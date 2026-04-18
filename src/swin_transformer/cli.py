import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

from swin_transformer.model_loader import get_model
from swin_transformer.split_data import split_dataset
from swin_transformer.data_loader import DynamicDataLoader
from swin_transformer.loss import focal_dice_loss
from keras_swin_unet import transformer_layers, swin_layers

import rasterio
from rasterio.enums import Resampling


import tensorflow as tf

class MeanIoUMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes=5, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.miou = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert one-hot to integer labels
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        self.miou.update_state(y_true, y_pred)

    def result(self):
        return self.miou.result()

    def reset_states(self):
        self.miou.reset_states()

def decode_mask(mask):
    colors = np.array([
        [255, 255, 255],        # Impervious surfaces - White
        [0, 0, 255],      # Building - Blue
        [0, 255, 255],      # Low Vegetation - Cyan
        [0, 255, 0],          # Tree - Gree
        [255, 255, 0],    # Car - Yellow
        [255, 0, 0],      # Background - Red
    ])

    return colors[mask]


def visualize_comparison(k, img, pred_mask,
                         true_mask=None,
                         valid_mask=None):

    # If no GT valid_mask (pure inference),
    # detect padding from black image region
    if valid_mask is None:
        valid_mask = ~np.all(img == 0, axis=-1)

    masked_pred = pred_mask.copy()
    masked_pred[~valid_mask] = -1

    plt.figure(figsize=(12, 6))

    display_img = img[..., :3]

    # ---- Original ----
    plt.subplot(1, 3, 1)
    plt.imshow(display_img)
    plt.title("Original")
    plt.axis("off")

    # ---- Prediction ----
    colored_pred = decode_mask(np.clip(masked_pred, 0, None))
    colored_pred[masked_pred == -1] = 0

    plt.subplot(1, 3, 2)
    plt.imshow(display_img)
    plt.imshow(colored_pred, alpha=0.5)
    plt.title("Prediction")
    plt.axis("off")

    # ---- Ground Truth ----
    if true_mask is not None:
        masked_gt = true_mask.copy()
        masked_gt[~valid_mask] = -1

        colored_gt = decode_mask(np.clip(masked_gt, 0, None))
        colored_gt[masked_gt == -1] = 0

        plt.subplot(1, 3, 3)
        plt.imshow(display_img)
        plt.imshow(colored_gt, alpha=0.5)
        plt.title("Ground Truth")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"comparison_{k}.png")
    plt.close()

    return k + 1


def plot_loss_curves(history, save_path=None):
    plt.figure(figsize=(8,6))

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_iou_curves(history, save_path=None):
    plt.figure(figsize=(8,6))

    plt.plot(history.history['mean_iou'], label='Training Mean IoU')
    plt.plot(history.history['val_mean_iou'], label='Validation Mean IoU')

    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.title('Mean IoU')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def run_train(args):

    ids = os.listdir(os.path.join(args.data, "images"))
    train_ids, val_ids = split_dataset(
        ids, train_frac=0.8, val_frac=0.2, seed=42
    )

    def make_loader(ids, mode):
        return DynamicDataLoader(
            data_dir=args.data,
            ids=ids,
            batch_size=args.bs,
            img_size=tuple(args.input_shape),
            mode=mode,
            image_dtype=np.float32,
            mask_dtype=np.int32,
            num_classes=args.num_classes,
            input_scale=args.input_scale,
            mask_scale=args.mask_scale,
        )

    train_loader = make_loader(train_ids, "train")
    val_loader = make_loader(val_ids, "val")

    os.makedirs(args.model_dir, exist_ok=True)

    best_ckpt_path = os.path.join(args.model_dir, "best_model.keras")
    last_ckpt_path = os.path.join(args.model_dir, "last_model.keras")

    strategy = tf.distribute.MirroredStrategy() #Multi GPU Strategy
    print("Number of devices:", strategy.num_replicas_in_sync)

    with strategy.scope():

        # Resume if checkpoint exists
        if os.path.exists(last_ckpt_path):
            print("Loading last checkpoint to resume training...")
            model = keras.models.load_model(
                last_ckpt_path,
                custom_objects={
                    "MeanIoUMetric": MeanIoUMetric,
                    "loss": focal_dice_loss(
                        alpha=args.alpha,
                        gamma=args.gamma,
                        dice_weight=args.dice_weight,
                        dice_class_weights=args.dice_class_weights
                    )
                }
            )
        else:
            print("Starting fresh training run...")
            model = get_model(
                input_size=tuple(args.input_shape),
                filter_num_begin=args.filter,
                depth=args.depth,
                stack_num_down=args.stack_down,
                stack_num_up=args.stack_up,
                patch_size=tuple(args.patch_size),
                num_heads=args.num_heads,
                window_size=args.window_size,
                num_mlp=args.num_mlp,
                num_classes=args.num_classes,
            )

            model.compile(
                    optimizer=keras.optimizers.Adam(1e-4, clipvalue=0.5),
                    loss=focal_dice_loss(
                        alpha=args.alpha,
                        gamma=args.gamma,
                        dice_weight=args.dice_weight,
                        dice_class_weights=args.dice_class_weights
                    ),
                    metrics=["accuracy", MeanIoUMetric(num_classes=args.num_classes)]
                )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_mean_iou",
            mode="max",
            patience=args.patience,
            restore_best_weights=True
        ),
        # Save best model (highest val IoU)
        keras.callbacks.ModelCheckpoint(
            best_ckpt_path,
            monitor="val_mean_iou",
            save_best_only=True,
            mode="max"
        ),
        # Save last epoch model (for true resume)
        keras.callbacks.ModelCheckpoint(
            last_ckpt_path,
            save_best_only=False
        ),
    ]

    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=args.epochs,
        callbacks=callbacks,
        #mode="max"
    )

    plot_loss_curves(history, save_path=os.path.join(args.model_dir, "loss_curve.png"))
    plot_iou_curves(history, save_path=os.path.join(args.model_dir, "iou_curve.png"))
    print("Saved loss curve and mean IoU curve")


    print("Training complete.")
    
    '''
    y_true, y_logits = [], []
    if args.visualize:
        k = 0
    for X, y in test_loader: #This line should be indented?
        preds = model.predict(X)
        y_true.extend(y)
        y_logits.extend(preds)
        if args.visualize:
            for i in range(min(args.visualize, X.shape[0])):
                k = visualize_comparison(k, X, y, preds, i, args.num_classes)
    '''
    # 5. Metrics
    '''
    y_t = np.concatenate([yt.argmax(-1).flatten() for yt in y_true])
    y_p = np.concatenate([pl.argmax(-1).flatten() for pl in y_logits])
    metrics = {
        "Accuracy": accuracy_score(y_t, y_p),
        "F1": f1_score(y_t, y_p, average="weighted"),
        "Precision": precision_score(y_t, y_p, average="weighted"),
        "Recall": recall_score(y_t, y_p, average="weighted"),
        "AUC": roc_auc_score(
            keras.utils.to_categorical(y_t),
            keras.utils.to_categorical(y_p),
            multi_class="ovr",
        ),
    }
    with open("model_evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("✅ Done. Metrics:", metrics)
'''


def run_infer(args):
    import matplotlib
    matplotlib.use('Agg')  # Ensures no Qt errors in headless environments
    import matplotlib.pyplot as plt

    #strategy = tf.distribute.MirroredStrategy()
    #print("Number of devices:", strategy.num_replicas_in_sync)

    # Load the model
    custom_objects = {
        **transformer_layers.__dict__,
        **swin_layers.__dict__,
    }
    #with strategy.scope():
    model = load_model(
        os.path.join(args.model_dir, "best_model.keras"), custom_objects=custom_objects, compile=False
    )

    '''
    # ----------- SINGLE IMAGE MODE ------------
    if args.image:
        with rasterio.open(args.image) as src:
            img = src.read(
            [1, 2, 3],
            out_shape=(3, args.input_shape[0], args.input_shape[1]),
            resampling=Resampling.bilinear
                ).transpose(1, 2, 0)
            
        inp = img.astype(np.float32)[None] / args.input_scale
        preds = model.predict(inp)[0]
        pred_mask = np.argmax(preds, axis=-1)

        if args.visualize:
            visualize_comparison(
                0,
                img,
                pred_mask,
                true_mask=None,
                valid_mask=None,
            )
        return
    '''
    # ----------- TEST LOADER MODE ------------
    print("Running inference on test dataset...")
    image_dir = os.path.join(args.data, "images")
    ids = sorted(os.listdir(image_dir))

    print("Image dir:", image_dir)
    print("Number of files found:", len(ids))
    print("First 10 files:", ids[:10])

    mode = "infer" if not args.evaluate else "test"

    test_loader = DynamicDataLoader(
        data_dir=args.data,
        ids=ids,
        batch_size=args.bs,
        img_size=tuple(args.input_shape),
        mode=mode,
        image_dtype=np.float32,
        mask_dtype=np.int32,
        num_classes=args.num_classes,
        input_scale=args.input_scale,
        dsm_scale=args.dsm_scale,
    )

    k = 0  # Visualization counter

    print("Total batches:", len(test_loader))

    num_classes = args.num_classes
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in test_loader:

        if args.evaluate:
            X_batch, y_batch, valid_batch = batch
        else:
            X_batch = batch
            y_batch = None
            valid_batch = None

        preds = model.predict(X_batch)

        for i in range(X_batch.shape[0]):

            img = X_batch[i]
            pred_mask = np.argmax(preds[i], axis=-1)

            if args.evaluate:
                valid_mask = valid_batch[i]
                true_mask = np.argmax(y_batch[i], axis=-1)

                # Remove ignored pixels from metrics
                valid_pixels = valid_mask

                true_flat = true_mask[valid_pixels].flatten()
                pred_flat = pred_mask[valid_pixels].flatten()

                cm = confusion_matrix(
                    true_flat,
                    pred_flat,
                    labels=list(range(num_classes))
                )

                conf_matrix += cm
            else:
                true_mask = None
                valid_mask = None

            # ---- Visualization ----
            if args.visualize and k < args.visualize:
                k = visualize_comparison(
                    k,
                    img,
                    pred_mask,
                    true_mask,
                    valid_mask
                )

        #if args.visualize and k >= args.visualize:
            #break
    if args.evaluate:

        TP = np.diag(conf_matrix)
        FP = conf_matrix.sum(axis=0) - TP
        FN = conf_matrix.sum(axis=1) - TP
        TN = conf_matrix.sum() - (TP + FP + FN)

        precision_per_class = TP / (TP + FP + 1e-7)
        recall_per_class    = TP / (TP + FN + 1e-7)
        iou_per_class       = TP / (TP + FP + FN + 1e-7)

        f1_per_class = 2 * precision_per_class * recall_per_class / (
            precision_per_class + recall_per_class + 1e-7
        )

        accuracy = TP.sum() / conf_matrix.sum()

        mean_precision = np.mean(precision_per_class)
        mean_recall    = np.mean(recall_per_class)
        mean_iou       = np.mean(iou_per_class)
        mean_f1        = np.mean(f1_per_class)

        metrics = {
            "Accuracy": float(accuracy),
            "Precision": float(mean_precision),
            "Precision per-class": precision_per_class.tolist(),
            "Recall": float(mean_recall),
            "Recall per-class": recall_per_class.tolist(),
            "Confusion Matrix": conf_matrix.tolist(),
            "IoU per-class": iou_per_class.tolist(),
            "Mean IoU": float(mean_iou),
            "F1": float(mean_f1),
            "F1 per-class": f1_per_class.tolist()
        }

        with open("model_evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("Inference Completed. Metrics saved to model_evaluation_metrics.json")


def main():
    p = argparse.ArgumentParser(
        prog="swin-unet", description="Train, evaluate+visualize, or infer"
    )
    sp = p.add_subparsers(dest="command", required=True)

    # train
    t = sp.add_parser("train", help="Train & evaluate (with optional visualize)")
    t.add_argument("--data", default="./data")
    t.add_argument("--model-dir", default="./checkpoint")
    t.add_argument("--num-classes", type=int, default=2)
    t.add_argument("--bs", type=int, default=64)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--filter", type=int, default=128)
    t.add_argument("--depth", type=int, default=4)
    t.add_argument("--stack-down", type=int, default=2)
    t.add_argument("--stack-up", type=int, default=2)
    t.add_argument("--patch-size", type=int, nargs=2, default=[4, 4])
    t.add_argument("--num-heads", type=int, nargs=4, default=[4, 8, 8, 8])
    t.add_argument("--window-size", type=int, nargs=4, default=[4, 2, 2, 2])
    t.add_argument("--num-mlp", type=int, default=512)
    t.add_argument("--gamma", type=float, required=True)
    t.add_argument("--alpha", type=float, nargs="+", required=True,  help="Class-wise alpha weights")
    t.add_argument("--dice_class_weights", type=float, nargs="+", default=None, help="Class-wise Dice weights")
    t.add_argument("--input-shape", type=int, nargs=3, default=[512, 512, 3])
    t.add_argument("--input-scale", type=int, default=255)
    t.add_argument("--mask-scale", type=int, default=255)
    t.add_argument("--dsm-dir", type=str, default="normalized_DSM") # Added
    t.add_argument("--dsm-scale", type=float, default=255.0)
    t.add_argument(
        "--visualize",
        type=int,
        default=0,
        help="How many test images to visualize (0 = none)",
    )
    t.set_defaults(func=run_train)

    # infer
    i = sp.add_parser("infer", help="Run inference on single image or test set")
    i.add_argument("--model-dir", default="./checkpoint")
    i.add_argument("--image", help="Path to single image (optional)")
    i.add_argument("--output", default="out.png")
    i.add_argument("--num-classes", type=int, default=2)
    i.add_argument("--gamma", type=float, default=0.25)
    i.add_argument("--alpha", type=float, default=2.0)
    i.add_argument("--input-scale", type=int, default=255)
    i.add_argument("--data", default="./data", help="Dataset folder path for test loader")
    i.add_argument("--bs", type=int, default=1)
    i.add_argument("--input-shape", type=int, nargs=3, default=[512, 512, 3])
    i.add_argument("--evaluate", type=int, default=0, help="1 to compute metrics")    
    i.add_argument("--visualize", type=int, default=0, help="How many images to visualize")
    i.add_argument("--dsm-dir", type=str, default="normalized_DSM") # Added
    i.add_argument("--dsm-scale", type=float, default=255.0)
    i.set_defaults(func=run_infer)


    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
