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


def compute_iou(y_true, y_pred, num_classes):

    iou_per_class = []

    for cls in range(num_classes):
        inter = np.sum((y_true == cls) & (y_pred == cls))
        union = np.sum((y_true == cls) | (y_pred == cls))
        iou_per_class.append(inter / union if union else 0)

    return iou_per_class

def decode_mask(mask):
    colors = np.array([
        [255, 255, 0],        # Background - yellow
        [255, 0, 0],      # Building - red
        [0, 255, 0],      # Road - green
        [255, 165, 0],          # Utilities - orange
        [0, 0, 255],    # Water - blue
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

    # ---- Original ----
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    # ---- Prediction ----
    colored_pred = decode_mask(np.clip(masked_pred, 0, None))
    colored_pred[masked_pred == -1] = 0

    plt.subplot(1, 3, 2)
    plt.imshow(img)
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
        plt.imshow(img)
        plt.imshow(colored_gt, alpha=0.5)
        plt.title("Ground Truth")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"comparison_{k}.png")
    plt.close()

    return k + 1


def run_train(args):
    # 1. Split
    ids = os.listdir(os.path.join(args.data, "images"))
    train_ids, val_ids = split_dataset(
        ids, train_frac=0.8, val_frac=0.2, seed=42
    )

    # 2. DataLoaders
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

    strategy = tf.distribute.MirroredStrategy() #Multi GPU Strategy
    print("Number of devices:", strategy.num_replicas_in_sync)

    with strategy.scope():

        # 3. Model
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
            loss=focal_dice_loss(alpha=args.alpha, gamma=args.gamma, dice_weight=args.dice_weight),
            metrics=["accuracy"]
        )

    os.makedirs(args.model_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            "val_loss", patience=args.patience, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.model_dir, "best_model.keras"),
            "val_loss",
            save_best_only=True,
        ),
    ]

    model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=args.epochs,
        callbacks=callbacks
    )

    print("Training complete.")
    
    # 4. Evaluate + visualize
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

    # ----------- SINGLE IMAGE MODE ------------
    if args.image:
        with rasterio.open(args.image) as src:
            img = src.read(
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
        #mask_scale=args.mask_scale,
    )

    y_true_all, y_pred_all = [], []
    k = 0  # Visualization counter

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
                y_true_all.extend(true_mask[valid_mask].flatten())
                y_pred_all.extend(pred_mask[valid_mask].flatten())
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
        # Calculate metrics on the entire test set
        accuracy = accuracy_score(y_true_all, y_pred_all)
        f1 = f1_score(y_true_all, y_pred_all, average="macro")
        precision = precision_score(y_true_all, y_pred_all, average="macro")
        recall = recall_score(y_true_all, y_pred_all, average="macro")
 
        precision_per_class = precision_score(
            y_true_all,
            y_pred_all,
            average=None,
            labels=list(range(args.num_classes))
        )

        recall_per_class = recall_score(
            y_true_all,
            y_pred_all,
            average=None,
            labels=list(range(args.num_classes))
        )

        conf_matrix = confusion_matrix(y_true_all, y_pred_all, labels=list(range(args.num_classes)))

        iou = compute_iou(
            np.array(y_true_all),
            np.array(y_pred_all),
            args.num_classes
        )

        f1_per_class = f1_score(
            y_true_all,
            y_pred_all,
            average=None,              # IMPORTANT
            labels=list(range(args.num_classes))
        )
        # Create metrics dictionary
        metrics = {
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Precision per-class": precision_per_class.tolist(),
            "Recall": float(recall),
            "Recall per-class": recall_per_class.tolist(),
            "Confusion Matrix": conf_matrix.tolist(),  # Save confusion matrix explicitly
            "IoU per-class": [float(x) for x in iou],
            "Mean IoU": float(np.mean(iou)),
            "F1": float(f1),
            "F1 per-class": f1_per_class.tolist()
        }

        # Save metrics to JSON
        with open("model_evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("✅ Inference Completed. Metrics saved to model_evaluation_metrics.json")


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
    t.add_argument("--alpha", type=float, required=True)
    t.add_argument("--dice_weight", type=float, default=0.4)
    t.add_argument("--input-shape", type=int, nargs=3, default=[512, 512, 3])
    t.add_argument("--input-scale", type=int, default=255)
    t.add_argument("--mask-scale", type=int, default=255)
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
    i.set_defaults(func=run_infer)


    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
