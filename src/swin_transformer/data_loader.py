import os
import warnings
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
import rasterio
from rasterio.enums import Resampling

class DynamicDataLoader(Sequence):
    def __init__(
        self,
        data_dir,
        ids,
        batch_size=2,
        img_size=(256, 256, 3), # Defaulted to 3 channels for Aerial dataset
        mode="train",
        image_dtype=np.float32,
        mask_dtype=np.int32,
        num_classes=None,
        input_scale=255,
        mask_scale=1,
        dsm_scale=255,
        dsm_dir="normalized_DSM"
    ):
        self.batch_size = batch_size
        self.mode = mode
        self.drop_remainder = (mode in ["train", "val"])

        if len(img_size) == 3:
            self.height, self.width, self.channels = img_size
        else:
            raise ValueError("img_size must be (H, W, C)")
        
        self.img_size = (self.height, self.width)

        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.dsm_dir = os.path.join(data_dir, dsm_dir) # Passed parameter
        self.ids = ids
        self.image_dtype = image_dtype
        self.mask_dtype = mask_dtype
        self.num_classes = num_classes
        self.input_scale = input_scale
        self.mask_scale = mask_scale
        self.dsm_scale = dsm_scale

        self.image_ext = ".tif" 
        self.dsm_ext = ".tif"
        self.mask_ext = ".tif" if mode != "infer" else None

    def __len__(self):
        if self.drop_remainder:
            return len(self.ids) // self.batch_size
        else:
            return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f"Index {index} is out of range for {len(self)} batches.")

        start_index = index * self.batch_size
        end_index = min(start_index + self.batch_size, len(self.ids))
        batch_ids = self.ids[start_index:end_index]

        if len(batch_ids) == 0:
            raise ValueError(
                f"Batch {index} has no data. This should not happen with correct batch calculation."
            )

        return self._data_generation(batch_ids)

    def _load_image(self, image_id):
        """
        Loads RGB. If channels == 4, also loads and stacks DSM.
        """

        image_path = os.path.join(self.image_dir, image_id)

        with rasterio.open(image_path) as src:
            image = src.read(
                [1, 2, 3],
                out_shape=(3, self.height, self.width),
                resampling=Resampling.bilinear
            ).transpose(1, 2, 0)

        image = image.astype(self.image_dtype) / self.input_scale

        # ---- Conditional DSM Loading ----
        if self.channels == 4:
            dsm_id = image_id.replace(self.image_ext, self.dsm_ext)
            dsm_path = os.path.join(self.dsm_dir, dsm_id)

            with rasterio.open(dsm_path) as src:
                dsm = src.read(
                    1, 
                    out_shape=(self.height, self.width),
                    resampling=Resampling.bilinear
                )
            
            # Add channel dim and normalize
            dsm = dsm.astype(self.image_dtype)[..., np.newaxis] / self.dsm_scale

            # Stack into 4-channel input (H, W, 4)
            final_image = np.concatenate([image, dsm], axis=-1)
        else:
            # Keep as 3-channel RGB
            final_image = image

        # ---- If inference-only mode, skip mask ----
        if self.mode == "infer":
            return final_image, None

        # Otherwise load mask
        mask_path = os.path.join(
            self.mask_dir, image_id.replace(self.image_ext, self.mask_ext)
        )

        with rasterio.open(mask_path) as src:
            mask_arr = src.read(
                1,
                out_shape=(self.height, self.width),
                resampling=Resampling.nearest
            )

        mask_arr = mask_arr.astype(self.mask_dtype)

        IGNORE_VALUE = 255
        valid_mask = mask_arr != IGNORE_VALUE
        safe_mask = mask_arr.copy()
        safe_mask[~valid_mask] = 0

        mask_onehot = to_categorical(safe_mask, num_classes=self.num_classes)
        mask_onehot[~valid_mask] = 0

        return final_image, mask_onehot, valid_mask
    
    def _data_generation(self, batch_ids):

        batch_size = len(batch_ids)

        # ---- Allocate image batch ----
        X = np.empty(
            (batch_size, self.height, self.width, self.channels),
            dtype=self.image_dtype
        )

        # ---- TRAIN MODE ----
        if self.mode in ["train", "val"]:

            y = np.empty(
                (batch_size, self.height, self.width, self.num_classes),
                dtype=np.float32
            )

            for i, ID in enumerate(batch_ids):
                img, mask, _ = self._load_image(ID)
                X[i] = img
                y[i] = mask

            return X, y


        # ---- TEST MODE (evaluation) ----
        elif self.mode == "test":

            y = np.empty(
                (batch_size, self.height, self.width, self.num_classes),
                dtype=np.float32
            )

            valid = np.empty(
                (batch_size, self.height, self.width),
                dtype=bool
            )

            for i, ID in enumerate(batch_ids):
                img, mask, valid_mask = self._load_image(ID)
                X[i] = img
                y[i] = mask
                valid[i] = valid_mask  # keep as bool for indexing

            return X, y, valid


        # ---- INFERENCE MODE ----
        elif self.mode == "infer":

            for i, ID in enumerate(batch_ids):
                img, _ = self._load_image(ID)
                X[i] = img

            return X