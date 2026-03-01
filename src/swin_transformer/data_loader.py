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
        img_size=(256, 256),
        mode="train",
        image_dtype=np.float32,
        mask_dtype=np.int32,
        num_classes=None,
        input_scale=255,
        mask_scale=1,
    ):
        self.batch_size = batch_size
        self.mode = mode


        if len(img_size) == 3:
            self.height, self.width, self.channels = img_size
        else:
            raise ValueError("img_size must be (H, W, C)")
        
        self.img_size = (self.height, self.width)

        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.ids = ids
        self.image_dtype = image_dtype
        self.mask_dtype = mask_dtype
        self.num_classes = num_classes
        self.input_scale = input_scale
        self.mask_scale = mask_scale

        self.image_ext = self._determine_extension(self.image_dir)

        if self.mode != "infer":
            self.mask_ext = self._determine_extension(self.mask_dir)
        else:
            self.mask_ext = None

    def _determine_extension(self, directory):
        for filename in os.listdir(directory):
            return os.path.splitext(filename)[1]
        raise FileNotFoundError(f"No files found in {directory}")

    def __len__(self):
        return len(self.ids) // self.batch_size

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

        image_path = os.path.join(self.image_dir, image_id)

        with rasterio.open(image_path) as src:
            image = src.read(
                out_shape=(3, self.height, self.width),
                resampling=Resampling.bilinear
            ).transpose(1, 2, 0)

        image = image.astype(self.image_dtype) / self.input_scale

        # ---- If inference-only mode, skip mask ----
        if self.mode == "infer":
            return image, None, None

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

        return image, mask_onehot, valid_mask
    
    def _data_generation(self, batch_ids):

        X = np.empty((len(batch_ids), self.height, self.width, 3),
                    dtype=self.image_dtype)

        if self.mode != "infer":
            y = np.empty((len(batch_ids),
                        self.height,
                        self.width,
                        self.num_classes),
                        dtype=np.float32)

            valid = np.empty((len(batch_ids),
                            self.height,
                            self.width),
                            dtype=bool)

        for i, ID in enumerate(batch_ids):

            image, mask, valid_mask = self._load_image(ID)

            X[i] = image

            if self.mode != "infer":
                y[i] = mask
                valid[i] = valid_mask

        if self.mode == "infer":
            return X

        return X, y, valid

