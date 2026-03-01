import random

def split_dataset(ids, train_frac=0.8, val_frac=0.2, seed=42):
    """
    Splits IDs into train and validation sets.
    No internal test split here.
    """

    assert abs(train_frac + val_frac - 1.0) < 1e-6, \
        "train_frac + val_frac must equal 1.0"

    ids = sorted(ids)
    random.seed(seed)
    random.shuffle(ids)

    total = len(ids)
    train_end = int(total * train_frac)

    train_ids = ids[:train_end]
    val_ids   = ids[train_end:]

    return train_ids, val_ids