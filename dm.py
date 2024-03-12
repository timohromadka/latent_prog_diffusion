import os
from omegaconf import DictConfig

from datasets import (
    load_from_disk,
    load_dataset,
    DatasetDict,
    Dataset
)
from torch.utils.data import DataLoader, Subset

from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

from lightning import (
    LightningDataModule
)


def load_hf_dataset(path_or_reponame: str, **kwargs):
    if os.path.exists(path_or_reponame):
        return load_from_disk(path_or_reponame)
    else:
        return load_dataset(
            path_or_reponame,
            cache_dir=kwargs.get('cache_dir', None)
        )


class ImageDatasets(LightningDataModule):
    def __init__(self, args) -> None:
        super().__init__()

        self.batch_size = args.batch_size
        self.data_dir = args.dataset
        self.image_resolution = args.image_resolution
        self.HF_DATASET_IMAGE_KEY = args.HF_DATASET_IMAGE_KEY
        self.num_train_samples = args.num_train_samples
        self.num_val_samples = args.num_val_samples

        # Preprocessing the datasets and DataLoaders creation.
        self.augmentations = Compose(
            [
                Resize(self.image_resolution,
                       interpolation=InterpolationMode.BILINEAR),
                CenterCrop(self.image_resolution),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )

    def setup(self, stage: str) -> None:
        dataset = load_hf_dataset(self.data_dir)
        dataset.set_transform(
            lambda sample: ImageDatasets._transforms(self, sample)
        )

        if isinstance(dataset, DatasetDict):
            self.train_dataset, self.valid_dataset = dataset['train'], dataset['test']
        elif isinstance(dataset, Dataset):
            split_datasets = dataset.train_test_split(0.1, 0.9, seed=42)
            self.train_dataset, self.valid_dataset = split_datasets['train'], split_datasets['test']

        # Subset selection if specified
        if self.num_train_samples is not None:
            self.train_dataset = Subset(self.train_dataset, range(self.num_train_samples))
        if self.num_val_samples is not None:
            self.valid_dataset = Subset(self.valid_dataset, range(self.num_val_samples))

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True,
                          drop_last=False
                          )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True,
                          drop_last=False
                          )

    def _transforms(self, sample):
        images = [
            self.augmentations(image.convert("RGB"))
            for image in sample[self.HF_DATASET_IMAGE_KEY]
        ]
        return {"images": images}