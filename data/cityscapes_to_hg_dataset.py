import os
import argparse
import numpy as np
from PIL import Image
from datasets import Dataset, DatasetDict

from tqdm import tqdm

# based on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L62
# the following are the labels that will be replaced by 0 (background)
ids_to_replace = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    9,
    10,
    14,
    15,
    16,
    18,
    29,
    30,
]
ids_to_remap = [idx for idx in range(34) if idx not in ids_to_replace]
ids_remap = {k: v for k, v in zip(ids_to_remap, range(1, 35))}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_cityscapes", type=str, required=False, default="tmp/data/cityscapes"
    )
    parser.add_argument(
        "--path_to_hgdataset",
        type=str,
        required=False,
        default="tmp/data/cityscapes_hg",
    )

    return parser.parse_args()


def get_all_filenames(args, dtype: str, split: str) -> list[str]:
    """
    dtype: gtFine, leftImg8bit
    split: train, val
    """
    path_to_split = f"{args.path_to_cityscapes}/{dtype}/{split}"
    ends = ".png" if dtype == "leftImg8bit" else "labelIds.png"
    cities = os.listdir(path_to_split)
    filenames = []
    for city in cities:
        tmp_filenames = os.listdir(f"{path_to_split}/{city}")
        filenames += [
            f"{path_to_split}/{city}/{filename}"
            for filename in tmp_filenames
            if filename.endswith(ends)
        ]

    return sorted(filenames)


def transform_label(label: Image.Image) -> Image.Image:
    """
    During training we have to ignore the 0 index
    However, the CityScapes dataset has more labels to ignore.
    torch.nn.CrossEntropyLoss allows only one label to ignore.
    After replacing these with 0, we remap the other labels to be in the range [1, n_classes]
    """
    array = np.array(label)
    array[np.isin(array, ids_to_replace)] = 0
    for k, v in ids_remap.items():
        array[array == k] = v

    return Image.fromarray(array)


def load_data(
    args,
) -> tuple[list[Image.Image], list[Image.Image], list[Image.Image], list[Image.Image],]:
    """
    Returns:
        train_images, train_labels, val_images, val_labels
    """
    train_images_files = get_all_filenames(args, "leftImg8bit", "train")
    train_labels_files = get_all_filenames(args, "gtFine", "train")
    val_images_files = get_all_filenames(args, "leftImg8bit", "val")
    val_labels_files = get_all_filenames(args, "gtFine", "val")

    train_images = []
    for filename in tqdm(train_images_files):
        with Image.open(filename) as image:
            train_image = image if image.mode == "RGB" else image.convert("RGB")
            train_images.append(train_image)

    val_images = []
    for filename in tqdm(val_images_files):
        with Image.open(filename) as image:
            val_image = image if image.mode == "RGB" else image.convert("RGB")
            val_images.append(val_image)

    train_labels = []
    for filename in tqdm(train_labels_files):
        with Image.open(filename) as train_label:
            train_labels.append(transform_label(train_label))

    val_labels = []
    for filename in tqdm(val_labels_files):
        with Image.open(filename) as val_label:
            val_labels.append(transform_label(val_label))

    return train_images, train_labels, val_images, val_labels


def create_dataset(args) -> DatasetDict:
    train_images, train_labels, val_images, val_labels = load_data(args)
    train_dataset = Dataset.from_dict(
        {
            "image": train_images,
            "label": train_labels,
        }
    )
    print(train_dataset)
    val_dataset = Dataset.from_dict(
        {
            "image": val_images,
            "label": val_labels,
        }
    )
    print(val_dataset)
    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "val": val_dataset,
        }
    )
    print(dataset_dict)
    return dataset_dict


def main():
    args = parse_args()
    dataset_dict = create_dataset(args)
    dataset_dict.save_to_disk(args.path_to_hgdataset)


if __name__ == "__main__":
    main()
