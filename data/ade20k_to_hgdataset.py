import os
import argparse
from PIL import Image
from datasets import Dataset, DatasetDict

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_ade20k", type=str, required=True)
    parser.add_argument(
        "--path_to_hgdataset", type=str, required=False, default="tmp/data/ade20k_hg"
    )

    return parser.parse_args()


def get_all_filenames(args, dtype: str, split: str) -> list[str]:
    """
    dtype: annotations, images
    split: training, validation
    """
    path_to_ade20k = args.path_to_ade20k
    path_to_split = f"{path_to_ade20k}/{dtype}/{split}"
    filenames = os.listdir(path_to_split)
    filenames = sorted([f"{path_to_split}/{filename}" for filename in filenames])
    return filenames


def load_data(
    args,
) -> tuple[list[Image.Image], list[Image.Image], list[Image.Image], list[Image.Image],]:
    """
    Returns:
        train_images, train_labels, val_images, val_labels
    """
    train_images_files = get_all_filenames(args, "images", "training")
    train_labels_files = get_all_filenames(args, "annotations", "training")
    val_images_files = get_all_filenames(args, "images", "validation")
    val_labels_files = get_all_filenames(args, "annotations", "validation")

    train_images = []
    for filename in train_images_files:
        with Image.open(filename) as image:
            train_image = image if image.mode == "RGB" else image.convert("RGB")
            train_images.append(train_image)

    val_images = []
    for filename in val_images_files:
        with Image.open(filename) as image:
            val_image = image if image.mode == "RGB" else image.convert("RGB")
            val_images.append(val_image)

    train_labels = []
    for filename in train_labels_files:
        with Image.open(filename) as train_label:
            train_labels.append(train_label)

    val_labels = []
    for filename in val_labels_files:
        with Image.open(filename) as val_label:
            val_labels.append(val_label)

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
