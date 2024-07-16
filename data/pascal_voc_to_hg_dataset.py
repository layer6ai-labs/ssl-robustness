import os
import argparse
import numpy as np
from PIL import Image
from datasets import Dataset, DatasetDict

from tqdm import tqdm

# based on http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html
# the following are the labels that will be replaced by 0 (background)
ids_to_replace = [0, 255]  # background, void


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_voc",
        type=str,
        required=False,
        default="tmp/data/voc/VOCdevkit/VOC2012",
    )
    parser.add_argument(
        "--path_to_hgdataset",
        type=str,
        required=False,
        default="tmp/data/pascal_voc_hg",
    )

    return parser.parse_args()


def get_all_filenames(args, dtype: str, split: str) -> list[str]:
    """
    dtype: JPEGImages, SegmentationClass
    split: train, val
    """
    path_to_split = f"{args.path_to_voc}/{dtype}"
    end = "jpg" if dtype == "JPEGImages" else "png"
    filenames = []
    with open(f"{args.path_to_voc}/ImageSets/Segmentation/{split}.txt", "r") as f:
        for line in f:
            filenames.append(f"{path_to_split}/{line.strip()}.{end}")

    return sorted(filenames)


def transform_label(label: Image.Image) -> Image.Image:
    """
    During training we have to ignore the 0 index
    However, the voc dataset has more labels to ignore (0, 255)
    """
    array = np.array(label)
    array[np.isin(array, ids_to_replace)] = 0

    return Image.fromarray(array)


def load_data(
    args,
) -> tuple[list[Image.Image], list[Image.Image], list[Image.Image], list[Image.Image],]:
    """
    Returns:
        train_images, train_labels, val_images, val_labels
    """
    train_images_files = get_all_filenames(args, "JPEGImages", "train")
    train_labels_files = get_all_filenames(args, "SegmentationClass", "train")
    val_images_files = get_all_filenames(args, "JPEGImages", "val")
    val_labels_files = get_all_filenames(args, "SegmentationClass", "val")

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
