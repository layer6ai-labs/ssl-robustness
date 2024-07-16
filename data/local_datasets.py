import os
from pathlib import Path
from typing import Callable, Optional, Union, Dict, List

import torch
import torchvision
from PIL import Image
from datasets import load_dataset, load_from_disk
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
import numpy as np


class HGDataset:
    name: str
    hg_name: str
    dataset_class: torchvision.datasets.VisionDataset
    image: Optional[str] = "image"
    label: Optional[str] = "label"
    train_split_name: Optional[str] = "train"
    test_split_name: Optional[str] = "test"
    classes_cnt: Optional[int] = None

    def __init__(self, root: str):
        self.root = root
        self.data = None

    def load_dataset(self, split: str) -> None:
        assert split in [
            "train",
            "test",
        ], f'Only split="train" and split="test" supported. Provided: split={split}'
        split = self.train_split_name if split == "train" else self.test_split_name
        self.data = load_dataset(self.hg_name, split=split, cache_dir=self.root)


class FashionMNIST(HGDataset):
    name = "fashion_mnist"
    hg_name = "fashion_mnist"
    classes_cnt = 10


class MNIST(HGDataset):
    name = "mnist"
    hg_name = "mnist"
    classes_cnt = 10


class Food101(HGDataset):
    name = "food101"
    hg_name = "food101"
    test_split_name = "validation"
    classes_cnt = 101


class ImageNet(HGDataset):
    name = "imagenet"
    hg_name = "imagenet-1k"
    classes_cnt = 1000


class STL10(HGDataset):
    name = "stl10"
    hg_name = "jxie/stl10"
    classes_cnt = 10


class CIFAR10(HGDataset):
    name = "cifar10"
    hg_name = "cifar10"
    image = "img"
    classes_cnt = 10


class CIFAR100(HGDataset):
    name = "cifar100"
    image = "img"
    label = "fine_label"
    hg_name = "cifar100"
    classes_cnt = 100


class Flowers102(HGDataset):
    name = "flowers102"
    hg_name = "nelorth/oxford-flowers"
    classes_cnt = 102


class FoodSeg103(HGDataset):
    name = "foodseg103"
    hg_name = "EduardoPacheco/FoodSeg103"
    test_split_name = "validation"
    classes_cnt = 104

    id2label = {
        0: "background",
        1: "candy",
        2: "egg tart",
        3: "french fries",
        4: "chocolate",
        5: "biscuit",
        6: "popcorn",
        7: "pudding",
        8: "ice cream",
        9: "cheese butter",
        10: "cake",
        11: "wine",
        12: "milkshake",
        13: "coffee",
        14: "juice",
        15: "milk",
        16: "tea",
        17: "almond",
        18: "red beans",
        19: "cashew",
        20: "dried cranberries",
        21: "soy",
        22: "walnut",
        23: "peanut",
        24: "egg",
        25: "apple",
        26: "date",
        27: "apricot",
        28: "avocado",
        29: "banana",
        30: "strawberry",
        31: "cherry",
        32: "blueberry",
        33: "raspberry",
        34: "mango",
        35: "olives",
        36: "peach",
        37: "lemon",
        38: "pear",
        39: "fig",
        40: "pineapple",
        41: "grape",
        42: "kiwi",
        43: "melon",
        44: "orange",
        45: "watermelon",
        46: "steak",
        47: "pork",
        48: "chicken duck",
        49: "sausage",
        50: "fried meat",
        51: "lamb",
        52: "sauce",
        53: "crab",
        54: "fish",
        55: "shellfish",
        56: "shrimp",
        57: "soup",
        58: "bread",
        59: "corn",
        60: "hamburg",
        61: "pizza",
        62: "hanamaki baozi",
        63: "wonton dumplings",
        64: "pasta",
        65: "noodles",
        66: "rice",
        67: "pie",
        68: "tofu",
        69: "eggplant",
        70: "potato",
        71: "garlic",
        72: "cauliflower",
        73: "tomato",
        74: "kelp",
        75: "seaweed",
        76: "spring onion",
        77: "rape",
        78: "ginger",
        79: "okra",
        80: "lettuce",
        81: "pumpkin",
        82: "cucumber",
        83: "white radish",
        84: "carrot",
        85: "asparagus",
        86: "bamboo shoots",
        87: "broccoli",
        88: "celery stick",
        89: "cilantro mint",
        90: "snow peas",
        91: "cabbage",
        92: "bean sprouts",
        93: "onion",
        94: "pepper",
        95: "green beans",
        96: "French beans",
        97: "king oyster mushroom",
        98: "shiitake",
        99: "enoki mushroom",
        100: "oyster mushroom",
        101: "white button mushroom",
        102: "salad",
        103: "other ingredients",
    }


class ADE20k(HGDataset):
    name = "ade20k"
    hg_name = None
    classes_cnt = 151
    test_split_name = "val"
    id2label = {
        0: "background",
        1: "wall",
        2: "building, edifice",
        3: "sky",
        4: "floor, flooring",
        5: "tree",
        6: "ceiling",
        7: "road, route",
        8: "bed ",
        9: "windowpane, window ",
        10: "grass",
        11: "cabinet",
        12: "sidewalk, pavement",
        13: "person, individual, someone, somebody, mortal, soul",
        14: "earth, ground",
        15: "door, double door",
        16: "table",
        17: "mountain, mount",
        18: "plant, flora, plant life",
        19: "curtain, drape, drapery, mantle, pall",
        20: "chair",
        21: "car, auto, automobile, machine, motorcar",
        22: "water",
        23: "painting, picture",
        24: "sofa, couch, lounge",
        25: "shelf",
        26: "house",
        27: "sea",
        28: "mirror",
        29: "rug, carpet, carpeting",
        30: "field",
        31: "armchair",
        32: "seat",
        33: "fence, fencing",
        34: "desk",
        35: "rock, stone",
        36: "wardrobe, closet, press",
        37: "lamp",
        38: "bathtub, bathing tub, bath, tub",
        39: "railing, rail",
        40: "cushion",
        41: "base, pedestal, stand",
        42: "box",
        43: "column, pillar",
        44: "signboard, sign",
        45: "chest of drawers, chest, bureau, dresser",
        46: "counter",
        47: "sand",
        48: "sink",
        49: "skyscraper",
        50: "fireplace, hearth, open fireplace",
        51: "refrigerator, icebox",
        52: "grandstand, covered stand",
        53: "path",
        54: "stairs, steps",
        55: "runway",
        56: "case, display case, showcase, vitrine",
        57: "pool table, billiard table, snooker table",
        58: "pillow",
        59: "screen door, screen",
        60: "stairway, staircase",
        61: "river",
        62: "bridge, span",
        63: "bookcase",
        64: "blind, screen",
        65: "coffee table, cocktail table",
        66: "toilet, can, commode, crapper, pot, potty, stool, throne",
        67: "flower",
        68: "book",
        69: "hill",
        70: "bench",
        71: "countertop",
        72: "stove, kitchen stove, range, kitchen range, cooking stove",
        73: "palm, palm tree",
        74: "kitchen island",
        75: "computer, computing machine, computing device, data processor, electronic computer, information processing system",
        76: "swivel chair",
        77: "boat",
        78: "bar",
        79: "arcade machine",
        80: "hovel, hut, hutch, shack, shanty",
        81: "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle",
        82: "towel",
        83: "light, light source",
        84: "truck, motortruck",
        85: "tower",
        86: "chandelier, pendant, pendent",
        87: "awning, sunshade, sunblind",
        88: "streetlight, street lamp",
        89: "booth, cubicle, stall, kiosk",
        90: "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box",
        91: "airplane, aeroplane, plane",
        92: "dirt track",
        93: "apparel, wearing apparel, dress, clothes",
        94: "pole",
        95: "land, ground, soil",
        96: "bannister, banister, balustrade, balusters, handrail",
        97: "escalator, moving staircase, moving stairway",
        98: "ottoman, pouf, pouffe, puff, hassock",
        99: "bottle",
        100: "buffet, counter, sideboard",
        101: "poster, posting, placard, notice, bill, card",
        102: "stage",
        103: "van",
        104: "ship",
        105: "fountain",
        106: "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
        107: "canopy",
        108: "washer, automatic washer, washing machine",
        109: "plaything, toy",
        110: "swimming pool, swimming bath, natatorium",
        111: "stool",
        112: "barrel, cask",
        113: "basket, handbasket",
        114: "waterfall, falls",
        115: "tent, collapsible shelter",
        116: "bag",
        117: "minibike, motorbike",
        118: "cradle",
        119: "oven",
        120: "ball",
        121: "food, solid food",
        122: "step, stair",
        123: "tank, storage tank",
        124: "trade name, brand name, brand, marque",
        125: "microwave, microwave oven",
        126: "pot, flowerpot",
        127: "animal, animate being, beast, brute, creature, fauna",
        128: "bicycle, bike, wheel, cycle ",
        129: "lake",
        130: "dishwasher, dish washer, dishwashing machine",
        131: "screen, silver screen, projection screen",
        132: "blanket, cover",
        133: "sculpture",
        134: "hood, exhaust hood",
        135: "sconce",
        136: "vase",
        137: "traffic light, traffic signal, stoplight",
        138: "tray",
        139: "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
        140: "fan",
        141: "pier, wharf, wharfage, dock",
        142: "crt screen",
        143: "plate",
        144: "monitor, monitoring device",
        145: "bulletin board, notice board",
        146: "shower",
        147: "radiator",
        148: "glass, drinking glass",
        149: "clock",
        150: "flag",
    }

    def load_dataset(self, split: str) -> None:
        assert split in [
            "train",
            "test",
        ], f'Only split="train" and split="test" supported. Provided: split={split}'
        split = self.train_split_name if split == "train" else self.test_split_name
        self.data = load_from_disk("tmp/data/ade20k_hg")[split]


class CityScapes(HGDataset):
    name = "cityscapes"
    hg_name = None
    classes_cnt = 20
    test_split_name = "val"
    id2label = {
        0: "background/unknown/unlabeled",
        1: "road",
        2: "sidewalk",
        3: "building",
        4: "wall",
        5: "fence",
        6: "pole",
        7: "traffic light",
        8: "traffic sign",
        9: "vegetation",
        10: "terrain",
        11: "sky",
        12: "person",
        13: "raider",
        14: "car",
        15: "truck",
        16: "bus",
        17: "train",
        18: "motorcycle",
        19: "bicycle",
    }

    def load_dataset(self, split: str) -> None:
        assert split in [
            "train",
            "test",
        ], f'Only split="train" and split="test" supported. Provided: split={split}'
        split = self.train_split_name if split == "train" else self.test_split_name
        self.data = load_from_disk("tmp/data/cityscapes_hg")[split]


class PascalVOC2012(HGDataset):
    name = "pascal_voc_2012"
    hg_name = None
    classes_cnt = 21
    test_split_name = "val"
    id2label = {
        0: "background/void",
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car ",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "potted plant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tv/monitor",
    }

    def load_dataset(self, split: str) -> None:
        assert split in [
            "train",
            "test",
        ], f'Only split="train" and split="test" supported. Provided: split={split}'
        split = self.train_split_name if split == "train" else self.test_split_name
        self.data = load_from_disk("tmp/data/pascal_voc_hg")[split]


class NYUDepthV2(HGDataset):
    name = "nyu_depth_v2"
    hg_name = "sayakpaul/nyu_depth_v2"
    classes_cnt = None
    test_split_name = "validation"
    label = "depth_map"
    depth_scale = 1


dataset_dict: Dict[str, HGDataset] = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "mnist": MNIST,
    "fashion-mnist": FashionMNIST,
    "imagenet": ImageNet,
    "foodseg103": FoodSeg103,
    "ade20k": ADE20k,
    "flowers102": Flowers102,
    "food101": Food101,
    "stl10": STL10,
    "cityscapes": CityScapes,
    "pascal_voc_2012": PascalVOC2012,
    "nyu_depth_v2": NYUDepthV2,
}


class CustomDatasetWithoutLabels(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = os.listdir(root)

    def __getitem__(self, index):
        path = self.root / self.images[index]
        x = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, -1

    def __len__(self):
        return len(self.images)


class HGClassificationDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        dataset: HGDataset,
        split: str,
        root: str,
        transform: Callable = None,
    ):
        self.root = Path(root)
        self.input_transform = transform
        self.dataset = dataset
        self.dataset.load_dataset(split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = (
            self.dataset.data[index][self.dataset.image],
            self.dataset.data[index][self.dataset.label],
        )
        if self.input_transform is not None:
            image = self.input_transform(image.convert("RGB"))
        return image, label

    def __len__(self):
        return len(self.dataset.data)


class HGSegmentationDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        dataset: HGDataset,
        split: str,
        root: str,
        transform: Callable = None,
    ):
        self.root = Path(root)
        self.transform = transform.transform
        self.dataset = dataset
        self.dataset.load_dataset(split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = (
            np.array(self.dataset.data[index][self.dataset.image]),
            np.array(self.dataset.data[index][self.dataset.label]),
        )
        if self.transform is not None:
            try:
                transformed = self.transform(image=image, mask=label)
            except:
                print(image.shape, label.shape, index)
                label = label.T  # for foodseg103 it happens only for some images
                print("Transposing label due to shape inconsistency, index:", index)
                transformed = self.transform(image=image, mask=label)

            image, label = torch.tensor(transformed["image"]), torch.LongTensor(
                transformed["mask"]
            )
            # convert to C, H, W
            image = image.permute(2, 0, 1)
        return image, label

    def __len__(self):
        return len(self.dataset.data)


class HGDepthDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        dataset: HGDataset,
        split: str,
        root: str,
        transform: Callable = None,
    ):
        self.root = Path(root)
        self.transform = transform.transform
        self.dataset = dataset
        self.dataset.load_dataset(split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = (
            np.array(self.dataset.data[index][self.dataset.image]),
            np.array(self.dataset.data[index][self.dataset.label]),
        )
        label = label / self.dataset.depth_scale
        if self.transform is not None:
            try:
                transformed = self.transform(image=image, mask=label)
            except:
                print(image.shape, label.shape, index)
                label = label.T
                print("Transposing label due to shape inconsistency, index:", index)
                transformed = self.transform(image=image, mask=label)

            image, label = torch.tensor(transformed["image"]), torch.tensor(
                transformed["mask"]
            )
            # convert to C, H, W
            image = image.permute(2, 0, 1)
        return image, label

    def __len__(self):
        return len(self.dataset.data)


def prepare_datasets(
    dataset: str,
    transform: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    sub_dir: Optional[Union[str, Path]] = None,
    no_labels: Optional[Union[str, Path]] = False,
    split: str = "train",
) -> Dataset:
    """Prepares the desired dataset.

    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        label_transform (Optional[Callable]): a transformation for label (segmentation task only).
        data_dir (Optional[Union[str, Path]], optional): the directory to load data from.
            Defaults to None.
        sub_dir (Optional[Union[str, Path]], optional): data directory to be appended to
        data_dir. Defaults to None.
        no_labels (Optional[bool], optional): if the custom dataset has no labels.
        split (str, optional): the split of the dataset. Defaults to "train".

    Returns:
        Dataset: the desired dataset with transformations.
    """

    if data_dir is None:
        sandbox_folder = Path(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        data_dir = sandbox_folder / "datasets"

    if sub_dir is not None:
        data_dir = Path(f"{data_dir}/{sub_dir}")

    if dataset in ("foodseg103", "ade20k", "cityscapes", "pascal_voc_2012"):
        hg_dataset = dataset_dict[dataset](
            root=data_dir,
        )
        dataset = HGSegmentationDataset(
            dataset=hg_dataset,
            split=split,
            root=data_dir,
            transform=transform,
        )
    elif dataset == "imagenet":
        return torchvision.datasets.ImageNet(
            data_dir, split=split, transform=transform
        )  # switch from hg, we don't want any discrepancies here.
    elif dataset == "nyu_depth_v2":
        hg_dataset = dataset_dict[dataset](
            root=data_dir,
        )
        dataset = HGDepthDataset(
            dataset=hg_dataset,
            split=split,
            root=data_dir,
            transform=transform,
        )
    elif dataset in dataset_dict.keys():
        hg_dataset = dataset_dict[dataset](
            root=data_dir,
        )
        dataset = HGClassificationDataset(
            dataset=hg_dataset,
            split=split,
            root=data_dir,
            transform=transform,
        )

    elif dataset not in dataset_dict.keys():
        assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist"
        dataset_class = CustomDatasetWithoutLabels if no_labels else ImageFolder
        dataset = dataset_class(data_dir, transform)
    return dataset
