# Data

This directory implements all logic regarding loading and transforming the datasets.

## datasets.py

Implements dataset helper classes: HGClassificationDataset and HGSegmentationDataset for two downstream tasks: classification and segmentation. All datasets for these tasks are unified to follow similar API.

### Datasets supported

Classification:

* cifar10
* cifar100
* mnist
* fashion
* imagenet
* flowers102
* food101
* stl10

Segmentation:

* foodseg103

## loader.py

Provides `prepare_dataloader` helper function for data loading.

## transforms.py

Provides all logic needed for the data transformation, dependent on dataset and downstream task.