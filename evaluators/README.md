# Evaluators

This directory implements all logic regarding downstream task performance evaluation, adversarial attack, and uniformity-allignment evaluation.

## attack.py

Implements all logic for adversarial attacks. Attack class is an abstract class derived by PGD class implementing the base PGD attack. It allows both attack on encoder's representation (classification) and attack on encoder's patch embeddings (segmentation), by changing `run(get_representation=encoder.get_patch_embeddings)` (ViT-based encoders only). DownstreamPGD attacks encoder+classification_head stack in downstream attack manner. All new attacks should be implemented in this file.

## evaluate.py

Implements evaluation methods for classification accuracy, alignment-uniformity, segmentation mIoU and accuracy, and adversarial classification accuracy and segmentation miou and accuracy.

## loss.py

Implements different optimization objectives for the adversarial representation shift: L2, CosineSimilarity, and CrossEntropy. For now it's established to use L2 distance as an attack objective. Supports normalization wrt representation space mean and std per dimension.

## metrics.py

Constains implementation of metrics used for alignment-uniformity evaluation