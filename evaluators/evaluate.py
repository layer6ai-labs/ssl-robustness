import os
import numpy as np
import torch
from evaluators.metrics import align_loss, uniform_loss, cosine_similarity
from evaluators.utils import get_macc_miou, sliding_window_inference
from models.downstream import Segmenter
from models.ssl_models.ssl import Encoder
from evaluators.attack import Attack
from tqdm import tqdm
from torch.utils.data import DataLoader

from typing import Callable

from models.downstream_models.depther.encoder_decoder import DepthEncoderDecoder

import numpy as np


def evaluate_alignment_uniformity(loader, ssl_model, device):
    aligns = []
    norm_aligns = []
    proj_aligns = []
    norm_proj_aligns = []

    uniforms = []
    norm_uniforms = []
    proj_uniforms = []
    norm_proj_uniforms = []

    pos_cosines = []
    neg_cosines = []
    proj_pos_cosines = []
    proj_neg_cosines = []

    norms = []
    proj_norms = []

    ssl_model = ssl_model.to(device)
    ssl_model.eval()
    for idx, (data, label) in enumerate(loader):
        im_x = data[0]
        im_y = data[1]
        with torch.no_grad():
            reps, projs = ssl_model.get_representations(
                torch.cat([im_x.to(device), im_y.to(device)]), with_projection=True
            )
            rep_x, rep_y = reps.chunk(2)
            proj_x, proj_y = projs.chunk(2)
            norm_rep_x = torch.nn.functional.normalize(rep_x, dim=1)
            norm_rep_y = torch.nn.functional.normalize(rep_y, dim=1)
            norm_proj_x = torch.nn.functional.normalize(proj_x, dim=1)
            norm_proj_y = torch.nn.functional.normalize(proj_y, dim=1)

            aligns.append(align_loss(rep_x, rep_y).item())
            norm_aligns.append(align_loss(norm_rep_x, norm_rep_y).item())
            proj_aligns.append(align_loss(proj_x, proj_y).item())
            norm_proj_aligns.append(align_loss(norm_proj_x, norm_proj_y).item())

            uniforms.append(uniform_loss(rep_x).item())
            norm_uniforms.append(uniform_loss(norm_rep_x).item())
            proj_uniforms.append(uniform_loss(proj_x).item())
            norm_proj_uniforms.append(uniform_loss(norm_proj_x).item())

            pos_cosine, neg_cosine = cosine_similarity(rep_x, rep_y)
            pos_cosines.append(pos_cosine.item())
            neg_cosines.append(neg_cosine.item())
            proj_pos_cosine, proj_neg_cosine = cosine_similarity(proj_x, proj_y)
            proj_pos_cosines.append(proj_pos_cosine.item())
            proj_neg_cosines.append(proj_neg_cosine.item())

            norms.append(torch.norm(rep_x, dim=1).mean().item())
            proj_norms.append(torch.norm(proj_x, dim=1).mean().item())

    avg_align = np.mean(aligns)
    avg_norm_align = np.mean(norm_aligns)
    avg_proj_align = np.mean(proj_aligns)
    avg_norm_proj_align = np.mean(norm_proj_aligns)
    avg_uniform = np.mean(uniforms)
    avg_norm_uniform = np.mean(norm_uniforms)
    avg_proj_uniform = np.mean(proj_uniforms)
    avg_norm_proj_uniform = np.mean(norm_proj_uniforms)
    avg_pos_cosine = np.mean(pos_cosines)
    avg_neg_cosine = np.mean(neg_cosines)
    avg_proj_pos_cosine = np.mean(proj_pos_cosines)
    avg_proj_neg_cosine = np.mean(proj_neg_cosines)
    avg_norm = np.mean(norms)
    avg_proj_norm = np.mean(proj_norms)
    print(
        f"Alignment Loss at Representation Layer: {avg_align}, "
        f"Normalized Alignment Loss at Representation Layer: {avg_norm_align}, "
        f"Alignment Loss after Projection: {avg_proj_align}, "
        f"Normalized Alignment Loss after Projection: {avg_norm_proj_align}, "
        f"Uniformity Loss at Representation Layer: {avg_uniform}, "
        f"Normalized Uniformity Loss at Representation Layer: {avg_norm_uniform}, "
        f"Uniformity Loss after Projection: {avg_proj_uniform}, "
        f"Normalized Uniformity Loss after Projection: {avg_norm_proj_uniform}, "
        f"Cosine Similarity Positive Pairs at Representation Layer: {avg_pos_cosine}, "
        f"Cosine Similarity Negative Pairs at Representation Layer: {avg_neg_cosine}, "
        f"Cosine Similarity Positive Pairs after Projection: {avg_proj_pos_cosine}, "
        f"Cosine Similarity Negative Pairs after Projection: {avg_proj_neg_cosine},",
        f"Average Norm at Representation Layer: {avg_norm},",
        f"Average Norm after Projections: {avg_proj_norm},",
    )

    return [
        avg_align,
        avg_norm_align,
        avg_proj_align,
        avg_norm_proj_align,
        avg_uniform,
        avg_norm_uniform,
        avg_proj_uniform,
        avg_norm_proj_uniform,
        avg_pos_cosine,
        avg_neg_cosine,
        avg_proj_pos_cosine,
        avg_proj_neg_cosine,
        avg_norm,
        avg_proj_norm,
    ]


def evaluate_accuracy(classifier, loader, device):
    classifier.clf.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = classifier(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    classifier.clf.train()
    return correct / total


def evaluate_segmenter_accuracy_miou(
    segmenter: Segmenter,
    dataloader: DataLoader,
    device,
    model_cfg: dict,
    task_cfg: dict,
) -> float:
    segmenter.head.eval()
    img_size = segmenter.encoder.image_size
    num_classes = segmenter.head.num_labels
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, label = batch
            inputs = inputs.to(device)
            pred = (
                sliding_window_inference(
                    model=segmenter,
                    num_classes=num_classes,
                    input_data=inputs,
                    crop_size=(img_size, img_size),
                    stride=(model_cfg.stride_size, model_cfg.stride_size),
                    batch_size=task_cfg.batch_size,
                    return_logits=True,
                )
                .cpu()
                .argmax(dim=1)[0]
            )
            predictions.append(pred)
            labels.append(label)

    segmenter.head.train()
    return get_macc_miou(
        preds=predictions, labels=labels, num_classes=segmenter.head.num_labels
    )


def adversarial_classification_accuracy(
    attack, encoder, data_loader, device, classifier, downstream=False
):
    correct = 0
    total = 0
    for idx, (images, labels) in tqdm(enumerate(data_loader)):
        if downstream:
            adv_img, success = attack.run(
                images=images.to(device),
                labels=labels.to(device),
                model=classifier,
                precomputed_original_representations=None,
                return_step_by_step=False,  # TODO returns error otherwise; fix it
            )
            correct += labels.size(0) - success.sum().item()
        else:
            adv_img, adv_representation = attack.run(
                images=images.to(device),
                model=encoder,
                precomputed_original_representations=None,
                return_step_by_step=False,  # TODO returns error otherwise; fix it
            )
            outputs = classifier(adv_img)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.to(device)).sum().item()
        total += labels.size(0)
        if (idx + 1) * data_loader.batch_size >= 1000:
            break
    return correct / total


def get_patch_embeddings_pgd(
    segmenter: Segmenter,
    encoder: Encoder,
    n_labels: int,
    device,
    model_cfg,
    task_cfg,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def _internal_get(image: torch.Tensor) -> torch.Tensor:
        return sliding_window_inference(
            segmenter,
            encoder=encoder,
            num_classes=n_labels,
            input_data=image.to(device),
            crop_size=(
                encoder.image_size,
                encoder.image_size,
            ),
            stride=(model_cfg.stride_size, model_cfg.stride_size),
            batch_size=task_cfg.batch_size,
            return_logits=False,
        )

    return _internal_get


def adversarial_segmenter_accuracy_miou(
    attack: Attack,
    encoder: Encoder,
    data_loader: DataLoader,
    device,
    segmenter: Segmenter,
    n_labels: int,
    model_cfg: dict,
    task_cfg: dict,
    downstream: bool = False,
) -> float:
    get_patch_embeddings = get_patch_embeddings_pgd(
        segmenter=segmenter,
        encoder=encoder,
        n_labels=n_labels,
        device=device,
        model_cfg=model_cfg,
        task_cfg=task_cfg,
    )

    predictions, labels_all = [], []
    for idx, (images, labels) in tqdm(enumerate(data_loader)):
        if downstream:
            adv_img = attack.run(
                image=images.to(device),
                model=segmenter,
                label=labels,
                device=device,
                model_cfg=model_cfg,
                task_cfg=task_cfg,
            )
        else:
            adv_img, _ = attack.run(
                images=images.to(device),
                model=encoder,
                precomputed_original_representations=None,
                get_representation=get_patch_embeddings,
                return_step_by_step=False,  # TODO returns error otherwise; fix it
            )
        with torch.no_grad():
            preds = (
                sliding_window_inference(
                    model=segmenter,
                    num_classes=n_labels,
                    input_data=adv_img.to(device),
                    crop_size=(
                        segmenter.encoder.image_size,
                        segmenter.encoder.image_size,
                    ),
                    stride=(model_cfg.stride_size, model_cfg.stride_size),
                    batch_size=task_cfg.batch_size,
                    return_logits=True,
                )
                .cpu()
                .argmax(dim=1)
            )

        predictions += [pred for pred in preds]
        labels_all += [label for label in labels]

    metrics = get_macc_miou(predictions, labels_all, n_labels)
    return metrics["mean_accuracy"], metrics["mean_iou"]


def adversarial_segmenter_all_per_step(
    attack: Attack,
    encoder: Encoder,
    data_loader: DataLoader,
    device,
    segmenter: Segmenter,
    n_labels: int,
    model_cfg: dict,
    task_cfg: dict,
    action_cfg: dict,
    downstream: bool = False,
) -> float:
    get_patch_embeddings = get_patch_embeddings_pgd(
        segmenter=segmenter,
        encoder=encoder,
        n_labels=n_labels,
        device=device,
        model_cfg=model_cfg,
        task_cfg=task_cfg,
    )

    images_all, adv_images_all, labels_all = [], [], []
    for idx, (image, label) in tqdm(enumerate(data_loader)):
        if downstream:
            adv_images = attack.run(
                image=image.to(device),
                model=segmenter,
                label=label,
                device=device,
                model_cfg=model_cfg,
                task_cfg=task_cfg,
                return_step_by_step=True,
            )
        else:
            adv_images, _ = attack.run(
                images=image.to(device),
                model=encoder,
                precomputed_original_representations=None,
                get_representation=get_patch_embeddings,
                return_step_by_step=True,
                steps_to_save=action_cfg.attack_num_steps,
            )
        images_all.append(image)
        adv_images_all.append(adv_images)
        labels_all.append(label)

        if (idx + 1) >= 100:
            break

    orig_embeddings = []
    for image in tqdm(images_all):
        with torch.no_grad():
            out = get_patch_embeddings(image).cpu()
            orig_embeddings.append(out)

    mious, accs, L2s = [], [], []
    for idx in tqdm(range(attack.num_steps + 1)):
        predictions, adv_embeddings = [], []
        for sample_idx in range(len(adv_images_all)):
            with torch.no_grad():
                for return_logits in (True, False):
                    out = sliding_window_inference(
                        model=segmenter,
                        encoder=encoder,
                        num_classes=n_labels,
                        input_data=adv_images_all[sample_idx][idx].to(device),
                        crop_size=(
                            segmenter.encoder.image_size,
                            segmenter.encoder.image_size,
                        ),
                        stride=(model_cfg.stride_size, model_cfg.stride_size),
                        batch_size=task_cfg.batch_size,
                        return_logits=return_logits,
                    ).cpu()
                    if return_logits:
                        predictions.append(out.argmax(dim=1))
                    else:
                        adv_embeddings.append(out)
        metrics = get_macc_miou(predictions, labels_all, n_labels)
        mious.append(metrics["mean_iou"])
        accs.append(metrics["mean_accuracy"])
        L2s.append(
            [
                torch.norm(orig_embeddings[sample_idx] - adv_embeddings[sample_idx])
                .mean()
                .item()
                for sample_idx in range(len(adv_images_all))
            ]
        )

    os.makedirs("tmp/out", exist_ok=True)
    np.savez(
        f"tmp/out/adversarial_segmenter_all_per_step_{encoder.name}_{task_cfg.dataset}_{attack.name}_{action_cfg.attack_epsilon}_{action_cfg.attack_num_steps}_{action_cfg.attack_lr}.npz",
        mious=mious,
        accs=accs,
        L2s=np.array(L2s),
    )


# Only evaluate on the batch size of 1
def evaluate_depth_estimation(
    encoder_decoder: DepthEncoderDecoder,
    dataLoader: DataLoader,
    device: torch.device,
):
    sum_loss = 0
    model = encoder_decoder.decode_head
    model.encoder.eval()
    model.encoder.freeze()
    mse = torch.nn.MSELoss()
    with torch.no_grad():
        for image, depth in tqdm(dataLoader, total=len(dataLoader)):
            assert image.shape[0] == 1
            image = image.to(device)
            depth = depth.cpu()
            valid_mask = depth > 0
            max_depth = encoder_decoder.decode_head.max_depth
            if max_depth is not None:
                valid_mask = torch.logical_and(depth > 0, depth <= max_depth)
            depth = depth[valid_mask]
            pred = encoder_decoder.whole_inference(image, None, rescale=True)
            pred = pred.squeeze(1).detach().cpu()
            pred = pred[valid_mask]
            rmse = torch.sqrt(mse(pred, depth)).item()
            sum_loss += rmse

    return sum_loss / len(dataLoader)


def adversarial_depth_estimation(
    attack, data_loader, device, encoder_decoder, task_cfg, downstream=False
):
    # mse = torch.nn.MSELoss(reduction='none')
    mse = torch.nn.MSELoss()
    sum_loss = 0
    for image, depth in tqdm(data_loader):
        assert image.shape[0] == 1
        if downstream:
            adv_img = attack.run(
                model=encoder_decoder,
                image=image.to(device),
                label=depth.to(device),
                device=device,
                task_cfg=task_cfg,
                return_step_by_step=False,
            )
        else:
            adv_img, adv_representation = attack.run(
                images=image.to(device),
                model=encoder_decoder,
                precomputed_original_representations=None,
                return_step_by_step=False,
            )
        with torch.no_grad():
            preds = encoder_decoder.whole_inference(
                adv_img, None, rescale=True
            ).squeeze(1)
            # rmse = torch.sqrt(mse(preds, depths).mean(dim=(1, 2))).sum().item()
            rmse = torch.sqrt(mse(preds.cpu(), depth.cpu())).item()
            sum_loss += rmse
    return sum_loss / len(data_loader)
