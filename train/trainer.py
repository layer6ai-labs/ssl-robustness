import torch
from tqdm import tqdm
import numpy as np
from evaluators.evaluate import (
    evaluate_accuracy,
    evaluate_segmenter_accuracy_miou,
    get_macc_miou,
    evaluate_depth_estimation,
)
from models.downstream import Segmenter, SegmenterHead, LinearClassifier
from models.downstream_models.depther.encoder_decoder import DepthEncoderDecoder
from models.ssl_models.ssl import Encoder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import numpy as np


def poly_lr_scheduler(  # from dino-v2 config files
    iter_num,
    total_iters,
    warmup_iters,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
):
    if iter_num < warmup_iters:
        factor = 1 - (iter_num / warmup_iters) * warmup_ratio
    else:
        factor = (1 - 1 / (total_iters - iter_num + 1)) ** power

    return factor


def train_classifier(
    classifier: LinearClassifier,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
):
    best_acc = 0
    if classifier.exits_clf():
        classifier.load_clf()
        val_acc = evaluate_accuracy(classifier, eval_dataloader, classifier.device)
        print(f"Epoch {classifier.epochs} accuracy: {val_acc}")
    else:
        optimizer = torch.optim.Adam(classifier.clf.parameters(), lr=classifier.lr)
        classifier.encoder.freeze()
        loss = torch.nn.CrossEntropyLoss()
        for epoch in tqdm(range(classifier.epochs)):
            for batch in train_dataloader:
                inputs, labels = batch
                inputs = inputs.to(classifier.device)
                labels = labels.to(classifier.device)
                logits = classifier(inputs)
                l = loss(logits, labels)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            val_acc = evaluate_accuracy(classifier, eval_dataloader, classifier.device)
            if val_acc > best_acc:
                best_acc = val_acc
                classifier.save_clf()
            print(f"Epoch {epoch} accuracy: {val_acc}")
    return val_acc


def train_segmenter(
    task_cfg: dict,
    model_cfg: dict,
    head: SegmenterHead,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    encoder: Encoder,
    device,
    n_labels: int,
) -> [float, float]:
    best_miou_eval = 0
    if task_cfg.eval_only:
        if not head.exists_segmenter():
            raise FileNotFoundError("No segmenter found")
        head.load_segmenter()
        eval_metrics = evaluate_segmenter_accuracy_miou(
            segmenter=Segmenter(num_labels=n_labels, encoder=encoder, head=head),
            dataloader=eval_dataloader,
            device=device,
            model_cfg=model_cfg,
            task_cfg=task_cfg,
        )
        print(
            f"mIoU: {round(eval_metrics['mean_iou'], 2)},",
            f"acc: {round(eval_metrics['mean_accuracy'], 2)},",
        )
        return eval_metrics["mean_accuracy"], eval_metrics["mean_iou"]
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=task_cfg.lr, weight_decay=1e-4, betas=(0.9, 0.999)
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda iter_num: poly_lr_scheduler(
            iter_num,
            len(train_dataloader) * task_cfg.epochs,
            task_cfg.warmup_iterations,
            task_cfg.warmup_ratio,
        ),
    )
    loss = torch.nn.CrossEntropyLoss()
    with tqdm(range(task_cfg.epochs), desc=f"{encoder.name}") as tt:
        for epoch in tt:
            epoch_predicted, epoch_labels = [], []
            for batch in train_dataloader:
                images, labels = batch
                images = images.to(device)
                with torch.no_grad():
                    embeddings = encoder.get_patch_embeddings(images)
                labels = labels.to(device)
                logits = head(embeddings)
                l = loss(logits, labels)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                # for evaluation
                epoch_predicted += [
                    logit.unsqueeze(0) for logit in logits.argmax(dim=1).cpu()
                ]
                epoch_labels += [label.unsqueeze(0) for label in labels.cpu()]
                scheduler.step()
                # opti_lr = optimizer.param_groups[0]["lr"]
                # tt.set_description(f"{opti_lr=}")

            if (epoch + 1) % task_cfg.eval_every_n_epochs:
                continue

            indices = np.random.permutation(len(epoch_predicted))[
                : task_cfg.n_samples_to_eval_train
            ]
            train_metrics = get_macc_miou(
                np.array(epoch_predicted)[indices],
                np.array(epoch_labels)[indices],
                n_labels,
            )

            eval_metrics = evaluate_segmenter_accuracy_miou(
                segmenter=Segmenter(num_labels=n_labels, encoder=encoder, head=head),
                dataloader=eval_dataloader,
                device=device,
                model_cfg=model_cfg,
                task_cfg=task_cfg,
            )
            print(
                f"\nEpoch {epoch}",
                f"mIoU_t: {round(train_metrics['mean_iou'], 2)},",
                f"mIoU_e: {round(eval_metrics['mean_iou'], 2)},",
                f"acc_t: {round(train_metrics['mean_accuracy'], 2)},",
                f"acc_e: {round(eval_metrics['mean_accuracy'], 2)}, {encoder.name}",
            )

            if eval_metrics["mean_iou"] > best_miou_eval:
                best_miou_eval = eval_metrics["mean_iou"]
                head.save_segmenter()
    eval_metrics = evaluate_segmenter_accuracy_miou(
                segmenter=Segmenter(num_labels=n_labels, encoder=encoder, head=head),
                dataloader=eval_dataloader,
                device=device,
                model_cfg=model_cfg,
                task_cfg=task_cfg,
            )
    return eval_metrics["mean_accuracy"], eval_metrics["mean_iou"]


def train_depth_estimator(
    encoder_decoder: DepthEncoderDecoder,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    task_cfg: dict,
    metrics: dict,
    device: torch.device,
):
    epochs = task_cfg["epochs"]
    model = encoder_decoder.decode_head

    if model.exists():
        model.load()
        rmse = evaluate_depth_estimation(encoder_decoder, eval_dataloader, device)
        print(
            f"RMSE: {round(rmse, 2)},",
        )
        metrics["rmse"].append(rmse)
        return

    if task_cfg["mode"] == "frozen":
        model.encoder.eval()
        model.encoder.freeze()
        optimizer = torch.optim.AdamW(
            model.conv_depth.parameters(),
            lr=task_cfg["lr"],
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
    elif task_cfg["mode"] == "finetune":
        model.encoder.train()
        model.encoder.unfreeze()
        optimizer = torch.optim.AdamW(
            model.conv_depth.parameters(),
            lr=task_cfg["lr"],
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
    else:
        raise ValueError("Invalid mode")

    best_rmse = 30

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        num_batches = 0
        for i, (image, depth) in enumerate(train_dataloader):
            image = image.to(device)
            depth = depth.to(device)

            # forward pass
            losses = encoder_decoder.forward_train(image, None, depth)

            # Backward and optimize
            optimizer.zero_grad()
            # loss = losses['decode.loss_depth']
            loss = losses["decode.loss_depth"] + losses["decode.loss_grad"]
            loss.backward()
            optimizer.step()

            # Calculate loss
            running_loss += loss.item()
            num_batches += 1
        # Report epoch loss
        print(
            "Epoch: [{}/{}] Epoch Loss: {:.4f}\n".format(
                (epoch + 1), epochs, (running_loss / num_batches)
            )
        )

        rmse = evaluate_depth_estimation(encoder_decoder, eval_dataloader, device)
        print(
            f"RMSE: {round(rmse, 4)},",
        )
        if rmse < best_rmse:
            best_rmse = rmse
            model.save()

    rmse = evaluate_depth_estimation(encoder_decoder, eval_dataloader, device)
    metrics["rmse"].append(rmse)
    print(f"RMSE: {round(rmse, 4)}")
