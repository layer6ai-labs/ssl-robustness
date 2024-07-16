import random

import hydra
import pandas
import torch
from omegaconf import DictConfig, OmegaConf

from data import local_datasets
from data import get_dataloader
from data.transforms import mean_dict, std_dict
from evaluators.attack import SegPGD, PGD_Sign, DownstreamPGD, AutoAttack, DepthPGD
from evaluators.evaluate import (
    evaluate_alignment_uniformity,
    adversarial_classification_accuracy,
    adversarial_segmenter_accuracy_miou,
    adversarial_segmenter_all_per_step,
    adversarial_depth_estimation,
)
from evaluators.loss import loss_dict, Loss, Cosine
from models.downstream import LinearClassifier, SegmenterHead, Segmenter
from models.ssl_models import load_ssl_models
from models.ssl_models.utils import encoder_sanity_check
from train.trainer import train_classifier, train_segmenter, train_depth_estimator
from utils import get_writer, get_ckpts
from models.downstream_models.depther.depth_head import BNHead
from models.downstream_models.depther.encoder_decoder import DepthEncoderDecoder

from fine_tune.deacl.run import run as deacl_run


@hydra.main(version_base=None, config_path="conf/")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    action_cfg = cfg.action
    task_cfg = cfg.task
    ssl_model_cfg = cfg.ssl_model

    # set seed for reproducibility
    torch.manual_seed(action_cfg.seed)
    torch.cuda.manual_seed(action_cfg.seed)
    random.seed(action_cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(action_cfg.device)
    class_cnt = local_datasets.dataset_dict[task_cfg.dataset].classes_cnt
    dataset_name = local_datasets.dataset_dict[task_cfg.dataset].name

    print("Start", ssl_model_cfg.name, task_cfg.dataset)
    writer = get_writer(
        {
            "action": dict(action_cfg),
            "task": dict(task_cfg),
        }
    )
    if action_cfg.action == "uniformity_alignment":
        print("uniformity_alignment")
        results, header = evaluate_uniformity_alignment(
            action_cfg, task_cfg, ssl_model_cfg, device, class_cnt
        )
        writer.write_pandas(
            f"alignment_uniformity", pandas.DataFrame(results, columns=header)
        )
    if action_cfg.action == "adversarial_finetuning":
        print("adversarial_finetuning")
        result = finetune_adversarially(
            action_cfg, task_cfg, ssl_model_cfg, device, class_cnt
        )
        writer.write_pandas(f"file_name", pandas.DataFrame(result))
    if action_cfg.action == "downstream_training":
        print("downstream_training")
        result = train_downstream(
            action_cfg, task_cfg, ssl_model_cfg, device, class_cnt, dataset_name
        )
        writer.write_pandas(f"acc", pandas.DataFrame(result))
    if action_cfg.action == "adversarial_attack":
        print("adversarial_attack")
        adv_acc = evaluate_adversarially(
            action_cfg, task_cfg, ssl_model_cfg, device, class_cnt
        )
        writer.write_pandas(f"adv_acc", pandas.DataFrame(adv_acc))


def evaluate_uniformity_alignment(
    action_cfg: DictConfig,
    task_cfg: DictConfig,
    ssl_model_cfg: DictConfig,
    device: torch.device,
    class_cnt: int,
):
    header = [
        "avg_align",
        "avg_norm_align",
        "avg_proj_align",
        "avg_norm_proj_align",
        "avg_uniform",
        "avg_norm_uniform",
        "avg_proj_uniform",
        "avg_norm_proj_uniform",
        "avg_pos_cosine",
        "avg_neg_cosine",
        "avg_proj_pos_cosine",
        "avg_proj_neg_cosine",
        "avg_norm",
        "avg_proj_norm",
        "ckpts",
    ]
    results = []
    loader = get_dataloader(
        task_cfg.dataset,
        action_cfg.data_root,
        batch_size=task_cfg.batch_size,
        two_views=True,
        split="train",
        crop_size=ssl_model_cfg.image_size,
    )
    for path in get_ckpts(
        ssl_model_cfg.ckpt.ckpts_folder, ssl_model_cfg.ckpt.final_ckpt
    ):
        ssl_model = load_ssl_models(ssl_model_cfg, load_weights=True, ckpt_path=path)
        res = evaluate_alignment_uniformity(loader, ssl_model, device)
        res.append(path)
        results.append(res)
    return results, header


def finetune_adversarially(
    action_cfg: DictConfig,
    task_cfg: DictConfig,
    ssl_model_cfg: DictConfig,
    device: torch.device,
    class_cnt: int,
):
    normalize = ssl_model_cfg.normalize if "normalize" in ssl_model_cfg else True
    mean = mean_dict["imagenet" if normalize else "no_norm"]
    std = std_dict["imagenet" if normalize else "no_norm"]
    train_loader = get_dataloader(
        task_cfg.dataset,
        action_cfg.data_root,
        batch_size=task_cfg.batch_size,
        two_views=False,
        augment=False,
        split="train",
        crop_size=ssl_model_cfg.image_size,
        num_workers=task_cfg.num_workers,
        normalize=normalize,
    )
    if task_cfg.task == "deacl":
        loss = Cosine()
        attack = PGD_Sign(
            loss=loss,
            num_steps=task_cfg.train_num_steps,
            lr=task_cfg.train_attack_lr,
            eps_budget=task_cfg.train_epsilon,
            mean=mean,
            std=std,
        )
        deacl_run(
            train_loader=train_loader,
            attack=attack,
            task_cfg=task_cfg,
            model_cfg=ssl_model_cfg,
            action_cfg=action_cfg,
        )


def train_downstream(
    action_cfg: DictConfig,
    task_cfg: DictConfig,
    ssl_model_cfg: DictConfig,
    device: torch.device,
    class_cnt: int,
    dataset_name: str,
):
    train_loader = get_dataloader(
        task_cfg.dataset,
        action_cfg.data_root,
        batch_size=task_cfg.batch_size,
        two_views=False,
        augment=False,
        split="train",
        crop_size=ssl_model_cfg.image_size,
        num_workers=action_cfg.num_workers,
        normalize=ssl_model_cfg.normalize if "normalize" in ssl_model_cfg else True,
    )
    eval_loader = get_dataloader(
        task_cfg.dataset,
        action_cfg.data_root,
        batch_size=(
            1 if task_cfg.task != "classification" else task_cfg.batch_size
        ),  # because of sliding window inference
        two_views=False,
        augment=False,
        split="test",
        crop_size=ssl_model_cfg.image_size,
        num_workers=action_cfg.num_workers,
        normalize=ssl_model_cfg.normalize if "normalize" in ssl_model_cfg else True,
    )
    if task_cfg.task == "classification":
        acc = {
            "val_acc": [],
            "ckpts": get_ckpts(
                ssl_model_cfg.ckpt.ckpts_folder, ssl_model_cfg.ckpt.final_ckpt
            ),
        }
        for path in acc["ckpts"]:
            print(f"Loading SSL Encoder from path: {path}")
            encoder = load_ssl_models(
                ssl_model_cfg, load_weights=True, ckpt_path=path
            ).encoder.to(device)
            clf = LinearClassifier(
                encoder,
                task_cfg.dataset,
                task_cfg.lr,
                task_cfg.epochs,
                class_cnt,
                device=device,
                is_train=True,
            )
            val_acc = train_classifier(clf, train_loader, eval_loader)
            encoder_sanity_check(ssl_model_cfg, path, device, clf)
            acc["val_acc"].append(val_acc)
            return acc

    elif task_cfg.task == "segmentation":
        acc = {
            "val_acc": [],
            "val_miou": [],
            "ckpts": get_ckpts(
                ssl_model_cfg.ckpt.ckpts_folder, ssl_model_cfg.ckpt.final_ckpt
            ),
        }
        for ckpt_name in acc["ckpts"]:
            print(f"Loading SSL Encoder from path: {ckpt_name}")
            encoder = load_ssl_models(
                ssl_model_cfg,
                load_weights=True,
                ckpt_path=ckpt_name,
            ).encoder.to(device)
            head = SegmenterHead(
                class_cnt,
                encoder,
                dataset_name=dataset_name,
                epochs=task_cfg.epochs,
                lr=task_cfg.lr,
            ).to(device)
            accuracy, miou = train_segmenter(
                task_cfg=task_cfg,
                model_cfg=ssl_model_cfg,
                head=head,
                train_dataloader=train_loader,
                eval_dataloader=eval_loader,
                encoder=encoder,
                device=device,
                n_labels=class_cnt,
            )
            encoder_sanity_check(
                ssl_model_cfg,
                ckpt_name,
                device,
                Segmenter(num_labels=class_cnt, encoder=encoder, head=head),
            )
            acc["val_acc"].append(accuracy)
            acc["val_miou"].append(miou)
        return acc

    elif task_cfg.task == "depth_estimation":
        metrics = {"rmse": []}
        metrics["ckpts"] = get_ckpts(
            ssl_model_cfg.ckpt.ckpts_folder, ssl_model_cfg.ckpt.final_ckpt
        )
        for ckpt_name in metrics["ckpts"]:
            print(f"Loading SSL Encoder from path: {ckpt_name}")
            encoder = load_ssl_models(
                ssl_model_cfg,
                load_weights=True,
                ckpt_path=ckpt_name,
            ).encoder.to(device)
            decode_head = BNHead(encoder, task_cfg, ssl_model_cfg, ckpt_path=ckpt_name, device=device)
            encoder_decoder = DepthEncoderDecoder(encoder, decode_head)
            train_depth_estimator(
                encoder_decoder,
                train_dataloader=train_loader,
                eval_dataloader=eval_loader,
                task_cfg=task_cfg,
                device=device,
                metrics=metrics,
            )

            encoder_sanity_check(ssl_model_cfg, ckpt_name, device, encoder_decoder)

        return metrics
    else:
        raise ValueError("Task not supported")


def evaluate_adversarially(
    action_cfg: DictConfig,
    task_cfg: DictConfig,
    ssl_model_cfg: DictConfig,
    device: torch.device,
    class_cnt: int,
):
    normalize = ssl_model_cfg.normalize if "normalize" in ssl_model_cfg else True
    mean = mean_dict["imagenet" if normalize else "no_norm"]
    std = std_dict["imagenet" if normalize else "no_norm"]
    eval_loader = get_dataloader(
        task_cfg.dataset,
        action_cfg.data_root,
        batch_size=task_cfg.batch_size if task_cfg.task == "classification" else 1,
        two_views=False,
        augment=False,
        split="test",
        crop_size=ssl_model_cfg.image_size,
        normalize=normalize,
    )
    if task_cfg.task == "classification":
        adv_acc = {
            "encoder_adv_acc": [],
            "downstream_adv_acc": [],
            "auto_adv_acc": [],
            "ckpts": get_ckpts(
                ssl_model_cfg.ckpt.ckpts_folder, ssl_model_cfg.ckpt.final_ckpt
            ),
        }
        for path in adv_acc["ckpts"]:
            print(f"Loading SSL Encoder from path: {path}")
            encoder = load_ssl_models(
                ssl_model_cfg,
                load_weights=True,
                ckpt_path=ssl_model_cfg.ckpt.final_ckpt,
            ).encoder.to(device)
            clf = LinearClassifier(
                encoder,
                task_cfg.dataset,
                task_cfg.lr,
                task_cfg.epochs,
                class_cnt,
                device=device,
                is_train=True,
            )
            if clf.exits_clf():
                clf.load_clf()
            else:
                raise RuntimeError("Please train the classifier before the attack")
            clf.encoder.unfreeze()
            # encoder attack
            loss: Loss = loss_dict[action_cfg.attack_loss]()
            # If planning to test on different attacks, pass this as an argument
            encoder_attack = PGD_Sign(
                loss=loss,
                num_steps=action_cfg.attack_num_steps,
                eps_budget=action_cfg.attack_epsilon,
                lr=action_cfg.attack_lr,
                mean=mean,
                std=std,
            )
            encoder_adv_acc = adversarial_classification_accuracy(
                encoder_attack, encoder, eval_loader, device, clf
            )
            adv_acc["encoder_adv_acc"].append(encoder_adv_acc)
            # downstream attack
            downstream_attack = DownstreamPGD(
                num_steps=action_cfg.attack_num_steps,
                eps_budget=action_cfg.attack_epsilon,
                lr=action_cfg.attack_lr,
                mean=mean,
                std=std,
            )
            downstream_adv_acc = adversarial_classification_accuracy(
                downstream_attack, encoder, eval_loader, device, clf, downstream=True
            )
            adv_acc["downstream_adv_acc"].append(downstream_adv_acc)
            print(
                f"encoder_adv_acc: {encoder_adv_acc}, "
                f"downstream_adv_acc: {downstream_adv_acc}, ",
                end="",
            )
            autoattack = AutoAttack(
                eps_budget=action_cfg.attack_epsilon,
                mean=mean,
                std=std,
            )
            auto_adv_acc = adversarial_classification_accuracy(
                autoattack, encoder, eval_loader, device, clf, downstream=True
            )
            adv_acc["auto_adv_acc"] = auto_adv_acc
            print(f"auto_adv_acc: {auto_adv_acc}")

    elif task_cfg.task == "segmentation":
        adv_acc = {
            "encoder_adv_acc": [],
            "downstream_adv_acc": [],
            "encoder_adv_miou": [],
            "downstream_adv_miou": [],
            "ckpts": get_ckpts(
                ssl_model_cfg.ckpt.ckpts_folder, ssl_model_cfg.ckpt.final_ckpt
            ),
        }
        for path in adv_acc["ckpts"]:
            print(f"Loading SSL Encoder from path: {path}")
            encoder = load_ssl_models(
                ssl_model_cfg,
                load_weights=True,
                ckpt_path=ssl_model_cfg.ckpt.final_ckpt,
            ).encoder.to(device)
            head: SegmenterHead = SegmenterHead(
                class_cnt,
                encoder,
                dataset_name=task_cfg.dataset,
                epochs=task_cfg.epochs,
                lr=task_cfg.lr,
            ).to(device)
            if head.exists_segmenter():
                head.load_segmenter()
            else:
                raise RuntimeError("Please train the segmenter before the attack")
            segmenter = Segmenter(class_cnt, encoder, head).to(device)
            encoder.unfreeze()
            encoder.eval()
            head.eval()
            # encoder attack
            loss: Loss = loss_dict[action_cfg.attack_loss]()
            encoder_attack = PGD_Sign(
                loss=loss,
                num_steps=action_cfg.attack_num_steps,
                eps_budget=action_cfg.attack_epsilon,
                lr=action_cfg.attack_lr,
                mean=mean,
                std=std,
            )
            encoder_adv_acc, encoder_adv_miu = adversarial_segmenter_accuracy_miou(
                encoder_attack,
                encoder,
                eval_loader,
                device,
                segmenter,
                n_labels=class_cnt,
                model_cfg=ssl_model_cfg,
                task_cfg=task_cfg,
            )
            adv_acc["encoder_adv_acc"].append(encoder_adv_acc)
            adv_acc["encoder_adv_miou"].append(encoder_adv_miu)
            # downstream attack
            downstream_attack = SegPGD(
                loss=loss,  # hardcoded cross entropy for now
                num_steps=action_cfg.attack_num_steps,
                eps_budget=action_cfg.attack_epsilon,
                lr=action_cfg.attack_lr,
                mean=mean,
                std=std,
            )
            (
                downstream_adv_acc,
                downstream_adv_miu,
            ) = adversarial_segmenter_accuracy_miou(
                downstream_attack,
                encoder,
                eval_loader,
                device,
                segmenter,
                n_labels=class_cnt,
                downstream=True,
                model_cfg=ssl_model_cfg,
                task_cfg=task_cfg,
            )
            adv_acc["downstream_adv_acc"].append(downstream_adv_acc)
            adv_acc["downstream_adv_miou"].append(downstream_adv_miu)
            print(
                f"encoder_adv_acc: {encoder_adv_acc}, downstream_adv_acc: {downstream_adv_acc}",
                "\n",
                f"encoder_adv_miou: {encoder_adv_miu}, downstream_adv_miu: {downstream_adv_miu}",
            )
    elif task_cfg.task == "eval_segmentation_adv":
        adv_acc = {"nope": []}
        for path in get_ckpts(
            ssl_model_cfg.ckpt.ckpts_folder, ssl_model_cfg.ckpt.final_ckpt
        ):
            print(f"Loading SSL Encoder from path: {path}")
            encoder = load_ssl_models(
                ssl_model_cfg,
                load_weights=True,
                ckpt_path=ssl_model_cfg.ckpt.final_ckpt,
            ).encoder.to(device)
            head: SegmenterHead = SegmenterHead(
                class_cnt,
                encoder,
                dataset_name=task_cfg.dataset,
                epochs=task_cfg.epochs,
                lr=task_cfg.lr,
            ).to(device)
            if head.exists_segmenter():
                head.load_segmenter()
            else:
                raise RuntimeError("Please train the segmenter before the attack")
            segmenter = Segmenter(class_cnt, encoder, head).to(device)
            encoder.unfreeze()
            encoder.eval()
            head.eval()
            # encoder attack
            loss: Loss = loss_dict[action_cfg.attack_loss]()
            if task_cfg.eval_embed:
                encoder_attack = PGD_Sign(
                    loss=loss,
                    num_steps=action_cfg.attack_num_steps,
                    eps_budget=action_cfg.attack_epsilon,
                    lr=action_cfg.attack_lr,
                    mean=mean,
                    std=std,
                )
                adversarial_segmenter_all_per_step(
                    encoder_attack,
                    encoder,
                    eval_loader,
                    device,
                    segmenter,
                    n_labels=class_cnt,
                    model_cfg=ssl_model_cfg,
                    task_cfg=task_cfg,
                    action_cfg=action_cfg,
                )
            # downstream attack
            if task_cfg.eval_downstream:
                downstream_attack = SegPGD(
                    loss=loss,  # hardcoded cross entropy for now
                    num_steps=action_cfg.attack_num_steps,
                    eps_budget=action_cfg.attack_epsilon,
                    lr=action_cfg.attack_lr,
                    mean=mean,
                    std=std,
                )
                adversarial_segmenter_all_per_step(
                    downstream_attack,
                    encoder,
                    eval_loader,
                    device,
                    segmenter,
                    n_labels=class_cnt,
                    downstream=True,
                    model_cfg=ssl_model_cfg,
                    task_cfg=task_cfg,
                    action_cfg=action_cfg,
                )

    elif task_cfg.task == "depth_estimation":
        adv_acc = {
            "encoder_adv_rmse": [],
            "downstream_adv_rmse": [],
            "ckpts": get_ckpts(
                ssl_model_cfg.ckpt.ckpts_folder, ssl_model_cfg.ckpt.final_ckpt
            ),
        }
        for path in adv_acc["ckpts"]:
            print(f"Loading SSL Encoder from path: {path}")
            encoder = load_ssl_models(
                ssl_model_cfg,
                load_weights=True,
                ckpt_path=path,
            ).encoder.to(device)
            decode_head = BNHead(encoder, task_cfg, ssl_model_cfg, ckpt_path=path, device=device)
            encoder_decoder = DepthEncoderDecoder(encoder, decode_head)
            if decode_head.exists():
                decode_head.load()
            else:
                raise RuntimeError("Please train the depther before the attack")
            encoder.unfreeze()
            encoder.eval()
            decode_head.conv_depth.eval()
            # encoder attack
            if action_cfg.eval_embed:
                loss: Loss = loss_dict[action_cfg.attack_loss]()
                encoder_attack = PGD_Sign(
                    loss=loss,
                    num_steps=action_cfg.attack_num_steps,
                    eps_budget=action_cfg.attack_epsilon,
                    lr=action_cfg.attack_lr,
                    mean=mean,
                    std=std,
                )
                en_rmse = adversarial_depth_estimation(
                    attack=encoder_attack,
                    data_loader=eval_loader,
                    device=device,
                    encoder_decoder=encoder_decoder,
                    task_cfg=task_cfg,
                    downstream=False,
                )
                adv_acc["encoder_adv_rmse"].append(en_rmse)
                print(f"encoder_adv_rmse: {en_rmse}")
            else:
                adv_acc["encoder_adv_rmse"].append(None)
            # downstream attack
            if action_cfg.eval_downstream:
                downstream_attack = DepthPGD(
                    loss=None,
                    num_steps=action_cfg.attack_num_steps,
                    eps_budget=action_cfg.attack_epsilon,
                    lr=action_cfg.attack_lr,
                    mean=mean,
                    std=std,
                )
                ds_rmse = adversarial_depth_estimation(
                    attack=downstream_attack,
                    data_loader=eval_loader,
                    device=device,
                    encoder_decoder=encoder_decoder,
                    task_cfg=task_cfg,
                    downstream=True,
                )
                adv_acc["downstream_adv_rmse"].append(ds_rmse)
                print(f"downstream_adv_rmse: {ds_rmse}")
            else:
                adv_acc["downstream_adv_rmse"].append(None)
    return adv_acc


if __name__ == "__main__":
    main()
