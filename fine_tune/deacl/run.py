# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from pprint import pprint
from omegaconf import OmegaConf
from argparse import Namespace

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from fine_tune.deacl.deacl import DeACL
from fine_tune.deacl.checkpointer import Checkpointer
from models.ssl_models.ssl import Encoder
from models.ssl_models.utils import load_ssl_models
from models.downstream import LinearClassifier
from evaluators.attack import Attack

import wandb

import shutil
import types

from torchvision import transforms
from torch.utils.data import DataLoader
import torch


def run(
    train_loader: DataLoader,
    attack: Attack,
    task_cfg: dict,
    model_cfg: dict,
    action_cfg: dict,
):
    seed_everything(5)
    run_args = Namespace(**OmegaConf.to_container(task_cfg))
    run_args.devices = eval(run_args.devices)

    callbacks = []

    model = DeACL(
        model_cfg=model_cfg,
        attack=attack,
        max_epochs=task_cfg.max_epochs,
        batch_size=task_cfg.batch_size,
        optimizer=task_cfg.optimizer,
        lr=task_cfg.lr,
        weight_decay=task_cfg.weight_decay,
        scheduler=task_cfg.scheduler,
        trades_k=task_cfg.trades_k,
    )

    # wandb logging
    if task_cfg.wandb:
        wandb.login()
        wandb_logger = WandbLogger(
            name=task_cfg.wandb_name,
            project=task_cfg.wandb_project,
            entity=None,
            offline=True,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(task_cfg)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    if task_cfg.save_checkpoint:
        ckpt = Checkpointer(
            args=run_args,
            model_name=model_cfg.name,
            logdir=os.path.join(action_cfg.output_models_root, task_cfg.task),
            frequency=task_cfg.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path = task_cfg.resume_checkpoint

    trainer = Trainer.from_argparse_args(
        run_args,
        logger=wandb_logger if task_cfg.wandb else None,
        callbacks=callbacks,
        enable_checkpointing=False,
        strategy="ddp_find_unused_parameters_false",
    )

    # File backup
    if task_cfg.wandb:
        experimentdir = f"{action_cfg.output_path}/_{trainer.logger.version}"
    else:
        experimentdir = action_cfg.output_path

    if os.path.exists(experimentdir):
        print(experimentdir + " : exists. overwrite it.")
        shutil.rmtree(experimentdir)
        os.mkdir(experimentdir)
    else:
        os.mkdir(experimentdir)

    trainer.fit(model, train_loader, ckpt_path=ckpt_path)
