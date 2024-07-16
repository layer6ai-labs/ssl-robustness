# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
import math

import torch
import warnings

import torch.nn.functional as F

from evaluators.depth_losses.sigloss import SigLoss
from evaluators.depth_losses.gradientloss import GradientLoss
from functools import partial


class DepthBaseDecodeHead:
    def __init__(self, encoder, task_cfg, ssl_cfg, ckpt_path, device):
        super(DepthBaseDecodeHead, self).__init__()
        self.encoder = encoder
        self.in_channels = [ssl_cfg.feature_dim]
        self.channels = ssl_cfg.head_channels
        self.conv_cfg = None
        self.act_cfg = dict(type="ReLU")
        self.loss_decode = [
            SigLoss(
                valid_mask=True,
                loss_weight=task_cfg.sig_loss_weight,
                warm_up=True,
                loss_name="loss_depth",
            ),
            GradientLoss(
                valid_mask=True,
                loss_weight=task_cfg.grad_loss_weight,
                loss_name="loss_grad",
            ),
        ]
        self.align_corners = False
        self.min_depth = 0.001
        self.max_depth = 10
        self.norm_cfg = None
        self.classify = True
        self.n_bins = 256
        self.scale_up = False
        self.dataset_name = task_cfg.dataset
        self.encoder_name = encoder.name
        self.lr = task_cfg.lr
        self.epochs = task_cfg.epochs

        self.official = task_cfg.load_official
        self.head = task_cfg.head

        self.bins_strategy = "UD"
        self.norm_strategy = "linear"
        self.softmax = torch.nn.Softmax(dim=1)
        torch.device(device)
        self.conv_depth = torch.nn.Conv2d(
            self.channels, self.n_bins, kernel_size=3, padding=1, stride=1
        )

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.encoder.forward = partial(
            self.encoder.model.get_intermediate_layers,
            n=[11],
            reshape=True,
            return_class_token=True,
            norm=False,
        )
        self.ckpt_path = "_".join(ckpt_path.split("/"))

    def extra_repr(self):
        """Extra repr."""
        s = f"align_corners={self.align_corners}"
        return s

    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, depth_gt):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            depth_gt (Tensor): GT depth
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        depth_pred = self.forward(inputs)
        losses = self.losses(depth_pred, depth_gt)

        return losses

    def forward_test(self, inputs):
        """Forward function for testing.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output depth map.
        """
        return self.forward(inputs)

    def depth_pred(self, feat):
        """Prediction each pixel."""
        logit = self.conv_depth(feat)

        bins = torch.linspace(
            self.min_depth, self.max_depth, self.n_bins, device=feat.device
        )
        # following Adabins, default linear
        logit = torch.relu(logit)
        eps = 0.1
        logit = logit + eps
        logit = logit / logit.sum(dim=1, keepdim=True)

        output = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(dim=1)
        return output

    def losses(self, depth_pred, depth_gt):
        """Compute depth loss."""
        loss = dict()
        depth_pred = resize(
            input=depth_pred,
            size=depth_gt.shape[1:],
            mode="bilinear",
            align_corners=self.align_corners,
            warning=False,
        ).squeeze(1)

        losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(depth_pred, depth_gt)
            else:
                loss[loss_decode.loss_name] += loss_decode(depth_pred, depth_gt)
        return loss

    def load(self):
        path = self.get_path()
        if os.path.exists(path):
            if self.official:
                loaded_model = torch.load(
                    path,
                    map_location=torch.device("cpu"),
                )["state_dict"]
                loaded_model["weight"] = loaded_model["decode_head.conv_depth.weight"]
                loaded_model["bias"] = loaded_model["decode_head.conv_depth.bias"]
                del loaded_model["decode_head.conv_depth.weight"]
                del loaded_model["decode_head.conv_depth.bias"]
                self.conv_depth.load_state_dict(loaded_model, strict=True)
            else:
                self.conv_depth.load_state_dict(
                    torch.load(
                        path,
                        map_location=torch.device("cpu"),
                    )
                )
        else:
            raise ValueError(f"Depth estimator not found at {path}")

    def get_path(self):
        if self.official:
            path = f"tmp/depth_estimators/{self.head}"
        else:
            path = f"tmp/depth_estimators/{self.encoder_name}_{self.dataset_name}_{self.lr}_{self.epochs}_{self.ckpt_path}.pth"
        return path

    def exists(self):
        print(self.get_path())
        return os.path.exists(self.get_path())

    def save(self):
        os.makedirs("tmp/depth_estimators", exist_ok=True)
        path = self.get_path()
        torch.save(self.conv_depth.state_dict(), path)


class BNHead(DepthBaseDecodeHead):
    """Just a batchnorm."""

    def __init__(self, encoder, task_cfg, ssl_cfg, ckpt_path, device):
        super().__init__(encoder, task_cfg, ssl_cfg, ckpt_path, device=device)
        self.input_transform = "resize_concat"
        self.in_index = [0]
        self.upsample = 4
        self.conv_depth = torch.nn.Conv2d(
            self.channels, self.n_bins, kernel_size=1, padding=0, stride=1
        )
        self.conv_depth.cuda()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if "concat" in self.input_transform:
            inputs = [inputs[i] for i in self.in_index]
            if "resize" in self.input_transform:
                inputs = [
                    resize(
                        input=x,
                        size=[s * self.upsample for s in inputs[0].shape[2:]],
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for x in inputs
                ]
            inputs = torch.cat(inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _forward_feature(self, inputs, **kwargs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # accept lists (for cls token)
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if len(x) == 2:
                x, cls_token = x[0], x[1]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                x = x[0]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                inputs[i] = x
        x = self._transform_inputs(inputs)
        return x

    def forward(self, inputs, **kwargs):
        """Forward function."""
        output = self._forward_feature(inputs, **kwargs)
        output = self.depth_pred(output)

        return output


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=False,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)
