# -*- coding: utf-8 -*-
import torch
import numpy as np
from typing import Dict
from torch import nn
# from typing import Any, Iterator, List, Union
# from detectron2.structures.boxes import matched_boxlist_iou
import pycocotools.mask as mask_utils
from torch.nn import functional as F
from detectron2.layers import Conv2d, ShapeSpec
import fvcore.nn.weight_init as weight_init

class DpSemSegFPNHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        num_classes           = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims             = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride    = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        norm                  = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight      = cfg.MODEL.ROI_DENSEPOSE_HEAD.SEMSEG_WEIGHTS
        # fmt: on
        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                # norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    # bias=not norm,
                    # norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        # self.semmask_features = Conv2d(conv_dims, feature_channels[self.in_features[0]]//4,
        #                                kernel_size=1, stride=1, padding=0)
        self.predictor = Conv2d(conv_dims, num_classes + 1, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features, targets=None):
        global_features = []
        for i, f in enumerate(self.in_features):

            if i == 0:
                x = self.scale_heads[i](features[i])
                global_features.append(x)
            else:
                latent_feats = self.scale_heads[i](features[i])
                global_features.append(latent_feats)
                x = x + latent_feats
            # global_features[-1] = F.relu(self.semmask_features(global_features[-1]))
        # global_features = torch.cat(global_features, 1)
        x = self.predictor(x)
        if self.training:
            losses = {}
            losses["loss_dp_sem_seg"] = (
                F.cross_entropy(x, targets, reduction="mean")
                * self.loss_weight
            )
            return F.softmax(x, 1)[:,1].unsqueeze(1), losses, global_features
        else:
            return F.softmax(x, 1)[:,1].unsqueeze(1), {}, global_features


class SemanticMaskDataFilter(object):
    def __init__(self, cfg):
        self.common_stride    = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE

    @torch.no_grad()
    def __call__(self, proposals_with_targets, im_h, im_w):
        gt_masks_for_all_images = []
        for proposals_per_image in proposals_with_targets:
            assert hasattr(proposals_per_image, "gt_boxes")
            assert hasattr(proposals_per_image, "proposal_boxes")
            gt_boxes = proposals_per_image.gt_boxes
            gt_masks = proposals_per_image.gt_masks

            gt_masks_per_image = torch.zeros((im_h, im_w))
            for i in range(len(proposals_per_image)):
                if i > 5:
                    break
                # if i == 0:
                #     unique_box = gt_boxes[i].tensor
                # if i > 0:
                #     exisits = torch.sum((unique_box - gt_boxes[i].tensor) == 0, 1)
                #     if torch.sum(exisits==4) > 0:
                #         continue
                #     else:
                #         unique_box = torch.cat([unique_box, gt_boxes[i].tensor], 0)
                mask = gt_masks[i]
                # box  = gt_boxes[i]
                # box_w = int((box.tensor[0, 2] - box.tensor[0, 0]).long().cpu().numpy())
                # box_h = int((box.tensor[0, 3] - box.tensor[0, 1]).long().cpu().numpy())

                gt_mask = polygons_to_bitmask(mask.polygons[0], im_h, im_w)
                gt_masks_per_image[gt_mask] = 1
            gt_masks_per_image = F.interpolate(gt_masks_per_image.unsqueeze(0).unsqueeze(0),
                                               (int(im_h / self.common_stride), int(im_w / self.common_stride)),
                                               mode="bilinear", align_corners=False)
            gt_masks_for_all_images.append(gt_masks_per_image)
        gt_masks_for_all_images = torch.cat(gt_masks_for_all_images, 0)
        gt_sem_seg = gt_masks_for_all_images.squeeze(1).to("cuda").long()
        return gt_sem_seg

class SemanticMaskDataFilterV2(object):
    def __init__(self, cfg):
        self.common_stride    = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE

    @torch.no_grad()
    def __call__(self, semseg_with_targets, im_h, im_w):
        
        gt_masks_for_all_images = F.interpolate(semseg_with_targets.tensor.unsqueeze(1),
                      (int(im_h / self.common_stride), int(im_w / self.common_stride)), mode="bilinear",
                      align_corners=False)
        gt_masks_for_all_images[gt_masks_for_all_images>0.5] = 1
        gt_sem_seg = gt_masks_for_all_images.squeeze(1).to("cuda").long()
        return gt_sem_seg

def polygons_to_bitmask(polygons, height, width):
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    assert len(polygons) > 0, "COCOAPI does not support empty polygons"
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(np.bool)

def build_semantic_mask_data_filter(cfg):
    mask_filter = SemanticMaskDataFilterV2(cfg)
    return mask_filter

