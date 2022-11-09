# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# File:

import torch
from torch import nn

from detectron2.layers import ShapeSpec

from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler, MultiROIPooler

from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals, select_proposals_with_visible_keypoints
from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_inference,keypoint_rcnn_loss
from .densepose_head import (
    build_densepose_data_filter,
    build_densepose_head,
    build_densepose_losses,
    build_densepose_predictor,
    densepose_inference,
    DensePoseInterLosses,
    DensePoseDataFilter,
    dp_keypoint_rcnn_loss,
    DensePoseKeypointsPredictor,
)
from .semantic_mask_head import (
    build_semantic_mask_data_filter,
    DpSemSegFPNHead
)




@ROI_HEADS_REGISTRY.register()
class DensePoseROIHeads(StandardROIHeads):
    """
    A Standard ROIHeads which contains an addition of DensePose head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_densepose_head(cfg, input_shape)
        self._init_dpsemseg_head(cfg)
        self._init_dp_keypoint_head(cfg,input_shape)

    def _init_dpsemseg_head(self, cfg):
        self.dp_semseg_on = cfg.MODEL.ROI_DENSEPOSE_HEAD.SEMSEG_ON

        if not self.dp_semseg_on:
            return
        self.common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        self.dp_semseg_head = DpSemSegFPNHead(cfg)
        self.sem_mask_data_filter = build_semantic_mask_data_filter(cfg)
      

    def _init_dp_keypoint_head(self, cfg,input_shape):
        # fmt: off

        self.dp_keypoint_on                         = cfg.MODEL.ROI_DENSEPOSE_HEAD.KPT_ON
        if not self.dp_keypoint_on:
            return
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight                = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        dp_pooler_resolution = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        # dp_multi_pooler_res        = ((28,28),(14,14),(14,14),(7,7))
        dp_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        dp_pooler_sampling_ratio = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
        dp_pooler_type = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        in_channels = [input_shape[f].channels for f in self.in_features][0]
        if not self.densepose_on:
            self.use_mid = cfg.MODEL.ROI_DENSEPOSE_HEAD.MID_ON

            if cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME == 'DensePoseAMAHead':
                self.densepose_pooler = MultiROIPooler(
                    output_size=[[28, 28], [14, 14], [14, 14], [7, 7]],
                    # output_size=[[28, 28], [28, 28], [28, 28], [28, 28]],
                    scales=dp_pooler_scales,
                    sampling_ratio=dp_pooler_sampling_ratio,
                    pooler_type=dp_pooler_type,
                )
            else:
                self.densepose_pooler = ROIPooler(
                    output_size=dp_pooler_resolution,
                    scales=dp_pooler_scales,
                    sampling_ratio=dp_pooler_sampling_ratio,
                    pooler_type=dp_pooler_type,
                )
            self.densepose_head = build_densepose_head(cfg, in_channels)
            # print(self.densepose_head)
            self.keypoint_predictor = DensePoseKeypointsPredictor(cfg, self.densepose_head.n_out_channels)

    def _init_densepose_head(self, cfg, input_shape):
        # fmt: off
        self.cfg                   = cfg
        self.densepose_on          = cfg.MODEL.DENSEPOSE_ON and cfg.MODEL.ROI_DENSEPOSE_HEAD.RCNN_HEAD_ON
        if not self.densepose_on:
            return
        self.densepose_data_filter = build_densepose_data_filter(cfg)
        dp_pooler_resolution       = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        dp_pooler_scales           = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        dp_pooler_sampling_ratio   = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
        dp_pooler_type             = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        self.use_mid = cfg.MODEL.ROI_DENSEPOSE_HEAD.MID_ON

        self.inter_super_on = False
        self.inter_weight          = cfg.MODEL.ROI_DENSEPOSE_HEAD.INTER_WEIGHTS
        if cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME == 'DensePosePIDHead':
            self.inter_super_on = True

        # fmt: on
        in_channels = [input_shape[f].channels for f in self.in_features][0]
        if cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME == 'DensePoseAMAHead':
            self.densepose_pooler = MultiROIPooler(
                output_size=[[32, 32], [16, 16], [16, 16], [8, 8]],
                scales=dp_pooler_scales,
                sampling_ratio=dp_pooler_sampling_ratio,
                pooler_type=dp_pooler_type,
            )
        else:
            self.densepose_pooler = ROIPooler(
                output_size=dp_pooler_resolution,
                scales=dp_pooler_scales,
                sampling_ratio=dp_pooler_sampling_ratio,
                pooler_type=dp_pooler_type,
            )

        self.densepose_head = build_densepose_head(cfg, in_channels)
        self.densepose_predictor = build_densepose_predictor(
            cfg, self.densepose_head.n_out_channels
        )
        self.densepose_losses = build_densepose_losses(cfg)
        self.densepose_inter_losses = DensePoseInterLosses(cfg)

    def _forward_semsegs(self, features, instances, extra):

        if not self.dp_semseg_on:
            return
        if self.training:
            im_h, im_w = int(features[0].size(2)* self.common_stride), \
                         int(features[0].size(3)* self.common_stride)
            
            gt_sem_seg = self.sem_mask_data_filter(extra, im_h, im_w)
            sem_seg_results, sem_seg_losses, latent_features = self.dp_semseg_head(features, gt_sem_seg)
            
        else:
            gt_sem_seg = None
            sem_seg_results, sem_seg_losses, latent_features = self.dp_semseg_head(features, gt_sem_seg)

        return sem_seg_results, sem_seg_losses, latent_features

    def _forward_dp_keypoint(self, keypoint_logits, instances):

        if not self.dp_keypoint_on:
            return {} if self.training else instances
        num_images = len(instances)
        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            normalizer = (
                num_images
                * self.batch_size_per_image
                * self.positive_sample_fraction
                * keypoint_logits.shape[1]
            )
            loss = dp_keypoint_rcnn_loss(
                keypoint_logits,
                instances,
                normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
            )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances

    def forward_dp_keypoint(self, features, instances):

        if not self.dp_keypoint_on:
            return {} if self.training else instances
        num_images = len(instances)
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            # proposals = self.keypoint_data_filter(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            
            if self.use_mid:
                features = [self.mid_decoder(features)]
            keypoint_features = self.densepose_pooler(features, proposal_boxes)
            keypoint_output = self.densepose_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_output)
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            normalizer = (
                num_images
                * self.batch_size_per_image
                * self.positive_sample_fraction
                * keypoint_logits.shape[1]
            )
           
            loss = keypoint_rcnn_loss(
                keypoint_logits,
                proposals,
                normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
            )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            if self.use_mid:
                features = [self.mid_decoder(features)]
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.densepose_pooler(features, pred_boxes)
            keypoint_output = self.densepose_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_output)
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances

    def _forward_densepose(self, features, instances, extra=None):
        """
        Forward logic of the densepose prediction branch.

        Args:
            features (list[Tensor]): #level input features for densepose prediction
            instances (list[Instances]): the per-image instances to train/predict densepose.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.densepose_on:
            return {} if self.training else instances
        bbox_locs_params = self.box_predictor.bbox_pred.weight
        bbox_cls_params = self.box_predictor.cls_score.weight
        with torch.no_grad():
            bbox_params = torch.cat([bbox_cls_params, bbox_locs_params], dim=0)
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals_dp = self.densepose_data_filter(proposals)
            if len(proposals_dp) > 0:
                proposal_boxes = [x.proposal_boxes for x in proposals_dp]
                if self.use_mid:
                    features = [self.mid_decoder(features)]
                    if self.dp_semseg_on:
                        segm_res, seg_losses, latent_features = self._forward_semsegs(features, instances, extra)
                else:
                    if self.dp_semseg_on:
                        segm_res, seg_losses, latent_features = self._forward_semsegs([features[-1]], instances, extra)
                features_dp = self.densepose_pooler(features, proposal_boxes)
                # print('roi pooler:',features_dp.size())
                if self.inter_super_on:
                    densepose_head_outputs, densepose_inter_head_outputs = self.densepose_head(features_dp)
                    densepose_outputs,  densepose_inter_outputs= self.densepose_predictor(densepose_head_outputs, densepose_inter_head_outputs)
                    densepose_loss_dict = self.densepose_losses(proposals_dp, densepose_outputs, cls_emb_loss_on=True)
                    inter_loss_dict = self.densepose_inter_losses(proposals_dp, densepose_inter_outputs, 'inter_')
                    for _, k in enumerate(inter_loss_dict.keys()):
                        if k == 'inter_loss_densepose_M':
                            densepose_loss_dict[k] = inter_loss_dict[k] * 1.
                        else:
                            densepose_loss_dict[k] = inter_loss_dict[k] * 0.
                else:
                    densepose_head_outputs = self.densepose_head(features_dp)
                    densepose_outputs, dp_outputs_from_kpt, dp_outputs_from_bbox = self.densepose_predictor(densepose_head_outputs, bbox_params)
                    if self.dp_keypoint_on:
                        keypoints_output = densepose_outputs[-1]
                        densepose_outputs = densepose_outputs[:-1]
                    densepose_loss_dict = self.densepose_losses(proposals_dp, densepose_outputs)
                    if self.dp_keypoint_on:
                        kpt_loss_dict = self._forward_dp_keypoint(keypoints_output, proposals_dp)
                        for _, k in enumerate(kpt_loss_dict.keys()):
                            densepose_loss_dict[k] = kpt_loss_dict[k]
                    if self.dp_semseg_on:
                        for _, k in enumerate(seg_losses.keys()):
                            densepose_loss_dict[k] = seg_losses[k]
                    if dp_outputs_from_kpt is not None:
                        dp_kpt_loss_dict = self.densepose_losses(proposals_dp, dp_outputs_from_kpt)
                        densepose_loss_dict['loss_densepose_I_from_kpt'] = dp_kpt_loss_dict['loss_densepose_I']
                    if dp_outputs_from_bbox is not None:
                        dp_box_loss_dict = self.densepose_losses(proposals_dp, dp_outputs_from_bbox)
                        densepose_loss_dict['loss_densepose_I_from_box'] = dp_box_loss_dict['loss_densepose_I']
                return densepose_loss_dict
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            if self.use_mid:
                features = [self.mid_decoder(features)]
            features_dp = self.densepose_pooler(features, pred_boxes)
            if len(features_dp) > 0:
                if self.inter_super_on:
                    densepose_head_outputs, densepose_inter_head_outputs = self.densepose_head(features_dp)
                    densepose_outputs, densepose_inter_outputs = self.densepose_predictor(densepose_head_outputs,
                                                                                          densepose_inter_head_outputs)
                else:
                    densepose_head_outputs = self.densepose_head(features_dp)
                    densepose_outputs, _, _ = self.densepose_predictor(densepose_head_outputs,bbox_params)
                    if self.dp_keypoint_on:
                        keypoints_output = densepose_outputs[-1]
                        densepose_outputs = densepose_outputs[:-1]
                        instances = self._forward_dp_keypoint(keypoints_output, instances)
            else:
                # If no detection occured instances
                # set densepose_outputs to empty tensors
                empty_tensor = torch.zeros(size=(0, 0, 0, 0), device=features_dp.device)
                densepose_outputs = tuple([empty_tensor] * 5)

            densepose_inference(densepose_outputs, instances)
            return instances


    def _forward_ama_densepose(self, features, instances, extra=None):
        """
        Forward logic of the densepose prediction branch.

        Args:
            features (list[Tensor]): #level input features for densepose prediction
            instances (list[Instances]): the per-image instances to train/predict densepose.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.densepose_on:
            return {} if self.training else instances
        latent_features = None
        if self.dp_semseg_on:
            segm_res, seg_losses, latent_features = self._forward_semsegs(features, instances, extra)
            # features = latent_features

        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals_dp = self.densepose_data_filter(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals_dp]
            features_dp = self.densepose_pooler(features, proposal_boxes)

            if len(proposals_dp) > 0:
                densepose_head_outputs, densepose_inter_head_outputs = self.densepose_head(features_dp, latent_features)
                bbox_locs_params = self.box_predictor.bbox_pred.weight
                bbox_cls_params  = self.box_predictor.cls_score.weight
                bbox_params = torch.cat([bbox_cls_params, bbox_locs_params], dim=0)
                densepose_outputs, inter_output_1, inter_output_2 = self.densepose_predictor(densepose_head_outputs,
                                                                                      densepose_inter_head_outputs, bbox_params)
                if self.dp_keypoint_on:
                    keypoints_output = densepose_outputs[-1]
                    densepose_outputs = densepose_outputs[:-1]
                densepose_loss_dict = self.densepose_losses(proposals_dp, densepose_outputs, cls_emb_loss_on=self.cfg.MODEL.ROI_DENSEPOSE_HEAD.IA_LOSS)

                inter_loss_dict = self.densepose_losses(proposals_dp, inter_output_1, prefix='inter_1_')

                for _, k in enumerate(inter_loss_dict.keys()):
                    if 'loss_densepose_I' in k:
                        densepose_loss_dict[k] = inter_loss_dict[k]
                    else:
                        densepose_loss_dict[k] = inter_loss_dict[k] * self.inter_weight
                inter_loss_dict = self.densepose_losses(proposals_dp, inter_output_2, prefix='inter_2_')
                for _, k in enumerate(inter_loss_dict.keys()):
                    if 'loss_densepose_I' in k:
                        densepose_loss_dict[k] = inter_loss_dict[k]
                    else:
                        densepose_loss_dict[k] = inter_loss_dict[k] * self.inter_weight
                if self.dp_semseg_on:
                    for _, k in enumerate(seg_losses.keys()):
                        densepose_loss_dict[k] = seg_losses[k]
                if self.dp_keypoint_on:

                    kpt_loss_dict = self._forward_dp_keypoint(keypoints_output, proposals_dp)
                    for _, k in enumerate(kpt_loss_dict.keys()):
                        densepose_loss_dict[k] = kpt_loss_dict[k]
                # box_features = self.box_pooler(features, proposal_boxes)
                # box_features = self.box_head(box_features)
                return densepose_loss_dict
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            features_dp = self.densepose_pooler(features, pred_boxes)
            if len(features_dp[0]) > 0 and len(pred_boxes) > 0:

                densepose_head_outputs, densepose_inter_head_outputs = self.densepose_head(features_dp, latent_features)
                bbox_locs_params = self.box_predictor.bbox_pred.weight
                bbox_cls_params = self.box_predictor.cls_score.weight
                bbox_params = torch.cat([bbox_cls_params, bbox_locs_params], dim=0)
                densepose_outputs, _, _ = self.densepose_predictor(densepose_head_outputs, densepose_inter_head_outputs,
                                                                   bbox_params)
                if self.dp_keypoint_on:
                    keypoints_output = densepose_outputs[-1]
                    densepose_outputs = densepose_outputs[:-1]
                    instances=self._forward_dp_keypoint(keypoints_output, instances)
            else:
                # If no detection occured instances
                # set densepose_outputs to empty tensors
                empty_tensor = torch.zeros(size=(0, 0, 0, 0), device=features_dp[0].device)
                densepose_outputs = tuple([empty_tensor] * 5)

            densepose_inference(densepose_outputs, instances)
            return instances

    def forward(self, images, features, proposals, targets=None, extra=None):
        features_list = [features[f] for f in self.in_features]

        instances, losses = super().forward(images, features, proposals, targets)

        del targets, images

        if self.training:
            if self.cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME == 'DensePoseAMAHead':
                losses.update(self._forward_ama_densepose(features_list, instances, extra))

            else:
                losses.update(self._forward_densepose(features_list, instances, extra))
                if self.dp_keypoint_on and not self.densepose_on:
                    losses.update(self.forward_dp_keypoint(features_list, instances))
        else:
            if self.cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME == 'DensePoseAMAHead':
                instances = self._forward_ama_densepose(features_list, instances)
            else:
                instances = self._forward_densepose(features_list, instances)
                if self.dp_keypoint_on and not self.densepose_on:
                    instances = self.forward_dp_keypoint(features_list, instances)
        return instances, losses


