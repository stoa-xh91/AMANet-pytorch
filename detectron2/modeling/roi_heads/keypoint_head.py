# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, interpolate
from detectron2.structures import heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
# import cv2
import numpy as np
import pickle
_TOTAL_SKIPPED = 0

ROI_KEYPOINT_HEAD_REGISTRY = Registry("ROI_KEYPOINT_HEAD")
ROI_KEYPOINT_HEAD_REGISTRY.__doc__ = """
Registry for keypoint heads, which make keypoint predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def build_keypoint_head(cfg, input_shape):
    """
    Build a keypoint head from `cfg.MODEL.ROI_KEYPOINT_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_KEYPOINT_HEAD.NAME
    return ROI_KEYPOINT_HEAD_REGISTRY.get(name)(cfg, input_shape)


def keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer):
    """
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.

    Returns a scalar tensor containing the loss.
    """
    heatmaps = []
    valid = []

    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_keypoints
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0

    N, K, H, W = pred_keypoint_logits.shape
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(
        pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer

    return keypoint_loss

def keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer):
    """
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.

    Returns a scalar tensor containing the loss.
    """
    heatmaps = []
    valid = []

    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_keypoints
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0

    N, K, H, W = pred_keypoint_logits.shape
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(
        pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer

    return keypoint_loss
def keypoint_rcnn_inter_part_loss(pred_keypoint_logits, pred_inter_keypoint_logits, instances, normalizer):
    """
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.

    Returns a scalar tensor containing the loss.
    """
    heatmaps = []
    inter_heatmaps = []
    valid = []

    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_keypoints
        inter_heatmaps_per_image, _ = keypoints.to_inter_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        tmp_inter_maps = inter_heatmaps_per_image[:,1:,:,:].reshape((inter_heatmaps_per_image.size(0),8,2,keypoint_side_len,keypoint_side_len))
        tmp_inter_maps = torch.sum(tmp_inter_maps, 2)
        inter_heatmaps_per_image = torch.cat([inter_heatmaps_per_image[:,0].unsqueeze(1), tmp_inter_maps], 1)
        inter_heatmaps_per_image = inter_heatmaps_per_image.clamp(min=0, max=1)
        inter_heatmaps.append(inter_heatmaps_per_image)

        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    inter_keypoint_targets = cat(inter_heatmaps, dim=0).to(dtype=torch.float32).to("cuda")

    inter_part_loss = F.binary_cross_entropy(pred_inter_keypoint_logits, inter_keypoint_targets)

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0

    N, K, H, W = pred_keypoint_logits.shape
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(
        pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer

    return keypoint_loss, inter_part_loss

def keypoint_rcnn_hybrid_loss(pred_keypoint_logits, instances, normalizer):
    heatmaps = []
    inter_heatmaps = []
    valid = []

    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_keypoints
        gaussian_heatmaps_per_image, _= keypoints.to_inter_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )

        inter_heatmaps.append(gaussian_heatmaps_per_image)

        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    inter_keypoint_targets = cat(inter_heatmaps, dim=0).to("cuda")
    # kpt_logits_size = pred_keypoint_logits.size()
    # inter_pred_keypoint_logits = pred_keypoint_logits.reshape((kpt_logits_size[0],kpt_logits_size[1],kpt_logits_size[2]*kpt_logits_size[3]))
    # inter_pred_keypoint_logits = F.softmax(inter_pred_keypoint_logits, dim=2)
    # inter_pred_keypoint_logits = inter_pred_keypoint_logits.reshape(kpt_logits_size)
    inter_part_loss = F.mse_loss(pred_keypoint_logits, inter_keypoint_targets)

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0

    N, K, H, W = pred_keypoint_logits.shape
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(
        pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer

    return keypoint_loss, inter_part_loss

def keypoint_rcnn_inference(pred_keypoint_logits, pred_instances):
    """
    Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score, prob)
        and add it to the `pred_instances` as a `pred_keypoints` field.

    Args:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
           of instances in the batch, K is the number of keypoints, and S is the side length of
           the keypoint heatmap. The values are spatial logits.
        pred_instances (list[Instances]): A list of M Instances, where M is the batch size.

    Returns:
        None. boxes will contain an extra "pred_keypoints" field.
            The field is a tensor of shape (#instance, K, 3) where the last
            dimension corresponds to (x, y, probability).
    """
    # flatten all bboxes from all images together (list[Boxes] -> Nx4 tensor)
    bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)

    keypoint_results = heatmaps_to_keypoints(pred_keypoint_logits.detach(), bboxes_flat.detach())
    num_instances_per_image = [len(i) for i in pred_instances]
    keypoint_results = keypoint_results.split(num_instances_per_image, dim=0)

    for keypoint_results_per_image, instances_per_image in zip(keypoint_results, pred_instances):
        # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score, prob)
        keypoint_xyp = keypoint_results_per_image[:, :, [0, 1, 3]]
        instances_per_image.pred_keypoints = keypoint_xyp


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class KRCNNConvDeconvUpsampleHead(nn.Module):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
            num_keypoints: number of keypoint heatmaps to predicts, determines the number of
                           channels in the final output.
        """
        super(KRCNNConvDeconvUpsampleHead, self).__init__()

        # fmt: off
        # default up_scale to 2 (this can eventually be moved to config)
        up_scale      = 2
        conv_dims     = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        in_channels   = input_shape.channels
        # fmt: on

        self.blocks = []
        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module("conv_fcn{}".format(idx), module)
            self.blocks.append(module)
            in_channels = layer_channels

        deconv_kernel = 4
        self.score_lowres = ConvTranspose2d(
            in_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.up_scale = up_scale

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        for layer in self.blocks:
            x = F.relu(layer(x))
        x = self.score_lowres(x)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x

@ROI_KEYPOINT_HEAD_REGISTRY.register()
class KeypointPRHead(nn.Module):
    """
    A standard keypoint rcnn head containing a part relation modeling
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
            num_keypoints: number of keypoint heatmaps to predicts, determines the number of
                           channels in the final output.
        """
        super(KeypointPRHead, self).__init__()

        # fmt: off
        # default up_scale to 2 (this can eventually be moved to config)
        up_scale      = 2
        layer_channels     = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_HEAD_DIM
        relation_dims = cfg.MODEL.ROI_KEYPOINT_HEAD.RELATION_DIM
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        self.n_stacked_convs = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_STACKED_CONVS
        num_parts = num_keypoints // 2 + 1
        in_channels   = input_shape.channels
        deconv_kernel = 4
        # fmt: on

        self.blocks = []
        for i in range(self.n_stacked_convs):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module(self._get_layer_name(i), module)
            self.blocks.append(module)
            in_channels = layer_channels
        for i in range(2):
            layer = ConvTranspose2d(
                layer_channels,
                layer_channels,
                kernel_size=deconv_kernel,
                stride=2,
                padding=int(deconv_kernel / 2 - 1),
            )
            layer_name = self._get_deconv_layer_name(i, 'PM')
            self.add_module(layer_name, layer)
        for i in range(2):
            layer = ConvTranspose2d(
                layer_channels,
                layer_channels,
                kernel_size=deconv_kernel,
                stride=2,
                padding=int(deconv_kernel / 2 - 1),
            )
            layer_name = self._get_deconv_layer_name(i, 'KM')
            self.add_module(layer_name, layer)
        self.inter_part_score = Conv2d(layer_channels, num_parts, 3, stride=1, padding=1)
        self.kpt_score = Conv2d(layer_channels + layer_channels, num_keypoints, 3, stride=1, padding=1)
        # self.kpt_score = Conv2d(relation_dims + layer_channels, num_keypoints, 3, stride=1, padding=1)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
    def _get_layer_name(self, i):
        layer_name = "conv_fcn{}".format(i + 1)
        return layer_name

    def _get_deconv_layer_name(self,i, prefix=''):
        layer_name = prefix + "deconv_fcn{}".format(i + 1)
        return layer_name
    def _forward_relation_embedding(self, part_scores, rel_matrix=None):
        '''

        :param part_scores: B x NUM_PARTS x W x H
        :param rel_matrix: NUM_PARTS x RELATION_DIM
        :return:
        '''
        B, n_parts, h, w = part_scores.size(0), part_scores.size(1),part_scores.size(2), part_scores.size(3)
        # rel_matrix = rel_matrix.unsqueeze(0).repeat(B,1,1)
        part_scores = part_scores.reshape((B, n_parts, h*w)).permute(0,2,1)
        rel_embs = torch.matmul(part_scores, rel_matrix)
        rel_embs = rel_embs.permute(0, 2, 1).reshape((B, n_parts, h, w))
        return rel_embs

    def forward(self, x, rel_matrix=None):
        # part module
        for i in range(self.n_stacked_convs // 2):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
        inter_x = x
        for i in range(2):
            layer_name = self._get_deconv_layer_name(i, 'PM')
            inter_x = getattr(self, layer_name)(inter_x)
            inter_x = F.relu(inter_x)
        part_scores = self.inter_part_score(inter_x)
        part_scores = F.sigmoid(part_scores)
        # rel_embs = self._forward_relation_embedding(part_scores, rel_matrix)
        # kpt module
        for i in range(self.n_stacked_convs // 2, self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
        for i in range(2):
            layer_name = self._get_deconv_layer_name(i, 'KM')
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
        x = torch.cat([x, inter_x], 1)
        x = self.kpt_score(x)
        return x, part_scores

@ROI_KEYPOINT_HEAD_REGISTRY.register()
class KRCNNPoseRelationHead(nn.Module):
    """
    A standard keypoint rcnn head containing a part relation modeling
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
            num_keypoints: number of keypoint heatmaps to predicts, determines the number of
                           channels in the final output.
        """
        super(KRCNNPoseRelationHead, self).__init__()

        # fmt: off
        # default up_scale to 2 (this can eventually be moved to config)
        rel_matrix_dir     = cfg.MODEL.ROI_KEYPOINT_HEAD.RELATION_DIR
        rel_matrix = pickle.load(open(rel_matrix_dir,'rb'))
        rel_matrix = torch.FloatTensor(rel_matrix)
        self.rel_matrix = nn.Parameter(data=rel_matrix, requires_grad=True)
        # word emb
        word_emb_dir = cfg.MODEL.ROI_KEYPOINT_HEAD.WORD_EMB_DIR
        word_emb = pickle.load(open(word_emb_dir, 'rb'))
        word_emb = torch.FloatTensor(word_emb)
        self.word_emb = nn.Parameter(data=word_emb, requires_grad=True)
        self.rel_scale = 0.5

        self.up_scale      = 2
        layer_channels     = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_HEAD_DIM
        self.relation_dims = cfg.MODEL.ROI_KEYPOINT_HEAD.RELATION_DIM
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        self.n_stacked_convs = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_STACKED_CONVS
        num_parts = num_keypoints // 2 + 1
        in_channels   = input_shape.channels
        deconv_kernel = 4
        # fmt: on

        self.blocks = []
        for i in range(self.n_stacked_convs):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module(self._get_layer_name(i), module)
            self.blocks.append(module)
            in_channels = layer_channels

        self.inter_part_score = ConvTranspose2d(
            layer_channels, num_parts, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        # self.inter_part_score = Conv2d(layer_channels, num_parts, 3, stride=1, padding=1)
        self.R_emb = nn.Sequential(
            nn.Conv2d(self.relation_dims, input_shape.channels, 1, stride=1, padding=0))
            # nn.ReLU(inplace=True))
        # self.kpt_score = Conv2d(layer_channels+ layer_channels, num_keypoints, 3, stride=1, padding=1)
        self.kpt_score = ConvTranspose2d(
            layer_channels+ layer_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )

        for name, param in self.named_parameters():
            if name =="rel_matrix" or name == "word_emb":
                continue
            print("init:", name)
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
    def _get_layer_name(self, i):
        layer_name = "conv_fcn{}".format(i + 1)
        return layer_name

    def _get_deconv_layer_name(self,i, prefix=''):
        layer_name = prefix + "deconv_fcn{}".format(i + 1)
        return layer_name
    def _forward_relation_embedding(self, part_scores, rel_matrix=None, word_emb=None):
        '''

        :param part_scores: B x NUM_PARTS x W x H
        :param rel_matrix: NUM_PARTS x NUM_PARTS
        :param word_emb: NUM_PARTS X RELATION DIM
        :return:
        '''
        rel_matrix = torch.matmul(rel_matrix, word_emb)
        B, n_parts, h, w = part_scores.size(0), part_scores.size(1),part_scores.size(2), part_scores.size(3)
        # rel_matrix = rel_matrix.unsqueeze(0).repeat(B,1,1)
        part_scores = part_scores.reshape((B, n_parts, h*w)).permute(0,2,1)
        rel_embs = torch.matmul(part_scores, rel_matrix)
        rel_embs = rel_embs.permute(0, 2, 1).reshape((B, self.relation_dims, h, w))
        rel_embs = self.R_emb(rel_embs)
        # rel_embs = interpolate(rel_embs,self.rel_scale)
        rel_embs = F.interpolate(rel_embs, (int(h*self.rel_scale), int(w*self.rel_scale)), mode="bilinear", align_corners=False)
        return rel_embs

    def forward(self, x, rel_matrix=None):
        # part module
        for i in range(self.n_stacked_convs // 2):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
        inter_x = x
        # for i in range(2):
        #     layer_name = self._get_deconv_layer_name(i, 'PM')
        #     inter_x = getattr(self, layer_name)(inter_x)
        #     inter_x = F.relu(inter_x)
        part_scores_logits = self.inter_part_score(inter_x)
        part_scores = F.sigmoid(part_scores_logits)
        rel_embs = self._forward_relation_embedding(part_scores, self.rel_matrix, self.word_emb)
        part_scores = interpolate(part_scores_logits, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        part_scores = F.sigmoid(part_scores)
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            rel_embs = getattr(self, layer_name)(rel_embs)
            rel_embs = F.relu(rel_embs)
        # kpt module
        for i in range(self.n_stacked_convs // 2, self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
        # for i in range(2):
        #     layer_name = self._get_deconv_layer_name(i, 'KM')
        #     x = getattr(self, layer_name)(x)
        #     x = F.relu(x)
        x = torch.cat([x, rel_embs], 1)
        x = self.kpt_score(x)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x, part_scores


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class KRCNNPoseRelationConveHead(nn.Module):
    """
    A standard keypoint rcnn head containing a part relation modeling
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
            num_keypoints: number of keypoint heatmaps to predicts, determines the number of
                           channels in the final output.
        """
        super(KRCNNPoseRelationConveHead, self).__init__()

        # fmt: off
        # default up_scale to 2 (this can eventually be moved to config)
        rel_matrix_dir     = cfg.MODEL.ROI_KEYPOINT_HEAD.RELATION_DIR
        rel_matrix = pickle.load(open(rel_matrix_dir,'rb'))
        print("relation:",rel_matrix.shape)
        rel_matrix = torch.FloatTensor(rel_matrix)
        self.rel_matrix = nn.Parameter(data=rel_matrix, requires_grad=True)


        self.up_scale      = 2
        layer_channels     = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_HEAD_DIM
        self.relation_dims = cfg.MODEL.ROI_KEYPOINT_HEAD.RELATION_DIM
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        self.n_stacked_convs = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_STACKED_CONVS
        num_parts = num_keypoints // 2 + 1
        in_channels   = input_shape.channels
        deconv_kernel = 4
        # fmt: on

        self.blocks = []
        for i in range(self.n_stacked_convs):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module(self._get_layer_name(i), module)
            self.blocks.append(module)
            in_channels = layer_channels

        # self.score_lowres = ConvTranspose2d(
        #     in_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        # )

        self.kpt_weight = torch.FloatTensor(np.zeros((in_channels, num_keypoints,deconv_kernel,deconv_kernel)))
        self.kpt_weight = nn.Parameter(data=self.kpt_weight, requires_grad=True)
        self.kpt_bias = torch.FloatTensor(np.zeros((num_keypoints)))
        self.kpt_bias = nn.Parameter(data=self.kpt_bias, requires_grad=True)
        for name, param in self.named_parameters():

            if name =="rel_matrix":
                continue
            print("init:", name)
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
    def _get_layer_name(self, i):
        layer_name = "conv_fcn{}".format(i + 1)
        return layer_name

    def _get_deconv_layer_name(self,i, prefix=''):
        layer_name = prefix + "deconv_fcn{}".format(i + 1)
        return layer_name
    def _forward_relation_embedding(self, weight, rel_matrix=None):
        '''

        :param part_scores: out x in x w x h
        :param rel_matrix: out x out
        :return:
        '''
        n_in, n_out, h, w = weight.size(0), weight.size(1),weight.size(2), weight.size(3)
        # print(weight.size())
        # rel_matrix = rel_matrix.unsqueeze(0).repeat(B,1,1)
        weight = weight.permute((1,0,2,3)).reshape((n_out, n_in*h*w))
        rel_weight = torch.matmul(rel_matrix, weight)
        rel_weight = rel_weight.reshape((n_out, n_in, h, w)).permute((1,0,2,3))
        return rel_weight

    def forward(self, x, rel_matrix=None):
        if len(x) == 0:
            return torch.zeros(size=(0, 0, 0, 0), device=x.device)
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)

        kpt_weight = self._forward_relation_embedding(self.kpt_weight, self.rel_matrix)
        x =  nn.functional.conv_transpose2d(x, weight=kpt_weight, bias=self.kpt_bias, padding=1, stride=2)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        # x = self.score_lowres(x)

        return x

@ROI_KEYPOINT_HEAD_REGISTRY.register()
class KRCNNPoseRelationConvHead(nn.Module):
    """
    A standard keypoint rcnn head containing a part relation modeling
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
            num_keypoints: number of keypoint heatmaps to predicts, determines the number of
                           channels in the final output.
        """
        super(KRCNNPoseRelationConvHead, self).__init__()

        # fmt: off
        # default up_scale to 2 (this can eventually be moved to config)
        # rel_matrix_dir     = cfg.MODEL.ROI_KEYPOINT_HEAD.PART_RELATION_DIR
        # rel_matrix = pickle.load(open(rel_matrix_dir,'rb'))
        # rel_matrix = torch.FloatTensor(rel_matrix)
        # self.rel_matrix = nn.Parameter(data=rel_matrix, requires_grad=False)
        # kpt relation matrix
        kpt_rel_matrix_dir = cfg.MODEL.ROI_KEYPOINT_HEAD.KPT_RELATION_DIR
        kpt_rel_matrix = pickle.load(open(kpt_rel_matrix_dir, 'rb'))
        kpt_rel_matrix = torch.FloatTensor(kpt_rel_matrix)
        self.kpt_rel_matrix = nn.Parameter(data=kpt_rel_matrix, requires_grad=True)

        # word emb
        # word_emb_dir = cfg.MODEL.ROI_KEYPOINT_HEAD.WORD_EMB_DIR
        # word_emb = pickle.load(open(word_emb_dir, 'rb'))
        # word_emb = torch.FloatTensor(word_emb)
        # self.word_emb = nn.Parameter(data=word_emb, requires_grad=True)
        self.rel_scale = 0.5
        # kpt word emb
        # kpt_word_emb_dir = cfg.MODEL.ROI_KEYPOINT_HEAD.KPT_WORD_EMB_DIR
        # kpt_word_emb = pickle.load(open(kpt_word_emb_dir, 'rb'))
        # kpt_word_emb = torch.FloatTensor(kpt_word_emb)
        # self.kpt_word_emb = nn.Parameter(data=kpt_word_emb, requires_grad=True)

        self.up_scale      = 2
        layer_channels     = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_HEAD_DIM
        self.relation_dims = cfg.MODEL.ROI_KEYPOINT_HEAD.RELATION_DIM
        # self.kpt_relation_dim = cfg.MODEL.ROI_KEYPOINT_HEAD.KPT_RELATION_DIM
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        self.n_stacked_convs = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_STACKED_CONVS
        num_parts = num_keypoints // 2 + 1
        in_channels   = input_shape.channels
        deconv_kernel = 4
        self.feat_channels = layer_channels
        self.deconv_kernel = deconv_kernel
        self.num_keypoints = num_keypoints
        # fmt: on

        self.blocks = []
        for i in range(self.n_stacked_convs):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module(self._get_layer_name(i), module)
            self.blocks.append(module)
            in_channels = layer_channels

        # self.inter_part_score = ConvTranspose2d(
        #     layer_channels, num_parts, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        # )
        # self.inter_part_score = Conv2d(layer_channels, num_parts, 3, stride=1, padding=1)
        self.R_emb = nn.Sequential(
            nn.Conv2d(layer_channels+layer_channels, layer_channels, 1, stride=1, padding=0))
            # nn.ReLU(inplace=True))
        # self.kpt_score = Conv2d(layer_channels+ layer_channels, num_keypoints, 3, stride=1, padding=1)
        self.kpt_score = ConvTranspose2d(
            layer_channels, num_keypoints+1, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.final_kpt_score = ConvTranspose2d(
            layer_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        # self.fcs = []
        # for k in range(2):
        #     fc1 = nn.Linear(layer_channels, layer_channels)
        #     self.add_module("fc1_levels_{}".format(k + 1), fc1)
        #     fc2 = nn.Linear(layer_channels, layer_channels)
        #     self.add_module("att_levels_{}".format(k + 1), fc2)
        #     self.fcs.append([fc1, nn.ReLU(), fc2, nn.Sigmoid()])
        # ama_conv_emb = Conv2d(layer_channels, layer_channels, 3, stride=1, padding=1)
        # self.add_module('ama_dynamic_conv_emb', ama_conv_emb)
        # weight_generator = []
        # param_size = (2 * layer_channels)*deconv_kernel*deconv_kernel
        # weight_generator.append(nn.Linear(self.kpt_relation_dim, param_size))
        # weight_generator.append(nn.LeakyReLU(0.02))
        # for i in range(3):
        #     weight_generator.append(nn.Linear(param_size, param_size))
        #     weight_generator.append(nn.LeakyReLU(0.02))
        # weight_generator.append(
        #     nn.Linear(param_size, param_size))
        # self.weight_generator = nn.Sequential(*weight_generator)
        for name, param in self.named_parameters():
            if name =="kpt_rel_matrix" or name == "kpt_word_emb":
                continue
            print("init:", name)
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
    def _get_layer_name(self, i):
        layer_name = "conv_fcn{}".format(i + 1)
        return layer_name

    def _get_deconv_layer_name(self,i, prefix=''):
        layer_name = prefix + "deconv_fcn{}".format(i + 1)
        return layer_name
    def _forward_relation_embedding(self, features, part_scores, rel_matrix=None, kpt_weight=None):
        '''

        :param part_scores: B x NUM_PARTS x W x H
        :param rel_matrix: NUM_PARTS x NUM_PARTS
        :param word_emb: NUM_PARTS X RELATION DIM
        :return:
        '''
        n_out, n_in, k_size_h, k_size_w = kpt_weight.size()
        kpt_weight = kpt_weight.permute((1,0,2,3))
        kpt_weight = torch.mean(kpt_weight,[2,3])
        kpt_weight = kpt_weight.reshape((n_in, n_out))
        rel_matrix = torch.matmul(rel_matrix, kpt_weight)
        B, n_parts, h, w = part_scores.size(0), part_scores.size(1),part_scores.size(2), part_scores.size(3)
        # rel_matrix = rel_matrix.unsqueeze(0).repeat(B,1,1)
        part_scores = part_scores.reshape((B, n_parts, h*w)).permute(0,2,1)
        rel_embs = torch.matmul(part_scores, rel_matrix)
        rel_embs = rel_embs.permute(0, 2, 1).reshape((B, n_out, h, w))
        rel_embs = F.interpolate(rel_embs, (int(h * self.rel_scale), int(w * self.rel_scale)), mode="bilinear",
                                 align_corners=False)
        vis_rel_embs = torch.cat([features, rel_embs], 1)
        vis_rel_embs = self.R_emb(vis_rel_embs)
        vis_rel_embs = F.relu(vis_rel_embs)
        # rel_embs = interpolate(rel_embs,self.rel_scale)

        return vis_rel_embs
    def _forward_kpt_weight_generate(self, kpt_rel_matrix=None, word_emb=None):
        '''

        :param weight: feat_dim x num_kpt x k_size x k_size
        :param kpt_rel_matrix: num_kpt x num_kpt
        :param word_emb: num_kpt x word_emb_dim
        :return:
        '''
        kpt_sem_embs = torch.matmul(kpt_rel_matrix, word_emb)

        weight = self.weight_generator(kpt_sem_embs)
        kpt_weight = weight.reshape((self.num_keypoints, self.feat_channels*2, self.deconv_kernel, self.deconv_kernel))
        kpt_weight = kpt_weight.permute((1,0,2,3))
        return kpt_weight
    def _ama_module_forward(self, features):
        assert len(features) > 0, 'invalid inputs for ama module'

        for i in range(len(features)):
            i_latent_feaure = torch.mean(features[i], [2, 3])
            i_latent_output = torch.flatten(i_latent_feaure, start_dim=1)
            for layer in self.fcs[i]:
                i_latent_output = layer(i_latent_output)
            features[i] = features[i] * (i_latent_output.unsqueeze(2).unsqueeze(3))
            if i == 0:
                out_features = features[i]
            else:
                out_features = out_features + features[i]
        # out_features = torch.cat(features, 1)
        out_features = getattr(self, 'ama_dynamic_conv_emb')(out_features)
        out_features = F.relu(out_features)
        return out_features
    def forward(self, features, rel_matrix=None):
        # part module
        x = features
        for i in range(self.n_stacked_convs // 2):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
        inter_x = x
        part_scores_logits = self.kpt_score(inter_x)
        # B, num_kpts, size_h, size_w = part_scores_logits.size(0), part_scores_logits.size(1),part_scores_logits.size(2),part_scores_logits.size(3)
        part_scores = F.softmax(part_scores_logits,dim=1)
        part_scores = part_scores[:,1:, :, :]
        # part_scores = part_scores.reshape((B,num_kpts,size_h,size_w))
        rel_embs = self._forward_relation_embedding(x, part_scores, self.kpt_rel_matrix, self.kpt_score.weight[:,1:,:,:])
        part_scores = interpolate(part_scores_logits, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        part_scores = part_scores[:, 1:, :, :]
        # print(part_scores.size())
        for i in range(self.n_stacked_convs // 2, self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            rel_embs = getattr(self, layer_name)(rel_embs)
            rel_embs = F.relu(rel_embs)
        # kpt module
        # for i in range(self.n_stacked_convs // 2, self.n_stacked_convs):
        #     layer_name = self._get_layer_name(i)
        #     x = getattr(self, layer_name)(x)
        #     x = F.relu(x)
        # x = [x, rel_embs]
        # x = self._ama_module_forward(x)
        # kpt_weight = self._forward_kpt_weight_generate(self.kpt_rel_matrix, self.kpt_word_emb)
        # x = nn.functional.conv_transpose2d(x, weight=kpt_weight, padding=1, stride=2)
        kpt_scores = self.final_kpt_score(rel_embs)
        kpt_scores = interpolate(kpt_scores, scale_factor=self.up_scale, mode="bilinear", align_corners=False)


        return kpt_scores, part_scores.contiguous()