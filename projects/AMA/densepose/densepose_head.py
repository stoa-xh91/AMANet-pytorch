# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from detectron2.layers import Conv2d, ConvTranspose2d, interpolate
from detectron2.structures.boxes import matched_boxlist_iou
from detectron2.utils.registry import Registry
# import cv2
import fvcore.nn.weight_init as weight_init
from .structures import DensePoseOutput
from .nonlocal_helper import NONLocalBlock2D
from .transformer_helper import MultiHeadAttention
import pickle
ROI_DENSEPOSE_HEAD_REGISTRY = Registry("ROI_DENSEPOSE_HEAD")


def initialize_module_params(module):
    for name, param in module.named_parameters():

        if 'deconv_p' in name and "norm" in name:
            continue
        if 'ASPP' in name and "norm" in name:
            continue
        if 'dp_sem_head' in name and "norm" in name:
            continue
        if 'body_kpt' in name or "dp_emb_layer" in name or "kpt_surface_transfer_matrix" in name:
            print('ignore init ',name)
            continue
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            if len(param.size())<2:
                print('ignore:',name)
                continue
            if 'transformer' in name:
                print('init:', name)
            nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
        elif "body_mask" in name or "body_part" in name or "bbox_surface_transfer_matrix" in name \
                or "part_surface_transfer_matrix" in name:
            print("init ",name)
            nn.init.normal_(param, std=0.001)

def gaussian_initialize_module_params(module):
    for name, param in module.named_parameters():
        if 'body_kpt' in name or "dp_emb_layer" in name:
            print('ignore init ', name)
            continue
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.normal_(param, std=0.001)
        elif "body_mask" in name or "body_part" in name or "kpt_surface_transfer_matrix" in name \
                or "bbox_surface_transfer_matrix" in name or "part_surface_transfer_matrix" in name:
            print("init ",name)
            nn.init.normal_(param, std=0.001)

@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseV1ConvXHead(nn.Module):
    def __init__(self, cfg, input_channels):
        super(DensePoseV1ConvXHead, self).__init__()
        # fmt: off
        hidden_dim           = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM
        kernel_size          = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL
        self.n_stacked_convs = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS
        # fmt: on
        pad_size = kernel_size // 2
        n_channels = input_channels
        for i in range(self.n_stacked_convs):
            layer = Conv2d(n_channels, hidden_dim, kernel_size, stride=1, padding=pad_size)
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        self.n_out_channels = n_channels
        initialize_module_params(self)

    def forward(self, features):
        x = features
        output = x
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
            output = x
        return output

    def _get_layer_name(self, i):
        layer_name = "body_conv_fcn{}".format(i + 1)
        return layer_name

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        ]

        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
            # nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseAMAHead(nn.Module):
    def __init__(self, cfg, input_channels):
        super(DensePoseAMAHead, self).__init__()
        # fmt: off
        self.dp_semseg_on = cfg.MODEL.ROI_DENSEPOSE_HEAD.SEMSEG_ON
        hidden_dim           = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM
        kernel_size          = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL
        deconv_kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        self.n_stacked_convs = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS

        self.scale_factors = [2, 4, 4, 8]
        self.up_idx = [[2], [1,2], [1,2], [0,1,2]]
        self.up_modules = []
        self.fcs = []
        # fmt: on
        pad_size = kernel_size // 2
        n_channels = input_channels
        for i in range(self.n_stacked_convs):
            layer = Conv2d(n_channels, hidden_dim, kernel_size, stride=1, padding=pad_size)
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        self._add_devonve_modules(hidden_dim, deconv_kernel_size)
        self._add_ama_modules(hidden_dim)
        layer = Conv2d(hidden_dim*2, hidden_dim, 1, stride=1, padding=0)
        self.i_emb_layer_name = 'instance_embedding_layer'
        self.add_module(self.i_emb_layer_name+'1', layer)
        self.n_out_channels = n_channels

        # gaussian_initialize_module_params(self)
        initialize_module_params(self)
    def _add_devonve_modules(self, hidden_dim, deconv_kernel_size):

        for k in range(3):
            tmp_ops = []
            deconv = ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=deconv_kernel_size,
                stride=2,
                padding=int(deconv_kernel_size / 2 - 1),
            )
            tmp_ops.append(deconv)
            tmp_ops.append(nn.ReLU())
            deconv_p1 = nn.Sequential(*tmp_ops)
            self.add_module('deconv_p' + str(k + 1), deconv_p1)
            tmp_ops.append(deconv_p1)
            self.up_modules.append(deconv_p1)
    def _add_ama_modules(self, hidden_dim):

        for k in range(4):
            fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.add_module("fc1_levels_{}".format(k + 1), fc1)
            fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.add_module("att_levels_{}".format(k + 1), fc2)
            self.fcs.append([fc1, nn.ReLU(), fc2, nn.Sigmoid()])
        ama_conv_emb = Conv2d(hidden_dim*4, hidden_dim, 3, stride=1, padding=1)
        self.add_module('ama_static_conv_emb', ama_conv_emb)
        ama_conv_emb = Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1)
        self.add_module('ama_dynamic_conv_emb', ama_conv_emb)

    def _ama_module_forward(self, features):
        assert len(features) > 0, 'invalid inputs for ama module'
        # static aggregation
        static_out_features = torch.cat(features, 1)
        static_out_features = getattr(self, 'ama_static_conv_emb')(static_out_features)
        static_out_features = F.relu(static_out_features)

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
        out_features = torch.cat([static_out_features, out_features], 1)
        return out_features
    def _ama_upsample_forward(self, features):
        for i in range(len(features)):
            for j in range(len(self.up_idx[i])):
                features[i] = self.up_modules[self.up_idx[i][j]](features[i])
        return features

    def forward(self, features, segm_features=None, forward_type='dp'):

        if forward_type == 'kp':
            x = features
            for j in range(self.n_stacked_convs):
                layer_name = self._get_layer_name(j)
                x = getattr(self, layer_name)(x)
                x = F.relu(x)
                return x
        # Multi Path
        lower_multi_roi_features = []
        for i in range(len(features)):
            x = features[i]
            for j in range(self.n_stacked_convs):
                layer_name = self._get_layer_name(j)
                x = getattr(self, layer_name)(x)
                x = F.relu(x)
            lower_multi_roi_features.append(x)

        multi_roi_features = self._ama_upsample_forward(lower_multi_roi_features)

        output = self._ama_module_forward(multi_roi_features)

        for i in range(1):
            output = getattr(self, self.i_emb_layer_name+str(i+1))(output)
            output = F.relu(output)
        return output, multi_roi_features

    def _get_layer_name(self, i):
        layer_name = "body_conv_fcn{}".format(i + 1)
        return layer_name

    def _get_deconv_layer_name(self,i, prefix=''):
        layer_name = prefix + "body_deconv_fcn{}".format(i + 1)
        return layer_name

@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePosePredictor(nn.Module):

    NUM_ANN_INDICES = 15

    def __init__(self, cfg, input_channels):
        super(DensePosePredictor, self).__init__()
        dim_in = input_channels
        dim_out_ann_index = self.NUM_ANN_INDICES
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        self.ann_index_lowres = ConvTranspose2d(
            dim_in, dim_out_ann_index, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.index_uv_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.u_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.v_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.m_lowres = ConvTranspose2d(
            dim_in, 2, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.scale_factor = cfg.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE
        initialize_module_params(self)

    def forward(self, head_outputs, params=None):
        ann_index_lowres = self.ann_index_lowres(head_outputs)
        index_uv_lowres = self.index_uv_lowres(head_outputs)
        u_lowres = self.u_lowres(head_outputs)
        v_lowres = self.v_lowres(head_outputs)
        m_lowres = self.m_lowres(head_outputs)

        def interp2d(input):
            return interpolate(
                input, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
            )

        ann_index = interp2d(ann_index_lowres)
        index_uv = interp2d(index_uv_lowres)
        u = interp2d(u_lowres)
        v = interp2d(v_lowres)
        m = interp2d(m_lowres)
        return (ann_index, index_uv, u, v, m), None, None #(ann_index_lowres, index_uv_lowres, u_lowres, v_lowres, m_lowres)

@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseRCNNStarPredictor(nn.Module):

    def __init__(self, cfg, input_channels):
        super(DensePoseRCNNStarPredictor, self).__init__()
        dim_in = input_channels
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        self.index_uv_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.u_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.v_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.m_lowres = ConvTranspose2d(
            dim_in, 2, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.scale_factor = cfg.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE
        initialize_module_params(self)

    def forward(self, head_outputs,params=None):
        ann_index_lowres = None
        index_uv_lowres = self.index_uv_lowres(head_outputs)
        u_lowres = self.u_lowres(head_outputs)
        v_lowres = self.v_lowres(head_outputs)
        m_lowres = self.m_lowres(head_outputs)
        def interp2d(input):
            return interpolate(
                input, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
            )
        ann_index = None
        index_uv = interp2d(index_uv_lowres)
        u = interp2d(u_lowres)
        v = interp2d(v_lowres)
        m = interp2d(m_lowres)
        return (ann_index, index_uv, u, v, m), None, None

@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePosePredictorV2(nn.Module):

    NUM_ANN_INDICES = 15

    def __init__(self, cfg, input_channels):
        super(DensePosePredictorV2, self).__init__()
        dim_in = input_channels
        dim_out_ann_index = self.NUM_ANN_INDICES
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        # self.segm_emb_layer = Conv2d(dim_in, dim_in, 3, stride=1, padding=1)
        self.dp_emb_layer = Conv2d(dim_in, dim_in, 3, stride=1, padding=1)

        self.ann_index_layer = Conv2d(dim_in, dim_out_ann_index, 3, stride=1, padding=1)
        self.index_uv_layer = Conv2d(dim_in, dim_out_patches, 3, stride=1, padding=1)
        self.u_layer = Conv2d(dim_in, dim_out_patches, 3, stride=1, padding=1)
        self.v_layer = Conv2d(dim_in, dim_out_patches, 3, stride=1, padding=1)
        self.m_layer = Conv2d(dim_in, 2, 3, stride=1, padding=1)
        #
        self.inter_m_layer = Conv2d(dim_in, 2, 3, stride=1, padding=1)
        # self.inter_ann_index_layer = Conv2d(dim_in, dim_out_ann_index, 3, stride=1, padding=1)
        # self.inter_index_uv_layer = Conv2d(dim_in, dim_out_patches, 3, stride=1, padding=1)
        # self.inter_u_layer = Conv2d(dim_in, dim_out_patches, 3, stride=1, padding=1)
        # self.inter_v_layer = Conv2d(dim_in, dim_out_patches, 3, stride=1, padding=1)

        initialize_module_params(self)

    def forward(self, head_outputs, inter_outputs):
        # inter_ann_index = self.inter_ann_index_layer(inter_outputs)
        # inter_index_uv = self.inter_index_uv_layer(inter_outputs)
        # inter_u = self.inter_u_layer(inter_outputs)
        # inter_v = self.inter_v_layer(inter_outputs)
        inter_m = self.inter_m_layer(inter_outputs)

        # seg_emb = self.dp_emb_layer(head_outputs)
        dp_emb = self.dp_emb_layer(head_outputs)
        dp_emb = F.relu(dp_emb)
        ann_index = self.ann_index_layer(dp_emb)
        index_uv = self.index_uv_layer(dp_emb)
        u = self.u_layer(dp_emb)
        v = self.v_layer(dp_emb)
        m = self.m_layer(dp_emb)
        return (ann_index, index_uv, u, v, m), inter_m #(inter_ann_index, inter_index_uv, inter_u, inter_v, inter_m)

@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class TaskTransformerAMAPredictorV2(nn.Module):

    NUM_ANN_INDICES = 15

    def __init__(self, cfg, input_channels):
        super(TaskTransformerAMAPredictorV2, self).__init__()
        self.dp_keypoints_on = cfg.MODEL.ROI_DENSEPOSE_HEAD.KPT_ON
        dim_in = input_channels
        dim_out_ann_index = self.NUM_ANN_INDICES
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        dim_out_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        self.dim_in = dim_in
        self.dp_emb_layer = Conv2d(dim_in, dim_in, 3, stride=1, padding=1)

        self.kernel_size = 3

        self.ann_index_layer = Conv2d(dim_in, dim_out_ann_index, 3, stride=1, padding=1)
        self.u_layer = Conv2d(dim_in, dim_out_patches, 3, stride=1, padding=1)
        self.v_layer = Conv2d(dim_in, dim_out_patches, 3, stride=1, padding=1)
        self.i_layer = Conv2d(dim_in, dim_out_patches, 3, stride=1, padding=1)
        self.m_layer = Conv2d(dim_in, 2, 3, stride=1, padding=1)
        self.k_layer = Conv2d(dim_in, dim_out_keypoints, 3, stride=1, padding=1)
        self.bbox_surface_transfer_matrix = Parameter(torch.Tensor(dim_out_patches, 6))

        bbox_weight_size = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        index_weight_size = dim_in * self.kernel_size * self.kernel_size

        bbox_surface_transformer = []
        bbox_surface_transformer.append(nn.Linear(bbox_weight_size, index_weight_size))
        bbox_surface_transformer.append(nn.LeakyReLU(0.02))
        bbox_surface_transformer.append(nn.Linear(index_weight_size, index_weight_size))
        self.bbox_surface_convertor = nn.Sequential(*bbox_surface_transformer)
        #
        self.part_surface_transfer_matrix = Parameter(torch.Tensor(
            dim_out_patches, dim_out_ann_index))
        part_weight_size = dim_in*3*3
        part_surface_transformer = []
        part_surface_transformer.append(nn.Linear(part_weight_size, index_weight_size))
        part_surface_transformer.append(nn.LeakyReLU(0.02))
        self.part_surface_convertor = nn.Sequential(*part_surface_transformer)
        #
        self.kpt_surface_transfer_matrix = Parameter(torch.Tensor(
            dim_out_patches, dim_out_keypoints))
        kpt_weight_size = dim_in * 3 * 3
        kpt_surface_transformer = []
        kpt_surface_transformer.append(nn.Linear(kpt_weight_size, index_weight_size))
        kpt_surface_transformer.append(nn.LeakyReLU(0.02))
        self.kpt_surface_convertor = nn.Sequential(*kpt_surface_transformer)
        #
        self.param_aggregator = nn.Sequential(*[nn.Conv1d(3*dim_out_patches,dim_out_patches, 1), nn.LeakyReLU(0.02)])
        self.parameter_encoder = MultiHeadAttention(index_weight_size, 1)
        self.parameter_decoder = MultiHeadAttention(index_weight_size, 1)
        self.param_predictor = nn.Linear(index_weight_size, index_weight_size)
        dim_in = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM
        for i in range(2):
            layer = Conv2d(dim_in, dim_out_ann_index, 3, stride=1, padding=1)
            layer_name = self._get_layer_name(i, 'ann_index')
            self.add_module(layer_name, layer)
            layer = Conv2d(dim_in, dim_out_patches, 3, stride=1, padding=1)
            layer_name = self._get_layer_name(i, 'u')
            self.add_module(layer_name, layer)
            layer = Conv2d(dim_in, dim_out_patches, 3, stride=1, padding=1)
            layer_name = self._get_layer_name(i, 'v')
            self.add_module(layer_name, layer)
            layer = Conv2d(dim_in, 2, 3, stride=1, padding=1)
            layer_name = self._get_layer_name(i, 'body_mask')
            self.add_module(layer_name, layer)
        initialize_module_params(self)

    def _get_layer_name(self, i,prefix=''):
        layer_name = prefix+"_inter_body_conv_fcn{}".format(i + 1)
        return layer_name

    def generate_surface_weights(self, bbox_weights, part_weights, kpt_weights):
        init_index_weight = self.i_layer.weight
        #
        bbox_weights = torch.matmul(self.bbox_surface_transfer_matrix, bbox_weights)
        bbox_weights = self.bbox_surface_convertor(bbox_weights)
        bbox_weights = bbox_weights[None, :, :]
        #
        n_out, n_in = part_weights.size(0), part_weights.size(1)
        part_weights = part_weights.reshape(n_out,-1)
        part_surface_weight = torch.matmul(self.part_surface_transfer_matrix, part_weights)
        part_surface_weight = self.part_surface_convertor(part_surface_weight)
        part_surface_weight = part_surface_weight[None, :, :]
        #
        n_out, n_in = kpt_weights.size(0), kpt_weights.size(1)
        kpt_weights = kpt_weights.reshape(n_out, -1)
        kpt_surface_weight = torch.matmul(self.kpt_surface_transfer_matrix, kpt_weights)
        kpt_surface_weight = self.kpt_surface_convertor(kpt_surface_weight)
        kpt_surface_weight = kpt_surface_weight[None, :, :]
        #
        params_pool = self.param_aggregator(torch.cat([bbox_weights, part_surface_weight, kpt_surface_weight], dim=1))
        surface_weights_candidates = self.parameter_encoder(params_pool, params_pool, init_index_weight)
        surface_weights_decoding = self.parameter_decoder(surface_weights_candidates, surface_weights_candidates,
                                                               init_index_weight)
        body_surface_weight = self.param_predictor(surface_weights_decoding.squeeze())
        body_surface_weight = body_surface_weight.reshape((self.part_surface_transfer_matrix.size(0), self.dim_in, 3, 3))
        return body_surface_weight

    def forward(self, head_outputs, inter_outputs, task_params=None):
        assert len(inter_outputs) == 4 or len(inter_outputs) == 5, "invalid number of inter outputs"
        inter_shallow_level_out = inter_outputs[0] + inter_outputs[1]
        inter_deep_level_out = inter_outputs[2] + inter_outputs[3]
        inter_features = [inter_shallow_level_out, inter_deep_level_out]
        inter_out1 = []
        inter_out2 = []
        # inter outputs
        dp_emb = self.dp_emb_layer(head_outputs)
        dp_emb = F.relu(dp_emb)
        u = self.u_layer(dp_emb)
        v = self.v_layer(dp_emb)
        m = self.m_layer(dp_emb)
        k = self.k_layer(dp_emb)
        index_uv = self.i_layer(dp_emb)
        ann_index = self.ann_index_layer(dp_emb)
        for i in range(2):
            if i == 0:
                inter_out = inter_out1
            else:
                inter_out = inter_out2
            layer_name = self._get_layer_name(i, 'ann_index')
            inter_ann_index = getattr(self, layer_name)(inter_features[i])
            inter_out.append(inter_ann_index)
            body_surface_weight = self.generate_surface_weights(task_params,
                                                                self.ann_index_layer.weight,
                                                                self.k_layer.weight)
            inter_index_uv = nn.functional.conv2d(dp_emb, weight=body_surface_weight, padding=1, stride=1)
            inter_out.append(inter_index_uv)
            layer_name = self._get_layer_name(i, 'u')
            inter_ann_index = getattr(self, layer_name)(inter_features[i])
            inter_out.append(inter_ann_index)
            layer_name = self._get_layer_name(i, 'v')
            inter_ann_index = getattr(self, layer_name)(inter_features[i])
            inter_out.append(inter_ann_index)
            layer_name = self._get_layer_name(i, 'body_mask')
            inter_ann_index = getattr(self, layer_name)(inter_features[i])
            inter_out.append(inter_ann_index)
        index_uv = (index_uv+inter_out1[1])*0.5
        return (ann_index, index_uv, u, v, m, k), inter_out1, inter_out2

class DensePoseKeypointsPredictor(nn.Module):

    def __init__(self, cfg, input_channels):
        super(DensePoseKeypointsPredictor, self).__init__()
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        self.up_scale = cfg.MODEL.ROI_DENSEPOSE_HEAD.KPT_UP_SCALE
        # fmt: on
        predictor = []
        if self.up_scale == 2:
            deconv_kernel = 4
            score_lowres = ConvTranspose2d(
                input_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
            )
            predictor.append(score_lowres)
            self.predictor = nn.Sequential(*predictor)
        elif self.up_scale == 4:
            deconv_kernel = 4
            predictor.append(ConvTranspose2d(
                input_channels, input_channels, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
            ))
            predictor.append(nn.ReLU())
            predictor.append(ConvTranspose2d(
                input_channels, input_channels, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
            ))
            predictor.append(nn.ReLU())
            predictor.append(Conv2d(input_channels, num_keypoints, 3, stride=1, padding=1))
            self.predictor = nn.Sequential(*predictor)
        else:
            predictor.append(Conv2d(input_channels, input_channels, 3, stride=1, padding=1))
            predictor.append(nn.ReLU())
            predictor.append(Conv2d(input_channels, num_keypoints, 3, stride=1, padding=1))
            self.predictor = nn.Sequential(*predictor)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.predictor(x)
        if self.up_scale > 2:
            x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)

        return x

class DensePoseDataFilter(object):
    def __init__(self, cfg, iou_threshold=0.7):
        self.iou_threshold = iou_threshold
        self.cfg = cfg

    @torch.no_grad()
    def __call__(self, proposals_with_targets):
        """
        Filters proposals with targets to keep only the ones relevant for
        DensePose training
        proposals: list(Instances), each element of the list corresponds to
            various instances (proposals, GT for boxes and densepose) for one
            image
        """
        proposals_filtered = []
        for proposals_per_image in proposals_with_targets:
            if not hasattr(proposals_per_image, "gt_densepose"):
                continue
            assert hasattr(proposals_per_image, "gt_boxes")
            assert hasattr(proposals_per_image, "proposal_boxes")
            gt_boxes = proposals_per_image.gt_boxes
            est_boxes = proposals_per_image.proposal_boxes
            # apply match threshold for densepose head
            iou = matched_boxlist_iou(gt_boxes, est_boxes)
            iou_select = iou > self.iou_threshold
            proposals_per_image = proposals_per_image[iou_select]
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.proposal_boxes)
            # filter out any target without densepose annotation
            gt_densepose = proposals_per_image.gt_densepose
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.gt_densepose)
            selected_indices = [
                i for i, dp_target in enumerate(gt_densepose) if dp_target is not None
            ]
            if len(selected_indices) != len(gt_densepose):
                proposals_per_image = proposals_per_image[selected_indices]
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.proposal_boxes)
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.gt_densepose)
            proposals_filtered.append(proposals_per_image)
            # print('per image:',len(proposals_per_image))
        return proposals_filtered


def build_densepose_head(cfg, input_channels):
    head_name = cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME
    return ROI_DENSEPOSE_HEAD_REGISTRY.get(head_name)(cfg, input_channels)

def build_densepose_predictor(cfg, input_channels):
    predictor_name = cfg.MODEL.ROI_DENSEPOSE_HEAD.PREDICTOR
    return ROI_DENSEPOSE_HEAD_REGISTRY.get(predictor_name)(cfg, input_channels)



def build_densepose_data_filter(cfg):
    dp_filter = DensePoseDataFilter(cfg, cfg.MODEL.ROI_DENSEPOSE_HEAD.FG_IOU_THRESHOLD)
    return dp_filter


def densepose_inference(densepose_outputs, detections):
    """
    Infer dense pose estimate based on outputs from the DensePose head
    and detections. The estimate for each detection instance is stored in its
    "pred_densepose" attribute.

    Args:
        densepose_outputs (tuple(`torch.Tensor`)): iterable containing 4 elements:
            - s (:obj: `torch.Tensor`): segmentation tensor of size (N, A, H, W),
            - i (:obj: `torch.Tensor`): classification tensor of size (N, C, H, W),
            - u (:obj: `torch.Tensor`): U coordinates for each class of size (N, C, H, W),
            - v (:obj: `torch.Tensor`): V coordinates for each class of size (N, C, H, W),
            where N is the total number of detections in a batch,
                  A is the number of segmentations classes (e.g. 15 for coarse body parts),
                  C is the number of labels (e.g. 25 for fine body parts),
                  W is the resolution along the X axis
                  H is the resolution along the Y axis
        detections (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Instances are modified by this method: "pred_densepose" attribute
            is added to each instance, the attribute contains the corresponding
            DensePoseOutput object.
    """

    # DensePose outputs: segmentation, body part indices, U, V
    s, index_uv, u, v, m = densepose_outputs
    k = 0
    # s = F.softmax(s, dim=1) if s is not None else s
    for detection in detections:
        n_i = len(detection)
        s_i = s[k : k + n_i] if s is not None else s
        # detection.pred_parsings = s_i if s_i is not None else None
        index_uv_i = index_uv[k : k + n_i]
        u_i = u[k : k + n_i]
        v_i = v[k : k + n_i]
        m_i = m[k : k + n_i] if m is not None else m
        densepose_output_i = DensePoseOutput(s_i, index_uv_i, u_i, v_i, m_i)
        detection.pred_densepose = densepose_output_i
        k += n_i


def _linear_interpolation_utilities(v_norm, v0_src, size_src, v0_dst, size_dst, size_z):
    """
    Computes utility values for linear interpolation at points v.
    The points are given as normalized offsets in the source interval
    (v0_src, v0_src + size_src), more precisely:
        v = v0_src + v_norm * size_src / 256.0
    The computed utilities include lower points v_lo, upper points v_hi,
    interpolation weights v_w and flags j_valid indicating whether the
    points falls into the destination interval (v0_dst, v0_dst + size_dst).

    Args:
        v_norm (:obj: `torch.Tensor`): tensor of size N containing
            normalized point offsets
        v0_src (:obj: `torch.Tensor`): tensor of size N containing
            left bounds of source intervals for normalized points
        size_src (:obj: `torch.Tensor`): tensor of size N containing
            source interval sizes for normalized points
        v0_dst (:obj: `torch.Tensor`): tensor of size N containing
            left bounds of destination intervals
        size_dst (:obj: `torch.Tensor`): tensor of size N containing
            destination interval sizes
        size_z (int): interval size for data to be interpolated

    Returns:
        v_lo (:obj: `torch.Tensor`): int tensor of size N containing
            indices of lower values used for interpolation, all values are
            integers from [0, size_z - 1]
        v_hi (:obj: `torch.Tensor`): int tensor of size N containing
            indices of upper values used for interpolation, all values are
            integers from [0, size_z - 1]
        v_w (:obj: `torch.Tensor`): float tensor of size N containing
            interpolation weights
        j_valid (:obj: `torch.Tensor`): uint8 tensor of size N containing
            0 for points outside the estimation interval
            (v0_est, v0_est + size_est) and 1 otherwise
    """
    v = v0_src + v_norm * size_src / 256.0
    j_valid = (v - v0_dst >= 0) * (v - v0_dst < size_dst)
    v_grid = (v - v0_dst) * size_z / size_dst
    v_lo = v_grid.floor().long().clamp(min=0, max=size_z - 1)
    v_hi = (v_lo + 1).clamp(max=size_z - 1)
    v_grid = torch.min(v_hi.float(), v_grid)
    v_w = v_grid - v_lo.float()
    return v_lo, v_hi, v_w, j_valid


def _grid_sampling_utilities(
    zh, zw, bbox_xywh_est, bbox_xywh_gt, index_gt, x_norm, y_norm, index_bbox
):
    """
    Prepare tensors used in grid sampling.

    Args:
        z_est (:obj: `torch.Tensor`): tensor of size (N,C,H,W) with estimated
            values of Z to be extracted for the points X, Y and channel
            indices I
        bbox_xywh_est (:obj: `torch.Tensor`): tensor of size (N, 4) containing
            estimated bounding boxes in format XYWH
        bbox_xywh_gt (:obj: `torch.Tensor`): tensor of size (N, 4) containing
            matched ground truth bounding boxes in format XYWH
        index_gt (:obj: `torch.Tensor`): tensor of size K with point labels for
            ground truth points
        x_norm (:obj: `torch.Tensor`): tensor of size K with X normalized
            coordinates of ground truth points. Image X coordinates can be
            obtained as X = Xbbox + x_norm * Wbbox / 255
        y_norm (:obj: `torch.Tensor`): tensor of size K with Y normalized
            coordinates of ground truth points. Image Y coordinates can be
            obtained as Y = Ybbox + y_norm * Hbbox / 255
        index_bbox (:obj: `torch.Tensor`): tensor of size K with bounding box
            indices for each ground truth point. The values are thus in
            [0, N-1]

    Returns:
        j_valid (:obj: `torch.Tensor`): uint8 tensor of size M containing
            0 for points to be discarded and 1 for points to be selected
        y_lo (:obj: `torch.Tensor`): int tensor of indices of upper values
            in z_est for each point
        y_hi (:obj: `torch.Tensor`): int tensor of indices of lower values
            in z_est for each point
        x_lo (:obj: `torch.Tensor`): int tensor of indices of left values
            in z_est for each point
        x_hi (:obj: `torch.Tensor`): int tensor of indices of right values
            in z_est for each point
        w_ylo_xlo (:obj: `torch.Tensor`): float tensor of size M;
            contains upper-left value weight for each point
        w_ylo_xhi (:obj: `torch.Tensor`): float tensor of size M;
            contains upper-right value weight for each point
        w_yhi_xlo (:obj: `torch.Tensor`): float tensor of size M;
            contains lower-left value weight for each point
        w_yhi_xhi (:obj: `torch.Tensor`): float tensor of size M;
            contains lower-right value weight for each point
    """

    x0_gt, y0_gt, w_gt, h_gt = bbox_xywh_gt[index_bbox].unbind(dim=1)
    x0_est, y0_est, w_est, h_est = bbox_xywh_est[index_bbox].unbind(dim=1)
    x_lo, x_hi, x_w, jx_valid = _linear_interpolation_utilities(
        x_norm, x0_gt, w_gt, x0_est, w_est, zw
    )
    y_lo, y_hi, y_w, jy_valid = _linear_interpolation_utilities(
        y_norm, y0_gt, h_gt, y0_est, h_est, zh
    )
    j_valid = jx_valid * jy_valid

    w_ylo_xlo = (1.0 - x_w) * (1.0 - y_w)
    w_ylo_xhi = x_w * (1.0 - y_w)
    w_yhi_xlo = (1.0 - x_w) * y_w
    w_yhi_xhi = x_w * y_w

    return j_valid, y_lo, y_hi, x_lo, x_hi, w_ylo_xlo, w_ylo_xhi, w_yhi_xlo, w_yhi_xhi


def _extract_at_points_packed(
    z_est,
    index_bbox_valid,
    slice_index_uv,
    y_lo,
    y_hi,
    x_lo,
    x_hi,
    w_ylo_xlo,
    w_ylo_xhi,
    w_yhi_xlo,
    w_yhi_xhi,
):
    """
    Extract ground truth values z_gt for valid point indices and estimated
    values z_est using bilinear interpolation over top-left (y_lo, x_lo),
    top-right (y_lo, x_hi), bottom-left (y_hi, x_lo) and bottom-right
    (y_hi, x_hi) values in z_est with corresponding weights:
    w_ylo_xlo, w_ylo_xhi, w_yhi_xlo and w_yhi_xhi.
    Use slice_index_uv to slice dim=1 in z_est
    """
    z_est_sampled = (
        z_est[index_bbox_valid, slice_index_uv, y_lo, x_lo] * w_ylo_xlo
        + z_est[index_bbox_valid, slice_index_uv, y_lo, x_hi] * w_ylo_xhi
        + z_est[index_bbox_valid, slice_index_uv, y_hi, x_lo] * w_yhi_xlo
        + z_est[index_bbox_valid, slice_index_uv, y_hi, x_hi] * w_yhi_xhi
    )
    return z_est_sampled


def _resample_data(
    z, bbox_xywh_src, bbox_xywh_dst, wout, hout, mode="nearest", padding_mode="zeros"
):
    """
    Args:
        z (:obj: `torch.Tensor`): tensor of size (N,C,H,W) with data to be
            resampled
        bbox_xywh_src (:obj: `torch.Tensor`): tensor of size (N,4) containing
            source bounding boxes in format XYWH
        bbox_xywh_dst (:obj: `torch.Tensor`): tensor of size (N,4) containing
            destination bounding boxes in format XYWH
    Return:
        zresampled (:obj: `torch.Tensor`): tensor of size (N, C, Hout, Wout)
            with resampled values of z, where D is the discretization size
    """
    n = bbox_xywh_src.size(0)
    assert n == bbox_xywh_dst.size(0), (
        "The number of "
        "source ROIs for resampling ({}) should be equal to the number "
        "of destination ROIs ({})".format(bbox_xywh_src.size(0), bbox_xywh_dst.size(0))
    )
    x0src, y0src, wsrc, hsrc = bbox_xywh_src.unbind(dim=1)
    x0dst, y0dst, wdst, hdst = bbox_xywh_dst.unbind(dim=1)
    x0dst_norm = 2 * (x0dst - x0src) / wsrc - 1
    y0dst_norm = 2 * (y0dst - y0src) / hsrc - 1
    x1dst_norm = 2 * (x0dst + wdst - x0src) / wsrc - 1
    y1dst_norm = 2 * (y0dst + hdst - y0src) / hsrc - 1
    grid_w = torch.arange(wout, device=z.device, dtype=torch.float) / wout
    grid_h = torch.arange(hout, device=z.device, dtype=torch.float) / hout
    grid_w_expanded = grid_w[None, None, :].expand(n, hout, wout)
    grid_h_expanded = grid_h[None, :, None].expand(n, hout, wout)
    dx_expanded = (x1dst_norm - x0dst_norm)[:, None, None].expand(n, hout, wout)
    dy_expanded = (y1dst_norm - y0dst_norm)[:, None, None].expand(n, hout, wout)
    x0_expanded = x0dst_norm[:, None, None].expand(n, hout, wout)
    y0_expanded = y0dst_norm[:, None, None].expand(n, hout, wout)
    grid_x = grid_w_expanded * dx_expanded + x0_expanded
    grid_y = grid_h_expanded * dy_expanded + y0_expanded
    grid = torch.stack((grid_x, grid_y), dim=3)
    # resample Z from (N, C, H, W) into (N, C, Hout, Wout)
    zresampled = F.grid_sample(z, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return zresampled


def _extract_single_tensors_from_matches_one_image(
    proposals_targets, bbox_with_dp_offset, bbox_global_offset
):
    i_gt_all = []
    x_norm_all = []
    y_norm_all = []
    u_gt_all = []
    v_gt_all = []
    s_gt_all = []
    m_gt_all = []
    bbox_xywh_gt_all = []
    bbox_xywh_est_all = []
    # Ibbox_all == k should be true for all data that corresponds
    # to bbox_xywh_gt[k] and bbox_xywh_est[k]
    # index k here is global wrt images
    i_bbox_all = []
    # at offset k (k is global) contains index of bounding box data
    # within densepose output tensor
    i_with_dp = []

    boxes_xywh_est = proposals_targets.proposal_boxes.clone()
    boxes_xywh_gt = proposals_targets.gt_boxes.clone()
    n_i = len(boxes_xywh_est)
    assert n_i == len(boxes_xywh_gt)

    if n_i:
        boxes_xywh_est.tensor[:, 2] -= boxes_xywh_est.tensor[:, 0]
        boxes_xywh_est.tensor[:, 3] -= boxes_xywh_est.tensor[:, 1]
        boxes_xywh_gt.tensor[:, 2] -= boxes_xywh_gt.tensor[:, 0]
        boxes_xywh_gt.tensor[:, 3] -= boxes_xywh_gt.tensor[:, 1]
        if hasattr(proposals_targets, "gt_densepose"):
            densepose_gt = proposals_targets.gt_densepose
            for k, box_xywh_est, box_xywh_gt, dp_gt in zip(
                range(n_i), boxes_xywh_est.tensor, boxes_xywh_gt.tensor, densepose_gt
            ):
                if (dp_gt is not None) and (len(dp_gt.x) > 0):
                    i_gt_all.append(dp_gt.i)
                    x_norm_all.append(dp_gt.x)
                    y_norm_all.append(dp_gt.y)
                    u_gt_all.append(dp_gt.u)
                    v_gt_all.append(dp_gt.v)
                    s_gt_all.append(dp_gt.segm.unsqueeze(0))
                    #
                    m_gt = dp_gt.segm.clone()
                    m_gt[m_gt>0] = 1
                    m_gt_all.append(m_gt.unsqueeze(0))
                    #
                    bbox_xywh_gt_all.append(box_xywh_gt.view(-1, 4))
                    bbox_xywh_est_all.append(box_xywh_est.view(-1, 4))
                    i_bbox_k = torch.full_like(dp_gt.i, bbox_with_dp_offset + len(i_with_dp))
                    i_bbox_all.append(i_bbox_k)
                    i_with_dp.append(bbox_global_offset + k)
    return (
        i_gt_all,
        x_norm_all,
        y_norm_all,
        u_gt_all,
        v_gt_all,
        s_gt_all,
        m_gt_all,
        bbox_xywh_gt_all,
        bbox_xywh_est_all,
        i_bbox_all,
        i_with_dp,
    )


def _extract_single_tensors_from_matches(proposals_with_targets):
    i_img = []
    i_gt_all = []
    x_norm_all = []
    y_norm_all = []
    u_gt_all = []
    v_gt_all = []
    s_gt_all = []
    m_gt_all = []
    bbox_xywh_gt_all = []
    bbox_xywh_est_all = []
    i_bbox_all = []
    i_with_dp_all = []
    n = 0
    for i, proposals_targets_per_image in enumerate(proposals_with_targets):
        n_i = proposals_targets_per_image.proposal_boxes.tensor.size(0)
        if not n_i:
            continue
        i_gt_img, x_norm_img, y_norm_img, u_gt_img, v_gt_img, s_gt_img, m_gt_img, bbox_xywh_gt_img, bbox_xywh_est_img, i_bbox_img, i_with_dp_img = _extract_single_tensors_from_matches_one_image(  # noqa
            proposals_targets_per_image, len(i_with_dp_all), n
        )
        i_gt_all.extend(i_gt_img)
        x_norm_all.extend(x_norm_img)
        y_norm_all.extend(y_norm_img)
        u_gt_all.extend(u_gt_img)
        v_gt_all.extend(v_gt_img)
        s_gt_all.extend(s_gt_img)
        m_gt_all.extend(m_gt_img)
        bbox_xywh_gt_all.extend(bbox_xywh_gt_img)
        bbox_xywh_est_all.extend(bbox_xywh_est_img)
        i_bbox_all.extend(i_bbox_img)
        i_with_dp_all.extend(i_with_dp_img)
        i_img.extend([i] * len(i_with_dp_img))
        n += n_i
    # concatenate all data into a single tensor
    if (n > 0) and (len(i_with_dp_all) > 0):
        i_gt = torch.cat(i_gt_all, 0).long()
        x_norm = torch.cat(x_norm_all, 0)
        y_norm = torch.cat(y_norm_all, 0)
        u_gt = torch.cat(u_gt_all, 0)
        v_gt = torch.cat(v_gt_all, 0)
        s_gt = torch.cat(s_gt_all, 0)
        m_gt = torch.cat(m_gt_all, 0)
        bbox_xywh_gt = torch.cat(bbox_xywh_gt_all, 0)
        bbox_xywh_est = torch.cat(bbox_xywh_est_all, 0)
        i_bbox = torch.cat(i_bbox_all, 0).long()
    else:
        i_gt = None
        x_norm = None
        y_norm = None
        u_gt = None
        v_gt = None
        s_gt = None
        m_gt = None
        bbox_xywh_gt = None
        bbox_xywh_est = None
        i_bbox = None
    return (
        i_img,
        i_with_dp_all,
        bbox_xywh_est,
        bbox_xywh_gt,
        i_gt,
        x_norm,
        y_norm,
        u_gt,
        v_gt,
        s_gt,
        m_gt,
        i_bbox,
    )


class DensePoseLosses(object):
    def __init__(self, cfg):
        # fmt: off
        self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        self.w_points     = cfg.MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS
        self.w_part       = cfg.MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS
        self.w_segm       = cfg.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS
        self.w_mask       = cfg.MODEL.ROI_DENSEPOSE_HEAD.BODY_MASK_WEIGHTS
        print('dp loss weight -> UV:%f, UV_index:%f, Part:%f, Mask:%f'%(self.w_points, self.w_part, self.w_segm, self.w_mask))
        # fmt: on

    def __call__(self, proposals_with_gt, densepose_outputs, prefix='', cls_emb_loss_on=False):
        losses = {}

        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7
        s, index_uv, u, v, m = densepose_outputs
        assert u.size(2) == v.size(2)
        assert u.size(3) == v.size(3)
        assert u.size(2) == index_uv.size(2)
        assert u.size(3) == index_uv.size(3)
        # print('UV size:', u.size(), v.size(), index_uv.size(), m.size())
        with torch.no_grad():
            index_uv_img, i_with_dp, bbox_xywh_est, bbox_xywh_gt, index_gt_all, x_norm, y_norm, u_gt_all, v_gt_all, s_gt, m_gt, index_bbox = _extract_single_tensors_from_matches(  # noqa
                proposals_with_gt
            )
        n_batch = len(i_with_dp)

        # NOTE: we need to keep the same computation graph on all the GPUs to
        # perform reduction properly. Hence even if we have no data on one
        # of the GPUs, we still need to generate the computation graph.
        # Add fake (zero) loss in the form Tensor.sum() * 0
        if not n_batch:
            losses[prefix+"loss_densepose_U"] = u.sum() * 0
            losses[prefix+"loss_densepose_V"] = v.sum() * 0
            losses[prefix+"loss_densepose_I"] = index_uv.sum() * 0
            if s is not None:
                losses[prefix+"loss_densepose_S"] = s.sum() * 0
            if m is not None:
                losses[prefix+"loss_densepose_M"] = m.sum() * 0
            if cls_emb_loss_on:
                losses[prefix + 'loss_push'] = m.sum() * 0
                losses[prefix + 'loss_pull'] = m.sum() * 0
            return losses

        zh = u.size(2)
        zw = u.size(3)

        j_valid, y_lo, y_hi, x_lo, x_hi, w_ylo_xlo, w_ylo_xhi, w_yhi_xlo, w_yhi_xhi = _grid_sampling_utilities(  # noqa
            zh, zw, bbox_xywh_est, bbox_xywh_gt, index_gt_all, x_norm, y_norm, index_bbox
        )

        j_valid_fg = j_valid * (index_gt_all > 0)

        u_gt = u_gt_all[j_valid_fg]
        u_est_all = _extract_at_points_packed(
            u[i_with_dp],
            index_bbox,
            index_gt_all,
            y_lo,
            y_hi,
            x_lo,
            x_hi,
            w_ylo_xlo,
            w_ylo_xhi,
            w_yhi_xlo,
            w_yhi_xhi,
        )
        u_est = u_est_all[j_valid_fg]

        v_gt = v_gt_all[j_valid_fg]
        v_est_all = _extract_at_points_packed(
            v[i_with_dp],
            index_bbox,
            index_gt_all,
            y_lo,
            y_hi,
            x_lo,
            x_hi,
            w_ylo_xlo,
            w_ylo_xhi,
            w_yhi_xlo,
            w_yhi_xhi,
        )
        v_est = v_est_all[j_valid_fg]

        index_uv_gt = index_gt_all[j_valid]
        index_uv_est_all = _extract_at_points_packed(
            index_uv[i_with_dp],
            index_bbox,
            slice(None),
            y_lo,
            y_hi,
            x_lo,
            x_hi,
            w_ylo_xlo[:, None],
            w_ylo_xhi[:, None],
            w_yhi_xlo[:, None],
            w_yhi_xhi[:, None],
        )
        index_uv_est = index_uv_est_all[j_valid, :]

        # Resample everything to the estimated data size, no need to resample
        # S_est then:
        if s is not None:
            s_est = s[i_with_dp]

        with torch.no_grad():
            s_gt = _resample_data(
                s_gt.unsqueeze(1),
                bbox_xywh_gt,
                bbox_xywh_est,
                self.heatmap_size,
                self.heatmap_size,
                mode="nearest",
                padding_mode="zeros",
            ).squeeze(1)
        # M_est then
        if m is not None:
            m_est = m[i_with_dp]
        m_gt = s_gt.clamp(min=0, max=1)
        # print('m_gt size:',m_gt.size())

        # add point-based losses:
        u_loss = F.smooth_l1_loss(u_est, u_gt, reduction="sum") * self.w_points
        losses[prefix+"loss_densepose_U"] = u_loss
        v_loss = F.smooth_l1_loss(v_est, v_gt, reduction="sum") * self.w_points
        losses[prefix+"loss_densepose_V"] = v_loss
        index_uv_loss = F.cross_entropy(index_uv_est, index_uv_gt.long()) * self.w_part
        losses[prefix+"loss_densepose_I"] = index_uv_loss

        if s is not None:
            s_loss = F.cross_entropy(s_est, s_gt.long()) * self.w_segm
            losses[prefix+"loss_densepose_S"] = s_loss
        if m is not None:
            m_loss = F.cross_entropy(m_est, m_gt.long()) * self.w_mask
            losses[prefix+"loss_densepose_M"] = m_loss
        if cls_emb_loss_on:
            emb_loss = class_emb_losses(m_est, m_gt)
            losses[prefix + 'loss_push'], losses[prefix + 'loss_pull'] = emb_loss
        return losses


def class_emb_losses(mask_emb, gt_labels):
    keep = (torch.sum(gt_labels, (1, 2)) > 0)
    keep_mask_emb = mask_emb[keep, :, :, :]
    keep_labels = gt_labels[keep, :, :]
    if keep_mask_emb.size(0) == 0 or torch.sum(keep).cpu().numpy() < 1:
        return 0
    keep_labels = keep_labels.clamp(min=0, max=1)
    fg_emb = keep_mask_emb * (keep_labels == 1).unsqueeze(1)
    fg_ref_emb = torch.sum(fg_emb,(2,3)) / torch.sum((keep_labels==1),(1,2)).unsqueeze(1)
    bg_emb = keep_mask_emb * (keep_labels == 0).unsqueeze(1)
    bg_ref_emb = torch.sum(bg_emb, (2, 3)) / torch.sum((keep_labels == 0), (1, 2)).unsqueeze(1)
    ref_embs = torch.cat([fg_ref_emb, bg_ref_emb], 0)
    labels = torch.cat([torch.ones((fg_ref_emb.size(0),)), torch.zeros(bg_ref_emb.size(0),)], 0)
    labels = labels.cuda(non_blocking=True)
    emb_loss = F.cross_entropy(ref_embs, labels.long()) * 0.5

    # fg push & aggregation
    '''
    fg_dist = (keep_mask_emb - fg_ref_emb.reshape((fg_ref_emb.size(0), fg_ref_emb.size(1), 1, 1))) ** 2
    fg_dist = torch.pow(torch.sum(fg_dist, 1), 0.5)
    fg_dist_logits = 2 / (1 + torch.exp(fg_dist))
    fg_dist_logits = fg_dist_logits.clamp(min=0, max=1)
    fg_dist_loss = F.binary_cross_entropy(fg_dist_logits.reshape((-1)), keep_labels.reshape((-1)).float())
    # bg push & aggregation
    bg_dist = (keep_mask_emb - bg_ref_emb.reshape((bg_ref_emb.size(0), bg_ref_emb.size(1), 1, 1))) ** 2
    bg_dist = torch.pow(torch.sum(bg_dist, 1), 0.5)
    bg_dist_logits = 2 / (1 + torch.exp(bg_dist))
    bg_dist_logits = bg_dist_logits.clamp(min=0, max=1)
    bg_dist_loss = F.binary_cross_entropy(bg_dist_logits.reshape((-1)), (1-keep_labels).reshape((-1)).float())
    # fg_dist = (keep_mask_emb - fg_ref_emb.reshape(
    #     (fg_ref_emb.size(0), fg_ref_emb.size(1), 1, 1))) ** 2 * keep_labels.unsqueeze(1)
    # mean_fg_dist = torch.sum(fg_dist, (2, 3)) / torch.sum((keep_labels == 1), (1, 2)).unsqueeze(1)
    # bg_dist = (keep_mask_emb - bg_ref_emb.reshape(
    #     (bg_ref_emb.size(0), bg_ref_emb.size(1), 1, 1))) ** 2 * (1-keep_labels).unsqueeze(1)
    # mean_bg_dist = torch.sum(bg_dist, (2, 3)) / torch.sum((keep_labels == 0), (1, 2)).unsqueeze(1)
    '''
    return emb_loss, 0.*labels.sum()#(fg_dist_loss + bg_dist_loss)

class DensePoseInterLosses(object):
    def __init__(self, cfg):
        # fmt: off
        self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        self.w_points     = cfg.MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS
        self.w_part       = cfg.MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS
        self.w_segm       = cfg.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS

    def __call__(self, proposals_with_gt, densepose_outputs, prefix='inter'):
        losses = {}
        m = densepose_outputs
        with torch.no_grad():
            index_uv_img, i_with_dp, bbox_xywh_est, bbox_xywh_gt, index_gt_all, x_norm, y_norm, u_gt_all, v_gt_all, s_gt, m_gt, index_bbox = _extract_single_tensors_from_matches(  # noqa
                proposals_with_gt
            )
        n_batch = len(i_with_dp)

        if not n_batch:
            losses[prefix + "loss_densepose_M"] = m.sum() * 0
            return losses

        # M_est then
        with torch.no_grad():
            s_gt = _resample_data(
                s_gt.unsqueeze(1),
                bbox_xywh_gt,
                bbox_xywh_est,
                self.heatmap_size,
                self.heatmap_size,
                mode="nearest",
                padding_mode="zeros",
            ).squeeze(1)
        m_est = m[i_with_dp]
        m_gt = s_gt.clamp(min=0, max=1)

        m_loss = F.cross_entropy(m_est, m_gt.long()) * self.w_segm
        losses[prefix+"loss_densepose_M"] = m_loss
        return losses

def dp_keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer):

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
        keypoint_targets = torch.cat(heatmaps, dim=0)
        valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:

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

def kpts_to_maps(kpts):
    side_len = 56
    import numpy as np
    map = np.zeros((side_len,side_len))
    for kp in kpts:
        if kp == 0:
            continue
        x = kp % side_len
        y = kp // side_len
        map[y, x] = 1
        map[y - 1, x] = 1
        map[y + 1, x] = 1
        map[y, x - 1] = 1
        map[y, x + 1] = 1
    return map
def build_densepose_losses(cfg):
    losses = DensePoseLosses(cfg)
    return losses

def build_densepose_inter_losses(cfg):
    losses = DensePoseInterLosses(cfg)
    return losses
