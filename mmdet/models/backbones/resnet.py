import logging

import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mmdet.ops import DeformConv, ModulatedDeformConv,ContextBlock2d,TemporalContextBlock,TemporalContextBlockshort,TemporalContextBlockshort_max,Correspondence
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 ct=None,
                 tc=None,
                 tcs=None,
                 tcsm=None,
                 video=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, "Not implemented yet."
        assert ct is None, "Not implemented yet"
        assert tc is None, "Not implemented yet"
        assert tcs is None, "Not implemented yet"
        assert tcsm is None, "Not implemented yet"
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 ct=None,
                 tc=None,
                 tcs=None,
                 tcsm=None,
                 video=False
                 ):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        self.inplanes = inplanes
        self.planes = planes
        self.video = video
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.ct = ct
        self.with_ct = ct is not None
        self.tc = tc
        self.with_tc = tc is not None
        self.tcs = tcs
        self.with_tcs = tcs is not None
        self.tcsm = tcsm
        self.with_tcsm = tcsm is not None
        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert conv_cfg is None, 'conv_cfg must be None for DCN'
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False)
        if self.with_ct:
            self.insert_pos = ct.get('insert_pos', 'after_add')
            assert self.insert_pos in ['after_1x1', 'after_3x3', 'after_add']
            if self.insert_pos == 'after_3x3':
                ct_inplanes = planes
            else:
                ct_inplanes = planes * self.expansion
            ct_ratio = ct.get('ratio', None)
            if ct_ratio is not None:
                ct_planes = int(ct_inplanes * ct_ratio)
            else:
                ct_planes = None
            pool = ct.get('pool', 'att')
            fusions = ct.get('fusions', 'channel_add')
            fusions = fusions if isinstance(fusions, list) else [fusions]
            self.context_block = ContextBlock2d(
                inplanes=ct_inplanes,
                planes=ct_planes,
                pool=pool,
                fusions=fusions,
            )
        if self.with_tc:
            self.tc_pos = tc.get('insert_pos', 'after_add')
            assert self.tc_pos in ['after_1x1', 'after_3x3', 'after_add']
            if self.tc_pos == 'after_3x3':
                tc_inplanes = planes
            else:
                tc_inplanes = planes * self.expansion

            self.share = tc.get('is_position_encoding', True)
            self.window_size=tc.get('window_size',5)
            if self.share:
                assert self.window_size in [3,5]
            self.snip_size=tc.get('snip_size',8)
            self.detach=tc.get('detach', False)
            self.local_mean=tc.get('local_mean',True)
            self.temporal_context_block = TemporalContextBlock(
                inplanes=tc_inplanes,
                snip_size=self.snip_size,
                window_size=self.window_size,
                detach= self.detach,
                is_position_encoding= self.share,
                local_mean=self.local_mean,
                repeat_mode=tc.get('repeat_mode',False),
                reduce=tc.get('reduce',False)
            )
        if self.with_tcs:
            self.tcs_pos = tcs.get('insert_pos', 'after_add')
            assert self.tcs_pos in ['after_1x1', 'after_3x3', 'after_add']
            if self.tcs_pos == 'after_3x3':
                tcs_inplanes = planes
            else:
                tcs_inplanes = planes * self.expansion
            self.snip_size=tcs.get('snip_size',8)
            _, self.tcs_norm = build_norm_layer(norm_cfg, tcs_inplanes, postfix='tcs')
            self.temporal_context_block_short = TemporalContextBlockshort(
                inplanes=tcs_inplanes,
                snip_size= self.snip_size,
                norm=self.tcs_norm
            )
        if self.with_tcsm:
            self.tcsm_pos = tcsm.get('insert_pos', 'after_add')
            assert self.tcsm_pos in ['after_1x1', 'after_3x3', 'after_add']
            if self.tcsm_pos == 'after_3x3':
                tcsm_inplanes = planes
            else:
                tcsm_inplanes = planes * self.expansion
            self.snip_size=tcsm.get('snip_size',8)
            _, self.tcsm_norm = build_norm_layer(norm_cfg, tcsm_inplanes, postfix='tcsm')
            self.temporal_context_block_short_max = TemporalContextBlockshort_max(
                inplanes=tcsm_inplanes,
                snip_size= self.snip_size,
                norm=self.tcsm_norm
            )

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            if self.video:
                x,snip=x
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if not self.with_dcn:
                out = self.conv2(out)
            elif self.with_modulated_dcn:
                offset_mask = self.conv2_offset(out)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = self.conv2_offset(out)
                out = self.conv2(out, offset)
            out = self.norm2(out)
            out = self.relu(out)
            if self.with_tc and self.tc_pos == 'after_3x3':
                out = self.temporal_context_block(out,snip)
            if self.with_tcs and self.tcs_pos == 'after_3x3':
                out = self.temporal_context_block_short(out)
            if self.with_tcsm and self.tcsm_pos == 'after_3x3':
                out = self.temporal_context_block_short_max(out,snip)
            if self.with_ct and self.insert_pos == 'after_3x3':
                out = self.context_block(out)
            out = self.conv3(out)
            out = self.norm3(out)
            if self.with_tc and self.tc_pos == 'after_1x1':
                out = self.temporal_context_block(out,snip)
            if self.with_tcs and self.tcs_pos == 'after_1x1':
                out = self.temporal_context_block_short(out)
            if self.with_tcsm and self.tcsm_pos == 'after_1x1':
                out = self.temporal_context_block_short_max(out,snip)
            if self.with_ct and self.insert_pos == 'after_1x1':
                out = self.context_block(out)
            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            if self.with_tc and self.tc_pos == 'after_add':
                out = self.temporal_context_block(out,snip)
            if self.with_tcs and self.tcs_pos == 'after_add':
                out = self.temporal_context_block_short(out)
            if self.with_tcsm and self.tcsm_pos == 'after_add':
                out = self.temporal_context_block_short_max(out,snip)
            if self.with_ct and self.insert_pos == 'after_add':
                out = self.context_block(out)
            out = self.relu(out)
            if self.video:
                out =(out,snip)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)


        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   video=False,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   dcn=None,
                   ct=None,
                   tc=None,
                   tcs=None,
                   tcsm=None):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1],
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            ct=ct,
            tc=tc,
            tcs=tcs,
            tcsm=tcsm,
            video=video))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                1,
                dilation,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                ct=ct,
                tc=tc,
                tcs=tcs,
                tcsm=tcsm,
                video=video))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 ct=None,
                 stage_with_ct=(False,False,False,False),
                 tc=None,
                 stage_with_tc=(False,False,False,False),
                 tcs=None,
                 stage_with_tcs=(False,False,False,False),
                 tcsm=None,
                 stage_with_tcsm=(False,False,False,False),
                 cp=None,
                 stage_with_cp=(False,False,False,False),
                 with_cp=False,
                 zero_init_residual=True,
                 video=False,
                 train_reduce=False,
                 test_reduce=False):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.ct = ct
        self.stage_with_ct = stage_with_ct
        if ct is not None:
            assert len(stage_with_ct) == num_stages
        self.tc = tc
        self.stage_with_tc = stage_with_tc
        if tc is not None:
            assert len(stage_with_tc) == num_stages
        self.tcs = tcs
        self.stage_with_tcs = stage_with_tcs
        if tcs is not None:
            assert len(stage_with_tcs) == num_stages
        self.tcsm = tcsm
        self.stage_with_tcsm = stage_with_tcsm
        if tcsm is not None:
            assert len(stage_with_tcsm) == num_stages
        self.cp=cp
        self.cp_stage=[]
        if cp is not None:
            assert len(stage_with_cp) == num_stages
            #init cp layer
            self.cp_stage=[ i for i in range(len(stage_with_cp)) if stage_with_cp[i] ]
            cp_mode=cp.get('mode', 'max')
            cp_topk=cp.get('topk','5')
            if stage_with_cp[0]:
                inplanes=256
                _, cp_norm0 = build_norm_layer(self.norm_cfg, inplanes, postfix='cp_norm0')
                self.correspondence_0=Correspondence(inplanes,cp_norm0,cp_topk,cp_mode)
            if stage_with_cp[1]:
                inplanes=512
                _, cp_norm1 = build_norm_layer(self.norm_cfg, inplanes, postfix='cp_norm1')
                self.correspondence_1=Correspondence(inplanes,cp_norm1,cp_topk,cp_mode)
            if stage_with_cp[2]:
                inplanes=1024
                _, cp_norm2 = build_norm_layer(self.norm_cfg, inplanes, postfix='cp_norm2')
                self.correspondence_2=Correspondence(inplanes,cp_norm2,cp_topk,cp_mode)
            if stage_with_cp[3]:
                inplanes=2048
                _, cp_norm3 = build_norm_layer(self.norm_cfg, inplanes, postfix='cp_norm3')
                self.correspondence_3=Correspondence(inplanes,cp_norm3,cp_topk,cp_mode)

        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64
        self.video=video
        self.train_reduce=train_reduce
        self.test_reduce=test_reduce

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            ct = self.ct if self.stage_with_ct[i] else None
            tc = self.tc if self.stage_with_tc[i] else None
            tcs = self.tcs if self.stage_with_tcs[i] else None
            tcsm = self.tcsm if self.stage_with_tcsm[i] else None
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                ct=ct,
                tc=tc,
                tcs=tcs,
                tcsm=tcsm,
                video=video)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')
    def reduce(self,out,snip_size):
        b,c,w,h=out.size()
        n=snip_size//2
        out=out.view(b//snip_size,snip_size,c,w,h)
        return out[:,n,...]

    def forward(self, x):
        if self.video:
            batch_size, snip_size, c_size, w_size, h_size=x.size()
            x=x.view(batch_size*snip_size,c_size,w_size,h_size)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.video:
            x=(x,snip_size)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            #for cp
            if i in self.cp_stage:
                #print('cp_stage',i)
                x=getattr(self,'correspondence_{}'.format(i))(x[0],x[1])
                x=(x,snip_size)
            if i in self.out_indices:
                if self.video:
                    outs.append(x[0])
                else:
                    outs.append(x)
        if self.video and self.train_reduce and self.training:
            outs=list(map(self.reduce,outs,[snip_size]*len(outs)))
        if self.video and self.test_reduce and not self.training:
            outs=list(map(self.reduce,outs,[snip_size]*len(outs)))
        return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
