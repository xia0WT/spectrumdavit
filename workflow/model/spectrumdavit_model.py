#v1.1 with dropblock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Optional, Tuple

from nwtk.norm import Norm1d
from nwtk.convposenc import ConvPosEnc1d
from nwtk.attention import ChannelAttention
from nwtk.drop import DropPath, DropBlock1d
from nwtk.mlp import Mlp
from nwtk.helpers import to_2tuple
from nwtk.downsample import DownSample1d
from nwtk._manipulate import checkpoint_seq

# dim must be divided by num_heads, dim also means channels.
class ChannelBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop_path=0.,
        drop_prob=0.,
        drop_func=nn.Dropout,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        ffn=True,
        cpe_act=False,
        ):
        super().__init__()
            
        self.cpe1 = ConvPosEnc1d(dim=dim, k=3, act=cpe_act)
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        attn_layer = ChannelAttention
        self.attn = attn_layer(
        dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = ConvPosEnc1d(dim=dim, k=3, act=cpe_act)

        if self.ffn:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop_prob=drop_prob,
                drop_func = drop_func,
            )
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        else:
            self.norm2 = None
            self.mlp = None
            self.drop_path2 = None
    def forward(self, x):
        B, C, N = x.shape

        x = self.cpe1(x).transpose(1, 2)

        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path1(cur)

        x = self.cpe2(x.transpose(1, 2).view(B, C, N))

        if self.mlp is not None:
            x = x.transpose(1, 2)
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
            x = x.transpose(1, 2).view(B, C, N)

        return x

class Stem(nn.Module):

    def __init__(
            self,
            in_chs=3,
            out_chs=96,
            stride=4,
            norm_layer=Norm1d,
    ):
        super().__init__()

        self.stride = stride
        self.in_chs = in_chs
        self.out_chs = out_chs
        assert stride == 4  # only setup for stride==4
        self.conv = nn.Conv1d(
            in_chs,
            out_chs,
            kernel_size=7,
            stride=stride,
            padding=3,
        )
        self.norm = norm_layer(out_chs)

    def forward(self, x: Tensor):
        B, C, N = x.shape
        pad_r = (self.stride - N % self.stride) % self.stride
        x = F.pad(x, (0, pad_r))
        x = self.conv(x)
        x = self.norm(x)
        return x

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, use_fused_attn=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = self.softmax(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

def window_partition(x: Tensor, window_size: int):

    B, N, C = x.shape
    x = x.view(B, N // window_size, window_size, C)
    windows = x.view(-1, window_size, C)
    return windows

def window_reverse(windows: Tensor, window_size: int, N: int):

    C = windows.shape[-1]
    x = windows.view(-1, N // window_size, window_size, C)
    x = x.view(-1, N, C)
    return x

class SelectAdaptivePool1d(nn.Module):
    def __init__(            
            self,
            output_size = 1,
            flatten: bool = False,
            input_fmt: str = 'NCHW',
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        
    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return 1
        
def _create_pool(
        num_features: int,
        use_conv: bool = False,
):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    global_pool = SelectAdaptivePool1d(
        flatten=flatten_in_pool,
    )
    num_pooled_features = num_features * global_pool.feat_mult()
    
    return global_pool, num_pooled_features


def _create_fc(num_features, num_classes: int, use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv1d(num_features, num_classes, 1, bias=True)
    else:
        fc = nn.Linear(num_features, num_classes, bias=True)
    return fc


def create_classifier(
        num_features: int,
        num_classes: int,
        use_conv: bool = False,
        drop_rate: Optional[float] = None,
):
    global_pool, num_pooled_features = _create_pool(
        num_features,
        use_conv=use_conv,
    )
    fc = _create_fc(
        num_pooled_features,
        num_classes,
        use_conv=use_conv,
    )
    if drop_rate is not None:
        dropout = nn.Dropout(drop_rate)
        return global_pool, dropout, fc
    return global_pool, fc


class ClassifierHead(nn.Module):
    def __init__(
            self,
            in_features: int,
            num_classes: int,
            drop_rate: float = 0.,
            use_conv: bool = False,
    ):

        super(ClassifierHead, self).__init__()
        self.in_features = in_features
        self.use_conv = use_conv

        global_pool, fc = create_classifier(
            in_features,
            num_classes,
            use_conv=use_conv,
            drop_rate = None,
        )
        self.global_pool = global_pool
        self.drop = nn.Dropout(drop_rate)
        self.fc = fc
        self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def reset(self, num_classes: int, pool_type: Optional[str] = None):
        if pool_type is not None and pool_type != self.global_pool.pool_type:
            self.global_pool, self.fc = create_classifier(
                self.in_features,
                num_classes,
                #pool_type=pool_type, #TODO
                use_conv=self.use_conv,
                #input_fmt=self.input_fmt, #TODO
            )
            self.flatten = nn.Flatten(1) if self.use_conv and pool_type else nn.Identity()
        else:
            num_pooled_features = self.in_features * self.global_pool.feat_mult()
            self.fc = _create_fc(
                num_pooled_features,
                num_classes,
                use_conv=self.use_conv,
            )

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.drop(x)
        if pre_logits:
            return self.flatten(x)
            
        x = self.fc(x)
        return self.flatten(x)

class SpatialBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            window_size=4,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_path=0.,
            drop_prob=0.,
            drop_func = nn.Dropout,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            ffn=True,
            cpe_act=False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.cpe1 = ConvPosEnc1d(dim=dim, k=3, act=cpe_act)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = ConvPosEnc1d(dim=dim, k=3, act=cpe_act)
        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_prob=drop_prob,
                drop_func=drop_func,
            )
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        else:
            self.norm2 = None
            self.mlp = None
            self.drop_path1 = None

    def forward(self, x):

        B, C, N = x.shape


        shortcut = self.cpe1(x).transpose(1, 2)

        x = self.norm1(shortcut)
        x = x.view(B, N, C)

        p = (self.window_size - N % self.window_size) #% self.window_size
        
        x = F.pad(x, (0, 0, 0, p))
        _, Np, _ = x.shape

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size, C)
        
        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Np)

        x = x[:, :N, :].contiguous()

        x = x.view(B, N, C)
        x = shortcut + self.drop_path1(x)

        x = self.cpe2(x.transpose(1, 2).view(B, C, N))

        if self.mlp is not None:
            x = x.transpose(1, 2)
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
            x = x.transpose(1, 2).view(B, C, N)

        return x

class DaVitStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            depth=1,
            downsample=True,
            attn_types=('spatial', 'channel'),
            num_heads=8,
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_path_rates=(0, 0),
            drop_prob=0.,
            drop_func=nn.Dropout,
            norm_layer=Norm1d,
            norm_layer_cl=nn.LayerNorm,
            ffn=True,
            cpe_act=False,
            down_kernel_size=2,
            named_blocks=False,
    ):
        super().__init__()

        self.grad_checkpointing = False

        # downsample embedding layer at the beginning of each stage
        if downsample:
            self.downsample = DownSample1d(in_chs, out_chs, kernel_size=down_kernel_size,)
        else:
            self.downsample = nn.Identity()

        stage_blocks = []
        for block_idx in range(depth):
            from collections import OrderedDict
            dual_attention_block = []
            for attn_idx, attn_type in enumerate(attn_types):
                if attn_type == 'spatial':
                    dual_attention_block.append(('spatial_block', SpatialBlock(
                        dim=out_chs,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=drop_path_rates[block_idx],
                        drop_prob=drop_prob,
                        drop_func=drop_func,
                        norm_layer=norm_layer_cl,
                        ffn=ffn,
                        cpe_act=cpe_act,
                        window_size=window_size,
                    )))
                elif attn_type == 'channel':
                    dual_attention_block.append(('channel_block', ChannelBlock(
                        dim=out_chs,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=drop_path_rates[block_idx],
                        drop_prob=drop_prob,
                        drop_func=drop_func,
                        norm_layer=norm_layer_cl,
                        ffn=ffn,
                        cpe_act=cpe_act,
                    )))
            if named_blocks:
                stage_blocks.append(nn.Sequential(OrderedDict(dual_attention_block)))
            else:
                stage_blocks.append(nn.Sequential(*[b[1] for b in dual_attention_block]))
        self.blocks = nn.Sequential(*stage_blocks)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, x: Tensor):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x

class SpectrumDaVit(nn.Module):

    def __init__(
            self,
            in_chans=3,
            depths=(1, 1, 3, 1),
            embed_dims=(96, 192, 384, 768),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=Norm1d,
            norm_layer_cl=nn.LayerNorm,
            norm_eps=1e-5,
            attn_types=('spatial', 'channel'),
            ffn=True,
            cpe_act=True,
            down_kernel_size=2,
            named_blocks=False,
            drop_rate=0.,
            drop_func=nn.Dropout,
            drop_path_rate=0.,
            num_classes=100,
            global_pool='avg',
            head_norm_first=True,
    ):
        super().__init__()
        num_stages = len(embed_dims)
        assert num_stages == len(num_heads) == len(depths)
        #norm_layer = partial(get_norm_layer(norm_layer), eps=norm_eps)
        #norm_layer_cl = partial(get_norm_layer(norm_layer_cl), eps=norm_eps)
        self.num_classes = num_classes
        self.num_features = self.head_hidden_size = embed_dims[-1]
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []

        self.stem = Stem(in_chans, embed_dims[0], norm_layer=norm_layer)
        in_chs = embed_dims[0]

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        for stage_idx in range(num_stages):
            out_chs = embed_dims[stage_idx]
            stage = DaVitStage(
                in_chs,
                out_chs,
                depth=depths[stage_idx],
                downsample=stage_idx > 0,
                attn_types=attn_types,
                num_heads=num_heads[stage_idx],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path_rates=dpr[stage_idx],
                drop_prob=drop_rate,
                drop_func=drop_func,
                norm_layer=norm_layer,
                norm_layer_cl=norm_layer_cl,
                ffn=ffn,
                cpe_act=cpe_act,
                down_kernel_size=down_kernel_size,
                named_blocks=named_blocks,
            )
            in_chs = out_chs
            stages.append(stage)
            self.feature_info += [dict(num_chs=out_chs, reduction=2, module=f'stages.{stage_idx}')]

        self.stages = nn.Sequential(*stages)

        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default DaViT order, similar to ConvNeXt
        # FIXME generalize this structure to ClassifierHead
        
        self.norm_pre = norm_layer(self.num_features) if head_norm_first else nn.Identity()
        self.head = ClassifierHead(
                    self.num_features,
                    num_classes,
                    #pool_type=global_pool,
                    drop_rate=self.drop_rate,
                    )


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
        for stage in self.stages:
            stage.set_grad_checkpointing(enable=enable)

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
