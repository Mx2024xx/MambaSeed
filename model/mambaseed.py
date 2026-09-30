#!/usr/bin/env python3

# Copyright Notice: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation,
# and any modifications thereto. Any use, reproduction, disclosure, or
# distribution of this software and related documentation without an
# express license agreement from NVIDIA CORPORATION is strictly prohibited.

# -----------------------------------------------------------------------------
# Attribution and License Notice for Academic Use
#
# This file is modified from code associated with NVIDIA's MambaVision project.
# Source project: https://github.com/NVlabs/MambaVision
#
# Please review and comply with the upstream license terms before any use,
# reproduction, distribution, or release of derivative works.
# In this workspace, the upstream project license is distributed as:
#   - /root/autodl-tmp/MambaVision/LICENSE
#
# Intended use for this modified file is non-commercial academic research.
# If your use case includes commercial use or public redistribution, obtain
# the necessary permissions and legal approval in advance.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg
try:
    from timm.models._builder import _update_default_kwargs as update_args
except Exception:  # noqa: E722
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

# In the original MV1/mambavision environment, this file used relative imports
# as an in-package module. It is now placed standalone under MambaSeed_224/model
# without registry.py. Training scripts only rely on timm's register_model,
# so register_pip_model is treated as optional for compatibility.
try:
    from .registry import register_pip_model  # type: ignore
except Exception:  # noqa: E722
    def register_pip_model(fn):
        """No-op decorator to keep backward compatibility when registry is absent."""
        return fn
from pathlib import Path
def _cfg(url='', **kwargs):
    """
    Helper function to build model config.

    Args:
        url: URL of pretrained model weights.
        **kwargs: Other config options.

    Returns:
        A dictionary containing model configuration.
    """
    return {'url': url,
            'num_classes': 10,             # Number of classes
            'input_size': (3, 224, 224),   # Input image size (C, H, W)
            'pool_size': None,             # Pooling layer size
            'crop_pct': 0.875,             # Crop percentage
            'interpolation': 'bicubic',    # Interpolation method
            'fixed_input_size': True,      # Whether input size is fixed
            'mean': (0.485, 0.456, 0.406), # Normalization mean
            'std': (0.229, 0.224, 0.225),  # Normalization std
            **kwargs
            }


default_cfgs = {
    'mamba_vision_deformable_T': _cfg(url='',
                                      crop_pct=1.0,
                                      input_size=(3, 224, 224),
                                      crop_mode='center'),
}


def _load_state_dict(module, state_dict, strict=False, logger=None):
    """
    Load a state dict into a module.

    This method is adapted from :meth:`torch.nn.Module.load_state_dict`.
    By default, ``strict`` is set to ``False`` and mismatch messages are still shown.

    Args:
        module: Module that receives the state dict.
        state_dict: Weight dictionary.
        strict: Whether to strictly enforce key matching. Default: ``False``.
        logger: Optional logger for warnings.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('Unexpected keys in source state_dict: ' + ", ".join(unexpected_keys))
    if missing_keys:
        err_msg.append('Missing keys in source state_dict: ' + ", ".join(missing_keys))

    
    if len(err_msg) > 0:
        err_msg.insert(0, 'The model and loaded state_dict do not match exactly')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """
    Load checkpoint from a file or URI.

    Args:
        model: Module to load checkpoint into.
        filename: File path, supports local path and URL.
        map_location: Same as :func:`torch.load`.
        strict: Whether to enforce exact parameter matching.
        logger: Logger used for error/warning messages.

    Returns:
        The loaded checkpoint dictionary.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class Downsample(nn.Module):
    """Simple 3x3 stride-2 downsampling block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5),
            nn.GELU(approximate='tanh'),
        )

    def forward(self, x):
        return self.reduction(x)


class MambaVisionMixer(nn.Module):
    """MambaVision mixer module based on state-space modeling (baseline-aligned)."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # State projection
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )

        # Time-step projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)

        # Time-step initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize time parameters
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # A parameters stored in log form
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D parameters
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # Two parallel 1D conv layers (baseline-aligned, kernel_size=d_conv)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """Forward pass, kept consistent with the baseline MambaVisionMixer."""
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        A = -torch.exp(self.A_log.float())

        # Apply SiLU activation and 1D conv to x and z separately (padding='same')
        x = F.silu(
            F.conv1d(
                input=x,
                weight=self.conv1d_x.weight,
                bias=self.conv1d_x.bias,
                padding="same",
                groups=self.d_inner // 2,
            )
        )
        z = F.silu(
            F.conv1d(
                input=z,
                weight=self.conv1d_z.weight,
                bias=self.conv1d_z.bias,
                padding="same",
                groups=self.d_inner // 2,
            )
        )

        # Projection operations
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        # Selective scan
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )

        # Merge outputs
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class LCBI_MambaBlock(nn.Module):
    """LCBI-Mamba block: inject a local convolution branch into the Mamba path."""

    def __init__(self, dim, drop_path=0., layer_scale=None, use_depthwise_conv=True, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # self.mixer = MambaVisionMixer(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mixer = MambaVisionMixer(d_model=dim, d_state=16, d_conv=4, expand=expand)
        # Lightweight local depthwise-separable conv branch for MambaVisionMixer compensation
        # DepthwiseConv (groups=dim) + pointwise 1x1 conv for channel mixing
        self.use_depthwise_conv = use_depthwise_conv
        if self.use_depthwise_conv:
            self.conv_branch = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim, eps=1e-5),
                nn.GELU(approximate='tanh'),
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            )
        else:
            self.conv_branch = None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        use_layer_scale = layer_scale is not None and isinstance(layer_scale, (int, float))
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1.0

    def forward(self, x):
        # Unified handling: residual in sequence space, then restore original x shape
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.view(B, C, -1).transpose(1, 2)      # (B, L, C)
            y = self.mixer(self.norm(x_flat))              # (B, L, C)
            y = self.drop_path(self.gamma * y)             # (B, L, C)
            y_img = y.transpose(1, 2).contiguous().view_as(x)  # Restore to x shape

            # Depthwise-separable conv branch (lightweight local information compensation)
            if self.conv_branch is not None:
                branch = self.conv_branch(x)
            else:
                branch = torch.zeros_like(x)

            return x + y_img + branch
        elif x.dim() == 3:
            y = self.mixer(self.norm(x))
            y = self.drop_path(self.gamma * y)
            return x + y
        else:
            raise ValueError(f"Unsupported input shape for LCBI_MambaBlock: {x.shape}")


class LCBI_AttnBlock(nn.Module):
    """LCBI-Attn block: inject a local convolution branch into the attention path."""

    def __init__(self, dim, num_heads=4, drop_path=0., mlp_ratio=2., qkv_bias=True, drop=0., use_conv_branch=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, bias=qkv_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, act_layer=nn.GELU, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # Simple spatial conv branch: 3x3 standard conv (no channel scaling)
        # Uses Conv2d + BN + GELU, with output channels matching input
        self.use_conv_branch = use_conv_branch
        if self.use_conv_branch:
            self.conv_branch = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dim, eps=1e-5),
                nn.GELU(approximate='tanh'),
            )
        else:
            self.conv_branch = None

    def forward(self, x):
        B, C, H, W = x.shape
        # Compute conv branch first (from original image features) for fusion with attention output
        if self.conv_branch is not None:
            conv_branch = self.conv_branch(x)
        else:
            conv_branch = torch.zeros_like(x)

        x_flat = x.view(B, C, -1).transpose(1, 2)  # (B, L, C)
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + self.drop_path(attn_out)
        x_flat = x_flat + self.drop_path(self.mlp(self.norm2(x_flat)))
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        # Add attention-path output and conv branch to form final residual output
        return x + conv_branch



class MambaSeed(nn.Module):
    """Tiny-level MambaSeed backbone based on MambaVisionMixer."""

    def __init__(self,
                 in_chans=3,
                 num_classes=10,
                 stem_dim=32,
                 dims=(32, 64, 128, 256),
                 depths=(2, 3, 6, 3),
                 drop_path_rate=0.2,
                 mlp_ratio=2.0,
                 use_attention: bool = True,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.use_attention = use_attention

        # Stem: 3x3 conv -> BN -> GELU, output dims[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim, eps=1e-5),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(stem_dim, dims[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dims[0], eps=1e-5),
            nn.GELU(approximate='tanh'),
        )

        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        cur = 0
        in_dim = dims[0]
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            blocks = []
            # Downsample before each stage except the first one
            if i > 0:
                blocks.append(Downsample(in_dim, dim))
            in_dim = dim

            # Stage composition: mostly LCBI_MambaBlock, with limited attention in high-level stage
            for j in range(depth):
                drop_p = dpr[cur + j]
                if self.use_attention and i == len(dims) - 1 and j == 0:
                    blocks.append(LCBI_AttnBlock(dim=dim, num_heads=4, drop_path=drop_p, mlp_ratio=mlp_ratio))
                else:
                    blocks.append(LCBI_MambaBlock(dim=dim, drop_path=drop_p, layer_scale=1e-2))
            cur += depth

            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.LayerNorm(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Weight initialization.

        Args:
            m: Module.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Return keywords that should not use weight decay."""
        return {'rpb'}

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)  # (B, L, C)
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class MambaSeed224(nn.Module):
    """Low-FLOPs MambaSeed variant: training and profiling use 224x224 resolution."""

    def __init__(self,
                 in_chans=3,
                 num_classes=10,
                 stem_dim=16,
                 dims=(16, 32, 64, 128, 192),
                #  depths=(2, 2, 3, 3),
                 depths=(1, 1, 1, 1, 1),
                 drop_path_rate=0.2,
                 mlp_ratio=2.0,
                 use_depthwise_conv=False,
                 use_attn=False,
                 use_conv_branch_in_attn=False,
                 expand=1,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.use_attn = use_attn
        self.use_depthwise_conv = use_depthwise_conv
        self.use_conv_branch_in_attn = use_conv_branch_in_attn
        self.expand = expand

        # Stem: reduce stem_dim and channels to lower compute cost
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim, eps=1e-5),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(stem_dim, dims[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dims[0], eps=1e-5),
            nn.GELU(approximate='tanh'),
        )

        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        cur = 0
        in_dim = dims[0]
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            blocks = []
            if i > 0:
                blocks.append(Downsample(in_dim, dim))
            in_dim = dim

            for j in range(depth):
                drop_p = dpr[cur + j]
                # Use LCBI_AttnBlock in the top stage and LCBI_MambaBlock elsewhere
                # Note: place LCBI_AttnBlock at the end of this stage
                if self.use_attn and i == len(dims) - 1 and j == depth - 1:
                    blocks.append(LCBI_AttnBlock(dim=dim, num_heads=4, drop_path=drop_p, mlp_ratio=mlp_ratio, use_conv_branch=self.use_conv_branch_in_attn))
                else:
                    blocks.append(LCBI_MambaBlock(dim=dim, drop_path=drop_p, layer_scale=1e-2, use_depthwise_conv=self.use_depthwise_conv, expand=self.expand))
            cur += depth

            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.LayerNorm(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x



@register_pip_model
@register_model
# def mamba_seed_224(pretrained=False, stem_dim=16, dims=(16, 32, 64, 128, 192), depths=(1, 1, 1, 1, 1), use_depthwise_conv=True, use_attn=True, use_conv_branch_in_attn=True, **kwargs):
def mamba_seed_224(pretrained=False, stem_dim=16, dims=(16, 32, 64, 128, 192), depths=(1, 1, 1, 1, 1), use_depthwise_conv=True, use_attn=True, use_conv_branch_in_attn=True, expand=3, **kwargs):
    """Low-FLOPs MambaSeed variant designed for 224x224 resolution."""
    model_path = kwargs.pop("model_path", "/tmp/mamba_seed_224.pth.tar")
    drop_path_rate = kwargs.pop("drop_path_rate", 0.15)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_deformable_T').to_dict()
    # Unify default input size to 224x224 for downstream tool compatibility
    pretrained_cfg["input_size"] = (3, 224, 224)
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)

    model = MambaSeed224(stem_dim=stem_dim, dims=dims, depths=depths, drop_path_rate=drop_path_rate, use_depthwise_conv=use_depthwise_conv, use_attn=use_attn, use_conv_branch_in_attn=use_conv_branch_in_attn, expand=expand, **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg

    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        _load_checkpoint(model, model_path, strict=False)
    return model

