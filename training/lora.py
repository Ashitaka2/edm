import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union
# from . import logger
import math

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from torch_utils import persistence


class FourierFeatures(nn.Module):
    """Random Fourier features.
    Args:
        frequency_matrix (torch.Tensor): Matrix of frequencies to use
            for Fourier features. Shape (num_frequencies, num_coordinates).
            This is referred to as B in the paper.
        learnable_features (bool): If True, fourier features are learnable,
            otherwise they are fixed.
    """
    def __init__(self, frequency_matrix, learnable_features=False):
        super(FourierFeatures, self).__init__()
        if learnable_features:
            self.frequency_matrix = nn.Parameter(frequency_matrix)
        else:
            # Register buffer adds a key to the state dict of the model. This will
            # track the attribute without registering it as a learnable parameter.
            # We require this so frequency matrix will also be moved to GPU when
            # we call .to(device) on the model
            self.register_buffer('frequency_matrix', frequency_matrix)
        self.learnable_features = learnable_features
        self.num_frequencies = frequency_matrix.shape[0]
        # Factor of 2 since we consider both a sine and cosine encoding
        self.feature_dim = 2 * self.num_frequencies

    def forward(self, coordinates):
        """Creates Fourier features from coordinates.

        Args:
            coordinates (torch.Tensor): Shape (num_points, coordinate_dim)
        """
        # The coordinates variable contains a batch of vectors of dimension
        # coordinate_dim. We want to perform a matrix multiply of each of these
        # vectors with the frequency matrix. I.e. given coordinates of
        # shape (num_points, coordinate_dim) we perform a matrix multiply by
        # the transposed frequency matrix of shape (coordinate_dim, num_frequencies)
        # to obtain an output of shape (num_points, num_frequencies).
        # print('self.frequency_matrix.shape: ', self.frequency_matrix.shape)
        # print('coordinates.shape: ', coordinates.shape)
        prefeatures = torch.matmul(coordinates, self.frequency_matrix.T)
        # print('prefeatures.shape: ', prefeatures.shape)
        # Calculate cosine and sine features
        cos_features = torch.cos(2 * math.pi * prefeatures)
        sin_features = torch.sin(2 * math.pi * prefeatures)
        # Concatenate sine and cosine features
        return torch.cat((cos_features, sin_features), dim=-1)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class SimpleEmbedding(nn.Module):
    def __init__(self, output_size):
        super(SimpleEmbedding, self).__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 128) 
        self.layer3 = nn.Linear(128, 64)  
        self.layer4 = nn.Linear(64, output_size)

        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.silu(self.layer1(x))
        x = self.silu(self.layer2(x))
        x = self.silu(self.layer3(x))
        x = self.layer4(x)
        return F.softmax(x, dim=1)
        # return x


class FourierEmbedding(nn.Module):
    def __init__(self, num_frequency, output_size):
        super(FourierEmbedding, self).__init__()
        frequency_matrix = torch.normal(mean=torch.zeros(num_frequency, 1), std=2.0)
        self.fourier_feature = FourierFeatures(frequency_matrix)
        
#         self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(2*num_frequency, 128) 
        self.layer3 = nn.Linear(128, 64)  
        self.layer4 = nn.Linear(64, output_size)

        self.silu = nn.SiLU()

    def forward(self, x):
        fourier_coords = self.fourier_feature(x)
#         x = self.silu(self.layer1(x))
        x = self.silu(self.layer2(fourier_coords))
        x = self.silu(self.layer3(x))
        x = self.layer4(x)
        return F.softmax(x, dim=1)
        # return x


class LoraInjectedConv2d(nn.Module): #for cLoRA
    
    def __init__(
            self, in_channels, out_channels, bias=False, r_t=4, r_c=None, scale=1.0, num_classes=None, num_timesteps=18,
            null_rate=0.0, interval=1, interpolate=None,
    ):
        super().__init__()
        
        assert (r_c is None) == (num_classes is None)
        
        self.r_c = r_c
        self.r_t = r_t
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv2d = conv_nd(2, in_channels, out_channels, 1, bias=bias) #conv_nd arguments: dim(selecting ConvNd), c_in, c_out, kernel_size, stride, padding, dilation, groups, bias
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        if r_c is not None:
            self.c_lora_down = conv_nd(2, in_channels, r_c * num_classes, 1, bias=False)
            self.c_lora_up = conv_nd(2, r_c * num_classes, out_channels, 1, bias=False)

            nn.init.normal_(self.c_lora_down.weight, std=1 / r_c)
            nn.init.zeros_(self.c_lora_up.weight)
        
        # self.t_lora_down = conv_nd(2, in_channels, r_t * num_timesteps, 1, bias=False)
        # self.t_lora_up = conv_nd(2, r_t * num_timesteps, out_channels, 1, bias=False)
        # nn.init.normal_(self.t_lora_down.weight, std=1 / r_t)
        # nn.init.zeros_(self.t_lora_up.weight)

        self.scale = scale
        self.c_selector = None
        self.t_selector = None
        self.mask = None
        self.bypass = False
        
        self.embedding = None

        self.interval = interval
        self.null_rate = null_rate
        self.interpolate = interpolate
        if self.interpolate == "train":
            # self.embedding = SimpleEmbedding(output_size = self.num_timesteps).to(self.conv2d.weight.device) #simple embedding using raw MLP
            self.embedding = FourierEmbedding(output_size = self.num_timesteps, num_frequency=64).to(self.conv2d.weight.device) #MLP with frozen RFF in first layer 
        self.c_bias = nn.Parameter(torch.zeros((self.r_c * self.num_classes, self.out_channels)))

        self.label_dropout = 0
    
    
    def set_t_selector(self, ts, reference_points):
        #find the closest value in reference_points and display it as one-hot style. self.num_timesteps should be 18 (=len(reference_points))
        reference_points = torch.tensor(reference_points, device = self.conv2d.weight.device)
        ts = ts.to(device=self.conv2d.weight.device, dtype=self.conv2d.weight.dtype).unsqueeze(1)  # Shape: [N, 1]
        reference_points = reference_points.unsqueeze(0)  # Shape: [1, M]
        differences = torch.abs(ts - reference_points)
        _, min_indices = torch.min(differences, dim=1)
        num_classes = reference_points.numel()
        mask = torch.nn.functional.one_hot(min_indices, num_classes=num_classes)
        
        mask = torch.repeat_interleave(mask, self.r_t, dim=1)
        self.t_selector = mask.unsqueeze(-1).unsqueeze(-1).to(self.conv2d.weight.device).to(self.conv2d.weight.dtype)
        return

    def select_class(self, class_labels):
        
        assert class_labels.size()[-1] == self.num_classes #10개
        # print(f"class_labels size is : {class_labels.size()}")
        # print(f"class_label: {class_labels}")
        
        self.mask = class_labels
        if self.label_dropout:
            self.mask = self.mask * (torch.rand([class_labels.shape[0], 1], device=class_labels.device) >= self.label_dropout).to(self.mask.dtype)
        self.mask = torch.repeat_interleave(self.mask, self.r_c, dim=1)
        # print(f"mask size: {self.mask.shape}") #[B, 40]
        self.c_selector = self.mask.unsqueeze(-1).unsqueeze(-1).to(self.conv2d.weight.device).to(self.conv2d.weight.dtype)
        # print(f"c_selector size: {self.c_selector.shape}")
        
    def simple_embedding(self, ts):
        ts = ts.to(device=self.conv2d.weight.device, dtype=self.conv2d.weight.dtype).unsqueeze(1)  # Shape: [N, 1]
        # dist.print0(f"ts size: {ts.size()}")
        self.mask = self.embedding(ts).to(self.conv2d.weight.device).to(self.conv2d.weight.dtype) 
        
        # dist.print0(mask)
                
        self.mask = torch.repeat_interleave(self.mask, self.r_t, dim=1) #size: (B, r_t * num_timesteps)
        # self.t_selector = mask.unsqueeze(2).to(self.conv2d.weight.device).to(self.conv2d.weight.dtype) # (B, r_t * num_timesteps, 1)
        return
    
    # def set_c_selector(self, classes):
    #     mask = nn.functional.one_hot(classes, self.num_classes)
    #     if self.null_rate > 0.0:
    #         null_idx = torch.randint(0, 100, size=(mask.shape[0],)).to(classes.device)
    #         null_idx = null_idx.ge(100 * self.null_rate)
    #         mask = null_idx.unsqueeze(1) * mask
    #     mask = torch.repeat_interleave(mask, self.r_c, dim=1)
    #     self.c_selector = mask.unsqueeze(2).to(self.conv1d.weight.device).to(self.conv1d.weight.dtype)
    #     return

    def set_bypass(self, bypass):
        self.bypass = bypass
        return

    def forward(self, input):
        # dist.print0(f"input size: {input.size()}")
        # dist.print0(f"c_selector size: {self.c_selector.size()}")
        # dist.print0(f"c_lora_down(input) size: {self.c_lora_down(input).size()}")
        return self.conv2d(input) \
            + self.c_lora_up(self.c_selector * self.c_lora_down(input)) \
            * self.scale + (torch.matmul(self.mask, self.c_bias)).unsqueeze(-1).unsqueeze(-1) * self.scale
    
    # def forward(self, input, emb):
    #     if isinstance(emb, tuple):
    #         t = emb[0]
    #         c = emb[1]
    #     else:
    #         t = emb
    #         c = None
        
    #     if self.bypass:
    #         return self.conv2d(input)
        
    #     elif self.interpolate == "train":
    #         # dist.print0("Trainable LoRA mask...")
    #         self.simple_embedding(t)
    #         # self.t_selector = torch.repeat_interleave(self.mask, self.r_t, dim=1) #size: (B, r_t * num_timesteps)
    #         self.t_selector = self.mask.unsqueeze(-1).unsqueeze(-1)
            
    #         # dist.print0(f"input size: {input.size()}")
    #         # dist.print0(f"t_selector size: {self.t_selector.size()}")
    #         # dist.print0(f"t_lora_down(input) size: {self.t_lora_down(input).size()}")
            
    #         return self.conv2d(input) \
    #             + self.t_lora_up(self.t_selector * self.t_lora_down(input)) \
    #             * self.scale + (torch.matmul(self.mask, self.bias)).unsqueeze(-1).unsqueeze(-1) * self.scale
              
    #     else:
    #         reference_points = [80.0, 57.58598472124816, 40.78557379650796, 28.374584604156844, 19.35245298032523, 
    #                             12.91008238075732, 8.400935309099816, 5.315194521796382, 3.256821519765537, 1.9233398370400518, 
    #                             1.088170636545279, 0.5853481231945422, 0.29644228447915727, 0.13951646873101678, 0.05994731123547159, 
    #                             0.022934518372333384, 0.0075280199627840785, 0.002000000000000003] #num_timesteps=18, EDM sampling sigma
    #         self.set_t_selector(t, reference_points)
            
    #         # dist.print0(f"input size: {input.size()}")
    #         # dist.print0(f"t_selector size: {self.t_selector.size()}")
    #         # dist.print0(f"t_lora_down(input) size: {self.t_lora_down(input).size()}")
            
    #         if c is not None:
    #             self.set_c_selector(c)
    #             return self.conv2d(input) \
    #                    + self.c_lora_up(self.c_selector * self.c_lora_down(input)) * self.scale \
    #                    + self.t_lora_up(self.t_selector * self.t_lora_down(input)) * self.scale
    #         else:
    #             self.t_selector = self.t_selector.unsqueeze(-1)
    #             return self.conv2d(input) \
    #                    + self.t_lora_up(self.t_selector * self.t_lora_down(input)) \
    #                    * self.scale


def extract_embedding_weights(model):
    embeddings = []
    for module in model.modules():
        # if isinstance(module, LoraInjectedConv1d):
        if isinstance(module, LoraInjectedConv2d) and module.t_embedding.weight.requires_grad:
            embeddings.append(module.t_embedding.weight.data.clone())
    return embeddings

DEFAULT_TARGET_REPLACE = {"UNetBlock"}


def _find_children(
    model,
    search_class: List[Type[nn.Module]] = [nn.Conv1d],
):
    for parent in model.modules():
        for name, module in parent.named_children():
            if any([isinstance(module, _class) for _class in search_class]):
                yield parent, name, module


def _find_modules(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Conv2d],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [
        LoraInjectedConv2d,
    ],
):
    if ancestor_class is not None:
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                # Otherwise, yield it
                yield parent, name, module


def set_bypass_for_lora(
    model: nn.Module,
    bypass: bool,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    verbose: bool = True,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[LoraInjectedConv2d], exclude_children_of=[]
    ):
        if name in ["qkv", "proj_out"]:
            _child_module.set_bypass(bypass)
            if verbose:
                print(f"Setting bypass as {bypass} for {name}")
    return


def select_class_for_lora(
    model: nn.Module,
    class_labels: torch.tensor,
    num_classes: int,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    verbose: bool = True,
):
    """
    inject lora into model, and returns lora parameter groups.
    """
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[LoraInjectedConv2d], exclude_children_of=[]
    ):
        if name in ["qkv", "proj"]:
            # class_idx = torch.matmul(class_labels, torch.arange(num_classes).to(device=class_labels.device))
            _child_module.select_class(class_labels)
            # if verbose:
            #     print(f"Selecting class {class_idx} for {name}")
    return


def inject_trainable_lora(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    r_t: int = 1,
    r_c: int = None,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    null_rate: float = 0.0,
    num_classes: int = None,
    num_timesteps: int = 11,
    scale: float = 1.0,
    interval: int = 1,
    interpolate = None,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    
    # #debugging!
    # dist.print0("Are you calling mee?????")
    # # ancestors = [module for module in model.modules()]
    # ancestor_class = "UNetBlock"
    # ancestors = (
    #         module
    #         for module in model.modules()
    #         if module.__class__.__name__ in ancestor_class
        # )
    
    # for ancestor in ancestors:
    #     # dist.print0(f"{model.named_modules()}")
    #     for fullname, module in model.named_modules():
    #         # dist.print0(f"{fullname}: {module}")
    #         from .networks import Conv2d
    #         if isinstance(module, Conv2d):
    #             # dist.print0(f"{fullname}: {module}!!?")
    #             *path, name = fullname.split(".")
    #             if name in ["qkv", "proj"]:
    #                 dist.print0(f"{fullname}: {module}!!?")


    
    
    
    # for _module, name, _child_module in _find_modules(
    #     model, target_replace_module, search_class=[nn.Conv2d]
    # ):
    from .networks import Conv2d
    for _module, name, _child_module in _find_modules(
        model, "UNetBlock", search_class=[Conv2d]
    ):
        # dist.print0(f"{name}")
        if name in ["qkv", "proj"]:
            
            weight = _child_module.weight
            bias = _child_module.bias
            if verbose:
                print("LoRA Injection : injecting lora into ", name)
                print("LoRA Injection : weight shape", weight.shape) 
                
            _tmp = LoraInjectedConv2d(
                _child_module.in_channels,
                _child_module.out_channels,
                _child_module.bias is not None,
                r_c=r_c,
                r_t=r_t,
                num_classes=num_classes,
                num_timesteps=num_timesteps,
                scale=scale,
                interval=interval,
                interpolate=interpolate,
                null_rate=null_rate,
            )
            _tmp.conv2d.weight = weight
            if bias is not None:
                _tmp.conv2d.bias = bias

            # switch the module
            _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
            _module._modules[name] = _tmp

            # require_grad_params.append(_module._modules[name].t_lora_up.parameters())
            # require_grad_params.append(_module._modules[name].t_lora_down.parameters())
            # if interpolate == "train":
            #     require_grad_params.append(_module._modules[name].embedding.parameters())

            if loras != None:
                _module._modules[name].t_lora_up.weight = loras.pop(0)
                _module._modules[name].t_lora_down.weight = loras.pop(0)

            #아래 두 줄은 필요 없는 것 같은데? (fine-tuning할 때 필요한 건가)
            # _module._modules[name].t_lora_up.weight.requires_grad = True
            # _module._modules[name].t_lora_down.weight.requires_grad = True
            names.append(name)

    return require_grad_params, names

