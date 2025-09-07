# TODO: Handle weight shape consistency when fan_in_fan_out=True. 
# TODO: Handle dropout

import math
import torch
import torch.nn.utils.parametrize as parameterize
from torch import nn


class LoraParameterization(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        fan_in_fan_out=False,
        r=0,
        lora_alpha=32,
        lora_dropout_p=0.05,
        **kwargs
    ):
        super().__init__()
        self.r = r
        if self.r > 0:
            self.fan_in_fan_out = fan_in_fan_out
            self.scaling = lora_alpha / r
            self.lora_dropout = nn.Dropout(lora_dropout_p) if lora_dropout_p > 0.0 else (lambda x: x)
            self.swap_fn = (lambda x: (x[1], x[0])) if self.fan_in_fan_out else (lambda x: x)
            self.lora_A = nn.Parameter(
                torch.zeros(self.swap_fn((r, in_features))), 
                True
            )
            self.lora_B = nn.Parameter(
                torch.zeros(self.swap_fn((out_features, r))),
                True
            )
            # In the LoRA paper, random Gaussian initialization was used for A, but the 
            # official implementation uses Kaiming Uniform Initialzation.
            # B is initialized to zero matrix (same as in paper)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.forward_fn = self.lora_forward if self.r > 0 else self.identity_forward
    
    def identity_forward(self, X):
        return X
    
    def lora_forward(self, X):
        return X + torch.matmul(*self.swap_fn((self.lora_B, self.lora_A))) * self.scaling
    
    def forward(self, X):
        return self.forward_fn(X)   

    @classmethod
    def add_lora_weights_to_linear(cls, layer, r=0, lora_alpha=32, lora_dropout_p=0.05):
        device = layer.weight.device
        out_features, in_features = layer.weight.shape
        return cls(
            in_features, out_features, fan_in_fan_out=False, r=r, lora_alpha=lora_alpha, lora_dropout_p=lora_dropout_p
        ).to(device) 


base_lora_config = {
    nn.Linear: {
        "weight": LoraParameterization.add_lora_weights_to_linear
    }
}

def add_lora_weights(layer, r=0, lora_alpha=32, lora_dropout_p=0.05):
    for attr, parameterization_fn in base_lora_config[type(layer)].items():
        # When accessing module.weight, the layer will return the parametrized version,
        # (parametrization_fn(module.weight)). If the original tensor requires a gradient,
        # the backward pass will differentiate through parametrization_fn, and the 
        # optimizer will update the tensor accordingly.
        parameterize.register_parametrization(
            layer, 
            attr, 
            parameterization_fn(
                layer, r, lora_alpha, lora_dropout_p
            )
        )
        
def attach_lora_to_layers(model, target_modules, r=0, lora_alpha=32, lora_dropout_p=0.05):
    for name, layer in model.named_modules():
        if any([t in name for t in target_modules]) and (type(layer) in base_lora_config):
            add_lora_weights(layer, r, lora_alpha, lora_dropout_p) # in-place