import math
import torch
from torch import nn

class LoraLinear(nn.Module):
    def __init__(self, layer, r, lora_alpha=32, lora_dropout=0.05):
        super().__init__()        
        self.layer = layer
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(lora_dropout) if (lora_dropout > 0.) else nn.Identity()

        dtype, device = layer.weight.dtype, layer.weight.device 
        self.lora_A = nn.Parameter(
            torch.zeros((r, layer.in_features), dtype=dtype, device=device),
            requires_grad=True
        )
        self.lora_B = nn.Parameter(
            torch.zeros((layer.out_features, r), dtype=dtype, device=device),
            requires_grad=True
        )
        # In the LoRA paper, random Gaussian initialization was used for A, but the 
        # official implementation uses Kaiming Uniform Initialzation.
        # B is initialized to zero matrix (same as in paper)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False

    def forward(self, x):
        x = self.lora_dropout(x)
        output = self.layer(x) + ((x @ self.lora_A.T @ self.lora_B.T) * self.scaling)
        return output
    
def attach_lora_weights(model, target_modules, r=8, lora_alpha=32, lora_dropout=0.05):
    assert r > 0, "r needs to be > 0"
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            # Traverse to parent (handle numeric indices)
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
            last = parts[-1]
            wrapper = LoraLinear(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            if last.isdigit():
                parent[int(last)] = wrapper
            else:
                setattr(parent, last, wrapper)
            print(f"[LoRA] wrapped: {name}")


def _get_parent_and_name(model, fullname):
    """
    Return (parent_module, last_name_or_index) so callers can set/replace child.
    Handles numeric module names (e.g. Sequential indices).
    """
    parent = model
    parts = fullname.split(".")
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]

def merge_and_unwrap_lora(model, set_trainable=True):
    for name, module in list(model.named_modules()):
        if isinstance(module, LoraLinear):
            delta = (module.lora_B @ module.lora_A) * module.scaling
            delta = delta.to(device=module.layer.weight.device, dtype=module.layer.weight.dtype)

            with torch.no_grad():
                module.layer.weight += delta.detach()

            module.layer.weight.requires_grad = bool(set_trainable)
            if module.layer.bias is not None:
                module.layer.bias.requires_grad = bool(set_trainable)

            # Replace wrapper with the original linear in the parent module
            parent, last = _get_parent_and_name(model, name)
            if last.isdigit():
                parent[int(last)] = module.layer
            else:
                setattr(parent, last, module.layer)
    return model