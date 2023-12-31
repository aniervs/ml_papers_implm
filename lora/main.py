import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


class LoRAParametrization(nn.Module):
    def __init__(self, in_features, out_features, rank=1, alpha=1, device=torch.device('cpu')):
        super().__init__()

        self.lora_A = nn.Parameter(torch.zeros(size=(rank, out_features), device=device, requires_grad=True))
        self.lora_B = nn.Parameter(torch.zeros(size=(in_features, rank), device=device, requires_grad=True))

        nn.init.normal_(self.lora_A, mean=0, std=1)

        self.lora_alpha = alpha
        self.lora_rank = rank
        self.enabled = True

    def forward(self, weights):
        if self.enabled:
            return weights + torch.matmul(self.lora_B, self.lora_A).view(weights.shape) * self.lora_alpha / self.lora_rank
        else:
            return weights


def linear_layer_parametrization(layer, device=torch.device('cpu'), rank=1, alpha=1):
    in_features, out_features = layer.weight.shape
    return LoRAParametrization(in_features, out_features, rank=rank, alpha=alpha, device=device)


def conv2d_layer_parametrization(layer, device=torch.device('cpu'), rank=1, alpha=1):
    out_channel, in_channel, kernel_1, kernel_2 = layer.weight.shape # TODO: test it. Not sure if it should be k1,k2 or k2,k1
    return LoRAParametrization(in_channel * kernel_1, out_channel * kernel_2, rank=rank, alpha=alpha, device=device) # TODO: check if the rank should be multiplied by the kernel size


def apply_lora_single_layer(layer, device=torch.device('cpu'), rank=1, alpha=1):
    name = str(layer)
    if "Linear" in name:
        parametrize.register_parametrization(
            layer, "weight", linear_layer_parametrization(layer, device=device, rank=rank, alpha=alpha)
        )
    elif "Conv2d" in name: # TODO; do the same for Conv1D, Conv3D (hint: maybe there's a general implm for all of them)
        parametrize.register_parametrization(
            layer, "weight", conv2d_layer_parametrization(layer, device=device, rank=rank, alpha=alpha)
        )


def apply_lora_all_params(module, device=torch.device('cpu'), rank=1, alpha=1):
    children = list(module.children())
    if len(children) == 0:
        apply_lora_single_layer(module, device=device, rank=rank, alpha=alpha)
    else:
        for layer in children:
            apply_lora_all_params(layer, device=device, rank=rank, alpha=alpha)


def freeze_non_lora_params(model):
    for name, param in model.named_parameters():
        if 'lora' not in name:  # TODO: fix this. Most likely, biases shouldn't be frozen
            param.requires_grad = False


def enable_disable_lora_single_layer(layer, enabled=True):
    name = str(layer)
    if "LoRA" in name or "lora" in name:
        layer.enabled = enabled


def enable_disable_lora_all_params(module, enabled=True):
    children = list(module.children())
    if len(children) == 0:
        enable_disable_lora_single_layer(module, enabled)
    else:
        for layer in children:
            enable_disable_lora_all_params(layer, enabled)
