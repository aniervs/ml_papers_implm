# LoRA implementation

This is with educational purposes.  The final goal is to have a generic library on LoRA that can be used with many mainstream layers out-of-the-box:

### Main goals right now:

- [x] Linear Layer
- [x] Conv2D
- [ ] Attention Layer

### Instructions on how to use

The `lora` package has on the `main` module:
- `LoRAParametrization` class: the matrices $A$ and $B$ such that $\Delta W = BA$
 
  - When `enabled` is to `True`, any inference will use $W + \frac{\alpha}{r}BA$. When set to `False`, any inference will use just `W`. 
- `linear_layer_parametrization` function: just creates a `LoRAParametrization` object for the corresponding `linear` layer.
- `apply_lora_single_layer` function: Given a childless layer (as a Pytorch module), it applies LoRA to it.
- `apply_lora_all_params` function: Recursive function that goes through the modules and submodules of a Pytorch Module, and it applies LoRA to all weights it finds on the way.
- `freeze_non_lora_params` function: freezes all original parameters, including their biases.
- `enable_disable_lora_all_params` function: Recursive function that enables LoRA inference in all layers.

--- 
To apply LoRA to your Pytorch model (as an `nn.Module`), just use:
```Python
from lora.main import apply_lora_all_params, freeze_non_lora_params
apply_lora_all_params(your_model, device)
freeze_non_lora_params(your_model)
```

and fine-tune it as you would usually do.
*Yes, just three lines!*

---

To do inference on the weights of the original model, just use:
```Python
from lora.main import apply_lora_all_params
enable_disable_lora_all_params(your_model, enabled=False)
```


### Details:
- Original paper: https://arxiv.org/abs/2106.09685
- Official repository: https://github.com/microsoft/LoRA
 
