# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any

import torch
from torch import Tensor

from slicegpt.modules import RMSN

from . import utils
from .config import config
from .model_adapter import LayerAdapter, ModelAdapter

from itertools import chain


def get_layer0_inputs(model_adapter: ModelAdapter, batch: Tensor) -> tuple[Tensor, tuple, dict[str, Any]]:
    """
    Returns the inputs to the first layer of the model (after embeddings).

    Also returns the additional args and kwargs that are passed to
    the first layer (such as the attention mask, or caches K/V values).

    This relies on all arguments to subsequent layers being the same.

    NB: this won't work from OPT 350m.
    """
    # Move embeddings to device.
    for W in model_adapter.get_embeddings():
        W.weight = torch.nn.Parameter(W.weight.to(config.device))

    class Catcher(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *args, **kwargs):
            self.saved_args = args
            self.saved_kwargs = kwargs
            raise ValueError

    layer0_adapter = model_adapter.get_layers()[0]
    layer0_catcher = Catcher()
    model_adapter.set_raw_layer_at(0, layer0_catcher)

    try:
        batch = utils.map_tensors(batch, device=config.device)
        model_adapter.model(**batch)
    except ValueError:
        pass

    # grab the inputs and caught arguments
    args = layer0_catcher.saved_args
    kwargs = layer0_catcher.saved_kwargs

    # put the caught stuff on cpu
    args = utils.map_tensors(args, device='cpu')
    kwargs = utils.map_tensors(kwargs, device='cpu')

    # put the layer back to normal
    model_adapter.set_raw_layer_at(0, layer0_adapter.layer)

    # Move embeddings back to cpu, and clear GPU cache.
    for W in model_adapter.get_embeddings():
        W.weight = torch.nn.Parameter(W.weight.to('cpu'))

    # Run GC and cleanup GPU memory
    utils.cleanup_memory()

    return args[layer0_adapter.hidden_states_args_position], args, kwargs


def get_signals(
    layer_adapter: LayerAdapter, layer_args: list[tuple], layer_kwargs: list[dict[str, Any]], truncate_fn = None) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output of the layer (not layernormed) and the input to the MLP (pre-layernorm).
    """
    mlp_ln_inputs = []
    outputs = []

    layer_adapter.layer.to(config.device)

    def hook_fn(_, args: tuple, _output: Any) -> None:
        inp = args[0]  # Position in RMSN.forward args
        mlp_ln_inputs.append(inp.cpu())

    def pre_hook_fn(_, args: tuple):
        inp = args[0]
        return (truncate_fn(inp),)

        

    second_layernorm = layer_adapter.get_second_layernorm()
    assert isinstance(second_layernorm, RMSN)
    pre_hook = None
    if truncate_fn is not None:
        attn_mlp_interface = layer_adapter.get_attn_mlp_interface()
        pre_hook = attn_mlp_interface.register_forward_pre_hook(pre_hook_fn)
    post_hook = second_layernorm.register_forward_hook(hook_fn)
    for i, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(layer_args, layer_kwargs)):
        layer_args_batch, layer_kwargs_batch = utils.map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=config.device
        )
        out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
        if isinstance(out, tuple):
            out = out[layer_adapter.hidden_states_output_position]
        out = out.cpu()
        outputs.append(out)

        if mlp_ln_inputs[i].ndim == 2:
            batch_size, seqlen, _ = out.shape  # both batch_size and seqlen are can vary from batch to batch
            mlp_ln_inputs[i] = mlp_ln_inputs[i].reshape(batch_size, seqlen, -1)

    for hook in filter(None, [pre_hook, post_hook]):
        hook.remove()
    return mlp_ln_inputs, outputs


class signal_catcher:
        def __init__(self):
            self.signals = []

        def __call__(self, module, args, output):
            self.signals.append(output.detach().cpu())


def get_attn_signals(
    layer_adapter: LayerAdapter, layer_args: list[tuple], layer_kwargs: list[dict[str, Any]]) -> dict[torch.nn.Module, list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output of the attention projections: Q,K,V
    """
    handles = []

    layer_adapter.layer.to(config.device)

    attn_projs = layer_adapter.get_attention_inputs()
    signal_catchers = {}
    for module in attn_projs:
        s = signal_catcher()
        signal_catchers[module] = s
        handles.append(module.register_forward_hook(s))
        
    # stop computation
    def stopper(module, args):
        raise ValueError
    handles.append(layer_adapter.get_attention_output().register_forward_pre_hook(stopper))


    for i, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(layer_args, layer_kwargs)):
        layer_args_batch, layer_kwargs_batch = utils.map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=config.device
        )
        try:
            layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
        except ValueError:
            pass

    for hook in handles:
        hook.remove()
    
    return dict((k, catcher.signals) for k, catcher in signal_catchers.items())


def get_mlp_signals(
    layer_adapter: LayerAdapter, layer_args: list[tuple], layer_kwargs: list[dict[str, Any]], truncate_fn = None) -> dict[torch.nn.Module, list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output of the attention projections: Q,K,V
    """
    handles = []
    pre_hook = None

    def pre_hook_fn(_, args: tuple):
        inp = args[0]
        return (truncate_fn(inp), *(args[1:]))

    layer_adapter.layer.to(config.device)

    pre_hook = None
    if truncate_fn is not None:
        attn_mlp_interface = layer_adapter.get_attn_mlp_interface()
        pre_hook = attn_mlp_interface.register_forward_pre_hook(pre_hook_fn)

    mlp_inputs = layer_adapter.get_mlp_inputs()
    signal_catchers = {}
    for module in mlp_inputs:
        s = signal_catcher()
        signal_catchers[module] = s
        handles.append(module.register_forward_hook(s))
    # stop computation
    def stopper(module, args):
        raise ValueError
    handles.append(layer_adapter.get_mlp_output().register_forward_pre_hook(stopper))

    for i, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(layer_args, layer_kwargs)):
        layer_args_batch, layer_kwargs_batch = utils.map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=config.device
        )
        try:
            layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
        except ValueError:
            pass

    handles.append(pre_hook)
    for hook in filter(None, handles):
        hook.remove()
    
    return dict((k, catcher.signals) for k, catcher in signal_catchers.items())

def get_attn_mlp_signals(layer_adapter: LayerAdapter, layer_args: list[tuple], layer_kwargs: list[dict[str, Any]]) -> dict[torch.nn.Module, list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output of the attention projections and mlp up projection
    """
    handles = []

    layer_adapter.layer.to(config.device)

    attn_projs = layer_adapter.get_attention_inputs()
    mlp_inputs = layer_adapter.get_mlp_inputs()
    signal_catchers = {}
    for module in chain(attn_projs, mlp_inputs):
        s = signal_catcher()
        signal_catchers[module] = s
        handles.append(module.register_forward_hook(s))
    outputs = []
    for i, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(layer_args, layer_kwargs)):
        layer_args_batch, layer_kwargs_batch = utils.map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=config.device
        )
        outputs.append(layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch))

    for hook in handles:
        hook.remove()
    
    return dict((k, catcher.signals) for k, catcher in signal_catchers.items()), outputs

def get_modules_and_layer_outputs_signals(modules: list[torch.nn.Module], layer_adapter: LayerAdapter, layer_args: list[tuple], layer_kwargs: list[dict[str, Any]]) -> dict[torch.nn.Module, list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output hidden states of the specified layer modules
    """
    handles = []

    layer_adapter.layer.to(config.device)

    signal_catchers = {}
    for module in modules:
        s = signal_catcher()
        signal_catchers[module] = s
        handles.append(module.register_forward_hook(s))
    outputs = []
    for i, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(layer_args, layer_kwargs)):
        layer_args_batch, layer_kwargs_batch = utils.map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=config.device
        )
        outputs.append(layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch))

    for hook in handles:
        hook.remove()
    
    return dict((k, catcher.signals) for k, catcher in signal_catchers.items()), outputs

class inputs_catcher:
    def __init__(self):
        self.signals = []

    def __call__(self, module, inputs):
        if isinstance(inputs, tuple):
            # usually inputs are pos 0 in linear args
            inputs = inputs[0]
        self.signals.append(inputs.detach().cpu())

def get_mlp_up_projs_inputs(layer_adapter, layer_args, layer_kwargs, mlp_in_mods = list(), truncate_fn = lambda x: x):
    handles = []

    layer_adapter.layer.to(config.device)

    if len(mlp_in_mods) == 0:
        mlp_in_mods = layer_adapter.get_mlp_inputs()
    signal_catchers = {}
    for module in mlp_in_mods:
        s = inputs_catcher()
        signal_catchers[module] = s
        handles.append(module.register_forward_pre_hook(s))
    
    pre_hook = None
    def pre_hook_fn(_, args: tuple):
        inp = args[0]
        return (truncate_fn(inp), *(args[1:]))
    if truncate_fn is not None:
        attn_mlp_interface = layer_adapter.get_attn_mlp_interface()
        pre_hook = attn_mlp_interface.register_forward_pre_hook(pre_hook_fn)
    
    # stop computation
    def stopper(module, args):
        raise ValueError
    handles.append(layer_adapter.get_mlp_output().register_forward_pre_hook(stopper))
    
    for i, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(layer_args, layer_kwargs)):
        layer_args_batch, layer_kwargs_batch = utils.map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=config.device
        )
        try:
            layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
        except ValueError:
            pass

    handles.append(pre_hook)
    for hook in filter(None, handles):
        hook.remove()
    
    return dict((k, catcher.signals) for k, catcher in signal_catchers.items())

def get_attn_projs_inputs(layer_adapter, layer_args, layer_kwargs, attn_mods=list()):
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output of the attention projections: Q,K,V
    """
    handles = []

    layer_adapter.layer.to(config.device)

    if len(attn_mods) == 0:
        attn_mods = layer_adapter.get_attention_inputs()
    signal_catchers = {}
    for module in attn_mods:
        s = inputs_catcher()
        signal_catchers[module] = s
        handles.append(module.register_forward_pre_hook(s))
        
    # stop computation
    def stopper(module, args):
        raise ValueError
    handles.append(layer_adapter.get_attention_output().register_forward_pre_hook(stopper))


    for i, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(layer_args, layer_kwargs)):
        layer_args_batch, layer_kwargs_batch = utils.map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=config.device
        )
        try:
            layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
        except ValueError:
            pass

    for hook in handles:
        hook.remove()
    
    return dict((k, catcher.signals) for k, catcher in signal_catchers.items())


def get_attn_output_inputs(layer_adapter, layer_args, layer_kwargs, truncate_fn = lambda x: x):
    handles = []

    layer_adapter.layer.to(config.device)

    catcher = inputs_catcher()
    out_proj = layer_adapter.get_attention_output()
    handles.append(out_proj.register_forward_pre_hook(catcher))
    
    # stop computation
    # this will fire after the pre-hook that registers input signals, so everything should work out fine
    def stopper(*args, **kwargs):
        raise ValueError
    handles.append(out_proj.register_forward_hook(stopper))
    
    for i, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(layer_args, layer_kwargs)):
        layer_args_batch, layer_kwargs_batch = utils.map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=config.device
        )
        try:
            layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
        except ValueError:
            pass

    for hook in handles:
        hook.remove()
    
    return catcher.signals

def get_mlp_output_inputs(layer_adapter, layer_args, layer_kwargs, truncate_fn = lambda x: x):
    handles = []

    layer_adapter.layer.to(config.device)

    mlp_output = layer_adapter.get_mlp_output()
    catcher = inputs_catcher()
    handles.append(mlp_output.register_forward_pre_hook(catcher))
    
    second_layernorm = layer_adapter.get_second_layernorm()
    assert isinstance(second_layernorm, RMSN)
    pre_hook = None
    def pre_hook_fn(_, args: tuple):
        inp = args[0]
        return (truncate_fn(inp), *(args[1:]))
    if truncate_fn is not None:
        attn_mlp_interface = layer_adapter.get_attn_mlp_interface()
        pre_hook = attn_mlp_interface.register_forward_pre_hook(pre_hook_fn)
    
    # stop computation
    # this will fire after the pre-hook that registers input signals, so everything should work out fine
    def stopper(*args, **kwargs):
        raise ValueError
    handles.append(mlp_output.register_forward_hook(stopper))
    
    for i, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(layer_args, layer_kwargs)):
        layer_args_batch, layer_kwargs_batch = utils.map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=config.device
        )
        try:
            layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
        except ValueError:
            pass

    handles.append(pre_hook)
    for hook in filter(None, handles):
        hook.remove()
    
    return catcher.signals