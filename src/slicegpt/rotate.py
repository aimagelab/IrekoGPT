# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter
from .model_utils import get_attn_projs_inputs, get_layer0_inputs, get_mlp_up_projs_inputs, get_signals, get_attn_signals, get_mlp_signals, get_attn_mlp_signals, get_modules_and_layer_outputs_signals, get_attn_output_inputs, get_mlp_output_inputs
from .slicing_scheduler import ConfigSlicingScheduler, ConstSlicingScheduler, SlicingScheduler
from .utils import cleanup_memory, map_tensors


def rotate_attention_inputs(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=config.device, dtype=torch.float64)
        Q = Q.to(W_.device)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_attention_inputs(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension

    layer_adapter.layer.attn_shortcut_Q = nn.Parameter(layer_adapter.layer.attn_shortcut_Q[:new_embedding_dimension, :])


def rotate_attention_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    Q = Q.to(W_.device)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_attention_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension


def rotate_mlp_input(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        Q = Q.to(W_.device)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension


def rotate_mlp_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    Q = Q.to(W_.device)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_mlp_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension


def rotate_embeddings(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in model_adapter.get_embeddings():
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        Q = Q.to(W_.device)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    # Run GC and cleanup GPU memory
    cleanup_memory()


def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimensions: dict[int, int]) -> None:
    # Slice the embeddings.
    for i, W in enumerate(model_adapter.get_embeddings()):
        W.weight.data = W.weight.data[:, : new_embedding_dimensions[i]]
        W.embedding_dim = new_embedding_dimensions[i]


def rotate_head(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_adapter.get_lm_head()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    Q = Q.to(W_.device)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_head(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    # Slice the head.
    lm_head = model_adapter.get_lm_head()
    lm_head.weight.data = lm_head.weight.data[:, :new_embedding_dimension]
    lm_head.in_features = new_embedding_dimension


def rotate_and_slice(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
    save_rotations: bool = False
) -> None | list:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations
    """
    if model_adapter.parallel_blocks:
        return rotate_and_slice_parallel(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)
    else:
        return rotate_and_slice_sequential(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation, save_rotations=save_rotations)

@torch.no_grad()
def rotate_sequential_ecd(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    save_rotations: bool = False,
    pca_calc_method: str = "reduce_all"
) -> None | list:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.

    Args:
        pca_calc_method (str): method with which we compute the PCA of activations. Accepted values:
        - "reduce_all": reduces both the batch and slice dimensions; given a calibration set with shape (B, N, L, D), where B is the number of examples and N is the number of model widths considered, the activations are reduced on both B and N. The effective calibration set shape is (BN, L, D).
        - "reduce_batch": reduces only the batch size; taking a calibration set with shape (B, N, L, D), only B is reduced while sequences all concatenated across the widths dimension. The effective calibration set shape is (B, NL, D).
    """

    if save_rotations:
        rotations = []
    else:
        class passer:
            def __getattr__(self, name):
                return self
            def __call__(self, *args, **kwds):
                pass
                
        rotations = passer()
    
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype
    total_slices = slicing_scheduler.n_targets + 1

    inps, args, kwargs, ignore_masks = [], [], [], []
    # we want to extend the calibration set by duplicating the activations for each target width we are using.
    # for instance: say the cal. set has shape (B, L, D) = (128, 2048, 768). we would like to extend for 2 target widths: {0.7, 0.8}. Then the new dataset would have shape (2, B, L, D'). The catch is that, for dataset[0], D' = 0.7 * D, while for dataset[1] D' = 0.8 * D.
    # Since tensors cannot have inhomogeneous dimensions, we can obtain the same effect of slicing weight matrices by setting to zero the appropriate number of elements at the end of each token, depending on the target compression ratio.
    def truncate_tensor(tensor: torch.Tensor, effective_last_shape: list[int]):
        assert len(effective_last_shape) == tensor.shape[0]
        # example of how to get the mask:
        # tensor.shape: (N, B, L, D)
        # effective_last_shape: (N, d)
        # suppose N = 2, D = 5, effective_last_shape = [2, 3]
        # first, build indices (broadcastable)
        # indices: torch.tensor[[0,1,2,3,4], [0,1,2,3,4]]
        indices = torch.arange(tensor.shape[-1], device=tensor.device).expand(tensor.shape[0], -1)
        # get mask telling which indices are above the threshold:
        # mask: torch.tensor[[0,0,1,1,1], [0,0,0,1,1]]
        mask = indices >= torch.tensor(effective_last_shape, device=tensor.device).unsqueeze(-1)
        # make dimensions broadcastable
        broadcastable_mask = mask.view([tensor.shape[0]] + [1 for _ in range(len(tensor.shape) - 2)] + [tensor.shape[-1]])
        # finally fill positions in the mask with zero
        return tensor.masked_fill(broadcastable_mask, 0.0)

    def expand_tensor_dim0(tensor: torch.Tensor, expand_size: int):
        
        return tensor.expand([expand_size] + [-1 for _ in range(len(tensor.shape))])
    
    def unflatten_first_axis(axis_shape, tensor):
        return tensor.view(axis_shape, -1, *(tensor.shape[1:]))

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=False)

    # function for reshaping activations in case of "reduce_batch" pca method
    def reduce_batch_act_reshape(x: torch.Tensor) -> torch.Tensor:
        full_x = unflatten_first_axis(slicing_scheduler.n_targets + 1, x)
        B, D = full_x.shape[1], full_x.shape[-1]
        # use reshape since transposing will break contiguity
        return full_x.transpose(0,1).reshape(B, -1, D)

    pca_fn = pca_calc
    if pca_calc_method == "reduce_all":
        act_reshape_fn = lambda x: x
        masks_reshape_fn = lambda m: m.flatten(end_dim=1)
    elif pca_calc_method == "reduce_batch":
        act_reshape_fn = reduce_batch_act_reshape
        masks_reshape_fn = lambda m: m.transpose(0,1).flatten(start_dim=1)
        pca_fn = pca_calc_2 if pca_calc_method == "experimental" else pca_calc
    elif pca_calc_method == "experimental":
        act_reshape_fn = lambda x: unflatten_first_axis(slicing_scheduler.n_targets + 1, x)
        masks_reshape_fn = lambda m: m
        pca_fn = pca_calc_2
    
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    # rotate and slice embeddings
    eig_val, Q = pca_calc(inps, ignore_masks)
    Q = Q.to(device=config.device)
    rotations.append(Q.cpu())
    rotate_embeddings(model_adapter, Q)

    # Rotation matrix between embeddings-first layer is always the same, indipendent of the slicing size
    logging.info("Rotate and slice layers")
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer_adapter, Q)
        # slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        # get signal between attention and mlp, rotate and slice
        for i, inp in enumerate(inps):
            # first rotate activations based on previously computed Q
            new_inps = torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype)).cpu()
            # split activations by slicing ratio only for first layer
            if idx == 0:
                new_inps = expand_tensor_dim0(new_inps, slicing_scheduler.n_targets + 1).flatten(end_dim=1)
                # update ignore masks
                ignore_masks[i] = masks_reshape_fn(expand_tensor_dim0(ignore_masks[i], slicing_scheduler.n_targets + 1))
                if "attention_mask" in kwargs[i] and kwargs[i]["attention_mask"] is not None:
                    kwargs[i]["attention_mask"] = expand_tensor_dim0(kwargs[i]["attention_mask"], total_slices).flatten(end_dim=1)
            # truncate tensor to simulate different slicing targets    
            new_inps = truncate_tensor(
                unflatten_first_axis(slicing_scheduler.n_targets +1, new_inps), [slicing_scheduler.hidden_size] + slicing_scheduler.get_mlp_input_dimension(idx)
            )
            args[i] = layer_adapter.get_updated_args(
                new_inps.flatten(end_dim=1),
                args[i],
            )
        
        mlp_ln_inputs, _ = get_signals(layer_adapter, args, kwargs)
        # reshape based on type of pca calc by passing the appropriate lambda closure
        # since the "width" batch dimension is flattened (to keep the 3-axis shape B,L,D when computing the forward pass), it must be first unflattened!
        eig_val, Q = pca_fn(mlp_ln_inputs, ignore_masks, act_reshape_fn)
        Q = Q.to(device=config.device, dtype=torch.float64)

        rotations.append(Q.cpu())
        layer.attn_shortcut_Q = nn.Parameter(
            torch.matmul(
                layer.attn_shortcut_Q,
                Q.to(dtype=dtype)
            )
        )
        rotate_attention_output(layer_adapter, Q)

        layer.mlp_shortcut_Q = nn.Parameter(
            Q.T.clone().to(dtype=dtype)
        )
        rotate_mlp_input(layer_adapter, Q)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        # activations must be truncated in this case!
        target_sizes = slicing_scheduler.get_mlp_output_dimension(idx)
        if not isinstance(target_sizes, list):
            target_sizes = [target_sizes for _ in range(slicing_scheduler.n_targets)]
        _, inps = get_signals(layer_adapter, args, kwargs, lambda x: truncate_tensor(unflatten_first_axis(slicing_scheduler.n_targets + 1, x), [slicing_scheduler.hidden_size] + target_sizes).flatten(end_dim=1))

        # as for the mlp inputs, we need to reshape accordingly
        eig_val, Q = pca_fn(inps, ignore_masks, act_reshape_fn)

        rotations.append(Q.cpu())
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q.to(dtype=dtype)))

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q)

        layer.to('cpu')
        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")

    return rotations if isinstance(rotations, list) else []

@torch.no_grad()
def rotate_sequential_ecd_with_ridge(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    save_rotations: bool = False,
    pca_calc_method: str = "reduce_all",
    lambda_reg: float = 0.0,
    layers_for_attn_correction: str | list = [],
    layers_for_mlp_correction: str | list = [],
    save_ridge_updates: bool = False,
    ridge_nsamples = 8192,
    attn_modules_for_correction: list[str] = ["v_proj"],
    mlp_modules_for_correction: list[str] = [],
) -> None | list:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.

    Args:
        pca_calc_method (str): method with which we compute the PCA of activations. Accepted values:
        - "reduce_all": reduces both the batch and slice dimensions; given a calibration set with shape (B, N, L, D), where B is the number of examples and N is the number of model widths considered, the activations are reduced on both B and N. The effective calibration set shape is (BN, L, D).
        - "reduce_batch": reduces only the batch size; taking a calibration set with shape (B, N, L, D), only B is reduced while sequences all concatenated across the widths dimension. The effective calibration set shape is (B, NL, D).
    """

    class passer:
        def __getattr__(self, name):
            return self
        def __call__(self, *args, **kwds):
            pass
    if save_rotations:
        rotations = []
    else:
        rotations = passer()
    ridge_updates = dict()
    
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype
    total_slices = slicing_scheduler.n_targets + 1

    inps, args, kwargs, ignore_masks = [], [], [], []
    # we want to extend the calibration set by duplicating the activations for each target width we are using.
    # for instance: say the cal. set has shape (B, L, D) = (128, 2048, 768). we would like to extend for 2 target widths: {0.7, 0.8}. Then the new dataset would have shape (2, B, L, D'). The catch is that, for dataset[0], D' = 0.7 * D, while for dataset[1] D' = 0.8 * D.
    # Since tensors cannot have inhomogeneous dimensions, we can obtain the same effect of slicing weight matrices by setting to zero the appropriate number of elements at the end of each token, depending on the target compression ratio.
    def truncate_tensor(tensor: torch.Tensor, effective_last_shape: list[int]):
        assert len(effective_last_shape) == tensor.shape[0]
        # example of how to get the mask:
        # tensor.shape: (N, B, L, D)
        # effective_last_shape: (N, d)
        # suppose N = 2, D = 5, effective_last_shape = [2, 3]
        # first, build indices (broadcastable)
        # indices: torch.tensor[[0,1,2,3,4], [0,1,2,3,4]]
        indices = torch.arange(tensor.shape[-1], device=tensor.device).expand(tensor.shape[0], -1)
        # get mask telling which indices are above the threshold:
        # mask: torch.tensor[[0,0,1,1,1], [0,0,0,1,1]]
        mask = indices >= torch.tensor(effective_last_shape, device=tensor.device).unsqueeze(-1)
        # make dimensions broadcastable
        broadcastable_mask = mask.view([tensor.shape[0]] + [1 for _ in range(len(tensor.shape) - 2)] + [tensor.shape[-1]])
        # finally fill positions in the mask with zero
        return tensor.masked_fill(broadcastable_mask, 0.0)

    def expand_tensor_dim0(tensor: torch.Tensor, expand_size: int):
        return tensor.expand([expand_size] + [-1 for _ in range(len(tensor.shape))])
    
    def unflatten_first_axis(axis_shape, tensor):
        return tensor.view(axis_shape, -1, *(tensor.shape[1:]))

    @torch.no_grad() 
    def apply_ridge_update_to_linear(linear_module, w_ridge):
        device = linear_module.weight.device
        dtype = linear_module.weight.dtype
        linear_module.weight.copy_((w_ridge.mT.to(device) @ linear_module.weight.to(w_ridge.dtype)).to(device=device, dtype=dtype))
    
    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=False)

    # function for reshaping activations in case of "reduce_batch" pca method
    def reduce_batch_act_reshape(x: torch.Tensor) -> torch.Tensor:
        full_x = unflatten_first_axis(total_slices, x)
        B, D = full_x.shape[1], full_x.shape[-1]
        # use reshape since transposing will break contiguity
        return full_x.transpose(0,1).reshape(B, -1, D)

    pca_fn = pca_calc
    if pca_calc_method == "reduce_all":
        act_reshape_fn = lambda x: x
        masks_reshape_fn = lambda m: m.flatten(end_dim=1)
    elif pca_calc_method == "reduce_batch":
        act_reshape_fn = reduce_batch_act_reshape
        masks_reshape_fn = lambda m: m.transpose(0,1).flatten(start_dim=1)
    elif pca_calc_method == "experimental":
        act_reshape_fn = lambda x: unflatten_first_axis(total_slices, x)
        masks_reshape_fn = lambda m: m
        pca_fn = pca_calc_2
    
    if layers_for_attn_correction == "all":
        layers_for_attn_correction = list(range(len(layers)))
    if layers_for_mlp_correction == "all":
        layers_for_mlp_correction = list(range(len(layers)))

    n_tokens = 0
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        n_tokens = n_tokens + inp_batch.shape[0] * inp_batch.shape[1]
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    # rotate and slice embeddings
    eig_val, Q = pca_calc(inps, ignore_masks)
    Q = Q.cpu()
    rotations.append(Q)
    rotate_embeddings(model_adapter, Q)

    ridge_dtype = torch.float32
    dev = torch.device("cuda")

    # Rotation matrix between embeddings-first layer is always the same, indipendent of the slicing size
    logging.info("Rotate and slice layers")
    full_inps = list(inps)
    L, D = full_inps[0].shape[-2:]
    full_model_args = list(args)
    full_model_kwargs = list(kwargs)
    # clone attention masks (that will need to be expanded) and leave all other tensors the same (saves memory)
    for i in range(len(full_model_kwargs)):
        # swallow copy dict so we can change one tensor and leave the rest untouched
        kwargs_dict = dict(full_model_kwargs[i])
        if "attention_mask" in kwargs_dict and kwargs_dict["attention_mask"] is not None:
            kwargs_dict["attention_mask"] = kwargs_dict["attention_mask"].clone()
        full_model_kwargs[i] = kwargs_dict
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        for i in range(len(full_inps)):
            full_inp = full_inps[i]
            full_model_args[i] = layer_adapter.get_updated_args(
                full_inp,
                args[i]
            )
        
        # will be needed to check which layers to
        layer_mods_names = dict((m, f"{idx}.{k}") for k,m in layer.named_modules())
            
        attn_inputs_mods_for_ridge = list()
        mlp_inputs_mods_for_ridge = list()
        modules_for_output_collect = list()
        should_compute_attn_inputs_ridge = False
        should_compute_attn_output_ridge = False
        should_compute_mlp_inputs_ridge = False
        should_compute_mlp_output_ridge = False
        if idx in layers_for_attn_correction:
            if isinstance(attn_modules_for_correction, str) and attn_modules_for_correction == "all":
                attn_inputs_mods_for_ridge.extend(layer_adapter.get_attention_inputs())
                should_compute_attn_inputs_ridge = True
                should_compute_attn_output_ridge = True
            else:
                attn_inputs_mods_for_ridge.extend(filter(lambda m: any(mod in layer_mods_names[m] for mod in attn_modules_for_correction), layer_adapter.get_attention_inputs()))
                should_compute_attn_inputs_ridge = len(attn_inputs_mods_for_ridge) > 0
                should_compute_attn_output_ridge = any(mod in layer_mods_names[layer_adapter.get_attention_output()] for mod in attn_modules_for_correction)
            modules_for_output_collect.extend(attn_inputs_mods_for_ridge + ([layer_adapter.get_attention_output()] if should_compute_attn_output_ridge else []))
                    
        if idx in layers_for_mlp_correction:
            if isinstance(mlp_modules_for_correction, str) and mlp_modules_for_correction == "all":
                mlp_output = [layer_adapter.get_mlp_output()] if ((idx < (len(layers) - 1)) or slicing_scheduler.do_slice_head) else []
                mlp_inputs_mods_for_ridge.extend(layer_adapter.get_mlp_inputs() + mlp_output)
                should_compute_mlp_inputs_ridge = True
                should_compute_mlp_output_ridge = len(mlp_output) > 0
            else:
                mlp_inputs_mods_for_ridge.extend(filter(lambda m: any(mod in layer_mods_names[m] for mod in mlp_modules_for_correction), layer_adapter.get_mlp_inputs()))
                should_compute_mlp_inputs_ridge = len(mlp_inputs_mods_for_ridge) > 0
                should_compute_mlp_output_ridge = any(mod in layer_mods_names[layer_adapter.get_mlp_output()] for mod in mlp_modules_for_correction) and ((idx < (len(layers) - 1)) or slicing_scheduler.do_slice_head)
            modules_for_output_collect.extend(mlp_inputs_mods_for_ridge + ([layer_adapter.get_mlp_output()] if should_compute_mlp_output_ridge else []))
        
        # does not matter if these activations are not computed with the full rotated model or a partially rotated model. these activations should not change if computed on the "full" inputs
        layer_activations, full_inps = get_modules_and_layer_outputs_signals(modules_for_output_collect, layer_adapter, full_model_args, full_model_kwargs)

        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(device=config.device, dtype=dtype))

        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer_adapter, Q)

        # get signal between attention and mlp, rotate and slice
        target_sizes = [slicing_scheduler.hidden_size] + slicing_scheduler.get_mlp_input_dimension(idx)
        new_inps = []
        for i in range(len(inps)):
            new_inp = inps.pop(0)
            # first rotate activations based on previously computed Q
            new_inp = torch.matmul(new_inp.to(device=config.device), Q.to(device=config.device, dtype=dtype)).cpu()
            # split activations by slicing ratio only for first layer
            if idx == 0:
                new_inp = expand_tensor_dim0(new_inp, total_slices).flatten(end_dim=1)
                # update ignore masks
                ignore_masks[i] = expand_tensor_dim0(ignore_masks[i], total_slices)
                if "attention_mask" in kwargs[i] and kwargs[i]["attention_mask"] is not None:
                    kwargs[i]["attention_mask"] = expand_tensor_dim0(kwargs[i]["attention_mask"], total_slices).flatten(end_dim=1)
            # truncate tensor to simulate different slicing targets    
            new_inp = truncate_tensor(
                unflatten_first_axis(total_slices, new_inp), target_sizes
            )
            new_inps.append(new_inp)
            args[i] = layer_adapter.get_updated_args(
                new_inp.flatten(end_dim=1),
                args[i],
            )
            
        if should_compute_attn_inputs_ridge:
            first_rmsn = layer_adapter.get_first_layernorm()
            use_all_tokens = ridge_nsamples >= n_tokens
            if use_all_tokens:
                logging.warning(f"layer {idx} attn-input ridge correction: ridge_nsamples ({ridge_nsamples}) >= available calibration tokens ({n_tokens}); using all available tokens instead of subsampling")
                selected_tokens = [first_rmsn(batch).flatten(start_dim=1, end_dim=2) for batch in new_inps]
            else:
                # uniform distribution without replacement trick
                tokens_subset = torch.randint(low=0, high=n_tokens - ridge_nsamples, size=(ridge_nsamples,))
                tokens_subset = torch.sort(tokens_subset, axis=0).values + torch.arange(ridge_nsamples)
                # dev = new_inps[0].device
                selected_tokens = []
                subset = tokens_subset
                processed_tokens = 0
                cut_idx = 0
                for batch in new_inps:
                    subset = subset[cut_idx:]
                    # select only activations from sliced models
                    batch = first_rmsn(batch)
                    batch = batch.flatten(start_dim=1, end_dim=2)
                    # after concatenating the batch dim and the seqlen dim (which are dim 1 and dim 2 in this case, since examples are further batched by slicing factor), the tot. number of tokens B*L is at shape 1
                    next_processed_tokens = processed_tokens + batch.shape[1]
                    cut_idx = (subset < next_processed_tokens).sum()
                    idxs_to_select = subset[:cut_idx] - processed_tokens
                    selected_tokens.append(batch[:,idxs_to_select])
                    processed_tokens = next_processed_tokens
            # concatenate along dim 1, dim 0 is for the slicing expansion
            X_s = torch.cat(selected_tokens, dim=1).flatten(end_dim=1).to(device=dev, dtype=ridge_dtype)
            new_inps = None

            for attn_proj in attn_inputs_mods_for_ridge:
                W = attn_proj.weight.data.to(device=dev, dtype=ridge_dtype)
                if attn_proj.bias is not None:
                    b = attn_proj.bias.data.to(device=dev, dtype=ridge_dtype)
                else:
                    b = torch.zeros(W.shape[0], device=dev, dtype=ridge_dtype)
                acts = layer_activations[attn_proj]
                if use_all_tokens:
                    n_acts = len(acts)
                    target_tokens = []
                    for _ in range(n_acts):
                        y = acts.pop(0)
                        # flatten tokens across sequences
                        y = y.view(-1, y.shape[-1])
                        target_tokens.append(y)
                else:
                    subset = tokens_subset
                    processed_tokens = 0
                    cut_idx = 0
                    target_tokens = []
                    for i in range(len(acts)):
                        subset = subset[cut_idx:]
                        y = acts.pop(0)
                        # flatten tokens across sequences
                        y = y.view(-1, y.shape[-1])
                        # same as X_s, we want the total number of tokens B*L
                        next_processed_tokens = processed_tokens + y.shape[0]
                        cut_idx = (subset < next_processed_tokens).sum()
                        idxs_to_select = subset[:cut_idx] - processed_tokens
                        y = y[idxs_to_select]
                        # subtract bias: only considering W for regression
                        target_tokens.append(y)
                        processed_tokens = next_processed_tokens
                # we need to expand the targets to match the number of slicing factors we are considering; then we flatten
                Y = expand_tensor_dim0(torch.cat(target_tokens, dim=0), total_slices).flatten(end_dim=1).to(dev, ridge_dtype)
                # subtract bias: only considering W for regression
                Y = Y - b
                # W = attn_proj.weight.data.to(device=dev, dtype=ridge_dtype)
                print(f"=== layer {layer_mods_names[attn_proj]} regression ===")
                # print(f"Target matrix rank: {torch.linalg.matrix_rank(Y).item()}")
                # print(f"Source matrix rank: {torch.linalg.matrix_rank(X_s).item()}")
                # print(f"Weights rank: {torch.linalg.matrix_rank(W).item()}")
                pre_mse = (Y - X_s @ W.T).pow(2).mean().item()
                print(f"MSE before regression: {pre_mse}")
                ridge_W = ridge_regression_matrix_low_rank(Y, X_s, W, lambda_reg)
                attn_proj.weight.data.add_(ridge_W.to(dtype=attn_proj.weight.dtype, device=attn_proj.weight.device))
                post_mse = (Y - X_s @ attn_proj.weight.data.T.to(device=dev, dtype=ridge_dtype)).pow(2).mean().item()
                print(f"MSE after regression: {post_mse} (diff: {post_mse - pre_mse})")
                if save_ridge_updates:
                    ridge_updates[f"{layer_mods_names[attn_proj]}.ridge_W"] = ridge_W.cpu()

            Y = None
            X_s = None
            ridge_W = None
            selected_tokens = None
            target_tokens = None
            tokens_subset = None

        if should_compute_attn_output_ridge:
            # no need to set truncate_fn here: layer_args already has hidden states truncated accordingly
            new_inps = get_attn_output_inputs(layer_adapter, args, kwargs)
            use_all_tokens = ridge_nsamples >= n_tokens
            if use_all_tokens:
                logging.warning(f"layer {idx} attn-output ridge correction: ridge_nsamples ({ridge_nsamples}) >= available calibration tokens ({n_tokens}); using all available tokens instead of subsampling")
                selected_tokens = [batch.view(total_slices, -1, batch.shape[-1]) for batch in new_inps]
            else:
                # uniform distribution without replacement trick
                tokens_subset = torch.randint(low=0, high=n_tokens - ridge_nsamples, size=(ridge_nsamples,))
                tokens_subset = torch.sort(tokens_subset, axis=0).values + torch.arange(ridge_nsamples)
                # dev = new_inps[0].device
                selected_tokens = []
                subset = tokens_subset
                processed_tokens = 0
                cut_idx = 0
                for batch in new_inps:
                    subset = subset[cut_idx:]
                    batch = batch.view(total_slices, -1, batch.shape[-1])
                    # after concatenating the batch dim and the seqlen dim (which are dim 1 and dim 2 in this case, since examples are further batched by slicing factor), the tot. number of tokens B*L is at shape 1
                    next_processed_tokens = processed_tokens + batch.shape[1]
                    cut_idx = (subset < next_processed_tokens).sum()
                    idxs_to_select = subset[:cut_idx] - processed_tokens
                    selected_tokens.append(batch[:,idxs_to_select])
                    processed_tokens = next_processed_tokens
            # concatenate along dim 1, dim 0 is for the slicing expansion
            X_s = torch.cat(selected_tokens, dim=1).flatten(end_dim=1).to(device=dev, dtype=ridge_dtype)
            new_inps = None

            attn_proj = layer_adapter.get_attention_output()
            W = attn_proj.weight.data.to(device=dev, dtype=ridge_dtype)
            if attn_proj.bias is not None:
                b = attn_proj.bias.data.to(device=dev, dtype=ridge_dtype)
            else:
                b = torch.zeros(W.shape[0], device=dev, dtype=ridge_dtype)
            acts = layer_activations[attn_proj]
            if use_all_tokens:
                n_acts = len(acts)
                target_tokens = []
                for _ in range(n_acts):
                    y = acts.pop(0)
                    # flatten tokens across sequences
                    y = y.view(-1, y.shape[-1])
                    target_tokens.append(y)
            else:
                subset = tokens_subset
                processed_tokens = 0
                cut_idx = 0
                target_tokens = []
                for i in range(len(acts)):
                    subset = subset[cut_idx:]
                    y = acts.pop(0)
                    # flatten tokens across sequences
                    y = y.view(-1, y.shape[-1])
                    # same as X_s, we want the total number of tokens B*L
                    next_processed_tokens = processed_tokens + y.shape[0]
                    cut_idx = (subset < next_processed_tokens).sum()
                    idxs_to_select = subset[:cut_idx] - processed_tokens
                    y = y[idxs_to_select]
                    # subtract bias: only considering W for regression
                    target_tokens.append(y)
                    processed_tokens = next_processed_tokens
            # we need to expand the targets to match the number of slicing factors we are considering; then we flatten
            Y = expand_tensor_dim0(torch.cat(target_tokens, dim=0), total_slices).flatten(end_dim=1).to(dev, ridge_dtype)
            # subtract bias: only considering W for regression
            Y = Y - b
            # W = attn_proj.weight.data.to(device=dev, dtype=ridge_dtype)
            print(f"=== layer {layer_mods_names[attn_proj]} regression ===")
            # print(f"Target matrix rank: {torch.linalg.matrix_rank(Y).item()}")
            # print(f"Source matrix rank: {torch.linalg.matrix_rank(X_s).item()}")
            # print(f"Weights rank: {torch.linalg.matrix_rank(W).item()}")
            pre_mse = (Y - X_s @ W.T).pow(2).mean().item()
            print(f"MSE before regression: {pre_mse}")
            ridge_W = ridge_regression_matrix_low_rank(Y, X_s, W, lambda_reg)
            attn_proj.weight.data.add_(ridge_W.to(dtype=attn_proj.weight.dtype, device=attn_proj.weight.device))
            post_mse = (Y - X_s @ attn_proj.weight.data.T.to(device=dev, dtype=ridge_dtype)).pow(2).mean().item()
            print(f"MSE after regression: {post_mse} (diff: {post_mse - pre_mse})")
            if save_ridge_updates:
                ridge_updates[f"{layer_mods_names[attn_proj]}.ridge_W"] = ridge_W.cpu()
            
            Y = None
            X_s = None
            ridge_W = None
            selected_tokens = None
            target_tokens = None
            tokens_subset = None
        
        cleanup_memory()

        mlp_ln_inputs, _ = get_signals(layer_adapter, args, kwargs)
        # reshape based on type of pca calc by passing the appropriate lambda closure
        # since the "width" batch dimension is flattened (to keep the 3-axis shape B,L,D when computing the forward pass), it must be first unflattened!
        eig_val, Q = pca_fn(mlp_ln_inputs, ignore_masks, act_reshape_fn, masks_reshape_fn)
        Q = Q.cpu()

        rotations.append(Q)
        layer.attn_shortcut_Q = nn.Parameter(
            torch.matmul(
                layer.attn_shortcut_Q,
                Q.to(device=layer.attn_shortcut_Q.device, dtype=dtype)
            )
        )
        rotate_attention_output(layer_adapter, Q)
        # after rotation

        layer.mlp_shortcut_Q = nn.Parameter(
            Q.T.clone().to(device=config.device, dtype=dtype)
        )
        rotate_mlp_input(layer_adapter, Q)
        Q = Q.to(config.device, dtype=dtype)
        for i in range(len(mlp_ln_inputs)):
            x = mlp_ln_inputs[i]
            x = truncate_tensor(unflatten_first_axis(total_slices, x), target_sizes)
            mlp_ln_inputs[i] = torch.matmul(x.to(Q.device), Q)
        Q = Q.cpu()

        if should_compute_mlp_inputs_ridge:
            second_rmsn = layer_adapter.get_second_layernorm()
            # dev = mlp_ln_inputs[0].device
            use_all_tokens = ridge_nsamples >= n_tokens
            if use_all_tokens:
                logging.warning(f"layer {idx} mlp-input ridge correction: ridge_nsamples ({ridge_nsamples}) >= available calibration tokens ({n_tokens}); using all available tokens instead of subsampling")
                selected_tokens = [second_rmsn(batch).flatten(start_dim=1, end_dim=2) for batch in mlp_ln_inputs]
            else:
                tokens_subset = torch.randint(low=0, high=n_tokens - ridge_nsamples, size=(ridge_nsamples,))
                tokens_subset = torch.sort(tokens_subset, axis=0).values + torch.arange(ridge_nsamples)
                selected_tokens = []
                subset = tokens_subset
                processed_tokens = 0
                cut_idx = 0
                for batch in mlp_ln_inputs:
                    subset = subset[cut_idx:]
                    # select only activations from sliced models
                    batch = second_rmsn(batch)
                    batch = batch.flatten(start_dim=1, end_dim=2)
                    next_processed_tokens = processed_tokens + batch.shape[1]
                    cut_idx = (subset < next_processed_tokens).sum()
                    # shift to 0-based indexing
                    idxs_to_select = subset[:cut_idx] - processed_tokens
                    selected_tokens.append(batch[:, idxs_to_select])
                    processed_tokens = next_processed_tokens
            X_s = torch.cat(selected_tokens, dim=1).flatten(end_dim=1).to(device=dev, dtype=ridge_dtype)

            for mlp_in in mlp_inputs_mods_for_ridge:
                W = mlp_in.weight.data.to(dev, ridge_dtype)
                if mlp_in.bias is not None:
                    b = mlp_in.bias.data.to(device=dev, dtype=ridge_dtype)
                else:
                    b = torch.zeros(W.shape[0], device=dev, dtype=ridge_dtype)
                acts = layer_activations[mlp_in]
                # embedding size in MLP is different from model's hiddend size D
                mlp_d = acts[0].shape[-1]
                if use_all_tokens:
                    n_acts = len(acts)
                    target_tokens = []
                    for _ in range(n_acts):
                        # reshape in case acts are still batched
                        y = acts.pop(0).view(-1, mlp_d)
                        target_tokens.append(y)
                else:
                    subset = tokens_subset
                    processed_tokens = 0
                    cut_idx = 0
                    target_tokens = []
                    for i in range(len(acts)):
                        # reshape in case acts are still batched
                        y = acts.pop(0).view(-1, mlp_d)
                        subset = subset[cut_idx:]
                        next_processed_tokens = processed_tokens + y.shape[0]
                        cut_idx = (subset < next_processed_tokens).sum()
                        idxs_to_select = subset[:cut_idx] - processed_tokens
                        y = y[idxs_to_select]
                        target_tokens.append(y)
                        processed_tokens = next_processed_tokens
                Y = expand_tensor_dim0(torch.cat(target_tokens, dim=0), total_slices).flatten(end_dim=1).to(device=dev, dtype=ridge_dtype)
                # once again, subtract the bias
                Y = Y - b

                # W = mlp_in.weight.data.to(dev, ridge_dtype)
                print(f"=== layer {layer_mods_names[mlp_in]} regression ===")
                # print(f"Target matrix rank: {torch.linalg.matrix_rank(Y).item()}")
                # print(f"Source matrix rank: {torch.linalg.matrix_rank(X_s).item()}")
                # print(f"Weights rank: {torch.linalg.matrix_rank(W).item()}")
                pre_mse = (Y - X_s @ W.T).pow(2).mean().item()
                print(f"MSE before regression: {pre_mse}")
                ridge_W = ridge_regression_matrix_low_rank(Y, X_s, W, lambda_reg)
                mlp_in.weight.data.add_(ridge_W.to(dtype=mlp_in.weight.dtype, device=mlp_in.weight.device))
                post_mse = (Y - X_s @ mlp_in.weight.data.T.to(device=dev, dtype=ridge_dtype)).pow(2).mean().item()
                print(f"MSE after regression: {post_mse} (diff: {post_mse - pre_mse})")
                # if save_ridge_updates:
                #     ridge_updates[f"{layer_mods_names[mlp_in]}.ridge_W"] = ridge_W.cpu()

            Y = None
            X_s = None
            ridge_W = None
            selected_tokens = None
            target_tokens = None
            tokens_subset = None
            
        if should_compute_mlp_output_ridge:
            target_sizes = [slicing_scheduler.hidden_size] + slicing_scheduler.get_mlp_input_dimension(idx)
            mlp_ln_inputs = get_mlp_output_inputs(layer_adapter, args, kwargs, truncate_fn=lambda x: truncate_tensor(unflatten_first_axis(total_slices, x), target_sizes).flatten(end_dim=1))

            use_all_tokens = ridge_nsamples >= n_tokens
            if use_all_tokens:
                logging.warning(f"layer {idx} mlp-output ridge correction: ridge_nsamples ({ridge_nsamples}) >= available calibration tokens ({n_tokens}); using all available tokens instead of subsampling")
                selected_tokens = [batch.view(total_slices, -1, batch.shape[-1]) for batch in mlp_ln_inputs]
            else:
                tokens_subset = torch.randint(low=0, high=n_tokens - ridge_nsamples, size=(ridge_nsamples,))
                tokens_subset = torch.sort(tokens_subset, axis=0).values + torch.arange(ridge_nsamples)
                selected_tokens = []
                subset = tokens_subset
                processed_tokens = 0
                cut_idx = 0
                for batch in mlp_ln_inputs:
                    subset = subset[cut_idx:]
                    # select only activations from sliced models
                    batch = batch.view(total_slices, -1, batch.shape[-1])
                    next_processed_tokens = processed_tokens + batch.shape[1]
                    cut_idx = (subset < next_processed_tokens).sum()
                    # shift to 0-based indexing
                    idxs_to_select = subset[:cut_idx] - processed_tokens
                    selected_tokens.append(batch[:, idxs_to_select])
                    processed_tokens = next_processed_tokens
            X_s = torch.cat(selected_tokens, dim=1).flatten(end_dim=1).to(device=dev, dtype=ridge_dtype)

            mlp_output = layer_adapter.get_mlp_output()
            W = mlp_output.weight.data.to(dev, ridge_dtype)
            if mlp_output.bias is not None:
                b = mlp_output.bias.data.to(device=dev, dtype=ridge_dtype)
            else:
                b = torch.zeros(W.shape[0], device=dev, dtype=ridge_dtype)
            acts = layer_activations[mlp_output]
            if use_all_tokens:
                n_acts = len(acts)
                target_tokens = []
                for _ in range(n_acts):
                    # reshape in case acts are still batched
                    y = acts.pop(0)
                    y = y.view(-1, y.shape[-1])
                    target_tokens.append(y)
            else:
                subset = tokens_subset
                processed_tokens = 0
                cut_idx = 0
                target_tokens = []
                for i in range(len(acts)):
                    # reshape in case acts are still batched
                    y = acts.pop(0)
                    y = y.view(-1, y.shape[-1])
                    subset = subset[cut_idx:]
                    next_processed_tokens = processed_tokens + y.shape[0]
                    cut_idx = (subset < next_processed_tokens).sum()
                    idxs_to_select = subset[:cut_idx] - processed_tokens
                    y = y[idxs_to_select]
                    target_tokens.append(y)
                    processed_tokens = next_processed_tokens
            Y = expand_tensor_dim0(torch.cat(target_tokens, dim=0), total_slices).flatten(end_dim=1).to(device=dev, dtype=ridge_dtype)
            # once again, subtract the bias
            Y = Y - b

            # W = mlp_output.weight.data.to(dev, ridge_dtype)
            print(f"=== layer {layer_mods_names[mlp_output]} regression ===")
            # print(f"Target matrix rank: {torch.linalg.matrix_rank(Y).item()}")
            # print(f"Source matrix rank: {torch.linalg.matrix_rank(X_s).item()}")
            # print(f"Weights rank: {torch.linalg.matrix_rank(W).item()}")
            pre_mse = (Y - X_s @ W.T).pow(2).mean().item()
            print(f"MSE before regression: {pre_mse}")
            ridge_W = ridge_regression_matrix_low_rank(Y, X_s, W, lambda_reg)
            mlp_output.weight.data.add_(ridge_W.to(dtype=mlp_output.weight.dtype, device=mlp_output.weight.device))
            post_mse = (Y - X_s @ mlp_output.weight.data.T.to(device=dev, dtype=ridge_dtype)).pow(2).mean().item()
            print(f"MSE after regression: {post_mse} (diff: {post_mse - pre_mse})")
            if save_ridge_updates:
                ridge_updates[f"{layer_mods_names[mlp_output]}.ridge_W"] = ridge_W.cpu()

            Y = None
            X_s = None
            ridge_W = None
            selected_tokens = None
            target_tokens = None
            tokens_subset = None
            
        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        # activations must be truncated in this case!
        mlp_out_sizes = slicing_scheduler.get_mlp_output_dimension(idx)
        if not isinstance(mlp_out_sizes, list):
            mlp_out_sizes = [mlp_out_sizes for _ in range(slicing_scheduler.n_targets)]
        target_sizes = [slicing_scheduler.hidden_size] + mlp_out_sizes
        _, inps = get_signals(layer_adapter, args, kwargs, lambda x: truncate_tensor(unflatten_first_axis(total_slices, x), target_sizes).flatten(end_dim=1))
        
        # as for the mlp inputs, we need to reshape accordingly
        eig_val, Q = pca_fn(inps, ignore_masks, act_reshape_fn, masks_reshape_fn)

        rotations.append(Q.cpu())
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q.to(device=layer.mlp_shortcut_Q.device, dtype=dtype)))

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q)

        layer.to('cpu')
        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")

    return rotations if save_rotations else [], ridge_updates

@torch.no_grad()
def rotate_and_slice_sequential(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
    save_rotations: bool = False
) -> None | list:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.
    """
    if save_rotations:
        rotations = []
    else:
        class passer:
            def __getattr__(self, name):
                return self
            def __call__(self, *args, **kwds):
                pass
                
        rotations = passer()
    
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=False)

    # rotate and slice embeddings
    eig_val, Q = pca_calc(inps, ignore_masks)
    Q = Q.to(device=config.device)
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)
    rotations.append(Q.cpu())
    rotate_embeddings(model_adapter, Q)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer_adapter, Q)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        # get signal between attention and mlp, rotate and slice
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
                    :, :, : slicing_scheduler.get_attention_input_dimension(idx)
                ].cpu(),
                args[i],
            )

        mlp_ln_inputs, _ = get_signals(layer_adapter, args, kwargs)
        eig_val, Q = pca_calc(mlp_ln_inputs, ignore_masks)
        Q = Q.to(device=config.device, dtype=torch.float64)
        if final_orientation == 'random':
            R = random_orthogonal_upper_left(
                Q.shape[0], slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
            )
            Q = Q @ R.to(Q.device)

        rotations.append(Q.cpu())
        layer.attn_shortcut_Q = nn.Parameter(
            torch.matmul(
                layer.attn_shortcut_Q,
                Q.to(dtype=dtype)[:, : slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)],
            )
        )
        rotate_attention_output(layer_adapter, Q)
        slice_attention_output(
            layer_adapter, slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
        )

        layer.mlp_shortcut_Q = nn.Parameter(
            Q.T.clone().to(dtype=dtype)[: slicing_scheduler.get_mlp_input_dimension(idx), :]
        )
        rotate_mlp_input(layer_adapter, Q)
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(idx))

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        _, inps = get_signals(layer_adapter, args, kwargs)
        eig_val, Q = pca_calc(inps, ignore_masks)
        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)

        rotations.append(Q.cpu())
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q.to(dtype=dtype)))

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)])

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")

    return rotations if isinstance(rotations, list) else []


@torch.no_grad()
def rotate_and_slice_parallel(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations

    This version works for models where the MLP block and the attention block are computed in parallel.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=True)

    # rotate and slice embeddings
    _, Q = pca_calc(inps, ignore_masks)
    Q = Q.to(device=config.device)
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)
    rotate_embeddings(model_adapter, Q)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    layers = model_adapter.get_layers()
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # rotate and slice the inputs to match previous layer (both attention and mlp)
        rotate_attention_inputs(layer_adapter, Q)
        rotate_mlp_input(layer_adapter, Q)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))
        slice_mlp_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        # update the input signals to this layer, and re-run it
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
                    :, :, : slicing_scheduler.get_attention_input_dimension(idx)
                ].cpu(),
                args[i],
            )

        # the simpler equivalent of get_signals
        outputs = []
        layer = layer.to(config.device)
        for layer_args_batch, layer_kwargs_batch in zip(args, kwargs):
            layer_args_batch, layer_kwargs_batch = map_tensors(
                [layer_args_batch, layer_kwargs_batch], device=config.device
            )
            out = layer(*layer_args_batch, **layer_kwargs_batch)
            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]
            out = out.cpu()
            outputs.append(out)

        inps = outputs
        _, Q = pca_calc(inps, ignore_masks)

        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)

        # update shortcut matrix
        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(layer.attn_shortcut_Q, Q.to(dtype=dtype)))

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q)
        rotate_attention_output(layer_adapter, Q)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        slice_attention_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))

        # slice the shortcut (there is only one, we use attn_shortcut buffer)
        layer.attn_shortcut_Q = nn.Parameter(
            layer.attn_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)]
        )

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate(model_adapter: ModelAdapter, dataloader: torch.utils.data.DataLoader[torch.Tensor]) -> None:
    """
    Rotate a model.
    TODO: Make this gpu memory efficient.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype  # Get the dtype of the model.

    # List of layers to rotate.
    layers = model_adapter.get_layers()

    # Get the input of the first layer norm and calculate the Q_1
    inps, args, kwargs = [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)

    _, Q_1 = pca_calc(inps)
    Q_1 = Q_1.to(device=config.device)

    # Rotate the embeddings.
    rotate_embeddings(model_adapter, Q_1)

    # Rotate the rest of the model.
    logging.info("Rotate layers")
    for layer_adapter in tqdm(layers, unit="layer", desc="Rotating"):
        layer = layer_adapter.layer
        # Extract the inputs and outputs of the second layernorm input and calculate the Q_3
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(inp, args[i])
        mlp_ln_inputs, outs = get_signals(layer_adapter, args, kwargs)
        _, Q_3 = pca_calc(mlp_ln_inputs)
        Q_3 = Q_3.to(device=config.device)
        _, Q_5 = pca_calc(outs)
        Q_5 = Q_5.to(device=config.device)

        # Rotate the Q, K and V matrices of the self-attention layer.
        rotate_attention_inputs(layer_adapter, Q_1)

        # Set the shortcut rotation matrix of the self-attention layer.
        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(Q_1.clone().T, Q_3.clone()).to(device="cpu", dtype=dtype))

        # Rotate the Attention output matrix
        rotate_attention_output(layer_adapter, Q_3)

        # Rotate the MLP input
        rotate_mlp_input(layer_adapter, Q_3)

        # Set the shortcut rotation matrix of the MLP.
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(Q_3.clone().T, Q_5.clone()).to(device="cpu", dtype=dtype))

        # Rotate MLP output
        rotate_mlp_output(layer_adapter, Q_5)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        inps = outs  # The inputs to the next layer are the outputs from this one!
        Q_1 = Q_5  # first rotation in the next layer is the last one in this...

    rotate_head(model_adapter, Q_5)
    logging.info("Rotate layers done")


def slice_rotated_model(model_adapter: ModelAdapter, slicing_scheduler: SlicingScheduler | None = None) -> None:
    """
    TODO: Make this gpu memory efficient.
    """
    model_adapter.model.eval()
    layers = model_adapter.get_layers()
    if not slicing_scheduler:
        if model_adapter.slicing_conf.const_dimension is not None:
            # backward compatibility for when no config is available
            slicing_scheduler = ConstSlicingScheduler(model_adapter.slicing_conf.const_dimension)
            slicing_scheduler.setup(
                hidden_size=model_adapter.hidden_size,
                layers_num=len(layers),
                parallel_blocks=model_adapter.parallel_blocks,
            )
        else:
            slicing_scheduler = ConfigSlicingScheduler(model_adapter.slicing_conf)

    # slice embeddings
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    # slice layers
    for i, layer_adapter in enumerate(layers):
        layer = layer_adapter.layer
        # slice attn weights 2nd dim, attn shortcut 1st dim
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(i))

        # slice mlp input 2nd dimension
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(i))

        # slice mlp shortcut 1st dimension
        # slice mlp shortcut
        if not model_adapter.parallel_blocks:
            layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[: slicing_scheduler.get_mlp_input_dimension(i), :])

        # slice mlp weights 1st dimension
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(i))

        if model_adapter.parallel_blocks:  # parallel case
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)]
            )
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)
            )
        else:  # sequential case
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)]
            )
            layer.mlp_shortcut_Q = nn.Parameter(
                layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(i)]
            )

            # slice attention weights 1st dimension
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)
            )

    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())


def random_orthogonal_upper_left(total_dim, upper_block_dim):
    """
    Create a square matrix where the upper left block is a random orthogonal matrix, and the remainder is the identity.
    """
    A = np.random.rand(upper_block_dim, upper_block_dim)
    Q, _ = np.linalg.qr(A)
    R = np.eye(total_dim)
    R[:upper_block_dim, :upper_block_dim] = Q
    return torch.from_numpy(R)


@torch.no_grad()
def pca_calc(
    X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None = None, reshape_fn = lambda x: x, mask_reshape_fn = lambda x: x
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run PCA on a list of batched data. Returns the eigenvalues and eigenvectors.
    """
    # Run GC and cleanup GPU memory
    cleanup_memory()

    H = None
    for idx, X_batch in enumerate(X):
        X_batch = reshape_fn(X_batch)
        if ignore_masks:
            X_batch[mask_reshape_fn(ignore_masks[idx]) == 0] = 0

        X_batch = X_batch.double().to(device=config.device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]
    return eig_val, eigen_vec


@torch.no_grad()
def pca_calc_2(
    X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None = None, reshape_fn = lambda x: x, mask_reshape_fn = lambda x: x
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run PCA on a list of batched data. Returns the eigenvalues and eigenvectors.
    """

    # Run GC and cleanup GPU memory
    cleanup_memory()

    H = []
    for idx, X_batch in enumerate(X):
        X_batch = reshape_fn(X_batch)
        assert X_batch.ndim == 4, f"Invalid axis number for batch: must be 4, found {len(X_batch.shape)}"
        if ignore_masks:
            X_batch[mask_reshape_fn(ignore_masks[idx]) == 0] = 0

        X_batch = X_batch.double().to(device=config.device)
        # sum covariance matrix over examples inside each width-set
        H.append((X_batch.mT @ X_batch).sum(dim=1))
    # stack all batches and put width-set dimension on first axis
    H = torch.stack(H).transpose(0,1)
    # sum all batches for each width-set
    H = H.sum(dim=1)
    # some magic to keep dims broadcastable
    damp = 0.01 * torch.mean(torch.diagonal(H, dim1=-2, dim2=-1), dim=-1, keepdim=True)
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[...,diag, diag] = H[...,diag, diag] + damp
    # batched PCA
    X_eig = torch.linalg.eigh(H)
    del H
    eigen_vec = X_eig.eigenvectors[0]
    for ev in X_eig.eigenvectors[1:]:
        eigen_vec = eigen_vec @ ev
    return None, eigen_vec

def ridge_regression_matrix_low_rank(Y: torch.Tensor, X_s: torch.Tensor, W: torch.Tensor, lambda_reg: float):
    # rhs = Y.mT @ X_s - W @ X_s.mT @ X_s
    # lhs = (X_s.mT @ X_s) + lambda_reg * torch.eye(X_s.shape[-1], dtype=X_s.dtype, device=X_s.device)
    # deltaW = torch.linalg.solve(lhs, rhs, left=False)
    # return deltaW

    # problem to solve: ||(Y - X_s @ W^T) - X_s @ dW^T||_F^2 + lambda * ||dW||_F^2
    # one obtains dW @ (X_s.mT @ X_s + lambda * I) = (Y - X_s @ W^T)^T @ X_s
    # X_s is low rank, so we use the pseudo-inverse to invert it
    rhs = Y.mT @ X_s - W @ X_s.mT @ X_s
    lhs = (X_s.mT @ X_s) + lambda_reg * torch.eye(X_s.shape[-1], dtype=X_s.dtype, device=X_s.device)
    deltaW = rhs @ torch.linalg.pinv(lhs, hermitian=True)
    return deltaW

