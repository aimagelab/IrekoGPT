# Copyright (c) 2026 Pietro Moriello
# Licensed under the MIT license.

import argparse
import json
import logging
import os
import pathlib
import shutil
from typing import Callable
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from slicegpt.slicing_scheduler import ConstSlicingScheduler

from itertools import chain
from collections import defaultdict
from contextlib import ExitStack
import tempfile
from safetensors.torch import save as safetensors_save
from safetensors import safe_open

VALID_METRICS = ("mse", "cka")

def compute_metrics_for_key(key, target_files, source_files_by_sparsity, metrics, device, dtype=torch.float64):
    """
    Streams a single module `key` across all already-open chunk files, one chunk at a time,
    accumulating every requested metric for every sparsity in one pass. This means each chunk
    (for the target, and for every sparsity's source) is read from disk and cast to `dtype`
    exactly once, no matter how many metrics are requested or how many sparsities are compared.
    """
    zero = lambda: torch.zeros((1,), device=device, dtype=dtype)
    cka_cross = defaultdict(zero)
    cka_source_gram = defaultdict(zero)
    cka_target_gram = defaultdict(zero)
    mse_sum = defaultdict(zero)
    mse_count = defaultdict(int)

    for chunk_idx, target_file in enumerate(target_files):
        t = target_file.get_tensor(key)
        t = t.view(-1, t.shape[-1]).to(device, dtype=dtype)
        for sparsity, source_files in source_files_by_sparsity.items():
            s = source_files[chunk_idx].get_tensor(key)
            s = s.view(-1, s.shape[-1]).to(device, dtype=dtype)
            if "cka" in metrics:
                cka_cross[sparsity] = cka_cross[sparsity] + (s.mT @ t)
                cka_source_gram[sparsity] = cka_source_gram[sparsity] + (s.mT @ s)
                cka_target_gram[sparsity] = cka_target_gram[sparsity] + (t.mT @ t)
            if "mse" in metrics:
                mse_sum[sparsity] = mse_sum[sparsity] + (s - t).pow(2).sum()
                mse_count[sparsity] = mse_count[sparsity] + s.numel()

    results_by_sparsity = defaultdict(dict)
    for sparsity in source_files_by_sparsity:
        if "cka" in metrics:
            results_by_sparsity[sparsity]["cka"] = (
                torch.norm(cka_cross[sparsity]).pow(2)
                / (torch.norm(cka_source_gram[sparsity]) * torch.norm(cka_target_gram[sparsity]))
            ).item()
        if "mse" in metrics:
            results_by_sparsity[sparsity]["mse"] = (mse_sum[sparsity] / mse_count[sparsity]).item()
    return results_by_sparsity

def _write_results_json(save_dir, results):
    """Atomically (over)writes the full results.json in its final format, so a crash mid-write
    never leaves a truncated/corrupt file behind for a later resume to choke on."""
    if save_dir is None:
        return None
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "results.json")
    results_json = json.dumps(_make_json_serializable(results), indent=2)
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(results_json)
    os.replace(tmp_path, save_path)
    return save_path

def _load_existing_metrics(save_dir, current_modules):
    """Loads the `metrics` section of a prior results.json at save_dir, if present, so a
    resumed run can skip module keys that were already fully computed."""
    if save_dir is None:
        return {}
    save_path = os.path.join(save_dir, "results.json")
    if not os.path.exists(save_path):
        return {}
    try:
        with open(save_path) as f:
            prior = json.load(f)
    except Exception as e:
        logging.warning(f"Could not parse existing results at {save_path} for resume, starting metrics fresh: {e}")
        return {}

    prior_modules = prior.get("metadata", {}).get("args", {}).get("modules")
    if prior_modules is not None and prior_modules != current_modules:
        logging.warning(
            f"Existing results at {save_path} were computed with different modules "
            f"({prior_modules}) than this run ({current_modules}); resume may be inconsistent."
        )
    return prior.get("metrics", {})

def _key_already_done(key, sparsities, metrics, existing_metrics):
    for sparsity in sparsities:
        sp_metrics = existing_metrics.get(str(sparsity))
        if not sp_metrics:
            return False
        for metric in metrics:
            if key not in sp_metrics.get(metric, {}):
                return False
    return True


def args_parser(interactive: bool = True) -> argparse.Namespace:
    class SplitArgs(argparse.Action):
        def __init__(self, map_type, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.map_type = map_type
        def __call__(self, parser, namespace, values, option_string=None):
            if values != "all":
                values = list(map(self.map_type, values.split(',')))
            setattr(namespace, self.dest, values)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Model type to load.",
    )
    parser.add_argument(
        "--target-model-state-path",
        type=str,
        help="Path to load the model state-dict from. Use this and the correct '--target-model-sparsity' to load sliced models from original SliceGPT.",
        default=None,
    )
    parser.add_argument(
        "--target-model-sparsity",
        type=float,
        default=0.0,
        help="Sparsity to apply to target model.",
    )
    parser.add_argument(
        "--target-model-round-interval",
        type=int,
        default=8,
        help="Round interval to apply to target model.",
    )
    parser.add_argument("--dtype", type=str, help="Data type to use.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to extract activations from and calculate perplexity on.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
    )
    parser.add_argument(
        "--dataset-local-path",
        type=str,
        help="Set this to load dataset from local path.",
        default=None,
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        help="Number of samples of the calibration data to load.",
        default=128,
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for loading the calibration data.")
    parser.add_argument(
        "--max-seqlen", type=int, default=2048, help="Maximum sequence length for the calibration data."
    )
    parser.add_argument(
        "--sparsities", action=lambda *args, **kwargs: SplitArgs(float, *args, **kwargs), default="1.0", 
        help="""Set of compressions used for eval/computing activations statistics.
 Should be interpreted as a ratio in [0,1] of model size reduction: for instance, '0.2' means '20 percent compression' (exaclty as original SliceGPT).
 The actual integer value will be obtained after rounding with the '--source-model-round-interval' argument."""
    )
    parser.add_argument(
        "--source-model-round-interval",
        type=int,
        default=8,
        help="Interval for rounding the weights (the best value may depend on your hardware)",
    )
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument("--save-dir", type=str, default=None, help="Path to save results.")

    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))

    parser.add_argument('--wandb-project', type=str, default="slicegpt", help="wandb project name.")
    parser.add_argument('--wandb-mode', type=str, default="disabled", help="Set wandb mode.")
    parser.add_argument(
        '--device',
        type=str,
        default="cuda",
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )

    parser.add_argument("--eval-source-model-only", action="store_true", default=False, help="Only run ppl evaluation on source model.")

    parser.add_argument("modules", nargs="*", type=str,  
    help=
    """List of patterns that specify what modules to consider for collecting activations.
All patterns must follow the structure \{input,output\}.<module_suffix>.
Every module name that ends with the suffix will be matched.
For instance: specifying 'v_proj' will match all v_proj modules in the model; if we wanted only layer.1's v_proj, we should specify
the longest suffix that uniquely identifies it, so: 'layers.1.self_attn.v_proj'."""
    )
    parser.add_argument(
        "--source-model-state-path",
        type=str,
        help="Path to load the model state-dict from. Use this and the correct '--target-model-sparsity' to load sliced models from original SliceGPT.",
        default=None,
    )
    parser.add_argument("--offload-activations-chunks", type=int, default=1, help="Controls how many chunks to use when offloading activations to disk.")
    parser.add_argument("--source-model-do-slice-head", action="store_true", default=False)
    parser.add_argument("--metrics", action=lambda *args, **kwargs: SplitArgs(str, *args, **kwargs), default=["mse", "cka"], help="Metrics to compute")

    return parser.parse_args() if interactive else parser.parse_args('')


def process_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if not 0 <= args.target_model_sparsity < 1:
        raise argparse.ArgumentTypeError(f"Sparsity should be in the range [0, 1)")

    if args.device:
        args.device = torch.device(args.device)

    if args.dtype:
        # retro-compatibility
        if args.dtype == "fp32":
            args.dtype = "float32"
        elif args.dtype == "fp16":
            args.dtype = "float16"

        dtype = getattr(torch, args.dtype, None)
        if dtype is None:
            raise argparse.ArgumentTypeError(f"Invalid datatype: torch.{args.dtype}")
        args.dtype = dtype
        

class ActivationsCollectionManager:

    def __init__(self, modules_to_hook: dict[str, torch.nn.Module], process_fn: None | dict[str, Callable], save_to_tmp=True):

        self.buffer = defaultdict(list)
        self.activations = list() 
        self.mods_to_names = dict((m,n) for n,m in modules_to_hook.items())
        self.process_fn = defaultdict(lambda: lambda x: x)
        if process_fn:
            self.process_fn.update(process_fn)
        
        self.handles = list()
        for key, module in modules_to_hook.items():
            if key.startswith("input"):
                handle = module.register_forward_pre_hook(self._input_hook_fn)
            elif key.startswith("output"):
                handle = module.register_forward_hook(self._output_hook_fn)
            else:
                raise ValueError(f"Formatting error: key should start with either 'input' or 'output', but found '{key}'")
            self.handles.append(handle)
            
        self.save_to_tmp = True

        
    def _output_hook_fn(self, module, args, output):
        name = self.mods_to_names[module]
        self.buffer[name].append(self.process_fn[name](output).detach().to("cpu", copy=True))
    def _input_hook_fn(self, module, args):
        name = self.mods_to_names[module]
        self.buffer[name].append(self.process_fn[name](args).detach().to("cpu", copy=True))

    def save_buffer(self):
        """
        Saves current buffers. If save_to_tmp was set, will create a tmp file entry
        """
        acts = dict()
        keys = list(self.buffer.keys())
        for k in keys:
            acts[k] = torch.cat(self.buffer.pop(k), dim=0)
        if self.save_to_tmp:
            acts = self._save_to_tmp_fn(acts)

            self.activations.append(acts)
            
    def remove(self):
        self.buffer = None
        del self.buffer
        for h in self.handles:
            h.remove()
    
    def clear(self):
        for act in self.activations:
            if isinstance(act, str):
                os.unlink(act)
        self.activations = list()
    
    def _save_to_tmp_fn(self, acts):
        tmp_dir = os.environ.get("SLICEGPT_TMPDIR")
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=tmp_dir) as tmp:
            tmp.write(safetensors_save(acts))
            filename = tmp.name
        return filename
        

@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module, pad_token_id: int | None, testloader: DataLoader[dict[str, torch.Tensor]], post_batch_processed_hook=None, device="cuda") -> float:
    """
    Evaluate the model's perplexity on the test set using batch processing.
    It is expected that model is already on the correct device.
    """
    gpu_utils.sync_gpus()

    start_time = time.time()

    model.eval()

    if pad_token_id:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    logging.info("Evaluating perplexity...")
    for batch_idx, batch in enumerate(testloader):
        logging.debug(f"Evaluating batch {len(nlls)}")
        batch = utils.map_tensors(batch, device)
        logits = model(**batch).logits

        if post_batch_processed_hook:
            post_batch_processed_hook(batch_idx)
        # shift outputs and labels autoregressively.
        logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]

        # CrossEntropyLoss demands data dimension is dimension 1.
        nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()

        mask = shift_labels != loss_fn.ignore_index
        nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
        nlls.append(nll_means)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())

    gpu_utils.sync_gpus()

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    return ppl.item()

def _inputs_process_fn(inputs):
    if isinstance(inputs, tuple):
        inputs = inputs[0]
    return inputs

def attach_activation_manager(model, modules_names_to_hook):
    all_mods = dict(model.named_modules())
    mods_to_hook = dict()
    process_fn = dict()
    for pattern in modules_names_to_hook:
        if not pattern.startswith("input.") and not pattern.startswith("output."):
            raise ValueError(f"Pattern should start with either 'input.' or 'output.', but found: '{pattern}'")
        hook_type_str, mod_name = pattern.split('.', maxsplit=1)
        for k in all_mods.keys():
            if k.endswith(mod_name):
                if k not in mods_to_hook:
                    key = '.'.join([hook_type_str, k])
                    mods_to_hook[key] = all_mods[k]
                    if hook_type_str == "input":
                        process_fn[key] = _inputs_process_fn
                else:
                    logging.info(f"Pattern '{key}' already taken: skipping...")

    return ActivationsCollectionManager(mods_to_hook, process_fn) 

class NestedDict(dict):
    def __missing__(self, key):
        self[key] = NestedDict()
        return self[key]


def _make_json_serializable(obj):
    """Convert an object to a JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, (defaultdict, NestedDict)):
        return _make_json_serializable(dict(obj))
    else:
        return obj

def main(args: argparse.Namespace) -> None:
    try:
        wandb.init(project=args.wandb_project, config=args, mode=args.wandb_mode)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    # Collect args metadata
    args_dict = vars(args).copy()
    # Convert non-serializable types to strings
    for k, v in args_dict.items():
        if isinstance(v, torch.device):
            args_dict[k] = str(v)
        elif isinstance(v, torch.dtype):
            args_dict[k] = str(v)
    results = NestedDict()
    results["metadata"]["args"] = args_dict

    for metric in args.metrics:
        if metric not in VALID_METRICS:
            raise ValueError(f"Unknown metric: '{metric}'")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=args.hf_token)
    logging.info("Loading dataset.")
    dataset = data_utils.get_dataset(args.dataset, args.dataset_local_path)["test"]
    logging.info("Preparing dataloader...")
    loader = data_utils.prepare_test_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        seqlen=args.max_seqlen,
        batch_size=args.batch_size,
    )

    n_minibatches = len(loader)
    minibatches_in_chunk = n_minibatches // args.offload_activations_chunks
    when_to_save = [i*minibatches_in_chunk - 1 for i in range(1, args.offload_activations_chunks)]
    when_to_save.append(n_minibatches - 1)
    def _chunk_activations(activations_mgr):
        def _offload_to_file(batch_idx):
            if batch_idx in when_to_save: 
                activations_mgr.save_buffer()
        return _offload_to_file

    target_activations_manager = None
    if not args.eval_source_model_only:
        # load the model from sliced_model_path to compute perplexity and skip rotation and slicing
        if args.target_model_state_path:
            target_model_adapter, tokenizer = hf_utils.load_rotated_model_state(
                args.model,
                args.target_model_state_path,
                dtype=args.dtype,
                sparsity=args.target_model_sparsity,
                round_interval=args.target_model_round_interval,
                token=args.hf_token,
            )
        else:
            target_model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
                args.model,
                dtype=args.dtype,
                token=args.hf_token
            )

        target_model = target_model_adapter.model
        target_model.eval()

        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(target_model_adapter)
        else:
            target_model.to(args.device)

        target_activations_manager = attach_activation_manager(target_model, args.modules)


        offload_to_file_fn = None
        if args.offload_activations_chunks:
            offload_to_file_fn = _chunk_activations(target_activations_manager)

        logging.info(f"Collecting activations for target model and computing perplexity...")
        target_ppl = evaluate_ppl(target_model, tokenizer.pad_token_id, loader, offload_to_file_fn, args.device)
        logging.info(f"Target model perplexity: {target_ppl}")
        results["target_model"]["ppl"] = target_ppl
        
        target_activations_manager.remove()

        target_model = None
        target_model_adapter = None
        utils.cleanup_memory()

    # Evaluate the source model at each sparsity via genuine weight-slicing
    source_managers_by_sparsity = {}
    for sparsity in args.sparsities:
        logging.info(f"Loading source model at sparsity '{sparsity}' (true slicing)...")
        source_model_adapter, _ = hf_utils.load_rotated_model_state(
            args.model,
            args.source_model_state_path,
            dtype=args.dtype,
            sparsity=sparsity,
            round_interval=args.source_model_round_interval,
            do_slice_head=args.source_model_do_slice_head,
            token=args.hf_token,
        )

        source_model = source_model_adapter.model
        source_model.eval()
        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(source_model_adapter)
        else:
            source_model.to(args.device)

        offload_to_file_fn = None
        source_activations_manager = None
        if not args.eval_source_model_only:
            source_activations_manager = attach_activation_manager(source_model, args.modules)
            if args.offload_activations_chunks:
                offload_to_file_fn = _chunk_activations(source_activations_manager)

        logging.info("Evaluating source model...")
        ppl = evaluate_ppl(source_model, tokenizer.pad_token_id, loader, offload_to_file_fn, args.device)
        logging.info(f"Source model perplexity at '{sparsity}' sparsity: {ppl}")
        results["source_model"][str(sparsity)]["ppl"] = ppl

        if source_activations_manager:
            source_activations_manager.remove()
            source_managers_by_sparsity[str(sparsity)] = source_activations_manager

        # Each sparsity is a differently-shaped model now (true slicing, not a shared masked
        # model), so it must be freed before the next one is constructed rather than reused.
        source_model = None
        source_model_adapter = None
        utils.cleanup_memory()

    # Load any already-computed metrics from a prior (e.g. timed-out) run at this save-dir
    # and merge them in BEFORE the next checkpoint write, so that write can't clobber them.
    existing_metrics = {}
    if target_activations_manager is not None:
        existing_metrics = _load_existing_metrics(args.save_dir, args.modules)
        for sparsity_str, metric_dict in existing_metrics.items():
            for metric_name, per_key in metric_dict.items():
                results["metrics"][sparsity_str][metric_name].update(per_key)

    # Save ppl results (and any resumed metrics) now, before the (potentially slow) metrics
    # phase, so a crash there doesn't lose them.
    _write_results_json(args.save_dir, results)

    if target_activations_manager is not None:
        keys = list(target_activations_manager.mods_to_names.values())

        logging.info(
            f"Computing metrics {args.metrics} for {len(keys)} module(s) "
            f"across {len(source_managers_by_sparsity)} sparsity level(s)..."
        )

        # Every chunk file (target's, and each sparsity's source) is opened once up front and
        # kept open for the whole metrics phase, instead of being reopened per metric/per key:
        # this is what lets a module key's chunk be read from disk exactly once and reused for
        # every requested metric, and lets the target's chunks be reused across every sparsity.
        with ExitStack() as stack:
            target_files = [
                stack.enter_context(safe_open(p, framework="pt")) for p in target_activations_manager.activations
            ]
            source_files_by_sparsity = {
                sparsity_str: [stack.enter_context(safe_open(p, framework="pt")) for p in mgr.activations]
                for sparsity_str, mgr in source_managers_by_sparsity.items()
            }

            for key in keys:
                if _key_already_done(key, args.sparsities, args.metrics, existing_metrics):
                    logging.info(f"Skipping already-computed key '{key}' (resumed from existing results).")
                    continue

                per_sparsity = compute_metrics_for_key(
                    key, target_files, source_files_by_sparsity, args.metrics, args.device
                )
                for sparsity_str, metric_values in per_sparsity.items():
                    for metric_name, value in metric_values.items():
                        results["metrics"][sparsity_str][metric_name][key] = value

                # Rewritten in full after every key, so a crash never loses more than the
                # keys not yet processed, and a resumed run can pick up where this left off.
                _write_results_json(args.save_dir, results)
                logging.info(f"Computed metrics for key '{key}'.")

                # PyTorch's caching allocator doesn't return freed blocks to the OS between
                # calls, so across many keys of varying width the cached-but-idle pool creeps
                # up over the run even though any single key's live tensors are modest.
                utils.cleanup_memory()

        for mgr in source_managers_by_sparsity.values():
            mgr.clear()
        target_activations_manager.clear()

    save_path = _write_results_json(args.save_dir, results)
    if save_path:
        logging.info(f"Results saved to {save_path}")

    logging.info(f"Results:\n{json.dumps(_make_json_serializable(results), indent=2)}")
    logging.info("Completed.")
        
if __name__ == "__main__":
    import os
    loglevel = os.environ.get("LOGLEVEL", "INFO")
    utils.configure_logging(log_to_console=True, log_to_file=False, level=getattr(logging, loglevel))
    args = args_parser()
    process_args(args)
    main(args)
