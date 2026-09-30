# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os
import pathlib
import shutil

import torch
import wandb

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from slicegpt.config import config
from slicegpt.slicing_scheduler import SlimmableConstantSlicingScheduler


def slicing_arg_parser(interactive: bool = True) -> argparse.Namespace:
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
        default="facebook/opt-125m",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    path_group.add_argument(
        "--sliced-model-path",
        type=str,
        help="Path to load the model to fine-tune (sliced) and tokenizer from",
        default=None,
    )
    # parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--dtype", type=str, help="Data type to use (all torch.<dtype> can be selected).", default="fp16")
    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset to calibrate on.",
        choices=["wikitext2", "ptb", "c4", "alpaca", "slim_pajama_200k"],
        default="wikitext2",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        help="Dataset to evaluate perplexity on. Falls back to --cal-dataset if not specified.",
        choices=["wikitext2", "ptb", "c4", "alpaca", "slim_pajama_200k"],
        default=None,
    )
    parser.add_argument(
        "--cal-nsamples",
        type=int,
        help="Number of samples of the calibration data to load.",
        default=128,
    )
    parser.add_argument("--cal-batch-size", type=int, default=4, help="Batch size for loading the calibration data.")
    parser.add_argument(
        "--cal-max-seqlen", type=int, default=2048, help="Maximum sequence length for the calibration data."
    )
    parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--round-interval",
        type=int,
        default=8,
        help="Interval for rounding the weights (the best value may depend on your hardware)",
    )
    parser.add_argument(
        "--final-orientation",
        type=str,
        default="random",
        choices=["random", "pca"],
        help="Final orientation of the sliced weights.",
    )
    parser.add_argument(
        "--ppl-eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity."
    )
    parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument(
        "--ppl-eval-nsamples", type=int, default=128, help="Number of samples to evaluate the perplexity on."
    )
    parser.add_argument("--eval-baseline", action="store_true", help="Evaluate the baseline model.")
    parser.add_argument("--eval-fused-model", action="store_true", help="Evaluate the fused model.")
    parser.add_argument("--ppl-only", action="store_true", help="Evaluate the loaded model without doing compression.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument("--save-dir", type=str, default=None, help="Path to save the model.")

    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))

    parser.add_argument('--wandb-project', type=str, default="slicegpt", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )

    # ---- CUSTOM ARGS ----

    parser.add_argument('--save-rotations', action="store_true", default=False, help="Save rotation matrices produced during slicing.")

    parser.add_argument("--target-compressions", action=lambda *args, **kwargs: SplitArgs(float, *args, **kwargs), help="Comma separated list of target ratios of compression.", required=True)

    parser.add_argument(
        "--variant",
        choices=["ecd", "ireko"],
        required=True,
        help="Which compression variant to run. \"ecd\": expanded-calibration PCA rotations only; \"ireko\": ECD rotations plus ridge-regression correction of the sliced projections.",
    )

    parser.add_argument("--pca-calc-method", choices=["reduce_all", "reduce_batch", "experimental"], help="Method for reducing activations before computing PCA. \"reduce_all\" will reduce both the example axis and the width axis; \"reduce_batch\" will reduce only the example axis, concatenating the width axis to the sequence length one (effectively expanding the calibration set size used for computing PCA).", default="reduce_all")

    parser.add_argument("--lambda-reg", type=float, help="Regularization coefficient for ridge regression.", default=0.0)

    parser.add_argument("--save-ridge-updates", action="store_true", default=False, help="Save regression weights computed by ridge regression at each layer")
    
    parser.add_argument("--layers-for-attn-correction", action=lambda *args, **kwargs: SplitArgs(int, *args, **kwargs), help="Comma separated list of layer indices on which to apply hidden activations correction for the self-attention component (Q,K,V) linears. If not specified, no correction is applied (like standard SliceGPT). If \"all\", all layers are selected.", default=list())
    parser.add_argument("--layers-for-mlp-correction", action=lambda *args, **kwargs: SplitArgs(int, *args, **kwargs), help="Comma separated list of layer indices on which to apply hidden activations correction for the mlp component (up_projs/fc1) linears. If not specified, no correction is applied (like standard SliceGPT). If \"all\", all layers are selected.", default=list())
    parser.add_argument("--attn-modules-for-correction", action=lambda *args, **kwargs: SplitArgs(str, *args, **kwargs), help="Comma separated list of attention module names to which correction is to be applied. If not specified, no correction is applied (like standard SliceGPT). If \"all\", all layers are selected.", default=["v_proj"])
    parser.add_argument("--mlp-modules-for-correction", action=lambda *args, **kwargs: SplitArgs(str, *args, **kwargs), help="Comma separated list of mlp module names to which correction is to be applied. If not specified, no correction is applied (like standard SliceGPT). If \"all\", all layers are selected.", default=list())

    parser.add_argument("--dump-cal-dataset", action="store_true", help="Saves calibration dataset examples used in a .pt file inside the target directory", default=False)
    
    parser.add_argument("--ridge-nsamples", type=int, help="Number of tokens to select for computing ridge. Sampled uniformly from all tokens available in calibration dataset.", default=8192)
    
    parser.add_argument("--model-state-path", help="Path to model state dict. To be used if we want to load a rotated-only model from a state dict.", default=None)
    
    parser.add_argument(
        "--cal-dataset-local-path",
        type=str,
        help="Set this to load cal dataset from local path.",
        default=None,
    )
    parser.add_argument(
        "--eval-dataset-local-path",
        type=str,
        help="Set this to load eval dataset from local path.",
        default=None,
    )

    parser.add_argument("--wandb-mode", help="Select wandb mode. Default disabled", default="disabled")

    return parser.parse_args() if interactive else parser.parse_args('')


def process_slicing_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if not 0 <= args.sparsity < 1:
        raise argparse.ArgumentTypeError(f"Sparsity should be in the range [0, 1)")

    if args.device:
        config.device = torch.device(args.device)

    config.dtype = None
    if args.dtype:
        # retro-compatibility
        if args.dtype == "fp32":
            args.dtype = "float32"
        elif args.dtype == "fp16":
            args.dtype = "float16"

        dtype = getattr(torch, args.dtype, None)
        if dtype is None:
            raise argparse.ArgumentTypeError(f"Invalid datatype: torch.{args.dtype}")
        config.dtype = dtype
    # if args.dtype == "fp16":
    #     config.dtype = torch.float16
    # elif args.dtype == "fp32":
    #     config.dtype = torch.float32
    # else:
    #     raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")


def slicing_main(args: argparse.Namespace) -> None:
    logging.info("Running SliceGPT experiment.")
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project=args.wandb_project, config=args, mode=args.wandb_mode)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    if args.sliced_model_path:
        # load the model from sliced_model_path to compute perplexity and skip rotation and slicing
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model,
            args.sliced_model_path,
            sparsity=args.sparsity,
            round_interval=args.round_interval,
            token=args.hf_token,
        )
    elif args.model_state_path:
        model_adapter, tokenizer = hf_utils.load_rotated_model_state(
            args.model,
            args.model_state_path,
            sparsity=args.sparsity,
            round_interval=args.round_interval,
            token=args.hf_token
        )
    else:
        # load one of the pre-trained models
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
            args.model, args.model_path, token=args.hf_token, dtype=config.dtype
        )

    model = model_adapter.model

    def reset_model_device() -> None:
        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(model_adapter)
        else:
            model.to(config.device)

    dataset = data_utils.get_dataset(args.cal_dataset, args.cal_dataset_local_path)
    train_dataset = dataset["train"]
    train_loader = data_utils.prepare_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_seqlen=args.cal_max_seqlen,
        batch_size=args.cal_batch_size,
        nsamples=args.cal_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )

    eval_dataset_name = args.eval_dataset or args.cal_dataset
    eval_dataset_local_path = args.eval_dataset_local_path or args.cal_dataset_local_path
    eval_test = data_utils.get_dataset(eval_dataset_name, eval_dataset_local_path)["test"]
    test_loader = data_utils.prepare_test_dataloader(
        dataset=eval_test, tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
    )

    # evaluate perplexity and exit if sliced model is loaded or if ppl_only is set
    if args.sliced_model_path or args.ppl_only:
        reset_model_device()
        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Loaded model perplexity: {dataset_ppl}')
        wandb.log({"original_ppl": dataset_ppl})
        return

    # original ppl
    if args.eval_baseline:
        reset_model_device()
        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Original ppl: {dataset_ppl:.4f}')
        wandb.log({"original_ppl": dataset_ppl})
        model.cpu()
        utils.cleanup_memory()

    if not (args.sliced_model_path or args.model_state_path):
        # replace modules with compressible equivalents
        layernorm_fusion.replace_layers(model_adapter)

        # fuse layernorms and add rotations to skip connections
        layernorm_fusion.fuse_modules(model_adapter)

    # don't run this on large and/or distributed models
    if args.eval_fused_model and not args.distribute_model:
        model.to(config.device)

        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Post-fusion: {dataset_ppl:.4f}')
        wandb.log({"post_fusion_ppl": dataset_ppl})

        model.cpu()

        # run GC and cleanup GPU memory
        utils.cleanup_memory()

    original_param_count = sum(int(p.nelement()) for p in model.parameters())
    logging.info(f'Original model parameters: {original_param_count:,d}')

    # compute new embedding dimension given the desired sparsity level
    new_embedding_dimension = [int(round(ratio * model_adapter.hidden_size)) for ratio in args.target_compressions]
    # round (down) to the nearest multiple of round_interval
    new_embedding_dimension = list(map(lambda dim: dim - (dim % args.round_interval), new_embedding_dimension))
    logging.info(
        f"New embedding dimensions: {new_embedding_dimension}"
    )

    scheduler = SlimmableConstantSlicingScheduler(new_embedding_dimension)
    ridge_ws = None
    if args.variant == "ecd":
        rotations = rotate.rotate_sequential_ecd(
            model_adapter,
            train_loader,
            scheduler,
            save_rotations=args.save_rotations,
            pca_calc_method=args.pca_calc_method,
        )
    elif args.variant == "ireko":
        rotations, ridge_ws = rotate.rotate_sequential_ecd_with_ridge(
            model_adapter,
            train_loader,
            scheduler,
            save_rotations=args.save_rotations,
            pca_calc_method=args.pca_calc_method,
            lambda_reg=args.lambda_reg,
            ridge_nsamples=args.ridge_nsamples,
            save_ridge_updates=args.save_ridge_updates,
            layers_for_attn_correction=args.layers_for_attn_correction,
            layers_for_mlp_correction=args.layers_for_mlp_correction,
            attn_modules_for_correction=args.attn_modules_for_correction,
            mlp_modules_for_correction=args.mlp_modules_for_correction,
        )
    else:
        raise ValueError(f"Unrecognized variant: {args.variant}")
    
    if args.save_dir:
        sliced_model_dir = pathlib.Path(args.save_dir)
        sliced_model_dir.mkdir(parents=True, exist_ok=True)

        sliced_model_name = sliced_model_dir / f'{pathlib.Path(args.model).name}_{args.sparsity}.pt'

        # Save the sliced model
        torch.save(model.state_dict(), sliced_model_name)

        # Save the rotation matrices
        if args.save_rotations and rotations is not None:
            torch.save(rotations, sliced_model_name.with_stem(sliced_model_name.stem + "_Qs"))
        
        if args.save_ridge_updates and ridge_ws is not None:
            torch.save(ridge_ws, sliced_model_name.with_stem(sliced_model_name.stem + "ridge_Ws"))

        # dump calibration dataset examples before starting if flag is set
        if args.dump_cal_dataset:
            torch.save(list(train_loader), sliced_model_dir / "cal_dataset.pt")
        # Save the slicing config
        config_path = sliced_model_name.with_suffix('.json')
        config_path.write_text(model_adapter.slicing_conf.to_json_string())

        # If slicing a local model, also save HF config files in sliced model dir
        if args.model_path:
            try:
                # copy all config files (tokenizer, model and slicing configs)
                for file in pathlib.Path(args.model_path).glob("*.json"):
                    if 'safetensors' not in str(file):
                        shutil.copy(str(file), sliced_model_dir)
                # copy all tokenizer models
                for file in pathlib.Path(args.model_path).glob("*token*.model"):
                    shutil.copy(str(file), sliced_model_dir)
                # copy vocab merges if any
                for file in pathlib.Path(args.model_path).glob("merges.txt"):
                    shutil.copy(str(file), sliced_model_dir)
            except OSError as e:
                logging.info(f'Failed to copy configs and tokenizer files: {e}')

        logging.info(f"Saved sliced model to {args.save_dir}")

    reset_model_device()
    dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    logging.info(f'After rotating and slicing {dataset_ppl:.4f}')
    wandb.log({"sliced_ppl": dataset_ppl})

    sliced_param_count = sum(int(p.nelement()) for p in model.parameters())
    sliced_fraction = 1.0 - sliced_param_count / original_param_count
    logging.info(f'Sliced model parameters: {sliced_param_count:,d} (sliced fraction {sliced_fraction:.4f})')


if __name__ == "__main__":
    utils.configure_logging(log_to_console=True, log_to_file=False, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    slicing_args = slicing_arg_parser()
    process_slicing_args(slicing_args)
    slicing_main(slicing_args)
