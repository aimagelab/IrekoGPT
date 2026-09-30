# IrekoGPT

**Authors:** Pietro Moriello, Pietro Buzzega, Angelo Porrello, Simone Calderara. AImageLab, University of Modena and Reggio Emilia, Italy.

This repository contains the code and the experiment scripts for **IrekoGPT**, a post-training
compression method for transformer language models. It is built on top of, and forked from, the
[SliceGPT](https://arxiv.org/abs/2401.15024) codebase, whose MIT license is retained in
[LICENSE](./LICENSE). We acknowledge and thank the original authors for publishing their work with a
permissive license. The original SliceGPT README is preserved in full at the
[end of this file](#original-slicegpt-readme).

IrekoGPT is a post-hoc method for converting pretrained LLMs into slimmable models whose width can be adjusted at inference time.
Building on SliceGPT, we retain its projection matrices without pruning them, allowing a single model to expose nested subnetworks at different widths.
We improve robustness by calibrating each layer across multiple compression ratios, and correct downstream linear layers through gradient-free ridge regression.
Across Llama and Qwen models, preliminary results show improvements over naive PCA-based slimming, with the largest gains at high compression.

The code is arranged as a package `slicegpt` in `/src`, and the experiment scripts are in
`/experiments`:

| script | role |
| --- | --- |
| [`experiments/run_slicegpt.py`](./experiments/run_slicegpt.py) | original SliceGPT, used here for the PCA baseline |
| [`experiments/run_irekogpt.py`](./experiments/run_irekogpt.py) | our variants, selected with `--variant {ecd,ireko}` |
| [`experiments/compare_acts.py`](./experiments/compare_acts.py) | activation-similarity measurements, and the multi-compression evaluation loop used for the paper's tables |

## Installation

```
    pip install -e .[experiment]
```

_Note:_ For models that require Hugging Face authentication, set the `--hf-token` argument manually
or using a key vault. Alternatively, set the environment variable `HF_TOKEN`.

## Reproducing the experiments

The commands below use the `meta-llama/Llama-2-7b-hf` setup from the paper. Values in
`<ANGLE_BRACKETS>` are placeholders you should fill in. Every command is run from the repository
root.

A note on `--sparsity 0.0`, which every slicing command below uses: the run computes and applies the
rotations but does **not** slice, and saves the full rotated checkpoint as
`<OUTPUT_DIR>/Llama-2-7b-hf_0.0.pt`. That single full-width checkpoint is what the later
multi-width slicing and the activation comparison consume, and it is what makes one ECD/IrekoGPT run
serve every target compression at once.

### SliceGPT

The original SliceGPT baseline. We followed the examples given in the upstream documentation, using
the same calibration setup as the runs below; the one thing to watch is that `--final-orientation`
must be set to `random`, which is the orientation the original method uses (and the default of
[`experiments/run_slicegpt.py`](./experiments/run_slicegpt.py)).

```bash
python experiments/run_slicegpt.py \
    --model meta-llama/Llama-2-7b-hf \
    --dtype float16 \
    --cal-dataset wikitext2 \
    --cal-nsamples 128 \
    --cal-batch-size 4 \
    --cal-max-seqlen 2048 \
    --sparsity <SPARSITY> \
    --final-orientation random \
    --save-dir <OUTPUT_DIR> \
    --wandb-mode <WANDB_MODE>
```

Unlike the runs below, this one actually slices: `--sparsity` is the fraction of the embedding
dimension removed, and one run produces one compressed model. Our comparisons use values in the
0.1-0.35 range. See also [Running SliceGPT](#running-slicegpt) in the original README below.

### PCA

The PCA-only baseline: rotations computed by PCA on the calibration set, without applying slicing.

```bash
python experiments/run_slicegpt.py \
    --model meta-llama/Llama-2-7b-hf \
    --dtype float16 \
    --cal-dataset wikitext2 \
    --cal-nsamples 128 \
    --cal-batch-size 4 \
    --cal-max-seqlen 2048 \
    --sparsity 0.0 \
    --final-orientation pca \
    --eval-baseline \
    --save-dir <OUTPUT_DIR> \
    --wandb-mode <WANDB_MODE>
```

This writes `<OUTPUT_DIR>/Llama-2-7b-hf_0.0.pt`, the PCA-rotated checkpoint. `--eval-baseline` additionally
reports the perplexity of the uncompressed model.

Also available: `--cal-dataset-local-path <PATH>` loads the calibration dataset from a directory on
disk instead of downloading it.

### ECD

Rotations computed by PCA over the *expanded* calibration set.

```bash
python experiments/run_irekogpt.py \
    --variant ecd \
    --model meta-llama/Llama-2-7b-hf \
    --dtype float16 \
    --cal-dataset wikitext2 \
    --cal-nsamples 128 \
    --cal-batch-size 4 \
    --cal-max-seqlen 2048 \
    --sparsity 0.0 \
    --target-compressions 0.7,0.8,0.9 \
    --eval-baseline \
    --save-dir <OUTPUT_DIR> \
    --wandb-mode <WANDB_MODE>
```

`--target-compressions` is the comma-separated list of width ratios the calibration set is
expanded over: `0.7,0.8,0.9` means the resulting checkpoint is calibrated at 70%,
80% or 90% of the original embedding dimension. Widths are rounded down to a multiple of
`--round-interval` (default 8).

Also available:
- `--save-rotations` additionally dumps every PCA rotation `Q` to `<OUTPUT_DIR>/Llama-2-7b-hf_0.0_Qs.pt`.
- `--cal-dataset-local-path <PATH>`, as above.

### IrekoGPT

ECD rotations plus the closed-form ridge correction.

```bash
python experiments/run_irekogpt.py \
    --variant ireko \
    --model meta-llama/Llama-2-7b-hf \
    --dtype float16 \
    --cal-dataset wikitext2 \
    --cal-nsamples 128 \
    --cal-batch-size 4 \
    --cal-max-seqlen 2048 \
    --sparsity 0.0 \
    --target-compressions 0.7,0.8,0.9 \
    --lambda-reg <LAMBDA_REG> \
    --ridge-nsamples <RIDGE_NSAMPLES> \
    --layers-for-attn-correction <LAYERS_FOR_ATTN_CORRECTION> \
    --attn-modules-for-correction <ATTN_MODULES_FOR_CORRECTION> \
    --layers-for-mlp-correction <LAYERS_FOR_MLP_CORRECTION> \
    --mlp-modules-for-correction <MLP_MODULES_FOR_CORRECTION> \
    --save-dir <OUTPUT_DIR> \
    --wandb-mode <WANDB_MODE>
```

The correction arguments:

- `--lambda-reg` — the ridge regularization coefficient. The experiments in the paper use `1e0`.
- `--ridge-nsamples` — how many tokens are sampled, uniformly from the calibration set, to solve each
  ridge problem. The experiments in the paper use `65536`.
- `--layers-for-attn-correction` / `--layers-for-mlp-correction` — comma-separated layer indices
  (e.g. `0,1,2`), or the literal `all` to select every layer.
- `--attn-modules-for-correction` / `--mlp-modules-for-correction` — comma-separated module-name
  suffixes to correct within the attention block (e.g. `v_proj`, or `v_proj,q_proj,k_proj`) and
  within the MLP block (e.g. `up_proj`, `gate_proj`, `down_proj`).

Leaving a correction group unset disables correction for it: with no MLP modules selected, for
instance, the MLP projections keep the plain SliceGPT behaviour and only the attention side is
corrected.

Also available:
- `--save-ridge-updates` dumps the per-layer ridge solutions to
  `<OUTPUT_DIR>/Llama-2-7b-hf_0.0ridge_Ws.pt`.
- `--save-rotations` and `--cal-dataset-local-path <PATH>`, as above.

### Compare activations

This script serves two purposes: it measures the similarity between the activations of a *source* and
a *target* model, and it runs the multi-compression perplexity evaluation loop used to produce the
paper's tables. Both models are given as full rotated checkpoints (the `_0.0.pt` files produced by
the runs above); the script slices them itself, once per requested sparsity.

```bash
python experiments/compare_acts.py \
    --model meta-llama/Llama-2-7b-hf \
    --dtype float16 \
    --dataset wikitext2 \
    --max-seqlen 2048 \
    --batch-size <BATCH_SIZE> \
    --sparsities 0.3,0.2,0.1 \
    --metrics <METRICS> \
    --offload-activations-chunks <OFFLOAD_ACTIVATIONS_CHUNKS> \
    --source-model-state-path <PCA_RUN_DIR>/Llama-2-7b-hf_0.0.pt \
    --target-model-state-path <TARGET_RUN_DIR>/Llama-2-7b-hf_0.0.pt \
    --save-dir <OUTPUT_DIR> \
    --wandb-mode <WANDB_MODE> \
    <MODULE_PATTERNS>
```

- `--sparsities` is a comma-separated list of sparsities — the fraction of the embedding dimension
  that is *removed* — at which the models are sliced and evaluated. Values are rounded with
  `--source-model-round-interval` (default 8).
- `<MODULE_PATTERNS>` are positional and **must come last**. Each is a
  `{input,output}.<module_suffix>` pattern, e.g. `output.v_proj input.down_proj`. Matching is by
  name suffix, so `output.v_proj` selects the `v_proj` of every layer, while
  `output.layers.1.self_attn.v_proj` selects a single module.
- `--metrics` accepts a comma-separated subset of `mse,cka` (default: both).
- `--offload-activations-chunks` splits the collected activations into that many chunks written to
  disk, which keeps memory bounded when comparing many modules at once.
- `--eval-source-model-only` skips the target model and runs only the source-model perplexity sweep.

Also available: `--dataset-local-path <PATH>`, the equivalent of `--cal-dataset-local-path` above.

## Acknowledgements

This code is a modified fork of [microsoft/TransformerCompression](https://github.com/microsoft/TransformerCompression),
the official implementation of SliceGPT, released under the MIT license. We thank its authors,
Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler and James Hensman,
and the Microsoft contributors to the repository:

> Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, James Hensman.
> *SliceGPT: Compress Large Language Models by Deleting Rows and Columns.* ICLR 2024.
> [arXiv:2401.15024](https://arxiv.org/abs/2401.15024)

---

## Original SliceGPT README

> Everything below this line is the README of the upstream SliceGPT repository that this code is
> forked from, preserved verbatim.

# Transformer Compression with SliceGPT

This repository contains the code for the paper [SliceGPT](https://arxiv.org/abs/2401.15024) (ICLR'24). Also discussed on [Hugging Face](https://huggingface.co/papers/2401.15024). 

SliceGPT is a new post-training sparsification scheme that makes transformer networks (including LLMs) smaller by 
first applying orthogonal transformations to each transformer layer that leave the model unchanged, and then slicing off the 
least-significant rows and columns (chosen by the eigenvalue decay) of the weight matrices. The model structure is 
left unchanged, but each weight matrix is replaced by a smaller (dense) weight matrix, reducing the embedding dimension 
of the model. This results in speedups (without any additional code optimization) and a reduced memory footprint.  

The code is arranged as a package `slicegpt` in `/src`, and scripts to replicate experiments from the paper are in 
`/experiments`. To install the `slicegpt` package, we recommend

```
    pip install -e .[experiment]
```

## Running SliceGPT

To run SliceGPT on `microsoft/phi-2`, from the `experiments` folder, run 
```
    python run_slicegpt.py \
           --model microsoft/phi-2 \
           --save-dir dir/to/save/sliced_model/in \
           --sparsity 0.25 \
           --device cuda:0 \
           --eval-baseline \
           --no-wandb
```

This will compress the `microsoft/phi-2` model and save the compressed model to the specified directory. Please consult 
the script for the full set of options.

_Note:_ For models that require Hugging Face authentication, set the `--hf-token` argument 
manually or using a key vault. Alternatively, set the environment variable `HF_TOKEN`.

### Recovery fine-tuning

To install additional dependencies required for post-slicing recovery fine-tuning (RFT):

```
    pip install -e .[experiment,finetune]
```

The following replicates the experiments in the paper (LoRA hyperparams valid for all Llama-2 and Phi-2 models): 
```
    python run_finetuning.py \
           --model microsoft/phi-2 \
           --sliced-model-path path/to/sliced \
           --save-dir dir/to/save/finetuned_model/in \
           --sparsity 0.25 \
           --device cuda:0 \
           --ppl-eval-dataset alpaca \
           --finetune-dataset alpaca \
           --finetune-train-nsamples 8000 \
           --finetune-train-seqlen 1024 \
           --finetune-train-batch-size 3 \
           --lora-alpha 10 \
           --lora-r 32 \
           --lora-dropout 0.05 \
           --lora-target-option attn_head_and_mlp \
           --eval-steps 16 \
           --save-steps 16 \
           --no-wandb
```

Notes: 
- The script [`bo_finetuning.py`](./experiments/bo_finetuning.py) can be used to run Bayesian optimization over the RFT hyperparameters.
- To run finetuning on the original model, specify `--model-path` instead of `--sliced-model-path`. 
- `sparsity` must be specified when specifying `sliced-model-path` to avoid default sparsity being used

### Evaluation using the [LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness) 
```
    python run_lm_eval.py \
           --model microsoft/phi-2 \
           --sliced-model-path path/to/sliced \
           --sparsity 0.25 \
           --tasks piqa \
           --no-wandb
```

Notes: 
- To run lm-eval on the original model, specify `--model-path` instead of `--sliced-model-path`. 
- `sparsity` must be specified when specifying `sliced-model-path` to avoid default sparsity being used

## Supported models

The following models from Hugging Face hub are currently supported
- [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
- [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b)
- [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b)
- [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b)
- [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
- [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
- [facebook/opt-125m](https://huggingface.co/facebook/opt-125m)
- [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b)
- [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b)
- [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b)
- [facebook/opt-13b](https://huggingface.co/facebook/opt-13b)
- [facebook/opt-30b](https://huggingface.co/facebook/opt-30b)
- [facebook/opt-66b](https://huggingface.co/facebook/opt-66b)

## Extending support to a new model type

The model you wish to support must be in Hugging Face Hub format. The model files can be downloaded from 
Hugging Face Hub by supplying `--model` argument, or accessed from local storage by using the `--model` and 
`--model-path` argument. To add SliceGPT support for a new model, one needs to implement a new model adapter 
and update `hf_utils.get_model_and_tokenizer` before slicing the new model.

### Implementing a new model adapter
- Implement the [ModelAdapter](./src/slicegpt/model_adapter.py) interface for the new model. The ModelAdapter class tells SliceGPT 
  how to interact with the model, an instance of which is stored at `self.model`. For example, 
  how to access each of the layers of the model.
- Implement the [LayerAdapter](./src/slicegpt/model_adapter.py) interface for the transformer layers. 
  The LayerAdapter class tells SliceGPT how to interact 
  with each transformer layer of the model, an instance of which is stored at `self.layer`. 
  For example, how to access the attention and MLP components of the transformer layer, and 
  how to update the arguments to the transformer layer's forward method.
- Implement a compressed transformer layer class that subclasses the transformer layer. 
  This class should also  provide an adapted `forward()` method to work with the compressed model. 
  This method should specify how the skip connection orthogonal matrices are used, depending on 
  whether MLP and attention blocks are sequential ([OPT](./src/slicegpt/adapters/opt_adapter.py), 
  [Llama-2/Llama-3](./src/slicegpt/adapters/llama_adapter.py)) or parallel 
  ([Phi-2](./src/slicegpt/adapters/phi2_adapter.py)). The `self.*_shortcut_Q` matrices are attached to the modules during
  slicing and are available in `forward()`. If the skip connection does not need modification, these matrices will be None, 
  and the `forward()` method can follow the original workflow. For more details on this, 
  please read Section 3 in [the paper](https://arxiv.org/abs/2401.15024).

Example: [llama_adapter.py](./src/slicegpt/adapters/llama_adapter.py)

### Using a new model adapter to slice a model
Once a model adapter is implemented, compressing the model involves three conceptual steps:
  - Replace modules with compressed equivalents (via `slicegpt.layernorm_fusion.replace_layers`)
  - Fuse layer norms and add rotations to skip connections (via `slicegpt.layernorm_fusion.fuse_modules`)
  - Rotate the inputs and slice the layers (via `slicegpt.rotate.rotate_and_slice`)

Example: [run_slicegpt.py](./experiments/run_slicegpt.py)

_Note:_ If the model you wish to support is not available in Hugging Face, you will also need to implement 
custom model loading and initialization functionality.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply 
Microsoft sponsorship.

Any use of third-party trademarks or logos are subject to those third-party's policies.

