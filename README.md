# Op-CAD: Benchmarking and Investigating Operation-oriented CAD Generation

Official implementation and evaluation code for the ICML 2026 paper **Op-CAD: Benchmarking and Investigating Operation-oriented CAD Generation**.

Op-CAD is the first large-scale multimodal dataset for operation-oriented CAD generation. It contains 128,449 modeling operations across four operation types and five modalities, together with a geometry-guided annotation pipeline, the CF-IoU evaluation metric, and Chain-of-Operation (COOP) prompting.

This repository contains the evaluation, reward, prompt construction, and inference scripts. Large assets are published separately:

- Paper: https://proceedings.mlr.press/v306/bai26a.html
- Project page: https://baiyixue01.github.io/op-cad/
- Model: https://huggingface.co/Biabai/op-llama-8b
- Dataset: https://huggingface.co/datasets/Biabai/op-cad
- Data rendering code: https://github.com/Baiyixue01/data_render

## Repository Layout

- `reward/`: prompt construction, model calls, local inference, repair, and evaluation utilities.
- `eval.Makefile`: reproducible evaluation entry points.
- `config.json`: safe example runtime configuration. It uses environment variables for external API keys.
- `test_run.sh`, `total.sh`, `vllm-run.sh`: example launch scripts from the original experiments.

Large training data, generated outputs, and intermediate CAD artifacts are intentionally not tracked in Git. Download them from the Hugging Face dataset instead.

## Installation

Create a Python environment with CadQuery and the ML dependencies:

```bash
conda create -n op-cad python=3.10 -y
conda activate op-cad
pip install -r requirements.txt
```

CadQuery and OpenCascade rendering can require system OpenGL libraries on Linux. If rendering fails in a headless environment, use an EGL/OSMesa-enabled environment or run on a machine with working OpenGL.

## Data

Download the dataset from Hugging Face:

```bash
huggingface-cli download Biabai/op-cad --repo-type dataset --local-dir data/op-cad
```

The public dataset contains:

- `shard_work_shards_op_image`: operation-oriented rendered image shards.
- `step_files_pc_2048_normalized`: normalized STEP point cloud data.
- `op_oriented_step_pc_2048_normalized`: operation-oriented STEP point cloud data.
- `op-cad-data`: CSV/JSON metadata used by this repository.

For legacy scripts that expect local paths, either set symbolic links into `data/` or update `config.json` and the Makefile variables.

## Model

Download the OP-CAD Llama model:

```bash
huggingface-cli download Biabai/op-llama-8b --local-dir models/op-llama-8b
```

Serve it with vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model models/op-llama-8b \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --port 8000 \
  --served-model-name op-llama-8b
```

Then point `config.json` at `http://localhost:8000/v1/chat/completions`.

## Evaluation

Build prompts:

```bash
python reward/build_prompt_json.py \
  --input data/op-cad/op-cad-data/split_result.json \
  --output outputs/prompts.json
```

Run one of the evaluation targets:

```bash
make -f eval.Makefile Qwen2.5-3b-coder-highlight
```

The Makefile targets mirror the original experiment setup. Before running them, update paths for your local model, dataset, and output directories.

## External APIs

The repository does not store API keys. If you use OpenAI-compatible endpoints, export keys in your shell and keep them out of Git:

```bash
export OPENAI_API_KEY=...
export OP_CAD_HTTP_BASE_URL=...
export OP_CAD_HTTP_MODEL=...
export OP_CAD_HTTP_API_KEY=...
```

## Citation

If you use the dataset, model, or evaluation code, please cite:

```bibtex
@InProceedings{pmlr-v306-bai26a,
  title     = {Op-{CAD}: Benchmarking and Investigating Operation-oriented {CAD} Generation},
  author    = {Bai, Yixue and Gu, Yufei and Xie, Zeke},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages     = {5245--5264},
  year      = {2026},
  volume    = {306},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v306/bai26a.html}
}
```
