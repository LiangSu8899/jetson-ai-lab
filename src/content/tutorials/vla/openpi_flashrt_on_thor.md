---
title: "OpenPi π₀.₅ on Jetson Thor with FlashRT TensorRT Plugins"
description: "Run OpenPi π₀.₅ on NVIDIA Jetson AGX Thor as a TensorRT engine built from FlashRT's fused NVFP4 and FlashAttention-4 kernels packaged as TensorRT plugins: the standard ONNX → trtexec workflow, about 2x lower latency than the FP8 + NVFP4 tutorial engine, same accuracy."
category: "VLA"
section: "Vision-Language-Action Models"
order: 3
tags: ["vla", "openpi", "pi0.5", "robotics", "jetson-thor", "tensorrt", "tensorrt-plugin", "flashrt", "nvfp4", "flashattention", "libero", "vision-language-action"]
authors:
  - name: "FlashRT"
    github: "LiangSu8899"
---

This tutorial builds on [OpenPi π₀.₅ on Jetson Thor](/tutorials/openpi_on_thor). It keeps the TensorRT workflow of that tutorial — ONNX, `trtexec`, a serialized engine loaded by openpi — and swaps the engine's contents for [FlashRT](https://github.com/flashrt-project/FlashRT)'s Thor kernels, packaged as TensorRT plugins.

## What changes, and what does not

| | OpenPi tutorial engine | FlashRT plugin engine |
|---|---|---|
| Build | ModelOpt FP8 + NVFP4 ONNX → `trtexec` | FlashRT calibration → ONNX with FlashRT plugins → `trtexec` |
| Layers | TensorRT kernels with Q/DQ | FlashRT fused kernels (NVFP4 GEMMs with fused GeGLU, FP8 attention projections, FlashAttention-4) inside three plugins |
| Shape | three camera slots, 208 prompt tokens | the unmasked cameras, the actual prompt (dynamic) |
| Runtime | `trtexec`, TensorRT Python, openpi `policy.infer` | same, plus the TensorRT plugin library |

## Performance

Measured with the OpenPi tutorial's own benchmark, `deployment_scripts/pi05_inference.py` (synthetic LIBERO example, 3 warmup + 10 timed runs, mean ± std), on the same Jetson AGX Thor Developer Kit (JetPack 7.2, MAXN), `pi05_libero`, action horizon 10:

| Inference Backend | Total Latency (ms) | Model Latency (ms) | Speedup |
|---|---|---|---|
| PyTorch BF16 | 132.08 ± 0.67 | 128.66 ± 0.50 | 1.0x |
| TensorRT FP8 + NVFP4 (OpenPi tutorial) | 48.83 ± 0.06 | 48.12 ± 0.04 | 2.7x |
| **TensorRT + FlashRT plugins** | **26.77 ± 0.05** | **26.02 ± 0.05** | **4.9x** |

The OpenPi tutorial engine row reproduces the tutorial's published result (48.84 / 48.08 ms). Step 6.1 runs both engines through the same script.

Action accuracy against the openpi PyTorch model (BF16) on 8 real LIBERO observations with their task prompts and pinned noise, cosine similarity of the raw actions over the 7 action dimensions (Step 6.2):

| Inference Backend | Mean | Min |
|---|---|---|
| TensorRT FP8 + NVFP4 (OpenPi tutorial) | 0.99962 | 0.99945 |
| **TensorRT + FlashRT plugins** | **0.99968** | **0.99946** |

Where the speedup comes from:

- **Kernels.** At the tutorial engine's own shape (three camera slots, 208 prompt tokens), `trtexec --useCudaGraph` measures 32.69 ms for the FlashRT engine against 47.76 ms. FlashRT's fused kernels merge what TensorRT runs as separate layers — for example the encoder FFN's gate and up projections, GeGLU and the activation quantization run as one NVFP4 kernel.
- **Shape.** LIBERO masks the third camera and pads the prompt; the FlashRT engine does not compute the masked camera or the padding.

The FlashRT engine's actions are bit-for-bit identical to FlashRT's own PyTorch runtime, which runs the same shape at the same speed. Latency varies by about 2 ms with prompt length (24–26 ms model time on LIBERO prompts, see Troubleshooting).

## How it works

```
                 TensorRT engine (ONNX, built by trtexec)
 images ──► [Pi05Siglip] ──► Concat ◄── Gather(embedding) ◄── lang_tokens
                               │
                RoPE Slice ──► [Pi05Encoder] ──► K, V
                                                   │
 noise ─────────────────────────────────────► [Pi05Decoder] ──► actions

 [ ] = FlashRT TensorRT plugin: FlashRT kernels, in FlashRT's call sequence
```

- **Plugins at fusion boundaries.** A plugin per operator would split the graph into small TensorRT islands and lose the fusions, which cross FFN, residual, normalization and quantization boundaries. The plugins cover whole stages (vision tower, prefix encoder, 10-step action decoder); per-layer versions exist for custom graphs.
- **Calibration stays in FlashRT.** FlashRT's Python runtime calibrates the static FP8 scales, AWQ and NVFP4 weights on real observations; the export records those weights into the ONNX file.
- **Verified bit for bit.** The build script checks the finished engine against FlashRT's runtime for several prompts.
- **Maintainable.** The plugins compile FlashRT's kernel sources directly. When FlashRT's kernels improve, rebuilding the plugin library and rerunning the build script carries the change into the engine, and the same bitwise check confirms it.

TensorRT still owns the graph, memory, execution context and CUDA graph capture, so any TensorRT host can run the engine.

## Prerequisites

### Hardware

- **NVIDIA Jetson AGX Thor** Developer Kit
- NVMe SSD recommended

### Software

| Component | Required Version |
|---|---|
| JetPack | 7.2 (L4T R39.x, CUDA 13.2, TensorRT 10.16) |
| Docker + NVIDIA Container Toolkit | as in the OpenPi tutorial |
| Python | 3.12 with `venv` |

### From the OpenPi tutorial

Complete **Steps 1–7** of [OpenPi π₀.₅ on Jetson Thor](/tutorials/openpi_on_thor):

- the `openpi-pi0.5:l4t-jp7.2` container image, and the `openpi/` checkout with the Jetson Thor deployment scripts;
- the `pi05_libero` PyTorch checkpoint at `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`.

The steps below run partly on the Thor host (FlashRT build and calibration) and partly inside the OpenPi container (engine build and inference).


## Step 1: Build FlashRT on the Host

FlashRT runs natively on Jetson (no container).

```bash
python3 -m venv ~/flashrt-venv
source ~/flashrt-venv/bin/activate
pip install torch --index-url https://pypi.jetson-ai-lab.io/sbsa/cu132

git clone -b feat/tensorrt-backend https://github.com/flashrt-project/FlashRT.git ~/FlashRT
cd ~/FlashRT
git submodule update --init third_party/cutlass

cmake -B build -S . -DGPU_ARCH=110
cmake --build build -j8
pip install -e ".[thor-fa4]"
pip install onnx sentencepiece safetensors
```

The PaliGemma tokenizer that openpi downloaded to `~/.cache/openpi/big_vision/` during the OpenPi tutorial is picked up automatically.

> See FlashRT's [Pi0.5 Thor guide](https://github.com/flashrt-project/FlashRT/blob/feat/tensorrt-backend/docs/pi05_thor.md) for the native runtime and [INSTALL.md](https://github.com/flashrt-project/FlashRT/blob/feat/tensorrt-backend/docs/INSTALL.md) for build details.


## Step 2: Build the TensorRT Plugin Library

```bash
cd ~/FlashRT
cmake -S backends/tensorrt -B build/tensorrt
cmake --build build/tensorrt -j 2
ls build/tensorrt/libflashrt_trt_pi05.so
```

The library contains FlashRT's kernels, the Pi0.5 stages and the FlashAttention-4 modules; at run time it needs only CUDA, cuBLAS and TensorRT.


## Step 3: Collect Calibration Observations (Inside the Container)

FlashRT calibrates on real observations. Start the OpenPi container as in the OpenPi tutorial (from the `openpi/` checkout), with FlashRT and an output directory mounted:

```bash
mkdir -p ~/flashrt_out
sudo docker run --rm -it --runtime nvidia \
  -v "$PWD":/workspace \
  -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/FlashRT":/flashrt \
  -v "$HOME/flashrt_out":/flashrt_out \
  -w /workspace \
  openpi-pi0.5:l4t-jp7.2
```

Inside the container:

```bash
export PYTHONPATH=packages/openpi-client/src:src:.:/flashrt/backends/tensorrt/integrations/openpi:$PYTHONPATH
cp -r ./src/openpi/models_pytorch/transformers_replace/* /usr/local/lib/python3.12/dist-packages/transformers/

python /flashrt/backends/tensorrt/tools/make_libero_fixture.py /flashrt_out/libero_calib_8.npz
```

This samples 8 frames evenly across the LIBERO dataset (the OpenPi tutorial's calibration dataset), downloading only the episodes it needs, and stores the images resized as openpi's input transform resizes them.

> **Tip:** if the dataset is already in `~/.cache/huggingface` and the Hub is unreachable, prefix the command with `HF_HUB_OFFLINE=1`.

Leave this container running; Steps 5 and 6 use it.


## Step 4: Calibrate, Export ONNX and Build the Engine (Host)

In a second terminal on the host:

```bash
source ~/flashrt-venv/bin/activate
cd ~/FlashRT
PYTHON=$(which python) backends/tensorrt/tools/build_pi05_engine.sh \
  ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  2 \
  ~/flashrt_out/libero_calib_8.npz \
  ~/flashrt_out/pi05_libero
```

Arguments: checkpoint, number of unmasked cameras (2 for LIBERO: base and wrist), calibration observations, output directory. The script takes about 3 minutes:

```
== calibrate + record SigLIP
per-layer reference vs library: True | eager tokens vs captured graph: True
== calibrate + record encoder
per-layer reference vs library (all layers): True
== calibrate + record decoder
step reference vs library decoder: actions+KV bitwise True
== prompt tables + references
== ONNX export
wrote .../pi05_libero/onnx/pi05.onnx initializers 590
== trtexec build
&&&& PASSED TensorRT.trtexec
== engine check
prompt 0: 14 tokens | actions bitwise eager=True graph=True | eager 25.44 ms, graph 24.00 ms
prompt 1: 8 tokens | actions bitwise eager=True graph=True | eager 27.00 ms, graph 25.79 ms
prompt 2: 14 tokens | actions bitwise eager=True graph=True | eager 25.44 ms, graph 24.00 ms
prompt 3: 6 tokens | actions bitwise eager=True graph=True | eager 25.40 ms, graph 23.98 ms
ENGINE_PROMPT_DYNAMIC_PASS
```

`bitwise=True` means the engine's actions equal FlashRT's runtime exactly, eagerly and under a CUDA graph.

The engine I/O:

| Tensor | Type | Shape | Meaning |
|---|---|---|---|
| `images` | fp16 | `[2, 224, 224, 3]` | base and wrist camera, HWC in [-1, 1] |
| `lang_tokens` | int32 | `[n]` | prompt tokens, unpadded |
| `noise` | fp16 | `[10, 32]` | initial flow-matching noise |
| `actions` | fp16 | `[10, 32]` | raw actions (openpi unnormalizes them) |


## Step 5: Build the Engine for the Container's TensorRT

TensorRT engines are tied to the exact TensorRT version, and the container ships its own. Rebuild the engine from the ONNX file inside the container:

```bash
trtexec --onnx=/flashrt_out/pi05_libero/onnx/pi05.onnx \
  --dynamicPlugins=/flashrt/build/tensorrt/libflashrt_trt_pi05.so \
  --stronglyTyped --builderOptimizationLevel=0 --memPoolSize=workspace:2048 \
  --minShapes=lang_tokens:2 --optShapes=lang_tokens:14 --maxShapes=lang_tokens:256 \
  --saveEngine=/flashrt_out/pi05_libero_container.engine
```

`--dynamicPlugins` loads the FlashRT plugins so the ONNX parser can resolve the `flashrt` operators. The build takes a few seconds.


## Step 6: Run π₀.₅ Through openpi (Inside the Container)

`openpi_flashrt.py` is the FlashRT counterpart of the tutorial's `setup_pi0_tensorrt_engine`: openpi's input and output transforms stay unchanged.

```python
import numpy as np
from openpi.policies import policy_config
from openpi.training import config as _config
from openpi_flashrt import setup_pi0_flashrt_engine

config = _config.get_config("pi05_libero")
checkpoint = "/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
policy = policy_config.create_trained_policy(config, checkpoint)
policy = setup_pi0_flashrt_engine(
    policy,
    "/flashrt_out/pi05_libero_container.engine",
    "/flashrt/build/tensorrt/libflashrt_trt_pi05.so",
)

obs = np.load("/flashrt_out/libero_calib_8.npz")
example = {
    "observation/image": obs["img_4"],
    "observation/wrist_image": obs["wrist_4"],
    "observation/state": obs["state_4"],
    "prompt": str(obs["prompt_4"]),
}
result = policy.infer(example)
print(result["actions"].shape, result["policy_timing"])   # (10, 7)
```

The first call captures a CUDA graph for the prompt length; later calls replay it.

### 6.1 Benchmark with the OpenPi tutorial's script

Run the OpenPi tutorial's `pi05_inference.py` for both engines. For the FlashRT engine, `run_official_pi05_inference.py` runs the same script unchanged and only points its TensorRT hook at the FlashRT engine:

```bash
C=/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch

# OpenPi tutorial engine (Step 10 of the OpenPi tutorial)
python deployment_scripts/pi05_inference.py \
  --config-name pi05_libero --checkpoint-dir $C \
  --engine-path $C/engine/model_fp8_nvfp4.engine \
  --inference-mode tensorrt --num-warmup 3 --num-test-runs 10

# FlashRT plugin engine
FLASHRT_TRT_PLUGIN=/flashrt/build/tensorrt/libflashrt_trt_pi05.so \
python /flashrt/backends/tensorrt/integrations/openpi/run_official_pi05_inference.py \
  --config-name pi05_libero --checkpoint-dir $C \
  --engine-path /flashrt_out/pi05_libero_container.engine \
  --inference-mode tensorrt --num-warmup 3 --num-test-runs 10
```

Expected results:

```
# OpenPi tutorial engine
Total inference time: 48.83 ± 0.06 ms
Model inference time: 48.12 ± 0.04 ms

# FlashRT plugin engine
Total inference time: 26.77 ± 0.05 ms
Model inference time: 26.02 ± 0.05 ms
```

### 6.2 Compare accuracy against PyTorch on real observations

```bash
python /flashrt/backends/tensorrt/integrations/openpi/compare_openpi_accuracy.py \
  /flashrt_out/libero_calib_8.npz $C /flashrt_out/pi05_libero_container.engine \
  /flashrt/build/tensorrt/libflashrt_trt_pi05.so \
  --tutorial-engine $C/engine/model_fp8_nvfp4.engine
```

```
FlashRT engine   vs openpi PyTorch, cosine over 7 action dims: mean 0.99968 min 0.99946 ...
tutorial engine  vs openpi PyTorch, cosine over 7 action dims: mean 0.99962 min 0.99945 ...
```

Each observation uses its own LIBERO task prompt and fixed noise; both engines and PyTorch see identical inputs. The observations are the 8 frames sampled in Step 3.

> **Note:** `pi05_inference.py --inference-mode compare` also works through `run_official_pi05_inference.py`. Its synthetic example uses random pixel images, which are far from the real observations both engines are calibrated on, and a new random example on every run unless NumPy is seeded (`EXAMPLE_SEED=<n>`), so its cosine is only comparable between runs with the same seed.


## (Optional) Other Ways to Run the Engine

- **trtexec:**
  ```bash
  trtexec --loadEngine=/flashrt_out/pi05_libero_container.engine \
    --dynamicPlugins=/flashrt/build/tensorrt/libflashrt_trt_pi05.so \
    --shapes=lang_tokens:14 --useCudaGraph
  ```
- **TensorRT Python or C++:** load the plugin library into the plugin registry (`trt.get_plugin_registry().load_library(...)`) before deserializing the engine.
- **TensorRT Edge-LLM:** FlashRT ships a `pi05_policy_inference` example for an Edge-LLM build, with image loading, tokenization and CUDA graphs in C++ ([integration guide](https://github.com/flashrt-project/FlashRT/tree/feat/tensorrt-backend/backends/tensorrt/integrations/edgellm)).


## Keeping Up with FlashRT

The engine is a snapshot of FlashRT's kernels and your calibration. After updating FlashRT:

```bash
cd ~/FlashRT && git pull
cmake --build build -j8 && cmake --build build/tensorrt -j 2
PYTHON=$(which python) backends/tensorrt/tools/build_pi05_engine.sh \
  ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 2 \
  ~/flashrt_out/libero_calib_8.npz ~/flashrt_out/pi05_libero
```

then repeat Step 5. The build script's bitwise check confirms the new engine still matches FlashRT's runtime. How the plugins are organized, the operator reference and how to package other pipelines are described in FlashRT's [TensorRT backend guide](https://github.com/flashrt-project/FlashRT/blob/feat/tensorrt-backend/docs/tensorrt_backend.md).


## Troubleshooting

| Symptom | Fix |
|---|---|
| `Plugin not found, are the plugin name, version, and namespace correct?` | pass the library with `--dynamicPlugins` (not `--staticPlugins`), or load it into the plugin registry first |
| `Failed to deserialize` / engine version error | the engine was built with a different TensorRT; rebuild it from ONNX where you run it (Step 5) |
| `CUTLASS v4.4.2 not found` | `git submodule update --init third_party/cutlass` |
| NVVM error `-arch=compute_a is an unsupported option` when FA4 compiles | `export CUTE_DSL_ARCH=sm_101a` (the build script sets it) |
| Engine check reports `bitwise=False` with differences around 1e-5 | the process loaded a different cuBLAS than PyTorch's; accuracy is unaffected |
| Latency ~2 ms higher for some prompts | decoder attention cuBLAS kernels are slower when `522 + prompt tokens` (rounded up to even) is not a multiple of 8 |


## References

- [OpenPi π₀.₅ on Jetson Thor](/tutorials/openpi_on_thor)
- [FlashRT](https://github.com/flashrt-project/FlashRT) and its [TensorRT backend](https://github.com/flashrt-project/FlashRT/tree/feat/tensorrt-backend/backends/tensorrt)
- [TensorRT plugins (IPluginV3)](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html)
- [OpenPi](https://github.com/Physical-Intelligence/openpi)
