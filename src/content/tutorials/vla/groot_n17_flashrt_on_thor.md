---
title: "Isaac GR00T 1.7 with FlashRT on Jetson Thor"
description: "Run the dual-camera LIBERO policy in Docker and compare FlashRT, TensorRT, and PyTorch using matched inputs."
category: "VLA"
section: "Vision-Language-Action Models"
order: 3
tags: ["vla", "gr00t", "jetson-thor", "flashrt", "tensorrt", "nvfp4", "libero"]
authors:
  - name: "Liang Su"
    github: "LiangSu8899"
---

Isaac GR00T 1.7 turns camera images, robot state, and a language instruction into an action chunk. This tutorial runs the LIBERO-10 checkpoint with its two training-time views, `image` and `wrist_image`. It follows the Jetson AI Lab TensorRT deployment recipe and adds a FlashRT path and a shared benchmark. [Isaac GR00T 1.7 on Jetson Thor](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/)

## Pipeline overview

The policy combines the two RGB views, the task instruction, and robot state. The ViT encodes the images, the LLM conditions the policy on the instruction, and the diffusion transformer (DiT) predicts the action chunk. Auxiliary modules prepare state and time embeddings and decode actions for the controller.

## Two-camera inference performance

On Jetson AGX Thor in MAXN with fixed clocks, FlashRT completed the matched two-camera inference call in **32.85 ms**. That is **12.3% lower latency** than JAL TensorRT and **3.11× the throughput** of eager PyTorch for serial batch-one calls.

| Inference backend | Median complete call | P95 complete call |
| --- | ---: | ---: |
| PyTorch eager BF16 | 102.13 ms | 111.67 ms |
| JAL optimized mixed NVFP4 | 37.48 ms | 38.55 ms |
| FlashRT FP8 backbone + NVFP4 action head | **32.85 ms** | **33.27 ms** |

![Bar chart showing median and P95 complete-call latency on Jetson AGX Thor](/code-samples/groot-n17-flashrt-thor/latency.svg)

All three used the same checkpoint, two current camera frames, prompt, state, supplied diffusion noise, batch size of one, four denoising steps, and two CPU threads. The timer covers a raw CPU observation through decoded CPU actions, with CUDA synchronization. It includes preprocessing and transfers. Model loading, engine building, calibration, graph preparation, simulator rendering, and warmup happen outside the timer.

Each backend ran independently in its own container, using the same saved observations from the first fixed initial state of LIBERO-10 tasks 4, 6, and 9. Each task had 20 warmup calls followed by 300 timed calls, for 900 measurements per backend. The benchmark used MAXN with CPU, GPU GPC, and EMC clock limits fixed at 2.601, 1.575, and 4.266 GHz, respectively. The timing run used PyTorch 2.12.0a0 and TensorRT 10.16.1.11.

## How the optimization works

### JAL TensorRT

JAL builds separate TensorRT engines for the backbone, action head, and auxiliary modules. Select the precision policy with `--quantization`.

| **Policy** | **ViT** | **LLM** | **DiT** | **Aux** |
| :--- | :--- | :--- | :--- | :--- |
| JAL baseline (`none`) | FP32 | BF16 | BF16 | BF16 |
| JAL `fp8` | FP8 | FP8 | FP8 | FP16 |
| JAL `mixed_nvfp4` **(benchmarked)** | FP8 | **NVFP4** | **NVFP4** | FP16 |

### FlashRT

| **Policy** | **ViT** | **LLM** | **DiT** | **Aux** |
| :--- | :--- | :--- | :--- | :--- |
| FlashRT 16-bit reference | FP16 | FP16 | BF16 | FP16 / BF16 |
| FlashRT FP8 | FP8 | FP8 | FP8 | FP8 / BF16 |
| FlashRT NVFP4 **(benchmarked)** | FP8 | FP8 | **NVFP4** | FP8 / BF16 |

In the FlashRT table, **Aux** lists the precision of VL self-attention first and the state encoder, action encoder, and action decoder second. The action encoder uses FP16 for SiLU. The NVFP4 DiT projections produce BF16 outputs. The latency comparison above covers the two marked quantized policies and official eager PyTorch.

## LIBERO-10 task success

| Backend | Successes / episodes | Mean ± run-to-run SD |
| --- | ---: | ---: |
| PyTorch eager BF16 | 4,457 / 5,000 | 89.14% ± 1.57 pp |
| JAL optimized mixed NVFP4 | 4,441 / 5,000 | 88.82% ± 1.10 pp |
| FlashRT FP8 backbone + NVFP4 action head | 4,463 / 5,000 | **89.26% ± 1.01 pp** |

Each backend ran ten rounds of 500 episodes. A round uses all ten tasks and every one of the 50 fixed initial states per task. Seeds vary across rounds. The reported SD is across the ten round success rates, in percentage points. The three backends had comparable task success in these runs. Task runs used two Thor systems with PyTorch 2.12 and 2.9 reference environments and TensorRT 10.16. [Per-round counts and runtime allocation](https://github.com/flashrt-project/FlashRT/blob/f2459ab18f168b549b2caced636193aecc97d0d7/evaluations/jal_groot/results/libero-ten-rounds.json)

## Prerequisites

- Jetson AGX Thor with **JetPack 7.2 / Jetson Linux R39**, a GPU-enabled Docker installation, and NVIDIA Container Toolkit. Follow the [Jetson setup guide](https://www.jetson-ai-lab.com/tutorials/getting-started-with-jetson/) for the host setup.
- Plan for at least **80 GB of free storage** for the container, model files, engine export, and benchmark output.
- A Hugging Face account with access to [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B). Accept the model terms before downloading.

## Step 1: Set Jetson to maximum performance and get the scripts

Run the commands on the Thor host in Bash. Use an idle GPU, select MAXN, and lock the CPU, GPU GPC, and EMC clocks before benchmarking. Save the current clock settings first and verify the fixed clocks with `jetson_clocks --show`. This follows the [Jetson AI Lab OpenPi setup](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/#step-1-set-jetson-to-maximum-performance).

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git git-lfs python3-venv
git lfs install
cat /etc/nv_tegra_release
docker --version
nvidia-smi
sudo nvpmodel -q
# Select the MAXN mode listed for your system (mode 0 on the measured kit).
sudo nvpmodel -m 0
clock_settings="$(mktemp -d)/jetson-clocks.conf"
sudo jetson_clocks --store "$clock_settings"
sudo jetson_clocks
sudo jetson_clocks --show

git clone --filter=blob:none --no-checkout https://github.com/flashrt-project/FlashRT.git flashrt-comparison
git -C flashrt-comparison sparse-checkout set evaluations/jal_groot
git -C flashrt-comparison checkout --detach f2459ab18f168b549b2caced636193aecc97d0d7
cd flashrt-comparison/evaluations/jal_groot
```

Download the two companion files into this directory as `Dockerfile.twocam` and `compare_latency.py`. [Dockerfile](/code-samples/groot-n17-flashrt-thor/Dockerfile) · [Latency runner](/code-samples/groot-n17-flashrt-thor/compare_latency.py)

```bash
SAMPLES=https://www.jetson-ai-lab.com/code-samples/groot-n17-flashrt-thor
curl -fL "$SAMPLES/Dockerfile" -o Dockerfile.twocam
curl -fL "$SAMPLES/compare_latency.py" -o compare_latency.py
```

> **Draft review:** The companion files are available from the preview links above. The `curl` URLs will serve them after publication.

## Step 2: Download the checkpoint and simulator assets

```bash
mkdir -p vendor artifacts/models/nvidia
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA/Isaac-GR00T.git vendor/Isaac-GR00T
GIT_LFS_SKIP_SMUDGE=1 git -C vendor/Isaac-GR00T checkout --detach 9c7e746b2cd37a810070a98ef41d290a07e806c2
git -C vendor/Isaac-GR00T lfs pull --include='demo_data/libero_demo/**' --exclude=''
git -C vendor/Isaac-GR00T submodule update --init --depth 1 external_dependencies/LIBERO
curl -fL https://www.jetson-ai-lab.com/code-samples/groot_n17_on_thor/groot-n17-trt-optimization.patch \
  -o artifacts/jal.patch
printf '%s\n' '6a818bfcbb149c0f0efa8567ecc0fd6909613bc53e00e08f66bb81c8c601d5a3  artifacts/jal.patch' | sha256sum --check -
git -C vendor/Isaac-GR00T apply ../../artifacts/jal.patch

python3 -m venv .hf
.hf/bin/pip install huggingface_hub==1.8.0
.hf/bin/hf auth login
.hf/bin/hf download nvidia/GR00T-N1.7-LIBERO \
  --revision 2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21 \
  --include 'libero_10/*' --local-dir artifacts/models/GR00T-N1.7-LIBERO
.hf/bin/hf download nvidia/Cosmos-Reason2-2B \
  --revision 9ce19a195e423419c349abfc86fd07178b230561 \
  --local-dir artifacts/models/nvidia/Cosmos-Reason2-2B
```

Make a small checkpoint overlay that points to the downloaded backbone. It links the original weight shards and retains both camera keys.

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

models = Path("artifacts/models")
source = models / "GR00T-N1.7-LIBERO/libero_10"
overlay = models / "groot-libero-twocam"
overlay.mkdir()
for path in source.iterdir():
    if path.name not in {"config.json", "processor_config.json", ".cache"}:
        (overlay / path.name).symlink_to(os.path.relpath(path, overlay))

backbone = "/comparison/artifacts/models/nvidia/Cosmos-Reason2-2B"
config = json.loads((source / "config.json").read_text())
processor = json.loads((source / "processor_config.json").read_text())
assert config["num_inference_timesteps"] == 4
kwargs = processor["processor_kwargs"]
video = kwargs["modality_configs"]["libero_sim"]["video"]
assert video["modality_keys"] == ["image", "wrist_image"] and video["delta_indices"] == [0]
config["model_name"] = kwargs["model_name"] = backbone
for name, data in [("config.json", config), ("processor_config.json", processor)]:
    (overlay / name).write_text(json.dumps(data, indent=2))
print("Cameras:", video["modality_keys"], "history indices:", video["delta_indices"])
PY
```

Expected: `Cameras: ['image', 'wrist_image'] history indices: [0]`.

## Step 3: Build and start the pinned Docker environment

All three backends use the same derived image. The FlashRT base is pinned by digest. The Dockerfile adds the LIBERO simulator and TensorRT export tools to its reference Python environment.

```bash
BASE=ghcr.io/flashrt-project/flashrt-thor@sha256:87ad7bce8635207425ac5e90a61d3d83c0ba1042fa28eef7d067474fee872c5c
docker pull "$BASE"
docker build --build-arg BASE_IMAGE="$BASE" \
  -f Dockerfile.twocam -t groot-twocam-thor:local scripts

run() {
  docker run --rm --pull never --network none --runtime nvidia --gpus all \
    --shm-size=8g -v "$PWD:/comparison" groot-twocam-thor:local "$@"
}
run -c 'import torch, tensorrt, modelopt; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(), torch.__version__, tensorrt.__version__, modelopt.__version__)'
run scripts/check_simulator_imports.py
```

Expect NVIDIA Thor with compute capability `(11, 0)`, PyTorch 2.12, TensorRT 10.16, ModelOpt 0.39.0, and ten simulator tasks. The container reads local model files with networking disabled.

## Step 4: Build the two-camera TensorRT engines

```bash
checkpoint=/comparison/artifacts/models/groot-libero-twocam
isaac=/comparison/vendor/Isaac-GR00T
trt_out=/comparison/artifacts/jal_trt_twocam

run "$isaac/scripts/deployment/build_trt_pipeline.py" \
  --model-path "$checkpoint" \
  --dataset-path "$isaac/demo_data/libero_demo" \
  --calib-dataset-path "$isaac/demo_data/libero_demo" \
  --embodiment-tag LIBERO_PANDA \
  --execution-profile optimized --quantization mixed_nvfp4 \
  --calib-size 10 --batch-size 1 --steps export,build --output-dir "$trt_out"
find artifacts/jal_trt_twocam/engines -name '*.engine'
```

Expect eight engine files. The JAL recipe uses ten official demo observations to set quantization scales. The engines bake in this calibration and the two-camera shapes.

## Step 5: Run inference and compare latency

Before timing, verify that the CPU, GPU GPC, and EMC minimum and maximum clocks match. MAXN alone leaves clocks free to vary. Keep the verification output with your timing results.

```bash
sudo jetson_clocks --show | tee artifacts/benchmark-clocks.txt
python3 - <<'PY_CLOCKS'
import re
from pathlib import Path

rows = Path("artifacts/benchmark-clocks.txt").read_text().splitlines()
clocks = [row for row in rows if re.match(r"^(cpu\d+:|gpu-gpc-\d+ |EMC )", row)]
assert any(row.startswith("cpu") for row in clocks), "CPU clocks unavailable"
assert any(row.startswith("gpu-gpc-") for row in clocks), "GPU GPC clocks unavailable"
assert any(row.startswith("EMC ") for row in clocks), "EMC clock unavailable"
for row in clocks:
    low, high = (int(re.search(key + r"=(\d+)", row)[1]) for key in ("MinFreq", "MaxFreq"))
    if low != high:
        raise SystemExit(f"Clock is unlocked: {row}. Run sudo jetson_clocks and check again.")
print("CPU, GPU GPC, and EMC clocks are fixed")
PY_CLOCKS
```

Prepare the three dual-camera observations once. The runner uses a separate initial state for FlashRT calibration. Then time each backend in its own container using the saved observations and diffusion noise.

```bash
run compare_latency.py prepare --checkpoint "$checkpoint" --output artifacts/latency
run compare_latency.py measure torch --checkpoint "$checkpoint" \
  --source "$isaac" --output artifacts/latency
run compare_latency.py measure trt --checkpoint "$checkpoint" \
  --source "$isaac" --engines "$trt_out/engines" --output artifacts/latency
run compare_latency.py measure flashrt --checkpoint "$checkpoint" --output artifacts/latency
run compare_latency.py summarize --output artifacts/latency
```

Each `measure` command prints the median for its three tasks. The summary checks that preprocessed inputs match and writes `artifacts/latency/latency.json` with 900 timings per backend, median and P95 latency, and per-task action cosine and absolute error against PyTorch. Expect finite actions of shape `[1, 16, 7]`. For a quick check, use `--smoke` with `prepare` and use a separate output directory throughout. Smoke mode times three calls per backend on one task.

Restore the saved host clocks after timing.

```bash
sudo jetson_clocks --restore "$clock_settings"
```

## Step 6: Reproduce the LIBERO-10 task scores

Run one seed first. It scores 500 episodes per backend and takes several hours on one Thor. The complete ten-seed run takes substantially longer.

```bash
captures=/comparison/artifacts/captures/libero-suite
run -m groot_compare.libero_suite calibrate \
  --output artifacts/rollouts/setup --checkpoint "$checkpoint" --source-dir "$isaac" \
  --calibration-dir "$captures" --seed 20261026

run_round() (
  set -e
  seed="$1"
  output="artifacts/rollouts/seed-$seed"
  for task in {0..9}; do
    for backend in torch trt flashrt; do
      run -m groot_compare.libero_suite "$backend" "$task" \
        --output "$output" --seed "$seed" --checkpoint "$checkpoint" \
        --source-dir "$isaac" --calibration-dir "$captures" --engine-dir "$trt_out/engines" || exit $?
    done
  done
  run -m groot_compare.verify_suite "$output" "$seed"
)
run_round 20261026
```

The runner executes eight actions per decision, uses five settling steps, and allows 720 control steps per episode. It reuses one reserved FlashRT calibration observation per task, separate from the 50 scored states. It keeps JAL engine calibration fixed. A successful verification writes `verified-summary.json` with `verified: true`, 1,500 scored episodes, and 500 unique task/state cases per backend.

To complete the same ten seeds and check the published counts:

```bash
for seed in 20261027 20261028 20261029 20261030 20261102 20261103 20261104 20261105 20261106; do
  run_round "$seed" || break
done
python3 results/verify_libero_counts.py
```

## Troubleshooting

### Higher latency than the benchmark

Check that the CPU, GPU GPC, and EMC are locked. MAXN alone allows the clocks to vary. Run `sudo jetson_clocks` on the host and repeat the clock check in Step 5. In a separate A/B on one dual-camera input, locking clocks reduced the median from **48.32 to 36.68 ms** for JAL TensorRT and from **44.45 to 33.71 ms** for FlashRT (100 calls per backend per condition).

| Symptom | Check |
| --- | --- |
| Hugging Face returns 401/403 | Accept the Cosmos model terms, then check `.hf/bin/hf auth whoami` on the host. |
| Unknown backbone or offline load error | Keep `nvidia/Cosmos-Reason2-2B` in the local backbone path. Check the overlay and Docker mount. |
| Camera mismatch or TensorRT shape error | Confirm the two camera names and rebuild engines from this checkpoint overlay. Use a fresh output directory for engine export. |
| EGL cannot initialize | Check the NVIDIA Container Toolkit runtime and graphics capability. Run the simulator import check before launching the benchmark. |
| Processed inputs differ | Check the checkpoint revision and camera order. The latency runner requires exact fingerprints; the rollout verifier reports bounded EGL rounding differences. |

The validation used an anonymous digest pull, a fresh derived container, and eight freshly built two-camera engines. Existing model files were verified against the pinned revisions and mounted read-only. The download commands above prepare the same assets for a new host.

## References

- [Isaac GR00T 1.7 on Jetson Thor](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/)
- [FlashRT GR00T Thor guide](https://github.com/flashrt-project/FlashRT/blob/e4bd9277521d137fcded8fa5ccb6091c05f5cd4b/docs/thor/groot-n17.md)
