"""Prepare shared GR00T inputs, time one backend per process, and compare results."""
import argparse
import json
import os
import time
from pathlib import Path


BACKENDS = ("torch", "trt", "flashrt")
ACTION_SHAPE = (16, 7)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def prepare(args):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    import numpy as np
    import torch
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv
    from libero.libero import benchmark
    from libero.libero.utils import get_libero_path
    from groot_compare.backends.flashrt import capture_calibration
    from groot_compare.closed_loop import case_seed, noise_record_id, task_camera_keys
    from groot_compare.common import shared_noise

    seed = 20261026
    task_camera_keys(args.checkpoint)
    torch.set_num_threads(2)
    suite = benchmark.get_benchmark_dict()["libero_10"](task_order_index=0)
    args.output.mkdir(parents=True)
    tasks = []
    for task_id in ([4] if args.smoke else [4, 6, 9]):
        task = suite.get_task(task_id)
        path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        states = torch.load(path, map_location="cpu", weights_only=False)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = LiberoEnv(task_bddl_file=str(bddl), task_description=task.language)
        try:
            frames = []
            for state_id in [0, 1]:
                env._env.seed(case_seed(seed, "libero_10", task.name, state_id))
                env._env.reset()
                raw = env._env.set_init_state(np.asarray(states[state_id]).copy())
                for _ in range(5):
                    raw, _, _, _ = env._env.step(np.zeros(7))
                flat = env._process_observation(raw)
                frames.append({
                    "video": {key: np.array(flat["video." + key][None, None], copy=True)
                              for key in ("image", "wrist_image")},
                    "state": {key: np.asarray(flat["state." + key], dtype=np.float32)[None, None]
                              for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")},
                    "language": {"annotation.human.action.task_description": [[task.language]]},
                })
        finally:
            env.close()
        capture = str(args.output / f"calibration-task{task_id}.pt")
        capture_calibration(args.checkpoint, frames[1], capture, cpu_threads=2)
        noise = shared_noise(noise_record_id("libero_10", task.name, 0, 0), seed)
        observation = frames[0]
        np.savez(args.output / f"task-{task_id}.npz", noise=noise,
                 **{f"{group}.{key}": value for group in ("video", "state")
                    for key, value in observation[group].items()})
        tasks.append({"id": task_id, "language": observation["language"]})
    write_json(args.output / "cases.json", {"seed": seed, "smoke": args.smoke, "tasks": tasks})



def load_observation(directory, task):
    import numpy as np

    with np.load(directory / f"task-{task['id']}.npz") as saved:
        observation = {"video": {}, "state": {}, "language": task["language"]}
        for name in saved.files:
            if name != "noise":
                group, key = name.split(".", 1)
                observation[group][key] = saved[name]
        return observation, saved["noise"]


def measure(args):
    import numpy as np
    import torch
    from groot_compare.metrics import action_array

    torch.set_num_threads(2)
    cases = json.loads((args.output / "cases.json").read_text())
    warmups, iterations = (2, 3) if cases["smoke"] else (20, 300)
    rows = {}
    runtime = {"torch": str(torch.__version__)}
    for task in cases["tasks"]:
        task_id = task["id"]
        observation, noise = load_observation(args.output, task)
        if args.backend == "flashrt":
            from groot_compare.backends.flashrt import FlashRTBackend

            capture = args.output / f"calibration-task{task_id}.pt"
            backend = FlashRTBackend(args.checkpoint, str(capture), cpu_threads=2)
        elif args.backend == "trt":
            import tensorrt
            from groot_compare.backends.jal import TensorRTBackend

            runtime["tensorrt"] = tensorrt.__version__
            backend = TensorRTBackend(checkpoint=args.checkpoint, source_dir=args.source,
                                      engine_dir=args.engines, cpu_threads=2)
        else:
            from groot_compare.backends.jal import TorchBackend

            backend = TorchBackend(checkpoint=args.checkpoint, source_dir=args.source, cpu_threads=2)

        try:
            fingerprint = backend.input_fingerprints(observation)
            reference = action_array(backend.get_action(observation, noise))
            if reference.shape != ACTION_SHAPE:
                raise ValueError(f"Unexpected action shape: {reference.shape}")
            for _ in range(warmups):
                backend.get_action(observation, noise)
            samples = []
            for _ in range(iterations):
                backend.synchronize()
                start = time.perf_counter()
                actions = backend.get_action(observation, noise)
                backend.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
                if action_array(actions).shape != ACTION_SHAPE:
                    raise ValueError("Unexpected action shape")
            rows[str(task_id)] = {"fingerprint": fingerprint, "samples_ms": samples,
                                  "actions": reference.tolist()}
            print(f"{args.backend} task {task_id}: {np.median(samples):.2f} ms", flush=True)
        finally:
            backend.close()
            del backend
            torch.cuda.empty_cache()
    write_json(args.output / f"{args.backend}.json", {"tasks": rows, "runtime": runtime,
                                                      "warmups": warmups})


def summarize(args):
    import numpy as np
    from groot_compare.metrics import error_metrics

    cases = json.loads((args.output / "cases.json").read_text())
    runs = {name: json.loads((args.output / f"{name}.json").read_text()) for name in BACKENDS}
    tasks = [str(task["id"]) for task in cases["tasks"]]
    expected_calls = 3 if cases["smoke"] else 300
    references = runs["torch"]["tasks"]
    samples, comparisons = {}, {}
    for name, run in runs.items():
        if set(run["tasks"]) != set(tasks):
            raise ValueError(f"Unexpected tasks for {name}")
        samples[name] = []
        for task in tasks:
            row = run["tasks"][task]
            if row["fingerprint"] != references[task]["fingerprint"]:
                raise ValueError(f"Preprocessed inputs differ for {name}, task {task}")
            timings = np.asarray(row["samples_ms"])
            if len(timings) != expected_calls or not np.isfinite(timings).all() or (timings <= 0).any():
                raise ValueError(f"Invalid latency samples for {name}, task {task}")
            samples[name].extend(row["samples_ms"])
            if name != "torch":
                comparisons[f"{name}/task{task}"] = error_metrics(row["actions"], references[task]["actions"])

    results = {name: {"calls": len(values), "median_ms": float(np.median(values)),
                      "p95_ms": float(np.percentile(values, 95))}
               for name, values in samples.items()}
    report = {"action_shape": [1, *ACTION_SHAPE], "matched_processed_inputs": True,
              "seed": cases["seed"], "tasks": [int(task) for task in tasks],
              "torch": runs["torch"]["runtime"]["torch"], "tensorrt": runs["trt"]["runtime"]["tensorrt"],
              "warmups": runs["torch"]["warmups"], "results": results,
              "action_agreement": comparisons, "samples_ms": samples}
    write_json(args.output / "latency.json", report)
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--checkpoint", required=True)
    preparation.add_argument("--smoke", action="store_true")
    timing = commands.add_parser("measure")
    timing.add_argument("backend", choices=BACKENDS)
    timing.add_argument("--checkpoint", required=True)
    timing.add_argument("--source")
    timing.add_argument("--engines")
    summary = commands.add_parser("summarize")
    for command in (preparation, timing, summary):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.output.exists():
            parser.error(f"Output already exists: {args.output}")
        prepare(args)
    elif args.command == "measure":
        if args.backend in ("torch", "trt") and not args.source:
            parser.error(f"measure {args.backend} requires --source")
        if args.backend == "trt" and not args.engines:
            parser.error("measure trt requires --engines")
        measure(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
