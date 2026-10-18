"""Execute one local GPU evaluation stage in a disposable interpreter."""

import json
import os
from pathlib import Path
import sys
import traceback
from contextlib import nullcontext

# Match imports used by KernelBench regardless of the caller's directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# The precision validator belongs to the surrounding PolyLLMAgent project.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def cpu_tree(value):
    """Detach nested fixture tensors from the reference CUDA context."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    return value


def save_reference_outputs(source, config, device):
    """Save one trial at a time; no previous trial's GPU output stays alive."""
    import torch
    from src import eval as kernel_eval

    directory = Path(config["correctness_artifact_dir"])
    precision = kernel_eval.get_torch_dtype_from_string(config["precision"])
    Model, get_init_inputs, get_inputs = kernel_eval.load_original_model_and_inputs(source, {})
    kernel_eval.set_seed(42)
    init_inputs = get_init_inputs()
    torch.save(cpu_tree(init_inputs), directory / "init.pt")
    init_inputs = [kernel_eval._process_input_tensor(x, device, config["backend"], precision)
                   for x in init_inputs]
    with torch.no_grad():
        kernel_eval.set_seed(42)
        model = Model(*init_inputs).to(device=device, dtype=precision)
        torch.manual_seed(42)
        seeds = [torch.randint(0, 2**32 - 1, (1,)).item()
                 for _ in range(config["num_correct_trials"])]
        for trial, seed in enumerate(seeds):
            kernel_eval.set_seed(seed)
            inputs = get_inputs()
            inputs = [kernel_eval._process_input_tensor(x, device, config["backend"], precision)
                      for x in inputs]
            # Preserve pre-forward inputs even if the reference mutates them.
            torch.save(cpu_tree(inputs), directory / f"inputs-{trial}.pt")
            kernel_eval.set_seed(seed)
            output = model(*inputs)
            torch.cuda.synchronize(device=device)
            torch.save(cpu_tree(output), directory / f"output-{trial}.pt")
            del output, inputs
        # Separate fixture used if correctness passes and custom timing starts.
        kernel_eval.set_seed(42)
        torch.save(cpu_tree(get_inputs()), directory / "performance-inputs.pt")
    return {"trials": len(seeds)}


def replay_reference_source(directory, num_trials):
    """A fixture reader replaces reference computation in the kernel process.

    KernelBench retains its existing comparison, tolerance, compilation and
    timing logic. ModelNew uses the same seed-42 initialization as before.
    """
    return f'''
import torch
from pathlib import Path
_directory = Path({str(directory)!r})
_trial = 0
def _load(name):
    return torch.load(_directory / name, map_location="cpu", weights_only=False)
def get_init_inputs():
    return _load("init.pt")
def get_inputs():
    if _trial >= {num_trials}:
        return _load("performance-inputs.pt")
    return _load(f"inputs-{{_trial}}.pt")
class Model(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
    def forward(self, *inputs):
        global _trial
        output = _load(f"output-{{_trial}}.pt")
        _trial += 1
        return output.to(device=inputs[0].device)
'''


def execute(request):
    import torch
    from scripts import run_and_check

    config = request["configs"]
    run_and_check.kernel_utils.set_gpu_arch(config["gpu_arch"])
    device = torch.device("cuda:0")
    stage = request["stage"]
    print(f"[INFO] Local {stage} stage pid={os.getpid()}", flush=True)
    if stage == "reference_outputs":
        return save_reference_outputs(request["ref_arch_src"], config, device)
    if stage == "kernel":
        contract = json.loads(config.get("precision_contract_json") or "null")
        capture = nullcontext([])
        if contract:
            from agent_tools.precision_validation import capture_generated_dots
            capture = capture_generated_dots(request["kernel_src"], contract)
        with capture as reports:
            result = run_and_check.evaluate_single_sample_src(
                replay_reference_source(config["correctness_artifact_dir"], config["num_correct_trials"]),
                request["kernel_src"], config, device
            )
        if result is None:
            raise RuntimeError("Kernel evaluation returned no result")
        return {"kernel_result": result.model_dump(), "instruction_reports": reports}
    if stage not in {"eager", "compile"}:
        raise ValueError(f"Unknown stage: {stage}")
    result = run_and_check.measure_program_time(
        ref_arch_name="Reference Program",
        ref_arch_src=request["ref_arch_src"],
        num_trials=config["num_perf_trials"],
        use_torch_compile=stage == "compile",
        torch_compile_backend="inductor",
        torch_compile_options="default",
        device=device,
        precision=run_and_check.kernel_eval.get_torch_dtype_from_string(config["precision"]),
    )
    if result is None:
        raise RuntimeError("Reference measurement returned no result; see worker output")
    return result


def main():
    request_path, result_path = map(Path, sys.argv[1:])
    try:
        result = execute(json.loads(request_path.read_text(encoding="utf-8")))
        payload = {"ok": True, "result": result}
        exit_code = 0
    except BaseException:
        payload = {"ok": False, "error": traceback.format_exc()}
        exit_code = 1
    result_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
