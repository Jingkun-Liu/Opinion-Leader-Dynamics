from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch


def _token_string(token_config_value: Any) -> Optional[str]:
    if token_config_value is None:
        return None
    if isinstance(token_config_value, dict):
        return token_config_value.get("content")
    return str(token_config_value)

def load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    path = Path(tokenizer_path)
    config_path = path / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            config = {}
        if "model_type" in config:
            return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)

    tokenizer_file = path / "tokenizer.json"
    tokenizer_config_file = path / "tokenizer_config.json"
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"No tokenizer.json under {path}")
    tokenizer_config = (
        json.loads(tokenizer_config_file.read_text())
        if tokenizer_config_file.is_file()
        else {}
    )
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        bos_token=_token_string(tokenizer_config.get("bos_token")),
        eos_token=_token_string(tokenizer_config.get("eos_token")),
        pad_token=_token_string(tokenizer_config.get("pad_token"))
        or _token_string(tokenizer_config.get("eos_token")),
        unk_token=_token_string(tokenizer_config.get("unk_token")),
        model_max_length=tokenizer_config.get("model_max_length", 1048576),
        clean_up_tokenization_spaces=tokenizer_config.get(
            "clean_up_tokenization_spaces", False
        ),
    )

def _find_complete_cuda_toolkit() -> Tuple[
    Optional[Path], Optional[Path], List[str]
]:
    candidates: List[Path] = []
    examined: List[str] = []

    def add_candidate(value: Optional[str | Path]) -> None:
        if not value:
            return
        candidate = Path(value).expanduser()
        try:
            candidate = candidate.resolve()
        except OSError:
            candidate = candidate.absolute()
        if candidate not in candidates:
            candidates.append(candidate)

    add_candidate(os.environ.get("CUDA_HOME"))
    add_candidate(os.environ.get("CUDA_PATH"))
    nvcc_on_path = shutil.which("nvcc")
    if nvcc_on_path:
        add_candidate(Path(nvcc_on_path).resolve().parent.parent)
    add_candidate(os.environ.get("CONDA_PREFIX"))
    add_candidate(sys.prefix)
    add_candidate("/usr/local/cuda-12.8")
    add_candidate("/usr/local/cuda")
    try:
        for path in sorted(Path("/usr/local").glob("cuda-*"), reverse=True):
            add_candidate(path)
    except OSError:
        pass

    for cuda_home in candidates:
        nvcc = cuda_home / "bin" / "nvcc"
        headers = (
            cuda_home / "include" / "cuda_runtime.h",
            cuda_home
            / "targets"
            / "x86_64-linux"
            / "include"
            / "cuda_runtime.h",
        )
        has_nvcc = nvcc.is_file()
        has_headers = any(path.is_file() for path in headers)
        examined.append(
            f"{cuda_home} (nvcc={'yes' if has_nvcc else 'no'}, "
            f"cuda_runtime.h={'yes' if has_headers else 'no'})"
        )
        if has_nvcc and has_headers:
            return cuda_home, nvcc, examined
    return None, None, examined

def load_dsv4(
    checkpoint_path: str,
    config_path: str,
    max_sequence_length: int = 1024,
    tokenizer_path: Optional[str] = None,
    tilelang_backend: str = "auto",
):
    inference_directory = str(Path(config_path).resolve().parent)
    encoding_directory = str(
        (Path(inference_directory).parent / "encoding").resolve()
    )
    for path in (inference_directory, encoding_directory):
        if path not in sys.path:
            sys.path.insert(0, path)

    import torch.distributed as dist

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA on this node")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA "
            "devices are visible"
        )
    torch.cuda.set_device(local_rank)

    major, minor = torch.cuda.get_device_capability(local_rank)
    architecture_suffix = "a" if (major, minor) == (9, 0) else ""
    tilelang_target = f"cuda -arch=sm_{major}{minor}{architecture_suffix}"
    os.environ["TILELANG_TARGET"] = tilelang_target
    os.environ["TILELANG_DEFAULT_TARGET"] = tilelang_target

    cuda_home, nvcc_path, examined_cuda_paths = _find_complete_cuda_toolkit()
    requested_backend = str(
        os.environ.get("DSV4_TILELANG_BACKEND", tilelang_backend)
    ).strip().lower()
    if requested_backend not in {"auto", "tvm_ffi", "nvrtc"}:
        raise ValueError("TileLang backend must be one of: auto, tvm_ffi, nvrtc")
    if requested_backend == "auto":
        selected_backend = "tvm_ffi" if cuda_home is not None else "nvrtc"
    else:
        selected_backend = requested_backend

    if selected_backend == "tvm_ffi" and cuda_home is None:
        examined = "\n  - ".join(examined_cuda_paths) or "no candidate paths"
        raise RuntimeError(
            "TileLang backend=tvm_ffi requires a host CUDA Toolkit containing "
            "both bin/nvcc and cuda_runtime.h, but none was found. Examined:\n"
            f"  - {examined}\n"
            "Either install CUDA Toolkit 12.8 and export CUDA_HOME/PATH, or "
            "run this script with --tilelang-backend nvrtc."
        )

    if cuda_home is not None:
        os.environ["CUDA_HOME"] = str(cuda_home)
        cuda_binary_directory = str(cuda_home / "bin")
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if cuda_binary_directory not in path_entries:
            os.environ["PATH"] = (
                cuda_binary_directory + os.pathsep + os.environ.get("PATH", "")
            )
    os.environ["TILELANG_EXECUTION_BACKEND"] = selected_backend
    cache_base = Path(os.environ.get("CONDA_PREFIX", inference_directory))
    cache_tag = (
        f"tilelang_dsv4_{selected_backend}_sm_"
        f"{major}{minor}{architecture_suffix}"
    )
    os.environ["TILELANG_CACHE_DIR"] = os.environ.get(
        "DSV4_TILELANG_CACHE_DIR", str(cache_base / cache_tag)
    )

    if rank == 0:
        toolkit_text = str(cuda_home) if cuda_home is not None else "not found"
        print(
            f"[TileLang] target={tilelang_target}, backend={selected_backend}, "
            f"host_toolkit={toolkit_text}, cache={os.environ['TILELANG_CACHE_DIR']}",
            flush=True,
        )
        if selected_backend == "nvrtc" and cuda_home is None:
            print(
                "[TileLang] nvcc was not found; using the NVRTC backend, which "
                "compiles through the CUDA runtime compiler.",
                flush=True,
            )
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")

    from safetensors.torch import load_model
    from model import ModelArgs, Transformer

    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    config["max_seq_len"] = int(max_sequence_length)
    config["max_batch_size"] = 1
    model_args = ModelArgs(
        **{
            key: value
            for key, value in config.items()
            if key in ModelArgs.__dataclass_fields__
        }
    )
    with torch.device("cuda"):
        model = Transformer(model_args)

    shard = os.path.join(
        checkpoint_path, f"model{rank}-mp{world_size}.safetensors"
    )
    if not os.path.isfile(shard):
        raise FileNotFoundError(
            f"Missing {shard}. The checkpoint must be converted with mp={world_size}."
        )
    load_model(model, shard, strict=False)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_path or checkpoint_path)
    torch.set_default_device("cuda")
    return model, tokenizer, rank, world_size
