from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

HELLASWAG_DEFAULT_PATH = "./datasets/hellaswag/data/test-00000-of-00001.parquet"
HELLASWAG_PROMPT_HEADER = (
    "Read the context carefully and choose the most plausible ending "
    "among the four options."
)


@dataclass(frozen=True)
class PromptJob:
    tag: str
    group: str
    text: str
    metadata: Dict[str, Any]

def build_hellaswag_prompt(context: str, endings: Sequence[str]) -> str:
    ctx = str(context).strip()
    options = [str(ending).strip() for ending in endings]
    if not ctx:
        raise ValueError("HellaSwag context is empty")
    if len(options) != 4:
        raise ValueError("HellaSwag item must contain exactly four endings")
    if any(not option for option in options):
        raise ValueError("HellaSwag endings must be nonempty")
    choice_block = "\n".join(
        f"({chr(ord('A') + index)}) {option}"
        for index, option in enumerate(options)
    )
    return (
        f"{HELLASWAG_PROMPT_HEADER}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Endings:\n{choice_block}\n\n"
        "Return only the option letter and the exact ending text.\nAnswer:"
    )

def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

def jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): jsonify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(value) for value in obj]
    if isinstance(obj, torch.Tensor):
        return jsonify(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

def _safe_sample_tag(value: str, fallback: str) -> str:
    text = str(value or "").strip()
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in text
    ).strip("._-")
    return cleaned or fallback

def _deduplicate_prompt_jobs(
    jobs: List[PromptJob],
) -> List[PromptJob]:
    seen: Dict[str, int] = {}
    unique: List[PromptJob] = []
    for index, job in enumerate(jobs):
        base = _safe_sample_tag(job.tag, f"sample_{index:04d}")
        count = seen.get(base, 0)
        seen[base] = count + 1
        final_tag = base if count == 0 else f"{base}_{count}"
        unique.append(
            PromptJob(
                tag=final_tag,
                group=str(job.group or "custom"),
                text=job.text.strip(),
                metadata=dict(job.metadata),
            )
        )
    return unique

def _parse_hellaswag_activities(value: str) -> Optional[List[str]]:
    requested = [part.strip() for part in str(value or "all").split(",")]
    requested = [part for part in requested if part]
    if not requested or requested == ["all"]:
        return None
    return list(dict.fromkeys(requested))

def _hellaswag_label_index(label: Any) -> Optional[int]:
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    try:
        index = int(label)
    except (TypeError, ValueError):
        return None
    return index if 0 <= index < 4 else None

def _hellaswag_job_from_record(
    *,
    source_path: Path,
    record_index: int,
    record: Dict[str, Any],
) -> PromptJob:
    context = str(record.get("ctx", "")).strip()
    endings_value = record.get("endings", [])
    if isinstance(endings_value, np.ndarray):
        endings_value = endings_value.tolist()
    if not isinstance(endings_value, (list, tuple)):
        raise ValueError(
            f"{source_path}:{record_index} has a non-list 'endings' field"
        )
    endings = [str(ending).strip() for ending in endings_value]
    activity = str(record.get("activity_label", "hellaswag")).strip() or "hellaswag"
    source_id = str(record.get("source_id", "")).strip()
    ind = record.get("ind", record_index)
    tag_seed = source_id or f"ind_{ind}"
    tag = f"hellaswag_{_safe_sample_tag(tag_seed, f'row_{record_index:04d}')}"
    label_index = _hellaswag_label_index(record.get("label"))
    gold_ending = endings[label_index] if label_index is not None else None
    return PromptJob(
        tag=tag,
        group="hellaswag",
        text=build_hellaswag_prompt(context, endings),
        metadata={
            "dataset": "HellaSwag",
            "activity_label": activity,
            "ind": ind,
            "source_id": source_id or None,
            "source_file": source_path.as_posix(),
            "source_format": "parquet",
            "record_index": int(record_index),
            "ctx": context,
            "ctx_a": record.get("ctx_a"),
            "ctx_b": record.get("ctx_b"),
            "endings": endings,
            "split": record.get("split"),
            "split_type": record.get("split_type"),
            "gold_label_index": label_index,
            "gold_ending": gold_ending,
        },
    )

def load_hellaswag_jobs(
    dataset_path: Path,
    activities: Optional[Sequence[str]] = None,
) -> List[PromptJob]:
    source_path = dataset_path.expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"HellaSwag parquet not found: {source_path}. "
            f"Expected {HELLASWAG_DEFAULT_PATH}"
        )
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to load HellaSwag parquet: pip install pandas pyarrow"
        ) from exc
    try:
        frame = pd.read_parquet(source_path)
    except Exception as exc:
        raise ValueError(f"Failed to read HellaSwag parquet {source_path}: {exc}") from exc
    if frame.empty:
        raise ValueError(f"No HellaSwag problems loaded from {source_path}")
    allowed = None if activities is None else set(activities)
    jobs: List[PromptJob] = []
    for record_index, row in frame.iterrows():
        record = row.to_dict()
        activity = str(record.get("activity_label", "")).strip()
        if allowed is not None and activity not in allowed:
            continue
        jobs.append(
            _hellaswag_job_from_record(
                source_path=source_path,
                record_index=(
                    int(record_index)
                    if isinstance(record_index, (int, np.integer))
                    else len(jobs)
                ),
                record=record,
            )
        )
    if not jobs:
        raise ValueError(f"No HellaSwag problems loaded from {source_path}")
    return jobs

def load_prompt_jobs(args: argparse.Namespace) -> List[PromptJob]:
    jobs: List[PromptJob] = []
    if args.prompt_file:
        path = Path(args.prompt_file)
        jobs.append(
            PromptJob(
                "custom_0",
                "custom",
                path.read_text(encoding="utf-8"),
                {"source_file": path.as_posix(), "source_format": "text"},
            )
        )
    elif args.prompt_jsonl:
        path = Path(args.prompt_jsonl)
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                raise ValueError(
                    f"{path}:{line_number} must contain an object with nonempty 'text'"
                )
            fallback = f"sample_{line_number - 1:04d}"
            tag = item.get("tag", item.get("id", fallback))
            group = item.get("group", "custom")
            jobs.append(
                PromptJob(
                    str(tag),
                    str(group),
                    str(item["text"]),
                    {
                        "source_file": path.as_posix(),
                        "source_format": "jsonl",
                        **{key: value for key, value in item.items() if key != "text"},
                    },
                )
            )
    elif args.prompt_dir:
        root = Path(args.prompt_dir)
        if not root.is_dir():
            raise NotADirectoryError(root)
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}
        )
        for index, path in enumerate(paths):
            relative = path.relative_to(root)
            group = relative.parts[0] if len(relative.parts) > 1 else "custom"
            jobs.append(
                PromptJob(
                    _safe_sample_tag(
                        relative.with_suffix("").as_posix(), f"sample_{index:04d}"
                    ),
                    group,
                    path.read_text(encoding="utf-8"),
                    {"source_file": path.as_posix(), "source_format": "text"},
                )
            )
    else:
        jobs.extend(
            load_hellaswag_jobs(
                Path(args.hellaswag_path or HELLASWAG_DEFAULT_PATH),
                _parse_hellaswag_activities(args.hellaswag_activities),
            )
        )

    jobs = _deduplicate_prompt_jobs(jobs)
    jobs = [job for job in jobs if job.text]
    if not jobs:
        raise ValueError("No nonempty prompt samples were found")

    limit = int(args.samples_per_group)
    if limit > 0:
        generator = np.random.default_rng(int(args.sample_seed))
        selected_tags = set()
        group_names = list(dict.fromkeys(job.group for job in jobs))
        for group in group_names:
            candidates = [job for job in jobs if job.group == group]
            if len(candidates) <= limit:
                selected_tags.update(job.tag for job in candidates)
            else:
                picks = generator.choice(len(candidates), size=limit, replace=False)
                selected_tags.update(candidates[int(index)].tag for index in picks)
        jobs = [job for job in jobs if job.tag in selected_tags]
    return jobs


def encode_prompt(tokenizer, text: str) -> List[int]:
    try:
        from encoding_dsv4 import encode_messages

        text = encode_messages(
            [{"role": "user", "content": text}], thinking_mode="chat"
        )
    except Exception:
        pass
    return tokenizer.encode(text)


def limit_prompt_tokens(
    input_ids: Sequence[int],
    max_tokens: int,
    policy: str,
    head_fraction: float = 0.50,
) -> Tuple[List[int], Dict[str, Any]]:
    token_ids = list(input_ids)
    limit = int(max_tokens)
    if limit < 4:
        raise ValueError("max_tokens must be at least four")
    if len(token_ids) <= limit:
        return token_ids, {
            "original_n_tokens": len(token_ids),
            "n_tokens": len(token_ids),
            "truncated": False,
            "truncation_policy": "none",
        }
    if policy == "error":
        raise ValueError(
            f"Encoded prompt has {len(token_ids)} tokens but the limit is {limit}. "
            "Increase --max-tokens and --max-seq-len, or use "
            "--long-prompt-policy head_tail."
        )
    if policy != "head_tail":
        raise ValueError(f"Unsupported long-prompt policy: {policy}")
    fraction = min(max(float(head_fraction), 0.10), 0.90)
    head_count = max(1, min(limit - 1, int(round(limit * fraction))))
    tail_count = limit - head_count
    limited = token_ids[:head_count] + token_ids[-tail_count:]
    return limited, {
        "original_n_tokens": len(token_ids),
        "n_tokens": len(limited),
        "truncated": True,
        "truncation_policy": "head_tail",
        "kept_head_tokens": int(head_count),
        "kept_tail_tokens": int(tail_count),
        "removed_middle_tokens": int(len(token_ids) - limit),
    }


def token_strings(tokenizer, input_ids: Sequence[int]) -> List[str]:
    ids = [int(token_id) for token_id in input_ids]
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(converter):
        try:
            converted = converter(ids)
            if isinstance(converted, list) and len(converted) == len(ids):
                return [str(token) for token in converted]
        except Exception:
            pass
    pieces: List[str] = []
    for token_id in ids:
        try:
            pieces.append(str(tokenizer.decode([token_id])))
        except Exception:
            pieces.append(f"<token:{token_id}>")
    return pieces


def attach_snapshot_tokens(
    result: Dict[str, Any],
    input_ids: Sequence[int],
    tokenizer,
) -> None:
    ids = [int(token_id) for token_id in input_ids]
    pieces = token_strings(tokenizer, ids)
    clustering = result.get("final_layer_clustering", {})
    final_labels = [int(label) for label in clustering.get("token_labels", [])]
    probabilities = list(clustering.get("membership_probabilities", []))
    outliers = list(clustering.get("outlier_scores", []))
    if final_labels and len(final_labels) != len(ids):
        raise ValueError("Final cluster labels and token ids must have the same length")
    result["tokens"] = [
        {
            "token_index": index,
            "token_id": token_id,
            "token_text": pieces[index],
            "final_cluster_label": final_labels[index] if final_labels else None,
            "final_membership_probability": (
                probabilities[index] if len(probabilities) == len(ids) else None
            ),
            "final_outlier_score": (
                outliers[index] if len(outliers) == len(ids) else None
            ),
        }
        for index, token_id in enumerate(ids)
    ]
    for snapshot in result.get("snapshots_sphere", []):
        indices = [int(index) for index in snapshot.get("token_index", [])]
        snapshot["token_id"] = [ids[index] for index in indices]
        snapshot["token_text"] = [pieces[index] for index in indices]
