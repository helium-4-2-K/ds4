#!/usr/bin/env python3
"""Generate DS4 GLM 5.2 TP4 rank-plan and layout files from Bird shards.

This importer targets the validated Bird/vLLM per-rank checkpoint layout:

  model-rank-{rank}-part-{0..19}.safetensors
  mtp-rank-{rank}-part-0.safetensors
  integrity-manifest.json

The emitted DS4 layout is intentionally a compact real-tensor frontier for the
first real-host loader run: it maps representative rank-local safetensors tensor
slices with exact dtype, shape, byte-range, and SHA-256 metadata while the rank
plan still declares the complete shard-file set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
from typing import Any

TP_SIZE = 4
DCP_SIZE = 4
PP_SIZE = 1
Q_HEADS = 64
EXPERTS = 256
VOCAB_SIZE = 154880
BASE_SHARDS = 20
MTP_SHARDS = 1
DTYPE_MAP = {
    "F32": "f32",
    "BF16": "bf16",
    "F8_E4M3": "fp8_e4m3",
    "F8_E4M3FN": "fp8_e4m3",
    "FP8_E4M3": "fp8_e4m3",
    "I32": "i32",
}
DTYPE_SIZE = {
    "f32": 4,
    "bf16": 2,
    "fp8_e4m3": 1,
    "i32": 4,
}


def load_safetensors_header(path: pathlib.Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as fp:
        raw_len = fp.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: missing safetensors header length")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = fp.read(header_len)
        if len(header) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
    obj = json.loads(header)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    return int(header_len), obj


def tensor_names(path: pathlib.Path) -> list[str]:
    _, header = load_safetensors_header(path)
    return [k for k in header.keys() if k != "__metadata__"]


def has_any(path: pathlib.Path, needles: tuple[str, ...]) -> bool:
    return any(any(needle in name for needle in needles)
               for name in tensor_names(path))


def rank_file_records(manifest: dict[str, Any],
                      rank: int,
                      key: str,
                      expected_count: int) -> list[dict[str, Any]]:
    records = [x for x in manifest.get(key, []) if x.get("rank") == rank]
    records.sort(key=lambda x: int(x.get("part", 0)))
    if len(records) != expected_count:
        raise ValueError(
            f"rank {rank}: expected {expected_count} {key}, found {len(records)}")
    return records


def dtype_name(raw: str) -> str | None:
    return DTYPE_MAP.get(raw)


def element_count(shape: list[Any]) -> int | None:
    if not shape:
        return None
    n = 1
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            return None
        n *= dim
    return n


def file_slice_sha256(path: pathlib.Path, offset: int, length: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        fp.seek(offset)
        remaining = length
        while remaining:
            chunk = fp.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{path}: truncated tensor slice")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def tensor_record(root: pathlib.Path,
                  file_name: str,
                  tensor_name: str,
                  metadata: dict[str, Any],
                  header_len: int) -> dict[str, Any]:
    raw_dtype = metadata.get("dtype")
    dtype = dtype_name(str(raw_dtype))
    shape = metadata.get("shape")
    offsets = metadata.get("data_offsets")
    if dtype is None:
        raise ValueError(f"{file_name}:{tensor_name}: unsupported dtype {raw_dtype}")
    if not isinstance(shape, list):
        raise ValueError(f"{file_name}:{tensor_name}: missing shape")
    elements = element_count(shape)
    if elements is None:
        raise ValueError(f"{file_name}:{tensor_name}: unsupported scalar/empty shape")
    if (not isinstance(offsets, list) or len(offsets) != 2 or
            not all(isinstance(x, int) and x >= 0 for x in offsets) or
            offsets[1] <= offsets[0]):
        raise ValueError(f"{file_name}:{tensor_name}: invalid data_offsets")
    byte_length = offsets[1] - offsets[0]
    expected = elements * DTYPE_SIZE[dtype]
    if byte_length != expected:
        raise ValueError(
            f"{file_name}:{tensor_name}: data length {byte_length} != {expected}")
    byte_offset = 8 + header_len + offsets[0]
    path = root / file_name
    return {
        "tensor_name": tensor_name,
        "file_path": file_name,
        "byte_offset": byte_offset,
        "byte_length": byte_length,
        "sha256": file_slice_sha256(path, byte_offset, byte_length),
        "dtype": dtype,
        "shape": shape,
        "element_count": elements,
    }


def find_tensor(root: pathlib.Path,
                records: list[dict[str, Any]],
                predicate) -> dict[str, Any] | None:
    for rec in records:
        file_name = str(rec["name"])
        path = root / file_name
        header_len, header = load_safetensors_header(path)
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if predicate(name, metadata):
                return tensor_record(root, file_name, name, metadata, header_len)
    return None


def choose_role_tensors(root: pathlib.Path,
                        base_records: list[dict[str, Any]],
                        mtp_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    all_records = base_records + mtp_records

    q_head = find_tensor(
        root,
        all_records,
        lambda name, meta: (
            ("mla_attn.q_b_proj.weight_packed" in name or
             "fused_qkv_a_proj.weight_packed" in name) and
            dtype_name(str(meta.get("dtype"))) is not None and
            element_count(meta.get("shape", [])) is not None))
    mla_kv = find_tensor(
        root,
        all_records,
        lambda name, meta: (
            "kv_b_proj.weight_packed" in name and
            dtype_name(str(meta.get("dtype"))) is not None and
            element_count(meta.get("shape", [])) is not None))
    expert = find_tensor(
        root,
        all_records,
        lambda name, meta: (
            ("routed_experts.w13_weight" in name or
             "routed_experts.w2_weight" in name) and
            dtype_name(str(meta.get("dtype"))) is not None and
            element_count(meta.get("shape", [])) is not None))
    vocab = find_tensor(
        root,
        all_records,
        lambda name, meta: (
            name == "lm_head.weight" and
            dtype_name(str(meta.get("dtype"))) is not None and
            element_count(meta.get("shape", [])) is not None))

    missing = [k for k, v in {
        "q_head": q_head,
        "mla_kv": mla_kv,
        "expert": expert,
        "vocab": vocab,
    }.items() if v is None]
    if missing:
        raise ValueError(f"cannot find representative tensors for: {', '.join(missing)}")
    return {"q_head": q_head, "mla_kv": mla_kv, "expert": expert, "vocab": vocab}


def write_rank_plan(path: pathlib.Path,
                    model_root: pathlib.Path,
                    rank: int,
                    fabric_addr: str,
                    fabric_data_plane: str,
                    base_records: list[dict[str, Any]],
                    mtp_records: list[dict[str, Any]]) -> None:
    lines = [
        "tp_size=4",
        "dcp_size=4",
        "pp_size=1",
        "rank_count=4",
        "host_count=4",
        f"rank={rank}",
        "model_name=glm-5.2",
        f"checkpoint_root={model_root}",
        f"fabric_addr={fabric_addr}",
        f"fabric_data_plane={fabric_data_plane}",
    ]
    for rec in base_records:
        lines.append(f"rank{rank}.base_shard={rec['name']}")
    for rec in mtp_records:
        lines.append(f"rank{rank}.mtp_shard={rec['name']}")
    path.write_text("\n".join(lines) + "\n")


def write_entry(lines: list[str],
                *,
                tensor: dict[str, Any],
                role: str,
                rank: int,
                q_span: tuple[int, int] | None = None,
                expert_span: tuple[int, int] | None = None,
                vocab_span: tuple[int, int] | None = None) -> None:
    lines.extend([
        "[entry]",
        f"tensor_name={tensor['tensor_name']}",
        f"role={role}",
        "distribution_scope=rank_local_shard",
        f"rank={rank}",
        f"file_path={tensor['file_path']}",
        f"byte_offset={tensor['byte_offset']}",
        f"byte_length={tensor['byte_length']}",
        f"sha256={tensor['sha256']}",
        "replicated=false",
        f"dtype={tensor['dtype']}",
        "shape=[" + ",".join(str(x) for x in tensor["shape"]) + "]",
        f"element_count={tensor['element_count']}",
    ])
    if q_span is not None:
        lines.extend([f"q_head_start={q_span[0]}", f"q_head_end={q_span[1]}"])
    if expert_span is not None:
        lines.extend([
            f"expert_start={expert_span[0]}",
            f"expert_end={expert_span[1]}",
        ])
    if vocab_span is not None:
        lines.extend([f"vocab_start={vocab_span[0]}", f"vocab_end={vocab_span[1]}"])


def write_layout(path: pathlib.Path,
                 manifest: dict[str, Any],
                 rank: int,
                 base_records: list[dict[str, Any]],
                 mtp_records: list[dict[str, Any]],
                 role_tensors: dict[str, dict[str, Any]]) -> None:
    q_start = rank * (Q_HEADS // TP_SIZE)
    q_end = q_start + (Q_HEADS // TP_SIZE)
    vocab_start = (VOCAB_SIZE * rank) // TP_SIZE
    vocab_end = (VOCAB_SIZE * (rank + 1)) // TP_SIZE

    lines = [
        "# Generated by misc/glm52_bird_layout.py from Bird/vLLM rank shards.",
        "# Entries map representative real safetensors tensor slices with",
        "# dtype, shape, byte-range, and slice SHA-256 metadata.",
        "format_version=ds4-shard-layout/v1",
        f"model_config_sha256={manifest.get('artifact_config_sha256', '')}",
        "tp_size=4",
        "dcp_size=4",
        "rank_count=4",
        "required_base_shards=20",
        "required_mtp_shards=1",
        "has_mtp=true",
    ]

    write_entry(lines,
                tensor=role_tensors["q_head"],
                role="q_head",
                rank=rank,
                q_span=(q_start, q_end))
    write_entry(lines,
                tensor=role_tensors["mla_kv"],
                role="mla_kv",
                rank=rank)
    write_entry(lines,
                tensor=role_tensors["expert"],
                role="expert",
                rank=rank,
                expert_span=(0, EXPERTS))
    write_entry(lines,
                tensor=role_tensors["vocab"],
                role="vocab",
                rank=rank,
                vocab_span=(vocab_start, vocab_end))
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", required=True, type=pathlib.Path)
    ap.add_argument("--rank", required=True, type=int)
    ap.add_argument("--fabric-addr", required=True)
    ap.add_argument("--fabric-data-plane", default="crs812-200g")
    ap.add_argument("--rank-plan-out", required=True, type=pathlib.Path)
    ap.add_argument("--layout-out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    if args.rank < 0 or args.rank >= TP_SIZE:
        raise SystemExit("--rank must be in [0,4)")
    root = args.model_root.expanduser().resolve()
    manifest_path = root / "integrity-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("tensor_parallel_size") != TP_SIZE:
        raise SystemExit("integrity manifest is not TP4")

    base_records = rank_file_records(manifest, args.rank, "base_files", BASE_SHARDS)
    mtp_records = rank_file_records(manifest, args.rank, "mtp_files", MTP_SHARDS)
    role_tensors = choose_role_tensors(root, base_records, mtp_records)
    write_rank_plan(args.rank_plan_out, root, args.rank, args.fabric_addr,
                    args.fabric_data_plane, base_records, mtp_records)
    write_layout(args.layout_out, manifest, args.rank, base_records,
                 mtp_records, role_tensors)
    print(f"rank_plan={args.rank_plan_out}")
    print(f"layout={args.layout_out}")
    print("roles=" + json.dumps({
        role: {
            "tensor_name": tensor["tensor_name"],
            "file_path": tensor["file_path"],
            "byte_length": tensor["byte_length"],
            "dtype": tensor["dtype"],
        }
        for role, tensor in role_tensors.items()
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
