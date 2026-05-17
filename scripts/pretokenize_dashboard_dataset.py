from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from huggingface_hub import HfApi
from sae_lens.config import PretokenizeRunnerConfig, special_token
from sae_lens.pretokenize_runner import pretokenize_dataset
from transformers import AutoTokenizer

from interpretune.utils.dashboard_pretokenization import (
    PretokenizationResult,
    build_dashboard_metadata,
    load_custom_pretokenization_module,
    materialize_tokenized_dataset,
    maybe_build_custom_metadata,
    maybe_configure_custom_parser,
    maybe_load_custom_pretokenization_settings,
)


DEFAULT_CUSTOM_DATASET_MODULE = "it_examples.utils.dashboard_pretokenization_rte"


def build_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretokenize datasets for Neuronpedia dashboard generation.")
    parser.add_argument("--custom-dataset-module")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-name", "--dataset-config-name", dest="dataset_name")
    parser.add_argument("--dataset-split", "--split", dest="split", default="train")
    parser.add_argument(
        "--dataset-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--data-files", nargs="+")
    parser.add_argument("--data-dir")
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--column-name", default="text")
    parser.add_argument(
        "--use-chat-formatting",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--pretokenize-batch-size", type=int, default=1000)
    parser.add_argument("--begin-batch-token", type=special_token, default="bos")
    parser.add_argument("--begin-sequence-token", type=special_token, default=None)
    parser.add_argument("--sequence-separator-token", type=special_token, default="bos")
    parser.add_argument(
        "--disable-concat-sequences",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--hf-num-shards", type=int, default=64)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument(
        "--hf-is-private-repo",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-tokenized-rows", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    base_parser = build_base_parser()
    known_args, _ = base_parser.parse_known_args(argv)
    custom_module = None
    if known_args.custom_dataset_module:
        custom_module = load_custom_pretokenization_module(known_args.custom_dataset_module)
    maybe_configure_custom_parser(custom_module, base_parser)
    args = base_parser.parse_args(argv)
    args.custom_module = custom_module
    return args


def build_pretokenize_config(args: argparse.Namespace) -> PretokenizeRunnerConfig:
    save_path = args.save_path or args.output_dir
    return PretokenizeRunnerConfig(
        tokenizer_name=args.tokenizer_name,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_trust_remote_code=args.dataset_trust_remote_code,
        split=args.split,
        data_files=args.data_files,
        data_dir=args.data_dir,
        num_proc=args.num_proc,
        context_size=args.context_size,
        column_name=args.column_name,
        use_chat_formatting=args.use_chat_formatting,
        shuffle=args.shuffle,
        seed=args.seed,
        streaming=args.streaming,
        pretokenize_batch_size=args.pretokenize_batch_size,
        begin_batch_token=args.begin_batch_token,
        begin_sequence_token=args.begin_sequence_token,
        sequence_separator_token=args.sequence_separator_token,
        disable_concat_sequences=args.disable_concat_sequences,
        save_path=str(save_path) if save_path is not None else None,
        hf_repo_id=args.hf_repo_id,
        hf_num_shards=args.hf_num_shards,
        hf_revision=args.hf_revision,
        hf_is_private_repo=args.hf_is_private_repo,
    )


def run_dashboard_pretokenization(
    args: argparse.Namespace,
) -> tuple[PretokenizationResult, PretokenizeRunnerConfig, dict[str, Any]]:
    cfg = build_pretokenize_config(args)
    dataset = load_dashboard_dataset(cfg, max_rows=args.max_rows)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.model_max_length = sys.maxsize

    if args.custom_module is not None:
        settings = maybe_load_custom_pretokenization_settings(args.custom_module, args)
        result = args.custom_module.pretokenize_custom_dataset(dataset, tokenizer, cfg, settings)
        custom_metadata = maybe_build_custom_metadata(
            args.custom_module,
            args=args,
            settings=settings,
            result=result,
            tokenizer=tokenizer,
            cfg=cfg,
        )
    else:
        tokenized_dataset = pretokenize_dataset(cast(Dataset, dataset), tokenizer, cfg)
        result = PretokenizationResult(
            tokenized_dataset=materialize_tokenized_dataset(
                tokenized_dataset,
                max_tokenized_rows=args.max_tokenized_rows,
            ),
            effective_context_size=cfg.context_size,
            prompt_lengths=None,
        )
        custom_metadata = {}

    if args.max_tokenized_rows is not None and len(result.tokenized_dataset) > args.max_tokenized_rows:
        result = PretokenizationResult(
            tokenized_dataset=result.tokenized_dataset.select(range(args.max_tokenized_rows)),
            effective_context_size=result.effective_context_size,
            prompt_lengths=(
                result.prompt_lengths[: args.max_tokenized_rows] if result.prompt_lengths is not None else None
            ),
        )

    metadata_cfg = replace(
        cfg,
        context_size=result.effective_context_size or cfg.context_size,
        disable_concat_sequences=bool(custom_metadata.get("disable_concat_sequences", cfg.disable_concat_sequences)),
    )
    metadata = build_dashboard_metadata(metadata_cfg, custom_metadata=custom_metadata)
    return result, metadata_cfg, metadata


def load_dashboard_dataset(
    cfg: PretokenizeRunnerConfig,
    *,
    max_rows: int | None,
) -> Dataset | IterableDataset:
    dataset = load_dataset(  # type: ignore[call-overload]
        cfg.dataset_path,
        name=cfg.dataset_name,
        data_dir=cfg.data_dir,
        data_files=cfg.data_files,
        split=cfg.split,  # type: ignore[arg-type]
        streaming=cfg.streaming,  # type: ignore[arg-type]
        trust_remote_code=cfg.dataset_trust_remote_code,
    )
    if isinstance(dataset, DatasetDict):
        raise ValueError("Dataset has multiple splits. Must provide a 'split' param.")
    if max_rows is None:
        return dataset
    if isinstance(dataset, Dataset):
        return dataset.select(range(min(len(dataset), max_rows)))
    return dataset.take(max_rows)


def persist_dashboard_dataset(
    dataset: Dataset,
    *,
    cfg: PretokenizeRunnerConfig,
    metadata: dict[str, Any],
    force: bool,
) -> None:
    if cfg.save_path is not None:
        save_path = Path(cfg.save_path)
        if save_path.exists():
            if not force:
                raise FileExistsError(f"Output directory already exists: {save_path}. Pass --force to replace it.")
            shutil.rmtree(save_path)
        dataset.save_to_disk(str(save_path))
        write_metadata_file(save_path, metadata)

    if cfg.hf_repo_id is not None:
        dataset.push_to_hub(
            repo_id=cfg.hf_repo_id,
            num_shards=cfg.hf_num_shards,
            private=cfg.hf_is_private_repo,
            revision=cfg.hf_revision,
        )
        upload_metadata_to_hugging_face_hub(cfg=cfg, metadata=metadata)


def write_metadata_file(output_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path = output_dir / "sae_lens.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def upload_metadata_to_hugging_face_hub(
    *,
    cfg: PretokenizeRunnerConfig,
    metadata: dict[str, Any],
) -> None:
    if cfg.hf_repo_id is None:
        return

    meta_io = io.BytesIO()
    meta_io.write(json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8"))
    meta_io.seek(0)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=meta_io,
        path_in_repo="sae_lens.json",
        repo_id=cfg.hf_repo_id,
        repo_type="dataset",
        commit_message="Add sae_lens metadata",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.save_path is None and args.output_dir is None and args.hf_repo_id is None:
        raise ValueError("Provide --save-path/--output-dir and/or --hf-repo-id.")

    result, cfg, metadata = run_dashboard_pretokenization(args)
    persist_dashboard_dataset(
        result.tokenized_dataset,
        cfg=cfg,
        metadata=metadata,
        force=args.force,
    )
    destination = cfg.save_path or cfg.hf_repo_id or "<in-memory>"
    print(
        f"Saved {len(result.tokenized_dataset)} tokenized prompts to {destination} "
        f"with context_size={result.effective_context_size or cfg.context_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
