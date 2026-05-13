import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset, load_from_disk

from run_inference import LatentReasoningInteractive

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from utils.reward_func import accuracy_reward


def _get_field(example: dict, field: str):
    value = example
    for part in field.split("."):
        value = value[part]
    return value


def _normalize_answer(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = value[0] if value else ""
    text = str(value).strip()
    if "####" in text:
        text = text.split("####")[-1].strip()
    return text


def _load_eval_dataset(args):
    if args.dataset_path:
        return load_from_disk(args.dataset_path)
    if args.dataset_config:
        return load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    return load_dataset(args.dataset_name, split=args.split)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BAT-LRT on a generic HF/local dataset.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--reasoning_net_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--reasoning_net_type", default="adaptive_anchor", choices=["fixed", "adaptive_anchor"])
    parser.add_argument("--latent_trajectory_length", type=int, default=384)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--min_latent_chunks", type=int, default=2)
    parser.add_argument("--router_tau", type=float, default=1.0)
    parser.add_argument("--fixed_latent_chunks", type=int, default=None)
    parser.add_argument("--print_latent_budget", action="store_true")

    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--prompt_field", default="problem")
    parser.add_argument("--answer_field", default="answer")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_jsonl", required=True)

    parser.add_argument("--prompt_suffix", default="")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--prompt_max_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.dataset_name and not args.dataset_path:
        raise ValueError("Provide either --dataset_name or --dataset_path.")
    return args


def main():
    args = parse_args()
    dataset = _load_eval_dataset(args)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    model = LatentReasoningInteractive(
        model_path=args.model_path,
        reasoning_net_path=args.reasoning_net_path,
        checkpoint_path=args.checkpoint_path,
        reasoning_net_type=args.reasoning_net_type,
        latent_trajectory_length=args.latent_trajectory_length,
        chunk_size=args.chunk_size,
        min_latent_chunks=args.min_latent_chunks,
        router_tau=args.router_tau,
        fixed_latent_chunks=args.fixed_latent_chunks,
        print_latent_budget=args.print_latent_budget,
        prompt_max_length=args.prompt_max_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0.0
    total = 0
    start_time = time.time()
    with output_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(dataset):
            prompt = str(_get_field(example, args.prompt_field)) + args.prompt_suffix
            gold = _normalize_answer(_get_field(example, args.answer_field))
            prediction = model.generate(prompt, temperature=args.temperature)
            reward = accuracy_reward(
                completions=[[{"role": "assistant", "content": prediction}]],
                solution=[gold],
            )[0]
            score = 0.0 if reward is None else float(reward)
            correct += score
            total += 1
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "prompt": prompt,
                        "gold": gold,
                        "prediction": prediction,
                        "score": score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            if total % 10 == 0:
                print(f"[{total}/{len(dataset)}] accuracy={correct / total:.4f}")

    summary = {
        "dataset": args.dataset_name or args.dataset_path,
        "split": args.split,
        "total": total,
        "accuracy": correct / max(total, 1),
        "seconds": time.time() - start_time,
        "output_jsonl": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
