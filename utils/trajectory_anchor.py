import random
import re
from typing import Iterable, Optional


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def extract_think_text(generation: str) -> Optional[str]:
    """Return the text inside a complete <think>...</think> block."""
    if not generation:
        return None
    match = THINK_RE.search(generation)
    if not match:
        return None
    text = re.sub(r"\s+", " ", match.group(1)).strip()
    return text or None


def _finish_reason_is_usable(reason) -> bool:
    if reason is None:
        return True
    if isinstance(reason, str):
        return reason == "" or reason.lower() == "stop"
    return False


def choose_teacher_trajectory(
    generations: Iterable[str] | None,
    correctness_math_verify: Iterable[bool] | None = None,
    correctness_llama: Iterable[bool] | None = None,
    finish_reasons: Iterable[str] | None = None,
    *,
    selection: str = "shortest",
    rng: random.Random | None = None,
) -> Optional[str]:
    """Pick one correct and complete DeepSeek-R1 trace for trajectory anchoring."""
    generations = _as_list(generations)
    if not generations:
        return None

    math_flags = _as_list(correctness_math_verify)
    llama_flags = _as_list(correctness_llama)
    finish_reasons = _as_list(finish_reasons)

    def flag_at(flags, index: int) -> bool:
        return bool(flags[index]) if index < len(flags) else False

    def finish_at(index: int):
        return finish_reasons[index] if index < len(finish_reasons) else None

    candidates_by_priority: list[list[str]] = [[], []]
    for index, generation in enumerate(generations):
        think_text = extract_think_text(generation)
        if think_text is None:
            continue
        if not _finish_reason_is_usable(finish_at(index)):
            continue
        if flag_at(math_flags, index):
            candidates_by_priority[0].append(think_text)
        elif flag_at(llama_flags, index):
            candidates_by_priority[1].append(think_text)

    candidates = candidates_by_priority[0] or candidates_by_priority[1]
    if not candidates:
        return None

    if selection == "shortest":
        return min(candidates, key=len)
    if selection == "longest":
        return max(candidates, key=len)
    if selection == "random":
        rng = rng or random
        return rng.choice(candidates)
    raise ValueError(
        f"Unsupported teacher_selection={selection!r}; choose shortest, random, or longest."
    )


STEP_BOUNDARY_RE = re.compile(
    r"(?:\n\s*\n+)|"
    r"(?=\n?\s*(?:\d+[\).]|[-*]\s+|Step\s+\d+[:.)]|First,|Second,|Third,|Finally,))",
    re.IGNORECASE,
)


def split_think_steps(
    think_text: str | None,
    *,
    max_steps: int = 24,
    min_chars: int = 24,
    max_chars: int = 900,
) -> list[str]:
    """Split a thinking trace into natural reasoning units."""
    if not think_text:
        return []

    text = think_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    coarse_parts = [part.strip() for part in STEP_BOUNDARY_RE.split(text) if part.strip()]
    steps: list[str] = []
    for part in coarse_parts:
        if len(part) <= max_chars:
            steps.append(part)
            continue

        sentences = re.split(r"(?<=[.!?。！？])\s+", part)
        buffer = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if buffer and len(buffer) + len(sentence) + 1 > max_chars:
                steps.append(buffer.strip())
                buffer = sentence
            else:
                buffer = f"{buffer} {sentence}".strip()
        if buffer:
            steps.append(buffer.strip())

    merged: list[str] = []
    for step in steps:
        if merged and len(step) < min_chars:
            merged[-1] = f"{merged[-1]} {step}".strip()
        else:
            merged.append(step)

    if len(merged) <= max_steps:
        return merged

    # Keep chronological order while merging adjacent over-segmented steps.
    bucket_size = len(merged) / max_steps
    compacted: list[str] = []
    for bucket_index in range(max_steps):
        start = int(round(bucket_index * bucket_size))
        end = int(round((bucket_index + 1) * bucket_size))
        end = max(end, start + 1)
        compacted.append(" ".join(merged[start:end]).strip())
    return [step for step in compacted if step]


def build_teacher_steps(
    example: dict,
    *,
    selection: str = "shortest",
    max_steps: int = 24,
) -> list[str]:
    teacher = choose_teacher_trajectory(
        example.get("generations"),
        example.get("correctness_math_verify"),
        example.get("correctness_llama"),
        example.get("finish_reasons"),
        selection=selection,
    )
    return split_think_steps(teacher, max_steps=max_steps)
