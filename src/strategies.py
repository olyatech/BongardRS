from collections.abc import Callable
from enum import Enum
from functools import wraps
from logging import getLogger
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Iterable

from results import BenchmarkResult, StrategyResult, AnswerItem

log = getLogger(__name__)

StrategyFuncUnwrapped = Callable[
    [Callable[[str, Path], str], Callable[[], None], List[str], Path],
    List[AnswerItem],
]

StrategyFunc = Callable[
    [Callable[[str, Path], str], Callable[[], None], List[str], Path],
    StrategyResult,
]


class StrategyName(str, Enum):
    DIRECT = "direct"
    DESCRIPTIVE_DIRECT = "descriptive-direct"
    DESCRIPTIVE_ITERATIVE = "descriptive-iterative"
    CONTRASTIVE_DIRECT = "contrastive-direct"
    CONTRASTIVE_ITERATIVE = "contrastive-iterative"


PROMPTS_PER_STRATEGY: Dict[StrategyName, int] = {
    StrategyName.DIRECT: 1,
    StrategyName.DESCRIPTIVE_DIRECT: 2,
    StrategyName.DESCRIPTIVE_ITERATIVE: 4,
    StrategyName.CONTRASTIVE_DIRECT: 2,
    StrategyName.CONTRASTIVE_ITERATIVE: 3,
}


class InvalidDataset(ValueError):
    """Raised when expected dataset is invalid. E.g. missing files or folders."""

    pass


def load_folder(folder: Path) -> List[Path]:
    if not folder.exists():
        raise InvalidDataset(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise InvalidDataset(f"Path is not a directory: {folder}")

    files = [file for file in folder.iterdir()]
    files = sorted(files, key=lambda file: file.name)

    return files


def get_descriptions(
    pics: List[Path], ask_model: Callable[[str, Path], str], prompt: str
) -> List[str]:
    answers = []
    for pic in pics:
        answers.append(ask_model(prompt, pic))

    return answers


def get_iterative_concept(
    pics: List[Path], ask_model: Callable[[str, Path], str], prompts: List[str]
) -> str:
    answer = ask_model(prompts[0], pics[0])
    for pair in pics[1:-1]:
        answer = ask_model(prompts[1], pair)

    answer = ask_model(prompts[2], pics[-1])
    return answer


def strategy_func(func: StrategyFuncUnwrapped):
    @wraps(func)
    def inner(
        ask_model: Callable[[str, Path], str],
        reload_context: Callable[[], None],
        prompts: List[str],
        dataset: Path,
    ) -> StrategyResult:
        strategy_name = func.__name__
        answers = func(ask_model, reload_context, prompts, dataset)
        return StrategyResult(strategy=strategy_name, prompts=prompts, answers=answers)

    return inner


@strategy_func
def direct(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    dataset: Path,
) -> List[AnswerItem]:
    tasks_folders = [file for file in dataset.iterdir()]
    tasks_folders = sorted(tasks_folders, key=lambda folder: folder.name)

    answers = []
    for problem in tqdm(tasks_folders, desc="Solving problems", unit="problem"):
        collage = problem / "collage.png"
        if not collage.is_file():
            log.error("Skipping problem %s: no collage.png", problem.name)
            continue

        reload_context()
        answer = ask_model(prompts[0], problem / "collage.png")
        answers.append(AnswerItem(problem=problem.name, answer=answer))

    return answers


@strategy_func
def descriptive_direct(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    dataset: Path,
) -> List[AnswerItem]:
    single_prompt = prompts[0]
    collage_prompt = prompts[1]

    try:
        tasks_folders = load_folder(dataset)
    except InvalidDataset as e:
        log.error("Dataset folder missing: %s", e)
        return []

    answers = []
    for problem in tqdm(tasks_folders, desc="Solving problems", unit="problem"):
        reload_context()

        collage = problem / "collage.png"
        if not collage.is_file():
            log.error("Skipping problem %s: no collage.png", problem.name)
            continue

        try:
            lefts = load_folder(problem / "left")
            rights = load_folder(problem / "right")
        except InvalidDataset:
            log.error(
                "Skipping problem %s: missing left/right subfolders", problem.name
            )
            continue

        lefts_desc = get_descriptions(lefts, ask_model, single_prompt)
        rights_desc = get_descriptions(rights, ask_model, single_prompt)

        answer = ask_model(collage_prompt.format(lefts_desc, rights_desc), collage)
        answers.append(AnswerItem(problem=problem.name, answer=answer))

    return answers


@strategy_func
def descriptive_iterative(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    dataset: Path,
) -> List[AnswerItem]:

    collage_prompt = prompts[3]
    tasks_folders = load_folder(dataset)

    answers = []
    for problem in tqdm(tasks_folders, desc="Solving problems", unit="problem"):
        reload_context()

        collage = problem / "collage.png"
        if not collage.is_file():
            log.error("Skipping problem %s: no collage.png", problem.name)
            continue

        try:
            lefts = load_folder(problem / "left")
            rights = load_folder(problem / "right")
        except InvalidDataset:
            log.error(
                "Skipping problem %s: missing left/right subfolders", problem.name
            )
            continue

        left_concept = get_iterative_concept(lefts, ask_model, prompts[:3])
        right_concept = get_iterative_concept(rights, ask_model, prompts[:3])

        answer = ask_model(collage_prompt.format(left_concept, right_concept), collage)
        answers.append(AnswerItem(problem=problem.name, answer=answer))

    return answers


@strategy_func
def contrastive_direct(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    dataset: Path,
) -> List[AnswerItem]:
    pair_prompt = prompts[0]
    collage_prompt = prompts[1]

    tasks_folders = load_folder(dataset)

    answers = []
    for problem in tqdm(tasks_folders, desc="Solving problems", unit="problem"):
        reload_context()

        collage = problem / "collage.png"
        if not collage.is_file():
            log.error("Skipping problem %s: no collage.png", problem.name)
            continue

        try:
            pairs = load_folder(problem / "pairs")
        except InvalidDataset:
            log.error("Skipping problem %s: missing pairs subfolder", problem.name)
            continue
        pairs_decs = get_descriptions(pairs, ask_model, pair_prompt)

        answer = ask_model(collage_prompt.format(pairs_decs), collage)
        answers.append(AnswerItem(problem=problem.name, answer=answer))

    return answers


@strategy_func
def contrastive_iterative(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    dataset: Path,
) -> List[AnswerItem]:
    tasks_folders = load_folder(dataset)

    answers = []
    for problem in tqdm(tasks_folders, desc="Solving problems", unit="problem"):
        reload_context()

        try:
            pairs = load_folder(problem / "pairs")
        except InvalidDataset:
            log.error("Skipping problem %s: missing pairs subfolder", problem.name)
            continue

        answer = get_iterative_concept(pairs, ask_model, prompts)
        answers.append(AnswerItem(problem=problem.name, answer=answer))

    return answers


STRATEGIES: Dict[StrategyName, StrategyFunc] = {
    StrategyName.DIRECT: direct,
    StrategyName.DESCRIPTIVE_DIRECT: descriptive_direct,
    StrategyName.DESCRIPTIVE_ITERATIVE: descriptive_iterative,
    StrategyName.CONTRASTIVE_DIRECT: contrastive_direct,
    StrategyName.CONTRASTIVE_ITERATIVE: contrastive_iterative,
}
