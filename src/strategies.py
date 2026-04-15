from collections.abc import Callable
from enum import Enum
from functools import wraps
from logging import getLogger
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from results import StrategyResult, AnswerItem

log = getLogger(__name__)

StrategyFuncUnwrapped = Callable[
    [Callable[[str, Path], str], Callable[[], None], List[str], Path],
    str,
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

COLLAGE_NAME = "collage.png"
LEFT_FOLDER = "left"
RIGHT_FOLDER = "right"
PAIRS_FOLDER = "pairs"


class InvalidDataset(ValueError):
    """Raised when expected dataset is invalid. E.g. missing files or folders."""

    pass


def load_file(file: Path) -> Path:
    if not file.is_file():
        raise InvalidDataset(f"File {file} does not exist")
    return file


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
    def wrapper(
        ask_model: Callable[[str, Path], str],
        reload_context: Callable[[], None],
        prompts: List[str],
        dataset: Path,
    ) -> StrategyResult:
        strategy_name = func.__name__

        try:
            tasks_folders = load_folder(dataset)
        except InvalidDataset as e:
            log.error("Dataset folder missing: %s", e)
            return StrategyResult(
                strategy=strategy_name, prompts=prompts, answers=[], skipped=[]
            )

        answers = []
        skipped = []
        for problem in tqdm(
            tasks_folders,
            desc=f"Benchmark for strategy {strategy_name:<25}",
            unit="problem",
        ):
            reload_context()

            try:
                answer = func(ask_model, reload_context, prompts, problem)
            except InvalidDataset as e:
                log.error(f"Error during solving problem {problem.name}: {str(e)}")
                skipped.append(problem.name)
                continue

            answers.append(AnswerItem(problem=problem.name, answer=answer))

        return StrategyResult(
            strategy=strategy_name, prompts=prompts, answers=answers, skipped=skipped
        )

    return wrapper


@strategy_func
def direct(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    problem: Path,
) -> str:
    collage = load_file(problem / COLLAGE_NAME)
    return ask_model(prompts[0], collage)


@strategy_func
def descriptive_direct(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    problem: Path,
) -> str:
    single_prompt = prompts[0]
    collage_prompt = prompts[1]

    collage = load_file(problem / COLLAGE_NAME)

    lefts = load_folder(problem / "left")
    rights = load_folder(problem / "right")

    lefts_desc = get_descriptions(lefts, ask_model, single_prompt)
    reload_context()
    rights_desc = get_descriptions(rights, ask_model, single_prompt)

    return ask_model(collage_prompt.format(lefts_desc, rights_desc), collage)


@strategy_func
def descriptive_iterative(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    problem: Path,
) -> str:
    iterative_prompts = prompts[:3]
    collage_prompt = prompts[3]

    collage = load_file(problem / COLLAGE_NAME)

    lefts = load_folder(problem / "left")
    rights = load_folder(problem / "right")

    left_concept = get_iterative_concept(lefts, ask_model, iterative_prompts)
    reload_context()
    right_concept = get_iterative_concept(rights, ask_model, iterative_prompts)

    return ask_model(collage_prompt.format(left_concept, right_concept), collage)


@strategy_func
def contrastive_direct(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    problem: Path,
) -> str:
    pair_prompt = prompts[0]
    collage_prompt = prompts[1]

    collage = load_file(problem / COLLAGE_NAME)
    pairs = load_folder(problem / PAIRS_FOLDER)

    pairs_decs = get_descriptions(pairs, ask_model, pair_prompt)

    return ask_model(collage_prompt.format(pairs_decs), collage)


@strategy_func
def contrastive_iterative(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    problem: Path,
) -> str:
    pairs = load_folder(problem / PAIRS_FOLDER)
    return get_iterative_concept(pairs, ask_model, prompts)


STRATEGIES: Dict[StrategyName, StrategyFunc] = {
    StrategyName.DIRECT: direct,
    StrategyName.DESCRIPTIVE_DIRECT: descriptive_direct,
    StrategyName.DESCRIPTIVE_ITERATIVE: descriptive_iterative,
    StrategyName.CONTRASTIVE_DIRECT: contrastive_direct,
    StrategyName.CONTRASTIVE_ITERATIVE: contrastive_iterative,
}
