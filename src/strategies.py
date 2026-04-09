import json

from collections.abc import Callable
from dataclasses import dataclass, asdict, field, fields
from datetime import datetime
from json import JSONDecodeError
from logging import getLogger
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Iterable
from typeguard import check_type, TypeCheckError


log = getLogger(__name__)


class InferenceResultLoadError(ValueError):
    """Raised when InferenceResult cannot be loaded correctly."""

    pass


class InvalidDataset(ValueError):
    """Raised when expected dataset is invalid. E.g. missing files or folders."""

    pass


class ConfigLoadError(ValueError):
    """Raised when benchmark config cannot be loaded from JSON."""

    pass


@dataclass
class AnswerItem:
    problem: str
    answer: str


@dataclass
class StrategySetup:
    """
    Configuration for one strategy of the benchmark.
    """

    strategy: str
    prompts: List[str]

    @classmethod
    def from_dict(cls, data: Dict):
        """Load setup from dictionary."""
        cls._validate_fields(data)

        field_names = {f.name for f in fields(cls)}
        init_kwargs = {name: data[name] for name in field_names}

        return cls(**init_kwargs)

    @classmethod
    def _validate_fields(cls, data: dict[str, Any]) -> None:
        required_fields = {f.name for f in fields(cls)}
        missing = required_fields - set(data.keys())
        if missing:
            raise ConfigLoadError(f"Setup missing required keys: {missing}")

        for f in fields(cls):
            value = data[f.name]
            try:
                check_type(value, f.type)
            except TypeCheckError as e:
                raise ConfigLoadError(
                    f"Type mismatch in setup field '{f.name}': "
                    f"expected {f.type.__name__}, got {type(value).__name__}. Message from type checker: {str(e)}."
                ) from None


@dataclass
class BenchmarkConfig:
    """
    Top-level benchmark configuration.
    Contains a list of per-strategy setups and global benchmark information.
    """

    strategies: List[StrategySetup]
    model: str
    dataset: Path

    @classmethod
    def load(cls, setup_path: str):
        """Load Config from file"""
        path = Path(setup_path)
        if not path.is_file():
            raise FileNotFoundError(f"Setup file not found as {path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except JSONDecodeError as e:
            raise ConfigLoadError(f"Invalid JSON in setup file {path}: {e}") from e

        data["strategies"] = cls._load_setups(data["strategies"])
        data["dataset"] = cls._load_dataset_path(data["dataset"])

        cls._validate_fields(data)

        field_names = {f.name for f in fields(cls)}
        init_kwargs = {name: data[name] for name in field_names}

        return cls(**init_kwargs)

    @classmethod
    def _validate_fields(cls, data: dict[str, Any]) -> None:
        required_fields = {f.name for f in fields(cls)}
        missing = required_fields - set(data.keys())
        if missing:
            raise ConfigLoadError(f"Setup file missing required keys: {missing}")

        for f in fields(cls):
            value = data[f.name]
            try:
                check_type(value, f.type)
            except TypeCheckError as e:
                raise ConfigLoadError(
                    f"Error during loading a setup file. Type mismatch in setup field '{f.name}': "
                    f"expected {f.type.__name__}, got {type(value).__name__}. Message from type checker: {str(e)}."
                ) from None

    @classmethod
    def _load_setups(cls, setups_list: Any) -> List[StrategySetup]:
        if not isinstance(setups_list, list):
            raise ConfigLoadError("Invalid config: 'setups' must be a list")

        setups: List[StrategySetup] = []
        for i, raw in enumerate(setups_list):
            try:
                setups.append(StrategySetup(**raw))
            except Exception as e:
                raise ConfigLoadError(
                    f"Error during config loading. Invalid strategy setup at index {i}. {str(e)}"
                ) from None

        return setups

    @classmethod
    def _load_dataset_path(cls, dataset_path) -> Path:
        """Convert and validate dataset path."""
        if dataset_path is None:
            raise ConfigLoadError(
                "Error during config loading. Path to dataset is required in field 'dataset'."
            )
        if not isinstance(dataset_path, str):
            raise ConfigLoadError(
                "Error during config loading. Field 'dataset' must contain a string path"
            )

        path = Path(dataset_path)
        if not path.exists():
            raise ConfigLoadError(
                f"Error during config loading. Dataset directory specified in config not found: {path}"
            )
        if not path.is_dir():
            raise ConfigLoadError(
                f"Error during config loading. Dataset path specified in config must be a directory: {path}"
            )
        return path


@dataclass 
class StrategyResult:
    strategy: str
    prompts: List[str]
    answers: List[AnswerItem]

    @classmethod
    def from_dict(cls, dict_data: Dict):
        """Create InferenceResult from a dictionary"""
        required_keys = {f.name for f in fields(cls)}

        if not required_keys.issubset(dict_data.keys()):
            missing = required_keys - dict_data.keys()
            raise InferenceResultLoadError(f"Missing required keys: {missing}")

        if not isinstance(dict_data["answers"], list):
            raise InferenceResultLoadError("'answers' must be a list")

        answers: List[AnswerItem] = []
        for i, ans in enumerate(dict_data["answers"]):
            if not isinstance(ans, dict) or {"problem", "answer"} - set(ans.keys()):
                raise InferenceResultLoadError(f"Invalid answer item at index {i}")
            answers.append(AnswerItem(problem=ans["problem"], answer=ans["answer"]))

        return cls(
            prompts=dict_data["prompts"], strategy=dict_data["strategy"], answers=answers
        )

    def save_as_json(self, file_path: str | None = None, indent=4):
        """Save results as a JSON"""
        json_string = self.to_json(indent)

        if not file_path:
            file_path = "results_" + self.model + "_" + self.end_time + ".json"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json_string)
        return file_path



@dataclass
class BenchmarkResult:
    model: str
    dataset: str
    end_time: str = field(init=False)
    results: List[StrategyResult]

    def __post_init__(self):
        self.end_time = datetime.now().isoformat()

    def to_json(self, indent=4):
        """Serialize results as a json-formatted string"""
        json_data = json.dumps(asdict(self), indent=indent, ensure_ascii=False)

        return json_data

    def save_as_json(self, file_path: str | None = None, indent=4):
        """Save results as a JSON"""
        json_string = self.to_json(indent)

        if not file_path:
            file_path = "results_" + self.model + "_" + self.end_time + ".json"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json_string)
        return file_path

    @classmethod
    def from_dict(cls, dict_data: Dict):
        """Create InferenceResult from a dictionary"""
        required_keys = {f.name for f in fields(cls)}

        if not required_keys.issubset(dict_data.keys()):
            missing = required_keys - dict_data.keys()
            raise InferenceResultLoadError(f"Missing required keys: {missing}")

        if not isinstance(dict_data["results"], list):
            raise InferenceResultLoadError("'results' must be a list of per-strategy results")

        results: List[StrategyResult] = []
        for i, ans in enumerate(dict_data["results"]):
            try:
                result = StrategyResult.from_dict(ans)
            except InferenceResultLoadError as e:
                raise InferenceResultLoadError(f"Invalid results for strategy at index {i}. {str(e)}") from None
            results.append(result)

        instance = cls(
            model=dict_data["model"], results=results, dataset=dict_data["dataset"]
        )

        instance.end_time = dict_data["end_time"]
        return instance

    @classmethod
    def load_from_json_file(cls, json_path: str):
        """Load InferenceResult from JSON file"""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"Path is not a file: {path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except JSONDecodeError as e:
            raise InferenceResultLoadError(
                f"JSON decode error for file {path}: {str(e)}"
            ) from e

        return cls.from_dict(data)


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


def direct(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    dataset: Path
) -> StrategyResult:
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

    return StrategyResult(prompts=prompts, answers=answers, strategy="direct")


def descriptive_direct(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    dataset: Path,
) -> StrategyResult:
    single_prompt = prompts[0]
    collage_prompt = prompts[1]

    try:
        tasks_folders = load_folder(dataset)
    except InvalidDataset as e:
        log.error("Dataset folder missing: %s", e)
        return StrategyResult(prompts=prompts, model=setup.model, answers=[])

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

    return StrategyResult(prompts=prompts, answers=answers, strategy="descriptive-direct")


def descriptive_iterative(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    dataset: Path,
) -> StrategyResult:
    
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

    return StrategyResult(prompts=prompts, answers=answers, strategy="descriptive-iterative")


def contrastive_direct(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    dataset: Path,
) -> StrategyResult:
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

    return StrategyResult(prompts=prompts, answers=answers, strategy="contrastive-direct")


def contrastive_iterative(
    ask_model: Callable[[str, Path], str],
    reload_context: Callable[[], None],
    prompts: List[str],
    dataset: Path,
) -> StrategyResult:
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

    return StrategyResult(prompts=prompts, answers=answers, strategy="contrastive-iterative")


StrategyFunc = Callable[
    [Callable[[str, Path], str], Callable[[], None], List[str], Path],
    StrategyResult,
]

class BongBench:
    STRATEGIES: Dict[str, StrategyFunc] = {
        "direct": direct,
        "descriptive-direct": descriptive_direct,
        "descriptive-iterative": descriptive_iterative,
        "contrastive-direct": contrastive_direct,
        "contrastive-iterative": contrastive_iterative,
    }

    def __init__(self, config: BenchmarkConfig):
        self.__config__ = config
        self.setups_by_strategy: Dict[str, List[StrategySetup]] = {}
        for setup in config.strategies:
            self.setups_by_strategy.setdefault(setup.strategy, []).append(setup)

    def run(
        self,
        ask_model: Callable[[str, Path], str],
        reload_context: Callable[[], None],
        strategies: Optional[Iterable[str]] = None,
    ) -> BenchmarkResult:
        if strategies is None:
            strategies = list(self.setups_by_strategy.keys())

        results: List[StrategyResult] = []
        for strategy_name in strategies:
            if strategy_name not in self.STRATEGIES:
                raise ValueError(f"Error during benchmark run. Unknown strategy: {strategy_name} (was set up in config).")

            strategy_func = self.STRATEGIES[strategy_name]

            for setup in self.setups_by_strategy[strategy_name]:
                result = strategy_func(ask_model, reload_context, setup.prompts, self.__config__.dataset)
                results.append(result)

        return BenchmarkResult(self.__config__.model, str(self.__config__.dataset), results)
