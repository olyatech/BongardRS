import json

from dataclasses import dataclass, asdict, field, fields
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import List, Dict

class InferenceResultLoadError(ValueError):
    """Raised when InferenceResult cannot be loaded correctly."""

    pass


@dataclass
class AnswerItem:
    problem: str
    answer: str


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
            prompts=dict_data["prompts"],
            strategy=dict_data["strategy"],
            answers=answers,
        )


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
            raise InferenceResultLoadError(
                "'results' must be a list of per-strategy results"
            )

        results: List[StrategyResult] = []
        for i, ans in enumerate(dict_data["results"]):
            try:
                result = StrategyResult.from_dict(ans)
            except InferenceResultLoadError as e:
                raise InferenceResultLoadError(
                    f"Invalid results for strategy at index {i}. {str(e)}"
                ) from None
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
