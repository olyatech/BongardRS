"""
Result types for the Bongard benchmark.

Defines:
- `AnswerItem`: one model answer for one problem,
- `StrategyResult`: all answers from one strategy,
- `BenchmarkResult`: a bundle of multiple strategy results that can be saved to JSON.
"""

import json

from dataclasses import dataclass, asdict, field, fields
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import List, Dict


class InferenceResultLoadError(ValueError):
    """
    Raised when saved benchmark results cannot be loaded correctly.

    This can happen if:
    - the JSON file is malformed,
    - any required field is missing,
    - or the structure does not match the expected schema.
    """

    pass


@dataclass
class AnswerItem:
    """
    Represents one answer produced by the model for a single Bongard problem.

    For each problem, the model receives a prompt and an image path and returns
    a text answer. This class bundles the problem identifier and the model’s answer.

    Example:
    ```
        AnswerItem(problem="001",
                   answer="Right side contains big objects, while left side contains small ones.")
    ```
    """

    problem: str
    answer: str


@dataclass
class StrategyResult:
    """
    Results produced by a single strategy during the benchmark.

    Each strategy (e.g., ``direct``, ``descriptive_direct``) is executed across
    all problems in the dataset, producing a list of answers and a list of
    problems that were skipped (e.g., missing files). Here these results are stored with
    the strategy name and prompts used.

    Example JSON-like structure:
    ```
        "strategy": "descriptive_direct",
        "prompts": ["Describe the left side.", "Describe the right side."],
        "answers": [
            "problem": "001", "answer": "Left: ...",
            "problem": "002", "answer": "Left: ..."
        ],
        "skipped": ["003"]
    ```
    """

    strategy: str
    prompts: List[str]
    answers: List[AnswerItem]
    skipped: List[str]

    @classmethod
    def from_dict(cls, dict_data: Dict) -> "StrategyResult":
        """
        Load a strategy result from a dictionary (e.g., loaded from JSON).

        The input dictionary must be structured as:
          - "strategy" (str): the strategy name.
          - "prompts" (List[str]): list of prompts used.
          - "answers" (List[Dict[str, str]]): list of answer items, each with
            "problem" (str) and "answer" (str).
          - "skipped" (List[str]): list of problem names that were skipped.

        Args:
            dict_data (Dict[str, Any]): the raw dictionary.

        Returns:
            StrategyResult: the parsed and validated result.

        Raises:
            InferenceResultLoadError: if required keys are missing, answers is not a list,
                or any answer item does not match the expected structure.
        """
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
            skipped=dict_data["skipped"],
        )


@dataclass
class BenchmarkResult:
    """
    Top-level result container for the Bongard benchmark.

    Stores:
    - model name and dataset path used,
    - end time of the run,
    - and a list of `StrategyResult` entries for each strategy executed.

    Example JSON-like structure:
    ```
          "model": "my-model",
          "dataset": "datasets/bongard",
          "end_time": "2026-04-16T12:34:56.789000",
          "results": [
            {
              "strategy": "direct",
              "prompts": ["..."],
              "answers": [{"problem" = "...",
                           "answer" = "..."}, ... ],
              "skipped": []
            },
            {
              "strategy": "descriptive_direct",
              "prompts": ["...", "..."],
              "answers": [{"problem" = "...",
                           "answer" = "..."}, ... ],
              "skipped": ["..."]
            }
          ]
    ```
    """

    model: str
    dataset: str
    end_time: str = field(init=False)
    results: List[StrategyResult]

    def __post_init__(self):
        """Set the end time when saving results"""
        self.end_time = datetime.now().isoformat()

    def to_json(self, indent=4):
        """
        Serialize this benchmark result to a JSON-formatted string.

        Args:
            indent (int): number of spaces to use for indentation (default 4).

        Returns:
            str: the JSON-formatted string representation of the result.
        """
        json_data = json.dumps(asdict(self), indent=indent, ensure_ascii=False)

        return json_data

    def save_as_json(self, file_path: str | Path | None = None, indent=4):
        """
        Save the benchmark result to a JSON file.

        If no file path is given, a default name is generated using
        the model name and end time.

        Args:
            file_path (Optional[str | Path]):
                Path to the JSON file. If None, a file name like
                ``results_my-model_2026-04-16T123456.789000.json`` is used.
            indent (int): indentation level for the JSON file.

        Returns:
            Path: the resolved path of the saved file.

        Example:
        ```
            result = benchmark.run(...)
            saved_path = result.save_as_json("results.json")
        ```
        """
        json_string = self.to_json(indent)

        if not file_path:
            file_path = "results_" + self.model + "_" + self.end_time + ".json"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json_string)
        return file_path

    @classmethod
    def from_dict(cls, dict_data: Dict):
        """
        Load a benchmark result from a dictionary (typically from JSON).

        The dictionary should have keys:
        - "model" (str): the model name,
        - "dataset" (str): the dataset path,
        - "end_time" (str): ISO-8601 timestamp,
        - "results" (List[Dict[str, Any]]): list of strategy results (each
          is a dictionary matching `StrategyResult` schema).

        Args:
            dict_data (Dict[str, Any]): the raw dictionary.

        Returns:
            BenchmarkResult: the parsed and validated result.

        Raises:
            InferenceResultLoadError: if fields are missing, results is not a list,
                or any inner strategy result fails to load.
        """
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
        """
        Load a benchmark result from a JSON file.

        Args:
            json_path (str): path to the JSON file.

        Returns:
            BenchmarkResult: the loaded result.

        Raises:
            FileNotFoundError: if the file does not exist.
            IsADirectoryError: if the path is a directory.
            InferenceResultLoadError: if the file is not valid JSON or the schema is incorrect.
        """
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

    def print_stats(self) -> None:
        """
        Print a short summary of the benchmark results to stdout.

        Includes:
        - model name,
        - dataset path,
        - number of strategies,
        - number of solved and skipped tasks per strategy.
        """
        print(
            f"Results of benchmark for model: '{self.model}' and dataset '{self.dataset}'"
        )
        print(
            f"Total amount of strategies run: {len(self.results)}. Per strategy information:"
        )
        for result in self.results:
            print(f"Strategy: {result.strategy}. Prompts: {result.prompts}.")
            print(
                f"Tasks solved: {len(result.answers)}, Tasks skipped: {len(result.skipped)}"
            )
