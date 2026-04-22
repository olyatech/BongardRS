"""Configuration module for the Bongard benchmark.

Defines the structure of the benchmark configuration file (JSON) and the classes
that parse, validate and store it for following usage in Bencmark run.
"""

import json

from dataclasses import dataclass, fields
from json import JSONDecodeError
from pathlib import Path
from typing import List, Dict, Any
from typeguard import check_type, TypeCheckError


class ConfigError(ValueError):
    """
    Raised when the benchmark configuration is invalid or inconsistent.

    Indicates errors in:
    - JSON structure,
    - missing or invalid fields,
    - unknown strategy name,
    - wrong prompt list length.
    """

    pass


from strategies import StrategyName, PROMPTS_PER_STRATEGY


@dataclass
class StrategySetup:
    """
    Configuration for one strategy of the Bongard benchmark.

    A strategy corresponds to a specific prompting method (e.g., ``direct``,
    ``descriptive-direct``, check readme for more information).
    This  class holds the strategy name and its list of prompts.

    Example config fragment:
    ```
        "strategy": "direct",
        "prompts": ["Describe this image."]
    ```
    """

    strategy: StrategyName
    prompts: List[str]

    @classmethod
    def from_dict(cls, data: Dict):
        """
        Load a strategy setup from a dictionary (e.g., parsed JSON).

        The strategy string is converted into a ``StrategyName`` enum member
        and validated; the prompts list is validated for prompts amount.
        Check available strategies at strategies.StrategyName, required len of list
        with prompts at strategies.PROMPTS_PER_STRATEGY.

        Args:
            data (Dict[str, Any]):
                A dictionary with keys:
                - "strategy" (str): name of the prompting strategy (must be
                one of the values in ``StrategyName``).
                - "prompts" (List[str]): list of prompts used for that strategy.

        Returns:
            StrategySetup: the validated strategy configuration.

        Raises:
            ConfigError: if the dictionary is missing required keys, contains
                an unknown strategy, or the prompts list has incorrect length.
        """
        data["strategy"] = cls._get_strategy_name(data)
        cls._validate_fields(data)
        instance = cls(**data)
        instance._check_prompts_len()
        return instance

    @classmethod
    def _get_strategy_name(cls, data: dict[str, Any]) -> StrategyName:
        if "strategy" not in data.keys():
            raise ConfigError("Setup missing required keys: 'strategy'")

        strategy = data["strategy"]
        try:
            strategy = StrategyName(strategy)
        except ValueError as e:
            raise ConfigError(f"Got invalid strategy '{strategy}'. {e}") from None

        return strategy

    @classmethod
    def _validate_fields(cls, data: dict[str, Any]) -> None:
        required_fields = {f.name for f in fields(cls)}
        missing = required_fields - set(data.keys())
        if missing:
            raise ConfigError(f"Setup missing required keys: {missing}")

        for f in fields(cls):
            value = data[f.name]
            try:
                check_type(value, f.type)
            except TypeCheckError as e:
                raise ConfigError(
                    f"Type mismatch in setup field '{f.name}': "
                    f"expected {f.type}, got {type(value).__name__}. Message from type checker: {str(e)}."
                ) from None

    def _check_prompts_len(self):
        required_len = PROMPTS_PER_STRATEGY[self.strategy]
        current_len = len(self.prompts)
        if current_len != required_len:
            raise ConfigError(
                f"Invalid prompts list for strategy {self.strategy}: {required_len} prompts required, got {current_len}."
            )


@dataclass
class BenchmarkConfig:
    """
    Top-level configuration for the Bongard benchmark.

    Contains:
    - global information (model name and dataset path),
    - and a list of per-strategy setups (``StrategySetup``).

    Example JSON config:
    ```
    "model": "model name",
    "dataset": "datasets/bongard",
    "strategies": [
      {"strategy": "direct", "prompts": ["sample prompt"]},
      {"strategy": "descriptive-direct", "prompts": ["sample prompt 1",
                                                     "sample prompt 2. left class: {}, right class: {}"]}
    ]
    ```
    """

    strategies: List[StrategySetup]
    model: str
    dataset: Path

    @classmethod
    def load(cls, setup_path: str):
        """
        Load the benchmark configuration from a JSON file.

        The file is parsed, validated, and converted into a ``BenchmarkConfig`` object
        that can be passed to benchmark runner ``BongBench``.

        Args:
            setup_path (str):
                Path to the JSON configuration file.

        Returns:
            BenchmarkConfig: the fully parsed and validated configuration.

        Raises:
            FileNotFoundError: if the file does not exist.
            JSONDecodeError: if the file is not valid JSON.
            ConfigError: if any part of the configuration is invalid or inconsistent.
        """
        path = Path(setup_path)
        if not path.is_file():
            raise FileNotFoundError(f"Setup file not found as {path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in setup file {path}: {e}") from None

        data["strategies"] = cls._load_setups(data["strategies"])
        data["dataset"] = cls._load_dataset_path(data["dataset"])

        cls._validate_fields(data)

        field_names = {f.name for f in fields(BenchmarkConfig)}
        init_kwargs = {name: data[name] for name in field_names}

        return cls(**init_kwargs)

    @classmethod
    def _validate_fields(cls, data: dict[str, Any]) -> None:
        """
        Validate that the top-level config dictionary has correct fields and types.

        Args:
            data (dict[str, Any]): the raw config dictionary.

        Raises:
            ConfigError: if any required field is missing or has a type mismatch.
        """
        required_fields = {f.name for f in fields(cls)}
        missing = required_fields - set(data.keys())
        if missing:
            raise ConfigError(f"Setup file missing required keys: {missing}")

        for f in fields(cls):
            value = data[f.name]
            try:
                check_type(value, f.type)
            except TypeCheckError as e:
                raise ConfigError(
                    f"Error during loading a setup file. Type mismatch in setup field '{f.name}': "
                    f"expected {f.type}, got {type(value).__name__}. Message from type checker: {str(e)}."
                ) from None

    @classmethod
    def _load_setups(cls, setups_list: Any) -> List[StrategySetup]:
        """
        Convert a list of raw strategy setups into validated ``StrategySetup`` objects.

        Args:
            setups_list (Any):
                A list of dictionaries as read from the JSON.

        Returns:
            List[StrategySetup]: the validated list of strategy setups.

        Raises:
            ConfigError: if ``setups_list`` is not a list, or if any setup is invalid.
        """
        if not isinstance(setups_list, list):
            raise ConfigError("Invalid config: 'setups' must be a list")

        setups: List[StrategySetup] = []
        for i, raw in enumerate(setups_list):
            try:
                setups.append(StrategySetup.from_dict(raw))
            except Exception as e:
                raise ConfigError(
                    f"Error during config loading. Invalid strategy setup at index {i}. {str(e)}"
                ) from None

        return setups

    @classmethod
    def _load_dataset_path(cls, dataset_path) -> Path:
        """
        Convert and validate the dataset path given in the config.

        Args:
            dataset_path (Any):
                The value of the ``dataset`` field from the JSON.

        Returns:
            Path: a ``pathlib.Path`` object pointing to the dataset directory.

        Raises:
            ConfigError: if the dataset path is absent, not a string, or does not
                exist as a directory.
        """
        if dataset_path is None:
            raise ConfigError(
                "Error during config loading. Path to dataset is required in field 'dataset'."
            )
        if not isinstance(dataset_path, str):
            raise ConfigError(
                "Error during config loading. Field 'dataset' must contain a string path"
            )

        path = Path(dataset_path)
        if not path.exists():
            raise ConfigError(
                f"Error during config loading. Dataset directory specified in config not found: {path}"
            )
        if not path.is_dir():
            raise ConfigError(
                f"Error during config loading. Dataset path specified in config must be a directory: {path}"
            )
        return path
