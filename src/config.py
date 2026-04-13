import json

from dataclasses import dataclass, fields
from json import JSONDecodeError
from pathlib import Path
from typing import List, Dict, Any
from typeguard import check_type, TypeCheckError


class ConfigError(ValueError):
    """Raised when benchmark config is invalid."""

    pass


from strategies import StrategyName


@dataclass
class StrategySetup:
    """
    Configuration for one strategy of the benchmark.
    """

    strategy: StrategyName
    prompts: List[str]

    @classmethod
    def from_dict(cls, data: Dict):
        """Load setup from dictionary."""
        cls._validate_fields(data)
        return cls(**data)

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
            raise ConfigError(f"Invalid JSON in setup file {path}: {e}") from e

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
            raise ConfigError(f"Setup file missing required keys: {missing}")

        for f in fields(cls):
            value = data[f.name]
            try:
                check_type(value, f.type)
            except TypeCheckError as e:
                raise ConfigError(
                    f"Error during loading a setup file. Type mismatch in setup field '{f.name}': "
                    f"expected {f.type.__name__}, got {type(value).__name__}. Message from type checker: {str(e)}."
                ) from None

    @classmethod
    def _load_setups(cls, setups_list: Any) -> List[StrategySetup]:
        if not isinstance(setups_list, list):
            raise ConfigError("Invalid config: 'setups' must be a list")

        setups: List[StrategySetup] = []
        for i, raw in enumerate(setups_list):
            try:
                setups.append(StrategySetup(**raw))
            except Exception as e:
                raise ConfigError(
                    f"Error during config loading. Invalid strategy setup at index {i}. {str(e)}"
                ) from None

        return setups

    @classmethod
    def _load_dataset_path(cls, dataset_path) -> Path:
        """Convert and validate dataset path."""
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
