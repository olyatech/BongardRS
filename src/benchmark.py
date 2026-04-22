"""
BongBench – runner for the Bongard benchmark.

This module provides the `BongBench` class, which orchestrates running one or more
prompting strategies (see `StrategyName`) on a model using a configuration defined
by `BenchmarkConfig`. The result is collected as a `BenchmarkResult` that can be
saved to JSON.

The typical workflow is:
- Load a benchmark configuration: `config = BenchmarkConfig.load("config.json")`
- Instantiate the runner: `bench = BongBench(config)`
- Run the benchmark: `result = bench.run(ask_model, reload_context)`
- Inspect or save `result` as JSON.

See:
- `config.BenchmarkConfig` for the configuration schema.
- `strategies` for available strategy names and implementations.
- `results.BenchmarkResult` for the output format.
"""

from collections.abc import Callable
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import List, Dict, Optional, Iterable

from results import BenchmarkResult, StrategyResult
from config import BenchmarkConfig, StrategySetup
from strategies import StrategyName, STRATEGIES


class BongBench:
    """
    Runner for the Bongard benchmark.

    Orchestrates running one or more strategies on a model using a configuration file
    that defines strategies, prompts, and dataset path. See config structure in config.BenchmarkConfig info.

    Usage example:
    ```
        config = BenchmarkConfig.load("config.json")
        bench = BongBench(config)
        result = bench.run(ask_model, reload_context)
    ```
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark runner.

        Args:
            config (BenchmarkConfig):
                The benchmark configuration loaded from a JSON file.
                Contains dataset path, model name, and list of strategy setups.
        """
        self.__config__ = config
        self.setups_by_strategy: Dict[StrategyName, List[StrategySetup]] = {}
        for setup in config.strategies:
            self.setups_by_strategy.setdefault(setup.strategy, []).append(setup)

    def run(
        self,
        ask_model: Callable[[str, Path], str],
        reload_context: Callable[[], None],
        strategies: Optional[Iterable[StrategyName]] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Run the benchmark for one or more strategies.

        Each strategy is executed against all problems in the dataset defined in the
        configuration. Multiple prompt setups for the same strategy may be present
        in the config, all would be run.

        Args:
            ask_model (Callable[[str, Path], str]):
                A function that sends a text prompt and an image path to a model
                and returns the model’s answer string.
            reload_context (Callable[[], None]):
                A function that resets or reloads any context (e.g., clears a chat history)
                before each problem is processed.
            strategies (Optional[Iterable[StrategyName]]):
                The list of strategy names to run. If None (default), all strategies
                defined in the config would be run.
            checkpoint_dir (Optional[str]):
                Directory path to save intermediate results. If None, no checkpointing
                is performed.

        Returns:
            BenchmarkResult: results for all strategies, can be saved as json.

        Raises:
            ValueError: if a strategy name is not known (not in ``StrategyName``).
        """
        if strategies is None:
            strategies = list(self.setups_by_strategy.keys())

        results: List[StrategyResult] = []
        checkpoint_dir_path: Optional[Path] = None

        if checkpoint_dir:
            checkpoint_dir_path = Path(checkpoint_dir)
            checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

        for strategy in strategies:
            if strategy not in StrategyName:
                raise ValueError(
                    f"Error during benchmark run. Unknown strategy: {strategy} (was set up in config)."
                )

            strategy_func = STRATEGIES[strategy]

            for setup in self.setups_by_strategy[strategy]:
                result = strategy_func(
                    ask_model, reload_context, setup.prompts, self.__config__.dataset
                )
                results.append(result)

                if checkpoint_dir_path is not None:
                    partial_result = BenchmarkResult(
                        model=self.__config__.model,
                        dataset=str(self.__config__.dataset),
                        results=results,
                    )
                    checkpoint_name = f"checkpoint_{partial_result.model}_{partial_result.end_time}.json"
                    partial_result.save_as_json(checkpoint_dir_path / checkpoint_name)

        return BenchmarkResult(
            model=self.__config__.model,
            dataset=str(self.__config__.dataset),
            results=results,
        )
