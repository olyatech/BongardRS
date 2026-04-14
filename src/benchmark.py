from collections.abc import Callable
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import List, Dict, Optional, Iterable

from results import BenchmarkResult, StrategyResult
from config import BenchmarkConfig, StrategySetup
from strategies import StrategyName, STRATEGIES


class BongBench:

    def __init__(self, config: BenchmarkConfig):
        self.__config__ = config
        self.setups_by_strategy: Dict[StrategyName, List[StrategySetup]] = {}
        for setup in config.strategies:
            self.setups_by_strategy.setdefault(setup.strategy, []).append(setup)

    def run(
        self,
        ask_model: Callable[[str, Path], str],
        reload_context: Callable[[], None],
        strategies: Optional[Iterable[StrategyName]] = None,
    ) -> BenchmarkResult:
        if strategies is None:
            strategies = list(self.setups_by_strategy.keys())

        results: List[StrategyResult] = []
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

        return BenchmarkResult(
            model=self.__config__.model,
            dataset=str(self.__config__.dataset),
            results=results,
        )
