from typing import List
from pathlib import Path

from benchmark import BenchmarkConfig, BenchmarkResult, BongBench
from strategies import StrategyName


def ask_model(prompt: str, image: Path):
    if image.exists():
        return f"answer for image `{image}` and prompt `{prompt}`"
    return "no image there :("


def ask_model_multiimage(prompt: str, images: List[Path]) -> str:
    return "answer from multi-image model"


def reload_model():
    pass


if __name__ == "__main__":
    # inference model
    config_path = "../prompts/sample_config.json"

    config = BenchmarkConfig.load(config_path)
    benchmark = BongBench(config)

    # run for single-image model
    results = benchmark.run(
        ask_model,
        reload_model,
        checkpoint_dir="bench_checkpoints",
        strategies=[StrategyName.CONTRASTIVE_ITERATIVE, StrategyName.CONTRASTIVE_DIRECT],
    )
    results.save_as_json("results.json")

    # run for multi-image model
    results = benchmark.run_multiimage(
        ask_model_multiimage,
        reload_model,
        checkpoint_dir="bench_checkpoints",
        strategies=[StrategyName.CONTRASTIVE_DIRECT_MULTIIMAGE, StrategyName.CONTRASTIVE_ITERATIVE_MULTIIMAGE],
    )
    results.save_as_json("results.json")

    # load inference results
    results_from_json = BenchmarkResult.load_from_json_file("results.json")
    results_from_json.print_stats()
