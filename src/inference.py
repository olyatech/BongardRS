import os
import pathlib
import json

from benchmark import BenchmarkConfig, BenchmarkResult, BongBench


def ask_model(prompt, image):
    if image.exists():
        return f"answer for image {image} and prompt {prompt}"
    return "no image there :("


def reload_model():
    pass


if __name__ == "__main__":
    # inference model
    setup = BenchmarkConfig.load("sample_setup.json")
    benchmark = BongBench(setup)
    results = benchmark.run(ask_model, reload_model)
    results.save_as_json("results.json")

    # load inference results
    results_from_json = BenchmarkResult.load_from_json_file("results.json")
