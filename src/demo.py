from benchmark import BenchmarkConfig, BenchmarkResult, BongBench


def ask_model(prompt, image):
    if image.exists():
        return f"answer for image `{image}` and prompt `{prompt}`"
    return "no image there :("


def reload_model():
    pass


if __name__ == "__main__":
    # inference model
    config_path = "../prompts/sample_config.json"

    config = BenchmarkConfig.load(config_path)
    benchmark = BongBench(config)
    results = benchmark.run(ask_model, reload_model)
    results.save_as_json("results.json")

    # load inference results
    results_from_json = BenchmarkResult.load_from_json_file("results.json")
    results_from_json.print_stats()
