import pandas as pd
from llm_judge import LLMJudge
import os
import argparse
import time
from datetime import datetime

def run_test(data_path, num_samples=5, model="gpt-4o", output_dir="test_results", random_seed=None, api_base_url=None, api_key=None):
    """Run a test evaluation on random samples from the provided dataset

    Args:
        data_path (str): Path to the CSV dataset
        num_samples (int): Number of random samples to test
        model (str): The LLM to use for judging (e.g., "gpt-4o", "qwen3-72b-chat")
        output_dir (str): Directory to save results
        random_seed (int, optional): Random seed for reproducibility
        api_base_url (str, optional): Base URL for the LLM API
        api_key (str, optional): API key for the LLM
    """
    if random_seed is None:
        random_seed = int(datetime.now().timestamp())
        print(f"Generated random seed: {random_seed}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded, contains {len(df)} samples")

    sampled_df = df.sample(min(num_samples, len(df)), random_state=random_seed)
    print(f"Randomly sampled {len(sampled_df)} examples for testing")

    model_columns = [col for col in df.columns if col.endswith('_generation')]
    model_names = [col.replace('_generation', '') for col in model_columns]
    print(f"Detected models to evaluate: {', '.join(model_names)}")

    if len(model_names) < 2:
        print("ERROR: Need at least 2 models to compare. Found only:", model_names)
        return

    judge = LLMJudge(model=model, random_seed=random_seed, api_base_url=api_base_url, api_key=api_key)
    print(f"Using judge model: {model}")
    if api_base_url:
        print(f"Using API base URL: {api_base_url}")


    print("\n-- Testing single example evaluation --")
    example = sampled_df.iloc[0]
    responses = {}
    for model_name, column in zip(model_names, model_columns):
        responses[model_name] = example[column]

    single_eval_result = judge.test_single_example(
        prompt=example["prompt"],
        ground_truth=example.get("ground_truth", "N/A"),
        responses=responses
    )

    print("\n-- Testing with multiple samples --")
    temp_csv_path = os.path.join(output_dir, "temp_sampled_data.csv")
    sampled_df.to_csv(temp_csv_path, index=False)

    try:
        start_time = time.time()
        results = judge.evaluate_dataset(
            data_path=temp_csv_path,
            output_dir=output_dir,
            limit=None
        )
        elapsed_time = time.time() - start_time

        print("\n***** TEST EVALUATION SUMMARY *****")
        print(f"Samples evaluated: {results['sample_size']}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        if results['sample_size'] > 0:
            print(f"Average time per sample: {elapsed_time/results['sample_size']:.2f} seconds")
        else:
            print("Average time per sample: N/A (no samples evaluated)")


        print("\nModel ranking as #1 counts:")
        for model_name, count in results["model_counts"].items():
            percentage = (count / results['sample_size']) * 100 if results['sample_size'] > 0 else 0
            print(f"  {model_name}: {count} ({percentage:.1f}%)")

        if "rank_counts" in results and results["rank_counts"]:
            print("\nRank distribution:")
            for model_name_key_eval in results["rank_counts"].keys():
                print(f"  {model_name_key_eval}:")
                for rank, count in results["rank_counts"][model_name_key_eval].items():
                    percentage = (count / results['sample_size']) * 100 if results['sample_size'] > 0 else 0
                    print(f"    Rank {rank}: {count} ({percentage:.1f}%)")
        else:
            print("\nRank distribution: No rank_counts data available.")


        if "average_ranks" in results and results["average_ranks"]:
            print("\nAverage ranks (lower is better):")
            for model_name_key_eval, avg_rank in results["average_ranks"].items():
                print(f"  {model_name_key_eval}: {avg_rank:.2f}")
        else:
            print("\nAverage ranks: No average_ranks data available.")


        print(f"Visualizations for test run saved to {output_dir}")


        response = input("\nDo you want to proceed with evaluating the full dataset? (y/n): ")
        if response.lower() == 'y':
            print("\n-- Evaluating full dataset --")
            full_results = judge.evaluate_dataset(
                data_path=data_path,
                output_dir=output_dir
            )
            print(f"\nFull evaluation complete. Results saved to: {output_dir}/judge_results_{full_results['run_id']}.json")
            print(f"Visualizations for full run saved to {output_dir}")

            if "average_scores" not in full_results:
                 print("Calculating average scores for the report (using rank-based scoring)...")
                 full_results["average_scores"] = calculate_average_scores_for_report(full_results)

            create_report(full_results, output_dir)

    finally:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
            print(f"Deleted temporary file: {temp_csv_path}")

def calculate_average_scores_for_report(results):
    """Calculate average scores for each model based on ranks for reporting."""
    model_names = list(results.get("model_counts", {}).keys())
    if not model_names:
        return {}

    num_models_evaluated = len(model_names)

    if results.get("evaluations"):
        first_eval_rankings = results["evaluations"][0].get("rankings", {})
        num_models_in_comparison = len(first_eval_rankings.keys()) if first_eval_rankings else num_models_evaluated
    else:
        num_models_in_comparison = num_models_evaluated

    rank_to_score = {1: 10.0}
    if num_models_in_comparison > 1:
        for i in range(2, num_models_in_comparison + 1):
            rank_to_score[i] = max(0, 10.0 - ((i-1) * (10.0 / (num_models_in_comparison -1 ))))
    elif num_models_in_comparison == 1:
        rank_to_score = {1:10.0}

    model_points = {model: 0.0 for model in model_names}
    model_appearances = {model: 0 for model in model_names}

    for eval_item in results.get("evaluations", []):
        rankings = eval_item.get("rankings", {})
        for model, rank in rankings.items():
            if model in model_points:
                model_points[model] += rank_to_score.get(rank, 0)
                model_appearances[model] += 1

    average_scores = {}
    for model in model_names:
        if model_appearances[model] > 0:
            average_scores[model] = model_points[model] / model_appearances[model]
        else:
            average_scores[model] = 0.0
    return average_scores


def create_report(results, output_dir):
    """Create a detailed analysis report."""
    report_path = os.path.join(output_dir, f"evaluation_report_{results.get('run_id', 'test')}.txt")

    with open(report_path, 'w') as f:
        f.write("LLM Judge Evaluation Report\n")
        f.write("==========================\n\n")

        f.write(f"Judge Model: {results.get('model', 'N/A')}\n")
        f.write(f"Dataset: {results.get('dataset', 'N/A')}\n")
        f.write(f"Number of Samples: {results.get('sample_size', 'N/A')}\n")
        f.write(f"Run Timestamp: {results.get('timestamp', 'N/A')}\n")
        f.write(f"Run ID: {results.get('run_id', 'N/A')}\n")
        f.write(f"Random Seed: {results.get('random_seed', 'N/A')}\n\n")

        f.write("Model Performance\n")
        f.write("-----------------\n")

        if results.get('model_counts'):
            f.write("\nBest Model Counts (Ranked #1):\n")
            for model, count in results["model_counts"].items():
                percentage = (count / results['sample_size']) * 100 if results['sample_size'] > 0 else 0
                f.write(f"  {model}: {count} ({percentage:.1f}%)\n")

        if results.get("average_scores"):
            f.write("\nAverage Scores (Rank-based, out of 10):\n")
            for model, score in results["average_scores"].items():
                f.write(f"  {model}: {score:.2f}\n")


        if results.get('rank_counts'):
            f.write("\nRank Distribution:\n")
            for model_name_key_report in results["rank_counts"]:
                f.write(f"  {model_name_key_report}:\n")
                for rank, count in results["rank_counts"][model_name_key_report].items():
                    percentage = (count / results['sample_size']) * 100 if results['sample_size'] > 0 else 0
                    f.write(f"    Rank {rank}: {count} ({percentage:.1f}%)\n")

        if results.get('average_ranks'):
            f.write("\nAverage Ranks (lower is better):\n")
            for model, avg_rank in results["average_ranks"].items():
                f.write(f"  {model}: {avg_rank:.2f}\n")

        if results.get('latency_stats'):
            f.write("\nLatency Statistics (Judge Model):\n")
            f.write(f"  Total: {results['latency_stats'].get('total', 0):.2f} seconds\n")
            f.write(f"  Average: {results['latency_stats'].get('average', 0):.2f} seconds per sample\n")
            f.write(f"  Min: {results['latency_stats'].get('min', 0):.2f} seconds\n")
            f.write(f"  Max: {results['latency_stats'].get('max', 0):.2f} seconds\n")

        f.write(f"\nFull results JSON: judge_results_{results.get('run_id', 'N/A')}.json\n")
    print(f"Detailed report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLM Judge with samples from your dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to your CSV dataset")
    parser.add_argument("--samples", type=int, default=5, help="Number of random samples to test with (default: 5)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM to use for judging (e.g., gpt-4o, qwen3-72b-chat, etc.)")
    parser.add_argument("--output", type=str, default="test_results", help="Output directory for results")
    parser.add_argument("--api_base_url", type=str, default=None, help="Optional: Base URL for the LLM API endpoint (e.g., for local models like Qwen on Ollama or vLLM)")
    parser.add_argument("--api_key", type=str, default=None, help="Optional: API key for the LLM API. If not provided, will try to use OPENAI_API_KEY or QWEN_API_KEY from environment variables.")
    parser.add_argument("--seed", type=int, default=None, help="Optional: Random seed for reproducibility")


    args = parser.parse_args()


    run_test(
        data_path=args.data,
        num_samples=args.samples,
        model=args.model,
        output_dir=args.output,
        random_seed=args.seed,
        api_base_url=args.api_base_url,
        api_key=args.api_key
    )