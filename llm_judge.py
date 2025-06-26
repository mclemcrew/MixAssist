import os
import pandas as pd
import json
import random
import uuid
import time
import re
from typing import Dict, Any
from openai import OpenAI, APITimeoutError
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from datetime import datetime

class LLMJudge:
    """A system that uses an LLM to rank model outputs based on specified criteria."""

    def __init__(self, model="gpt-4o", random_seed=None, api_base_url=None, api_key=None, 
                 timeout: int = 60, max_retries_on_timeout: int = 0, enable_retry_on_timeout: bool = False):
        self.model = model
        self.timeout = timeout
        self.max_retries_on_timeout = max_retries_on_timeout
        self.enable_retry_on_timeout = enable_retry_on_timeout

        if random_seed is None:
            random_seed = int(datetime.now().timestamp())
        random.seed(random_seed)
        self.random_seed = random_seed
        print(f"Using random seed: {random_seed}")

        effective_api_key = api_key
        effective_base_url = api_base_url

        is_likely_local_model_pattern = ":" in self.model and not \
                                     (self.model.lower().startswith("gpt-") or \
                                      self.model.lower().startswith("text-") or \
                                      self.model.lower().startswith("ada") or \
                                      self.model.lower().startswith("babbage") or \
                                      self.model.lower().startswith("curie") or \
                                      self.model.lower().startswith("davinci") or \
                                      self.model.lower().startswith("o3-"))

        if is_likely_local_model_pattern:
            print(f"Model name '{self.model}' suggests a local model (e.g., Ollama).")
            if not effective_base_url:
                effective_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
                print(f"No API base URL provided, defaulting to Ollama standard: {effective_base_url}")
            if not effective_api_key:
                effective_api_key = os.getenv("OLLAMA_API_KEY", "ollama") 
                print(f"No API key provided for local model, defaulting to: '{effective_api_key}'")
        
        elif effective_base_url:
            print(f"Custom API base URL provided: {effective_base_url}")
            if not effective_api_key:
                effective_api_key = "ollama"
                print(f"No API key provided with custom base URL, defaulting to: '{effective_api_key}'")
        
        else:
            print("Assuming OpenAI model configuration.")
            if not effective_api_key:
                effective_api_key = os.getenv("OPENAI_API_KEY")
            if not effective_api_key:
                print("Warning: OPENAI_API_KEY not found in environment and no API key provided. This may fail if the model requires OpenAI authentication.")
            print("Using default OpenAI API base URL (or expecting OPENAI_BASE_URL from environment).")

        try:
            self.client = OpenAI(api_key=effective_api_key, base_url=effective_base_url)
            print(f"LLM Judge Client initialized. Target model: {self.model}")
            if effective_base_url:
                 print(f"Client configured with API Base URL: {effective_base_url}")

        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            print("Please check your API key, base URL, and network connection.")
            raise

        self.results = {} 
        self.run_id = str(uuid.uuid4())
        self.skipped_samples = []
    
    def _get_ranking_prompt(self) -> str:
        return """
You are an impartial judge evaluating how closely each response aligns with the way an expert producer would assist an amateur during a co-creative music mixing session.

Your PRIMARY goal is to rank three given responses based on the following evaluation criteria:

CRITICAL EVALUATION CRITERIA (in order of importance):
1. TECHNICAL ACCURACY - Information should be accurate within a music production context
2. HELPFULNESS - Addresses the issue at hand with practical, actionable advice or suitable explanation given the user’s behavior
3. CONVERSATION FLUENCY - Response is natural, concise, and decisive

EXAMPLES:

This is an example of an explanation with GOOD technical accuracy: "By pulling down the levels in the 400 to 600 Hz range and adding some air around 1.3k to 2k, you're creating more room for the guitar to breathe and stand out in the mix."

Using the standard ranges of frequencies for different instruments while suggesting an EQ is helpful to amateurs in understanding both the technical implementation as well as the reasoning behind why these techniques are useful.  This explanation provides the correct ranges technically as well as providing a reasoning for what adding these elements will do for the mix.

This is an example of an explanation with BAD technical accuracy: "I don't understand what you mean by 'pan the crash and ride'."

Panning is a technique to help increase the stereo image of the mix by moving different instruments around the placement of the stereo field. This explanation does not understand this concept of panning, which is a foundation in mixing.  For this reason, this explanation is bad in terms of technical accuracy.

This is an example of an explanation with GOOD helpfulness: "To adjust compression, first, locate the compressor on your DAW (Digital Audio Workstation). It usually has a ‘Compressor’ or ‘Clipping’ icon. You can drag the ‘Wet’ control to the left or right to make the compression more or less pronounced. The wet control determines how much of the signal is being compressed."

This explanation describes to the amateur music producer exactly what to do without questioning their creative decisions and provides a greater depth of detail for what a compressor does such that the amateur can then use this knowledge within their exploration of the sonic space.

This is another example of GOOD helpfulness: "I need more information before I can respond. Please elaborate."

In some cases, this can be viewed as helpful such that the expert is asking clarifying questions as opposed to assuming what the amateur meant and providing a reason for something that may not be correct.

This is an example of an explanation with BAD helpfulness: "You can't just say, ’I want a really low reverb,’ because you can't just turn it off."

This is another example of BAD helpfulness: "It sounds like you are trying to make the drum loop tighter and more focused, while also reducing its energy or intensity. This could be achieved by adjusting the tempo, adding effects such as compression or EQ, or using a different drum sample with a different sound."

While this explanation may appear helpful on the surface, as opposed to asking a follow-up question before providing an answer, this explanation assumes what the user means instead of providing a clarifying question to resolve the discrepancy before continuing.

This kind of explanation belittles the user and undermines their learning potential throughout this process of mixing.  This explanation does not help the user understand what may be wrong with their explanation of the reverb and instead does not encourage learning.

This is an example of an explanation with GOOD naturalness: "You're gonna need a little bit of compression on that."

Slang terms such as gonna, gotta, sick, fire, or many other terms help promote a sense of naturalness within the conversation overall.  These terms and other music production jargon may provide a sense of ease and naturalness for the amateur who is receiving instructions. This wording helps provide a sense of naturalness throughout the conversation flow.

This is an example of an explanation with BAD naturalness: "I understand, thank you for letting me know. Is there anything else I can help with? If not, have a great day and let me know if you need any assistance in the future."

This explanation halts the communication between the expert and the amateur entirely and may also deter the amateur from asking follow-up questions due to the cold way in which the explanation is stated.

This is an example of an explanation with GOOD consciousness: "Yes, a compressor could help in controlling the dynamic range and making sure the volume levels are even across different parts of the song."

This explanation provides accurate information without being too verbose and encourages follow-up questions from the amateur.  Specifics such as a compressor and what it does are addressed well without being overly wordy.

This is an example of an explanation with BAD consciousness: "Yes, I think they could be a bit quieter to better balance with the guitar and vocals in the mix. It might also help to adjust the levels of each instrument to create more clarity and separation between them. Additionally, experimenting with different drum samples or effects can add depth and complexity to the overall sound."

This explanation wanders around the main argument and keeps adding sentences after initially providing a suggestion at the beginning.  The addition of these sentences might distract or confuse the amateur with too many additions to the central concept.

This is an example of an explanation with GOOD decisiveness: "Yeah, so we would put the high pass on guitar three then."

This explanation is succinct and provides a clear and actionable path forward for the amateur when implementing this suggestion.

This is an example of an explanation with BAD decisiveness: "That's a good question. I think it's more of a matter of what you can do with the tube screamer."

This explanation does not provide the amateur with a sense of certainty for why something was selected and instead provides a generic response overall and turns the onus back toward the amateur for what they can do with the effect.

You MUST rank these explanations, even if the difference is slight.

Output ONLY in JSON format: ‘{’rank_1’: {A, B, or C}, ‘rank_2’: {A, B, or C}; ’rank_3’: {A, B, or C}}; with no explanation.

==================
"""

    def _get_ranked_responses(self, prompt: str, ground_truth: str,
                           responses: Dict[str, str]) -> Dict[str, Any]:
        model_names = list(responses.keys())
        if len(model_names) < 2:
            return {"rankings": {model_names[0]: 1} if model_names else {}, "errors": ["Not enough models to rank"], "skipped": False}

        model_letter_map = {}
        letter_model_map = {}
        shuffled_models = model_names.copy()
        random.shuffle(shuffled_models)
        for i, model_name_key in enumerate(shuffled_models):
            letter = chr(65 + i)
            model_letter_map[model_name_key] = letter
            letter_model_map[letter] = model_name_key

        content = f"PROMPT: {prompt}\n\n"
        for model_name_key, letter in model_letter_map.items():
            content += f"RESPONSE {letter}: {responses[model_name_key]}\n\n"

        base_params = { # Renamed to base_params for clarity before temp adjustment
            "model": self.model,
            "messages": [{"role": "system", "content": self._get_ranking_prompt()}, {"role": "user", "content": content}],
            "timeout": self.timeout
        }
        
        current_sample_id_log = "N/A (single run)"
        if hasattr(self, 'results') and self.results and "evaluations" in self.results:
            if not self.results["evaluations"]:
                 current_sample_id_log = 0
            elif self.results["evaluations"][-1].get("placeholder"):
                 current_sample_id_log = self.results["evaluations"][-1].get("sample_id", -1) +1 if self.results["evaluations"][-1].get("sample_id", -1) != -1 else 0
            elif "sample_id" in self.results["evaluations"][-1]:
                 current_sample_id_log = self.results["evaluations"][-1].get("sample_id")
                 if isinstance(current_sample_id_log, (int, float)):
                      current_sample_id_log = int(current_sample_id_log) + 1
            if "test_single" == current_sample_id_log and "original_df_index" in self.results["evaluations"][-1]:
                current_sample_id_log = self.results["evaluations"][-1]["original_df_index"]


        if self.enable_retry_on_timeout:
            last_error = None
            for attempt in range(self.max_retries_on_timeout + 1):
                params = base_params.copy()
                if not (self.model.lower().startswith("gpt-") or self.model.lower().startswith("o3-")):
                    params["temperature"] = 0.0
                
                print(f"\n***** Sending Prompt to Judge (Sample ID: {current_sample_id_log}, Attempt {attempt + 1}/{self.max_retries_on_timeout + 1}) *****")
                if "temperature" in params:
                    print(f"Setting temperature to {params['temperature']} for model: {self.model}")
                try:
                    chat_completion = self.client.chat.completions.create(**params)
                    response_content = chat_completion.choices[0].message.content
                    json_str = re.search(r'\{.*\}', response_content.strip(), re.DOTALL)
                    if not json_str: raise ValueError("No JSON found in response")
                    json_str = json_str.group(0).replace("'", '"').replace(";", ",")
                    ranking_results = json.loads(json_str)
                    model_rankings = {letter_model_map[letter]: int(rank.split('_')[1]) for rank, letter in ranking_results.items()}
                    best_model_name = next((m for m, r in model_rankings.items() if r == 1), None)
                    return {
                        "rankings": model_rankings, "best_model": best_model_name,
                        "model_letter_map": model_letter_map, "letter_model_map": letter_model_map,
                        "ranking_response": response_content, "skipped": False
                    }
                except APITimeoutError as e:
                    last_error = e
                    print(f"Attempt {attempt + 1} timed out. (Sample ID: {current_sample_id_log})")
                    if attempt < self.max_retries_on_timeout:
                        print(f"Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        print(f"All {self.max_retries_on_timeout + 1} attempts timed out. Marking as skipped. (Sample ID: {current_sample_id_log})")
                        return {"skipped": True, "error": f"Request timed out after {self.max_retries_on_timeout + 1} attempts. Last error: {str(last_error)}"}
                except Exception as e:
                    print(f"Error on attempt {attempt + 1} (Sample ID: {current_sample_id_log}): {str(e)}. Falling back to random ranking.")
                    rankings = {model: pos for i, (model, pos) in enumerate(zip(model_names, random.sample(range(1, len(model_names) + 1), len(model_names))))}
                    return {"rankings": rankings, "best_model": next((m for m, r in rankings.items() if r == 1), None),
                            "model_letter_map": model_letter_map, "letter_model_map": {v:k for k,v in model_letter_map.items()},
                            "errors": [str(e)], "skipped": False}
            return {"skipped": True, "error": f"Retry logic completed without success. Last error: {str(last_error)}"}
        else:
            params = base_params.copy()
            if not (self.model.lower().startswith("gpt-") or self.model.lower().startswith("o3-")):
                params["temperature"] = 0.0

            print(f"\n***** Sending Prompt to Judge (Sample ID: {current_sample_id_log}, Single Attempt) *****")
            if "temperature" in params:
                 print(f"Setting temperature to {params['temperature']} for model: {self.model}")
            try:
                chat_completion = self.client.chat.completions.create(**params)
                response_content = chat_completion.choices[0].message.content
                json_str = re.search(r'\{.*\}', response_content.strip(), re.DOTALL)
                if not json_str: raise ValueError("No JSON found in response")
                json_str = json_str.group(0).replace("'", '"').replace(";", ",")
                ranking_results = json.loads(json_str)
                model_rankings = {letter_model_map[letter]: int(rank.split('_')[1]) for rank, letter in ranking_results.items()}
                best_model_name = next((m for m, r in model_rankings.items() if r == 1), None)
                return {
                    "rankings": model_rankings, "best_model": best_model_name,
                    "model_letter_map": model_letter_map, "letter_model_map": letter_model_map,
                    "ranking_response": response_content, "skipped": False
                }
            except APITimeoutError:
                print(f"Request timed out after {self.timeout} seconds. Skipping sample. (Sample ID: {current_sample_id_log})")
                return {"skipped": True, "error": f"Request timed out after {self.timeout} seconds."}
            except Exception as e:
                print(f"Error calling LLM API or parsing response (Sample ID: {current_sample_id_log}): {str(e)}. Falling back to random ranking.")
                rankings = {model: pos for i, (model, pos) in enumerate(zip(model_names, random.sample(range(1, len(model_names) + 1), len(model_names))))}
                return {
                    "rankings": rankings, "best_model": next((m for m, r in rankings.items() if r == 1), None),
                    "model_letter_map": model_letter_map, "letter_model_map": {v: k for k, v in model_letter_map.items()},
                    "errors": [str(e)], "skipped": False
                }

    def evaluate_sample(self, prompt: str, ground_truth: str,
                        responses: Dict[str, str]) -> Dict[str, Any]:
        start_time = time.time()
        evaluation_results = self._get_ranked_responses(prompt, ground_truth, responses)
        if not evaluation_results.get("skipped") or (evaluation_results.get("skipped") is False and "rankings" in evaluation_results) :
             evaluation_results["latency"] = time.time() - start_time
        return evaluation_results

    def evaluate_dataset(self, data_path: str, output_dir: str = "results",
                         limit: int = None, random_sample: bool = False) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(data_path)
        initial_sample_count = len(df)
        print(f"Loaded dataset with {initial_sample_count} samples")

        if limit and limit < len(df):
            current_df = df.sample(limit, random_state=self.random_seed) if random_sample else df.head(limit)
            print(f"Processing {len(current_df)} samples (limited/sampled).")
        else:
            current_df = df
            print(f"Processing all {len(current_df)} samples.")
        
        self.run_id = str(uuid.uuid4()) # Ensure a fresh run_id for each dataset evaluation
        self.results = {
            "run_id": self.run_id, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model, "dataset": data_path,
            "random_seed": self.random_seed, "evaluations": [],
            "model_counts": {}, "latency_stats": {"total": 0, "average": 0, "min": float('inf'), "max": 0},
            "initial_sample_count": initial_sample_count,
            "processed_sample_count_before_skips": len(current_df)
        }
        self.skipped_samples = []

        model_columns = [col for col in current_df.columns if col.endswith('_generation')]
        model_names = [col.replace('_generation', '') for col in model_columns]
        if not model_names:
            print("Error: No model generation columns (ending with '_generation') found.")
            return self.results # Return early if no models to evaluate
        print(f"Found models to evaluate: {', '.join(model_names)}")

        self.results["model_counts"] = {model_name: 0 for model_name in model_names}
        rank_counts = {model_name: {rank: 0 for rank in range(1, len(model_names) + 1)} for model_name in model_names}
        
        total_latency = 0.0

        self.results["evaluations"] = [] 


        for idx_val, row_content in tqdm(current_df.iterrows(), total=len(current_df), desc="Evaluating samples"):
            prompt_text = row_content.get("prompt", "")
            ground_truth_text = row_content.get("ground_truth", "")
            current_responses = {model_n: row_content.get(col_n, "") for model_n, col_n in zip(model_names, model_columns)}

            placeholder_idx = len(self.results["evaluations"])
            self.results["evaluations"].append({"sample_id": placeholder_idx -1 if placeholder_idx > 0 else -1 , "placeholder": True, "original_df_index": idx_val})


            evaluation_dict = self.evaluate_sample(prompt_text, ground_truth_text, current_responses)

            if self.results["evaluations"] and self.results["evaluations"][-1].get("placeholder"):
                self.results["evaluations"].pop()


            if evaluation_dict.get("skipped"):
                self.skipped_samples.append({
                    "original_df_index": idx_val, 
                    "conversation_id": row_content.get("conversation_id", "unknown"), 
                    "turn_id": row_content.get("turn_id", "unknown"), 
                    "prompt": prompt_text, 
                    "reason": evaluation_dict.get("error")
                })
                continue

            evaluation_dict.update({
                "sample_id": len(self.results["evaluations"]),
                "original_df_index": idx_val,
                "conversation_id": row_content.get("conversation_id", "unknown"),
                "turn_id": row_content.get("turn_id", "unknown"), 
                "topic": row_content.get("topic", "unknown"),
                "prompt": prompt_text, "ground_truth": ground_truth_text, "model_responses": current_responses
            })
            self.results["evaluations"].append(evaluation_dict)

            if (best_m := evaluation_dict.get("best_model")) and best_m in self.results["model_counts"]:
                self.results["model_counts"][best_m] += 1
            
            for m_name, r_val in evaluation_dict.get("rankings", {}).items():
                if m_name in rank_counts and r_val in rank_counts[m_name]:
                    rank_counts[m_name][r_val] += 1

            current_latency = evaluation_dict.get("latency", 0.0)
            if current_latency > 0:
                total_latency += current_latency
                self.results["latency_stats"]["min"] = min(self.results["latency_stats"]["min"], current_latency)
                self.results["latency_stats"]["max"] = max(self.results["latency_stats"]["max"], current_latency)

            if (len(self.results["evaluations"]) % 10 == 0) and len(self.results["evaluations"]) > 0:
                 self._save_results(output_dir)

        num_successfully_evaluated = len(self.results["evaluations"])
        self.results.update({
            "rank_counts": rank_counts, 
            "final_evaluated_count": num_successfully_evaluated,
            "skipped_count": len(self.skipped_samples), 
            "skipped_samples_details": self.skipped_samples
        })
        
        if num_successfully_evaluated > 0:
            self.results["latency_stats"]["total"] = total_latency
            self.results["latency_stats"]["average"] = total_latency / num_successfully_evaluated
            self.results["latency_stats"]["min"] = 0.0 if self.results["latency_stats"]["min"] == float('inf') else self.results["latency_stats"]["min"]
            avg_ranks_calc = {}
            for model_n_key in model_names:
                ranks_list = [eval_item["rankings"].get(model_n_key) for eval_item in self.results["evaluations"] \
                              if eval_item.get("rankings") and eval_item["rankings"].get(model_n_key) is not None]
                avg_ranks_calc[model_n_key] = sum(ranks_list) / len(ranks_list) if ranks_list else 0
            self.results["average_ranks"] = avg_ranks_calc
        else:
            self.results["latency_stats"] = {"total": 0, "average": 0, "min": 0, "max": 0}
            self.results["average_ranks"] = {model_n_key: 0 for model_n_key in model_names}

        # This happens occasionally depending on the model selection
        if self.skipped_samples:
            print("\n***** SKIPPED SAMPLES *****")
            print(f"A total of {len(self.skipped_samples)} samples were skipped.")
            for skipped_item in self.skipped_samples:
                print(f"  - Original DF Index: {skipped_item['original_df_index']} (Conv ID: {skipped_item.get('conversation_id', 'N/A')}) - Reason: {skipped_item['reason']}")
            print("-----------------------\n")

        self._save_results(output_dir)
        if num_successfully_evaluated > 0:
            self._generate_visualizations(output_dir)
        else:
            print("No samples were successfully evaluated after processing, skipping visualizations.")
        return self.results

    def _save_results(self, output_dir: str) -> None:
        filepath = os.path.join(output_dir, f"judge_results_{self.run_id}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"Results saved to {filepath}")
        except Exception as e:
            print(f"Error saving results to {filepath}: {e}")

    def _generate_visualizations(self, output_dir: str) -> None:
        if not self.results.get("evaluations") and not self.results.get("final_evaluated_count", 0) > 0:
            print("No evaluations to visualize.")
            return

        model_names = list(self.results.get("model_counts", {}).keys())
        if not model_names:
            if self.results.get("evaluations") and "rankings" in self.results["evaluations"][0]:
                model_names = list(self.results["evaluations"][0]["rankings"].keys())
            if not model_names:
                 print("No models found in results for visualization.")
                 return

        model_counts_viz = {model_name: self.results.get("model_counts", {}).get(model_name, 0) for model_name in model_names}

        plt.figure(figsize=(10, 6))
        counts = [model_counts_viz.get(model, 0) for model in model_names]
        plt.bar(model_names, counts)
        plt.title("Number of Times Each Model Was Ranked #1")
        plt.xlabel("Model")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"best_model_counts_{self.run_id}.png"))
        plt.close()

        if self.results.get("average_ranks"):
            plt.figure(figsize=(10, 6))
            avg_ranks = [self.results["average_ranks"].get(model, 0) for model in model_names]
            plt.bar(model_names, avg_ranks)
            plt.title("Average Rank for Each Model (lower is better)")
            plt.xlabel("Model")
            plt.ylabel("Average Rank")
            plt.ylim(0, len(model_names) + 0.5 if model_names else 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"average_ranks_{self.run_id}.png"))
            plt.close()

        rank_counts_viz = self.results.get("rank_counts", {})
        if rank_counts_viz and model_names and any(m in rank_counts_viz for m in model_names) and rank_counts_viz.get(next(m for m in model_names if m in rank_counts_viz), {}):
            plt.figure(figsize=(12, 8))
            all_ranks_present = set()
            for model_name_key in model_names:
                if model_name_key in rank_counts_viz:
                    all_ranks_present.update(rank_counts_viz[model_name_key].keys())
            
            if not all_ranks_present:
                print("No rank data to visualize for rank distribution (stacked).")
                plt.close()
            else:
                ranks_to_plot = sorted([r for r in all_ranks_present if isinstance(r, (int, str))])
                bottom_data = np.zeros(len(model_names))
                for rank_val_str in ranks_to_plot:
                    rank_val = int(rank_val_str) if isinstance(rank_val_str, str) and rank_val_str.isdigit() else rank_val_str
                    rank_data = [rank_counts_viz.get(model, {}).get(str(rank_val), 0) for model in model_names]
                    plt.bar(model_names, rank_data, width=0.65, bottom=bottom_data, label=f'Rank {rank_val}')
                    bottom_data = [sum(x) for x in zip(bottom_data, rank_data)]
                
                plt.title("Rank Distribution by Model (Stacked)")
                plt.xlabel("Model")
                plt.ylabel("Count")
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"rank_distribution_stacked_{self.run_id}.png"))
            plt.close()

            rank_data_for_boxplot_final = []
            boxplot_labels_final = []
            if self.results.get("evaluations"):
                for model_name_key_bp in model_names:
                    model_specific_ranks = [eval_item_bp["rankings"][model_name_key_bp] for eval_item_bp in self.results["evaluations"] \
                                            if eval_item_bp.get("rankings") and model_name_key_bp in eval_item_bp["rankings"]]
                    if model_specific_ranks:
                        rank_data_for_boxplot_final.append(model_specific_ranks)
                        boxplot_labels_final.append(model_name_key_bp)

            if rank_data_for_boxplot_final: 
                plt.figure(figsize=(12, 6))
                plt.boxplot(rank_data_for_boxplot_final, labels=boxplot_labels_final)
                plt.title("Rank Distribution by Model (Boxplot)")
                plt.xlabel("Model")
                plt.ylabel("Rank (lower is better)")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"rank_distribution_boxplot_{self.run_id}.png"))
            else:
                print("No data for rank distribution boxplot after filtering.")
            plt.close()
        else:
            print("Skipping rank distribution plots due to missing rank_counts or model data.")
        
        print(f"Visualizations saved to {output_dir}")

    def test_single_example(self, prompt: str, ground_truth: str,
                           responses: Dict[str, str]) -> Dict[str, Any]:
        print("\n=== TESTING JUDGE WITH SINGLE EXAMPLE ===\n")
        print(f"PROMPT: {prompt}\n\nGROUND TRUTH: {ground_truth}")
        for model, response in responses.items():
            print(f"\n{model.upper()} RESPONSE: {response}")

        print("\n***** JUDGE EVALUATION *****")
        original_evaluations_temp = self.results.get("evaluations")
        if not hasattr(self, 'results') or not self.results: self.results = {}
        self.results["evaluations"] = [{"sample_id": "test_single", "placeholder": False, "original_df_index": "test_single"}]


        result = self.evaluate_sample(prompt, ground_truth, responses)
        
        if original_evaluations_temp is not None:
            self.results["evaluations"] = original_evaluations_temp
        elif "evaluations" in self.results:
            del self.results["evaluations"]


        if result.get("skipped"):
            print(f"\nSAMPLE SKIPPED: {result.get('error')}")
            print("\n=====================================")
            return result

        print("\nRANKINGS:")
        if "rankings" in result and result["rankings"]:
            sorted_models = sorted(result["rankings"].items(), key=lambda x: x[1])
            for model, rank in sorted_models:
                letter = result.get("model_letter_map", {}).get(model, "?")
                print(f"  Rank {rank}: {model} (Response {letter})")
        else:
            print("  No rankings available.")

        print(f"\nBEST MODEL: {result.get('best_model', 'N/A')}")
        if "latency" in result:
             print(f"Latency: {result.get('latency', 0):.2f} seconds")
        
        if "model_letter_map" in result and result["model_letter_map"]:
            print("\nMODEL ASSIGNMENT:")
            for model, letter in result["model_letter_map"].items():
                print(f"  {model} was presented as Response {letter}")
        if "ranking_response" in result:
            print(f"\nRaw ranking response from {self.model}:\n{result['ranking_response']}")
        if "errors" in result and result["errors"]:
            print(f"\nERRORS DURING JUDGING: {result['errors']}")
        
        print("\n=====================================")
        return result