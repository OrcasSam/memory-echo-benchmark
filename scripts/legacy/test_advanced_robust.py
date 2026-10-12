"""
Evaluation harness using the robust memory layer (no JSON schema dependency).
Drop-in replacement for test_advanced.py.

Usage:
    python -m scripts.legacy.test_advanced_robust --backend openai --model gpt-4o-mini --dataset data/locomo10.json
    python -m scripts.legacy.test_advanced_robust --backend deepseek --model deepseek-chat --dataset data/locomo10.json
    python -m scripts.legacy.test_advanced_robust --backend ollama --model qwen2.5:3b --dataset data/locomo10.json
"""

from memory_layer_robust import RobustLLMController, RobustAgenticMemorySystem
from llm_text_parsers import (
    parse_plain_text_answer,
    parse_relevant_parts,
    parse_keywords_response,
)
import os
import json
import argparse
import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from utils import (
    aggregate_metrics,
    calculate_bleu_scores,
    calculate_meteor_score,
    calculate_metrics,
    calculate_rouge_scores,
    simple_tokenize,
)
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

logger = logging.getLogger("amem_robust")


def safe_cache_tag(value: str) -> str:
    """Make a filesystem-safe tag for cache directories."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def collect_llm_usage(agent: "RobustAdvancedMemAgent") -> Dict[str, int]:
    """Collect usage from both LLM controllers used by the agent."""
    total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }
    controllers = [
        agent.memory_system.llm_controller.llm,
        agent.retriever_llm.llm,
    ]
    for controller in controllers:
        usage = getattr(controller, "usage", None)
        if not usage:
            continue
        for key in total:
            total[key] += int(usage.get(key, 0) or 0)
    return total


def add_usage(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + int(value or 0)


def remove_if_exists(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        logger.warning("Could not remove cache checkpoint: %s", path)


def empty_metrics(metrics_mode: str) -> Dict[str, float]:
    if metrics_mode == "none":
        return {}
    base = {
        "exact_match": 0,
        "f1": 0.0,
        "rouge1_f": 0.0,
        "rouge2_f": 0.0,
        "rougeL_f": 0.0,
        "bleu1": 0.0,
        "bleu2": 0.0,
        "bleu3": 0.0,
        "bleu4": 0.0,
        "meteor": 0.0,
    }
    if metrics_mode == "full":
        base.update({
            "bert_precision": 0.0,
            "bert_recall": 0.0,
            "bert_f1": 0.0,
            "sbert_similarity": 0.0,
        })
    return base


def calculate_basic_metrics(prediction: str, reference: str) -> Dict[str, float]:
    prediction = str(prediction).strip()
    reference = str(reference).strip()
    if not prediction or not reference:
        return empty_metrics("basic")

    exact_match = int(prediction.lower() == reference.lower())
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "exact_match": exact_match,
        "f1": f1,
        **calculate_rouge_scores(prediction, reference),
        **calculate_bleu_scores(prediction, reference),
        "meteor": calculate_meteor_score(prediction, reference),
    }


def calculate_selected_metrics(prediction: str, reference: str, metrics_mode: str) -> Dict[str, float]:
    if metrics_mode == "none":
        return {}
    if not reference:
        return empty_metrics(metrics_mode)
    if metrics_mode == "basic":
        return calculate_basic_metrics(prediction, reference)
    return calculate_metrics(prediction, reference)


def build_results_payload(
    model: str,
    dataset_path: str,
    metrics_mode: str,
    quiet_qa_log: bool,
    total_questions: int,
    category_counts: Dict[int, int],
    memories_dir: str,
    cache_stats: Dict[str, int],
    api_usage: Dict[str, int],
    all_metrics: List[Dict[str, float]],
    all_categories: List[int],
    results: List[Dict],
    completed_samples: int,
    target_samples: int,
    is_partial: bool,
) -> Dict:
    return {
        "model": model,
        "dataset": dataset_path,
        "memory_layer": "robust",
        "metrics_mode": metrics_mode,
        "quiet_qa_log": quiet_qa_log,
        "completed_samples": completed_samples,
        "target_samples": target_samples,
        "is_partial": is_partial,
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "memory_cache": {
            "directory": memories_dir,
            **cache_stats,
        },
        "api_usage": api_usage,
        "aggregate_metrics": aggregate_metrics(all_metrics, all_categories),
        "individual_results": results,
    }


class RobustAdvancedMemAgent:
    """Agent using the robust memory system with plain-text LLM calls."""

    def __init__(self, model, backend, retrieve_k, temperature_c5,
                 sglang_host="http://localhost", sglang_port=30000):
        self.memory_system = RobustAgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retriever_llm = RobustLLMController(
            backend=backend,
            model=model,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(content, k=k)

    def retrieve_memory_llm(self, memories_text, query):
        """Select relevant parts of conversation memories — plain text, no JSON schema."""
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts of the conversation that would help answer this specific question.
If no parts are relevant, return the input unchanged."""

        response = self.retriever_llm.llm.get_completion(prompt)
        return parse_relevant_parts(response)

    def generate_query_llm(self, question):
        """Generate query keywords — plain text, no JSON schema."""
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

        response = self.retriever_llm.llm.get_completion(prompt)
        result = parse_keywords_response(response)
        logger.debug("generate_query_llm response: %s", result)
        return result

    def answer_question(self, question: str, category: int, answer: str) -> tuple:
        """Generate answer for a question — plain text, no JSON schema."""
        keywords = self.generate_query_llm(question)
        raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)
        context = raw_context

        assert category in [1, 2, 3, 4, 5]

        if category == 5:
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:"""
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
            temperature = 0.7
        elif category == 3:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7

        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt, temperature=temperature,
            )
        except Exception as e:
            logger.warning("answer_question failed: %s — returning empty", e)
            response = ""
        return response, user_prompt, raw_context


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    eval_logger = logging.getLogger('locomo_eval_robust')
    eval_logger.setLevel(logging.INFO)
    eval_logger.handlers.clear()
    eval_logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    eval_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        eval_logger.addHandler(file_handler)

    return eval_logger


def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None,
                     ratio: float = 1.0, backend: str = "sglang",
                     temperature_c5: float = 0.5, retrieve_k: int = 10,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000,
                     memory_progress_interval: int = 10,
                     metrics_mode: str = "basic",
                     quiet_qa_log: bool = False,
                     max_samples: Optional[int] = None):
    """Evaluate the robust agent on the LoComo dataset."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_robust_{model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(str(Path(__file__).resolve().parents[2]), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_logger = setup_logger(log_path)
    eval_logger.info(f"Loading dataset from {dataset_path}")
    eval_logger.info(f"Using ROBUST memory layer (no JSON schema dependency)")
    eval_logger.info(f"Metrics mode: {metrics_mode}")
    eval_logger.info(f"Quiet QA log: {quiet_qa_log}")
    eval_logger.info(f"Max samples: {max_samples if max_samples else 'all'}")

    samples = load_locomo_dataset(dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")
    if max_samples is not None:
        samples = samples[:max(1, min(max_samples, len(samples)))]
        eval_logger.info(f"Limiting run to {len(samples)} samples")

    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)
    resume_from_sample = 0

    i = 0
    error_num = 0
    memories_dir = os.path.join(
        str(Path(__file__).resolve().parents[2]),
        "cached_memories_robust_{}_{}_{}".format(
            safe_cache_tag(backend),
            safe_cache_tag(model),
            safe_cache_tag(Path(dataset_path).stem),
        ),
    )
    os.makedirs(memories_dir, exist_ok=True)
    eval_logger.info(f"Memory cache directory: {memories_dir}")
    allow_categories = [1, 2, 3, 4, 5]
    cache_stats = {"hits": 0, "misses": 0, "created_memories": 0}
    api_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }

    if output_path and os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                previous_results = json.load(f)
            if previous_results.get("is_partial") and previous_results.get("individual_results"):
                resume_from_sample = int(previous_results.get("completed_samples", 0) or 0)
                results = previous_results.get("individual_results", [])
                all_metrics = [item.get("metrics", {}) for item in results]
                all_categories = [item.get("category") for item in results]
                total_questions = int(previous_results.get("total_questions", len(results)) or len(results))
                category_counts = defaultdict(int)
                for item in results:
                    category_counts[item.get("category")] += 1
                previous_usage = previous_results.get("api_usage", {})
                for key in api_usage:
                    api_usage[key] = int(previous_usage.get(key, 0) or 0)
                previous_cache = previous_results.get("memory_cache", {})
                for key in ("hits", "misses", "created_memories"):
                    cache_stats[key] = int(previous_cache.get(key, cache_stats.get(key, 0)) or 0)
                eval_logger.info(
                    f"Resuming result file from sample {resume_from_sample}: "
                    f"{total_questions} questions already evaluated"
                )
        except Exception as e:
            eval_logger.warning(f"Could not resume previous result file: {e}")

    for sample_idx, sample in enumerate(samples):
        if sample_idx < resume_from_sample:
            eval_logger.info(f"Skipping completed sample {sample_idx}")
            continue

        agent = RobustAdvancedMemAgent(model, backend, retrieve_k, temperature_c5,
                                       sglang_host, sglang_port)

        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(
            memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )
        partial_memory_cache_file = os.path.join(memories_dir, f"partial_memory_cache_sample_{sample_idx}.pkl")
        partial_retriever_cache_file = os.path.join(memories_dir, f"partial_retriever_cache_sample_{sample_idx}.pkl")
        partial_retriever_cache_embeddings_file = os.path.join(
            memories_dir, f"partial_retriever_cache_embeddings_sample_{sample_idx}.npy"
        )

        if os.path.exists(memory_cache_file):
            cache_stats["hits"] += 1
            eval_logger.info(f"Loading cached memories for sample {sample_idx}")
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            if os.path.exists(retriever_cache_file):
                eval_logger.info(f"Found retriever cache files")
                agent.memory_system.retriever = agent.memory_system.retriever.load(
                    retriever_cache_file, retriever_cache_embeddings_file
                )
            else:
                eval_logger.info(f"No retriever cache found, loading from memory")
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                    cached_memories, 'all-MiniLM-L6-v2'
                )
            eval_logger.info(f"Successfully loaded {len(cached_memories)} memories")
        else:
            cache_stats["misses"] += 1
            flat_turns = [
                (turns.date_time, turn)
                for turns in sample.conversation.sessions.values()
                for turn in turns.turns
            ]
            start_turn = 0
            if os.path.exists(partial_memory_cache_file):
                eval_logger.info(f"Found partial memory checkpoint for sample {sample_idx}. Resuming.")
                with open(partial_memory_cache_file, 'rb') as f:
                    partial_payload = pickle.load(f)
                cached_memories = partial_payload.get("memories", {})
                start_turn = int(partial_payload.get("processed_turns", len(cached_memories)) or 0)
                agent.memory_system.memories = cached_memories
                if os.path.exists(partial_retriever_cache_file):
                    agent.memory_system.retriever = agent.memory_system.retriever.load(
                        partial_retriever_cache_file,
                        partial_retriever_cache_embeddings_file,
                    )
                else:
                    agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                        cached_memories, 'all-MiniLM-L6-v2'
                    )
                eval_logger.info(
                    f"Resumed sample {sample_idx}: {start_turn}/{len(flat_turns)} turns already cached"
                )
            else:
                eval_logger.info(f"No cached memories found for sample {sample_idx}. Creating new memories.")

            for turn_idx, (turn_datatime, turn) in enumerate(flat_turns[start_turn:], start=start_turn + 1):
                conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
                agent.add_memory(conversation_tmp, time=turn_datatime)
                if turn_idx % memory_progress_interval == 0 or turn_idx == len(flat_turns):
                    eval_logger.info(
                        f"Sample {sample_idx} memory build progress: "
                        f"{turn_idx}/{len(flat_turns)} turns, {len(agent.memory_system.memories)} memories"
                    )
                    with open(partial_memory_cache_file, 'wb') as f:
                        pickle.dump({
                            "processed_turns": turn_idx,
                            "memories": agent.memory_system.memories,
                        }, f)
                    agent.memory_system.retriever.save(
                        partial_retriever_cache_file,
                        partial_retriever_cache_embeddings_file,
                    )

            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            remove_if_exists(partial_memory_cache_file)
            remove_if_exists(partial_retriever_cache_file)
            remove_if_exists(partial_retriever_cache_embeddings_file)
            cache_stats["created_memories"] += len(memories_to_cache)
            eval_logger.info(f"Successfully cached {len(memories_to_cache)} memories")

        eval_logger.info(f"Processing sample {sample_idx + 1}/{len(samples)}")
        already_answered_in_sample = sum(
            1 for item in results if int(item.get("sample_id", -1)) == sample_idx
        )
        eligible_seen_in_sample = 0
        if already_answered_in_sample:
            eval_logger.info(
                f"Resuming sample {sample_idx} QA: "
                f"{already_answered_in_sample} answers already checkpointed"
            )

        for qa in sample.qa:
            if int(qa.category) in allow_categories:
                eligible_seen_in_sample += 1
                if eligible_seen_in_sample <= already_answered_in_sample:
                    continue

                total_questions += 1
                category_counts[qa.category] += 1

                prediction, user_prompt, raw_context = agent.answer_question(
                    qa.question, qa.category, qa.final_answer
                )

                # Parse the prediction (handles both JSON and plain text)
                prediction = parse_plain_text_answer(prediction)

                eval_logger.info(f"Question {total_questions}: {qa.question}")
                eval_logger.info(f"Prediction: {prediction}")
                eval_logger.info(f"Reference: {qa.final_answer}")
                eval_logger.info(f"Category: {qa.category}")
                if not quiet_qa_log:
                    eval_logger.info(f"User Prompt: {user_prompt}")
                    eval_logger.info(f"Raw Context: {raw_context}")

                metrics = calculate_selected_metrics(prediction, qa.final_answer, metrics_mode)

                all_metrics.append(metrics)
                all_categories.append(qa.category)

                result = {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": qa.category,
                    "metrics": metrics,
                }
                results.append(result)

                if total_questions % 10 == 0:
                    eval_logger.info(f"Processed {total_questions} questions")
                if output_path and total_questions % 5 == 0:
                    checkpoint_results = build_results_payload(
                        model=model,
                        dataset_path=dataset_path,
                        metrics_mode=metrics_mode,
                        quiet_qa_log=quiet_qa_log,
                        total_questions=total_questions,
                        category_counts=category_counts,
                        memories_dir=memories_dir,
                        cache_stats=cache_stats,
                        api_usage=api_usage,
                        all_metrics=all_metrics,
                        all_categories=all_categories,
                        results=results,
                        completed_samples=sample_idx,
                        target_samples=len(samples),
                        is_partial=True,
                    )
                    with open(output_path, 'w') as f:
                        json.dump(checkpoint_results, f, indent=2)

        add_usage(api_usage, collect_llm_usage(agent))

        if output_path:
            checkpoint_results = build_results_payload(
                model=model,
                dataset_path=dataset_path,
                metrics_mode=metrics_mode,
                quiet_qa_log=quiet_qa_log,
                total_questions=total_questions,
                category_counts=category_counts,
                memories_dir=memories_dir,
                cache_stats=cache_stats,
                api_usage=api_usage,
                all_metrics=all_metrics,
                all_categories=all_categories,
                results=results,
                completed_samples=sample_idx + 1,
                target_samples=len(samples),
                is_partial=(sample_idx + 1) < len(samples),
            )
            with open(output_path, 'w') as f:
                json.dump(checkpoint_results, f, indent=2)
            eval_logger.info(
                f"Checkpoint results saved to {output_path} "
                f"({sample_idx + 1}/{len(samples)} samples)"
            )

    final_results = build_results_payload(
        model=model,
        dataset_path=dataset_path,
        metrics_mode=metrics_mode,
        quiet_qa_log=quiet_qa_log,
        total_questions=total_questions,
        category_counts=category_counts,
        memories_dir=memories_dir,
        cache_stats=cache_stats,
        api_usage=api_usage,
        all_metrics=all_metrics,
        all_categories=all_categories,
        results=results,
        completed_samples=len(samples),
        target_samples=len(samples),
        is_partial=False,
    )
    aggregate_results = final_results["aggregate_metrics"]
    eval_logger.info(f"Error number: {error_num}")
    eval_logger.info(f"Memory cache stats: {cache_stats}")
    eval_logger.info(f"API usage: {api_usage}")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        eval_logger.info(f"Results saved to {output_path}")

    eval_logger.info("Evaluation Summary:")
    eval_logger.info(f"Total questions evaluated: {total_questions}")
    eval_logger.info("Category Distribution:")
    for category, count in sorted(category_counts.items()):
        eval_logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")

    eval_logger.info("Aggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        eval_logger.info(f"{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            eval_logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                eval_logger.info(f"    {stat_name}: {value:.4f}")

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate robust text-only agent on LoComo dataset (no JSON schema dependency)"
    )
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="openai",
                        help="Backend to use (openai, deepseek, ollama, sglang, or vllm)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                        help="Temperature for category 5 questions")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang server port (for sglang backend)")
    parser.add_argument("--memory_progress_interval", type=int, default=10,
                        help="How often to log and checkpoint memory-building turns")
    parser.add_argument("--metrics_mode", choices=["full", "basic", "none"], default="basic",
                        help="Metrics to compute during the run: full includes BERTScore/SBERT, basic skips them")
    parser.add_argument("--quiet_qa_log", action="store_true",
                        help="Do not log full prompts and raw retrieved context for every question")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on the number of samples to evaluate")
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    dataset_path = os.path.join(str(Path(__file__).resolve().parents[2]), args.dataset)
    output_path = os.path.join(str(Path(__file__).resolve().parents[2]), args.output) if args.output else None

    evaluate_dataset(
        dataset_path, args.model, output_path, args.ratio,
        args.backend, args.temperature_c5, args.retrieve_k,
        args.sglang_host, args.sglang_port, args.memory_progress_interval,
        args.metrics_mode, args.quiet_qa_log, args.max_samples,
    )


if __name__ == "__main__":
    main()
