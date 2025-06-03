import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import google.generativeai as genai
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    SemanticSimilarity
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings, time
warnings.filterwarnings('ignore')

DELAY_TIME = 10  # Delay time between evaluations

class RAGASEvaluator:
    """
    RAGAS Evaluator for Vietnamese Legal Q&A System using RAGAS v0.2+
    Evaluates: Faithfulness, Answer Relevancy, Semantic Similarity.
    NoiseSensitivity has been completely removed.
    Enhanced with API key rotation functionality.
    """

    def __init__(self, google_api_keys: List[str], embedding_model_name: str = "intfloat/multilingual-e5-base", use_google_embeddings: bool = False):
        """
        Initialize RAGAS evaluator with Google Generative AI and embeddings.

        Args:
            google_api_keys (List[str]): List of Google API keys for rotation.
            embedding_model_name (str): Name of the embedding model to use (if not using Google embeddings).
            use_google_embeddings (bool): Whether to use Google's embeddings or HuggingFace.
        """
        self.google_api_keys = google_api_keys
        self.current_key_index = 0
        self.current_api_key = self.google_api_keys[0]
        self.embedding_model_name = embedding_model_name
        self.use_google_embeddings = use_google_embeddings
        self.setup_llm_and_embeddings()
        self.setup_metrics()
        print(f"🔑 Initialized with {len(self.google_api_keys)} API keys for rotation")

    def rotate_api_key(self):
        """Rotate to the next API key in the list."""
        self.current_key_index = (self.current_key_index + 1) % len(self.google_api_keys)
        self.current_api_key = self.google_api_keys[self.current_key_index]
        print(f"🔄 Switched to API key #{self.current_key_index + 1}")

        # Reinitialize LLM and embeddings with new API key
        self.setup_llm_and_embeddings()
        self.setup_metrics()

    def setup_llm_and_embeddings(self):
        """Setup LLM and embeddings for RAGAS evaluation."""
        # Configure Google Generative AI with current API key
        genai.configure(api_key=self.current_api_key)

        # Initialize LLM (using Gemini Flash) with RAGAS wrapper
        langchain_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=self.current_api_key,
            temperature=0.1
        )
        self.llm = LangchainLLMWrapper(langchain_llm)

        # Initialize embeddings with RAGAS wrapper
        if self.use_google_embeddings:
            # Use Google's embeddings
            langchain_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.current_api_key
            )
        else:
            # Use HuggingFace embeddings (doesn't need API key rotation)
            langchain_embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU is available
                encode_kwargs={'normalize_embeddings': True}
            )
        self.embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

    def setup_metrics(self):
        """Setup RAGAS metrics with custom LLM and embeddings."""
        # Initialize metrics with LLM and embeddings
        self.faithfulness = Faithfulness(llm=self.llm)
        self.answer_relevancy = AnswerRelevancy(
            llm=self.llm,
            embeddings=self.embeddings
        )
        self.semantic_similarity = SemanticSimilarity(embeddings=self.embeddings)

        self.metrics = [
            self.faithfulness,
            self.answer_relevancy,
            self.semantic_similarity
        ]

    def prepare_single_turn_sample(self, json_data: Dict[str, Any]) -> SingleTurnSample:
        """
        Prepare SingleTurnSample from JSON format for RAGAS evaluation.

        Args:
            json_data (Dict): Input JSON data.

        Returns:
            SingleTurnSample: Formatted sample for RAGAS.
        """
        user_input = json_data.get("question", "")
        response = json_data.get("answer_generation", {}).get("generated_answer", "")
        reference = json_data.get("truth_answer", "")

        retrieved_contexts = []
        if "retrival_results" in json_data:
            for result in json_data["retrival_results"]:
                context_content = result.get('content', '')
                # Ensure context is not empty, provide a placeholder if necessary
                context = f"Điều {result.get('dieu_number', 'N/A')}: {context_content if context_content else 'Không có nội dung.'}"
                retrieved_contexts.append(context)

        # If no contexts found, use the truth answer as context, or an empty string if truth_answer is also missing.
        if not retrieved_contexts:
            retrieved_contexts = [reference if reference else "Không có ngữ cảnh được cung cấp."]

        return SingleTurnSample(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
            reference=reference
        )

    def evaluate_batch(self, json_data_list: List[Dict[str, Any]], save_to_csv: bool = True, csv_filename: str = "ragas_evaluation_results.csv") -> pd.DataFrame:
        """
        Evaluate multiple samples and save detailed results to CSV.
        Rotates API key every 5 evaluations.

        Args:
            json_data_list (List): List of sample data.
            save_to_csv (bool): Whether to save results to CSV file.
            csv_filename (str): Name of the CSV file to save results.

        Returns:
            pd.DataFrame: Evaluation results.
        """
        results_list = []
        for i, json_data in enumerate(json_data_list):
            # Rotate API key every 5 evaluations
            if i > 0 and i % 5 == 0:
                print(f"🔄 Rotating API key after {i} evaluations...")
                self.rotate_api_key()
                time.sleep(2)  # Brief pause after key rotation

            print(f"Evaluating sample {i+1}/{len(json_data_list)} (API Key #{self.current_key_index + 1})...")
            try:
                # comprehensive_evaluation now directly returns the summary scores dict
                scores_summary = self.comprehensive_evaluation(json_data)
                sample_result = {
                    "sample_id": i + 1,
                    "question": json_data.get("question", ""),
                    "generated_answer": json_data.get("answer_generation", {}).get("generated_answer", ""),
                    "truth_answer": json_data.get("truth_answer", ""),
                    "model_used": json_data.get("answer_generation", {}).get("model", ""),
                    "generation_time": json_data.get("answer_generation", {}).get("generation_time", 0),
                    "faithfulness": scores_summary.get("faithfulness")[0],
                    "answer_relevancy": scores_summary.get("answer_relevancy", 0.0)[0],
                    "semantic_similarity": scores_summary.get("semantic_similarity", 0.0)[0],
                    "num_retrieved_contexts": len(json_data.get("retrival_results", [])),
                    "retrieved_contexts": "; ".join([f"Điều {res.get('dieu_number', 'N/A')}" for res in json_data.get("retrival_results", [])]),
                    "evaluation_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "api_key_used": self.current_key_index + 1,  # Track which API key was used
                    'overall_quality': "K"
                }
                if json_data.get("retrival_results"):
                    first_result = json_data["retrival_results"][0]
                    sample_result.update({
                        "bm25_score": first_result.get("bm25_score", 0),
                        "vector_similarity_score": first_result.get("vector_similarity_score", 0),
                        "combined_score": first_result.get("combined_score", 0)
                    })
                print("Query:", sample_result['question'])
                print("Truth Answer:", sample_result['truth_answer'][:50])
                print("Gen Answer:", sample_result['generated_answer'][:50])
                print('faithfulness         ','answer_relevancy          ','semantic_similarity')
                print(sample_result['faithfulness'], sample_result['answer_relevancy'], sample_result['semantic_similarity'])
                results_list.append(sample_result)
            except Exception as e:
                print(f"❌ Error evaluating sample {i+1}: {str(e)}")
                error_result = {
                    "sample_id": i + 1, "question": json_data.get("question", ""),
                    "generated_answer": json_data.get("answer_generation", {}).get("generated_answer", ""),
                    "truth_answer": json_data.get("truth_answer", ""),
                    "model_used": json_data.get("answer_generation", {}).get("model", ""),
                    "generation_time": json_data.get("answer_generation", {}).get("generation_time", 0),
                    "faithfulness": 0.0, "answer_relevancy": 0.0, "semantic_similarity": 0.0,
                    "num_retrieved_contexts": len(json_data.get("retrival_results", [])),
                    "retrieved_contexts": "ERROR", "overall_quality": "ERROR",
                    "evaluation_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "api_key_used": self.current_key_index + 1,
                    "error_message": str(e),
                    'overall_quality': "ERROR"
                }
                results_list.append(error_result)

            if i < len(json_data_list) - 1:
                print(f"Waiting {DELAY_TIME} seconds before next evaluation...")
                time.sleep(DELAY_TIME)

        df_results = pd.DataFrame(results_list)
        if save_to_csv and not df_results.empty:
            df_results.to_csv(csv_filename, index=False, encoding='utf-8')
            print(f"✅ Detailed evaluation results saved to: {csv_filename}")
        return df_results

    def comprehensive_evaluation(self, json_data: Dict[str, Any]):
        """
        Comprehensive evaluation for a single sample.

        Args:
            json_data (Dict): Input data.

        Returns:
            Dict: Summary of evaluation scores (faithfulness, answer_relevancy, semantic_similarity).
        """
        original_sample = self.prepare_single_turn_sample(json_data)
        original_dataset = EvaluationDataset(samples=[original_sample])

        # self.metrics already contains only the desired metrics
        evaluation_result_object = evaluate(
            dataset=original_dataset,
            metrics=self.metrics
        )

        summary_scores = evaluation_result_object.to_pandas()
        return summary_scores # Directly return the scores dictionary

    def generate_report(self, summary_scores: Dict[str, float]) -> str:
        """
        Generate evaluation report from summary scores.

        Args:
            summary_scores (Dict): Summary of evaluation scores.

        Returns:
            str: Formatted report.
        """
        model_name = self.embedding_model_name if not self.use_google_embeddings else "Google Embeddings (models/embedding-001)"
        report = f"""
=== RAGAS EVALUATION REPORT FOR VIETNAMESE LEGAL Q&A ===
Embedding Model: {model_name}
API Keys Used: {len(self.google_api_keys)} keys with rotation every 5 evaluations

OVERALL SCORES:
- Faithfulness: {summary_scores.get('faithfulness', 'N/A'):.4f}
  (Measures how grounded the answer is in the given context)

- Answer Relevancy: {summary_scores.get('answer_relevancy', 'N/A'):.4f}
  (Measures how relevant the answer is to the given question)

- Semantic Similarity: {summary_scores.get('semantic_similarity', 'N/A'):.4f}
  (Measures semantic similarity between generated and ground truth answers)

INTERPRETATION:
- Faithfulness: Higher is better (0.0 - 1.0)
- Answer Relevancy: Higher is better (0.0 - 1.0)
- Semantic Similarity: Higher is better (0.0 - 1.0)

        """
        return report.strip()

    def calculate_average_scores(self, df_results: pd.DataFrame) -> Dict[str, Any]:
        """Calculate average scores and statistics from evaluation results."""
        valid_results = df_results[df_results['overall_quality'] != 'ERROR']
        if valid_results.empty:
            return {
                "error": "No valid evaluation results found",
                "total_samples": len(df_results),
                "successful_evaluations": 0,
                "failed_evaluations": len(df_results)
            }

        metrics_to_average = ['faithfulness', 'answer_relevancy', 'semantic_similarity']
        average_scores_stats = {}
        for metric in metrics_to_average:
            if metric in valid_results.columns:
                # Ensure scores are numeric and drop NaN before calculation
                scores = pd.to_numeric(valid_results[metric], errors='coerce').dropna()
                if not scores.empty:
                    average_scores_stats[f"{metric}_mean"] = float(scores.mean())
                    average_scores_stats[f"{metric}_std"] = float(scores.std())
                    average_scores_stats[f"{metric}_min"] = float(scores.min())
                    average_scores_stats[f"{metric}_max"] = float(scores.max())
                    average_scores_stats[f"{metric}_median"] = float(scores.median())
                else:
                    for stat in ['mean', 'std', 'min', 'max', 'median']:
                        average_scores_stats[f"{metric}_{stat}"] = 0.0

        overall_avg_components = [average_scores_stats.get(f'{m}_mean', 0.0) for m in metrics_to_average]
        valid_overall_components = [s for s in overall_avg_components if s > 0]
        overall_avg = np.mean(valid_overall_components) if valid_overall_components else 0.0

        quality_counts = valid_results['overall_quality'].value_counts().to_dict()
        model_performance = {}
        if 'model_used' in valid_results.columns:
            for model in valid_results['model_used'].unique():
                model_data = valid_results[valid_results['model_used'] == model]
                if not model_data.empty:
                    model_metrics_means = []
                    for metric in metrics_to_average:
                         # Ensure scores are numeric and drop NaN before calculation
                        metric_scores = pd.to_numeric(model_data[metric], errors='coerce').dropna()
                        if not metric_scores.empty:
                            model_metrics_means.append(metric_scores.mean())

                    valid_model_metric_means = [m for m in model_metrics_means if pd.notna(m) and m > 0]

                    model_performance[model] = {
                        'count': len(model_data),
                        'avg_faithfulness': float(pd.to_numeric(model_data['faithfulness'], errors='coerce').mean() if not pd.to_numeric(model_data['faithfulness'], errors='coerce').dropna().empty else 0.0),
                        'avg_answer_relevancy': float(pd.to_numeric(model_data['answer_relevancy'], errors='coerce').mean() if not pd.to_numeric(model_data['answer_relevancy'], errors='coerce').dropna().empty else 0.0),
                        'avg_semantic_similarity': float(pd.to_numeric(model_data['semantic_similarity'], errors='coerce').mean() if not pd.to_numeric(model_data['semantic_similarity'], errors='coerce').dropna().empty else 0.0),
                        'overall_avg': float(np.mean(valid_model_metric_means)) if valid_model_metric_means else 0.0
                    }

        # Add API key usage statistics
        api_key_stats = {}
        if 'api_key_used' in valid_results.columns:
            api_key_counts = valid_results['api_key_used'].value_counts().to_dict()
            api_key_stats = {f"api_key_{key}": count for key, count in api_key_counts.items()}

        generation_time_stats = {}
        if 'generation_time' in valid_results.columns:
            gen_times = pd.to_numeric(valid_results['generation_time'], errors='coerce').dropna()
            if not gen_times.empty:
                generation_time_stats = {
                    'mean_generation_time': float(gen_times.mean()),
                    'std_generation_time': float(gen_times.std()),
                    'min_generation_time': float(gen_times.min()),
                    'max_generation_time': float(gen_times.max()),
                    'median_generation_time': float(gen_times.median())
                }
        return {
            "evaluation_summary": {
                "total_samples": len(df_results),
                "successful_evaluations": len(valid_results),
                "failed_evaluations": len(df_results) - len(valid_results),
                "success_rate": (len(valid_results) / len(df_results)) if len(df_results) > 0 else 0.0,
                "overall_average_score": float(overall_avg),
                "evaluation_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "api_keys_count": len(self.google_api_keys)
            },
            "metrics_statistics": average_scores_stats,
            "quality_distribution": quality_counts,
            "model_performance": model_performance,
            "generation_time_statistics": generation_time_stats,
            "api_key_usage": api_key_stats,
            "configuration": {
                "embedding_model": self.embedding_model_name if not self.use_google_embeddings else "Google Embeddings",
                "use_google_embeddings": self.use_google_embeddings,
                "llm_model": "gemini-2.0-flash",
                "api_keys_count": len(self.google_api_keys),
                "rotation_frequency": 5
            }
        }

    def save_average_scores_to_json(self, average_scores: Dict[str, Any], json_filename: str = "ragas_average_scores.json"):
        try:
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(average_scores, f, indent=2, ensure_ascii=False)
            print(f"✅ Average scores and statistics saved to: {json_filename}")
        except Exception as e:
            print(f"❌ Error saving average scores to JSON: {str(e)}")

    def evaluate_dataset_complete(self, json_data_list: List[Dict[str, Any]],
                                  csv_filename: str = "ragas_evaluation_results.csv",
                                  json_filename: str = "ragas_average_scores.json") -> tuple[pd.DataFrame, Dict[str, Any]]:
        print(f"🚀 Starting comprehensive evaluation of {len(json_data_list)} samples...")
        print(f"📊 Using embedding model: {'Google Embeddings' if self.use_google_embeddings else self.embedding_model_name}")
        print(f"🔑 Using {len(self.google_api_keys)} API keys with rotation every 5 evaluations")
        print("\n📋 Step 1: Evaluating all samples...")
        df_results = self.evaluate_batch(json_data_list, save_to_csv=True, csv_filename=csv_filename)
        print("\n📊 Step 2: Calculating average scores and statistics...")
        average_scores = self.calculate_average_scores(df_results)
        print("\n💾 Step 3: Saving average scores to JSON...")
        self.save_average_scores_to_json(average_scores, json_filename)
        print("\n" + "="*80)
        print("📈 EVALUATION SUMMARY")
        print("="*80)
        if "evaluation_summary" in average_scores:
            summary = average_scores["evaluation_summary"]
            print(f"Total Samples: {summary.get('total_samples', 'N/A')}")
            print(f"Successful Evaluations: {summary.get('successful_evaluations', 'N/A')}")
            print(f"Failed Evaluations: {summary.get('failed_evaluations', 'N/A')}")
            success_rate = summary.get('success_rate', 0.0)
            print(f"Success Rate: {success_rate:.2%}")
            overall_avg_score = summary.get('overall_average_score', 0.0)
            print(f"Overall Average Score: {overall_avg_score:.4f}")
            print(f"API Keys Used: {summary.get('api_keys_count', 'N/A')}")

            if "api_key_usage" in average_scores:
                print(f"\n🔑 API KEY USAGE:")
                for key, count in average_scores["api_key_usage"].items():
                    print(f"  • {key}: {count} evaluations")

            if "metrics_statistics" in average_scores:
                print(f"\n📊 AVERAGE METRICS:")
                ms = average_scores["metrics_statistics"]
                print(f"  • Faithfulness: {ms.get('faithfulness_mean', 0.0):.4f} ± {ms.get('faithfulness_std', 0.0):.4f}")
                print(f"  • Answer Relevancy: {ms.get('answer_relevancy_mean', 0.0):.4f} ± {ms.get('answer_relevancy_std', 0.0):.4f}")
                print(f"  • Semantic Similarity: {ms.get('semantic_similarity_mean', 0.0):.4f} ± {ms.get('semantic_similarity_std', 0.0):.4f}")
            if "quality_distribution" in average_scores:
                print(f"\n🎯 QUALITY DISTRIBUTION:")
                for quality, count in average_scores["quality_distribution"].items():
                    print(f"  • {quality}: {count} samples")
        print("\n✅ Evaluation completed successfully!")
        print(f"📄 Detailed results: {csv_filename}")
        print(f"📊 Summary statistics: {json_filename}")
        return df_results, average_scores

    def evaluate_dataset_complete_with_report(self, json_data_list: List[Dict[str, Any]],
                                              csv_filename: str = "ragas_evaluation_results.csv",
                                              json_filename: str = "ragas_average_scores.json",
                                              report_filename: str = "ragas_detailed_report.txt") -> tuple[pd.DataFrame, Dict[str, Any], str]:
        df_results, average_scores = self.evaluate_dataset_complete(
            json_data_list, csv_filename, json_filename
        )
        print("\n📄 Step 4: Generating detailed report...") # Adjusted step number
        report_content = self.generate_detailed_report(df_results, average_scores)
        self.save_detailed_report(df_results, average_scores, report_filename)
        print(f"📋 Comprehensive report: {report_filename}")
        return df_results, average_scores, report_content

    def generate_detailed_report(self, df_results: pd.DataFrame, average_scores: Dict[str, Any]) -> str:
        """Generate a detailed evaluation report."""
        report_lines = [
            "=" * 80,
            "RAGAS EVALUATION DETAILED REPORT",
            "=" * 80,
            "",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Embedding Model: {self.embedding_model_name if not self.use_google_embeddings else 'Google Embeddings'}",
            f"API Keys Used: {len(self.google_api_keys)} with rotation every 5 evaluations",
            "",
            "EVALUATION SUMMARY:",
            "-" * 40,
        ]

        if "evaluation_summary" in average_scores:
            summary = average_scores["evaluation_summary"]
            report_lines.extend([
                f"Total Samples: {summary.get('total_samples', 'N/A')}",
                f"Successful Evaluations: {summary.get('successful_evaluations', 'N/A')}",
                f"Failed Evaluations: {summary.get('failed_evaluations', 'N/A')}",
                f"Success Rate: {summary.get('success_rate', 0.0):.2%}",
                f"Overall Average Score: {summary.get('overall_average_score', 0.0):.4f}",
            ])

        return "\n".join(report_lines)

    def save_detailed_report(self, df_results: pd.DataFrame, average_scores: Dict[str, Any], report_filename: str):
        """Save detailed report to file."""
        try:
            report_content = self.generate_detailed_report(df_results, average_scores)
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✅ Detailed report saved to: {report_filename}")
        except Exception as e:
            print(f"❌ Error saving detailed report: {str(e)}")

def read_json_file(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_to_json(data, output_path):
    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Example usage with multiple API keys
    google_api_keys = [
                "AIzaSyCKtN98H-n2idRhIgWpvzcw-4cqdzik9rE",
                "AIzaSyAhZsYmuI9Waxj1o4ZXcT6lCYszhmVpWcM",
                "AIzaSyClqpWZjhwiFJ7kXJdalC-HOQ4GzNbGkq8",
                "AIzaSyAdis532XF3hKGdIlZ7PjvT0U4pi1FhWDw",
                "AIzaSyA8HYpppKJ84wAveUCdeMy6mTgETPNbQmw",
    ]

    # Filter out placeholder keys
    valid_api_keys = [key for key in google_api_keys if not key.startswith("YOUR_") and key != ""]

    if not valid_api_keys:
        print("⚠️ WARNING: Please replace placeholder API keys with your actual Google API keys.")
        valid_api_keys = ["AIzaSyClqpWZjhwiFJ7kXJdalC-HOQ4GzNbGkq8"]  # Fallback to original key

    dataset = read_json_file("data/evalations/answer_evaluation/answers_gemini.json")
    # dataset = dataset[:100]
    print("🔧 Initializing RAGAS Evaluator with API Key Rotation...")
    try:
        evaluator = RAGASEvaluator(
            google_api_keys=valid_api_keys,
            use_google_embeddings=False,
            embedding_model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
        )
        print("\n🚀 Running complete evaluation pipeline with detailed report...")
        df_results, average_scores, report_content = evaluator.evaluate_dataset_complete_with_report(
            json_data_list=dataset,
            csv_filename="vietnamese_legal_qa_eval.csv",
            json_filename="vietnamese_legal_qa_avg.json",
            report_filename="vietnamese_legal_qa_report.txt"
        )
        print("\n📋 SAMPLE DETAILED RESULTS (DataFrame):")
        print("="*80)
        if not df_results.empty:
            display_cols = ['sample_id', 'faithfulness', 'answer_relevancy', 'semantic_similarity', 'api_key_used', 'overall_quality']
            print(df_results[display_cols].to_string(index=False))
        print("\n📊 AVERAGE SCORES (JSON Content Preview):")
        print("="*80)
        print(json.dumps(average_scores, indent=2, ensure_ascii=False))
        print("\n📄 DETAILED REPORT PREVIEW (First 50 lines):")
        print("="*80)
        report_lines = report_content.split('\n')
        for line in report_lines[:50]: print(line)
        if len(report_lines) > 50: print("... (see full report in file)")
    except Exception as e:
        import traceback
        print(f"❌ Error during evaluation: {str(e)}")
        print(traceback.format_exc())
        print("Check API keys, internet connection, and model compatibility.")

    print("\n🔍 AVAILABLE EMBEDDING MODELS:")
    print("="*80)
    print("\n" + "="*80)
    print("💡 USAGE TIPS:")
    print("- Add multiple Google API keys to the google_api_keys list for rotation.")
    print("- API keys will rotate every 5 evaluations to distribute load.")
    print("- Try 'vietnamese_specific' or 'labse' embedding models for Vietnamese.")
    print("- Check generated CSV, JSON, and TXT files for detailed results.")
    print("- The 'api_key_used' column in CSV shows which API key was used for each evaluation.")
    print("="*80)