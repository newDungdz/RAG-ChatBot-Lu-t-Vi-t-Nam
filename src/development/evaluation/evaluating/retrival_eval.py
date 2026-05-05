import json
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple
import time
import logging
from collections import defaultdict
import os

class RAGEvaluator:
    def __init__(self, ground_truth_data: str, pre_retrieved_data: str, output_dir: str = "evaluation_results"):
        """
        Initialize the RAG evaluation system
        
        Parameters:
        -----------
        ground_truth_data : str
            Path to JSON file containing ground truth queries with retrieved_laws
        pre_retrieved_data : str
            Path to JSON file containing pre-retrieved results from your pipeline
        output_dir : str
            Directory to save evaluation results
        """
        self.ground_truth_data = ground_truth_data
        self.pre_retrieved_data = pre_retrieved_data
        self.output_dir = output_dir
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        self.intermediate_dir = os.path.join(output_dir, "intermediate_results")
        os.makedirs(self.intermediate_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{output_dir}/evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_ground_truth_queries(self) -> List[Dict]:
        """
        Load ground truth queries from JSON file
        
        Expected format:
        {
            "id": 617,
            "retrieved_laws": [...],
            "retrieved_laws": [...]
        }
        
        Returns:
        --------
        List[Dict] : List of queries with at least 3 relevant laws
        """
        try:
            data = self.ground_truth_data
            
            if not isinstance(data, list):
                data = [data]
            
            # Filter queries with at least 3 relevant laws
            filtered_queries = []
            for item in data:
                if "id" not in item or "retrieved_laws" not in item:
                    self.logger.warning(f"Skipping item missing required fields: {item}")
                    continue
                
                retrieved_laws = item["retrieved_laws"]
                if len(retrieved_laws) < 3:
                    self.logger.info(f"Skipping query {item['id']} - only {len(retrieved_laws)} relevant laws (minimum 3 required)")
                    continue
                
                filtered_queries.append({
                    "id": item["id"],
                    "retrieved_laws": retrieved_laws,
                    "processed": False
                })
            
            self.logger.info(f"Loaded {len(filtered_queries)} queries with at least 3 relevant laws from {len(data)} total queries")
            return filtered_queries
            
        except Exception as e:
            self.logger.error(f"Error loading ground truth queries: {str(e)}")
            raise
    
    def load_pre_retrieved_results(self) -> Dict[int, List[str]]:
        """
        Load pre-retrieved results from JSON file
        
        Expected format:
        {
            "id": 0,
            "question": "...",
            "results": [
                {
                    "chunk_id": "100/2015/QH13_d_332",
                    ...
                },
                ...
            ]
        }
        
        Returns:
        --------
        Dict[int, List[str]] : Mapping from query ID to list of retrieved chunk IDs
        """
        try:
            data = self.pre_retrieved_data
            
            if not isinstance(data, list):
                data = [data]
            
            retrieved_results = {}
            for item in data:
                if "results" in item:
                    result_field = "results"
                elif "retrival_results" in item:
                    result_field = "retrival_results"
                else:
                    self.logger.warning(f"Skipping pre-retrieved item missing required fields")
                    continue
                query_id = item['id']
                chunk_ids = []
                
                for result in item[result_field]:
                    if "chunk_id" in result:
                        chunk_ids.append(result["chunk_id"])
                
                retrieved_results[query_id] = chunk_ids
            
            self.logger.info(f"Loaded pre-retrieved results for {len(retrieved_results)} queries")
            return retrieved_results
            
        except Exception as e:
            self.logger.error(f"Error loading pre-retrieved results: {str(e)}")
            raise
    
    def calculate_k_precision(self, retrieved_chunk_ids: List[str], 
                            correct_chunk_ids: List[str], 
                            k: int = None) -> Dict[str, float]:
        """
        Calculate K-Precision for retrieval evaluation
        
        Parameters:
        -----------
        retrieved_chunk_ids : List[str]
            List of retrieved chunk IDs (in order of relevance)
        correct_chunk_ids : List[str]
            List of correct/relevant chunk IDs
        k : int, optional
            Number of top results to consider. If None, considers all retrieved chunks
            
        Returns:
        --------
        Dict[str, float] : Precision metrics
        """
        if not retrieved_chunk_ids:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "total_retrieved": 0,
                "total_relevant": len(correct_chunk_ids),
                "relevant_retrieved": 0
            }
        
        # Limit to top-k if specified
        if k is not None:
            retrieved_chunk_ids = retrieved_chunk_ids[:k]
        
        # Convert to sets for intersection calculation
        retrieved_set = set(retrieved_chunk_ids)
        correct_set = set(correct_chunk_ids)
        
        # Calculate metrics
        relevant_retrieved = len(retrieved_set.intersection(correct_set))
        total_retrieved = len(retrieved_chunk_ids)
        total_relevant = len(correct_chunk_ids)
        
        # Precision: What fraction of retrieved chunks are relevant?
        precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0
        
        # Recall: What fraction of relevant chunks were retrieved?
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_retrieved": total_retrieved,
            "total_relevant": total_relevant,
            "relevant_retrieved": relevant_retrieved
        }
    
    def evaluate_single_query(self, ground_truth_query: Dict, 
                            retrieved_results: Dict[int, List[str]]) -> Dict:
        """
        Evaluate a single query
        
        Parameters:
        -----------
        ground_truth_query : Dict
            Ground truth query data
        retrieved_results : Dict[int, List[str]]
            Pre-retrieved results mapping
            
        Returns:
        --------
        Dict : Evaluation results for the query
        """
        query_id = ground_truth_query["id"]
        relevant_chunk_ids = ground_truth_query["retrieved_laws"]
        
        self.logger.info(f"Evaluating query {query_id}")
        
        start_time = time.time()
        
        try:
            # Get retrieved chunk IDs for this query
            retrieved_chunk_ids = retrieved_results.get(query_id, [])
            if not retrieved_chunk_ids:
                self.logger.warning(f"No retrieved results found for query {query_id}")
            
            evaluation_time = time.time() - start_time
            print(len(retrieved_chunk_ids))
            # Calculate precision metrics for different k values
            precision_metrics = {}
            k_values = [1, 3, 5, 10, 20, len(retrieved_chunk_ids)] if retrieved_chunk_ids else [0]
            
            for k in k_values:
                if k == 0:
                    continue
                precision_metrics[f"precision@{k}"] = self.calculate_k_precision(
                    retrieved_chunk_ids, relevant_chunk_ids, k
                )
            
            # Overall precision (all retrieved chunks)
            overall_precision = self.calculate_k_precision(
                retrieved_chunk_ids, relevant_chunk_ids
            )
            
            # Mark query as processed
            ground_truth_query["processed"] = True
            
            return {
                "query_id": query_id,
                "relevant_chunk_ids": relevant_chunk_ids,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "evaluation_time": evaluation_time,
                "precision_at_k": precision_metrics,
                "overall_precision": overall_precision,
                "success": True,
                "error": None
            }
                
        except Exception as e:
            self.logger.error(f"Error evaluating query {query_id}: {str(e)}")
            return {
                "query_id": query_id,
                "relevant_chunk_ids": relevant_chunk_ids,
                "retrieved_chunk_ids": [],
                "evaluation_time": time.time() - start_time,
                "precision_at_k": {},
                "overall_precision": self.calculate_k_precision([], relevant_chunk_ids),
                "success": False,
                "error": str(e)
            }
    
    def run_evaluation(self, sample_size: int = None) -> Dict:
        """
        Run evaluation on all queries
        
        Parameters:
        -----------
        sample_size : int, optional
            Number of queries to evaluate (for testing). If None, evaluates all
            
        Returns:
        --------
        Dict : Complete evaluation results
        """
        self.logger.info("Starting RAG retrieval evaluation...")
        
        # Load data
        ground_truth_queries = self.load_ground_truth_queries()
        retrieved_results = self.load_pre_retrieved_results()
        
        # Sample if specified
        if sample_size and sample_size < len(ground_truth_queries):
            ground_truth_queries = ground_truth_queries[:sample_size]
            self.logger.info(f"Evaluating sample of {sample_size} queries")
        
        # Filter queries that have pre-retrieved results
        valid_queries = []
        for query in ground_truth_queries:
            if query["id"] in retrieved_results:
                valid_queries.append(query)
            else:
                self.logger.warning(f"No pre-retrieved results for query {query['id']}, skipping")
        
        self.logger.info(f"Evaluating {len(valid_queries)} queries with both ground truth and pre-retrieved results")
        
        # Evaluate each query
        results = []
        total_queries = len(valid_queries)
        
        for i, query_data in enumerate(valid_queries):
            self.logger.info(f"Progress: {i+1}/{total_queries}")
            result = self.evaluate_single_query(query_data, retrieved_results)
            results.append(result)
            
            # Save intermediate results every 10 queries
            if (i + 1) % 10 == 0:
                self._save_intermediate_results(results, i + 1)
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        # Prepare final results
        final_results = {
            "evaluation_config": {
                "total_ground_truth_queries": len(ground_truth_queries),
                "total_pre_retrieved_queries": len(retrieved_results),
                "evaluated_queries": total_queries,
                "sample_size": sample_size,
                "k_values": [1, 3, 5, 10, 20]
            },
            "individual_results": results,
            "aggregate_metrics": aggregate_metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save final results
        self._save_final_results(final_results)
        
        return final_results
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all queries"""
        successful_results = [r for r in results if r["success"]]
        
        if not successful_results:
            return {"error": "No successful evaluations"}
        
        # Calculate average metrics
        avg_metrics = {}
        k_values = [1, 3, 5, 10, 20]
        
        for k in k_values:
            precisions = []
            recalls = []
            f1_scores = []
            
            for result in successful_results:
                precision_data = result["precision_at_k"].get(f"precision@{k}")
                if precision_data:
                    precisions.append(precision_data["precision"])
                    recalls.append(precision_data["recall"])
                    f1_scores.append(precision_data["f1"])
            
            if precisions:
                avg_metrics[f"avg_precision@{k}"] = np.mean(precisions)
                avg_metrics[f"avg_recall@{k}"] = np.mean(recalls)
                avg_metrics[f"avg_f1@{k}"] = np.mean(f1_scores)
                avg_metrics[f"std_precision@{k}"] = np.std(precisions)
                avg_metrics[f"std_recall@{k}"] = np.std(recalls)
                avg_metrics[f"std_f1@{k}"] = np.std(f1_scores)
                avg_metrics[f"median_precision@{k}"] = np.median(precisions)
                avg_metrics[f"median_recall@{k}"] = np.median(recalls)
                avg_metrics[f"median_f1@{k}"] = np.median(f1_scores)
        
        # Overall metrics
        overall_precisions = [r["overall_precision"]["precision"] for r in successful_results]
        overall_recalls = [r["overall_precision"]["recall"] for r in successful_results]
        overall_f1s = [r["overall_precision"]["f1"] for r in successful_results]
        
        avg_metrics.update({
            "avg_overall_precision": np.mean(overall_precisions),
            "avg_overall_recall": np.mean(overall_recalls),
            "avg_overall_f1": np.mean(overall_f1s),
            "std_overall_precision": np.std(overall_precisions),
            "std_overall_recall": np.std(overall_recalls),
            "std_overall_f1": np.std(overall_f1s),
            "median_overall_precision": np.median(overall_precisions),
            "median_overall_recall": np.median(overall_recalls),
            "median_overall_f1": np.median(overall_f1s),
            "avg_evaluation_time": np.mean([r["evaluation_time"] for r in successful_results]),
            "success_rate": len(successful_results) / len(results),
            "total_evaluated": len(results),
            "successful_evaluations": len(successful_results)
        })
        
        return avg_metrics
    
    def _save_intermediate_results(self, results: List[Dict], count: int):
        """Save intermediate results to the intermediate folder"""
        filename = os.path.join(self.intermediate_dir, f"intermediate_results_{count}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved intermediate results for {count} queries to {filename}")
    
    def _save_final_results(self, results: Dict):
        """Save final results with better organization"""
        # Save detailed results
        detailed_filename = os.path.join(self.output_dir, "detailed_evaluation_results.json")
        with open(detailed_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save summary report
        summary_filename = os.path.join(self.output_dir, "evaluation_summary.txt")
        with open(summary_filename, 'w', encoding='utf-8') as f:
            self._write_summary_report(f, results)
        
        # Save individual query metrics CSV
        individual_csv_filename = os.path.join(self.output_dir, "individual_query_metrics.csv")
        self._save_individual_metrics_csv(results, individual_csv_filename)
        
        # Save detailed precision/recall metrics CSV (each row is a k_value)
        detailed_csv_filename = os.path.join(self.output_dir, "detailed_precision_recall_metrics.csv")
        self._save_detailed_metrics_csv(results, detailed_csv_filename)
        
        # Save aggregate metrics CSV
        aggregate_csv_filename = os.path.join(self.output_dir, "aggregate_metrics.csv")
        self._save_aggregate_metrics_csv(results, aggregate_csv_filename)
        
        self.logger.info(f"All results saved to {self.output_dir}/")
        self.logger.info(f"Intermediate results saved to {self.intermediate_dir}/")
    
    def _write_summary_report(self, file, results: Dict):
        """Write summary report"""
        config = results["evaluation_config"]
        metrics = results["aggregate_metrics"]
        
        file.write("RAG RETRIEVAL EVALUATION SUMMARY\n")
        file.write("=" * 50 + "\n\n")
        
        file.write("CONFIGURATION:\n")
        file.write(f"  Ground Truth Queries: {config['total_ground_truth_queries']}\n")
        file.write(f"  Pre-retrieved Queries: {config['total_pre_retrieved_queries']}\n")
        file.write(f"  Evaluated Queries: {config['evaluated_queries']}\n")
        file.write(f"  Sample Size: {config.get('sample_size', 'All')}\n")
        file.write(f"  K Values: {config['k_values']}\n\n")
        
        file.write("AGGREGATE METRICS:\n")
        file.write(f"  Success Rate: {metrics.get('success_rate', 0):.4f}\n")
        file.write(f"  Avg Evaluation Time: {metrics.get('avg_evaluation_time', 0):.4f}s\n\n")
        
        file.write("PRECISION METRICS (Mean ± Std):\n")
        for k in config['k_values']:
            precision = metrics.get(f'avg_precision@{k}', 0)
            recall = metrics.get(f'avg_recall@{k}', 0)
            f1 = metrics.get(f'avg_f1@{k}', 0)
            std_precision = metrics.get(f'std_precision@{k}', 0)
            std_recall = metrics.get(f'std_recall@{k}', 0)
            std_f1 = metrics.get(f'std_f1@{k}', 0)
            file.write(f"  P@{k}: {precision:.4f} (±{std_precision:.4f}), R@{k}: {recall:.4f} (±{std_recall:.4f}), F1@{k}: {f1:.4f} (±{std_f1:.4f})\n")
        
        file.write(f"\nOVERALL METRICS (Mean ± Std):\n")
        file.write(f"  Precision: {metrics.get('avg_overall_precision', 0):.4f} (±{metrics.get('std_overall_precision', 0):.4f})\n")
        file.write(f"  Recall: {metrics.get('avg_overall_recall', 0):.4f} (±{metrics.get('std_overall_recall', 0):.4f})\n")
        file.write(f"  F1-Score: {metrics.get('avg_overall_f1', 0):.4f} (±{metrics.get('std_overall_f1', 0):.4f})\n")
        
        file.write(f"\nMEDIAN METRICS:\n")
        for k in config['k_values']:
            med_precision = metrics.get(f'median_precision@{k}', 0)
            med_recall = metrics.get(f'median_recall@{k}', 0)
            med_f1 = metrics.get(f'median_f1@{k}', 0)
            file.write(f"  Median P@{k}: {med_precision:.4f}, R@{k}: {med_recall:.4f}, F1@{k}: {med_f1:.4f}\n")
        
        file.write(f"\nOVERALL MEDIAN METRICS:\n")
        file.write(f"  Median Precision: {metrics.get('median_overall_precision', 0):.4f}\n")
        file.write(f"  Median Recall: {metrics.get('median_overall_recall', 0):.4f}\n")
        file.write(f"  Median F1-Score: {metrics.get('median_overall_f1', 0):.4f}\n")
    
    def _save_individual_metrics_csv(self, results: Dict, filename: str):
        """Save individual query metrics to CSV"""
        rows = []
        for result in results["individual_results"]:
            if not result["success"]:
                continue
            
            # Get overall precision metrics
            overall_metrics = result["overall_precision"]
            
            row = {
                "query_id": result["query_id"],
                "evaluation_time": result["evaluation_time"],
                "total_retrieved": overall_metrics["total_retrieved"],
                "total_relevant": overall_metrics["total_relevant"],
                "relevant_retrieved": overall_metrics["relevant_retrieved"],
                "overall_precision": overall_metrics["precision"],
                "overall_recall": overall_metrics["recall"],
                "overall_f1": overall_metrics["f1"]
            }
            
            # Add precision@k metrics for each k value
            k_values = [1, 3, 5, 10, 20]
            for k in k_values:
                precision_at_k = result["precision_at_k"].get(f"precision@{k}", {})
                row[f"precision@{k}"] = precision_at_k.get("precision", 0)
                row[f"recall@{k}"] = precision_at_k.get("recall", 0)
                row[f"f1@{k}"] = precision_at_k.get("f1", 0)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False, encoding='utf-8')
        self.logger.info(f"Saved individual query metrics to {filename}")
    
    def _save_detailed_metrics_csv(self, results: Dict, filename: str):
        """Save detailed precision/recall metrics CSV where each row is a k_value"""
        metrics = results["aggregate_metrics"]
        k_values = results["evaluation_config"]["k_values"]
        
        rows = []
        for k in k_values:
            row = {
                "k_value": k,
                "avg_precision": metrics.get(f'avg_precision@{k}', 0),
                "avg_recall": metrics.get(f'avg_recall@{k}', 0),
                "avg_f1": metrics.get(f'avg_f1@{k}', 0),
                "std_precision": metrics.get(f'std_precision@{k}', 0),
                "std_recall": metrics.get(f'std_recall@{k}', 0),
                "std_f1": metrics.get(f'std_f1@{k}', 0),
                "median_precision": metrics.get(f'median_precision@{k}', 0),
                "median_recall": metrics.get(f'median_recall@{k}', 0),
                "median_f1": metrics.get(f'median_f1@{k}', 0)
            }
            rows.append(row)
        
        # Add overall metrics row
        overall_row = {
            "k_value": "overall",
            "avg_precision": metrics.get('avg_overall_precision', 0),
            "avg_recall": metrics.get('avg_overall_recall', 0),
            "avg_f1": metrics.get('avg_overall_f1', 0),
            "std_precision": metrics.get('std_overall_precision', 0),
            "std_recall": metrics.get('std_overall_recall', 0),
            "std_f1": metrics.get('std_overall_f1', 0),
            "median_precision": metrics.get('median_overall_precision', 0),
            "median_recall": metrics.get('median_overall_recall', 0),
            "median_f1": metrics.get('median_overall_f1', 0)
        }
        rows.append(overall_row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False, encoding='utf-8')
        self.logger.info(f"Saved detailed precision/recall metrics to {filename}")
    
    def _save_aggregate_metrics_csv(self, results: Dict, filename: str):
        """Save aggregate metrics to CSV"""
        metrics = results["aggregate_metrics"]
        config = results["evaluation_config"]
        
        # Create a summary row
        summary_data = {
            "timestamp": results["timestamp"],
            "total_ground_truth_queries": config["total_ground_truth_queries"],
            "total_pre_retrieved_queries": config["total_pre_retrieved_queries"],
            "evaluated_queries": config["evaluated_queries"],
            "sample_size": config.get("sample_size", "All"),
            "success_rate": metrics.get("success_rate", 0),
            "avg_evaluation_time": metrics.get("avg_evaluation_time", 0),
            "successful_evaluations": metrics.get("successful_evaluations", 0),
            "total_evaluated": metrics.get("total_evaluated", 0)
        }
        
        df = pd.DataFrame([summary_data])
        df.to_csv(filename, index=False, encoding='utf-8')
        self.logger.info(f"Saved aggregate metrics to {filename}")

def read_json_file(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data
def save_to_json(data, output_path):
    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
# Example usage function
def run_evaluation_example():
    """
    Example of how to use the evaluation system
    """
    root_evaluation_folder = "data/evalations/eval_bm25_1000"
    # Initialize evaluator
    evaluator = RAGEvaluator(
        ground_truth_data=read_json_file("data/json_data/QnA/processed_qna_data.json"),  # File with retrieved_laws
        pre_retrieved_data=read_json_file(f"{root_evaluation_folder}/pre_retrive_query.json")['queries'],  # File with your pipeline results
        output_dir=f"{root_evaluation_folder}/retrival_eval"
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(
        sample_size=None  # Set to number for testing with subset
    )
    
    print("Evaluation completed!")
    print(f"Average Precision@10: {results['aggregate_metrics'].get('avg_precision@10', 0):.4f}")
    print(f"Average Recall@10: {results['aggregate_metrics'].get('avg_recall@10', 0):.4f}")
    print(f"Average F1@10: {results['aggregate_metrics'].get('avg_f1@10', 0):.4f}")
    print(f"Evaluated {results['evaluation_config']['evaluated_queries']} queries")
    
    print(f"\nFiles saved to:")
    print(f"  Main results: {evaluator.output_dir}/")
    print(f"  Intermediate results: {evaluator.intermediate_dir}/")


if __name__ == "__main__":
    run_evaluation_example()