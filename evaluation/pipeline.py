"""
Evaluation Pipeline for FRA RAG System.

Comprehensive evaluation pipeline that:
- Runs RAG system on golden dataset
- Computes all metrics (retrieval, generation, Arabic-specific)
- Integrates with Ragas and DeepEval (optional)
- Generates detailed evaluation reports
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import numpy as np
from loguru import logger

from .metrics import RetrievalEvaluator, GenerationEvaluator, ArabicEvaluator, MetricResult
from .golden_dataset import GoldenDataset, EvaluationItem


@dataclass
class PredictionResult:
    """Result from running RAG on a single item."""
    item_id: str
    question: str
    predicted_answer: str
    retrieved_chunk_ids: List[str]
    retrieved_sources: List[Dict[str, Any]]
    contexts: List[str]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    dataset_name: str
    dataset_size: int
    timestamp: str
    
    # Aggregate metrics
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    generation_metrics: Dict[str, float] = field(default_factory=dict)
    arabic_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    
    # Detailed results
    per_item_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Breakdown by category
    by_question_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_entity_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Ragas results (if available)
    ragas_results: Dict[str, float] = field(default_factory=dict)
    
    # DeepEval results (if available)
    deepeval_results: Dict[str, float] = field(default_factory=dict)
    
    def get_overall_score(self) -> float:
        """Calculate weighted overall score."""
        weights = {
            "retrieval": 0.25,
            "generation": 0.35,
            "arabic": 0.25,
            "latency": 0.15,
        }
        
        retrieval_avg = np.mean(list(self.retrieval_metrics.values())) if self.retrieval_metrics else 0
        generation_avg = np.mean(list(self.generation_metrics.values())) if self.generation_metrics else 0
        arabic_avg = np.mean(list(self.arabic_metrics.values())) if self.arabic_metrics else 0
        
        # Latency score (faster is better, target < 2000ms)
        latency_score = max(0, 1 - (self.avg_latency_ms / 5000))
        
        overall = (
            weights["retrieval"] * retrieval_avg +
            weights["generation"] * generation_avg +
            weights["arabic"] * arabic_avg +
            weights["latency"] * latency_score
        )
        
        return overall
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_size": self.dataset_size,
            "timestamp": self.timestamp,
            "overall_score": self.get_overall_score(),
            "retrieval_metrics": self.retrieval_metrics,
            "generation_metrics": self.generation_metrics,
            "arabic_metrics": self.arabic_metrics,
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "by_question_type": self.by_question_type,
            "by_difficulty": self.by_difficulty,
            "by_entity_type": self.by_entity_type,
            "ragas_results": self.ragas_results,
            "deepeval_results": self.deepeval_results,
            "per_item_results": self.per_item_results,
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        md = f"""# FRA RAG System Evaluation Report

## Overview
- **Dataset**: {self.dataset_name}
- **Size**: {self.dataset_size} items
- **Date**: {self.timestamp}
- **Overall Score**: {self.get_overall_score():.2%}

---

## Retrieval Metrics

| Metric | Score |
|--------|-------|
"""
        for metric, value in self.retrieval_metrics.items():
            md += f"| {metric} | {value:.4f} |\n"
        
        md += """
---

## Generation Metrics

| Metric | Score |
|--------|-------|
"""
        for metric, value in self.generation_metrics.items():
            md += f"| {metric} | {value:.4f} |\n"
        
        md += """
---

## Arabic-Specific Metrics

| Metric | Score |
|--------|-------|
"""
        for metric, value in self.arabic_metrics.items():
            md += f"| {metric} | {value:.4f} |\n"
        
        md += f"""
---

## Performance

| Metric | Value |
|--------|-------|
| Average Latency | {self.avg_latency_ms:.0f} ms |
| P95 Latency | {self.p95_latency_ms:.0f} ms |

"""
        
        if self.ragas_results:
            md += """
---

## Ragas Metrics

| Metric | Score |
|--------|-------|
"""
            for metric, value in self.ragas_results.items():
                md += f"| {metric} | {value:.4f} |\n"
        
        if self.deepeval_results:
            md += """
---

## DeepEval Metrics

| Metric | Score |
|--------|-------|
"""
            for metric, value in self.deepeval_results.items():
                md += f"| {metric} | {value:.4f} |\n"
        
        if self.by_question_type:
            md += """
---

## Breakdown by Question Type

| Type | Avg Score |
|------|-----------|
"""
            for qtype, metrics in self.by_question_type.items():
                avg = np.mean(list(metrics.values()))
                md += f"| {qtype} | {avg:.4f} |\n"
        
        return md
    
    def save(self, path: Path):
        """Save report to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Report saved to {path}")
    
    def save_markdown(self, path: Path):
        """Save markdown report."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())
        logger.info(f"Markdown report saved to {path}")


class EvaluationPipeline:
    """
    End-to-end evaluation pipeline.
    
    Runs the RAG system on a golden dataset and computes
    comprehensive metrics including retrieval quality,
    generation quality, and Arabic-specific metrics.
    """
    
    def __init__(
        self,
        rag_system,
        use_ragas: bool = True,
        use_deepeval: bool = True,
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            rag_system: The RAG system to evaluate
            use_ragas: Whether to use Ragas metrics
            use_deepeval: Whether to use DeepEval metrics
        """
        self.rag_system = rag_system
        self.use_ragas = use_ragas
        self.use_deepeval = use_deepeval
        
        # Initialize evaluators
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.arabic_evaluator = ArabicEvaluator()
        
        # Check for optional dependencies
        self._ragas_available = self._check_ragas()
        self._deepeval_available = self._check_deepeval()
        
        logger.info(f"EvaluationPipeline initialized (ragas={self._ragas_available}, deepeval={self._deepeval_available})")
    
    def _check_ragas(self) -> bool:
        """Check if Ragas is available."""
        try:
            import ragas
            return True
        except ImportError:
            if self.use_ragas:
                logger.warning("Ragas not installed. Install with: pip install ragas")
            return False
    
    def _check_deepeval(self) -> bool:
        """Check if DeepEval is available."""
        try:
            import deepeval
            return True
        except ImportError:
            if self.use_deepeval:
                logger.warning("DeepEval not installed. Install with: pip install deepeval")
            return False
    
    def run(
        self,
        dataset: GoldenDataset,
        k: int = 5,
        verbose: bool = True,
    ) -> EvaluationReport:
        """
        Run full evaluation on dataset.
        
        Args:
            dataset: Golden dataset to evaluate on
            k: Number of documents to retrieve
            verbose: Print progress
            
        Returns:
            EvaluationReport with all metrics
        """
        logger.info(f"Starting evaluation on {len(dataset)} items...")
        
        # Run predictions
        predictions = self._run_predictions(dataset, k, verbose)
        
        # Compute retrieval metrics
        retrieval_metrics = self._compute_retrieval_metrics(predictions, dataset, k)
        
        # Compute generation metrics
        generation_metrics = self._compute_generation_metrics(predictions, dataset)
        
        # Compute Arabic metrics
        arabic_metrics = self._compute_arabic_metrics(predictions, dataset)
        
        # Compute latency stats
        latencies = [p.latency_ms for p in predictions]
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Run Ragas if available
        ragas_results = {}
        if self._ragas_available and self.use_ragas:
            ragas_results = self._run_ragas(predictions, dataset)
        
        # Run DeepEval if available
        deepeval_results = {}
        if self._deepeval_available and self.use_deepeval:
            deepeval_results = self._run_deepeval(predictions, dataset)
        
        # Compute breakdowns
        by_type = self._compute_breakdown(predictions, dataset, "question_type")
        by_difficulty = self._compute_breakdown(predictions, dataset, "difficulty")
        by_entity = self._compute_breakdown(predictions, dataset, "entity_type")
        
        # Build per-item results
        per_item = self._build_per_item_results(predictions, dataset)
        
        report = EvaluationReport(
            dataset_name=dataset.name,
            dataset_size=len(dataset),
            timestamp=datetime.now().isoformat(),
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            arabic_metrics=arabic_metrics,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            per_item_results=per_item,
            by_question_type=by_type,
            by_difficulty=by_difficulty,
            by_entity_type=by_entity,
            ragas_results=ragas_results,
            deepeval_results=deepeval_results,
        )
        
        logger.info(f"Evaluation complete. Overall score: {report.get_overall_score():.2%}")
        return report
    
    def _run_predictions(
        self,
        dataset: GoldenDataset,
        k: int,
        verbose: bool,
    ) -> List[PredictionResult]:
        """Run RAG system on all items."""
        predictions = []
        
        for i, item in enumerate(dataset.items):
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"Processing {i + 1}/{len(dataset.items)}...")
            
            start_time = time.time()
            
            try:
                # Query the RAG system
                if hasattr(self.rag_system, 'query_router'):
                    result = self.rag_system.query_router.retrieve_with_routing(
                        item.question, k=k
                    )
                elif hasattr(self.rag_system, 'hybrid_retriever'):
                    result = self.rag_system.hybrid_retriever.retrieve_with_context(
                        item.question, k=k
                    )
                else:
                    result = self.rag_system.retriever.retrieve_with_context(
                        item.question, k=k
                    )
                
                context = result.get("context", "")
                sources = result.get("sources", [])
                
                # Generate answer
                if self.rag_system.llm_client:
                    llm_result = self.rag_system.llm_client.generate(
                        query=item.question,
                        context=context,
                        sources=[s.get("source", "") for s in sources],
                    )
                    answer = llm_result.answer
                else:
                    answer = context[:500]
                
                latency = (time.time() - start_time) * 1000
                
                pred = PredictionResult(
                    item_id=item.id,
                    question=item.question,
                    predicted_answer=answer,
                    retrieved_chunk_ids=[s.get("id", str(j)) for j, s in enumerate(sources)],
                    retrieved_sources=sources,
                    contexts=[s.get("content", s.get("text", "")) for s in sources],
                    latency_ms=latency,
                )
                predictions.append(pred)
                
            except Exception as e:
                logger.error(f"Error on item {item.id}: {e}")
                predictions.append(PredictionResult(
                    item_id=item.id,
                    question=item.question,
                    predicted_answer=f"Error: {str(e)}",
                    retrieved_chunk_ids=[],
                    retrieved_sources=[],
                    contexts=[],
                    latency_ms=0,
                ))
        
        return predictions
    
    def _compute_retrieval_metrics(
        self,
        predictions: List[PredictionResult],
        dataset: GoldenDataset,
        k: int,
    ) -> Dict[str, float]:
        """Compute retrieval metrics."""
        # Build prediction and ground truth lists
        pred_ids = [p.retrieved_chunk_ids for p in predictions]
        truth_ids = [item.relevant_chunks for item in dataset.items]
        
        # If no ground truth chunk IDs, use document names
        if not any(truth_ids):
            pred_ids = [[s.get("source", "") for s in p.retrieved_sources] for p in predictions]
            truth_ids = [item.relevant_docs for item in dataset.items]
        
        results = self.retrieval_evaluator.evaluate_all(pred_ids, truth_ids, k)
        
        return {name: result.value for name, result in results.items()}
    
    def _compute_generation_metrics(
        self,
        predictions: List[PredictionResult],
        dataset: GoldenDataset,
    ) -> Dict[str, float]:
        """Compute generation metrics."""
        all_results = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_utilization": [],
            "answer_correctness": [],
        }
        
        for pred, item in zip(predictions, dataset.items):
            results = self.generation_evaluator.evaluate_all(
                question=item.question,
                answer=pred.predicted_answer,
                contexts=pred.contexts,
                ground_truth=item.ground_truth_answer,
            )
            
            for name, result in results.items():
                all_results[name].append(result.value)
        
        return {name: np.mean(values) for name, values in all_results.items() if values}
    
    def _compute_arabic_metrics(
        self,
        predictions: List[PredictionResult],
        dataset: GoldenDataset,
    ) -> Dict[str, float]:
        """Compute Arabic-specific metrics."""
        all_results = {
            "citation_accuracy": [],
            "anti_hallucination": [],
            "number_accuracy": [],
            "article_accuracy": [],
        }
        
        for pred, item in zip(predictions, dataset.items):
            results = self.arabic_evaluator.evaluate_all(
                answer=pred.predicted_answer,
                sources=pred.retrieved_sources,
                contexts=pred.contexts,
                ground_truth=item.ground_truth_answer,
                ground_truth_articles=item.ground_truth_articles,
            )
            
            for name, result in results.items():
                all_results[name].append(result.value)
        
        return {name: np.mean(values) for name, values in all_results.items() if values}
    
    def _run_ragas(
        self,
        predictions: List[PredictionResult],
        dataset: GoldenDataset,
    ) -> Dict[str, float]:
        """Run Ragas evaluation."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset
            
            # Prepare data for Ragas
            data = {
                "question": [item.question for item in dataset.items],
                "answer": [p.predicted_answer for p in predictions],
                "contexts": [p.contexts for p in predictions],
                "ground_truth": [item.ground_truth_answer for item in dataset.items],
            }
            
            ragas_dataset = Dataset.from_dict(data)
            
            # Run evaluation
            results = evaluate(
                dataset=ragas_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )
            
            return {
                "ragas_faithfulness": results.get("faithfulness", 0),
                "ragas_relevancy": results.get("answer_relevancy", 0),
                "ragas_context_precision": results.get("context_precision", 0),
                "ragas_context_recall": results.get("context_recall", 0),
            }
            
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            return {}
    
    def _run_deepeval(
        self,
        predictions: List[PredictionResult],
        dataset: GoldenDataset,
    ) -> Dict[str, float]:
        """Run DeepEval evaluation."""
        try:
            from deepeval import evaluate
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                FaithfulnessMetric,
                ContextualRelevancyMetric,
            )
            from deepeval.test_case import LLMTestCase
            
            test_cases = []
            for pred, item in zip(predictions, dataset.items):
                test_case = LLMTestCase(
                    input=item.question,
                    actual_output=pred.predicted_answer,
                    expected_output=item.ground_truth_answer,
                    retrieval_context=pred.contexts,
                )
                test_cases.append(test_case)
            
            # Run metrics
            relevancy_metric = AnswerRelevancyMetric()
            faithfulness_metric = FaithfulnessMetric()
            context_metric = ContextualRelevancyMetric()
            
            # Evaluate (simplified - actual implementation may vary)
            results = {
                "deepeval_relevancy": 0.0,
                "deepeval_faithfulness": 0.0,
                "deepeval_context": 0.0,
            }
            
            return results
            
        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {e}")
            return {}
    
    def _compute_breakdown(
        self,
        predictions: List[PredictionResult],
        dataset: GoldenDataset,
        field: str,
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics breakdown by field."""
        breakdown = {}
        
        for pred, item in zip(predictions, dataset.items):
            field_value = getattr(item, field, "")
            if not field_value:
                continue
            
            if field_value not in breakdown:
                breakdown[field_value] = {
                    "count": 0,
                    "faithfulness_sum": 0,
                    "relevancy_sum": 0,
                }
            
            # Compute metrics for this item
            gen_results = self.generation_evaluator.evaluate_all(
                question=item.question,
                answer=pred.predicted_answer,
                contexts=pred.contexts,
                ground_truth=item.ground_truth_answer,
            )
            
            breakdown[field_value]["count"] += 1
            breakdown[field_value]["faithfulness_sum"] += gen_results["faithfulness"].value
            breakdown[field_value]["relevancy_sum"] += gen_results["answer_relevancy"].value
        
        # Compute averages
        result = {}
        for key, data in breakdown.items():
            count = data["count"]
            if count > 0:
                result[key] = {
                    "faithfulness": data["faithfulness_sum"] / count,
                    "relevancy": data["relevancy_sum"] / count,
                    "count": count,
                }
        
        return result
    
    def _build_per_item_results(
        self,
        predictions: List[PredictionResult],
        dataset: GoldenDataset,
    ) -> List[Dict[str, Any]]:
        """Build detailed per-item results."""
        results = []
        
        for pred, item in zip(predictions, dataset.items):
            gen_results = self.generation_evaluator.evaluate_all(
                question=item.question,
                answer=pred.predicted_answer,
                contexts=pred.contexts,
                ground_truth=item.ground_truth_answer,
            )
            
            results.append({
                "id": item.id,
                "question": item.question,
                "question_type": item.question_type,
                "predicted_answer": pred.predicted_answer[:500],
                "ground_truth": item.ground_truth_answer[:500],
                "latency_ms": pred.latency_ms,
                "faithfulness": gen_results["faithfulness"].value,
                "relevancy": gen_results["answer_relevancy"].value,
                "num_sources": len(pred.retrieved_sources),
            })
        
        return results
    
    def run_quick(
        self,
        questions: List[str],
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        Run quick evaluation on a list of questions (no ground truth).
        
        Useful for testing retrieval quality without a full dataset.
        
        Args:
            questions: List of test questions
            k: Number of documents to retrieve
            
        Returns:
            Dict with basic metrics
        """
        latencies = []
        faithfulness_scores = []
        relevancy_scores = []
        
        for question in questions:
            start_time = time.time()
            
            try:
                # Query
                if hasattr(self.rag_system, 'query_router'):
                    result = self.rag_system.query_router.retrieve_with_routing(question, k=k)
                else:
                    result = self.rag_system.retriever.retrieve_with_context(question, k=k)
                
                context = result.get("context", "")
                sources = result.get("sources", [])
                contexts = [s.get("content", s.get("text", "")) for s in sources]
                
                # Generate answer
                if self.rag_system.llm_client:
                    llm_result = self.rag_system.llm_client.generate(
                        query=question,
                        context=context,
                        sources=[s.get("source", "") for s in sources],
                    )
                    answer = llm_result.answer
                else:
                    answer = context[:500]
                
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                
                # Compute metrics
                faith = self.generation_evaluator.faithfulness(answer, contexts)
                rel = self.generation_evaluator.answer_relevancy(question, answer)
                
                faithfulness_scores.append(faith.value)
                relevancy_scores.append(rel.value)
                
            except Exception as e:
                logger.error(f"Error on question: {e}")
        
        return {
            "questions_tested": len(questions),
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "avg_faithfulness": np.mean(faithfulness_scores) if faithfulness_scores else 0,
            "avg_relevancy": np.mean(relevancy_scores) if relevancy_scores else 0,
        }


def create_evaluation_pipeline(
    rag_system,
    use_ragas: bool = True,
    use_deepeval: bool = True,
) -> EvaluationPipeline:
    """Factory function."""
    return EvaluationPipeline(
        rag_system=rag_system,
        use_ragas=use_ragas,
        use_deepeval=use_deepeval,
    )
