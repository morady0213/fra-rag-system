"""
FRA RAG System Evaluation Framework.

Comprehensive evaluation suite including:
- Retrieval metrics (Hit Rate, MRR, Precision, Recall)
- Generation metrics (Faithfulness, Relevancy, Correctness)
- Ragas integration
- DeepEval integration
- Arabic-specific metrics
"""

from .metrics import RetrievalEvaluator, GenerationEvaluator, ArabicEvaluator
from .golden_dataset import GoldenDataset, DatasetGenerator
from .pipeline import EvaluationPipeline, EvaluationReport

__all__ = [
    "RetrievalEvaluator",
    "GenerationEvaluator", 
    "ArabicEvaluator",
    "GoldenDataset",
    "DatasetGenerator",
    "EvaluationPipeline",
    "EvaluationReport",
]
