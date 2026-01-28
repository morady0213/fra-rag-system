"""
Evaluation Metrics for FRA RAG System.

Includes:
- Retrieval Metrics: Hit Rate, MRR, Precision, Recall, NDCG
- Generation Metrics: Faithfulness, Answer Relevancy, Correctness
- Arabic-specific Metrics: Citation accuracy, Article reference accuracy
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from loguru import logger


@dataclass
class MetricResult:
    """Result from a metric computation."""
    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"{self.name}: {self.value:.4f}"


class RetrievalEvaluator:
    """
    Evaluate retrieval quality using standard IR metrics.
    
    Metrics:
    - Hit Rate @ K (Recall @ K): % of queries with at least one relevant doc in top K
    - MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant doc
    - Precision @ K: % of retrieved docs that are relevant
    - Recall @ K: % of relevant docs that were retrieved
    - NDCG @ K: Normalized Discounted Cumulative Gain
    - MAP: Mean Average Precision
    """
    
    def __init__(self):
        logger.info("RetrievalEvaluator initialized")
    
    def hit_rate_at_k(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        k: int = 5,
    ) -> MetricResult:
        """
        Calculate Hit Rate @ K (also known as Recall @ K in some contexts).
        
        Measures the percentage of queries where at least one relevant
        document appears in the top K retrieved documents.
        
        Args:
            predictions: List of retrieved doc IDs for each query
            ground_truth: List of relevant doc IDs for each query
            k: Number of top documents to consider
            
        Returns:
            MetricResult with hit rate value
        """
        if not predictions or not ground_truth:
            return MetricResult("hit_rate@k", 0.0)
        
        hits = 0
        for pred, truth in zip(predictions, ground_truth):
            pred_set = set(pred[:k])
            truth_set = set(truth)
            if pred_set & truth_set:  # Intersection
                hits += 1
        
        hit_rate = hits / len(predictions)
        
        return MetricResult(
            name=f"hit_rate@{k}",
            value=hit_rate,
            details={"hits": hits, "total": len(predictions), "k": k}
        )
    
    def mrr(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
    ) -> MetricResult:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Measures the average of reciprocal ranks of the first relevant
        document. Higher is better (max 1.0).
        
        Args:
            predictions: List of retrieved doc IDs for each query
            ground_truth: List of relevant doc IDs for each query
            
        Returns:
            MetricResult with MRR value
        """
        if not predictions or not ground_truth:
            return MetricResult("mrr", 0.0)
        
        reciprocal_ranks = []
        
        for pred, truth in zip(predictions, ground_truth):
            truth_set = set(truth)
            rr = 0.0
            for rank, doc_id in enumerate(pred, 1):
                if doc_id in truth_set:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
        
        mrr_value = np.mean(reciprocal_ranks)
        
        return MetricResult(
            name="mrr",
            value=mrr_value,
            details={"reciprocal_ranks": reciprocal_ranks}
        )
    
    def precision_at_k(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        k: int = 5,
    ) -> MetricResult:
        """
        Calculate Precision @ K.
        
        Measures the percentage of retrieved documents (in top K)
        that are actually relevant.
        
        Args:
            predictions: List of retrieved doc IDs for each query
            ground_truth: List of relevant doc IDs for each query
            k: Number of top documents to consider
            
        Returns:
            MetricResult with precision value
        """
        if not predictions or not ground_truth:
            return MetricResult("precision@k", 0.0)
        
        precisions = []
        
        for pred, truth in zip(predictions, ground_truth):
            pred_k = pred[:k]
            truth_set = set(truth)
            relevant_count = sum(1 for p in pred_k if p in truth_set)
            precision = relevant_count / k if k > 0 else 0.0
            precisions.append(precision)
        
        avg_precision = np.mean(precisions)
        
        return MetricResult(
            name=f"precision@{k}",
            value=avg_precision,
            details={"precisions": precisions, "k": k}
        )
    
    def recall_at_k(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        k: int = 5,
    ) -> MetricResult:
        """
        Calculate Recall @ K.
        
        Measures the percentage of relevant documents that were
        retrieved in the top K.
        
        Args:
            predictions: List of retrieved doc IDs for each query
            ground_truth: List of relevant doc IDs for each query
            k: Number of top documents to consider
            
        Returns:
            MetricResult with recall value
        """
        if not predictions or not ground_truth:
            return MetricResult("recall@k", 0.0)
        
        recalls = []
        
        for pred, truth in zip(predictions, ground_truth):
            if not truth:
                continue
            pred_k = set(pred[:k])
            truth_set = set(truth)
            relevant_retrieved = len(pred_k & truth_set)
            recall = relevant_retrieved / len(truth_set)
            recalls.append(recall)
        
        avg_recall = np.mean(recalls) if recalls else 0.0
        
        return MetricResult(
            name=f"recall@{k}",
            value=avg_recall,
            details={"recalls": recalls, "k": k}
        )
    
    def ndcg_at_k(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        k: int = 5,
    ) -> MetricResult:
        """
        Calculate Normalized Discounted Cumulative Gain @ K.
        
        Measures ranking quality with position-based discounting.
        Higher positions contribute more to the score.
        
        Args:
            predictions: List of retrieved doc IDs for each query
            ground_truth: List of relevant doc IDs for each query
            k: Number of top documents to consider
            
        Returns:
            MetricResult with NDCG value
        """
        if not predictions or not ground_truth:
            return MetricResult("ndcg@k", 0.0)
        
        def dcg(relevances: List[int], k: int) -> float:
            """Calculate DCG."""
            relevances = relevances[:k]
            if not relevances:
                return 0.0
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
        
        ndcg_scores = []
        
        for pred, truth in zip(predictions, ground_truth):
            truth_set = set(truth)
            
            # Binary relevance
            relevances = [1 if doc_id in truth_set else 0 for doc_id in pred[:k]]
            
            # Calculate DCG
            dcg_score = dcg(relevances, k)
            
            # Calculate ideal DCG (all relevant docs at top)
            ideal_relevances = sorted(relevances, reverse=True)
            idcg_score = dcg(ideal_relevances, k)
            
            # Calculate NDCG
            ndcg = dcg_score / idcg_score if idcg_score > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        avg_ndcg = np.mean(ndcg_scores)
        
        return MetricResult(
            name=f"ndcg@{k}",
            value=avg_ndcg,
            details={"ndcg_scores": ndcg_scores, "k": k}
        )
    
    def mean_average_precision(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
    ) -> MetricResult:
        """
        Calculate Mean Average Precision (MAP).
        
        Average of precision values at each relevant document position.
        
        Args:
            predictions: List of retrieved doc IDs for each query
            ground_truth: List of relevant doc IDs for each query
            
        Returns:
            MetricResult with MAP value
        """
        if not predictions or not ground_truth:
            return MetricResult("map", 0.0)
        
        average_precisions = []
        
        for pred, truth in zip(predictions, ground_truth):
            if not truth:
                continue
            
            truth_set = set(truth)
            hits = 0
            precision_sum = 0.0
            
            for rank, doc_id in enumerate(pred, 1):
                if doc_id in truth_set:
                    hits += 1
                    precision_sum += hits / rank
            
            # AP should be precision_sum / number of relevant docs
            # Cap at 1.0 to handle edge cases
            ap = min(1.0, precision_sum / len(truth_set)) if truth_set else 0.0
            average_precisions.append(ap)
        
        # MAP should be between 0 and 1
        map_value = min(1.0, np.mean(average_precisions)) if average_precisions else 0.0
        
        return MetricResult(
            name="map",
            value=map_value,
            details={"average_precisions": average_precisions}
        )
    
    def evaluate_all(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        k: int = 5,
    ) -> Dict[str, MetricResult]:
        """Run all retrieval metrics."""
        return {
            "hit_rate": self.hit_rate_at_k(predictions, ground_truth, k),
            "mrr": self.mrr(predictions, ground_truth),
            "precision": self.precision_at_k(predictions, ground_truth, k),
            "recall": self.recall_at_k(predictions, ground_truth, k),
            "ndcg": self.ndcg_at_k(predictions, ground_truth, k),
            "map": self.mean_average_precision(predictions, ground_truth),
        }


class GenerationEvaluator:
    """
    Evaluate generation quality.
    
    Metrics:
    - Faithfulness: Does answer rely only on context? (Anti-hallucination)
    - Answer Relevancy: Does answer address the question?
    - Answer Correctness: Is answer factually correct?
    - Context Utilization: How much of context is used?
    """
    
    def __init__(self, embedding_model=None):
        """
        Initialize generation evaluator.
        
        Args:
            embedding_model: Optional embedding model for semantic similarity
        """
        self.embedding_model = embedding_model
        logger.info("GenerationEvaluator initialized")
    
    def faithfulness(
        self,
        answer: str,
        contexts: List[str],
        llm_client=None,
    ) -> MetricResult:
        """
        Measure faithfulness (anti-hallucination).
        
        Checks if all claims in the answer are supported by the context.
        
        Args:
            answer: Generated answer
            contexts: Retrieved context passages
            llm_client: Optional LLM for claim extraction
            
        Returns:
            MetricResult with faithfulness score
        """
        if not answer or not contexts:
            return MetricResult("faithfulness", 0.0)
        
        # Handle NO_CONTEXT responses - model correctly refusing to answer
        # This is GOOD behavior and should be rewarded with high faithfulness
        no_context_patterns = [
            "NO_CONTEXT",
            "لم أجد",
            "لا يوجد نص صريح",
            "لا تتوفر معلومات",
            "غير متوفر في المستندات",
            "I did not find",
            "No explicit text",
            "not found in the documents",
        ]
        
        answer_lower = answer.lower()
        for pattern in no_context_patterns:
            if pattern.lower() in answer_lower:
                return MetricResult(
                    name="faithfulness",
                    value=1.0,  # Perfect faithfulness - model refused to hallucinate
                    details={"no_context_response": True, "pattern_matched": pattern}
                )
        
        combined_context = " ".join(contexts)
        
        # Extract key phrases from answer
        answer_phrases = self._extract_key_phrases(answer)
        
        if not answer_phrases:
            return MetricResult("faithfulness", 1.0)  # No claims to verify
        
        # Check each phrase against context
        supported_count = 0
        unsupported_phrases = []
        
        for phrase in answer_phrases:
            if self._phrase_in_context(phrase, combined_context):
                supported_count += 1
            else:
                unsupported_phrases.append(phrase)
        
        faithfulness_score = supported_count / len(answer_phrases)
        
        return MetricResult(
            name="faithfulness",
            value=faithfulness_score,
            details={
                "total_phrases": len(answer_phrases),
                "supported": supported_count,
                "unsupported_phrases": unsupported_phrases[:5],
            }
        )
    
    def answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> MetricResult:
        """
        Measure answer relevancy to the question.
        
        Checks if the answer addresses what was asked.
        
        Args:
            question: Original question
            answer: Generated answer
            
        Returns:
            MetricResult with relevancy score
        """
        if not question or not answer:
            return MetricResult("answer_relevancy", 0.0)
        
        # Extract key terms from question
        question_terms = self._extract_key_terms(question)
        answer_terms = self._extract_key_terms(answer)
        
        if not question_terms:
            return MetricResult("answer_relevancy", 0.5)
        
        # Check overlap
        overlap = len(question_terms & answer_terms)
        relevancy = overlap / len(question_terms)
        
        # Boost if answer contains question-specific patterns
        if self._contains_answer_pattern(question, answer):
            relevancy = min(1.0, relevancy + 0.2)
        
        return MetricResult(
            name="answer_relevancy",
            value=relevancy,
            details={
                "question_terms": list(question_terms),
                "overlap": overlap,
            }
        )
    
    def answer_correctness(
        self,
        answer: str,
        ground_truth: str,
    ) -> MetricResult:
        """
        Measure answer correctness against ground truth.
        
        Uses token overlap and key fact matching.
        
        Args:
            answer: Generated answer
            ground_truth: Expected correct answer
            
        Returns:
            MetricResult with correctness score
        """
        if not answer or not ground_truth:
            return MetricResult("answer_correctness", 0.0)
        
        # Normalize texts
        answer_norm = self._normalize_arabic(answer.lower())
        truth_norm = self._normalize_arabic(ground_truth.lower())
        
        # Token-level F1
        answer_tokens = set(answer_norm.split())
        truth_tokens = set(truth_norm.split())
        
        if not truth_tokens:
            return MetricResult("answer_correctness", 0.0)
        
        overlap = answer_tokens & truth_tokens
        precision = len(overlap) / len(answer_tokens) if answer_tokens else 0
        recall = len(overlap) / len(truth_tokens)
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Check for key numbers/values
        answer_numbers = set(re.findall(r'\d+', answer))
        truth_numbers = set(re.findall(r'\d+', ground_truth))
        
        number_match = len(answer_numbers & truth_numbers) / len(truth_numbers) if truth_numbers else 1.0
        
        # Combined score
        correctness = 0.7 * f1 + 0.3 * number_match
        
        return MetricResult(
            name="answer_correctness",
            value=correctness,
            details={
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "number_match": number_match,
            }
        )
    
    def context_utilization(
        self,
        answer: str,
        contexts: List[str],
    ) -> MetricResult:
        """
        Measure how much of the context is utilized in the answer.
        
        Args:
            answer: Generated answer
            contexts: Retrieved context passages
            
        Returns:
            MetricResult with utilization score
        """
        if not answer or not contexts:
            return MetricResult("context_utilization", 0.0)
        
        answer_terms = self._extract_key_terms(answer)
        
        contexts_used = 0
        for ctx in contexts:
            ctx_terms = self._extract_key_terms(ctx)
            if ctx_terms & answer_terms:
                contexts_used += 1
        
        utilization = contexts_used / len(contexts)
        
        return MetricResult(
            name="context_utilization",
            value=utilization,
            details={"contexts_used": contexts_used, "total_contexts": len(contexts)}
        )
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        # Split by sentence boundaries
        sentences = re.split(r'[.،؟!]', text)
        
        phrases = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:
                # Extract noun phrases or key segments
                phrases.append(sent)
        
        return phrases
    
    def _phrase_in_context(self, phrase: str, context: str) -> bool:
        """Check if phrase is supported by context."""
        # Normalize
        phrase_norm = self._normalize_arabic(phrase.lower())
        context_norm = self._normalize_arabic(context.lower())
        
        # Direct substring
        if phrase_norm in context_norm:
            return True
        
        # Token overlap threshold
        phrase_tokens = set(phrase_norm.split())
        context_tokens = set(context_norm.split())
        
        overlap = len(phrase_tokens & context_tokens) / len(phrase_tokens) if phrase_tokens else 0
        return overlap > 0.5
    
    def _extract_key_terms(self, text: str) -> set:
        """Extract key terms from text."""
        # Remove stop words
        stop_words = {'ما', 'هي', 'هو', 'في', 'من', 'إلى', 'على', 'عن', 'مع', 'أن', 'التي', 'الذي', 'هذا', 'هذه', 'و', 'أو'}
        
        text_norm = self._normalize_arabic(text.lower())
        words = re.findall(r'[\u0600-\u06FF\w]{3,}', text_norm)
        
        return {w for w in words if w not in stop_words}
    
    def _contains_answer_pattern(self, question: str, answer: str) -> bool:
        """Check if answer contains expected patterns based on question type."""
        # What/ما questions should have descriptive answers
        if re.search(r'^ما\s|^what', question.lower()):
            return len(answer) > 50
        
        # Yes/No questions
        if re.search(r'^هل\s|^is\s|^can\s|^does\s', question.lower()):
            return bool(re.search(r'نعم|لا|yes|no', answer.lower()))
        
        return True
    
    def _normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text for comparison."""
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ة', 'ه').replace('ى', 'ي')
        return text
    
    def evaluate_all(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None,
    ) -> Dict[str, MetricResult]:
        """Run all generation metrics."""
        results = {
            "faithfulness": self.faithfulness(answer, contexts),
            "answer_relevancy": self.answer_relevancy(question, answer),
            "context_utilization": self.context_utilization(answer, contexts),
        }
        
        if ground_truth:
            results["answer_correctness"] = self.answer_correctness(answer, ground_truth)
        
        return results


class ArabicEvaluator:
    """
    Arabic-specific evaluation metrics for regulatory RAG.
    
    Metrics:
    - Citation Accuracy: Do citations match actual sources?
    - Article Reference Accuracy: Are article numbers correct?
    - Number Accuracy: Are numerical values correct?
    - Legal Term Accuracy: Are legal terms used correctly?
    """
    
    def __init__(self):
        logger.info("ArabicEvaluator initialized")
    
    def citation_accuracy(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
    ) -> MetricResult:
        """
        Check if citations in answer match actual sources.
        
        Args:
            answer: Generated answer with citations [1], [2], etc.
            sources: List of source documents
            
        Returns:
            MetricResult with citation accuracy
        """
        # Extract citation numbers from answer
        citations = re.findall(r'\[(\d+)\]', answer)
        
        if not citations:
            # No citations - could be good or bad depending on context
            return MetricResult("citation_accuracy", 0.5, details={"no_citations": True})
        
        valid_citations = 0
        invalid_citations = []
        
        for cite_num in citations:
            idx = int(cite_num) - 1
            if 0 <= idx < len(sources):
                # Citation index is valid
                source = sources[idx]
                source_text = source.get("content", source.get("text", ""))
                
                # Check if answer contains text that could come from this source
                if self._citation_supported(answer, source_text, cite_num):
                    valid_citations += 1
                else:
                    invalid_citations.append(cite_num)
            else:
                invalid_citations.append(cite_num)
        
        accuracy = valid_citations / len(set(citations)) if citations else 0.0
        
        return MetricResult(
            name="citation_accuracy",
            value=accuracy,
            details={
                "total_citations": len(set(citations)),
                "valid": valid_citations,
                "invalid": invalid_citations,
            }
        )
    
    def article_reference_accuracy(
        self,
        answer: str,
        ground_truth_articles: List[str],
    ) -> MetricResult:
        """
        Check if answer references the correct articles/clauses.
        
        Args:
            answer: Generated answer
            ground_truth_articles: List of correct article references
            
        Returns:
            MetricResult with article accuracy
        """
        # Extract article references from answer
        found_articles = re.findall(
            r'(?:المادة|مادة|البند|الفقرة|Article)\s*\(?\s*(\d+)\s*\)?',
            answer,
            re.IGNORECASE
        )
        
        if not ground_truth_articles:
            return MetricResult("article_accuracy", 1.0 if not found_articles else 0.5)
        
        found_set = set(found_articles)
        truth_set = set(ground_truth_articles)
        
        correct = len(found_set & truth_set)
        precision = correct / len(found_set) if found_set else 0.0
        recall = correct / len(truth_set) if truth_set else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return MetricResult(
            name="article_accuracy",
            value=f1,
            details={
                "found_articles": list(found_set),
                "expected_articles": list(truth_set),
                "correct": correct,
                "precision": precision,
                "recall": recall,
            }
        )
    
    def number_accuracy(
        self,
        answer: str,
        ground_truth: str,
    ) -> MetricResult:
        """
        Check if numerical values in answer are correct.
        
        Critical for regulatory content (capital requirements, fees, etc.)
        
        Args:
            answer: Generated answer
            ground_truth: Expected answer with correct numbers
            
        Returns:
            MetricResult with number accuracy
        """
        # Extract numbers with context
        answer_numbers = self._extract_numbers_with_context(answer)
        truth_numbers = self._extract_numbers_with_context(ground_truth)
        
        if not truth_numbers:
            return MetricResult("number_accuracy", 1.0 if not answer_numbers else 0.5)
        
        # Match numbers (exact match required for regulatory content)
        matched = 0
        for truth_num in truth_numbers:
            if any(self._numbers_match(truth_num, ans_num) for ans_num in answer_numbers):
                matched += 1
        
        accuracy = matched / len(truth_numbers)
        
        return MetricResult(
            name="number_accuracy",
            value=accuracy,
            details={
                "answer_numbers": answer_numbers,
                "truth_numbers": truth_numbers,
                "matched": matched,
            }
        )
    
    def anti_hallucination_score(
        self,
        answer: str,
        contexts: List[str],
    ) -> MetricResult:
        """
        Detect potential hallucinations in the answer.
        
        Checks for:
        - Numbers not in context
        - Specific claims not supported
        - Made-up entity names
        
        Args:
            answer: Generated answer
            contexts: Retrieved context passages
            
        Returns:
            MetricResult with anti-hallucination score (higher is better)
        """
        combined_context = " ".join(contexts)
        
        issues = []
        score = 1.0
        
        # Check numbers
        answer_numbers = set(re.findall(r'\d+', answer))
        context_numbers = set(re.findall(r'\d+', combined_context))
        
        unsupported_numbers = answer_numbers - context_numbers
        if unsupported_numbers:
            # Allow common numbers (1, 2, 3, years)
            truly_unsupported = {n for n in unsupported_numbers if int(n) > 10 and int(n) < 1900}
            if truly_unsupported:
                score -= 0.2 * min(len(truly_unsupported), 3)
                issues.append(f"Unsupported numbers: {truly_unsupported}")
        
        # Check for uncertainty markers (good sign)
        uncertainty_markers = ['قد', 'ربما', 'يحتمل', 'غير متأكد', 'لم أجد']
        has_uncertainty = any(marker in answer for marker in uncertainty_markers)
        
        # Check for "لا يوجد" when making claims
        if 'لا يوجد' in answer and not any('لا يوجد' in ctx or 'لا توجد' in ctx for ctx in contexts):
            score -= 0.3
            issues.append("Claims non-existence without context support")
        
        score = max(0.0, score)
        
        return MetricResult(
            name="anti_hallucination",
            value=score,
            details={
                "issues": issues,
                "has_uncertainty_markers": has_uncertainty,
            }
        )
    
    def _citation_supported(self, answer: str, source_text: str, cite_num: str) -> bool:
        """Check if citation is supported by source content."""
        # Find text near citation
        pattern = rf'\[{cite_num}\]'
        matches = list(re.finditer(pattern, answer))
        
        if not matches:
            return False
        
        # Get surrounding text
        for match in matches:
            start = max(0, match.start() - 100)
            end = min(len(answer), match.end() + 50)
            surrounding = answer[start:end]
            
            # Check if surrounding text has overlap with source
            surrounding_terms = set(re.findall(r'[\u0600-\u06FF\w]{4,}', surrounding.lower()))
            source_terms = set(re.findall(r'[\u0600-\u06FF\w]{4,}', source_text.lower()))
            
            overlap = len(surrounding_terms & source_terms)
            if overlap >= 2:
                return True
        
        return False
    
    def _extract_numbers_with_context(self, text: str) -> List[Tuple[str, str]]:
        """Extract numbers with surrounding context."""
        results = []
        
        # Find numbers with units
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(مليون|ألف|جنيه|%|يوم|شهر|سنة)',
            r'(\d+(?:\.\d+)?)\s*(million|thousand|EGP|%|days?|months?|years?)',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                results.append((match.group(1), match.group(2)))
        
        return results
    
    def _numbers_match(self, num1: Tuple[str, str], num2: Tuple[str, str]) -> bool:
        """Check if two number-unit pairs match."""
        val1, unit1 = num1
        val2, unit2 = num2
        
        # Normalize units
        unit_map = {
            'مليون': 'million', 'ألف': 'thousand', 'جنيه': 'egp',
            'يوم': 'day', 'شهر': 'month', 'سنة': 'year'
        }
        
        unit1_norm = unit_map.get(unit1.lower(), unit1.lower())
        unit2_norm = unit_map.get(unit2.lower(), unit2.lower())
        
        return val1 == val2 and unit1_norm == unit2_norm
    
    def evaluate_all(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        contexts: List[str],
        ground_truth: str = None,
        ground_truth_articles: List[str] = None,
    ) -> Dict[str, MetricResult]:
        """Run all Arabic-specific metrics."""
        results = {
            "citation_accuracy": self.citation_accuracy(answer, sources),
            "anti_hallucination": self.anti_hallucination_score(answer, contexts),
        }
        
        if ground_truth:
            results["number_accuracy"] = self.number_accuracy(answer, ground_truth)
        
        if ground_truth_articles:
            results["article_accuracy"] = self.article_reference_accuracy(answer, ground_truth_articles)
        
        return results
