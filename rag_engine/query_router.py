"""
Query Router and Decomposition Module.

Provides intelligent query routing and decomposition for complex questions.

Features:
- Query complexity detection
- Sub-query generation for multi-part questions
- Retrieval strategy routing (standard vs decomposed)
- Result aggregation from multiple sub-queries
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class QueryType(Enum):
    """Types of queries based on complexity."""
    SIMPLE = "simple"           # Single topic, direct question
    COMPARISON = "comparison"   # Comparing two or more things
    MULTI_PART = "multi_part"   # Multiple distinct questions
    PROCEDURAL = "procedural"   # Step-by-step process question


@dataclass
class SubQuery:
    """Represents a decomposed sub-query."""
    query: str
    intent: str
    weight: float = 1.0


@dataclass
class RoutingDecision:
    """Routing decision for a query."""
    query_type: QueryType
    use_decomposition: bool
    sub_queries: List[SubQuery]
    reasoning: str


class QueryAnalyzer:
    """
    Analyzes queries to determine complexity and optimal retrieval strategy.
    """
    
    # Arabic comparison indicators
    COMPARISON_PATTERNS_AR = [
        r'الفرق بين',
        r'ما الفرق',
        r'قارن بين',
        r'مقارنة بين',
        r'أيهما',
        r'ايهما',
        r'بالمقارنة مع',
        r'مقابل',
        r'في مقابل',
        r'عكس',
        r'بينما',
        r'أم\s',
        r'ام\s',
    ]
    
    # Arabic multi-part indicators
    MULTI_PART_PATTERNS_AR = [
        r'وكذلك',
        r'وأيضا',
        r'وايضا',
        r'بالإضافة إلى',
        r'بالاضافة الى',
        r'علاوة على',
        r'فضلا عن',
        r'وما هي',
        r'وما هو',
        r'وكيف',
        r'ومتى',
        r'وأين',
        r'واين',
    ]
    
    # Arabic procedural indicators
    PROCEDURAL_PATTERNS_AR = [
        r'خطوات',
        r'إجراءات',
        r'اجراءات',
        r'كيفية',
        r'كيف يتم',
        r'ما هي طريقة',
        r'عملية',
        r'مراحل',
    ]
    
    # English patterns
    COMPARISON_PATTERNS_EN = [
        r'difference between',
        r'compare',
        r'comparison',
        r'versus',
        r'vs\.?',
        r'or\s',
        r'which one',
        r'better than',
    ]
    
    MULTI_PART_PATTERNS_EN = [
        r'and also',
        r'in addition',
        r'furthermore',
        r'as well as',
        r'and what',
        r'and how',
        r'and when',
    ]
    
    def analyze(self, query: str) -> QueryType:
        """
        Analyze query to determine its type.
        
        Args:
            query: User query
            
        Returns:
            QueryType enum value
        """
        query_lower = query.lower()
        
        # Check for comparison
        for pattern in self.COMPARISON_PATTERNS_AR + self.COMPARISON_PATTERNS_EN:
            if re.search(pattern, query_lower):
                return QueryType.COMPARISON
        
        # Check for multi-part
        for pattern in self.MULTI_PART_PATTERNS_AR + self.MULTI_PART_PATTERNS_EN:
            if re.search(pattern, query_lower):
                return QueryType.MULTI_PART
        
        # Check for procedural
        for pattern in self.PROCEDURAL_PATTERNS_AR:
            if re.search(pattern, query_lower):
                return QueryType.PROCEDURAL
        
        return QueryType.SIMPLE


class QueryDecomposer:
    """
    Decomposes complex queries into simpler sub-queries.
    """
    
    def decompose(self, query: str, query_type: QueryType) -> List[SubQuery]:
        """
        Decompose a query based on its type.
        
        Args:
            query: Original query
            query_type: Detected query type
            
        Returns:
            List of SubQuery objects
        """
        if query_type == QueryType.COMPARISON:
            return self._decompose_comparison(query)
        elif query_type == QueryType.MULTI_PART:
            return self._decompose_multi_part(query)
        elif query_type == QueryType.PROCEDURAL:
            return self._decompose_procedural(query)
        else:
            return [SubQuery(query=query, intent="direct", weight=1.0)]
    
    def _decompose_comparison(self, query: str) -> List[SubQuery]:
        """Decompose comparison queries."""
        sub_queries = []
        
        # Try to extract the two subjects being compared
        # Pattern: "الفرق بين X و Y" or "difference between X and Y"
        
        # Arabic pattern
        ar_match = re.search(
            r'(?:الفرق بين|ما الفرق بين|قارن بين|مقارنة بين)\s*(.+?)\s*(?:و|وبين)\s*(.+?)(?:\?|؟|$)',
            query
        )
        
        # English pattern
        en_match = re.search(
            r'(?:difference between|compare)\s*(.+?)\s*(?:and|vs\.?|versus)\s*(.+?)(?:\?|$)',
            query,
            re.IGNORECASE
        )
        
        match = ar_match or en_match
        
        if match:
            subject1 = match.group(1).strip()
            subject2 = match.group(2).strip()
            
            # Create sub-queries for each subject
            sub_queries.append(SubQuery(
                query=f"ما هي متطلبات {subject1}؟" if self._is_arabic(query) else f"What are the requirements for {subject1}?",
                intent=f"requirements_{subject1}",
                weight=1.0
            ))
            sub_queries.append(SubQuery(
                query=f"ما هي متطلبات {subject2}؟" if self._is_arabic(query) else f"What are the requirements for {subject2}?",
                intent=f"requirements_{subject2}",
                weight=1.0
            ))
            
            # Also search for direct comparison
            sub_queries.append(SubQuery(
                query=query,
                intent="direct_comparison",
                weight=0.5
            ))
        else:
            # Fallback: just use original query
            sub_queries.append(SubQuery(query=query, intent="comparison", weight=1.0))
        
        return sub_queries
    
    def _decompose_multi_part(self, query: str) -> List[SubQuery]:
        """Decompose multi-part queries."""
        sub_queries = []
        
        # Split by common conjunctions
        parts = re.split(r'[،,]\s*(?:و|and)\s*|[،,]\s*(?:وكذلك|وأيضا|بالإضافة)\s*', query)
        
        for i, part in enumerate(parts):
            part = part.strip()
            if part and len(part) > 5:  # Skip very short parts
                sub_queries.append(SubQuery(
                    query=part,
                    intent=f"part_{i+1}",
                    weight=1.0 / len(parts)
                ))
        
        if not sub_queries:
            sub_queries.append(SubQuery(query=query, intent="multi_part", weight=1.0))
        
        return sub_queries
    
    def _decompose_procedural(self, query: str) -> List[SubQuery]:
        """Decompose procedural queries into steps."""
        # For procedural queries, we search for:
        # 1. The overall process
        # 2. Specific requirements/documents
        # 3. Conditions/prerequisites
        
        sub_queries = [
            SubQuery(query=query, intent="process_overview", weight=1.0),
        ]
        
        # Extract the main subject
        subject_match = re.search(r'(?:خطوات|إجراءات|كيفية|كيف يتم)\s*(.+?)(?:\?|؟|$)', query)
        if subject_match:
            subject = subject_match.group(1).strip()
            sub_queries.append(SubQuery(
                query=f"ما هي المستندات المطلوبة لـ{subject}؟",
                intent="required_documents",
                weight=0.8
            ))
            sub_queries.append(SubQuery(
                query=f"ما هي شروط {subject}؟",
                intent="conditions",
                weight=0.6
            ))
        
        return sub_queries
    
    def _is_arabic(self, text: str) -> bool:
        """Check if text is primarily Arabic."""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        return arabic_chars > len(text) * 0.3


class QueryRouter:
    """
    Routes queries to appropriate retrieval strategies.
    
    Decides between:
    - Standard retrieval (single query)
    - Decomposed retrieval (multiple sub-queries)
    """
    
    def __init__(
        self,
        retriever,
        decomposition_threshold: float = 0.7,
    ):
        """
        Initialize query router.
        
        Args:
            retriever: Base retriever (HybridRetriever or Retriever)
            decomposition_threshold: Confidence threshold for using decomposition
        """
        self.retriever = retriever
        self.analyzer = QueryAnalyzer()
        self.decomposer = QueryDecomposer()
        self.decomposition_threshold = decomposition_threshold
        
        logger.info("QueryRouter initialized with decomposition support")
    
    def route(self, query: str) -> RoutingDecision:
        """
        Analyze query and decide on retrieval strategy.
        
        Args:
            query: User query
            
        Returns:
            RoutingDecision with strategy details
        """
        query_type = self.analyzer.analyze(query)
        
        # Decide if decomposition is needed
        use_decomposition = query_type in [QueryType.COMPARISON, QueryType.MULTI_PART]
        
        if use_decomposition:
            sub_queries = self.decomposer.decompose(query, query_type)
            reasoning = f"Query detected as {query_type.value}, decomposed into {len(sub_queries)} sub-queries"
        else:
            sub_queries = [SubQuery(query=query, intent="direct", weight=1.0)]
            reasoning = f"Query detected as {query_type.value}, using standard retrieval"
        
        logger.info(reasoning)
        
        return RoutingDecision(
            query_type=query_type,
            use_decomposition=use_decomposition,
            sub_queries=sub_queries,
            reasoning=reasoning
        )
    
    def retrieve_with_routing(
        self,
        query: str,
        k: int = 5,
        force_decomposition: bool = False,
    ) -> Dict[str, Any]:
        """
        Retrieve documents with intelligent routing.
        
        Args:
            query: User query
            k: Number of results per sub-query
            force_decomposition: Force decomposition regardless of query type
            
        Returns:
            Combined retrieval results with context
        """
        decision = self.route(query)
        
        if force_decomposition or decision.use_decomposition:
            return self._decomposed_retrieval(query, decision, k)
        else:
            return self._standard_retrieval(query, k)
    
    def _standard_retrieval(self, query: str, k: int) -> Dict[str, Any]:
        """Standard single-query retrieval."""
        results = self.retriever.retrieve_with_context(query, k=k)
        results["retrieval_strategy"] = "standard"
        results["sub_queries"] = [query]
        return results
    
    def _decomposed_retrieval(
        self,
        original_query: str,
        decision: RoutingDecision,
        k: int
    ) -> Dict[str, Any]:
        """
        Decomposed multi-query retrieval.
        
        Executes sub-queries and aggregates results.
        """
        all_results = []
        all_sources = []
        sub_query_results = {}
        
        # Execute each sub-query
        for sub_query in decision.sub_queries:
            try:
                result = self.retriever.retrieve_with_context(sub_query.query, k=k)
                sub_query_results[sub_query.intent] = {
                    "query": sub_query.query,
                    "sources": result.get("sources", []),
                    "context": result.get("context", ""),
                }
                
                # Collect sources with weight adjustment
                for source in result.get("sources", []):
                    adjusted_source = source.copy()
                    adjusted_source["score"] = source["score"] * sub_query.weight
                    adjusted_source["sub_query"] = sub_query.query
                    adjusted_source["intent"] = sub_query.intent
                    all_results.append(adjusted_source)
                    
            except Exception as e:
                logger.error(f"Sub-query retrieval error for '{sub_query.query}': {e}")
        
        # Deduplicate and merge results
        merged_results = self._merge_results(all_results, k * 2)  # Get more for diversity
        
        # Build combined context
        context_parts = []
        seen_content = set()
        
        for result in merged_results:
            content_hash = hash(result.get("content", "")[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                source_info = f"[المصدر: {result.get('source', 'unknown')}]"
                if "sub_query" in result:
                    source_info += f" (من استعلام: {result['intent']})"
                context_parts.append(f"{source_info}\n{result.get('content', '')}")
        
        combined_context = "\n\n---\n\n".join(context_parts[:k])
        
        return {
            "context": combined_context,
            "sources": merged_results[:k],
            "retrieval_strategy": "decomposed",
            "query_type": decision.query_type.value,
            "sub_queries": [sq.query for sq in decision.sub_queries],
            "sub_query_results": sub_query_results,
            "reasoning": decision.reasoning,
        }
    
    def _merge_results(
        self,
        results: List[Dict],
        k: int
    ) -> List[Dict]:
        """
        Merge and deduplicate results from multiple sub-queries.
        
        Uses content-based deduplication and score aggregation.
        """
        # Group by content hash
        content_groups: Dict[str, List[Dict]] = {}
        
        for result in results:
            content = result.get("content", "")
            content_key = hash(content[:200])
            
            if content_key not in content_groups:
                content_groups[content_key] = []
            content_groups[content_key].append(result)
        
        # Aggregate scores for duplicates
        merged = []
        for content_key, group in content_groups.items():
            best_result = max(group, key=lambda x: x.get("score", 0))
            # Boost score if found by multiple sub-queries
            if len(group) > 1:
                best_result["score"] = min(1.0, best_result["score"] * (1 + 0.2 * len(group)))
                best_result["found_by_multiple"] = True
            merged.append(best_result)
        
        # Sort by score
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return merged[:k]


def create_query_router(retriever, **kwargs) -> QueryRouter:
    """Factory function to create QueryRouter."""
    return QueryRouter(retriever=retriever, **kwargs)
