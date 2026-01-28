"""
Grok (xAI) Client Module.

Provides integration with xAI's Grok API for answer generation.
Constructs prompts with retrieved Arabic context and enforces
that answers are grounded in the provided context only.

Endpoint: https://api.x.ai/v1/chat/completions
Models: grok-beta, grok-4 (when available)
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from loguru import logger

import requests

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import XAI_API_KEY, XAI_API_ENDPOINT, GROK_MODEL


@dataclass
class GenerationResult:
    """Represents the result of a generation request."""
    answer: str
    model: str
    usage: Dict[str, int]
    sources: List[str]
    query: str
    context_used: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "model": self.model,
            "usage": self.usage,
            "sources": self.sources,
            "query": self.query,
            "context_used": self.context_used,
        }


class GrokClient:
    """
    Client for xAI's Grok API.
    
    Designed for Arabic RAG applications with:
    - Context-grounded responses
    - Arabic language support
    - Regulatory/legal document focus
    """
    
    # System prompt for Arabic regulatory RAG
    # Instructs the model to answer ONLY based on provided context with proper citations
    SYSTEM_PROMPT_AR = """<role>محلل قانوني للهيئة العامة للرقابة المالية المصرية</role>

<constraints>
1. أجب فقط من السياق المقدم - ممنوع استخدام معرفة خارجية
2. إذا لم تجد الإجابة في السياق، قل: "NO_CONTEXT"
3. ممنوع: "عادةً"، "غالباً"، "من المتوقع"، أي تفسير أو استنتاج
4. لكل معلومة: اذكر المصدر (المادة/القرار)
</constraints>

<format>
[الإجابة المباشرة]
[المصدر: اسم المستند - المادة X]
</format>"""

    SYSTEM_PROMPT_EN = """<role>Legal analyst for Egyptian Financial Regulatory Authority (FRA)</role>

<constraints>
1. Answer ONLY from provided context - external knowledge forbidden
2. If answer not in context, respond: "NO_CONTEXT"
3. Forbidden: "usually", "typically", "expected", any interpretation
4. For every fact: cite source (Article/Decision)
</constraints>

<format>
[Direct answer]
[Source: Document name - Article X]
</format>"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GROK_MODEL,
        endpoint: str = XAI_API_ENDPOINT,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        use_arabic_prompt: bool = True,
    ):
        """
        Initialize the Grok client.
        
        Args:
            api_key: xAI API key (reads from env if not provided)
            model: Model name (grok-beta or grok-4)
            endpoint: API endpoint URL
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            use_arabic_prompt: Use Arabic system prompt
        """
        self.api_key = api_key or XAI_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "xAI API key not found. "
                "Set XAI_API_KEY environment variable or pass api_key parameter."
            )
        
        self.model = model
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = (
            self.SYSTEM_PROMPT_AR if use_arabic_prompt else self.SYSTEM_PROMPT_EN
        )
        
        # Request headers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        logger.info(f"GrokClient initialized with model: {model}")
    
    def filter_relevant_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        min_chunks: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Filter chunks to keep only those relevant to the query.
        Uses LLM to identify which chunks contain the answer.
        
        Args:
            query: User's question
            chunks: List of retrieved chunks with 'content' and 'source' keys
            min_chunks: Minimum number of chunks to keep (prevents over-filtering)
            
        Returns:
            Filtered list of relevant chunks
        """
        if not chunks:
            return []
        
        # Don't filter if only 1-2 chunks
        if len(chunks) <= 2:
            return chunks
        
        # Build chunk list for LLM
        chunk_descriptions = []
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", chunk.get("text", ""))[:400]  # Truncate for efficiency
            source = chunk.get("source", "unknown")
            chunk_descriptions.append(f"[{i}] Source: {source}\n{content}")
        
        chunks_text = "\n\n".join(chunk_descriptions)
        
        filter_prompt = f"""<task>Rate each chunk's relevance to answering the query.</task>

<query>{query}</query>

<chunks>
{chunks_text}
</chunks>

<instruction>
Return chunk numbers that MAY contain useful information for answering.
Be INCLUSIVE - include chunks that are even partially relevant.
Format: comma-separated numbers (e.g., "0,1,2")
Include at least the top 1-2 most relevant chunks.
</instruction>"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": filter_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 50,
        }
        
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            
            logger.info(f"Context filter response: {answer}")
            
            # Parse chunk indices
            try:
                # Extract all numbers from response
                import re
                indices = [int(x) for x in re.findall(r'\d+', answer)]
                indices = list(dict.fromkeys(indices))  # Remove duplicates, preserve order
                filtered = [chunks[i] for i in indices if 0 <= i < len(chunks)]
                
                # Ensure minimum chunks
                if len(filtered) < min_chunks:
                    # Add back highest-scored chunks
                    for chunk in chunks:
                        if chunk not in filtered:
                            filtered.append(chunk)
                        if len(filtered) >= min_chunks:
                            break
                
                logger.info(f"Filtered {len(chunks)} chunks to {len(filtered)} relevant chunks")
                return filtered if filtered else chunks
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse filter response: {e}, using all chunks")
                return chunks
                
        except Exception as e:
            logger.warning(f"Context filter failed: {e}, using all chunks")
            return chunks
    
    def generate(
        self,
        query: str,
        context: str,
        sources: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate an answer based on the query and context.
        
        Args:
            query: User's question
            context: Retrieved context from documents
            sources: List of source documents used
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            GenerationResult with the answer and metadata
        """
        sources = sources or []
        
        # Construct the user message with context
        user_message = self._build_user_message(query, context)
        
        # Build the request payload
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        logger.info(f"Generating answer for: {query[:100]}...")
        logger.debug(f"Context length: {len(context)} chars")
        
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=60,
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the answer
            answer = result["choices"][0]["message"]["content"]
            
            # Extract usage stats
            usage = result.get("usage", {})
            
            logger.info(
                f"Generated response: {len(answer)} chars, "
                f"tokens: {usage.get('total_tokens', 'N/A')}"
            )
            
            return GenerationResult(
                answer=answer,
                model=result.get("model", self.model),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                sources=sources,
                query=query,
                context_used=bool(context),
            )
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            logger.error(f"Response: {e.response.text if e.response else 'N/A'}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing response: {e}")
            raise
    
    def _build_user_message(self, query: str, context: str, language: str = "ar") -> str:
        """
        Build the user message with context and query.
        
        Args:
            query: User's question
            context: Retrieved context
            language: Response language ('ar' for Arabic, 'en' for English)
            
        Returns:
            Formatted user message
        """
        if context:
            if language == "ar":
                # Arabic message format with context
                message = f"""## السياق من الوثائق الرسمية للهيئة العامة للرقابة المالية:

{context}

---

## السؤال: {query}

## تعليمات الإجابة:
1. أجب بناءً على السياق المقدم أعلاه **فقط**
2. اقتبس النصوص ذات الصلة مباشرةً باستخدام «علامات التنصيص»
3. اذكر اسم المستند ورقم المادة/البند لكل اقتباس
4. إذا لم تجد إجابة صريحة، صرّح بذلك بوضوح واقترح مواد ذات صلة إن وُجدت
5. لا تقدم أي معلومات من خارج السياق المقدم"""
            else:
                # English message format
                message = f"""## Context from Official FRA Documents:

{context}

---

## Question: {query}

## Response Instructions:
1. Answer based **ONLY** on the context above
2. Quote relevant texts directly using "quotation marks"
3. Cite document name and article/clause number for each quote
4. If no explicit answer exists, state this clearly and suggest related articles if any
5. Do not provide any information from outside the given context"""
        else:
            if language == "ar":
                message = f"""السؤال: {query}

⚠️ **تنبيه**: لا يوجد سياق متاح من الوثائق لهذا السؤال.
الرجاء الرد بأنه لا توجد معلومات متاحة في المستندات الحالية."""
            else:
                message = f"""Question: {query}

⚠️ **Note**: No context available from documents for this question.
Please respond that no information is available in current documents."""
        
        return message
    
    def generate_with_retrieval(
        self,
        query: str,
        retriever,
        k: int = 5,
    ) -> GenerationResult:
        """
        Generate answer with automatic retrieval.
        
        Convenience method that handles retrieval and generation in one call.
        
        Args:
            query: User's question
            retriever: Retriever instance
            k: Number of documents to retrieve
            
        Returns:
            GenerationResult with the answer
        """
        # Retrieve relevant context
        retrieval_result = retriever.retrieve_with_context(query, k=k)
        
        context = retrieval_result["context"]
        sources = [s["source"] for s in retrieval_result["sources"]]
        
        # Generate answer
        return self.generate(
            query=query,
            context=context,
            sources=sources,
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request with custom messages.
        
        For advanced use cases where you need full control over the conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Assistant's response text
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise


def generate_answer(
    query: str,
    context: str,
    sources: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Convenience function to generate an answer.
    
    Args:
        query: User's question
        context: Retrieved context
        sources: Source documents
        api_key: Optional API key
        
    Returns:
        Generated answer text
    """
    client = GrokClient(api_key=api_key)
    result = client.generate(query, context, sources)
    return result.answer


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage (requires valid API key)
    try:
        client = GrokClient()
        
        # Sample context
        sample_context = """
        [المصدر 1: legislation/law_10_2009.pdf]
        المادة الأولى: تنشأ هيئة عامة تسمى "الهيئة العامة للرقابة المالية" تكون لها الشخصية الاعتبارية العامة.
        
        [المصدر 2: about.md]
        تختص الهيئة بالرقابة والإشراف على الأسواق والأدوات المالية غير المصرفية بما في ذلك أسواق رأس المال وأنشطة التأمين والتمويل العقاري.
        """
        
        query = "ما هي اختصاصات الهيئة العامة للرقابة المالية؟"
        
        result = client.generate(
            query=query,
            context=sample_context,
            sources=["legislation/law_10_2009.pdf", "about.md"],
        )
        
        print("=" * 60)
        print("Query:", result.query)
        print("=" * 60)
        print("\nAnswer:")
        print(result.answer)
        print("\n" + "-" * 60)
        print(f"Model: {result.model}")
        print(f"Tokens used: {result.usage}")
        print(f"Sources: {result.sources}")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set XAI_API_KEY environment variable.")
    except Exception as e:
        print(f"Error: {e}")
