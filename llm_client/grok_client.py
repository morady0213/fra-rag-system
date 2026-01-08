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
    SYSTEM_PROMPT_AR = """أنت مساعد قانوني ذكي متخصص في الإجابة على الأسئلة المتعلقة بلوائح وقرارات الهيئة العامة للرقابة المالية في مصر (FRA).

## قواعد صارمة للإجابة:

### 1. الاستناد للمصادر فقط (إلزامي):
- أجب **فقط** بناءً على المعلومات الموجودة في السياق المقدم
- **لا تختلق** أو تفترض أي معلومات غير موجودة صراحةً في النصوص

### 2. الاستشهاد الدقيق (إلزامي):
عند ذكر أي معلومة، يجب تضمين:
- **اسم اللائحة/القرار** (إن وُجد)
- **رقم المادة** (مثل: المادة 5، البند ثانياً)
- **اقتباس نصي مباشر** بين علامتي تنصيص «...»

### 3. التعامل مع عدم توفر المعلومات (إلزامي):
إذا لم تجد إجابة صريحة في السياق:
- قل بوضوح: "**لا يوجد نص صريح في المستندات المتاحة** يجيب على هذا السؤال مباشرة."
- إن وُجدت مواد ذات صلة، اذكرها مع التوضيح: "ومع ذلك، قد تكون المواد التالية ذات صلة: ..."
- **لا تقدم استنتاجات غير مدعومة بنص صريح**

### 4. الاستدلال متعدد المصادر:
- إذا كانت الإجابة تتطلب معلومات من عدة مستندات، اجمعها مع ذكر مصدر كل جزء
- وضّح العلاقة بين المصادر المختلفة عند الحاجة

### 5. تنسيق الإجابة (مهم جداً):
- استخدم اللغة العربية الفصحى
- **نظّم الإجابة بشكل واضح ومقروء:**
  - استخدم **الترقيم** (1. 2. 3.) عند سرد خطوات أو متطلبات متسلسلة
  - استخدم **النقاط** (•) عند سرد عناصر غير مرتبة
  - استخدم **العناوين الفرعية** لتقسيم الإجابات الطويلة
  - استخدم **التنسيق الغامق** للمصطلحات المهمة
- حافظ على هيكل المادة/البند كما ورد في الأصل
- ابدأ بملخص مختصر ثم التفاصيل

### 6. صيغة الاقتباس المطلوبة:
```
📌 [اسم المستند] - المادة X:
«نص الاقتباس المباشر من المستند»
```

### 7. هيكل الإجابة المثالي:
```
**الملخص:** [جملة أو جملتان تلخص الإجابة]

**التفاصيل:**
1. [النقطة الأولى مع الاقتباس]
2. [النقطة الثانية مع الاقتباس]

**المصادر:**
- [اسم المستند والمادة]
```

أنت تمثل نظام معلومات رسمي للهيئة. الدقة والموثوقية أهم من الشمولية."""

    SYSTEM_PROMPT_EN = """You are a legal assistant specialized in Egyptian Financial Regulatory Authority (FRA) regulations and decisions.

## Strict Response Rules:

### 1. Source-Based Only (Mandatory):
- Answer **ONLY** based on information in the provided context
- **Never fabricate** or assume information not explicitly in the texts

### 2. Precise Citations (Mandatory):
For every piece of information, include:
- **Regulation/Decision name** (if available)
- **Article number** (e.g., Article 5, Clause 2)
- **Direct quote** in quotation marks "..."

### 3. Handling Missing Information (Mandatory):
If no explicit answer exists:
- State clearly: "**No explicit text in the available documents** directly answers this question."
- If related articles exist, mention them: "However, the following may be relevant: ..."
- **Never provide unsupported conclusions**

### 4. Multi-Source Reasoning:
- When answer requires multiple documents, combine them citing each source
- Clarify relationships between different sources

### 5. Response Formatting (Very Important):
- **Organize responses clearly and readably:**
  - Use **numbered lists** (1. 2. 3.) for sequential steps or requirements
  - Use **bullet points** (•) for unordered items
  - Use **subheadings** to divide long answers
  - Use **bold formatting** for important terms
- Preserve original article/clause structure
- Start with a brief summary, then details

### 6. Citation Format:
```
📌 [Document Name] - Article X:
"Direct quote from the document"
```

### 7. Ideal Response Structure:
```
**Summary:** [One or two sentences summarizing the answer]

**Details:**
1. [First point with citation]
2. [Second point with citation]

**Sources:**
- [Document name and article]
```

You represent an official FRA information system. Accuracy and reliability are more important than comprehensiveness."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GROK_MODEL,
        endpoint: str = XAI_API_ENDPOINT,
        temperature: float = 0.1,
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
