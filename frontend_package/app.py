import gradio as gr
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from config import SAMPLE_DOCS_DIR, RAW_PDFS_DIR

# Configure logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}", level="INFO")

# Initialize RAG system (lazy loading)
_rag_system = None
_last_evidence = []  # Store last retrieved evidence for "Show Evidence" feature
_query_history = []  # Store query history
_feedback_log = []  # Store user feedback

# Feedback storage file
FEEDBACK_FILE = PROJECT_ROOT / "data" / "feedback.json"


def save_feedback(query: str, answer: str, feedback: str, timestamp: str):
    """Save user feedback to file."""
    import json
    from datetime import datetime
    
    feedback_entry = {
        "timestamp": timestamp or datetime.now().isoformat(),
        "query": query,
        "answer": answer[:500],  # Truncate for storage
        "feedback": feedback,  # "positive" or "negative"
    }
    
    _feedback_log.append(feedback_entry)
    
    # Save to file
    try:
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.append(feedback_entry)
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        logger.info(f"Feedback saved: {feedback}")
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")

def _build_metadata_filter(entity_filter: str, doc_type_filter: str, topic_filter: str) -> dict:
    """
    Build metadata filter dictionary from UI filter selections.
    
    Args:
        entity_filter: Selected entity type filter
        doc_type_filter: Selected document type filter
        topic_filter: Selected topic filter
        
    Returns:
        Dictionary with filter conditions for retrieval
    """
    filters = {}
    
    # Parse entity filter (extract Arabic key before parenthesis)
    if entity_filter and "الكل" not in entity_filter:
        entity_key = entity_filter.split(" (")[0].strip()
        filters["entity_type"] = entity_key
    
    # Parse document type filter
    if doc_type_filter and "الكل" not in doc_type_filter:
        doc_key = doc_type_filter.split(" (")[0].strip()
        filters["document_type"] = doc_key
    
    # Parse topic filter
    if topic_filter and "الكل" not in topic_filter:
        topic_key = topic_filter.split(" (")[0].strip()
        filters["topic"] = topic_key
    
    return filters if filters else None


def get_rag_system():
    """Get or initialize the RAG system."""
    global _rag_system
    if _rag_system is None:
        from main import FRARAGSystem
        logger.info("Initializing RAG system...")
        _rag_system = FRARAGSystem()
        
        # Auto-ingest if no documents
        if not _rag_system.is_indexed():
            has_docs = any(SAMPLE_DOCS_DIR.glob("*")) if SAMPLE_DOCS_DIR.exists() else False
            has_pdfs = any(RAW_PDFS_DIR.glob("*.pdf")) if RAW_PDFS_DIR.exists() else False
            if has_docs or has_pdfs:
                logger.info("Ingesting documents...")
                _rag_system.ingest_documents()
    
    return _rag_system


def process_query(
    message: str, 
    history: list, 
    language: str = "العربية", 
    num_sources: int = 5,
    use_hybrid: bool = True,
    use_rerank: bool = True,
    entity_filter: str = "الكل (All)",
    doc_type_filter: str = "الكل (All)",
    topic_filter: str = "الكل (All)",
    use_react_agent: bool = False,
) -> Tuple[str, str, str]:
    """
    Process a user query and return the answer with evidence.
    
    Args:
        message: User's question
        history: Chat history
        language: Response language
        num_sources: Number of sources to retrieve
        use_hybrid: Enable hybrid search (vector + BM25)
        use_rerank: Enable cross-encoder reranking
        entity_filter: Filter by entity type
        doc_type_filter: Filter by document type
        topic_filter: Filter by topic
        use_react_agent: Use ReAct agent for multi-hop reasoning
        
    Returns:
        Tuple of (answer, evidence_text, agent_trace)
    """
    global _last_evidence
    
    if not message.strip():
        no_question = "الرجاء إدخال سؤال." if language == "العربية" else "Please enter a question."
        return no_question, "", ""
    
    try:
        system = get_rag_system()
        
        # Build metadata filter from UI selections
        metadata_filter = _build_metadata_filter(entity_filter, doc_type_filter, topic_filter)
        
        # Use ReAct agent for multi-hop reasoning if enabled
        if use_react_agent and hasattr(system, 'react_agent') and system.react_agent:
            logger.info("Using ReAct agent for multi-hop reasoning")
            agent_result = system.react_agent.run(message, k=num_sources)
            
            # Build agent trace HTML for display
            agent_trace = _build_agent_trace_html(agent_result, language)
            
            # Store evidence
            _last_evidence = agent_result.sources
            evidence_text = _build_evidence_text(agent_result.sources, language, message)
            
            # Build sources text
            sources_text = ""
            if agent_result.sources:
                if language == "العربية":
                    sources_text = "\n\n---\n### 📚 المصادر المستخدمة:\n"
                else:
                    sources_text = "\n\n---\n### 📚 Sources Used:\n"
                for i, source in enumerate(agent_result.sources[:5], 1):
                    sources_text += f"**{i}.** {source.get('source', 'unknown')}\n"
            
            return agent_result.answer + sources_text, evidence_text, agent_trace
        
        # Use query router for intelligent retrieval (handles complex/comparison queries)
        if use_hybrid and hasattr(system, 'query_router'):
            retrieval_result = system.query_router.retrieve_with_routing(message, k=num_sources)
            strategy = retrieval_result.get("retrieval_strategy", "unknown")
            logger.info(f"Using query router (strategy={strategy})")
        elif use_hybrid and hasattr(system, 'hybrid_retriever'):
            retrieval_result = system.hybrid_retriever.retrieve_with_context(message, k=num_sources)
            logger.info(f"Using hybrid search (rerank={use_rerank})")
        else:
            retrieval_result = system.retriever.retrieve_with_context(message, k=num_sources)
            logger.info("Using basic vector search")
        context = retrieval_result["context"]
        sources = retrieval_result["sources"]
        
        # Store evidence for "Show Evidence" feature
        _last_evidence = sources
        
        # Build sources text with citations
        sources_text = ""
        if sources:
            if language == "العربية":
                sources_text = "\n\n---\n### 📚 المصادر المستخدمة:\n"
            else:
                sources_text = "\n\n---\n### 📚 Sources Used:\n"
            
            for i, source in enumerate(sources, 1):
                sources_text += f"**{i}.** {source['source']}\n"
        
        # Build evidence text for expandable section (with query for highlighting)
        evidence_text = _build_evidence_text(sources, language, message)
        
        # Handle no context found (anti-hallucination)
        if not context:
            if language == "العربية":
                no_info = "**⚠️ لا يوجد نص صريح في المستندات المتاحة** يجيب على هذا السؤال مباشرة.\n\nالرجاء صياغة السؤال بطريقة مختلفة أو التأكد من توفر المستندات ذات الصلة."
            else:
                no_info = "**⚠️ No explicit text in the available documents** directly answers this question.\n\nPlease rephrase your question or ensure relevant documents are available."
            return no_info, evidence_text, ""
        
        # Generate answer with LLM
        if system.llm_client:
            # Filter chunks to keep only relevant ones (reduces hallucination)
            if sources and len(sources) > 1:
                filtered_sources = system.llm_client.filter_relevant_chunks(message, sources)
                if filtered_sources:
                    # Rebuild context from filtered sources
                    context = "\n\n---\n\n".join([
                        f"📄 المصدر: {s.get('source', 'unknown')}\n{s.get('content', s.get('text', ''))}"
                        for s in filtered_sources
                    ])
                    sources = filtered_sources
                    logger.info(f"Using {len(filtered_sources)} filtered chunks for generation")
            
            result = system.llm_client.generate(
                query=message,
                context=context,
                sources=[s["source"] for s in sources],
            )
            return result.answer + sources_text, evidence_text, ""
        else:
            if language == "العربية":
                return f"⚠️ LLM غير متاح\n\n**السياق:**\n{context[:1500]}..." + sources_text, evidence_text, ""
            else:
                return f"⚠️ LLM not available\n\n**Context:**\n{context[:1500]}..." + sources_text, evidence_text, ""
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_msg = f"حدث خطأ: {e}" if language == "العربية" else f"An error occurred: {e}"
        return error_msg, "", ""


def _build_agent_trace_html(agent_result, language: str) -> str:
    """Build HTML display of ReAct agent reasoning steps."""
    if not agent_result or not agent_result.steps:
        return ""
    
    # Action icons
    action_icons = {
        "retrieve": "🔍",
        "calculate": "🧮", 
        "compare": "⚖️",
        "answer": "✅",
        "lookup": "📖",
    }
    
    # Action colors
    action_colors = {
        "retrieve": "#3498db",  # Blue
        "calculate": "#9b59b6",  # Purple
        "compare": "#e67e22",  # Orange
        "answer": "#27ae60",  # Green
        "lookup": "#1abc9c",  # Teal
    }
    
    if language == "العربية":
        html = '<div style="direction: rtl; text-align: right; font-family: Arial, sans-serif;">'
        html += '<h3>🧠 مسار الاستدلال (Reasoning Trace)</h3>'
        html += f'<p style="color: #666;">عدد الخطوات: {len(agent_result.steps)} | عمليات البحث: {agent_result.total_retrievals}</p>'
    else:
        html = '<div style="font-family: Arial, sans-serif;">'
        html += '<h3>🧠 Reasoning Trace</h3>'
        html += f'<p style="color: #666;">Steps: {len(agent_result.steps)} | Retrievals: {agent_result.total_retrievals}</p>'
    
    for step in agent_result.steps:
        action_name = step.action.value if hasattr(step.action, 'value') else str(step.action)
        icon = action_icons.get(action_name, "❓")
        color = action_colors.get(action_name, "#666")
        
        html += f'''
<div style="border-left: 4px solid {color}; padding: 12px; margin: 10px 0; background: #f8f9fa; border-radius: 0 8px 8px 0;">
    <div style="display: flex; align-items: center; margin-bottom: 8px;">
        <span style="background: {color}; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; margin-left: 10px;">
            الخطوة {step.step_number}
        </span>
        <span style="font-size: 1.5em; margin: 0 8px;">{icon}</span>
        <span style="background: #eee; padding: 4px 8px; border-radius: 4px; font-weight: 500;">
            {action_name.upper()}
        </span>
    </div>
    <div style="margin: 8px 0;">
        <strong>💭 التفكير:</strong>
        <p style="margin: 4px 0; color: #444;">{step.thought[:200]}{'...' if len(step.thought) > 200 else ''}</p>
    </div>
    <div style="margin: 8px 0;">
        <strong>🎯 المدخلات:</strong>
        <p style="margin: 4px 0; color: #555; background: #fff; padding: 8px; border-radius: 4px; font-family: monospace;">
            {step.action_input[:150]}{'...' if len(step.action_input) > 150 else ''}
        </p>
    </div>
    <div style="margin: 8px 0;">
        <strong>👁️ الملاحظة:</strong>
        <p style="margin: 4px 0; color: #666; font-size: 0.9em;">
            {step.observation[:200]}{'...' if len(step.observation) > 200 else ''}
        </p>
    </div>
</div>
'''
    
    # Summary
    status = "✅ اكتمل الاستدلال" if agent_result.reasoning_complete else "⚠️ توقف قبل الاكتمال"
    html += f'<p style="text-align: center; padding: 10px; background: #e8f5e9; border-radius: 8px;"><strong>{status}</strong></p>'
    html += '</div>'
    
    return html


def _build_evidence_text(sources: List[Dict], language: str, query: str = "") -> str:
    """Build formatted evidence text from sources with interactive cards."""
    if not sources:
        if language == "العربية":
            return '<div class="no-evidence">📭 لا توجد أدلة متاحة.</div>'
        return '<div class="no-evidence">📭 No evidence available.</div>'
    
    # Header
    if language == "العربية":
        html = '<div style="direction: rtl; text-align: right;">'
        html += '<h3>📖 الأدلة والنصوص المسترجعة</h3>'
        html += '<p style="color: #666;">انقر على أي مصدر لعرض النص الكامل مع تمييز الكلمات المطابقة</p>'
    else:
        html = '<div>'
        html += '<h3>📖 Retrieved Evidence & Texts</h3>'
        html += '<p style="color: #666;">Click any source to view full text with highlighted matches</p>'
    
    # Build source cards
    for i, source in enumerate(sources, 1):
        source_name = source.get('source', 'Unknown')
        score = source.get('score', 0)
        content = source.get('content', source.get('text', ''))
        
        # Highlight matching terms from query
        highlighted_content = _highlight_matches(content, query) if query else content
        
        # Score color based on relevance
        score_pct = int(score * 100)
        if score_pct >= 70:
            score_color = "#28a745"  # Green
        elif score_pct >= 50:
            score_color = "#ffc107"  # Yellow
        else:
            score_color = "#dc3545"  # Red
        
        html += f'''
<details class="source-card" style="margin-bottom: 12px; border: 1px solid #dee2e6; border-radius: 12px; overflow: hidden;">
    <summary class="source-header" style="padding: 12px 16px; cursor: pointer; display: flex; align-items: center; background: white; list-style: none;">
        <span class="source-number" style="background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); color: white; padding: 4px 10px; border-radius: 20px; font-weight: bold; margin-left: 10px;">[{i}]</span>
        <span class="source-name" style="flex-grow: 1; font-weight: 500; color: #333;">📄 {source_name}</span>
    </summary>
    <div class="source-content" style="padding: 16px; background: #fafbfc; direction: rtl; text-align: right; max-height: 300px; overflow-y: auto;">
        <div class="source-text" style="font-size: 0.95em; line-height: 1.8; color: #444; white-space: pre-wrap;">
{highlighted_content[:1500]}{'<br><br><em>... (نص مقتطع)</em>' if len(content) > 1500 else ''}
        </div>
    </div>
</details>
'''
    
    html += '</div>'
    return html


def _highlight_matches(text: str, query: str) -> str:
    """Highlight matching words from query in text."""
    import re
    
    if not query or not text:
        return text
    
    # Extract meaningful words from query (skip common Arabic stop words)
    stop_words = {'ما', 'هي', 'هو', 'في', 'من', 'إلى', 'على', 'عن', 'مع', 'أن', 'التي', 'الذي', 'هذا', 'هذه', 'كان', 'كانت', 'يكون', 'تكون', 'أو', 'و', 'ل', 'ب', 'ك', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'which', 'who', 'for', 'of', 'to', 'in', 'and', 'or'}
    
    # Normalize Arabic text
    query_normalized = query.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا').replace('ة', 'ه').replace('ى', 'ي')
    
    # Extract words (minimum 3 characters)
    words = [w for w in re.findall(r'[\u0600-\u06FF\w]{3,}', query_normalized) if w.lower() not in stop_words]
    
    if not words:
        return text
    
    # Create regex pattern for matching
    highlighted = text
    for word in words[:5]:  # Limit to 5 words to avoid over-highlighting
        # Build pattern with Arabic character variations
        pattern = word.replace('ا', '[اأإآ]').replace('ه', '[هة]').replace('ي', '[يى]')
        try:
            highlighted = re.sub(
                f'({pattern})',
                r'<mark style="background: linear-gradient(120deg, #fff3cd 0%, #ffeeba 100%); padding: 2px 4px; border-radius: 3px; font-weight: 500;">\1</mark>',
                highlighted,
                flags=re.IGNORECASE
            )
        except:
            pass  # Skip if regex fails
    
    return highlighted


def get_stats() -> str:
    """Get vector store statistics."""
    try:
        system = get_rag_system()
        stats = system.vector_store.get_stats()
        return f"""
**📊 إحصائيات قاعدة البيانات (Database Statistics)**

| الحقل | القيمة |
|-------|--------|
| المجموعة (Collection) | {stats.get('collection_name', 'N/A')} |
| عدد المستندات (Documents) | {stats.get('document_count', 0)} |
| نموذج التضمين (Embedding) | {stats.get('embedding_model', 'N/A')} |
| أبعاد المتجه (Vector Size) | {stats.get('vector_size', 'N/A')} |
"""
    except Exception as e:
        return f"خطأ: {e}"


def browse_chunks(limit: int = 10) -> str:
    """Browse chunks stored in Qdrant."""
    try:
        system = get_rag_system()
        client = system.vector_store.client
        
        # Scroll through chunks
        points, _ = client.scroll(
            collection_name="fra_documents",
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        if not points:
            return "❌ لا توجد مستندات في قاعدة البيانات"
        
        html = f"**📚 عرض {len(points)} من الأجزاء المخزنة (Chunks)**\n\n"
        
        for i, point in enumerate(points, 1):
            payload = point.payload
            source = payload.get('source', 'Unknown')
            entity = payload.get('entity_type', 'N/A')
            doc_type = payload.get('doc_type', 'N/A')
            content = payload.get('content', payload.get('text', ''))[:300]
            
            html += f"""
---
### 📄 Chunk {i} (ID: `{str(point.id)[:8]}...`)

| Field | Value |
|-------|-------|
| **Source** | {source} |
| **Entity** | {entity} |
| **Doc Type** | {doc_type} |

**Content Preview:**
> {content}...

"""
        
        return html
    except Exception as e:
        return f"❌ خطأ: {e}"


def upload_and_index_documents(files) -> str:
    """
    Upload documents and index them into the RAG system.
    
    Args:
        files: List of uploaded files from Gradio
        
    Returns:
        Status message
    """
    if not files:
        return "⚠️ لم يتم اختيار ملفات. (No files selected)"
    
    import shutil
    from pathlib import Path
    
    try:
        system = get_rag_system()
        uploaded_files = []
        
        for file in files:
            # Get file path and name
            file_path = Path(file.name) if hasattr(file, 'name') else Path(file)
            file_name = file_path.name
            
            # Copy to sample_docs directory
            dest_path = SAMPLE_DOCS_DIR / file_name
            shutil.copy(str(file_path), str(dest_path))
            uploaded_files.append(file_name)
            logger.info(f"Uploaded: {file_name}")
        
        # Re-ingest documents
        logger.info("Re-indexing documents...")
        count = system.ingest_documents(force=True)
        
        # Reset hybrid retriever BM25 index
        if hasattr(system, 'hybrid_retriever'):
            system.hybrid_retriever._bm25_synced = False
        
        files_list = "\n".join([f"- {f}" for f in uploaded_files])
        return f"""✅ **تم الرفع بنجاح! (Upload Successful!)**

**الملفات المرفوعة ({len(uploaded_files)}):**
{files_list}

**إجمالي المستندات المفهرسة:** {count}
"""
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return f"❌ خطأ في الرفع: {e}\n(Upload error: {e})"


def get_available_documents() -> List[str]:
    """Get list of available document names."""
    try:
        system = get_rag_system()
        # This would ideally query the vector store for unique sources
        docs = list(SAMPLE_DOCS_DIR.glob("*.*")) if SAMPLE_DOCS_DIR.exists() else []
        return [doc.name for doc in docs if not doc.name.startswith('.')]
    except:
        return []


# Custom CSS for RTL Arabic support and interactive sources
custom_css = """
.rtl-text {
    direction: rtl;
    text-align: right;
    font-family: 'Segoe UI', 'Arial', 'Tahoma', sans-serif;
}
.gradio-container {
    font-family: 'Segoe UI', 'Arial', 'Tahoma', sans-serif !important;
}
.message {
    direction: rtl;
    text-align: right;
}
.chatbot .message {
    direction: rtl;
    text-align: right;
}
.evidence-box {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    margin-top: 10px;
    direction: rtl;
}
.source-tag {
    background-color: #e7f3ff;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.9em;
}

/* Interactive source cards */
.source-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border: 1px solid #dee2e6;
    border-radius: 12px;
    margin-bottom: 12px;
    overflow: hidden;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.source-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    border-color: #007bff;
}
.source-header {
    padding: 12px 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: white;
    border-bottom: 1px solid #eee;
}
.source-header:hover {
    background: #f0f7ff;
}
.source-number {
    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
    color: white;
    padding: 4px 10px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.9em;
    margin-left: 10px;
}
.source-name {
    flex-grow: 1;
    font-weight: 500;
    color: #333;
    direction: rtl;
    text-align: right;
}
.relevance-badge {
    background: #28a745;
    color: white;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    margin-right: 10px;
}
.source-content {
    padding: 16px;
    background: #fafbfc;
    direction: rtl;
    text-align: right;
    max-height: 300px;
    overflow-y: auto;
}
.source-text {
    font-size: 0.95em;
    line-height: 1.8;
    color: #444;
    white-space: pre-wrap;
}
.highlight-match {
    background: linear-gradient(120deg, #fff3cd 0%, #ffeeba 100%);
    padding: 2px 4px;
    border-radius: 3px;
    font-weight: 500;
    border-bottom: 2px solid #ffc107;
}
.expand-icon {
    transition: transform 0.3s ease;
    font-size: 1.2em;
}
.expand-icon.expanded {
    transform: rotate(180deg);
}
.citation-link {
    color: #007bff;
    text-decoration: none;
    font-weight: bold;
    cursor: pointer;
    padding: 0 2px;
}
.citation-link:hover {
    text-decoration: underline;
    background: #e7f3ff;
    border-radius: 3px;
}
.no-evidence {
    text-align: center;
    padding: 40px;
    color: #6c757d;
}
"""

# Create Gradio interface
with gr.Blocks(title="نظام FRA RAG") as demo:
    
    # State for storing evidence and last response
    evidence_state = gr.State("")
    last_query_state = gr.State("")
    last_answer_state = gr.State("")
    
    gr.Markdown(
        """
        # 🏛️ نظام الاسترجاع المعزز للهيئة العامة للرقابة المالية
        ## FRA RAG System - Financial Regulatory Authority
        
        نظام ذكي للإجابة على الأسئلة المتعلقة بلوائح وقرارات الهيئة مع استشهادات دقيقة
        
        Intelligent Q&A system for FRA regulations with precise citations
        """,
        elem_classes=["rtl-text"]
    )
    
    with gr.Row():
        # Main chat column
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="المحادثة (Chat)",
                height=400,
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="سؤالك (Your Question)",
                    placeholder="اكتب سؤالك هنا... (Type your question here...)",
                    lines=2,
                    rtl=True,
                    scale=4,
                )
                submit_btn = gr.Button("إرسال\nSend", variant="primary", scale=1)
            
            # Feedback and action buttons
            with gr.Row():
                thumbs_up_btn = gr.Button("👍 مفيد (Helpful)", variant="secondary", scale=1)
                thumbs_down_btn = gr.Button("👎 غير مفيد (Not Helpful)", variant="secondary", scale=1)
                clear_btn = gr.Button("🗑️ مسح (Clear)", scale=1)
                show_evidence_btn = gr.Button("📖 الأدلة (Evidence)", scale=1)
            
            feedback_status = gr.Markdown("", visible=True)
            
            # Expandable evidence section with interactive HTML
            with gr.Accordion("📖 الأدلة والنصوص المسترجعة (Retrieved Evidence)", open=False):
                evidence_output = gr.HTML(
                    value='<div class="no-evidence">اطرح سؤالاً لعرض الأدلة المسترجعة...<br>Ask a question to see retrieved evidence...</div>',
                    elem_classes=["evidence-box"]
                )
            
            # Agent reasoning trace section
            with gr.Accordion("🧠 مسار الاستدلال (Agent Reasoning)", open=False):
                agent_trace_output = gr.HTML(
                    value='<div style="color: #666; text-align: center; padding: 20px;">فعّل وكيل ReAct لرؤية مسار الاستدلال<br>Enable ReAct agent to see reasoning trace</div>',
                    elem_classes=["agent-trace-box"]
                )
            
            # Query history section
            with gr.Accordion("📜 سجل الأسئلة (Query History)", open=False):
                history_output = gr.Markdown(
                    value="لم يتم طرح أي أسئلة بعد.\nNo questions asked yet."
                )
        
        # Settings column
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ الإعدادات (Settings)")
            
            language_select = gr.Radio(
                choices=["العربية", "English"],
                value="العربية",
                label="لغة الإجابة (Response Language)",
            )
            
            num_sources_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="عدد المصادر (Number of Sources)",
            )
            
            use_hybrid_checkbox = gr.Checkbox(
                value=True,
                label="🔀 بحث هجين (Hybrid Search)",
                info="دمج البحث الدلالي مع البحث بالكلمات المفتاحية"
            )
            
            use_rerank_checkbox = gr.Checkbox(
                value=True,
                label="🎯 إعادة الترتيب (Reranking)",
                info="تحسين دقة النتائج باستخدام نموذج إعادة الترتيب"
            )
            
            use_react_checkbox = gr.Checkbox(
                value=False,
                label="🧠 ReAct Agent (Multi-Hop Reasoning)",
                info="Enable multi-step reasoning for complex questions"
            )
            
            gr.Markdown("---")
            gr.Markdown("### 🔍 فلاتر البحث (Search Filters)")
            
            entity_filter = gr.Dropdown(
                choices=[
                    "الكل (All)",
                    "تمويل_استهلاكي (Consumer Finance)",
                    "تمويل_متناهي_الصغر (Microfinance)",
                    "تأجير_تمويلي (Leasing)",
                    "سمسرة (Brokerage)",
                    "تأمين (Insurance)",
                    "توريق (Securitization)",
                    "بنك (Bank)",
                ],
                value="الكل (All)",
                label="🏢 نوع الجهة (Entity Type)",
                info="تصفية حسب نوع الجهة الخاضعة"
            )
            
            doc_type_filter = gr.Dropdown(
                choices=[
                    "الكل (All)",
                    "نموذج (Form)",
                    "لائحة (Regulation)",
                    "قرار (Decision)",
                    "تعميم (Circular)",
                    "دليل (Guide)",
                ],
                value="الكل (All)",
                label="📄 نوع المستند (Document Type)",
                info="تصفية حسب نوع المستند"
            )
            
            topic_filter = gr.Dropdown(
                choices=[
                    "الكل (All)",
                    "ترخيص (Licensing)",
                    "رأس_المال (Capital)",
                    "فروع (Branches)",
                    "سندات (Bonds)",
                    "تسجيل (Registration)",
                ],
                value="الكل (All)",
                label="📑 الموضوع (Topic)",
                info="تصفية حسب الموضوع"
            )
            
            gr.Markdown("---")
            
            # Document upload section
            gr.Markdown("### 📤 رفع مستند (Upload Document)")
            file_upload = gr.File(
                label="اختر ملف (Select File)",
                file_types=[".docx", ".pdf", ".txt", ".md"],
                file_count="multiple",
            )
            upload_btn = gr.Button("📥 رفع وفهرسة (Upload & Index)", variant="secondary")
            upload_status = gr.Markdown("")
            
            gr.Markdown("---")
            
            stats_btn = gr.Button("📊 إحصائيات (Statistics)")
            stats_output = gr.Markdown("")
            
            browse_btn = gr.Button("📚 عرض الأجزاء (Browse Chunks)")
            chunks_output = gr.Markdown("")
            
            gr.Markdown(
                """
                ---
                ### 📝 أمثلة (Examples)
                
                - ما هي مستندات إصدار سندات التوريق؟
                - ما هي متطلبات إصدار السندات الخضراء؟
                - ما هي إجراءات غلق فرع لشركة تمويل؟
                
                ---
                ### ℹ️ ملاحظات
                - الإجابات مبنية على النصوص الرسمية فقط
                - استخدم 👍/👎 لتقييم الإجابات
                - البحث الهجين يحسّن النتائج للاستعلامات العربية
                """
            )
    
    # Hidden state for agent trace
    agent_trace_state = gr.State("")
    
    # Event handlers
    def respond(message, chat_history, language, num_sources, use_hybrid, use_rerank, use_react, entity_f, doc_type_f, topic_f):
        if not message.strip():
            return "", chat_history, "", "", "", get_history_text(), ""
        
        answer, evidence, agent_trace = process_query(
            message, chat_history, language, num_sources, use_hybrid, use_rerank,
            entity_f, doc_type_f, topic_f, use_react
        )
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": answer})
        
        # Update query history
        from datetime import datetime
        _query_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "query": message[:100],
            "language": language,
            "hybrid": use_hybrid,
            "rerank": use_rerank,
            "react": use_react,
        })
        
        return "", chat_history, evidence, message, answer, get_history_text(), agent_trace
    
    def get_history_text():
        """Format query history for display."""
        if not _query_history:
            return "لم يتم طرح أي أسئلة بعد.\nNo questions asked yet."
        
        history_md = "| الوقت | السؤال |\n|-------|--------|\n"
        for item in reversed(_query_history[-10:]):  # Last 10 queries
            history_md += f"| {item['timestamp']} | {item['query'][:50]}... |\n"
        return history_md
    
    def update_evidence(evidence):
        return evidence if evidence else "لا توجد أدلة متاحة حالياً.\nNo evidence available yet."
    
    def handle_feedback_positive(query, answer):
        """Handle positive feedback."""
        from datetime import datetime
        if query and answer:
            save_feedback(query, answer, "positive", datetime.now().isoformat())
            return "✅ شكراً لتقييمك! (Thank you for your feedback!)"
        return ""
    
    def handle_feedback_negative(query, answer):
        """Handle negative feedback."""
        from datetime import datetime
        if query and answer:
            save_feedback(query, answer, "negative", datetime.now().isoformat())
            return "📝 شكراً لتقييمك، سنعمل على التحسين! (Thank you, we'll improve!)"
        return ""
    
    def clear_chat():
        """Clear chat and reset states."""
        return [], "", "", "", get_history_text(), ""
    
    def update_agent_trace(trace):
        """Update agent trace display."""
        if trace:
            return trace
        return '<div style="color: #666; text-align: center; padding: 20px;">فعّل وكيل ReAct لرؤية مسار الاستدلال<br>Enable ReAct agent to see reasoning trace</div>'
    
    # Connect events
    msg.submit(
        respond, 
        [msg, chatbot, language_select, num_sources_slider, use_hybrid_checkbox, use_rerank_checkbox, use_react_checkbox, entity_filter, doc_type_filter, topic_filter], 
        [msg, chatbot, evidence_state, last_query_state, last_answer_state, history_output, agent_trace_state]
    ).then(
        update_evidence,
        [evidence_state],
        [evidence_output]
    ).then(
        update_agent_trace,
        [agent_trace_state],
        [agent_trace_output]
    )
    
    submit_btn.click(
        respond, 
        [msg, chatbot, language_select, num_sources_slider, use_hybrid_checkbox, use_rerank_checkbox, use_react_checkbox, entity_filter, doc_type_filter, topic_filter], 
        [msg, chatbot, evidence_state, last_query_state, last_answer_state, history_output, agent_trace_state]
    ).then(
        update_evidence,
        [evidence_state],
        [evidence_output]
    ).then(
        update_agent_trace,
        [agent_trace_state],
        [agent_trace_output]
    )
    
    # Feedback buttons
    thumbs_up_btn.click(
        handle_feedback_positive,
        [last_query_state, last_answer_state],
        [feedback_status]
    )
    
    thumbs_down_btn.click(
        handle_feedback_negative,
        [last_query_state, last_answer_state],
        [feedback_status]
    )
    
    clear_btn.click(clear_chat, None, [chatbot, evidence_state, last_query_state, last_answer_state, history_output, agent_trace_state])
    show_evidence_btn.click(update_evidence, [evidence_state], [evidence_output])
    stats_btn.click(get_stats, None, stats_output)
    browse_btn.click(browse_chunks, None, chunks_output)
    
    # Upload button
    upload_btn.click(upload_and_index_documents, [file_upload], [upload_status])


if __name__ == "__main__":
    logger.info("Starting FRA RAG Web UI (Enhanced)...")
    logger.info("Features: Hybrid Search, Reranking, Caching, Feedback, Query History")
    logger.info("Open http://localhost:7860 in your browser")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=custom_css,
    )
