"""
FRA RAG System - Enhanced Web UI

A Gradio-based web interface for the FRA Regulatory RAG system.
Features:
- Arabic RTL display with proper formatting
- Cited answers with regulation names, article numbers, and exact quotes
- Show Evidence expandable feature
- Anti-hallucination with explicit "not found" responses
- Multi-document reasoning
- Document filtering by type
- Bilingual support (Arabic/English)

Usage:
    python app.py
"""

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
) -> Tuple[str, str]:
    """
    Process a user query and return the answer with evidence.
    
    Args:
        message: User's question
        history: Chat history
        language: Response language
        num_sources: Number of sources to retrieve
        use_hybrid: Enable hybrid search (vector + BM25)
        use_rerank: Enable cross-encoder reranking
        
    Returns:
        Tuple of (answer, evidence_text)
    """
    global _last_evidence
    
    if not message.strip():
        no_question = "الرجاء إدخال سؤال." if language == "العربية" else "Please enter a question."
        return no_question, ""
    
    try:
        system = get_rag_system()
        
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
                sources_text += f"**{i}.** {source['source']} (relevance: {source['score']:.1%})\n"
        
        # Build evidence text for expandable section
        evidence_text = _build_evidence_text(sources, language)
        
        # Handle no context found (anti-hallucination)
        if not context:
            if language == "العربية":
                no_info = "**⚠️ لا يوجد نص صريح في المستندات المتاحة** يجيب على هذا السؤال مباشرة.\n\nالرجاء صياغة السؤال بطريقة مختلفة أو التأكد من توفر المستندات ذات الصلة."
            else:
                no_info = "**⚠️ No explicit text in the available documents** directly answers this question.\n\nPlease rephrase your question or ensure relevant documents are available."
            return no_info, evidence_text
        
        # Generate answer with LLM
        if system.llm_client:
            result = system.llm_client.generate(
                query=message,
                context=context,
                sources=[s["source"] for s in sources],
            )
            return result.answer + sources_text, evidence_text
        else:
            if language == "العربية":
                return f"⚠️ LLM غير متاح\n\n**السياق:**\n{context[:1500]}..." + sources_text, evidence_text
            else:
                return f"⚠️ LLM not available\n\n**Context:**\n{context[:1500]}..." + sources_text, evidence_text
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_msg = f"حدث خطأ: {e}" if language == "العربية" else f"An error occurred: {e}"
        return error_msg, ""


def _build_evidence_text(sources: List[Dict], language: str) -> str:
    """Build formatted evidence text from sources."""
    if not sources:
        if language == "العربية":
            return "لا توجد أدلة متاحة."
        return "No evidence available."
    
    if language == "العربية":
        evidence = "## 📖 الأدلة والنصوص المسترجعة\n\n"
        evidence += "هذه النصوص الأصلية التي تم استخدامها للإجابة على سؤالك:\n\n"
    else:
        evidence = "## 📖 Retrieved Evidence & Texts\n\n"
        evidence += "These are the original texts used to answer your question:\n\n"
    
    for i, source in enumerate(sources, 1):
        evidence += f"---\n### 📌 المصدر {i}: {source['source']}\n"
        evidence += f"**Relevance Score:** {source['score']:.1%}\n\n"
        
        # Get the actual content
        content = source.get('content', source.get('text', 'N/A'))
        if content and content != 'N/A':
            evidence += f"```\n{content[:1000]}{'...' if len(content) > 1000 else ''}\n```\n\n"
        else:
            evidence += "_No content available_\n\n"
    
    return evidence


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


# Custom CSS for RTL Arabic support
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
            
            # Expandable evidence section
            with gr.Accordion("📖 الأدلة والنصوص المسترجعة (Retrieved Evidence)", open=False):
                evidence_output = gr.Markdown(
                    value="اطرح سؤالاً لعرض الأدلة المسترجعة...\nAsk a question to see retrieved evidence...",
                    elem_classes=["evidence-box"]
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
    
    # Event handlers
    def respond(message, chat_history, language, num_sources, use_hybrid, use_rerank):
        if not message.strip():
            return "", chat_history, "", "", "", get_history_text()
        
        answer, evidence = process_query(
            message, chat_history, language, num_sources, use_hybrid, use_rerank
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
        })
        
        return "", chat_history, evidence, message, answer, get_history_text()
    
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
        return [], "", "", "", get_history_text()
    
    # Connect events
    msg.submit(
        respond, 
        [msg, chatbot, language_select, num_sources_slider, use_hybrid_checkbox, use_rerank_checkbox], 
        [msg, chatbot, evidence_state, last_query_state, last_answer_state, history_output]
    ).then(
        update_evidence,
        [evidence_state],
        [evidence_output]
    )
    
    submit_btn.click(
        respond, 
        [msg, chatbot, language_select, num_sources_slider, use_hybrid_checkbox, use_rerank_checkbox], 
        [msg, chatbot, evidence_state, last_query_state, last_answer_state, history_output]
    ).then(
        update_evidence,
        [evidence_state],
        [evidence_output]
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
    
    clear_btn.click(clear_chat, None, [chatbot, evidence_state, last_query_state, last_answer_state, history_output])
    show_evidence_btn.click(update_evidence, [evidence_state], [evidence_output])
    stats_btn.click(get_stats, None, stats_output)
    
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
