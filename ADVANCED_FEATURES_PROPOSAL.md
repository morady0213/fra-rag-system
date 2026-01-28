# FRA RAG System - Advanced Features Proposal

## Executive Summary

This proposal outlines a phased approach to evolve the FRA RAG system from a solid retrieval-augmented generation platform into an enterprise-grade **Legal AI Assistant** with advanced reasoning, knowledge graphs, and rigorous evaluation capabilities.

---

## Part 1: Technical Advancements

### 1. Hierarchical Document Chunking (Parent-Document Retrieval)

#### Current Limitation
```
Chunk 1: "المادة 5: يجب على الشركة..."
Chunk 2: "...استثناء من ذلك في حالة..."  ← Context lost!
```

#### Proposed Architecture: Small-to-Big Retrieval

```
┌─────────────────────────────────────────────────────────────────┐
│                     Document: قانون_التمويل.docx                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Chapter: الباب الأول - التراخيص                ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │           Article: المادة 5 - متطلبات رأس المال         │││
│  │  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ │││
│  │  │  │ Clause 1      │ │ Clause 2      │ │ Exception     │ │││
│  │  │  │ (Indexed)     │ │ (Indexed)     │ │ (Indexed)     │ │││
│  │  │  │ Vector: ✓     │ │ Vector: ✓     │ │ Vector: ✓     │ │││
│  │  │  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ │││
│  │  │          └─────────────────┼─────────────────┘          │││
│  │  │                            ▼                            │││
│  │  │               Parent Reference: article_5_id            │││
│  │  └─────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation Plan

**Phase 1: Hierarchical Parser**

```python
# New file: ingestion/hierarchical_chunker.py

@dataclass
class DocumentNode:
    """Hierarchical document node."""
    id: str
    content: str
    level: str  # "document" | "chapter" | "article" | "clause"
    parent_id: Optional[str]
    children_ids: List[str]
    metadata: Dict[str, Any]

class HierarchicalChunker:
    """
    Parse legal documents into hierarchical structure.
    
    Detection patterns:
    - الباب / الفصل → Chapter
    - المادة → Article
    - البند / الفقرة → Clause
    - أولاً، ثانياً → Numbered items
    """
    
    HIERARCHY_PATTERNS = {
        "chapter": [r"الباب\s+(\w+)", r"الفصل\s+(\w+)"],
        "article": [r"مادة\s*\(?\s*(\d+)\s*\)?", r"المادة\s+(\d+)"],
        "clause": [r"البند\s+(\d+)", r"الفقرة\s+(\w+)"],
        "item": [r"(أولاً|ثانياً|ثالثاً|رابعاً|خامساً)"],
    }
    
    def parse(self, text: str, source: str) -> List[DocumentNode]:
        """Parse document into hierarchical nodes."""
        # 1. Detect structure markers
        # 2. Build tree
        # 3. Return flattened nodes with parent references
```

**Phase 2: Dual-Index Storage**

```python
# Store in Qdrant with parent references

# Small chunks (for precise retrieval)
small_chunk = {
    "id": "clause_5_1_uuid",
    "text": "الحد الأدنى لرأس المال 50 مليون جنيه",
    "level": "clause",
    "parent_id": "article_5_uuid",
    "grandparent_id": "chapter_1_uuid",
    "root_id": "document_uuid",
}

# Parent documents (for context expansion)
parent_doc = {
    "id": "article_5_uuid",
    "text": "المادة 5 - متطلبات رأس المال\n1. الحد الأدنى...\n2. استثناءات...",
    "level": "article",
    "children_ids": ["clause_5_1_uuid", "clause_5_2_uuid"],
}
```

**Phase 3: Smart Retrieval**

```python
class ParentDocumentRetriever:
    """
    Retrieve small chunks, expand to parent context.
    """
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        # 1. Search small chunks (high precision)
        small_results = self.vector_store.search(
            query, k=k*2, filter={"level": ["clause", "item"]}
        )
        
        # 2. Get unique parent articles
        parent_ids = set(r["parent_id"] for r in small_results)
        
        # 3. Fetch full parent content
        parents = self.vector_store.get_by_ids(list(parent_ids))
        
        # 4. Return parents with matched clause highlighted
        return self._merge_with_highlights(parents, small_results)
```

#### Benefits
- ✅ High precision search on specific clauses
- ✅ Full article context for LLM
- ✅ Preserves legal document structure
- ✅ Enables "show me the full article" feature

#### Effort Estimate
- **Development**: 2-3 weeks
- **Testing**: 1 week
- **Priority**: **HIGH** (foundational improvement)

---

### 2. Semantic Metadata Extraction

#### Current Limitation
- No filtering by date, entity type, or document status
- Cannot answer "What applies to Banks in 2024?"

#### Proposed Architecture

```
Document Ingestion
       ↓
┌──────────────────────────────────────────────────┐
│           Metadata Extraction (Small LLM)         │
│                                                   │
│  Input: Document text                            │
│  Output: {                                       │
│    "law_status": "active",                       │
│    "effective_date": "2024-01-15",               │
│    "issuing_authority": "FRA",                   │
│    "entity_types": ["Bank", "Microfinance"],     │
│    "document_type": "regulation",                │
│    "topics": ["licensing", "capital", "branches"]│
│  }                                               │
└──────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────┐
│              Qdrant Storage                       │
│                                                   │
│  payload: {                                      │
│    "text": "...",                                │
│    "metadata": {                                 │
│      "law_status": "active",                     │
│      "effective_date": "2024-01-15",             │
│      "entity_types": ["Bank", "Microfinance"],   │
│      ...                                         │
│    }                                             │
│  }                                               │
└──────────────────────────────────────────────────┘
```

#### Implementation

```python
# ingestion/metadata_extractor.py

class MetadataExtractor:
    """Extract semantic metadata using LLM."""
    
    EXTRACTION_PROMPT = """
    حلل النص التنظيمي التالي واستخرج البيانات الوصفية:
    
    النص: {text}
    
    استخرج المعلومات التالية بصيغة JSON:
    {{
        "law_status": "active" أو "repealed" أو "amended",
        "effective_date": "YYYY-MM-DD" أو null,
        "amendment_date": "YYYY-MM-DD" أو null,
        "issuing_authority": "اسم الجهة المصدرة",
        "entity_types": ["قائمة أنواع الجهات الخاضعة"],
        "document_type": "regulation" أو "decision" أو "circular" أو "form",
        "topics": ["قائمة المواضيع الرئيسية"],
        "penalties_mentioned": true أو false,
        "capital_requirements_mentioned": true أو false
    }}
    """
    
    def extract(self, text: str) -> Dict[str, Any]:
        # Use small/fast LLM for extraction
        response = self.llm.generate(
            self.EXTRACTION_PROMPT.format(text=text[:4000])
        )
        return json.loads(response)
```

#### UI Filter Components

```python
# In app.py - Add filter dropdowns

with gr.Row():
    entity_filter = gr.Dropdown(
        choices=["الكل", "بنوك", "تمويل استهلاكي", "تمويل متناهي الصغر", "سمسرة", "تأمين"],
        value="الكل",
        label="🏢 نوع الجهة (Entity Type)"
    )
    doc_type_filter = gr.Dropdown(
        choices=["الكل", "لائحة", "قرار", "تعميم", "نموذج"],
        value="الكل",
        label="📄 نوع المستند (Document Type)"
    )
    date_filter = gr.Dropdown(
        choices=["الكل", "2024", "2023", "2022", "2021", "قبل 2021"],
        value="الكل",
        label="📅 السنة (Year)"
    )
```

#### Benefits
- ✅ Precise filtering by entity type
- ✅ Date-based queries ("What changed in 2024?")
- ✅ Document type filtering (forms vs regulations)
- ✅ Better retrieval precision

#### Effort Estimate
- **Development**: 1-2 weeks
- **Testing**: 1 week
- **Priority**: **HIGH** (addresses key user need)

---

### 3. Knowledge Graph / GraphRAG

#### Current Limitation
- Vector search misses conceptually related but semantically different content
- Cannot answer "What are ALL obligations of Microfinance companies?"

#### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Knowledge Graph (Neo4j/NetworkX)              │
│                                                                  │
│  ┌─────────────┐         REQUIRES          ┌─────────────────┐  │
│  │ Microfinance│─────────────────────────▶│ License         │  │
│  │ Company     │                           │                 │  │
│  └─────────────┘                           └────────┬────────┘  │
│         │                                           │           │
│         │ MUST_HAVE                                 │ NEEDS     │
│         ▼                                           ▼           │
│  ┌─────────────┐                           ┌─────────────────┐  │
│  │ Capital     │                           │ Documents       │  │
│  │ 50M EGP     │                           │ - Application   │  │
│  └─────────────┘                           │ - Articles      │  │
│         │                                  │ - Security Check│  │
│         │ EXCEPTION_IF                     └─────────────────┘  │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ Government  │                                               │
│  │ Ownership   │                                               │
│  └─────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation

**Phase 1: Graph Schema Definition**

```python
# rag_engine/knowledge_graph.py

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class EntityType(Enum):
    ENTITY = "entity"           # Bank, Microfinance, Insurance
    REQUIREMENT = "requirement" # License, Capital, Documents
    DOCUMENT = "document"       # Forms, Regulations
    CONDITION = "condition"     # Exceptions, Prerequisites
    PENALTY = "penalty"         # Fines, Revocations

class RelationType(Enum):
    REQUIRES = "REQUIRES"
    MUST_HAVE = "MUST_HAVE"
    NEEDS_DOCUMENT = "NEEDS_DOCUMENT"
    EXCEPTION_IF = "EXCEPTION_IF"
    SUBJECT_TO = "SUBJECT_TO"
    DEFINED_IN = "DEFINED_IN"
    AMENDS = "AMENDS"
    REPEALS = "REPEALS"

@dataclass
class GraphNode:
    id: str
    type: EntityType
    name: str
    name_ar: str
    attributes: Dict[str, Any]
    source_chunks: List[str]  # Links to vector store

@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relation: RelationType
    attributes: Dict[str, Any]
    source_chunk: str  # Where this relation was found
```

**Phase 2: Automatic Graph Construction**

```python
class GraphBuilder:
    """Build knowledge graph from document chunks."""
    
    EXTRACTION_PROMPT = """
    من النص التنظيمي التالي، استخرج الكيانات والعلاقات:
    
    النص: {text}
    
    استخرج بصيغة JSON:
    {{
        "entities": [
            {{"name": "...", "type": "entity|requirement|document|condition|penalty", "attributes": {{}}}}
        ],
        "relations": [
            {{"source": "...", "target": "...", "relation": "REQUIRES|MUST_HAVE|EXCEPTION_IF|..."}}
        ]
    }}
    
    أنواع العلاقات المتاحة:
    - REQUIRES: X يتطلب Y
    - MUST_HAVE: X يجب أن يمتلك Y
    - NEEDS_DOCUMENT: X يحتاج مستند Y
    - EXCEPTION_IF: استثناء من X إذا تحقق شرط Y
    - SUBJECT_TO: X خاضع لـ Y
    """
    
    def build_from_chunks(self, chunks: List[Dict]) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Extract entities and relations from all chunks."""
        all_nodes = []
        all_edges = []
        
        for chunk in chunks:
            extracted = self.llm.generate(
                self.EXTRACTION_PROMPT.format(text=chunk["text"])
            )
            nodes, edges = self._parse_extraction(extracted, chunk["id"])
            all_nodes.extend(nodes)
            all_edges.extend(edges)
        
        # Deduplicate and merge nodes
        return self._merge_nodes(all_nodes), all_edges
```

**Phase 3: Graph-Enhanced Retrieval**

```python
class GraphRAGRetriever:
    """Combine vector search with graph traversal."""
    
    def retrieve(self, query: str, k: int = 5) -> Dict[str, Any]:
        # 1. Vector search for initial nodes
        vector_results = self.vector_store.search(query, k=k)
        
        # 2. Extract entities from query
        query_entities = self.extract_entities(query)
        
        # 3. Graph traversal from matched entities
        graph_context = []
        for entity in query_entities:
            # Find related nodes (1-2 hops)
            related = self.graph.traverse(
                start=entity,
                max_hops=2,
                relations=["REQUIRES", "MUST_HAVE", "EXCEPTION_IF"]
            )
            graph_context.extend(related)
        
        # 4. Fetch source chunks for graph nodes
        graph_chunks = self.vector_store.get_by_ids(
            [node.source_chunks[0] for node in graph_context]
        )
        
        # 5. Merge and deduplicate
        return self._merge_results(vector_results, graph_chunks)
```

#### Example Query Flow

```
Query: "ما هي التزامات شركات التمويل متناهي الصغر؟"

1. Vector Search → Finds chunks mentioning "التمويل متناهي الصغر"

2. Entity Extraction → Identifies: "شركة التمويل متناهي الصغر"

3. Graph Traversal:
   Microfinance Company
   ├── REQUIRES → License (source: article_1)
   ├── MUST_HAVE → Capital 50M (source: article_5)
   ├── MUST_HAVE → Branch Manager (source: article_10)
   ├── SUBJECT_TO → Annual Audit (source: article_15)
   └── EXCEPTION_IF → Government Ownership (source: article_5_exception)

4. Fetch ALL related chunks → Complete context

5. LLM generates comprehensive answer with ALL obligations
```

#### Benefits
- ✅ Discovers related concepts not found by vector search
- ✅ Handles "global" questions (summarize all X)
- ✅ Explicit relationship tracking
- ✅ Explainable retrieval path

#### Effort Estimate
- **Development**: 4-6 weeks
- **Testing**: 2 weeks
- **Priority**: **MEDIUM** (high value but complex)

---

### 4. Agentic Workflow (ReAct Agent)

#### Current Limitation
- Single retrieval step cannot handle multi-hop reasoning
- Cannot answer: "Does Company X qualify for License Y given condition Z?"

#### Proposed Architecture: ReAct Agent

```
User Query: "هل تحتاج شركة تأمين برأس مال 30 مليون جنيه إلى موافقة مسبقة؟"

┌─────────────────────────────────────────────────────────────────┐
│                        ReAct Agent Loop                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Iteration 1:                                                    │
│   Thought: أحتاج معرفة الحد الأدنى لرأس المال لشركات التأمين    │
│   Action: retrieve("الحد الأدنى رأس المال شركات التأمين")       │
│   Observation: "الحد الأدنى 60 مليون جنيه (المادة 5)"           │
│                                                                  │
│ Iteration 2:                                                    │
│   Thought: رأس المال 30 مليون أقل من الحد الأدنى 60 مليون،      │
│            أحتاج معرفة هل هناك استثناءات                        │
│   Action: retrieve("استثناءات رأس المال شركات التأمين")         │
│   Observation: "لا توجد استثناءات للحد الأدنى"                  │
│                                                                  │
│ Iteration 3:                                                    │
│   Thought: لدي معلومات كافية للإجابة                           │
│   Action: answer()                                              │
│                                                                  │
│ Final Answer:                                                   │
│   "لا، شركة التأمين برأس مال 30 مليون جنيه لا تستوفي الحد       │
│    الأدنى المطلوب وهو 60 مليون جنيه، ولا توجد استثناءات."      │
│                                                                  │
│   📌 المصادر:                                                   │
│   - المادة 5: الحد الأدنى لرأس المال                           │
│   - المادة 5 فقرة 3: عدم وجود استثناءات                        │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation

```python
# rag_engine/react_agent.py

from enum import Enum
from typing import List, Tuple, Optional

class ActionType(Enum):
    RETRIEVE = "retrieve"
    CALCULATE = "calculate"
    COMPARE = "compare"
    ANSWER = "answer"

@dataclass
class AgentStep:
    thought: str
    action: ActionType
    action_input: str
    observation: str

class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent for multi-hop legal reasoning.
    """
    
    SYSTEM_PROMPT = """
    أنت وكيل ذكي متخصص في تحليل الأسئلة القانونية والتنظيمية.
    
    لكل سؤال، اتبع هذا النمط:
    
    Thought: [تحليلك للموقف وما تحتاج معرفته]
    Action: [retrieve|calculate|compare|answer]
    Action Input: [استعلام البحث أو المعلومات]
    
    الإجراءات المتاحة:
    - retrieve: البحث عن معلومات في المستندات
    - calculate: حساب رياضي بسيط
    - compare: مقارنة قيمتين
    - answer: تقديم الإجابة النهائية (فقط عندما تملك معلومات كافية)
    
    استمر في التفكير والبحث حتى تجمع معلومات كافية للإجابة.
    لا تخمن - إذا لم تجد المعلومة، قل ذلك.
    """
    
    def __init__(self, retriever, llm, max_iterations: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.max_iterations = max_iterations
    
    def run(self, query: str) -> Tuple[str, List[AgentStep]]:
        """Execute agent loop."""
        steps = []
        context = f"السؤال: {query}\n\n"
        
        for i in range(self.max_iterations):
            # Get next action from LLM
            response = self.llm.generate(
                self.SYSTEM_PROMPT + context
            )
            
            # Parse thought, action, action_input
            thought, action, action_input = self._parse_response(response)
            
            if action == ActionType.ANSWER:
                # Final answer
                return action_input, steps
            
            # Execute action
            observation = self._execute_action(action, action_input)
            
            # Record step
            step = AgentStep(thought, action, action_input, observation)
            steps.append(step)
            
            # Update context
            context += f"""
Thought: {thought}
Action: {action.value}
Action Input: {action_input}
Observation: {observation}

"""
        
        # Max iterations reached
        return self._synthesize_answer(query, steps), steps
    
    def _execute_action(self, action: ActionType, input: str) -> str:
        if action == ActionType.RETRIEVE:
            results = self.retriever.retrieve(input, k=3)
            return self._format_retrieval(results)
        elif action == ActionType.CALCULATE:
            return self._safe_calculate(input)
        elif action == ActionType.COMPARE:
            return self._compare_values(input)
        return ""
```

#### Benefits
- ✅ Multi-hop reasoning for complex questions
- ✅ Step-by-step transparency
- ✅ Reduces hallucination through verification
- ✅ Handles conditional logic

#### Effort Estimate
- **Development**: 2-3 weeks
- **Testing**: 2 weeks
- **Priority**: **HIGH** (significant capability upgrade)

---

### 5. Clickable Sources with Highlighting

#### Current Limitation
- Sources are displayed but not interactive
- User cannot quickly verify the cited text

#### Proposed UI Enhancement

```
┌─────────────────────────────────────────────────────────────────┐
│  Answer:                                                         │
│                                                                  │
│  الحد الأدنى لرأس المال هو 50 مليون جنيه [1].                    │
│  يجب تقديم طلب الترخيص إلى الهيئة [2].                          │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  [1] ← Clickable                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 📄 نموذج-قيد-فرع-لشركة-تمويل-الاستهلاكى.docx             │   │
│  │                                                          │   │
│  │ المادة 5:                                                │   │
│  │ "يشترط لقيد فرع شركة التمويل الاستهلاكي أن يكون رأس    │   │
│  │  المال المدفوع لا يقل عن ██ 50 مليون جنيه مصري ██"      │   │
│  │                         ↑ Highlighted match              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation

```python
# In app.py

def create_source_html(sources: List[Dict], answer: str) -> str:
    """Create HTML with clickable source references."""
    
    # Extract citation numbers from answer
    citations = re.findall(r'\[(\d+)\]', answer)
    
    html_parts = ['<div class="sources-container">']
    
    for i, source in enumerate(sources, 1):
        # Find the matched text in the answer
        matched_text = find_matched_phrase(answer, source["content"])
        
        # Highlight matched text in source
        highlighted_content = highlight_text(
            source["content"],
            matched_text,
            highlight_class="bg-yellow-200"
        )
        
        html_parts.append(f'''
        <div class="source-card" id="source-{i}">
            <div class="source-header" onclick="toggleSource({i})">
                <span class="source-number">[{i}]</span>
                <span class="source-name">📄 {source["source"]}</span>
                <span class="relevance-badge">{source["score"]*100:.0f}%</span>
            </div>
            <div class="source-content" id="source-content-{i}" style="display:none;">
                <div class="source-text" dir="rtl">
                    {highlighted_content}
                </div>
            </div>
        </div>
        ''')
    
    html_parts.append('</div>')
    
    # Add JavaScript for interactivity
    html_parts.append('''
    <script>
    function toggleSource(id) {
        const content = document.getElementById('source-content-' + id);
        content.style.display = content.style.display === 'none' ? 'block' : 'none';
    }
    
    function scrollToSource(id) {
        const element = document.getElementById('source-' + id);
        element.scrollIntoView({ behavior: 'smooth' });
        toggleSource(id);
    }
    </script>
    ''')
    
    return '\n'.join(html_parts)

def format_answer_with_clickable_refs(answer: str) -> str:
    """Make citation numbers clickable."""
    return re.sub(
        r'\[(\d+)\]',
        r'<a href="#" onclick="scrollToSource(\1); return false;" class="citation-link">[\1]</a>',
        answer
    )
```

#### Benefits
- ✅ Quick verification of sources
- ✅ Highlighted matched text
- ✅ Better user experience
- ✅ Increased trust in system

#### Effort Estimate
- **Development**: 1 week
- **Testing**: 2-3 days
- **Priority**: **HIGH** (easy win, high impact)

---

## Part 2: Evaluation Framework

### 1. Golden Dataset Creation

#### Dataset Structure

```json
{
    "id": "q_001",
    "question": "ما هو الحد الأدنى لرأس المال لشركة تمويل استهلاكي؟",
    "question_type": "factual",
    "language": "ar",
    "ground_truth_answer": "الحد الأدنى لرأس المال هو 50 مليون جنيه مصري.",
    "relevant_chunks": ["chunk_id_1", "chunk_id_2"],
    "relevant_docs": ["نموذج-قيد-فرع-لشركة-تمويل-الاستهلاكى.docx"],
    "difficulty": "easy",
    "requires_multi_hop": false,
    "metadata": {
        "entity_type": "consumer_finance",
        "topic": "capital_requirements"
    }
}
```

#### Question Types

| Type | Description | Example |
|------|-------------|---------|
| **factual** | Single fact retrieval | ما هو الحد الأدنى...؟ |
| **comparison** | Compare two concepts | ما الفرق بين X و Y؟ |
| **procedural** | Steps/process | ما هي خطوات...؟ |
| **conditional** | If-then logic | هل يجوز X إذا كان Y؟ |
| **aggregation** | Summarize multiple items | ما هي جميع التزامات...؟ |
| **temporal** | Date-based | ما الذي تغير في 2024؟ |
| **negation** | What is NOT allowed | ما هي المحظورات؟ |

#### Dataset Generation Strategy

```python
# evaluation/dataset_generator.py

class GoldenDatasetGenerator:
    """Generate evaluation dataset from documents."""
    
    QUESTION_GENERATION_PROMPT = """
    أنت خبير في إنشاء أسئلة تقييم لأنظمة الذكاء الاصطناعي.
    
    بناءً على النص التالي، أنشئ {n} أسئلة متنوعة:
    
    النص: {text}
    
    لكل سؤال، قدم:
    1. السؤال
    2. نوع السؤال (factual/comparison/procedural/conditional/aggregation)
    3. الإجابة الصحيحة
    4. مستوى الصعوبة (easy/medium/hard)
    5. هل يتطلب استدلال متعدد الخطوات (true/false)
    
    تأكد من تنويع أنواع الأسئلة.
    """
    
    def generate_from_chunks(self, chunks: List[Dict], questions_per_chunk: int = 3) -> List[Dict]:
        """Generate questions from document chunks."""
        dataset = []
        
        for chunk in chunks:
            questions = self.llm.generate(
                self.QUESTION_GENERATION_PROMPT.format(
                    text=chunk["text"],
                    n=questions_per_chunk
                )
            )
            
            for q in questions:
                q["relevant_chunks"] = [chunk["id"]]
                q["relevant_docs"] = [chunk["source"]]
                dataset.append(q)
        
        return dataset
    
    def generate_comparison_questions(self, entity_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Generate comparison questions for entity pairs."""
        # e.g., ("تمويل استهلاكي", "تمويل متناهي الصغر")
        pass
    
    def generate_from_user_logs(self, feedback_file: str) -> List[Dict]:
        """Extract high-quality questions from user feedback."""
        # Use questions that got positive feedback
        pass
```

#### Target Dataset Size

| Category | Count | Priority |
|----------|-------|----------|
| Factual questions | 40 | High |
| Comparison questions | 20 | High |
| Procedural questions | 15 | Medium |
| Conditional questions | 10 | High |
| Aggregation questions | 10 | Medium |
| Edge cases (not in docs) | 5 | High |
| **Total** | **100** | |

---

### 2. Evaluation Metrics Implementation

#### A. Retrieval Metrics

```python
# evaluation/retrieval_metrics.py

from typing import List, Dict
import numpy as np

class RetrievalEvaluator:
    """Evaluate retrieval quality."""
    
    def hit_rate_at_k(self, predictions: List[List[str]], ground_truth: List[List[str]], k: int = 5) -> float:
        """
        Percentage of queries where at least one relevant doc is in top-k.
        
        Args:
            predictions: List of retrieved chunk IDs for each query
            ground_truth: List of relevant chunk IDs for each query
        """
        hits = 0
        for pred, truth in zip(predictions, ground_truth):
            if any(p in truth for p in pred[:k]):
                hits += 1
        return hits / len(predictions)
    
    def mrr(self, predictions: List[List[str]], ground_truth: List[List[str]]) -> float:
        """
        Mean Reciprocal Rank - measures how high the first relevant result is.
        """
        rr_sum = 0
        for pred, truth in zip(predictions, ground_truth):
            for rank, p in enumerate(pred, 1):
                if p in truth:
                    rr_sum += 1 / rank
                    break
        return rr_sum / len(predictions)
    
    def precision_at_k(self, predictions: List[List[str]], ground_truth: List[List[str]], k: int = 5) -> float:
        """
        Percentage of retrieved docs that are relevant.
        """
        precisions = []
        for pred, truth in zip(predictions, ground_truth):
            relevant = sum(1 for p in pred[:k] if p in truth)
            precisions.append(relevant / k)
        return np.mean(precisions)
    
    def recall_at_k(self, predictions: List[List[str]], ground_truth: List[List[str]], k: int = 5) -> float:
        """
        Percentage of relevant docs that were retrieved.
        """
        recalls = []
        for pred, truth in zip(predictions, ground_truth):
            if len(truth) == 0:
                continue
            relevant = sum(1 for p in pred[:k] if p in truth)
            recalls.append(relevant / len(truth))
        return np.mean(recalls)
```

#### B. Generation Metrics (Ragas Integration)

```python
# evaluation/generation_metrics.py

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset

class GenerationEvaluator:
    """Evaluate LLM generation quality using Ragas."""
    
    def __init__(self, llm=None, embeddings=None):
        self.llm = llm
        self.embeddings = embeddings
    
    def evaluate_batch(self, data: List[Dict]) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.
        
        Args:
            data: List of dicts with keys:
                - question: str
                - answer: str (system generated)
                - contexts: List[str] (retrieved chunks)
                - ground_truth: str (expected answer)
        """
        # Convert to Ragas dataset format
        dataset = Dataset.from_dict({
            'question': [d['question'] for d in data],
            'answer': [d['answer'] for d in data],
            'contexts': [d['contexts'] for d in data],
            'ground_truth': [d['ground_truth'] for d in data],
        })
        
        # Run evaluation
        results = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,       # Does answer stick to context?
                answer_relevancy,   # Does answer address the question?
                context_precision,  # Are retrieved contexts relevant?
                context_recall,     # Did we retrieve all needed context?
                answer_correctness, # Is the answer correct?
            ],
            llm=self.llm,
            embeddings=self.embeddings,
        )
        
        return {
            'faithfulness': results['faithfulness'],
            'answer_relevancy': results['answer_relevancy'],
            'context_precision': results['context_precision'],
            'context_recall': results['context_recall'],
            'answer_correctness': results['answer_correctness'],
        }
    
    def evaluate_single(self, question: str, answer: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
        """Evaluate a single prediction."""
        return self.evaluate_batch([{
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'ground_truth': ground_truth,
        }])
```

#### C. Custom Arabic Metrics

```python
# evaluation/arabic_metrics.py

class ArabicEvaluator:
    """Arabic-specific evaluation metrics."""
    
    def citation_accuracy(self, answer: str, sources: List[Dict]) -> float:
        """
        Check if citations in answer match actual sources.
        """
        # Extract citation numbers from answer
        citations = re.findall(r'\[(\d+)\]', answer)
        
        # Check if each citation has corresponding source
        valid_citations = 0
        for cite_num in citations:
            idx = int(cite_num) - 1
            if idx < len(sources):
                # Check if cited text appears in source
                # ... fuzzy matching logic
                valid_citations += 1
        
        return valid_citations / max(len(citations), 1)
    
    def article_reference_accuracy(self, answer: str, ground_truth_articles: List[str]) -> float:
        """
        Check if answer references the correct articles/clauses.
        """
        # Extract article references from answer
        # e.g., "المادة 5", "البند 3"
        found_articles = re.findall(r'(?:المادة|البند|الفقرة)\s+(\d+)', answer)
        
        correct = sum(1 for a in found_articles if a in ground_truth_articles)
        return correct / max(len(ground_truth_articles), 1)
    
    def anti_hallucination_score(self, answer: str, contexts: List[str]) -> float:
        """
        Check if answer contains information not in contexts.
        Uses embedding similarity to detect potential hallucinations.
        """
        # Split answer into claims
        claims = self._extract_claims(answer)
        
        grounded_claims = 0
        for claim in claims:
            # Check if claim is supported by any context
            max_similarity = max(
                self._semantic_similarity(claim, ctx)
                for ctx in contexts
            )
            if max_similarity > 0.7:
                grounded_claims += 1
        
        return grounded_claims / max(len(claims), 1)
```

---

### 3. Evaluation Pipeline

```python
# evaluation/pipeline.py

class EvaluationPipeline:
    """End-to-end evaluation pipeline."""
    
    def __init__(self, rag_system, golden_dataset_path: str):
        self.system = rag_system
        self.dataset = self._load_dataset(golden_dataset_path)
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.arabic_evaluator = ArabicEvaluator()
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation and generate report."""
        
        # 1. Run system on all questions
        predictions = []
        for item in self.dataset:
            result = self.system.query_with_sources(item['question'])
            predictions.append({
                'question': item['question'],
                'answer': result['answer'],
                'contexts': [s['content'] for s in result['sources']],
                'retrieved_ids': [s['id'] for s in result['sources']],
                'ground_truth': item['ground_truth_answer'],
                'relevant_ids': item['relevant_chunks'],
            })
        
        # 2. Retrieval metrics
        retrieval_metrics = {
            'hit_rate@5': self.retrieval_evaluator.hit_rate_at_k(
                [p['retrieved_ids'] for p in predictions],
                [p['relevant_ids'] for p in predictions],
                k=5
            ),
            'mrr': self.retrieval_evaluator.mrr(
                [p['retrieved_ids'] for p in predictions],
                [p['relevant_ids'] for p in predictions]
            ),
            'precision@5': self.retrieval_evaluator.precision_at_k(
                [p['retrieved_ids'] for p in predictions],
                [p['relevant_ids'] for p in predictions],
                k=5
            ),
        }
        
        # 3. Generation metrics (Ragas)
        generation_metrics = self.generation_evaluator.evaluate_batch(predictions)
        
        # 4. Arabic-specific metrics
        arabic_metrics = {
            'citation_accuracy': np.mean([
                self.arabic_evaluator.citation_accuracy(p['answer'], p['contexts'])
                for p in predictions
            ]),
            'anti_hallucination': np.mean([
                self.arabic_evaluator.anti_hallucination_score(p['answer'], p['contexts'])
                for p in predictions
            ]),
        }
        
        # 5. Aggregate results
        return {
            'retrieval': retrieval_metrics,
            'generation': generation_metrics,
            'arabic': arabic_metrics,
            'overall_score': self._calculate_overall_score(
                retrieval_metrics, generation_metrics, arabic_metrics
            ),
            'detailed_results': predictions,
        }
    
    def generate_report(self, results: Dict) -> str:
        """Generate markdown evaluation report."""
        return f"""
# FRA RAG System Evaluation Report

## Date: {datetime.now().strftime('%Y-%m-%d')}

## Summary
- **Overall Score**: {results['overall_score']:.2%}

## Retrieval Metrics
| Metric | Score |
|--------|-------|
| Hit Rate@5 | {results['retrieval']['hit_rate@5']:.2%} |
| MRR | {results['retrieval']['mrr']:.3f} |
| Precision@5 | {results['retrieval']['precision@5']:.2%} |

## Generation Metrics
| Metric | Score |
|--------|-------|
| Faithfulness | {results['generation']['faithfulness']:.2%} |
| Answer Relevancy | {results['generation']['answer_relevancy']:.2%} |
| Context Precision | {results['generation']['context_precision']:.2%} |

## Arabic-Specific Metrics
| Metric | Score |
|--------|-------|
| Citation Accuracy | {results['arabic']['citation_accuracy']:.2%} |
| Anti-Hallucination | {results['arabic']['anti_hallucination']:.2%} |
        """
```

---

## Part 3: Implementation Roadmap

### Phase 1: Quick Wins (2-3 weeks)
| Feature | Effort | Impact | Priority |
|---------|--------|--------|----------|
| Clickable sources with highlighting | 1 week | High | ✅ DO FIRST |
| Semantic metadata extraction | 1-2 weeks | High | ✅ DO FIRST |
| Basic evaluation framework | 1 week | High | ✅ DO FIRST |
| New document ingestion | 1 day | Medium | ✅ DO NOW |

### Phase 2: Core Improvements (4-6 weeks)
| Feature | Effort | Impact | Priority |
|---------|--------|--------|----------|
| Hierarchical chunking (parent-document) | 2-3 weeks | Very High | ✅ |
| ReAct agent for multi-hop | 2-3 weeks | Very High | ✅ |
| Golden dataset creation (100 QA pairs) | 2 weeks | High | ✅ |

### Phase 3: Advanced Features (6-8 weeks)
| Feature | Effort | Impact | Priority |
|---------|--------|--------|----------|
| Knowledge Graph / GraphRAG | 4-6 weeks | High | ⚠️ Complex |
| Full Ragas integration | 2 weeks | High | ✅ |
| Automated regression testing | 1 week | Medium | ✅ |

---

## Additional Enhancement Suggestions

### 1. Query Understanding Enhancement

```python
# Add query classification for better routing

class QueryClassifier:
    """Classify query intent for optimized handling."""
    
    INTENTS = {
        "definition": ["ما هو", "ما المقصود", "عرف"],
        "requirement": ["ما هي المتطلبات", "ما هي الشروط", "ما يلزم"],
        "procedure": ["كيف", "ما هي الخطوات", "ما هي الإجراءات"],
        "comparison": ["ما الفرق", "قارن", "أيهما"],
        "eligibility": ["هل يجوز", "هل يحق", "هل يمكن"],
        "penalty": ["ما هي العقوبة", "ما هي الغرامة"],
        "timeline": ["متى", "ما هي المدة", "كم يستغرق"],
    }
```

### 2. Confidence Scoring

```python
# Add confidence score to answers

def calculate_confidence(retrieval_scores: List[float], answer: str, contexts: List[str]) -> float:
    """
    Calculate confidence score for answer.
    
    Factors:
    1. Retrieval scores (are results highly relevant?)
    2. Citation density (is answer well-supported?)
    3. Context coverage (did we find enough context?)
    """
    retrieval_confidence = np.mean([s for s in retrieval_scores if s > 0.5])
    citation_count = len(re.findall(r'\[(\d+)\]', answer))
    coverage = min(citation_count / 3, 1.0)  # Expect at least 3 citations
    
    return (retrieval_confidence * 0.5 + coverage * 0.5)
```

### 3. Feedback-Driven Improvement

```python
# Use feedback to improve system

class FeedbackAnalyzer:
    """Analyze user feedback for improvement opportunities."""
    
    def identify_weak_topics(self, feedback_data: List[Dict]) -> List[str]:
        """Find topics with low satisfaction."""
        # Group by detected topic
        # Calculate negative feedback rate per topic
        # Return topics needing improvement
        pass
    
    def generate_improvement_suggestions(self) -> List[str]:
        """Generate actionable improvement suggestions."""
        pass
```

---

## Conclusion

This proposal outlines a comprehensive path to transform the FRA RAG system from a solid retrieval system into an advanced **Legal AI Assistant**. The key improvements are:

1. **Hierarchical Chunking**: Preserve legal document structure
2. **Semantic Metadata**: Enable powerful filtering
3. **ReAct Agent**: Multi-hop reasoning for complex questions
4. **Knowledge Graph**: Discover related concepts (future phase)
5. **Evaluation Framework**: Measure and improve systematically
6. **UI Enhancements**: Clickable sources with highlighting

**Recommended Starting Point**: 
1. Ingest new documents
2. Implement clickable sources (quick win)
3. Add metadata extraction
4. Set up basic evaluation
5. Implement hierarchical chunking
6. Add ReAct agent

Would you like me to begin implementing any of these features?
