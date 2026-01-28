"""
Golden Dataset for FRA RAG System Evaluation.

Provides structures and utilities for creating and managing
evaluation datasets with question-answer pairs and ground truth.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from loguru import logger


@dataclass
class EvaluationItem:
    """A single evaluation item in the golden dataset."""
    id: str
    question: str
    question_type: str  # factual, comparison, procedural, conditional, aggregation, temporal, negation
    language: str  # ar, en
    ground_truth_answer: str
    relevant_chunks: List[str] = field(default_factory=list)  # Chunk IDs
    relevant_docs: List[str] = field(default_factory=list)  # Document names
    ground_truth_articles: List[str] = field(default_factory=list)  # Article numbers
    difficulty: str = "medium"  # easy, medium, hard
    requires_multi_hop: bool = False
    entity_type: str = ""  # consumer_finance, microfinance, insurance, etc.
    topic: str = ""  # licensing, capital, branches, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationItem":
        return cls(**data)


@dataclass
class GoldenDataset:
    """
    Golden dataset containing evaluation items.
    
    A golden dataset is a curated set of question-answer pairs
    with ground truth for evaluating RAG system performance.
    """
    name: str
    version: str
    created_at: str
    items: List[EvaluationItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.items)
    
    def add_item(self, item: EvaluationItem):
        """Add an evaluation item."""
        self.items.append(item)
    
    def get_by_type(self, question_type: str) -> List[EvaluationItem]:
        """Get items by question type."""
        return [item for item in self.items if item.question_type == question_type]
    
    def get_by_difficulty(self, difficulty: str) -> List[EvaluationItem]:
        """Get items by difficulty."""
        return [item for item in self.items if item.difficulty == difficulty]
    
    def get_by_entity(self, entity_type: str) -> List[EvaluationItem]:
        """Get items by entity type."""
        return [item for item in self.items if item.entity_type == entity_type]
    
    def get_multi_hop(self) -> List[EvaluationItem]:
        """Get items requiring multi-hop reasoning."""
        return [item for item in self.items if item.requires_multi_hop]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "total_items": len(self.items),
            "by_type": {
                qtype: len(self.get_by_type(qtype))
                for qtype in set(item.question_type for item in self.items)
            },
            "by_difficulty": {
                diff: len(self.get_by_difficulty(diff))
                for diff in set(item.difficulty for item in self.items)
            },
            "by_entity": {
                entity: len(self.get_by_entity(entity))
                for entity in set(item.entity_type for item in self.items) if entity
            },
            "multi_hop_count": len(self.get_multi_hop()),
            "languages": list(set(item.language for item in self.items)),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "items": [item.to_dict() for item in self.items],
            "metadata": self.metadata,
            "statistics": self.get_statistics(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenDataset":
        items = [EvaluationItem.from_dict(item) for item in data.get("items", [])]
        return cls(
            name=data["name"],
            version=data["version"],
            created_at=data["created_at"],
            items=items,
            metadata=data.get("metadata", {}),
        )
    
    def save(self, path: Path):
        """Save dataset to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Dataset saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "GoldenDataset":
        """Load dataset from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Dataset loaded from {path}")
        return cls.from_dict(data)


class DatasetGenerator:
    """
    Generate evaluation datasets from documents.
    
    Methods:
    - Generate from document chunks using LLM
    - Generate from user query logs
    - Manual creation with templates
    """
    
    # Question templates for different types
    QUESTION_TEMPLATES = {
        "factual": [
            "ما هو {concept}؟",
            "ما هي {concept}؟",
            "ما المقصود بـ {concept}؟",
            "ما هو الحد الأدنى لـ {concept}؟",
            "كم {concept}؟",
        ],
        "comparison": [
            "ما الفرق بين {concept_a} و {concept_b}؟",
            "قارن بين {concept_a} و {concept_b}",
            "ما هي أوجه الاختلاف بين {concept_a} و {concept_b}؟",
        ],
        "procedural": [
            "ما هي خطوات {process}؟",
            "كيف يتم {process}؟",
            "ما هي إجراءات {process}؟",
            "ما هي المستندات المطلوبة لـ {process}؟",
        ],
        "conditional": [
            "هل يجوز {action} في حالة {condition}؟",
            "متى يمكن {action}؟",
            "ما هي شروط {action}؟",
            "هل يحق لـ {entity} {action}؟",
        ],
        "aggregation": [
            "ما هي جميع {items}؟",
            "اذكر كل {items}",
            "ما هي قائمة {items}؟",
        ],
        "temporal": [
            "ما الذي تغير في {date}؟",
            "ما هي المتطلبات الجديدة في {date}؟",
            "متى تم تعديل {concept}؟",
        ],
        "negation": [
            "ما هي المحظورات في {context}؟",
            "ما لا يجوز في {context}؟",
            "ما هي القيود على {entity}؟",
        ],
    }
    
    # Entity types for FRA domain
    ENTITY_TYPES = [
        ("تمويل_استهلاكي", "consumer_finance"),
        ("تمويل_متناهي_الصغر", "microfinance"),
        ("تأجير_تمويلي", "leasing"),
        ("سمسرة", "brokerage"),
        ("تأمين", "insurance"),
        ("توريق", "securitization"),
    ]
    
    # Topics
    TOPICS = [
        ("ترخيص", "licensing"),
        ("رأس_المال", "capital"),
        ("فروع", "branches"),
        ("غلق", "closure"),
        ("نقل", "transfer"),
        ("سندات", "bonds"),
        ("مدير", "management"),
    ]
    
    def __init__(self, llm_client=None):
        """
        Initialize dataset generator.
        
        Args:
            llm_client: Optional LLM for question generation
        """
        self.llm_client = llm_client
        logger.info("DatasetGenerator initialized")
    
    def create_empty_dataset(self, name: str, version: str = "1.0") -> GoldenDataset:
        """Create an empty dataset."""
        return GoldenDataset(
            name=name,
            version=version,
            created_at=datetime.now().isoformat(),
        )
    
    def generate_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        questions_per_chunk: int = 2,
        dataset_name: str = "auto_generated",
    ) -> GoldenDataset:
        """
        Generate evaluation items from document chunks using LLM.
        
        Args:
            chunks: List of document chunks with text and metadata
            questions_per_chunk: Number of questions to generate per chunk
            dataset_name: Name for the dataset
            
        Returns:
            GoldenDataset with generated items
        """
        dataset = self.create_empty_dataset(dataset_name)
        
        for chunk in chunks:
            text = chunk.get("text", chunk.get("content", ""))
            source = chunk.get("source", chunk.get("metadata", {}).get("source", "unknown"))
            
            if len(text) < 100:
                continue
            
            # Generate questions using LLM
            if self.llm_client:
                items = self._generate_questions_llm(text, source, questions_per_chunk)
            else:
                items = self._generate_questions_template(text, source)
            
            for item in items:
                dataset.add_item(item)
        
        logger.info(f"Generated {len(dataset)} evaluation items from {len(chunks)} chunks")
        return dataset
    
    def generate_from_user_logs(
        self,
        feedback_file: Path,
        filter_positive: bool = True,
    ) -> GoldenDataset:
        """
        Generate evaluation items from user query logs.
        
        Uses queries that received positive feedback as gold standard.
        
        Args:
            feedback_file: Path to feedback JSON file
            filter_positive: Only use positively rated queries
            
        Returns:
            GoldenDataset with items from logs
        """
        dataset = self.create_empty_dataset("user_queries")
        
        if not feedback_file.exists():
            logger.warning(f"Feedback file not found: {feedback_file}")
            return dataset
        
        with open(feedback_file, "r", encoding="utf-8") as f:
            feedback_data = json.load(f)
        
        for entry in feedback_data:
            if filter_positive and entry.get("feedback") != "positive":
                continue
            
            item = EvaluationItem(
                id=str(uuid.uuid4()),
                question=entry.get("query", ""),
                question_type="user_query",
                language="ar" if any('\u0600' <= c <= '\u06FF' for c in entry.get("query", "")) else "en",
                ground_truth_answer=entry.get("answer", ""),
                difficulty="medium",
                metadata={"source": "user_feedback", "timestamp": entry.get("timestamp")},
            )
            dataset.add_item(item)
        
        logger.info(f"Generated {len(dataset)} items from user logs")
        return dataset
    
    def create_manual_item(
        self,
        question: str,
        answer: str,
        question_type: str = "factual",
        language: str = "ar",
        relevant_docs: List[str] = None,
        ground_truth_articles: List[str] = None,
        difficulty: str = "medium",
        requires_multi_hop: bool = False,
        entity_type: str = "",
        topic: str = "",
    ) -> EvaluationItem:
        """Create a manual evaluation item."""
        return EvaluationItem(
            id=str(uuid.uuid4()),
            question=question,
            question_type=question_type,
            language=language,
            ground_truth_answer=answer,
            relevant_docs=relevant_docs or [],
            ground_truth_articles=ground_truth_articles or [],
            difficulty=difficulty,
            requires_multi_hop=requires_multi_hop,
            entity_type=entity_type,
            topic=topic,
        )
    
    def create_starter_dataset(self) -> GoldenDataset:
        """
        Create a starter dataset with common FRA regulatory questions.
        
        Returns a dataset with manually curated questions covering
        different question types and entity types.
        """
        dataset = self.create_empty_dataset("fra_starter", version="1.0")
        
        # Factual questions
        factual_items = [
            {
                "question": "ما هو الحد الأدنى لرأس المال لشركة تمويل استهلاكي؟",
                "answer": "الحد الأدنى لرأس المال لشركة التمويل الاستهلاكي هو 50 مليون جنيه مصري.",
                "entity_type": "تمويل_استهلاكي",
                "topic": "رأس_المال",
                "difficulty": "easy",
            },
            {
                "question": "ما هي المستندات المطلوبة لإصدار سندات التوريق؟",
                "answer": "تشمل المستندات المطلوبة: نشرة الاكتتاب، تقرير مدير الإصدار، تقييم الضمانات، وموافقة الهيئة.",
                "entity_type": "توريق",
                "topic": "سندات",
                "difficulty": "medium",
            },
            {
                "question": "ما هي إجراءات غلق فرع لشركة سمسرة؟",
                "answer": "يجب تقديم طلب غلق الفرع مرفقاً به موافقة مجلس الإدارة وخطة نقل العملاء.",
                "entity_type": "سمسرة",
                "topic": "غلق",
                "difficulty": "medium",
            },
        ]
        
        for item_data in factual_items:
            item = self.create_manual_item(
                question=item_data["question"],
                answer=item_data["answer"],
                question_type="factual",
                entity_type=item_data["entity_type"],
                topic=item_data["topic"],
                difficulty=item_data["difficulty"],
            )
            dataset.add_item(item)
        
        # Comparison questions
        comparison_items = [
            {
                "question": "ما الفرق بين متطلبات قيد فرع تمويل استهلاكي وفرع تمويل متناهي الصغر؟",
                "answer": "يختلف الحد الأدنى لرأس المال والمستندات المطلوبة بين النوعين.",
                "requires_multi_hop": True,
                "difficulty": "hard",
            },
        ]
        
        for item_data in comparison_items:
            item = self.create_manual_item(
                question=item_data["question"],
                answer=item_data["answer"],
                question_type="comparison",
                requires_multi_hop=item_data.get("requires_multi_hop", False),
                difficulty=item_data.get("difficulty", "medium"),
            )
            dataset.add_item(item)
        
        # Conditional questions
        conditional_items = [
            {
                "question": "هل يجوز لشركة التأمين فتح فرع جديد بدون موافقة الهيئة؟",
                "answer": "لا، يجب الحصول على موافقة الهيئة العامة للرقابة المالية قبل فتح أي فرع جديد.",
                "entity_type": "تأمين",
                "topic": "فروع",
            },
        ]
        
        for item_data in conditional_items:
            item = self.create_manual_item(
                question=item_data["question"],
                answer=item_data["answer"],
                question_type="conditional",
                entity_type=item_data.get("entity_type", ""),
                topic=item_data.get("topic", ""),
            )
            dataset.add_item(item)
        
        # Anti-hallucination test (question with no answer in docs)
        edge_case_items = [
            {
                "question": "ما هي عقوبة تجاوز سرعة 120 كم/ساعة؟",
                "answer": "لا توجد معلومات في المستندات المتاحة حول هذا الموضوع.",
                "question_type": "edge_case",
                "difficulty": "hard",
            },
        ]
        
        for item_data in edge_case_items:
            item = self.create_manual_item(
                question=item_data["question"],
                answer=item_data["answer"],
                question_type=item_data.get("question_type", "factual"),
                difficulty=item_data.get("difficulty", "medium"),
            )
            dataset.add_item(item)
        
        logger.info(f"Created starter dataset with {len(dataset)} items")
        return dataset
    
    def _generate_questions_llm(
        self,
        text: str,
        source: str,
        num_questions: int,
    ) -> List[EvaluationItem]:
        """Generate questions using LLM."""
        if not self.llm_client:
            return []
        
        prompt = f"""
أنت خبير في إنشاء أسئلة تقييم لأنظمة الذكاء الاصطناعي.

بناءً على النص التالي، أنشئ {num_questions} أسئلة متنوعة:

النص:
{text[:2000]}

لكل سؤال، قدم:
1. السؤال
2. الإجابة الصحيحة
3. نوع السؤال (factual/comparison/procedural/conditional)
4. مستوى الصعوبة (easy/medium/hard)

قدم الإجابة بصيغة JSON.
"""
        
        try:
            # Call LLM - simplified for now
            # In production, parse LLM response
            pass
        except Exception as e:
            logger.error(f"LLM question generation failed: {e}")
        
        return []
    
    def _generate_questions_template(
        self,
        text: str,
        source: str,
    ) -> List[EvaluationItem]:
        """Generate questions using templates (fallback)."""
        import re
        
        items = []
        
        # Extract potential concepts
        # Look for numbers with units
        capital_match = re.search(r'(\d+)\s*(مليون|ألف)\s*جنيه', text)
        if capital_match:
            entity_match = re.search(r'(تمويل\s*استهلاكي|تمويل\s*متناهي|سمسرة|تأمين)', text)
            entity = entity_match.group(1) if entity_match else "الشركة"
            
            item = EvaluationItem(
                id=str(uuid.uuid4()),
                question=f"ما هو الحد الأدنى لرأس المال لـ {entity}؟",
                question_type="factual",
                language="ar",
                ground_truth_answer=f"{capital_match.group(1)} {capital_match.group(2)} جنيه",
                relevant_docs=[source],
                difficulty="easy",
                topic="رأس_المال",
            )
            items.append(item)
        
        return items


def create_dataset_generator(llm_client=None) -> DatasetGenerator:
    """Factory function."""
    return DatasetGenerator(llm_client=llm_client)
