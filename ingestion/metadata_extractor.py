"""
Semantic Metadata Extractor for FRA RAG System.

Extracts structured metadata from regulatory documents using LLM or rule-based methods.
Enables filtering by entity type, document type, effective date, and topics.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger


@dataclass
class DocumentMetadata:
    """Structured metadata for a regulatory document."""
    # Document identification
    document_type: str = "unknown"  # regulation, decision, circular, form, guide
    law_status: str = "active"  # active, repealed, amended
    
    # Dates
    effective_date: Optional[str] = None  # YYYY-MM-DD
    amendment_date: Optional[str] = None
    
    # Entity information
    entity_types: List[str] = field(default_factory=list)  # Bank, Microfinance, Insurance, Brokerage, etc.
    issuing_authority: str = "FRA"
    
    # Content classification
    topics: List[str] = field(default_factory=list)  # licensing, capital, branches, penalties, etc.
    
    # Flags for common regulatory concepts
    has_penalties: bool = False
    has_capital_requirements: bool = False
    has_licensing_requirements: bool = False
    has_branch_requirements: bool = False
    
    # Original source
    source_file: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "document_type": self.document_type,
            "law_status": self.law_status,
            "effective_date": self.effective_date,
            "amendment_date": self.amendment_date,
            "entity_types": self.entity_types,
            "issuing_authority": self.issuing_authority,
            "topics": self.topics,
            "has_penalties": self.has_penalties,
            "has_capital_requirements": self.has_capital_requirements,
            "has_licensing_requirements": self.has_licensing_requirements,
            "has_branch_requirements": self.has_branch_requirements,
            "source_file": self.source_file,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Create from dictionary."""
        return cls(
            document_type=data.get("document_type", "unknown"),
            law_status=data.get("law_status", "active"),
            effective_date=data.get("effective_date"),
            amendment_date=data.get("amendment_date"),
            entity_types=data.get("entity_types", []),
            issuing_authority=data.get("issuing_authority", "FRA"),
            topics=data.get("topics", []),
            has_penalties=data.get("has_penalties", False),
            has_capital_requirements=data.get("has_capital_requirements", False),
            has_licensing_requirements=data.get("has_licensing_requirements", False),
            has_branch_requirements=data.get("has_branch_requirements", False),
            source_file=data.get("source_file", ""),
        )


class MetadataExtractor:
    """
    Extract semantic metadata from regulatory documents.
    
    Uses rule-based extraction with Arabic/English pattern matching.
    Can optionally use LLM for enhanced extraction.
    """
    
    # Entity type patterns (Arabic)
    ENTITY_PATTERNS = {
        "تمويل_استهلاكي": [
            r"تمويل\s*استهلاكي",
            r"التمويل\s*الاستهلاكي",
            r"شركة?\s*تمويل\s*استهلاكي",
            r"consumer\s*finance",
        ],
        "تمويل_متناهي_الصغر": [
            r"متناهي?\s*الصغر",
            r"التمويل\s*متناهي\s*الصغر",
            r"microfinance",
        ],
        "تأجير_تمويلي": [
            r"تأجير\s*تمويلي",
            r"التأجير\s*التمويلي",
            r"leasing",
        ],
        "سمسرة": [
            r"سمسرة",
            r"شركة?\s*سمسرة",
            r"brokerage",
        ],
        "تأمين": [
            r"تأمين",
            r"شركة?\s*تأمين",
            r"insurance",
        ],
        "توريق": [
            r"توريق",
            r"سندات\s*توريق",
            r"securitization",
        ],
        "بنك": [
            r"بنك",
            r"مصرف",
            r"bank",
        ],
    }
    
    # Document type patterns
    DOCUMENT_TYPE_PATTERNS = {
        "نموذج": [r"نموذج", r"استمارة", r"form", r"template"],
        "لائحة": [r"لائحة", r"regulation", r"قانون"],
        "قرار": [r"قرار", r"decision", r"decree"],
        "تعميم": [r"تعميم", r"circular", r"إعلان"],
        "دليل": [r"دليل", r"guide", r"إرشادات"],
    }
    
    # Topic patterns
    TOPIC_PATTERNS = {
        "ترخيص": [r"ترخيص", r"تأسيس", r"قيد", r"license", r"registration"],
        "رأس_المال": [r"رأس\s*المال", r"capital", r"مال\s*مدفوع"],
        "فروع": [r"فرع", r"فروع", r"branch"],
        "غلق": [r"غلق", r"إغلاق", r"close", r"closure"],
        "نقل": [r"نقل", r"transfer", r"تحويل"],
        "مدير": [r"مدير", r"إدارة", r"manager", r"director"],
        "مجلس_الإدارة": [r"مجلس\s*الإدارة", r"board", r"أعضاء"],
        "سندات": [r"سندات", r"صكوك", r"bonds", r"securities"],
        "أمين_حفظ": [r"أمين\s*حفظ", r"custodian", r"حفظ"],
        "تسجيل": [r"تسجيل", r"register", r"قيد"],
        "استعلام_أمني": [r"استعلام\s*أمني", r"أمنية", r"security\s*check"],
    }
    
    # Penalty indicators
    PENALTY_PATTERNS = [
        r"غرامة",
        r"عقوبة",
        r"جزاء",
        r"إلغاء\s*الترخيص",
        r"penalty",
        r"fine",
        r"sanction",
    ]
    
    # Capital requirement indicators
    CAPITAL_PATTERNS = [
        r"الحد\s*الأدنى.*رأس\s*المال",
        r"رأس\s*مال.*مدفوع",
        r"minimum\s*capital",
        r"جنيه\s*مصري",
        r"\d+\s*مليون",
    ]
    
    # Date patterns
    DATE_PATTERNS = [
        r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})",  # YYYY-MM-DD
        r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})",  # DD-MM-YYYY
        r"بتاريخ\s*(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})",
    ]
    
    def __init__(self, use_llm: bool = False, llm_client=None):
        """
        Initialize metadata extractor.
        
        Args:
            use_llm: Whether to use LLM for enhanced extraction
            llm_client: LLM client instance (optional)
        """
        self.use_llm = use_llm
        self.llm_client = llm_client
        logger.info(f"MetadataExtractor initialized (use_llm={use_llm})")
    
    def extract(self, text: str, source_file: str = "") -> DocumentMetadata:
        """
        Extract metadata from document text.
        
        Args:
            text: Document text content
            source_file: Source file name for context
            
        Returns:
            DocumentMetadata object with extracted information
        """
        metadata = DocumentMetadata(source_file=source_file)
        
        # Use filename for initial hints
        if source_file:
            metadata = self._extract_from_filename(source_file, metadata)
        
        # Extract from text content
        metadata = self._extract_entity_types(text, metadata)
        metadata = self._extract_document_type(text, source_file, metadata)
        metadata = self._extract_topics(text, metadata)
        metadata = self._extract_flags(text, metadata)
        metadata = self._extract_dates(text, metadata)
        
        # LLM enhancement if enabled
        if self.use_llm and self.llm_client:
            metadata = self._llm_enhance(text, metadata)
        
        logger.debug(f"Extracted metadata for {source_file}: {metadata.to_dict()}")
        return metadata
    
    def _extract_from_filename(self, filename: str, metadata: DocumentMetadata) -> DocumentMetadata:
        """Extract hints from filename."""
        filename_lower = filename.lower()
        
        # Document type from filename
        if "نموذج" in filename or "form" in filename_lower:
            metadata.document_type = "نموذج"
        elif "قرار" in filename:
            metadata.document_type = "قرار"
        elif "لائحة" in filename:
            metadata.document_type = "لائحة"
        elif "مستندات" in filename:
            metadata.document_type = "قائمة_مستندات"
        elif "استعلام" in filename or "استطلاع" in filename:
            metadata.document_type = "استعلام"
        elif "اقرار" in filename or "إقرار" in filename:
            metadata.document_type = "إقرار"
        
        return metadata
    
    def _extract_entity_types(self, text: str, metadata: DocumentMetadata) -> DocumentMetadata:
        """Extract entity types mentioned in document."""
        text_lower = text.lower()
        entities = []
        
        for entity, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    if entity not in entities:
                        entities.append(entity)
                    break
        
        metadata.entity_types = entities if entities else ["عام"]
        return metadata
    
    def _extract_document_type(self, text: str, filename: str, metadata: DocumentMetadata) -> DocumentMetadata:
        """Extract document type."""
        if metadata.document_type != "unknown":
            return metadata
        
        combined = f"{filename} {text[:500]}".lower()
        
        for doc_type, patterns in self.DOCUMENT_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    metadata.document_type = doc_type
                    return metadata
        
        return metadata
    
    def _extract_topics(self, text: str, metadata: DocumentMetadata) -> DocumentMetadata:
        """Extract topics discussed in document."""
        topics = []
        
        for topic, patterns in self.TOPIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if topic not in topics:
                        topics.append(topic)
                    break
        
        metadata.topics = topics
        return metadata
    
    def _extract_flags(self, text: str, metadata: DocumentMetadata) -> DocumentMetadata:
        """Extract boolean flags for common regulatory concepts."""
        # Check for penalties
        for pattern in self.PENALTY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                metadata.has_penalties = True
                break
        
        # Check for capital requirements
        for pattern in self.CAPITAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                metadata.has_capital_requirements = True
                break
        
        # Licensing requirements
        if re.search(r"ترخيص|تأسيس|قيد|license|registration", text, re.IGNORECASE):
            metadata.has_licensing_requirements = True
        
        # Branch requirements
        if re.search(r"فرع|فروع|branch", text, re.IGNORECASE):
            metadata.has_branch_requirements = True
        
        return metadata
    
    def _extract_dates(self, text: str, metadata: DocumentMetadata) -> DocumentMetadata:
        """Extract dates from document."""
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                try:
                    # Try to construct a date
                    if len(groups[0]) == 4:  # YYYY first
                        year, month, day = groups[0], groups[1], groups[2]
                    else:  # DD first
                        day, month, year = groups[0], groups[1], groups[2]
                    
                    metadata.effective_date = f"{year}-{int(month):02d}-{int(day):02d}"
                    break
                except:
                    pass
        
        return metadata
    
    def _llm_enhance(self, text: str, metadata: DocumentMetadata) -> DocumentMetadata:
        """Use LLM to enhance metadata extraction."""
        if not self.llm_client:
            return metadata
        
        prompt = f"""
حلل النص التنظيمي التالي واستخرج البيانات الوصفية:

النص (أول 2000 حرف):
{text[:2000]}

البيانات المستخرجة حالياً:
- نوع المستند: {metadata.document_type}
- أنواع الجهات: {', '.join(metadata.entity_types)}
- المواضيع: {', '.join(metadata.topics)}

هل هناك تصحيحات أو إضافات؟ أجب بصيغة JSON فقط:
{{"entity_types": [], "topics": [], "document_type": ""}}
"""
        
        try:
            # This would call the LLM for enhanced extraction
            # For now, return as-is
            pass
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
        
        return metadata


class MetadataFilter:
    """
    Filter documents based on metadata criteria.
    
    Used during retrieval to filter by entity type, document type, date, etc.
    """
    
    def __init__(self):
        """Initialize metadata filter."""
        pass
    
    def build_qdrant_filter(
        self,
        entity_type: Optional[str] = None,
        document_type: Optional[str] = None,
        topic: Optional[str] = None,
        year: Optional[int] = None,
        has_penalties: Optional[bool] = None,
        has_capital_requirements: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build Qdrant filter from metadata criteria.
        
        Args:
            entity_type: Filter by entity type (e.g., "تمويل_استهلاكي")
            document_type: Filter by document type (e.g., "نموذج")
            topic: Filter by topic (e.g., "ترخيص")
            year: Filter by effective year
            has_penalties: Filter for documents with penalties
            has_capital_requirements: Filter for documents with capital requirements
            
        Returns:
            Qdrant filter dictionary or None
        """
        conditions = []
        
        if entity_type and entity_type != "الكل":
            conditions.append({
                "key": "metadata.entity_types",
                "match": {"any": [entity_type]}
            })
        
        if document_type and document_type != "الكل":
            conditions.append({
                "key": "metadata.document_type",
                "match": {"value": document_type}
            })
        
        if topic and topic != "الكل":
            conditions.append({
                "key": "metadata.topics",
                "match": {"any": [topic]}
            })
        
        if year:
            conditions.append({
                "key": "metadata.effective_date",
                "match": {"text": str(year)}
            })
        
        if has_penalties is not None:
            conditions.append({
                "key": "metadata.has_penalties",
                "match": {"value": has_penalties}
            })
        
        if has_capital_requirements is not None:
            conditions.append({
                "key": "metadata.has_capital_requirements",
                "match": {"value": has_capital_requirements}
            })
        
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return {"must": conditions}
        
        return {"must": conditions}


def create_metadata_extractor(use_llm: bool = False, llm_client=None) -> MetadataExtractor:
    """Factory function to create metadata extractor."""
    return MetadataExtractor(use_llm=use_llm, llm_client=llm_client)
