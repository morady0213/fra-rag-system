"""
Hierarchical Document Chunker for FRA RAG System.

Implements Parent-Document Retrieval (Small-to-Big) strategy:
- Index small chunks (clauses/sentences) for precise vector search
- Retrieve parent context (full articles/sections) for LLM generation
- Preserves legal document hierarchy: Document -> Chapter -> Article -> Clause
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from loguru import logger


@dataclass
class DocumentNode:
    """Hierarchical document node."""
    id: str
    content: str
    level: str  # "document", "chapter", "article", "clause", "item"
    title: str = ""
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "level": self.level,
            "title": self.title,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class HierarchicalDocument:
    """A document with hierarchical structure."""
    source: str
    root_id: str
    nodes: Dict[str, DocumentNode] = field(default_factory=dict)
    
    def get_node(self, node_id: str) -> Optional[DocumentNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_parent(self, node_id: str) -> Optional[DocumentNode]:
        """Get parent node."""
        node = self.nodes.get(node_id)
        if node and node.parent_id:
            return self.nodes.get(node.parent_id)
        return None
    
    def get_children(self, node_id: str) -> List[DocumentNode]:
        """Get child nodes."""
        node = self.nodes.get(node_id)
        if node:
            return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]
        return []
    
    def get_ancestors(self, node_id: str, max_levels: int = 3) -> List[DocumentNode]:
        """Get ancestor nodes up to max_levels."""
        ancestors = []
        current_id = node_id
        
        for _ in range(max_levels):
            node = self.nodes.get(current_id)
            if not node or not node.parent_id:
                break
            parent = self.nodes.get(node.parent_id)
            if parent:
                ancestors.append(parent)
                current_id = parent.id
            else:
                break
        
        return ancestors
    
    def get_leaf_nodes(self) -> List[DocumentNode]:
        """Get all leaf nodes (smallest chunks for indexing)."""
        return [n for n in self.nodes.values() if not n.children_ids]
    
    def get_nodes_by_level(self, level: str) -> List[DocumentNode]:
        """Get all nodes at a specific level."""
        return [n for n in self.nodes.values() if n.level == level]


class HierarchicalChunker:
    """
    Parse legal documents into hierarchical structure.
    
    Hierarchy levels:
    - document: Full document
    - chapter: الباب / الفصل (Chapter)
    - article: المادة (Article)
    - clause: البند / الفقرة (Clause)
    - item: أولاً، ثانياً (Numbered items)
    
    Strategy: Small-to-Big
    - Index small chunks (clauses/items) for precise search
    - Store parent references for context expansion
    """
    
    # Hierarchy detection patterns
    CHAPTER_PATTERNS = [
        r"^(الباب\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|\d+))\s*[:\-]?\s*(.*?)$",
        r"^(الفصل\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|\d+))\s*[:\-]?\s*(.*?)$",
        r"^(Chapter\s+\d+)\s*[:\-]?\s*(.*?)$",
    ]
    
    ARTICLE_PATTERNS = [
        r"^(مادة\s*\(?\s*(\d+)\s*\)?)\s*[:\-]?\s*(.*?)$",
        r"^(المادة\s+(\d+))\s*[:\-]?\s*(.*?)$",
        r"^(Article\s+(\d+))\s*[:\-]?\s*(.*?)$",
    ]
    
    CLAUSE_PATTERNS = [
        r"^(البند\s+(\d+))\s*[:\-]?\s*(.*?)$",
        r"^(الفقرة\s+(\w+))\s*[:\-]?\s*(.*?)$",
        r"^(\d+\s*[-\)\.]\s*)(.*?)$",
    ]
    
    ITEM_PATTERNS = [
        r"^(أولاً|ثانياً|ثالثاً|رابعاً|خامساً|سادساً|سابعاً|ثامناً|تاسعاً|عاشراً)\s*[:\-]?\s*(.*?)$",
        r"^([أ-ي]\s*[-\)\.]\s*)(.*?)$",
        r"^([a-z]\s*[-\)\.]\s*)(.*?)$",
    ]
    
    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 500,
        overlap: int = 50,
        index_level: str = "clause",  # Level to index for search
    ):
        """
        Initialize hierarchical chunker.
        
        Args:
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size for leaf nodes
            overlap: Character overlap between chunks
            index_level: Which level to index for vector search
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.index_level = index_level
        
        logger.info(f"HierarchicalChunker initialized (index_level={index_level})")
    
    def parse(self, text: str, source: str, metadata: Dict[str, Any] = None) -> HierarchicalDocument:
        """
        Parse document text into hierarchical structure.
        
        Args:
            text: Document text content
            source: Source file name
            metadata: Additional metadata
            
        Returns:
            HierarchicalDocument with parsed structure
        """
        metadata = metadata or {}
        
        # Create root document node
        root_id = str(uuid.uuid4())
        root = DocumentNode(
            id=root_id,
            content=text,
            level="document",
            title=source,
            metadata={"source": source, **metadata},
            start_char=0,
            end_char=len(text),
        )
        
        doc = HierarchicalDocument(source=source, root_id=root_id)
        doc.nodes[root_id] = root
        
        # Parse structure
        self._parse_structure(text, root_id, doc, metadata)
        
        # If no structure found, create chunks from text
        if len(doc.nodes) == 1:
            self._create_flat_chunks(text, root_id, doc, metadata)
        
        logger.info(f"Parsed {source}: {len(doc.nodes)} nodes, {len(doc.get_leaf_nodes())} leaf nodes")
        return doc
    
    def _parse_structure(
        self,
        text: str,
        parent_id: str,
        doc: HierarchicalDocument,
        metadata: Dict[str, Any],
    ):
        """Parse hierarchical structure from text."""
        lines = text.split('\n')
        current_chapter_id = parent_id
        current_article_id = None
        current_clause_id = None
        
        current_content = []
        current_start = 0
        char_pos = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for chapter
            chapter_match = self._match_pattern(line_stripped, self.CHAPTER_PATTERNS)
            if chapter_match:
                # Save previous content
                if current_content and current_article_id:
                    self._finalize_node(current_article_id, current_content, doc)
                
                # Create chapter node
                chapter_id = str(uuid.uuid4())
                chapter = DocumentNode(
                    id=chapter_id,
                    content=line_stripped,
                    level="chapter",
                    title=chapter_match.group(1) if chapter_match else line_stripped[:50],
                    parent_id=parent_id,
                    metadata={"source": doc.source, **metadata},
                    start_char=char_pos,
                )
                doc.nodes[chapter_id] = chapter
                doc.nodes[parent_id].children_ids.append(chapter_id)
                
                current_chapter_id = chapter_id
                current_article_id = None
                current_content = []
                char_pos += len(line) + 1
                continue
            
            # Check for article
            article_match = self._match_pattern(line_stripped, self.ARTICLE_PATTERNS)
            if article_match:
                # Save previous article content
                if current_content and current_article_id:
                    self._finalize_node(current_article_id, current_content, doc)
                
                # Create article node
                article_id = str(uuid.uuid4())
                article_num = article_match.group(2) if article_match.lastindex >= 2 else ""
                article = DocumentNode(
                    id=article_id,
                    content=line_stripped,
                    level="article",
                    title=f"المادة {article_num}" if article_num else line_stripped[:50],
                    parent_id=current_chapter_id,
                    metadata={"source": doc.source, "article_number": article_num, **metadata},
                    start_char=char_pos,
                )
                doc.nodes[article_id] = article
                doc.nodes[current_chapter_id].children_ids.append(article_id)
                
                current_article_id = article_id
                current_clause_id = None
                current_content = [line_stripped]
                char_pos += len(line) + 1
                continue
            
            # Check for clause/item within article
            clause_match = self._match_pattern(line_stripped, self.CLAUSE_PATTERNS + self.ITEM_PATTERNS)
            if clause_match and current_article_id:
                # Create clause node
                clause_id = str(uuid.uuid4())
                clause = DocumentNode(
                    id=clause_id,
                    content=line_stripped,
                    level="clause",
                    title=clause_match.group(1) if clause_match else line_stripped[:30],
                    parent_id=current_article_id,
                    metadata={"source": doc.source, **metadata},
                    start_char=char_pos,
                )
                doc.nodes[clause_id] = clause
                doc.nodes[current_article_id].children_ids.append(clause_id)
                current_clause_id = clause_id
            
            # Accumulate content
            if line_stripped:
                current_content.append(line_stripped)
            
            char_pos += len(line) + 1
        
        # Finalize last node
        if current_content and current_article_id:
            self._finalize_node(current_article_id, current_content, doc)
    
    def _match_pattern(self, text: str, patterns: List[str]) -> Optional[re.Match]:
        """Try to match text against patterns."""
        for pattern in patterns:
            match = re.match(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                return match
        return None
    
    def _finalize_node(self, node_id: str, content_lines: List[str], doc: HierarchicalDocument):
        """Finalize node with accumulated content."""
        if node_id in doc.nodes:
            full_content = '\n'.join(content_lines)
            doc.nodes[node_id].content = full_content
            doc.nodes[node_id].end_char = doc.nodes[node_id].start_char + len(full_content)
    
    def _create_flat_chunks(
        self,
        text: str,
        parent_id: str,
        doc: HierarchicalDocument,
        metadata: Dict[str, Any],
    ):
        """Create flat chunks when no structure is detected."""
        # Split by paragraphs or sentences
        paragraphs = re.split(r'\n\s*\n', text)
        
        char_pos = 0
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < self.min_chunk_size:
                char_pos += len(para) + 2
                continue
            
            # Create chunk node
            chunk_id = str(uuid.uuid4())
            chunk = DocumentNode(
                id=chunk_id,
                content=para,
                level="clause",
                title=f"فقرة {i + 1}",
                parent_id=parent_id,
                metadata={"source": doc.source, "chunk_index": i, **metadata},
                start_char=char_pos,
                end_char=char_pos + len(para),
            )
            doc.nodes[chunk_id] = chunk
            doc.nodes[parent_id].children_ids.append(chunk_id)
            
            char_pos += len(para) + 2
    
    def get_indexable_chunks(
        self,
        doc: HierarchicalDocument,
        include_parent_context: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get chunks suitable for vector indexing.
        
        Args:
            doc: Hierarchical document
            include_parent_context: Whether to include parent reference
            
        Returns:
            List of chunk dictionaries for indexing
        """
        chunks = []
        leaf_nodes = doc.get_leaf_nodes()
        
        for node in leaf_nodes:
            chunk = {
                "id": node.id,
                "text": node.content,
                "level": node.level,
                "title": node.title,
                "parent_id": node.parent_id,
                "source": doc.source,
                "metadata": {
                    **node.metadata,
                    "node_id": node.id,
                    "parent_id": node.parent_id,
                    "level": node.level,
                },
            }
            
            # Add parent context reference
            if include_parent_context and node.parent_id:
                parent = doc.get_node(node.parent_id)
                if parent:
                    chunk["parent_content"] = parent.content
                    chunk["parent_title"] = parent.title
                    chunk["metadata"]["parent_level"] = parent.level
            
            chunks.append(chunk)
        
        return chunks
    
    def expand_to_parent(
        self,
        doc: HierarchicalDocument,
        node_id: str,
        target_level: str = "article",
    ) -> Optional[str]:
        """
        Expand a leaf node to its parent at target level.
        
        Args:
            doc: Hierarchical document
            node_id: Node ID to expand
            target_level: Target level to expand to
            
        Returns:
            Parent content at target level
        """
        node = doc.get_node(node_id)
        if not node:
            return None
        
        # Walk up the tree to find target level
        current = node
        while current:
            if current.level == target_level:
                return current.content
            if current.parent_id:
                current = doc.get_node(current.parent_id)
            else:
                break
        
        # Return highest parent if target not found
        ancestors = doc.get_ancestors(node_id)
        if ancestors:
            return ancestors[-1].content
        
        return node.content


class ParentDocumentRetriever:
    """
    Retriever that fetches parent context for matched chunks.
    
    Strategy:
    1. Search small chunks (clauses) for precise matching
    2. Expand to parent (article) for full context
    3. Return combined context for LLM
    """
    
    def __init__(
        self,
        vector_store,
        hierarchical_docs: Dict[str, HierarchicalDocument] = None,
    ):
        """
        Initialize parent document retriever.
        
        Args:
            vector_store: Vector store instance
            hierarchical_docs: Pre-parsed hierarchical documents
        """
        self.vector_store = vector_store
        self.hierarchical_docs = hierarchical_docs or {}
        logger.info("ParentDocumentRetriever initialized")
    
    def add_document(self, doc: HierarchicalDocument):
        """Add a hierarchical document."""
        self.hierarchical_docs[doc.source] = doc
    
    def retrieve_with_parent(
        self,
        query: str,
        k: int = 5,
        expand_level: str = "article",
    ) -> Dict[str, Any]:
        """
        Retrieve chunks and expand to parent context.
        
        Args:
            query: Search query
            k: Number of results
            expand_level: Level to expand to
            
        Returns:
            Retrieved results with expanded context
        """
        # Search small chunks
        results = self.vector_store.search(query, k=k * 2)  # Get more for deduplication
        
        expanded_results = []
        seen_parents = set()
        
        for result in results:
            source = result.get("source", "")
            node_id = result.get("metadata", {}).get("node_id")
            parent_id = result.get("metadata", {}).get("parent_id")
            
            # Skip if we've already included this parent
            if parent_id and parent_id in seen_parents:
                continue
            
            # Try to expand to parent
            expanded_content = result.get("content", result.get("text", ""))
            parent_title = ""
            
            if source in self.hierarchical_docs and node_id:
                doc = self.hierarchical_docs[source]
                parent_content = doc.expand_to_parent(node_id, expand_level)
                if parent_content:
                    expanded_content = parent_content
                    parent = doc.get_parent(node_id)
                    if parent:
                        parent_title = parent.title
                        seen_parents.add(parent.id)
            
            expanded_results.append({
                "source": source,
                "content": expanded_content,
                "matched_chunk": result.get("content", result.get("text", "")),
                "score": result.get("score", 0),
                "parent_title": parent_title,
                "metadata": result.get("metadata", {}),
            })
            
            if len(expanded_results) >= k:
                break
        
        # Build context
        context_parts = []
        for r in expanded_results:
            if r["parent_title"]:
                context_parts.append(f"[{r['source']} - {r['parent_title']}]\n{r['content']}")
            else:
                context_parts.append(f"[{r['source']}]\n{r['content']}")
        
        return {
            "results": expanded_results,
            "context": "\n\n---\n\n".join(context_parts),
            "sources": [
                {
                    "source": r["source"],
                    "content": r["content"],
                    "matched_chunk": r["matched_chunk"],
                    "score": r["score"],
                }
                for r in expanded_results
            ],
        }


def create_hierarchical_chunker(
    min_chunk_size: int = 100,
    max_chunk_size: int = 500,
    index_level: str = "clause",
) -> HierarchicalChunker:
    """Factory function to create hierarchical chunker."""
    return HierarchicalChunker(
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        index_level=index_level,
    )
