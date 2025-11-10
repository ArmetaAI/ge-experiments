"""
LLM utilities for the Госэкспертиза completeness check application.

This module provides helper functions for working with LLMs if needed,
such as text processing, prompt generation, and context management.

Note: The current implementation focuses on fuzzy string matching for
file comparison and does not heavily rely on LLM calls for the core
completeness check logic. However, these utilities can be extended
for future enhancements like intelligent document classification.
"""

from typing import List, Dict, Any, Optional
import re

#TODO: This is an AI generated code. Needs improvements and checking
class ContextManager:
    """
    Manages context for LLM interactions with large documents.
    
    This class helps prevent context overflow by:
    - Chunking large texts
    - Tracking token usage
    - Prioritizing important information
    - Generating summaries
    """
    
    def __init__(self, max_context_length: int = 100000):
        """
        Initialize context manager.
        
        Args:
            max_context_length: Maximum context length in characters
        """
        self.max_context_length = max_context_length
        self.current_context_length = 0
    
    def chunk_text(self, text: str, chunk_size: int = 5000, overlap: int = 500) -> List[str]:
        """
        Split large text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                sentence_end = text.rfind('. ', start, end)
                if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation).
        
        For accurate token counting, use tiktoken library.
        This is a simple approximation: ~4 chars per token.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def truncate_to_token_limit(self, text: str, max_tokens: int = 25000) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum token count
            
        Returns:
            Truncated text
        """
        max_chars = max_tokens * 4  # Rough approximation
        
        if len(text) <= max_chars:
            return text
        
        # Truncate and add indicator
        return text[:max_chars] + "\n\n[... truncated for context limit ...]"
    
    def extract_key_sections(self, text: str, section_markers: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Extract key sections from structured documents.
        
        Args:
            text: Document text
            section_markers: List of section headers to look for
            
        Returns:
            Dictionary mapping section names to content
        """
        if section_markers is None:
            # Default Kazakhstan construction document sections
            section_markers = [
                "1. Общие положения",
                "2. Исходные данные",
                "3. Проектные решения",
                "4. Технические характеристики",
                "Состав проекта",
                "Комплектность",
                "Прилагаемые документы"
            ]
        
        sections = {}
        
        for marker in section_markers:
            # Case-insensitive search
            pattern = re.escape(marker)
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                start = match.start()
                # Find next section or end of text
                end = len(text)
                for other_marker in section_markers:
                    if other_marker == marker:
                        continue
                    other_match = re.search(re.escape(other_marker), text[start+len(marker):], re.IGNORECASE)
                    if other_match:
                        potential_end = start + len(marker) + other_match.start()
                        if potential_end < end:
                            end = potential_end
                
                sections[marker] = text[start:end].strip()
        
        return sections
    
    def summarize_for_context(self, text: str, max_length: int = 1000) -> str:
        """
        Create a brief summary for context preservation.
        
        This is a simple extractive summarization. For better results,
        use an LLM to generate summaries.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        if len(text) <= max_length:
            return text
        
        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Take first few sentences and last few sentences
        num_sentences = max(3, max_length // 100)
        
        if len(sentences) <= num_sentences * 2:
            summary = ' '.join(sentences)
        else:
            first_part = ' '.join(sentences[:num_sentences])
            last_part = ' '.join(sentences[-num_sentences:])
            summary = f"{first_part}\n\n[...]\n\n{last_part}"
        
        # Truncate if still too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
    
    def build_context_window(self, 
                            system_prompt: str,
                            document_chunks: List[str],
                            max_tokens: int = 25000) -> List[str]:
        """
        Build context windows that fit within token limits.
        
        Args:
            system_prompt: System prompt to include in each window
            document_chunks: List of document chunks
            max_tokens: Maximum tokens per window
            
        Returns:
            List of complete context windows
        """
        windows = []
        system_tokens = self.estimate_tokens(system_prompt)
        available_tokens = max_tokens - system_tokens - 500  # Buffer
        
        current_window = []
        current_tokens = 0
        
        for chunk in document_chunks:
            chunk_tokens = self.estimate_tokens(chunk)
            
            if current_tokens + chunk_tokens > available_tokens:
                # Start new window
                if current_window:
                    windows.append(system_prompt + "\n\n" + "\n\n".join(current_window))
                current_window = [chunk]
                current_tokens = chunk_tokens
            else:
                current_window.append(chunk)
                current_tokens += chunk_tokens
        
        # Add final window
        if current_window:
            windows.append(system_prompt + "\n\n" + "\n\n".join(current_window))
        
        return windows


class DocumentMemory:
    """
    Memory system for preserving important information across agent calls.
    
    This is useful when processing multiple documents sequentially and
    needing to maintain context about previous documents.
    """
    
    def __init__(self, max_entries: int = 100):
        """
        Initialize document memory.
        
        Args:
            max_entries: Maximum number of document summaries to keep
        """
        self.max_entries = max_entries
        self.memory: List[Dict[str, Any]] = []
    
    def add_document(self, filename: str, summary: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add document summary to memory.
        
        Args:
            filename: Document filename
            summary: Document summary
            metadata: Optional metadata
        """
        entry = {
            "filename": filename,
            "summary": summary,
            "metadata": metadata or {}
        }
        
        self.memory.append(entry)
        
        # Keep only recent entries
        if len(self.memory) > self.max_entries:
            self.memory = self.memory[-self.max_entries:]
    
    def get_context_summary(self) -> str:
        """
        Get a summary of all documents in memory.
        
        Returns:
            Formatted summary string
        """
        if not self.memory:
            return "No documents in memory."
        
        summary_parts = [f"Documents in memory: {len(self.memory)}"]
        
        for entry in self.memory[-10:]:  # Last 10
            summary_parts.append(
                f"- {entry['filename']}: {entry['summary'][:100]}..."
            )
        
        return "\n".join(summary_parts)
    
    def search_memory(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Search memory for documents matching keywords.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of matching document entries
        """
        matches = []
        
        for entry in self.memory:
            for keyword in keywords:
                if keyword.lower() in entry["summary"].lower() or \
                   keyword.lower() in entry["filename"].lower():
                    matches.append(entry)
                    break
        
        return matches


# Utility functions for prompt engineering

def create_structure_extraction_prompt(text_preview: str) -> str:
    """
    Create prompt for extracting document structure from ОПЗ.
    
    Args:
        text_preview: Preview text from ОПЗ document
        
    Returns:
        Formatted prompt
    """
    return f"""
Ты эксперт по анализу проектной документации Республики Казахстан.

Проанализируй следующий фрагмент Общей Пояснительной Записки (ОПЗ) и извлеки:
1. Список обязательных разделов проекта (ИРД, ПСД, и т.д.)
2. Список обязательных документов в каждом разделе
3. Требования к комплектности документации

Текст документа:
{text_preview}

Верни структурированный ответ в формате JSON с полями:
- required_sections: список обязательных разделов
- required_documents: список обязательных документов
- completeness_requirements: текстовое описание требований
"""


def create_classification_prompt(filename: str, text_preview: str) -> str:
    """
    Create prompt for document classification.
    
    Args:
        filename: Document filename
        text_preview: Preview text from document
        
    Returns:
        Formatted prompt
    """
    return f"""
Ты эксперт по классификации строительной документации Республики Казахстан.

Классифицируй следующий документ по одной из категорий:
- ПСД (Проектно-сметная документация)
- ИРД (Исходно-разрешительная документация)
- Архитектурные решения
- Конструктивные решения
- Электротехнические решения
- Водоснабжение и водоотведение
- Теплоснабжение
- Другое

Имя файла: {filename}

Содержимое (фрагмент):
{text_preview}

Верни:
1. Категорию документа
2. Уровень уверенности (0-1)
3. Краткое обоснование
"""
