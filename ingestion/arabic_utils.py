"""
Arabic Text Normalization Utilities.

This module provides strict normalization for Arabic text, specifically
designed for processing Egyptian regulatory and legal documents.

Normalization steps:
1. Unify Alef variants (أ, إ, آ, ٱ → ا)
2. Unify Yeh variants (ى → ي)
3. Remove Tatweel/Kashida (ـ)
4. Remove Tashkeel/Diacritics (حركات)
5. Normalize whitespace

These normalizations are crucial for:
- Consistent text matching in retrieval
- Reducing vocabulary size for embeddings
- Handling OCR errors common in Arabic documents
"""

import re
import unicodedata
from typing import Optional


# ============================================================================
# ARABIC CHARACTER MAPPINGS
# ============================================================================

# Alef variants → Alef (ا)
# أ = Alef with Hamza Above (U+0623)
# إ = Alef with Hamza Below (U+0625)
# آ = Alef with Madda Above (U+0622)
# ٱ = Alef Wasla (U+0671)
ALEF_VARIANTS = {
    '\u0623': '\u0627',  # أ → ا
    '\u0625': '\u0627',  # إ → ا
    '\u0622': '\u0627',  # آ → ا
    '\u0671': '\u0627',  # ٱ → ا
}

# Yeh variants → Yeh (ي)
# ى = Alef Maksura (U+0649) - often confused with final Yeh
YEH_VARIANTS = {
    '\u0649': '\u064A',  # ى → ي
}

# Tatweel/Kashida (ـ) - used for text justification, has no semantic meaning
TATWEEL = '\u0640'

# Arabic diacritical marks (Tashkeel/Harakat)
# These are the short vowel marks and other diacritics
ARABIC_DIACRITICS = [
    '\u064B',  # Fathatan (ً)
    '\u064C',  # Dammatan (ٌ)
    '\u064D',  # Kasratan (ٍ)
    '\u064E',  # Fatha (َ)
    '\u064F',  # Damma (ُ)
    '\u0650',  # Kasra (ِ)
    '\u0651',  # Shadda (ّ)
    '\u0652',  # Sukun (ْ)
    '\u0653',  # Maddah Above (ٓ)
    '\u0654',  # Hamza Above (ٔ)
    '\u0655',  # Hamza Below (ٕ)
    '\u0656',  # Subscript Alef (ٖ)
    '\u0657',  # Inverted Damma (ٗ)
    '\u0658',  # Mark Noon Ghunna (٘)
    '\u0659',  # Zwarakay (ٙ)
    '\u065A',  # Vowel Sign Small V Above (ٚ)
    '\u065B',  # Vowel Sign Inverted Small V Above (ٛ)
    '\u065C',  # Vowel Sign Dot Below (ٜ)
    '\u065D',  # Reversed Damma (ٝ)
    '\u065E',  # Fatha with Two Dots (ٞ)
    '\u065F',  # Wavy Hamza Below (ٟ)
    '\u0670',  # Superscript Alef (ٰ)
]

# Compile regex patterns for efficiency
DIACRITICS_PATTERN = re.compile('[' + ''.join(ARABIC_DIACRITICS) + ']')
TATWEEL_PATTERN = re.compile(TATWEEL)
WHITESPACE_PATTERN = re.compile(r'\s+')


def normalize_text(text: str) -> str:
    """
    Normalize Arabic text with strict rules for legal/regulatory documents.
    
    This function applies the following normalizations in order:
    1. Unicode NFKC normalization (handles compatibility characters)
    2. Unify Alef variants (أ, إ, آ, ٱ → ا)
    3. Unify Yeh variants (ى → ي)
    4. Remove Tatweel/Kashida (ـ)
    5. Remove all Tashkeel/diacritics (حركات)
    6. Normalize whitespace (multiple spaces → single space)
    7. Strip leading/trailing whitespace
    
    Args:
        text: Input Arabic text string
        
    Returns:
        Normalized text string
        
    Example:
        >>> normalize_text("القَرَارُ رَقْم ١٢٣")
        'القرار رقم ١٢٣'
        >>> normalize_text("الإدارة المــالية")
        'الادارة المالية'
    """
    if not text:
        return ""
    
    # Step 1: Unicode NFKC normalization
    # This handles compatibility characters and composed forms
    text = unicodedata.normalize('NFKC', text)
    
    # Step 2: Unify Alef variants
    # Replace أ, إ, آ, ٱ with ا
    for variant, replacement in ALEF_VARIANTS.items():
        text = text.replace(variant, replacement)
    
    # Step 3: Unify Yeh variants
    # Replace ى with ي
    for variant, replacement in YEH_VARIANTS.items():
        text = text.replace(variant, replacement)
    
    # Step 4: Remove Tatweel (Kashida)
    # This character is used for text justification and has no meaning
    text = TATWEEL_PATTERN.sub('', text)
    
    # Step 5: Remove Tashkeel (diacritics/harakat)
    # Removes all short vowel marks
    text = DIACRITICS_PATTERN.sub('', text)
    
    # Step 6: Normalize whitespace
    # Replace multiple spaces/tabs/newlines with single space
    text = WHITESPACE_PATTERN.sub(' ', text)
    
    # Step 7: Strip leading/trailing whitespace
    text = text.strip()
    
    return text


class ArabicTextNormalizer:
    """
    Class-based Arabic text normalizer with configurable options.
    
    Provides more control over which normalizations to apply.
    Useful when you need to preserve certain features (e.g., diacritics).
    """
    
    def __init__(
        self,
        normalize_alef: bool = True,
        normalize_yeh: bool = True,
        remove_tatweel: bool = True,
        remove_diacritics: bool = True,
        normalize_whitespace: bool = True,
    ):
        """
        Initialize the normalizer with configuration options.
        
        Args:
            normalize_alef: Unify Alef variants
            normalize_yeh: Unify Yeh variants  
            remove_tatweel: Remove Tatweel/Kashida
            remove_diacritics: Remove Tashkeel/diacritics
            normalize_whitespace: Normalize whitespace
        """
        self.normalize_alef = normalize_alef
        self.normalize_yeh = normalize_yeh
        self.remove_tatweel = remove_tatweel
        self.remove_diacritics = remove_diacritics
        self.normalize_whitespace = normalize_whitespace
    
    def __call__(self, text: str) -> str:
        """
        Normalize text according to configured options.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        return self.normalize(text)
    
    def normalize(self, text: str) -> str:
        """
        Normalize text according to configured options.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Unicode normalization always applied
        text = unicodedata.normalize('NFKC', text)
        
        if self.normalize_alef:
            for variant, replacement in ALEF_VARIANTS.items():
                text = text.replace(variant, replacement)
        
        if self.normalize_yeh:
            for variant, replacement in YEH_VARIANTS.items():
                text = text.replace(variant, replacement)
        
        if self.remove_tatweel:
            text = TATWEEL_PATTERN.sub('', text)
        
        if self.remove_diacritics:
            text = DIACRITICS_PATTERN.sub('', text)
        
        if self.normalize_whitespace:
            text = WHITESPACE_PATTERN.sub(' ', text)
            text = text.strip()
        
        return text


def remove_diacritics_only(text: str) -> str:
    """
    Remove only diacritics from text, keeping other characters.
    
    Useful when you want to remove harakat but preserve
    other Arabic text features.
    
    Args:
        text: Input text
        
    Returns:
        Text with diacritics removed
    """
    return DIACRITICS_PATTERN.sub('', text)


def is_arabic(text: str) -> bool:
    """
    Check if text contains Arabic characters.
    
    Args:
        text: Input text
        
    Returns:
        True if text contains Arabic characters
    """
    # Arabic Unicode block: U+0600 to U+06FF
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    return bool(arabic_pattern.search(text))


def get_arabic_ratio(text: str) -> float:
    """
    Calculate the ratio of Arabic characters in text.
    
    Useful for detecting if a document is primarily in Arabic.
    
    Args:
        text: Input text
        
    Returns:
        Ratio of Arabic characters (0.0 to 1.0)
    """
    if not text:
        return 0.0
    
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    arabic_chars = len(arabic_pattern.findall(text))
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    
    if total_chars == 0:
        return 0.0
    
    return arabic_chars / total_chars


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Test examples
    test_cases = [
        "القَرَارُ رَقْم ١٢٣",  # With diacritics
        "الإدارة المــالية",  # With tatweel
        "أحكام القانون رقم ٢٠٢٤",  # With Alef variants
        "الهيئة العامة للرقابة المالية",  # Standard text
        "المادة الأولى: ...",  # Legal article format
    ]
    
    print("Arabic Text Normalization Examples:")
    print("=" * 60)
    
    for text in test_cases:
        normalized = normalize_text(text)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print(f"Arabic ratio: {get_arabic_ratio(text):.2%}")
        print("-" * 60)
