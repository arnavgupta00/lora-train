#!/usr/bin/env python3
"""
Deduplicator Module

Deduplication logic for error-correction examples:
- Exact key dedup: (question_id, db_id, broken_sql_hash)
- Normalized broken-SQL dedup
- Normalized corrected-SQL dedup
- Near-duplicate semantic checks
"""

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from .metadata import RepairExample
from .verifier import normalize_sql


@dataclass
class DedupStats:
    """Statistics from deduplication."""
    
    total_input: int = 0
    total_output: int = 0
    exact_key_dupes: int = 0
    normalized_broken_dupes: int = 0
    normalized_corrected_dupes: int = 0
    semantic_dupes: int = 0
    cross_pool_similar: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "duplicates_removed": self.total_input - self.total_output,
            "duplicate_rate": round(100 * (self.total_input - self.total_output) / max(1, self.total_input), 2),
            "exact_key_dupes": self.exact_key_dupes,
            "normalized_broken_dupes": self.normalized_broken_dupes,
            "normalized_corrected_dupes": self.normalized_corrected_dupes,
            "semantic_dupes": self.semantic_dupes,
            "cross_pool_similar": self.cross_pool_similar,
        }


def sql_hash(sql: str) -> str:
    """Generate hash of normalized SQL."""
    normalized = normalize_sql(sql)
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def compute_similarity(s1: str, s2: str) -> float:
    """Compute similarity ratio between two strings."""
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def extract_key_tokens(text: str) -> Set[str]:
    """Extract key tokens for semantic comparison."""
    if not text:
        return set()
    
    # Normalize
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Extract words longer than 2 chars
    words = set(w for w in text.split() if len(w) > 2)
    
    return words


def semantic_similarity(
    q1: str, b1: str, c1: str,
    q2: str, b2: str, c2: str,
) -> float:
    """
    Compute semantic similarity between two examples.
    
    Based on (question, broken_sql, corrected_sql) tuples.
    """
    # Question similarity (weighted higher)
    q_sim = compute_similarity(q1, q2)
    
    # SQL similarities
    b_sim = compute_similarity(normalize_sql(b1), normalize_sql(b2))
    c_sim = compute_similarity(normalize_sql(c1), normalize_sql(c2))
    
    # Weighted average
    return 0.4 * q_sim + 0.3 * b_sim + 0.3 * c_sim


class Deduplicator:
    """Deduplicator for error-correction examples."""
    
    def __init__(self, semantic_threshold: float = 0.85):
        """
        Args:
            semantic_threshold: Similarity threshold for semantic dedup (0.85 = 85%)
        """
        self.semantic_threshold = semantic_threshold
        self.stats = DedupStats()
        
        # Tracking sets for each pool
        self._clean_examples: List[RepairExample] = []
        self._internal_examples: List[RepairExample] = []
        
        # Dedup indices
        self._exact_keys: Set[Tuple[Optional[int], str, str]] = set()
        self._broken_hashes: Set[str] = set()
        self._corrected_hashes: Set[str] = set()
    
    def _get_exact_key(self, ex: RepairExample) -> Tuple[Optional[int], str, str]:
        """Get exact dedup key for an example."""
        return (
            ex.metadata.original_question_id,
            ex.metadata.db_family,
            sql_hash(ex.broken_sql),
        )
    
    def _check_exact_dupe(self, ex: RepairExample) -> bool:
        """Check if example is an exact duplicate."""
        key = self._get_exact_key(ex)
        if key in self._exact_keys:
            return True
        return False
    
    def _check_normalized_broken_dupe(self, ex: RepairExample) -> bool:
        """Check if broken SQL is a duplicate (normalized)."""
        h = sql_hash(ex.broken_sql)
        if h in self._broken_hashes:
            return True
        return False
    
    def _check_normalized_corrected_dupe(self, ex: RepairExample) -> bool:
        """Check if corrected SQL is a duplicate (normalized)."""
        h = sql_hash(ex.corrected_sql)
        
        # Allow some corrected SQL overlap (same fix for different broken)
        # But track for reporting
        return False  # Don't filter on corrected SQL alone
    
    def _check_semantic_dupe(
        self,
        ex: RepairExample,
        pool: List[RepairExample],
    ) -> bool:
        """Check if example is semantically too similar to existing."""
        # Only check last N examples for efficiency
        check_window = 100
        recent = pool[-check_window:] if len(pool) > check_window else pool
        
        for other in recent:
            sim = semantic_similarity(
                ex.question, ex.broken_sql, ex.corrected_sql,
                other.question, other.broken_sql, other.corrected_sql,
            )
            if sim >= self.semantic_threshold:
                return True
        
        return False
    
    def add_example(self, ex: RepairExample) -> bool:
        """
        Add an example, checking for duplicates.
        
        Returns:
            True if added, False if duplicate
        """
        self.stats.total_input += 1
        
        pool = ex.metadata.pool
        
        # Check exact key
        if self._check_exact_dupe(ex):
            self.stats.exact_key_dupes += 1
            return False
        
        # Check normalized broken SQL
        if self._check_normalized_broken_dupe(ex):
            self.stats.normalized_broken_dupes += 1
            return False
        
        # Check semantic similarity within pool
        if pool == "clean":
            if self._check_semantic_dupe(ex, self._clean_examples):
                self.stats.semantic_dupes += 1
                return False
        else:
            if self._check_semantic_dupe(ex, self._internal_examples):
                self.stats.semantic_dupes += 1
                return False
        
        # Not a duplicate - add it
        self._exact_keys.add(self._get_exact_key(ex))
        self._broken_hashes.add(sql_hash(ex.broken_sql))
        self._corrected_hashes.add(sql_hash(ex.corrected_sql))
        
        if pool == "clean":
            self._clean_examples.append(ex)
        else:
            self._internal_examples.append(ex)
        
        self.stats.total_output += 1
        return True
    
    def deduplicate_batch(
        self,
        examples: List[RepairExample],
    ) -> Tuple[List[RepairExample], List[RepairExample]]:
        """
        Deduplicate a batch of examples.
        
        Returns:
            (clean_examples, internal_examples)
        """
        for ex in examples:
            self.add_example(ex)
        
        return self._clean_examples.copy(), self._internal_examples.copy()
    
    def check_cross_pool_similarity(self) -> List[Dict[str, Any]]:
        """
        Check for similar examples across clean and internal pools.
        
        Returns report of similar pairs (for tracking, not rejection).
        """
        similar_pairs = []
        
        for clean_ex in self._clean_examples:
            for internal_ex in self._internal_examples:
                sim = semantic_similarity(
                    clean_ex.question, clean_ex.broken_sql, clean_ex.corrected_sql,
                    internal_ex.question, internal_ex.broken_sql, internal_ex.corrected_sql,
                )
                if sim >= 0.8:  # Slightly lower threshold for cross-pool
                    similar_pairs.append({
                        "clean_id": clean_ex.metadata.example_id,
                        "internal_id": internal_ex.metadata.example_id,
                        "similarity": round(sim, 3),
                    })
                    self.stats.cross_pool_similar += 1
        
        return similar_pairs
    
    def get_clean_examples(self) -> List[RepairExample]:
        """Get deduplicated clean examples."""
        return self._clean_examples.copy()
    
    def get_internal_examples(self) -> List[RepairExample]:
        """Get deduplicated internal examples."""
        return self._internal_examples.copy()
    
    def get_stats(self) -> DedupStats:
        """Get deduplication statistics."""
        return self.stats
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate deduplication report."""
        cross_pool = self.check_cross_pool_similarity()
        
        return {
            "statistics": self.stats.to_dict(),
            "clean_count": len(self._clean_examples),
            "internal_count": len(self._internal_examples),
            "cross_pool_similar_pairs": len(cross_pool),
            "cross_pool_sample": cross_pool[:20],  # First 20 pairs
        }
