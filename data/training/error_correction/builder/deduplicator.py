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
from concurrent.futures import ThreadPoolExecutor
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


@dataclass
class ExampleSignature:
    """Cached normalized fields used for semantic dedup checks."""

    question_lc: str
    broken_norm: str
    corrected_norm: str
    tokens: Set[str]


@dataclass
class PreparedExample:
    """Precomputed values used by dedup checks."""

    exact_key: Tuple[Optional[int], str, str]
    broken_hash: str
    corrected_hash: str
    signature: ExampleSignature


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


def semantic_similarity_from_signatures(sig1: ExampleSignature, sig2: ExampleSignature) -> float:
    """Compute semantic similarity from pre-normalized signatures."""
    q_sim = compute_similarity(sig1.question_lc, sig2.question_lc)
    b_sim = compute_similarity(sig1.broken_norm, sig2.broken_norm)
    c_sim = compute_similarity(sig1.corrected_norm, sig2.corrected_norm)
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
        self._clean_signatures: List[ExampleSignature] = []
        self._internal_signatures: List[ExampleSignature] = []
        
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

    def _prepare_example(self, ex: RepairExample) -> PreparedExample:
        """Precompute hashes and signatures once per example."""
        broken_hash = sql_hash(ex.broken_sql)
        return PreparedExample(
            exact_key=(
                ex.metadata.original_question_id,
                ex.metadata.db_family,
                broken_hash,
            ),
            broken_hash=broken_hash,
            corrected_hash=sql_hash(ex.corrected_sql),
            signature=self._build_signature(ex),
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
        ex_sig: ExampleSignature,
        pool_signatures: List[ExampleSignature],
    ) -> bool:
        """Check if example is semantically too similar to existing."""
        # Only check last N examples for efficiency
        check_window = 100
        recent = (
            pool_signatures[-check_window:]
            if len(pool_signatures) > check_window
            else pool_signatures
        )

        ex_len = max(1, len(ex_sig.broken_norm))
        
        for other_sig in recent:
            # Cheap filter 1: skip if broken SQL lengths are too different.
            other_len = max(1, len(other_sig.broken_norm))
            if abs(ex_len - other_len) / max(ex_len, other_len) > 0.65:
                continue

            # Cheap filter 2: require minimum token overlap before expensive match.
            if ex_sig.tokens and other_sig.tokens:
                overlap = len(ex_sig.tokens & other_sig.tokens) / max(
                    1, len(ex_sig.tokens | other_sig.tokens)
                )
                if overlap < 0.10:
                    continue

            sim = semantic_similarity_from_signatures(ex_sig, other_sig)
            if sim >= self.semantic_threshold:
                return True
        
        return False

    def _build_signature(self, ex: RepairExample) -> ExampleSignature:
        """Build cached normalized representation for semantic checks."""
        question_lc = (ex.question or "").lower()
        broken_norm = normalize_sql(ex.broken_sql)
        corrected_norm = normalize_sql(ex.corrected_sql)
        tokens = extract_key_tokens(question_lc)
        tokens.update(extract_key_tokens(broken_norm))
        tokens.update(extract_key_tokens(corrected_norm))
        return ExampleSignature(
            question_lc=question_lc,
            broken_norm=broken_norm,
            corrected_norm=corrected_norm,
            tokens=tokens,
        )
    
    def _add_prepared(self, ex: RepairExample, prepared: PreparedExample) -> bool:
        """Add an example using precomputed dedup fields."""
        self.stats.total_input += 1

        pool = ex.metadata.pool

        # Check exact key
        if prepared.exact_key in self._exact_keys:
            self.stats.exact_key_dupes += 1
            return False

        # Check normalized broken SQL
        if prepared.broken_hash in self._broken_hashes:
            self.stats.normalized_broken_dupes += 1
            return False

        # Check semantic similarity within pool
        if pool == "clean":
            if self._check_semantic_dupe(prepared.signature, self._clean_signatures):
                self.stats.semantic_dupes += 1
                return False
        else:
            if self._check_semantic_dupe(prepared.signature, self._internal_signatures):
                self.stats.semantic_dupes += 1
                return False

        # Not a duplicate - add it
        self._exact_keys.add(prepared.exact_key)
        self._broken_hashes.add(prepared.broken_hash)
        self._corrected_hashes.add(prepared.corrected_hash)

        if pool == "clean":
            self._clean_examples.append(ex)
            self._clean_signatures.append(prepared.signature)
        else:
            self._internal_examples.append(ex)
            self._internal_signatures.append(prepared.signature)

        self.stats.total_output += 1
        return True

    def add_example(self, ex: RepairExample) -> bool:
        """
        Add an example, checking for duplicates.
        
        Returns:
            True if added, False if duplicate
        """
        prepared = self._prepare_example(ex)
        return self._add_prepared(ex, prepared)
    
    def deduplicate_batch(
        self,
        examples: List[RepairExample],
        progress_every: int = 0,
        progress_callback=None,
        prepare_workers: int = 1,
    ) -> Tuple[List[RepairExample], List[RepairExample]]:
        """
        Deduplicate a batch of examples.
        
        Returns:
            (clean_examples, internal_examples)
        """
        if prepare_workers > 1 and len(examples) > 1:
            with ThreadPoolExecutor(max_workers=prepare_workers) as executor:
                prepared_examples = list(executor.map(self._prepare_example, examples))
        else:
            prepared_examples = [self._prepare_example(ex) for ex in examples]

        total = len(examples)
        for idx, (ex, prepared) in enumerate(zip(examples, prepared_examples), start=1):
            self._add_prepared(ex, prepared)
            if progress_callback and progress_every > 0 and idx % progress_every == 0:
                progress_callback(idx, total, self.stats)

        if progress_callback and progress_every > 0 and total % progress_every != 0:
            progress_callback(total, total, self.stats)
        
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
