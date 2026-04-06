#!/usr/bin/env python3
"""
Splitter Module

Train/dev splitting with stratification and quota enforcement.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import DatabaseFamilyQuotas, FailureFamilyQuotas
from .metadata import RepairExample


@dataclass
class SplitStats:
    """Statistics from splitting."""
    
    total_input: int = 0
    train_count: int = 0
    dev_count: int = 0
    
    train_by_db: Dict[str, int] = field(default_factory=dict)
    dev_by_db: Dict[str, int] = field(default_factory=dict)
    train_by_failure_type: Dict[str, int] = field(default_factory=dict)
    dev_by_failure_type: Dict[str, int] = field(default_factory=dict)
    
    quota_violations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_input": self.total_input,
            "train_count": self.train_count,
            "dev_count": self.dev_count,
            "train_ratio": round(self.train_count / max(1, self.total_input), 3),
            "dev_ratio": round(self.dev_count / max(1, self.total_input), 3),
            "train_by_db": self.train_by_db,
            "dev_by_db": self.dev_by_db,
            "train_by_failure_type": self.train_by_failure_type,
            "dev_by_failure_type": self.dev_by_failure_type,
            "quota_violations": self.quota_violations,
        }


class Splitter:
    """
    Train/dev splitter with stratification.
    
    Features:
    - Stratified by db_family and failure_type
    - Respects minimum quotas
    - Separate splitting for clean and internal pools
    """
    
    def __init__(
        self,
        dev_ratio: float = 0.1,
        db_quotas: Optional[DatabaseFamilyQuotas] = None,
        failure_quotas: Optional[FailureFamilyQuotas] = None,
        seed: int = 42,
    ):
        """
        Args:
            dev_ratio: Fraction for dev set (default 0.1 = 10%)
            db_quotas: Minimum database family quotas
            failure_quotas: Minimum failure type quotas
            seed: Random seed for reproducibility
        """
        self.dev_ratio = dev_ratio
        self.db_quotas = db_quotas
        self.failure_quotas = failure_quotas
        self.seed = seed
        
        self.stats = SplitStats()
        
        random.seed(seed)
    
    def _get_db_quota(self, db_id: str) -> int:
        """Get minimum quota for a database family."""
        if not self.db_quotas:
            return 0
        
        # Try to get from quotas
        db_lower = db_id.lower().replace('-', '_')
        return getattr(self.db_quotas, db_lower, 0)
    
    def _get_failure_quota(self, failure_type: str) -> int:
        """Get minimum quota for a failure type."""
        if not self.failure_quotas:
            return 0
        
        ft_lower = failure_type.lower().replace('-', '_')
        return getattr(self.failure_quotas, ft_lower, 0)
    
    def split(
        self,
        examples: List[RepairExample],
        is_clean: bool = False,
    ) -> Tuple[List[RepairExample], List[RepairExample]]:
        """
        Split examples into train and dev sets.
        
        Args:
            examples: Examples to split
            is_clean: Whether this is the clean pool (for dev constraints)
        
        Returns:
            (train_examples, dev_examples)
        """
        if not examples:
            return [], []
        
        self.stats.total_input = len(examples)
        
        # Group by (db_family, failure_type) for stratification
        groups: Dict[Tuple[str, str], List[RepairExample]] = defaultdict(list)
        
        for ex in examples:
            key = (ex.metadata.db_family, ex.metadata.failure_type)
            groups[key].append(ex)
        
        train = []
        dev = []
        
        # Split each group
        for (db_id, failure_type), group_examples in groups.items():
            # Shuffle group
            random.shuffle(group_examples)
            
            # Calculate split point
            n = len(group_examples)
            n_dev = max(1, int(n * self.dev_ratio)) if n > 1 else 0
            
            # Split
            group_dev = group_examples[:n_dev]
            group_train = group_examples[n_dev:]
            
            # Update splits
            dev.extend(group_dev)
            train.extend(group_train)
            
            # Update example metadata
            for ex in group_dev:
                ex.metadata.split = "dev"
            for ex in group_train:
                ex.metadata.split = "train"
        
        # Check quotas
        self._check_quotas(train, is_clean)
        
        # Update stats
        self.stats.train_count = len(train)
        self.stats.dev_count = len(dev)
        
        for ex in train:
            db = ex.metadata.db_family
            ft = ex.metadata.failure_type
            self.stats.train_by_db[db] = self.stats.train_by_db.get(db, 0) + 1
            self.stats.train_by_failure_type[ft] = \
                self.stats.train_by_failure_type.get(ft, 0) + 1
        
        for ex in dev:
            db = ex.metadata.db_family
            ft = ex.metadata.failure_type
            self.stats.dev_by_db[db] = self.stats.dev_by_db.get(db, 0) + 1
            self.stats.dev_by_failure_type[ft] = \
                self.stats.dev_by_failure_type.get(ft, 0) + 1
        
        return train, dev
    
    def _check_quotas(self, train: List[RepairExample], is_clean: bool) -> None:
        """Check if quotas are met and record violations."""
        if not self.db_quotas and not self.failure_quotas:
            return
        
        # Count by db and failure type
        db_counts: Dict[str, int] = defaultdict(int)
        ft_counts: Dict[str, int] = defaultdict(int)
        
        for ex in train:
            db_counts[ex.metadata.db_family] += 1
            ft_counts[ex.metadata.failure_type] += 1
        
        # Check db quotas
        if self.db_quotas:
            for db_id in [
                "california_schools", "financial", "formula_1",
                "thrombosis_prediction", "debit_card_specializing",
                "card_games", "toxicology", "codebase_community",
                "european_football_2", "student_club", "superhero",
            ]:
                quota = self._get_db_quota(db_id)
                actual = db_counts.get(db_id, 0)
                if quota > 0 and actual < quota:
                    self.stats.quota_violations.append(
                        f"DB {db_id}: {actual}/{quota} (shortfall: {quota - actual})"
                    )
        
        # Check failure type quotas
        if self.failure_quotas:
            for ft in [
                "wrong_return_field", "wrong_count_granularity",
                "wrong_denominator", "wrong_cohort_definition",
                "join_path_error", "temporal_anchor_error",
                "table_family_confusion", "alias_error",
                "wrong_table_side_error", "missing_distinct",
                "syntax_local_error", "degenerate_or_truncated_sql",
            ]:
                quota = self._get_failure_quota(ft)
                actual = ft_counts.get(ft, 0)
                if quota > 0 and actual < quota:
                    self.stats.quota_violations.append(
                        f"Failure type {ft}: {actual}/{quota} (shortfall: {quota - actual})"
                    )
    
    def get_stats(self) -> SplitStats:
        """Get splitting statistics."""
        return self.stats
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate splitting report."""
        return self.stats.to_dict()


def split_pools(
    clean_examples: List[RepairExample],
    internal_examples: List[RepairExample],
    dev_ratio: float = 0.1,
    db_quotas: Optional[DatabaseFamilyQuotas] = None,
    failure_quotas: Optional[FailureFamilyQuotas] = None,
    seed: int = 42,
) -> Dict[str, List[RepairExample]]:
    """
    Convenience function to split both pools.
    
    Returns:
        {
            "clean_train": [...],
            "clean_dev": [...],
            "internal_train": [...],
            "internal_dev": [...],
        }
    """
    # Split clean pool
    clean_splitter = Splitter(
        dev_ratio=dev_ratio,
        db_quotas=db_quotas,
        failure_quotas=failure_quotas,
        seed=seed,
    )
    clean_train, clean_dev = clean_splitter.split(clean_examples, is_clean=True)
    
    # Split internal pool
    internal_splitter = Splitter(
        dev_ratio=dev_ratio,
        db_quotas=db_quotas,
        failure_quotas=failure_quotas,
        seed=seed + 1,  # Different seed for internal
    )
    internal_train, internal_dev = internal_splitter.split(internal_examples, is_clean=False)
    
    return {
        "clean_train": clean_train,
        "clean_dev": clean_dev,
        "internal_train": internal_train,
        "internal_dev": internal_dev,
        "clean_stats": clean_splitter.get_stats().to_dict(),
        "internal_stats": internal_splitter.get_stats().to_dict(),
    }
