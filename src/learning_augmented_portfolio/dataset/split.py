"""
Chronological split assignment and temporal ordering integrity.
Enforces the strict invariant: max(Date_Train) < min(Date_Validation) < min(Date_Test).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Sequence, Tuple, Any
import numpy as np


class DatasetSplit(str, Enum):
    """Dataset partition enum."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class DatasetSplitConfig:
    """
    Chronological partition boundaries.
    
    Attributes:
        train_end: Inclusive boundary for training set (calendar date string or integer index).
        validation_end: Inclusive boundary for validation set.
        test_end: Optional inclusive boundary for test set.
    """
    train_end: Union[int, str]
    validation_end: Union[int, str]
    test_end: Optional[Union[int, str]] = None

    def __post_init__(self) -> None:
        if self.train_end >= self.validation_end:
            raise ValueError(
                f"Invalid chronological split: train_end ({self.train_end}) must precede "
                f"validation_end ({self.validation_end})"
            )
        if self.test_end is not None and self.validation_end >= self.test_end:
            raise ValueError(
                f"Invalid chronological split: validation_end ({self.validation_end}) must precede "
                f"test_end ({self.test_end})"
            )

    def assign_split(self, date_or_index: Union[int, str]) -> DatasetSplit:
        """Assign an instance to train, validation, or test based on its timestamp/index."""
        if date_or_index <= self.train_end:
            return DatasetSplit.TRAIN
        elif date_or_index <= self.validation_end:
            return DatasetSplit.VALIDATION
        else:
            if self.test_end is not None and date_or_index > self.test_end:
                raise ValueError(
                    f"Timestamp {date_or_index} exceeds configured test_end {self.test_end}"
                )
            return DatasetSplit.TEST

    @classmethod
    def from_proportions(
        cls,
        total_steps: int,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> DatasetSplitConfig:
        """Create split boundaries from step counts and relative ratios."""
        if train_ratio <= 0 or val_ratio <= 0 or (train_ratio + val_ratio) >= 1.0:
            raise ValueError(f"Invalid split ratios: train={train_ratio}, val={val_ratio}")
        n_train = int(np.floor(total_steps * train_ratio))
        n_val = int(np.floor(total_steps * val_ratio))
        train_end = n_train - 1
        val_end = train_end + n_val
        test_end = total_steps - 1
        return cls(train_end=train_end, validation_end=val_end, test_end=test_end)


def validate_chronological_integrity(
    split_records: Sequence[Tuple[Union[int, str], DatasetSplit]],
) -> None:
    """
    Verify that the chronological ordering invariant holds across all records:
        max(Date_train) < min(Date_validation)
        max(Date_validation) < min(Date_test)
    """
    train_dates = [d for d, s in split_records if s == DatasetSplit.TRAIN]
    val_dates = [d for d, s in split_records if s == DatasetSplit.VALIDATION]
    test_dates = [d for d, s in split_records if s == DatasetSplit.TEST]

    if train_dates and val_dates:
        max_train = max(train_dates)
        min_val = min(val_dates)
        if max_train >= min_val:
            raise ValueError(
                f"Chronological leak detected: max(Date_train)={max_train} >= min(Date_val)={min_val}"
            )

    if val_dates and test_dates:
        max_val = max(val_dates)
        min_test = min(test_dates)
        if max_val >= min_test:
            raise ValueError(
                f"Chronological leak detected: max(Date_val)={max_val} >= min(Date_test)={min_test}"
            )

    if train_dates and test_dates and not val_dates:
        max_train = max(train_dates)
        min_test = min(test_dates)
        if max_train >= min_test:
            raise ValueError(
                f"Chronological leak detected: max(Date_train)={max_train} >= min(Date_test)={min_test}"
            )
