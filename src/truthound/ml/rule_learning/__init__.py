"""Rule learning module for Truthound.

Automatically learns validation rules from data characteristics.
Supports:
- Profile-based rule generation
- Constraint mining
- Pattern-based rules

Example:
    >>> from truthound.ml.rule_learning import DataProfileRuleLearner
    >>> learner = DataProfileRuleLearner()
    >>> result = learner.learn_rules(data)
    >>> for rule in result.rules:
    ...     print(f"{rule.name}: {rule.condition}")
"""

from truthound.ml.rule_learning.profile_learner import DataProfileRuleLearner
from truthound.ml.rule_learning.constraint_miner import ConstraintMiner
from truthound.ml.rule_learning.pattern_learner import PatternRuleLearner

__all__ = [
    "DataProfileRuleLearner",
    "ConstraintMiner",
    "PatternRuleLearner",
]
