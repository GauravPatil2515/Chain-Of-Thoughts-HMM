"""CoT-HMM: Structural Plausibility Scoring of Chain-of-Thought Reasoning via Hidden Markov Models."""

from src.tagging.rule_based_tagger import segment_and_clean, tag_steps
from src.hmm.scorer import EnsembleScorer
from src.hmm.trainer import EnsembleHMMTrainer

__version__ = "1.0.0"

__all__ = [
    "segment_and_clean",
    "tag_steps",
    "EnsembleScorer",
    "EnsembleHMMTrainer",
]
