"""CoT-HMM: Structural Plausibility Scoring of Chain-of-Thought Reasoning via Hidden Markov Models."""

from pathlib import Path
from src.tagging.rule_based_tagger import segment_and_clean, tag_steps
from src.hmm.scorer import EnsembleScorer
from src.hmm.trainer import EnsembleHMMTrainer

__version__ = "1.0.0"

_DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "ensemble_scorer.pkl"


def load_default_scorer() -> EnsembleScorer:
    """Load the bundled pre-trained 12-HMM ensemble scorer."""
    if not _DEFAULT_MODEL_PATH.exists():
        repo_fallback = Path(__file__).parent.parent / "models" / "production_ensemble" / "ensemble_scorer.pkl"
        if repo_fallback.exists():
            return EnsembleScorer.load(str(repo_fallback))
        raise FileNotFoundError(
            f"Bundled model not found at {_DEFAULT_MODEL_PATH}. "
            "Train a new model using EnsembleHMMTrainer."
        )
    return EnsembleScorer.load(str(_DEFAULT_MODEL_PATH))


__all__ = [
    "segment_and_clean",
    "tag_steps",
    "EnsembleScorer",
    "EnsembleHMMTrainer",
    "load_default_scorer",
]
