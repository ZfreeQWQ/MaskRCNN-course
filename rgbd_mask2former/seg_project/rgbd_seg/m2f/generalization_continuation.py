"""Preserve optimizer moments while retaining the focus mapper and mature LRs."""
from .algorithm_continuation import ContinuationTrainer
from .generalization import GeneralizationTrainer


class GeneralizationContinuationTrainer(ContinuationTrainer, GeneralizationTrainer):
    """MRO combines audited full-state restore with the opt-in training recipe.

    ContinuationTrainer's super() enters GeneralizationTrainer, retaining the
    focus mapper, diagnostics and mature-module optimizer LR configuration.
    """
    pass
