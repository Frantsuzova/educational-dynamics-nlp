"""Four-component scoring with auditable normalization.

IMPORTANT: these newly operationalized scores are not interchangeable with
historical score_objects/score_norms/score_meaning/score_system in the
existing processed_responses.csv; that file has no underlying raw texts.
"""
from __future__ import annotations
from collections import defaultdict
from .lexicon_matching import match_lexicon
from .preprocessing import lemmas

COMPONENT_NAMES = {
    'Наследие и объекты': 'score_objects',
    'Нормы и практики': 'score_norms',
    'Смыслы и интерпретация': 'score_meaning',
    'Социальная система и связи': 'score_system',
}

def score_components(text: str, rows: list[dict], index: dict | None = None,
                     *, scale: float = 100.0) -> tuple[dict, list[dict]]:
    """Weighted matches per `scale` tokens (unbounded; usually per 100 words).

    Separate component scores can overlap. Longest-per-component matching
    avoids double counting nested expressions inside one component. This is
    NOT a probability, a fraction of the dictionary, or a 0..1 metric.
    """
    hits = match_lexicon(text, rows, index)
    totals = defaultdict(float)
    for h in hits:
        totals[h['component']] += h['weight']
    size = len(lemmas(text))
    scores = {field: round(scale * totals[name] / size, 6) if size else 0.0
              for name, field in COMPONENT_NAMES.items()}
    return scores, hits

def score_components_relative(text: str, rows: list[dict], index: dict | None = None) -> dict:
    """Optional relative profile: each component's share of total weighted hits.

    Values sum to 1 when any matches exist; 0 otherwise. Relative profiles
    obscure absolute coverage, so use alongside raw density scores.
    """
    scores, _ = score_components(text, rows, index)
    total = sum(scores.values())
    return {k + '_relative': round(v/total, 6) if total else 0.0
            for k, v in scores.items()}
