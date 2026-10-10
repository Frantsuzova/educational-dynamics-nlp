"""Four-component weighted binary profile, exactly as defined on slide 9.

S_k = sum_i(w_ik*x_i) / sum_i(w_ik), x_i = 1 when the term
or its approved variant occurs at least once, 0 otherwise.
Scores are independent, between 0 and 1, and may overlap.
"""
from __future__ import annotations
from collections import defaultdict
from .lexicon_matching import match_lexicon

COMPONENT_NAMES = {
    'Наследие и объекты': 'score_objects',
    'Нормы и практики': 'score_norms',
    'Смыслы и интерпретация': 'score_meaning',
    'Социальная система и связи': 'score_system',
}

def score_components(text: str, rows: list[dict], index: dict | None = None,
                     *, scale: float = 1.0) -> tuple[dict, list[dict]]:
    if scale != 1.0:
        raise ValueError('Slide 9 uses a normalized 0..1 score; scale must be 1.')
    hits = match_lexicon(text, rows, index)
    present = {h['row_id'] for h in hits}
    denominator = defaultdict(float)
    numerator = defaultdict(float)
    for i, row in enumerate(rows):
        category = row['компонент']
        weight = float(row.get('вес_черновой') or 1)
        if weight < 0:
            raise ValueError('Weights must be nonnegative')
        denominator[category] += weight
        if i in present:
            numerator[category] += weight
    scores = {
        field: round(numerator[name] / denominator[name], 6)
        if denominator[name] else 0.0
        for name, field in COMPONENT_NAMES.items()
    }
    return scores, hits

def score_components_relative(text: str, rows: list[dict], index: dict | None = None) -> dict:
    """Optional diagnostics, NOT one of the presentation's scores."""
    scores, _ = score_components(text, rows, index)
    total = sum(scores.values())
    return {k + '_relative': round(v / total, 6) if total else 0.0
            for k, v in scores.items()}
