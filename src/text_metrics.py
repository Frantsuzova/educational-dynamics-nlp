"""Transparent operational proxies for four text metrics (not historical calibration).

All four scores are in [0, 1]. They are text features, NOT validated measures
of student understanding. Formulae are documented below.
"""
from __future__ import annotations
from collections import Counter
from itertools import combinations
from .preprocessing import lemmas, sentences
from .lexicon_matching import match_lexicon

# Explicit, editable lexical markers. The interpretation score counts
# explanatory constructions, not the correctness of an interpretation.
ABSTRACT_ROOTS = frozenset('культура общество ценность норма смысл значение система процесс изменение развитие отношение связь идентичность коммуникация взаимодействие поведение представление сознание традиция память информация интерпретация символ правило практика функция'.split())
EXPLANATORY_PATTERNS = (
    ('потому', 'что'), ('так', 'как'), ('для', 'того', 'чтобы'),
    ('это', 'означать'), ('это', 'позволять'), ('это', 'приводить'),
    ('в', 'результат'), ('в', 'следствие'), ('например',),
    ('следовательно',), ('поэтому',), ('поскольку',), ('благодаря',),
    ('влиять', 'на'), ('связанный', 'с'), ('зависеть', 'от'),
)

def _bounded(x: float) -> float:
    return round(max(0.0, min(1.0, x)), 6)

def _contains_pattern(seq: list[str], pattern: tuple[str, ...]) -> bool:
    n = len(pattern)
    return any(tuple(seq[i:i+n]) == pattern for i in range(len(seq)-n+1))

def _concept_spans(hits: list[dict]) -> list[tuple[int, int, str]]:
    """Deduplicate spans shared across components."""
    return sorted({(h['start'], h['end'], h['term']) for h in hits})

def _concept_graph_density(ls: list[str], hits: list[dict], window: int = 8) -> tuple[float, int, int]:
    """Undirected concept co-occurrence graph; window in lemmatized tokens.

    Vertices are distinct canonical concepts; edges exist if occurrences are
    at most `window` token positions apart. Density = 2E / V(V-1).
    V<2 yields 0. This measures co-occurrence, not logical coherence.
    """
    spans = _concept_spans(hits)
    vertices = {term for _, _, term in spans}
    if len(vertices) < 2:
        return 0.0, len(vertices), 0
    edges = set()
    for a, b in combinations(spans, 2):
        if a[2] == b[2]:
            continue
        gap = max(0, max(a[0], b[0]) - min(a[1], b[1]))
        if gap <= window:
            edges.add(tuple(sorted((a[2], b[2]))))
    v = len(vertices)
    return _bounded(2*len(edges)/(v*(v-1))), v, len(edges)

def calculate_text_metrics(text: str, rows: list[dict], index: dict | None = None,
                           *, graph_window: int = 8) -> dict:
    """Four operational scores plus audit fields.

    course_concept_share: unique token positions covered by dictionary hits /
      total tokens. A multiword hit covers each of its tokens once.
    abstraction_score: proportion of content tokens whose lemma is in the
      editable ABSTRACT_ROOTS list (heuristic, not an abstractness classifier).
    relational_density: graph density of dictionary concept co-occurrences
      within graph_window tokens.
    interpretation_score: proportion of sentences containing >=1 explanatory
      pattern; 0 if no sentences. No claim about argument quality.
    """
    ls = lemmas(text)
    hits = match_lexicon(text, rows, index)
    covered = {p for h in hits for p in range(h['start'], h['end'])}
    share = _bounded(len(covered)/len(ls)) if ls else 0.0
    abstract = _bounded(sum(w in ABSTRACT_ROOTS for w in ls)/len(ls)) if ls else 0.0
    density, vertices, edges = _concept_graph_density(ls, hits, graph_window)
    sentence_ls = [lemmas(s) for s in sentences(text)]
    sentence_ls = [s for s in sentence_ls if s]
    explained = sum(any(_contains_pattern(s, p) for p in EXPLANATORY_PATTERNS) for s in sentence_ls)
    interpretation = _bounded(explained/len(sentence_ls)) if sentence_ls else 0.0
    return {'course_concept_share': share, 'abstraction_score': abstract,
            'relational_density': density, 'interpretation_score': interpretation,
            'token_count': len(ls), 'covered_tokens': len(covered),
            'graph_vertices': vertices, 'graph_edges': edges,
            'sentence_count': len(sentence_ls), 'explanatory_sentences': explained}
