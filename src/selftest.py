"""Quick smoke test that does not require a downloaded Word2Vec model."""
from . import preprocessing as pp
from .lexicon_matching import build_index, match_lexicon
from .concept_scoring import score_components, score_components_relative
from .text_metrics import calculate_text_metrics

def run():
    # Use an identity lemmatizer for a pure-Python logic test. This test checks
    # matching, n-grams and formula plumbing, not Russian morphology quality.
    original = pp.lemma
    pp.lemma = lambda x: x.lower().replace('ё','е')
    try:
        rows = [
            {'компонент':'Наследие и объекты','термин':'музей', 'вес_черновой':'1'},
            {'компонент':'Нормы и практики','термин':'социальные нормы','вес_черновой':'0.5'},
            {'компонент':'Смыслы и интерпретация','термин':'культурная память','вес_черновой':'1'},
            {'компонент':'Социальная система и связи','термин':'общество','вес_черновой':'1'},
        ]
        idx = build_index(rows)
        text = 'Музей и общество. Культурная память влияет на социальные нормы.'
        p = pp.preprocess(text, 4)
        assert p['bigrams'] and p['trigrams'] and p['fourgrams']
        hits = match_lexicon(text, rows, idx)
        assert len(hits) == 4, hits
        scores, _ = score_components(text, rows, idx)
        assert all(v >= 0 for v in scores.values())
        rel = score_components_relative(text, rows, idx)
        assert abs(sum(rel.values())-1) < 1e-5
        metrics = calculate_text_metrics(text, rows, idx)
        assert all(0 <= metrics[k] <= 1 for k in ('course_concept_share','abstraction_score','relational_density','interpretation_score'))
        assert metrics['graph_vertices'] == 4
        print('Smoke test passed: ngrams, matching, scoring, metrics.')
    finally:
        pp.lemma = original

if __name__ == '__main__':
    run()
