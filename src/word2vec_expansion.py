"""Word2Vec candidate generation; human approval required before matching.

A Word2Vec neighbor is semantically related, NOT necessarily a synonym.
Compatible with plain lemmatized Russian Word2Vec keyed vectors; POS-tagged
models require a matching tokenizer/adapter.
"""
from pathlib import Path
from .preprocessing import lemmas

def load_model(path: str):
    from gensim.models import KeyedVectors
    p = Path(path)
    if p.suffix == '.kv':
        return KeyedVectors.load(str(p), mmap='r')
    if p.suffix == '.bin':
        return KeyedVectors.load_word2vec_format(str(p), binary=True)
    return KeyedVectors.load_word2vec_format(str(p), binary=False)

def _vec(model, text):
    import numpy as np
    words = lemmas(text)
    vectors = [model[w] for w in words if w in model]
    if not vectors:
        return None
    v = np.mean(vectors, axis=0)
    return v / (np.linalg.norm(v) + 1e-12)

def candidates(rows, model, topn=10, min_similarity=0.72):
    """Generate up to `topn` neighbors per entry, with similarity scores.

    For phrases, a mean of constituent word vectors is used, so neighbors
    may be single words; this is a review queue, not an automatic match list.
    """
    for row in rows:
        v = _vec(model, row['термин'])
        if v is None:
            continue
        neighbors = model.similar_by_vector(v, topn=max(topn*8, 40))
        found = []
        original = set(lemmas(row['термин']))
        for token, sim in neighbors:
            if sim < min_similarity:
                continue
            if token in original or not token.replace('_', '').isalpha():
                continue
            if token not in [t for t, _ in found]:
                found.append((token, round(float(sim), 5)))
            if len(found) >= topn:
                break
        yield {'component': row['компонент'], 'seed': row['термин'],
               'candidates': ';'.join(t for t, _ in found),
               'similarities': ';'.join(str(s) for _, s in found),
               'source': 'word2vec_unreviewed'}
