"""Lemmatized dictionary matching; exact and approved variants only."""
import csv
from collections import defaultdict
from pathlib import Path
from .preprocessing import lemmas, ngrams

def load_lexicon(path: str | Path) -> list[dict]:
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def build_index(rows: list[dict], include_variants: bool=True) -> dict:
    idx=defaultdict(list)
    for i,row in enumerate(rows):
        terms=[row['термин']]
        if include_variants:
            terms += [s.strip() for s in row.get('синонимы_и_варианты','').split(';') if s.strip()]
        for term in terms:
            key=tuple(lemmas(term))
            if key:
                idx[key].append((i,term))
    return dict(idx)

def match_lexicon(text: str, rows: list[dict], index: dict | None=None) -> list[dict]:
    index = index if index is not None else build_index(rows)
    ls=lemmas(text)
    hits=[]
    for n in sorted({len(k) for k in index}):
        for start,gram in enumerate(ngrams(ls,n)):
            for i,term in index.get(gram,[]):
                hits.append({'row_id':i,'component':rows[i]['компонент'],
                             'term':rows[i]['термин'],'matched_as':term,
                             'start':start,'end':start+n,
                             'weight':float(rows[i].get('вес_черновой') or 1)})
    # Slide 9: each lexicon entry is an independent binary feature.
    # Never suppress a shorter term because a longer expression overlaps it:
    # otherwise x_i depends on matching order and dictionary composition.
    unique = {(h['row_id'],h['start'],h['end']):h for h in hits}
    return sorted(unique.values(), key=lambda x:(x['start'],x['component'],x['row_id']))
