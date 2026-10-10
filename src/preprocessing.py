"""Russian lemmatization and contiguous n-grams (with stopwords retained)."""
import re
from functools import lru_cache
from itertools import chain

TOKEN_RE = re.compile(r"[а-яёa-z]+(?:-[а-яёa-z]+)*", re.I)

@lru_cache(maxsize=1)
def _morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()

@lru_cache(maxsize=150_000)
def lemma(token: str) -> str:
    return _morph().parse(token.lower().replace('ё','е'))[0].normal_form.replace('ё','е')

def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower().replace('ё','е'))

def lemmas(text: str) -> list[str]:
    return [lemma(t) for t in tokens(text)]

def ngrams(words: list[str], n: int) -> list[tuple[str,...]]:
    return list(zip(*(words[i:] for i in range(n)))) if len(words)>=n else []

def sentences(text: str) -> list[str]:
    """Simple sentence segmentation; punctuation-delimited, not a syntax parser."""
    return [s.strip() for s in re.split(r"[.!?]+", str(text)) if s.strip()]

def preprocess(text: str, max_n: int = 3) -> dict:
    ls=lemmas(text)
    return {'tokens':tokens(text), 'lemmas':ls,
            'bigrams':[' '.join(x) for x in ngrams(ls,2)],
            'trigrams':[' '.join(x) for x in ngrams(ls,3)],
            'fourgrams':[' '.join(x) for x in ngrams(ls,4)],
            'ngrams':list(chain.from_iterable(ngrams(ls,n) for n in range(1,max_n+1)))}
