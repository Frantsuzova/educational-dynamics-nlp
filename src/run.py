"""CLI: run from repository root with `python -m src.run ...`."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from .preprocessing import preprocess
from .lexicon_matching import load_lexicon, build_index
from .concept_scoring import score_components, score_components_relative
from .text_metrics import calculate_text_metrics
from .aggregation import aggregate, TEXT_METRICS, COMPONENT_METRICS
from .word2vec_expansion import load_model, candidates

def write_rows(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def main():
    p = argparse.ArgumentParser(description='Educational dynamics NLP pipeline')
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('analyze', help='Compute features from RAW response texts')
    a.add_argument('--input', required=True)
    a.add_argument('--lexicon', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--text-column', default='response_text')
    a.add_argument('--approved-w2v', help='CSV: component,seed,approved_terms (semicolon-separated)')
    a.add_argument('--graph-window', type=int, default=8)
    b = sub.add_parser('aggregate', help='Aggregate existing processed features')
    b.add_argument('--input', required=True)
    b.add_argument('--out-dir', required=True)
    c = sub.add_parser('w2v-candidates', help='Generate review queue from pretrained vectors')
    c.add_argument('--lexicon', required=True)
    c.add_argument('--model', required=True)
    c.add_argument('--output', required=True)
    c.add_argument('--threshold', type=float, default=.72)
    c.add_argument('--topn', type=int, default=10)
    args = p.parse_args()
    if args.cmd == 'aggregate':
        d = Path(args.out_dir)
        d.mkdir(parents=True, exist_ok=True)
        aggregate(args.input, d/'text_metrics_dynamics.csv', TEXT_METRICS)
        aggregate(args.input, d/'concept_components_dynamics.csv', COMPONENT_METRICS)
        return
    rows = load_lexicon(args.lexicon)
    if args.cmd == 'w2v-candidates':
        model = load_model(args.model)
        found = list(candidates(rows, model, topn=args.topn, min_similarity=args.threshold))
        write_rows(args.output, found, ['component','seed','candidates','similarities','source'])
        return
    if args.graph_window < 0:
        p.error('--graph-window must be >= 0')
    if args.approved_w2v:
        with open(args.approved_w2v, encoding='utf-8-sig', newline='') as f:
            approved_rows = list(csv.DictReader(f))
        approved = {(r['component'], r['seed']): [x.strip() for x in r['approved_terms'].split(';') if x.strip()]
                    for r in approved_rows}
        for row in rows:
            extra = approved.get((row['компонент'], row['термин']), [])
            if extra:
                row['синонимы_и_варианты'] = ';'.join(filter(None, [row.get('синонимы_и_варианты',''), *extra]))
    idx = build_index(rows)
    with open(args.input, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or args.text_column not in reader.fieldnames:
            raise ValueError(f'Missing raw text column {args.text_column!r}. '
                             'processed_responses.csv contains derived scores, not raw responses.')
        data = list(reader)
    output = []
    for row in data:
        text = row[args.text_column] or ''
        prep = preprocess(text, max_n=4)
        components, hits = score_components(text, rows, idx)
        relative = score_components_relative(text, rows, idx)
        metrics = calculate_text_metrics(text, rows, idx, graph_window=args.graph_window)
        output.append({**row, 'lemmas': ' '.join(prep['lemmas']),
                       'bigrams': ' | '.join(prep['bigrams']),
                       'trigrams': ' | '.join(prep['trigrams']),
                       'fourgrams': ' | '.join(prep['fourgrams']),
                       'lexicon_hits': ' | '.join(f"{h['component']}:{h['term']}" for h in hits),
                       **metrics, **components, **relative})
    columns = list(data[0]) if data else []
    columns += [k for k in (output[0] if output else {}) if k not in columns]
    write_rows(args.output, output, columns or [args.text_column])

if __name__ == '__main__':
    main()
