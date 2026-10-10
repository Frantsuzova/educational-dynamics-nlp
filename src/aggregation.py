"""Recreate cohort/timepoint summaries from existing processed_responses.csv."""
import csv
import statistics
from collections import defaultdict

TEXT_METRICS=['course_concept_share','abstraction_score','relational_density','interpretation_score']
COMPONENT_METRICS=['score_objects','score_norms','score_meaning','score_system']

def aggregate(input_csv,output_csv,metrics):
    with open(input_csv,encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    out=[]
    for cohort in sorted({r['cohort'] for r in rows})+['all']:
        for tp in ['T0','T1','T2','T3']:
            subset=[r for r in rows if (cohort=='all' or r['cohort']==cohort) and r['timepoint']==tp]
            for metric in metrics:
                values=[float(r[metric]) for r in subset if r.get(metric) not in ('',None)]
                if not values: continue
                out.append({'cohort':cohort,'timepoint':tp,'metric':metric,
                            'n':len(values),'mean':round(statistics.mean(values),6),
                            'std':round(statistics.stdev(values),6) if len(values)>1 else 0,
                            'min':min(values),'max':max(values)})
    with open(output_csv,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['cohort','timepoint','metric','n','mean','std','min','max'])
        w.writeheader();w.writerows(out)
    return out
