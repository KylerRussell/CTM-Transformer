"""Map-level behavioral summaries for exhaustive one-hop query interventions."""
from collections import Counter,defaultdict
from itertools import combinations

NODES='ABCDEFGH'


def counts(rows):
    n=len(rows);correct=sum(r['correct'] for r in rows)
    return {'examples':n,'correct':correct,'exact_match':correct/n if n else None,
            'terminated':sum(r['terminated'] for r in rows)}


def summarize_grid(rows):
    maps=defaultdict(list)
    for r in rows:maps[r['id']].append(r)
    if not maps:raise ValueError('Empty query grid')
    confusion={n:{k:0 for k in NODES+'?'} for n in NODES}
    correct_hist=Counter();distinct_hist=Counter();map_rows=[]
    collisions=0;persisted=0;invalid=0
    for key,group in maps.items():
        if len(group)!=8 or {r['start'] for r in group}!=set(NODES):raise ValueError('Each map requires exactly eight unique starts')
        original_starts={r['original_start'] for r in group}
        if len(original_starts)!=1:raise ValueError('Original query annotation differs within map')
        original=next(r for r in group if r['start']==r['original_start'])
        mapping=original['successors']
        if set(mapping)!=set(NODES) or set(mapping.values())!=set(NODES):raise ValueError('Expected permutation map')
        inverse={v:k for k,v in mapping.items()}
        for row in group:
            if row['successors']!=mapping or row['answer']!=mapping[row['start']]:raise ValueError('Inconsistent map or target')
            expected_correct=row['terminated'] and row['prediction']==row['answer']
            if row['correct']!=expected_correct:raise ValueError('Incorrect exact-match flag')
            source=inverse.get(row['prediction'],'?') if row['terminated'] else '?'
            confusion[row['start']][source]+=1;invalid+=int(source=='?')
            if row['start']!=row['original_start']:
                persisted+=int(row['terminated'] and row['prediction']==original['answer'])
        same=sum((a['prediction'],a['terminated'])==(b['prediction'],b['terminated']) for a,b in combinations(group,2))
        collisions+=same
        correct=sum(r['correct'] for r in group)
        distinct=len({r['prediction'] for r in group if r['terminated'] and r['prediction'] in NODES and len(r['prediction'])==1})
        correct_hist[correct]+=1;distinct_hist[distinct]+=1
        map_rows.append({'id':key,'correct_queries':correct,'all_correct':correct==8,
                         'same_output_pairs':same,'distinct_valid_answers':distinct})
    exposures=sorted({r['query_exposure'] for r in rows})
    return {'overall':counts(rows),'by_start':{n:counts([r for r in rows if r['start']==n]) for n in NODES},
            'by_exposure':{e:counts([r for r in rows if r['query_exposure']==e]) for e in exposures},
            'maps':len(maps),'all_queries_correct_maps':correct_hist[8],
            'all_queries_correct_map_fraction':correct_hist[8]/len(maps),
            'correct_queries_histogram':{str(n):correct_hist[n] for n in range(9)},
            'distinct_valid_answers_histogram':{str(n):distinct_hist[n] for n in range(9)},
            'same_output_pairs':collisions,'total_start_pairs':28*len(maps),
            'same_output_pair_rate':collisions/(28*len(maps)),
            'changed_queries':7*len(maps),'changed_queries_output_original_answer':persisted,
            'original_answer_persistence_rate':persisted/(7*len(maps)),
            'invalid_outputs':invalid,'output_implied_source_counts':confusion,'per_map':map_rows}
