"""Independent perfect, constant, and invalid predictors audit paired summaries."""
import pytest
from ctm_transformer.query_analysis import summarize_grid,NODES


def rows(mode):
    mapping={n:NODES[(i+1)%8] for i,n in enumerate(NODES)}
    result=[]
    for n in NODES:
        prediction=mapping[n] if mode=='perfect' else mapping['A'] if mode=='constant' else 'Z'
        result.append({'id':'one-map','start':n,'original_start':'A','successors':mapping,
            'answer':mapping[n],'prediction':prediction,'terminated':True,'correct':prediction==mapping[n],
            'query_exposure':'trained_query' if n=='A' else 'changed_training_map_query'})
    return result


def test_perfect_predictor():
    s=summarize_grid(rows('perfect'))
    assert s['overall']['exact_match']==1 and s['all_queries_correct_maps']==1
    assert s['same_output_pairs']==0 and s['changed_queries_output_original_answer']==0
    assert s['distinct_valid_answers_histogram']['8']==1
    for n in NODES:assert s['output_implied_source_counts'][n][n]==1


def test_constant_predictor_is_one_correct_and_twenty_eight_collisions():
    s=summarize_grid(rows('constant'))
    assert s['overall']['exact_match']==1/8 and s['all_queries_correct_maps']==0
    assert s['correct_queries_histogram']['1']==1
    assert s['same_output_pairs']==28 and s['changed_queries_output_original_answer']==7
    for n in NODES:assert s['output_implied_source_counts'][n]['A']==1


def test_invalid_output_and_missing_eos_are_not_source_retrieval():
    records=rows('invalid');records[0].update(prediction='B',terminated=False,correct=False)
    s=summarize_grid(records)
    assert s['invalid_outputs']==8 and s['overall']['correct']==0
    assert s['distinct_valid_answers_histogram']['0']==1
    for n in NODES:assert s['output_implied_source_counts'][n]['?']==1


def test_rejects_incomplete_grid_and_false_metric():
    with pytest.raises(ValueError,match='eight'):summarize_grid(rows('perfect')[:-1])
    records=rows('perfect');records[0]['correct']=False
    with pytest.raises(ValueError,match='exact-match'):summarize_grid(records)
