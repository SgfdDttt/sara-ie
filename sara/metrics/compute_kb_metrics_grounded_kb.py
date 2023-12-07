import sys
import json
import argparse
import glob
import os.path as path
import problog
from set_based_f1 import SetBasedF1Measure
from problog.program import PrologString
from problog.logic import term2str

parser = argparse.ArgumentParser(description='Compute metrics against full KB.')
parser.add_argument('--reference', type=str, help='path to directory containing reference cases', required=True)
parser.add_argument('--prediction', type=str, help='path to file containing model predictions', required=True)
args = parser.parse_args()

def list2tuple(l):
    if not isinstance(l, list):
        return l
    return tuple([list2tuple(x) for x in l])

def span2tuple(x):
    assert x.functor == 'span'
    a = x.args
    assert len(a) == 3
    output = tuple([z.compute_value() for z in a])
    type_checking_arg(output), output
    return output

def prolog2tuple(kb):
    output = set()
    for kbe in kb:
        ps = PrologString(kbe)
        ps = [x for x in ps]
        assert len(ps) == 1
        ps = ps.pop()
        f = ps.functor
        a = ps.args # each of a is a span(...)
        at = tuple([span2tuple(x) for x in a])
        tuple_kbe = (f,) + at
        type_checking_kbe(tuple_kbe)
        output.add(tuple_kbe)
    # end for kbe in kb:
    return output

def type_checking_kb(kb):
    assert isinstance(kb, set), kb
    for kbe in kb:
        type_checking_kbe(kbe)
# end def type_checking(kb):

def type_checking_kbe(kbe):
    assert isinstance(kbe, tuple), kbe
    assert len(kbe) in [2, 3], kbe
    predicate = kbe[0]
    args = kbe[1:]
    assert isinstance(predicate, str), predicate
    for a in args:
        type_checking_arg(a)
# end def type_checking(kbe):

def type_checking_arg(a):
    assert len(a) == 3, a
    value, start_idx, end_idx = a
    assert isinstance(value, str) or isinstance(value, int), "{} ({})".format(value, type(value))
    assert isinstance(start_idx, int), start_idx
    assert isinstance(end_idx, int), end_idx
# end def type_checking_arg(a):

# read reference
reference = {} # map caseid -> set of tuples
for filename in glob.glob(path.join(args.reference, '*.json')):
    stuff = json.load(open(filename, 'r'))
    kb = stuff['facts']
    case_id = path.basename(filename)[:-5]
    assert case_id not in reference
    tuple_kb = set(prolog2tuple(kb))
    type_checking_kb(tuple_kb)
    reference[case_id] = tuple_kb
# end for filename in glob.glob(path.join(args.reference, '*.json')):
# read predictions
predictions = {} # map caseid -> set of tuples
capped_kbs = {} # map caseid -> capped KB, without the unextractable spans
for line in open(args.prediction):
    stuff = json.loads(line.strip('\n'))
    kbs = stuff['kbs']
    metadata = stuff['metadata']
    assert len(kbs) == len(metadata)
    for kb, md in zip(kbs, metadata):
        tuple_kb = set(list2tuple(x) for x in kb)
        type_checking_kb(tuple_kb)
        case_id = md['id'].split('-')[0].strip()
        predictions.setdefault(case_id, set())
        predictions[case_id] = predictions[case_id] | tuple_kb
        capped_kbs.setdefault(case_id, set())
        tuple_ckb = set(list2tuple(x) for x in md['kbes'])
        type_checking_kb(tuple_ckb)
        capped_kbs[case_id] = capped_kbs[case_id] | tuple_ckb
    # end for kb, md in zip(kbs, metadata):
# end for line in open(args.prediction):
relevant_caseids = set(reference.keys()) & set(predictions.keys())
relevant_caseids = sorted(relevant_caseids)
# feed to metrics
refs = [reference[caseid] for caseid in relevant_caseids]
preds = [predictions[caseid] for caseid in relevant_caseids]
capped_refs = [capped_kbs[caseid] for caseid in relevant_caseids]
# sanity check
for thing in [refs, preds, capped_refs]:
    for kb in thing:
        type_checking_kb(kb)
for r, cr in zip(refs, capped_refs):
    assert cr <= r
def compute_metric(pred, ref):
    metric = SetBasedF1Measure()
    metric(pred, ref)
    # print metric
    print(metric.num_correct)
    m = metric.get_metric(True)
    def format_percentage(x):
        y = round(100*x, 1)
        return y
    for k, v in m.items():
        m[k] = format_percentage(v)
    m['N'] = len(relevant_caseids)
    return m
print('prediction vs reference - how well our model does against full KB')
print(json.dumps(compute_metric(preds, refs),sort_keys=True,indent=2))
print('prediction vs capped reference - these were computed as part of AllenNLP model')
print(json.dumps(compute_metric(preds, capped_refs),sort_keys=True,indent=2))
print('capped reference vs reference - the best metrics we can hope to achieve against real reference')
print(json.dumps(compute_metric(capped_refs, refs),sort_keys=True,indent=2))
