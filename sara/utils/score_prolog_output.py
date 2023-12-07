import argparse
import re
import pdb
import glob
import os
import sys
from case import Case
import numpy as np
import scipy.stats as st
import json

parser = argparse.ArgumentParser(description='Score output log of Prolog run.')
parser.add_argument('--predictions', type=str, help='path to file output by running Prolog', required=True)
parser.add_argument('--cases', type=str, help='path to folder containing original cases', required=True)
args = parser.parse_args()

integer_regexp = re.compile(r"[\d]+")

def is_case_path(x):
    y=x.split('/')[-1]
    is_tax_case = y.startswith('tax_case')
    is_other_case = y.startswith('s')
    is_case_path = xor(is_tax_case, is_other_case)
    is_case = y.endswith('.pl')
    return is_case and is_case_path

def is_prediction(x):
    is_binary = x=="success"
    is_integer = integer_regexp.match(x) is not None
    return xor(is_binary,is_integer)

def xor(a,b):
    assert isinstance(a,bool)
    assert isinstance(b,bool)
    return a!=b

# parse log into useful data
# produce tuples of (caseid, *answers)
data = []
current = []
for line in open(args.predictions,'r'):
    x=line.strip('\n')
    assert xor(is_case_path(x),is_prediction(x))
    if is_case_path(x):
        caseid=x.split('/')[-1][:-3]
        if len(current)>0:
            data.append(tuple(current))
        current = [caseid]
    elif is_prediction(x):
        assert len(current)>0, pdb.set_trace()
        current.append(x)
if len(current)>0:
    data.append(tuple(current)) # append last tuple extracted

# load numerical cases to get extract ground truth
ground_truth={}
for filename in glob.glob(os.path.join(args.cases,'tax_case_*.pl')):
    key=os.path.basename(filename)[:-3]
    case = Case.from_file(filename)
    query = list(case.query)
    tax_term = query[0].args[1]
    tax_amount = tax_term.args[2].value
    assert isinstance(tax_amount, int)
    ground_truth.setdefault(key, tax_amount)
    assert ground_truth[key] == tax_amount

def evaluate_numerical_answer(output, gold):
    assert isinstance(output,float) or isinstance(output,int)
    assert isinstance(gold,int)
    delta=abs(output-gold)/max(0.1*gold,5000)
    return delta<1

# compute metrics, with confidence intervals
metrics = {'binary': [], 'numerical': []}
for data_point in data:
    caseid=data_point[0]
    assert xor(caseid.startswith('tax_case'), caseid.startswith('s'))
    if caseid.startswith('tax_case'):
        response=data_point[-1]
        if response==caseid:
            metrics['numerical'].append(0) # this might happen in case of a timeout
        else:
            gold=ground_truth[caseid]
            correct=evaluate_numerical_answer(int(response),gold)
            metrics['numerical'].append(int(correct))
    elif caseid.startswith('s'):
        response=data_point[-1]
        assert response in ['success',caseid]
        correct=response=='success'
        metrics['binary'].append(int(correct))
# end for data_point in data:
# compute metrics and confidence intervals
def compute_metrics(samples):
    m = np.mean(samples)
    lower, upper = st.t.interval(alpha=0.9, df=len(samples)-1, loc=m,
                scale=st.sem(samples))
    # check confidence interval
    delta = abs(2*m-lower-upper)/m
    assert delta < 1e-2
    pm = m-lower # this 'pm' is what will be written before and after the metric
    return {'mean': m, 'confidence interval': pm}
metrics['binary'] = compute_metrics(metrics['binary'])
metrics['numerical'] = compute_metrics(metrics['numerical'])
num_places=1
def format_percentage(x):
    assert isinstance(x,float)
    return round(100*x,num_places)
for k,d in metrics.items():
    for k2,v in d.items():
        metrics[k][k2]=format_percentage(v)
print("metrics in percentage and rounded to {} figure(s) after the comma".format(num_places))
print(json.dumps(metrics,indent=2,sort_keys=True))
for key in ['binary', 'numerical']:
    print(key)
    print('{} +/- {}'.format(metrics[key]['mean'], metrics[key]['confidence interval']))
