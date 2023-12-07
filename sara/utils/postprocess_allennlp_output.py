import argparse
import math
import json
import glob
import os
import sys
from case import Case
import copy
import pdb
from problog.logic import Term, Constant, Var, term2str
from problog.program import PrologString

parser = argparse.ArgumentParser(description='Turn AllenNLP model output into a Prolog-compatible case.')
parser.add_argument('--predictions', type=str, help='path to file output by the allennlp evaluate command', required=True)
parser.add_argument('--output', type=str, help='path to folder to write to', required=True)
parser.add_argument('--cases', type=str, help='path to folder containing original cases', required=True)
parser.add_argument('--empty', action='store_true', help='whether to write the predicted facts at all. Serves to write "empty" cases')
args = parser.parse_args()

def list2tuple(x):
    # x is a nested list. We need to turn each item into a tuple.
    if not isinstance(x, list):
        return x
    return tuple(list2tuple(y) for y in x)

def tuple2kbe(x):
    # turn a tuple into a Prolog object
    # x is either a single place or 2 place predicate,
    # and each argument is a span
    arity = len(x)-1
    functor = x[0]
    arg1 = x[1]
    arg1 = Term('span', *[Constant(y) for y in arg1])
    if arity == 2:
        arg2 = x[2]
        arg2 = Term('span', *[Constant(y) for y in arg2])
    args = [arg1] if arity == 1 else [arg1, arg2]
    term = str(Term(functor, *args))
    return term

def make_tax_query(query):
    # write out query as findall(X, tax("Alice",2019,X), A), min_list(A,Y), write(Y), nl.
    assert len(query.args) == 2
    _, tax_term = query.args
    assert tax_term.functor == 'tax'
    assert len(tax_term.args) == 3
    tax_args = tax_term.args[:2]
    # run 2 queries: the first one is faster, the second one is slower. That way we can add a
    # timeout, and take the second answer if provided, else the first one.
    new_query = [':- tax({},{},X), !, write(X), nl.'.format(*tax_args),
            ':- findall(X, tax({},{},X), A), min_list(A,Y), write(Y), nl.'.format(*tax_args)]
    new_query = '\n'.join(new_query)
    new_query = [str(x) for x in PrologString(new_query)]
    assert isinstance(new_query, list)
    return new_query

data = {} # map case id -> predictions
# reorganize predictions
for line in open(args.predictions,'r'):
    batch=json.loads(line)
    for kb, spans, logprobs, metadata in zip(batch['kbs'], batch['spans'], batch['probabilities'], batch['metadata']):
        caseid = metadata['id'].split(' - ')[0]
        data.setdefault(caseid, {})
        case_data = data[caseid]
        case_data.setdefault('case text', metadata['case text'])
        assert case_data['case text'] == metadata['case text']
        case_data.setdefault('kb', set())
        case_data['kb'] |= set(tuple2kbe(x) for x in list2tuple(kb))
        case_data.setdefault('spans', set())
        case_data['spans'] |= set(list2tuple(spans))
        case_data.setdefault('logprobs', {})
        for k, v in logprobs.items():
            case_data['logprobs'].setdefault(k, v)
            assert case_data['logprobs'][k] == v
        data[caseid] = copy.deepcopy(case_data)
# end for line in open(args.predictions,'r'):

cases = {}
for filename in glob.glob(os.path.join(args.cases,'*.pl')):
    case = Case.from_file(filename)
    relevant_facts = list()
    # make sure these statements are adapted to Prolog
    # from the text of the case add in 'discontiguous' stuff
    discontiguous = filter(lambda x: 'discontiguous' in x, case.raw_facts.split('\n'))
    discontiguous = map(lambda x: x.strip('.\n'), discontiguous)
    relevant_facts.extend(list(discontiguous))
    # from the case, get:
    for fact in case.facts:
        # - imports eg :- [statutes/...]
        if (fact.functor == ':-') and (len(fact.args) == 2):
            if (term2str(fact.args[0]) == '_directive') and (term2str(fact.args[1]).startswith('[')):
                # then it's a ":- [statutes/prolog/init]" statement
                relevant_facts.append(fact)
        # - s-predicates eg s151_a("Alice", 2015)
        def is_s_predicate(term):
            x=term.functor
            first_char, second_char = term2str(x)[:2]
            return (first_char == 's') and (second_char in [str(ii) for ii in range(10)])
        if is_s_predicate(fact):
            relevant_facts.append(fact)
        elif is_s_predicate(fact.args[0]):
            relevant_facts.append(fact)
    # end for fact in case.facts:
    # - the query
    relevant_facts = list(map(lambda x: str(x), relevant_facts))
    assert case.name not in cases
    # if it's a tax query, modify the query to explicitly compute the amount and write it out
    query = list(case.query)
    is_tax_query=False
    if (len(query) == 1):
        q = query[0]
        if (term2str(q.args[0]) == '_directive') and (q.args[1].functor == 'tax'):
            is_tax_query=True
    if is_tax_query:
        query = make_tax_query(q)
    else:
        query = [str(x) for x in query]
        query[-1] = query[-1] + ', write("success"), nl'
    # end if is_tax_query:
    cases[case.name] = (relevant_facts, [str(x) for x in query])
# end for filename in glob.glob(os.path.join(args.cases,'*.pl')):

# write cases to individual files in folder
if not os.path.isdir(args.output):
    os.mkdir(args.output)
for caseid in data:
    assert caseid in cases, '{} not in {}'.format(caseid, sorted(cases.keys()))
    gold_facts, query = cases[caseid]
    predictions = data[caseid]['kb']
    output_text = ['% ' + data[caseid]['case text']]
    output_text.append('')
    output_text.extend(list(x+'.' for x in gold_facts))
    if not args.empty:
        output_text.extend(list(x+'.' for x in predictions))
    output_text.append('')
    output_text.extend(list(x+'.' for x in query))
    output_text.append(':- halt.')
    output_file = os.path.join(args.output, caseid + '.pl')
    output_text = '\n'.join(output_text)
    # small postprocessing step: negation is best surrounded by spaces
    output_text = output_text.replace("\\+"," \\+ ")
    with open(output_file, 'w') as f:
        assert isinstance(output_text,str)
        f.write(output_text+'\n')
# end for caseid in data:
