# This takes an original case, and expands the knowledge base
# to all possible base facts that can be generated with it. This makes
# the job of the data loader more straightforward.
import sys
import os
import glob
import os
import sys
import json
from problog.logic import Term, Constant, Var, term2str
from problog.program import PrologString, PrologFile
from problog.core import ProbLog
from problog import get_evaluatable
from sara.data.dataset.case import Case
from pyswip import Prolog, Variable
import re
import argparse

parser = argparse.ArgumentParser(description="Expand input case's KB")
parser.add_argument('--statutes', type=str, required=True,
                            help='path to folder containing Prolog program')
parser.add_argument('--case', type=str, required=True,
                            help='path to input case')
parser.add_argument('--output', type=str, required=True,
                            help='path to output folder')
args = parser.parse_args()

input_file = args.case
output_folder = args.output
statutes_folder = args.statutes

# Inventory all predicates
statutes_str = []
for filename in glob.glob(os.path.join(statutes_folder,'*.pl')):
    statutes_str.extend(list(open(filename,'r')))
statutes_str = ''.join(statutes_str)
prolog_program = PrologString(statutes_str)
all_predicates = set()
for filename in glob.glob(os.path.join(statutes_folder,'*.pl')):
    if filename.endswith('init.pl'):
        continue
    if filename.endswith('events.pl'):
        predicates = open(filename,'r')
        predicates = map(lambda x: x.rstrip('.\n').split(' ')[-1].split('/'), predicates)
        predicates = map(lambda x: (x[0],int(x[1])), predicates)
        all_predicates |= set(predicates)
        continue
    prolog_program = PrologFile(filename)
    for thing in prolog_program:
        if thing.functor != ':-':
            continue
        str_args = tuple([term2str(a) for a in thing.args])
        if str_args[0] == "_directive":
            continue
        arg1 = thing.args[0] # this should be a term
        assert isinstance(arg1,Term)
        functor = arg1.functor
        arity = len(arg1.args)
        all_predicates.add((functor, arity))
    # end for thing in prolog_program:
# end for filename in glob.glob(os.path.join(statutes_folder,'*.pl')):

prolog_string = []
# Declare predicates as dynamic
for p, a in all_predicates:
    prolog_string.append(':- discontiguous {}/{}.'.format(p,a))
# Add entire program
for filename in glob.glob(os.path.join(statutes_folder,'*.pl')):
    if filename.endswith('init.pl'):
        continue
    if filename.endswith('events.pl'):
        continue
    prolog_program = PrologFile(filename)
    for thing in prolog_program:
        prolog_string.append(str(thing) + '.')
# Add case
case = Case.from_file(input_file)
for fact in case.facts:
    if fact.functor == ':-':
        str_args = tuple([term2str(a) for a in fact.args])
        if str_args[0] == "_directive":
            continue
    prolog_string.append(str(fact)+'.')
# Consult case
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
tmp_file = os.path.join(output_folder, '{}.tmp'.format(os.path.basename(input_file)))
with open(tmp_file,'w') as f:
    f.write('\n'.join(prolog_string))
prolog_engine = Prolog()
prolog_engine.consult(tmp_file)
# Query for all base facts

# helper functions

def unquote(y):
    done = False
    while not done:
        single_quotes = (y[0]=="'") and (y[-1]=="'")
        double_quotes = (y[0]=='"') and (y[-1]=='"')
        done = (not single_quotes) and (not double_quotes)
        if not done:
            y = y[1:-1]
    # end while not done:
    return y

def remap_arg(x):
    if isinstance(x,str):
        y=unquote(x)
        return '"'+y+'"'
    if not isinstance(x,bytes):
        return x
    y = byte2str(x)
    return '"'+y+'"'
# end def remap_arg(x):

def byte2str(x):
    # turn b'Alice' or b"Nando's Chicken" into
    # Alice and Nando's Chicken respectively
    assert isinstance(x,bytes)
    y = str(x)
    assert y[0] == 'b'
    y = y[1:]
    single_quotes = (y[0]=="'") and (y[-1]=="'")
    double_quotes = (y[0]=='"') and (y[-1]=='"')
    assert single_quotes or double_quotes, y
    y = y[1:-1]
    return y
# end def byte2str(x):

# find all events
events = open(os.path.join(statutes_folder,'events.pl'),'r')
events = map(lambda x: x.split(' ')[2].rstrip('.\n').split('/'), events)
events = map(lambda x: (x[0],int(x[1])), events)
events = filter(lambda x: x[0][-1]=='_', events)
events = list(events)

# query
generated_facts = set()
for event,arity in events:
    assert arity in [1, 2]
    if arity == 1:
        query_str = "{}(span(A,B,C))".format(event)
    elif arity == 2:
        query_str = "{}(span(A,B,C),span(X,Y,Z))".format(event)
    results = prolog_engine.query(query_str)
    results = list(results)
    for r in results:
        args = [r[x] for x in ['A','B','C']]
        all_good = all(not isinstance(x,Variable) for x in args)
        if not all_good:
            continue
        args = map(remap_arg, args)
        args = list(map(lambda x: Constant(x), args))
        arg1 = Term("span", *args)
        arg2 = None
        if arity == 2:
            args = [r[x] for x in ['X','Y','Z']]
            all_good = all(not isinstance(x,Variable) for x in args)
            if not all_good:
                continue
            args = map(remap_arg, args)
            args = list(map(lambda x: Constant(x), args))
            arg2 = Term("span", *args)
        # end if arity == 2:
        kbe = Term(event, arg1) if arg2 is None else Term(event, arg1, arg2)
        generated_facts.add(kbe)
    # end for r in results:
# end for event,arity in events:
n=0
for f in case.facts:
    if f.functor == ':-':
        continue
    if not f.functor.endswith('_'):
        continue
    assert f in generated_facts, str(generated_facts)
    n+=1
assert len(generated_facts) >= n, '{} vs {}'.format(len(generated_facts),n)

# delete tmp file
os.remove(tmp_file)

# Write new case to file
def sorting_key(fact):
    arity = len(fact.args)
    span1 = fact.args[0]
    key = list(span1.args[1:]) # just keep the span
    if arity == 1:
        key.extend([0, 0])
    else:
        span2 = fact.args[1]
        key.extend(span2.args[1:])
    assert len(key) == 4
    key.append(fact.functor)
    key = [x.value if isinstance(x,Constant) else x for x in key]
    assert all(isinstance(x,str) or isinstance(x,int) for x in key), fact
    return tuple(key)
# end def sorting_key(fact):

def find_index(l,k):
    assert k in l
    return l.index(k)
# end def find_index(l,k):
sorted_facts = sorted(generated_facts, key=sorting_key)
sorted_facts = list(map(lambda x: '{}.'.format(x), sorted_facts))
case_text = list(map(lambda x: x.rstrip('\n'), open(input_file,'r')))
facts_index = find_index(case_text, '% Facts')
text_index = find_index(case_text, '% Text')
question_index = find_index(case_text, '% Question')
text = case_text[text_index+1:question_index]
text = [x[2:] for x in text]
text = ' '.join(text).strip(' ')
output_json = {'text': text, 'facts': sorted_facts}
output_file = os.path.join(output_folder,os.path.basename(input_file))
assert output_file.endswith('.pl')
output_file = output_file[:-3] + '.json'
with open(output_file,'w') as f:
    json.dump(output_json,f,indent=2,sort_keys=True)
