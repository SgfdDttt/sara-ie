from typing import *
import logging
import collections
import json
from overrides import overrides
import itertools
import os
from problog.logic import Term, Constant, Var, term2str
from problog.program import PrologString
import datetime
import re
import copy

import torch
from allennlp.data import DatasetReader, Field, Instance
from allennlp.data.fields import MetadataField, ListField, SpanField, TextField, SequenceLabelField, LabelField
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers.token_class import Token
from allennlp.common.util import sanitize
from transformers import BertTokenizer
from allennlp.common.util import sanitize_wordpiece

import pdb

logger = logging.getLogger(__name__)

integer_regexp = re.compile(r"[\d]+")

def remove_enclosing_quotes(string):
    # only remove one level of quotes at a time
    if len(string) == 0:
        return string
    in_double = (string[0] == '"') and (string[-1] == '"')
    in_single = (string[0] == "'") and (string[-1] == "'")
    output = string
    if in_double or in_single:
        output = string[1:-1]
    return output

def unquote(x):
    done=False
    y=str(x)
    while not done:
        done=y==remove_enclosing_quotes(y)
        y=remove_enclosing_quotes(y)
    # end while not done:
    return y

class LegalBertIndexer(PretrainedTransformerIndexer):
    # this is used to wrap LegalBert tokenizer into an indexer
    def __init__(
        self,
        model_name: str,
        namespace: str = "tags",
        max_length: int = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
        ):
        super().__init__('bert-base-cased', **kwargs) # transformer model name is irrelevant because it gets overwritten
        self._namespace = namespace
        self._allennlp_tokenizer = BertTokenizer.from_pretrained(model_name)
        self._tokenizer = self._allennlp_tokenizer
        self._added_to_vocabulary = False

        self._num_added_start_tokens = 1
        self._num_added_end_tokens = 1

        self._max_length = max_length
        if self._max_length is not None:
            num_added_tokens = len(self._allennlp_tokenizer.tokenize("a")) - 1
            self._effective_max_length = (
                self._max_length - num_added_tokens
            )
            if self._effective_max_length <= 0:
                raise ValueError(
                    "max_length needs to be greater than the number of special tokens inserted."
                )


@DatasetReader.register('sara-bio-ie')
class SaraBioIeDataset(DatasetReader):
    def __init__(self,
                 transformer_model_name: str,
                 data_folder: str,
                 max_length: int = None,
                 max_instances: int = None,
                 is_training: bool = True,
                 output_file: str = None,
                 **kwargs):
        super().__init__()
        assert os.path.isdir(data_folder)
        self.data_folder = data_folder
        self.max_length = max_length
        self.max_instances = max_instances
        self.is_training = is_training
        self.output_file = output_file

        if transformer_model_name.endswith('LegalBert'):
            self.tokenizer = BertTokenizer.from_pretrained(transformer_model_name)
        else:
            self.tokenizer = PretrainedTransformerTokenizer(
                transformer_model_name,
                add_special_tokens=False,
                max_length=max_length
            )
        if transformer_model_name.endswith('LegalBert'):
            indexer = LegalBertIndexer(transformer_model_name)
        else:
            indexer = PretrainedTransformerIndexer(transformer_model_name)
        self.token_indexers = {
            "tokens": indexer
        }

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # file_path is a path to a file containing relevant cases
        assert os.path.isfile(file_path)
        instance_counter = 0
        self.kbe_stats = {'accepted': {}, 'rejected': {}}
        for line in open(file_path, 'r'):
            basename = line.strip('\n')
            basename = basename.split('.')[0] + '.json'
            filename = os.path.join(self.data_folder, basename)
            case = json.load(open(filename,'r'))
            case['facts'] = PrologString('\n'.join(case['facts']))
            case['name'] = basename[:-5]
            kbes = set()
            for fact in case['facts']:
                # each fact is a tuple of (predicate, *args)
                # each arg is a triple (surface form, start span, end span)
                def map_arg(arg):
                    assert arg.functor == 'span'
                    assert len(arg.args) == 3
                    assert all(isinstance(a, Constant) for a in arg.args)
                    plain_args = [a.value for a in arg.args]
                    surface_form, start_ind, end_ind = plain_args
                    return (surface_form, start_ind, end_ind)
                # end def map_arg(arg):
                kbes.add((fact.functor,) + tuple(map(map_arg, fact.args)))
            # first round: find events
            if self.is_training:
                events = list(filter(lambda x: len(x) == 2, kbes))
            else:
                events = kbes # if in dev mode, the model needs to generate the full KB
            instance = self.text_to_instance(case, events)
            if instance is None:
                # don't generate the arguments if the event is not decodable
                self.kbe_stats['rejected'].setdefault(case['name'], 0)
                self.kbe_stats['rejected'][case['name']] += len(kbes)
                continue
            else:
                y = instance.human_readable_dict()
                caseid = case['name']
                self.kbe_stats['accepted'].setdefault(caseid, 0)
                self.kbe_stats['accepted'][caseid] += len(y['metadata']['accepted kbes'])
                self.kbe_stats['rejected'].setdefault(caseid, 0)
                self.kbe_stats['rejected'][caseid] += len(y['metadata']['rejected kbes'])
                yield instance
            instance_counter += 1
            if not self.is_training: # if in dev mode, the model needs to generate the full KB
                continue
            # second round: find arguments of each event
            for event in sorted(events):
                e = event[1] # event marker
                arguments = list(filter(lambda x: (len(x) == 3) and (x[1] == e), kbes))
                instance = self.text_to_instance(case, set(arguments), root_span=e)
                if instance is None:
                    self.kbe_stats['rejected'].setdefault(case['name'], 0)
                    self.kbe_stats['rejected'][case['name']] += len(set(arguments))
                else:
                    y = instance.human_readable_dict()
                    self.kbe_stats['accepted'].setdefault(case['name'], 0)
                    self.kbe_stats['accepted'][case['name']] += len(y['metadata']['accepted kbes'])
                    self.kbe_stats['rejected'].setdefault(case['name'], 0)
                    self.kbe_stats['rejected'][case['name']] += len(y['metadata']['rejected kbes'])
                    yield instance
                instance_counter += 1
            if (self.max_instances is not None) and (instance_counter >= self.max_instances):
                break

    @overrides
    def text_to_instance(
            self,
            case,
            raw_kbes: set,
            root_span = None
            ):
        case_text = case['text']
        instance_id = '{} - {}'.format(case['name'], root_span)
        metadata = {
            'id': instance_id,
            'case text': case_text,
            'root span': root_span,
            'rejected kbes': set(),
            'accepted kbes': set(),
            'full kb': set()
        }
        # filter out kbes that can't be generated by the model
        # - underlying representations that can't be generated from the span
        # - overlapping spans
        # - edges with multiple labels
        # That 3rd condition is implicitly checked by the 2nd condition. Still
        # keeping the dedicated function "conflicting_labels" around.
        metadata['full kb'] = copy.deepcopy(set(raw_kbes))
        kbes = set()
        def is_decodable(predicate,argument):
            gold_argument, start_char, end_char = argument
            decoded_argument=decode_span(case_text,predicate,(start_char,end_char))
            return gold_argument==decoded_argument
        if root_span is not None:
            if not is_decodable("dummy_",root_span):
                return None
        # remove anything that is not decodable
        for kbe in raw_kbes:
            predicate = kbe[0]
            arity = len(kbe[1:])
            assert arity in [1,2]
            if arity == 1:
                # root event is decodable
                decodable = is_decodable(predicate,kbe[1])
            elif arity == 2:
                # root event is decodable, and arg is decodable
                decodable = (is_decodable("dummy_",kbe[1])) and \
                        (is_decodable(predicate,kbe[2]))
            if decodable:
                kbes.add(kbe)
            else:
                metadata['rejected kbes'].add(kbe)
        # end for kbe in raw_kbes:
        # look for overlapping spans in arguments to be predicted,
        # and for multiple labels for the same edge,
        # and keep the first in lexicographic order
        nu_kbes = set()
        def is_overlapping_span(span1,span2):
            assert len(span1) == 3
            assert len(span2) == 3
            _, start1, end1 = span1
            _, start2, end2 = span2
            assert all(isinstance(x,int) for x in [start1, end1, start2, end2]), \
                    '{} and {}'.format(span1,span2)
            not_overlapping = (end1<=start2) or (end2<=start1)
            return not not_overlapping
        # end def is_overlapping_span(span1,span2):
        def is_overlapping_arg(kbe1, kbe2):
            # kbe2 overlaps with kbe1 if
            # 1. they are both events and their spans overlap
            # 2. they are both arguments, and they are the same event, and their arguments overlap
            if len(kbe1) != len(kbe2):
                return False
            elif len(kbe1) == 2: # case 1
                return is_overlapping_span(kbe1[1], kbe2[1])
            else: # case 2
                assert len(kbe1) == 3
                same_event = kbe1[1] == kbe2[1]
                overlapping_arg = is_overlapping_span(kbe1[2], kbe2[2])
                return same_event and overlapping_arg
        # end def is_overlapping_arg(kbe1, kbe2):
        def conflicting_labels(kbe1,kbe2):
            if len(kbe1) != len(kbe2):
                return False
            pred1 = kbe1[0]
            pred2 = kbe2[0]
            indices1 = [x[1:] for x in kbe1[1:]]
            indices2 = [x[1:] for x in kbe2[1:]]
            conflict = (indices1 == indices2) and (pred1 != pred2)
            return conflict
        # end def conflicting_labels(kbe1,kbe2):
        def conflict_with_existing(kbe,kb):
            for okbe in kb:
                if is_overlapping_arg(kbe,okbe):
                    return True
                if conflicting_labels(kbe,okbe):
                    return True
            # end for okbe in kb:
            return False
        # end def conflict_with_existing(kbe,kb):
        def sorting_key(kbe):
            key = []
            assert len(kbe) in [2,3]
            if len(kbe) == 3:
                items = kbe[1:]
            else:
                items = [("", -1, -1), kbe[1]] # this is like prepending a virtual root. It ensures events are kept before arguments.
            for x in items:
                # sorting by value at last ensures that 'start_' predicates are before 'end_'
                key.extend(list(x[1:])+[x[0]])
            key.append(kbe[0]) # finally, add name of predicate, so that if patient_(spanA,spanB) and purpose_(spanA,spanB) both occur we can deterministically choose
            assert len(key) == 2*3+1 # 2 spans and 1 predicate
            return tuple(key)
        # end def sorting_key(kbe):
        for kbe in sorted(kbes,key=sorting_key):
            if not conflict_with_existing(kbe, nu_kbes):
                metadata['accepted kbes'].add(kbe)
                nu_kbes.add(kbe)
            else:
                metadata['rejected kbes'].add(kbe)
        # end for kbe in kbes:
        # sanity checks
        assert len(metadata['rejected kbes'] & metadata['accepted kbes']) == 0
        assert metadata['rejected kbes'] | metadata['accepted kbes'] == metadata['full kb'], pdb.set_trace()
        assert metadata['accepted kbes'] == nu_kbes
        kbes = copy.deepcopy(nu_kbes)
        assert len(kbes)<=len(raw_kbes)
        metadata['kbes'] = kbes
        # tokenize case
        tokens = self.tokenizer.tokenize(case_text)
        if isinstance(self.tokenizer, BertTokenizer):
            # Andrew's LegalBERT
            tokens = self.get_legal_bert_tokens(case_text)
        else:
            tokens = self.tokenizer.add_special_tokens(tokens)
        token_indices = self.estimate_character_indices(case_text, tokens)
        # reflect token indices in the token objects
        assert len(token_indices) == len(tokens)
        for token, tok_ind in zip(tokens, token_indices):
            tok_start, tok_end = None, None
            if tok_ind is not None:
                tok_start, tok_end = tok_ind
            token.idx = tok_start
            token.idx_end = tok_end
        assert len(tokens) <= self.max_length
        metadata['tokens'] = tokens
        # convert arguments to spans in tokens
        arg2span = {}
        def find_token(char_index):
            # for a given char_index, return the token it is contained in
            output = None
            for ii, tok_ind in enumerate(token_indices):
                if tok_ind is None:
                    assert (ii==0) or (ii==len(token_indices)-1)
                    continue
                elif (char_index>=tok_ind[0]) and (char_index<tok_ind[1]):
                    output = ii
                    break
            # end for ii, tok_ind in enumerate(token_indices):
            assert output is not None, pdb.set_trace()
            return output
        # end def find_token(char_index):
        def get_token_spans(arg):
            assert len(arg) == 3, arg
            char_start, char_stop = arg[1:]
            assert isinstance(char_start, int), char_start
            assert isinstance(char_stop, int), char_stop
            # convert char spans to token spans
            token_start = find_token(char_start)
            assert token_start is not None, pdb.set_trace()
            token_stop = find_token(char_stop) # char spans are inclusive
            assert token_stop is not None, pdb.set_trace()
            token_stop += 1 # token spans are exclusive
            return token_start, token_stop
        # end def get_token_spans(arg):
        if root_span is not None:
            token_span = get_token_spans(root_span)
            arg2span[root_span] = token_span
        for kbe in kbes:
            if root_span is not None:
                assert kbe[1] == root_span
            for arg in kbe[1:]:
                token_span = get_token_spans(arg)
                arg2span.setdefault(arg, token_span)
                assert arg2span[arg] == token_span # check for different entries under same key
            # end for arg in kbe[1:]:
        # end for kbe in kbes"""
        input_text = TextField(tokens)
        metadata_field = MetadataField(metadata)
        # establish sequence of tags
        tags = ['O' for _ in tokens]
        classification_samples = set() # triples of (start_span, stop_span, label)
        for kbe in kbes:
            predicate = kbe[0]
            if self.is_training:
                expected_arity = 2 if root_span is None else 3
                assert len(kbe) == expected_arity
            # the stuff that needs to be tagged and classified is the last argument
            arg = kbe[-1]
            # each argument or event has exactly 1 relevant span
            # set tags
            relevant_span = arg2span[arg]
            for ii in range(*relevant_span):
                if self.is_training:
                    # at dev time, there may be overlapping spans, because they are decoded
                    # separately (eg same span argument to 2 events)
                    # The BIO tags are not used at dev time anyway
                    assert tags[ii] == 'O', tags # check we're not overwriting something
                tags[ii] = 'I'
            tags[relevant_span[0]] = 'B'
            # set classification
            classification_samples.add(relevant_span+(predicate,))
            # end for span in arg2spans[arg]:
        # end for kbe in kbes:
        classification_samples = sorted(classification_samples)
        metadata['spans'] = set((start, stop) for start, stop, _ in classification_samples)
        tags = [LabelField(t, "tag_labels") for t in tags]
        tags_field = ListField(tags) #, input_text)
        input_spans = [
                SpanField(span_start=start, span_end=stop, sequence_field=input_text)
                for start,stop,_ in classification_samples ]
        if len(input_spans)>0:
            input_span_field = ListField(input_spans)
        else:
            dummy_list = ListField([SpanField(span_start=0, span_end=0, sequence_field=input_text)])
            input_span_field = dummy_list.empty_field()
        label_space = "arg_labels" # we're merging all arg and event labels
        span_labels = [LabelField(label, label_space) for _,_,label in classification_samples]
        assert len(input_spans) == len(span_labels)
        if len(span_labels)>0:
            span_label_field = ListField(span_labels)
        else:
            dummy_list = ListField([LabelField("x", label_space)])
            span_label_field = dummy_list.empty_field()
        # if no root is specified, take [CLS] as the root
        if root_span is None:
            root_span = (-1,-1)
        elif root_span not in arg2span:
            """this happens when root_span doesn't occur in the text. This training example
            might confuse the model, so we're skipping it entirely. The model wasn't trained
            to detect the root event in the first place anyway, so this would have no value
            at eval time"""
            return None
        else:
            root_span = arg2span[root_span]
        assert isinstance(root_span, tuple)
        assert len(root_span) == 2
        root_span_field = SpanField(span_start=root_span[0],
                span_end=root_span[1], sequence_field=input_text)
        fields: Dict[str, Field] = {
            'metadata': metadata_field,
            'text': input_text,
            'root_span': root_span_field
        }
        if self.is_training:
            fields.update({
                'tags': tags_field,
                'spans': input_span_field,
                'labels': span_label_field
                })
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["text"].token_indexers = self.token_indexers

    def get_legal_bert_tokens(self, case_text):
        # Andrew's LegalBERT is not in AllenNLP format so we do something custom
        # This is partially copied from https://github.com/allenai/allennlp/blob/main/allennlp/data/tokenizers/pretrained_transformer_tokenizer.py#L235
        encoded_tokens = self.tokenizer.encode_plus(case_text, text_pair=None,
                add_special_tokens=True, truncation=True, max_length=self.max_length)
        token_ids = encoded_tokens["input_ids"]
        token_type_ids = encoded_tokens["token_type_ids"]

        tokens = []
        for token_id, token_type_id in zip(token_ids, token_type_ids):
            # In `special_tokens_mask`, 1s indicate special tokens and 0s indicate regular tokens.
            # NOTE: in transformers v3.4.0 (and probably older versions) the docstring
            # for `encode_plus` was incorrect as it had the 0s and 1s reversed.
            # https://github.com/huggingface/transformers/pull/7949 fixed this.
            start = None
            end = None

            tokens.append(
                Token(
                    text=self.tokenizer.convert_ids_to_tokens(token_id, skip_special_tokens=False),
                    text_id=token_id,
                    type_id=token_type_id,
                    idx=start,
                    idx_end=end,
                )
            )

        return tokens


    def estimate_character_indices_legal_bert(self, text, token_ids):
        # taken straight from https://github.com/allenai/allennlp/blob/main/allennlp/data/tokenizers/pretrained_transformer_tokenizer.py#L295
        token_texts = [
            sanitize_wordpiece(t) for t in self.tokenizer.convert_ids_to_tokens(token_ids)
        ]
        token_offsets: List[Optional[Tuple[int, int]]] = [None] * len(token_ids)

        min_allowed_skipped_whitespace = 3
        allowed_skipped_whitespace = min_allowed_skipped_whitespace

        text_index = 0
        token_index = 0
        while text_index < len(text) and token_index < len(token_ids):
            token_text = token_texts[token_index]
            token_start_index = text.find(token_text, text_index)

            # Did we not find it at all?
            if token_start_index < 0:
                token_index += 1
                # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
                allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                continue

            # Did we jump too far?
            non_whitespace_chars_skipped = sum(
                1 for c in text[text_index:token_start_index] if not c.isspace()
            )
            if non_whitespace_chars_skipped > allowed_skipped_whitespace:
                # Too many skipped characters. Something is wrong. Ignore this token.
                token_index += 1
                # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
                allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                continue
            allowed_skipped_whitespace = min_allowed_skipped_whitespace

            token_offsets[token_index] = (
                token_start_index,
                token_start_index + len(token_text),
            )
            text_index = token_start_index + len(token_text)
            token_index += 1
        return token_offsets


    def estimate_character_indices(self, case_text, tokens):
        # the token vocabulary was produced with WordPiece. So if something is an [UNK], it must be a single character.
        # So we just need to find spaces, and the rest is fine.
        if isinstance(self.tokenizer, BertTokenizer):
            token_indices = self.estimate_character_indices_legal_bert(case_text, [x.text_id for x in tokens])
        else:
            token_indices = self.tokenizer._estimate_character_indices(case_text, [x.text_id for x in tokens])
        assert token_indices is not None
        for ii, tok_ind in enumerate(token_indices):
            if tok_ind is not None:
                continue
            if (ii==0) or (ii==len(token_indices)-1): # those tend to be separator tokens so no need to change them
                continue
            prev_token = token_indices[ii-1]
            assert prev_token is not None
            next_token_index = ii+1
            while token_indices[next_token_index] is None:
                next_token_index += 1
                if next_token_index == len(token_indices):
                    break
            if next_token_index == len(token_indices):
                case_span = (prev_token[1], len(case_text))
            else:
                case_span = (prev_token[1], token_indices[next_token_index][0])
            span_text = case_text[case_span[0]:case_span[1]]
            index = self.force_align(span_text)
            token_indices[ii] = (case_span[0]+index, case_span[0]+index+1)
            assert token_indices[ii][0]>=case_span[0]
            assert token_indices[ii][1]<=case_span[1]
        # end for ii, tok_ind in enumerate(token_indices):
        # sanity checks
        assert all(x is not None for x in token_indices[1:-1])
        for ii, tok_ind in enumerate(token_indices):
            if ii in [0, len(token_indices)-2, len(token_indices)-1]:
                continue
            assert tok_ind[1]<=token_indices[ii+1][0]
        return token_indices

    def force_align(self, case_text):
        # just pick the first non-space token
        assert len(case_text)>0
        index=0
        while case_text[index] == ' ':
            index+=1
            if index==len(case_text):
                break
        index=min(index,len(case_text)-1)
        assert case_text[index] != ' '
        return index


""" DECODING FUNCTIONS AND CONSTANTS """
def decode_span(case_text, predicate, span):
    """ This returns a single output that is the decode of
    span in case_text, to be used in mutiple places in the code """
    """ case_text: string representation of the case
    predicate: string representation of the predicate whose last argument
    is span
    span: a pair of inclusive indices into case_text
    output: a well formed string or integer such that
    span(output,span[0],span[1]) is a valid last argument to predicate """
    string_form = case_text[span[0]:span[1]+1]
    output = None
    # try to recast based on name of predicate
    if predicate in ['amount_']:
        possible_matches = integer_regexp.findall(string_form)
        refined_string = None
        for pm in possible_matches:
            if refined_string is None:
                refined_string = pm
            elif len(refined_string)<len(pm):
                refined_string=pm
        if refined_string is None:
            output = 1 # default integer value
        else:
            output = int(refined_string)
    elif predicate in ['start_', 'end_']:
        output = convert_to_date((span[0], span[1]), case_text, predicate)
    elif predicate == 'purpose_':
        # if the second argument to 'purpose_' is something related to a plan,
        # make sure to convert it to the proper string. The mapping was established
        # based on observations in the data, and is gold.
        output = {
                "retirement fund": "make provisions for employees in case of retirement",
                "life insurance": "make provisions for employees in case of death",
                "health insurance": "make provisions for employees in case of sickness",
                "disability plan": "make provisions for employees or dependents"
                }.get(string_form, string_form)
        output = '"{}"'.format(unquote(output))
    else:
        output = '"{}"'.format(unquote(string_form))
    assert output is not None
    return output

DATE_FORMATS=[ # ordered by priority
        '%b %d, %Y', # Jan 1, 2021
        '%B %d, %Y', # January 1, 2021
        '%b %dst, %Y', # Feb 1st, 2017
        '%B %dst, %Y', # October 1st, 2013
        '%b %dnd, %Y', # Mar 2nd, 2015
        '%B %dnd, %Y', # March 2nd, 2015
        '%b %drd, %Y', # Feb 3rd, 1992
        '%B %drd, %Y', # May 29th, 2008
        '%b %dth, %Y',
        '%B %dth, %Y', # July 9th, 2014
        '%d %B %Y', # 2 February 2015
        '%d %b %Y', # 2 Feb 2015

        '%b %d', # Jan 1
        '%B %d', # January 1
        '%b %dst', # Feb 1st
        '%B %dst', # October 1st
        '%b %dnd', # Mar 2nd
        '%B %dnd', # March 2nd
        '%b %drd', # Feb 3rd
        '%B %drd', # May 29th
        '%b %dth',
        '%B %dth', # July 9th

        '%B', # September
        '%b', # Sep

        '%Y' # 2017
        ]

def convert_to_date(char_span, case_text, predicate):
    # turn a string argument into an integer that
    # represents a date as YYYYMMDD
    if (char_span[0] is None) or (char_span[1] is None):
        arg_string = ""
    else:
        arg_string = case_text[char_span[0]:char_span[1]+1].strip()
        # strip punctuation - T5 sometimes keeps punctuation behind a number
        arg_string = arg_string.strip('.,')
    assert predicate in ['start_', 'end_']
    # 1. try to parse into the most accurate format possible
    date_formatted = None
    for form in DATE_FORMATS:
        try:
            date_formatted = datetime.datetime.strptime(arg_string, form)
        except ValueError:
            pass
        if date_formatted is not None:
            break
    # end for form in DATE_FORMATS:
    if date_formatted is not None:
        year = date_formatted.year if '%Y' in form else None
        month = date_formatted.month if '%b' in form.lower() else None
        day = date_formatted.day if '%d' in form else None
    else:
        year, month, day = None, None, None
    # end for form in DATE_FORMATS:
    # 2. try to find missing values in text
    if year is None:
        # match sequences of 4 digits starting with 1 or 2
        # that are not preceded by a dollar sign
        year_regexp = re.compile(r"[^\$]([12]\d{3})")
        years = year_regexp.finditer(case_text)
        def span2position(span):
            if span is None:
                return 0.5
            elif len(span) != 2:
                return 0.5
            elif not all(isinstance(x, int) for x in span):
                return 0.5
            else:
                return sum(span)/len(span)
        # end def span2position(span):
        def algebraic_distance(span1, span2):
            # positive if span1 is before span2
            p1 = span2position(span1)
            p2 = span2position(span2)
            return p2-p1
        # end def algebraic_distance(span1, span2):
        years = [(algebraic_distance(char_span, m.span()), m.group().strip()) for m in years]
        # pick closest year that occurs after the span in question, and if there is no
        # such year, pick the one that is closest
        years = [(x[0] if x[0]>=0 else float('inf'), abs(x[0]), x[1]) for x in years]
        year = min(years)
        year = year if year is None else int(year[2])
    # 3. complete with default values
    year = 0 if year is None else year
    if month is None:
        month = 1 if predicate == 'start_' else 12
    if day is None:
        day = 1 if predicate == 'start_' else 31
    for x in [day, month, year]:
        assert x is not None
        assert isinstance(x, int)
    output = day + 100*month + 10000*year
    return output
# end def convert_to_date(char_span, case_text, predicate):
""" END DECODING FUNCTIONS AND CONSTANTS """

""" STATISTICS UTILS """
def avg(l):
    return sum(l) / len(l)

def stddev(l):
    m = avg(l)
    l2 = [(x - m) ** 2 for x in l]
    sig = avg(l2)
    sig = sig ** 0.5
    return sig

def median(l):
    l2 = sorted(l)
    L = len(l2)
    if L % 2 == 1:
        return l2[L // 2]
    else:
        return 0.5 * (l2[L // 2 - 1] + l2[L // 2])

def histogram(l):
    output = {}
    for x in l:
        output.setdefault(x, 0)
        output[x] += 1
    return output

def compute_stats(l):
    return {'min': min(l), 'max': max(l), 'avg': avg(l),
            'stddev': stddev(l), 'median': median(l),
            'hist': histogram(l), 'N': len(l)}
""" END STATISTICS UTILS """

if __name__ == '__main__':
    from allennlp.data import Vocabulary
    print("by default, read train, dev and test, and print statistics")

    data_path = "resources/sara_v3"
    dataset = SaraBioIeDataset(
        transformer_model_name="t5-base",
        max_length=512,
        data_folder=os.path.join(data_path, "grounded_cases")
    )
    output_folder = os.path.join(data_path, "ie_compatible_cases")
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    # statistics to be collected
    # because instances are sub-case, I need to first create a mapping of case -> counts for some
    cases = {} # case id -> KBEs
    predictions = { # individual predictions, in the style produced by allennlp evaluate
            'kbs': [],
            'spans': [],
            'probabilities': [],
            'metadata': []
            }
    stats = {}
    for split in ['train','dev','test']:
        print('============ ' + split.upper() + ' ============')
        data = {
                'kbes': {
                    'number': {}, # number of kbes for each case
                    'events': {}, # number of events for each case
                    'arguments': {} # number of arguments for each case
                    },
                'predicates': {
                    'events': {}, # mapping of event types to their number of occurrence eg 'patient_' -> 59
                    'arguments': {} # same as above but for arguments
                    },
                'spans': {
                    'number': {}, # number of spans for each case
                    'length': {
                        'words': [], # length of spans in words
                        'characters': [] # length of spans in characters
                        },
                    'total length': {
                        'words': {}, # total number of words in a case that belong to a span
                        'characters': {} # total number of characters in a case that belong to a span
                        },
                    'types': set() # all the different spans that come up
                    },
                'text': {
                    'length': {
                        'words': {}, # number of words in case text
                        'characters': {} # number of characters in case text
                        }
                    }
                }
        for x in dataset.read('resources/data/{}'.format(split)):
            y = x.human_readable_dict()
            caseid = y['metadata']['id'].split(' - ')[0]
            cases.setdefault(caseid,set())
            kbes = y['metadata']['kbes']
            cases[caseid].update(kbes)
            predictions['kbs'].append(kbes)
            predictions['spans'].append(y['metadata']['spans'])
            logprobs = {}
            for x in kbes:
                logprobs[str(x)] = 0
            predictions['probabilities'].append(copy.deepcopy(logprobs))
            predictions['metadata'].append(y['metadata'])
            events = list(filter(lambda x: len(x)==2, kbes))
            arguments = list(filter(lambda x: len(x)==3, kbes))
            assert len(events) + len(arguments) == len(kbes)
            labels = list(filter(lambda x: x != -1, y['labels'])) # -1 is padding
            assert len(labels) == len(kbes), pdb.set_trace()
            spans = list(filter(lambda x: (x[0]>=0) and (x[1]>=0), y['spans'])) # (-1,-1) is padding
            assert len(spans) == len(kbes)
            text = y['metadata']['case text']
            for k, v in [('number', kbes), ('events', events), ('arguments', arguments)]:
                data['kbes'][k].setdefault(caseid, 0)
                data['kbes'][k][caseid] += len(v)
            # end for k, v in [('number', kbes), ('events', events), ('arguments', arguments)]:
            for k, l in [('events', events), ('arguments', arguments)]:
                for x in l:
                    p = x[0]
                    data['predicates'][k].setdefault(p, 0)
                    data['predicates'][k][p] += 1
                # end for x in l:
            # end for k, l in [('events', events), ('arguments', arguments)]:
            data['spans']['number'].setdefault(caseid, 0)
            data['spans']['number'][caseid] += 1
            for kbe in kbes:
                arg = kbe[-1] # it's always the last arg that's relevant
                _, start_char, end_char = arg
                len_chars = end_char - start_char + 1 # inclusive spans
                data['spans']['length']['characters'].append(len_chars)
                data['spans']['total length']['characters'].setdefault(caseid, 0)
                data['spans']['total length']['characters'][caseid] += len_chars
                span_text = text[start_char:end_char+1]
                len_words = len(span_text.strip('\n ').split())
                data['spans']['length']['words'].append(len_words)
                data['spans']['total length']['words'].setdefault(caseid, 0)
                data['spans']['total length']['words'][caseid] += len_words
                data['spans']['types'].add(arg)
            # end for kbe in kbes:
            for k in ['characters', 'words']:
                n = len(text) if k=='characters' else len(text.split())
                data['text']['length'][k].setdefault(caseid, n)
                assert data['text']['length'][k][caseid] == n
        # end for x in dataset.read('resources/data/{}'.format(split)):
        # in the end, they must represent the same cases
        caseids = set(data['kbes']['number'].keys())
        assert set(data['kbes']['events'].keys()) == caseids
        assert set(data['kbes']['arguments'].keys()) == caseids
        assert set(data['spans']['number'].keys()) == caseids
        assert set(data['text']['length']['words'].keys()) == caseids
        assert set(data['text']['length']['characters'].keys()) == caseids
        # must have counted the same number of samples
        assert len(data['spans']['length']['words']) == len(data['spans']['length']['characters'])
        n_spans = sum(v for _, v in set(data['spans']['number'].items()))
        n_spans = len(data['spans']['length']['words'])
        # total number of types must be equal to number of spans
        n_events = sum(v for _,v in data['predicates']['events'].items())
        n_args = sum(v for _,v in data['predicates']['arguments'].items())
        n_spans == n_events + n_args
        # compute statistics
        # 1. statistics in terms of min/max/avg/stdev
        def get_stats(d):
            samples = list(v for _, v in d.items())
            return compute_stats(samples)
        # end def get_stats(d):
        def print_stats(s):
            assert isinstance(s,dict)
            print(json.dumps(s,indent=2,sort_keys=True))
        print('kbes'.upper())
        print('    number'.upper())
        data['kbes']['number'] = get_stats(data['kbes']['number'])
        print_stats(data['kbes']['number'])
        print('    events'.upper())
        data['kbes']['events'] = get_stats(data['kbes']['events'])
        print_stats(data['kbes']['events'])
        print('    arguments'.upper())
        data['kbes']['arguments'] = get_stats(data['kbes']['arguments'])
        print_stats(data['kbes']['arguments'])
        print('spans'.upper())
        print('    number'.upper())
        data['spans']['number'] = get_stats(data['spans']['number'])
        print_stats(data['spans']['number'])
        print('    length - characters'.upper())
        data['spans']['length']['characters'] = compute_stats(data['spans']['length']['characters'])
        print_stats(data['spans']['length']['characters'])
        print('    length - words'.upper())
        data['spans']['length']['words'] = compute_stats(data['spans']['length']['words'])
        print_stats(data['spans']['length']['words'])
        for k in ['characters', 'words']:
            print('    coverage - {}'.format(k).upper())
            samples = []
            for caseid in data['spans']['total length'][k]:
                len_spans = data['spans']['total length'][k][caseid]
                len_text = data['text']['length'][k][caseid]
                samples.append(len_spans/len_text)
            # end for caseid in data['spans']['total length']['characters']:
            print_stats(compute_stats(samples))
        # end for k in ['characters', 'words']:
        print('text'.upper())
        print('    length - characters'.upper())
        data['text']['length']['characters'] = get_stats(data['text']['length']['characters'])
        print_stats(data['text']['length']['characters'])
        print('    length - words'.upper())
        data['text']['length']['words'] = get_stats(data['text']['length']['words'])
        print_stats(data['text']['length']['words'])
        # 2. histograms/distributions
        print('predicates'.upper())
        print('    events'.upper())
        print_stats(data['predicates']['events'])
        print('    arguments'.upper())
        print_stats(data['predicates']['arguments'])
        print('spans - number of types'.upper())
        print(len(data['spans']['types']))
        print('rejected and accepted kbes'.upper())
        for key, d in dataset.kbe_stats.items():
            print(key.upper())
            s = sorted(d.items(), key=lambda x: (-x[1], x[0]))
            top_10p = int(round(len(s)/10))
            print(s[:top_10p])
            top_10p_number = sum(x[1] for x in s[:top_10p])
            total_number = sum(x[1] for x in s)
            print('{} vs {}'.format(top_10p_number, total_number))
            #print(sorted(d.items(), key=lambda x: (-x[1], x[0])))
        del data['spans']['types']
        stats[split] = copy.deepcopy(data)
    # end for split in ['train','dev','test']:
    with open("data_statistics.json","w") as f:
        json.dump(stats,f,indent=2,sort_keys=True)
    # write IE-compatible cases to file, in the style of model predictions, to compute a Prolog "topline"
    def make_span(arg):
        assert len(arg) == 3
        value, start_char, end_char = arg
        assert isinstance(value, int) or isinstance(value, str)
        assert isinstance(start_char,int)
        assert isinstance(end_char,int)
        if isinstance(value,str):
            value = '"' + unquote(value) + '"'
        output = 'span({},{},{})'.format(value, start_char, end_char)
        return output
    # end def make_span(arg):
    def make_prolog_compatible_kb(kbes):
        output = set()
        for kbe in kbes:
            assert len(kbe) in [2,3]
            predicate = unquote(kbe[0])
            arguments = kbe[1:]
            prolog_args = [make_span(a) for a in arguments]
            prolog_str = '{}({}).'.format(predicate, ','.join(prolog_args))
            output.add(prolog_str)
        # end for kbe in kbes:
        return output
    # end def make_prolog_compatible_kb(kbes):
    for caseid, case_kb in cases.items():
        prolog_kb = make_prolog_compatible_kb(case_kb)
        output_file = os.path.join(output_folder, caseid + '.pl')
        with open(output_file, 'w') as f:
            f.write('\n'.join(prolog_kb)+'\n')
    # end for caseid, case_kb in cases.items():
    # also print as a prediction in the style of the output of the "allennlp evaluate" command
    ref_k = sorted(predictions.keys())[0]
    ref_len = len(predictions[ref_k])
    for k, v in predictions.items():
        assert ref_len == len(v)
    output_file = os.path.join(output_folder, 'predictions.json')
    with open(output_file, 'w') as f:
        json.dump(sanitize(predictions),f)
