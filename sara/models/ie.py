import copy
import re
import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy, F1Measure, SpanBasedF1Measure
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules import FeedForward
from allennlp.nn import Activation
from allennlp.nn.initializers import NormalInitializer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers.token_class import Token
import torch
import torch.nn.functional as F
import allennlp.modules.conditional_random_field as crf
from transformers import BertModel

from sara.metrics import SetBasedF1Measure
import sara.data.dataset.ie as Ie

# the following two sets are used in decoding
EVENT_PREDICATES = set([
    "american_employer_",
    "attending_classes_",
    "birth_",
    "blindness_",
    "brother_",
    "business_",
    "business_trust_",
    "citizenship_",
    "daughter_",
    "death_",
    "deduction_",
    "disability_",
    "educational_institution_",
    "enrollment_",
    "father_",
    "hospital_",
    "incarceration_",
    "income_",
    "international_organization_",
    "itemize_deductions_",
    "joint_return_",
    "legal_separation_",
    "marriage_",
    "medical_institution_",
    "medical_patient_",
    "migration_",
    "mother_",
    "nonresident_alien_",
    "nurses_training_school_",
    "payment_",
    "penal_institution_",
    "plan_",
    "residence_",
    "retirement_",
    "service_",
    "sibling_",
    "sister_",
    "son_",
    "termination_",
    "unemployment_compensation_agreement_"
    ])
ARGUMENT_PREDICATES = set([
    "agent_",
    "amount_",
    "beneficiary_",
    "country_",
    "destination_",
    "end_",
    "location_",
    "means_",
    "patient_",
    "purpose_",
    "reason_",
    "start_",
    "type_",
    ])

def tensor2list(t):
    return t.detach().cpu().numpy().tolist()

def find_close_span(logprobs, span):
    # This is called if span is not in logprobs
    assert span not in logprobs
    # I've seen this happen where ('""', 117, 117) was in
    # logprobs but ('"\'"', 117, 117) wasn't
    # Find a matching logprob based on indices only
    for key, value in logprobs.items():
        if key[1:] == span[1:]:
            return value
    return None

def compute_logprobs(predictions, logprobs, bb, is_first_round):
    predicted_spans = predictions['predicted spans'][bb]
    span_probabilities = predictions['span probabilities'][bb]
    predicted_kbes = predictions['kbs'][bb]
    label_probabilities = predictions['label probabilities'][bb]
    ref_length = len(predicted_kbes)
    assert len(predicted_spans) == ref_length
    assert len(span_probabilities) == ref_length
    assert len(label_probabilities) == ref_length
    for kbe, span, span_lp, label_lp in zip(predicted_kbes, predicted_spans, span_probabilities, label_probabilities):
        if is_first_round:
            # first round, so these are events
            _, kbe_span = kbe
            logprobs[kbe_span] = span_lp # store it for use in second round
            # lp(P,A) = lp(P|A) + lp(A)
            kbe_lp = label_lp + logprobs[kbe_span]
            logprobs.setdefault(kbe, kbe_lp)
            if logprobs[kbe] != kbe_lp:
                print('WARNING: ({},{}) and {}'.format(kbe, kbe_lp, logprobs))
        else:
            # second round, so these are arguments
            _, kbe_span1, _ = kbe
            if kbe_span1 in logprobs:
                span1_lp = logprobs[kbe_span1]
            else:
                span1_lp = find_close_span(logprobs, kbe_span1)
            if span1_lp is None:
                span1_lp = 0 # this will inflate probabilities but is better than setting it to -inf
            # lp(P,A1,A2) = lp(P|A1,A2) + lp(A2|A1) + lp(A1)
            kbe_lp = label_lp + span_lp + span1_lp
            logprobs.setdefault(kbe, kbe_lp)
            if logprobs[kbe] != kbe_lp:
                print('WARNING: ({},{}) and {}'.format(kbe, kbe_lp, logprobs))
    # end for kbe, span, label_lp in zip(predicted_kbes, predicted_spans, label_probabilities):
    return logprobs
# end def compute_logprobs(predictions, logprobs, bb, is_first_round):

@Model.register('sara-bio-ie')
class SaraBioIeModel(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            transformer_model_name: str = "bert-base-cased",
            num_layers_tagger: int = 0,
            num_units_tagger: int = 1024,
            num_layers_classifier: int = 0,
            num_units_classifier: int = 1024,
            feature_dropout = 0.2,
            loss_tradeoff: float = 0.5,
            distance_feature_size = 512,
            max_length = 512,
            stddev = 0.02,
            span_representation_mode: str = "average",
            ):
        super().__init__(vocab)
        assert loss_tradeoff <= 1
        assert loss_tradeoff >= 0
        self.loss_tradeoff = loss_tradeoff
        assert span_representation_mode in ["extremities", "average"]
        self.span_representation_mode = span_representation_mode
        self.distance_feature_size = distance_feature_size
        self.max_length = max_length
        self.non_allennlp = transformer_model_name.endswith('LegalBert')
        if self.non_allennlp:
            embedder = BertModel.from_pretrained(transformer_model_name)
            self.text_field_embedder = embedder
        else:
            embedder = PretrainedTransformerEmbedder(transformer_model_name)
            self.text_field_embedder = BasicTextFieldEmbedder(
                {
                    "tokens": embedder
                }
            )
        if ('t0' in transformer_model_name.lower()) or ('t5' in transformer_model_name.lower()):
            # only keep the transformer encoder
            self.text_field_embedder.token_embedder_tokens.transformer_model \
                    = self.text_field_embedder.token_embedder_tokens.transformer_model.encoder
        if self.non_allennlp:
            base_feature_size = embedder.config.hidden_size
        else:
            base_feature_size = self.text_field_embedder.get_output_dim()
        if self.distance_feature_size is not None:
            base_feature_size += self.distance_feature_size
        t = torch.zeros((base_feature_size,)).cuda()
        self.virtual_root_representation = torch.nn.Parameter(t)
        initializer = NormalInitializer(mean=0, std=stddev)
        initializer(self.virtual_root_representation)
        # representation of one token, and of root span representation
        if self.span_representation_mode == 'extremities':
            input_size = 3*base_feature_size
        elif self.span_representation_mode == 'average':
            input_size = 2*base_feature_size
        self.num_tags = vocab.get_vocab_size("tag_labels")
        self.num_arg_labels = vocab.get_vocab_size("arg_labels")
        self.num_event_labels = vocab.get_vocab_size("event_labels")
        output_size = self.num_tags
        if num_layers_tagger > 0:
            feature_ffnn = FeedForward(input_dim=input_size,
                                       hidden_dims=num_units_tagger,
                                       num_layers=num_layers_tagger,
                                       activations=Activation.by_name('tanh')(),
                                       dropout=feature_dropout)
            output_affine = torch.nn.Linear(feature_ffnn.get_output_dim(), output_size)
            self.tagger = torch.nn.Sequential(feature_ffnn, output_affine)
        else:
            self.tagger = torch.nn.Linear(input_size, output_size)
        # representation of one span, and of root span
        if self.span_representation_mode == 'extremities':
            input_size = 4*base_feature_size
        elif self.span_representation_mode == 'average':
            input_size = 2*base_feature_size
        output_size = self.num_arg_labels + self.num_event_labels
        if num_layers_classifier > 0:
            feature_ffnn = FeedForward(input_dim=input_size,
                                       hidden_dims=num_units_classifier,
                                       num_layers=num_layers_classifier,
                                       activations=Activation.by_name('tanh')(),
                                       dropout=feature_dropout)
            output_affine = torch.nn.Linear(feature_ffnn.get_output_dim(), output_size)
            self.classifier = torch.nn.Sequential(feature_ffnn, output_affine)
        else:
            self.classifier = torch.nn.Linear(input_size, output_size)
        bio_transitions = crf.allowed_transitions(constraint_type='BIO', \
                            labels=vocab.get_index_to_token_vocabulary("tag_labels"))
        self.crf = crf.ConditionalRandomField(num_tags=self.num_tags,constraints=bio_transitions)
        if self.distance_feature_size is not None:
            self.distance_embedding = torch.nn.Embedding(
                    embedding_dim=self.distance_feature_size, num_embeddings=self.max_length, padding_idx=-1)
        self.metrics = {
                'loss': Average(),
                'label accuracy': CategoricalAccuracy(),
                'tag accuracy': CategoricalAccuracy(),
                'span': SetBasedF1Measure(),
                'kbe': SetBasedF1Measure(),
                'full kb': SetBasedF1Measure(),
                'label': SetBasedF1Measure() # the name is slightly misleading
                }
        # the 'label' metric measures on a per-sample basis whether the correct labels have been predicted,
        # irrespective of whether the underlying span is correct

    def compute_distance_features(self,
                                  embedded_question: torch.Tensor,
                                  trigger_span: torch.Tensor) -> torch.Tensor:
        max_length = embedded_question.size(1)
        # shape (1, sequence_length)
        basic_range = torch.arange(max_length, device=embedded_question.device)
        basic_range = torch.unsqueeze(basic_range, dim=0)
        # shape (batch_size, 1)
        start_indices = torch.unsqueeze(trigger_span[:, 0], dim=1)
        end_indices = torch.unsqueeze(trigger_span[:, 1], dim=1)
        # shape (batch_size, sequence_length)
        distance_feature = F.relu(start_indices - basic_range) \
                + F.relu(basic_range - end_indices)
        # spans set to (-1, -1) should be ignored
        mask = torch.logical_or(start_indices == -1, end_indices == -1)
        distance_feature = torch.masked_fill(distance_feature, mask, max_length - 1)
        # find closest span
        assert torch.all(distance_feature >= 0)
        assert torch.all(distance_feature < max_length)
        distance_feature = self.distance_embedding(distance_feature)
        # If for a given example, the trigger span is (-1, -1), then switch the distance
        # feature off. This can happen if the template anchor is not within the window.
        mask = torch.unsqueeze(mask, dim=2)
        distance_feature = torch.masked_fill(distance_feature, mask, 0)
        return distance_feature

    def compute_root_span_rep(
            self,
            embedded_text: torch.Tensor,
            root_span: torch.LongTensor
            ):
        # extract root span representation
        batch_size, _ = root_span.size()
        _, _, dim = embedded_text.size()
        root_span_reps = self.compute_span_representations(embedded_text,
                torch.unsqueeze(root_span,dim=1)) # batch x dim
        root_span_reps = torch.squeeze(root_span_reps, dim=1)
        # wherever necessary, plug in the virtual root representation
        virtual_root_mask = root_span[:,0] == -1
        assert (virtual_root_mask == (root_span[:,1] == -1)).all()
        virtual_root_mask = torch.unsqueeze(virtual_root_mask, dim=1).float()
        root_span_reps = (1-virtual_root_mask)*root_span_reps + virtual_root_mask*torch.unsqueeze(self.virtual_root_representation,dim=0)
        assert not torch.any(torch.isnan(root_span_reps))
        if self.span_representation_mode == "extremities":
            assert root_span_reps.size() == (batch_size, 2*dim)
        elif self.span_representation_mode == "average":
            assert root_span_reps.size() == (batch_size, dim)
        return root_span_reps

    def compute_bio_logits(
            self,
            embedded_text,
            root_span,
            root_span_rep
            ):
        batch_size, sequence_length, _ = embedded_text.size()
        _, dim = root_span_rep.size()
        assert root_span_rep.size(0) == batch_size
        expanded_root_span_rep = torch.unsqueeze(root_span_rep,dim=1)
        expanded_root_span_rep = expanded_root_span_rep.expand(
                (batch_size, sequence_length, dim)) # batch x seq len x dim
        assert expanded_root_span_rep.size(0) == embedded_text.size(0)
        assert expanded_root_span_rep.size(1) == embedded_text.size(1)
        augmented_input_features = torch.cat([embedded_text, expanded_root_span_rep], dim=2)
        assert not torch.any(torch.isnan(augmented_input_features))
        # predict tags
        bio_logits = self.tagger(augmented_input_features)
        return bio_logits

    def predict_labels(
            self,
            embedded_text,
            spans,
            root_span_rep
            ):
        span_representations = self.compute_span_representations(embedded_text, spans) # batch x num spans x dim
        assert not torch.any(torch.isnan(span_representations))
        num_spans = span_representations.size(1)
        batch_size, num_spans, _ = span_representations.size()
        _, dim = root_span_rep.size()
        expanded_root_span_rep = torch.unsqueeze(root_span_rep, dim=1)
        expanded_root_span_rep = expanded_root_span_rep.expand(
                (batch_size, num_spans, dim)) # batch x num_spans x dim
        assert expanded_root_span_rep.size(0) == span_representations.size(0)
        assert expanded_root_span_rep.size(1) == span_representations.size(1)
        classifier_features = torch.cat([span_representations, expanded_root_span_rep], dim=2)
        assert not torch.any(torch.isnan(classifier_features))
        label_logits = self.classifier(classifier_features)
        return label_logits

    def make_virtual_root(
            self,
            batch_size,
            device
            ):
        # this returns a span at (0,1)
        output = torch.zeros((batch_size, 2), dtype=torch.long, device=device)
        output[:,0] = -1
        output[:,1] = -1
        return output

    def make_span_tensor(
            self,
            span,
            device
            ):
        # batch size is 1, num spans is 1
        output = torch.zeros((1, 2), dtype=torch.long, device=device)
        output[:,0] = span[0]
        output[:,1] = span[1]
        return output

    def decode_span(self, span, metadata):
        output = metadata['tokens'][span[0]:span[1]]
        assert output
        start_char = list(filter(lambda x: x is not None, [tok.idx for tok in output]))
        start_char = min(start_char) if start_char else None
        stop_char = list(filter(lambda x: x is not None, [tok.idx_end for tok in output]))
        stop_char = max(stop_char) if stop_char else None
        if (start_char is None) or (stop_char is None):
            # this can happen if for example the model selects "[SEP]"
            output = ""
            start_char = -1
            stop_char = 0
        else:
            output = metadata['case text'][start_char:stop_char]
        return output, (start_char, stop_char-1) # make char spans inclusive

    def decode_kbe(self, span, label_predictions, root_span, metadata):
        assert span is not None
        predicate_strings = [
                (self.vocab.get_token_from_index(cc, "arg_labels").strip('"'), x)
                for cc, x in enumerate(label_predictions)
                ]
        # sorted in increasing order of likelihood
        predicate_strings = sorted(predicate_strings, key=lambda x: (x[1], x[0]))
        # 1. determine the predicate
        relevant_predicates = EVENT_PREDICATES if root_span is None else ARGUMENT_PREDICATES
        relevant_predicate_logprobs = []
        for p_str, lp in predicate_strings:
            if p_str in relevant_predicates:
                relevant_predicate_logprobs.append(lp)
        # some of the relevant predicates simply aren't part of the vocab of the model because they don't appear in train
        assert len(relevant_predicate_logprobs) <= len(relevant_predicates)
        predicate = None
        while predicate not in relevant_predicates:
            predicate, logprob = predicate_strings.pop()
        assert predicate is not None
        logprob = logprob - tensor2list(torch.logsumexp(torch.tensor(relevant_predicate_logprobs),dim=0))
        assert logprob <= 0
        _, (start_char, stop_char) = self.decode_span(span, metadata)
        arg2 = Ie.decode_span(metadata['case text'], predicate, (start_char, stop_char))
        if root_span is None: # then it's an event
            kbe = (predicate, (arg2,start_char,stop_char))
        else:
            arg1 = '"{}"'.format(root_span[0].strip('"'))
            kbe = (predicate, (arg1,)+root_span[1:], (arg2,start_char,stop_char))
        return kbe, logprob

    def compute_text_features(self, text, root_span):
        # run encoder
        if self.non_allennlp:
            # remap keys
            tokens=text['tokens']
            bert_outputs = self.text_field_embedder(tokens['token_ids'],
                    token_type_ids=tokens['type_ids'], attention_mask=tokens['mask'],)
            embedded_text = bert_outputs.last_hidden_state
        else:
            embedded_text = self.text_field_embedder(text)
        batch_size = embedded_text.size(0)
        device = embedded_text.device
        assert not torch.any(torch.isnan(embedded_text))
        # Computes distance features if they are on
        if self.distance_feature_size is not None:
            # Shape: (batch_size, seq_len, distance_feature_size)
            # distance features are switched off in case of a virtual root span
            distance_feature = self.compute_distance_features(
                    embedded_text, root_span)
            token_features = torch.cat([embedded_text, distance_feature], dim=2)
        else:
            token_features = embedded_text
        return token_features


    def teacher_forcing(
            self,
            text: Dict[str, Dict[str, torch.LongTensor]],
            root_span: torch.LongTensor, # shape is batch x 2
            tags: torch.LongTensor, # shape is batch size x seq len
            spans: torch.LongTensor, # shape is batch x num spans x 2
            labels: torch.LongTensor, # shape is batch size x num spans
            metadata: List[Dict[str, Any]]
            ):
        output = {}
        device = root_span.device
        embedded_text = self.compute_text_features(text, root_span)
        root_span_rep = self.compute_root_span_rep(embedded_text, root_span)
        bio_logits = self.compute_bio_logits(embedded_text, root_span, root_span_rep)
        # predict spans
        predicted_spans, _ = self.decode_spans(bio_logits, text['tokens']['mask'], compute_probabilities=False)
        output['predicted spans'] = predicted_spans
        self.metrics['tag accuracy'](bio_logits, tags, mask=text['tokens']['mask'])
        gold_spans = tensor2list(spans)
        gold_spans = list(map(lambda x: set(map(lambda y: tuple(y), x)), gold_spans))
        gold_spans = list(set(filter(lambda x: (x[0]!=-1) and (x[1]!=-1), y)) for y in gold_spans)
        self.metrics['span'](list(map(lambda x: set(x), predicted_spans)), gold_spans)
        # predict labels
        label_logits = self.predict_labels(embedded_text, spans, root_span_rep)
        label_predictions = torch.argmax(label_logits, dim=2) # batch size x num spans
        output['label predictions'] = label_predictions
        labels_for_metrics = zip(tensor2list(label_predictions), tensor2list(labels))
        labels_for_metrics = list(zip(a,b) for a,b in labels_for_metrics)
        labels_for_metrics = list(list(filter(lambda x: x[1]>-1, l))
                for l in labels_for_metrics)
        predicted_labels_for_metrics = list(list(x[0] for x in y)
                for y in labels_for_metrics)
        gold_labels_for_metrics = list(list(x[1] for x in y)
                for y in labels_for_metrics)
        self.metrics['label'](predicted_labels_for_metrics, gold_labels_for_metrics)
        assert (text['tokens']['mask'] == (tags!=-1)).all()
        tag_probability = self.crf(bio_logits,
                tags.masked_fill(tags==-1, 0), text['tokens']['mask'])
        num_tokens = torch.sum(text['tokens']['mask'])
        assert num_tokens > 0
        tag_loss = -tag_probability / max(1,num_tokens) # normalize by number of labels
        assert not torch.isnan(tag_loss)
        self.metrics['label accuracy'](label_logits, labels, mask=labels != -1)
        num_labels = torch.sum(labels != -1)
        if num_labels > 0:
            label_loss = F.cross_entropy(torch.reshape(label_logits,
                (-1,self.num_arg_labels+self.num_event_labels)),
                torch.reshape(labels, (-1,)), ignore_index=-1, reduction='sum')
            assert not torch.isnan(label_loss)
            label_loss /= max(1, num_labels) # normalize by number of samples
            loss = self.loss_tradeoff*tag_loss + (1-self.loss_tradeoff)*label_loss
        else:
            loss = self.loss_tradeoff*tag_loss
        assert not torch.isnan(loss)
        self.metrics['loss'](loss)
        output['loss'] = loss
        return output


    def inference(
            self,
            text: Dict[str, Dict[str, torch.LongTensor]],
            metadata: List[Dict[str, Any]] = None
            ):
        mask_len = tensor2list(torch.sum(text['tokens']['mask'], dim=1))
        token_len = [len(x['tokens']) for x in metadata]
        assert len(mask_len) == len(token_len)
        assert all(x==y for x,y in zip(token_len, mask_len))
        text_mask = text['tokens']['mask']
        text_mask = text_mask.contiguous()
        batch_size = text_mask.size(0)
        logprobs = [{} for _ in range(batch_size)] # keep track of log-probs for kbes
        device = text_mask.device
        # first round, decode with virtual root
        root_span = self.make_virtual_root(batch_size, device)
        embedded_text = self.compute_text_features(text, root_span)
        assert not torch.any(torch.isnan(embedded_text))
        first_round = self.one_step_inference(embedded_text, text_mask,
                root_span, None, metadata)
        # get log-probs of first-round spans and kbes
        for bb in range(batch_size):
            logprobs[bb] = compute_logprobs(first_round, logprobs[bb], bb, True)
        # end for bb in range(batch_size):
        # second round, decode each item separately
        kbs = [set(x) for x in first_round['kbs']]
        spans = [set(x) for x in first_round['predicted spans']]
        mask_len = tensor2list(torch.sum(text_mask, dim=1))
        token_len = [len(x['tokens']) for x in metadata]
        assert len(mask_len) == len(token_len)
        assert all(x==y for x,y in zip(token_len, mask_len))
        for bb in range(batch_size):
            queue = first_round['predicted spans'][bb]
            while queue:
                current = queue.pop()
                root_span_tensor = self.make_span_tensor(current, embedded_text.device)
                root_span_text, index_tuple = self.decode_span(current, metadata[bb]) 
                result = self.one_step_inference(embedded_text[bb:bb+1,:,:],
                        text_mask[bb:bb+1,:],
                        root_span_tensor, (root_span_text,)+index_tuple, metadata[bb:bb+1])
                # save log-probs of dependent spans
                assert len(result['kbs']) == 1
                assert len(result['predicted spans']) == 1
                logprobs[bb] = compute_logprobs(result, logprobs[bb], 0, False)
                kbs[bb] |= set(result['kbs'][0])
                spans[bb] |= set(result['predicted spans'][0])
            # end while queue:
        # end for bb in range(batch_size):
        # final sanity check: check that each predicted KBE has a probability
        for bb in range(batch_size):
            for kbe in kbs[bb]:
                assert kbe in logprobs[bb]
        # end for bb in range(batch_size):
        return {'kbs': kbs, 'spans': spans, 'probabilities': logprobs}

    def one_step_inference(
            self,
            embedded_text,
            text_mask,
            root_span: torch.LongTensor,
            root_span_tuple: tuple,
            metadata: List[Dict[str, Any]] = None
            ):
        if root_span_tuple is not None:
            assert len(root_span_tuple) == 3, root_span_tuple
        output = {}
        device = root_span.device
        batch_size = embedded_text.size(0)
        # check that all things have the correct dimensions
        assert text_mask.size(0) == batch_size
        assert len(metadata) == batch_size
        assert embedded_text.size(1) == text_mask.size(1)
        # check validity of mask
        mask_len = tensor2list(torch.sum(text_mask, dim=1))
        token_len = [len(x['tokens']) for x in metadata]
        assert len(mask_len) == len(token_len)
        assert all(x==y for x,y in zip(token_len, mask_len))
        # run encoder
        root_span_rep = self.compute_root_span_rep(embedded_text, root_span)
        bio_logits = self.compute_bio_logits(embedded_text, root_span, root_span_rep)
        # turn predicted spans into a tensor
        predicted_spans, log_probabilities = self.decode_spans(bio_logits, text_mask, compute_probabilities=True)
        assert len(predicted_spans) == batch_size
        output["predicted spans"] = predicted_spans
        output["span probabilities"] = log_probabilities
        num_spans = max(1,max(len(x) for x in predicted_spans))
        spans = torch.zeros((batch_size,num_spans,2), dtype=torch.long, device=device)
        spans.fill_(-1) # mask value
        for ii, _spans in enumerate(predicted_spans):
            for jj, s in enumerate(_spans):
                spans[ii,jj,0] = s[0]
                spans[ii,jj,1] = s[1]
        # end for ii, _spans in enumerate(predicted_spans):
        label_logits = self.predict_labels(embedded_text, spans, root_span_rep)
        label_predictions = torch.argmax(label_logits, dim=2) # batch size x num spans
        output['label predictions'] = label_predictions
        kbes = [ list() for _ in range(batch_size) ]
        label_logprobs = [ list() for _ in range(batch_size) ]
        label_predictions = tensor2list(label_logits)
        assert len(label_predictions) == len(predicted_spans)
        for ii, decoded_spans in enumerate(predicted_spans):
            assert len(label_predictions[ii]) >= len(decoded_spans)
            for jj, span in enumerate(decoded_spans):
                # sanity check
                seq_len = len(metadata[ii]['tokens'])
                assert span[0] in range(seq_len)
                assert span[1] in range(seq_len+1)
                assert span[1] > span[0]
                kbe, label_logprob = self.decode_kbe(span, label_predictions[ii][jj],
                        root_span_tuple, metadata[ii])
                kbes[ii].append(kbe)
                label_logprobs[ii].append(label_logprob)
            # end for jj, span in enumerate(decoded_spans):
        # end for ii, decoded_spans in enumerate(predicted_spans):
        output['kbs'] = kbes
        output['label probabilities'] = label_logprobs
        return output

    def forward(
            self,
            text: Dict[str, Dict[str, torch.LongTensor]],
            root_span: torch.LongTensor, # shape is batch x 2
            tags: torch.LongTensor = None, # shape is batch size x seq len
            spans: torch.LongTensor = None, # shape is batch x num spans x 2
            labels: torch.LongTensor = None, # shape is batch size x num spans
            metadata: List[Dict[str, Any]] = None
            ):
        mask_len = tensor2list(torch.sum(text['tokens']['mask'], dim=1))
        token_len = [len(x['tokens']) for x in metadata]
        assert len(mask_len) == len(token_len)
        assert all(x==y for x,y in zip(token_len, mask_len))
        if self.training:
            output = self.teacher_forcing(text, root_span, tags, spans, labels, metadata)
        else:
            output = self.inference(text, metadata)
            self.metrics['kbe'](output['kbs'], [m['kbes'] for m in metadata])
            self.metrics['span'](output['spans'], [m['spans'] for m in metadata])
            predicted_labels = [ set(x[0] for x in kbe) for kbe in output['kbs'] ]
            expected_labels = [ set(x[0] for x in m['kbes']) for m in metadata ]
            self.metrics['label'](predicted_labels, expected_labels)
            self.metrics['full kb'](output['kbs'], [m['full kb'] for m in metadata])
        # end if self.training:
        # tuples as keys are problematic, so we need to change them to something else
        if 'probabilities' in output: # typically this will only be the case at eval time
            for ii,_ in enumerate(output['probabilities']):
                probabilities = {}
                for k, v in output['probabilities'][ii].items():
                    assert isinstance(k, tuple)
                    key = str(k)
                    probabilities[key] = v
                output['probabilities'][ii] = copy.deepcopy(probabilities)
            # end for ii,_ in enumerate(output['probabilities']):
        # save other relevant information for downstream use
        output['metadata'] = copy.deepcopy(metadata)
        return output

    def compute_span_representations(self, input_features, span_indices):
        if self.span_representation_mode =="extremities":
            return self.compute_span_representations_select_extremities(
                    input_features, span_indices)
        elif self.span_representation_mode =="average":
            return self.compute_span_representations_average_pool(
                    input_features, span_indices)

    def compute_span_representations_select_extremities(self, input_features, span_indices):
        # input_features is batch size x seq len x dim
        # span indices is batch size x num spans x 2
        # output is batch size x num spans x 2*dim
        assert len(input_features.size()) == 3
        assert len(span_indices.size()) == 3
        device = input_features.device
        batch_size, sequence_length, dim = input_features.size()
        rectified_indices = span_indices.clone()
        # offset by one because span indices are exclusive
        rectified_indices[:,:,1] = rectified_indices[:,:,1]-1
        # avoid selecting indices outside the allowed range
        rectified_indices = rectified_indices.masked_fill(span_indices<0, 0)
        rectified_indices = rectified_indices.masked_fill(rectified_indices>sequence_length-1, sequence_length-1)
        selected_features = input_features[
            torch.arange(batch_size, device=device).unsqueeze(-1).unsqueeze(-1),
            rectified_indices
        ]
        output_features = torch.cat([selected_features[:,:,0,:], selected_features[:,:,1,:]], dim=2)
        assert output_features.size(0) == batch_size
        assert output_features.size(1) == span_indices.size(1)
        assert output_features.size(2) == 2*dim
        return output_features

    def compute_span_representations_average_pool(self, input_features, span_indices):
        # input_features is batch size x seq len x dim
        # span indices is batch size x num spans x 2
        # output is batch size x num spans x dim
        assert len(input_features.size()) == 3
        assert len(span_indices.size()) == 3
        device = input_features.device
        batch_size, sequence_length, dim = input_features.size()
        # create tensor of indices
        num_spans = span_indices.size(1)
        indices = torch.arange(0, sequence_length, device=device) # indices are exclusive
        indices = torch.tile(indices, (batch_size, num_spans, 1)) # size batch x num root spans x seq len
        assert span_indices[:,:,0].size() == indices[:,:,0].size()
        # create mask over relevant indices
        # batch size x num spans x seq len
        span_mask = torch.logical_and(indices >= span_indices[:,:,0:1], indices < span_indices[:,:,1:2]) 
        # compute masked mean
        assert span_mask.size() == (batch_size, num_spans, sequence_length)
        # batch x num spans x seq len x dim
        span_mask = torch.unsqueeze(span_mask, dim=3)
        masked_features = torch.unsqueeze(input_features, dim=1) * span_mask
        span_reps = torch.sum(masked_features, dim=2) # batch x num spans x dim
        numerator = torch.sum(span_mask, dim=2)
        numerator = numerator.masked_fill(numerator == 0, 1)
        assert torch.all(numerator > 0)
        span_reps = span_reps / numerator
        assert span_reps.size() == (batch_size, num_spans, dim)
        assert not torch.any(torch.isnan(span_reps))
        assert span_reps.size(0) == batch_size
        assert span_reps.size(1) == span_indices.size(1)
        assert span_reps.size(2) == dim
        return span_reps

    def compute_span_probability(self, logits: torch.Tensor, mask: torch.BoolTensor, span):
        # Some of this code was copy-pasted from AllenNLP source code:
        # https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py#L217
        # This uses the CRF to compute the probability of a specific span in a sequence
        restricted_logits = torch.clone(logits)
        logits_len, num_tags = logits.size()
        seq_len = tensor2list(sum(mask))
        assert span[0]<seq_len
        assert span[1]<=seq_len
        tag_map = [self.vocab.get_token_from_index(x, "tag_labels") for x in range(3)]
        # Make a mask of all allowed emissions
        emission_mask = torch.ones((logits_len,num_tags), dtype=torch.bool, device=logits.device)
        # beginning of span must be a 'B'
        emission_mask[span[0],tag_map.index('I')]=False
        emission_mask[span[0],tag_map.index('O')]=False
        # 2nd tag of span until the end must be 'I' (spans are exclusive)
        for ii in range(span[0]+1, span[1]):
            emission_mask[ii,tag_map.index('B')]=False
            emission_mask[ii,tag_map.index('O')]=False
        # Following tag must be either 'B' or 'O'
        if span[1] < seq_len:
            emission_mask[span[1],tag_map.index('I')]=False
        # set forbidden emissions to a large negative emission score
        # using -float('inf') cause nan. -10000.0 is what AllenNLP uses to enforce decoding constraints.
        large_negative_number = -10000.0
        restricted_logits = restricted_logits.masked_fill(torch.logical_not(emission_mask), large_negative_number)
        numerator = self.crf._input_likelihood(
                torch.unsqueeze(restricted_logits,dim=0), torch.unsqueeze(mask,dim=0))
        denominator = self.crf._input_likelihood(
                torch.unsqueeze(logits,dim=0), torch.unsqueeze(mask,dim=0))
        assert (numerator<=denominator).all(), '{} vs {}'.format(numerator,denominator)
        log_prob = tensor2list(numerator-denominator) # we don't need the gradient
        assert len(log_prob) == 1
        log_prob = log_prob.pop()
        assert isinstance(log_prob,float)
        assert log_prob<=0
        return log_prob


    def decode_spans(self, bio_logits, mask, compute_probabilities=False):
        batch_size = bio_logits.size(0)
        viterbi_tags = self.crf.viterbi_tags(bio_logits, mask=mask, top_k=1)
        assert batch_size == len(viterbi_tags)
        output = [ list() for _ in viterbi_tags ]
        for ii, top_k in enumerate(viterbi_tags):
            # we only care about the top decode
            tag_seq, _ = top_k[0]
            # find spans
            current_span = None
            for tt, tag_int in enumerate(tag_seq):
                tag_letter = self.vocab.get_token_from_index(tag_int, "tag_labels")
                if current_span is not None: # we're inside a span
                    if tag_letter in ['B', 'O']: # then close the span
                        output[ii].append((current_span, tt)) # exclusive spans
                        current_span = None
                else: # we're not inside a span
                    if tag_letter == 'B':
                        current_span = tt
                # end if current_span is not None: # we're inside a span
            if current_span is not None: # close final span if still open
                output[ii].append((current_span, tt+1))
            # end for tt, tag_int in enumerate(tag_seq):
        # end for ii, top_k in enumerate(viterbi_tags):
        # sanity check
        for ii, spans in enumerate(output):
            assert sorted(set(spans)) == spans # no duplicates, and ordered naturally
            for start, stop in spans:
                seq_len = tensor2list(sum(mask[ii,:]))
                assert start in range(seq_len)
                assert stop in range(seq_len+1) # exclusive spans
                assert stop > start # exclusive spans
                tag_seq = viterbi_tags[ii][0][0]
                tag_start = tag_seq[start]
                assert self.vocab.get_token_from_index(tag_start, "tag_labels") == 'B'
                tag_stop = tag_seq[stop-1] # exclusive indices
                if stop>start+1: # otherwise when stop == start+1, the stop tag is a B
                    assert self.vocab.get_token_from_index(tag_stop, "tag_labels") in ['I', 'O']
                if tag_stop-1 > tag_start+1:
                    for tag in tag_seq[tag_start+1:tag_stop-1]:
                        assert self.vocab.get_token_from_index(tag, "tag_labels") == 'I'
            # end for start, stop in spans:
        # end for ii, spans in enumerate(output):
        # only decode probabilities in eval mode
        log_probabilities = None
        if compute_probabilities:
            log_probabilities = [ [] for _ in output ]
            for ii,outp in enumerate(output):
                for span in outp:
                    lp = self.compute_span_probability(bio_logits[ii,:,:], mask[ii,:], span)
                    log_probabilities[ii].append(lp)
            assert all(len(x) == len(y) for x,y in zip(output,log_probabilities))
        return output, log_probabilities

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {}
        for key, value in self.metrics.items():
            val = value.get_metric(reset)
            if isinstance(val, dict):
                for key2, value2 in val.items():
                    output['{}-{}'.format(key, key2)] = value2
            else:
                output[key] = val
        for k, v in output.items():
            if 'loss' in k:
                continue
            if v is None:
                continue
            output[k] = round(100*v, 1)
        return output
