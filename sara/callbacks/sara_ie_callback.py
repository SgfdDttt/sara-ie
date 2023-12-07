import logging
import os
import json
from typing import Optional, List, Dict, Any, Tuple, Set
from overrides import overrides
import pdb

import torch
from allennlp.training import TrainerCallback

logger = logging.getLogger(__name__)

def format_kbe(kbe):
    if isinstance(kbe,str):
        return kbe
    assert isinstance(kbe,tuple)
    predicate = kbe[0]
    arguments = kbe[1:]
    argument_string = ', '.join([str(x) for x in arguments])
    output = '{}({})'.format(predicate, argument_string)
    return output

@TrainerCallback.register('batch_results_printer')
class BatchResultsPrinter(TrainerCallback):
    def __init__(self,
                 serialization_dir: str,
                 using_logger: bool = True,
                 output_file_path: Optional[str] = None,
                 validation_only: bool = True):
        super().__init__(serialization_dir)
        self._using_logger = using_logger
        self._validation_only = validation_only
        # self._output_file_path = output_file_path
        if output_file_path is None:
            self._output_file_path = None
        else:
            self._output_file_path = os.path.join(serialization_dir, output_file_path)
            os.makedirs(self._output_file_path, exist_ok=True)

        self._file_handler = None
        self._current_epoch = -1
        self._current_stage = True  # is_training?

        self.last_entry_id = None

    def _get_file_handler(self, epoch: int, is_training: bool):
        if self._output_file_path is None:
            return None
        if epoch == self._current_epoch and self._current_stage == is_training:
            return self._file_handler
        file_path = os.path.join(self._output_file_path,
                                 f'model-outputs-{epoch}-' + ('train' if is_training else 'validation') + '.txt')
        if self._file_handler is None:
            self._file_handler = open(file_path, 'w')
        elif epoch != self._current_epoch or self._current_stage != is_training:
            self._current_epoch = epoch
            self._current_stage = is_training
            self._file_handler = open(file_path, 'w')
        return self._file_handler

    def _log(self, log_str):
        if self._using_logger:
            logger.info(log_str)
        if self._file_handler is not None:
            self._file_handler.write(f'{log_str}\n')

    @overrides
    def on_batch(
            self,
            trainer,
            batch_inputs: List[List[torch.Tensor]] = None,
            batch_outputs: List[Dict[str, Any]] = None,
            metrics: Dict[str, Any] = None,
            epoch: int = None,
            batches_in_epoch_completed=None,
            is_training: bool = None,
            is_primary: bool = None,
            batch_grad_norm=None,
            **kwargs
    ) -> None:
        if self._validation_only and is_training:
            return

        file_handler = self._get_file_handler(epoch=epoch, is_training=is_training)

        assert len(batch_outputs) == len(batch_inputs), \
            '{} vs {}'.format(len(batch_outputs), len(batch_inputs))
        for inputs, outputs in zip(batch_inputs, batch_outputs):
            for ii, metadata in enumerate(inputs['metadata']):
                case_id = metadata['id']
                case_text = metadata['case text']
                case_tokens = metadata['tokens']
                root_span = metadata['root span']
                gold_kbes = '. '.join(format_kbe(kbe) for kbe in sorted(metadata['kbes']))
                predicted_kbes = '. '.join(format_kbe(kbe) for kbe in sorted(outputs['kbs'][ii]))
                gold_spans = ', '.join(str(x) for x in sorted(metadata['spans']))
                predicted_spans = ', '.join(str(x) for x in sorted(outputs['spans'][ii]))
                log_probs = dict((format_kbe(k),v) for k,v in outputs['probabilities'][ii].items())
                log_probs = json.dumps(log_probs, indent=2, sort_keys=True)
                self._log('============')
                self._log(f'Case id={case_id}')
                self._log(f'Case={case_text}')
                self._log(f'Tokens={case_tokens}')
                self._log(f'Root span={root_span}')
                self._log(f'Gold KB={gold_kbes}')
                self._log(f'Predicted KB={predicted_kbes}')
                self._log(f'Gold spans={gold_spans}')
                self._log(f'Predicted spans={predicted_spans}')
                self._log(f'Log probabilities={log_probs}')
                self._log('------------')
            # end for ii, metadata in enumerate(inputs['metadata']):
            if file_handler is not None:
                file_handler.flush()
        # end for inputs, outputs in zip(batch_inputs, batch_outputs):


def len_vs(a, b):
    return '{} vs {}'.format(len(a), len(b))
