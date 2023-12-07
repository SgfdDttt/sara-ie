"""
Written based on AllenNLP example from Optuna Github repo
"""

import os
import random
import shutil
import sys
import copy
import logging
import json
import glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy
import optuna
from optuna.integration import AllenNLPPruningCallback
from packaging import version
import torch

import allennlp
import allennlp.data
import allennlp.models
import allennlp.modules
import sara
import sara.data
import sara.data.dataset
import sara.data.dataset.ie
import sara.models
import sara.models.ie
import sara.callbacks
import sara.callbacks.sara_ie_callback
from allennlp.common.params import Params

DATA_DIR = "resources/"
TARGET_METRICS = ("kbe-f1", "label-f1", "span-f1") # in order of importance
TARGET_METRIC = "kbe-f1"

assert os.path.isdir(DATA_DIR)

def median(l):
    if len(l)==0:
        return 0
    assert len(l)>0
    l2 = sorted(l)
    L = len(l2)
    m = L%2
    k = L//2
    if m == 1:
        return l2[k]
    else:
        return 0.5*(l2[k-1]+l2[k])

def mean(l):
    if len(l) == 0:
        return 0
    assert len(l)>0
    return float(sum(l)) / len(l)

def get_config(config, trial):
    local_config = copy.deepcopy(config)
    local_config['model'].update({
        'num_layers_classifier': trial.suggest_int('num_layers_classifier', 1, 4),
        'num_units_classifier': 2**trial.suggest_int('log_num_units_classifier', 5, 11),
        'num_layers_tagger': trial.suggest_int('num_layers_tagger', 1, 4),
        'num_units_tagger': 2**trial.suggest_int('log_num_units_tagger', 5, 11),
        'distance_feature_size': 2**trial.suggest_int('log_distance_feature_size', 3, 9),
        'loss_tradeoff': trial.suggest_float('loss_tradeoff', 0, 1),
        'feature_dropout': trial.suggest_float('dropout', 0, 0.8)
        })
    batch_size=2**trial.suggest_int('log_batch_size', 3, 8)
    max_batch_size = 32
    if batch_size <= max_batch_size:
        real_batch_size, num_accumulation_steps = batch_size, 1
    else:
        real_batch_size, num_accumulation_steps = max_batch_size, batch_size//max_batch_size
    assert real_batch_size*num_accumulation_steps == batch_size
    local_config['data_loader']['batch_sampler']['batch_size'] = real_batch_size
    local_config['trainer']['num_gradient_accumulation_steps'] = num_accumulation_steps
    local_config['trainer']['optimizer'].update({
        'lr': trial.suggest_float("lr", 1e-6, 1e-3, log=True),
        'weight_decay': 0
        })
    return local_config

def objective(config, trial):
    local_config = get_config(config, trial)
    root_output_dir = os.path.join(OUTPUT_DIR,
            "trial_{}".format(trial.number))
    final_metric=dict((m,[]) for m in TARGET_METRICS)
    for ii in range(3): # three trials per set of hyperparams
        config_copy = copy.deepcopy(local_config) # building train_loop empties this dict
        seed = ii*17
        output_dir = os.path.join(root_output_dir,"seed_{}".format(seed))
        random.seed(seed)
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        print(json.dumps(config_copy, sort_keys=True, indent=2))
        train_loop = allennlp.commands.train.TrainModel.from_params(
                params=Params(config_copy),
                serialization_dir=output_dir,
                local_rank=0)
        metrics = train_loop.run()
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(local_config, f, indent=2, sort_keys=True)
        # archive final model
        # allennlp.models.archival.archive_model(output_dir)
        files_to_remove = sorted(glob.glob(output_dir+'/*.th'))
        for filename in files_to_remove: # make room to not clog folders
            os.remove(filename)
        for tgt_metric in TARGET_METRICS:
            final_metric[tgt_metric].append(metrics["best_validation_" + tgt_metric])
    # end for ii in range(3): # three trials per set of hyperparams
    print('===== metrics before aggregation ====='.upper())
    print(final_metric)
    print('-----')
    final_metric=tuple(mean(final_metric[m]) for m in TARGET_METRICS)
    return final_metric

def get_base_config(model_name=None):

    seed = 0

    # Training configs
    epochs = 200
    batch_size = 64

    # Model configs
    transformer_model = "t5-base" if model_name is None else model_name
    length_limit = 512

    # Path configs
    train_data_path = os.path.join(DATA_DIR, "data", "train")
    assert os.path.isfile(train_data_path), train_data_path
    dev_data_path = os.path.join(DATA_DIR, "data", "dev")
    assert os.path.isfile(dev_data_path), dev_data_path

    dataset_reader = {
      "type": "sara-bio-ie",
      "transformer_model_name": transformer_model,
      "data_folder": os.path.join(DATA_DIR, "sara_v3", "grounded_cases"),
      "max_length": length_limit
    }

    validation_dataset_reader = copy.deepcopy(dataset_reader)
    validation_dataset_reader["is_training"] = False

    model = {
      "type": "sara-bio-ie",
      "transformer_model_name": transformer_model,
      "max_length": 512,
      }

    output = {
      "dataset_reader": copy.deepcopy(dataset_reader),
      "validation_dataset_reader": copy.deepcopy(validation_dataset_reader),
      "train_data_path": train_data_path,
      "validation_data_path": dev_data_path,
      "vocabulary": {
          "type": "from_instances",
      },
      "model": copy.deepcopy(model),
      "data_loader": {
        "batch_sampler": {
          "type": "bucket",
          "batch_size": batch_size,
        }
      },
      "trainer": {
        "optimizer": {
          "type": "huggingface_adamw",
          "weight_decay": 0,
          "lr": 5e-5,
          "eps": 1e-8
        },
        "grad_clipping": 1.0,
        "num_epochs": epochs,
        "validation_metric": ["+{}".format(m) for m in TARGET_METRICS],
        "patience": 20,
        "num_gradient_accumulation_steps": 1,
        "learning_rate_scheduler": {
          "type": "reduce_on_plateau"
        },
        "callbacks": [
            {
                "type": "batch_results_printer",
                "output_file_path": "batch_results_printer",
                "using_logger": False,
                "validation_only": True
            }
        ],
      },
      "random_seed": seed,
      "numpy_seed": seed,
      "pytorch_seed": seed,
    }
    return copy.deepcopy(output)

if __name__ == "__main__":
    if version.parse(allennlp.__version__) < version.parse("2.0.0"):
        raise RuntimeError(
            "`allennlp>=2.0.0` is required for this example."
            " If you want to use `allennlp<2.0.0`, please install `optuna==2.5.0`"
            " and refer to the following example:"
            " https://github.com/optuna/optuna/blob/v2.5.0/examples/allennlp/allennlp_simple.py"
        )
    model_name=None
    if len(sys.argv)>1:
        model_name=sys.argv[1]
    seed=0
    if len(sys.argv)>2:
        seed=int(sys.argv[2])
    sys.stdout.flush()
    short_model_name=model_name.split('/')[-1]
    assert isinstance(short_model_name,str)
    assert len(short_model_name)>0
    assert '/' not in short_model_name
    OUTPUT_DIR = "/srv/local1/nholzen1/exp/optuna-sara-ie-{}".format(short_model_name)
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    STUDY_NAME = "optuna-sara-ie-{}.db".format(short_model_name)
    CONFIG = get_base_config(model_name)

    random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["pytorch_seed"])
    numpy.random.seed(CONFIG["numpy_seed"])
    for k in ["random_seed", "pytorch_seed", "numpy_seed"]:
        del CONFIG[k]

    #print('using seed {}'.format(seed))
    sampler = optuna.samplers.MOTPESampler(n_startup_trials=10)
    directions = ["maximize" for _ in TARGET_METRICS]
    study = optuna.create_study(study_name='study',
            storage='sqlite:////home/nholzen1/'+STUDY_NAME,
            directions=directions, sampler=sampler, load_if_exists=True)
    study.optimize(lambda x: objective(CONFIG, x), n_trials=1)

    print("Number of finished trials: ", len(study.trials))
    print("Best trials:")
    trials = study.best_trials

    for trial in trials:
        print('=====')
        print("  Values: ", trial.values)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        print('-----')
