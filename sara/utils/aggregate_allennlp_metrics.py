# This script aggregates scores found under <exp_folder>
# <exp_folder> is supposed to be the folder containing all experiment outputs produced by allennlp
# Usage: python sara/utils/aggregate_allennlp_metrics.py <exp_folder>
import sys
import glob
import os
import json
import copy
exp_folder=sys.argv[1]
metric_suffixes = ['f1', 'precision', 'recall']
splits = ['dev', 'test']
metric_prefixes = ['kbe', 'full kb']
seeds = [0, 17, 34, 51, 68]
metrics = {}
# load data
for split in splits:
    for filename in glob.glob(os.path.join(exp_folder, 'seed_*', '{}_metrics.json'.format(split))):
        print(filename)
        seed=filename.split('/')[-2]
        assert seed.startswith('seed_')
        seed=int(seed[5:])
        assert seed in seeds
        m = json.load(open(filename,'r'))
        key = (seed, split)
        assert key not in metrics
        metrics[key] = copy.deepcopy(m)
# format data
output = []
for seed in seeds:
    line = []
    for prefix in metric_prefixes:
        for split in splits:
            for suffix in metric_suffixes:
                key1 = (seed, split)
                assert key1 in metrics
                key2 = '-'.join((prefix, suffix))
                assert key2 in metrics[key1]
                value = metrics[key1][key2]
                line.append(value)
            # end for suffix in metric_suffixes:
        # end for split in splits:
    # end for prefix in metric_prefixes:
    # sanity check 1: precision of full kb is equal to that of partial KB
    dev_full_kb = line[7]
    dev_partial_kb = line[1]
    precision_equal = dev_full_kb == dev_partial_kb
    if not precision_equal:
        print('{} vs {}'.format(dev_full_kb,dev_partial_kb))
    assert line[4] == line[10] # test
    # sanity check 2: recall of full kb is lower than that of partial KB
    assert line[8] <= line[2] # dev
    assert line[11] <= line[5] # test
    line = [str(x) for x in line]
    output.append('\t'.join(line))
# end for seed in seeds:
print('\n'.join(output))
