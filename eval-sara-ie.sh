# this script runs for a specific config file (see variable $config_file below)
root_output_folder=exp/sara-ie-LegalBert-alt-config-final-models
mkdir -p $root_output_folder
config_file=config-files/legalbert.json
seed=$SEED
output_folder=$root_output_folder"/seed_"$seed
mkdir -p $output_folder
rm -r $output_folder
allennlp train $config_file -s $output_folder \
    --include-package sara.data.dataset.ie --include-package sara.models.ie \
    --include-package sara.metrics.set_based_f1 --include-package sara.callbacks.sara_ie_callback \
    --overrides '{"random_seed": '$seed', "numpy_seed": '$seed', "pytorch_seed": '$seed'}'
# decode train, dev and test
for split in train dev test
do
    eval_file=resources/data/$split
    score_file=$output_folder/$split"_metrics.json"
    predictions_file=$output_folder/$split"_predictions.json"
    model_file=$output_folder/model.tar.gz
    allennlp evaluate $model_file $eval_file \
        --output-file $score_file \
        --include-package sara.data.dataset.ie --include-package sara.models.ie \
        --include-package sara.metrics.set_based_f1 --include-package sara.callbacks.sara_ie_callback \
        --predictions-output-file $predictions_file \
        --overrides '{"data_loader.batch_sampler.batch_size": 32, "dataset_reader.is_training": false}' || exit 0
done
date
