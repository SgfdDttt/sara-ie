reference=resources/sara_v3/grounded_cases/
script=sara/metrics/compute_kb_metrics_grounded_kb.py
for model in sara-ie-bert-base-cased-final-models sara-ie-custom-legalbert-alt-config-final-models sara-ie-t5-base-final-models sara-ie-roberta-base-final-models sara-ie-LegalBert-alt-config-final-models
do
    for seed in 0 17 34 51 68
    do
        prefix=/brtx/604-nvme1/nholzen1/exp/$model/seed_$seed
        for split in dev test
        do
            prediction=$prefix/"$split"_predictions.json
            #echo $prediction
            echo $model $split $seed
            python $script --reference $reference --prediction $prediction || exit 0
        done
    done
done
