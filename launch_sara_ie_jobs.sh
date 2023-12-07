# This script launches multiple optuna jobs in parallel. It must be adapted to your environment
# before it can be used. This is just meant as a helpful example.
model_name="roberta-base"
log_dir=sara-ie-"$model_name"-logs
mkdir -p $log_dir
# delete all old jobs
rm -r exp/optuna-sara-ie-"$model_name"/*
rm $log_dir/*
rm optuna-sara-ie-"$model_name".db

script=search-sara-ie.sh

# first experiments to start the db
logfile=$log_dir"/ie_0.log"
sbatch -o $logfile --export=MODEL_NAME=$model_name --job-name="ie_0_"$model_name $script
sleep 60

# other experiments
sbatch -o $log_dir"/ie_%a.log" --array=1-99%5 --export=MODEL_NAME=$model_name --job-name="ie_"$model_name $script
