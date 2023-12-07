# This script launches a single optuna job on a compute grid. It must be adapted to your environment
# before it can be used. This is just meant as a helpful example.

# delay onset of array jobs to avoid collisions when reading/writing to the database
if [ -z ${SLURM_ARRAY_TASK_ID+x} ]
then
    echo not sleeping
    optuna_seed=0
else
    sleep_time=$(expr $SLURM_ARRAY_TASK_ID % 20)
    sleep_time=$(expr $sleep_time "*" 10)
    echo sleeping for $sleep_time
    sleep $sleep_time
    optuna_seed=$SLURM_ARRAY_TASK_ID
fi
python hparams_search_sara_ie.py $MODEL_NAME $optuna_seed
