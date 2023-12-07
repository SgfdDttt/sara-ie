# This script launches multiple eval jobs in parallel. It must be adapted to your environment
# before it can be used. This is just meant as a helpful example.
log_dir=sara-ie-LegalBert-final-model-logs
mkdir -p $log_dir
# remove old logs
rm $log_dir/*
script=eval-sara-ie.sh
for seed in 0 17 34 51 68
do
    logfile=$log_dir"/"$seed".log"
    if [ ! -f "$logfile" ]
    then
        echo $seed
        sbatch --export=SEED=$seed --output=$logfile $script
        sleep 1
    fi
done
