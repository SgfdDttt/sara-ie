sara_folder=resources/sara_v3
output_folder=$sara_folder/grounded_cases
mkdir -p $output_folder
rm -r $output_folder
mkdir -p $output_folder
for file in $sara_folder/cases/prolog/*.pl
do
    echo $file
    python sara/data/dataset/ground_case.py \
        --statutes $sara_folder/statutes/prolog/ \
        --case $file \
        --output $output_folder || exit 0
done
