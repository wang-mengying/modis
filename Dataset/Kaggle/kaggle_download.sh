#!/bin/bash

count=0
limit=2000
max_size=1048576

for page in {1..100}
do
    kaggle datasets list --file-type csv --max-size $max_size --csv --page $page > datasets.csv

    while IFS=, read -r ref title size url rest
    do
        if [[ "$ref" != "ref" ]]; then
            kaggle datasets download "$ref" --unzip -p ./extra1/
            ((count++))
            if [[ $count -ge $limit ]]; then
                break 2
            fi
        fi
    done < datasets.csv

    rm datasets.csv
done

echo "Downloaded $count datasets."