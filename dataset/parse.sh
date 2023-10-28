#/bin/bash

while read line
do
    victim_meth_coverage=$(echo $line | cut -d',' -f8)
    polluter_meth_coverage=$(echo $line | cut -d',' -f9)
    c_or_nc_coverage=$(echo $line | cut -d',' -f10)

    if [[ $victim_meth_coverage == "" ]]; then
        continue
    fi
    if [[ $polluter_meth_coverage == "" ]]; then
        continue
    fi
    if [[ $c_or_nc_coverage == "" ]]; then
        continue
    fi
    
    echo $line >> tmp.csv

done < $1
