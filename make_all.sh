#!/bin/bash

declare -a models=('constant' 'linear' 'kernel' 'tree' 'forest')
data_type='synthetic'

for j in "${models[@]}"
do
	python main.py $j $data_type -o 3
done
