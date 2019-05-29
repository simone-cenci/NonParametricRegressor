#!/bin/bash

declare -a models=('constant' 'linear' 'kernel' 'tree' 'forest')
data_type='empirical'

for j in "${models[@]}"
do
	python length_robustness.py $j $data_type -o 3
done
