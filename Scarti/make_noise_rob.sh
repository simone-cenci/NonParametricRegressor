#!/bin/bash

declare -a models=('constant' 'linear' 'kernel' 'tree' 'forest')

for j in "${models[@]}"
do
	python noise_robustness.py $j -o 3
done
