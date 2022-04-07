#!/bin/bash

conda_env=$CONDA_DEFAULT_ENV
conda deactivate

archs=$(seq 0 6)

for gen_arch in ${archs[@]}
do
	for reg_arch in ${archs[@]}
	do
		name=ts_${gen_arch}_${reg_arch}
		tmux kill-session -t $name
		tmux new -d -s $name
		tmux send-keys -t ${name}.0 "clear" ENTER
		tmux send-keys -t ${name}.0 "conda activate $conda_env" ENTER
		tmux send-keys -t ${name}.0 "bash train.sh $gen_arch $reg_arch" ENTER
	done
done

conda activate $conda_env
