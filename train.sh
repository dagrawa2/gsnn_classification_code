#!/usr/bin/bash
set -e

G=C_6
archidxs=(3 4)
seeds=(0 1 2 3 4 5)

for genidx in ${archidxs[@]}
do
	for regidx in ${archidxs[@]}
	do
		for genseed in ${seeds[@]}
		do
			for regseed in ${seeds[@]}
			do
				python train.py \
					--dimension=6 \
					--n_points=10 \
					--input_dir=results/$G/npzs \
					--generator_idx=$genidx \
					--regressor_idx=$regidx \
					--generator_seed=$genseed \
					--regressor_seed=$regseed \
					--batch_size=10000 \
					--epochs=2 \
					--lr=5e-3 \
					--val_batch_size=100000 \
					--val_interval=1 \
					--device=cpu \
					--output_dir=results/$G/gsnns/archs${genidx}_${regidx}_seeds${genseed}_${regseed} \
					--exist_ok
			done
		done
	done
done

echo All done!
