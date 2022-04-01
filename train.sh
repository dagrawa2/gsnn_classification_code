#!/usr/bin/bash
set -e

groups=("C_6")

for G in ${groups[@]}
do
	python train.py \
		--dimension=6 \
		--n_points=10 \
		--input_dir=results/$G/npzs \
		--generator_idx=0 \
		--regressor_idx=0 \
		--seed=0 \
		--batch_size=10000 \
		--epochs=25 \
		--lr=5e-3 \
		--val_batch_size=100000 \
		--val_interval=1 \
		--device=cpu \
		--output_dir=results/$G/gsnns \
		--exist_ok
done

echo All done!
