#!/usr/bin/bash
set -e

gen_arch=$1
reg_arch=$2

G=C_6
ranks=(6 4 5 3 2 2 1)

gen_rank=${ranks[gen_arch]}
reg_rank=${ranks[reg_arch]}

gen_inits=$(seq -$gen_rank $gen_rank)
reg_inits=$(seq -$reg_rank $reg_rank)

for gen_init in ${gen_inits[@]/0}
do
	for reg_init in ${reg_inits[@]/0}
	do
		python train.py \
			--dimension=6 \
			--n_points=8 \
			--input_dir=results/$G/npzs \
			--generator_arch_idx=$gen_arch \
			--regressor_arch_idx=$reg_arch \
			--generator_init_w=$gen_init \
			--regressor_init_w=$reg_init \
			--batch_size=4096 \
			--epochs=20 \
			--lr=5e-3 \
			--val_batch_size=65536 \
			--val_interval=1 \
			--device=cpu \
			--output_dir=results/$G/gsnns/archs${gen_arch}_${reg_arch}/inits${gen_init}_${reg_init} \
			--exist_ok
	done
done

echo All done!
