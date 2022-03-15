#!/usr/bin/bash
set -e

groups=("C_4" "C_8" "D_3" "D_6")

for G in ${groups[@]}
do
	python main.py \
		--group=$G \
		--results_dir=results \
		--overwrite
done

echo All done!
