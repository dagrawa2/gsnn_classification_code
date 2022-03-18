#!/usr/bin/bash
set -e

groups=(\
"C_2" "C_3" "C_4" "C_5" "C_6" "C_7" "C_8" \
"D_3" "D_4" "D_5" "D_6" \
"C_2xC_2" "C_2xC_2xC_2" "C_2xC_4" \
"Q_8")

for G in ${groups[@]}
do
	python main.py \
		--group=$G \
		--results_dir=results \
		--overwrite
done

echo All done!
