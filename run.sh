#!/usr/bin/bash
set -e

G=C_4

python main.py \
	--group=$G \
	--results_dir=results \
	--overwrite

