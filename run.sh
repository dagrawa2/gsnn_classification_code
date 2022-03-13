#!/usr/bin/bash
set -e

G=C_8

python main.py \
	--group=$G \
	--output_dir=results/$G
