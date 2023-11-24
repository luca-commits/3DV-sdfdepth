#!/bin/bash

#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH -A ls_polle
#SBATCH --job-name=render_masks
#SBATCH --output=render_masks_%A_%a.out

python render_masks.py
