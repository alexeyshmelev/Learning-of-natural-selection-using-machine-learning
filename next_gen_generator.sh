#!/bin/bash
SELAM -d demography.txt -o output.txt -s selection.txt -h -c 1 0.01
SELAM_STATS -i population_output.txt -a 0.0001 > next_gen_simulation/$1
rm population_output.txt
