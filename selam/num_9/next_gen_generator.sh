#!/bin/bash
SELAM -d demography.txt -o output.txt -s selection.txt -h -c 1 0.01
SELAM_STATS -i population_output.txt -a 0.00001 > ../next_gen_simulation_finalt/$1
rm population_output.txt