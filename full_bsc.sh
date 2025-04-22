#!/bin/bash
for pn in 0.25
do
	for n in -1
	do
		for p in 0.25 0.5
		do
			for s in -1
			do
				sbatch run_bsc.sh $s $p $n 42 $pn
			done
		done
	done
done
