#!/bin/bash
for n in -1
do
	for delta in 2
	do
		for s in 1 
		do
			for d in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
			do
				for v in 1000
				do
					for p in 0.25
					do
						for h in 2 5 10 20
						do
							sbatch run_gmm_nn.sh $n $delta $d 42 $s $h $v ccg_results ccg_models $p
						done
					done
				done
			done
		done
	done
done
