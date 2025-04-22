#!/bin/bash
for n in -1
do
	for delta in 2
	do
		for s in 0.5 1 1.5 2 3 4
		do
			for d in 2 3 4 5
			do
				for v in 1000
				do
					for h in 2 5 10
					do
						sbatch run_xor.sh $n $delta $d 42 $s $h $v xor_results_nn xor_models_nn $p
					done
				done
			done
		done
	done
done
