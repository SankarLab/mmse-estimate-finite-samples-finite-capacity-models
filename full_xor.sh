#!/bin/bash
for n in -1
do
	for delta in 1 2
	do
		for s in 0.5 1 1.5 2 
		do
			for d in 2 3 4 5
			do
				for v in 1000
				do
#					for h in 2 5 10
					#do
						# sbatch run_xor_nn.sh $n $delta $d 42 $s $h $v xor_nn_results xor_nn_models
					#done
					sbatch run_xor_linear.sh $n $delta $d 42 $s 1 $v xor_linear_results xor_linear_models
				done
			done
		done
	done
done
