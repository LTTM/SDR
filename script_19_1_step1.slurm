#!/bin/bash
#SBATCH --job-name=C_19_1
#SBATCH --error=C_19_1.%j.err
#SBATCH --output=C_19_1.%j.out
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:titan_rtx:1
#SBATCH --ntasks=2
#SBATCH --mem=40G
#SBATCH --time=002:00:00

# parameters
batch_size=8
task='19-1'

names='SDR'

loss_featspars=(0.001)
lfs_normalization=('max_maskedforclass')
lfs_shrinkingfn=('exponential')
lfs_loss_fn_touse='ratio'

loss_de_prototypes=(0.01)


loss_fc=(0.001)
lfc_sep_clust=(0.001)
loss_kd=(100)

steps=(1)

epochs=30
lr_step0=0.001
lr_stepN=0.0001


for (( l=0; l<${#loss_featspars[*]}; l++ )); do
	for (( m=0; m<${#lfs_normalization[*]}; m++ )); do
		for (( n=0; n<${#lfs_shrinkingfn[*]}; n++ )); do
			for (( o=0; o<${#loss_de_prototypes[*]}; o++ )); do
				for (( p=0; p<${#loss_fc[*]}; p++ )); do
					for (( t=0; t<${#loss_kd[*]}; t++ )); do
						for (( s=0; s<${#steps[*]}; s++ )); do
							START=`/bin/date +%s`  ### fake start time, for initialization purposes
							while [ $(( $(/bin/date +%s) - 2 )) -lt $START ]; do
								START=`/bin/date +%s` ### true start time
								echo $START
								log_dir='logs/'$task'/'$task'_'$names'/'
								out_file='outputs/'$task'/output_'$task'_'$names'.txt'

								echo $out_file
								
								/bin/sleep 2
								if [ ! -f "$out_file" ]; then

									singularity exec --nv pytorch_v2.sif \
									python -u -m torch.distributed.launch 1> $out_file  2>&1 \
									--nproc_per_node=1 run.py \
									--batch_size $batch_size \
									--logdir $log_dir \
									--dataset voc \
									--name $names \
									--task $task \
									--step ${steps[s]} \
									--lr $lr_stepN \
									--epochs $epochs \
									--debug \
									--method $names \
									--sample_num 10 \
									--where_to_sim GPU_windows \
									--step_ckpt 'logs/'$task'/'$task'-voc_FT/'$task'-voc_FT_0.pth'
								fi
							done
						done
					done
				done
			done
		done
	done
done
