#!/bin/bash
#SBATCH --job-name=C_19_1
#SBATCH --error=C_19_1.%j.err
#SBATCH --output=C_19_1.%j.out
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:titan_rtx:1
#SBATCH --ntasks=2
#SBATCH --mem=40G
#SBATCH --time=010:00:00

# parameters
batch_size=8
task='19-1'

names='FT'

task='19-1'
steps=1

epochs=30
lr_step0=0.001
lr_stepN=0.0001


singularity exec --nv pytorch_v2.sif \
python -u -m torch.distributed.launch 1> 'outputs/'$task'/output_'$task'_'$names'_step0.txt' 2>&1 \
--nproc_per_node=1 run.py \
--batch_size $batch_size \
--logdir logs/$task/ \
--dataset voc \
--name $names \
--task $task \
--step 0 \
--lr $lr_step0 \
--epochs $epochs \
--debug \
--sample_num 10 \
--unce \
--loss_de_prototypes 1 \
--where_to_sim GPU_windows
