REM@echo off 
setLocal EnableDelayedExpansion

SET /A batch_size=8
SET /A epochs=30
SET lr_step0=0.001
SET lr_stepN=0.0001

SET names=SDR
SET task=19-1
SET steps=1

REM 1-st step for all the scenarios in task 19-1
for %%s in (%steps%) do (
	for %%n in (%names%) do (
	   SET logdir=logs/%task%/
	   SET out_file=outputs/%task%/output_%task%_%%n_step%%s.txt
	   echo !out_file!
	   
	   CALL python -u -m torch.distributed.launch 1> !out_file! 2>&1 ^
	   --nproc_per_node=1 run.py ^
	   --batch_size %batch_size% ^
	   --logdir !logdir! ^
	   --dataset voc ^
	   --name %%n ^
	   --task %task% ^
	   --step %%s ^
	   --lr %lr_stepN% ^
	   --epochs %epochs% ^
	   --method %%n ^
	   --debug ^
	   --sample_num 10 ^
	   --where_to_sim GPU_windows ^
	   --step_ckpt logs/%task%/%task%-voc_FT/%task%-voc_FT_0.pth
	)
)
