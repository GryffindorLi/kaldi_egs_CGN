#!/bin/sh

# You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=general

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=long

# The default run (wall-clock) time is 1 minute
#SBATCH --time=144:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# The default number of CPUs per task is 1 (note: CPUs are always allocated to jobs per 2)
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
#SBATCH --cpus-per-task=32

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=40G

# Request a GPU
#SBATCH --gres=gpu:3

# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --mail-type=END

# Measure GPU usage of your job (initialization)
#previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

# Use this simple command to check that your sbatch settings are working (it should show the GPU that you requested)
#/usr/bin/nvidia-smi

# Your job commands go below here

# Uncomment these lines and adapt them to load the software that your job requires
module use /opt/insy/modulefiles
module load cuda/11.2 cudnn/11.2-8.1.1.33 devtoolset/8 intel/oneapi nccl/11.0-2.9.8

# Computations should be started with 'srun'. For example:
srun sh ./local/nnet3/run_ivector_common.sh --stage 0 --nj 1 --min-seg-len 1.55 --train-set train --gmm tri4 --num-threads-ubm 32 --nnet3-affix ""

srun sh ./run_tdnn_1a_updated.sh

# Your job commands go above here

# Measure GPU usage of your job (result)
#/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
