#PBS -l nodes=1:ppn=1:gpus=1:exclusive_process,mem=12gb
#PBS -S /bin/bash
#PBS -N LJ_data_ML
#PBS -j oe
#PBS -o LOG
#PBS -n
#PBS -l walltime=720:00:00
module purge
module load devel/cuda/8.0
cd $PBS_O_WORKDIR
make -B
./out 
