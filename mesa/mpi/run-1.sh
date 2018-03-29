#!/bin/bash
#SBATCH -J mesa          # job name
#SBATCH -o mesa-1.o%j             # output file name (%j expands to jobID)
#SBATCH -e mesa-1.e%j             # error file name (%j expands to jobID)
#SBATCH -n 1
#SBATCH -t 32:00:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/tidal-circularization/mesa/1Msun

date

./rn

date
