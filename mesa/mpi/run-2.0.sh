#!/bin/bash
#SBATCH -J mesa          # job name
#SBATCH -o mesa-2.0.o%j             # output file name (%j expands to jobID)
#SBATCH -e mesa-2.0.e%j             # error file name (%j expands to jobID)
#SBATCH -n 1
#SBATCH -t 12:00:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/tidal-circularization/mesa/2.0Msun

date

./rn

date
