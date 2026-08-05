#!/bin/bash

#SBATCH --job-name=build_benchmark
#SBATCH --output=logs/build_benchmark_%j.log
#SBATCH --error=logs/build_benchmark_%j.err
#SBATCH -p general
#SBATCH -G a100:1
#SBATCH -q public
#SBATCH -c 16
#SBATCH --mem 30G
#SBATCH -t 0-02:00:00   # time in d-hh:mm:ss (2 hours is plenty)

# Setup environment
module purge
module load mamba/latest
source activate /data/hkerner/ush/tram-ovs/.conda/envs/ovs_env

# Ensure logs directory exists
mkdir -p logs

BASE_DIR='/data/hkerner/ush/taxonomy-driven-ovs-models'

cd ${BASE_DIR}
echo "Benchmark Builder"

PYTHONPATH=${BASE_DIR} /data/hkerner/ush/tram-ovs/.conda/envs/ovs_env/bin/python expanded_benchmark_builder.py
