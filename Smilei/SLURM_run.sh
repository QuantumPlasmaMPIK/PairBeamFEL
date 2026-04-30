#!/bin/bash
#SBATCH --job-name=erciyes_smilei
#SBATCH --time=360:00:00
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=16    # number of threads per MPI process
#SBATCH --qos=standard


## Further bash options
# make sure the script stops if any command in it fails
set -euo pipefail
set -x

## additional SBATCH commands...
#SBATCH --error=log.err


# avoid side-effects initially
module purge

## Virtual python environment activation should be done manually once per session, as it can be used for multiple purposes, not only for Smilei
source /home/erciyes/virt_environments/Smilei_python/bin/activate

## Smilei environment variables and required modules
source /home/erciyes/virt_environments/Smilei_vars
module list

## Interface
usage="$(basename "$0") [-h]  [-s|--smilei_version <SMILEI_DIR>] [-n|--namelist <NAMELIST_FILE>] [-f|--subfolder <OUTPUT_SUBFOLDER>] [--test]
where:
	-h: show the usage help text and exit
	-s|--smilei_version: Directory of the Smilei Git repo under the /bin directory of the QUANTUMPLASMA, default is v4.8 
	-n|--namelist: namelist python script name to be used by the SMILEI binary, default is namelist.py
	-f|--subfolder: create a specific subfolder under the mirrored location for the simulation results, default is None
	--test: run smilei_test only to check parameters first, default is false"

## Set common variables
QUANTUMPLASMA_ROOT_FOLDER=/lfs/l8/theo/quantumplasma/
SMILEI_GITHUB_REFERENCE_RELATIVE_FOLDER=src/Smilei_FEL/Smilei/
SMILEI_RELATIVE_FOLDER=src/Smilei_FEL/FreeElectronLaser/Smilei/
#SMILEI_RELATIVE_FOLDER=$SMILEI_GITHUB_REFERENCE_RELATIVE_FOLDER
IO_FOLDER=/nfs/scratch/erciyes/simulations/Smilei/
BASH_SCRIPT_DIR="${PWD}"
# actual call location of the installed script --> ${BASEDIR}
NAMELIST_FILE=namelist_POC.py
OUTPUT_SUBFOLDER="main"
run_smilei_test=false

## Parse the relevant options & arguments
while [[ "$#" -gt 0 ]]; do
    key="$1"
    case $key in
	-h|--help)  echo "$usage";  exit 0 ;;
	-s|--smilei_version) SMILEI_RELATIVE_FOLDER=bin/"$2"/Smilei/; shift ;;
	-n|--namelist) NAMELIST_FILE="$2"; shift ;;
	-f|--subfolder) OUTPUT_SUBFOLDER="$2"; shift ;;
	--test)  run_smilei_test=true;  ;;
	*) ;;
    esac
    shift
done



## SMILEI Optimizations
# Dynamic scheduling for patch-spec loop
export OMP_SCHEDULE=dynamic
# number of OpenMP threads per MPI process
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
# Binding OpenMP Threads of each MPI process on cores
export OMP_PLACES=cores


## Create corresponding simulation subfolder(as a swapped of the analysis dir.) and copy aux. scripts (job-submit, namelist, postprocess)
rm -fr $BASH_SCRIPT_DIR/$OUTPUT_SUBFOLDER
mkdir -p $BASH_SCRIPT_DIR/$OUTPUT_SUBFOLDER
cp $BASH_SCRIPT_DIR/$NAMELIST_FILE $BASH_SCRIPT_DIR/postprocess.py $BASH_SCRIPT_DIR/postprocess_3d.py $BASH_SCRIPT_DIR/FEL_analysis.py $BASH_SCRIPT_DIR/$OUTPUT_SUBFOLDER
F=${BASH_SCRIPT_DIR//analysis/simulations}/$OUTPUT_SUBFOLDER/simulation_results
#F=$IO_FOLDER/$OUTPUT_SUBFOLDER/simulation_results
rm -fr $F
mkdir -p $F
cp $BASH_SCRIPT_DIR/SLURM_run.sh $F/
# smilei namelist file is as a default, namelist.py file in the same subfolder with this runner script
cp $BASH_SCRIPT_DIR/$NAMELIST_FILE  $F/
cd $F/ || exit -1


## Call SLURM job allocation for the SMILEI execution
# --hint=nomultithread --distribution=block:block
if [ "$run_smilei_test" = true ] ; then
    srun  --hint=nomultithread --distribution=block:block  $QUANTUMPLASMA_ROOT_FOLDER/$SMILEI_RELATIVE_FOLDER/smilei_test $NAMELIST_FILE
else
    srun  --hint=nomultithread --distribution=block:block  $QUANTUMPLASMA_ROOT_FOLDER/$SMILEI_RELATIVE_FOLDER/smilei $NAMELIST_FILE
fi

# copy the job log output into the corresponding analysis subfolder
mv $BASH_SCRIPT_DIR/slurm-$SLURM_JOB_ID.out $BASH_SCRIPT_DIR/$OUTPUT_SUBFOLDER/SLURM_log.out

