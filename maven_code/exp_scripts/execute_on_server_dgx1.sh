#!/bin/bash
# Kills the existing docker container and restarts a new detached container

GPU_ID=$1
N_REPEAT=$2
PARAMS="${@:3}"

# I am not sshed into the server in my home directory

if [ -z "$EXP_DIR" ]
then
    EXP_DIR=~
fi

echo "EXP_DIR: $EXP_DIR"

# Update the git repo
echo "Updating git repo"
#mkdir -p $EXP_DIR/deepmarl
cd $EXP_DIR/Anuj-MAVEN
# echo "REMEMBER TO ALLOW UPDATING OF THE REPO IN execute_on_server.sh AFTER TESTING!"
git fetch -q origin
git checkout master
git reset --hard origin/master -q

# Run the experiment $N_REPEAT times on GPU $GPU
bash ./run_dgx1.sh $GPU_ID "bash exp_scripts/repeat_exp.sh $N_REPEAT $PARAMS"