#!/bin/bash
# Kills all docker containers running the deepmarl/pytorch image.
# Rebuilds docker image

if [ -z "$EXP_DIR" ]
then
    EXP_DIR=~
fi

echo "EXP_DIR: $EXP_DIR"
cd $EXP_DIR

# Check for the deepmarl repo, if it is not there then clone it
if [ ! -d "pymarl" ]; then
    echo "Cloning pymarl repo"
    # If this doesn't work on a server you might have to manually connect and clone the repo for the first time
    git clone git@github.com:oxwhirl/pymarl-dev.git
fi

cd $EXP_DIR/Anuj-MAVEN

git fetch -q origin
git checkout master
git reset --hard origin/master -q

# Kill the docker container
./kill.sh

# Rebuild the docker container
#./build.sh

# Install sc2
./install_sc2.sh