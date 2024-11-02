#!/usr/bin/env bash

set -euo pipefail

echo "THIS SCRIPT IS WIP AND HAS NOT BEEN TESTED!"
read -rp "Are you sure you want to continue? [yN]
>" toContinue

if [[ ! ($toContinue == "yes" || $toContinue == "y") ]]; then
	echo "Exiting"
	exit 0
fi

if ! git rev-parse &>/dev/null --is-inside-work-tree; then
	echo "Please run this script inside the repo"
	exit 1
fi

# screw good practice (it's probably just ironclaw anyway)
read -rp "Please enter the device password
>" PASSWORD

echo -n "Checking if password is correct... "

if sudo -kS <<<"$PASSWORD" &>>/dev/null true; then
	echo "Done"
else
	echo "Failed"
	exit 1
fi

echo -n "Checking for internet connection (trying to connect to github.com)... "

if ping -i 0.2 -w 1 github.com &>>/dev/null; then
	echo "Done"
else
	echo "Failed"
	exit 1
fi

REPOSITORY="$(git rev-parse --show-toplevel)"
DATA_DIR="$HOME/Downloads"
LOG_FILE="$REPOSITORY/setup.log"

mkdir -p "$DATA_DIR"

if [[ $REPOSITORY == "" ]]; then
	echo "Repository is null, exiting"
	exit 1
fi

if [[ $DATA_DIR == "" ]]; then
	echo "Temp dir is null, exiting"
	exit 1
fi

echo "Logging to $LOG_FILE, please use that for detailed info"

function runCommand() {
	echo -n "Running \"$*\"... "
	echo "**Running \"$*\" in \"$(pwd)\"**" >> "$LOG_FILE"
	if "$@" &>>"$LOG_FILE"; then
		echo "Done"
	else
		echo "Failed -- check log for more info"
		exit 1
	fi
}

# same as runCommand but configurable message
function assertCommand() {
	echo -n "Checking $1... "
	echo "**Running \"${*:2}\" in \"$(pwd)\"**" >> "$LOG_FILE"
	if "${@:2}" &>>"$LOG_FILE"; then
		echo "Done"
	else
		echo "Failed -- check log for more info"
		exit 1
	fi
}

function runAsRoot() {
	echo -n "Running \"$*\"... "
	echo "**Running \"$*\" in \"$(pwd)\" as root**" >> "$LOG_FILE"
	# shellcheck disable=SC2024
	if sudo <<<"$PASSWORD" -E -S "$@" &>>"$LOG_FILE"; then
		echo "Done"
	else
		echo "Failed -- check log for more info"
		exit 1
	fi
}

function cloneIf() {
	echo "**Cloning \"$1\" to dir \"$2\"**" >> "$LOG_FILE"
	echo -n "Cloning \"$1\"... "
	if git -C "$2" pull &>>"$LOG_FILE"; then
		echo "Skipped"
	else
		runCommand git clone --depth 1 --shallow-submodules --recursive "$1" "$2"
		echo "Done"
	fi
}

export USE_PRIORITIZED_TEXT_FOR_LD=1
# TODO: check both of these locations
export CMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc/"
export CUDA_HOME="/usr/local/cuda"
export MAX_JOBS=2 # for memory reasons
#export FORCE_CUDA=1

cd "$REPOSITORY"

echo "RUNNING SYSTEM UPGRADE"
runAsRoot apt-get update -y
runAsRoot apt-get upgrade -y

echo "INSTALLING JETPACK"
runAsRoot apt-get install -y nvidia-jetpack

echo "RUNNING PIP COMMANDS"
# TODO: don't recreate env if it already exists
runCommand python3 -m venv venv
# shellcheck disable=SC1091
source ./venv/bin/activate
runCommand pip install -r requirements.txt
runCommand pip install setuptools
runCommand pip uninstall --yes torch torchvision # we'll install those manually

echo "CLONING PYTORCH"
cd "$DATA_DIR"
cloneIf "https://github.com/pytorch/pytorch.git" pytorch

echo "BUILDING PYTORCH"
cd pytorch
runAsRoot apt-get install build-essential cmake ninja-build
runCommand pip install -r requirements.txt
runAsRoot python3 setup.py bdist_wheel
runCommand pip install ./dist/*.whl
assertCommand "pytorch install" python3 -c <<EOF
import torch
if torch.cuda.device_count() == 1:
	exit(0)
else:
	exit(1)
EOF

echo "CLONING TORCHVISION"
cd "$DATA_DIR"
cloneIf "https://github.com/pytorch/vision.git" vision

echo "BUILDING TORCHVISION"
cd vision
runCommand pip install -r requirements.txt
runCommand python3 setup.py bdist_wheel
runCommand pip install ./dist/*.whl
assertCommand "torchvision install" python3 -c <<EOF
import torch
import torchvision
EOF

echo "SETUP COMPLETE"
echo "Please restart now."

