#!/usr/bin/env bash

set -euo pipefail

function yell() {
	echo -e "\e[1m$*\e[0m"
}

yell "THIS SCRIPT IS WIP AND HAS NOT BEEN TESTED!"
echo "Only run this script on the coprocessor."
yell "DO NOT RUN THIS SCRIPT ON YOUR LAPTOP"
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

NO_CLONE=${NO_CLONE:-}
NO_BUILD=${NO_BUILD:-}

# some env vars
if [[ -n "$NO_CLONE" ]]; then
	yell "NO_CLONE set -- skipping clone/pull"
fi

if [[ -n "$NO_BUILD" ]]; then
	yell "NO_BUILD set -- skipping build"
fi

# just in case
if [[ $REPOSITORY == "" ]]; then
	echo "Repository is null, exiting"
	exit 1
fi

# I'm really paranoid about this
if [[ $DATA_DIR == "" ]]; then
	echo "Data dir is null, exiting"
	exit 1
fi

mkdir -p "$DATA_DIR"

if [[ -f "$LOG_FILE" ]]; then
	echo "Overwriting $LOG_FILE"
	echo -n "" > "$LOG_FILE"
fi

echo "Logging to $LOG_FILE, please use that for detailed info"

function runCommand() {
	echo -n "Running \"$*\"... "
	yell "Running \"$*\" in \"$(pwd)\"" >> "$LOG_FILE"
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
	yell "Running \"${*:2}\" in \"$(pwd)\"" >> "$LOG_FILE"
	if "${@:2}" &>>"$LOG_FILE"; then
		echo "Done"
	else
		echo "Failed"
		exit 1
	fi
}

function runAsRoot() {
	echo -n "Running \"$*\"... "
	yell "Running \"$*\" in \"$(pwd)\" as root" >> "$LOG_FILE"
	# shellcheck disable=SC2024
	if sudo <<<"$PASSWORD" -E -S "$@" &>>"$LOG_FILE"; then
		echo "Done"
	else
		echo "Failed -- check log for more info"
		exit 1
	fi
}

function cloneIf() {
	yell "Cloning \"$1\" to dir \"$2\"" >> "$LOG_FILE"
	echo -n "Cloning \"$1\"... "
	if git -C "$2" pull &>>"$LOG_FILE"; then
		echo "Skipped"
	else
		runCommand git clone --depth 1 --shallow-submodules --recursive "$1" "$2" &>/dev/null
		echo "Done"
	fi
}

export USE_PRIORITIZED_TEXT_FOR_LD=1
export CMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc/"
#assertCommand "if cuda compiler exists" test -f "$CMAKE_CUDA_COMPILER"
export CUDA_HOME="/usr/local/cuda"
#assertCommand "if cuda home exists" test -d "$CUDA_HOME"
export MAX_JOBS=3 # for memory reasons
#export FORCE_CUDA=1

cd "$REPOSITORY"

yell "RUNNING SYSTEM UPGRADE"
runAsRoot apt-get update -y
runAsRoot apt-get upgrade -y

yell "INSTALLING JETPACK"
runAsRoot apt-get install -y nvidia-jetpack

yell "RUNNING PIP COMMANDS"
if [[ -f ./venv/bin/activate ]]; then
	echo "venv already exists -- skipping creation"
else
	runCommand python3 -m venv venv
fi
# shellcheck disable=SC1091
source ./venv/bin/activate
runCommand pip install --upgrade pip
runCommand pip install pyntcore "--index-url=https://wpilib.jfrog.io/artifactory/api/pypi/wpilib-python-release-2024/simple"
runCommand pip install -r requirements.txt
runCommand pip uninstall --yes torch torchvision # we'll install those manually

if [[ -z $NO_CLONE ]]; then
	yell "CLONING PYTORCH"
	cd "$DATA_DIR"
	cloneIf "https://github.com/pytorch/pytorch.git" pytorch
	cd pytorch
	runCommand git submodule update --init --recursive --depth 1 # in case they add more submodules
fi

yell "BUILDING AND INSTALLING PYTORCH"
cd "$DATA_DIR/pytorch"
runAsRoot apt-get install build-essential cmake ninja-build python3-pip
runCommand pip install -r requirements.txt
runAsRoot apt-get install python3-setuptools # workaround https://github.com/pytorch/pytorch/issues/129304
if [[ -z "$NO_BUILD" ]]; then
	runAsRoot /usr/bin/python3 setup.py bdist_wheel
fi
runCommand pip install ./dist/*.whl
cd "$REPOSITORY" # can't be run in the pytorch dir for reasons
assertCommand "pytorch uses cuda" python3 -c "
import torch
if torch.cuda.device_count() == 1:
	exit(0)
else:
	exit(1)
"

if [[ -z $NO_CLONE ]]; then
	yell "CLONING TORCHVISION"
	cd "$DATA_DIR"
	cloneIf "https://github.com/pytorch/vision.git" vision
	cd vision
	runCommand git submodule update --init --recursive --depth 1
fi

yell "BUILDING AND INSTALLING TORCHVISION"
cd "$DATA_DIR/vision"
if [[ -z "$NO_BUILD" ]]; then
	runCommand /usr/bin/python3 setup.py bdist_wheel
fi
runCommand pip install ./dist/*.whl
cd "$REPOSITORY"
assertCommand "torchvision starts" python3 -c "
import torch
import torchvision
"

yell "SETTING UP NETPLAN"
runAsRoot apt install netplan.io
runAsRoot mkdir -p /etc/netplan
runAsRoot touch /etc/netplan/50-robot.yaml
runAsRoot chmod 600 /etc/netplan/50-robot.yaml
runAsRoot tee /etc/netplan/50-robot.yaml <<EOF
network:
    version: 2
    renderer: NetworkManager
    ethernets:
        enP8p1s0:
            addresses:
                - 10.9.72.10/24
            dhcp4: True
            optional: True
EOF

yell "SETUP COMPLETE!"
echo "Remember to source $REPOSITORY/venv/bin/activate before trying to run the code."

