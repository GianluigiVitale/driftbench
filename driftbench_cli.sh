#!/bin/bash
# DriftBench CLI Wrapper Script
# Makes it easy to run DriftBench commands from the repo_driftcli folder

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add driftbench to Python path
export PYTHONPATH="${SCRIPT_DIR}/driftbench:${PYTHONPATH}"

# Run the CLI with all arguments
python3 -m driftbench.cli "$@"
