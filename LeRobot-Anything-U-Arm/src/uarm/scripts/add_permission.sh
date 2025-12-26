#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

find $SCRIPT_DIR -type f -exec chmod +x {} \;

echo "[INFO] All files in $SCRIPT_DIR have been granted executable permissions."
