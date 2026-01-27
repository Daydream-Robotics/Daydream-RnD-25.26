#!/bin/bash
BASEDIR=$(dirname $0)
source ~/cam_venv/bin/activate
python3 ${BASEDIR}/pi_vex_serial.py > ~/Pi.out 2>&1