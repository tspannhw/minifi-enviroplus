#!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M")
source /opt/intel/openvino/bin/setupvars.sh 
fswebcam -q -r 1280x720 --no-banner /opt/demo/images/$DATE.jpg
python3 -W ignore /opt/intel/openvino/build/test.py /opt/demo/images/$DATE.jpg 2>/dev/null 
#python3 /opt/intel/openvino/build/test.py /opt/demo/images/$DATE.jpg
