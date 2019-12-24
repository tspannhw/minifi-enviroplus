import cv2 as cv
import time
import uuid
from time import gmtime, strftime
import json
import argparse
import os.path
import re
import sys
import tarfile
import os
import datetime
import math
import random, string
import base64
import psutil
import socket


def getCPUtemperature():
  res = os.popen('vcgencmd measure_temp').readline()
  return (res.replace("temp=", "").replace("'C\n", ""))

def IP_address():
    try:
      external_IP_and_port = ('198.41.0.4', 53)  # a.root-servers.net
      socket_family = socket.AF_INET
      s = socket.socket(socket_family, socket.SOCK_DGRAM)
      s.connect(external_IP_and_port)
      answer = s.getsockname()
      s.close()
      return answer[0] if answer else None
    except socket.error:
      return None

# Load the model.
net = cv.dnn.readNet('/opt/intel/openvino/build/face-detection-adas-0001.xml',
                     '/opt/intel/openvino/build/face-detection-adas-0001.bin')
# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
# input image
inputImage = sys.argv[1]

# Read an image.
frame = cv.imread(inputImage)
if frame is None:
    raise Exception('Image not found!')
# Prepare input blob and perform an inference.
blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U)
net.setInput(blob)
out = net.forward()

row = {}
counter = 0
# yyyy-mm-dd hh:mm:ss
currenttime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
host = os.uname()[1]
starttime = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
start = time.time()
ipaddress = IP_address()
end = time.time()
usage = psutil.disk_usage("/")
row['host'] = os.uname()[1]
uuid2 = '{0}_{1}'.format(strftime("%Y%m%d%H%M%S", gmtime()), uuid.uuid4())
cpuTemp = int(float(getCPUtemperature()))
row['cputemp'] = str(round(cpuTemp, 2))
row['ipaddress'] = str(ipaddress)
row['endtime'] = '{0:.2f}'.format(end)
row['runtime'] = '{0:.2f}'.format(end - start)
row['systemtime'] = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
row['starttime'] = str(starttime)
row['diskfree'] = "{:.1f}".format(float(usage.free) / 1024 / 1024)
row['memory'] = str(psutil.virtual_memory().percent)
row['uuid'] = str(uuid2)

# Draw detected faces on the frame.
for detection in out.reshape(-1, 7):
    confidence = float(detection[2])
    xmin = int(detection[3] * frame.shape[1])
    ymin = int(detection[4] * frame.shape[0])
    xmax = int(detection[5] * frame.shape[1])
    ymax = int(detection[6] * frame.shape[0])
    if confidence > 0.5:
        print("Confidence:" + str(confidence) + " XMin: " + str(xmin) + " YMIN:" + str(ymin) + " XMAX:" + str(xmax) + " YMAX:" + str(ymax)) 
        row['confidence' + counter] = str(confidence)
        row['xmin' + counter] = str(xmin)
        row['ymin' + counter] = str(ymin)
        row['xmax' + counter] = str(xmax)
        row['ymax' + counter] = str(ymax)
        counter = counter + 1
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
# Save the frame to an image file.
# PNG?
outputFilename =  '{0}_{1}.jpg'.format(strftime("%Y%m%d%H%M%S", gmtime()), uuid.uuid4())
cv.imwrite('/opt/demo/images/' + outputFilename, frame)
row['image_filename'] = outputFilename

json_string = json.dumps(row) 
print(json_string)
