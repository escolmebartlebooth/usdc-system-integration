#!/bin/sh

# This script file automates all the steps required to run on the simulator
# this allows for easy reproduction when the weights file changes.
# -N indicates, don't download unless there is a new file available.
wget -N https://s3.amazonaws.com/sdc3-capstone/30k/frozen_inference_graph.pb
pip install -r requirements.txt
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch