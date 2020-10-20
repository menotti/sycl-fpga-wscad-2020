#!/bin/bash
#source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
echo "##" $(whoami) is working oneAPI_Toolkits_with_FPGAs Bonus_FPGA lab step $(pwd)
./hough_transform.fpga 
echo "########## Done with the run"
