#!/bin/bash
#source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
echo "##" $(whoami) is working oneAPI_Toolkits_with_FPGAs Bonus_FPGA lab step $(pwd)
dpcpp -fintelfpga hough_transform.cpp -DFPGA -o hough_transform.fpga -Xshardware
echo "########## Done with the compile"
