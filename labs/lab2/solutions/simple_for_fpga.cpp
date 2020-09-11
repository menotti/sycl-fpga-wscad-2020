//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
using namespace sycl;
static const int N = 100;
int main(){
    
  //# Create an array of 100 incrementing numbers
  //# The sum should be 5050
  int summands[100];
  for (int i=0;i<100;i++) summands[i]=i+1;
    
  //# Create a variable to hold the sum
  int sum = 0;
    
  //# A -D switch will define which device we choose
  #if defined(FPGA_EMULATOR)
    intel::fpga_emulator_selector device_selector;
  #else
    intel::fpga_selector device_selector;
  #endif

  //# Buffers are used to share data between the host and the FPGA
  buffer<int, 1> buffer_summands(summands, 100);
  buffer<int, 1> buffer_sum(&sum, 1);

  //# define queue which has default device associated for offload
  //# The queue is used by the host to kick off code on the FPGA
  queue q(device_selector);
    
  //# Send the values to the FPGA or the FPGA emulator to calculate the sum
  //# You can think of the handler as a proxy for everything behind the scenes
  //#   that needs to happen between the host and the FPGA
  q.submit([&](handler &h) {
    //# The FPGA needs to have access to the buffers set up earlier
    //# The access is defined in terms of the FPGA's access
    auto acc_summands = buffer_summands.get_access<access::mode::read>(h);
    auto acc_sum = buffer_sum.get_access<access::mode::write>(h);
      
    //# This the code that gets executed on the FPGA
    //# This is often referred to as a kernel
    //# If you wanted to make simple_sum a function, you could,
    //#   and the FPGA Tutorials are written in this manner
    h.single_task<class simple_sum>([=]() {
      //# Kernel to add things up using FPGA or FPGA emulator
      //# Code inside here becomes hardware
      int kernel_sum = 0;
      for (int i=0;i<100;i++) kernel_sum = kernel_sum + acc_summands[i];
      acc_sum[0] = kernel_sum;
    });
  }).wait();

  //# Print Output
  std::cout << "The calculation is finished. The sum is ";
  std::cout << sum;
  std::cout << "." << std::endl;

  return 0;
}
