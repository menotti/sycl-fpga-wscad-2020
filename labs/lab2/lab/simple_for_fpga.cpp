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
    
  //# Cria uma matriz de 100 números incrementais
  //# A soma deve ser 5050
  int summands[100];
  for (int i=0;i<100;i++) summands[i]=i+1;
    
  //# Cria uma variável para guardar a soma
  int sum = 0;
    
  //# Um flag -D definirá qual dispositivo escolheremos
  #if defined(FPGA_EMULATOR)
    intel::fpga_emulator_selector device_selector;
  #else
    intel::fpga_selector device_selector;
  #endif

  //# Buffers são usados ​​para compartilhar dados entre o host e o FPGA
  buffer<int, 1> buffer_summands(summands, 100);
  buffer<int, 1> buffer_sum(&sum, 1);

  //# define a fila que tem dispositivo padrão associado para descarregamento
  //# A fila é usada pelo host para iniciar o código no FPGA
  queue q(device_selector);
    
  //# Envia os valores para o FPGA ou o emulador FPGA para calcular a soma
  //# Você pode pensar no manipulador como um intermediário para tudo 
  //# o que precisa acontecer entre o host e o FPGA
  q.submit([&](handler &h) {
    //# O FPGA precisa ter acesso aos buffers configurados anteriormente
    //# O acesso é definido em termos de acesso do lado do FPGA
    auto acc_summands = buffer_summands.get_access<access::mode::read>(h);
    auto acc_sum = buffer_sum.get_access<access::mode::write>(h);
      
    //# Este é o código que é executado no FPGA
    //# Isso geralmente é referido como um kernel
    //# Se você quisesse tornar simple_sum uma função, você poderia,
    //# e os Tutoriais FPGA são escritos desta maneira
    h.single_task<class simple_sum>([=]() {
      //# Kernel para adicionar coisas usando FPGA ou emulador FPGA
      //# O código aqui se torna hardware
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
