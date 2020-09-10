
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iomanip>
#include <vector>
#include "Complex.hpp"

using namespace sycl;
using namespace std;

// Quantidade de números complexos passando para o código SYCL
static const int num_elements = 100;

class custom_device_selector : public device_selector {
public:
    custom_device_selector(std::string vendorName) : vendorName_(vendorName){};
    int operator()(const device& dev) const override {
    int rating = 0;    
    //Estamos pesquisando o dispositivo personalizado específico de um fornecedor e, 
    //se for um dispositivo GPU, estamos dando a classificação mais alta, 3. 
    //A segunda preferência é dada a qualquer dispositivo GPU e 
    //a terceira preferência é dada a dispositivo CPU.
    //**************Etapa 1: Remova o comentário das seguintes linhas, onde você está definindo a classificação dos dispositivos********
    /*if (dev.is_gpu() & (dev.get_info<info::device::name>().find(vendorName_) != std::string::npos))
        rating = 3;
    else if (dev.is_gpu()) rating = 2;
    else if (dev.is_cpu()) rating = 1; */
    return rating;
    };
    
private:
    std::string vendorName_;
};

// in_vect1 e in_vect2 são os vetores com num_elements, números complexos 
// e são entradas para a função paralela
void DpcppParallel(queue &q, std::vector<Complex2> &in_vect1,
                   std::vector<Complex2> &in_vect2, std::vector<Complex2> &out_vect) {
  // Configurar buffers de entrada
  buffer<Complex2, 1> bufin_vect1(in_vect1.data(), range<1>(num_elements));
  buffer<Complex2, 1> bufin_vect2(in_vect2.data(), range<1>(num_elements));

  // Configurar buffers de saída
  buffer<Complex2, 1> bufout_vect(out_vect.data(), range<1>(num_elements));

  std::cout << "Target Device: "
            << q.get_device().get_info<info::device::name>() <<"\n";
  // Enviar objeto de função do grupo Comando para a fila
  q.submit([&](handler &h) {
    // Acessores configurados em modo leitura
    auto V1 = bufin_vect1.get_access<access::mode::read>(h);
    auto V2 = bufin_vect2.get_access<access::mode::read>(h);
    // Acessor configurado em modo escrita
    //**************Etapa 2: Remova o comentário da linha abaixo para definir o acessor de escrita********************      
    //auto V3 = bufout_vect.get_access<access::mode::write>(h); 
      
    h.parallel_for(range<1>(num_elements), [=](id<1> i) {      
    //**************Etapa 3: Remova o comentário da linha abaixo para chamar a função complex_mul 
    //que calcula a multiplicação dos números complexos********************
        
    //V3[i] = V1[i].complex_mul(V2[i]); 
    });
  });
  q.wait_and_throw();
}
void DpcppScalar(std::vector<Complex2> &in_vect1, std::vector<Complex2> &in_vect2,
                 std::vector<Complex2> &out_vect) {
  for (int i = 0; i < num_elements; i++) {
    out_vect[i] = in_vect1[i].complex_mul(in_vect2[i]);
  }
}
//Compare os resultados dos dois vetores de saída de paralelo e escalar. Eles devem ser iguais
int Compare(std::vector<Complex2> &v1, std::vector<Complex2> &v2) {
  int ret_code = 1;
  for (int i = 0; i < num_elements; i++) {
    if (v1[i] != v2[i]) {
      ret_code = -1;
      break;
    }
  }
  return ret_code;
}
int main() {
  // Declare seus vetores de entrada e saída da classe Complex2
  vector<Complex2> input_vect1;
  vector<Complex2> input_vect2;
  vector<Complex2> out_vect_parallel;
  vector<Complex2> out_vect_scalar; 
  

  for (int i = 0; i < num_elements; i++) {
    input_vect1.push_back(Complex2(i + 2, i + 4));
    input_vect2.push_back(Complex2(i + 4, i + 6));
    out_vect_parallel.push_back(Complex2(0, 0));
    out_vect_scalar.push_back(Complex2(0, 0));
  }

  // este manipulador de exceção captura exceções assíncronas
  auto exception_handler = [&](cl::sycl::exception_list eList) {
    for (std::exception_ptr const &e : eList) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception const &e) {
        std::cout << "Failure" << std::endl;
        std::terminate();
      }
    }
  };

  // Inicialize seus vetores de entrada e saída. As entradas são inicializadas conforme abaixo.
  // As saídas são inicializadas com 0
  try {
    //Passe o nome do fornecedor para o qual o dispositivo que você deseja consultar
    std::string vendor_name = "Intel";
    //std::string vendor_name = "AMD";
    //std::string vendor_name = "Nvidia";
    // queue constructor passed exception handler
    custom_device_selector selector(vendor_name);
    queue q(selector, exception_handler);     
    // Chame DpcppParallel com as entradas e saídas necessárias
    DpcppParallel(q, input_vect1, input_vect2, out_vect_parallel);
  } catch (...) {
    // alguma outra exceção detectada
    std::cout << "Failure" << std::endl;
    std::terminate();
  }

  cout << "****************************************Multiplying Complex numbers "
          "in Parallel********************************************************"
       << std::endl;
  // Imprima as saídas da função Paralela
  for (int i = 0; i < num_elements; i++) {
    cout << out_vect_parallel[i] << ' ';
    if (i == num_elements - 1) {
      cout << "\n\n";
    }
  }
  cout << "****************************************Multiplying Complex numbers "
          "in Serial***********************************************************"
       << std::endl;
  // Chame a função DpcppScalar com as entradas e saídas necessárias
  DpcppScalar(input_vect1, input_vect2, out_vect_scalar);
  for (auto it = out_vect_scalar.begin(); it != out_vect_scalar.end(); it++) {
    cout << *it << ' ';
    if (it == out_vect_scalar.end() - 1) {
      cout << "\n\n";
    }
  }

  // Compare as saídas das funções paralela e escalar. Eles deveriam ser iguais
  int ret_code = Compare(out_vect_parallel, out_vect_scalar);
  if (ret_code == 1) {
    cout << "********************************************Success. Results are "
            "matched******************************"
         << "\n";
  } else
    cout << "*********************************************Failed. Results are "
            "not matched**************************"
         << "\n";

  return 0;
}
