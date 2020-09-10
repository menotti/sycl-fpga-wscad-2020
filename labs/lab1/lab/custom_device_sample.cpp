//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;
class my_device_selector : public device_selector {
public:
    my_device_selector(std::string vendorName) : vendorName_(vendorName){};
    int operator()(const device& dev) const override {
    int rating = 0;
    //Estamos pesquisando o dispositivo personalizado específico de um fornecedor e, 
    //se for um dispositivo GPU, estamos dando a classificação mais alta, 3. 
    //A segunda preferência é dada a qualquer dispositivo GPU 
    //e a terceira preferência é dada a dispositivo CPU.
    if (dev.is_gpu() & (dev.get_info<info::device::name>().find(vendorName_) != std::string::npos))
        rating = 3;
    else if (dev.is_gpu()) rating = 2;
    else if (dev.is_cpu()) rating = 1;
    return rating;
    };
    
private:
    std::string vendorName_;
};
int main() {
    //passe o nome do fornecedor para o qual o dispositivo que você deseja consultar
    std::string vendor_name = "Intel";
    //std::string vendor_name = "AMD";
    //std::string vendor_name = "Nvidia";
    my_device_selector selector(vendor_name);
    queue q(selector);
    std::cout << "Device: "
    << q.get_device().get_info<info::device::name>() << std::endl;
    return 0;
}
