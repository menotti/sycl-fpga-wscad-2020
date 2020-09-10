//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace sycl;

int main() {
    constexpr int N = 16;
    auto R = range<1>(N);
    std::vector<double> v(N, 10);
    queue q;
    // O buffer assume a propriedade dos dados armazenados no vetor
    buffer<double, 1> buf(v.data(), R);
    q.submit([&](handler& h) {
        auto a = buf.get_access<access::mode::read_write>(h);
        h.parallel_for(R, [=](id<1> i) { a[i] -= 2; });
    });
    // A criação do acessor de host é uma chamada blocante e só retornará depois que 
    // todos os kernels syCL enfileirados que modificam o mesmo buffer em qualquer fila 
    // concluírem a execução e os dados estiverem disponíveis para o host.
    auto b = buf.get_access<access::mode::read>();
    for (int i = 0; i < N; i++) 
        std::cout << v[i] << "\n";
    return 0;
}
