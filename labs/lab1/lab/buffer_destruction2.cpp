//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
constexpr int N = 16;
using namespace sycl;

// A criação do buffer ocorre dentro de um escopo de função separado.
void dpcpp_code(std::vector<double> &v, queue &q) {
    auto R = range<1>(N);
    buffer<double, 1> buf(v.data(), R);
    q.submit([&](handler &h) {
    auto a = buf.get_access<access::mode::read_write>(h);
        h.parallel_for(R, [=](id<1> i) { a[i] -= 2; });
    });
}
int main() {
    std::vector<double> v(N, 10);
    queue q;
    dpcpp_code(v, q);
    // Quando a execução avança além desse escopo de função, 
    // o destruidor de buffer é invocado, cedendo a propriedade
    // dos dados e os copia de volta para a memória do host.
    for (int i = 0; i < N; i++) 
        std::cout << v[i] << "\n";
    return 0;
}
