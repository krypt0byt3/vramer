#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int a = 0;
    int b = 0;

    cudaDeviceCanAccessPeer(&a, 0, 1);
    cudaDeviceCanAccessPeer(&b, 1, 0);

    std::cout << "GPU0 -> GPU1: " << a << "\n";
    std::cout << "GPU1 -> GPU0: " << b << "\n";
}
