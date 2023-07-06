#include <iostream>
#include <random>
#include "top.hpp"


int main()
{
    std::cout << std::endl;
    std::cout << "********************************************" << std::endl;
    std::cout << "TESTING: ops::hls::stencil2dCore impl" << std::endl;
    std::cout << "********************************************" << std::endl << std::endl;

    dut();

    return 0;
}
