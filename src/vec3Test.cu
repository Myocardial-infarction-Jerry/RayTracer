#include <iostream>
#include <cstdlib>
#include <ctime>

#include "Vec3.cuh"

int main(int argc, char const *argv[]) {
    srand(time(NULL));
    Vec3 a(rand(), rand(), rand());
    Vec3 b(rand(), rand(), rand());
    Vec3 c = a + b;
    std::cerr << a << " + " << b << " = " << c << "\n";

    std::cout << "Press any key to continue...";
    std::cin.ignore();

    return 0;
}
