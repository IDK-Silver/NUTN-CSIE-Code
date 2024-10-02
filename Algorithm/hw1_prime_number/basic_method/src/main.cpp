#include <iostream>
#include <cmath>
#include <vector>
#include <random>

std::random_device rd;
std::mt19937 gen(rd());

unsigned long long power_mod(unsigned long long base, unsigned long long exp, unsigned long long mod) {
    unsigned long long result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1)
            result = (result * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return result;
}

bool fermat_primality_testing(const int & input_value, const int & num_of_test) {

    // handle special case
    if (input_value <= 1 or input_value == 4) {
        return false;
    }
    if (input_value <= 3) {
        return true;
    }


    std::uniform_int_distribution<> dis(2, input_value - 2);

    // running prime testing
    for (unsigned int _ = 0; _ < num_of_test; _++) {

        int random_number = dis(gen);

        if (power_mod(random_number, input_value - 1, input_value) != 1)
            return false;
    }
    return true;
}


int main() {
    int input_size = 0;
    int input_number = 0;
    std::cin >> input_size;
    for (int i = 0; i < input_size; i++) {
        std::cin >> input_number;
        if (fermat_primality_testing(input_number, 5))
            std::cout << "Prime" << std::endl;
        else
            std::cout << "Composite" << std::endl;
    }

    return 0;
}