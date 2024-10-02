#include <iostream>
#include <cmath>
#include <vector>
#include <random>

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

bool fermat_primality_testing(const unsigned int & input_value, const unsigned int & num_of_test) {

    // handle special case
    if (input_value <= 1 or input_value == 4) {
        return false;
    }
    if (input_value <= 3) {
        return true;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(2, input_value - 2);

    // runing prime testing
    for (unsigned int _ = 0; _ < num_of_test; _++) {

        int random_number = dis(gen);
        
        auto try_number = static_cast<unsigned int>(std::pow(random_number, input_value - 1));

        if (power_mod(random_number, input_value - 1, input_value) != 1)
            return false;
    }
    return true;
}


int main() {
    unsigned int input_value = 0;
    std::cin >> input_value;
    std::cout << input_value << (fermat_primality_testing(input_value, 100) ? " is a " : " is not a ") \
              << "prime number." << std::endl;
    return 0;
}