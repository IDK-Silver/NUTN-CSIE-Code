#include <iostream>
#include <big_int>
#include <prime>
#include <iostream>
#include <vector>

#include "core/prime_runner.hpp"
#include "core/config.hpp"

// 函數來生成 2^p - 1
BigInt generate_mersenne_numer(BigInt exponent) {
    BigInt result = 1;
    for (BigInt i = 0; i < exponent; i++) {
        result = result * 2;
    }
    result = result - 1;
    return result;
}

int main() {

    core::BasicConfig config;
    config.load_config("../config.txt");
    std::cout << config.get_value("algorithm") << std::endl;

    // auto primes = core::find_primes_in_range("fermat", 0, 100);
    // for (auto prime : primes) {
    //     std::cout << prime << std::endl;
    // }
    
    return 0;
}