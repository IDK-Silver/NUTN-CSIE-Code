#ifndef PRIME_HPP
#define PRIME_HPP

#include <big_int.hpp>
#include <cmath>
#include <vector>
#include <random>
#include <limits>

static std::random_device __prime_rd;
static std::mt19937 __prime_gen(__prime_rd());

namespace prime {
    bool fermat_primality_testing(const BigInt & input_value, const int & num_of_test);
    bool miller_rabin_primality_testing(const BigInt & input_value, const int& num_of_test);
    bool sieve_of_eratosthenes(const BigInt& input_value);
    bool basic_prime_test(const BigInt& input_value);
    BigInt power_mod(const BigInt& base, const BigInt& exp, const BigInt& mod);
}

#endif