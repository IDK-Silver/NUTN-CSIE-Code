#ifndef PRIME_RUNNER_HPP
#define PRIME_RUNNER_HPP

#include <string>
#include <vector>
#include <prime>
#include <big_int>

using namespace prime;

namespace core {

std::vector<BigInt> find_primes_in_range(const std::string& algorithm, const BigInt& start, const BigInt& end, const int& try_times = 5) {
    std::vector<BigInt> primes;
    
    for (BigInt number = start; number <= end; number = number + 1) {
        bool isPrime = false;
        if (algorithm == "fermat") {
            isPrime = fermat_primality_testing(number, try_times);
        } 
        else if (algorithm == "miller-rabin") {
            isPrime = miller_rabin_primality_testing(number, try_times);
        } 
        else if (algorithm == "sieve") {
            isPrime = sieve_of_eratosthenes(number);
        } 
        else if (algorithm == "basic") {
            isPrime = basic_prime_test(number);
        } 
        else {
            throw std::invalid_argument("不支持的算法: " + algorithm);
        }
        
        if (isPrime) {
            primes.push_back(number);
        }
    }

    return primes;
}

// 新增的 is_prime 函數，用於判斷單個數字是否為質數
bool is_prime(const std::string& algorithm, const BigInt& number, const int& try_times = 5) {
    if (algorithm == "fermat") {
        return fermat_primality_testing(number, try_times);
    } 
    else if (algorithm == "miller-rabin") {
        return miller_rabin_primality_testing(number, try_times);
    } 
    else if (algorithm == "sieve") {
        return sieve_of_eratosthenes(number);
    } 
    else if (algorithm == "basic") {
        return basic_prime_test(number);
    } 
    else {
        throw std::invalid_argument("不支持的算法: " + algorithm);
    }
}

} // namespace prime

#endif // PRIME_RUNNER_HPP
