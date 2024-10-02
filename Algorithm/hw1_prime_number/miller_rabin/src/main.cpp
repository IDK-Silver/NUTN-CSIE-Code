#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <limits>
#include <big_int.hpp> // 引入 BigInt 類

std::random_device rd;
std::mt19937 gen(rd());

// 使用 BigInt 進行模指數運算
BigInt power_mod(const BigInt& base, const BigInt& exp, const BigInt& mod) {
    BigInt result = 1;
    BigInt base_mod = base % mod;
    BigInt exponent = exp;

    while (exponent > 0) {
        // 檢查 exponent 是否為奇數
        if ((exponent % 2) == 1) {
            result = (result * base_mod) % mod;
        }
        base_mod = (base_mod * base_mod) % mod;
        exponent = exponent / 2;
    }
    return result;
}

bool miller_rabin_primality_testing(const BigInt& input_value, const int& num_of_test) {

    if (input_value < 2) {
        return false;
    }
    if (input_value == 2 || input_value == 3) {
        return true;
    }
    if ((input_value % 2) == 0) {
        return false;
    }

    // 寫出 input_value - 1 為 2^R * D
    BigInt D = input_value - 1;
    int R = 0;
    while ((D % 2) == 0) {
        D = D / 2;
        R += 1;
    }

    std::uniform_int_distribution<long long> dis(2, __LONG_LONG_MAX__);

    for (int _ = 0; _ < num_of_test; _++) {
        BigInt A(std::to_string(dis(gen)));
        BigInt X = power_mod(A, D, input_value);

        if (X == BigInt("1") || X == (input_value - BigInt("1")))
            continue;

        bool cont_outer = false;
        for (int i = 1; i < R; i++) {
            X = power_mod(X, BigInt("2"), input_value);

            if (X == (input_value - BigInt("1"))) {
                cont_outer = true;
                break;
            }
            if (X == BigInt("1"))
                return false;
        }

        if (cont_outer)
            continue;

        return false;
    }
    return true;
}

int main() {
    std::string input_str;
    std::cin >> input_str;
    BigInt input_value(input_str);
    bool is_prime = miller_rabin_primality_testing(input_value, 5);
    std::cout << input_str << (is_prime ? " 是 " : " 不是 ") << "質數。" << std::endl;
    return 0;
}