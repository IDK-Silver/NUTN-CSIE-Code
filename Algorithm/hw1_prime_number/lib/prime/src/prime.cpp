#include "prime.hpp"

namespace prime {


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

        BigInt upper_bound = std::min(input_value - 2, BigInt(__LONG_LONG_MAX__));
        std::uniform_int_distribution<long long> dis(2, std::stoll(upper_bound.to_string()));

        for (int _ = 0; _ < num_of_test; _++) {
            BigInt A(std::to_string(dis(__prime_gen)));
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

    
    bool fermat_primality_testing(const BigInt & input_value, const int & num_of_test) {

        // 處理特殊情況
        if (input_value <= BigInt("1") || input_value == BigInt("4")) {
            return false;
        }
        if (input_value <= BigInt("3")) {
            return true;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        BigInt upper_bound = std::min(input_value - 2, BigInt(__LONG_LONG_MAX__));
        std::uniform_int_distribution<long long> dis(2, std::stoll(upper_bound.to_string()));

        // 執行質數測試
        for (int _ = 0; _ < num_of_test; _++) {

            BigInt random_number(std::to_string(dis(gen)));
            
            // 使用 BigInt 進行計算
            if (power_mod(random_number, input_value - BigInt("1"), input_value) != BigInt("1"))
                return false;
        }
        return true;
    }

    
    bool sieve_of_eratosthenes(const BigInt & input_value) {
        if (input_value <= BigInt("1")) return false;
        if (input_value == BigInt("2")) return true;

        if (input_value > __LONG_LONG_MAX__) 
            throw std::runtime_error("input_value is too large for sieve_of_eratosthenes");

        std::vector<bool> pass_list(std::stoull(input_value.to_string()) + 1, false);
        const auto sqrt_input_value = static_cast<unsigned long long>(sqrt(std::stoull(input_value.to_string())));

        for (auto try_num = 2; try_num <= sqrt_input_value; try_num++) {
            if (!pass_list[try_num]) {
                for (auto scale_num = try_num; scale_num * try_num <= std::stoull(input_value.to_string()); scale_num++) {
                    pass_list[try_num * scale_num] = true;
                }
            }
        }
        return !pass_list[std::stoull(input_value.to_string())];
    }


    
    bool basic_prime_test(const BigInt& input_value) {
        // 處理特殊情況
        if (input_value <= BigInt("1")) {
            return false;
        }
        if (input_value <= BigInt("3")) {
            return true;
        }
        if (input_value % BigInt("2") == BigInt("0")) {
            return false;
        }

        // 從3開始，每次加2，檢查到sqrt(input_value)
        BigInt i("3");
        BigInt sqrt_input = input_value.sqrt();
        while (i <= sqrt_input) {
            if (input_value % i == BigInt("0")) {
                return false;
            }
            i.add(2);
        }
        return true;
    }

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

}