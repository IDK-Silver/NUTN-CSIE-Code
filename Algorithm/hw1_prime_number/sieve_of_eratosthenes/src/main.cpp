#include <iostream>
#include <cmath>
#include <vector>

bool sieve_of_eratosthenes(const unsigned int & input_value) {
    if (input_value <= 1) return false;
    if (input_value == 2) return true;

    std::vector<bool> pass_list(input_value + 1, false);
    const auto sqrt_input_value = static_cast<unsigned int>(sqrt(input_value));

    for (auto try_num = 2; try_num <= sqrt_input_value; try_num++) {
        if (not pass_list[try_num]) {
            for (auto scale_num = try_num; scale_num * try_num <= input_value; scale_num++) {
                pass_list[try_num * scale_num] = true;
            }
        }
    }
    return !pass_list[input_value];
}


int main() {
    unsigned int input_value = 0;
    std::cin >> input_value;
    std::cout << input_value << (sieve_of_eratosthenes(input_value) ? " is a " : " is not a ") \
              << "prime number." << std::endl;
    return 0;
}