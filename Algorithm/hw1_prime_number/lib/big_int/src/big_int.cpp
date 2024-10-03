#include "big_int.hpp"
#include <algorithm>


BigInt::BigInt() {
    this->number = "";
    this->is_negative = false;
}

BigInt::BigInt(const std::string & num) {
    
    if (is_bigint(num))
    {
        if (num.at(0) == '-') {
            this->number = num.substr(1);
            this->is_negative = true;
        }
        else {
            this->number = num;
            this->is_negative = false;
        }
    }
    else
    {
        throw std::invalid_argument("Number is not valid");
    }
}

BigInt::BigInt(const long long & num) {
    this->number = std::to_string(num);
    this->is_negative = num < 0;
}

// Checks if the current number is valid Number or not.
bool BigInt::is_bigint() {
    return BigInt::is_bigint(
        this->is_negative ? "-"  + this->number : this->number
    );
}

BigInt & BigInt::operator++() {
    *this = *this + 1;
    return *this;
}

BigInt BigInt::operator++(int) {
    BigInt temp = *this;
    ++(*this);
    return temp;
}

BigInt & BigInt::operator--() {
    *this = *this - 1;
    return *this;
}

BigInt BigInt::operator--(int) {
    BigInt temp = *this;
    --(*this);
    return temp;
}

// Checks if the feeded integer is valid Number or not.
bool BigInt::is_bigint(const std::string & input_string) {
    // If the number is negative, we need to skip the first character.
    std::size_t start_index = 0;
    if (input_string.at(0) == '-') {
        start_index = 1;
    }

    // Check if all characters in the string are digits.
    for (std::size_t i = start_index; i < input_string.length(); ++i) {
        if (!isdigit(input_string.at(i))) {
            return false;
        }
    }

    // If all characters are digits, return true.
    return true;
}

// 重載 < 運算符
bool BigInt::operator<(const BigInt& other) const {
    // 處理正負號
    if (this->is_negative && !other.is_negative) {
        return true;
    }
    if (!this->is_negative && other.is_negative) {
        return false;
    }

    // 兩數同號時
    bool bothNegative = this->is_negative && other.is_negative;
    if (this->number.length() != other.number.length()) {
        return bothNegative ? 
               (this->number.length() > other.number.length()) : 
               (this->number.length() < other.number.length());
    }

    // 長度相同時，逐位比較
    for (size_t i = 0; i < this->number.length(); ++i) {
        if (this->number[i] != other.number[i]) {
            return bothNegative ? 
                   (this->number[i] > other.number[i]) : 
                   (this->number[i] < other.number[i]);
        }
    }

    // 完全相等
    return false;
}

// 重載 > 運算符
bool BigInt::operator>(const BigInt& other) const {
    return other < *this;
}

// 重載 >= 運算符
bool BigInt::operator>=(const BigInt& other) const {
    return !(*this < other);
}

// 重載 <= 運算符
bool BigInt::operator<=(const BigInt& other) const {
    return !(other < *this);
}

bool BigInt::operator==(const BigInt& other) const {
    return this->number == other.number && this->is_negative == other.is_negative;
}

bool BigInt::operator!=(const BigInt& other) const {
    return !(*this == other);
}

bool BigInt::add(const BigInt & other) {
    // 如果兩個數字符號相同，直接相加
    if (this->is_negative == other.is_negative) {
        std::string num1 = this->number;
        std::string num2 = other.number;
        this->number = BigInt::simplify_add(num1, num2);
        this->is_negative = this->is_negative;
        return true;
    }
    else {
        bool sign_from_this = true;
        BigInt larger = BigInt(this->number);
        BigInt smaller = BigInt(other.number);
        if (larger < smaller) {
            std::swap(larger, smaller);
            sign_from_this = false;
        }
        this->number = BigInt::simplify_subtract(larger.number, smaller.number);
        if (sign_from_this) {
            this->is_negative = larger.is_negative;
        }
        else {
            this->is_negative = !larger.is_negative;
        }
        return true;
    }
}

bool BigInt::subtract(const BigInt & other) {
    BigInt sub_object(other);
    sub_object.is_negative = !sub_object.is_negative;
    this->add(sub_object);
    return true;
}



std::ostream& operator<<(std::ostream& os, const BigInt& bigint) {
    os << bigint.to_string();
    return os;
}

std::string BigInt::to_string() const {
    if (this->is_negative) {
        return "-" + this->number;
    }
    return this->number;
}

std::string BigInt::simplify_add(const std::string& num1, const std::string& num2) {
    std::string result;
    int carry = 0;
    int i = num1.length() - 1;
    int j = num2.length() - 1;

    while (i >= 0 || j >= 0 || carry > 0) {
        int sum = carry;
        if (i >= 0) sum += num1[i] - '0';
        if (j >= 0) sum += num2[j] - '0';

        result.push_back(sum % 10 + '0');
        carry = sum / 10;

        i--;
        j--;
    }

    std::reverse(result.begin(), result.end());
    return result;
}

BigInt BigInt::sqrt() const {
    if (*this < 0) {
        throw std::invalid_argument("Cannot compute the square root of a negative number.");
    }

    BigInt low = 0;
    BigInt high = *this;
    BigInt mid;
    BigInt result;

    while (low <= high) {
        mid = (low + high) / 2;
        BigInt mid_squared = mid * mid;

        if (mid_squared == *this) {
            return mid;
        } else if (mid_squared < *this) {
            low = mid + 1;
            result = mid;
        } else {
            high = mid - 1;
        }
    }

    return result;
}

std::string BigInt::simplify_subtract(const std::string& num1, const std::string& num2) {
    std::string result;
    int borrow = 0;
    int i = num1.length() - 1;
    int j = num2.length() - 1;

    while (i >= 0 || j >= 0) {
        int diff = borrow;
        if (i >= 0) diff += num1[i] - '0';
        if (j >= 0) diff -= num2[j] - '0';

        if (diff < 0) {
            diff += 10;
            borrow = -1;
        } else {
            borrow = 0;
        }

        result.push_back(diff + '0');

        i--;
        j--;
    }

    // 移除前導零
    while (result.length() > 1 && result.back() == '0') {
        result.pop_back();
    }

    std::reverse(result.begin(), result.end());
    return result;
}

BigInt BigInt::operator+(const BigInt& other) const {
    BigInt result(*this);
    result.add(other);
    return result;
}

BigInt BigInt::operator-(const BigInt& other) const {
    BigInt result(*this);
    result.subtract(other);
    return result;
}

BigInt BigInt::operator*(const BigInt& other) const {
    // 確定結果的符號
    bool result_negative = this->is_negative ^ other.is_negative;

    std::string num1 = this->number;
    std::string num2 = other.number;

    // 反轉數字以便從最低位開始相乘
    std::reverse(num1.begin(), num1.end());
    std::reverse(num2.begin(), num2.end());

    std::vector<int> result(num1.size() + num2.size(), 0);

    // 執行乘法運算
    for (size_t i = 0; i < num1.size(); ++i) {
        int digit1 = num1[i] - '0';
        for (size_t j = 0; j < num2.size(); ++j) {
            int digit2 = num2[j] - '0';
            result[i + j] += digit1 * digit2;
            // 處理進位
            result[i + j + 1] += result[i + j] / 10;
            result[i + j] %= 10;
        }
    }

    // 移除前導零
    while (result.size() > 1 && result.back() == 0) {
        result.pop_back();
    }

    // 將結果轉換回字符串
    std::string product = "";
    for (auto it = result.rbegin(); it != result.rend(); ++it) {
        product += std::to_string(*it);
    }

    BigInt result_bigint;
    result_bigint.number = product;
    result_bigint.is_negative = result_negative;

    return result_bigint;
}


BigInt BigInt::operator/(const BigInt& other) const {
    if (other.number == "0") {
        throw std::invalid_argument("除數不能為零");
    }

    // 確定結果的符號
    bool result_negative = this->is_negative ^ other.is_negative;

    // 拷貝絕對值進行運算
    std::string dividend = this->number;
    std::string divisor = other.number;

    // 如果被除數小於除數，結果為0
    if (BigInt(dividend) < BigInt(divisor)) {
        return BigInt("0");
    }

    std::string quotient = "";
    std::string remainder = "";

    size_t n = dividend.length();
    // size_t m = divisor.length();

    std::string current = "";
    size_t i = 0;

    while (i < n) {
        current += dividend[i];
        // 移除前導零
        size_t start = current.find_first_not_of('0');
        if (start != std::string::npos) {
            current = current.substr(start);
        } else {
            current = "0";
        }

        BigInt current_bigint(current);
        BigInt divisor_bigint(divisor);

        int count = 0;
        while (current_bigint >= divisor_bigint) {
            current_bigint = current_bigint - divisor_bigint;
            count++;
        }

        quotient += std::to_string(count);
        current = current_bigint.number;
        i++;
    }

    // 移除前導零
    size_t pos = quotient.find_first_not_of('0');
    if (pos != std::string::npos) {
        quotient = quotient.substr(pos);
    } else {
        quotient = "0";
    }

    BigInt result_bigint;
    result_bigint.number = quotient;
    result_bigint.is_negative = result_negative;

    return result_bigint;
}


BigInt BigInt::operator%(const BigInt& other) const {
    if (other.number == "0") {
        throw std::invalid_argument("除數不能為零");
    }

    // 計算商
    BigInt quotient = (*this) / other;
    // 計算商與除數的乘積
    BigInt product = quotient * other;
    // 計算餘數
    BigInt remainder = (*this) - product;

    // 根據C++的取餘數定義，餘數的符號與被除數相同
    if (this->is_negative && remainder.number != "0") {
        remainder.is_negative = true;
    } else {
        remainder.is_negative = false;
    }

    return remainder;
}