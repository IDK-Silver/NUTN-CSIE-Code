#ifndef BIG_INT_HPP
#define BIG_INT_HPP

#include <vector>
#include <iostream>
#include <string>

class BigInt {
public:
    // Default constructor.
    BigInt();
    // Constructor that takes a string as an argument.
    BigInt(const std::string & num);
    BigInt(const long long & num);

    // Checks if the current number is valid Number or not.
    bool is_bigint();
    // Checks if the feeded integer is valid Number or not.
    static bool is_bigint(const std::string & input_string);

    // Adds two BigInt objects.
    bool add(const BigInt & other);
    // Subtracts two BigInt objects.
    bool subtract(const BigInt & other);

    // 重載 + 運算符
    BigInt operator+(const BigInt& other) const;

    // 重載 - 運算符
    BigInt operator-(const BigInt& other) const;

    // 重載 * 運算符
    BigInt operator*(const BigInt& other) const;

    // 重載 / 運算符
    BigInt operator/(const BigInt& other) const;
    BigInt operator%(const BigInt& other) const;

    // 重載 < 運算符
    bool operator<(const BigInt& other) const;

    // 重載 > 運算符
    bool operator>(const BigInt& other) const;

    // 重載 >= 運算符
    bool operator>=(const BigInt& other) const;

    // 重載 <= 運算符
    bool operator<=(const BigInt& other) const;
    bool operator==(const BigInt& other) const;

    // 重載 << 運算符
    friend std::ostream& operator<<(std::ostream& os, const BigInt& bigint);


private:
    std::string number;
    bool is_negative;

    // 只能夠用於兩個正整數的加法
    static std::string simplify_add(const std::string & num1, const std::string & num2);
    
    // 只能夠用於 num1 >= num2 的減法, 且 num1 和 num2 都是正整數
    static std::string simplify_subtract(const std::string & num1, const std::string & num2);




};

#endif // BIG_INT_HPP