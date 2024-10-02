#include <gtest/gtest.h>
#include <big_int.hpp>

TEST(BigIntTest, Addition) {
    BigInt a("12345678901234567890");
    BigInt b("98765432109876543210");
    BigInt result = a + b;
    BigInt expected("111111111011111111100");
    EXPECT_EQ(result, expected);
}

TEST(BigIntTest, Subtraction) {
    BigInt a("98765432109876543210");
    BigInt b("12345678901234567890");
    BigInt result = a - b;
    BigInt expected("86419753208641975320");
    EXPECT_EQ(result, expected);
}

TEST(BigIntTest, Multiplication) {
    BigInt a("12345678901234567890");
    BigInt b("98765432109876543210");
    BigInt result = a * b;
    BigInt expected("1219326311370217952237463801111263526900");
    EXPECT_EQ(result, expected);
}

TEST(BigIntTest, Division) {
    BigInt a("98765432109876543210");
    BigInt b("465465464654");
    BigInt result = a / b;
    BigInt expected("212186380");
    EXPECT_EQ(result, expected);
}

TEST(BigIntTest, Modulus) {
    BigInt a("98765432109876543210");
    BigInt b("465465464654");
    BigInt result = a % b;
    BigInt expected("149926330690");
    EXPECT_EQ(result, expected);
}

TEST(BigIntTest, Comparison) {
    BigInt a("12345678901234567890");
    BigInt b("98765432109876543210");
    EXPECT_LT(a, b);
}

TEST(BigIntTest, Assignment) {
    BigInt a("12345678901234567890");
    BigInt b("12345678901234567890");
    b = a;
    EXPECT_EQ(b, a);
}

TEST(BigIntTest, Output) {
    BigInt a("12345678901234567890");
    std::stringstream ss;
    ss << a;
    EXPECT_EQ(ss.str(), "12345678901234567890");
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
