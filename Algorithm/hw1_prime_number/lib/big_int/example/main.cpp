#include <big_int.hpp>
#include <iostream>
using namespace std;
int main() {
    BigInt num1("21474836480");
    BigInt num2("2147483648");

    cout << num1  + num2 << endl;
    cout << num1  - num2 << endl;
    cout << num1 * num2 << endl;
    cout << num1 / num2 << endl;
    cout << num1 % num2 << endl;
    return 0;
}