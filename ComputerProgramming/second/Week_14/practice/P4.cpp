#include <iostream>

using namespace std;

void print_v(int n)
{
    int ns_tens = 1;
    int left = n;

    // 眔Τ碭计
    while (left >= 10)
    {
        left /= 10;
        ns_tens *= 10;
    }

    for (int p = ns_tens; p > 0; p /= 10)
    {
        // n 埃 p => 程オ娩计
        cout << (n / p) << endl;

        // р程オ娩计奔 & p /= 10 秈︽近
        n %= p;
    }
}


int main()
{

    print_v(123);
}
