#include <iostream>

using namespace std;

void print_v(int n)
{
    int ns_tens = 1;
    int left = n;

    // �o�X���X���
    while (left >= 10)
    {
        left /= 10;
        ns_tens *= 10;
    }

    for (int p = ns_tens; p > 0; p /= 10)
    {
        // n �� p => �̥��䪺�Ʀr
        cout << (n / p) << endl;

        // ��̥���Ʀr�h�� & p /= 10 �h�i��U�@��
        n %= p;
    }
}


int main()
{

    print_v(123);
}
