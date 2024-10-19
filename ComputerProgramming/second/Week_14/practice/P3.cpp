#include <iostream>

using namespace std;


void print_v(int n)
{
    int r = n % 10;
    int q = n / 10;

    if (q == 0)
    {
        cout << r << "\n";
        return;
    }

    print_v(q);
    cout << r << "\n";
    return;
}

int main()
{
    print_v(456);
    return 0;
}
