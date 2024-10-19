#include <iostream>

using namespace std;


int power(int n, int p)
{
    if (p < 0)
    {
        cerr << "Error";
        return -1;
    }

    if (p == 1)
    {
        return n;
    }

    return n * power(n, p - 1);
}


int main()
{

    cout << power(2, 10);
    return 0;
}
