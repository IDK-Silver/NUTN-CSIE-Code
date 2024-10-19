
#include <iostream>

using namespace std;


int factorial(int n)
{
    if (n <= 1)
        return 1;

    return n * factorial(n - 1);
}

int main()
{
    cout << "Input number : ";
    int n = 0;
    cin >> n;
    cout << "ans : " << factorial(n) << "\n";
    return 0;
}
