#include <iostream>

using namespace std;

int main()
{
    cout << "Input number : ";
    int n = 0;
    cin >> n;


    int ans = 1;

    while (n > 1)
    {
        ans *= n--;
    };

    cout << "ans : " << ans << "\n";


    return 0;
}
