#include <iostream>
#include <string>
using namespace std;

double unit_price(int  d, int p);

int main()
{
    double d = 0, p = 0;
    char ans;

    do 
    {
        cout << "Enter d and p \n";
        cin >> d >> p;
        cout << "unit price is $" << unit_price(d, p) << "\n";

        cout << "Test again? (y/n)";
        cin >> ans;
    } while (toupper(ans) == 'Y');
    return 0;
}

double unit_price(int  d, int p)
{
    const double PI = 3.14;
    double radius = d / static_cast<double>(2);
    double area = PI * radius * radius;
    return p / area;
}