#include <iostream>
#include <cmath>
#include <stdexcept>
using namespace std;

bool is_prime(int num)
{
    if (num <= 1)
        return false;
    
    for (int i = 2; i <= sqrt(num); i++)
    {
        if (num % i == 0)
            return false;
    }
    return true;
}

double Division(int divisor, int dividend)
{
    
    if (dividend == 0)
        throw runtime_error("Error: Attempted to divide by zero.");
   
    else if (is_prime(divisor))
        throw runtime_error("Error: Divisor is prime.");

    else if (divisor % dividend)
    {
        throw runtime_error("Error: Inaccurate result.");
    }

    return divisor / dividend;
}

int main()
{
    int divisor = 0, dividend = 0;
    while (cin >> divisor >> dividend)
    {
        try
        {
            int ans = Division(divisor, dividend);
            cout << divisor << " / " << dividend << " = " << ans << "\n";
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        
    }
}