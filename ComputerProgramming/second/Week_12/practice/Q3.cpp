#include<math.h>
#include<iostream>

using namespace std;

namespace my_std {

    double pow(double a, double b)
    {
        double sum = a;

        while (b > 1)
        {
            sum *= a;
            b--;
        }
        return sum;
    }
}

int main()
{
    cout << my_std::pow(2, 31);
}