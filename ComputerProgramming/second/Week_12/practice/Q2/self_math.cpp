#include "self_math.h"

int self_pow(int a, int b)
{
    int sum = a;

    while (b > 1)
    {
        sum *= a;
        b--;
    }
        
        
    return sum;
}