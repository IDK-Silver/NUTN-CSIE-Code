#include <iostream>

using namespace std;

int main()
{
    const int arr_size = 10;

    int *arr_a = new int[arr_size];

    for (int i = 0; i < arr_size; i++)
    {
        arr_a[i] = i;
    }

    int *p = arr_a;

    for (int i = 0; i < arr_size; i++)
    {
        p[i] += i;
    }


    for (int i = 0; i < arr_size; i++)
    {
        cout << arr_a[i] << " ";
    }
    cout << "\n";
    return 0;
}