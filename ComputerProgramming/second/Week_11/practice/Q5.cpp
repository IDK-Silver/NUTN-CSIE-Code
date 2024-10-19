#include <iostream>
#include <string>

using namespace std;

int* doubler(const int *arr, int size)
{
    int *m_arr = new int [size];

    for (int i = 0; i < size; i++)
    {
        m_arr[i] = arr[i] * 2;
    }
    return m_arr;
}

void print_arr(const string & name, const int *arr, const int & size) 
{
    cout << name << " : { ";
    for (int i = 0; i < size; i++)
    {
        cout << arr[i];
        if (i == size - 1)
            cout  << " }\n";
        else
            cout << ", ";
    }
    return;
}

int main()
{
    const int size = 5;
    const int org[size] = {1, 2, 3, 4, 5};

    int* m_arr = doubler(org, size);

    print_arr("A", org, size);
    print_arr("B", m_arr, size);
    delete [] m_arr;
    return 0;
}