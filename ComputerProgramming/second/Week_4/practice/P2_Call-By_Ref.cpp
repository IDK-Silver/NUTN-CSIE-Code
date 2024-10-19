#include <iostream>

using namespace std;

template <class T> void get_num(T & i1, T & i2)
{
    cout << "Enter two number : ";
    cin >> i1 >> i2;
};

template <class T> void swap_value(T & i1, T & i2)
{
    T temp = i1;
    i1 = i2;
    i2 = temp;
}

template <class T> void show_result(const T & i1, const T & i2)
{
    cout << "In reverse order the number are : " \
        << i1 << " " << i2 << "\n";
}

int main()
{
    int f = 0, s = 0;
    get_num(f, s);
    swap_value(f, s);
    show_result(f, s);
}