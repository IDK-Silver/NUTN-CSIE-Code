#include <iostream>
#include <string>
using namespace std;
int input(const string & str) {
    cout << str;
    int temp;
    cin >> temp;
    return temp;
}

void fill_array(int *arr, int size)
{
    cout << "input " << size << "'s element : \n";
    for (int i = 0; i < size; i++)
    {
        string display = "\tinput index " + std::to_string(i) + "'s element : ";
        arr[i] = input(display);
    }
    cout << "array is {";
    for (int i = 0; i < size; i++)
    {
        cout << arr[i];
        if (i == size - 1)
            cout  << "}\n";
        else
            cout << ", ";
    }
    return;
}

int search(int *arr, int size, int target)
{
    for (int i = 0; i < size; i++)
        if (arr[i] == target)
            return i;
    return -1;
}

int main()
{
    cout << "This program search a list of numbers\n";
    int arr_size = input("How many numbers will be on the list?\n");
    int* d_arr = new int [arr_size];
    fill_array(d_arr, arr_size);
    int target = input("Enter a value to search for : ");
    int local = search(d_arr, arr_size, target);
    if (local == -1)
        cout << target << " is not in the array\n";
    else
        cout << target << " is element " << local << " in the array\n";
    delete [] d_arr;
    return 0;
}