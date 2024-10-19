#include <iostream>


void fill_up(int *arr, int size)
{
    std::cout << "Enter " << size << " num : ";

    for (int i = 0; i < size; i++)
        std::cin >> arr[i];

    std::cout << "Last index element is " << arr[size - 1] << ".\n";
}


int main()
{
    int arr_1[5] = {0};
    int arr_2[10] = {0};

    fill_up(arr_1, 5);
    fill_up(arr_2, 10);

    return 0;
}