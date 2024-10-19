#include <iostream>

void bubble_sort(int* arr, int len)
{
    for (int i = 0; i < len; i++)
    {
        for (int j = i + 1; j < len; j++)
        {
            if (arr[i] > arr[j])
            {
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }

}
void fill_up(int *arr, int size)
{
    std::cout << "Enter " << size << " num : ";

    for (int i = 0; i < size; i++)
        std::cin >> arr[i];

    std::cout << "Last index element is " << arr[size - 1] << ".\n";
}
int main()
{
    int arr[5] = {0};
    fill_up(arr, 5);
    bubble_sort(arr, 5);

    for (int i = 0; i < 5; i++)
        std::cout << arr[i] << " ";
    
    std::cout << "\n";
    return 0;
}