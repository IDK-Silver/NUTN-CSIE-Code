#include <iostream>
#include <array>

int main()
{
    std::array<int, 5> arr = {10, 20, 30, 40, 50};
    // int arr[] = {10, 20, 30, 40, 50};

    for (const auto & num : arr)
        std::cout << num << "\n";

    return 0;
}