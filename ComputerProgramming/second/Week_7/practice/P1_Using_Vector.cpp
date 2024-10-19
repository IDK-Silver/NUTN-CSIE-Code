#include <iostream>
#include <vector>

using namespace std;

int main()
{
    vector<int> vec;

    int input_num = 0;

    cout << "Inpute positive number\n";

    while (cin >> input_num && input_num > 0)
    {
        vec.push_back(input_num);
        cout << "Your input number is : " << input_num << ".\n" \
             << "the vector size is : " << vec.size() << "\n";

    }
    
    cout << "\nend of input\ninput log : \n";

    for (const auto & num : vec)
    {
        cout << "\t" << num << "\n";
    }
    
    return 0;
}