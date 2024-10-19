#include <iostream>
#include <vector>
using namespace std;
int main()
{
    int input_number = 0;
    vector<int> vec;
    while (cin >> input_number)
    {
        vec.push_back(input_number);
    }

    cout << "First In First Out (Queue) order : ";
    for (auto iter = vec.begin(); iter != vec.end(); iter++)
        cout << *iter << " ";

    cout << "\n";
    cout << "First In Last Out (Stack) order : ";
    for (auto iter = vec.rbegin(); iter != vec.rend(); iter++)
        cout << *iter << " ";
}