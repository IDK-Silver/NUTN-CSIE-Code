#include <iostream>
#include <deque>
#include <stack>

using namespace std;

int main()
{
    deque<int> q;
    stack<int> s;

    int input_number = 0;
    while (cin >> input_number)
    {
        q.push_back(input_number);
        s.push(input_number);
    }

    cout << "First In First Out (Queue) order : ";
    while (!q.empty())
    {
        cout << q.front() << " ";
        q.pop_front();
    }
    cout << "\n";
    cout << "First In Last Out (Stack) order : ";
    while (!s.empty())
    {
        cout << s.top() << " ";
        s.pop();
    }

    return 0;
}