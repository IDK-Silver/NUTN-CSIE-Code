#include <iostream>
#include <memory>

using namespace std;

void sneaky(shared_ptr<int>);

int main()
{
    auto p = make_shared<int>();

    *p = 77;

    cout << "Before call to function *p == " << *p << "\n";
    sneaky(p);

    cout << "After call to function *p == " << *p << "\n";
    
    return 0;
}

void sneaky(shared_ptr<int> p)
{
    *p = 99;
    cout << "Inside function call *p == " << *p << "\n";
}