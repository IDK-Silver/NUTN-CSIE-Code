#include <iostream>
using namespace std;

class BasicClass {

private:

public:
    BasicClass() = default;
    ~BasicClass() = default;


    void basic_class_info()
    {
        cout << "This is basic class\n";
    };
};


class SuperClass : public BasicClass {

private:

public:
    SuperClass() = default;

    ~SuperClass() = default;

    void super_class_info()
    {
        cout << "This is super class\n";
    };
};

int main()
{
    SuperClass super_obj;
    super_obj.basic_class_info();
    super_obj.super_class_info();

    return 0;
}
