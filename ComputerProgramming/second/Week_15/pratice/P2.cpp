#include <iostream>

using namespace std;

class Cat {
public:
    void meow() {
        cout << "Meow\n";
        this->private_function();
    }

public:
    void private_function() {
        cout << "This is private function\n";
    };
};

class LongLegCat : public Cat {
public:
    void jump() {
        cout << "Jump\n.";
    }
};

int main()
{
    LongLegCat cat;
    cat.meow();
    cat.jump();

}
