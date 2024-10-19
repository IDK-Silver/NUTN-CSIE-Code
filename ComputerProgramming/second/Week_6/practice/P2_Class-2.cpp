#include <iostream>
#include <string>

using namespace std;

class Student
{
private:
    
public:
    int age;
    string name;
    Student() = default;
    ~Student() = default;
    void show_info();
};

void Student::show_info()
{
    cout << "Info\n";
    cout << "\tName:" << this->name << "\n" \
         << "\tAge:" << this->age << "\n";
}

int main()
{
    Student stu;
    stu.name = "Kano";
    stu.age = 20;

    stu.show_info();
}