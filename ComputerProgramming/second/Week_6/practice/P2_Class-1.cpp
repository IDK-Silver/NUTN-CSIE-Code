#include <iostream>
#include <string>

using namespace std;

class Student
{
private:
    int age;
    string name;
public:
    Student() = default;
    ~Student() = default;
    void set_name(const string & name);
    void set_age(const int & age);
    void show_info();
};

void Student::set_name(const string & name)
{
    this->name = name;
}

void Student::set_age(const int & age)
{
    this->age = age;
}

void Student::show_info()
{
    cout << "Info\n";
    cout << "\tName:" << this->name << "\n" \
         << "\tAge:" << this->age << "\n";
}

int main()
{
    Student stu;
    stu.set_name("Kano");
    stu.set_age(20);

    stu.show_info();
}