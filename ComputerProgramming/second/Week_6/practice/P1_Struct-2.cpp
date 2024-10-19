#include <iostream>
#include <string>

using namespace std;

struct Student
{
    string name;
    int age;
    double grade;
};

void scanf(Student & stu);
void print(const Student & stu);

int main()
{
    Student stu;

    scanf(stu);
    print(stu);

    return 0;
}


void scanf(Student & stu)
{
    cout << "Enter namem age and grade : ";
    cin >> stu.name;
    cin >> stu.age;
    cin >> stu.grade;
}

void print(const Student & stu)
{
    cout << stu.name << "\n";
    cout << stu.age << "\n";
    cout << stu.grade << "\n";
}