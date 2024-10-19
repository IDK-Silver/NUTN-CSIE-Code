#include <iostream>
#include <string>
#include <cstdio>
#include <memory>

using namespace std;

class Person {
protected:

    string user_name;

public:
    Person() {
        cout << "Input Name : ";
        cin >> this->user_name;
    };
    ~Person() = default;

    virtual void pay(int price)
    {
        cout << this->user_name << " 您好：本次消費為 " << price << "元\n";
    }
};


class Member : public Person
{
protected:
    double discount = 0.8;

public:
    Member() : Person() {
    };

    void pay(int price) {
        printf("親愛的會員%s您好，本次消費為 %d 元，打%d折後實收 %d 元。",
                user_name.c_str() , price, (int)(discount * 10), (int)(price * discount));
    }
};


class GoldMember : public Member {
public:
    GoldMember() : Member() {
        this->discount = 0.7;
    }
};

int main()
{
    cout << "輸入你想要的會員等級 : (P, M, G) : ";
    char ans = 0;
    cin >> ans;

    std::unique_ptr<Person> customers;

    switch (ans)
    {
    case 'P':
        customers = make_unique<Person>();
        break;

    case 'M':
        customers = make_unique<Member>();
        break;

    case 'G':
        customers = make_unique<GoldMember>();
        break;

    default:
        cout << "Input error, using default\n";
        customers = make_unique<Person>();
        break;
    }

    customers->pay(1000);
    return 0;
}
