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
        cout << this->user_name << " �z�n�G�������O�� " << price << "��\n";
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
        printf("�˷R���|��%s�z�n�A�������O�� %d ���A��%d���ꦬ %d ���C",
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
    cout << "��J�A�Q�n���|������ : (P, M, G) : ";
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
