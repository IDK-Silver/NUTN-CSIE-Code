#include <iostream>

using namespace std;

struct ShoeType
{
    char style;
    double price = 0;
};

void print(const ShoeType & shoe)
{
    cout.setf(ios::fixed);
    cout.setf(ios::showpoint);
    cout.precision(2);
    cout << "\tStyle " << shoe.style << "\n" \
         << "\tPrice " << shoe.price << "\n";
}


void set_discount(ShoeType & shoe)
{
    double discount = 0;
    cout << "Enter Discount : ";
    cin >> discount;
    
    cout << "Origin \n";
    print(shoe);
    shoe.price *= (1.0 - discount);
    cout << "Discount\n";
    print(shoe);
}

int main()
{
    ShoeType shoe_1 = {'S', 29.6};
    set_discount(shoe_1);
}