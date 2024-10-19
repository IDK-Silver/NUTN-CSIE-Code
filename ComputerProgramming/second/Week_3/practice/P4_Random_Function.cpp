#include <iostream>

using namespace std;

double total_cost(int np, double pp)
{
    const double TAXRATE = 0.05;
    double sub_total = 0;

    sub_total = np * pp;

    return sub_total + (sub_total * TAXRATE);
}

int main()
{
    double price = 0, bill = 0;

    int number;

    cout << "Enter the number of items purchased : ";
    cin >> number;

    cout << "Enter the price per item $ : ";
    cin >> price;

    bill = total_cost(number, price);

    cout.setf(ios::fixed);
    cout.setf(ios::showpoint);
    cout.precision(2);

    cout << number << " items at "
        << "$" << price << " each.\n"
        << "Final bill, includeing tax, is $" << bill
        << endl;
    return 0; 
}
