#include <iostream>

using namespace std;

const double RATE = 150.00;

double fee(const double  & hours_worked, const int & minutes_worked);

int main()
{
    int hour = 0, minutes = 0;
    double bill = 0;

    cout << "Input hour & minutes : \n\t>";
    cin >> hour >> minutes;

    bill = fee(hour, minutes);

    cout.setf(ios::fixed);
    cout.setf(ios::showpoint);
    cout.precision(2);

    cout << "For " << hour << " hour and " << minutes \
    << " minutes, you bill is $" << bill << "\n";

    return 0;
}

double fee(const double  & hours_worked, const int & minutes_worked)
{
    return ( ((hours_worked * 60.0) + minutes_worked) / 15.0 ) * RATE;
}