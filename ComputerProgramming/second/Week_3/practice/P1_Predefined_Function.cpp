#include <iostream>
#include <cmath>

using namespace std;

int main()
{
    const double CONST_PRE_SQ_FT = 10.5;

    double budget = 0, area = 0, length_side = 0;

    cout << "Enter the amount budgeted for your doghouse $ ";
    cin >> budget;

    area = budget / CONST_PRE_SQ_FT;
    length_side = sqrt(area);

    cout.setf(ios::fixed);
    cout.setf(ios::showpoint);
    cout.precision(2);

    cout << "For a price of $ " << budget << "\n"
        << "I can build you a luxurious square doghouse \n"
        << "that is  " << length_side
        << " feet on each side.\n";
    return 0;
}
