#include <iostream>

using namespace std;

struct CDAccountV1
{
    double balance = 0;
    double interest_rate = 0;
    int term = 0;
};

void get_data(CDAccountV1 &account);

int main()
{
    CDAccountV1 account;
    get_data(account);

    double rate_fraction = 0, interest = 0;

    rate_fraction = account.interest_rate / 100.0;
    interest = account.balance * \
             (rate_fraction * (account.term / 12.0));
    account.balance = account.balance + interest;

    cout.setf(ios::fixed);
    cout.setf(ios::showpoint);
    cout.precision(2);

    cout << "When your CD matures in " \
         << account.term << " mounts,\n" \
         << "it will have a balance of $ " \
         << account.balance << "\n";
    
    return 0;
}

void get_data(CDAccountV1 &account)
{
    cout << "Enter account balance : $ ";
    cin >> account.balance;

    cout << "Enter account interest rate : ";
    cin >> account.interest_rate;

    cout << "Enter the number of months untill maturity : ";
    cin >> account.term;
}