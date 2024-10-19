#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

class Money
{
private:
    int dollars = 0;
    int cents = 0;

    const int dollar_part(double amount)
    {
        return static_cast<int>(amount);
    };
    const int cents_part(double amount)
    {
        int double_cents = amount * 100;
        int int_cents = (round(fabs(double_cents))) % 100;
        if (amount < 0)
            int_cents = -int_cents;
        return int_cents;
    };
    const int round(double number)
    {
        return static_cast<int>(floor(number + 0.5));
    };
public:

    Money() = default;
    Money(double amount) : dollars(dollar_part(amount)), cents(cents_part(amount)) {};
    Money(int the_dollars, int the_cents){
        if ((the_dollars < 0 && the_cents > 0) || (the_dollars > 0 && the_cents < 0))
        {
            cout << "Inconsistent money data.\n";
            exit(1);
        }
        dollars = the_dollars;
        cents = the_cents;
    };
    Money(int the_dollars) : dollars(the_dollars), cents(0) {};
    ~Money() = default;

    double get_amount() const 
    {
        return this->cents * 0.01 + this->dollars;
    };
    int get_cents() const
    {
        return this->cents;
    };
    int get_dollars() const
    {
        return this->dollars;
    };
    void intput()
    {
        char dollar_sign;
        cin >> dollar_sign;

        if (dollar_sign != '$')
        {
            cout << "No dollar sign in Money input.\n";
            exit(1);
        }

        double amount_as_double = 0;
        cin >> amount_as_double;
        dollars = dollar_part(amount_as_double);
        cents = cents_part(amount_as_double);
    };
    void output()
    {
        
    };

    Money operator+(const Money &amount_2)
    {
        int all_cents_1 = get_cents() + get_dollars() * 100;
        int all_cents_2 = amount_2.get_cents() + amount_2.get_dollars() * 100;

        int sum_all_cents = all_cents_1 + all_cents_2;
        int abs_all_cents = abs(sum_all_cents);
        int final_dollars = abs_all_cents / 100;
        int final_cents = abs_all_cents % 100;

        if (sum_all_cents < 0)
        {
            final_dollars = -final_dollars;
            final_cents = - final_cents;
        }

        return Money(final_dollars, final_cents);
    }
    const Money operator-(const Money &amount_2)
    {
        int all_cents_1 = get_cents() - get_dollars() * 100;
        int all_cents_2 = amount_2.get_cents() - amount_2.get_dollars() * 100;

        int sum_all_cents = all_cents_1 + all_cents_2;
        int abs_all_cents = abs(sum_all_cents);
        int final_dollars = abs_all_cents / 100;
        int final_cents = abs_all_cents % 100;

        if (sum_all_cents < 0)
        {
            final_dollars = -final_dollars;
            final_cents = - final_cents;
        }
        return Money(final_dollars, final_cents);
    }
    const bool operator==(const Money &amount_2)
    {
        return ((get_dollars() == amount_2.get_dollars()) &&
                (get_cents() == amount_2.get_cents()));
    }  
    const Money operator-()
    {
        return Money(get_dollars(), get_cents());
    }
};




int main()
{
    Money your_amount, my_amount(10, 9);

    cout << "Enter an amount of money ";
    your_amount.intput();
    cout << "Yout amount is ";
    your_amount.output();
    cout << "\n";

    cout << "My amount is ";
    my_amount.output();
    cout << "\n";

    if (your_amount == my_amount)
        cout << "We have same amounts.\n";
    else
        cout << "One of us is richer.\n";

    Money our_amount = your_amount + my_amount;
    your_amount.output();
    cout << " + ";
    my_amount.output();
    cout << " is equal ";
    our_amount.output();
    cout << "\n";

    Money diff_amount = your_amount - my_amount;
    your_amount.output();
    cout << " - ";
    my_amount.output();
    cout << " is equal ";
    diff_amount.output();
    cout << "\n";

    return 0;
}