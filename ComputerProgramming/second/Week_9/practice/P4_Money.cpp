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
    const int cents_part(double amount) const
    {
        int double_cents = amount * 100;
        int int_cents = (round(fabs(double_cents))) % 100;
        if (amount < 0)
            int_cents = -int_cents;
        return int_cents;
    };
    const int round(double number) const
    {
        return static_cast<int>(floor(number + 0.5));
    };

public:
    friend const Money operator-(const Money &amount_1, const Money &amount_2)
    {
        int all_cents_1 = amount_1.get_cents() + amount_1.get_dollars() * 100;
        int all_cents_2 = amount_2.get_cents() + amount_2.get_dollars() * 100;

        int sum_all_cents = all_cents_1 - all_cents_2;
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
    friend Money operator+(const Money &amount_1, const Money &amount_2)
    {
        int all_cents_1 = amount_1.get_cents() + amount_1.get_dollars() * 100;
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
    friend const bool operator==(const Money &amount_1, const Money &amount_2)
    {
        return ((amount_1.get_dollars() == amount_2.get_dollars()) &&
                (amount_1.get_cents() == amount_2.get_cents()));
    }
    friend const Money operator-(const Money &amount)
    {
        return Money(-amount.get_dollars(), -amount.get_cents());
    }

    friend ostream& operator<<(ostream &output_stream, const Money &amount)
    {
        int abs_dollars = abs(amount.dollars);
        int abs_cents = abs(amount.cents);

        if (amount.dollars < 0 || amount.cents < 0)
            output_stream << "$-";
        else 
            output_stream << "$";

        output_stream << abs_dollars;

        if (abs_cents >= 10)
            output_stream << "." << abs_cents;
        else
            output_stream << ".0" << abs_cents;
        
        return output_stream;
    }

    friend istream& operator>>(istream & input_stream, Money & amount)
    {
        char dollar_sign;
        input_stream >> dollar_sign;

        if (dollar_sign != '$')
        {
            cout << "No dollar sign in Money input.\n";
            exit(1);
        }

        double amount_as_double = 0;
        input_stream >> amount_as_double;
        amount.dollars = amount.dollar_part(amount_as_double);
        amount.cents = amount.cents_part(amount_as_double);
        return input_stream;
    }


    Money() = default;
    Money(double amount) : dollars(dollar_part(amount)), cents(cents_part(amount)) {};
    Money(int the_dollars, int the_cents)
    {
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
        cout << "$" << this->dollars << "." << this->cents;
    };

};

int main()
{
    Money your_amount, my_amount(10.1);

    cout << "Enter an amount of money ";
    cin >> your_amount;
    
    cout << "Yout amount is " << your_amount << "\n";

    cout << "My amount is " << my_amount << "\n";

    if (your_amount == my_amount)
        cout << "We have same amounts.\n";
    else
        cout << "One of us is richer.\n";

    Money our_amount = your_amount + my_amount;
    cout << your_amount << "+" << my_amount << " is equal " << our_amount << "\n";

    Money diff_amount = your_amount - my_amount;
    cout << your_amount << "-" << my_amount << " is equal " << diff_amount << "\n";

    return 0;
}