#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;

class Decimal {
private:
    long long int base_;
    long long int integerPart_;
    long long int fractionalPart_;
public:

    Decimal(long long int integer, long long int fractional, long long int base) {
        integerPart_= integer;
        fractionalPart_ = fractional;
        base_ = base;
    }

    ~Decimal() = default;
    
    Decimal operator+(Decimal obj) {

        if (obj.integerPart_ < 0)
        {
            obj.fractionalPart_ = -obj.fractionalPart_;
        }

        Decimal sum(0, 0, 0);
        long long int i_fp;
        if (base_ == obj.base_)
        {
            if (fractionalPart_ + obj.fractionalPart_ >= base_)
            {
                sum.integerPart_ = integerPart_ + obj.integerPart_ + 1;
                sum.fractionalPart_ = (fractionalPart_ + obj.fractionalPart_) - base_ ;
                sum.base_ = base_;
                return sum;
            }
            else
            {
                sum.integerPart_ = integerPart_ + obj.integerPart_ ;
                sum.fractionalPart_ = fractionalPart_ + obj.fractionalPart_ ;
                sum.base_ = base_;
                return sum;
            }
        }
        else if(base_ > obj.base_)
        {

            
            i_fp = (fractionalPart_ * obj.base_) + (obj.fractionalPart_ * this->base_);
            sum.integerPart_ = integerPart_ + obj.integerPart_;

            if (i_fp < 0 && sum.integerPart_ > 0)
            {
                    sum.integerPart_ -= 1;
                    i_fp += obj.base_ * base_;;
            }
            
            if (i_fp < 0 && sum.integerPart_ < 0)
            {
                i_fp = abs(i_fp);
            }

            while (i_fp % 10 == 0)
            {
                i_fp /= 10;
            }

             if (i_fp > base_)
            {
                sum.integerPart_ = integerPart_ + obj.integerPart_ + 1;
                sum.fractionalPart_ = i_fp - base_ ;
                sum.base_ = base_;
                return sum;
            }

                sum.integerPart_ = integerPart_ + obj.integerPart_ ;
                sum.fractionalPart_ = i_fp ;
                sum.base_ = base_;
                return sum;


        }
        else if(base_ < obj.base_)
        {

            i_fp = (fractionalPart_ * obj.base_) + (obj.fractionalPart_ * this->base_);
            sum.integerPart_ = integerPart_ + obj.integerPart_;

            

            if (i_fp < 0 && sum.integerPart_ > 0)
            {
                    sum.integerPart_ -= 1;
                    i_fp += obj.base_ * base_;;
            }
            
            if (i_fp < 0 && sum.integerPart_ < 0)
            {
                i_fp = abs(i_fp);
            }

            while (i_fp % 10 == 0)
            {
                i_fp /= 10;
            }

            if (i_fp > obj.base_)
            {
                sum.integerPart_ += 1;
                sum.fractionalPart_ = i_fp - base_ ;
                sum.base_ = obj.base_;
                return sum;
            }
            
            sum.fractionalPart_ = i_fp;
            sum.base_ = obj.base_;
            return sum;
        }
    }

    Decimal operator-(Decimal obj){
        Decimal sub(0, 0, 0);

        if (obj.integerPart_ < 0 && obj.fractionalPart_ > 0)
        {
            obj.fractionalPart_ = -obj.fractionalPart_;
        }

        obj.integerPart_ = -obj.integerPart_;
        obj.fractionalPart_ = -obj.fractionalPart_;

        return operator+(obj);

        // if (base_ == obj.base_)
        // {
        //     if (fractionalPart_ > obj.fractionalPart_)
        //     {
        //         sub.integerPart_ = integerPart_ - obj.integerPart_;
        //         sub.fractionalPart_ = fractionalPart_ - obj.fractionalPart_;
        //         sub.base_ = base_;
        //         return sub;
        //     }
        //     else 
        //     {
        //         sub.integerPart_ = integerPart_ - obj.integerPart_ - 1;
        //         sub.fractionalPart_ = fractionalPart_ + base_ - obj.fractionalPart_;
        //         sub.base_ = base_;
        //         return sub;
        //     }
        // }
        // else if (base_ > obj.base_)
        // {
        //     obj.fractionalPart_ * (base_ / obj.base_);
        //     if (fractionalPart_ > obj.fractionalPart_ )
        //     {
        //         sub.integerPart_ = integerPart_ + obj.integerPart_ ;
        //         sub.fractionalPart_ = fractionalPart_ - obj.fractionalPart_;
        //         sub.base_ = base_;
        //         return sub;
        //     }
        //     else
        //     {
        //         sub.integerPart_ = integerPart_ - obj.integerPart_ - 1;
        //         sub.fractionalPart_ = fractionalPart_ + base_ - obj.fractionalPart_;
        //         sub.base_ = base_;
        //         return sub;
        //     }

        // }
        // else if (base_ < obj.base_)
        {
            fractionalPart_ * (obj.base_ / base_);
            if (fractionalPart_ > obj.fractionalPart_ )
            {
                sub.integerPart_ = integerPart_ + obj.integerPart_ ;
                sub.fractionalPart_ = fractionalPart_ - obj.fractionalPart_;
                sub.base_ = obj.base_;
                return sub;
            }
            else
            {
                sub.integerPart_ = integerPart_ - obj.integerPart_ - 1;
                sub.fractionalPart_ = fractionalPart_ + base_ - obj.fractionalPart_;
                sub.base_ = obj.base_;
                return sub;
            }
        }
    }

    Decimal operator*(Decimal obj){
        Decimal mul(0, 0, 0);
        long long int n1, n2;
        n1 = integerPart_ * base_ + fractionalPart_;
        n2 = obj.integerPart_ * obj.base_ + obj.fractionalPart_;

        mul.integerPart_ = (n1 * n2) / (base_ * obj.base_);
        mul.fractionalPart_ = (n1 * n2) - mul.integerPart_ * (base_ * obj.base_);
        mul.base_ = base_ * obj.base_;
        return mul;
    }

    friend ostream& operator<<(ostream& os, Decimal& decimal) {
        // I help you finish decimal complement 0, but the decimal mantissa part need you to clean.
	long long int frac = log10(decimal.base_);
        os << decimal.integerPart_ << '.' << setfill('0') << setw(frac) << abs(decimal.fractionalPart_);
        return os;
    }
};

int main() {
    int integerpart, fractionalpart, base;
    cin >> integerpart >> fractionalpart >> base;
    Decimal num1(integerpart, fractionalpart, base); 
    cin >> integerpart >> fractionalpart >> base;
    Decimal num2(integerpart, fractionalpart, base);

    Decimal sum = num1 + num2;
    Decimal sub = num1 - num2;
    Decimal mul = num1 * num2;
    cout << "Sum : " << sum << endl;
    cout << "Sub : " << sub << endl;
    cout << "Mul : " << mul << endl;
    return 0;
}