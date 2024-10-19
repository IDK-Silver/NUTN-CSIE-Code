#include <iostream>

using namespace std;


class DayOfYear
{
    private:

        int month = 1;
        int day = 1;

    public:
        DayOfYear(int month, int day)
        {
            this->month = month;
            this->day = day;
        }
        void output_info();
};

void DayOfYear::output_info()
{
    cout << "Month : " << this->month << "\tDay : " << this->day << "\n";
}



int main()
{
    DayOfYear d1(12, 24), d2(1, 10);

    d1.output_info();
    d2.output_info();

    return 0;
}