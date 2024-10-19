#include<iostream>
#include<cstdlib>
using namespace std;

int main()
{
    int month = 0, day = 0;

    cout << "Enter two int : ";

    cin >> month;
    cin >> day;
    
    srand(month * day);

    int prediction = 0;

    char ans;

    cout << "Weather for today :\n";

    do
    {
        prediction = rand() % 3;

        switch (prediction)
        {
            case 0:
                cout << "The day will be sunny!!\n";
                break;
            case 1:
                cout << "The day will be cloudy.\n";
                break;
            case 2:
                cout << "The day will be stormy!\n";
                break;
            default:
                cout << "Error\n";
                break;

        }

        cout << "Want the weather for the next day (y/n)";
        cin >> ans;
    } while(ans == 'y' || ans == 'Y');

    cout << "That's it from your 24 hour weather program\n";
    return 0;
    
}
