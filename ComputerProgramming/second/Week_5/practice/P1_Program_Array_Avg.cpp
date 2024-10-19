#include <iostream>

using namespace std;

int main()
{
    int score[5] = {0};

    int sum = 0;

    cout << "Enter 5 scores:\n";
    cin >> score[0];

    for (int i = 1; i < 5; i++)
    {
        cin >> score[i];
        sum += score[i];
    }

    double avg = sum / 5.0;
    

    cout << "The avg score is " << avg << "\n" \
        << "Thew scores and their\n"
        << "differences from the highest are : \n";

    for (int i = 0; i < 5; i++)
        cout << score[i] << " off by "
            << (avg - score[i]) << "\n";
}