#include <iostream>

using namespace std;

int main()
{
    int score[5] = {0};

    int max = 0;

    cout << "Enter 5 scores:\n";
    cin >> score[0];

    max = score[0];


    for (int i = 1; i < 5; i++)
    {
        cin >> score[i];
        if (score[i] > max);
            max = score[i];
    }

    cout << "The max score is " << max << "\n" \
        << "Thew scores and their\n"
        << "differences from the highest are : \n";

    for (int i = 0; i < 5; i++)
        cout << score[i] << " off by "
            << (max - score[i]) << "\n";
}