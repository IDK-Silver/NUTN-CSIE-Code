#include <iostream>
#include <fstream>

using namespace std;

int main()
{
    string user_input;
    cout << "input some words : ";
    getline(cin, user_input);

    ofstream out("outfile");

    if (out.is_open())
    {
        out << user_input;
        out.close();
        cout << "success" << "\n";
    }
    else
    {
        cout << "fail opened " << endl;
    }

    return 0;
}