#include <iostream>
#include <fstream>

using namespace std;

int main()
{
    cout << "opening data.txt for appending.\n";
    ofstream fout;
    fout.open("data.txt", ios::app);

    fout << "5 6\n" \
         << "7 8\n";
        
    fout.close();

    cout << "end of appending to file\n";

    return 0;
}