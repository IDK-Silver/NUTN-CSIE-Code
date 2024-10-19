#include <iostream>
#include <cstdlib>
#include <fstream>

using namespace std;

int main()
{
    ifstream in;
    ofstream out;

    in.open("in");
    if (in.fail())
    {
        cout << "input file opening failed.\n";
        exit(1);
    }

    out.open("out");
    if (out.fail())
    {
        cout << "output file opening failed.\n";
        exit(1);
    }

    int input_num = 0;
    int sum = 0;
    while (in >> input_num)
        sum += input_num;

    out << "sum number is " << sum << "\n";

    in.close();
    out.close();

    return 0;
}