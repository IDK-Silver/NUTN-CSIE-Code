#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
using namespace std;
void make_neat(ifstream & in, ofstream & out, const int & d_p, const int & width)
{
    out.setf(ios::fixed);
    out.setf(ios::showpoint);
    out.precision(d_p);

    cout.setf(ios::fixed);
    cout.setf(ios::showpoint);
    cout.precision(d_p);

    double next = 0;
    while (in >> next)
    {
        out << setw(width) << next << '\n';
        cout << setw(width) << next << '\n';
    }
}
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
    make_neat(in, out, 5, 12);
    in.close();
    out.close();
    return 0;
}
