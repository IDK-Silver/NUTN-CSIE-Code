#include <fstream>

using namespace std;

int main()
{
    ifstream in_stream;
    ofstream out_stream;

    in_stream.open("input_file.txt");
    out_stream.open("output_file.txt");

    int input_num = 3;
    int sum = 0;
    while (in_stream >> input_num)
    {
        sum += input_num;
    }

    out_stream << "sum from inputfile is : " << sum << "\n";

    in_stream.close();
    out_stream.close();
    return 0;
}
