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

    string str;
    int num = 0;
    // 如果沒有讀到 end of file
    while (!in.eof())
    {
        // 取得一行
        getline(in, str);

        // 輸出 行編號 + 取得的文字
        out << ++num << " " << str << "\n";
    }

    in.close();
    out.close();

    return 0;
}