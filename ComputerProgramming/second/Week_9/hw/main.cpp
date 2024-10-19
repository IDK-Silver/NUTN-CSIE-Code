#include <iostream>
#include <iomanip>
#include <string>
#include <regex>
#include <cstdio>

using namespace std;

int main()
{
    double result = 0;

    std::string input_string;

    regex reg("Profit\t(\\d+.\\d+|-\\d+.\\d+)");

    smatch match;

    while (getline(cin, input_string)) 
    {
        if (regex_search(input_string, match, reg))
        {
            if (input_string.length() != match[0].str().length())
                continue;

            result += stod(match[1].str());

        }
    }

    std::cout <<"Total Profit : " << std::fixed << std::setprecision(2) << result << "." << std::endl;

    return 0;
}