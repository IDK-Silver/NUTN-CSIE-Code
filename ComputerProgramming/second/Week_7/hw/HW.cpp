#include <iostream>
#include <map>
#include <string>
#include <ctype.h>
using namespace std;
int main()
{
    string str;
    string temp;
    while (cin >> temp)
    {
        str += temp;
    }

    map<char, int> result;



    for (char w = '0'; w <= '9'; w++)
    {
        result[w] = 0;
    }
    
    for (char w = 'A'; w <= 'Z'; w++)
    {
        result[w] = 0;
    }
    

    for (char w = 'a'; w <= 'z'; w++)
    {
        result[w] = 0;
    }

    for (auto word : str)
    {
        if (isalpha(word) || isdigit(word))
        {
            result[word] += 1;
        }
    }

    for (auto s : result)
    {
        cout << s.first << "\t" << s.second << "\n";
    }

}