#include <iostream>
#include <string>
#include <map>

using namespace std;

int get_public_key(const string & str)
{

    map<char, int> split_map;

    for (const auto & w : str)
    {
        if (isalpha(w) || isdigit(w))
            split_map[w] += 1;
    }

    int w = 0;

    for (const auto & s : split_map)
    {
        w += (int) s.first * s.second;
    }

    return w % str.length();
}

string get_encryption(const string & str, const int  & public_key)
{
    string encryption;

    encryption += str.substr(public_key, str.length() - public_key);
    encryption += str.substr(0, public_key);
    
    return encryption;
}


string get_decrypt(const string & str, int public_key)
{
    public_key = str.length() - public_key;
    string decrypt;

    decrypt += str.substr(public_key, str.length() - public_key);
    decrypt += str.substr(0, public_key);

    return decrypt;
}


int main()
{
    string encryption;
    string decryption;
    getline(cin, encryption);
    getline(cin, decryption);
    cout << get_encryption(encryption, get_public_key(encryption)) << "\n";
    cout << get_decrypt(decryption, get_public_key(decryption)) << "\n";
}