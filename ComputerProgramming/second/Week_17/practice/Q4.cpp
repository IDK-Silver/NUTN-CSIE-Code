#include <iostream>
#include <string>

using namespace std;

void change_to_x(string & str) {

  for (auto & word : str) {
    word = 'X';
  }

}

int main()
{
  string str = "ABCDE";

  cout << str << "\n";

  change_to_x(str);

  cout << str << "\n";

  return 0;

}
