#include <iostream>

using namespace std;

int main() {

  int size = 0;

  cin >> size;

  for (int n = size; n >= 1; n--)
  {
    if (1 == n)
      cout << "Hurray\n";
    else 
    {
      cout << "Hip, ";
    }
  }
  return 0;
}
