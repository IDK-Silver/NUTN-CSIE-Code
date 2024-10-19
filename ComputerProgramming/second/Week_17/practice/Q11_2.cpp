#include <iostream>

using namespace std;

int main()
{
  int n = 5;

  while (true)
  {
    if (n == 1) 
    {
      cout << "Hurray!" << endl;
      break;
    }
    else {
      cout << "Hip, " << endl;
      n--;
    }
  }


}
