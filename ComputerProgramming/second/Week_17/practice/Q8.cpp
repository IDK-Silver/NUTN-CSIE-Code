#include <iostream>

using namespace std;


int main()
{
  int size = 10;
  int *entry = new int [size];

  cout << "Enter " << size << " number : \n";

  for (int i = 0; i < size; i++)
    cin >> entry[i];

  for (int i = 0; i < size; i++)
     cout << entry[i] << " ";
  cout << "\n";

  return 0;
}
