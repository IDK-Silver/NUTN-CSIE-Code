#include <iostream>

using namespace std;

int main() {
  int *number = nullptr;
  number = new int[5];

  for (int i = 0; i < 5; i++) {
    cin >> number[i];
  }

  for (int i = 0; i < 5; i++)
    cout << number[i] << " ";
  cout << "\n";
  
  delete number;

  return 0;

}
