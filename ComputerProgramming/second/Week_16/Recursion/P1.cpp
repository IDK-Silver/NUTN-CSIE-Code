#include <iostream>

void recursive(int i) {
  using namespace std;
  if (i < 8) {
    cout << i << " ";
    recursive(i + 1); // Recursive call with an updated value of i
  }
}

int main() {
  recursive(4); // Initial call to the recursive function
  return 0;
}
