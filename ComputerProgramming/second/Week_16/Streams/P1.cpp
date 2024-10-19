#include <iostream>
#include <iomanip>

using namespace std;

void ans_a() {
  cout.setf(ios::fixed);
  cout.setf(ios::showpoint);
  cout.precision(2);
  cout << 12.22233 << "\n";
};

void ans_b() {
  cout.setf(ios::fixed | ios::showpoint);
  cout << setprecision(2) << 12.22233 << "\n";

}

void ans_c() {
  cout << setiosflags(ios::fixed);
  cout << setiosflags(ios::showpoint);
  cout << setprecision(2) << 12.22233 << "\n";
}

void ans_d() {
  cout.flags(ios::fixed);
  cout.flags(ios::showpoint);
  cout.precision(2);
  cout << 12.22233 << "\n";
}

int main()
{
  ans_a();
  ans_b();
  ans_c();
  ans_d();
  cout << "\n";
  return 0;
}
