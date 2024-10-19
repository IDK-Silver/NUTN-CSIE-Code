#include <iostream>

using namespace std;

int main()
{
	int *p1, *p2;
	p1 = new int;
	p2 = new int;
	*p1 = 10;
	*p2 = 20;
	cout << *p1 << "" << *p2 << endl;
	*p1 = *p2;
	cout << *p1 << "" << *p2 << endl;
	*p1 = 30;
	cout << *p1 << "" << *p2 << endl;
	return 0;
}
