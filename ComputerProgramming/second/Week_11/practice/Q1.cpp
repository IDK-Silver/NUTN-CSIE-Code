#include <iostream>
#include <memory>

using namespace std;

void print_ptr(const shared_ptr<int> & p1, const shared_ptr<int> & p2)
{
    cout << *p1 << " " << *p2 << "\n";
    return;
}

int main()
{
    shared_ptr<int> p1, p2;

    /* p1 pointer to a new memory sapce */
    p1 = make_shared<int>();

    /* p2 pointer to p1 addres */
    p2 = p1;

    /* set p1's memory space value to 42 */
    *p1 = 42;

    /* print p1 and p2, p1's memory space value is 42, p2 is also 42 because p2 and p1 pointer to same memory space */
    print_ptr(p1, p2);
    
    /* set p2's memory space value to 53 */
    *p2 = 53;
    /* print p1 and p2, p1's memory space value is 53, p2 is also 53 because p2 and p1 pointer to same memory space */
    print_ptr(p1, p2);

    /* set p1 pointer to new memory space, and set value 88 */
    p1 = make_shared<int>(88);

    /* print p1, and p2, p1's memeory space value is 88, p2 is 53, because p1, and p2 is not same memory space */
    print_ptr(p1, p2);
    
    return 0;
}