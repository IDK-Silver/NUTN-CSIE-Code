#include "employee.h"

int main()
{
    SavitchEmployees::Employee person("IDK", "12");
    person.setNetPay(5000);

    person.printCheck();
    return 0;
}
