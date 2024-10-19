#include <iostream>
#include <string>
using namespace std;

class Account {
private:
    
public:

    string name;
    int balance;

    Account() {
        // Finish this part
    }

    string getname() {
        return this->name;
    }

    void deposit(int amount) {
        this->balance += amount;
        cout << this->name << " deposit " << amount << ".\n";
    }

    bool withdraw(int amount) {
        
        if (this->balance < amount)
        {
            cout << this->name << "'s account balance is insufficient.\n";
            return false;
        }
        this->balance -= amount;
        cout << this->name << " withdraw " << amount << " from his/her account.\n";
        return true;
    }

    bool transfer(Account* transfer, int amount) {

        if (this->name == transfer->getname())
        {
            cout << "You transfer to yourself.\n";
        }
        else if (this->balance >= amount)
        {
            cout << "Transfer successful.\n";
            transfer->balance += amount;
            this->balance -= amount;
        }
        else
        {
            cout << "Transfer failed.\n";
        }
        return true;
    }

    void getaccInfo() {
        cout << this->name << "\t" << this->balance << "\n";
    }

    void set_info()
    {
        cin >> this->name;
        cin >> this->balance;
    }

};

int main() {
    int amt;
    cin >> amt;
    Account acc[amt];
    for(int i = 0; i < amt; i++) acc[i].set_info();
	
    string command, who, object;
    int amount, tmp_withdraw, tmp_transfer;
    while (cin >> who >> command >>  amount) {
        // find acc index
        
        auto find_who = [&](string find_name)
        {

            for (int i = 0; i < amt; i++)
            {
                if (acc[i].getname() == find_name)
                {
                    return i;
                }
            }
            return 0;
        };

        int index = find_who(who);

        if (command == "deposit")
        {
            acc[index].deposit(amount);
        }
        else if (command == "withdraw")
        {
            acc[index].withdraw(amount);
        }
        else if (command == "transfer")
        {
            string receiver;
            cin >> receiver;
            acc[index].transfer(&acc[find_who(receiver)], amount);
        }


    }
    cout << "\nName\tBalance" << endl;
    for(int i = 0; i < amt; i++) acc[i].getaccInfo();
    return 0;
}