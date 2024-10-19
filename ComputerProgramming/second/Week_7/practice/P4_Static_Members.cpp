#include <iostream>
#include <string>

using namespace std;

class Server
{
private:
    std::string name;
    static int turn;
    static int last_served;
    static bool status_open;

public:
    Server(const string & name);
    ~Server() = default;
    static int get_turn();
    void serve_one();
    static bool is_open();
};

int Server::turn = 0;
int Server::last_served = 0;
bool Server::status_open = true;

int main()
{   
    Server s1("A"), s2("B");

    int number = 0, count = 0;

    do
    {   
        // get the num that need to turn 
        cout << "How many in your group ? ";
        cin >> number;

        // Server get turn 
        cout << "Your turns are : ";
        for (count = 0; count < number; count++)
        {
            cout << Server::get_turn() << " ";
        }

        // serve - to make last turn + 1
        cout << "\n";
        s1.serve_one();
        s2.serve_one();

    } while (Server::is_open());

    cout << "Now closing service.\n";

    return 0;
}


Server::Server(const string & name)
{
    // return server name
    this->name = name;
}

bool Server::is_open()
{
    // open status
    return status_open;
}
int Server::get_turn()
{
    // turn + 1
    turn++;
    return turn;
}

void Server::serve_one()
{
    // if server is open, and has people need to serve
    if (is_open() && last_served < turn)
    {
        last_served++;
        cout << "Server " << name << " Now Serving " \
             << last_served << ".\n";
    }
    // if not have any turn need to serve, server can close
    else if (last_served == turn)
    {
        Server::status_open = false;
    }
}