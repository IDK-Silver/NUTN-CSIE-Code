#include <iostream>
#include <memory>
#include <string>
#include <sstream>

using namespace std;

class Node {
public:
    Node() = default;
    ~Node() = default;
    int value = 0;
    shared_ptr<Node> left;
    shared_ptr<Node> right;
};

class LinkedList {


private:
    int size = 0;
    shared_ptr<Node> root_node;
    shared_ptr<Node> end_node;

    string push_mode = "FIFO";

public:

    void set_psuh_mode(string mode)
    {
        if (mode == "FIFO")
            push_mode = "FIFO";

        else if (mode == "FILO")
            push_mode = "FILO";

        else
            push_mode = "FIFO";
    };

    void push(int value)
    {
        if (push_mode == "FIFO")
            push_back_node(value);

        else if (push_mode == "FILO")
            push_front_node(value);
    };

    void push_back_node(int value) {

        auto new_node = make_shared<Node>();
        new_node->value = value;
        size++;

        if (root_node == nullptr)
        {
            root_node = new_node;
            end_node = root_node;
            return;
        }


        auto current_node = root_node;

        while (current_node->right != nullptr)
        {
            current_node = current_node->right;
        }
        new_node->left = current_node;
        current_node->right = new_node;
        end_node = new_node;
    };



    void push_front_node(int value)
    {
        auto new_node = make_shared<Node>();
        new_node->value = value;
        size++;

        if (root_node == nullptr)
        {
            root_node = new_node;
            end_node = root_node;
            return;
        }

        auto current_node = root_node;
        current_node->left = new_node;
        new_node->right = current_node;
        root_node = new_node;
    };

    void clear()
    {
        root_node = nullptr;
        end_node = nullptr;
        size = 0;
    }

    void print_node()
    {
        auto current_node = root_node;

        while(current_node != nullptr)
        {
            cout << current_node->value << " ";
            current_node = current_node->right;
        }
        cout << "\n";
    }
};


int main()
{
    string input_type;
    LinkedList list;
    while (cin >> input_type)
    {

        list.set_psuh_mode(input_type);

        string input_str;
        getline(cin, input_str);

        stringstream ss;
        ss << input_str;

        int value = 0;
        while(ss >> value)
        {
            list.push(value);
        }
        cout << input_type << " Linked-list : ";
        list.print_node();
    }
}
