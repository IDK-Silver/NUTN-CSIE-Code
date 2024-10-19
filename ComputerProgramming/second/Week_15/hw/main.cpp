#include <iostream>
#include <string>
#include <memory>
using namespace std;

class Node {
public:
    int value = 0;
    shared_ptr<Node> next = nullptr;
    Node(int value) : value(value) {};
    ~Node() = default;
};

class LinkedList {

private:
    int size = 0;
    string mode;
    shared_ptr<Node> head;

public:

    LinkedList(string input_mode)
    {
        mode = input_mode;
    };

    void insert(int value)
    {

        auto new_node = make_shared<Node>(value);
        size++;

        if (head == nullptr)
        {
            head = new_node;
            return;
        }

        if (mode == "FILO")
        {
            new_node->next = head;
            head = new_node;
        }
        else
        {
            shared_ptr<Node> current_node = head;

            while (current_node->next != nullptr)
                current_node = current_node->next;

            current_node->next = new_node;
        }
    };

    bool remove(int index)
    {
        if (size == 0)
            return false;

        // cout << size << " " << index;
        if (size <= index || index < 0)
            return false;

        if (index == 0)
        {
            head = head->next;
            size--;
            return true;
        }

        auto current_node = head;

        shared_ptr<Node> r_node = nullptr;

        while (index--)
        {
            r_node = current_node;
            current_node = current_node->next;
        }

        r_node->next = current_node->next;
        size--;
        return true;

    };

    int search(int target)
    {
        auto current_node = head;

        int find_index = 0;

        while (current_node != nullptr)
        {
            if (current_node->value == target)
                return find_index;

            current_node = current_node->next;
            find_index++;
        }

        return -1;
    };

    void display()
    {
        cout << this->mode << " Linked-list : ";


        auto current_node = head;


        while (current_node != nullptr)
        {
            cout << current_node->value << " ";
            current_node = current_node->next;
        }
        cout << endl;
    };
};

int main() {
	string mode;
	cin >> mode;
    LinkedList list(mode);
	int num;
	while(cin >> num) {
		if (num != -1) list.insert(num);
		else break;
	}
    list.display();
	string command;
	int value;
	while (cin >> command >> value) {
		//cout << command << value;
		if (command == "remove") {
			int remove = list.remove(value);
			if (remove) list.display();
			else cout << "Index exceed linked-list length."<< endl;
		} else if (command == "search") {

			int search = list.search(value);
			if (search!=-1) cout << "The value " << value << " at the index " << search << endl;
			else cout << "The value doesn't exist in the linked-list."<< endl;
		}
	}
	list.display();
    return 0;
}
