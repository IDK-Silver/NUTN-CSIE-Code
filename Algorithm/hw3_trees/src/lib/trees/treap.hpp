#ifndef YUFENG_TREAP_H
#define YUFENG_TREAP_H

#include <iostream>
#include <cstdlib>  // For rand()
#include <ctime>    // For seeding rand()

template <typename T>
class Treap
{
private:
    struct Node
    {
        T data;         // Data stored in the node
        int priority;   // Priority of the node
        Node *left;     // Left child
        Node *right;    // Right child

        Node(const T &value)
            : data(value), priority(rand()), left(nullptr), right(nullptr) {}
    };

    Node *root;

    // Right rotation
    void rotate_right(Node *&node)
    {
        Node *left_child = node->left;
        node->left = left_child->right;
        left_child->right = node;
        node = left_child;
    }

    // Left rotation
    void rotate_left(Node *&node)
    {
        Node *right_child = node->right;
        node->right = right_child->left;
        right_child->left = node;
        node = right_child;
    }

    // Recursive insertion
    void insert_node(Node *&node, const T &value)
    {
        if (node == nullptr)
        {
            node = new Node(value);
        }
        else if (value < node->data)
        {
            insert_node(node->left, value);
            if (node->left->priority > node->priority)
            {
                rotate_right(node);
            }
        }
        else if (value > node->data)
        {
            insert_node(node->right, value);
            if (node->right->priority > node->priority)
            {
                rotate_left(node);
            }
        }
        // If the value is equal, do not insert (no duplicates allowed)
    }

    // Recursive deletion
    void delete_node(Node *&node, const T &value)
    {
        if (node == nullptr)
            return;
        if (value < node->data)
        {
            delete_node(node->left, value);
        }
        else if (value > node->data)
        {
            delete_node(node->right, value);
        }
        else
        {
            // Node found
            if (node->left == nullptr && node->right == nullptr)
            {
                delete node;
                node = nullptr;
            }
            else if (node->left == nullptr)
            {
                rotate_left(node);
                delete_node(node->left, value);
            }
            else if (node->right == nullptr)
            {
                rotate_right(node);
                delete_node(node->right, value);
            }
            else
            {
                if (node->left->priority > node->right->priority)
                {
                    rotate_right(node);
                    delete_node(node->right, value);
                }
                else
                {
                    rotate_left(node);
                    delete_node(node->left, value);
                }
            }
        }
    }

    // In-order traversal
    void in_order_traversal(Node *node) const
    {
        if (node != nullptr)
        {
            in_order_traversal(node->left);
            std::cout << node->data << " ";
            in_order_traversal(node->right);
        }
    }

    // Search for a node
    Node *search_node(Node *node, const T &value) const
    {
        if (node == nullptr)
            return nullptr;
        if (value == node->data)
            return node;
        else if (value < node->data)
            return search_node(node->left, value);
        else
            return search_node(node->right, value);
    }

    // Delete the entire tree
    void delete_tree(Node *node)
    {
        if (node != nullptr)
        {
            delete_tree(node->left);
            delete_tree(node->right);
            delete node;
        }
    }

public:
    // Constructor, initializes the root to nullptr
    Treap() : root(nullptr)
    {
        srand(static_cast<unsigned int>(time(0))); // Seed for random priorities
    }

    // Destructor, frees all nodes' memory
    ~Treap()
    {
        delete_tree(root);
    }

    // Public insert function
    void insert_node(const T &value)
    {
        insert_node(root, value);
    }

    // Public delete function
    void remove(const T &value)
    {
        delete_node(root, value);
    }

    // Public in-order traversal
    void in_order_traversal() const
    {
        in_order_traversal(root);
        std::cout << std::endl;
    }

    // Public search function
    bool search_node(const T &value) const
    {
        return search_node(root, value) != nullptr;
    }
};

#endif // YUFENG_TREAP_H