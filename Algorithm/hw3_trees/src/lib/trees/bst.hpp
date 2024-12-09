#ifndef YUFENG_BINARYSEARCHTREE_H
#define YUFENG_BINARYSEARCHTREE_H

#include <iostream>

template <typename T>
class BinarySearchTree
{
private:
    struct Node
    {
        T data;      // Data stored in the node
        Node *left;  // Left child
        Node *right; // Right child

        Node(const T &value)
            : data(value), left(nullptr), right(nullptr) {}
    };

    Node *root;

    void insert_node(Node *&node, const T &value)
    {
        if (node == nullptr)
        {
            node = new Node(value);
        }
        else if (value < node->data)
        {
            insert_node(node->left, value);
        }
        else if (value > node->data)
        {
            insert_node(node->right, value);
        }
        // If the value is equal, do not insert (no duplicates allowed)
    }

    // Private in-order traversal function, recursively traverses the tree
    void in_order_traversal(Node *node) const
    {
        if (node != nullptr)
        {
            in_order_traversal(node->left);
            std::cout << node->data << " ";
            in_order_traversal(node->right);
        }
    }

    // Private search function, recursively searches for a value
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

    // Private function to delete the tree, recursively frees memory
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
    BinarySearchTree() : root(nullptr) {}

    // Destructor, frees all nodes' memory
    ~BinarySearchTree()
    {
        delete_tree(root);
    }

    // Public insert function, user interface
    void insert_node(const T &value)
    {
        insert_node(root, value);
    }

    // Public in-order traversal function, user interface
    void in_order_traversal() const
    {
        in_order_traversal(root);
        std::cout << std::endl;
    }

    // Public search function, user interface
    bool search_node(const T &value) const
    {
        return search_node(root, value) != nullptr;
    }
};

#endif // YUFENG_BINARYSEARCHTREE_H