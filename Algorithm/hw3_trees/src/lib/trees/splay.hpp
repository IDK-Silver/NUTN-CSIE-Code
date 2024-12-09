#ifndef SPLAYTREE_H
#define SPLAYTREE_H

#include <iostream>

// Define the template SplayTree class
template <typename T>
class SplayTree {
private:
    // Define the structure of a tree node
    struct Node {
        T data;          // Data stored in the node
        Node* left;      // Left child
        Node* right;     // Right child

        // Constructor
        Node(const T& value)
            : data(value), left(nullptr), right(nullptr) {}
    };

    Node* root; // Root of the tree

    // Right rotate the subtree rooted with x
    Node* right_rotate(Node* x) {
        Node* y = x->left;
        if (y == nullptr) return x;
        x->left = y->right;
        y->right = x;
        return y;
    }

    // Left rotate the subtree rooted with x
    Node* left_rotate(Node* x) {
        Node* y = x->right;
        if (y == nullptr) return x;
        x->right = y->left;
        y->left = x;
        return y;
    }

    // Splay operation: bring the key to root
    Node* splay(Node* root, const T& key) {
        if (root == nullptr || root->data == key)
            return root;

        // Key lies in left subtree
        if (key < root->data) {
            // Key not in tree
            if (root->left == nullptr)
                return root;

            // Zig-Zig (Left Left)
            if (key < root->left->data) {
                root->left->left = splay(root->left->left, key);
                root = right_rotate(root);
            }
            // Zig-Zag (Left Right)
            else if (key > root->left->data) {
                root->left->right = splay(root->left->right, key);
                if (root->left->right != nullptr)
                    root->left = left_rotate(root->left);
            }

            // Second rotation
            return (root->left == nullptr) ? root : right_rotate(root);
        }
        // Key lies in right subtree
        else {
            // Key not in tree
            if (root->right == nullptr)
                return root;

            // Zag-Zag (Right Right)
            if (key > root->right->data) {
                root->right->right = splay(root->right->right, key);
                root = left_rotate(root);
            }
            // Zag-Zig (Right Left)
            else if (key < root->right->data) {
                root->right->left = splay(root->right->left, key);
                if (root->right->left != nullptr)
                    root->right = right_rotate(root->right);
            }

            // Second rotation
            return (root->right == nullptr) ? root : left_rotate(root);
        }
    }

    // Recursive insert function
    Node* insert_node(Node* root, const T& key) {
        if (root == nullptr)
            return new Node(key);

        root = splay(root, key);

        if (key == root->data)
            return root; // No duplicates

        Node* new_node = new Node(key);

        if (key < root->data) {
            new_node->right = root;
            new_node->left = root->left;
            root->left = nullptr;
        }
        else {
            new_node->left = root;
            new_node->right = root->right;
            root->right = nullptr;
        }

        return new_node;
    }

    // Recursive search function
    Node* search_node(Node* root, const T& key) {
        return splay(root, key);
    }

    // Private in-order traversal function
    void in_order_traversal(Node* node) const {
        if (node != nullptr) {
            in_order_traversal(node->left);
            std::cout << node->data << " ";
            in_order_traversal(node->right);
        }
    }

    // Private function to delete the tree
    void delete_tree(Node* node) {
        if (node != nullptr) {
            delete_tree(node->left);
            delete_tree(node->right);
            delete node;
        }
    }

public:
    // Constructor
    SplayTree() : root(nullptr) {}

    // Destructor
    ~SplayTree() {
        delete_tree(root);
    }

    // Public insert function
    void insert_node(const T& key) {
        root = insert_node(root, key);
    }

    // Public search function
    bool search_node(const T& key) {
        root = search_node(root, key);
        return (root != nullptr && root->data == key);
        // return root;
    }

    // Public in-order traversal
    void in_order_traversal() const {
        in_order_traversal(root);
        std::cout << std::endl;
    }
};

#endif // SPLAYTREE_H