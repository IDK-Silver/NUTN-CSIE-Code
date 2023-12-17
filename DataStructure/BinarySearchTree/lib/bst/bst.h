//
// Created by idk on 2023/12/13.
//

#ifndef BINARY_SEARCH_TREE_H
#define BINARY_SEARCH_TREE_H

#include <stdlib.h>

/**
 * @bst_head: binary search tree head pointer
 */
struct bst_head {
    struct bst_head* left, *right, *parent;
};

/**
 * INIT_BST_HEAD - to init tree root
 * @root: tree root pointer.
 */
#define INIT_BST_HEAD(root) do {   \
    root.left = NULL;              \
    root.right = NULL;             \
    root.parent = NULL;            \
}   while(0)


/**
 * bst_entry - get the struct for this entry
 * @ptr:	the &struct list_head pointer.
 * @type:	the type of the struct this is embedded in.
 * @member:	the name of the list_struct within the struct.
 */
#define bst_entry(ptr, type, member) \
	((type *)((char *)(ptr)-(unsigned long)(&((type *)0)->member)))


/**
 * bst insert - insert a new entry to binary tree that belong to root
 * @ptr: new entry to be insert
 * @root: tree root
 * @compare_func: compare entry function
 */
void bst_insert(struct bst_head *ptr, struct bst_head **root,
                int (*compare_func)(struct bst_head*, struct bst_head*)) {
    struct bst_head *_x = *root;
    struct bst_head *_y = NULL;
    struct bst_head *_z = ptr;
    while (_x != NULL)
    {
        _y = _x;
        if (compare_func(ptr, _x) == -1)
            _x = _x->left;
        else
            _x = _x->right;
    }
    _z->parent = _y;
    if (_y == NULL)
        *root = _z;
    else if (compare_func(_z, _y) == -1)
        _y->left = _z;
    else
         _y->right = _z;
}


/**
 * bst search - search ptr in tree that belong to root
 * @ptr: want to find target
 * @root: tree root
 * @compare_func: compare entry function
 */
struct bst_head* bst_search(struct bst_head *ptr, struct bst_head *root,
                            int (*compare_func)(struct bst_head*, struct bst_head*))
{
    while (root != NULL && compare_func(ptr, root) != 0)
    {
        if (compare_func(ptr, root) == -1)
            root = root->left;
        else
            root = root->right;
    }
    return root;
}

/**
 * bst minimum - find the minimum entry in tree that belong to ptr
 * @ptr: the root
 * @return return minimum entry in ptr
 */
struct bst_head* bst_minimum(struct bst_head *ptr)
{
    while (ptr != NULL && ptr->left != NULL)
        ptr = ptr->left;
    return  ptr;
}

/**
 * bst bst_maximum - find the bst_maximum entry in tree that belong to ptr
 * @ptr: the root
 * @return return bst_maximum entry in ptr
 */
struct bst_head* bst_maximum(struct bst_head *ptr)
{
    while (ptr != NULL && ptr->right != NULL)
        ptr = ptr->right;
    return  ptr;
}

/**
 * bst successor - find the entry with the smallest key greater than ptr key
 * @ptr: the current node.
 */
struct bst_head* bst_successor(struct bst_head *ptr)
{
    if (ptr->right != NULL)
        return bst_minimum(ptr->right);
    else
    {
        struct bst_head* y = ptr->parent;
        while (y != NULL && ptr == y->right)
        {
            ptr = y;
            y = ptr->parent;
        }
        return  y;
    }
}

/**
 * bst predecessor - finds the entry with the largest key smaller than the ptr
 * @ptr: the current node.
 */
struct bst_head* bst_predecessor(struct  bst_head *ptr)
{
    if (ptr->left != NULL)
        return bst_maximum(ptr->left);
    else
    {
        struct bst_head *y = ptr->parent;
        while (y != NULL && ptr == y->left)
        {
            ptr = y;
            y = ptr->parent;
        }
        return  y;
    }
}

/**
 * bst transplant - replace u by v
 * @u: the subtree u
 * @v: the subtree v
 * @root: the tree root
 */
void bst_transplant(struct bst_head *u, struct  bst_head *v, struct bst_head **root)
{
    if (u->parent == NULL)
        *root = v;
    else if (u == u->parent->left)
        u->parent->left = v;
    else
        u->parent->right = v;

    if (v != NULL)
        v->parent = u->parent;
}

/**
 * bst_delete - delete ptr in root
 * @ptr: the entry that want to delete
 * @root: the tree root
 */
void bst_delete(struct bst_head *ptr, struct bst_head **root)
{
    if (ptr->left == NULL)
        bst_transplant(ptr, ptr->right, root);
    else if (ptr->right == NULL)
        bst_transplant(ptr, ptr->left, root);
    else
    {
        struct bst_head *y = bst_minimum(ptr->right);

        if (y != NULL)
        {
            bst_transplant(y, y->right, root);
            y->right = ptr->right;
            y->right->parent = y;
        }

        bst_transplant(ptr, y, root);
        y->left = ptr->left;
        y->left->parent = y;
    }
}


#endif //BINARY_SEARCH_TREE_H
