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
#define INIT_BST_HEAD(root) do { \
    \
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
#define bst_insert(ptr, root, compare_func) do {                                \
    struct bst_head *_x = root;                                                 \
    struct bst_head *_y = NULL;                                                 \
    struct bst_head *_z = ptr;                                                  \
    while (_x != NULL)                                                          \
    {                                                                           \
        _y = _x;                                                                \
        if (compare_func(ptr, _x) == -1)                                        \
            _x = _x->left;                                                      \
        else                                                                    \
            _x = _x->right;                                                     \
    }                                                                           \
    _z->parent = _y;                                                            \
    if (_y == NULL)                                                             \
        root = _z;                                                              \
    else if (compare_func(_z, _y) == -1)                                        \
        _y->left = _z;                                                          \
    else                                                                        \
         _y->right = _z;                                                        \
} while(0)



#endif //BINARY_SEARCH_TREE_H
