//
// Created by idk on 2023/12/17.
//

#ifndef BINARY_SEARCH_TREE_H_EXPANSION_H
#define BINARY_SEARCH_TREE_H_EXPANSION_H

#include "bst.h"

struct bst_head* bst_minimum_by_index(unsigned int index, struct bst_head* root)
{
    struct bst_head *min = bst_minimum(root);
    for (int current = 0; current < index; current++) { min = bst_successor(min); }
    return min;
}

struct bst_head* bst_maximum_by_index(unsigned int index, struct bst_head* root)
{
    struct bst_head *max = bst_maximum(root);
    for (int current = 0; current < index; current++) { max = bst_predecessor(max); }
    return max;
}


#endif //BINARY_SEARCH_TREE_H_EXPANSION_H
