#ifndef YUFENG_VIRTUAL_TREE_H
#define YUFENG_VIRTUAL_TREE_H

#include <iostream>

template <typename T>
class VirtualTree
{
private:
    struct Node
    {
    };

public:

    virtual void insert_node(const T &value);
    virtual bool search_node(T value);
};

#endif // YUFENG_VIRTUAL_TREE_H