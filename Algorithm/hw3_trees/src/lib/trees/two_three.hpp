// TwoThreeTree.hpp
#ifndef TWOTHREETREE_HPP
#define TWOTHREETREE_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>

// Define the TwoThreeTree class template
template <typename T>
class TwoThreeTree
{
private:
    struct Node
    {
        Node *parent;
        std::vector<T> datas;
        std::vector<Node *> childs;

        Node()
        {
            parent = nullptr;
            datas.clear();
            childs.clear();
        }

        void insert(const T element)
        {
            datas.push_back(element);
            size_t i = datas.size() - 1;
            while (i > 0 && datas[i] < datas[i - 1])
                std::swap(datas[i], datas[i - 1]), i--;
        }

        void insert(Node *node)
        {
            childs.push_back(node);
            size_t i = childs.size() - 1;
            while (i > 0 && childs[i]->datas[0] <= childs[i - 1]->datas[0])
            {
                std::swap(childs[i], childs[i - 1]);
                i--;
            }
        }

        void delete_tree(const T element)
        {
            for (size_t i = 0; i < datas.size() - 1; i++)
                if (datas[i] == element)
                    std::swap(datas[i], datas[i + 1]);
            datas.pop_back();
        }

        void delete_tree(Node *node)
        {
            for (size_t i = 0; i < childs.size() - 1; i++)
                if (childs[i] == node)
                    std::swap(childs[i], childs[i + 1]);
            childs.pop_back();
        }

        void pull()
        {
            Node *a = new Node, *b = new Node;
            a->insert(datas[0]), b->insert(datas[2]);
            if (!childs.empty())
            {
                a->insert(childs[0]), b->insert(childs[2]);
                a->insert(childs[1]), b->insert(childs[3]);
                childs[0]->parent = childs[1]->parent = a;
                childs[2]->parent = childs[3]->parent = b;
            }
            if (parent == nullptr)
            {
                parent = new Node();
                parent->insert(this);
            }
            parent->insert(datas[1]);
            parent->delete_tree(this);
            parent->insert(a);
            parent->insert(b);
            a->parent = b->parent = parent;
        }

        void push()
        {
            Node *a, *b;
            int num;
            for (int i = 0; i < childs.size(); i++)
            {
                if (childs[i]->datas.size() == 0)
                {
                    a = childs[i];
                    if (i == childs.size() - 1)
                    {
                        b = childs[i - 1];
                        num = datas[i - 1];
                    }
                    else
                    {
                        b = childs[i + 1];
                        num = datas[i];
                    }
                }
            }
            if (a->childs.size() != 0)
                b->insert(a->childs[0]), a->childs[0]->parent = b;
            b->insert(num), delete_tree(num);
            delete_tree(a);
            if (b->datas.size() == 3)
                b->pull();
        }

    }
    *root = new Node;
public:
    void insert_node(T k, Node *cur)
    {
        if (cur->childs.empty())
        {
            cur->insert(k);
            while (cur && cur->datas.size() == 3)
            {
                cur->pull();
                Node *tmp = cur;
                cur = cur->parent;
                delete tmp;
            }
            if (cur->parent == nullptr)
                root = cur;
            return;
        }
        for (int i = 0; i < cur->datas.size(); i++)
        {
            if (k <= cur->datas[i])
            {
                insert_node(k, cur->childs[i]);
                return;
            }
        }
        insert_node(k, cur->childs[cur->datas.size()]);
    }
    void insert_node(T k) { insert_node(k, root); }

    Node *search_node(T k, Node *cur)
    {
        for (int i = 0; i < cur->datas.size(); i++)
        {
            if (cur->datas[i] == k)
                return cur;
        }
        if (cur->childs.empty())
            return nullptr;
        for (int i = 0; i < cur->datas.size(); i++)
        {
            if (k <= cur->datas[i])
                return search_node(k, cur->childs[i]);
        }
        return search_node(k, cur->childs[cur->datas.size()]);
    }
    bool search_node(T k) { return search_node(k, root) != nullptr; }

    void Delete(T k)
    {
        Node *cur = search_node(k);
        if (cur == nullptr)
            return;
        if (cur->childs.size() != 0)
        {
            int i = 0;
            for (; i < cur->datas.size(); i++)
                if (cur->datas[i] == k)
                    break;
            Node *tmp = cur->childs[i + 1];
            while (!tmp->childs.empty())
                tmp = tmp->childs[0];
            cur->delete_tree(k), cur->insert(tmp->datas[0]), tmp->delete_tree(tmp->datas[0]);
            cur = tmp;
        }
        else
            cur->delete_tree(k);
        while (cur->parent != nullptr && cur->datas.empty())
        {
            cur = cur->parent, cur->push();
        }
        if (cur->parent == nullptr && cur->datas.empty())
        {
            root = cur->childs[0];
            cur->childs[0]->parent = nullptr;
            delete cur;
        }
    }

    TwoThreeTree() {};
    ~TwoThreeTree() {};
};

#endif // TWOTHREETREE_HPP