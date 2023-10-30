//
// Created by idk on 2023/10/25.
//
#include <stdio.h>

#ifndef NUTN_IDK_LIST_H
#define NUTN_IDK_LIST_H



typedef struct listNode {
    int data;
    struct listNode *prev;
    struct listNode *next;
} listNode, *listNodePtr;

typedef struct linkedList {
    listNodePtr head;
    listNodePtr tail;
} list, *listPtr;

void freeListNode(listNodePtr ptr) {

}

void insertListNode(listNodePtr node, listNodePtr insetNode) {
    insetNode->prev = node;
    insetNode->next = node->next;

    if (node->next != NULL) {
        node->next->prev = insetNode;
    }

    node->next = insetNode;
}

void deleteListNode(listPtr L, listNodePtr node) {
    if (node->prev != NULL) {
        node->prev->next = node->next;
    }
    else {
        L->head = node->next;
    }

    if (node->next != NULL) {
        node->next->prev = node.prev;
    }
    freeListNode(node);
}



#endif //NUTN_IDK_LIST_H
