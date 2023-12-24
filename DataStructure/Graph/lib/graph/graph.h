//
// Created by idk on 2023/12/24.
//

#ifndef IDK_GRAPH_H
#define IDK_GRAPH_H

#include <inttypes.h>
#include <stdlib.h>
#include <list/list.h>

typedef uint32_t graph_size;

enum graph_color {
    WHITE,
    GRAY,
    BLACK
};

struct graph_vertex {
    graph_size id;
    graph_size *distance;
    enum graph_color color;
    struct list_head list;
};

void INIT_GRAPH_VERTEX(struct graph_vertex *ptr)
{
    ptr->id = 0;
    ptr->distance = NULL;
    ptr->color = WHITE;
}


struct graph_vertex_ptr {
    struct graph_vertex *ptr;
    struct list_head list;
};


void INIT_GRAPH_VERTEX_PTR(struct graph_vertex_ptr *ptr)
{
    ptr->ptr = NULL;
}

#define graph_vertex_list list_head


void INIT_GRAPH_VERTEX_LIST(struct graph_vertex_list *ptr)
{
    INIT_LIST_HEAD(ptr);
}

struct graph_vertex_ptr_list {
    struct list_head data;
    struct list_head list;
};

void INIT_GRAPH_VERTEX_PTR_LIST(struct graph_vertex_ptr_list *ptr)
{
    INIT_LIST_HEAD(&ptr->data);
}

struct graph {
    graph_size vertex_num;
    // storage by linked list
    struct list_head _adjacency_list;
    struct graph_vertex_list vertex_list;
};

void INIT_GRAPH(struct graph *ptr) {
    ptr->vertex_num = 0;
    INIT_LIST_HEAD(&ptr->_adjacency_list);
    INIT_GRAPH_VERTEX_LIST(&ptr->vertex_list);
}


struct graph_vertex* graph_get_vertex_by_id(struct graph *gh, graph_size id)
{
    struct list_head *vertex_head = &gh->vertex_list;

    for (graph_size i = 0; i < id; i++)
        vertex_head = vertex_head->next;

    return list_first_entry(vertex_head, struct graph_vertex, list);
}


void graph_add_vertex(struct graph *ptr)
{
    // create new vertex
    struct graph_vertex *vertex = (struct graph_vertex*) malloc(sizeof(struct graph_vertex));
    INIT_GRAPH_VERTEX(vertex);
    vertex->id = ptr->vertex_num;

    // graph vertex number += 1
    ptr->vertex_num += 1;
    list_add_tail(&vertex->list, &ptr->vertex_list);

    // add adjacency list
    struct graph_vertex_ptr_list *adjacency_list = (struct graph_vertex_ptr_list *) malloc(sizeof(struct graph_vertex_ptr_list));
    INIT_GRAPH_VERTEX_PTR_LIST(adjacency_list);

    // add adjacency list to graph
    list_add_tail(&adjacency_list->list, &ptr->_adjacency_list);
};

struct graph_vertex_ptr_list* graph_get_adjacency_by_vertex(struct graph *gh,  struct graph_vertex *u)
{
    struct list_head *adjacency_list_head = &gh->_adjacency_list;

    for (graph_size i = 0; i < u->id; i++)
        adjacency_list_head = adjacency_list_head->next;

    struct graph_vertex_ptr_list *adjacency_list = list_first_entry(adjacency_list_head, struct graph_vertex_ptr_list, list);
    return adjacency_list;
}


/**
 * connect two vertex, make (u) have a edge connect to (v)
 * @param ptr  the graph pointer
 * @param u    the target
 * @param v    the be connect
 */

void graph_add_edge(struct graph *ptr, struct graph_vertex *u, struct graph_vertex *v)
{
    struct graph_vertex_ptr_list *adjacency_list = graph_get_adjacency_by_vertex(ptr, u);

//    struct graph_vertex_ptr_list *adjacency_list = list_first_entry(adjacency_list_head, struct graph_vertex_ptr_list, list);

    struct graph_vertex_ptr *vertex_ptr = (struct  graph_vertex_ptr *) malloc(sizeof(struct graph_vertex_ptr));
    vertex_ptr->ptr = v;

    list_add_tail(&vertex_ptr->list, &adjacency_list->data);
}



#define graph_for_vertex_ptr_in_vertex_ptr_list(pos, head)				    \
	for (pos = list_entry((&head->data)->next, typeof(*pos), list),	    \
		     prefetch(pos->list.next);			                    \
	     &pos->list != (&head->data); 					                    \
	     pos = list_entry(pos->list.next, typeof(*pos), list),   	\
		     prefetch(pos->list.next))


#define graph_for_vertex_in_graph(pos, head)				    \
	for (pos = list_entry((&head->vertex_list)->next, typeof(*pos), list),	    \
		     prefetch(pos->list.next);			                    \
	     &pos->list != (&head->vertex_list); 					                    \
	     pos = list_entry(pos->list.next, typeof(*pos), list),   	\
		     prefetch(pos->list.next))




#endif //IDK_GRAPH_H


