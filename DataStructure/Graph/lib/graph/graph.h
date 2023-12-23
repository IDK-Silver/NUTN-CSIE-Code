//
// Created by idk on 2023/12/20.
//

#include <inttypes.h>
#include <stdlib.h>
#include <list/list.h>

#ifndef IDK_GRAPH_H
#define IDK_GRAPH_H

typedef int graph_size;

enum graph_color {
    WHITE,
    GRAY,
    BLACK
};



struct graph_vertex_head {
    graph_size id;
    graph_size *distance;
    enum graph_color color;
    struct list_head list;
};





void INIT_GRAPH_VERTEX_HEAD(struct graph_vertex_head *ptr)
{
    ptr->id = 0;
    ptr->distance = NULL;
    ptr->color = WHITE;
}

struct graph_vertex_list {
    struct list_head data;
    struct list_head list;
};

void INIT_GRAPH_VERTEX_LIST(struct graph_vertex_list *ptr)
{
    INIT_LIST_HEAD(&(ptr->data));
}

struct graph {
    graph_size vertex_num;

    // storage by linked list
    struct list_head adjacency_list;
};

void INIT_GRAPH(struct graph *ptr) {
    ptr->vertex_num = 0;
    INIT_LIST_HEAD(&ptr->adjacency_list);
}


/**
 * list_for_each_entry	-	iterate over list of given type
 * @pos:	the type * to use as a loop counter.
 * @head:	the head for your list.
 * @member:	the name of the list_struct within the struct.
 */
#define graph_for_each_adjacency_list(pos, head)				    \
	for (pos = list_entry((&head)->next, typeof(*pos), list),	    \
		     prefetch(pos->list.next);			                    \
	     &pos->list != (&head); 					                    \
	     pos = list_entry(pos->list.next, typeof(*pos), list),   	\
		     prefetch(pos->list.next))


/**
 * list_for_each_entry	-	iterate over list of given type
 * @pos:	the type * to use as a loop counter.
 * @head:	the head for your list.
 * @member:	the name of the list_struct within the struct.
 */
#define graph_for_vertex_list(pos, head)				    \
	for (pos = list_entry((&head->data)->next, typeof(*pos), list),	    \
		     prefetch(pos->list.next);			                    \
	     &pos->list != (&head->data); 					                    \
	     pos = list_entry(pos->list.next, typeof(*pos), list),   	\
		     prefetch(pos->list.next))


#endif //IDK_GRAPH_H
