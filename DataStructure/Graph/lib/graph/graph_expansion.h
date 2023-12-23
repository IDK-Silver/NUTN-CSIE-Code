//
// Created by idk on 2023/12/21.
//

#ifndef IDK_GRAPH_EXPANSION_H
#define IDK_GRAPH_EXPANSION_H

#include "graph.h"

struct graph* graph_create_by_adjacency_list_array(const graph_size *array, graph_size size)
{

    struct list_head *pos;

    // init graph
    struct graph *gh = (struct graph *) malloc(sizeof(struct graph));
    INIT_GRAPH(gh);


    // if it empties array return null
    if (size == 0)
        return NULL;

    // get vertex number
    graph_size vertex_num = array[0] - 1;

    // create adjacency list
    gh->vertex_num = vertex_num;



    // decode array
    for (int vertex_index = 0; vertex_index < vertex_num; vertex_index++)
    {
        // get how many node connect by edge
        graph_size range = array[vertex_index + 1] - array[vertex_index];

        // get start index
        graph_size start_index = array[vertex_index];

        // to storage each vertex
        struct graph_vertex_list *vertex_list = (struct graph_vertex_list*) malloc(sizeof(struct graph_vertex_list));
        INIT_GRAPH_VERTEX_LIST(vertex_list);

        // for each node
        for (graph_size node_index = start_index; node_index < start_index + range; node_index++)
        {
            // create vertex
            struct graph_vertex_head *vertex = (struct graph_vertex_head*) malloc(sizeof(struct graph_vertex_head));
            INIT_GRAPH_VERTEX_HEAD(vertex);

            // set vertex node
            vertex->id = array[node_index];

            // add vertex
            list_add_tail(&(vertex->list), &vertex_list->data);
        }

        INIT_LIST_HEAD(&vertex_list->list);


        // add vertex list to adjacency list
        list_add_tail(&vertex_list->list, &gh->adjacency_list);
    }


//            list_entry(&gh->adjacency_list, struct graph_vertex_list, list);


//        vertex_list = list_first_entry(pos, struct graph_vertex_list, list);


//        printf("%p \n", vertex_list);


//        struct graph_vertex_head *h = list_entry(&vertex_list->data, struct graph_vertex_head, list);
//        printf("%p", h);


//        printf("%d \n", h->id);
//        list_del(pos);
//        struct list_head *vertex = pos;
//        list_del(pos);
//        list_move_tail(&vertex_list.list, pos);
//        list_add_tail(vertex, &vertex_list.list);



    // return result
    return gh;
}


#endif //IDK_GRAPH_EXPANSION_H
