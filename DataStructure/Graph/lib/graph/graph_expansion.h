//
// Created by idk on 2023/12/24.
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

    // create vertex
    for (graph_size i = 0; i < vertex_num; i++)
    {
        graph_add_vertex(gh);
    }


    // decode array
    for (int vertex_index = 0; vertex_index < vertex_num; vertex_index++)
    {
        // get how many node connect by edge
        graph_size range = array[vertex_index + 1] - array[vertex_index];

        // get start index
        graph_size start_index = array[vertex_index];


        struct graph_vertex *u = graph_get_vertex_by_id(gh, vertex_index);

        // for each node
        for (graph_size node_index = start_index; node_index < start_index + range; node_index++)
        {
            struct graph_vertex *v = graph_get_vertex_by_id(gh, array[node_index]);

            graph_add_edge(gh, u, v);
        }

    }

    // return result
    return gh;
}



#endif //IDK_GRAPH_EXPANSION_H
