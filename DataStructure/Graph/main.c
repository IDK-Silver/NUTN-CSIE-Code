#include <stdio.h>
#include <stdlib.h>
#include <list/list.h>
#include <string.h>
#include <unistd.h>
#include <graph/graph.h>
#include <graph/graph_expansion.h>

/**
 * to list node that to storage char type
 * @data: char value
 * @list: the list need variable
 */
struct list_char_node {
    char data;
    struct list_head list;
};

/**
 * to list node that to storage graph size type
 * @data: graph size value
 * @list: the list need variable
 */
struct list_graph_size_node {
    graph_size data;
    struct list_head list;
};

/**
 * ask user input testfile path
 * @param file_path_ptr  pointer to the file path
 * @param file_size      pointer to file path size
 */
void get_file_path_by_user(char **file_path_ptr, uint16_t *file_size)
{
    char *file_path = NULL;
    uint16_t file_path_size = 0;
    // list iter pointer
    struct list_head *n, *pos;

    // create a list for storage file path
    struct list_head *file_path_list = (struct list_head *) malloc(sizeof(struct list_head));
    INIT_LIST_HEAD(file_path_list);

    // display a message to notice user input file path
    printf("Input Graph adjacency list array file path : ");
    char input_char = 0;

    // storage each input char to list
    while (scanf("%c", &input_char)) {
        // exist condition
        if (input_char == '\n')
            break;

        // create list node
        struct list_char_node *node = (struct list_char_node *) malloc(sizeof(struct list_char_node));
        INIT_LIST_HEAD(&node->list);
        node->data = input_char;

        // add to a file path list
        list_add_tail(&node->list, file_path_list);
        file_path_size++;
    }

    // malloc with file path size
    file_path = (char *) malloc(sizeof(char) * file_path_size + 1);

    // reset file path size for file path index
    file_path_size = 0;

    // for each file path list copy to file a path array
    list_for_each_safe(pos, n, file_path_list) {
        // get node
        struct list_char_node *node = list_entry(pos, struct list_char_node, list);

        // copy data to file's path
        file_path[file_path_size++] = node->data;

        // free node
        list_del(pos);
        free(node);
    }

    // add an end of string
    file_path[file_path_size + 1] = '\0';

    // free file path list
    free(file_path_list);

    // check file exists
    if (strlen(file_path) > 0 && access(file_path, F_OK) == 0)
    {
        /* check file extensions */
        char *extension = strrchr(file_path, '.');
        if (extension == NULL || strcmp(extension + 1, "txt") != 0)
        {
            fprintf(stderr, "the file extension is not valid\n");
            exit(0);
        }
    }
    // file does not exist
    else
    {
        printf("the file not exist in %s\n", file_path);
        exit(0);
    }

    // move data to input pointer
    *file_path_ptr = file_path;
    *file_size = file_path_size;
}

/**
 * read the file that user input and storage to array
 * @param file_path  the file path
 * @param array      a pointer to the result array
 * @param size       the file path size
 */
void get_adjacency_list_array(char *file_path, graph_size **array, graph_size *size)
{
    /* for list each */
    struct list_head *pos, *n;

    /* the result array */
    graph_size *result = NULL;

    /* to storage user input */
    struct list_head input_list;
    INIT_LIST_HEAD(&input_list);

    // result array size
    graph_size array_size = 0;

    // read file
    FILE *file = NULL;
    file = fopen(file_path, "r");
    if (file == NULL) {
        fprintf(stderr, "can't open file %s\n", file_path);
        exit(0);
    }

    // read each file data
    graph_size graph_id = 0;
    while(fscanf(file, "%u", &graph_id) != EOF)
    {
        // storage to list
        struct list_graph_size_node *node = (struct list_graph_size_node*) malloc(sizeof(struct list_graph_size_node));
        node->data = graph_id;
        list_add_tail(&node->list, &input_list);

        // array size += 1
        array_size++;
    }

    // create a result array by input size
    result = (graph_size *) malloc(sizeof(graph_size) * array_size);
    array_size = 0;

    // for each list node copy to result array
    list_for_each_safe(pos, n, &input_list) {
        // get node
        struct list_graph_size_node *node = list_entry(pos, struct list_graph_size_node, list);

        // copy data to result
        result[array_size++] = node->data;

        // free node
        list_del(pos);
        free(node);
    }

    // close file
    fclose(file);

    // move data to input pointer
    *array = result;
    *size = array_size;
}

/**
 * Well, this is the BFS algorithm...
 * @param gh the graph
 * @param s the start vertex
 */
void bfs(struct graph *gh, struct graph_vertex *s)
{
    // print message
    printf("BFS : ");

    // init each vertex info
    struct graph_vertex *vertex = NULL;
    graph_for_vertex_in_graph(vertex, gh)
    {
        vertex->color = WHITE;
        if (vertex->distance != NULL)
            free(vertex->distance);
    }

    // change start vertex info
    s->color = GRAY;
    s->distance = (graph_size *) malloc(sizeof(graph_size));
    *s->distance = 0;

    // make a queue to storage the node
    struct list_head queue = {0};
    INIT_LIST_HEAD(&queue);

    // add start node to queue
    struct list_graph_size_node *node = (struct list_graph_size_node *) malloc(sizeof(struct  list_graph_size_node));
    node->data = s->id;
    list_add_tail(&node->list, &queue);

    // if the queue not empty, it means the bfs visit is not an end
    while (list_empty(&queue) != 1)
    {
        // get the first element in queue
        struct list_graph_size_node* f_q = list_first_entry(&queue, struct list_graph_size_node, list);
        list_del(&f_q->list);

        // get vertex by f_q
        struct graph_vertex *u = graph_get_vertex_by_id(gh, f_q->data);

        // free f_q
        free(f_q);

        // the vertex that u connect vertex
        struct graph_vertex *v = NULL;
        struct graph_vertex_ptr *v_ptr = NULL;

        // get the u's adjacency
        struct graph_vertex_ptr_list *adjacency = graph_get_adjacency_by_vertex(gh, u);

        // for each vertex in u's adjacency
        graph_for_vertex_ptr_in_vertex_ptr_list(v_ptr, adjacency)
        {
            // get v by v_ptr (by the lib of graph ask)
            // I don't have enough time to design this library, so the design should be simplified._QQ
            v = v_ptr->ptr;

            // if not, add to the queue
            if (v->color == WHITE)
            {
                v->color = GRAY;
                if (v->distance == NULL)
                {
                    v->distance = (graph_size *) malloc(sizeof(graph_size));
                    *v->distance = 0;
                }
                *v->distance += 1;

                node = (struct list_graph_size_node *) malloc(sizeof(struct list_graph_size_node));
                node->data = v->id;

                list_add_tail(&node->list, &queue);
            }
        }

        // end of a visit
        u->color = BLACK;
        printf("%u ", u->id);
    }
    printf("\n");
}


/**
 * Well, this is the part of DFS algorithm, to visit by part of vertex
 * @param gh the graph
 * @param u the part of vertex
 */
void dfs_visit(struct graph *gh, struct graph_vertex *u)
{
    // to storage thr dfs time, another mean is order of a visit
    static graph_size dfs_time;
    dfs_time += 1;

    // check and change the vertex distance
    if (u->distance == NULL)
        u->distance = (graph_size *) malloc(sizeof(graph_size));
    *u->distance = dfs_time;

    // mark the status
    u->color = GRAY;
    struct graph_vertex *v = NULL;
    struct graph_vertex_ptr *v_ptr = NULL;
    graph_for_vertex_ptr_in_vertex_ptr_list(v_ptr, graph_get_adjacency_by_vertex(gh, u))
    {
        v = v_ptr->ptr;

        if (v->color == WHITE)
            dfs_visit(gh, v);
    }

    dfs_time += 1;
    printf("%u ", u->id);
}

void dfs(struct graph *gh)
{
    static graph_size dfs_time = 0;
    printf("DFS : ");

    struct graph_vertex *vertex = NULL;

    // init all vertexes in graph
    graph_for_vertex_in_graph(vertex, gh)
    {
        vertex->color = WHITE;
        if (vertex->distance != NULL)
            free(vertex->distance);
    }

    // dfs visit all vertex
    graph_for_vertex_in_graph(vertex, gh)
    {
        if (vertex->color == WHITE)
            dfs_visit(gh, vertex);
    }

}



int main()
{

    // file a path with array with c style
    char *file_path = NULL;
    uint16_t file_path_size = 0;

    // get a file path
    get_file_path_by_user(&file_path, &file_path_size);

    // get adjacency list array by file path
    graph_size *adjacency_list_array = NULL;
    graph_size adjacency_list_array_size = 0;

    // get adjacency list array by file path
    get_adjacency_list_array(file_path, &adjacency_list_array, &adjacency_list_array_size);

    // create graph by decode adjacency list array
    struct graph *gh = graph_create_by_adjacency_list_array(adjacency_list_array, adjacency_list_array_size);

    // get the first vertex in graph
    struct graph_vertex *start_vertex = list_first_entry(&gh->vertex_list, struct graph_vertex, list);
    
    // bfs
    bfs(gh, start_vertex);

    // dfs
    dfs(gh);

    return 0;
}
