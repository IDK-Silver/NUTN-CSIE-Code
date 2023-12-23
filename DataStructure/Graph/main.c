#include <stdio.h>
#include <stdlib.h>
#include <list/list.h>
#include <inttypes.h>
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

void get_file_path_by_user(char **file_path_ptr, uint16_t *file_size)
{
    char *file_path = NULL;
    uint16_t file_path_size = 0;
    // list iter pointer
    struct list_head *n, *pos;

    // create list for storage file path
    struct list_head *file_path_list = (struct list_head *) malloc(sizeof(struct list_head));
    INIT_LIST_HEAD(file_path_list);

    // display message to notice user input file path
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

        // add to file path list
        list_add_tail(&node->list, file_path_list);
        file_path_size++;
    }

    // malloc with file path size
    file_path = (char *) malloc(sizeof(char) * file_path_size + 1);

    // reset file path size for file path index
    file_path_size = 0;

    // for each file path list copy to file path array
    list_for_each_safe(pos, n, file_path_list) {
        // get node
        struct list_char_node *node = list_entry(pos, struct list_char_node, list);

        // copy data to file path
        file_path[file_path_size++] = node->data;

        // free node
        list_del(pos);
        free(node);
    }

    // add end of string
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
    // file is not exist
    else
    {
        printf("the file not exist in %s\n", file_path);
        exit(0);
    }

    // move to input pointer
    *file_path_ptr = file_path;
    *file_size = file_path_size;
}

int main()
{

//    struct list_head *pos, * n;

    // file path with array with c style
    char *file_path = NULL;
    uint16_t file_path_size = 0;

    // get file path
//    get_file_path_by_user(&file_path, &file_path_size);

    graph_size array[] = {5, 8, 10, 13, 15, 2, 3, 4, 1, 3, 1, 2, 4,1, 3};

    struct graph *gp = graph_create_by_adjacency_list_array(array, 15);
    struct graph_vertex_head *pos;
    struct graph_vertex_list *vertex_list = NULL;

    struct graph_vertex_list *as = {0};

    graph_for_each_adjacency_list(vertex_list, gp->adjacency_list) {
        graph_for_vertex_list(pos, vertex_list) {
            printf("%u ", pos->id);
        }
        printf("\n");
    }



//    printf("%s", file_path);

    return 0;
}
