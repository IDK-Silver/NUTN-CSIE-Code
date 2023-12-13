#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "lib/bst/bst.h"

/**
 * the binary search tree with double data
 */
struct bst_node_double {
    double data;
    struct bst_head bst;
};



/**
 * the function to compare bst_node_double entry by they data
 * @x: the first need compare entry
 * @y: the second need compare entry
 *
 * @return :
 *      -1: if the value of first is less than the value of the second.
 *       0 - if the value of first is equal than the value of the second.
 *      +1 - if the value of first is more than the value of the second.
 */
int bst_node_double_compare(struct bst_head *x, struct bst_head *y)
{
    // the entry value
    double l = bst_entry(x, struct bst_node_double, bst)->data;
    double r = bst_entry(y, struct bst_node_double, bst)->data;

    // compare code
    if (l == r)
        return 0;

    else if (l > r)
        return 1;

    return -1;
}




/**
 * the function to decode program argument
 * @argc: the number of argument
 * @argv: array of argument
 * @file_path: the path of load file, the variable may set by argument
 *
 */
void decode_argument(int argc, char *argv[], char **file_path)
{
    // read argument
    for (int i = 0; i < argc; i++)
    {
        char *argument = argv[i];

        // the argument that file_path
        if (strcmp(argument, "-f") == 0)
        {
            // check index range
            if (i + 1 < argc)
                *file_path = argv[++i];

            // if out range print error message
            else
            {
                fprintf(stderr, "argument input error\n");
                exit(0);
            }
        }
    }
}


/**
 * the function to check argument can use in program
 * @file_path: the path of load file, the variable may set by argument
 *
 */
void program_argument_check_and_set(char **file_path)
{
    // not set file_path by program argument
    if (*file_path == NULL)
    {
        printf("Input file path : ");
        scanf("%s", *file_path);
    }

    // check file exists
    if (strlen(*file_path) > 0 && access(*file_path, F_OK) == 0)
    {
        /* check file extensions */
        char *extension = strrchr(*file_path, '.');
        if (extension == NULL || strcmp(extension + 1, "txt") != 0)
        {
            fprintf(stderr, "the file extension is not valid\n");
            exit(0);
        }
    }
    // file is not exist
    else
    {
        fprintf(stderr, "the file is not exist in %s\n", *file_path);
        exit(0);
    }

}

int main(int argc, char *argv[])
{
    /* the path of load file */
    char *file_path = NULL;

    /* decode argument and get value  */
    decode_argument(argc, argv, &file_path);

    /* check argument can use in program */
    program_argument_check_and_set(&file_path);

    /* */
    if (file_path == NULL) {
        printf("%s", file_path);
    }

    // create and init the binary search tree
    struct bst_head *root = NULL;
    INIT_BST_HEAD(root);

    struct bst_node_double *d1 = (struct bst_node_double*)malloc(sizeof(struct bst_node_double));
    d1->data = 5;

    struct bst_node_double *d2 = (struct bst_node_double*)malloc(sizeof(struct bst_node_double));
    d2->data = 3;

    struct bst_node_double *d3 = (struct bst_node_double*)malloc(sizeof(struct bst_node_double));
    d3->data = 2;

    struct bst_node_double *d4 = (struct bst_node_double*)malloc(sizeof(struct bst_node_double));
    d4->data = 7;


    bst_insert(&(d1->bst), root, bst_node_double_compare);
    bst_insert(&(d2->bst), root, bst_node_double_compare);
    bst_insert(&(d3->bst), root, bst_node_double_compare);
    bst_insert(&(d4->bst), root, bst_node_double_compare);

//    inorder_tree_walk(root);

//    printf("Hello, World!\n");
    return 0;
}


void inorder_tree_walk(struct bst_head *x) {
    if (x != NULL)
    {
        inorder_tree_walk(x->left);
        double i = bst_entry(x, struct bst_node_double, bst)->data;
        double  p = -1, r = -1, l = -1;
        if (x->parent != NULL)
            p = bst_entry(x->parent, struct bst_node_double, bst)->data;

        if (x->left != NULL)
            l = bst_entry(x->left, struct bst_node_double, bst)->data;

        if (x->right != NULL)
            r = bst_entry(x->right, struct bst_node_double, bst)->data;


        printf("Node %f\t Parent is %f\t L : %.0f \t R : %.0f\n", i, p, l, r);


        inorder_tree_walk(x->right);
    }
}