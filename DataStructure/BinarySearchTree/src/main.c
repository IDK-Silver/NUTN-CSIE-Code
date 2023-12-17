/*
 * !!!!! Note :
 *     the compilation must be done under Linux or using Msys2;
 *     otherwise, comments and modifications are required to disable the parts
 *     that utilize the 'access' function, unistd lib
 */

#include <stdio.h>
#include <string.h>
#include <bst/bst.h>
#include <bst/bst_expansion.h>
#include <stdbool.h>
#include <unistd.h>


#define MaxInputSize 256

/**
 * the binary search tree with double data
 */
struct bst_node_double {
    double data;
    struct bst_head bst;
};


void check_file_path(char *file_path);

void bst_node_double_show_info(struct bst_head *node) {

    if (node == NULL) {
        printf("node is NULL\n");
    }

    double i = bst_entry(node, struct bst_node_double, bst)->data;
    double  p = -1, r = -1, l = -1;
    if (node->parent != NULL)
        p = bst_entry(node->parent, struct bst_node_double, bst)->data;

    if (node->left != NULL)
        l = bst_entry(node->left, struct bst_node_double, bst)->data;


    if (node->right != NULL)
        r = bst_entry(node->right, struct bst_node_double, bst)->data;


    printf("Node %f\t Parent is %f\t L : %.0f \t R : %.0f\n", i, p, l, r);
}


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
            {
                *file_path = argv[++i];
            }

            // if out range print error message
            else
            {
                fprintf(stderr, "argument input error\n");
                exit(0);
            }
        }
    }

    /* check argument can use in program */

}


/**
 * the function to check argument can use in program
 * @file_path: the path of load file, the variable may set by argument
 *
 */
void check_file_path(char *file_path)
{
    // not set file_path by program argument
    if (file_path == NULL)
    {
        fprintf(stderr, "Not valid argument\n");
    }

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
        fprintf(stderr, "the file is not exist in %s\n", file_path);
        exit(0);
    }

}

void inorder(struct bst_head *node)
{
    if (node == NULL)
        return;

    inorder(node->left);
    printf("%.2lf\n", bst_entry(node, struct bst_node_double, bst)->data);
    inorder(node->right);
}

void import_file(char *file_path, struct bst_head **root)
{
    // check file path
    check_file_path(file_path);
    FILE *file;

    file = fopen(file_path, "r");
    if (file == NULL) {
        fprintf(stderr, "can't open file %s\n", file_path);
        exit(0);
    }

    double data = 0;
    while(fscanf(file, "%lf", &data) != EOF)
    {
        struct bst_node_double *node = (struct bst_node_double*) malloc(sizeof(struct bst_node_double));
        node->data = data;
        INIT_BST_HEAD(node->bst);
        bst_insert(&(node->bst), root, bst_node_double_compare);
    }

    fclose(file);
    printf("import file successful\n");
}

void insert_node(struct bst_head **root)
{
    char buffer[MaxInputSize] = {0};
    char input = 0;
    int current_index = 0;

    // clear blank char
    while (1)
    {
        scanf("%c", &input);
        if (input != ' ')
        {
            buffer[0] = input;
            current_index++;
            break;
        }
    } ;

    // for each char
    while (scanf("%c", &input))
    {

        // end of the number than convertor char array to number
        if (input == ' ' || input == '\n')
        {
            struct bst_node_double *node = (struct bst_node_double*) malloc(sizeof(struct bst_node_double));
            buffer[current_index] = '\n';
            node->data = atof(buffer);
            INIT_BST_HEAD(node->bst);
            bst_insert(&(node->bst), root, bst_node_double_compare);
            current_index = 0;

            if (input == '\n')
            {
                printf("insert node successful\n");
                return;
            }
        }
        else
        {
            buffer[current_index] = input;
            current_index++;
        }
    }


}

void exist(struct bst_head *root)
{
    double data = 0;
    scanf("%lf", &data);

    struct bst_node_double *node = (struct bst_node_double*) malloc(sizeof(struct bst_node_double));
    node->data = data;


    bst_search(&(node->bst), root, bst_node_double_compare);

    struct bst_head *find = bst_search(&(node->bst), root, bst_node_double_compare);
    free(node);

    if (find != NULL)
        printf("exist\n");
    else
        printf("not exist\n");
}

void rank(struct bst_head *root)
{
    unsigned int index = 0;
    scanf("%d", &index);

    struct bst_head *node = bst_minimum_by_index(index, root);

    if (node == NULL) {
        printf("out of range\n");
        return;
    }

    printf("The value is : %f\n", bst_entry(node, struct bst_node_double, bst)->data);

}


void delete(struct bst_head **root)
{
    double data = 0;
    scanf("%lf", &data);

    struct bst_node_double *node = (struct bst_node_double*) malloc(sizeof(struct bst_node_double));
    node->data = data;
    INIT_BST_HEAD(node->bst);

    struct bst_head *find = bst_search(&(node->bst), *root, bst_node_double_compare);
    free(node);

    if (find != NULL)
    {
        bst_delete(find, root);
        free(bst_entry(find, struct bst_node_double, bst));
        printf("delete node successful\n");
    }
    else
        printf("not exist\n");
}


void help()
{
    printf("\'insert\' can insert value ex : insert 66\n");
    printf("\'delete\' can delete node by give value ex : delete 66\n");
    printf("\'show\' inorder traversal the tree ex : show\n");
    printf("\'rank\' find the i-th smallest node value ,i begin 0. ex : rank 0\n");
    printf("\'find\' check whether the node with the value provided by the user exists in the tree or not ex : find 66\n");
    printf("\'import\' import the txt file ex : import kano_is_cute.txt\n");
    printf("\'exit\' exit the program ex : exit\n");
    printf("\'help\' show command ex : help\n");
}

int main(int argc, char *argv[])
{

    // create tree
    struct bst_head *root = NULL;

    /* the path of load file */
    char *file_path = NULL;

    /* decode argument and get value  */
    decode_argument(argc, argv, &file_path);

    /* import file by argument */
    if (file_path != NULL)
        import_file(file_path, &root);


    // create variable storage file path
    file_path = (char *) malloc(sizeof(char) * MaxInputSize);

    // create variable storage command
    char *command = (char *) malloc(sizeof(char) * MaxInputSize);

    help();


    while (true)
    {
        // read command
        scanf("%s", command);

        if (strcmp(command, "help") == 0)
            help();
        // insert node command
        else if (strcmp(command, "insert") == 0 || strcmp(command, "i") == 0)
            insert_node(&root);

        // import file command
        else if (strcmp(command, "import") == 0)
        {
            scanf("%s", file_path);
            import_file(file_path, &root);
        }

        // find node command
        else if (strcmp(command, "exist") == 0 || strcmp(command, "find") == 0)
            exist(root);

        // find rank node command
        else if (strcmp(command, "rank") == 0)
            rank(root);

        // delete node command
        else if (strcmp(command, "delete") == 0)
            delete(&root);

        // inorder display
        else if (strcmp(command, "inorder") == 0 || strcmp(command, "show") == 0)
            inorder(root);

        // exit command
        else if (strcmp(command, "exit") == 0)
        {
            printf("exit program\n");
            break;
        }

        else
        {
            printf("command not found : %s\n", command);
        }


    }

    return 0;
}



