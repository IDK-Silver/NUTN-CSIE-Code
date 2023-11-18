#include <list/list.h>
#include <algorithm/ExpressionConverter.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    // show message
    printf("Enter Your Expression, Ex : (10^2)+(1/0.3)+21+30\nExpression :");

    // the user input
    char input_char = 0;

    // input char list
    struct list_head list = {0};

    // to for each element of list
    struct list_head *pos, *n;

    // init list
    INIT_LIST_HEAD(&list);

    // to counting pair
    unsigned left_pair_count = 0, right_pari_count = 0;

    // it will calculate user input math equation unit press enter
    while (scanf("%c", &input_char)) {

        // the new input char
        struct list_node_char *node = (struct list_node_char*)malloc(sizeof(struct list_node_char));

        // user press enter, out the wille loop
        if (input_char == '\n')
        {
            node->data = EedOfNumber;
            list_add_tail(&node->list, &list);
            break;
        }

        // counting pair
        if (input_char == LeftPare)
            left_pair_count++;

        else if (input_char == RightPare)
            right_pari_count++;

        // if input char is operator than add end of number to list
        if (is_math_operator(input_char) && input_char != LeftPare && input_char != RightPare)
        {
            node->data = EedOfNumber;
            list_add_tail(&node->list, &list);
        }

        // if user input is either number or operation, print the waring message and exit program
        if (!is_math_number(input_char) && !is_math_operator(input_char)  && input_char != ' ')
        {
            // print message
            fprintf(stderr, "Warring input is not a number or valid operator.");
            exit(0);
        }

        // add to list
        node = (struct list_node_char*)malloc(sizeof(struct list_node_char));
        node->data = input_char;
        list_add_tail(&node->list, &list);
    }


    // if left pair != right pari it mean the math expression is not valid
    if (left_pair_count != right_pari_count) {
        fprintf(stderr, "The expression is not valid\ncheck the right pair and left pair.");
        exit(0);
    }


    // the postfix format list
    struct list_head postfix_list = {0};

    // init list
    INIT_LIST_HEAD(&postfix_list);

    // infix to postfix
    infix_to_postfix(&list, &postfix_list);


    // print ans
    printf("Result : %lf\n", postfix_to_value(&postfix_list));


    // exit
     printf("Enter any to exit.");
    char exit_char = 0;
    scanf("%c", &exit_char);

    // free list
    list_for_each_safe(pos, n, &list)
    {
        struct list_node_char *st = list_entry(pos,struct list_node_char,list);
        list_del(pos);
        free(st);
    }

    // free postfix
    list_for_each_safe(pos, n, &postfix_list)
    {
        struct list_node_char *st = list_entry(pos,struct list_node_char,list);
        list_del(pos);
        free(st);
    }

}
