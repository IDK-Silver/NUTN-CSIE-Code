//
// Created by idk on 2023/11/7.
//

#ifndef CALCULATOR_EXPRESSION_CONVERTER_H
#define CALCULATOR_EXPRESSION_CONVERTER_H

#define CALCULATOR_BUFFER_SIZE 128

#include <list/list.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctype.h>
#include <math.h>

// char list struct
struct list_node_char {
    char data;
    struct list_head list;
};


struct  list_node_double {
    double data;
    struct list_head list;
};

// the math operator
enum MathOperator {
    Add = '+',
    Sub = '-',
    Mul = '*',
    Div = '/',
    Mod = '%',
    Pow = '^',
    LeftPare = '(',
    RightPare = ')',
    EedOfNumber = '@',
    UnDefined = -127,
};

bool is_math_operator(char alpha) {
    
    // check whether input alpha is operator
    switch (alpha) {
        case Add:
        case Sub:
        case Mul:
        case Div:
        case Mod:
        case Pow:
        case LeftPare:
        case RightPare:
            return true;
            
        default:
            return false;
    }

}

bool is_math_number(char  alpha) {
    if (!is_math_operator(alpha) && alpha != EedOfNumber && isdigit(alpha) || alpha == '.') {
        return true;
    }
    return  false;
}

// to get math operator priority
int get_math_operator_priority(enum MathOperator operator) {

    // to match each operator priority
    switch (operator) {

        case Add:
        case Sub:
            return 1;

        case Mul:
        case Div:
            return 2;

        case Mod:
        case Pow:
            return 3;

        case LeftPare:
        case RightPare:
            return 4;

        case EedOfNumber:
            return -1;

        default:
            // operator is not exists, exit program and print error message
            fprintf(stderr, "ExpressionConverter : unable to get math operator priority input operator is undefined");
            exit(-1);
    }
}

/* compare math operator priority
 *  if r > l return 1
 *  if r = l return 0
 *  if r < l return -1
 */
int compare_math_operator_priority(enum MathOperator r, enum MathOperator l) {

    // the priority of r
    int pr = get_math_operator_priority(r);

    // the priority of l
    int pl = get_math_operator_priority(l);

    // return compare result;
    return pr == pl ? 0 : pr > pl ? 1 : -1;

}

void infix_to_postfix(struct list_head* list, struct list_head* result_list) {
    // the operator stack list
    struct list_head operator_stack;

    // init list
    INIT_LIST_HEAD(&operator_stack);

    // the pos to each element
    struct list_head* pos = NULL, *n = NULL;

    // to storage input number it mean that each input must in 128 letter
    char buffer[CALCULATOR_BUFFER_SIZE] = {0};

    // for each node in input list
    list_for_each(pos, list)
    {
        // get node
        struct list_node_char *_node = list_entry(pos, struct list_node_char, list);

        struct list_node_char *node = (struct list_node_char*)malloc(sizeof(struct list_node_char));

        node->data = _node->data;


        // add alpha to result
        if (!is_math_operator(node->data)) {
            list_add_tail(&node->list, result_list);
        }

        // meet left pare than push operator stack
        else if (node->data == LeftPare || list_empty(&operator_stack))
        {
            list_add(&node->list, &operator_stack);
        }

        // stack is not empty
        else if (!list_empty(&operator_stack))
        {

            // get element that the top of stack
            struct list_node_char* stack_top_element = list_first_entry(&operator_stack, struct list_node_char, list);

            // match left and right pare, then push the operator of pare context to result
            if (node->data == RightPare)
            {

                // push the operator of pare context to result
                while (stack_top_element->data != LeftPare) {
                    list_move_tail(&stack_top_element->list, result_list);
                    stack_top_element = list_first_entry(&operator_stack, struct list_node_char, list);
                }

                // del right pare
                list_del(&stack_top_element->list);
                free(stack_top_element);
                free(node);
            }
            else if (stack_top_element->data == LeftPare) {
                list_add(&node->list, &operator_stack);
            }
            else
            {
                // compare operator priority
                switch (compare_math_operator_priority(node->data, stack_top_element->data))
                {

                    case -1:
                    case 0:
                        /// remove form stack, then add to result
                        list_move_tail(&stack_top_element->list, result_list);
                        list_add(&node->list, &operator_stack);
                        break;

                    case 1:
                        // add operator to stack
                        list_add(&node->list, &operator_stack);
                        break;
                }
            }
        }

    }

    // end of the input list , so push each operator of stack to result
    list_for_each_safe(pos, n, &operator_stack) {
        list_move_tail(pos, result_list);
    }
}



double postfix_to_value(struct list_head* list) {

    // the result stack
    struct list_head result_stack = {0};
    struct list_head *pos = NULL;

    // init list
    INIT_LIST_HEAD(&result_stack);

    // the char to number buffer it mean the number len can't more than CALCULATOR_BUFFER_SIZE -1
    char number_buffer[CALCULATOR_BUFFER_SIZE] = {0};

    // the counting buffer index
    int buffer_index = 0;

    // for each char element to convert to value
    list_for_each(pos, list) {

        // char node
        struct list_node_char* node = list_entry(pos, struct list_node_char, list);

        // is char is number push to buffer
        if (is_math_number(node->data)) {
            number_buffer[buffer_index++] = node->data;
        }

        // if not meet number, let buffer to number and push the number to result stack
        else if (buffer_index >= 1)
        {
            // set the end of buffer
            number_buffer[buffer_index] = '\0';
            char **ptr = NULL;

            // init buffer index
            buffer_index = 0;

            // create new stack node
            struct list_node_double *num_node = (struct list_node_double*)malloc(sizeof(struct list_node_double));

            // set node value to buffer number
            num_node->data = strtod(number_buffer, ptr);

            // push  number node to result stack
            list_add(&num_node->list, &result_stack);
        }

        // if meet operator to calculate value
        if (is_math_operator(node->data))
        {

            // get number from stack top
            double r = 0, l = 0, cal_result = 0;

            // the calculate result node
            struct list_node_double *num_node = {0};

            // get stack top element
            num_node = list_first_entry(&result_stack, struct list_node_double, list);
            r = num_node->data;

            // free num_node
            list_del(&num_node->list);
            free(num_node);

            // get stack top element
            num_node = list_first_entry(&result_stack, struct list_node_double, list);
            l = list_first_entry(&result_stack, struct list_node_double, list)->data;

            // free num_node
            list_del(&num_node->list);
            free(num_node);

            // create new num node
            num_node  = (struct list_node_double*)malloc(sizeof(struct list_node_double));


            // match each operation
            switch (node->data) {

                case Add:
                    cal_result = l + r;
                    break;

                case Sub:
                    cal_result = l - r;
                    break;

                case Mul:
                    cal_result = l * r;
                    break;

                case Div:

                    // if div by zero print to stderr to note user, exit the program
                    if (r == 0) {
                        fprintf(stderr, "Expression exists the problem : div by zero\ncheck your expression");
                        exit(0);
                    }
                    cal_result = l / r;
                    break;

                case Pow:
                    cal_result = pow(l, r);
                    break;

                case Mod:
                    cal_result = fmod(l, r);
                    break;

            }

            // add result to stack
            num_node->data = cal_result;
            list_add(&num_node->list, &result_stack);
        }

    }

    // get result value
    struct list_node_double *result_node = list_first_entry(&result_stack, struct list_node_double, list);
    double result = result_node->data;

    // free result
    list_del(&result_node->list);
    free(result_node);

    // return result
    return result;
}





#endif //CALCULATOR_EXPRESSION_CONVERTER_H
