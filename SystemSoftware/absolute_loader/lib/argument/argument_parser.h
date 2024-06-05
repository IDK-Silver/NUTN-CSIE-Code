//
// Created by idk on 2024/6/3.
//

#ifndef ARGUMENT_H
#define ARGUMENT_H

#include "argument_options.h"
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

struct args {
    char* program_path;
    char* start_address;
    char* file_path;
    bool is_out_result;
};


#define args_init(arg) do {             \
    (arg)->program_path = NULL;         \
    (arg)->start_address = NULL;        \
    (arg)->file_path = NULL;            \
    (arg)->is_out_result = false;       \
} while(0)


int args_parse(int argc, char *argv[], struct args *result);


int get_arg(const int argc, char *argv[], uint32_t arg_index, char **arg, uint32_t *arg_len);


#endif //ARGUMENT_H
