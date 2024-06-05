//
// Created by idk on 2024/6/3.
//
#include "argument_options.h"

int argtoc(enum ARG_OPTS options, char **result) {
    if (result == NULL || *result == NULL) {
        fprintf(stderr, "argument to string failed : null pointer\n");
        return -1;
    }

    switch (options) {

        case START_ADDRESS:
            strcpy(*result, "-A");
        return 0;

        case IS_OUT_RESULT:
            strcpy(*result, "-f");
        return 0;

        default:
            fprintf(stderr, "argument to string failed : unknow option\n");
        return -1;
    }
}

