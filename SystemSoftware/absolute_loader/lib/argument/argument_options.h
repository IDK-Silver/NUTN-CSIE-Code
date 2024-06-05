//
// Created by idk on 2024/6/3.
//

#ifndef ARGUMENT_OPTIONS_H
#define ARGUMENT_OPTIONS_H

#include <stdio.h>
#include <string.h>

enum ARG_OPTS {
    START_ADDRESS,
    IS_OUT_RESULT,
};

int argtoc(enum ARG_OPTS options, char **result);

#endif //ARGUMENT_OPTIONS_H
