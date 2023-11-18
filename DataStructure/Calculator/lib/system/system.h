//
// Created by idk on 2023/11/18.
//

#ifndef CALCULATOR_SYSTEM_H
#define CALCULATOR_SYSTEM_H

#include <stdlib.h>
#include <stdio.h>

void exit_with_press_any_key(int __status) {

    // clear buffer
    fflush(stderr);

    // print message
    printf("\nEnter any to exit.");

    // input char
    char exit_char = 0;

    // clear the input buffer and wait user press enter
    scanf("%c", &exit_char);
    while (exit_char != '\n') {
        scanf("%c", &exit_char);
    };

    // exit program
    exit(__status);
}

#endif //CALCULATOR_SYSTEM_H
