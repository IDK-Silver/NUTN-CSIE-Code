//
// Created by idk on 2023/11/18.
//

#ifndef CALCULATOR_SYSTEM_H
#define CALCULATOR_SYSTEM_H

#include <stdlib.h>
#include <stdio.h>

void exit_with_press_any_key(int __status) {
    printf("Enter any to exit.");
    char exit_char = 0;
    scanf("%c", &exit_char);
    exit(__status);
}

#endif //CALCULATOR_SYSTEM_H
