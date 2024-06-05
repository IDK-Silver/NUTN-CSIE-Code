//
// Created by idk on 2024/6/5.
//

#ifndef UTILITY_NUMERIC_H
#define UTILITY_NUMERIC_H


#include <stdio.h>

#include <inttypes.h>
#include <string.h>

struct hex_wf {
    uint8_t rf : 4;
    uint8_t lf : 4;
};

#define hex_wf_init(var) do {              \
    (var)->lf = 0;                   \
    (var)->rf = 0;                   \
} while(0)

int num_from_ascii(const char* accii, const uint32_t ascii_len, const int base);


#endif //UTILITY_NUMERIC_H
