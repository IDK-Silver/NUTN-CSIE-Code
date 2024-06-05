//
// Created by idk on 2024/6/5.
//
#include "file.h"

#include <stddef.h>
#include <inttypes.h>
#include <string.h>

int filename_without_extension(char **dest, const char *src) {

    if (src == NULL || dest == NULL || *dest == NULL) {
        return -1;
    }

    const uint32_t scr_len = strlen(src);

    uint32_t last_dot_len = scr_len;

    for (;last_dot_len >= 1;last_dot_len--) {
        if (src[last_dot_len - 1] == '.') {
            break;
        }
    }

    strncpy(*dest, src, last_dot_len - 1);

    return 0;
}
