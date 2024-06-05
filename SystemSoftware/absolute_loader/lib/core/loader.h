//
// Created by idk on 2024/6/6.
//

#ifndef LOADER_H
#define LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <utility/numeric.h>
#include <inttypes.h>
#include <list/list.h>

struct list_node_hex_wf {
    struct hex_wf *data;
    struct list_head list;
};

void conversion_hex(const char *file_path, uint8_t **dest, uint32_t *dest_len);
void sprint_memory_info(char **dest, int address, const uint32_t max_address_len, const uint8_t content);


#endif //LOADER_H
