//
// Created by idk on 2024/6/6.
//
#include "loader.h"


void conversion_hex(const char *file_path, uint8_t** dest, uint32_t *dest_len) {

    FILE *file = NULL;

    // open source file
    if (file_path != NULL) {

        // open file
        file = fopen(file_path,"r");

        // ensure file is valid
        if(file == NULL)
        {
            fprintf(stderr, "faild to open file\n");
            exit(1);
        }

    }

    // the hex with half list
    struct list_head hex_wf_list = {0};

    // init list
    INIT_LIST_HEAD(&hex_wf_list);

    // for each char in file
    bool is_need_merg = false;

    *dest_len = 0;
    struct list_node_hex_wf *node = NULL;

    for (char fc = 0; fscanf(file, "%c", &fc) == 1;) {
        // printf("%c", fc);


        if (!is_need_merg) {
            node = (struct list_node_hex_wf*) malloc(sizeof(struct list_node_hex_wf) * 1);
            node->data = (struct hex_wf *) malloc(sizeof(struct hex_wf));
            hex_wf_init(node->data);
            node->data->lf = num_from_ascii(&fc, 1,16);
            is_need_merg = true;
        }
        else {
            node->data->rf = num_from_ascii(&fc, 1,16);
            list_add_tail(&node->list, &hex_wf_list);
            *dest_len = *dest_len + 1;
            is_need_merg = false;
        }

    }

    *dest = (uint8_t*) malloc(sizeof(uint8_t) * (*dest_len));


    // to for each element of list
    struct list_head *pos, *n;

    uint32_t index = 0;
    list_for_each_safe(pos, n, &hex_wf_list)
    {
        struct list_node_hex_wf *st = list_entry(pos,struct list_node_hex_wf, list);
        uint8_t content = *((uint8_t*) st->data);
        (*dest)[index] = content;
        index++;
        list_del(pos);
        free(st);
    }
}

void sprint_memory_info(char **dest, int address, const uint32_t max_address_len, const uint8_t content) {

    int address_len = 0;

    // check address is not zero
    if (address <= 0 || max_address_len <= 0) {
        address_len = 0;

    }
    else {

        // get the address len by log_16 (address) and ceiling that
        const double log_target = log(address);
        const double log_base = log(0x10);
        address_len = ((log_target - 1) / (log_base + 1));
    }

    // fill zero
    for (int i = 0 ; i < max_address_len - address_len - 1; i++) {
        sprintf((*dest) + i, "0");
    }

    // get hex info
    const struct hex_wf hex = *(struct hex_wf*)(&content);

    // if hex left half is 0 need to fill 0
    if (hex.lf <= 0) {
        sprintf((*dest) + (max_address_len - address_len - 2), "%X\t0%X\n", address, content);
    }
    else
        sprintf((*dest) + (max_address_len - address_len + -2), "%X\t%X\n", address, content);
}

