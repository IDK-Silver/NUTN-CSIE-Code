#include <stdio.h>
#include <argument/argument_parser.h>
#include <utility/numeric.h>
#include <inttypes.h>
#include <core/loader.h>

#include "absolute_loader_config.h"


int main(int argc, char *argv[]) {

    // storage arument parse result
    struct args arg_result;
    args_init(&arg_result);

    // parseing argument
    args_parse(argc, argv, &arg_result);

    // storage loading file
    uint8_t *hex_array  = NULL;
    uint32_t hex_len = 0;

    // conversion ascii to hex (byte)
    conversion_hex(arg_result.file_path, &hex_array,  &hex_len);

    // get memory start address by string that arument give
    uint32_t address = num_from_ascii(
        arg_result.start_address, strlen(arg_result.start_address), 16
    );


    // init buffer
    char *buffer = (char *) malloc(sizeof(char) * CONSOLE_MESSAGE_BUFFER_LEN);
    memset(buffer, '\0', CONSOLE_MESSAGE_BUFFER_LEN);


    // show memory data
    for (int i = 0; i < hex_len; i++) {
        sprint_memory_info(&buffer, address, 5, hex_array[i]);
        address += sizeof(uint8_t);
        if (arg_result.is_out_result) {
            // write to file
        }
        else {
            fprintf(stdout, "%s", buffer);
        }
    }




    return 0;
}


