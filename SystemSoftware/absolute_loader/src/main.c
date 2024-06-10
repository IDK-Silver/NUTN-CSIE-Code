#include <inttypes.h>
#include <stdio.h>

#include <argument/argument_parser.h>
#include <build_config/absolute_loader_config.h>
#include <core/loader.h>
#include <utility/numeric.h>

int main(int argc, char *argv[]) {

    // storage arument parse result
    struct args arg_result;
    args_init(&arg_result);

    // parseing argument
    if (args_parse(argc, argv, &arg_result) < 0) {
        exit(-1);
    }

    // storage loading file
    uint8_t *hex_array  = NULL;
    uint32_t hex_len = 0;

    // conversion ascii to hex (byte)
    conversion_hex(arg_result.file_path, &hex_array,  &hex_len);

    // get memory start address by string that argument give
    uint32_t address = num_from_ascii(
        arg_result.start_address, strlen(arg_result.start_address), 16
    );


    // init memory info str buffer
    char *buffer = (char *) malloc(sizeof(char) * CONSOLE_MESSAGE_BUFFER_LEN);
    memset(buffer, '\0', CONSOLE_MESSAGE_BUFFER_LEN);


    // if need, try to open output file
    FILE *out_file = NULL;
    if (arg_result.is_out_result) {
        out_file = open_output_file(
            arg_result.file_path, DEFAULT_MEMORY_OUTFILE_EXTENSION
        );
    }

    // show memory data or wirte to output file
    for (int i = 0; i < hex_len; i++) {

        // print info to buffer
        sprint_memory_info(&buffer, address, 5, hex_array[i]);

        // add memory address
        address += sizeof(uint8_t);

        // select IO (stdout, or outputfile)
        if (arg_result.is_out_result) {
            fprintf(out_file, "%s", buffer);
        }
        else {
            fprintf(stdout, "%s", buffer);
        }
    }

    return 0;
}


