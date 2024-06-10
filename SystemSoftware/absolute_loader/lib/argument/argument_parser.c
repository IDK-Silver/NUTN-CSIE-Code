//
// Created by idk on 2024/6/3.
//

#include "argument_parser.h"

#include <build_config/absolute_loader_config.h>


int args_parse(int argc, char *argv[], struct args *result) {

    // the number of arugment not match any option
    uint32_t cn_option = 0;

    char *opt_str = malloc(sizeof(char) * 10);
    uint32_t opt_len;

    // argument and that length
    char *arg = NULL;
    uint32_t arg_len = 0;

    for (int arg_index = 0; arg_index < argc; arg_index++) {

        // get argment form argv
        if (get_arg(argc, argv, arg_index, &arg, &arg_len) < 0) {
            return -1;
        }

        // check option (start address)
        {
            argtoc(START_ADDRESS, &opt_str);
            opt_len = strlen(opt_str);


            if (strncmp(arg, opt_str, opt_len) == 0) {

                if (arg_len == opt_len) {
                    // move to next argument to get the file path
                    arg_index++;

                    if (get_arg(argc, argv, arg_index, &arg, &arg_len) < 0) {
                        exit(-1);
                    }

                    result->start_address = arg;
                    continue;
                }

                result->start_address = (arg + opt_len);
                continue;
            }

        }


        // check option (out result)
        {
            // get the arguemnt option string
            argtoc(IS_OUT_RESULT, &opt_str);
            opt_len = strlen(opt_str);

            if (strncmp(arg, opt_str, opt_len) == 0) {
                result->is_out_result = true;
                // move to next argument to get the file path
                // arg_index++;
                //
                // if (get_arg(argc, argv, arg_index, &arg, &arg_len) < 0) {
                //     exit(-1);
                // }
                //
                // result->file_path = arg;
                continue;
            }
        }

        // if run here it mean not match any option
        cn_option++;

        // assign default like program path or input source path
        switch (cn_option) {

            case 1:
                result->program_path = arg;

            case 2:
                result->file_path = arg;
                break;

            default:
                fprintf(stderr, "too much argument\n");
                exit(-1);
        }


    }


    if (result->start_address == NULL) {
        fprintf(stderr,
            "must given memory start info like : \n\t \'%s -A 05279 \n",
            PROGRAM_NAME_IN_RUNTIME
        );
        return -1;
    }

    return 0;
}

int get_arg(const int argc, char *argv[], uint32_t arg_index, char **arg, uint32_t *arg_len) {

    if (arg_index > argc) {
        fprintf(stderr, "get_next_arg : out of range\n");
        return -1;
    }

    *arg_len = strlen(argv[arg_index]);
    *arg = argv[arg_index];

    return  0;
}