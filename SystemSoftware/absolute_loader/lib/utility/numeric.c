//
// Created by idk on 2024/6/5.
//
#include "numeric.h"



int num_from_ascii(const char* accii, const uint32_t ascii_len, const int base) {

    int result = 0;
    // uint32_t ascii_len = strlen(accii);

    switch (base) {
        case 10:

            for (uint32_t index = 0; index < ascii_len; index++) {

                // get the character of string
                char c = accii[index];

                // get dec num from ascii
                c -= '0';

                // carrying
                result *= base;

                // add number
                result += c;
            }
            break;

        case 0x10:
            for (uint32_t index = 0; index < ascii_len; index++) {

                // get the character of string
                char c = accii[index];

                // get dec num from ascii
                c -= '0';

                // more the pure number (9)
                c = (c > (10 - 1)) ? c - (0x10 - (10 - 1)) : c;

                // if character is upper case, sub offset
                c = (c > (0x10 - 1)) ? c - ('a' - 'A') : c;

                // carrying
                result *= base;

                // add number
                result += c;
            }
            break;

        case 010:
            for (uint32_t index = 0; index < ascii_len; index++) {

                // get the character of string
                char c = accii[index];

                // get dec num from ascii
                c -= '0';

                // carrying
                result *= base;

                // add number
                result += c;
            }
        break;

        case 0b10:
            for (uint32_t index = 0; index < ascii_len; index++) {

                // get the character of string
                char c = accii[index];

                // get dec num from ascii
                c -= '0';

                // carrying
                result *= base;

                // add number
                result += c;
            }
        break;

        default:
            fprintf(stderr, "num_from_ascii : unknow base");
            return 0;
    }

    return result;
}
