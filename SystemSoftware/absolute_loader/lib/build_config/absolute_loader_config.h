//
// Created by idk on 2024/6/6.
//

#ifndef ABSOLUTE_LOADER_CONFIG_H
#define ABSOLUTE_LOADER_CONFIG_H

#define CONSOLE_MESSAGE_BUFFER_LEN 512
#define DEFAULT_MEMORY_OUTFILE_EXTENSION ".mem"

#ifdef _WIN32
    #define PROGRAM_NAME_IN_RUNTIME "absolute_loader.exe"
#endif

#ifdef linux
    #define PROGRAM_NAME_IN_RUNTIME "absolute_loader"
#endif


#endif //ABSOLUTE_LOADER_CONFIG_H
