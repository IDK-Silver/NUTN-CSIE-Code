#if defined(__linux__)
#define _GNU_SOURCE
#endif

#if defined(__linux__)
#include <sched.h>
#elif defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__) && defined(__x86_64__)
#include <cpuid.h>
#endif

#include <omp.h>

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_help_requested(int argc, char *argv[]) {
    return argc == 2
        && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0);
}

static void print_usage(const char *program_name, FILE *stream) {
    fprintf(
        stream,
        "Usage: %s <thread_count> <iteration_count>\n"
        "       %s --help\n"
        "\n"
        "Arguments:\n"
        "  thread_count    Number of OpenMP threads to create.\n"
        "  iteration_count Number of loop iterations processed by the parallel for.\n"
        "\n"
        "This demo prints which thread handled each iteration and, when supported,\n"
        "the current CPU/core number.\n",
        program_name,
        program_name
    );
}

static long long parse_long_long(
    const char *text,
    const char *argument_name,
    long long min_value,
    long long max_value
) {
    char *end = NULL;
    long long value = 0;

    errno = 0;
    value = strtoll(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "Invalid %s: %s\n", argument_name, text);
        exit(EXIT_FAILURE);
    }

    if (value < min_value || value > max_value) {
        fprintf(
            stderr,
            "%s must be between %lld and %lld\n",
            argument_name,
            min_value,
            max_value
        );
        exit(EXIT_FAILURE);
    }

    return value;
}

static int get_current_cpu(void) {
#if defined(__linux__)
    return sched_getcpu();
#elif defined(_WIN32)
    return (int)GetCurrentProcessorNumber();
#elif defined(__APPLE__)
    #if defined(__x86_64__)
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;

    if (__get_cpuid_max(0, NULL) >= 0x0B) {
        __cpuid_count(0x0B, 0, eax, ebx, ecx, edx);
        if (ebx != 0) {
            return (int)edx;
        }
    }

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (int)((ebx >> 24) & 0xFF);
    }
    #endif

    return -1;
#else
    return -1;
#endif
}

int main(int argc, char *argv[]) {
    int thread_count = 0;
    uint32_t iteration_count = 0;

    if (is_help_requested(argc, argv)) {
        print_usage(argv[0], stdout);
        return EXIT_SUCCESS;
    }

    if (argc != 3) {
        print_usage(argv[0], stderr);
        return EXIT_FAILURE;
    }

    thread_count = (int)parse_long_long(argv[1], "thread_count", 1, INT_MAX);
    iteration_count = (uint32_t)parse_long_long(
        argv[2],
        "iteration_count",
        1,
        UINT32_MAX
    );

    omp_set_num_threads(thread_count);

    printf("# of proc = %d\n", omp_get_num_procs());
    printf("# of loop iterations = %u\n", iteration_count);

    #pragma omp parallel for
    for (uint32_t iteration_index = 0; iteration_index < iteration_count; ++iteration_index) {
        int current_thread = omp_get_thread_num();
        int current_cpu = get_current_cpu();

        printf(
            "thread %d runs index %u.\n",
            current_thread,
            iteration_index
        );

        if (current_cpu >= 0) {
            printf(
                "thread %d runs index %u in core %d.\n",
                current_thread,
                iteration_index,
                current_cpu
            );
        } else {
            printf(
                "thread %d runs index %u. CPU info is not available on this platform.\n",
                current_thread,
                iteration_index
            );
        }
    }

    return 0;
}
