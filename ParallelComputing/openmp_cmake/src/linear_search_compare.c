#include <omp.h>

#include <assert.h>
#include <errno.h>
#include <limits.h>
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
        "Usage: %s <thread_count> <element_count> <target_value> [repeat_count]\n"
        "       %s --help\n"
        "\n"
        "Arguments:\n"
        "  thread_count  Number of OpenMP threads used by the parallel search.\n"
        "  element_count Number of integers allocated in the search array.\n"
        "  target_value  Value to search for in the array.\n"
        "  repeat_count  Optional number of timing repetitions for averaging.\n"
        "                Default: 1\n"
        "\n"
        "The program initializes data[i] = i, then runs both serial and parallel\n"
        "linear search and prints average execution time plus speedup.\n",
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

static void initialize_search_data(int *search_data, size_t element_count) {
    for (size_t element_index = 0; element_index < element_count; ++element_index) {
        search_data[element_index] = (int)element_index;
    }
}

static long long serial_linear_search(
    const int *search_data,
    size_t element_count,
    int target_value
) {
    for (size_t element_index = 0; element_index < element_count; ++element_index) {
        if (search_data[element_index] == target_value) {
            return (long long)element_index;
        }
    }

    return -1;
}

static long long parallel_linear_search(
    const int *search_data,
    size_t element_count,
    int target_value,
    int thread_count
) {
    long long first_match_index = (long long)element_count;
    long long total_elements = (long long)element_count;

    #pragma omp parallel for num_threads(thread_count) reduction(min:first_match_index)
    for (long long element_index = 0; element_index < total_elements; ++element_index) {
        if (search_data[element_index] == target_value) {
            first_match_index = element_index;
        }
    }

    if (first_match_index == total_elements) {
        return -1;
    }

    return first_match_index;
}

int main(int argc, char *argv[]) {
    int *search_data = NULL;
    int thread_count = 0;
    int target_value = 0;
    int repeat_count = 1;
    size_t element_count = 0;
    long long serial_match_index = -1;
    long long parallel_match_index = -1;
    double serial_elapsed_seconds = 0.0;
    double parallel_elapsed_seconds = 0.0;

    if (is_help_requested(argc, argv)) {
        print_usage(argv[0], stdout);
        return EXIT_SUCCESS;
    }

    if (argc < 4 || argc > 5) {
        print_usage(argv[0], stderr);
        return EXIT_FAILURE;
    }

    thread_count = (int)parse_long_long(argv[1], "thread_count", 1, INT_MAX);
    element_count = (size_t)parse_long_long(argv[2], "element_count", 1, INT_MAX);
    target_value = (int)parse_long_long(argv[3], "target_value", INT_MIN, INT_MAX);

    if (argc == 5) {
        repeat_count = (int)parse_long_long(argv[4], "repeat_count", 1, INT_MAX);
    }

    search_data = malloc(element_count * sizeof(*search_data));
    if (search_data == NULL) {
        perror("malloc");
        return EXIT_FAILURE;
    }

    initialize_search_data(search_data, element_count);
    omp_set_dynamic(0);

    for (int iteration = 0; iteration < repeat_count; ++iteration) {
        double start_time = omp_get_wtime();
        serial_match_index = serial_linear_search(
            search_data,
            element_count,
            target_value
        );
        serial_elapsed_seconds += omp_get_wtime() - start_time;
    }

    for (int iteration = 0; iteration < repeat_count; ++iteration) {
        double start_time = omp_get_wtime();
        parallel_match_index = parallel_linear_search(
            search_data,
            element_count,
            target_value,
            thread_count
        );
        parallel_elapsed_seconds += omp_get_wtime() - start_time;
    }

    printf("thread_count = %d\n", thread_count);
    printf("element_count = %zu\n", element_count);
    printf("target_value = %d\n", target_value);
    printf("repeat_count = %d\n", repeat_count);
    printf("serial_match_index = %lld\n", serial_match_index);
    printf("parallel_match_index = %lld\n", parallel_match_index);
    printf(
        "serial_average_seconds = %.9f\n",
        serial_elapsed_seconds / repeat_count
    );
    printf(
        "parallel_average_seconds = %.9f\n",
        parallel_elapsed_seconds / repeat_count
    );

    if (parallel_elapsed_seconds > 0.0) {
        printf(
            "speedup = %.3fx\n",
            serial_elapsed_seconds / parallel_elapsed_seconds
        );
    }

    assert(serial_match_index == parallel_match_index);

    free(search_data);
    return EXIT_SUCCESS;
}
