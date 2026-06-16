#include <omp.h>

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef long long (*SumFunction)(
    const int *matrix,
    size_t matrix_size,
    int thread_count
);

typedef struct {
    const char *name;
    SumFunction sum;
    long long result;
    double total_seconds;
    double best_seconds;
} BenchmarkCase;

static int is_help_requested(int argc, char *argv[]) {
    return argc == 2
        && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0);
}

static void print_usage(const char *program_name, FILE *stream) {
    fprintf(
        stream,
        "Usage: %s <thread_count> <matrix_size> [repeat_count]\n"
        "       %s --help\n"
        "\n"
        "Arguments:\n"
        "  thread_count Number of OpenMP threads used by parallel cases.\n"
        "  matrix_size  Matrix dimension N for summing an NxN integer matrix.\n"
        "  repeat_count Optional number of timing repetitions for averaging.\n"
        "               Default: 1\n"
        "\n"
        "The benchmark compares a serial sum with OpenMP reduction and atomic\n"
        "updates over the same matrix.\n",
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

static size_t checked_matrix_element_count(size_t matrix_size) {
    if (matrix_size != 0 && matrix_size > SIZE_MAX / matrix_size) {
        fprintf(stderr, "matrix_size is too large.\n");
        exit(EXIT_FAILURE);
    }

    return matrix_size * matrix_size;
}

static int *allocate_matrix(size_t element_count) {
    int *matrix = NULL;

    if (element_count > SIZE_MAX / sizeof(*matrix)) {
        fprintf(stderr, "matrix is too large to allocate.\n");
        exit(EXIT_FAILURE);
    }

    matrix = malloc(element_count * sizeof(*matrix));
    if (matrix == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    return matrix;
}

static void initialize_matrix(int *matrix, size_t matrix_size) {
    for (size_t row = 0; row < matrix_size; ++row) {
        for (size_t col = 0; col < matrix_size; ++col) {
            size_t index = row * matrix_size + col;

            matrix[index] = (int)((row * 3 + col * 5 + 7) % 19);
        }
    }
}

static long long serial_sum_matrix(
    const int *matrix,
    size_t matrix_size,
    int thread_count
) {
    long long sum = 0;

    (void)thread_count;

    for (size_t row = 0; row < matrix_size; ++row) {
        for (size_t col = 0; col < matrix_size; ++col) {
            sum += matrix[row * matrix_size + col];
        }
    }

    return sum;
}

static long long reduction_sum_matrix(
    const int *matrix,
    size_t matrix_size,
    int thread_count
) {
    long long sum = 0;
    long long dimension = (long long)matrix_size;

    #pragma omp parallel num_threads(thread_count)
    {
        #pragma omp for reduction(+:sum)
        for (long long row = 0; row < dimension; ++row) {
            size_t row_index = (size_t)row;

            for (size_t col = 0; col < matrix_size; ++col) {
                sum += matrix[row_index * matrix_size + col];
            }
        }
    }

    return sum;
}

static long long atomic_sum_matrix(
    const int *matrix,
    size_t matrix_size,
    int thread_count
) {
    long long sum = 0;
    long long dimension = (long long)matrix_size;

    #pragma omp parallel num_threads(thread_count)
    {
        #pragma omp for
        for (long long row = 0; row < dimension; ++row) {
            size_t row_index = (size_t)row;

            for (size_t col = 0; col < matrix_size; ++col) {
                #pragma omp atomic update
                sum += matrix[row_index * matrix_size + col];
            }
        }
    }

    return sum;
}

static void run_benchmark_case(
    BenchmarkCase *benchmark_case,
    const int *matrix,
    size_t matrix_size,
    int thread_count,
    int repeat_count,
    long long expected_sum
) {
    benchmark_case->total_seconds = 0.0;
    benchmark_case->best_seconds = -1.0;

    for (int repetition = 0; repetition < repeat_count; ++repetition) {
        double elapsed_seconds = 0.0;
        double start_time = omp_get_wtime();

        benchmark_case->result = benchmark_case->sum(
            matrix,
            matrix_size,
            thread_count
        );
        elapsed_seconds = omp_get_wtime() - start_time;

        if (benchmark_case->result != expected_sum) {
            fprintf(
                stderr,
                "%s sum mismatch: expected %lld, got %lld\n",
                benchmark_case->name,
                expected_sum,
                benchmark_case->result
            );
            exit(EXIT_FAILURE);
        }

        benchmark_case->total_seconds += elapsed_seconds;

        if (
            benchmark_case->best_seconds < 0.0
            || elapsed_seconds < benchmark_case->best_seconds
        ) {
            benchmark_case->best_seconds = elapsed_seconds;
        }
    }
}

int main(int argc, char *argv[]) {
    BenchmarkCase benchmark_cases[] = {
        { "serial", serial_sum_matrix, 0, 0.0, 0.0 },
        { "reduction", reduction_sum_matrix, 0, 0.0, 0.0 },
        { "atomic", atomic_sum_matrix, 0, 0.0, 0.0 },
    };
    const int benchmark_case_count = (int)(
        sizeof(benchmark_cases) / sizeof(benchmark_cases[0])
    );
    int *matrix = NULL;
    int thread_count = 0;
    int repeat_count = 1;
    size_t matrix_size = 0;
    size_t element_count = 0;
    long long expected_sum = 0;

    if (is_help_requested(argc, argv)) {
        print_usage(argv[0], stdout);
        return EXIT_SUCCESS;
    }

    if (argc < 3 || argc > 4) {
        print_usage(argv[0], stderr);
        return EXIT_FAILURE;
    }

    thread_count = (int)parse_long_long(argv[1], "thread_count", 1, INT_MAX);
    matrix_size = (size_t)parse_long_long(argv[2], "matrix_size", 1, INT_MAX);

    if (argc == 4) {
        repeat_count = (int)parse_long_long(argv[3], "repeat_count", 1, INT_MAX);
    }

    element_count = checked_matrix_element_count(matrix_size);
    matrix = allocate_matrix(element_count);
    initialize_matrix(matrix, matrix_size);
    omp_set_dynamic(0);

    expected_sum = serial_sum_matrix(matrix, matrix_size, thread_count);

    for (int case_index = 0; case_index < benchmark_case_count; ++case_index) {
        run_benchmark_case(
            &benchmark_cases[case_index],
            matrix,
            matrix_size,
            thread_count,
            repeat_count,
            expected_sum
        );
    }

    printf("Sum reduction vs atomic benchmark\n");
    printf("thread_count = %d\n", thread_count);
    printf("matrix_size = %zu\n", matrix_size);
    printf("repeat_count = %d\n", repeat_count);
    printf("\n");
    printf("%-10s %16s %16s %16s\n", "method", "sum", "avg_seconds", "best_seconds");

    for (int case_index = 0; case_index < benchmark_case_count; ++case_index) {
        printf(
            "%-10s %16lld %16.9f %16.9f\n",
            benchmark_cases[case_index].name,
            benchmark_cases[case_index].result,
            benchmark_cases[case_index].total_seconds / repeat_count,
            benchmark_cases[case_index].best_seconds
        );
    }

    free(matrix);
    return EXIT_SUCCESS;
}
