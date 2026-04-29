#include <omp.h>

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*MultiplyFunction)(
    const double *a,
    const double *b,
    double *c,
    size_t matrix_size,
    int thread_count
);

typedef struct {
    const char *name;
    MultiplyFunction multiply;
    double total_seconds;
    double best_seconds;
    double checksum;
    double max_error;
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
        "  matrix_size  Matrix dimension N for NxN matrix multiplication.\n"
        "  repeat_count Optional number of timing repetitions for averaging.\n"
        "               Default: 1\n"
        "\n"
        "The program runs four implementations in one benchmark:\n"
        "  seq_inner: C[i][j] = sum_k A[i][k] * B[k][j]\n"
        "  seq_outer: C[i][j] += A[i][k] * B[k][j], using i-k-j loop order\n"
        "  omp_inner: parallel version of seq_inner\n"
        "  omp_outer: parallel version of seq_outer, partitioned by rows\n",
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

static double *allocate_matrix(size_t element_count, const char *matrix_name) {
    double *matrix = NULL;

    if (element_count > SIZE_MAX / sizeof(*matrix)) {
        fprintf(stderr, "%s is too large to allocate.\n", matrix_name);
        exit(EXIT_FAILURE);
    }

    matrix = malloc(element_count * sizeof(*matrix));
    if (matrix == NULL) {
        fprintf(stderr, "Failed to allocate %s.\n", matrix_name);
        exit(EXIT_FAILURE);
    }

    return matrix;
}

static void initialize_matrices(double *a, double *b, size_t matrix_size) {
    for (size_t row = 0; row < matrix_size; ++row) {
        for (size_t col = 0; col < matrix_size; ++col) {
            size_t index = row * matrix_size + col;
            int a_token = (int)(((row % 23) * 3 + (col % 23) * 5 + 1) % 23);
            int b_token = (int)(((row % 29) * 7 + (col % 29) * 2 + 3) % 29);

            a[index] = (double)(a_token - 11) / 7.0;
            b[index] = (double)(b_token - 14) / 9.0;
        }
    }
}

static void seq_inner_product(
    const double *a,
    const double *b,
    double *c,
    size_t matrix_size,
    int thread_count
) {
    (void)thread_count;

    for (size_t row = 0; row < matrix_size; ++row) {
        for (size_t col = 0; col < matrix_size; ++col) {
            double sum = 0.0;

            for (size_t k = 0; k < matrix_size; ++k) {
                sum += a[row * matrix_size + k] * b[k * matrix_size + col];
            }

            c[row * matrix_size + col] = sum;
        }
    }
}

static void seq_outer_product(
    const double *a,
    const double *b,
    double *c,
    size_t matrix_size,
    int thread_count
) {
    (void)thread_count;

    for (size_t row = 0; row < matrix_size; ++row) {
        double *c_row = c + row * matrix_size;

        for (size_t col = 0; col < matrix_size; ++col) {
            c_row[col] = 0.0;
        }

        for (size_t k = 0; k < matrix_size; ++k) {
            const double aik = a[row * matrix_size + k];
            const double *b_row = b + k * matrix_size;

            for (size_t col = 0; col < matrix_size; ++col) {
                c_row[col] += aik * b_row[col];
            }
        }
    }
}

static void omp_inner_product(
    const double *a,
    const double *b,
    double *c,
    size_t matrix_size,
    int thread_count
) {
    long long dimension = (long long)matrix_size;

    #pragma omp parallel for collapse(2) num_threads(thread_count) schedule(static)
    for (long long row = 0; row < dimension; ++row) {
        for (long long col = 0; col < dimension; ++col) {
            double sum = 0.0;
            size_t row_index = (size_t)row;
            size_t col_index = (size_t)col;

            for (size_t k = 0; k < matrix_size; ++k) {
                sum += a[row_index * matrix_size + k]
                    * b[k * matrix_size + col_index];
            }

            c[row_index * matrix_size + col_index] = sum;
        }
    }
}

static void omp_outer_product(
    const double *a,
    const double *b,
    double *c,
    size_t matrix_size,
    int thread_count
) {
    long long dimension = (long long)matrix_size;

    #pragma omp parallel for num_threads(thread_count) schedule(static)
    for (long long row = 0; row < dimension; ++row) {
        size_t row_index = (size_t)row;
        double *c_row = c + row_index * matrix_size;

        for (size_t col = 0; col < matrix_size; ++col) {
            c_row[col] = 0.0;
        }

        for (size_t k = 0; k < matrix_size; ++k) {
            const double aik = a[row_index * matrix_size + k];
            const double *b_row = b + k * matrix_size;

            for (size_t col = 0; col < matrix_size; ++col) {
                c_row[col] += aik * b_row[col];
            }
        }
    }
}

static double absolute_double(double value) {
    if (value < 0.0) {
        return -value;
    }

    return value;
}

static double max_abs_difference(
    const double *expected,
    const double *actual,
    size_t element_count
) {
    double max_error = 0.0;

    for (size_t index = 0; index < element_count; ++index) {
        double error = absolute_double(expected[index] - actual[index]);

        if (error > max_error) {
            max_error = error;
        }
    }

    return max_error;
}

static double matrix_checksum(const double *matrix, size_t element_count) {
    double checksum = 0.0;

    for (size_t index = 0; index < element_count; ++index) {
        checksum += matrix[index];
    }

    return checksum;
}

static void run_benchmark_case(
    BenchmarkCase *benchmark_case,
    const double *a,
    const double *b,
    const double *reference,
    double *work,
    size_t matrix_size,
    size_t element_count,
    int thread_count,
    int repeat_count
) {
    benchmark_case->total_seconds = 0.0;
    benchmark_case->best_seconds = -1.0;

    for (int repetition = 0; repetition < repeat_count; ++repetition) {
        double start_time = omp_get_wtime();

        benchmark_case->multiply(a, b, work, matrix_size, thread_count);

        double elapsed_seconds = omp_get_wtime() - start_time;
        benchmark_case->total_seconds += elapsed_seconds;

        if (
            benchmark_case->best_seconds < 0.0
            || elapsed_seconds < benchmark_case->best_seconds
        ) {
            benchmark_case->best_seconds = elapsed_seconds;
        }
    }

    benchmark_case->checksum = matrix_checksum(work, element_count);
    benchmark_case->max_error = max_abs_difference(reference, work, element_count);
}

int main(int argc, char *argv[]) {
    BenchmarkCase benchmark_cases[] = {
        { "seq_inner", seq_inner_product, 0.0, 0.0, 0.0, 0.0 },
        { "seq_outer", seq_outer_product, 0.0, 0.0, 0.0, 0.0 },
        { "omp_inner", omp_inner_product, 0.0, 0.0, 0.0, 0.0 },
        { "omp_outer", omp_outer_product, 0.0, 0.0, 0.0, 0.0 },
    };
    const int benchmark_case_count = (int)(
        sizeof(benchmark_cases) / sizeof(benchmark_cases[0])
    );
    const double tolerance = 1.0e-8;
    int thread_count = 0;
    int repeat_count = 1;
    size_t matrix_size = 0;
    size_t element_count = 0;
    double *a = NULL;
    double *b = NULL;
    double *reference = NULL;
    double *work = NULL;
    double baseline_average = 0.0;
    double total_mib = 0.0;

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
    a = allocate_matrix(element_count, "A");
    b = allocate_matrix(element_count, "B");
    reference = allocate_matrix(element_count, "reference C");
    work = allocate_matrix(element_count, "work C");

    initialize_matrices(a, b, matrix_size);
    omp_set_dynamic(0);

    seq_inner_product(a, b, reference, matrix_size, 1);

    printf("Matrix multiplication benchmark\n");
    printf("thread_count = %d\n", thread_count);
    printf("matrix_size = %zu x %zu\n", matrix_size, matrix_size);
    printf("repeat_count = %d\n", repeat_count);
    total_mib = ((double)element_count * (double)sizeof(double) * 4.0)
        / (1024.0 * 1024.0);
    printf("allocated_matrix_memory_mib = %.2f\n", total_mib);
    printf("\n");

    for (int case_index = 0; case_index < benchmark_case_count; ++case_index) {
        run_benchmark_case(
            &benchmark_cases[case_index],
            a,
            b,
            reference,
            work,
            matrix_size,
            element_count,
            thread_count,
            repeat_count
        );
    }

    baseline_average = benchmark_cases[0].total_seconds / repeat_count;

    for (int case_index = 0; case_index < benchmark_case_count; ++case_index) {
        const BenchmarkCase *benchmark_case = &benchmark_cases[case_index];
        double average_seconds = benchmark_case->total_seconds / repeat_count;

        printf("%s\n", benchmark_case->name);
        printf("  average_seconds = %.9f\n", average_seconds);
        printf("  best_seconds = %.9f\n", benchmark_case->best_seconds);

        if (average_seconds > 0.0) {
            printf("  speedup_vs_seq_inner = %.3fx\n", baseline_average / average_seconds);
        }

        printf("  checksum = %.9f\n", benchmark_case->checksum);
        printf("  max_error = %.12f\n", benchmark_case->max_error);

        if (benchmark_case->max_error > tolerance) {
            fprintf(
                stderr,
                "%s produced a result outside tolerance %.12f.\n",
                benchmark_case->name,
                tolerance
            );
            free(work);
            free(reference);
            free(b);
            free(a);
            return EXIT_FAILURE;
        }

        printf("\n");
    }

    free(work);
    free(reference);
    free(b);
    free(a);

    return EXIT_SUCCESS;
}
