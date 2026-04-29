#include <omp.h>

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    omp_sched_t kind;
} ScheduleCase;

typedef struct {
    const char *schedule_name;
    long long primes_found;
    double total_seconds;
    double best_seconds;
} BenchmarkResult;

static int is_help_requested(int argc, char *argv[]) {
    return argc == 2
        && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0);
}

static void print_usage(const char *program_name, FILE *stream) {
    fprintf(
        stream,
        "Usage: %s <thread_count> <max_number> <chunk_size> [repeat_count]\n"
        "       %s --help\n"
        "\n"
        "Arguments:\n"
        "  thread_count Number of OpenMP threads used by the benchmark.\n"
        "  max_number   Count prime numbers from 2 to this value.\n"
        "  chunk_size   Chunk size used by static, dynamic, and guided schedules.\n"
        "  repeat_count Optional number of timing repetitions for averaging.\n"
        "               Default: 1\n"
        "\n"
        "The benchmark runs the same prime-number workload with schedule(static),\n"
        "schedule(dynamic), and schedule(guided), then prints timing results.\n",
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

static int test_for_prime(long long value) {
    if (value < 2) {
        return 0;
    }

    if (value == 2) {
        return 1;
    }

    if (value % 2 == 0) {
        return 0;
    }

    for (long long factor = 3; factor <= value / factor; factor += 2) {
        if (value % factor == 0) {
            return 0;
        }
    }

    return 1;
}

static long long count_primes(
    long long max_number,
    int thread_count,
    omp_sched_t schedule_kind,
    int chunk_size,
    double *elapsed_seconds
) {
    long long odd_primes_found = 0;
    double start_time = 0.0;

    omp_set_schedule(schedule_kind, chunk_size);
    start_time = omp_get_wtime();

    #pragma omp parallel for num_threads(thread_count) schedule(runtime)
    for (long long candidate = 3; candidate <= max_number; candidate += 2) {
        if (test_for_prime(candidate)) {
            #pragma omp atomic update
            odd_primes_found++;
        }
    }

    *elapsed_seconds = omp_get_wtime() - start_time;

    if (max_number >= 2) {
        return odd_primes_found + 1;
    }

    return odd_primes_found;
}

int main(int argc, char *argv[]) {
    const ScheduleCase schedule_cases[] = {
        { "static", omp_sched_static },
        { "dynamic", omp_sched_dynamic },
        { "guided", omp_sched_guided },
    };
    const int schedule_case_count = (int)(
        sizeof(schedule_cases) / sizeof(schedule_cases[0])
    );
    BenchmarkResult results[3] = { 0 };
    int thread_count = 0;
    int chunk_size = 0;
    int repeat_count = 1;
    long long max_number = 0;
    long long expected_primes_found = -1;

    if (is_help_requested(argc, argv)) {
        print_usage(argv[0], stdout);
        return EXIT_SUCCESS;
    }

    if (argc < 4 || argc > 5) {
        print_usage(argv[0], stderr);
        return EXIT_FAILURE;
    }

    thread_count = (int)parse_long_long(argv[1], "thread_count", 1, INT_MAX);
    max_number = parse_long_long(argv[2], "max_number", 2, LLONG_MAX - 2);
    chunk_size = (int)parse_long_long(argv[3], "chunk_size", 1, INT_MAX);

    if (argc == 5) {
        repeat_count = (int)parse_long_long(argv[4], "repeat_count", 1, INT_MAX);
    }

    omp_set_dynamic(0);

    for (int schedule_index = 0; schedule_index < schedule_case_count; ++schedule_index) {
        results[schedule_index].schedule_name = schedule_cases[schedule_index].name;
        results[schedule_index].best_seconds = -1.0;

        for (int repetition = 0; repetition < repeat_count; ++repetition) {
            double elapsed_seconds = 0.0;
            long long primes_found = count_primes(
                max_number,
                thread_count,
                schedule_cases[schedule_index].kind,
                chunk_size,
                &elapsed_seconds
            );

            if (expected_primes_found < 0) {
                expected_primes_found = primes_found;
            } else if (primes_found != expected_primes_found) {
                fprintf(stderr, "Prime count mismatch while benchmarking.\n");
                return EXIT_FAILURE;
            }

            results[schedule_index].primes_found = primes_found;
            results[schedule_index].total_seconds += elapsed_seconds;

            if (
                results[schedule_index].best_seconds < 0.0
                || elapsed_seconds < results[schedule_index].best_seconds
            ) {
                results[schedule_index].best_seconds = elapsed_seconds;
            }
        }
    }

    printf("Prime schedule benchmark\n");
    printf("thread_count = %d\n", thread_count);
    printf("max_number = %lld\n", max_number);
    printf("chunk_size = %d\n", chunk_size);
    printf("repeat_count = %d\n", repeat_count);
    printf("\n");
    printf("%-10s %14s %16s %16s\n", "schedule", "primes_found", "avg_seconds", "best_seconds");

    for (int schedule_index = 0; schedule_index < schedule_case_count; ++schedule_index) {
        printf(
            "%-10s %14lld %16.9f %16.9f\n",
            results[schedule_index].schedule_name,
            results[schedule_index].primes_found,
            results[schedule_index].total_seconds / repeat_count,
            results[schedule_index].best_seconds
        );
    }

    return EXIT_SUCCESS;
}
