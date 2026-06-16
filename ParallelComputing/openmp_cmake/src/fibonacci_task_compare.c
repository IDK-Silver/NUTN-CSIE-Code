#include <omp.h>

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long long (*FibFunction)(int n, int thread_count, int task_cutoff);

typedef struct {
    const char *name;
    FibFunction fib;
    unsigned long long result;
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
        "Usage: %s <thread_count> <n> <task_cutoff> [repeat_count]\n"
        "       %s --help\n"
        "\n"
        "Arguments:\n"
        "  thread_count Number of OpenMP threads used by the task case.\n"
        "  n            Fibonacci index. Maximum: 93 for unsigned long long.\n"
        "  task_cutoff  Use serial recursion when n <= task_cutoff.\n"
        "  repeat_count Optional number of timing repetitions for averaging.\n"
        "               Default: 1\n"
        "\n"
        "The benchmark compares serial recursive Fibonacci with an OpenMP task\n"
        "version that uses parallel, single, task, firstprivate, shared, and\n"
        "taskwait. This is intentionally the inefficient O(2^n) Fibonacci.\n",
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

static unsigned long long serial_fibonacci_recursive(int n) {
    if (n < 2) {
        return (unsigned long long)n;
    }

    return serial_fibonacci_recursive(n - 1) + serial_fibonacci_recursive(n - 2);
}

static unsigned long long serial_fibonacci(
    int n,
    int thread_count,
    int task_cutoff
) {
    (void)thread_count;
    (void)task_cutoff;

    return serial_fibonacci_recursive(n);
}

static unsigned long long task_fibonacci_recursive(int n, int task_cutoff) {
    unsigned long long x = 0;
    unsigned long long y = 0;

    if (n < 2) {
        return (unsigned long long)n;
    }

    if (n <= task_cutoff) {
        return serial_fibonacci_recursive(n);
    }

    #pragma omp task shared(x) firstprivate(n, task_cutoff)
    {
        x = task_fibonacci_recursive(n - 1, task_cutoff);
    }

    #pragma omp task shared(y) firstprivate(n, task_cutoff)
    {
        y = task_fibonacci_recursive(n - 2, task_cutoff);
    }

    #pragma omp taskwait

    return x + y;
}

static unsigned long long task_fibonacci(
    int n,
    int thread_count,
    int task_cutoff
) {
    unsigned long long result = 0;

    #pragma omp parallel num_threads(thread_count)
    {
        #pragma omp single
        {
            result = task_fibonacci_recursive(n, task_cutoff);
        }
    }

    return result;
}

static void run_benchmark_case(
    BenchmarkCase *benchmark_case,
    int n,
    int thread_count,
    int task_cutoff,
    int repeat_count,
    unsigned long long expected_result
) {
    benchmark_case->total_seconds = 0.0;
    benchmark_case->best_seconds = -1.0;

    for (int repetition = 0; repetition < repeat_count; ++repetition) {
        double elapsed_seconds = 0.0;
        double start_time = omp_get_wtime();

        benchmark_case->result = benchmark_case->fib(n, thread_count, task_cutoff);
        elapsed_seconds = omp_get_wtime() - start_time;

        if (benchmark_case->result != expected_result) {
            fprintf(
                stderr,
                "%s result mismatch: expected %llu, got %llu\n",
                benchmark_case->name,
                expected_result,
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
        { "serial", serial_fibonacci, 0, 0.0, 0.0 },
        { "task", task_fibonacci, 0, 0.0, 0.0 },
    };
    const int benchmark_case_count = (int)(
        sizeof(benchmark_cases) / sizeof(benchmark_cases[0])
    );
    int thread_count = 0;
    int n = 0;
    int task_cutoff = 0;
    int repeat_count = 1;
    unsigned long long expected_result = 0;

    if (is_help_requested(argc, argv)) {
        print_usage(argv[0], stdout);
        return EXIT_SUCCESS;
    }

    if (argc < 4 || argc > 5) {
        print_usage(argv[0], stderr);
        return EXIT_FAILURE;
    }

    thread_count = (int)parse_long_long(argv[1], "thread_count", 1, INT_MAX);
    n = (int)parse_long_long(argv[2], "n", 0, 93);
    task_cutoff = (int)parse_long_long(argv[3], "task_cutoff", 0, n);

    if (argc == 5) {
        repeat_count = (int)parse_long_long(argv[4], "repeat_count", 1, INT_MAX);
    }

    omp_set_dynamic(0);
    expected_result = serial_fibonacci_recursive(n);

    for (int case_index = 0; case_index < benchmark_case_count; ++case_index) {
        run_benchmark_case(
            &benchmark_cases[case_index],
            n,
            thread_count,
            task_cutoff,
            repeat_count,
            expected_result
        );
    }

    printf("OpenMP task Fibonacci benchmark\n");
    printf("thread_count = %d\n", thread_count);
    printf("n = %d\n", n);
    printf("task_cutoff = %d\n", task_cutoff);
    printf("repeat_count = %d\n", repeat_count);
    printf("\n");
    printf("%-10s %20s %16s %16s\n", "method", "fib(n)", "avg_seconds", "best_seconds");

    for (int case_index = 0; case_index < benchmark_case_count; ++case_index) {
        printf(
            "%-10s %20llu %16.9f %16.9f\n",
            benchmark_cases[case_index].name,
            benchmark_cases[case_index].result,
            benchmark_cases[case_index].total_seconds / repeat_count,
            benchmark_cases[case_index].best_seconds
        );
    }

    return EXIT_SUCCESS;
}
