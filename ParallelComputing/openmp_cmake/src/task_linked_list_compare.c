#include <omp.h>

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Node {
    size_t index;
    int value;
    int work_units;
    struct Node *next;
} Node;

typedef void (*ProcessFunction)(
    const Node *head,
    uint64_t *results,
    int thread_count,
    long long *task_counts
);

typedef struct {
    const char *name;
    ProcessFunction process;
    uint64_t checksum;
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
        "Usage: %s <thread_count> <node_count> <work_scale> [repeat_count]\n"
        "       %s --help\n"
        "\n"
        "Arguments:\n"
        "  thread_count Number of OpenMP threads used by the task case.\n"
        "  node_count   Number of nodes in the linked list.\n"
        "  work_scale   Base CPU work per node; each node gets a different amount.\n"
        "  repeat_count Optional number of timing repetitions for averaging.\n"
        "               Default: 1\n"
        "\n"
        "The benchmark compares serial linked-list processing with an OpenMP task\n"
        "version that uses parallel, single, task, firstprivate, and taskwait.\n",
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

static Node *allocate_nodes(size_t node_count) {
    Node *nodes = NULL;

    if (node_count > SIZE_MAX / sizeof(*nodes)) {
        fprintf(stderr, "node_count is too large to allocate.\n");
        exit(EXIT_FAILURE);
    }

    nodes = malloc(node_count * sizeof(*nodes));
    if (nodes == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    return nodes;
}

static uint64_t *allocate_results(size_t node_count) {
    uint64_t *results = NULL;

    if (node_count > SIZE_MAX / sizeof(*results)) {
        fprintf(stderr, "node_count is too large to allocate results.\n");
        exit(EXIT_FAILURE);
    }

    results = malloc(node_count * sizeof(*results));
    if (results == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    return results;
}

static long long *allocate_task_counts(int thread_count) {
    long long *task_counts = calloc((size_t)thread_count, sizeof(*task_counts));

    if (task_counts == NULL) {
        perror("calloc");
        exit(EXIT_FAILURE);
    }

    return task_counts;
}

static void initialize_nodes(Node *nodes, size_t node_count, int work_scale) {
    for (size_t index = 0; index < node_count; ++index) {
        int multiplier = (int)((index % 17) + 1);

        if (index % 11 == 0) {
            multiplier += 7;
        }

        nodes[index].index = index;
        nodes[index].value = (int)((index * 37 + 101) % 1009);
        nodes[index].work_units = work_scale * multiplier;
        nodes[index].next = index + 1 < node_count ? &nodes[index + 1] : NULL;
    }
}

static uint64_t process_node(const Node *node) {
    uint64_t state = (uint64_t)node->value + (uint64_t)(node->index + 1) * 97ULL;

    for (int iteration = 0; iteration < node->work_units; ++iteration) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state += (uint64_t)iteration + 0x9E3779B97F4A7C15ULL;
    }

    return state;
}

static void serial_process_list(
    const Node *head,
    uint64_t *results,
    int thread_count,
    long long *task_counts
) {
    const Node *current = head;

    (void)thread_count;
    (void)task_counts;

    while (current != NULL) {
        results[current->index] = process_node(current);
        current = current->next;
    }
}

static void task_process_list(
    const Node *head,
    uint64_t *results,
    int thread_count,
    long long *task_counts
) {
    #pragma omp parallel num_threads(thread_count)
    {
        #pragma omp single
        {
            const Node *current = head;

            while (current != NULL) {
                #pragma omp task firstprivate(current) shared(results, task_counts)
                {
                    int thread_id = omp_get_thread_num();

                    results[current->index] = process_node(current);
                    task_counts[thread_id]++;
                }

                current = current->next;
            }

            #pragma omp taskwait
        }
    }
}

static uint64_t checksum_results(const uint64_t *results, size_t node_count) {
    uint64_t checksum = 0;

    for (size_t index = 0; index < node_count; ++index) {
        checksum ^= results[index] + 0x9E3779B97F4A7C15ULL
            + (checksum << 6) + (checksum >> 2);
    }

    return checksum;
}

static void run_benchmark_case(
    BenchmarkCase *benchmark_case,
    const Node *head,
    uint64_t *results,
    size_t node_count,
    int thread_count,
    int repeat_count,
    uint64_t expected_checksum,
    long long *task_counts
) {
    benchmark_case->total_seconds = 0.0;
    benchmark_case->best_seconds = -1.0;

    for (int repetition = 0; repetition < repeat_count; ++repetition) {
        double elapsed_seconds = 0.0;
        double start_time = 0.0;

        memset(task_counts, 0, (size_t)thread_count * sizeof(*task_counts));

        start_time = omp_get_wtime();
        benchmark_case->process(head, results, thread_count, task_counts);
        elapsed_seconds = omp_get_wtime() - start_time;

        benchmark_case->checksum = checksum_results(results, node_count);
        if (benchmark_case->checksum != expected_checksum) {
            fprintf(
                stderr,
                "%s checksum mismatch: expected %llu, got %llu\n",
                benchmark_case->name,
                (unsigned long long)expected_checksum,
                (unsigned long long)benchmark_case->checksum
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
        { "serial", serial_process_list, 0, 0.0, 0.0 },
        { "task", task_process_list, 0, 0.0, 0.0 },
    };
    const int benchmark_case_count = (int)(
        sizeof(benchmark_cases) / sizeof(benchmark_cases[0])
    );
    Node *nodes = NULL;
    uint64_t *results = NULL;
    long long *task_counts = NULL;
    int thread_count = 0;
    int repeat_count = 1;
    int work_scale = 0;
    size_t node_count = 0;
    uint64_t expected_checksum = 0;

    if (is_help_requested(argc, argv)) {
        print_usage(argv[0], stdout);
        return EXIT_SUCCESS;
    }

    if (argc < 4 || argc > 5) {
        print_usage(argv[0], stderr);
        return EXIT_FAILURE;
    }

    thread_count = (int)parse_long_long(argv[1], "thread_count", 1, INT_MAX);
    node_count = (size_t)parse_long_long(argv[2], "node_count", 1, INT_MAX);
    work_scale = (int)parse_long_long(argv[3], "work_scale", 1, INT_MAX / 32);

    if (argc == 5) {
        repeat_count = (int)parse_long_long(argv[4], "repeat_count", 1, INT_MAX);
    }

    nodes = allocate_nodes(node_count);
    results = allocate_results(node_count);
    task_counts = allocate_task_counts(thread_count);
    initialize_nodes(nodes, node_count, work_scale);
    omp_set_dynamic(0);

    serial_process_list(nodes, results, thread_count, task_counts);
    expected_checksum = checksum_results(results, node_count);

    for (int case_index = 0; case_index < benchmark_case_count; ++case_index) {
        run_benchmark_case(
            &benchmark_cases[case_index],
            nodes,
            results,
            node_count,
            thread_count,
            repeat_count,
            expected_checksum,
            task_counts
        );
    }

    printf("OpenMP task linked-list benchmark\n");
    printf("thread_count = %d\n", thread_count);
    printf("node_count = %zu\n", node_count);
    printf("work_scale = %d\n", work_scale);
    printf("repeat_count = %d\n", repeat_count);
    printf("\n");
    printf("%-10s %20s %16s %16s\n", "method", "checksum", "avg_seconds", "best_seconds");

    for (int case_index = 0; case_index < benchmark_case_count; ++case_index) {
        printf(
            "%-10s %20llu %16.9f %16.9f\n",
            benchmark_cases[case_index].name,
            (unsigned long long)benchmark_cases[case_index].checksum,
            benchmark_cases[case_index].total_seconds / repeat_count,
            benchmark_cases[case_index].best_seconds
        );
    }

    printf("\n");
    printf("task executions by thread from the last task run:\n");
    for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
        printf("thread %d: %lld\n", thread_index, task_counts[thread_index]);
    }

    free(task_counts);
    free(results);
    free(nodes);
    return EXIT_SUCCESS;
}
