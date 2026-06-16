#include "stencil_common.hpp"

#include <iomanip>

void print_usage(const char* program) {
    std::cout << "Usage:\n"
              << "  " << program << " [N] [benchmark_runs] [seed]\n\n";
    print_common_stencil_options();
    std::cout << "This target runs the 1D stencil sequentially on the CPU.\n";
}

int main(int argc, char** argv) {
    if (wants_help(argc, argv)) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    StencilArgs args;
    if (!parse_stencil_args(argc, argv, args)) {
        std::cerr << "Usage: " << argv[0] << " [N] [benchmark_runs] [seed]" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<int> input;
    std::vector<int> output;
    std::vector<int> expected;
    fill_input(input, args.n, args.seed);

    double avg_ms = benchmark_stencil_cpu(input, output, args.n, args.benchmark_runs);
    stencil_cpu_reference(input, expected, args.n);
    bool ok = verify_result(expected, output);

    std::cout << "1D stencil sequential CPU\n";
    std::cout << "N: " << args.n << '\n';
    std::cout << "Radius: " << STENCIL_RADIUS << '\n';
    std::cout << "Benchmark runs: " << args.benchmark_runs << '\n';
    std::cout << "Timing: stencil loop only\n\n";

    std::cout << std::left << std::setw(24) << "Version" << std::setw(18) << "Avg time"
              << std::setw(18) << "Checksum" << "Result\n";
    std::cout << std::string(64, '-') << '\n';
    std::cout << std::left << std::setw(24) << "CPU sequential" << std::setw(18)
              << (std::to_string(avg_ms) + " ms") << std::setw(18) << checksum(output)
              << (ok ? "OK" : "FAILED") << '\n';

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
