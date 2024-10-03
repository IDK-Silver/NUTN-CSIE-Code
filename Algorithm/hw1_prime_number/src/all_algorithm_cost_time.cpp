#include <iostream>
#include <big_int>
#include <prime>
#include <vector>
#include <filesystem>
#include <system_error>
#include <fstream>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <set>
#include <chrono> // 已包含
#include "core/prime_runner.hpp"
#include "core/config.hpp"

namespace fs = std::filesystem;

#define __is_print_log__ true

// 遞歸創建目錄
bool create_directories_recursively(const fs::path& path) {
    std::error_code ec;
    if (fs::create_directories(path, ec)) {
        std::cout << "成功創建資料夾：" << path << std::endl;
        return true;
    } else if (fs::exists(path)) {
        std::cout << "資料夾已存在：" << path << std::endl;
        return true;
    } else {
        std::cerr << "創建資料夾失敗：" << path << " - " << ec.message() << std::endl;
        return false;
    }
}

std::mutex cout_mutex;
std::atomic<bool> should_stop(false);


void sieve_of_eratosthenes(const size_t & start, const size_t & end, std::vector<std::pair<BigInt, double>> &primes_with_times) {
    if (end < 2) return;

    // 初始化 sieve
    std::vector<bool> sieve(end + 1, true);
    sieve[0] = sieve[1] = false;

    // 開始計時
    auto sieve_start_time = std::chrono::high_resolution_clock::now();

    for (size_t p = 2; p * p <= end; ++p) {
        if (sieve[p]) {
            for (size_t i = p * p; i <= end; i += p) {
                sieve[i] = false;
            }

        }
    }
    
    
    

    auto current_time = std::chrono::high_resolution_clock::now();
    size_t num_of_prime = 0;
    // 收集所有質數並記錄時間
    for (size_t p = (start > 2 ? start : 2); p <= end; ++p) {
        if (sieve[p]) {
            num_of_prime++;
        }
    }
    
    std::chrono::duration<double, std::milli> elapsed = (current_time - sieve_start_time) / num_of_prime;


    // 收集所有質數並記錄時間
    for (size_t p = (start > 2 ? start : 2); p <= end; ++p) {
        if (sieve[p]) {
            primes_with_times.emplace_back(BigInt(p), elapsed.count());
        }
    }
}
// 修改後的 find_primes 函數，記錄每個質數的查找時間（非篩法算法）
void find_primes(const std::string& algorithm, const BigInt& start, const BigInt& end, const int& try_times, std::vector<std::pair<BigInt, double>>& primes_with_times) {
    BigInt current = start;

    while (current <= end && !should_stop.load()) {
        auto prime_start = std::chrono::high_resolution_clock::now();

        // 使用 is_prime 函數判斷 current 是否為質數
        bool isPrime = core::is_prime(algorithm, current, try_times);

        auto prime_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = prime_end - prime_start;

        if (isPrime) {
            primes_with_times.emplace_back(current, elapsed.count());

        }

        current = current + 1;
    }
}

// 新增的 find_primes_sieve 函數，用於篩法算法，記錄每個質數的查找時間
void find_primes_sieve(const std::string& algorithm, const BigInt& start, const BigInt& end, std::vector<std::pair<BigInt, double>>& primes_with_times) {
    // 將 BigInt 轉換為 size_t
    size_t start_size_t = std::stoull(start.to_string());
    size_t end_size_t = std::stoull(end.to_string());

    // 確保轉換後的值不超過 size_t 的範圍
    if (start > BigInt(std::to_string(std::numeric_limits<size_t>::max())) ||
        end > BigInt(std::to_string(std::numeric_limits<size_t>::max()))) {
        throw std::runtime_error("輸入值超出 size_t 的範圍");
    }

    // 執行篩法並記錄每個質數的查找時間
    sieve_of_eratosthenes(start_size_t, end_size_t, primes_with_times);
}

// 修改後的 run_prime_test_return 函數，返回質數列表及每個質數的查找時間
std::pair<std::vector<std::pair<BigInt, double>>, double> run_prime_test_return(const std::string& algorithm, const BigInt& start, const BigInt& end, const int& try_times) {
    if(__is_print_log__)
        std::cout << "開始使用 " << algorithm << " 算法查找質數..." << std::endl;
    std::vector<std::pair<BigInt, double>> all_primes_with_times;

    // 如果算法是 sieve，直接調用 find_primes_sieve
    if (algorithm == "sieve") {
        auto sieve_start_time = std::chrono::high_resolution_clock::now();
        find_primes_sieve(algorithm, start, end, all_primes_with_times);
        auto sieve_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = sieve_end_time - sieve_start_time;
        double elapsed_ms = elapsed.count() * 1000.0; // 轉換為毫秒

        if(__is_print_log__)
            std::cout << algorithm << " 質數查找完成，共找到 " << all_primes_with_times.size() << " 個質數。" << std::endl;
        if(__is_print_log__)
            std::cout << algorithm << " 運行時間：" << elapsed_ms << " 毫秒" << std::endl;

        return { all_primes_with_times, elapsed_ms };
    }

    // 非篩法算法處理
    unsigned int thread_count = std::thread::hardware_concurrency();
    unsigned int task_count = thread_count * 1; // 可根據需要調整倍數
    std::vector<std::thread> threads;
    std::vector<std::vector<std::pair<BigInt, double>>> thread_primes_with_times(task_count);

    BigInt range = end - start + 1;
    BigInt chunk_size = range / task_count;

    // 開始計時
    auto start_time = std::chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < task_count; i++) {
        BigInt thread_start = start + (chunk_size * i);
        BigInt thread_end = (i == task_count - 1) ? end : thread_start + chunk_size - 1;
        threads.emplace_back(find_primes, algorithm, thread_start, thread_end, try_times, std::ref(thread_primes_with_times[i]));
    }

    // 等待所有線程完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 合併所有質數和時間
    for (const auto& thread_primes : thread_primes_with_times) {
        all_primes_with_times.insert(all_primes_with_times.end(), thread_primes.begin(), thread_primes.end());
    }

    // 按質數大小排序
    std::sort(all_primes_with_times.begin(), all_primes_with_times.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // 結束計時
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double elapsed_ms = elapsed.count() * 1000.0; // 轉換為毫秒

    if(__is_print_log__)
        std::cout << algorithm << " 質數查找完成，共找到 " << all_primes_with_times.size() << " 個質數。" << std::endl;
    if(__is_print_log__)
        std::cout << algorithm << " 運行時間：" << elapsed_ms << " 毫秒" << std::endl;

    return { all_primes_with_times, elapsed_ms };
}

int main(int argc, char* argv[]) {
    std::cout << "argc: " << argc << std::endl;
    if (argc != 9) {
        std::cerr << "用法: " << argv[0] << " --n_start <n_start> --n_end <n_end> --fermat_try_time <fermat_try_time> --miller_rabin_try_time <miller_rabin_try_time>" << std::endl;
        return 1;
    }

    BigInt n_start, n_end;
    int fermat_try_time, miller_rabin_try_time;

    for (int i = 1; i < argc; i += 1) {
        std::string arg = argv[i];
        if (arg == "--n_start") {
            n_start = BigInt(argv[i + 1]);
        } else if (arg == "--n_end") {
            n_end = BigInt(argv[i + 1]);
        } else if (arg == "--fermat_try_time") {
            fermat_try_time = std::stoi(argv[i + 1]);
        } else if (arg == "--miller_rabin_try_time") {
            miller_rabin_try_time = std::stoi(argv[i + 1]);
        }
    }

    if (n_start < 2 || n_end <= n_start) {
        std::cerr << "錯誤：n_start 必須大於等於 2，且 n_end 必須大於 n_start" << std::endl;
        return 1;
    }

    if (fermat_try_time < 1 || miller_rabin_try_time < 1) {
        std::cerr << "錯誤：try_time 必須大於等於 1" << std::endl;
        return 1;
    }


    // 定義算法列表，包含 sieve
    std::vector<std::string> algorithms = {"fermat", "miller-rabin", "basic", "sieve"};

    auto result_folder_path = fs::path("result/all_algorithm");
    fs::create_directories(result_folder_path); // 確保目錄存在
    
    auto result_file = result_folder_path / "cost_time.csv";

    // 創建並打開文件
    std::ofstream result_file_stream(result_file);
    if (!result_file_stream.is_open()) {
        std::cerr << "無法創建總結果文件：" << result_file << std::endl;
        return 1;
    }

    // 紀錄算法參數
    std::ofstream param_file(result_folder_path / "algorithm_params.txt");
    if (param_file.is_open()) {
        param_file << "fermat_try_time: " << fermat_try_time << "\n";
        param_file << "miller_rabin_try_time: " << miller_rabin_try_time << "\n";
        param_file << "n_start: " << n_start.to_string() << "\n";
        param_file << "n_end: " << n_end.to_string() << "\n";
        param_file.close();
    } else {
        std::cerr << "無法創建參數文件" << std::endl;
    }
    // 寫入 CSV 標題
    result_file_stream << "算法,n,time_ms\n";

    for (const auto& algorithm : algorithms) {
        // 設定 try_times 根據算法
        int try_times = 1; // 預設值
        if (algorithm == "fermat") {
            try_times = fermat_try_time;
        } else if (algorithm == "miller-rabin") {
            try_times = miller_rabin_try_time;
        }

        // 定義任務範圍
        unsigned int num_cores = std::thread::hardware_concurrency();
        const int k = 1;
        unsigned int total_tasks = (algorithm != "sieve") ? (num_cores * k) : 1; // 篩法單獨處理

        std::vector<std::pair<BigInt, BigInt>> tasks;
        if (algorithm == "sieve") {
            tasks.emplace_back(n_start, n_end);
        } else {
            BigInt range = n_end - n_start + 1;
            BigInt chunk_size = range / total_tasks;
            for (unsigned int i = 0; i < total_tasks; i++) {
                BigInt task_start = n_start + (chunk_size * i);
                BigInt task_end = (i == total_tasks - 1) ? n_end : task_start + chunk_size - 1;
                tasks.emplace_back(task_start, task_end);
            }
        }
        
        // 對每個任務運行質數測試
        for (const auto& task : tasks) {
            int current_try_times = try_times;
            if (algorithm == "sieve") {
                current_try_times = 0; // 篩法不使用 try_times
            }

            auto [primes_with_times, elapsed_ms] = run_prime_test_return(algorithm, task.first, task.second, current_try_times);
            
            // 將結果寫入 CSV 文件
            for (const auto& [prime, time_ms] : primes_with_times) {
                result_file_stream << algorithm << "," << prime << "," << time_ms << "\n";
            }

            // 如果是篩法，因為每個質數已經有自己的 time_ms，所以無需額外處理
        }

        // 刷新文件流
        result_file_stream.flush();
    }

    result_file_stream.close();
    std::cout << "所有結果已保存到文件：" << result_file << std::endl;

    return 0;
}