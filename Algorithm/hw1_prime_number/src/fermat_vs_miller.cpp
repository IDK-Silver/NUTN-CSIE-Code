// src/fermat_vs_miller.cpp

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
#include "core/prime_runner.hpp"
#include "core/config.hpp"

namespace fs = std::filesystem;

#define __is_print_log__ false

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

// 重構的 find_primes 函數
void find_primes(const std::string& algorithm, const BigInt& start, const BigInt& end, const int& try_times, std::vector<BigInt>& primes) {
    std::vector<BigInt> thread_primes = core::find_primes_in_range(algorithm, start, end, try_times);
    
    {
        if (__is_print_log__) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            for (const auto& prime : thread_primes) {
                std::cout << algorithm << ": " << prime << " 是質數" << std::endl;
            }
        }
    }
    
    primes = std::move(thread_primes);
}

// 修改後的 run_prime_test_return 函數，直接返回質數列表
std::vector<BigInt> run_prime_test_return(const std::string& algorithm, const BigInt& start, const BigInt& end, const int& try_times) {

    std::cout << "開始使用 " << algorithm << " 算法查找質數..." << std::endl;

    std::vector<BigInt> all_primes;

    unsigned int thread_count = std::thread::hardware_concurrency();
    unsigned int task_count = thread_count * 4;
    std::vector<std::thread> threads;
    std::vector<std::vector<BigInt>> thread_primes(task_count);

    BigInt range = end - start + 1;
    BigInt chunk_size = range / task_count;

    for (unsigned int i = 0; i < task_count; ++i) {
        BigInt thread_start = start + (chunk_size * i);
        BigInt thread_end = (i == task_count - 1) ? end : thread_start + chunk_size - 1;
        threads.emplace_back(find_primes, algorithm, thread_start, thread_end, try_times, std::ref(thread_primes[i]));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (const auto& primes : thread_primes) {
        all_primes.insert(all_primes.end(), primes.begin(), primes.end());
    }

    std::sort(all_primes.begin(), all_primes.end());

    if (__is_print_log__) {
        std::cout << algorithm << " 質數查找完成。" << std::endl;
    }

    return all_primes;
}

int main(int argc, char* argv[]) {
    // 定義默認值
    std::vector<int> n_list = {100, 1000, 10000};
    int try_time_start = 1;
    int try_time_end = 10;

    // 解析命令行參數
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--n_list") {
            n_list.clear();
            std::stringstream ss(argv[++i]);
            int n;
            while (ss >> n) {
                n_list.push_back(n);
                if (ss.peek() == ',')
                    ss.ignore();
            }
        } else if (arg == "--try_time_start") {
            try_time_start = std::stoi(argv[++i]);
        } else if (arg == "--try_time_end") {
            try_time_end = std::stoi(argv[++i]);
        }
    }

    fs::path folder_path = "result/fermat_vs_miller";
    
    std::cout << "準備創建資料夾：" << folder_path << std::endl;
    if (!create_directories_recursively(folder_path)) {
        std::cerr << "無法創建資料夾：" << folder_path << std::endl;
        return {};
    }

    // 輸出解析後的參數值（用於調試）
    std::cout << "n_list: ";
    for (int n : n_list) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    std::cout << "try_time_start: " << try_time_start << std::endl;
    std::cout << "try_time_end: " << try_time_end << std::endl;


    // 定義算法列表
    std::vector<std::string> algorithms = {"fermat", "miller-rabin"};

    // 遍歷每個 n
    for (const auto& n : n_list) {
        BigInt start = 2;
        BigInt end = n;

        // 首先運行基本測試，並保存結果
        std::string basic_output_file = "result/fermat_vs_miller/basic_" + std::to_string(n) + ".txt";
        std::vector<BigInt> basic_primes = core::find_primes_in_range("basic", start, end, 1);
        
        // 將 basic_primes 寫入文件
        std::ofstream basic_outfile(basic_output_file);
        if (basic_outfile.is_open()) {
            for (const auto& prime : basic_primes) {
                basic_outfile << prime << std::endl;
            }
            basic_outfile.close();
            std::cout << "基本測試結果已保存到：" << basic_output_file << std::endl;
        } else {
            std::cerr << "無法打開基本測試輸出文件：" << basic_output_file << std::endl;
            continue;
        }

        std::set<BigInt> basic_set(basic_primes.begin(), basic_primes.end());

        // 定義每個 n 的結果文件
        std::string result_filename = "result/fermat_vs_miller/n_" + std::to_string(n) + "_results.txt";
        std::ofstream result_file(result_filename);
        if (!result_file.is_open()) {
            std::cerr << "無法打開結果文件：" << result_filename << std::endl;
            continue;
        }

        // 寫入結果文件的標題
        result_file << "n: " << n << "\n";
        result_file << "try_time_start: " << try_time_start << "\n";
        result_file << "try_time_end: " << try_time_end << "\n\n";
        result_file << "算法, try_time, num_of_correct, num_of_total, accuracy(%)\n";

        // 遍歷每個 try_time
        for (int try_time = try_time_start; try_time <= try_time_end; ++try_time) {
            for (const auto& algorithm : algorithms) {
                // 運行質數測試並獲取結果
                std::vector<BigInt> test_primes = run_prime_test_return(algorithm, start, end, try_time);

                // 計算正確數量
                int correct = 0;
                for (const auto& prime : test_primes) {
                    if (basic_set.count(prime) > 0) {
                        correct++;
                    }
                }

                // 總數量
                int total = test_primes.size();

                // 計算準確率
                double accuracy = (total > 0) ? (static_cast<double>(correct) / total) * 100.0 : 0.0;

                // 寫入結果到結果文件
                result_file << algorithm << ", " << try_time << ", " << correct << ", " << total << ", " << accuracy << "\n";

                std::cout << "n=" << n << ", try_time=" << try_time << ", algorithm=" << algorithm 
                          << " 正確率: " << accuracy << "%" << std::endl;
            }
        }

        result_file.close();
        if (result_file) {
            std::cout << "結果已輸出到文件：" << result_filename << std::endl;
        } else {
            std::cerr << "無法正確寫入結果文件：" << result_filename << std::endl;
        }
    }

    return 0;
}