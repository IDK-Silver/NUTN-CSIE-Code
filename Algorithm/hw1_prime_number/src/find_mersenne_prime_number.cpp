#include <iostream>
#include <big_int>
#include <prime>
#include <vector>
#include <filesystem>
#include <fstream>
#include <chrono>
#include "core/prime_runner.hpp"
#include "core/config.hpp"

namespace fs = std::filesystem;

// 函數來生成 2^p - 1
BigInt generate_mersenne_numer(BigInt exponent) {
    BigInt result = 1;
    for (BigInt i = 0; i < exponent; i++) {
        result = result * 2;
    }
    result = result - 1;
    return result;
}

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

int main(int argc, char* argv[]) {



    // 創建結果資料夾
    fs::path result_path = "result/find_mersenne_prime_number";
    create_directories_recursively(result_path);

    fs::path mersenne_file_path = result_path / "mersenne_prime_number.txt";
    fs::path last_try_file_path = result_path / "last_try_number.txt";

    // 讀取 last_try_number.txt 以獲取最後嘗試的數字
    std::fstream last_try_file(last_try_file_path.string(), std::ios::in);
    BigInt last_p = 0;

    if (last_try_file.is_open()) {
        std::string last_try_line;
        if (std::getline(last_try_file, last_try_line)) {
            if (!last_try_line.empty()) {
                last_p = BigInt(last_try_line);
            }
        }
        last_try_file.close();
    } else {
        // 如果 last_try_number.txt 不存在，檢查 mersenne_prime_number.txt
        std::fstream mersenne_file(mersenne_file_path.string(), std::ios::in);
        if (mersenne_file.is_open()) {
            std::string last_line;
            while (std::getline(mersenne_file, last_line)) {
                // 讀取最後一行
            }
            if (!last_line.empty()) {
                last_p = BigInt(last_line);
            }
            mersenne_file.close();
        }
    }

    std::cout << "last_try_number: " << last_p << std::endl;

    // 開啟 last_try_number.txt 以寫入最後嘗試的數字
    std::fstream last_try_out(last_try_file_path.string(), std::ios::out | std::ios::trunc);
    if (!last_try_out.is_open()) {
        std::cerr << "無法打開文件：" << last_try_file_path << std::endl;
        return 1;
    }

    // 開啟 mersenne_prime_number.txt 以追加新找到的素數
    std::fstream mersenne_file(mersenne_file_path.string(), std::ios::out | std::ios::app);
    if (!mersenne_file.is_open()) {
        std::cerr << "無法打開文件：" << mersenne_file_path << std::endl;
        return 1;
    }
    last_p = last_p + 1;
    // 設定執行時間限制（例如：1小時）

    // 預設為12小時
    int run_minutes = 1;  
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--run_min" && i + 1 < argc) {
            run_minutes = std::stoi(argv[i + 1]);
            break;
        }
    }
    
    // 將分鐘轉換為小時
    const auto time_limit = std::chrono::minutes(run_minutes);
    auto start_time = std::chrono::steady_clock::now();
    auto current_time = std::chrono::steady_clock::now();
    std::cout << "設定執行時間限制為: " << run_minutes << " 分鐘" << std::endl;
    for (BigInt i = last_p; ; i++) {
        // 檢查是否超過時間限制
        current_time = std::chrono::steady_clock::now();
        if (current_time - start_time > time_limit) {
            std::cout << "已達到時間限制，停止執行。" << std::endl;
            break;
        }

        auto mersenne_number = generate_mersenne_numer(i);

        // 更新 last_try_number.txt
        last_try_out.seekp(0);
        last_try_out << i << std::endl;

        if (core::is_prime("miller-rabin", mersenne_number, 3)) {
            mersenne_file << i << std::endl;
            std::cout << "找到梅森素數：2^" << i << " - 1" << std::endl;
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "實際執行時間：" << duration.count() << " 秒" << std::endl;

    last_try_out.close();
    mersenne_file.close();
    return 0;
}