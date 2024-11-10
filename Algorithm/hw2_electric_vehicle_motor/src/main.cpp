#include <iostream>
#include <algorithm>
#include <fstream>
#include <core>
#include <string>

void print_help() {
    std::cout << "使用方式: program [選項]\n"
              << "選項:\n"
              << "  --help                顯示此幫助訊息\n"
              << "  --stand <檔案路徑>    指定充電站資料檔案 (預設: stands.csv)\n"
              << "  --people <檔案路徑>   指定電動機車資料檔案 (預設: people.csv)\n";
    exit(0);
}

int main(int argc, char* argv[]) {
    std::string stands_file = "stands.csv";
    std::string people_file = "people.csv";


    if (argc == 1) {
        std::cerr << "錯誤：未提供任何參數\n"
                  << "使用 --help 查看使用說明\n";
        return 1;
    }

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_help();
        }

        if (i + 1 >= argc) {
            std::cerr << "錯誤：參數 " << arg << " 需要一個值\n"
                      << "使用 --help 查看使用說明\n";
            return 1;
        }
        
        if (arg == "--stand") {
            stands_file = argv[++i];
        } else if (arg == "--people") {
            people_file = argv[++i];
        } else {
            std::cerr << "未知的參數: " << arg << "\n"
                      << "使用 --help 查看使用說明\n";
            return 1;
        }
    }

    // Check if files exist
    std::ifstream stands_check(stands_file);
    if (!stands_check.good()) {
        std::cerr << "錯誤：找不到充電站資料檔案 " << stands_file << "\n";
        return 1;
    }
    stands_check.close();

    std::ifstream people_check(people_file); 
    if (!people_check.good()) {
        std::cerr << "錯誤：找不到電動機車資料檔案 " << people_file << "\n";
        return 1;
    }
    people_check.close();

    // 讀取充電站資料
    std::vector<MotorStand> stands = read_motor_stands(stands_file);
    std::cout << "充電站列表:\n";
    for (const auto& stand : stands) {
        std::cout << stand << std::endl;
    }

    // 讀取電動機車資料
    std::vector<MotorPeople> people = read_motor_people(people_file);
    std::cout << "\n電動機車列表:\n";
    for (auto& person : people) {
        person.set_cost_kwh_per_km(10);
        std::cout << person << std::endl;
    }

    // (1) 哪兩個充電站靠最近? 距離多少? (brute-force algorithm)
    auto closest_stands = find_closest_points(stands);
    std::cout << "\n(1) 最接近的兩個充電站:\n\t"
              << closest_stands.first.get_name() << " 和 " 
              << closest_stands.second.get_name() 
              << "，距離: " << closest_stands.first.get_distance(closest_stands.second) << "\n";

    // (2) 哪兩台電動機車靠最近? 距離多少? (brute-force algorithm)
    auto closest_people = find_closest_points(people);
    std::cout << "\n(2) 最接近的兩台電動機車:\n\t"
              << closest_people.first.get_name() << " 和 " 
              << closest_people.second.get_name() 
              << "，距離: " << closest_people.first.get_distance(closest_people.second) << "\n";

    // (3) m 個充電站的範圍有多大(Convex-Hull，它的面積以及最遠的距離)? (brute-force algorithm)
    auto hull_stands = find_convex_hull(stands);
    double area_stands = calculate_area(hull_stands);
    auto diameter_stands = find_diameter(hull_stands);
    std::cout << "\n(3) 充電站的 Convex Hull 包含 " << hull_stands.size() << " 個點：\n";
    for (const auto& point : hull_stands) {
        std::cout << point << "\n";
    }
    std::cout << "充電站覆蓋面積: " << area_stands << " 平方單位\n";
    std::cout << "充電站中最遠的兩個點是:\n\t" 
              << diameter_stands.first << "\n\t和\n\t" 
              << diameter_stands.second 
              << "\n距離: " << diameter_stands.first.get_distance(diameter_stands.second) << "\n";

    // (4) n 台電動機車的範圍有多大(Convex-Hull，它的面積以及最遠的距離)? (brute-force algorithm)
    auto hull_people = find_convex_hull(people);
    double area_people = calculate_area(hull_people);
    auto diameter_people = find_diameter(hull_people);
    std::cout << "\n(4) 電動機車的 Convex Hull 包含 " << hull_people.size() << " 個點：\n";
    for (const auto& point : hull_people) {
        std::cout << point << "\n";
    }
    std::cout << "電動機車覆蓋面積: " << area_people << " 平方單位\n";
    std::cout << "電動機車中最遠的兩個點是:\n\t" 
              << diameter_people.first << "\n\t和\n\t" 
              << diameter_people.second 
              << "\n距離: " << diameter_people.first.get_distance(diameter_people.second) << "\n";

    // (5) 假設 n=m 且每一個充電站只有一個空位，要以最省電的方式 n 個人騎 u-motor 到充電站，如何配對，以及求出最少的總耗能? (Hungarian method)
    if (stands.size() == people.size()) {
        // 建立成本矩陣
        std::vector<std::vector<double>> cost_matrix;
        for (const auto& person : people) {
            std::vector<double> costs;
            for (const auto& stand : stands) {
                double cost = person.get_cost_to_point(stand);
                costs.push_back(cost);
            }
            cost_matrix.push_back(costs);
        }

        // 應用匈牙利演算法
        HungarianAlgorithm hungarian(cost_matrix);
        std::vector<int> assignment;
        double total_cost = hungarian.solve(assignment);

        // 輸出配對結果
        std::cout << "\n(5) 最省電的配對結果:\n";
        for (size_t i = 0; i < assignment.size(); ++i) {
            if (assignment[i] != -1) {
                std::cout << "\t" << people[i].get_name() 
                          << " 分配到 " << stands[assignment[i]].get_name() 
                          << "，耗能: " << cost_matrix[i][assignment[i]] << "\n";
            } else {
                std::cout << "\t" << people[i].get_name() << " 未分配到任何充電站。\n";
            }
        }
        std::cout << "總最小耗能: " << total_cost << "\n";
    } else {
        std::cout << "\n(5) 無法應用匈牙利演算法，因為充電站和電動機車的數量不相等。\n";
    }

    // (6) 如(5)所述，假設 n≠m，第 i 個充電站空位有v_i 個, v_i ≥ 0, 1 ≤ i ≤ m，如何解決這個問題? (Hungarian method)
    // 假設充電站的空位數量已在 stands 中設定
    std::vector<int> stand_slots;
    for (const auto& stand : stands) {
        stand_slots.push_back(stand.get_capacity());
    }
    int total_slots = 0;
    for (const auto& slot : stand_slots) {
        total_slots += slot;
    }

    if (people.size() != stands.size() || std::any_of(stand_slots.begin(), stand_slots.end(), [](int v){ return v > 1; })) {
        // 擴展充電站空位
        std::vector<MotorStand> expanded_stands;
        std::vector<std::string> stand_names;
        for (size_t i = 0; i < stands.size(); ++i) {
            for (int j = 0; j < stand_slots[i]; ++j) {
                MotorStand expanded = stands[i];
                expanded.set_name(stands[i].get_name() + "_" + std::to_string(j));
                expanded_stands.push_back(expanded);
                stand_names.push_back(expanded.get_name());
            }
        }

        if (people.size() > expanded_stands.size()) {
            std::cerr << "\n(6) 錯誤：充電站空位不足以容納所有人。\n";
        } else {
            // 建立成本矩陣
            std::vector<std::vector<double>> extended_cost_matrix;
            for (const auto& person : people) {
                std::vector<double> costs;
                for (const auto& stand : expanded_stands) {
                    double cost = person.get_cost_to_point(stand);
                    costs.push_back(cost);
                }
                extended_cost_matrix.push_back(costs);
            }

            // 應用擴展的匈牙利演算法
            HungarianAlgorithm hungarian_extended(extended_cost_matrix);
            std::vector<int> assignment_extended;
            double total_cost_extended = hungarian_extended.solve(assignment_extended);

            // 輸出配對結果
            std::cout << "\n(6) 最省電的配對結果 (n ≠ m):\n";
            for (size_t i = 0; i < assignment_extended.size(); ++i) {
                if (assignment_extended[i] != -1) {
                    std::cout << "\t" << people[i].get_name() 
                              << " 分配到 " << stand_names[assignment_extended[i]] 
                              << "，耗能: " << extended_cost_matrix[i][assignment_extended[i]] << "\n";
                } else {
                    std::cout << "\t" << people[i].get_name() << " 未分配到任何充電站。\n";
                }
            }
            std::cout << "總最小耗能: " << total_cost_extended << "\n";
        }
    }

    return 0;
}