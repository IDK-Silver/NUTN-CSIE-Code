#include "config.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

namespace core {

BasicConfig::BasicConfig() {}

BasicConfig::~BasicConfig() {}

void BasicConfig::load_config(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "無法打開配置文件: " << config_file << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key, value;

        if (std::getline(iss, key, ':') && std::getline(iss >> std::ws, value)) {
            // 去除鍵和值的前後空格
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            datas[key] = value;
        }
    }

    file.close();
}

std::string BasicConfig::get_value(const std::string& key) const {
    auto it = datas.find(key);
    if (it != datas.end()) {
        return it->second;
    }
    return "";  // 如果找不到鍵，返回空字符串
}

}  // namespace core
