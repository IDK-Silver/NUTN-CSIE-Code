#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <map>

namespace core {

class BasicConfig {
public:
    BasicConfig();
    ~BasicConfig();

    void load_config(const std::string& config_file);
    std::string get_value(const std::string& key) const;

private:
    std::map<std::string, std::string> datas;
};

}  // namespace core

#endif // CONFIG_HPP