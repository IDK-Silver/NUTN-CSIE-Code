#include "core_process_io.hpp"
#include <fstream>
#include <sstream>
#include <string>

std::vector<MotorStand> read_motor_stands(std::filesystem::path path) {
    std::vector<MotorStand> stands;
    std::ifstream file(path);
    std::string line;
    
    // Skip header line
    std::getline(file, line);
    

    while (std::getline(file, line)) {
        std::vector<std::string> values;
        std::string value;
        for (auto& token : line) {
            if (token == ',') {
                values.push_back(value);
                value.clear();
            } else {
                value += token;
            }
        }
        values.push_back(value);

        MotorStand stand(0, 0);
        stand.set_name(values.at(0));
        stand.set_point(std::stod(values.at(1)), std::stod(values.at(2)));
        stand.set_capacity(std::stoul(values.at(3)));
        stand.set_motor_count(std::stoul(values.at(4)));

        stands.push_back(stand);    
    }
    
    return stands;
}


std::vector<MotorPeople> read_motor_people(std::filesystem::path path) {
    std::vector<MotorPeople> people;
    std::ifstream file(path);
    std::string line;
    
    // Skip header line
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        
        // Read name
        std::getline(ss, token, ',');
        std::string name = token;
        
        // Read x
        std::getline(ss, token, ',');
        uint32_t x = std::stoul(token);
        
        // Read y
        std::getline(ss, token);
        uint32_t y = std::stoul(token);
        
        // Create and configure MotorPeople
        MotorPeople person(x, y);
        person.set_name(name);
        
        people.push_back(person);
    }
    
    return people;
}
