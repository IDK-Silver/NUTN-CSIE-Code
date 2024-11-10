#ifndef CORE_PROCESS_IO_HPP
#define CORE_PROCESS_IO_HPP


#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "core_structure.hpp"

std::vector<MotorStand> read_motor_stands(std::filesystem::path);
std::vector<MotorPeople> read_motor_people(std::filesystem::path);

#endif // CORE_PROCESS_IO_HPP
