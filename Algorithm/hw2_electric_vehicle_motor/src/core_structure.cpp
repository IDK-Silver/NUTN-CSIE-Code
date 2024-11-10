#include "core_structure.hpp"

Point::Point(double x, double y) : x(x), y(y) {
    
}

Point::~Point() {}

void Point::set_point(double x, double y) {
    this->x = x;
    this->y = y;
}

void Point::set_x(double x) {
    this->x = x;
}

void Point::set_y(double y) {
    this->y = y;
}

double Point::get_x() const {
    return x;
}

double Point::get_y() const {
    return y;
}

double Point::get_distance(const Point& other) const {
    return sqrt(pow(abs(x - other.x), 2) + pow(abs(y - other.y), 2));
}

MotorStand::MotorStand(double x, double y) : Point(x, y), capacity(0), motor_count(0) {
}

MotorStand::~MotorStand() {
}

void MotorStand::set_capacity(uint32_t capacity) {
    this->capacity = capacity;
}

uint32_t MotorStand::get_capacity() const {
    return capacity;
}

void MotorStand::set_motor_count(uint32_t motor_count) {
    this->motor_count = motor_count;
}

uint32_t MotorStand::get_motor_count() const {
    return motor_count;
}

uint32_t MotorStand::get_remaining_capacity() const {
    return (motor_count <= capacity) ? capacity - motor_count : 0;
}

void MotorStand::set_name(std::string name) {
    this->name = name;
}

std::string MotorStand::get_name() const {
    return name;
}

MotorPeople::MotorPeople(double x, double y) : Point(x, y), cost_kwh_per_km(0) {
}

MotorPeople::~MotorPeople() {
}

void MotorPeople::set_cost_kwh_per_km(double cost_kwh_per_km) {
    this->cost_kwh_per_km = cost_kwh_per_km;
}

double MotorPeople::get_cost_kwh_per_km() const {
    return cost_kwh_per_km;
}

double MotorPeople::get_cost_to_point(const Point& other) const {
    uint32_t distance = this->get_distance(other);
    return distance * cost_kwh_per_km;
}

void MotorPeople::set_name(std::string name) {
    this->name = name;
}

std::string MotorPeople::get_name() const {
    return name;
}

std::ostream& operator<<(std::ostream& os, const MotorStand& stand) {
    os << "Stand[" << stand.name << "] at (" 
       << stand.get_x() << "," << stand.get_y() 
       << ") capacity:" << stand.capacity 
       << " current:" << stand.motor_count;
    return os;
}

std::ostream& operator<<(std::ostream& os, const MotorPeople& people) {
    os << "People[" << people.name << "] at (" 
       << people.get_x() << "," << people.get_y() 
       << ") cost/km:" << people.cost_kwh_per_km;
    return os;
}

