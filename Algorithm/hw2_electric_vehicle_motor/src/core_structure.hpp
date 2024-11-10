#ifndef CORE_STRUCTURE_HPP
#define CORE_STRUCTURE_HPP
#include <cstdint>
#include <cmath>
#include <string>
#include <iostream> 
class Point {
public:
    Point(double x, double y);
    ~Point();
    void set_point(double x, double y);
    void set_x(double x);
    void set_y(double y);
    double get_x() const;
    double get_y() const;
    double get_distance(const Point& other) const;
private:
    double x;
    double y;
};

class MotorStand : public Point {
public:
    MotorStand(double x, double y);
    ~MotorStand();
    void set_capacity(uint32_t capacity);
    uint32_t get_capacity() const;
    void set_motor_count(uint32_t motor_count);
    uint32_t get_motor_count() const;
    // Get remaining capacity of motor stand
    uint32_t get_remaining_capacity() const;
    void set_name(std::string name);
    std::string get_name() const;
    friend std::ostream& operator<<(std::ostream& os, const MotorStand& stand);
private:
    std::string name;
    uint32_t capacity;
    uint32_t motor_count;
};


class MotorPeople : public Point {
public:
    MotorPeople(double x, double y);
    ~MotorPeople();
    void set_cost_kwh_per_km(double cost_kwh_per_km);
    double get_cost_kwh_per_km() const;
    double get_cost_to_point(const Point& other) const;
    void set_name(std::string name);
    std::string get_name() const;
    friend std::ostream& operator<<(std::ostream& os, const MotorPeople& people);
    
private:
    std::string name;
    double cost_kwh_per_km;
};
#endif // CORE_STRUCTURE_HPP
