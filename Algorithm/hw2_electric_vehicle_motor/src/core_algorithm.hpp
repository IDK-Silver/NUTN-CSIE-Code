#ifndef CORE_ALGORITHM_HPP
#define CORE_ALGORITHM_HPP

#include "core_structure.hpp"
#include <vector>
#include <utility>
#include <limits>

template<typename T>
std::pair<T, T> find_closest_points(const std::vector<T>& points) {

    // If there are less than 2 points, return the same point twice
    if (points.size() < 2) {
        return {points.at(0), points.at(0)};
    }

    // Initialize the closest pair with the first two points
    std::pair<T, T> closest_pair = {points.at(0), points.at(1)};
    uint32_t min_distance = points.at(0).get_distance(points.at(1));

    // Iterate through all pairs of points to find the closest pair
    for (size_t i = 0; i < points.size(); i++) {
        for (size_t j = i + 1; j < points.size(); j++) {
            uint32_t dist = points.at(i).get_distance(points.at(j));
            if (dist < min_distance) {
                min_distance = dist;
                closest_pair = {points.at(i), points.at(j)};
            }
        }
    }

    return closest_pair;
}

template<typename T>
int orientation(const T& p, const T& q, const T& r) {
    int val = (q.get_y() - p.get_y()) * (r.get_x() - q.get_x()) -
              (q.get_x() - p.get_x()) * (r.get_y() - q.get_y());
    
    if (val == 0) return 0;
    return (val > 0) ? 1 : -1;
}

// Find convex hull of points
template<typename T>
std::vector<T> find_convex_hull(const std::vector<T>& points) {
    if (points.size() < 3) return points;
    
    // Find leftmost point
    size_t leftmost = 0;
    for (size_t i = 1; i < points.size(); i++) {
        if (points[i].get_x() < points[leftmost].get_x()) {
            leftmost = i;
        }
    }
    
    std::vector<T> hull;
    size_t p = leftmost;
    
    // Keep adding points to hull
    do {
        hull.push_back(points[p]);
        size_t q = (p + 1) % points.size();
        
        for (size_t i = 0; i < points.size(); i++) {
            if (orientation(points[p], points[i], points[q]) == -1) {
                q = i;
            }
        }
        
        p = q;
    } while (p != leftmost);
    
    return hull;
}


// Find maximum distance between any two points in hull
template<typename T>
std::pair<T, T> find_diameter(const std::vector<T>& hull) {
    if (hull.size() < 2) return {hull[0], hull[0]};
    
    std::pair<T, T> max_pair = {hull[0], hull[1]};
    uint32_t max_dist = hull[0].get_distance(hull[1]);
    
    for (size_t i = 0; i < hull.size(); i++) {
        for (size_t j = i + 1; j < hull.size(); j++) {
            uint32_t dist = hull[i].get_distance(hull[j]);
            if (dist > max_dist) {
                max_dist = dist;
                max_pair = {hull[i], hull[j]};
            }
        }
    }
    
    return max_pair;
}

template<typename T>
double calculate_area(const std::vector<T>& hull) {
    if (hull.size() < 3) return 0.0;
    
    double area = 0.0;
    size_t j = hull.size() - 1;
    
    for (size_t i = 0; i < hull.size(); i++) {
        area += (hull[j].get_x() + hull[i].get_x()) * 
                (hull[j].get_y() - hull[i].get_y());
        j = i;
    }
    
    return std::abs(area) / 2.0;
}

// 匈牙利演算法來解決最小總耗能配對問題（擴展版）
// 支援 n != m 且充電站有多個空位
class HungarianAlgorithm {
public:
    HungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix);
    double solve(std::vector<int>& assignment);
private:
    int n; // 人數
    int m; // 充電站空位總數
    std::vector<std::vector<double>> cost;
    std::vector<double> u, v;
    std::vector<int> p, way;
};


#endif // CORE_ALGORITHM_HPP

