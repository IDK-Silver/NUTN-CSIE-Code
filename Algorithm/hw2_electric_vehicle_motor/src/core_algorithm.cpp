#include "core_algorithm.hpp"



HungarianAlgorithm::HungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix)
    : cost(cost_matrix) {
    n = cost.size();           // 人數
    if (n == 0) return;
    m = cost[0].size();        // 充電站空位總數
    // 初始化拉格朗日乘數
    u.assign(n + 1, 0.0);
    v.assign(m + 1, 0.0);
    p.assign(m + 1, 0);
    way.assign(m + 1, 0);
}

double HungarianAlgorithm::solve(std::vector<int>& assignment) {
    int inf = std::numeric_limits<int>::max();
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(m + 1, inf);
        std::vector<bool> used(m + 1, false);
        int i0 = i;
        int j1;
        double delta;
        do {
            used[j0] = true;
            int i1 = p[j0];
            delta = inf;
            int j2 = 0;
            for (int j = 1; j <= m; ++j) {
                if (!used[j]) {
                    double cur = cost[i1 - 1][j - 1] - u[i1] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j2 = j;
                    }
                }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else {
                    minv[j] -= delta;
                }
            }
            j0 = j2;
        } while (p[j0] != 0);
        
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    
    assignment.assign(n, -1);
    double total_cost = 0.0;
    for (int j = 1; j <= m; ++j) {
        if (p[j] > 0 && p[j] <= n && j <= m) {
            assignment[p[j] - 1] = j - 1;
            total_cost += cost[p[j] - 1][j - 1];
        }
    }
    return total_cost;
}