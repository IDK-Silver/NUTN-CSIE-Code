//
// Created by idk on 2023/11/1.
//

#ifndef MAZE_MAZE_ALGORITHM_H
#define MAZE_MAZE_ALGORITHM_H

#include "Maze.h"
#include <memory>

using  namespace  std;

namespace MazeAlgorithm {
    struct Point {
        int x = 0;
        int y = 0;
    };

    std::vector<MazeAlgorithm::Point> getRoadHit(const std::shared_ptr<Maze> &maze);
    void bfs_maze(const std::shared_ptr<Maze> &maze, Point startPoint, Point endPoint, vector<vector<Point>>& last_point);
    void
    mark_maze(Point enter_point, Point start_point, vector<vector<Point>> last_point,
              vector<vector<MazeObject>> maze_data, std::vector<Point>& result);
}



#endif //MAZE_MAZE_ALGORITHM_H
