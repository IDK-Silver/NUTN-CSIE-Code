//
// Created by idk on 2023/11/1.
//

#include "MazeAlgorithm.h"
#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <algorithm>
#include "Maze.h"
#include <QDebug>

using namespace std;



void MazeAlgorithm::bfs_maze(const std::shared_ptr<Maze>& maze, Point startPoint, Point endPoint, vector<vector<Point>>& last_point) {

    std::deque<Point> path_queue;
    std::map<std::pair<int, int>, bool> visited;


    const std::vector<Point> move_vec = {
            {0, 1}, {0, -1}, {-1, 0}, {1, 0}
    };


    auto bsf = [&](
            Point sp, Point ep, std::vector<std::vector<MazeObject>> m,
            std::vector<std::vector<Point>> & last_point) {

        // add start point
        path_queue.push_back(sp);

        last_point.at(sp.x).at(sp.y) = sp;

        // for each point
        while (!path_queue.empty())
        {

            // get the front element
            const auto & current_point = path_queue.front();
            path_queue.pop_front();

            visited[std::pair<int, int>(current_point.x, current_point.y)] = true;

            // if not meet the end point
            if (current_point.x != ep.x || current_point.y != ep.y)
            {
                // for each vector point
                for (const auto & vec : move_vec)
                {
                    // np = current + vector
                    Point np = current_point;
                    np.x += vec.x;
                    np.y += vec.y;


                    // check the rule
                    if (np.x < 0 || np.x >= m.size() || np.y < 0 || np.y >= m.at(0).size())
                        continue;

                    if (m.at(np.x).at(np.y) == MazeObject::Wall)
                        continue;

                    if (visited.find(std::pair<int, int>(np.x, np.y)) != visited.end())
                        continue;

                    // push the queue
                    path_queue.push_back(np);
                    last_point.at(np.x).at(np.y) = current_point;
                }
            }
        }
    };

    // bfs maze
    bsf(startPoint, endPoint, maze->getMazeMatrix(), last_point);
}

std::vector<MazeAlgorithm::Point> MazeAlgorithm::getRoadHit(const shared_ptr<Maze> &maze) {
    std::vector<Point> result;

    vector<vector<Point>> last_point(maze->getSize().first, vector<Point>(maze->getSize().second));

    Point startPoint, endPoint;

    for (int row = 0; row < maze->getSize().first; row++) {
        for (int column = 0; column < maze->getSize().second; column++) {
            if (maze->at(row, column) == MazeObject::Start) {
                startPoint.x = row;
                startPoint.y = column;
            }

            if (maze->at(row, column) == MazeObject::End) {
                endPoint.x = row;
                endPoint.y = column;
            }
        }
    }

    bfs_maze(maze, startPoint, endPoint, last_point);

    mark_maze(endPoint, startPoint, last_point, maze->getMazeMatrix(), result);
    return result;
}

void MazeAlgorithm::mark_maze(Point enter_point, Point start_point, vector<vector<Point>> last_point,
                              vector<vector<MazeObject>> maze_data, std::vector<Point>& result) {

    Point f_p = last_point.at(enter_point.x).at(enter_point.y);

    if (
            f_p.x != enter_point.x || f_p.y != enter_point.y
            && !(enter_point.x == start_point.x && enter_point.y == start_point.y)
                                      )
        MazeAlgorithm::mark_maze(f_p, start_point, last_point, maze_data, result);

    result.push_back(f_p);
}

