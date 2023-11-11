//
// Created by idk on 2023/10/28.
//

#include "Maze.h"


std::vector<MazeObject> Maze::at(size_t row) {
    return this->map->at(row);
}

MazeObject& Maze::at(size_t row, size_t column) {
    return this->map->at(row).at(column);
}

void Maze::setMazeObject(size_t row, size_t column, const MazeObject &obj){
    this->map->at(row).at(column) = obj;
}

Maze::Maze(size_t M, size_t N) : map_size_M(M), map_size_N(N) {

    /* create maze map obj */
    this->map = std::make_shared<std::vector<std::vector<MazeObject>>>();

    /* create each maze obj to maze map */
    for (size_t row = 0; row < M; row++) {

        std::vector<MazeObject> row_vec;

        for (size_t column = 0; column < N; column++) {
            row_vec.push_back(MazeObject::Wall);
        }

        this->map->push_back(row_vec);
    }
}

std::pair<int, int> Maze::getSize() {
    return std::pair<int, int>(this->map_size_M, this->map_size_N);
}

std::ostream &operator<<(std::ostream &os, const Maze &obj) {

    os << "Maze Size : " << obj.map_size_M << ", " << obj.map_size_N;

    for (size_t row = 0; row < obj.map_size_M; row++) {

        for (size_t column = 0; column < obj.map_size_N; column++) {
            os << obj.map->at(row).at(column) << " ";
        }
        os << "\n";
    }

    return os;
}

std::vector<std::vector<MazeObject>> Maze::getMazeMatrix() {
    return *this->map;
}

