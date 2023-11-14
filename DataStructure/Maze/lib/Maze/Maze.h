//
// Created by idk on 2023/10/28.
//

#ifndef MAZE_MAZE_H
#define MAZE_MAZE_H

#include <vector>
#include <memory>
#include <ostream>

/* the object of maze map */
enum MazeObject {
    Wall,
    Blank,
    Player,
    RoadHint,
    Start,
    End
};


/* the maze */
class Maze {
public:
    Maze(size_t M, size_t N);
    ~Maze() = default;

    /* get map object method */
    std::vector<MazeObject> at(size_t row);
    MazeObject& at(size_t row, size_t column);

    /* get map object method */
    void setMazeObject(size_t row, size_t column, const MazeObject & obj);
    std::pair<int, int> getSize();

    // override operator
    friend std::ostream& operator<<(std::ostream& os, const Maze& obj);

    std::vector<std::vector<MazeObject>> getMazeMatrix();
    std::shared_ptr<std::vector<std::vector<MazeObject>>> getMazeMatrixByRef();

private:
    const size_t map_size_M;
    const size_t map_size_N;

    /* the maze map */
    std::shared_ptr<std::vector<std::vector<MazeObject>>> map;



};



#endif //MAZE_MAZE_H
