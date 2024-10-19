#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <algorithm>

using namespace std;

struct Point {
    int x = 0;
    int y = 0;
};

deque<Point> path_queue;

map<pair<int, int>, bool> visited;

const vector<Point> move_vec = {
        {0, 1}, {0, -1}, {-1, 0}, {1, 0}
};

void bfs_maze(Point sp, Point ep, vector<vector<char>> m, vector<vector<Point>> & last_point)
{
    // add start point
    path_queue.push_back(sp);

    last_point.at(sp.x).at(sp.y) = sp;

    // for each point
    while (!path_queue.empty())
    {

        // get the front element
        const auto & current_point = path_queue.front();
        path_queue.pop_front();

        visited[pair<int, int>(current_point.x, current_point.y)] = true;

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

                if (m.at(np.x).at(np.y) == '1')
                    continue;

                if (visited.find(pair<int, int>(np.x, np.y)) != visited.end())
                    continue;

                // push the queue
                path_queue.push_back(np);
                last_point.at(np.x).at(np.y) = current_point;
            }
        }
    }


}

void mark_maze (Point enter_point, Point start_point, vector<vector<Point>> last_point, vector<vector<char>> & maze_data)
{
    Point f_p = last_point.at(enter_point.x).at(enter_point.y);
    if (
            f_p.x != enter_point.x || f_p.y != enter_point.y
            && !(enter_point.x == start_point.x && enter_point.y == start_point.y)
    )
        mark_maze(f_p, start_point, last_point, maze_data);

    maze_data.at(f_p.x).at(f_p.y) = '*';
//    cout << f_p.x << ", " << f_p.y << "\n";
};

int main()
{
    int w = 0, h = 0;
    cin >> w >> h;

    Point start_point, end_point;

    vector<vector<char>> maze_data;
    for (int x = 0; x < w; x++)
    {
        vector<char> line;
        for (int y = 0; y < h; y++)
        {
            char input_num = 0;
            cin >> input_num;

            if (input_num == 'S')
            {
                start_point = {x, y};
            }
            else if (input_num == 'E')
            {
                end_point = { x, y};
            }

            line.push_back(input_num);
        }
        maze_data.push_back(line);
    }

    vector<vector<Point>> last_point(w, vector<Point>(h));

    bfs_maze(start_point, end_point, maze_data, last_point);








    mark_maze(end_point, start_point, last_point, maze_data);
    maze_data.at(start_point.x).at(start_point.y) = 'S';
    maze_data.at(end_point.x).at(end_point.y) = 'E';

    for (const auto & x_data : maze_data)
    {
        for (const auto & data : x_data)
        {
            cout << data;
        }
        cout << "\n";
    }

    long step_num = 1;
    for (auto const & vec : maze_data)
    {
        step_num += count(vec.begin(), vec.end(), '*');
    }


    cout << "Total " << step_num << " steps.";

    // find the start point




}
