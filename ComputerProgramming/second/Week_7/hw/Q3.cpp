#include <iostream>
#include <map>
#include <vector>

using namespace std;

vector<vector<int>> ans;

void q_solution(int row, int n, vector<int> chess_point)
{
    if (row == n)
    {
        ans.push_back(chess_point);
        return;
    }

    auto is_valid = [](int row_lim, int try_col, vector<int> chess_point) {

        for (int row = 0; row < row_lim; row++)
        {
            if (chess_point.at(row) == try_col || (abs(row_lim - row) == abs(chess_point.at(row) - try_col)))
                return false;
        }

        return true;
    };

    for (int try_col = 0; try_col < n; try_col++)
    {
        if (is_valid(row, try_col, chess_point))
        {
            chess_point.at(row) = try_col;
            q_solution(row + 1, n, chess_point);
        }
    }

}

int main()
{
    int size = 0;
    cin >> size;

    // x = index, y = arr[x]
    vector<int> chess_point(size);


    q_solution(0, size, chess_point);


    map<string , int> ans_map;

    int index = 0;
    for (const auto & point : ans)
    {
        string key;
        for (const auto & col : point)
        {
            key += (char) col + 1;
        }

        if (ans_map.find(key) == ans_map.end())
            ans_map[key] = index++;
        else
        {
            cout << key;
            exit(0);
        }
    }

    for (const auto & pair : ans_map)
    {
        for (const auto & point : ans.at(pair.second))
        {
            cout << point + 1 << " ";
        }
        cout << "\n";
    }
    cout << "A total of " << ans.size() << " solutions for " << size << "-Queens problem.";
    return 0;
}