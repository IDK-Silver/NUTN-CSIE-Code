#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int binary_search(const vector<int> & arr, int target)
{
    int r = 0, q = arr.size() - 1;

    while (r <= q)
    {
        int m = r + (q - r) / 2;

        if (arr.at(m) == target)
            return m;

        else if (arr.at(m) < target)
            r = m + 1;

        else
            q = m - 1;
    }

    return -1;
}

int partition(int p, int r, vector<int> & arr)
{
    int last = arr.at(r);
    int i = p - 1;

    for (int j = p; j < r - 1; j++)
    {
        if (arr.at(j) <= last)
        {
            i++;
            swap(arr.at(j), arr.at(i));
        }
    }
    swap(arr.at(i + 1), arr.at(last));
    return  i + 1;

}

void quick_sort(int p, int r, vector<int> & arr)
{
    if (p >= r)
        return;

    int q = partition(p, r, arr);
    quick_sort(p, q - 1, arr);
    quick_sort(q + 1, r, arr);
}

int main()
{
    vector<int> vec = {1, 2, 5, 7, 8, 10, 2};
    quick_sort(0, 6, vec);


    for (const auto & n : vec)
    {
        cout << n << " ";
    }

    cout << "\n" << binary_search(vec, 8) << "\n";
    return 0;
}






