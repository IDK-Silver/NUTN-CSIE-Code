#include <iostream>

using namespace std;

void showVolume(int length, int width = 1, int height = 1);

int main()
{
    showVolume(2, 2, 2);
     showVolume(2, 2);
     showVolume(1);

    return 0;
}

void showVolume(int length, int width, int height)
{
    cout << "Volume of a box with \n"
    << length << " " << width << " " << height << "\n";
}