#include <iostream>
#include <cmath>
a
using namespace std;

const double PI = 3.1459;

double area(double radius);
double volume(double radius);

int main()
{
    double radius_both = 0, area_circle = 0, volume_sphere = 0;

    cout << "Enter a radius to use for both a circle\n"
        << "and a sphere (in inches) : ";
    cin >> radius_both;

    area_circle = area(radius_both); 
    volume_sphere = volume(radius_both);

    cout << "Radius = " << radius_both << " inches\n"
        << " Area of circle = " << area_circle
        << " square inches\n"
        << "Volume of sphere = " << volume_sphere
        << " cubic inches\n";

    return 0;
}

double area(double radius)
{
    return (PI * pow(radius, 2));
}

double volume(double radius)
{
    return ((4.0 / 3.0) * PI * pow(radius, 3));
}
