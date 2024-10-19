#include <iostream>
#include <string>
#include <cstdio>
#include <iomanip>
#include <cmath>

using namespace std;

class Rectangle {
    protected:
        double width = 0;
        double height = 0;
        double high = 0;
        double coefficient = 1;
        string shap_str = "";

    public:

        Rectangle(double width, double height, double high)
        {
            this->width = width;
            this->height = height;
            this->high = high;
            this->coefficient = 1;
            shap_str =  "The Rectangle's width is " + to_string((int) this->width) + ", height is " + to_string((int) this->height) + ", high is "+ to_string((int) this->high) + ".\n";
        }

        double get_area() {
            return this->width * this->height * this->coefficient;
        };

        double get_volume() {
            return this->get_area() * this->high;
        };

        friend ostream& operator<<(ostream & stream, Rectangle & t) {
            stream << t.shap_str << "Area = " << t.get_area() << "\nVolume = "<< t.get_volume() << "\n";
            return stream;
        };
};

class Triangle : public Rectangle {
    public:
        Triangle(double width, double height, double high) : Rectangle(width, height, high) {
            this->shap_str =  "The Triangle's bottom is " + to_string((int) this->width) + ", width is " + to_string((int) this->height) + ", high is "+ to_string((int) this->high) + ".\n";
            this->coefficient = 0.5;
        }
};

class Circle : public Rectangle {
    public:
        Circle(double width, double high) : Rectangle(width, width, high) {
            this->shap_str =  "The Circle's radius is " + to_string((int) this->width) + ", high is "+ to_string((int) this->high) + ".\n";
            this->coefficient = M_PI;
        };
};

int main()
{
    string in;
    cin >> in;

    if (in == "rectangle")
    {
        double width = 0;
        double height = 0;
        double high = 0;
        cin >> width >> height >> high;
        Rectangle shape(width, height, high);
        cout << shape;
    }
    else if (in == "triangle")
    {
        double width = 0;
        double height = 0;
        double high = 0;
        cin >> width >> height >> high;
        Triangle shape(width, height, high);
        cout << shape;
    }
    else 
    {
        double width = 0;
        double high = 0;
        cin >> width >> high;
        Circle shape(width, high);
        cout << shape;
    }
    return 0;
}