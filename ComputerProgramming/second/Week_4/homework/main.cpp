#include <iostream>
using namespace std;

class Rectangle {
    private:
        int width, height, index;
    public:
        Rectangle(int i, int w, int h) {
            width = w;
            height = h;
	    index = i;
        }
        int area() {
            return this->width * this->height;
        }
		
	string name() {
	    return "Rectangle " + std::to_string(this->index);
	}
		
	bool suit(Rectangle obj) {
	    // Finish this part
	}
		
	int blocks(Rectangle obj) {
	    // Finish this part
	}
		
    int remain(Rectangle obj) {
	    // Finish this part
	}
		
	void insert(Rectangle obj) {
        
        int fw = obj.width, fh = obj.height;
        {
            double d = (this->width / obj.width) * (this->height / obj.height);
            double r = (this->width / obj.height) * (this->height / obj.width);

            if (d < r)
            {
                fw = obj.height;
                fh = obj.width;
            }
        }

        int tb = int(height / fh) * int(width / fw);
        int la = this->area() - (tb * obj.area());


        if (tb > 0)
        {
            if (la > 0)
            {
                cout << "There are " << tb << " Rectangle " << obj.index << " stuffed into Rectangle " << index << ", and remain " \
                                << la << " areas.\n";
            }
            else
            {
                cout << "There are " << tb << " Rectangle " << obj.index << " stuffed into Rectangle " << index << ".\n";
            }
        }
        else
        {
            cout << "Rectangle " << this->index << " is smaller than Rectangle " << obj.index << ".\n";
        }
    }
};

int main() {
	int w, h;
	cin >> w >> h;
        Rectangle rect1(1, w, h);
	cin >> w >> h;
        Rectangle rect2(2, w, h);

        cout << "The area of " << rect1.name() << " is " << rect1.area() << "." << endl;
        cout << "The area of " << rect2.name() << " is " << rect2.area() << "." << endl;
	rect1.insert(rect2);
	rect2.insert(rect1);



        return 0;
}