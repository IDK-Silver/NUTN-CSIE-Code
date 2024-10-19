#ifndef DIGITAL_H
#define DIGITAL_H

#include <iostream>
using namespace std;
class DigitalTime
{
private:
    int hour = 0;
    int minute = 0;

    friend bool operator== (const DigitalTime & obj_1,const DigitalTime & obj_2);
    friend istream & operator>> (istream& ins, DigitalTime& theObject);
    friend ostream & operator<< (ostream& outs, const DigitalTime& theObject);

public:
    
    DigitalTime(int hour, int min);
    DigitalTime();
    ~DigitalTime() = default;

    int get_hour();
    int get_minute();
    void advance(int hour, int min);
    void advance(int min);

    static void read_minute(int& theHour);
    static void read_hour(int& theMinut);
    static int digital_to_int(char );
};

#endif 