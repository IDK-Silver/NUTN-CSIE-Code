#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <cmath>
#include <stdio.h>

using namespace std;

struct Person {
    string name;
    int age = 0;
    float height = 0;
    float weight = 0;
    float BMI = 0;
};

shared_ptr<Person> createPerson()
{
    auto data = make_shared<Person>();
    cin >> data->name;
    cin >> data->age;
    cin >> data->height;
    cin >> data->weight;
    data->BMI = data->weight / pow(data->height / 100.0, 2);
    return data;
};

void printPerson(shared_ptr<Person> data)
{
    cout << data->name << "\t" << data->age << "\t" << data->height << "\t" << data->weight << "\t";
    printf("%.1f\t", data->BMI);
    string BMI_status;
    if (data->BMI < 18.5)
    {
        BMI_status = "Too Thin";
    } 
    else if (data->BMI < 24)
    {
        BMI_status = "Normal";
    } 
    else if (data->BMI < 27)
    {
        BMI_status = "Too Heavy";
    }
    else if (data->BMI < 30)
    {
        BMI_status = "Too Too Heavy";
    }
    else if (data->BMI < 35)
    {
        BMI_status = "Too Too Too Heavy";
    }
    else
    {
        BMI_status = "Too Too Too Too Heavy";
    }
    cout << BMI_status << "\n";
};

int main()
{
    int size = 0;
    vector<shared_ptr<Person>> datas;
    cin >> size;

    for (int i = 0; i < size; i++)
    {
        datas.push_back(createPerson());
    }

    cout << "Name\tAge\tHeight\tWeight\tBMI\tStatus\n";

    for (const auto data : datas)
    {
        printPerson(data);
    }

    return 0;
}

/*
Name	Age	Height	Weight	BMI	Status
Niny	27	176	54	17.4	Too Thin
Nick	21	181	65	19.8	Normal
Judy	21	169	58	20.3	Normal
*/
