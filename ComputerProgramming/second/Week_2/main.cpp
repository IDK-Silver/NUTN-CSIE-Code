#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdio.h>
#include <math.h>

int main()
{
    const int subject_size = 6;
    std::vector<std::string> subject_name = {
        "DM", "DS", "LA", "ALGO", "CO", "OS"
    };

    int data_size = 0;
    std::cin >> data_size;
    
    std::vector<std::vector<double>> grades;

    for (int index = 0; index < data_size; index++)
    {
        std::vector<double> grade;

        for (int j = 0; j < subject_size; j++)
        {
            double number = 0;

            std::cin >> number;
            grade.push_back(number);
        }
        grades.push_back(grade);
    }

    // print the title
    std::cout << "No.\t";
    for (int i = 0; i < subject_size; i++)
    {
        std::cout << subject_name.at(i) << "\t";
    }
    std::cout << "Avg.\tVar.\n";
    

    for (int i = 0; i < data_size; i++)
    {
        std::cout << i + 1 << "\t";
        double avg = 0;
        double var = 0;
        auto grade = grades.at(i);
        for (const auto & num : grade)
        {
            std::cout << num << "\t";
            avg += num;
        }

        avg /= subject_size;
        
        printf("%.1f\t", avg);
        // std::cout << std::setprecision(1) << avg << "\t";

        for (const auto & num : grade)
        {
            var += pow(num - avg, 2) / subject_size;
        }
        printf("%.1f\n", var);
        // std::cout << std::setprecision(1) << var << "\n";
    }
}
