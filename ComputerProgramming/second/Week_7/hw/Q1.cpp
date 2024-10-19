#include <iostream>
#include <vector>

using namespace std;

class Matrix {
    private:
        int row_size = 0;
        int col_size = 0;
        
    public:
        vector<vector<int>> datas;
        Matrix(const int row, const int col, bool is_cin = false)
        {
            this->row_size = row;
            this->col_size = col;

            
            for (int r = 0; r < row; r++)
            {
                vector<int> col_data;
                for (int c = 0; c < col; c++)
                {
                    int input_num = 0;
                    if (is_cin)
                    {
                        cin >> input_num;
                    }
                    col_data.push_back(input_num);
                }
                this->datas.push_back(col_data);
            }
            
        };

        void fill_value(const int value)
        {
            for (auto & cols : this->datas)
            {
                for (auto & data : cols)
                {
                    data = value;
                }
            }
        }

        ~Matrix() = default;

                // Finish this part
        friend ostream& operator<<(ostream& os, Matrix& matrix) {
            for (const auto & cs : matrix.datas)
            {
                for (const auto & data : cs)
                    os << data << " ";
                os << endl;
            }
            
            return os;
        }



        // Because you haven't learn throw exception, if you want to have more challenging, you can try it yourself.
        Matrix operator+(const Matrix& other) {
            if (this->col_size != other.col_size || this->row_size != other.row_size) {
                throw runtime_error("Matrix dimensions must match for addition.\n");
            }
            Matrix result = *this;

            for (int r = 0; r < result.row_size; r++)
                for (int c = 0; c < result.col_size; c++)
                {
                    result.datas.at(r).at(c) += other.datas.at(r).at(c);
                }
            return result;
        }

        // Because you haven't learn throw exception, if you want to have more challenging, you can try it yourself.
        Matrix operator-(const Matrix& other) {
            if (this->col_size != other.col_size || this->row_size != other.row_size) {
                throw runtime_error("Matrix dimensions must match for subtraction.\n");
            }

            Matrix result = *this;

            for (int r = 0; r < result.row_size; r++)
                for (int c = 0; c < result.col_size; c++)
                {
                    result.datas.at(r).at(c) -= other.datas.at(r).at(c);
                }
            return result;
        }

        // Because you haven't learn throw exception, if you want to have more challenging, you can try it yourself.
        Matrix operator*(const Matrix& other) {
            
            if (this->col_size != other.row_size) {
                throw runtime_error("Matrix dimensions must match for multiplication.\n");
            }

            Matrix result(this->row_size, other.col_size);

            result.fill_value(0);

            for (int r = 0; r < result.row_size; r++)
            {
                for (int c = 0; c < result.col_size; c++)
                {
                    for (int k = 0; k < this->col_size; k++)
                    {
                        result.datas.at(r).at(c) += this->datas.at(r).at(k) * other.datas.at(k).at(c);
                    }
                }
            }

            return result;
        }


};


int main()
{
    int r = 0, c = 0;
    
    cin >> r >> c;
    Matrix A(r, c, true);

    cin >> r >> c;
    Matrix B(r, c, true);

    cout << "Matrix A\n" << A << "\n";
    cout << "Matrix B\n" << B << "\n";

    // Finish this part
    // If you want to use my hint please just copy the following part.
    try {
	Matrix C = A + B;
	cout << "Add" << endl;
	cout << C << endl;
    } catch (const runtime_error& e) {
	cout << e.what();
    }
    try {
        Matrix D = A - B;
        cout << "Sub" << endl;
        cout << D << endl;
    } catch (const runtime_error& e) {
        cout << e.what();
    }
    try {
        Matrix E = A * B;
        cout << "Mul" << endl;
        cout << E << endl;
    } catch (const runtime_error& e) {
	cout << e.what();
    }
    return 0;
}