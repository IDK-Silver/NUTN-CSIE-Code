#include <fstream>
using std::ofstream;
using std::ios;

int main()
{
	ofstream outStream;
	outStream.open("myFile.txt", ios::app);
	
	return 0;
}
