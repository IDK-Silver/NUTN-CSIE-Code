#include <iostream>
using namespace std;

int main(){
	
	int w[2],number[2]={1,2},h[2],area[2];
	int remain,stuff;
	for(int i=0;i<2;i++){
		cin >> w[i] >> h[i];
		area[i]=w[i]*h[i];
		cout << "The area of Rectangle "<<number[i] << " is " << area[i]<<"."<< endl;
	}
	
	if((w[0] >= w[1] && h[0] >= h[1]) || (w[0] >= h[1] && h[0] >= w[1]))
	{
		stuff=area[0]/area[1];
		remain=area[0]%area[1];
		
		if((remain == 0 )&& (stuff != 1)){
			
			cout << "There are " << stuff <<" Rectangle 2 stuffed into Rectangle 1."<< endl;
			cout << "Rectangle 2 is smaller than Rectangle 1."<< endl;
		}else{
			if((stuff==1) && (remain == 0)){
				cout << "There are " << stuff <<" Rectangle 2 stuffed into Rectangle 1."<< endl;
				cout << "There are " << stuff <<" Rectangle 1 stuffed into Rectangle 2."<< endl;
			}
			else{
				cout << "There are " << stuff<<" Rectangle 2 stuffed into Rectangle 1, and remain "<<remain<<" areas."<<endl;
				cout << "Rectangle 2 is smaller than Rectangle 1."<< endl;
				
			}
			
			
		}
		}
	else 
	{
		cout << "Rectangle 1 is smaller than Rectangle 2."<< endl; 
		cout << "Rectangle 2 is smaller than Rectangle 1."<< endl;
		

		
	}
	return 0;
}

