#include <iostream>
#include <string.h>



using namespace std;
struct info{
	string name;
	int money;
};

int main(){
	
	int num;
	string who,command,who_1;
	int mny;
	cin>>num;
	info bankinfo[num];
	
	for (int i=0;i<num;i++){
		cin >> bankinfo[i].name >> bankinfo[i].money;
		
		
	}
	while(cin >>who>> command>>mny){
		for(int i=0;i<=num;i++){
			if(who==bankinfo[i].name){
				if(command=="deposit"){
					bankinfo[i].money=bankinfo[i].money+mny;
					cout << bankinfo[i].name<< " deposit " << mny<<"."<<endl;
				}else if (command=="withdraw"){
					bankinfo[i].money=bankinfo[i].money-mny;
					if(bankinfo[i].money >= 0){
						cout << bankinfo[i].name<< " withdraw "<< mny <<" from his/her account.\n";
						
					}else{
						bankinfo[i].money=bankinfo[i].money+mny;
						cout << bankinfo[i].name<<"'s account balance is insufficient.\n";
					}
				}else if(command=="transfer"){
					cin >> who_1;
					
					if(who==who_1){
						bankinfo[i].money=bankinfo[i].money;
						cout << "You transfer to yourself.\n";
						
					}else{
						bankinfo[i].money=bankinfo[i].money-mny;
						if(bankinfo[i].money>=0){
							cout << "Transfer successful.\n";
						}else{
							bankinfo[i].money=bankinfo[i].money+mny;
							cout<<"Transfer failed.\n";
						}
					}
					
				}
			}
		}
	}
	
	cout << "\nName\tBalance" << endl;
	for(int i=0;i<num;i++){
		cout << bankinfo[i].name <<"\t"<<bankinfo[i].money<<endl;
	}
	
	return 0;
	
}


