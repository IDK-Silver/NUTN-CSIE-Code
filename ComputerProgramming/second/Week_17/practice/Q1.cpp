class Money
{
public:
  Money( );
  Money(int dollars, int cents); 
  Money(int dollars);
  Money(double amount);
// other public members
  const Money operator+(const MoneyADD& amt2)ADD;
>>Need to add & and semicolon
  int getCents( ) const;
  int getDollars( ) const;
private:
  int dollars;
  int cents;
  //other private members
};
/*
	Note that * is not overloaded in the class, but operator + is overloaded using an operator function with the following declaration:
	const Money operator+(const Money& amt2); 
	The question is: Given the declarations,
	Money baseAmount(100, 60); // $100.60
	Money fullAmount; 
*/
