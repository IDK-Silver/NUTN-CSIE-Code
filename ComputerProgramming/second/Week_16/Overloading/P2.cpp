class Money {
public:
  Money( );
  Money(int dollars, int cents);
  Money(int dollars);
  Money(double amount);
  
  // other public members
  const Money operator+(const Money& amt2);
  int getCents( ) const;
  int getDollars( ) const;
private:
  int dollars;
  int cents;
  //other private members
};

int main()
{

  return 0;
}
