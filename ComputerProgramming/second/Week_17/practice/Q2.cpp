class IntPair
{
  int first;
  int second;
public:
  IntPair(int firstValue, int secondValue);
  const IntPair operator++( ); //Prefix version
  const IntPair operator++(int); //Postfix version
  int getFirst( ) const;
  int getSecond( ) const;
};


int main() {

  IntPair a(1, 2);
  (a++);

}
