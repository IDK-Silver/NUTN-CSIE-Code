class B
{
public:
  B();
  B(int nn);
  void f();
  void g();
private:
  int n;
};
class D: public B
{
public:
  D(int nn, float dd);
  void h();
private:
  double d;
};


int main()
{
	D d(1, 2);
  
}
