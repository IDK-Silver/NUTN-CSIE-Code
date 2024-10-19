#include <iostream>

class Obj {
public:
  Obj(int value) : value(value) {};
  Obj() = default;
  ~Obj() = default;
private:
  int value = 0;
};

Obj create_obj(int value) {
  return Obj(value);
};

int main()
{
  auto obj = create_obj(10);
  return 0;
}
