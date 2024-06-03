library ieee;
use ieee.std_logic_1164.all;

entity fa is port (
	A, B, Cin : in  std_logic;
	Sum, Carry: out std_logic
);

end fa;


architecture bhr of fa is 

signal t1 : std_logic;
signal t2 : std_logic;

begin
	t1 <= (A xor B);
	t2 <= (A and B);
	
	Sum <= t1 xor Cin;
	Carry <= (t1 and Cin) xor t2;
end bhr;