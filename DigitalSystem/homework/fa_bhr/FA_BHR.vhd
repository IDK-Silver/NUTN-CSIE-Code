library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity FA_BHR is port (
	A, B, Cin : in std_logic;
	Sum, Carry : out std_logic
);
end FA_BHR;

architecture bhr of FA_BHR is

signal ca, cb, cc, r : unsigned (1 downto 0);

begin
	ca(0) <= A;
	cb(0) <= B;
	cc(0) <= Cin;
	
	r <= ca +cb +cc;
	
	Sum <= r(0);
	Carry <= r(1);
	
end bhr;
