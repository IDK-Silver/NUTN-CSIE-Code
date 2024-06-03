library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity FA_BHR_3Bit is port (
	A, B  : in  std_logic_vector(2 downto 0);
	Cin   : in  std_logic;
	Sum   : out std_logic_vector(2 downto 0);
	Carry : out std_logic
);
end FA_BHR_3Bit;

architecture bhr of FA_BHR_3Bit is

signal ca, cb, cc, r : unsigned (3 downto 0);

begin
	ca(2 downto 0) <= unsigned(A);
	cb(2 downto 0) <= unsigned(B);
	cc(0) 		   <= Cin;
	
	r <= ca +cb +cc;
	
	Sum <= std_logic_vector(r(2 downto 0));
	Carry <= r(3);
	
end bhr;
