library ieee;
use ieee.std_logic_1164.all;

entity fa_3b is port (
	A, B   : in  std_logic_vector(2 downto 0);
	Cin    : in  std_logic;
	Sum    : out std_logic_vector(2 downto 0);
	Carry  : out std_logic
);
end fa_3b;

architecture bhr of fa_3b is

component fa port (
	A, B, Cin : in  std_logic;
	Sum, Carry: out std_logic
); end component;

signal ct : std_logic_vector(2 downto 0);

begin

	L1 : fa port map(A(0), B(0), Cin, Sum(0), ct(0));
	L2 : fa port map(A(1), B(1), ct(0), Sum(1), ct(1));
	L3 : fa port map(A(2), B(2), ct(1), Sum(2), Carry);

end bhr;

