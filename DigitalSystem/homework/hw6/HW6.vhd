library ieee;
use ieee.std_logic_1164.all;

entity HW6 is port (
	A, B : in  std_logic_vector(2 downto 0);
	R_S  : out std_logic_vector(5 downto 0)
);
end HW6;

architecture bhr of HW6 is 


component fa_3b port (
	A, B   : in  std_logic_vector(2 downto 0);
	Cin    : in  std_logic;
	Sum    : out std_logic_vector(2 downto 0);
	Carry  : out std_logic
); end component;

signal t1 : std_logic_vector(2 downto 0);
signal t2 : std_logic_vector(2 downto 0);

begin
	
	
	
end bhr;