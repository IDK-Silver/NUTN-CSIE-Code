library ieee;
use ieee.std_logic_1164.all;

entity HW7 is port (
	hin, clk : in  std_logic;
	Q        : out std_logic_vector(3 downto 0)
); end HW7;

architecture bhr of HW7 is

signal qt : std_logic_vector(3 downto 0);

component dff port (
	d, clk : in  std_logic;
	q      : out std_logic
); end component;

begin

F: dff port map (
			hin, clk, qt(0)
	);

GF : for i in 0 to 2 generate
		L2: dff port map (
			qt(i), clk, qt(i + 1)
		);
	 end generate GF ; 
	 
	 Q <= qt;


end bhr;