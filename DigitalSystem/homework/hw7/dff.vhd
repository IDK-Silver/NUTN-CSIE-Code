library ieee;
use ieee.std_logic_1164.all;

entity dff is port (
	d, clk : in  std_logic;
	q      : out std_logic
); end dff;


architecture bhr of dff is


begin

process (clk) begin

if (rising_edge(clk)) then
	
	q <= d;

end if;


end process;


end bhr;