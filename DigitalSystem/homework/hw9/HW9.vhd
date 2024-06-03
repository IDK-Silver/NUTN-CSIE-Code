library ieee;
use ieee.std_logic_1164.all;


entity HW9 is port (
	hin : in  std_logic_vector(4 downto 0);
	hout: out std_logic
);
end HW9;

architecture bhr of HW9 is 

	function FN1 (
		F1, F2, F3, F4 : std_logic
	) return std_logic is variable result : std_logic;
	begin

		result := std_logic(
			(F1 xor F2) xor (F3 xor F4)
		);
		return result;
	end function FN1;


begin

	hout <= (
		Fn1(hin(4), hin(3), hin(2), hin(1))
		xor
		hin(0)
	);


end bhr;
