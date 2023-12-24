package SecondPractice;
import SecondPractice.Role;
import SecondPractice.Warrior;

public class Witcher extends Role {

	public Witcher() {
		this("undefine");
	}

	public Witcher(String name) {
		this(name, 400, 100);
	};

	public Witcher(String Name, int Life, int Magic) {
		super(Name, Life, Magic, 10);
	}

	
	public void SmallFire(Role obj) {
		if (obj instanceof  Witcher) {
			this.attack(obj, 40);
		}
		else {
			this.attack(obj, 60);
		}
	};


}