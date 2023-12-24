package Project_Third;
import Project_Third.Role;
import Project_Third.Warrior;

public class Witcher extends Role implements MagicRecoverable{

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

	@Override
	public double recoverMagic() {
		final int recoverTotal = this.getLife() + MAGICRATE;
		this.setMagic(recoverTotal);
		return recoverTotal;
	}

}



