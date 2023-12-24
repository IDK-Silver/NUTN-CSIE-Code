package Project_Third;
import Project_Third.Role;
import Project_Third.Witcher;
public class Warrior extends Role implements LifeRecoverable {

	public Warrior() {
		this("undefine");
	}

	public Warrior(String name) {
		this(name, 400, 100);
	};

	public Warrior(String Name, int Life, int Magic) {
		super(Name, Life, Magic, 10);
	}

	public void NewMoon(Role obj) {
		if (obj instanceof Warrior) {
			this.attack(obj, 25);
		}
		else {
			this.attack(obj, 40);
		}

	}

	@Override
	public double recoverLife() {
		final int recoverTotal = this.getLife() + LIFERATE;
		this.setLife(recoverTotal);
		return recoverTotal;
	}
}
