package first_practice;



public class Witcher {
	
	private String Name;
	private int Life;
	private int Magic;
	
	final int constMagic;
	
	public Witcher() {
		this("undefine");
	}
	
	public Witcher(String name) {
		this(name, 400, 100);
	};
	
	public Witcher(String Name, int Life, int Magic) {
		this.Life = Life;
		this.Magic = Magic;
		this.Name = Name;
		this.constMagic = 10;
	}
	
	public void setName(String name) { this.Name = name; }
	public String getName() { return this.Name; }
	
	public void setLife(int life) { this.Life = life; }
	public int getLife() { return this.Life; }
	
	public void setMagic(int magic) { this.Magic = magic; }
	public int getMagic() { return this.Magic; }
	
	public void SmallFire(Warrior obj) {
		
		if (this.Magic < this.constMagic) {
			System.out.println("Magic is not enough");
		}
		
		this.Magic = this.constMagic;
		
		System.out.printf("%s is pk with %s\n", this.Name, obj.getName());
		obj.setLife(obj.getLife() - 40);
		
		if (obj.getLife() <= 0) {
			System.out.printf("%s is dead\n", obj.getName());
			System.exit(0);
		}
		
	};
	
	public void SmallFire(Witcher obj) {
		
		if (this.Magic < this.constMagic) {
			System.out.println("Magic is not enough");
		}
		
		this.Magic = this.constMagic;
		
		System.out.printf("%s is pk with %s\n", this.Name, obj.getName());
		obj.setLife(obj.getLife() - 60);
		
		if (obj.getLife() <= 0) {
			System.out.printf("%s is dead\n", obj.Name);
			System.exit(0);
		}
		
	};
}