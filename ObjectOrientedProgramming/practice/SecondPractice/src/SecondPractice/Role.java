package SecondPractice;

public class Role {
    private String Name;
    private int Life;
    private int Magic;
    private final int constMagic;

    public Role(String Name, int Life, int Magic, int constMagic) {
        this.Life = Life;
        this.Magic = Magic;
        this.Name = Name;
        this.constMagic = constMagic;
    }


    public void setName(String name) { this.Name = name; }
    public String getName() { return this.Name; }

    public void setLife(int life) { this.Life = life; }
    public int getLife() { return this.Life; }

    public void setMagic(int magic) { this.Magic = magic; }
    public int getMagic() { return this.Magic; }

    protected void attack(Role obj, int damage) {
        if (this.Magic < this.constMagic) {
            System.out.println("Magic is not enough");
        }

        this.Magic -= this.constMagic;

        System.out.printf("%s is pk with %s : damage is %d\n", this.Name, obj.getName(), damage);
        obj.setLife(obj.getLife() - damage);

        if (obj.getLife() <= 0) {
            System.out.printf("%s is dead\n", obj.getName());
            System.exit(0);
        }
    }

    @Override
    public String toString() {
        return this.Name + " " + this.Magic + " " + this.Life;
    }
}
