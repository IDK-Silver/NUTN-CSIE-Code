package Project_Third;
enum SizeOfDrug {
    Large,
    Mid,
    Small
}
abstract public class Drug {
    public Drug() {
        this(SizeOfDrug.Large);
    }

    public Drug(SizeOfDrug size) {
        this.size = size;
    }

    public void setDrugSize(SizeOfDrug size) {
        this.size = size;
    }

    protected SizeOfDrug size;

    public abstract int getDrugValue();
}
