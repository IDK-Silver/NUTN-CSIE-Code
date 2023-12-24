package Project_Third;

public class RedDru extends Drug{
    public RedDru() {

    };

    @Override
    public int getDrugValue() {
        if (this.size == SizeOfDrug.Large) {
            return 120;
        } else if (this.size == SizeOfDrug.Mid) {
            return 80;
        } else if (this.size == SizeOfDrug.Small) {
            return 50;
        }
        return -1;
    };


}
