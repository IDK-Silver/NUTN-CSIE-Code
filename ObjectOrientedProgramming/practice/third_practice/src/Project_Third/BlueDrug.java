package Project_Third;

import Project_Third.Drug;

public class BlueDrug extends Drug {

    BlueDrug() {

    }
    @Override
    public int getDrugValue() {
        if (this.size == SizeOfDrug.Large) {
            return 100;
        } else if (this.size == SizeOfDrug.Mid) {
            return 60;
        } else if (this.size == SizeOfDrug.Small) {
            return 30;
        }
        return -1;
    };

}
