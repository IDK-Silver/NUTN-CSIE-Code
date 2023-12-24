package Project_Third;
import java.util.Random;
import Project_Third.RedDru;
import Project_Third.BlueDrug;
public class Main {

    public static void main(String[] args) {

        Role[] role_arr = {
                new Warrior("A"),
                new Warrior("B"),
                new Warrior("C"),
                new Witcher("D"),
                new Witcher("E"),
                new Witcher("F")
        };


        while (true) {
            int a = 0, b = 0;

            Random r = new Random();

            a = r.nextInt(5+ 1); // 0 ~ 5
            b = r.nextInt(5+ 1);

//            System.out.printf("%d %d\n", a, b);

            if (a == b)
                continue;

            Role r1 = role_arr[a];
            Role r2 = role_arr[b];

            if (r1 instanceof Warrior) {
                ((Warrior) r1).NewMoon(r2);
                ((Warrior) r1).recoverLife();
            }
            else if (r1 instanceof Witcher){
                ((Witcher) r1).SmallFire(r2);
                ((Witcher) r1).recoverMagic();
            }
            else {
                System.out.println("Error");
            }
//            isNeedDrug(r1);
            isNeedDrug(r2);
        }

    }


    public static void isNeedDrug(Role obj) {
        Random r = new Random();
        int drug_rng = r.nextInt(9+ 1);

        SizeOfDrug size;

        if (drug_rng == 0) {
            size = SizeOfDrug.Large;
        }
        else if (drug_rng <= 3) {
            size = SizeOfDrug.Mid;
        }
        else if (drug_rng <= 7) {
            size = SizeOfDrug.Small;
        }
        else {
            return;
        }

        if (obj.getLife() < 60) {
            Drug d = new RedDru();
            d.setDrugSize(size);
            obj.Drink(d);
        }
        else if (obj.getMagic() < 30) {
            Drug d = new BlueDrug();
            d.setDrugSize(size);
            obj.Drink(d);
        }
    }

}

