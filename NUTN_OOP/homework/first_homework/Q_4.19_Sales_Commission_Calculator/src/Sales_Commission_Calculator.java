import java.util.Hashtable;
import java.util.Scanner;

public class Sales_Commission_Calculator {
    public static void main(String[] args) {

        // the ans
        double totalGetPay = 200;

        // init sale table
        Hashtable<Integer, Double> saleTable;
        saleTable = new Hashtable<>();
        saleTable.put(1, 239.99);
        saleTable.put(2, 129.75);
        saleTable.put(3, 99.95);
        saleTable.put(4, 350.89);
//        saleTable.put(4, 5000.0);

        // the input formant and told user how to exit
        System.out.print(
                        "Input your want to cal : \n " +
                        "ex : <item> <num> -> 1 5\n" +
                        "if your want to exit enter -1 to exit\n"
                );

        // get the System in
        Scanner scanner = new Scanner(System.in);

        // the sale item and num
        int item, num;

        // cal core
        while (true) {

            // get input
            item = scanner.nextInt();

            // check is exit command
            if (item == -1) {
                System.out.println("Exit");
                break;
            }

            num = scanner.nextInt();

            // check input item is in saleTable
            if (!saleTable.containsKey(item)) {
                System.out.print("Item is not in table\n");
                continue;
            }


            totalGetPay += saleTable.get(item) * ((double) 9 / 100) * num;

        }

        System.out.printf("Total pay is : %f", totalGetPay);


    }


}