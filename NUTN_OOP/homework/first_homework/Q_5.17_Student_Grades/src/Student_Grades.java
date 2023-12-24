import java.util.Scanner;


public class Student_Grades {
    public static void main(String[] args) {

        // student size
        final int inputSize = 5;

        // storage each grade number
        int[] gradesResult = new int[4];

        // input obj
        Scanner scanner = new Scanner(System.in);

        for (int i = 0; i < inputSize; i++) {
            // print info
            System.out.printf("The num %d student\n", i + 1);



            System.out.print("\tname : ");
            String inputName = scanner.nextLine();

            // read the user input
            System.out.print("\tgrade : ");
            char inputGrade = scanner.nextLine().charAt(0);


            // classify
            switch (inputGrade) {
                case 'A':
                    gradesResult[0] += 1;
                    break;

                case 'B':
                    gradesResult[1] += 1;
                    break;

                case  'C':
                    gradesResult[2] += 1;
                    break;

                case  'D':
                    gradesResult[3] += 1;
                    break;

                default:
                    System.out.println("Input is not valid, Must be : (A, B, C, D)\n");
                    i--;
                    continue;
            }

        }

        // print the result
        for (int i = 0; i < gradesResult.length; i++) {

            // The question did not specify that the student's name should be print
            System.out.printf("%c : %d\n", 'A' + i, gradesResult[i]);
        }

    }
}