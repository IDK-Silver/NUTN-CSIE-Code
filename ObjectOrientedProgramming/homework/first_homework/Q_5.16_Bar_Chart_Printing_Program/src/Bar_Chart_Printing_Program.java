import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Scanner;

public class Bar_Chart_Printing_Program {

    public static void main(String[] args) {

        // input size
        final int inputSize = 5;

        // the number of asterisks that user want
        int inputNum = 0;

        // input buffer
        ArrayDeque<Integer> queue = new ArrayDeque<>();

        // get system in obj
        Scanner scanner = new Scanner(System.in);

        for (int i = 0; i < inputSize; i++)
        {
            inputNum = scanner.nextInt();

            // check input is valid
            if (inputNum < 1 || inputNum > 30)
            {
                System.err.println("Input is not valid");
                i--;
                continue;
            }

            // add to the queue
            queue.add(inputNum);
        }

        // print each line
        for (Integer num : queue) {
            System.out.printf("%s\n",  String.join("", Collections.nCopies(num, "*")));
        }
    }
}