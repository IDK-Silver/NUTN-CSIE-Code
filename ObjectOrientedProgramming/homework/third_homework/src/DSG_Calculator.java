import javax.swing.*;
import java.awt.*;
import javax.swing.SpringLayout;

public class DSG_Calculator extends JFrame {

    DSG_Calculator() {
        initUI();
    }

    private void initUI() {

        setTitle("DSG Calculator");
        setSize(480, 340);
        setLocationRelativeTo(null);
        setResizable(false);
        setDefaultCloseOperation(EXIT_ON_CLOSE);

        var pane = getContentPane();
        var layout = new SpringLayout();
        pane.setLayout(layout);

        JTextField valueTextField = new JTextField("0");
        valueTextField.setBackground(new Color(224, 130, 130));
        valueTextField.setForeground(Color.black);
        valueTextField.setHorizontalAlignment(SwingConstants.RIGHT);
        valueTextField.setFont(new Font("Sanserif", Font.PLAIN, 24));
        valueTextField.setEditable(false);
        valueTextField.setPreferredSize(new Dimension(280, 60));

        JLabel programNameJLabel = new JLabel("DSG Calculator");
        programNameJLabel.setPreferredSize(new Dimension(110, valueTextField.getPreferredSize().height));
        programNameJLabel.setFont( new Font ("Sanserif", Font.ITALIC + Font.BOLD, 14));
        JButton[] numberButtons = new JButton[11];



        pane.add(valueTextField);
        pane.add(programNameJLabel);

        for (int i = 0; i < 11; i++) {
            numberButtons[i] = new JButton(Integer.toString(i + 1));
            numberButtons[i].setPreferredSize(new Dimension(60, 48));
            numberButtons[i].setFont(new Font("Sanserif", Font.BOLD, 16));
            pane.add(numberButtons[i]);
        }

        numberButtons[10].setText(".");
        numberButtons[9].setText("0");
        numberButtons[9].setPreferredSize(new Dimension(numberButtons[0].getPreferredSize().width * 2, numberButtons[0].getPreferredSize().height));


        JButton equalButton = new JButton("=");
        equalButton.setFont( new Font("Sanserif", Font.BOLD, 32));
        equalButton.setForeground(Color.RED);
        equalButton.setPreferredSize( new Dimension((int)(numberButtons[1].getPreferredSize().width * 1.6), numberButtons[0].getPreferredSize().height * 2 - 3));
        pane.add(equalButton);

        JButton clearButton = new JButton("Clear");
        clearButton.setFont( new Font("Sanserif", Font.BOLD, 24));
        clearButton.setForeground(Color.RED);
        clearButton.setPreferredSize(equalButton.getPreferredSize());
        pane.add(clearButton);


        JButton[] operatorButtons = new JButton[4];

        for (int i = 0; i < 4; i++) {
            operatorButtons[i] = new JButton("");
            operatorButtons[i].setPreferredSize(numberButtons[1].getPreferredSize());
            operatorButtons[i].setFont(new Font("Sanserif", Font.BOLD, 28));
            operatorButtons[i].setForeground(Color.BLUE);
            pane.add(operatorButtons[i]);
        }

        operatorButtons[0].setText("+");
        operatorButtons[1].setText("-");
        operatorButtons[2].setText("x");
        operatorButtons[3].setText("/");


        layout.putConstraint(SpringLayout.WEST, valueTextField, 30, SpringLayout.WEST, pane);
        layout.putConstraint(SpringLayout.NORTH, valueTextField, 10, SpringLayout.NORTH, pane);

        layout.putConstraint(SpringLayout.WEST, programNameJLabel, 30, SpringLayout.EAST, valueTextField);
        layout.putConstraint(SpringLayout.NORTH, programNameJLabel, 0, SpringLayout.NORTH, valueTextField);

        layout.putConstraint(SpringLayout.NORTH, numberButtons[0], 10, SpringLayout.SOUTH, valueTextField);
        layout.putConstraint(SpringLayout.WEST, numberButtons[0], 0, SpringLayout.WEST, valueTextField);

        for (int i = 1; i < 11; i++) {
            if ((i % 3) == 0) {
                layout.putConstraint(SpringLayout.NORTH, numberButtons[i], 0, SpringLayout.SOUTH, numberButtons[i - 3]);
                layout.putConstraint(SpringLayout.WEST, numberButtons[i], 0, SpringLayout.WEST, numberButtons[i - 3]);
            }
            else {
                layout.putConstraint(SpringLayout.WEST, numberButtons[i], 0, SpringLayout.EAST, numberButtons[i - 1]);
                layout.putConstraint(SpringLayout.NORTH, numberButtons[i], 0, SpringLayout.NORTH, numberButtons[i - 1]);
            }
        }

        layout.putConstraint(SpringLayout.NORTH, operatorButtons[0], 0, SpringLayout.NORTH, numberButtons[1]);
        layout.putConstraint(SpringLayout.EAST, operatorButtons[0], 0, SpringLayout.EAST, valueTextField);

        for (int i = 1; i < 4; i++) {
            layout.putConstraint(SpringLayout.NORTH, operatorButtons[i], 0, SpringLayout.SOUTH, operatorButtons[i - 1]);
            layout.putConstraint(SpringLayout.WEST, operatorButtons[i], 0, SpringLayout.WEST, operatorButtons[i - 1]);
        }


        layout.putConstraint(SpringLayout.WEST, equalButton, 0, SpringLayout.WEST, programNameJLabel);
        layout.putConstraint(SpringLayout.NORTH, equalButton, 0, SpringLayout.NORTH, operatorButtons[0]);

        layout.putConstraint(SpringLayout.WEST, clearButton, 0, SpringLayout.WEST, programNameJLabel);
        layout.putConstraint(SpringLayout.SOUTH, clearButton, 0, SpringLayout.SOUTH, operatorButtons[3]);
    }

    
    public static void main(String[] args) {
        EventQueue.invokeLater( () -> {
                var ex = new DSG_Calculator();
                ex.setVisible(true);
            }
        );
    }



}