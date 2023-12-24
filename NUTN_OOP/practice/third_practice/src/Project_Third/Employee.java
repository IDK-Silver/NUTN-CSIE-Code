package Project_Third;

public abstract class Employee implements Payable {

    public abstract double earnings();

    public double getPaymentAmount() {
        return earnings();
    }
}
