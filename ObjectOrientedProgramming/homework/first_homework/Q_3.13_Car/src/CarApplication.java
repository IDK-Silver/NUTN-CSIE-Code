class Car {
    private String model;
    public void setModel(String model) {
        this.model = model;
    }
    public String getModel() {
        return this.model;
    }

    private String year;
    public void setYear(String year) {
        this.year = year;
    }
    public String getYear() {
        return this.year;
    }

    private double price;

    public void setPrice(double price) {

        // check input price is valid
        if (price < 0) {
            System.err.printf("Fail to set new price : %f is not valid\n", price);
            return;
        }

        // apply the new price
        this.price = price;
    }
    public double getPrice() {
        return this.price;
    }

    public Car(String model, String year, double price)  {
        this.model = model;
        this.year = year;
        this.price = price;
    }


    public void showInfo() {
        System.out.printf(
                "Model: %s, Year: %s, Price: %f\n",
                this.model, this.year, this.price
        );
    }
}

public class CarApplication {
    public static void main(String[] args) {
        // create two obj that type is Car
        Car Car1 = new Car("Kano", "2020", 2525.25);
        Car Car2 = new Car("Nana", "2021", 5252.25);

        // display each object’s price
        Car1.showInfo();
        Car2.showInfo();

        // Then apply a 5% discount on the price of the first car
        Car1.setPrice(Car1.getPrice() * (100 - 5) / 100);

        //  and a 7% discount on the price of the second
        Car2.setPrice(Car2.getPrice() * (100 - 7) / 100);

        //  Display each Car’s price again.
        System.out.println();
        Car1.showInfo();
        Car2.showInfo();
    }
}