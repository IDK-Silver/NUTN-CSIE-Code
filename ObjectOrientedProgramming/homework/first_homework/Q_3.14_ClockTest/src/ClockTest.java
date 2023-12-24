class Clock {

    private int hour;

    public void setHour(int hour) {

        this.hour = hour;
        checkTimeValueValid();
    }

    public int getHour() {
        return hour;
    }

    private int minute;

    public void setMinute(int minute) {
        this.minute = minute;
        checkTimeValueValid();
    }

    public int getMinute() {
        return minute;
    }

    private int second;

    public void setSecond(int second) {

        this.second = second;
        checkTimeValueValid();
    }

    public int getSecond() {
        return second;
    }

    private void checkTimeValueValid()
    {
        if (this.hour >= 24 || this.minute >= 60|| this.second >= 60)
        {
            System.err.println("Error : time value is not valid, set to default value");
            this.hour = 0;
            this.minute = 0;
            this.second = 0;
        }
    }

    public Clock(int hour, int minute, int second) {
        this.hour = hour;
        this.minute = hour;
        this.second = second;
    }


    public void displayTime() {
        System.out.printf(
                "Time -> %2d : %2d : %2d\n",
                this.hour, this.minute, this.second
        );
    }
}


public class ClockTest {
    public static void main(String[] args) {
        // create clock
        Clock clock = new Clock(9, 10, 20);

        // display clock time
        clock.displayTime();

        // set the valid time
        System.out.println("\nSet the valid time");
        clock.setHour(23);
        clock.displayTime();


        // set the not valid time
        System.out.println("\nSet the not valid time ");
        clock.setHour(24);
        clock.displayTime();

    }
}