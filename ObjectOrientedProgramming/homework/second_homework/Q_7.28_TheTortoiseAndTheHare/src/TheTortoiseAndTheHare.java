import java.util.ArrayList;
import java.util.Random;

class moveInfo {
	
	moveInfo(int leftRange, int rightRange, int offset)
	{
		this.leftRange = leftRange;
		this.rightRange = rightRange;
		this.offset = offset;
	}
	
	int leftRange;
	int rightRange;
	int offset;
	
	Boolean isMatch(int position) {
		
		if (position >= this.leftRange && position <= this.rightRange)
		{
			return true;
		}
		return false;
	}
}

class Entity {
	
	Entity(String name, ArrayList<String> bindingMap)
	{
		this.name = name;
		this.moveData = new ArrayList<moveInfo>();
		this.map = bindingMap;
		this.mapSign = this.name.charAt(0);
	}
	
	ArrayList<String> map;
	
	String name;
	String winSlogan;
	Character mapSign;
	
	void setName(String name) {
		this.name = name;
	}
	
	String getName() {
		return this.name;
	}
	
	
	int position = 0;
	
	void setPosition(int position) {
		this.position = position;
	}
	
	int getPosition() {
		return this.position;
	}
	
	ArrayList<moveInfo> moveData;
	
	int rand() {
		Random r = new Random();
		return r.nextInt(9) + 1;
	}
	
	void randMove() {
		
		// get random number by entity itself random function
		int randNumber = this.rand();
		
		// the position offset
		int offset = 0;
		
		Boolean isMoveInfoExists = false;;
		
		
		// get the position offset 
		for (moveInfo info : moveData)
		{
			if (info.isMatch(randNumber))
			{
				offset = info.offset;
				isMoveInfoExists = true;
			}
		}
		
		// can't find offset
		if (!isMoveInfoExists) {
			System.err.printf("Error : the entity %s try to random move but can't find vaild offset with random number "
					+ "check the random function and moveData\n", this.name);
			System.exit(-1);
		}
		
		// get new position by current position add offset
		int newPosition = this.position + offset;
		
		
		
		// check newPosition is in range, if not fix it
		if (newPosition < 0)
		{
			newPosition = 0;
		}
		else if (newPosition >= this.map.size())
		{
			newPosition = map.size() - 1;
		}
		
		// if two  entity is in same position call crash function to solve it.
		if (!this.map.get(newPosition).isEmpty() && this.map.get(newPosition) != Character.toString(this.mapSign))
		{
			this.crash(newPosition);
		}
		// mark map
		else
		{
			
			this.map.set(newPosition, Character.toString(this.mapSign));
		}
		
		// clear the old position
		this.map.set(this.position, "");
		
		// change current position to new position
		this.position = newPosition;
		
	}
	
	// the method that how to solving two entity is in same position.
	void crash(int position)
	{
		this.map.set(position, "OUCH!");
//		System.out.println("OUCH!");
	}
}



class Tortoise extends Entity
{

	Tortoise(ArrayList<String> bindingMap) {
		super("Tortoise", bindingMap);

		this.moveData.add(new moveInfo(1, 5, 3));
		this.moveData.add(new moveInfo(6, 7, -6));
		this.moveData.add(new moveInfo(8, 10, 1));
		this.winSlogan = "TORTOISE WINS!!! YAY!!!";
	}
	
}


class Hare extends Entity
{

	Hare(ArrayList<String> bindingMap) {
		super("Hare", bindingMap);

		this.moveData.add(new moveInfo(1, 2, 0));
		this.moveData.add(new moveInfo(3, 4, 9));
		this.moveData.add(new moveInfo(5, 5, -12));
		this.moveData.add(new moveInfo(6, 8, 1));
		this.moveData.add(new moveInfo(9, 10, -2));
		this.winSlogan = "Hare wins. Yuch.";
	}
	
}


public class TheTortoiseAndTheHare {

	public static void main(String[] args) {
		
		// map size
		final int mapSize = 70;
		
		// initial the map 
		ArrayList<String> map = new ArrayList<String>();
		
		for (int i = 0; i < mapSize; i++)
		{
			map.add("");
		}
		
		// create each entity 
		ArrayList<Entity> entityList = new ArrayList<Entity>();
		
		entityList.add(new Tortoise(map));
		entityList.add(new Hare(map));
		
		//  run game until the winner is appear 
		System.out.println("BANG !!!!! AND THEY'RE OFF!!!!!");
		while (map.get(mapSize - 1).isEmpty())
		{	
			// each entity run
			for (Entity entity : entityList)
			{
				entity.randMove();
			}
			
			// print map
			System.out.printf("|");
			for (String sign : map)
			{
				System.out.printf("%1s|", sign);
			}
			System.out.println();
			
		}
		
		// list  the  winners
		ArrayList<Entity> winnerList = new ArrayList<Entity>();
		for (Entity entity : entityList)
		{
			if (entity.position == mapSize -1)
			{
				winnerList.add(entity);
			}
		}
		
		
		// print winners
		if (winnerList.size() == 1)
		{
			System.out.println(winnerList.get(0).winSlogan);
		}
		else if (winnerList.size() == entityList.size()) {
			System.out.println("It's a tie");
		}
		else
		{
			System.out.print("Winers are  : ");
			for (Entity entity : winnerList)
			{
				System.out.printf("%s ",entity.name);
			}
			System.out.println();
		}
		
		
	
	}

}
