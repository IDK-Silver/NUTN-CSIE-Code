import java.lang.reflect.Array;
import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

class Card {
	private final String face;
	private final String suit;
	
	public Card(String cardFace, String cardSuit) {
		this.face = cardFace;
		this.suit = cardSuit;
	}
	
	public String toString() {
		return this.face + " of " + suit;
	}
	
	public String getFace() { return this.face; }
	public String getSuit() { return this.suit; }
	
}

enum CardContains {
	None,
	Pair,
	TwoPair,
	ThreeOfKind,
	Straight,
	Flush,
	FullHouse,
	FourOfKind
}

class DeckOfCards {
	private static final SecureRandom randomNumbers = new SecureRandom();
	private static final int NUMBER_OF_CARDS = 52;
	
	private Card[] deck = new Card[NUMBER_OF_CARDS];
	private int currentCard = 0;
	
	int getCurrentCard() { return this.currentCard; }
	
	public DeckOfCards() {
		String[] faces = { "Ace", "Deuce", "Three", "Four", "Five", "Six", "Seven",
														"Eight", "Nine", "Ten", "Jack", "Queen", "King" };
		
		String[] suits = { "Hearts", "Diamonds", "Clubs", "Spades" };
		
		for (int count = 0; count < this.deck.length; count++) {
			deck[count] = new Card(faces[count % 13], suits[count / 13]);
		}
	}
	
	public void shuffle() {
		this.currentCard = 0;
		
		for (int first = 0; first < this.deck.length; first++) {
			int second = randomNumbers.nextInt(NUMBER_OF_CARDS);
			
			Card temp = this.deck[first];
			this.deck[first] = this.deck[second];
			this.deck[second] = temp;
			
		}
	}
	
	public Card dealCard() {
		if (this.currentCard < this.deck.length) {
			return this.deck[this.currentCard++];
		}
		else {
			return null;
		}
	}
	
	public static CardContains getCardContains(Card[] cards) {
		
		HashMap<String, Integer> stackBySuit = new 	HashMap<String, Integer>();
		HashMap<String, Integer> stackByFace = new 	HashMap<String, Integer>();
		HashMap<Card, Integer> stackByCard = new 	HashMap<Card, Integer>();
		
		
		
		for (Card card : cards) {
			stackBySuit.merge(card.getSuit(), 1, Integer::sum);
			stackByFace.merge(card.getFace(), 1, Integer::sum);
			stackByCard.merge(card, 1, Integer::sum);
		}
		
		// FourOfKind
		if (Collections.max(stackByFace.values()) >= 4)
		{
			return CardContains.FourOfKind;
		}
	    
		// Full House
		Integer[] stackByFaceArray = stackByFace.values().toArray(new Integer[0]);
		Arrays.sort(stackByFaceArray);
		
		if (stackByFaceArray.length == 2) {
			if (stackByFaceArray[0] == 3 && stackByFaceArray[1] == 2)
			{
				return CardContains.FullHouse;
			}
		}
		
		// Flush
		if (Collections.max(stackBySuit.values()) == 5) {
			return CardContains.Flush;
		}
		
		// Straight
		Integer[] stackByCardArray = stackByCard.values().toArray(new Integer[0]);
		Arrays.sort(stackByCardArray);
		if (stackByFaceArray.length == 5) {
			int currentIndex = 0;
			
			// compare i_th + 1 is equal (i + 1)_th ,  for all 0 <= i <= container size
			while (currentIndex + 1 < stackByCardArray.length) {
				if (stackByCardArray[currentIndex] + 1 == stackByCardArray[currentIndex + 1]) {
					currentIndex++;
					continue;
				}
				break;
			}
			
			// After compare all element, if cards is straight then currentIndex must be stackByFaceArray.length - 1
			if (currentIndex == stackByFaceArray.length - 1) {
				return CardContains.Straight;
			}
		}
		
		// ThreeofKind
		for (int num : stackByFaceArray) {
			if (num > 3)
				return CardContains.ThreeOfKind;
		}
		
		// TwoPair and Pair 		
		int maxStackSameCarNumber = 0;
		
		for (int num : stackByFaceArray) {
			maxStackSameCarNumber += num / 2;
		}
		
		if (maxStackSameCarNumber == 2) {
			return CardContains.TwoPair;
		}
		else if (maxStackSameCarNumber == 1){
			return CardContains.Pair;
		}
		
		// None 
		return CardContains.None;
	}
	
}

public class CardShufflingAndDealing {

	public static void main(String[] args) {
		
		final int stackOfCardSize = 5;
		DeckOfCards cards = new DeckOfCards();
		cards.shuffle();
		
		Card[] Player_1 = new Card[stackOfCardSize];
		Card[] Player_2 = new Card[stackOfCardSize];
		
		// each player deal card 
		while (cards.getCurrentCard() + 10 <= 52 - 1) {
			for (int i = 0; i < stackOfCardSize; i++) {
				Player_1[i] = cards.dealCard();
				Player_2[i] = cards.dealCard();
			}
			
			// show card
			CardContains p1= DeckOfCards.getCardContains(Player_1);
			System.out.printf("Player 1 : %-10s : ", p1);
				
			for (int j = 0; j < stackOfCardSize; j++) {
				System.out.printf("%-19s", Player_1[j]);
			}
				
			CardContains p2= DeckOfCards.getCardContains(Player_2);
			System.out.printf("\nPlayer 2 : %-10s : ", p2);
				
			for (int j = 0; j < stackOfCardSize; j++) {
				System.out.printf("%-19s", Player_2[j]);
			}
				
			if (p1.ordinal() == p2.ordinal()) {
				System.out.println("\nTie\n");
			}
			else {
					System.out.printf("\nWinner is %s\n\n",
					(p1.ordinal() > p2.ordinal() ? "Player 1":"Player 2"));
			}		
		}
	}			
}