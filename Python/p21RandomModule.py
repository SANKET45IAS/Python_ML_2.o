# ---------------------------------------------------------
# ✅ Python Random Module Example (Detailed)
# ---------------------------------------------------------

# 1. Import the random module
import random

# 🔹 2. Generate Random Numbers
print("\n--- Random Numbers ---")

# Random float: 0 ≤ x < 1
rand_float = random.random()
print("Random float (0 to 1):", rand_float)

# Random float within a range
rand_uniform = random.uniform(10, 20)  # 10 ≤ x < 20
print("Random float between 10 and 20:", rand_uniform)

# Random integer within a range
rand_int = random.randint(1, 100)      # inclusive of both ends
print("Random integer (1 to 100):", rand_int)

# Random integer with step
rand_range = random.randrange(0, 50, 5) # 0,5,10,...45
print("Random integer from range(0,50,5):", rand_range)

# 🔹 3. Random Selection from a List
print("\n--- Random Selection ---")
items = ["apple", "banana", "cherry", "orange"]

single_choice = random.choice(items)     # pick one item
print("Random choice:", single_choice)

multiple_choice = random.choices(items, k=3)  # pick 3 items with replacement
print("Random choices (with replacement, k=3):", multiple_choice)

# Random sample (without replacement)
sample_choice = random.sample(items, k=2)
print("Random sample (without replacement, k=2):", sample_choice)

# 🔹 4. Shuffle a List
print("\n--- Shuffle List ---")
numbers = [1, 2, 3, 4, 5]
print("Original List:", numbers)
random.shuffle(numbers)                   # shuffle in place
print("Shuffled List:", numbers)

# 🔹 5. Random Seed
print("\n--- Random Seed ---")
# Seed makes random numbers reproducible
random.seed(42)
print("Random integer with seed 42:", random.randint(1, 100))
random.seed(42)
print("Random integer with seed 42 (again):", random.randint(1, 100))

# 🔹 6. Practical Uses
print("\n--- Practical Uses ---")

# 1. Simulate dice roll
dice_roll = random.randint(1, 6)
print("Dice Roll:", dice_roll)

# 2. Random password generator
import string
password_chars = string.ascii_letters + string.digits + string.punctuation
password = ''.join(random.choices(password_chars, k=10))
print("Random Password:", password)

# 3. Randomly select a winner
participants = ["Sanket", "Aayush", "Raj", "Rohit"]
winner = random.choice(participants)
print("Randomly Selected Winner:", winner)

# 4. Random shuffle in games
deck = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
random.shuffle(deck)
print("Shuffled Deck:", deck)
