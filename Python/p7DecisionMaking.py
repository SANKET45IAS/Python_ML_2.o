# -------------------------------
# Python Decision Making Statements Example
# -------------------------------

# 1️⃣ Simple if statement
number = int(input("Enter a number: "))
if number > 0:  # condition
    print("The number is positive.")  # executed only if condition is True

print()

# 2️⃣ if-else statement
number = int(input("Enter another number: "))
if number % 2 == 0:  # check even number
    print(number, "is Even")
else:
    print(number, "is Odd")

print()

# 3️⃣ if-elif-else statement
marks = int(input("Enter your marks (0-100): "))
if marks >= 90:
    print("Grade: A")
elif marks >= 75:
    print("Grade: B")
elif marks >= 60:
    print("Grade: C")
elif marks >= 50:
    print("Grade: D")
else:
    print("Grade: F")

print()

# 4️⃣ Nested if statement
age = int(input("Enter your age: "))
if age >= 18:
    print("You are eligible to vote.")
    citizenship = input("Are you a citizen? (yes/no): ")
    if citizenship.lower() == "yes":
        print("You can cast your vote.")
    else:
        print("You cannot vote, not a citizen.")
else:
    print("You are not eligible to vote yet.")

print()

# 5️⃣ Conditional (ternary) operator
num1 = int(input("Enter first number for comparison: "))
num2 = int(input("Enter second number for comparison: "))
max_num = num1 if num1 > num2 else num2
print("The maximum number is:", max_num)

print()

# 6️⃣ Using logical operators in decision making
temperature = int(input("Enter the temperature (°C): "))
if temperature > 30 and temperature <= 45:
    print("It's a hot day.")
elif temperature > 20 and temperature <= 30:
    print("It's a pleasant day.")
elif temperature > 0 and temperature <= 20:
    print("It's a cold day.")
else:
    print("Temperature is extreme, stay safe!")

print()

# 7️⃣ Membership operator in decision making
fruits = ["apple", "banana", "mango", "orange"]
fruit_input = input("Enter a fruit name: ")
if fruit_input.lower() in fruits:
    print(fruit_input, "is available in the list.")
else:
    print(fruit_input, "is not available in the list.")
