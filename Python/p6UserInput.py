# -------------------------------
# Python User Input Example
# -------------------------------

# 1️⃣ Getting input as a string (default)
name = input("Enter your name: ")  # input() always returns a string
print("Hello,", name, "!")          # prints the string entered

# 2️⃣ Getting integer input
age_str = input("Enter your age: ")
age = int(age_str)  # convert string input to integer
print("Your age is:", age)

# 3️⃣ Getting float input
height_str = input("Enter your height in meters: ")
height = float(height_str)  # convert string input to float
print("Your height is:", height, "meters")

# 4️⃣ Getting multiple inputs in a single line
# User can enter numbers separated by space
numbers = input("Enter 3 numbers separated by space: ")  # e.g. 5 10 15
num_list = numbers.split()  # split string into list of strings
num1 = int(num_list[0])
num2 = int(num_list[1])
num3 = int(num_list[2])
print("Numbers entered:", num1, num2, num3)
print("Sum of numbers:", num1 + num2 + num3)

# 5️⃣ Getting boolean input
is_student_input = input("Are you a student? (yes/no): ")
# convert yes/no to boolean
is_student = True if is_student_input.lower() == 'yes' else False
print("Student status:", is_student)

# 6️⃣ Input with validation (loop until correct input)
while True:
    try:
        score = int(input("Enter your score (0-100): "))
        if 0 <= score <= 100:
            print("Valid score entered:", score)
            break
        else:
            print("Score must be between 0 and 100. Try again.")
    except ValueError:
        print("Invalid input! Please enter a number.")

# 7️⃣ Using input for arithmetic operation
num_a = int(input("Enter first number: "))
num_b = int(input("Enter second number: "))
print(f"{num_a} + {num_b} =", num_a + num_b)
print(f"{num_a} - {num_b} =", num_a - num_b)
print(f"{num_a} * {num_b} =", num_a * num_b)
print(f"{num_a} / {num_b} =", num_a / num_b) #give in float
print(f"{num_a} // {num_b} =", num_a // num_b) #give in it 
print(f"{num_a} % {num_b} =", num_a % num_b)         