# -------------------------------
# Python Loops Example
# -------------------------------

# 1️⃣ For loop with list
print("For loop over list:")
fruits = ["apple", "banana", "mango", "orange"]
for fruit in fruits:  # iterates over each element in the list
    print("Fruit:", fruit)
print()

# 2️⃣ For loop with range()
print("For loop with range():")
for i in range(1, 6):  # numbers from 1 to 5
    print("i =", i)
print()

# 3️⃣ While loop
print("While loop example:")
count = 1
while count <= 5:  # condition checked before each iteration
    print("Count =", count)
    count += 1
print()

# 4️⃣ Nested loops
print("Nested loops example:")
for i in range(1, 4):  # outer loop
    for j in range(1, 4):  # inner loop
        print(f"i = {i}, j = {j}")
print()

# 5️⃣ Loop with break
print("For loop with break:")
for num in range(1, 10):
    if num == 5:
        print("Breaking the loop at num =", num)
        break
    print("Current num:", num)
print()

# 6️⃣ Loop with continue
print("For loop with continue:")
for num in range(1, 10):
    if num % 2 == 0:  # skip even numbers
        continue
    print("Odd number:", num)
print()

# 7️⃣ Loop with else
print("Loop with else:")
for i in range(1, 6):
    print("i =", i)
else:
    print("Loop completed normally (else executed since no break occurred)")
print()

# 8️⃣ While loop with else
print("While loop with else:")
count = 1
while count <= 5:
    print("Count =", count)
    count += 1
else:
    print("While loop completed normally (else executed since no break occurred)")
print()

# 9️⃣ Using nested loops with break and continue
print("Nested loops with break and continue:")
for i in range(1, 4):
    for j in range(1, 6):
        if j == 4:
            print("Skipping j =", j, "using continue")
            continue
        if j == 5:
            print("Breaking inner loop at j =", j)
            break
        print(f"i={i}, j={j}")
print("Nested loop demo completed")
