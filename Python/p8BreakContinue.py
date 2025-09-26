# -------------------------------
# Break and Continue Example
# -------------------------------

# 1️⃣ Using break
print("Example of break:")
for num in range(1, 11):
    if num == 5:
        print("Breaking the loop at num =", num)
        break  # exits the loop immediately
    print("Current number:", num)
print("Loop exited using break\n")

# Output explanation:
# The loop prints numbers 1 to 4, and breaks at 5, so 5 is not printed in the loop.

# 2️⃣ Using continue
print("Example of continue:")
for num in range(1, 11):
    if num % 2 == 0:  # skip even numbers
        continue  # skips current iteration and moves to next
    print("Odd number:", num)
print("Loop skipped even numbers using continue\n")

# Output explanation:
# The loop prints only odd numbers: 1, 3, 5, 7, 9. Even numbers are skipped.

# 3️⃣ Using break in a while loop
print("While loop with break:")
i = 1
while i <= 10:
    if i == 7:
        print("Breaking while loop at i =", i)
        break
    print("i =", i)
    i += 1
print("Exited while loop\n")

# 4️⃣ Using continue in a while loop
print("While loop with continue:")
i = 0
while i < 10:
    i += 1
    if i % 3 == 0:
        continue  # skip numbers divisible by 3
    print("Number not divisible by 3:", i)
print("Skipped numbers divisible by 3 using continue")
