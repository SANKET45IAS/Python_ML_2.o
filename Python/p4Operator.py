# -------------------------------
# Python Operators Example
# -------------------------------

# Arithmetic Operators
a = 10
b = 3

print("Arithmetic Operators:")
print("a + b =", a + b)  # Addition
print("a - b =", a - b)  # Subtraction
print("a * b =", a * b)  # Multiplication
print("a / b =", a / b)  # Division (float)
print("a // b =", a // b) # Floor Division (integer result)
print("a % b =", a % b)  # Modulus (remainder)
print("a ** b =", a ** b) # Exponentiation (power)
print()

# Assignment Operators
c = 5
print("Assignment Operators:")
c += 2  # c = c + 2
print("c += 2 ->", c)
c *= 3  # c = c * 3
print("c *= 3 ->", c)
c -= 4  # c = c - 4
print("c -= 4 ->", c)
c /= 2  # c = c / 2
print("c /= 2 ->", c)
print()

# Comparison Operators
x = 10
y = 20
print("Comparison Operators:")
print("x == y:", x == y)  # equal
print("x != y:", x != y)  # not equal
print("x > y:", x > y)    # greater than
print("x < y:", x < y)    # less than
print("x >= y:", x >= y)  # greater than or equal
print("x <= y:", x <= y)  # less than or equal
print()

# Logical Operators
p = True
q = False
print("Logical Operators:")
print("p and q:", p and q)  # AND
print("p or q:", p or q)    # OR
print("not p:", not p)      # NOT
print()

# Identity Operators
print("Identity Operators:")
print("x is y:", x is y)          # True if same object(value)
print("x is not y:", x is not y)  # True if not same object(value)
print()

# Membership Operators
list_numbers = [1, 2, 3, 4, 5]
print("Membership Operators:")
print("3 in list_numbers:", 3 in list_numbers)       # True if 3 is in list
print("10 not in list_numbers:", 10 not in list_numbers) # True if 10 not in list
print()

# Bitwise Operators
m = 5   # binary 101
n = 3   # binary 011
print("Bitwise Operators:")
print("m & n =", m & n)  # AND
print("m | n =", m | n)  # OR
print("m ^ n =", m ^ n)  # XOR
print("~m =", ~m)        # NOT
print("m << 1 =", m << 1)  # Left Shift
print("m >> 1 =", m >> 1)  # Right Shift
print()
