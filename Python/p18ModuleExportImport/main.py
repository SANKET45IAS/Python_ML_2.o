# ----------------------------
# File: main.py
# ----------------------------

# 1. Import the whole module
import mymodule

print("Module Variable PI:", mymodule.PI)
print("Module Variable Greeting:", mymodule.greeting)

# Call function from module
print("Add 5 + 10:", mymodule.add(5, 10))
print("Factorial of 5:", mymodule.factorial(5))

# Use class from module
circle1 = mymodule.Circle(4)
print("Area of Circle with radius 4:", circle1.area())

# --------------------------------------------------
# 2. Import specific items from module
from mymodule import add, factorial, greeting

print("\nUsing specific imports:")
print("Add 20 + 30:", add(20, 30))
print("Factorial of 6:", factorial(6))
print("Greeting:", greeting)

# --------------------------------------------------
# 3. Import with alias
import mymodule as mm

print("\nUsing module alias:")
print("Add 2 + 3 using alias:", mm.add(2, 3))
