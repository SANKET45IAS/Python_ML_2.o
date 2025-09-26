# ----------------------------
# File: mymodule.py
# ----------------------------

# Variables
PI = 3.1415
greeting = "Hello, Sanket!"

# Functions
def add(a, b):
    """Returns the sum of a and b"""
    return a + b

def factorial(n):
    """Returns factorial of n using recursion"""
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)

# Class
class Circle:
    """Represents a circle with radius"""
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return PI * self.radius ** 2
