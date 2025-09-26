# ----------------------------------------
# Recursive Function: Factorial
# ----------------------------------------

def factorial(n):
    """
    Recursive function to find factorial of n.
    factorial(n) = n * factorial(n-1)
    Base case: factorial(0) = 1
    """
    if n == 0:   # base case (stop condition)
        return 1
    else:
        return n * factorial(n - 1)  # recursive call

# Example calls
print("Factorial of 5:", factorial(5))   # 5*4*3*2*1 = 120
print("Factorial of 0:", factorial(0))   # Base case → 1
