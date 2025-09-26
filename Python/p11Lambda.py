# -----------------------------------------
# Lambda Functions - Complete Example
# -----------------------------------------

# 1. Basic lambda vs normal function
def square(x):
    return x ** 2

square_lambda = lambda x: x ** 2

print("Square using normal function:", square(5))
print("Square using lambda:", square_lambda(5))


# 2. Lambda with multiple arguments
add = lambda a, b: a + b
multiply = lambda x, y: x * y

print("Addition (10+5):", add(10, 5))
print("Multiplication (6*7):", multiply(6, 7))


# 3. Lambda without arguments
hello = lambda: "Hello, Sanket!"
print("Lambda with no args:", hello())


# 4. Lambda inside another function
def make_incrementer(n):
    """Returns a lambda that increments input by n"""
    return lambda x: x + n

inc5 = make_incrementer(5)
print("10 incremented by 5:", inc5(10))


# 5. Using lambda with map(), filter(), reduce()
numbers = [1, 2, 3, 4, 5]

# map -> apply function to each element
squares = list(map(lambda x: x ** 2, numbers))
print("Squares using map+lambda:", squares)

# filter -> keep elements matching condition
even_nums = list(filter(lambda x: x % 2 == 0, numbers))
print("Even numbers:", even_nums)

# reduce -> apply rolling operation
from functools import reduce
product = reduce(lambda x, y: x * y, numbers)
print("Product of list:", product)


# 6. Lambda with sorted()
students = [("Sanket", 20), ("Aayush", 19), ("Raj", 21)]
sorted_students = sorted(students, key=lambda student: student[1])
print("Students sorted by age:", sorted_students)


# 7. Lambda with conditional expression
even_or_odd = lambda x: "Even" if x % 2 == 0 else "Odd"
print("5 is:", even_or_odd(5))
print("10 is:", even_or_odd(10))
