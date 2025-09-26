# -------------------------------
# Example of variable types in Python
# -------------------------------

# Global variables
x = 100                  # integer literal
name = "Sanket"          # string literal
pi = 3.1415              # float literal
is_student = True        # boolean literal

# Numeric literals - new additions
complex_num = 2 + 3j      # complex literal
hex_num = 0x1A            # hexadecimal literal (26 in decimal)
oct_num = 0o12            # octal literal (10 in decimal)
bin_num = 0b1010          # binary literal (10 in decimal)

# Sequence types
numbers_list = [1, 2, 3, 4]   # list
numbers_tuple = (5, 6, 7)     # tuple
range_example = range(5)      # range (0,1,2,3,4)

# Mapping type
student = {"id": 1, "name": "Sanket"}  # dictionary

# Set types
unique_numbers = {1, 2, 3, 3, 4}       # set (duplicates auto removed)
frozen_numbers = frozenset([5, 6, 7])  # immutable set

# Binary types
byte_data = b"Hello"                   # bytes
mutable_bytes = bytearray(b"Hi")       # bytearray
memory_view = memoryview(byte_data)    # memoryview object

# -------------------------------
# Function demonstrating local variable
# -------------------------------
def demo_function():
    y = 50  # local variable
    print("Inside function -> local y:", y)
    print("Inside function -> global x (accessible):", x)

demo_function()

print("\nOutside function -> global variables:")
print("x:", x)
print("name:", name)
print("pi:", pi)
print("is_student:", is_student)
print("complex_num:", complex_num)
print("hex_num (0x1A):", hex_num)
print("oct_num (0o12):", oct_num)
print("bin_num (0b1010):", bin_num)
print("numbers_list:", numbers_list)
print("numbers_tuple:", numbers_tuple)
print("range_example:", list(range_example))
print("student dict:", student)
print("unique_numbers:", unique_numbers)
print("frozen_numbers:", frozen_numbers)
print("byte_data:", byte_data)
print("mutable_bytes:", mutable_bytes)
print("memory_view (as list of ASCII codes):", list(memory_view))

# -------------------------------
# Triple Quotes Example
# -------------------------------

# Triple quotes can be used for multi-line strings
multi_line_str = '''This is a string
that spans multiple lines
using triple single quotes.'''

print("\nMulti-line String (single quotes):")
print(multi_line_str)

# Triple double quotes can also be used
multi_line_str2 = """This is another string
spanning multiple lines
using triple double quotes."""

print("\nMulti-line String (double quotes):")
print(multi_line_str2)

# Triple quotes are also used for docstrings
def greet(name):
    """
    This function takes a name as input
    and prints a greeting message.
    """
    print(f"Hello, {name}!")

# Calling the function
greet("Sanket")

# Accessing the docstring
print("\nDocstring of greet function:")
print(greet.__doc__)  # prints the docstring inside triple quotes

# -------------------------------
# Scope Variables Example
# -------------------------------
x = 10  # global variable

def outer_function():
    # Enclosed scope
    y = 20
    
    def inner_function():
        # Local scope
        z = 30
        print("\nInside inner_function:")
        print("Local z:", z)           # Local available here
        print("Enclosed y:", y)        # Can access enclosed variable
        print("Global x:", x)          # Can access global variable
        print("Built-in len:", len([1, 2, 3]))  # Built-in available everywhere
        
    inner_function()
    
    print("\nInside outer_function:")
    print("Enclosed y:", y)  # accessible here
    print("Global x:", x)    # accessible here
    # print("Local z:", z)   # ❌ NOT accessible → z is local to inner_function only

outer_function()

print("\nOutside all functions:")
print("Global x:", x)  # accessible everywhere
# print("Enclosed y:", y)  # ❌ NOT accessible → y is local to outer_function
# print("Local z:", z)     # ❌ NOT accessible → z is local to inner_function
