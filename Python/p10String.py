# -----------------------------------------
# STRING BASICS
# -----------------------------------------
s = "  Hello, Sanket! Welcome to Python.  "

print("Original String:", repr(s))
print("Length:", len(s))  # total characters including spaces

# -----------------------------------------
# ACCESSING BY INDEX
# -----------------------------------------
print("\n--- Accessing characters by index ---")
print("First character:", s[0])       # index 0 → ' '
print("7th character:", s[6])         # index 6
print("Last character:", s[-1])       # last index
print("Second last character:", s[-2])  

# -----------------------------------------
# STRING SLICING
# -----------------------------------------
print("\n--- Slicing examples ---")
print("s[2:9] ->", s[2:9])          # from index 2 to 8
print("s[:5] ->", s[:5])            # from start to index 4
print("s[7:] ->", s[7:])            # from index 7 till end
print("s[::2] ->", s[::2])          # every 2nd character
print("s[::-1] ->", s[::-1])        # reverse string

# -----------------------------------------
# IMPORTANT STRING METHODS
# -----------------------------------------

print("\n--- Case methods ---")
print("Lowercase:", s.lower())
print("Uppercase:", s.upper())
print("Title case:", s.title())       # Each word first letter capital
print("Capitalize:", s.capitalize())  # Only first letter capital
print("Swapcase:", s.swapcase())      # Swap upper <-> lower

print("\n--- Searching & Finding ---")
print("Count of 'o':", s.count("o"))
print("Find 'Sanket':", s.find("Sanket"))   # index of first occurrence
print("Index of 'Python':", s.index("Python"))  # like find() but error if not found
print("startswith 'Hello':", s.strip().startswith("Hello"))
print("endswith 'Python.':", s.strip().endswith("Python."))

print("\n--- Checking content ---")
print("Is alphanumeric?:", "Sanket123".isalnum())
print("Is alpha?:", "Sanket".isalpha())
print("Is digit?:", "12345".isdigit())
print("Is decimal?:", "99".isdecimal())
print("Is numeric?:", "½".isnumeric())   # True for fractions too
print("Is space?:", "   ".isspace())
print("Is lower?:", "python".islower())
print("Is upper?:", "PYTHON".isupper())
print("Is title?:", "Python Programming".istitle())

print("\n--- Modification ---")
print("Replace 'Sanket' with 'Friend':", s.replace("Sanket", "Friend"))
print("Strip (remove spaces):", s.strip())
print("Left strip:", s.lstrip())
print("Right strip:", s.rstrip())
print("Center aligned (width 40):", s.center(40, "-"))
print("Left just (width 30):", s.ljust(30, "*"))
print("Right just (width 30):", s.rjust(30, "*"))
print("Zero fill (width 10):", "42".zfill(10))

print("\n--- Splitting & Joining ---")
print("Split by spaces:", s.split())
print("Split by comma:", s.split(","))
print("Partition by 'Welcome':", s.partition("Welcome"))   # returns tuple
words = ["Python", "is", "fun"]
print("Join words:", " ".join(words))

print("\n--- Encoding & Formatting ---")
print("Encode to bytes:", "Sanket".encode("utf-8"))
print("Format string:", "My name is {}, age {}".format("Sanket", 20))
print("Formatted with f-string:", f"My name is {'Sanket'}, age {20}")

print("\n--- Misc ---")
print("Expandtabs:", "a\tb\tc".expandtabs(4))  # replace tab with spaces
print("Translate:", "abc".translate({97: "1", 98: "2"}))  # map ascii to chars
print("maketrans + translate:", "hello".translate(str.maketrans("hel", "xyz")))

 