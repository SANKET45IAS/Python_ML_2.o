# Except 0 every negative and positive integer are consider as True in every programming language.
# -------------------------------
# Python Tokens Example
# -------------------------------

# ---- 1. Keywords ----
# Keywords are reserved words in Python with special meaning.
def greet():   # 'def' is a keyword used to define a function
    # 'return' is also a keyword (will use later)

    # ---- 2. Identifiers ----
    # Identifiers are names given to variables, functions, classes, etc.
    message = "Hello, Sanket!"   # 'message' is an identifier (variable name)

    # ---- 3. Literals ----
    # Literals are constant values written directly in code.
    num = 10          # integer literal
    pi = 3.14         # float literal
    is_fun = True     # boolean literal
    text = "Python"   # string literal

    # ---- 4. Operators ----
    # Operators are symbols that perform operations.
    result = num + pi  # '=' is assignment operator, '+' is arithmetic operator

    # ---- 5. Punctuators (Separators) ----
    # Punctuators are symbols that structure code (colons, commas, brackets, etc.)
    if is_fun:   # ':' is a punctuator
        print(message, text, result)  # '()' and ',' are punctuators

    # Using return keyword
    return result   # 'return' is a keyword

# ---- Function call (Identifier + Punctuators) ----
output = greet()  # 'greet' is an identifier, '()' punctuator for function call
print("Returned Value:", output)  # prints final returned result
