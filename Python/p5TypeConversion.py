# -------------------------------
# Python Type Conversion Example
# -------------------------------

# Original values
int_num = 10
float_num = 3.14
complex_num = 2 + 3j
str_num = "25"
str_text = "Python"
bool_val = True

# -------------------------------
# 1. Converting to Integer
# -------------------------------
print("Converting to Integer:")
print("int(float_num) ->", int(float_num))   # float to int (decimal part truncated)
print("int(str_num) ->", int(str_num))       # string to int
print("int(bool_val) ->", int(bool_val))     # boolean to int (True -> 1, False -> 0)
# print(int(str_text))  # ❌ Error → string "Python" cannot convert to int
print()

# -------------------------------
# 2. Converting to Float
# -------------------------------
print("Converting to Float:")
print("float(int_num) ->", float(int_num))   # int to float
print("float(str_num) ->", float(str_num))   # string to float
print("float(bool_val) ->", float(bool_val)) # boolean to float
# print(float(str_text)) # ❌ Error → "Python" cannot convert to float
print()

# -------------------------------
# 3. Converting to Complex
# -------------------------------
print("Converting to Complex:")
print("complex(int_num) ->", complex(int_num))         # int to complex (imag = 0)
print("complex(float_num) ->", complex(float_num))     # float to complex
print("complex(str_num) ->", complex(str_num))         # string number to complex
print("complex(bool_val) ->", complex(bool_val))       # boolean to complex
# print(complex(str_text))  # ❌ Error → "Python" cannot convert to complex
print()

# -------------------------------
# 4. Converting to String
# -------------------------------
print("Converting to String:")
print("str(int_num) ->", str(int_num))          # int to string
print("str(float_num) ->", str(float_num))      # float to string
print("str(complex_num) ->", str(complex_num))  # complex to string
print("str(bool_val) ->", str(bool_val))        # boolean to string
print()

# -------------------------------
# 5. Converting to Boolean
# -------------------------------
print("Converting to Boolean:")
print("bool(0) ->", bool(0))         # 0 is False
print("bool(25) ->", bool(25))       # non-zero int is True
print("bool('') ->", bool(''))       # empty string is False
print("bool('Python') ->", bool('Python')) # non-empty string is True
print("bool([]) ->", bool([]))       # empty list is False
print("bool([1,2,3]) ->", bool([1,2,3]))   # non-empty list is True
print("bool(None) ->", bool(None))   # None is False
