# ---------------------------------------------------------
# ✅ Python Math Module Example (Detailed)
# ---------------------------------------------------------

# 1. Importing the math module
import math

# 🔹 2. Constants
print("Math Constants:")
print("Pi (π):", math.pi)         # π ≈ 3.14159
print("Euler's number (e):", math.e)  # e ≈ 2.71828
print("Tau (τ):", math.tau)       # τ = 2π

# 🔹 3. Basic Mathematical Functions
print("\nBasic Math Functions:")
num = -15.7
print("Absolute value of", num, ":", math.fabs(num))  # Absolute value (float)
print("Ceiling of", num, ":", math.ceil(num))         # Smallest integer ≥ num
print("Floor of", num, ":", math.floor(num))          # Largest integer ≤ num
print("Round of", num, ":", round(num))               # Round to nearest integer

# 🔹 4. Power and Square Root
x = 5
y = 3
print("\nPower and Square Root:")
print(f"{x} raised to {y}:", math.pow(x, y))         # x^y as float
print("Square root of", x, ":", math.sqrt(x))         # √x

# 🔹 5. Logarithmic Functions
print("\nLogarithmic Functions:")
num = 10
print("Natural log of", num, ":", math.log(num))      # ln(x)
print("Log base 10 of", num, ":", math.log10(num))    # log10(x)
print("Log base 2 of", num, ":", math.log2(num))      # log2(x)

# 🔹 6. Trigonometric Functions
print("\nTrigonometric Functions:")
angle_deg = 45
angle_rad = math.radians(angle_deg)  # Convert degrees to radians

print(f"sin({angle_deg}°):", math.sin(angle_rad))
print(f"cos({angle_deg}°):", math.cos(angle_rad))
print(f"tan({angle_deg}°):", math.tan(angle_rad))

# 🔹 7. Inverse Trigonometric Functions
value = 0.7071
print("\nInverse Trigonometric Functions:")
print("arcsin(0.7071) in radians:", math.asin(value))
print("arccos(0.7071) in radians:", math.acos(value))
print("arctan(1) in radians:", math.atan(1))

# 🔹 8. Other Useful Functions
numbers = [4, 7, 1, 9, 12]
print("\nOther Useful Functions:")
print("Minimum:", min(numbers))
print("Maximum:", max(numbers))
print("Factorial of 5:", math.factorial(5))          # 5! = 120
print("GCD of 12 and 18:", math.gcd(12, 18))        # Greatest common divisor
print("Degrees to radians (180°):", math.radians(180))
print("Radians to degrees (π radians):", math.degrees(math.pi))

# 🔹 9. Special Functions
print("\nSpecial Functions:")
print("Exponential of 2:", math.exp(2))             # e^2
print("Log1p(0.5) (ln(1+0.5)):", math.log1p(0.5))   # more precise for small values

# 🔹 10. Notes
# 1. math functions mostly return float values.
# 2. Some functions like factorial(), gcd() only accept integers.
# 3. Trigonometric functions use radians, not degrees.
# 4. Use math.radians() and math.degrees() to convert between degrees and radians.
