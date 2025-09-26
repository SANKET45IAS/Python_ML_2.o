# ---------------------------------------------------------
# ✅ Python Statistics Module Example (Detailed)
# ---------------------------------------------------------

# 1. Import the statistics module
import statistics

# Sample data
data = [10, 15, 20, 15, 25, 30, 15]

print("Sample Data:", data)

# 🔹 2. Measures of Central Tendency
print("\n--- Central Tendency ---")
mean_value = statistics.mean(data)          # Average
print("Mean:", mean_value)

median_value = statistics.median(data)      # Middle value
print("Median:", median_value)

mode_value = statistics.mode(data)          # Most frequent value
print("Mode:", mode_value)

# Harmonic mean
harmonic_mean_value = statistics.harmonic_mean(data)
print("Harmonic Mean:", harmonic_mean_value)

# Geometric mean (Python 3.8+)
try:
    geometric_mean_value = statistics.geometric_mean(data)
    print("Geometric Mean:", geometric_mean_value)
except AttributeError:
    print("Geometric Mean requires Python 3.8+")

# 🔹 3. Measures of Spread / Variability
print("\n--- Variability ---")
variance_value = statistics.variance(data)       # Sample variance
print("Variance:", variance_value)

stdev_value = statistics.stdev(data)             # Sample standard deviation
print("Standard Deviation:", stdev_value)

# Population variance / std dev
pvariance_value = statistics.pvariance(data)     # Population variance
pstdev_value = statistics.pstdev(data)           # Population standard deviation
print("Population Variance:", pvariance_value)
print("Population Standard Deviation:", pstdev_value)

# 🔹 4. Other Useful Functions
print("\n--- Other Useful Functions ---")
median_low = statistics.median_low(data)   # Lower median for even-sized data
median_high = statistics.median_high(data) # Upper median for even-sized data
print("Median Low:", median_low)
print("Median High:", median_high)

# 🔹 5. Notes and Practical Uses
# 1. statistics.mean() → Average of data
# 2. statistics.median() → Middle value
# 3. statistics.mode() → Most frequent value
# 4. statistics.variance() → Sample variance
# 5. statistics.stdev() → Sample standard deviation
# 6. Use population functions for full population data (pvariance, pstdev)
# 7. Useful for **data analysis, probability, and scientific calculations**

# 🔹 6. Example Use Case: Exam Scores
scores = [85, 90, 78, 92, 88, 90, 85]
print("\nExam Scores:", scores)
print("Average Score:", statistics.mean(scores))
print("Highest Frequency Score:", statistics.mode(scores))
print("Score Variability (Std Dev):", statistics.stdev(scores))
