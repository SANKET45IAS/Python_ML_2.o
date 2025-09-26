# -----------------------------------------
# 1. Basic Class and Object
# -----------------------------------------
class Student:
    # Class variable (shared by all objects)
    school_name = "SOA University"

    # Constructor -> called when object created
    def __init__(self, name, age):
        self.name = name        # instance variable
        self.age = age

    # Instance method
    def display_info(self):
        print(f"Name: {self.name}, Age: {self.age}, School: {Student.school_name}")

    # Class method -> works on class level
    @classmethod
    def change_school(cls, new_name):
        cls.school_name = new_name

    # Static method -> general utility (not tied to object/class variables)
    @staticmethod
    def welcome():
        print("Welcome to the Student Portal!")


# Creating objects
s1 = Student("Sanket", 20)
s2 = Student("Aayush", 19)

# Accessing instance & class methods
Student.welcome()
s1.display_info()
s2.display_info()

# Changing class variable using classmethod
Student.change_school("ITER Bhubaneswar")
s1.display_info()
s2.display_info()


# -----------------------------------------
# 2. Encapsulation (Public, Protected, Private)
# -----------------------------------------
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner            # public variable
        self._account_type = "Saving" # protected variable (convention only)
        self.__balance = balance      # private variable (name mangling)

    def deposit(self, amount):
        self.__balance += amount

    def get_balance(self):
        return self.__balance

acc = BankAccount("Sanket", 5000)
print("\nAccount Owner (public):", acc.owner)
print("Account Type (protected):", acc._account_type)
# print(acc.__balance) ❌ Not accessible (private)
print("Account Balance (via method):", acc.get_balance())


# -----------------------------------------
# 3. Inheritance
# -----------------------------------------
# Base class
class Person:
    def __init__(self, name):
        self.name = name

    def show(self):
        print("Person:", self.name)

# Single Inheritance
class Employee(Person):
    def __init__(self, name, emp_id):
        super().__init__(name)
        self.emp_id = emp_id

    def show(self):
        print(f"Employee: {self.name}, ID: {self.emp_id}")

# Multiple Inheritance
class Coder:
    def code(self):
        print(f"{self.name} is coding...")

class Programmer(Employee, Coder):
    def __init__(self, name, emp_id, language):
        super().__init__(name, emp_id)
        self.language = language

    def show(self):
        print(f"Programmer: {self.name}, ID: {self.emp_id}, Language: {self.language}")

# Creating objects
p1 = Person("Raj")
e1 = Employee("Sanket", 101)
pr1 = Programmer("Aayush", 102, "Python")

print()
p1.show()
e1.show()
pr1.show()
pr1.code()


# -----------------------------------------
# 4. Polymorphism
# -----------------------------------------
# (a) Method overriding
class Animal:
    def sound(self):
        print("Animals make sound")

class Dog(Animal):
    def sound(self):
        print("Dog barks")

class Cat(Animal):
    def sound(self):
        print("Cat meows")

print()
for animal in [Dog(), Cat()]:
    animal.sound()  # same method, different behavior

# (b) Operator Overloading
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):  # Overloading +
        return Vector(self.x + other.x, self.y + other.y)

    def __str__(self):  # Overloading print()
        return f"Vector({self.x}, {self.y})"

v1 = Vector(2, 3)
v2 = Vector(4, 5)
v3 = v1 + v2
print("\nVector Addition:", v3)


# -----------------------------------------
# 5. Special Methods
# -----------------------------------------
class MyList:
    def __init__(self, items):
        self.items = items

    def __len__(self):  # Overloading len()
        return len(self.items)

    def __str__(self):
        return f"MyList: {self.items}"

ml = MyList([1, 2, 3, 4, 5])
print("\nLength of MyList:", len(ml))
print("Printing MyList object:", ml)


# ✅ Example: Class, Object, and use of 'self'

class Car:
    # Constructor (called automatically when object is created)
    def __init__(self, brand, model):
        # 'self.brand' and 'self.model' are **object attributes**
        # They belong to the specific object (car1, car2, etc.)
        self.brand = brand
        self.model = model
    
    # Method to display info about the car
    def display_info(self):
        # 'self' lets us access object-specific attributes
        print("Brand:", self.brand)
        print("Model:", self.model)

    # Method to show difference between self.attribute and local variable
    def compare_self_and_local(self):
        year = 2025   # Local variable (accessible only inside this method)
        self.year = 2024  # Object attribute (belongs to the object)
        
        print("Local variable 'year':", year)           # Works only here
        print("Object attribute 'self.year':", self.year)  # Accessible outside too

# ✅ Create objects (Python automatically passes the object as 'self')
car1 = Car("Tesla", "Model S")
car2 = Car("Toyota", "Corolla")

# Call methods
print("Car 1 Information:")
car1.display_info()   # self refers to car1 here
print("\nCar 2 Information:")
car2.display_info()   # self refers to car2 here

print("\nComparing 'self' vs Local Variable (for car1):")
car1.compare_self_and_local()

# Access object attribute outside the class
print("Accessing car1.year outside the class:", car1.year)

# ❌ Local variable 'year' cannot be accessed outside method
# print(car1.year_local)   # ERROR: AttributeError
# Reason: 'year' was a local variable, not tied to 'self'

