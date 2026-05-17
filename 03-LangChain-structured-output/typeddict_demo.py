"""
Demonstration of TypedDict and Inheritance in Python.
"""
from typing import TypedDict

# 1. INHERITANCE
# Inheritance is an Object-Oriented concept where one class gets all the features of another.
# Here, 'Person' is inheriting from 'TypedDict'. 
# This means 'Person' IS a dictionary, but it has strict rules added to it.
class Person(TypedDict):
    # 2. TYPE HINTS
    # We are declaring the blueprint. Every 'Person' dictionary MUST have:
    # A key 'name' with a string value.
    # A key 'age' with an integer value.
    name: str
    age: int

# 3. USAGE
# Here we create a dictionary and specify its type is 'Person'.
# If you run this through a type checker (like mypy), the IDE will throw an 
# error because '35' is a string, but the schema demands an integer.
new_person: Person = {'name': 'nitish', 'age': 35}

print("Created a TypedDict successfully:")
print(f"Name: {new_person['name']}")
print(f"Age: {new_person['age']}")