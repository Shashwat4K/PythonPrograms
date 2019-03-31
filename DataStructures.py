import collections
import operator
import os
import heapq
import sys

# Unpacking from iterables of arbitrary length
print('Unpacking from iterables of arbitrary length')
data = [1,2.5,True, "Shashwat", "Kadam", "CSE"]
data2 = ["String1", "String2", "String3", 2, 5, False]
number1, number2, number3, *strings = data

for i in (number1, number2, number3, *strings):
    print("Type: {} Value: {}".format(type(i), i))

print("----------------------------------")
*strs, num1, num2, num3 = data2
for i in (*strs, num1, num2, num3):
    print("Type: {} Value: {}".format(type(i), i))   

# You can also unpack in loop, e.g. for x, *y in data:

# Largest and Smallest N values
print("----------------------------------")
print('Largest and Smallest N values\n')
nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]

print('Top 3 numbers', end=" ")
print(heapq.nlargest(3,nums))
print()
print('Botom 4 numbers', end=" ")
print(heapq.nsmallest(4, nums))

rows = [
    {"name": "Shashwat", "age": 21, "dept": "CSE"},
    {"name": "Suyash", "age": 20, "dept": "CIV"},
    {"name": "Om", "age": 22, "dept": "MIN"},
    {"name": "Rahul", "age": 22, "dept": "MEC"},
    {"name": "Rajat", "age": 21, "dept": "MME"},
    {"name": "Gyan", "age": 22, "dept": "CME"},
    {"name": "Swapnil", "age": 21, "dept": "EEE"},
]

print('Persons with largest lexicographical names')
for i in heapq.nlargest(len(rows), rows, key=lambda x: x['name']):
    print("{}->".format(i['name']), end=" ")
print("/")   
print('Before Heapify:') 
print(list(nums))
print('After Heapify:')
heap = list(nums)
heapq.heapify(heap)
print(heap)
while heap.__len__() > 0:
    print("{} -> ".format((heapq.heappop(heap))), end=" ")
print()    
print("----------------------------------")

# Implementing a Priority Queue

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item)) # DOing a negation because higher the number, higher the priority
        # Introducing extra _index value because to sidtinguish between elements with same priority
        # Compared according to priority first, if the priorities are same then one who came first is smaller than others
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

# Playing with Dictionaries
print("----------------------------------")
print('defaultdict() OrderedDict()')

d = collections.defaultdict(list)
d['fname'].extend(['Shashwat', 'Suyash', 'Rajat', 'Om'])
d['lname'].extend(['Kadam', 'Nikalje', 'Sorde', 'Thakare'])

for f,l in zip(d['fname'], d['lname']):
    print('{} {}'.format(f, l))

d2 = collections.OrderedDict()

d2['x'] = 7541513
d2['z'] = -5121
d2['a'] = 4631.55
d2['b'] = True and False

print('OrderedDict()->', d2)
print("----------------------------------")
