import csv
import ast

reader = csv.reader(open('values.csv', 'r'))
d = {}
row = next(reader)
print(row)
#['id', 'R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']

AD = 1
NC = 0

sum_equal = 0
sum_equal_mult = 0
sum_equal_avg = 0

lists = {}

for R in range(10):
	for K in range(5):
		key = "R" + str(R) + "K" + str(K)
		lists[key] = []

for row in reader:
	for R in range(10):
		key = "R" + str(R) + "K" + str(row[R + 2])
		lists[key].append(row[0] + "_" + row[1] + ".png")

print("lists= { ")
for k in lists.keys():
	print
	print("'" + k + "' : " + str(lists[k]) + ",")
print("}")