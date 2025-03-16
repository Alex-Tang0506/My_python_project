import os
import csv
title = ['ISBN']
a = ['9781604134339', '160413433X']
b = ['034080095X', '9780340800959']

current_path = os.getcwd()
print(current_path)
f = open(current_path + "\\test_1.csv", mode="w", encoding='utf-8-sig', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(title)
csv_writer.writerow(a)
csv_writer.writerow(b)
f.close()