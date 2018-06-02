# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:19:13 2018
This is a program to load CSV file
@author: 15065
"""
import csv
csvFile = open('C:/Users/15065/Downloads/T10Y2Y.csv','r')
reader = csv.reader(csvFile)
#用字典存储数据
result = dict()
for item in reader:
    if reader.line_num == 1:
        continue
    result[item[0]] = item[1]
    
csvFile.close()
result_value = list(result.values())
n = len(result_value)
##将缺失值转化为相邻时间的值
for i in range(n):
    if result_value[i] == '.':
        result_value[i] = result_value[i-1]
        

results = [float(item) for item in result_value]

print(len(results))

