from cv2.ximgproc import *
import cv2
import pdb
print(callable(guidedFilter))

img = cv2.imread('disp3.pgm')

# img3 = img[:,:,::-1]

# print(img3.shape)

# cv2.imshow('img3', img3)
# cv2.imshow('img', img)
# cv2.waitKey(0)

# guide = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# dst1 = cv2.ximgproc.guidedFilter(guide=guide, src=img, radius=16, eps=50, dDepth=-1)

# dst1 = cv2.ximgproc.guidedFilter(guide=guide, src=guide, radius=16, eps=50, dDepth=-1)

# cv2.imshow('guided', dst1)
# cv2.waitKey(0)

pdb.set_trace()
# import requests
# from bs4 import BeautifulSoup
# from difflib import SequenceMatcher
# from selenium import webdriver
# import time
# import pinyin

# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# driver = webdriver.Chrome()
# driver.get("http://gra108.aca.ntu.edu.tw/regchk/stu_query.asp")
# driver.find_element_by_xpath("//select[@name='DEP']").click()

# time.sleep(1)
# driver.find_element_by_xpath("//option[text()='9010電機工程學系碩士班甲組(主修自動控制)']").click()
# driver.find_element_by_xpath("//input[@name='qry']").click()

# time.sleep(1)
# status = driver.find_element_by_xpath('//table[1]/tbody').find_elements_by_tag_name('tr')
# _not = 0
# _alr = 0
# _alrbn = 0
# _giu = 0

# a = pinyin.get("未報到", format = 'numerical')
# b = pinyin.get("志願1", format = 'numerical')
# c = pinyin.get("放棄", format = 'numerical')
# d = pinyin.get("志願2", format = 'numerical')
# e = pinyin.get("志願3", format = 'numerical')

# care = 53

# index = 0

# for row in status:

# 	if index is care:

# 		print("已放棄:", _giu)
# 		print("未報到:", _not)
# 		print("志願2:", _alrbn)
# 		print("志願1:", _alr)

# 	cells = row.find_elements_by_tag_name('td')
# 	_cells = pinyin.get(str(cells[4].text), format = 'numerical')


# 	if similar(_cells, a) > 0.9:
# 		_not += 1
# 	if similar(_cells, b) > 0.99:
# 		_alr += 1
# 	if similar(_cells, c) > 0.6:
# 		_giu += 1
# 	if similar(_cells, d) > 0.99:
# 		_alrbn += 1

# 	if similar(_cells, e) > 0.99:
# 		_alrbn += 1



# 	index += 1

# driver.quit()

 


