import csv
import sys
import pandas as pd


# Horizontal Flip
# x = 1-x   y = y
Two = pd.read_csv("./Label/001.csv")
type(Two)                      # pandas.core.frame.DataFrame
Two.dtypes                     # DataFrame中包含的每个对象都被看成Numpy对象
#print(Two.columns.tolist())    # 得到所有列名
Two.values.tolist()       

Two["x1"] = 1- Two["x1"]
Two["x2"] = 1- Two["x2"]
Two["x3"] = 1- Two["x3"]

Two["x1"].replace(1, 0, inplace=True)
Two["x2"].replace(1, 0, inplace=True)
Two["x3"].replace(1, 0, inplace=True)

Two.to_csv("./Label/002.csv")

# Vertical Flip
# x = x   y = 1-y
Three = pd.read_csv("./Label/001.csv")
type(Three)                      # pandas.core.frame.DataFrame
Three.dtypes                     # DataFrame中包含的每个对象都被看成Numpy对象
Three.values.tolist()       

Three["y1"] = 1- Three["y1"]
Three["y2"] = 1- Three["y2"]
Three["y3"] = 1- Three["y3"]

Three["y1"].replace(1, 0, inplace=True)
Three["y2"].replace(1, 0, inplace=True)
Three["y3"].replace(1, 0, inplace=True)

Three .to_csv("./Label/003.csv")

# Rotate 90
# x = y   y = 1-x
Four = pd.read_csv("./Label/001.csv")
O = pd.read_csv("./Label/001.csv")
type(Four)                      # pandas.core.frame.DataFrame
Four.dtypes                     # DataFrame中包含的每个对象都被看成Numpy对象
Four.values.tolist()   

Four["x1"] = O["y1"]
Four["x2"] = O["y2"]
Four["x3"] = O["y3"]
Four["y1"] = 1 - O["x1"]
Four["y2"] = 1 - O["x2"]
Four["y3"] = 1 - O["x3"]

Four["x1"].replace(1, 0, inplace=True)
Four["x2"].replace(1, 0, inplace=True)
Four["x3"].replace(1, 0, inplace=True)
Four["y1"].replace(1, 0, inplace=True)
Four["y2"].replace(1, 0, inplace=True)
Four["y3"].replace(1, 0, inplace=True)

Four.to_csv("./Label/004.csv")

# Rotate 270
# x = 1 - y y = x
Five = pd.read_csv("./Label/001.csv")
O = pd.read_csv("./Label/001.csv")
type(Five)                      # pandas.core.frame.DataFrame
Five.dtypes                     # DataFrame中包含的每个对象都被看成Numpy对象
Five.values.tolist()   

Five["x1"] = 1 - O["y1"]
Five["x2"] = 1 - O["y2"]
Five["x3"] = 1 - O["y3"]
Five["y1"] = O["x1"]
Five["y2"] = O["x2"]
Five["y3"] = O["x3"]

Five["x1"].replace(1, 0, inplace=True)
Five["x2"].replace(1, 0, inplace=True)
Five["x3"].replace(1, 0, inplace=True)
Five["y1"].replace(1, 0, inplace=True)
Five["y2"].replace(1, 0, inplace=True)
Five["y3"].replace(1, 0, inplace=True)

Five.to_csv("./Label/005.csv")

# Color
# x = x y = y
Six = pd.read_csv("./Label/001.csv")

Six.to_csv("./Label/006.csv")
