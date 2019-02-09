from configparser import ConfigParser, ExtendedInterpolation
import numpy as np

parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read('test.ini')
sections = parser.sections()

items = parser.items(sections[0])

parameters = {}
for i in range(len(items)):
    try:
        parameters[items[i][0]] = eval(items[i][1])
    except:
        parameters[items[i][0]] = (items[i][1])