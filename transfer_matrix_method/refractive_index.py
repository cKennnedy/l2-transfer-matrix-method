from io import TextIOWrapper
import yaml
import csv


class RefractiveIndex:
    def __init__(self, data_file: TextIOWrapper):
        data_wrapper = yaml.load(data_file.read())
        data = csv.reader(data_wrapper["DATA"][0]["data"].split("\n"), delimiter=" ")
        self._data = {}
        for data_point in data:
            if not len(data_point) == 0:
                self._data[data_point[0]] = {"n": data_point[1], "k": data_point[2]}

