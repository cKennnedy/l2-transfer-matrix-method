import csv
import pathlib

filepathv = pathlib.Path(__file__).parent

class dataReader():
    current_dict = {}

    def create_dict(self,filename):
        with open(filename, encoding="utf-8-sig") as f:
            file = csv.reader(f.readlines(), delimiter='\t')
        for line in file:
            print(line)
        

    def get_dict(self):
        return self.current_dict

data = dataReader()

data.create_dict(filepathv/"data"/"Co.txt")
