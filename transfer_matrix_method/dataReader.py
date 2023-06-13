import csv

class dataReader():
    current_dict = {}

    def create_dict(self,filename):
        file = csv.reader(filename, delimiter='\t')
        print(file)

    def get_dict(self):
        return self.current_dict

data = dataReader()

data.create_dict("data\Co.txt")
