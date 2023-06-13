from io import TextIOWrapper
import yaml
import csv


class RefractiveIndex:
    def __init__(self, data: dict[float,dict[str,float]]):
        self._data = data

    def __getitem__(self, wavelength:float):
        wavelengths = self._data.keys()
        below = max([w for w in wavelengths if w <= wavelength])
        above = min([w for w in wavelengths if w > wavelength])
        interp_coeff = (wavelength-below) / (above - below)
        return {
            "n": self._data[below]["n"] + interp_coeff * (self._data[above]["n"] - self._data[below]["n"]), 
            "k": self._data[below]["k"] + interp_coeff * (self._data[above]["k"] - self._data[below]["k"])
        }
    
class YAMLRefractiveIndex(RefractiveIndex):
    def __init__(self, data_file: TextIOWrapper):
        data_wrapper = yaml.load(data_file.read())
        data = csv.reader(data_wrapper["DATA"][0]["data"].split("\n"), delimiter=" ")
        prepared_data = {}
        for data_point in data:
                if not len(data_point) == 0:
                    prepared_data[float(data_point[0])] = {"n": float(data_point[1]), "k": float(data_point[2])}

        super().__init__(prepared_data)

class CSVRefractiveIndex(RefractiveIndex):
    def __init__(self, data_file: TextIOWrapper):
        csv_data = csv.DictReader(data_file.readlines(), delimiter="\t")
        prepared_data = {}
        for line in csv_data:
            prepared_data[float(line["Wavelength(nm)"])/1e9] = {
                "n": float(line["n"]),
                "k": float(line["k"])
            }

        super().__init__(prepared_data)