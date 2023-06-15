from .get_material_data import get_filepath

from io import TextIOWrapper
import yaml
import csv


class RefractiveIndex:
    def __init__(self, data: dict[float,dict[str,float]]):
        self._data = data

    def __getitem__(self, wavelength:float):
        wavelengths = self._data.keys()
        if wavelength > max(wavelengths) or wavelength < min(wavelengths):
            raise KeyError("Index out of Data Range")
        below = max([w for w in wavelengths if w <= wavelength])
        above = min([w for w in wavelengths if w > wavelength])
        interp_coeff = (wavelength-below) / (above - below)
        lininterp = lambda value: self._data[below][value] + interp_coeff * (self._data[above][value] - self._data[below][value])
        return lininterp("n") + lininterp("k")*1j
    
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

class MaterialRefractiveIndex(CSVRefractiveIndex):
    def __init__(self, material_name: str):
        with open(get_filepath(material=material_name), encoding="utf-8-sig") as f:
            super().__init__(f)