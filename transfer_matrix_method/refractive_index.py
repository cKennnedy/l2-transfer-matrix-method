from io import TextIOWrapper
import yaml
import csv


class RefractiveIndex:
    def __init__(self, data_file: TextIOWrapper):
        data_wrapper = yaml.load(data_file.read(), Loader=yaml.FullLoader)
        data = csv.reader(data_wrapper["DATA"][0]["data"].split("\n"), delimiter=" ")
        self._data = {}
        for data_point in data:
            if not len(data_point) == 0:
                self._data[float(data_point[0])] = {"n": float(data_point[1]), "k": float(data_point[2])}

    def __getitem__(self, wavelength:float):
        wavelengths = self._data.keys()
        below = max([w for w in wavelengths if w <= wavelength])
        above = min([w for w in wavelengths if w > wavelength])
        interp_coeff = (wavelength-below) / (above - below)
        lininterp = lambda value: self._data[below][value] + interp_coeff * (self._data[above][value] - self._data[below]["k"])
        return {
            "n": lininterp("n"), 
            "k": lininterp("k")
        }

