import pathlib

directory = pathlib.Path(__file__).parent/"data"

def get_filepath(material: str) -> str:
    path = directory/f"{material}.txt"
    if not path.is_file():
        raise ValueError("Material not found in data library")
    
    return path