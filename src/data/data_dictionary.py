from typing import List
import pandas as pd
import re
import io

def remove_headers(df_info: str) -> str:
    """Removes the header and footer section of the info and strip the white space
    at the start and end of the string

    Parameters:
        `df_info (str)`: information returned from `pandas.Dataframe.info()`
    Returns:
        `df_info (str)`: the info stripped of the header and footer
    """
    df_info = df_info.rsplit("----", 1)[1].split("dtypes")[0].strip()
    return df_info

def create_data_dictionary(df: pd.DataFrame, filename: str = "output.txt", header: List[str] = [], as_strings: bool = False, examples=True) -> None:
    """Creates a data dictionary using information from the dataframe info object

    Parameters:
        `df (pandas.DataFrame)`: a dataframe to extract the information object
        `header (List[str])`: a list of headers used in the output definition defaults to `["Index", "FieldName", "NonNullCount", "NonNull", "DataType"]`
        `filename (str)`: the file name (path) for the output. defaults to `"output.txt"`
        `as_strings (bool)`: determines if `object` types should be outputed as `strings`. defaults to `False`
        `examples (bool)`: determines if exmaples should be included in the output. defaults to `True`
    Returns:
        An output file containing the data dictionary
    """
    if(len(header) == 0):
        header = ["Index", "FieldName", "NonNullCount", "NonNull", "DataType"]
    
    # get the info as a string into a buffer
    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()
    info = remove_headers(info)

    # get all the lines in the text as a list
    lines = [re.sub("\s+", ",", l.strip()) for l in info.split("\n")]

    # add examples to the output
    if(examples):
        header.append("Example")
        examples_list = df.iloc[0].values.tolist()
        lines = [line + f",{str(examples_list[i])}" for i, line in enumerate(lines)]
    
    # write lines to file
    textfile = open(filename, mode="w")
    textfile.write(",".join(header) + "\n")
    for line in lines:
        line = re.sub("object", "string", line) if as_strings else line
        textfile.write(line + "\n")
    # close the file buffer
    textfile.close()

def main(df: pd.DataFrame, header: List[str] = [], filename: str = "output.txt", as_strings: bool = False, examples=True):
    create_data_dictionary(df=df, header=header, filename=filename, as_strings=as_strings, examples=examples)

if __name__ == "__main__":
    dfs_locations = ["complaints_inspections_registry.csv", "complaints_registry.csv", "complaints_sanctions_registry.csv", "sanctions_inspections_registry.csv", "sanctions_registry.csv"]
    loc = "../../data/processed/db/"
    dfs = [pd.read_csv(loc+f) for f in dfs_locations]
    for idx, df in enumerate(dfs):
        filename = loc + "data_dictionary/" + re.sub("\.csv", ".txt", dfs_locations[idx])
        main(df=df, filename=filename, as_strings=True)
