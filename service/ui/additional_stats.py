import uuid
import pandas as pd
import pydqc
import os
import streamlit as st
from pydqc.data_compare import data_compare

OUTPUT_DIR = "/tmp/"

def map_type(original_type):
    if original_type in ["int64", "float64"]:
        return "numeric"
    else:
        return "str"

def getSchema(df):
    _schema = df.dtypes.to_dict()
    _col_names = list(_schema.keys())
    _col_types = list(_schema.values())
    _col_include = [1 for _ in range(len(_col_names))]
    schema_df = pd.DataFrame(dict(column=_col_names, type=_col_types, include=_col_include))
    schema_df['type'] = schema_df['type'].map(lambda t: map_type(t))
    return schema_df

def get_full_path_2_xlsx(_filename_part):
    return f"{OUTPUT_DIR}/data_compare_{_filename_part}.xlsx"

def process_2_xlsx(original, processed, name = ""):
    _filename_part = f"{name}_{str(uuid.uuid4())}"
    _original_schema = getSchema(original)
    _processed_schema = getSchema(processed)
    pydqc.data_compare.data_compare(original, processed, _original_schema, _processed_schema, _filename_part,
                                    output_root=OUTPUT_DIR)
    return get_full_path_2_xlsx(_filename_part)

def get_xlsx_markdown(original, processed):
    import base64
    _xlsx_file = process_2_xlsx(original, processed)
    with open(_xlsx_file, mode='rb') as file:
        _file_content = file.read()
    b64 = base64.b64encode(_file_content).decode()
    href = f'<a href="data:file/xlsx;base64,{b64}" download="compare.xlsx">Download XLS File</a> (right-click and save as &lt;some_name&gt;.xlsx)'
    os.remove(_xlsx_file)
    return href


