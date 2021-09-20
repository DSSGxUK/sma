import os
import io
import base64
import streamlit as st
import pandas as pd
from data_preprocess import (import_df, merge_dfs, normalize)

uploaded_files = {}
if('merged_files' not in st.session_state):
    st.session_state.current_file = {}
def set_page_metadata():
    st.title("Data merging tool")
    
def data_merging_dashboard():
    data_files = st.file_uploader("Choose csv file(s)", accept_multiple_files=True)
    if(data_files):
        for data_file in data_files:
            file_name = data_file.name.rsplit(".", 1)[0]
            file_extension = data_file.name.rsplit(".", 1)[1]
            if(file_extension == "csv"):
                uploaded_files[file_name] = import_df(data_file)
            elif(file_extension == "xls" or file_extension == "xlsx"):
                sheets = {}
                xls = pd.ExcelFile(data_file)
                if(len(xls.sheet_names) > 1):
                    for _sheet in xls.sheet_names:
                        sheets[f'{_sheet}'] = pd.read_excel(xls, sheet_name=_sheet)
                    uploaded_files.update(sheets)
                else:
                    uploaded_files[file_name] = pd.read_excel(xls)
            else:
                print('')

def merge_files_selector():
    if (len(uploaded_files.keys()) > 0):
        st.write("Select two files or sheets to merge")
        file1 = st.selectbox("Merge", list(uploaded_files.keys()))
        file2 = st.selectbox("With", list(uploaded_files.keys()))

        merge_col_intersect = [value for value in list(uploaded_files[file1].columns) if value in list(uploaded_files[file2].columns)]
        merge_on = st.multiselect("On", merge_col_intersect)

        merge_method = st.selectbox("Merge method", ["left", "right", "outer", "inner", "cross"])
        initiate_merge = st.button("Merge")

        if(initiate_merge):
            data_frames = [uploaded_files[file1], uploaded_files[file2]]
            merged_data = merge_dfs(dfs=data_frames, keys=[merge_on], method=merge_method)
            st.session_state.current_file["last_merge"] = merged_data
            uploaded_files.update(st.session_state.current_file)

            normalize_cols = st.multiselect("Normalize columns", list(merged_data.columns))
            # merged_data_norm = normalize(merged_data, normalize_cols)
            output_file_name = f"{file1}_x_{file2}.csv"
            st.text_input("File path", value='/files/') + output_file_name
            merged_data.to_csv(f"/files/data_merge/{output_file_name}", encoding="utf-8", index=False)

            # towrite = io.BytesIO()
            # downloaded_file = merged_data_norm.to_csv(towrite, encoding='utf-8', index=False, header=True)
            # towrite.seek(0)
            # base64_encoded_data = base64.b64encode(towrite.read()).decode()
            # download_link = f'<a href="data:file/csv;base64,{base64_encoded_data}" download="output.csv">Download csv file</a>'
            # st.markdown(download_link, unsafe_allow_html=True)
def main():
    set_page_metadata()
    data_merging_dashboard()
    merge_files_selector()
if __name__ == "__main__":
    main()