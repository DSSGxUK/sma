import numpy as np
import pandas as pd
import argparse
from functools import reduce
from unidecode import unidecode
from typing import List
import glob
from tqdm import tqdm
from pathlib import Path

MERGE_KEYS = {
    "complaints_registry": "ComplaintId",
    "facilities_inspection": "FacilityId",
    "facilities_sanction": "FacilityId",
    "facilities_registry": "FacilityId",
    "sanctions_registry": "SanctionId",
    "inspections_registry": "InspectionId"
}

def character_replacement(text: str):
    text = str(text)
    return unidecode(f"{text}", "utf-8")

def normalize(df: pd.DataFrame, columns: list = []):
    for column in columns:
        df[column] = df[column].astype(str).apply(character_replacement)
    return df.replace('nan', np.nan)

def import_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def merge_dfs(dfs: List[pd.DataFrame], keys: List[str], method: str = "left") -> pd.DataFrame:
    if(len(dfs) == 2):
        left, right = dfs
        df = left.merge(right, on=keys[0], how=method)
        return df
    left, middle, right = dfs
    df = left.merge(middle, on=keys[0], how=method).merge(right, on=keys[1], how=method)
    #df = reduce(lambda left,right: pd.merge(left,right, on=key, how=method), dfs)
    return df

def export_df(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False, header=True)

def add_binary_column(EndType: str):
    if(EndType == "Derivacion Total a Organismo Competente" or EndType == "Archivo I"):
        return "Irrelevant"
    elif (EndType == "Formulacion de Cargos" or EndType == "Archivo II"):
        return "Relevant"
    else: 
        return np.nan

def merge_etl_data(data_folder_path, merge_keys = MERGE_KEYS, include_territories=False):
    all_csv_files = glob.glob(f"{data_folder_path}/*.csv")    
    all_files = {}

    for file_path in tqdm(all_csv_files):
        # @HACK - Excel only allows a max of 31 characters to be used as sheet names
        file_name = Path(file_path).stem[:31]
        all_files[file_name] = pd.read_csv(file_path).reset_index(drop = True)
        # pd.read_csv(file_path, index_col=0,  error_bad_lines=False)

    complaint_files = [all_files[file_key] for file_key in [
        'Resumen_Denuncia',
        'Detalle_DenunciaHecho',
        'Detalle_DenunciaMateriaAmbienta',
        'Detalle_DenunciaColumnasExtraI',
        'Detalle_DenunciaPoblacionAfecta',
        'Detallle_DenunciaImpactoSalud',
        'Detalle_DenunciaEfectoMedioAmbi',
        'Detalle_DenunciaGeorreferencia'
        ]]
    
    sanction_files = [all_files[file_key] for file_key in [
        'Resumen_ProcesoSancion',
        'Detalle_ProcesoSancionHechoInst'
        ]]
    inspection_files = [all_files[file_key] for file_key in ['ReporteProcesoFiscalizacion', 'Detalle_ProcesoFiscalizacionUni']]
    facility_files = [all_files[file_key] for file_key in ['Resumen_UnidadFiscalizable']]

    complaints_registry = reduce(lambda left, right: pd.merge(left, right, how="left", on=merge_keys["complaints_registry"]), complaint_files)
    sanctions_registry = reduce(lambda left, right: pd.merge(left, right, how="left", on=merge_keys["sanctions_registry"]), sanction_files)
    inspections_registry = reduce(lambda left, right: pd.merge(left, right, how="left", on=merge_keys["inspections_registry"]), inspection_files)
    if(len(facility_files) == 1):
        facilities_registry = facility_files[0]
    else:
        facilities_registry = reduce(lambda left, right: pd.merge(left, right, how="left", on=merge_keys["facilities_registry"]), facility_files)
    complaints_facilities_registry = complaints_registry.merge(all_files["Detalle_DenunciaUnidadFiscaliza"], on = "ComplaintId").merge(facilities_registry, on="FacilityId", how="left")
    complaints_inspections_registry = complaints_registry.merge(all_files["Detalle_DenunciaProcesoFiscaliz"], on = "ComplaintId").merge(inspections_registry, how="left", on="InspectionId")
    complaints_sanctions_registry = complaints_registry.merge(all_files["Detalle_DenunciaProcesoSancion"], on = "ComplaintId").merge(sanctions_registry, how="left", on="SanctionId")
    #inspections_sanctions_registry = inspections_registry.merge(all_files["Detalle_ProcesoSancionProcesoFi"], on = "InspectionId").merge(sanctions_registry, how="left", on="SanctionId")
    #facility_sanctions = facilities_registry.merge(all_files["Detalle_ProcesoSancionUnidadFis"], on="FacilityId").merge(sanctions_registry, how="left", on="SanctionId")
    facilities_sanctions_id = facilities_registry.merge(all_files["Detalle_ProcesoSancionUnidadFis"], on="FacilityId")

    if(include_territories):
        complaints_facilities_registry = complaints_facilities_registry.merge(all_files["Variables_territoriales"], how="left", left_on="FacilityDistrict", right_on="District")
    
    ## write csv files to processed:
    # complaints_registry = complaints_registry.drop_duplicates(subset=['ComplaintId'])
    complaints_registry.to_csv("/home/<USERNAME>/data/processed/complaints_registry.csv", index = True)
    complaints_facilities_registry.to_csv("/home/<USERNAME>/data/processed/complaints_facilities_registry.csv", index = True)
    facilities_sanctions_id.to_csv("/home/<USERNAME>/data/processed/facilities_sanction_id.csv", index = True)
    sanctions_registry.to_csv("/home/<USERNAME>/data/processed/sanctions_registry.csv", index = True)
    complaints_sanctions_registry.to_csv("/home/<USERNAME>/data/processed/complaints_sanctions_registry.csv", index = True)

    # return {
    #     "complaints_registry": complaints_registry.drop_duplicates(subset=['ComplaintId']),
    #     "sanctions_registry": sanctions_registry,
    #     "inspections_registry": inspections_registry,
    #     "facilities_registry": facilities_registry,
    #     "facilities_sanction_id": facilities_sanctions_id,
    #     "complaints_facilities_registry": complaints_facilities_registry,
    #     "complaints_inspections_registry": complaints_inspections_registry,
    #     "complaints_sanctions_registry": complaints_sanctions_registry
    #     # "inspections_sanctions_registry": inspections_sanctions_registry,
    #     # "facilities_sanctions": facility_sanctions
    # }

def main(input_paths: List[str], output_path: str, merge_keys: str, col_names=[]):
    dfs = [import_df(filename) for filename in input_paths]
    final_df = merge_dfs(dfs, keys=merge_keys, method="left")
    final_df = normalize(final_df, columns=col_names)
    export_df(final_df, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input paths for the csv files")
    parser.add_argument("--output", help="output path for processed data")
    parser.add_argument("--keys", help="the keys used in merging the data files")
    parser.add_argument("--normalize", help="columns where the data should be normalized to ascii")

    args = parser.parse_args()
    input_paths = args.input.split()
    output_path = args.output if args.output is not None else "../data/processed/db/output.csv"
    merge_keys = args.keys.split()

    main(input_paths = input_paths, output_path=output_path, merge_keys=merge_keys, col_names=args.normalize.split())
