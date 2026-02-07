import pandas as pd
import numpy as np
import os
from .preprocess import detect_center_shift, apply_center_shift

def read_and_clean_data(file_path, head_skip=0, data_skip=0):
    """
    Read and clean EPM trial data from Excel file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
        
    try:
        df_preview = pd.read_excel(file_path, header=None, nrows=50)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
        
    header_row_idx = None
    target_cols = ["Trial time", "X center", "Y center"]
    
    for idx, row in df_preview.iterrows():
        row_vals = [str(x).strip() for x in row.values]
        if all(col in row_vals for col in target_cols):
            header_row_idx = idx
            break
            
    if header_row_idx is None:
        header_row_idx = head_skip

    offset = max(0, data_skip - head_skip)
    
    df = pd.read_excel(file_path, header=header_row_idx)
    
    if offset > 0:
        df = df.iloc[offset:].reset_index(drop=True)
        
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    numeric_cols = ['trial time', 'x center', 'y center']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    required = ['trial time', 'x center', 'y center']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
        
    def find_col(pattern):
        for c in df.columns:
            if pattern in c:
                return c
        return None
        
    open_1 = find_col("open_arm_1")
    open_2 = find_col("open_arm_2")
    closed_1 = find_col("closed_arm_1")
    closed_2 = find_col("closed_arm_2")
    center = find_col("centre_zone")
    
    def clean_zone_col(col_name):
        if col_name and col_name in df.columns:
            s = df[col_name].astype(str).replace('-', '0')
            return pd.to_numeric(s, errors='coerce').fillna(0)
        return pd.Series(0, index=df.index)

    df['InOpen'] = 0
    df['InClosed'] = 0
    df['InCentre'] = 0
    
    val_open_1 = clean_zone_col(open_1)
    val_open_2 = clean_zone_col(open_2)
    val_closed_1 = clean_zone_col(closed_1)
    val_closed_2 = clean_zone_col(closed_2)
    val_center = clean_zone_col(center)
    
    df['InOpen'] = ((val_open_1 == 1) | (val_open_2 == 1)).astype(int)
    df['InClosed'] = ((val_closed_1 == 1) | (val_closed_2 == 1)).astype(int)
    df['InCentre'] = (val_center == 1).astype(int)
        
    clean_df = pd.DataFrame({
        'Time': df['trial time'],
        'X': df['x center'],
        'Y': df['y center'],
        'InOpen': df['InOpen'],
        'InClosed': df['InClosed'],
        'InCentre': df['InCentre']
    })
    
    clean_df = clean_df.dropna(subset=['X', 'Y'])
    
    shift_res = detect_center_shift(clean_df)
    clean_df = apply_center_shift(clean_df, shift_res)
    
    clean_df['dx'] = clean_df['x_m'].diff()
    clean_df['dy'] = clean_df['y_m'].diff()
    clean_df['dt'] = clean_df['Time'].diff()
    
    clean_df['distance'] = np.sqrt(clean_df['dx']**2 + clean_df['dy']**2)
    clean_df['velocity'] = clean_df['distance'] / clean_df['dt']
    
    clean_df = clean_df.drop(columns=['dx', 'dy', 'dt'])
    
    return clean_df
