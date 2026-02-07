import pandas as pd
import numpy as np
import os

class TrackingData:
    def __init__(self, data_dict, frames, fps, filename):
        self.data = data_dict
        self.frames = frames
        self.fps = fps
        self.seconds = frames / fps
        self.filename = filename
        self.median_data = self._calculate_median_data()
        self.distance_units = "pixel"
        self.object_type = "TrackingData"
        self.zones = {}

    def _calculate_median_data(self):
        median_list = []
        for point_name, df in self.data.items():
            median_list.append({
                'PointName': point_name,
                'x': df['x'].median(),
                'y': df['y'].median()
            })
        return pd.DataFrame(median_list).set_index('PointName')

def read_dlc_csv(file_path, fps=1):
    """
    Reads DLC Tracking data from a csv file and returns a TrackingData object.
    
    Args:
        file_path (str): path to a DLC tracking .csv file
        fps (float): frames per second of the recording.
    
    Returns:
        TrackingData object
    """
    if fps == 1:
        print("Warning: no fps set. setting fps to 1. keep in mind that time based analyses are resolved in frames / second")

    try:
        df_raw = pd.read_csv(file_path, header=[0, 1, 2], index_col=0)
    except Exception as e:
        df_raw = pd.read_csv(file_path, header=[0, 1], index_col=0)
    
    out_data = {}
    
    if isinstance(df_raw.columns, pd.MultiIndex):
        bodyparts = df_raw.columns.get_level_values(1).unique()
        
        for bp in bodyparts:
            try:
                bp_data = df_raw.xs(bp, axis=1, level=1)
                if 'x' in bp_data.columns and 'y' in bp_data.columns:
                    out_data[bp] = bp_data.copy()
                    out_data[bp]['frame'] = bp_data.index
                else:
                    subset = bp_data.iloc[:, :3].copy()
                    subset.columns = ['x', 'y', 'likelihood']
                    subset['frame'] = subset.index
                    out_data[bp] = subset
            except Exception:
                continue
                
    frames = np.arange(len(df_raw))
    filename = os.path.basename(file_path)
    
    return TrackingData(out_data, frames, fps, filename)
