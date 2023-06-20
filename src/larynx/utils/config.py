from pathlib import Path
import os

class Config:
    def __init__(self):
        self.seed = 555
        self.project_path = self._get_root_path()
        self.data_raw_path = os.path.join(self.project_path, 'data/', 'raw/')
        self.data_processed_path = os.path.join(self.project_path, 'data/', 'processed/')
        self.figures_path = os.path.join(self.project_path, 'reports/figures/')
        self.temp_figures_path = os.path.join(self.project_path, 'reports/figures/temp/')
        self.models_path = os.path.join(self.project_path, 'models/')
        self.data_path = os.path.join(self.project_path, 'data/')

        self.min_window_level = -155
        self.max_window_level = 268

        self.roi_2d = [96, 96]

    def _get_root_path(self):
            return str(Path(__file__).resolve().parents[3])
    
    def join_data_path_with(self, folder: str):
         pth = os.path.join(self.data_path, folder)
         if os.path.isdir(pth):
            return pth
         assert('The path is not valid; check the folder: ' + folder)

    def get_ct_window_level(self):
         print('CT window level: ', self.min_window_level, self.max_window_level)
         return self.min_window_level, self.max_window_level