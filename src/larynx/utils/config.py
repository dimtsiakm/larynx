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

    def _get_root_path(self):
            return str(Path(__file__).resolve().parents[3])
