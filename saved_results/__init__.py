from config import Config
import os
import pandas as pd

class SavedResults(Config):

    dataframe = None

    def __init__(self, id=None, mse=None, ssim=None, parameters=None) -> None:
        super().__init__()

        if id==None and self.dataframe == None and self.csv_exists():
            
            self.dataframe = pd.read_csv(f'{self.results_seach}/all_results.csv')

        else:
            if not self.csv_exists():
                self.dataframe = pd.DataFrame({
                    "id": [id,],
                    "mse": [mse,],
                    "ssim": [ssim, ],
                    "parameters": [parameters, ]
                })

            else:
                if self.dataframe == None:
                    self.dataframe = pd.read_csv(f'{self.results_seach}/all_results.csv')
                
                new_row = {
                    "id": id,
                    "mse": mse,
                    "ssim": ssim,
                    "parameters": parameters
                }
                
                self.dataframe = pd.concat([self.dataframe, pd.DataFrame([new_row])])
            
            self.dataframe.to_csv(f'{self.results_seach}/all_results.csv', index=False)

    def csv_exists(self):
        file_path = os.path.join(self.results_seach, f'all_results.csv')
        return os.path.isfile(file_path)
    
    def show_dataframe_sorted_by_mse(self,):
        self.dataframe =  self.dataframe.sort_values(by="mse")
        return self.dataframe