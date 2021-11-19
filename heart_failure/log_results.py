import pandas as pd
from openpyxl import load_workbook


class Logger():
    def __init__(self,*args,**kwargs):
        # Set default parameters
        self.excel_filepath = 'results/df/test_log.xlsx'
        self.auto_train = False
        self.added_features = 0
        self.oversample = False
        self.forced_features = None
        self.train_size = 0.66
        self.robustness_iterations = 100
        # Update with provided parameters
        self.__dict__.update(kwargs)
    
    
    def update_log(self):
        line_info = {
            'auto_train': self.auto_train,
            'n_features': self.added_features,
            'oversample': self.oversample,
            'forced_features': self.forced_features,
            'train_size': self.train_size,
            'robustness_iterations': self.robustness_iterations
            }        
        try:
            book = load_workbook(self.excel_filepath)
            book = load_workbook(self.excel_filepath)
            writer = pd.ExcelWriter(self.excel_filepath, engine='openpyxl') 
            writer.book = book
            writer.book = book
            writer.sheets = {ws.title: ws for ws in book.worksheets}
            for sheetname in writer.sheets:
                pd.DataFrame([line_info]).to_excel(writer,sheet_name=sheetname,startrow=writer.sheets[sheetname].max_row,
                                                   index=True,header=False)

            writer.save()
        except: 
            print(f'{self.excel_filepath} not found. {self.excel_filepath} created.')
            pd.DataFrame([line_info]).to_excel(self.excel_filepath)