import sys
import pandas as pd
import argparse

from data_eng.loading import GetData

class LoadData:
    '''
    The main functionality of this class is to load the data to the folder path
    we have assigned in params.yaml file
    function return data and save it to folder we have assigned
    '''

    def __init__(self):
        self.getdata = GetData()

    def load_data(self, input_path, output_path):
            
        try:
            print(f"Loading data from the source")
            
            self.data = self.getdata.get_data(input_path)
            self.raw_data_path = output_path
            self.data.to_csv(self.raw_data_path, sep=',', encoding='utf-8', index=False)
            print(f"Data Loaded from the source Successfully !!!")
        
        except Exception as e:
            print(
                f"Exception Occurred while loading data from the source -->{e}")
            raise Exception(e, sys) from e



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input_path', default='data/external/Consignment_pricing.csv', help='Ruta del archivo CSV de entrada')
    args.add_argument('--output_path', default='data/raw/dataset.csv', help='Ruta de salida para guardar los datos')

    parsed_args = args.parse_args()
    
    LoadData().load_data(input_path=parsed_args.input_path, output_path=parsed_args.output_path)


