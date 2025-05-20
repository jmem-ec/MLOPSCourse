import sys
import pandas as pd
import argparse

class GetData:
    '''
    The main functionality is to get data from external source 
    Function return None 
    '''

    def __init__(self):
        pass


    def get_data(self, input_path):
        try:
            print("getting the data from the source")
            #self.config = self.read_params(config_path)
            

            self.data_path = input_path
            self.data = pd.read_csv(self.data_path, sep=',', encoding='utf-8')
        
            print(f"Data Fetched from the source Successfully !!!")
            return self.data


        except Exception as e:
            print(
                f"Exception Occurred while getting data from the source -->{e}")
            raise Exception(e, sys) from e


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--input_path', default='data/external/Consignment_pricing.csv')
    parsed_args = args.parse_args()
    data = GetData().get_data(input_path=parsed_args.input_path)

