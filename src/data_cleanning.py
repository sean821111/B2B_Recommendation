import pandas as pd
import numpy as np
import re
import statistics
from sklearn.preprocessing import LabelEncoder

class DataCleaning():
    def __init__(self, data) -> None:
        self.data = data 
        self.SKUs = data["WTPARTNUMBER"]
        self.label_encoder = LabelEncoder()

    def isNaN(self, num):
        return num != num
    
    # Standardization 
    def z_score(self, nparr):
        nparr = np.array(nparr, dtype='float')
        return [(x-nparr.mean())/nparr.std() for x in nparr]
    
    def encoding(self):
        data = self.data
        # initialize 
        df = pd.DataFrame()

        df["WTPARTNUMBER"] = data["WTPARTNUMBER"]

        df['APPLICATION'] = data['APPLICATION']
        df.loc[self.isNaN(df["APPLICATION"]) , "APPLICATION"] =  statistics.mode(data['APPLICATION'])
        df['APPLICATION'] = self.label_encoder.fit_transform(df['APPLICATION'])

        df['COLORGAMUT'] = data['COLORGAMUT']
        df["COLORGAMUT"].fillna(df["COLORGAMUT"].mode()[0], inplace=True)
        df['COLORGAMUT'] = self.z_score(df['COLORGAMUT'].to_numpy())

        # df['COLOR_NUMBER'] = data['COLOR_NUMBER']
        # df['BRIGHTNESS'] = data['BRIGHTNESS']
        df['RESPONSE_TIME_TYP'] = data['RESPONSE_TIME_TYP']
        df["RESPONSE_TIME_TYP"].fillna(df["RESPONSE_TIME_TYP"].mode()[0], inplace=True)

        df['VIEWING_DIRECTION'] = data['VIEWING_DIRECTION']
        df.loc[self.isNaN(df["VIEWING_DIRECTION"]) , "VIEWING_DIRECTION"] = 'Free'
        df['VIEWING_DIRECTION'] = self.label_encoder.fit_transform(df['VIEWING_DIRECTION'])

        # df['OPERATION_TEMP'] = data['OPERATION_TEMP']
        # df['STORAGE_TEMP']= data['STORAGE_TEMP']

        # processing OUTLINE_TYP_HV
        OUTLINE_H = []
        OUTLINE_V = []
        OUTLINE_D = []
        for x in data["OUTLINE_TYP_HV"]:
            OUTLINE_TYP = re.split(r'x|X|\*', x)
            H = float(re.findall(r"[-+]?(?:\d*\.*\d+)", OUTLINE_TYP[0])[0])
            V = float(re.findall(r"[-+]?(?:\d*\.*\d+)",OUTLINE_TYP[1])[0])
            if len(OUTLINE_TYP)<3 or OUTLINE_TYP[2] =="--":
                D = statistics.mode(OUTLINE_D) # using typical value assign
            else:
                D = float(re.findall(r"[-+]?(?:\d*\.*\d+)",OUTLINE_TYP[2])[0])
            
            OUTLINE_H.append(H)
            OUTLINE_V.append(V)
            OUTLINE_D.append(D)
        df['OUTLINE_H'] = self.z_score(OUTLINE_H)
        df['OUTLINE_V'] = self.z_score(OUTLINE_V)
        df['OUTLINE_D'] = self.z_score(OUTLINE_D)

        # RESOLUTION
        RESOLUTION_H = []
        RESOLUTION_V = []
        for x in data["RESOLUTION"]:
            if not self.isNaN(x):
                RH, RV = re.split(r'x', x)
                RESOLUTION_H.append(int(RH))
                RESOLUTION_V.append(int(RV))
            else:
                RESOLUTION_H.append(statistics.mode(RESOLUTION_H))
                RESOLUTION_V.append(statistics.mode(RESOLUTION_V))
        df['RESOLUTION_H'] = self.z_score(RESOLUTION_H)
        df['RESOLUTION_V'] = self.z_score(RESOLUTION_V)

        # GLASS THICKNESS
        try:
            GLASS_THICKNESS = []
            for x in data["GLASS_THICKNESS"]:
                if not self.isNaN(x):
                    GLASS_THICKNESS.append(float(re.split(r'mm| ',x)[0]))
                else:
                    GLASS_THICKNESS.append(statistics.mode(GLASS_THICKNESS)) # using typical value assign
            df['GLASS_THICKNESS'] = self.z_score(GLASS_THICKNESS)
        except Exception as exp:
            print(exp)

        # CONTRAST RATIO
        CONTRAST_RATIO = []
        for x in data["CONTRAST_RATIO"]:
            if not self.isNaN(x):
                CONTRAST_RATIO.append(int(re.split(r':', x)[0]))
            else: 
                CONTRAST_RATIO.append(statistics.mode(CONTRAST_RATIO))
        df['CONTRAST_RATIO'] = self.z_score(CONTRAST_RATIO)


        # VIEW ANGLE (H/V)
        VIEW_ANGLE_H = []
        VIEW_ANGLE_V = []
        data["VIEW_ANGLE_H_V"].fillna(data["VIEW_ANGLE_H_V"].mode()[0], inplace=True)
        for x in data["VIEW_ANGLE_H_V"]:
            H = int(re.split(r'/| ', x)[0])
            V = int(re.split(r'/| ', x)[1])
            VIEW_ANGLE_H.append(H)
            VIEW_ANGLE_V.append(H)
        df['VIEW_ANGLE_H'] = self.z_score(VIEW_ANGLE_H)
        df['VIEW_ANGLE_V'] = self.z_score(VIEW_ANGLE_V)

        # SCREEN ORIENTATION
        SCREEN_ORIENTATION = []
        for x in data["SCREEN_ORIENTATION"]:
            if x == "Portrait":
                SCREEN_ORIENTATION.append(1)
            else: 
                SCREEN_ORIENTATION.append(0)
        df['SCREEN_ORIENTATION'] = SCREEN_ORIENTATION        


        # RELIABLE TIME
        RA_TIME = []
        for x in data["RA_TIME"]:
            if not self.isNaN(x):
                RA_TIME.append(int(re.split(r'hrs| ',x)[0]))
            else:
                # only 1 missing value currently, and according to it's APPLICATION assign to 1000hrs
                RA_TIME.append(1000)
                
        df['RA_TIME'] = self.z_score(RA_TIME)


        # -----------------------------------
        # Categorical type

        # 14 types in LCM INTERFACE
        categories = []
        for x in data["LCM_INTERFACE"]:
            if not self.isNaN(x):
                cats = re.split(r', ', x)
                for cat in cats:
                    if cat not in categories:
                        categories.append(cat)

        # Apply one hot encoding
        # initialize a dictionary for each category 
        l = len(data["LCM_INTERFACE"])
        d2 = {cat:[0 for i in range(l)] for cat in categories}

        # assign value
        for i, x in enumerate(data["LCM_INTERFACE"]):
            if not self.isNaN(x):
                cats = re.split(r', ', x)
                for cat in cats:
                    d2[cat][i] = 1

        for k in d2.keys():
            df[k] = d2[k]

        # TOUCH_STRUCTURE
        try:
            TOUCH_STRUCTURE = []
            for x in data["TOUCH_STRUCTURE"]:
                if not self.isNaN(x):
                    TOUCH_STRUCTURE.append(x)
                else:
                    TOUCH_STRUCTURE.append("Non_Touch")

            df['TOUCH_STRUCTURE'] = TOUCH_STRUCTURE
        except Exception as exp:
            print(exp)

        # LCD_TECHNOLOGY
        d3 = pd.get_dummies(data["LCD_TECHNOLOGY"])
        for k in d3.keys():
            df[k] = d3[k]
    
        # 'COLOR_NUMBER', 'OPERATION_TEMP', 'STORAGE_TEMP'
        special_features = ['COLOR_NUMBER', 'OPERATION_TEMP', 'STORAGE_TEMP']
        for col in special_features:
            data[col].fillna(data[col].mode()[0], inplace=True)
            df[col] = self.label_encoder.fit_transform(data[col])

        return df

if __name__ == '__main__':
    CELL_path = "../../PLM/Cell產品清單20221205.xlsx"
    CELL_data = pd.read_excel(CELL_path)
        
    data_cleaning = DataCleaning(CELL_data)
    df = data_cleaning.encoding()
    print(df.head())