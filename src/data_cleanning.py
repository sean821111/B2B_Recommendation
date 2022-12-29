import pandas as pd
import numpy as np
import re
import statistics
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder

class DataCleaning():
    def __init__(self, data:pd.DataFrame) -> None:
        self.data = data 
        self.SKUs = data["WTPARTNUMBER"]
        self.label_encoder = LabelEncoder()
        self.mutilabel_encoder = MultiLabelBinarizer()
        self.onehot_encoder = OneHotEncoder()

    def isNaN(self, num):
        return num != num
    
    # Standardization 
    def z_score(self, nparr):
        nparr = np.array(nparr, dtype='float')
        return [(x-nparr.mean())/nparr.std() for x in nparr]
    
    def encoding(self, test:bool=True):
        data = self.data.copy()
        # initialize 
        df = pd.DataFrame()

        # if not test:
        try:
            df['APPLICATION'] = data['APPLICATION']
            df.loc[self.isNaN(df["APPLICATION"]) , "APPLICATION"] =  statistics.mode(data['APPLICATION'])
            df['APPLICATION'] = self.label_encoder.fit_transform(df['APPLICATION'])

            df['COLORGAMUT'] = data['COLORGAMUT']
            df["COLORGAMUT"].fillna(df["COLORGAMUT"].mode()[0], inplace=True)
            df['COLORGAMUT'] = self.z_score(df['COLORGAMUT'].to_numpy())
        except Exception as exp:
            print(exp)

        # df['BRIGHTNESS'] = data['BRIGHTNESS']
        try:
            df['RESPONSE_TIME_TYP'] = data['RESPONSE_TIME_TYP']
            df["RESPONSE_TIME_TYP"].fillna(df["RESPONSE_TIME_TYP"].mode()[0], inplace=True)
            df['VIEWING_DIRECTION'] = data['VIEWING_DIRECTION']
            df.loc[self.isNaN(df["VIEWING_DIRECTION"]) , "VIEWING_DIRECTION"] = 'Free'
            df['VIEWING_DIRECTION'] = self.label_encoder.fit_transform(df['VIEWING_DIRECTION'])
        except Exception as exp:
            print(exp)

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
        # RESOLUTION_H = []
        # RESOLUTION_V = []

        # for x in data["RESOLUTION"]:
        #     if not self.isNaN(x):
        #         RH, RV = re.split(r'x', x)
        #         RESOLUTION_H.append(int(RH))
        #         RESOLUTION_V.append(int(RV))
        #     else:
        #         RESOLUTION_H.append(statistics.mode(RESOLUTION_H))
        #         RESOLUTION_V.append(statistics.mode(RESOLUTION_V))

        # using regular expression 
        data['RESOLUTION'].fillna(data['RESOLUTION'].mode()) # 因為解析度是固定組合所以直接用 mode填缺失值
        RESOLUTION_H = data['RESOLUTION'].apply(lambda x: re.findall(r"\d+\.*\d*", x)[0])
        RESOLUTION_V = data['RESOLUTION'].apply(lambda x: re.findall(r"\d+\.*\d*", x)[1])

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

        except Exception as exp:
            print(exp)

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
        try:
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

        except Exception as exp:
            print(exp)


        # TOUCH_STRUCTURE
        try:
            TOUCH_STRUCTURE = []
            for x in data["TOUCH_STRUCTURE"]:
                if not self.isNaN(x):
                    TOUCH_STRUCTURE.append(x)
                else:
                    TOUCH_STRUCTURE.append("Non_Touch")

            df['TOUCH_STRUCTURE'] = TOUCH_STRUCTURE

            # LCD_TECHNOLOGY
            d3 = pd.get_dummies(data["LCD_TECHNOLOGY"])
            for k in d3.keys():
                df[k] = d3[k]

        except Exception as exp:
            print(exp)

        # 'COLOR_NUMBER', 'OPERATION_TEMP', 'STORAGE_TEMP'
        special_features = ['COLOR_NUMBER', 'OPERATION_TEMP', 'STORAGE_TEMP']
        for col in special_features:
            data[col].fillna(data[col].mode()[0], inplace=True)
            df[col] = self.label_encoder.fit_transform(data[col])


        # Solution Multi-Label categorical attribute [APPLICATION_TYPE, APPLICATION_PRODUCTION, TOUCH_TYPE]
        special_features = ['APPLICATION_TYPE', 'APPLICATION_PRODUCTION', 'TOUCH_TYPE']
        for col in special_features:
            data[col].fillna(data[col].mode()[0], inplace=True)
            _encode = self.mutilabel_encoder.fit_transform(data[col].apply(lambda x: [s.strip() for s in str(x).split(',')]))
            df[col] = _encode.tolist()

        # Solution Single-Label categorical attribute[CPU, GPD_IO]
        special_features = ['CPU', 'GPD_IO']
        for col in special_features:
            data[col].fillna(data[col].mode()[0], inplace=True)
            _encode = self.mutilabel_encoder.fit_transform(data[col].apply(lambda x: [x]))
            df[col] = _encode.tolist()

        # Solution numerical attribute [PANNEL_SIZE_MIN, PANNEL_SIZE_MAX, POWER_SUPPLY, POWER_CONSUMPTION_FG]
        # PANNEL_SIZE_MIN, PANNEL_SIZE_MAX
        special_features = ['PANNEL_SIZE_MIN', 'PANNEL_SIZE_MAX', 'POWER_CONSUMPTION_FG']
        for col in special_features:
            data[col].fillna(data[col].mode()[0], inplace=True)
            re_data = data[col].apply(lambda x: float(re.findall(r"^\d*\.*\d+[Ww]|^\d+\.*\d*$", str(x))[0].replace('W', '')))
            df[col] = self.z_score(re_data)
        
        # POWER_SUPPLY
        def unit_convertion(num_unit):
            num_unit = str(num_unit)
            ec = re.findall(r"\d+\.*\d*", num_unit) # number of electric current
            unit = num_unit
            ec_ = ec.copy()
            ec_ += ['-', '~']
            for n in ec_:
                unit = unit.replace(n, '') # drop electric current str to take unit
            ec = [float(e) for e in ec]
            avg_ec = sum(ec)/len(ec) # avg electric current

            if unit.lower() == 'ma': # change unit
                avg_ec = avg_ec/1000.0
            return avg_ec

        volt = data['POWER_SUPPLY'].apply(lambda x: str(x).split('/')[0]).apply(lambda x: np.mean([ float(ele) for ele in re.findall(r"\d+\.*\d*", x)]))
        df['POWER_SUPPLY_VOLT'] = self.z_score(volt)
        ampere = data['POWER_SUPPLY'].apply(lambda x: str(x).split('/')[1]).apply(unit_convertion)
        df['POWER_SUPPLY_AMPERE'] = self.z_score(ampere)

        return df

if __name__ == '__main__':
    solution_path = "./data/方案產品清單20221205.xlsx"
    solution_data = pd.read_excel(solution_path)
    data_cleaning = DataCleaning(solution_data)
    df = data_cleaning.encoding()
    print(df.head(15))