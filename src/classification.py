import pandas as pd
import numpy as np
import json


class Classification():
    # json格式 = {"tftdisplay_優選":{}, "tftdisplay_客製":{}, "paperdisplay_優選":{}, "paperdisplay_客製":{}, "方案產品暨應用":{}, "顯示紙暨系統整合方案":{}, "顯示紙":{}}

    def __init__(self,dir_cell,dir_lcm,dir_tp,dir_systemBoard,dir_solution,dir_hannspree):
        # Cell + Lcm + Tp 分出 TFT 顯示器、Paper 顯示器
        self.cell_data_dir = dir_cell
        self.lcm_data_dir = dir_lcm
        self.tp_data_dir = dir_tp      
        # 方案
        self.systemBoard_data_dir = dir_systemBoard
        self.solution_data_dir = dir_solution  
        # hannspree
        self.hannspree_data_dir = dir_hannspree

    def load_excel(self,file):
        df_ = pd.read_excel(file,header=0,index_col=0)
        return df_

    def Display_data(self):
        def filter(df):
            #篩出RELEASED之"WTPARTNUMBER","A3_PRODUCT_TYPE","SHIPMENT_TYPE","LCD_TECHNOLOGY"欄位
            filt = (df["STATESTATE"]=="RELEASED")
            df_ = df.loc[filt,["WTPARTNUMBER","A3_PRODUCT_TYPE","SHIPMENT_TYPE","LCD_TECHNOLOGY"]]
            return df_
        #收集CELL、LCM、TP 中 RELEASED SKU號碼
        cell_data = self.load_excel(self.cell_data_dir)
        lcm_data = self.load_excel(self.lcm_data_dir)
        tp_data = self.load_excel(self.tp_data_dir)
        cell_df =  filter(cell_data) 
        lcm_df =  filter(lcm_data)
        tp_df =  filter(tp_data)
        display_df = pd.concat([cell_df, lcm_df], ignore_index=True)
        display_df = pd.concat([display_df, tp_df], ignore_index=True)
        return display_df

    def tft_data(self):
        def classify(df):
            Preferred_f1 = ((df["A3_PRODUCT_TYPE"]=="HSD") & ((df["SHIPMENT_TYPE"]=="LCM") | (df["SHIPMENT_TYPE"]=="TP") | (df["SHIPMENT_TYPE"]=="FOG")) & ((df["LCD_TECHNOLOGY"]=="TN") | (df["LCD_TECHNOLOGY"]=="IPS")))
            Preferred_f2 = (((df["A3_PRODUCT_TYPE"]=="ODM") | (df["A3_PRODUCT_TYPE"]=="ODM PDBed(-P)")) & ((df["SHIPMENT_TYPE"]=="LCM") | (df["SHIPMENT_TYPE"]=="TP") | (df["SHIPMENT_TYPE"]=="FOG")) & ((df["LCD_TECHNOLOGY"]=="TN") | (df["LCD_TECHNOLOGY"]=="IPS")))
            Preferred_f3 = ((df["A3_PRODUCT_TYPE"]=="ODM PDBing(-PX)") & ((df["SHIPMENT_TYPE"]=="LCM") | (df["SHIPMENT_TYPE"]=="TP") | (df["SHIPMENT_TYPE"]=="FOG")) & ((df["LCD_TECHNOLOGY"]=="TN") | (df["LCD_TECHNOLOGY"]=="IPS")))
            Preferred_f4 = ((df["A3_PRODUCT_TYPE"]=="ODM PDBing(-SX)") & ((df["SHIPMENT_TYPE"]=="LCM") | (df["SHIPMENT_TYPE"]=="TP") | (df["SHIPMENT_TYPE"]=="FOG")) & ((df["LCD_TECHNOLOGY"]=="TN") | (df["LCD_TECHNOLOGY"]=="IPS")))
            Preferred = Preferred_f1 | Preferred_f2 | Preferred_f3 | Preferred_f4
            Custom_f1 = ((df["A3_PRODUCT_TYPE"]=="ODM PDBing(-XX)") & ((df["SHIPMENT_TYPE"]=="LCM") | (df["SHIPMENT_TYPE"]=="TP") | (df["SHIPMENT_TYPE"]=="FOG")) & ((df["LCD_TECHNOLOGY"]=="TN") | (df["LCD_TECHNOLOGY"]=="IPS")))
            Custom_f2 = ((df["A3_PRODUCT_TYPE"]=="HSD") & (df["SHIPMENT_TYPE"]=="Cell") & ((df["LCD_TECHNOLOGY"]=="TN") | (df["LCD_TECHNOLOGY"]=="IPS")))
            Custom = Preferred | Custom_f1 | Custom_f2

            Preferred_df = df.loc[Preferred,["WTPARTNUMBER"]]
            Custom_df = df.loc[Custom,["WTPARTNUMBER"]]
            return Preferred_df, Custom_df

        display_df = self.Display_data()
        tft_Preferred_df, tft_Custom_df = classify(display_df)
        return tft_Preferred_df, tft_Custom_df

    def paper_data(self):
        def classify(df):
            Preferred_f1 = ((df["A3_PRODUCT_TYPE"]=="HSD") & ((df["SHIPMENT_TYPE"]=="LCM") | (df["SHIPMENT_TYPE"]=="TP") | (df["SHIPMENT_TYPE"]=="FOG")) & ((df["LCD_TECHNOLOGY"]=="Reflective") | (df["LCD_TECHNOLOGY"]=="Transflective")))
            Preferred_f2 = (((df["A3_PRODUCT_TYPE"]=="ODM") | (df["A3_PRODUCT_TYPE"]=="ODM PDBed(-P)")) & ((df["SHIPMENT_TYPE"]=="LCM") | (df["SHIPMENT_TYPE"]=="TP") | (df["SHIPMENT_TYPE"]=="FOG")) & ((df["LCD_TECHNOLOGY"]=="Reflective") | (df["LCD_TECHNOLOGY"]=="Transflective")))
            Preferred_f3 = ((df["A3_PRODUCT_TYPE"]=="ODM PDBing(-PX)") & ((df["SHIPMENT_TYPE"]=="LCM") | (df["SHIPMENT_TYPE"]=="TP") | (df["SHIPMENT_TYPE"]=="FOG")) & ((df["LCD_TECHNOLOGY"]=="Reflective") | (df["LCD_TECHNOLOGY"]=="Transflective")))
            Preferred_f4 = ((df["A3_PRODUCT_TYPE"]=="ODM PDBing(-SX)") & ((df["SHIPMENT_TYPE"]=="LCM") | (df["SHIPMENT_TYPE"]=="TP") | (df["SHIPMENT_TYPE"]=="FOG")) & ((df["LCD_TECHNOLOGY"]=="Reflective") | (df["LCD_TECHNOLOGY"]=="Transflective")))
            Preferred = Preferred_f1 | Preferred_f2 | Preferred_f3 | Preferred_f4
            Custom_f1 = ((df["A3_PRODUCT_TYPE"]=="ODM PDBing(-XX)") & ((df["SHIPMENT_TYPE"]=="LCM") | (df["SHIPMENT_TYPE"]=="TP") | (df["SHIPMENT_TYPE"]=="FOG")) & ((df["LCD_TECHNOLOGY"]=="Reflective") | (df["LCD_TECHNOLOGY"]=="Transflective")))
            Custom_f2 = ((df["A3_PRODUCT_TYPE"]=="HSD") & (df["SHIPMENT_TYPE"]=="Cell") & ((df["LCD_TECHNOLOGY"]=="Reflective") | (df["LCD_TECHNOLOGY"]=="Transflective")))
            Custom = Preferred | Custom_f1 | Custom_f2

            Preferred_df = df.loc[Preferred,["WTPARTNUMBER"]]
            Custom_df = df.loc[Custom,["WTPARTNUMBER"]]
            return Preferred_df, Custom_df

        display_df = self.Display_data()
        paper_Preferred_df, paper_Custom_df = classify(display_df)
        return paper_Preferred_df, paper_Custom_df

    def hannspree_data(self):
        def filter(df):
            #篩出RELEASED之"WTPARTNUMBER"欄位
            filt = (df["STATESTATE"]=="RELEASED")
            df_ = df.loc[filt,["WTPARTNUMBER"]]
            return df_

        hannspree_data = self.load_excel(self.hannspree_data_dir)
        hannspree_df = filter(hannspree_data)
        return hannspree_df

    def solution_data(self):
        def filter(df):
            #篩出RELEASED之"WTPARTNUMBER"欄位
            filt = (df["STATESTATE"]=="RELEASED")
            df_ = df.loc[filt,["WTPARTNUMBER"]]
            return df_

        systemBoard_data = self.load_excel(self.systemBoard_data_dir)
        solution_data = self.load_excel(self.solution_data_dir)

        systemBoard_df = filter(systemBoard_data)
        solution_df = filter(solution_data)

        hannspree_df = self.hannspree_data()
        solution_df = pd.concat([solution_df, hannspree_df], ignore_index=True)

        return systemBoard_df, solution_df


    def output_folder(self):
        all_data = {}

        # 各產品func.回傳 df,(df to list)，此使用list存字典
        tft_p_df, tft_c_df = self.tft_data()
        paper_p_df, paper_c_df = self.paper_data()
        solution_df, solution_hannspree_df = self.solution_data()
        hannspree_df = self.hannspree_data()
        # print("hannspree_df:",hannspree_df["WTPARTNUMBER"].values.tolist())

        all_data["tftdisplay_Preferred"] = tft_p_df["WTPARTNUMBER"].values.tolist()
        all_data["tftdisplay_Custom"] = tft_c_df["WTPARTNUMBER"].values.tolist()
        all_data["paperdisplay_Preferred"] = paper_p_df["WTPARTNUMBER"].values.tolist()
        all_data["paperdisplay_Custom"] = paper_c_df["WTPARTNUMBER"].values.tolist() 
        all_data["systemBoard"] = solution_df["WTPARTNUMBER"].values.tolist()   
        all_data["solution_hannspree"] = solution_hannspree_df["WTPARTNUMBER"].values.tolist()     
        all_data["hannspree"] = hannspree_df["WTPARTNUMBER"].values.tolist()
        # print("all_data_df:\n",all_data)
        out_json = json.dumps(all_data)

        return tft_p_df,tft_c_df,paper_p_df,paper_c_df,solution_df,solution_hannspree_df,hannspree_df,out_json

def main():
    # Cell + Lcm + Tp 分出 TFT 顯示器、Paper 顯示器
    cell_data_dir = "../data/PLM/Cell產品清單20221229.xlsx"
    lcm_data_dir = "../data/PLM/LCM產品清單20221229.xlsx"
    tp_data_dir = "../data/PLM/TP產品清單20221229.xlsx"
    # 方案
    systemBoard_data_dir = "../data/PLM/方案產品清單20221229.xlsx"
    solution_data_dir = "../data/PLM/方案整機產品清單20221229.xlsx"
    # hannspree
    hannspree_data_dir = "../data/PLM/Hannspree整機產品清單20221229.xlsx"

    x = Classification(cell_data_dir,lcm_data_dir,tp_data_dir,systemBoard_data_dir,solution_data_dir,hannspree_data_dir)
    tft_p_df,tft_c_df,paper_p_df,paper_c_df,solution_df,solution_hannspree_df,hannspree_df,out_json = x.output_folder()
    # 對應df:{"tftdisplay_Preferred","tftdisplay_Custom","paperdisplay_Preferred","paperdisplay_Custom","systemBoard","solution_hannspree","hannspree"}
    print(f"\ntftdisplay_Preferred:\n{tft_p_df}\ntftdisplay_Custom:\n{tft_c_df}\npaperdisplay_Preferred:\n{paper_p_df}\npaperdisplay_Custom:\n{paper_c_df}")
    print(f"\nsystemBoard:\n{solution_df}\nsolution_hannspree:\n{solution_hannspree_df}\npaperdisplay_Custom:{hannspree_df}")
    print(f"\njson_data:\n{out_json}")

    return tft_p_df,tft_c_df,paper_p_df,paper_c_df,solution_df,solution_hannspree_df,hannspree_df

if __name__=="__main__":
    main()
