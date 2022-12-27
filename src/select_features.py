import pandas as pd
import re
import matplotlib.pyplot as plt
import plotly.express as px

path = "D:\\Junxiang\Digital Platform\Data\PLM\\"

df_cell = pd.read_excel(path+"Cell產品清單20221205.xlsx")
df_lcm = pd.read_excel(path+"LCM產品清單20221205.xlsx")
df_tp = pd.read_excel(path+"TP產品清單20221205.xlsx")

# intersection of columns
filter_cols = set(df_cell.columns)&set(df_lcm.columns)&set(df_tp.columns)

# importance columns
filter_cols = set([ 'WTPARTNUMBER'
                    ,'DISPLAY_AREA_DIAGONAL_SIZE'
                    ,'RESOLUTION'
                    ,'ASPECT_RATIO'
                    ,'OUTLINE_TYP_HV'
                    ,'GLASS_THICKNESS'
                    ,'COLORGAMUT'
                    ,'COLOR_NUMBER'
                    ,'CONTRAST_RATIO'
                    ,'RESPONSE_TIME_TYP'
                    ,'BRIGHTNESS'
                    ,'SCREEN_ORIENTATION'
                    ,'VIEW_ANGLE_H_V'
                    ,'VIEWING_DIRECTION'
                    ,'LCM_INTERFACE'
                    ,'OPERATION_TEMP'
                    ,'STORAGE_TEMP'
                    ,'RA_TIME'
                    ,'TOUCH_STRUCTURE'
                    ,'APPLICATION'
                    ,'LCD_TECHNOLOGY'
                ])&filter_cols

df_cell_ = df_cell[filter_cols]
df_lcm_ = df_lcm[filter_cols]
df_tp_ = df_tp[filter_cols]

# Combine lcm, cell, tp
df_sku = pd.concat([df_lcm_, df_cell_, df_tp_], axis=0)
df_sku.reset_index(inplace=True)

df_sku.to_excel(path+"magento_SKU.xlsx", index=False)

# Pending features
df_sku_pending = df_sku[['COLOR_NUMBER', 'BRIGHTNESS', 'OPERATION_TEMP', 'STORAGE_TEMP']]

# Insight - bar plot of columns
def bar_plot(x, y):
    fig = px.bar(y=y,
                x=x,
                text=x/df_sku_pending.shape[0],
                orientation='h')
    fig.show()


n_nan = df_sku_pending.isna().sum()
bar_plot(n_nan, df_sku_pending.columns)

n_unique = df_sku_pending.nunique()
bar_plot(n_unique, df_sku_pending.columns)

