import pandas as pd
import numpy as np 
import data_cleanning as dc
import classification
from recommendation_model import Recommend

class Dissimilarity():
    """
    num_dissim : func, default: euclidian_dissim
        Dissimilarity function used by the algorithm for numerical variables.
        Defaults to the Euclidian dissimilarity function.
    cat_dissim : func, default: matching_dissim
        Dissimilarity function used by the kmodes algorithm for categorical variables.
        Defaults to the matching dissimilarity function.
    gamma : float, default: None
        Weighing factor that determines relative importance of numerical vs.
        categorical attributes (see discussion in Huang [1997]). By default,
        automatically calculated from data.
    """
    def __init__(self, df) -> None:
        self.df = df
        self.SKUs = df["WTPARTNUMBER"]

    def matching_dissim(self, a, b, **_):
        """Simple matching dissimilarity function"""
        return np.sum(a != b, axis=1)

    def euclidean_dissim(self, a, b, **_):
        """Euclidean distance dissimilarity function"""
        if np.isnan(a).any() or np.isnan(b).any():
            raise ValueError("Missing values detected in numerical columns.")
        return np.sum((a - b) ** 2, axis=1)

    def _split_num_cat(self, X, categorical):
        """Extract numerical and categorical columns.
        Convert to numpy arrays, if needed.
        :param X: Feature matrix
        :param categorical: Indices of categorical columns
        """
        Xnum = np.asanyarray(X[:, [ii for ii in range(X.shape[1])
                                if ii not in categorical]]).astype(np.float64)
        Xcat = np.asanyarray(X[:, categorical])
        return Xnum, Xcat

    def similarItems(self, sku):
        dfMatrix = self.df.to_numpy()
        catColumnsPos = [self.df.columns.get_loc(col) for col in list(self.df.select_dtypes('object').columns)]

        Xnum, Xcat = self._split_num_cat(dfMatrix, catColumnsPos)

        gamma = 0.5 * np.mean(Xnum.std(axis=0))

        input_sku_attr = self.df.loc[self.df["WTPARTNUMBER"] == sku, :].drop("WTPARTNUMBER", axis=1)

        num_dissim = self.euclidean_dissim(input_sku_attr, Xnum)
        cat_dissim = self.matching_dissim(input_sku_attr, Xcat)

        dissim = num_dissim + gamma * cat_dissim
        sortedIdx = np.argsort(dissim)

        sortedSKUs = np.empty(len(self.SKUs),  dtype='object')
        for i,v in enumerate(sortedIdx):
            sortedSKUs[v] = self.SKUs[i]
        return sortedSKUs

    def batch_similarItems(self, type_="tftdisplay_Preferred", topN=10):
        """JSON format
        
        ```Json
        {
            "sku1": {
                "tftdisplay_Preferred": ["sku1", "sku2", ...],
                "tftdisplay_Custom": ["sku1", "sku3"],
                "paperdispla_Preferred": ["sku1"],
                "paperdisplay_Custom": [],
                "systemBoard": [],
                "solution": [],
                "hannspree": [] 
            },
        }
        ```
        """
        skuSimilar = {}

        for sku in self.SKUs:
            skuSimilar[sku] = self.similarItems(sku)[0:topN]

        return skuSimilar

def content_based(df : pd.DataFrame,
                  target_feature : str="WTPARTNUMBER",
                  filter_limit : int=10):

    Recom = Recommend(df, target_feature)

    # using the target_attr_table to calculate the target similarity
    cosine_sim = Recom.cosine_similarity()

    items_name = df[target_feature]
    df_sim = pd.DataFrame(cosine_sim, columns=items_name)
    df_sim.index = items_name

    return df_sim

if __name__ == '__main__':
    # (LCM CELL TP) data
    filePath = "../data/CELL_LCM_TP.xlsx"
    data = pd.read_excel(filePath)

    # clean and encode
    data_cleaning = dc.DataCleaning(data)
    df = data_cleaning.encoding()

    for col in df.columns.drop("WTPARTNUMBER"):
        df[col] = df[col].astype(float)

    # separte to tdtdisplay, ... (total 7 types)
    *lcm_cell_tp, solution_df, solution_hannspree_df, hannspree_df = classification.main()

    df_tft_p, df_tft_c, df_paper_p, df_paper_c = [pd.merge(df, tmp, how="inner", on="WTPARTNUMBER") for tmp in lcm_cell_tp]

    # calculate similarity
    dissim = Dissimilarity(df_tft_p)

    # single SKU 
    dissim.similarItems("010GPW2-900001-PX")

    df_tft_p.dtypes
