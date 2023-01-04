import pandas as pd
import numpy as np 
import data_cleanning as dc
import classification
from recommendation_model import Recommend
import json
from multiprocessing.dummy import Pool as ThreadPool

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


    def similarItems(self, skuIndex):
        dfMatrix = df.to_numpy()
        catColumnsPos = [df.columns.get_loc(col) for col in list(df.select_dtypes('object').columns)]


        Xnum, Xcat = self._split_num_cat(dfMatrix, catColumnsPos)

        gamma = 0.5 * np.mean(Xnum.std(axis=0))


        num_dissim = self.euclidean_dissim(Xnum[skuIndex], Xnum)
        cat_dissim = self.matching_dissim(Xcat[skuIndex], Xcat)


        dissim = num_dissim + gamma * cat_dissim
        sortedIdx = np.argsort(dissim)

        sortedSKUs = np.empty(len(self.SKUs),  dtype='object')
        for i,v in enumerate(sortedIdx):
            sortedSKUs[v] = self.SKUs[i]
        return sortedSKUs

    @staticmethod
    def batch_similarItems(df, topN=15):
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
        skus_similar = {}

        # separte to tdtdisplay, ... (total 7 types)
        *lcm_cell_tp, solution_df, solution_hannspree_df, hannspree_df = classification.main()

        def content_based(df : pd.DataFrame,
                        target_feature : str="WTPARTNUMBER"):

            Recom = Recommend(df, target_feature)

            # using the target_attr_table to calculate the target similarity
            cosine_sim = Recom.cosine_similarity()

            sku = df[target_feature]
            df_simMatrix = pd.DataFrame(cosine_sim, columns=sku)
            df_simMatrix["SKU"] = sku

            return df_simMatrix

        with ThreadPool(12) as pool:
            df_tft_p, df_tft_c, df_paper_p, df_paper_c = pool.map(content_based, [pd.merge(df, tmp, how="inner", on="WTPARTNUMBER") for tmp in lcm_cell_tp])

        magento_skus = set(df_tft_p["SKU"])|\
                        set(df_tft_c["SKU"])|\
                        set(df_paper_p["SKU"])|\
                        set(df_paper_c["SKU"])|\
                        set(solution_df["WTPARTNUMBER"])|\
                        set(solution_hannspree_df["WTPARTNUMBER"])|\
                        set(hannspree_df["WTPARTNUMBER"])

        def helper(df_type, type_, sku):
            if sku not in skus_similar: skus_similar[sku] = {}

            if sku in df_type['SKU'].values:
                sku_similar = df_type.sort_values(by=[sku], ascending=[False])[["SKU", sku]]
                sku_list = sku_similar["SKU"].iloc[:topN].to_list()

                # keep input_sku be first element
                sku_list.remove(sku)
                sku_list.insert(0, sku)

                skus_similar[sku][type_] = sku_list
            else:
                skus_similar[sku][type_] = []

        def helper_rule(df_type, type_):
            for sku in magento_skus:
                if sku in df_type["WTPARTNUMBER"].values:
                    if df_type.shape[0] > topN:
                        skus_similar[sku][type_] = pd.read_csv(f"../data/PLM/{type_}.csv")["WTPARTNUMBER"].iloc[:topN].to_list()
                    else:
                        skus_similar[sku][type_] = pd.read_csv(f"../data/PLM/{type_}.csv")["WTPARTNUMBER"].to_list()
                else:
                    skus_similar[sku][type_] = []

        def rule_thread(sku):
            helper(df_tft_p, "tftdisplay_Preferred", sku)
            helper(df_tft_c, "tftdisplay_Custom", sku)
            helper(df_paper_p, "paperdisplay_Preferred", sku)
            helper(df_paper_c, "paperdisplay_Custom", sku)
        
        with ThreadPool(16) as pool:
            pool.map(rule_thread, [sku for sku in magento_skus])
        
        helper_rule(solution_df, "systemBoard")
        helper_rule(solution_hannspree_df, "solution_hannspree")
        helper_rule(hannspree_df, "hannspree")

        return skus_similar

if __name__ == '__main__':
    # (LCM CELL TP) data
    filePath = "../data/PLM/CELL_LCM_TP.xlsx"
    data = pd.read_excel(filePath)

    # clean and encode
    data_cleaning = dc.DataCleaning(data)
    df = data_cleaning.encoding()

    for col in df.columns.drop("WTPARTNUMBER"):
        df[col] = df[col].astype(float)

    skus_similar = Dissimilarity.batch_similarItems(df)

    with open("../data/PLM/SKU_Similar_thread.json", "w") as outfile:
        json.dump(skus_similar, outfile)
