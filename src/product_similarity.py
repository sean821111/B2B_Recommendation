import pandas as pd
import numpy as np 
import csv
import data_cleanning as dc
import classification

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
        SKUs = df["WTPARTNUMBER"]

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
        dfMatrix = df.to_numpy()
        catColumnsPos = [df.columns.get_loc(col) for col in list(df.select_dtypes('object').columns)]

        Xnum, Xcat = self._split_num_cat(dfMatrix, catColumnsPos)

        gamma = 0.5 * np.mean(Xnum.std(axis=0))

        num_dissim = self.euclidean_dissim(sku, Xnum)
        cat_dissim = self.matching_dissim(sku, Xcat)

        dissim = num_dissim + gamma* cat_dissim
        sortedIdx = np.argsort(dissim)

        sortedSKUs = np.empty(len(self.SKUs),  dtype='object')
        for i,v in enumerate(sortedIdx):
            sortedSKUs[v] = self.SKUs[i]
        return sortedSKUs
    def batch_similarItems(self, topN=10):
        for sku in self.SKUs:
            topSimilarItems = self.similarItems(sku)
            
            return topSimilarItems[0:topN]
            
if __name__ == '__main__':
    filePath = "./data/CELL_LCM_TP.xlsx"
    data = pd.read_excel(filePath)
    
    # combine LCM CELL TP data 

    
    data_cleaning = dc.DataCleaning(data)
    df = data_cleaning.encoding()
    
    dissim = Dissimilarity(df)
    
    # single SKU 
    dissim.similarItems("010GPW2-900001-PX")
    
    