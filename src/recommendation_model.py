import pandas as pd

class Recommend:
    def __init__(self, target_attr_table : pd.DataFrame, target_feature : str) -> None:
        """
        Descriptions
        ------------
        See the "workflow.drawio"

        Parameters
        ----------
        target_feature : we focus this feature which should be recommend target, ex: movie, sku, customers...

        target_attr_table : the attribution of target_feature should be groupby
                             in this term, we choose "mode" agg (don't use median, 
                             cause of the "label encoder features" will get "float" value)
                             (see the workflow.drawio)

        """
        self.target_feature = target_feature

        # groupy the target feature -> drop duplications
        self.target_attr_table = target_attr_table

        # keep the index-value (index - movie) which have been drop duplicated
        # that could let us "find the index" by item name in the "cosine_matrix"
        self.indices = pd.Series(self.target_attr_table.index, index=self.target_attr_table[target_feature])

    def cosine_similarity(self):
        """
        https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system

        Returns
        -------
        Consine similarity matrix for target feature.
        (m x m, m: class of target feature)

        """

        from sklearn.metrics.pairwise import cosine_similarity

        target_only_attr_table = self.target_attr_table.drop(self.target_feature, axis=1)

        cosine_sim = cosine_similarity(target_only_attr_table)

        return cosine_sim

