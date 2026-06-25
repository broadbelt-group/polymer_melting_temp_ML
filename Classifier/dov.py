import numpy as np
import pandas as pd

class DomainOfValidity:
    """
    DoV for defining applicability domain:
        mean Tanimoto to k nearest training neighbors >= threshold
    """
    def __init__(self, k=5, threshold_percentile=5, fp_bits=1024):
        self.k                    = k
        self.threshold_percentile = threshold_percentile
        self.fp_bits              = fp_bits
        self.train_fps            = None
        self.threshold_           = None
        self.loo_scores_          = None

    def _get_gen(self):
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        return GetMorganGenerator(radius=2, fpSize=self.fp_bits)

    def _smiles_to_fp(self, smiles_list):
        from rdkit import Chem
        gen = self._get_gen()
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(str(smi))
            fp  = (np.array(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
                   if mol is not None else np.zeros(self.fp_bits, dtype=np.float32))
            fps.append(fp)
        return np.array(fps)

    def _tanimoto_batch(self, query_fp, ref_fps):
        intersect = (query_fp * ref_fps).sum(axis=1)
        union     = (query_fp + ref_fps - query_fp * ref_fps).sum(axis=1)
        return intersect / (union + 1e-8)

    def fit(self, train_smiles):
        """
        Parameters
        ----------
        train_smiles : list of repeat unit SMILES strings
        """
        self.train_fps    = self._smiles_to_fp(train_smiles)

        loo_scores = []
        for i, fp in enumerate(self.train_fps):
            others = np.delete(self.train_fps, i, axis=0)
            sims   = self._tanimoto_batch(fp, others)
            top_k  = np.sort(sims)[::-1][:self.k]
            loo_scores.append(top_k.mean())

        self.loo_scores_ = np.array(loo_scores)
        self.threshold_  = np.percentile(self.loo_scores_, self.threshold_percentile)
        print(f"  DoV threshold (p{self.threshold_percentile}): {self.threshold_:.3f}")
        return self

    def score(self, query_smiles):
        """
        Returns
        -------
        scores         : float array, mean Tanimoto to k nearest neighbors
        in_domain_chem : bool array, chemistry threshold met
        """
        query_fps = self._smiles_to_fp(query_smiles)
        scores = []
        for fp in query_fps:
            sims  = self._tanimoto_batch(fp, self.train_fps)
            top_k = np.sort(sims)[::-1][:self.k]
            scores.append(top_k.mean())

        scores         = np.array(scores)
        in_domain_chem = scores >= self.threshold_
        return scores, in_domain_chem

    def score_df(self, query_smiles, query_names=None):
        scores, in_domain_chem = self.score(
            query_smiles
        )
        result = pd.DataFrame({
            "smiles":         query_smiles,
            "dov_score":      scores.round(4),
            "in_domain_chem": in_domain_chem,
            "reliability":    pd.cut(
                scores,
                bins=[0, 0.3, 0.5, 0.7, 1.01],
                labels=["Very low", "Low", "Moderate", "High"]
            ),
        })
        if query_names is not None:
            result.insert(0, "name", query_names)
        return result