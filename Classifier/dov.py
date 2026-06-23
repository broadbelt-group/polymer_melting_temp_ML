import numpy as np
import pandas as pd

class DomainOfValidity:
    """
    DoV with two-level applicability domain:
      Level 1 (chemistry): mean Tanimoto to k nearest training neighbors >= threshold
      Level 2 (stereo):    query stereo class was seen in training set
    
    in_domain_chem  : chemistry only (same as V1)
    in_domain_full  : chemistry AND stereo class both covered
    """
    def __init__(self, k=5, threshold_percentile=5, fp_bits=1024):
        self.k                    = k
        self.threshold_percentile = threshold_percentile
        self.fp_bits              = fp_bits
        self.train_fps            = None
        self.train_stereo_        = None
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

    def fit(self, train_smiles, train_stereo):
        """
        Parameters
        ----------
        train_smiles : list of repeat unit SMILES strings
        train_stereo : list of stereo class strings (e.g. 'isotactic')
        """
        self.train_fps    = self._smiles_to_fp(train_smiles)
        self.train_stereo_ = set(str(s).strip().lower() for s in train_stereo)

        loo_scores = []
        for i, fp in enumerate(self.train_fps):
            others = np.delete(self.train_fps, i, axis=0)
            sims   = self._tanimoto_batch(fp, others)
            top_k  = np.sort(sims)[::-1][:self.k]
            loo_scores.append(top_k.mean())

        self.loo_scores_ = np.array(loo_scores)
        self.threshold_  = np.percentile(self.loo_scores_, self.threshold_percentile)
        print(f"  DoV threshold (p{self.threshold_percentile}): {self.threshold_:.3f}")
        # print(f"  Training self-similarity: "
        #       f"{self.loo_scores_.mean():.3f} ± {self.loo_scores_.std():.3f}")
        # print(f"  Stereo classes in training: {sorted(self.train_stereo_)}")
        return self

    def score(self, query_smiles, query_stereo):
        """
        Returns
        -------
        scores         : float array, mean Tanimoto to k nearest neighbors
        in_domain_chem : bool array, chemistry threshold met
        in_domain_full : bool array, chemistry AND stereo both covered
        stereo_covered : bool array, stereo class seen in training
        """
        query_fps = self._smiles_to_fp(query_smiles)
        scores = []
        for fp in query_fps:
            sims  = self._tanimoto_batch(fp, self.train_fps)
            top_k = np.sort(sims)[::-1][:self.k]
            scores.append(top_k.mean())

        scores         = np.array(scores)
        in_domain_chem = scores >= self.threshold_
        stereo_covered = np.array([
            str(s).strip().lower() in self.train_stereo_
            for s in query_stereo
        ])
        in_domain_full = in_domain_chem & stereo_covered
        return scores, in_domain_chem, in_domain_full, stereo_covered

    def score_df(self, query_smiles, query_stereo, query_names=None):
        scores, in_domain_chem, in_domain_full, stereo_covered = self.score(
            query_smiles, query_stereo
        )
        result = pd.DataFrame({
            "smiles":         query_smiles,
            "stereo":         query_stereo,
            "dov_score":      scores.round(4),
            "in_domain_chem": in_domain_chem,
            "stereo_covered": stereo_covered,
            "in_domain_full": in_domain_full,
            "reliability":    pd.cut(
                scores,
                bins=[0, 0.3, 0.5, 0.7, 1.01],
                labels=["Very low", "Low", "Moderate", "High"]
            ),
        })
        if query_names is not None:
            result.insert(0, "name", query_names)
        return result