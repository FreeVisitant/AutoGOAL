from autogoal.kb import (
    AlgorithmBase,
    Seq,
    Sentence,
    Supervised,
    VectorCategorical,
    MatrixContinuousSparse,
)
from autogoal.grammar import BooleanValue
from autogoal_sklearn._generated import TfidfVectorizer

class MyTfidfVectorizerBridge(AlgorithmBase):

    @classmethod
    def is_upscalable(cls) -> bool:
        return False

    def __init__(
        self,
        lowercase:    BooleanValue(),
        binary:       BooleanValue(),
        use_idf:      BooleanValue(),
        smooth_idf:   BooleanValue(),
        sublinear_tf: BooleanValue()
    ):
        self.lowercase    = lowercase
        self.binary       = binary
        self.use_idf      = use_idf
        self.smooth_idf   = smooth_idf
        self.sublinear_tf = sublinear_tf

        self.vec = TfidfVectorizer(
            lowercase=self.lowercase,
            binary=self.binary,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
        )

    def run(
        self,
        X: Seq[Sentence],
        Y: Supervised[VectorCategorical]
    ) -> (MatrixContinuousSparse, Supervised[VectorCategorical]):
        X_vec = self.vec.run(X)
        return (X_vec, Y)
