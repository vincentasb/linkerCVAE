import re
import numpy as np
from numpy import ndarray


AMINO_ACID_SET = np.asarray(list("TSPGDKQNAVERI"))


class OHAAE():
    num_layers = 1
    embed_dimension = len(AMINO_ACID_SET)
    embed_name = "one_hot_aa_encoding"

    def embbed_sequence(self, sequence: str) -> ndarray:
        if not sequence:
            return np.zeros((len(AMINO_ACID_SET), 0))
        sequence = re.sub(r"[UZOB]", "X", sequence)
        one_hot_encoded = [AMINO_ACID_SET == aa for aa in sequence]
        return np.stack(one_hot_encoded).astype(np.float32).flatten()

