# Define the codon table
codon_table = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'N': ['AAT', 'AAC'],
    'D': ['GAT', 'GAC'],
    'C': ['TGT', 'TGC'],
    'Q': ['CAA', 'CAG'],
    'E': ['GAA', 'GAG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'K': ['AAA', 'AAG'],
    'M': ['ATG'],
    'F': ['TTT', 'TTC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'W': ['TGG'],
    'Y': ['TAT', 'TAC'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    '*': ['TAA', 'TAG', 'TGA']
}

illegal_sequences = ["GAATTC", "TCTAGA", "ACTAGT", "CTGCAG", "GCGGCCGC"]

def contains_illegal_sequence(dna_sequence):
    for illegal in illegal_sequences:
        if illegal in dna_sequence:
            return True
    return False

def reverse_translate(aa_sequence):
    dna_sequence = ''
    for aa in aa_sequence:
        codons = codon_table[aa]
        for codon in codons:
            if not contains_illegal_sequence(dna_sequence + codon):
                dna_sequence += codon
                break
    return dna_sequence
