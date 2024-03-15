import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-in","--path_input", type=str, help="the path of input FASTA file")
parser.add_argument("-out","--path_output", type=str, help="the path of output Binary matrix file")

def read_fasta(fasta_file):
    seq = ""
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                continue
            else:
                seq += line.strip()
    #print(seq)
    return seq

def gen_one_hot(seq):
    one_hot_matrix = np.zeros((len(seq),20))
    A = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    C = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    D = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    E = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    F = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    G = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    H = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
    I = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    K = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
    L = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
    M = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
    N = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
    P = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
    Q = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
    R = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
    S = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
    T = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
    V = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
    W = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
    Y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]

    for idx, ele in enumerate(seq):
        if ele == 'A':
            one_hot_matrix[idx] = A
        elif ele == 'C':
            one_hot_matrix[idx] = C
        elif ele == 'D':
            one_hot_matrix[idx] = D
        elif ele == 'E':
            one_hot_matrix[idx] = E
        elif ele == 'F':
            one_hot_matrix[idx] = F
        elif ele == 'G':
            one_hot_matrix[idx] = G
        elif ele == 'H':
            one_hot_matrix[idx] = H
        elif ele == 'I':
            one_hot_matrix[idx] = I
        elif ele == 'K':
            one_hot_matrix[idx] = K
        elif ele == 'L':
            one_hot_matrix[idx] = L
        elif ele == 'M':
            one_hot_matrix[idx] = M
        elif ele == 'N':
            one_hot_matrix[idx] = N
        elif ele == 'P':
            one_hot_matrix[idx] = P
        elif ele == 'Q':
            one_hot_matrix[idx] = Q
        elif ele == 'R':
            one_hot_matrix[idx] = R
        elif ele == 'S':
            one_hot_matrix[idx] = S
        elif ele == 'T':
            one_hot_matrix[idx] = T
        elif ele == 'V':
            one_hot_matrix[idx] = V
        elif ele == 'W':
            one_hot_matrix[idx] = W
        elif ele == 'Y':
            one_hot_matrix[idx] = Y

    return one_hot_matrix

def save_binary_matrix(matrix, output_path):
    np.savetxt(output_path, matrix, fmt="%.3f")

def main(fasta_file, output_file):
    seq = read_fasta(fasta_file)
    one_hot_matrix = gen_one_hot(seq)
    save_binary_matrix(one_hot_matrix, output_file)
    
import os
if __name__ == "__main__":
    args = parser.parse_args()
    input=os.listdir(args.path_input)
    str=".fasta"
   
    for i in input:
        if i.endswith(str):
            file_name="/"+i.split(".")[0]
            print(args.path_input+"/"+i)
            
            main(args.path_input+"/"+i, args.path_output+file_name+".binary")
    