import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-in","--path_input",help="input your folder of fasta",type=str)
parser.add_argument("-out","--path_output",help="the path of output mmseqs2 without label",type=str)
parser.add_argument("-db_path","--database_path",help="input the path of your database" ,type=str,default="your database")
args=parser.parse_args()

import os
input=os.listdir(args.path_input)
args.path_output
tmp_dir=args.path_output+"/tmp"
finished_dir=args.path_output+"/finished"
processing_dir=args.path_output+"/processing"
mmseqs2_dir=args.path_output+"/mmseqs2"

db_path=args.database_path

os.system("mkdir "+tmp_dir)
os.system("mkdir "+finished_dir)
os.system("mkdir "+processing_dir)
os.system("mkdir "+mmseqs2_dir)

import numpy as np

for i in input:
    str=".fasta"
    if i.endswith(str):
        file_name="/"+i.split(".")[0]
        print(i)
        os.system("mmseqs createdb "+args.path_input+"/"+i+" "+processing_dir+file_name+".db")
        os.system("mmseqs search "+processing_dir+file_name+".db"+" "+db_path+" "+processing_dir+file_name+".outdb"+" tmp --num-iterations 3")
        os.system("mmseqs result2profile "+processing_dir+file_name+".db"+" "+db_path+" "+processing_dir+file_name+".outdb"+" "+processing_dir+file_name+".qp")
        os.system("mmseqs profile2pssm "+processing_dir+file_name+".qp"+" "+finished_dir+file_name+".mmseqpssm")

        f=np.genfromtxt(finished_dir+file_name+".mmseqpssm",delimiter="\t",skip_header=2)
        data = np.zeros((len(f), 20), dtype=np.float16 )
        m=0
        for row in f:
            data[m]=np.array(row[2:]).astype('float16')
            m+=1
        np.savetxt(mmseqs2_dir+"/"+file_name+".mmseqs2",data)
        print(i+"-------------done")
print("All the fasta file in the input folder are been transfer Successfully!!!")
