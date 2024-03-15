# Sequence-Based Prediction of ATP Binding Sites using Pre-Trained Language Models and Multi-Window Convolutional Neural Networks


## Abstract <a name="abstract"></a>
Identifying the binding sites of adenosine triphosphate (ATP) in proteins is critical for understanding binding mechanisms and functional roles. The increasing amount of protein sequence data necessitates computational methods for identifying binding sites. In this work, we will present a multi-window convolutional neural network model using pre-trained protein language model embeddings to predict ATP-binding residues from a protein sequence.

Adenosine triphosphate (ATP) plays a vital role in providing energy and enabling key cellular processes through interactions with binding proteins. However, experimental identification of ATP-binding residues remains challenging. To address the challenge, we developed a multi-window convolutional neural network (CNN) architecture taking pre-trained protein language model embeddings as input features. In particular, multiple parallel convolutional layers scan for motifs localized to different window sizes. Max pooling extracts salient features concatenated across windows into a final multi-scale representation for residue-level classification.

On benchmark ATP-binding protein datasets, our model achieves an AUC of 0.97, significantly improving on prior sequence-based models and outperforming CNN baselines. This demonstrates the utility of pre-trained language models and multi-window CNNs for advanced sequence-based prediction of ATP-binding residues. Our approach provides a promising new direction for elucidating binding mechanisms and interactions from primary structure.

<br>
![workflow](https://github.com/B1607/ATPirP/blob/aa273fb569ce027b34fac6a3ec3024f36d890ca4/Other/Figure.png)

## Dataset <a name="Dataset"></a>

| Dataset            | Protein Sequence | DNA Interacting Residues | Non-Interacting Residues |
|--------------------|------------------|--------------------------|--------------------------|
| Training data      | 388              | 5657                     | 142086                   |
| Independent data   | 41               | 681                      | 14152                    |
| Total              | 429              | 6338                     | 156238                   |


## Quick start <a name="quickstart"></a>

### Step 1: Generate Data Features

Navigate to the data folder and utilize the FASTA file to produce additional data features, saving them in the dataset folder.

Example usage:
```bash
python get_Binary_Matrix.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_mmseqs2.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_ProtTrans.py -in "Your FASTA file folder" -out "The destination folder of your output"
```
"Note: Ensure to update the path to your protein sequence database within get_mmseqs2.py as necessary."
### Step 2: Generate Dataset Using Data Features

Transition to the dataset folder and utilize the data features to produce a dataset.

Example usage:
```bash
python batch_get_series_feature.py -in "Your data feature Folder" -out "The destination folder of your output" -script get_series_feature.py -num 10 -old_ext "The data format of your data feature" -new_ext ".set" -w "num_dependent"
```
Alternative example:
```bash
python batch_get_series_feature.py -in Test -out Series11\ProtTrans\Test -script get_series_feature.py -num 10 -old_ext ".porttrans" -new_ext ".set" -w 5
```

### Step 3: Execute Prediction

Navigate to the code folder to execute the prediction.

Command-line usage:
```bash
python main.py -n_dep 5 -n_fil 256 -n_hid 1000 -bs 256 -ws 2 4 6 8 10 -nf 20 -e 20 -df "ProtTrans" -val "independent"
```
Alternatively, utilize the Jupyter notebook:
```bash
main.ipynb
```


