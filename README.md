# Unleashing the Power of LLMs as Multi-Modal Encoders for Text and Graph-Structured Data

This repository is the official implementation of [Unleashing the Power of LLMs as Multi-Modal Encoders for Text and Graph-Structured Data](https://arxiv.org/pdf/2410.11235). 

## Get Started
### KG-Contextualized QA
Run the following code
```bash
bash scripts/train/mistral/qa/dual_view/run_dual_view__csqa.sh

bash scripts/train/mistral/qa/dual_view/run_dual_view__obqa.sh

bash scripts/train/mistral/qa/dual_view/run_dual_view__medqa_usmle.sh
```

Note that, you should modify the max length in line 1114 of file `utils/data_utils.py`, 128 for CSQA and OBQA, 256 for MedQA.

### Graph-Text Pair Classification
Run the code
```bash
bash scripts/train/mistral/pair_classification/dual_view/run_dual_view.sh
```

### Retrieval
Run the code
```bash
bash scripts/train/mistral/retrieval/dual_view/run_dual_view__fiqa.sh

bash scripts/train/mistral/retrieval/dual_view/run_dual_view__scifact.sh
```

## Results
See more experimental details in our paper.
### KG-Contextualized QA
#### Test accuracy comparison on CommonsenseQA
| Methods                  | Test Accuracy     |
|--------------------------|-------------------|
| **Language Models Only** |                   |
| RoBERTa-Large      | 68.69 ± 0.56      |
| E5-Mistral         | 69.49 ± 0.28      |
| **LM + KG**              |                   |
| RGCN           | 68.41 ± 0.66      |
| GconAttn            | 68.59 ± 0.39      |
| KagNet              | 69.01 ± 0.22      |
| RN                 | 69.08 ± 0.91      |
| MHGRN                | 71.11 ± 0.10      |
| QA-GNN               | 73.41 ± 0.92      |
| GreaseLM             | 74.20 ± 0.40      |
| **Janus (Ours)**         | **81.09 ± 0.73**  |

#### Test accuracy comparison on OpenBookQA
| Methods                                  | Test Accuracy      |
|------------------------------------------|--------------------|
| **Language Models Only**                 |                    |
| RoBERTa-Large                        | 64.80 ± 2.37       |
| AristoRoBERTa                         | 78.40 ± 1.64       |
| E5-Mistral                          | 74.80 ± 0.35       |
| **LM + KG + Scientific Facts**           |                    |
| RGCN                                 | 74.60 ± 2.53       |
| GconAttn                            | 71.80 ± 1.21       |
| RN                                  | 75.35 ± 1.39       |
| MHGRN                                | 81.87 ± 1.86       |
| QA-GNN                               | 82.77 ± 1.56       |
| GreaseLM                             | 83.87 ± 1.29       |
| **Janus (Ours)**                         | 86.67 ± 1.10       |
| **Janus (Ours) + Scientific Facts**      | **93.33 ± 0.42**   |

####  Test accuracy comparison on MedQA-USMLE
| Methods            | Test Accuracy |
|--------------------|---------------|
| **Language Models Only**            |               |
| BERT-Base          | 34.3          |
| BioBERT-Base       | 34.1          |
| RoBERTa-Large      | 35            |
| BioBERT-Large      | 36.7          |
| SapBERT            | 37.2          |
| E5-Mistral         | 38.3          |
| **LM + KG**                         |               |
| QA-GNN             | 38            |
| GreaseLM           | 38.5          |
| **Janus (Ours)**   | 49.9 ± 0.88   |


### Graph-Text Pair Classification
#### Test accuracy comparison on WebNLG dataset
| Methods        | Test Accuracy    |
|----------------|------------------|
| **LM + KG**    |                  |
| RGCN           | 63.20 ± 0.49     |
| MHGRN          | 84.98 ± 0.53     |
| QA-GNN         | 75.55 ± 3.54     |
| GreaseLM       | 82.50 ± 4.29     |
| **Janus (Ours)** | **88.43 ± 1.33** |


### Retrieval
####  Comparision results (NDCG@10) on retrieval tasks
| Methods           | SciFact | FIQA  |
|-------------------|---------|-------|
| **Language Models Only** |         |       |
| BERT-Base         | 13.3    | 2.2   |
| RoBERTa-Large     | 43.3    | 20.4  |
| E5-Small          | 65.6    | 34.8  |
| E5-Base           | 73.1    | 36.4  |
| E5-Large          | 72.6    | 38.6  |
| GTR-XXL           | 66.2    | 46.7  |
| SGPT              | 74.7    | 37.2  |
| E5-Mistral        | 76.1    | 53.5  |
| **LM + KG**       |         |       |
| QA-GNN            | 41.4    | 19.5  |
| GreaseLM          | 48.9    | 29.3  |
| **Janus (Ours)**  | 82.9 ± 1.9 | 54.1 ± 0.1 |


## Acknowledgement
* Our implementation is partly based on Michihiro's [code](https://github.com/michiyasunaga/qagnn).

## Contact
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.