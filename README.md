# stVAT: Spatial Domains and Spatially Variable Genes Identification with Functional Analysis Based on the VAEViT Algorithm

## Requirements
- Python==3.10  
- PyTorch==2.4.1+cu121  
- CUDA==12.1  
- scanpy==1.10.3  
- anndata==0.10.9  
- numpy==2.0

## Dataset
- [STOMICS DataBase - PDAC](https://db.cngb.org/stomics/search?query=PDAC)  
- [Spatial Research - DOI: 10.1158/0008-5472.CAN-18-0747](https://www.spatialresearch.org/resources-published-datasets/doi-10-1158-0008-5472-can-18-0747/)  
- [10x Genomics - Human Breast Cancer (Block A Section 1)](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0)  
- [10x Genomics Support - V1 Human Invasive Ductal Carcinoma](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.2.0/V1_Human_Invasive_Ductal_Carcinoma)  

---

## Run
To run the analysis using the provided datasets, navigate to the project directory and execute the following command:

```bash
python testMEL.py
