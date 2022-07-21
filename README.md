# Singlecell-protein-subcellular-location
This is a deep learning-based protein subcellular localization pipline for predicting protein subcellular locations on single cells in immunofluorescence images.
## 1 part1 preprocessing
Kaggle training set can be accessed by [link](https://www.kaggle.com/competitions/hpa-single-cell-image-classification/data?select=train). Run the code in this folder setp by step to obtain the IF data in the HPA database and obtain the index file (*.csv*) of the data, all data are stored in the folder **HPA_data**, and all processed index files are stored in the folder **data_csv**.
## 2 part2 MIL
Multi-instance learning model based on IF images.Run ***main.py*** to get image-based model. Multiple parameters for model training are saved in the configuration file **./configs/config.yaml**. Before running the code, add the absolute path of the directory of the cell images to the variable **data:data_root** in the configuration file. Modify the parameter **model:name** in the configuration file to get different MIL models. After running part 3, the value 'imagelabel' of  variable **data:celllabel** in configuration file can be modified to 'pseudolabel' to strengthen the MIL model.
## 3 part3 AssignedPseudoLabel
Run ***./Clustering/clustering_main.py*** to get the pseudo-label by clustering method, and run codes in ./Heuristic step by step to get the pseudo-label by heuristic method. Run ***./S1_CombinedPseudo.py*** to get the cell label for cell-based model (part4)
## 4 part4 Cellmodel
Cell-based model based on cell images and pseudo-label (part3) .Run ***main.py*** to get cell-based model. Multiple parameters for model training are saved in the configuration file **./configs/config.yaml**. Before running the code, add the absolute path of the directory of the cell images to the variable **data:data_root** in the configuration file.
## 5 part5 Ensemble-Validation
Run ***S1-validation-ensemble.py*** to test the performance of ensemble model on manual test set. Multiple parameters for model validation are saved in the configuration file **./configs/config.yaml**. Before running the code, add the absolute path of the directory of the test images to  **data:root_data**  and path of model's weight to **MILmodel:pth_path** and **CellModel:pth_path** in the configuration file.
Validation of the ensemble model on the Kaggle test set is available on the Kaggle platform.
