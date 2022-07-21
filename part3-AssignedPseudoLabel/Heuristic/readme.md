Run S1:

Obtain cell confidence in each subcellular location class. Multiple parameters for model are saved in the configuration file **./configs/config.yaml**. 
Before running the code, add the absolute path of the directory of the IF images and masks to the parameter **data:root_data** and **data:root_mask**  in the configuration file, respectively. 
Modify the parameter **MILmodel:name**, and add path of model's weight to **MILmodel:pth_path** in the configuration file to get different MIL models. 

Run S2:

Assign pseudo-labels to each cell using heuristics.
