Note: Kaggle training set needs to be stored in **'./HPA_data/IF/data'**.

1.Run S1:

Download the IF images in **.jpg** from HPA database.

2.Run S2:

Download the cell masks of IF images from HPA database, and download IF images only in **.tif** format no **.jpg** format from database.

3.Run S3:

Convert IF images with rgb channels to IF images with single-channel.

4.Run S4:

Use [cell segmentation tool](https://github.com/CellProfiling/HPA-Cell-Segmentation) to segment Kaggle training data set and IF images in .tif format from HPA database to obtain cell masks.

5.Run S5:

Obtain single-cell images from IF images using cell masks and save cell images with the four channels in **'./HPA_data/Cell/data'** in **.npy** format.
Obtain index files (.csv) of single-cell images.
