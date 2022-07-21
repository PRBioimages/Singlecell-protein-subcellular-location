import pandas as pd
from utils.Evaluator import *


def validate_mAP(savePath, Dec_path, root_dec_mask, GT_path, root_gt_mask):
    def getBoundingBoxes(directory,
                         isGT,
                         allBoundingBoxes=None,
                         allClasses=None):
        """Read txt files containing bounding boxes (ground truth and detections)."""
        if allBoundingBoxes is None:
            allBoundingBoxes = BoundingBoxes()
        if allClasses is None:
            allClasses = []
        # Read ground truths
        files = pd.read_csv(directory)

        # Read GT detections from txt file
        # Each line of the files in the groundtruths folder represents a ground truth bounding box
        # (bounding boxes that a detector should detect)
        # Each value of each line is  "class_id, x, y, width, height" respectively
        # Class_id represents the class of the bounding box
        # x, y represents the most top-left coordinates of the bounding box
        # x2, y2 represents the most bottom-right coordinates of the bounding box

        for _, line in files.iterrows():
            if isGT:
                nameOfImage = line['ID']
                # idClass = int(splitLine[0]) #class
                idClass = line['Cell_Label']  # class
                idIndex = line['Cell_index']  # index
                # idGene = line['Gene']  # gene
                mask = os.path.join(root_gt_mask, nameOfImage + '_mask.png')
                mask = cv2.imread(mask, 0)
                mask = np.where(mask == idIndex)
                xmin, ymin = np.min(mask, axis=1)
                xmax, ymax = np.max(mask, axis=1)
                for c in idClass.split(';'):
                    c = int(c)
                    bb = BoundingBox(nameOfImage,
                                     c,
                                     xmin, ymin,
                                     xmax, ymax,
                                     typeCoordinates=CoordinatesType.Absolute,
                                     imgSize=None,
                                     bbType=BBType.GroundTruth,
                                     format=BBFormat.XYX2Y2)
                    allBoundingBoxes.addBoundingBox(bb)
                    if c not in allClasses:
                        allClasses.append(c)
            else:
                nameOfImage = line['ID']
                # idClass = int(splitLine[0]) #class
                # idClass = line['Cell_Label']  # class
                idIndex = line['Cell_index']  # index
                # idGene = line['Gene']  # gene
                mask = os.path.join(root_dec_mask, nameOfImage + '_mask.png')
                # mask = cv2.imread(mask, 0)
                # mask = np.where(mask == idIndex, 1, 0)
                mask = cv2.imread(mask, 0)
                mask = np.where(mask == idIndex)
                xmin, ymin = np.min(mask, axis=1)
                xmax, ymax = np.max(mask, axis=1)
                for c in range(19):
                    bb = BoundingBox(nameOfImage,
                                     c,
                                     xmin, ymin,
                                     xmax, ymax,
                                     typeCoordinates=CoordinatesType.Absolute,
                                     imgSize=None,
                                     bbType=BBType.Detected,
                                     classConfidence=line[str(c)],
                                     format=BBFormat.XYX2Y2)
                    allBoundingBoxes.addBoundingBox(bb)
                    if c not in allClasses:
                        allClasses.append(c)


        return allBoundingBoxes, allClasses

    # Get groundtruth boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(GT_path, True)

    # Get detected boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(Dec_path, False, allBoundingBoxes, allClasses)



    allClasses.sort()

    evaluator = Evaluator()
    acc_AP = 0
    validClasses = 0

    iouThreshold = 0.6


    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iouThreshold,  # IOU threshold
        # method=MethodAveragePrecision.EveryPointInterpolation,
        method=MethodAveragePrecision.ElevenPointInterpolation,
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
        savePath=savePath,
        showGraphic=False)
