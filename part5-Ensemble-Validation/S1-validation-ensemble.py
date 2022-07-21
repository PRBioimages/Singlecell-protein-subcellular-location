from utils.valid_models import *
from utils.valid_dataload import *
from utils.mAP_metrics import validate_mAP
from utils import parse_args
from batchgenerators.utilities.file_and_folder_operations import *
from tqdm import tqdm
import torch
import os


def main():
    cfg = parse_args()

    result_path = cfg.data.save_dir


    File_name = __file__.split("/")[-1].split(".")[0]
    print('[!] Runing', File_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### In order to facilitate the experiment, the test set obtained the mask through the HPA segmentation tool in advance,
    # and the number of cells in each image was counted in the .csv file.
    meta_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../")), 'data_csv')
    Cell_df_path = join(meta_dir, 'RandomSelect_MamualReset.csv')
    Cell_df = pd.read_csv(Cell_df_path)
    label_cols = list(range(19))
    for i in label_cols:
        Cell_df[i] = 0


    root_mask = cfg.data.root_mask
    root_data = cfg.data.root_data
    if len(subfiles(root_mask)) < 328:
        raise Exception('Mask of test IF images should be pre-segmented in dir:', root_mask)



    inference_model_MIL1 = []
    cfg.model.name = cfg.MILmodel.name1
    print('IMAGE_Model_name:', cfg.model.name)
    model = get_model(cfg).to(device)
    load_matched_state(model, torch.load(cfg.MILmodel.path1))
    _ = model.eval()
    inference_model_MIL1.append(model)

    inference_model_MIL2 = []
    cfg.model.name = cfg.MILmodel.name2
    print('IMAGE_Model_name:', cfg.model.name)
    model = get_model(cfg).to(device)
    load_matched_state(model, torch.load(cfg.MILmodel.path2))
    _ = model.eval()
    inference_model_MIL1.append(model)

    inference_model_cell = []
    cfg.model.name = cfg.CellModel.name
    print('CELL_Model_name:', cfg.model.name)
    model = get_model(cfg).to(device)
    load_matched_state(model, torch.load(cfg.CellModel.path))
    _ = model.eval()
    inference_model_cell.append(model)


    IndFirst = 0
    hpadataset = TestDataset_ensemble(Cell_df, root_mask, root_data)
    dl = torch.utils.data.DataLoader(hpadataset, shuffle=False, batch_size=1, num_workers=2)
    for ipts_img, ipts_cell, ID, num in tqdm(dl, total=len(dl)):
        ipts_img = ipts_img[0].to(device)
        ipts_cell = ipts_cell[0].to(device)

        assert ID[0] == Cell_df.loc[IndFirst, 'ID'], print('%s,ID is not match,%s!==%s' %
                                                                 (ID[0], ID[0], Cell_df.loc[IndFirst, 'ID']))

        res = []
        exp = []
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for model in inference_model_MIL1:
                    ifr = model(ipts_img, num)
                    res.append(ifr[0].float())
                    exp.append(ifr[1].float())
                for model in inference_model_MIL2:
                    ifr = model(ipts_img, num)
                    res.append(ifr[0].float())
                    exp.append(ifr[1].float())

                res_cell = [model(ipts_cell).float() for model in inference_model_cell]

        predict_p = np.stack([torch.sigmoid(r.cpu()) for r in res]).mean(0)
        exp_p = np.stack([torch.sigmoid(e.cpu()) for e in exp]).mean(0)

        predict_p_cell = np.stack([torch.sigmoid(r.cpu()) for r in res_cell]).mean(0)
        batch_o_preds = (predict_p * cfg.MILmodel.frac + predict_p_cell * cfg.CellModel.frac) * exp_p
        # batch_o_preds = (predict_p) * exp_p
        # batch_o_preds = predict_p

        IndLast = IndFirst + len(batch_o_preds)
        Cell_df.loc[IndFirst:IndLast-1, label_cols] = batch_o_preds
        IndFirst = IndLast


    save_path = join(result_path, 'ensemble',
                     'Pseudo-max-attent-cell')
    print('Saving config.json in', save_path)


    os.makedirs(save_path, exist_ok=True)
    cfg.dump_json(join(save_path, 'config.json'))
    confidence_path = join(save_path + '/confidence_cell.csv')
    Cell_df.to_csv(confidence_path, index=False)

    GT_path = join(meta_dir, 'MamualReset.csv')
    root_gt_mask = './test_data/Ground_true/MaskMutual'

    print('Waiting for Calculating mAP')
    validate_mAP(save_path, confidence_path, root_mask, GT_path, root_gt_mask)
    print('Saving mAP Results in', save_path)


if __name__ == '__main__':
    main()

