from utils import *
from utils.valid_models import *
from utils.valid_dataload import *
from batchgenerators.utilities.file_and_folder_operations import *
from tqdm import tqdm
import torch
import pandas as pd


def get_bbox(mask):
    coor = []
    for i in range(1, mask.max()+1):
        m = np.where(mask == i)
        xmin, ymin = np.min(m, axis=1)
        xmax, ymax = np.max(m, axis=1)
        # coor.append([xmin, ymin, xmax, ymax])
        coor.append([ymin, xmin, ymax+1, xmax+1])
    return sorted(coor, key=lambda x: (x[1], x[0], x[3], x[2]))


def main():
    cfg = parse_args()

    Result_TestDir = './Results'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CELL_Model_DIR = cfg.MILmodel.path1

    print('CELL_Model_DIR:', CELL_Model_DIR)

    meta_dir = join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'data_csv')
    Cell_df_path = join(meta_dir, 'SCV_notest.csv')
    Cell_df = pd.read_csv(Cell_df_path)
    Cell_df = Cell_df.drop_duplicates('ID', keep='last', ignore_index=True).reset_index(drop=True)


    TEST_IMG_DIR = cfg.data.root_data
    root_mask = cfg.data.root_mask
    modelname = cfg.MILmodel.name1

    cfg.model.name = modelname



    label_cols = list(range(19))
    for i in label_cols:
        Cell_df[i] = 0


    inference_model = get_model(cfg).to(device)
    load_matched_state(inference_model, torch.load(CELL_Model_DIR, map_location='cpu'))
    _ = inference_model.eval()

    valid_ds = HPADataSET(df=Cell_df, TEST_IMG_DIR=TEST_IMG_DIR, root_mask=root_mask)
    valid_onlyclassor = DataLoader(dataset=valid_ds, batch_size=1, shuffle=False,drop_last=True, num_workers=6, pin_memory=True)

    df = pd.DataFrame()
    for (img, indx, cnt) in tqdm(valid_onlyclassor, total=len(valid_onlyclassor)):
        ID = Cell_df.loc[indx, 'ID'].values[0]
        l = Cell_df.loc[indx, 'Label'].values[0]
        img = img.view(-1, img.shape[-3], img.shape[-2], img.shape[-1])

        # temp = pd.Series([ID, indx]+[0 for i in range(19)], index=index)
        batch_o_preds = []
        with torch.no_grad():
        # with torch.cuda.amp.autocast():
            cell, exp = inference_model(img.to(device), cnt)
            predict_p = torch.sigmoid(cell.cpu())
            exp_p = torch.sigmoid(exp.cpu())
            batch_o_preds.append(predict_p*exp_p)
        batch_o_preds = np.stack(batch_o_preds)
        batch_o_preds = np.mean(batch_o_preds, axis=0)
        # l = list(map(lambda x: int(x), l.split('|')))
        batch_o_preds = np.minimum(batch_o_preds, 1)
        batch_o_preds = np.concatenate([np.array([ID]*cnt).reshape(-1,1), np.arange(cnt).reshape(-1,1), np.array([l]*cnt).reshape(-1,1),batch_o_preds], axis=1)
        df1 = pd.DataFrame(batch_o_preds, columns=['ID', 'idx', 'ImageLabel'] + [f'class{i}' for i in range(19)])
        df = df.append(df1)

    maybe_mkdir_p(Result_TestDir)
    df.to_csv(join(Result_TestDir, f'scv_heuristic.csv'), index=False)


if __name__ == '__main__':
    main()

