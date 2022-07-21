from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans, KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
import cv2
import joblib
from tqdm import tqdm
from Feature_Selector import Feature_Selector
from util import *
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


def label2nu(x, num1, num2):
    if str(x) == str(num1):
        return 0
    if str(x) == str(num2):
        return 1
    return 2


def label2nu3conbination(x, num1, num2, num3):
    if str(x) == str(num1):
        return 0
    if str(x) == str(num2):
        return 1
    if str(x) == str(num3):
        return 2
    return 3


def softmax(x):
    e_x = np.exp(x)
    return e_x / np.expand_dims(np.sum(e_x, axis=1), 1)


def main():
    n_components = [60]
    repeated = 1
    meta_dir = join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'data_csv')
    scv_df = pd.read_csv(join(meta_dir, 'SCV_notest_meta.csv'))
    scv_df['num_classes'] = scv_df['Label'].apply(lambda r: len(r.split(';')))
    scv_df = scv_df[scv_df['num_classes'] > 1]
    a = scv_df.Label.value_counts()
    label_com = a.index.to_list()

    for lc in tqdm(label_com):
        if len(lc.split(';')) == 2:
            num1, num2 = lc.split(';')
            lc = '|'.join(lc.split(';'))
            for r in range(repeated):
                for dim in n_components:
                    ms = StandardScaler()
                    outPath = './Cluster_results/scv/models'
                    normalPath = join(os.path.abspath(os.path.join(os.getcwd(), "./")), 'SLFs', 'normal')
                    out_dir_all = './Cluster_results/scv/update_results'
                    normalFilepath = [join(normalPath, f'{i}_features.csv') for i in [num1, num2]]
                    # data1 = pd.read_csv(outputFilepath, header=0)  # Load the train file into a dataframe
                    data = pd.concat([preprocessing_df(i) for i in normalFilepath], axis=0).reset_index(drop=True)
                    data.dropna(how='any', axis=1, inplace=True)
                    data_train = data.copy()

                    ####  feature selection  ###

                    out_select_path = join(outPath, f'{num1}|{num2}', 'FeatureSelect')
                    maybe_mkdir_p(out_select_path)

                    features_name = list(data_train.iloc[:, 2:].columns)
                    joblib.dump(features_name, join(out_select_path, f'raw_features_dimension{dim}_repeated{r}.pkl'))
                    # data_ = train.iloc[:, :2]
                    data_train[features_name] = ms.fit_transform(data_train[features_name])  # maxmin-score normal
                    joblib.dump(ms, join(out_select_path, f'StandardScaler_dimension{dim}_repeated{r}.pkl'))
                    data_train['Label_'] = data_train.Label.apply(lambda x: label2nu(x, num1, num2))

                    fs = Feature_Selector()
                    # select_fname = fs.chi2_selector(train, features_name, "label", i)
                    select_fname = fs.rfe_selector(data_train, features_name, "Label_", dim, scale=False)
                    joblib.dump(select_fname, join(out_select_path, f'select_features_dimension{dim}_repeated{r}.pkl'))

                    data_train_ = data_train.iloc[:, :2]
                    data_pca = data_train.loc[:, select_fname]
                    data_pca = pd.DataFrame(data_pca)
                    data_train = data_train_.join(data_pca, how='outer')

                    data_train.to_csv(join(out_select_path, f'RFE_SLFs_features_dimension{dim}_repeated{r}.csv'), index=False, na_rep='NaN')

                    #### clustering ###

                    data_train = pd.read_csv(join(out_select_path, f'RFE_SLFs_features_dimension{dim}_repeated{r}.csv'))
                    cluster_data = data_train.copy()
                    # x = random.sample(range(0, len(cluster_data)), 1000)
                    # cluster_data = cluster_data.iloc[x,:]
                    features_name = cluster_data.iloc[:, 2:].columns
                    target = cluster_data.Label.apply(lambda x: label2nu(x, num1, num2))
                    features = cluster_data.loc[:, features_name]

                    mbk = MiniBatchKMeans(init='k-means++', n_clusters=2, batch_size=100,
                                          n_init=10, max_no_improvement=10, verbose=0)
                    # mbk = KMeans(init='k-means++', n_clusters=4, n_init=10,  verbose=0, n_jobs=10)

                    mbk.fit(features)
                    y_predict = mbk.predict(features)
                    flag = MatchLabel(y_predict, target)
                    if flag != list(range(len(lc.split('|')))):
                        mbk.cluster_centers_ = mbk.cluster_centers_[flag, :]
                    print("k-means ARI ：", metrics.adjusted_rand_score(target, y_predict))
                    print("k-means NMI：",
                          metrics.adjusted_mutual_info_score(target, y_predict))
                    print("k-means homogeneity：", metrics.homogeneity_score(target, y_predict))
                    print("k-means completeness：", metrics.completeness_score(target, y_predict))
                    # cluster_data_new = cluster_data[['ID_idx', 'Label', 'New_Label']]
                    cluster_data['New_Label'] = y_predict
                    out_cluster_path = os.path.join(outPath, f'{num1}|{num2}', 'Cluster_SLFs')
                    maybe_mkdir_p(out_cluster_path)
                    joblib.dump(mbk, join(out_cluster_path, f'KMeans_model_dimension{dim}_repeated{r}.pkl'))
                    cluster_data.to_csv(join(out_cluster_path, f'Kmeans_dimension{dim}_repeated{r}.csv'), index=False, na_rep='NaN')

                    np.save(out_cluster_path + f'/Kmeans_center_dimension{dim}_repeated{r}.npy', mbk.cluster_centers_)

                    # The SLFs features of the cells have been saved in a csv format file
                    scv_df_path = join(os.path.abspath(os.path.join(os.getcwd(), "./")), 'SLFs', 'scv', 'SCV_all_features.csv')
                    df = pd.read_csv(scv_df_path)
                    df = df[df.Label.isin([lc])].reset_index(drop=True)
                    df['Label'] = df['Label'].apply(lambda x: str(x))

                    ##  SLF features  ##
                    raw_features = subfiles(out_select_path, prefix=f'raw_features_dimension{dim}')[0]
                    SLF_Features = load_pickle(raw_features)

                    ##  feature selection  ##
                    select_features = subfiles(out_select_path, prefix=f'select_features_dimension{dim}')[0]
                    select_features = load_pickle(select_features)
                    if type(select_features[0]) is not str:
                        select_features = list(map(lambda x: str(x), select_features))

                    ##  scale feature  ##
                    sc = subfiles(out_select_path, prefix=f'StandardScaler_dimension{dim}')[0]
                    sc = joblib.load(sc)

                    ##  SLF features's scale, deduction  ##
                    if type(SLF_Features[-1]) is not str:
                        SLF_Features = list(map(lambda x: str(x), SLF_Features))
                    df[SLF_Features] = sc.transform(df[SLF_Features])
                    df = df[['ID_idx'] + ['Label'] + select_features]
                    df.columns = ['ID_idx'] + ['Label'] + list(map(lambda x: str(x), select_features))

                    # single class #
                    Fileter_Data_df = subfiles(out_select_path, prefix=f'RFE_SLFs_features_dimension{dim}')[0]
                    # select_features = list(map(lambda x: str(x), select_features))
                    Data_single_df = pd.read_csv(Fileter_Data_df)[['ID_idx'] + ['Label'] + select_features]
                    Data_single_df['New_Label'] = Data_single_df.Label.apply(lambda x: label2nu(x, num1, num2))
                    Data_single_df[[f'kmeans_{INT_2_STR[int(num1)]}', f'kmeans_{INT_2_STR[int(num2)]}']] = 0
                    Data_single_df.loc[Data_single_df.Label.isin([num1]), f'kmeans_{INT_2_STR[int(num1)]}'] = 1
                    Data_single_df.loc[Data_single_df.Label.isin([num2]), f'kmeans_{INT_2_STR[int(num2)]}'] = 1

                    # multi class #
                    ##  SLF feature's scale, deduction  ##
                    Model_cluster = subfiles(out_cluster_path, prefix=f'KMeans_model_dimension{dim}')[0]
                    mbk = joblib.load(Model_cluster)
                    flag = MatchLabel(mbk.predict(Data_single_df[select_features]), Data_single_df['New_Label'])
                    if flag != list(range(len(lc.split('|')))):
                        mbk.cluster_centers_ = mbk.cluster_centers_[flag, :]

                    cluster_centers = mbk.cluster_centers_
                    # A is a matrix：distance of euclidean has been used
                    dist = cdist(df[select_features], cluster_centers, metric='euclidean')

                    target = df.Label.apply(lambda x: label2nu(x, num1, num2))
                    df['New_Label'] = target
                    label_cell = np.minimum(1 - softmax(dist), 1.0)
                    label_cell[np.where(target == 2)[0], :] = label_cell[np.where(target == 2)[0], :] * 2.0
                    label_cell = np.minimum(label_cell, 1.0)
                    label_cell = pd.DataFrame(label_cell,
                                              columns=[f'kmeans_{INT_2_STR[int(num1)]}', f'kmeans_{INT_2_STR[int(num2)]}'])
                    # cluster_multi_df = df.iloc[:, :2].join(label_cell, how='outer').sort_values(by=['ID_idx'])
                    cluster_multi_df = df.join(label_cell, how='outer').sort_values(by=['ID_idx'])

                    # Data_dfs = pd.concat([cluster_multi_df, Data_single_df], ignore_index=True)
                    # out_dir_all = join(outPath, 'summary_cluster')

                    out_path = join(out_dir_all, f'{num1}_{num2}')
                    maybe_mkdir_p(out_path)
                    # cluster_multi_df.to_csv(join(out_path, f'{lc}_dimension{dim}.csv'), index=False)

        elif len(lc.split(';')) == 3:
            num1, num2, num3 = lc.split(';')
            lc = '|'.join(lc.split(';'))
            for r in range(repeated):
                for dim in n_components:
                    ms = StandardScaler()
                    outPath = './Cluster_results/scv/models'
                    normalPath = join(os.path.abspath(os.path.join(os.getcwd(), "./")), 'SLFs', 'normal')
                    out_dir_all = './Cluster_results/scv/update_results'
                    normalFilepath = [join(normalPath, f'{i}_features.csv') for i in lc.split('|')]
                    # data1 = pd.read_csv(outputFilepath, header=0)  # Load the train file into a dataframe
                    data = pd.concat([preprocessing_df(i) for i in normalFilepath], axis=0).reset_index(drop=True)
                    data.dropna(how='any', axis=1, inplace=True)
                    data_train = data.copy()

                    # ####  feature selection  ###

                    out_select_path = join(outPath, f'{num1}|{num2}|{num3}', 'FeatureSelect')
                    maybe_mkdir_p(out_select_path)

                    features_name = list(data_train.iloc[:, 2:].columns)
                    joblib.dump(features_name, join(out_select_path, f'raw_features_dimension{dim}_repeated{r}.pkl'))
                    # data_ = train.iloc[:, :2]
                    data_train[features_name] = ms.fit_transform(data_train[features_name])  # maxmin-score normalization
                    joblib.dump(ms, join(out_select_path, f'StandardScaler_dimension{dim}_repeated{r}.pkl'))
                    # data = pd.DataFrame(data)
                    # data.columns = features_name
                    #
                    # data_train = data_.join(data, how='outer')
                    data_train['Label_'] = data_train.Label.apply(lambda x: label2nu3conbination(x, num1, num2, num3))

                    fs = Feature_Selector()
                    # select_fname = fs.chi2_selector(train, features_name, "label", i)
                    select_fname = fs.rfe_selector(data_train, features_name, "Label_", dim, scale=False)
                    joblib.dump(select_fname, join(out_select_path, f'select_features_dimension{dim}_repeated{r}.pkl'))

                    data_train_ = data_train.iloc[:, :2]
                    data_pca = data_train.loc[:, select_fname]
                    data_pca = pd.DataFrame(data_pca)
                    data_train = data_train_.join(data_pca, how='outer')

                    data_train.to_csv(join(out_select_path, f'RFE_SLFs_features_dimension{dim}_repeated{r}.csv'), index=False, na_rep='NaN')


                    #### clustering ###

                    data_train = pd.read_csv(join(out_select_path, f'RFE_SLFs_features_dimension{dim}_repeated{r}.csv'))
                    cluster_data = data_train.copy()
                    # x = random.sample(range(0, len(cluster_data)), 1000)
                    # cluster_data = cluster_data.iloc[x,:]
                    features_name = cluster_data.iloc[:, 2:].columns
                    target = cluster_data.Label.apply(lambda x: label2nu3conbination(x, num1, num2, num3))
                    features = cluster_data.loc[:, features_name]

                    mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=100,
                                          n_init=10, max_no_improvement=10, verbose=0)
                    # mbk = KMeans(init='k-means++', n_clusters=4, n_init=10,  verbose=0, n_jobs=10)

                    mbk.fit(features)
                    y_predict = mbk.predict(features)
                    flag = MatchLabel(y_predict, target)
                    if flag != list(range(len(lc.split('|')))):
                        mbk.cluster_centers_ = mbk.cluster_centers_[flag, :]
                    print("k-means ARI：", metrics.adjusted_rand_score(target, y_predict))
                    print("k-means NMI：",
                          metrics.adjusted_mutual_info_score(target, y_predict))
                    print("k-means homogeneity：", metrics.homogeneity_score(target, y_predict))
                    print("k-means completeness：", metrics.completeness_score(target, y_predict))
                    # cluster_data_new = cluster_data[['ID_idx', 'Label', 'New_Label']]
                    cluster_data['New_Label'] = y_predict
                    out_cluster_path = os.path.join(outPath, f'{num1}|{num2}|{num3}', 'Cluster_SLFs')
                    maybe_mkdir_p(out_cluster_path)
                    joblib.dump(mbk, join(out_cluster_path, f'KMeans_model_dimension{dim}_repeated{r}.pkl'))
                    cluster_data.to_csv(join(out_cluster_path, f'Kmeans_dimension{dim}_repeated{r}.csv'), index=False, na_rep='NaN')

                    np.save(out_cluster_path + f'/Kmeans_center_dimension{dim}_repeated{r}.npy', mbk.cluster_centers_)

                    # The SLFs features of the cells have been saved in a csv format file
                    scv_df_path = join(os.path.abspath(os.path.join(os.getcwd(), "./")), 'SLFs', 'scv', 'SCV_all_features.csv')
                    df = pd.read_csv(scv_df_path)
                    df = df[df.Label.isin([lc])].reset_index(drop=True)
                    df['Label'] = df['Label'].apply(lambda x: str(x))

                    ##  SLF features  ##
                    raw_features = subfiles(out_select_path, prefix=f'raw_features_dimension{dim}')[0]
                    SLF_Features = load_pickle(raw_features)

                    ##  features selection  ##
                    select_features = subfiles(out_select_path, prefix=f'select_features_dimension{dim}')[0]
                    select_features = load_pickle(select_features)
                    if type(select_features[0]) is not str:
                        select_features = list(map(lambda x: str(x), select_features))

                    ##  scale features  ##
                    sc = subfiles(out_select_path, prefix=f'StandardScaler_dimension{dim}')[0]
                    sc = joblib.load(sc)

                    ##  SLF features's scale, deduction  ##
                    if type(SLF_Features[-1]) is not str:
                        SLF_Features = list(map(lambda x: str(x), SLF_Features))
                    df[SLF_Features] = sc.transform(df[SLF_Features])
                    df = df[['ID_idx'] + ['Label'] + select_features]
                    df.columns = ['ID_idx'] + ['Label'] + list(map(lambda x: str(x), select_features))

                    # single class #
                    Fileter_Data_df = subfiles(out_select_path, prefix=f'RFE_SLFs_features_dimension{dim}')[0]
                    # select_features = list(map(lambda x: str(x), select_features))
                    Data_single_df = pd.read_csv(Fileter_Data_df)[['ID_idx'] + ['Label'] + select_features]
                    Data_single_df['New_Label'] = Data_single_df.Label.apply(lambda x: label2nu3conbination(x, num1, num2, num3))
                    Data_single_df[[f'kmeans_{INT_2_STR[int(num1)]}', f'kmeans_{INT_2_STR[int(num2)]}', f'kmeans_{INT_2_STR[int(num3)]}']] = 0
                    Data_single_df.loc[Data_single_df.Label.isin([num1]), f'kmeans_{INT_2_STR[int(num1)]}'] = 1
                    Data_single_df.loc[Data_single_df.Label.isin([num2]), f'kmeans_{INT_2_STR[int(num2)]}'] = 1
                    Data_single_df.loc[Data_single_df.Label.isin([num3]), f'kmeans_{INT_2_STR[int(num3)]}'] = 1

                    # multi class #
                    Model_cluster = subfiles(out_cluster_path, prefix=f'KMeans_model_dimension{dim}')[0]
                    mbk = joblib.load(Model_cluster)
                    flag = MatchLabel(mbk.predict(Data_single_df[select_features]), Data_single_df['New_Label'])
                    if flag != list(range(len(lc.split('|')))):
                        mbk.cluster_centers_ = mbk.cluster_centers_[flag, :]

                    cluster_centers = mbk.cluster_centers_
                    # A is a matrix：distance of euclidean is used
                    dist = cdist(df[select_features], cluster_centers, metric='euclidean')

                    target = df.Label.apply(lambda x: label2nu3conbination(x, num1, num2, num3))
                    df['New_Label'] = target
                    label_cell = np.minimum(1 - softmax(dist), 1.0)
                    label_cell[np.where(target == 3)[0], :] = label_cell[np.where(target == 3)[0], :] * 3.0
                    label_cell = np.minimum(label_cell, 1.0)
                    label_cell = pd.DataFrame(label_cell,
                                              columns=[f'kmeans_{INT_2_STR[int(num1)]}', f'kmeans_{INT_2_STR[int(num2)]}', f'kmeans_{INT_2_STR[int(num3)]}'])
                    # cluster_multi_df = df.iloc[:, :2].join(label_cell, how='outer').sort_values(by=['ID_idx'])
                    cluster_multi_df = df.join(label_cell, how='outer').sort_values(by=['ID_idx'])

                    # Data_dfs = pd.concat([cluster_multi_df, Data_single_df], ignore_index=True)
                    # out_dir_all = join(outPath, 'summary_cluster')

                    out_path = join(out_dir_all, f'{num1}_{num2}_{num3}')
                    maybe_mkdir_p(out_path)
                    cluster_multi_df.to_csv(join(out_path, f'{lc}_dimension{dim}.csv'), index=False)


if __name__ == '__main__':
    main()
