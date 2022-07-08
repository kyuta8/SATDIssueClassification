import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN


class DataProcessing(object):

    def __init__(self) -> None:
        pass
    
    def undersampling(self, train_x, train_label, counts=True, ratio=0.1, n_ratio=50, p_ratio=1):
        """
            Undersampling : This function is to reduce negative data in imbalanced data using randam sampling
        """
        if counts:
            num = np.sum(train_label)
        else:
            num = len(train_label) - np.sum(train_label)
        # rus = RandomUnderSampler(ratio='majority', random_state=42)
        # rus = RandomUnderSampler(sampling_strategy={0:int(num*n_ratio), 1:int(num*p_ratio)}, random_state=42)
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
        resampled_train_x, resampled_train_label = rus.fit_resample(train_x, train_label)
        print('-'*30)
        print('Train Data')
        print('  - Resampled Positive:', np.sum(resampled_train_label))
        print('  - Resampled Negative:', len(resampled_train_label)-np.sum(resampled_train_label))
        print('-'*30)
        print()
        return resampled_train_x, resampled_train_label

    def oversampling(self, train_x, train_label, counts=False, ratio=0.1, n_ratio=1, p_ratio=1/9):
        """
            Oversampling : This function is to increase positive data in imbalanced data using randam sampling
        """
        if counts:
            num = np.sum(train_label)
        else:
            num = len(train_label) - np.sum(train_label)
        ros = RandomOverSampler(sampling_strategy={0:int(num*n_ratio), 1:int(num*p_ratio)}, random_state=42)
        # ros = RandomOverSampler(sampling_strategy='auto', random_state=0)
        resampled_train_x, resampled_train_label = ros.fit_resample(train_x, train_label)
        print('-'*30)
        print('Train Data')
        print('  - Resampled Positive:', np.sum(resampled_train_label))
        print('  - Resampled Negative:', len(resampled_train_label)-np.sum(resampled_train_label))
        print('-'*30)
        print()
        return resampled_train_x, resampled_train_label

    def SMOTE(self, train_x, train_label, counts=False, ratio=0.1, n_ratio=1, p_ratio=1/9):
        """
            SMOTE : This function is to increase positive data in imbalanced data.
                    SMOTE uses data distances based on K-means to increase positive data, unlike oversampling.
        """
        if counts:
            num = np.sum(train_label)
        else:
            num = len(train_label) - np.sum(train_label)
        smote = SMOTE(sampling_strategy=ratio, random_state=42)
        # smote = SMOTE(sampling_strategy={0:int(num*n_ratio), 1:int(num*p_ratio)}, random_state=42)
        resampled_train_x, resampled_train_label = smote.fit_resample(train_x, train_label)
        print('-'*30)
        print('Train Data')
        print('  - Resampled Positive:', np.sum(resampled_train_label))
        print('  - Resampled Negative:', len(resampled_train_label)-np.sum(resampled_train_label))
        print('-'*30)
        print()
        return resampled_train_x, resampled_train_label

    def SMOTEENN(self, train_x, train_label, counts=False, ratio=0.1, n_ratio=1, p_ratio=1/9):
        """
            SMOTE-ENN : This function is to increase positive data and to reduce negative data in imbalanced data.
        """
        if counts:
            num = np.sum(train_label)
        else:
            num = len(train_label) - np.sum(train_label)
        smote = SMOTEENN(sampling_strategy=ratio, random_state=42)
        # smote = SMOTEENN(sampling_strategy='auto', random_state=42)
        resampled_train_x, resampled_train_label = smote.fit_resample(train_x, train_label)
        print('-'*30)
        print('Train Data')
        print('  - Resampled Positive:', np.sum(resampled_train_label))
        print('  - Resampled Negative:', len(resampled_train_label)-np.sum(resampled_train_label))
        print('-'*30)
        print()

        return resampled_train_x, resampled_train_label