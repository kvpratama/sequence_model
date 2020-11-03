import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import pdb


class UrbanSoundDataset(Dataset):
    # rapper for the UrbanSound8K dataset
    # Argument List
    #  path to the UrbanSound8K csv file
    #  path to the UrbanSound8K audio files
    #  list of folders to use in the dataset

    def __init__(self, csv_path, file_path, folderList):
        csvData = pd.read_csv(csv_path)
        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(0, len(csvData)):
            if csvData.iloc[i, 5] in folderList:
                self.file_names.append(csvData.iloc[i, 0])
                self.labels.append(csvData.iloc[i, 6])
                self.folders.append(csvData.iloc[i, 5])

        self.file_path = file_path
        self.folderList = folderList

    def __getitem__(self, index):
        # format the file path and load the file
        path = self.file_path + "fold" + str(self.folders[index]) + "/" + self.file_names[index]
        sound, sample_rate = torchaudio.load(path, out=None, normalization=True)
        # pdb.set_trace()
        # load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
        soundData = torch.mean(sound, dim=0, keepdim=True)
        # tempData = torch.zeros([1, 160000])  # tempData accounts for audio clips that are too short

        # repeat short sound
        while len(soundData[0]) < 160000:
            soundData = torch.cat([soundData, soundData], axis=1)

        soundData = soundData[:, :160000]

        # if soundData.numel() < 160000:
        # tempData[:soundData.numel()] = soundData[:]
        # tempData[:, :soundData.numel()] = soundData
        # else:
        # tempData[:] = soundData[:160000]
        # tempData = soundData[:, :160000]

        # soundData = tempData

        # soundFormatted = torch.zeros([160, 1])
        # soundFormatted[:160] = soundData[::1000]  # take every fifth sample of soundData
        # torchaudio.save('save3.wav', soundData.permute(1,0)[:,:160000][::5], sound[1])
        # soundFormatted = soundFormatted.permute(1, 0)

        mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(soundData)  # (channel, n_mels, time)
        mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
        mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate)(soundData)  # (channel, n_mfcc, time)
        mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
        # spectogram = torchaudio.transforms.Spectrogram(sample_rate=sample_rate)(soundData)
        feature = torch.cat([mel_specgram_norm, mfcc_norm], axis=1)
        # norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # feature_norm = norm(feature)
        # pdb.set_trace()
        # return soundFormatted, self.labels[index]
        # return mel_specgram[0].permute(1, 0), self.labels[index]
        # return mfcc[0].permute(1, 0), self.labels[index]
        return feature[0].permute(1, 0), self.labels[index]

    def __len__(self):
        return len(self.file_names)