import os
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
import japanize_matplotlib
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import IPython.display as ipd

# 音声ファイルのスペクトログラムを画像として保存する
# 引数
#   dir: 音声ファイルが存在するディレクトリ
#   file_name: 音声ファイルの名前


def Spectrogram(dir, file_name, winsize=1024):
    fs, y = wavfile.read(dir + file_name)
    y = np.mean(y, axis=1) if len(y.shape) > 1 else y
    window = signal.windows.hann(winsize)
    f, t, Y = signal.stft(y, fs, window=window,
                          nperseg=winsize, noverlap=winsize-0.01*fs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("時間[s]")
    ax.set_ylabel("周波数[Hz]")
    ax.set_title("\"" + file_name + "\"のスペクトログラム")
    ax.pcolormesh(t, f, np.log(np.abs(Y)+1))
    # plt.rcParams["svg.fonttype"] = "none"
    fig.savefig("./png-images/Spectrograms/" + file_name.replace(".wav",
                "_SPECTROGRAM.png"), format="png", dpi=1200)
    plt.close()

    # 実際に生成

# 音声ファイルの振幅スペクトルをFFTで計算し、画像として保存する
# 引数
#   dir: 音声ファイルが存在するディレクトリ
#   file_name: 音声ファイルの名前


def FFTPlot(dir, file_name):
    fs, y = wavfile.read(dir + file_name)
    y = np.mean(y, axis=1) if len(y.shape) > 1 else y
    y1 = y[0: fs]
    Y1 = fft(y1)
    Y1_half = Y1[0: int(len(Y1)/9)]
    spec = np.abs(Y1_half)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("周波数[Hz]")
    ax.set_ylabel("振幅")
    ax.set_title("\"" + file_name + "\"の振幅スペクトル")
    ax.plot(spec)
    # plt.rcParams["svg.fonttype"] = "none"
    fig.savefig("./png-images/amp-spectrum/" +
                file_name.replace(".wav", "_FFT.png"), format="png", dpi=1200)
    plt.close()

# 音声ファイルの波形を画像として保存する
# 引数
#   dir: 音声ファイルが存在するディレクトリ
#   file_name: 音声ファイルの名前


def WavePlot(dir, file_name):
    fs, y = wavfile.read(dir + file_name)
    t = np.arange(0, len(y)/fs, 1/fs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("時間[s]")
    ax.set_ylabel("振幅")
    ax.set_title("\"" + file_name + "\"の波形グラフ")
    ax.plot(t, y)
    # plt.rcParams["svg.fonttype"] = "none"
    fig.savefig("./png-images/Waveforms/" + file_name.replace(".wav",
                "_WAVEFORM.png"), format="png", dpi=1200)
    plt.close()


dir = "./wavdata/"
for i in os.listdir('./wavdata/'):
    if (i.replace(".wav", "_FFT.png") not in os.listdir('./png-images/amp-spectrum/')):
        print(i)
        FFTPlot(dir, i)
        WavePlot(dir, i)
        Spectrogram(dir, i)
