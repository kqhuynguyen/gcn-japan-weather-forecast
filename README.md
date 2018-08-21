# What is this?
A repository which contains the implementation for a forecasting task.

**The task**: Given the previous 5 hours, predict the temperatures recorded from a set of sensors in in next hour.

**The architecture**: Youngjoo Seo, Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst, [Structured Sequence Modeling With Graph Convolutional Recurrent Networks][paper]

[paper]: https://arxiv.org/abs/1612.07659
I also implemented another architecture similar to the proposed gconvLSTM, with the LSTM cell replaced by a basic RNN cell.

**The result**:

Model | k | knn | RMSE
------------ | ------------- | ------------ | -------------
lstm |  |  | 1.5619
glstm | 1 | 4 | 0.1749
glstm | 2 | 4 | **0.1674**
glstm | 3 | 4 | 0.1701
glstm | 4 | 4 | 0.1756
grnn | 3 | 8 | 0.1954
grnn | 1 | 4 | 0.1747
grnn | 2 | 4 | **0.1700**
# Dataset

[Sensor data, preprocessed][data]. This should be put into `datasets/japan`.

[data]: https://drive.google.com/open?id=18_0xnKfjj7_PDXn58Dp3FPtAAMvpfoPB
# Reproduction

Requirements: 
```
Python 2.7
Tensorflow 1.1.0+
```
Make sure you have the dataset ready. The settings in `config.py` should be handled. Run:
```
pip install -r requirements.txt
python gconv_main.py
```
