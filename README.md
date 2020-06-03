# TSC-KNN

**基于KNN(k-NearestNeighbor，k最近邻)和DTW（Dynamic Time Warping，动态时间归整）的时间序列分类方法**

绝大多数信号数据都可以看成1维的时间序列，包括基于时间序列数据的诸如故障诊断，语音识别等应用均涉及时间序列的分类，本项目提供了一种简单高效的基于KNN和DTW的时间序列分类方法。

DTW采用动态规划来计算两个时间序列之间的相似性，算法复杂度为O(N2)。考虑到代码运行效率，这里采用基于C编译的快速DTW方法。

KNN算法采用DTW运算结果作为距离值，采用vote方法进行分类。

## data数据集
Notebook | data
-------- | ------
[data](https://github.com/vvanggeng/TSC-KNN/blob/master/data/data.zip) | Data exploration


提供了两组数据集，每组均包含训练数据和测试数据，也可自行划分。

该数据集为压电传感器所采集冲击信号，分类标签对应于冲击区域编号A，B，C...
