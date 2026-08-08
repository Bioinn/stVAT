# 1.背景介绍
空间转录组研究中的两个重要分析任务：**识别显示空间表达模式的空间变异基因（SVGs）和检测具有一致基因表达的空间域**。空间变异基因通常指的是在组织中表达水平存在明显差异的基因，这些基因可能在空间上呈现出特定的分布。识别SVG能系统分析特定位置的细胞状态、推断细胞间的通讯等。空间域定义为基因表达和组织学在空间上一致的区域，检测空间域能够更好理解组织结构和功能分区。
# 2.设计思路
目前识别SVGs的方法主要是基于统计模型的方法，它们首先构建基因表达谱与空间位置之间相关性的统计模型，然后返回一个p值来指示基因表达的空间变异性。只使用统计显著性来衡量基因表达中的空间模式，则难以从空间角度解释SVGs内的空间异质性和同质性。对于检测空间域，ST数据的高维度和稀疏性使得传统算法不适合有效地识别具有生物一致性的空间域。新兴方法则在细胞分辨率下表现不佳，仍然很难检测复杂组织的空间域。并且目前的算法对这两个问题大都是当成完全割裂的两个问题进行处理的。

因此作者提出了PROST计算框架，包括（i）通过PROST指数定量表征基因表达模式中的空间变化；（ii）通过自注意机制对空间域进行无监督聚类。PROST框架包含计算SVGs的PROST Index (PI)模块和检测空间域的PROST Neural Network (PNN)模块，这两个工作流程的联合利用使PROST能够整合空间信息和基因表达谱，以检测具有组织学特征的一致表达模式的空间域。
# 3.环境准备
1.安装Conda/Miniconda环境,教程可参考：[https://zhuanlan.zhihu.com/p/1978239735307708129](https://zhuanlan.zhihu.com/p/1978239735307708129)
2.开始栏输入`cmd`进入命令行界面，以此分别输入以下两行命令创建并激活PROST环境：
```python
conda create -n stVAT python=3.10
conda activate stVAT
```
3.在`PROST_ENV`里创建R环境，依次输入以下命令
```python
conda install -c conda-forge r-base
conda install -c conda-forge r-mclust=5.4.10
```
4.安装相关依赖包 cmd切换路径到解压后的~PROST-master`根目录文下，以下做Windows/Linux系统区分
如果是Windows系统输入以下命令：
```python
pip install -r requirements_win.txt
pip install rpy2-2.9.5-cp37-cp37m-win_amd64.whl
```
如果是Linux系统输入以下命令：
```python
pip install -r requirements.txt
```
5.安装PROST项目包,三行命令依次执行：
```python
pip install setuptools==58.2.0
python setup.py build
python setup.py install
```
***
到此完成环境准备。
***
## 基于PROST的人类大脑背外侧前额叶皮层(DLPFC) 10x Genomics Visium 空间转录组数据分析
DLPFC 是神经科学研究中的经典样本,因为这个脑区具有明显的分层结构(大脑皮层有6层细胞组织),非常适合用来验证空间聚类算法能否正确识别出这些解剖学上已知的功能分区。

现在可以打开`test`文件夹下的`testMEL.py`，按照注释的步骤逐个运行代码，完成基于PROST的人类大脑背外侧前额叶皮层(DLPFC) 10x Genomics Visium 空间转录组数据分析。

通过这个示例，你将能够了解如何使用PROST框架来识别空间变异基因和检测空间域，从而更好地理解组织结构和功能分区。

另外若想尝试其他数据集，可参考官方文档：[https://prost-doc.readthedocs.io/en/latest/index.html](https://prost-doc.readthedocs.io/en/latest/index.html)
