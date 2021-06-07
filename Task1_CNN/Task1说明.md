# 任务一：卷积神经网络
要求理解卷积神经网络（CNN）的基本思想与操作，通过基于Pytorch框架的代码填空加深理解，并实现简单的demo。

## 参考资料
1. \*[Bilibili 台大李宏毅 卷积神经网络讲解 约1小时](https://www.bilibili.com/video/BV1Lb411b7BS?from=search&seid=41737188674805425) - 学习CNN基础知识
2. 《神经网络与深度学习》第5章 卷积神经网络 - 学习CNN基础知识
3. \*[PyTorch 1.8.1 documentation](https://pytorch.org/docs/stable/index.html) - PyTorch官方文档 用以查询API的参数与函数信息
4. 善用搜索引擎 - 各种疑难杂症解决

## 任务

- \*基础任务：
    1. 利用卷积神经网络，实现对MNIST数据集的分类问题。（CNN.py代码填空，最终准确率应在96%以上）
    
    首次运行时的超参（learning_rate、keep_prob_rate等）建议按照默认值进行。在得到较高的准确率后可以尝试对超参数进行更改来探索不同参数对模型所造成的影响。
    
        - 数据集：MINIST数据集 
            - 下载方式1：网盘链接：https://pan.baidu.com/s/1xjyY9849eAdTaMKxyHazZg 提取码：thzy 
            - 下载方式2: 通过代码下载(提供的代码里面已经包含了)
        - *需要掌握的核心知识点：
            - CNN中的基本组件：卷积（包含卷积核尺寸kernel size或patch、padding、stride等参数）、池化
            - 全连接
        - 其他可补充知识点：
            - 数据集：训练集/测试集的划分
            - ReLU 激活函数
            - Dropout 防止过拟合
            - Softmax 用于多分类
            - Mini-batches
    
    2. 利用卷积神经网络，实现对Scratch数据集的图片分类问题。（将先前代码应用到新的数据集中）
        - 数据集: 网盘链接：https://pan.baidu.com/s/1cTVuPiBhaa_PgW3EMQ8mhg
    3. 自行了解机器学习基础知识


- 可选任务（组会交流内容）：
    - 要求：
        - 2~3人一组
        - 选择其中一项可选任务，阅读该模型的经典文献，通过搜索引擎查阅资料，了解模型细节，实现再MINST数据集上的分类
        - 准备组会PPT，交流在过程中了解到的重要的知识
    1. VGG（构建模型的思路：使用重复的块）
    2. NiN（构建模型的思路：网络中的网络，串联多个由卷积层和全连接层构成的小网络来构成网络）
    3. ResNet（残差层）
    
- 时间：两周
- 组会时间：6.20 13：30pm

