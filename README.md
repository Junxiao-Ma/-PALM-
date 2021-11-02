# 飞桨常规赛：PALM眼底彩照视盘探测与分割 - 10月第四名方案


**赛题简述**
	
    PALM眼底视盘检测与分割常规赛的重点是研究和发展与患者眼底照片结构分割相关的算法。该常规赛的目标是评估和比较在一个常见的视网膜眼底图像数据集上分割视盘的自动算法。该任务目的是对眼底图像的视盘进行检测，若存在视盘结构，需从眼底图像中分割出视盘区域；若无视盘结构，分割结果直接置全背景。

![](https://ai-studio-static-online.cdn.bcebos.com/938ab4fac88e44969e61f8f10181ca1366c53fbc3d6147ff80487b03c964543e)


**数据基本标签**

	标签为 0 代表视盘（黑色区域）；标签为 255 代表其他（白色区域）。
    

**训练数据集**

文件名称：Train

Train文件夹里有fundus_images文件夹和Disc_Masks文件夹。

* fundus_images文件夹内包含800张眼底彩照，分辨率为1444×1444，或2124×2056。命名形如H0001.jpg、N0001.jpg、P0001.jpg和V0001.jpg。

* Disc_Masks文件夹内包含fundus_images里眼底彩照的视盘分割金标准，大小与对应的眼底彩照一致。命名前缀和对应的fundus_images文件夹里的图像命名一致，后缀为bmp。

**测试数据集**

文件名称：PALM-Testing400-Images

* 包含400张眼底彩照，命名形如T0001.jpg。

# 一、官方基线方案内容

* 1、解压数据与数据划分

	-- # 解压数据集
    
	-- !unzip -oq /home/aistudio/data/data85136/常规赛：PALM眼底彩照视盘探测与分割.zip -d PaddleSeg/data
    
   -- # 划分数据
   
	-- !python utils/dataset_splited.py

* 2、数据标签预处理

   -- # 转换标签
   
   -- !python utils/dataset_pretrans.py

	* 原分类为1分类问题，为了问题研究的充分性和更大程度上利用多分类间的类别竞争对分类结构有一个更好的指导
   
   * 二分类问题描述，原标签为0不变，将255无效值转换为1值
   
   * 后期提交前会后处理，消去1值，换回赛题需要的255值
   
* 3、利用PaddleSeg套件加速赛题开发与测试: 可参考套件config中的yml，结合动态图API进行快速高效的实验开发

* 4、 实现训练流程

* 5、 实现预测流程

* 6、完成提交结果 -- 基线方案为0.86431的得分(iters:2400)，可从**训练迭代次数**、**损失函数**、**模型**入手
	
   -- # 提交结果后处理
   
	-- utils/post_process.py

**部分训练参数**

![](https://ai-studio-static-online.cdn.bcebos.com/19c1febc87ec4fd78bb5b4a1966389c0979588525df44379885e5543c41d363c)

![](https://ai-studio-static-online.cdn.bcebos.com/d8ff821a46ab43df8e13fda1d2e5752bd4a54eeb3ae643dbad0288ff41f0c6cd)


## 准备数据集与PaddleSeg
为了大家下载方便，这里把PaddleSeg已经为大家准备好了压缩包，大家解压就好


```python
# 解压PaddleSeg压缩包
!unzip -oq data/data114877/PaddleSeg.zip -d /home/aistudio/
```

上一步mv，可以将PaddleSeg加压后的文件目录改成PaddleSeg

> PaddleSeg下载至github的release2.0版本，为了方便大家使用，已添加在了数据集中供大家使用


```python
# 解压数据集到PaddleSeg目录下的data文件夹
!unzip -oq /home/aistudio/data/data114783/常规赛：PALM眼底彩照视盘探测与分割.zip -d PaddleSeg/data
```


```python
# 查看数据集文件的树形结构
!tree -d PaddleSeg/data/常规赛：PALM眼底彩照视盘探测与分割
```

# 二、比赛数据集情况
PALM-Testing400-Images : 测试数据集文件夹

Train : 训练数据集文件夹

* Disc_Masks   ; 标注图片
* fundus_image  : 原始图片

> 注意没有验证数据集，这里提供一个简单的划分程序，划分比例为0.7

通过PIL的Image读取图片查看以下原数据与Label标注情况


```python
from PIL import Image

# 读取图片
png_img = Image.open('PaddleSeg/data/常规赛：PALM眼底彩照视盘探测与分割/Train/fundus_image/H0003.jpg')
png_img  # 展示图片
```


```python
bmp_img = Image.open('PaddleSeg/data/常规赛：PALM眼底彩照视盘探测与分割/Train/Disc_Masks/H0003.bmp')
bmp_img   # 展示图片
```

可以看出，白色部分全是255，黑色为有效标注区域(0值)

# 三、划分数据集与数据预处置


当前划分比例为0.7——可在`utils`文件夹下的`dataset_splited.py`修改`train_percent`为其它值

数据预处置-可在`utils`文件夹下的`dataset_pretrans.py`中查看相关代码


```python
# 保证路径为初始路径
%cd /home/aistudio/PaddleSeg

# 划分数据
!python utils/dataset_splited.py

# 转换标签--预处置
!python utils/dataset_pretrans.py
```

移除原数据，减小项目空间，减少下一次进入和退出保存时花的时间


```python
# 移除’常规赛：PALM眼底彩照视盘探测与分割‘文件夹
!rm -rf PaddleSeg/data/常规赛：PALM眼底彩照视盘探测与分割
!rm -rf PaddleSeg/data/__MACOSX
```

# 四、下载依赖项

> 平台可以不用下载，但是如果在本地可能需要执行这一步


```python
# 下载依赖项，保证PaddleSeg正常运行
%pwd
!pip install -r requirements.txt
```

# 五、开始构建比赛模型

## 1. 导入需要的库


```python
# 当前套件下切换目录到PaddleSeg下，才能使用paddleseg
%pwd
import paddle                                     # paddle基本框架
from paddleseg import models as M                 # paddleseg的模型库--对应模型源代码：PaddleSeg/paddleseg/models
from paddleseg.models import backbones as B       # 分割模型需要的骨干网络--对应模型源代码：PaddleSeg/paddleseg/models/backbones
from paddleseg.models import losses as L          # 分割模型需要的损失函数--对应模型源代码：PaddleSeg/paddleseg/models/losses
from paddleseg import transforms as T             # 分割模型需要的数据预处理方法(图像)--对应模型源代码：PaddleSeg/paddleseg/transforms/transforms.py
from paddleseg.datasets import OpticDiscSeg       # paddleseg对应的数据加载机制--对应模型源代码：PaddleSeg/paddleseg/datasets/optic_disc_seg.py
from paddleseg.core import train, evaluate, predict  # 训练、评估、预测接口--对应模型源代码：PaddleSeg/paddleseg/core

import os                                         # 必要的文件处理
```

## 2 . 创建模型与Dataset


```python
# 实例创建一个模型——当前演示基线模型EMANet
# 其它模型可前往PaddleSeg/paddleseg/models下查看相应的其它.py文件查阅参数，进行配置
# 也可通过help(M.EMANet)查看类文档描述
# 必要的，可以参考PaddleSeg/configs中同名模型yml文件配置
model = M.EMANet(
                 num_classes=2,                  # 类别数，这里已经转换为2分类问题了
                 backbone=B.ResNet50_vd(),       # 选用骨干网络--注意骨干网络和backbone_indices的搭配(注意传入的是实例化的对象哦，不要传成类了)
                 backbone_indices=[2, 3]         # backbone流向分割部分的特征通道index，必要时可以使用[1,], [1, 2], [2, 3], [1, 3], [1, 2, 3]
)
```

### EMANet完整参数介绍如下：

`source: PaddleSeg/paddleseg/models/emanet.py`

```python
num_classes,
backbone,
backbone_indices=(2, 3),       
ema_channels=512,            # ema输入通道——backbone输出-->经过ema输入得到ema_channels的通道数
gc_channels=256,             # ema输出编码通道(不等于num_classes)
num_bases=64,               # 注意力参数个数
stage_num=3,                # 编码状态(次数)
momentum=0.1,               # 动量--与注意力有关
concat_input=True,           # 拼接输入
enable_auxiliary_loss=True,     # 组合损失
align_corners=False,          # 居中对齐
pretrained=None              # 是否加载预处理
```


```python
# 配置dataset以及相应的transform
train_transforms = [
    T.RandomHorizontalFlip(),          # 水平翻转
    T.Resize(target_size=(800, 800)),
    T.Normalize()
]

# 验证处理可以不添加额外的方式，确定形状即可
eval_transforms = [
    T.Resize(target_size=(800, 800)),  # 缩放大小
    T.Normalize()
]

# 创建数据集
train_dataset = OpticDiscSeg(
    dataset_root='data',             # PaddleSeg/data目录下存在与mode对应的train_list.txt
    transforms=train_transforms,
    mode='train'                     # mode不要写错了
)

eval_dataset = OpticDiscSeg(
    dataset_root='data',
    transforms=eval_transforms,
    mode='val'
)
```

## 3. 配置学习率与损失计算方式


```python
base_lr = 0.01          
# 多项式学习率，由decay_steps决定base_lr到base_lr*0.01的间隔迭代次数
lr = paddle.optimizer.lr.PolynomialDecay(base_lr, power=0.9, decay_steps=2000, end_lr=base_lr*0.01)

# 优化器--可以换用Adam
# 正则项可以调高一些
optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=4.0e-5)
```


```python
# paddleseg支持多损失，所以采用字典的方式配置损失
losses = {}  # 创建loss字典
losses['types'] = [L.CrossEntropyLoss(), L.DiceLoss()]  # 将需要的多个损失进行添加

# 每一个损失计算的结果的权重
# all_loss = losses['types'][0] * losses['coef'][0] + losses['types'][2] * losses['coef'][2]
losses['coef'] = [4.0, 2.0] 
```

* DiceLoss比较符合赛题

* CrossEntropyLoss适合多分类损失计算

* 如果是相同的损失可以通过: [L.CrossEntropyLoss()] * 2 实现 [L.CrossEntropyLoss(), L.CrossEntropyLoss()]

* coef系数值，不是越大越好，控制适当的比例就可以了

## 4. 开始训练


```python
train(
    model=model,                        # 创建的模型
    train_dataset=train_dataset,        # 训练数据集
    val_dataset=eval_dataset,           # 验证数据集
    optimizer=optimizer,                # 优化器
    save_dir='output',                  # 保存路径--不必该，否则后边程序也需要修正
    iters=2400,                         # 训练迭代次数（这里不是轮次）
    batch_size=4,                       # 批大小
    save_interval=200,                  # 验证+保存的迭代周期
    log_iters=40,                       # 日志输出的迭代周期
    num_workers=0,                      # 多线程关闭(0)--在平台上开启可能会断掉
    losses=losses,                      # 损失字典
    use_vdl=True)                       # 是否记录训练参数
```

## 5. 开始预测

预测的配置略微不同，需要读取`test_list.txt`中的文件进入list中，然后传入list以及Image_dir进行预测

> 前面的训练与验证是通过给dir，自动搜寻，这里不一样，要注意一下哦


```python
test_list = []
test_root = 'data'      # 之前划分数据图像保存的根路径
with open('data/test_list.txt') as f: 
    for i in f.readlines():
        test_list.append(os.path.join(test_root, i[:-1]))   # 逐行写入，-1是为了去掉 \n

# 预测的transform也略有不同，前边时list，这里需要严格的传入transform格式
# 利用Compose实现多处理
transforms = T.Compose([
    T.Resize(target_size=(800, 800)),
    T.Normalize()
])
```

使用`predict`接口进行预测


```python
predict(
        model,                                           # 创建的模型
        model_path='output/best_model/model.pdparams',   # 模型参数
        transforms=transforms,                           # 数据处理方式--尽量避免高斯那些操作，在这里效果不好
        image_list=test_list,                            # 上边生成的图片list
        image_dir=test_root,                             # 图片保存的形式--保证预测结果保存在同名的结构中，但不是在test_root目录下，而是output/results中
        save_dir='output/results'                        # 保存路径——PaddleSeg/output/results/pseudo_color_prediction：为真实预测结果
    )
```

## 6. 后处理并生成提交文件(提交时忘记保存checkpoint)


```python
!python utils/post_process.py
```


```python
# 复制文件到最顶层目录
!cp -r output/results/pseudo_color_prediction/Image/ Disc_Segmentation
# 压缩文件
!zip -r Disc_Segmentation.zip Disc_Segmentation
# 删除复制的文件
!rm -rf Disc_Segmentation
```

其它一些清理步骤选择性使用即可


```python
# 删除zip的文件--丢失提交结果，需重新后处理生成
# !rm -rf Disc_Segmentation.zip
# 删除预测结果--丢失预测结果，需重新预测
# !rm -rf PaddleSeg/output/results
# 删除output文件夹--丢失模型参数，需重新训练
# !rm -rf PaddleSeg/output
# 删除data文件夹--数据将丢失，需要重新解压，划分，预处置
# !rm -rf PaddleSeg/data
```

# 六、其它建议

- 1. 模型建议：注意力模型或者经典的unet模型

- 2. 损失建议：多损失结构，不同的ccoef，针对赛题的特殊损失等

- 3. 模型魔改建议：尝试对Unet添加注意力模块，修改参数，或者调整不同的backbone与indices组合

- 4. 优化器与学习率策略的调整


最后，祝大家Paddle越用越顺手，比赛越打越顺利——取得理想的成绩！
> 有问题欢迎评论区讨论

## 七、个人简介
本人来自江苏科技大学本科三年级，刚刚接触深度学习不久希望大家多多关注

感兴趣的方向：目标检测，强化学习，自然语言处理、

个人链接：
[马骏骁](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/824948)

<https://aistudio.baidu.com/aistudio/personalcenter/thirdview/824948>
