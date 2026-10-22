CV编码方向深度学习工程师4个月成长计划

方向解析
端到端编码     → 图像/视频压缩的神经网络化（替代传统H.264/H.265）

前处理        → 编码前的图像增强、去噪、超分、HDR处理

后处理        → 解码后的质量恢复、伪影消除、增强

代理编码器    → 用神经网络模拟/替代传统编码器行为

--------------------------------------------------------------------------------

前置知识摸底（第1周，自测）
需要确认自己的起点
自测问题：
□ 是否了解图像的RGB/YUV格式？
□ 是否知道PSNR/SSIM是什么？
□ 是否用过PyTorch写过任何网络？
□ 是否知道H.264/H.265是什么？
□ 是否了解卷积神经网络基本原理？
根据自测结果决定起点
全部不了解 → 从第0阶段开始
了解图像基础但不懂深度学习 → 从第0阶段第2部分开始  
懂深度学习但不懂编码 → 从第1阶段开始

--------------------------------------------------------------------------------

第0阶段：基础补全（第1-2周）
如果是完全初学者，这两周非常关键，不能跳过

Week 1：图像与视频基础
需要学习的知识点
图像基础：
- 图像的数字表示：像素、位深度、色彩空间
- RGB vs YCbCr/YUV的区别和转换
- 图像的频域概念：空间域和频率域
- 离散余弦变换DCT的基本原理
- 图像质量评价指标：PSNR、SSIM、MS-SSIM
​
视频基础：
- 视频的帧结构：I帧、P帧、B帧
- 运动估计与运动补偿基本概念
- 传统编解码流程：变换→量化→熵编码
- 码率、量化步长、质量的关系
- 率失真理论基本概念（Rate-Distortion Theory）
学习资料
书籍：
- 《数字图像处理》Rafael C. Gonzalez（前5章）
  重点读：第2章图像基础、第3章空间域处理、第4章频率域处理
​
视频课程：
- B站搜索"图像处理基础"- 中国大学MOOC相关课程
- 斯坦福CS231n第一讲（图像基础部分）
​
文档：
- FFmpeg官方文档中的概念介绍部分
- Wikipedia: YCbCr词条、DCT词条（看懂公式）
本周实践任务
任务1：用Python+OpenCV
  - 读取一张图像，转换RGB到YUV
  - 计算两张图像之间的PSNR和SSIM
  - 对图像做DCT变换并可视化频谱
​
任务2：用ffmpeg命令行
  - 安装ffmpeg
  - 用不同QP值编码同一视频，观察质量和码率变化
  - 学会用ffprobe查看视频信息
    
Week 2：深度学习基础快速入门
需要学习的知识点
核心概念：
- 神经网络前向传播和反向传播原理
- 卷积层、池化层、激活函数、BN层
- 损失函数：MSE、L1、感知损失
- 优化器：SGD、Adam的基本原理
- 过拟合与正则化：Dropout、权重衰减
​
PyTorch基础：
- Tensor操作
- Dataset和DataLoader
- 定义网络（nn.Module）
- 训练循环的标准写法
- 模型保存和加载
- GPU使用

学习资料
首选课程：
- 李沐《动手学深度学习》（[d2l.ai](https://zh.d2l.ai/index.html)）
  重点章节：第3章线性网络、第5章深度学习计算、第6章卷积神经网络
  这本书配套有视频，B站搜索"李沐动手学深度学习"
​
PyTorch官方：
- PyTorch官网 60分钟入门教程（pytorch.org/tutorials）
  按顺序完成：
  Tensor Tutorial(https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
  → Autograd(https://docs.pytorch.org/tutorials/beginner/former_torchies/autograd_tutorial_old.html#autograd)
  → Neural Networks(https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#neural-networks)
  → Training a Classifier(https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-a-classifier)
​
补充：
- 3Blue1Brown YouTube神经网络系列（4个视频，非常直观）
- 训练经验 https://karpathy.github.io/2019/04/25/recipe/  
本周实践任务
任务：在CIFAR-10数据集上
  - 实现一个简单CNN分类器（3-5层）
  - 跑通完整训练流程
  - 用TensorBoard记录loss曲线
  目标：训练精度达到80%以上
  
  重要：这个任务不是为了精度，是为了熟悉完整训练流程

--------------------------------------------------------------------------------
  
第1阶段：图像质量增强基础（第3-5周）
这是进入编码方向的必经之路，前处理后处理都建立在这个基础上

Week 3：图像复原与超分辨率基础
需要学习的论文（按顺序）
论文1：SRCNN（最简单的起点）
标题："Learning a Deep Convolutional Network for Image Super-Resolution"
作者：Dong et al.
发表：ECCV 2014 / TPAMI 2015
学习重点：
  - 为什么CNN能做超分
  - 端到端学习的思路
  - 如何定义超分任务的输入输出
​
论文2：VDSR
标题："Accurate Image Super-Resolution Using Very Deep Convolutional Networks"
作者：Kim et al.
发表：CVPR 2016
学习重点：
  - 残差学习在低级视觉的应用
  - 为什么学习残差比学习映射更容易
​
论文3：EDSR（重要）
标题："Enhanced Deep Residual Networks for Single Image Super-Resolution"
作者：Lim et al.
发表：CVPRW 2017
学习重点：
  - 去掉BN层的原因（这在图像复原任务中非常重要）
  - 残差缩放的技巧
  - 这是很多后续工作的baseline
需要研究的代码
代码库1：BasicSR
地址：https://github.com/XPixelGroup/BasicSR
学习内容：
  - 整体项目结构怎么组织
  - 数据加载Pipeline怎么写（degradation流程）
  - 训练框架怎么搭
  重点看：basicsr/models/ 和 basicsr/data/ 目录
​
具体操作：
  Step1：克隆仓库，按文档配置环境
  Step2：跑通EDSR的训练脚本
  Step3：读懂SRDataset的数据加载代码
  Step4：读懂训练循环和验证代码
本周需要掌握的技能
技术点：
- 如何构建图像的训练数据对（HR图像→下采样→LR图像）
- 图像的随机裁剪、翻转等数据增强
- 如何在图像任务中正确计算PSNR/SSIM
- 像素值归一化的注意事项（[0,1]还是[-1,1]）
- 如何可视化训练结果（保存中间图像）
Week 4：感知损失与图像质量评价
需要学习的论文
论文4：感知损失（非常重要）
标题："Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
作者：Johnson et al.
发表：ECCV 2016
学习重点：
  - 像素级MSE损失的缺陷（为什么产生模糊）
  - 用预训练VGG的特征图计算损失
  - 这个思路在编码后处理中大量使用
​
论文5：SRGAN
标题："Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
作者：Ledig et al.
发表：CVPR 2017
学习重点：
  - GAN在图像恢复中的应用
  - 感知质量vs.PSNR的权衡（这在编码领域非常核心）
  - 判别器的设计
​
论文6：ESRGAN（重要）
标题："ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
作者：Wang et al.
发表：ECCVW 2018
学习重点：
  - RRDB网络结构
  - Relativistic GAN损失
  - 特征图匹配损失的改进
需要研究的代码
继续深入BasicSR：
  - 读懂GAN训练的代码（两个优化器交替更新）
  - 读懂感知损失（PerceptualLoss）的实现
  - 理解如何用VGG提取特征
​
实践任务：
  - 分别用L1 Loss、感知损失训练超分模型
  - 对比两种损失下图像的视觉质量差异
  - 记录PSNR/SSIM和视觉质量是否一致（理解两者的矛盾）
Week 5：图像去噪与去伪影
需要学习的论文
论文7：DnCNN
标题："Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
作者：Zhang et al.
发表：TIP 2017
学习重点：
  - 盲去噪 vs. 非盲去噪
  - 残差学习在去噪中的应用
  - 这是后处理去伪影网络的重要参考
​
论文8：FFDNet
标题："FFDNet: Toward a Fast and Flexible Solution for CNN-Based Image Denoising"
作者：Zhang et al.
发表：TIP 2018
学习重点：
  - 噪声级别图作为输入（可控去噪）
  - 这个思路在编码后处理中被广泛借鉴
​
论文9：ARCNN（编码伪影去除）
标题："Deep Convolution Networks for Compression Artifacts Reduction"
作者：Dong et al.
发表：ICCV 2015
学习重点：
  - 第一篇专门针对编码伪影的深度学习方法
  - 块效应（blocking artifact）的成因和消除
  - 这是编码后处理方向的直接入门论文
第2阶段：传统视频编码深入理解（第6-7周）
做编码方向必须真正理解传统编码，否则无法理解"端到端"在改进什么

Week 6：传统图像编码标准
需要学习的知识点
JPEG编码流程（必须深入理解）：
  - 色彩空间转换 YCbCr
  - 8x8块分割
  - DCT变换（每个系数的物理含义）
  - 量化表（亮度/色度量化表）
  - Zigzag扫描
  - 霍夫曼熵编码
  重点：量化步骤是有损的根源，理解QP参数的意义
​
HEVC/H.265关键技术（概念理解层面）：
  - CTU/CU/PU/TU的层次结构
  - 帧内预测的方向模式（35种角度）
  - 帧间预测：运动估计、运动补偿
  - 变换：DST/DCT多种变换核
  - 量化：QP与Lambda的关系
  - 熵编码：CABAC原理
  - 环路滤波：去块滤波、SAO滤波
  
重要概念：率失真优化（RDO）
  - 拉格朗日优化：D + λR
  - 这是所有编码决策的核心
  - 端到端学习就是在端到端地优化这个目标
学习资料
书籍：
- 《新一代高效视频编码H.265/HEVC》
  重点读：变换量化章节、帧内预测章节、率失真优化章节
​
标准文档（不需要全读，查阅用）：
- HEVC Overview论文：
  "Overview of the High Efficiency Video Coding (HEVC) Standard"
  作者：Sullivan et al., TCSVT 2012
  这篇综述非常好，建议精读
​
视频课程：
- Coursera: "Digital Image and Video Processing" 
  （西北大学课程，有编码相关章节）
​
实践工具：
- HM（HEVC参考软件）：https://hevc.hhi.fraunhofer.de/
  任务：编译并运行HM，用不同QP编码测试序列
  观察：码率、PSNR的变化曲线（Rate-Distortion Curve）
- VVC参考软件VTM也可以了解
Week 7：传统编码的深入实践与分析
实践任务
任务1：深度分析编码结果
  - 使用HM编码标准测试序列（从Xiph.org下载）
  - 在QP=22/27/32/37下分别编码
  - 分析每种QP下的：总码率、PSNR(Y/U/V)、各类型帧的码率分布
  - 画出Rate-Distortion曲线，理解Bjontegaard Delta计算方法
​
任务2：可视化编码信息
  - 使用HM的统计输出，分析编码决策分布
  - 可视化CTU划分模式
  - 观察不同内容（平坦区域vs.纹理区域）的编码行为差异
​
任务3：理解编码伪影
  - 收集不同QP下的编码重建图像
  - 分析块效应、振铃效应、色度模糊的产生位置
  - 这为后处理方向奠定认知基础
​
工具：
  - Python写分析脚本，用matplotlib画RD曲线
  - YUView软件：可视化YUV序列和编码信息
需要理解的关键概念
对神经网络编码非常重要的传统概念：
  - 率失真理论：香农限、率失真函数R(D)
  - 信息熵与编码效率
  - 量化误差的统计特性
  - 运动场的稀疏性
  - 残差信号的分布特性（近似拉普拉斯分布）

参考论文（概念层面）：
  "Rate-Distortion Theory" 相关综述
  重点理解：为什么端到端编码要联合优化率失真
第3阶段：端到端图像编码（第8-10周）
这是核心方向，需要重点投入

Week 8：端到端编码基础
必读论文（严格按顺序）
论文1：端到端编码开山之作
标题："End-to-end Optimized Image Compression"
作者：Ballé et al.
发表：ICLR 2017
学习重点：
  - 为什么需要端到端优化
  - 如何处理量化不可微问题（加性均匀噪声近似）
  - 广义分水岭变换（GDN激活函数）
  - 率失真联合优化的损失函数设计
  - 理解编码器/解码器/熵编码三部分
  重要：这篇论文的附录数学推导要看懂
​
论文2：超先验模型
标题："Variational Image Compression with a Scale Hyperprior"  
作者：Ballé et al.
发表：ICLR 2018
学习重点：
  - 什么是超先验（hyperprior）
  - 侧信息（side information）的编码
  - 用变分自编码器框架理解图像编码
  - 高斯尺度混合模型对残差的建模
​
论文3：自回归上下文模型（重要）
标题："Joint Autoregressive and Hierarchical Priors for Learned Image Compression"
作者：Minnen et al.
发表：NeurIPS 2018
学习重点：
  - 自回归上下文模型：用已解码符号预测未解码符号
  - 联合超先验+上下文的框架
  - 这是之后大量工作的基础框架
  - 理解为什么上下文模型能提升压缩率
需要研究的代码
代码库1：CompressAI（最重要）
地址：https://github.com/InterDigitalCom/CompressAI
这是端到端编码方向最重要的代码库
​
学习步骤：
  Step1：阅读文档，理解整体架构
  Step2：运行官方示例，复现Ballé2018的RD性能
  Step3：深入阅读以下文件：
    - compressai/models/google.py（各种压缩模型）
    - compressai/entropy_models/（熵编码模型）
    - compressai/layers/（关键层实现）
    - compressai/ans.py（算术编码接口）
  Step4：理解量化和熵编码的接口
​
重点理解代码中的：
  - EntropyBottleneck的实现
  - GaussianConditional的实现
  - 训练时和推理时量化行为的差异
  - 如何计算码率（bpp: bits per pixel）
Week 9：端到端编码进阶
必读论文
论文4：注意力机制引入
标题："Image Compression with Recurrent Neural Networks"
作者：Toderici et al.
发表：CVPR 2016 / TPAMI 2017
学习重点：
  - RNN在图像编码中的应用思路
  - 渐进式编码的概念
  - 二值量化的处理方式
​
论文5：channel-wise上下文
标题："Channel-wise Autoregressive Entropy Models for Learned Image Compression"
作者：Minnen & Singh
发表：ICIP 2020
学习重点：
  - 通道方向的自回归（比空间自回归快很多）
  - 如何在速度和性能间平衡
  - 并行解码的实现方式
​
论文6：Transformer引入编码（重要）
标题："Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules"
作者：Cheng et al.
发表：CVPR 2020
学习重点：
  - 注意力模块在编码中的应用
  - 高斯混合模型（GMM）建模
  - 这是CVPR2020的重要工作，CompressAI中有实现
​
论文7：VVC可学习组件
标题："Learned Image Compression with Hierarchical Residual Units"
或
标题："SIC: Scalable Image Compression"
（选一篇阅读即可，了解可扩展编码方向）
实践任务
任务1：复现端到端编码结果
  - 在Kodak数据集上评测CompressAI中各个模型
  - 画出RD曲线（bpp vs PSNR/MS-SSIM）
  - 与JPEG/JPEG2000/BPG/HM进行对比
  - 理解Bjontegaard Delta-Rate（BD-Rate）计算方法
​
任务2：分析码率分布
  - 可视化latent表示的分布
  - 分析不同内容图像的latent分布特性
  - 理解熵编码估计的码率与实际码率的关系
​
任务3：模型修改实验
  - 在Ballé2018基础上，增加一个注意力模块
  - 观察修改后RD性能的变化
  - 记录训练时间和参数量变化
Week 10：端到端视频编码
必读论文
论文8：DVC（第一篇端到端视频编码）
标题："DVC: An End-to-end Deep Video Compression Framework"
作者：Lu et al.
发表：CVPR 2019
学习重点：
  - 光流估计替代运动估计
  - 运动压缩的流程
  - 残差编码
  - 整体框架与传统视频编码的对应关系
  - 联合优化的损失函数设计
​
论文9：DCVC系列（当前重要工作）
标题："Deep Contextual Video Compression"
作者：Li et al.
发表：NeurIPS 2021
学习重点：
  - 基于上下文的视频编码
  - 条件编码框架
  - 比DVC的改进点
​
论文10：DCVC-DC（重要，接近SOTA）
标题："Neural Video Compression with Diverse Contexts"
作者：Li et al.
发表：CVPR 2023
学习重点：
  - 多样化上下文
  - 如何在实用性上进一步提升
  代码：https://github.com/microsoft/DCVC
需要研究的代码
代码库：DCVC系列
地址：https://github.com/microsoft/DCVC
学习步骤：
  Step1：跑通DVC的推理代码
  Step2：理解光流网络（通常用SPyNet或PWCNet）的接口
  Step3：读懂运动压缩和残差压缩的代码
  Step4：理解参考帧管理和编码顺序
​
相关代码：光流网络
  SPyNet：https://github.com/sniklaus/pytorch-spynet
  理解：光流是什么，如何用光流做运动补偿
第4阶段：前处理方向（第11周）
Week 11：
编码前处理：让图像/视频在编码前更"友好"于编码器

需要学习的核心概念
前处理的目标：
  - 在相同码率下获得更高质量（或相同质量用更少码率）
  - 针对编码器的特性进行优化
​
主要任务：
  1. 编码感知的降噪（去除对编码不友好的噪声）
  2. 编码感知的超分（上采样后编码再下采样）
  3. HDR/SDR转换预处理
  4. 内容自适应的预处理（根据内容决定处理强度）
必读论文
论文1：编码感知预处理思路
标题："Enhancing HEVC Compressed Videos with a Partition-Masked Convolutional Neural Network"
作者：Dai et al.
发表：ICIP 2018
学习重点：编码感知处理的基本思路
​
论文2：感知优化预处理
标题："Pre-processing for HEVC using Deep Learning"
（搜索相关工作，这个方向有很多ICIP/ICME论文）
​
论文3：基于内容的自适应编码（重要思路）
标题："Content-Adaptive Optimization for Neural Image Compression"
作者：Campos et al.
发表：CVPRW 2019
学习重点：
  - 根据内容调整编码策略
  - 实例自适应优化的思想
​
论文4：编码前超分
标题："Toward Robust Neural Image Compression: Discretization and Quantization"
或搜索 "pre-processing super-resolution video coding"相关论文
学习重点：先超分再编码降分辨率的编码增益
实践任务
任务：设计一个简单的编码感知前处理网络
  - 输入：原始图像
  - 目标：处理后用HEVC编码，在相同码率下获得更高PSNR
  - 损失函数设计：需要考虑编码损失（用可微近似或强化学习思路）
  - 评测：计算BD-Rate相对于无前处理的增益
  
难点：前处理网络训练时需要模拟编码过程
  这里引入"代理编码器"的概念（下一阶段重点）
第5阶段：后处理方向（第12周）
week 12:
解码后处理：消除编码引入的失真，恢复图像质量

必读论文
论文1：ARCNN（已在Week5提到，再深入）
标题："Deep Convolution Networks for Compression Artifacts Reduction"
作者：Dong et al.
发表：ICCV 2015
这次重点：理解训练数据的构造方式
​
论文2：DCAD
标题："A Convolutional Neural Network Approach for Post-Processing in HEVC Intra Coding"
作者：Dai et al.
发表：MMM 2017
学习重点：
  - 针对HEVC帧内编码的后处理
  - 如何利用QP信息（QP-adaptive后处理）
​
论文3：VRCNN
标题："Variable-Rate Deep Image Compression with a Conditional Autoencoder"
注意：搜索针对视频编码后处理的VRCNN相关论文
或
标题："Residual Highway Convolutional Neural Networks for in-loop Filtering in HEVC"
作者：Park & Kim
发表：TIP 2018
​
论文4：环路滤波神经网络（重要）
标题："CNN-based In-loop Filtering for Coding Efficiency Improvement"
作者：Kang et al.
发表：IVMSP 2017
学习重点：
  - In-loop filter（环路滤波）vs. Post filter的区别
  - 如何集成到编解码器中
​
论文5：基于Transformer的后处理（前沿）
标题："STDF: Spatio-Temporal Deformable Fusion for Video Compression Artifact Reduction"
作者：Deng et al.
发表：AAAI 2020
代码：https://github.com/ryanxingql/stdf-pytorch
学习重点：
  - 利用多帧信息做视频后处理
  - 可变形卷积在时序对齐中的应用
​
论文6：BasicVSR思路应用于后处理
标题："BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond"
作者：Chan et al.
发表：CVPR 2021
代码：https://github.com/ckkelvinchan/BasicVSR-PlusPlus
学习重点：
  - 视频处理的双向传播框架
  - 这个框架可以迁移到视频后处理任务
实践任务
任务：实现一个视频编码后处理网络
  Step1：收集训练数据
    - 下载UHD/HD视频序列（Vimeo-90K数据集）
    - 用ffmpeg以不同QP编码，得到原始/压缩图像对
  
  Step2：实现基础后处理网络
    - 参考ARCNN架构
    - 加入QP条件（QP作为网络输入）
  
  Step3：评测
    - 在JCT-VC测试序列上评测
    - 计算相对于无后处理的BD-Rate增益
    - 目标：在QP32下PSNR提升0.5dB以上
  
  Step4：改进
    - 引入时序信息（多帧输入）
    - 引入可变形卷积做时序对齐
第6阶段：代理编码器
week 13:
这是连接前后处理与端到端编码的关键技术

核心概念理解
为什么需要代理编码器：
  问题：训练前处理/后处理网络时，需要通过编码过程求梯度
  困难：传统编码器（HM/VTM）不可微，无法反向传播
  解决：用神经网络模拟编码器行为，使整个流程可微
​
代理编码器的本质：
用神经网络 f_θ 近似传统编码器 E 的行为
使得：f_θ(x) ≈ E(x)，且 f_θ 对 x 可微
​
三大应用场景：
┌─────────────────────────────────────────────────────┐
│  场景1：前处理训练                                    │
│  原始图像 → 前处理网络 → [代理编码器] → 重建图像      │
│  反向传播可以穿过代理编码器，优化前处理网络            │
│                                                     │
│  场景2：码率控制                                      │
│  用代理编码器预测编码码率，指导编码参数选择            │
│                                                     │
│  场景3：编码决策加速                                  │
│  用神经网络预测编码器内部决策（CU划分、预测模式等）    │
└─────────────────────────────────────────────────────┘
第13周：代理编码器基础理论
Week 13 Part 1：理解问题本质（前2天）
需要建立的认知框架
传统编码器的不可微问题分析：
​
1. 量化不可微
   连续值 x → round(x) → 整数
   导数：几乎处处为0，无法传播梯度
​
2. 编码决策不可微
   RDO选择最优模式：argmin(D + λR)
   argmin操作不可微
​
3. 熵编码不可微
   CABAC的上下文建模是离散过程
​
4. 环路滤波不可微
   SAO/去块滤波有阈值判断
​
代理编码器的目标：
  针对以上问题，构建可微近似
需要学习的背景知识
知识点1：直通估计器（Straight-Through Estimator, STE）
来源论文：
  标题："Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation"
  作者：Bengio et al.
  发表：arXiv 2013
学习重点：
  - 前向传播用量化值
  - 反向传播用恒等梯度（假装没有量化）
  - 这是最简单的量化梯度近似方法
  - 在很多代理编码器中被使用
​
知识点2：量化的可微近似方法综述
参考：
  标题："A Survey of Quantization Methods for Efficient Neural Network Inference"
  作者：Gholami et al.
  发表：arXiv 2021
  重点读：Section 3（量化方法）和 Section 5（量化感知训练）
  这篇综述帮助理解量化近似的各种思路
​
知识点3：知识蒸馏基础
参考：
  标题："Distilling the Knowledge in a Neural Network"
  作者：Hinton et al.
  发表：arXiv 2015
学习重点：
  - 用教师网络（传统编码器）指导学生网络（代理编码器）
  - 软标签的概念
  - 这是训练代理编码器的核心方法之一
Week 13 Part 2：代理编码器论文精读（第3-4天）
第一类：模拟编码失真的代理编码器
论文1：开创性工作（必读）
标题："Generating Videos with Scene Dynamics"
作者：Vondrick et al.
发表：NIPS 2016
注意：这篇不是编码论文，但其中可微渲染的思路
对理解代理编码器的设计有启发
学习重点：如何用神经网络近似不可微的物理过程
​
论文2：用于前处理训练的代理编码器（核心论文）
标题："Preprocessing Enhanced Image Compression for Machine Vision"
作者：Yan et al.
发表：arXiv 2021
学习重点：
  - 如何设计一个模拟HEVC失真的神经网络
  - 训练数据的构造：原始图像和编码重建图像对
  - 代理编码器的网络结构选择
  - 如何验证代理编码器的近似精度
​
论文3：可微JPEG代理（结构清晰，适合入门）
标题："JPEG-resistant Adversarial Images"
作者：Shin & Song
发表：NIPS Workshop 2017
学习重点：
  - JPEG各个步骤的可微近似
  - DCT：本身可微，直接实现
  - 量化：用软量化近似
  - 这是最简单的代理编码器案例，适合手写复现
​
论文4：可微JPEG的系统实现
标题："Towards Robust Neural Networks via Random Self-ensemble"
注意：搜索"differentiable JPEG"相关工作
或参考：
标题："DiffJPEG: A Differentiable JPEG Encoding/Decoding"
（GitHub上有对应实现：https://github.com/mlomnitz/DiffJPEG）
学习重点：
  - 系统性地将JPEG每个步骤替换为可微版本
  - 色彩空间转换：线性变换，可微
  - 下采样：双线性插值近似
  - DCT：作为固定权重的线性层实现
  - 量化：软量化或STE
  - 代码复现重点：手写每个可微步骤
第二类：用于码率预测的代理编码器
论文5：码率预测代理编码器
标题："Complexity Reduction of the CU Partition and Mode Decision in the HEVC Inter Coding"
作者及相关：搜索 "CNN-based rate prediction HEVC" 相关工作
​
更直接的论文：
标题："A Neural Network-Based Rate-Distortion Optimization for HEVC Intra Coding"
作者：Huo et al.
发表：相关期刊/会议
学习重点：
  - 用CNN预测给定内容在特定QP下的编码码率
  - 输入特征的设计（原始块特征 + QP）
  - 输出：预测码率或率失真代价
​
论文6：内容感知码率控制
标题："Content-Aware Rate Distortion Optimization Using Scene Complexity Measures"
（搜索相关工作理解码率预测的应用场景）
​
更重要的论文：
标题："Attention-based Neural Networks for Chroma Intra Prediction in Video Coding"
作者：Hu et al.
发表：JETCAS 2020
学习重点：
  - 如何将神经网络预测集成到编码器决策流程
  - 这类工作本质上是局部代理模型
第三类：编码决策代理（加速编码）
论文7：CU划分预测（重要方向）
标题："Fast HEVC Intra Coding Based on Convolutional Neural Network"
作者：Xu et al.
发表：VCIP 2017 / 相关工作
学习重点：
  - 用CNN预测CTU的最优划分深度
  - 跳过不必要的RDO计算
  - 训练标签的获取：运行HM收集最优划分决策
​
论文8：更完整的CU决策代理
标题："Reducing Complexity of HEVC: A Deep Learning Approach"
作者：Liu et al.
发表：TCSVT 2018
学习重点：
  - 多任务学习同时预测多种编码决策
  - 如何构建大规模训练数据集
  - 代理模型的精度-速度权衡
​
论文9：VVC/H.266加速（前沿）
标题："Deep Learning Based Fast HEVC Intra Coding"
作者及延伸：搜索 "CNN VVC partition decision"相关工作
更具体：
标题："Partition Map Prediction for Fast Block Partitioning in VVC"
作者：Song et al.
发表：TCSVT 2022
学习重点：
  - VVC更复杂的四叉树+二叉树+三叉树划分的预测
  - 分层预测策略
  - 如何集成到VTM参考软件
第四类：面向机器视觉的代理编码器（最前沿）
论文10：面向机器的编码（重要新方向）
标题："Towards Coding for Human and Machine Vision: A Scalable Image Coding Approach"
作者：Choi et al.
发表：ICME 2021
学习重点：
  - 人眼观看 vs. 机器分析的不同需求
  - 代理任务网络（检测/分割网络）作为编码目标
  - 率失真-准确率联合优化
​
论文11：特征压缩与代理编码器
标题："Image Coding for Machines with Omnipotent Feature"
作者：Liu et al.
发表：arXiv 2021
学习重点：
  - 面向下游任务的特征压缩
  - 任务网络作为"软标签"生成器（代理角色）
​
论文12：VCM（Video Coding for Machines）方向综述
标题："Video Coding for Machines: A Paradigm of Collaborative Compression and Intelligent Analytics"
作者：Duan et al.
发表：TIP 2020
学习重点：
  - VCM的整体框架
  - 代理任务模型在VCM中的角色
  - 这个方向是MPEG标准化的重点
第14周：代理编码器进阶与联合优化
Week 14 Part 1：与前处理联合优化（第1-2天）
必读论文
论文13：前处理+代理编码器联合优化（核心）
标题："Optimizing Video Prediction via Video Frame Interpolation"
（这篇偏向理解联合优化思路，非编码专向）
​
更直接的工作：
标题："Perception-Distortion Tradeoff"
作者：Blau & Michaeli
发表：CVPR 2018
学习重点：
  - 感知质量和失真之间的根本矛盾
  - 代理编码器训练时如何选择优化目标
  - 这个权衡在前处理设计中非常重要
​
论文14：端到端前处理优化（最直接）
标题："Learning-Based Preprocessing for Video Compression"
（搜索相关工作，这个方向在ICIP/DCC/PCS会议上有很多论文）
​
具体可参考：
标题："Encoder-Side Deep Metric Learning for Video Compression Enhancement"
或
标题："Pre- and Post-Filtering for the Versatile Video Coding Standard"
学习重点：
  - 如何在有代理编码器的情况下端到端训练前处理网络
  - 训练稳定性问题及解决方案
  - 代理编码器的精度如何影响前处理效果
​
论文15：图像前处理与神经编码联合优化
标题："Variable-Rate Deep Image Compression through Spatially-Adaptive Feature Transform"
作者：Song et al.
发表：ICCV 2021
代码：https://github.com/micmic123/QmapCompression
学习重点：
  - 空间自适应的码率分配
  - 这是前处理思想在端到端编码中的体现
  - QMap的设计思路
Week 14 Part 2：与后处理联合优化（第3-4天）
必读论文
论文16：In-loop神经网络滤波集成（重要）
标题："Deep Neural Network-Based Luma and Chroma QP Offset Decision for Versatile Video Coding"
作者：相关VVC工作
更重要的：
标题："CNN-based Intra-prediction for Lossless HEVC"
作者：Song et al.
发表：TCSVT 2019
​
核心论文：
标题："DeepSIC: Deep Joint Source-Channel Image Communication"
（理解编码与后处理联合优化的思路框架）
​
更直接的工作：
标题："Jointly Optimizing Codewords and Post-Processing for JPEG Image Compression"
（搜索类似工作）
​
实际重要论文：
标题："CVEGAN: A Perceptually-inspired GAN for Compressed Video Enhancement"
作者：Ma et al.
发表：arXiv 2021
代码：可在GitHub搜索CVEGAN
学习重点：
  - 后处理GAN的训练策略
  - 如何将编码器特性（QP、分区信息）注入后处理网络
  - 主观质量优化 vs. 客观指标优化的权衡
​
论文17：后处理与编码器联合优化
标题："Enhancing Quality for VVC Compressed Videos by Jointly Exploiting Spatial Details and Temporal Structure"
（搜索 "joint optimization post-processing VVC" 相关工作）
学习重点：
  - 如何将后处理网络的梯度反传到编码参数
  - 代理编码器在此扮演的角色
  - 实际工程中的近似策略
Week 14 Part 3：强化学习视角的代理编码器（第5天）
论文18：强化学习用于编码决策
标题："Reinforcement Learning for Variable Bit Rate Image Compression"
作者：Mentzer et al. 相关方向
更具体：
标题："A Reinforcement Learning Approach to Bit-allocation in Image Compression"
（搜索RL-based rate control相关工作）
​
重要论文：
标题："Adaptive Bitrate Video Streaming with Super Resolution"
（理解RL在码率自适应中的应用，间接涉及代理编码器）
​
核心视角论文：
标题："Optimizing Video Encoding with Temporal Information Using Reinforcement Learning"
（搜索相关工作，理解RL如何绕过不可微问题）
学习重点：
  - 策略梯度方法绕过不可微问题
  - 奖励函数设计（BD-Rate增益作为奖励）
  - 这是代理编码器的另一种思路：不近似，而是绕过
第15周：代理编码器工程实现
Week 15 Part 1：手写可微JPEG代理（第1-2天）
实现步骤与学习顺序
任务：从零实现完整的可微JPEG编码器
​
Step1：实现各个可微模块
需要实现的模块清单：
  模块1：RGB to YCbCr变换
    - 线性变换，天然可微
    - 注意数值范围处理
  
  模块2：色度下采样（4:2:0）
    - 用双线性插值实现可微下采样
    - 理解下采样对梯度传播的影响
  
  模块3：8x8分块DCT
    - DCT是线性变换，可以用矩阵乘法实现
    - 预计算8x8 DCT基矩阵作为固定权重
    - 实现技巧：用F.conv2d实现分块DCT
  
  模块4：量化（核心难点）
    实现三种近似方式并对比：
    方式A：直通估计器（STE）
      前向：round(x / Q_table) * Q_table
      反向：梯度直接通过，忽略round操作
    方式B：软量化
      用sin函数近似：x - (Q/2π)·sin(2πx/Q)
    方式C：加性均匀噪声
      训练时：x + Uniform(-Q/2, Q/2)
      推理时：round(x/Q)*Q
  
  模块5：逆量化+逆DCT+合并
    - 这些步骤都是可微的
​
Step2：端到端测试可微性
  用torch.autograd.gradcheck验证梯度正确性
​
Step3：训练验证
  - 训练一个简单前处理网络（3层CNN）
  - 损失函数：前处理→可微JPEG→重建图像 的MSE
  - 对比：有/无可微JPEG训练的前处理效果差异
​
参考代码：
  DiffJPEG：https://github.com/mlomnitz/DiffJPEG
  要求：先不看源码，自己实现后再对比
Week 15 Part 2：HEVC代理编码器实现（第3-5天）
实现策略
HEVC代理编码器比JPEG复杂得多，
工程上有两种实现路线：
​
路线A：黑盒近似（推荐初学者）
  思路：不模拟内部流程，直接学习
        输入图像/块 → 代理网络 → 编码重建结果
  
  实现步骤：
  Step1：数据收集
    - 下载标准测试序列（Xiph.org / CLIC数据集）
    - 用HM/FFmpeg以QP=22/27/32/37编码所有图像
    - 构建数据集：{原始图像, QP, 重建图像, 实际码率}
  
  Step2：代理网络设计
    输入：原始图像块(64x64) + QP嵌入向量
    网络结构：
      - QP嵌入：将QP标量映射为通道维度向量（类似条件BN）
      - 主干网络：5-8层残差网络
      - 输出：预测的编码重建图像块
    
    QP条件注入方式（两种，都要理解）：
      方式1：QP→Embedding→加到特征图
      方式2：QP→预测BN的gamma/beta参数（SPADE思路）
  
  Step3：损失函数设计
    L_total = λ1 * L_distortion + λ2 * L_rate
    
    L_distortion：预测重建 vs. 真实重建的MSE/L1
    L_rate：预测码率 vs. 真实码率的MSE
    （码率可以用bpp表示：bits / (H*W)）
  
  Step4：评测代理精度
    指标1：重建质量近似误差
      PSNR(代理输出, HM真实输出) > 40dB为合格
    指标2：码率预测误差
      |预测bpp - 真实bpp| / 真实bpp < 10%为合格
    指标3：RD曲线相似性
      代理模型的RD曲线 vs. HM的RD曲线的形状相似性
​
路线B：白盒近似（理解原理后尝试）
  思路：模拟HEVC内部关键步骤
  简化版流程：
    原始块 → 帧内预测近似 → DCT → 软量化 → 逆DCT → 重建
  
  仅实现帧内预测+变换量化部分（跳过熵编码）
  码率用量化后非零系数数量近似估计
Week 15 Part 3：代理编码器质量验证（补充）
验证实验设计：
​
实验1：代理编码器近似质量
  - 在Kodak测试集上对比代理输出 vs. 真实HM输出
  - 画散点图：横轴真实PSNR，纵轴代理预测PSNR
  - 计算相关系数，越高越好
​
实验2：用代理训练前处理网络
  - 训练一个简单前处理CNN（使用代理编码器梯度）
  - 在真实HM上评测前处理效果
  - 对比：随机前处理 vs. 代理优化前处理 的BD-Rate
​
实验3：代理精度对优化效果的影响
  - 训练精度不同的代理模型（通过限制训练数据量）
  - 分析代理精度 vs. 前处理BD-Rate增益的关系
  - 这个实验帮助理解代理编码器的"瓶颈"在哪里
第16周：前沿方向与综合实战
Week 16 Part 1：前沿论文精读（第1-3天）
端到端系统中的代理角色
论文19：编码感知图像增强（最相关）
标题："Proximal Splitting Networks for Image Restoration"
（理解代理优化框架）
​
更直接：
标题："Encoding Enhancement for Learned Image Compression via Domain Adaptation"
（搜索相关工作）
​
重要论文：
标题："Sandwiched Image Compression: Extending the Usage of Standard Codecs with Neural Network Wrappers"
作者：Feng et al.
发表：相关顶会/期刊
学习重点：
  - 在标准编解码器外层包裹神经网络
  - 前处理网络 + 标准编码器 + 后处理网络
  - 代理编码器在训练此类系统中的核心作用
  - 这是目前工业界非常实用的方案
​
论文20：可学习量化与代理结合
标题："Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules"
作者：Cheng et al.
发表：CVPR 2020
（已在第3阶段提到，这次重点看量化部分）
学习重点：
  - 量化的各种近似在端到端编码中的实际效果对比
  - 推理时真实量化 vs. 训练时近似量化的gap
​
论文21：NIC（神经图像编码）中的代理问题综述视角
标题："Neural Image Compression: Generalization, Robustness, and Spectral Biases"
作者：Cao et al.
发表：NeurIPS 2022
学习重点：
  - 端到端编码训练中量化近似的影响分析
  - 泛化性问题：训练图像分布 vs. 测试图像分布
  - 这篇论文对理解代理编码器的局限性很有帮助
VCM方向的代理编码器（最前沿）
论文22：任务导向编码中的代理
标题："Task-Driven Image Compression"
作者：Chamain et al. 或类似工作
发表：相关顶会
​
更重要：
标题："Learned Image Compression for Machine Perception"
作者：Fischer et al.
发表：arXiv 2021
学习重点：
  - 检测/分割网络作为"代理任务模型"
  - 如何用任务网络的梯度优化编码器
  - 人眼质量 vs. 机器视觉精度的权衡
​
论文23：MPEG-VCM标准化方向
标题："An Overview of Core Experiments for Video Coding for Machines"
作者：MPEG工作组文档（可在网上搜索）
学习重点：
  - 了解VCM标准化的测试条件
  - 理解工业界对代理编码器的实际需求
Week 16 Part 2：核心代码库深度学习（第4-5天）
必须研究的代码库
代码库1：CompressAI中的代理相关部分
地址：https://github.com/InterDigitalCom/CompressAI
重点研究文件：
  - compressai/entropy_models/entropy_models.py
    重点：量化的训练/推理不同行为的实现
    找到：_quantize方法，理解训练时加噪声 vs. 推理时round
  - compressai/models/google.py
    重点：EntropyBottleneck中的量化近似
  
研究任务：
  在CompressAI中找到所有与量化相关的代码
  理解：为什么要用加性均匀噪声代替round
  实验：将加性噪声替换为STE，对比训练效果
​
代码库2：DiffJPEG
地址：https://github.com/mlomnitz/DiffJPEG
研究任务：
  - 逐行阅读所有源码
  - 理解每个可微近似的实现细节
  - 在此基础上添加可控质量因子（quality factor作为输入）
​
代码库3：DCVC（端到端视频编码）
地址：https://github.com/microsoft/DCVC
重点研究：
  - 训练代码中的量化处理
  - 运动编码和残差编码中各自的代理策略
  - 联合优化的损失函数实现
​
代码库4：VCM相关代码
地址：https://github.com/JiangWeibeta/VCM-related（搜索相关仓库）
或：FCM（Feature Compression for Machines）相关开源代码
研究任务：
  - 理解检测/分割任务网络如何接入编码训练流程
  - 理解特征量化的代理实现
代理编码器学习路径总结图
Week 13：理论基础
    ↓
  理解不可微问题（量化/决策/熵编码）
    ↓
  学习三种解决方案（STE/软量化/加噪声）
    ↓
  阅读可微JPEG、HEVC代理相关论文
​
Week 14：联合优化
    ↓
  理解代理编码器如何嵌入训练流程
    ↓
  学习前处理/后处理与代理联合优化论文
    ↓
  理解RL绕过不可微的另一种思路
​
Week 15：工程实现
    ↓
  手写可微JPEG（从零实现）
    ↓
  实现HEVC黑盒代理编码器
    ↓
  验证代理精度和优化效果
​
Week 16：前沿与综合
    ↓
  Sandwiched系统（工业界方案）
    ↓
  VCM方向（学术前沿）
    ↓
  综合实战项目
综合实战项目（贯穿理解）
项目：基于代理编码器的端到端前处理系统
项目目标：
  训练一个前处理网络，使得
  "前处理后用标准HEVC编码"的RD性能
  优于"直接用HEVC编码"
​
系统结构：
  原始图像
      ↓
  前处理网络（待优化，3-5层CNN+残差）
      ↓
  代理HEVC编码器（固定，已训练好）
      ↓
  预测重建图像
      ↓
  损失函数：L = MSE(预测重建, 原始) + β * 预测码率
​
训练流程：
  Phase1（第15周）：
    - 训练代理HEVC编码器
    - 数据：HM编码的{原始,重建,码率}三元组
    - 验证代理精度达标
​
  Phase2（第16周）：
    - 冻结代理编码器
    - 训练前处理网络
    - 损失函数中加入感知损失项
​
  Phase3（评测）：
    - 将训练好的前处理网络应用到真实HM编码
    - 计算BD-Rate增益
    - 对比有/无代理训练的差异
​
评测指标：
  - BD-Rate（相对于无前处理的HM）
  - 主观质量（VMAF评分）
  - 前处理网络推理时间（实时性要求）
  
期望结果：
  - BD-Rate节省 > 3%
  - 代理预测PSNR误差 < 0.3dB
  - 前处理时间 < 10ms/frame (1080p)
重要补充：论文查找方法
由于这个方向论文较新且细分，
部分论文需要主动搜索：
搜索渠道：
1. Google Scholar关键词组合：
   "surrogate encoder video coding"
   "differentiable video coding"
   "proxy codec neural network"
   "coding-aware preprocessing"
   "CNN-based in-loop filter HEVC/VVC"
   "rate distortion optimization deep learning"
​
2. 顶会/期刊重点关注：
   会议：CVPR / ICCV / ECCV / NeurIPS / ICLR
        DCC（Data Compression Conference）专业对口
        ICIP / ICME / PCS
        ISCAS / VCIP
   期刊：IEEE TIP（图像处理汇刊）
        IEEE TCSVT（视频技术汇刊）
        IEEE TPAMI
​
3. arXiv每日追踪：
   订阅 cs.CV 和 eess.IV 方向
   关键词过滤：compression / codec / encoding
​
4. 论文引用链追踪：
   找到一篇核心论文后
   → 看它引用了哪些论文（往前追）
   → 看哪些论文引用了它（往后追，用Google Scholar）
   这是最高效的文献调研方式
四个月完整学习计划总览
整体时间轴
Month 1（第1-4周）：基础建设期
├── Week 1：图像视频基础 + 传统编码概念
├── Week 2：PyTorch深度学习基础
├── Week 3：图像超分与复原基础
└── Week 4：感知损失与图像质量评价
​
Month 2（第5-8周）：编码专项期
├── Week 5：图像去噪与编码伪影去除
├── Week 6：传统编码标准深入（HEVC/VVC）
├── Week 7：传统编码实践与分析
└── Week 8：端到端图像编码基础
​
Month 3（第9-12周）：核心技术期
├── Week 9：端到端编码进阶
├── Week 10：端到端视频编码
├── Week 11：前处理方向
└── Week 12：后处理方向
​
Month 4（第13-16周）：代理编码器专项期
├── Week 13：代理编码器基础理论与论文
├── Week 14：联合优化框架
├── Week 15：工程实现
└── Week 16：前沿方向与综合实战
Month 1 详细计划（补全版）
Week 1：图像视频基础
每日任务分解
Day 1-2：图像基础
学习内容：
  - 图像数字表示：像素、位深度、动态范围
  - 色彩空间：RGB原理、YCbCr原理
  - RGB到YCbCr的转换公式（BT.601 / BT.709 / BT.2020区别）
  - 图像的内存布局：HWC vs. CHW
  - 常见图像格式：PNG（无损）/ JPEG（有损）/ RAW
​
学习资料：
  主要：《数字图像处理》第2章
  补充：OpenCV官方文档 Color Space Conversions部分
  
实践：
  用OpenCV读取图像
  手写RGB↔YCbCr转换（不用cv2.cvtColor，手写矩阵乘法）
  验证手写结果与cv2.cvtColor一致
​
Day 3-4：频域基础
学习内容：
  - 傅里叶变换的直觉理解（频率=变化快慢）
  - 2D DFT：图像的频谱含义
  - DCT vs. DFT的区别（为什么编码用DCT）
  - 8x8 DCT的64个基函数的视觉含义
  - 低频系数 vs. 高频系数对应的图像内容
​
学习资料：
  3Blue1Brown：傅里叶变换视频（YouTube）
  《数字图像处理》第4章（频率域处理）
  
实践：
  用numpy实现2D DCT
  对一张图像做DCT，可视化频谱
  将高频系数置零，观察图像变化（理解JPEG压缩原理）
​
Day 5-7：视频基础与质量评价
学习内容：
  - 视频的帧率、分辨率、时长
  - 帧类型：I帧/P帧/B帧的概念
  - 图像质量评价：
    PSNR：计算公式、典型值范围（>40dB为高质量）
    SSIM：结构相似性，比PSNR更符合人眼
    MS-SSIM：多尺度版本
    VMAF：Netflix开发的视频质量指标
    LPIPS：基于深度特征的感知距离
  - 码率单位：bpp（bits per pixel）/ kbps / Mbps
​
学习资料：
  VMAF官方文档：https://github.com/Netflix/vmaf
  SSIM原始论文：
    标题："Image Quality Assessment: From Error Visibility to Structural Similarity"
    作者：Wang et al.
    发表：TIP 2004
    （这篇必读，SSIM是整个方向的基础评价指标）
​
实践：
  手写PSNR计算函数
  用scikit-image计算SSIM
  安装VMAF，评测一段视频
  用ffmpeg以不同码率编码同一视频，计算各质量指标
  画出码率-质量曲线（Rate-Quality Curve）
Week 2：PyTorch深度学习基础
每日任务分解
Day 1-2：PyTorch核心操作
学习内容：
  - Tensor创建、索引、切片、reshape
  - 广播机制（Broadcasting）
  - 自动求导：requires_grad、backward、grad
  - GPU操作：.cuda() / .to(device)
  - 内存管理：detach、no_grad
​
学习资料：
  PyTorch官方Tutorial：
    "Learn the Basics"全部6个章节
    地址：pytorch.org/tutorials/beginner/basics
​
实践：
  完成官方Tutorial的所有代码练习
  手写一个标量函数的梯度验证：
    f(x) = x^3 + 2x^2 + x
    用PyTorch自动求导 vs. 手算导数对比
​
Day 3-4：网络定义与训练
学习内容：
  - nn.Module的定义方式
  - 常用层：Conv2d、BatchNorm2d、ReLU、Linear
  - 损失函数：MSELoss、L1Loss、BCELoss
  - 优化器：Adam、SGD
  - 学习率调度：StepLR、CosineAnnealingLR
  - 标准训练循环结构
​
学习资料：
  李沐《动手学深度学习》第5章、第6章
  B站视频配套
​
实践：
  在CIFAR-10上实现5层CNN分类器
  要求代码包含：
    完整训练循环
    验证集评测
    TensorBoard记录loss和accuracy
    模型checkpoint保存/加载
​
Day 5-7：图像任务专项训练
学习内容：
  - 图像数据集的Dataset/DataLoader写法
  - 图像增强：torchvision.transforms
  - 图像任务的评测：PSNR/SSIM在PyTorch中的计算
  - 可视化：torchvision.utils.make_grid
  - 混合精度训练：torch.cuda.amp（了解概念）
​
实践：
  实现一个图像去噪数据集类：
    读取干净图像
    动态添加高斯噪声
    返回（噪声图像，干净图像）对
  训练一个5层残差去噪网络
  评测PSNR：噪声图像 vs. 网络输出 vs. 原始图像
  可视化：用make_grid展示一批结果
Week 3-4：详细补全
Week 3 每日任务：
​
Day 1-2：SRCNN精读与复现
  - 精读论文，理解三个卷积层的分工
  - 手写SRCNN（不参考任何代码）
  - 在Set5数据集上评测
  - 理解双三次插值上采样的位置（在网络输入前）
​
Day 3-4：EDSR精读与BasicSR环境搭建
  - 精读EDSR论文，重点理解去BN的原因
  - 搭建BasicSR环境
  - 跑通EDSR训练脚本
  - 对比有BN和无BN的训练曲线
​
Day 5-7：数据Pipeline深入理解
  - 阅读BasicSR的data目录所有代码
  - 理解paired图像数据集的构建
  - 理解图像patch随机裁剪的实现
  - 理解数据增强（随机翻转、旋转）
  - 实践：修改BasicSR数据加载，加入自定义增强
​
Week 4 每日任务：
​
Day 1-2：感知损失深入
  - 精读Johnson et al. 2016论文
  - 理解VGG各层特征的语义层次
  - 手写PerceptualLoss类
  - 实验：L1 Loss vs. 感知损失的视觉效果对比
​
Day 3-5：GAN训练
  - 精读SRGAN论文
  - 理解判别器的设计（PatchGAN思路）
  - 理解GAN训练的不稳定性问题
  - 阅读BasicSR中的GAN训练代码
  - 实践：跑通ESRGAN训练，理解两个优化器的交替更新
​
Day 6-7：质量指标深入
  - 安装并使用LPIPS
  - 理解PSNR高但视觉效果差的原因
  - 收集不同模型的输出，做主观质量对比
  - 理解感知质量-失真权衡（Perception-Distortion Tradeoff）
Month 2 详细计划（补全版）
Week 5：去噪与编码伪影
每日任务分解
Day 1-2：DnCNN精读
  学习内容：
    - 残差学习用于去噪的原理
    - 批归一化在去噪中的作用
    - 盲去噪：单个模型处理多种噪声级别
  实践：
    手写DnCNN
    在BSD68数据集上评测
    对比sigma=15/25/50的去噪效果
Day 3-4：FFDNet精读
  学习内容：
    - 噪声级别图（noise level map）作为额外输入
    - 空间变化噪声的处理
    - 下采样加速推理的技巧
    - 可控去噪强度的设计思路
      （这个思路直接对应后处理中QP条件输入）
  实践：
    手写FFDNet核心结构
    实验：固定噪声级别 vs. 空间变化噪声级别图输入
    对比FFDNet和DnCNN在不同噪声级别下的性能
​
Day 5-6：ARCNN精读（编码伪影去除）
  学习内容：
    - JPEG/HEVC编码引入的三类伪影：
      块效应（Blocking）：8x8块边界不连续
      振铃效应（Ringing）：边缘附近的震荡
      色度模糊（Chroma Blurring）：色度分量模糊
    - 训练数据构造：
      原始图像 → JPEG压缩（不同quality factor）→ 压缩图像
      构成（压缩图像, 原始图像）训练对
    - 网络结构：4层卷积，学习残差
​
  实践：
    构建JPEG伪影去除数据集
      使用PIL以quality=10/20/30/40压缩图像
      构建训练/验证/测试集
    手写ARCNN并训练
    评测：在LIVE1数据集上计算PSNR/SSIM
    可视化：放大观察块边界的伪影去除效果
​
Day 7：综合实验
  任务：对比实验
    同一张HEVC压缩图像，分别用：
      DnCNN后处理
      ARCNN后处理
      无后处理
    计算PSNR/SSIM/LPIPS
    分析：专门针对编码伪影设计的网络
          vs. 通用去噪网络的效果差异
    记录实验结论，这是理解"任务专用"设计的重要认知
Week 6：传统编码标准深入
每日任务分解
Day 1-2：JPEG编码完整流程
  学习内容（按流程顺序）：
    Step1. 色彩空间转换：RGB → YCbCr
    Step2. 色度下采样：4:2:0采样格式
    Step3. 8x8分块
    Step4. DCT变换：每个8x8块的64个系数含义
    Step5. 量化：
      亮度量化表 vs. 色度量化表
      Quality Factor如何缩放量化表
      量化步长 = 量化表值 × 缩放因子
    Step6. Zigzag扫描：从低频到高频
    Step7. 游程编码（RLE）
    Step8. 霍夫曼编码
​
  学习资料：
    《数字图像处理》第8章
    JPEG标准文档摘要（网上有很多教程版本）
    推荐：
      标题："The JPEG Still Picture Compression Standard"
      作者：Wallace
      发表：CACM 1991
      这是JPEG的经典综述论文，必读
​
  实践：
    用Python从零实现JPEG编码器（不用PIL）
    实现每个步骤为独立函数
    编码一张灰度图像，与PIL的JPEG输出对比
    重点：手写量化步骤，理解量化误差的产生
​
Day 3-5：HEVC关键技术
  学习内容：
​
  第一部分：编码结构
    - CTU（Coding Tree Unit）：最大64×64
    - 四叉树划分：CTU→CU（Coding Unit）
    - PU（Prediction Unit）：预测的基本单元
    - TU（Transform Unit）：变换的基本单元
    - 理解这四层结构的层次关系
​
  第二部分：帧内预测
    - 35种预测方向（DC/Planar + 33个角度方向）
    - 参考像素的获取和滤波
    - 最可能模式（MPM）机制
​
  第三部分：帧间预测
    - 运动估计：在参考帧中找最相似块
    - 运动矢量（MV）的精度：1/4像素精度
    - 运动补偿插值滤波器
    - Merge模式 vs. AMVP模式
​
  第四部分：变换与量化
    - DCT/DST多种变换核
    - QP与量化步长的关系（每6个QP步长翻倍）
    - 率失真优化（RDO）：D + λR
    - Lambda的计算方式
​
  第五部分：环路滤波
    - 去块滤波（Deblocking Filter）：消除块效应
    - SAO（Sample Adaptive Offset）：补偿统计偏差
    - 这两个滤波器的位置：在重建帧写入参考帧之前
​
  学习资料：
    必读综述论文：
      标题："Overview of the High Efficiency Video Coding (HEVC) Standard"
      作者：Sullivan et al.
      发表：IEEE TCSVT 2012
      这篇30页的综述是HEVC最权威的入门材料
​
    补充：
      《新一代高效视频编码H.265/HEVC原理、标准与实现》
      重点章节：帧内预测、帧间预测、变换量化
​
Day 6-7：HM参考软件实践
  环境搭建：
    下载HM：https://hevc.hhi.fraunhofer.de/
    按文档编译（Linux推荐，Windows也可以）
    下载标准测试序列：
      HEVC CTC测试序列（可从网上搜索获取）
      或使用Xiph.org的开源视频序列
​
  实践任务：
    Task1：基础编码
      用encoder_intra_main.cfg配置
      以QP=22/27/32/37编码BasketballDrill序列
      记录每种QP的：码率(kbps)、PSNR(Y/U/V)、编码时间
​
    Task2：画RD曲线
      以码率为横轴、PSNR为纵轴
      画出本序列的Rate-Distortion曲线
      用Python matplotlib实现
​
    Task3：理解BD-Rate
      学习BD-Rate计算方法
      推荐工具：https://github.com/Anserw/Bjontegaard_metric
      计算两组RD曲线之间的BD-Rate差值
Week 7：传统编码深入实践
每日任务分解
Day 1-2：编码信息可视化分析
  工具准备：
    安装YUView：https://github.com/IENT/YUView
    这是最好的YUV视频分析工具，可以显示编码统计信息
​
  分析任务：
    Task1：CU划分可视化
      用YUView打开HM编码输出
      观察不同内容区域的CU划分深度
      记录：平坦区域（天空）vs. 纹理区域（草地）的划分差异
      这个观察对理解"为什么用神经网络预测划分"很重要
​
    Task2：预测模式分布
      统计帧内预测模式的分布直方图
      哪些角度方向最常用？
      平坦图像 vs. 纹理图像的预测模式分布差异
​
    Task3：残差分布分析
      提取编码残差信号
      画残差值的直方图
      验证：残差信号接近拉普拉斯分布
      这是熵编码模型设计的数学基础
​
Day 3-4：率失真理论深入
  学习内容：
    - 香农率失真函数R(D)的数学定义
    - 拉格朗日优化：min D + λR
    - Lambda参数的物理意义
    - QP和Lambda的对应关系：λ = 0.85 × 2^((QP-12)/3)
    - 为什么端到端学习要用同样的率失真目标
​
  必读论文：
    标题："Rate-Distortion Optimized Image Compression
           using Learned Encoder Enhancements"
    作者：Klopp et al.
    发表：ECCV 2020 Workshop
    学习重点：
      如何用率失真理论指导神经网络编码的设计
​
  实践：
    用不同λ值训练同一个简单编码网络
    观察λ变化对码率-失真权衡的影响
    理解：大λ → 更注重码率压缩，小λ → 更注重质量
Day 5：FFmpeg基础操作
  学习内容：
    - FFmpeg架构：libavcodec/libavformat/libavfilter
    - 常用命令行操作：
      转码：ffmpeg -i input.mp4 -c:v libx265 -crf 28 output.mp4
      提取帧：ffmpeg -i input.mp4 -f image2 frames/%04d.png
      YUV操作：ffmpeg -i input.mp4 -pix_fmt yuv420p output.yuv
      码率控制：-b:v 1M（固定码率） vs. -crf 23（固定质量）
    - 理解CRF/QP/码率三种控制方式的区别
    - 常用滤镜：scale/crop/pad
​
  实践任务：
    用ffmpeg构建一套自动化评测脚本
    输入：原始视频序列
    输出：在多个QP下的编码结果 + PSNR/SSIM报告
    用Python subprocess调用ffmpeg
    自动收集结果并画RD曲线
​
Day 6：用Python控制FFmpeg编码
  学习内容：
    - ffmpeg-python库的使用
    - 如何用Python构建编码Pipeline
    - 如何解析ffmpeg输出获取编码统计信息
    - 如何批量处理视频数据集
​
  学习资料：
    ffmpeg-python文档：
      https://github.com/kkroening/ffmpeg-python
    
  实践任务：
    写一个Python类：VideoCodecWrapper
    功能：
      encode(input_path, qp) → 返回重建视频路径和码率
      evaluate(original, reconstructed) → 返回PSNR/SSIM/VMAF
      sweep_qp(input_path, qp_list) → 批量编码并返回RD数据
    这个工具类在后续所有实验中都会用到
​
Day 7：构建标准测试数据集
  任务：搭建完整的测试评测环境
​
  数据集准备：
    图像数据集：
      Kodak：24张无损图像，标准图像压缩测试集
        下载：http://r0k.us/graphics/kodak/
      CLIC：图像压缩挑战赛数据集
        下载：https://www.compression.cc/
      DIV2K：高质量图像数据集
        下载：https://data.vision.ee.ethz.ch/cvl/DIV2K/
​
    视频数据集：
      UVG：高质量视频序列
        下载：http://ultravideo.fi/
      HEVC CTC序列：Class A-E标准测试序列
      MCL-JCV：面向视频压缩的主观质量数据集
​
  评测框架搭建：
    建立统一的评测脚本结构：
    ├── evaluate_image_codec.py   # 图像编码评测
    ├── evaluate_video_codec.py   # 视频编码评测
    ├── compute_bdrate.py         # BD-Rate计算
    ├── plot_rd_curve.py          # RD曲线绘制
    └── compare_methods.py        # 多方法对比
​
    评测指标统一：
      客观指标：PSNR-Y / MS-SSIM / LPIPS / VMAF
      码率单位：统一用bpp（图像）或kbps（视频）
      对比基准：JPEG / BPG / HEVC HM / VTM
Week 8 每日任务分解
Day 1-2：变分自编码器基础（VAE）
  学习内容（端到端编码的理论基础）：
    - 自编码器（AE）的基本结构
    - 变分自编码器（VAE）的概率解释
    - 重参数化技巧（Reparameterization Trick）
    - ELBO（证据下界）的推导
    - VAE与图像压缩的关系：
      编码器 = 分析变换
      潜变量 = 压缩表示
      解码器 = 合成变换
      先验分布 = 熵模型
​
  必读论文：
    标题："Auto-Encoding Variational Bayes"
    作者：Kingma & Welling
    发表：ICLR 2014
    学习重点：
      ELBO推导过程
      重参数化技巧
      这是理解端到端编码理论的必要基础
​
  实践：
    手写VAE在MNIST上训练
    理解KL散度项对应码率，重建损失对应失真
    这个对应关系是端到端编码的核心思想
​
Day 3-4：Ballé2017精读（端到端编码第一篇）
  论文：
    标题："End-to-end Optimized Image Compression"
    作者：Ballé et al.
    发表：ICLR 2017
​
  精读步骤：
    第一遍（1小时）：
      理解整体框架：分析变换→量化→熵编码→合成变换
      理解与传统编码的对应关系
      记录不懂的公式
​
    第二遍（2小时）：
      推导量化近似：为什么用加性均匀噪声
        训练时：ŷ = y + U(-0.5, 0.5)
        推理时：ŷ = round(y)
      推导GDN激活函数的公式
      理解为什么用GDN而不是ReLU
​
    第三遍（1小时）：
      阅读附录数学推导
      理解率失真损失函数：
        L = λ·D + R
        D = MSE(x, x̂)
        R = -log p(ŷ)（熵）
​
  实践：
    在CompressAI中找到对应实现
    对照论文逐行阅读代码
    运行训练脚本，观察loss曲线
​
Day 5：Ballé2018精读（超先验模型）
  论文：
    标题："Variational Image Compression with a Scale Hyperprior"
    作者：Ballé et al.
    发表：ICLR 2018
​
  精读重点：
    理解超先验的动机：
      为什么ŷ的各元素不独立？
      空间相关性如何影响编码效率？
    理解侧信息z的编码：
      z = h_a(y)（超分析变换）
      z̃ = quantize(z)
      z̃通过另一个熵模型编码
    理解条件高斯模型：
      p(ŷ|z̃) = N(0, σ²)
      σ由z̃预测
​
  与Ballé2017对比：
    画出两个模型的架构对比图（手绘）
    理解超先验增加了什么、解决了什么问题
​
Day 6：Minnen2018精读（自回归上下文）
  论文：
    标题："Joint Autoregressive and Hierarchical Priors
           for Learned Image Compression"
    作者：Minnen et al.
    发表：NeurIPS 2018
​
  精读重点：
    自回归上下文模型的工作方式：
      已解码的ŷ_{i,j}作为上下文
      预测下一个符号的分布参数
    与超先验的结合：
      联合预测μ和σ（均值和方差）
      比Ballé2018只预测σ更精确
    
    重要认知：
      自回归解码是串行的（慢）
      这是后续工作（channel-wise等）要解决的问题
​
  实践：
    在CompressAI中运行Minnen2018模型
    对比Ballé2017/Ballé2018/Minnen2018的RD性能
    画三条RD曲线在同一图中，与HEVC HM对比
​
Day 7：搭建完整端到端编码实验环境
  任务：
    配置完整的CompressAI实验环境
​
  步骤：
    Step1：环境配置
      pip install compressai
      验证安装：运行官方示例
​
    Step2：训练数据准备
      下载CLIC训练集
      写DataLoader：
        随机裁剪256×256 patch
        归一化到[0,1]
​
    Step3：训练Ballé2018模型
      使用λ=0.0483（对应中等质量）
      训练至少100epoch
      记录：bpp loss、distortion loss、total loss曲线
​
    Step4：评测
      在Kodak数据集上评测
      计算每张图像的bpp和PSNR
      画RD曲线，与JPEG/BPG对比
      计算BD-Rate相对于JPEG的增益
Week 9：端到端编码进阶
Day 1-2：Cheng2020精读（注意力机制）
  论文：
    标题："Learned Image Compression with Discretized
           Gaussian Mixture Likelihoods and Attention Modules"
    作者：Cheng et al.
    发表：CVPR 2020
  
  精读重点：
    注意力模块的设计：
      残差注意力块（Residual Attention Block）
      通道注意力 + 空间注意力的组合
      为什么注意力有助于提升压缩性能
    高斯混合模型（GMM）：
      用K个高斯分布的混合建模符号分布
      比单高斯模型更灵活准确
      GMM的参数预测：K个(π, μ, σ)
​
  实践：
    在CompressAI中运行Cheng2020模型
    对比加入注意力前后的RD性能差异
    可视化注意力图：哪些区域获得更多关注？
​
Day 3-4：可变码率编码
  必读论文：
    标题："Variable-Rate Deep Image Compression
           with a Conditional Autoencoder"
    作者：Cui et al.
    发表：CVPR 2021
  
  学习重点：
    传统方法：训练多个λ对应多个模型
    可变码率：单个模型支持所有码率点
    条件自编码器的实现方式：
      λ值嵌入为条件向量
      注入到每个卷积层（类似条件BN）
    训练策略：随机采样λ值训练
​
  补充论文：
    标题："Slimmable Compressive Autoencoders
           for Practical Neural Image Compression"
    作者：Yang et al.
    发表：CVPR 2021
    学习重点：可切换网络宽度实现可变码率
​
  实践：
    修改CompressAI中的模型
    加入λ条件嵌入
    训练可变码率模型
    在λ=0.0018/0.0035/0.0067/0.0130/0.025/0.0483上评测
​
Day 5-6：空间自适应码率分配
  必读论文：
    标题："Variable-Rate Deep Image Compression through
           Spatially-Adaptive Feature Transform"
    作者：Song et al.
    发表：ICCV 2021
    代码：https://github.com/micmic123/QmapCompression
  
  学习重点：
    空间QMap的概念：
      不同区域分配不同码率
      显著区域分配更多比特
      平坦区域分配更少比特
    特征变换的条件注入方式
    QMap的生成策略（手动 or 自动）
​
  实践：
    克隆QmapCompression仓库
    运行推理代码
    可视化QMap：不同内容区域的码率分配
    实验：手动设置QMap，观察对局部质量的影响
​
Day 7：端到端图像编码综合对比实验
  实验设计：
    在Kodak测试集上系统对比：
      传统方法：JPEG / JPEG2000 / BPG(HEVC) / VTM(VVC)
      神经网络方法：
        Ballé2017 / Ballé2018 / Minnen2018
        Cheng2020 / 可变码率模型
    
    评测维度：
      客观：PSNR / MS-SSIM / LPIPS（分别画RD曲线）
      效率：编码时间 / 解码时间
      BD-Rate：相对HEVC BM的增益
​
    产出：
      一份完整的实验报告
      包含RD曲线图、BD-Rate表格、速度对比
      这份报告是非常好的学习总结
Week 10：端到端视频编码
Day 1-2：视频编码基础复习与光流
  学习内容：
    光流（Optical Flow）：
      定义：相邻帧间像素的运动场
      与运动向量（MV）的关系和区别
      经典算法：Lucas-Kanade / Farneback
      深度学习光流：FlowNet / PWCNet / SPyNet
​
  必读论文（光流）：
    标题："SpyNet: Spatial Pyramid Network for
           Optical Flow Estimation"
    作者：Ranjan & Black
    发表：CVPR 2017
    代码：https://github.com/sniklaus/pytorch-spynet
    学习重点：
      空间金字塔光流估计
      轻量化设计（参数少，适合嵌入视频编码框架）
​
  实践：
    运行SPyNet推理代码
    可视化光流（HSV颜色编码）
    理解光流箭头的方向和幅度含义
​
Day 3-4：DVC精读（端到端视频编码第一篇）
  论文：
    标题："DVC: An End-to-end Deep Video
           Compression Framework"
    作者：Lu et al.
    发表：CVPR 2019
    代码：https://github.com/GuoLusjtu/DVC
​
  精读步骤：
    第一遍：理解整体框架
      与传统视频编码的对比：
        光流估计 ≈ 运动估计
        光流压缩 ≈ 运动信息编码
        运动补偿 ≈ 预测信号生成
        残差压缩 ≈ 残差编码
      
    第二遍：理解各个模块
      运动估计网络：SPyNet变体
      运动压缩：把图像压缩用于光流压缩
      运动补偿：可变形卷积 or 双线性插值
      残差压缩：基于Ballé2018的图像压缩
​
    第三遍：理解训练策略
      两阶段训练：
        Stage1：先训练图像压缩部分
        Stage2：端到端联合训练
      损失函数：同时优化运动和残差的率失真
​
  实践：
    克隆DVC仓库并配置环境
    在UVG数据集上评测
    可视化：光流图、运动补偿结果、残差图
​
Day 5：DCVC精读
  论文：
    标题："Deep Contextual Video Compression"
    作者：Li et al.
    发表：NeurIPS 2021
    代码：https://github.com/microsoft/DCVC
​
  精读重点：
    与DVC的核心区别：
      DVC：压缩残差（与传统相同思路）
      DCVC：条件编码（更本质的改进）
    上下文特征（Context Feature）：
      用对齐后的参考帧特征作为条件
      不是预测残差，而是在特征空间条件编码
    为什么条件编码比残差编码更有效：
      残差：像素域，信息冗余高
      条件：特征域，更紧凑的表示
​
Day 6：DCVC-DC精读（当前接近SOTA）
  论文：
    标题："Neural Video Compression with Diverse Contexts"
    作者：Li et al.
    发表：CVPR 2023
    代码：https://github.com/microsoft/DCVC
​
  精读重点：
    多样化上下文的设计：
      时序上下文：来自参考帧
      空间上下文：来自当前帧已编码区域
      通道上下文：通道方向自回归
    如何提升实用性：
      编解码速度优化
      GOP结构设计
      帧级码率控制
​
Day 7：视频编码综合实验
  实验：
    在UVG和HEVC CTC序列上评测：
      HEVC HM（基准）
      VTM（最新传统编码器）
      DVC
      DCVC / DCVC-DC
​
    评测指标：
      PSNR-Y（Y通道）
      MS-SSIM
      编解码速度（fps）
      BD-Rate相对HM
​
    分析：
      端到端视频编码在什么分辨率/内容上优势明显？
      速度上的差距有多大？（这是工业化的关键障碍）
Week 11：前处理方向
Day 1-2：前处理问题定义与综述
  学习内容：
    前处理的分类：
      1. 编码感知降噪：去除对编码不友好的信号成分
      2. 内容自适应平滑：在不重要区域减少纹理
      3. 编码前超分：提高分辨率后编码再下采样
      4. HDR/SDR转换：处理高动态范围内容
​
    关键认知：
      前处理 ≠ 图像增强
      目标是"让编码器更高效"而非"让图像看起来更好"
      前处理引入的改变在解码端通过后处理恢复
​
  必读论文：
    标题："Perception-Distortion Tradeoff"
    作者：Blau & Michaeli
    发表：CVPR 2018
    学习重点：
      感知质量-失真的根本矛盾
      不可能三角：感知质量/失真/码率
      这个理论框架指导前处理的设计目标
​
Day 3-4：编码感知前处理论文
  论文1：
    标题："Encoder-Side Deep Metric Learning for
           Video Compression Enhancement"
    （搜索相关方向论文）
​
  更直接论文：
    标题："Content-Adaptive Optimization for
           Neural Image Compression"
    作者：Campos et al.
    发表：CVPRW 2019
    学习重点：
      实例自适应优化（Instance-Adaptive）
      每张图像单独优化编码表示
      与训练时优化的区别
​
  论文2：
    标题："Improving Inference for Neural Image
           Compression"
    作者：Campos et al.
    发表：NeurIPS 2019
    学习重点：
      推理时的迭代优化
      如何在不改变网络参数下提升压缩性能
​
  论文3：
    标题："Sandwiched Image Compression: Extending
           the Usage of Standard Codecs with
           Neural Network Wrappers"
    作者：Feng et al.
    发表：ICME 2023（或相近版本）
    学习重点：
      三明治结构：前处理+标准编码+后处理
      端到端训练此类系统的方法
      代理编码器在其中的关键角色
​
Day 5-6：前处理实践
  实践任务：设计并训练编码前处理网络
​
  数据准备：
    使用DIV2K训练集（800张高质量图像）
    对每张图像用ffmpeg以QP=27/32/37编码
    构建三元组：{原始图像, QP, 编码重建图像}
​
  网络设计：
    前处理网络结构：
      输入：原始图像(3, H, W)
      架构：U-Net变体（7层，跳跃连接）
      输出：处理后图像(3, H, W)（与输入同分辨率）
      条件：QP值通过嵌入层注入
​
    损失函数设计（两种方案）：
      方案A（无代理编码器）：
        L = MSE(前处理输出, 原始图像) 
        问题：不能直接优化编码后的质量
​
      方案B（有代理编码器）：
        L = MSE(代理编码(前处理输出), 原始图像)
          + β × 预测码率
        优势：直接优化编码后质量
​
    先实现方案A，记录基线
    在Week13-15实现代理编码器后，升级为方案B
​
  评测：
    将训练好的前处理网络应用到真实HEVC编码流程
    计算BD-Rate增益（相对无前处理）
​
Day 7：前处理的局限性分析
  分析任务：
    实验1：前处理对不同内容的效果
      分别测试：平坦内容（风景）/ 高频内容（纹理）/ 人脸
      前处理对哪类内容效果最好？
​
    实验2：前处理 vs. 编码质量的关系
      低QP（高质量）时前处理效果如何？
      高QP（低质量）时前处理效果如何？
​
    实验3：主观质量影响
      前处理是否引入可见的视觉变化？
      用LPIPS评测前处理引入的感知变化量
​
    记录分析结论，形成对前处理局限性的认知
Week 12：后处理方向
Day 1-2：后处理系统综述
  学习内容：
    后处理的分类：
      1. Post-loop filter：解码后独立处理
      2. In-loop filter：集成在编码环路内
      区别：In-loop影响参考帧，影响后续帧的编码
​
    现有传统环路滤波的局限：
      HEVC去块滤波：基于规则，固定强度
      SAO：统计补偿，不针对内容
      神经网络后处理的优势：自适应、内容感知
​
  论文复习（在Week5已读，现在加深理解）：
    ARCNN / DnCNN / FFDNet的后处理应用视角
​
Day 3-4：视频后处理论文
  论文1（重要）：
    标题："STDF: Spatio-Temporal Deformable Fusion
           for Video Compression Artifact Reduction"
    作者：Deng et al.
    发表：AAAI 2020
    代码：https://github.com/ryanxingql/stdf-pytorch
    
    精读重点：
      可变形卷积（Deformable Convolution）原理：
        普通卷积：固定采样位置
        可变形卷积：学习偏移量，自适应采样位置
      时序信息的利用方式：
        用相邻帧作为辅助输入
        可变形卷积对齐相邻帧
      为什么多帧信息有帮助：
        单帧后处理：只有空间信息
        多帧后处理：时序冗余提供额外信息
​
  论文2：
    标题："CVEGAN: A Perceptually-inspired GAN
           for Compressed Video Enhancement"
    作者：Ma et al.
    发表：arXiv 2021
​
    精读重点：
      GAN在视频后处理中的训练稳定性
      时序一致性损失：防止帧间闪烁
      感知损失在视频后处理中的设计
​
  论文3：
    标题："BasicVSR++: Improving Video
           Super-Resolution with Enhanced
           Propagation and Alignment"
    作者：Chan et al.
    发表：CVPR 2022
    代码：https://github.com/ckkelvinchan/BasicVSR-PlusPlus
    
    学习重点：
      双向传播框架（前向+后向）
      二阶网格传播
      流引导可变形对齐
      这个框架可直接迁移到视频后处理
​
Day 5：QP条件后处理
  学习内容：
    为什么后处理需要QP条件：
      不同QP的失真类型和强度不同
      固定强度后处理在低QP下过处理
      QP条件网络自适应调整处理强度
​
  实现设计：
    QP条件注入方式：
      方案1：QP → MLP → 加到特征图
      方案2：QP → 预测Affine变换参数（推荐）
        γ = MLP_γ(QP)
        β = MLP_β(QP)
        output = γ × feature + β
      方案3：为每个QP训练独立的BN参数
​
  实践：
    修改STDF代码
    加入QP条件模块
    在QP=22/27/32/37上联合训练
    评测：QP条件网络 vs. 固定网络的性能差异
​
Day 6-7：后处理完整实践
  完整任务：训练视频后处理网络
​
  数据构建：
    下载Vimeo-90K数据集：
      http://toflow.csail.mit.edu/
    用ffmpeg以QP=27/32/37编码所有视频
    构建：{原始帧序列, QP, 压缩帧序列} 三元组
​
  训练配置：
    网络：STDF变体（加QP条件）
    输入：当前帧 + 前后各2帧 = 5帧
    输出：当前帧的后处理结果
    损失：MSE + 感知损失 + 时序一致性损失
​
  评测：
    在HEVC CTC序列上评测
    计算BD-Rate增益（相对无后处理）
    目标：BD-Rate节省 > 5%
​
  分析：
    后处理对不同帧类型(I/P/B帧)效果差异
    时序上下文帧数对效果的影响(1/3/5帧对比)
Month 4 详细计划（代理编码器专项，完整版）
Week 13：代理编码器基础理论
Day 1：不可微问题深入分析
  学习内容：
    传统编码器的四个不可微点：
      1. 量化：round(x)导数为0
      2. 模式决策：argmin操作不可微
      3. 熵编码：CABAC是符号操作
      4. 阈值判断：环路滤波中的条件分支
​
    三种解决策略的完整对比：
      策略A：可微近似（Differentiable Approximation）
        代表：STE / 软量化 / 加性噪声
        优点：训练时梯度流动自然
        缺点：训练/推理行为不一致（train-test gap）
​
      策略B：无梯度优化（Gradient-Free Optimization）
        代表：强化学习 / 进化算法 / 贝叶斯优化
        优点：不需要近似，直接优化真实目标
        缺点：样本效率低，训练慢
​
      策略C：代理模型（Surrogate Model）
        代表：用神经网络模拟编码器
        优点：精度可控，训练高效
        缺点：代理本身存在误差，优化可能过拟合代理
​
  记录：对三种策略各写一段理解笔记
​
Day 2：STE与软量化深入
  必读论文：
    论文1：
      标题："Estimating or Propagating Gradients
             Through Stochastic Neurons for
             Conditional Computation"
      作者：Bengio et al.
      发表：arXiv 2013
      重点：STE的理论依据和适用条件
​
    论文2：
      标题："Learned Step Size Quantization"
      作者：Esser et al.
      发表：ICLR 2020
      重点：可学习量化步长 + STE训练技巧
​
  实践（重要）：
实验：STE vs. 软量化 vs. 加性噪声对比
  设置：
    使用Ballé2017框架
    只修改量化近似方式
    其他超参数完全相同
    在Kodak上评测RD性能
​
  三种量化实现：
    实现A：STE
      前向：y_hat = torch.round(y)
      反向：梯度直通（用detach技巧实现）
​
    实现B：软量化
      y_hat = y - torch.sin(2π*y) / (2π)
      平滑近似round函数
​
    实现C：加性均匀噪声（CompressAI默认）
      训练：y_hat = y + torch.zeros_like(y).uniform_(-0.5, 0.5)
      推理：y_hat = torch.round(y)
​
  评测维度：
    训练收敛速度（loss曲线）
    最终RD性能（bpp vs PSNR）
    训练/推理行为差异（train-test gap大小）
​
  记录结论：
    哪种方法收敛最快？
    哪种方法最终性能最好？
    gap最小的是哪种？
    形成对量化近似的直观认知
Day 3-4：代理编码器论文精读
Day 3：可微JPEG代理
  论文1：
    标题："JPEG-resistant Adversarial Images"
    作者：Shin & Song
    发表：NIPS Workshop 2017
    精读重点：
      JPEG各步骤的可微化方式
      DCT：线性变换，直接实现
      量化：用soft round近似
      为什么需要可微JPEG（对抗样本攻击场景）
      思考：这个场景与前处理训练的相似性
​
  论文2（重要）：
    标题："Towards Robust Neural Networks
           via Random Self-ensemble"
    作者：Liu & Cheng
    发表：ECCV 2018
    精读重点：
      可微图像变换的通用框架
      如何将任意图像处理步骤近似为可微操作
​
  代码研究：
    DiffJPEG实现：
      地址：https://github.com/mlomnitz/DiffJPEG
      研究步骤：
        Step1：阅读README，理解整体设计
        Step2：逐文件阅读：
          compress_jpeg.py：压缩流程
          decompress_jpeg.py：解压流程
          utils.py：DCT矩阵等工具
          differentiable_round.py：可微量化
        Step3：跑通示例代码
        Step4：验证可微性：
          对输入图像求梯度
          确认梯度能从输出传回输入
        Step5：实验：
          用DiffJPEG训练一个对JPEG鲁棒的分类器
          对比：有/无DiffJPEG训练的模型
          在真实JPEG压缩图像上测试分类精度
​
Day 4：HEVC代理编码器论文
  论文3：
    标题："Fast HEVC Intra Coding Based on
           Convolutional Neural Network"
    作者：Xu et al.
    发表：VCIP 2017
    精读重点：
      CNN预测CU划分深度的框架
      训练数据收集：运行HM获取最优决策标签
      多类分类问题的损失函数设计
      如何将预测结果集成到HM中
​
  论文4（重要）：
    标题："Reducing Complexity of HEVC: A Deep
           Learning Approach"
    作者：Liu et al.
    发表：IEEE TCSVT 2018
    精读重点：
      多任务学习框架：
        同时预测CU深度、预测模式、跳过标志
      如何构建大规模训练数据集
      代理精度与编码加速的权衡分析
      这篇论文的实验分析非常详细，值得反复阅读
​
  论文5：
    标题："Partition Map Prediction for Fast
           Block Partitioning in VVC"
    作者：Song et al.
    发表：IEEE TCSVT 2022
    精读重点：
      VVC更复杂的QTMT划分结构
      如何用分割图（Partition Map）表示划分结果
      分层预测策略：从粗到细
      集成到VTM的工程细节
Day 5-6：代理编码器论文（编码失真模拟）
Day 5：黑盒代理编码器
  论文6（核心）：
    标题："Preprocessing Enhanced Image Compression
           for Machine Vision"
    作者：Yan et al.
    发表：arXiv 2021
    精读重点：
      代理HEVC编码器的网络结构设计
      训练数据构造方式
      如何验证代理精度
      代理编码器用于前处理训练的完整流程
​
  论文7：
    标题："Sandwiched Image Compression: Extending
           the Usage of Standard Codecs with
           Neural Network Wrappers"
    作者：Feng et al.
    发表：ICME 2023
    精读重点：
      三明治结构的完整训练方案
      代理编码器精度对前后处理效果的影响分析
      工业界实用方案的设计考量
      与纯端到端方案的性能对比
​
Day 6：面向机器视觉的代理编码器
  论文8：
    标题："Task-Driven Image Compression"
    作者：Chamain et al.
    发表：相关顶会
    精读重点：
      任务网络（检测/分割）作为代理
      如何用任务损失优化编码器
      率失真-精度联合优化
​
  论文9：
    标题："Video Coding for Machines: A Paradigm
           of Collaborative Compression and
           Intelligent Analytics"
    作者：Duan et al.
    发表：IEEE TIP 2020
    精读重点：
      VCM整体框架
      人眼质量与机器精度的双重需求
      代理任务模型在VCM中的角色
      MPEG-VCM标准化的测试条件
​
  论文10：
    标题："Image Coding for Machines with
           Omnipotent Feature"
    作者：Liu et al.
    发表：arXiv 2021
    精读重点：
      面向多种下游任务的通用特征压缩
      任务无关特征的学习方式
      代理任务集合的构建
Day 7：Week 13综合整理
整理任务：
  Task1：绘制代理编码器分类图
    按照功能分类：
      失真模拟类（黑盒/白盒）
      码率预测类
      决策加速类
      任务导向类
    对每类总结：核心思路/代表论文/适用场景
​
  Task2：整理关键技术点笔记
    量化近似方法对比表
    代理精度评测指标体系
    训练技巧总结
​
  Task3：规划Week14-15的实现任务
    确定要实现的代理编码器类型
    准备数据集和实验环境
Week 14：联合优化框架
Day 1-2：前处理联合优化理论
必读论文：
  论文1：
    标题："Perception-Distortion Tradeoff"
    作者：Blau & Michaeli
    发表：CVPR 2018
    精读重点（深度版）：
      感知质量的数学定义：
        p(x̂) ≈ p(x)（分布匹配）
      不可能三角证明：
        当失真D固定时，
        提升感知质量会增加方差
      对前处理设计的启示：
        前处理不应该追求完美重建
        应该针对编码场景优化目标
​
  论文2：
    标题："Optimally Compressed Representations
           with Information-Theoretic Lower Bounds"
    （搜索率失真感知理论相关工作）
​
  更直接论文：
    标题："Variable Rate ROI Image Compression
           Optimized for Visual Quality"
    学习重点：
      感兴趣区域（ROI）的差异化码率分配
      视觉注意力模型的应用
      这是前处理思想的一种具体形式
​
学习内容：联合优化的数学框架
  问题定义：
    给定：前处理网络 f_θ，代理编码器 C_φ（固定）
    优化：min_θ E[D(x, C_φ(f_θ(x))) + λ·R(C_φ(f_θ(x)))]
    
  训练策略：
    阶段1：训练代理编码器 C_φ
      固定：无
      优化：min_φ ||C_φ(x) - E(x)||² + α·|R_φ(x) - R_E(x)|
      其中E(x)是真实编码器输出，R_E是真实码率
​
    阶段2：固定代理，训练前处理
      固定：C_φ（代理编码器参数）
      优化：min_θ E[MSE(x, C_φ(f_θ(x))) + λ·R_φ(f_θ(x))]
​
    阶段3（可选）：端到端微调
      联合微调 θ 和 φ（小学习率）
      防止前处理过度适配代理误差
Day 3-4：后处理联合优化
必读论文：
  论文1：
    标题："Deep Neural Network-Based Luma and Chroma
           QP Offset Decision for Versatile Video Coding"
    精读重点：
      QP偏移决策的神经网络预测
      如何将后处理集成到编码环路
      In-loop vs. Post-loop的性能差异
​
  论文2（重要）：
    标题："Jointly Optimizing Preprocessing and
           Postprocessing for Compressed Image
           Enhancement"
    （搜索前后处理联合优化相关工作）
​
  论文3：
    标题："FBCNN: Flexible Blind Convolutional
           Neural Network for Image Compression
           Artifact Suppression"
    作者：Jiang et al.
    发表：ICCV 2021
    代码：https://github.com/jiaxi-jiang/FBCNN
    精读重点：
      盲质量因子预测（不需要知道QP）
      灵活的处理强度控制
      解耦图像质量和内容的设计思路
​
联合优化实验设计：
  前处理+后处理联合系统：
    系统结构：
      原始图像 x
        ↓ 前处理网络 f_θ
      处理图像 x'
        ↓ 代理编码器 C_φ（固定）
      重建图像 x̂'
        ↓ 后处理网络 g_ψ
      最终输出 x̂
​
    损失函数：
      L = MSE(x, g_ψ(C_φ(f_θ(x))))
        + λ · R_φ(f_θ(x))
        + μ · ||f_θ(x) - x||²  （约束前处理幅度）
​
    训练策略：
      Step1：单独训练代理编码器
      Step2：固定代理，训练前处理
      Step3：固定代理，训练后处理
      Step4：固定代理，联合微调前后处理
​
    评测：
      将训练好的系统应用到真实HM编码
      计算端到端BD-Rate增益
Day 5：强化学习视角
学习内容：
  为什么考虑强化学习：
    RL不需要梯度，可以直接优化离散决策
    适用于编码模式选择等离散优化问题
​
必读论文：
  论文1：
    标题："Reinforcement Learning for Variable
           Bit Rate Image Compression"
    （搜索相关工作）
​
  论文2：
    标题："Adaptive Bitrate Video Streaming
           with Neural Networks"
    学习重点：
      状态空间设计（网络状态、缓冲区状态）
      动作空间设计（码率选择）
      奖励函数设计（QoE指标）
      这个框架可以迁移到编码决策
​
  论文3：
    标题："Optimizing Video Encoding with
           Temporal Information Using
           Reinforcement Learning"
    精读重点：
      状态：当前帧内容特征
      动作：QP选择 / 编码模式
      奖励：PSNR增益 - λ × 码率增量
      如何处理奖励的延迟性（未来帧的影响）
​
RL vs. 代理模型的对比认知：
  代理模型优势：
    训练样本效率高
    可以利用梯度信息
    容易控制优化目标
​
  RL优势：
    不需要可微近似
    可以优化长期目标
    自然处理离散决策
​
  工程选择：
    大多数工业场景：代理模型
    复杂序列级决策：RL
    两者结合也是常见方案
Day 6-7：前沿论文与系统设计
Day 6：前沿论文精读
  论文1（重要）：
    标题："Neural Image Compression: Generalization,
           Robustness, and Spectral Biases"
    作者：Cao et al.
    发表：NeurIPS 2022
    精读重点：
      量化近似对泛化性的影响
      不同图像分布下的性能差异
      频谱偏差对压缩的影响
      对代理编码器精度要求的理论分析
​
  论文2：
    标题："ELIC: Efficient Learned Image Compression
           with Unevenly Grouped Space-Channel
           Contextual Adaptive Coding"
    作者：He et al.
    发表：CVPR 2022
    代码：相关开源实现可在GitHub搜索
    精读重点：
      不均匀分组的通道上下文
      空间-通道联合上下文
      接近VTM的性能，推理速度合理
      这是目前图像端到端编码的重要里程碑
​
  论文3：
    标题："EVC: Towards Real-Time Neural Image
           Compression with Mask Decay"
    作者：相关工作
    发表：ICLR 2023
    精读重点：
      实时推理的工程优化
      掩码衰减训练策略
      端到端编码工业化的关键考量
​
Day 7：系统设计规划
  任务：设计完整的代理编码器系统方案
​
  系统方案文档：
    1. 系统目标定义
       目标：BD-Rate节省 > 5%（相对HEVC HM）
       约束：前处理延迟 < 10ms（1080p）
             后处理延迟 < 5ms（1080p）
             总系统延迟 < 50ms
​
    2. 代理编码器规格
       输入：图像块（128×128）+ QP（22/27/32/37）
       输出：重建图像块 + 预测码率
       精度要求：
         PSNR误差 < 0.3dB
         码率预测误差 < 10%
​
    3. 前处理网络规格
       架构：轻量级U-Net（参数量 < 1M）
       条件：QP嵌入
       推理速度：GPU上 < 5ms（1080p）
​
    4. 后处理网络规格
       架构：带时序输入的残差网络
       输入：当前帧 + 前后各2帧
       推理速度：GPU上 < 3ms（1080p）
​
    5. 训练流程规划
       Stage1（1周）：训练代理编码器
       Stage2（3天）：训练前处理网络
       Stage3（3天）：训练后处理网络
       Stage4（2天）：联合微调
       Stage5（1天）：完整评测
Week 15：工程实现
Day 1-2：可微JPEG完整实现
实现目标：
  从零写一个完整的可微JPEG编码器
  支持任意quality factor输入
  梯度可以完整反向传播
​
实现规范：
  文件结构：
    diff_jpeg/
    ├── __init__.py
    ├── color_transform.py    # RGB↔YCbCr
    ├── chroma_subsampling.py # 4:2:0下采样
    ├── dct.py               # 可微DCT
    ├── quantization.py      # 软量化
    ├── jpeg_pipeline.py     # 完整流水线
    └── test_gradient.py     # 梯度验证
​
  各模块实现要点：
    color_transform.py：
      实现BT.601标准的转换矩阵
      转换为矩阵乘法（天然可微）
      注意数值范围的处理
​
    chroma_subsampling.py：
      下采样：用双线性插值（可微）
      上采样：双线性插值（可微）
​
    dct.py：
      预计算8×8 DCT基矩阵（64个基函数）
      用F.conv2d实现分块DCT
      注意padding和stride的设置
​
    quantization.py：
      实现三种量化方式：
        soft_round(x) = x - sin(2πx)/(2π)
        ste_round(x)：前向round，反向恒等
        noise_round(x)：加均匀噪声
      quality_to_qtable(quality)：
        将quality factor转换为量化表
​
    jpeg_pipeline.py：
      串联所有模块
      forward(image, quality) → 重建图像
      支持batch处理
​
梯度验证：
  使用torch.autograd.gradcheck
  验证每个模块的梯度正确性
  验证完整Pipeline的梯度流动
​
应用实验：
  实验1：对抗JPEG攻击
    目标：训练鲁棒分类器
    方法：在分类器前接DiffJPEG
    评测：在JPEG压缩图像上的分类精度
​
  实验2：前处理优化
    目标：优化图像使JPEG压缩后PSNR更高
    方法：固定DiffJPEG，对输入图像求梯度
    评测：优化后图像的JPEG压缩质量
Day 3-4：HEVC黑盒代理编码器实现
实现目标：
  训练一个神经网络模拟HEVC编码失真
  支持QP=22/27/32/37条件输入
  在Kodak测试集上代理精度达标
​
数据收集Pipeline：
  Step1：下载训练图像
    使用DIV2K训练集（800张）
    使用CLIC训练集（补充多样性）
    总计约2000张高质量图像
​
  Step2：用HM编码收集数据
    写Python脚本批量调用HM：
      for image in dataset:
        for qp in [22, 27, 32, 37]:
          encoded = hm_encode(image, qp)
          reconstructed = hm_decode(encoded)
          bpp = file_size / (H * W)
          save(image, qp, reconstructed, bpp)
​
    注意：
      图像裁剪为64的倍数（HM要求）
      保存时用PNG避免二次压缩
​
  Step3：构建数据集类
    SurrogateDataset：
      输入：{原始patch(128×128), QP}
      输出：{HM重建patch, 实际bpp}
      随机裁剪：在线裁剪，增加数据多样性
​
代理网络设计：
  class SurrogateHEVC(nn.Module):
    架构规格：
      QP嵌入层：
        输入：QP标量
        输出：64维向量
        实现：nn.Embedding(4, 64)
              或 nn.Linear(1, 64) + ReLU
​
      主干网络：
        8层残差块
        通道数：64
        条件注入：每个残差块注入QP嵌入
          注入方式：Affine变换
            γ = Linear(64, 64)(qp_embed)
            β = Linear(64, 64)(qp_embed)
            feature = γ * feature + β
​
      码率预测头：
        全局平均池化
        2层MLP
        输出：预测bpp
​
      重建输出头：
        1×1卷积
        输出：重建图像（与输入同分辨率）
​
  损失函数：
    L_recon = L1(predicted_recon, hm_recon)
    L_rate = MSE(predicted_bpp, actual_bpp)
    L_total = L_recon + 0.1 * L_rate
​
训练配置：
  优化器：Adam，lr=1e-4
  Batch size：32
  训练轮数：100 epoch
  学习率衰减：CosineAnnealingLR
  数据增强：随机裁剪、随机翻转
​
代理精度评测（Kodak测试集）：
  指标1：重建质量
    计算PSNR(代理输出, HM输出)
    目标：> 40dB
​
  指标2：码率预测
    计算|预测bpp - 真实bpp| / 真实bpp
    目标：< 10%
​
  指标3：RD曲线形状
    画代理模型 vs. HM的RD曲线
    视觉检查两条曲线的形状和间距
​
  指标4：梯度质量验证
    对输入图像求梯度
    可视化梯度图：应该突出图像的边缘和纹理
    梯度消失检验：确认梯度幅度合理
Day 5-6：前处理网络完整训练
基于代理编码器训练前处理网络
​
前处理网络架构：
  class PreprocessNet(nn.Module):
    设计原则：
      1. 轻量化（推理速度要求）
      2. 输出与输入分辨率相同
      3. 输出幅度受限（不能改变太多）
​
    架构：
      编码器：3层卷积，步长2，通道[32, 64, 128]
      瓶颈层：4个残差块，128通道
      解码器：3层转置卷积，通道[64, 32, 3]
      跳跃连接：编解码器之间
      QP条件：在瓶颈层注入
      输出约束：tanh激活 × 0.1（限制输出幅度）
                最终输出 = 原始图像 + 残差（幅度受限）
​
训练流程：
  Phase1：热身训练（10 epoch）
    仅用重建损失
    L = MSE(代理(前处理(x)), x)
    目的：让网络学会基本的压缩感知处理
​
  Phase2：加入码率约束（40 epoch）
    L = MSE(代理(前处理(x)), x) + λ × 预测码率
    λ从小到大逐渐增加（课程学习）
​
  Phase3：加入感知损失（50 epoch）
    L = MSE + λ × 码率 + μ × 感知损失
    感知损失：VGG特征图MSE
​
  Phase4：对抗训练（可选，20 epoch）
    加入判别器，提升主观质量
​
评测（关键步骤）：
  在真实HM上评测（不是代理）：
    Step1：将前处理网络输出保存为PNG
    Step2：用HM编码处理后的图像
    Step3：计算编码结果的PSNR和码率
    Step4：与无前处理的基线对比
    Step5：计算BD-Rate增益
​
  预期结果：
    BD-Rate节省 3-8%（相对于无前处理HM）
    LPIPS略有提升（感知质量）
    前处理引入的视觉变化不可见
​
  对比实验：
    A：无前处理（基线）
    B：前处理（无代理，仅重建损失）
    C：前处理（有代理，完整损失）
    D：前处理（有代理，加感知损失）
    分析各方案的BD-Rate和主观质量
Day 7：代理精度与优化效果关系分析
关键实验：代理精度对前处理效果的影响
​
实验设计：
  训练5个代理模型（代理精度递增）：
    代理1：训练100张图像（精度最低）
    代理2：训练300张图像
    代理3：训练600张图像
    代理4：训练1000张图像
    代理5：训练2000张图像（精度最高）
​
  用每个代理训练对应的前处理网络
​
  在真实HM上评测各前处理网络：
    记录代理PSNR误差 vs. BD-Rate增益
​
  预期观察：
    代理精度越高，前处理效果越好
    存在精度阈值：超过阈值后效果不再提升
    这个阈值是工程设计的重要参考
​
  分析报告内容：
    代理精度-BD-Rate增益曲线
    代理误差对哪种内容影响最大
    是否存在"代理过拟合"现象
    改进建议：代理需要多精确才够用
Week 16：前沿与综合实战
Day 1-2：VCM与SAM相关前沿
论文1（VCM核心）：
  标题："An Overview of Core Experiments for
         Video Coding for Machines"
  精读重点：
    VCM的标准化测试条件
    Anchor系统的定义
    评测指标：BD-Rate for Machines
    代理任务：检测（COCO AP）/分割/分类
​
论文2：
  标题："Towards Coding for Human and Machine Vision:
         A Scalable Image Coding Approach"
  作者：Choi et al.
  发表：ICME 2021
  精读重点：
    可扩展编码：基础层用于机器，增强层用于人眼
    两个任务的联合率失真优化
​
论文3（最新方向）：
  标题："Segment Anything"
  作者：Kirillov et al.
  发表：ICCV 2023
  代码：https://github.com/facebookresearch/segment-anything
  与编码的关联：
    SAM作为代理任务模型
    图像编码是否保留SAM所需特征
    这是当前VCM方向的热点问题
​
论文4：
  标题："Scalable Human-Machine Point Cloud
         Compression"
  （了解点云编码中的类似问题，拓宽视野）
​
实践：
  运行SAM模型
  实验：
    原始图像 → SAM分割
    HEVC压缩图像（不同QP）→ SAM分割
    对比分割结果的变化
    分析：哪个QP开始导致分割质量明显下降
    这个实验建立对"编码对机器视觉影响"的直观认知
Day 3-4：扩散模型与编码的结合（前沿方向）
学习内容：
  扩散模型用于图像压缩的思路
​
论文1：
  标题："High-Fidelity Generative Image Compression"
  作者：Mentzer et al.
  发表：NeurIPS 2020
  精读重点：
    生成模型在超低码率编码中的应用
    如何在极低码率下保持感知质量
    与传统率失真优化的关系
论文2（重要前沿）：
  标题："Lossy Compression with Gaussian Diffusion"
  作者：Theis & Salimans
  发表：arXiv 2022
  精读重点：
    扩散模型作为解码器的理论框架
    如何用扩散过程建模解码不确定性
    与VAE编码框架的结合方式
    率失真感知三元组在扩散模型下的重新理解
    数学推导：
      编码：x → 潜变量z（确定性压缩）
      解码：z → 扩散采样 → x̂（随机生成）
      码率由z的熵决定
      失真由采样质量决定
​
论文3：
  标题："Towards Extreme Image Compression with
         Latent Feature Guidance and Diffusion
         Prior"
  作者：Pi et al.
  发表：arXiv 2023
  精读重点：
    扩散先验用于超低码率图像重建
    潜在特征引导扩散采样方向
    如何控制生成内容的保真度
    实验：0.1bpp以下的图像重建质量
​
论文4（系统性工作）：
  标题："CDC: Contextual Diffusion Compression"
  （搜索diffusion-based image compression相关最新工作）
  精读重点：
    上下文信息如何引导扩散解码
    与确定性解码方案的性能对比
    主观质量 vs. 客观指标的差异
​
论文5：
  标题："Diffusion-based Image Compression with
         Soft Context Filtering"
  （搜索相关arXiv 2023工作）
  精读重点：
    软上下文滤波机制
    扩散步数与重建质量的权衡
    实时性问题的解决思路
​
论文6（稳定扩散与压缩）：
  标题："Stable Diffusion is a Natural Generative
         Prior for Image Compression"
  （搜索stable diffusion compression相关工作）
  精读重点：
    预训练扩散模型作为强先验
    如何利用Stable Diffusion的大规模知识
    超低码率场景下的实际应用
扩散模型基础补充学习
如果对扩散模型基础不熟悉，需要先补充：
​
必读论文：
  论文A：
    标题："Denoising Diffusion Probabilistic Models"
    作者：Ho et al.
    发表：NeurIPS 2020（DDPM）
    精读重点：
      前向过程：逐步加噪
        q(x_t|x_{t-1}) = N(√(1-β_t)x_{t-1}, β_t·I)
      反向过程：逐步去噪
        p_θ(x_{t-1}|x_t) = N(μ_θ(x_t,t), Σ_θ(x_t,t))
      训练目标：预测噪声ε
      与VAE的关系：
        VAE：一步编解码
        DDPM：T步逐步编解码
​
  论文B：
    标题："Denoising Diffusion Implicit Models"
    作者：Song et al.
    发表：ICLR 2021（DDIM）
    精读重点：
      非马尔可夫链加速采样
      确定性采样 vs. 随机采样
      步数从1000→50的加速原理
​
  代码：
    DDPM官方实现：
      https://github.com/hojonathanho/diffusion
    HuggingFace diffusers库：
      https://github.com/huggingface/diffusers
    学习任务：
      跑通DDPM在CIFAR-10上的训练
      理解采样过程的代码实现
扩散模型与编码结合的实践
实验1：扩散模型作为后处理
  思路：
    传统后处理：确定性映射（低质量→高质量）
    扩散后处理：随机生成（从噪声中恢复细节）
  
  实现步骤：
    Step1：训练条件扩散模型
      条件：HEVC压缩图像（低质量）
      目标：生成高质量重建
      条件注入方式：
        将压缩图像特征作为引导信号
        用classifier-free guidance
​
    Step2：与确定性后处理对比
      指标：PSNR / SSIM（客观）
      指标：LPIPS / FID（感知）
      预期：扩散方法PSNR可能不如确定性方法
            但LPIPS和视觉质量更好
​
    Step3：分析采样步数与质量的权衡
      步数：10 / 20 / 50 / 100 / 1000
      记录每种步数的：推理时间 / PSNR / LPIPS
      画出步数-质量-速度三者关系图
​
实验2：扩散先验用于超低码率
  场景：bpp < 0.1（极低码率）
  思路：
    传统方法在超低码率下严重模糊
    扩散模型可以"幻想"合理的细节
  
  实现步骤：
    Step1：训练极低码率编码器
      修改CompressAI模型
      设置λ=0.0001（极低，偏向码率）
      训练后在Kodak上评测
​
    Step2：加入扩散解码器
      固定编码器（已训练）
      训练扩散解码器
      以编码器输出的潜变量为条件
​
    Step3：主观质量评测
      请3-5人进行主观打分
      对比：传统解码器 vs. 扩散解码器
      分析：哪些内容类型受益最多？
​
关键认知：
  扩散模型在编码领域的地位：
    优势：超低码率下的感知质量
    劣势：采样速度慢（工业应用的主要障碍）
    趋势：加速采样技术（Consistency Models等）
          将推动扩散模型在编码中的实用化
Day 5：NeRF与3D内容编码（前沿拓展）
学习目标：
  了解3D内容压缩的新兴方向
  理解代理编码器思想在3D领域的延伸
​
必读论文：
  论文1：
    标题："NeRF: Representing Scenes as Neural
           Radiance Fields for View Synthesis"
    作者：Mildenhall et al.
    发表：ECCV 2020
    代码：https://github.com/bmild/nerf
    精读重点：
      隐式神经表示的概念
      体积渲染原理
      与传统3D表示的对比
      NeRF本身就是一种压缩表示
        网络参数 = 压缩后的场景表示
​
  论文2：
    标题："3D Gaussian Splatting for Real-Time
           Radiance Field Rendering"
    作者：Kerbl et al.
    发表：SIGGRAPH 2023
    代码：https://github.com/graphdeco-inria/gaussian-splatting
    精读重点：
      高斯点云的场景表示
      实时渲染的实现原理
      与NeRF的主要区别
​
  论文3（与压缩直接相关）：
    标题："Compact 3D Gaussian Representation for
           Radiance Field"
    作者：Lee et al.
    发表：CVPR 2024
    精读重点：
      3DGS的压缩问题
      神经网络压缩高斯属性
      代理编码器思想在3D场景中的应用
​
  论文4：
    标题："NeRF-Compression: Compressing Neural
           Radiance Fields"
    （搜索NeRF compression相关最新工作）
    精读重点：
      NeRF参数的熵编码
      重要性加权的网络剪枝
      与传统3D内容压缩的对比
​
实践：
  Task1：跑通NeRF
    使用tiny-nerf示例：
      https://github.com/krrish94/nerf-pytorch
    在lego数据集上训练
    理解隐式表示的压缩本质
​
  Task2：3DGS压缩实验
    跑通3DGS官方代码
    统计不同场景的高斯数量和存储大小
    思考：如何用学习型方法压缩高斯参数
Day 6：综合实战项目实施
项目：完整的编码增强系统
项目名称：
  Surrogate-Guided Video Compression Enhancement System
  基于代理编码器引导的视频编码增强系统
​
系统目标：
  在标准HEVC编码基础上
  通过前处理+后处理提升压缩效率
  目标BD-Rate节省 > 8%（相对HEVC HM）
  前后处理总延迟 < 20ms（1080p，GPU）
​
完整系统架构：
  ┌─────────────────────────────────────────────┐
  │              输入视频帧序列                   │
  └──────────────────┬──────────────────────────┘
                     ↓
  ┌─────────────────────────────────────────────┐
  │         前处理网络（PreprocessNet）            │
  │  输入：原始帧 + QP                           │
  │  输出：编码友好的处理帧                        │
  │  延迟要求：< 8ms/帧                          │
  └──────────────────┬──────────────────────────┘
                     ↓
  ┌─────────────────────────────────────────────┐
  │         标准HEVC编码器（HM/FFmpeg）           │
  │  固定编码器，不修改                           │
  │  QP：22/27/32/37（四个质量点）               │
  └──────────────────┬──────────────────────────┘
                     ↓
  ┌─────────────────────────────────────────────┐
  │         后处理网络（PostprocessNet）           │
  │  输入：解码帧 + 相邻帧 + QP                   │
  │  输出：增强后的帧                             │
  │  延迟要求：< 12ms/帧                         │
  └──────────────────┬──────────────────────────┘
                     ↓
  ┌─────────────────────────────────────────────┐
  │              输出增强视频                     │
  └─────────────────────────────────────────────┘
​
训练流程详细规划：
​
  Stage1：代理编码器训练（3天）
    数据：DIV2K + CLIC（约2000张图像）
    每张图像用HM以QP=22/27/32/37编码
    网络：SurrogateHEVC（已在Week15设计）
    训练：100 epoch
    验收：Kodak上PSNR误差 < 0.3dB，码率误差 < 10%
​
  Stage2：前处理网络训练（2天）
    数据：Vimeo-90K训练集
    代理编码器：固定（Stage1训练好）
    网络：PreprocessNet（轻量U-Net）
    损失：率失真损失（通过代理反传）
    训练：50 epoch
    验收：代理模型上BD-Rate节省 > 5%
​
  Stage3：后处理网络训练（2天）
    数据：Vimeo-90K训练集
      用真实HM编码获取{原始帧，压缩帧}对
      （不用代理，直接用真实HM收集数据）
    网络：PostprocessNet（时序残差网络）
    损失：MSE + 感知损失 + 时序一致性
    训练：50 epoch
    验收：单独后处理BD-Rate节省 > 5%
​
  Stage4：联合微调（1天）
    固定代理编码器
    联合训练前处理+后处理
    小学习率：原始学习率的0.1倍
    训练：10 epoch
    目的：前后处理协同工作
​
  Stage5：完整评测（1天）
    测试集：HEVC CTC标准序列
      Class A：2560×1600（4K内容）
      Class B：1920×1080（1080p）
      Class C：832×480（标清）
      Class D：416×240（低分辨率）
    评测指标：
      BD-Rate（PSNR-Y）
      BD-Rate（MS-SSIM）
      BD-Rate（VMAF）
      前处理延迟（ms/帧）
      后处理延迟（ms/帧）
项目评测与分析报告
报告结构：
​
1. 实验环境
   GPU：型号/显存
   CPU：型号/核数
   框架：PyTorch版本
   编码器：HM版本/FFmpeg版本
​
2. 各模块独立性能
   前处理单独贡献：BD-Rate节省 X%
   后处理单独贡献：BD-Rate节省 Y%
   前后处理联合：BD-Rate节省 Z%
   理想情况：Z > X + Y（协同效应）
​
3. 不同分辨率分析
   4K / 1080p / 480p下的BD-Rate对比
   分析：分辨率对效果的影响规律
​
4. 不同内容类型分析
   运动快/运动慢序列
   纹理丰富/平坦内容
   人脸/自然风景
​
5. 主观质量评测
   选取代表性片段
   用VMAF评分
   用LPIPS评分
   人工主观对比（如有条件）
​
6. 速度分析
   前处理：X ms/帧（GPU）
   后处理：Y ms/帧（GPU）
   总开销对比：原始编解码时间
​
7. 消融实验（Ablation Study）
   去掉代理编码器的影响
   去掉QP条件的影响
   去掉感知损失的影响
   去掉时序信息的影响
​
8. 与相关工作对比
   对比STDF（纯后处理）
   对比BasicVSR++（迁移用于后处理）
   对比端到端编码（DVC/DCVC）
Day 7：学习总结与进阶规划
四个月学习成果总结
技能矩阵自评（1-5分）：
​
基础技能：
  □ 图像/视频基础知识          目标：5分
  □ PyTorch熟练度              目标：5分
  □ 传统编码标准理解            目标：4分
  □ 数据集管理和评测框架        目标：5分
​
核心技术：
  □ 端到端图像编码              目标：4分
  □ 端到端视频编码              目标：3分
  □ 图像前处理设计              目标：4分
  □ 视频后处理设计              目标：4分
  □ 代理编码器实现              目标：4分
  □ 联合优化系统                目标：3分
​
前沿方向了解：
  □ 扩散模型与编码              目标：3分
  □ VCM/面向机器编码            目标：3分
  □ NeRF/3DGS压缩              目标：2分
四个月后的进阶路线
根据兴趣和方向选择深入路线：
​
路线A：学术研究方向
  重点：发表高质量论文
  
  Month 5-6：
    深入一个细分方向（选其一）：
      A1：极低码率感知编码（扩散模型方向）
      A2：面向机器视觉的编码（VCM方向）
      A3：3D内容的神经压缩（NeRF/3DGS）
​
    具体行动：
      精读目标方向近3年所有顶会论文（约30篇）
      复现1-2篇代表性工作
      找到可以改进的点
      设计实验验证idea
      撰写论文投稿CVPR/ICCV/NeurIPS
​
    推荐跟踪会议：
      DCC（Data Compression Conference）
        最专业的数据压缩会议
        每年3月，论文质量高
      VCIP（Visual Communications and Image Processing）
      PCS（Picture Coding Symposium）
      以上三个是编码专业会议
      CVPR/ICCV/ECCV中的compression track
​
路线B：工业工程方向
  重点：解决实际问题，追求工程落地
​
  Month 5-6：
    工程化专项：
      B1：模型轻量化与加速
        知识蒸馏：将大模型压缩为小模型
        模型量化：INT8/INT4推理
        TensorRT部署：GPU推理加速
        目标：前后处理网络在GPU上实现实时（>30fps，1080p）
​
      B2：多平台部署
        学习ONNX导出
        学习TensorRT优化
        学习移动端部署（CoreML/TFLite）
        实现：将后处理网络部署到手机
​
      B3：码率控制工程
        学习真实编码器的码率控制接口
        设计神经网络辅助的内容自适应码率控制
        与现有码率控制算法（ABR/VBR/CQP）集成
​
    必学工具：
      TensorRT：https://developer.nvidia.com/tensorrt
      ONNX Runtime：https://onnxruntime.ai/
      相关论文（模型压缩）：
        标题："Once-for-All: Train One Network and
               Specialize it for Efficient Deployment"
        作者：Cai et al.
        发表：ICLR 2020
        学习轻量化网络设计方法
​
路线C：全栈方向（兼顾研究和工程）
  Month 5：深化一个研究方向（参考路线A）
  Month 6：将研究成果工程化（参考路线B）
  目标：既能发论文，又能做产品
持续学习的资源与习惯
每日必做（30分钟）：
  1. 检查arXiv cs.CV + eess.IV最新论文
     地址：https://arxiv.org/list/eess.IV/recent
     筛选关键词：
       compression / codec / encoding
       rate-distortion / image quality
       neural video / learned compression
​
  2. 追踪以下研究组的最新工作：
     Johannes Ballé（Google）：端到端编码奠基人
     Eirikur Agustsson（Google）：感知压缩
     Luc Van Gool（ETH Zurich）：通用视觉压缩
     Yan Lu团队（Microsoft）：DCVC系列
     Dong Liu团队（USTC）：视频编码加速
     Xin Lu团队（字节跳动）：工业界编码增强
​
每周必做：
  1. 精读1-2篇新论文（完整推导+代码）
  2. 更新实验记录和笔记
  3. 复现或验证论文中的一个关键实验
​
每月必做：
  1. 回顾当月学习，更新技能矩阵
  2. 整理代码：将实验代码整理为可复用模块
  3. 参与社区讨论：
     Papers With Code讨论区
     OpenReview评论区
     GitHub Issues/Discussions
​
重要竞赛（参与实战）：
  CLIC（Challenge on Learned Image Compression）
    每年CVPR Workshop举办
    地址：https://www.compression.cc/
    强烈推荐参加：直接接触领域顶尖方法
​
  HEVC/VVC编码优化竞赛：
    每年JVET会议相关竞赛
​
  AIM（Advances in Image Manipulation）Workshop：
    每年ECCV举办
    包含压缩增强赛道
四个月完整学习资源汇总
核心论文清单（按学习顺序）
基础部分（Month 1）：
  Week 1-2：无特定论文，以教材为主
  Week 3：
    [1] SRCNN - Dong et al., ECCV 2014
    [2] VDSR - Kim et al., CVPR 2016
    [3] EDSR - Lim et al., CVPRW 2017
  Week 4：
    [4] 感知损失 - Johnson et al., ECCV 2016
    [5] SRGAN - Ledig et al., CVPR 2017
    [6] ESRGAN - Wang et al., ECCVW 2018
​
Month 2：
  Week 5：
    [7] DnCNN - Zhang et al., TIP 2017
    [8] FFDNet - Zhang et al., TIP 2018
    [9] ARCNN - Dong et al., ICCV 2015
  Week 6：
    [10] HEVC Overview - Sullivan et al., TCSVT 2012
    [11] JPEG Standard - Wallace, CACM 1991
  Week 7：
    [12] SSIM - Wang et al., TIP 2004
  Week 8：
    [13] VAE - Kingma & Welling, ICLR 2014
    [14] Ballé2017 - ICLR 2017
    [15] Ballé2018 - ICLR 2018
    [16] Minnen2018 - NeurIPS 2018
​
Month 3：
  Week 9：
    [17] Cheng2020 - CVPR 2020
    [18] 可变码率 - Cui et al., CVPR 2021
    [19] QMap - Song et al., ICCV 2021
  Week 10：
    [20] SPyNet - Ranjan & Black, CVPR 2017
    [21] DVC - Lu et al., CVPR 2019
    [22] DCVC - Li et al., NeurIPS 2021
    [23] DCVC-DC - Li et al., CVPR 2023
  Week 11：
    [24] Perception-Distortion - Blau & Michaeli, CVPR 2018
    [25] Sandwiched - Feng et al., ICME 2023
    [26] Content-Adaptive - Campos et al., CVPRW 2019
  Week 12：
    [27] STDF - Deng et al., AAAI 2020
    [28] CVEGAN - Ma et al., arXiv 2021
    [29] BasicVSR++ - Chan et al., CVPR 2022
    [30] FBCNN - Jiang et al., ICCV 2021
​
Month 4：
  Week 13：
    [31] STE - Bengio et al., arXiv 2013
    [32] LSQ - Esser et al., ICLR 2020
    [33] DiffJPEG - Shin & Song, NIPS Workshop 2017
    [34] 快速HEVC - Xu et al., VCIP 2017
    [35] 复杂度降低 - Liu et al., TCSVT 2018
    [36] VVC划分预测 - Song et al., TCSVT 2022
  Week 14：
    [37] 联合优化前处理 - 相关工作
    [38] ELIC - He et al., CVPR 2022
    [39] Neural Image Compression泛化 - Cao et al., NeurIPS 2022
  Week 15：
    实践为主，复现前述论文
  Week 16：
    [40] HFGIC - Mentzer et al., NeurIPS 2020
    [41] Lossy Diffusion - Theis et al., arXiv 2022
    [42] DDPM - Ho et al., NeurIPS 2020
    [43] NeRF - Mildenhall et al., ECCV 2020
    [44] 3DGS - Kerbl et al., SIGGRAPH 2023
    [45] VCM Overview - Duan et al., TIP 2020
核心代码库汇总
图像质量基础：
  BasicSR：https://github.com/XPixelGroup/BasicSR
  FBCNN：https://github.com/jiaxi-jiang/FBCNN
​
端到端编码：
  CompressAI：https://github.com/InterDigitalCom/CompressAI
  DCVC系列：https://github.com/microsoft/DCVC
​
可微编码器：
  DiffJPEG：https://github.com/mlomnitz/DiffJPEG
​
视频后处理：
  STDF：https://github.com/ryanxingql/stdf-pytorch
  BasicVSR++：https://github.com/ckkelvinchan/BasicVSR-PlusPlus
​
传统编码工具：
  HM参考软件：https://hevc.hhi.fraunhofer.de/
  VTM参考软件：https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM
  BD-Rate计算：https://github.com/Anserw/Bjontegaard_metric
  YUView可视化：https://github.com/IENT/YUView
​
评测工具：
  VMAF：https://github.com/Netflix/vmaf
  LPIPS：https://github.com/richzhang/PerceptualSimilarity
​
前沿方向：
  SAM：https://github.com/facebookresearch/segment-anything
  NeRF PyTorch：https://github.com/krrish94/nerf-pytorch
  3DGS：https://github.com/graphdeco-inria/gaussian-splatting
  HuggingFace Diffusers：https://github.com/huggingface/diffusers
硬件与环境配置建议
最低配置（能跑通所有实验）：
  GPU：NVIDIA RTX 3080（10GB显存）
  CPU：8核以上
  内存：32GB
  存储：2TB SSD（数据集较大）
​
推荐配置（训练速度合理）：
  GPU：NVIDIA RTX 3090 / 4090（24GB显存）
  CPU：16核以上
  内存：64GB
  存储：4TB SSD
​
软件环境：
  操作系统：Ubuntu 20.04/22.04（推荐）
  CUDA：11.8 或 12.x
  Python：3.9 或 3.10
  PyTorch：2.0以上
  关键库：
    compressai >= 1.2
    basicsr >= 1.4
    ffmpeg-python >= 0.2
    opencv-python >= 4.7
    scikit-image >= 0.19（SSIM计算）
    pytorch-msssim（MS-SSIM）
    lpips >= 0.1.4
​
环境搭建顺序：
  Step1：安装CUDA和cuDNN
  Step2：创建conda虚拟环境
  Step3：安装PyTorch
  Step4：安装ffmpeg（系统级）
  Step5：安装HM/VTM（需要编译）
  Step6：安装Python依赖库
  Step7：验证所有工具可用
最终学习建议
核心原则（按重要性排序）：
​
原则1：动手优先
  每读一篇论文，必须配合代码
  不能复现的论文算没读懂
  建议：论文:代码 = 1:2（代码时间是论文的两倍）
​
原则2：记录驱动
  建立个人知识库（Notion/Obsidian）
  每篇论文：一页笔记（核心贡献+关键公式+实验结果）
  每次实验：记录配置+结果+分析
  四个月后这些笔记是最宝贵的财富
​
原则3：问题导向
  不要为了学而学
  每个技术点都问：这解决了什么问题？
  这样学到的知识才能活用
​
原则4：循序渐进
  不要跳跃学习
  基础不扎实，高级内容看不懂
  严格按照计划顺序执行
最终学习建议完整版
原则5-10（续）
原则5：对比思维
  学习每个新方法时，必须问：
    它比之前的方法好在哪里？
    它解决了什么之前没解决的问题？
    它的局限性是什么？
  建立对比表格：
    ┌──────────┬────────┬────────┬────────┬────────┐
    │ 方法     │ 核心贡献│ 优势   │ 局限性 │ 适用场景│
    ├──────────┼────────┼────────┼────────┼────────┤
    │ Ballé17  │ 端到端 │ 首次   │ 性能差 │ 研究基线│
    │ Ballé18  │ 超先验 │ 更准确 │ 慢     │ 研究基线│
    │ Minnen18 │ 上下文 │ 性能好 │ 串行慢 │ 高性能  │
    │ Cheng20  │ 注意力 │ 更高效 │ 复杂   │ 高性能  │
    │ ELIC     │ 通道组 │ 接近VTM│ 较慢   │ 接近实用│
    └──────────┴────────┴────────┴────────┴────────┘
  这种表格比单独笔记更有价值
​
原则6：工程意识
  学术代码 ≠ 工程代码
  任何方法落地前需要考虑：
    推理速度：能否满足实时要求？
    显存占用：能否在目标设备运行？
    稳定性：边界情况如何处理？
    可维护性：代码是否清晰？
  培养习惯：每个实验记录推理时间和显存占用
​
原则7：社区参与
  不要孤立学习
  参与方式：
    GitHub Issues：遇到问题先搜索Issues
    Papers With Code：查看论文复现情况
    知乎/Reddit：搜索相关技术讨论
    Twitter/X：关注领域研究者
    OpenReview：阅读顶会论文的审稿意见
      审稿意见往往指出论文的真正弱点
      这是快速理解论文局限性的捷径
​
原则8：输出倒逼输入
  学习的最好方式是输出
  输出形式（从易到难）：
    Level1：整理笔记（每篇论文）
    Level2：写技术博客（每个主题）
    Level3：录制讲解视频（某个核心技术）
    Level4：开源复现代码（某篇论文）
    Level5：参加竞赛（CLIC等）
    Level6：投稿论文（自己的改进）
  建议：从第一个月开始写技术博客
        四个月后博客就是最好的简历
​
原则9：深度优于广度
  编码方向的陷阱：论文太多，容易贪多
  正确策略：
    每个方向选2-3篇最重要的论文精读
    精读 = 推导公式 + 复现代码 + 理解局限
    宁可10篇精读，不要100篇泛读
  识别"最重要论文"的方法：
    被引次数高（Google Scholar）
    出现在综述论文的必读列表
    领域内研究者反复提及
​
原则10：接受不确定性
  四个月不可能精通所有内容
  诚实评估自己的掌握程度
  允许有不懂的地方，记录下来
  持续学习是常态，不是例外
每周执行模板
标准周学习流程
周一（论文日）：
  上午（3小时）：
    精读本周核心论文第一篇
    第一遍：通读，不追求全懂
    记录：3个核心贡献点，3个不懂的地方
​
  下午（3小时）：
    精读第一篇（第二遍）
    重点攻克不懂的公式
    手推核心公式，写在纸上
    对照代码理解实现细节
​
  晚上（1小时）：
    整理笔记
    更新论文对比表格
​
周二（代码日）：
  上午（3小时）：
    克隆本周相关代码库
    配置环境，跑通基础示例
    阅读核心模块代码
​
  下午（3小时）：
    逐行阅读关键实现
    对照论文理解设计决策
    记录不理解的代码段
​
  晚上（1小时）：
    整理代码阅读笔记
    记录关键API和使用方式
​
周三（实验日）：
  上午（3小时）：
    设计本周核心实验
    准备数据集和评测脚本
    运行基础实验（基线）
​
  下午（3小时）：
    运行对比实验
    记录实验结果
    分析异常结果
​
  晚上（1小时）：
    整理实验日志
    记录：配置/结果/分析/结论
​
周四（论文+实验日）：
  上午（3小时）：
    精读第二篇论文
    重点关注与第一篇的区别
​
  下午（3小时）：
    运行本周进阶实验
    尝试修改代码验证理解
​
  晚上（1小时）：
    对比两篇论文的异同
    更新对比表格
​
周五（整合日）：
  上午（2小时）：
    回顾本周所有笔记
    整理成结构化文档
​
  下午（2小时）：
    写本周技术博客
    内容：本周学到的最重要的一个技术点
    格式：背景→问题→解决方案→实验→结论
​
  下午（2小时）：
    整理本周代码
    写清晰的注释
    推送到个人GitHub
​
  晚上（1小时）：
    预习下周计划
    下载相关论文和代码
    确认数据集已准备
​
周末（灵活时间）：
  周六上午（可选）：
    补充本周未完成内容
    或深入一个感兴趣的细节
​
  周日（休息+规划）：
    完全休息（防止burnout）
    晚上30分钟：
      回顾本周进度
      更新四个月总进度追踪表
      规划下周重点
实验记录模板
每次实验的标准记录格式
===========================================
实验名称：[描述性名称，例如"Ballé2018在Kodak上的RD性能"]
实验日期：YYYY-MM-DD
实验目的：[一句话说明为什么做这个实验]
===========================================
​
1. 环境信息：
   GPU：[型号]
   PyTorch版本：[版本号]
   代码库版本：[commit hash或版本号]
​
2. 数据集：
   训练集：[名称 + 样本数量]
   测试集：[名称 + 样本数量]
​
3. 实验配置：
   关键超参数：
     λ = [值]
     batch_size = [值]
     learning_rate = [值]
     epochs = [值]
   
   与默认配置的差异：
     [列出所有修改的配置项]
​
4. 实验结果：
​
   定量结果：
   ┌────────┬──────────┬──────────┬──────────┐
   │ 方法   │ bpp      │ PSNR(dB) │ MS-SSIM  │
   ├────────┼──────────┼──────────┼──────────┤
   │ JPEG   │ 0.456    │ 33.12    │ 0.956    │
   │ BPG    │ 0.412    │ 35.67    │ 0.971    │
   │ 本方法  │ 0.389    │ 35.23    │ 0.969    │
   └────────┴──────────┴──────────┴──────────┘
   
   BD-Rate相对BPG：[+X% 或 -X%]
​
   训练曲线：
     最终训练loss：[值]
     最终验证loss：[值]
     是否收敛：[是/否/未知]
​
5. 现象分析：
   [描述观察到的现象，例如"在高码率点性能接近BPG，低码率点有明显差距"]
​
6. 原因分析：
   [分析现象的可能原因]
​
7. 结论：
   [这个实验告诉我们什么]
​
8. 遗留问题：
   [还有哪些不理解的地方，下次实验要验证什么]
​
9. 下一步：
   [基于这个结果，下一步做什么实验]
===========================================
论文笔记模板
每篇论文的标准笔记格式
===========================================
论文标题：[完整标题]
作者：[第一作者 et al.]
发表：[期刊/会议 + 年份]
论文链接：[arXiv或官方链接]
代码链接：[GitHub链接，若无则注明]
阅读日期：YYYY-MM-DD
重要程度：⭐⭐⭐⭐⭐（1-5星）
===========================================
​
1. 解决的问题：
   [这篇论文要解决什么问题？]
   [这个问题为什么重要？]
   [之前的方法为什么解决不好？]
​
2. 核心贡献（3点以内）：
   贡献1：[简洁描述]
   贡献2：[简洁描述]
   贡献3：[简洁描述]
​
3. 方法概述：
   [用自己的话描述方法，不超过200字]
   [包含关键的架构图或流程图（手绘后拍照）]
​
4. 关键公式：
   公式1：[写出公式，解释每个符号的含义]
   公式2：[写出公式，解释每个符号的含义]
   
   推导过程（重要公式）：
   [手推关键推导步骤]
​
5. 实验结果：
   数据集：[使用的测试集]
   主要结果：[最重要的数字/曲线]
   与之前SOTA的对比：[性能提升幅度]
​
6. 个人理解：
   最巧妙的设计：[你认为最聪明的部分]
   可能的局限：[你认为的弱点]
   
7. 与相关工作的关系：
   改进自：[哪篇论文]
   被哪些工作改进：[后续工作]
​
8. 代码对应关系：
   核心模块在代码中的位置：
     公式X → 文件名.py 第X行
     公式Y → 文件名.py 第X行
​
9. 遗留问题：
   [读完后还不明白的地方]
   [需要进一步验证的疑问]
===========================================
个人GitHub仓库组织建议
推荐仓库结构：
cv-compression-learning/
├── README.md                    # 学习路线总览
├── notes/                       # 论文笔记
│   ├── 01_basics/              # 基础知识笔记
│   │   ├── image_basics.md
│   │   └── video_basics.md
│   ├── 02_image_compression/   # 图像编码论文笔记
│   │   ├── balle2017.md
│   │   ├── balle2018.md
│   │   └── minnen2018.md
│   ├── 03_video_compression/   # 视频编码论文笔记
│   │   ├── dvc.md
│   │   └── dcvc.md
│   ├── 04_preprocessing/       # 前处理笔记
│   ├── 05_postprocessing/      # 后处理笔记
│   └── 06_surrogate/           # 代理编码器笔记
│
├── experiments/                 # 实验代码
│   ├── week01_image_basics/
│   │   ├── yuv_conversion.py
│   │   ├── dct_visualization.py
│   │   └── psnr_ssim.py
│   ├── week03_srcnn/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── week08_e2e_compression/
│   │   ├── train_balle2018.py
│   │   └── evaluate_kodak.py
│   └── week15_surrogate/
│       ├── diff_jpeg/
│       ├── surrogate_hevc/
│       └── joint_optimization/
│
├── tools/                       # 通用工具
│   ├── video_codec_wrapper.py  # FFmpeg封装
│   ├── bdrate.py               # BD-Rate计算
│   ├── rd_curve.py             # RD曲线绘制
│   ├── metrics.py              # 评测指标
│   └── visualization.py        # 可视化工具
│
├── blog/                        # 技术博客草稿
│   ├── 01_end_to_end_compression_intro.md
│   ├── 02_differentiable_jpeg.md
│   └── 03_surrogate_encoder.md
│
└── results/                     # 实验结果（不上传大文件）
    ├── rd_curves/
    └── visual_examples/
​
提交习惯：
  每天提交：实验代码和结果记录
  提交信息格式：
    [Week X] 描述内容
    例：[Week 8] Add Ballé2018 training code and Kodak evaluation
  避免：一次性提交大量内容
  原因：清晰的提交历史是学习过程的记录
常见困难与解决方案
-
困难1：论文公式看不懂
  症状：看到数学公式直接跳过
  危害：对方法一知半解，无法改进
  解决：
    Step1：先接受不懂，继续读完全文
    Step2：找到公式对应的代码实现
    Step3：代码→公式对应理解
    Step4：用简单数字代入公式验算
    Step5：查阅公式中用到的数学概念
    耐心：一篇论文的公式可能需要3-5天才完全理解
​
困难2：代码跑不通
  症状：按文档操作，各种报错
  解决步骤：
    Step1：仔细阅读错误信息（不要直接跳过）
    Step2：搜索错误信息（GitHub Issues优先）
    Step3：检查环境版本（CUDA/PyTorch版本冲突最常见）
    Step4：从最简单的示例开始，逐步复杂化
    Step5：实在不行，联系作者（GitHub Issue）
  预防：
    每个项目用独立conda环境
    记录每个项目的完整依赖版本
​
困难3：训练loss不收敛
  症状：loss震荡或不下降
  常见原因和解决：
    学习率太大：降低10倍试试
    梯度爆炸：加梯度裁剪（clip_grad_norm）
    数据预处理错误：检查输入值域
    网络结构问题：先用最简单的结构验证
    损失函数错误：检查损失计算代码
  调试顺序：
    先用小数据集（10张图）验证能过拟合
    能过拟合说明网络和代码没问题
    再解决泛化问题
​
困难4：实验结果与论文差距大
  症状：复现论文结果，PSNR差1-2dB
  常见原因：
    训练数据不同（论文可能用私有数据）
    训练轮数不够（有的模型需要训练很久）
    超参数差异（论文未公开所有细节）
    代码实现细节差异
  解决：
    先确认方向正确（趋势一致即可）
    查看论文是否有官方代码
    查看Issues中其他人的复现结果
    接受：能达到论文90%的性能就算复现成功
​
困难5：不知道如何选择下一步方向
  症状：学了很多，不知道做什么
  解决：
    回顾学习记录，找出最感兴趣的方向
    看该方向最新论文，找未解决的问题
    与导师/同事讨论（如果有的话）
    参加CLIC竞赛，实战中找方向
    从改进现有工作开始，不要求全新方向
​
困难6：时间不够用
  症状：计划总是完不成
  解决：
    重新评估时间预估（通常需要乘以2）
    砍掉非核心内容
    保持核心实践任务不变
    理论学习可以压缩
    原则：宁可少学但学透，不要贪多嚼不烂
四个月里程碑检查点
Month 1结束检查（第4周末）：
  □ 能用Python计算PSNR/SSIM/LPIPS
  □ 能用FFmpeg编码视频并提取帧
  □ 能在CIFAR-10上训练CNN分类器（>85%精度）
  □ 能在超分任务上训练EDSR并计算PSNR
  □ 理解感知损失和GAN损失的区别
  □ 建立了论文笔记仓库（至少6篇笔记）
  如果以上有未完成项：本月继续补，不要强行进入Month 2
​
Month 2结束检查（第8周末）：
  □ 能用HM以不同QP编码视频序列
  □ 能画Rate-Distortion曲线
  □ 能计算BD-Rate
  □ 理解HEVC的CTU划分和帧内预测
  □ 能在CompressAI中训练Ballé2018
  □ 画出至少3条RD曲线并与HEVC对比
  □ 技术博客至少2篇
​
Month 3结束检查（第12周末）：
  □ 能跑通DVC的推理代码
  □ 训练了前处理网络（即使BD-Rate很小）
  □ 训练了后处理网络（BD-Rate节省>3%）
  □ 理解端到端视频编码的完整框架
  □ 技术博客至少4篇
  □ 参与了至少一个开源项目的Issue讨论
​
Month 4结束检查（第16周末）：
  □ 手写实现了可微JPEG并验证梯度
  □ 训练了HEVC黑盒代理编码器（精度达标）
  □ 基于代理编码器训练了前处理网络
  □ 完成了完整的端到端系统评测
  □ 写了完整的实验报告（包含消融实验）
  □ 技术博客至少6篇
  □ GitHub仓库整洁，有完整文档
  □ 了解至少一个前沿方向（扩散/VCM/3D）
​
四个月总结里程碑：
  □ 精读论文 >= 30篇（有完整笔记）
  □ 复现实验 >= 10个
  □ 技术博客 >= 6篇
  □ 开源代码 >= 1个完整项目
  □ 参加或了解CLIC竞赛的规则和基线
资深工程师的最终寄语
关于"4个月成为资深工程师"的诚实评估：
​
4个月能达到的真实水平：
  ✓ 能独立阅读和理解该方向的顶会论文
  ✓ 能复现主流方法并进行合理改进
  ✓ 能设计和实施完整的实验
  ✓ 对编码方向有系统性认知
  ✓ 能与领域专家进行有深度的技术讨论
  ✓ 具备独立开展研究或工程工作的能力
​
4个月不能达到的：
  ✗ 对所有细节都深入理解
  ✗ 能提出全新的突破性方法
  ✗ 工业级系统的优化经验
  ✗ 对领域历史演进的深刻感悟
​
真正的资深需要：
  2-3年的持续深耕
  参与过真实产品的压缩优化
  发表过有影响力的论文或开源项目
  对领域局限性有清醒认识
​
但4个月的系统学习：
  是成为资深工程师最快的起点
  按照这个计划执行完毕
  你将具备资深工程师所需的知识框架
  剩下的是时间和实践的积累
​
最重要的一句话：
  开始执行比完美计划更重要
  从今天的Week 1 Day 1开始
  每天进步一点点
  四个月后回头看，会惊讶于自己的成长
