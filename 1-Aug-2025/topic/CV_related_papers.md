# Stereo 3D Gaussian Splatting SLAM for Outdoor Urban Scenes 

**Title (ZH)**: 户外城市场景的立体3D高斯散射SLAM 

**Authors**: Xiaohan Li, Ziren Gong, Fabio Tosi, Matteo Poggi, Stefano Mattoccia, Dong Liu, Jun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23677)  

**Abstract**: 3D Gaussian Splatting (3DGS) has recently gained popularity in SLAM applications due to its fast rendering and high-fidelity representation. However, existing 3DGS-SLAM systems have predominantly focused on indoor environments and relied on active depth sensors, leaving a gap for large-scale outdoor applications. We present BGS-SLAM, the first binocular 3D Gaussian Splatting SLAM system designed for outdoor scenarios. Our approach uses only RGB stereo pairs without requiring LiDAR or active sensors. BGS-SLAM leverages depth estimates from pre-trained deep stereo networks to guide 3D Gaussian optimization with a multi-loss strategy enhancing both geometric consistency and visual quality. Experiments on multiple datasets demonstrate that BGS-SLAM achieves superior tracking accuracy and mapping performance compared to other 3DGS-based solutions in complex outdoor environments. 

**Abstract (ZH)**: 基于双目的3D高斯斑点SLAM（BGS-SLAM）：复杂户外环境下的追踪精度和建图性能优于其他基于3D高斯斑点的方法 

---
# GSFusion:Globally Optimized LiDAR-Inertial-Visual Mapping for Gaussian Splatting 

**Title (ZH)**: GSFusion：全局优化的LiDAR-惯性-视觉融合建图方法用于高斯点云渲染 

**Authors**: Jaeseok Park, Chanoh Park, Minsu Kim, Soohwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.23273)  

**Abstract**: While 3D Gaussian Splatting (3DGS) has revolutionized photorealistic mapping, conventional approaches based on camera sensor, even RGB-D, suffer from fundamental limitations such as high computational load, failure in environments with poor texture or illumination, and short operational ranges. LiDAR emerges as a robust alternative, but its integration with 3DGS introduces new challenges, such as the need for exceptional global alignment for photorealistic quality and prolonged optimization times caused by sparse data. To address these challenges, we propose GSFusion, an online LiDAR-Inertial-Visual mapping system that ensures high-precision map consistency through a surfel-to-surfel constraint in the global pose-graph optimization. To handle sparse data, our system employs a pixel-aware Gaussian initialization strategy for efficient representation and a bounded sigmoid constraint to prevent uncontrolled Gaussian growth. Experiments on public and our datasets demonstrate our system outperforms existing 3DGS SLAM systems in terms of rendering quality and map-building efficiency. 

**Abstract (ZH)**: 尽管3D高斯散射（3DGS）已经革命性地推动了逼真 mapping 的发展，传统的基于相机传感器的方法，甚至包括RGB-D方法，仍面临着根本性的局限性，如计算负载高、在纹理或光照不良的环境中失效，以及短的操作范围。激光雷达（LiDAR）作为一个稳健的替代方案出现，但其与3DGS的集成引入了新的挑战，如为了保证逼真的质量需要优秀的全局对齐，并且由于稀疏数据导致优化时间延 长。为了解决这些挑战，我们提出了一种基于激光雷达-惯性-视觉的在线映射系统——GSFusion，该系统通过全局姿态图优化中的散el到散el约束来确保高精度的地图一致性。为了处理稀疏数据，我们的系统采用了一种像素感知的高斯初始化策略以实现高效表示，并引入了有界的Sigmoid约束以防止高斯的无控制增长。在公开数据集和我们自己的数据集上的实验表明，我们的系统在渲染质量和建图效率方面优于现有3DGS SLAM系统。 

---
# Online Estimation of Table-Top Grown Strawberry Mass in Field Conditions with Occlusions 

**Title (ZH)**: 田间遮挡条件下台上栽培草莓质量的在线估计 

**Authors**: Jinshan Zhen, Yuanyue Ge, Tianxiao Zhu, Hui Zhao, Ya Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.23487)  

**Abstract**: Accurate mass estimation of table-top grown strawberries under field conditions remains challenging due to frequent occlusions and pose variations. This study proposes a vision-based pipeline integrating RGB-D sensing and deep learning to enable non-destructive, real-time and online mass estimation. The method employed YOLOv8-Seg for instance segmentation, Cycle-consistent generative adversarial network (CycleGAN) for occluded region completion, and tilt-angle correction to refine frontal projection area calculations. A polynomial regression model then mapped the geometric features to mass. Experiments demonstrated mean mass estimation errors of 8.11% for isolated strawberries and 10.47% for occluded cases. CycleGAN outperformed large mask inpainting (LaMa) model in occlusion recovery, achieving superior pixel area ratios (PAR) (mean: 0.978 vs. 1.112) and higher intersection over union (IoU) scores (92.3% vs. 47.7% in the [0.9-1] range). This approach addresses critical limitations of traditional methods, offering a robust solution for automated harvesting and yield monitoring with complex occlusion patterns. 

**Abstract (ZH)**: 在田间条件下，基于视觉的草莓质量估计pipeline结合RGB-D感知和深度学习方法，以实现非破坏性、实时和在线质量估计仍具有挑战性。由于频繁遮挡和姿态变化，桌面生长的草莓的质量精确估计仍然具有挑战性。本文提出了一种结合RGB-D传感和深度学习的视觉pipeline，用于在田间条件下实现非破坏性、实时和在线的质量估计。方法采用了YOLOv8-Seg进行实例分割，Cycle-consistent生成对抗网络（CycleGAN）进行遮挡区域补全，并通过倾斜角度校正来细化正面投影面积计算。然后使用多项式回归模型将几何特征映射到质量上。实验结果显示，孤立草莓的质量估计均值误差为8.11%，遮挡情况下的均值误差为10.47%。CycleGAN在遮挡恢复方面优于Large Mask Inpainting（LaMa）模型，实现了更高的像素面积比率（PAR，均值：0.978 对比 1.112）和更高的交并比（IoU，0.9-1范围内为92.3% 对比 47.7%）。该方法克服了传统方法的关键局限性，为复杂遮挡模式下的自动化收获和产量监控提供了稳健的解决方案。 

---
# Vision-Language Fusion for Real-Time Autonomous Driving: Goal-Centered Cross-Attention of Camera, HD-Map, & Waypoints 

**Title (ZH)**: 基于视觉-语言融合的实时自动驾驶：目标导向的摄像机、高精度地图及航路点跨注意力机制 

**Authors**: Santosh Patapati, Trisanth Srinivasan, Murari Ambati  

**Link**: [PDF](https://arxiv.org/pdf/2507.23064)  

**Abstract**: Autonomous cars need geometric accuracy and semantic understanding to navigate complex environments, yet most stacks handle them separately. We present XYZ-Drive, a single vision-language model that reads a front-camera frame, a 25m $\times$ 25m overhead map, and the next waypoint, then outputs steering and speed. A lightweight goal-centered cross-attention layer lets waypoint tokens highlight relevant image and map patches, supporting both action and textual explanations, before the fused tokens enter a partially fine-tuned LLaMA-3.2 11B model.
On the MD-NEX Outdoor-Driving benchmark XYZ-Drive attains 95% success and 0.80 Success weighted by Path Length (SPL), surpassing PhysNav-DG by 15%. and halving collisions, all while significantly improving efficiency by using only a single branch. Sixteen ablations explain the gains. Removing any modality (vision, waypoint, map) drops success by up to 11%, confirming their complementary roles and rich connections. Replacing goal-centered attention with simple concatenation cuts 3% in performance, showing query-based fusion injects map knowledge more effectively. Keeping the transformer frozen loses 5%, showing the importance of fine-tuning when applying VLMs for specific tasks such as autonomous driving. Coarsening map resolution from 10 cm to 40 cm blurs lane edges and raises crash rate.
Overall, these results demonstrate that early, token-level fusion of intent and map layout enables accurate, transparent, real-time driving. 

**Abstract (ZH)**: 自主驾驶汽车需要几何精度和语义理解来导航复杂环境，但大多数系统分别处理这两方面。我们提出了XYZ-Drive，这是一种单一的视觉语言模型，它读取前视摄像头帧、25m×25m的鸟瞰图地图以及下一个行驶点，然后输出转向和速度。一个轻量级的目标中心交叉注意力层使行驶点标记突出相关图像和地图片段，支持行动和文本解释，之后融合后的标记进入部分微调的LLaMA-3.2 11B模型。
在MD-NEX室外驾驶基准测试中，XYZ-Drive实现95%的成功率和0.80的成功加权路径长度（SPL），超过PhysNav-DG 15%，同时减少碰撞率，并显著提高效率，仅使用单一分支。对六个消融实验解释了增益。去除任何模态（视觉、行驶点、地图）都会使成功率最多降低11%，确认它们的互补作用和丰富的联系。用简单的串联替换目标中心注意力会降低3%的性能，表明基于查询的融合更有效地注入地图知识。冻结变压器会降低5%的性能，表明在为特定任务（如自动驾驶）应用VLM时微调的重要性。将地图分辨率从10 cm减少到40 cm会模糊车道边缘并提高碰撞率。综上所述，这些结果表明，早期、标记级的意图与地图布局融合能够实现准确、透明、实时驾驶。 

---
# Phi-Ground Tech Report: Advancing Perception in GUI Grounding 

**Title (ZH)**: Phi-Ground 技术报告：提升GUI定位感知技术 

**Authors**: Miaosen Zhang, Ziqiang Xu, Jialiang Zhu, Qi Dai, Kai Qiu, Yifan Yang, Chong Luo, Tianyi Chen, Justin Wagle, Tim Franklin, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.23779)  

**Abstract**: With the development of multimodal reasoning models, Computer Use Agents (CUAs), akin to Jarvis from \textit{"Iron Man"}, are becoming a reality. GUI grounding is a core component for CUAs to execute actual actions, similar to mechanical control in robotics, and it directly leads to the success or failure of the system. It determines actions such as clicking and typing, as well as related parameters like the coordinates for clicks. Current end-to-end grounding models still achieve less than 65\% accuracy on challenging benchmarks like ScreenSpot-pro and UI-Vision, indicating they are far from being ready for deployment. % , as a single misclick can result in unacceptable consequences. In this work, we conduct an empirical study on the training of grounding models, examining details from data collection to model training. Ultimately, we developed the \textbf{Phi-Ground} model family, which achieves state-of-the-art performance across all five grounding benchmarks for models under $10B$ parameters in agent settings. In the end-to-end model setting, our model still achieves SOTA results with scores of \textit{\textbf{43.2}} on ScreenSpot-pro and \textit{\textbf{27.2}} on UI-Vision. We believe that the various details discussed in this paper, along with our successes and failures, not only clarify the construction of grounding models but also benefit other perception tasks. Project homepage: \href{this https URL}{this https URL} 

**Abstract (ZH)**: 随着多模态推理模型的发展，计算机使用代理（CUAs），类似于《钢铁侠》中的Jarvis，正在成为现实。GUI定位是CUAs执行实际操作的核心组件，类似于机器人领域的机械控制，直接关系到系统的成功或失败。它决定了点击和输入等相关操作及其参数，如点击坐标。当前的端到端定位模型在ScreenSpot-pro和UI-Vision等具有挑战性的基准测试中仍未能达到65%以上的准确率，表明它们尚未准备好部署。因此，单次误操作可能导致不可接受的后果。在本文中，我们对定位模型的训练进行了实证研究，从数据收集到模型训练进行了详细探讨。最终，我们开发了Phi-Ground模型系列，在agent设置下，该模型在所有五个定位基准测试中都达到了参数量低于10B的模型的最优性能。在端到端模型设置中，我们的模型在ScreenSpot-pro和UI-Vision上的得分分别为43.2和27.2，仍然取得了最优结果。我们相信，本文中讨论的各个方面，以及我们的成功与失败，不仅阐明了定位模型的构建，还对其他感知任务有益。项目主页：this https URL 

---
# Enhanced Velocity Field Modeling for Gaussian Video Reconstruction 

**Title (ZH)**: 增强速度场建模以实现高斯视频重建 

**Authors**: Zhenyang Li, Xiaoyang Bai, Tongchen Zhang, Pengfei Shen, Weiwei Xu, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2507.23704)  

**Abstract**: High-fidelity 3D video reconstruction is essential for enabling real-time rendering of dynamic scenes with realistic motion in virtual and augmented reality (VR/AR). The deformation field paradigm of 3D Gaussian splatting has achieved near-photorealistic results in video reconstruction due to the great representation capability of deep deformation networks. However, in videos with complex motion and significant scale variations, deformation networks often overfit to irregular Gaussian trajectories, leading to suboptimal visual quality. Moreover, the gradient-based densification strategy designed for static scene reconstruction proves inadequate to address the absence of dynamic content. In light of these challenges, we propose a flow-empowered velocity field modeling scheme tailored for Gaussian video reconstruction, dubbed FlowGaussian-VR. It consists of two core components: a velocity field rendering (VFR) pipeline which enables optical flow-based optimization, and a flow-assisted adaptive densification (FAD) strategy that adjusts the number and size of Gaussians in dynamic regions. We validate our model's effectiveness on multi-view dynamic reconstruction and novel view synthesis with multiple real-world datasets containing challenging motion scenarios, demonstrating not only notable visual improvements (over 2.5 dB gain in PSNR) and less blurry artifacts in dynamic textures, but also regularized and trackable per-Gaussian trajectories. 

**Abstract (ZH)**: 高保真3D视频重建对于在虚拟现实（VR）和增强现实（AR）中实时渲染具有现实运动的动态场景至关重要。基于变形场的3D高斯散点图方法由于深层变形网络的强大表示能力，在视频重建中实现了接近照片级的真实结果。然而，在具有复杂运动和显著尺度变化的视频中，变形网络往往会过度拟合不规则的高斯轨迹，导致视觉质量不佳。此外，为静态场景重建设计的基于梯度的密度增强策略无法有效解决动态内容的缺失问题。针对这些挑战，我们提出了一种流增强的速度场建模方案，名为FlowGaussian-VR。该方案包含两个核心组件：速度场渲染（VFR）管道，支持基于光流的优化，以及流辅助自适应密度增强（FAD）策略，该策略根据动态区域调整高斯的数量和大小。我们在包含具有挑战性运动场景的多个真实世界数据集的多视图动态重建和新颖视图合成中验证了我们模型的有效性，不仅展示了显著的视觉改善（PSNR增益超过2.5 dB）和动态纹理中较少的模糊 artefacts，而且还展示了规整且可追踪的每高斯轨迹。 

---
# Efficient Masked Attention Transformer for Few-Shot Classification and Segmentation 

**Title (ZH)**: 面向少样本分类和分割的高效遮蔽注意力变压器 

**Authors**: Dustin Carrión-Ojeda, Stefan Roth, Simone Schaub-Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2507.23642)  

**Abstract**: Few-shot classification and segmentation (FS-CS) focuses on jointly performing multi-label classification and multi-class segmentation using few annotated examples. Although the current state of the art (SOTA) achieves high accuracy in both tasks, it struggles with small objects. To overcome this, we propose the Efficient Masked Attention Transformer (EMAT), which improves classification and segmentation accuracy, especially for small objects. EMAT introduces three modifications: a novel memory-efficient masked attention mechanism, a learnable downscaling strategy, and parameter-efficiency enhancements. EMAT outperforms all FS-CS methods on the PASCAL-5$^i$ and COCO-20$^i$ datasets, using at least four times fewer trainable parameters. Moreover, as the current FS-CS evaluation setting discards available annotations, despite their costly collection, we introduce two novel evaluation settings that consider these annotations to better reflect practical scenarios. 

**Abstract (ZH)**: Few-shot 分类与分割 (FS-CS) 研究旨在使用少量标注样本同时进行多标签分类和多类分割。尽管当前最佳方法 (SOTA) 在两项任务上都达到了较高的准确性，但它在处理小物体时表现不佳。为了解决这一问题，我们提出了高效掩码注意力变换器 (EMAT)，该方法能够提高分类和分割的准确性，尤其适用于小物体。EMAT 引入了三种改进：一种新的内存高效掩码注意力机制、可学习的下采样策略和参数效率增强。EMAT 在 PASCAL-5$^i$ 和 COCO-20$^i$ 数据集上的表现优于所有 FS-CS 方法，使用了至少四倍少的可训练参数。此外，鉴于当前 FS-CS 评估设置忽略了可用的标注信息，尽管这些标注信息的收集成本高昂，我们引入了两种新的评估设置，考虑这些标注信息以更好地反映实际场景。 

---
# ART: Adaptive Relation Tuning for Generalized Relation Prediction 

**Title (ZH)**: 自适应关系调谐以实现通用关系预测 

**Authors**: Gopika Sudhakaran, Hikaru Shindo, Patrick Schramowski, Simone Schaub-Meyer, Kristian Kersting, Stefan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2507.23543)  

**Abstract**: Visual relation detection (VRD) is the task of identifying the relationships between objects in a scene. VRD models trained solely on relation detection data struggle to generalize beyond the relations on which they are trained. While prompt tuning has been used to adapt vision-language models (VLMs) for VRD, it uses handcrafted prompts and struggles with novel or complex relations. We argue that instruction tuning offers a more effective solution by fine-tuning VLMs on diverse instructional data. We thus introduce ART, an Adaptive Relation Tuning framework that adapts VLMs for VRD through instruction tuning and strategic instance selection. By converting VRD datasets into an instruction tuning format and employing an adaptive sampling algorithm, ART directs the VLM to focus on informative relations while maintaining generalizability. Specifically, we focus on the relation classification, where subject-object boxes are given and the model predicts the predicate between them. We tune on a held-in set and evaluate across multiple held-out datasets of varying complexity. Our approach strongly improves over its baselines and can infer unseen relation concepts, a capability absent in mainstream VRD methods. We demonstrate ART's practical value by using the predicted relations for segmenting complex scenes. 

**Abstract (ZH)**: 视觉关系检测中的自适应关系微调（ART） 

---
# I Am Big, You Are Little; I Am Right, You Are Wrong 

**Title (ZH)**: 我大你小；我对你说错 

**Authors**: David A. Kelly, Akchunya Chanchal, Nathan Blake  

**Link**: [PDF](https://arxiv.org/pdf/2507.23509)  

**Abstract**: Machine learning for image classification is an active and rapidly developing field. With the proliferation of classifiers of different sizes and different architectures, the problem of choosing the right model becomes more and more important.
While we can assess a model's classification accuracy statistically, our understanding of the way these models work is unfortunately limited. In order to gain insight into the decision-making process of different vision models, we propose using minimal sufficient pixels sets to gauge a model's `concentration': the pixels that capture the essence of an image through the lens of the model. By comparing position, overlap, and size of sets of pixels, we identify that different architectures have statistically different concentration, in both size and position. In particular, ConvNext and EVA models differ markedly from the others. We also identify that images which are misclassified are associated with larger pixels sets than correct classifications. 

**Abstract (ZH)**: 机器学习在图像分类领域的应用是一个活跃且快速发展的领域。随着不同大小和架构分类器的增多，选择合适的模型问题越来越重要。虽然我们可以从统计上评估模型的分类准确性，但对这些模型工作方式的理解是有限的。为了深入了解不同视觉模型的决策过程，我们提出使用最小充分像素集来衡量模型的“集中度”：即通过模型视角捕捉图像本质的像素集合。通过比较像素集的位置、重叠和大小，我们发现不同的架构在集中度上存在统计上的显著差异，尤其是在大小和位置上。特别是，ConvNext和EVA模型与其他模型有明显不同。我们还发现，误分类的图像与更大的像素集合相关，而正确分类的图像则与较小的像素集合相关。 

---
# Machine learning and machine learned prediction in chest X-ray images 

**Title (ZH)**: 机器学习与胸部X光图像的机器学习预测 

**Authors**: Shereiff Garrett, Abhinav Adhikari, Sarina Gautam, DaShawn Marquis Morris, Chandra Mani Adhikari  

**Link**: [PDF](https://arxiv.org/pdf/2507.23455)  

**Abstract**: Machine learning and artificial intelligence are fast-growing fields of research in which data is used to train algorithms, learn patterns, and make predictions. This approach helps to solve seemingly intricate problems with significant accuracy without explicit programming by recognizing complex relationships in data. Taking an example of 5824 chest X-ray images, we implement two machine learning algorithms, namely, a baseline convolutional neural network (CNN) and a DenseNet-121, and present our analysis in making machine-learned predictions in predicting patients with ailments. Both baseline CNN and DenseNet-121 perform very well in the binary classification problem presented in this work. Gradient-weighted class activation mapping shows that DenseNet-121 correctly focuses on essential parts of the input chest X-ray images in its decision-making more than the baseline CNN. 

**Abstract (ZH)**: 机器学习和人工智能是快速发展的研究领域，通过数据来训练算法、学习模式并进行预测。这种方法通过识别数据中的复杂关系，在无需显式编程的情况下，能够以显著的准确性解决看似复杂的问题。通过5824张胸部X光图像的例子，我们实现了两种机器学习算法，即基本卷积神经网络（CNN）和DenseNet-121，并在此工作中展示了在预测患者疾病方面的机器学习预测分析。基本CNN和DenseNet-121在本文呈现的二分类问题中表现都非常出色。梯度加权类激活映射显示，DenseNet-121在决策过程中更准确地聚焦于输入胸部X光图像的关键部分，远超基本CNN。 

---
# FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning 

**Title (ZH)**: FastDriveVLA: 高效端到端驾驶通过即插即用重建基础上的令牌剪枝 

**Authors**: Jiajun Cao, Qizhe Zhang, Peidong Jia, Xuhui Zhao, Bo Lan, Xiaoan Zhang, Xiaobao Wei, Sixiang Chen, Zhuo Li, Yang Wang, Liyun Li, Xianming Liu, Ming Lu, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.23318)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated significant potential in complex scene understanding and action reasoning, leading to their increasing adoption in end-to-end autonomous driving systems. However, the long visual tokens of VLA models greatly increase computational costs. Current visual token pruning methods in Vision-Language Models (VLM) rely on either visual token similarity or visual-text attention, but both have shown poor performance in autonomous driving scenarios. Given that human drivers concentrate on relevant foreground areas while driving, we assert that retaining visual tokens containing this foreground information is essential for effective decision-making. Inspired by this, we propose FastDriveVLA, a novel reconstruction-based vision token pruning framework designed specifically for autonomous driving. FastDriveVLA includes a plug-and-play visual token pruner called ReconPruner, which prioritizes foreground information through MAE-style pixel reconstruction. A novel adversarial foreground-background reconstruction strategy is designed to train ReconPruner for the visual encoder of VLA models. Once trained, ReconPruner can be seamlessly applied to different VLA models with the same visual encoder without retraining. To train ReconPruner, we also introduce a large-scale dataset called nuScenes-FG, consisting of 241K image-mask pairs with annotated foreground regions. Our approach achieves state-of-the-art results on the nuScenes closed-loop planning benchmark across different pruning ratios. 

**Abstract (ZH)**: 基于视觉-语言-动作的快速决策驱动框架FastDriveVLA 

---
# Impact of Hyperparameter Optimization on the Accuracy of Lightweight Deep Learning Models for Real-Time Image Classification 

**Title (ZH)**: 轻量级深度学习模型实时图像分类精度的超参数优化影响研究 

**Authors**: Vineet Kumar Rakesh, Soumya Mazumdar, Tapas Samanta, Sarbajit Pal, Amitabha Das  

**Link**: [PDF](https://arxiv.org/pdf/2507.23315)  

**Abstract**: Lightweight convolutional and transformer-based models have become vital for real-time image classification in resource-constrained applications, such as embedded systems and edge devices. This work analyzes the influence of hyperparameter adjustment on the accuracy and convergence behavior of seven efficient deep learning architectures: EfficientNetV2-S, ConvNeXt-T, MobileViT v2 (XXS/XS/S), MobileNetV3-L, TinyViT-21M, and RepVGG-A2. All models are trained on the ImageNet-1K dataset under consistent training settings, with an emphasis on real-time practicality. An comprehensive ablation study is undertaken to separate the effect of critical hyperparameters, including learning rate schedules, batch sizes, input resolution, data augmentation, regularization approaches, and optimizer choice. To assess appropriateness for real-time applications, each model is assessed not only in terms of Top-1 and Top-5 classification accuracy, but also in terms of inference time, parameter count, model size, and frames-per-second (FPS) on a GPU-accelerated edge deployment simulation. Results demonstrate that cosine learning rate decay and adjustable batch size may greatly boost both accuracy and convergence speed, while keeping low latency and memory cost. Notably, RepVGG-A2 achieves over 80% Top-1 accuracy with efficient inference performance, offering a compelling balance between accuracy and deployment cost for VGG-style models. The results give practical guidance for constructing resource-efficient deep learning models appropriate for real-time image processing pipelines. All code and training logs are publicly accessible at this https URL. 

**Abstract (ZH)**: 轻量级的卷积和变压器模型已成为嵌入式系统和边缘设备等资源受限应用中实时图像分类的关键。本研究分析了超参数调整对七种高效的深度学习架构（EfficientNetV2-S、ConvNeXt-T、MobileViT v2 (XXS/XS/S)、MobileNetV3-L、TinyViT-21M、RepVGG-A2）准确性和收敛行为的影响。所有模型均在一致的训练设置下使用ImageNet-1K数据集进行训练，并强调实时实用性。进行了一项全面的消融研究，以分离关键超参数（包括学习率调度、批量大小、输入分辨率、数据增强、正则化方法和优化器选择）的效果。为了评估其适用于实时应用的适宜性，每个模型不仅从Top-1和Top-5分类准确性，还从推理时间、参数量、模型大小和每秒帧数（FPS）的角度在GPU加速的边缘部署仿真中进行了评估。结果显示，余弦衰减学习率和可调批量大小可以显著提高准确性和收敛速度，同时保持低延迟和低成本。值得注意的是，RepVGG-A2在高效推理性能下实现了超过80%的Top-1准确率，为VGG风格的模型提供了准确性和部署成本之间具有竞争力的平衡。结果为构建适用于实时图像处理管道的资源高效深度学习模型提供了实用指导。所有代码和训练日志均可在该网址访问。 

---
# Towards Affordable Tumor Segmentation and Visualization for 3D Breast MRI Using SAM2 

**Title (ZH)**: 基于SAM2的可负担的3D乳腺MRI肿瘤分割与可视化研究 

**Authors**: Solha Kang, Eugene Kim, Joris Vankerschaver, Utku Ozbulak  

**Link**: [PDF](https://arxiv.org/pdf/2507.23272)  

**Abstract**: Breast MRI provides high-resolution volumetric imaging critical for tumor assessment and treatment planning, yet manual interpretation of 3D scans remains labor-intensive and subjective. While AI-powered tools hold promise for accelerating medical image analysis, adoption of commercial medical AI products remains limited in low- and middle-income countries due to high license costs, proprietary software, and infrastructure demands. In this work, we investigate whether the Segment Anything Model 2 (SAM2) can be adapted for low-cost, minimal-input 3D tumor segmentation in breast MRI. Using a single bounding box annotation on one slice, we propagate segmentation predictions across the 3D volume using three different slice-wise tracking strategies: top-to-bottom, bottom-to-top, and center-outward. We evaluate these strategies across a large cohort of patients and find that center-outward propagation yields the most consistent and accurate segmentations. Despite being a zero-shot model not trained for volumetric medical data, SAM2 achieves strong segmentation performance under minimal supervision. We further analyze how segmentation performance relates to tumor size, location, and shape, identifying key failure modes. Our results suggest that general-purpose foundation models such as SAM2 can support 3D medical image analysis with minimal supervision, offering an accessible and affordable alternative for resource-constrained settings. 

**Abstract (ZH)**: Segment Anything Model 2 (SAM2) 适用于低剂量输入的低成本三维肿瘤分割在乳腺MRI中的应用探究 

---
# Modeling Human Gaze Behavior with Diffusion Models for Unified Scanpath Prediction 

**Title (ZH)**: 使用扩散模型建模人类注视行为以实现统一的扫视路径预测 

**Authors**: Giuseppe Cartella, Vittorio Cuculo, Alessandro D'Amelio, Marcella Cornia, Giuseppe Boccignone, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2507.23021)  

**Abstract**: Predicting human gaze scanpaths is crucial for understanding visual attention, with applications in human-computer interaction, autonomous systems, and cognitive robotics. While deep learning models have advanced scanpath prediction, most existing approaches generate averaged behaviors, failing to capture the variability of human visual exploration. In this work, we present ScanDiff, a novel architecture that combines diffusion models with Vision Transformers to generate diverse and realistic scanpaths. Our method explicitly models scanpath variability by leveraging the stochastic nature of diffusion models, producing a wide range of plausible gaze trajectories. Additionally, we introduce textual conditioning to enable task-driven scanpath generation, allowing the model to adapt to different visual search objectives. Experiments on benchmark datasets show that ScanDiff surpasses state-of-the-art methods in both free-viewing and task-driven scenarios, producing more diverse and accurate scanpaths. These results highlight its ability to better capture the complexity of human visual behavior, pushing forward gaze prediction research. Source code and models are publicly available at this https URL. 

**Abstract (ZH)**: 预测人类视扫描路径对于理解视觉注意至关重要，并在人机交互、自主系统和认知机器人等领域有着广泛的应用。尽管深度学习模型在扫描路径预测方面取得了进展，但现有方法大多生成平均行为，无法捕捉人类视觉探索的变异性。在这项工作中，我们提出了一种名为ScanDiff的新架构，该架构结合了扩散模型和Vision Transformers，以生成多样且真实的人类视扫描路径。我们的方法通过利用扩散模型的随机性质，明确建模扫描路径的变异性，产生多种可能的眼球运动轨迹。此外，我们引入了文本条件，以实现任务驱动的扫描路径生成，使模型能够适应不同的视觉搜索目标。基准数据集上的实验结果显示，ScanDiff在自由观看和任务驱动场景中均超越了现有最先进的方法，生成的扫描路径更具多样性和准确性。这些结果突显了其更好地捕捉人类视觉行为复杂性的能力，推动了视扫描预测研究的发展。源代码和模型已在以下网址公开。 

---
# From Propagator to Oscillator: The Dual Role of Symmetric Differential Equations in Neural Systems 

**Title (ZH)**: 从传播子到振荡器：对称微分方程在神经系统中的双重角色 

**Authors**: Kun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22916)  

**Abstract**: In our previous work, we proposed a novel neuron model based on symmetric differential equations and demonstrated its potential as an efficient signal propagator. Building upon that foundation, the present study delves deeper into the intrinsic dynamics and functional diversity of this model. By systematically exploring the parameter space and employing a range of mathematical analysis tools, we theoretically reveal the system 's core property of functional duality. Specifically, the model exhibits two distinct trajectory behaviors: one is asymptotically stable, corresponding to a reliable signal propagator; the other is Lyapunov stable, characterized by sustained self-excited oscillations, functioning as a signal generator. To enable effective monitoring and prediction of system states during simulations, we introduce a novel intermediate-state metric termed on-road energy. Simulation results confirm that transitions between the two functional modes can be induced through parameter adjustments or modifications to the connection structure. Moreover, we show that oscillations can be effectively suppressed by introducing external signals. These findings draw a compelling parallel to the dual roles of biological neurons in both information transmission and rhythm generation, thereby establishing a solid theoretical basis and a clear functional roadmap for the broader application of this model in neuromorphic engineering. 

**Abstract (ZH)**: 基于对称微分方程的新颖神经元模型的内在动力学与功能多样性研究 

---
