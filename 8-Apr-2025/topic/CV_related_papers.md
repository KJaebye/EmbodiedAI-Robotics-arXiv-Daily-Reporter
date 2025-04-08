# CloSE: A Compact Shape- and Orientation-Agnostic Cloth State Representation 

**Title (ZH)**: CloSE: 一种紧凑的不依赖形状和方向的衣物状态表示方法 

**Authors**: Jay Kamat, Júlia Borràs, Carme Torras  

**Link**: [PDF](https://arxiv.org/pdf/2504.05033)  

**Abstract**: Cloth manipulation is a difficult problem mainly because of the non-rigid nature of cloth, which makes a good representation of deformation essential. We present a new representation for the deformation-state of clothes. First, we propose the dGLI disk representation, based on topological indices computed for segments on the edges of the cloth mesh border that are arranged on a circular grid. The heat-map of the dGLI disk uncovers patterns that correspond to features of the cloth state that are consistent for different shapes, sizes of positions of the cloth, like the corners and the fold locations. We then abstract these important features from the dGLI disk onto a circle, calling it the Cloth StatE representation (CloSE). This representation is compact, continuous, and general for different shapes. Finally, we show the strengths of this representation in two relevant applications: semantic labeling and high- and low-level planning. The code, the dataset and the video can be accessed from : this https URL 

**Abstract (ZH)**: 布料操纵是一个困难的问题，主要是由于布料的非刚性性质，使得变形的良好表示至关重要。我们提出了一种新的布料变形状态表示方法。首先，我们基于布料网格边界边缘上的段在圆形网格上的排列，提出了dGLI圆盘表示法。dGLI圆盘的热图揭示了与不同形状、大小和位置的布料状态特征一致的模式，如角落和折叠位置。然后，我们将这些重要特征从dGLI圆盘抽象到一个圆上，称为布料状态表示（CloSE）。这种表示方法紧凑、连续且适用于不同形状。最后，我们展示了此表示方法在两个相关应用中的优势：语义标注和高低级规划。代码、数据集和视频可以从以下链接访问：this https URL 

---
# SELC: Self-Supervised Efficient Local Correspondence Learning for Low Quality Images 

**Title (ZH)**: SELCl：自监督高效局部对应学习在低质量图像中的应用 

**Authors**: Yuqing Wang, Yan Wang, Hailiang Tang, Xiaoji Niu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04497)  

**Abstract**: Accurate and stable feature matching is critical for computer vision tasks, particularly in applications such as Simultaneous Localization and Mapping (SLAM). While recent learning-based feature matching methods have demonstrated promising performance in challenging spatiotemporal scenarios, they still face inherent trade-offs between accuracy and computational efficiency in specific settings. In this paper, we propose a lightweight feature matching network designed to establish sparse, stable, and consistent correspondence between multiple frames. The proposed method eliminates the dependency on manual annotations during training and mitigates feature drift through a hybrid self-supervised paradigm. Extensive experiments validate three key advantages: (1) Our method operates without dependency on external prior knowledge and seamlessly incorporates its hybrid training mechanism into original datasets. (2) Benchmarked against state-of-the-art deep learning-based methods, our approach maintains equivalent computational efficiency at low-resolution scales while achieving a 2-10x improvement in computational efficiency for high-resolution inputs. (3) Comparative evaluations demonstrate that the proposed hybrid self-supervised scheme effectively mitigates feature drift in long-term tracking while maintaining consistent representation across image sequences. 

**Abstract (ZH)**: 准确且稳定的特征匹配对于计算机视觉任务至关重要，特别是在Simultaneous Localization and Mapping (SLAM)等应用中。尽管近年来基于学习的特征匹配方法在复杂的时空场景中展现了令人鼓舞的性能，但在特定情况下它们仍然面临准确性和计算效率之间的固有权衡。在本文中，我们提出了一种轻量级特征匹配网络，旨在建立多帧之间的稀疏、稳定且一致的对应关系。所提出的方法在训练过程中消除对外部先验知识的依赖，并通过混合自监督范式来减轻特征漂移。广泛的实验验证了三个关键优势：(1) 该方法不依赖外部先验知识，并能无缝将其实现机制整合到原始数据集中。(2) 与最先进的基于深度学习的方法相比，我们的方法在低分辨率尺度下保持了相当的计算效率，并且在高分辨率输入下可实现2-10倍的计算效率提升。(3) 比较评估表明，提出的混合自监督方案有效地减轻了长时间跟踪中的特征漂移，并保持了图像序列中的一致表示。 

---
# eKalibr-Stereo: Continuous-Time Spatiotemporal Calibration for Event-Based Stereo Visual Systems 

**Title (ZH)**: eKalibr-立体视觉系统基于事件的连续时空校准 

**Authors**: Shuolong Chen, Xingxing Li, Liu Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.04451)  

**Abstract**: The bioinspired event camera, distinguished by its exceptional temporal resolution, high dynamic range, and low power consumption, has been extensively studied in recent years for motion estimation, robotic perception, and object detection. In ego-motion estimation, the stereo event camera setup is commonly adopted due to its direct scale perception and depth recovery. For optimal stereo visual fusion, accurate spatiotemporal (extrinsic and temporal) calibration is required. Considering that few stereo visual calibrators orienting to event cameras exist, based on our previous work eKalibr (an event camera intrinsic calibrator), we propose eKalibr-Stereo for accurate spatiotemporal calibration of event-based stereo visual systems. To improve the continuity of grid pattern tracking, building upon the grid pattern recognition method in eKalibr, an additional motion prior-based tracking module is designed in eKalibr-Stereo to track incomplete grid patterns. Based on tracked grid patterns, a two-step initialization procedure is performed to recover initial guesses of piece-wise B-splines and spatiotemporal parameters, followed by a continuous-time batch bundle adjustment to refine the initialized states to optimal ones. The results of extensive real-world experiments show that eKalibr-Stereo can achieve accurate event-based stereo spatiotemporal calibration. The implementation of eKalibr-Stereo is open-sourced at (this https URL) to benefit the research community. 

**Abstract (ZH)**: 受生物启发的事件相机因其卓越的时间分辨率、高动态范围和低功耗，在近期被广泛研究用于运动估计、机器人感知和物体检测。在自我运动估计中，由于其直接尺度感知和深度恢复能力，立体事件相机设置被广泛应用。为了实现最佳立体视觉融合，需要进行精确的空间-时间（外在和时间）校准。鉴于针对事件相机的立体视觉校准工具较少，基于我们之前的工作eKalibr（事件相机内在校准器），我们提出eKalibr-Stereo用于立体事件视觉系统的精确空间-时间校准。为了提高网格图案跟踪的连续性，基于eKalibr中的网格图案识别方法，在eKalibr-Stereo中设计了一个基于运动先验的跟踪模块来跟踪不完整的网格图案。基于跟踪的网格图案，我们执行两步初始化程序来恢复分段B样条和空间-时间参数的初始猜测，然后进行连续时间批量多项式调整以优化初始化状态。广泛的实地实验结果表明，eKalibr-Stereo可以实现精确的事件驱动立体空间-时间校准。eKalibr-Stereo的实现已经开源（https://this-url/），以造福研究界。 

---
# A Convex and Global Solution for the P$n$P Problem in 2D Forward-Looking Sonar 

**Title (ZH)**: 2D 前向声纳 P^nP 问题的凸全局解 

**Authors**: Jiayi Su, Jingyu Qian, Liuqing Yang, Yufan Yuan, Yanbing Fu, Jie Wu, Yan Wei, Fengzhong Qu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04445)  

**Abstract**: The perspective-$n$-point (P$n$P) problem is important for robotic pose estimation. It is well studied for optical cameras, but research is lacking for 2D forward-looking sonar (FLS) in underwater scenarios due to the vastly different imaging principles. In this paper, we demonstrate that, despite the nonlinearity inherent in sonar image formation, the P$n$P problem for 2D FLS can still be effectively addressed within a point-to-line (PtL) 3D registration paradigm through orthographic approximation. The registration is then resolved by a duality-based optimal solver, ensuring the global optimality. For coplanar cases, a null space analysis is conducted to retrieve the solutions from the dual formulation, enabling the methods to be applied to more general cases. Extensive simulations have been conducted to systematically evaluate the performance under different settings. Compared to non-reprojection-optimized state-of-the-art (SOTA) methods, the proposed approach achieves significantly higher precision. When both methods are optimized, ours demonstrates comparable or slightly superior precision. 

**Abstract (ZH)**: 基于点到线（PtL）三维配准范式的2D前向声呐（FLS）视角-n点（P$n$P）问题研究 

---
# Data Scaling Laws for End-to-End Autonomous Driving 

**Title (ZH)**: 端到端自动驾驶的数据标度律 

**Authors**: Alexander Naumann, Xunjiang Gu, Tolga Dimlioglu, Mariusz Bojarski, Alperen Degirmenci, Alexander Popov, Devansh Bisla, Marco Pavone, Urs Müller, Boris Ivanovic  

**Link**: [PDF](https://arxiv.org/pdf/2504.04338)  

**Abstract**: Autonomous vehicle (AV) stacks have traditionally relied on decomposed approaches, with separate modules handling perception, prediction, and planning. However, this design introduces information loss during inter-module communication, increases computational overhead, and can lead to compounding errors. To address these challenges, recent works have proposed architectures that integrate all components into an end-to-end differentiable model, enabling holistic system optimization. This shift emphasizes data engineering over software integration, offering the potential to enhance system performance by simply scaling up training resources. In this work, we evaluate the performance of a simple end-to-end driving architecture on internal driving datasets ranging in size from 16 to 8192 hours with both open-loop metrics and closed-loop simulations. Specifically, we investigate how much additional training data is needed to achieve a target performance gain, e.g., a 5% improvement in motion prediction accuracy. By understanding the relationship between model performance and training dataset size, we aim to provide insights for data-driven decision-making in autonomous driving development. 

**Abstract (ZH)**: 自主驾驶系统堆栈传统上依赖于分解的方法，每个模块分别进行感知、预测和规划。然而，这种设计在模块间通信时会导致信息丢失，增加计算负担，并可能导致累积错误。为了解决这些问题，近期的研究提出了将所有组件整合到端到端可微分模型中的架构，从而实现整体系统优化。这种转变更注重数据工程而非软件集成，可以通过简单地扩展训练资源来提升系统性能。在本工作中，我们评估了一种简单的端到端驾驶架构在从16小时到8192小时不等内部驾驶数据集上的性能，包括开环度量和闭环仿真。具体地，我们探讨了实现目标性能提升（例如在运动预测准确性方面提高5%）所需的额外训练数据量。通过理解模型性能与训练数据集规模之间的关系，我们旨在为自主驾驶开发中的数据驱动决策提供见解。 

---
# Stereo-LiDAR Fusion by Semi-Global Matching With Discrete Disparity-Matching Cost and Semidensification 

**Title (ZH)**: 基于半全局匹配的离散视差匹配代价与半稀疏化立体-LiDAR融合 

**Authors**: Yasuhiro Yao, Ryoichi Ishikawa, Takeshi Oishi  

**Link**: [PDF](https://arxiv.org/pdf/2504.05148)  

**Abstract**: We present a real-time, non-learning depth estimation method that fuses Light Detection and Ranging (LiDAR) data with stereo camera input. Our approach comprises three key techniques: Semi-Global Matching (SGM) stereo with Discrete Disparity-matching Cost (DDC), semidensification of LiDAR disparity, and a consistency check that combines stereo images and LiDAR data. Each of these components is designed for parallelization on a GPU to realize real-time performance. When it was evaluated on the KITTI dataset, the proposed method achieved an error rate of 2.79\%, outperforming the previous state-of-the-art real-time stereo-LiDAR fusion method, which had an error rate of 3.05\%. Furthermore, we tested the proposed method in various scenarios, including different LiDAR point densities, varying weather conditions, and indoor environments, to demonstrate its high adaptability. We believe that the real-time and non-learning nature of our method makes it highly practical for applications in robotics and automation. 

**Abstract (ZH)**: 我们提出了一种实时、非学习的深度估计方法，该方法结合了Light Detection and Ranging (LiDAR) 数据和立体相机输入。该方法包含三种关键技术：半全局匹配（SGM）立体视觉与离散视差匹配成本（DDC）、LiDAR视差半稠密化以及一个结合立体图像和LiDAR数据的一致性检查。这些组件均设计为可在GPU上并行化以实现实时性能。当在KITTI数据集上进行评估时，所提出的方法实现了2.79%的误差率，优于之前实时立体视觉-LiDAR融合的最佳方法（误差率为3.05%）。此外，我们在不同的LiDAR点密度、不同天气条件以及室内环境中测试了所提出的方法，以展示其高适应性。我们认为，本方法的实时和非学习特性使其在机器人技术和自动化领域具有很高的实用性。 

---
# GAMDTP: Dynamic Trajectory Prediction with Graph Attention Mamba Network 

**Title (ZH)**: GAMDTP：基于图注意力Mamba网络的动力学轨迹预测 

**Authors**: Yunxiang Liu, Hongkuo Niu, Jianlin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04862)  

**Abstract**: Accurate motion prediction of traffic agents is crucial for the safety and stability of autonomous driving systems. In this paper, we introduce GAMDTP, a novel graph attention-based network tailored for dynamic trajectory prediction. Specifically, we fuse the result of self attention and mamba-ssm through a gate mechanism, leveraging the strengths of both to extract features more efficiently and accurately, in each graph convolution layer. GAMDTP encodes the high-definition map(HD map) data and the agents' historical trajectory coordinates and decodes the network's output to generate the final prediction results. Additionally, recent approaches predominantly focus on dynamically fusing historical forecast results and rely on two-stage frameworks including proposal and refinement. To further enhance the performance of the two-stage frameworks we also design a scoring mechanism to evaluate the prediction quality during the proposal and refinement processes. Experiments on the Argoverse dataset demonstrates that GAMDTP achieves state-of-the-art performance, achieving superior accuracy in dynamic trajectory prediction. 

**Abstract (ZH)**: 基于图注意机制的GAMDTP网络在交通代理动态轨迹预测中的应用 

---
# Inverse++: Vision-Centric 3D Semantic Occupancy Prediction Assisted with 3D Object Detection 

**Title (ZH)**: Inverse++: 以视觉为中心的3D语义占有预测辅助以3D物体检测 

**Authors**: Zhenxing Ming, Julie Stephany Berrio, Mao Shan, Stewart Worrall  

**Link**: [PDF](https://arxiv.org/pdf/2504.04732)  

**Abstract**: 3D semantic occupancy prediction aims to forecast detailed geometric and semantic information of the surrounding environment for autonomous vehicles (AVs) using onboard surround-view cameras. Existing methods primarily focus on intricate inner structure module designs to improve model performance, such as efficient feature sampling and aggregation processes or intermediate feature representation formats. In this paper, we explore multitask learning by introducing an additional 3D supervision signal by incorporating an additional 3D object detection auxiliary branch. This extra 3D supervision signal enhances the model's overall performance by strengthening the capability of the intermediate features to capture small dynamic objects in the scene, and these small dynamic objects often include vulnerable road users, i.e. bicycles, motorcycles, and pedestrians, whose detection is crucial for ensuring driving safety in autonomous vehicles. Extensive experiments conducted on the nuScenes datasets, including challenging rainy and nighttime scenarios, showcase that our approach attains state-of-the-art results, achieving an IoU score of 31.73% and a mIoU score of 20.91% and excels at detecting vulnerable road users (VRU). The code will be made available at:this https URL 

**Abstract (ZH)**: 基于车载全景相机的3D语义 occupancy 预测旨在利用自动驾驶车辆上的全景相机，预测周围环境的详细几何和语义信息。现有方法主要集中在精细的内部结构模块设计以提高模型性能，例如高效的特征采样和聚合过程或中间特征表示格式。在本文中，我们通过引入附加的3D监督信号，探索多任务学习，该附加信号通过结合一个额外的3D物体检测辅助分支来实现。该附加的3D监督信号通过增强中间特征捕捉场景中小动态物体的能力而增强了模型的整体性能，这些小动态物体通常包括弱势道路使用者，即自行车、摩托车和行人，其检测对于确保自动驾驶车辆的驾驶安全至关重要。在nuScenes数据集上的广泛实验，包括恶劣的雨天和夜间场景，展示了我们的方法取得了最先进的成果，实现了IoU分数为31.73%和mIoU分数为20.91%，并擅长检测弱势道路使用者（VRU）。代码将在该链接处提供：this https URL。 

---
# WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments 

**Title (ZH)**: WildGS-SLAM: 动态环境中的单目高斯点云SLAM 

**Authors**: Jianhao Zheng, Zihan Zhu, Valentin Bieri, Marc Pollefeys, Songyou Peng, Iro Armeni  

**Link**: [PDF](https://arxiv.org/pdf/2504.03886)  

**Abstract**: We present WildGS-SLAM, a robust and efficient monocular RGB SLAM system designed to handle dynamic environments by leveraging uncertainty-aware geometric mapping. Unlike traditional SLAM systems, which assume static scenes, our approach integrates depth and uncertainty information to enhance tracking, mapping, and rendering performance in the presence of moving objects. We introduce an uncertainty map, predicted by a shallow multi-layer perceptron and DINOv2 features, to guide dynamic object removal during both tracking and mapping. This uncertainty map enhances dense bundle adjustment and Gaussian map optimization, improving reconstruction accuracy. Our system is evaluated on multiple datasets and demonstrates artifact-free view synthesis. Results showcase WildGS-SLAM's superior performance in dynamic environments compared to state-of-the-art methods. 

**Abstract (ZH)**: WildGS-SLAM：一种通过利用不确定性意识几何映射来处理动态环境的鲁棒高效单目RGB SLAM系统 

---
# 3D Universal Lesion Detection and Tagging in CT with Self-Training 

**Title (ZH)**: 3D自训练CT病变检测与标记 

**Authors**: Jared Frazier, Tejas Sudharshan Mathai, Jianfei Liu, Angshuman Paul, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2504.05201)  

**Abstract**: Radiologists routinely perform the tedious task of lesion localization, classification, and size measurement in computed tomography (CT) studies. Universal lesion detection and tagging (ULDT) can simultaneously help alleviate the cumbersome nature of lesion measurement and enable tumor burden assessment. Previous ULDT approaches utilize the publicly available DeepLesion dataset, however it does not provide the full volumetric (3D) extent of lesions and also displays a severe class imbalance. In this work, we propose a self-training pipeline to detect 3D lesions and tag them according to the body part they occur in. We used a significantly limited 30\% subset of DeepLesion to train a VFNet model for 2D lesion detection and tagging. Next, the 2D lesion context was expanded into 3D, and the mined 3D lesion proposals were integrated back into the baseline training data in order to retrain the model over multiple rounds. Through the self-training procedure, our VFNet model learned from its own predictions, detected lesions in 3D, and tagged them. Our results indicated that our VFNet model achieved an average sensitivity of 46.9\% at [0.125:8] false positives (FP) with a limited 30\% data subset in comparison to the 46.8\% of an existing approach that used the entire DeepLesion dataset. To our knowledge, we are the first to jointly detect lesions in 3D and tag them according to the body part label. 

**Abstract (ZH)**: Radiologists 常规地在计算机断层扫描（CT）研究中执行病变定位、分类和尺寸测量的任务。通用病变检测和标记（ULDT）可以同时减轻病变测量的繁琐性质，使肿瘤负担评估成为可能。之前的 ULDT 方法利用了公开的 DeepLesion 数据集，但该数据集并未提供病变的完整体积（3D）范围，并且严重存在类别不平衡问题。在本工作中，我们提出了一种自训练管道来检测 3D 病变并根据其发生的部位对其进行标记。我们使用 DeepLesion 的显著局限的 30% 子集来训练一个 VFNet 模型进行 2D 病变检测和标记。接下来，2D 病变上下文扩展到 3D，并提取的 3D 病变建议被重新集成到基础训练数据中，以便在多轮迭代中重新训练模型。通过自训练过程，我们的 VFNet 模型从自己的预测中学习，在有限的 30% 数据子集中实现了在 [0.125:8] 假阳性（FP）下的平均灵敏度为 46.9%，与使用整个 DeepLesion 数据集的现有方法相比为 46.8%。据我们所知，我们是第一个联合检测 3D 病变并根据解剖部位标签对其进行标记的研究工作。 

---
# Universal Lymph Node Detection in Multiparametric MRI with Selective Augmentation 

**Title (ZH)**: 基于选择性增强的多参数MRI通用淋巴结检测 

**Authors**: Tejas Sudharshan Mathai, Sungwon Lee, Thomas C. Shen, Zhiyong Lu, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2504.05196)  

**Abstract**: Robust localization of lymph nodes (LNs) in multiparametric MRI (mpMRI) is critical for the assessment of lymphadenopathy. Radiologists routinely measure the size of LN to distinguish benign from malignant nodes, which would require subsequent cancer staging. Sizing is a cumbersome task compounded by the diverse appearances of LNs in mpMRI, which renders their measurement difficult. Furthermore, smaller and potentially metastatic LNs could be missed during a busy clinical day. To alleviate these imaging and workflow problems, we propose a pipeline to universally detect both benign and metastatic nodes in the body for their ensuing measurement. The recently proposed VFNet neural network was employed to identify LN in T2 fat suppressed and diffusion weighted imaging (DWI) sequences acquired by various scanners with a variety of exam protocols. We also use a selective augmentation technique known as Intra-Label LISA (ILL) to diversify the input data samples the model sees during training, such that it improves its robustness during the evaluation phase. We achieved a sensitivity of $\sim$83\% with ILL vs. $\sim$80\% without ILL at 4 FP/vol. Compared with current LN detection approaches evaluated on mpMRI, we show a sensitivity improvement of $\sim$9\% at 4 FP/vol. 

**Abstract (ZH)**: 在多参数MRI (mpMRI) 中稳健定位淋巴结 (LNs) 对淋巴腺病的评估至关重要。我们提出了一种管道来普遍检测体内良性及转移性淋巴结，以便随后进行测量。我们采用了最近提出的VFNet神经网络，该网络能够在不同扫描器和多种检查协议下识别T2脂肪抑制和扩散加权成像 (DWI) 序列中的淋巴结。此外，我们还使用了一种名为Intra-Label LISA (ILL) 的选择性增强技术，在模型训练过程中多样化其输入数据样本，从而提高其评估阶段的稳健性。与当前的淋巴结检测方法在mpMRI上的评估相比，我们在每升4个假阳性下的敏感性提高了约9%。 

---
# SSLFusion: Scale & Space Aligned Latent Fusion Model for Multimodal 3D Object Detection 

**Title (ZH)**: SSLFusion：面向多模态3D物体检测的尺度与空间对齐潜在融合模型 

**Authors**: Bonan Ding, Jin Xie, Jing Nie, Jiale Cao  

**Link**: [PDF](https://arxiv.org/pdf/2504.05170)  

**Abstract**: Multimodal 3D object detection based on deep neural networks has indeed made significant progress. However, it still faces challenges due to the misalignment of scale and spatial information between features extracted from 2D images and those derived from 3D point clouds. Existing methods usually aggregate multimodal features at a single stage. However, leveraging multi-stage cross-modal features is crucial for detecting objects of various scales. Therefore, these methods often struggle to integrate features across different scales and modalities effectively, thereby restricting the accuracy of detection. Additionally, the time-consuming Query-Key-Value-based (QKV-based) cross-attention operations often utilized in existing methods aid in reasoning the location and existence of objects by capturing non-local contexts. However, this approach tends to increase computational complexity. To address these challenges, we present SSLFusion, a novel Scale & Space Aligned Latent Fusion Model, consisting of a scale-aligned fusion strategy (SAF), a 3D-to-2D space alignment module (SAM), and a latent cross-modal fusion module (LFM). SAF mitigates scale misalignment between modalities by aggregating features from both images and point clouds across multiple levels. SAM is designed to reduce the inter-modal gap between features from images and point clouds by incorporating 3D coordinate information into 2D image features. Additionally, LFM captures cross-modal non-local contexts in the latent space without utilizing the QKV-based attention operations, thus mitigating computational complexity. Experiments on the KITTI and DENSE datasets demonstrate that our SSLFusion outperforms state-of-the-art methods. Our approach obtains an absolute gain of 2.15% in 3D AP, compared with the state-of-art method GraphAlign on the moderate level of the KITTI test set. 

**Abstract (ZH)**: 基于深度神经网络的多模态3D目标检测已取得显著进展，但由于从2D图像中提取的特征与从3D点云中提取的特征在尺度和空间信息上的不对齐，仍面临挑战。现有方法通常在单个阶段聚合多模态特征。然而，利用多阶段跨模态特征对于检测不同尺度的目标至关重要。因此，这些方法往往难以有效地跨尺度和模态整合特征，从而限制了检测的准确性。此外，现有方法常用耗时的Query-Key-Value（QKV）基跨注意力操作捕获非局部上下文，推理目标的位置和存在，但这种方法增加了计算复杂度。为了解决这些挑战，我们提出了SSLFusion，一种新颖的尺度与空间对齐潜在融合模型，包括尺度对齐融合策略（SAF）、3D到2D空间对齐模块（SAM）和潜在跨模态融合模块（LFM）。SAF 通过在多个级别上从图像和点云中聚合特征来缓解模态之间的尺度不一致性。SAM 通过将3D坐标信息融入2D图像特征，旨在减少图像和点云特征之间的跨模态间隙。此外，LFM 在潜在空间中捕获跨模态的非局部上下文，而不使用QKV基注意力操作，从而减轻计算复杂度。在Kitti和Dense数据集上的实验表明，我们的SSLFusion 超过了现有方法。在KITTI测试集的中等水平上，我们的方法在3D AP上相对于现有方法GraphAlign 的绝对增益为2.15%。 

---
# EffOWT: Transfer Visual Language Models to Open-World Tracking Efficiently and Effectively 

**Title (ZH)**: EffOWT: 有效地高效转移视觉语言模型到开放世界跟踪 

**Authors**: Bingyang Wang, Kaer Huang, Bin Li, Yiqiang Yan, Lihe Zhang, Huchuan Lu, You He  

**Link**: [PDF](https://arxiv.org/pdf/2504.05141)  

**Abstract**: Open-World Tracking (OWT) aims to track every object of any category, which requires the model to have strong generalization capabilities. Trackers can improve their generalization ability by leveraging Visual Language Models (VLMs). However, challenges arise with the fine-tuning strategies when VLMs are transferred to OWT: full fine-tuning results in excessive parameter and memory costs, while the zero-shot strategy leads to sub-optimal performance. To solve the problem, EffOWT is proposed for efficiently transferring VLMs to OWT. Specifically, we build a small and independent learnable side network outside the VLM backbone. By freezing the backbone and only executing backpropagation on the side network, the model's efficiency requirements can be met. In addition, EffOWT enhances the side network by proposing a hybrid structure of Transformer and CNN to improve the model's performance in the OWT field. Finally, we implement sparse interactions on the MLP, thus reducing parameter updates and memory costs significantly. Thanks to the proposed methods, EffOWT achieves an absolute gain of 5.5% on the tracking metric OWTA for unknown categories, while only updating 1.3% of the parameters compared to full fine-tuning, with a 36.4% memory saving. Other metrics also demonstrate obvious improvement. 

**Abstract (ZH)**: Efficient Transfer of Visual Language Models to Open-World Tracking 

---
# RCCFormer: A Robust Crowd Counting Network Based on Transformer 

**Title (ZH)**: RCCFormer：基于 transformer 的鲁棒人流计数网络 

**Authors**: Peng Liu, Heng-Chao Li, Sen Lei, Nanqing Liu, Bin Feng, Xiao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04935)  

**Abstract**: Crowd counting, which is a key computer vision task, has emerged as a fundamental technology in crowd analysis and public safety management. However, challenges such as scale variations and complex backgrounds significantly impact the accuracy of crowd counting. To mitigate these issues, this paper proposes a robust Transformer-based crowd counting network, termed RCCFormer, specifically designed for background suppression and scale awareness. The proposed method incorporates a Multi-level Feature Fusion Module (MFFM), which meticulously integrates features extracted at diverse stages of the backbone architecture. It establishes a strong baseline capable of capturing intricate and comprehensive feature representations, surpassing traditional baselines. Furthermore, the introduced Detail-Embedded Attention Block (DEAB) captures contextual information and local details through global self-attention and local attention along with a learnable manner for efficient fusion. This enhances the model's ability to focus on foreground regions while effectively mitigating background noise interference. Additionally, we develop an Adaptive Scale-Aware Module (ASAM), with our novel Input-dependent Deformable Convolution (IDConv) as its fundamental building block. This module dynamically adapts to changes in head target shapes and scales, significantly improving the network's capability to accommodate large-scale variations. The effectiveness of the proposed method is validated on the ShanghaiTech Part_A and Part_B, NWPU-Crowd, and QNRF datasets. The results demonstrate that our RCCFormer achieves excellent performance across all four datasets, showcasing state-of-the-art outcomes. 

**Abstract (ZH)**: 基于Transformer的鲁棒 crowd counting 网络 RCCFormer：背景抑制与尺度awareness的设计 

---
# Video-Bench: Human-Aligned Video Generation Benchmark 

**Title (ZH)**: Video-Bench: 人体对齐的视频生成基准 

**Authors**: Hui Han, Siyuan Li, Jiaqi Chen, Yiwen Yuan, Yuling Wu, Chak Tou Leong, Hanwen Du, Junchen Fu, Youhua Li, Jie Zhang, Chi Zhang, Li-jia Li, Yongxin Ni  

**Link**: [PDF](https://arxiv.org/pdf/2504.04907)  

**Abstract**: Video generation assessment is essential for ensuring that generative models produce visually realistic, high-quality videos while aligning with human expectations. Current video generation benchmarks fall into two main categories: traditional benchmarks, which use metrics and embeddings to evaluate generated video quality across multiple dimensions but often lack alignment with human judgments; and large language model (LLM)-based benchmarks, though capable of human-like reasoning, are constrained by a limited understanding of video quality metrics and cross-modal consistency. To address these challenges and establish a benchmark that better aligns with human preferences, this paper introduces Video-Bench, a comprehensive benchmark featuring a rich prompt suite and extensive evaluation dimensions. This benchmark represents the first attempt to systematically leverage MLLMs across all dimensions relevant to video generation assessment in generative models. By incorporating few-shot scoring and chain-of-query techniques, Video-Bench provides a structured, scalable approach to generated video evaluation. Experiments on advanced models including Sora demonstrate that Video-Bench achieves superior alignment with human preferences across all dimensions. Moreover, in instances where our framework's assessments diverge from human evaluations, it consistently offers more objective and accurate insights, suggesting an even greater potential advantage over traditional human judgment. 

**Abstract (ZH)**: 视频生成评估对于确保生成模型产生视觉上真实、高质量的视频并与人类期望保持一致至关重要。当前视频生成基准主要分为两类：传统的基准，使用多种度量和嵌入来评估生成视频的质量，但往往缺乏与人类判断的对齐；以及基于大规模语言模型（LLM）的基准，尽管具备类似人类的推理能力，但在理解和视频质量度量以及跨模态一致性方面能力有限。为了应对这些挑战并建立一个更符合人类偏好的基准，本文引入了Video-Bench，一个涵盖丰富提示集和广泛评估维度的综合基准。Video-Bench是首次尝试系统地在所有与视频生成评估相关的维度中利用MLLMs。通过结合少量示例评分和查询链技术，Video-Bench提供了一种结构化、可扩展的生成视频评估方法。实验表明，Video-Bench在所有维度中都实现了与人类偏好的更好对齐。此外，在我们的框架评估与人类评价出现分歧的情况下，它始终提供了更加客观和准确的洞察，表明在传统人类判断之外可能存在更大的优势。 

---
# From Specificity to Generality: Revisiting Generalizable Artifacts in Detecting Face Deepfakes 

**Title (ZH)**: 从具体性到普遍性：重新审视检测人脸深fake的可迁移 artifacts 

**Authors**: Long Ma, Zhiyuan Yan, Yize Chen, Jin Xu, Qinglang Guo, Hu Huang, Yong Liao, Hui Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.04827)  

**Abstract**: Detecting deepfakes has been an increasingly important topic, especially given the rapid development of AI generation techniques. In this paper, we ask: How can we build a universal detection framework that is effective for most facial deepfakes? One significant challenge is the wide variety of deepfake generators available, resulting in varying forgery artifacts (e.g., lighting inconsistency, color mismatch, etc). But should we ``teach" the detector to learn all these artifacts separately? It is impossible and impractical to elaborate on them all. So the core idea is to pinpoint the more common and general artifacts across different deepfakes. Accordingly, we categorize deepfake artifacts into two distinct yet complementary types: Face Inconsistency Artifacts (FIA) and Up-Sampling Artifacts (USA). FIA arise from the challenge of generating all intricate details, inevitably causing inconsistencies between the complex facial features and relatively uniform surrounding areas. USA, on the other hand, are the inevitable traces left by the generator's decoder during the up-sampling process. This categorization stems from the observation that all existing deepfakes typically exhibit one or both of these artifacts. To achieve this, we propose a new data-level pseudo-fake creation framework that constructs fake samples with only the FIA and USA, without introducing extra less-general artifacts. Specifically, we employ a super-resolution to simulate the USA, while design a Blender module that uses image-level self-blending on diverse facial regions to create the FIA. We surprisingly found that, with this intuitive design, a standard image classifier trained only with our pseudo-fake data can non-trivially generalize well to unseen deepfakes. 

**Abstract (ZH)**: 构建一种针对大多数面部深fake通用且有效的检测框架 

---
# Dynamic Vision Mamba 

**Title (ZH)**: 动态视见矛头蛇 

**Authors**: Mengxuan Wu, Zekai Li, Zhiyuan Liang, Moyang Li, Xuanlei Zhao, Samir Khaki, Zheng Zhu, Xiaojiang Peng, Konstantinos N. Plataniotis, Kai Wang, Wangbo Zhao, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2504.04787)  

**Abstract**: Mamba-based vision models have gained extensive attention as a result of being computationally more efficient than attention-based models. However, spatial redundancy still exists in these models, represented by token and block redundancy. For token redundancy, we analytically find that early token pruning methods will result in inconsistency between training and inference or introduce extra computation for inference. Therefore, we customize token pruning to fit the Mamba structure by rearranging the pruned sequence before feeding it into the next Mamba block. For block redundancy, we allow each image to select SSM blocks dynamically based on an empirical observation that the inference speed of Mamba-based vision models is largely affected by the number of SSM blocks. Our proposed method, Dynamic Vision Mamba (DyVM), effectively reduces FLOPs with minor performance drops. We achieve a reduction of 35.2\% FLOPs with only a loss of accuracy of 1.7\% on Vim-S. It also generalizes well across different Mamba vision model architectures and different vision tasks. Our code will be made public. 

**Abstract (ZH)**: 基于Mamba的视觉模型由于具有比基于注意力的模型更高效的计算特性而获得了广泛关注。然而，这些模型仍存在空间冗余，表现为令牌和块冗余。对于令牌冗余，我们分析发现，早期的令牌修剪方法会导致训练和推断之间不一致或在推断过程中增加额外的计算量。因此，我们通过在输送到下一个Mamba块之前重新排列修剪序列，定制了令牌修剪以适应Mamba结构。对于块冗余，我们允许每张图像根据Mamba基于视觉模型的推断速度主要受SSM块数量影响的经验观察，动态选择SSM块。我们提出的方法，动态视觉Mamba（DyVM），能有效地减少FLOPs，同时仅产生轻微的性能下降。我们在Vim-S上实现了35.2%的FLOPs减少，准确率下降了1.7%。该方法还能够在不同Mamba视觉模型架构和不同视觉任务上表现出良好的泛化能力。我们的代码将公开发布。 

---
# Enhancing Leaf Disease Classification Using GAT-GCN Hybrid Model 

**Title (ZH)**: 使用GAT-GCN混合模型增强叶片疾病分类 

**Authors**: Shyam Sundhar, Riya Sharma, Priyansh Maheshwari, Suvidha Rupesh Kumar, T. Sunil Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2504.04764)  

**Abstract**: Agriculture plays a critical role in the global economy, providing livelihoods and ensuring food security for billions. As innovative agricultural practices become more widespread, the risk of crop diseases has increased, highlighting the urgent need for efficient, low-intervention disease identification methods. This research presents a hybrid model combining Graph Attention Networks (GATs) and Graph Convolution Networks (GCNs) for leaf disease classification. GCNs have been widely used for learning from graph-structured data, and GATs enhance this by incorporating attention mechanisms to focus on the most important neighbors. The methodology integrates superpixel segmentation for efficient feature extraction, partitioning images into meaningful, homogeneous regions that better capture localized features. The authors have employed an edge augmentation technique to enhance the robustness of the model. The edge augmentation technique has introduced a significant degree of generalization in the detection capabilities of the model. To further optimize training, weight initialization techniques are applied. The hybrid model is evaluated against the individual performance of the GCN and GAT models and the hybrid model achieved a precision of 0.9822, recall of 0.9818, and F1-score of 0.9818 in apple leaf disease classification, a precision of 0.9746, recall of 0.9744, and F1-score of 0.9743 in potato leaf disease classification, and a precision of 0.8801, recall of 0.8801, and F1-score of 0.8799 in sugarcane leaf disease classification. These results demonstrate the robustness and performance of the model, suggesting its potential to support sustainable agricultural practices through precise and effective disease detection. This work is a small step towards reducing the loss of crops and hence supporting sustainable goals of zero hunger and life on land. 

**Abstract (ZH)**: 农业在全球经济中扮演着关键角色，为 billions 提供生计并确保粮食安全。随着创新农业生产实践的普及，作物病害的风险增加，强调了迫切需要高效、低干预的病害识别方法。本研究提出了一种结合图注意网络（GATs）和图卷积网络（GCNs）的混合模型，用于叶片病害分类。GCNs 广泛用于从结构化数据中学习，而 GATs 通过引入注意机制关注最重要的邻居从而增强了这一过程。该方法整合了超像素分割以实现高效的特征提取，将图像分区为更具意义的、同质的区域，更好地捕捉局部特征。作者采用了边增强技术以提高模型的鲁棒性。边增强技术显著增强了模型检测能力的一般性。为了进一步优化训练，应用了权重初始化技术。该混合模型在苹果叶片病害分类中的精度为 0.9822、召回率为 0.9818、F1 分数为 0.9818；在马铃薯叶片病害分类中的精度为 0.9746、召回率为 0.9744、F1 分数为 0.9743；在甘蔗叶片病害分类中的精度为 0.8801、召回率为 0.8801、F1 分数为 0.8799。这些结果展示了该模型的鲁棒性和性能，表明其在通过精确有效的病害检测支持可持续农业生产方面的潜力。这项工作是朝着减少作物损失和支持零饥饿和陆地生命可持续目标迈出的一个小步骤。 

---
# Bridging Knowledge Gap Between Image Inpainting and Large-Area Visible Watermark Removal 

**Title (ZH)**: 填补图像修复与大面积可见水印移除之间的知识差距 

**Authors**: Yicheng Leng, Chaowei Fang, Junye Chen, Yixiang Fang, Sheng Li, Guanbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04687)  

**Abstract**: Visible watermark removal which involves watermark cleaning and background content restoration is pivotal to evaluate the resilience of watermarks. Existing deep neural network (DNN)-based models still struggle with large-area watermarks and are overly dependent on the quality of watermark mask prediction. To overcome these challenges, we introduce a novel feature adapting framework that leverages the representation modeling capacity of a pre-trained image inpainting model. Our approach bridges the knowledge gap between image inpainting and watermark removal by fusing information of the residual background content beneath watermarks into the inpainting backbone model. We establish a dual-branch system to capture and embed features from the residual background content, which are merged into intermediate features of the inpainting backbone model via gated feature fusion modules. Moreover, for relieving the dependence on high-quality watermark masks, we introduce a new training paradigm by utilizing coarse watermark masks to guide the inference process. This contributes to a visible image removal model which is insensitive to the quality of watermark mask during testing. Extensive experiments on both a large-scale synthesized dataset and a real-world dataset demonstrate that our approach significantly outperforms existing state-of-the-art methods. The source code is available in the supplementary materials. 

**Abstract (ZH)**: 可见水印去除中的水印清理和背景内容恢复对于评估水印的鲁棒性至关重要。现有的基于深度神经网络（DNN）的模型在处理大面积水印时仍存在问题，并且过度依赖水印掩模预测的质量。为克服这些挑战，我们提出了一种新的特征自适应框架，利用预训练图像修复模型的表示建模能力。我们的方法通过融合水印下方残留背景内容的信息，弥合了图像修复与水印去除之间的知识差距。我们建立了一种双分支系统来捕捉和嵌入残留背景内容的特征，并通过门控特征融合模块将这些特征合并到修复骨干模型的中间特征中。此外，为了减轻对高质量水印掩模的依赖，我们提出了一种新的训练范式，利用粗糙的水印掩模来指导推断过程，从而在测试过程中对水印掩模的质量具有鲁棒性。在大规模合成数据集和真实世界数据集上的广泛实验表明，我们的方法显著优于现有最先进的方法。源代码可在附录材料中获取。 

---
# Here Comes the Explanation: A Shapley Perspective on Multi-contrast Medical Image Segmentation 

**Title (ZH)**: Here Comes the Explanation: 从Shapley值视角看多对比医学图像分割 

**Authors**: Tianyi Ren, Juampablo Heras Rivera, Hitender Oswal, Yutong Pan, Agamdeep Chopra, Jacob Ruzevick, Mehmet Kurt  

**Link**: [PDF](https://arxiv.org/pdf/2504.04645)  

**Abstract**: Deep learning has been successfully applied to medical image segmentation, enabling accurate identification of regions of interest such as organs and lesions. This approach works effectively across diverse datasets, including those with single-image contrast, multi-contrast, and multimodal imaging data. To improve human understanding of these black-box models, there is a growing need for Explainable AI (XAI) techniques for model transparency and accountability. Previous research has primarily focused on post hoc pixel-level explanations, using methods gradient-based and perturbation-based apporaches. These methods rely on gradients or perturbations to explain model predictions. However, these pixel-level explanations often struggle with the complexity inherent in multi-contrast magnetic resonance imaging (MRI) segmentation tasks, and the sparsely distributed explanations have limited clinical relevance. In this study, we propose using contrast-level Shapley values to explain state-of-the-art models trained on standard metrics used in brain tumor segmentation. Our results demonstrate that Shapley analysis provides valuable insights into different models' behavior used for tumor segmentation. We demonstrated a bias for U-Net towards over-weighing T1-contrast and FLAIR, while Swin-UNETR provided a cross-contrast understanding with balanced Shapley distribution. 

**Abstract (ZH)**: 深度学习已在医学图像分割中成功应用，能够准确识别如器官和病灶等区域。该方法在单对比度图像、多对比度图像和多模态成像数据等多样化的数据集上均能有效工作。为了提高对这些黑箱模型的人类理解，需要可解释人工智能（XAI）技术以增加模型的透明度和责任感。前期研究主要关注后验像素级解释，使用基于梯度和扰动的方法。然而，这些像素级解释在多对比度磁共振成像（MRI）分割任务的复杂性面前效果有限，且稀疏的解释缺乏临床意义。本研究提出使用对比度级Shapley值来解释基于标准脑肿瘤分割指标训练的最先进模型。研究结果表明，Shapley分析为不同模型在肿瘤分割中的行为提供了有价值的见解。我们发现U-Net倾向于过度重视T1对比度和FLAIR，而Swin-UNETR则提供了跨对比度的理解并具有平衡的Shapley分布。 

---
# Your Image Generator Is Your New Private Dataset 

**Title (ZH)**: 您的图像生成器即是您的新私人数据集 

**Authors**: Nicolo Resmini, Eugenio Lomurno, Cristian Sbrolli, Matteo Matteucci  

**Link**: [PDF](https://arxiv.org/pdf/2504.04582)  

**Abstract**: Generative diffusion models have emerged as powerful tools to synthetically produce training data, offering potential solutions to data scarcity and reducing labelling costs for downstream supervised deep learning applications. However, effectively leveraging text-conditioned image generation for building classifier training sets requires addressing key issues: constructing informative textual prompts, adapting generative models to specific domains, and ensuring robust performance. This paper proposes the Text-Conditioned Knowledge Recycling (TCKR) pipeline to tackle these challenges. TCKR combines dynamic image captioning, parameter-efficient diffusion model fine-tuning, and Generative Knowledge Distillation techniques to create synthetic datasets tailored for image classification. The pipeline is rigorously evaluated on ten diverse image classification benchmarks. The results demonstrate that models trained solely on TCKR-generated data achieve classification accuracies on par with (and in several cases exceeding) models trained on real images. Furthermore, the evaluation reveals that these synthetic-data-trained models exhibit substantially enhanced privacy characteristics: their vulnerability to Membership Inference Attacks is significantly reduced, with the membership inference AUC lowered by 5.49 points on average compared to using real training data, demonstrating a substantial improvement in the performance-privacy trade-off. These findings indicate that high-fidelity synthetic data can effectively replace real data for training classifiers, yielding strong performance whilst simultaneously providing improved privacy protection as a valuable emergent property. The code and trained models are available in the accompanying open-source repository. 

**Abstract (ZH)**: 基于文本条件的生成扩散模型在图像分类训练集构建中的应用：Text-Conditioned Knowledge Recycling (TCKR) 管道的研究 

---
# SnapPix: Efficient-Coding--Inspired In-Sensor Compression for Edge Vision 

**Title (ZH)**: SnapPix：受高效编码启发的边缘视觉传感器内压缩 

**Authors**: Weikai Lin, Tianrui Ma, Adith Boloor, Yu Feng, Ruofan Xing, Xuan Zhang, Yuhao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04535)  

**Abstract**: Energy-efficient image acquisition on the edge is crucial for enabling remote sensing applications where the sensor node has weak compute capabilities and must transmit data to a remote server/cloud for processing. To reduce the edge energy consumption, this paper proposes a sensor-algorithm co-designed system called SnapPix, which compresses raw pixels in the analog domain inside the sensor. We use coded exposure (CE) as the in-sensor compression strategy as it offers the flexibility to sample, i.e., selectively expose pixels, both spatially and temporally. SNAPPIX has three contributions. First, we propose a task-agnostic strategy to learn the sampling/exposure pattern based on the classic theory of efficient coding. Second, we co-design the downstream vision model with the exposure pattern to address the pixel-level non-uniformity unique to CE-compressed images. Finally, we propose lightweight augmentations to the image sensor hardware to support our in-sensor CE compression. Evaluating on action recognition and video reconstruction, SnapPix outperforms state-of-the-art video-based methods at the same speed while reducing the energy by up to 15.4x. We have open-sourced the code at: this https URL. 

**Abstract (ZH)**: 边缘节点上的能效图像获取对于传感器节点计算能力较弱且必须将数据传输到远程服务器/云端进行处理的远程 sensing 应用至关重要。为了减少边缘节点的能耗，本文提出了一种名为 SnapPix 的传感器-算法协同设计系统，在传感器内部的模拟域对原始像素进行压缩。我们使用编码曝光（CE）作为内部压缩策略，因为它可以在空间和时间上灵活地采样，即选择性地曝光像素。SnapPix 有三个贡献。首先，我们提出了一种任务无关的策略，基于高效的编码经典理论来学习采样/曝光模式。其次，我们与曝光模式共同设计下游的视觉模型，以解决 CE 压缩图像特有的像素级非均匀性问题。最后，我们提出了轻量级的图像传感器硬件增强措施，以支持我们的内部 CE 压缩。在动作识别和视频重建的评估中，SnapPix 在相同速度下优于最先进的基于视频的方法，能耗最多可降低 15.4 倍。我们已在以下链接开源了代码：this https URL。 

---
# Enhance Then Search: An Augmentation-Search Strategy with Foundation Models for Cross-Domain Few-Shot Object Detection 

**Title (ZH)**: 增强后再搜索：基于基础模型的跨域少样本对象检测增广-搜索策略 

**Authors**: Jiancheng Pan, Yanxing Liu, Xiao He, Long Peng, Jiahao Li, Yuze Sun, Xiaomeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04517)  

**Abstract**: Foundation models pretrained on extensive datasets, such as GroundingDINO and LAE-DINO, have performed remarkably in the cross-domain few-shot object detection (CD-FSOD) task. Through rigorous few-shot training, we found that the integration of image-based data augmentation techniques and grid-based sub-domain search strategy significantly enhances the performance of these foundation models. Building upon GroundingDINO, we employed several widely used image augmentation methods and established optimization objectives to effectively navigate the expansive domain space in search of optimal sub-domains. This approach facilitates efficient few-shot object detection and introduces an approach to solving the CD-FSOD problem by efficiently searching for the optimal parameter configuration from the foundation model. Our findings substantially advance the practical deployment of vision-language models in data-scarce environments, offering critical insights into optimizing their cross-domain generalization capabilities without labor-intensive retraining. Code is available at this https URL. 

**Abstract (ZH)**: 基础模型在大规模数据集上预训练，如GroundingDINO和LAE-DINO，在跨域少样本目标检测（CD-FSOD）任务中表现卓越。通过严格的少样本训练，我们发现基于图像的数据增强技术与基于网格的子域搜索策略的结合显著提升了这些基础模型的性能。基于GroundingDINO，我们采用了多种常用的图像增强方法并建立了优化目标，以有效探索广阔的数据域空间，寻找最优子域。该方法促进了高效的少样本目标检测，并提出了通过高效搜索基础模型的最佳参数配置来解决CD-FSOD问题的方法。我们的研究显著推动了在数据稀缺环境中视觉-语言模型的实际部署，并提供了关于优化其跨域泛化能力的重要洞见，而无需进行劳动密集型的重新训练。代码可在此处访问：this https URL。 

---
# Statistical Guarantees Of False Discovery Rate In Medical Instance Segmentation Tasks Based on Conformal Risk Control 

**Title (ZH)**: 基于形验风险控制的医疗Instance分割任务中错误发现率的统计保证 

**Authors**: Mengxia Dai, Wenqian Luo, Tianyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04482)  

**Abstract**: Instance segmentation plays a pivotal role in medical image analysis by enabling precise localization and delineation of lesions, tumors, and anatomical structures. Although deep learning models such as Mask R-CNN and BlendMask have achieved remarkable progress, their application in high-risk medical scenarios remains constrained by confidence calibration issues, which may lead to misdiagnosis. To address this challenge, we propose a robust quality control framework based on conformal prediction theory. This framework innovatively constructs a risk-aware dynamic threshold mechanism that adaptively adjusts segmentation decision boundaries according to clinical this http URL, we design a \textbf{calibration-aware loss function} that dynamically tunes the segmentation threshold based on a user-defined risk level $\alpha$. Utilizing exchangeable calibration data, this method ensures that the expected FNR or FDR on test data remains below $\alpha$ with high probability. The framework maintains compatibility with mainstream segmentation models (e.g., Mask R-CNN, BlendMask+ResNet-50-FPN) and datasets (PASCAL VOC format) without requiring architectural modifications. Empirical results demonstrate that we rigorously bound the FDR metric marginally over the test set via our developed calibration framework. 

**Abstract (ZH)**: 实例分割在医疗图像分析中发挥着关键作用，通过实现病变、肿瘤和解剖结构的精确定位和勾勒。尽管诸如Mask R-CNN和BlendMask等深度学习模型取得了显著进展，但在高风险医疗场景中的应用仍受限于置信度校准问题，这可能导致误诊。为应对这一挑战，我们提出了一种基于可构造预测理论的稳健质量控制框架。该框架创新性地构建了一种风险感知动态阈值机制，根据临床需求自适应调整分割决策边界。为此，我们设计了一种**置信度感知损失函数**，根据用户定义的风险水平$\alpha$动态调整分割阈值。利用可交换的校准数据，该方法确保在测试数据上的预期FNR或FDR低于$\alpha$的概率很高。该框架与主流的分割模型（如Mask R-CNN、BlendMask+ResNet-50-FPN）及数据集（PASCAL VOC格式）保持兼容，无需对架构进行修改。实验证明，我们通过开发的校准框架严格界定了测试集上的FDR指标。 

---
# EclipseNETs: Learning Irregular Small Celestial Body Silhouettes 

**Title (ZH)**: EclipseNETs：学习不规则小型天体轮廓 

**Authors**: Giacomo Acciarini, Dario Izzo, Francesco Biscani  

**Link**: [PDF](https://arxiv.org/pdf/2504.04455)  

**Abstract**: Accurately predicting eclipse events around irregular small bodies is crucial for spacecraft navigation, orbit determination, and spacecraft systems management. This paper introduces a novel approach leveraging neural implicit representations to model eclipse conditions efficiently and reliably. We propose neural network architectures that capture the complex silhouettes of asteroids and comets with high precision. Tested on four well-characterized bodies - Bennu, Itokawa, 67P/Churyumov-Gerasimenko, and Eros - our method achieves accuracy comparable to traditional ray-tracing techniques while offering orders of magnitude faster performance. Additionally, we develop an indirect learning framework that trains these models directly from sparse trajectory data using Neural Ordinary Differential Equations, removing the requirement to have prior knowledge of an accurate shape model. This approach allows for the continuous refinement of eclipse predictions, progressively reducing errors and improving accuracy as new trajectory data is incorporated. 

**Abstract (ZH)**: 准确预测不规则小体附近的日食事件对于 spacecraft 导航、轨道确定以及 spacecraft 系统管理至关重要。本文介绍了一种利用神经隐式表示的新方法，以高效可靠地建模日食条件。我们提出了能够以高精度捕捉小行星和彗星复杂剪影的神经网络架构。在对班努小行星、伊托卡瓦小行星、67P/丘留莫夫-格拉西缅科彗星和爱神星这四个已充分-characterized的天体进行测试后，我们的方法在准确度上达到了与传统射线跟踪技术相当的水平，同时提供了数量级更优的性能。此外，我们开发了一种间接学习框架，通过使用神经常微分方程直接从稀疏轨迹数据训练这些模型，无需事先具备精确形状模型的知识。该方法允许对日食预测进行持续优化，随着新轨迹数据的加入，逐步减少误差并提高准确度。 

---
# FluentLip: A Phonemes-Based Two-stage Approach for Audio-Driven Lip Synthesis with Optical Flow Consistency 

**Title (ZH)**: FluentLip: 基于音素的两阶段音频驱动唇型合成方法，具有光流一致性 

**Authors**: Shiyan Liu, Rui Qu, Yan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.04427)  

**Abstract**: Generating consecutive images of lip movements that align with a given speech in audio-driven lip synthesis is a challenging task. While previous studies have made strides in synchronization and visual quality, lip intelligibility and video fluency remain persistent challenges. This work proposes FluentLip, a two-stage approach for audio-driven lip synthesis, incorporating three featured strategies. To improve lip synchronization and intelligibility, we integrate a phoneme extractor and encoder to generate a fusion of audio and phoneme information for multimodal learning. Additionally, we employ optical flow consistency loss to ensure natural transitions between image frames. Furthermore, we incorporate a diffusion chain during the training of Generative Adversarial Networks (GANs) to improve both stability and efficiency. We evaluate our proposed FluentLip through extensive experiments, comparing it with five state-of-the-art (SOTA) approaches across five metrics, including a proposed metric called Phoneme Error Rate (PER) that evaluates lip pose intelligibility and video fluency. The experimental results demonstrate that our FluentLip approach is highly competitive, achieving significant improvements in smoothness and naturalness. In particular, it outperforms these SOTA approaches by approximately $\textbf{16.3%}$ in Fréchet Inception Distance (FID) and $\textbf{35.2%}$ in PER. 

**Abstract (ZH)**: 基于音频驱动的唇动合成生成与给定语音连续对齐的图像，并保持唇部可读性和视频流畅性是一项具有挑战性的任务。尽管先前的研究在同步性和视觉质量方面取得了进展，但唇部可读性和视频流畅性依然存在持续的挑战。本文提出了一种名为FluentLip的两阶段方法，结合了三种特色策略。为了提高唇部同步性和可读性，我们集成了一个音素提取器和编码器，生成音素信息和音频信息的融合，进行多模态学习。此外，我们利用光学流一致性损失来确保图像帧间的自然过渡。进一步地，在生成对抗网络（GAN）的训练中引入了扩散链，以提高稳定性和效率。我们通过广泛的实验评估了提出的FluentLip方法，将其与五个最先进的（SOTA）方法在五个指标上进行了比较，包括一个新的名为音素错误率（PER）的指标，该指标评估了唇形姿态的可读性和视频流畅性。实验结果表明，我们的FluentLip方法具有很强的竞争性，明显提高了平滑度和自然度。特别是在弗雷谢特入胜距离（FID）和PER上，分别优于这些SOTA方法约16.3%和35.2%。 

---
# Progressive Multi-Source Domain Adaptation for Personalized Facial Expression Recognition 

**Title (ZH)**: 渐进多源域适应的个性化面部表情识别 

**Authors**: Muhammad Osama Zeeshan, Marco Pedersoli, Alessandro Lameiras Koerich, Eric Grange  

**Link**: [PDF](https://arxiv.org/pdf/2504.04252)  

**Abstract**: Personalized facial expression recognition (FER) involves adapting a machine learning model using samples from labeled sources and unlabeled target domains. Given the challenges of recognizing subtle expressions with considerable interpersonal variability, state-of-the-art unsupervised domain adaptation (UDA) methods focus on the multi-source UDA (MSDA) setting, where each domain corresponds to a specific subject, and improve model accuracy and robustness. However, when adapting to a specific target, the diverse nature of multiple source domains translates to a large shift between source and target data. State-of-the-art MSDA methods for FER address this domain shift by considering all the sources to adapt to the target representations. Nevertheless, adapting to a target subject presents significant challenges due to large distributional differences between source and target domains, often resulting in negative transfer. In addition, integrating all sources simultaneously increases computational costs and causes misalignment with the target. To address these issues, we propose a progressive MSDA approach that gradually introduces information from subjects based on their similarity to the target subject. This will ensure that only the most relevant sources from the target are selected, which helps avoid the negative transfer caused by dissimilar sources. We first exploit the closest sources to reduce the distribution shift with the target and then move towards the furthest while only considering the most relevant sources based on the predetermined threshold. Furthermore, to mitigate catastrophic forgetting caused by the incremental introduction of source subjects, we implemented a density-based memory mechanism that preserves the most relevant historical source samples for adaptation. Our experiments show the effectiveness of our proposed method on pain datasets: Biovid and UNBC-McMaster. 

**Abstract (ZH)**: 个性化面部表情识别（FER）涉及使用标记源和未标记目标域的样本调整机器学习模型。鉴于识别细微表情存在显著个体差异的挑战，最新的无监督域适应（UDA）方法集中在多源UDA（MSDA）设置上，每个域对应特定的个体，并提高模型准确性和鲁棒性。然而，在适应特定目标时，多个源域的多样性会导致源域和目标域之间出现较大的数据差异。最先进的FER MSDA方法通过考虑所有源域以适应目标表示来解决这一域差异问题。尽管如此，适应特定目标主体因源域和目标域之间巨大的分布差异而面临重大挑战，往往导致负迁移。此外，同时集成所有源域会增加计算成本并导致与目标主体的对齐偏差。为了应对这些问题，我们提出了一种渐进式的MSDA方法，根据其与目标主体的相似性逐步引入主体信息。这将确保仅选择与目标最相关的源，从而避免因不相似的源导致的负迁移。我们首先利用最近的源域减少与目标之间的分布差异，然后逐步向最远的源域推进，同时只考虑基于预设阈值的最相关源域。此外，为了缓解逐步引入源主体引起的灾难性遗忘，我们实现了一种基于密度的记忆机制，保留最相关的历史源样本以进行适应。我们的实验表明，该方法在疼痛数据集Biovid和UNBC-McMaster上具有有效性。 

---
# Multi-identity Human Image Animation with Structural Video Diffusion 

**Title (ZH)**: 多身份人体图像动画生成中的结构视频扩散 

**Authors**: Zhenzhi Wang, Yixuan Li, Yanhong Zeng, Yuwei Guo, Dahua Lin, Tianfan Xue, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2504.04126)  

**Abstract**: Generating human videos from a single image while ensuring high visual quality and precise control is a challenging task, especially in complex scenarios involving multiple individuals and interactions with objects. Existing methods, while effective for single-human cases, often fail to handle the intricacies of multi-identity interactions because they struggle to associate the correct pairs of human appearance and pose condition and model the distribution of 3D-aware dynamics. To address these limitations, we present Structural Video Diffusion, a novel framework designed for generating realistic multi-human videos. Our approach introduces two core innovations: identity-specific embeddings to maintain consistent appearances across individuals and a structural learning mechanism that incorporates depth and surface-normal cues to model human-object interactions. Additionally, we expand existing human video dataset with 25K new videos featuring diverse multi-human and object interaction scenarios, providing a robust foundation for training. Experimental results demonstrate that Structural Video Diffusion achieves superior performance in generating lifelike, coherent videos for multiple subjects with dynamic and rich interactions, advancing the state of human-centric video generation. 

**Abstract (ZH)**: 从单张图像生成高质量、精确控制的多个人体视频是一项具有挑战性的任务，尤其是在涉及多人和物体交互的复杂场景中。现有方法虽然在单人体案例中有效，但在处理多身份间的复杂交互时往往失效，因为它们难以正确关联人体外观和姿态条件，并建模3D动态分布。为解决这些限制，我们提出了一种新型框架——结构化视频扩散（Structural Video Diffusion），专门用于生成逼真的人体视频。该方法引入了两项核心创新：身份特定嵌入以保持不同个体的外观一致，并结合深度和法线线索，引入结构学习机制以建模人体与物体的交互。此外，我们扩展了现有的人体视频数据集，新增了25K个包含多样化多人和物体交互场景的新视频，为训练提供了坚实的基础。实验结果表明，结构化视频扩散在生成具备动态和丰富交互的多主体逼真、连贯视频方面表现出优越性能，推动了以人体为中心的视频生成技术的发展。 

---
# DocSAM: Unified Document Image Segmentation via Query Decomposition and Heterogeneous Mixed Learning 

**Title (ZH)**: DocSAM：基于查询分解和异构混合学习的统一文档图像分割 

**Authors**: Xiao-Hui Li, Fei Yin, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04085)  

**Abstract**: Document image segmentation is crucial for document analysis and recognition but remains challenging due to the diversity of document formats and segmentation tasks. Existing methods often address these tasks separately, resulting in limited generalization and resource wastage. This paper introduces DocSAM, a transformer-based unified framework designed for various document image segmentation tasks, such as document layout analysis, multi-granularity text segmentation, and table structure recognition, by modelling these tasks as a combination of instance and semantic segmentation. Specifically, DocSAM employs Sentence-BERT to map category names from each dataset into semantic queries that match the dimensionality of instance queries. These two sets of queries interact through an attention mechanism and are cross-attended with image features to predict instance and semantic segmentation masks. Instance categories are predicted by computing the dot product between instance and semantic queries, followed by softmax normalization of scores. Consequently, DocSAM can be jointly trained on heterogeneous datasets, enhancing robustness and generalization while reducing computational and storage resources. Comprehensive evaluations show that DocSAM surpasses existing methods in accuracy, efficiency, and adaptability, highlighting its potential for advancing document image understanding and segmentation across various applications. Codes are available at this https URL. 

**Abstract (ZH)**: 文档图像分割对于文档分析和识别至关重要，但由于文档格式和分割任务的多样性，这仍然具有挑战性。现有方法通常针对这些任务分别处理，导致泛化能力有限和资源浪费。本文介绍了DocSAM，这是一种基于变压器的统一框架，旨在通过将这些任务建模为实例分割和语义分割的组合，用于各种文档图像分割任务，如文档布局分析、多粒度文本分割和表格结构识别。具体而言，DocSAM 使用 Sentence-BERT 将每个数据集的类别名称映射为语义查询，使其与实例查询的维度相匹配。这两组查询通过注意机制相互作用，并与图像特征进行交叉注意，以预测实例和语义分割掩码。实例类别通过计算实例查询和语义查询之间的点积并应用softmax归一化来预测。因此，DocSAM 可以在异构数据集上联合训练，增强稳健性和泛化能力，同时减少计算和存储资源。全面的评估表明，DocSAM 在准确性、效率和适应性方面超过了现有方法，突显了其在各种应用中推动文档图像理解和分割的潜力。代码可在以下网址获取：这个 https URL。 

---
# A Survey of Pathology Foundation Model: Progress and Future Directions 

**Title (ZH)**: 病理学基础模型综述：进展与未来方向 

**Authors**: Conghao Xiong, Hao Chen, Joseph J. Y. Sung  

**Link**: [PDF](https://arxiv.org/pdf/2504.04045)  

**Abstract**: Computational pathology, analyzing whole slide images for automated cancer diagnosis, relies on the multiple instance learning framework where performance heavily depends on the feature extractor and aggregator. Recent Pathology Foundation Models (PFMs), pretrained on large-scale histopathology data, have significantly enhanced capabilities of extractors and aggregators but lack systematic analysis frameworks. This survey presents a hierarchical taxonomy organizing PFMs through a top-down philosophy that can be utilized to analyze FMs in any domain: model scope, model pretraining, and model design. Additionally, we systematically categorize PFM evaluation tasks into slide-level, patch-level, multimodal, and biological tasks, providing comprehensive benchmarking criteria. Our analysis identifies critical challenges in both PFM development (pathology-specific methodology, end-to-end pretraining, data-model scalability) and utilization (effective adaptation, model maintenance), paving the way for future directions in this promising field. Resources referenced in this survey are available at this https URL. 

**Abstract (ZH)**: 计算病理学，基于整个切片图像的自动化癌症诊断，依赖于多实例学习框架，其性能高度依赖于特征提取器和聚合器。大规模组织病理学数据预训练的近期病理学基础模型（PFMs）显著增强了提取器和聚合器的能力，但缺乏系统的分析框架。本文综述提出了一种自上而下的层级分类体系，用于组织PFMs，并可应用于任何领域：模型范围、模型预训练和模型设计。此外，我们系统地将PFM评估任务分类为切片级、 patch级、多模态和生物任务，并提供了全面的基准评估标准。我们的分析指出了PFM开发和利用中关键的挑战（特定于病理的方法学、端到端预训练、数据-模型可扩展性）和利用中的挑战（有效适应、模型维护），为这一有前景领域的未来方向指明了道路。文中引用的资源可访问此链接：此 https URL。 

---
# Simultaneous Motion And Noise Estimation with Event Cameras 

**Title (ZH)**: 事件相机中同时运动与噪声估计 

**Authors**: Shintaro Shiba, Yoshimitsu Aoki, Guillermo Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2504.04029)  

**Abstract**: Event cameras are emerging vision sensors, whose noise is challenging to characterize. Existing denoising methods for event cameras consider other tasks such as motion estimation separately (i.e., sequentially after denoising). However, motion is an intrinsic part of event data, since scene edges cannot be sensed without motion. This work proposes, to the best of our knowledge, the first method that simultaneously estimates motion in its various forms (e.g., ego-motion, optical flow) and noise. The method is flexible, as it allows replacing the 1-step motion estimation of the widely-used Contrast Maximization framework with any other motion estimator, such as deep neural networks. The experiments show that the proposed method achieves state-of-the-art results on the E-MLB denoising benchmark and competitive results on the DND21 benchmark, while showing its efficacy on motion estimation and intensity reconstruction tasks. We believe that the proposed approach contributes to strengthening the theory of event-data denoising, as well as impacting practical denoising use-cases, as we release the code upon acceptance. Project page: this https URL 

**Abstract (ZH)**: 事件相机是新兴的视觉传感器，其噪声难以表征。现有事件相机去噪方法在去噪后才考虑其他任务（如运动估计）。然而，运动是事件数据的一个内在部分，因为不能在没有运动的情况下感知场景边缘。本文提出了一种，据我们所知，第一个能够在其各种形式（如自我运动、光学流）中同时估计运动和噪声的方法。该方法具有灵活性，允许用任何其他运动估计器（如深度神经网络）代替广泛应用的对比最大化框架中的单步运动估计。实验结果显示，所提出的方法在E-MLB去噪基准上达到最先进的效果，在DND21基准上表现竞争力，并在运动估计和强度重建任务中展示了其有效性。我们认为，所提出的方法不仅加强了事件数据去噪的理论，而且还影响了实际的去噪应用，因为接受后我们将发布代码。项目页面：这个 https URL 

---
# Edge Approximation Text Detector 

**Title (ZH)**: 边缘近似文本检测器 

**Authors**: Chuang Yang, Xu Han, Tao Han, Han Han, Bingxuan Zhao, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04001)  

**Abstract**: Pursuing efficient text shape representations helps scene text detection models focus on compact foreground regions and optimize the contour reconstruction steps to simplify the whole detection pipeline. Current approaches either represent irregular shapes via box-to-polygon strategy or decomposing a contour into pieces for fitting gradually, the deficiency of coarse contours or complex pipelines always exists in these models. Considering the above issues, we introduce EdgeText to fit text contours compactly while alleviating excessive contour rebuilding processes. Concretely, it is observed that the two long edges of texts can be regarded as smooth curves. It allows us to build contours via continuous and smooth edges that cover text regions tightly instead of fitting piecewise, which helps avoid the two limitations in current models. Inspired by this observation, EdgeText formulates the text representation as the edge approximation problem via parameterized curve fitting functions. In the inference stage, our model starts with locating text centers, and then creating curve functions for approximating text edges relying on the points. Meanwhile, truncation points are determined based on the location features. In the end, extracting curve segments from curve functions by using the pixel coordinate information brought by truncation points to reconstruct text contours. Furthermore, considering the deep dependency of EdgeText on text edges, a bilateral enhanced perception (BEP) module is designed. It encourages our model to pay attention to the recognition of edge features. Additionally, to accelerate the learning of the curve function parameters, we introduce a proportional integral loss (PI-loss) to force the proposed model to focus on the curve distribution and avoid being disturbed by text scales. 

**Abstract (ZH)**: 追求高效的文本形状表示有助于场景文本检测模型集中在紧凑的前景区域并优化轮廓重构步骤，简化整个检测管道。当前的方法要么通过盒状到多边形的策略表示不规则形状，要么逐步分解轮廓为片段进行拟合，这些模型中粗略的轮廓或复杂的管道始终存在缺陷。考虑到上述问题，我们介绍了EdgeText以紧致地拟合文本轮廓，同时减轻过多的轮廓重建过程。具体而言，观察到文本的两条长边可以被视为平滑曲线。这使我们能够通过连续和平滑的边缘构建紧密覆盖文本区域的轮廓，而不是分段拟合，从而避免当前模型中的两个局限性。受此观察的启发，EdgeText将文本表示形式化为参数化曲线拟合函数的边缘近似问题。在推理阶段，我们的模型首先定位文本中心，然后基于点创建曲线函数以逼近文本边缘，并根据位置特征确定截断点。最后，通过截断点带来的像素坐标信息从曲线函数中提取曲线段，重建文本轮廓。此外，考虑到EdgeText对文本边缘的深度依赖性，我们设计了一个双边增强感知（BEP）模块，以促使模型关注边缘特征的识别。另外，为了加速曲线函数参数的学习，我们引入了比例积分损失（PI-loss），以迫使提出模型关注曲线分布，避免被文本尺度干扰。 

---
# VideoComp: Advancing Fine-Grained Compositional and Temporal Alignment in Video-Text Models 

**Title (ZH)**: VideoComp: 促进视频-文本模型中的细粒度组成和时间对齐advance 

**Authors**: Dahun Kim, AJ Piergiovanni, Ganesh Mallya, Anelia Angelova  

**Link**: [PDF](https://arxiv.org/pdf/2504.03970)  

**Abstract**: We introduce VideoComp, a benchmark and learning framework for advancing video-text compositionality understanding, aimed at improving vision-language models (VLMs) in fine-grained temporal alignment. Unlike existing benchmarks focused on static image-text compositionality or isolated single-event videos, our benchmark targets alignment in continuous multi-event videos. Leveraging video-text datasets with temporally localized event captions (e.g. ActivityNet-Captions, YouCook2), we construct two compositional benchmarks, ActivityNet-Comp and YouCook2-Comp. We create challenging negative samples with subtle temporal disruptions such as reordering, action word replacement, partial captioning, and combined disruptions. These benchmarks comprehensively test models' compositional sensitivity across extended, cohesive video-text sequences. To improve model performance, we propose a hierarchical pairwise preference loss that strengthens alignment with temporally accurate pairs and gradually penalizes increasingly disrupted ones, encouraging fine-grained compositional learning. To mitigate the limited availability of densely annotated video data, we introduce a pretraining strategy that concatenates short video-caption pairs to simulate multi-event sequences. We evaluate video-text foundational models and large multimodal models (LMMs) on our benchmark, identifying both strengths and areas for improvement in compositionality. Overall, our work provides a comprehensive framework for evaluating and enhancing model capabilities in achieving fine-grained, temporally coherent video-text alignment. 

**Abstract (ZH)**: VideoComp：用于提升细粒度时空对齐的视频-文本组成性理解基准及学习框架 

---
# TGraphX: Tensor-Aware Graph Neural Network for Multi-Dimensional Feature Learning 

**Title (ZH)**: TGraphX：张量感知图神经网络多维特征学习 

**Authors**: Arash Sajjadi, Mark Eramian  

**Link**: [PDF](https://arxiv.org/pdf/2504.03953)  

**Abstract**: TGraphX presents a novel paradigm in deep learning by unifying convolutional neural networks (CNNs) with graph neural networks (GNNs) to enhance visual reasoning tasks. Traditional CNNs excel at extracting rich spatial features from images but lack the inherent capability to model inter-object relationships. Conversely, conventional GNNs typically rely on flattened node features, thereby discarding vital spatial details. TGraphX overcomes these limitations by employing CNNs to generate multi-dimensional node features (e.g., (3*128*128) tensors) that preserve local spatial semantics. These spatially aware nodes participate in a graph where message passing is performed using 1*1 convolutions, which fuse adjacent features while maintaining their structure. Furthermore, a deep CNN aggregator with residual connections is used to robustly refine the fused messages, ensuring stable gradient flow and end-to-end trainability. Our approach not only bridges the gap between spatial feature extraction and relational reasoning but also demonstrates significant improvements in object detection refinement and ensemble reasoning. 

**Abstract (ZH)**: TGraphX通过将卷积神经网络（CNNs）与图神经网络（GNNs）统一起来，以增强视觉推理任务，在深度学习中提出了一种新颖的范式。 

---
# Detection Limits and Statistical Separability of Tree Ring Watermarks in Rectified Flow-based Text-to-Image Generation Models 

**Title (ZH)**: 校准流基于文本生成图像模型中树轮水印的检测限和统计可分辨性 

**Authors**: Ved Umrajkar, Aakash Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2504.03850)  

**Abstract**: Tree-Ring Watermarking is a significant technique for authenticating AI-generated images. However, its effectiveness in rectified flow-based models remains unexplored, particularly given the inherent challenges of these models with noise latent inversion. Through extensive experimentation, we evaluated and compared the detection and separability of watermarks between SD 2.1 and FLUX.1-dev models. By analyzing various text guidance configurations and augmentation attacks, we demonstrate how inversion limitations affect both watermark recovery and the statistical separation between watermarked and unwatermarked images. Our findings provide valuable insights into the current limitations of Tree-Ring Watermarking in the current SOTA models and highlight the critical need for improved inversion methods to achieve reliable watermark detection and separability. The official implementation, dataset release and all experimental results are available at this \href{this https URL}{\textbf{link}}. 

**Abstract (ZH)**: 树轮水印技术是认证AI生成图像的重要方法。然而，该技术在纠正的基于流的模型中的有效性尚未被探索，尤其是考虑到这些模型中存在的固有噪声反向转换难题。通过广泛的实验，我们评估并比较了SD 2.1和FLUX.1-dev模型之间水印的检测能力和可分离性。通过对各种文本引导配置和增强攻击的分析，我们展示了反向转换限制如何影响水印恢复以及带水印和不带水印图像之间的统计分离。我们的研究结果为当前Tree-Ring水印技术在当前SOTA模型中的局限性提供了有价值的洞见，并强调了改进反向转换方法对于实现可靠的水印检测和可分离性至关重要。官方实现、数据集发布和所有实验结果可在以下链接获取：\href{this https URL}{\textbf{link}}。 

---
# PointSplit: Towards On-device 3D Object Detection with Heterogeneous Low-power Accelerators 

**Title (ZH)**: PointSplit: 向着使用异构低功耗加速器的设备端3D物体检测 

**Authors**: Keondo Park, You Rim Choi, Inhoe Lee, Hyung-Sin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.03654)  

**Abstract**: Running deep learning models on resource-constrained edge devices has drawn significant attention due to its fast response, privacy preservation, and robust operation regardless of Internet connectivity. While these devices already cope with various intelligent tasks, the latest edge devices that are equipped with multiple types of low-power accelerators (i.e., both mobile GPU and NPU) can bring another opportunity; a task that used to be too heavy for an edge device in the single-accelerator world might become viable in the upcoming heterogeneous-accelerator this http URL realize the potential in the context of 3D object detection, we identify several technical challenges and propose PointSplit, a novel 3D object detection framework for multi-accelerator edge devices that addresses the problems. Specifically, our PointSplit design includes (1) 2D semantics-aware biased point sampling, (2) parallelized 3D feature extraction, and (3) role-based group-wise quantization. We implement PointSplit on TensorFlow Lite and evaluate it on a customized hardware platform comprising both mobile GPU and EdgeTPU. Experimental results on representative RGB-D datasets, SUN RGB-D and Scannet V2, demonstrate that PointSplit on a multi-accelerator device is 24.7 times faster with similar accuracy compared to the full-precision, 2D-3D fusion-based 3D detector on a GPU-only device. 

**Abstract (ZH)**: 基于多加速器边缘设备的点分割三维目标检测方法：利用低功耗移动GPU和NPU实现快速精准的三维目标检测 

---
