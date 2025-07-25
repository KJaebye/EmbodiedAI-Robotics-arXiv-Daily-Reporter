# G2S-ICP SLAM: Geometry-aware Gaussian Splatting ICP SLAM 

**Title (ZH)**: G2S-ICP SLAM：几何aware的高斯点云ICP SLAM 

**Authors**: Gyuhyeon Pak, Hae Min Cho, Euntai Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.18344)  

**Abstract**: In this paper, we present a novel geometry-aware RGB-D Gaussian Splatting SLAM system, named G2S-ICP SLAM. The proposed method performs high-fidelity 3D reconstruction and robust camera pose tracking in real-time by representing each scene element using a Gaussian distribution constrained to the local tangent plane. This effectively models the local surface as a 2D Gaussian disk aligned with the underlying geometry, leading to more consistent depth interpretation across multiple viewpoints compared to conventional 3D ellipsoid-based representations with isotropic uncertainty. To integrate this representation into the SLAM pipeline, we embed the surface-aligned Gaussian disks into a Generalized ICP framework by introducing anisotropic covariance prior without altering the underlying registration formulation. Furthermore we propose a geometry-aware loss that supervises photometric, depth, and normal consistency. Our system achieves real-time operation while preserving both visual and geometric fidelity. Extensive experiments on the Replica and TUM-RGBD datasets demonstrate that G2S-ICP SLAM outperforms prior SLAM systems in terms of localization accuracy, reconstruction completeness, while maintaining the rendering quality. 

**Abstract (ZH)**: 基于几何感知的RGB-D高斯点云SLAM系统：G2S-ICP SLAM 

---
# Evaluation of facial landmark localization performance in a surgical setting 

**Title (ZH)**: 手术环境下面部特征点定位性能评估 

**Authors**: Ines Frajtag, Marko Švaco, Filip Šuligoj  

**Link**: [PDF](https://arxiv.org/pdf/2507.18248)  

**Abstract**: The use of robotics, computer vision, and their applications is becoming increasingly widespread in various fields, including medicine. Many face detection algorithms have found applications in neurosurgery, ophthalmology, and plastic surgery. A common challenge in using these algorithms is variable lighting conditions and the flexibility of detection positions to identify and precisely localize patients. The proposed experiment tests the MediaPipe algorithm for detecting facial landmarks in a controlled setting, using a robotic arm that automatically adjusts positions while the surgical light and the phantom remain in a fixed position. The results of this study demonstrate that the improved accuracy of facial landmark detection under surgical lighting significantly enhances the detection performance at larger yaw and pitch angles. The increase in standard deviation/dispersion occurs due to imprecise detection of selected facial landmarks. This analysis allows for a discussion on the potential integration of the MediaPipe algorithm into medical procedures. 

**Abstract (ZH)**: 机器人、计算机视觉及其在各领域的应用，包括医学领域日益普及。许多面部检测算法在神经外科、眼科和整形外科中得到了應用。使用这些算法时的一个常见挑战是在不同光照条件下和检测位置的灵活性，以识别和精确定位患者。本实验旨在在受控环境中测试MediaPipe算法在面部特征点检测中的性能，利用机械臂自动调整位置，同时保持手术灯光和phantom（模拟人体器官的模型）固定。研究结果表明，手术照明下面部特征点检测精度的提升，显著改善了在较大偏航和俯仰角度下的检测性能。标准偏差/分散度的增加是由于选定面部特征点检测不够精确。这一分析为进一步探讨将MediaPipe算法集成到医疗程序中提供可能。 

---
# Autonomous UAV Navigation for Search and Rescue Missions Using Computer Vision and Convolutional Neural Networks 

**Title (ZH)**: 使用计算机视觉和卷积神经网络的自主无人机搜救导航 

**Authors**: Luka Šiktar, Branimir Ćaran, Bojan Šekoranja, Marko Švaco  

**Link**: [PDF](https://arxiv.org/pdf/2507.18160)  

**Abstract**: In this paper, we present a subsystem, using Unmanned Aerial Vehicles (UAV), for search and rescue missions, focusing on people detection, face recognition and tracking of identified individuals. The proposed solution integrates a UAV with ROS2 framework, that utilizes multiple convolutional neural networks (CNN) for search missions. System identification and PD controller deployment are performed for autonomous UAV navigation. The ROS2 environment utilizes the YOLOv11 and YOLOv11-pose CNNs for tracking purposes, and the dlib library CNN for face recognition. The system detects a specific individual, performs face recognition and starts tracking. If the individual is not yet known, the UAV operator can manually locate the person, save their facial image and immediately initiate the tracking process. The tracking process relies on specific keypoints identified on the human body using the YOLOv11-pose CNN model. These keypoints are used to track a specific individual and maintain a safe distance. To enhance accurate tracking, system identification is performed, based on measurement data from the UAVs IMU. The identified system parameters are used to design PD controllers that utilize YOLOv11-pose to estimate the distance between the UAVs camera and the identified individual. The initial experiments, conducted on 14 known individuals, demonstrated that the proposed subsystem can be successfully used in real time. The next step involves implementing the system on a large experimental UAV for field use and integrating autonomous navigation with GPS-guided control for rescue operations planning. 

**Abstract (ZH)**: 基于无人机的搜索与救援子系统：人员检测、人脸识别与跟踪技术研发 

---
# DSFormer: A Dual-Scale Cross-Learning Transformer for Visual Place Recognition 

**Title (ZH)**: DSFormer: 一种双尺度跨学习变换器用于视觉地点识别 

**Authors**: Haiyang Jiang, Songhao Piao, Chao Gao, Lei Yu, Liguo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.18444)  

**Abstract**: Visual Place Recognition (VPR) is crucial for robust mobile robot localization, yet it faces significant challenges in maintaining reliable performance under varying environmental conditions and viewpoints. To address this, we propose a novel framework that integrates Dual-Scale-Former (DSFormer), a Transformer-based cross-learning module, with an innovative block clustering strategy. DSFormer enhances feature representation by enabling bidirectional information transfer between dual-scale features extracted from the final two CNN layers, capturing both semantic richness and spatial details through self-attention for long-range dependencies within each scale and shared cross-attention for cross-scale learning. Complementing this, our block clustering strategy repartitions the widely used San Francisco eXtra Large (SF-XL) training dataset from multiple distinct perspectives, optimizing data organization to further bolster robustness against viewpoint variations. Together, these innovations not only yield a robust global embedding adaptable to environmental changes but also reduce the required training data volume by approximately 30\% compared to previous partitioning methods. Comprehensive experiments demonstrate that our approach achieves state-of-the-art performance across most benchmark datasets, surpassing advanced reranking methods like DELG, Patch-NetVLAD, TransVPR, and R2Former as a global retrieval solution using 512-dim global descriptors, while significantly improving computational efficiency. 

**Abstract (ZH)**: 视觉地点识别（VPR）对于鲁棒的移动机器人定位至关重要，但其在不同环境条件和视角下保持可靠性能面临显著挑战。为此，我们提出了一种新型框架，该框架结合了Dual-Scale-Former（DSFormer）——一种基于Transformer的跨学习模块——以及创新的块聚类策略。DSFormer通过在最终两层CNN提取的双尺度特征之间实现双向信息传递，增强特征表示，并通过自注意力机制捕捉每尺度内的长期依赖关系，通过共享的跨注意力机制进行跨尺度学习，从而同时捕获语义丰富性和空间细节。此外，我们的块聚类策略多角度重新 partition 了广泛使用的San Francisco eXtra Large（SF-XL）训练数据集，优化数据组织以进一步增强对视角变化的鲁棒性。这些创新不仅提供了适应环境变化的鲁棒全局嵌入表示，还比之前的方法减少了约30%的训练数据量。全面的实验表明，在大多数基准数据集上，我们的方法达到了最先进的性能，超越了包括DELG、Patch-NetVLAD、TransVPR和R2Former在内的高级重排序方法，使用512维全局描述符时具有显著的计算效率提升。 

---
# FishDet-M: A Unified Large-Scale Benchmark for Robust Fish Detection and CLIP-Guided Model Selection in Diverse Aquatic Visual Domains 

**Title (ZH)**: FishDet-M：统一的大规模基准，用于多样 aquatic 视觉域中鲁棒鱼类检测和 CLIP 引导模型选择。 

**Authors**: Muayad Abujabal, Lyes Saad Saoud, Irfan Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2507.17859)  

**Abstract**: Accurate fish detection in underwater imagery is essential for ecological monitoring, aquaculture automation, and robotic perception. However, practical deployment remains limited by fragmented datasets, heterogeneous imaging conditions, and inconsistent evaluation protocols. To address these gaps, we present \textit{FishDet-M}, the largest unified benchmark for fish detection, comprising 13 publicly available datasets spanning diverse aquatic environments including marine, brackish, occluded, and aquarium scenes. All data are harmonized using COCO-style annotations with both bounding boxes and segmentation masks, enabling consistent and scalable cross-domain evaluation. We systematically benchmark 28 contemporary object detection models, covering the YOLOv8 to YOLOv12 series, R-CNN based detectors, and DETR based models. Evaluations are conducted using standard metrics including mAP, mAP@50, and mAP@75, along with scale-specific analyses (AP$_S$, AP$_M$, AP$_L$) and inference profiling in terms of latency and parameter count. The results highlight the varying detection performance across models trained on FishDet-M, as well as the trade-off between accuracy and efficiency across models of different architectures. To support adaptive deployment, we introduce a CLIP-based model selection framework that leverages vision-language alignment to dynamically identify the most semantically appropriate detector for each input image. This zero-shot selection strategy achieves high performance without requiring ensemble computation, offering a scalable solution for real-time applications. FishDet-M establishes a standardized and reproducible platform for evaluating object detection in complex aquatic scenes. All datasets, pretrained models, and evaluation tools are publicly available to facilitate future research in underwater computer vision and intelligent marine systems. 

**Abstract (ZH)**: 准确的水下图像鱼类检测对于生态监控、水产自动化和机器人感知至关重要。然而，实际部署受限于碎片化的数据集、异质的成像条件和不一致的评估协议。为了解决这些问题，我们提出了FishDet-M，这是目前最大的统一鱼类检测基准，包含13个公开数据集，涵盖了包括海洋、半咸水、遮挡和水族箱在内的多种水生环境。所有数据均使用COCO样式注释，包括边界框和分割掩模，以实现跨域一致和可扩展的评估。我们系统性地评估了28个当代目标检测模型，涵盖了从YOLOv8到YOLOv12系列、基于R-CNN的检测器和基于DETR的模型。评估使用标准指标包括mAP、mAP@50和mAP@75，以及针对不同尺度的分析（AP$_S$、AP$_M$、AP$_L$）和推理时间及参数量的评估。结果突出了在FishDet-M上训练的不同模型之间的检测性能差异，以及不同架构模型之间的准确性和效率之间的权衡。为了支持自适应部署，我们引入了一种基于CLIP的模型选择框架，利用视觉-语言对齐动态识别每个输入图像的最语义合适的检测器。这种零样本选择策略无需集成计算即可实现高性能，为实时应用提供了一种可扩展的解决方案。FishDet-M为评估复杂水生环境中的目标检测建立了标准化和可重复的平台。所有数据集、预训练模型和评估工具均公开，以促进水下计算机视觉和智能海洋系统领域的未来研究。标题：FishDet-M：最大的统一鱼类检测基准 

---
# SynC: Synthetic Image Caption Dataset Refinement with One-to-many Mapping for Zero-shot Image Captioning 

**Title (ZH)**: SynC: 一种基于一对一映射合成图像标题数据集精炼方法的零样本图像描述研究 

**Authors**: Si-Woo Kim, MinJu Jeon, Ye-Chan Kim, Soeun Lee, Taewhan Kim, Dong-Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.18616)  

**Abstract**: Zero-shot Image Captioning (ZIC) increasingly utilizes synthetic datasets generated by text-to-image (T2I) models to mitigate the need for costly manual annotation. However, these T2I models often produce images that exhibit semantic misalignments with their corresponding input captions (e.g., missing objects, incorrect attributes), resulting in noisy synthetic image-caption pairs that can hinder model training. Existing dataset pruning techniques are largely designed for removing noisy text in web-crawled data. However, these methods are ill-suited for the distinct challenges of synthetic data, where captions are typically well-formed, but images may be inaccurate representations. To address this gap, we introduce SynC, a novel framework specifically designed to refine synthetic image-caption datasets for ZIC. Instead of conventional filtering or regeneration, SynC focuses on reassigning captions to the most semantically aligned images already present within the synthetic image pool. Our approach employs a one-to-many mapping strategy by initially retrieving multiple relevant candidate images for each caption. We then apply a cycle-consistency-inspired alignment scorer that selects the best image by verifying its ability to retrieve the original caption via image-to-text retrieval. Extensive evaluations demonstrate that SynC consistently and significantly improves performance across various ZIC models on standard benchmarks (MS-COCO, Flickr30k, NoCaps), achieving state-of-the-art results in several scenarios. SynC offers an effective strategy for curating refined synthetic data to enhance ZIC. 

**Abstract (ZH)**: 零样本图像描述（ZIC）越来越多地利用由文本到图像（T2I）模型生成的合成数据集，以减少对昂贵的手动标注的需求。然而，这些T2I模型生成的图像常常与对应的输入描述存在语义不一致（例如，缺少对象、属性错误），导致噪声合成图像-描述对，这可能会阻碍模型训练。现有的数据集精简技术大多设计用于去除从网络抓取数据中的噪声文本。然而，这些方法不适用于合成数据的特定挑战，其中描述通常结构良好，但图像可能是不准确的表示。为此，我们提出了SynC，一种专门设计用于 refinement合成图像-描述数据集以供ZIC使用的新型框架。相较传统的过滤或再生，SynC专注于将每个描述重新分配给合成图像池中与其最语义匹配的图像。我们的方法采用一种一对一多映射策略，初始阶段为每个描述检索多个相关的候选图像。随后，我们应用一个受循环一致性启发的对齐评分器，通过图像到文本的检索验证图像能否恢复原始描述，从而选择最佳图像。广泛评估表明，SynC在多个标准基准（MS-COCO、Flickr30k、NoCaps）上的一系列ZIC模型中一致且显著地提高了性能，并在多种场景中达到了最先进的结果。SynC提供了一种有效的方法来筛选精炼的合成数据，以提升零样本图像描述的性能。 

---
# DRWKV: Focusing on Object Edges for Low-Light Image Enhancement 

**Title (ZH)**: DRWKV：聚焦物体边缘的低光图像增强 

**Authors**: Xuecheng Bai, Yuxiang Wang, Boyu Hu, Qinyuan Jie, Chuanzhi Xu, Hongru Xiao, Kechen Li, Vera Chung  

**Link**: [PDF](https://arxiv.org/pdf/2507.18594)  

**Abstract**: Low-light image enhancement remains a challenging task, particularly in preserving object edge continuity and fine structural details under extreme illumination degradation. In this paper, we propose a novel model, DRWKV (Detailed Receptance Weighted Key Value), which integrates our proposed Global Edge Retinex (GER) theory, enabling effective decoupling of illumination and edge structures for enhanced edge fidelity. Secondly, we introduce Evolving WKV Attention, a spiral-scanning mechanism that captures spatial edge continuity and models irregular structures more effectively. Thirdly, we design the Bilateral Spectrum Aligner (Bi-SAB) and a tailored MS2-Loss to jointly align luminance and chrominance features, improving visual naturalness and mitigating artifacts. Extensive experiments on five LLIE benchmarks demonstrate that DRWKV achieves leading performance in PSNR, SSIM, and NIQE while maintaining low computational complexity. Furthermore, DRWKV enhances downstream performance in low-light multi-object tracking tasks, validating its generalization capabilities. 

**Abstract (ZH)**: 低光图像增强仍是一项具有挑战性的任务，特别是在极端光照降级情况下保持对象边缘连续性和精细结构细节方面。本文提出了一种新型模型DRWKV（详细接收率加权键值），该模型整合了我们提出的全局边缘_retinex_理论，可以有效解耦光照和边缘结构，从而增强边缘保真度。其次，我们引入了演化WKV注意力机制，这是一种螺旋扫描机制，可以捕捉空间边缘连续性并更有效地建模不规则结构。第三，我们设计了双边光谱对齐器（Bi-SAB）和特定的MS2损失，以联合对亮度和色度特征进行对齐，从而改善视觉自然度并减轻伪影。在五个LLIE基准上的广泛实验表明，DRWKV在PSNR、SSIM和NIQE上取得了领先性能，同时保持较低的计算复杂度。此外，DRWKV在低光多对象跟踪任务中的下游性能得到了增强，验证了其泛化能力。 

---
# Revisiting Physically Realizable Adversarial Object Attack against LiDAR-based Detection: Clarifying Problem Formulation and Experimental Protocols 

**Title (ZH)**: 重新审视基于物理可实现的LiDAR目标攻击的对抗性物体攻击：明确问题陈述与实验协议 

**Authors**: Luo Cheng, Hanwei Zhang, Lijun Zhang, Holger Hermanns  

**Link**: [PDF](https://arxiv.org/pdf/2507.18457)  

**Abstract**: Adversarial robustness in LiDAR-based 3D object detection is a critical research area due to its widespread application in real-world scenarios. While many digital attacks manipulate point clouds or meshes, they often lack physical realizability, limiting their practical impact. Physical adversarial object attacks remain underexplored and suffer from poor reproducibility due to inconsistent setups and hardware differences. To address this, we propose a device-agnostic, standardized framework that abstracts key elements of physical adversarial object attacks, supports diverse methods, and provides open-source code with benchmarking protocols in simulation and real-world settings. Our framework enables fair comparison, accelerates research, and is validated by successfully transferring simulated attacks to a physical LiDAR system. Beyond the framework, we offer insights into factors influencing attack success and advance understanding of adversarial robustness in real-world LiDAR perception. 

**Abstract (ZH)**: 基于LiDAR的3D物体检测对抗鲁棒性是一个关键的研究领域，由于其在实际场景中的广泛应用。虽然许多数字攻击操控点云或网格，但它们往往缺乏物理可行性，限制了其实际影响。物理对抗物体攻击尚未得到充分探索，且由于设置不一致和硬件差异，重现性较差。为解决这一问题，我们提出了一种设备无关的标准框架，该框架抽象了物理对抗物体攻击的关键要素，支持多种方法，并提供了在仿真和实际场景中进行基准测试的开源代码。该框架使比较公平、加速了研究，并通过成功将模拟攻击转移到物理LiDAR系统中得到验证。此外，我们还探讨了影响攻击成功的关键因素，深化了对实际LiDAR感知中对抗鲁棒性的理解。 

---
# Improving Bird Classification with Primary Color Additives 

**Title (ZH)**: 使用主要颜色添加剂提高鸟类分类效果 

**Authors**: Ezhini Rasendiran R, Chandresh Kumar Maurya  

**Link**: [PDF](https://arxiv.org/pdf/2507.18334)  

**Abstract**: We address the problem of classifying bird species using their song recordings, a challenging task due to environmental noise, overlapping vocalizations, and missing labels. Existing models struggle with low-SNR or multi-species recordings. We hypothesize that birds can be classified by visualizing their pitch pattern, speed, and repetition, collectively called motifs. Deep learning models applied to spectrogram images help, but similar motifs across species cause confusion. To mitigate this, we embed frequency information into spectrograms using primary color additives. This enhances species distinction and improves classification accuracy. Our experiments show that the proposed approach achieves statistically significant gains over models without colorization and surpasses the BirdCLEF 2024 winner, improving F1 by 7.3%, ROC-AUC by 6.2%, and CMAP by 6.6%. These results demonstrate the effectiveness of incorporating frequency information via colorization. 

**Abstract (ZH)**: 使用彩色谱图图示化频率信息进行鸟类物种分类 

---
# Exploiting Gaussian Agnostic Representation Learning with Diffusion Priors for Enhanced Infrared Small Target Detection 

**Title (ZH)**: 基于扩散先验的高斯agnostic表示学习增强红外小目标检测 

**Authors**: Junyao Li, Yahao Lu, Xingyuan Guo, Xiaoyu Xian, Tiantian Wang, Yukai Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.18260)  

**Abstract**: Infrared small target detection (ISTD) plays a vital role in numerous practical applications. In pursuit of determining the performance boundaries, researchers employ large and expensive manual-labeling data for representation learning. Nevertheless, this approach renders the state-of-the-art ISTD methods highly fragile in real-world challenges. In this paper, we first study the variation in detection performance across several mainstream methods under various scarcity -- namely, the absence of high-quality infrared data -- that challenge the prevailing theories about practical ISTD. To address this concern, we introduce the Gaussian Agnostic Representation Learning. Specifically, we propose the Gaussian Group Squeezer, leveraging Gaussian sampling and compression for non-uniform quantization. By exploiting a diverse array of training samples, we enhance the resilience of ISTD models against various challenges. Then, we introduce two-stage diffusion models for real-world reconstruction. By aligning quantized signals closely with real-world distributions, we significantly elevate the quality and fidelity of the synthetic samples. Comparative evaluations against state-of-the-art detection methods in various scarcity scenarios demonstrate the efficacy of the proposed approach. 

**Abstract (ZH)**: 红外小目标检测（ISTD）在众多实际应用中发挥着重要作用。为了确定性能边界，研究者们使用大量昂贵的手动标注数据进行表示学习。然而，这种方法使得最先进的ISTD方法在实际挑战中非常脆弱。在本文中，我们首先研究了在各种数据稀缺条件下（即缺乏高质量红外数据）主流方法的检测性能变异，从而挑战了现有关于实际ISTD的理论。为解决这一问题，我们引入了高斯无偏表示学习。具体来说，我们提出了高斯组挤压器，利用高斯采样和压缩进行非均匀量化。通过利用多样化的训练样本，我们增强了ISTD模型对各种挑战的鲁棒性。然后，我们介绍了两级扩散模型进行实际重构。通过使量化信号紧密符合实际分布，我们显著提高了合成样本的质量和保真度。在各种数据稀缺条件下的对比评估证明了所提出方法的有效性。 

---
# DepthDark: Robust Monocular Depth Estimation for Low-Light Environments 

**Title (ZH)**: 深度黑暗：低光照环境下的鲁棒单目深度估计 

**Authors**: Longjian Zeng, Zunjie Zhu, Rongfeng Lu, Ming Lu, Bolun Zheng, Chenggang Yan, Anke Xue  

**Link**: [PDF](https://arxiv.org/pdf/2507.18243)  

**Abstract**: In recent years, foundation models for monocular depth estimation have received increasing attention. Current methods mainly address typical daylight conditions, but their effectiveness notably decreases in low-light environments. There is a lack of robust foundational models for monocular depth estimation specifically designed for low-light scenarios. This largely stems from the absence of large-scale, high-quality paired depth datasets for low-light conditions and the effective parameter-efficient fine-tuning (PEFT) strategy. To address these challenges, we propose DepthDark, a robust foundation model for low-light monocular depth estimation. We first introduce a flare-simulation module and a noise-simulation module to accurately simulate the imaging process under nighttime conditions, producing high-quality paired depth datasets for low-light conditions. Additionally, we present an effective low-light PEFT strategy that utilizes illumination guidance and multiscale feature fusion to enhance the model's capability in low-light environments. Our method achieves state-of-the-art depth estimation performance on the challenging nuScenes-Night and RobotCar-Night datasets, validating its effectiveness using limited training data and computing resources. 

**Abstract (ZH)**: 低光环境单目深度估计的鲁棒基础模型：DepthDark 

---
# Parameter-Efficient Fine-Tuning of 3D DDPM for MRI Image Generation Using Tensor Networks 

**Title (ZH)**: 使用张量网络的3D DDPM参数高效微调以生成MRI图像 

**Authors**: Binghua Li, Ziqing Chang, Tong Liang, Chao Li, Toshihisa Tanaka, Shigeki Aoki, Qibin Zhao, Zhe Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.18112)  

**Abstract**: We address the challenge of parameter-efficient fine-tuning (PEFT) for three-dimensional (3D) U-Net-based denoising diffusion probabilistic models (DDPMs) in magnetic resonance imaging (MRI) image generation. Despite its practical significance, research on parameter-efficient representations of 3D convolution operations remains limited. To bridge this gap, we propose Tensor Volumetric Operator (TenVOO), a novel PEFT method specifically designed for fine-tuning DDPMs with 3D convolutional backbones. Leveraging tensor network modeling, TenVOO represents 3D convolution kernels with lower-dimensional tensors, effectively capturing complex spatial dependencies during fine-tuning with few parameters. We evaluate TenVOO on three downstream brain MRI datasets-ADNI, PPMI, and BraTS2021-by fine-tuning a DDPM pretrained on 59,830 T1-weighted brain MRI scans from the UK Biobank. Our results demonstrate that TenVOO achieves state-of-the-art performance in multi-scale structural similarity index measure (MS-SSIM), outperforming existing approaches in capturing spatial dependencies while requiring only 0.3% of the trainable parameters of the original model. Our code is available at: this https URL 

**Abstract (ZH)**: 我们针对基于3D U-Net的去噪扩散概率模型（DDPMs）在磁共振成像（MRI）图像生成中的参数高效微调（PEFT）挑战进行了研究。通过引入张量体元算子（TenVOO）这一新型PEFT方法，特别是在3D卷积骨干网络下的微调，我们利用张量网络建模，以低维张量表示3D卷积核，在微调过程中有效捕捉复杂的空间依赖性，同时仅使用较少的参数。我们在ADNI、PPMI和BraTS2021三个下游脑MRI数据集上评估了TenVOO，通过对英国生物银行中59,830张T1加权脑MRI扫描数据预训练的DDPM进行微调。实验结果表明，TenVOO在多尺度结构相似性指数度量（MS-SSIM）上取得了最佳性能，同时在捕捉空间依赖性方面优于现有方法，仅需原始模型可训练参数的0.3%。代码现可从以下链接获取：this https URL。 

---
# Datasets and Recipes for Video Temporal Grounding via Reinforcement Learning 

**Title (ZH)**: 用于强化学习的视频时间定位数据集和食谱 

**Authors**: Ruizhe Chen, Zhiting Fan, Tianze Luo, Heqing Zou, Zhaopeng Feng, Guiyang Xie, Hansheng Zhang, Zhuochen Wang, Zuozhu Liu, Huaijian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18100)  

**Abstract**: Video Temporal Grounding (VTG) aims to localize relevant temporal segments in videos given natural language queries. Despite recent progress with large vision-language models (LVLMs) and instruction-tuning, existing approaches often suffer from limited temporal awareness and poor generalization. In this work, we introduce a two-stage training framework that integrates supervised fine-tuning with reinforcement learning (RL) to improve both the accuracy and robustness of VTG models. Our approach first leverages high-quality curated cold start data for SFT initialization, followed by difficulty-controlled RL to further enhance temporal localization and reasoning abilities. Comprehensive experiments on multiple VTG benchmarks demonstrate that our method consistently outperforms existing models, particularly in challenging and open-domain scenarios. We conduct an in-depth analysis of training strategies and dataset curation, highlighting the importance of both high-quality cold start data and difficulty-controlled RL. To facilitate further research and industrial adoption, we release all intermediate datasets, models, and code to the community. 

**Abstract (ZH)**: 视频时间定位（VTG）的目标是给定自然语言查询，在视频中定位相关的时序段。尽管大规模视觉-语言模型（LVLMs）和指令调优取得了进步，现有的方法往往存在时间感知有限和泛化能力差的问题。在本工作中，我们提出了一种两阶段训练框架，将监督微调与强化学习（RL）相结合，以提高VTG模型的准确性和鲁棒性。我们的方法首先利用高质量策划的冷启动数据进行监督微调初始化，随后通过难度可控的RL进一步增强时间定位和推理能力。在多个VTG基准上的全面实验表明，我们的方法在各种场景下（尤其是挑战性和开放域场景）均优于现有模型。我们对训练策略和数据策划进行了深入分析，强调了高质量冷启动数据和难度可控的RL的重要性。为了促进进一步的研究和工业应用，我们向社区提供了所有中间数据集、模型和代码。 

---
# TextSAM-EUS: Text Prompt Learning for SAM to Accurately Segment Pancreatic Tumor in Endoscopic Ultrasound 

**Title (ZH)**: TextSAM-EUS: 文本提示学习以提高萨姆分割胰腺肿瘤的准确性在内镜超声中的应用 

**Authors**: Pascal Spiegler, Taha Koleilat, Arash Harirpoush, Corey S. Miller, Hassan Rivaz, Marta Kersten-Oertel, Yiming Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.18082)  

**Abstract**: Pancreatic cancer carries a poor prognosis and relies on endoscopic ultrasound (EUS) for targeted biopsy and radiotherapy. However, the speckle noise, low contrast, and unintuitive appearance of EUS make segmentation of pancreatic tumors with fully supervised deep learning (DL) models both error-prone and dependent on large, expert-curated annotation datasets. To address these challenges, we present TextSAM-EUS, a novel, lightweight, text-driven adaptation of the Segment Anything Model (SAM) that requires no manual geometric prompts at inference. Our approach leverages text prompt learning (context optimization) through the BiomedCLIP text encoder in conjunction with a LoRA-based adaptation of SAM's architecture to enable automatic pancreatic tumor segmentation in EUS, tuning only 0.86% of the total parameters. On the public Endoscopic Ultrasound Database of the Pancreas, TextSAM-EUS with automatic prompts attains 82.69% Dice and 85.28% normalized surface distance (NSD), and with manual geometric prompts reaches 83.10% Dice and 85.70% NSD, outperforming both existing state-of-the-art (SOTA) supervised DL models and foundation models (e.g., SAM and its variants). As the first attempt to incorporate prompt learning in SAM-based medical image segmentation, TextSAM-EUS offers a practical option for efficient and robust automatic EUS segmentation. Our code will be publicly available upon acceptance. 

**Abstract (ZH)**: 基于文本驱动的TextSAM-EUS在内镜超声影像中自动胰腺肿瘤分割 

---
# Enhancing Scene Transition Awareness in Video Generation via Post-Training 

**Title (ZH)**: 通过后训练增强视频生成中的场景过渡意识 

**Authors**: Hanwen Shen, Jiajie Lu, Yupeng Cao, Xiaonan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18046)  

**Abstract**: Recent advances in AI-generated video have shown strong performance on \emph{text-to-video} tasks, particularly for short clips depicting a single scene. However, current models struggle to generate longer videos with coherent scene transitions, primarily because they cannot infer when a transition is needed from the prompt. Most open-source models are trained on datasets consisting of single-scene video clips, which limits their capacity to learn and respond to prompts requiring multiple scenes. Developing scene transition awareness is essential for multi-scene generation, as it allows models to identify and segment videos into distinct clips by accurately detecting transitions.
To address this, we propose the \textbf{Transition-Aware Video} (TAV) dataset, which consists of preprocessed video clips with multiple scene transitions. Our experiment shows that post-training on the \textbf{TAV} dataset improves prompt-based scene transition understanding, narrows the gap between required and generated scenes, and maintains image quality. 

**Abstract (ZH)**: 近期AI生成视频的研究在文本到视频任务上取得了显著性能，尤其是对于描绘单个场景的短片段。然而，当前模型在生成长视频和连贯场景过渡方面表现不佳，主要原因是它们无法从提示中推断出何时需要进行过渡。大多数开源模型都是在仅包含单场景视频片段的数据集上训练的，这限制了它们在处理需要多个场景的提示时的学习和响应能力。为了应对这一挑战，我们提出了**具有场景意识的视频**（Transition-Aware Video, TAV）数据集，该数据集包含多个场景过渡的预处理视频片段。我们的实验表明，基于TAV数据集进行训练可以提高基于提示的场景过渡理解能力，缩小所需场景和生成场景之间的差距，并保持图像质量。 

---
# ViGText: Deepfake Image Detection with Vision-Language Model Explanations and Graph Neural Networks 

**Title (ZH)**: ViGText：结合视觉-语言模型解释和图神经网络的深仿图像检测 

**Authors**: Ahmad ALBarqawi, Mahmoud Nazzal, Issa Khalil, Abdallah Khreishah, NhatHai Phan  

**Link**: [PDF](https://arxiv.org/pdf/2507.18031)  

**Abstract**: The rapid rise of deepfake technology, which produces realistic but fraudulent digital content, threatens the authenticity of media. Traditional deepfake detection approaches often struggle with sophisticated, customized deepfakes, especially in terms of generalization and robustness against malicious attacks. This paper introduces ViGText, a novel approach that integrates images with Vision Large Language Model (VLLM) Text explanations within a Graph-based framework to improve deepfake detection. The novelty of ViGText lies in its integration of detailed explanations with visual data, as it provides a more context-aware analysis than captions, which often lack specificity and fail to reveal subtle inconsistencies. ViGText systematically divides images into patches, constructs image and text graphs, and integrates them for analysis using Graph Neural Networks (GNNs) to identify deepfakes. Through the use of multi-level feature extraction across spatial and frequency domains, ViGText captures details that enhance its robustness and accuracy to detect sophisticated deepfakes. Extensive experiments demonstrate that ViGText significantly enhances generalization and achieves a notable performance boost when it detects user-customized deepfakes. Specifically, average F1 scores rise from 72.45% to 98.32% under generalization evaluation, and reflects the model's superior ability to generalize to unseen, fine-tuned variations of stable diffusion models. As for robustness, ViGText achieves an increase of 11.1% in recall compared to other deepfake detection approaches. When facing targeted attacks that exploit its graph-based architecture, ViGText limits classification performance degradation to less than 4%. ViGText uses detailed visual and textual analysis to set a new standard for detecting deepfakes, helping ensure media authenticity and information integrity. 

**Abstract (ZH)**: 快速崛起的深度伪造技术产生了逼真但虚假的数字内容，威胁到媒体的真实性。传统深度伪造检测方法往往难以应对复杂的、定制化的深度伪造，特别是在泛化能力和抵御恶意攻击的鲁棒性方面。本文介绍了一种名为ViGText的新型方法，该方法将图像与Vision Large Language Model (VLLM) 文本解释结合在图框架中，以提高深度伪造检测能力。ViGText的创新之处在于它将详细的解释与视觉数据相结合，提供比缺乏具体性且难以揭示细微不一致性的字幕更丰富的上下文分析。ViGText系统性地将图像分割成块，构建图像和文本图，并使用图神经网络（GNNs）进行分析以识别深度伪造。通过跨空间和频域的多级特征提取，ViGText捕获了增强其鲁棒性和检测复杂深度伪造准确性的细节。广泛的实验表明，ViGText显著提高了泛化能力，并在检测用户定制的深度伪造时实现了显著的性能提升。具体来说，在泛化评估中，平均F1分数从72.45%提高到98.32%，反映了模型在泛化到未见过的、微调后的稳定扩散模型变体方面的优越能力。对于鲁棒性，ViGText相较于其他深度伪造检测方法召回率提高了11.1%。面对利用其图架构的针对性攻击，ViGText将分类性能下降限制在少于4%。ViGText通过详细的视觉和文本分析设立了检测深度伪造的新标准，有助于确保媒体的真实性与信息完整性。 

---
# Detail++: Training-Free Detail Enhancer for Text-to-Image Diffusion Models 

**Title (ZH)**: Detail++: 不需训练的文本到图像扩散模型细节增强器 

**Authors**: Lifeng Chen, Jiner Wang, Zihao Pan, Beier Zhu, Xiaofeng Yang, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17853)  

**Abstract**: Recent advances in text-to-image (T2I) generation have led to impressive visual results. However, these models still face significant challenges when handling complex prompt, particularly those involving multiple subjects with distinct attributes. Inspired by the human drawing process, which first outlines the composition and then incrementally adds details, we propose Detail++, a training-free framework that introduces a novel Progressive Detail Injection (PDI) strategy to address this limitation. Specifically, we decompose a complex prompt into a sequence of simplified sub-prompts, guiding the generation process in stages. This staged generation leverages the inherent layout-controlling capacity of self-attention to first ensure global composition, followed by precise refinement. To achieve accurate binding between attributes and corresponding subjects, we exploit cross-attention mechanisms and further introduce a Centroid Alignment Loss at test time to reduce binding noise and enhance attribute consistency. Extensive experiments on T2I-CompBench and a newly constructed style composition benchmark demonstrate that Detail++ significantly outperforms existing methods, particularly in scenarios involving multiple objects and complex stylistic conditions. 

**Abstract (ZH)**: Recent Advances in Text-to-Image Generation: Detail++ Framework for Handling Complex Prompts 

---
# SV3.3B: A Sports Video Understanding Model for Action Recognition 

**Title (ZH)**: SV3.3B: 一种用于动作识别的体育视频理解模型 

**Authors**: Sai Varun Kodathala, Yashwanth Reddy Vutukoori, Rakesh Vunnam  

**Link**: [PDF](https://arxiv.org/pdf/2507.17844)  

**Abstract**: This paper addresses the challenge of automated sports video analysis, which has traditionally been limited by computationally intensive models requiring server-side processing and lacking fine-grained understanding of athletic movements. Current approaches struggle to capture the nuanced biomechanical transitions essential for meaningful sports analysis, often missing critical phases like preparation, execution, and follow-through that occur within seconds. To address these limitations, we introduce SV3.3B, a lightweight 3.3B parameter video understanding model that combines novel temporal motion difference sampling with self-supervised learning for efficient on-device deployment. Our approach employs a DWT-VGG16-LDA based keyframe extraction mechanism that intelligently identifies the 16 most representative frames from sports sequences, followed by a V-DWT-JEPA2 encoder pretrained through mask-denoising objectives and an LLM decoder fine-tuned for sports action description generation. Evaluated on a subset of the NSVA basketball dataset, SV3.3B achieves superior performance across both traditional text generation metrics and sports-specific evaluation criteria, outperforming larger closed-source models including GPT-4o variants while maintaining significantly lower computational requirements. Our model demonstrates exceptional capability in generating technically detailed and analytically rich sports descriptions, achieving 29.2% improvement over GPT-4o in ground truth validation metrics, with substantial improvements in information density, action complexity, and measurement precision metrics essential for comprehensive athletic analysis. Model Available at this https URL. 

**Abstract (ZH)**: 本论文探讨了自动化体育视频分析的挑战，传统上由于计算密集型模型需要服务器端处理且缺乏对运动细节的精细理解而受到限制。当前的方法难以捕捉到有意义的体育分析所需的关键生物力学转换细节，往往忽略了如准备、执行和跟进等在几秒内发生的关键阶段。为了解决这些限制，我们介绍了SV3.3B，一个轻量级的3.3B参数视频理解模型，该模型结合了新型时间运动差异采样和自监督学习，以实现高效设备端部署。我们的方法采用基于DWT-VGG16-LDA的关键帧提取机制，从体育序列中智能地识别出最具代表性的16帧，随后通过掩码去噪目标预训练的V-DWT-JEPA2编码器和用于体育动作描述生成的LLM解码器的微调版本。在NSVA篮球数据集的一个子集上进行评估，SV3.3B在传统文本生成指标和专门的体育评估标准方面表现出色，优于包括GPT-4o变体在内的更大规模的封闭源模型，同时计算要求显著降低。我们的模型展示了在生成技术上详细和分析上丰富的体育描述方面的出色能力，在地面真实验证指标上相较于GPT-4o提升了29.2%，并在涉及全面运动员分析的信息密度、动作复杂性和测量精度指标方面取得了显著改进。模型可访问地址：this https URL 

---
# Learning from Heterogeneity: Generalizing Dynamic Facial Expression Recognition via Distributionally Robust Optimization 

**Title (ZH)**: 从异质性中学习：基于分布鲁棒优化的动态面部表情识别泛化 

**Authors**: Feng-Qi Cui, Anyang Tong, Jinyang Huang, Jie Zhang, Dan Guo, Zhi Liu, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15765)  

**Abstract**: Dynamic Facial Expression Recognition (DFER) plays a critical role in affective computing and human-computer interaction. Although existing methods achieve comparable performance, they inevitably suffer from performance degradation under sample heterogeneity caused by multi-source data and individual expression variability. To address these challenges, we propose a novel framework, called Heterogeneity-aware Distributional Framework (HDF), and design two plug-and-play modules to enhance time-frequency modeling and mitigate optimization imbalance caused by hard samples. Specifically, the Time-Frequency Distributional Attention Module (DAM) captures both temporal consistency and frequency robustness through a dual-branch attention design, improving tolerance to sequence inconsistency and visual style shifts. Then, based on gradient sensitivity and information bottleneck principles, an adaptive optimization module Distribution-aware Scaling Module (DSM) is introduced to dynamically balance classification and contrastive losses, enabling more stable and discriminative representation learning. Extensive experiments on two widely used datasets, DFEW and FERV39k, demonstrate that HDF significantly improves both recognition accuracy and robustness. Our method achieves superior weighted average recall (WAR) and unweighted average recall (UAR) while maintaining strong generalization across diverse and imbalanced scenarios. Codes are released at this https URL. 

**Abstract (ZH)**: 异构分布感知动态面部表情识别框架（Heterogeneity-aware Distributional Framework for Dynamic Facial Expression Recognition） 

---
