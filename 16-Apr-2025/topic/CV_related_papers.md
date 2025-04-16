# CAP-Net: A Unified Network for 6D Pose and Size Estimation of Categorical Articulated Parts from a Single RGB-D Image 

**Title (ZH)**: CAP-Net：从单张RGB-D图像中估计类别化articulated部分6D姿态和尺寸的一体化网络 

**Authors**: Jingshun Huang, Haitao Lin, Tianyu Wang, Yanwei Fu, Xiangyang Xue, Yi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11230)  

**Abstract**: This paper tackles category-level pose estimation of articulated objects in robotic manipulation tasks and introduces a new benchmark dataset. While recent methods estimate part poses and sizes at the category level, they often rely on geometric cues and complex multi-stage pipelines that first segment parts from the point cloud, followed by Normalized Part Coordinate Space (NPCS) estimation for 6D poses. These approaches overlook dense semantic cues from RGB images, leading to suboptimal accuracy, particularly for objects with small parts. To address these limitations, we propose a single-stage Network, CAP-Net, for estimating the 6D poses and sizes of Categorical Articulated Parts. This method combines RGB-D features to generate instance segmentation and NPCS representations for each part in an end-to-end manner. CAP-Net uses a unified network to simultaneously predict point-wise class labels, centroid offsets, and NPCS maps. A clustering algorithm then groups points of the same predicted class based on their estimated centroid distances to isolate each part. Finally, the NPCS region of each part is aligned with the point cloud to recover its final pose and size. To bridge the sim-to-real domain gap, we introduce the RGBD-Art dataset, the largest RGB-D articulated dataset to date, featuring photorealistic RGB images and depth noise simulated from real sensors. Experimental evaluations on the RGBD-Art dataset demonstrate that our method significantly outperforms the state-of-the-art approach. Real-world deployments of our model in robotic tasks underscore its robustness and exceptional sim-to-real transfer capabilities, confirming its substantial practical utility. Our dataset, code and pre-trained models are available on the project page. 

**Abstract (ZH)**: 本文解决了机器人操作任务中刚性部件类别级姿态估计问题，并引入了一个新的基准数据集。虽然最近的方法在类别级别估计部件姿态和尺寸，但它们通常依赖于几何线索和复杂的多阶段管道，首先从点云中分割部件，然后使用归一化部分坐标空间(NPCS)估计6D姿态。这些方法忽视了从RGB图像中提取的密集语义线索，导致精度欠佳，特别是对于具有小部件的对象。为了解决这些局限性，我们提出了一种单阶段网络CAP-Net，用于估计类别级可变形部件的6D姿态和尺寸。该方法结合RGB-D特征，以端到端的方式生成每个部件的实例分割和NPCS表示。CAP-Net使用统一网络同时预测点级类别标签、质心偏移和NPCS图。然后，使用聚类算法根据估计的质心距离对具有相同预测类别的点进行分组，以隔离每个部件。最后，将每个部件的NPCS区域与点云对齐以恢复其最终姿态和尺寸。为弥合仿真到现实的领域差距，我们引入了RGBD-Art数据集，这是迄今为止最大的RGB-D可变形数据集，包含逼真的RGB图像和从真实传感器模拟的深度噪声。在RGBD-Art数据集上的实验评估表明，我们的方法显著优于现有最先进的方法。在实际机器人任务中的应用部署证明了其稳健性和出色的仿真实现迁移能力，确认了其重大的实用价值。我们的数据集、代码和预训练模型可在项目页面获取。 

---
# Real-time Seafloor Segmentation and Mapping 

**Title (ZH)**: 实时海底分割与制图 

**Authors**: Michele Grimaldi, Nouf Alkaabi, Francesco Ruscio, Sebastian Realpe Rua, Rafael Garcia, Nuno Gracias  

**Link**: [PDF](https://arxiv.org/pdf/2504.10750)  

**Abstract**: Posidonia oceanica meadows are a species of seagrass highly dependent on rocks for their survival and conservation. In recent years, there has been a concerning global decline in this species, emphasizing the critical need for efficient monitoring and assessment tools. While deep learning-based semantic segmentation and visual automated monitoring systems have shown promise in a variety of applications, their performance in underwater environments remains challenging due to complex water conditions and limited datasets. This paper introduces a framework that combines machine learning and computer vision techniques to enable an autonomous underwater vehicle (AUV) to inspect the boundaries of Posidonia oceanica meadows autonomously. The framework incorporates an image segmentation module using an existing Mask R-CNN model and a strategy for Posidonia oceanica meadow boundary tracking. Furthermore, a new class dedicated to rocks is introduced to enhance the existing model, aiming to contribute to a comprehensive monitoring approach and provide a deeper understanding of the intricate interactions between the meadow and its surrounding environment. The image segmentation model is validated using real underwater images, while the overall inspection framework is evaluated in a realistic simulation environment, replicating actual monitoring scenarios with real underwater images. The results demonstrate that the proposed framework enables the AUV to autonomously accomplish the main tasks of underwater inspection and segmentation of rocks. Consequently, this work holds significant potential for the conservation and protection of marine environments, providing valuable insights into the status of Posidonia oceanica meadows and supporting targeted preservation efforts 

**Abstract (ZH)**: Posidonia oceanica 沼泽的监测与评估：结合机器学习和计算机视觉的自主水下车辆框架 

---
# MARVIS: Motion & Geometry Aware Real and Virtual Image Segmentation 

**Title (ZH)**: MARVIS: 动态与几何感知的现实与虚拟图像分割 

**Authors**: Jiayi Wu, Xiaomin Lin, Shahriar Negahdaripour, Cornelia Fermüller, Yiannis Aloimonos  

**Link**: [PDF](https://arxiv.org/pdf/2403.09850)  

**Abstract**: Tasks such as autonomous navigation, 3D reconstruction, and object recognition near the water surfaces are crucial in marine robotics applications. However, challenges arise due to dynamic disturbances, e.g., light reflections and refraction from the random air-water interface, irregular liquid flow, and similar factors, which can lead to potential failures in perception and navigation systems. Traditional computer vision algorithms struggle to differentiate between real and virtual image regions, significantly complicating tasks. A virtual image region is an apparent representation formed by the redirection of light rays, typically through reflection or refraction, creating the illusion of an object's presence without its actual physical location. This work proposes a novel approach for segmentation on real and virtual image regions, exploiting synthetic images combined with domain-invariant information, a Motion Entropy Kernel, and Epipolar Geometric Consistency. Our segmentation network does not need to be re-trained if the domain changes. We show this by deploying the same segmentation network in two different domains: simulation and the real world. By creating realistic synthetic images that mimic the complexities of the water surface, we provide fine-grained training data for our network (MARVIS) to discern between real and virtual images effectively. By motion & geometry-aware design choices and through comprehensive experimental analysis, we achieve state-of-the-art real-virtual image segmentation performance in unseen real world domain, achieving an IoU over 78% and a F1-Score over 86% while ensuring a small computational footprint. MARVIS offers over 43 FPS (8 FPS) inference rates on a single GPU (CPU core). Our code and dataset are available here this https URL. 

**Abstract (ZH)**: 水表面附近自主导航、三维重建和物体识别任务在水下机器人应用中至关重要。然而，由于动态干扰因素，如随机气-水界面的光反射和折射、不规则液体流动等，感知和导航系统可能失效。传统计算机视觉算法难以区分真实和虚拟图像区域，显著增加了任务难度。虚拟图像区域是光线经过反射或折射重定向后形成的表象，并不真实存在。本文提出了一种结合合成图像、领域不变信息、运动熵核和极线几何一致性的新颖方法，用于真实和虚拟图像区域的分割。我们的分割网络在领域变化时无需重新训练。通过在模拟和真实世界两个不同领域中部署相同的分割网络，我们展示了这一点。通过生成模拟真实水面复杂性的逼真合成图像，为网络（MARVIS）提供精细的训练数据，以有效区分真实和虚拟图像。通过运动与几何感知设计选择和全面的实验分析，在未见的真实世界领域中实现了最先进的真实虚拟图像分割性能，IoU超过78%，F1-Score超过86%，同时保持较小的计算开销。MARVIS在单块GPU（CPU核心）上实现超过43 FPS（8 FPS）的推理速率。我们的代码和数据集可在此处获取：https://。 

---
# ADT: Tuning Diffusion Models with Adversarial Supervision 

**Title (ZH)**: ADT：通过对抗监督调优扩散模型 

**Authors**: Dazhong Shen, Guanglu Song, Yi Zhang, Bingqi Ma, Lujundong Li, Dongzhi Jiang, Zhuofan Zong, Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11423)  

**Abstract**: Diffusion models have achieved outstanding image generation by reversing a forward noising process to approximate true data distributions. During training, these models predict diffusion scores from noised versions of true samples in a single forward pass, while inference requires iterative denoising starting from white noise. This training-inference divergences hinder the alignment between inference and training data distributions, due to potential prediction biases and cumulative error accumulation. To address this problem, we propose an intuitive but effective fine-tuning framework, called Adversarial Diffusion Tuning (ADT), by stimulating the inference process during optimization and aligning the final outputs with training data by adversarial supervision. Specifically, to achieve robust adversarial training, ADT features a siamese-network discriminator with a fixed pre-trained backbone and lightweight trainable parameters, incorporates an image-to-image sampling strategy to smooth discriminative difficulties, and preserves the original diffusion loss to prevent discriminator hacking. In addition, we carefully constrain the backward-flowing path for back-propagating gradients along the inference path without incurring memory overload or gradient explosion. Finally, extensive experiments on Stable Diffusion models (v1.5, XL, and v3), demonstrate that ADT significantly improves both distribution alignment and image quality. 

**Abstract (ZH)**: 对抗扩散调优（ADT）：优化过程中的对抗性监督以改善分布对齐和图像质量 

---
# VideoPanda: Video Panoramic Diffusion with Multi-view Attention 

**Title (ZH)**: VideoPanda：基于多视图Attention的全景视频扩散 

**Authors**: Kevin Xie, Amirmojtaba Sabour, Jiahui Huang, Despoina Paschalidou, Greg Klar, Umar Iqbal, Sanja Fidler, Xiaohui Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2504.11389)  

**Abstract**: High resolution panoramic video content is paramount for immersive experiences in Virtual Reality, but is non-trivial to collect as it requires specialized equipment and intricate camera setups. In this work, we introduce VideoPanda, a novel approach for synthesizing 360$^\circ$ videos conditioned on text or single-view video data. VideoPanda leverages multi-view attention layers to augment a video diffusion model, enabling it to generate consistent multi-view videos that can be combined into immersive panoramic content. VideoPanda is trained jointly using two conditions: text-only and single-view video, and supports autoregressive generation of long-videos. To overcome the computational burden of multi-view video generation, we randomly subsample the duration and camera views used during training and show that the model is able to gracefully generalize to generating more frames during inference. Extensive evaluations on both real-world and synthetic video datasets demonstrate that VideoPanda generates more realistic and coherent 360$^\circ$ panoramas across all input conditions compared to existing methods. Visit the project website at this https URL for results. 

**Abstract (ZH)**: 基于文本或单视角视频数据合成360°视频的novel方法：VideoPanda 

---
# Explicit and Implicit Representations in AI-based 3D Reconstruction for Radiology: A systematic literature review 

**Title (ZH)**: 基于AI的3D重建在放射学中显式和隐式表示：一项系统文献综述 

**Authors**: Yuezhe Yang, Boyu Yang, Yaqian Wang, Yang He, Xingbo Dong, Zhe Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.11349)  

**Abstract**: The demand for high-quality medical imaging in clinical practice and assisted diagnosis has made 3D reconstruction in radiological imaging a key research focus. Artificial intelligence (AI) has emerged as a promising approach to enhancing reconstruction accuracy while reducing acquisition and processing time, thereby minimizing patient radiation exposure and discomfort and ultimately benefiting clinical diagnosis. This review explores state-of-the-art AI-based 3D reconstruction algorithms in radiological imaging, categorizing them into explicit and implicit approaches based on their underlying principles. Explicit methods include point-based, volume-based, and Gaussian representations, while implicit methods encompass implicit prior embedding and neural radiance fields. Additionally, we examine commonly used evaluation metrics and benchmark datasets. Finally, we discuss the current state of development, key challenges, and future research directions in this evolving field. Our project available on: this https URL. 

**Abstract (ZH)**: 高质医学成像在临床实践和辅助诊断中的需求使得放射影像三维重建成为关键研究方向。人工智能（AI）作为一种有望提高重建精度、减少获取和处理时间的方法，从而减少患者辐射暴露和不适，最终有利于临床诊断。本文综述了基于AI的放射影像三维重建最新算法，基于其基本原理将其分类为显式方法和隐式方法。显式方法包括基于点、基于体素和高斯表示法，而隐式方法包括隐式先验嵌入和神经辐射场。此外，本文还探讨了常用评估指标和基准数据集。最后，本文讨论了该领域当前的发展状况、关键挑战及未来研究方向。我们的项目可在以下链接获取：this https URL。 

---
# CFIS-YOLO: A Lightweight Multi-Scale Fusion Network for Edge-Deployable Wood Defect Detection 

**Title (ZH)**: CFIS-YOLO：一种适用于边缘部署的多尺度融合轻量级木材缺陷检测网络 

**Authors**: Jincheng Kang, Yi Cen, Yigang Cen, Ke Wang, Yuhan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11305)  

**Abstract**: Wood defect detection is critical for ensuring quality control in the wood processing industry. However, current industrial applications face two major challenges: traditional methods are costly, subjective, and labor-intensive, while mainstream deep learning models often struggle to balance detection accuracy and computational efficiency for edge deployment. To address these issues, this study proposes CFIS-YOLO, a lightweight object detection model optimized for edge devices. The model introduces an enhanced C2f structure, a dynamic feature recombination module, and a novel loss function that incorporates auxiliary bounding boxes and angular constraints. These innovations improve multi-scale feature fusion and small object localization while significantly reducing computational overhead. Evaluated on a public wood defect dataset, CFIS-YOLO achieves a mean Average Precision (mAP@0.5) of 77.5\%, outperforming the baseline YOLOv10s by 4 percentage points. On SOPHON BM1684X edge devices, CFIS-YOLO delivers 135 FPS, reduces power consumption to 17.3\% of the original implementation, and incurs only a 0.5 percentage point drop in mAP. These results demonstrate that CFIS-YOLO is a practical and effective solution for real-world wood defect detection in resource-constrained environments. 

**Abstract (ZH)**: 木材缺陷检测对于确保木制品加工业的质量控制至关重要。然而，当前工业应用面临两大挑战：传统方法成本高、主观性强且劳动密集；主流的深度学习模型往往难以在保持检测准确率的同时兼顾计算效率以适用于边缘部署。为解决这些问题，本研究提出了一种针对边缘设备优化的轻量化目标检测模型CFIS-YOLO。该模型引入了增强的C2f结构、动态特征重组模块以及新的损失函数，该损失函数结合了辅助边界框和角度约束。这些创新提高了多尺度特征融合和小目标定位能力，显著减少了计算开销。在公开的木材缺陷数据集上，CFIS-YOLO的mAP@0.5达到77.5%，比基准YOLOv10s高出4个百分点。在SOPHON BM1684X边缘设备上，CFIS-YOLO实现了135 FPS，将功耗降低至原实现的17.3%，同时mAP仅有0.5个百分点的下降。这些结果表明，CFIS-YOLO是适用于资源受限环境的实际有效解决方案。 

---
# Diversity-Driven Learning: Tackling Spurious Correlations and Data Heterogeneity in Federated Models 

**Title (ZH)**: 多样性驱动的学习：应对联邦模型中的假相关和数据异质性 

**Authors**: Gergely D. Németh, Eros Fanì, Yeat Jeng Ng, Barbara Caputo, Miguel Ángel Lozano, Nuria Oliver, Novi Quadrianto  

**Link**: [PDF](https://arxiv.org/pdf/2504.11216)  

**Abstract**: Federated Learning (FL) enables decentralized training of machine learning models on distributed data while preserving privacy. However, in real-world FL settings, client data is often non-identically distributed and imbalanced, resulting in statistical data heterogeneity which impacts the generalization capabilities of the server's model across clients, slows convergence and reduces performance. In this paper, we address this challenge by first proposing a characterization of statistical data heterogeneity by means of 6 metrics of global and client attribute imbalance, class imbalance, and spurious correlations. Next, we create and share 7 computer vision datasets for binary and multiclass image classification tasks in Federated Learning that cover a broad range of statistical data heterogeneity and hence simulate real-world situations. Finally, we propose FedDiverse, a novel client selection algorithm in FL which is designed to manage and leverage data heterogeneity across clients by promoting collaboration between clients with complementary data distributions. Experiments on the seven proposed FL datasets demonstrate FedDiverse's effectiveness in enhancing the performance and robustness of a variety of FL methods while having low communication and computational overhead. 

**Abstract (ZH)**: 联邦学习（FL）通过在分布式数据上进行去中心化训练来保持隐私，同时训练机器学习模型。然而，在实际的FL设置中，客户端数据往往是非同分布且不平衡的，导致统计数据异质性，影响服务器模型在客户端之间的泛化能力，减缓收敛速度并降低性能。本文通过首先提出使用6个全局和客户端属性不平衡、类别不平衡以及假相关性的指标来表征统计数据异质性，接着创建并共享了7个用于二分类和多分类图像识别任务的联邦学习数据集，模拟了广泛的统计数据异质性情况，最后提出了FedDiverse，一种用于联邦学习的新型客户端选择算法，通过促进具有互补数据分布的客户端之间的协作来管理和利用数据异质性。实验结果表明，FedDiverse在多种FL方法中提高了性能和 robustness，同时具有较低的通信和计算开销。 

---
# DMAGaze: Gaze Estimation Based on Feature Disentanglement and Multi-Scale Attention 

**Title (ZH)**: DMAGaze：基于特征解缠和多尺度注意力的眼球估计 

**Authors**: Haohan Chen, Hongjia Liu, Shiyong Lan, Wenwu Wang, Yixin Qiao, Yao Li, Guonan Deng  

**Link**: [PDF](https://arxiv.org/pdf/2504.11160)  

**Abstract**: Gaze estimation, which predicts gaze direction, commonly faces the challenge of interference from complex gaze-irrelevant information in face images. In this work, we propose DMAGaze, a novel gaze estimation framework that exploits information from facial images in three aspects: gaze-relevant global features (disentangled from facial image), local eye features (extracted from cropped eye patch), and head pose estimation features, to improve overall performance. Firstly, we design a new continuous mask-based Disentangler to accurately disentangle gaze-relevant and gaze-irrelevant information in facial images by achieving the dual-branch disentanglement goal through separately reconstructing the eye and non-eye regions. Furthermore, we introduce a new cascaded attention module named Multi-Scale Global Local Attention Module (MS-GLAM). Through a customized cascaded attention structure, it effectively focuses on global and local information at multiple scales, further enhancing the information from the Disentangler. Finally, the global gaze-relevant features disentangled by the upper face branch, combined with head pose and local eye features, are passed through the detection head for high-precision gaze estimation. Our proposed DMAGaze has been extensively validated on two mainstream public datasets, achieving state-of-the-art performance. 

**Abstract (ZH)**: DMAGaze：一种多视角解耦的眼球追踪框架 

---
# GATE3D: Generalized Attention-based Task-synergized Estimation in 3D* 

**Title (ZH)**: GATE3D: 基于通用注意力的任务协同三维估计 

**Authors**: Eunsoo Im, Jung Kwon Lee, Changhyun Jee  

**Link**: [PDF](https://arxiv.org/pdf/2504.11014)  

**Abstract**: The emerging trend in computer vision emphasizes developing universal models capable of simultaneously addressing multiple diverse tasks. Such universality typically requires joint training across multi-domain datasets to ensure effective generalization. However, monocular 3D object detection presents unique challenges in multi-domain training due to the scarcity of datasets annotated with accurate 3D ground-truth labels, especially beyond typical road-based autonomous driving contexts. To address this challenge, we introduce a novel weakly supervised framework leveraging pseudo-labels. Current pretrained models often struggle to accurately detect pedestrians in non-road environments due to inherent dataset biases. Unlike generalized image-based 2D object detection models, achieving similar generalization in monocular 3D detection remains largely unexplored. In this paper, we propose GATE3D, a novel framework designed specifically for generalized monocular 3D object detection via weak supervision. GATE3D effectively bridges domain gaps by employing consistency losses between 2D and 3D predictions. Remarkably, our model achieves competitive performance on the KITTI benchmark as well as on an indoor-office dataset collected by us to evaluate the generalization capabilities of our framework. Our results demonstrate that GATE3D significantly accelerates learning from limited annotated data through effective pre-training strategies, highlighting substantial potential for broader impacts in robotics, augmented reality, and virtual reality applications. Project page: this https URL 

**Abstract (ZH)**: 新兴的计算机视觉趋势强调开发能够同时处理多种多样任务的通用模型。这种通用性通常需要跨多域数据集进行联合训练，以确保有效的泛化能力。然而，单目三维物体检测在多域训练中面临独特挑战，主要原因是在非道路环境中的准确三维地面真值标签数据集十分稀缺。为解决这一挑战，我们提出了一种利用伪标签的新颖弱监督框架。当前的预训练模型在非道路环境中的行人检测方面往往表现不佳，这是因为数据集中的偏见问题。与通用的基于图像的二维对象检测模型不同，在单目三维检测中实现类似的泛化能力仍然尚未得到充分探索。在本文中，我们提出了GATE3D，这是一种专门用于通过弱监督进行通用单目三维物体检测的新型框架。GATE3D 通过在二维和三维预测之间使用一致性损失有效地弥合了领域差距。令人惊讶的是，我们的模型在KITTIData基准测试以及我们在室内办公环境收集的数据集上均取得了竞争力的表现，用于评估我们框架的泛化能力。我们的结果表明，GATE3D 通过有效的预训练策略显著加速了从有限标注数据的学习过程，突显了其在机器人、增强现实和虚拟现实应用中的广泛应用潜力。项目页面: [this URL](this https URL) 

---
# MediSee: Reasoning-based Pixel-level Perception in Medical Images 

**Title (ZH)**: MediSee: 基于推理的医学图像像素级感知 

**Authors**: Qinyue Tong, Ziqian Lu, Jun Liu, Yangming Zheng, Zheming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11008)  

**Abstract**: Despite remarkable advancements in pixel-level medical image perception, existing methods are either limited to specific tasks or heavily rely on accurate bounding boxes or text labels as input prompts. However, the medical knowledge required for input is a huge obstacle for general public, which greatly reduces the universality of these methods. Compared with these domain-specialized auxiliary information, general users tend to rely on oral queries that require logical reasoning. In this paper, we introduce a novel medical vision task: Medical Reasoning Segmentation and Detection (MedSD), which aims to comprehend implicit queries about medical images and generate the corresponding segmentation mask and bounding box for the target object. To accomplish this task, we first introduce a Multi-perspective, Logic-driven Medical Reasoning Segmentation and Detection (MLMR-SD) dataset, which encompasses a substantial collection of medical entity targets along with their corresponding reasoning. Furthermore, we propose MediSee, an effective baseline model designed for medical reasoning segmentation and detection. The experimental results indicate that the proposed method can effectively address MedSD with implicit colloquial queries and outperform traditional medical referring segmentation methods. 

**Abstract (ZH)**: 基于逻辑推理的医疗图像分割与检测（MedSD） 

---
# TMCIR: Token Merge Benefits Composed Image Retrieval 

**Title (ZH)**: TMCIR: 词元合并优化组合图像检索 

**Authors**: Chaoyang Wang, Zeyu Zhang, Long Teng, Zijun Li, Shichao Kan  

**Link**: [PDF](https://arxiv.org/pdf/2504.10995)  

**Abstract**: Composed Image Retrieval (CIR) retrieves target images using a multi-modal query that combines a reference image with text describing desired modifications. The primary challenge is effectively fusing this visual and textual information. Current cross-modal feature fusion approaches for CIR exhibit an inherent bias in intention interpretation. These methods tend to disproportionately emphasize either the reference image features (visual-dominant fusion) or the textual modification intent (text-dominant fusion through image-to-text conversion). Such an imbalanced representation often fails to accurately capture and reflect the actual search intent of the user in the retrieval results. To address this challenge, we propose TMCIR, a novel framework that advances composed image retrieval through two key innovations: 1) Intent-Aware Cross-Modal Alignment. We first fine-tune CLIP encoders contrastively using intent-reflecting pseudo-target images, synthesized from reference images and textual descriptions via a diffusion model. This step enhances the encoder ability of text to capture nuanced intents in textual descriptions. 2) Adaptive Token Fusion. We further fine-tune all encoders contrastively by comparing adaptive token-fusion features with the target image. This mechanism dynamically balances visual and textual representations within the contrastive learning pipeline, optimizing the composed feature for retrieval. Extensive experiments on Fashion-IQ and CIRR datasets demonstrate that TMCIR significantly outperforms state-of-the-art methods, particularly in capturing nuanced user intent. 

**Abstract (ZH)**: 综合图像检索（CIR）使用结合参考图像和描述所需修改的文本的多模态查询来检索目标图像。主要挑战在于有效地融合这种视觉和文本信息。当前的跨模态特征融合方法在意图解释上存在固有的偏见。这些方法往往过度强调参考图像特征（视觉主导融合）或通过图像到文本转换的文本修改意图（文本主导融合）。这种不平衡的表示往往无法准确捕捉和反映用户的实际检索意图。为了解决这一挑战，我们提出了一种名为TMCIR的新框架，通过两项创新推进了综合图像检索：1）意图感知的跨模态对齐。我们首先使用从参考图像和文本描述通过扩散模型合成的反映意图的伪目标图像，以对比的方式微调CLIP编码器。这一步骤增强了编码器捕捉文本描述中细微意图的能力。2）自适应 token 融合。我们进一步通过将自适应 token 融合特征与目标图像进行对比来对比微调所有编码器。这种机制在对比学习管道中动态平衡视觉和文本表示，优化组合特征以进行检索。在 Fashion-IQ 和 CIRR 数据集上的广泛实验表明，TMCIR 显著优于当前最先进的方法，尤其是在捕捉细微用户意图方面。 

---
# Bringing together invertible UNets with invertible attention modules for memory-efficient diffusion models 

**Title (ZH)**: 将可逆UNet与可逆注意力模块结合以实现_MEMORY-EFFICIENT_扩散模型 

**Authors**: Karan Jain, Mohammad Nayeem Teli  

**Link**: [PDF](https://arxiv.org/pdf/2504.10883)  

**Abstract**: Diffusion models have recently gained state of the art performance on many image generation tasks. However, most models require significant computational resources to achieve this. This becomes apparent in the application of medical image synthesis due to the 3D nature of medical datasets like CT-scans, MRIs, electron microscope, etc. In this paper we propose a novel architecture for a single GPU memory-efficient training for diffusion models for high dimensional medical datasets. The proposed model is built by using an invertible UNet architecture with invertible attention modules. This leads to the following two contributions: 1. denoising diffusion models and thus enabling memory usage to be independent of the dimensionality of the dataset, and 2. reducing the energy usage during training. While this new model can be applied to a multitude of image generation tasks, we showcase its memory-efficiency on the 3D BraTS2020 dataset leading to up to 15\% decrease in peak memory consumption during training with comparable results to SOTA while maintaining the image quality. 

**Abstract (ZH)**: 基于单GPU内存高效训练的 invertible UNet 架构在高维医疗图像生成中的应用 

---
# Can Vision-Language Models Understand and Interpret Dynamic Gestures from Pedestrians? Pilot Datasets and Exploration Towards Instructive Nonverbal Commands for Cooperative Autonomous Vehicles 

**Title (ZH)**: 视觉-语言模型能否理解并解释行人动态手势？试点数据集及向配合型自主车辆指示性非言语命令的探索 

**Authors**: Tonko E. W. Bossen, Andreas Møgelmose, Ross Greer  

**Link**: [PDF](https://arxiv.org/pdf/2504.10873)  

**Abstract**: In autonomous driving, it is crucial to correctly interpret traffic gestures (TGs), such as those of an authority figure providing orders or instructions, or a pedestrian signaling the driver, to ensure a safe and pleasant traffic environment for all road users. This study investigates the capabilities of state-of-the-art vision-language models (VLMs) in zero-shot interpretation, focusing on their ability to caption and classify human gestures in traffic contexts. We create and publicly share two custom datasets with varying formal and informal TGs, such as 'Stop', 'Reverse', 'Hail', etc. The datasets are "Acted TG (ATG)" and "Instructive TG In-The-Wild (ITGI)". They are annotated with natural language, describing the pedestrian's body position and gesture. We evaluate models using three methods utilizing expert-generated captions as baseline and control: (1) caption similarity, (2) gesture classification, and (3) pose sequence reconstruction similarity. Results show that current VLMs struggle with gesture understanding: sentence similarity averages below 0.59, and classification F1 scores reach only 0.14-0.39, well below the expert baseline of 0.70. While pose reconstruction shows potential, it requires more data and refined metrics to be reliable. Our findings reveal that although some SOTA VLMs can interpret zero-shot human traffic gestures, none are accurate and robust enough to be trustworthy, emphasizing the need for further research in this domain. 

**Abstract (ZH)**: 在自主驾驶中，正确解释交通手势（TGs），如权威人物的指令或行人的驾驶信号，对于确保所有道路使用者的安全和愉快交通环境至关重要。本研究探讨了最先进的视觉-语言模型（VLMs）在零样本解释中的能力，重点在于其在交通场景中描述和分类人类手势的能力。我们创建并公开分享了两个自定义数据集，包含正式和非正式的交通手势，如“停止”、“倒车”、“招手”等。这些数据集分别为“行为交通手势（ATG）”和“野生指导性交通手势（ITGI）”。它们用自然语言标注了行人的身体位置和手势。我们使用三种方法评估模型，利用专家生成的描述作为基线和对照：（1）句子相似度，（2）手势分类，（3）姿态序列重构相似度。结果表明，当前的VLMs在手势理解方面存在困难：句子相似度平均值低于0.59，分类F1分数仅为0.14-0.39，远低于专家基线0.70。尽管姿态重建显示出潜力，但需要更多数据和精细的评估指标才能可靠。我们的研究发现，尽管一些最先进的VLMs能够理解和解释零样本的人类交通手势，但没有一种模型既准确又足够可靠，这强调了在此领域进一步研究的必要性。 

---
# PatrolVision: Automated License Plate Recognition in the wild 

**Title (ZH)**: 巡逻视界：野外自动化车牌识别 

**Authors**: Anmol Singhal Navya Singhal  

**Link**: [PDF](https://arxiv.org/pdf/2504.10810)  

**Abstract**: Adoption of AI driven techniques in public services remains low due to challenges related to accuracy and speed of information at population scale. Computer vision techniques for traffic monitoring have not gained much popularity despite their relative strength in areas such as autonomous driving. Despite large number of academic methods for Automatic License Plate Recognition (ALPR) systems, very few provide an end to end solution for patrolling in the city. This paper presents a novel prototype for a low power GPU based patrolling system to be deployed in an urban environment on surveillance vehicles for automated vehicle detection, recognition and tracking. In this work, we propose a complete ALPR system for Singapore license plates having both single and double line creating our own YOLO based network. We focus on unconstrained capture scenarios as would be the case in real world application, where the license plate (LP) might be considerably distorted due to oblique views. In this work, we first detect the license plate from the full image using RFB-Net and rectify multiple distorted license plates in a single image. After that, the detected license plate image is fed to our network for character recognition. We evaluate the performance of our proposed system on a newly built dataset covering more than 16,000 images. The system was able to correctly detect license plates with 86\% precision and recognize characters of a license plate in 67\% of the test set, and 89\% accuracy with one incorrect character (partial match). We also test latency of our system and achieve 64FPS on Tesla P4 GPU 

**Abstract (ZH)**: 基于GPU的低功耗巡逻系统在城市环境中的自动车辆检测、识别与跟踪技术研究 

---
# Visual anemometry of natural vegetation from their leaf motion 

**Title (ZH)**: 自然植被叶片运动的视觉风速测量 

**Authors**: Roni H. Goldshmid, John O. Dabiri, John E. Sader  

**Link**: [PDF](https://arxiv.org/pdf/2504.10584)  

**Abstract**: High-resolution, near-ground wind-speed data are critical for improving the accuracy of weather predictions and climate models,$^{1-3}$ supporting wildfire control efforts,$^{4-7}$ and ensuring the safe passage of airplanes during takeoff and landing maneouvers.$^{8,9}$ Quantitative wind speed anemometry generally employs on-site instrumentation for accurate single-position data or sophisticated remote techniques such as Doppler radar for quantitative field measurements. It is widely recognized that the wind-induced motion of vegetation depends in a complex manner on their structure and mechanical properties, obviating their use in quantitative anemometry.$^{10-14}$ We analyze measurements on a host of different vegetation showing that leaf motion can be decoupled from the leaf's branch and support structure, at low-to-moderate wind speed, $U_{wind}$. This wind speed range is characterized by a leaf Reynolds number, enabling the development of a remote, quantitative anemometry method based on the formula, $U_{wind}\approx740\sqrt{{\mu}U_{leaf}/{\rho}D}$, that relies only on the leaf size $D$, its measured fluctuating (RMS) speed $U_{leaf}$, the air viscosity $\mu$, and its mass density $\rho$. This formula is corroborated by a first-principles model and validated using a host of laboratory and field tests on diverse vegetation types, ranging from oak, olive, and magnolia trees through to camphor and bullgrass. The findings of this study open the door to a new paradigm in anemometry, using natural vegetation to enable remote and rapid quantitative field measurements at global locations with minimal cost. 

**Abstract (ZH)**: 高分辨率近地风速数据对于提高天气预测和气候模型的准确性至关重要，支持野火控制努力，并确保飞机在起飞和降落时的安全。基于叶片运动的远程定量风速仪方法：一种新的基于自然植被的远程快速定量场测量方法。 

---
# AB-Cache: Training-Free Acceleration of Diffusion Models via Adams-Bashforth Cached Feature Reuse 

**Title (ZH)**: AB-Cache: 不需训练的扩散模型加速通过Adams-Bashforth缓存特征重用 

**Authors**: Zichao Yu, Zhen Zou, Guojiang Shao, Chengwei Zhang, Shengze Xu, Jie Huang, Feng Zhao, Xiaodong Cun, Wenyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10540)  

**Abstract**: Diffusion models have demonstrated remarkable success in generative tasks, yet their iterative denoising process results in slow inference, limiting their practicality. While existing acceleration methods exploit the well-known U-shaped similarity pattern between adjacent steps through caching mechanisms, they lack theoretical foundation and rely on simplistic computation reuse, often leading to performance degradation. In this work, we provide a theoretical understanding by analyzing the denoising process through the second-order Adams-Bashforth method, revealing a linear relationship between the outputs of consecutive steps. This analysis explains why the outputs of adjacent steps exhibit a U-shaped pattern. Furthermore, extending Adams-Bashforth method to higher order, we propose a novel caching-based acceleration approach for diffusion models, instead of directly reusing cached results, with a truncation error bound of only \(O(h^k)\) where $h$ is the step size. Extensive validation across diverse image and video diffusion models (including HunyuanVideo and FLUX.1-dev) with various schedulers demonstrates our method's effectiveness in achieving nearly $3\times$ speedup while maintaining original performance levels, offering a practical real-time solution without compromising generation quality. 

**Abstract (ZH)**: 扩散模型在生成任务中展现了显著的成功，但其迭代去噪过程导致推断缓慢，限制了其实用性。虽然现有的加速方法通过缓存机制利用了相邻步骤间已知的U形相似模式，但缺乏理论基础，依赖于简单的计算复用，经常会降低性能。在本工作中，我们通过分析连续步骤之间的输出关系，利用第二阶阿德ams- Bashforth方法提供了一个理论理解，揭示了连续步骤输出间的线性关系。这种分析解释了相邻步骤输出为何呈现U形模式。此外，将阿德ams-巴斯福德方法扩展到更高阶，我们提出了一种基于缓存的新型加速方法，而不是直接复用缓存结果，仅带有限截断误差边界\(O(h^k)\)。在不同的图像和视频扩散模型（包括HunyuanVideo和FLUX.1-dev）以及多种调度器上进行的广泛验证表明，我们的方法能够实现接近3倍的速度提升，同时保持原始性能水平，提供了一种在不牺牲生成质量的情况下实现实时处理的实用解决方案。 

---
# Focal Loss based Residual Convolutional Neural Network for Speech Emotion Recognition 

**Title (ZH)**: 基于焦点损失的残差卷积神经网络在语音情绪识别中的应用 

**Authors**: Suraj Tripathi, Abhay Kumar, Abhiram Ramesh, Chirag Singh, Promod Yenigalla  

**Link**: [PDF](https://arxiv.org/pdf/1906.05682)  

**Abstract**: This paper proposes a Residual Convolutional Neural Network (ResNet) based on speech features and trained under Focal Loss to recognize emotion in speech. Speech features such as Spectrogram and Mel-frequency Cepstral Coefficients (MFCCs) have shown the ability to characterize emotion better than just plain text. Further Focal Loss, first used in One-Stage Object Detectors, has shown the ability to focus the training process more towards hard-examples and down-weight the loss assigned to well-classified examples, thus preventing the model from being overwhelmed by easily classifiable examples. 

**Abstract (ZH)**: 基于语音特征和焦损失的残差卷积神经网络情感识别研究成果 

---
