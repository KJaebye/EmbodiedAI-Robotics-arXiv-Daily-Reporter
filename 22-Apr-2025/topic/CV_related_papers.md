# SuFIA-BC: Generating High Quality Demonstration Data for Visuomotor Policy Learning in Surgical Subtasks 

**Title (ZH)**: SuFIA-BC：为手术子任务的视运动策略学习生成高质量示范数据 

**Authors**: Masoud Moghani, Nigel Nelson, Mohamed Ghanem, Andres Diaz-Pinto, Kush Hari, Mahdi Azizian, Ken Goldberg, Sean Huver, Animesh Garg  

**Link**: [PDF](https://arxiv.org/pdf/2504.14857)  

**Abstract**: Behavior cloning facilitates the learning of dexterous manipulation skills, yet the complexity of surgical environments, the difficulty and expense of obtaining patient data, and robot calibration errors present unique challenges for surgical robot learning. We provide an enhanced surgical digital twin with photorealistic human anatomical organs, integrated into a comprehensive simulator designed to generate high-quality synthetic data to solve fundamental tasks in surgical autonomy. We present SuFIA-BC: visual Behavior Cloning policies for Surgical First Interactive Autonomy Assistants. We investigate visual observation spaces including multi-view cameras and 3D visual representations extracted from a single endoscopic camera view. Through systematic evaluation, we find that the diverse set of photorealistic surgical tasks introduced in this work enables a comprehensive evaluation of prospective behavior cloning models for the unique challenges posed by surgical environments. We observe that current state-of-the-art behavior cloning techniques struggle to solve the contact-rich and complex tasks evaluated in this work, regardless of their underlying perception or control architectures. These findings highlight the importance of customizing perception pipelines and control architectures, as well as curating larger-scale synthetic datasets that meet the specific demands of surgical tasks. Project website: this https URL 

**Abstract (ZH)**: 行为 cloning 有助于灵巧操作技能的学习，但在手术环境中复杂的任务需求、获取患者数据的难度和成本以及机器人校准误差为手术机器人学习带来了独特挑战。我们提供一种增强的手术数字孪生体，集成具有超高逼真度的人体解剖器官，并结合了一个综合模拟器，以生成高质量的合成数据以解决手术自主性的基本任务。我们提出了 SuFIA-BC：用于手术首次互动自主助理的视觉行为克隆策略。我们研究了包括多视角摄像头和从单个内窥镜相机视角提取的三维视觉表示在内的视觉观察空间。通过系统的评估，我们发现本工作中引入的多样的逼真手术任务能够全面评估行为克隆模型在手术环境中面临的独特挑战。我们观察到，当前最先进的行为克隆技术在解决本研究中评估的接触丰富且复杂的任务时存在困难，不考虑其底层的感知或控制架构。这些发现突显了定制感知管道和控制架构的重要性，以及编目规模更大的满足特定手术任务需求的合成数据集的重要性。项目网站：this https URL 

---
# Accelerating Visual Reinforcement Learning with Separate Primitive Policy for Peg-in-Hole Tasks 

**Title (ZH)**: 基于独立基本策略加速视觉强化学习：以孔插针任务为例 

**Authors**: Zichun Xu, Zhaomin Wang, Yuntao Li, Lei Zhuang, Zhiyuan Zhao, Guocai Yang, Jingdong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.14820)  

**Abstract**: For peg-in-hole tasks, humans rely on binocular visual perception to locate the peg above the hole surface and then proceed with insertion. This paper draws insights from this behavior to enable agents to learn efficient assembly strategies through visual reinforcement learning. Hence, we propose a Separate Primitive Policy (S2P) to simultaneously learn how to derive location and insertion actions. S2P is compatible with model-free reinforcement learning algorithms. Ten insertion tasks featuring different polygons are developed as benchmarks for evaluations. Simulation experiments show that S2P can boost the sample efficiency and success rate even with force constraints. Real-world experiments are also performed to verify the feasibility of S2P. Ablations are finally given to discuss the generalizability of S2P and some factors that affect its performance. 

**Abstract (ZH)**: 基于 peg-in-hole 任务，人类依赖双眼视觉感知来定位 peg 上方的孔位，然后进行插入。本文从这一行为中获得灵感，通过视觉强化学习使代理学习高效的装配策略。因此，我们提出了一种分离基础政策（S2P）来同时学习如何获取定位和插入动作。S2P 兼容无模型的强化学习算法。开发了包含不同多边形的十种插入任务作为评估基准。仿真实验表明，即使在力约束下，S2P 也能提高样本效率和成功率。我们也进行了实际实验验证 S2P 的可行性。最后给出了消融实验以讨论 S2P 的泛化能力和影响其性能的一些因素。 

---
# An Iterative Task-Driven Framework for Resilient LiDAR Place Recognition in Adverse Weather 

**Title (ZH)**: 一种用于恶劣天气条件下的 resilient LiDAR 地点识别的迭代任务驱动框架 

**Authors**: Xiongwei Zhao, Yang Wang, Qihao Sun, Haojie Bai, Xingxiang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.14806)  

**Abstract**: LiDAR place recognition (LPR) plays a vital role in autonomous navigation. However, existing LPR methods struggle to maintain robustness under adverse weather conditions such as rain, snow, and fog, where weather-induced noise and point cloud degradation impair LiDAR reliability and perception accuracy. To tackle these challenges, we propose an Iterative Task-Driven Framework (ITDNet), which integrates a LiDAR Data Restoration (LDR) module and a LiDAR Place Recognition (LPR) module through an iterative learning strategy. These modules are jointly trained end-to-end, with alternating optimization to enhance performance. The core rationale of ITDNet is to leverage the LDR module to recover the corrupted point clouds while preserving structural consistency with clean data, thereby improving LPR accuracy in adverse weather. Simultaneously, the LPR task provides feature pseudo-labels to guide the LDR module's training, aligning it more effectively with the LPR task. To achieve this, we first design a task-driven LPR loss and a reconstruction loss to jointly supervise the optimization of the LDR module. Furthermore, for the LDR module, we propose a Dual-Domain Mixer (DDM) block for frequency-spatial feature fusion and a Semantic-Aware Generator (SAG) block for semantic-guided restoration. In addition, for the LPR module, we introduce a Multi-Frequency Transformer (MFT) block and a Wavelet Pyramid NetVLAD (WPN) block to aggregate multi-scale, robust global descriptors. Finally, extensive experiments on the Weather-KITTI, Boreas, and our proposed Weather-Apollo datasets demonstrate that, demonstrate that ITDNet outperforms existing LPR methods, achieving state-of-the-art performance in adverse weather. The datasets and code will be made publicly available at this https URL. 

**Abstract (ZH)**: 基于迭代任务驱动框架的LiDAR位置识别方法研究：应对恶劣天气条件下的鲁棒性挑战 

---
# DRAWER: Digital Reconstruction and Articulation With Environment Realism 

**Title (ZH)**: DRAWER: 数字重建与环境真实感articulation 

**Authors**: Hongchi Xia, Entong Su, Marius Memmel, Arhan Jain, Raymond Yu, Numfor Mbiziwo-Tiapo, Ali Farhadi, Abhishek Gupta, Shenlong Wang, Wei-Chiu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.15278)  

**Abstract**: Creating virtual digital replicas from real-world data unlocks significant potential across domains like gaming and robotics. In this paper, we present DRAWER, a novel framework that converts a video of a static indoor scene into a photorealistic and interactive digital environment. Our approach centers on two main contributions: (i) a reconstruction module based on a dual scene representation that reconstructs the scene with fine-grained geometric details, and (ii) an articulation module that identifies articulation types and hinge positions, reconstructs simulatable shapes and appearances and integrates them into the scene. The resulting virtual environment is photorealistic, interactive, and runs in real time, with compatibility for game engines and robotic simulation platforms. We demonstrate the potential of DRAWER by using it to automatically create an interactive game in Unreal Engine and to enable real-to-sim-to-real transfer for robotics applications. 

**Abstract (ZH)**: 从现实世界数据创建虚拟数字复制品在游戏和机器人等领域展现出巨大潜力。本文提出DRAWER框架，该框架能够将静态室内场景视频转换为照片级真实且可交互的数字环境。我们的方法主要贡献包括：（i）基于双场景表示的重建模块，能够精细重建几何细节；（ii）关节模块，用于识别关节类型和绞点位置，重构可模拟的形状和外观，并将其整合到场景中。所生成的虚拟环境具有照片级真实感、交互性，并可实时运行，兼容游戏引擎和机器人仿真平台。我们通过在Unreal Engine中自动生成互动游戏以及支持从真实到模拟再到真实的机器人应用转移，展示了DRAWER的潜力。 

---
# Locate 3D: Real-World Object Localization via Self-Supervised Learning in 3D 

**Title (ZH)**: Locate 3D：通过3D自主学习进行真实世界物体定位 

**Authors**: Sergio Arnaud, Paul McVay, Ada Martin, Arjun Majumdar, Krishna Murthy Jatavallabhula, Phillip Thomas, Ruslan Partsey, Daniel Dugas, Abha Gejji, Alexander Sax, Vincent-Pierre Berges, Mikael Henaff, Ayush Jain, Ang Cao, Ishita Prasad, Mrinal Kalakrishnan, Michael Rabbat, Nicolas Ballas, Mido Assran, Oleksandr Maksymets, Aravind Rajeswaran, Franziska Meier  

**Link**: [PDF](https://arxiv.org/pdf/2504.14151)  

**Abstract**: We present LOCATE 3D, a model for localizing objects in 3D scenes from referring expressions like "the small coffee table between the sofa and the lamp." LOCATE 3D sets a new state-of-the-art on standard referential grounding benchmarks and showcases robust generalization capabilities. Notably, LOCATE 3D operates directly on sensor observation streams (posed RGB-D frames), enabling real-world deployment on robots and AR devices. Key to our approach is 3D-JEPA, a novel self-supervised learning (SSL) algorithm applicable to sensor point clouds. It takes as input a 3D pointcloud featurized using 2D foundation models (CLIP, DINO). Subsequently, masked prediction in latent space is employed as a pretext task to aid the self-supervised learning of contextualized pointcloud features. Once trained, the 3D-JEPA encoder is finetuned alongside a language-conditioned decoder to jointly predict 3D masks and bounding boxes. Additionally, we introduce LOCATE 3D DATASET, a new dataset for 3D referential grounding, spanning multiple capture setups with over 130K annotations. This enables a systematic study of generalization capabilities as well as a stronger model. 

**Abstract (ZH)**: LOCATE 3D：一种基于引用表达在3D场景中定位对象的模型 

---
# Bringing Diversity from Diffusion Models to Semantic-Guided Face Asset Generation 

**Title (ZH)**: 将扩散模型引入语义引导的面部资产生成以增加多样性 

**Authors**: Yunxuan Cai, Sitao Xiang, Zongjian Li, Haiwei Chen, Yajie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15259)  

**Abstract**: Digital modeling and reconstruction of human faces serve various applications. However, its availability is often hindered by the requirements of data capturing devices, manual labor, and suitable actors. This situation restricts the diversity, expressiveness, and control over the resulting models. This work aims to demonstrate that a semantically controllable generative network can provide enhanced control over the digital face modeling process. To enhance diversity beyond the limited human faces scanned in a controlled setting, we introduce a novel data generation pipeline that creates a high-quality 3D face database using a pre-trained diffusion model. Our proposed normalization module converts synthesized data from the diffusion model into high-quality scanned data. Using the 44,000 face models we obtained, we further developed an efficient GAN-based generator. This generator accepts semantic attributes as input, and generates geometry and albedo. It also allows continuous post-editing of attributes in the latent space. Our asset refinement component subsequently creates physically-based facial assets. We introduce a comprehensive system designed for creating and editing high-quality face assets. Our proposed model has undergone extensive experiment, comparison and evaluation. We also integrate everything into a web-based interactive tool. We aim to make this tool publicly available with the release of the paper. 

**Abstract (ZH)**: 基于语义控制的数字人脸建模与重建：增强控制能力的研究 

---
# An Efficient Aerial Image Detection with Variable Receptive Fields 

**Title (ZH)**: 具有可变 receptive fields 的高效航测图像检测 

**Authors**: Liu Wenbin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15165)  

**Abstract**: Aerial object detection using unmanned aerial vehicles (UAVs) faces critical challenges including sub-10px targets, dense occlusions, and stringent computational constraints. Existing detectors struggle to balance accuracy and efficiency due to rigid receptive fields and redundant architectures. To address these limitations, we propose Variable Receptive Field DETR (VRF-DETR), a transformer-based detector incorporating three key components: 1) Multi-Scale Context Fusion (MSCF) module that dynamically recalibrates features through adaptive spatial attention and gated multi-scale fusion, 2) Gated Convolution (GConv) layer enabling parameter-efficient local-context modeling via depthwise separable operations and dynamic gating, and 3) Gated Multi-scale Fusion (GMCF) Bottleneck that hierarchically disentangles occluded objects through cascaded global-local interactions. Experiments on VisDrone2019 demonstrate VRF-DETR achieves 51.4\% mAP\textsubscript{50} and 31.8\% mAP\textsubscript{50:95} with only 13.5M parameters. This work establishes a new efficiency-accuracy Pareto frontier for UAV-based detection tasks. 

**Abstract (ZH)**: 基于无人驾驶航空车辆的航空目标检测面临包括亚10像素目标、密集遮挡和严格的计算约束在内的关键挑战。现有检测器由于固有的感受野约束和冗余架构难以在准确性和效率之间取得平衡。为解决这些局限性，我们提出了可变感受野DETR（VRF-DETR），这是一种基于变压器的检测器，包含三个关键组件：1）多尺度上下文融合（MSCF）模块，通过自适应空间注意力和门控多尺度融合动态校准特征；2）门控卷积（GConv）层，通过深度可分离操作和动态门控实现参数高效的地方上下文建模；3）门控多尺度融合（GMCF）瓶颈，通过级联的全局-局部交互逐级解开遮挡物体。在VisDrone2019数据集上的实验表明，VRF-DETR仅使用13.5M参数便实现了51.4%的mAP50和31.8%的mAP50:95。这项工作为基于无人驾驶航空车辆的检测任务建立了新的效率-准确性的帕累托前沿。 

---
# Landmark-Free Preoperative-to-Intraoperative Registration in Laparoscopic Liver Resection 

**Title (ZH)**: 基于腹腔镜肝切除术的无 Landmark 预手术至术中配准 

**Authors**: Jun Zhou, Bingchen Gao, Kai Wang, Jialun Pei, Pheng-Ann Heng, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15152)  

**Abstract**: Liver registration by overlaying preoperative 3D models onto intraoperative 2D frames can assist surgeons in perceiving the spatial anatomy of the liver clearly for a higher surgical success rate. Existing registration methods rely heavily on anatomical landmark-based workflows, which encounter two major limitations: 1) ambiguous landmark definitions fail to provide efficient markers for registration; 2) insufficient integration of intraoperative liver visual information in shape deformation modeling. To address these challenges, in this paper, we propose a landmark-free preoperative-to-intraoperative registration framework utilizing effective self-supervised learning, termed \ourmodel. This framework transforms the conventional 3D-2D workflow into a 3D-3D registration pipeline, which is then decoupled into rigid and non-rigid registration subtasks. \ourmodel~first introduces a feature-disentangled transformer to learn robust correspondences for recovering rigid transformations. Further, a structure-regularized deformation network is designed to adjust the preoperative model to align with the intraoperative liver surface. This network captures structural correlations through geometry similarity modeling in a low-rank transformer network. To facilitate the validation of the registration performance, we also construct an in-vivo registration dataset containing liver resection videos of 21 patients, called \emph{P2I-LReg}, which contains 346 keyframes that provide a global view of the liver together with liver mask annotations and calibrated camera intrinsic parameters. Extensive experiments and user studies on both synthetic and in-vivo datasets demonstrate the superiority and potential clinical applicability of our method. 

**Abstract (ZH)**: 基于自监督学习的无 landmark 术前到术中肝注册框架 

---
# A triple-branch network for latent fingerprint enhancement guided by orientation fields and minutiae 

**Title (ZH)**: 基于方向场和细节指导的三支路网络用于潜在指纹增强 

**Authors**: Yurun Wang, Zerong Qi, Shujun Fu, Mingzheng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15105)  

**Abstract**: Latent fingerprint enhancement is a critical step in the process of latent fingerprint identification. Existing deep learning-based enhancement methods still fall short of practical application requirements, particularly in restoring low-quality fingerprint regions. Recognizing that different regions of latent fingerprints require distinct enhancement strategies, we propose a Triple Branch Spatial Fusion Network (TBSFNet), which simultaneously enhances different regions of the image using tailored strategies. Furthermore, to improve the generalization capability of the network, we integrate orientation field and minutiae-related modules into TBSFNet and introduce a Multi-Level Feature Guidance Network (MLFGNet). Experimental results on the MOLF and MUST datasets demonstrate that MLFGNet outperforms existing enhancement algorithms. 

**Abstract (ZH)**: latent指纹增强是latent指纹识别过程中的关键步骤。现有的基于深度学习的增强方法仍未能满足实际应用要求，特别是在恢复低质量指纹区域方面。鉴于latent指纹的不同区域需要不同的增强策略，我们提出了三支路空间融合网络（TBSFNet），该网络使用定制的策略同时增强图像的不同区域。此外，为了提高网络的泛化能力，我们将方向场和细节相关模块集成到TBSFNet中，并引入多层特征引导网络（MLFGNet）。实验结果表明，MLFGNet在MOLF和MUST数据集上的表现优于现有的增强算法。 

---
# Fast Adversarial Training with Weak-to-Strong Spatial-Temporal Consistency in the Frequency Domain on Videos 

**Title (ZH)**: 在频域中具有从弱到强空间-时间一致性的快速对抗训练 

**Authors**: Songping Wang, Hanqing Liu, Yueming Lyu, Xiantao Hu, Ziwen He, Wei Wang, Caifeng Shan, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14921)  

**Abstract**: Adversarial Training (AT) has been shown to significantly enhance adversarial robustness via a min-max optimization approach. However, its effectiveness in video recognition tasks is hampered by two main challenges. First, fast adversarial training for video models remains largely unexplored, which severely impedes its practical applications. Specifically, most video adversarial training methods are computationally costly, with long training times and high expenses. Second, existing methods struggle with the trade-off between clean accuracy and adversarial robustness. To address these challenges, we introduce Video Fast Adversarial Training with Weak-to-Strong consistency (VFAT-WS), the first fast adversarial training method for video data. Specifically, VFAT-WS incorporates the following key designs: First, it integrates a straightforward yet effective temporal frequency augmentation (TF-AUG), and its spatial-temporal enhanced form STF-AUG, along with a single-step PGD attack to boost training efficiency and robustness. Second, it devises a weak-to-strong spatial-temporal consistency regularization, which seamlessly integrates the simpler TF-AUG and the more complex STF-AUG. Leveraging the consistency regularization, it steers the learning process from simple to complex augmentations. Both of them work together to achieve a better trade-off between clean accuracy and robustness. Extensive experiments on UCF-101 and HMDB-51 with both CNN and Transformer-based models demonstrate that VFAT-WS achieves great improvements in adversarial robustness and corruption robustness, while accelerating training by nearly 490%. 

**Abstract (ZH)**: Video Fast Adversarial Training with Weak-to-Strong Consistency (VFAT-WS) 

---
# Object-Level Verbalized Confidence Calibration in Vision-Language Models via Semantic Perturbation 

**Title (ZH)**: 基于语义扰动的视觉-语言模型对象级语义化置信校准 

**Authors**: Yunpu Zhao, Rui Zhang, Junbin Xiao, Ruibo Hou, Jiaming Guo, Zihao Zhang, Yifan Hao, Yunji Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14848)  

**Abstract**: Vision-language models (VLMs) excel in various multimodal tasks but frequently suffer from poor calibration, resulting in misalignment between their verbalized confidence and response correctness. This miscalibration undermines user trust, especially when models confidently provide incorrect or fabricated information. In this work, we propose a novel Confidence Calibration through Semantic Perturbation (CSP) framework to improve the calibration of verbalized confidence for VLMs in response to object-centric queries. We first introduce a perturbed dataset where Gaussian noise is applied to the key object regions to simulate visual uncertainty at different confidence levels, establishing an explicit mapping between visual ambiguity and confidence levels. We further enhance calibration through a two-stage training process combining supervised fine-tuning on the perturbed dataset with subsequent preference optimization. Extensive experiments on popular benchmarks demonstrate that our method significantly improves the alignment between verbalized confidence and response correctness while maintaining or enhancing overall task performance. These results highlight the potential of semantic perturbation as a practical tool for improving the reliability and interpretability of VLMs. 

**Abstract (ZH)**: 视觉语言模型(VLMs)在多模态任务中表现出色，但经常遭受校准不佳的困扰，导致其口头表达的信心与响应的正确性不一致。这种校准不当会削弱用户的信任，尤其是在模型自信地提供错误或虚假信息时。在本文中，我们提出了一种新的基于语义扰动的信心校准(CSP)框架，以提高VLMs对以物体为中心的查询响应中口头表达的信心的校准。我们首先引入了一个扰动数据集，其中在关键物体区域应用高斯噪声以模拟不同信心水平下的视觉不确定性，建立了视觉模糊性和信心水平之间的明确映射。我们进一步通过结合扰动数据集上的监督微调和后续的偏好优化的两阶段训练过程来增强校准。在流行的基准测试上的广泛实验表明，我们的方法显著提高了口头表达的信心与响应正确性之间的对齐，同时保持或提高了整体任务性能。这些结果突显了语义扰动作为提高VLMs可靠性和可解释性的实际工具的潜力。 

---
# ECViT: Efficient Convolutional Vision Transformer with Local-Attention and Multi-scale Stages 

**Title (ZH)**: ECViT: 高效的局部注意和多尺度阶段卷积视觉变换器 

**Authors**: Zhoujie Qian  

**Link**: [PDF](https://arxiv.org/pdf/2504.14825)  

**Abstract**: Vision Transformers (ViTs) have revolutionized computer vision by leveraging self-attention to model long-range dependencies. However, ViTs face challenges such as high computational costs due to the quadratic scaling of self-attention and the requirement of a large amount of training data. To address these limitations, we propose the Efficient Convolutional Vision Transformer (ECViT), a hybrid architecture that effectively combines the strengths of CNNs and Transformers. ECViT introduces inductive biases such as locality and translation invariance, inherent to Convolutional Neural Networks (CNNs) into the Transformer framework by extracting patches from low-level features and enhancing the encoder with convolutional operations. Additionally, it incorporates local-attention and a pyramid structure to enable efficient multi-scale feature extraction and representation. Experimental results demonstrate that ECViT achieves an optimal balance between performance and efficiency, outperforming state-of-the-art models on various image classification tasks while maintaining low computational and storage requirements. ECViT offers an ideal solution for applications that prioritize high efficiency without compromising performance. 

**Abstract (ZH)**: 高效的卷积视觉变换器（ECViT）：结合CNN和Transformer的优势以实现高效性能权衡 

---
# SuperCL: Superpixel Guided Contrastive Learning for Medical Image Segmentation Pre-training 

**Title (ZH)**: SuperCL：基于超像素的对比学习医疗图像分割预训练 

**Authors**: Shuang Zeng, Lei Zhu, Xinliang Zhang, Hangzhou He, Yanye Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14737)  

**Abstract**: Medical image segmentation is a critical yet challenging task, primarily due to the difficulty of obtaining extensive datasets of high-quality, expert-annotated images. Contrastive learning presents a potential but still problematic solution to this issue. Because most existing methods focus on extracting instance-level or pixel-to-pixel representation, which ignores the characteristics between intra-image similar pixel groups. Moreover, when considering contrastive pairs generation, most SOTA methods mainly rely on manually setting thresholds, which requires a large number of gradient experiments and lacks efficiency and generalization. To address these issues, we propose a novel contrastive learning approach named SuperCL for medical image segmentation pre-training. Specifically, our SuperCL exploits the structural prior and pixel correlation of images by introducing two novel contrastive pairs generation strategies: Intra-image Local Contrastive Pairs (ILCP) Generation and Inter-image Global Contrastive Pairs (IGCP) Generation. Considering superpixel cluster aligns well with the concept of contrastive pairs generation, we utilize the superpixel map to generate pseudo masks for both ILCP and IGCP to guide supervised contrastive learning. Moreover, we also propose two modules named Average SuperPixel Feature Map Generation (ASP) and Connected Components Label Generation (CCL) to better exploit the prior structural information for IGCP. Finally, experiments on 8 medical image datasets indicate our SuperCL outperforms existing 12 methods. i.e. Our SuperCL achieves a superior performance with more precise predictions from visualization figures and 3.15%, 5.44%, 7.89% DSC higher than the previous best results on MMWHS, CHAOS, Spleen with 10% annotations. Our code will be released after acceptance. 

**Abstract (ZH)**: 一种用于医学图像分割预训练的新型对比学习方法：SuperCL 

---
# Time Frequency Analysis of EMG Signal for Gesture Recognition using Fine grained Features 

**Title (ZH)**: 基于细粒度特征的EMG信号时频分析手势识别 

**Authors**: Parshuram N. Aarotale, Ajita Rattani  

**Link**: [PDF](https://arxiv.org/pdf/2504.14708)  

**Abstract**: Electromyography (EMG) based hand gesture recognition converts forearm muscle activity into control commands for prosthetics, rehabilitation, and human computer interaction. This paper proposes a novel approach to EMG-based hand gesture recognition that uses fine-grained classification and presents XMANet, which unifies low-level local and high level semantic cues through cross layer mutual attention among shallow to deep CNN experts. Using stacked spectrograms and scalograms derived from the Short Time Fourier Transform (STFT) and Wavelet Transform (WT), we benchmark XMANet against ResNet50, DenseNet-121, MobileNetV3, and EfficientNetB0. Experimental results on the Grabmyo dataset indicate that, using STFT, the proposed XMANet model outperforms the baseline ResNet50, EfficientNetB0, MobileNetV3, and DenseNet121 models with improvement of approximately 1.72%, 4.38%, 5.10%, and 2.53%, respectively. When employing the WT approach, improvements of around 1.57%, 1.88%, 1.46%, and 2.05% are observed over the same baselines. Similarly, on the FORS EMG dataset, the XMANet(ResNet50) model using STFT shows an improvement of about 5.04% over the baseline ResNet50. In comparison, the XMANet(DenseNet121) and XMANet(MobileNetV3) models yield enhancements of approximately 4.11% and 2.81%, respectively. Moreover, when using WT, the proposed XMANet achieves gains of around 4.26%, 9.36%, 5.72%, and 6.09% over the baseline ResNet50, DenseNet121, MobileNetV3, and EfficientNetB0 models, respectively. These results confirm that XMANet consistently improves performance across various architectures and signal processing techniques, demonstrating the strong potential of fine grained features for accurate and robust EMG classification. 

**Abstract (ZH)**: 基于 electromyography (EMG) 的手部手势识别将前臂肌活动转化为假肢控制命令、康复和人机交互的控制指令。本文提出了一种基于 EMG 的手部手势识别的新方法，并介绍了 XMANet，该方法通过浅层到深层 CNN 专家之间的跨层互注意力机制统一了低级局部和高级语义线索。使用短时傅里叶变换（STFT）和小波变换（WT）得到的堆叠频谱图和小波变化图，我们将 XMANet 与 ResNet50、DenseNet-121、MobileNetV3 和 EfficientNetB0 进行基准测试。实验结果表明，使用 STFT 时，提出的 XMANet 模型分别在基准 ResNet50、EfficientNetB0、MobileNetV3 和 DenseNet121 模型上获得了约 1.72%、4.38%、5.10% 和 2.53% 的性能提升。使用 WT 时，分别获得了约 1.57%、1.88%、1.46% 和 2.05% 的性能提升。同样，在 FORS EMG 数据集上，使用 STFT 的 XMANet(ResNet50) 模型相对于基准 ResNet50 模型提升了约 5.04%。相比之下，XMANet(DenseNet121) 和 XMANet(MobileNetV3) 模型分别获得了约 4.11% 和 2.81% 的提升。此外，使用 WT 时，提出的 XMANet 分别相对于基准 ResNet50、DenseNet121、MobileNetV3 和 EfficientNetB0 模型获得了约 4.26%、9.36%、5.72% 和 6.09% 的性能提升。这些结果证实了 XMANet 在不同架构和信号处理技术下 consistently 提高了性能，展示了精细特征在准确且稳健的 EMG 分类中的强大潜力。 

---
# IXGS-Intraoperative 3D Reconstruction from Sparse, Arbitrarily Posed Real X-rays 

**Title (ZH)**: IXGS-手术中从稀疏、任意姿态的真实X射线进行的3D重建 

**Authors**: Sascha Jecklin, Aidana Massalimova, Ruyi Zha, Lilian Calvet, Christoph J. Laux, Mazda Farshad, Philipp Fürnstahl  

**Link**: [PDF](https://arxiv.org/pdf/2504.14699)  

**Abstract**: Spine surgery is a high-risk intervention demanding precise execution, often supported by image-based navigation systems. Recently, supervised learning approaches have gained attention for reconstructing 3D spinal anatomy from sparse fluoroscopic data, significantly reducing reliance on radiation-intensive 3D imaging systems. However, these methods typically require large amounts of annotated training data and may struggle to generalize across varying patient anatomies or imaging conditions. Instance-learning approaches like Gaussian splatting could offer an alternative by avoiding extensive annotation requirements. While Gaussian splatting has shown promise for novel view synthesis, its application to sparse, arbitrarily posed real intraoperative X-rays has remained largely unexplored. This work addresses this limitation by extending the $R^2$-Gaussian splatting framework to reconstruct anatomically consistent 3D volumes under these challenging conditions. We introduce an anatomy-guided radiographic standardization step using style transfer, improving visual consistency across views, and enhancing reconstruction quality. Notably, our framework requires no pretraining, making it inherently adaptable to new patients and anatomies. We evaluated our approach using an ex-vivo dataset. Expert surgical evaluation confirmed the clinical utility of the 3D reconstructions for navigation, especially when using 20 to 30 views, and highlighted the standardization's benefit for anatomical clarity. Benchmarking via quantitative 2D metrics (PSNR/SSIM) confirmed performance trade-offs compared to idealized settings, but also validated the improvement gained from standardization over raw inputs. This work demonstrates the feasibility of instance-based volumetric reconstruction from arbitrary sparse-view X-rays, advancing intraoperative 3D imaging for surgical navigation. 

**Abstract (ZH)**: 基于实例的学习方法在稀疏视图X射线三维重建中的应用：改进脊柱手术导航成像 

---
# VM-BHINet:Vision Mamba Bimanual Hand Interaction Network for 3D Interacting Hand Mesh Recovery From a Single RGB Image 

**Title (ZH)**: VM-BHINet: Vision Mamba Bimanual Hand Interaction Network for 3D Interacting Hand Mesh Recovery from a Single RGB Image 

**Authors**: Han Bi, Ge Yu, Yu He, Wenzhuo Liu, Zijie Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14618)  

**Abstract**: Understanding bimanual hand interactions is essential for realistic 3D pose and shape reconstruction. However, existing methods struggle with occlusions, ambiguous appearances, and computational inefficiencies. To address these challenges, we propose Vision Mamba Bimanual Hand Interaction Network (VM-BHINet), introducing state space models (SSMs) into hand reconstruction to enhance interaction modeling while improving computational efficiency. The core component, Vision Mamba Interaction Feature Extraction Block (VM-IFEBlock), combines SSMs with local and global feature operations, enabling deep understanding of hand interactions. Experiments on the InterHand2.6M dataset show that VM-BHINet reduces Mean per-joint position error (MPJPE) and Mean per-vertex position error (MPVPE) by 2-3%, significantly surpassing state-of-the-art methods. 

**Abstract (ZH)**: 理解双手中手交互对于实现逼真的3D姿态和形状重建是必不可少的。然而，现有方法在处理遮挡、模糊外观和计算效率低下方面存在困难。为了解决这些挑战，我们提出了Vision Mamba双手中手交互网络（VM-BHINet），通过引入状态空间模型（SSMs）来增强交互建模并提高计算效率。核心组件，Vision Mamba交互特征提取块（VM-IFEBlock），将SSMs与局部和全局特征操作相结合，实现了对手交互的深层次理解。实验结果显示，VM-BHINet在InterHand2.6M数据集上的Mean per-joint position error（MPJPE）和Mean per-vertex position error（MPVPE）分别减少了2-3%，显著优于现有最佳方法。 

---
# VGNC: Reducing the Overfitting of Sparse-view 3DGS via Validation-guided Gaussian Number Control 

**Title (ZH)**: VGNC: 通过验证引导的高斯数字控制减少稀视角3DGS过拟合 

**Authors**: Lifeng Lin, Rongfeng Lu, Quan Chen, Haofan Ren, Ming Lu, Yaoqi Sun, Chenggang Yan, Anke Xue  

**Link**: [PDF](https://arxiv.org/pdf/2504.14548)  

**Abstract**: Sparse-view 3D reconstruction is a fundamental yet challenging task in practical 3D reconstruction applications. Recently, many methods based on the 3D Gaussian Splatting (3DGS) framework have been proposed to address sparse-view 3D reconstruction. Although these methods have made considerable advancements, they still show significant issues with overfitting. To reduce the overfitting, we introduce VGNC, a novel Validation-guided Gaussian Number Control (VGNC) approach based on generative novel view synthesis (NVS) models. To the best of our knowledge, this is the first attempt to alleviate the overfitting issue of sparse-view 3DGS with generative validation images. Specifically, we first introduce a validation image generation method based on a generative NVS model. We then propose a Gaussian number control strategy that utilizes generated validation images to determine the optimal Gaussian numbers, thereby reducing the issue of overfitting. We conducted detailed experiments on various sparse-view 3DGS baselines and datasets to evaluate the effectiveness of VGNC. Extensive experiments show that our approach not only reduces overfitting but also improves rendering quality on the test set while decreasing the number of Gaussian points. This reduction lowers storage demands and accelerates both training and rendering. The code will be released. 

**Abstract (ZH)**: 基于验证引导的高斯点数控制（VGNC）：生成式新视角合成在稀疏视角3D重建中的应用 

---
# DreamID: High-Fidelity and Fast diffusion-based Face Swapping via Triplet ID Group Learning 

**Title (ZH)**: DreamID: 基于三重ID组学习的高保真快速人脸互换 

**Authors**: Fulong Ye, Miao Hua, Pengze Zhang, Xinghui Li, Qichao Sun, Songtao Zhao, Qian He, Xinglong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14509)  

**Abstract**: In this paper, we introduce DreamID, a diffusion-based face swapping model that achieves high levels of ID similarity, attribute preservation, image fidelity, and fast inference speed. Unlike the typical face swapping training process, which often relies on implicit supervision and struggles to achieve satisfactory results. DreamID establishes explicit supervision for face swapping by constructing Triplet ID Group data, significantly enhancing identity similarity and attribute preservation. The iterative nature of diffusion models poses challenges for utilizing efficient image-space loss functions, as performing time-consuming multi-step sampling to obtain the generated image during training is impractical. To address this issue, we leverage the accelerated diffusion model SD Turbo, reducing the inference steps to a single iteration, enabling efficient pixel-level end-to-end training with explicit Triplet ID Group supervision. Additionally, we propose an improved diffusion-based model architecture comprising SwapNet, FaceNet, and ID Adapter. This robust architecture fully unlocks the power of the Triplet ID Group explicit supervision. Finally, to further extend our method, we explicitly modify the Triplet ID Group data during training to fine-tune and preserve specific attributes, such as glasses and face shape. Extensive experiments demonstrate that DreamID outperforms state-of-the-art methods in terms of identity similarity, pose and expression preservation, and image fidelity. Overall, DreamID achieves high-quality face swapping results at 512*512 resolution in just 0.6 seconds and performs exceptionally well in challenging scenarios such as complex lighting, large angles, and occlusions. 

**Abstract (ZH)**: 基于扩散模型的DreamID面部替换模型：高身份相似度、属性保真度、图像 fidelity 和快速推断速度 

---
# Adversarial Attack for RGB-Event based Visual Object Tracking 

**Title (ZH)**: 基于RGB-事件的视觉目标跟踪的对抗攻击 

**Authors**: Qiang Chen, Xiao Wang, Haowen Wang, Bo Jiang, Lin Zhu, Dawei Zhang, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14423)  

**Abstract**: Visual object tracking is a crucial research topic in the fields of computer vision and multi-modal fusion. Among various approaches, robust visual tracking that combines RGB frames with Event streams has attracted increasing attention from researchers. While striving for high accuracy and efficiency in tracking, it is also important to explore how to effectively conduct adversarial attacks and defenses on RGB-Event stream tracking algorithms, yet research in this area remains relatively scarce. To bridge this gap, in this paper, we propose a cross-modal adversarial attack algorithm for RGB-Event visual tracking. Because of the diverse representations of Event streams, and given that Event voxels and frames are more commonly used, this paper will focus on these two representations for an in-depth study. Specifically, for the RGB-Event voxel, we first optimize the perturbation by adversarial loss to generate RGB frame adversarial examples. For discrete Event voxel representations, we propose a two-step attack strategy, more in detail, we first inject Event voxels into the target region as initialized adversarial examples, then, conduct a gradient-guided optimization by perturbing the spatial location of the Event voxels. For the RGB-Event frame based tracking, we optimize the cross-modal universal perturbation by integrating the gradient information from multimodal data. We evaluate the proposed approach against attacks on three widely used RGB-Event Tracking datasets, i.e., COESOT, FE108, and VisEvent. Extensive experiments show that our method significantly reduces the performance of the tracker across numerous datasets in both unimodal and multimodal scenarios. The source code will be released on this https URL 

**Abstract (ZH)**: 跨模态的RGB-事件流视觉跟踪对抗攻击算法 

---
# LOOPE: Learnable Optimal Patch Order in Positional Embeddings for Vision Transformers 

**Title (ZH)**: LOOPE: 可学习的位置嵌入最优patches顺序在视觉变换器中的应用 

**Authors**: Md Abtahi Majeed Chowdhury, Md Rifat Ur Rahman, Akil Ahmad Taki  

**Link**: [PDF](https://arxiv.org/pdf/2504.14386)  

**Abstract**: Positional embeddings (PE) play a crucial role in Vision Transformers (ViTs) by providing spatial information otherwise lost due to the permutation invariant nature of self attention. While absolute positional embeddings (APE) have shown theoretical advantages over relative positional embeddings (RPE), particularly due to the ability of sinusoidal functions to preserve spatial inductive biases like monotonicity and shift invariance, a fundamental challenge arises when mapping a 2D grid to a 1D sequence. Existing methods have mostly overlooked or never explored the impact of patch ordering in positional embeddings. To address this, we propose LOOPE, a learnable patch-ordering method that optimizes spatial representation for a given set of frequencies, providing a principled approach to patch order optimization. Empirical results show that our PE significantly improves classification accuracy across various ViT architectures. To rigorously evaluate the effectiveness of positional embeddings, we introduce the "Three Cell Experiment", a novel benchmarking framework that assesses the ability of PEs to retain relative and absolute positional information across different ViT architectures. Unlike standard evaluations, which typically report a performance gap of 4 to 6% between models with and without PE, our method reveals a striking 30 to 35% difference, offering a more sensitive diagnostic tool to measure the efficacy of PEs. Our experimental analysis confirms that the proposed LOOPE demonstrates enhanced effectiveness in retaining both relative and absolute positional information. 

**Abstract (ZH)**: 位置嵌入（PE）在视觉变压器（ViTs）中通过提供由于自注意力的排列不变性而丢失的空间信息发挥着关键作用。虽然绝对位置嵌入（APE）在理论上展现出相对于相对位置嵌入（RPE）的优势，尤其是在保持诸如单调性和移位不变性等空间诱导偏置方面，当将2D网格映射到1D序列时，一个基本的挑战随之而来。现有的方法大多忽视或从未探索过位置嵌入中切片顺序的影响。为了解决这一问题，我们提出了一种可学习的切片顺序方法LOOPE，它针对给定的频率集优化空间表示，为切片顺序的优化提供了一种原则性的方法。我们的实验结果显示，位置嵌入显著提高了各种ViT架构的分类准确性。为了严格评估位置嵌入的有效性，我们引入了“The Three Cell Experiment”这一新的基准框架，评估位置嵌入在不同ViT架构中保留相对和绝对位置信息的能力。与标准评估相比，我们的方法揭示了高达30%到35%的显著差异，提供了更敏感的诊断工具来衡量位置嵌入的效果。我们的实验分析证实，所提出的LOOPE在保留相对和绝对位置信息方面表现出更优的效果。 

---
# Visual Prompting for One-shot Controllable Video Editing without Inversion 

**Title (ZH)**: 无需倒置的一次性可控视频编辑的视觉提示方法 

**Authors**: Zhengbo Zhang, Yuxi Zhou, Duo Peng, Joo-Hwee Lim, Zhigang Tu, De Wen Soh, Lin Geng Foo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14335)  

**Abstract**: One-shot controllable video editing (OCVE) is an important yet challenging task, aiming to propagate user edits that are made -- using any image editing tool -- on the first frame of a video to all subsequent frames, while ensuring content consistency between edited frames and source frames. To achieve this, prior methods employ DDIM inversion to transform source frames into latent noise, which is then fed into a pre-trained diffusion model, conditioned on the user-edited first frame, to generate the edited video. However, the DDIM inversion process accumulates errors, which hinder the latent noise from accurately reconstructing the source frames, ultimately compromising content consistency in the generated edited frames. To overcome it, our method eliminates the need for DDIM inversion by performing OCVE through a novel perspective based on visual prompting. Furthermore, inspired by consistency models that can perform multi-step consistency sampling to generate a sequence of content-consistent images, we propose a content consistency sampling (CCS) to ensure content consistency between the generated edited frames and the source frames. Moreover, we introduce a temporal-content consistency sampling (TCS) based on Stein Variational Gradient Descent to ensure temporal consistency across the edited frames. Extensive experiments validate the effectiveness of our approach. 

**Abstract (ZH)**: 基于视觉提示的一次性可控视频编辑（OCVE）：内容一致性的采样方法 

---
# Balancing Privacy and Action Performance: A Penalty-Driven Approach to Image Anonymization 

**Title (ZH)**: 平衡隐私与行动性能：一种惩罚驱动的图像匿名化方法 

**Authors**: Nazia Aslam, Kamal Nasrollahi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14301)  

**Abstract**: The rapid development of video surveillance systems for object detection, tracking, activity recognition, and anomaly detection has revolutionized our day-to-day lives while setting alarms for privacy concerns. It isn't easy to strike a balance between visual privacy and action recognition performance in most computer vision models. Is it possible to safeguard privacy without sacrificing performance? It poses a formidable challenge, as even minor privacy enhancements can lead to substantial performance degradation. To address this challenge, we propose a privacy-preserving image anonymization technique that optimizes the anonymizer using penalties from the utility branch, ensuring improved action recognition performance while minimally affecting privacy leakage. This approach addresses the trade-off between minimizing privacy leakage and maintaining high action performance. The proposed approach is primarily designed to align with the regulatory standards of the EU AI Act and GDPR, ensuring the protection of personally identifiable information while maintaining action performance. To the best of our knowledge, we are the first to introduce a feature-based penalty scheme that exclusively controls the action features, allowing freedom to anonymize private attributes. Extensive experiments were conducted to validate the effectiveness of the proposed method. The results demonstrate that applying a penalty to anonymizer from utility branch enhances action performance while maintaining nearly consistent privacy leakage across different penalty settings. 

**Abstract (ZH)**: 视频监控系统中基于对象检测、跟踪、行为识别和异常检测的快速发展已经革新了我们的日常生活，同时也引发了隐私担忧。在大多数计算机视觉模型中，要在视觉隐私和行为识别性能之间找到平衡颇不易。是否可以在不牺牲性能的前提下保护隐私？这是一个严峻的挑战，因为即使是微小的隐私增强也可能导致性能大幅度下降。为应对这一挑战，我们提出了一种隐私保护图像匿名化技术，通过使用来自效用分支的惩罚优化匿名器，从而在最小影响隐私泄露的同时提升行为识别性能。该方法旨在平衡减少隐私泄露和保持高水平行为性能之间的权衡。我们主要设计该方法以符合欧盟AI法案和GDPR的监管标准，确保在保护个人可识别信息的同时维持行为性能。据我们所知，我们首次提出了基于特征的惩罚方案，该方案专门控制行为特征，允许对私有属性进行匿名化而不受限制。进行了大量实验证明所提方法的有效性。结果表明，来自效用分支的惩罚应用于匿名器可以提高行为识别性能，同时在不同惩罚设置下保持接近一致的隐私泄露水平。 

---
# Learning Joint ID-Textual Representation for ID-Preserving Image Synthesis 

**Title (ZH)**: ID保真的联合身份-文本表示学习 

**Authors**: Zichuan Liu, Liming Jiang, Qing Yan, Yumin Jia, Hao Kang, Xin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14202)  

**Abstract**: We propose a novel framework for ID-preserving generation using a multi-modal encoding strategy rather than injecting identity features via adapters into pre-trained models. Our method treats identity and text as a unified conditioning input. To achieve this, we introduce FaceCLIP, a multi-modal encoder that learns a joint embedding space for both identity and textual semantics. Given a reference face and a text prompt, FaceCLIP produces a unified representation that encodes both identity and text, which conditions a base diffusion model to generate images that are identity-consistent and text-aligned. We also present a multi-modal alignment algorithm to train FaceCLIP, using a loss that aligns its joint representation with face, text, and image embedding spaces. We then build FaceCLIP-SDXL, an ID-preserving image synthesis pipeline by integrating FaceCLIP with Stable Diffusion XL (SDXL). Compared to prior methods, FaceCLIP-SDXL enables photorealistic portrait generation with better identity preservation and textual relevance. Extensive experiments demonstrate its quantitative and qualitative superiority. 

**Abstract (ZH)**: 我们提出了一种新的框架，采用多模态编码策略实现身份保留生成，而不是通过适配器将身份特征注入预训练模型。我们的方法将身份和文本视为统一的条件输入。为此，我们引入了FaceCLIP，这是一种多模态编码器，学习身份和文本语义的联合嵌入空间。给定一个参考人脸和文本提示，FaceCLIP生成一个统一表示，同时编码身份和文本，用于条件基扩散模型生成与身份一致且与文本对齐的图像。我们还提出了一种多模态对齐算法来训练FaceCLIP，使用一种损失函数，使其联合表示与人脸、文本和图像嵌入空间对齐。然后，我们构建了FaceCLIP-SDXL，这是一种通过将FaceCLIP与Stable Diffusion XL (SDXL) 结合的具有身份保留的图像合成管线。与先前的方法相比，FaceCLIP-SDXL 能够生成更具真实感的肖像，同时保持更好的身份一致性与文本相关性。大量实验证明了其在定量和定性上的优越性。 

---
# Breaking the Diffraction Barrier for Passive Sources: Parameter-Decoupled Superresolution Assisted by Physics-Informed Machine Learning 

**Title (ZH)**: 突破衍射极限的被动源：参数解耦超分辨辅助下的物理知情机器学习 

**Authors**: Abdelali Sajia, Bilal Benzimoun, Pawan Khatiwada, Guogan Zhao, Xiao-Feng Qian  

**Link**: [PDF](https://arxiv.org/pdf/2504.14156)  

**Abstract**: We present a parameter-decoupled superresolution framework for estimating sub-wavelength separations of passive two-point sources without requiring prior knowledge or control of the source. Our theoretical foundation circumvents the need to estimate multiple challenging parameters such as partial coherence, brightness imbalance, random relative phase, and photon statistics. A physics-informed machine learning (ML) model (trained with a standard desktop workstation), synergistically integrating this theory, further addresses practical imperfections including background noise, photon loss, and centroid/orientation misalignment. The integrated parameter-decoupling superresolution method achieves resolution 14 and more times below the diffraction limit (corresponding to ~ 13.5 nm in optical microscopy) on experimentally generated realistic images with >82% fidelity, performance rivaling state-of-the-art techniques for actively controllable sources. Critically, our method's robustness against source parameter variability and source-independent noises enables potential applications in realistic scenarios where source control is infeasible, such as astrophysical imaging, live-cell microscopy, and quantum metrology. This work bridges a critical gap between theoretical superresolution limits and practical implementations for passive systems. 

**Abstract (ZH)**: 无源两点源超分辨框架：无需先验知识或源控制的亚波长分离估计 

---
# ThyroidEffi 1.0: A Cost-Effective System for High-Performance Multi-Class Thyroid Carcinoma Classification 

**Title (ZH)**: ThyroidEffi 1.0: 一种高性能多类甲状腺癌分类的成本-effective系统 

**Authors**: Hai Pham-Ngoc, De Nguyen-Van, Dung Vu-Tien, Phuong Le-Hong  

**Link**: [PDF](https://arxiv.org/pdf/2504.14139)  

**Abstract**: Background: Automated classification of thyroid fine needle aspiration biopsy (FNAB) images faces challenges in limited data, inter-observer variability, and computational cost. Efficient, interpretable models are crucial for clinical support. Objective: To develop and externally validate a deep learning system for the multi-class classification of thyroid FNAB images into three key categories that directly guide post-biopsy treatment decisions in Vietnam: benign (B2), suspicious for malignancy (B5), and malignant (B6), while achieving high diagnostic accuracy with low computational overhead. Methods: Our framework features: (1) YOLOv10-based cell cluster detection for informative sub-region extraction and noise reduction; (2) a curriculum learning-inspired protocol sequencing localized crops to full images for multi-scale feature capture; (3) adaptive lightweight EfficientNetB0 (4 millions parameters) selection balancing performance and efficiency; and (4) a Transformer-inspired module for multi-scale, multi-region analysis. External validation used 1,015 independent FNAB images. Results: ThyroidEffi Basic achieved a macro F1 of 89.19\% and AUCs of 0.98 (B2), 0.95 (B5), and 0.96 (B6) on the internal test set. External validation yielded AUCs of 0.9495 (B2), 0.7436 (B5), and 0.8396 (B6). ThyroidEffi Premium improved macro F1 to 89.77\%. Grad-CAM highlighted key diagnostic regions, confirming interpretability. The system processed 1000 cases in 30 seconds, demonstrating feasibility on widely accessible hardware like a 12-core CPU. Conclusions: This work demonstrates that high-accuracy, interpretable thyroid FNAB image classification is achievable with minimal computational demands. 

**Abstract (ZH)**: 背景:甲状腺细针穿刺活检（FNAB）图像的自动分类面临数据有限、观察者间变异性以及计算成本高的挑战。高效的可解释模型对于临床支持至关重要。目的:开发并外部验证一个深度学习系统，用于将甲状腺FNAB图像分为直接指导术后治疗决策的三个关键类别：良性（B2）、可疑恶性（B5）和恶性（B6），同时实现高诊断准确性并具备低计算开销。方法:我们的框架包括：（1）基于YOLOv10的细胞簇检测，用于信息子区域提取和噪声减少；（2）基于曲率学习的协议，按局部裁剪到全图像的序列进行多尺度特征捕捉；（3）自适应的轻量级EfficientNetB0（400万个参数）选择，平衡性能和效率；（4）基于Transformer的设计模块进行多尺度、多区域分析。外部验证使用了1015张独立的FNAB图像。结果: ThyroidEffi Basic在内部测试集上实现了宏F1值89.19%和AUC值分别为0.98（B2）、0.95（B5）和0.96（B6）。外部验证AUC值分别为0.9495（B2）、0.7436（B5）和0.8396（B6）。ThyroidEffi Premium将宏F1提高到了89.77%。Grad-CAM高亮了关键诊断区域，证实了系统的可解释性。该系统在12核CPU等广泛可用的硬件上每秒处理1000个案例，展示了其实现的可行性。结论:本研究证明，即使在低计算需求下，高准确度和可解释的甲状腺FNAB图像分类也是可行的。 

---
# Occlusion-Ordered Semantic Instance Segmentation 

**Title (ZH)**: 遮挡有序语义实例分割 

**Authors**: Soroosh Baselizadeh, Cheuk-To Yu, Olga Veksler, Yuri Boykov  

**Link**: [PDF](https://arxiv.org/pdf/2504.14054)  

**Abstract**: Standard semantic instance segmentation provides useful, but inherently 2D information from a single image. To enable 3D analysis, one usually integrates absolute monocular depth estimation with instance segmentation. However, monocular depth is a difficult task. Instead, we leverage a simpler single-image task, occlusion-based relative depth ordering, providing coarser but useful 3D information. We show that relative depth ordering works more reliably from occlusions than from absolute depth. We propose to solve the joint task of relative depth ordering and segmentation of instances based on occlusions. We call this task Occlusion-Ordered Semantic Instance Segmentation (OOSIS). We develop an approach to OOSIS that extracts instances and their occlusion order simultaneously from oriented occlusion boundaries and semantic segmentation. Unlike popular detect-and-segment framework for instance segmentation, combining occlusion ordering with instance segmentation allows a simple and clean formulation of OOSIS as a labeling problem. As a part of our solution for OOSIS, we develop a novel oriented occlusion boundaries approach that significantly outperforms prior work. We also develop a new joint OOSIS metric based both on instance mask accuracy and correctness of their occlusion order. We achieve better performance than strong baselines on KINS and COCOA datasets. 

**Abstract (ZH)**: 基于遮挡顺序的语义实例分割（OOSIS） 

---
# LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models 

**Title (ZH)**: LoftUp: 基于坐标的学习特征上采样器用于视觉基础模型 

**Authors**: Haiwen Huang, Anpei Chen, Volodymyr Havrylov, Andreas Geiger, Dan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14032)  

**Abstract**: Vision foundation models (VFMs) such as DINOv2 and CLIP have achieved impressive results on various downstream tasks, but their limited feature resolution hampers performance in applications requiring pixel-level understanding. Feature upsampling offers a promising direction to address this challenge. In this work, we identify two critical factors for enhancing feature upsampling: the upsampler architecture and the training objective. For the upsampler architecture, we introduce a coordinate-based cross-attention transformer that integrates the high-resolution images with coordinates and low-resolution VFM features to generate sharp, high-quality features. For the training objective, we propose constructing high-resolution pseudo-groundtruth features by leveraging class-agnostic masks and self-distillation. Our approach effectively captures fine-grained details and adapts flexibly to various input and feature resolutions. Through experiments, we demonstrate that our approach significantly outperforms existing feature upsampling techniques across various downstream tasks. Our code is released at this https URL. 

**Abstract (ZH)**: 基于视觉的础模型（VFMs）如DINOv2和CLIP已在各种下游任务中取得了 impressive 的成果，但它们有限的特征分辨率阻碍了在需要像素级理解的应用中的性能。特征上采样为解决这一挑战提供了有希望的方向。在本文中，我们确定了增强特征上采样的两个关键因素：上采样器架构和训练目标。对于上采样器架构，我们引入了一种基于坐标的交叉注意变换器，它可以将高分辨率图像与坐标和低分辨率的VFM特征结合，生成清晰的高质量特征。对于训练目标，我们提出通过利用类无感知掩模和自我蒸馏构建高分辨率伪 ground-truth 特征。我们的方法有效地捕捉到了细微的细节，并且能够灵活适应各种输入和特征分辨率。通过实验，我们展示了我们的方法在各种下游任务中显著优于现有特征上采样技术。我们的代码已发布在 this https URL。 

---
# Multiscale Tensor Summation Factorization as a New Neural Network Layer (MTS Layer) for Multidimensional Data Processing 

**Title (ZH)**: 多尺度张量求和因子分解作为新型多维数据处理的神经网络层（MTS层） 

**Authors**: Mehmet Yamaç, Muhammad Numan Yousaf, Serkan Kiranyaz, Moncef Gabbouj  

**Link**: [PDF](https://arxiv.org/pdf/2504.13975)  

**Abstract**: Multilayer perceptrons (MLP), or fully connected artificial neural networks, are known for performing vector-matrix multiplications using learnable weight matrices; however, their practical application in many machine learning tasks, especially in computer vision, can be limited due to the high dimensionality of input-output pairs at each layer. To improve efficiency, convolutional operators have been utilized to facilitate weight sharing and local connections, yet they are constrained by limited receptive fields. In this paper, we introduce Multiscale Tensor Summation (MTS) Factorization, a novel neural network operator that implements tensor summation at multiple scales, where each tensor to be summed is obtained through Tucker-decomposition-like mode products. Unlike other tensor decomposition methods in the literature, MTS is not introduced as a network compression tool; instead, as a new backbone neural layer. MTS not only reduces the number of parameters required while enhancing the efficiency of weight optimization compared to traditional dense layers (i.e., unfactorized weight matrices in MLP layers), but it also demonstrates clear advantages over convolutional layers. The proof-of-concept experimental comparison of the proposed MTS networks with MLPs and Convolutional Neural Networks (CNNs) demonstrates their effectiveness across various tasks, such as classification, compression, and signal restoration. Additionally, when integrated with modern non-linear units such as the multi-head gate (MHG), also introduced in this study, the corresponding neural network, MTSNet, demonstrates a more favorable complexity-performance tradeoff compared to state-of-the-art transformers in various computer vision applications. The software implementation of the MTS layer and the corresponding MTS-based networks, MTSNets, is shared at this https URL. 

**Abstract (ZH)**: 多层张量求和因子化（MTS因子化）：一种多尺度张量求和的新神经网络运算符 

---
# Evaluating Menu OCR and Translation: A Benchmark for Aligning Human and Automated Evaluations in Large Vision-Language Models 

**Title (ZH)**: 评估菜单OCR和翻译：大型视觉-语言模型中人工评估与自动化评估对齐的标准 

**Authors**: Zhanglin Wu, Tengfei Song, Ning Xie, Weidong Zhang, Mengli Zhu, Shuang Wu, Shiliang Sun, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13945)  

**Abstract**: The rapid advancement of large vision-language models (LVLMs) has significantly propelled applications in document understanding, particularly in optical character recognition (OCR) and multilingual translation. However, current evaluations of LVLMs, like the widely used OCRBench, mainly focus on verifying the correctness of their short-text responses and long-text responses with simple layout, while the evaluation of their ability to understand long texts with complex layout design is highly significant but largely overlooked. In this paper, we propose Menu OCR and Translation Benchmark (MOTBench), a specialized evaluation framework emphasizing the pivotal role of menu translation in cross-cultural communication. MOTBench requires LVLMs to accurately recognize and translate each dish, along with its price and unit items on a menu, providing a comprehensive assessment of their visual understanding and language processing capabilities. Our benchmark is comprised of a collection of Chinese and English menus, characterized by intricate layouts, a variety of fonts, and culturally specific elements across different languages, along with precise human annotations. Experiments show that our automatic evaluation results are highly consistent with professional human evaluation. We evaluate a range of publicly available state-of-the-art LVLMs, and through analyzing their output to identify the strengths and weaknesses in their performance, offering valuable insights to guide future advancements in LVLM development. MOTBench is available at this https URL. 

**Abstract (ZH)**: 大规模视觉语言模型的快速发展极大地推动了文档理解的应用，特别是在光学字符识别（OCR）和多语言翻译方面。然而，现有的大规模视觉语言模型评估，如广泛使用的OCRBench，主要集中在验证其短文本和简单布局长文本响应的正确性，而对其理解和翻译复杂布局长文本能力的评估则显得至关重要但被忽视。本文提出了一种专门的评估框架——菜单OCR和翻译基准（MOTBench），强调菜单翻译在跨文化沟通中的关键作用。MOTBench要求大规模视觉语言模型准确识别和翻译菜单上每道菜、价格以及单位项目，从而全面评估其视觉理解和语言处理能力。我们的基准数据集包含中文和英文菜单，布局复杂，字体多样，并且包含不同语言中的文化特定元素，同时还附有人工精确标注。实验结果显示，我们的自动评估结果与专业的人类评估高度一致。我们评估了多种公开的领先大规模视觉语言模型，并通过分析其输出来识别其性能的优点和不足，为未来的模型开发提供了有价值的见解。MOTBench可访问 [此链接]。 

---
