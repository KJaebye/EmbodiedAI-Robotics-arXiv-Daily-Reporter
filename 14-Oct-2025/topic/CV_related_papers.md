# Controllable Generative Trajectory Prediction via Weak Preference Alignment 

**Title (ZH)**: 可控生成轨迹预测通过弱偏好对齐 

**Authors**: Yongxi Cao, Julian F. Schumann, Jens Kober, Joni Pajarinen, Arkady Zgonnikov  

**Link**: [PDF](https://arxiv.org/pdf/2510.10731)  

**Abstract**: Deep generative models such as conditional variational autoencoders (CVAEs) have shown great promise for predicting trajectories of surrounding agents in autonomous vehicle planning. State-of-the-art models have achieved remarkable accuracy in such prediction tasks. Besides accuracy, diversity is also crucial for safe planning because human behaviors are inherently uncertain and multimodal. However, existing methods generally lack a scheme to generate controllably diverse trajectories, which is arguably more useful than randomly diversified trajectories, to the end of safe planning. To address this, we propose PrefCVAE, an augmented CVAE framework that uses weakly labeled preference pairs to imbue latent variables with semantic attributes. Using average velocity as an example attribute, we demonstrate that PrefCVAE enables controllable, semantically meaningful predictions without degrading baseline accuracy. Our results show the effectiveness of preference supervision as a cost-effective way to enhance sampling-based generative models. 

**Abstract (ZH)**: 基于弱标签偏好配对的增强条件变分自编码器 

---
# SpikeGrasp: A Benchmark for 6-DoF Grasp Pose Detection from Stereo Spike Streams 

**Title (ZH)**: SpikeGrasp: 6-DoF 抓取姿态检测基准从立体尖峰流中 

**Authors**: Zhuoheng Gao, Jiyao Zhang, Zhiyong Xie, Hao Dong, Zhaofei Yu, Rongmei Chen, Guozhang Chen, Tiejun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10602)  

**Abstract**: Most robotic grasping systems rely on converting sensor data into explicit 3D point clouds, which is a computational step not found in biological intelligence. This paper explores a fundamentally different, neuro-inspired paradigm for 6-DoF grasp detection. We introduce SpikeGrasp, a framework that mimics the biological visuomotor pathway, processing raw, asynchronous events from stereo spike cameras, similarly to retinas, to directly infer grasp poses. Our model fuses these stereo spike streams and uses a recurrent spiking neural network, analogous to high-level visual processing, to iteratively refine grasp hypotheses without ever reconstructing a point cloud. To validate this approach, we built a large-scale synthetic benchmark dataset. Experiments show that SpikeGrasp surpasses traditional point-cloud-based baselines, especially in cluttered and textureless scenes, and demonstrates remarkable data efficiency. By establishing the viability of this end-to-end, neuro-inspired approach, SpikeGrasp paves the way for future systems capable of the fluid and efficient manipulation seen in nature, particularly for dynamic objects. 

**Abstract (ZH)**: 基于神经启发的6-自由度抓取检测框架：SpikeGrasp 

---
# Fast Vision in the Dark: A Case for Single-Photon Imaging in Planetary Navigation 

**Title (ZH)**: 快速暗夜视觉：单光子成像在行星导航中的应用案例 

**Authors**: David Rodríguez-Martínez, C.J. Pérez del Pulgar  

**Link**: [PDF](https://arxiv.org/pdf/2510.10597)  

**Abstract**: Improving robotic navigation is critical for extending exploration range and enhancing operational efficiency. Vision-based navigation relying on traditional CCD or CMOS cameras faces major challenges when complex illumination conditions are paired with motion, limiting the range and accessibility of mobile planetary robots. In this study, we propose a novel approach to planetary navigation that leverages the unique imaging capabilities of Single-Photon Avalanche Diode (SPAD) cameras. We present the first comprehensive evaluation of single-photon imaging as an alternative passive sensing technology for robotic exploration missions targeting perceptually challenging locations, with a special emphasis on high-latitude lunar regions. We detail the operating principles and performance characteristics of SPAD cameras, assess their advantages and limitations in addressing key perception challenges of upcoming exploration missions to the Moon, and benchmark their performance under representative illumination conditions. 

**Abstract (ZH)**: 提高机器人导航性能对于扩展探索范围和提升操作效率至关重要。基于视觉的导航依赖传统的CCD或CMOS相机，在复杂光照条件与运动结合时面临重大挑战，限制了移动行星机器人的作用范围和可达性。本研究表明，通过利用单光子雪崩二极管（SPAD）相机的独特成像能力，可以提出一种新的行星导航方法。我们首次全面评估了单光子成像作为一种替代性被动传感技术在针对感知挑战性地点的机器人探索任务中的应用，特别强调了高纬度月球地区的应用。我们详细介绍了SPAD相机的工作原理和性能特征，评估了其在应对即将对月球进行的探索任务中的关键感知挑战方面的优势和局限性，并在代表性光照条件下对其性能进行了基准测试。 

---
# sqrtVINS: Robust and Ultrafast Square-Root Filter-based 3D Motion Tracking 

**Title (ZH)**: sqrtVINS: 均方根滤波器基于的稳健和超快速三维运动跟踪 

**Authors**: Yuxiang Peng, Chuchu Chen, Kejian Wu, Guoquan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10346)  

**Abstract**: In this paper, we develop and open-source, for the first time, a square-root filter (SRF)-based visual-inertial navigation system (VINS), termed sqrtVINS, which is ultra-fast, numerically stable, and capable of dynamic initialization even under extreme conditions (i.e., extremely small time window). Despite recent advancements in VINS, resource constraints and numerical instability on embedded (robotic) systems with limited precision remain critical challenges. A square-root covariance-based filter offers a promising solution by providing numerical stability, efficient memory usage, and guaranteed positive semi-definiteness. However, canonical SRFs suffer from inefficiencies caused by disruptions in the triangular structure of the covariance matrix during updates. The proposed method significantly improves VINS efficiency with a novel Cholesky decomposition (LLT)-based SRF update, by fully exploiting the system structure to preserve the structure. Moreover, we design a fast, robust, dynamic initialization method, which first recovers the minimal states without triangulating 3D features and then efficiently performs iterative SRF update to refine the full states, enabling seamless VINS operation. The proposed LLT-based SRF is extensively verified through numerical studies, demonstrating superior numerical stability and achieving robust efficient performance on 32-bit single-precision floats, operating at twice the speed of state-of-the-art (SOTA) methods. Our initialization method, tested on both mobile workstations and Jetson Nano computers, achieving a high success rate of initialization even within a 100 ms window under minimal conditions. Finally, the proposed sqrtVINS is extensively validated across diverse scenarios, demonstrating strong efficiency, robustness, and reliability. The full open-source implementation is released to support future research and applications. 

**Abstract (ZH)**: 基于平方根滤波器的超快视觉惯性导航系统（sqrtVINS） 

---
# VG-Mapping: Variation-Aware 3D Gaussians for Online Semi-static Scene Mapping 

**Title (ZH)**: VG-Mapping: 基于变异性意识的在线半静态场景3D高斯映射 

**Authors**: Yicheng He, Jingwen Yu, Guangcheng Chen, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09962)  

**Abstract**: Maintaining an up-to-date map that accurately reflects recent changes in the environment is crucial, especially for robots that repeatedly traverse the same space. Failing to promptly update the changed regions can degrade map quality, resulting in poor localization, inefficient operations, and even lost robots. 3D Gaussian Splatting (3DGS) has recently seen widespread adoption in online map reconstruction due to its dense, differentiable, and photorealistic properties, yet accurately and efficiently updating the regions of change remains a challenge. In this paper, we propose VG-Mapping, a novel online 3DGS-based mapping system tailored for such semi-static scenes. Our approach introduces a hybrid representation that augments 3DGS with a TSDF-based voxel map to efficiently identify changed regions in a scene, along with a variation-aware density control strategy that inserts or deletes Gaussian primitives in regions undergoing change. Furthermore, to address the absence of public benchmarks for this task, we construct a RGB-D dataset comprising both synthetic and real-world semi-static environments. Experimental results demonstrate that our method substantially improves the rendering quality and map update efficiency in semi-static scenes. The code and dataset are available at this https URL. 

**Abstract (ZH)**: 维持一个准确反映环境近期变化的最新地图对于重复穿越同一空间的机器人至关重要。未能及时更新变化区域会导致地图质量下降，从而造成定位不佳、操作不效率以及机器人丢失等问题。3D高斯斑图化（3DGS）由于其稠密、可微和逼真的特性，在在线地图重建中得到了广泛应用，但准确而高效地更新变化区域仍然是一个挑战。本文提出VG-Mapping，这是一种针对此类半静态场景的新型在线3DGS基地图制作系统。我们的方法引入了一种混合表示，将3DGS与基于TSDF的体素地图相结合，以高效地识别场景中的变化区域，并提出了一种变化感知的密度控制策略，在发生变化的区域插入或删除高斯原语。此外，为了解决这一任务缺乏公开基准的问题，我们构建了一个包含合成和真实世界半静态环境的RGB-D数据集。实验结果表明，我们的方法在半静态场景中显著提高了渲染质量和地图更新效率。代码和数据集可在以下链接获取：this https URL。 

---
# DKPMV: Dense Keypoints Fusion from Multi-View RGB Frames for 6D Pose Estimation of Textureless Objects 

**Title (ZH)**: DKPMV：无纹理物体六自由度姿态估计的多视图RGB帧密集关键点融合 

**Authors**: Jiahong Chen, Jinghao Wang, Zi Wang, Ziwen Wang, Banglei Guan, Qifeng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10933)  

**Abstract**: 6D pose estimation of textureless objects is valuable for industrial robotic applications, yet remains challenging due to the frequent loss of depth information. Current multi-view methods either rely on depth data or insufficiently exploit multi-view geometric cues, limiting their performance. In this paper, we propose DKPMV, a pipeline that achieves dense keypoint-level fusion using only multi-view RGB images as input. We design a three-stage progressive pose optimization strategy that leverages dense multi-view keypoint geometry information. To enable effective dense keypoint fusion, we enhance the keypoint network with attentional aggregation and symmetry-aware training, improving prediction accuracy and resolving ambiguities on symmetric objects. Extensive experiments on the ROBI dataset demonstrate that DKPMV outperforms state-of-the-art multi-view RGB approaches and even surpasses the RGB-D methods in the majority of cases. The code will be available soon. 

**Abstract (ZH)**: 无纹理对象的6D姿态估计对于工业机器人应用具有重要意义，但由于深度信息的频繁丢失，依然具有挑战性。当前的多视角方法要么依赖于深度数据，要么未能充分利用多视角几何线索，限制了它们的性能。本文提出DKPMV管道，仅使用多视角RGB图像作为输入实现密集关键点级融合。我们设计了一种三阶段渐进姿态优化策略，利用密集的多视角关键点几何信息。为了实现有效的密集关键点融合，我们通过注意力聚合和对称意识训练增强了关键点网络，提高了预测精度并解决了对称对象上的歧义性。在ROBI数据集上的 extensive 实验表明，DKPMV 在大多数情况下优于最先进的多视角RGB方法，并且甚至在某些情况下超越了RGB-D方法。代码即将公开。 

---
# MonoSE(3)-Diffusion: A Monocular SE(3) Diffusion Framework for Robust Camera-to-Robot Pose Estimation 

**Title (ZH)**: MonoSE(3)-Diffusion：一种用于稳健相机到机器人姿态估计的一目测程SE(3)扩散框架 

**Authors**: Kangjian Zhu, Haobo Jiang, Yigong Zhang, Jianjun Qian, Jian Yang, Jin Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.10434)  

**Abstract**: We propose MonoSE(3)-Diffusion, a monocular SE(3) diffusion framework that formulates markerless, image-based robot pose estimation as a conditional denoising diffusion process. The framework consists of two processes: a visibility-constrained diffusion process for diverse pose augmentation and a timestep-aware reverse process for progressive pose refinement. The diffusion process progressively perturbs ground-truth poses to noisy transformations for training a pose denoising network. Importantly, we integrate visibility constraints into the process, ensuring the transformations remain within the camera field of view. Compared to the fixed-scale perturbations used in current methods, the diffusion process generates in-view and diverse training poses, thereby improving the network generalization capability. Furthermore, the reverse process iteratively predicts the poses by the denoising network and refines pose estimates by sampling from the diffusion posterior of current timestep, following a scheduled coarse-to-fine procedure. Moreover, the timestep indicates the transformation scales, which guide the denoising network to achieve more accurate pose predictions. The reverse process demonstrates higher robustness than direct prediction, benefiting from its timestep-aware refinement scheme. Our approach demonstrates improvements across two benchmarks (DREAM and RoboKeyGen), achieving a notable AUC of 66.75 on the most challenging dataset, representing a 32.3% gain over the state-of-the-art. 

**Abstract (ZH)**: 无标记图像引导机器人姿态估计的MonoSE(3)-Diffusion框架 

---
# Bridging Perspectives: Foundation Model Guided BEV Maps for 3D Object Detection and Tracking 

**Title (ZH)**: 视角融合：基础模型引导的BEV地图在3D物体检测与跟踪中的应用 

**Authors**: Markus Käppeler, Özgün Çiçek, Daniele Cattaneo, Claudius Gläser, Yakov Miron, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2510.10287)  

**Abstract**: Camera-based 3D object detection and tracking are essential for perception in autonomous driving. Current state-of-the-art approaches often rely exclusively on either perspective-view (PV) or bird's-eye-view (BEV) features, limiting their ability to leverage both fine-grained object details and spatially structured scene representations. In this work, we propose DualViewDistill, a hybrid detection and tracking framework that incorporates both PV and BEV camera image features to leverage their complementary strengths. Our approach introduces BEV maps guided by foundation models, leveraging descriptive DINOv2 features that are distilled into BEV representations through a novel distillation process. By integrating PV features with BEV maps enriched with semantic and geometric features from DINOv2, our model leverages this hybrid representation via deformable aggregation to enhance 3D object detection and tracking. Extensive experiments on the nuScenes and Argoverse 2 benchmarks demonstrate that DualViewDistill achieves state-of-the-art performance. The results showcase the potential of foundation model BEV maps to enable more reliable perception for autonomous driving. We make the code and pre-trained models available at this https URL . 

**Abstract (ZH)**: 基于相机的3D物体检测与跟踪是自动驾驶感知中的关键。当前最先进的方法通常依赖于透视视图（PV）或鸟瞰视图（BEV）特征中的任一方，限制了其同时利用细粒度物体细节和空间结构化场景表示的能力。本文提出了一种名为DualViewDistill的混合检测与跟踪框架，该框架融合了PV和BEV相机图像特征，以充分发挥其互补优势。我们的方法通过一种新颖的蒸馏过程，利用基础模型引导的BEV图，并借助描述性强的DINOv2特征进行蒸馏，构建BEV表示。通过将PV特征与富含语义和几何特征的DINOv2增强的BEV图相结合，我们的模型通过变形聚合利用这种混合表示，以增强3D物体检测与跟踪。在nuScenes和Argoverse 2基准上的广泛实验表明，DualViewDistill实现了最先进的性能。结果展示了基础模型BEV图在实现更可靠自动驾驶感知方面的潜力。我们将在以下网址提供代码和预训练模型：this https URL。 

---
# CharCom: Composable Identity Control for Multi-Character Story Illustration 

**Title (ZH)**: CharCom: 可组合的身份控制for 多角色故事插画 

**Authors**: Zhongsheng Wang, Ming Lin, Zhedong Lin, Yaser Shakib, Qian Liu, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10135)  

**Abstract**: Ensuring character identity consistency across varying prompts remains a fundamental limitation in diffusion-based text-to-image generation. We propose CharCom, a modular and parameter-efficient framework that achieves character-consistent story illustration through composable LoRA adapters, enabling efficient per-character customization without retraining the base model. Built on a frozen diffusion backbone, CharCom dynamically composes adapters at inference using prompt-aware control. Experiments on multi-scene narratives demonstrate that CharCom significantly enhances character fidelity, semantic alignment, and temporal coherence. It remains robust in crowded scenes and enables scalable multi-character generation with minimal overhead, making it well-suited for real-world applications such as story illustration and animation. 

**Abstract (ZH)**: 确保在变化的提示下角色身份一致性是基于扩散的文本到图像生成中的一个基本限制。我们提出CharCom，一种模块化和参数高效框架，通过可组合的LoRA适配器实现角色一致性故事插图，能够在不重新训练基模型的情况下进行高效的角色个性化定制。CharCom基于冻结的扩散骨干，在推理时使用提示感知控制动态组合适配器。实验表明，CharCom在多场景叙事中显著提高了角色保真度、语义对齐和时间连贯性，并在拥挤场景中保持鲁棒性，能够以最小开销实现可扩展的多角色生成，使其适用于故事插图和动画等实际应用。 

---
# NV3D: Leveraging Spatial Shape Through Normal Vector-based 3D Object Detection 

**Title (ZH)**: NV3D：基于法向量的空间形状在三维物体检测中的应用 

**Authors**: Krittin Chaowakarn, Paramin Sangwongngam, Nang Htet Htet Aung, Chalie Charoenlarpnopparut  

**Link**: [PDF](https://arxiv.org/pdf/2510.11632)  

**Abstract**: Recent studies in 3D object detection for autonomous vehicles aim to enrich features through the utilization of multi-modal setups or the extraction of local patterns within LiDAR point clouds. However, multi-modal methods face significant challenges in feature alignment, and gaining features locally can be oversimplified for complex 3D object detection tasks. In this paper, we propose a novel model, NV3D, which utilizes local features acquired from voxel neighbors, as normal vectors computed per voxel basis using K-nearest neighbors (KNN) and principal component analysis (PCA). This informative feature enables NV3D to determine the relationship between the surface and pertinent target entities, including cars, pedestrians, or cyclists. During the normal vector extraction process, NV3D offers two distinct sampling strategies: normal vector density-based sampling and FOV-aware bin-based sampling, allowing elimination of up to 55% of data while maintaining performance. In addition, we applied element-wise attention fusion, which accepts voxel features as the query and value and normal vector features as the key, similar to the attention mechanism. Our method is trained on the KITTI dataset and has demonstrated superior performance in car and cyclist detection owing to their spatial shapes. In the validation set, NV3D without sampling achieves 86.60% and 80.18% mean Average Precision (mAP), greater than the baseline Voxel R-CNN by 2.61% and 4.23% mAP, respectively. With both samplings, NV3D achieves 85.54% mAP in car detection, exceeding the baseline by 1.56% mAP, despite roughly 55% of voxels being filtered out. 

**Abstract (ZH)**: 基于局部特征的新型3D物体检测模型NV3D 

---
# LikePhys: Evaluating Intuitive Physics Understanding in Video Diffusion Models via Likelihood Preference 

**Title (ZH)**: LikePhys: 通过概率偏好评估视频扩散模型中的直观物理理解 

**Authors**: Jianhao Yuan, Fabio Pizzati, Francesco Pinto, Lars Kunze, Ivan Laptev, Paul Newman, Philip Torr, Daniele De Martini  

**Link**: [PDF](https://arxiv.org/pdf/2510.11512)  

**Abstract**: Intuitive physics understanding in video diffusion models plays an essential role in building general-purpose physically plausible world simulators, yet accurately evaluating such capacity remains a challenging task due to the difficulty in disentangling physics correctness from visual appearance in generation. To the end, we introduce LikePhys, a training-free method that evaluates intuitive physics in video diffusion models by distinguishing physically valid and impossible videos using the denoising objective as an ELBO-based likelihood surrogate on a curated dataset of valid-invalid pairs. By testing on our constructed benchmark of twelve scenarios spanning over four physics domains, we show that our evaluation metric, Plausibility Preference Error (PPE), demonstrates strong alignment with human preference, outperforming state-of-the-art evaluator baselines. We then systematically benchmark intuitive physics understanding in current video diffusion models. Our study further analyses how model design and inference settings affect intuitive physics understanding and highlights domain-specific capacity variations across physical laws. Empirical results show that, despite current models struggling with complex and chaotic dynamics, there is a clear trend of improvement in physics understanding as model capacity and inference settings scale. 

**Abstract (ZH)**: 无 Fairfax, 一种用于评估视频扩散模型直观物理理解的无需训练方法：通过使用基于ELBO的似然代理区分有效和无效视频以评估视频扩散模型中的直观物理理解 

---
# Uncertainty-Aware ControlNet: Bridging Domain Gaps with Synthetic Image Generation 

**Title (ZH)**: 面向不确定性的ControlNet：借助合成图像生成弥合领域差距 

**Authors**: Joshua Niemeijer, Jan Ehrhardt, Heinz Handels, Hristina Uzunova  

**Link**: [PDF](https://arxiv.org/pdf/2510.11346)  

**Abstract**: Generative Models are a valuable tool for the controlled creation of high-quality image data. Controlled diffusion models like the ControlNet have allowed the creation of labeled distributions. Such synthetic datasets can augment the original training distribution when discriminative models, like semantic segmentation, are trained. However, this augmentation effect is limited since ControlNets tend to reproduce the original training distribution.
This work introduces a method to utilize data from unlabeled domains to train ControlNets by introducing the concept of uncertainty into the control mechanism. The uncertainty indicates that a given image was not part of the training distribution of a downstream task, e.g., segmentation. Thus, two types of control are engaged in the final network: an uncertainty control from an unlabeled dataset and a semantic control from the labeled dataset. The resulting ControlNet allows us to create annotated data with high uncertainty from the target domain, i.e., synthetic data from the unlabeled distribution with labels. In our scenario, we consider retinal OCTs, where typically high-quality Spectralis images are available with given ground truth segmentations, enabling the training of segmentation networks. The recent development in Home-OCT devices, however, yields retinal OCTs with lower quality and a large domain shift, such that out-of-the-pocket segmentation networks cannot be applied for this type of data. Synthesizing annotated images from the Home-OCT domain using the proposed approach closes this gap and leads to significantly improved segmentation results without adding any further supervision. The advantage of uncertainty-guidance becomes obvious when compared to style transfer: it enables arbitrary domain shifts without any strict learning of an image style. This is also demonstrated in a traffic scene experiment. 

**Abstract (ZH)**: 利用不确定性引导的数据增强方法训练ControlNet以生成带有高不确定性标注的数据 

---
# When Does Supervised Training Pay Off? The Hidden Economics of Object Detection in the Era of Vision-Language Models 

**Title (ZH)**: 监督训练何时见效？视觉语言模型时代的目标检测隐含经济探讨 

**Authors**: Samer Al-Hamadani  

**Link**: [PDF](https://arxiv.org/pdf/2510.11302)  

**Abstract**: Object detection systems have traditionally relied on supervised learning with manually annotated bounding boxes, achieving high accuracy at the cost of substantial annotation investment. The emergence of Vision-Language Models (VLMs) offers an alternative paradigm enabling zero-shot detection through natural language queries, eliminating annotation requirements but operating with reduced accuracy. This paper presents the first comprehensive cost-effectiveness analysis comparing supervised detection (YOLO) with zero-shot VLM inference (Gemini Flash 2.5). Through systematic evaluation on 1,000 stratified COCO images and 200 diverse product images spanning consumer electronics and rare categories, combined with detailed Total Cost of Ownership modeling, we establish quantitative break-even thresholds governing architecture selection. Our findings reveal that supervised YOLO achieves 91.2% accuracy versus 68.5% for zero-shot Gemini on standard categories, representing a 22.7 percentage point advantage that costs $10,800 in annotation for 100-category systems. However, this advantage justifies investment only beyond 55 million inferences, equivalent to 151,000 images daily for one year. Zero-shot Gemini demonstrates 52.3% accuracy on diverse product categories (ranging from highly web-prevalent consumer electronics at 75-85% to rare specialized equipment at 25-40%) where supervised YOLO achieves 0% due to architectural constraints preventing detection of untrained classes. Cost per Correct Detection analysis reveals substantially lower per-detection costs for Gemini ($0.00050 vs $0.143) at 100,000 inferences despite accuracy deficits. We develop decision frameworks demonstrating that optimal architecture selection depends critically on deployment volume, category stability, budget constraints, and accuracy requirements rather than purely technical performance metrics. 

**Abstract (ZH)**: 监督检测系统与零样本VLM推理的成本效益分析：以YOLO与Gemini Flash 2.5为例 

---
# Generalisation of automatic tumour segmentation in histopathological whole-slide images across multiple cancer types 

**Title (ZH)**: 跨多种癌症类型Histopathological全切片图像中自动肿瘤分割的泛化研究 

**Authors**: Ole-Johan Skrede, Manohar Pradhan, Maria Xepapadakis Isaksen, Tarjei Sveinsgjerd Hveem, Ljiljana Vlatkovic, Arild Nesbakken, Kristina Lindemann, Gunnar B Kristensen, Jenneke Kasius, Alain G Zeimet, Odd Terje Brustugun, Lill-Tove Rasmussen Busund, Elin H Richardsen, Erik Skaaheim Haug, Bjørn Brennhovd, Emma Rewcastle, Melinda Lillesand, Vebjørn Kvikstad, Emiel Janssen, David J Kerr, Knut Liestøl, Fritz Albregtsen, Andreas Kleppe  

**Link**: [PDF](https://arxiv.org/pdf/2510.11182)  

**Abstract**: Deep learning is expected to aid pathologists by automating tasks such as tumour segmentation. We aimed to develop one universal tumour segmentation model for histopathological images and examine its performance in different cancer types. The model was developed using over 20 000 whole-slide images from over 4 000 patients with colorectal, endometrial, lung, or prostate carcinoma. Performance was validated in pre-planned analyses on external cohorts with over 3 000 patients across six cancer types. Exploratory analyses included over 1 500 additional patients from The Cancer Genome Atlas. Average Dice coefficient was over 80% in all validation cohorts with en bloc resection specimens and in The Cancer Genome Atlas cohorts. No loss of performance was observed when comparing the universal model with models specialised on single cancer types. In conclusion, extensive and rigorous evaluations demonstrate that generic tumour segmentation by a single model is possible across cancer types, patient populations, sample preparations, and slide scanners. 

**Abstract (ZH)**: 深度学习有望通过自动化如肿瘤分割等任务来辅助病理学家。我们旨在开发一个统一的肿瘤分割模型以应用于组织病理图像，并考察其在不同癌症类型中的性能表现。该模型使用了来自4000多名患者（包括结直肠癌、子宫内膜癌、肺癌和前列腺癌）的超过20,000张全切片图像进行开发。性能在外部多癌种队列中进行了预设分析验证，涵盖超过3,000名患者的六种癌症类型。探索性分析还包括来自The Cancer Genome Atlas的1,500多名额外患者。所有验证队列（包括整块切除标本和The Cancer Genome Atlas队列）的平均Dice系数超过80%。与专门针对单一癌症类型的模型相比，通用模型未观察到性能下降。总之，广泛的严格评估表明，单一模型在不同癌症类型、患者群体、样本制备和切片扫描仪中实现通用肿瘤分割是可行的。 

---
# Text-Enhanced Panoptic Symbol Spotting in CAD Drawings 

**Title (ZH)**: CAD绘图中的文本增强全景符号检测 

**Authors**: Xianlin Liu, Yan Gong, Bohao Li, Jiajing Huang, Bowen Du, Junchen Ye, Liyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11091)  

**Abstract**: With the widespread adoption of Computer-Aided Design(CAD) drawings in engineering, architecture, and industrial design, the ability to accurately interpret and analyze these drawings has become increasingly critical. Among various subtasks, panoptic symbol spotting plays a vital role in enabling downstream applications such as CAD automation and design retrieval. Existing methods primarily focus on geometric primitives within the CAD drawings to address this task, but they face following major problems: they usually overlook the rich textual annotations present in CAD drawings and they lack explicit modeling of relationships among primitives, resulting in incomprehensive understanding of the holistic drawings. To fill this gap, we propose a panoptic symbol spotting framework that incorporates textual annotations. The framework constructs unified representations by jointly modeling geometric and textual primitives. Then, using visual features extract by pretrained CNN as the initial representations, a Transformer-based backbone is employed, enhanced with a type-aware attention mechanism to explicitly model the different types of spatial dependencies between various primitives. Extensive experiments on the real-world dataset demonstrate that the proposed method outperforms existing approaches on symbol spotting tasks involving textual annotations, and exhibits superior robustness when applied to complex CAD drawings. 

**Abstract (ZH)**: 基于文本注释的综合符号检测框架：解决CAD图纸综合理解问题 

---
# Source-Free Object Detection with Detection Transformer 

**Title (ZH)**: 无源物体检测：检测变换器方法 

**Authors**: Huizai Yao, Sicheng Zhao, Shuo Lu, Hui Chen, Yangyang Li, Guoping Liu, Tengfei Xing, Chenggang Yan, Jianhua Tao, Guiguang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.11090)  

**Abstract**: Source-Free Object Detection (SFOD) enables knowledge transfer from a source domain to an unsupervised target domain for object detection without access to source data. Most existing SFOD approaches are either confined to conventional object detection (OD) models like Faster R-CNN or designed as general solutions without tailored adaptations for novel OD architectures, especially Detection Transformer (DETR). In this paper, we introduce Feature Reweighting ANd Contrastive Learning NetworK (FRANCK), a novel SFOD framework specifically designed to perform query-centric feature enhancement for DETRs. FRANCK comprises four key components: (1) an Objectness Score-based Sample Reweighting (OSSR) module that computes attention-based objectness scores on multi-scale encoder feature maps, reweighting the detection loss to emphasize less-recognized regions; (2) a Contrastive Learning with Matching-based Memory Bank (CMMB) module that integrates multi-level features into memory banks, enhancing class-wise contrastive learning; (3) an Uncertainty-weighted Query-fused Feature Distillation (UQFD) module that improves feature distillation through prediction quality reweighting and query feature fusion; and (4) an improved self-training pipeline with a Dynamic Teacher Updating Interval (DTUI) that optimizes pseudo-label quality. By leveraging these components, FRANCK effectively adapts a source-pre-trained DETR model to a target domain with enhanced robustness and generalization. Extensive experiments on several widely used benchmarks demonstrate that our method achieves state-of-the-art performance, highlighting its effectiveness and compatibility with DETR-based SFOD models. 

**Abstract (ZH)**: 源数据无从获取的目标检测（Source-Free Object Detection, SFOD）使知识能够在无需访问源数据的情况下，从一个源领域转移到一个未监督的目标领域进行目标检测。目前大多数SFOD方法要么局限于传统的对象检测（OD）模型如Faster R-CNN，要么作为通用解决方案设计，缺乏针对新型OD架构，尤其是检测变换器（DETR）的针对性调整。在本文中，我们引入了特征重赋权重和对比学习网络（FRANCK），这是一种专门设计用于对DETR进行查询中心特征增强的新型SFOD框架。FRANCK包含四个关键组件：基于对象概率的采样重赋权重模块（OSSR），计算多尺度编码特征图上的注意力对象概率分数，并重新加权检测损失以强调未充分识别的区域；具有匹配增强的记忆银行的对比学习模块（CMMB），将多层级特征整合进记忆银行中，增强类别间的对比学习；通过预测质量加权和查询特征融合改进的不确定性加权查询融合特征蒸馏模块（UQFD）；以及带有动态教师更新间隔改进的自我训练管道（DTUI），优化伪标签质量。通过利用这些组件，FRANCK有效地将源预训练的DETR模型适应到具有增强鲁棒性和泛化能力的目标领域。在几个广泛应用的标准上的广泛实验表明，我们的方法达到了最先进的性能，突显了其在基于DETR的SFOD模型中的有效性和兼容性。 

---
# GeoVLMath: Enhancing Geometry Reasoning in Vision-Language Models via Cross-Modal Reward for Auxiliary Line Creation 

**Title (ZH)**: GeoVLMath：通过跨模态奖励辅助直线创建增强视觉-语言模型中的几何推理能力 

**Authors**: Shasha Guo, Liang Pang, Xi Wang, Yanling Wang, Huawei Shen, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11020)  

**Abstract**: Auxiliary lines are essential for solving complex geometric problems but remain challenging for large vision-language models (LVLMs). Rather than editing diagrams to draw auxiliary lines, which current image editing models struggle to render with geometric precision, we generate textual descriptions of auxiliary-line constructions to better align with the representational strengths of LVLMs. To bridge the gap between textual descriptions and spatial structure, we propose a reinforcement learning framework that enhances diagram-text alignment. At the core of our approach is a cross-modal reward that evaluates how well the generated auxiliary-line description for an original diagram matches a ground-truth auxiliary-line diagram. Built on this reward, we present GeoVLMath, an open-source LVLM tailored to auxiliary-line reasoning in solid geometry. This fine-grained signal drives a GRPO-based RL stage, yielding precise diagram-text alignment. To support training, we develop a scalable data creation pipeline and construct AuxSolidMath, a dataset of 3,018 real-exam geometry problems with paired diagrams and aligned textual fields. At the 3B and 7B scales, GeoVLMath achieves competitive and often superior performance compared with strong open-source and proprietary LVLMs on auxiliary-line reasoning benchmarks. 

**Abstract (ZH)**: 辅助线对于解决复杂的几何问题至关重要，但对大型视觉-语言模型（LVLMs）来说仍具有挑战性。我们不通过编辑图形来绘制辅助线，而是生成辅助线构建的文字描述，以更好地与LVLMs的表征优势相契合。为了弥合文本描述与空间结构之间的差距，我们提出了一种强化学习框架，以增强图形-文本对齐。该方法的核心是一个跨模态奖励，用于评估生成的辅助线描述与真实辅助线图形之间的匹配程度。基于这一奖励，我们提出了GeoVLMath，一个适用于固态几何辅助线推理的开源LVLM。这一精细信号驱动基于GRPO的强化学习阶段，实现精确的图形-文本对齐。为了支持训练，我们开发了一个可扩展的数据创建管道，并构建了包含3,018个实际考试几何问题的AuxSolidMath数据集，这些问题配有配对的图形和对齐的文字字段。在3B和7B规模下，GeoVLMath在辅助线推理基准测试中与强大的开源和专有LVLMs相比，表现出竞争力甚至更优的性能。 

---
# Unify Variables in Neural Scaling Laws for General Audio Representations via Embedding Effective Rank 

**Title (ZH)**: 通过嵌入有效秩统一神经尺度定律中的变量以实现通用音频表示 

**Authors**: Xuyao Deng, Yanjie Sun, Yong Dou, Kele Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10948)  

**Abstract**: Scaling laws have profoundly shaped our understanding of model performance in computer vision and natural language processing, yet their application to general audio representation learning remains underexplored. A key challenge lies in the multifactorial nature of general audio representation-representation quality is jointly influenced by variables such as audio length, embedding dimensionality, model depth, model architecture, data volume, etc., many of which are difficult to isolate or express analytically. In this work, we present a systematic study of scaling laws for general audio representations by utilizing embedding effective rank (RankMe) as a unifying metric that encapsulates the impact of diverse variables on representation quality. RankMe enables a label-free, information-theoretic quantification of audio embeddings, allowing us to examine scaling behaviors across a wide hyper-parameter space, including model size, training data volume, computational budget, architectural configurations, etc. Our empirical findings reveal a consistent power-law relationship between RankMe and representation quality, suggesting that embedding effective rank serves as a reliable proxy for assessing and predicting model performance in audio representation learning. This work not only validates the applicability of classical scaling principles to the general audio domain but also offers a theoretically grounded and empirically robust framework for guiding future model scaling strategies in audio foundation models. 

**Abstract (ZH)**: 尺度规律在塑造计算机视觉和自然语言处理中模型性能理解方面发挥了深远影响，但在通用音频表示学习中的应用仍鲜有人探索。一个关键挑战在于多因素性，通用音频表示的质量由音频长度、嵌入维度、模型深度、模型架构、数据量等多种变量共同影响，其中许多变量难以分离或通过分析表达。在本文中，我们通过使用嵌入有效秩（RankMe）作为一个统一的指标，系统地研究了通用音频表示的尺度规律，RankMe能综合反映多种变量对表示质量的影响。RankMe允许我们通过信息论方式无标签地量化音频嵌入，从而使我们能够跨越包括模型大小、训练数据量、计算预算、架构配置等广泛超参数空间来研究规模效应。我们的实证研究发现，RankMe与表示质量之间存在一致的幂律关系，这表明嵌入有效秩可以作为评估和预测音频表示学习中模型性能的可靠代理。本研究不仅验证了经典尺度原则在通用音频领域的适用性，还为指导未来音频基础模型的规模策略提供了理论依据和实证支持框架。 

---
# MSCloudCAM: Cross-Attention with Multi-Scale Context for Multispectral Cloud Segmentation 

**Title (ZH)**: MSCloudCAM: 多尺度上下文的交叉注意力多光谱云分割 

**Authors**: Md Abdullah Al Mazid, Liangdong Deng, Naphtali Rishe  

**Link**: [PDF](https://arxiv.org/pdf/2510.10802)  

**Abstract**: Clouds remain a critical challenge in optical satellite imagery, hindering reliable analysis for environmental monitoring, land cover mapping, and climate research. To overcome this, we propose MSCloudCAM, a Cross-Attention with Multi-Scale Context Network tailored for multispectral and multi-sensor cloud segmentation. Our framework exploits the spectral richness of Sentinel-2 (CloudSEN12) and Landsat-8 (L8Biome) data to classify four semantic categories: clear sky, thin cloud, thick cloud, and cloud shadow. MSCloudCAM combines a Swin Transformer backbone for hierarchical feature extraction with multi-scale context modules ASPP and PSP for enhanced scale-aware learning. A Cross-Attention block enables effective multisensor and multispectral feature fusion, while the integration of an Efficient Channel Attention Block (ECAB) and a Spatial Attention Module adaptively refine feature representations. Comprehensive experiments on CloudSEN12 and L8Biome demonstrate that MSCloudCAM delivers state-of-the-art segmentation accuracy, surpassing leading baseline architectures while maintaining competitive parameter efficiency and FLOPs. These results underscore the model's effectiveness and practicality, making it well-suited for large-scale Earth observation tasks and real-world applications. 

**Abstract (ZH)**: 风云遥感影像中的云掩模问题仍然是光学卫星影像分析的关键挑战，阻碍了环境监测、土地覆盖制图和气候研究的可靠分析。为了解决这一问题，我们提出了MSCloudCAM，一种适用于多光谱和多传感器云分割的跨注意力多尺度上下文网络。该框架利用Sentinel-2（CloudSEN12）和Landsat-8（L8Biome）数据的光谱丰富性，用于分类四种语义类别：晴空、薄云、厚云和云影。MSCloudCAM 结合了Swin Transformer骨干网进行分层特征提取，以及 ASPP 和 PSP 多尺度上下文模块以增强尺度感知学习。跨注意力块实现了有效的多传感器和多光谱特征融合，而Efficient Channel Attention Block (ECAB) 和 Spatial Attention Module 的集成则适应性地细化特征表示。CloudSEN12 和 L8Biome 上的综合实验表明，MSCloudCAM 在分割精度上达到了最先进的水平，超越了领先的基本架构，同时保持了具有竞争力的参数效率和FLOPs。这些结果突显了该模型的有效性和实用性，使其适合于大规模地球观测任务和实际应用。 

---
# DISC-GAN: Disentangling Style and Content for Cluster-Specific Synthetic Underwater Image Generation 

**Title (ZH)**: DISC-GAN: 解耦风格与内容以生成簇特定的合成水下图像 

**Authors**: Sneha Varur, Anirudh R Hanchinamani, Tarun S Bagewadi, Uma Mudenagudi, Chaitra D Desai, Sujata C, Padmashree Desai, Sumit Meharwade  

**Link**: [PDF](https://arxiv.org/pdf/2510.10782)  

**Abstract**: In this paper, we propose a novel framework, Disentangled Style-Content GAN (DISC-GAN), which integrates style-content disentanglement with a cluster-specific training strategy towards photorealistic underwater image synthesis. The quality of synthetic underwater images is challenged by optical due to phenomena such as color attenuation and turbidity. These phenomena are represented by distinct stylistic variations across different waterbodies, such as changes in tint and haze. While generative models are well-suited to capture complex patterns, they often lack the ability to model the non-uniform conditions of diverse underwater environments. To address these challenges, we employ K-means clustering to partition a dataset into style-specific domains. We use separate encoders to get latent spaces for style and content; we further integrate these latent representations via Adaptive Instance Normalization (AdaIN) and decode the result to produce the final synthetic image. The model is trained independently on each style cluster to preserve domain-specific characteristics. Our framework demonstrates state-of-the-art performance, obtaining a Structural Similarity Index (SSIM) of 0.9012, an average Peak Signal-to-Noise Ratio (PSNR) of 32.5118 dB, and a Frechet Inception Distance (FID) of 13.3728. 

**Abstract (ZH)**: 一种新颖的解耦风格-内容GAN框架：_DISC-GAN及其在水下图像合成中的应用 

---
# Scalable Face Security Vision Foundation Model for Deepfake, Diffusion, and Spoofing Detection 

**Title (ZH)**: 可扩展的人脸安全视觉基础模型用于深伪、扩散和欺骗检测 

**Authors**: Gaojian Wang, Feng Lin, Tong Wu, Zhisheng Yan, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.10663)  

**Abstract**: With abundant, unlabeled real faces, how can we learn robust and transferable facial representations to boost generalization across various face security tasks? We make the first attempt and propose FS-VFM, a scalable self-supervised pre-training framework, to learn fundamental representations of real face images. We introduce three learning objectives, namely 3C, that synergize masked image modeling (MIM) and instance discrimination (ID), empowering FS-VFM to encode both local patterns and global semantics of real faces. Specifically, we formulate various facial masking strategies for MIM and devise a simple yet effective CRFR-P masking, which explicitly prompts the model to pursue meaningful intra-region Consistency and challenging inter-region Coherency. We present a reliable self-distillation mechanism that seamlessly couples MIM with ID to establish underlying local-to-global Correspondence. After pre-training, vanilla vision transformers (ViTs) serve as universal Vision Foundation Models for downstream Face Security tasks: cross-dataset deepfake detection, cross-domain face anti-spoofing, and unseen diffusion facial forensics. To efficiently transfer the pre-trained FS-VFM, we further propose FS-Adapter, a lightweight plug-and-play bottleneck atop the frozen backbone with a novel real-anchor contrastive objective. Extensive experiments on 11 public benchmarks demonstrate that our FS-VFM consistently generalizes better than diverse VFMs, spanning natural and facial domains, fully, weakly, and self-supervised paradigms, small, base, and large ViT scales, and even outperforms SOTA task-specific methods, while FS-Adapter offers an excellent efficiency-performance trade-off. The code and models are available on this https URL. 

**Abstract (ZH)**: 如何利用丰富的无标签真实人脸图像学习鲁棒且可迁移的面部表示以增强各类面部安全任务的一般化能力？我们首次进行尝试并提出FS-VFM，一种可扩展的自监督预训练框架，以学习真实人脸图像的基本表示。我们引入了三种学习目标，即3C，融合了掩码图像建模（MIM）和实例区分（ID），使FS-VFM能够编码真实人脸的局部模式和全局语义。具体而言，我们为MIM制定了各种面部掩码策略，并设计了一种简单有效的CRFR-P掩码，明确地促使模型追求有意义的区域内一致性以及具有挑战性的区域间一致性。我们提出了一种可靠的自蒸馏机制，无缝地将MIM与ID耦合起来，建立底层的局部到全局对应关系。预训练后，通用的视觉基础模型（ViTs）作为下游面部安全任务：跨数据集深度伪造检测、跨域人脸防篡改和未知扩散面部法医检验的通用组件。为了高效地迁移预训练的FS-VFM，我们进一步提出FS-Adapter，这是一种基于冻结主干的轻量级即插即用瓶颈结构，并带有新的真实锚点对比目标。在11个公开基准上的广泛实验表明，我们的FS-VFM在多种视觉变换器规模和自监督范式下，以及自然和面部领域内，均表现出更出色的泛化能力，甚至超越了现有的特定任务方法，而FS-Adapter则提供了卓越的效率-性能权衡。代码和模型可在以下链接获取：this https URL。 

---
# DEMO: Disentangled Motion Latent Flow Matching for Fine-Grained Controllable Talking Portrait Synthesis 

**Title (ZH)**: DEMO: 解耦运动潜在流匹配的细腻粒度可控 Talking Portrait 合成 

**Authors**: Peiyin Chen, Zhuowei Yang, Hui Feng, Sheng Jiang, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2510.10650)  

**Abstract**: Audio-driven talking-head generation has advanced rapidly with diffusion-based generative models, yet producing temporally coherent videos with fine-grained motion control remains challenging. We propose DEMO, a flow-matching generative framework for audio-driven talking-portrait video synthesis that delivers disentangled, high-fidelity control of lip motion, head pose, and eye gaze. The core contribution is a motion auto-encoder that builds a structured latent space in which motion factors are independently represented and approximately orthogonalized. On this disentangled motion space, we apply optimal-transport-based flow matching with a transformer predictor to generate temporally smooth motion trajectories conditioned on audio. Extensive experiments across multiple benchmarks show that DEMO outperforms prior methods in video realism, lip-audio synchronization, and motion fidelity. These results demonstrate that combining fine-grained motion disentanglement with flow-based generative modeling provides a powerful new paradigm for controllable talking-head video synthesis. 

**Abstract (ZH)**: 基于音频驱动的Head Motion生成，扩散型生成模型推动了头像视频生成的快速发展，但实时前后一致的、细粒度运动控制的视频生成仍然具有挑战性。我们提出了一种基于流匹配的生成框架DEMO，用于音频驱动的肖像视频合成，提供了解耦、高保真度的唇动、头部姿态和眼睛注视控制。核心贡献是一种运动自编码器，它构建了一个结构化的潜在空间，在该空间中，运动因素独立表示并通过近似正交化。在这一解耦的运动空间上，我们应用基于运输最优的流匹配和变压器预测器，根据音频条件生成平滑的运动轨迹。多基准实验显示，DEMO在视频真实性、唇音同步和运动保真度方面优于先前方法。这些结果表明，将细粒度运动解耦与基于流的生成建模相结合，为可控头像视频合成提供了一个强大新的范式。 

---
# Mesh-Gait: A Unified Framework for Gait Recognition Through Multi-Modal Representation Learning from 2D Silhouettes 

**Title (ZH)**: Mesh-Gait：一种基于多模态表示学习的2D轮廓统一框架的人体姿态识别方法 

**Authors**: Zhao-Yang Wang, Jieneng Chen, Jiang Liu, Yuxiang Guo, Rama Chellappa  

**Link**: [PDF](https://arxiv.org/pdf/2510.10406)  

**Abstract**: Gait recognition, a fundamental biometric technology, leverages unique walking patterns for individual identification, typically using 2D representations such as silhouettes or skeletons. However, these methods often struggle with viewpoint variations, occlusions, and noise. Multi-modal approaches that incorporate 3D body shape information offer improved robustness but are computationally expensive, limiting their feasibility for real-time applications. To address these challenges, we introduce Mesh-Gait, a novel end-to-end multi-modal gait recognition framework that directly reconstructs 3D representations from 2D silhouettes, effectively combining the strengths of both modalities. Compared to existing methods, directly learning 3D features from 3D joints or meshes is complex and difficult to fuse with silhouette-based gait features. To overcome this, Mesh-Gait reconstructs 3D heatmaps as an intermediate representation, enabling the model to effectively capture 3D geometric information while maintaining simplicity and computational efficiency. During training, the intermediate 3D heatmaps are gradually reconstructed and become increasingly accurate under supervised learning, where the loss is calculated between the reconstructed 3D joints, virtual markers, and 3D meshes and their corresponding ground truth, ensuring precise spatial alignment and consistent 3D structure. Mesh-Gait extracts discriminative features from both silhouettes and reconstructed 3D heatmaps in a computationally efficient manner. This design enables the model to capture spatial and structural gait characteristics while avoiding the heavy overhead of direct 3D reconstruction from RGB videos, allowing the network to focus on motion dynamics rather than irrelevant visual details. Extensive experiments demonstrate that Mesh-Gait achieves state-of-the-art accuracy. The code will be released upon acceptance of the paper. 

**Abstract (ZH)**: 步态识别，一种基本的生物识别技术，利用独特的行走模式进行个体识别，通常使用2D表示，如轮廓或骨架。然而，这些方法往往难以处理视角变化、遮挡和噪声。多模态方法结合3D身体形状信息可以提高鲁棒性，但计算成本高，限制了其在实时应用中的可行性。为应对这些挑战，我们提出了一种新颖的端到端多模态步态识别框架Mesh-Gait，直接从2D轮廓重建3D表示，有效结合了两种模态的优势。与现有方法相比，直接从3D关节或网格中学习3D特征并将其与基于轮廓的步态特征融合既复杂又困难。为克服这一挑战，Mesh-Gait 构建了3D热图作为中间表示，使模型能够有效捕捉3D几何信息，同时保持简洁性和计算效率。在训练过程中，中间的3D热图在监督学习下逐渐重建并变得越来越准确，损失函数在重建的3D关节、虚拟标记、3D网格及其对应的地面真相之间计算，确保精确的空间对齐和一致的3D结构。Mesh-Gait 以计算高效的方式从轮廓和重建的3D热图中提取判别性特征。该设计使模型能够捕获空间和结构的步态特征，同时避免了从RGB视频直接进行3D重建时的沉重开销，使网络能够专注于运动动态而非无关的视觉细节。大量实验表明，Mesh-Gait 达到了最先进的准确率。代码将在论文被接受后发布。 

---
# Identifying bias in CNN image classification using image scrambling and transforms 

**Title (ZH)**: 使用图像杂乱和变换识别CNN图像分类中的偏差 

**Authors**: Sai Teja Erukude  

**Link**: [PDF](https://arxiv.org/pdf/2510.10383)  

**Abstract**: CNNs are now prevalent as the primary choice for most machine vision problems due to their superior rate of classification and the availability of user-friendly libraries. These networks effortlessly identify and select features in a non-intuitive data-driven manner, making it difficult to determine which features were most influential. That leads to a ``black box", where users cannot know how the image data are analyzed but rely on empirical results. Therefore the decision-making process can be biased by background information that is difficult to detect. Here we discuss examples of such hidden biases and propose techniques for identifying them, methods to distinguish between contextual information and background noise, and explore whether CNNs learn from irrelevant features. One effective approach to identify dataset bias is to classify blank background parts of the images. However, in some situations a blank background in the images is not available, making it more difficult to separate the foreground information from the blank background. Such parts of the image can also be considered contextual learning, not necessarily bias. To overcome this, we propose two approaches that were tested on six different datasets, including natural, synthetic, and hybrid datasets. The first method involves dividing images into smaller, non-overlapping tiles of various sizes, which are then shuffled randomly, making classification more challenging. The second method involves the application of several image transforms, including Fourier, Wavelet transforms, and Median filter, and their combinations. These transforms help recover background noise information used by CNN to classify images. Results indicate that this method can effectively distinguish between contextual information and background noise, and alert on the presence of background noise even without the need to use background information. 

**Abstract (ZH)**: CNNs在图像数据分类中的隐藏偏差识别与区分方法 

---
# Ortho-Fuse: Orthomosaic Generation for Sparse High-Resolution Crop Health Maps Through Intermediate Optical Flow Estimation 

**Title (ZH)**: 正交融合：通过中间光学流估计生成稀疏高分辨率作物健康图的正交全景图生成方法 

**Authors**: Rugved Katole, Christopher Stewart  

**Link**: [PDF](https://arxiv.org/pdf/2510.10360)  

**Abstract**: AI-driven crop health mapping systems offer substantial advantages over conventional monitoring approaches through accelerated data acquisition and cost reduction. However, widespread farmer adoption remains constrained by technical limitations in orthomosaic generation from sparse aerial imagery datasets. Traditional photogrammetric reconstruction requires 70-80\% inter-image overlap to establish sufficient feature correspondences for accurate geometric registration. AI-driven systems operating under resource-constrained conditions cannot consistently achieve these overlap thresholds, resulting in degraded reconstruction quality that undermines user confidence in autonomous monitoring technologies. In this paper, we present Ortho-Fuse, an optical flow-based framework that enables the generation of a reliable orthomosaic with reduced overlap requirements. Our approach employs intermediate flow estimation to synthesize transitional imagery between consecutive aerial frames, artificially augmenting feature correspondences for improved geometric reconstruction. Experimental validation demonstrates a 20\% reduction in minimum overlap requirements. We further analyze adoption barriers in precision agriculture to identify pathways for enhanced integration of AI-driven monitoring systems. 

**Abstract (ZH)**: 基于AI的作物健康映射系统通过加速数据获取和降低成本提供了显著优势，但广泛农民采纳受到稀疏航拍影像数据正射影像生成技术限制。传统摄影测量重建需要70-80%的影像重叠以建立足够的特征对应关系，确保几何注册精度。在资源受限条件下运行的AI驱动系统无法一致地达到这些重叠阈值，导致重建质量下降，损害用户对自主监测技术的信心。本文介绍了一种基于光学流的框架Ortho-Fuse，能够以减少的重叠要求生成可靠的正射影像。通过中间流估计合成连续航拍帧之间的过渡影像，人为增强特征对应关系，提高几何重建效果。实验验证显示重叠要求减少了20%。我们进一步分析了精准农业中的采纳障碍，以识别增强AI驱动监测系统集成的途径。 

---
# From Programs to Poses: Factored Real-World Scene Generation via Learned Program Libraries 

**Title (ZH)**: 从程序到姿态：通过学习程序库生成事实化的现实场景 

**Authors**: Joy Hsu, Emily Jin, Jiajun Wu, Niloy J. Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2510.10292)  

**Abstract**: Real-world scenes, such as those in ScanNet, are difficult to capture, with highly limited data available. Generating realistic scenes with varied object poses remains an open and challenging task. In this work, we propose FactoredScenes, a framework that synthesizes realistic 3D scenes by leveraging the underlying structure of rooms while learning the variation of object poses from lived-in scenes. We introduce a factored representation that decomposes scenes into hierarchically organized concepts of room programs and object poses. To encode structure, FactoredScenes learns a library of functions capturing reusable layout patterns from which scenes are drawn, then uses large language models to generate high-level programs, regularized by the learned library. To represent scene variations, FactoredScenes learns a program-conditioned model to hierarchically predict object poses, and retrieves and places 3D objects in a scene. We show that FactoredScenes generates realistic, real-world rooms that are difficult to distinguish from real ScanNet scenes. 

**Abstract (ZH)**: FactoredScenes：通过房间程序和物体姿态的分解生成真实的3D场景 

---
# MRI Brain Tumor Detection with Computer Vision 

**Title (ZH)**: 基于计算机视觉的MRI脑肿瘤检测 

**Authors**: Jack Krolik, Jake Lynn, John Henry Rudden, Dmytro Vremenko  

**Link**: [PDF](https://arxiv.org/pdf/2510.10250)  

**Abstract**: This study explores the application of deep learning techniques in the automated detection and segmentation of brain tumors from MRI scans. We employ several machine learning models, including basic logistic regression, Convolutional Neural Networks (CNNs), and Residual Networks (ResNet) to classify brain tumors effectively. Additionally, we investigate the use of U-Net for semantic segmentation and EfficientDet for anchor-based object detection to enhance the localization and identification of tumors. Our results demonstrate promising improvements in the accuracy and efficiency of brain tumor diagnostics, underscoring the potential of deep learning in medical imaging and its significance in improving clinical outcomes. 

**Abstract (ZH)**: 本研究探讨了深度学习技术在自动检测和分割MRI扫描中脑肿瘤的应用。我们采用多种机器学习模型，包括基础逻辑回归、卷积神经网络（CNN）和残差网络（ResNet），以有效分类脑肿瘤。此外，我们还研究了U-Net在语义分割中的应用和EfficientDet在基于锚点的对象检测中的应用，以提高肿瘤的定位和识别。研究结果表明，深度学习在脑肿瘤诊断中的准确性和效率方面取得了令人鼓舞的改进，突显了其在医学成像中的潜力及其对改善临床结果的重要性。 

---
# HccePose(BF): Predicting Front \& Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation 

**Title (ZH)**: HccePose(BF): 预测前后表面以构建超密集2D-3D对应关系进行姿态估计 

**Authors**: Yulin Wang, Mengting Hu, Hongli Li, Chen Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.10177)  

**Abstract**: In pose estimation for seen objects, a prevalent pipeline involves using neural networks to predict dense 3D coordinates of the object surface on 2D images, which are then used to establish dense 2D-3D correspondences. However, current methods primarily focus on more efficient encoding techniques to improve the precision of predicted 3D coordinates on the object's front surface, overlooking the potential benefits of incorporating the back surface and interior of the object. To better utilize the full surface and interior of the object, this study predicts 3D coordinates of both the object's front and back surfaces and densely samples 3D coordinates between them. This process creates ultra-dense 2D-3D correspondences, effectively enhancing pose estimation accuracy based on the Perspective-n-Point (PnP) algorithm. Additionally, we propose Hierarchical Continuous Coordinate Encoding (HCCE) to provide a more accurate and efficient representation of front and back surface coordinates. Experimental results show that, compared to existing state-of-the-art (SOTA) methods on the BOP website, the proposed approach outperforms across seven classic BOP core datasets. Code is available at this https URL. 

**Abstract (ZH)**: 在已见物体的姿势估计中，一个常见的流程是使用神经网络在二维图像上预测物体表面的密集3D坐标，进而建立密集的2D-3D对应关系。然而，当前的方法主要侧重于更高效的编码技术以提高预测的3D坐标精度，忽略了结合物体背面和内部信息的潜在好处。为了更好地利用物体的完整表面和内部，本研究预测了物体正面和背面表面的3D坐标，并在两者之间进行密集采样。这一过程创建了超密集的2D-3D对应关系，基于视角点法（PnP）算法有效提升了姿势估计的准确性。此外，我们提出了层次连续坐标编码（HCCE）以提供更准确和高效的正面和背面表面坐标表示。实验结果显示，与BOP网站上现有的最先进的（SOTA）方法相比，所提出的方法在七个经典BOP核心数据集上表现更优。代码详见此链接。 

---
# SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation 

**Title (ZH)**: SaFiRe: 眯眼- fixation 重迭代用于图像分割的引用方法 

**Authors**: Zhenjie Mao, Yuhuan Yang, Chaofan Ma, Dongsheng Jiang, Jiangchao Yao, Ya Zhang, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10160)  

**Abstract**: Referring Image Segmentation (RIS) aims to segment the target object in an image given a natural language expression. While recent methods leverage pre-trained vision backbones and more training corpus to achieve impressive results, they predominantly focus on simple expressions--short, clear noun phrases like "red car" or "left girl". This simplification often reduces RIS to a key word/concept matching problem, limiting the model's ability to handle referential ambiguity in expressions. In this work, we identify two challenging real-world scenarios: object-distracting expressions, which involve multiple entities with contextual cues, and category-implicit expressions, where the object class is not explicitly stated. To address the challenges, we propose a novel framework, SaFiRe, which mimics the human two-phase cognitive process--first forming a global understanding, then refining it through detail-oriented inspection. This is naturally supported by Mamba's scan-then-update property, which aligns with our phased design and enables efficient multi-cycle refinement with linear complexity. We further introduce aRefCOCO, a new benchmark designed to evaluate RIS models under ambiguous referring expressions. Extensive experiments on both standard and proposed datasets demonstrate the superiority of SaFiRe over state-of-the-art baselines. 

**Abstract (ZH)**: 参考图像分割（RIS）旨在给定自然语言表达的情况下对图像中的目标对象进行分割。尽管近期方法利用预训练的视觉 backbone 和更多的训练语料取得了显著成果，但它们主要侧重于简单的表达——如“红色汽车”或“左侧的女孩”这样的短且清晰的名词短语。这种简化往往将RIS简化为关键词/概念匹配问题，限制了模型处理表达中的引用歧义的能力。在本文中，我们识别了两个具有挑战性的现实场景：具有上下文提示的对象分散表达，以及包含未明确陈述的对象类别的情况。为应对这些挑战，我们提出了一种新的框架 SaFiRe，模拟人类的两阶段认知过程——首先形成全局理解，然后通过细节审视进行细化。Mamba 的扫描-更新特性天然支持这一设计，使我们的多阶段设计能够高效地进行线性复杂度的多轮细化。此外，我们引入了 aRefCOCO，这是一个新的基准，用于评估在模糊引用表达下的RIS模型性能。广泛的实验结果表明，SaFiRe 在标准数据集和提出的数据集上均优于现有的基线方法。 

---
# DeepFusionNet: Autoencoder-Based Low-Light Image Enhancement and Super-Resolution 

**Title (ZH)**: DeepFusionNet：基于自编码器的低光照图像增强与超分辨率 

**Authors**: Halil Hüseyin Çalışkan, Talha Koruk  

**Link**: [PDF](https://arxiv.org/pdf/2510.10122)  

**Abstract**: Computer vision and image processing applications suffer from dark and low-light images, particularly during real-time image transmission. Currently, low light and dark images are converted to bright and colored forms using autoencoders; however, these methods often achieve low SSIM and PSNR scores and require high computational power due to their large number of parameters. To address these challenges, the DeepFusionNet architecture has been developed. According to the results obtained with the LOL-v1 dataset, DeepFusionNet achieved an SSIM of 92.8% and a PSNR score of 26.30, while containing only approximately 2.5 million parameters. On the other hand, conversion of blurry and low-resolution images into high-resolution and blur-free images has gained importance in image processing applications. Unlike GAN-based super-resolution methods, an autoencoder-based super resolution model has been developed that contains approximately 100 thousand parameters and uses the DeepFusionNet architecture. According to the results of the tests, the DeepFusionNet based super-resolution method achieved a PSNR of 25.30 and a SSIM score of 80.7 percent according to the validation set. 

**Abstract (ZH)**: 计算机视觉和图像处理应用程序在处理暗光和低光图像时遇到挑战，特别是在实时图像传输中。目前，通过自编码器将低光照和暗光图像转换为明亮彩色的形式，但这些方法往往获得较低的SSIM和PSNR分数，并且由于参数众多需要大量计算资源。为解决这些问题，开发了DeepFusionNet架构。根据使用LOL-v1数据集获得的结果，DeepFusionNet实现了92.8%的SSIM和26.30的PSNR分数，其参数数量约为250万。另一方面，高分辨率和无模糊图像的转换在图像处理应用中变得重要。不同于基于GAN的超分辨率方法，开发了一种基于自编码器的超分辨率模型，参数数量约为10万，并采用DeepFusionNet架构。根据测试结果，基于DeepFusionNet的超分辨率方法在验证集上实现了25.30的PSNR和80.7%的SSIM分数。 

---
# Uncertainty-Aware Post-Detection Framework for Enhanced Fire and Smoke Detection in Compact Deep Learning Models 

**Title (ZH)**: 考虑不确定性后处理框架以增强紧凑深度学习模型中的火灾和烟雾检测 

**Authors**: Aniruddha Srinivas Joshi, Godwyn James William, Shreyas Srinivas Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2510.10108)  

**Abstract**: Accurate fire and smoke detection is critical for safety and disaster response, yet existing vision-based methods face challenges in balancing efficiency and reliability. Compact deep learning models such as YOLOv5n and YOLOv8n are widely adopted for deployment on UAVs, CCTV systems, and IoT devices, but their reduced capacity often results in false positives and missed detections. Conventional post-detection methods such as Non-Maximum Suppression and Soft-NMS rely only on spatial overlap, which can suppress true positives or retain false alarms in cluttered or ambiguous fire scenes. To address these limitations, we propose an uncertainty aware post-detection framework that rescales detection confidences using both statistical uncertainty and domain relevant visual cues. A lightweight Confidence Refinement Network integrates uncertainty estimates with color, edge, and texture features to adjust detection scores without modifying the base model. Experiments on the D-Fire dataset demonstrate improved precision, recall, and mean average precision compared to existing baselines, with only modest computational overhead. These results highlight the effectiveness of post-detection rescoring in enhancing the robustness of compact deep learning models for real-world fire and smoke detection. 

**Abstract (ZH)**: 准确的火灾和烟雾检测对于安全和灾害响应至关重要，但现有的基于视觉的方法在效率和可靠性之间难以平衡。紧凑型深度学习模型如YOLOv5n和YOLOv8n广泛应用在无人机、闭路电视系统和物联网设备上，但它们的容量缩减往往导致误报和漏报。传统后检测方法如非最大抑制和Soft-NMS仅依赖于空间重叠，这在复杂或多义的火灾场景中可能抑制真实阳性或保留误报。为解决这些问题，我们提出一个意识不确定性的后检测框架，使用统计不确定性和领域相关视觉线索重新标定检测置信度。一个轻量级的置信度精炼网络将不确定性估计与颜色、边缘和纹理特征整合，调整检测分数而不修改基础模型。在D-Fire数据集上的实验表明，与现有基线相比，该方法在精度、召回率和平均精度上有所提高，且计算 overhead 较小。这些结果突显了后检测重新评分在增强紧凑型深度学习模型在实际火灾和烟雾检测中的鲁棒性方面的有效性。 

---
# Translution: Unifying Self-attention and Convolution for Adaptive and Relative Modeling 

**Title (ZH)**: 统一自注意力与卷积以实现自适应和相对建模 

**Authors**: Hehe Fan, Yi Yang, Mohan Kankanhalli, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10060)  

**Abstract**: When modeling a given type of data, we consider it to involve two key aspects: 1) identifying relevant elements (e.g., image pixels or textual words) to a central element, as in a convolutional receptive field, or to a query element, as in self-attention, and 2) encoding these tokens effectively. Self-attention can adaptively identify these elements but relies on absolute positional embedding for structural representation learning. In contrast, convolution encodes elements in a relative manner, yet their fixed kernel size limits their ability to adaptively select the relevant elements. In this paper, we introduce Translution, an operation that unifies the adaptive identification capability of self-attention and the relative encoding advantage of convolution. However, this integration leads to a substantial increase in the number of parameters, exceeding most currently available computational resources. Therefore, we propose a lightweight variant of Translution, named {\alpha}-Translution. Experiments on computer vision and natural language processing tasks show that Translution (including {\alpha}-Translution) achieves superior accuracy compared to self-attention. The code is available at this https URL. 

**Abstract (ZH)**: 在建模给定类型的数据时，我们认为涉及两个关键方面：1）识别与中心元素或查询元素相关的元素（例如，在卷积感受野中识别图像像素，在自我注意中识别文本单词），2）有效地编码这些标记。自我注意可以自适应地识别这些元素，但依赖绝对位置嵌入进行结构表示学习。相比之下，卷积以相对方式编码元素，但其固定的核尺寸限制了其自适应选择相关元素的能力。在本文中，我们引入了Translution操作，它统一了自我注意的自适应识别能力和卷积的相对编码优势。然而，这种整合导致参数数量显著增加，超过了大多数当前可用的计算资源。因此，我们提出了一种轻量级的Translution变体，命名为{\alpha}-Translution。实验表明，Translution（包括{\alpha}-Translution）在计算机视觉和自然语言处理任务中实现了优于自我注意的准确性。相关代码可以在以下网址获取。 

---
# Think Twice to See More: Iterative Visual Reasoning in Medical VLMs 

**Title (ZH)**: 慎重思考以洞察更多：医疗VLM中的迭代视觉推理 

**Authors**: Kaitao Chen, Shaohao Rui, Yankai Jiang, Jiamin Wu, Qihao Zheng, Chunfeng Song, Xiaosong Wang, Mu Zhou, Mianxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10052)  

**Abstract**: Medical vision-language models (VLMs) excel at image-text understanding but typically rely on a single-pass reasoning that neglects localized visual cues. In clinical practice, however, human experts iteratively scan, focus, and refine the regions of interest before reaching a final diagnosis. To narrow this machine-human perception gap, we introduce ViTAR, a novel VLM framework that emulates the iterative reasoning process of human experts through a cognitive chain of "think-act-rethink-answer". ViTAR treats medical images as interactive objects, enabling models to engage multi-step visual reasoning. To support this approach, we curate a high-quality instruction dataset comprising 1K interactive examples that encode expert-like diagnostic behaviors. In addition, a 16K visual question answering training data has been curated towards fine-grained visual diagnosis. We introduce a two-stage training strategy that begins with supervised fine-tuning to guide cognitive trajectories, followed by the reinforcement learning to optimize decision-making. Extensive evaluations demonstrate that ViTAR outperforms strong state-of-the-art models. Visual attention analysis reveals that from the "think" to "rethink" rounds, ViTAR increasingly anchors visual grounding to clinically critical regions and maintains high attention allocation to visual tokens during reasoning, providing mechanistic insight into its improved performance. These findings demonstrate that embedding expert-style iterative thinking chains into VLMs enhances both performance and trustworthiness of medical AI. 

**Abstract (ZH)**: 医学视觉语言模型（VLMs）在图像-文本理解方面表现出色，但通常依赖于忽视局部视觉提示的单次推理。然而，在临床实践中，人类专家会逐步扫描、聚焦并精炼感兴趣区域，最终得出诊断结果。为了缩小机器与人类感知之间的差距，我们提出了ViTAR，这是一种新的VLM框架，通过“思考-行动-再思考-回答”的认知链模拟人类专家的迭代推理过程。ViTAR将医学图像视为可交互的对象，使模型能够进行多步视觉推理。为了支持这一方法，我们构建了一个高质量的指令数据集，包含1000个交互式示例，这些示例编码了专家级的诊断行为。此外，我们还构建了一个16000个视觉问答训练数据集，支持精细视觉诊断。我们提出了一种两阶段训练策略，首先通过监督微调引导认知轨迹，然后通过强化学习优化决策过程。广泛评估表明，ViTAR优于现有的强大模型。视觉注意分析显示，从“思考”到“再思考”阶段，ViTAR逐渐将视觉 grounding 聚焦于临床关键区域，并在推理过程中保持对视觉令牌的高注意分配，提供了其性能改进的机制见解。这些发现表明，将专家级的迭代思维链嵌入VLMs可以提高医学AI的性能和可信度。 

---
# Explainable Human-in-the-Loop Segmentation via Critic Feedback Signals 

**Title (ZH)**: 可解释的人机环分割通过评论反馈信号 

**Authors**: Pouya Shaeri, Ryan T. Woo, Yasaman Mohammadpour, Ariane Middel  

**Link**: [PDF](https://arxiv.org/pdf/2510.09945)  

**Abstract**: Segmentation models achieve high accuracy on benchmarks but often fail in real-world domains by relying on spurious correlations instead of true object boundaries. We propose a human-in-the-loop interactive framework that enables interventional learning through targeted human corrections of segmentation outputs. Our approach treats human corrections as interventional signals that show when reliance on superficial features (e.g., color or texture) is inappropriate. The system learns from these interventions by propagating correction-informed edits across visually similar images, effectively steering the model toward robust, semantically meaningful features rather than dataset-specific artifacts. Unlike traditional annotation approaches that simply provide more training data, our method explicitly identifies when and why the model fails and then systematically corrects these failure modes across the entire dataset. Through iterative human feedback, the system develops increasingly robust representations that generalize better to novel domains and resist artifactual correlations. We demonstrate that our framework improves segmentation accuracy by up to 9 mIoU points (12-15\% relative improvement) on challenging cubemap data and yields 3-4$\times$ reductions in annotation effort compared to standard retraining, while maintaining competitive performance on benchmark datasets. This work provides a practical framework for researchers and practitioners seeking to build segmentation systems that are accurate, robust to dataset biases, data-efficient, and adaptable to real-world domains such as urban climate monitoring and autonomous driving. 

**Abstract (ZH)**: 基于人类循环交互的干预学习框架：提高分割模型的准确性和鲁棒性 

---
# SpectralCA: Bi-Directional Cross-Attention for Next-Generation UAV Hyperspectral Vision 

**Title (ZH)**: SpectralCA: 双向交叉注意力机制用于下一代无人机 hyperspectral 视觉 

**Authors**: D.V. Brovko  

**Link**: [PDF](https://arxiv.org/pdf/2510.09912)  

**Abstract**: The relevance of this research lies in the growing demand for unmanned aerial vehicles (UAVs) capable of operating reliably in complex environments where conventional navigation becomes unreliable due to interference, poor visibility, or camouflage. Hyperspectral imaging (HSI) provides unique opportunities for UAV-based computer vision by enabling fine-grained material recognition and object differentiation, which are critical for navigation, surveillance, agriculture, and environmental monitoring. The aim of this work is to develop a deep learning architecture integrating HSI into UAV perception for navigation, object detection, and terrain classification. Objectives include: reviewing existing HSI methods, designing a hybrid 2D/3D convolutional architecture with spectral-spatial cross-attention, training, and benchmarking. The methodology is based on the modification of the Mobile 3D Vision Transformer (MDvT) by introducing the proposed SpectralCA block. This block employs bi-directional cross-attention to fuse spectral and spatial features, enhancing accuracy while reducing parameters and inference time. Experimental evaluation was conducted on the WHU-Hi-HongHu dataset, with results assessed using Overall Accuracy, Average Accuracy, and the Kappa coefficient. The findings confirm that the proposed architecture improves UAV perception efficiency, enabling real-time operation for navigation, object recognition, and environmental monitoring tasks.
Keywords: SpectralCA, deep learning, computer vision, hyperspectral imaging, unmanned aerial vehicle, object detection, semi-supervised learning. 

**Abstract (ZH)**: 基于航摄高光谱成像的无人机导航与目标检测深度学习架构研究 

---
# Harnessing Self-Supervised Deep Learning and Geostationary Remote Sensing for Advancing Wildfire and Associated Air Quality Monitoring: Improved Smoke and Fire Front Masking using GOES and TEMPO Radiance Data 

**Title (ZH)**: 利用自我监督深度学习和地球静止轨道遥感促进野火及关联空气质量监测：使用GOES和TEMPO辐射数据优化烟雾和火线掩码 

**Authors**: Nicholas LaHaye, Thilanka Munashinge, Hugo Lee, Xiaohua Pan, Gonzalo Gonzalez Abad, Hazem Mahmoud, Jennifer Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.09845)  

**Abstract**: This work demonstrates the possibilities for improving wildfire and air quality management in the western United States by leveraging the unprecedented hourly data from NASA's TEMPO satellite mission and advances in self-supervised deep learning. Here we demonstrate the efficacy of deep learning for mapping the near real-time hourly spread of wildfire fronts and smoke plumes using an innovative self-supervised deep learning-system: successfully distinguishing smoke plumes from clouds using GOES-18 and TEMPO data, strong agreement across the smoke and fire masks generated from different sensing modalities as well as significant improvement over operational products for the same cases. 

**Abstract (ZH)**: 本研究展示了通过利用NASA TEMPO卫星任务前所未有的每小时数据以及自主监督深度学习的进步，提高美国西部地区野火和空气质量管理的可能性。我们证明了深度学习在使用创新的自主监督深度学习系统实时绘制野火前沿和烟霾分布图方面的有效性：成功使用GOES-18和TEMPO数据区分烟霾和云，不同传感模态生成的烟霾和火mask之间具有较强的协议，并且在相同情况下显著优于现有产品。 

---
# Vanishing Contributions: A Unified Approach to Smoothly Transition Neural Models into Compressed Form 

**Title (ZH)**: 消失的贡献：平滑过渡神经模型至压缩形式的一体化方法 

**Authors**: Lorenzo Nikiforos, Charalampos Antoniadis, Luciano Prono, Fabio Pareschi, Riccardo Rovatti, Gianluca Setti  

**Link**: [PDF](https://arxiv.org/pdf/2510.09696)  

**Abstract**: The increasing scale of deep neural networks has led to a growing need for compression techniques such as pruning, quantization, and low-rank decomposition. While these methods are very effective in reducing memory, computation and energy consumption, they often introduce severe accuracy degradation when applied directly. We introduce Vanishing Contributions (VCON), a general approach for smoothly transitioning neural models into compressed form. Rather than replacing the original network directly with its compressed version, VCON executes the two in parallel during fine-tuning. The contribution of the original (uncompressed) model is progressively reduced, while that of the compressed model is gradually increased. This smooth transition allows the network to adapt over time, improving stability and mitigating accuracy degradation. We evaluate VCON across computer vision and natural language processing benchmarks, in combination with multiple compression strategies. Across all scenarios, VCON leads to consistent improvements: typical gains exceed 3%, while some configuration exhibits accuracy boosts of 20%. VCON thus provides a generalizable method that can be applied to the existing compression techniques, with evidence of consistent gains across multiple benchmarks. 

**Abstract (ZH)**: 深度神经网络规模的增加导致了压缩技术（如剪枝、量化和低秩分解）需求的增长。虽然这些方法在减少内存、计算和能耗方面非常有效，但在直接应用时通常会导致严重的准确率下降。我们引入了消失贡献（VCON）方法，这是一种使神经网络平滑过渡到压缩形式的通用方法。VCON不直接用压缩网络替换原始网络，而是在微调过程中并行执行两者。原始（未压缩）模型的贡献逐渐减少，而压缩模型的贡献逐渐增加。这种平滑过渡使网络能够随着时间调整，提高稳定性和减轻准确率下降。我们在计算机视觉和自然语言处理基准上评估了VCON，结合了多种压缩策略。在所有场景中，VCON都带来了持续改进：典型增益超过3%，而某些配置的准确率提升高达20%。因此，VCON提供了一种可应用于现有压缩技术的通用方法，并且在多个基准测试中表现出一致的增益。 

---
# Learning What Matters: Steering Diffusion via Spectrally Anisotropic Forward Noise 

**Title (ZH)**: 学习重要性：通过谱各向异性前向噪声引导扩散 

**Authors**: Luca Scimeca, Thomas Jiralerspong, Berton Earnshaw, Jason Hartford, Yoshua Bengio  

**Link**: [PDF](https://arxiv.org/pdf/2510.09660)  

**Abstract**: Diffusion Probabilistic Models (DPMs) have achieved strong generative performance, yet their inductive biases remain largely implicit. In this work, we aim to build inductive biases into the training and sampling of diffusion models to better accommodate the target distribution of the data to model. We introduce an anisotropic noise operator that shapes these biases by replacing the isotropic forward covariance with a structured, frequency-diagonal covariance. This operator unifies band-pass masks and power-law weightings, allowing us to emphasize or suppress designated frequency bands, while keeping the forward process Gaussian. We refer to this as spectrally anisotropic Gaussian diffusion (SAGD). In this work, we derive the score relation for anisotropic covariances and show that, under full support, the learned score converges to the true data score as $t\!\to\!0$, while anisotropy reshapes the probability-flow path from noise to data. Empirically, we show the induced anisotropy outperforms standard diffusion across several vision datasets, and enables selective omission: learning while ignoring known corruptions confined to specific bands. Together, these results demonstrate that carefully designed anisotropic forward noise provides a simple, yet principled, handle to tailor inductive bias in DPMs. 

**Abstract (ZH)**: 谱各向异性高斯扩散（SAGD）：在扩散模型中构建诱导偏置 

---
# Ultralytics YOLO Evolution: An Overview of YOLO26, YOLO11, YOLOv8 and YOLOv5 Object Detectors for Computer Vision and Pattern Recognition 

**Title (ZH)**: YOLO演化超纲：YOLO26、YOLO11、YOLOv8和YOLOv5物体检测器综述 

**Authors**: Ranjan Sapkota, Manoj Karkee  

**Link**: [PDF](https://arxiv.org/pdf/2510.09653)  

**Abstract**: This paper presents a comprehensive overview of the Ultralytics YOLO(You Only Look Once) family of object detectors, focusing the architectural evolution, benchmarking, deployment perspectives, and future challenges. The review begins with the most recent release, YOLO26 (YOLOv26), which introduces key innovations including Distribution Focal Loss (DFL) removal, native NMS-free inference, Progressive Loss Balancing (ProgLoss), Small-Target-Aware Label Assignment (STAL), and the MuSGD optimizer for stable training. The progression is then traced through YOLO11, with its hybrid task assignment and efficiency-focused modules; YOLOv8, which advanced with a decoupled detection head and anchor-free predictions; and YOLOv5, which established the modular PyTorch foundation that enabled modern YOLO development. Benchmarking on the MS COCO dataset provides a detailed quantitative comparison of YOLOv5, YOLOv8, YOLO11, and YOLO26, alongside cross-comparisons with YOLOv12, YOLOv13, RT-DETR, and DEIM. Metrics including precision, recall, F1 score, mean Average Precision, and inference speed are analyzed to highlight trade-offs between accuracy and efficiency. Deployment and application perspectives are further discussed, covering export formats, quantization strategies, and real-world use in robotics, agriculture, surveillance, and manufacturing. Finally, the paper identifies challenges and future directions, including dense-scene limitations, hybrid CNN-Transformer integration, open-vocabulary detection, and edge-aware training approaches. 

**Abstract (ZH)**: 本论文对Ultralytics YOLO(你只看一次)家族的目标检测器进行了全面综述，重点介绍了架构演化、基准测试、部署视角以及未来挑战。综述从最新的YOLO26（YOLOv26）开始，该版本引入了关键创新，包括分布焦损（DFL）移除、原生无NMS推断、渐进损失平衡（ProgLoss）、小目标感知标签分配（STAL）以及MuSGD优化器以实现稳定的训练。随后追溯了从YOLO11及其混合任务分配和高效模块，到YOLOv8的解耦检测头和无锚预测，再到YOLOv5的模块化PyTorch基础，这些基础推动了现代YOLO的发展。在MS COCO数据集上的基准测试提供了YOLOv5、YOLOv8、YOLO11和YOLO26的详细定量比较，同时还与YOLOv12、YOLOv13、RT-DETR和DEIM进行了交叉比较。分析包括精度、召回率、F1分数、平均精确度和推理速度等指标，以强调准确性和效率之间的权衡。进一步讨论了部署和应用视角，包括导出格式、量化策略以及在机器人技术、农业、监控和制造等实际应用中的使用。最后，论文指出了挑战和未来方向，包括密集场景限制、CNN-Transformer结合、开放词汇检测以及边缘感知训练方法。 

---
# TinyViT-Batten: Few-Shot Vision Transformer with Explainable Attention for Early Batten-Disease Detection on Pediatric MRI 

**Title (ZH)**: TinyViT-Batten：具有可解释注意力的少量样本视觉变压器在儿童MRI早期巴特病检测中的应用 

**Authors**: Khartik Uppalapati, Bora Yimenicioglu, Shakeel Abdulkareem, Adan Eftekhari, Bhavya Uppalapati, Viraj Kamath  

**Link**: [PDF](https://arxiv.org/pdf/2510.09649)  

**Abstract**: Batten disease (neuronal ceroid lipofuscinosis) is a rare pediatric neurodegenerative disorder whose early MRI signs are subtle and often missed. We propose TinyViT-Batten, a few-shot Vision Transformer (ViT) framework to detect early Batten disease from pediatric brain MRI with limited training cases. We distill a large teacher ViT into a 5 M-parameter TinyViT and fine-tune it using metric-based few-shot learning (prototypical loss with 5-shot episodes). Our model achieves high accuracy (approximately 91%) and area under ROC of at least 0.95 on a multi-site dataset of 79 genetically confirmed Batten-disease MRIs (27 CLN3 from the Hochstein natural-history study, 32 CLN2 from an international longitudinal cohort, 12 early-manifestation CLN2 cases reported by Cokal et al., and 8 public Radiopaedia scans) together with 90 age-matched controls, outperforming a 3D-ResNet and Swin-Tiny baseline. We further integrate Gradient-weighted Class Activation Mapping (Grad-CAM) to highlight disease-relevant brain regions, enabling explainable predictions. The model's small size and strong performance (sensitivity greater than 90%, specificity approximately 90%) demonstrates a practical AI solution for early Batten disease detection. 

**Abstract (ZH)**: 基于少量样本学习的TinyViT-Batten框架在有限训练案例下从儿童脑MRI中检测早期Batten病 

---
