# DeepVL: Dynamics and Inertial Measurements-based Deep Velocity Learning for Underwater Odometry 

**Title (ZH)**: DeepVL：基于动力学和惯性测量的深水里程计深度速度学习 

**Authors**: Mohit Singh, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2502.07726)  

**Abstract**: This paper presents a learned model to predict the robot-centric velocity of an underwater robot through dynamics-aware proprioception. The method exploits a recurrent neural network using as inputs inertial cues, motor commands, and battery voltage readings alongside the hidden state of the previous time-step to output robust velocity estimates and their associated uncertainty. An ensemble of networks is utilized to enhance the velocity and uncertainty predictions. Fusing the network's outputs into an Extended Kalman Filter, alongside inertial predictions and barometer updates, the method enables long-term underwater odometry without further exteroception. Furthermore, when integrated into visual-inertial odometry, the method assists in enhanced estimation resilience when dealing with an order of magnitude fewer total features tracked (as few as 1) as compared to conventional visual-inertial systems. Tested onboard an underwater robot deployed both in a laboratory pool and the Trondheim Fjord, the method takes less than 5ms for inference either on the CPU or the GPU of an NVIDIA Orin AGX and demonstrates less than 4% relative position error in novel trajectories during complete visual blackout, and approximately 2% relative error when a maximum of 2 visual features from a monocular camera are available. 

**Abstract (ZH)**: 一种基于动力学aware proprioception的_learned模型用于海底机器人机器人-centric速度预测 

---
# Next Block Prediction: Video Generation via Semi-Auto-Regressive Modeling 

**Title (ZH)**: 下一个块的预测：通过半自动回归建模进行视频生成 

**Authors**: Shuhuai Ren, Shuming Ma, Xu Sun, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.07737)  

**Abstract**: Next-Token Prediction (NTP) is a de facto approach for autoregressive (AR) video generation, but it suffers from suboptimal unidirectional dependencies and slow inference speed. In this work, we propose a semi-autoregressive (semi-AR) framework, called Next-Block Prediction (NBP), for video generation. By uniformly decomposing video content into equal-sized blocks (e.g., rows or frames), we shift the generation unit from individual tokens to blocks, allowing each token in the current block to simultaneously predict the corresponding token in the next block. Unlike traditional AR modeling, our framework employs bidirectional attention within each block, enabling tokens to capture more robust spatial dependencies. By predicting multiple tokens in parallel, NBP models significantly reduce the number of generation steps, leading to faster and more efficient inference. Our model achieves FVD scores of 103.3 on UCF101 and 25.5 on K600, outperforming the vanilla NTP model by an average of 4.4. Furthermore, thanks to the reduced number of inference steps, the NBP model generates 8.89 frames (128x128 resolution) per second, achieving an 11x speedup. We also explored model scales ranging from 700M to 3B parameters, observing significant improvements in generation quality, with FVD scores dropping from 103.3 to 55.3 on UCF101 and from 25.5 to 19.5 on K600, demonstrating the scalability of our approach. 

**Abstract (ZH)**: Next-Block Prediction (NBP)：一种用于视频生成的半自回归框架 

---
# Exoplanet Transit Candidate Identification in TESS Full-Frame Images via a Transformer-Based Algorithm 

**Title (ZH)**: 基于变压器算法的TESS全帧图像外行星凌星候选体识别 

**Authors**: Helem Salinas, Rafael Brahm, Greg Olmschenk, Richard K. Barry, Karim Pichara, Stela Ishitani Silva, Vladimir Araujo  

**Link**: [PDF](https://arxiv.org/pdf/2502.07542)  

**Abstract**: The Transiting Exoplanet Survey Satellite (TESS) is surveying a large fraction of the sky, generating a vast database of photometric time series data that requires thorough analysis to identify exoplanetary transit signals. Automated learning approaches have been successfully applied to identify transit signals. However, most existing methods focus on the classification and validation of candidates, while few efforts have explored new techniques for the search of candidates. To search for new exoplanet transit candidates, we propose an approach to identify exoplanet transit signals without the need for phase folding or assuming periodicity in the transit signals, such as those observed in multi-transit light curves. To achieve this, we implement a new neural network inspired by Transformers to directly process Full Frame Image (FFI) light curves to detect exoplanet transits. Transformers, originally developed for natural language processing, have recently demonstrated significant success in capturing long-range dependencies compared to previous approaches focused on sequential data. This ability allows us to employ multi-head self-attention to identify exoplanet transit signals directly from the complete light curves, combined with background and centroid time series, without requiring prior transit parameters. The network is trained to learn characteristics of the transit signal, like the dip shape, which helps distinguish planetary transits from other variability sources. Our model successfully identified 214 new planetary system candidates, including 122 multi-transit light curves, 88 single-transit and 4 multi-planet systems from TESS sectors 1-26 with a radius > 0.27 $R_{\mathrm{Jupiter}}$, demonstrating its ability to detect transits regardless of their periodicity. 

**Abstract (ZH)**: TESS中基于Transformer的新兴行星凌星信号识别方法 

---
# VidCRAFT3: Camera, Object, and Lighting Control for Image-to-Video Generation 

**Title (ZH)**: VidCRAFT3: 基于图像到视频生成的摄像头、对象和 lighting 控制 

**Authors**: Sixiao Zheng, Zimian Peng, Yanpeng Zhou, Yi Zhu, Hang Xu, Xiangru Huang, Yanwei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2502.07531)  

**Abstract**: Recent image-to-video generation methods have demonstrated success in enabling control over one or two visual elements, such as camera trajectory or object motion. However, these methods are unable to offer control over multiple visual elements due to limitations in data and network efficacy. In this paper, we introduce VidCRAFT3, a novel framework for precise image-to-video generation that enables control over camera motion, object motion, and lighting direction simultaneously. To better decouple control over each visual element, we propose the Spatial Triple-Attention Transformer, which integrates lighting direction, text, and image in a symmetric way. Since most real-world video datasets lack lighting annotations, we construct a high-quality synthetic video dataset, the VideoLightingDirection (VLD) dataset. This dataset includes lighting direction annotations and objects of diverse appearance, enabling VidCRAFT3 to effectively handle strong light transmission and reflection effects. Additionally, we propose a three-stage training strategy that eliminates the need for training data annotated with multiple visual elements (camera motion, object motion, and lighting direction) simultaneously. Extensive experiments on benchmark datasets demonstrate the efficacy of VidCRAFT3 in producing high-quality video content, surpassing existing state-of-the-art methods in terms of control granularity and visual coherence. All code and data will be publicly available. Project page: this https URL. 

**Abstract (ZH)**: Recent image-to-video generation方法在控制一个或两个视觉元素（如相机轨迹或对象运动）方面取得了成功，但在控制多个视觉元素方面受到数据和网络效果的限制。本文介绍了VidCRAFT3，一种新的精确图像到视频生成框架，能够同时控制相机运动、对象运动和照明方向。为了更好地分离对每个视觉元素的控制，我们提出了空间三重注意变换器，它以对称方式结合了照明方向、文本和图像。由于大多数真实世界的视频数据集缺乏照明标注，我们构建了一个高质量的合成视频数据集——视频照明方向（VLD）数据集。该数据集包含了照明方向标注和多样外观的对象，使VidCRAFT3能够有效地处理强烈的透射和反射效果。此外，我们提出了一种三阶段训练策略，可以消除同时标注有多个视觉元素（相机运动、对象运动和照明方向）的需求。基准数据集上的广泛实验表明，VidCRAFT3在生成高质量视频内容方面优于现有最先进的方法，在控制细度和视觉连贯性方面表现出色。所有代码和数据将公开提供。项目页面：this https URL。 

---
# No Data, No Optimization: A Lightweight Method To Disrupt Neural Networks With Sign-Flips 

**Title (ZH)**: 没有数据，没有优化：一种基于符号翻转的轻量级神经网络干扰方法 

**Authors**: Ido Galil, Moshe Kimhi, Ran El-Yaniv  

**Link**: [PDF](https://arxiv.org/pdf/2502.07408)  

**Abstract**: Deep Neural Networks (DNNs) can be catastrophically disrupted by flipping only a handful of sign bits in their parameters. We introduce Deep Neural Lesion (DNL), a data-free, lightweight method that locates these critical parameters and triggers massive accuracy drops. We validate its efficacy on a wide variety of computer vision models and datasets. The method requires no training data or optimization and can be carried out via common exploits software, firmware or hardware based attack vectors. An enhanced variant that uses a single forward and backward pass further amplifies the damage beyond DNL's zero-pass approach. Flipping just two sign bits in ResNet50 on ImageNet reduces accuracy by 99.8\%. We also show that selectively protecting a small fraction of vulnerable sign bits provides a practical defense against such attacks. 

**Abstract (ZH)**: 深度神经网络（DNNs）的少量权重符号位翻转可以导致灾难性的破坏。我们介绍了一种无数据、轻量级的方法Deep Neural Lesion（DNL），该方法可以定位这些关键参数并触发巨大的准确率下降。我们在多种计算机视觉模型和数据集上验证了其有效性。该方法无需训练数据或优化，可以通过常见的漏洞利用软件、固件或硬件攻击向量来执行。一个增强版本仅使用一次前向和反向传播进一步放大了DNL零次方法的破坏性。在ImageNet上翻转ResNet50的两个符号位可使准确率降低99.8%。我们还展示了有选择地保护一小部分脆弱的符号位可以提供对抗此类攻击的实用防御措施。 

---
# Multi-Task-oriented Nighttime Haze Imaging Enhancer for Vision-driven Measurement Systems 

**Title (ZH)**: 面向多任务的夜间雾霾成像增强器用于视觉驱动的测量系统 

**Authors**: Ai Chen, Yuxu Lu, Dong Yang, Junlin Zhou, Yan Fu, Duanbing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.07351)  

**Abstract**: Salient object detection (SOD) plays a critical role in vision-driven measurement systems (VMS), facilitating the detection and segmentation of key visual elements in an image. However, adverse imaging conditions such as haze during the day, low light, and haze at night severely degrade image quality, and complicating the SOD process. To address these challenges, we propose a multi-task-oriented nighttime haze imaging enhancer (MToIE), which integrates three tasks: daytime dehazing, low-light enhancement, and nighttime dehazing. The MToIE incorporates two key innovative components: First, the network employs a task-oriented node learning mechanism to handle three specific degradation types: day-time haze, low light, and night-time haze conditions, with an embedded self-attention module enhancing its performance in nighttime imaging. In addition, multi-receptive field enhancement module that efficiently extracts multi-scale features through three parallel depthwise separable convolution branches with different dilation rates, capturing comprehensive spatial information with minimal computational overhead. To ensure optimal image reconstruction quality and visual characteristics, we suggest a hybrid loss function. Extensive experiments on different types of weather/imaging conditions illustrate that MToIE surpasses existing methods, significantly enhancing the accuracy and reliability of vision systems across diverse imaging scenarios. The code is available at this https URL. 

**Abstract (ZH)**: 多任务导向的夜间 haze 特征增强器（MToIE）：白天去 haz、低光照增强和夜间去 haz 的集成解决方案 

---
# TRAVEL: Training-Free Retrieval and Alignment for Vision-and-Language Navigation 

**Title (ZH)**: 旅行：无需训练的视觉-语言导航检索与对齐 

**Authors**: Navid Rajabi, Jana Kosecka  

**Link**: [PDF](https://arxiv.org/pdf/2502.07306)  

**Abstract**: In this work, we propose a modular approach for the Vision-Language Navigation (VLN) task by decomposing the problem into four sub-modules that use state-of-the-art Large Language Models (LLMs) and Vision-Language Models (VLMs) in a zero-shot setting. Given navigation instruction in natural language, we first prompt LLM to extract the landmarks and the order in which they are visited. Assuming the known model of the environment, we retrieve the top-k locations of the last landmark and generate $k$ path hypotheses from the starting location to the last landmark using the shortest path algorithm on the topological map of the environment. Each path hypothesis is represented by a sequence of panoramas. We then use dynamic programming to compute the alignment score between the sequence of panoramas and the sequence of landmark names, which match scores obtained from VLM. Finally, we compute the nDTW metric between the hypothesis that yields the highest alignment score to evaluate the path fidelity. We demonstrate superior performance compared to other approaches that use joint semantic maps like VLMaps \cite{vlmaps} on the complex R2R-Habitat \cite{r2r} instruction dataset and quantify in detail the effect of visual grounding on navigation performance. 

**Abstract (ZH)**: 本研究提出了一种模块化的方法来解决视觉语言导航（VLN）任务，通过将问题分解为四个子模块，这些子模块在零样本设置中使用最先进的大型语言模型（LLMs）和视觉语言模型（VLMs）。给定自然语言的导航指令，我们首先使用LLM提取地标及其访问顺序。在已知环境模型的情况下，我们检索最后一个地标附近的前k个位置，并使用环境拓扑图上的最短路径算法从起始位置生成到最后一个地标之间的k条路径假设。每条路径假设由全景图序列表示。然后，我们使用动态规划计算全景图序列与地标名称序列之间的对齐得分，该得分与VLM获得的匹配得分进行比较。最后，我们计算生成最高对齐得分假设的nDTW度量来评估路径准确性。我们在复杂的R2R-Habitat指令数据集上展示了优于使用联合语义图（如VLMaps）的其他方法的性能，并详细量化了视觉定位对导航性能的影响。 

---
# KPIs 2024 Challenge: Advancing Glomerular Segmentation from Patch- to Slide-Level 

**Title (ZH)**: KPIs 2024 挑战：从Patch级到Slide级的肾小球分割 advancements 

**Authors**: Ruining Deng, Tianyuan Yao, Yucheng Tang, Junlin Guo, Siqi Lu, Juming Xiong, Lining Yu, Quan Huu Cap, Pengzhou Cai, Libin Lan, Ze Zhao, Adrian Galdran, Amit Kumar, Gunjan Deotale, Dev Kumar Das, Inyoung Paik, Joonho Lee, Geongyu Lee, Yujia Chen, Wangkai Li, Zhaoyang Li, Xuege Hou, Zeyuan Wu, Shengjin Wang, Maximilian Fischer, Lars Kramer, Anghong Du, Le Zhang, Maria Sanchez Sanchez, Helena Sanchez Ulloa, David Ribalta Heredia, Carlos Perez de Arenaza Garcia, Shuoyu Xu, Bingdou He, Xinping Cheng, Tao Wang, Noemie Moreau, Katarzyna Bozek, Shubham Innani, Ujjwal Baid, Kaura Solomon Kefas, Bennett A. Landman, Yu Wang, Shilin Zhao, Mengmeng Yin, Haichun Yang, Yuankai Huo  

**Link**: [PDF](https://arxiv.org/pdf/2502.07288)  

**Abstract**: Chronic kidney disease (CKD) is a major global health issue, affecting over 10% of the population and causing significant mortality. While kidney biopsy remains the gold standard for CKD diagnosis and treatment, the lack of comprehensive benchmarks for kidney pathology segmentation hinders progress in the field. To address this, we organized the Kidney Pathology Image Segmentation (KPIs) Challenge, introducing a dataset that incorporates preclinical rodent models of CKD with over 10,000 annotated glomeruli from 60+ Periodic Acid Schiff (PAS)-stained whole slide images. The challenge includes two tasks, patch-level segmentation and whole slide image segmentation and detection, evaluated using the Dice Similarity Coefficient (DSC) and F1-score. By encouraging innovative segmentation methods that adapt to diverse CKD models and tissue conditions, the KPIs Challenge aims to advance kidney pathology analysis, establish new benchmarks, and enable precise, large-scale quantification for disease research and diagnosis. 

**Abstract (ZH)**: 慢性肾病（CKD）是全球重要的健康问题，影响超过10%的人口并导致显著的死亡率。尽管肾活检仍然是CKD诊断和治疗的金标准，但缺乏全面的肾病理分割基准阻碍了该领域的发展。为了解决这一问题，我们组织了肾病理图像分割（KPIs）挑战，并引入了一个数据集，该数据集结合了预临床CKD小鼠模型，并包含来自60多例 periodic acid Schiff（PAS）染色全切片图像的超过10,000个标注的肾小体。该挑战包含两个任务——patch-level分割和全切片图像分割与检测，评估指标为Dice相似性系数（DSC）和F1分数。通过鼓励适应多种CKD模型和组织条件的创新分割方法，KPIs挑战旨在推进肾病理分析、建立新基准，并实现疾病研究和诊断中的精确、大规模量化。 

---
# Enhancing Video Understanding: Deep Neural Networks for Spatiotemporal Analysis 

**Title (ZH)**: 增强视频理解：用于时空分析的深度神经网络 

**Authors**: Amir Hosein Fadaei, Mohammad-Reza A. Dehaqani  

**Link**: [PDF](https://arxiv.org/pdf/2502.07277)  

**Abstract**: It's no secret that video has become the primary way we share information online. That's why there's been a surge in demand for algorithms that can analyze and understand video content. It's a trend going to continue as video continues to dominate the digital landscape. These algorithms will extract and classify related features from the video and will use them to describe the events and objects in the video. Deep neural networks have displayed encouraging outcomes in the realm of feature extraction and video description. This paper will explore the spatiotemporal features found in videos and recent advancements in deep neural networks in video understanding. We will review some of the main trends in video understanding models and their structural design, the main problems, and some offered solutions in this topic. We will also review and compare significant video understanding and action recognition datasets. 

**Abstract (ZH)**: 视频已成为我们在线分享信息的主要方式：视频内容分析与理解的最新进展 

---
# Vevo: Controllable Zero-Shot Voice Imitation with Self-Supervised Disentanglement 

**Title (ZH)**: Vevo:可控的自监督解耦零样本语音模仿 

**Authors**: Xueyao Zhang, Xiaohui Zhang, Kainan Peng, Zhenyu Tang, Vimal Manohar, Yingru Liu, Jeff Hwang, Dangna Li, Yuhao Wang, Julian Chan, Yuan Huang, Zhizheng Wu, Mingbo Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.07243)  

**Abstract**: The imitation of voice, targeted on specific speech attributes such as timbre and speaking style, is crucial in speech generation. However, existing methods rely heavily on annotated data, and struggle with effectively disentangling timbre and style, leading to challenges in achieving controllable generation, especially in zero-shot scenarios. To address these issues, we propose Vevo, a versatile zero-shot voice imitation framework with controllable timbre and style. Vevo operates in two core stages: (1) Content-Style Modeling: Given either text or speech's content tokens as input, we utilize an autoregressive transformer to generate the content-style tokens, which is prompted by a style reference; (2) Acoustic Modeling: Given the content-style tokens as input, we employ a flow-matching transformer to produce acoustic representations, which is prompted by a timbre reference. To obtain the content and content-style tokens of speech, we design a fully self-supervised approach that progressively decouples the timbre, style, and linguistic content of speech. Specifically, we adopt VQ-VAE as the tokenizer for the continuous hidden features of HuBERT. We treat the vocabulary size of the VQ-VAE codebook as the information bottleneck, and adjust it carefully to obtain the disentangled speech representations. Solely self-supervised trained on 60K hours of audiobook speech data, without any fine-tuning on style-specific corpora, Vevo matches or surpasses existing methods in accent and emotion conversion tasks. Additionally, Vevo's effectiveness in zero-shot voice conversion and text-to-speech tasks further demonstrates its strong generalization and versatility. Audio samples are available at this https URL. 

**Abstract (ZH)**: 目标语音特定位数模仿：可控音色和风格的通用零样本语音生成框架 

---
# Contextual Gesture: Co-Speech Gesture Video Generation through Context-aware Gesture Representation 

**Title (ZH)**: 上下文手势：基于上下文感知手势表示的共时手势视频生成 

**Authors**: Pinxin Liu, Pengfei Zhang, Hyeongwoo Kim, Pablo Garrido, Ari Sharpio, Kyle Olszewski  

**Link**: [PDF](https://arxiv.org/pdf/2502.07239)  

**Abstract**: Co-speech gesture generation is crucial for creating lifelike avatars and enhancing human-computer interactions by synchronizing gestures with speech. Despite recent advancements, existing methods struggle with accurately identifying the rhythmic or semantic triggers from audio for generating contextualized gesture patterns and achieving pixel-level realism. To address these challenges, we introduce Contextual Gesture, a framework that improves co-speech gesture video generation through three innovative components: (1) a chronological speech-gesture alignment that temporally connects two modalities, (2) a contextualized gesture tokenization that incorporate speech context into motion pattern representation through distillation, and (3) a structure-aware refinement module that employs edge connection to link gesture keypoints to improve video generation. Our extensive experiments demonstrate that Contextual Gesture not only produces realistic and speech-aligned gesture videos but also supports long-sequence generation and video gesture editing applications, shown in Fig.1 Project Page: this https URL. 

**Abstract (ZH)**: 同步言语手势生成对于创建生动的虚拟角色并增强人机交互至关重要，通过将手势与言语同步。尽管近期取得了进展，现有方法在从音频中准确识别节奏或语义触发点以生成情境化手势模式并实现像素级真实性方面仍存在挑战。为解决这些挑战，我们提出了Contextual Gesture框架，该框架通过三种创新组件改进了同步言语手势视频生成：（1）时间上连接言语和手势的对齐，（2）包含语音上下文的上下文化手势标记化，通过精炼将语音上下文融入到运动模式表示中，（3）结构感知精炼模块，通过边缘连接将手势关键点相连以改善视频生成。广泛的实验表明，Contextual Gesture不仅生成了真实且与言语对齐的手势视频，还支持长序列生成和视频手势编辑应用，如图1所示。项目页面：点击此处。 

---
# Diffusion Suction Grasping with Large-Scale Parcel Dataset 

**Title (ZH)**: 大规模包裹数据集驱动的扩散吸取抓取 

**Authors**: Ding-Tao Huang, Xinyi He, Debei Hua, Dongfang Yu, En-Te Lin, Long Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2502.07238)  

**Abstract**: While recent advances in object suction grasping have shown remarkable progress, significant challenges persist particularly in cluttered and complex parcel handling scenarios. Two fundamental limitations hinder current approaches: (1) the lack of a comprehensive suction grasp dataset tailored for parcel manipulation tasks, and (2) insufficient adaptability to diverse object characteristics including size variations, geometric complexity, and textural diversity. To address these challenges, we present Parcel-Suction-Dataset, a large-scale synthetic dataset containing 25 thousand cluttered scenes with 410 million precision-annotated suction grasp poses. This dataset is generated through our novel geometric sampling algorithm that enables efficient generation of optimal suction grasps incorporating both physical constraints and material properties. We further propose Diffusion-Suction, an innovative framework that reformulates suction grasp prediction as a conditional generation task through denoising diffusion probabilistic models. Our method iteratively refines random noise into suction grasp score maps through visual-conditioned guidance from point cloud observations, effectively learning spatial point-wise affordances from our synthetic dataset. Extensive experiments demonstrate that the simple yet efficient Diffusion-Suction achieves new state-of-the-art performance compared to previous models on both Parcel-Suction-Dataset and the public SuctionNet-1Billion benchmark. 

**Abstract (ZH)**: 面向包裹操作的吸盘抓取数据集与Diffusion-Suction框架 

---
# SparseFormer: Detecting Objects in HRW Shots via Sparse Vision Transformer 

**Title (ZH)**: SparseFormer：通过稀疏视觉Transformer在HRW镜头中检测物体 

**Authors**: Wenxi Li, Yuchen Guo, Jilai Zheng, Haozhe Lin, Chao Ma, Lu Fang, Xiaokang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.07216)  

**Abstract**: Recent years have seen an increase in the use of gigapixel-level image and video capture systems and benchmarks with high-resolution wide (HRW) shots. However, unlike close-up shots in the MS COCO dataset, the higher resolution and wider field of view raise unique challenges, such as extreme sparsity and huge scale changes, causing existing close-up detectors inaccuracy and inefficiency. In this paper, we present a novel model-agnostic sparse vision transformer, dubbed SparseFormer, to bridge the gap of object detection between close-up and HRW shots. The proposed SparseFormer selectively uses attentive tokens to scrutinize the sparsely distributed windows that may contain objects. In this way, it can jointly explore global and local attention by fusing coarse- and fine-grained features to handle huge scale changes. SparseFormer also benefits from a novel Cross-slice non-maximum suppression (C-NMS) algorithm to precisely localize objects from noisy windows and a simple yet effective multi-scale strategy to improve accuracy. Extensive experiments on two HRW benchmarks, PANDA and DOTA-v1.0, demonstrate that the proposed SparseFormer significantly improves detection accuracy (up to 5.8%) and speed (up to 3x) over the state-of-the-art approaches. 

**Abstract (ZH)**: 近年来， gigapixel 级图像和视频捕获系统和高分辨率宽场景（HRW）基准的应用日益增多。然而，与 MS COCO 数据集中的近距拍摄相比，更高的分辨率和更宽的视野带来了独特的挑战，如极端稀疏性和巨大的尺度变化，导致现有的近距检测器出现不准确和低效的问题。本文提出了一种新型的模型无关稀疏视觉变换器，命名为 SparseFormer，以弥合近距拍摄与 HRW 拍摄之间的检测差距。SparseFormer 选择性地使用注意力标记来仔细审查可能包含目标的稀疏分布窗口，从而通过融合粗粒度和细粒度特征来共同探索全局和局部注意力，以处理巨大的尺度变化。SparseFormer 还受益于一种新颖的跨层非最大抑制（C-NMS）算法，可以精确地从嘈杂的窗口中定位目标，并采用一种简单而有效的多尺度策略来提高准确率。在两个 HRW 基准 PANDA 和 DOTA-v1.0 上的广泛实验表明，提出的 SparseFormer 相较于现有最先进的方法显著提高了检测准确率（高达 5.8%）和速度（高达 3 倍）。 

---
# Dense Object Detection Based on De-homogenized Queries 

**Title (ZH)**: 基于去同质化查询的密集目标检测 

**Authors**: Yueming Huang, Chenrui Ma, Hao Zhou, Hao Wu, Guowu Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.07194)  

**Abstract**: Dense object detection is widely used in automatic driving, video surveillance, and other fields. This paper focuses on the challenging task of dense object detection. Currently, detection methods based on greedy algorithms, such as non-maximum suppression (NMS), often produce many repetitive predictions or missed detections in dense scenarios, which is a common problem faced by NMS-based algorithms. Through the end-to-end DETR (DEtection TRansformer), as a type of detector that can incorporate the post-processing de-duplication capability of NMS, etc., into the network, we found that homogeneous queries in the query-based detector lead to a reduction in the de-duplication capability of the network and the learning efficiency of the encoder, resulting in duplicate prediction and missed detection problems. To solve this problem, we propose learnable differentiated encoding to de-homogenize the queries, and at the same time, queries can communicate with each other via differentiated encoding information, replacing the previous self-attention among the queries. In addition, we used joint loss on the output of the encoder that considered both location and confidence prediction to give a higher-quality initialization for queries. Without cumbersome decoder stacking and guaranteeing accuracy, our proposed end-to-end detection framework was more concise and reduced the number of parameters by about 8% compared to deformable DETR. Our method achieved excellent results on the challenging CrowdHuman dataset with 93.6% average precision (AP), 39.2% MR-2, and 84.3% JI. The performance overperformed previous SOTA methods, such as Iter-E2EDet (Progressive End-to-End Object Detection) and MIP (One proposal, Multiple predictions). In addition, our method is more robust in various scenarios with different densities. 

**Abstract (ZH)**: 密集目标检测在自动驾驶、视频监控等领域中有广泛应用。本文专注于密集目标检测这一具有挑战性的任务。目前基于贪婪算法的检测方法，如非极大值抑制（NMS），在密集场景中经常产生许多重复预测或漏检，这是NMS基算法常见的问题。通过端到端的DETR（检测变换器），作为一种可以将NMS等后处理去重能力融入网络的检测器，我们发现基于查询的检测器中的同质查询降低了网络的去重能力和编码器的学习效率，导致重复预测和漏检问题。为了解决这个问题，我们提出可学习的差异化编码以去同质化查询，并且通过差异化编码信息使查询之间可以相互通信，取代了之前的查询自我注意力机制。此外，我们在编码器的输出上使用联合损失，同时考虑位置和置信度预测，为查询提供更高的初始化质量。在不增加复杂解码器堆叠且保证准确性的情况下，我们的端到端检测框架更为简洁，并且参数数量减少了约8%相比可变形DETR。我们的方法在具有挑战性的CrowdHuman数据集上取得了优异的结果，平均精度（AP）为93.6%，MR-2为39.2%，JI为84.3%，性能超越了之前的SOTA方法，如Iter-E2EDet（渐进式端到端目标检测）和MIP（一提议多预测）。此外，我们的方法在不同密度的多种场景中更加稳健。 

---
# Improved YOLOv7 model for insulator defect detection 

**Title (ZH)**: 改进的YOLOv7模型在绝缘子缺陷检测中的应用 

**Authors**: Zhenyue Wang, Guowu Yuan, Hao Zhou, Yi Ma, Yutang Ma, Dong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.07179)  

**Abstract**: Insulators are crucial insulation components and structural supports in power grids, playing a vital role in the transmission lines. Due to temperature fluctuations, internal stress, or damage from hail, insulators are prone to injury. Automatic detection of damaged insulators faces challenges such as diverse types, small defect targets, and complex backgrounds and shapes. Most research for detecting insulator defects has focused on a single defect type or a specific material. However, the insulators in the grid's transmission lines have different colors and materials. Various insulator defects coexist, and the existing methods have difficulty meeting the practical application requirements. Current methods suffer from low detection accuracy and mAP0.5 cannot meet application requirements. This paper proposes an improved YOLOv7 model for multi-type insulator defect detection. First, our model replaces the SPPCSPC module with the RFB module to enhance the network's feature extraction capability. Second, a CA mechanism is introduced into the head part to enhance the network's feature representation ability and to improve detection accuracy. Third, a WIoU loss function is employed to address the low-quality samples hindering model generalization during training, thereby improving the model's overall performance. The experimental results indicate that the proposed model exhibits enhancements across various performance metrics. Specifically, there is a 1.6% advancement in mAP_0.5, a corresponding 1.6% enhancement in mAP_0.5:0.95, a 1.3% elevation in precision, and a 1% increase in recall. Moreover, the model achieves parameter reduction by 3.2 million, leading to a decrease of 2.5 GFLOPS in computational cost. Notably, there is also an improvement of 2.81 milliseconds in single-image detection speed. 

**Abstract (ZH)**: 一种改进的YOLOv7多类型绝缘子缺陷检测方法 

---
# Foreign-Object Detection in High-Voltage Transmission Line Based on Improved YOLOv8m 

**Title (ZH)**: 基于改进YOLOv8m的高压输电线路异物检测 

**Authors**: Zhenyue Wang, Guowu Yuan, Hao Zhou, Yi Ma, Yutang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.07175)  

**Abstract**: The safe operation of high-voltage transmission lines ensures the power grid's security. Various foreign objects attached to the transmission lines, such as balloons, kites and nesting birds, can significantly affect the safe and stable operation of high-voltage transmission lines. With the advancement of computer vision technology, periodic automatic inspection of foreign objects is efficient and necessary. Existing detection methods have low accuracy because foreign objects at-tached to the transmission lines are complex, including occlusions, diverse object types, significant scale variations, and complex backgrounds. In response to the practical needs of the Yunnan Branch of China Southern Power Grid Co., Ltd., this paper proposes an improved YOLOv8m-based model for detecting foreign objects on transmission lines. Experiments are conducted on a dataset collected from Yunnan Power Grid. The proposed model enhances the original YOLOv8m by in-corporating a Global Attention Module (GAM) into the backbone to focus on occluded foreign objects, replacing the SPPF module with the SPPCSPC module to augment the model's multiscale feature extraction capability, and introducing the Focal-EIoU loss function to address the issue of high- and low-quality sample imbalances. These improvements accelerate model convergence and enhance detection accuracy. The experimental results demonstrate that our proposed model achieves a 2.7% increase in mAP_0.5, a 4% increase in mAP_0.5:0.95, and a 6% increase in recall. 

**Abstract (ZH)**: 高压输电线路的安全运行保障了电网的安全。附着在输电线路上的各种异物，如气球、风筝和筑巢的鸟类，会对高压输电线路的安全稳定运行产生显著影响。随着计算机视觉技术的发展，定期自动检测异物是高效且必要的。由于输电线路上的异物复杂多样，包括遮挡、多样的物体类型、显著的尺度变化以及复杂的背景，现有检测方法的准确性较低。针对中国南方电网有限责任公司云南分公司实际需求，本文提出了一种基于改进YOLOv8m的检测模型，用于识别输电线路上的异物。实验在云南电网数据集上进行。该模型通过在骨干网络中引入全局注意力模块（GAM）来专注于遮挡的异物，用SPPCSPC模块替换SPPF模块以增强模型的多尺度特征提取能力，并引入Focal-EIoU损失函数以解决高质量和低质量样本不平衡的问题。这些改进加快了模型的收敛速度并提升了检测精度。实验结果表明，所提出的模型在mAP_0.5上提高了2.7%，在mAP_0.5:0.95上提高了4%，在召回率上提高了6%。 

---
# SemiHMER: Semi-supervised Handwritten Mathematical Expression Recognition using pseudo-labels 

**Title (ZH)**: SemiHMER: 使用伪标签的半监督手写数学表达式识别 

**Authors**: Kehua Chen, Haoyang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.07172)  

**Abstract**: In recent years, deep learning with Convolutional Neural Networks (CNNs) has achieved remarkable results in the field of HMER (Handwritten Mathematical Expression Recognition). However, it remains challenging to improve performance with limited labeled training data. This paper presents, for the first time, a simple yet effective semi-supervised HMER framework by introducing dual-branch semi-supervised learning. Specifically, we simplify the conventional deep co-training from consistency regularization to cross-supervised learning, where the prediction of one branch is used as a pseudo-label to supervise the other branch directly end-to-end. Considering that the learning of the two branches tends to converge in the later stages of model optimization, we also incorporate a weak-to-strong strategy by applying different levels of augmentation to each branch, which behaves like expanding the training data and improving the quality of network training. Meanwhile, We propose a novel module, Global Dynamic Counting Module(GDCM), to enhance the performance of the HMER decoder, which alleviates recognition inaccuracies in long-distance formula recognition and the occurrence of repeated characters. We release our code at this https URL. 

**Abstract (ZH)**: 近年来，基于卷积神经网络（CNNs）的深度学习在手写数学表达识别（HMER）领域取得了显著成果。然而，有限带标签训练数据的情况下提高性能仍具有挑战性。本文首次提出了一种简单有效的半监督HMER框架，通过引入双支路半监督学习。具体来说，我们将传统的深共训练从一致性正则化简化为交叉监督学习，其中一个支路的预测结果作为伪标签直接监督另一个支路的全过程训练。考虑到模型优化后期两个支路的学习趋于收敛，我们还引入了从弱到强的策略，对每个支路应用不同级别的数据增强，这种策略类似于扩展训练数据并提高网络训练质量。同时，我们提出了一种新型模块——全局动态计数模块（GDCM），以增强HMER解码器的性能，缓解长距离公式识别中的识别不准确和重复字符出现的问题。我们已将代码发布于此 <https://www.example.com>。 

---
# A Survey on Mamba Architecture for Vision Applications 

**Title (ZH)**: Mamba架构综述：面向视觉应用 

**Authors**: Fady Ibrahim, Guangjun Liu, Guanghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.07161)  

**Abstract**: Transformers have become foundational for visual tasks such as object detection, semantic segmentation, and video understanding, but their quadratic complexity in attention mechanisms presents scalability challenges. To address these limitations, the Mamba architecture utilizes state-space models (SSMs) for linear scalability, efficient processing, and improved contextual awareness. This paper investigates Mamba architecture for visual domain applications and its recent advancements, including Vision Mamba (ViM) and VideoMamba, which introduce bidirectional scanning, selective scanning mechanisms, and spatiotemporal processing to enhance image and video understanding. Architectural innovations like position embeddings, cross-scan modules, and hierarchical designs further optimize the Mamba framework for global and local feature extraction. These advancements position Mamba as a promising architecture in computer vision research and applications. 

**Abstract (ZH)**: Transformers在视觉任务中的应用已成为基础，但其注意力机制中的 Quadratic 复杂性提出了可扩展性挑战。为解决这些限制，Mamba 架构利用状态空间模型 (SSMs) 实现线性可扩展性、高效处理和增强的上下文意识。本文研究了 Mamba 架构在视觉领域中的应用及其近期进展，包括 Vision Mamba (ViM) 和 VideoMamba，它们引入了双向扫描、选择性扫描机制和时空处理，以增强图像和视频理解。架构创新如位置嵌入、跨扫描模块和分层设计进一步优化了 Mamba 框架中的全局和局部特征提取。这些进展使 Mamba 成为计算机视觉研究和应用中的有前途的架构。 

---
# Explaining 3D Computed Tomography Classifiers with Counterfactuals 

**Title (ZH)**: 用反事实解释3D计算机断层分类器 

**Authors**: Joseph Paul Cohen, Louis Blankemeier, Akshay Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2502.07156)  

**Abstract**: Counterfactual explanations in medical imaging are critical for understanding the predictions made by deep learning models. We extend the Latent Shift counterfactual generation method from 2D applications to 3D computed tomography (CT) scans. We address the challenges associated with 3D data, such as limited training samples and high memory demands, by implementing a slice-based approach. This method leverages a 2D encoder trained on CT slices, which are subsequently combined to maintain 3D context. We demonstrate this technique on two models for clinical phenotype prediction and lung segmentation. Our approach is both memory-efficient and effective for generating interpretable counterfactuals in high-resolution 3D medical imaging. 

**Abstract (ZH)**: 医疗影像中的反事实解释对于理解深度学习模型的预测至关重要。我们将2D应用程序中的Latent Shift反事实生成方法扩展到3D计算机断层扫描（CT）扫描。通过实施基于切片的方法解决3D数据的挑战，如训练样本有限和高内存需求，该方法利用在CT切片上训练的2D编码器，随后将切片组合以保持3D上下文。我们在这两种临床表型预测和肺部分割模型上展示了该技术。我们的方法在高分辨率3D医疗影像中生成可解释的反事实既节能又有效。 

---
# Few-Shot Multi-Human Neural Rendering Using Geometry Constraints 

**Title (ZH)**: 基于几何约束的少样本多人体神经渲染 

**Authors**: Qian li, Victoria Fernàndez Abrevaya, Franck Multon, Adnane Boukhayma  

**Link**: [PDF](https://arxiv.org/pdf/2502.07140)  

**Abstract**: We present a method for recovering the shape and radiance of a scene consisting of multiple people given solely a few images. Multi-human scenes are complex due to additional occlusion and clutter. For single-human settings, existing approaches using implicit neural representations have achieved impressive results that deliver accurate geometry and appearance. However, it remains challenging to extend these methods for estimating multiple humans from sparse views. We propose a neural implicit reconstruction method that addresses the inherent challenges of this task through the following contributions: First, we propose to use geometry constraints by exploiting pre-computed meshes using a human body model (SMPL). Specifically, we regularize the signed distances using the SMPL mesh and leverage bounding boxes for improved rendering. Second, we propose a ray regularization scheme to minimize rendering inconsistencies, and a saturation regularization for robust optimization in variable illumination. Extensive experiments on both real and synthetic datasets demonstrate the benefits of our approach and show state-of-the-art performance against existing neural reconstruction methods. 

**Abstract (ZH)**: 一种基于神经隐式表示的多人体场景形状和辐射量恢复方法 

---
# Unconstrained Body Recognition at Altitude and Range: Comparing Four Approaches 

**Title (ZH)**: 高空远距离不受约束的身体识别：四种方法的比较 

**Authors**: Blake A Myers, Matthew Q Hill, Veda Nandan Gandi, Thomas M Metz, Alice J O'Toole  

**Link**: [PDF](https://arxiv.org/pdf/2502.07130)  

**Abstract**: This study presents an investigation of four distinct approaches to long-term person identification using body shape. Unlike short-term re-identification systems that rely on temporary features (e.g., clothing), we focus on learning persistent body shape characteristics that remain stable over time. We introduce a body identification model based on a Vision Transformer (ViT) (Body Identification from Diverse Datasets, BIDDS) and on a Swin-ViT model (Swin-BIDDS). We also expand on previous approaches based on the Linguistic and Non-linguistic Core ResNet Identity Models (LCRIM and NLCRIM), but with improved training. All models are trained on a large and diverse dataset of over 1.9 million images of approximately 5k identities across 9 databases. Performance was evaluated on standard re-identification benchmark datasets (MARS, MSMT17, Outdoor Gait, DeepChange) and on an unconstrained dataset that includes images at a distance (from close-range to 1000m), at altitude (from an unmanned aerial vehicle, UAV), and with clothing change. A comparative analysis across these models provides insights into how different backbone architectures and input image sizes impact long-term body identification performance across real-world conditions. 

**Abstract (ZH)**: 本研究探讨了四种基于身体形状的长期人脸识别方法。不同于依赖临时特征（如衣物）的短期重识别系统，我们专注于学习能够长时间保持稳定的持续身体形状特征。我们提出了一种基于视觉变换器（ViT）的身体识别模型（多种数据集的身体识别，BIDDS）和一种基于Swin-ViT模型的身体识别模型（Swin-BIDDS）。我们还在此基础上扩展了基于语言核心和非语言核心ResNet身份模型（LCRIM和NLCRIM）的先前方法，但改进了训练方法。所有模型均在包含约5k个体身份跨越9个数据库的超过190万张图像的大规模多样数据集上进行训练。性能在标准重识别基准数据集（MARS、MSMT17、户外步态、DeepChange）及包含从近距离至1000米、无人机拍摄及着装变化的非约束性数据集上进行了评估。这些模型的对比分析揭示了不同的骨干架构和输入图像大小在实际条件下的长期身体识别性能差异。 

---
# From Image to Video: An Empirical Study of Diffusion Representations 

**Title (ZH)**: 从图像到视频：关于扩散表示的实证研究 

**Authors**: Pedro Vélez, Luisa F. Polanía, Yi Yang, Chuhan Zhang, Rishab Kabra, Anurag Arnab, Mehdi S. M. Sajjadi  

**Link**: [PDF](https://arxiv.org/pdf/2502.07001)  

**Abstract**: Diffusion models have revolutionized generative modeling, enabling unprecedented realism in image and video synthesis. This success has sparked interest in leveraging their representations for visual understanding tasks. While recent works have explored this potential for image generation, the visual understanding capabilities of video diffusion models remain largely uncharted. To address this gap, we systematically compare the same model architecture trained for video versus image generation, analyzing the performance of their latent representations on various downstream tasks including image classification, action recognition, depth estimation, and tracking. Results show that video diffusion models consistently outperform their image counterparts, though we find a striking range in the extent of this superiority. We further analyze features extracted from different layers and with varying noise levels, as well as the effect of model size and training budget on representation and generation quality. This work marks the first direct comparison of video and image diffusion objectives for visual understanding, offering insights into the role of temporal information in representation learning. 

**Abstract (ZH)**: 扩散模型颠覆了生成建模，使得图像和视频合成前所未有的逼真。这一成功激发了利用其表示方法进行视觉理解任务的兴趣。虽然近期研究探索了其在图像生成方面的潜力，但视频扩散模型的视觉理解能力尚待充分挖掘。为解决这一差距，我们系统地比较了用于视频和图像生成的相同模型架构，并对各种下游任务（包括图像分类、动作识别、深度估计和跟踪）中其隐含表示的性能进行了分析。结果表明，视频扩散模型在各个任务上普遍优于其图像对应模型，尽管其优越性在不同任务上存在显著差异。我们进一步分析了不同层提取的特征及不同噪声水平下的特征，并探讨了模型规模和训练预算对表示和生成质量的影响。本研究标志着首次直接比较视频和图像扩散目标在视觉理解中的表现，提供了关于时间信息在表示学习中作用的见解。 

---
# PyPotteryInk: One-Step Diffusion Model for Sketch to Publication-ready Archaeological Drawings 

**Title (ZH)**: PyPotteryInk：从素描到考古学-ready 图表的一步扩散模型 

**Authors**: Lorenzo Cardarelli  

**Link**: [PDF](https://arxiv.org/pdf/2502.06897)  

**Abstract**: Archaeological pottery documentation traditionally requires a time-consuming manual process of converting pencil sketches into publication-ready inked drawings. I present PyPotteryInk, an open-source automated pipeline that transforms archaeological pottery sketches into standardised publication-ready drawings using a one-step diffusion model. Built on a modified img2img-turbo architecture, the system processes drawings in a single forward pass while preserving crucial morphological details and maintaining archaeologic documentation standards and analytical value. The model employs an efficient patch-based approach with dynamic overlap, enabling high-resolution output regardless of input drawing size. I demonstrate the effectiveness of the approach on a dataset of Italian protohistoric pottery drawings, where it successfully captures both fine details like decorative patterns and structural elements like vessel profiles or handling elements. Expert evaluation confirms that the generated drawings meet publication standards while significantly reducing processing time from hours to seconds per drawing. The model can be fine-tuned to adapt to different archaeological contexts with minimal training data, making it versatile across various pottery documentation styles. The pre-trained models, the Python library and comprehensive documentation are provided to facilitate adoption within the archaeological research community. 

**Abstract (ZH)**: 考古陶器记录传统上需要耗时的手动过程，将铅笔草图转换为出版-ready 的墨绘图纸。我提出了PyPotteryInk，这是一个开源的自动化管道，使用单步扩散模型将考古陶器草图转换为标准化的出版-ready 绘图纸。该系统基于修改后的img2img-turbo架构，一次前向传播即可处理绘图，同时保留关键的形态学细节并维持考古记录标准和分析价值。模型采用高效的基于补丁的方法，具有动态重叠，能够在不考虑输入绘图纸大小的情况下生成高分辨率输出。在意大利前史陶器绘图的数据集上展示了该方法的有效性，成功捕捉到了精细细节如装饰图案和结构要素如容器轮廓或握持元素。专家评估证实，生成的绘图纸符合出版标准，同时将处理时间从每张绘图纸数小时缩短到数秒。该模型可以少量训练数据微调以适应不同的考古背景，使其适用于各种陶器记录风格。预训练模型、Python库和全面的文档已提供，以方便在考古研究社区中的采纳。 

---
# Topological derivative approach for deep neural network architecture adaptation 

**Title (ZH)**: 拓扑导数方法在深度神经网络架构适应中的应用 

**Authors**: C G Krishnanunni, Tan Bui-Thanh, Clint Dawson  

**Link**: [PDF](https://arxiv.org/pdf/2502.06885)  

**Abstract**: This work presents a novel algorithm for progressively adapting neural network architecture along the depth. In particular, we attempt to address the following questions in a mathematically principled way: i) Where to add a new capacity (layer) during the training process? ii) How to initialize the new capacity? At the heart of our approach are two key ingredients: i) the introduction of a ``shape functional" to be minimized, which depends on neural network topology, and ii) the introduction of a topological derivative of the shape functional with respect to the neural network topology. Using an optimal control viewpoint, we show that the network topological derivative exists under certain conditions, and its closed-form expression is derived. In particular, we explore, for the first time, the connection between the topological derivative from a topology optimization framework with the Hamiltonian from optimal control theory. Further, we show that the optimality condition for the shape functional leads to an eigenvalue problem for deep neural architecture adaptation. Our approach thus determines the most sensitive location along the depth where a new layer needs to be inserted during the training phase and the associated parametric initialization for the newly added layer. We also demonstrate that our layer insertion strategy can be derived from an optimal transport viewpoint as a solution to maximizing a topological derivative in $p$-Wasserstein space, where $p>= 1$. Numerical investigations with fully connected network, convolutional neural network, and vision transformer on various regression and classification problems demonstrate that our proposed approach can outperform an ad-hoc baseline network and other architecture adaptation strategies. Further, we also demonstrate other applications of topological derivative in fields such as transfer learning. 

**Abstract (ZH)**: 本文提出了一种沿深度渐进适应神经网络架构的新算法。特别是在数学原理上尝试解决以下问题：i）在训练过程中何时添加新的能力（层）？ii）如何初始化新的能力？我们方法的核心包括两个关键成分：i）引入一个依赖于神经网络拓扑结构的“形状泛函”并将其最小化，ii）引入形状泛函相对于神经网络拓扑结构的拓扑导数。从最优控制的角度出发，我们证明在某些条件下网络拓扑导数存在，并推导出其闭式表达式。特别是，我们首次探讨了拓扑优化框架中的拓扑导数与最优控制理论中的哈密顿量之间的联系。进一步地，我们表明形状泛函的最优性条件导致了深度神经架构适应的特征值问题。因此，我们的方法确定了训练阶段神经网络深度中需要插入新层的最敏感位置以及新添加层的参数初始化。我们还证明，从最优传输的观点来看，我们的层插入策略可以通过最大化p-Wasserstein空间中的拓扑导数来推导。数值实验表明，在各种回归和分类问题上，我们的方法可以超越随机基线网络和其他架构适应策略。此外，我们还展示了拓扑导数在领域适应等领域的其他应用。 

---
# BF-GAN: Development of an AI-driven Bubbly Flow Image Generation Model Using Generative Adversarial Networks 

**Title (ZH)**: BF-GAN：基于生成对抗网络的气泡流图像生成模型开发 

**Authors**: Wen Zhou, Shuichiro Miwa, Yang Liu, Koji Okamoto  

**Link**: [PDF](https://arxiv.org/pdf/2502.06863)  

**Abstract**: A generative AI architecture called bubbly flow generative adversarial networks (BF-GAN) is developed, designed to generate realistic and high-quality bubbly flow images through physically conditioned inputs, jg and jf. Initially, 52 sets of bubbly flow experiments under varying conditions are conducted to collect 140,000 bubbly flow images with physical labels of jg and jf for training data. A multi-scale loss function is then developed, incorporating mismatch loss and pixel loss to enhance the generative performance of BF-GAN further. Regarding evaluative metrics of generative AI, the BF-GAN has surpassed conventional GAN. Physically, key parameters of bubbly flow generated by BF-GAN are extracted and compared with measurement values and empirical correlations, validating BF-GAN's generative performance. The comparative analysis demonstrate that the BF-GAN can generate realistic and high-quality bubbly flow images with any given jg and jf within the research scope.
BF-GAN offers a generative AI solution for two-phase flow research, substantially lowering the time and cost required to obtain high-quality data. In addition, it can function as a benchmark dataset generator for bubbly flow detection and segmentation algorithms, enhancing overall productivity in this research domain. The BF-GAN model is available online (this https URL). 

**Abstract (ZH)**: 一种名为泡沫流生成对抗网络（BF-GAN）的生成式AI架构被开发出来，设计用于通过物理条件输入jg和jf生成高质量的泡沫流图像。初始阶段，进行了52组不同条件下的泡沫流实验，收集了140,000张带有物理标签jg和jf的泡沫流图像作为训练数据。随后，开发了一种多尺度损失函数，结合不匹配损失和像素损失，进一步提升BF-GAN的生成性能。在生成式AI评价指标方面，BF-GAN超越了传统GAN。从物理角度，BF-GAN生成的泡沫流的关键参数被提取并与测量值和经验相关性进行了比较，验证了BF-GAN的生成性能。比较分析表明，BF-GAN可以在研究范围内生成任意给定jg和jf的高质量和逼真的泡沫流图像。

BF-GAN为两相流研究提供了生成式AI解决方案，大幅降低了获取高质量数据所需的时间和成本。此外，BF-GAN还可以作为泡沫流检测和分割算法的基准数据集生成器，提高该研究领域的整体生产力。BF-GAN模型已上线（此链接）。 

---
# Learning to Synthesize Compatible Fashion Items Using Semantic Alignment and Collocation Classification: An Outfit Generation Framework 

**Title (ZH)**: 基于语义对齐和共现分类的服装项合成兼容学习：一套装生成框架 

**Authors**: Dongliang Zhou, Haijun Zhang, Kai Yang, Linlin Liu, Han Yan, Xiaofei Xu, Zhao Zhang, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2502.06827)  

**Abstract**: The field of fashion compatibility learning has attracted great attention from both the academic and industrial communities in recent years. Many studies have been carried out for fashion compatibility prediction, collocated outfit recommendation, artificial intelligence (AI)-enabled compatible fashion design, and related topics. In particular, AI-enabled compatible fashion design can be used to synthesize compatible fashion items or outfits in order to improve the design experience for designers or the efficacy of recommendations for customers. However, previous generative models for collocated fashion synthesis have generally focused on the image-to-image translation between fashion items of upper and lower clothing. In this paper, we propose a novel outfit generation framework, i.e., OutfitGAN, with the aim of synthesizing a set of complementary items to compose an entire outfit, given one extant fashion item and reference masks of target synthesized items. OutfitGAN includes a semantic alignment module, which is responsible for characterizing the mapping correspondence between the existing fashion items and the synthesized ones, to improve the quality of the synthesized images, and a collocation classification module, which is used to improve the compatibility of a synthesized outfit. In order to evaluate the performance of our proposed models, we built a large-scale dataset consisting of 20,000 fashion outfits. Extensive experimental results on this dataset show that our OutfitGAN can synthesize photo-realistic outfits and outperform state-of-the-art methods in terms of similarity, authenticity and compatibility measurements. 

**Abstract (ZH)**: 时尚兼容性学习领域近年来吸引了学术界和工业界的广泛关注。许多研究针对时尚兼容性预测、配对服装推荐、人工智能-enable的兼容服装设计及相关主题进行了探讨。特别是在人工智能-enable的兼容服装设计方面，可以合成兼容的服装单品或搭配，以提高设计师的设计体验或提升对客户的推荐效果。然而，现有的配对服装合成生成模型通常仅专注于上下装之间的图像到图像的转化。本文提出了一种新的服装生成框架，即OutfitGAN，旨在给定一个现有的时尚单品及目标合成单品的参考掩码时，合成一系列互补单品以组成完整的服装搭配。OutfitGAN包括一个语义对齐模块，用于表征现有单品与合成单品之间的映射对应关系，以提高合成图像的质量；以及一个配对分类模块，用于提升合成服装搭配的兼容性。为了评估所提模型的性能，我们构建了一个包含20,000个时尚搭配的大规模数据集。在此数据集上的大量实验结果表明，我们的OutfitGAN能够生成逼真的服装搭配，并在相似度、真实性和兼容性测量方面优于现有方法。 

---
# Diffusion Instruction Tuning 

**Title (ZH)**: 扩散指令调整 

**Authors**: Chen Jin, Ryutaro Tanno, Amrutha Saseendran, Tom Diethe, Philip Teare  

**Link**: [PDF](https://arxiv.org/pdf/2502.06814)  

**Abstract**: We introduce Lavender, a simple supervised fine-tuning (SFT) method that boosts the performance of advanced vision-language models (VLMs) by leveraging state-of-the-art image generation models such as Stable Diffusion. Specifically, Lavender aligns the text-vision attention in the VLM transformer with the equivalent used by Stable Diffusion during SFT, instead of adapting separate encoders. This alignment enriches the model's visual understanding and significantly boosts performance across in- and out-of-distribution tasks. Lavender requires just 0.13 million training examples, 2.5% of typical large-scale SFT datasets, and fine-tunes on standard hardware (8 GPUs) in a single day. It consistently improves state-of-the-art open-source multimodal LLMs (e.g., Llama-3.2-11B, MiniCPM-Llama3-v2.5), achieving up to 30% gains and a 68% boost on challenging out-of-distribution medical QA tasks. By efficiently transferring the visual expertise of image generators with minimal supervision, Lavender offers a scalable solution for more accurate vision-language systems. All code, training data, and models will be shared at this https URL. 

**Abstract (ZH)**: Lavender：一种通过利用先进图像生成模型提升高级视觉-语言模型性能的简单监督微调方法 

---
