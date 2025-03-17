# BEVDiffLoc: End-to-End LiDAR Global Localization in BEV View based on Diffusion Model 

**Title (ZH)**: BEVDiffLoc：基于扩散模型的端到端LiDAR全局局部化在BEV视图中 

**Authors**: Ziyue Wang, Chenghao Shi, Neng Wang, Qinghua Yu, Xieyuanli Chen, Huimin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11372)  

**Abstract**: Localization is one of the core parts of modern robotics. Classic localization methods typically follow the retrieve-then-register paradigm, achieving remarkable success. Recently, the emergence of end-to-end localization approaches has offered distinct advantages, including a streamlined system architecture and the elimination of the need to store extensive map data. Although these methods have demonstrated promising results, current end-to-end localization approaches still face limitations in robustness and accuracy. Bird's-Eye-View (BEV) image is one of the most widely adopted data representations in autonomous driving. It significantly reduces data complexity while preserving spatial structure and scale consistency, making it an ideal representation for localization tasks. However, research on BEV-based end-to-end localization remains notably insufficient. To fill this gap, we propose BEVDiffLoc, a novel framework that formulates LiDAR localization as a conditional generation of poses. Leveraging the properties of BEV, we first introduce a specific data augmentation method to significantly enhance the diversity of input data. Then, the Maximum Feature Aggregation Module and Vision Transformer are employed to learn robust features while maintaining robustness against significant rotational view variations. Finally, we incorporate a diffusion model that iteratively refines the learned features to recover the absolute pose. Extensive experiments on the Oxford Radar RobotCar and NCLT datasets demonstrate that BEVDiffLoc outperforms the baseline methods. Our code is available at this https URL. 

**Abstract (ZH)**: 基于BEV的端到端局部化：BEVDiffLoc框架 

---
# Enhancing Hand Palm Motion Gesture Recognition by Eliminating Reference Frame Bias via Frame-Invariant Similarity Measures 

**Title (ZH)**: 通过使用帧不变相似度度量消除参考帧偏差以增强手掌运动手势识别 

**Authors**: Arno Verduyn, Maxim Vochten, Joris De Schutter  

**Link**: [PDF](https://arxiv.org/pdf/2503.11352)  

**Abstract**: The ability of robots to recognize human gestures facilitates a natural and accessible human-robot collaboration. However, most work in gesture recognition remains rooted in reference frame-dependent representations. This poses a challenge when reference frames vary due to different work cell layouts, imprecise frame calibrations, or other environmental changes. This paper investigated the use of invariant trajectory descriptors for robust hand palm motion gesture recognition under reference frame changes. First, a novel dataset of recorded Hand Palm Motion (HPM) gestures is introduced. The motion gestures in this dataset were specifically designed to be distinguishable without dependence on specific reference frames or directional cues. Afterwards, multiple invariant trajectory descriptor approaches were benchmarked to assess how their performances generalize to this novel HPM dataset. After this offline benchmarking, the best scoring approach is validated for online recognition by developing a real-time Proof of Concept (PoC). In this PoC, hand palm motion gestures were used to control the real-time movement of a manipulator arm. The PoC demonstrated a high recognition reliability in real-time operation, achieving an $F_1$-score of 92.3%. This work demonstrates the effectiveness of the invariant descriptor approach as a standalone solution. Moreover, we believe that the invariant descriptor approach can also be utilized within other state-of-the-art pattern recognition and learning systems to improve their robustness against reference frame variations. 

**Abstract (ZH)**: 基于不变轨迹描述符的人手掌运动手势识别在参考框架变化下的稳健性研究 

---
# Image-Goal Navigation Using Refined Feature Guidance and Scene Graph Enhancement 

**Title (ZH)**: 基于精炼特征引导和场景图增强的目标导向导航 

**Authors**: Zhicheng Feng, Xieyuanli Chen, Chenghao Shi, Lun Luo, Zhichao Chen, Yun-Hui Liu, Huimin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10986)  

**Abstract**: In this paper, we introduce a novel image-goal navigation approach, named RFSG. Our focus lies in leveraging the fine-grained connections between goals, observations, and the environment within limited image data, all the while keeping the navigation architecture simple and lightweight. To this end, we propose the spatial-channel attention mechanism, enabling the network to learn the importance of multi-dimensional features to fuse the goal and observation features. In addition, a selfdistillation mechanism is incorporated to further enhance the feature representation capabilities. Given that the navigation task needs surrounding environmental information for more efficient navigation, we propose an image scene graph to establish feature associations at both the image and object levels, effectively encoding the surrounding scene information. Crossscene performance validation was conducted on the Gibson and HM3D datasets, and the proposed method achieved stateof-the-art results among mainstream methods, with a speed of up to 53.5 frames per second on an RTX3080. This contributes to the realization of end-to-end image-goal navigation in realworld scenarios. The implementation and model of our method have been released at: this https URL. 

**Abstract (ZH)**: 基于RFSG的空间注意力机制及其在图像目标导航中的应用 

---
# Disentangled Object-Centric Image Representation for Robotic Manipulation 

**Title (ZH)**: 去耦对象中心图像表示在机器人操作中的应用 

**Authors**: David Emukpere, Romain Deffayet, Bingbing Wu, Romain Brégier, Michael Niemaz, Jean-Luc Meunier, Denys Proux, Jean-Michel Renders, Seungsu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.11565)  

**Abstract**: Learning robotic manipulation skills from vision is a promising approach for developing robotics applications that can generalize broadly to real-world scenarios. As such, many approaches to enable this vision have been explored with fruitful results. Particularly, object-centric representation methods have been shown to provide better inductive biases for skill learning, leading to improved performance and generalization. Nonetheless, we show that object-centric methods can struggle to learn simple manipulation skills in multi-object environments. Thus, we propose DOCIR, an object-centric framework that introduces a disentangled representation for objects of interest, obstacles, and robot embodiment. We show that this approach leads to state-of-the-art performance for learning pick and place skills from visual inputs in multi-object environments and generalizes at test time to changing objects of interest and distractors in the scene. Furthermore, we show its efficacy both in simulation and zero-shot transfer to the real world. 

**Abstract (ZH)**: 从视觉学习机器人 manipulation 技能是开发能够在多种现实场景中泛化的机器人应用的一种有前景的方法。因此，许多促进这一目标的方法已被探索并取得了丰硕的成果。特别是以对象为中心的表示方法已被证明能够为技能学习提供更好的归纳偏置，从而提高性能和泛化能力。然而，我们发现以对象为中心的方法在多对象环境中学习简单的 manipulation 技能可能存在困难。因此，我们提出了一种以对象为中心的 DOCIR 框架，该框架引入了对象、障碍物和机器人主体的解耦表示。实验表明，该方法在多对象环境中从视觉输入学习 pick and place 技能方面达到了最先进的性能，并且在测试时能够泛化到场景中变化的对象和干扰物。此外，我们在仿真和零样本到真实世界的转移中展示了其有效性。 

---
# TASTE-Rob: Advancing Video Generation of Task-Oriented Hand-Object Interaction for Generalizable Robotic Manipulation 

**Title (ZH)**: TASTE-Rob：推进面向任务的手物交互视频生成以提高通用化机器人操作能力 

**Authors**: Hongxiang Zhao, Xingchen Liu, Mutian Xu, Yiming Hao, Weikai Chen, Xiaoguang Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.11423)  

**Abstract**: We address key limitations in existing datasets and models for task-oriented hand-object interaction video generation, a critical approach of generating video demonstrations for robotic imitation learning. Current datasets, such as Ego4D, often suffer from inconsistent view perspectives and misaligned interactions, leading to reduced video quality and limiting their applicability for precise imitation learning tasks. Towards this end, we introduce TASTE-Rob -- a pioneering large-scale dataset of 100,856 ego-centric hand-object interaction videos. Each video is meticulously aligned with language instructions and recorded from a consistent camera viewpoint to ensure interaction clarity. By fine-tuning a Video Diffusion Model (VDM) on TASTE-Rob, we achieve realistic object interactions, though we observed occasional inconsistencies in hand grasping postures. To enhance realism, we introduce a three-stage pose-refinement pipeline that improves hand posture accuracy in generated videos. Our curated dataset, coupled with the specialized pose-refinement framework, provides notable performance gains in generating high-quality, task-oriented hand-object interaction videos, resulting in achieving superior generalizable robotic manipulation. The TASTE-Rob dataset will be made publicly available upon publication to foster further advancements in the field. 

**Abstract (ZH)**: 我们解决现有面向任务的手物交互视频生成数据集和模型的关键限制，这是用于机器人模仿学习的视频演示生成关键方法的一个重要方面。当前的数据集，如Ego4D，通常存在视角不一致和交互不匹配的问题，这降低了视频质量，并限制了其在精确模仿学习任务中的应用。为此，我们引入TASTE-Rob——一个包含100,856个自我中心手物交互视频的开创性大规模数据集。每个视频都与语言指令精确对齐，并从一致的摄像机视角录制，以确保交互清晰度。通过在TASTE-Rob上微调视频扩散模型（VDM），我们实现了现实的手物交互，虽然观察到手部抓握姿态偶尔存在不一致性。为了增强现实感，我们引入了一个三阶段姿态精炼管道，提高了生成视频中手部姿态的准确性。我们精心策划的数据集与专门的姿态精炼框架相结合，在生成高质量、面向任务的手物交互视频方面取得了显著性能提升，从而实现了更可泛化的机器人操作。TASTE-Rob数据集将在发表后公开，以促进该领域的进一步发展。 

---
# LuSeg: Efficient Negative and Positive Obstacles Segmentation via Contrast-Driven Multi-Modal Feature Fusion on the Lunar 

**Title (ZH)**: LuSeg: 通过月球对比驱动多模态特征融合的负障碍和正障碍高效分割 

**Authors**: Shuaifeng Jiao, Zhiwen Zeng, Zhuoqun Su, Xieyuanli Chen, Zongtan Zhou, Huimin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11409)  

**Abstract**: As lunar exploration missions grow increasingly complex, ensuring safe and autonomous rover-based surface exploration has become one of the key challenges in lunar exploration tasks. In this work, we have developed a lunar surface simulation system called the Lunar Exploration Simulator System (LESS) and the LunarSeg dataset, which provides RGB-D data for lunar obstacle segmentation that includes both positive and negative obstacles. Additionally, we propose a novel two-stage segmentation network called LuSeg. Through contrastive learning, it enforces semantic consistency between the RGB encoder from Stage I and the depth encoder from Stage II. Experimental results on our proposed LunarSeg dataset and additional public real-world NPO road obstacle dataset demonstrate that LuSeg achieves state-of-the-art segmentation performance for both positive and negative obstacles while maintaining a high inference speed of approximately 57\,Hz. We have released the implementation of our LESS system, LunarSeg dataset, and the code of LuSeg at:this https URL. 

**Abstract (ZH)**: 随着月球探测任务日益复杂，确保安全自主的月球车表面探测已成为月球探测任务中的关键挑战之一。本文开发了名为月球探测仿真系统（LESS）的月球表面仿真系统和提供了包含正负障碍物的RGB-D数据集LunarSeg，并提出了一种新型两阶段分割网络LuSeg。通过对比学习，它在第一阶段的RGB编码器和第二阶段的深度编码器之间强制执行语义一致性。实验结果表明，LuSeg在我们提出的LunarSeg数据集和额外的公开真实世界NPO道路障碍数据集上实现了正负障碍物的最先进的分割性能，同时保持了约57 Hz的高推理速度。我们已在此处发布了LESS系统、LunarSeg数据集以及LuSeg的代码：this https URL。 

---
# 3D Extended Object Tracking based on Extruded B-Spline Side View Profiles 

**Title (ZH)**: 基于扩展B样条侧面视图轮廓的3D扩展对象跟踪 

**Authors**: Longfei Han, Klaus Kefferpütz, Jürgen Beyerer  

**Link**: [PDF](https://arxiv.org/pdf/2503.10730)  

**Abstract**: Object tracking is an essential task for autonomous systems. With the advancement of 3D sensors, these systems can better perceive their surroundings using effective 3D Extended Object Tracking (EOT) methods. Based on the observation that common road users are symmetrical on the right and left sides in the traveling direction, we focus on the side view profile of the object. In order to leverage of the development in 2D EOT and balance the number of parameters of a shape model in the tracking algorithms, we propose a method for 3D extended object tracking (EOT) by describing the side view profile of the object with B-spline curves and forming an extrusion to obtain a 3D extent. The use of B-spline curves exploits their flexible representation power by allowing the control points to move freely. The algorithm is developed into an Extended Kalman Filter (EKF). For a through evaluation of this method, we use simulated traffic scenario of different vehicle models and realworld open dataset containing both radar and lidar data. 

**Abstract (ZH)**: 基于2D EOT发展的3D扩域目标跟踪方法：利用B样条曲线描述侧视轮廓并形成 extrusion 

---
# Video Individual Counting for Moving Drones 

**Title (ZH)**: 移动无人机的视频个体计数 

**Authors**: Yaowu Fan, Jia Wan, Tao Han, Antoni B. Chan, Andy J. Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.10701)  

**Abstract**: Video Individual Counting (VIC) has received increasing attentions recently due to its importance in intelligent video surveillance. Existing works are limited in two aspects, i.e., dataset and method. Previous crowd counting datasets are captured with fixed or rarely moving cameras with relatively sparse individuals, restricting evaluation for a highly varying view and time in crowded scenes. While VIC methods have been proposed based on localization-then-association or localization-then-classification, they may not perform well due to difficulty in accurate localization of crowded and small targets under challenging scenarios. To address these issues, we collect a MovingDroneCrowd Dataset and propose a density map based VIC method. Different from existing datasets, our dataset consists of videos captured by fast-moving drones in crowded scenes under diverse illuminations, shooting heights and angles. Other than localizing individuals, we propose a Depth-wise Cross-Frame Attention (DCFA) module, which directly estimate inflow and outflow density maps through learning shared density maps between consecutive frames. The inflow density maps across frames are summed up to obtain the number of unique pedestrians in a video. Experiments on our datasets and publicly available ones show the superiority of our method over the state of the arts for VIC in highly dynamic and complex crowded scenes. Our dataset and codes will be released publicly. 

**Abstract (ZH)**: 视频个体计数（VIC）由于其在智能视频监控中的重要性，最近受到了越来越多的关注。现有的工作在数据集和方法上存在两个局限性。以往的群体计数数据集大多由固定或移动缓慢的相机在相对稀疏的人群中拍摄，限制了对视角和时间变化较大的拥挤场景的评估。虽然已经提出了基于定位-关联或定位-分类的方法来进行视频个体计数（VIC），但在挑战性场景下，由于难以准确定位拥挤和小的目标，这些方法可能表现不佳。为了解决这些问题，我们收集了一个移动无人机 crowd 数据集，并提出了一种基于密度图的视频个体计数方法。不同于现有的数据集，我们的数据集由快速移动的无人机在多种照明、拍摄高度和角度下拍摄的拥挤场景视频组成。除了定位个体，我们还提出了一种深度可分离帧间注意力模块（DCFA），直接通过学习连续帧之间共享的密度图来估计流入和流出密度图。帧间流入密度图的累加得到视频中唯一行人的数量。在我们自己的数据集和公开可用的数据集上的实验表明，与最先进的方法相比，我们的方法在高度动态和复杂拥挤场景中的视频个体计数（VIC）中具有优越性。我们的数据集和代码将公开发布。 

---
# Enhancing Deep Learning Based Structured Illumination Microscopy Reconstruction with Light Field Awareness 

**Title (ZH)**: 基于光源场意识增强的深度学习结构光显微镜重建 

**Authors**: Long-Kun Shan, Ze-Hao Wang, Tong-Tian Weng, Xiang-Dong Chen, Fang-Wen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.11640)  

**Abstract**: Structured illumination microscopy (SIM) is a pivotal technique for dynamic subcellular imaging in live cells. Conventional SIM reconstruction algorithms depend on accurately estimating the illumination pattern and can introduce artefacts when this estimation is imprecise. Although recent deep learning-based SIM reconstruction methods have improved speed, accuracy, and robustness, they often struggle with out-of-distribution data. To address this limitation, we propose an Awareness-of-Light-field SIM (AL-SIM) reconstruction approach that directly estimates the actual light field to correct for errors arising from data distribution shifts. Through comprehensive experiments on both simulated filament structures and live BSC1 cells, our method demonstrates a 7% reduction in the normalized root mean square error (NRMSE) and substantially lowers reconstruction artefacts. By minimizing these artefacts and improving overall accuracy, AL-SIM broadens the applicability of SIM for complex biological systems. 

**Abstract (ZH)**: 基于光照场感知的结构照明 microscopy (AL-SIM) 重建方法 

---
# RASA: Replace Anyone, Say Anything -- A Training-Free Framework for Audio-Driven and Universal Portrait Video Editing 

**Title (ZH)**: RASA：Replace Anyone, Say Anything — 一种无需训练的音频驱动通用portrait视频编辑框架 

**Authors**: Tianrui Pan, Lin Liu, Jie Liu, Xiaopeng Zhang, Jie Tang, Gangshan Wu, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2503.11571)  

**Abstract**: Portrait video editing focuses on modifying specific attributes of portrait videos, guided by audio or video streams. Previous methods typically either concentrate on lip-region reenactment or require training specialized models to extract keypoints for motion transfer to a new identity. In this paper, we introduce a training-free universal portrait video editing framework that provides a versatile and adaptable editing strategy. This framework supports portrait appearance editing conditioned on the changed first reference frame, as well as lip editing conditioned on varied speech, or a combination of both. It is based on a Unified Animation Control (UAC) mechanism with source inversion latents to edit the entire portrait, including visual-driven shape control, audio-driven speaking control, and inter-frame temporal control. Furthermore, our method can be adapted to different scenarios by adjusting the initial reference frame, enabling detailed editing of portrait videos with specific head rotations and facial expressions. This comprehensive approach ensures a holistic and flexible solution for portrait video editing. The experimental results show that our model can achieve more accurate and synchronized lip movements for the lip editing task, as well as more flexible motion transfer for the appearance editing task. Demo is available at this https URL. 

**Abstract (ZH)**: 面部视频编辑专注于通过音频或视频流引导修改面部视频的特定属性。先前的方法通常要么专注于唇部区域再现，要么需要训练专门的模型来提取关键点以将运动转移到新身份上。在本文中，我们介绍了一种无需训练的通用面部视频编辑框架，提供了一种灵活且可适应的编辑策略。该框架支持以改变的第一参考帧为条件的面部外观编辑，以及以变化的语音为条件的唇部编辑，也可以同时结合两者。该框架基于统一动画控制（UAC）机制，利用源反转潜在变量来编辑整个面部，包括视觉驱动的形状控制、音频驱动的说话控制和区间时间控制。此外，通过调整初始参考帧，我们的方法可以适应不同的场景，使面部视频编辑具有特定的头部旋转和面部表情的详细编辑能力。这种方法确保了面部视频编辑的一个全面和灵活的解决方案。实验结果表明，我们的模型在唇部编辑任务中可以实现更准确和同步的唇部运动，在外观编辑任务中可以实现更灵活的运动转移。演示可在以下链接查看：这个 https URL。 

---
# FLASHμ: Fast Localizing And Sizing of Holographic Microparticles 

**Title (ZH)**: FLASHμ：快速定位和定容全息微粒子 

**Authors**: Ayush Paliwal, Oliver Schlenczek, Birte Thiede, Manuel Santos Pereira, Katja Stieger, Eberhard Bodenschatz, Gholamhossein Bagheri, Alexander Ecker  

**Link**: [PDF](https://arxiv.org/pdf/2503.11538)  

**Abstract**: Reconstructing the 3D location and size of microparticles from diffraction images - holograms - is a computationally expensive inverse problem that has traditionally been solved using physics-based reconstruction methods. More recently, researchers have used machine learning methods to speed up the process. However, for small particles in large sample volumes the performance of these methods falls short of standard physics-based reconstruction methods. Here we designed a two-stage neural network architecture, FLASH$\mu$, to detect small particles (6-100$\mu$m) from holograms with large sample depths up to 20cm. Trained only on synthetic data with added physical noise, our method reliably detects particles of at least 9$\mu$m diameter in real holograms, comparable to the standard reconstruction-based approaches while operating on smaller crops, at quarter of the original resolution and providing roughly a 600-fold speedup. In addition to introducing a novel approach to a non-local object detection or signal demixing problem, our work could enable low-cost, real-time holographic imaging setups. 

**Abstract (ZH)**: 从衍射图像重建微粒子的3D位置和尺寸是一个计算密集型的逆问题，传统上使用基于物理的方法求解。最近，研究人员使用机器学习方法加速此过程。然而，对于大样本体积中的小粒子，这些方法的表现不如标准的基于物理的重建方法。我们设计了一种两阶段神经网络架构FLASH$\mu$，用于从具有大样本深度（最多20cm）的全息图中检测小粒子（6-100$\mu$m）。仅通过在具有添加物理噪声的合成数据上训练，我们的方法在真实全息图中可靠地检测出至少9$\mu$m直径的粒子，性能与标准的基于重建的方法相当，同时处理更小的图像区域，原始分辨率的四分之一，并提供约600倍的速度提升。除了提出一种新的非局域对象检测或信号去混方法外，我们的工作还有可能使低成本、实时全息成像系统成为可能。 

---
# HiTVideo: Hierarchical Tokenizers for Enhancing Text-to-Video Generation with Autoregressive Large Language Models 

**Title (ZH)**: HiTVideo：层次化分词器用于增强基于自回归大规模语言模型的文本到视频生成 

**Authors**: Ziqin Zhou, Yifan Yang, Yuqing Yang, Tianyu He, Houwen Peng, Kai Qiu, Qi Dai, Lili Qiu, Chong Luo, Lingqiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11513)  

**Abstract**: Text-to-video generation poses significant challenges due to the inherent complexity of video data, which spans both temporal and spatial dimensions. It introduces additional redundancy, abrupt variations, and a domain gap between language and vision tokens while generation. Addressing these challenges requires an effective video tokenizer that can efficiently encode video data while preserving essential semantic and spatiotemporal information, serving as a critical bridge between text and vision. Inspired by the observation in VQ-VAE-2 and workflows of traditional animation, we propose HiTVideo for text-to-video generation with hierarchical tokenizers. It utilizes a 3D causal VAE with a multi-layer discrete token framework, encoding video content into hierarchically structured codebooks. Higher layers capture semantic information with higher compression, while lower layers focus on fine-grained spatiotemporal details, striking a balance between compression efficiency and reconstruction quality. Our approach efficiently encodes longer video sequences (e.g., 8 seconds, 64 frames), reducing bits per pixel (bpp) by approximately 70\% compared to baseline tokenizers, while maintaining competitive reconstruction quality. We explore the trade-offs between compression and reconstruction, while emphasizing the advantages of high-compressed semantic tokens in text-to-video tasks. HiTVideo aims to address the potential limitations of existing video tokenizers in text-to-video generation tasks, striving for higher compression ratios and simplify LLMs modeling under language guidance, offering a scalable and promising framework for advancing text to video generation. Demo page: this https URL. 

**Abstract (ZH)**: 文本到视频生成由于视频数据固有的时空复杂性而面临重大挑战，需要有效的时间级和空间级视频分词器来高效编码视频数据并保留关键的语义和时空信息，成为文本与视觉之间的关键桥梁。受VQ-VAE-2和传统动画工作流的启发，我们提出HiTVideo，用于具有层次分词器的文本到视频生成。它采用三维因果VAE与多层离散分词框架，将视频内容编码为分层结构的词汇表。较高层次捕获更高压缩比的语义信息，较低层次关注精细的时空细节，在压缩效率和重建质量之间取得平衡。该方法能够有效编码较长的视频序列（如8秒，64帧），与基准分词器相比，降低每像素位数（bpp）约70%，同时保持竞争力的重建质量。我们探讨了压缩和重建之间的权衡，强调了高压缩语义分词在文本到视频任务中的优势。HiTVideo旨在解决现有视频分词器在文本到视频生成任务中的潜在局限性，目标是更高的压缩比和在语言引导下简化LLM建模，提供一个可扩展且有前景的框架以推进文本到视频生成。 

---
# Learning to reset in target search problems 

**Title (ZH)**: 在目标搜索问题中学习重置 

**Authors**: Gorka Muñoz-Gil, Hans J. Briegel, Michele Caraglio  

**Link**: [PDF](https://arxiv.org/pdf/2503.11330)  

**Abstract**: Target search problems are central to a wide range of fields, from biological foraging to the optimization algorithms. Recently, the ability to reset the search has been shown to significantly improve the searcher's efficiency. However, the optimal resetting strategy depends on the specific properties of the search problem and can often be challenging to determine. In this work, we propose a reinforcement learning (RL)-based framework to train agents capable of optimizing their search efficiency in environments by learning how to reset. First, we validate the approach in a well-established benchmark: the Brownian search with resetting. There, RL agents consistently recover strategies closely resembling the sharp resetting distribution, known to be optimal in this scenario. We then extend the framework by allowing agents to control not only when to reset, but also their spatial dynamics through turning actions. In this more complex setting, the agents discover strategies that adapt both resetting and turning to the properties of the environment, outperforming the proposed benchmarks. These results demonstrate how reinforcement learning can serve both as an optimization tool and a mechanism for uncovering new, interpretable strategies in stochastic search processes with resetting. 

**Abstract (ZH)**: 基于强化学习的重置策略优化研究：从布朗搜索到复杂环境中的目标搜索 

---
# MEET: A Million-Scale Dataset for Fine-Grained Geospatial Scene Classification with Zoom-Free Remote Sensing Imagery 

**Title (ZH)**: MEET：一种基于无缩放遥感图像的百万规模细粒度地理场景分类数据集 

**Authors**: Yansheng Li, Yuning Wu, Gong Cheng, Chao Tao, Bo Dang, Yu Wang, Jiahao Zhang, Chuge Zhang, Yiting Liu, Xu Tang, Jiayi Ma, Yongjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11219)  

**Abstract**: Accurate fine-grained geospatial scene classification using remote sensing imagery is essential for a wide range of applications. However, existing approaches often rely on manually zooming remote sensing images at different scales to create typical scene samples. This approach fails to adequately support the fixed-resolution image interpretation requirements in real-world scenarios. To address this limitation, we introduce the Million-scale finE-grained geospatial scEne classification dataseT (MEET), which contains over 1.03 million zoom-free remote sensing scene samples, manually annotated into 80 fine-grained categories. In MEET, each scene sample follows a scene-inscene layout, where the central scene serves as the reference, and auxiliary scenes provide crucial spatial context for finegrained classification. Moreover, to tackle the emerging challenge of scene-in-scene classification, we present the Context-Aware Transformer (CAT), a model specifically designed for this task, which adaptively fuses spatial context to accurately classify the scene samples. CAT adaptively fuses spatial context to accurately classify the scene samples by learning attentional features that capture the relationships between the center and auxiliary scenes. Based on MEET, we establish a comprehensive benchmark for fine-grained geospatial scene classification, evaluating CAT against 11 competitive baselines. The results demonstrate that CAT significantly outperforms these baselines, achieving a 1.88% higher balanced accuracy (BA) with the Swin-Large backbone, and a notable 7.87% improvement with the Swin-Huge backbone. Further experiments validate the effectiveness of each module in CAT and show the practical applicability of CAT in the urban functional zone mapping. The source code and dataset will be publicly available at this https URL. 

**Abstract (ZH)**: 大规模无缩放遥感细粒度地理场景分类数据集（MEET）：基于上下文感知变换器的细粒度地理场景分类 

---
# Multi-Stage Generative Upscaler: Reconstructing Football Broadcast Images via Diffusion Models 

**Title (ZH)**: 多阶段生成上放大模型：基于扩散模型的足球广播图像重构 

**Authors**: Luca Martini, Daniele Zolezzi, Saverio Iacono, Gianni Viardo Vercelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.11181)  

**Abstract**: The reconstruction of low-resolution football broadcast images presents a significant challenge in sports broadcasting, where detailed visuals are essential for analysis and audience engagement. This study introduces a multi-stage generative upscaling framework leveraging Diffusion Models to enhance degraded images, transforming inputs as small as $64 \times 64$ pixels into high-fidelity $1024 \times 1024$ outputs. By integrating an image-to-image pipeline, ControlNet conditioning, and LoRA fine-tuning, our approach surpasses traditional upscaling methods in restoring intricate textures and domain-specific elements such as player details and jersey logos. The custom LoRA is trained on a custom football dataset, ensuring adaptability to sports broadcast needs. Experimental results demonstrate substantial improvements over conventional models, with ControlNet refining fine details and LoRA enhancing task-specific elements. These findings highlight the potential of diffusion-based image reconstruction in sports media, paving the way for future applications in automated video enhancement and real-time sports analytics. 

**Abstract (ZH)**: 低分辨率足球直播图像的重建在体育广播中是一项重要的挑战，详细的视觉内容对于分析和观众参与至关重要。本研究提出了一种基于扩散模型的多阶段生成放大框架，通过提升降级图像，将输入的最小尺寸从64×64像素转换为高质量的1024×1024输出。通过整合图像到图像的处理管道、ControlNet条件控制和LoRA微调，我们的方法在恢复复杂的纹理和特定领域的元素（如球员细节和球衣商标）方面超过了传统的放大方法。定制的LoRA在定制的足球数据集上进行训练，确保适应体育广播的需求。实验结果表明，与传统模型相比，我们的方法取得了显著的改进，ControlNet细化了细部细节，而LoRA增强了特定任务的元素。这些发现突显了基于扩散模型的图像重建在体育媒体中的潜力，为未来的自动视频增强和实时体育分析应用铺平了道路。 

---
# Zero-TIG: Temporal Consistency-Aware Zero-Shot Illumination-Guided Low-light Video Enhancement 

**Title (ZH)**: Zero-TIG：面向时间一致性约束的零-shot 背景光引导低光视频增强 

**Authors**: Yini Li, Nantheera Anantrasirichai  

**Link**: [PDF](https://arxiv.org/pdf/2503.11175)  

**Abstract**: Low-light and underwater videos suffer from poor visibility, low contrast, and high noise, necessitating enhancements in visual quality. However, existing approaches typically rely on paired ground truth, which limits their practicality and often fails to maintain temporal consistency. To overcome these obstacles, this paper introduces a novel zero-shot learning approach named Zero-TIG, leveraging the Retinex theory and optical flow techniques. The proposed network consists of an enhancement module and a temporal feedback module. The enhancement module comprises three subnetworks: low-light image denoising, illumination estimation, and reflection denoising. The temporal enhancement module ensures temporal consistency by incorporating histogram equalization, optical flow computation, and image warping to align the enhanced previous frame with the current frame, thereby maintaining continuity. Additionally, we address color distortion in underwater data by adaptively balancing RGB channels. The experimental results demonstrate that our method achieves low-light video enhancement without the need for paired training data, making it a promising and applicable method for real-world scenario enhancement. 

**Abstract (ZH)**: 低光照和水下视频由于能见度低、对比度低和噪声高，需要在视觉质量上进行增强。为克服现有方法对配对 ground truth 的依赖和时间一致性问题，本文提出了一种名为 Zero-TIG 的零样本学习方法，利用 Retinex 理论和光学流技术。所提出的网络包括增强模块和时间反馈模块。增强模块由三个子网络组成：低光照图像去噪、照度估计和反射去噪。时间增强模块通过引入直方图均衡化、光学流计算和图像变形，确保增强的前一帧与当前帧对齐，从而保持连续性。此外，本文通过自适应平衡 RGB 通道解决了水下数据的色彩失真问题。实验结果表明，该方法能够在无需配对训练数据的情况下实现低光照视频增强，具有在实际场景中增强应用的潜力。 

---
# Neurons: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction 

**Title (ZH)**: 神经元：模拟人类视觉皮层提高fMRI到视频重建的 fidelity 和可解释性 

**Authors**: Haonan Wang, Qixiang Zhang, Lehan Wang, Xuanqi Huang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.11167)  

**Abstract**: Decoding visual stimuli from neural activity is essential for understanding the human brain. While fMRI methods have successfully reconstructed static images, fMRI-to-video reconstruction faces challenges due to the need for capturing spatiotemporal dynamics like motion and scene transitions. Recent approaches have improved semantic and perceptual alignment but struggle to integrate coarse fMRI data with detailed visual features. Inspired by the hierarchical organization of the visual system, we propose NEURONS, a novel framework that decouples learning into four correlated sub-tasks: key object segmentation, concept recognition, scene description, and blurry video reconstruction. This approach simulates the visual cortex's functional specialization, allowing the model to capture diverse video content. In the inference stage, NEURONS generates robust conditioning signals for a pre-trained text-to-video diffusion model to reconstruct the videos. Extensive experiments demonstrate that NEURONS outperforms state-of-the-art baselines, achieving solid improvements in video consistency (26.6%) and semantic-level accuracy (19.1%). Notably, NEURONS shows a strong functional correlation with the visual cortex, highlighting its potential for brain-computer interfaces and clinical applications. Code and model weights will be available at: this https URL. 

**Abstract (ZH)**: 从神经活动解码视觉刺激是理解人类大脑的关键。尽管fMRI方法成功地重建了静态图像，但由于需要捕获如运动和场景转换的时空动态，fMRI到视频的重建面临着挑战。近期的方法在语义和感知一致性方面取得了进步，但仍难以将粗粒度的fMRI数据与详细的视觉特征整合。受视觉系统层次组织的启发，我们提出了一种名为NEURONS的新框架，将学习任务分解为四个相关子任务：关键对象分割、概念识别、场景描述和模糊视频重建。该方法模拟了视觉皮层的功能专业化，使模型能够捕捉多种视频内容。在推理阶段，NEURONS为预训练的文本到视频扩散模型生成稳健的条件信号，以重建视频。广泛实验表明，NEURONS优于现有 baseline，视频一致性提高26.6%，语义水平准确性提高19.1%。值得注意的是，NEURONS与视觉皮层的功能相关性较强，突显了其在脑机接口和临床应用中的潜力。代码和模型权重将在以下网址获取：this https URL。 

---
# Direction-Aware Diagonal Autoregressive Image Generation 

**Title (ZH)**: 方向意识的对角自回归图像生成 

**Authors**: Yijia Xu, Jianzhong Ju, Jian Luan, Jinshi Cui  

**Link**: [PDF](https://arxiv.org/pdf/2503.11129)  

**Abstract**: The raster-ordered image token sequence exhibits a significant Euclidean distance between index-adjacent tokens at line breaks, making it unsuitable for autoregressive generation. To address this issue, this paper proposes Direction-Aware Diagonal Autoregressive Image Generation (DAR) method, which generates image tokens following a diagonal scanning order. The proposed diagonal scanning order ensures that tokens with adjacent indices remain in close proximity while enabling causal attention to gather information from a broader range of directions. Additionally, two direction-aware modules: 4D-RoPE and direction embeddings are introduced, enhancing the model's capability to handle frequent changes in generation direction. To leverage the representational capacity of the image tokenizer, we use its codebook as the image token embeddings. We propose models of varying scales, ranging from 485M to 2.0B. On the 256$\times$256 ImageNet benchmark, our DAR-XL (2.0B) outperforms all previous autoregressive image generators, achieving a state-of-the-art FID score of 1.37. 

**Abstract (ZH)**: 方向感知对角自回归图像生成方法（DAR） 

---
# FMNet: Frequency-Assisted Mamba-Like Linear Attention Network for Camouflaged Object Detection 

**Title (ZH)**: FMNet: 频率辅助的类似Mamba的线性注意力网络用于伪装目标检测 

**Authors**: Ming Deng, Sijin Sun, Zihao Li, Xiaochuan Hu, Xing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11030)  

**Abstract**: Camouflaged Object Detection (COD) is challenging due to the strong similarity between camouflaged objects and their surroundings, which complicates identification. Existing methods mainly rely on spatial local features, failing to capture global information, while Transformers increase computational this http URL address this, the Frequency-Assisted Mamba-Like Linear Attention Network (FMNet) is proposed, which leverages frequency-domain learning to efficiently capture global features and mitigate ambiguity between objects and the background. FMNet introduces the Multi-Scale Frequency-Assisted Mamba-Like Linear Attention (MFM) module, integrating frequency and spatial features through a multi-scale structure to handle scale variations while reducing computational complexity. Additionally, the Pyramidal Frequency Attention Extraction (PFAE) module and the Frequency Reverse Decoder (FRD) enhance semantics and reconstruct features. Experimental results demonstrate that FMNet outperforms existing methods on multiple COD datasets, showcasing its advantages in both performance and efficiency. Code available at this https URL. 

**Abstract (ZH)**: 伪装目标检测（COD）由于伪装目标与其环境之间的强烈相似性而具有挑战性，这增加了识别的复杂性。现有方法主要依赖于空间局部特征，无法捕捉全局信息，而Transformer提高计算复杂度对此有所改善。为了解决这一问题，提出了一种频率辅助的类似Mamba的线性注意力网络（FMNet），它利用频域学习有效地捕获全局特征并减轻伪装目标与背景之间的模糊性。FMNet引入了多尺度频率辅助的类似Mamba的线性注意力（MFM）模块，通过多尺度结构整合频率和空间特征，以处理尺度变化并降低计算复杂度。此外，金字塔频域注意提取（PFAE）模块和频率逆向解码器（FRD）增强了语义并重构特征。实验结果表明，FMNet在多个伪装目标检测数据集上优于现有方法，展示了其在性能和效率方面的优势。代码可在以下网址获取。 

---
# Observation-Graph Interaction and Key-Detail Guidance for Vision and Language Navigation 

**Title (ZH)**: 观察图交互与关键细节指导下的视觉语言导航 

**Authors**: Yifan Xie, Binkai Ou, Fei Ma, Yaohua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11006)  

**Abstract**: Vision and Language Navigation (VLN) requires an agent to navigate through environments following natural language instructions. However, existing methods often struggle with effectively integrating visual observations and instruction details during navigation, leading to suboptimal path planning and limited success rates. In this paper, we propose OIKG (Observation-graph Interaction and Key-detail Guidance), a novel framework that addresses these limitations through two key components: (1) an observation-graph interaction module that decouples angular and visual information while strengthening edge representations in the navigation space, and (2) a key-detail guidance module that dynamically extracts and utilizes fine-grained location and object information from instructions. By enabling more precise cross-modal alignment and dynamic instruction interpretation, our approach significantly improves the agent's ability to follow complex navigation instructions. Extensive experiments on the R2R and RxR datasets demonstrate that OIKG achieves state-of-the-art performance across multiple evaluation metrics, validating the effectiveness of our method in enhancing navigation precision through better observation-instruction alignment. 

**Abstract (ZH)**: 基于视觉与语言的导航（VLN）要求代理遵循自然语言指令在环境中导航。然而，现有方法在导航过程中往往难以有效整合视觉观察和指令细节，导致路径规划效果不佳，成功率有限。本文提出了一种新颖的框架OIKG（Observation-graph Interaction and Key-detail Guidance），通过两个关键组件解决这些局限性：（1）一个观测图交互模块，解耦角度和视觉信息的同时增强导航空间中的边表示；（2）一个关键细节引导模块，动态提取和利用指令中的细粒度位置和对象信息。通过实现更精确的跨模态对齐和动态指令解释，我们的方法显著提高了代理遵循复杂导航指令的能力。在R2R和RxR数据集上的广泛实验表明，OIKG在多个评估指标上达到了最佳性能，验证了我们方法在通过更好观测-指令对齐提升导航精度方面的有效性。 

---
# JPEG Compliant Compression for Both Human and Machine, A Report 

**Title (ZH)**: 符合人类和机器的JPEG规范压缩技术报告 

**Authors**: Linfeng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.10912)  

**Abstract**: Deep Neural Networks (DNNs) have become an integral part of our daily lives, especially in vision-related applications. However, the conventional lossy image compression algorithms are primarily designed for the Human Vision System (HVS), which can non-trivially compromise the DNNs' validation accuracy after compression, as noted in \cite{liu2018deepn}. Thus developing an image compression algorithm for both human and machine (DNNs) is on the horizon.
To address the challenge mentioned above, in this paper, we first formulate the image compression as a multi-objective optimization problem which take both human and machine prespectives into account, then we solve it by linear combination, and proposed a novel distortion measure for both human and machine, dubbed Human and Machine-Oriented Error (HMOE). After that, we develop Human And Machine Oriented Soft Decision Quantization (HMOSDQ) based on HMOE, a lossy image compression algorithm for both human and machine (DNNs), and fully complied with JPEG format. In order to evaluate the performance of HMOSDQ, finally we conduct the experiments for two pre-trained well-known DNN-based image classifiers named Alexnet \cite{Alexnet} and VGG-16 \cite{simonyan2014VGG} on two subsets of the ImageNet \cite{deng2009imagenet} validation set: one subset included images with shorter side in the range of 496 to 512, while the other included images with shorter side in the range of 376 to 384. Our results demonstrate that HMOSDQ outperforms the default JPEG algorithm in terms of rate-accuracy and rate-distortion performance. For the Alexnet comparing with the default JPEG algorithm, HMOSDQ can improve the validation accuracy by more than $0.81\%$ at $0.61$ BPP, or equivalently reduce the compression rate of default JPEG by $9.6\times$ while maintaining the same validation accuracy. 

**Abstract (ZH)**: 深度神经网络（DNNs）已成为我们日常生活中不可或缺的一部分，尤其是在视觉相关应用中。然而，传统的有损图像压缩算法主要是针对人类视觉系统（HVS）设计的，这可能会在压缩后显著降低DNNs的验证准确性，如文献\[liu2018deepn\]中所述。因此，开发同时适用于人类和机器（DNNs）的图像压缩算法迫在眉睫。
为了解决上述挑战，本文首先将图像压缩问题形式化为一个多目标优化问题，同时考虑了人类和机器的视角，然后通过线性组合解决这个问题，并提出了一种适用于人类和机器的新失真度量，称为人类和机器导向误差（HMOE）。之后，基于HMOE开发了人类和机器导向软决策量化（HMOSDQ），这是一种同时适用于人类和机器（DNNs）的有损图像压缩算法，完全符合JPEG格式。为了评估HMOSDQ的性能，我们对两个预先训练好的基于DNN的图像分类器Alexnet\[Alexnet\]和VGG-16\[simonyan2014VGG\]在ImageNet\[deng2009imagenet\]验证集的两个子集中进行了实验：一个子集包含较短边在496到512范围内的图像，另一个子集包含较短边在376到384范围内的图像。实验结果表明，HMOSDQ在速率-准确性性能和速率-失真性能方面优于标准JPEG算法。与标准JPEG算法相比，在0.61 BPP时，HMOSDQ可以将验证准确性提高超过0.81%，或者等效地将标准JPEG的压缩率减少9.6倍，同时保持相同的验证准确性。 

---
# TacticExpert: Spatial-Temporal Graph Language Model for Basketball Tactics 

**Title (ZH)**: TacticExpert: 空间-时间图语言模型在篮球战术中的应用 

**Authors**: Xu Lingrui, Liu Mandi, Zhang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2503.10722)  

**Abstract**: The core challenge in basketball tactic modeling lies in efficiently extracting complex spatial-temporal dependencies from historical data and accurately predicting various in-game events. Existing state-of-the-art (SOTA) models, primarily based on graph neural networks (GNNs), encounter difficulties in capturing long-term, long-distance, and fine-grained interactions among heterogeneous player nodes, as well as in recognizing interaction patterns. Additionally, they exhibit limited generalization to untrained downstream tasks and zero-shot scenarios. In this work, we propose a Spatial-Temporal Propagation Symmetry-Aware Graph Transformer for fine-grained game modeling. This architecture explicitly captures delay effects in the spatial space to enhance player node representations across discrete-time slices, employing symmetry-invariant priors to guide the attention mechanism. We also introduce an efficient contrastive learning strategy to train a Mixture of Tactics Experts module, facilitating differentiated modeling of offensive tactics. By integrating dense training with sparse inference, we achieve a 2.4x improvement in model efficiency. Moreover, the incorporation of Lightweight Graph Grounding for Large Language Models enables robust performance in open-ended downstream tasks and zero-shot scenarios, including novel teams or players. The proposed model, TacticExpert, delineates a vertically integrated large model framework for basketball, unifying pretraining across multiple datasets and downstream prediction tasks. Fine-grained modeling modules significantly enhance spatial-temporal representations, and visualization analyzes confirm the strong interpretability of the model. 

**Abstract (ZH)**: 基于时空传播对称性的图变压器在篮球战术建模中的细粒度游戏建模 

---
# Team NYCU at Defactify4: Robust Detection and Source Identification of AI-Generated Images Using CNN and CLIP-Based Models 

**Title (ZH)**: NYCU团队在Defactify4中的研究：使用CNN和CLIP基模型的AI生成图像的稳健检测及其源识别 

**Authors**: Tsan-Tsung Yang, I-Wei Chen, Kuan-Ting Chen, Shang-Hsuan Chiang, Wen-Chih Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.10718)  

**Abstract**: With the rapid advancement of generative AI, AI-generated images have become increasingly realistic, raising concerns about creativity, misinformation, and content authenticity. Detecting such images and identifying their source models has become a critical challenge in ensuring the integrity of digital media. This paper tackles the detection of AI-generated images and identifying their source models using CNN and CLIP-ViT classifiers. For the CNN-based classifier, we leverage EfficientNet-B0 as the backbone and feed with RGB channels, frequency features, and reconstruction errors, while for CLIP-ViT, we adopt a pretrained CLIP image encoder to extract image features and SVM to perform classification. Evaluated on the Defactify 4 dataset, our methods demonstrate strong performance in both tasks, with CLIP-ViT showing superior robustness to image perturbations. Compared to baselines like AEROBLADE and OCC-CLIP, our approach achieves competitive results. Notably, our method ranked Top-3 overall in the Defactify 4 competition, highlighting its effectiveness and generalizability. All of our implementations can be found in this https URL 

**Abstract (ZH)**: 随着生成式AI的迅速发展，AI生成的图像日益逼真，引发了关于创造力、错误信息和内容真实性的问题。检测这些图像并识别其源模型已成为确保数字媒体完整性的关键挑战。本文使用CNN和CLIP-ViT分类器来解决AI生成图像的检测和源模型识别问题。对于基于CNN的分类器，我们采用EfficientNet-B0作为主干，并输入RGB通道、频率特征和重构误差；对于CLIP-ViT，我们采用预训练的CLIP图像编码器提取图像特征，并使用SVM进行分类。在Defactify 4数据集上的评估表明，我们的方法在这两项任务上都表现出强大性能，CLIP-ViT对图像扰动具有更高的鲁棒性。与AEROBLADE和OCC-CLIP等基线方法相比，我们的方法取得了竞争力的结果。值得注意的是，在Defactify 4竞赛中，我们的方法总体排名前三，突显了其有效性与泛化能力。所有我们的实现都可以在以下链接找到：https://github.com/alibaba/Qwen-Languages-Assistant 

---
# Deep Learning-Based Automated Workflow for Accurate Segmentation and Measurement of Abdominal Organs in CT Scans 

**Title (ZH)**: 基于深度学习的自动化工作流用于CT扫描中腹部器官的准确分割与测量 

**Authors**: Praveen Shastry, Ashok Sharma, Kavya Mohan, Naveen Kumarasami, Anandakumar D, Mounigasri M, Keerthana R, Kishore Prasath Venkatesh, Bargava Subramanian, Kalyan Sivasailam  

**Link**: [PDF](https://arxiv.org/pdf/2503.10717)  

**Abstract**: Background: Automated analysis of CT scans for abdominal organ measurement is crucial for improving diagnostic efficiency and reducing inter-observer variability. Manual segmentation and measurement of organs such as the kidneys, liver, spleen, and prostate are time-consuming and subject to inconsistency, underscoring the need for automated approaches.
Purpose: The purpose of this study is to develop and validate an automated workflow for the segmentation and measurement of abdominal organs in CT scans using advanced deep learning models, in order to improve accuracy, reliability, and efficiency in clinical evaluations.
Methods: The proposed workflow combines nnU-Net, U-Net++ for organ segmentation, followed by a 3D RCNN model for measuring organ volumes and dimensions. The models were trained and evaluated on CT datasets with metrics such as precision, recall, and Mean Squared Error (MSE) to assess performance. Segmentation quality was verified for its adaptability to variations in patient anatomy and scanner settings.
Results: The developed workflow achieved high precision and recall values, exceeding 95 for all targeted organs. The Mean Squared Error (MSE) values were low, indicating a high level of consistency between predicted and ground truth measurements. The segmentation and measurement pipeline demonstrated robust performance, providing accurate delineation and quantification of the kidneys, liver, spleen, and prostate.
Conclusion: The proposed approach offers an automated, efficient, and reliable solution for abdominal organ measurement in CT scans. By significantly reducing manual intervention, this workflow enhances measurement accuracy and consistency, with potential for widespread clinical implementation. Future work will focus on expanding the approach to other organs and addressing complex pathological cases. 

**Abstract (ZH)**: 背景：自动分析CT扫描以测量腹部器官对于提高诊断效率和减少观察者间变异至关重要。手动分割和测量肾脏、肝脏、脾脏和前列腺等器官耗时且不一致，凸显了需要自动方法的需求。
目的：本研究旨在开发并验证一种基于先进深度学习模型的自动化工作流，用于CT扫描中腹部器官的分割和测量，以提高临床评估的准确度、可靠性和效率。
方法：所提出的工作流结合了nnU-Net和U-Net++进行器官分割，随后使用3D RCNN模型测量器官体积和尺寸。通过精度、召回率和均方误差（MSE）等指标对模型进行训练和评估，以评估其性能。验证分割质量以适应不同患者解剖结构和扫描设置的变化。
结果：开发的工作流在所有目标器官上的精度和召回率均超过95%。均方误差（MSE）值较低，表明预测值与真实值之间的测量一致性较高。分割和测量管道表现出稳健的性能，提供了对肾脏、肝脏、脾脏和前列腺的准确勾勒和量化。
结论：所提出的方法提供了一种自动化、高效且可靠的解决方案，用于CT扫描中的腹部器官测量。通过显著减少手动干预，该工作流提高了测量准确度和一致性，并有望在临床中广泛应用。未来工作将集中在将该方法扩展到其他器官以及处理复杂的病理病例上。 

---
# End-to-end Learning of Sparse Interventions on Activations to Steer Generation 

**Title (ZH)**: 端到端学习稀疏干预以引导生成 

**Authors**: Pau Rodriguez, Michal Klein, Eleonora Gualdoni, Arno Blaas, Luca Zappella, Marco Cuturi, Xavier Suau  

**Link**: [PDF](https://arxiv.org/pdf/2503.10679)  

**Abstract**: The growing use of generative models in daily life calls for efficient mechanisms to control their generation, to e.g., produce safe content or provide users with tools to explore style changes. Ideally, such mechanisms should be cheap, both at train and inference time, while preserving output quality. Recent research has shown that such mechanisms can be obtained by intervening exclusively on model activations, with the goal of correcting distributional differences between activations seen when using prompts from a source vs. a target set (e.g., toxic and non-toxic sentences). While cheap, these fast methods are inherently crude: their maps are tuned locally, not accounting for their impact on downstream layers, resulting in interventions that cause unintended shifts when used out-of-sample. We propose in this work linear end-to-end activation steering (LinEAS), an approach trained with a global loss that accounts simultaneously for all layerwise distributional shifts. In addition to being more robust, the loss used to train LinEAS can be regularized with sparsifying norms, which can automatically carry out neuron and layer selection. Empirically, LinEAS only requires a handful of samples to be effective, and beats similar baselines on toxicity mitigation, while performing on par with far more involved finetuning approaches. We show that LinEAS interventions can be composed, study the impact of sparsity on their performance, and showcase applications in text-to-image diffusions. 

**Abstract (ZH)**: 生成模型在日常生活中的 Growing 使用呼唤高效的生成控制机制：从源集到目标集（如有毒与非有毒句子）之间的激活分布差异校正 

---
# Text-to-3D Generation using Jensen-Shannon Score Distillation 

**Title (ZH)**: 基于杰森-香农分数蒸馏的文本到3D生成 

**Authors**: Khoi Do, Binh-Son Hua  

**Link**: [PDF](https://arxiv.org/pdf/2503.10660)  

**Abstract**: Score distillation sampling is an effective technique to generate 3D models from text prompts, utilizing pre-trained large-scale text-to-image diffusion models as guidance. However, the produced 3D assets tend to be over-saturating, over-smoothing, with limited diversity. These issues are results from a reverse Kullback-Leibler (KL) divergence objective, which makes the optimization unstable and results in mode-seeking behavior. In this paper, we derive a bounded score distillation objective based on Jensen-Shannon divergence (JSD), which stabilizes the optimization process and produces high-quality 3D generation. JSD can match well generated and target distribution, therefore mitigating mode seeking. We provide a practical implementation of JSD by utilizing the theory of generative adversarial networks to define an approximate objective function for the generator, assuming the discriminator is well trained. By assuming the discriminator following a log-odds classifier, we propose a minority sampling algorithm to estimate the gradients of our proposed objective, providing a practical implementation for JSD. We conduct both theoretical and empirical studies to validate our method. Experimental results on T3Bench demonstrate that our method can produce high-quality and diversified 3D assets. 

**Abstract (ZH)**: 基于Jensen-Shannon散度的得分蒸馏采样：稳定3D生成并提高质量 

---
