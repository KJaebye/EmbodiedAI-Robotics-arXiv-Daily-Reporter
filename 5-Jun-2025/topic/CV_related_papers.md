# Confidence-Guided Human-AI Collaboration: Reinforcement Learning with Distributional Proxy Value Propagation for Autonomous Driving 

**Title (ZH)**: 基于信心引导的人机协作：分布代理价值传播的强化学习在自动驾驶中的应用 

**Authors**: Li Zeqiao, Wang Yijing, Wang Haoyu, Li Zheng, Li Peng, Zuo zhiqiang, Hu Chuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.03568)  

**Abstract**: Autonomous driving promises significant advancements in mobility, road safety and traffic efficiency, yet reinforcement learning and imitation learning face safe-exploration and distribution-shift challenges. Although human-AI collaboration alleviates these issues, it often relies heavily on extensive human intervention, which increases costs and reduces efficiency. This paper develops a confidence-guided human-AI collaboration (C-HAC) strategy to overcome these limitations. First, C-HAC employs a distributional proxy value propagation method within the distributional soft actor-critic (DSAC) framework. By leveraging return distributions to represent human intentions C-HAC achieves rapid and stable learning of human-guided policies with minimal human interaction. Subsequently, a shared control mechanism is activated to integrate the learned human-guided policy with a self-learning policy that maximizes cumulative rewards. This enables the agent to explore independently and continuously enhance its performance beyond human guidance. Finally, a policy confidence evaluation algorithm capitalizes on DSAC's return distribution networks to facilitate dynamic switching between human-guided and self-learning policies via a confidence-based intervention function. This ensures the agent can pursue optimal policies while maintaining safety and performance guarantees. Extensive experiments across diverse driving scenarios reveal that C-HAC significantly outperforms conventional methods in terms of safety, efficiency, and overall performance, achieving state-of-the-art results. The effectiveness of the proposed method is further validated through real-world road tests in complex traffic conditions. The videos and code are available at: this https URL. 

**Abstract (ZH)**: 自主驾驶有望在移动性、道路安全和交通效率方面带来重大进展，然而强化学习和模仿学习面临着安全探索和分布转移的挑战。尽管人机协作可以减轻这些问题，但往往需要大量的人工干预，增加了成本并降低了效率。本文提出了一种基于信心引导的人机协作（C-HAC）策略以克服这些限制。首先，C-HAC 在分布软演员-评论家 (DSAC) 框架内采用了分布代理价值传播方法。通过利用回报分布来表示人类意图，C-HAC 实现了在最少人类干预下的人类指导政策的快速稳定学习。随后，激活了一种共决策机制，将所学的人类指导策略与最大化累积奖励的自我学习策略相结合。这使智能体能够独立探索并持续提升其性能，超越人类指导。最后，基于 DSAC 的回报分布网络开发了一种策略信心评估算法，通过基于信心的干预函数动态切换人类指导和自我学习策略。这确保智能体能够在保证安全和性能的同时追求最优策略。跨多种驾驶场景的广泛实验表明，C-HAC 在安全、效率和总体性能方面显著优于传统方法，达到顶级性能。所提方法的有效性通过在复杂交通条件下的实车测试进一步验证。相关视频和代码可在以下链接获取：this https URL。 

---
# Zero-Shot Temporal Interaction Localization for Egocentric Videos 

**Title (ZH)**: 零样本自视点视频时空交互定位 

**Authors**: Erhang Zhang, Junyi Ma, Yin-Dong Zheng, Yixuan Zhou, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03662)  

**Abstract**: Locating human-object interaction (HOI) actions within video serves as the foundation for multiple downstream tasks, such as human behavior analysis and human-robot skill transfer. Current temporal action localization methods typically rely on annotated action and object categories of interactions for optimization, which leads to domain bias and low deployment efficiency. Although some recent works have achieved zero-shot temporal action localization (ZS-TAL) with large vision-language models (VLMs), their coarse-grained estimations and open-loop pipelines hinder further performance improvements for temporal interaction localization (TIL). To address these issues, we propose a novel zero-shot TIL approach dubbed EgoLoc to locate the timings of grasp actions for human-object interaction in egocentric videos. EgoLoc introduces a self-adaptive sampling strategy to generate reasonable visual prompts for VLM reasoning. By absorbing both 2D and 3D observations, it directly samples high-quality initial guesses around the possible contact/separation timestamps of HOI according to 3D hand velocities, leading to high inference accuracy and efficiency. In addition, EgoLoc generates closed-loop feedback from visual and dynamic cues to further refine the localization results. Comprehensive experiments on the publicly available dataset and our newly proposed benchmark demonstrate that EgoLoc achieves better temporal interaction localization for egocentric videos compared to state-of-the-art baselines. We will release our code and relevant data as open-source at this https URL. 

**Abstract (ZH)**: 基于视频的人-物交互（HOI）动作定位为多个下游任务（如人类行为分析和人机技能转移）奠定了基础。当前的时间动作定位方法通常依赖于交互的标注动作和对象类别进行优化，这导致了领域偏差和低部署效率。尽管一些近期工作使用大型视觉-语言模型（VLMs）实现了零样本时间动作定位（ZS-TAL），但它们粗粒度的估计和开环管道阻碍了进一步的时间交互定位（TIL）性能改进。为了解决这些问题，我们提出了一种名为EgoLoc的新型零样本TIL方法，用于在第一人称视频中定位人-物交互的抓取动作时间。EgoLoc引入了一种自我适应的采样策略，以生成合理的视觉提示供VLM推理。通过结合2D和3D观察，它根据3D手速度在可能的接触/分离时间戳周围直接抽样高质量的初始猜测，从而提高了推理的准确性和效率。此外，EgoLoc从视觉和动态线索中生成闭环反馈以进一步细化定位结果。在公开数据集和我们新提出的基准上的全面实验表明，EgoLoc在第一人称视频的时间交互定位方面优于最新的基线方法。我们将在该网址发布我们的代码和相关数据：this https URL。 

---
# SplArt: Articulation Estimation and Part-Level Reconstruction with 3D Gaussian Splatting 

**Title (ZH)**: SplArt: 基于3D 高斯点云的艺术关节估计与部分级别重建 

**Authors**: Shengjie Lin, Jiading Fang, Muhammad Zubair Irshad, Vitor Campagnolo Guizilini, Rares Andrei Ambrus, Greg Shakhnarovich, Matthew R. Walter  

**Link**: [PDF](https://arxiv.org/pdf/2506.03594)  

**Abstract**: Reconstructing articulated objects prevalent in daily environments is crucial for applications in augmented/virtual reality and robotics. However, existing methods face scalability limitations (requiring 3D supervision or costly annotations), robustness issues (being susceptible to local optima), and rendering shortcomings (lacking speed or photorealism). We introduce SplArt, a self-supervised, category-agnostic framework that leverages 3D Gaussian Splatting (3DGS) to reconstruct articulated objects and infer kinematics from two sets of posed RGB images captured at different articulation states, enabling real-time photorealistic rendering for novel viewpoints and articulations. SplArt augments 3DGS with a differentiable mobility parameter per Gaussian, achieving refined part segmentation. A multi-stage optimization strategy is employed to progressively handle reconstruction, part segmentation, and articulation estimation, significantly enhancing robustness and accuracy. SplArt exploits geometric self-supervision, effectively addressing challenging scenarios without requiring 3D annotations or category-specific priors. Evaluations on established and newly proposed benchmarks, along with applications to real-world scenarios using a handheld RGB camera, demonstrate SplArt's state-of-the-art performance and real-world practicality. Code is publicly available at this https URL. 

**Abstract (ZH)**: 基于3D高斯束的自监督骨架物体重建框架SplArt 

---
# Person Re-Identification System at Semantic Level based on Pedestrian Attributes Ontology 

**Title (ZH)**: 基于行人属性本体的语义级别行人重识别系统 

**Authors**: Ngoc Q. Ly, Hieu N. M. Cao, Thi T. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.04143)  

**Abstract**: Person Re-Identification (Re-ID) is a very important task in video surveillance systems such as tracking people, finding people in public places, or analysing customer behavior in supermarkets. Although there have been many works to solve this problem, there are still remaining challenges such as large-scale datasets, imbalanced data, viewpoint, fine grained data (attributes), the Local Features are not employed at semantic level in online stage of Re-ID task, furthermore, the imbalanced data problem of attributes are not taken into consideration. This paper has proposed a Unified Re-ID system consisted of three main modules such as Pedestrian Attribute Ontology (PAO), Local Multi-task DCNN (Local MDCNN), Imbalance Data Solver (IDS). The new main point of our Re-ID system is the power of mutual support of PAO, Local MDCNN and IDS to exploit the inner-group correlations of attributes and pre-filter the mismatch candidates from Gallery set based on semantic information as Fashion Attributes and Facial Attributes, to solve the imbalanced data of attributes without adjusting network architecture and data augmentation. We experimented on the well-known Market1501 dataset. The experimental results have shown the effectiveness of our Re-ID system and it could achieve the higher performance on Market1501 dataset in comparison to some state-of-the-art Re-ID methods. 

**Abstract (ZH)**: 基于统一框架的行人再识别系统：融合行人属性本体、局部多任务DCNN和不平衡数据解决方法 

---
# Recent Advances in Medical Image Classification 

**Title (ZH)**: 最近在医学图像分类领域的进展 

**Authors**: Loan Dao, Ngoc Quoc Ly  

**Link**: [PDF](https://arxiv.org/pdf/2506.04129)  

**Abstract**: Medical image classification is crucial for diagnosis and treatment, benefiting significantly from advancements in artificial intelligence. The paper reviews recent progress in the field, focusing on three levels of solutions: basic, specific, and applied. It highlights advances in traditional methods using deep learning models like Convolutional Neural Networks and Vision Transformers, as well as state-of-the-art approaches with Vision Language Models. These models tackle the issue of limited labeled data, and enhance and explain predictive results through Explainable Artificial Intelligence. 

**Abstract (ZH)**: 医学图像分类对于诊断和治疗至关重要，受益于人工智能的发展。本文回顾了该领域的最新进展，重点关注三个层面的解决方案：基础层、具体层和应用层。文章 Highlights 传统方法在使用深度学习模型如卷积神经网络和视觉变换器方面的进展，以及使用视觉语言模型的先进方法。这些模型解决了标注数据有限的问题，并通过可解释的人工智能增强和解释预测结果。 

---
# A Comprehensive Study on Medical Image Segmentation using Deep Neural Networks 

**Title (ZH)**: 基于深度神经网络的医学图像分割综述研究 

**Authors**: Loan Dao, Ngoc Quoc Ly  

**Link**: [PDF](https://arxiv.org/pdf/2506.04121)  

**Abstract**: Over the past decade, Medical Image Segmentation (MIS) using Deep Neural Networks (DNNs) has achieved significant performance improvements and holds great promise for future developments. This paper presents a comprehensive study on MIS based on DNNs. Intelligent Vision Systems are often evaluated based on their output levels, such as Data, Information, Knowledge, Intelligence, and Wisdom (DIKIW),and the state-of-the-art solutions in MIS at these levels are the focus of research. Additionally, Explainable Artificial Intelligence (XAI) has become an important research direction, as it aims to uncover the "black box" nature of previous DNN architectures to meet the requirements of transparency and ethics. The study emphasizes the importance of MIS in disease diagnosis and early detection, particularly for increasing the survival rate of cancer patients through timely diagnosis. XAI and early prediction are considered two important steps in the journey from "intelligence" to "wisdom." Additionally, the paper addresses existing challenges and proposes potential solutions to enhance the efficiency of implementing DNN-based MIS. 

**Abstract (ZH)**: 过去十年，基于深度神经网络的医学图像分割（MIS）取得了显著的性能提升，并为未来的发展带来了巨大潜力。本文对基于深度神经网络的医学图像分割进行了全面研究。智能视觉系统通常根据其输出水平，如数据、信息、知识、智能和智慧（DIKIW）进行评估，这些水平上的最先进解决方案是本研究的重点。此外，可解释的人工智能（XAI）已成为一个重要的研究方向，因为它旨在揭开之前深度神经网络架构的“黑匣子”性质，以满足透明性和伦理要求。研究强调了医学图像分割在疾病诊断和早期检测中的重要性，特别是通过及时诊断提高癌症患者的生存率。可解释性人工智能和早期预测被认为是“智能”到“智慧”旅程中的两个重要步骤。此外，本文还讨论了现有挑战，并提出了潜在解决方案，以提高基于深度神经网络的医学图像分割的效率。 

---
# JointSplat: Probabilistic Joint Flow-Depth Optimization for Sparse-View Gaussian Splatting 

**Title (ZH)**: JointSplat: 概率联合流-深度优化Sparse-视图ガウシアンスプラッティング 

**Authors**: Yang Xiao, Guoan Xu, Qiang Wu, Wenjing Jia  

**Link**: [PDF](https://arxiv.org/pdf/2506.03872)  

**Abstract**: Reconstructing 3D scenes from sparse viewpoints is a long-standing challenge with wide applications. Recent advances in feed-forward 3D Gaussian sparse-view reconstruction methods provide an efficient solution for real-time novel view synthesis by leveraging geometric priors learned from large-scale multi-view datasets and computing 3D Gaussian centers via back-projection. Despite offering strong geometric cues, both feed-forward multi-view depth estimation and flow-depth joint estimation face key limitations: the former suffers from mislocation and artifact issues in low-texture or repetitive regions, while the latter is prone to local noise and global inconsistency due to unreliable matches when ground-truth flow supervision is unavailable. To overcome this, we propose JointSplat, a unified framework that leverages the complementarity between optical flow and depth via a novel probabilistic optimization mechanism. Specifically, this pixel-level mechanism scales the information fusion between depth and flow based on the matching probability of optical flow during training. Building upon the above mechanism, we further propose a novel multi-view depth-consistency loss to leverage the reliability of supervision while suppressing misleading gradients in uncertain areas. Evaluated on RealEstate10K and ACID, JointSplat consistently outperforms state-of-the-art (SOTA) methods, demonstrating the effectiveness and robustness of our proposed probabilistic joint flow-depth optimization approach for high-fidelity sparse-view 3D reconstruction. 

**Abstract (ZH)**: 从稀疏视角重建3D场景是长期存在的挑战，具有广泛的应用前景。近年来，基于前向传递的3D高斯稀疏视点重建方法通过利用大规模多视角数据集中学到的几何先验，并通过反投影计算3D高斯中心，提供了实时新颖视角合成的有效解决方案。尽管提供了强大的几何线索，但前向多视角深度估计和联合流-深度估计仍面临关键限制：前者在低纹理或重复区域遭受位置错误和伪影问题，而后者由于当缺乏地面真实流监督时难以可靠的匹配而导致局部噪声和全局不一致。为了克服这些限制，我们提出了一种联合点积（JointSplat）统一框架，通过一种新颖的概率优化机制利用流和深度之间的互补性。具体而言，在训练过程中，这种亚像素机制基于光学流的匹配概率来缩放深度和流之间的信息融合。在此机制的基础上，我们进一步提出了一种新颖的多视角一致深度损失，以利用监督的可靠性并抑制不确定区域中的误导梯度。在RealEstate10K和ACID上进行评估，JointSplat始终优于现有最佳方法，证明了我们提出的概率联合流-深度优化方法在高保真稀疏视点3D重建中的有效性和鲁棒性。 

---
# SAAT: Synergistic Alternating Aggregation Transformer for Image Super-Resolution 

**Title (ZH)**: SAAT: 协同交替聚合变换器用于图像超分辨率 

**Authors**: Jianfeng Wu, Nannan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03740)  

**Abstract**: Single image super-resolution is a well-known downstream task which aims to restore low-resolution images into high-resolution images. At present, models based on Transformers have shone brightly in the field of super-resolution due to their ability to capture long-term dependencies in information. However, current methods typically compute self-attention in nonoverlapping windows to save computational costs, and the standard self-attention computation only focuses on its results, thereby neglecting the useful information across channels and the rich spatial structural information generated in the intermediate process. Channel attention and spatial attention have, respectively, brought significant improvements to various downstream visual tasks in terms of extracting feature dependency and spatial structure relationships, but the synergistic relationship between channel and spatial attention has not been fully explored this http URL address these issues, we propose a novel model. Synergistic Alternating Aggregation Transformer (SAAT), which can better utilize the potential information of features. In SAAT, we introduce the Efficient Channel & Window Synergistic Attention Group (CWSAG) and the Spatial & Window Synergistic Attention Group (SWSAG). On the one hand, CWSAG combines efficient channel attention with shifted window attention, enhancing non-local feature fusion, and producing more visually appealing results. On the other hand, SWSAG leverages spatial attention to capture rich structured feature information, thereby enabling SAAT to more effectively extract structural this http URL experimental results and ablation studies demonstrate the effectiveness of SAAT in the field of super-resolution. SAAT achieves performance comparable to that of the state-of-the-art (SOTA) under the same quantity of parameters. 

**Abstract (ZH)**: 单图像超分辨率是一种广为人知的下游任务，旨在将低分辨率图像恢复为高分辨率图像。现有的基于Transformer的模型在超分辨率领域因其能够捕获长程依赖关系而表现出色。然而，当前的方法通常通过在非重叠窗口中计算自注意力来节约计算成本，而标准的自注意力计算仅关注其结果，从而忽视了通道间有用的信息以及中间过程中生成的丰富空间结构信息。通道注意力和空间注意力在提升各种下游视觉任务的特征依赖性和空间结构关系方面分别带来了显著的改进，但通道和空间注意力之间的协同关系尚未得到充分探索。为了应对这些挑战，我们提出了一种新颖的模型——协同交替聚集变换器（SAAT），该模型能够更好地利用特征的潜在信息。在SAAT中，我们引入了高效通道与窗口协同注意力组（CWSAG）和空间与窗口协同注意力组（SWSAG）。一方面，CWSAG结合了高效的通道注意力与移位窗口注意力，增强非局部特征融合，产生更令人满意的结果。另一方面，SWSAG利用空间注意力来捕获丰富的结构化特征信息，从而使SAAT更有效地提取结构信息。实验结果和消融研究证明了SAAT在超分辨率领域的有效性。SAAT在相同参数量的情况下，性能达到了与现有最佳方法（SOTA）相当的水平。 

---
# OSGNet @ Ego4D Episodic Memory Challenge 2025 

**Title (ZH)**: OSGNet @ Ego4D 回忆挑战2025 

**Authors**: Yisen Feng, Haoyu Zhang, Qiaohui Chu, Meng Liu, Weili Guan, Yaowei Wang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.03710)  

**Abstract**: In this report, we present our champion solutions for the three egocentric video localization tracks of the Ego4D Episodic Memory Challenge at CVPR 2025. All tracks require precise localization of the interval within an untrimmed egocentric video. Previous unified video localization approaches often rely on late fusion strategies, which tend to yield suboptimal results. To address this, we adopt an early fusion-based video localization model to tackle all three tasks, aiming to enhance localization accuracy. Ultimately, our method achieved first place in the Natural Language Queries, Goal Step, and Moment Queries tracks, demonstrating its effectiveness. Our code can be found at this https URL. 

**Abstract (ZH)**: 在Ego4D Episodic Memory Challenge 2025的三项以自我为中心的视频定位赛道中，我们呈现了我们的冠军解决方案。所有赛道均要求在未剪辑的以自我为中心的视频中精确定位时间区间。以往的统一视频定位方法通常依赖于后期融合策略，这往往会导致次优结果。为了解决这一问题，我们采用了基于早期融合的视频定位模型来应对所有三个任务，旨在提高定位准确性。最终，我们的方法在自然语言查询、目标步骤和时刻查询赛道中均获得了第一名，证明了其有效性。我们的代码可在以下链接找到：这个 https URL。 

---
# How PARTs assemble into wholes: Learning the relative composition of images 

**Title (ZH)**: How PARTs组装成 wholes: 学习图像的相对组成 

**Authors**: Melika Ayoughi, Samira Abnar, Chen Huang, Chris Sandino, Sayeri Lala, Eeshan Gunesh Dhekane, Dan Busbridge, Shuangfei Zhai, Vimal Thilak, Josh Susskind, Pascal Mettes, Paul Groth, Hanlin Goh  

**Link**: [PDF](https://arxiv.org/pdf/2506.03682)  

**Abstract**: The composition of objects and their parts, along with object-object positional relationships, provides a rich source of information for representation learning. Hence, spatial-aware pretext tasks have been actively explored in self-supervised learning. Existing works commonly start from a grid structure, where the goal of the pretext task involves predicting the absolute position index of patches within a fixed grid. However, grid-based approaches fall short of capturing the fluid and continuous nature of real-world object compositions. We introduce PART, a self-supervised learning approach that leverages continuous relative transformations between off-grid patches to overcome these limitations. By modeling how parts relate to each other in a continuous space, PART learns the relative composition of images-an off-grid structural relative positioning process that generalizes beyond occlusions and deformations. In tasks requiring precise spatial understanding such as object detection and time series prediction, PART outperforms strong grid-based methods like MAE and DropPos, while also maintaining competitive performance on global classification tasks with minimal hyperparameter tuning. By breaking free from grid constraints, PART opens up an exciting new trajectory for universal self-supervised pretraining across diverse datatypes-from natural images to EEG signals-with promising potential in video, medical imaging, and audio. 

**Abstract (ZH)**: 对象及其部件的组成和对象间的空间关系提供了丰富的信息来源，用于表示学习。因此，具有空间意识的预训练任务在自我监督学习中被积极研究。现有工作通常基于网格结构，预训练任务的目标是在固定网格内预测补丁的绝对位置索引。然而，基于网格的方法难以捕捉现实世界中对象组成流体和连续的特性。我们提出了PART，一种利用离网补丁之间连续相对变换的自我监督学习方法，以克服这些限制。通过在连续空间中建模部件之间的关系，PART 学习图像的相对组成——一种离网结构的相对定位过程，可以超越遮挡和变形进行泛化。在需要精确空间理解的任务如物体检测和时间序列预测中，PART 在各类任务中均优于强大的基于网格方法如MAE和DropPos，同时在全局分类任务中保持了竞争力，且无需大量超参数调整。通过摆脱网格约束，PART 为跨多种数据类型的通用自我监督预训练开辟了一条新的路径，包括自然图像、EEG信号等，并在视频、医学成像和音频领域展现出巨大的潜力。 

---
# MambaNeXt-YOLO: A Hybrid State Space Model for Real-time Object Detection 

**Title (ZH)**: MambaNeXt-YOLO: 一种用于实时目标检测的混合状态空间模型 

**Authors**: Xiaochun Lei, Siqi Wu, Weilin Wu, Zetao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03654)  

**Abstract**: Real-time object detection is a fundamental but challenging task in computer vision, particularly when computational resources are limited. Although YOLO-series models have set strong benchmarks by balancing speed and accuracy, the increasing need for richer global context modeling has led to the use of Transformer-based architectures. Nevertheless, Transformers have high computational complexity because of their self-attention mechanism, which limits their practicality for real-time and edge deployments. To overcome these challenges, recent developments in linear state space models, such as Mamba, provide a promising alternative by enabling efficient sequence modeling with linear complexity. Building on this insight, we propose MambaNeXt-YOLO, a novel object detection framework that balances accuracy and efficiency through three key contributions: (1) MambaNeXt Block: a hybrid design that integrates CNNs with Mamba to effectively capture both local features and long-range dependencies; (2) Multi-branch Asymmetric Fusion Pyramid Network (MAFPN): an enhanced feature pyramid architecture that improves multi-scale object detection across various object sizes; and (3) Edge-focused Efficiency: our method achieved 66.6\% mAP at 31.9 FPS on the PASCAL VOC dataset without any pre-training and supports deployment on edge devices such as the NVIDIA Jetson Xavier NX and Orin NX. 

**Abstract (ZH)**: 基于Mamba的实时物体检测框架MambaNeXt-YOLO：平衡准确性和效率的新方法 

---
# Spatial Understanding from Videos: Structured Prompts Meet Simulation Data 

**Title (ZH)**: 基于视频的空间理解：结构化提示与模拟数据相结合 

**Authors**: Haoyu Zhang, Meng Liu, Zaijing Li, Haokun Wen, Weili Guan, Yaowei Wang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.03642)  

**Abstract**: Visual-spatial understanding, the ability to infer object relationships and layouts from visual input, is fundamental to downstream tasks such as robotic navigation and embodied interaction. However, existing methods face spatial uncertainty and data scarcity, limiting the 3D spatial reasoning capability of pre-trained vision-language models (VLMs). To address these challenges, we present a unified framework for enhancing 3D spatial reasoning in pre-trained VLMs without modifying their architecture. This framework combines SpatialMind, a structured prompting strategy that decomposes complex scenes and questions into interpretable reasoning steps, with ScanForgeQA, a scalable question-answering dataset built from diverse 3D simulation scenes through an automated construction process designed for fine-tuning. Extensive experiments across multiple benchmarks demonstrate the individual and combined effectiveness of our prompting and fine-tuning strategies, and yield insights that may inspire future research on visual-spatial understanding. 

**Abstract (ZH)**: 视觉空间理解能力，即从视觉输入中推断物体关系和布局的能力，是诸如机器人导航和实体交互等下游任务的基础。然而，现有的方法面临空间不确定性与数据稀缺性的问题，限制了预训练视觉-语言模型（VLMs）的3D空间推理能力。为解决这些挑战，我们提出了一种无需修改架构即可增强预训练VLMs的3D空间推理能力的统一框架。该框架结合了SpatialMind，一种结构化的提示策略，将复杂场景和问题分解为可解释的推理步骤，以及ScanForgeQA，一种通过自动化构建过程从多种3D模拟场景中生成、适用于微调的可扩展问答数据集。跨多个基准的广泛实验表明，我们的提示和微调策略的个体及联合效果，并为未来视觉空间理解研究提供了启示。 

---
# DiagNet: Detecting Objects using Diagonal Constraints on Adjacency Matrix of Graph Neural Network 

**Title (ZH)**: DiagNet：基于图神经网络邻接矩阵对角约束的物体检测方法 

**Authors**: Chong Hyun Lee, Kibae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.03571)  

**Abstract**: We propose DaigNet, a new approach to object detection with which we can detect an object bounding box using diagonal constraints on adjacency matrix of a graph convolutional network (GCN). We propose two diagonalization algorithms based on hard and soft constraints on adjacency matrix and two loss functions using diagonal constraint and complementary constraint. The DaigNet eliminates the need for designing a set of anchor boxes commonly used. To prove feasibility of our novel detector, we adopt detection head in YOLO models. Experiments show that the DiagNet achieves 7.5% higher mAP50 on Pascal VOC than YOLOv1. The DiagNet also shows 5.1% higher mAP on MS COCO than YOLOv3u, 3.7% higher mAP than YOLOv5u, and 2.9% higher mAP than YOLOv8. 

**Abstract (ZH)**: 我们提出DaigNet，这是一种使用图卷积网络（GCN）邻接矩阵对角约束进行对象检测的新方法。我们提出了基于邻接矩阵硬约束和软约束的两种对角化算法，并使用对角约束和互补约束提出了两种损失函数。DaigNet消除了常用锚框集的设计需求。为了证明我们新型检测器的可行性，我们在YOLO模型中采用检测头部。实验结果显示，DiagNet在Pascal VOC上的mAP50比YOLOv1高7.5%。DiagNet在MS COCO上的mAP比YOLOv3u高5.1%，比YOLOv5u高3.7%，比YOLOv8高2.9%。 

---
# Video-Skill-CoT: Skill-based Chain-of-Thoughts for Domain-Adaptive Video Reasoning 

**Title (ZH)**: Video-Skill-CoT: 基于技能的链式思考在领域自适应视频推理中的应用 

**Authors**: Daeun Lee, Jaehong Yoon, Jaemin Cho, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2506.03525)  

**Abstract**: Recent advances in Chain-of-Thought (CoT) reasoning have improved complex video understanding, but existing methods often struggle to adapt to domain-specific skills (e.g., event detection, spatial relation understanding, emotion understanding) over various video content. To address this, we propose Video-Skill-CoT (a.k.a. Video-SKoT), a framework that automatically constructs and leverages skill-aware CoT supervisions for domain-adaptive video reasoning. First, we construct skill-based CoT annotations: we extract domain-relevant reasoning skills from training questions, cluster them into a shared skill taxonomy, and create detailed multi-step CoT rationale tailored to each video-question pair for training. Second, we introduce a skill-specific expert learning framework. Each expert module specializes in a subset of reasoning skills and is trained with lightweight adapters using the collected CoT supervision. We demonstrate the effectiveness of the proposed approach on three video understanding benchmarks, where Video-SKoT consistently outperforms strong baselines. We also provide in-depth analyses on comparing different CoT annotation pipelines and learned skills over multiple video domains. 

**Abstract (ZH)**: Recent advances in Chain-of-Thought (CoT) reasoning have improved complex video understanding, but existing methods often struggle to adapt to domain-specific skills (e.g., event detection, spatial relation understanding, emotion understanding) over various video content. To address this, we propose Video-Skill-CoT (a.k.a. Video-SKoT), a framework that automatically constructs and leverages skill-aware CoT supervisions for domain-adaptive video reasoning. First, we construct skill-based CoT annotations: we extract domain-relevant reasoning skills from training questions, cluster them into a shared skill taxonomy, and create detailed multi-step CoT rationale tailored to each video-question pair for training. Second, we introduce a skill-specific expert learning framework. Each expert module specializes in a subset of reasoning skills and is trained with lightweight adapters using the collected CoT supervision. We demonstrate the effectiveness of the proposed approach on three video understanding benchmarks, where Video-SKoT consistently outperforms strong baselines. We also provide in-depth analyses on comparing different CoT annotation pipelines and learned skills over multiple video domains。翻译标题：

最近在Chain-of-Thought (CoT)推理方面的进展提高了复杂视频的理解能力，但现有方法往往难以适应各种视频内容中的领域特定技能（例如，事件检测、空间关系理解、情感理解）。为此，我们提出了一种名为Video-Skill-CoT（简称Video-SKoT）的方法，该框架能够自动构建并利用领域感知的CoT监督，促进领域适应的视频推理。首先，我们构建了基于技能的CoT注释：从训练问题中提取领域相关的推理技能，将它们聚类成共享技能分类法，并为每个视频-问题对创建详细的多步CoT推理。其次，我们引入了一种针对特定技能的专家学习框架。每个专家模块专门处理一组推理技能，并使用收集到的CoT监督与轻量级适配器进行训练。我们通过三个视频理解基准测试验证了所提出方法的有效性，其中Video-SKoT在所有基准测试中均优于强基线模型。我们还对不同CoT注释管道进行了深入分析，并研究了多个视频领域中学习到的技能。 

---
# POLARIS: A High-contrast Polarimetric Imaging Benchmark Dataset for Exoplanetary Disk Representation Learning 

**Title (ZH)**: POLARIS：一种用于外行星盘表示学习的高对比度偏振成像基准数据集 

**Authors**: Fangyi Cao, Bin Ren, Zihao Wang, Shiwei Fu, Youbin Mo, Xiaoyang Liu, Yuzhou Chen, Weixin Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.03511)  

**Abstract**: With over 1,000,000 images from more than 10,000 exposures using state-of-the-art high-contrast imagers (e.g., Gemini Planet Imager, VLT/SPHERE) in the search for exoplanets, can artificial intelligence (AI) serve as a transformative tool in imaging Earth-like exoplanets in the coming decade? In this paper, we introduce a benchmark and explore this question from a polarimetric image representation learning perspective. Despite extensive investments over the past decade, only a few new exoplanets have been directly imaged. Existing imaging approaches rely heavily on labor-intensive labeling of reference stars, which serve as background to extract circumstellar objects (disks or exoplanets) around target stars. With our POLARIS (POlarized Light dAta for total intensity Representation learning of direct Imaging of exoplanetary Systems) dataset, we classify reference star and circumstellar disk images using the full public SPHERE/IRDIS polarized-light archive since 2014, requiring less than 10 percent manual labeling. We evaluate a range of models including statistical, generative, and large vision-language models and provide baseline performance. We also propose an unsupervised generative representation learning framework that integrates these models, achieving superior performance and enhanced representational power. To our knowledge, this is the first uniformly reduced, high-quality exoplanet imaging dataset, rare in astrophysics and machine learning. By releasing this dataset and baselines, we aim to equip astrophysicists with new tools and engage data scientists in advancing direct exoplanet imaging, catalyzing major interdisciplinary breakthroughs. 

**Abstract (ZH)**: 基于偏振图像表示学习：阿尔文智像 dataset 在直接成像地球类系外行星中的潜力探讨 

---
# Multi-Spectral Gaussian Splatting with Neural Color Representation 

**Title (ZH)**: 多光谱高斯散列与神经颜色表示 

**Authors**: Lukas Meyer, Josef Grün, Maximilian Weiherer, Bernhard Egger, Marc Stamminger, Linus Franke  

**Link**: [PDF](https://arxiv.org/pdf/2506.03407)  

**Abstract**: We present MS-Splatting -- a multi-spectral 3D Gaussian Splatting (3DGS) framework that is able to generate multi-view consistent novel views from images of multiple, independent cameras with different spectral domains. In contrast to previous approaches, our method does not require cross-modal camera calibration and is versatile enough to model a variety of different spectra, including thermal and near-infra red, without any algorithmic changes.
Unlike existing 3DGS-based frameworks that treat each modality separately (by optimizing per-channel spherical harmonics) and therefore fail to exploit the underlying spectral and spatial correlations, our method leverages a novel neural color representation that encodes multi-spectral information into a learned, compact, per-splat feature embedding. A shallow multi-layer perceptron (MLP) then decodes this embedding to obtain spectral color values, enabling joint learning of all bands within a unified representation.
Our experiments show that this simple yet effective strategy is able to improve multi-spectral rendering quality, while also leading to improved per-spectra rendering quality over state-of-the-art methods. We demonstrate the effectiveness of this new technique in agricultural applications to render vegetation indices, such as normalized difference vegetation index (NDVI). 

**Abstract (ZH)**: 多光谱3D高斯点云渲染框架：MS-Splatting 

---
# Rethinking Whole-Body CT Image Interpretation: An Abnormality-Centric Approach 

**Title (ZH)**: 整身CT图像解释的重新思考：基于异常的方法 

**Authors**: Ziheng Zhao, Lisong Dai, Ya Zhang, Yanfeng Wang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.03238)  

**Abstract**: Automated interpretation of CT images-particularly localizing and describing abnormal findings across multi-plane and whole-body scans-remains a significant challenge in clinical radiology. This work aims to address this challenge through four key contributions: (i) On taxonomy, we collaborate with senior radiologists to propose a comprehensive hierarchical classification system, with 404 representative abnormal findings across all body regions; (ii) On data, we contribute a dataset containing over 14.5K CT images from multiple planes and all human body regions, and meticulously provide grounding annotations for over 19K abnormalities, each linked to the detailed description and cast into the taxonomy; (iii) On model development, we propose OminiAbnorm-CT, which can automatically ground and describe abnormal findings on multi-plane and whole-body CT images based on text queries, while also allowing flexible interaction through visual prompts; (iv) On benchmarks, we establish three representative evaluation tasks based on real clinical scenarios. Through extensive experiments, we show that OminiAbnorm-CT can significantly outperform existing methods on all the tasks and metrics. 

**Abstract (ZH)**: 自动解读CT图像，特别是在多平面和全身扫描中定位和描述异常发现，仍然是临床放射学中的一个重要挑战。本文通过四个关键贡献来应对这一挑战：（i）在分类学上，我们与资深放射科医生合作，提出了一种全面的分层次分类系统，涵盖404种代表性全身各区域的异常发现；（ii）在数据方面，我们提供了一个包含超过14500张多平面和全身各区域CT图像的数据集，并详细标注了超过19000个异常，每个异常都与详细的描述相链接，并纳入分类系统；（iii）在模型开发方面，我们提出了OminiAbnorm-CT，该模型可以根据文本查询自动在多平面和全身CT图像中定位和描述异常发现，同时通过视觉提示支持灵活交互；（iv）在基准测试方面，我们基于真实临床场景建立了三个代表性评估任务。通过广泛的实验，我们表明OminiAbnorm-CT在所有任务和指标上显著优于现有方法。 

---
# Multi-Analyte, Swab-based Automated Wound Monitor with AI 

**Title (ZH)**: 基于 swab 的多分析物自动伤口监测系统与 AI 

**Authors**: Madhu Babu Sikha, Lalith Appari, Gurudatt Nanjanagudu Ganesh, Amay Bandodkar, Imon Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2506.03188)  

**Abstract**: Diabetic foot ulcers (DFUs), a class of chronic wounds, affect ~750,000 individuals every year in the US alone and identifying non-healing DFUs that develop to chronic wounds early can drastically reduce treatment costs and minimize risks of amputation. There is therefore a pressing need for diagnostic tools that can detect non-healing DFUs early. We develop a low cost, multi-analyte 3D printed assays seamlessly integrated on swabs that can identify non-healing DFUs and a Wound Sensor iOS App - an innovative mobile application developed for the controlled acquisition and automated analysis of wound sensor data. By comparing both the original base image (before exposure to the wound) and the wound-exposed image, we developed automated computer vision techniques to compare density changes between the two assay images, which allow us to automatically determine the severity of the wound. The iOS app ensures accurate data collection and presents actionable insights, despite challenges such as variations in camera configurations and ambient conditions. The proposed integrated sensor and iOS app will allow healthcare professionals to monitor wound conditions real-time, track healing progress, and assess critical parameters related to wound care. 

**Abstract (ZH)**: 糖尿病足溃疡（DFUs）的低成本多分析物3D打印检测 assay 及集成伤口传感器iOS应用：早期识别非愈合DFUs以大幅降低治疗成本并减少截肢风险 

---
# Lightweight Convolutional Neural Networks for Retinal Disease Classification 

**Title (ZH)**: 轻量级卷积神经网络在视网膜疾病分类中的应用 

**Authors**: Duaa Kareem Qasim, Sabah Abdulazeez Jebur, Lafta Raheem Ali, Abdul Jalil M. Khalaf, Abir Jaafar Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2506.03186)  

**Abstract**: Retinal diseases such as Diabetic Retinopathy (DR) and Macular Hole (MH) significantly impact vision and affect millions worldwide. Early detection is crucial, as DR, a complication of diabetes, damages retinal blood vessels, potentially leading to blindness, while MH disrupts central vision, affecting tasks like reading and facial recognition. This paper employed two lightweight and efficient Convolution Neural Network architectures, MobileNet and NASNetMobile, for the classification of Normal, DR, and MH retinal images. The models were trained on the RFMiD dataset, consisting of 3,200 fundus images, after undergoing preprocessing steps such as resizing, normalization, and augmentation. To address data scarcity, this study leveraged transfer learning and data augmentation techniques, enhancing model generalization and performance. The experimental results demonstrate that MobileNetV2 achieved the highest accuracy of 90.8%, outperforming NASNetMobile, which achieved 89.5% accuracy. These findings highlight the effectiveness of CNNs in retinal disease classification, providing a foundation for AI-assisted ophthalmic diagnosis and early intervention. 

**Abstract (ZH)**: Retinal疾病如糖尿病视网膜病变(DR)和黄斑裂孔(MH)严重影响视力并影响数以百万计的人。早期诊断至关重要，因为糖尿病视网膜病变会损害视网膜血管，可能导致失明，而黄斑裂孔会破坏中央视力，影响阅读和面部识别。本文采用了两种轻量级高效的卷积神经网络架构-MobileNet和NASNetMobile-对正常、DR和MH视网膜图像进行分类。模型在经过大小调整、归一化和增强等预处理步骤的RFMiD数据集中进行训练，该数据集包含3,200张底片图像。为解决数据稀缺问题，本研究利用了迁移学习和数据增强技术，提高了模型的泛化能力和性能。实验结果表明，MobileNetV2的准确率最高，达到90.8%，优于NASNetMobile的89.5%准确率。这些发现突显了CNN在视网膜疾病分类中的有效性，为AI辅助眼科诊断和早期干预奠定了基础。 

---
# Impact of Tuning Parameters in Deep Convolutional Neural Network Using a Crack Image Dataset 

**Title (ZH)**: 使用裂纹图像数据集探究深度卷积神经网络调参影响 

**Authors**: Mahe Zabin, Ho-Jin Choi, Md. Monirul Islam, Jia Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2506.03184)  

**Abstract**: The performance of a classifier depends on the tuning of its parame ters. In this paper, we have experimented the impact of various tuning parameters on the performance of a deep convolutional neural network (DCNN). In the ex perimental evaluation, we have considered a DCNN classifier that consists of 2 convolutional layers (CL), 2 pooling layers (PL), 1 dropout, and a dense layer. To observe the impact of pooling, activation function, and optimizer tuning pa rameters, we utilized a crack image dataset having two classes: negative and pos itive. The experimental results demonstrate that with the maxpooling, the DCNN demonstrates its better performance for adam optimizer and tanh activation func tion. 

**Abstract (ZH)**: 深卷积神经网络参数调整对其性能的影响：基于裂缝图像数据集的实验研究 

---
# Deep Learning-Based Breast Cancer Detection in Mammography: A Multi-Center Validation Study in Thai Population 

**Title (ZH)**: 基于深度学习的乳腺癌在乳腺X线摄影中的检测：泰国人群多中心验证研究 

**Authors**: Isarun Chamveha, Supphanut Chaiyungyuen, Sasinun Worakriangkrai, Nattawadee Prasawang, Warasinee Chaisangmongkon, Pornpim Korpraphong, Voraparee Suvannarerg, Shanigarn Thiravit, Chalermdej Kannawat, Kewalin Rungsinaporn, Suwara Issaragrisil, Payia Chadbunchachai, Pattiya Gatechumpol, Chawiporn Muktabhant, Patarachai Sereerat  

**Link**: [PDF](https://arxiv.org/pdf/2506.03177)  

**Abstract**: This study presents a deep learning system for breast cancer detection in mammography, developed using a modified EfficientNetV2 architecture with enhanced attention mechanisms. The model was trained on mammograms from a major Thai medical center and validated on three distinct datasets: an in-domain test set (9,421 cases), a biopsy-confirmed set (883 cases), and an out-of-domain generalizability set (761 cases) collected from two different hospitals. For cancer detection, the model achieved AUROCs of 0.89, 0.96, and 0.94 on the respective datasets. The system's lesion localization capability, evaluated using metrics including Lesion Localization Fraction (LLF) and Non-Lesion Localization Fraction (NLF), demonstrated robust performance in identifying suspicious regions. Clinical validation through concordance tests showed strong agreement with radiologists: 83.5% classification and 84.0% localization concordance for biopsy-confirmed cases, and 78.1% classification and 79.6% localization concordance for out-of-domain cases. Expert radiologists' acceptance rate also averaged 96.7% for biopsy-confirmed cases, and 89.3% for out-of-domain cases. The system achieved a System Usability Scale score of 74.17 for source hospital, and 69.20 for validation hospitals, indicating good clinical acceptance. These results demonstrate the model's effectiveness in assisting mammogram interpretation, with the potential to enhance breast cancer screening workflows in clinical practice. 

**Abstract (ZH)**: 乳腺癌在乳房X光摄影中的检测：基于改进EfficientNetV2架构的深度学习系统的研究 

---
# PALADIN : Robust Neural Fingerprinting for Text-to-Image Diffusion Models 

**Title (ZH)**: PALADIN：稳健的神经指纹技术用于文本到图像扩散模型 

**Authors**: Murthy L, Subarna Tripathi  

**Link**: [PDF](https://arxiv.org/pdf/2506.03170)  

**Abstract**: The risk of misusing text-to-image generative models for malicious uses, especially due to the open-source development of such models, has become a serious concern. As a risk mitigation strategy, attributing generative models with neural fingerprinting is emerging as a popular technique. There has been a plethora of recent work that aim for addressing neural fingerprinting. A trade-off between the attribution accuracy and generation quality of such models has been studied extensively. None of the existing methods yet achieved $100\%$ attribution accuracy. However, any model with less than \emph{perfect} accuracy is practically non-deployable. In this work, we propose an accurate method to incorporate neural fingerprinting for text-to-image diffusion models leveraging the concepts of cyclic error correcting codes from the literature of coding theory. 

**Abstract (ZH)**: 利用循环错误校正码概念将神经指纹印技术应用于文本到图像扩散模型以减轻误用风险 

---
# Improvement of human health lifespan with hybrid group pose estimation methods 

**Title (ZH)**: 基于混合群体姿态估计方法的人类健康寿命提升 

**Authors**: Arindam Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2506.03169)  

**Abstract**: Human beings rely heavily on estimation of poses in order to access their body movements. Human pose estimation methods take advantage of computer vision advances in order to track human body movements in real life applications. This comes from videos which are recorded through available devices. These para-digms provide potential to make human movement measurement more accessible to users. The consumers of pose estimation movements believe that human poses content tend to supplement available videos. This has increased pose estimation software usage to estimate human poses. In order to address this problem, we develop hybrid-ensemble-based group pose estimation method to improve human health. This proposed hybrid-ensemble-based group pose estimation method aims to detect multi-person poses using modified group pose estimation and modified real time pose estimation. This ensemble allows fusion of performance of stated methods in real time. The input poses from images are fed into individual meth-ods. The pose transformation method helps to identify relevant features for en-semble to perform training effectively. After this, customized pre-trained hybrid ensemble is trained on public benchmarked datasets which is being evaluated through test datasets. The effectiveness and viability of proposed method is estab-lished based on comparative analysis of group pose estimation methods and ex-periments conducted on benchmarked datasets. It provides best optimized results in real-time pose estimation. It makes pose estimation method more robust to oc-clusion and improves dense regression accuracy. These results have affirmed po-tential application of this method in several real-time situations with improvement in human health life span 

**Abstract (ZH)**: 基于混合集成的人群姿态估计方法以提升人类健康 

---
# Dual Branch VideoMamba with Gated Class Token Fusion for Violence Detection 

**Title (ZH)**: 具有门控类令牌融合的双分支VideoMamba暴力检测 

**Authors**: Damith Chamalke Senadeera, Xiaoyun Yang, Dimitrios Kollias, Gregory Slabaugh  

**Link**: [PDF](https://arxiv.org/pdf/2506.03162)  

**Abstract**: The rapid proliferation of surveillance cameras has increased the demand for automated violence detection. While CNNs and Transformers have shown success in extracting spatio-temporal features, they struggle with long-term dependencies and computational efficiency. We propose Dual Branch VideoMamba with Gated Class Token Fusion (GCTF), an efficient architecture combining a dual-branch design and a state-space model (SSM) backbone where one branch captures spatial features, while the other focuses on temporal dynamics, with continuous fusion via a gating mechanism. We also present a new benchmark by merging RWF-2000, RLVS, and VioPeru datasets in video violence detection, ensuring strict separation between training and testing sets. Our model achieves state-of-the-art performance on this benchmark offering an optimal balance between accuracy and computational efficiency, demonstrating the promise of SSMs for scalable, real-time surveillance violence detection. 

**Abstract (ZH)**: 监视摄像头的快速普及增加了自动化暴力检测的需求。虽然CNN和Transformer在提取时空特征方面取得了成功，但在处理长期依赖性和计算效率方面仍然存在挑战。我们提出了一个高效的Dual Branch VideoMamba with Gated Class Token Fusion (GCTF) 架构，结合了双分支设计和状态空间模型（SSM）骨干网络，其中一个分支捕获空间特征，而另一个分支专注于时间动态，并通过门控机制实现连续融合。我们还提出一个新的基准，通过将RWF-2000、RLVS和VioPeru数据集合并，在视频暴力检测中确保训练集和测试集之间严格分离。我们的模型在这一基准上达到了最优的准确性和计算效率之间的平衡，证明了SSM在可扩展、实时 surveillance 暴力检测中的潜力。 

---
