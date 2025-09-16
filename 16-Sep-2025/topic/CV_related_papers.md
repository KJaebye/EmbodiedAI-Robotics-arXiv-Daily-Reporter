# ViSTR-GP: Online Cyberattack Detection via Vision-to-State Tensor Regression and Gaussian Processes in Automated Robotic Operations 

**Title (ZH)**: ViSTR-GP: 自动化机器人操作中基于视觉至状态张量回归和高斯过程的在线网络攻击检测 

**Authors**: Navid Aftabi, Philip Samaha, Jin Ma, Long Cheng, Ramy Harik, Dan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.10948)  

**Abstract**: Industrial robotic systems are central to automating smart manufacturing operations. Connected and automated factories face growing cybersecurity risks that can potentially cause interruptions and damages to physical operations. Among these attacks, data-integrity attacks often involve sophisticated exploitation of vulnerabilities that enable an attacker to access and manipulate the operational data and are hence difficult to detect with only existing intrusion detection or model-based detection. This paper addresses the challenges in utilizing existing side-channels to detect data-integrity attacks in robotic manufacturing processes by developing an online detection framework, ViSTR-GP, that cross-checks encoder-reported measurements against a vision-based estimate from an overhead camera outside the controller's authority. In this framework, a one-time interactive segmentation initializes SAM-Track to generate per-frame masks. A low-rank tensor-regression surrogate maps each mask to measurements, while a matrix-variate Gaussian process models nominal residuals, capturing temporal structure and cross-joint correlations. A frame-wise test statistic derived from the predictive distribution provides an online detector with interpretable thresholds. We validate the framework on a real-world robotic testbed with synchronized video frame and encoder data, collecting multiple nominal cycles and constructing replay attack scenarios with graded end-effector deviations. Results on the testbed indicate that the proposed framework recovers joint angles accurately and detects data-integrity attacks earlier with more frequent alarms than all baselines. These improvements are most evident in the most subtle attacks. These results show that plants can detect data-integrity attacks by adding an independent physical channel, bypassing the controller's authority, without needing complex instrumentation. 

**Abstract (ZH)**: 基于视觉的在线数据完整性攻击检测框架：应用于机器人制造过程的安全性增强 

---
# FastTrack: GPU-Accelerated Tracking for Visual SLAM 

**Title (ZH)**: FastTrack：视觉SLAM中的GPU加速跟踪 

**Authors**: Kimia Khabiri, Parsa Hosseininejad, Shishir Gopinath, Karthik Dantu, Steven Y. Ko  

**Link**: [PDF](https://arxiv.org/pdf/2509.10757)  

**Abstract**: The tracking module of a visual-inertial SLAM system processes incoming image frames and IMU data to estimate the position of the frame in relation to the map. It is important for the tracking to complete in a timely manner for each frame to avoid poor localization or tracking loss. We therefore present a new approach which leverages GPU computing power to accelerate time-consuming components of tracking in order to improve its performance. These components include stereo feature matching and local map tracking. We implement our design inside the ORB-SLAM3 tracking process using CUDA. Our evaluation demonstrates an overall improvement in tracking performance of up to 2.8x on a desktop and Jetson Xavier NX board in stereo-inertial mode, using the well-known SLAM datasets EuRoC and TUM-VI. 

**Abstract (ZH)**: 视觉惯性SLAM系统中基于GPU的跟踪模块利用GPU计算能力加速耗时的跟踪组件，以提高跟踪性能。这些组件包括立体特征匹配和局部地图跟踪。在ORB-SLAM3跟踪过程中使用CUDA实现我们的设计。评估结果显示，在桌面和Jetson Xavier NX板的立体惯性模式下，使用著名的SLAM数据集EuRoC和TUM-VI，跟踪性能整体提升最高可达2.8倍。 

---
# Learning to Generate 4D LiDAR Sequences 

**Title (ZH)**: 学习生成4D LiDAR序列 

**Authors**: Ao Liang, Youquan Liu, Yu Yang, Dongyue Lu, Linfeng Li, Lingdong Kong, Huaici Zhao, Wei Tsang Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2509.11959)  

**Abstract**: While generative world models have advanced video and occupancy-based data synthesis, LiDAR generation remains underexplored despite its importance for accurate 3D perception. Extending generation to 4D LiDAR data introduces challenges in controllability, temporal stability, and evaluation. We present LiDARCrafter, a unified framework that converts free-form language into editable LiDAR sequences. Instructions are parsed into ego-centric scene graphs, which a tri-branch diffusion model transforms into object layouts, trajectories, and shapes. A range-image diffusion model generates the initial scan, and an autoregressive module extends it into a temporally coherent sequence. The explicit layout design further supports object-level editing, such as insertion or relocation. To enable fair assessment, we provide EvalSuite, a benchmark spanning scene-, object-, and sequence-level metrics. On nuScenes, LiDARCrafter achieves state-of-the-art fidelity, controllability, and temporal consistency, offering a foundation for LiDAR-based simulation and data augmentation. 

**Abstract (ZH)**: 尽管生成型世界模型在视频和占用数据合成方面取得了进展，但LiDAR生成仍然未被充分探索，尽管其对于准确的3D感知非常重要。将生成扩展到4D LiDAR数据引入了可控性、时间稳定性以及评估方面的挑战。我们介绍了一种统一框架LiDARCrafter，该框架将自由形式的语言转换为可编辑的LiDAR序列。指令被解析为以自我为中心的场景图，随后由三分支扩散模型转换为对象布局、轨迹和形状。范围图像扩散模型生成初始扫描，而自回归模块将其扩展为时间连贯的序列。明确的设计布局进一步支持对象级别的编辑，如插入或重新定位。为了实现公平的评估，我们提供了EvalSuite基准，该基准涵盖了场景、对象和序列级别的指标。在nuScenes数据集上，LiDARCrafter在保真度、可控性和时间一致性方面获得了最先进的性能，为基于LiDAR的模拟和数据增强提供了基础。 

---
# Cross-Platform Scaling of Vision-Language-Action Models from Edge to Cloud GPUs 

**Title (ZH)**: 边缘到云GPU跨平台扩展的视觉-语言-动作模型 

**Authors**: Amir Taherin, Juyi Lin, Arash Akbari, Arman Akbari, Pu Zhao, Weiwei Chen, David Kaeli, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11480)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as powerful generalist policies for robotic control, yet their performance scaling across model architectures and hardware platforms, as well as their associated power budgets, remain poorly understood. This work presents an evaluation of five representative VLA models -- spanning state-of-the-art baselines and two newly proposed architectures -- targeting edge and datacenter GPU platforms. Using the LIBERO benchmark, we measure accuracy alongside system-level metrics, including latency, throughput, and peak memory usage, under varying edge power constraints and high-performance datacenter GPU configurations. Our results identify distinct scaling trends: (1) architectural choices, such as action tokenization and model backbone size, strongly influence throughput and memory footprint; (2) power-constrained edge devices exhibit non-linear performance degradation, with some configurations matching or exceeding older datacenter GPUs; and (3) high-throughput variants can be achieved without significant accuracy loss. These findings provide actionable insights when selecting and optimizing VLAs across a range of deployment constraints. Our work challenges current assumptions about the superiority of datacenter hardware for robotic inference. 

**Abstract (ZH)**: Vision-Language-Action (VLA)模型已发展成为机器人控制的强大通用政策，但其在不同模型架构和硬件平台上的性能扩展性以及相关的功耗预算仍不甚明了。本研究对五个代表性的VLA模型进行了评估——这些模型涵盖了最先进的基线模型和两种新提出的架构，并针对边缘设备和数据中心GPU平台进行靶向优化。使用LIBERO基准测试，我们在不同的边缘功耗限制和高性能数据中心GPU配置下，测量了模型的准确性以及系统级指标，包括延迟、吞吐量和峰值内存使用量。我们的研究结果揭示了不同的扩展趋势：（1）架构选择，如动作标记化和模型主干大小，强烈影响吞吐量和内存占用；（2）功耗受限的边缘设备表现出非线性性能下降，某些配置甚至可匹配或超过较老的数据中心GPU；（3）可以实现高吞吐量变体而不显著牺牲准确性。这些发现为在各种部署约束条件下选择和优化VLA模型提供了可操作的见解。本研究质疑了数据中心硬件在机器人推理方面优势的现有假设。 

---
# Beyond Frame-wise Tracking: A Trajectory-based Paradigm for Efficient Point Cloud Tracking 

**Title (ZH)**: 超越帧级跟踪：一种高效的点云轨迹导向跟踪范式 

**Authors**: BaiChen Fan, Sifan Zhou, Jian Li, Shibo Zhao, Muqing Cao, Qin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11453)  

**Abstract**: LiDAR-based 3D single object tracking (3D SOT) is a critical task in robotics and autonomous systems. Existing methods typically follow frame-wise motion estimation or a sequence-based paradigm. However, the two-frame methods are efficient but lack long-term temporal context, making them vulnerable in sparse or occluded scenes, while sequence-based methods that process multiple point clouds gain robustness at a significant computational cost. To resolve this dilemma, we propose a novel trajectory-based paradigm and its instantiation, TrajTrack. TrajTrack is a lightweight framework that enhances a base two-frame tracker by implicitly learning motion continuity from historical bounding box trajectories alone-without requiring additional, costly point cloud inputs. It first generates a fast, explicit motion proposal and then uses an implicit motion modeling module to predict the future trajectory, which in turn refines and corrects the initial proposal. Extensive experiments on the large-scale NuScenes benchmark show that TrajTrack achieves new state-of-the-art performance, dramatically improving tracking precision by 4.48% over a strong baseline while running at 56 FPS. Besides, we also demonstrate the strong generalizability of TrajTrack across different base trackers. Video is available at this https URL. 

**Abstract (ZH)**: 基于LiDAR的3D单目标跟踪（3D SOT）是机器人和自主系统中的一个关键任务。现有的方法通常遵循帧级运动估计或基于序列的范式。然而，两帧方法效率高但缺乏长期时序上下文，在稀疏或被遮挡的场景中容易失效，而处理多个点云的序列方法虽然具有鲁棒性，但在计算成本上显著增加。为了解决这一矛盾，我们提出了一种新型的轨迹导向范式及其具体实例TrajTrack。TrajTrack是一种轻量级框架，通过仅从历史边界框轨迹中隐式学习运动连续性来增强基础的两帧跟踪器，无需额外的成本高昂的点云输入。它首先生成一个快速的显式运动提案，然后使用隐式运动建模模块预测未来的轨迹，进而细化和纠正初始提案。在大规模NuScenes基准测试上的大量实验表明，TrajTrack达到了新的最佳性能，在运行速度达到56 FPS的情况下，比强大基准提高了4.48%的跟踪精度。此外，我们还展示了TrajTrack在不同基础跟踪器上的强大泛化能力。视频可访问此链接。 

---
# Point-Plane Projections for Accurate LiDAR Semantic Segmentation in Small Data Scenarios 

**Title (ZH)**: 点面投影用于小数据场景下的LiDAR语义分割 

**Authors**: Simone Mosco, Daniel Fusaro, Wanmeng Li, Emanuele Menegatti, Alberto Pretto  

**Link**: [PDF](https://arxiv.org/pdf/2509.10841)  

**Abstract**: LiDAR point cloud semantic segmentation is essential for interpreting 3D environments in applications such as autonomous driving and robotics. Recent methods achieve strong performance by exploiting different point cloud representations or incorporating data from other sensors, such as cameras or external datasets. However, these approaches often suffer from high computational complexity and require large amounts of training data, limiting their generalization in data-scarce scenarios. In this paper, we improve the performance of point-based methods by effectively learning features from 2D representations through point-plane projections, enabling the extraction of complementary information while relying solely on LiDAR data. Additionally, we introduce a geometry-aware technique for data augmentation that aligns with LiDAR sensor properties and mitigates class imbalance. We implemented and evaluated our method that applies point-plane projections onto multiple informative 2D representations of the point cloud. Experiments demonstrate that this approach leads to significant improvements in limited-data scenarios, while also achieving competitive results on two publicly available standard datasets, as SemanticKITTI and PandaSet. The code of our method is available at this https URL 

**Abstract (ZH)**: LiDAR点云语义分割对于在自动驾驶和机器人等领域解释3D环境至关重要。近期的方法通过利用不同的点云表示或结合其他传感器（如摄像头或外部数据集）的数据，实现了较强的性能。然而，这些方法通常面临高计算复杂度和需要大量训练数据的问题，限制了它们在数据稀缺场景下的泛化能力。本文通过有效学习点平面投影的2D表示特征，提高了基于点的方法的性能，从而仅依赖LiDAR数据即可提取互补信息。此外，我们还引入了一种几何感知的数据增强技术，该技术符合LiDAR传感器的特性并缓解了类别不平衡问题。我们在多个信息性的2D点云表示上应用点平面投影的方法进行了实施和评估。实验结果表明，该方法在数据稀缺场景中取得了显著的改进，并在SemanticKITTI和PandaSet等两个公开的标准数据集上达到了具有竞争力的结果。我们的方法代码可在此处访问：this https URL。 

---
# Learning Representations in Video Game Agents with Supervised Contrastive Imitation Learning 

**Title (ZH)**: 基于监督对比模仿学习的视频游戏代理表示学习 

**Authors**: Carlos Celemin, Joseph Brennan, Pierluigi Vito Amadori, Tim Bradley  

**Link**: [PDF](https://arxiv.org/pdf/2509.11880)  

**Abstract**: This paper introduces a novel application of Supervised Contrastive Learning (SupCon) to Imitation Learning (IL), with a focus on learning more effective state representations for agents in video game environments. The goal is to obtain latent representations of the observations that capture better the action-relevant factors, thereby modeling better the cause-effect relationship from the observations that are mapped to the actions performed by the demonstrator, for example, the player jumps whenever an obstacle appears ahead. We propose an approach to integrate the SupCon loss with continuous output spaces, enabling SupCon to operate without constraints regarding the type of actions of the environment. Experiments on the 3D games Astro Bot and Returnal, and multiple 2D Atari games show improved representation quality, faster learning convergence, and better generalization compared to baseline models trained only with supervised action prediction loss functions. 

**Abstract (ZH)**: 本文介绍了一种将监督对比学习（SupCon）应用于模仿学习（IL）的新颖应用，重点关注在视频游戏环境中为智能体学习更有效的状态表示。目标是获得更能捕捉行动相关因素的潜在表示，从而更好地建模从观察到演示者执行的动作所映射的原因-结果关系，例如，每当出现障碍物时，玩家就会跳跃。本文提出了一种将SupCon损失与连续输出空间集成的方法，使SupCon能够在不受到环境动作类型限制的情况下运行。在3D游戏Astro Bot和Returnal以及多种2D Atari 游戏上的实验结果显示，与仅使用监督动作预测损失函数训练的基础模型相比，该方法能提高表示质量、加快学习收敛速度并更好地泛化。 

---
# VideoAgent: Personalized Synthesis of Scientific Videos 

**Title (ZH)**: VideoAgent: 个性化科学视频合成 

**Authors**: Xiao Liang, Bangxin Li, Zixuan Chen, Hanyue Zheng, Zhi Ma, Di Wang, Cong Tian, Quan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11253)  

**Abstract**: Automating the generation of scientific videos is a crucial yet challenging task for effective knowledge dissemination. However, existing works on document automation primarily focus on static media such as posters and slides, lacking mechanisms for personalized dynamic orchestration and multimodal content synchronization. To address these challenges, we introduce VideoAgent, a novel multi-agent framework that synthesizes personalized scientific videos through a conversational interface. VideoAgent parses a source paper into a fine-grained asset library and, guided by user requirements, orchestrates a narrative flow that synthesizes both static slides and dynamic animations to explain complex concepts. To enable rigorous evaluation, we also propose SciVidEval, the first comprehensive suite for this task, which combines automated metrics for multimodal content quality and synchronization with a Video-Quiz-based human evaluation to measure knowledge transfer. Extensive experiments demonstrate that our method significantly outperforms existing commercial scientific video generation services and approaches human-level quality in scientific communication. 

**Abstract (ZH)**: 自动化生成科学视频是有效知识传播的关键但具有挑战性的任务。现有文档自动化工作主要集中在海报和幻灯片等静态媒体上，缺乏个性化动态编排和多模态内容同步的机制。为了解决这些挑战，我们引入了VideoAgent，这是一种新颖的多agent框架，通过对话界面合成个性化科学视频。VideoAgent将源论文解析为精细粒度的资产库，并在用户要求的引导下，编排叙述流程，综合静态幻灯片和动态动画来解释复杂概念。为了实现严格的评估，我们还提出了SciVidEval，这是首个针对此任务的综合套件，它结合了多模态内容质量和同步的自动化指标，以及基于视频测验的人类评估来衡量知识转移。大量实验表明，我们的方法在科学视频生成方面显著优于现有商业服务，并达到与人类水平相当的质量。 

---
# HoloGarment: 360° Novel View Synthesis of In-the-Wild Garments 

**Title (ZH)**: 全息服装：野外服饰的360°新型视图合成 

**Authors**: Johanna Karras, Yingwei Li, Yasamin Jafarian, Ira Kemelmacher-Shlizerman  

**Link**: [PDF](https://arxiv.org/pdf/2509.12187)  

**Abstract**: Novel view synthesis (NVS) of in-the-wild garments is a challenging task due significant occlusions, complex human poses, and cloth deformations. Prior methods rely on synthetic 3D training data consisting of mostly unoccluded and static objects, leading to poor generalization on real-world clothing. In this paper, we propose HoloGarment (Hologram-Garment), a method that takes 1-3 images or a continuous video of a person wearing a garment and generates 360° novel views of the garment in a canonical pose. Our key insight is to bridge the domain gap between real and synthetic data with a novel implicit training paradigm leveraging a combination of large-scale real video data and small-scale synthetic 3D data to optimize a shared garment embedding space. During inference, the shared embedding space further enables dynamic video-to-360° NVS through the construction of a garment "atlas" representation by finetuning a garment embedding on a specific real-world video. The atlas captures garment-specific geometry and texture across all viewpoints, independent of body pose or motion. Extensive experiments show that HoloGarment achieves state-of-the-art performance on NVS of in-the-wild garments from images and videos. Notably, our method robustly handles challenging real-world artifacts -- such as wrinkling, pose variation, and occlusion -- while maintaining photorealism, view consistency, fine texture details, and accurate geometry. Visit our project page for additional results: this https URL 

**Abstract (ZH)**: 野外穿着服装的新型视角合成（NVS）是一项具有显著遮挡、复杂人体姿态和服装变形挑战的任务。先前的方法依赖于合成的3D训练数据，这些数据主要由大部分无遮挡和静止的物体组成，导致在真实世界服装上的泛化能力较差。在本文中，我们提出了一种HoloGarment（全息服装）方法，该方法接受一张或多张穿着衣服的人的图像或连续视频，并生成穿着者在标准姿态下的360°新型视角服装图像。我们的关键洞察是，通过结合大规模的现实视频数据和小型的合成3D数据，利用新颖的隐式训练范式优化共享的服装嵌入空间，以桥接现实和合成数据之间的领域差距。在推理过程中，共享的嵌入空间进一步通过细化特定现实世界视频上的服装嵌入来构建“服装地图”表示形式，从而实现动态视频到360°的新型视角合成。该地图捕获了所有视角下特有的服装几何和纹理，与身体姿态或运动无关。大量实验表明，HoloGarment在图像和视频中实现野外穿着服装的新型视角合成上达到了最先进的性能。值得注意的是，我们的方法稳健地处理了现实世界中的挑战性特征——如褶皱、姿态变化和遮挡——同时保持了逼真的外观、视图一致性和精细的纹理细节以及准确的几何结构。访问我们的项目页面获取更多结果：this https URL 

---
# 3DViT-GAT: A Unified Atlas-Based 3D Vision Transformer and Graph Learning Framework for Major Depressive Disorder Detection Using Structural MRI Data 

**Title (ZH)**: 基于3D视图变换器和图学习的统一脑图谱框架：结构MRI数据中重度抑郁障碍检测的3D维特Transformer和图注意力机制模型 

**Authors**: Nojod M. Alotaibi, Areej M. Alhothali, Manar S. Ali  

**Link**: [PDF](https://arxiv.org/pdf/2509.12143)  

**Abstract**: Major depressive disorder (MDD) is a prevalent mental health condition that negatively impacts both individual well-being and global public health. Automated detection of MDD using structural magnetic resonance imaging (sMRI) and deep learning (DL) methods holds increasing promise for improving diagnostic accuracy and enabling early intervention. Most existing methods employ either voxel-level features or handcrafted regional representations built from predefined brain atlases, limiting their ability to capture complex brain patterns. This paper develops a unified pipeline that utilizes Vision Transformers (ViTs) for extracting 3D region embeddings from sMRI data and Graph Neural Network (GNN) for classification. We explore two strategies for defining regions: (1) an atlas-based approach using predefined structural and functional brain atlases, and (2) an cube-based method by which ViTs are trained directly to identify regions from uniformly extracted 3D patches. Further, cosine similarity graphs are generated to model interregional relationships, and guide GNN-based classification. Extensive experiments were conducted using the REST-meta-MDD dataset to demonstrate the effectiveness of our model. With stratified 10-fold cross-validation, the best model obtained 78.98% accuracy, 76.54% sensitivity, 81.58% specificity, 81.58% precision, and 78.98% F1-score. Further, atlas-based models consistently outperformed the cube-based approach, highlighting the importance of using domain-specific anatomical priors for MDD detection. 

**Abstract (ZH)**: 重大抑郁障碍（MDD）是一种广泛存在的心理健康状况，对个体福祉和全球公共卫生产生负面影响。利用结构磁共振成像（sMRI）和深度学习（DL）方法自动检测MDD具有提高诊断准确性和促进早期干预的潜在价值。现有大多数方法要么使用体素级特征，要么基于预定义的大脑atlase构建手工制作的区域表示，限制了其捕捉复杂大脑模式的能力。本文开发了一种统一的管道，利用Vision Transformers（ViTs）从sMRI数据中提取三维区域嵌入，并使用Graph Neural Network（GNN）进行分类。我们探索了两种区域定义策略：（1）基于atlase的方法，使用预定义的结构和功能atlase；（2）基于立方体的方法，其中ViTs直接训练以识别从均匀抽取的3D片段中识别区域。此外，生成余弦相似度图来建模区域间的关系，并指导基于GNN的分类。通过使用REST-meta-MDD数据集进行了广泛实验，以展示我们模型的有效性。在分层10折交叉验证中，最佳模型获得了78.98%的准确率、76.54%的敏感性、81.58%的特异性、81.58%的精确率和78.98%的F1分数。此外，基于atlase的模型始终优于立方体方法，强调了使用针对MDD检测的领域特定解剖先验的重要性。 

---
# U-Mamba2: Scaling State Space Models for Dental Anatomy Segmentation in CBCT 

**Title (ZH)**: U-Mamba2: 扩展状态空间模型在CBCT牙科解剖分割中的应用 

**Authors**: Zhi Qin Tan, Xiatian Zhu, Owen Addison, Yunpeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.12069)  

**Abstract**: Cone-Beam Computed Tomography (CBCT) is a widely used 3D imaging technique in dentistry, providing volumetric information about the anatomical structures of jaws and teeth. Accurate segmentation of these anatomies is critical for clinical applications such as diagnosis and surgical planning, but remains time-consuming and challenging. In this paper, we present U-Mamba2, a new neural network architecture designed for multi-anatomy CBCT segmentation in the context of the ToothFairy3 challenge. U-Mamba2 integrates the Mamba2 state space models into the U-Net architecture, enforcing stronger structural constraints for higher efficiency without compromising performance. In addition, we integrate interactive click prompts with cross-attention blocks, pre-train U-Mamba2 using self-supervised learning, and incorporate dental domain knowledge into the model design to address key challenges of dental anatomy segmentation in CBCT. Extensive experiments, including independent tests, demonstrate that U-Mamba2 is both effective and efficient, securing top 3 places in both tasks of the Toothfairy3 challenge. In Task 1, U-Mamba2 achieved a mean Dice of 0.792, HD95 of 93.19 with the held-out test data, with an average inference time of XX (TBC during the ODIN workshop). In Task 2, U-Mamba2 achieved the mean Dice of 0.852 and HD95 of 7.39 with the held-out test data. The code is publicly available at this https URL. 

**Abstract (ZH)**: Cone-Beam 计算机断层成像（CBCT）在牙科中的应用及其多 Anatomy 分割的 U-Mamba2 神经网络架构 

---
# A Computer Vision Pipeline for Individual-Level Behavior Analysis: Benchmarking on the Edinburgh Pig Dataset 

**Title (ZH)**: 基于爱丁堡猪数据集的个体水平行为分析计算机视觉流程：基准测试 

**Authors**: Haiyu Yang, Enhong Liu, Jennifer Sun, Sumit Sharma, Meike van Leerdam, Sebastien Franceschini, Puchun Niu, Miel Hostens  

**Link**: [PDF](https://arxiv.org/pdf/2509.12047)  

**Abstract**: Animal behavior analysis plays a crucial role in understanding animal welfare, health status, and productivity in agricultural settings. However, traditional manual observation methods are time-consuming, subjective, and limited in scalability. We present a modular pipeline that leverages open-sourced state-of-the-art computer vision techniques to automate animal behavior analysis in a group housing environment. Our approach combines state-of-the-art models for zero-shot object detection, motion-aware tracking and segmentation, and advanced feature extraction using vision transformers for robust behavior recognition. The pipeline addresses challenges including animal occlusions and group housing scenarios as demonstrated in indoor pig monitoring. We validated our system on the Edinburgh Pig Behavior Video Dataset for multiple behavioral tasks. Our temporal model achieved 94.2% overall accuracy, representing a 21.2 percentage point improvement over existing methods. The pipeline demonstrated robust tracking capabilities with 93.3% identity preservation score and 89.3% object detection precision. The modular design suggests potential for adaptation to other contexts, though further validation across species would be required. The open-source implementation provides a scalable solution for behavior monitoring, contributing to precision pig farming and welfare assessment through automated, objective, and continuous analysis. 

**Abstract (ZH)**: 动物行为分析在理解动物福利、健康状态和生产性能中的作用至关重要。然而，传统的手工观察方法耗时、主观且可扩展性有限。我们提出了一种模块化管道，利用开源的先进计算机视觉技术来自动化群体饲养环境中的动物行为分析。我们的方法结合了零样本对象检测、运动感知跟踪和分割的先进模型，以及使用视觉变换器进行高级特征提取的稳健行为识别。该管道解决了包括动物遮挡和群体饲养场景在内的挑战，如在室内猪监测中所示。我们在爱丁堡猪行为视频数据集上对我们的系统进行了多任务验证。我们的时序模型overall accuracy达到94.2%，比现有方法高出21.2个百分点。管道表现出强大的跟踪能力，身份保留得分为93.3%，物体检测精确度为89.3%。模块化设计表明该系统具有适应其他情境的潜力，但需要在其他物种中进行进一步验证。开源实现提供了可扩展的行为监测解决方案，通过自动、客观和连续的分析，推动精准养猪和福利评估。 

---
# Exploring Efficient Open-Vocabulary Segmentation in the Remote Sensing 

**Title (ZH)**: 探索遥感中的高效开放式词汇分割方法 

**Authors**: Bingyu Li, Haocheng Dong, Da Zhang, Zhiyuan Zhao, Junyu Gao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.12040)  

**Abstract**: Open-Vocabulary Remote Sensing Image Segmentation (OVRSIS), an emerging task that adapts Open-Vocabulary Segmentation (OVS) to the remote sensing (RS) domain, remains underexplored due to the absence of a unified evaluation benchmark and the domain gap between natural and RS images. To bridge these gaps, we first establish a standardized OVRSIS benchmark (\textbf{OVRSISBench}) based on widely-used RS segmentation datasets, enabling consistent evaluation across methods. Using this benchmark, we comprehensively evaluate several representative OVS/OVRSIS models and reveal their limitations when directly applied to remote sensing scenarios. Building on these insights, we propose \textbf{RSKT-Seg}, a novel open-vocabulary segmentation framework tailored for remote sensing. RSKT-Seg integrates three key components: (1) a Multi-Directional Cost Map Aggregation (RS-CMA) module that captures rotation-invariant visual cues by computing vision-language cosine similarities across multiple directions; (2) an Efficient Cost Map Fusion (RS-Fusion) transformer, which jointly models spatial and semantic dependencies with a lightweight dimensionality reduction strategy; and (3) a Remote Sensing Knowledge Transfer (RS-Transfer) module that injects pre-trained knowledge and facilitates domain adaptation via enhanced upsampling. Extensive experiments on the benchmark show that RSKT-Seg consistently outperforms strong OVS baselines by +3.8 mIoU and +5.9 mACC, while achieving 2x faster inference through efficient aggregation. Our code is \href{this https URL}{\textcolor{blue}{here}}. 

**Abstract (ZH)**: 开放词汇遥感图像分割：面向遥感领域的新兴任务（OVRSIS）及其评估基准（OVRSISBench）与新型开放词汇分割框架（RSKT-Seg） 

---
# Integrating Prior Observations for Incremental 3D Scene Graph Prediction 

**Title (ZH)**: 集成先验观察以进行增量三维场景图预测 

**Authors**: Marian Renz, Felix Igelbrink, Martin Atzmueller  

**Link**: [PDF](https://arxiv.org/pdf/2509.11895)  

**Abstract**: 3D semantic scene graphs (3DSSG) provide compact structured representations of environments by explicitly modeling objects, attributes, and relationships. While 3DSSGs have shown promise in robotics and embodied AI, many existing methods rely mainly on sensor data, not integrating further information from semantically rich environments. Additionally, most methods assume access to complete scene reconstructions, limiting their applicability in real-world, incremental settings. This paper introduces a novel heterogeneous graph model for incremental 3DSSG prediction that integrates additional, multi-modal information, such as prior observations, directly into the message-passing process. Utilizing multiple layers, the model flexibly incorporates global and local scene representations without requiring specialized modules or full scene reconstructions. We evaluate our approach on the 3DSSG dataset, showing that GNNs enriched with multi-modal information such as semantic embeddings (e.g., CLIP) and prior observations offer a scalable and generalizable solution for complex, real-world environments. The full source code of the presented architecture will be made available at this https URL. 

**Abstract (ZH)**: 3D语义场景图的增量预测异构图模型 

---
# Bridging the Gap Between Sparsity and Redundancy: A Dual-Decoding Framework with Global Context for Map Inference 

**Title (ZH)**: 填补稀疏性和冗余性之间的差距：一种带有全局上下文的双重解码框架用于地图推断 

**Authors**: Yudong Shen, Wenyu Wu, Jiali Mao, Yixiao Tong, Guoping Liu, Chaoya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11731)  

**Abstract**: Trajectory data has become a key resource for automated map in-ference due to its low cost, broad coverage, and continuous availability. However, uneven trajectory density often leads to frag-mented roads in sparse areas and redundant segments in dense regions, posing significant challenges for existing methods. To address these issues, we propose DGMap, a dual-decoding framework with global context awareness, featuring Multi-scale Grid Encoding, Mask-enhanced Keypoint Extraction, and Global Context-aware Relation Prediction. By integrating global semantic context with local geometric features, DGMap improves keypoint detection accuracy to reduce road fragmentation in sparse-trajectory areas. Additionally, the Global Context-aware Relation Prediction module suppresses false connections in dense-trajectory regions by modeling long-range trajectory patterns. Experimental results on three real-world datasets show that DGMap outperforms state-of-the-art methods by 5% in APLS, with notable performance gains on trajectory data from the Didi Chuxing platform 

**Abstract (ZH)**: 轨迹数据已成为自动地图推理的关键资源，由于其低成本、广覆盖和持续可用性。然而，不均匀的轨迹密度往往导致稀疏区域道路断裂和稠密区域冗余路段，给现有方法带来了重大挑战。为了解决这些问题，我们提出DGMap，一种具有全局上下文意识的双解码框架，包含多尺度网格编码、掩码增强关键点提取和全局上下文意识关系预测。通过结合全局语义上下文和局部几何特征，DGMap 提高了关键点检测准确性，从而减少稀疏轨迹区域的道路断裂。此外，全局上下文意识关系预测模块通过建模长距离轨迹模式来抑制稠密轨迹区域中的虚假连接。在三个真实世界的数据集上的实验结果显示，DGMap 在APLS指标上比现有最佳方法提高了5%，特别是在滴滴出行平台的轨迹数据上表现出显著的性能提升。 

---
# Microsurgical Instrument Segmentation for Robot-Assisted Surgery 

**Title (ZH)**: 机器人辅助手术中的微外科器械分割 

**Authors**: Tae Kyeong Jeong, Garam Kim, Juyoun Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.11727)  

**Abstract**: Accurate segmentation of thin structures is critical for microsurgical scene understanding but remains challenging due to resolution loss, low contrast, and class imbalance. We propose Microsurgery Instrument Segmentation for Robotic Assistance(MISRA), a segmentation framework that augments RGB input with luminance channels, integrates skip attention to preserve elongated features, and employs an Iterative Feedback Module(IFM) for continuity restoration across multiple passes. In addition, we introduce a dedicated microsurgical dataset with fine-grained annotations of surgical instruments including thin objects, providing a benchmark for robust evaluation Dataset available at this https URL. Experiments demonstrate that MISRA achieves competitive performance, improving the mean class IoU by 5.37% over competing methods, while delivering more stable predictions at instrument contacts and overlaps. These results position MISRA as a promising step toward reliable scene parsing for computer-assisted and robotic microsurgery. 

**Abstract (ZH)**: 微手术器械分割以增强机器人辅助(MISRA)：一种针对微手术场景理解的细粒度分割框架 

---
# Hierarchical Identity Learning for Unsupervised Visible-Infrared Person Re-Identification 

**Title (ZH)**: 无监督可见光-红外人体重新识别的分层身份学习 

**Authors**: Haonan Shi, Yubin Wang, De Cheng, Lingfeng He, Nannan Wang, Xinbo Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.11587)  

**Abstract**: Unsupervised visible-infrared person re-identification (USVI-ReID) aims to learn modality-invariant image features from unlabeled cross-modal person datasets by reducing the modality gap while minimizing reliance on costly manual annotations. Existing methods typically address USVI-ReID using cluster-based contrastive learning, which represents a person by a single cluster center. However, they primarily focus on the commonality of images within each cluster while neglecting the finer-grained differences among them. To address the limitation, we propose a Hierarchical Identity Learning (HIL) framework. Since each cluster may contain several smaller sub-clusters that reflect fine-grained variations among images, we generate multiple memories for each existing coarse-grained cluster via a secondary clustering. Additionally, we propose Multi-Center Contrastive Learning (MCCL) to refine representations for enhancing intra-modal clustering and minimizing cross-modal discrepancies. To further improve cross-modal matching quality, we design a Bidirectional Reverse Selection Transmission (BRST) mechanism, which establishes reliable cross-modal correspondences by performing bidirectional matching of pseudo-labels. Extensive experiments conducted on the SYSU-MM01 and RegDB datasets demonstrate that the proposed method outperforms existing approaches. The source code is available at: this https URL. 

**Abstract (ZH)**: 无监督可见光-红外人体重识别（USVI-ReID）旨在通过减少模态差距并最小化对昂贵的手动标注依赖来从未标注的跨模态人体数据集中学习模态不变的图像特征。现有方法通常使用基于聚类的对比学习来解决USVI-ReID问题，通过单个聚类中心表示一个人。然而，这些方法主要关注每个聚类内部图像的共同点，而忽略它们之间的微细差异。为了解决这一局限，我们提出了层次身份学习（HIL）框架。由于每个聚类可能包含反映图像微细变化的几个较小的子聚类，我们通过二次聚类为每个现有的粗粒度聚类生成多个记忆。此外，我们提出了多中心对比学习（MCCL）以细化表示，增强同模态聚类并最小化跨模态差异。为进一步提高跨模态匹配质量，我们设计了一种双向反向选择传输（BRST）机制，通过双向匹配伪标签建立可靠的跨模态对应关系。在SYSU-MM01和RegDB数据集上的广泛实验表明，所提出的方法优于现有方法。源代码可在以下链接获取：this <https://github.com/...>。 

---
# Promoting Shape Bias in CNNs: Frequency-Based and Contrastive Regularization for Corruption Robustness 

**Title (ZH)**: 基于频率和对比正则化促进CNN的形状偏见以增强对抗扰动的鲁棒性 

**Authors**: Robin Narsingh Ranabhat, Longwei Wang, Amit Kumar Patel, KC santosh  

**Link**: [PDF](https://arxiv.org/pdf/2509.11355)  

**Abstract**: Convolutional Neural Networks (CNNs) excel at image classification but remain vulnerable to common corruptions that humans handle with ease. A key reason for this fragility is their reliance on local texture cues rather than global object shapes -- a stark contrast to human perception. To address this, we propose two complementary regularization strategies designed to encourage shape-biased representations and enhance robustness. The first introduces an auxiliary loss that enforces feature consistency between original and low-frequency filtered inputs, discouraging dependence on high-frequency textures. The second incorporates supervised contrastive learning to structure the feature space around class-consistent, shape-relevant representations. Evaluated on the CIFAR-10-C benchmark, both methods improve corruption robustness without degrading clean accuracy. Our results suggest that loss-level regularization can effectively steer CNNs toward more shape-aware, resilient representations. 

**Abstract (ZH)**: 卷积神经网络（CNNs）在图像分类方面表现出色，但仍然容易受到人类可以轻松处理的常见 corruption 的影响。这一脆弱性的一个关键原因是它们依赖局部纹理线索而非全局对象形状——这与人类感知形成了鲜明对比。为了应对这一挑战，我们提出了两种互补的正则化策略，旨在促进形状偏向的表示并提高鲁棒性。第一种策略引入了辅助损失，强制原始输入和低频过滤输入之间的特征一致性，从而减少对高频纹理的依赖。第二种策略结合了监督对比学习，以构建围绕类一致且形状相关表示的特征空间。在 CIFAR-10-C 基准上评估表明，这两种方法在不牺牲干净准确性的情况下提高了对 corruption 的鲁棒性。我们的结果表明，损失层面的正则化可以有效地引导 CNNs 向更注重形状、更具鲁棒性的表示方向发展。 

---
# MIS-LSTM: Multichannel Image-Sequence LSTM for Sleep Quality and Stress Prediction 

**Title (ZH)**: MIS-LSTM：多通道图像序列LSTM在睡眠质量与压力预测中的应用 

**Authors**: Seongwan Park, Jieun Woo, Siheon Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11232)  

**Abstract**: This paper presents MIS-LSTM, a hybrid framework that joins CNN encoders with an LSTM sequence model for sleep quality and stress prediction at the day level from multimodal lifelog data. Continuous sensor streams are first partitioned into N-hour blocks and rendered as multi-channel images, while sparse discrete events are encoded with a dedicated 1D-CNN. A Convolutional Block Attention Module fuses the two modalities into refined block embeddings, which an LSTM then aggregates to capture long-range temporal dependencies. To further boost robustness, we introduce UALRE, an uncertainty-aware ensemble that overrides lowconfidence majority votes with high-confidence individual predictions. Experiments on the 2025 ETRI Lifelog Challenge dataset show that Our base MISLSTM achieves Macro-F1 0.615; with the UALRE ensemble, the score improves to 0.647, outperforming strong LSTM, 1D-CNN, and CNN baselines. Ablations confirm (i) the superiority of multi-channel over stacked-vertical imaging, (ii) the benefit of a 4-hour block granularity, and (iii) the efficacy of modality-specific discrete encoding. 

**Abstract (ZH)**: 本文提出了一种将CNN编码器与LSTM序列模型结合的混合框架MIS-LSTM，用于从多模态日志数据中按日预测睡眠质量和压力。连续传感器流被分割成N小时块并表示为多通道图像，稀疏离散事件则通过专用的1D-CNN进行编码。卷积块注意力模块将两种模态融合成精细的块嵌入，随后LSTM聚合这些嵌入以捕获长时序依赖关系。为了进一步增强鲁棒性，引入了UALRE（不确定性意识集成），它用高置信度的个体预测覆盖低置信度的多数投票。在2025 ETRI日志挑战数据集上的实验表明，我们的基线MISLSTM实现了宏F1分数0.615；加入UALRE集成后，分数提升到0.647，优于强大的LSTM、1D-CNN和CNN基线。消融实验确认了(i) 多通道优于堆叠垂直成像、(ii) 4小时块粒度的优势以及(iii) 模态特定离散编码的有效性。 

---
# Geometrically Constrained and Token-Based Probabilistic Spatial Transformers 

**Title (ZH)**: 几何约束和基于token的概率空间变换器 

**Authors**: Johann Schmidt, Sebastian Stober  

**Link**: [PDF](https://arxiv.org/pdf/2509.11218)  

**Abstract**: Fine-grained visual classification (FGVC) remains highly sensitive to geometric variability, where objects appear under arbitrary orientations, scales, and perspective distortions. While equivariant architectures address this issue, they typically require substantial computational resources and restrict the hypothesis space. We revisit Spatial Transformer Networks (STNs) as a canonicalization tool for transformer-based vision pipelines, emphasizing their flexibility, backbone-agnostic nature, and lack of architectural constraints. We propose a probabilistic, component-wise extension that improves robustness. Specifically, we decompose affine transformations into rotation, scaling, and shearing, and regress each component under geometric constraints using a shared localization encoder. To capture uncertainty, we model each component with a Gaussian variational posterior and perform sampling-based canonicalization during inference.A novel component-wise alignment loss leverages augmentation parameters to guide spatial alignment. Experiments on challenging moth classification benchmarks demonstrate that our method consistently improves robustness compared to other STNs. 

**Abstract (ZH)**: 细粒度视觉分类（FGVC）仍高度敏感于几何变异，其中对象以任意方向、尺度和视角变形出现。虽然_equivariant_架构可以解决这一问题，但它们通常需要大量的计算资源并限制假设空间。我们重新审视Spatial Transformer Networks（STN）作为基于转换器的视觉管道的标准规范化工具，强调其灵活性、后端无关性和缺乏架构约束。我们提出了一种概率性的组件级扩展，以提高鲁棒性。具体而言，我们将仿射变换分解为旋转、缩放和剪切，并在几何约束下使用共享定位编码器回归每个组件。为了捕获不确定性，我们用高斯变分后验概率建模每个组件，并在推断过程中通过采样进行规范化。一种新颖的组件级对齐损失利用增强参数来引导空间对齐。在具有挑战性的蛾类分类基准测试中，我们的方法在鲁棒性方面始终优于其他STN。 

---
# PanoLora: Bridging Perspective and Panoramic Video Generation with LoRA Adaptation 

**Title (ZH)**: PanoLora：视角与全景视频生成的LoRA适应性桥梁 

**Authors**: Zeyu Dong, Yuyang Yin, Yuqi Li, Eric Li, Hao-Xiang Guo, Yikai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11092)  

**Abstract**: Generating high-quality 360° panoramic videos remains a significant challenge due to the fundamental differences between panoramic and traditional perspective-view projections. While perspective videos rely on a single viewpoint with a limited field of view, panoramic content requires rendering the full surrounding environment, making it difficult for standard video generation models to adapt. Existing solutions often introduce complex architectures or large-scale training, leading to inefficiency and suboptimal results. Motivated by the success of Low-Rank Adaptation (LoRA) in style transfer tasks, we propose treating panoramic video generation as an adaptation problem from perspective views. Through theoretical analysis, we demonstrate that LoRA can effectively model the transformation between these projections when its rank exceeds the degrees of freedom in the task. Our approach efficiently fine-tunes a pretrained video diffusion model using only approximately 1,000 videos while achieving high-quality panoramic generation. Experimental results demonstrate that our method maintains proper projection geometry and surpasses previous state-of-the-art approaches in visual quality, left-right consistency, and motion diversity. 

**Abstract (ZH)**: 生成高质量的360°全景视频仍然是一个重大挑战，因为全景和传统视角投影之间存在根本差异。虽然传统视角视频依赖于单一视角和有限的视场，全景内容需要渲染整个环境，使得标准视频生成模型难以适应。现有解决方案往往引入复杂架构或大规模训练，导致效率低下且结果不佳。受Low-Rank Adaptation (LoRA)在风格转换任务中成功应用的启发，我们提出将全景视频生成视为从视角视图的适应问题。通过理论分析，我们证明当LoRA的秩超过任务自由度时，它可以有效地建模这两种投影之间的转换。我们的方法仅使用约1,000个视频对预训练的视频扩散模型进行高效微调，从而实现高質量的全景生成。实验结果表明，我们的方法保持了正确的投影几何结构，并在视觉质量、左右一致性以及运动多样性方面超越了先前的最佳方法。 

---
# An Advanced Convolutional Neural Network for Bearing Fault Diagnosis under Limited Data 

**Title (ZH)**: 基于有限数据的先进卷积神经网络轴承故障诊断 

**Authors**: Shengke Sun, Shuzhen Han, Ziqian Luan, Xinghao Qin, Jiao Yin, Zhanshan Zhao, Jinli Cao, Hua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11053)  

**Abstract**: In the area of bearing fault diagnosis, deep learning (DL) methods have been widely used recently. However, due to the high cost or privacy concerns, high-quality labeled data are scarce in real world scenarios. While few-shot learning has shown promise in addressing data scarcity, existing methods still face significant limitations in this domain. Traditional data augmentation techniques often suffer from mode collapse and generate low-quality samples that fail to capture the diversity of bearing fault patterns. Moreover, conventional convolutional neural networks (CNNs) with local receptive fields makes them inadequate for extracting global features from complex vibration signals. Additionally, existing methods fail to model the intricate relationships between limited training samples. To solve these problems, we propose an advanced data augmentation and contrastive fourier convolution framework (DAC-FCF) for bearing fault diagnosis under limited data. Firstly, a novel conditional consistent latent representation and reconstruction generative adversarial network (CCLR-GAN) is proposed to generate more diverse data. Secondly, a contrastive learning based joint optimization mechanism is utilized to better model the relations between the available training data. Finally, we propose a 1D fourier convolution neural network (1D-FCNN) to achieve a global-aware of the input data. Experiments demonstrate that DAC-FCF achieves significant improvements, outperforming baselines by up to 32\% on case western reserve university (CWRU) dataset and 10\% on a self-collected test bench. Extensive ablation experiments prove the effectiveness of the proposed components. Thus, the proposed DAC-FCF offers a promising solution for bearing fault diagnosis under limited data. 

**Abstract (ZH)**: 在轴承故障诊断领域，深度学习方法最近得到了广泛应用。然而，由于成本高或隐私问题，现实场景中高质量的标注数据稀缺。虽然少样本学习在解决数据稀缺性方面前景广阔，但现有方法在此领域仍面临诸多局限。传统数据增强技术常常遭受模式崩溃的问题，生成的样本质量较低，无法捕获轴承故障模式的多样性。此外，传统的具有局部感受野的卷积神经网络（CNN）对于提取复杂振动信号的全局特征不足。更重要的是，现有方法无法有效建模有限训练样本之间的复杂关系。为了解决这些问题，我们提出了一种先进的数据增强和对比傅里叶卷积框架（DAC-FCF）以应对有限数据下的轴承故障诊断。首先，提出了一种新颖的条件一致潜在表示和重建生成对抗网络（CCLR-GAN）以生成更多样化的数据。其次，利用基于对比学习的联合优化机制更好地建模可用训练数据之间的关系。最后，提出了一种1D傅里叶卷积神经网络（1D-FCNN）以实现对输入数据的整体感知。实验表明，DAC-FCF在Case Western Reserve University（CWRU）数据集和自收集测试台上的表现分别超过了基线32%和10%。大量消融实验证明了所提组件的有效性。因此，提出的DAC-FCF为有限数据下的轴承故障诊断提供了一个有前景的解决方案。 

---
# ToMA: Token Merge with Attention for Image Generation with Diffusion Models 

**Title (ZH)**: ToMA: 基于注意力的token合并方法在扩散模型图像生成中的应用 

**Authors**: Wenbo Lu, Shaoyi Zheng, Yuxuan Xia, Shengjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10918)  

**Abstract**: Diffusion models excel in high-fidelity image generation but face scalability limits due to transformers' quadratic attention complexity. Plug-and-play token reduction methods like ToMeSD and ToFu reduce FLOPs by merging redundant tokens in generated images but rely on GPU-inefficient operations (e.g., sorting, scattered writes), introducing overheads that negate theoretical speedups when paired with optimized attention implementations (e.g., FlashAttention). To bridge this gap, we propose Token Merge with Attention (ToMA), an off-the-shelf method that redesigns token reduction for GPU-aligned efficiency, with three key contributions: 1) a reformulation of token merge as a submodular optimization problem to select diverse tokens; 2) merge/unmerge as an attention-like linear transformation via GPU-friendly matrix operations; and 3) exploiting latent locality and sequential redundancy (pattern reuse) to minimize overhead. ToMA reduces SDXL/Flux generation latency by 24%/23%, respectively (with DINO $\Delta < 0.07$), outperforming prior methods. This work bridges the gap between theoretical and practical efficiency for transformers in diffusion. 

**Abstract (ZH)**: Token Merge with Attention (ToMA)：一种面向GPU优化的 tokens 融合方法 

---
# A Comparison and Evaluation of Fine-tuned Convolutional Neural Networks to Large Language Models for Image Classification and Segmentation of Brain Tumors on MRI 

**Title (ZH)**: 细调卷积神经网络与大型语言模型在MRI脑肿瘤图像分类与分割中的对比与评估 

**Authors**: Felicia Liu, Jay J. Yoo, Farzad Khalvati  

**Link**: [PDF](https://arxiv.org/pdf/2509.10683)  

**Abstract**: Large Language Models (LLMs) have shown strong performance in text-based healthcare tasks. However, their utility in image-based applications remains unexplored. We investigate the effectiveness of LLMs for medical imaging tasks, specifically glioma classification and segmentation, and compare their performance to that of traditional convolutional neural networks (CNNs). Using the BraTS 2020 dataset of multi-modal brain MRIs, we evaluated a general-purpose vision-language LLM (LLaMA 3.2 Instruct) both before and after fine-tuning, and benchmarked its performance against custom 3D CNNs. For glioma classification (Low-Grade vs. High-Grade), the CNN achieved 80% accuracy and balanced precision and recall. The general LLM reached 76% accuracy but suffered from a specificity of only 18%, often misclassifying Low-Grade tumors. Fine-tuning improved specificity to 55%, but overall performance declined (e.g., accuracy dropped to 72%). For segmentation, three methods - center point, bounding box, and polygon extraction, were implemented. CNNs accurately localized gliomas, though small tumors were sometimes missed. In contrast, LLMs consistently clustered predictions near the image center, with no distinction of glioma size, location, or placement. Fine-tuning improved output formatting but failed to meaningfully enhance spatial accuracy. The bounding polygon method yielded random, unstructured outputs. Overall, CNNs outperformed LLMs in both tasks. LLMs showed limited spatial understanding and minimal improvement from fine-tuning, indicating that, in their current form, they are not well-suited for image-based tasks. More rigorous fine-tuning or alternative training strategies may be needed for LLMs to achieve better performance, robustness, and utility in the medical space. 

**Abstract (ZH)**: 大型语言模型在基于图像的医疗影像任务中的有效性及其与传统卷积神经网络的比较 

---
# Spectral and Rhythm Features for Audio Classification with Deep Convolutional Neural Networks 

**Title (ZH)**: 基于深卷积神经网络的音频分类的光谱和节奏特征 

**Authors**: Friedrich Wolf-Monheim  

**Link**: [PDF](https://arxiv.org/pdf/2410.06927)  

**Abstract**: Convolutional neural networks (CNNs) are widely used in computer vision. They can be used not only for conventional digital image material to recognize patterns, but also for feature extraction from digital imagery representing spectral and rhythm features extracted from time-domain digital audio signals for the acoustic classification of sounds. Different spectral and rhythm feature representations like mel-scaled spectrograms, mel-frequency cepstral coefficients (MFCCs), cyclic tempograms, short-time Fourier transform (STFT) chromagrams, constant-Q transform (CQT) chromagrams and chroma energy normalized statistics (CENS) chromagrams are investigated in terms of the audio classification performance using a deep convolutional neural network. It can be clearly shown that the mel-scaled spectrograms and the mel-frequency cepstral coefficients (MFCCs) perform significantly better than the other spectral and rhythm features investigated in this research for audio classification tasks using deep CNNs. The experiments were carried out with the aid of the ESC-50 dataset with 2,000 labeled environmental audio recordings. 

**Abstract (ZH)**: 卷积神经网络（CNNs）在计算机视觉中广泛应用。它们不仅可以用于模式识别的传统数字图像材料，还可以用于从代表时域数字音频信号的光谱和节律特征的数字图像中提取特征，以进行声音的声学分类。研究表明，在使用深度卷积神经网络进行音频分类时，梅尔标度频谱图和梅尔频率倒谱系数（MFCCs）的表现显著优于其他研究中探究的光谱和节律特征。实验使用包含2000个标记环境音频记录的ESC-50数据集进行。 

---
