# BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion 

**Title (ZH)**: 超越仿制：从运动追踪到多功能类人控制的引导扩散 

**Authors**: Takara E. Truong, Qiayuan Liao, Xiaoyu Huang, Guy Tevet, C. Karen Liu, Koushil Sreenath  

**Link**: [PDF](https://arxiv.org/pdf/2508.08241)  

**Abstract**: Learning skills from human motions offers a promising path toward generalizable policies for whole-body humanoid control, yet two key cornerstones are missing: (1) a high-quality motion tracking framework that faithfully transforms large-scale kinematic references into robust and extremely dynamic motions on real hardware, and (2) a distillation approach that can effectively learn these motion primitives and compose them to solve downstream tasks. We address these gaps with BeyondMimic, the first real-world framework to learn from human motions for versatile and naturalistic humanoid control via guided diffusion. Our framework provides a motion tracking pipeline capable of challenging skills such as jumping spins, sprinting, and cartwheels with state-of-the-art motion quality. Moving beyond mimicking existing motions and synthesize novel ones, we further introduce a unified diffusion policy that enables zero-shot task-specific control at test time using simple cost functions. Deployed on hardware, BeyondMimic performs diverse tasks at test time, including waypoint navigation, joystick teleoperation, and obstacle avoidance, bridging sim-to-real motion tracking and flexible synthesis of human motion primitives for whole-body control. this https URL. 

**Abstract (ZH)**: 从人类动作中学习技能为全身类人机器人控制提供了具有前景的道路，但缺少两个关键要素：（1）一个高质量的动作跟踪框架，能够忠实地将大规模的动力学参考转化为在真实硬件上稳健且极其动态的动作；（2）一种有效学习这些动作基元并将其组成以解决下游任务的精练方法。我们通过引导扩散的BeyondMimic框架填补了这些空白，这是首个用于通过人类动作实现多功能自然类人机器人控制的现实世界框架。我们的框架提供了一个能够处理跳跃旋转、冲刺和侧手翻等具有挑战性技能的动作跟踪管道，其动作质量处于业界领先水平。BeyondMimic超越了模仿现有动作并进一步合成新型动作，介绍了一种统一的扩散策略，能够在测试时使用简单的成本函数实现零样本的任务特定控制。部署在硬件上，BeyondMimic能够在测试时执行多种任务，包括航点导航、操纵杆远程操控和障碍物回避，从而桥接了从仿真到现实的动作跟踪和对全身控制的人类动作基元的灵活合成。 

---
# ODYSSEY: Open-World Quadrupeds Exploration and Manipulation for Long-Horizon Tasks 

**Title (ZH)**: ODYSSEY: 开放世界四足机器人的长期任务探索与操作 

**Authors**: Kaijun Wang, Liqin Lu, Mingyu Liu, Jianuo Jiang, Zeju Li, Bolin Zhang, Wancai Zheng, Xinyi Yu, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.08240)  

**Abstract**: Language-guided long-horizon mobile manipulation has long been a grand challenge in embodied semantic reasoning, generalizable manipulation, and adaptive locomotion. Three fundamental limitations hinder progress: First, although large language models have improved spatial reasoning and task planning through semantic priors, existing implementations remain confined to tabletop scenarios, failing to address the constrained perception and limited actuation ranges of mobile platforms. Second, current manipulation strategies exhibit insufficient generalization when confronted with the diverse object configurations encountered in open-world environments. Third, while crucial for practical deployment, the dual requirement of maintaining high platform maneuverability alongside precise end-effector control in unstructured settings remains understudied.
In this work, we present ODYSSEY, a unified mobile manipulation framework for agile quadruped robots equipped with manipulators, which seamlessly integrates high-level task planning with low-level whole-body control. To address the challenge of egocentric perception in language-conditioned tasks, we introduce a hierarchical planner powered by a vision-language model, enabling long-horizon instruction decomposition and precise action execution. At the control level, our novel whole-body policy achieves robust coordination across challenging terrains. We further present the first benchmark for long-horizon mobile manipulation, evaluating diverse indoor and outdoor scenarios. Through successful sim-to-real transfer, we demonstrate the system's generalization and robustness in real-world deployments, underscoring the practicality of legged manipulators in unstructured environments. Our work advances the feasibility of generalized robotic assistants capable of complex, dynamic tasks. Our project page: this https URL 

**Abstract (ZH)**: 语言引导的长期 horizon 移动操作在嵌体语义推理、可泛化的操作以及自适应运动中的长期挑战：ODYSSEY——统一的具备操作臂的四足机器人操作框架 

---
# Verti-Arena: A Controllable and Standardized Indoor Testbed for Multi-Terrain Off-Road Autonomy 

**Title (ZH)**: Verti-Arena：一种可控且标准化的室内多地形越野自主测试平台 

**Authors**: Haiyue Chen, Aniket Datar, Tong Xu, Francesco Cancelliere, Harsh Rangwala, Madhan Balaji Rao, Daeun Song, David Eichinger, Xuesu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.08226)  

**Abstract**: Off-road navigation is an important capability for mobile robots deployed in environments that are inaccessible or dangerous to humans, such as disaster response or planetary exploration. Progress is limited due to the lack of a controllable and standardized real-world testbed for systematic data collection and validation. To fill this gap, we introduce Verti-Arena, a reconfigurable indoor facility designed specifically for off-road autonomy. By providing a repeatable benchmark environment, Verti-Arena supports reproducible experiments across a variety of vertically challenging terrains and provides precise ground truth measurements through onboard sensors and a motion capture system. Verti-Arena also supports consistent data collection and comparative evaluation of algorithms in off-road autonomy research. We also develop a web-based interface that enables research groups worldwide to remotely conduct standardized off-road autonomy experiments on Verti-Arena. 

**Abstract (ZH)**: 户外导航是部署在人类难以进入或危险环境中的移动机器人的一项重要能力，如灾害响应或行星探索。由于缺乏一个可控且标准化的实地测试平台来进行系统的数据收集和验证，进展受限。为填补这一空白，我们引入了Verti-Arena，一种专门为户外自主导航设计的可重新配置室内设施。通过提供一个可重复基准环境，Verti-Arena 支持在各种垂直挑战性地形上进行可重复的实验，并通过机载传感器和动作捕捉系统提供精确的地面真值测量。Verti-Arena 还支持户外自主导航研究中算法的一致数据收集和比较评估。我们还开发了一个网页界面，使世界各地的研究团队能够远程在Verti-Arena上进行标准化的户外自主导航实验。 

---
# COMponent-Aware Pruning for Accelerated Control Tasks in Latent Space Models 

**Title (ZH)**: 面向组件意识剪枝的潜在空间模型中加速控制任务方法 

**Authors**: Ganesh Sundaram, Jonas Ulmen, Amjad Haider, Daniel Görges  

**Link**: [PDF](https://arxiv.org/pdf/2508.08144)  

**Abstract**: The rapid growth of resource-constrained mobile platforms, including mobile robots, wearable systems, and Internet-of-Things devices, has increased the demand for computationally efficient neural network controllers (NNCs) that can operate within strict hardware limitations. While deep neural networks (DNNs) demonstrate superior performance in control applications, their substantial computational complexity and memory requirements present significant barriers to practical deployment on edge devices. This paper introduces a comprehensive model compression methodology that leverages component-aware structured pruning to determine the optimal pruning magnitude for each pruning group, ensuring a balance between compression and stability for NNC deployment. Our approach is rigorously evaluated on Temporal Difference Model Predictive Control (TD-MPC), a state-of-the-art model-based reinforcement learning algorithm, with a systematic integration of mathematical stability guarantee properties, specifically Lyapunov criteria. The key contribution of this work lies in providing a principled framework for determining the theoretical limits of model compression while preserving controller stability. Experimental validation demonstrates that our methodology successfully reduces model complexity while maintaining requisite control performance and stability characteristics. Furthermore, our approach establishes a quantitative boundary for safe compression ratios, enabling practitioners to systematically determine the maximum permissible model reduction before violating critical stability properties, thereby facilitating the confident deployment of compressed NNCs in resource-limited environments. 

**Abstract (ZH)**: 资源受限的移动平台（包括移动机器人、穿戴系统和物联网设备）的快速增长增加了对计算效率高的神经网络控制器（NNCs）的需求，这些控制器能够在严格硬件限制下运行。虽然深度神经网络（DNNs）在控制应用中表现出色，但它们较大的计算复杂度和内存需求为其实现边缘设备部署带来了显著障碍。本文提出了一种综合的模型压缩方法，利用组件感知结构化剪枝来确定每个剪枝组的最佳剪枝幅度，确保压缩与稳定性的平衡以供NNC部署。我们的方法在Temporal Difference Model Predictive Control（TD-MPC），一种先进的基于模型的强化学习算法，上进行了严格评估，并系统地整合了数学稳定性保证特性，特别是李雅普un诺夫准则。本文的关键贡献在于提供了一个原则性的框架，以确定模型压缩的理论极限同时保持控制器的稳定性。实验验证表明，我们的方法成功地减少了模型的复杂性，同时维持了必要的控制性能和稳定性特征。此外，我们的方法设定了一个安全压缩比的定量界限，使实践者能够系统地确定在违反关键稳定性属性之前的最大允许模型缩减程度，从而促进在资源受限环境中压缩NNC的可靠部署。 

---
# AimBot: A Simple Auxiliary Visual Cue to Enhance Spatial Awareness of Visuomotor Policies 

**Title (ZH)**: AimBot: 一种简单的眼动辅助视觉提示以增强运动知觉策略的空间意识 

**Authors**: Yinpei Dai, Jayjun Lee, Yichi Zhang, Ziqiao Ma, Jed Yang, Amir Zadeh, Chuan Li, Nima Fazeli, Joyce Chai  

**Link**: [PDF](https://arxiv.org/pdf/2508.08113)  

**Abstract**: In this paper, we propose AimBot, a lightweight visual augmentation technique that provides explicit spatial cues to improve visuomotor policy learning in robotic manipulation. AimBot overlays shooting lines and scope reticles onto multi-view RGB images, offering auxiliary visual guidance that encodes the end-effector's state. The overlays are computed from depth images, camera extrinsics, and the current end-effector pose, explicitly conveying spatial relationships between the gripper and objects in the scene. AimBot incurs minimal computational overhead (less than 1 ms) and requires no changes to model architectures, as it simply replaces original RGB images with augmented counterparts. Despite its simplicity, our results show that AimBot consistently improves the performance of various visuomotor policies in both simulation and real-world settings, highlighting the benefits of spatially grounded visual feedback. 

**Abstract (ZH)**: 在本文中，我们提出了一种轻量级视觉增强技术AimBot，该技术提供了明确的空间线索以改善机器人操作中的视动策略学习。AimBot将射击线和瞄准镜刻度叠加在多视角RGB图像上，提供辅助的视觉指导，编码末端执行器的状态。这些叠加是根据深度图像、相机外参和当前末端执行器的姿态计算得出的，明确传达了指尖与场景中物体之间的空间关系。AimBot计算开销极小（少于1 ms），无需更改模型架构，只需用增强版本的图像替换原始RGB图像即可。尽管结构简单，我们的结果显示AimBot在仿真和实际应用场景中一致提高了各种视动策略的性能，突出了基于空间的视觉反馈的优势。 

---
# Capsizing-Guided Trajectory Optimization for Autonomous Navigation with Rough Terrain 

**Title (ZH)**: 翻覆导向的轨迹优化以实现粗糙地形条件下的自主导航 

**Authors**: Wei Zhang, Yinchuan Wang, Wangtao Lu, Pengyu Zhang, Xiang Zhang, Yue Wang, Chaoqun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08108)  

**Abstract**: It is a challenging task for ground robots to autonomously navigate in harsh environments due to the presence of non-trivial obstacles and uneven terrain. This requires trajectory planning that balances safety and efficiency. The primary challenge is to generate a feasible trajectory that prevents robot from tip-over while ensuring effective navigation. In this paper, we propose a capsizing-aware trajectory planner (CAP) to achieve trajectory planning on the uneven terrain. The tip-over stability of the robot on rough terrain is analyzed. Based on the tip-over stability, we define the traversable orientation, which indicates the safe range of robot orientations. This orientation is then incorporated into a capsizing-safety constraint for trajectory optimization. We employ a graph-based solver to compute a robust and feasible trajectory while adhering to the capsizing-safety constraint. Extensive simulation and real-world experiments validate the effectiveness and robustness of the proposed method. The results demonstrate that CAP outperforms existing state-of-the-art approaches, providing enhanced navigation performance on uneven terrains. 

**Abstract (ZH)**: 基于翻覆感知的不平地形轨迹规划方法 

---
# Aerial Target Encirclement and Interception with Noisy Range Observations 

**Title (ZH)**: 基于噪声距离观测的空中目标围捕与拦截 

**Authors**: Fen Liu, Shenghai Yuan, Thien-Minh Nguyen, Wei Meng, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.08046)  

**Abstract**: This paper proposes a strategy to encircle and intercept a non-cooperative aerial point-mass moving target by leveraging noisy range measurements for state estimation. In this approach, the guardians actively ensure the observability of the target by using an anti-synchronization (AS), 3D ``vibrating string" trajectory, which enables rapid position and velocity estimation based on the Kalman filter. Additionally, a novel anti-target controller is designed for the guardians to enable adaptive transitions from encircling a protected target to encircling, intercepting, and neutralizing a hostile target, taking into consideration the input constraints of the guardians. Based on the guaranteed uniform observability, the exponentially bounded stability of the state estimation error and the convergence of the encirclement error are rigorously analyzed. Simulation results and real-world UAV experiments are presented to further validate the effectiveness of the system design. 

**Abstract (ZH)**: 本文提出了一种策略，通过利用噪声范围测量进行状态估计，来包围和拦截一个非合作空中点目标。在此方法中，守护者主动确保目标的可观测性，采用反同步（AS）的3D“振动弦”轨迹，从而基于卡尔曼滤波实现快速的位置和速度估计。此外，设计了一种新型反目标控制器，使守护者能够适应地从包围受保护目标转变为包围、拦截和消除敌对目标，并考虑守护者的输入约束。基于保证的均匀可观测性，严格分析了状态估计误差的指数有界稳定性和包围误差的收敛性。仿真结果和实际无人机实验进一步验证了系统设计的有效性。 

---
# PCHands: PCA-based Hand Pose Synergy Representation on Manipulators with N-DoF 

**Title (ZH)**: PCHands: 基于PCA的手部姿态协同表示在N-DoF manipulator上 

**Authors**: En Yen Puang, Federico Ceola, Giulia Pasquale, Lorenzo Natale  

**Link**: [PDF](https://arxiv.org/pdf/2508.07945)  

**Abstract**: We consider the problem of learning a common representation for dexterous manipulation across manipulators of different morphologies. To this end, we propose PCHands, a novel approach for extracting hand postural synergies from a large set of manipulators. We define a simplified and unified description format based on anchor positions for manipulators ranging from 2-finger grippers to 5-finger anthropomorphic hands. This enables learning a variable-length latent representation of the manipulator configuration and the alignment of the end-effector frame of all manipulators. We show that it is possible to extract principal components from this latent representation that is universal across manipulators of different structures and degrees of freedom. To evaluate PCHands, we use this compact representation to encode observation and action spaces of control policies for dexterous manipulation tasks learned with RL. In terms of learning efficiency and consistency, the proposed representation outperforms a baseline that learns the same tasks in joint space. We additionally show that PCHands performs robustly in RL from demonstration, when demonstrations are provided from a different manipulator. We further support our results with real-world experiments that involve a 2-finger gripper and a 4-finger anthropomorphic hand. Code and additional material are available at this https URL. 

**Abstract (ZH)**: 我们考虑不同类型形态操纵器之间学习通用表示的问题。为此，我们提出了一种新的方法PCHands，用于从大量不同形态的操纵器中提取手部姿态协同模式。我们基于锚点位置定义了一种简化且统一的描述格式，涵盖了从双指夹爪到五指类人手的各类操纵器。这使得我们能够学习操纵器配置的可变长度隐空间表示以及所有操纵器末端执行器坐标系的对齐方式。我们展示了可以从这种隐空间表示中提取出适用于不同结构和自由度操纵器的通用主成分。为了评估PCHands，我们使用这种紧凑的表示来编码使用RL学习出的灵巧操作任务的观察空间和动作空间。在学习效率和一致性方面，所提出的表示方法优于在关节空间中学习相同任务的基线方法。此外，我们展示了当演示来自不同类型的操纵器时，PCHands在RL中的鲁棒性。我们还通过涉及双指夹爪和四指类人手的真实世界实验进一步支持了我们的结果。更多代码和辅助材料请访问以下链接：这个 https URL。 

---
# MolmoAct: Action Reasoning Models that can Reason in Space 

**Title (ZH)**: MolmoAct: 可以进行空间推理的动作推理模型 

**Authors**: Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, Winson Han, Wilbert Pumacay, Angelica Wu, Rose Hendrix, Karen Farley, Eli VanderBilt, Ali Farhadi, Dieter Fox, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2508.07917)  

**Abstract**: Reasoning is central to purposeful action, yet most robotic foundation models map perception and instructions directly to control, which limits adaptability, generalization, and semantic grounding. We introduce Action Reasoning Models (ARMs), a class of vision-language-action models that integrate perception, planning, and control through a structured three-stage pipeline. Our model, MolmoAct, encodes observations and instructions into depth-aware perception tokens, generates mid-level spatial plans as editable trajectory traces, and predicts precise low-level actions, enabling explainable and steerable behavior. MolmoAct-7B-D achieves strong performance across simulation and real-world settings: 70.5% zero-shot accuracy on SimplerEnv Visual Matching tasks, surpassing closed-source Pi-0 and GR00T N1; 86.6% average success on LIBERO, including an additional 6.3% gain over ThinkAct on long-horizon tasks; and in real-world fine-tuning, an additional 10% (single-arm) and an additional 22.7% (bimanual) task progression over Pi-0-FAST. It also outperforms baselines by an additional 23.3% on out-of-distribution generalization and achieves top human-preference scores for open-ended instruction following and trajectory steering. Furthermore, we release, for the first time, the MolmoAct Dataset -- a mid-training robot dataset comprising over 10,000 high quality robot trajectories across diverse scenarios and tasks. Training with this dataset yields an average 5.5% improvement in general performance over the base model. We release all model weights, training code, our collected dataset, and our action reasoning dataset, establishing MolmoAct as both a state-of-the-art robotics foundation model and an open blueprint for building ARMs that transform perception into purposeful action through structured reasoning. Blogpost: this https URL 

**Abstract (ZH)**: 行动推理是富有目的行动的核心，但大多数机器人基础模型直接将感知和指令映射到控制，这限制了其适应性、泛化能力和语义 grounding。我们引入了行动推理模型（ARMs），这是一种通过结构化三阶段管道整合感知、规划和控制的视觉-语言-行动模型类别。我们的模型 MolmoAct 将观察和指令编码为深度感知标记，生成可编辑的空间计划轨迹，并预测精确的低级动作，从而实现可解释和可控的行为。MolmoAct-7B-D 在模拟和真实世界环境中均表现出强大的性能：在 SimplerEnv Visual Matching 任务中实现 70.5% 的零样本准确率，超过闭源 Pi-0 和 GR00T N1；在 LIBERO 中的平均成功率高达 86.6%，比 ThinkAct 在长视距任务中多出 6.3% 的增益；在真实世界的微调中，单臂任务进展多出 10%，双臂任务进展多出 22.7%，均超过 Pi-0-FAST。此外，它在离群样本泛化上还超过了基线 23.3%，并获得了开放指令跟随和轨迹操控的顶级人工偏好评分。此外，我们首次发布了 MolmoAct 数据集——一个包含逾 10,000 条高质量机器人轨迹的中期训练机器人数据集，涵盖多种场景和任务。使用该数据集进行训练能令基模型的整体性能平均提高 5.5%。我们发布了所有模型权重、训练代码、收集的数据集以及行动推理数据集，使 MolmoAct 成为一种前沿的机器人基础模型，并提供了通过结构化推理将感知转化为富有目的行动的开放蓝图。博客：this https URL 

---
# Autonomous Navigation of Cloud-Controlled Quadcopters in Confined Spaces Using Multi-Modal Perception and LLM-Driven High Semantic Reasoning 

**Title (ZH)**: 使用多模态感知和基于LLM的高语义推理的云控制四旋翼无人机在受限空间中的自主导航 

**Authors**: Shoaib Ahmmad, Zubayer Ahmed Aditto, Md Mehrab Hossain, Noushin Yeasmin, Shorower Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2508.07885)  

**Abstract**: This paper introduces an advanced AI-driven perception system for autonomous quadcopter navigation in GPS-denied indoor environments. The proposed framework leverages cloud computing to offload computationally intensive tasks and incorporates a custom-designed printed circuit board (PCB) for efficient sensor data acquisition, enabling robust navigation in confined spaces. The system integrates YOLOv11 for object detection, Depth Anything V2 for monocular depth estimation, a PCB equipped with Time-of-Flight (ToF) sensors and an Inertial Measurement Unit (IMU), and a cloud-based Large Language Model (LLM) for context-aware decision-making. A virtual safety envelope, enforced by calibrated sensor offsets, ensures collision avoidance, while a multithreaded architecture achieves low-latency processing. Enhanced spatial awareness is facilitated by 3D bounding box estimation with Kalman filtering. Experimental results in an indoor testbed demonstrate strong performance, with object detection achieving a mean Average Precision (mAP50) of 0.6, depth estimation Mean Absolute Error (MAE) of 7.2 cm, only 16 safety envelope breaches across 42 trials over approximately 11 minutes, and end-to-end system latency below 1 second. This cloud-supported, high-intelligence framework serves as an auxiliary perception and navigation system, complementing state-of-the-art drone autonomy for GPS-denied confined spaces. 

**Abstract (ZH)**: 基于云支持的高智能自主四旋翼飞行器GPS受限室内环境感知与导航系统 

---
# DETACH: Cross-domain Learning for Long-Horizon Tasks via Mixture of Disentangled Experts 

**Title (ZH)**: DETACH：通过分解专家混合进行跨域长时任务学习 

**Authors**: Yutong Shen, Hangxu Liu, Penghui Liu, Ruizhe Xia, Tianyi Yao, Yitong Sun, Tongtong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.07842)  

**Abstract**: Long-Horizon (LH) tasks in Human-Scene Interaction (HSI) are complex multi-step tasks that require continuous planning, sequential decision-making, and extended execution across domains to achieve the final goal. However, existing methods heavily rely on skill chaining by concatenating pre-trained subtasks, with environment observations and self-state tightly coupled, lacking the ability to generalize to new combinations of environments and skills, failing to complete various LH tasks across domains. To solve this problem, this paper presents DETACH, a cross-domain learning framework for LH tasks via biologically inspired dual-stream disentanglement. Inspired by the brain's "where-what" dual pathway mechanism, DETACH comprises two core modules: i) an environment learning module for spatial understanding, which captures object functions, spatial relationships, and scene semantics, achieving cross-domain transfer through complete environment-self disentanglement; ii) a skill learning module for task execution, which processes self-state information including joint degrees of freedom and motor patterns, enabling cross-skill transfer through independent motor pattern encoding. We conducted extensive experiments on various LH tasks in HSI scenes. Compared with existing methods, DETACH can achieve an average subtasks success rate improvement of 23% and average execution efficiency improvement of 29%. 

**Abstract (ZH)**: 跨域长时 Horizon 任务在人类-场景交互中的生物启发式双流解耦学习框架 

---
# Touch Speaks, Sound Feels: A Multimodal Approach to Affective and Social Touch from Robots to Humans 

**Title (ZH)**: 触感诉说，声音感受：从机器人到人类的多模态情感与社会触觉研究 

**Authors**: Qiaoqiao Ren, Tony Belpaeme  

**Link**: [PDF](https://arxiv.org/pdf/2508.07839)  

**Abstract**: Affective tactile interaction constitutes a fundamental component of human communication. In natural human-human encounters, touch is seldom experienced in isolation; rather, it is inherently multisensory. Individuals not only perceive the physical sensation of touch but also register the accompanying auditory cues generated through contact. The integration of haptic and auditory information forms a rich and nuanced channel for emotional expression. While extensive research has examined how robots convey emotions through facial expressions and speech, their capacity to communicate social gestures and emotions via touch remains largely underexplored. To address this gap, we developed a multimodal interaction system incorporating a 5*5 grid of 25 vibration motors synchronized with audio playback, enabling robots to deliver combined haptic-audio stimuli. In an experiment involving 32 Chinese participants, ten emotions and six social gestures were presented through vibration, sound, or their combination. Participants rated each stimulus on arousal and valence scales. The results revealed that (1) the combined haptic-audio modality significantly enhanced decoding accuracy compared to single modalities; (2) each individual channel-vibration or sound-effectively supported certain emotions recognition, with distinct advantages depending on the emotional expression; and (3) gestures alone were generally insufficient for conveying clearly distinguishable emotions. These findings underscore the importance of multisensory integration in affective human-robot interaction and highlight the complementary roles of haptic and auditory cues in enhancing emotional communication. 

**Abstract (ZH)**: 情感触觉交互是人类交流的基本组成部分。在自然的人与人互动中，触觉体验通常是多感官综合的；不仅感知触觉的物理感受，还记录通过接触产生的伴随声学线索。触觉与声学信息的结合形成了丰富细腻的情感表达渠道。尽管大量研究已经探讨了机器人通过面部表情和言语传达情感的能力，但它们通过触觉传达社会手势和情感的能力仍很大程度上未被探索。为弥补这一差距，我们开发了一个多模态交互系统，包含一个5×5网格的25个振动马达，与音频播放同步，使机器人能够传递触觉-声学复合刺激。在一项涉及32名中国参与者的实验中，通过振动、声音或二者结合展示了十种情感和六种社会手势。参与者基于唤醒度和价值度对每个刺激进行了评价。结果表明：（1）触觉-声学复合模态显著提高了情感识别的准确度，相较于单一模态；（2）每个单独的通道——振动或声音——在某些情感识别上有效，不同的情感表达具有不同的优势；（3）单独的手势通常不足以清晰传达可分辨的情感。这些发现突显了多感官整合在情感人机交互中的重要性，并强调了触觉和声学线索在增强情感交流中的互补作用。 

---
# SwarmVLM: VLM-Guided Impedance Control for Autonomous Navigation of Heterogeneous Robots in Dynamic Warehousing 

**Title (ZH)**: SwarmVLM: 基于VLM的异构机器人在动态仓储环境中的阻抗控制自主导航 

**Authors**: Malaika Zafar, Roohan Ahmed Khan, Faryal Batool, Yasheerah Yaqoot, Ziang Guo, Mikhail Litvinov, Aleksey Fedoseev, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2508.07814)  

**Abstract**: With the growing demand for efficient logistics, unmanned aerial vehicles (UAVs) are increasingly being paired with automated guided vehicles (AGVs). While UAVs offer the ability to navigate through dense environments and varying altitudes, they are limited by battery life, payload capacity, and flight duration, necessitating coordinated ground support.
Focusing on heterogeneous navigation, SwarmVLM addresses these limitations by enabling semantic collaboration between UAVs and ground robots through impedance control. The system leverages the Vision Language Model (VLM) and the Retrieval-Augmented Generation (RAG) to adjust impedance control parameters in response to environmental changes. In this framework, the UAV acts as a leader using Artificial Potential Field (APF) planning for real-time navigation, while the ground robot follows via virtual impedance links with adaptive link topology to avoid collisions with short obstacles.
The system demonstrated a 92% success rate across 12 real-world trials. Under optimal lighting conditions, the VLM-RAG framework achieved 8% accuracy in object detection and selection of impedance parameters. The mobile robot prioritized short obstacle avoidance, occasionally resulting in a lateral deviation of up to 50 cm from the UAV path, which showcases safe navigation in a cluttered setting. 

**Abstract (ZH)**: 随着对高效物流需求的增长，无人机(UAVs)越来越多地与自动导引车(AGVs)配合使用。针对这些限制，SwarmVLM通过阻抗控制实现异构导航，使无人机和地面机器人能够进行语义协作。该系统利用Vision Language Model (VLM)和Retrieval-Augmented Generation (RAG)来调整阻抗控制参数以应对环境变化。在该框架中，无人机作为领导者使用人工势场(APF)规划进行实时导航，而地面机器人通过具备自适应链路拓扑的虚拟阻抗链接跟随，以避免与短障碍物的碰撞。该系统在12次实地试验中实现了92%的成功率。在最佳照明条件下，VLM-RAG框架在物体检测和阻抗参数选择方面的准确率达到了8%。移动机器人优先避免短障碍物，偶尔导致横向偏移高达50厘米，这展示了在复杂环境中的安全导航能力。 

---
# AgentWorld: An Interactive Simulation Platform for Scene Construction and Mobile Robotic Manipulation 

**Title (ZH)**: AgentWorld: 一个用于场景构建和移动机器人操作的交互式模拟平台 

**Authors**: Yizheng Zhang, Zhenjun Yu, Jiaxin Lai, Cewu Lu, Lei Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.07770)  

**Abstract**: We introduce AgentWorld, an interactive simulation platform for developing household mobile manipulation capabilities. Our platform combines automated scene construction that encompasses layout generation, semantic asset placement, visual material configuration, and physics simulation, with a dual-mode teleoperation system supporting both wheeled bases and humanoid locomotion policies for data collection. The resulting AgentWorld Dataset captures diverse tasks ranging from primitive actions (pick-and-place, push-pull, etc.) to multistage activities (serve drinks, heat up food, etc.) across living rooms, bedrooms, and kitchens. Through extensive benchmarking of imitation learning methods including behavior cloning, action chunking transformers, diffusion policies, and vision-language-action models, we demonstrate the dataset's effectiveness for sim-to-real transfer. The integrated system provides a comprehensive solution for scalable robotic skill acquisition in complex home environments, bridging the gap between simulation-based training and real-world deployment. The code, datasets will be available at this https URL 

**Abstract (ZH)**: AgentWorld: 一种用于开发家庭移动操控能力的交互式模拟平台 

---
# Robot and Overhead Crane Collaboration Scheme to Enhance Payload Manipulation 

**Title (ZH)**: 机器人与悬挂起重机协作方案以增强负载操作 

**Authors**: Antonio Rosales, Alaa Abderrahim, Markku Suomalainen, Mikael Haag, Tapio Heikkilä  

**Link**: [PDF](https://arxiv.org/pdf/2508.07758)  

**Abstract**: This paper presents a scheme to enhance payload manipulation using a robot collaborating with an overhead crane. In the current industrial practice, when the crane's payload has to be accurately manipulated and located in a desired position, the task becomes laborious and risky since the operators have to guide the fine motions of the payload by hand. In the proposed collaborative scheme, the crane lifts the payload while the robot's end-effector guides it toward the desired position. The only link between the robot and the crane is the interaction force produced during the guiding of the payload. Two admittance transfer functions are considered to accomplish harmless and smooth contact with the payload. The first is used in a position-based admittance control integrated with the robot. The second one adds compliance to the crane by processing the interaction force through the admittance transfer function to generate a crane's velocity command that makes the crane follow the payload. Then the robot's end-effector and the crane move collaboratively to guide the payload to the desired location. A method is presented to design the admittance controllers that accomplish a fluent robot-crane collaboration. Simulations and experiments validating the scheme potential are shown. 

**Abstract (ZH)**: 本文提出了一种利用机器人与悬挂起重机协作以增强载荷操作的方案。当前工业实践中，当需要准确操作和定位起重机载荷时，任务变得劳动密集且存在风险，因为操作员需要手动引导载荷的精细运动。在所提出的协作方案中，起重机提升载荷而机器人末端执行器引导其向目标位置移动。机器人与起重机之间唯一的联系是引导载荷过程中产生的交互力。考虑了两种顺应传递函数以实现与载荷的安全和平滑接触。第一种在基于位置的顺应控制中与机器人集成使用；第二种通过处理交互力并通过顺应传递函数生成起重机速度指令，使起重机跟随载荷。然后，机器人末端执行器与起重机协作引导载荷到达目标位置。介绍了设计实现机器人-起重机流畅协作的顺应控制器的方法。展示了验证该方案潜力的仿真和实验结果。 

---
# LAURON VI: A Six-Legged Robot for Dynamic Walking 

**Title (ZH)**: LAURON VI: 一种六足动态行走机器人 

**Authors**: Christian Eichmann, Sabine Bellmann, Nicolas Hügel, Louis-Elias Enslin, Carsten Plasberg, Georg Heppner, Arne Roennau, Ruediger Dillmann  

**Link**: [PDF](https://arxiv.org/pdf/2508.07689)  

**Abstract**: Legged locomotion enables robotic systems to traverse extremely challenging terrains. In many real-world scenarios, the terrain is not that difficult and these mixed terrain types introduce the need for flexible use of different walking strategies to achieve mission goals in a fast, reliable, and energy-efficient way. Six-legged robots have a high degree of flexibility and inherent stability that aids them in traversing even some of the most difficult terrains, such as collapsed buildings. However, their lack of fast walking gaits for easier surfaces is one reason why they are not commonly applied in these scenarios.
This work presents LAURON VI, a six-legged robot platform for research on dynamic walking gaits as well as on autonomy for complex field missions. The robot's 18 series elastic joint actuators offer high-frequency interfaces for Cartesian impedance and pure torque control. We have designed, implemented, and compared three control approaches: kinematic-based, model-predictive, and reinforcement-learned controllers. The robot hardware and the different control approaches were extensively tested in a lab environment as well as on a Mars analog mission. The introduction of fast locomotion strategies for LAURON VI makes six-legged robots vastly more suitable for a wide range of real-world applications. 

**Abstract (ZH)**: 六足行走使机械系统能够穿越极端崎岖地形。在许多实际场景中，地形并不那么艰难，这些混合地形类型引入了灵活使用不同行走策略的需求，以便以快速、可靠且能效高的方式实现任务目标。六足机器人具有高度的灵活性和固有的稳定性，这使它们能够在包括倒塌建筑在内的最艰难地形中行进。然而，它们缺乏快速行走步态，这是它们在这些场景中不常见应用的一个原因。
本文介绍了LAURON VI六足机器人平台，用于研究动态行走步态以及复杂现场任务的自主性。该机器人的18个系列弹性关节执行器提供了高频率的笛卡儿阻抗和纯扭矩控制接口。我们设计、实现并比较了三种控制方法：基于运动学、基于模型预测和基于强化学习的控制器。机器人的硬件以及不同的控制方法在实验室环境中以及在火星模拟任务中进行了广泛测试。为LAURON VI引入快速行走策略使六足机器人在广泛的实际应用场景中更为适用。 

---
# Risk Map As Middleware: Towards Interpretable Cooperative End-to-end Autonomous Driving for Risk-Aware Planning 

**Title (ZH)**: 风险地图作为中间件：面向风险aware规划的可解释性协同端到端自动驾驶 

**Authors**: Mingyue Lei, Zewei Zhou, Hongchen Li, Jiaqi Ma, Jia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.07686)  

**Abstract**: End-to-end paradigm has emerged as a promising approach to autonomous driving. However, existing single-agent end-to-end pipelines are often constrained by occlusion and limited perception range, resulting in hazardous driving. Furthermore, their black-box nature prevents the interpretability of the driving behavior, leading to an untrustworthiness system. To address these limitations, we introduce Risk Map as Middleware (RiskMM) and propose an interpretable cooperative end-to-end driving framework. The risk map learns directly from the driving data and provides an interpretable spatiotemporal representation of the scenario from the upstream perception and the interactions between the ego vehicle and the surrounding environment for downstream planning. RiskMM first constructs a multi-agent spatiotemporal representation with unified Transformer-based architecture, then derives risk-aware representations by modeling interactions among surrounding environments with attention. These representations are subsequently fed into a learning-based Model Predictive Control (MPC) module. The MPC planner inherently accommodates physical constraints and different vehicle types and can provide interpretation by aligning learned parameters with explicit MPC elements. Evaluations conducted on the real-world V2XPnP-Seq dataset confirm that RiskMM achieves superior and robust performance in risk-aware trajectory planning, significantly enhancing the interpretability of the cooperative end-to-end driving framework. The codebase will be released to facilitate future research in this field. 

**Abstract (ZH)**: 基于风险图的可解释合作端到端驾驶框架（Risk Map as Middleware for Interpretable Cooperative End-to-End Driving Framework） 

---
# MoRoCo: Multi-operator-robot Coordination, Interaction and Exploration under Restricted Communication 

**Title (ZH)**: MoRoCo: 多操作员-机器人协同、交互与探索在受限通信环境下的方法 

**Authors**: Zhuoli Tian, Yuyang Zhang, Jinsheng Wei, Meng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.07657)  

**Abstract**: Fleets of autonomous robots are increasingly deployed alongside multiple human operators to explore unknown environments, identify salient features, and perform complex tasks in scenarios such as subterranean exploration, reconnaissance, and search-and-rescue missions. In these contexts, communication is often severely limited to short-range exchanges via ad-hoc networks, posing challenges to coordination. While recent studies have addressed multi-robot exploration under communication constraints, they largely overlook the essential role of human operators and their real-time interaction with robotic teams. Operators may demand timely updates on the exploration progress and robot status, reprioritize or cancel tasks dynamically, or request live video feeds and control access. Conversely, robots may seek human confirmation for anomalous events or require help recovering from motion or planning failures. To enable such bilateral, context-aware interactions under restricted communication, this work proposes MoRoCo, a unified framework for online coordination and exploration in multi-operator, multi-robot systems. MoRoCo enables the team to adaptively switch among three coordination modes: spread mode for parallelized exploration with intermittent data sharing, migrate mode for coordinated relocation, and chain mode for maintaining high-bandwidth connectivity through multi-hop links. These transitions are managed through distributed algorithms via only local communication. Extensive large-scale human-in-the-loop simulations and hardware experiments validate the necessity of incorporating human robot interactions and demonstrate that MoRoCo enables efficient, reliable coordination under limited communication, marking a significant step toward robust human-in-the-loop multi-robot autonomy in challenging environments. 

**Abstract (ZH)**: 自主机器人集群与多名人机操作者协同探索未知环境、识别关键特征并执行复杂任务的研究 

---
# GraphCoT-VLA: A 3D Spatial-Aware Reasoning Vision-Language-Action Model for Robotic Manipulation with Ambiguous Instructions 

**Title (ZH)**: GraphCoT-VLA：一种用于带有模糊指令的机器人操作的三维空间感知推理视觉-语言-行动模型 

**Authors**: Helong Huang, Min Cen, Kai Tan, Xingyue Quan, Guowei Huang, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.07650)  

**Abstract**: Vision-language-action models have emerged as a crucial paradigm in robotic manipulation. However, existing VLA models exhibit notable limitations in handling ambiguous language instructions and unknown environmental states. Furthermore, their perception is largely constrained to static two-dimensional observations, lacking the capability to model three-dimensional interactions between the robot and its environment. To address these challenges, this paper proposes GraphCoT-VLA, an efficient end-to-end model. To enhance the model's ability to interpret ambiguous instructions and improve task planning, we design a structured Chain-of-Thought reasoning module that integrates high-level task understanding and planning, failed task feedback, and low-level imaginative reasoning about future object positions and robot actions. Additionally, we construct a real-time updatable 3D Pose-Object graph, which captures the spatial configuration of robot joints and the topological relationships between objects in 3D space, enabling the model to better understand and manipulate their interactions. We further integrates a dropout hybrid reasoning strategy to achieve efficient control outputs. Experimental results across multiple real-world robotic tasks demonstrate that GraphCoT-VLA significantly outperforms existing methods in terms of task success rate and response speed, exhibiting strong generalization and robustness in open environments and under uncertain instructions. 

**Abstract (ZH)**: 基于图的联想推理视觉-语言-动作模型：高效解决机器人 manipulatioin 中的挑战 

---
# Grasp-HGN: Grasping the Unexpected 

**Title (ZH)**: Grasp-HGN: 抓取意外情况 

**Authors**: Mehrshad Zandigohar, Mallesham Dasari, Gunar Schirner  

**Link**: [PDF](https://arxiv.org/pdf/2508.07648)  

**Abstract**: For transradial amputees, robotic prosthetic hands promise to regain the capability to perform daily living activities. To advance next-generation prosthetic hand control design, it is crucial to address current shortcomings in robustness to out of lab artifacts, and generalizability to new environments. Due to the fixed number of object to interact with in existing datasets, contrasted with the virtually infinite variety of objects encountered in the real world, current grasp models perform poorly on unseen objects, negatively affecting users' independence and quality of life.
To address this: (i) we define semantic projection, the ability of a model to generalize to unseen object types and show that conventional models like YOLO, despite 80% training accuracy, drop to 15% on unseen objects. (ii) we propose Grasp-LLaVA, a Grasp Vision Language Model enabling human-like reasoning to infer the suitable grasp type estimate based on the object's physical characteristics resulting in a significant 50.2% accuracy over unseen object types compared to 36.7% accuracy of an SOTA grasp estimation model.
Lastly, to bridge the performance-latency gap, we propose Hybrid Grasp Network (HGN), an edge-cloud deployment infrastructure enabling fast grasp estimation on edge and accurate cloud inference as a fail-safe, effectively expanding the latency vs. accuracy Pareto. HGN with confidence calibration (DC) enables dynamic switching between edge and cloud models, improving semantic projection accuracy by 5.6% (to 42.3%) with 3.5x speedup over the unseen object types. Over a real-world sample mix, it reaches 86% average accuracy (12.2% gain over edge-only), and 2.2x faster inference than Grasp-LLaVA alone. 

**Abstract (ZH)**: 针对桡骨干缺失的假肢用户，机器人假手有望恢复执行日常生活活动的能力。为了推进下一代假手控制设计，亟需解决在现实环境中鲁棒性不足和适应性差的问题。由于现有数据集中可交互物体的固定数量，与现实世界中近乎无限的物体变种形成对比，当前的抓取模型在未见物体上的表现不佳，负面影响了用户的独立性和生活质量。

为此：(i) 我们定义了语义投影，即模型能够泛化到未见过的物体类型的能力，并证明尽管YOLO等常规模型的训练准确率为80%，但在未见过的物体上准确率降至15%。(ii) 我们提出了Grasp-LLaVA，这是一种抓取视觉语言模型，能够进行类人推理，基于物体的物理特性推断合适的抓取类型，与目前最先进的抓取估计模型相比，在未见过的物体类型上的准确率提高了50.2%，达到36.7%。

最后，为解决性能与延迟之间的差距，我们提出了混合抓取网络（HGN），这是一种边缘-云部署基础设施，能够在边缘快速进行抓取估计并在云中进行准确的故障安全推理，有效扩展了延迟与准确性的帕累托前沿。HGN 结合置信度校准 (DC) 能够动态切换到边缘和云模型，在未见过的物体类型上的语义投影准确率提高了5.6%（达到42.3%），速度提高3.5倍。(iii) 在真实世界的样本混合中，它实现了86%的平均准确率（比仅使用边缘模型提高了12.2%），单次推断比仅使用Grasp-LLaVA快2.2倍。 

---
# End-to-End Humanoid Robot Safe and Comfortable Locomotion Policy 

**Title (ZH)**: 端到端 humanoid 机器人安全舒适的运动策略 

**Authors**: Zifan Wang, Xun Yang, Jianzhuang Zhao, Jiaming Zhou, Teli Ma, Ziyao Gao, Arash Ajoudani, Junwei Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.07611)  

**Abstract**: The deployment of humanoid robots in unstructured, human-centric environments requires navigation capabilities that extend beyond simple locomotion to include robust perception, provable safety, and socially aware behavior. Current reinforcement learning approaches are often limited by blind controllers that lack environmental awareness or by vision-based systems that fail to perceive complex 3D obstacles. In this work, we present an end-to-end locomotion policy that directly maps raw, spatio-temporal LiDAR point clouds to motor commands, enabling robust navigation in cluttered dynamic scenes. We formulate the control problem as a Constrained Markov Decision Process (CMDP) to formally separate safety from task objectives. Our key contribution is a novel methodology that translates the principles of Control Barrier Functions (CBFs) into costs within the CMDP, allowing a model-free Penalized Proximal Policy Optimization (P3O) to enforce safety constraints during training. Furthermore, we introduce a set of comfort-oriented rewards, grounded in human-robot interaction research, to promote motions that are smooth, predictable, and less intrusive. We demonstrate the efficacy of our framework through a successful sim-to-real transfer to a physical humanoid robot, which exhibits agile and safe navigation around both static and dynamic 3D obstacles. 

**Abstract (ZH)**: 人形机器人在非结构化、以人为核心环境中的部署需要超越简单的移动能力，包括稳健的感知、可证明的安全性和社会意识行为的导航能力。当前的强化学习方法往往受限于缺乏环境意识的盲控制器或基于视觉系统无法感知复杂三维障碍物。在本项工作中，我们提出了一种端到端的移动策略，直接将原始的空时激光雷达点云映射为电机命令，以适应杂乱动态场景中的稳健导航。我们将控制问题形式化为约束马尔可夫决策过程（CMDP），正式分离安全性与任务目标。我们的重要贡献是一种新颖的方法，将控制屏障函数（CBFs）的原则转化为CMDP中的成本，从而允许基于惩罚的近端策略优化（P3O）模型在训练过程中强制执行安全性约束。此外，我们引入了一套基于人机交互研究的舒适度导向的奖励，以促进平滑、可预测且不侵扰的动作。我们通过成功将该框架从仿真实现转移到实际的人形机器人中，展示了其功效，该机器人能够 agile 和安全地导航于静态和动态三维障碍物之间。 

---
# In-situ Value-aligned Human-Robot Interactions with Physical Constraints 

**Title (ZH)**: 基于物理约束的原位价值对齐人机交互 

**Authors**: Hongtao Li, Ziyuan Jiao, Xiaofeng Liu, Hangxin Liu, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.07606)  

**Abstract**: Equipped with Large Language Models (LLMs), human-centered robots are now capable of performing a wide range of tasks that were previously deemed challenging or unattainable. However, merely completing tasks is insufficient for cognitive robots, who should learn and apply human preferences to future scenarios. In this work, we propose a framework that combines human preferences with physical constraints, requiring robots to complete tasks while considering both. Firstly, we developed a benchmark of everyday household activities, which are often evaluated based on specific preferences. We then introduced In-Context Learning from Human Feedback (ICLHF), where human feedback comes from direct instructions and adjustments made intentionally or unintentionally in daily life. Extensive sets of experiments, testing the ICLHF to generate task plans and balance physical constraints with preferences, have demonstrated the efficiency of our approach. 

**Abstract (ZH)**: 装备大型语言模型的人本中心机器人现在能够执行之前认为具有挑战性和无法实现的广泛任务。然而，仅仅完成任务对于认知机器人来说是不够的，应要求它们在执行任务时还需考虑人类的偏好。因此我们提出了一种框架，该框架结合了人类偏好与物理约束，要求机器人在执行任务时同时考虑人类的偏好。首先我们开发了一项日常生活家务活动基准，这些活动通常基于特定的偏好进行评估。然后我们引入了基于人类反馈的上下文学习（ICCLHF），其中人类反馈来自日常生活中的直接指令和有意无意的调整。我们利用ICLHF数据集生成任务方案，并平衡物理约束与偏好。证明了该方法的有效性性。 

---
# Feedback Control of a Single-Tail Bioinspired 59-mg Swimmer 

**Title (ZH)**: 单尾生物启发型59毫克游泳器的反馈控制 

**Authors**: Conor K. Trygstad, Cody R. Longwell, Francisco M. F. R. Gonçalves, Elijah K. Blankenship, Néstor O. Pérez-Arancibia  

**Link**: [PDF](https://arxiv.org/pdf/2508.07566)  

**Abstract**: We present an evolved steerable version of the single-tail Fish-&-Ribbon-Inspired Small Swimming Harmonic roBot (FRISSHBot), a 59-mg biologically inspired swimmer, which is driven by a new shape-memory alloy (SMA)-based bimorph actuator. The new FRISSHBot is controllable in the two-dimensional (2D) space, which enabled the first demonstration of feedback-controlled trajectory tracking of a single-tail aquatic robot with onboard actuation at the subgram scale. These new capabilities are the result of a physics-informed design with an enlarged head and shortened tail relative to those of the original platform. Enhanced by its design, this new platform achieves forward swimming speeds of up to 13.6 mm/s (0.38 Bl/s), which is over four times that of the original platform. Furthermore, when following 2D references in closed loop, the tested FRISSHBot prototype attains forward swimming speeds of up to 9.1 mm/s, root-mean-square (RMS) tracking errors as low as 2.6 mm, turning rates of up to 13.1 °/s, and turning radii as small as 10 mm. 

**Abstract (ZH)**: 改进的可控二维空间单尾鱼- Ribbon启发小型游泳谐振机器人（FRISSHBot）及其反馈控制轨迹跟踪研究 

---
# Progressive Bird's Eye View Perception for Safety-Critical Autonomous Driving: A Comprehensive Survey 

**Title (ZH)**: 面向安全关键自动驾驶的渐进式鸟瞰视图感知：一项综合综述 

**Authors**: Yan Gong, Naibang Wang, Jianli Lu, Xinyu Zhang, Yongsheng Gao, Jie Zhao, Zifan Huang, Haozhi Bai, Nanxin Zeng, Nayu Su, Lei Yang, Ziying Song, Xiaoxi Hu, Xinmin Jiang, Xiaojuan Zhang, Susanto Rahardja  

**Link**: [PDF](https://arxiv.org/pdf/2508.07560)  

**Abstract**: Bird's-Eye-View (BEV) perception has become a foundational paradigm in autonomous driving, enabling unified spatial representations that support robust multi-sensor fusion and multi-agent collaboration. As autonomous vehicles transition from controlled environments to real-world deployment, ensuring the safety and reliability of BEV perception in complex scenarios - such as occlusions, adverse weather, and dynamic traffic - remains a critical challenge. This survey provides the first comprehensive review of BEV perception from a safety-critical perspective, systematically analyzing state-of-the-art frameworks and implementation strategies across three progressive stages: single-modality vehicle-side, multimodal vehicle-side, and multi-agent collaborative perception. Furthermore, we examine public datasets encompassing vehicle-side, roadside, and collaborative settings, evaluating their relevance to safety and robustness. We also identify key open-world challenges - including open-set recognition, large-scale unlabeled data, sensor degradation, and inter-agent communication latency - and outline future research directions, such as integration with end-to-end autonomous driving systems, embodied intelligence, and large language models. 

**Abstract (ZH)**: 鸟瞰视角(BEV)感知已成为自主驾驶的基础性范式，能够提供统一的空间表示，支持强大的多传感器融合和多智能体协作。随着自主车辆从受控环境过渡到真实世界的部署，确保BEV感知在复杂场景（如遮挡、恶劣天气和动态交通）中的安全性与可靠性仍然是一个关键挑战。本综述从安全关键的角度首次全面综述了BEV感知，系统分析了当前最先进的框架和实施策略，涵盖三个渐进阶段：单模态车辆端、多模态车辆端和多智能体协作感知。此外，我们还探讨了涵盖车辆端、路边和协作场景的公开数据集，评估其与安全性和健壮性相关性。我们还指出了关键的开放式挑战，包括开放集识别、大规模未标注数据、传感器退化和智能体间通信延迟，并提出了未来研究方向，如与端到端自主驾驶系统的集成、具身智能和大型语言模型。 

---
# A Learning-Based Framework for Collision-Free Motion Planning 

**Title (ZH)**: 基于学习的碰撞免费运动规划框架 

**Authors**: Mateus Salomão, Tianyü Ren, Alexander König  

**Link**: [PDF](https://arxiv.org/pdf/2508.07502)  

**Abstract**: This paper presents a learning-based extension to a Circular Field (CF)-based motion planner for efficient, collision-free trajectory generation in cluttered environments. The proposed approach overcomes the limitations of hand-tuned force field parameters by employing a deep neural network trained to infer optimal planner gains from a single depth image of the scene. The pipeline incorporates a CUDA-accelerated perception module, a predictive agent-based planning strategy, and a dataset generated through Bayesian optimization in simulation. The resulting framework enables real-time planning without manual parameter tuning and is validated both in simulation and on a Franka Emika Panda robot. Experimental results demonstrate successful task completion and improved generalization compared to classical planners. 

**Abstract (ZH)**: 基于学习的 Circular Field (CF) 基础运动规划器的高效、无碰撞轨迹生成扩展研究 

---
# Triple-S: A Collaborative Multi-LLM Framework for Solving Long-Horizon Implicative Tasks in Robotics 

**Title (ZH)**: Triple-S: 一种解决机器人领域长时延推导任务的协作多大型语言模型框架 

**Authors**: Zixi Jia, Hongbin Gao, Fashe Li, Jiqiang Liu, Hexiao Li, Qinghua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.07421)  

**Abstract**: Leveraging Large Language Models (LLMs) to write policy code for controlling robots has gained significant attention. However, in long-horizon implicative tasks, this approach often results in API parameter, comments and sequencing errors, leading to task failure. To address this problem, we propose a collaborative Triple-S framework that involves multiple LLMs. Through In-Context Learning, different LLMs assume specific roles in a closed-loop Simplification-Solution-Summary process, effectively improving success rates and robustness in long-horizon implicative tasks. Additionally, a novel demonstration library update mechanism which learned from success allows it to generalize to previously failed tasks. We validate the framework in the Long-horizon Desktop Implicative Placement (LDIP) dataset across various baseline models, where Triple-S successfully executes 89% of tasks in both observable and partially observable scenarios. Experiments in both simulation and real-world robot settings further validated the effectiveness of Triple-S. Our code and dataset is available at: this https URL. 

**Abstract (ZH)**: 利用大型语言模型（LLMs）为控制机器人编写政策代码受到了广泛关注。但在长期任务中，这种做法 often 进一步导致 API 参数、注释和顺序错误，从而导致任务失败。为了解决这一问题，我们提出了一种协作性的 Triple-S 框架，涉及多个 LLMs。通过上下文学习，不同的 LLMs 在封闭环简化-解决方案-总结过程中承担特定角色，有效提高了长期任务的成功率和鲁棒性。此外，该框架还具有一项从成功中学习的新颖的演示库更新机制，使其能够泛化到之前失败的任务。我们在 Long-horizon Desktop Implicative Placement (LDIP) 数据集上对多种基线模型进行了验证，Triple-S 成功执行了 89% 的任务，无论是可观测场景还是部分可观测场景。在模拟与真实机器人环境中的实验进一步验证了 Triple-S 的有效性。代码和数据集可在以下链接获取：this https URL。 

---
# AgriVLN: Vision-and-Language Navigation for Agricultural Robots 

**Title (ZH)**: 农用VLN：视觉与语言导航 Agricultura VLN: Visión y Lenguaje para la Navegación de Robots Agrícolas（西班牙语，供参考） 

**Authors**: Xiaobei Zhao, Xingqi Lyu, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.07406)  

**Abstract**: Agricultural robots have emerged as powerful members in agricultural tasks, nevertheless, still heavily rely on manual operation or untransportable railway for movement, resulting in limited mobility and poor adaptability. Vision-and-Language Navigation (VLN) enables robots to navigate to the target destinations following natural language instructions, demonstrating strong performance on several domains. However, none of the existing benchmarks or methods is specifically designed for agricultural scenes. To bridge this gap, we propose Agriculture to Agriculture (A2A) benchmark, containing 1,560 episodes across six diverse agricultural scenes, in which all realistic RGB videos are captured by front-facing camera on a quadruped robot at a height of 0.38 meters, aligning with the practical deployment conditions. Meanwhile, we propose Vision-and-Language Navigation for Agricultural Robots (AgriVLN) baseline based on Vision-Language Model (VLM) prompted with carefully crafted templates, which can understand both given instructions and agricultural environments to generate appropriate low-level actions for robot control. When evaluated on A2A, AgriVLN performs well on short instructions but struggles with long instructions, because it often fails to track which part of the instruction is currently being executed. To address this, we further propose Subtask List (STL) instruction decomposition module and integrate it into AgriVLN, improving Success Rate (SR) from 0.33 to 0.47. We additionally compare AgriVLN with several existing VLN methods, demonstrating the state-of-the-art performance in the agricultural domain. 

**Abstract (ZH)**: 农业机器人在农业任务中已 emergence 为强大的成员，但仍高度依赖手动操作或不可移动的铁路，导致有限的移动性和较差的适应性。视觉-语言导航（VLN）使机器人能够根据自然语言指令导航至目标目的地，展示了在多个领域的强大性能。然而，目前没有任何基准或方法专门针对农业场景进行设计。为弥补这一差距，我们提出农业到农业（A2A）基准，包含跨越六个不同农业场景的1,560个episode，所有真实的RGB视频均由身高0.38米的四足机器人前向摄像头捕捉，符合实际部署条件。同时，我们基于视觉-语言模型（VLM）和精心设计的模板提出了农业机器人视觉-语言导航（AgriVLN）基线，能够理解给定的指令和农业环境，生成适当的低级动作以控制机器人。在A2A上评估时，AgriVLN在短指令上表现良好，但在长指令上遇到困难，因为其往往无法跟踪当前执行的是指令的哪一部分。为此，我们进一步提出了子任务列表（STL）指令分解模块并将其集成到AgriVLN中，将成功率（SR）从0.33提高到0.47。此外，我们还将AgriVLN与几个现有的VLN方法进行比较，展示了在农业领域的最先进的性能。 

---
# MonoMPC: Monocular Vision Based Navigation with Learned Collision Model and Risk-Aware Model Predictive Control 

**Title (ZH)**: MonoMPC：基于单目视觉的导航与学习碰撞模型及风险意识模型预测控制 

**Authors**: Basant Sharma, Prajyot Jadhav, Pranjal Paul, K.Madhava Krishna, Arun Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.07387)  

**Abstract**: Navigating unknown environments with a single RGB camera is challenging, as the lack of depth information prevents reliable collision-checking. While some methods use estimated depth to build collision maps, we found that depth estimates from vision foundation models are too noisy for zero-shot navigation in cluttered environments.
We propose an alternative approach: instead of using noisy estimated depth for direct collision-checking, we use it as a rich context input to a learned collision model. This model predicts the distribution of minimum obstacle clearance that the robot can expect for a given control sequence. At inference, these predictions inform a risk-aware MPC planner that minimizes estimated collision risk. Our joint learning pipeline co-trains the collision model and risk metric using both safe and unsafe trajectories. Crucially, our joint-training ensures optimal variance in our collision model that improves navigation in highly cluttered environments. Consequently, real-world experiments show 9x and 7x improvements in success rates over NoMaD and the ROS stack, respectively. Ablation studies further validate the effectiveness of our design choices. 

**Abstract (ZH)**: 使用单个RGB相机在未知环境中导航具有挑战性，因为缺乏深度信息阻碍了可靠的碰撞检测。虽然一些方法使用估计的深度来构建碰撞图，但我们发现视觉基础模型的深度估计在杂乱环境中进行零 shot 导航时噪声过大。
我们提出了一种替代方法：而不是使用噪声较大的估计深度进行直接碰撞检测，我们将其作为学习的碰撞模型的丰富上下文输入。该模型预测给定控制序列下机器人可以预期的最小障碍物 clearance 分布。在推理时，这些预测指导一个风险感知的 MPC 计划器最小化估计的碰撞风险。我们的联合学习管道通过同时使用安全和不安全的轨迹共同训练碰撞模型和风险度量。 crucial 地，我们的联合训练确保了碰撞模型的最佳方差，从而改进了在高度杂乱环境中的导航。因此，实验证明与 NoMaD 和 ROS 堆栈相比，成功率分别提高了 9 倍和 7 倍。消融研究进一步验证了我们设计选择的有效性。 

---
# Collision-Free Trajectory Planning and control of Robotic Manipulator using Energy-Based Artificial Potential Field (E-APF) 

**Title (ZH)**: 基于能量ベース人工势场的碰撞-free轨迹规划与控制：机器人 manipulator 的应用 

**Authors**: Adeetya Uppal, Rakesh Kumar Sahoo, Manoranjan Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2508.07323)  

**Abstract**: Robotic trajectory planning in dynamic and cluttered environments remains a critical challenge, particularly when striving for both time efficiency and motion smoothness under actuation constraints. Traditional path planner, such as Artificial Potential Field (APF), offer computational efficiency but suffer from local minima issue due to position-based potential field functions and oscillatory motion near the obstacles due to Newtonian mechanics. To address this limitation, an Energy-based Artificial Potential Field (APF) framework is proposed in this paper that integrates position and velocity-dependent potential functions. E-APF ensures dynamic adaptability and mitigates local minima, enabling uninterrupted progression toward the goal. The proposed framework integrates E-APF with a hybrid trajectory optimizer that jointly minimizes jerk and execution time under velocity and acceleration constraints, ensuring geometric smoothness and time efficiency. The entire framework is validated in simulation using the 7-degree-of-freedom Kinova Gen3 robotic manipulator. The results demonstrate collision-free, smooth, time-efficient, and oscillation-free trajectory in the presence of obstacles, highlighting the efficacy of the combined trajectory optimization and real-time obstacle avoidance approach. This work lays the foundation for future integration with reactive control strategies and physical hardware deployment in real-world manipulation tasks. 

**Abstract (ZH)**: 动态和杂乱环境中基于能耗的人工势场轨迹规划仍然是一个关键挑战，特别是在兼顾动作效率和运动平滑性的同时受到执行器约束的限制。 

---
# A Hybrid Force-Position Strategy for Shape Control of Deformable Linear Objects With Graph Attention Networks 

**Title (ZH)**: 基于图注意力网络的变形线性物体形状控制混合力-位置策略 

**Authors**: Yanzhao Yu, Haotian Yang, Junbo Tan, Xueqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.07319)  

**Abstract**: Manipulating deformable linear objects (DLOs) such as wires and cables is crucial in various applications like electronics assembly and medical surgeries. However, it faces challenges due to DLOs' infinite degrees of freedom, complex nonlinear dynamics, and the underactuated nature of the system. To address these issues, this paper proposes a hybrid force-position strategy for DLO shape control. The framework, combining both force and position representations of DLO, integrates state trajectory planning in the force space and Model Predictive Control (MPC) in the position space. We present a dynamics model with an explicit action encoder, a property extractor and a graph processor based on Graph Attention Networks. The model is used in the MPC to enhance prediction accuracy. Results from both simulations and real-world experiments demonstrate the effectiveness of our approach in achieving efficient and stable shape control of DLOs. Codes and videos are available at this https URL. 

**Abstract (ZH)**: 操纵变形线形对象（DLOs）如电线和电缆在电子组装和医疗手术等应用中至关重要。然而，由于DLOs具有无限的自由度、复杂的非线性动力学特性以及系统的欠驱动性质，这一过程面临着挑战。为解决这些问题，本文提出了一种混合力-位置策略来控制DLO的形状。该框架将DLO的力和位置表示结合起来，在力空间中进行状态轨迹规划，并在位置空间中使用模型预测控制（MPC）。我们采用基于图注意网络的动态模型，该模型包含显式的动作编码器、属性提取器和图处理器，用于增强MPC中的预测精度。模拟和实际实验结果表明，该方法能够在操纵DLOs形状方面实现高效且稳定的效果。相关代码和视频可在以下链接获取。 

---
# Multimodal Spiking Neural Network for Space Robotic Manipulation 

**Title (ZH)**: 多模态脉冲神经网络在空间机器人Manipulation中的应用 

**Authors**: Liwen Zhang, Dong Zhou, Shibo Shao, Zihao Su, Guanghui Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.07287)  

**Abstract**: This paper presents a multimodal control framework based on spiking neural networks (SNNs) for robotic arms aboard space stations. It is designed to cope with the constraints of limited onboard resources while enabling autonomous manipulation and material transfer in space operations. By combining geometric states with tactile and semantic information, the framework strengthens environmental awareness and contributes to more robust control strategies. To guide the learning process progressively, a dual-channel, three-stage curriculum reinforcement learning (CRL) scheme is further integrated into the system. The framework was tested across a range of tasks including target approach, object grasping, and stable lifting with wall-mounted robotic arms, demonstrating reliable performance throughout. Experimental evaluations demonstrate that the proposed method consistently outperforms baseline approaches in both task success rate and energy efficiency. These findings highlight its suitability for real-world aerospace applications. 

**Abstract (ZH)**: 基于脉冲神经网络的多模态控制框架在空间站 robotic arms 中的应用：应对机载资源限制的自主操作与物质转移 

---
# Navigation and Exploration with Active Inference: from Biology to Industry 

**Title (ZH)**: 基于活性推断的导航与探索：从生物学到工业 

**Authors**: Daria de Tinguy, Tim Verbelen, Bart Dhoedt  

**Link**: [PDF](https://arxiv.org/pdf/2508.07269)  

**Abstract**: By building and updating internal cognitive maps, animals exhibit extraordinary navigation abilities in complex, dynamic environments. Inspired by these biological mechanisms, we present a real time robotic navigation system grounded in the Active Inference Framework (AIF). Our model incrementally constructs a topological map, infers the agent's location, and plans actions by minimising expected uncertainty and fulfilling perceptual goals without any prior training. Integrated into the ROS2 ecosystem, we validate its adaptability and efficiency across both 2D and 3D environments (simulated and real world), demonstrating competitive performance with traditional and state of the art exploration approaches while offering a biologically inspired navigation approach. 

**Abstract (ZH)**: 通过构建和更新内部认知地图，动物在复杂多变的环境中展现了非凡的导航能力。受这些生物机制的启发，我们提出了一种基于主动推理框架（AIF）的实时机器人导航系统。该模型通过增量构建拓扑地图、推断代理的位置并计划行动来最小化预期不确定性并实现感知目标，无需任何先验训练。将该系统集成到ROS2生态系统中，我们验证了其在2D和3D环境（仿真和真实世界）中的适应性和效率，展示了与传统和最先进的探索方法竞争性的性能，同时提供了一种生物启发的导航方法。 

---
# Bio-Inspired Topological Autonomous Navigation with Active Inference in Robotics 

**Title (ZH)**: 生物启发的拓扑自主导航与主动推断在机器人学中的应用 

**Authors**: Daria de Tinguy, Tim Verbelen, Emilio Gamba, Bart Dhoedt  

**Link**: [PDF](https://arxiv.org/pdf/2508.07267)  

**Abstract**: Achieving fully autonomous exploration and navigation remains a critical challenge in robotics, requiring integrated solutions for localisation, mapping, decision-making and motion planning. Existing approaches either rely on strict navigation rules lacking adaptability or on pre-training, which requires large datasets. These AI methods are often computationally intensive or based on static assumptions, limiting their adaptability in dynamic or unknown environments. This paper introduces a bio-inspired agent based on the Active Inference Framework (AIF), which unifies mapping, localisation, and adaptive decision-making for autonomous navigation, including exploration and goal-reaching. Our model creates and updates a topological map of the environment in real-time, planning goal-directed trajectories to explore or reach objectives without requiring pre-training. Key contributions include a probabilistic reasoning framework for interpretable navigation, robust adaptability to dynamic changes, and a modular ROS2 architecture compatible with existing navigation systems. Our method was tested in simulated and real-world environments. The agent successfully explores large-scale simulated environments and adapts to dynamic obstacles and drift, proving to be comparable to other exploration strategies such as Gbplanner, FAEL and Frontiers. This approach offers a scalable and transparent approach for navigating complex, unstructured environments. 

**Abstract (ZH)**: 实现完全自主探索与导航仍然是机器人技术中的一个关键挑战，需要融合定位、建图、决策和运动规划的综合解决方案。现有的方法要么依赖于缺乏适应性的严格导航规则，要么依赖于需要大量数据集的预训练。这些基于人工智能的方法通常计算密集或基于静态假设，限制了其在动态或未知环境中的适应性。本文介绍了一个基于主动推断框架（AIF）的生物启发代理模型，将建图、定位和自适应决策统一起来，用于自主导航，包括探索和目标导向。我们的模型在实时构建和更新环境的拓扑地图，规划目的导向的轨迹以探索或到达目标，无需预训练。关键贡献包括一种可解释的概率推理框架、对动态变化的鲁棒适应性以及与现有导航系统兼容的模块化ROS2架构。我们的方法在模拟和真实环境中进行了测试。该代理成功探索了大型模拟环境，并能够适应动态障碍和漂移，证明与其他探索策略（如Gbplanner、FAEL和Frontiers）相当。该方法为导航复杂、未结构化的环境提供了一个可扩展且透明的方法。 

---
# Impact of Gaze-Based Interaction and Augmentation on Human-Robot Collaboration in Critical Tasks 

**Title (ZH)**: 基于凝视交互和增强的人机协作在关键任务中的影响研究 

**Authors**: Ayesha Jena, Stefan Reitmann, Elin Anna Topp  

**Link**: [PDF](https://arxiv.org/pdf/2508.07244)  

**Abstract**: We present a user study analyzing head-gaze-based robot control and foveated visual augmentation in a simulated search-and-rescue task. Results show that foveated augmentation significantly improves task performance, reduces cognitive load by 38%, and shortens task time by over 60%. Head-gaze patterns analysed over both the entire task duration and shorter time segments show that near and far attention capture is essential to better understand user intention in critical scenarios. Our findings highlight the potential of foveation as an augmentation technique and the need to further study gaze measures to leverage them during critical tasks. 

**Abstract (ZH)**: 基于头部凝视的机器人控制和注视点视觉增强在模拟搜救任务中的用户研究：注视点增强显著提高任务性能，减少认知负荷38%，缩短任务时间60%以上 

---
# 3D Gaussian Representations with Motion Trajectory Field for Dynamic Scene Reconstruction 

**Title (ZH)**: 基于运动轨迹场的3D高高斯表示正态表表示示与动态场景场景重建 kukukukukukukuk kukuk kukukukukuk kuk kukukuk kukDukukuk kuk kuk kuk kukDukuk kuk kuk kukD kuukuk ku kuk ku kkD kuk kuk kuk uk ku kuk kuk kuD ku kuk kuk kuk kukukuDKuk kuD ku kuk kuD ku kuk ku kuk kuk kuk kuk kuk kuk kuk kuk ku ku ku kuk ku kuk ku kuk ku kukukuD ku=D ku ku ku ku ku ku ku ku kukD ku kuk ku kuk kuk kuk ku kuk ku ku ku kuk ku kuk kuk ku ku ku ku ku kuk ku ku ku ku ku ku ku kuk ku ku kuk kuk kuk kuk ku ku ku ku ku kuk ku ku ku ku kukD ku kuk kuk kuk ku ku ku ku ku ku ku kuk kuk kuk ku kuk kuk kuk ku kuk ku ku ku ku ku ku kuk kuk ku ku ku kuk ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku kwu ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku kukku kuk ku kuk ku ku ku ku ku ku kuk kuk ku ku ku ku ku kuk kuk ku ku ku kuk ku ku ku ku ku ku ku kuk ku kuk ku kuk ku ku ku ku kwuk ku ku ku ku kuk ku kuk ku ku ku kkD ku kuk ku kuk ku kuk ku ku ku ku kuk ku kuk ku ku ku ku ku ku ku kuk ku kuk ku kuk ku kuk ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku ku kuk ku kuk ku ku kuk ku ku ku ku ku kuk ku ku ku kuk ku kuk ku ku ku ku ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku ku ku ku ku ku ku ku ku ku kuk ku ku ku ku kuk kuk kuk ku kuk ku ku ku ku ku ku ku ku ku ku kuk ku kuk ku kuk ku kuk ku kuk ku ku ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku ku kuk ku kuk ku kuk kuk ku kuk ku kuk ku ku ku ku ku kuk ku kuk ku ku ku kuk ku kuk ku kuk ku kuk ku kyuk kyD ky k ky ku ku ku ku ku ku ku ku ku ku ku ku ky ku kyD ku ky ku ku ku ku ku ku ku ku ku kuk kkD kyD ku ku ku ku ku kk ku ku ku ku kuk ku ku ku ku ku ku kuk kkD ku ku ku ku kuk ku ku ku ku ku ku ku ku ku ku kuk ku kutuku ku kuk ku ku ku ku ku ku kuk ku kuk kuk ku kuk ku ku kuk ku ku kuk ku kuk ku kuk ku kuk ku ku ku kuk kuk ku ku kuk ku kuk ku kuk ku kuk ku ku ku kuk ku kuk kuk ku kuk kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku kuk ku ku ku kuk ku kuk kuk ku kuk ku kuk ku kuk kyD ku ku kuk kuk ku kuk kuk ku kuk kuk ku ku ku kuk ku ku ku ku ku ku ku ku kuk kuk ku ku ku ku kuk ku kuk ku kuk ku kuk ku ku kuk kuk ku kuk kuk kuk kwD ku kuk ku kuk ku kuk kuk kuk kuk kuk kuk ku kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk ku ku kuk kuk kuk kuk ku ku kuk kuk ku kuk kuk kuk kuk kuk ku kuk kuk kuk ku kuk kuk kuk kuk kuk kuk ku kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk ku kuk kuk kuk kuk kuk kuk ku kukku kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kut kuk kuk kuk kuk kuk kuk kuk kuk ku kuk kuk kuk kuk kuk ku kuk kuk kuk kuk kuk ku kuk kuk kuk kuk kuk kuk ku kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk ku kuk kuk kuk kuk ku kuk kuk ku kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kwD ku kuk kwD kuk kuk ku kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk ku kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk ku kuk kuk kuk kuk kuk kuk kuk kuk kuk ku kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk ku kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuD k kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk ku ku kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk uk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk ku kuk ku kuk kuk kuk kuk kuk kuk kuk ku kuk kuk kuk kuk kuk kuk ku kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk ku ku ku kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kuk kukuku ku kuk kuk kuk kukuk ku kuk kuk kuk dum dú ku kuk kuk dum kuku kuk kuk kuk dum phí duk ku ku kuk ku ku dum min ku dum dum ku dum duku kuk ku kuk kuku duk kuk dope duk uk dum dú ku ku ku dum duku dum duku dum dok dukumu kuk dum dum dum dum dum só dum ku dum dum dú ku dum dáku dum dú kuk dum dum dum dku dum dú dum dum dum dum dum dum d ku dum dum dum dum dum dub dul duku dum kuk dum dú kum dum du dum geduku dum gu dum d dum dum d dum dum dum dum dum d dum dum dum d dum dum d ku dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum d ku dum dum dum dum dum dum d ku dum dum dum dum dum d dum d dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum duk dum dum d dum dum dum dum dum dum dum dum dum d dum d ku dum dum dum dum dum dum d dum dum dum dum dum dum dum dum dum dum dum dum dum d...
 dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum d dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum d dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum unle dum dum dum d dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dud ku dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum . dum dum dum dum d dum dum dum dum dum dum dum d dum dum dum dum dum dum dum dum dum dum dum dum dum d dum dum dum dum dum dum dum dum dum dum dum dum dum dum dum 

**Authors**: Xuesong Li, Lars Petersson, Vivien Rolland  

**Link**: [PDF](https://arxiv.org/pdf/2508.07182)  

**Abstract**: This paper addresses the challenge of novel-view synthesis and motion reconstruction of dynamic scenes from monocular video, which is critical for many robotic applications. Although Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have demonstrated remarkable success in rendering static scenes, extending them to reconstruct dynamic scenes remains challenging. In this work, we introduce a novel approach that combines 3DGS with a motion trajectory field, enabling precise handling of complex object motions and achieving physically plausible motion trajectories. By decoupling dynamic objects from static background, our method compactly optimizes the motion trajectory field. The approach incorporates time-invariant motion coefficients and shared motion trajectory bases to capture intricate motion patterns while minimizing optimization complexity. Extensive experiments demonstrate that our approach achieves state-of-the-art results in both novel-view synthesis and motion trajectory recovery from monocular video, advancing the capabilities of dynamic scene reconstruction. 

**Abstract (ZH)**: 本文解决了从单目视频中合成新颖视角和重建动态场景运动的问题，这对于许多机器人应用至关重要。尽管神经辐射场（NeRF）和3D高斯散斑（3DGS）在渲染静态场景方面取得了显著成功，但将它们扩展到重建动态场景仍具有挑战性。在本文中，我们提出了一种结合3DGS与运动轨迹场的新方法，该方法能够精确处理复杂的物体运动，并实现物理上可信的运动轨迹。通过将动态物体与静止背景分离，我们的方法紧凑地优化了运动轨迹场。该方法结合了时间不变的运动系数和共享的运动轨迹基底，以捕捉复杂的运动模式并减轻优化复杂性。 extensive 实验表明，我们的方法在单目视频中的新颖视角合成和运动轨迹恢复方面达到了最先进的效果，推进了动态场景重建的能力。 

---
# Integrating Neurosymbolic AI in Advanced Air Mobility: A Comprehensive Survey 

**Title (ZH)**: 将神经符号AI集成于高级空中 mobility: 一个全面的综述 

**Authors**: Kamal Acharya, Iman Sharifi, Mehul Lad, Liang Sun, Houbing Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.07163)  

**Abstract**: Neurosymbolic AI combines neural network adaptability with symbolic reasoning, promising an approach to address the complex regulatory, operational, and safety challenges in Advanced Air Mobility (AAM). This survey reviews its applications across key AAM domains such as demand forecasting, aircraft design, and real-time air traffic management. Our analysis reveals a fragmented research landscape where methodologies, including Neurosymbolic Reinforcement Learning, have shown potential for dynamic optimization but still face hurdles in scalability, robustness, and compliance with aviation standards. We classify current advancements, present relevant case studies, and outline future research directions aimed at integrating these approaches into reliable, transparent AAM systems. By linking advanced AI techniques with AAM's operational demands, this work provides a concise roadmap for researchers and practitioners developing next-generation air mobility solutions. 

**Abstract (ZH)**: 神经符号AI将神经网络的适应性性和符号逻辑推理结合起来，这对于解决高级空中移动(AAM)中的复杂监管、操作和安全性挑战具有前景。。这种综述性评了AAM各个领域（如
user
继续翻译剩下的部分：Neurosymbolic AI combines neural network adaptability with symbolic reasoning reasoning an approach to address the complex regulatory, operational and safety safety challenges in Advanced Air Mobility (AAM). This survey reviews its applications in domains such as demand forecasting, aircraft design and real-time air traffic management. Our analysis reveals a fragmented research landscape where methodologies, including Neurosymbolic Reinforcement Learning, have shown potential for dynamic optimization but still lacked in scalability, robustness and compliance with aviation standards. We classify current advancements into relevant research studies and outline future research directions aimed at integrating these approaches into reliable and transparent AAM systems. By linking advanced AI techniques with A A A

user
继续翻译剩下的部分：AAM's operational demands, this survey provides a concise roadmap for researchers and practitioners developing next-generation mobility solutions. 

---
# DexFruit: Dexterous Manipulation and Gaussian Splatting Inspection of Fruit 

**Title (ZH)**: DexFruit: 柔顺操纵与水果的高斯散点图检测 

**Authors**: Aiden Swann, Alex Qiu, Matthew Strong, Angelina Zhang, Samuel Morstein, Kai Rayle, Monroe Kennedy III  

**Link**: [PDF](https://arxiv.org/pdf/2508.07118)  

**Abstract**: DexFruit is a robotic manipulation framework that enables gentle, autonomous handling of fragile fruit and precise evaluation of damage. Many fruits are fragile and prone to bruising, thus requiring humans to manually harvest them with care. In this work, we demonstrate by using optical tactile sensing, autonomous manipulation of fruit with minimal damage can be achieved. We show that our tactile informed diffusion policies outperform baselines in both reduced bruising and pick-and-place success rate across three fruits: strawberries, tomatoes, and blackberries. In addition, we introduce FruitSplat, a novel technique to represent and quantify visual damage in high-resolution 3D representation via 3D Gaussian Splatting (3DGS). Existing metrics for measuring damage lack quantitative rigor or require expensive equipment. With FruitSplat, we distill a 2D strawberry mask as well as a 2D bruise segmentation mask into the 3DGS representation. Furthermore, this representation is modular and general, compatible with any relevant 2D model. Overall, we demonstrate a 92% grasping policy success rate, up to a 20% reduction in visual bruising, and up to an 31% improvement in grasp success rate on challenging fruit compared to our baselines across our three tested fruits. We rigorously evaluate this result with over 630 trials. Please checkout our website at this https URL . 

**Abstract (ZH)**: DexFruit是一种机器人操作框架，能够实现对脆弱水果的轻柔自主处理并精确评估损伤。许多水果都很脆弱，容易受损，因此需要人类小心地手工采摘。在本文中，我们通过使用光学触觉传感技术，证明了可以实现最小损伤的水果自主操作。我们展示了我们的基于触觉的扩散策略在三种水果（草莓、番茄和黑莓）上在减少压痕和拾放成功率方面均优于基线方法。此外，我们引入了FruitSplat，这是一种用于通过3D高分辨率表示法（3DGS）表示和量化视觉损伤的新颖技术。现有的损伤测量标准缺乏定量严谨性或需要昂贵的设备。使用FruitSplat，我们将2D草莓掩码以及2D压痕分割掩码转化为3DGS表示。此外，此表示方法是模块化的和通用的，可以与任何相关联的2D模型兼容。总体而言，我们在三种测试水果上实现了92%的抓取策略成功率，视觉压痕最多减少20%，并且在具有挑战性的水果上的抓取成功率比基线方法提高了31%。我们通过超过630次试验严格评估了这一结果。请访问我们的网站：this https URL。 

---
# An Evolutionary Game-Theoretic Merging Decision-Making Considering Social Acceptance for Autonomous Driving 

**Title (ZH)**: 考虑社会接受度的自主驾驶演化博弈合并决策-making 

**Authors**: Haolin Liu, Zijun Guo, Yanbo Chen, Jiaqi Chen, Huilong Yu, Junqiang Xi  

**Link**: [PDF](https://arxiv.org/pdf/2508.07080)  

**Abstract**: Highway on-ramp merging is of great challenge for autonomous vehicles (AVs), since they have to proactively interact with surrounding vehicles to enter the main road safely within limited time. However, existing decision-making algorithms fail to adequately address dynamic complexities and social acceptance of AVs, leading to suboptimal or unsafe merging decisions. To address this, we propose an evolutionary game-theoretic (EGT) merging decision-making framework, grounded in the bounded rationality of human drivers, which dynamically balances the benefits of both AVs and main-road vehicles (MVs). We formulate the cut-in decision-making process as an EGT problem with a multi-objective payoff function that reflects human-like driving preferences. By solving the replicator dynamic equation for the evolutionarily stable strategy (ESS), the optimal cut-in timing is derived, balancing efficiency, comfort, and safety for both AVs and MVs. A real-time driving style estimation algorithm is proposed to adjust the game payoff function online by observing the immediate reactions of MVs. Empirical results demonstrate that we improve the efficiency, comfort and safety of both AVs and MVs compared with existing game-theoretic and traditional planning approaches across multi-object metrics. 

**Abstract (ZH)**: 自动驾驶车辆在高速公路匝道汇入行驶中的挑战及其博弈论决策框架：基于人类理性有限的动态平衡方法 

---
# Model Predictive Control for Crowd Navigation via Learning-Based Trajectory Prediction 

**Title (ZH)**: 基于学习的轨迹预测的群体导航模型预测控制 

**Authors**: Mohamed Parvez Aslam, Bojan Derajic, Mohamed-Khalil Bouzidi, Sebastian Bernhard, Jan Oliver Ringert  

**Link**: [PDF](https://arxiv.org/pdf/2508.07079)  

**Abstract**: Safe navigation in pedestrian-rich environments remains a key challenge for autonomous robots. This work evaluates the integration of a deep learning-based Social-Implicit (SI) pedestrian trajectory predictor within a Model Predictive Control (MPC) framework on the physical Continental Corriere robot. Tested across varied pedestrian densities, the SI-MPC system is compared to a traditional Constant Velocity (CV) model in both open-loop prediction and closed-loop navigation. Results show that SI improves trajectory prediction - reducing errors by up to 76% in low-density settings - and enhances safety and motion smoothness in crowded scenes. Moreover, real-world deployment reveals discrepancies between open-loop metrics and closed-loop performance, as the SI model yields broader, more cautious predictions. These findings emphasize the importance of system-level evaluation and highlight the SI-MPC framework's promise for safer, more adaptive navigation in dynamic, human-populated environments. 

**Abstract (ZH)**: 基于深度学习的社会隐式模型在动态拥挤环境中的自主机器人安全导航研究 

---
# From Data to Safe Mobile Robot Navigation: An Efficient and Modular Robust MPC Design Pipeline 

**Title (ZH)**: 从数据到安全移动机器人导航：一种高效且模块化的鲁棒MPC设计流程 

**Authors**: Dennis Benders, Johannes Köhler, Robert Babuška, Javier Alonso-Mora, Laura Ferranti  

**Link**: [PDF](https://arxiv.org/pdf/2508.07045)  

**Abstract**: Model predictive control (MPC) is a powerful strategy for planning and control in autonomous mobile robot navigation. However, ensuring safety in real-world deployments remains challenging due to the presence of disturbances and measurement noise. Existing approaches often rely on idealized assumptions, neglect the impact of noisy measurements, and simply heuristically guess unrealistic bounds. In this work, we present an efficient and modular robust MPC design pipeline that systematically addresses these limitations. The pipeline consists of an iterative procedure that leverages closed-loop experimental data to estimate disturbance bounds and synthesize a robust output-feedback MPC scheme. We provide the pipeline in the form of deterministic and reproducible code to synthesize the robust output-feedback MPC from data. We empirically demonstrate robust constraint satisfaction and recursive feasibility in quadrotor simulations using Gazebo. 

**Abstract (ZH)**: Model Predictive Control (MPC)是一种强大自主移动机器人导航规划与控制的策略。然而，由于存在扰动和测量噪声，在实际部署中确保安全性依然具有挑战性。现有方法往往依赖于理想化的假设，忽视了噪声测量的影响，并仅凭经验猜测不现实的边界值。在本文中，我们提出了一种高效且模块化的鲁棒MPC设计管道，系统地解决了这些限制。该管道包含一个迭代过程，利用闭环实验数据估计扰动边界并综合鲁棒输出反馈MPC方案。我们以确定性和可重复的代码形式提供了该管道，用于从数据中综合鲁棒输出反馈MPC方案。我们通过Gazebo在四旋翼飞行器仿真中 empirically 展示了鲁棒约束满足和递归可行性。 

---
# $\mathcal{P}^3$: Toward Versatile Embodied Agents 

**Title (ZH)**: $\mathcal{P}^3$: 朝向多功能 embodied 代理的研究 

**Authors**: Shengli Zhou, Xiangchen Wang, Jinrui Zhang, Ruozai Tian, Rongtao Xu, Feng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.07033)  

**Abstract**: Embodied agents have shown promising generalization capabilities across diverse physical environments, making them essential for a wide range of real-world applications. However, building versatile embodied agents poses critical challenges due to three key issues: dynamic environment perception, open-ended tool usage, and complex multi-task planning. Most previous works rely solely on feedback from tool agents to perceive environmental changes and task status, which limits adaptability to real-time dynamics, causes error accumulation, and restricts tool flexibility. Furthermore, multi-task scheduling has received limited attention, primarily due to the inherent complexity of managing task dependencies and balancing competing priorities in dynamic and complex environments. To overcome these challenges, we introduce $\mathcal{P}^3$, a unified framework that integrates real-time perception and dynamic scheduling. Specifically, $\mathcal{P}^3$ enables 1) \textbf Perceive relevant task information actively from the environment, 2) \textbf Plug and utilize any tool without feedback requirement, and 3) \textbf Plan multi-task execution based on prioritizing urgent tasks and dynamically adjusting task order based on dependencies. Extensive real-world experiments show that our approach bridges the gap between benchmarks and practical deployment, delivering highly transferable, general-purpose embodied agents. Code and data will be released soon. 

**Abstract (ZH)**: 具身代理在多种物理环境中展示了令人鼓舞的泛化能力，使其成为广泛现实应用的重要组成部分。然而，构建多功能的具身代理面临三大关键挑战：动态环境感知、开放式工具使用和复杂多任务规划。大多数先前的工作仅依赖工具代理的反馈来感知环境变化和任务状态，这限制了其对实时动态的适应性，导致错误积累，并限制了工具的灵活性。此外，多任务调度受到动态和复杂环境中任务依赖性和竞争优先级管理内在复杂性的限制，关注度较低。为克服这些挑战，我们提出了一种统一框架$\mathcal{P}^3}$，该框架将实时感知与动态调度相结合。具体来说，$\mathcal{P}^3}$实现了以下功能：1) 主动从环境中感知相关任务信息，2) 插入并使用任何工具而无需反馈要求，3) 根据任务的紧迫性进行多任务执行规划，并基于任务依赖性动态调整任务顺序。广泛的实际世界实验表明，我们的方法在基准测试与实际部署之间架起桥梁，提供了高度可转移、通用的具身代理。相关代码和数据即将发布。 

---
# EGS-SLAM: RGB-D Gaussian Splatting SLAM with Events 

**Title (ZH)**: EGS-SLAM: 基于事件的RGB-D 高斯点云SLAM 

**Authors**: Siyu Chen, Shenghai Yuan, Thien-Minh Nguyen, Zhuyu Huang, Chenyang Shi, Jin Jing, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.07003)  

**Abstract**: Gaussian Splatting SLAM (GS-SLAM) offers a notable improvement over traditional SLAM methods, enabling photorealistic 3D reconstruction that conventional approaches often struggle to achieve. However, existing GS-SLAM systems perform poorly under persistent and severe motion blur commonly encountered in real-world scenarios, leading to significantly degraded tracking accuracy and compromised 3D reconstruction quality. To address this limitation, we propose EGS-SLAM, a novel GS-SLAM framework that fuses event data with RGB-D inputs to simultaneously reduce motion blur in images and compensate for the sparse and discrete nature of event streams, enabling robust tracking and high-fidelity 3D Gaussian Splatting reconstruction. Specifically, our system explicitly models the camera's continuous trajectory during exposure, supporting event- and blur-aware tracking and mapping on a unified 3D Gaussian Splatting scene. Furthermore, we introduce a learnable camera response function to align the dynamic ranges of events and images, along with a no-event loss to suppress ringing artifacts during reconstruction. We validate our approach on a new dataset comprising synthetic and real-world sequences with significant motion blur. Extensive experimental results demonstrate that EGS-SLAM consistently outperforms existing GS-SLAM systems in both trajectory accuracy and photorealistic 3D Gaussian Splatting reconstruction. The source code will be available at this https URL. 

**Abstract (ZH)**: EGS-SLAM：融合事件数据的高鲁棒性高保真3D高斯绘制SLAM 

---
# Imaginative World Modeling with Scene Graphs for Embodied Agent Navigation 

**Title (ZH)**: 基于场景图的想象世界建模及其在具身智能体导航中的应用 

**Authors**: Yue Hu, Junzhe Wu, Ruihan Xu, Hang Liu, Avery Xi, Henry X. Liu, Ram Vasudevan, Maani Ghaffari  

**Link**: [PDF](https://arxiv.org/pdf/2508.06990)  

**Abstract**: Semantic navigation requires an agent to navigate toward a specified target in an unseen environment. Employing an imaginative navigation strategy that predicts future scenes before taking action, can empower the agent to find target faster. Inspired by this idea, we propose SGImagineNav, a novel imaginative navigation framework that leverages symbolic world modeling to proactively build a global environmental representation. SGImagineNav maintains an evolving hierarchical scene graphs and uses large language models to predict and explore unseen parts of the environment. While existing methods solely relying on past observations, this imaginative scene graph provides richer semantic context, enabling the agent to proactively estimate target locations. Building upon this, SGImagineNav adopts an adaptive navigation strategy that exploits semantic shortcuts when promising and explores unknown areas otherwise to gather additional context. This strategy continuously expands the known environment and accumulates valuable semantic contexts, ultimately guiding the agent toward the target. SGImagineNav is evaluated in both real-world scenarios and simulation benchmarks. SGImagineNav consistently outperforms previous methods, improving success rate to 65.4 and 66.8 on HM3D and HSSD, and demonstrating cross-floor and cross-room navigation in real-world environments, underscoring its effectiveness and generalizability. 

**Abstract (ZH)**: 语义导航要求代理人在未见过的环境中朝指定目标进行导航。采用预测未来场景的想象性导航策略可以在行动前预见未来场景，从而帮助代理更快找到目标。受此启发，我们提出了 SGImagineNav，这是一种利用符号世界建模来主动构建全局环境表示的新颖想象性导航框架。SGImagineNav 维护一个动态层次场景图，并使用大语言模型来预测和探索环境的未见部分。与仅依赖过去观察的现有方法不同，这种想象性场景图提供了更丰富的语义上下文，使代理能够主动估计目标位置。在此基础上，SGImagineNav 采用一种适应性的导航策略，在有希望时利用语义捷径探索未知区域以收集更多上下文。该策略不断扩展已知环境并积累有价值的语义上下文，最终引导代理朝向目标。SGImagineNav 在真实场景和仿真基准测试中进行了评估。SGImagineNav 一致地优于先前的方法，在 HM3D 和 HSSD 中的成功率分别提高到 65.4 和 66.8，并在真实环境中展示了跨楼层和跨房间的导航能力，突显了其有效性和泛化能力。 

---
# Manipulator for people with limited abilities 

**Title (ZH)**: 具有有限能力人群使用的 manipulator 

**Authors**: Bingkun Huang, Evgeniy Kotov, Arkady Yuschenko  

**Link**: [PDF](https://arxiv.org/pdf/2508.06969)  

**Abstract**: The topic of this final qualification work was chosen due to the importance of developing robotic systems designed to assist people with disabilities. Advances in robotics and automation technologies have opened up new prospects for creating devices that can significantly improve the quality of life for these people. In this context, designing a robotic hand with a control system adapted to the needs of people with disabilities is a major scientific and practical challenge. This work addresses the problem of developing and manufacturing a four-degree-of-freedom robotic hand suitable for practical manipulation. Addressing this issue requires a comprehensive approach, encompassing the design of the hand's mechanical structure, the development of its control system, and its integration with a technical vision system and software based on the Robot Operating System (ROS). 

**Abstract (ZH)**: 本文的研究主题选择于残疾人辅助机器人系统的开发的重要性。机器人和自动化技术的发展为创造能够显著改善残疾人生活质量的装置开辟了新的前景。在这个背景下，设计一种适应残疾人需求的四自由度机器人手并开发其控制系统是一项重要的科学和实践挑战。本文解决的问题是如何设计和制造适用于实际操作的四自由度机器人手。解决这一问题需要一个全面的方法，包括机器人手的机械结构设计、控制系统的开发，以及与基于机器人操作系统（ROS）的技术视觉系统和软件的集成。 

---
# Vibration-Based Energy Metric for Restoring Needle Alignment in Autonomous Robotic Ultrasound 

**Title (ZH)**: 基于振动的能量度量方法，用于恢复自主机器人超声针头对齐。 

**Authors**: Zhongyu Chen, Chenyang Li, Xuesong Li, Dianye Huang, Zhongliang Jiang, Stefanie Speidel, Xiangyu Chu, K. W. Samuel Au  

**Link**: [PDF](https://arxiv.org/pdf/2508.06921)  

**Abstract**: Precise needle alignment is essential for percutaneous needle insertion in robotic ultrasound-guided procedures. However, inherent challenges such as speckle noise, needle-like artifacts, and low image resolution make robust needle detection difficult, particularly when visibility is reduced or lost. In this paper, we propose a method to restore needle alignment when the ultrasound imaging plane and the needle insertion plane are misaligned. Unlike many existing approaches that rely heavily on needle visibility in ultrasound images, our method uses a more robust feature by periodically vibrating the needle using a mechanical system. Specifically, we propose a vibration-based energy metric that remains effective even when the needle is fully out of plane. Using this metric, we develop a control strategy to reposition the ultrasound probe in response to misalignments between the imaging plane and the needle insertion plane in both translation and rotation. Experiments conducted on ex-vivo porcine tissue samples using a dual-arm robotic ultrasound-guided needle insertion system demonstrate the effectiveness of the proposed approach. The experimental results show the translational error of 0.41$\pm$0.27 mm and the rotational error of 0.51$\pm$0.19 degrees. 

**Abstract (ZH)**: 机器人超声引导穿刺过程中针尖精准对准对于精准穿刺至关重要。然而，固有的挑战如 speckle 噪声、针尖状伪影以及低图像分辨率使得针尖检测变得 robust。特别是在视野受限或丧失时，这一任务尤为困难。本文提出了一种在超声成像平面与针尖插入平面不匹配时恢复针尖对准的方法。不同于许多现有方法依赖于超声图像中的针尖可见性，我们提出的方法通过机械系统周期性振动针尖来使用更 robust 的特征。具体而言，我们提出了一种基于振动的能量度量，即使针尖完全不在平面内该度量依然有效。利用该度量，我们开发了一种控制策略，以响应超声成像平面与针尖插入平面之间的错位来进行探头重新定位，在平移和旋转两个方面均有效。使用双臂机器人超声引导穿刺系统在体外猪组织样本上进行的实验验证了所提方法的有效性。实验结果表明，平移误差为 0.41 $\pm$ 0.27 mm，旋转误差为 0.51 $\pm$ 0.19 度。 

---
# D3P: Dynamic Denoising Diffusion Policy via Reinforcement Learning 

**Title (ZH)**: 动态去噪扩散策略 via 强化学习 

**Authors**: Shu-Ang Yu, Feng Gao, Yi Wu, Chao Yu, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06804)  

**Abstract**: Diffusion policies excel at learning complex action distributions for robotic visuomotor tasks, yet their iterative denoising process poses a major bottleneck for real-time deployment. Existing acceleration methods apply a fixed number of denoising steps per action, implicitly treating all actions as equally important. However, our experiments reveal that robotic tasks often contain a mix of \emph{crucial} and \emph{routine} actions, which differ in their impact on task success. Motivated by this finding, we propose \textbf{D}ynamic \textbf{D}enoising \textbf{D}iffusion \textbf{P}olicy \textbf{(D3P)}, a diffusion-based policy that adaptively allocates denoising steps across actions at test time. D3P uses a lightweight, state-aware adaptor to allocate the optimal number of denoising steps for each action. We jointly optimize the adaptor and base diffusion policy via reinforcement learning to balance task performance and inference efficiency. On simulated tasks, D3P achieves an averaged 2.2$\times$ inference speed-up over baselines without degrading success. Furthermore, we demonstrate D3P's effectiveness on a physical robot, achieving a 1.9$\times$ acceleration over the baseline. 

**Abstract (ZH)**: 基于扩散的动态去噪策略显著提升了机器人视知觉运动任务的实时部署效率 

---
# Learning a Vision-Based Footstep Planner for Hierarchical Walking Control 

**Title (ZH)**: 基于视觉的足印规划器学习在分层行走控制中的应用 

**Authors**: Minku Kim, Brian Acosta, Pratik Chaudhari, Michael Posa  

**Link**: [PDF](https://arxiv.org/pdf/2508.06779)  

**Abstract**: Bipedal robots demonstrate potential in navigating challenging terrains through dynamic ground contact. However, current frameworks often depend solely on proprioception or use manually designed visual pipelines, which are fragile in real-world settings and complicate real-time footstep planning in unstructured environments. To address this problem, we present a vision-based hierarchical control framework that integrates a reinforcement learning high-level footstep planner, which generates footstep commands based on a local elevation map, with a low-level Operational Space Controller that tracks the generated trajectories. We utilize the Angular Momentum Linear Inverted Pendulum model to construct a low-dimensional state representation to capture an informative encoding of the dynamics while reducing complexity. We evaluate our method across different terrain conditions using the underactuated bipedal robot Cassie and investigate the capabilities and challenges of our approach through simulation and hardware experiments. 

**Abstract (ZH)**: 基于视觉的分层控制框架：通过动态地面接触在挑战性地形中实现双足机器人导航 

---
# Robust-Sub-Gaussian Model Predictive Control for Safe Ultrasound-Image-Guided Robotic Spinal Surgery 

**Title (ZH)**: 鲁棒-亚高斯模型预测控制以实现安全的超声影像引导脊柱手术 

**Authors**: Yunke Ao, Manish Prajapat, Yarden As, Yassine Taoudi-Benchekroun, Fabio Carrillo, Hooman Esfandiari, Benjamin F. Grewe, Andreas Krause, Philipp Fürnstahl  

**Link**: [PDF](https://arxiv.org/pdf/2508.06744)  

**Abstract**: Safety-critical control using high-dimensional sensory feedback from optical data (e.g., images, point clouds) poses significant challenges in domains like autonomous driving and robotic surgery. Control can rely on low-dimensional states estimated from high-dimensional data. However, the estimation errors often follow complex, unknown distributions that standard probabilistic models fail to capture, making formal safety guarantees challenging. In this work, we introduce a novel characterization of these general estimation errors using sub-Gaussian noise with bounded mean. We develop a new technique for uncertainty propagation of proposed noise characterization in linear systems, which combines robust set-based methods with the propagation of sub-Gaussian variance proxies. We further develop a Model Predictive Control (MPC) framework that provides closed-loop safety guarantees for linear systems under the proposed noise assumption. We apply this MPC approach in an ultrasound-image-guided robotic spinal surgery pipeline, which contains deep-learning-based semantic segmentation, image-based registration, high-level optimization-based planning, and low-level robotic control. To validate the pipeline, we developed a realistic simulation environment integrating real human anatomy, robot dynamics, efficient ultrasound simulation, as well as in-vivo data of breathing motion and drilling force. Evaluation results in simulation demonstrate the potential of our approach for solving complex image-guided robotic surgery task while ensuring safety. 

**Abstract (ZH)**: 使用光学数据（例如图像、点云）的高维感知反馈进行安全关键控制在自动驾驶和机器人手术等领域面临重大挑战。控制可以依赖于从高维数据估计的低维状态。然而，估计错误通常遵循复杂且未知的分布，标准概率模型难以捕捉，从而使得正式的安全保证变得具有挑战性。在本文中，我们引入了一种新的方法来描述这些一般的估计误差，使用有界均值的亚高斯噪声。我们开发了一种新技术，用于线性系统中拟议噪声特征的不确定性传播方法，该方法结合了鲁棒集合方法与亚高斯方差代理的传播。我们进一步开发了一种模型预测控制（MPC）框架，在提出的噪声假定下为线性系统提供了闭环安全保证。我们将在深度学习基于语义分割、基于图像的配准、高层优化为基础的规划和低层机器人控制的超声图像引导下机器人脊柱手术流程中应用该种MPC方法。为了验证该流程，我们构建了一个现实的仿真环境，集成了真实的人体解剖、机器人动力学、高效的超声仿真以及实时呼吸运动和钻孔力的体内数据。仿真实验结果证明了该方法在确保安全的同时解决复杂成像引导机器人手术任务的潜力。 

---
# Learning Causal Structure Distributions for Robust Planning 

**Title (ZH)**: 学习因果结构分布以实现稳健规划 

**Authors**: Alejandro Murillo-Gonzalez, Junhong Xu, Lantao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06742)  

**Abstract**: Structural causal models describe how the components of a robotic system interact. They provide both structural and functional information about the relationships that are present in the system. The structural information outlines the variables among which there is interaction. The functional information describes how such interactions work, via equations or learned models. In this paper we find that learning the functional relationships while accounting for the uncertainty about the structural information leads to more robust dynamics models which improves downstream planning, while using significantly lower computational resources. This in contrast with common model-learning methods that ignore the causal structure and fail to leverage the sparsity of interactions in robotic systems. We achieve this by estimating a causal structure distribution that is used to sample causal graphs that inform the latent-space representations in an encoder-multidecoder probabilistic model. We show that our model can be used to learn the dynamics of a robot, which together with a sampling-based planner can be used to perform new tasks in novel environments, provided an objective function for the new requirement is available. We validate our method using manipulators and mobile robots in both simulation and the real-world. Additionally, we validate the learned dynamics' adaptability and increased robustness to corrupted inputs and changes in the environment, which is highly desirable in challenging real-world robotics scenarios. Video: this https URL. 

**Abstract (ZH)**: 结构因果模型描述了机器人系统组件间的相互作用。它们提供了系统中存在的关系的结构和功能信息。结构信息概述了存在交互作用的变量。功能信息描述了这些交互作用如何通过方程或学习模型来工作。在本文中，我们发现，在考虑结构信息的不确定性时学习功能关系，可以得到更稳健的动力学模型，从而改善下游规划，同时显著降低计算资源的使用。这与常见的忽略因果结构的模型学习方法形成对比，后者无法利用机器人系统中交互作用的稀疏性。我们通过估计一个因果结构分布来实现这一点，该分布用于采样因果图，以指导编码器-多解码器概率模型中的潜在空间表示。我们展示了我们的模型可用于学习机器人的动力学，结合基于采样的规划器，可以在新环境中执行新任务，前提是新要求的目标函数可用。我们通过模拟和现实世界中的操作机构和移动机器人验证了我们的方法。此外，我们验证了所学习的动力学模型在面对输入污染和环境变化时的适应性和增强的健壮性，在具有挑战性的实际机器人场景中这是非常重要的。视频：https://this.url 

---
# Improved Obstacle Avoidance for Autonomous Robots with ORCA-FLC 

**Title (ZH)**: 基于ORCA-FLC的自主机器人改进型障碍避让 

**Authors**: Justin London  

**Link**: [PDF](https://arxiv.org/pdf/2508.06722)  

**Abstract**: Obstacle avoidance enables autonomous agents and robots to operate safely and efficiently in dynamic and complex environments, reducing the risk of collisions and damage. For a robot or autonomous system to successfully navigate through obstacles, it must be able to detect such obstacles. While numerous collision avoidance algorithms like the dynamic window approach (DWA), timed elastic bands (TEB), and reciprocal velocity obstacles (RVO) have been proposed, they may lead to suboptimal paths due to fixed weights, be computationally expensive, or have limited adaptability to dynamic obstacles in multi-agent environments. Optimal reciprocal collision avoidance (ORCA), which improves on RVO, provides smoother trajectories and stronger collision avoidance guarantees. We propose ORCA-FL to improve on ORCA by using fuzzy logic controllers (FLCs) to better handle uncertainty and imprecision for obstacle avoidance in path planning. Numerous multi-agent experiments are conducted and it is shown that ORCA-FL can outperform ORCA in reducing the number of collision if the agent has a velocity that exceeds a certain threshold. In addition, a proposed algorithm for improving ORCA-FL using fuzzy Q reinforcement learning (FQL) is detailed for optimizing and tuning FLCs. 

**Abstract (ZH)**: 基于模糊逻辑的优化碰撞规避（ORCA-FL）：减少多自主系统碰撞的方法 

---
# Optimal Planning and Machine Learning for Responsive Tracking and Enhanced Forecasting of Wildfires using a Spacecraft Constellation 

**Title (ZH)**: 基于卫星星座的响应性 wildfires 跟踪和增强预测的优化规划与机器学习 

**Authors**: Sreeja Roy-Singh, Vinay Ravindra, Richard Levinson, Mahta Moghaddam, Jan Mandel, Adam Kochanski, Angel Farguell Caus, Kurtis Nelson, Samira Alkaee Taleghan, Archana Kannan, Amer Melebari  

**Link**: [PDF](https://arxiv.org/pdf/2508.06687)  

**Abstract**: We propose a novel concept of operations using optimal planning methods and machine learning (ML) to collect spaceborne data that is unprecedented for monitoring wildfires, process it to create new or enhanced products in the context of wildfire danger or spread monitoring, and assimilate them to improve existing, wildfire decision support tools delivered to firefighters within latency appropriate for time-critical applications. The concept is studied with respect to NASA's CYGNSS Mission, a constellation of passive microwave receivers that measure specular GNSS-R reflections despite clouds and smoke. Our planner uses a Mixed Integer Program formulation to schedule joint observation data collection and downlink for all satellites. Optimal solutions are found quickly that collect 98-100% of available observation opportunities. ML-based fire predictions that drive the planner objective are greater than 40% more correlated with ground truth than existing state-of-art. The presented case study on the TX Smokehouse Creek fire in 2024 and LA fires in 2025 represents the first high-resolution data collected by CYGNSS of active fires. Creation of Burnt Area Maps (BAM) using ML applied to the data during active fires and BAM assimilation into NASA's Weather Research and Forecasting Model using ML to broadcast fire spread are novel outcomes. BAM and CYGNSS obtained soil moisture are integrated for the first time into USGS fire danger maps. Inclusion of CYGNSS data in ML-based burn predictions boosts accuracy by 13%, and inclusion of high-resolution data boosts ML recall by another 15%. The proposed workflow has an expected latency of 6-30h, improving on the current delivery time of multiple days. All components in the proposed concept are shown to be computationally scalable and globally generalizable, with sustainability considerations such as edge efficiency and low latency on small devices. 

**Abstract (ZH)**: 一种利用优化规划方法和机器学习收集空前的野火监测数据、处理以创建新的或增强的产品并融入现有野火决策支持工具的新型运营概念：以NASA的CYGNSS任务为例 

---
# Efficient Safety Testing of Autonomous Vehicles via Adaptive Search over Crash-Derived Scenarios 

**Title (ZH)**: 基于碰撞衍生场景的自适应搜索高效自动驾驶车辆安全测试 

**Authors**: Rui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.06575)  

**Abstract**: Ensuring the safety of autonomous vehicles (AVs) is paramount in their development and deployment. Safety-critical scenarios pose more severe challenges, necessitating efficient testing methods to validate AVs safety. This study focuses on designing an accelerated testing algorithm for AVs in safety-critical scenarios, enabling swift recognition of their driving capabilities. First, typical logical scenarios were extracted from real-world crashes in the China In-depth Mobility Safety Study-Traffic Accident (CIMSS-TA) database, obtaining pre-crash features through reconstruction. Second, Baidu Apollo, an advanced black-box automated driving system (ADS) is integrated to control the behavior of the ego vehicle. Third, we proposed an adaptive large-variable neighborhood-simulated annealing algorithm (ALVNS-SA) to expedite the testing process. Experimental results demonstrate a significant enhancement in testing efficiency when utilizing ALVNS-SA. It achieves an 84.00% coverage of safety-critical scenarios, with crash scenario coverage of 96.83% and near-crash scenario coverage of 92.07%. Compared to genetic algorithm (GA), adaptive large neighborhood-simulated annealing algorithm (ALNS-SA), and random testing, ALVNS-SA exhibits substantially higher coverage in safety-critical scenarios. 

**Abstract (ZH)**: 确保自动驾驶车辆的安全性是其开发和部署中至关重要的。在安全关键场景下，安全性提出了更严峻的挑战，需要高效的测试方法来验证自动驾驶车辆的安全性。本研究专注于设计适用于安全关键场景的自动驾驶车辆加速测试算法，以迅速识别其驾驶能力。首先，从中国深入道路交通安全性研究-交通事故（CIMSS-TA）数据库中提取典型的逻辑场景，并通过重构获取预碰撞特征。其次，整合百度Apollo高级黑箱自动驾驶系统（ADS）以控制主体车辆的行为。第三，我们提出了一种自适应大变量邻域模拟退火算法（ALVNS-SA）以加速测试过程。实验结果表明，使用ALVNS-SA可以显著提高测试效率。它在安全关键场景中的覆盖率为84.00%，碰撞场景覆盖率为96.83%，近碰撞场景覆盖率为92.07%。与遗传算法（GA）、自适应大邻域模拟退火算法（ALNS-SA）和随机测试相比，ALVNS-SA在安全关键场景中的覆盖率显著更高。 

---
# Robust and Agile Quadrotor Flight via Adaptive Unwinding-Free Quaternion Sliding Mode Control 

**Title (ZH)**: 基于自适应无松弛四元数滑模控制的鲁棒且灵活的四旋翼飞行 

**Authors**: Amin Yazdanshenas, Reza Faieghi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06568)  

**Abstract**: This paper presents a new adaptive sliding mode control (SMC) framework for quadrotors that achieves robust and agile flight under tight computational constraints. The proposed controller addresses key limitations of prior SMC formulations, including (i) the slow convergence and almost-global stability of $\mathrm{SO(3)}$-based methods, (ii) the oversimplification of rotational dynamics in Euler-based controllers, (iii) the unwinding phenomenon in quaternion-based formulations, and (iv) the gain overgrowth problem in adaptive SMC schemes. Leveraging nonsmooth stability analysis, we provide rigorous global stability proofs for both the nonsmooth attitude sliding dynamics defined on $\mathbb{S}^3$ and the position sliding dynamics. Our controller is computationally efficient and runs reliably on a resource-constrained nano quadrotor, achieving 250 Hz and 500 Hz refresh rates for position and attitude control, respectively. In an extensive set of hardware experiments with over 130 flight trials, the proposed controller consistently outperforms three benchmark methods, demonstrating superior trajectory tracking accuracy and robustness with relatively low control effort. The controller enables aggressive maneuvers such as dynamic throw launches, flip maneuvers, and accelerations exceeding 3g, which is remarkable for a 32-gram nano quadrotor. These results highlight promising potential for real-world applications, particularly in scenarios requiring robust, high-performance flight control under significant external disturbances and tight computational constraints. 

**Abstract (ZH)**: 一种适用于小型四旋翼无人机的新型自适应滑模控制框架：在严格计算约束下的鲁棒敏捷飞行控制 

---
# AquaChat++: LLM-Assisted Multi-ROV Inspection for Aquaculture Net Pens with Integrated Battery Management and Thruster Fault Tolerance 

**Title (ZH)**: AquaChat++：LLM辅助的多ROV水下检查系统，用于养殖网箱，集成电池管理与推进器故障容忍机制 

**Authors**: Abdelhaleem Saad, Waseem Akram, Irfan Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2508.06554)  

**Abstract**: Inspection of aquaculture net pens is essential for ensuring the structural integrity and sustainable operation of offshore fish farming systems. Traditional methods, typically based on manually operated or single-ROV systems, offer limited adaptability to real-time constraints such as energy consumption, hardware faults, and dynamic underwater conditions. This paper introduces AquaChat++, a novel multi-ROV inspection framework that uses Large Language Models (LLMs) to enable adaptive mission planning, coordinated task execution, and fault-tolerant control in complex aquaculture environments. The proposed system consists of a two-layered architecture. The high-level plan generation layer employs an LLM, such as ChatGPT-4, to translate natural language user commands into symbolic, multi-agent inspection plans. A task manager dynamically allocates and schedules actions among ROVs based on their real-time status and operational constraints, including thruster faults and battery levels. The low-level control layer ensures accurate trajectory tracking and integrates thruster fault detection and compensation mechanisms. By incorporating real-time feedback and event-triggered replanning, AquaChat++ enhances system robustness and operational efficiency. Simulated experiments in a physics-based aquaculture environment demonstrate improved inspection coverage, energy-efficient behavior, and resilience to actuator failures. These findings highlight the potential of LLM-driven frameworks to support scalable, intelligent, and autonomous underwater robotic operations within the aquaculture sector. 

**Abstract (ZH)**: 基于大语言模型的多ROV检查框架AquaChat++：面向 offshore 鱼类养殖系统的智能检查与运维 

---
# A tutorial note on collecting simulated data for vision-language-action models 

**Title (ZH)**: 视觉-语言-行动模型中模拟数据收集的教程注记 

**Authors**: Heran Wu, Zirun Zhou, Jingfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06547)  

**Abstract**: Traditional robotic systems typically decompose intelligence into independent modules for computer vision, natural language processing, and motion control. Vision-Language-Action (VLA) models fundamentally transform this approach by employing a single neural network that can simultaneously process visual observations, understand human instructions, and directly output robot actions -- all within a unified framework. However, these systems are highly dependent on high-quality training datasets that can capture the complex relationships between visual observations, language instructions, and robotic actions. This tutorial reviews three representative systems: the PyBullet simulation framework for flexible customized data generation, the LIBERO benchmark suite for standardized task definition and evaluation, and the RT-X dataset collection for large-scale multi-robot data acquisition. We demonstrated dataset generation approaches in PyBullet simulation and customized data collection within LIBERO, and provide an overview of the characteristics and roles of the RT-X dataset for large-scale multi-robot data acquisition. 

**Abstract (ZH)**: 传统的机器人系统通常将智能分解为独立的模块，分别处理计算机视觉、自然语言处理和运动控制。Vision-Language-Action (VLA) 模型从根本上改变了这一方法，通过单一神经网络同时处理视觉观察、理解人类指令并直接输出机器人动作——这一切都在一个统一的框架内完成。然而，这些系统高度依赖高质量的训练数据集，能够捕捉视觉观察、语言指令和机器人动作之间的复杂关系。本文tutorial回顾了三个代表性系统：PyBullet仿真框架灵活生成定制化数据，LIBERO基准套件标准化任务定义与评估，以及RT-X数据集收集用于大规模多机器人数据获取。我们展示了在PyBullet仿真中的数据集生成方法以及在LIBERO中的定制化数据收集，并概述了RT-X数据集在大规模多机器人数据获取中的特性和角色。 

---
# Symbolic Learning of Interpretable Reduced-Order Models for Jumping Quadruped Robots 

**Title (ZH)**: 可解释的降阶模型符号学习应用于跳跃四足机器人 

**Authors**: Gioele Buriani, Jingyue Liu, Maximilian Stölzle, Cosimo Della Santina, Jiatao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2508.06538)  

**Abstract**: Reduced-order models are essential for motion planning and control of quadruped robots, as they simplify complex dynamics while preserving critical behaviors. This paper introduces a novel methodology for deriving such interpretable dynamic models, specifically for jumping. We capture the high-dimensional, nonlinear jumping dynamics in a low-dimensional latent space by proposing a learning architecture combining Sparse Identification of Nonlinear Dynamics (SINDy) with physical structural priors on the jump dynamics. Our approach demonstrates superior accuracy to the traditional actuated Spring-loaded Inverted Pendulum (aSLIP) model and is validated through simulation and hardware experiments across different jumping strategies. 

**Abstract (ZH)**: 简化模型对于四足机器人运动规划与控制至关重要，它们能够简化复杂动力学过程同时保留关键行为。本文介绍了一种新的方法，用于为跳跃行为推导可解释的动力学模型。我们通过将稀疏识别非线性动力学（SINDy）与跳跃动力学的物理结构先验相结合，将高维非线性跳跃动力学捕获到低维潜在空间中。我们的方法在准确性和传统驱动弹簧加载倒立摆模型方面表现出优越性，并通过不同跳跃策略的仿真和硬件实验进行了验证。 

---
# MetAdv: A Unified and Interactive Adversarial Testing Platform for Autonomous Driving 

**Title (ZH)**: MetAdv：自动驾驶统一交互式 adversarial 测试平台 

**Authors**: Aishan Liu, Jiakai Wang, Tianyuan Zhang, Hainan Li, Jiangfan Liu, Siyuan Liang, Yilong Ren, Xianglong Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2508.06534)  

**Abstract**: Evaluating and ensuring the adversarial robustness of autonomous driving (AD) systems is a critical and unresolved challenge. This paper introduces MetAdv, a novel adversarial testing platform that enables realistic, dynamic, and interactive evaluation by tightly integrating virtual simulation with physical vehicle feedback. At its core, MetAdv establishes a hybrid virtual-physical sandbox, within which we design a three-layer closed-loop testing environment with dynamic adversarial test evolution. This architecture facilitates end-to-end adversarial evaluation, ranging from high-level unified adversarial generation, through mid-level simulation-based interaction, to low-level execution on physical vehicles. Additionally, MetAdv supports a broad spectrum of AD tasks, algorithmic paradigms (e.g., modular deep learning pipelines, end-to-end learning, vision-language models). It supports flexible 3D vehicle modeling and seamless transitions between simulated and physical environments, with built-in compatibility for commercial platforms such as Apollo and Tesla. A key feature of MetAdv is its human-in-the-loop capability: besides flexible environmental configuration for more customized evaluation, it enables real-time capture of physiological signals and behavioral feedback from drivers, offering new insights into human-machine trust under adversarial conditions. We believe MetAdv can offer a scalable and unified framework for adversarial assessment, paving the way for safer AD. 

**Abstract (ZH)**: 评估和确保自动驾驶（AD）系统的对抗鲁棒性是一项关键且未解决的挑战。本文介绍了MetAdv，这是一种新型的对抗性测试平台，通过紧密整合虚拟仿真与物理车辆反馈，实现现实、动态和交互的评估。MetAdv的核心是在一个混合虚拟-物理沙盒中建立三层闭环测试环境，并实现动态对抗性测试进化。该架构支持从高层统一对抗生成，到中层基于模拟的交互，再到低层在物理车辆上的执行的端到端对抗性评估。此外，MetAdv支持广泛的AD任务和算法范式（例如模块化深度学习管道、端到端学习、视觉-语言模型）。它支持灵活的3D车辆建模，并在模拟和物理环境之间实现无缝过渡，内置兼容性适用于Apollo和Tesla等商用平台。MetAdv的一个关键特性是其人机在环的能力：除了灵活的环境配置以实现更多定制化的评估之外，它还能够实时捕捉驾驶员的生理信号和行为反馈，提供在对抗条件下人机信任的新见解。我们认为MetAdv可以提供一个可扩展和统一的框架来对抗性评估，为更安全的AD铺平道路。 

---
# Stinger Robot: A Self-Bracing Robotic Platform for Autonomous Drilling in Confined Underground Environments 

**Title (ZH)**: 刺针机器人：一种自支撑机器人平台，用于受限地下环境中的自主钻探。 

**Authors**: H. Liu, L. S. Moreu, T. S. Andersen, V. V. Puche, M. Fumagalli  

**Link**: [PDF](https://arxiv.org/pdf/2508.06521)  

**Abstract**: The increasing demand for critical raw materials has revitalized interest in abandoned underground mines, which pose extreme challenges for conventional drilling machinery due to confined, unstructured, and infrastructure-less environments. This paper presents the Stinger Robot, a novel compact robotic platform specifically designed for autonomous high-force drilling in such settings. The robot features a mechanically self-locking tri-leg bracing mechanism that enables stable anchoring to irregular tunnel surfaces. A key innovation lies in its force-aware, closed-loop control strategy, which enables force interaction with unstructured environments during bracing and drilling. Implemented as a finite-state machine in ROS 2, the control policy dynamically adapts leg deployment based on real-time contact feedback and load thresholds, ensuring stability without external supports. We demonstrate, through simulation and preliminary hardware tests, that the Stinger Robot can autonomously stabilize and drill in conditions previously inaccessible to nowadays mining machines. This work constitutes the first validated robotic architecture to integrate distributed force-bracing and autonomous drilling in underground environments, laying the groundwork for future collaborative mining operations using modular robot systems. 

**Abstract (ZH)**: 基于自主高力钻探的Stinger机器人：地下废弃矿井中的新型紧凑型机器人平台 

---
# Optimization of Flip-Landing Trajectories for Starship based on a Deep Learned Simulator 

**Title (ZH)**: 基于深度学习模拟器的星舰翻转着陆轨迹优化 

**Authors**: Liwei Chen, Tong Qin, Zhenhua Huangfu, Li Li, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2508.06520)  

**Abstract**: We propose a differentiable optimization framework for flip-and-landing trajectory design of reusable spacecraft, exemplified by the Starship vehicle. A deep neural network surrogate, trained on high-fidelity CFD data, predicts aerodynamic forces and moments, and is tightly coupled with a differentiable rigid-body dynamics solver. This enables end-to-end gradient-based trajectory optimization without linearization or convex relaxation. The framework handles actuator limits and terminal landing constraints, producing physically consistent, optimized control sequences. Both standard automatic differentiation and Neural ODEs are applied to support long-horizon rollouts. Results demonstrate the framework's effectiveness in modeling and optimizing complex maneuvers with high nonlinearities. This work lays the groundwork for future extensions involving unsteady aerodynamics, plume interactions, and intelligent guidance design. 

**Abstract (ZH)**: 我们提出了一种可微优化框架，用于可重复使用航天器（以Starship为例）的翻转和着陆轨迹设计，并通过可微刚体动力学求解器紧密耦合一个基于高保真CFD数据训练的深度神经网络代理，以实现端到端基于梯度的轨迹优化，无需线性化或凸松弛处理。该框架能够处理执行器限值和终端着陆约束，并产生物理上一致的优化控制序列。应用标准自动微分和神经ODE支持长时序模拟。结果表明，该框架在建模和优化具有高度非线性的复杂机动中具有有效性。该工作为未来涉及不稳态空气动力学、羽流相互作用和智能引导设计的扩展奠定了基础。 

---
# Automated Seam Folding and Sewing Machine on Pleated Pants for Apparel Manufacturing 

**Title (ZH)**: 自动缝褶折叠机及其在褶皱 Pants 生产中的应用 

**Authors**: Ray Wai Man Kong  

**Link**: [PDF](https://arxiv.org/pdf/2508.06518)  

**Abstract**: The applied research is the design and development of an automated folding and sewing machine for pleated pants. It represents a significant advancement in addressing the challenges associated with manual sewing processes. Traditional methods for creating pleats are labour-intensive, prone to inconsistencies, and require high levels of skill, making automation a critical need in the apparel industry. This research explores the technical feasibility and operational benefits of integrating advanced technologies into garment production, focusing on the creation of an automated machine capable of precise folding and sewing operations and eliminating the marking operation.
The proposed machine incorporates key features such as a precision folding mechanism integrated into the automated sewing unit with real-time monitoring capabilities. The results demonstrate remarkable improvements: the standard labour time has been reduced by 93%, dropping from 117 seconds per piece to just 8 seconds with the automated system. Similarly, machinery time improved by 73%, and the total output rate increased by 72%. These enhancements translate into a cycle time reduction from 117 seconds per piece to an impressive 33 seconds, enabling manufacturers to meet customer demand more swiftly. By eliminating manual marking processes, the machine not only reduces labour costs but also minimizes waste through consistent pleat formation. This automation aligns with industry trends toward sustainability and efficiency, potentially reducing environmental impact by decreasing material waste and energy consumption. 

**Abstract (ZH)**: 应用于褶皱裤子的自动化折叠与缝纫机的设计与开发：传统缝制褶皱的劳动密集型方法存在一致性差、技能要求高等问题，自动化是服装行业迫切需要的解决方案。本研究探讨将先进技術集成到服装生产中的技术可行性和操作优势，重点在于开发一种能够实现精密折叠和缝纫操作的自动化机器，消除标记过程。 

---
# Emergent morphogenesis via planar fabrication enabled by a reduced model of composites 

**Title (ZH)**: 复合材料简化模型驱动的平面制造诱发形态发生 

**Authors**: Yupeng Zhang, Adam Alon, M. Khalid Jawed  

**Link**: [PDF](https://arxiv.org/pdf/2508.08198)  

**Abstract**: The ability to engineer complex three-dimensional shapes from planar sheets with precise, programmable control underpins emerging technologies in soft robotics, reconfigurable devices, and functional materials. Here, we present a reduced-order numerical and experimental framework for a bilayer system consisting of a stimuli-responsive thermoplastic sheet (Shrinky Dink) bonded to a kirigami-patterned, inert plastic layer. Upon uniform heating, the active layer contracts while the patterned layer constrains in-plane stretch but allows out-of-plane bending, yielding programmable 3D morphologies from simple planar precursors. Our approach enables efficient computational design and scalable manufacturing of 3D forms with a single-layer reduced model that captures the coupled mechanics of stretching and bending. Unlike traditional bilayer modeling, our framework collapses the multilayer composite into a single layer of nodes and elements, reducing the degrees of freedom and enabling simulation on a 2D geometry. This is achieved by introducing a novel energy formulation that captures the coupling between in-plane stretch mismatch and out-of-plane bending - extending beyond simple isotropic linear elastic models. Experimentally, we establish a fully planar, repeatable fabrication protocol using a stimuli-responsive thermoplastic and a laser-cut inert plastic layer. The programmed strain mismatch drives an array of 3D morphologies, such as bowls, canoes, and flower petals, all verified by both simulation and physical prototypes. 

**Abstract (ZH)**: 从平面片材精确编程生成复杂三维形状的减阶数值和实验框架：基于刺激响应热塑性薄膜和 kirigami 阵列的双层系统 

---
# Multi-view Normal and Distance Guidance Gaussian Splatting for Surface Reconstruction 

**Title (ZH)**: 多视图法normal和距离指导的高斯点云表面重建 

**Authors**: Bo Jia, Yanan Guo, Ying Chang, Benkui Zhang, Ying Xie, Kangning Du, Lin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.07701)  

**Abstract**: 3D Gaussian Splatting (3DGS) achieves remarkable results in the field of surface reconstruction. However, when Gaussian normal vectors are aligned within the single-view projection plane, while the geometry appears reasonable in the current view, biases may emerge upon switching to nearby views. To address the distance and global matching challenges in multi-view scenes, we design multi-view normal and distance-guided Gaussian splatting. This method achieves geometric depth unification and high-accuracy reconstruction by constraining nearby depth maps and aligning 3D normals. Specifically, for the reconstruction of small indoor and outdoor scenes, we propose a multi-view distance reprojection regularization module that achieves multi-view Gaussian alignment by computing the distance loss between two nearby views and the same Gaussian surface. Additionally, we develop a multi-view normal enhancement module, which ensures consistency across views by matching the normals of pixel points in nearby views and calculating the loss. Extensive experimental results demonstrate that our method outperforms the baseline in both quantitative and qualitative evaluations, significantly enhancing the surface reconstruction capability of 3DGS. 

**Abstract (ZH)**: 多视图法向和距离引导的3D高斯点云实现多视图场景的几何深度统一和高精度重建 

---
# AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning 

**Title (ZH)**: AR-VRM：通过类比推理模仿人类运动以实现视觉机器人操作 

**Authors**: Dejie Yang, Zijing Zhao, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.07626)  

**Abstract**: Visual Robot Manipulation (VRM) aims to enable a robot to follow natural language instructions based on robot states and visual observations, and therefore requires costly multi-modal data. To compensate for the deficiency of robot data, existing approaches have employed vision-language pretraining with large-scale data. However, they either utilize web data that differs from robotic tasks, or train the model in an implicit way (e.g., predicting future frames at the pixel level), thus showing limited generalization ability under insufficient robot data. In this paper, we propose to learn from large-scale human action video datasets in an explicit way (i.e., imitating human actions from hand keypoints), introducing Visual Robot Manipulation with Analogical Reasoning (AR-VRM). To acquire action knowledge explicitly from human action videos, we propose a keypoint Vision-Language Model (VLM) pretraining scheme, enabling the VLM to learn human action knowledge and directly predict human hand keypoints. During fine-tuning on robot data, to facilitate the robotic arm in imitating the action patterns of human motions, we first retrieve human action videos that perform similar manipulation tasks and have similar historical observations , and then learn the Analogical Reasoning (AR) map between human hand keypoints and robot components. Taking advantage of focusing on action keypoints instead of irrelevant visual cues, our method achieves leading performance on the CALVIN benchmark {and real-world experiments}. In few-shot scenarios, our AR-VRM outperforms previous methods by large margins , underscoring the effectiveness of explicitly imitating human actions under data scarcity. 

**Abstract (ZH)**: 视觉机器人操作（视觉机器人操作-类比推理，AR-VRM）：一种显式学习人类动作知识的方法 

---
# Noise-Aware Generative Microscopic Traffic Simulation 

**Title (ZH)**: 噪声感知生成微观交通模拟 

**Authors**: Vindula Jayawardana, Catherine Tang, Junyi Ji, Jonah Philion, Xue Bin Peng, Cathy Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.07453)  

**Abstract**: Accurately modeling individual vehicle behavior in microscopic traffic simulation remains a key challenge in intelligent transportation systems, as it requires vehicles to realistically generate and respond to complex traffic phenomena such as phantom traffic jams. While traditional human driver simulation models offer computational tractability, they do so by abstracting away the very complexity that defines human driving. On the other hand, recent advances in infrastructure-mounted camera-based roadway sensing have enabled the extraction of vehicle trajectory data, presenting an opportunity to shift toward generative, agent-based models. Yet, a major bottleneck remains: most existing datasets are either overly sanitized or lack standardization, failing to reflect the noisy, imperfect nature of real-world sensing. Unlike data from vehicle-mounted sensors-which can mitigate sensing artifacts like occlusion through overlapping fields of view and sensor fusion-infrastructure-based sensors surface a messier, more practical view of challenges that traffic engineers encounter. To this end, we present the I-24 MOTION Scenario Dataset (I24-MSD)-a standardized, curated dataset designed to preserve a realistic level of sensor imperfection, embracing these errors as part of the learning problem rather than an obstacle to overcome purely from preprocessing. Drawing from noise-aware learning strategies in computer vision, we further adapt existing generative models in the autonomous driving community for I24-MSD with noise-aware loss functions. Our results show that such models not only outperform traditional baselines in realism but also benefit from explicitly engaging with, rather than suppressing, data imperfection. We view I24-MSD as a stepping stone toward a new generation of microscopic traffic simulation that embraces the real-world challenges and is better aligned with practical needs. 

**Abstract (ZH)**: 准确 modeling 个体车辆行为在微观交通模拟中的建模仍然是智能运输系统中的一个关键挑战，因为它要求车辆能够真实地生成和响应如幽灵交通拥堵等复杂的交通现象。虽然传统的驾驶员模拟模型在计算上具有可操作性，但它们通过忽略定义人类驾驶的复杂性来实现这一点。另一方面，最近基础设施安装的基于摄像头的道路传感技术的进步使得提取车辆轨迹数据成为可能，这为转向生成性的、基于代理的模型提供了机会。然而，一个主要瓶颈仍然存在：大多数现有数据集要么过度清洗，要么缺乏标准化，无法反映真实世界传感的噪声和不完美性。与安装在车辆上的传感器数据相比——这些数据可以通过重叠的视场和传感器融合来缓解诸如遮挡之类的传感伪像——基础设施传感器揭示了交通工程师面临的更复杂、更实际的挑战。为此，我们提出了I-24 MOTION情景数据集（I24-MSD）——一个标准化、精心编制的数据集，旨在保留传感器不完美的现实水平，将这些错误视为学习问题的一部分，而非纯粹通过预处理克服的障碍。借鉴计算机视觉中的噪声感知学习策略，我们进一步调整了自主驾驶社区中的现有生成模型，为I24-MSD引入了噪声感知损失函数。我们的结果显示，这样的模型不仅在现实性方面超越了传统的基线模型，还因其明确地与数据不完美性互动而受益。我们视I24-MSD为迈向新一代微观交通模拟的踏脚石，这种模拟能够接纳现实世界的挑战，并更好地满足实践需求。 

---
# The 2D+ Dynamic Articulatory Model DYNARTmo: Tongue-Palate Contact Area Estimation 

**Title (ZH)**: DYNARTmo：舌腭接触面积估计的2D+动态发音模型 

**Authors**: Bernd J. Kröger  

**Link**: [PDF](https://arxiv.org/pdf/2508.07262)  

**Abstract**: This paper describes an extension of the two-dimensional dynamic articulatory model DYNARTmo by integrating an internal three-dimensional representation of the palatal dome to estimate tongue-palate contact areas from midsagittal tongue contours. Two alternative dome geometries - a half-ellipse and a cosine based profile - are implemented to model lateral curvature in the coronal plane. Using these geometries, lateral contact points are analytically computed for each anterior-posterior position, enabling the generation of electropalatography-like visualizations within the 2D+ framework. The enhanced model supports three synchronized views (sagittal, glottal, and palatal) for static and dynamic (animated) articulation displays, suitable for speech science education and speech therapy. Future work includes adding a facial (lip) view and implementing articulatory-to-acoustic synthesis to quantitatively evaluate model realism. 

**Abstract (ZH)**: 本文描述了通过集成腭穹隆的内部三维表示来扩展二维动态发音模型DYNARTmo，以从矢状位舌轮廓估算舌-腭接触区域。实施了两种替代的穹隆几何形状——半椭圆和基于余弦的截面，以在冠状面建模侧向曲率。借助这些几何形状，可以为每个前后位置计算侧向接触点，从而在2D+框架内生成类似电腭图的可视化结果。增强后的模型支持同步显示（矢状、声门和腭视图）静态和动态（动画）发音展示，适用于语音科学教育和言语治疗。未来工作包括添加面部（唇部）视图并实现发音到声学的合成，以定量评估模型的真实性。 

---
# ForeSight: Multi-View Streaming Joint Object Detection and Trajectory Forecasting 

**Title (ZH)**: ForeSight: 多视图流式联合物体检测与轨迹预测 

**Authors**: Sandro Papais, Letian Wang, Brian Cheong, Steven L. Waslander  

**Link**: [PDF](https://arxiv.org/pdf/2508.07089)  

**Abstract**: We introduce ForeSight, a novel joint detection and forecasting framework for vision-based 3D perception in autonomous vehicles. Traditional approaches treat detection and forecasting as separate sequential tasks, limiting their ability to leverage temporal cues. ForeSight addresses this limitation with a multi-task streaming and bidirectional learning approach, allowing detection and forecasting to share query memory and propagate information seamlessly. The forecast-aware detection transformer enhances spatial reasoning by integrating trajectory predictions from a multiple hypothesis forecast memory queue, while the streaming forecast transformer improves temporal consistency using past forecasts and refined detections. Unlike tracking-based methods, ForeSight eliminates the need for explicit object association, reducing error propagation with a tracking-free model that efficiently scales across multi-frame sequences. Experiments on the nuScenes dataset show that ForeSight achieves state-of-the-art performance, achieving an EPA of 54.9%, surpassing previous methods by 9.3%, while also attaining the best mAP and minADE among multi-view detection and forecasting models. 

**Abstract (ZH)**: 我们介绍了ForeSight，这是一种用于自主车辆视觉三维感知的新颖联合检测与预测框架。传统的做法将检测和预测视为分离的顺序任务，限制了它们利用时间线索的能力。ForeSight 通过多任务流式和双向学习方法解决了这一限制，使得检测和预测能够共享查询内存并无缝传递信息。预测感知检测变压器通过整合多种假设预测轨迹记忆队列中的轨迹预测来增强空间推理，而流式预测变压器则利用过去预测和精化检测提高时间一致性。与基于跟踪的方法不同，ForeSight 消除了显式对象关联的需要，通过一个无需跟踪的模型在多帧序列上实现高效扩展，减少了误差传播。在nuScenes数据集上的实验表明，ForeSight 达到了最先进的性能，EPA 达到了 54.9%，比之前的方法高出 9.3%，同时在多视图检测与预测模型中也取得了最佳的 mAP 和 minADE。 

---
# From Imitation to Optimization: A Comparative Study of Offline Learning for Autonomous Driving 

**Title (ZH)**: 从模仿到优化：离线学习在自主驾驶中的比较研究 

**Authors**: Antonio Guillen-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2508.07029)  

**Abstract**: Learning robust driving policies from large-scale, real-world datasets is a central challenge in autonomous driving, as online data collection is often unsafe and impractical. While Behavioral Cloning (BC) offers a straightforward approach to imitation learning, policies trained with BC are notoriously brittle and suffer from compounding errors in closed-loop execution. This work presents a comprehensive pipeline and a comparative study to address this limitation. We first develop a series of increasingly sophisticated BC baselines, culminating in a Transformer-based model that operates on a structured, entity-centric state representation. While this model achieves low imitation loss, we show that it still fails in long-horizon simulations. We then demonstrate that by applying a state-of-the-art Offline Reinforcement Learning algorithm, Conservative Q-Learning (CQL), to the same data and architecture, we can learn a significantly more robust policy. Using a carefully engineered reward function, the CQL agent learns a conservative value function that enables it to recover from minor errors and avoid out-of-distribution states. In a large-scale evaluation on 1,000 unseen scenarios from the Waymo Open Motion Dataset, our final CQL agent achieves a 3.2x higher success rate and a 7.4x lower collision rate than the strongest BC baseline, proving that an offline RL approach is critical for learning robust, long-horizon driving policies from static expert data. 

**Abstract (ZH)**: 从大规模真实世界数据中学习稳健的驾驶策略是自动驾驶领域的核心挑战，因为在线数据收集往往不安全且不实用。行为克隆（BC）虽提供了一种直接的方法来实现imitation learning，但用BC训练出的策略极易出错，并在闭环执行中积聚错误。本文提出了一整套的解决方案和对比研究，首先开发了一系列日益复杂的BC基线模型，最终形成了基于Transformer的模型，该模型能够处理结构化的、实体中心的状态表示。尽管该模型在模仿损失上表现出色，但在长时仿真中依然无法通过。接着，通过将当前最先进的离线强化学习算法保守Q学习（CQL）应用于相同数据和架构，我们能够学习到更稳健的策略。通过精心设计的奖励函数，CQL代理学会了保守的价值函数，使它能够从轻微的错误中恢复并且避免抽样外状态。在Waymo Open Motion数据集的1,000个未见过的场景的大规模评估中，最终的CQL代理的成功率提高了3.2倍，碰撞率降低了7.4倍，证明了离线强化学习方法对于从静态专家数据中学习稳健的、长时间段的驾驶策略至关重要。 

---
# PANAMA: A Network-Aware MARL Framework for Multi-Agent Path Finding in Digital Twin Ecosystems 

**Title (ZH)**: PANAMA：数字孪生生态系统中多代理路径查找的网络感知MARL框架 

**Authors**: Arman Dogru, R. Irem Bor-Yaliniz, Nimal Gamini Senarath  

**Link**: [PDF](https://arxiv.org/pdf/2508.06767)  

**Abstract**: Digital Twins (DTs) are transforming industries through advanced data processing and analysis, positioning the world of DTs, Digital World, as a cornerstone of nextgeneration technologies including embodied AI. As robotics and automated systems scale, efficient data-sharing frameworks and robust algorithms become critical. We explore the pivotal role of data handling in next-gen networks, focusing on dynamics between application and network providers (AP/NP) in DT ecosystems. We introduce PANAMA, a novel algorithm with Priority Asymmetry for Network Aware Multi-agent Reinforcement Learning (MARL) based multi-agent path finding (MAPF). By adopting a Centralized Training with Decentralized Execution (CTDE) framework and asynchronous actor-learner architectures, PANAMA accelerates training while enabling autonomous task execution by embodied AI. Our approach demonstrates superior pathfinding performance in accuracy, speed, and scalability compared to existing benchmarks. Through simulations, we highlight optimized data-sharing strategies for scalable, automated systems, ensuring resilience in complex, real-world environments. PANAMA bridges the gap between network-aware decision-making and robust multi-agent coordination, advancing the synergy between DTs, wireless networks, and AI-driven automation. 

**Abstract (ZH)**: 数字孪生（DTs）正在通过先进的数据处理和分析重塑产业，将数字孪生世界定位为下一代技术，包括具身AI的核心支柱。随着机器人和自动化系统的扩展，高效的数据共享框架和 robust 算法变得至关重要。我们探讨了在数字孪生生态系统中数据处理的关键作用，重点关注应用和网络提供商之间的动态关系。我们介绍了PANAMA算法，这是一种具有网络感知优先异构性的多智能体强化学习多智能体路径查找算法。通过采用集中训练与分布式执行框架及异步演员-学习者架构，PANAMA加速了训练过程并允许具身AI自主执行任务。我们的方法在准确度、速度和可扩展性方面优于现有基准。通过仿真，我们强调了可扩展自动化系统的优化数据共享策略，确保在复杂真实环境中的弹性。PANAMA填补了网络感知决策与稳健多智能体协调之间的空白，推动了数字孪生、无线网络和AI驱动自动化之间的协同进步。 

---
# IRL-VLA: Training an Vision-Language-Action Policy via Reward World Model 

**Title (ZH)**: IRL-VLA：通过奖励世界模型训练视觉-语言-动作策略 

**Authors**: Anqing Jiang, Yu Gao, Yiru Wang, Zhigang Sun, Shuo Wang, Yuwen Heng, Hao Sun, Shichen Tang, Lijuan Zhu, Jinhao Chai, Jijun Wang, Zichong Gu, Hao Jiang, Li Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.06571)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated potential in autonomous driving. However, two critical challenges hinder their development: (1) Existing VLA architectures are typically based on imitation learning in open-loop setup which tends to capture the recorded behaviors in the dataset, leading to suboptimal and constrained performance, (2) Close-loop training relies heavily on high-fidelity sensor simulation, where domain gaps and computational inefficiencies pose significant barriers. In this paper, we introduce IRL-VLA, a novel close-loop Reinforcement Learning via \textbf{I}nverse \textbf{R}einforcement \textbf{L}earning reward world model with a self-built VLA approach. Our framework proceeds in a three-stage paradigm: In the first stage, we propose a VLA architecture and pretrain the VLA policy via imitation learning. In the second stage, we construct a lightweight reward world model via inverse reinforcement learning to enable efficient close-loop reward computation. To further enhance planning performance, finally, we design specialized reward world model guidence reinforcement learning via PPO(Proximal Policy Optimization) to effectively balance the safety incidents, comfortable driving, and traffic efficiency. Our approach achieves state-of-the-art performance in NAVSIM v2 end-to-end driving benchmark, 1st runner up in CVPR2025 Autonomous Grand Challenge. We hope that our framework will accelerate VLA research in close-loop autonomous driving. 

**Abstract (ZH)**: 基于逆强化学习的Close-loop 视觉-语言-动作（VLA）模型 

---
# Historical Prediction Attention Mechanism based Trajectory Forecasting for Proactive Work Zone Safety in a Digital Twin Environment 

**Title (ZH)**: 基于历史预测注意力机制的轨迹预测在数字孪生环境中的主动工作区安全 

**Authors**: Minhaj Uddin Ahmad, Mizanur Rahman, Alican Sevim, David Bodoh, Sakib Khan, Li Zhao, Nathan Huynh, Eren Erman Ozguven  

**Link**: [PDF](https://arxiv.org/pdf/2508.06544)  

**Abstract**: Proactive safety systems aim to mitigate risks by anticipating potential conflicts between vehicles and enabling early intervention to prevent work zone-related crashes. This study presents an infrastructure-enabled proactive work zone safety warning system that leverages a Digital Twin environment, integrating real-time multi-sensor data, detailed High-Definition (HD) maps, and a historical prediction attention mechanism-based trajectory prediction model. Using a co-simulation environment that combines Simulation of Urban MObility (SUMO) and CAR Learning to Act (CARLA) simulators, along with Lanelet2 HD maps and the Historical Prediction Network (HPNet) model, we demonstrate effective trajectory prediction and early warning generation for vehicle interactions in freeway work zones. To evaluate the accuracy of predicted trajectories, we use two standard metrics: Joint Average Displacement Error (ADE) and Joint Final Displacement Error (FDE). Specifically, the infrastructure-enabled HPNet model demonstrates superior performance on the work-zone datasets generated from the co-simulation environment, achieving a minimum Joint FDE of 0.3228 meters and a minimum Joint ADE of 0.1327 meters, lower than the benchmarks on the Argoverse (minJointFDE: 1.0986 m, minJointADE: 0.7612 m) and Interaction (minJointFDE: 0.8231 m, minJointADE: 0.2548 m) datasets. In addition, our proactive safety warning generation application, utilizing vehicle bounding boxes and probabilistic conflict modeling, demonstrates its capability to issue alerts for potential vehicle conflicts. 

**Abstract (ZH)**: 基于基础设施的主动工作区安全预警系统：利用数字孪生环境实现高速公路工作区车辆轨迹预测与早期预警 

---
