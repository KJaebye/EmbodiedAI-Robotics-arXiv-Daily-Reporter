# SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning 

**Title (ZH)**: SimpleVLA-RL：通过强化学习扩展VLA训练 

**Authors**: Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.09674)  

**Abstract**: Vision-Language-Action (VLA) models have recently emerged as a powerful paradigm for robotic manipulation. Despite substantial progress enabled by large-scale pretraining and supervised fine-tuning (SFT), these models face two fundamental challenges: (i) the scarcity and high cost of large-scale human-operated robotic trajectories required for SFT scaling, and (ii) limited generalization to tasks involving distribution shift. Recent breakthroughs in Large Reasoning Models (LRMs) demonstrate that reinforcement learning (RL) can dramatically enhance step-by-step reasoning capabilities, raising a natural question: Can RL similarly improve the long-horizon step-by-step action planning of VLA? In this work, we introduce SimpleVLA-RL, an efficient RL framework tailored for VLA models. Building upon veRL, we introduce VLA-specific trajectory sampling, scalable parallelization, multi-environment rendering, and optimized loss computation. When applied to OpenVLA-OFT, SimpleVLA-RL achieves SoTA performance on LIBERO and even outperforms $\pi_0$ on RoboTwin 1.0\&2.0 with the exploration-enhancing strategies we introduce. SimpleVLA-RL not only reduces dependence on large-scale data and enables robust generalization, but also remarkably surpasses SFT in real-world tasks. Moreover, we identify a novel phenomenon ``pushcut'' during RL training, wherein the policy discovers previously unseen patterns beyond those seen in the previous training process. Github: this https URL 

**Abstract (ZH)**: 基于视觉-语言-动作的强化学习框架：SimpleVLA-RL 

---
# Dexplore: Scalable Neural Control for Dexterous Manipulation from Reference-Scoped Exploration 

**Title (ZH)**: Dexplore: 面向参考范围探索的可扩展神经控制方法用于灵巧操作 

**Authors**: Sirui Xu, Yu-Wei Chao, Liuyu Bian, Arsalan Mousavian, Yu-Xiong Wang, Liang-Yan Gui, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09671)  

**Abstract**: Hand-object motion-capture (MoCap) repositories offer large-scale, contact-rich demonstrations and hold promise for scaling dexterous robotic manipulation. Yet demonstration inaccuracies and embodiment gaps between human and robot hands limit the straightforward use of these data. Existing methods adopt a three-stage workflow, including retargeting, tracking, and residual correction, which often leaves demonstrations underused and compound errors across stages. We introduce Dexplore, a unified single-loop optimization that jointly performs retargeting and tracking to learn robot control policies directly from MoCap at scale. Rather than treating demonstrations as ground truth, we use them as soft guidance. From raw trajectories, we derive adaptive spatial scopes, and train with reinforcement learning to keep the policy in-scope while minimizing control effort and accomplishing the task. This unified formulation preserves demonstration intent, enables robot-specific strategies to emerge, improves robustness to noise, and scales to large demonstration corpora. We distill the scaled tracking policy into a vision-based, skill-conditioned generative controller that encodes diverse manipulation skills in a rich latent representation, supporting generalization across objects and real-world deployment. Taken together, these contributions position Dexplore as a principled bridge that transforms imperfect demonstrations into effective training signals for dexterous manipulation. 

**Abstract (ZH)**: Dexplore: A Unified Optimization for Scaling Hand-Object Motion-Capture Demonstrations to Dexterous Robotic Manipulation 

---
# MOFU: Development of a MOrphing Fluffy Unit with Expansion and Contraction Capabilities and Evaluation of the Animacy of Its Movements 

**Title (ZH)**: MOFU：具有扩展与收缩能力的变形毛绒单元开发及其运动拟人性评价 

**Authors**: Taisei Mogi, Mari Saito, Yoshihiro Nakata  

**Link**: [PDF](https://arxiv.org/pdf/2509.09613)  

**Abstract**: Robots for therapy and social interaction are often intended to evoke "animacy" in humans. While many robots imitate appearance and joint movements, little attention has been given to whole-body expansion-contraction, volume-changing movements observed in living organisms, and their effect on animacy perception. We developed a mobile robot called "MOFU (Morphing Fluffy Unit)," capable of whole-body expansion-contraction with a single motor and covered with a fluffy exterior. MOFU employs a "Jitterbug" structure, a geometric transformation mechanism that enables smooth volume change in diameter from 210 to 280 mm using one actuator. It is also equipped with a differential two-wheel drive mechanism for locomotion. To evaluate the effect of expansion-contraction movements, we conducted an online survey using videos of MOFU's behavior. Participants rated impressions with the Godspeed Questionnaire Series. First, we compared videos of MOFU in a stationary state with and without expansion-contraction and turning, finding that expansion-contraction significantly increased perceived animacy. Second, we hypothesized that presenting two MOFUs would increase animacy compared with a single robot; however, this was not supported, as no significant difference emerged. Exploratory analyses further compared four dual-robot motion conditions. Third, when expansion-contraction was combined with locomotion, animacy ratings were higher than locomotion alone. These results suggest that volume-changing movements such as expansion and contraction enhance perceived animacy in robots and should be considered an important design element in future robot development aimed at shaping human impressions. 

**Abstract (ZH)**: 用于治疗和社会互动的机器人常常旨在唤起人类的“生命力”。尽管许多机器人模仿外观和关节运动，但很少有关注全身扩张收缩、体积变化的运动及其对生命力感知的影响。我们开发了一种名为“MOFU（形态绒毛单元）”的移动机器人，能够通过单一马达实现全身扩张收缩，并覆盖有绒毛外层。MOFU采用“Jitterbug”结构，这是一种几何变换机制，可在直径从210毫米至280毫米之间平滑变化，仅需一个执行器。它还配备了差动双轮驱动机制以实现移动。为了评估扩张收缩运动的效果，我们使用MOFU行为的视频进行了在线调查，并使用Godspeed问卷系列对参与者进行了评价。首先，我们将MOFU在静止状态和有无扩张收缩及转向的视频进行比较，发现扩张收缩显著提高了感知的生命力。其次，我们假设展示两个MOFU会比单个机器人增加生命力，但这一假设未得到支持，因为没有显著差异。进一步的探索性分析比较了四种双机器人运动条件。第三，当扩张收缩与移动结合时，生命力评价高于仅移动的情况。这些结果表明，体积变化运动如扩张和收缩能够增强对机器人生命力的感知，应被视为未来旨在塑造人类印象的机器人设计中一个重要设计元素。 

---
# ObjectReact: Learning Object-Relative Control for Visual Navigation 

**Title (ZH)**: ObjectReact: 学习对象相对控制的视觉导航 

**Authors**: Sourav Garg, Dustin Craggs, Vineeth Bhat, Lachlan Mares, Stefan Podgorski, Madhava Krishna, Feras Dayoub, Ian Reid  

**Link**: [PDF](https://arxiv.org/pdf/2509.09594)  

**Abstract**: Visual navigation using only a single camera and a topological map has recently become an appealing alternative to methods that require additional sensors and 3D maps. This is typically achieved through an "image-relative" approach to estimating control from a given pair of current observation and subgoal image. However, image-level representations of the world have limitations because images are strictly tied to the agent's pose and embodiment. In contrast, objects, being a property of the map, offer an embodiment- and trajectory-invariant world representation. In this work, we present a new paradigm of learning "object-relative" control that exhibits several desirable characteristics: a) new routes can be traversed without strictly requiring to imitate prior experience, b) the control prediction problem can be decoupled from solving the image matching problem, and c) high invariance can be achieved in cross-embodiment deployment for variations across both training-testing and mapping-execution settings. We propose a topometric map representation in the form of a "relative" 3D scene graph, which is used to obtain more informative object-level global path planning costs. We train a local controller, dubbed "ObjectReact", conditioned directly on a high-level "WayObject Costmap" representation that eliminates the need for an explicit RGB input. We demonstrate the advantages of learning object-relative control over its image-relative counterpart across sensor height variations and multiple navigation tasks that challenge the underlying spatial understanding capability, e.g., navigating a map trajectory in the reverse direction. We further show that our sim-only policy is able to generalize well to real-world indoor environments. Code and supplementary material are accessible via project page: this https URL 

**Abstract (ZH)**: 仅使用单个相机和拓扑地图的视觉导航 recently 成为一种有吸引力的替代方法，无需额外传感器和 3D 地图。 

---
# A Neuromorphic Incipient Slip Detection System using Papillae Morphology 

**Title (ZH)**: 基于乳头形态的神经形态早期打滑检测系统 

**Authors**: Yanhui Lu, Zeyu Deng, Stephen J. Redmond, Efi Psomopoulou, Benjamin Ward-Cherrier  

**Link**: [PDF](https://arxiv.org/pdf/2509.09546)  

**Abstract**: Detecting incipient slip enables early intervention to prevent object slippage and enhance robotic manipulation safety. However, deploying such systems on edge platforms remains challenging, particularly due to energy constraints. This work presents a neuromorphic tactile sensing system based on the NeuroTac sensor with an extruding papillae-based skin and a spiking convolutional neural network (SCNN) for slip-state classification. The SCNN model achieves 94.33% classification accuracy across three classes (no slip, incipient slip, and gross slip) in slip conditions induced by sensor motion. Under the dynamic gravity-induced slip validation conditions, after temporal smoothing of the SCNN's final-layer spike counts, the system detects incipient slip at least 360 ms prior to gross slip across all trials, consistently identifying incipient slip before gross slip occurs. These results demonstrate that this neuromorphic system has stable and responsive incipient slip detection capability. 

**Abstract (ZH)**: 基于NeuroTac传感器的神经形态触觉传感系统及其在滑动状态分类中的应用：边缘平台上的早期滑动检测以增强机器人操作安全 

---
# SMapper: A Multi-Modal Data Acquisition Platform for SLAM Benchmarking 

**Title (ZH)**: SMapper：用于SLAM基准测试的多模态数据采集平台 

**Authors**: Pedro Miguel Bastos Soares, Ali Tourani, Miguel Fernandez-Cortizas, Asier Bikandi Noya, Jose Luis Sanchez-Lopez, Holger Voos  

**Link**: [PDF](https://arxiv.org/pdf/2509.09509)  

**Abstract**: Advancing research in fields like Simultaneous Localization and Mapping (SLAM) and autonomous navigation critically depends on reliable and reproducible multimodal datasets. While several influential datasets have driven progress in these domains, they often suffer from limitations in sensing modalities, environmental diversity, and the reproducibility of the underlying hardware setups. To address these challenges, this paper introduces SMapper, a novel open-hardware, multi-sensor platform designed explicitly for, though not limited to, SLAM research. The device integrates synchronized LiDAR, multi-camera, and inertial sensing, supported by a robust calibration and synchronization pipeline that ensures precise spatio-temporal alignment across modalities. Its open and replicable design allows researchers to extend its capabilities and reproduce experiments across both handheld and robot-mounted scenarios. To demonstrate its practicality, we additionally release SMapper-light, a publicly available SLAM dataset containing representative indoor and outdoor sequences. The dataset includes tightly synchronized multimodal data and ground-truth trajectories derived from offline LiDAR-based SLAM with sub-centimeter accuracy, alongside dense 3D reconstructions. Furthermore, the paper contains benchmarking results on state-of-the-art LiDAR and visual SLAM frameworks using the SMapper-light dataset. By combining open-hardware design, reproducible data collection, and comprehensive benchmarking, SMapper establishes a robust foundation for advancing SLAM algorithm development, evaluation, and reproducibility. 

**Abstract (ZH)**: 先进同时定位与地图构建（SLAM）和自主导航领域研究的进步关键依赖于可靠且可再现的多模态数据集。为了应对这些挑战，本文介绍了一种新颖的开源硬件多传感器平台SMapper，该平台专门用于SLAM研究，虽然不局限于SLAM研究领域。设备集成了同步 Lidar、多摄像头和惯性传感器，并通过一个稳健的校准和同步管线确保各模态之间的精确时空对齐。其开源和可再现的设计使研究人员能够扩展其功能并在手持和机器人搭载的场景中再现实验。为了证明其实用性，我们还发布了SMapper-light，这是一个包含代表性室内和室外序列的公开SLAM数据集。该数据集包括精确到亚厘米级别的离线LiDAR SLAM的参考轨迹和密集三维重建。此外，论文使用SMapper-light数据集对先进的LiDAR和视觉SLAM框架进行了基准测试。通过结合开源硬件设计、可再现的数据收集和全面的基准测试，SMapper为SLAM算法的发展、评估和再现性奠定了坚实的基础。 

---
# BagIt! An Adaptive Dual-Arm Manipulation of Fabric Bags for Object Bagging 

**Title (ZH)**: BagIt！一种适应性双臂 manipulator 对织物袋中物体封装的操控方法 

**Authors**: Peng Zhou, Jiaming Qi, Hongmin Wu, Chen Wang, Yizhou Chen, Zeqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09484)  

**Abstract**: Bagging tasks, commonly found in industrial scenarios, are challenging considering deformable bags' complicated and unpredictable nature. This paper presents an automated bagging system from the proposed adaptive Structure-of-Interest (SOI) manipulation strategy for dual robot arms. The system dynamically adjusts its actions based on real-time visual feedback, removing the need for pre-existing knowledge of bag properties. Our framework incorporates Gaussian Mixture Models (GMM) for estimating SOI states, optimization techniques for SOI generation, motion planning via Constrained Bidirectional Rapidly-exploring Random Tree (CBiRRT), and dual-arm coordination using Model Predictive Control (MPC). Extensive experiments validate the capability of our system to perform precise and robust bagging across various objects, showcasing its adaptability. This work offers a new solution for robotic deformable object manipulation (DOM), particularly in automated bagging tasks. Video of this work is available at this https URL. 

**Abstract (ZH)**: 工业场景中常见的袋装任务因其可变形袋子的复杂和不可预测性而具有挑战性。本文提出了一种基于可适应的兴趣结构（SOI） manipulation 策略的自动袋装系统，适用于双臂机器人。系统根据实时视觉反馈动态调整其操作，无需预先了解袋子的性质。我们的框架包括使用高斯混合模型（GMM）估计 SOI 状态、使用优化技术生成 SOI、使用约束双向快速扩展随机树（CBiRRT）进行运动规划，以及使用模型预测控制（MPC）进行双臂协调。广泛实验验证了该系统在各种物体上执行精确和稳健袋装的能力，展示了其适应性。本文为机器人可变形物体操纵（DOM），特别是在自动化袋装任务中提供了一种新的解决方案。相关视频可通过以下链接查看：this https URL。 

---
# A Hybrid Hinge-Beam Continuum Robot with Passive Safety Capping for Real-Time Fatigue Awareness 

**Title (ZH)**: 一种带有被动安全封顶装置的混合铰接梁连续体机器人及其实时疲劳感知技术 

**Authors**: Tongshun Chen, Zezhou Sun, Yanhan Sun, Yuhao Wang, Dezhen Song, Ke Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09404)  

**Abstract**: Cable-driven continuum robots offer high flexibility and lightweight design, making them well-suited for tasks in constrained and unstructured environments. However, prolonged use can induce mechanical fatigue from plastic deformation and material degradation, compromising performance and risking structural failure. In the state of the art, fatigue estimation of continuum robots remains underexplored, limiting long-term operation. To address this, we propose a fatigue-aware continuum robot with three key innovations: (1) a Hybrid Hinge-Beam structure where TwistBeam and BendBeam decouple torsion and bending: passive revolute joints in the BendBeam mitigate stress concentration, while TwistBeam's limited torsional deformation reduces BendBeam stress magnitude, enhancing durability; (2) a Passive Stopper that safely constrains motion via mechanical constraints and employs motor torque sensing to detect corresponding limit torque, ensuring safety and enabling data collection; and (3) a real-time fatigue-awareness method that estimates stiffness from motor torque at the limit pose, enabling online fatigue estimation without additional sensors. Experiments show that the proposed design reduces fatigue accumulation by about 49% compared with a conventional design, while passive mechanical limiting combined with motor-side sensing allows accurate estimation of structural fatigue and damage. These results confirm the effectiveness of the proposed architecture for safe and reliable long-term operation. 

**Abstract (ZH)**: 基于缆驱动的连续体机器人提供了高柔性和轻量化设计，使其适用于受限和未结构化的环境。然而，长期使用会导致由塑性变形和材料老化引起的机械疲劳，这会损害性能并可能导致结构失效。当前技术中，连续体机器人的疲劳估算研究相对不足，限制了其长期运行能力。为应对这一挑战，我们提出了一种具有三个关键创新的设计：（1）一种混合铰链-梁结构，其中TwistBeam和BendBeam解耦扭转和弯曲：BendBeam中的被动回转关节缓解了应力集中，而TwistBeam的有限扭转变形减少了BendBeam的应力幅度，增强了耐用性；（2）一种安全的运动约束器，通过机械约束确保安全，并利用电机扭矩传感检测相应的极限扭矩，确保安全并实现数据收集；（3）一种实时疲劳感知方法，能够从极限姿态的电机扭矩中估算刚度，无需额外传感器即可实现在线疲劳估算。实验结果显示，与传统设计相比，所提出的设计疲劳累积减少了约49%，而结合机械约束和电机侧感知可以准确估计结构疲劳和损伤。这些结果证实了所提出架构在安全可靠的长期运行中的有效性。 

---
# VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model 

**Title (ZH)**: VLA-适配器：一种有效的细粒度多模态模型范式 

**Authors**: Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, Siteng Huang, Yifan Tang, Wenhui Wang, Ru Zhang, Jianyi Liu, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09372)  

**Abstract**: Vision-Language-Action (VLA) models typically bridge the gap between perceptual and action spaces by pre-training a large-scale Vision-Language Model (VLM) on robotic data. While this approach greatly enhances performance, it also incurs significant training costs. In this paper, we investigate how to effectively bridge vision-language (VL) representations to action (A). We introduce VLA-Adapter, a novel paradigm designed to reduce the reliance of VLA models on large-scale VLMs and extensive pre-training. To this end, we first systematically analyze the effectiveness of various VL conditions and present key findings on which conditions are essential for bridging perception and action spaces. Based on these insights, we propose a lightweight Policy module with Bridge Attention, which autonomously injects the optimal condition into the action space. In this way, our method achieves high performance using only a 0.5B-parameter backbone, without any robotic data pre-training. Extensive experiments on both simulated and real-world robotic benchmarks demonstrate that VLA-Adapter not only achieves state-of-the-art level performance, but also offers the fast inference speed reported to date. Furthermore, thanks to the proposed advanced bridging paradigm, VLA-Adapter enables the training of a powerful VLA model in just 8 hours on a single consumer-grade GPU, greatly lowering the barrier to deploying the VLA model. Project page: this https URL. 

**Abstract (ZH)**: Vision-Language-Action (VLA) 模型通常通过在机器人数据上预训练大规模视觉-语言模型（VLM）来弥合感知和动作空间的差距。虽然这种方法大大提升了性能，但也带来了显著的训练成本。本文研究了如何有效将视觉-语言（VL）表示桥接到动作（A）。我们引入了VLA-Adapter，这是一种新型的范式，旨在减少VLA模型对大规模VLM和长时间预训练的依赖。为此，我们首先系统分析了各种VL条件的有效性，并提出了对于弥合感知和动作空间的关键发现。基于这些见解，我们提出了一个轻量级的Policy模块，其中包含Bridge Attention，该模块可自主注入最合适的条件到动作空间中。通过这种方式，我们的方法仅使用一个0.5B参数的骨干模型即可实现高性能，无需任何机器人数据预训练。广泛实验证明，VLA-Adapter不仅达到了最先进的性能，而且还提供了迄今为止报告的最快推理速度。此外，由于提出的先进桥接范式，VLA-Adapter使得仅在一个消费级GPU上只需8小时即可训练出强大的VLA模型，大大降低了部署VLA模型的门槛。 

---
# AGILOped: Agile Open-Source Humanoid Robot for Research 

**Title (ZH)**: AGILOped：敏捷开源人形机器人用于研究 

**Authors**: Grzegorz Ficht, Luis Denninger, Sven Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2509.09364)  

**Abstract**: With academic and commercial interest for humanoid robots peaking, multiple platforms are being developed. Through a high level of customization, they showcase impressive performance. Most of these systems remain closed-source or have high acquisition and maintenance costs, however. In this work, we present AGILOped - an open-source humanoid robot that closes the gap between high performance and accessibility. Our robot is driven by off-the-shelf backdrivable actuators with high power density and uses standard electronic components. With a height of 110 cm and weighing only 14.5 kg, AGILOped can be operated without a gantry by a single person. Experiments in walking, jumping, impact mitigation and getting-up demonstrate its viability for use in research. 

**Abstract (ZH)**: 基于通用智能的开源人形机器人AGILOped：高性能与易用性的桥梁 

---
# OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning 

**Title (ZH)**: OmniEVA：基于任务自适应3D接地和体感意识推理的通用智能体规划器 

**Authors**: Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yuzheng Zhuang, Bowen Yang, He Zhu, Lingfeng Zhang, Pengwei Xie, David Gamaliel Arcos Bravo, Yingxue Zhang, Jianye Hao, Xingyue Quan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09332)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically this http URL address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: this https URL 

**Abstract (ZH)**: 最近的多模态大语言模型（MLLMs）进展为 embodied 智能开启了新机遇，使其能够实现多模态理解、推理和交互，以及连续的空间决策。然而，当前基于MLLM的 embodied 系统面临两个关键限制。首先，几何适应性缺口：仅在二维输入上训练或以硬编码三维几何注入方式训练的模型要么缺乏足够的空间信息，要么在二维泛化上受到限制，导致在具有不同空间需求的任务之间适应性较差。其次， embodied 约束性缺口：前期工作往往忽视了真实机器人物理约束和能力，导致理论上有效的任务计划但在实践中难以实施。为了解决这些缺口，我们引入了 OmniEVA -- 一种通过两项关键创新实现先进 embodied 推理和任务规划的 embodied 多能规划器：（1）任务自适应三维标注机制，引入门控路由器根据上下文要求进行显式的选择性三维融合调节，实现面向多样 embodied 任务的上下文感知三维标注。（2）体态感知推理框架，该框架将任务目标和体态约束联合纳入推理循环中，从而实现既目标导向又可执行的规划决策。广泛的经验结果表明，OmniEVA 不仅在通用体态推理性能上达到了最先进的水平，还在多种下游场景中表现出强大的能力。一系列提出的体态基准测试评估，包括基础任务和复合任务，证实了其稳健的多能规划能力。项目页面：this https URL 

---
# RENet: Fault-Tolerant Motion Control for Quadruped Robots via Redundant Estimator Networks under Visual Collapse 

**Title (ZH)**: RENet: 视觉失效情况下 quadruped 机器人冗余估计算法的容错运动控制 

**Authors**: Yueqi Zhang, Quancheng Qian, Taixian Hou, Peng Zhai, Xiaoyi Wei, Kangmai Hu, Jiafu Yi, Lihua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09283)  

**Abstract**: Vision-based locomotion in outdoor environments presents significant challenges for quadruped robots. Accurate environmental prediction and effective handling of depth sensor noise during real-world deployment remain difficult, severely restricting the outdoor applications of such algorithms. To address these deployment challenges in vision-based motion control, this letter proposes the Redundant Estimator Network (RENet) framework. The framework employs a dual-estimator architecture that ensures robust motion performance while maintaining deployment stability during onboard vision failures. Through an online estimator adaptation, our method enables seamless transitions between estimation modules when handling visual perception uncertainties. Experimental validation on a real-world robot demonstrates the framework's effectiveness in complex outdoor environments, showing particular advantages in scenarios with degraded visual perception. This framework demonstrates its potential as a practical solution for reliable robotic deployment in challenging field conditions. Project website: this https URL 

**Abstract (ZH)**: 基于视觉的四肢机器人在室外环境中的运动控制面临显著挑战。在现场部署中，准确的环境预测和深度传感器噪声的有效处理依然困难，严重限制了此类算法在室外的应用。为解决基于视觉的运动控制在实际部署中的这些挑战，本文提出了冗余估计网络（RENet）框架。该框架采用双估计器架构，确保在机载视觉故障时仍能保持鲁棒的运动性能和部署稳定性。通过在线估计器适应，我们的方法能在处理视觉感知不确定性时实现估计模块的无缝过渡。在真实机器人上的实验验证表明，该框架在复杂室外环境中的有效性，特别是在视觉感知退化的场景中显示出明显优势。该框架展示了解决在恶劣现场条件下可靠的机器人部署问题的潜在实用方案。项目网站：这个 https URL。 

---
# Occupancy-aware Trajectory Planning for Autonomous Valet Parking in Uncertain Dynamic Environments 

**Title (ZH)**: occupancy-aware 自动代客泊车中不确定动态环境下的路径规划 

**Authors**: Farhad Nawaz, Faizan M. Tariq, Sangjae Bae, David Isele, Avinash Singh, Nadia Figueroa, Nikolai Matni, Jovin D'sa  

**Link**: [PDF](https://arxiv.org/pdf/2509.09206)  

**Abstract**: Accurately reasoning about future parking spot availability and integrated planning is critical for enabling safe and efficient autonomous valet parking in dynamic, uncertain environments. Unlike existing methods that rely solely on instantaneous observations or static assumptions, we present an approach that predicts future parking spot occupancy by explicitly distinguishing between initially vacant and occupied spots, and by leveraging the predicted motion of dynamic agents. We introduce a probabilistic spot occupancy estimator that incorporates partial and noisy observations within a limited Field-of-View (FoV) model and accounts for the evolving uncertainty of unobserved regions. Coupled with this, we design a strategy planner that adaptively balances goal-directed parking maneuvers with exploratory navigation based on information gain, and intelligently incorporates wait-and-go behaviors at promising spots. Through randomized simulations emulating large parking lots, we demonstrate that our framework significantly improves parking efficiency, safety margins, and trajectory smoothness compared to existing approaches. 

**Abstract (ZH)**: 准确推理未来停车位可用性并进行集成规划对于在动态和不确定环境中实现安全高效的自主代客泊车至关重要。我们提出了一种方法，通过明确区分初始空闲和占用的停车位，并利用动态代理的预测运动来预测未来停车位占用情况。我们引入了一种概率性停车位占用估计器，该估计器在一个有限视场（FoV）模型中整合了部分和噪声观察数据，并考虑了未观察区域的不断变化的不确定性。结合这一点，我们设计了一种策略规划器，该规划器根据信息增益自适应地平衡目标导向的泊车操作和探索性导航，并在有希望的停车位上智能地采用等待和继续的行为。通过模拟大型停车场的随机仿真，我们展示了与现有方法相比，我们的框架显著提高了泊车效率、安全余量和轨迹平滑度。 

---
# AEOS: Active Environment-aware Optimal Scanning Control for UAV LiDAR-Inertial Odometry in Complex Scenes 

**Title (ZH)**: AEOS: 主动环境aware最优扫描控制for UAV LiDAR-惯性里程计在复杂场景中 

**Authors**: Jianping Li, Xinhang Xu, Zhongyuan Liu, Shenghai Yuan, Muqing Cao, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.09141)  

**Abstract**: LiDAR-based 3D perception and localization on unmanned aerial vehicles (UAVs) are fundamentally limited by the narrow field of view (FoV) of compact LiDAR sensors and the payload constraints that preclude multi-sensor configurations. Traditional motorized scanning systems with fixed-speed rotations lack scene awareness and task-level adaptability, leading to degraded odometry and mapping performance in complex, occluded environments. Inspired by the active sensing behavior of owls, we propose AEOS (Active Environment-aware Optimal Scanning), a biologically inspired and computationally efficient framework for adaptive LiDAR control in UAV-based LiDAR-Inertial Odometry (LIO). AEOS combines model predictive control (MPC) and reinforcement learning (RL) in a hybrid architecture: an analytical uncertainty model predicts future pose observability for exploitation, while a lightweight neural network learns an implicit cost map from panoramic depth representations to guide exploration. To support scalable training and generalization, we develop a point cloud-based simulation environment with real-world LiDAR maps across diverse scenes, enabling sim-to-real transfer. Extensive experiments in both simulation and real-world environments demonstrate that AEOS significantly improves odometry accuracy compared to fixed-rate, optimization-only, and fully learned baselines, while maintaining real-time performance under onboard computational constraints. The project page can be found at this https URL. 

**Abstract (ZH)**: 基于LiDAR的无人机（UAV）三维感知与定位受到紧凑LiDAR传感器狭窄视野和载荷限制的制约，无法配置多传感器系统。传统具有固定旋转速度的电动扫描系统缺乏场景意识和任务级适应性，在复杂遮挡环境中导致里程计和建图性能下降。受猫头鹰主动感测行为的启发，我们提出了AEOS（Active Environment-aware Optimal Scanning），一种生物启发且计算高效的框架，用于基于无人机的LiDAR-惯性里程计（LIO）的自适应LiDAR控制。AEOS结合了模型预测控制（MPC）和强化学习（RL）：通过分析不确定性模型预测未来姿态可观测性，而轻量级神经网络则从全景深度图中学习隐式成本地图以引导探索。为支持可扩展的训练和泛化，我们开发了一个基于点云的模拟环境，包含不同场景的真实LiDAR地图，实现模拟到现实的转移。在仿真和实际环境中的 extensive 实验表明，AEOS 相较于固定速率、仅优化和完全学习基准显著提高了里程计准确性，同时在机载计算约束下保持实时性能。项目页面可访问此网址。 

---
# LIPM-Guided Reinforcement Learning for Stable and Perceptive Locomotion in Bipedal Robots 

**Title (ZH)**: 基于LIPM的强化学习方法实现 bipedal 机器人稳定且具备感知能力的运动控制 

**Authors**: Haokai Su, Haoxiang Luo, Shunpeng Yang, Kaiwen Jiang, Wei Zhang, Hua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.09106)  

**Abstract**: Achieving stable and robust perceptive locomotion for bipedal robots in unstructured outdoor environments remains a critical challenge due to complex terrain geometry and susceptibility to external disturbances. In this work, we propose a novel reward design inspired by the Linear Inverted Pendulum Model (LIPM) to enable perceptive and stable locomotion in the wild. The LIPM provides theoretical guidance for dynamic balance by regulating the center of mass (CoM) height and the torso orientation. These are key factors for terrain-aware locomotion, as they help ensure a stable viewpoint for the robot's camera. Building on this insight, we design a reward function that promotes balance and dynamic stability while encouraging accurate CoM trajectory tracking. To adaptively trade off between velocity tracking and stability, we leverage the Reward Fusion Module (RFM) approach that prioritizes stability when needed. A double-critic architecture is adopted to separately evaluate stability and locomotion objectives, improving training efficiency and robustness. We validate our approach through extensive experiments on a bipedal robot in both simulation and real-world outdoor environments. The results demonstrate superior terrain adaptability, disturbance rejection, and consistent performance across a wide range of speeds and perceptual conditions. 

**Abstract (ZH)**: 在未结构化户外环境中的 bipedal 机器人感知性稳定行走 remains a critical challenge due to complex terrain geometry and susceptibility to external disturbances. In this work, we propose a novel reward design inspired by the Linear Inverted Pendulum Model (LIPM) to enable perceptive and stable locomotion in the wild. 

---
# Kinetostatics and Particle-Swarm Optimization of Vehicle-Mounted Underactuated Metamorphic Loading Manipulators 

**Title (ZH)**: 车载欠驱动 metamorphic 负载 manipulator 的运动静力学与粒子群优化 

**Authors**: Nan Mao, Guanglu Jia, Junpeng Chen, Emmanouil Spyrakos-Papastavridis, Jian S. Dai  

**Link**: [PDF](https://arxiv.org/pdf/2509.09093)  

**Abstract**: Fixed degree-of-freedom (DoF) loading mechanisms often suffer from excessive actuators, complex control, and limited adaptability to dynamic tasks. This study proposes an innovative mechanism of underactuated metamorphic loading manipulators (UMLM), integrating a metamorphic arm with a passively adaptive gripper. The metamorphic arm exploits geometric constraints, enabling the topology reconfiguration and flexible motion trajectories without additional actuators. The adaptive gripper, driven entirely by the arm, conforms to diverse objects through passive compliance. A structural model is developed, and a kinetostatics analysis is conducted to investigate isomorphic grasping configurations. To optimize performance, Particle-Swarm Optimization (PSO) is utilized to refine the gripper's dimensional parameters, ensuring robust adaptability across various applications. Simulation results validate the UMLM's easily implemented control strategy, operational versatility, and effectiveness in grasping diverse objects in dynamic environments. This work underscores the practical potential of underactuated metamorphic mechanisms in applications requiring efficient and adaptable loading solutions. Beyond the specific design, this generalized modeling and optimization framework extends to a broader class of manipulators, offering a scalable approach to the development of robotic systems that require efficiency, flexibility, and robust performance. 

**Abstract (ZH)**: 一种少自由度 metamorphic 加载 manipulator 的创新机制：具有被动自适应 Gripper 的少自由度 metamorphic 加载 manipulator（UMLM） 

---
# KoopMotion: Learning Almost Divergence Free Koopman Flow Fields for Motion Planning 

**Title (ZH)**: KoopMotion: 学习几乎无散度的柯普曼流动场以进行运动规划 

**Authors**: Alice Kate Li, Thales C Silva, Victoria Edwards, Vijay Kumar, M. Ani Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2509.09074)  

**Abstract**: In this work, we propose a novel flow field-based motion planning method that drives a robot from any initial state to a desired reference trajectory such that it converges to the trajectory's end point. Despite demonstrated efficacy in using Koopman operator theory for modeling dynamical systems, Koopman does not inherently enforce convergence to desired trajectories nor to specified goals -- a requirement when learning from demonstrations (LfD). We present KoopMotion which represents motion flow fields as dynamical systems, parameterized by Koopman Operators to mimic desired trajectories, and leverages the divergence properties of the learnt flow fields to obtain smooth motion fields that converge to a desired reference trajectory when a robot is placed away from the desired trajectory, and tracks the trajectory until the end point. To demonstrate the effectiveness of our approach, we show evaluations of KoopMotion on the LASA human handwriting dataset and a 3D manipulator end-effector trajectory dataset, including spectral analysis. We also perform experiments on a physical robot, verifying KoopMotion on a miniature autonomous surface vehicle operating in a non-static fluid flow environment. Our approach is highly sample efficient in both space and time, requiring only 3\% of the LASA dataset to generate dense motion plans. Additionally, KoopMotion provides a significant improvement over baselines when comparing metrics that measure spatial and temporal dynamics modeling efficacy. 

**Abstract (ZH)**: 基于流场的运动规划方法：KoopMotion及其在动态系统中的应用 

---
# Rapid Manufacturing of Lightweight Drone Frames Using Single-Tow Architected Composites 

**Title (ZH)**: 使用单丝架构复合材料快速制造轻质无人机框架 

**Authors**: Md Habib Ullah Khan, Kaiyue Deng, Ismail Mujtaba Khan, Kelvin Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09024)  

**Abstract**: The demand for lightweight and high-strength composite structures is rapidly growing in aerospace and robotics, particularly for optimized drone frames. However, conventional composite manufacturing methods struggle to achieve complex 3D architectures for weight savings and rely on assembling separate components, which introduce weak points at the joints. Additionally, maintaining continuous fiber reinforcement remains challenging, limiting structural efficiency. In this study, we demonstrate the lightweight Face Centered Cubic (FFC) lattice structured conceptualization of drone frames for weight reduction and complex topology fabrication through 3D Fiber Tethering (3DFiT) using continuous single tow fiber ensuring precise fiber alignment, eliminating weak points associated with traditional composite assembly. Mechanical testing demonstrates that the fabricated drone frame exhibits a high specific strength of around four to eight times the metal and thermoplastic, outperforming other conventional 3D printing methods. The drone frame weighs only 260 g, making it 10% lighter than the commercial DJI F450 frame, enhancing structural integrity and contributing to an extended flight time of three minutes, while flight testing confirms its stability and durability under operational conditions. The findings demonstrate the potential of single tow lattice truss-based drone frames, with 3DFiT serving as a scalable and efficient manufacturing method. 

**Abstract (ZH)**: 轻质高强复合结构在 aerospace 和机器人领域的需求迅速增长，特别是在优化无人机框架方面。然而，传统的复合材料制造方法难以实现复杂的三维架构以减轻重量，并且依赖于组装独立部件，这会在接头处引入弱点。此外，保持连续纤维增强仍然是一个挑战，限制了结构效率。在本研究中，我们通过3D纤维约束（3DFiT）技术，利用连续单束纤维确保精确的纤维对齐，展示了基于体心立方（FFC）格子结构概念的无人机框架，实现重量减轻和复杂拓扑结构的制造。机械测试表明，制造的无人机框架具有约四到八倍于金属和热塑性材料的高比强度，超越了其他传统的3D打印方法。该无人机框架仅重260克，比商用DJI F450框架轻10%，提高了结构完整性并延长了飞行时间至三分钟，而飞行测试验证了其在运行条件下的稳定性和耐用性。研究结果展示了基于单束纤维格子桁架无人机框架的潜力，而3DFiT作为可规模化和高效的制造方法具有重要意义。 

---
# Multi Robot Coordination in Highly Dynamic Environments: Tackling Asymmetric Obstacles and Limited Communication 

**Title (ZH)**: 在高度动态环境中的多机器人协调：应对不对称障碍和有限通信 

**Authors**: Vincenzo Suriani, Daniele Affinita, Domenico D. Bloisi, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2509.08859)  

**Abstract**: Coordinating a fully distributed multi-agent system (MAS) can be challenging when the communication channel has very limited capabilities in terms of sending rate and packet payload. When the MAS has to deal with active obstacles in a highly partially observable environment, the communication channel acquires considerable relevance. In this paper, we present an approach to deal with task assignments in extremely active scenarios, where tasks need to be frequently reallocated among the agents participating in the coordination process. Inspired by market-based task assignments, we introduce a novel distributed coordination method to orchestrate autonomous agents' actions efficiently in low communication scenarios. In particular, our algorithm takes into account asymmetric obstacles. While in the real world, the majority of obstacles are asymmetric, they are usually treated as symmetric ones, thus limiting the applicability of existing methods. To summarize, the presented architecture is designed to tackle scenarios where the obstacles are active and asymmetric, the communication channel is poor and the environment is partially observable. Our approach has been validated in simulation and in the real world, using a team of NAO robots during official RoboCup competitions. Experimental results show a notable reduction in task overlaps in limited communication settings, with a decrease of 52% in the most frequent reallocated task. 

**Abstract (ZH)**: 完全分布式多agent系统的任务协调：在通信能力受限且存在活跃非对称障碍物的高部分可观测环境中 

---
# Visual Grounding from Event Cameras 

**Title (ZH)**: 事件相机的视觉定位 

**Authors**: Lingdong Kong, Dongyue Lu, Ao Liang, Rong Li, Yuhao Dong, Tianshuai Hu, Lai Xing Ng, Wei Tsang Ooi, Benoit R. Cottereau  

**Link**: [PDF](https://arxiv.org/pdf/2509.09584)  

**Abstract**: Event cameras capture changes in brightness with microsecond precision and remain reliable under motion blur and challenging illumination, offering clear advantages for modeling highly dynamic scenes. Yet, their integration with natural language understanding has received little attention, leaving a gap in multimodal perception. To address this, we introduce Talk2Event, the first large-scale benchmark for language-driven object grounding using event data. Built on real-world driving scenarios, Talk2Event comprises 5,567 scenes, 13,458 annotated objects, and more than 30,000 carefully validated referring expressions. Each expression is enriched with four structured attributes -- appearance, status, relation to the viewer, and relation to surrounding objects -- that explicitly capture spatial, temporal, and relational cues. This attribute-centric design supports interpretable and compositional grounding, enabling analysis that moves beyond simple object recognition to contextual reasoning in dynamic environments. We envision Talk2Event as a foundation for advancing multimodal and temporally-aware perception, with applications spanning robotics, human-AI interaction, and so on. 

**Abstract (ZH)**: 事件相机以微秒级精度捕捉亮度变化，即使在运动模糊和复杂光照条件下仍保持可靠性能，为建模高度动态场景提供了明显优势。然而，它们与自然语言理解的集成方面尚未得到充分关注，这在多模态感知方面留下了一定差距。为解决这一问题，我们引入了Talk2Event，这是一个基于事件数据的语言驱动对象定位的首个大规模基准。Talk2Event涵盖了5,567个场景、13,458个标注对象以及超过30,000个仔细验证的指示表达式。每个表达式都包含四个结构化属性——外观、状态、与观者的相关性以及与周围对象的相关性——这些属性明确捕捉了空间、时间及关系线索。该属性中心设计支持可解释和组合式定位，能够超越简单的对象识别，实现动态环境中的上下文推理。我们设想Talk2Event将成为推动多模态及时间敏感感知的基础，应用于机器人技术、人机交互等领域。 

---
# Curriculum-Based Multi-Tier Semantic Exploration via Deep Reinforcement Learning 

**Title (ZH)**: 基于课程的时间递进多层语义探索通过深度强化学习 

**Authors**: Abdel Hakim Drid, Vincenzo Suriani, Daniele Nardi, Abderrezzak Debilou  

**Link**: [PDF](https://arxiv.org/pdf/2509.09356)  

**Abstract**: Navigating and understanding complex and unknown environments autonomously demands more than just basic perception and movement from embodied agents. Truly effective exploration requires agents to possess higher-level cognitive abilities, the ability to reason about their surroundings, and make more informed decisions regarding exploration strategies. However, traditional RL approaches struggle to balance efficient exploration and semantic understanding due to limited cognitive capabilities embedded in the small policies for the agents, leading often to human drivers when dealing with semantic exploration. In this paper, we address this challenge by presenting a novel Deep Reinforcement Learning (DRL) architecture that is specifically designed for resource efficient semantic exploration. A key methodological contribution is the integration of a Vision-Language Model (VLM) common-sense through a layered reward function. The VLM query is modeled as a dedicated action, allowing the agent to strategically query the VLM only when deemed necessary for gaining external guidance, thereby conserving resources. This mechanism is combined with a curriculum learning strategy designed to guide learning at different levels of complexity to ensure robust and stable learning. Our experimental evaluation results convincingly demonstrate that our agent achieves significantly enhanced object discovery rates and develops a learned capability to effectively navigate towards semantically rich regions. Furthermore, it also shows a strategic mastery of when to prompt for external environmental information. By demonstrating a practical and scalable method for embedding common-sense semantic reasoning with autonomous agents, this research provides a novel approach to pursuing a fully intelligent and self-guided exploration in robotics. 

**Abstract (ZH)**: 自主导航和理解复杂未知环境不仅需要实体代理的基本感知和运动，还需要具备更高层次的认知能力，能够对其周边环境进行推理，并作出更有前瞻性的探索策略决策。然而，传统的强化学习（RL）方法因代理中小政策嵌入的认知能力有限，难以平衡高效的探索与语义理解，常常需要人类驾驶员介入进行语义探索。本文通过提出一种新型的资源高效语义探索深度强化学习（DRL）架构来解决这一挑战，关键方法论贡献在于通过分层奖励函数整合了一种视觉-语言模型（VLM）的常识。VLM 查询被建模为专用动作，使得代理仅在必要时策略性地查询VLM，以获得外部指导，从而节省资源。该机制结合了一种Curriculum学习策略，引导在不同复杂程度下的学习，确保学习的鲁棒性和稳定性。实验结果表明，我们的代理显著提高了对象发现率，并发展出有效导航至语义丰富区域的能力，还展示了其在何时请求外部环境信息方面的战略掌握。通过展示一种实用且可扩展的方法，将常识语义推理嵌入自主代理，本研究为在机器人领域追求真正智能和自主指导的探索提供了一个新的方法。 

---
# Classification of Driver Behaviour Using External Observation Techniques for Autonomous Vehicles 

**Title (ZH)**: 基于外部观测技术的驾驶员行为分类 

**Authors**: Ian Nell, Shane Gilroy  

**Link**: [PDF](https://arxiv.org/pdf/2509.09349)  

**Abstract**: Road traffic accidents remain a significant global concern, with human error, particularly distracted and impaired driving, among the leading causes. This study introduces a novel driver behavior classification system that uses external observation techniques to detect indicators of distraction and impairment. The proposed framework employs advanced computer vision methodologies, including real-time object tracking, lateral displacement analysis, and lane position monitoring. The system identifies unsafe driving behaviors such as excessive lateral movement and erratic trajectory patterns by implementing the YOLO object detection model and custom lane estimation algorithms. Unlike systems reliant on inter-vehicular communication, this vision-based approach enables behavioral analysis of non-connected vehicles. Experimental evaluations on diverse video datasets demonstrate the framework's reliability and adaptability across varying road and environmental conditions. 

**Abstract (ZH)**: 基于外部观察的新型驾驶员行为分类系统：检测分心和受酒精影响的指标 

---
# Model-Agnostic Open-Set Air-to-Air Visual Object Detection for Reliable UAV Perception 

**Title (ZH)**: 模型无关的开放集空对空视觉目标检测以实现可靠无人机感知 

**Authors**: Spyridon Loukovitis, Anastasios Arsenos, Vasileios Karampinis, Athanasios Voulodimos  

**Link**: [PDF](https://arxiv.org/pdf/2509.09297)  

**Abstract**: Open-set detection is crucial for robust UAV autonomy in air-to-air object detection under real-world conditions. Traditional closed-set detectors degrade significantly under domain shifts and flight data corruption, posing risks to safety-critical applications. We propose a novel, model-agnostic open-set detection framework designed specifically for embedding-based detectors. The method explicitly handles unknown object rejection while maintaining robustness against corrupted flight data. It estimates semantic uncertainty via entropy modeling in the embedding space and incorporates spectral normalization and temperature scaling to enhance open-set discrimination. We validate our approach on the challenging AOT aerial benchmark and through extensive real-world flight tests. Comprehensive ablation studies demonstrate consistent improvements over baseline methods, achieving up to a 10\% relative AUROC gain compared to standard YOLO-based detectors. Additionally, we show that background rejection further strengthens robustness without compromising detection accuracy, making our solution particularly well-suited for reliable UAV perception in dynamic air-to-air environments. 

**Abstract (ZH)**: 开放集检测是真实环境中空中目标检测实现无人机 robust 自主性的关键。传统的封闭集检测器在领域偏移和飞行数据污染下表现显著下降，这对安全关键应用构成了风险。我们提出了一种适用于嵌入式检测器的新型、模型无关的开放集检测框架。该方法明确处理未知目标的拒绝，同时保持对污染飞行数据的鲁棒性。通过嵌入空间中的熵建模估计语义不确定性，并结合频谱规范化和温度缩放以增强开放集辨别能力。我们在具有挑战性的 AOT 航空基准测试和广泛的实地飞行试验中验证了该方法。全面的消融研究证明，相对于基准方法实现了持续改进，与标准 YOLO 基础检测器相比，相对 AUROC 提升高达 10%。此外，我们展示了背景拒绝进一步增强了鲁棒性而不牺牲检测准确性，使我们的解决方案特别适合动态的空战环境中的可靠无人机感知。 

---
# Global Optimization of Stochastic Black-Box Functions with Arbitrary Noise Distributions using Wilson Score Kernel Density Estimation 

**Title (ZH)**: 使用威尔逊分数核密度估计全局优化任意噪声分布的随机黑盒函数 

**Authors**: Thorbjørn Mosekjær Iversen, Lars Carøe Sørensen, Simon Faarvang Mathiesen, Henrik Gordon Petersen  

**Link**: [PDF](https://arxiv.org/pdf/2509.09238)  

**Abstract**: Many optimization problems in robotics involve the optimization of time-expensive black-box functions, such as those involving complex simulations or evaluation of real-world experiments. Furthermore, these functions are often stochastic as repeated experiments are subject to unmeasurable disturbances. Bayesian optimization can be used to optimize such methods in an efficient manner by deploying a probabilistic function estimator to estimate with a given confidence so that regions of the search space can be pruned away. Consequently, the success of the Bayesian optimization depends on the function estimator's ability to provide informative confidence bounds. Existing function estimators require many function evaluations to infer the underlying confidence or depend on modeling of the disturbances. In this paper, it is shown that the confidence bounds provided by the Wilson Score Kernel Density Estimator (WS-KDE) are applicable as excellent bounds to any stochastic function with an output confined to the closed interval [0;1] regardless of the distribution of the output. This finding opens up the use of WS-KDE for stable global optimization on a wider range of cost functions. The properties of WS-KDE in the context of Bayesian optimization are demonstrated in simulation and applied to the problem of automated trap design for vibrational part feeders. 

**Abstract (ZH)**: 机器人领域中涉及的时间昂贵的黑盒函数优化问题及其Bayesian优化方法的研究：Wilson Score内核密度估计在[0;1]区间输出的随机函数中的应用 

---
# ProgD: Progressive Multi-scale Decoding with Dynamic Graphs for Joint Multi-agent Motion Forecasting 

**Title (ZH)**: ProgD：基于动态图的分阶段多尺度解码联合多 Agents 动态预测 

**Authors**: Xing Gao, Zherui Huang, Weiyao Lin, Xiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.09210)  

**Abstract**: Accurate motion prediction of surrounding agents is crucial for the safe planning of autonomous vehicles. Recent advancements have extended prediction techniques from individual agents to joint predictions of multiple interacting agents, with various strategies to address complex interactions within future motions of agents. However, these methods overlook the evolving nature of these interactions. To address this limitation, we propose a novel progressive multi-scale decoding strategy, termed ProgD, with the help of dynamic heterogeneous graph-based scenario modeling. In particular, to explicitly and comprehensively capture the evolving social interactions in future scenarios, given their inherent uncertainty, we design a progressive modeling of scenarios with dynamic heterogeneous graphs. With the unfolding of such dynamic heterogeneous graphs, a factorized architecture is designed to process the spatio-temporal dependencies within future scenarios and progressively eliminate uncertainty in future motions of multiple agents. Furthermore, a multi-scale decoding procedure is incorporated to improve on the future scenario modeling and consistent prediction of agents' future motion. The proposed ProgD achieves state-of-the-art performance on the INTERACTION multi-agent prediction benchmark, ranking $1^{st}$, and the Argoverse 2 multi-world forecasting benchmark. 

**Abstract (ZH)**: 准确预测周围代理的运动对于自主车辆的安全规划至关重要。近期进展将预测技术从单个代理扩展到了多个相互作用代理的联合预测，并采用各种策略来应对未来的复杂交互。然而，这些方法忽视了这些交互的进化性质。为解决这一局限，我们提出了一种名为ProgD的新型渐进多尺度解码策略，并借助动态异构图基场景建模。特别地，为了明确定性和全面地捕捉未来场景中逐渐演化的社会交互（考虑到他们固有的不确定性），我们设计了一种动态异构图的渐进建模方法。随着动态异构图的展开，我们设计了一种因子化解构来处理未来场景中的时空依赖性，并逐步消除多代理未来运动中的不确定性。此外，我们整合了一种多尺度解码过程，以提高未来场景建模的准确性并实现一致的代理未来运动预测。所提出的ProgD在INTERACTION多代理预测基准测试和Argoverse 2多世界预测基准测试中实现了最优性能，排名第一。 

---
