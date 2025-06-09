# PyGemini: Unified Software Development towards Maritime Autonomy Systems 

**Title (ZH)**: PyGemini: 一体化软件开发面向海洋自主系统 

**Authors**: Kjetil Vasstein, Christian Le, Simon Lervåg Breivik, Trygve Maukon Myhr, Annette Stahl, Edmund Førland Brekke  

**Link**: [PDF](https://arxiv.org/pdf/2506.06262)  

**Abstract**: Ensuring the safety and certifiability of autonomous surface vessels (ASVs) requires robust decision-making systems, supported by extensive simulation, testing, and validation across a broad range of scenarios. However, the current landscape of maritime autonomy development is fragmented -- relying on disparate tools for communication, simulation, monitoring, and system integration -- which hampers interdisciplinary collaboration and inhibits the creation of compelling assurance cases, demanded by insurers and regulatory bodies. Furthermore, these disjointed tools often suffer from performance bottlenecks, vendor lock-in, and limited support for continuous integration workflows. To address these challenges, we introduce PyGemini, a permissively licensed, Python-native framework that builds on the legacy of Autoferry Gemini to unify maritime autonomy development. PyGemini introduces a novel Configuration-Driven Development (CDD) process that fuses Behavior-Driven Development (BDD), data-oriented design, and containerization to support modular, maintainable, and scalable software architectures. The framework functions as a stand-alone application, cloud-based service, or embedded library -- ensuring flexibility across research and operational contexts. We demonstrate its versatility through a suite of maritime tools -- including 3D content generation for simulation and monitoring, scenario generation for autonomy validation and training, and generative artificial intelligence pipelines for augmenting imagery -- thereby offering a scalable, maintainable, and performance-oriented foundation for future maritime robotics and autonomy research. 

**Abstract (ZH)**: 确保自主水面船舶（ASVs）的安全性和可验证性需要 robust 的决策系统，这些系统依托于广泛场景下的仿真、测试和验证。然而，当前海上自主性开发领域存在碎片化现象——依赖于不同的工具进行通信、仿真、监控和系统集成，这阻碍了跨学科的合作，并限制了保险公司和监管机构要求的令人信服的保障案例的创建。此外，这些分离的工具往往存在性能瓶颈、供应商锁定以及对持续集成工作流程支持有限的问题。为解决这些问题，我们介绍了 PyGemini，这是一个基于宽松许可、原生的 Python 框架，继承了 Autoferry Gemini 的传统，旨在统一海上自主性开发。PyGemini 引入了一种新颖的配置驱动开发（CDD）过程，融合了行为驱动开发（BDD）、数据导向设计和容器化技术，以支持模块化、可维护和可扩展的软件架构。该框架可以作为独立应用、云服务或嵌入式库运行，确保其在研究和运营情境下的灵活性。我们通过一系列海上工具（包括用于仿真的 3D 内容生成、用于自主性验证和培训的场景生成，以及用于增强图像的生成型人工智能流水线）展示了其多功能性，从而为未来的海洋机器人和自主性研究提供了可扩展、可维护且性能优化的基础。 

---
# From NLVO to NAO: Reactive Robot Navigation using Velocity and Acceleration Obstacles 

**Title (ZH)**: 从NLVO到NAO：基于速度和加速度障碍物的反应式机器人导航 

**Authors**: Asher Stern, Zvi Shiller  

**Link**: [PDF](https://arxiv.org/pdf/2506.06255)  

**Abstract**: This paper introduces a novel approach for robot navigation in challenging dynamic environments. The proposed method builds upon the concept of Velocity Obstacles (VO) that was later extended to Nonlinear Velocity Obstacles (NLVO) to account for obstacles moving along nonlinear trajectories. The NLVO is extended in this paper to Acceleration Obstacles (AO) and Nonlinear Acceleration Obstacles (NAO) that account for velocity and acceleration constraints. Multi-robot navigation is achieved by using the same avoidance algorithm by all robots. At each time step, the trajectories of all robots are predicted based on their current velocity and acceleration to allow the computation of their respective NLVO, AO and NAO.
The introduction of AO and NAO allows the generation of safe avoidance maneuvers that account for the robot dynamic constraints better than could be done with the NLVO alone. This paper demonstrates the use of AO and NAO for robot navigation in challenging environments. It is shown that using AO and NAO enables simultaneous real-time collision avoidance while accounting for robot kinematics and a direct consideration of its dynamic constraints. The presented approach enables reactive and efficient navigation, with potential application for autonomous vehicles operating in complex dynamic environments. 

**Abstract (ZH)**: 一种用于应对挑战性动态环境的机器人导航新方法：加速障碍与非线性加速障碍在多机器人导航中的应用 

---
# BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly 

**Title (ZH)**: BiAssemble: 学习双手几何装配的合作功能 

**Authors**: Yan Shen, Ruihai Wu, Yubin Ke, Xinyuan Song, Zeyi Li, Xiaoqi Li, Hongwei Fan, Haoran Lu, Hao dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.06221)  

**Abstract**: Shape assembly, the process of combining parts into a complete whole, is a crucial robotic skill with broad real-world applications. Among various assembly tasks, geometric assembly--where broken parts are reassembled into their original form (e.g., reconstructing a shattered bowl)--is particularly challenging. This requires the robot to recognize geometric cues for grasping, assembly, and subsequent bimanual collaborative manipulation on varied fragments. In this paper, we exploit the geometric generalization of point-level affordance, learning affordance aware of bimanual collaboration in geometric assembly with long-horizon action sequences. To address the evaluation ambiguity caused by geometry diversity of broken parts, we introduce a real-world benchmark featuring geometric variety and global reproducibility. Extensive experiments demonstrate the superiority of our approach over both previous affordance-based and imitation-based methods. Project page: this https URL. 

**Abstract (ZH)**: 几何组装：一种将破碎部件重新组合成原形的机器人技能，在广泛的实际应用中至关重要。在各类组装任务中，几何组装尤其具有挑战性——需要机器人识别几何线索以进行抓取、组装及后续的双臂协作操作。本文利用点级 affordance 的几何泛化，在长时序行动序列中学习具有双臂协作意识的 affordance，以应对碎片几何多样性带来的评估模糊性问题。我们引入了一个包含几何多样性和全局可复现性的实际环境基准。大量实验表明，本文方法在与基于 affordance 和基于模仿的方法的对比中均表现出优越性。项目页面：这个 https://url.cn/3Jn8XhK。 

---
# Astra: Toward General-Purpose Mobile Robots via Hierarchical Multimodal Learning 

**Title (ZH)**: Astra：通过层次多模态学习迈向通用型移动机器人 

**Authors**: Sheng Chen, Peiyu He, Jiaxin Hu, Ziyang Liu, Yansheng Wang, Tao Xu, Chi Zhang, Chongchong Zhang, Chao An, Shiyu Cai, Duo Cao, Kangping Chen, Shuai Chu, Tianwei Chu, Mingdi Dan, Min Du, Weiwei Fang, Pengyou Fu, Junkai Hu, Xiaowei Jiang, Zhaodi Jiang, Fuxuan Li, Jun Li, Minghui Li, Mingyao Li, Yanchang Li, Zhibin Li, Guangming Liu, Kairui Liu, Lihao Liu, Weizhi Liu, Xiaoshun Liu, Yufei Liu, Yunfei Liu, Qiang Lu, Yuanfei Luo, Xiang Lv, Hongying Ma, Sai Ma, Lingxian Mi, Sha Sa, Hongxiang Shu, Lei Tian, Chengzhi Wang, Jiayu Wang, Kaijie Wang, Qingyi Wang, Renwen Wang, Tao Wang, Wei Wang, Xirui Wang, Chao Wei, Xuguang Wei, Zijun Xia, Zhaohao Xiao, Tingshuai Yan, Liyan Yang, Yifan Yang, Zhikai Yang, Zhong Yin, Li Yuan, Liuchun Yuan, Chi Zhang, Jinyang Zhang, Junhui Zhang, Linge Zhang, Zhenyi Zhang, Zheyu Zhang, Dongjie Zhu, Hang Li, Yangang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06205)  

**Abstract**: Modern robot navigation systems encounter difficulties in diverse and complex indoor environments. Traditional approaches rely on multiple modules with small models or rule-based systems and thus lack adaptability to new environments. To address this, we developed Astra, a comprehensive dual-model architecture, Astra-Global and Astra-Local, for mobile robot navigation. Astra-Global, a multimodal LLM, processes vision and language inputs to perform self and goal localization using a hybrid topological-semantic graph as the global map, and outperforms traditional visual place recognition methods. Astra-Local, a multitask network, handles local path planning and odometry estimation. Its 4D spatial-temporal encoder, trained through self-supervised learning, generates robust 4D features for downstream tasks. The planning head utilizes flow matching and a novel masked ESDF loss to minimize collision risks for generating local trajectories, and the odometry head integrates multi-sensor inputs via a transformer encoder to predict the relative pose of the robot. Deployed on real in-house mobile robots, Astra achieves high end-to-end mission success rate across diverse indoor environments. 

**Abstract (ZH)**: 现代机器人导航系统在多样且复杂的室内环境中遇到困难。传统的方法依赖多个小型模型或基于规则的系统模块，因此缺乏对新环境的适应性。为解决这一问题，我们开发了Astra，一个全面的双模型架构，包括Astra-Global和Astra-Local，用于移动机器人导航。Astra-Global是一个多模态大型语言模型，通过混合拓扑语义图作为全局地图来处理视觉和语言输入，实现自我和目标定位，并优于传统的视觉地方识别方法。Astra-Local是一个多任务网络，处理局部路径规划和里程计估计。其4D空间-时间编码器通过自监督学习训练，生成稳健的4D特征以用于下游任务。规划头部利用流匹配和新颖的掩码ESDF损失来最小化碰撞风险以生成局部轨迹，而里程计头部通过变压器编码器整合多传感器输入以预测机器人相对姿态。在实时内部移动机器人上部署，Astra在多种室内环境中实现了高端到端任务成功率。 

---
# 3DFlowAction: Learning Cross-Embodiment Manipulation from 3D Flow World Model 

**Title (ZH)**: 3DFlowAction：从3D流世界模型学习跨身体操纵技能 

**Authors**: Hongyan Zhi, Peihao Chen, Siyuan Zhou, Yubo Dong, Quanxi Wu, Lei Han, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06199)  

**Abstract**: Manipulation has long been a challenging task for robots, while humans can effortlessly perform complex interactions with objects, such as hanging a cup on the mug rack. A key reason is the lack of a large and uniform dataset for teaching robots manipulation skills. Current robot datasets often record robot action in different action spaces within a simple scene. This hinders the robot to learn a unified and robust action representation for different robots within diverse scenes. Observing how humans understand a manipulation task, we find that understanding how the objects should move in the 3D space is a critical clue for guiding actions. This clue is embodiment-agnostic and suitable for both humans and different robots. Motivated by this, we aim to learn a 3D flow world model from both human and robot manipulation data. This model predicts the future movement of the interacting objects in 3D space, guiding action planning for manipulation. Specifically, we synthesize a large-scale 3D optical flow dataset, named ManiFlow-110k, through a moving object auto-detect pipeline. A video diffusion-based world model then learns manipulation physics from these data, generating 3D optical flow trajectories conditioned on language instructions. With the generated 3D object optical flow, we propose a flow-guided rendering mechanism, which renders the predicted final state and leverages GPT-4o to assess whether the predicted flow aligns with the task description. This equips the robot with a closed-loop planning ability. Finally, we consider the predicted 3D optical flow as constraints for an optimization policy to determine a chunk of robot actions for manipulation. Extensive experiments demonstrate strong generalization across diverse robotic manipulation tasks and reliable cross-embodiment adaptation without hardware-specific training. 

**Abstract (ZH)**: 基于人类和机器人数据的学习三维流动世界模型：引导操作规划与通用化 

---
# Bridging Perception and Action: Spatially-Grounded Mid-Level Representations for Robot Generalization 

**Title (ZH)**: 感知与行动的桥梁：基于空间的中间表示形式促进机器人泛化 

**Authors**: Jonathan Yang, Chuyuan Kelly Fu, Dhruv Shah, Dorsa Sadigh, Fei Xia, Tingnan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06196)  

**Abstract**: In this work, we investigate how spatially grounded auxiliary representations can provide both broad, high-level grounding as well as direct, actionable information to improve policy learning performance and generalization for dexterous tasks. We study these mid-level representations across three critical dimensions: object-centricity, pose-awareness, and depth-awareness. We use these interpretable mid-level representations to train specialist encoders via supervised learning, then feed them as inputs to a diffusion policy to solve dexterous bimanual manipulation tasks in the real world. We propose a novel mixture-of-experts policy architecture that combines multiple specialized expert models, each trained on a distinct mid-level representation, to improve policy generalization. This method achieves an average success rate that is 11% higher than a language-grounded baseline and 24 percent higher than a standard diffusion policy baseline on our evaluation tasks. Furthermore, we find that leveraging mid-level representations as supervision signals for policy actions within a weighted imitation learning algorithm improves the precision with which the policy follows these representations, yielding an additional performance increase of 10%. Our findings highlight the importance of grounding robot policies not only with broad perceptual tasks but also with more granular, actionable representations. For further information and videos, please visit this https URL. 

**Abstract (ZH)**: 本工作中，我们探讨了基于空间的辅助表示如何提供广泛的高层次 grounding 以及直接的可操作信息，以提高灵巧任务的策略学习性能和泛化能力。我们从三个关键维度研究这些中层表示：对象中心性、姿态感知和深度感知。我们使用这些可解释的中层表示，通过监督学习训练专业编码器，然后将它们作为输入馈入扩散策略，以解决真实世界中的双臂灵巧操作任务。我们提出了一种新颖的专家混合策略架构，该架构结合了多个训练在不同中层表示上的专业专家模型，以提高策略的泛化能力。该方法在我们的评估任务中，平均成功率比基于语言的基线高11%，比标准的扩散策略基线高24%。此外，我们发现，在加权模仿学习算法中利用中层表示作为策略行动的监督信号，可以提高策略跟随这些表示的精确度，从而进一步提高10%的性能。我们的研究结果强调，不仅需要通过广泛的感知任务来地化机器人策略，还需要通过更细粒度的、可操作的表示来地化策略。欲了解更多信息和视频，请访问此链接：https://xxxxxx。 

---
# UAV-UGV Cooperative Trajectory Optimization and Task Allocation for Medical Rescue Tasks in Post-Disaster Environments 

**Title (ZH)**: 灾害后环境中无人机-地面机器人协同轨迹优化与任务分配 

**Authors**: Kaiyuan Chen, Wanpeng Zhao, Yongxi Liu, Yuanqing Xia, Wannian Liang, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06136)  

**Abstract**: In post-disaster scenarios, rapid and efficient delivery of medical resources is critical and challenging due to severe damage to infrastructure. To provide an optimized solution, we propose a cooperative trajectory optimization and task allocation framework leveraging unmanned aerial vehicles (UAVs) and unmanned ground vehicles (UGVs). This study integrates a Genetic Algorithm (GA) for efficient task allocation among multiple UAVs and UGVs, and employs an informed-RRT* (Rapidly-exploring Random Tree Star) algorithm for collision-free trajectory generation. Further optimization of task sequencing and path efficiency is conducted using Covariance Matrix Adaptation Evolution Strategy (CMA-ES). Simulation experiments conducted in a realistic post-disaster environment demonstrate that our proposed approach significantly improves the overall efficiency of medical rescue operations compared to traditional strategies, showing substantial reductions in total mission completion time and traveled distance. Additionally, the cooperative utilization of UAVs and UGVs effectively balances their complementary advantages, highlighting the system' s scalability and practicality for real-world deployment. 

**Abstract (ZH)**: 在灾后场景中，利用无人机和地面无人车进行高效协同路径优化与任务分配，以优化医疗资源快速精准配送的研究 

---
# On-board Mission Replanning for Adaptive Cooperative Multi-Robot Systems 

**Title (ZH)**: 在分支任务规划适应性合作多机器人系统中的车载任务重规划 

**Authors**: Elim Kwan, Rehman Qureshi, Liam Fletcher, Colin Laganier, Victoria Nockles, Richard Walters  

**Link**: [PDF](https://arxiv.org/pdf/2506.06094)  

**Abstract**: Cooperative autonomous robotic systems have significant potential for executing complex multi-task missions across space, air, ground, and maritime domains. But they commonly operate in remote, dynamic and hazardous environments, requiring rapid in-mission adaptation without reliance on fragile or slow communication links to centralised compute. Fast, on-board replanning algorithms are therefore needed to enhance resilience. Reinforcement Learning shows strong promise for efficiently solving mission planning tasks when formulated as Travelling Salesperson Problems (TSPs), but existing methods: 1) are unsuitable for replanning, where agents do not start at a single location; 2) do not allow cooperation between agents; 3) are unable to model tasks with variable durations; or 4) lack practical considerations for on-board deployment. Here we define the Cooperative Mission Replanning Problem as a novel variant of multiple TSP with adaptations to overcome these issues, and develop a new encoder/decoder-based model using Graph Attention Networks and Attention Models to solve it effectively and efficiently. Using a simple example of cooperative drones, we show our replanner consistently (90% of the time) maintains performance within 10% of the state-of-the-art LKH3 heuristic solver, whilst running 85-370 times faster on a Raspberry Pi. This work paves the way for increased resilience in autonomous multi-agent systems. 

**Abstract (ZH)**: 合作自主机器人系统在空间、空气、地面和 maritime 领域执行复杂多任务使命具有巨大的潜力，但它们通常在远程、动态和危险的环境中操作，需要快速进行在任务中的适应而不依赖于脆弱或慢速的通信链路与集中计算。因此需要快速的机载重规划算法以增强系统的韧性。强化学习在将使命规划任务表述为旅行商问题（TSP）时显示出强大的潜力，但现有方法：1) 不适用于重规划，因代理不从单一位置开始；2) 不允许代理之间的合作；3) 无法建模具有变化持续时间的任务；或 4) 缺乏针对机载部署的实际考虑。在这里，我们将合作使命重规划问题定义为多TSP的新型变体，并通过Graph Attention Networks和注意力模型开发了一种新的编码器/解码器模型，以有效地解决该问题。以合作无人机的简单示例为例，我们展示了我们的重规划器在90%的时间内保持与最先进的LKH3启发式求解器相当的性能，同时在Raspberry Pi上运行速度快85-370倍。这项工作为自主多代理系统增加了韧性奠定了基础。 

---
# Self driving algorithm for an active four wheel drive racecar 

**Title (ZH)**: 主动四轮驱动赛车的自动驾驶算法 

**Authors**: Gergely Bari, Laszlo Palkovics  

**Link**: [PDF](https://arxiv.org/pdf/2506.06077)  

**Abstract**: Controlling autonomous vehicles at their handling limits is a significant challenge, particularly for electric vehicles with active four wheel drive (A4WD) systems offering independent wheel torque control. While traditional Vehicle Dynamics Control (VDC) methods use complex physics-based models, this study explores Deep Reinforcement Learning (DRL) to develop a unified, high-performance controller. We employ the Proximal Policy Optimization (PPO) algorithm to train an agent for optimal lap times in a simulated racecar (TORCS) at the tire grip limit. Critically, the agent learns an end-to-end policy that directly maps vehicle states, like velocities, accelerations, and yaw rate, to a steering angle command and independent torque commands for each of the four wheels. This formulation bypasses conventional pedal inputs and explicit torque vectoring algorithms, allowing the agent to implicitly learn the A4WD control logic needed for maximizing performance and stability. Simulation results demonstrate the RL agent learns sophisticated strategies, dynamically optimizing wheel torque distribution corner-by-corner to enhance handling and mitigate the vehicle's inherent understeer. The learned behaviors mimic and, in aspects of grip utilization, potentially surpass traditional physics-based A4WD controllers while achieving competitive lap times. This research underscores DRL's potential to create adaptive control systems for complex vehicle dynamics, suggesting RL is a potent alternative for advancing autonomous driving in demanding, grip-limited scenarios for racing and road safety. 

**Abstract (ZH)**: 基于深度强化学习的四轮独立驱动自主车辆handling极限控制研究 

---
# BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning 

**Title (ZH)**: BEAST: 效率较高的B-样条编码行动序列的分词方法在模仿学习中的应用 

**Authors**: Hongyi Zhou, Weiran Liao, Xi Huang, Yucheng Tang, Fabian Otto, Xiaogang Jia, Xinkai Jiang, Simon Hilber, Ge Li, Qian Wang, Ömer Erdinç Yağmurlu, Nils Blank, Moritz Reuss, Rudolf Lioutikov  

**Link**: [PDF](https://arxiv.org/pdf/2506.06072)  

**Abstract**: We present the B-spline Encoded Action Sequence Tokenizer (BEAST), a novel action tokenizer that encodes action sequences into compact discrete or continuous tokens using B-splines. In contrast to existing action tokenizers based on vector quantization or byte pair encoding, BEAST requires no separate tokenizer training and consistently produces tokens of uniform length, enabling fast action sequence generation via parallel decoding. Leveraging our B-spline formulation, BEAST inherently ensures generating smooth trajectories without discontinuities between adjacent segments. We extensively evaluate BEAST by integrating it with three distinct model architectures: a Variational Autoencoder (VAE) with continuous tokens, a decoder-only Transformer with discrete tokens, and Florence-2, a pretrained Vision-Language Model with an encoder-decoder architecture, demonstrating BEAST's compatibility and scalability with large pretrained models. We evaluate BEAST across three established benchmarks consisting of 166 simulated tasks and on three distinct robot settings with a total of 8 real-world tasks. Experimental results demonstrate that BEAST (i) significantly reduces both training and inference computational costs, and (ii) consistently generates smooth, high-frequency control signals suitable for continuous control tasks while (iii) reliably achieves competitive task success rates compared to state-of-the-art methods. 

**Abstract (ZH)**: B-spline编码动作序列分词器（BEAST）：一种新颖的动作分词方法 

---
# End-to-End Framework for Robot Lawnmower Coverage Path Planning using Cellular Decomposition 

**Title (ZH)**: 基于细胞分解的无人驾驶割草机器人全场路径规划端到端框架 

**Authors**: Nikunj Shah, Utsav Dey, Kenji Nishimiya  

**Link**: [PDF](https://arxiv.org/pdf/2506.06028)  

**Abstract**: Efficient Coverage Path Planning (CPP) is necessary for autonomous robotic lawnmowers to effectively navigate and maintain lawns with diverse and irregular shapes. This paper introduces a comprehensive end-to-end pipeline for CPP, designed to convert user-defined boundaries on an aerial map into optimized coverage paths seamlessly. The pipeline includes user input extraction, coordinate transformation, area decomposition and path generation using our novel AdaptiveDecompositionCPP algorithm, preview and customization through an interactive coverage path visualizer, and conversion to actionable GPS waypoints. The AdaptiveDecompositionCPP algorithm combines cellular decomposition with an adaptive merging strategy to reduce non-mowing travel thereby enhancing operational efficiency. Experimental evaluations, encompassing both simulations and real-world lawnmower tests, demonstrate the effectiveness of the framework in coverage completeness and mowing efficiency. 

**Abstract (ZH)**: 高效的覆盖路径规划（CPP）对于自主割草机器人有效导航和维护具有多样化和不规则形状的草坪是必要的。本文介绍了用于CPP的全面端到端管道，旨在将用户定义的边界无缝转换为优化的覆盖路径。该管道包括用户输入提取、坐标变换、区域分解和路径生成（使用我们提出的AdaptiveDecompositionCPP算法）、通过交互式的覆盖路径可视化器进行预览和定制，以及转换为可执行的GPS航点。AdaptiveDecompositionCPP算法结合了细胞分解与自适应合并策略，以减少非割草行驶，从而提升操作效率。实验评估，包括模拟和实际割草机测试，证明了该框架在覆盖完整性和割草效率方面的有效性。 

---
# Enhanced Trust Region Sequential Convex Optimization for Multi-Drone Thermal Screening Trajectory Planning in Urban Environments 

**Title (ZH)**: 增强信任域序列凸优化在城市环境中的多无人机热筛查轨迹规划 

**Authors**: Kaiyuan Chen, Zhengjie Hu, Shaolin Zhang, Yuanqing Xia, Wannian Liang, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06012)  

**Abstract**: The rapid detection of abnormal body temperatures in urban populations is essential for managing public health risks, especially during outbreaks of infectious diseases. Multi-drone thermal screening systems offer promising solutions for fast, large-scale, and non-intrusive human temperature monitoring. However, trajectory planning for multiple drones in complex urban environments poses significant challenges, including collision avoidance, coverage efficiency, and constrained flight environments. In this study, we propose an enhanced trust region sequential convex optimization (TR-SCO) algorithm for optimal trajectory planning of multiple drones performing thermal screening tasks. Our improved algorithm integrates a refined convex optimization formulation within a trust region framework, effectively balancing trajectory smoothness, obstacle avoidance, altitude constraints, and maximum screening coverage. Simulation results demonstrate that our approach significantly improves trajectory optimality and computational efficiency compared to conventional convex optimization methods. This research provides critical insights and practical contributions toward deploying efficient multi-drone systems for real-time thermal screening in urban areas. For reader who are interested in our research, we release our source code at this https URL. 

**Abstract (ZH)**: 城市人群中快速检测异常体温对于管理公共卫生风险至关重要，尤其是在传染病暴发期间。多无人机热筛查系统为快速、大规模和非侵入性的人体温度监控提供了有前景的解决方案。然而，在复杂的城市环境中进行多无人机轨迹规划面临着巨大的挑战，包括避碰、覆盖率和飞行限制。在本研究中，我们提出了一种增强的信任区域序列凸优化（TR-SCO）算法，用于多无人机热筛查任务的最优轨迹规划。我们改进的算法结合了细化的凸优化公式于信任区域框架内，有效地平衡了轨迹平滑性、障碍物回避、高度限制和最大筛查覆盖率。仿真结果表明，与传统的凸优化方法相比，我们的方法显著提高了轨迹优化效果和计算效率。本研究为在城市区域能够进行实时热筛查的高效多无人机系统部署提供了关键见解和实用贡献。对我们的研究感兴趣的读者可以通过此链接获取我们开源的代码：此链接。 

---
# Improving Long-Range Navigation with Spatially-Enhanced Recurrent Memory via End-to-End Reinforcement Learning 

**Title (ZH)**: 基于端到端强化学习的时空增强递归记忆在长距离导航中的改进 

**Authors**: Fan Yang, Per Frivik, David Hoeller, Chen Wang, Cesar Cadena, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2506.05997)  

**Abstract**: Recent advancements in robot navigation, especially with end-to-end learning approaches like reinforcement learning (RL), have shown remarkable efficiency and effectiveness. Yet, successful navigation still relies on two key capabilities: mapping and planning, whether explicit or implicit. Classical approaches use explicit mapping pipelines to register ego-centric observations into a coherent map frame for the planner. In contrast, end-to-end learning achieves this implicitly, often through recurrent neural networks (RNNs) that fuse current and past observations into a latent space for planning. While architectures such as LSTM and GRU capture temporal dependencies, our findings reveal a key limitation: their inability to perform effective spatial memorization. This skill is essential for transforming and integrating sequential observations from varying perspectives to build spatial representations that support downstream planning. To address this, we propose Spatially-Enhanced Recurrent Units (SRUs), a simple yet effective modification to existing RNNs, designed to enhance spatial memorization capabilities. We introduce an attention-based architecture with SRUs, enabling long-range navigation using a single forward-facing stereo camera. Regularization techniques are employed to ensure robust end-to-end recurrent training via RL. Experimental results show our approach improves long-range navigation by 23.5% compared to existing RNNs. Furthermore, with SRU memory, our method outperforms the RL baseline with explicit mapping and memory modules, achieving a 29.6% improvement in diverse environments requiring long-horizon mapping and memorization. Finally, we address the sim-to-real gap by leveraging large-scale pretraining on synthetic depth data, enabling zero-shot transfer to diverse and complex real-world environments. 

**Abstract (ZH)**: 近期机器人导航的进展，尤其是端到端学习方法（如强化学习RL），显示出了显著的效率和效果。然而，成功的导航仍依赖于两个关键能力：建图和规划，无论是显式的还是隐式的。经典方法使用显式的建图管道，将第一人称观测注册为规划器所需的连贯的地图框架。相比之下，端到端学习隐式地实现了这一点，通常通过融合当前和过去观测的递归神经网络（RNN），在潜在空间中进行规划。虽然LSTM和GRU等架构捕捉了时间依赖性，但我们的研究揭示了一个关键限制：它们无法有效地进行空间记忆。这种技能对于将不同视角下的序列观测转换和整合以构建支持下游规划的空间表示是必不可少的。为了应对这一挑战，我们提出了一种增强空间记忆能力的简单而有效的递归单元改进方案——空间增强递归单元（SRUs）。我们介绍了一种基于注意力机制的SRU架构，使得仅用一个前方双目摄像头实现远程导航成为可能。通过正则化技术，我们确保了通过RL实现端到端递归训练的鲁棒性。实验结果显示，与现有RNN相比，我们的方法在远程导航性能上提高了23.5%。此外，借助SRU记忆，我们的方法在需要长期映射和记忆的多样化环境中优于带有显式建图和记忆模块的RL基线，取得了29.6%的性能提升。最后，我们通过大规模预训练合成深度数据，解决了仿真到现实世界的差距，使方法能够零样本传输到多样化和复杂的实际环境。 

---
# Object Navigation with Structure-Semantic Reasoning-Based Multi-level Map and Multimodal Decision-Making LLM 

**Title (ZH)**: 基于结构语义推理的多层级地图与多模态决策导航 

**Authors**: Chongshang Yan, Jiaxuan He, Delun Li, Yi Yang, Wenjie Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.05896)  

**Abstract**: The zero-shot object navigation (ZSON) in unknown open-ended environments coupled with semantically novel target often suffers from the significant decline in performance due to the neglect of high-dimensional implicit scene information and the long-range target searching task. To address this, we proposed an active object navigation framework with Environmental Attributes Map (EAM) and MLLM Hierarchical Reasoning module (MHR) to improve its success rate and efficiency. EAM is constructed by reasoning observed environments with SBERT and predicting unobserved ones with Diffusion, utilizing human space regularities that underlie object-room correlations and area adjacencies. MHR is inspired by EAM to perform frontier exploration decision-making, avoiding the circuitous trajectories in long-range scenarios to improve path efficiency. Experimental results demonstrate that the EAM module achieves 64.5\% scene mapping accuracy on MP3D dataset, while the navigation task attains SPLs of 28.4\% and 26.3\% on HM3D and MP3D benchmarks respectively - representing absolute improvements of 21.4\% and 46.0\% over baseline methods. 

**Abstract (ZH)**: 未知开放环境中的零样本物体导航（ZSON）常常由于忽略高维隐式场景信息和长距离目标搜索任务的影响而性能显著下降。为解决这一问题，我们提出了一种结合环境属性图（EAM）和MLLM层次推理模块（MHR）的主动物体导航框架，以提高其成功率和效率。实验结果表明，EAM模块在MP3D数据集上实现了64.5%的场景映射准确率，而导航任务分别在HM3D和MP3D基准上取得了28.4%和26.3%的SPL值，相对于基线方法分别绝对提高了21.4%和46.0%。 

---
# Optimal Robotic Velcro Peeling with Force Feedback 

**Title (ZH)**: 带有力反馈的最优机器人钩快剥离技术 

**Authors**: Jiacheng Yuan, Changhyun Choi, Volkan Isler  

**Link**: [PDF](https://arxiv.org/pdf/2506.05812)  

**Abstract**: We study the problem of peeling a Velcro strap from a surface using a robotic manipulator. The surface geometry is arbitrary and unknown. The robot has access to only the force feedback and its end-effector position. This problem is challenging due to the partial observability of the environment and the incompleteness of the sensor feedback. To solve it, we first model the system with simple analytic state and action models based on quasi-static dynamics assumptions. We then study the fully-observable case where the state of both the Velcro and the robot are given. For this case, we obtain the optimal solution in closed-form which minimizes the total energy cost. Next, for the partially-observable case, we design a state estimator which estimates the underlying state using only force and position feedback. Then, we present a heuristics-based controller that balances exploratory and exploitative behaviors in order to peel the velcro efficiently. Finally, we evaluate our proposed method in environments with complex geometric uncertainties and sensor noises, achieving 100% success rate with less than 80% increase in energy cost compared to the optimal solution when the environment is fully-observable, outperforming the baselines by a large margin. 

**Abstract (ZH)**: 使用机器人 manipulator 剥离 Velcro 条带的问题：从几何未知的表面剥离，基于部分可观测性和传感器反馈不完全性分析与解决 

---
# Where Do We Look When We Teach? Analyzing Human Gaze Behavior Across Demonstration Devices in Robot Imitation Learning 

**Title (ZH)**: 我们教学时看向何处？在机器人模仿学习中演示设备的人类凝视行为分析 

**Authors**: Yutaro Ishida, Takamitsu Matsubara, Takayuki Kanai, Kazuhiro Shintani, Hiroshi Bito  

**Link**: [PDF](https://arxiv.org/pdf/2506.05808)  

**Abstract**: Imitation learning for acquiring generalizable policies often requires a large volume of demonstration data, making the process significantly costly. One promising strategy to address this challenge is to leverage the cognitive and decision-making skills of human demonstrators with strong generalization capability, particularly by extracting task-relevant cues from their gaze behavior. However, imitation learning typically involves humans collecting data using demonstration devices that emulate a robot's embodiment and visual condition. This raises the question of how such devices influence gaze behavior. We propose an experimental framework that systematically analyzes demonstrators' gaze behavior across a spectrum of demonstration devices. Our experimental results indicate that devices emulating (1) a robot's embodiment or (2) visual condition impair demonstrators' capability to extract task-relevant cues via gaze behavior, with the extent of impairment depending on the degree of emulation. Additionally, gaze data collected using devices that capture natural human behavior improves the policy's task success rate from 18.8% to 68.8% under environmental shifts. 

**Abstract (ZH)**: 基于认知和决策技能的人类演示者在获取可泛化策略中的模仿学习往往需要大量示例数据，使这一过程变得成本高昂。一种有潜力的策略是利用具有强泛化能力的人类演示者认知和决策技能，特别是通过提取与任务相关的线索来利用他们的注视行为。然而，模仿学习通常涉及人类使用模拟机器人身体和视觉条件的设备来收集数据。这引发了这些设备如何影响注视行为的问题。我们提出了一种实验框架，系统分析不同演示设备下演示者注视行为的变化。实验结果表明，模拟（1）机器人身体或（2）视觉条件的设备会妨碍演示者通过注视行为提取与任务相关的线索，这种妨碍的程度取决于模拟的程度。此外，在环境变化下，使用捕捉自然人类行为的设备收集的眼动数据可以使策略的任务成功率从18.8%提高到68.8%。 

---
# A Soft Robotic Module with Pneumatic Actuation and Enhanced Controllability Using a Shape Memory Alloy Wire 

**Title (ZH)**: 一种使用形状记忆合金丝增强可控性的气动软机器人模块 

**Authors**: Mohammadnavid Golchin  

**Link**: [PDF](https://arxiv.org/pdf/2506.05741)  

**Abstract**: In this paper, a compressed air-actuated soft robotic module was developed by incorporating a shape memory alloy (SMA) wire into its structure to achieve the desired bending angle with greater precision. First, a fiber-reinforced bending module with a strain-limiting layer made of polypropylene was fabricated. The SMA wire was then placed in a silicon matrix, which was used as a new strain-limiting layer. A simple closed-loop control algorithm was used to regulate the bending angle of the soft robot within its workspace. A camera was utilized to measure the angular changes in the vertical plane. Different angles, ranging from 0 to 65 degrees, were covered to evaluate the performance of the module and the bending angle control algorithm. The experimental tests demonstrate that using the SMA wire results in more precise control of bending in the vertical plane. In addition, it is possible to bend more with less working pressure. The error range was reduced from an average of 5 degrees to 2 degrees, and the rise time was reduced from an average of 19 seconds to 3 seconds. 

**Abstract (ZH)**: 一种集成形状记忆合金丝的压缩空气驱动柔性机器人模块及其精确弯曲角度控制研究 

---
# Advancement and Field Evaluation of a Dual-arm Apple Harvesting Robot 

**Title (ZH)**: 双臂苹果采摘机器人技术进展与田间评估 

**Authors**: Keyi Zhu, Kyle Lammers, Kaixiang Zhang, Chaaran Arunachalam, Siddhartha Bhattacharya, Jiajia Li, Renfu Lu, Zhaojian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05714)  

**Abstract**: Apples are among the most widely consumed fruits worldwide. Currently, apple harvesting fully relies on manual labor, which is costly, drudging, and hazardous to workers. Hence, robotic harvesting has attracted increasing attention in recent years. However, existing systems still fall short in terms of performance, effectiveness, and reliability for complex orchard environments. In this work, we present the development and evaluation of a dual-arm harvesting robot. The system integrates a ToF camera, two 4DOF robotic arms, a centralized vacuum system, and a post-harvest handling module. During harvesting, suction force is dynamically assigned to either arm via the vacuum system, enabling efficient apple detachment while reducing power consumption and noise. Compared to our previous design, we incorporated a platform movement mechanism that enables both in-out and up-down adjustments, enhancing the robot's dexterity and adaptability to varying canopy structures. On the algorithmic side, we developed a robust apple localization pipeline that combines a foundation-model-based detector, segmentation, and clustering-based depth estimation, which improves performance in orchards. Additionally, pressure sensors were integrated into the system, and a novel dual-arm coordination strategy was introduced to respond to harvest failures based on sensor feedback, further improving picking efficiency. Field demos were conducted in two commercial orchards in MI, USA, with different canopy structures. The system achieved success rates of 0.807 and 0.797, with an average picking cycle time of 5.97s. The proposed strategy reduced harvest time by 28% compared to a single-arm baseline. The dual-arm harvesting robot enhances the reliability and efficiency of apple picking. With further advancements, the system holds strong potential for autonomous operation and commercialization for the apple industry. 

**Abstract (ZH)**: 双臂苹果采摘机器人开发与评价 

---
# Towards Autonomous In-situ Soil Sampling and Mapping in Large-Scale Agricultural Environments 

**Title (ZH)**: 面向大规模农业环境中的自主就位土壤采样与制图 

**Authors**: Thien Hoang Nguyen, Erik Muller, Michael Rubin, Xiaofei Wang, Fiorella Sibona, Salah Sukkarieh  

**Link**: [PDF](https://arxiv.org/pdf/2506.05653)  

**Abstract**: Traditional soil sampling and analysis methods are labor-intensive, time-consuming, and limited in spatial resolution, making them unsuitable for large-scale precision agriculture. To address these limitations, we present a robotic solution for real-time sampling, analysis and mapping of key soil properties. Our system consists of two main sub-systems: a Sample Acquisition System (SAS) for precise, automated in-field soil sampling; and a Sample Analysis Lab (Lab) for real-time soil property analysis. The system's performance was validated through extensive field trials at a large-scale Australian farm. Experimental results show that the SAS can consistently acquire soil samples with a mass of 50g at a depth of 200mm, while the Lab can process each sample within 10 minutes to accurately measure pH and macronutrients. These results demonstrate the potential of the system to provide farmers with timely, data-driven insights for more efficient and sustainable soil management and fertilizer application. 

**Abstract (ZH)**: 传统的土壤采样与分析方法劳动密集、耗时且空间分辨率有限，不适合大规模精准农业的应用。为解决这些局限性，我们提出了一种机器人解决方案，用于实时采集、分析和制图关键土壤属性。该系统包括两个主要子系统：一个精确的自动化现场土壤采样系统（SAS）；和一个实时土壤属性分析实验室（Lab）。该系统的性能通过在澳大利亚大型农场进行的广泛田间试验得到了验证。实验结果表明，SAS可以一致地在200mm深度处采集50g的土壤样本，而Lab可以在10分钟内处理每个样本以准确测量pH值和宏量元素。这些结果展示了该系统为农民提供及时的数据驱动见解以实现更高效和可持续的土壤管理和肥料施用的潜力。 

---
# TD-TOG Dataset: Benchmarking Zero-Shot and One-Shot Task-Oriented Grasping for Object Generalization 

**Title (ZH)**: TD-TOG数据集：面向对象泛化的零-shot和少-shot任务导向抓取基准测试 

**Authors**: Valerija Holomjova, Jamie Grech, Dewei Yi, Bruno Yun, Andrew Starkey, Pascal Meißner  

**Link**: [PDF](https://arxiv.org/pdf/2506.05576)  

**Abstract**: Task-oriented grasping (TOG) is an essential preliminary step for robotic task execution, which involves predicting grasps on regions of target objects that facilitate intended tasks. Existing literature reveals there is a limited availability of TOG datasets for training and benchmarking despite large demand, which are often synthetic or have artifacts in mask annotations that hinder model performance. Moreover, TOG solutions often require affordance masks, grasps, and object masks for training, however, existing datasets typically provide only a subset of these annotations. To address these limitations, we introduce the Top-down Task-oriented Grasping (TD-TOG) dataset, designed to train and evaluate TOG solutions. TD-TOG comprises 1,449 real-world RGB-D scenes including 30 object categories and 120 subcategories, with hand-annotated object masks, affordances, and planar rectangular grasps. It also features a test set for a novel challenge that assesses a TOG solution's ability to distinguish between object subcategories. To contribute to the demand for TOG solutions that can adapt and manipulate previously unseen objects without re-training, we propose a novel TOG framework, Binary-TOG. Binary-TOG uses zero-shot for object recognition, and one-shot learning for affordance recognition. Zero-shot learning enables Binary-TOG to identify objects in multi-object scenes through textual prompts, eliminating the need for visual references. In multi-object settings, Binary-TOG achieves an average task-oriented grasp accuracy of 68.9%. Lastly, this paper contributes a comparative analysis between one-shot and zero-shot learning for object generalization in TOG to be used in the development of future TOG solutions. 

**Abstract (ZH)**: 面向任务的抓取（TOG）数据集：Top-down Task-oriented Grasping (TD-TOG) 

---
# Learning to Recover: Dynamic Reward Shaping with Wheel-Leg Coordination for Fallen Robots 

**Title (ZH)**: 基于轮腿协调的摔倒机器人动态奖励塑形学习恢复方法 

**Authors**: Boyuan Deng, Luca Rossini, Jin Wang, Weijie Wang, Nikolaos Tsagarakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.05516)  

**Abstract**: Adaptive recovery from fall incidents are essential skills for the practical deployment of wheeled-legged robots, which uniquely combine the agility of legs with the speed of wheels for rapid recovery. However, traditional methods relying on preplanned recovery motions, simplified dynamics or sparse rewards often fail to produce robust recovery policies. This paper presents a learning-based framework integrating Episode-based Dynamic Reward Shaping and curriculum learning, which dynamically balances exploration of diverse recovery maneuvers with precise posture refinement. An asymmetric actor-critic architecture accelerates training by leveraging privileged information in simulation, while noise-injected observations enhance robustness against uncertainties. We further demonstrate that synergistic wheel-leg coordination reduces joint torque consumption by 15.8% and 26.2% and improves stabilization through energy transfer mechanisms. Extensive evaluations on two distinct quadruped platforms achieve recovery success rates up to 99.1% and 97.8% without platform-specific tuning. The supplementary material is available at this https URL 

**Abstract (ZH)**: 基于 episodes 动态奖励塑形和 Curriculum 学习的自适应跌倒恢复学习框架：轮腿机器人快速恢复的精确姿态调控与探索平衡 

---
# Trajectory Optimization for UAV-Based Medical Delivery with Temporal Logic Constraints and Convex Feasible Set Collision Avoidance 

**Title (ZH)**: 基于时间逻辑约束和凸可行集碰撞 avoidance 的无人机医疗配送轨迹优化 

**Authors**: Kaiyuan Chen, Yuhan Suo, Shaowei Cui, Yuanqing Xia, Wannian Liang, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06038)  

**Abstract**: This paper addresses the problem of trajectory optimization for unmanned aerial vehicles (UAVs) performing time-sensitive medical deliveries in urban environments. Specifically, we consider a single UAV with 3 degree-of-freedom dynamics tasked with delivering blood packages to multiple hospitals, each with a predefined time window and priority. Mission objectives are encoded using Signal Temporal Logic (STL), enabling the formal specification of spatial-temporal constraints. To ensure safety, city buildings are modeled as 3D convex obstacles, and obstacle avoidance is handled through a Convex Feasible Set (CFS) method. The entire planning problem-combining UAV dynamics, STL satisfaction, and collision avoidance-is formulated as a convex optimization problem that ensures tractability and can be solved efficiently using standard convex programming techniques. Simulation results demonstrate that the proposed method generates dynamically feasible, collision-free trajectories that satisfy temporal mission goals, providing a scalable and reliable approach for autonomous UAV-based medical logistics. 

**Abstract (ZH)**: 本文探讨了在城市环境中执行时间敏感医疗配送任务的无人驾驶飞行器（UAV）轨迹优化问题。具体而言，我们考虑了一架具有3自由度动力学的单个UAV，其任务是向多个具有预定义时间和优先级的医院配送血包。使用信号时态逻辑（STL）编码任务目标，使得能够正式规定空间-时间约束。为了确保安全，将城市建筑物建模为3D凸障碍，并通过凸可行集（CFS）方法处理障碍物避免问题。整个规划问题结合UAV动力学、STL满足性和碰撞避免，被形式化为一个凸优化问题，该问题具有可处理性，并且可以使用标准凸规划技术高效地求解。仿真实验结果表明，所提出的方法能够生成动态可行、无碰撞的轨迹，满足时间任务目标，提供了一种可扩展且可靠的基于自主UAV的医疗物流方法。 

---
# Equivariant Filter for Relative Attitude and Target Angular Velocity Estimation 

**Title (ZH)**: 相对姿态和目标角速度 estimation 的协变滤波器 

**Authors**: Gil Serrano, Bruno J. Guerreiro, Pedro Lourenço, Rita Cunha  

**Link**: [PDF](https://arxiv.org/pdf/2506.06016)  

**Abstract**: Accurate estimation of the relative attitude and angular velocity between two rigid bodies is fundamental in aerospace applications such as spacecraft rendezvous and docking. In these scenarios, a chaser vehicle must determine the orientation and angular velocity of a target object using onboard sensors. This work addresses the challenge of designing an Equivariant Filter (EqF) that can reliably estimate both the relative attitude and the target angular velocity using noisy observations of two known, non-collinear vectors fixed in the target frame. To derive the EqF, a symmetry for the system is proposed and an equivariant lift onto the symmetry group is calculated. Observability and convergence properties are analyzed. Simulations demonstrate the filter's performance, with Monte Carlo runs yielding statistically significant results. The impact of low-rate measurements is also examined and a strategy to mitigate this effect is proposed. Experimental results, using fiducial markers and both conventional and event cameras for measurement acquisition, further validate the approach, confirming its effectiveness in a realistic setting. 

**Abstract (ZH)**: 在航天航空应用中基于两刚体之间相对姿态和角速度的准确估计方法 

---
# Dynamic Mixture of Progressive Parameter-Efficient Expert Library for Lifelong Robot Learning 

**Title (ZH)**: lifelong 机器人学习中渐进参数高效专家库的动态混合 

**Authors**: Yuheng Lei, Sitong Mao, Shunbo Zhou, Hongyuan Zhang, Xuelong Li, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05985)  

**Abstract**: A generalist agent must continuously learn and adapt throughout its lifetime, achieving efficient forward transfer while minimizing catastrophic forgetting. Previous work within the dominant pretrain-then-finetune paradigm has explored parameter-efficient fine-tuning for single-task adaptation, effectively steering a frozen pretrained model with a small number of parameters. However, in the context of lifelong learning, these methods rely on the impractical assumption of a test-time task identifier and restrict knowledge sharing among isolated adapters. To address these limitations, we propose Dynamic Mixture of Progressive Parameter-Efficient Expert Library (DMPEL) for lifelong robot learning. DMPEL progressively learn a low-rank expert library and employs a lightweight router to dynamically combine experts into an end-to-end policy, facilitating flexible behavior during lifelong adaptation. Moreover, by leveraging the modular structure of the fine-tuned parameters, we introduce coefficient replay to guide the router in accurately retrieving frozen experts for previously encountered tasks, thereby mitigating catastrophic forgetting. This method is significantly more storage- and computationally-efficient than applying demonstration replay to the entire policy. Extensive experiments on the lifelong manipulation benchmark LIBERO demonstrate that our framework outperforms state-of-the-art lifelong learning methods in success rates across continual adaptation, while utilizing minimal trainable parameters and storage. 

**Abstract (ZH)**: 一种通用智能体必须在其生命周期内持续学习和适应，实现高效的前向迁移同时尽量减少灾难性遗忘。在占主导地位的预训练-然后微调范式中，先前工作探索了单任务适应的参数高效微调方法，有效利用少量参数引导冻结的预训练模型。然而，在终身学习的背景下，这些方法依赖于在测试时需任务标识符这一 impractical 的假设，并限制了隔离适配器之间知识的共享。为了解决这些限制，我们提出了一种动态渐进参数高效专家库（DMPEL）方法，用于终身机器人学习。DMPEL 逐步学习一个低秩专家库，并采用一个轻量级路由器动态组合专家，促进终身适应过程中灵活的行为。此外，利用微调参数的模块化结构，我们引入了系数重播，指导路由器准确检索之前遇到的任务的冻结专家，从而减轻灾难性遗忘。该方法在整体策略上应用演示重播方面更为存储和计算高效。在终身操作基准 LIBERO 上的广泛实验表明，我们的框架在持续适应过程中成功率达到最新水平，同时使用最少的可训练参数和存储空间。 

---
# Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning 

**Title (ZH)**: 在线强化学习中贝叶斯最优性运算子向贝叶斯运算子的渐进转换 

**Authors**: Motoki Omura, Kazuki Ota, Takayuki Osa, Yusuke Mukuta, Tatsuya Harada  

**Link**: [PDF](https://arxiv.org/pdf/2506.05968)  

**Abstract**: For continuous action spaces, actor-critic methods are widely used in online reinforcement learning (RL). However, unlike RL algorithms for discrete actions, which generally model the optimal value function using the Bellman optimality operator, RL algorithms for continuous actions typically model Q-values for the current policy using the Bellman operator. These algorithms for continuous actions rely exclusively on policy updates for improvement, which often results in low sample efficiency. This study examines the effectiveness of incorporating the Bellman optimality operator into actor-critic frameworks. Experiments in a simple environment show that modeling optimal values accelerates learning but leads to overestimation bias. To address this, we propose an annealing approach that gradually transitions from the Bellman optimality operator to the Bellman operator, thereby accelerating learning while mitigating bias. Our method, combined with TD3 and SAC, significantly outperforms existing approaches across various locomotion and manipulation tasks, demonstrating improved performance and robustness to hyperparameters related to optimality. 

**Abstract (ZH)**: 将贝尔曼最优算子融入actor-critic框架的有效性研究：加速学习并减轻偏差 

---
# Trajectory Entropy: Modeling Game State Stability from Multimodality Trajectory Prediction 

**Title (ZH)**: 轨迹熵：基于多模态轨迹预测的游戏状态稳定性建模 

**Authors**: Yesheng Zhang, Wenjian Sun, Yuheng Chen, Qingwei Liu, Qi Lin, Rui Zhang, Xu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05810)  

**Abstract**: Complex interactions among agents present a significant challenge for autonomous driving in real-world scenarios. Recently, a promising approach has emerged, which formulates the interactions of agents as a level-k game framework. It effectively decouples agent policies by hierarchical game levels. However, this framework ignores both the varying driving complexities among agents and the dynamic changes in agent states across game levels, instead treating them uniformly. Consequently, redundant and error-prone computations are introduced into this framework. To tackle the issue, this paper proposes a metric, termed as Trajectory Entropy, to reveal the game status of agents within the level-k game framework. The key insight stems from recognizing the inherit relationship between agent policy uncertainty and the associated driving complexity. Specifically, Trajectory Entropy extracts statistical signals representing uncertainty from the multimodality trajectory prediction results of agents in the game. Then, the signal-to-noise ratio of this signal is utilized to quantify the game status of agents. Based on the proposed Trajectory Entropy, we refine the current level-k game framework through a simple gating mechanism, significantly improving overall accuracy while reducing computational costs. Our method is evaluated on the Waymo and nuPlan datasets, in terms of trajectory prediction, open-loop and closed-loop planning tasks. The results demonstrate the state-of-the-art performance of our method, with precision improved by up to 19.89% for prediction and up to 16.48% for planning. 

**Abstract (ZH)**: 基于轨迹熵改进的level-k游戏框架在自主驾驶中的应用 

---
# EqCollide: Equivariant and Collision-Aware Deformable Objects Neural Simulator 

**Title (ZH)**: EqCollide: 具有equivariance和碰撞意识的可变形对象神经模拟器 

**Authors**: Qianyi Chen, Tianrun Gao, Chenbo Jiang, Tailin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05797)  

**Abstract**: Simulating collisions of deformable objects is a fundamental yet challenging task due to the complexity of modeling solid mechanics and multi-body interactions. Existing data-driven methods often suffer from lack of equivariance to physical symmetries, inadequate handling of collisions, and limited scalability. Here we introduce EqCollide, the first end-to-end equivariant neural fields simulator for deformable objects and their collisions. We propose an equivariant encoder to map object geometry and velocity into latent control points. A subsequent equivariant Graph Neural Network-based Neural Ordinary Differential Equation models the interactions among control points via collision-aware message passing. To reconstruct velocity fields, we query a neural field conditioned on control point features, enabling continuous and resolution-independent motion predictions. Experimental results show that EqCollide achieves accurate, stable, and scalable simulations across diverse object configurations, and our model achieves 24.34% to 35.82% lower rollout MSE even compared with the best-performing baseline model. Furthermore, our model could generalize to more colliding objects and extended temporal horizons, and stay robust to input transformed with group action. 

**Abstract (ZH)**: 基于等变神经场的可变形物体及其碰撞的端到端模拟 

---
# Robust sensor fusion against on-vehicle sensor staleness 

**Title (ZH)**: 车载传感器陈旧性抵抗的传感器融合robust sensor fusion against on-vehicle sensor staleness 

**Authors**: Meng Fan, Yifan Zuo, Patrick Blaes, Harley Montgomery, Subhasis Das  

**Link**: [PDF](https://arxiv.org/pdf/2506.05780)  

**Abstract**: Sensor fusion is crucial for a performant and robust Perception system in autonomous vehicles, but sensor staleness, where data from different sensors arrives with varying delays, poses significant challenges. Temporal misalignment between sensor modalities leads to inconsistent object state estimates, severely degrading the quality of trajectory predictions that are critical for safety. We present a novel and model-agnostic approach to address this problem via (1) a per-point timestamp offset feature (for LiDAR and radar both relative to camera) that enables fine-grained temporal awareness in sensor fusion, and (2) a data augmentation strategy that simulates realistic sensor staleness patterns observed in deployed vehicles. Our method is integrated into a perspective-view detection model that consumes sensor data from multiple LiDARs, radars and cameras. We demonstrate that while a conventional model shows significant regressions when one sensor modality is stale, our approach reaches consistently good performance across both synchronized and stale conditions. 

**Abstract (ZH)**: 传感器融合对于自动驾驶车辆高效稳健的感知系统至关重要，但不同传感器数据的时延差异构成了重大挑战。传感器模态之间的时间不对齐导致物体状态估计不一致，严重降低了对安全至关重要的轨迹预测质量。我们提出了一种新颖且模型无关的方法，通过（1）一个点级别的时间戳偏移特征（LiDAR和雷达相对于相机的时间偏移），实现传感器融合中的细粒度时间感知，以及（2）一种模拟部署车辆中观察到的传感器时延模式的数据增强策略来解决这一问题。该方法被集成到一个使用多LiDAR、雷达和摄像头传感器数据的视角检测模型中。我们证明，在一种传感器模态失效的情况下，传统模型会出现显著退化，而我们的方法在同步和失效条件下都能保持一致的良好性能。 

---
# You Only Estimate Once: Unified, One-stage, Real-Time Category-level Articulated Object 6D Pose Estimation for Robotic Grasping 

**Title (ZH)**: 只需一次估计：统一的一阶段实时类别级 articulated 对象 6D 姿态估计用于机器人抓取 

**Authors**: Jingshun Huang, Haitao Lin, Tianyu Wang, Yanwei Fu, Yu-Gang Jiang, Xiangyang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2506.05719)  

**Abstract**: This paper addresses the problem of category-level pose estimation for articulated objects in robotic manipulation tasks. Recent works have shown promising results in estimating part pose and size at the category level. However, these approaches primarily follow a complex multi-stage pipeline that first segments part instances in the point cloud and then estimates the Normalized Part Coordinate Space (NPCS) representation for 6D poses. These approaches suffer from high computational costs and low performance in real-time robotic tasks. To address these limitations, we propose YOEO, a single-stage method that simultaneously outputs instance segmentation and NPCS representations in an end-to-end manner. We use a unified network to generate point-wise semantic labels and centroid offsets, allowing points from the same part instance to vote for the same centroid. We further utilize a clustering algorithm to distinguish points based on their estimated centroid distances. Finally, we first separate the NPCS region of each instance. Then, we align the separated regions with the real point cloud to recover the final pose and size. Experimental results on the GAPart dataset demonstrate the pose estimation capabilities of our proposed single-shot method. We also deploy our synthetically-trained model in a real-world setting, providing real-time visual feedback at 200Hz, enabling a physical Kinova robot to interact with unseen articulated objects. This showcases the utility and effectiveness of our proposed method. 

**Abstract (ZH)**: 本文解决了机器人操作任务中articulated对象类别级别姿态估计的问题。最近的研究在估计部件姿态和大小方面取得了令人鼓舞的结果。然而，这些方法主要遵循一个复杂多阶段的流程，首先对点云中的部件实例进行分割，然后估计归一化部件坐标空间(NPCS)表示以获取6D姿态。这些方法在实时机器人任务中面临高计算成本和低性能的限制。为了解决这些问题，我们提出了YOEO，一种单阶段方法，在端到端的方式下同时输出实例分割和NPCS表示。我们使用一个统一网络生成点级语义标签和质心偏移量，使得同一部件实例的点可以投票支持同一个质心。我们进一步利用聚类算法根据点的估计质心距离对点进行区分。最后，我们首先分离每个实例的NPCS区域，然后将分离的区域与真实点云对齐以恢复最终的姿态和尺寸。在GAPart数据集上的实验结果展示了我们提出的单次检测方法的姿态估计能力。我们还将合成训练的模型部署到实际应用场景中，以200Hz的速度提供实时视觉反馈，使物理Kinova机器人能够与未见过的articulated对象进行互动。这展示了我们提出方法的实用性和有效性。 

---
# A Modular Haptic Display with Reconfigurable Signals for Personalized Information Transfer 

**Title (ZH)**: 一种可重构信号的模块化触觉显示装置，实现个性化信息传输 

**Authors**: Antonio Alvarez Valdivia, Benjamin A. Christie, Dylan P. Losey, Laura H. Blumenschein  

**Link**: [PDF](https://arxiv.org/pdf/2506.05648)  

**Abstract**: We present a customizable soft haptic system that integrates modular hardware with an information-theoretic algorithm to personalize feedback for different users and tasks. Our platform features modular, multi-degree-of-freedom pneumatic displays, where different signal types, such as pressure, frequency, and contact area, can be activated or combined using fluidic logic circuits. These circuits simplify control by reducing reliance on specialized electronics and enabling coordinated actuation of multiple haptic elements through a compact set of inputs. Our approach allows rapid reconfiguration of haptic signal rendering through hardware-level logic switching without rewriting code. Personalization of the haptic interface is achieved through the combination of modular hardware and software-driven signal selection. To determine which display configurations will be most effective, we model haptic communication as a signal transmission problem, where an agent must convey latent information to the user. We formulate the optimization problem to identify the haptic hardware setup that maximizes the information transfer between the intended message and the user's interpretation, accounting for individual differences in sensitivity, preferences, and perceptual salience. We evaluate this framework through user studies where participants interact with reconfigurable displays under different signal combinations. Our findings support the role of modularity and personalization in creating multimodal haptic interfaces and advance the development of reconfigurable systems that adapt with users in dynamic human-machine interaction contexts. 

**Abstract (ZH)**: 可定制的软触觉系统：基于模块化硬件和信息论算法的个性化反馈集成平台 

---
# A Compendium of Autonomous Navigation using Object Detection and Tracking in Unmanned Aerial Vehicles 

**Title (ZH)**: 基于对象检测与跟踪的自主导航综合研究（应用于无人驾驶航空车辆） 

**Authors**: Mohit Arora, Pratyush Shukla, Shivali Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2506.05378)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are one of the most revolutionary inventions of 21st century. At the core of a UAV lies the central processing system that uses wireless signals to control their movement. The most popular UAVs are quadcopters that use a set of four motors, arranged as two on either side with opposite spin. An autonomous UAV is called a drone. Drones have been in service in the US army since the 90's for covert missions critical to national security. It would not be wrong to claim that drones make up an integral part of the national security and provide the most valuable service during surveillance operations. While UAVs are controlled using wireless signals, there reside some challenges that disrupt the operation of such vehicles such as signal quality and range, real time processing, human expertise, robust hardware and data security. These challenges can be solved by programming UAVs to be autonomous, using object detection and tracking, through Computer Vision algorithms. Computer Vision is an interdisciplinary field that seeks the use of deep learning to gain a high-level understanding of digital images and videos for the purpose of automating the task of human visual system. Using computer vision, algorithms for detecting and tracking various objects can be developed suitable to the hardware so as to allow real time processing for immediate judgement. This paper attempts to review the various approaches several authors have proposed for the purpose of autonomous navigation of UAVs by through various algorithms of object detection and tracking in real time, for the purpose of applications in various fields such as disaster management, dense area exploration, traffic vehicle surveillance etc. 

**Abstract (ZH)**: 无人驾驶航空车辆（UAVs）是21世纪最革命性的发明之一。UAV的核心是中央处理系统，通过无线信号控制其运动。最流行的UAV是四旋翼无人机，它配备了一组四个电机，每边两个，旋转方向相反。自主无人机称为无人机。无人机自20世纪90年代以来一直在美国军队中服役，用于关键性的隐蔽任务，对国家安全至关重要。可以说，无人机是国家安全不可或缺的一部分，并在 surveillance 操作中提供了最宝贵的服务。虽然UAV是通过无线信号控制的，但存在一些挑战，如信号质量与范围、实时处理、人力专业知识、鲁棒硬件和数据安全等问题。通过编程使无人机自主，使用计算机视觉算法进行目标检测与跟踪，可以解决这些问题。计算机视觉是一个跨学科领域，利用深度学习来理解数字图像和视频，旨在自动化人类视觉系统完成的任务。利用计算机视觉，可以开发适合硬件的检测与跟踪算法，以实现实时处理并立即做出判断。本文旨在回顾若干作者提出的多种方法，通过实时目标检测与跟踪算法实现无人机的自主导航，以应用于灾害管理、密集区域探索、交通车辆监控等各个领域。 

---
