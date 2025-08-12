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
# MolmoAct: Action Reasoning Models that can Reason in Space 

**Title (ZH)**: MolmoAct: 可以进行空间推理的动作推理模型 

**Authors**: Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, Winson Han, Wilbert Pumacay, Angelica Wu, Rose Hendrix, Karen Farley, Eli VanderBilt, Ali Farhadi, Dieter Fox, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2508.07917)  

**Abstract**: Reasoning is central to purposeful action, yet most robotic foundation models map perception and instructions directly to control, which limits adaptability, generalization, and semantic grounding. We introduce Action Reasoning Models (ARMs), a class of vision-language-action models that integrate perception, planning, and control through a structured three-stage pipeline. Our model, MolmoAct, encodes observations and instructions into depth-aware perception tokens, generates mid-level spatial plans as editable trajectory traces, and predicts precise low-level actions, enabling explainable and steerable behavior. MolmoAct-7B-D achieves strong performance across simulation and real-world settings: 70.5% zero-shot accuracy on SimplerEnv Visual Matching tasks, surpassing closed-source Pi-0 and GR00T N1; 86.6% average success on LIBERO, including an additional 6.3% gain over ThinkAct on long-horizon tasks; and in real-world fine-tuning, an additional 10% (single-arm) and an additional 22.7% (bimanual) task progression over Pi-0-FAST. It also outperforms baselines by an additional 23.3% on out-of-distribution generalization and achieves top human-preference scores for open-ended instruction following and trajectory steering. Furthermore, we release, for the first time, the MolmoAct Dataset -- a mid-training robot dataset comprising over 10,000 high quality robot trajectories across diverse scenarios and tasks. Training with this dataset yields an average 5.5% improvement in general performance over the base model. We release all model weights, training code, our collected dataset, and our action reasoning dataset, establishing MolmoAct as both a state-of-the-art robotics foundation model and an open blueprint for building ARMs that transform perception into purposeful action through structured reasoning. Blogpost: this https URL 

**Abstract (ZH)**: 行动推理是富有目的行动的核心，但大多数机器人基础模型直接将感知和指令映射到控制，这限制了其适应性、泛化能力和语义 grounding。我们引入了行动推理模型（ARMs），这是一种通过结构化三阶段管道整合感知、规划和控制的视觉-语言-行动模型类别。我们的模型 MolmoAct 将观察和指令编码为深度感知标记，生成可编辑的空间计划轨迹，并预测精确的低级动作，从而实现可解释和可控的行为。MolmoAct-7B-D 在模拟和真实世界环境中均表现出强大的性能：在 SimplerEnv Visual Matching 任务中实现 70.5% 的零样本准确率，超过闭源 Pi-0 和 GR00T N1；在 LIBERO 中的平均成功率高达 86.6%，比 ThinkAct 在长视距任务中多出 6.3% 的增益；在真实世界的微调中，单臂任务进展多出 10%，双臂任务进展多出 22.7%，均超过 Pi-0-FAST。此外，它在离群样本泛化上还超过了基线 23.3%，并获得了开放指令跟随和轨迹操控的顶级人工偏好评分。此外，我们首次发布了 MolmoAct 数据集——一个包含逾 10,000 条高质量机器人轨迹的中期训练机器人数据集，涵盖多种场景和任务。使用该数据集进行训练能令基模型的整体性能平均提高 5.5%。我们发布了所有模型权重、训练代码、收集的数据集以及行动推理数据集，使 MolmoAct 成为一种前沿的机器人基础模型，并提供了通过结构化推理将感知转化为富有目的行动的开放蓝图。博客：this https URL 

---
# Autonomous Navigation of Cloud-Controlled Quadcopters in Confined Spaces Using Multi-Modal Perception and LLM-Driven High Semantic Reasoning 

**Title (ZH)**: 基于多模态感知和 大语言模型驱动高语义推理的受限空间内受控四关于我们群控无人机自主导航标题 

**Authors**: Shoaib Ahmmad, Zubayer Ahmed Aditto, Md Mehrab Hossain, Noushin Yeasmin, Shorower Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2508.07885)  

**Abstract**: This paper introduces an advanced AI-driven perception system for autonomous quadcopter navigation in GPS-denied indoor environments. The proposed framework leverages cloud computing to offload computationally intensive tasks and incorporates a custom-designed printed circuit board (PCB) for efficient sensor data acquisition, enabling robust navigation in confined spaces. The system integrates YOLOv11 for object detection, Depth Anything V2 for monocular depth estimation, a PCB equipped with Time-of-Flight (ToF) sensors and an Inertial Measurement Unit (IMU), and a cloud-based Large Language Model (LLM) for context-aware decision-making. A virtual safety envelope, enforced by calibrated sensor offsets, ensures collision avoidance, while a multithreaded architecture achieves low-latency processing. Enhanced spatial awareness is facilitated by 3D bounding box estimation with Kalman filtering. Experimental results in an indoor testbed demonstrate strong performance, with object detection achieving a mean Average Precision (mAP50) of 0.6, depth estimation Mean Absolute Error (MAE) of 7.2 cm, only 16 safety envelope breaches across 42 trials over approximately 11 minutes, and end-to-end system latency below 1 second. This cloud-supported, high-intelligence framework serves as an auxiliary perception and navigation system, complementing state-of-the-art drone autonomy for GPS-denied confined spaces. 

**Abstract (ZH)**: 一种基于云的支持高intelligence的自主室内GPS受限四旋翼机感知与导航系统 

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

**Abstract (ZH)**: 对于 radial 截肢患者，仿人手假肢有望恢复进行日常生活的能力。为了推进下一代假手控制设计，亟需解决在实验室外环境鲁棒性差和对新环境的一般性差的问题。由于现有数据集中交互对象数量固定，而现实世界中遇到的对象几乎无限多变，当前的抓取模型在未见过的对象上表现不佳，严重影响用户的独立性和生活质量。

为了应对这一挑战：（i）我们定义了语义投影，即模型具备将技巧推广到未见过的对象类型的能力，并证明了尽管 YOLO 模型在训练集上的准确率达到 80%，但在未见过的对象上仅达到 15%。 （ii）我们提出了 Grasp-LLaVA，这是一种抓取视觉语言模型，能够进行类人的逻辑推理，基于对象的物理特征推断合适的抓取类型，相比当前最先进的抓取估计模型，在未见过的对象类型上的准确率提高了 33.5%。

最后，为了弥合性能与延迟之间的差距，我们提出了一种混合抓取网络（HGN），这是一种边缘-云部署基础设施，可以在边缘进行快速抓取估计，并在云端提供准确的推断作为冗余，从而有效扩展了延迟与准确性的 Pareto 效应。HGN 通过置信度校准（DC）能够动态切换边缘和云模型之间的使用，对于未见过的对象典型样例，其语义投影准确率提升了 5.6%（提高到 42.3%），并且估计速度提高了 3.5 倍。在实际样本混合中，HGN 达到 86% 的平均准确率（与仅使用边缘模型相比提高了 12.2%），并且比单独使用 Grasp-LLaVA 的推断速度更快两倍多。

翻译后的标题：
语义投影与Grasp-LLaVA：实现未见过对象的抓取类型估计 

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
# Triple-S: A Collaborative Multi-LLM Framework for Solving Long-Horizon Implicative Tasks in Robotics 

**Title (ZH)**: Triple-S: 一种解决机器人领域长时延推导任务的协作多大型语言模型框架 

**Authors**: Zixi Jia, Hongbin Gao, Fashe Li, Jiqiang Liu, Hexiao Li, Qinghua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.07421)  

**Abstract**: Leveraging Large Language Models (LLMs) to write policy code for controlling robots has gained significant attention. However, in long-horizon implicative tasks, this approach often results in API parameter, comments and sequencing errors, leading to task failure. To address this problem, we propose a collaborative Triple-S framework that involves multiple LLMs. Through In-Context Learning, different LLMs assume specific roles in a closed-loop Simplification-Solution-Summary process, effectively improving success rates and robustness in long-horizon implicative tasks. Additionally, a novel demonstration library update mechanism which learned from success allows it to generalize to previously failed tasks. We validate the framework in the Long-horizon Desktop Implicative Placement (LDIP) dataset across various baseline models, where Triple-S successfully executes 89% of tasks in both observable and partially observable scenarios. Experiments in both simulation and real-world robot settings further validated the effectiveness of Triple-S. Our code and dataset is available at: this https URL. 

**Abstract (ZH)**: 利用大型语言模型（LLMs）为控制机器人编写政策代码受到了广泛关注。但在长期任务中，这种做法 often 进一步导致 API 参数、注释和顺序错误，从而导致任务失败。为了解决这一问题，我们提出了一种协作性的 Triple-S 框架，涉及多个 LLMs。通过上下文学习，不同的 LLMs 在封闭环简化-解决方案-总结过程中承担特定角色，有效提高了长期任务的成功率和鲁棒性。此外，该框架还具有一项从成功中学习的新颖的演示库更新机制，使其能够泛化到之前失败的任务。我们在 Long-horizon Desktop Implicative Placement (LDIP) 数据集上对多种基线模型进行了验证，Triple-S 成功执行了 89% 的任务，无论是可观测场景还是部分可观测场景。在模拟与真实机器人环境中的实验进一步验证了 Triple-S 的有效性。代码和数据集可在以下链接获取：this https URL。 

---
# AgriVLN: Vision-and-Language Navigation for Agricultural Robots 

**Title (ZH)**: 农用VLN：农业机器人视觉与语言导航 

**Authors**: Xiaobei Zhao, Xingqi Lyu, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.07406)  

**Abstract**: Agricultural robots have emerged as powerful members in agricultural tasks, nevertheless, still heavily rely on manual operation or untransportable railway for movement, resulting in limited mobility and poor adaptability. Vision-and-Language Navigation (VLN) enables robots to navigate to the target destinations following natural language instructions, demonstrating strong performance on several domains. However, none of the existing benchmarks or methods is specifically designed for agricultural scenes. To bridge this gap, we propose Agriculture to Agriculture (A2A) benchmark, containing 1,560 episodes across six diverse agricultural scenes, in which all realistic RGB videos are captured by front-facing camera on a quadruped robot at a height of 0.38 meters, aligning with the practical deployment conditions. Meanwhile, we propose Vision-and-Language Navigation for Agricultural Robots (AgriVLN) baseline based on Vision-Language Model (VLM) prompted with carefully crafted templates, which can understand both given instructions and agricultural environments to generate appropriate low-level actions for robot control. When evaluated on A2A, AgriVLN performs well on short instructions but struggles with long instructions, because it often fails to track which part of the instruction is currently being executed. To address this, we further propose Subtask List (STL) instruction decomposition module and integrate it into AgriVLN, improving Success Rate (SR) from 0.33 to 0.47. We additionally compare AgriVLN with several existing VLN methods, demonstrating the state-of-the-art performance in the agricultural domain. 

**Abstract (ZH)**: 农业机器人在农业任务中已展现出强大的能力，然而仍高度依赖手动操作或不可移动的轨道进行移动，导致其移动能力有限且适应性差。基于视觉-语言导航（VLN）使机器人能够遵循自然语言指令导航至目标位置，已经在多个领域展现出强大性能。然而，目前所有现有的基准或方法均未专门设计用于农业场景。为填补这一空白，我们提出了农业到农业（A2A）基准，包含涵盖六大不同农业场景的1,560个 episodes，并且所有真实的RGB视频均由四足机器人前向摄像头在0.38米的高度拍摄，符合实际部署条件。同时，我们基于视觉-语言模型（VLM）提出了精心设计模板的农业机器人基于视觉-语言导航（AgriVLN）基线，能够理解给定的指令和农业环境以生成适当的低级动作以控制机器人。评估结果显示，AgriVLN在短指令上表现良好，但在长指令上表现出困难，因为它经常无法正确追踪当前执行的指令部分。为此，我们进一步提出了子任务列表（STL）指令分解模块，并将其整合到AgriVLN中，将成功率（SR）从0.33提高到0.47。此外，我们还将AgriVLN与其他现有的VLN方法进行了比较，展示了在农业领域中的先进性能。 

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
# $\mathcal{P}^3$: Toward Versatile Embodied Agents 

**Title (ZH)**: $\mathcal{P}^3$: 朝向多功能 embodied 代理的研究 

**Authors**: Shengli Zhou, Xiangchen Wang, Jinrui Zhang, Ruozai Tian, Rongtao Xu, Feng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.07033)  

**Abstract**: Embodied agents have shown promising generalization capabilities across diverse physical environments, making them essential for a wide range of real-world applications. However, building versatile embodied agents poses critical challenges due to three key issues: dynamic environment perception, open-ended tool usage, and complex multi-task planning. Most previous works rely solely on feedback from tool agents to perceive environmental changes and task status, which limits adaptability to real-time dynamics, causes error accumulation, and restricts tool flexibility. Furthermore, multi-task scheduling has received limited attention, primarily due to the inherent complexity of managing task dependencies and balancing competing priorities in dynamic and complex environments. To overcome these challenges, we introduce $\mathcal{P}^3$, a unified framework that integrates real-time perception and dynamic scheduling. Specifically, $\mathcal{P}^3$ enables 1) \textbf Perceive relevant task information actively from the environment, 2) \textbf Plug and utilize any tool without feedback requirement, and 3) \textbf Plan multi-task execution based on prioritizing urgent tasks and dynamically adjusting task order based on dependencies. Extensive real-world experiments show that our approach bridges the gap between benchmarks and practical deployment, delivering highly transferable, general-purpose embodied agents. Code and data will be released soon. 

**Abstract (ZH)**: 具身代理在多种物理环境中展示了令人鼓舞的泛化能力，使其成为广泛现实应用的重要组成部分。然而，构建多功能的具身代理面临三大关键挑战：动态环境感知、开放式工具使用和复杂多任务规划。大多数先前的工作仅依赖工具代理的反馈来感知环境变化和任务状态，这限制了其对实时动态的适应性，导致错误积累，并限制了工具的灵活性。此外，多任务调度受到动态和复杂环境中任务依赖性和竞争优先级管理内在复杂性的限制，关注度较低。为克服这些挑战，我们提出了一种统一框架$\mathcal{P}^3}$，该框架将实时感知与动态调度相结合。具体来说，$\mathcal{P}^3}$实现了以下功能：1) 主动从环境中感知相关任务信息，2) 插入并使用任何工具而无需反馈要求，3) 根据任务的紧迫性进行多任务执行规划，并基于任务依赖性动态调整任务顺序。广泛的实际世界实验表明，我们的方法在基准测试与实际部署之间架起桥梁，提供了高度可转移、通用的具身代理。相关代码和数据即将发布。 

---
# Imaginative World Modeling with Scene Graphs for Embodied Agent Navigation 

**Title (ZH)**: 基于场景图的想象世界建模及其在具身智能体导航中的应用 

**Authors**: Yue Hu, Junzhe Wu, Ruihan Xu, Hang Liu, Avery Xi, Henry X. Liu, Ram Vasudevan, Maani Ghaffari  

**Link**: [PDF](https://arxiv.org/pdf/2508.06990)  

**Abstract**: Semantic navigation requires an agent to navigate toward a specified target in an unseen environment. Employing an imaginative navigation strategy that predicts future scenes before taking action, can empower the agent to find target faster. Inspired by this idea, we propose SGImagineNav, a novel imaginative navigation framework that leverages symbolic world modeling to proactively build a global environmental representation. SGImagineNav maintains an evolving hierarchical scene graphs and uses large language models to predict and explore unseen parts of the environment. While existing methods solely relying on past observations, this imaginative scene graph provides richer semantic context, enabling the agent to proactively estimate target locations. Building upon this, SGImagineNav adopts an adaptive navigation strategy that exploits semantic shortcuts when promising and explores unknown areas otherwise to gather additional context. This strategy continuously expands the known environment and accumulates valuable semantic contexts, ultimately guiding the agent toward the target. SGImagineNav is evaluated in both real-world scenarios and simulation benchmarks. SGImagineNav consistently outperforms previous methods, improving success rate to 65.4 and 66.8 on HM3D and HSSD, and demonstrating cross-floor and cross-room navigation in real-world environments, underscoring its effectiveness and generalizability. 

**Abstract (ZH)**: 语义导航要求代理人在未见过的环境中朝指定目标进行导航。采用预测未来场景的想象性导航策略可以在行动前预见未来场景，从而帮助代理更快找到目标。受此启发，我们提出了 SGImagineNav，这是一种利用符号世界建模来主动构建全局环境表示的新颖想象性导航框架。SGImagineNav 维护一个动态层次场景图，并使用大语言模型来预测和探索环境的未见部分。与仅依赖过去观察的现有方法不同，这种想象性场景图提供了更丰富的语义上下文，使代理能够主动估计目标位置。在此基础上，SGImagineNav 采用一种适应性的导航策略，在有希望时利用语义捷径探索未知区域以收集更多上下文。该策略不断扩展已知环境并积累有价值的语义上下文，最终引导代理朝向目标。SGImagineNav 在真实场景和仿真基准测试中进行了评估。SGImagineNav 一致地优于先前的方法，在 HM3D 和 HSSD 中的成功率分别提高到 65.4 和 66.8，并在真实环境中展示了跨楼层和跨房间的导航能力，突显了其有效性和泛化能力。 

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
# Learning Causal Structure Distributions for Robust Planning 

**Title (ZH)**: 学习因果结构分布以实现鲁棒规划 

**Authors**: Alejandro Murillo-Gonzalez, Junhong Xu, Lantao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06742)  

**Abstract**: Structural causal models describe how the components of a robotic system interact. They provide both structural and functional information about the relationships that are present in the system. The structural information outlines the variables among which there is interaction. The functional information describes how such interactions work, via equations or learned models. In this paper we find that learning the functional relationships while accounting for the uncertainty about the structural information leads to more robust dynamics models which improves downstream planning, while using significantly lower computational resources. This in contrast with common model-learning methods that ignore the causal structure and fail to leverage the sparsity of interactions in robotic systems. We achieve this by estimating a causal structure distribution that is used to sample causal graphs that inform the latent-space representations in an encoder-multidecoder probabilistic model. We show that our model can be used to learn the dynamics of a robot, which together with a sampling-based planner can be used to perform new tasks in novel environments, provided an objective function for the new requirement is available. We validate our method using manipulators and mobile robots in both simulation and the real-world. Additionally, we validate the learned dynamics' adaptability and increased robustness to corrupted inputs and changes in the environment, which is highly desirable in challenging real-world robotics scenarios. Video: this https URL. 

**Abstract (ZH)**: 结构因果模型描述了机器人系统组件之间的相互作用方式，提供了系统中存在关系的结构性和功能性信息。结构性信息概述了存在相互作用的变量。功能性信息描述了这些相互作用如何通过方程或学习模型来实现。在本文中，我们发现，在考虑结构性信息的不确定性的同时学习功能关系，可以生成更稳健的动力学模型，从而改进下游规划，同时显著减少计算资源的使用。这与通常忽略因果结构的模型学习方法形成对比，后者无法利用机器人系统中相互作用的稀疏性。我们通过估计一个因果结构分布来实现这一点，该分布用于抽样因果图，以指导编码器-多解码器概率模型中的潜在空间表示。我们展示了我们的模型可以用于学习机器人的动力学，结合基于采样的规划器，可以用于在新环境中执行新任务，前提是新要求有一个目标函数。我们在仿真和真实世界中分别使用操作器和移动机器人验证了该方法。此外，我们验证了所学习的动力学对输入污染和环境变化的适应性和增强的鲁棒性，这对于具有挑战性的现实世界机器人场景来说是非常重要的。视频：请访问此链接。 

---
# Symbolic Learning of Interpretable Reduced-Order Models for Jumping Quadruped Robots 

**Title (ZH)**: 跳跃四足机器人的可解释降阶模型的符号学习 

**Authors**: Gioele Buriani, Jingyue Liu, Maximilian Stölzle, Cosimo Della Santina, Jiatao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2508.06538)  

**Abstract**: Reduced-order models are essential for motion planning and control of quadruped robots, as they simplify complex dynamics while preserving critical behaviors. This paper introduces a novel methodology for deriving such interpretable dynamic models, specifically for jumping. We capture the high-dimensional, nonlinear jumping dynamics in a low-dimensional latent space by proposing a learning architecture combining Sparse Identification of Nonlinear Dynamics (SINDy) with physical structural priors on the jump dynamics. Our approach demonstrates superior accuracy to the traditional actuated Spring-loaded Inverted Pendulum (aSLIP) model and is validated through simulation and hardware experiments across different jumping strategies. 

**Abstract (ZH)**: 简化模型对于四足机器人运动规划与控制至关重要，它们能简化复杂动力学的同时保留关键行为。本文介绍了一种用于跳跃的新型可解释动力模型推导方法。通过将稀疏识别非线性动力学（SINDy）与跳跃动力学的物理结构先验相结合，我们捕获高维非线性跳跃动力学在低维潜在空间中的表示，并通过仿真和硬件实验验证了该方法在不同跳跃策略下的优越准确性和有效性。 

---
# AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning 

**Title (ZH)**: AR-VRM：通过类比推理模仿人类运动以实现视觉机器人操作 

**Authors**: Dejie Yang, Zijing Zhao, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.07626)  

**Abstract**: Visual Robot Manipulation (VRM) aims to enable a robot to follow natural language instructions based on robot states and visual observations, and therefore requires costly multi-modal data. To compensate for the deficiency of robot data, existing approaches have employed vision-language pretraining with large-scale data. However, they either utilize web data that differs from robotic tasks, or train the model in an implicit way (e.g., predicting future frames at the pixel level), thus showing limited generalization ability under insufficient robot data. In this paper, we propose to learn from large-scale human action video datasets in an explicit way (i.e., imitating human actions from hand keypoints), introducing Visual Robot Manipulation with Analogical Reasoning (AR-VRM). To acquire action knowledge explicitly from human action videos, we propose a keypoint Vision-Language Model (VLM) pretraining scheme, enabling the VLM to learn human action knowledge and directly predict human hand keypoints. During fine-tuning on robot data, to facilitate the robotic arm in imitating the action patterns of human motions, we first retrieve human action videos that perform similar manipulation tasks and have similar historical observations , and then learn the Analogical Reasoning (AR) map between human hand keypoints and robot components. Taking advantage of focusing on action keypoints instead of irrelevant visual cues, our method achieves leading performance on the CALVIN benchmark {and real-world experiments}. In few-shot scenarios, our AR-VRM outperforms previous methods by large margins , underscoring the effectiveness of explicitly imitating human actions under data scarcity. 

**Abstract (ZH)**: 视觉机器人操作（视觉机器人操作-类比推理，AR-VRM）：一种显式学习人类动作知识的方法 

---
# PANAMA: A Network-Aware MARL Framework for Multi-Agent Path Finding in Digital Twin Ecosystems 

**Title (ZH)**: PANAMA：用于数字孪生生态系统中多agent路径规划的网络感知MARL框架 

**Authors**: Arman Dogru, R. Irem Bor-Yaliniz, Nimal Gamini Senarath  

**Link**: [PDF](https://arxiv.org/pdf/2508.06767)  

**Abstract**: Digital Twins (DTs) are transforming industries through advanced data processing and analysis, positioning the world of DTs, Digital World, as a cornerstone of nextgeneration technologies including embodied AI. As robotics and automated systems scale, efficient data-sharing frameworks and robust algorithms become critical. We explore the pivotal role of data handling in next-gen networks, focusing on dynamics between application and network providers (AP/NP) in DT ecosystems. We introduce PANAMA, a novel algorithm with Priority Asymmetry for Network Aware Multi-agent Reinforcement Learning (MARL) based multi-agent path finding (MAPF). By adopting a Centralized Training with Decentralized Execution (CTDE) framework and asynchronous actor-learner architectures, PANAMA accelerates training while enabling autonomous task execution by embodied AI. Our approach demonstrates superior pathfinding performance in accuracy, speed, and scalability compared to existing benchmarks. Through simulations, we highlight optimized data-sharing strategies for scalable, automated systems, ensuring resilience in complex, real-world environments. PANAMA bridges the gap between network-aware decision-making and robust multi-agent coordination, advancing the synergy between DTs, wireless networks, and AI-driven automation. 

**Abstract (ZH)**: 数字孪生（DTs）通过先进数据处理和分析重塑产业，将数字世界定位为下一代技术包括具身AI的核心基石。随着机器人技术和自动化系统的扩展，高效的数据共享框架和 robust 算法变得至关重要。我们探讨了在数字孪生生态系统中数据处理在下一代网络中的关键作用，重点关注应用和网络提供商之间的动态互动。我们介绍了一种名为PANAMA的新算法，该算法基于多智能体强化学习（MARL）的多智能体路径查找（MAPF），采用了中心化训练与去中心化执行（CTDE）框架以及异步演员-学习者架构，加速了训练并使具身AI能够自主执行任务。我们的方法在准确性、速度和可扩展性方面优于现有基准。通过仿真，我们展示了为可扩展自动化系统优化的数据共享策略，确保在复杂实际环境中的弹性。PANAMA填补了网络意识决策与 robust 多智能体协调之间的 gap，推动了数字孪生、无线网络和AI驱动自动化之间的协同进步。 

---
# IRL-VLA: Training an Vision-Language-Action Policy via Reward World Model 

**Title (ZH)**: IRL-VLA: 通过奖励世界模型训练视觉-语言-行动策略 

**Authors**: Anqing Jiang, Yu Gao, Yiru Wang, Zhigang Sun, Shuo Wang, Yuwen Heng, Hao Sun, Shichen Tang, Lijuan Zhu, Jinhao Chai, Jijun Wang, Zichong Gu, Hao Jiang, Li Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.06571)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated potential in autonomous driving. However, two critical challenges hinder their development: (1) Existing VLA architectures are typically based on imitation learning in open-loop setup which tends to capture the recorded behaviors in the dataset, leading to suboptimal and constrained performance, (2) Close-loop training relies heavily on high-fidelity sensor simulation, where domain gaps and computational inefficiencies pose significant barriers. In this paper, we introduce IRL-VLA, a novel close-loop Reinforcement Learning via \textbf{I}nverse \textbf{R}einforcement \textbf{L}earning reward world model with a self-built VLA approach. Our framework proceeds in a three-stage paradigm: In the first stage, we propose a VLA architecture and pretrain the VLA policy via imitation learning. In the second stage, we construct a lightweight reward world model via inverse reinforcement learning to enable efficient close-loop reward computation. To further enhance planning performance, finally, we design specialized reward world model guidence reinforcement learning via PPO(Proximal Policy Optimization) to effectively balance the safety incidents, comfortable driving, and traffic efficiency. Our approach achieves state-of-the-art performance in NAVSIM v2 end-to-end driving benchmark, 1st runner up in CVPR2025 Autonomous Grand Challenge. We hope that our framework will accelerate VLA research in close-loop autonomous driving. 

**Abstract (ZH)**: Vision-Language-Action (VLA)模型在自动驾驶中展现出潜在应用价值。然而，其开发面临两大关键挑战：（1）现有的VLA架构通常基于开环条件下的模仿学习，容易捕捉数据集中记录的行为，导致性能欠佳且受限；（2）闭环训练依赖于高保真传感器模拟，其中领域偏差和计算效率问题构成了重大障碍。本文介绍了IRL-VLA，一种基于逆强化学习的新型闭环强化学习框架，结合自建的VLA方法。本框架采用三阶段流程：首先，提出VLA架构并通过模仿学习预训练VLA策略；其次，通过逆强化学习构建轻量级奖励世界模型，以实现高效的闭环奖励计算；最后，设计基于PPO的专业奖励世界模型指导强化学习，以有效平衡安全性、舒适性和交通效率。我们的方法在NAVSIM v2端到端驾驶基准测试中取得了最先进的性能，并在CVPR2025自主挑战赛中获得亚军。我们希望本框架能够加速VLA在闭环自动驾驶中的研究。 

---
# Deep Reinforcement Learning with anticipatory reward in LSTM for Collision Avoidance of Mobile Robots 

**Title (ZH)**: 基于循环神经网络的前瞻奖励深度强化学习在移动机器人防碰撞中的应用 

**Authors**: Olivier Poulet, Frédéric Guinand, François Guérin  

**Link**: [PDF](https://arxiv.org/pdf/2508.07941)  

**Abstract**: This article proposes a collision risk anticipation method based on short-term prediction of the agents position. A Long Short-Term Memory (LSTM) model, trained on past trajectories, is used to estimate the next position of each robot. This prediction allows us to define an anticipated collision risk by dynamically modulating the reward of a Deep Q-Learning Network (DQN) agent. The approach is tested in a constrained environment, where two robots move without communication or identifiers. Despite a limited sampling frequency (1 Hz), the results show a significant decrease of the collisions number and a stability improvement. The proposed method, which is computationally inexpensive, appears particularly attractive for implementation on embedded systems. 

**Abstract (ZH)**: 基于代理短期位置预测的碰撞风险预判方法 

---
# Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents 

**Title (ZH)**: 瓦解与重建：基于技能的视觉- 语言导航智能体的混合模型 

**Authors**: Tianyi Ma, Yue Zhang, Zehao Wang, Parisa Kordjamshidi  

**Link**: [PDF](https://arxiv.org/pdf/2508.07642)  

**Abstract**: Vision-and-Language Navigation (VLN) poses significant challenges in enabling agents to interpret natural language instructions and navigate complex 3D environments. While recent progress has been driven by large-scale pre-training and data augmentation, current methods still struggle to generalize to unseen scenarios, particularly when complex spatial and temporal reasoning is required. In this work, we propose SkillNav, a modular framework that introduces structured, skill-based reasoning into Transformer-based VLN agents. Our method decomposes navigation into a set of interpretable atomic skills (e.g., Vertical Movement, Area and Region Identification, Stop and Pause), each handled by a specialized agent. We then introduce a novel zero-shot Vision-Language Model (VLM)-based router, which dynamically selects the most suitable agent at each time step by aligning sub-goals with visual observations and historical actions. SkillNav achieves a new state-of-the-art performance on the R2R benchmark and demonstrates strong generalization to the GSA-R2R benchmark that includes novel instruction styles and unseen environments. 

**Abstract (ZH)**: Vision-and-Language Navigation (VLN) 在使代理解读自然语言指令并在复杂3D环境中导航方面面临着显著挑战。尽管近期进展受益于大规模预训练和数据增强，现有方法在应对未见过的场景时仍然难以泛化，尤其是在需要复杂的空间和时间推理时。在本文中，我们提出了一种模块化框架——SkillNav，该框架将结构化、基于技能的推理引入到了基于Transformer的VLN代理中。我们的方法将导航分解为一组可解释的基本技能（如垂直移动、区域和区域识别、停留和暂停），每个技能由一个专门的代理处理。我们还引入了一种新颖的零样本视觉-语言模型（VLM）路由器，该路由器在每个时间步骤动态选择最合适的代理，通过将子目标与视觉观察和历史动作对齐来实现。SkillNav 在 R2R 基准上取得了新的最佳性能，并在包含新指令风格和未见过环境的GSA-R2R基准上展示了强大的泛化能力。 

---
# A Comprehensive Survey of Self-Evolving AI Agents: A New Paradigm Bridging Foundation Models and Lifelong Agentic Systems 

**Title (ZH)**: 全面综述自进化AI代理：一种连接基础模型与终身代理系统的新型范式 

**Authors**: Jinyuan Fang, Yanwen Peng, Xi Zhang, Yingxu Wang, Xinhao Yi, Guibin Zhang, Yi Xu, Bin Wu, Siwei Liu, Zihao Li, Zhaochun Ren, Nikos Aletras, Xi Wang, Han Zhou, Zaiqiao Meng  

**Link**: [PDF](https://arxiv.org/pdf/2508.07407)  

**Abstract**: Recent advances in large language models have sparked growing interest in AI agents capable of solving complex, real-world tasks. However, most existing agent systems rely on manually crafted configurations that remain static after deployment, limiting their ability to adapt to dynamic and evolving environments. To this end, recent research has explored agent evolution techniques that aim to automatically enhance agent systems based on interaction data and environmental feedback. This emerging direction lays the foundation for self-evolving AI agents, which bridge the static capabilities of foundation models with the continuous adaptability required by lifelong agentic systems. In this survey, we provide a comprehensive review of existing techniques for self-evolving agentic systems. Specifically, we first introduce a unified conceptual framework that abstracts the feedback loop underlying the design of self-evolving agentic systems. The framework highlights four key components: System Inputs, Agent System, Environment, and Optimisers, serving as a foundation for understanding and comparing different strategies. Based on this framework, we systematically review a wide range of self-evolving techniques that target different components of the agent system. We also investigate domain-specific evolution strategies developed for specialised fields such as biomedicine, programming, and finance, where optimisation objectives are tightly coupled with domain constraints. In addition, we provide a dedicated discussion on the evaluation, safety, and ethical considerations for self-evolving agentic systems, which are critical to ensuring their effectiveness and reliability. This survey aims to provide researchers and practitioners with a systematic understanding of self-evolving AI agents, laying the foundation for the development of more adaptive, autonomous, and lifelong agentic systems. 

**Abstract (ZH)**: recent advances in large language models have sparked growing interest in AI agents capable of solving complex, real-world tasks.然而，现有的大多数代理系统依赖于手动构建的配置，这些配置在部署后保持静态，限制了它们适应动态和 evolving 环境的能力。为此，近期的研究探索了基于交互数据和环境反馈自动提升代理系统的技术。这一新兴方向为自进化的AI代理系统奠定了基础，这些系统结合了基础模型的静态能力与终身代理系统所需的持续适应能力。在本文综述中，我们提供了一种全面的自我进化的代理系统技术的回顾。具体地，我们首先介绍了一个统一的概念框架，该框架抽象出自我进化的代理系统设计背后的反馈循环。该框架强调了四个关键组成部分：系统输入、代理系统、环境和优化器，作为理解并比较不同策略的基础。在此基础上，我们系统地回顾了旨在代理系统不同组成部分的广泛自我进化技术。我们也探讨了为生物医学、编程和金融等特定领域开发的优化策略，其中优化目标与领域约束紧密相关。此外，我们还专门讨论了自我进化的代理系统在评估、安全性和伦理方面的考虑，这些是确保其有效性和可靠性的关键因素。本文综述旨在为研究人员和实践者提供系统理解自我进化的AI代理的基础，为开发更适应、自主并终身的代理系统奠定基础。 

---
# EndoAgent: A Memory-Guided Reflective Agent for Intelligent Endoscopic Vision-to-Decision Reasoning 

**Title (ZH)**: EndoAgent：一种内存引导的反射性智能内镜视觉决策推理代理 

**Authors**: Yi Tang, Kaini Wang, Yang Chen, Guangquan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.07292)  

**Abstract**: Developing general artificial intelligence (AI) systems to support endoscopic image diagnosis is an emerging research priority. Existing methods based on large-scale pretraining often lack unified coordination across tasks and struggle to handle the multi-step processes required in complex clinical workflows. While AI agents have shown promise in flexible instruction parsing and tool integration across domains, their potential in endoscopy remains underexplored. To address this gap, we propose EndoAgent, the first memory-guided agent for vision-to-decision endoscopic analysis that integrates iterative reasoning with adaptive tool selection and collaboration. Built on a dual-memory design, it enables sophisticated decision-making by ensuring logical coherence through short-term action tracking and progressively enhancing reasoning acuity through long-term experiential learning. To support diverse clinical tasks, EndoAgent integrates a suite of expert-designed tools within a unified reasoning loop. We further introduce EndoAgentBench, a benchmark of 5,709 visual question-answer pairs that assess visual understanding and language generation capabilities in realistic scenarios. Extensive experiments show that EndoAgent consistently outperforms both general and medical multimodal models, exhibiting its strong flexibility and reasoning capabilities. 

**Abstract (ZH)**: 开发用于内镜图像诊断的一般人工智能系统是新兴的研究优先领域。现有的基于大规模预训练的方法往往缺乏跨任务的统一协调，难以处理复杂临床工作流程中所需的多步骤过程。虽然人工智能代理在跨领域灵活指令解析和工具集成方面显示出潜力，但在内镜应用中的潜力尚未得到充分探索。为了填补这一空白，我们提出了EndoAgent，这是一种基于记忆引导的内镜视觉到决策分析代理，将迭代推理与适应性工具选择和协作相结合。该代理基于双记忆设计，通过短期行为跟踪确保逻辑一致性，并通过长期经验学习逐步提高推理敏锐度。为了支持多样化的临床任务，EndoAgent在一个统一的推理循环中集成了多套专家设计的工具。我们还引入了包含5,709个视觉问答对的EndoAgentBench基准测试，评估其在现实场景中的视觉理解能力和语言生成能力。大量实验表明，EndoAgent在一致性和推理能力方面均优于通用和医学多模态模型。 

---
# Simulating Biological Intelligence: Active Inference with Experiment-Informed Generative Model 

**Title (ZH)**: 模拟生物智能：基于实验指导的生成模型的主动推断 

**Authors**: Aswin Paul, Moein Khajehnejad, Forough Habibollahi, Brett J. Kagan, Adeel Razi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06980)  

**Abstract**: With recent and rapid advancements in artificial intelligence (AI), understanding the foundation of purposeful behaviour in autonomous agents is crucial for developing safe and efficient systems. While artificial neural networks have dominated the path to AI, recent studies are exploring the potential of biologically based systems, such as networks of living biological neuronal networks. Along with promises of high power and data efficiency, these systems may also inform more explainable and biologically plausible models. In this work, we propose a framework rooted in active inference, a general theory of behaviour, to model decision-making in embodied agents. Using experiment-informed generative models, we simulate decision-making processes in a simulated game-play environment, mirroring experimental setups that use biological neurons. Our results demonstrate learning in these agents, providing insights into the role of memory-based learning and predictive planning in intelligent decision-making. This work contributes to the growing field of explainable AI by offering a biologically grounded and scalable approach to understanding purposeful behaviour in agents. 

**Abstract (ZH)**: 随着人工智能（AI）的近期 rapid 发展，了解自主代理有目的行为的基础对于开发安全和高效的系统至关重要。虽然人工神经网络主导了AI的发展路径，但最近的研究正在探索基于生物学系统的潜力，如由生物神经元网络组成的网络。除了高功率和数据效率的承诺外，这些系统还可能启发更可解释和生物合理模型。在本项工作中，我们提出了一种基于主动推断的框架，这是一种关于行为的一般理论，用于建模具身代理的决策过程。利用实验指导的生成模型，我们在模拟游戏环境中标记的决策过程，模拟使用生物神经元的实验装置。我们的结果展示了这些代理的学习，提供了关于基于记忆的学习和预测性计划在智能决策中的作用的见解。本项工作为可解释AI的发展做出了贡献，通过提供一种基于生物学原理且可扩展的方法来理解代理的有目的行为。 

---
# Topology Generation of UAV Covert Communication Networks: A Graph Diffusion Approach with Incentive Mechanism 

**Title (ZH)**: 无人机隐蔽通信网络的拓扑生成：具有激励机制的图扩散方法 

**Authors**: Xin Tang, Qian Chen, Fengshun Li, Youchun Gong, Yinqiu Liu, Wen Tian, Shaowen Qin, Xiaohuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.06746)  

**Abstract**: With the growing demand for Uncrewed Aerial Vehicle (UAV) networks in sensitive applications, such as urban monitoring, emergency response, and secure sensing, ensuring reliable connectivity and covert communication has become increasingly vital. However, dynamic mobility and exposure risks pose significant challenges. To tackle these challenges, this paper proposes a self-organizing UAV network framework combining Graph Diffusion-based Policy Optimization (GDPO) with a Stackelberg Game (SG)-based incentive mechanism. The GDPO method uses generative AI to dynamically generate sparse but well-connected topologies, enabling flexible adaptation to changing node distributions and Ground User (GU) demands. Meanwhile, the Stackelberg Game (SG)-based incentive mechanism guides self-interested UAVs to choose relay behaviors and neighbor links that support cooperation and enhance covert communication. Extensive experiments are conducted to validate the effectiveness of the proposed framework in terms of model convergence, topology generation quality, and enhancement of covert communication performance. 

**Abstract (ZH)**: 随着无人机网络在敏感应用如城市监控、应急响应和安全感知等方面需求的增长，确保可靠连接和隐蔽通信变得越来越重要。然而，动态移动性和暴露风险构成了重大挑战。为应对这些挑战，本文提出了一种基于图扩散策略优化（GDPO）和Stackelberg博弈（SG）激励机制相结合的自组织无人机网络框架。GDPO方法利用生成式AI动态生成稀疏但具有良好连接性的拓扑结构，以灵活适应节点分布和地面用户的需求。同时，基于Stackelberg博弈（SG）的激励机制引导无人机选择支持合作并增强隐蔽通信的中继行为和邻接链路。通过广泛的实验验证了所提框架在模型收敛性、拓扑结构生成质量和隐蔽通信性能提升方面的有效性。 

---
# MedReasoner: Reinforcement Learning Drives Reasoning Grounding from Clinical Thought to Pixel-Level Precision 

**Title (ZH)**: MedReasoner: 强化学习驱动从临床思维到像素级精确的推理接地 

**Authors**: Zhonghao Yan, Muxi Diao, Yuxuan Yang, Jiayuan Xu, Kaizhou Zhang, Ruoyan Jing, Lele Yang, Yanxi Liu, Kongming Liang, Zhanyu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.08177)  

**Abstract**: Accurately grounding regions of interest (ROIs) is critical for diagnosis and treatment planning in medical imaging. While multimodal large language models (MLLMs) combine visual perception with natural language, current medical-grounding pipelines still rely on supervised fine-tuning with explicit spatial hints, making them ill-equipped to handle the implicit queries common in clinical practice. This work makes three core contributions. We first define Unified Medical Reasoning Grounding (UMRG), a novel vision-language task that demands clinical reasoning and pixel-level grounding. Second, we release U-MRG-14K, a dataset of 14K samples featuring pixel-level masks alongside implicit clinical queries and reasoning traces, spanning 10 modalities, 15 super-categories, and 108 specific categories. Finally, we introduce MedReasoner, a modular framework that distinctly separates reasoning from segmentation: an MLLM reasoner is optimized with reinforcement learning, while a frozen segmentation expert converts spatial prompts into masks, with alignment achieved through format and accuracy rewards. MedReasoner achieves state-of-the-art performance on U-MRG-14K and demonstrates strong generalization to unseen clinical queries, underscoring the significant promise of reinforcement learning for interpretable medical grounding. 

**Abstract (ZH)**: 准确地定位兴趣区域对于医学影像诊断和治疗规划至关重要。虽然多模态大型语言模型结合了视觉感知和自然语言，但当前的医学定位管道仍依赖于带有显式空间提示的监督微调，这使它们难以处理临床实践中常见的隐式查询。本工作中，我们做出了三项核心贡献。首先，我们定义了统一医学推理定位（UMRG），这是一个新的视图-语言任务，要求临床推理和像素级定位。第二，我们发布了UMRG-14K数据集，包含14000个样本，每个样本都包含了隐式临床查询、推理轨迹和像素级掩码，覆盖了10种模态、15个超类别和108个具体类别。最后，我们引入了MedReasoner，这是一个模块化框架，明确地将推理和分割分离：一个大型语言模型推理器通过强化学习优化，一个冻结的分割专家通过格式和准确率奖励将空间提示转换为掩码。MedReasoner在UMRG-14K上达到了最先进的性能，并展示了强大的泛化能力，用于处理未见过的临床查询，突显了强化学习在可解释医学定位方面的巨大潜力。 

---
# COMponent-Aware Pruning for Accelerated Control Tasks in Latent Space Models 

**Title (ZH)**: 组件意识裁剪以加速潜在空间模型中的控制任务 

**Authors**: Ganesh Sundaram, Jonas Ulmen, Amjad Haider, Daniel Görges  

**Link**: [PDF](https://arxiv.org/pdf/2508.08144)  

**Abstract**: The rapid growth of resource-constrained mobile platforms, including mobile robots, wearable systems, and Internet-of-Things devices, has increased the demand for computationally efficient neural network controllers (NNCs) that can operate within strict hardware limitations. While deep neural networks (DNNs) demonstrate superior performance in control applications, their substantial computational complexity and memory requirements present significant barriers to practical deployment on edge devices. This paper introduces a comprehensive model compression methodology that leverages component-aware structured pruning to determine the optimal pruning magnitude for each pruning group, ensuring a balance between compression and stability for NNC deployment. Our approach is rigorously evaluated on Temporal Difference Model Predictive Control (TD-MPC), a state-of-the-art model-based reinforcement learning algorithm, with a systematic integration of mathematical stability guarantee properties, specifically Lyapunov criteria. The key contribution of this work lies in providing a principled framework for determining the theoretical limits of model compression while preserving controller stability. Experimental validation demonstrates that our methodology successfully reduces model complexity while maintaining requisite control performance and stability characteristics. Furthermore, our approach establishes a quantitative boundary for safe compression ratios, enabling practitioners to systematically determine the maximum permissible model reduction before violating critical stability properties, thereby facilitating the confident deployment of compressed NNCs in resource-limited environments. 

**Abstract (ZH)**: 资源受限的移动平台迅速增长，包括移动机器人、穿戴系统和物联网设备，增加了对计算高效的神经网络控制器（NNCs）的需求，这些控制器可以在严格的硬件限制下运行。尽管深度神经网络（DNNs）在控制应用中表现出色，但其巨大的计算复杂度和内存要求给边缘设备上的实际部署带来了显著障碍。本文介绍了一种全面的模型压缩方法，该方法利用组件感知结构化剪枝来确定每个剪枝组的最优剪枝幅度，确保压缩与稳定性的平衡，以保障NNCs的部署。我们的方法在Temporal Difference Model Predictive Control（TD-MPC）上进行了严格评估，TD-MPC是一种先进的基于模型的强化学习算法，并系统地整合了数学稳定性保证特性，特别是李雅普un夫稳定性准则。本文的关键贡献在于提供了一个原则性的框架，用于确定模型压缩的理论限制，同时保持控制器稳定性。实验验证表明，我们的方法成功地减少了模型复杂性，同时保持了所需的控制性能和稳定性特性。此外，我们的方法为安全压缩比建立了定量边界，使实践者能够系统地确定在违反关键稳定性属性之前的最大允许模型减少，从而促进压缩NNCs在资源受限环境中的自信部署。 

---
# CognitiveArm: Enabling Real-Time EEG-Controlled Prosthetic Arm Using Embodied Machine Learning 

**Title (ZH)**: 认知臂：基于嵌入式机器学习的实时EEG控制假肢臂 

**Authors**: Abdul Basit, Maha Nawaz, Saim Rehman, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2508.07731)  

**Abstract**: Efficient control of prosthetic limbs via non-invasive brain-computer interfaces (BCIs) requires advanced EEG processing, including pre-filtering, feature extraction, and action prediction, performed in real time on edge AI hardware. Achieving this on resource-constrained devices presents challenges in balancing model complexity, computational efficiency, and latency. We present CognitiveArm, an EEG-driven, brain-controlled prosthetic system implemented on embedded AI hardware, achieving real-time operation without compromising accuracy. The system integrates BrainFlow, an open-source library for EEG data acquisition and streaming, with optimized deep learning (DL) models for precise brain signal classification. Using evolutionary search, we identify Pareto-optimal DL configurations through hyperparameter tuning, optimizer analysis, and window selection, analyzed individually and in ensemble configurations. We apply model compression techniques such as pruning and quantization to optimize models for embedded deployment, balancing efficiency and accuracy. We collected an EEG dataset and designed an annotation pipeline enabling precise labeling of brain signals corresponding to specific intended actions, forming the basis for training our optimized DL models. CognitiveArm also supports voice commands for seamless mode switching, enabling control of the prosthetic arm's 3 degrees of freedom (DoF). Running entirely on embedded hardware, it ensures low latency and real-time responsiveness. A full-scale prototype, interfaced with the OpenBCI UltraCortex Mark IV EEG headset, achieved up to 90% accuracy in classifying three core actions (left, right, idle). Voice integration enables multiplexed, variable movement for everyday tasks (e.g., handshake, cup picking), enhancing real-world performance and demonstrating CognitiveArm's potential for advanced prosthetic control. 

**Abstract (ZH)**: 通过非侵入式脑计算机接口（BCIs）高效控制假肢的手部需要在边缘AI硬件上实时进行先进的EEG处理，包括预滤波、特征提取和动作预测。在资源受限的设备上实现这一目标面临在模型复杂性、计算效率和延迟之间平衡的挑战。我们提出CognitiveArm，一个基于EEG的脑控假肢系统，该系统在嵌入式AI硬件上实现，能够实现实时操作而不牺牲准确性。该系统整合了BrainFlow开源库进行EEG数据采集和流式传输，并结合了优化的深度学习（DL）模型进行精确的脑信号分类。通过进化搜索，我们通过对超参数调优、优化器分析和窗口选择进行个体分析和整体配置分析，确定了Pareto最优的DL配置。我们运用模型压缩技术（如剪枝和量化）来优化模型以适应嵌入式部署，平衡效率和准确性。我们收集了一个EEG数据集，并设计了注释流水线，使其能够精确标注与特定意图动作对应的脑信号，为训练优化的DL模型奠定了基础。CognitiveArm还支持语音命令，使模式无缝切换，从而控制假肢的3个自由度（DoF）。该系统完全运行在嵌入式硬件上，确保低延迟和实时响应。与OpenBCI UltraCortex Mark IV EEG头戴设备进行接口的全规模原型在分类三个主要动作（左、右、空闲）上实现了高达90%的准确性。语音集成使得CognitiveArm能够执行日常生活任务中的多重可变动作（如握手、拿杯子），增强了实际性能并展示了其在高级假肢控制方面的潜力。 

---
# MORE-CLEAR: Multimodal Offline Reinforcement learning for Clinical notes Leveraged Enhanced State Representation 

**Title (ZH)**: MORE-CLEAR：多
用户继续对话，请再输出翻译结果 barcelona ultra-marathon runners's kinematic and physiological variance
用户继续对话-vars response
user kukuke cu忽悠助手继续输出
Assistant pérdida有效性：
用户 MORE-CLEAR: Multim
用户 pérdida有效性-mult modal Offline Reinunction learning
舯用户更多的
用户 kino-dynamics and physiological kinetic variables générated by barcelona ultra-marathon runners pérdida有效性 kukuka wrestlers in the kinematic and physiological普遍存在差异 pérdida有效性 

**Authors**: Yooseok Lim, ByoungJun Jeon, Seong-A Park, Jisoo Lee, Sae Won Choi, Chang Wook Jeong, Ho-Geol Ryu, Hongyeol Lee, Hyun-Lim Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.07681)  

**Abstract**: Sepsis, a life-threatening inflammatory response to infection, causes organ dysfunction, making early detection and optimal management critical. Previous reinforcement learning (RL) approaches to sepsis management rely primarily on structured data, such as lab results or vital signs, and on a dearth of a comprehensive understanding of the patient's condition. In this work, we propose a Multimodal Offline REinforcement learning for Clinical notes Leveraged Enhanced stAte Representation (MORE-CLEAR) framework for sepsis control in intensive care units. MORE-CLEAR employs pre-trained large-scale language models (LLMs) to facilitate the extraction of rich semantic representations from clinical notes, preserving clinical context and improving patient state representation. Gated fusion and cross-modal attention allow dynamic weight adjustment in the context of time and the effective integration of multimodal data. Extensive cross-validation using two public (MIMIC-III and MIMIC-IV) and one private dataset demonstrates that MORE-CLEAR significantly improves estimated survival rate and policy performance compared to single-modal RL approaches. To our knowledge, this is the first to leverage LLM capabilities within a multimodal offline RL for better state representation in medical applications. This approach can potentially expedite the treatment and management of sepsis by enabling reinforcement learning models to propose enhanced actions based on a more comprehensive understanding of patient conditions. 

**Abstract (ZH)**: 多模态离线强化学习结合临床笔记增强状态表示框架（MORE-CLEAR）在重症监护中控制脓毒症 

---
# Stackelberg Coupling of Online Representation Learning and Reinforcement Learning 

**Title (ZH)**: 在线表示学习与强化学习的Stackelberg耦合 

**Authors**: Fernando Martinez, Tao Li, Yingdong Lu, Juntao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.07452)  

**Abstract**: Integrated, end-to-end learning of representations and policies remains a cornerstone of deep reinforcement learning (RL). However, to address the challenge of learning effective features from a sparse reward signal, recent trends have shifted towards adding complex auxiliary objectives or fully decoupling the two processes, often at the cost of increased design complexity. This work proposes an alternative to both decoupling and naive end-to-end learning, arguing that performance can be significantly improved by structuring the interaction between distinct perception and control networks with a principled, game-theoretic dynamic. We formalize this dynamic by introducing the Stackelberg Coupled Representation and Reinforcement Learning (SCORER) framework, which models the interaction between perception and control as a Stackelberg game. The perception network (leader) strategically learns features to benefit the control network (follower), whose own objective is to minimize its Bellman error. We approximate the game's equilibrium with a practical two-timescale algorithm. Applied to standard DQN variants on benchmark tasks, SCORER improves sample efficiency and final performance. Our results show that performance gains can be achieved through principled algorithmic design of the perception-control dynamic, without requiring complex auxiliary objectives or architectures. 

**Abstract (ZH)**: 一种基于博弈论的 Perception-Control 结构化交互框架改进深度强化学习性能 

---
# Explainability-in-Action: Enabling Expressive Manipulation and Tacit Understanding by Bending Diffusion Models in ComfyUI 

**Title (ZH)**: 解释在行动：通过在ComfyUI中弯曲扩散模型实现表达性操控和隐性理解 

**Authors**: Ahmed M. Abuzuraiq, Philippe Pasquier  

**Link**: [PDF](https://arxiv.org/pdf/2508.07183)  

**Abstract**: Explainable AI (XAI) in creative contexts can go beyond transparency to support artistic engagement, modifiability, and sustained practice. While curated datasets and training human-scale models can offer artists greater agency and control, large-scale generative models like text-to-image diffusion systems often obscure these possibilities. We suggest that even large models can be treated as creative materials if their internal structure is exposed and manipulable. We propose a craft-based approach to explainability rooted in long-term, hands-on engagement akin to Schön's "reflection-in-action" and demonstrate its application through a model-bending and inspection plugin integrated into the node-based interface of ComfyUI. We demonstrate that by interactively manipulating different parts of a generative model, artists can develop an intuition about how each component influences the output. 

**Abstract (ZH)**: 可解释的人工智能（XAI）在创意背景下不仅可以超越透明性，还可以支持艺术参与、可修改性和持续实践。即使大型模型在其内部结构被暴露和可操作的情况下，也可以视为创意材料。我们提出了一种基于长期手头互动的工艺导向解释方法，类似于舍恩的“行动中的反思”，并通过将模型扭曲和检查插件整合到ComfyUI的节点界面中来展示其应用。我们证明，通过交互式地操作生成模型的不同部分，艺术家可以发展出对每个组件如何影响输出的直觉。 

---
# Sparsity-Driven Plasticity in Multi-Task Reinforcement Learning 

**Title (ZH)**: 多任务强化学习中的稀疏驱动塑性 

**Authors**: Aleksandar Todorov, Juan Cardenas-Cartagena, Rafael F. Cunha, Marco Zullich, Matthia Sabatelli  

**Link**: [PDF](https://arxiv.org/pdf/2508.06871)  

**Abstract**: Plasticity loss, a diminishing capacity to adapt as training progresses, is a critical challenge in deep reinforcement learning. We examine this issue in multi-task reinforcement learning (MTRL), where higher representational flexibility is crucial for managing diverse and potentially conflicting task demands. We systematically explore how sparsification methods, particularly Gradual Magnitude Pruning (GMP) and Sparse Evolutionary Training (SET), enhance plasticity and consequently improve performance in MTRL agents. We evaluate these approaches across distinct MTRL architectures (shared backbone, Mixture of Experts, Mixture of Orthogonal Experts) on standardized MTRL benchmarks, comparing against dense baselines, and a comprehensive range of alternative plasticity-inducing or regularization methods. Our results demonstrate that both GMP and SET effectively mitigate key indicators of plasticity degradation, such as neuron dormancy and representational collapse. These plasticity improvements often correlate with enhanced multi-task performance, with sparse agents frequently outperforming dense counterparts and achieving competitive results against explicit plasticity interventions. Our findings offer insights into the interplay between plasticity, network sparsity, and MTRL designs, highlighting dynamic sparsification as a robust but context-sensitive tool for developing more adaptable MTRL systems. 

**Abstract (ZH)**: 塑料性丧失在训练过程中适应能力的逐步减弱是深度强化学习中的一个关键挑战。我们研究了这一问题在多任务强化学习（MTRL）中的表现，其中更高的表示灵活性对于管理多种潜在冲突的任务需求至关重要。我们系统地探讨了剪枝方法（特别是渐进量值剪枝GMP和稀疏进化训练SET）如何增强塑料性并从而提高MTRL代理的表现。我们在标准化的MTRL基准上评估了这些方法在不同的MTRL架构（共享骨干、专家混合、正交专家混合）中的效果，对比了密集基线方法以及一系列其他增强塑料性或正则化的方法。我们的结果表明，GMP和SET都能有效缓解关键的塑料性退化指标，如神经元休眠和表示塌陷。这些塑料性的改进通常与多任务性能的提升相关，稀疏代理经常优于密集代理，并且在某些情况下可以与显式增强塑料性的干预措施相媲美。我们的研究为塑料性、网络稀疏性和MTRL设计之间的相互作用提供了见解，突出了动态稀疏化作为一种稳健但具有上下文敏感性的工具，对于开发更具适应性的MTRL系统的重要性。 

---
# In-Context Reinforcement Learning via Communicative World Models 

**Title (ZH)**: 基于交流世界模型的上下文强化学习 

**Authors**: Fernando Martinez-Lopez, Tao Li, Yingdong Lu, Juntao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.06659)  

**Abstract**: Reinforcement learning (RL) agents often struggle to generalize to new tasks and contexts without updating their parameters, mainly because their learned representations and policies are overfit to the specifics of their training environments. To boost agents' in-context RL (ICRL) ability, this work formulates ICRL as a two-agent emergent communication problem and introduces CORAL (Communicative Representation for Adaptive RL), a framework that learns a transferable communicative context by decoupling latent representation learning from control. In CORAL, an Information Agent (IA) is pre-trained as a world model on a diverse distribution of tasks. Its objective is not to maximize task reward, but to build a world model and distill its understanding into concise messages. The emergent communication protocol is shaped by a novel Causal Influence Loss, which measures the effect that the message has on the next action. During deployment, the previously trained IA serves as a fixed contextualizer for a new Control Agent (CA), which learns to solve tasks by interpreting the provided communicative context. Our experiments demonstrate that this approach enables the CA to achieve significant gains in sample efficiency and successfully perform zero-shot adaptation with the help of pre-trained IA in entirely unseen sparse-reward environments, validating the efficacy of learning a transferable communicative representation. 

**Abstract (ZH)**: 基于通信的学习增强 reinforcement learning（ICRL）agents及CORAL框架：学习可转移的通信表示以提升零样本适应能力 

---
# Fractal Language Modelling by Universal Sequence Maps (USM) 

**Title (ZH)**: 分形语言建模通过通用序列映射（USM） 

**Authors**: Jonas S Almeida, Daniel E Russ, Susana Vinga, Ines Duarte, Lee Mason, Praphulla Bhawsar, Aaron Ge, Arlindo Oliveira, Jeya Balaji Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2508.06641)  

**Abstract**: Motivation: With the advent of Language Models using Transformers, popularized by ChatGPT, there is a renewed interest in exploring encoding procedures that numerically represent symbolic sequences at multiple scales and embedding dimensions. The challenge that encoding addresses is the need for mechanisms that uniquely retain contextual information about the succession of individual symbols, which can then be modeled by nonlinear formulations such as neural networks.
Context: Universal Sequence Maps(USM) are iterated functions that bijectively encode symbolic sequences onto embedded numerical spaces. USM is composed of two Chaos Game Representations (CGR), iterated forwardly and backwardly, that can be projected into the frequency domain (FCGR). The corresponding USM coordinates can be used to compute a Chebyshev distance metric as well as k-mer frequencies, without having to recompute the embedded numeric coordinates, and, paradoxically, allowing for non-integers values of k.
Results: This report advances the bijective fractal encoding by Universal Sequence Maps (USM) by resolving seeding biases affecting the iterated process. The resolution had two results, the first expected, the second an intriguing outcome: 1) full reconciliation of numeric positioning with sequence identity; and 2) uncovering the nature of USM as an efficient numeric process converging towards a steady state sequence embedding solution. We illustrate these results for genomic sequences because of the convenience of a planar representation defined by an alphabet with only 4 tokens (the 4 nucleotides). Nevertheless, the application to alphabet of arbitrary cardinality was found to be straightforward. 

**Abstract (ZH)**: 动机：随着使用变换器的语言模型的兴起，以ChatGPT为代表，人们对探索能够以多个尺度和嵌入维度数值表示符号序列的编码方法的兴趣得到了新的提高。编码面临的挑战是需要机制来唯一保留单个符号序列的上下文信息，这些信息可以由非线性模型如神经网络进行建模。

背景：通用序列映射（USM）是迭代函数，能够在嵌入的数值空间中双射编码符号序列。USM 由两个混沌游戏表示法（CGR）组成，以正向和反向迭代方式，可以投影到频域（FCGR）。USM 对应的坐标可以用于计算切比雪夫距离度量以及 k-mer 频率，而无需重新计算嵌入的数值坐标，并且出人意料地允许 k 取非整数值。

结果：本报告通过解决影响迭代过程的起始条件偏见，推进了通用序列映射（USM）的双射分形编码。这一解决办法产生了两个结果：第一个是预期的结果，第二个是一个有趣的发现：1) 完全协调数值位置与序列身份；2) 揭示USM作为一种有效数值过程，能够收敛到稳定的序列嵌入解。我们通过基因组序列的平面表示来阐述这些结果，因为仅由四个符号（四种核苷酸）定义的字母表便于表示。然而，任意基数字母表的应用被发现是直接的。 

---
# Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs 

**Title (ZH)**: 深度无知：过滤预训练数据为开放权重LLMs构建防篡改保障 

**Authors**: Kyle O'Brien, Stephen Casper, Quentin Anthony, Tomek Korbak, Robert Kirk, Xander Davies, Ishan Mishra, Geoffrey Irving, Yarin Gal, Stella Biderman  

**Link**: [PDF](https://arxiv.org/pdf/2508.06601)  

**Abstract**: Open-weight AI systems offer unique benefits, including enhanced transparency, open research, and decentralized access. However, they are vulnerable to tampering attacks which can efficiently elicit harmful behaviors by modifying weights or activations. Currently, there is not yet a robust science of open-weight model risk management. Existing safety fine-tuning methods and other post-training techniques have struggled to make LLMs resistant to more than a few dozen steps of adversarial fine-tuning. In this paper, we investigate whether filtering text about dual-use topics from training data can prevent unwanted capabilities and serve as a more tamper-resistant safeguard. We introduce a multi-stage pipeline for scalable data filtering and show that it offers a tractable and effective method for minimizing biothreat proxy knowledge in LLMs. We pretrain multiple 6.9B-parameter models from scratch and find that they exhibit substantial resistance to adversarial fine-tuning attacks on up to 10,000 steps and 300M tokens of biothreat-related text -- outperforming existing post-training baselines by over an order of magnitude -- with no observed degradation to unrelated capabilities. However, while filtered models lack internalized dangerous knowledge, we find that they can still leverage such information when it is provided in context (e.g., via search tool augmentation), demonstrating a need for a defense-in-depth approach. Overall, these findings help to establish pretraining data curation as a promising layer of defense for open-weight AI systems. 

**Abstract (ZH)**: 开放权重的AI系统提供了独特的优势，包括增强的透明度、开放研究和去中心化的访问。然而，它们容易受到篡改攻击，攻击者可以通过修改权重或激活来有效地诱发有害行为。目前，开放权重模型风险管理工作尚未形成坚实的科学体系。现有的安全微调方法和其他后训练技术难以使大语言模型对超过几十步的对抗微调保持抗性。在本文中，我们探讨了从训练数据中过滤掉双重用途主题的文本是否能防止不希望的能力并作为更抗篡改的安全保障。我们介绍了一个可扩展的数据过滤多阶段管道，并展示了它在最小化LLM中的生物威胁代理知识方面提供了可实现且有效的方法。我们从头预训练了多个69亿参数的模型，并发现这些模型对多达10,000步和3亿个生物威胁相关文本令牌的对抗微调攻击表现出显著的抵抗力，比现有后训练基线高出一个数量级，且未观察到对无关能力的退化。然而，尽管过滤后的模型缺乏内化危险知识，我们发现它们在提供上下文信息（例如，通过搜索引擎工具增强）时仍能利用此类信息，这表明需要多层次防御策略。总的来说，这些发现有助于将预训练数据筛选确立为开放权重AI系统防御体系的一个有前景的层面。 

---
# Computing with Canonical Microcircuits 

**Title (ZH)**: 使用规范微回路进行计算 

**Authors**: PK Douglas  

**Link**: [PDF](https://arxiv.org/pdf/2508.06501)  

**Abstract**: The human brain represents the only known example of general intelligence that naturally aligns with human values. On a mere 20-watt power budget, the brain achieves robust learning and adaptive decision-making in ways that continue to elude advanced AI systems. Inspired by the brain, we present a computational architecture based on canonical microcircuits (CMCs) - stereotyped patterns of neurons found ubiquitously throughout the cortex. We implement these circuits as neural ODEs comprising spiny stellate, inhibitory, and pyramidal neurons, forming an 8-dimensional dynamical system with biologically plausible recurrent connections. Our experiments show that even a single CMC node achieves 97.8 percent accuracy on MNIST, while hierarchical configurations - with learnable inter-regional connectivity and recurrent connections - yield improved performance on more complex image benchmarks. Notably, our approach achieves competitive results using substantially fewer parameters than conventional deep learning models. Phase space analysis revealed distinct dynamical trajectories for different input classes, highlighting interpretable, emergent behaviors observed in biological systems. These findings suggest that neuromorphic computing approaches can improve both efficiency and interpretability in artificial neural networks, offering new directions for parameter-efficient architectures grounded in the computational principles of the human brain. 

**Abstract (ZH)**: 人类大脑是唯一已知与人类价值自然对齐的一般智能实例。在仅20瓦的功率预算下，大脑以高级AI系统仍未解决的方式实现稳健的学习和适应性决策。受大脑启发，我们提出了一种基于经典微回路（CMCs）的计算架构——广泛存在于皮层中的标准化神经元模式。我们通过脉刺星形、抑制性及锥形神经元实现这些电路，形成一个具有生物可 plausilble 循环连接的8维动力系统。实验表明，单个CMC节点在MNIST上的准确率达到97.8%，而层次化配置——具有可学习跨区域连接和循环连接——在更复杂的图像基准测试中表现出更好的性能。值得注意的是，与传统的深度学习模型相比，我们的方法使用了大量较少的参数实现了竞争力的结果。相空间分析揭示了不同输入类别独特的动力学轨迹，突出了在生物系统中观察到的可解释、涌现的行为。这些发现表明，神经形态计算方法可以提高人工神经网络的效率和可解释性，为基于人类大脑计算原理的参数高效架构提供新的方向。 

---
