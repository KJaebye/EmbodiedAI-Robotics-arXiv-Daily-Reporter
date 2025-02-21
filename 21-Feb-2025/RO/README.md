# VB-Com: Learning Vision-Blind Composite Humanoid Locomotion Against Deficient Perception 

**Title (ZH)**: 视觉盲复合 humanoid 行走学习: 对抗感知不足的学习方法 

**Authors**: Junli Ren, Tao Huang, Huayi Wang, Zirui Wang, Qingwei Ben, Jiangmiao Pang, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.14814)  

**Abstract**: The performance of legged locomotion is closely tied to the accuracy and comprehensiveness of state observations. Blind policies, which rely solely on proprioception, are considered highly robust due to the reliability of proprioceptive observations. However, these policies significantly limit locomotion speed and often require collisions with the terrain to adapt. In contrast, Vision policies allows the robot to plan motions in advance and respond proactively to unstructured terrains with an online perception module. However, perception is often compromised by noisy real-world environments, potential sensor failures, and the limitations of current simulations in presenting dynamic or deformable terrains. Humanoid robots, with high degrees of freedom and inherently unstable morphology, are particularly susceptible to misguidance from deficient perception, which can result in falls or termination on challenging dynamic terrains. To leverage the advantages of both vision and blind policies, we propose VB-Com, a composite framework that enables humanoid robots to determine when to rely on the vision policy and when to switch to the blind policy under perceptual deficiency. We demonstrate that VB-Com effectively enables humanoid robots to traverse challenging terrains and obstacles despite perception deficiencies caused by dynamic terrains or perceptual noise. 

**Abstract (ZH)**: 基于视觉和盲政策的复合框架（VB-Com）：在感知不足时使类人机器人有效穿越复杂地形和障碍 

---
# Planning, scheduling, and execution on the Moon: the CADRE technology demonstration mission 

**Title (ZH)**: 月球上的规划、排程与执行：CADRE技术演示任务 

**Authors**: Gregg Rabideau, Joseph Russino, Andrew Branch, Nihal Dhamani, Tiago Stegun Vaquero, Steve Chien, Jean-Pierre de la Croix, Federico Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14803)  

**Abstract**: NASA's Cooperative Autonomous Distributed Robotic Exploration (CADRE) mission, slated for flight to the Moon's Reiner Gamma region in 2025/2026, is designed to demonstrate multi-agent autonomous exploration of the Lunar surface and sub-surface. A team of three robots and a base station will autonomously explore a region near the lander, collecting the data required for 3D reconstruction of the surface with no human input; and then autonomously perform distributed sensing with multi-static ground penetrating radars (GPR), driving in formation while performing coordinated radar soundings to create a map of the subsurface. At the core of CADRE's software architecture is a novel autonomous, distributed planning, scheduling, and execution (PS&E) system. The system coordinates the robots' activities, planning and executing tasks that require multiple robots' participation while ensuring that each individual robot's thermal and power resources stay within prescribed bounds, and respecting ground-prescribed sleep-wake cycles. The system uses a centralized-planning, distributed-execution paradigm, and a leader election mechanism ensures robustness to failures of individual agents. In this paper, we describe the architecture of CADRE's PS&E system; discuss its design rationale; and report on verification and validation (V&V) testing of the system on CADRE's hardware in preparation for deployment on the Moon. 

**Abstract (ZH)**: NASA的CADRE（协作自主分布式机器人勘探）月球任务（计划于2025/2026年发射至雷inerγ区域）旨在演示多智能体自主月球表面及亚表面探索。该任务将由三台机器人及一个基站自主探索着陆器附近的区域，收集用于3D重建表面所需的数据，并自主执行多静态地面穿透雷达分布式传感任务，在行进成队形的同时进行协同雷达探测，以创建亚表面地图。CADRE软件架构的核心是一个新颖的自主分布式规划、调度与执行（PS&E）系统。该系统协调机器人的活动，规划和执行需要多机器人参与的任务，同时确保每个机器人自身的热能和电力资源保持在限定范围内，并遵守指定的休眠-唤醒周期。系统采用集中规划、分散执行的范式，并通过领导者选举机制确保对个体智能体故障的鲁棒性。本文描述了CADRE PS&E系统的架构；讨论其设计原理；并报告了在月球部署之前于CADRE硬件上进行的验证与验证（V&V）测试。 

---
# Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration 

**Title (ZH)**: 类人视觉集成控制：通往通用类人控制的途径 

**Authors**: Pengxiang Ding, Jianfei Ma, Xinyang Tong, Binghong Zou, Xinxin Luo, Yiguo Fan, Ting Wang, Hongchao Lu, Panzhong Mo, Jinxin Liu, Yuefan Wang, Huaicheng Zhou, Wenshuo Feng, Jiacheng Liu, Siteng Huang, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14795)  

**Abstract**: This paper addresses the limitations of current humanoid robot control frameworks, which primarily rely on reactive mechanisms and lack autonomous interaction capabilities due to data scarcity. We propose Humanoid-VLA, a novel framework that integrates language understanding, egocentric scene perception, and motion control, enabling universal humanoid control. Humanoid-VLA begins with language-motion pre-alignment using non-egocentric human motion datasets paired with textual descriptions, allowing the model to learn universal motion patterns and action semantics. We then incorporate egocentric visual context through a parameter efficient video-conditioned fine-tuning, enabling context-aware motion generation. Furthermore, we introduce a self-supervised data augmentation strategy that automatically generates pseudoannotations directly derived from motion data. This process converts raw motion sequences into informative question-answer pairs, facilitating the effective use of large-scale unlabeled video data. Built upon whole-body control architectures, extensive experiments show that Humanoid-VLA achieves object interaction and environment exploration tasks with enhanced contextual awareness, demonstrating a more human-like capacity for adaptive and intelligent engagement. 

**Abstract (ZH)**: 本文针对当前类人机器人控制框架主要依赖于反应机制且因数据稀缺而缺乏自主交互能力的局限性，提出了一种新型框架Humanoid-VLA，该框架整合了语言理解、第一人称场景感知和运动控制，实现通用类人控制。Humanoid-VLA 通过使用非第一人称的人类运动数据集与文本描述配对，进行语言-运动前期对齐，从而使模型学习到通用的运动模式和动作语义。我们随后通过参数高效的视频条件微调引入第一人称视觉上下文，使运动生成具备上下文感知能力。此外，我们引入了一种自监督数据增强策略，自动从运动数据中生成伪标注，将原始运动序列转换为信息丰富的问答对，便于有效利用大规模的未标注视频数据。基于全身控制架构，广泛实验表明，Humanoid-VLA 在对象交互和环境探索任务中表现出增强的上下文感知能力，展示出更强的适应性和智能交互能力。 

---
# Real-world Troublemaker: A Novel Track Testing Framework for Automated Driving Systems in Safety-critical Interaction Scenarios 

**Title (ZH)**: 现实世界中的麻烦制造者：一种新型 Tracks 测试框架，用于安全关键交互场景中的自动驾驶系统评估 

**Authors**: Xinrui Zhang, Lu Xiong, Peizhi Zhang, Junpeng Huang, Yining Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.14574)  

**Abstract**: Track testing plays a critical role in the safety evaluation of autonomous driving systems (ADS), as it provides real-world object targets and a safety-controllable interaction environment. However, existing track testing scenarios are often pre-fixed and limited, primarily due to the inflexibility of object target control methods and the lack of intelligent interactive behaviors. To overcome this limitation, we propose a novel track testing framework, Real-world Troublemaker, which can generate adversarial object target motion trajectories and facilitate intelligent interactions with the vehicle under test (VUT), creating a more realistic and dynamic testing environment. To enable flexible motion trajectories, cloud-controlled technology is utilized to remotely and dynamically control object targets to create a realistic traffic environment. To achieve intelligent interactions, an interactive concrete scenario generation method is introduced within a game-theoretic structure. The proposed framework has been successfully implemented at the Tongji University Intelligent Connected Vehicle Evaluation Base. Field test results demonstrate that Troublemaker can perform dynamic interactive testing of ADS accurately and effectively. Compared to traditional track testing methods, Troublemaker improves scenario reproduction accuracy by 65.2\%, increases the diversity of target vehicle interaction strategies by approximately 9.2 times, and enhances exposure frequency of safety-critical scenarios by 3.5 times in unprotected left-turn scenarios. 

**Abstract (ZH)**: 现实世界Troublemaker在自动驾驶系统安全评估中的关键作用及其新型赛道测试框架 

---
# A Mobile Robotic Approach to Autonomous Surface Scanning in Legal Medicine 

**Title (ZH)**: 移动机器人在法医表面扫描中的自主探测方法 

**Authors**: Sarah Grube, Sarah Latus, Martin Fischer, Vidas Raudonis, Axel Heinemann, Benjamin Ondruschka, Alexander Schlaefer  

**Link**: [PDF](https://arxiv.org/pdf/2502.14514)  

**Abstract**: Purpose: Comprehensive legal medicine documentation includes both an internal but also an external examination of the corpse. Typically, this documentation is conducted manually during conventional autopsy. A systematic digital documentation would be desirable, especially for the external examination of wounds, which is becoming more relevant for legal medicine analysis. For this purpose, RGB surface scanning has been introduced. While a manual full surface scan using a handheld camera is timeconsuming and operator dependent, floor or ceiling mounted robotic systems require substantial space and a dedicated room. Hence, we consider whether a mobile robotic system can be used for external documentation. Methods: We develop a mobile robotic system that enables full-body RGB-D surface scanning. Our work includes a detailed configuration space analysis to identify the environmental parameters that need to be considered to successfully perform a surface scan. We validate our findings through an experimental study in the lab and demonstrate the system's application in a legal medicine environment. Results: Our configuration space analysis shows that a good trade-off between coverage and time is reached with three robot base positions, leading to a coverage of 94.96 %. Experiments validate the effectiveness of the system in accurately capturing body surface geometry with an average surface coverage of 96.90 +- 3.16 % and 92.45 +- 1.43 % for a body phantom and actual corpses, respectively. Conclusion: This work demonstrates the potential of a mobile robotic system to automate RGB-D surface scanning in legal medicine, complementing the use of post-mortem CT scans for inner documentation. Our results indicate that the proposed system can contribute to more efficient and autonomous legal medicine documentation, reducing the need for manual intervention. 

**Abstract (ZH)**: 目的：综合法医学文档记录包括内部和外部尸体检查。通常，这种记录在传统尸检过程中手动进行。系统化的数字化记录尤其对于伤口的外部检查更为重要，这部分在法医学分析中变得越来越关键。为此，已经引入了RGB表面扫描技术。尽管手持相机进行全表面扫描耗时且依赖操作者，但安装在天花板或地面上的机器人系统需要大量的空间和专用房间。因此，我们考虑是否可以使用移动机器人系统进行外部记录。方法：我们开发了一种移动机器人系统，以实现全身RGB-D表面扫描。我们的工作包括详细的配置空间分析，以确定需要考虑的环境参数，以便成功执行表面扫描。我们通过实验室实验验证了这些发现，并在法医学环境中展示了系统的应用。结果：配置空间分析结果显示，使用三个机器人基座位置可以达到良好的时间和覆盖率之间的权衡，覆盖率为94.96%。实验验证了系统在准确捕捉身体表面几何形状方面的有效性，平均覆盖率为96.90±3.16%（尸体模型）和92.45±1.43%（实际尸体）。结论：本研究展示了移动机器人系统在法医学中自动化RGB-D表面扫描的潜力，可以与死后CT扫描结合使用，用于内部记录。我们的结果表明，所提出系统可以有助于更高效和自主的法医学文档记录，减少手动干预的需要。 

---
# Watch Less, Feel More: Sim-to-Real RL for Generalizable Articulated Object Manipulation via Motion Adaptation and Impedance Control 

**Title (ZH)**: 减少观看，增强感受：基于运动适应和阻抗控制的拟真到现实的RL通用 articulated 物体操纵研究 

**Authors**: Tan-Dzung Do, Nandiraju Gireesh, Jilong Wang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14457)  

**Abstract**: Articulated object manipulation poses a unique challenge compared to rigid object manipulation as the object itself represents a dynamic environment. In this work, we present a novel RL-based pipeline equipped with variable impedance control and motion adaptation leveraging observation history for generalizable articulated object manipulation, focusing on smooth and dexterous motion during zero-shot sim-to-real transfer. To mitigate the sim-to-real gap, our pipeline diminishes reliance on vision by not leveraging the vision data feature (RGBD/pointcloud) directly as policy input but rather extracting useful low-dimensional data first via off-the-shelf modules. Additionally, we experience less sim-to-real gap by inferring object motion and its intrinsic properties via observation history as well as utilizing impedance control both in the simulation and in the real world. Furthermore, we develop a well-designed training setting with great randomization and a specialized reward system (task-aware and motion-aware) that enables multi-staged, end-to-end manipulation without heuristic motion planning. To the best of our knowledge, our policy is the first to report 84\% success rate in the real world via extensive experiments with various unseen objects. 

**Abstract (ZH)**: 具有关节的物体操作与刚体物体操作相比具有独特的挑战性，因为物体本身代表了一个动态环境。本文提出了一种新颖的基于RL的处理流程，该流程结合了可变阻抗控制和运动适应性，利用观测历史数据实现通用的具有关节的物体操作，专注于零样本模拟到现实的过渡中流畅和灵巧的运动。为了缓解模拟与现实之间的差距，我们的处理流程通过不直接将视觉数据特征（RGBD/点云）作为策略输入，而是先通过现成模块提取有用的低维数据来减少对视觉数据的依赖。此外，我们通过利用观测历史数据推断物体运动及其内在属性，并在模拟和现实世界中都使用阻抗控制，从而减少了模拟与现实之间的差距。进一步地，我们开发了一个具有良好随机性的训练设置和专门的奖励系统（任务感知和运动感知），这使得多阶段的端到端操作得以实现，而无需启发式运动规划。据我们所知，我们的策略是通过各种未见过的物体进行广泛实验后，在现实世界中首次报道了84%的成功率。 

---
# An Efficient Ground-aerial Transportation System for Pest Control Enabled by AI-based Autonomous Nano-UAVs 

**Title (ZH)**: 基于AI自主纳米无人机的高效地面-空中害虫控制运输系统 

**Authors**: Luca Crupi, Luca Butera, Alberto Ferrante, Alessandro Giusti, Daniele Palossi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14455)  

**Abstract**: Efficient crop production requires early detection of pest outbreaks and timely treatments; we consider a solution based on a fleet of multiple autonomous miniaturized unmanned aerial vehicles (nano-UAVs) to visually detect pests and a single slower heavy vehicle that visits the detected outbreaks to deliver treatments. To cope with the extreme limitations aboard nano-UAVs, e.g., low-resolution sensors and sub-100 mW computational power budget, we design, fine-tune, and optimize a tiny image-based convolutional neural network (CNN) for pest detection. Despite the small size of our CNN (i.e., 0.58 GOps/inference), on our dataset, it scores a mean average precision (mAP) of 0.79 in detecting harmful bugs, i.e., 14% lower mAP but 32x fewer operations than the best-performing CNN in the literature. Our CNN runs in real-time at 6.8 frame/s, requiring 33 mW on a GWT GAP9 System-on-Chip aboard a Crazyflie nano-UAV. Then, to cope with in-field unexpected obstacles, we leverage a global+local path planner based on the A* algorithm. The global path planner determines the best route for the nano-UAV to sweep the entire area, while the local one runs up to 50 Hz aboard our nano-UAV and prevents collision by adjusting the short-distance path. Finally, we demonstrate with in-simulator experiments that once a 25 nano-UAVs fleet has combed a 200x200 m vineyard, collected information can be used to plan the best path for the tractor, visiting all and only required hotspots. In this scenario, our efficient transportation system, compared to a traditional single-ground vehicle performing both inspection and treatment, can save up to 20 h working time. 

**Abstract (ZH)**: 高效的农作物生产需要早期发现害虫爆发并及时治疗；我们提出了一种基于多架自主微型无人机（nano-UAVs）的解决方案，用于视觉检测害虫，以及一个较慢的重型车辆访问检测到的爆发区域进行治疗。为应对纳米无人机上的极端限制，如低分辨率传感器和不到100毫瓦的计算能力预算，我们设计、微调并优化了一个小型基于图像的卷积神经网络（CNN）用于害虫检测。尽管我们的CNN规模较小（即每推断0.58 GOps），但在我们的数据集上，它在检测有害昆虫方面的mAP得分为0.79，即比文献中表现最好的CNN低14%的mAP，但操作次数少32倍。我们的CNN在搭载于Crazyflie纳米无人机的GWT GAP9片上系统上实时运行，帧率为6.8帧/秒，耗电33 mW。为了应对现场意外障碍物，我们利用基于A*算法的全局+局部路径规划器。全局路径规划器确定纳米无人机清扫整个区域的最佳路线，而局部路径规划器以高达50 Hz的频率运行在我们的纳米无人机上，并通过调整短距离路径来防止碰撞。最后，通过仿真实验展示了当25架纳米无人机团队扫描200x200米的葡萄园并收集信息后，可以规划拖拉机的最佳路径，访问所有且仅访问必要的热点。在这种情况下，与传统的执行检查和治疗的一架地面车辆相比，我们的高效运输系统可以节省最多20小时的工作时间。 

---
# ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model 

**Title (ZH)**: ChatVLA：统一的多模态理解和机器人控制的视觉-语言-动作模型 

**Authors**: Zhongyi Zhou, Yichen Zhu, Minjie Zhu, Junjie Wen, Ning Liu, Zhiyuan Xu, Weibin Meng, Ran Cheng, Yaxin Peng, Chaomin Shen, Feifei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.14420)  

**Abstract**: Humans possess a unified cognitive ability to perceive, comprehend, and interact with the physical world. Why can't large language models replicate this holistic understanding? Through a systematic analysis of existing training paradigms in vision-language-action models (VLA), we identify two key challenges: spurious forgetting, where robot training overwrites crucial visual-text alignments, and task interference, where competing control and understanding tasks degrade performance when trained jointly. To overcome these limitations, we propose ChatVLA, a novel framework featuring Phased Alignment Training, which incrementally integrates multimodal data after initial control mastery, and a Mixture-of-Experts architecture to minimize task interference. ChatVLA demonstrates competitive performance on visual question-answering datasets and significantly surpasses state-of-the-art vision-language-action (VLA) methods on multimodal understanding benchmarks. Notably, it achieves a six times higher performance on MMMU and scores 47.2% on MMStar with a more parameter-efficient design than ECoT. Furthermore, ChatVLA demonstrates superior performance on 25 real-world robot manipulation tasks compared to existing VLA methods like OpenVLA. Our findings highlight the potential of our unified framework for achieving both robust multimodal understanding and effective robot control. 

**Abstract (ZH)**: 人类拥有感知、理解和与物理世界交互的统一认知能力。为何大型语言模型无法复制这种整体理解能力？通过对视觉-语言-动作模型（VLA）现有训练范式的系统分析，我们识别出两个关键挑战：伪遗忘，即机器人训练会覆盖关键的视觉-文本对齐；任务干扰，竞争控制和理解任务在联合训练时会降低性能。为克服这些限制，我们提出了ChatVLA，这是一种新颖的框架，具备分阶段对齐训练，该框架在初始控制掌握后逐步整合多模态数据，以及混合专家架构以最小化任务干扰。ChatVLA在视觉问答数据集上表现出竞争性性能，并在多模态理解基准上显著超越最先进的视觉-语言-动作（VLA）方法。值得注意的是，它在MMMU上的性能提高了六倍，并在MMStar上的得分为47.2%，其参数效率设计优于ECoT。此外，ChatVLA在25个实际机器人操作任务上的表现优于现有的VLA方法如OpenVLA。我们的发现强调了我们统一框架在实现稳健的多模态理解与有效的机器人控制方面的潜力。 

---
# Mem2Ego: Empowering Vision-Language Models with Global-to-Ego Memory for Long-Horizon Embodied Navigation 

**Title (ZH)**: Mem2Ego: 通过全局到自我记忆增强视觉语言模型在长期 horizons 体态导航中的能力 

**Authors**: Lingfeng Zhang, Yuecheng Liu, Zhanguang Zhang, Matin Aghaei, Yaochen Hu, Hongjian Gu, Mohammad Ali Alomrani, David Gamaliel Arcos Bravo, Raika Karimi, Atia Hamidizadeh, Haoping Xu, Guowei Huang, Zhanpeng Zhang, Tongtong Cao, Weichao Qiu, Xingyue Quan, Jianye Hao, Yuzheng Zhuang, Yingxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14254)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and Vision-Language Models (VLMs) have made them powerful tools in embodied navigation, enabling agents to leverage commonsense and spatial reasoning for efficient exploration in unfamiliar environments. Existing LLM-based approaches convert global memory, such as semantic or topological maps, into language descriptions to guide navigation. While this improves efficiency and reduces redundant exploration, the loss of geometric information in language-based representations hinders spatial reasoning, especially in intricate environments. To address this, VLM-based approaches directly process ego-centric visual inputs to select optimal directions for exploration. However, relying solely on a first-person perspective makes navigation a partially observed decision-making problem, leading to suboptimal decisions in complex environments. In this paper, we present a novel vision-language model (VLM)-based navigation framework that addresses these challenges by adaptively retrieving task-relevant cues from a global memory module and integrating them with the agent's egocentric observations. By dynamically aligning global contextual information with local perception, our approach enhances spatial reasoning and decision-making in long-horizon tasks. Experimental results demonstrate that the proposed method surpasses previous state-of-the-art approaches in object navigation tasks, providing a more effective and scalable solution for embodied navigation. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）和视觉-语言模型（VLMs）使其成为实体导航的强大工具，使代理能够利用常识和空间推理在陌生环境中进行高效探索。现有的基于LLM的方法将全局记忆（如语义或拓扑图）转换为语言描述以指导导航，这虽然提高了效率并减少了重复探索，但基于语言的表示形式损失了几何信息，特别是在复杂环境中妨碍了空间推理。为解决这一问题，基于VLM的方法直接处理自视点视觉输入以选择探索的最佳方向。然而，仅依赖第一视角使导航成为一个部分可观测的决策问题，导致在复杂环境中做出次优决策。在本文中，我们提出了一种新颖的基于VLM的导航框架，通过自适应地从全局记忆模块检索与任务相关的信息并将其与代理的自视点观察结合，来应对这些挑战。通过动态对齐全局上下文信息与局部感知，该方法在长时任务中增强了空间推理与决策能力。实验结果表明，所提出的方法在物体导航任务中超过了先前的最佳方法，提供了更有效和更具扩展性的实体导航解决方案。 

---
# No Minima, No Collisions: Combining Modulation and Control Barrier Function Strategies for Feasible Dynamical Collision Avoidance 

**Title (ZH)**: 没有临界点，没有碰撞：结合调制和控制障壁函数策略实现可行的动力学避碰 

**Authors**: Yifan Xue, Nadia Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2502.14238)  

**Abstract**: As prominent real-time safety-critical reactive control techniques, Control Barrier Function Quadratic Programs (CBF-QPs) work for control affine systems in general but result in local minima in the generated trajectories and consequently cannot ensure convergence to the goals. Contrarily, Modulation of Dynamical Systems (Mod-DSs), including normal, reference, and on-manifold Mod-DS, achieve obstacle avoidance with few and even no local minima but have trouble optimally minimizing the difference between the constrained and the unconstrained controller outputs, and its applications are limited to fully-actuated systems. We dive into the theoretical foundations of CBF-QP and Mod-DS, proving that despite their distinct origins, normal Mod-DS is a special case of CBF-QP, and reference Mod-DS's solutions are mathematically connected to that of the CBF-QP through one equation. Building on top of the unveiled theoretical connections between CBF-QP and Mod-DS, reference Mod-based CBF-QP and on-manifold Mod-based CBF-QP controllers are proposed to combine the strength of CBF-QP and Mod-DS approaches and realize local-minimum-free reactive obstacle avoidance for control affine systems in general. We validate our methods in both simulated hospital environments and real-world experiments using Ridgeback for fully-actuated systems and Fetch robots for underactuated systems. Mod-based CBF-QPs outperform CBF-QPs as well as the optimally constrained-enforcing Mod-DS approaches we proposed in all experiments. 

**Abstract (ZH)**: 基于控制 barrier 函数二次规划的模态动态系统理论基础及其应用 

---
# Real-Time Sampling-based Online Planning for Drone Interception 

**Title (ZH)**: 基于采样的实时在线规划以实现无人机拦截 

**Authors**: Gilhyun Ryou, Lukas Lao Beyer, Sertac Karaman  

**Link**: [PDF](https://arxiv.org/pdf/2502.14231)  

**Abstract**: This paper studies high-speed online planning in dynamic environments. The problem requires finding time-optimal trajectories that conform to system dynamics, meeting computational constraints for real-time adaptation, and accounting for uncertainty from environmental changes. To address these challenges, we propose a sampling-based online planning algorithm that leverages neural network inference to replace time-consuming nonlinear trajectory optimization, enabling rapid exploration of multiple trajectory options under uncertainty. The proposed method is applied to the drone interception problem, where a defense drone must intercept a target while avoiding collisions and handling imperfect target predictions. The algorithm efficiently generates trajectories toward multiple potential target drone positions in parallel. It then assesses trajectory reachability by comparing traversal times with the target drone's predicted arrival time, ultimately selecting the minimum-time reachable trajectory. Through extensive validation in both simulated and real-world environments, we demonstrate our method's capability for high-rate online planning and its adaptability to unpredictable movements in unstructured settings. 

**Abstract (ZH)**: 本文研究动态环境下的高速在线规划问题。该问题要求找到符合系统动力学的时间最优轨迹，满足实时适应的计算约束，并考虑到环境变化带来的不确定性。为了解决这些挑战，我们提出了一种基于采样的在线规划算法，该算法利用神经网络推理来替代耗时的非线性轨迹优化，从而能够在不确定性条件下快速探索多个轨迹选项。所提出的方法应用于无人机拦截问题，在该问题中，一台防御无人机需要拦截目标，同时避免碰撞并处理不完美目标预测。该算法并行生成多个潜在目标无人机位置的轨迹。然后通过比较穿越时间与目标无人机预测到达时间来评估轨迹可达性，最终选择时间最短的可到达轨迹。通过在仿真和真实环境中的广泛验证，我们展示了该方法在高通量在线规划方面的性能及其在无结构环境中对不可预测运动的适应性。 

---
# REFLEX Dataset: A Multimodal Dataset of Human Reactions to Robot Failures and Explanations 

**Title (ZH)**: REFLEX数据集：人类对机器人故障反应及解释的多模态数据集 

**Authors**: Parag Khanna, Andreas Naoum, Elmira Yadollahi, Mårten Björkman, Christian Smith  

**Link**: [PDF](https://arxiv.org/pdf/2502.14185)  

**Abstract**: This work presents REFLEX: Robotic Explanations to FaiLures and Human EXpressions, a comprehensive multimodal dataset capturing human reactions to robot failures and subsequent explanations in collaborative settings. It aims to facilitate research into human-robot interaction dynamics, addressing the need to study reactions to both initial failures and explanations, as well as the evolution of these reactions in long-term interactions. By providing rich, annotated data on human responses to different types of failures, explanation levels, and explanation varying strategies, the dataset contributes to the development of more robust, adaptive, and satisfying robotic systems capable of maintaining positive relationships with human collaborators, even during challenges like repeated failures. 

**Abstract (ZH)**: REFLEX: 机器人故障及其人类解释的综合多模态数据集，用于协作环境中的机器人与人类交互动态研究 

---
# Hybrid Visual Servoing of Tendon-driven Continuum Robots 

**Title (ZH)**: 肌腱驱动连续体机器人的混合视觉伺服控制 

**Authors**: Rana Danesh, Farrokh Janabi-Sharifi, Farhad Aghili  

**Link**: [PDF](https://arxiv.org/pdf/2502.14092)  

**Abstract**: This paper introduces a novel Hybrid Visual Servoing (HVS) approach for controlling tendon-driven continuum robots (TDCRs). The HVS system combines Image-Based Visual Servoing (IBVS) with Deep Learning-Based Visual Servoing (DLBVS) to overcome the limitations of each method and improve overall performance. IBVS offers higher accuracy and faster convergence in feature-rich environments, while DLBVS enhances robustness against disturbances and offers a larger workspace. By enabling smooth transitions between IBVS and DLBVS, the proposed HVS ensures effective control in dynamic, unstructured environments. The effectiveness of this approach is validated through simulations and real-world experiments, demonstrating that HVS achieves reduced iteration time, faster convergence, lower final error, and smoother performance compared to DLBVS alone, while maintaining DLBVS's robustness in challenging conditions such as occlusions, lighting changes, actuator noise, and physical impacts. 

**Abstract (ZH)**: 本文介绍了一种新型混合视觉伺服（HVS）方法，用于控制腱驱动连续机器人（TDCRs）。HVS系统将基于图像的视觉伺服（IBVS）与基于深度学习的视觉伺服（DLBVS）相结合，以克服每种方法的局限性和提高整体性能。IBVS在特征丰富环境中提供更高的精度和更快的收敛速度，而DLBVS增强了对干扰的鲁棒性和更大的工作空间。通过在IBVS和DLBVS之间实现平滑过渡，所提出的HVS确保在动态、非结构化环境中有效控制。通过仿真和实际实验验证了该方法的有效性，结果显示HVS在减少迭代时间、加快收敛速度、降低最终误差和维持更平滑性能方面优于单独使用DLBVS，同时保持DLBVS在遮挡、光照变化、执行器噪声和物理冲击等挑战性条件下的鲁棒性。 

---
# Building reliable sim driving agents by scaling self-play 

**Title (ZH)**: 通过扩展自我对弈构建可靠的模拟驾驶代理 

**Authors**: Daphne Cornelisse, Aarav Pandya, Kevin Joseph, Joseph Suárez, Eugene Vinitsky  

**Link**: [PDF](https://arxiv.org/pdf/2502.14706)  

**Abstract**: Simulation agents are essential for designing and testing systems that interact with humans, such as autonomous vehicles (AVs). These agents serve various purposes, from benchmarking AV performance to stress-testing the system's limits, but all use cases share a key requirement: reliability. A simulation agent should behave as intended by the designer, minimizing unintended actions like collisions that can compromise the signal-to-noise ratio of analyses. As a foundation for reliable sim agents, we propose scaling self-play to thousands of scenarios on the Waymo Open Motion Dataset under semi-realistic limits on human perception and control. Training from scratch on a single GPU, our agents nearly solve the full training set within a day. They generalize effectively to unseen test scenes, achieving a 99.8% goal completion rate with less than 0.8% combined collision and off-road incidents across 10,000 held-out scenarios. Beyond in-distribution generalization, our agents show partial robustness to out-of-distribution scenes and can be fine-tuned in minutes to reach near-perfect performance in those cases. Demonstrations of agent behaviors can be found at this link. We open-source both the pre-trained agents and the complete code base. Demonstrations of agent behaviors can be found at \url{this https URL}. 

**Abstract (ZH)**: 仿真代理对于设计和测试与人类交互的系统（如自动驾驶车辆）至关重要。这些代理承担多种功能，从自动驾驶性能基准测试到压力测试系统的极限，但所有应用场景都共享一个关键要求：可靠性。可靠的仿真代理应该如设计者所期望般行为，尽量减少可能导致分析信噪比下降的意外动作。为构建可靠仿真代理的基础，我们提出在受到人类感知和控制半真实限制的Waymo Open Motion数据集上，将自我对弈扩展到数千种场景。在单块GPU上从头开始训练，我们的代理几乎在一天内解决了整个训练集。它们在未见过的测试场景中表现出有效的泛化能力，针对10,000个保留场景，完成率达99.8%，总计碰撞和离路事故占比不到0.8%。我们的代理还展示了部分对未见过场景的鲁棒性，并能在几分钟内微调以在这些情况下达到近乎完美的性能。代理行为演示可在该链接找到。我们开源了预训练的代理和完整的代码库。代理行为演示可访问此链接：\url{this https URL}。 

---
# ModSkill: Physical Character Skill Modularization 

**Title (ZH)**: ModSkill: 物理特性技能模块化 

**Authors**: Yiming Huang, Zhiyang Dou, Lingjie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14140)  

**Abstract**: Human motion is highly diverse and dynamic, posing challenges for imitation learning algorithms that aim to generalize motor skills for controlling simulated characters. Previous methods typically rely on a universal full-body controller for tracking reference motion (tracking-based model) or a unified full-body skill embedding space (skill embedding). However, these approaches often struggle to generalize and scale to larger motion datasets. In this work, we introduce a novel skill learning framework, ModSkill, that decouples complex full-body skills into compositional, modular skills for independent body parts. Our framework features a skill modularization attention layer that processes policy observations into modular skill embeddings that guide low-level controllers for each body part. We also propose an Active Skill Learning approach with Generative Adaptive Sampling, using large motion generation models to adaptively enhance policy learning in challenging tracking scenarios. Our results show that this modularized skill learning framework, enhanced by generative sampling, outperforms existing methods in precise full-body motion tracking and enables reusable skill embeddings for diverse goal-driven tasks. 

**Abstract (ZH)**: 基于模块化技能的学习框架在精确全身运动跟踪中的表现及其应用 

---
