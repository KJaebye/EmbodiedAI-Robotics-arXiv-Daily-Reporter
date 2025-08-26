# FlowVLA: Thinking in Motion with a Visual Chain of Thought 

**Title (ZH)**: FlowVLA: 在视觉链思过程中思考运动 

**Authors**: Zhide Zhong, Haodong Yan, Junfeng Li, Xiangchen Liu, Xin Gong, Wenxuan Song, Jiayi Chen, Haoang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.18269)  

**Abstract**: Many Vision-Language-Action (VLA) models rely on an internal world model trained via next-frame prediction. This approach, however, struggles with physical reasoning as it entangles static appearance with dynamic motion, often resulting in implausible visual forecasts and inefficient policy learning. To address these limitations, we introduce the Visual Chain of Thought (Visual CoT): a pre-training framework that encourages a model to reason about how a scene evolves before predicting what it will look like. We instantiate this principle in FlowVLA, which predicts a future frame ($v_{t+1}$) only after generating an intermediate optical flow representation ($f_t$) that encodes motion dynamics. This ``$v_t \rightarrow f_t \rightarrow v_{t+1}$'' reasoning process is implemented within a single autoregressive Transformer, guiding the model to learn disentangled dynamics. As a result, FlowVLA produces coherent visual predictions and facilitates more efficient policy learning. Experiments on challenging robotics manipulation benchmarks demonstrate state-of-the-art performance with substantially improved sample efficiency, pointing toward a more principled foundation for world modeling. Project page: this https URL 

**Abstract (ZH)**: 许多视觉-语言-动作（VLA）模型依赖于通过下一帧预测训练的内部世界模型。然而，这种方法在物理推理方面存在困难，因为它将静态外观与动态运动纠缠在一起，往往导致不现实的视觉预测和低效的策略学习。为了解决这些问题，我们引入了视觉链式思维（Visual CoT）：一种预训练框架，鼓励模型在预测未来视觉状态之前先推断场景的演变过程。我们以此原则为基础，在FlowVLA中通过生成一种中间的光学流表示（$f_t$）来预测未来帧（$v_{t+1}$），其中$ f_t $编码了运动动态。这一“$v_t \rightarrow f_t \rightarrow v_{t+1$”推理过程在单一的自回归Transformer中实现，引导模型学习解纠缠的动力学。因此，FlowVLA产生了连贯的视觉预测并促进了更高效的策略学习。实验结果表明，FlowVLA在具有挑战性的机器人操控基准测试中表现出最先进的性能，并且显著提高了样本效率，朝着更为原则的世界建模奠定了基础。项目页面: [这里](this https URL)。 

---
# SafeBimanual: Diffusion-based Trajectory Optimization for Safe Bimanual Manipulation 

**Title (ZH)**: SafeBimanual: 基于扩散的轨迹优化以实现安全双臂操作 

**Authors**: Haoyuan Deng, Wenkai Guo, Qianzhun Wang, Zhenyu Wu, Ziwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18268)  

**Abstract**: Bimanual manipulation has been widely applied in household services and manufacturing, which enables the complex task completion with coordination requirements. Recent diffusion-based policy learning approaches have achieved promising performance in modeling action distributions for bimanual manipulation. However, they ignored the physical safety constraints of bimanual manipulation, which leads to the dangerous behaviors with damage to robots and objects. To this end, we propose a test-time trajectory optimization framework named SafeBimanual for any pre-trained diffusion-based bimanual manipulation policies, which imposes the safety constraints on bimanual actions to avoid dangerous robot behaviors with improved success rate. Specifically, we design diverse cost functions for safety constraints in different dual-arm cooperation patterns including avoidance of tearing objects and collision between arms and objects, which optimizes the manipulator trajectories with guided sampling of diffusion denoising process. Moreover, we employ a vision-language model (VLM) to schedule the cost functions by specifying keypoints and corresponding pairwise relationship, so that the optimal safety constraint is dynamically generated in the entire bimanual manipulation process. SafeBimanual demonstrates superiority on 8 simulated tasks in RoboTwin with a 13.7% increase in success rate and a 18.8% reduction in unsafe interactions over state-of-the-art diffusion-based methods. Extensive experiments on 4 real-world tasks further verify its practical value by improving the success rate by 32.5%. 

**Abstract (ZH)**: 双臂操作已广泛应用于家庭服务和制造领域， Enabled复杂任务的协调完成。近年来，基于扩散的方法在模拟双臂操作中的动作分布方面取得了显著成效。然而，它们忽略了双臂操作的物理安全约束，导致了对机器人和物体造成损害的危险行为。为了解决这一问题，我们提出了一种名为SafeBimanual的测试时轨迹优化框架，可以应用于任何预训练的基于扩散的双臂操作策略，该框架在提高成功率的同时对双臂动作施加安全约束以避免危险的机器人行为。具体地，我们为不同双臂合作模式设计了多样的成本函数，包括避免物体撕裂和双臂与物体之间的碰撞，通过指导扩散去噪过程中的采样来优化操作轨迹。此外，我们采用了一种视觉语言模型（VLM）来调度这些成本函数，通过指定关键点和相应的成对关系生成动态的安全约束，从而在整个双臂操作过程中实现最优的安全约束。SafeBimanual在RoboTwin中8个模拟任务上展示了优越性，与最先进的基于扩散的方法相比，成功率达到13.7%的提升，不安全交互减少18.8%。在4个实际任务上的广泛实验进一步验证了其实际价值，成功率达到32.5%的提升。 

---
# Scene-Agnostic Traversability Labeling and Estimation via a Multimodal Self-supervised Framework 

**Title (ZH)**: 基于多模态自监督框架的场景无关通行性标签与估计 

**Authors**: Zipeng Fang, Yanbo Wang, Lei Zhao, Weidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18249)  

**Abstract**: Traversability estimation is critical for enabling robots to navigate across diverse terrains and environments. While recent self-supervised learning methods achieve promising results, they often fail to capture the characteristics of non-traversable regions. Moreover, most prior works concentrate on a single modality, overlooking the complementary strengths offered by integrating heterogeneous sensory modalities for more robust traversability estimation. To address these limitations, we propose a multimodal self-supervised framework for traversability labeling and estimation. First, our annotation pipeline integrates footprint, LiDAR, and camera data as prompts for a vision foundation model, generating traversability labels that account for both semantic and geometric cues. Then, leveraging these labels, we train a dual-stream network that jointly learns from different modalities in a decoupled manner, enhancing its capacity to recognize diverse traversability patterns. In addition, we incorporate sparse LiDAR-based supervision to mitigate the noise introduced by pseudo labels. Finally, extensive experiments conducted across urban, off-road, and campus environments demonstrate the effectiveness of our approach. The proposed automatic labeling method consistently achieves around 88% IoU across diverse datasets. Compared to existing self-supervised state-of-the-art methods, our multimodal traversability estimation network yields consistently higher IoU, improving by 1.6-3.5% on all evaluated datasets. 

**Abstract (ZH)**: 多模态自监督框架在非通行区域标注与估计中的应用 

---
# DANCeRS: A Distributed Algorithm for Negotiating Consensus in Robot Swarms with Gaussian Belief Propagation 

**Title (ZH)**: DANCeRS：基于高斯信念传播的机器人 swarm 中一致性协商的分布式算法 

**Authors**: Aalok Patwardhan, Andrew J. Davison  

**Link**: [PDF](https://arxiv.org/pdf/2508.18153)  

**Abstract**: Robot swarms require cohesive collective behaviour to address diverse challenges, including shape formation and decision-making. Existing approaches often treat consensus in discrete and continuous decision spaces as distinct problems. We present DANCeRS, a unified, distributed algorithm leveraging Gaussian Belief Propagation (GBP) to achieve consensus in both domains. By representing a swarm as a factor graph our method ensures scalability and robustness in dynamic environments, relying on purely peer-to-peer message passing. We demonstrate the effectiveness of our general framework through two applications where agents in a swarm must achieve consensus on global behaviour whilst relying on local communication. In the first, robots must perform path planning and collision avoidance to create shape formations. In the second, we show how the same framework can be used by a group of robots to form a consensus over a set of discrete decisions. Experimental results highlight our method's scalability and efficiency compared to recent approaches to these problems making it a promising solution for multi-robot systems requiring distributed consensus. We encourage the reader to see the supplementary video demo. 

**Abstract (ZH)**: 机器人群需要具有汇聚的集体行为以应对多样化的挑战，包括形状形成和决策制定。现有方法往往将离散和连续决策空间中的共识视为两个独立的问题。我们提出了DANCeRS统一分布式算法，利用高斯信念传播（GBP）在两个领域中实现共识。通过将群簇表示为因子图，我们的方法确保在动态环境中具有可扩展性和鲁棒性，依赖于纯粹的点对点消息传递。我们通过两个应用展示了我们通用框架的有效性，在这些应用中，群中的代理必须依靠局部通信实现对全局行为的一致性。在第一个应用中，机器人必须进行路径规划和碰撞避免以形成形状。在第二个应用中，我们展示了同一框架如何用于一组机器人以形成对一组离散决策的一致性共识。实验结果突出了我们方法与解决这些问题的近期方法相比的可扩展性和效率，使其成为需要分布式一致性的多机器人系统的一种有前景的解决方案。我们鼓励读者观看补充视频演示。 

---
# Analysis of Harpy's Constrained Trotting and Jumping Maneuver 

**Title (ZH)**: 分析哈皮鸟的受限行走和跳跃机动 Maneuver 

**Authors**: Prathima Ananda Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.18139)  

**Abstract**: This study presents an analysis of experimental data from Harpy, a thruster-assisted bipedal robot developed at Northeastern University. The study examines data sets from trotting and jumping experiments to understand the fundamental principles governing hybrid leg-thruster locomotion. Through data analysis across multiple locomotion modes, this research reveals that Harpy achieves stable locomotion with bounded trajectories and consistent foot placement through strategic leg-thruster synergy. The results demonstrate controlled joint behavior with low torques and symmetric tracking, accurate foot placement within kinematic constraints despite phase-transition perturbations, and underactuated degree-of-freedom stability without divergence. Energy level analysis reveals that legs provide primary propulsion, while the thrusters enable additional aerial phase control. The analysis identifies critical body-leg coupling dynamics during aerial phases that require phase-specific control strategies. Consistent repeatability and symmetry across experiments validate the robustness of the hybrid actuation approach. 

**Abstract (ZH)**: 本研究分析了东北大学开发的Harpy助推双足机器人实验数据，探讨了跑步和跳跃实验数据，以理解混合腿-助推器运动原理。通过跨多种运动模式的数据分析，本研究揭示Harpy通过策略性腿部-助推器协同作用实现了稳定的有限轨迹和一致的脚部定位。结果表明，关节行为受到控制且扭矩较低、跟踪对称，尽管有相位过渡干扰，脚部仍在运动学约束内精确定位，且欠驱动自由度保持稳定而不发散。能量水平分析显示，腿部提供主要推力，助推器允许额外的空中相位控制。分析识别了空中相位期间的关键身体-腿部耦合动力学，需要特定相位的控制策略。一致的重复性和对称性验证了混合驱动方法的稳健性。 

---
# The Effects of Communication Delay on Human Performance and Neurocognitive Responses in Mobile Robot Teleoperation 

**Title (ZH)**: 通信延迟对移动机器人遥控中人类表现及神经认知反应的影响 

**Authors**: Zhaokun Chen, Wenshuo Wang, Wenzhuo Liu, Yichen Liu, Junqiang Xi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18074)  

**Abstract**: Communication delays in mobile robot teleoperation adversely affect human-machine collaboration. Understanding delay effects on human operational performance and neurocognition is essential for resolving this issue. However, no previous research has explored this. To fill this gap, we conduct a human-in-the-loop experiment involving 10 participants, integrating electroencephalography (EEG) and robot behavior data under varying delays (0-500 ms in 100 ms increments) to systematically investigate these effects. Behavior analysis reveals significant performance degradation at 200-300 ms delays, affecting both task efficiency and accuracy. EEG analysis discovers features with significant delay dependence: frontal $\theta/\beta$-band and parietal $\alpha$-band power. We also identify a threshold window (100-200 ms) for early perception of delay in humans, during which these EEG features first exhibit significant differences. When delay exceeds 400 ms, all features plateau, indicating saturation of cognitive resource allocation at physiological limits. These findings provide the first evidence of perceptual and cognitive delay thresholds during teleoperation tasks in humans, offering critical neurocognitive insights for the design of delay compensation strategies. 

**Abstract (ZH)**: 移动机器人远程操作中的通信延迟影响人机协作。探索延迟对人类操作性能和神经认知影响的规律对于解决这一问题至关重要。然而，此前的研究尚未涉及此领域。为填补这一空白，我们进行了10名参与者参与的包含电encephalography（EEG）和机器人行为数据的人机闭环实验，探索不同延迟（0-500 ms，间隔100 ms）下的这些影响。行为分析显示，在200-300 ms延迟时，任务效率和准确性显著下降。EEG分析发现具有显著延迟依赖性的特征：前额θ/β频段和顶叶α频段的功率。我们还确定了一个感知延迟的阈值窗口（100-200 ms），在此期间这些EEG特征首次表现出显著差异。当延迟超过400 ms时，所有特征均达到饱和状态，表明认知资源分配在生理极限处饱和。这些发现提供了人类在远程操作任务中感知和认知延迟阈值的首个证据，为设计延迟补偿策略提供了关键的神经认知洞察。 

---
# Arnold: a generalist muscle transformer policy 

**Title (ZH)**: Arnold: 通用肌肉转换器政策 

**Authors**: Alberto Silvio Chiappa, Boshi An, Merkourios Simos, Chengkun Li, Alexander Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2508.18066)  

**Abstract**: Controlling high-dimensional and nonlinear musculoskeletal models of the human body is a foundational scientific challenge. Recent machine learning breakthroughs have heralded policies that master individual skills like reaching, object manipulation and locomotion in musculoskeletal systems with many degrees of freedom. However, these agents are merely "specialists", achieving high performance for a single skill. In this work, we develop Arnold, a generalist policy that masters multiple tasks and embodiments. Arnold combines behavior cloning and fine-tuning with PPO to achieve expert or super-expert performance in 14 challenging control tasks from dexterous object manipulation to locomotion. A key innovation is Arnold's sensorimotor vocabulary, a compositional representation of the semantics of heterogeneous sensory modalities, objectives, and actuators. Arnold leverages this vocabulary via a transformer architecture to deal with the variable observation and action spaces of each task. This framework supports efficient multi-task, multi-embodiment learning and facilitates rapid adaptation to novel tasks. Finally, we analyze Arnold to provide insights into biological motor control, corroborating recent findings on the limited transferability of muscle synergies across tasks. 

**Abstract (ZH)**: 控制高维度和非线性的人体 musculoskeletal 模型是一项基础科学挑战。近期的机器学习突破已经使人们能够使用多自由度 musculoskeletal 系统掌握诸如抓取、物体操作和行走等单一技能。然而，这些代理仅仅是在单一技能上达到高性能的“专家”。在本文中，我们开发了 Arnold，一种能够在多个任务和实体上掌握技能的通用代理。Arnold 结合了行为克隆、微调与 PPO，以在从灵巧物体操作到行走的 14 项具有挑战性的控制任务中达到专家或超专家性能。一个关键创新在于 Arnold 的运动感知词汇表，这是一种组合表示异构传感模态、目标和执行器语义的表示方法。Arnold 通过基于变换器的架构利用这一词汇表，以应对每个任务观测和动作空间的变化。该框架支持高效的多任务、多实体学习，并促进对新任务的快速适应。最后，我们对 Arnold 进行分析，以提供关于生物运动控制的见解，验证了最近关于跨任务肌肉协同工作转换有限的研究发现。 

---
# Modeling and Control Framework for Autonomous Space Manipulator Handover Operations 

**Title (ZH)**: 自主太空 manipulator 手蒯交接操作的建模与控制框架 

**Authors**: Diego Quevedo, Sarah Hudson, Donghoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.18039)  

**Abstract**: Autonomous space robotics is poised to play a vital role in future space missions, particularly for In-space Servicing, Assembly, and Manufacturing (ISAM). A key capability in such missions is the Robot-to-Robot (R2R) handover of mission-critical objects. This work presents a dynamic model of a dual-arm space manipulator system and compares various tracking control laws. The key contributions of this work are the development of a cooperative manipulator dynamic model and the comparative analysis of control laws to support autonomous R2R handovers in ISAM scenarios. 

**Abstract (ZH)**: 自主空间机器人在future空间任务中的关键作用，特别是在空间服务、组装与制造（ISAM）领域，尤其体现在机器人到机器人（R2R）任务关键对象的手递手操作中。本文提出了一个双臂空间操作器系统的动力学模型，并比较了多种跟踪控制律。本文的主要贡献在于开发了协同操作器动力学模型，并对支持ISAM场景中自主R2R手递手操作的控制律进行了比较分析。 

---
# No Need to Look! Locating and Grasping Objects by a Robot Arm Covered with Sensitive Skin 

**Title (ZH)**: 无需观察！基于敏感皮肤覆盖的机器人手臂定位与抓取物体 

**Authors**: Karel Bartunek, Lukas Rustler, Matej Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2508.17986)  

**Abstract**: Locating and grasping of objects by robots is typically performed using visual sensors. Haptic feedback from contacts with the environment is only secondary if present at all. In this work, we explored an extreme case of searching for and grasping objects in complete absence of visual input, relying on haptic feedback only. The main novelty lies in the use of contacts over the complete surface of a robot manipulator covered with sensitive skin. The search is divided into two phases: (1) coarse workspace exploration with the complete robot surface, followed by (2) precise localization using the end-effector equipped with a force/torque sensor. We systematically evaluated this method in simulation and on the real robot, demonstrating that diverse objects can be located, grasped, and put in a basket. The overall success rate on the real robot for one object was 85.7\% with failures mainly while grasping specific objects. The method using whole-body contacts is six times faster compared to a baseline that uses haptic feedback only on the end-effector. We also show locating and grasping multiple objects on the table. This method is not restricted to our specific setup and can be deployed on any platform with the ability of sensing contacts over the entire body surface. This work holds promise for diverse applications in areas with challenging visual perception (due to lighting, dust, smoke, occlusion) such as in agriculture when fruits or vegetables need to be located inside foliage and picked. 

**Abstract (ZH)**: 基于全身体感反馈的物体定位与抓取 

---
# Integration of Computer Vision with Adaptive Control for Autonomous Driving Using ADORE 

**Title (ZH)**: 基于ADORE的计算机视觉与自适应控制集成在自主驾驶中的应用 

**Authors**: Abu Shad Ahammed, Md Shahi Amran Hossain, Sayeri Mukherjee, Roman Obermaisser, Md. Ziaur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2508.17985)  

**Abstract**: Ensuring safety in autonomous driving requires a seamless integration of perception and decision making under uncertain conditions. Although computer vision (CV) models such as YOLO achieve high accuracy in detecting traffic signs and obstacles, their performance degrades in drift scenarios caused by weather variations or unseen objects. This work presents a simulated autonomous driving system that combines a context aware CV model with adaptive control using the ADORE framework. The CARLA simulator was integrated with ADORE via the ROS bridge, allowing real-time communication between perception, decision, and control modules. A simulated test case was designed in both clear and drift weather conditions to demonstrate the robust detection performance of the perception model while ADORE successfully adapted vehicle behavior to speed limits and obstacles with low response latency. The findings highlight the potential of coupling deep learning-based perception with rule-based adaptive decision making to improve automotive safety critical system. 

**Abstract (ZH)**: 确保自动驾驶安全需要在不确定条件下实现感知与决策的无缝集成。虽然YOLO等计算机视觉模型在检测交通标志和障碍物方面具有高精度，但在由天气变化或未见过的物体引起的漂移场景中，其性能会下降。本研究提出了一种结合基于上下文的计算机视觉模型和ADORE框架自适应控制的模拟自动驾驶系统。通过ROS桥将CARLA模拟器与ADORE集成，实现了感知、决策和控制模块之间的实时通信。在晴朗和漂移天气条件下设计了模拟测试案例，展示了感知模型的稳健检测性能，同时ADORE成功地适应了速度限制和障碍物，并具有低响应延迟。研究结果突显了将基于深度学习的感知与基于规则的自适应决策相结合以提高汽车安全关键系统性能的潜力。 

---
# A holistic perception system of internal and external monitoring for ground autonomous vehicles: AutoTRUST paradigm 

**Title (ZH)**: 基于内部和外部监测的地面自主车辆全面感知系统：AutoTRUST框架 

**Authors**: Alexandros Gkillas, Christos Anagnostopoulos, Nikos Piperigkos, Dimitris Tsiktsiris, Theofilos Christodoulou, Theofanis Siamatras, Dimitrios Triantafyllou, Christos Basdekis, Theoktisti Marinopoulou, Panagiotis Lepentsiotis, Elefterios Blitsis, Aggeliki Zacharaki, Nearchos Stylianidis, Leonidas Katelaris, Lamberto Salvan, Aris S. Lalos, Christos Laoudias, Antonios Lalas, Konstantinos Votis  

**Link**: [PDF](https://arxiv.org/pdf/2508.17969)  

**Abstract**: This paper introduces a holistic perception system for internal and external monitoring of autonomous vehicles, with the aim of demonstrating a novel AI-leveraged self-adaptive framework of advanced vehicle technologies and solutions that optimize perception and experience on-board. Internal monitoring system relies on a multi-camera setup designed for predicting and identifying driver and occupant behavior through facial recognition, exploiting in addition a large language model as virtual assistant. Moreover, the in-cabin monitoring system includes AI-empowered smart sensors that measure air-quality and perform thermal comfort analysis for efficient on and off-boarding. On the other hand, external monitoring system perceives the surrounding environment of vehicle, through a LiDAR-based cost-efficient semantic segmentation approach, that performs highly accurate and efficient super-resolution on low-quality raw 3D point clouds. The holistic perception framework is developed in the context of EU's Horizon Europe programm AutoTRUST, and has been integrated and deployed on a real electric vehicle provided by ALKE. Experimental validation and evaluation at the integration site of Joint Research Centre at Ispra, Italy, highlights increased performance and efficiency of the modular blocks of the proposed perception architecture. 

**Abstract (ZH)**: 基于人工智能的自主车辆全方位感知系统：一种先进的自我适应框架研究 

---
# Egocentric Instruction-oriented Affordance Prediction via Large Multimodal Model 

**Title (ZH)**: 以自我中心指令为导向的大规模多模态可用性预测 

**Authors**: Bokai Ji, Jie Gu, Xiaokang Ma, Chu Tang, Jingmin Chen, Guangxia Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17922)  

**Abstract**: Affordance is crucial for intelligent robots in the context of object manipulation. In this paper, we argue that affordance should be task-/instruction-dependent, which is overlooked by many previous works. That is, different instructions can lead to different manipulation regions and directions even for the same object. According to this observation, we present a new dataset comprising fifteen thousand object-instruction-affordance triplets. All scenes in the dataset are from an egocentric viewpoint, designed to approximate the perspective of a human-like robot. Furthermore, we investigate how to enable large multimodal models (LMMs) to serve as affordance predictors by implementing a ``search against verifiers'' pipeline. An LMM is asked to progressively predict affordances, with the output at each step being verified by itself during the iterative process, imitating a reasoning process. Experiments show that our method not only unlocks new instruction-oriented affordance prediction capabilities, but also achieves outstanding performance broadly. 

**Abstract (ZH)**: 智能机器人在物体操作情境下，功能感知至关重要。本文认为，功能感知应具有任务/指令依赖性，这是许多先前工作的不足之处。即，即使面对同一个物体，不同的指令也会导致不同的操作区域和方向。基于这一观察，我们提出了一包含十五 thousand 物体-指令-功能感知三元组的新数据集。所有场景均从第一人称视角设计，旨在模拟类人机器人视角。此外，我们探讨了通过实施“搜索对抗验证者”管道，使大规模多模态模型（LMM）能够作为功能感知预测器的可能性。在迭代过程中，LMM 被要求逐步预测功能，并在每一阶段输出被自身验证，模仿推理过程。实验表明，我们的方法不仅解锁了新的以指令为导向的功能感知预测能力，还在广泛的应用中取得了优异的表现。 

---
# CubeDN: Real-time Drone Detection in 3D Space from Dual mmWave Radar Cubes 

**Title (ZH)**: CubeDN：来自双毫米波雷达立方体的三维空间实时无人机检测 

**Authors**: Yuan Fang, Fangzhan Shi, Xijia Wei, Qingchao Chen, Kevin Chetty, Simon Julier  

**Link**: [PDF](https://arxiv.org/pdf/2508.17831)  

**Abstract**: As drone use has become more widespread, there is a critical need to ensure safety and security. A key element of this is robust and accurate drone detection and localization. While cameras and other optical sensors like LiDAR are commonly used for object detection, their performance degrades under adverse lighting and environmental conditions. Therefore, this has generated interest in finding more reliable alternatives, such as millimeter-wave (mmWave) radar. Recent research on mmWave radar object detection has predominantly focused on 2D detection of road users. Although these systems demonstrate excellent performance for 2D problems, they lack the sensing capability to measure elevation, which is essential for 3D drone detection. To address this gap, we propose CubeDN, a single-stage end-to-end radar object detection network specifically designed for flying drones. CubeDN overcomes challenges such as poor elevation resolution by utilizing a dual radar configuration and a novel deep learning pipeline. It simultaneously detects, localizes, and classifies drones of two sizes, achieving decimeter-level tracking accuracy at closer ranges with overall $95\%$ average precision (AP) and $85\%$ average recall (AR). Furthermore, CubeDN completes data processing and inference at 10Hz, making it highly suitable for practical applications. 

**Abstract (ZH)**: 随着无人机使用范围的扩大，确保安全与security的需要变得至关重要。一个重要方面是 robust 和 accurate 的无人机检测与定位。尽管相机和其他光学传感器如 LiDAR 广泛用于目标检测，但在不良光照和环境条件下，它们的表现会下降。因此，寻找更可靠替代方案，如毫米波（mmWave）雷达，引起了人们的兴趣。最近关于 mmWave 雷达目标检测的研究主要集中在2D检测道路使用者上。虽然这些系统在2D问题上表现出色，但缺乏测量海拔的能力，这是3D无人机检测必不可少的。为了解决这个差距，我们提出 CubeDN，这是一种专门用于飞行无人机的一阶段端到端雷达目标检测网络。CubeDN 通过使用双雷达配置和一种新颖的深度学习流水线，克服了低海拔分辨率等问题。它同时检测、定位和分类两种尺寸的无人机，在近距离范围内实现了分米级的跟踪精度，总体平均精度（AP）为95%，平均召回率（AR）为85%。此外，CubeDN 在10Hz 的数据处理和推理速度下运行，使其非常适用于实际应用。 

---
# Effect of Performance Feedback Timing on Motor Learning for a Surgical Training Task 

**Title (ZH)**: 手术训练任务中绩效反馈时间对运动学习的影响 

**Authors**: Mary Kate Gale, Kailana Baker-Matsuoka, Ilana Nisky, Allison Okamura  

**Link**: [PDF](https://arxiv.org/pdf/2508.17830)  

**Abstract**: Objective: Robot-assisted minimally invasive surgery (RMIS) has become the gold standard for a variety of surgical procedures, but the optimal method of training surgeons for RMIS is unknown. We hypothesized that real-time, rather than post-task, error feedback would better increase learning speed and reduce errors. Methods: Forty-two surgical novices learned a virtual version of the ring-on-wire task, a canonical task in RMIS training. We investigated the impact of feedback timing with multi-sensory (haptic and visual) cues in three groups: (1) real-time error feedback, (2) trial replay with error feedback, and (3) no error feedback. Results: Participant performance was evaluated based on the accuracy of ring position and orientation during the task. Participants who received real-time feedback outperformed other groups in ring orientation. Additionally, participants who received feedback in replay outperformed participants who did not receive any error feedback on ring orientation during long, straight path sections. There were no significant differences between groups for ring position overall, but participants who received real-time feedback outperformed the other groups in positional accuracy on tightly curved path sections. Conclusion: The addition of real-time haptic and visual error feedback improves learning outcomes in a virtual surgical task over error feedback in replay or no error feedback at all. Significance: This work demonstrates that multi-sensory error feedback delivered in real time leads to better training outcomes as compared to the same feedback delivered after task completion. This novel method of training may enable surgical trainees to develop skills with greater speed and accuracy. 

**Abstract (ZH)**: 客观目标：机器人辅助微创手术（RMIS）已成为多种外科手术的标准，但最佳的外科医生培训方法尚未确定。我们假设实时错误反馈而非任务后错误反馈能够更好地提高学习速度并减少错误。方法：42名手术初学者学习了一个虚拟的环在绳上的任务，这是RMIS培训中的一个经典任务。我们通过多感官（触觉和视觉）提示在三个组中研究了反馈时间的影响：（1）实时错误反馈，（2）试次回放带有错误反馈，（3）无错误反馈。结果：根据任务中环的位置和方向准确性评估参与者的表现。接受实时反馈的参与者在环的方向上表现最好。此外，在长直线路径段上，接受回放反馈的参与者在环的方向上表现优于未接受错误反馈的参与者。总体上，各组在环的位置上没有显著差异，但接受实时反馈的参与者在紧弯路径段上的位置准确性上表现最好。结论：实时的触觉和视觉错误反馈在虚拟外科任务中的学习效果优于任务后错误反馈或完全没有错误反馈。意义：这项工作表明，与任务完成后提供的反馈相比，实时的多感官错误反馈能够更好地提高培训效果。这一新的培训方法可以使外科训练生以更快和更准确的方式发展技能。 

---
# Adaptive Output Steps: FlexiSteps Network for Dynamic Trajectory Prediction 

**Title (ZH)**: 自适应输出步长：FlexiSteps 网络用于动态轨迹预测 

**Authors**: Yunxiang Liu, Hongkuo Niu, Jianlin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17797)  

**Abstract**: Accurate trajectory prediction is vital for autonomous driving, robotics, and intelligent decision-making systems, yet traditional models typically rely on fixed-length output predictions, limiting their adaptability to dynamic real-world scenarios. In this paper, we introduce the FlexiSteps Network (FSN), a novel framework that dynamically adjusts prediction output time steps based on varying contextual conditions. Inspired by recent advancements addressing observation length discrepancies and dynamic feature extraction, FSN incorporates an pre-trained Adaptive Prediction Module (APM) to evaluate and adjust the output steps dynamically, ensuring optimal prediction accuracy and efficiency. To guarantee the plug-and-play of our FSN, we also design a Dynamic Decoder(DD). Additionally, to balance the prediction time steps and prediction accuracy, we design a scoring mechanism, which not only introduces the Fréchet distance to evaluate the geometric similarity between the predicted trajectories and the ground truth trajectories but the length of predicted steps is also considered. Extensive experiments conducted on benchmark datasets including Argoverse and INTERACTION demonstrate the effectiveness and flexibility of our proposed FSN framework. 

**Abstract (ZH)**: 灵活步长网络：基于动态条件调整预测时间步的轨迹预测方法 

---
# Talking to Robots: A Practical Examination of Speech Foundation Models for HRI Applications 

**Title (ZH)**: 与机器人对话：对用于人机交互应用的语音基础模型的实践考察 

**Authors**: Theresa Pekarek Rosin, Julia Gachot, Henri-Leon Kordt, Matthias Kerzel, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2508.17753)  

**Abstract**: Automatic Speech Recognition (ASR) systems in real-world settings need to handle imperfect audio, often degraded by hardware limitations or environmental noise, while accommodating diverse user groups. In human-robot interaction (HRI), these challenges intersect to create a uniquely challenging recognition environment. We evaluate four state-of-the-art ASR systems on eight publicly available datasets that capture six dimensions of difficulty: domain-specific, accented, noisy, age-variant, impaired, and spontaneous speech. Our analysis demonstrates significant variations in performance, hallucination tendencies, and inherent biases, despite similar scores on standard benchmarks. These limitations have serious implications for HRI, where recognition errors can interfere with task performance, user trust, and safety. 

**Abstract (ZH)**: 自动语音识别（ASR）系统在实际应用场景中需要处理受硬件限制或环境噪声影响的不完美音频，并适应多样的用户群体。在人机交互（HRI）中，这些挑战交织在一起，形成了一个独特的挑战性识别环境。我们评估了四种最新的ASR系统在八个公开可用数据集上的性能，这些数据集涵盖了六种难度维度：领域特定的、带口音的、嘈杂的、年龄变化的、受损的和自发的语音。我们的分析表明，尽管在标准基准上的得分相似，但在性能、幻觉倾向和固有偏差方面存在显著差异。这些局限性对HRI有严重影响，因为识别错误可能会干扰任务性能、用户信任和安全性。 

---
# MEVITA: Open-Source Bipedal Robot Assembled from E-Commerce Components via Sheet Metal Welding 

**Title (ZH)**: MEVITA：通过板材焊接组装的基于电子商务组件的开源双足机器人 

**Authors**: Kento Kawaharazuka, Shogo Sawaguchi, Ayumu Iwata, Keita Yoneda, Temma Suzuki, Kei Okada  

**Link**: [PDF](https://arxiv.org/pdf/2508.17684)  

**Abstract**: Various bipedal robots have been developed to date, and in recent years, there has been a growing trend toward releasing these robots as open-source platforms. This shift is fostering an environment in which anyone can freely develop bipedal robots and share their knowledge, rather than relying solely on commercial products. However, most existing open-source bipedal robots are designed to be fabricated using 3D printers, which limits their scalability in size and often results in fragile structures. On the other hand, some metal-based bipedal robots have been developed, but they typically involve a large number of components, making assembly difficult, and in some cases, the parts themselves are not readily available through e-commerce platforms. To address these issues, we developed MEVITA, an open-source bipedal robot that can be built entirely from components available via e-commerce. Aiming for the minimal viable configuration for a bipedal robot, we utilized sheet metal welding to integrate complex geometries into single parts, thereby significantly reducing the number of components and enabling easy assembly for anyone. Through reinforcement learning in simulation and Sim-to-Real transfer, we demonstrated robust walking behaviors across various environments, confirming the effectiveness of our approach. All hardware, software, and training environments can be obtained from this https URL . 

**Abstract (ZH)**: 开源电商组件可构建的 bipedal 机器人 MEVITA：基于板材焊接的简约设计与强化学习验证 

---
# SEBVS: Synthetic Event-based Visual Servoing for Robot Navigation and Manipulation 

**Title (ZH)**: SEBVS: 合成事件驱动的视觉伺服在机器人导航与操作中的应用 

**Authors**: Krishna Vinod, Prithvi Jai Ramesh, Pavan Kumar B N, Bharatesh Chakravarthi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17643)  

**Abstract**: Event cameras offer microsecond latency, high dynamic range, and low power consumption, making them ideal for real-time robotic perception under challenging conditions such as motion blur, occlusion, and illumination changes. However, despite their advantages, synthetic event-based vision remains largely unexplored in mainstream robotics simulators. This lack of simulation setup hinders the evaluation of event-driven approaches for robotic manipulation and navigation tasks. This work presents an open-source, user-friendly v2e robotics operating system (ROS) package for Gazebo simulation that enables seamless event stream generation from RGB camera feeds. The package is used to investigate event-based robotic policies (ERP) for real-time navigation and manipulation. Two representative scenarios are evaluated: (1) object following with a mobile robot and (2) object detection and grasping with a robotic manipulator. Transformer-based ERPs are trained by behavior cloning and compared to RGB-based counterparts under various operating conditions. Experimental results show that event-guided policies consistently deliver competitive advantages. The results highlight the potential of event-driven perception to improve real-time robotic navigation and manipulation, providing a foundation for broader integration of event cameras into robotic policy learning. The GitHub repo for the dataset and code: this https URL 

**Abstract (ZH)**: 事件相机提供了微秒级延迟、高动态范围和低功耗，使其在运动模糊、遮挡和光照变化等挑战条件下进行实时机器人感知的理想选择。然而，尽管具有这些优点，合成事件驱动视觉在主流机器人模拟器中仍 largely unexplored。本工作 presents 一个开源、用户友好的从RGB相机馈送生成事件流的v2e机器人操作系统(ROS)包，以用于Gazebo模拟。该包用于研究事件驱动机器人策略(ERP)在实时导航和 manipulation 任务中的应用。评估了两个代表性场景：（1）移动机器人物体跟随和（2）机器人 manipulator 物体检测与抓取。基于Transformer的 ERP 通过行为克隆训练，并在不同操作条件下与基于RGB的同类进行比较。实验结果表明，事件引导策略一致地提供了竞争力的优势。结果强调了事件驱动感知在提高实时机器人导航和 manipulation 方面的潜力，为更广泛地将事件相机整合到机器人策略学习中奠定了基础。GitHub 数据集和代码仓库: this https URL。 

---
# GWM: Towards Scalable Gaussian World Models for Robotic Manipulation 

**Title (ZH)**: GWM：面向可扩展的机器人操作高斯世界模型 

**Authors**: Guanxing Lu, Baoxiong Jia, Puhao Li, Yixin Chen, Ziwei Wang, Yansong Tang, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17600)  

**Abstract**: Training robot policies within a learned world model is trending due to the inefficiency of real-world interactions. The established image-based world models and policies have shown prior success, but lack robust geometric information that requires consistent spatial and physical understanding of the three-dimensional world, even pre-trained on internet-scale video sources. To this end, we propose a novel branch of world model named Gaussian World Model (GWM) for robotic manipulation, which reconstructs the future state by inferring the propagation of Gaussian primitives under the effect of robot actions. At its core is a latent Diffusion Transformer (DiT) combined with a 3D variational autoencoder, enabling fine-grained scene-level future state reconstruction with Gaussian Splatting. GWM can not only enhance the visual representation for imitation learning agent by self-supervised future prediction training, but can serve as a neural simulator that supports model-based reinforcement learning. Both simulated and real-world experiments depict that GWM can precisely predict future scenes conditioned on diverse robot actions, and can be further utilized to train policies that outperform the state-of-the-art by impressive margins, showcasing the initial data scaling potential of 3D world model. 

**Abstract (ZH)**: 基于学习的世界模型内训练机器人策略正逐渐成为趋势，以应对真实世界交互的低效问题。现有的基于图像的世界模型和策略虽取得了一定成功，但缺乏 robust 的几何信息，这种信息需要一致的空间和物理理解三维世界，即使是在大规模互联网视频数据预训练的情况下。为了解决这一问题，我们提出了一种新的世界模型分支，即高斯世界模型（GWM），该模型通过在机器人动作影响下推断高斯原元素的传播来重建未来状态。其核心是一个潜变量扩散变换器（DiT）结合3D 变分自动编码器，实现基于高斯体绘制的细粒度场景级未来状态重建。GWM 不仅可以通过自监督的未来预测训练增强视觉表示，还可以作为神经模拟器支持基于模型的强化学习。模拟和现实世界实验表明，GWM 能在不同机器人动作条件下精确预测未来场景，并可通过进一步训练超越现有最佳方法的策略，展示了三维世界模型的初步数据规模潜力。 

---
# LodeStar: Long-horizon Dexterity via Synthetic Data Augmentation from Human Demonstrations 

**Title (ZH)**: LodeStar: 长期灵活性通过人类演示的合成数据扩增获取 

**Authors**: Weikang Wan, Jiawei Fu, Xiaodi Yuan, Yifeng Zhu, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.17547)  

**Abstract**: Developing robotic systems capable of robustly executing long-horizon manipulation tasks with human-level dexterity is challenging, as such tasks require both physical dexterity and seamless sequencing of manipulation skills while robustly handling environment variations. While imitation learning offers a promising approach, acquiring comprehensive datasets is resource-intensive. In this work, we propose a learning framework and system LodeStar that automatically decomposes task demonstrations into semantically meaningful skills using off-the-shelf foundation models, and generates diverse synthetic demonstration datasets from a few human demos through reinforcement learning. These sim-augmented datasets enable robust skill training, with a Skill Routing Transformer (SRT) policy effectively chaining the learned skills together to execute complex long-horizon manipulation tasks. Experimental evaluations on three challenging real-world long-horizon dexterous manipulation tasks demonstrate that our approach significantly improves task performance and robustness compared to previous baselines. Videos are available at this http URL. 

**Abstract (ZH)**: 开发能够在长时间尺度上以人类级灵巧性稳健执行操作任务的机器人系统是一项挑战，因为这类任务要求兼具物理灵巧性和在处理环境变化时无缝衔接操作技能的能力。虽然模仿学习提供了有前景的方法，但获取全面的数据集需要大量资源。在本文中，我们提出了一种学习框架和系统LodeStar，该系统利用现成的基础模型自动将任务演示分解成语义上有意义的技能，并通过强化学习从少量的人类演示生成多样化的合成演示数据集。这些强化学习增强的数据集能够实现技能的稳健训练，Skill Routing Transformer (SRT) 策略能够有效地将学到的技能串联起来以执行复杂的长时间尺度操作任务。在三个具有挑战性的现实世界长时间尺度灵巧操作任务上的实验评估表明，与先前的基本方法相比，我们的方法显著改善了任务性能和稳健性。视频可在以下网址获取：this http URL。 

---
# Variational Shape Inference for Grasp Diffusion on SE(3) 

**Title (ZH)**: SE(3) 上抓取扩散的变分形状推断 

**Authors**: S. Talha Bukhari, Kaivalya Agrawal, Zachary Kingston, Aniket Bera  

**Link**: [PDF](https://arxiv.org/pdf/2508.17482)  

**Abstract**: Grasp synthesis is a fundamental task in robotic manipulation which usually has multiple feasible solutions. Multimodal grasp synthesis seeks to generate diverse sets of stable grasps conditioned on object geometry, making the robust learning of geometric features crucial for success. To address this challenge, we propose a framework for learning multimodal grasp distributions that leverages variational shape inference to enhance robustness against shape noise and measurement sparsity. Our approach first trains a variational autoencoder for shape inference using implicit neural representations, and then uses these learned geometric features to guide a diffusion model for grasp synthesis on the SE(3) manifold. Additionally, we introduce a test-time grasp optimization technique that can be integrated as a plugin to further enhance grasping performance. Experimental results demonstrate that our shape inference for grasp synthesis formulation outperforms state-of-the-art multimodal grasp synthesis methods on the ACRONYM dataset by 6.3%, while demonstrating robustness to deterioration in point cloud density compared to other approaches. Furthermore, our trained model achieves zero-shot transfer to real-world manipulation of household objects, generating 34% more successful grasps than baselines despite measurement noise and point cloud calibration errors. 

**Abstract (ZH)**: 多模态抓取合成是机器人操作中的一个基础任务，通常有多重可行的解决方案。多模态抓取合成旨在基于物体几何形状生成多样化的稳定抓取集合，因此学习几何特征的稳健性对于成功至关重要。为解决这一挑战，我们提出了一个利用变分形状推断的框架，以增强对形状噪声和测量稀疏性的鲁棒性。我们的方法首先使用隐式神经表示训练一个变分自编码器进行形状推断，然后使用这些学习到的几何特征引导SE(3)流形上的抓取合成。此外，我们引入了一种测试时抓取优化技术，该技术可以作为插件进一步增强抓取性能。实验结果表明，我们的抓取合成形状推断方法在ACRONYM数据集上优于最先进的多模态抓取合成方法，性能高出6.3%，并且在点云密度恶化的情况下比其他方法更具有鲁棒性。进一步而言，我们训练的模型实现了对家庭用品的真实世界操作的零样本迁移，尽管存在测量噪声和点云校准误差的情况下，生成的成功抓取数量比基线方法多34%。 

---
# Morphological Cognition: Classifying MNIST Digits Through Morphological Computation Alone 

**Title (ZH)**: 形态认知：仅通过形态计算分类MNIST数字 

**Authors**: Alican Mertan, Nick Cheney  

**Link**: [PDF](https://arxiv.org/pdf/2508.17469)  

**Abstract**: With the rise of modern deep learning, neural networks have become an essential part of virtually every artificial intelligence system, making it difficult even to imagine different models for intelligent behavior. In contrast, nature provides us with many different mechanisms for intelligent behavior, most of which we have yet to replicate. One of such underinvestigated aspects of intelligence is embodiment and the role it plays in intelligent behavior. In this work, we focus on how the simple and fixed behavior of constituent parts of a simulated physical body can result in an emergent behavior that can be classified as cognitive by an outside observer. Specifically, we show how simulated voxels with fixed behaviors can be combined to create a robot such that, when presented with an image of an MNIST digit zero, it moves towards the left; and when it is presented with an image of an MNIST digit one, it moves towards the right. Such robots possess what we refer to as ``morphological cognition'' -- the ability to perform cognitive behavior as a result of morphological processes. To the best of our knowledge, this is the first demonstration of a high-level mental faculty such as image classification performed by a robot without any neural circuitry. We hope that this work serves as a proof-of-concept and fosters further research into different models of intelligence. 

**Abstract (ZH)**: 基于模拟物理体的形态认知：无需神经电路的图像分类示例 

---
# Optimizing Grasping in Legged Robots: A Deep Learning Approach to Loco-Manipulation 

**Title (ZH)**: 基于深度学习的腿式机器人抓取优化：灵巧操控方法 

**Authors**: Dilermando Almeida, Guilherme Lazzarini, Juliano Negri, Thiago H. Segreto, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2508.17466)  

**Abstract**: Quadruped robots have emerged as highly efficient and versatile platforms, excelling in navigating complex and unstructured terrains where traditional wheeled robots might fail. Equipping these robots with manipulator arms unlocks the advanced capability of loco-manipulation to perform complex physical interaction tasks in areas ranging from industrial automation to search-and-rescue missions. However, achieving precise and adaptable grasping in such dynamic scenarios remains a significant challenge, often hindered by the need for extensive real-world calibration and pre-programmed grasp configurations. This paper introduces a deep learning framework designed to enhance the grasping capabilities of quadrupeds equipped with arms, focusing on improved precision and adaptability. Our approach centers on a sim-to-real methodology that minimizes reliance on physical data collection. We developed a pipeline within the Genesis simulation environment to generate a synthetic dataset of grasp attempts on common objects. By simulating thousands of interactions from various perspectives, we created pixel-wise annotated grasp-quality maps to serve as the ground truth for our model. This dataset was used to train a custom CNN with a U-Net-like architecture that processes multi-modal input from an onboard RGB and depth cameras, including RGB images, depth maps, segmentation masks, and surface normal maps. The trained model outputs a grasp-quality heatmap to identify the optimal grasp point. We validated the complete framework on a four-legged robot. The system successfully executed a full loco-manipulation task: autonomously navigating to a target object, perceiving it with its sensors, predicting the optimal grasp pose using our model, and performing a precise grasp. This work proves that leveraging simulated training with advanced sensing offers a scalable and effective solution for object handling. 

**Abstract (ZH)**: 基于仿真的四足机器人 manipulator 手臂 grasping 能力增强：精准与适应性并重的方法 

---
# Evolutionary Brain-Body Co-Optimization Consistently Fails to Select for Morphological Potential 

**Title (ZH)**: 进化性的脑体共优化一致性地未能选择出形态潜力。 

**Authors**: Alican Mertan, Nick Cheney  

**Link**: [PDF](https://arxiv.org/pdf/2508.17464)  

**Abstract**: Brain-body co-optimization remains a challenging problem, despite increasing interest from the community in recent years. To understand and overcome the challenges, we propose exhaustively mapping a morphology-fitness landscape to study it. To this end, we train controllers for each feasible morphology in a design space of 1,305,840 distinct morphologies, constrained by a computational budget. First, we show that this design space constitutes a good model for studying the brain-body co-optimization problem, and our attempt to exhaustively map it roughly captures the landscape. We then proceed to analyze how evolutionary brain-body co-optimization algorithms work in this design space. The complete knowledge of the morphology-fitness landscape facilitates a better understanding of the results of evolutionary brain-body co-optimization algorithms and how they unfold over evolutionary time in the morphology space. This investigation shows that the experimented algorithms cannot consistently find near-optimal solutions. The search, at times, gets stuck on morphologies that are sometimes one mutation away from better morphologies, and the algorithms cannot efficiently track the fitness gradient in the morphology-fitness landscape. We provide evidence that experimented algorithms regularly undervalue the fitness of individuals with newly mutated bodies and, as a result, eliminate promising morphologies throughout evolution. Our work provides the most concrete demonstration of the challenges of evolutionary brain-body co-optimization. Our findings ground the trends in the literature and provide valuable insights for future work. 

**Abstract (ZH)**: 脑体协同优化仍然是一个具有挑战性的问题，尽管近年来该领域引起了越来越多社区的关注。为了了解和克服这些挑战，我们提出全面映射形态-适应度景观以研究该问题。为此，我们在一个包含1,305,840种不同形态的设计空间中，受到计算预算的限制，为每种可行的形态训练控制器。首先，我们展示了该设计空间构成研究脑体协同优化问题的良好模型，我们尝试全面映射它大致捕捉了该景观。然后，我们分析了在该设计空间中进化脑体协同优化算法的工作原理。全面了解形态-适应度景观有助于更好地理解进化脑体协同优化算法的结果及其在形态空间中的演化过程。这项研究显示，所试验的算法无法一致地找到近似最优解。搜索有时会在接近更好形态但仅相差一个突变的形态上停滞，且算法无法有效地追踪形态-适应度景观中的适应度梯度。我们提供了证据表明，所试验的算法经常低估新突变体型个体的适应度，并因此在整个进化过程中淘汰有前途的形态。我们工作提供了进化脑体协同优化挑战的最直接证据。我们的发现为文献中的趋势提供了依据，并为进一步研究提供了宝贵的见解。 

---
# Robotic Manipulation via Imitation Learning: Taxonomy, Evolution, Benchmark, and Challenges 

**Title (ZH)**: 基于模仿学习的机器人操作：分类、演进、基准与挑战 

**Authors**: Zezeng Li, Alexandre Chapin, Enda Xiang, Rui Yang, Bruno Machado, Na Lei, Emmanuel Dellandrea, Di Huang, Liming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.17449)  

**Abstract**: Robotic Manipulation (RM) is central to the advancement of autonomous robots, enabling them to interact with and manipulate objects in real-world environments. This survey focuses on RM methodologies that leverage imitation learning, a powerful technique that allows robots to learn complex manipulation skills by mimicking human demonstrations. We identify and analyze the most influential studies in this domain, selected based on community impact and intrinsic quality. For each paper, we provide a structured summary, covering the research purpose, technical implementation, hierarchical classification, input formats, key priors, strengths and limitations, and citation metrics. Additionally, we trace the chronological development of imitation learning techniques within RM policy (RMP), offering a timeline of key technological advancements. Where available, we report benchmark results and perform quantitative evaluations to compare existing methods. By synthesizing these insights, this review provides a comprehensive resource for researchers and practitioners, highlighting both the state of the art and the challenges that lie ahead in the field of robotic manipulation through imitation learning. 

**Abstract (ZH)**: 机器人操作中的模仿学习方法综述：从实世界环境中物体的交互与操作到基于模仿学习的机器人操作方法学的研究 

---
# OVITA: Open-Vocabulary Interpretable Trajectory Adaptations 

**Title (ZH)**: OVITA: 开词汇量可解释轨迹适应 

**Authors**: Anurag Maurya, Tashmoy Ghosh, Anh Nguyen, Ravi Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2508.17260)  

**Abstract**: Adapting trajectories to dynamic situations and user preferences is crucial for robot operation in unstructured environments with non-expert users. Natural language enables users to express these adjustments in an interactive manner. We introduce OVITA, an interpretable, open-vocabulary, language-driven framework designed for adapting robot trajectories in dynamic and novel situations based on human instructions. OVITA leverages multiple pre-trained Large Language Models (LLMs) to integrate user commands into trajectories generated by motion planners or those learned through demonstrations. OVITA employs code as an adaptation policy generated by an LLM, enabling users to adjust individual waypoints, thus providing flexible control. Another LLM, which acts as a code explainer, removes the need for expert users, enabling intuitive interactions. The efficacy and significance of the proposed OVITA framework is demonstrated through extensive simulations and real-world environments with diverse tasks involving spatiotemporal variations on heterogeneous robotic platforms such as a KUKA IIWA robot manipulator, Clearpath Jackal ground robot, and CrazyFlie drone. 

**Abstract (ZH)**: 适配动态情况和用户偏好的轨迹调整对于在非结构化环境中由非专家用户操作的机器人至关重要。自然语言使用户能够以交互方式表达这些调整。我们提出了OVITA，一种基于人类指令、可解释的、含开放词汇表的、语言驱动的框架，用于在动态和新颖情况下调整机器人轨迹。OVITA利用多个预训练的大规模语言模型（LLMs）将用户命令整合到路径规划器生成的轨迹或通过示范学习的轨迹中。OVITA采用由LLM生成的代码作为适应策略，允许用户调整单个航点，从而提供灵活控制。另一个LLM作为代码解释器，消除了对专家用户的需求，使交互更加直观。通过广泛的仿真实验和涉及多种任务、异构机器人平台（如KUKA IIWA机械臂、Clearpath Jackal地面机器人和CrazyFlie无人机）的时空变化的实际环境，展示了提出的OVITA框架的有效性和重要性。 

---
# LaGarNet: Goal-Conditioned Recurrent State-Space Models for Pick-and-Place Garment Flattening 

**Title (ZH)**: LaGarNet: 基于目标条件的递归状态空间模型用于衣物整理的拾取与放置 

**Authors**: Halid Abdulrahim Kadi, Kasim Terzić  

**Link**: [PDF](https://arxiv.org/pdf/2508.17070)  

**Abstract**: We present a novel goal-conditioned recurrent state space (GC-RSSM) model capable of learning latent dynamics of pick-and-place garment manipulation. Our proposed method LaGarNet matches the state-of-the-art performance of mesh-based methods, marking the first successful application of state-space models on complex garments. LaGarNet trains on a coverage-alignment reward and a dataset collected through a general procedure supported by a random policy and a diffusion policy learned from few human demonstrations; it substantially reduces the inductive biases introduced in the previous similar methods. We demonstrate that a single-policy LaGarNet achieves flattening on four different types of garments in both real-world and simulation settings. 

**Abstract (ZH)**: 我们提出了一种新颖的基于目标条件的递归状态空间（GC-RSSM）模型，该模型能够学习拾取和放置服装操作的潜在动力学。我们提出的LaGarNet方法匹配了基于网格方法的最先进性能，标志着状态空间模型首次成功应用于复杂服装。LaGarNet通过覆盖对齐奖励在由随机策略和支持的有限人类示范学习的扩散策略支持的一般程序下进行训练，显著降低了先前类似方法引入的归纳偏置。我们展示了单策略的LaGarNet在现实世界和仿真环境中均实现了四种不同类型的服装的平整化。 

---
# A Rapid Iterative Trajectory Planning Method for Automated Parking through Differential Flatness 

**Title (ZH)**: 基于微分平坦性的快速迭代轨迹规划方法用于自动停车 

**Authors**: Zhouheng Li, Lei Xie, Cheng Hu, Hongye Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.17038)  

**Abstract**: As autonomous driving continues to advance, automated parking is becoming increasingly essential. However, significant challenges arise when implementing path velocity decomposition (PVD) trajectory planning for automated parking. The primary challenge is ensuring rapid and precise collision-free trajectory planning, which is often in conflict. The secondary challenge involves maintaining sufficient control feasibility of the planned trajectory, particularly at gear shifting points (GSP). This paper proposes a PVD-based rapid iterative trajectory planning (RITP) method to solve the above challenges. The proposed method effectively balances the necessity for time efficiency and precise collision avoidance through a novel collision avoidance framework. Moreover, it enhances the overall control feasibility of the planned trajectory by incorporating the vehicle kinematics model and including terminal smoothing constraints (TSC) at GSP during path planning. Specifically, the proposed method leverages differential flatness to ensure the planned path adheres to the vehicle kinematic model. Additionally, it utilizes TSC to maintain curvature continuity at GSP, thereby enhancing the control feasibility of the overall trajectory. The simulation results demonstrate superior time efficiency and tracking errors compared to model-integrated and other iteration-based trajectory planning methods. In the real-world experiment, the proposed method was implemented and validated on a ROS-based vehicle, demonstrating the applicability of the RITP method for real vehicles. 

**Abstract (ZH)**: 随着自动驾驶技术的不断进步，自动泊车变得 increasingly 重要。然而，在实施路径速度分解（PVD）轨迹规划时，自动泊车面临着显著的挑战。主要挑战是确保快速而精确的无碰撞轨迹规划，这往往存在冲突。次要挑战在于在换挡点（GSP）保持计划轨迹的充分控制可行性。本文提出了一种基于PVD的快速迭代轨迹规划（RITP）方法来解决上述挑战。所提出的方法通过一种新的碰撞规避框架有效地平衡了时间效率和精确碰撞规避的必要性。此外，通过结合车辆运动学模型并在路径规划过程中在换挡点（GSP）纳入终端平滑约束（TSC），该方法还增强了计划轨迹的整体控制可行性。具体而言，所提出的方法利用微分平坦性确保计划路径遵循车辆运动学模型。此外，利用TSC保持换挡点处的曲率连续性，从而提高整体轨迹的控制可行性。仿真结果表明，与集成模型和其他基于迭代的轨迹规划方法相比，该方法具有更好的时间效率和跟踪误差。在实际试验中，所提出的方法已在基于ROS的车辆上实现并验证，展示了RITP方法在实际车辆上的适用性。 

---
# DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration 

**Title (ZH)**: 双空间过滤与强化刚性注册 

**Authors**: Jiayi Li, Yuxin Yao, Qiuhang Lu, Juyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17034)  

**Abstract**: Rigid registration, aiming to estimate a rigid transformation to align source and target data, play a crucial role in applications such as SLAM and 3D reconstruction. However, noisy, partially overlapping data and the need for real-time processing pose major challenges for rigid registration. Considering that feature-based matching can handle large transformation differences but suffers from limited accuracy, while local geometry-based matching can achieve fine-grained local alignment but relies heavily on a good initial transformation, we propose a novel dual-space paradigm to fully leverage the strengths of both approaches. First, we introduce an efficient filtering mechanism that incorporates a computationally lightweight single-point RANSAC algorithm followed by a refinement module to eliminate unreliable feature-based correspondences. Subsequently, we treat filtered correspondences as anchor points, extract geometric proxies, and formulates an effective objective function with a tailored solver to estimate the transformation. Experiments verify our method's effectiveness, as shown by achieving up to a 32x CPU-time speedup over MAC on KITTI with comparable accuracy. 

**Abstract (ZH)**: 基于双空间 paradigm 的刚体注册方法：融合特征匹配与局部几何匹配优势 

---
# LLM-based Human-like Traffic Simulation for Self-driving Tests 

**Title (ZH)**: 基于LLM的人类似交通模拟用于自动驾驶测试 

**Authors**: Wendi Li, Hao Wu, Han Gao, Bing Mao, Fengyuan Xu, Sheng Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2508.16962)  

**Abstract**: Ensuring realistic traffic dynamics is a prerequisite for simulation platforms to evaluate the reliability of self-driving systems before deployment in the real world. Because most road users are human drivers, reproducing their diverse behaviors within simulators is vital. Existing solutions, however, typically rely on either handcrafted heuristics or narrow data-driven models, which capture only fragments of real driving behaviors and offer limited driving style diversity and interpretability. To address this gap, we introduce HDSim, an HD traffic generation framework that combines cognitive theory with large language model (LLM) assistance to produce scalable and realistic traffic scenarios within simulation platforms. The framework advances the state of the art in two ways: (i) it introduces a hierarchical driver model that represents diverse driving style traits, and (ii) it develops a Perception-Mediated Behavior Influence strategy, where LLMs guide perception to indirectly shape driver actions. Experiments reveal that embedding HDSim into simulation improves detection of safety-critical failures in self-driving systems by up to 68% and yields realism-consistent accident interpretability. 

**Abstract (ZH)**: 确保真实的交通动态是评估自动驾驶系统可靠性的仿真平台在实际部署前的先决条件。由于大多数道路使用者是人类驾驶员，因此在模拟器中重现其多样行为至关重要。现有解决方案通常依赖于手工构建的启发式方法或狭窄的数据驱动模型，这些方法只能捕捉现实驾驶行为的碎片，并且提供的驾驶风格多样性有限且可解释性差。为解决这一问题，我们引入了HDSim，这是一种结合认知理论和大型语言模型（LLM）辅助的高精度交通生成框架，可在仿真平台中生成可扩展且真实的交通场景。该框架以两种方式推动了技术进步：（i）引入了一种分层驾驶员模型，以表示多样的驾驶风格特征；（ii）开发了一种感知中介行为影响策略，其中LLM引导感知以间接塑造驾驶员行为。实验表明，将HDSim嵌入仿真可以将自动驾驶系统中安全关键故障的检测率提高多达68%，并提供与现实一致的事故可解释性。 

---
# Drive As You Like: Strategy-Level Motion Planning Based on A Multi-Head Diffusion Model 

**Title (ZH)**: 随心驾驶：基于多头扩散模型的策略级运动规划 

**Authors**: Fan Ding, Xuewen Luo, Hwa Hui Tew, Ruturaj Reddy, Xikun Wang, Junn Yong Loo  

**Link**: [PDF](https://arxiv.org/pdf/2508.16947)  

**Abstract**: Recent advances in motion planning for autonomous driving have led to models capable of generating high-quality trajectories. However, most existing planners tend to fix their policy after supervised training, leading to consistent but rigid driving behaviors. This limits their ability to reflect human preferences or adapt to dynamic, instruction-driven demands. In this work, we propose a diffusion-based multi-head trajectory planner(M-diffusion planner). During the early training stage, all output heads share weights to learn to generate high-quality trajectories. Leveraging the probabilistic nature of diffusion models, we then apply Group Relative Policy Optimization (GRPO) to fine-tune the pre-trained model for diverse policy-specific behaviors. At inference time, we incorporate a large language model (LLM) to guide strategy selection, enabling dynamic, instruction-aware planning without switching models. Closed-loop simulation demonstrates that our post-trained planner retains strong planning capability while achieving state-of-the-art (SOTA) performance on the nuPlan val14 benchmark. Open-loop results further show that the generated trajectories exhibit clear diversity, effectively satisfying multi-modal driving behavior requirements. The code and related experiments will be released upon acceptance of the paper. 

**Abstract (ZH)**: Recent Advances in Motion Planning for Autonomous Driving Have Led to Models Capable of Generating High-Quality Trajectories: A Diffusion-Based Multi-Head Trajectory Planner (M-diffusion Planner) with Dynamic Policy Adaptation 

---
# HumanoidVerse: A Versatile Humanoid for Vision-Language Guided Multi-Object Rearrangement 

**Title (ZH)**: HumanoidVerse: 一种适用于视觉-语言引导多物体重排的多功能人形机器人 

**Authors**: Haozhuo Zhang, Jingkai Sun, Michele Caprio, Jian Tang, Shanghang Zhang, Qiang Zhang, Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.16943)  

**Abstract**: We introduce HumanoidVerse, a novel framework for vision-language guided humanoid control that enables a single physically simulated robot to perform long-horizon, multi-object rearrangement tasks across diverse scenes. Unlike prior methods that operate in fixed settings with single-object interactions, our approach supports consecutive manipulation of multiple objects, guided only by natural language instructions and egocentric camera RGB observations. HumanoidVerse is trained via a multi-stage curriculum using a dual-teacher distillation pipeline, enabling fluid transitions between sub-tasks without requiring environment resets. To support this, we construct a large-scale dataset comprising 350 multi-object tasks spanning four room layouts. Extensive experiments in the Isaac Gym simulator demonstrate that our method significantly outperforms prior state-of-the-art in both task success rate and spatial precision, and generalizes well to unseen environments and instructions. Our work represents a key step toward robust, general-purpose humanoid agents capable of executing complex, sequential tasks under real-world sensory constraints. The video visualization results can be found on the project page: this https URL. 

**Abstract (ZH)**: 基于视觉语言引导的人形机器人控制新框架：HumanoidVerse跨场景多对象重排任务的研究 

---
# Relative Navigation and Dynamic Target Tracking for Autonomous Underwater Proximity Operations 

**Title (ZH)**: 自主水下近距离操作中的相对导航与动态目标跟踪 

**Authors**: David Baxter, Aldo Terán Espinoza, Antonio Terán Espinoza, Amy Loutfi, John Folkesson, Peter Sigray, Stephanie Lowry, Jakob Kuttenkeuler  

**Link**: [PDF](https://arxiv.org/pdf/2508.16901)  

**Abstract**: Estimating a target's 6-DoF motion in underwater proximity operations is difficult because the chaser lacks target-side proprioception and the available relative observations are sparse, noisy, and often partial (e.g., Ultra-Short Baseline (USBL) positions). Without a motion prior, factor-graph maximum a posteriori estimation is underconstrained: consecutive target states are weakly linked and orientation can drift. We propose a generalized constant-twist motion prior defined on the tangent space of Lie groups that enforces temporally consistent trajectories across all degrees of freedom; in SE(3) it couples translation and rotation in the body frame. We present a ternary factor and derive its closed-form Jacobians based on standard Lie group operations, enabling drop-in use for trajectories on arbitrary Lie groups. We evaluate two deployment modes: (A) an SE(3)-only representation that regularizes orientation even when only position is measured, and (B) a mode with boundary factors that switches the target representation between SE(3) and 3D position while applying the same generalized constant-twist prior across representation changes. Validation on a real-world dynamic docking scenario dataset shows consistent ego-target trajectory estimation through USBL-only and optical relative measurement segments with an improved relative tracking accuracy compared to the noisy measurements to the target. Because the construction relies on standard Lie group primitives, it is portable across state manifolds and sensing modalities. 

**Abstract (ZH)**: 基于Lie群切空间的通用固定旋扭转运动先验在水下近程操作中目标6-DOF运动估计 

---
# A Workflow for Map Creation in Autonomous Vehicle Simulations 

**Title (ZH)**: 自主车辆仿真中地图创建的工作流 

**Authors**: Zubair Islam, Ahmaad Ansari, George Daoud, Mohamed El-Darieby  

**Link**: [PDF](https://arxiv.org/pdf/2508.16856)  

**Abstract**: The fast development of technology and artificial intelligence has significantly advanced Autonomous Vehicle (AV) research, emphasizing the need for extensive simulation testing. Accurate and adaptable maps are critical in AV development, serving as the foundation for localization, path planning, and scenario testing. However, creating simulation-ready maps is often difficult and resource-intensive, especially with simulators like CARLA (CAR Learning to Act). Many existing workflows require significant computational resources or rely on specific simulators, limiting flexibility for developers. This paper presents a custom workflow to streamline map creation for AV development, demonstrated through the generation of a 3D map of a parking lot at Ontario Tech University. Future work will focus on incorporating SLAM technologies, optimizing the workflow for broader simulator compatibility, and exploring more flexible handling of latitude and longitude values to enhance map generation accuracy. 

**Abstract (ZH)**: 技术与人工智能的快速发展显著推动了自动驾驶车辆（AV）的研究，强调了广泛进行仿真测试的必要性。准确且灵活的地图在AV研发中至关重要，它是定位、路径规划和场景测试的基础。然而，创建适用于仿真的地图往往耗时且资源密集，尤其是使用CARLA等仿真器时。许多现有的工作流程需要大量的计算资源或依赖特定的仿真器，限制了开发者的灵活性。本文提出了一种定制的工作流来简化AV开发中的地图创建过程，并通过安大略理工大学停车场的3D地图生成过程进行了演示。未来工作将侧重于整合SLAM技术、优化工作流以扩大其与更广泛仿真器的兼容性，并探索更多灵活处理纬度和经度值的方法，以提高地图生成的准确性。 

---
# Autonomous UAV Flight Navigation in Confined Spaces: A Reinforcement Learning Approach 

**Title (ZH)**: 自主无人机在受限空间内的飞行导航：一种强化学习方法 

**Authors**: Marco S. Tayar, Lucas K. de Oliveira, Juliano D. Negri, Thiago H. Segreto, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2508.16807)  

**Abstract**: Inspecting confined industrial infrastructure, such as ventilation shafts, is a hazardous and inefficient task for humans. Unmanned Aerial Vehicles (UAVs) offer a promising alternative, but GPS-denied environments require robust control policies to prevent collisions. Deep Reinforcement Learning (DRL) has emerged as a powerful framework for developing such policies, and this paper provides a comparative study of two leading DRL algorithms for this task: the on-policy Proximal Policy Optimization (PPO) and the off-policy Soft Actor-Critic (SAC). The training was conducted with procedurally generated duct environments in Genesis simulation environment. A reward function was designed to guide a drone through a series of waypoints while applying a significant penalty for collisions. PPO learned a stable policy that completed all evaluation episodes without collision, producing smooth trajectories. By contrast, SAC consistently converged to a suboptimal behavior that traversed only the initial segments before failure. These results suggest that, in hazard-dense navigation, the training stability of on-policy methods can outweigh the nominal sample efficiency of off-policy algorithms. More broadly, the study provides evidence that procedurally generated, high-fidelity simulations are effective testbeds for developing and benchmarking robust navigation policies. 

**Abstract (ZH)**: 基于政策的Proximal Policy Optimization方法在受限工业基础设施检查中的稳定性优于基于软目标吸收器的Soft Actor-Critic方法：来自程序生成高保真模拟的有效导航策略研究 

---
# A Dataset and Benchmark for Robotic Cloth Unfolding Grasp Selection: The ICRA 2024 Cloth Competition 

**Title (ZH)**: 用于机器人布料展开抓取选择的数据集与基准：ICRA 2024 布料竞赛 

**Authors**: Victor-Louis De Gusseme, Thomas Lips, Remko Proesmans, Julius Hietala, Giwan Lee, Jiyoung Choi, Jeongil Choi, Geon Kim, Phayuth Yonrith, Domen Tabernik, Andrej Gams, Peter Nimac, Matej Urbas, Jon Muhovič, Danijel Skočaj, Matija Mavsar, Hyojeong Yu, Minseo Kwon, Young J. Kim, Yang Cong, Ronghan Chen, Yu Ren, Supeng Diao, Jiawei Weng, Jiayue Liu, Haoran Sun, Linhan Yang, Zeqing Zhang, Ning Guo, Lei Yang, Fang Wan, Chaoyang Song, Jia Pan, Yixiang Jin, Yong A, Jun Shi, Dingzhe Li, Yong Yang, Kakeru Yamasaki, Takumi Kajiwara, Yuki Nakadera, Krati Saxena, Tomohiro Shibata, Chongkun Xia, Kai Mo, Yanzhao Yu, Qihao Lin, Binqiang Ma, Uihun Sagong, JungHyun Choi, JeongHyun Park, Dongwoo Lee, Yeongmin Kim, Myun Joong Hwang, Yusuke Kuribayashi, Naoki Hiratsuka, Daisuke Tanaka, Solvi Arnold, Kimitoshi Yamazaki, Carlos Mateo-Agullo, Andreas Verleysen, Francis Wyffels  

**Link**: [PDF](https://arxiv.org/pdf/2508.16749)  

**Abstract**: Robotic cloth manipulation suffers from a lack of standardized benchmarks and shared datasets for evaluating and comparing different approaches. To address this, we created a benchmark and organized the ICRA 2024 Cloth Competition, a unique head-to-head evaluation focused on grasp pose selection for in-air robotic cloth unfolding. Eleven diverse teams participated in the competition, utilizing our publicly released dataset of real-world robotic cloth unfolding attempts and a variety of methods to design their unfolding approaches. Afterwards, we also expanded our dataset with 176 competition evaluation trials, resulting in a dataset of 679 unfolding demonstrations across 34 garments. Analysis of the competition results revealed insights about the trade-off between grasp success and coverage, the surprisingly strong achievements of hand-engineered methods and a significant discrepancy between competition performance and prior work, underscoring the importance of independent, out-of-the-lab evaluation in robotic cloth manipulation. The associated dataset is a valuable resource for developing and evaluating grasp selection methods, particularly for learning-based approaches. We hope that our benchmark, dataset and competition results can serve as a foundation for future benchmarks and drive further progress in data-driven robotic cloth manipulation. The dataset and benchmarking code are available at this https URL. 

**Abstract (ZH)**: 机器人布料操作缺乏标准化基准和共享数据集来评估和比较不同方法。为解决这一问题，我们创建了一个基准并组织了ICRA 2024布料竞赛，这是一个专注于空中机器人布料展开中抓取姿态选择的unique头对头评估。来自世界各地的十一支队伍参与了竞赛，利用我们公开发布的实际机器人布料展开数据集以及各种方法设计其展开策略。随后，我们还扩充了数据集，增加了176次竞赛评估试验，共包含34件服装的679次展开演示。竞赛结果分析揭示了抓取成功率与覆盖范围之间的权衡，手工程设计方法的惊人表现，以及竞赛表现与之前工作之间的显著差异，强调了在机器人布料操作中独立于实验室外评估的重要性。该相关数据集是开发和评估抓取选择方法（特别是基于学习的方法）的重要资源。我们希望我们的基准、数据集和竞赛结果能为未来基准的建立提供基础，并推动数据驱动的机器人布料操作的进一步进步。数据集和基准代码可在以下链接获取：this https URL。 

---
# COSMO-Bench: A Benchmark for Collaborative SLAM Optimization 

**Title (ZH)**: COSMO-Bench: 一种协作SLAM优化基准 

**Authors**: Daniel McGann, Easton R. Potokar, Michael Kaess  

**Link**: [PDF](https://arxiv.org/pdf/2508.16731)  

**Abstract**: Recent years have seen a focus on research into distributed optimization algorithms for multi-robot Collaborative Simultaneous Localization and Mapping (C-SLAM). Research in this domain, however, is made difficult by a lack of standard benchmark datasets. Such datasets have been used to great effect in the field of single-robot SLAM, and researchers focused on multi-robot problems would benefit greatly from dedicated benchmark datasets. To address this gap, we design and release the Collaborative Open-Source Multi-robot Optimization Benchmark (COSMO-Bench) -- a suite of 24 datasets derived from a state-of-the-art C-SLAM front-end and real-world LiDAR data. Data DOI: this https URL 

**Abstract (ZH)**: 近年来，分布式优化算法在多robot协作的同时定位与建图（C-SLAM）研究中受到了广泛关注。然而，由于缺乏标准基准数据集，该领域的研究面临较大困难。单robot SLAM领域中已经有效地应用了此类数据集，专注于多robot问题的研究人员将受益于专门的数据集。为填补这一空白，我们设计并发布了协作开源多robot优化基准（COSMO-Bench）——一套基于最先进的C-SLAM前端和真实LiDAR数据的24个数据集。数据DOI: this https URL。 

---
# BirdRecorder's AI on Sky: Safeguarding birds of prey by detection and classification of tiny objects around wind turbines 

**Title (ZH)**: BirdRecorder的AI翱翔于天空：通过风力发电机周围小型物体的检测与分类保护猛禽 

**Authors**: Nico Klar, Nizam Gifary, Felix P. G. Ziegler, Frank Sehnke, Anton Kaifel, Eric Price, Aamir Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2508.18136)  

**Abstract**: The urgent need for renewable energy expansion, particularly wind power, is hindered by conflicts with wildlife conservation. To address this, we developed BirdRecorder, an advanced AI-based anti-collision system to protect endangered birds, especially the red kite (Milvus milvus). Integrating robotics, telemetry, and high-performance AI algorithms, BirdRecorder aims to detect, track, and classify avian species within a range of 800 m to minimize bird-turbine collisions.
BirdRecorder integrates advanced AI methods with optimized hardware and software architectures to enable real-time image processing. Leveraging Single Shot Detector (SSD) for detection, combined with specialized hardware acceleration and tracking algorithms, our system achieves high detection precision while maintaining the speed necessary for real-time decision-making. By combining these components, BirdRecorder outperforms existing approaches in both accuracy and efficiency.
In this paper, we summarize results on field tests and performance of the BirdRecorder system. By bridging the gap between renewable energy expansion and wildlife conservation, BirdRecorder contributes to a more sustainable coexistence of technology and nature. 

**Abstract (ZH)**: 迫切需要扩展可再生能源，尤其是风能，但受到了与野生动物保护之间的冲突的阻碍。为了解决这一问题，我们开发了BirdRecorder，一种基于先进AI的防撞系统，旨在保护濒危鸟类，尤其是红尾鹰（Milvus milvus）。BirdRecorder 结合了机器人技术、遥感和高性能AI算法，旨在在800米范围内检测、跟踪和分类鸟类，以最小化鸟与风力涡轮机的碰撞。

BirdRecorder 将先进的AI方法与优化的硬件和软件架构相结合，实现实时图像处理。利用单-shot检测器（SSD）进行检测，并结合专门的硬件加速和跟踪算法，我们的系统在保持足够速度以进行即时决策的同时，实现了高检测精度。通过结合这些组件，BirdRecorder 在准确性和效率上均优于现有方法。

在本文中，我们总结了BirdRecorder系统在实地测试中的结果和性能。通过弥合可再生能源扩展与野生动物保护之间的差距，BirdRecorder 促进了技术与自然之间更可持续的共存。 

---
# Neural Algorithmic Reasoners informed Large Language Model for Multi-Agent Path Finding 

**Title (ZH)**: 神经算法推理引导的大型语言模型在多代理路径寻找中的应用 

**Authors**: Pu Feng, Size Wang, Yuhong Cao, Junkang Liang, Rongye Shi, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17971)  

**Abstract**: The development and application of large language models (LLM) have demonstrated that foundational models can be utilized to solve a wide array of tasks. However, their performance in multi-agent path finding (MAPF) tasks has been less than satisfactory, with only a few studies exploring this area. MAPF is a complex problem requiring both planning and multi-agent coordination. To improve the performance of LLM in MAPF tasks, we propose a novel framework, LLM-NAR, which leverages neural algorithmic reasoners (NAR) to inform LLM for MAPF. LLM-NAR consists of three key components: an LLM for MAPF, a pre-trained graph neural network-based NAR, and a cross-attention mechanism. This is the first work to propose using a neural algorithmic reasoner to integrate GNNs with the map information for MAPF, thereby guiding LLM to achieve superior performance. LLM-NAR can be easily adapted to various LLM models. Both simulation and real-world experiments demonstrate that our method significantly outperforms existing LLM-based approaches in solving MAPF problems. 

**Abstract (ZH)**: 大型语言模型的发展及其应用表明，基础模型可以用于解决广泛的任务。然而，在多智能体路径规划（MAPF）任务上的表现不尽如人意，相关研究较少。MAPF是一个需要规划和多智能体协调的复杂问题。为了提高大型语言模型在MAPF任务中的表现，我们提出了一种名为LLM-NAR的新框架，该框架利用神经算法推理器（NAR）来指导大型语言模型解决MAPF问题。LLM-NAR包括三个关键组件：一个用于MAPF的大型语言模型、一个基于图神经网络的预训练神经算法推理器以及一种交叉注意力机制。这是首次提出使用神经算法推理器将GNN与地图信息结合起来解决MAPF问题的研究，从而引导大型语言模型实现更好的性能。LLM-NAR可以轻松适应各种大型语言模型。仿真与实际实验均表明，我们的方法在解决MAPF问题上显著优于现有的基于大型语言模型的方法。 

---
# Physical Embodiment Enables Information Processing Beyond Explicit Sensing in Active Matter 

**Title (ZH)**: 物理体型使活性物质中的信息处理超越显式传感成为可能 

**Authors**: Diptabrata Paul, Nikola Milosevic, Nico Scherf, Frank Cichos  

**Link**: [PDF](https://arxiv.org/pdf/2508.17921)  

**Abstract**: Living microorganisms have evolved dedicated sensory machinery to detect environmental perturbations, processing these signals through biochemical networks to guide behavior. Replicating such capabilities in synthetic active matter remains a fundamental challenge. Here, we demonstrate that synthetic active particles can adapt to hidden hydrodynamic perturbations through physical embodiment alone, without explicit sensing mechanisms. Using reinforcement learning to control self-thermophoretic particles, we show that they learn navigation strategies to counteract unobserved flow fields by exploiting information encoded in their physical dynamics. Remarkably, particles successfully navigate perturbations that are not included in their state inputs, revealing that embodied dynamics can serve as an implicit sensing mechanism. This discovery establishes physical embodiment as a computational resource for information processing in active matter, with implications for autonomous microrobotic systems and bio-inspired computation. 

**Abstract (ZH)**: 活化的微生物已经进化出专门的感测装置来探测环境变化，并通过生物化学网络处理这些信号以指导行为。在合成活性物质中复制这种能力仍然是一个基本挑战。在这里，我们证明合成活性颗粒仅通过物理体现就可以适应隐藏的水动力扰动，而无需明确的感测机制。通过使用强化学习来控制自热泳颗粒，我们展示它们通过利用其物理动力学中编码的信息学习导航策略以对抗未观察到的流场。令人惊讶的是，颗粒成功导航了不在其状态输入中的扰动，表明物理动力学可以作为隐式感测机制。这一发现将物理体现确立为活性物质中信息处理的计算资源，并对自主微机器人系统和生物启发计算具有重要意义。 

---
# SoK: Cybersecurity Assessment of Humanoid Ecosystem 

**Title (ZH)**: SoK: 人类身态生态系统网络安全评估 

**Authors**: Priyanka Prakash Surve, Asaf Shabtai, Yuval Elovici  

**Link**: [PDF](https://arxiv.org/pdf/2508.17481)  

**Abstract**: Humanoids are progressing toward practical deployment across healthcare, industrial, defense, and service sectors. While typically considered cyber-physical systems (CPSs), their dependence on traditional networked software stacks (e.g., Linux operating systems), robot operating system (ROS) middleware, and over-the-air update channels, creates a distinct security profile that exposes them to vulnerabilities conventional CPS models do not fully address. Prior studies have mainly examined specific threats, such as LiDAR spoofing or adversarial machine learning (AML). This narrow focus overlooks how an attack targeting one component can cascade harm throughout the robot's interconnected systems. We address this gap through a systematization of knowledge (SoK) that takes a comprehensive approach, consolidating fragmented research from robotics, CPS, and network security domains. We introduce a seven-layer security model for humanoid robots, organizing 39 known attacks and 35 defenses across the humanoid ecosystem-from hardware to human-robot interaction. Building on this security model, we develop a quantitative 39x35 attack-defense matrix with risk-weighted scoring, validated through Monte Carlo analysis. We demonstrate our method by evaluating three real-world robots: Pepper, G1 EDU, and Digit. The scoring analysis revealed varying security maturity levels, with scores ranging from 39.9% to 79.5% across the platforms. This work introduces a structured, evidence-based assessment method that enables systematic security evaluation, supports cross-platform benchmarking, and guides prioritization of security investments in humanoid robotics. 

**Abstract (ZH)**: 类人机器人在医疗、工业、国防和服务领域中的实践部署正在不断进步。虽然类人机器人通常被视为网络物理系统（CPS），但它们依赖于传统的网络软件栈（如Linux操作系统）、机器人操作系统（ROS）中间件以及空中更新通道，这些特性为其带来了不同于传统CPS模型的安全特性，并使其暴露于常规CPS模型未能充分解决的漏洞中。之前的研究主要关注特定威胁，如激光雷达欺骗或对抗性机器学习（AML）。这种狭窄的视角忽视了针对一个组件的攻击如何在机器人相互连接的系统中引发连锁反应。我们通过系统知识综合（SoK）研究，采用全面的方法，将来自机器人学、CPS和网络安全部门的零散研究整合起来。我们为类人机器人引入了一个七层安全模型，将39种已知攻击和35种防御措施按硬件到人机交互的全生态系统进行了组织。基于该安全模型，我们开发了一个量化39×35攻击-防御矩阵，并通过蒙特卡洛分析进行了验证。通过评估三个实际机器人（Pepper、G1 EDU和Digit）来展示我们的方法，评分分析显示不同平台的安全部成熟度存在差异，得分范围从39.9%到79.5%不等。该项工作引入了一种结构化、基于证据的评估方法，能够实现系统的安全评估、跨平台基准测试，并指导类人机器人安全投资的优先级确定。 

---
# A Synthetic Dataset for Manometry Recognition in Robotic Applications 

**Title (ZH)**: 一种用于机器人应用中的食道测压识别合成数据集 

**Authors**: Pedro Antonio Rabelo Saraiva, Enzo Ferreira de Souza, Joao Manoel Herrera Pinheiro, Thiago H. Segreto, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2508.17468)  

**Abstract**: This work addresses the challenges of data scarcity and high acquisition costs for training robust object detection models in complex industrial environments, such as offshore oil platforms. The practical and economic barriers to collecting real-world data in these hazardous settings often hamper the development of autonomous inspection systems. To overcome this, in this work we propose and validate a hybrid data synthesis pipeline that combines procedural rendering with AI-driven video generation. Our methodology leverages BlenderProc to create photorealistic images with precise annotations and controlled domain randomization, and integrates NVIDIA's Cosmos-Predict2 world-foundation model to synthesize physically plausible video sequences with temporal diversity, capturing rare viewpoints and adverse conditions. We demonstrate that a YOLO-based detection network trained on a composite dataset, blending real images with our synthetic data, achieves superior performance compared to models trained exclusively on real-world data. Notably, a 1:1 mixture of real and synthetic data yielded the highest accuracy, surpassing the real-only baseline. These findings highlight the viability of a synthetic-first approach as an efficient, cost-effective, and safe alternative for developing reliable perception systems in safety-critical and resource-constrained industrial applications. 

**Abstract (ZH)**: 本研究解决了在复杂工业环境（如 offshore 石油平台）中训练稳健的目标检测模型时遇到的数据稀缺性和高昂获取成本的挑战。在这些危险环境中的实际数据收集受到实践和经济的限制，往往阻碍了自主检测系统的开发。为克服这一难题，本研究提出并验证了一种结合过程渲染与 AI 驱动视频生成的混合数据合成管道。我们的方法利用 BlenderProc 创建具有精确标注和受控领域随机化的照片级真实图像，并结合 NVIDIA 的 Cosmos-Predict2 世界基础模型来合成具有时间多样性的物理上可验证的视频序列，捕捉稀有视角和不利条件。研究结果表明，基于 YOLO 的检测网络在综合数据集上训练，该数据集融合了真实图像和合成数据，相比仅使用真实世界数据训练的模型，表现出更优的性能。值得注意的是，真实数据与合成数据 1:1 的混合比例获得了最高的准确性，超过了仅使用真实数据的基线。这些发现突显了合成优先方法在安全关键和资源受限的工业应用中开发可靠感知系统方面的可行性和经济高效性及安全性。 

---
# Robust Point Cloud Registration via Geometric Overlapping Guided Rotation Search 

**Title (ZH)**: 基于几何重叠引导旋转搜索的鲁棒点云注册 

**Authors**: Zhao Zheng, Jingfan Fan, Long Shao, Hong Song, Danni Ai, Tianyu Fu, Deqiang Xiao, Yongtian Wang, Jian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17427)  

**Abstract**: Point cloud registration based on correspondences computes the rigid transformation that maximizes the number of inliers constrained within the noise threshold. Current state-of-the-art (SOTA) methods employing spatial compatibility graphs or branch-and-bound (BnB) search mainly focus on registration under high outlier ratios. However, graph-based methods require at least quadratic space and time complexity for graph construction, while multi-stage BnB search methods often suffer from inaccuracy due to local optima between decomposed stages. This paper proposes a geometric maximum overlapping registration framework via rotation-only BnB search. The rigid transformation is decomposed using Chasles' theorem into a translation along rotation axis and a 2D rigid transformation. The optimal rotation axis and angle are searched via BnB, with residual parameters formulated as range maximum query (RMQ) problems. Firstly, the top-k candidate rotation axes are searched within a hemisphere parameterized by cube mapping, and the translation along each axis is estimated through interval stabbing of the correspondences projected onto that axis. Secondly, the 2D registration is relaxed to 1D rotation angle search with 2D RMQ of geometric overlapping for axis-aligned rectangles, which is solved deterministically in polynomial time using sweep line algorithm with segment tree. Experimental results on 3DMatch, 3DLoMatch, and KITTI datasets demonstrate superior accuracy and efficiency over SOTA methods, while the time complexity is polynomial and the space complexity increases linearly with the number of points, even in the worst case. 

**Abstract (ZH)**: 基于对应点的点云注册通过旋转 Only  branch-and-bound 搜索实现几何最大重叠变换 

---
# SEER-VAR: Semantic Egocentric Environment Reasoner for Vehicle Augmented Reality 

**Title (ZH)**: SEER-VAR: 基于语义自.cent环境推理器 for 车辆增强现实 

**Authors**: Yuzhi Lai, Shenghai Yuan, Peizheng Li, Jun Lou, Andreas Zell  

**Link**: [PDF](https://arxiv.org/pdf/2508.17255)  

**Abstract**: We present SEER-VAR, a novel framework for egocentric vehicle-based augmented reality (AR) that unifies semantic decomposition, Context-Aware SLAM Branches (CASB), and LLM-driven recommendation. Unlike existing systems that assume static or single-view settings, SEER-VAR dynamically separates cabin and road scenes via depth-guided vision-language grounding. Two SLAM branches track egocentric motion in each context, while a GPT-based module generates context-aware overlays such as dashboard cues and hazard alerts. To support evaluation, we introduce EgoSLAM-Drive, a real-world dataset featuring synchronized egocentric views, 6DoF ground-truth poses, and AR annotations across diverse driving scenarios. Experiments demonstrate that SEER-VAR achieves robust spatial alignment and perceptually coherent AR rendering across varied environments. As one of the first to explore LLM-based AR recommendation in egocentric driving, we address the lack of comparable systems through structured prompting and detailed user studies. Results show that SEER-VAR enhances perceived scene understanding, overlay relevance, and driver ease, providing an effective foundation for future research in this direction. Code and dataset will be made open source. 

**Abstract (ZH)**: SEER-VAR：一种统一语义分解、上下文感知SLAM分支和LLM驱动推荐的自车视角增强现实框架 

---
# Collaborative-Online-Learning-Enabled Distributionally Robust Motion Control for Multi-Robot Systems 

**Title (ZH)**: 基于协作在线学习的分布鲁棒多机器人系统运动控制 

**Authors**: Chao Ning, Han Wang, Longyan Li, Yang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17173)  

**Abstract**: This paper develops a novel COllaborative-Online-Learning (COOL)-enabled motion control framework for multi-robot systems to avoid collision amid randomly moving obstacles whose motion distributions are partially observable through decentralized data streams. To address the notable challenge of data acquisition due to occlusion, a COOL approach based on the Dirichlet process mixture model is proposed to efficiently extract motion distribution information by exchanging among robots selected learning structures. By leveraging the fine-grained local-moment information learned through COOL, a data-stream-driven ambiguity set for obstacle motion is constructed. We then introduce a novel ambiguity set propagation method, which theoretically admits the derivation of the ambiguity sets for obstacle positions over the entire prediction horizon by utilizing obstacle current positions and the ambiguity set for obstacle motion. Additionally, we develop a compression scheme with its safety guarantee to automatically adjust the complexity and granularity of the ambiguity set by aggregating basic ambiguity sets that are close in a measure space, thereby striking an attractive trade-off between control performance and computation time. Then the probabilistic collision-free trajectories are generated through distributionally robust optimization problems. The distributionally robust obstacle avoidance constraints based on the compressed ambiguity set are equivalently reformulated by deriving separating hyperplanes through tractable semi-definite programming. Finally, we establish the probabilistic collision avoidance guarantee and the long-term tracking performance guarantee for the proposed framework. The numerical simulations are used to demonstrate the efficacy and superiority of the proposed approach compared with state-of-the-art methods. 

**Abstract (ZH)**: 基于狄利克雷过程混合模型的COllaborative-Online-Learning (COOL) 启发的多机器人系统避碰运动控制框架 

---
# DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method 

**Title (ZH)**: DeltaFlow：一种高效的多帧场景流估计方法 

**Authors**: Qingwen Zhang, Xiaomeng Zhu, Yushan Zhang, Yixi Cai, Olov Andersson, Patric Jensfelt  

**Link**: [PDF](https://arxiv.org/pdf/2508.17054)  

**Abstract**: Previous dominant methods for scene flow estimation focus mainly on input from two consecutive frames, neglecting valuable information in the temporal domain. While recent trends shift towards multi-frame reasoning, they suffer from rapidly escalating computational costs as the number of frames grows. To leverage temporal information more efficiently, we propose DeltaFlow ($\Delta$Flow), a lightweight 3D framework that captures motion cues via a $\Delta$ scheme, extracting temporal features with minimal computational cost, regardless of the number of frames. Additionally, scene flow estimation faces challenges such as imbalanced object class distributions and motion inconsistency. To tackle these issues, we introduce a Category-Balanced Loss to enhance learning across underrepresented classes and an Instance Consistency Loss to enforce coherent object motion, improving flow accuracy. Extensive evaluations on the Argoverse 2 and Waymo datasets show that $\Delta$Flow achieves state-of-the-art performance with up to 22% lower error and $2\times$ faster inference compared to the next-best multi-frame supervised method, while also demonstrating a strong cross-domain generalization ability. The code is open-sourced at this https URL along with trained model weights. 

**Abstract (ZH)**: 基于多帧场景流估计的DeltaFlow：轻量级高效框架及优化策略 

---
# M3DMap: Object-aware Multimodal 3D Mapping for Dynamic Environments 

**Title (ZH)**: M3DMap: 具有物体意识的多模态动态环境3D地图构建 

**Authors**: Dmitry Yudin  

**Link**: [PDF](https://arxiv.org/pdf/2508.17044)  

**Abstract**: 3D mapping in dynamic environments poses a challenge for modern researchers in robotics and autonomous transportation. There are no universal representations for dynamic 3D scenes that incorporate multimodal data such as images, point clouds, and text. This article takes a step toward solving this problem. It proposes a taxonomy of methods for constructing multimodal 3D maps, classifying contemporary approaches based on scene types and representations, learning methods, and practical applications. Using this taxonomy, a brief structured analysis of recent methods is provided. The article also describes an original modular method called M3DMap, designed for object-aware construction of multimodal 3D maps for both static and dynamic scenes. It consists of several interconnected components: a neural multimodal object segmentation and tracking module; an odometry estimation module, including trainable algorithms; a module for 3D map construction and updating with various implementations depending on the desired scene representation; and a multimodal data retrieval module. The article highlights original implementations of these modules and their advantages in solving various practical tasks, from 3D object grounding to mobile manipulation. Additionally, it presents theoretical propositions demonstrating the positive effect of using multimodal data and modern foundational models in 3D mapping methods. Details of the taxonomy and method implementation are available at this https URL. 

**Abstract (ZH)**: 动态环境下的3D建图是现代机器人与自主运输领域研究人员面临的挑战。缺乏能够整合图像、点云和文本等多种模式数据的通用动态3D场景表示方法。本文朝着解决这一问题迈出了一步，提出了一种构造多模式3D地图的方法 taxonomy，基于场景类型、表示方法、学习方法和实际应用对当前方法进行分类。利用此 taxonomy，简要分析了近期的方法。文中还描述了一个原创的模块化方法 M3DMap，旨在构建适合静态和动态场景的物体感知多模式3D地图。该方法由几个相互关联的组件组成：一个神经多模式物体分割和跟踪模块；一个里程计估计模块，包括可训练算法；一个3D地图构建和更新模块，根据所需的场景表示有不同的实现方式；一个多模式数据检索模块。本文突出了这些模块的原创实现及其在从3D物体定位到移动操作等各种实际任务中的优势。此外，还提出了理论命题，展示了使用多模式数据和现代基础模型在3D建图方法中发挥的积极效果。更多详细内容和方法实现细节请参见此 <https://> 地址。 

---
# Fiducial Marker Splatting for High-Fidelity Robotics Simulations 

**Title (ZH)**: 信标标记点绘制用于高保真机器人模拟 

**Authors**: Diram Tabaa, Gianni Di Caro  

**Link**: [PDF](https://arxiv.org/pdf/2508.17012)  

**Abstract**: High-fidelity 3D simulation is critical for training mobile robots, but its traditional reliance on mesh-based representations often struggle in complex environments, such as densely packed greenhouses featuring occlusions and repetitive structures. Recent neural rendering methods, like Gaussian Splatting (GS), achieve remarkable visual realism but lack flexibility to incorporate fiducial markers, which are essential for robotic localization and control. We propose a hybrid framework that combines the photorealism of GS with structured marker representations. Our core contribution is a novel algorithm for efficiently generating GS-based fiducial markers (e.g., AprilTags) within cluttered scenes. Experiments show that our approach outperforms traditional image-fitting techniques in both efficiency and pose-estimation accuracy. We further demonstrate the framework's potential in a greenhouse simulation. This agricultural setting serves as a challenging testbed, as its combination of dense foliage, similar-looking elements, and occlusions pushes the limits of perception, thereby highlighting the framework's value for real-world applications. 

**Abstract (ZH)**: 高保真3D仿真对于训练移动机器人至关重要，但其传统的基于网格的表示形式在复杂环境中往往难以应对，例如包括遮挡和重复结构的密集温室。最近的神经渲染方法，如高斯绘画（GS），实现了令人印象深刻的视觉真实感，但在整合用于机器人定位和控制的特征标记方面缺乏灵活性。我们提出了一种混合框架，将GS的视觉真实感与结构化标记表示相结合。我们的核心贡献是一种新型算法，用于在杂乱场景中高效生成基于GS的特征标记（如AprilTags）。实验表明，我们的方法在效率和姿态估计精度上均优于传统的图像匹配技术。我们进一步展示了该框架在温室仿真中的潜力。这种农业设置作为一个具有挑战性的测试平台，其密集植被、相似元素和遮挡的组合，对感知提出了极限挑战，从而突显了该框架在实际应用中的价值。 

---
# Observations of atypical users from a pilot deployment of a public-space social robot in a church 

**Title (ZH)**: 公共空间社会机器人在教堂试点部署中的非典型用户观察 

**Authors**: Andrew Blair, Peggy Gregory, Mary Ellen Foster  

**Link**: [PDF](https://arxiv.org/pdf/2508.16622)  

**Abstract**: Though a goal of HRI is the natural integration of social robots into everyday public spaces, real-world studies still occur mostly within controlled environments with predetermined participants. True public spaces present an environment which is largely unconstrained and unpredictable, frequented by a diverse range of people whose goals can often conflict with those of the robot. When combined with the general unfamiliarity most people have with social robots, this leads to unexpected human-robot interactions in these public spaces that are rarely discussed or detected in other contexts. In this paper, we describe atypical users we observed interacting with our robot, and those who did not, during a three-day pilot deployment within a large working church and visitor attraction. We then discuss theoretical future advances in the field that could address these challenges, as well as immediate practical mitigations and strategies to help improve public space human-robot interactions in the present. This work contributes empirical insights into the dynamics of human-robot interaction in public environments and offers actionable guidance for more effective future deployments for social robot designers. 

**Abstract (ZH)**: 虽然人机交互的目标是将社会机器人自然地整合到日常公共空间中，但现实世界的研究仍主要在受控环境中进行，参与者事先确定。真正的公共空间提供了一个基本不受限制且难以预测的环境，来往的人群多样，他们的目标有时会与机器人相冲突。当结合大多数人对社会机器人的普遍陌生感时，这导致在这些公共空间中出现了非典型的、难以预料的人机互动，而在其他情况下这些互动往往未被讨论或检测到。在本文中，我们描述了在一大型工作教堂和旅游景点为期三天的试点部署中观察到的非典型用户及其未进行人机互动的情况。随后我们讨论了可以通过理论上的未来发展来应对这些挑战的方法，同时也提出了一些即时的实践缓解措施和策略，以帮助改善当前公共空间中的人机互动。本文提供了有关公共环境中人机互动动态的实证见解，并为社会机器人设计师提供了可操作的指导，以实现更有效的未来部署。 

---
# Social Identity in Human-Agent Interaction: A Primer 

**Title (ZH)**: 人类与智能体互动中的社会身份： Primer导论 

**Authors**: Katie Seaborn  

**Link**: [PDF](https://arxiv.org/pdf/2508.16609)  

**Abstract**: Social identity theory (SIT) and social categorization theory (SCT) are two facets of the social identity approach (SIA) to understanding social phenomena. SIT and SCT are models that describe and explain how people interact with one another socially, connecting the individual to the group through an understanding of underlying psychological mechanisms and intergroup behaviour. SIT, originally developed in the 1970s, and SCT, a later, more general offshoot, have been broadly applied to a range of social phenomena among people. The rise of increasingly social machines embedded in daily life has spurned efforts on understanding whether and how artificial agents can and do participate in SIA activities. As agents like social robots and chatbots powered by sophisticated large language models (LLMs) advance, understanding the real and potential roles of these technologies as social entities is crucial. Here, I provide a primer on SIA and extrapolate, through case studies and imagined examples, how SIT and SCT can apply to artificial social agents. I emphasize that not all human models and sub-theories will apply. I further argue that, given the emerging competence of these machines and our tendency to be taken in by them, we experts may need to don the hat of the uncanny killjoy, for our own good. 

**Abstract (ZH)**: 社会身份理论（SIT）和社会分类理论（SCT）是社会身份方法（SIA）理解社会现象的两个方面。 

---
# Dimension-Decomposed Learning for Quadrotor Geometric Attitude Control with Almost Global Exponential Convergence on SO(3) 

**Title (ZH)**: 四旋翼几何姿态控制的维度分解学习及其在SO(3)上的几乎全局指数收敛 

**Authors**: Tianhua Gao, Masashi Izumita, Kohji Tomita, Akiya Kamimura  

**Link**: [PDF](https://arxiv.org/pdf/2508.14422)  

**Abstract**: This paper introduces a lightweight and interpretable online learning approach called Dimension-Decomposed Learning (DiD-L) for disturbance identification in quadrotor geometric attitude control. As a module instance of DiD-L, we propose the Sliced Adaptive-Neuro Mapping (SANM). Specifically, to address underlying underfitting problems, the high-dimensional mapping for online identification is axially ``sliced" into multiple low-dimensional submappings (slices). In this way, the complex high-dimensional problem is decomposed into a set of simple low-dimensional subtasks addressed by shallow neural networks and adaptive laws. These neural networks and adaptive laws are updated online via Lyapunov-based adaptation without the persistent excitation (PE) condition. To enhance the interpretability of the proposed approach, we prove that the state solution of the rotational error dynamics exponentially converges into an arbitrarily small ball within an almost global attraction domain, despite time-varying disturbances and inertia uncertainties. This result is novel as it demonstrates exponential convergence without requiring pre-training for unseen disturbances and specific knowledge of the model. To our knowledge in the quadrotor control field, DiD-L is the first online learning approach that is lightweight enough to run in real-time at 400 Hz on microcontroller units (MCUs) such as STM32, and has been validated through real-world experiments. 

**Abstract (ZH)**: 一种轻量级可解释的在线学习方法——维度分解学习（DiD-L）及其在四旋翼几何姿态控制中的扰动识别应用 

---
