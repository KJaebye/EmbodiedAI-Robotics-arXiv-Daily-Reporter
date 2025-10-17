# VT-Refine: Learning Bimanual Assembly with Visuo-Tactile Feedback via Simulation Fine-Tunin 

**Title (ZH)**: VT-Refine: 通过仿真微调学习双臂装配的视觉-触觉反馈方法 

**Authors**: Binghao Huang, Jie Xu, Iretiayo Akinola, Wei Yang, Balakumar Sundaralingam, Rowland O'Flaherty, Dieter Fox, Xiaolong Wang, Arsalan Mousavian, Yu-Wei Chao, Yunzhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14930)  

**Abstract**: Humans excel at bimanual assembly tasks by adapting to rich tactile feedback -- a capability that remains difficult to replicate in robots through behavioral cloning alone, due to the suboptimality and limited diversity of human demonstrations. In this work, we present VT-Refine, a visuo-tactile policy learning framework that combines real-world demonstrations, high-fidelity tactile simulation, and reinforcement learning to tackle precise, contact-rich bimanual assembly. We begin by training a diffusion policy on a small set of demonstrations using synchronized visual and tactile inputs. This policy is then transferred to a simulated digital twin equipped with simulated tactile sensors and further refined via large-scale reinforcement learning to enhance robustness and generalization. To enable accurate sim-to-real transfer, we leverage high-resolution piezoresistive tactile sensors that provide normal force signals and can be realistically modeled in parallel using GPU-accelerated simulation. Experimental results show that VT-Refine improves assembly performance in both simulation and the real world by increasing data diversity and enabling more effective policy fine-tuning. Our project page is available at this https URL. 

**Abstract (ZH)**: 人类在双臂装配任务中通过适应丰富的触觉反馈表现出色——这一能力仅通过行为克隆难以在机器人上复制，因为人类示范的不足和有限多样性。本文提出了一种结合实际示范、高保真触觉模拟和强化学习的视触觉策略学习框架VT-Refine，以应对精确的、接触丰富的双臂装配任务。我们首先使用同步的视觉和触觉输入对一个小规模示范集进行扩散策略训练。然后，将该策略转移到配备了模拟触觉传感器的模拟数字双体内，并通过大规模强化学习进一步精炼，以提高鲁棒性和泛化能力。为了使仿真实际转移更加准确，我们利用高分辨率压阻式触觉传感器，该传感器提供了法向力信号，并可通过GPU加速模拟并进行现实建模。实验结果表明，VT-Refine通过增加数据多样性并使策略微调更加有效，在仿真和现实世界中均提高了装配性能。我们的项目页面可访问此链接：这个 https URL。 

---
# STITCHER: Constrained Trajectory Planning in Known Environments with Real-Time Motion Primitive Search 

**Title (ZH)**: Stitcher: 有限环境中的实时运动元搜索受限轨迹规划 

**Authors**: Helene J. Levy, Brett T. Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2510.14893)  

**Abstract**: Autonomous high-speed navigation through large, complex environments requires real-time generation of agile trajectories that are dynamically feasible, collision-free, and satisfy state or actuator constraints. Modern trajectory planning techniques primarily use numerical optimization, as they enable the systematic computation of high-quality, expressive trajectories that satisfy various constraints. However, stringent requirements on computation time and the risk of numerical instability can limit the use of optimization-based planners in safety-critical scenarios. This work presents an optimization-free planning framework called STITCHER that stitches short trajectory segments together with graph search to compute long-range, expressive, and near-optimal trajectories in real-time. STITCHER outperforms modern optimization-based planners through our innovative planning architecture and several algorithmic developments that make real-time planning possible. Extensive simulation testing is performed to analyze the algorithmic components that make up STITCHER, along with a thorough comparison with two state-of-the-art optimization planners. Simulation tests show that safe trajectories can be created within a few milliseconds for paths that span the entirety of two 50 m x 50 m environments. Hardware tests with a custom quadrotor verify that STITCHER can produce trackable paths in real-time while respecting nonconvex constraints, such as limits on tilt angle and motor forces, which are otherwise hard to include in optimization-based planners. 

**Abstract (ZH)**: 自主高速导航通过大型复杂环境需要实时生成敏捷、动态可行、无碰撞且满足状态或执行器约束的轨迹。STITCHER：一种基于图搜索的无优化实时轨迹规划框架及其性能分析 

---
# Multi Agent Switching Mode Controller for Sound Source localization 

**Title (ZH)**: 多代理切换模式控制器在声源定位中的应用 

**Authors**: Marcello Sorge, Nicola Cigarini, Riccardo Lorigiola, Giulia Michieletto, Andrea Masiero, Angelo Cenedese, Alberto Guarnieri  

**Link**: [PDF](https://arxiv.org/pdf/2510.14849)  

**Abstract**: Source seeking is an important topic in robotic research, especially considering sound-based sensors since they allow the agents to locate a target even in critical conditions where it is not possible to establish a direct line of sight. In this work, we design a multi- agent switching mode control strategy for acoustic-based target localization. Two scenarios are considered: single source localization, in which the agents are driven maintaining a rigid formation towards the target, and multi-source scenario, in which each agent searches for the targets independently from the others. 

**Abstract (ZH)**: 基于声源的多agent切换模式目标定位研究 

---
# RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning 

**Title (ZH)**: RL-100: 实用的机器人 manipulotion 与现实世界的强化学习 

**Authors**: Kun Lei, Huanyu Li, Dongjie Yu, Zhenyu Wei, Lingxiao Guo, Zhennan Jiang, Ziyu Wang, Shiyu Liang, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14830)  

**Abstract**: Real-world robotic manipulation in homes and factories demands reliability, efficiency, and robustness that approach or surpass skilled human operators. We present RL-100, a real-world reinforcement learning training framework built on diffusion visuomotor policies trained bu supervised learning. RL-100 introduces a three-stage pipeline. First, imitation learning leverages human priors. Second, iterative offline reinforcement learning uses an Offline Policy Evaluation procedure, abbreviated OPE, to gate PPO-style updates that are applied in the denoising process for conservative and reliable improvement. Third, online reinforcement learning eliminates residual failure modes. An additional lightweight consistency distillation head compresses the multi-step sampling process in diffusion into a single-step policy, enabling high-frequency control with an order-of-magnitude reduction in latency while preserving task performance. The framework is task-, embodiment-, and representation-agnostic and supports both 3D point clouds and 2D RGB inputs, a variety of robot platforms, and both single-step and action-chunk policies. We evaluate RL-100 on seven real-robot tasks spanning dynamic rigid-body control, such as Push-T and Agile Bowling, fluids and granular pouring, deformable cloth folding, precise dexterous unscrewing, and multi-stage orange juicing. RL-100 attains 100\% success across evaluated trials for a total of 900 out of 900 episodes, including up to 250 out of 250 consecutive trials on one task. The method achieves near-human teleoperation or better time efficiency and demonstrates multi-hour robustness with uninterrupted operation lasting up to two hours. 

**Abstract (ZH)**: 基于扩散视觉运动策略的鲁棒实时强化学习训练框架：RL-100 

---
# Open TeleDex: A Hardware-Agnostic Teleoperation System for Imitation Learning based Dexterous Manipulation 

**Title (ZH)**: Open TeleDex：一种硬件无关的模仿学习灵巧操作远程操作系统 

**Authors**: Xu Chi, Chao Zhang, Yang Su, Lingfeng Dou, Fujia Yang, Jiakuo Zhao, Haoyu Zhou, Xiaoyou Jia, Yong Zhou, Shan An  

**Link**: [PDF](https://arxiv.org/pdf/2510.14771)  

**Abstract**: Accurate and high-fidelity demonstration data acquisition is a critical bottleneck for deploying robot Imitation Learning (IL) systems, particularly when dealing with heterogeneous robotic platforms. Existing teleoperation systems often fail to guarantee high-precision data collection across diverse types of teleoperation devices. To address this, we developed Open TeleDex, a unified teleoperation framework engineered for demonstration data collection. Open TeleDex specifically tackles the TripleAny challenge, seamlessly supporting any robotic arm, any dexterous hand, and any external input device. Furthermore, we propose a novel hand pose retargeting algorithm that significantly boosts the interoperability of Open TeleDex, enabling robust and accurate compatibility with an even wider spectrum of heterogeneous master and slave equipment. Open TeleDex establishes a foundational, high-quality, and publicly available platform for accelerating both academic research and industry development in complex robotic manipulation and IL. 

**Abstract (ZH)**: 准确且高保真的示范数据采集是部署机器人模仿学习系统的关键瓶颈，尤其是在处理异构机器人平台时。现有的远程操纵系统往往无法保证在多种类型的远程操纵设备之间进行高精度的数据采集。为解决这一问题，我们开发了Open TeleDex，这是一种统一的远程操纵框架，专为示范数据采集而设计。Open TeleDex特别解决了TripleAny挑战，无缝支持任何类型的机器人臂、任何灵巧手以及任何外部输入设备。此外，我们提出了一种新颖的手部姿态重新目标算法，显著提升了Open TeleDex的互操作性，使其能够与更广泛的异构主从设备实现更 robust 和准确的兼容性。Open TeleDex 为加速复杂机器人操作和模仿学习的学术研究和工业发展奠定了坚实、高质量且公开可用的基础平台。 

---
# Spatially anchored Tactile Awareness for Robust Dexterous Manipulation 

**Title (ZH)**: 空间锚定的触觉感知以实现稳健的灵巧操作 

**Authors**: Jialei Huang, Yang Ye, Yuanqing Gong, Xuezhou Zhu, Yang Gao, Kaifeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14647)  

**Abstract**: Dexterous manipulation requires precise geometric reasoning, yet existing visuo-tactile learning methods struggle with sub-millimeter precision tasks that are routine for traditional model-based approaches. We identify a key limitation: while tactile sensors provide rich contact information, current learning frameworks fail to effectively leverage both the perceptual richness of tactile signals and their spatial relationship with hand kinematics. We believe an ideal tactile representation should explicitly ground contact measurements in a stable reference frame while preserving detailed sensory information, enabling policies to not only detect contact occurrence but also precisely infer object geometry in the hand's coordinate system. We introduce SaTA (Spatially-anchored Tactile Awareness for dexterous manipulation), an end-to-end policy framework that explicitly anchors tactile features to the hand's kinematic frame through forward kinematics, enabling accurate geometric reasoning without requiring object models or explicit pose estimation. Our key insight is that spatially grounded tactile representations allow policies to not only detect contact occurrence but also precisely infer object geometry in the hand's coordinate system. We validate SaTA on challenging dexterous manipulation tasks, including bimanual USB-C mating in free space, a task demanding sub-millimeter alignment precision, as well as light bulb installation requiring precise thread engagement and rotational control, and card sliding that demands delicate force modulation and angular precision. These tasks represent significant challenges for learning-based methods due to their stringent precision requirements. Across multiple benchmarks, SaTA significantly outperforms strong visuo-tactile baselines, improving success rates by up to 30 percentage while reducing task completion times by 27 percentage. 

**Abstract (ZH)**: 灵巧操作需要精确的几何推理，而现有的基于视觉-触觉学习的方法在亚毫米级精度的任务上仍存在困难，这是传统基于模型的方法所擅长的。我们识别出一个关键限制：尽管触觉传感器提供了丰富的接触信息，当前的学习框架未能有效利用触觉信号的感知丰富性和其与手部运动学的空间关系。我们认为理想的触觉表示方式应该明确地将接触测量值锚定在一个稳定的参考框架中，同时保留详细的感官信息，使策略不仅能检测接触的发生，还能精确推断物体在手部坐标系统中的几何形状。我们引入了SaTA（空间锚定的触觉意识，用于灵巧操作）——一个端到端的策略框架，通过正向运动学将触觉特征明确锚定到手部的运动学框架，从而在无需物体模型或显式姿态估计的情况下实现精确的几何推理。我们的核心见解是，空间锚定的触觉表示使策略不仅能检测接触的发生，还能精确推断物体在手部坐标系统中的几何形状。我们在一系列挑战性的灵巧操作任务上验证了SaTA，包括自由空间中的双臂USB-C连接、需要亚毫米级对准精度的任务，以及要求精确牙纹配合和旋转控制的灯泡安装，还有需要精细力调节和角度精度的卡片滑动。这些任务对基于学习的方法构成了重大挑战，因为它们具有严格的技术要求。在多个基准测试中，SaTA 显著优于强大的基于视觉-触觉基线，成功率提高了30个百分点，完成任务时间减少了27个百分点。 

---
# Stability Criteria and Motor Performance in Delayed Haptic Dyadic Interactions Mediated by Robots 

**Title (ZH)**: 机器人介导的延迟触觉双人互动中的稳定性标准与电机性能 

**Authors**: Mingtian Du, Suhas Raghavendra Kulkarni, Simone Kager, Domenico Campolo  

**Link**: [PDF](https://arxiv.org/pdf/2510.14511)  

**Abstract**: This paper establishes analytical stability criteria for robot-mediated human-human (dyadic) interaction systems, focusing on haptic communication under network-induced time delays. Through frequency-domain analysis supported by numerical simulations, we identify both delay-independent and delay-dependent stability criteria. The delay-independent criterion guarantees stability irrespective of the delay, whereas the delay-dependent criterion is characterised by a maximum tolerable delay before instability occurs. The criteria demonstrate dependence on controller and robot dynamic parameters, where increasing stiffness reduces the maximum tolerable delay in a non-linear manner, thereby heightening system vulnerability. The proposed criteria can be generalised to a wide range of robot-mediated interactions and serve as design guidelines for stable remote dyadic systems. Experiments with robots performing human-like movements further illustrate the correlation between stability and motor performance. The findings of this paper suggest the prerequisites for effective delay-compensation strategies. 

**Abstract (ZH)**: 本文建立了由机器人介导的人与人（双人）交互系统的稳定性准则，重点讨论了网络引起的时延下的触觉通信稳定性。通过结合数值仿真支持的频域分析，我们确定了既依赖时延又独立于时延的稳定性准则。既不依赖于时延的准则保证了无论时延如何系统都是稳定的，而依赖于时延的准则则通过存在导致不稳定的最大容忍时延来表征。这些准则显示出对控制器和机器人动力学参数的依赖性，其中增加刚度以非线性方式减少了最大容忍时延，从而提高了系统的脆弱性。提出的标准可以推广到各种由机器人介导的交互，并作为稳定远程双人系统的设汁指南。机器人执行人类动作的实验进一步证明了稳定性和运动性能之间的关系。本文的研究结果表明了有效的时延补偿策略的前提条件。 

---
# RoboANKLE: Design, Development, and Functional Evaluation of a Robotic Ankle with a Motorized Compliant Unit 

**Title (ZH)**: RoboANKLE：具有电机化顺应单元的机器人踝关节的设计、开发与功能评价 

**Authors**: Baris Baysal, Omid Arfaie, Ramazan Unal  

**Link**: [PDF](https://arxiv.org/pdf/2510.14414)  

**Abstract**: This study presents a powered transtibial prosthesis with complete push-off assistance, RoboANKLE. The design aims to fulfill specific requirements, such as a sufficient range of motion (RoM) while providing the necessary torque for achieving natural ankle motion in daily activities. Addressing the challenges faced in designing active transtibial prostheses, such as maintaining energetic autonomy and minimizing weight, is vital for the study. With this aim, we try to imitate the human ankle by providing extensive push-off assistance to achieve a natural-like torque profile. Thus, Energy Store and Extended Release mechanism (ESER) is employed with a novel Extra Energy Storage (EES) mechanism. Kinematic and kinetic analyses are carried out to determine the design parameters and assess the design performance. Subsequently, a Computer-Aided Design (CAD) model is built and used in comprehensive dynamic and structural analyses. These analyses are used for the design performance evaluation and determine the forces and torques applied to the prosthesis, which aids in optimizing the design for minimal weight via structural analysis and topology optimization. The design of the prototype is then finalized and manufactured for experimental evaluation to validate the design and functionality. The prototype is realized with a mass of 1.92 kg and dimensions of 261x107x420 mm. The Functional evaluations of the RoboANKLE revealed that it is capable of achieving the natural maximum dorsi-flexion angle with 95% accuracy. Also, Thanks to the implemented mechanisms, the results show that RoboANKLE can generate 57% higher than the required torque for natural walking. The result of the power generation capacity of the RoboANKLE is 10% more than the natural power during the gait cycle. 

**Abstract (ZH)**: 一种提供完全推离辅助的电动�ModelError假肢RoboANKLE的研究 

---
# Prescribed Performance Control of Deformable Object Manipulation in Spatial Latent Space 

**Title (ZH)**: 空间潜在空间中可变形物体操作的指定性能控制 

**Authors**: Ning Han, Gu Gong, Bin Zhang, Yuexuan Xu, Bohan Yang, Yunhui Liu, David Navarro-Alarcon  

**Link**: [PDF](https://arxiv.org/pdf/2510.14234)  

**Abstract**: Manipulating three-dimensional (3D) deformable objects presents significant challenges for robotic systems due to their infinite-dimensional state space and complex deformable dynamics. This paper proposes a novel model-free approach for shape control with constraints imposed on key points. Unlike existing methods that rely on feature dimensionality reduction, the proposed controller leverages the coordinates of key points as the feature vector, which are extracted from the deformable object's point cloud using deep learning methods. This approach not only reduces the dimensionality of the feature space but also retains the spatial information of the object. By extracting key points, the manipulation of deformable objects is simplified into a visual servoing problem, where the shape dynamics are described using a deformation Jacobian matrix. To enhance control accuracy, a prescribed performance control method is developed by integrating barrier Lyapunov functions (BLF) to enforce constraints on the key points. The stability of the closed-loop system is rigorously analyzed and verified using the Lyapunov method. Experimental results further demonstrate the effectiveness and robustness of the proposed method. 

**Abstract (ZH)**: 基于关键点约束的三维可变形物体形状控制新方法 

---
# Partial Feedback Linearization Control of a Cable-Suspended Multirotor Platform for Stabilization of an Attached Load 

**Title (ZH)**: 基于部分反馈线性化控制的电缆悬吊多旋翼平台附载物稳定控制 

**Authors**: Hemjyoti Das, Christian Ott  

**Link**: [PDF](https://arxiv.org/pdf/2510.14072)  

**Abstract**: In this work, we present a novel control approach based on partial feedback linearization (PFL) for the stabilization of a suspended aerial platform with an attached load. Such systems are envisioned for various applications in construction sites involving cranes, such as the holding and transportation of heavy objects. Our proposed control approach considers the underactuation of the whole system while utilizing its coupled dynamics for stabilization. We demonstrate using numerical stability analysis that these coupled terms are crucial for the stabilization of the complete system. We also carried out robustness analysis of the proposed approach in the presence of external wind disturbances, sensor noise, and uncertainties in system dynamics. As our envisioned target application involves cranes in outdoor construction sites, our control approaches rely on only onboard sensors, thus making it suitable for such applications. We carried out extensive simulation studies and experimental tests to validate our proposed control approach. 

**Abstract (ZH)**: 基于部分反馈线性化的悬吊空中平台载重稳定控制新方法 

---
# Spatially Intelligent Patrol Routes for Concealed Emitter Localization by Robot Swarms 

**Title (ZH)**: 机器人 swarm 在隐蔽发射源定位中的智能巡逻路径规划 

**Authors**: Adam Morris, Timothy Pelham, Edmund R. Hunt  

**Link**: [PDF](https://arxiv.org/pdf/2510.14018)  

**Abstract**: This paper introduces a method for designing spatially intelligent robot swarm behaviors to localize concealed radio emitters. We use differential evolution to generate geometric patrol routes that localize unknown signals independently of emitter parameters, a key challenge in electromagnetic surveillance. Patrol shape and antenna type are shown to influence information gain, which in turn determines the effective triangulation coverage. We simulate a four-robot swarm across eight configurations, assigning pre-generated patrol routes based on a specified patrol shape and sensing capability (antenna type: omnidirectional or directional). An emitter is placed within the map for each trial, with randomized position, transmission power and frequency. Results show that omnidirectional localization success rates are driven primarily by source location rather than signal properties, with failures occurring most often when sources are placed in peripheral areas of the map. Directional antennas are able to overcome this limitation due to their higher gain and directivity, with an average detection success rate of 98.75% compared to 80.25% for omnidirectional. Average localization errors range from 1.01-1.30 m for directional sensing and 1.67-1.90 m for omnidirectional sensing; while directional sensing also benefits from shorter patrol edges. These results demonstrate that a swarm's ability to predict electromagnetic phenomena is directly dependent on its physical interaction with the environment. Consequently, spatial intelligence, realized here through optimized patrol routes and antenna selection, is a critical design consideration for effective robotic surveillance. 

**Abstract (ZH)**: 基于空间智能的机器人 swarm 定位隐蔽电磁发射器行为设计方法 

---
# Design of Paper Robot Building Kits 

**Title (ZH)**: 纸机器人搭建套件的设计 

**Authors**: Ruhan Yang, Ellen Yi-Luen Do  

**Link**: [PDF](https://arxiv.org/pdf/2510.14914)  

**Abstract**: Building robots is an engaging activity that provides opportunities for hands-on learning. However, traditional robot-building kits are usually costly with limited functionality due to material and technology constraints. To improve the accessibility and flexibility of such kits, we take paper as the building material and extensively explore the versatility of paper-based interactions. Based on an analysis of current robot-building kits and paper-based interaction research, we propose a design space for devising paper robots. We also analyzed our building kit designs using this design space, where these kits demonstrate the potential of paper as a cost-effective material for robot building. As a starting point, our design space and building kit examples provide a guideline that inspires and informs future research and development of novel paper robot-building kits. 

**Abstract (ZH)**: 将纸张作为构建材料探索纸基交互的潜力：构建成本-effective的纸机器人 

---
# Combining Reinforcement Learning and Behavior Trees for NPCs in Video Games with AMD Schola 

**Title (ZH)**: 使用AMD Schola结合强化学习和行为树为视频游戏中的NPCs设计智能行为 

**Authors**: Tian Liu, Alex Cann, Ian Colbert, Mehdi Saeedi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14154)  

**Abstract**: While the rapid advancements in the reinforcement learning (RL) research community have been remarkable, the adoption in commercial video games remains slow. In this paper, we outline common challenges the Game AI community faces when using RL-driven NPCs in practice, and highlight the intersection of RL with traditional behavior trees (BTs) as a crucial juncture to be explored further. Although the BT+RL intersection has been suggested in several research papers, its adoption is rare. We demonstrate the viability of this approach using AMD Schola -- a plugin for training RL agents in Unreal Engine -- by creating multi-task NPCs in a complex 3D environment inspired by the commercial video game ``The Last of Us". We provide detailed methodologies for jointly training RL models with BTs while showcasing various skills. 

**Abstract (ZH)**: 尽管强化学习（RL）研究领域的快速发展令人瞩目，但在商用视频游戏中的应用依然缓慢。本文概述了游戏AI社区在实践中使用基于RL的NPC所面临的常见挑战，并强调RL与传统行为树（BT）的交集是一个亟待深入探索的关键领域。尽管已有数篇研究论文建议采用BT+RL的交集方法，但实际上的应用仍然很少。我们通过使用AMD Schola（Unreal Engine中的一个训练RL代理的插件），在受商业视频游戏《最后生还者》启发的复杂3D环境中创建多任务NPC，展示了这一方法的可行性，并详细介绍了在同时训练RL模型与BT时的各种方法和技术。 

---
