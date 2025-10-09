# HyPlan: Hybrid Learning-Assisted Planning Under Uncertainty for Safe Autonomous Driving 

**Title (ZH)**: HyPlan: 综合学习辅助规划以应对不确定性实现安全自动驾驶 

**Authors**: Donald Pfaffmann, Matthias Klusch, Marcel Steinmetz  

**Link**: [PDF](https://arxiv.org/pdf/2510.07210)  

**Abstract**: We present a novel hybrid learning-assisted planning method, named HyPlan, for solving the collision-free navigation problem for self-driving cars in partially observable traffic environments. HyPlan combines methods for multi-agent behavior prediction, deep reinforcement learning with proximal policy optimization and approximated online POMDP planning with heuristic confidence-based vertical pruning to reduce its execution time without compromising safety of driving. Our experimental performance analysis on the CARLA-CTS2 benchmark of critical traffic scenarios with pedestrians revealed that HyPlan may navigate safer than selected relevant baselines and perform significantly faster than considered alternative online POMDP planners. 

**Abstract (ZH)**: 基于学习辅助规划的HyPlan方法：解决部分可观测交通环境中自动驾驶车辆的无碰撞导航问题 

---
# COMPAct: Computational Optimization and Automated Modular design of Planetary Actuators 

**Title (ZH)**: COMPAct: 计算优化与行星执行器自动化模块化设计 

**Authors**: Aman Singh, Deepak Kapa, Suryank Joshi, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2510.07197)  

**Abstract**: The optimal design of robotic actuators is a critical area of research, yet limited attention has been given to optimizing gearbox parameters and automating actuator CAD. This paper introduces COMPAct: Computational Optimization and Automated Modular Design of Planetary Actuators, a framework that systematically identifies optimal gearbox parameters for a given motor across four gearbox types, single-stage planetary gearbox (SSPG), compound planetary gearbox (CPG), Wolfrom planetary gearbox (WPG), and double-stage planetary gearbox (DSPG). The framework minimizes mass and actuator width while maximizing efficiency, and further automates actuator CAD generation to enable direct 3D printing without manual redesign. Using this framework, optimal gearbox designs are explored over a wide range of gear ratios, providing insights into the suitability of different gearbox types across various gear ratio ranges. In addition, the framework is used to generate CAD models of all four gearbox types with varying gear ratios and motors. Two actuator types are fabricated and experimentally evaluated through power efficiency, no-load backlash, and transmission stiffness tests. Experimental results indicate that the SSPG actuator achieves a mechanical efficiency of 60-80 %, a no-load backlash of 0.59 deg, and a transmission stiffness of 242.7 Nm/rad, while the CPG actuator demonstrates 60 % efficiency, 2.6 deg backlash, and a stiffness of 201.6 Nm/rad. Code available at: this https URL Video: this https URL 

**Abstract (ZH)**: 基于计算优化和自动模块化设计的行星_actuator参数优化框架：COMPAct 

---
# TIGeR: Tool-Integrated Geometric Reasoning in Vision-Language Models for Robotics 

**Title (ZH)**: TIGeR: 工具集成几何推理在视觉-语言模型中的机器人应用 

**Authors**: Yi Han, Cheng Chi, Enshen Zhou, Shanyu Rong, Jingkun An, Pengwei Wang, Zhongyuan Wang, Lu Sheng, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07181)  

**Abstract**: Vision-Language Models (VLMs) have shown remarkable capabilities in spatial reasoning, yet they remain fundamentally limited to qualitative precision and lack the computational precision required for real-world robotics. Current approaches fail to leverage metric cues from depth sensors and camera calibration, instead reducing geometric problems to pattern recognition tasks that cannot deliver the centimeter-level accuracy essential for robotic manipulation. We present TIGeR (Tool-Integrated Geometric Reasoning), a novel framework that transforms VLMs from perceptual estimators to geometric computers by enabling them to generate and execute precise geometric computations through external tools. Rather than attempting to internalize complex geometric operations within neural networks, TIGeR empowers models to recognize geometric reasoning requirements, synthesize appropriate computational code, and invoke specialized libraries for exact calculations. To support this paradigm, we introduce TIGeR-300K, a comprehensive tool-invocation-oriented dataset covering point transformations, pose estimation, trajectory generation, and spatial compatibility verification, complete with tool invocation sequences and intermediate computations. Through a two-stage training pipeline combining supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) with our proposed hierarchical reward design, TIGeR achieves SOTA performance on geometric reasoning benchmarks while demonstrating centimeter-level precision in real-world robotic manipulation tasks. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在空间推理方面展现了出色的能力，但在根本上仍受限于定性精度，缺乏用于实际机器人操作所需的计算精度。当前的方法未能利用深度传感器和相机校准带来的度量线索，而是将几何问题简化为模式识别任务，无法提供机器人操作所必需的厘米级精度。我们提出了TIGeR（Tool-Integrated Geometric Reasoning）框架，该框架通过使视觉-语言模型生成并执行精确的几何计算，从而将它们从感知估计器转变为几何计算器。TIGeR 不试图在神经网络内部实现复杂的几何操作，而是赋予模型识别几何推理需求的能力，合成适当的计算代码，并调用专门的库进行精确计算。为支持这一范式，我们引入了TIGeR-300K数据集，该数据集涵盖了点变换、姿态估计、轨迹生成和空间兼容性验证，还包含工具调用序列和中间计算。通过结合我们提出的层次奖励设计的监督微调（SFT）和强化微调（RFT）两阶段训练管道，TIGeR 在几何推理基准测试中达到了最佳性能，并在实际机器人操作任务中展示了厘米级精度。 

---
# A Narwhal-Inspired Sensing-to-Control Framework for Small Fixed-Wing Aircraft 

**Title (ZH)**: 受独角鲸启发的感测到控制框架：小型固定翼飞行器 

**Authors**: Fengze Xie, Xiaozhou Fan, Jacob Schuster, Yisong Yue, Morteza Gharib  

**Link**: [PDF](https://arxiv.org/pdf/2510.07160)  

**Abstract**: Fixed-wing unmanned aerial vehicles (UAVs) offer endurance and efficiency but lack low-speed agility due to highly coupled dynamics. We present an end-to-end sensing-to-control pipeline that combines bio-inspired hardware, physics-informed dynamics learning, and convex control allocation. Measuring airflow on a small airframe is difficult because near-body aerodynamics, propeller slipstream, control-surface actuation, and ambient gusts distort pressure signals. Inspired by the narwhal's protruding tusk, we mount in-house multi-hole probes far upstream and complement them with sparse, carefully placed wing pressure sensors for local flow measurement. A data-driven calibration maps probe pressures to airspeed and flow angles. We then learn a control-affine dynamics model using the estimated airspeed/angles and sparse sensors. A soft left/right symmetry regularizer improves identifiability under partial observability and limits confounding between wing pressures and flaperon inputs. Desired wrenches (forces and moments) are realized by a regularized least-squares allocator that yields smooth, trimmed actuation. Wind-tunnel studies across a wide operating range show that adding wing pressures reduces force-estimation error by 25-30%, the proposed model degrades less under distribution shift (about 12% versus 44% for an unstructured baseline), and force tracking improves with smoother inputs, including a 27% reduction in normal-force RMSE versus a plain affine model and 34% versus an unstructured baseline. 

**Abstract (ZH)**: 基于生物启发硬件、物理知情动力学习和凸控制分配的固定翼无人机端到端传感到控制管道 

---
# DPL: Depth-only Perceptive Humanoid Locomotion via Realistic Depth Synthesis and Cross-Attention Terrain Reconstruction 

**Title (ZH)**: DPL: 仅深度感知的人形机器人行走通过现实深度合成和跨注意力地形重建 

**Authors**: Jingkai Sun, Gang Han, Pihai Sun, Wen Zhao, Jiahang Cao, Jiaxu Wang, Yijie Guo, Qiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07152)  

**Abstract**: Recent advancements in legged robot perceptive locomotion have shown promising progress. However, terrain-aware humanoid locomotion remains largely constrained to two paradigms: depth image-based end-to-end learning and elevation map-based methods. The former suffers from limited training efficiency and a significant sim-to-real gap in depth perception, while the latter depends heavily on multiple vision sensors and localization systems, resulting in latency and reduced robustness. To overcome these challenges, we propose a novel framework that tightly integrates three key components: (1) Terrain-Aware Locomotion Policy with a Blind Backbone, which leverages pre-trained elevation map-based perception to guide reinforcement learning with minimal visual input; (2) Multi-Modality Cross-Attention Transformer, which reconstructs structured terrain representations from noisy depth images; (3) Realistic Depth Images Synthetic Method, which employs self-occlusion-aware ray casting and noise-aware modeling to synthesize realistic depth observations, achieving over 30\% reduction in terrain reconstruction error. This combination enables efficient policy training with limited data and hardware resources, while preserving critical terrain features essential for generalization. We validate our framework on a full-sized humanoid robot, demonstrating agile and adaptive locomotion across diverse and challenging terrains. 

**Abstract (ZH)**: 最近在腿式机器人知觉运动方面的进展显示出有希望的进步。然而，地形感知的人形运动主要局限于两种范式：基于深度图像的端到端学习和基于高程图的方法。前者由于训练效率有限和深度感知中的显著仿真实验差距而受到影响，而后者则严重依赖多个视觉传感器和定位系统，导致延迟并降低了鲁棒性。为克服这些挑战，我们提出了一种新的框架，该框架紧密整合了三个关键组件：（1）带有盲式主干的地形感知运动策略，利用预训练的高程图感知来引导强化学习，同时提供最少的视觉输入；（2）多模态跨注意力变压器，从噪声深度图像中重构结构化的地形表示；（3）真实的深度图像合成方法，采用自遮挡感知的射线投射和噪声感知建模来合成真实的深度观测，地形重建误差降低超过30%。该组合能够在有限的数据和硬件资源下实现高效的策略训练，同时保留对于泛化至关重要的关键地形特征。我们在一个全尺寸的人形机器人上验证了该框架，展示了其在各种复杂地形上的灵活和适应性运动。 

---
# TrackVLA++: Unleashing Reasoning and Memory Capabilities in VLA Models for Embodied Visual Tracking 

**Title (ZH)**: TrackVLA++: 在VLA模型中释放推理和记忆能力以实现感知实体视觉跟踪 

**Authors**: Jiahang Liu, Yunpeng Qi, Jiazhao Zhang, Minghan Li, Shaoan Wang, Kui Wu, Hanjing Ye, Hong Zhang, Zhibo Chen, Fangwei Zhong, Zhizheng Zhang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07134)  

**Abstract**: Embodied Visual Tracking (EVT) is a fundamental ability that underpins practical applications, such as companion robots, guidance robots and service assistants, where continuously following moving targets is essential. Recent advances have enabled language-guided tracking in complex and unstructured scenes. However, existing approaches lack explicit spatial reasoning and effective temporal memory, causing failures under severe occlusions or in the presence of similar-looking distractors. To address these challenges, we present TrackVLA++, a novel Vision-Language-Action (VLA) model that enhances embodied visual tracking with two key modules, a spatial reasoning mechanism and a Target Identification Memory (TIM). The reasoning module introduces a Chain-of-Thought paradigm, termed Polar-CoT, which infers the target's relative position and encodes it as a compact polar-coordinate token for action prediction. Guided by these spatial priors, the TIM employs a gated update strategy to preserve long-horizon target memory, ensuring spatiotemporal consistency and mitigating target loss during extended occlusions. Extensive experiments show that TrackVLA++ achieves state-of-the-art performance on public benchmarks across both egocentric and multi-camera settings. On the challenging EVT-Bench DT split, TrackVLA++ surpasses the previous leading approach by 5.1 and 12, respectively. Furthermore, TrackVLA++ exhibits strong zero-shot generalization, enabling robust real-world tracking in dynamic and occluded scenarios. 

**Abstract (ZH)**: 具身视觉跟踪 (EVT) 是一种基本能力，支撑着诸如伴侣机器人、引导机器人和服务助理等实际应用，其中连续跟踪移动目标至关重要。近期进展使得在复杂且未结构化的场景中实现语言引导的跟踪成为可能。然而，现有方法缺乏明确的空间推理能力和有效的时序记忆，导致在严重遮挡或存在外观相似的干扰物情况下出现故障。为解决这些挑战，我们提出了 TrackVLA++，这是一种新型的视觉-语言-行动 (VLA) 模型，通过引入空间推理机制和目标识别记忆 (TIM) 两个关键模块来增强具身视觉跟踪。推理模块引入了称为极坐标思维过程 (Polar-CoT) 的思维链框架，通过推断目标的相对位置并将其编码为紧凑的极坐标标记来预测行动。在这些空间先验的引导下，TIM 采用门控更新策略来保存长时目标记忆，确保时空一致性并在长时间遮挡期间减轻目标丢失。广泛实验表明，TrackVLA++ 在公共基准数据集上实现了最先进的性能，无论是在第一人称视角还是多摄像机设置下。在挑战性的 EVT-Bench DT 分割数据集上，TrackVLA++ 分别超过了之前的领先方法 5.1 和 12。此外，TrackVLA++ 演示了强大的零样本泛化能力，能够在动态和遮挡场景中实现稳健的实际世界跟踪。 

---
# A Digital Twin Framework for Metamorphic Testing of Autonomous Driving Systems Using Generative Model 

**Title (ZH)**: 基于生成模型的自主驾驶系统 metamorphic 测试的数字孪生框架 

**Authors**: Tony Zhang, Burak Kantarci, Umair Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2510.07133)  

**Abstract**: Ensuring the safety of self-driving cars remains a major challenge due to the complexity and unpredictability of real-world driving environments. Traditional testing methods face significant limitations, such as the oracle problem, which makes it difficult to determine whether a system's behavior is correct, and the inability to cover the full range of scenarios an autonomous vehicle may encounter. In this paper, we introduce a digital twin-driven metamorphic testing framework that addresses these challenges by creating a virtual replica of the self-driving system and its operating environment. By combining digital twin technology with AI-based image generative models such as Stable Diffusion, our approach enables the systematic generation of realistic and diverse driving scenes. This includes variations in weather, road topology, and environmental features, all while maintaining the core semantics of the original scenario. The digital twin provides a synchronized simulation environment where changes can be tested in a controlled and repeatable manner. Within this environment, we define three metamorphic relations inspired by real-world traffic rules and vehicle behavior. We validate our framework in the Udacity self-driving simulator and demonstrate that it significantly enhances test coverage and effectiveness. Our method achieves the highest true positive rate (0.719), F1 score (0.689), and precision (0.662) compared to baseline approaches. This paper highlights the value of integrating digital twins with AI-powered scenario generation to create a scalable, automated, and high-fidelity testing solution for autonomous vehicle safety. 

**Abstract (ZH)**: 基于数字孪生的 metamorphic 测试框架：自动驾驶汽车安全测试的新挑战与解决方案 

---
# Sampling Strategies for Robust Universal Quadrupedal Locomotion Policies 

**Title (ZH)**: 稳健通用四足运动策略的采样策略 

**Authors**: David Rytz, Kim Tien Ly, Ioannis Havoutis  

**Link**: [PDF](https://arxiv.org/pdf/2510.07094)  

**Abstract**: This work focuses on sampling strategies of configuration variations for generating robust universal locomotion policies for quadrupedal robots. We investigate the effects of sampling physical robot parameters and joint proportional-derivative gains to enable training a single reinforcement learning policy that generalizes to multiple parameter configurations. Three fundamental joint gain sampling strategies are compared: parameter sampling with (1) linear and polynomial function mappings of mass-to-gains, (2) performance-based adaptive filtering, and (3) uniform random sampling. We improve the robustness of the policy by biasing the configurations using nominal priors and reference models. All training was conducted on RaiSim, tested in simulation on a range of diverse quadrupeds, and zero-shot deployed onto hardware using the ANYmal quadruped robot. Compared to multiple baseline implementations, our results demonstrate the need for significant joint controller gains randomization for robust closing of the sim-to-real gap. 

**Abstract (ZH)**: 本文集中在配置变异的采样策略上，以生成四足机器人的稳健通用行走策略。我们调查了采样物理机器人参数和关节比例-微分增益的影响，以使训练单个强化学习策略能够泛化到多种参数配置。三种基本的关节增益采样策略进行了比较：参数采样具有（1）质量到增益的线性和多项式函数映射，（2）基于性能的自适应过滤，以及（3）均匀随机采样。通过使用先验知识和参考模型对配置进行偏差调整，以提高策略的稳健性。研究在RaiSim上进行训练，在多种多样的四足机器人仿真中进行测试，并使用ANYmal四足机器人进行零样本部署。与多个基线实施相比，我们的结果表明，为了稳健地缩小仿真到现实的差距，关节控制器增益的显著随机化是必需的。 

---
# Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications 

**Title (ZH)**: 机器人视觉-语言-行动模型：面向实际应用的综述 

**Authors**: Kento Kawaharazuka, Jihoon Oh, Jun Yamada, Ingmar Posner, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07077)  

**Abstract**: Amid growing efforts to leverage advances in large language models (LLMs) and vision-language models (VLMs) for robotics, Vision-Language-Action (VLA) models have recently gained significant attention. By unifying vision, language, and action data at scale, which have traditionally been studied separately, VLA models aim to learn policies that generalise across diverse tasks, objects, embodiments, and environments. This generalisation capability is expected to enable robots to solve novel downstream tasks with minimal or no additional task-specific data, facilitating more flexible and scalable real-world deployment. Unlike previous surveys that focus narrowly on action representations or high-level model architectures, this work offers a comprehensive, full-stack review, integrating both software and hardware components of VLA systems. In particular, this paper provides a systematic review of VLAs, covering their strategy and architectural transition, architectures and building blocks, modality-specific processing techniques, and learning paradigms. In addition, to support the deployment of VLAs in real-world robotic applications, we also review commonly used robot platforms, data collection strategies, publicly available datasets, data augmentation methods, and evaluation benchmarks. Throughout this comprehensive survey, this paper aims to offer practical guidance for the robotics community in applying VLAs to real-world robotic systems. All references categorized by training approach, evaluation method, modality, and dataset are available in the table on our project website: this https URL . 

**Abstract (ZH)**: 随着日益增长的努力利用大型语言模型（LLMs）和视觉-语言模型（VLMs）来推动机器人技术的发展，视觉-语言-动作（VLA）模型最近已经获得了显著的关注。通过大规模统一研究历来单独研究的视觉、语言和动作数据，VLA模型旨在学习能在多任务、多物体、多实体和多环境条件下泛化的策略。这种泛化能力有望使机器人能够使用最少或无需额外任务特定数据来解决新的下游任务，从而促进更为灵活和可扩展的实际应用部署。不同于过去专注于动作表示或高层模型架构的综述，本文提供了一个全面的端到端回顾，涵盖了VLA系统的软件和硬件组件。特别是，本文系统地回顾了VLA，涵盖了它们的策略和架构转变、架构和构建块、模态特定处理技术以及学习范式。此外，为了支持VLA在实际机器人应用中的部署，本文还回顾了常用的机器人平台、数据收集策略、公开可用的数据集、数据增强方法以及评估基准。在本文全面的综述中，旨在为机器人界如何将VLA应用于实际机器人系统提供实用指导。所有按训练方法、评估方法、模态和数据集分类的参考文献详见我们项目网站上的表格：this https URL。 

---
# Bring the Apple, Not the Sofa: Impact of Irrelevant Context in Embodied AI Commands on VLA Models 

**Title (ZH)**: bring the apple, not the sofa: 不相关背景对 embodied AI 命令影响的研究——基于视觉语言模型的视角 

**Authors**: Daria Pugacheva, Andrey Moskalenko, Denis Shepelev, Andrey Kuznetsov, Vlad Shakhuro, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2510.07067)  

**Abstract**: Vision Language Action (VLA) models are widely used in Embodied AI, enabling robots to interpret and execute language instructions. However, their robustness to natural language variability in real-world scenarios has not been thoroughly investigated. In this work, we present a novel systematic study of the robustness of state-of-the-art VLA models under linguistic perturbations. Specifically, we evaluate model performance under two types of instruction noise: (1) human-generated paraphrasing and (2) the addition of irrelevant context. We further categorize irrelevant contexts into two groups according to their length and their semantic and lexical proximity to robot commands. In this study, we observe consistent performance degradation as context size expands. We also demonstrate that the model can exhibit relative robustness to random context, with a performance drop within 10%, while semantically and lexically similar context of the same length can trigger a quality decline of around 50%. Human paraphrases of instructions lead to a drop of nearly 20%. To mitigate this, we propose an LLM-based filtering framework that extracts core commands from noisy inputs. Incorporating our filtering step allows models to recover up to 98.5% of their original performance under noisy conditions. 

**Abstract (ZH)**: 基于视觉-语言-动作模型在语言 perturbations 下的鲁棒性研究 

---
# Diffusing Trajectory Optimization Problems for Recovery During Multi-Finger Manipulation 

**Title (ZH)**: 多指 manipulation 过程中恢复的扩散轨迹优化问题 

**Authors**: Abhinav Kumar, Fan Yang, Sergio Aguilera Marinovic, Soshi Iba, Rana Soltani Zarrin, Dmitry Berenson  

**Link**: [PDF](https://arxiv.org/pdf/2510.07030)  

**Abstract**: Multi-fingered hands are emerging as powerful platforms for performing fine manipulation tasks, including tool use. However, environmental perturbations or execution errors can impede task performance, motivating the use of recovery behaviors that enable normal task execution to resume. In this work, we take advantage of recent advances in diffusion models to construct a framework that autonomously identifies when recovery is necessary and optimizes contact-rich trajectories to recover. We use a diffusion model trained on the task to estimate when states are not conducive to task execution, framed as an out-of-distribution detection problem. We then use diffusion sampling to project these states in-distribution and use trajectory optimization to plan contact-rich recovery trajectories. We also propose a novel diffusion-based approach that distills this process to efficiently diffuse the full parameterization, including constraints, goal state, and initialization, of the recovery trajectory optimization problem, saving time during online execution. We compare our method to a reinforcement learning baseline and other methods that do not explicitly plan contact interactions, including on a hardware screwdriver-turning task where we show that recovering using our method improves task performance by 96% and that ours is the only method evaluated that can attempt recovery without causing catastrophic task failure. Videos can be found at this https URL. 

**Abstract (ZH)**: 多指灵巧手在进行精密操作任务时正逐渐成为强大的平台，包括工具使用。然而，环境干扰或执行错误会阻碍任务性能，促使采取恢复行为以恢复正常的任务执行。在本文中，我们利用最近在扩散模型方面的进展构建了一个框架，该框架能够自主识别何时需要恢复，并优化富含接触的轨迹以进行恢复。我们使用基于任务训练的扩散模型来估计哪些状态不利于任务执行，将其作为分布外检测问题。然后，我们使用扩散采样将这些状态映射到分布内，并使用轨迹优化来规划富含接触的恢复轨迹。我们还提出了一种基于扩散的新颖方法，该方法能够高效地扩散恢复轨迹优化问题的全部参数化，包括约束、目标状态和初始化，从而在线执行时节省时间。我们将我们的方法与强化学习基线以及其他未显式规划接触交互的方法进行比较，在一个硬件螺丝刀旋转任务中，我们证明使用我们的方法进行恢复可以将任务性能提高96%，并且这是唯一一个能够尝试恢复而不导致任务灾难性失败的方法。相关视频可以在以下链接找到：这个 https URL。 

---
# Temporal-Prior-Guided View Planning for Periodic 3D Plant Reconstruction 

**Title (ZH)**: 基于时间先验的视图规划以实现周期性三维植物重建 

**Authors**: Sicong Pan, Xuying Huang, Maren Bennewitz  

**Link**: [PDF](https://arxiv.org/pdf/2510.07028)  

**Abstract**: Periodic 3D reconstruction is essential for crop monitoring, but costly when each cycle restarts from scratch, wasting resources and ignoring information from previous captures. We propose temporal-prior-guided view planning for periodic plant reconstruction, in which a previously reconstructed model of the same plant is non-rigidly aligned to a new partial observation to form an approximation of the current geometry. To accommodate plant growth, we inflate this approximation and solve a set covering optimization problem to compute a minimal set of views. We integrated this method into a complete pipeline that acquires one additional next-best view before registration for robustness and then plans a globally shortest path to connect the planned set of views and outputs the best view sequence. Experiments on maize and tomato under hemisphere and sphere view spaces show that our system maintains or improves surface coverage while requiring fewer views and comparable movement cost compared to state-of-the-art baselines. 

**Abstract (ZH)**: 基于时间先验的周期性植物重建视图规划 

---
# Tailoring materials into kirigami robots 

**Title (ZH)**: 将材料裁剪成 kirigami 机器人 

**Authors**: Saravana Prashanth Murali Babu, Aida Parvaresh, Ahmad Rafsanjani  

**Link**: [PDF](https://arxiv.org/pdf/2510.07027)  

**Abstract**: Kirigami, the traditional paper-cutting craft, holds immense potential for revolutionizing robotics by providing multifunctional, lightweight, and adaptable solutions. Kirigami structures, characterized by their bending-dominated deformation, offer resilience to tensile forces and facilitate shape morphing under small actuation forces. Kirigami components such as actuators, sensors, batteries, controllers, and body structures can be tailored to specific robotic applications by optimizing cut patterns. Actuators based on kirigami principles exhibit complex motions programmable through various energy sources, while kirigami sensors bridge the gap between electrical conductivity and compliance. Kirigami-integrated batteries enable energy storage directly within robot structures, enhancing flexibility and compactness. Kirigami-controlled mechanisms mimic mechanical computations, enabling advanced functionalities such as shape morphing and memory functions. Applications of kirigami-enabled robots include grasping, locomotion, and wearables, showcasing their adaptability to diverse environments and tasks. Despite promising opportunities, challenges remain in the design of cut patterns for a given function and streamlining fabrication techniques. 

**Abstract (ZH)**: kirigami在传统纸切割工艺的基础上为机器人技术带来了巨大的潜力，提供了多功能、轻量化和适应性强的解决方案。kirigami结构以其弯曲主导的变形特性，对拉力具有良好的适应性，并且在小驱动力作用下能够实现形状变化。kirigami组件，如执行器、传感器、电池、控制器和机体结构，可以通过优化切割模式来针对特定的机器人应用进行定制。基于kirigami原理的执行器能够通过多种能量源编程实现复杂的运动，而kirigami传感器则在电导性和顺应性之间建立了桥梁。通过kirigami整合的电池可以直接在机器人结构内部实现能量存储，提高灵活性和紧凑性。kirigami控制机制模拟机械计算，使机器人具备形状变化和记忆功能等高级功能。 kirigami使能的机器人应用包括抓取、移动和穿戴设备，展示了其在多种环境和任务中的适应性。尽管前景广阔，但在给定功能下设计切割模式以及简化制造技术方面仍存在挑战。 

---
# Distributed 3D Source Seeking via SO(3) Geometric Control of Robot Swarms 

**Title (ZH)**: 基于SO(3)几何控制的分布式三维源搜索机器人 swarm 控制方法 

**Authors**: Jesús Bautista, Héctor García de Marina  

**Link**: [PDF](https://arxiv.org/pdf/2510.06836)  

**Abstract**: This paper presents a geometric control framework on the Lie group SO(3) for 3D source-seeking by robots with first-order attitude dynamics and constant translational speed. By working directly on SO(3), the approach avoids Euler-angle singularities and quaternion ambiguities, providing a unique, intrinsic representation of orientation. We design a proportional feed-forward controller that ensures exponential alignment of each agent to an estimated ascending direction toward a 3D scalar field source. The controller adapts to bounded unknown variations and preserves well-posed swarm formations. Numerical simulations demonstrate the effectiveness of the method, with all code provided open source for reproducibility. 

**Abstract (ZH)**: 基于Lie群SO(3)的几何控制框架：具有一阶姿态动力学和恒定平移速度的机器人三维源寻觅控制 

---
# UniFField: A Generalizable Unified Neural Feature Field for Visual, Semantic, and Spatial Uncertainties in Any Scene 

**Title (ZH)**: UniFField：一种通用的统一神经特征场，用于任意场景中的视觉、语义和空间不确定性 

**Authors**: Christian Maurer, Snehal Jauhri, Sophie Lueth, Georgia Chalvatzaki  

**Link**: [PDF](https://arxiv.org/pdf/2510.06754)  

**Abstract**: Comprehensive visual, geometric, and semantic understanding of a 3D scene is crucial for successful execution of robotic tasks, especially in unstructured and complex environments. Additionally, to make robust decisions, it is necessary for the robot to evaluate the reliability of perceived information. While recent advances in 3D neural feature fields have enabled robots to leverage features from pretrained foundation models for tasks such as language-guided manipulation and navigation, existing methods suffer from two critical limitations: (i) they are typically scene-specific, and (ii) they lack the ability to model uncertainty in their predictions. We present UniFField, a unified uncertainty-aware neural feature field that combines visual, semantic, and geometric features in a single generalizable representation while also predicting uncertainty in each modality. Our approach, which can be applied zero shot to any new environment, incrementally integrates RGB-D images into our voxel-based feature representation as the robot explores the scene, simultaneously updating uncertainty estimation. We evaluate our uncertainty estimations to accurately describe the model prediction errors in scene reconstruction and semantic feature prediction. Furthermore, we successfully leverage our feature predictions and their respective uncertainty for an active object search task using a mobile manipulator robot, demonstrating the capability for robust decision-making. 

**Abstract (ZH)**: 全面的视觉、几何和语义理解对于机器人在不规则和复杂环境中成功执行任务至关重要。此外，为了做出稳健的决策，机器人需要评估其感知信息的可靠性。虽然近期在三维神经特征场方面的进展使机器人能够利用预训练基础模型的特征进行语言引导的操作和导航等任务，但现有方法存在两个关键局限性：（i）它们通常是场景特定的；（ii）缺乏在其预测中建模不确定性的能力。我们提出了一种统一的不确定性感知神经特征场UniFField，该方法将视觉、语义和几何特征结合在一个通用表示中，同时预测每个模态的不确定性。我们的方法可以零样本迁移到任何新的环境，在机器人探索场景时，逐步将RGB-D图像集成到基于体素的特征表示中，并同时更新不确定性估计。我们评估了不确定性估计，以准确描述场景重建和语义特征预测中的模型预测误差。此外，我们成功利用特征预测及其各自的不确定性进行了一个基于移动操作机器人的主动对象搜索任务，展示了其进行稳健决策的能力。 

---
# SanDRA: Safe Large-Language-Model-Based Decision Making for Automated Vehicles Using Reachability Analysis 

**Title (ZH)**: SanDRA: 基于可达性分析的大型语言模型驱动的自动驾驶安全决策方法 

**Authors**: Yuanfei Lin, Sebastian Illing, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2510.06717)  

**Abstract**: Large language models have been widely applied to knowledge-driven decision-making for automated vehicles due to their strong generalization and reasoning capabilities. However, the safety of the resulting decisions cannot be ensured due to possible hallucinations and the lack of integrated vehicle dynamics. To address this issue, we propose SanDRA, the first safe large-language-model-based decision making framework for automated vehicles using reachability analysis. Our approach starts with a comprehensive description of the driving scenario to prompt large language models to generate and rank feasible driving actions. These actions are translated into temporal logic formulas that incorporate formalized traffic rules, and are subsequently integrated into reachability analysis to eliminate unsafe actions. We validate our approach in both open-loop and closed-loop driving environments using off-the-shelf and finetuned large language models, showing that it can provide provably safe and, where possible, legally compliant driving actions, even under high-density traffic conditions. To ensure transparency and facilitate future research, all code and experimental setups are publicly available at this http URL. 

**Abstract (ZH)**: 基于可达性分析的安全large语言模型驱动的自动驾驶决策框架SanDRA 

---
# RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training 

**Title (ZH)**: RLinf-VLA：一种统一高效的VLA+RL训练框架 

**Authors**: Hongzhi Zang, Mingjie Wei, Si Xu, Yongji Wu, Zhen Guo, Yuanqing Wang, Hao Lin, Liangzhi Shi, Yuqing Xie, Zhexuan Xu, Zhihao Liu, Kang Chen, Wenhao Tang, Quanlu Zhang, Weinan Zhang, Chao Yu, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06710)  

**Abstract**: Recent progress in vision and language foundation models has significantly advanced multimodal understanding, reasoning, and generation, inspiring a surge of interest in extending such capabilities to embodied settings through vision-language-action (VLA) models. Yet, most VLA models are still trained with supervised fine-tuning (SFT), which struggles to generalize under distribution shifts due to error accumulation. Reinforcement learning (RL) offers a promising alternative by directly optimizing task performance through interaction, but existing attempts remain fragmented and lack a unified platform for fair and systematic comparison across model architectures and algorithmic designs. To address this gap, we introduce RLinf-VLA, a unified and efficient framework for scalable RL training of VLA models. The system adopts a highly flexible resource allocation design that addresses the challenge of integrating rendering, training, and inference in RL+VLA training. In particular, for GPU-parallelized simulators, RLinf-VLA implements a novel hybrid fine-grained pipeline allocation mode, achieving a 1.61x-1.88x speedup in training. Through a unified interface, RLinf-VLA seamlessly supports diverse VLA architectures (e.g., OpenVLA, OpenVLA-OFT), multiple RL algorithms (e.g., PPO, GRPO), and various simulators (e.g., ManiSkill, LIBERO). In simulation, a unified model achieves 98.11\% across 130 LIBERO tasks and 97.66\% across 25 ManiSkill tasks. Beyond empirical performance, our study distills a set of best practices for applying RL to VLA training and sheds light on emerging patterns in this integration. Furthermore, we present preliminary deployment on a real-world Franka robot, where RL-trained policies exhibit stronger generalization than those trained with SFT. We envision RLinf-VLA as a foundation to accelerate and standardize research on embodied intelligence. 

**Abstract (ZH)**: 近期视觉与语言基础模型的进展显著推进了多模态理解、推理和生成，激发了通过视觉-语言-动作（VLA）模型将此类能力扩展到具身环境中的兴趣。然而，大多数VLA模型仍使用监督微调（SFT）训练，这在分布迁移时由于误差累积难以泛化。强化学习（RL）通过交互直接优化任务性能提供了有前景的替代方案，但现有尝试仍碎片化且缺乏针对不同模型架构和算法设计进行全面公平比较的统一平台。为填补这一空白，我们引入了RLinf-VLA，这是一个用于可扩展的VLA模型强化学习训练的统一高效框架。该系统采用高度灵活的资源分配设计，解决了RL+VLA训练中渲染、训练和推理整合的挑战。特别是，对于GPU并行化模拟器，RLinf-VLA 实现了一种新型细粒度混合管道分配模式，实现1.61x-1.88x的训练速度提升。通过统一接口，RLinf-VLA 紧密支持多种VLA架构（如OpenVLA、OpenVLA-OFT）、多种RL算法（如PPO、GRPO）和各种模拟器（如ManiSkill、LIBERO）。在模拟中，统一模型在130个LIBERO任务中达到了98.11%的表现，在25个ManiSkill任务中达到了97.66%的表现。除实证性能外，我们的研究提炼了将RL应用于VLA训练的最佳实践，并揭示了这一集成中的新兴模式。此外，我们初步展示了在真实世界Franka机器人上的部署，其中RL训练策略的泛化能力优于SFT训练策略。我们期望RLinf-VLA成为加速和标准化具身智能研究的基础。 

---
# Assist-As-Needed: Adaptive Multimodal Robotic Assistance for Medication Management in Dementia Care 

**Title (ZH)**: 按需辅助：适应性多模态机器人辅助在失智照护中的药物管理 

**Authors**: Kruthika Gangaraju, Tanmayi Inaparthy, Jiaqi Yang, Yihao Zheng, Fengpei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.06633)  

**Abstract**: People living with dementia (PLWDs) face progressively declining abilities in medication management-from simple forgetfulness to complete task breakdown-yet most assistive technologies fail to adapt to these changing needs. This one-size-fits-all approach undermines autonomy, accelerates dependence, and increases caregiver burden. Occupational therapy principles emphasize matching assistance levels to individual capabilities: minimal reminders for those who merely forget, spatial guidance for those who misplace items, and comprehensive multimodal support for those requiring step-by-step instruction. However, existing robotic systems lack this adaptive, graduated response framework essential for maintaining PLWD independence. We present an adaptive multimodal robotic framework using the Pepper robot that dynamically adjusts assistance based on real-time assessment of user needs. Our system implements a hierarchical intervention model progressing from (1) simple verbal reminders, to (2) verbal + gestural cues, to (3) full multimodal guidance combining physical navigation to medication locations with step-by-step verbal and gestural instructions. Powered by LLM-driven interaction strategies and multimodal sensing, the system continuously evaluates task states to provide just-enough assistance-preserving autonomy while ensuring medication adherence. We conducted a preliminary study with healthy adults and dementia care stakeholders in a controlled lab setting, evaluating the system's usability, comprehensibility, and appropriateness of adaptive feedback mechanisms. This work contributes: (1) a theoretically grounded adaptive assistance framework translating occupational therapy principles into HRI design, (2) a multimodal robotic implementation that preserves PLWD dignity through graduated support, and (3) empirical insights into stakeholder perceptions of adaptive robotic care. 

**Abstract (ZH)**: 适应性多模态机器人框架：基于帕金森病患者认知功能变化的动态照料 

---
# Safe Obstacle-Free Guidance of Space Manipulators in Debris Removal Missions via Deep Reinforcement Learning 

**Title (ZH)**: 基于深度强化学习的空间 manipulation 安全无障碍引导以应对碎片清除任务 

**Authors**: Vincent Lam, Robin Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2510.06566)  

**Abstract**: The objective of this study is to develop a model-free workspace trajectory planner for space manipulators using a Twin Delayed Deep Deterministic Policy Gradient (TD3) agent to enable safe and reliable debris capture. A local control strategy with singularity avoidance and manipulability enhancement is employed to ensure stable execution. The manipulator must simultaneously track a capture point on a non-cooperative target, avoid self-collisions, and prevent unintended contact with the target. To address these challenges, we propose a curriculum-based multi-critic network where one critic emphasizes accurate tracking and the other enforces collision avoidance. A prioritized experience replay buffer is also used to accelerate convergence and improve policy robustness. The framework is evaluated on a simulated seven-degree-of-freedom KUKA LBR iiwa mounted on a free-floating base in Matlab/Simulink, demonstrating safe and adaptive trajectory generation for debris removal missions. 

**Abstract (ZH)**: 本研究的目标是开发一种基于TD3代理的无模型工作空间轨迹规划器，以空间 manipulator 为对象，实现对未配合目标的碎片捕获，确保安全可靠。该计划器采用具有奇异性避免和操作性增强的局部控制策略，以确保执行的稳定性。机械臂必须同时在非配合目标上跟踪捕获点，避免自碰撞，并防止意外接触目标。为应对这些挑战，我们提出了一种基于课程的学习多评价格值网络，其中一个是强调准确跟踪，另一个是强制执行碰撞避免。同时，使用优先经验重放缓冲区以加快收敛速度并提高策略的鲁棒性。该框架在Matlab/Simulink中对一个自由浮动底座上安装的七个自由度的KUKA LBR iiwa进行了仿真评估，展示了适用于碎片清除任务的安全自适应轨迹生成能力。 

---
# RAISE: A self-driving laboratory for interfacial property formulation discovery 

**Title (ZH)**: RAISE：一种自驾驶实验室，用于界面性质表征发现 

**Authors**: Mohammad Nazeri, Sheldon Mei, Jeffrey Watchorn, Alex Zhang, Erin Ng, Tao Wen, Abhijoy Mandal, Kevin Golovin, Alan Aspuru-Guzik, Frank Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06546)  

**Abstract**: Surface wettability is a critical design parameter for biomedical devices, coatings, and textiles. Contact angle measurements quantify liquid-surface interactions, which depend strongly on liquid formulation. Herein, we present the Robotic Autonomous Imaging Surface Evaluator (RAISE), a closed-loop, self-driving laboratory that is capable of linking liquid formulation optimization with surface wettability assessment. RAISE comprises a full experimental orchestrator with the ability of mixing liquid ingredients to create varying formulation cocktails, transferring droplets of prepared formulations to a high-throughput stage, and using a pick-and-place camera tool for automated droplet image capture. The system also includes an automated image processing pipeline to measure contact angles. This closed loop experiment orchestrator is integrated with a Bayesian Optimization (BO) client, which enables iterative exploration of new formulations based on previous contact angle measurements to meet user-defined objectives. The system operates in a high-throughput manner and can achieve a measurement rate of approximately 1 contact angle measurement per minute. Here we demonstrate RAISE can be used to explore surfactant wettability and how surfactant combinations create tunable formulations that compensate for purity-related variations. Furthermore, multi-objective BO demonstrates how precise and optimal formulations can be reached based on application-specific goals. The optimization is guided by a desirability score, which prioritizes formulations that are within target contact angle ranges, minimize surfactant usage and reduce cost. This work demonstrates the capabilities of RAISE to autonomously link liquid formulations to contact angle measurements in a closed-loop system, using multi-objective BO to efficiently identify optimal formulations aligned with researcher-defined criteria. 

**Abstract (ZH)**: 表面润湿性是生物医学设备、涂层和纺织品设计中的关键参数。接触角测量量化了液体-表面的相互作用，这取决于液体的配方。在此，我们介绍了一种名为自主成像表面评估器（RAISE）的闭环、自动驾驶实验室，它能够将液体配方优化与表面润湿性评估联系起来。RAISE包括一个完整的实验调度器，能够混合液体成分以创建不同的配方混合物，将准备好的配方滴转移到高通量平台上，并使用拣放相机工具实现自动液滴图像捕获。系统还包含一个自动图像处理流水线来测量接触角。该闭环实验调度器与贝叶斯优化（BO）客户端集成，可以根据之前的接触角测量结果进行迭代探索，以满足用户定义的目标。系统以高通量方式运行，每分钟可以完成约一次接触角测量。我们演示了RAISE可以用于探索表面活性剂的润湿性及其组合如何创建可调配方以补偿纯度相关变化。此外，多目标BO展示了如何基于应用特定目标精确并优化配方，优化由可实现性评分指导，优先考虑目标接触角范围内的配方、减少表面活性剂使用并降低成本。本研究证明了RAISE在闭环系统中自主地将液体配方与接触角测量关联起来的能力，并使用多目标BO高效识别符合研究人员定义标准的最优配方。 

---
# Real-Time Glass Detection and Reprojection using Sensor Fusion Onboard Aerial Robots 

**Title (ZH)**: 基于机载机器人传感器融合的实时玻璃检测与重塑投影 

**Authors**: Malakhi Hopkins, Varun Murali, Vijay Kumar, Camillo J Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2510.06518)  

**Abstract**: Autonomous aerial robots are increasingly being deployed in real-world scenarios, where transparent obstacles present significant challenges to reliable navigation and mapping. These materials pose a unique problem for traditional perception systems because they lack discernible features and can cause conventional depth sensors to fail, leading to inaccurate maps and potential collisions. To ensure safe navigation, robots must be able to accurately detect and map these transparent obstacles. Existing methods often rely on large, expensive sensors or algorithms that impose high computational burdens, making them unsuitable for low Size, Weight, and Power (SWaP) robots. In this work, we propose a novel and computationally efficient framework for detecting and mapping transparent obstacles onboard a sub-300g quadrotor. Our method fuses data from a Time-of-Flight (ToF) camera and an ultrasonic sensor with a custom, lightweight 2D convolution model. This specialized approach accurately detects specular reflections and propagates their depth into corresponding empty regions of the depth map, effectively rendering transparent obstacles visible. The entire pipeline operates in real-time, utilizing only a small fraction of a CPU core on an embedded processor. We validate our system through a series of experiments in both controlled and real-world environments, demonstrating the utility of our method through experiments where the robot maps indoor environments containing glass. Our work is, to our knowledge, the first of its kind to demonstrate a real-time, onboard transparent obstacle mapping system on a low-SWaP quadrotor using only the CPU. 

**Abstract (ZH)**: 自主飞行机器人在实际应用场景中越来越多，透明障碍物对可靠的导航和建图构成了显著挑战。这些材料给传统的感知系统带来了独特的问题，因为它们缺乏可区分的特征，并可能导致传统的深度传感器失效，从而产生不准确的地图并引发潜在碰撞。为了确保安全导航，机器人必须能够准确检测和测绘这些透明障碍物。现有方法通常依赖于大型、昂贵的传感器或计算负担高的算法，这使得它们不适合尺寸、重量和功率（SWaP）小巧的机器人。在这项工作中，我们提出了一种新颖且计算高效的框架，在一个重量小于300克的四旋翼飞行器上检测和测绘透明障碍物。该方法结合了飞行时间（ToF）相机数据和超声波传感器数据，并利用一种定制的轻量级二维卷积模型。这种专业方法准确检测镜面反射，并将它们的深度传播到深度图中的相应空白区域，有效使透明障碍物变得可见。整个管道实时运行，仅使用嵌入式处理器上的一小部分CPU核心。我们通过在受控和实际环境中的系列实验验证了我们的系统，展示了该方法在机器人测绘包含玻璃的室内环境中的实用性。据我们所知，这项工作是首次在低-SWaP四旋翼飞行器上仅使用CPU实现实时、机载透明障碍物测绘系统的案例。 

---
# What You Don't Know Can Hurt You: How Well do Latent Safety Filters Understand Partially Observable Safety Constraints? 

**Title (ZH)**: 你不知晓的可能会对你造成伤害：潜藏的安全过滤器对部分可观测安全约束的理解程度如何？ 

**Authors**: Matthew Kim, Kensuke Nakamura, Andrea Bajcsy  

**Link**: [PDF](https://arxiv.org/pdf/2510.06492)  

**Abstract**: Safe control techniques, such as Hamilton-Jacobi reachability, provide principled methods for synthesizing safety-preserving robot policies but typically assume hand-designed state spaces and full observability. Recent work has relaxed these assumptions via latent-space safe control, where state representations and dynamics are learned jointly through world models that reconstruct future high-dimensional observations (e.g., RGB images) from current observations and actions. This enables safety constraints that are difficult to specify analytically (e.g., spilling) to be framed as classification problems in latent space, allowing controllers to operate directly from raw observations. However, these methods assume that safety-critical features are observable in the learned latent state. We ask: when are latent state spaces sufficient for safe control? To study this, we examine temperature-based failures, comparable to overheating in cooking or manufacturing tasks, and find that RGB-only observations can produce myopic safety behaviors, e.g., avoiding seeing failure states rather than preventing failure itself. To predict such behaviors, we introduce a mutual information-based measure that identifies when observations fail to capture safety-relevant features. Finally, we propose a multimodal-supervised training strategy that shapes the latent state with additional sensory inputs during training, but requires no extra modalities at deployment, and validate our approach in simulation and on hardware with a Franka Research 3 manipulator preventing a pot of wax from overheating. 

**Abstract (ZH)**: 基于潜在状态空间的安全控制技术：以高温故障为例的研究 

---
# Active Next-Best-View Optimization for Risk-Averse Path Planning 

**Title (ZH)**: 风险规避路径规划中的主动下一个最佳视图优化 

**Authors**: Amirhossein Mollaei Khass, Guangyi Liu, Vivek Pandey, Wen Jiang, Boshu Lei, Kostas Daniilidis, Nader Motee  

**Link**: [PDF](https://arxiv.org/pdf/2510.06481)  

**Abstract**: Safe navigation in uncertain environments requires planning methods that integrate risk aversion with active perception. In this work, we present a unified framework that refines a coarse reference path by constructing tail-sensitive risk maps from Average Value-at-Risk statistics on an online-updated 3D Gaussian-splat Radiance Field. These maps enable the generation of locally safe and feasible trajectories. In parallel, we formulate Next-Best-View (NBV) selection as an optimization problem on the SE(3) pose manifold, where Riemannian gradient descent maximizes an expected information gain objective to reduce uncertainty most critical for imminent motion. Our approach advances the state-of-the-art by coupling risk-averse path refinement with NBV planning, while introducing scalable gradient decompositions that support efficient online updates in complex environments. We demonstrate the effectiveness of the proposed framework through extensive computational studies. 

**Abstract (ZH)**: 在不确定环境中安全导航需要结合风险规避与主动感知的规划方法。本文提出了一种统一框架，通过构建基于在线更新的3D高斯点辐射场平均价值-at-风险统计的尾敏感风险地图来细化粗参考路径。这些地图能够生成局部安全且可行的轨迹。同时，我们将最佳视图（NBV）选择形式化为SE(3)姿态流形上的优化问题，其中刘维尔梯度下降最大化预期信息增益目标，以减少即将发生的运动中最为关键的不确定性。我们的方法通过将风险规避路径细化与NBV规划耦合，并引入可扩展的梯度分解，支持复杂环境中高效的在线更新，从而超越了现有技术。我们通过广泛的眼动计算研究展示了所提出框架的有效性。 

---
# Constrained Natural Language Action Planning for Resilient Embodied Systems 

**Title (ZH)**: 面向 resilient 体域系统的约束自然语言动作规划 

**Authors**: Grayson Byrd, Corban Rivera, Bethany Kemp, Meghan Booker, Aurora Schmidt, Celso M de Melo, Lalithkumar Seenivasan, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2510.06357)  

**Abstract**: Replicating human-level intelligence in the execution of embodied tasks remains challenging due to the unconstrained nature of real-world environments. Novel use of large language models (LLMs) for task planning seeks to address the previously intractable state/action space of complex planning tasks, but hallucinations limit their reliability, and thus, viability beyond a research context. Additionally, the prompt engineering required to achieve adequate system performance lacks transparency, and thus, repeatability. In contrast to LLM planning, symbolic planning methods offer strong reliability and repeatability guarantees, but struggle to scale to the complexity and ambiguity of real-world tasks. We introduce a new robotic planning method that augments LLM planners with symbolic planning oversight to improve reliability and repeatability, and provide a transparent approach to defining hard constraints with considerably stronger clarity than traditional prompt engineering. Importantly, these augmentations preserve the reasoning capabilities of LLMs and retain impressive generalization in open-world environments. We demonstrate our approach in simulated and real-world environments. On the ALFWorld planning benchmark, our approach outperforms current state-of-the-art methods, achieving a near-perfect 99% success rate. Deployment of our method to a real-world quadruped robot resulted in 100% task success compared to 50% and 30% for pure LLM and symbolic planners across embodied pick and place tasks. Our approach presents an effective strategy to enhance the reliability, repeatability and transparency of LLM-based robot planners while retaining their key strengths: flexibility and generalizability to complex real-world environments. We hope that this work will contribute to the broad goal of building resilient embodied intelligent systems. 

**Abstract (ZH)**: 在执行具身任务中复制人类级别的智能仍然具有挑战性，因为现实世界环境的约束不定。新颖地使用大型语言模型（LLMs）进行任务规划旨在解决复杂规划任务中难以处理的状态/动作空间问题，但幻觉限制了它们在研究环境之外的可靠性与可行性。此外，为实现足够的系统性能所需的提示工程缺乏透明度，从而导致可重复性问题。与LLM规划相比，符号规划方法提供了强大的可靠性和可重复性保证，但难以扩展到现实世界任务的复杂性和模糊性。我们介绍了一种新的机器人规划方法，该方法通过符号规划监督来增强LLM规划者，从而提高可靠性和可重复性，并提供了一种定义硬约束的透明方法，比传统提示工程具有更强的明晰性。重要的是，这些增强保持了LLMs的推理能力，并在开放环境下保持了出色的泛化能力。我们在模拟和现实环境中的方法演示证明了这一点。在ALFWorld规划基准测试中，我们的方法超越了当前最先进的方法，实现了近乎完美的99%成功率。将我们的方法部署到一个现实世界的四足机器人上，在具身拾放任务中，其任务成功率达到了100%，而纯LLM和符号规划者分别仅为50%和30%。我们的方法为增强LLM基机器人规划者的可靠性和可重复性，同时保留其核心优势：灵活性和面向复杂现实环境的泛化能力，提供了一种有效策略。我们希望这项工作能为构建鲁棒的具身智能系统做出贡献。 

---
# A Formal gatekeeper Framework for Safe Dual Control with Active Exploration 

**Title (ZH)**: 一种正式的守门人框架，用于安全的双回路控制并伴随主动探索 

**Authors**: Kaleb Ben Naveed, Devansh R. Agrawal, Dimitra Panagou  

**Link**: [PDF](https://arxiv.org/pdf/2510.06351)  

**Abstract**: Planning safe trajectories under model uncertainty is a fundamental challenge. Robust planning ensures safety by considering worst-case realizations, yet ignores uncertainty reduction and leads to overly conservative behavior. Actively reducing uncertainty on-the-fly during a nominal mission defines the dual control problem. Most approaches address this by adding a weighted exploration term to the cost, tuned to trade off the nominal objective and uncertainty reduction, but without formal consideration of when exploration is beneficial. Moreover, safety is enforced in some methods but not in others. We propose a framework that integrates robust planning with active exploration under formal guarantees as follows: The key innovation and contribution is that exploration is pursued only when it provides a verifiable improvement without compromising safety. To achieve this, we utilize our earlier work on gatekeeper as an architecture for safety verification, and extend it so that it generates both safe and informative trajectories that reduce uncertainty and the cost of the mission, or keep it within a user-defined budget. The methodology is evaluated via simulation case studies on the online dual control of a quadrotor under parametric uncertainty. 

**Abstract (ZH)**: 在模型不确定性下的安全轨迹规划是基本挑战。通过鲁棒规划确保安全性需要考虑最坏情况的实现，但忽略了不确定性减少，导致行为过于保守。在名义任务中实时减少不确定性定义了双重控制问题。大多数方法通过添加加权探索项到成本中来解决这一问题，以平衡名义目标和不确定性减少，但没有正式考虑探索何时有益。此外，某些方法确保安全性，而其他方法则不确保。我们提出了一种框架，将鲁棒规划与在形式保证下的主动探索结合起来，如下：关键创新和贡献在于，仅在探索提供可验证的改进且不牺牲安全性时才进行探索。为了实现这一点，我们利用我们之前的工作作为安全性验证的架构，并将其扩展为生成既能保证安全又能提供信息的轨迹，这些轨迹可以减少不确定性并降低任务成本，或在用户定义的预算范围内。该方法通过模拟案例研究评估了四旋翼在参数不确定性下的在线双重控制。 

---
# Vi-TacMan: Articulated Object Manipulation via Vision and Touch 

**Title (ZH)**: Vi-TacMan: 通过视觉和触觉进行articulated对象操作 

**Authors**: Leiyao Cui, Zihang Zhao, Sirui Xie, Wenhuan Zhang, Zhi Han, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06339)  

**Abstract**: Autonomous manipulation of articulated objects remains a fundamental challenge for robots in human environments. Vision-based methods can infer hidden kinematics but can yield imprecise estimates on unfamiliar objects. Tactile approaches achieve robust control through contact feedback but require accurate initialization. This suggests a natural synergy: vision for global guidance, touch for local precision. Yet no framework systematically exploits this complementarity for generalized articulated manipulation. Here we present Vi-TacMan, which uses vision to propose grasps and coarse directions that seed a tactile controller for precise execution. By incorporating surface normals as geometric priors and modeling directions via von Mises-Fisher distributions, our approach achieves significant gains over baselines (all p<0.0001). Critically, manipulation succeeds without explicit kinematic models -- the tactile controller refines coarse visual estimates through real-time contact regulation. Tests on more than 50,000 simulated and diverse real-world objects confirm robust cross-category generalization. This work establishes that coarse visual cues suffice for reliable manipulation when coupled with tactile feedback, offering a scalable paradigm for autonomous systems in unstructured environments. 

**Abstract (ZH)**: 基于视觉与触觉的自适应物体 manipulation：细粒度执行的全局引导与局部精控 

---
# WristWorld: Generating Wrist-Views via 4D World Models for Robotic Manipulation 

**Title (ZH)**: 腕部世界：通过4D世界模型生成腕部视角以实现机器人 manipulation 

**Authors**: Zezhong Qian, Xiaowei Chi, Yuming Li, Shizun Wang, Zhiyuan Qin, Xiaozhu Ju, Sirui Han, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07313)  

**Abstract**: Wrist-view observations are crucial for VLA models as they capture fine-grained hand-object interactions that directly enhance manipulation performance. Yet large-scale datasets rarely include such recordings, resulting in a substantial gap between abundant anchor views and scarce wrist views. Existing world models cannot bridge this gap, as they require a wrist-view first frame and thus fail to generate wrist-view videos from anchor views alone. Amid this gap, recent visual geometry models such as VGGT emerge with geometric and cross-view priors that make it possible to address extreme viewpoint shifts. Inspired by these insights, we propose WristWorld, the first 4D world model that generates wrist-view videos solely from anchor views. WristWorld operates in two stages: (i) Reconstruction, which extends VGGT and incorporates our Spatial Projection Consistency (SPC) Loss to estimate geometrically consistent wrist-view poses and 4D point clouds; (ii) Generation, which employs our video generation model to synthesize temporally coherent wrist-view videos from the reconstructed perspective. Experiments on Droid, Calvin, and Franka Panda demonstrate state-of-the-art video generation with superior spatial consistency, while also improving VLA performance, raising the average task completion length on Calvin by 3.81% and closing 42.4% of the anchor-wrist view gap. 

**Abstract (ZH)**: 腕视图观察对于VLA模型至关重要，因为它们捕捉到精细的手-object交互，直接提升了操作表现。然而，大规模数据集很少包含此类记录，导致丰富锚视图与稀缺腕视图之间存在巨大差距。现有世界模型无法弥合这一差距，因为它们需要腕视图初始帧，从而无法仅从锚视图生成腕视图视频。在此差距中，最近的视觉几何模型如VGGT凭借几何先验和跨视图先验，使得应对极端视角转换成为可能。受此启发，我们提出了WristWorld，这是首个仅从锚视图生成腕视图视频的4D世界模型。WristWorld分为两个阶段：(i) 重构阶段，该阶段扩展了VGGT并结合了我们的空间投影一致性（SPC）损失以估计几何上一致的腕视图姿态和4D点云；(ii) 生成阶段，该阶段使用我们的视频生成模型从重构视点合成时空连贯的腕视图视频。实验在Droid、Calvin和Franka Panda上展示了最先进的视频生成，具有优越的空间一致性，同时提升了VLA性能，提高了Calvin上任务平均完成长度3.81%，并关闭了42.4%的锚视图-腕视图差距。 

---
# ELMUR: External Layer Memory with Update/Rewrite for Long-Horizon RL 

**Title (ZH)**: ELMUR：用于长期展望RL的外部层记忆更新/重写方法 

**Authors**: Egor Cherepanov, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2510.07151)  

**Abstract**: Real-world robotic agents must act under partial observability and long horizons, where key cues may appear long before they affect decision making. However, most modern approaches rely solely on instantaneous information, without incorporating insights from the past. Standard recurrent or transformer models struggle with retaining and leveraging long-term dependencies: context windows truncate history, while naive memory extensions fail under scale and sparsity. We propose ELMUR (External Layer Memory with Update/Rewrite), a transformer architecture with structured external memory. Each layer maintains memory embeddings, interacts with them via bidirectional cross-attention, and updates them through an Least Recently Used (LRU) memory module using replacement or convex blending. ELMUR extends effective horizons up to 100,000 times beyond the attention window and achieves a 100% success rate on a synthetic T-Maze task with corridors up to one million steps. In POPGym, it outperforms baselines on more than half of the tasks. On MIKASA-Robo sparse-reward manipulation tasks with visual observations, it nearly doubles the performance of strong baselines. These results demonstrate that structured, layer-local external memory offers a simple and scalable approach to decision making under partial observability. 

**Abstract (ZH)**: 基于外部层次记忆更新/重写的真实世界机器人代理超越观察极限的Transformer架构 

---
# Generative World Modelling for Humanoids: 1X World Model Challenge Technical Report 

**Title (ZH)**: 人形机器人生成式世界建模：1X世界模型挑战技术报告 

**Authors**: Riccardo Mereu, Aidan Scannell, Yuxin Hou, Yi Zhao, Aditya Jitta, Antonio Dominguez, Luigi Acerbi, Amos Storkey, Paul Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07092)  

**Abstract**: World models are a powerful paradigm in AI and robotics, enabling agents to reason about the future by predicting visual observations or compact latent states. The 1X World Model Challenge introduces an open-source benchmark of real-world humanoid interaction, with two complementary tracks: sampling, focused on forecasting future image frames, and compression, focused on predicting future discrete latent codes. For the sampling track, we adapt the video generation foundation model Wan-2.2 TI2V-5B to video-state-conditioned future frame prediction. We condition the video generation on robot states using AdaLN-Zero, and further post-train the model using LoRA. For the compression track, we train a Spatio-Temporal Transformer model from scratch. Our models achieve 23.0 dB PSNR in the sampling task and a Top-500 CE of 6.6386 in the compression task, securing 1st place in both challenges. 

**Abstract (ZH)**: World模型是人工智能和机器人领域的强大范式，能够使智能体通过预测视觉观察或紧凑的潜在状态来推理未来。1X World模型挑战引入了一个开放源代码的现实世界类人交互基准，包含两个互补赛道：采样赛道，侧重于预测未来图像帧；压缩赛道，侧重于预测未来离散潜在代码。在采样赛道中，我们基于视频生成基础模型Wan-2.2 TI2V-5B对视频状态条件下的未来帧预测进行了调整，使用AdaLN-Zero对机器人状态进行条件设置，并进一步使用LoRA对模型进行后训练。在压缩赛道中，我们从零开始训练了一个时空变换器模型。我们的模型在采样任务中实现了23.0 dB的PSNR，在压缩任务中获得了6.6386的Top-500 CE，分别在两个挑战中获得第一名。 

---
# Artists' Views on Robotics Involvement in Painting Productions 

**Title (ZH)**: 艺术家对机器人参与绘画生产的看法 

**Authors**: Francesca Cocchella, Nilay Roy Choudhury, Eric Chen, Patrícia Alves-Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2510.07063)  

**Abstract**: As robotic technologies evolve, their potential in artistic creation becomes an increasingly relevant topic of inquiry. This study explores how professional abstract artists perceive and experience co-creative interactions with an autonomous painting robotic arm. Eight artists engaged in six painting sessions -- three with a human partner, followed by three with the robot -- and subsequently participated in semi-structured interviews analyzed through reflexive thematic analysis. Human-human interactions were described as intuitive, dialogic, and emotionally engaging, whereas human-robot sessions felt more playful and reflective, offering greater autonomy and prompting for novel strategies to overcome the system's limitations. This work offers one of the first empirical investigations into artists' lived experiences with a robot, highlighting the value of long-term engagement and a multidisciplinary approach to human-robot co-creation. 

**Abstract (ZH)**: 随着机器人技术的发展，其在艺术创作中的潜在价值成为了一个日益相关的研究话题。本研究探讨了专业抽象艺术家如何感知和体验与自主画笔机器人进行共创互动的过程。八位艺术家参与了六次绘画会话——三次与人类伙伴合作，随后三次与机器人合作——并随后参与了半结构化访谈，通过反思性主题分析进行分析。人类-人类互动被描述为直观的、对话式的和情感丰富的，而人-机器人的会话则感觉更加轻松和反思性的，提供了更多的自主性，并促使提出新的策略来克服系统的限制。本研究提供了关于艺术家与机器人共创造生活体验的第一个实证研究之一，突显了长期参与和跨学科方法在人类-机器人共创造过程中的价值。 

---
# Introspection in Learned Semantic Scene Graph Localisation 

**Title (ZH)**: 基于学习的语义场景图定位中的内省分析 

**Authors**: Manshika Charvi Bissessur, Efimia Panagiotaki, Daniele De Martini  

**Link**: [PDF](https://arxiv.org/pdf/2510.07053)  

**Abstract**: This work investigates how semantics influence localisation performance and robustness in a learned self-supervised, contrastive semantic localisation framework. After training a localisation network on both original and perturbed maps, we conduct a thorough post-hoc introspection analysis to probe whether the model filters environmental noise and prioritises distinctive landmarks over routine clutter. We validate various interpretability methods and present a comparative reliability analysis. Integrated gradients and Attention Weights consistently emerge as the most reliable probes of learned behaviour. A semantic class ablation further reveals an implicit weighting in which frequent objects are often down-weighted. Overall, the results indicate that the model learns noise-robust, semantically salient relations about place definition, thereby enabling explainable registration under challenging visual and structural variations. 

**Abstract (ZH)**: 本研究探讨语义如何影响学习自监督对比语义局部化框架中的局部化性能和鲁棒性。通过在原始和扰动地图上训练局部化网络后，我们进行详细的事后内省分析，探究模型是否过滤环境噪声并优先处理独特地标而非常规杂乱。我们验证了多种可解释性方法，并进行了可靠性对比分析。集成梯度和注意力权重一致地显示出对学习行为最可靠的探针。语义类别消融进一步揭示了一个隐含的加权方式，其中频繁出现的物体通常被降低权重。总体而言，结果表明模型学习了对噪声具有鲁棒性且语义上显著的地方定义关系，从而在具有挑战性的视觉和结构变化下实现可解释的配准。 

---
# DecompGAIL: Learning Realistic Traffic Behaviors with Decomposed Multi-Agent Generative Adversarial Imitation Learning 

**Title (ZH)**: DecompGAIL：分解多智能体生成对抗模仿学习中的现实交通行为学习 

**Authors**: Ke Guo, Haochen Liu, Xiaojun Wu, Chen Lv  

**Link**: [PDF](https://arxiv.org/pdf/2510.06913)  

**Abstract**: Realistic traffic simulation is critical for the development of autonomous driving systems and urban mobility planning, yet existing imitation learning approaches often fail to model realistic traffic behaviors. Behavior cloning suffers from covariate shift, while Generative Adversarial Imitation Learning (GAIL) is notoriously unstable in multi-agent settings. We identify a key source of this instability: irrelevant interaction misguidance, where a discriminator penalizes an ego vehicle's realistic behavior due to unrealistic interactions among its neighbors. To address this, we propose Decomposed Multi-agent GAIL (DecompGAIL), which explicitly decomposes realism into ego-map and ego-neighbor components, filtering out misleading neighbor: neighbor and neighbor: map interactions. We further introduce a social PPO objective that augments ego rewards with distance-weighted neighborhood rewards, encouraging overall realism across agents. Integrated into a lightweight SMART-based backbone, DecompGAIL achieves state-of-the-art performance on the WOMD Sim Agents 2025 benchmark. 

**Abstract (ZH)**: DecompGAIL: Decomposing Realism for Multi-Agent GAIL in Traffic Simulation 

---
# HARP-NeXt: High-Speed and Accurate Range-Point Fusion Network for 3D LiDAR Semantic Segmentation 

**Title (ZH)**: HARP-NeXt：高-Speed和高精度的3D LiDAR语义分割范围点融合网络 

**Authors**: Samir Abou Haidar, Alexandre Chariot, Mehdi Darouich, Cyril Joly, Jean-Emmanuel Deschaud  

**Link**: [PDF](https://arxiv.org/pdf/2510.06876)  

**Abstract**: LiDAR semantic segmentation is crucial for autonomous vehicles and mobile robots, requiring high accuracy and real-time processing, especially on resource-constrained embedded systems. Previous state-of-the-art methods often face a trade-off between accuracy and speed. Point-based and sparse convolution-based methods are accurate but slow due to the complexity of neighbor searching and 3D convolutions. Projection-based methods are faster but lose critical geometric information during the 2D projection. Additionally, many recent methods rely on test-time augmentation (TTA) to improve performance, which further slows the inference. Moreover, the pre-processing phase across all methods increases execution time and is demanding on embedded platforms. Therefore, we introduce HARP-NeXt, a high-speed and accurate LiDAR semantic segmentation network. We first propose a novel pre-processing methodology that significantly reduces computational overhead. Then, we design the Conv-SE-NeXt feature extraction block to efficiently capture representations without deep layer stacking per network stage. We also employ a multi-scale range-point fusion backbone that leverages information at multiple abstraction levels to preserve essential geometric details, thereby enhancing accuracy. Experiments on the nuScenes and SemanticKITTI benchmarks show that HARP-NeXt achieves a superior speed-accuracy trade-off compared to all state-of-the-art methods, and, without relying on ensemble models or TTA, is comparable to the top-ranked PTv3, while running 24$\times$ faster. The code is available at this https URL 

**Abstract (ZH)**: LiDAR语义分割对自主车辆和移动机器人至关重要，需要高精度和实时处理，特别是在资源受限的嵌入式系统中。先前的先进方法往往在精度和速度之间存在权衡。基于点的方法和稀疏卷积方法虽然准确但速度较慢，因为邻居搜索和三维卷积的复杂性。投影方法速度快但会在二维投影过程中丢失重要的几何信息。此外，许多最近的方法依赖测试时增强（TTA）来提高性能，这进一步减慢了推理速度。而且，所有方法的预处理阶段都会增加执行时间，并对嵌入式平台提出更高要求。因此，我们引入了HARP-NeXt，一种高速高精度的LiDAR语义分割网络。我们首先提出了一种新的预处理方法，大幅减少了计算开销。然后，我们设计了Conv-SE-NeXt特征提取块，以有效地捕获表示而不必在网络的每一阶段进行深层次层堆叠。我们还采用了一种多尺度范围点融合骨干网络，利用多个抽象层次上的信息来保留关键的几何细节，从而提高准确性。在nuScenes和SemanticKITTI基准上的实验表明，HARP-NeXt在速度和精度之间实现了优于所有先进方法的妥协，并且在无需依赖集成模型或TTA的情况下，与排名靠前的PTv3相当，但运行速度快24倍。代码可在以下链接获取。 

---
# Through the Perspective of LiDAR: A Feature-Enriched and Uncertainty-Aware Annotation Pipeline for Terrestrial Point Cloud Segmentation 

**Title (ZH)**: 从LiDAR视角出发：一种用于地表点云分割的特征丰富且考虑不确定性注释管道 

**Authors**: Fei Zhang, Rob Chancia, Josie Clapp, Amirhossein Hassanzadeh, Dimah Dera, Richard MacKenzie, Jan van Aardt  

**Link**: [PDF](https://arxiv.org/pdf/2510.06582)  

**Abstract**: Accurate semantic segmentation of terrestrial laser scanning (TLS) point clouds is limited by costly manual annotation. We propose a semi-automated, uncertainty-aware pipeline that integrates spherical projection, feature enrichment, ensemble learning, and targeted annotation to reduce labeling effort, while sustaining high accuracy. Our approach projects 3D points to a 2D spherical grid, enriches pixels with multi-source features, and trains an ensemble of segmentation networks to produce pseudo-labels and uncertainty maps, the latter guiding annotation of ambiguous regions. The 2D outputs are back-projected to 3D, yielding densely annotated point clouds supported by a three-tier visualization suite (2D feature maps, 3D colorized point clouds, and compact virtual spheres) for rapid triage and reviewer guidance. Using this pipeline, we build Mangrove3D, a semantic segmentation TLS dataset for mangrove forests. We further evaluate data efficiency and feature importance to address two key questions: (1) how much annotated data are needed and (2) which features matter most. Results show that performance saturates after ~12 annotated scans, geometric features contribute the most, and compact nine-channel stacks capture nearly all discriminative power, with the mean Intersection over Union (mIoU) plateauing at around 0.76. Finally, we confirm the generalization of our feature-enrichment strategy through cross-dataset tests on ForestSemantic and Semantic3D.
Our contributions include: (i) a robust, uncertainty-aware TLS annotation pipeline with visualization tools; (ii) the Mangrove3D dataset; and (iii) empirical guidance on data efficiency and feature importance, thus enabling scalable, high-quality segmentation of TLS point clouds for ecological monitoring and beyond. The dataset and processing scripts are publicly available at this https URL. 

**Abstract (ZH)**: 一种基于球面投影和特征增强的半自动化激光扫描点云语义分割方法：减少标注努力并维持高精度 

---
# Terrain-Aided Navigation Using a Point Cloud Measurement Sensor 

**Title (ZH)**: 基于点云测量传感器的地形辅助导航 

**Authors**: Abdülbaki Şanlan, Fatih Erol, Murad Abu-Khalaf, Emre Koyuncu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06470)  

**Abstract**: We investigate the use of a point cloud measurement in terrain-aided navigation. Our goal is to aid an inertial navigation system, by exploring ways to generate a useful measurement innovation error for effective nonlinear state estimation. We compare two such measurement models that involve the scanning of a digital terrain elevation model: a) one that is based on typical ray-casting from a given pose, that returns the predicted point cloud measurement from that pose, and b) another computationally less intensive one that does not require raycasting and we refer to herein as a sliding grid. Besides requiring a pose, it requires the pattern of the point cloud measurement itself and returns a predicted point cloud measurement. We further investigate the observability properties of the altitude for both measurement models. As a baseline, we compare the use of a point cloud measurement performance to the use of a radar altimeter and show the gains in accuracy. We conclude by showing that a point cloud measurement outperforms the use of a radar altimeter, and the point cloud measurement model to use depends on the computational resources 

**Abstract (ZH)**: 基于地形辅助导航中点云测量的应用研究：一种提高非线性状态估计有效性的测量创新误差方法探讨 

---
# Three-dimensional Integrated Guidance and Control for Leader-Follower Flexible Formation of Fixed Wing UAVs 

**Title (ZH)**: 固定翼无人机领导者-跟随者柔性编队的三维集成指导与控制 

**Authors**: Praveen Kumar Ranjan, Abhinav Sinha, Yongcan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06394)  

**Abstract**: This paper presents a nonlinear integrated guidance and control (IGC) approach for flexible leader-follower formation flight of fixed-wing unmanned aerial vehicles (UAVs) while accounting for high-fidelity aerodynamics and thrust dynamics. Unlike conventional leader-follower schemes that fix the follower's position relative to the leader, the follower is steered to maintain range and bearing angles (which is the angle between its velocity vector and its line-of-sight (LOS) with respect to the leader) arbitrarily close to the prescribed values, enabling the follower to maintain formation on a hemispherical region behind the leader. The proposed IGC framework directly maps leader-follower relative range dynamics to throttle commands, and the follower's velocity orientation relative to the LOS to aerodynamic control surface deflections. This enables synergism between guidance and control subsystems. The control design uses a dynamic surface control-based backstepping approach to achieve convergence to the desired formation set, where Lyapunov barrier functions are incorporated to ensure the follower's bearing angle is constrained within specified bounds. Rigorous stability analysis guarantees uniform ultimate boundedness of all error states and strict constraint satisfaction in the presence of aerodynamic nonlinearities. The proposed flexible formation scheme allows the follower to have an orientation mismatch relative to the leader to execute anticipatory reconfiguration by transitioning between the relative positions in the admissible formation set when the leader aggressively maneuvers. The proposed IGC law relies only on relative information and onboard sensors without the information about the leader's maneuver, making it suitable for GPS-denied or non-cooperative scenarios. Finally, we present simulation results to vindicate the effectiveness and robustness of our approach. 

**Abstract (ZH)**: 本文提出了一种考虑高保真气动和推力动力学的柔性 líder-follower编队飞行非线性集成导航与控制方法，实现了固定翼无人机编队飞行中跟随器相对于领导者任意接近预定距离和航向角的维持，使得跟随器可以保持在领导者后方的半球区域内飞行。该提出的集成导航与控制框架直接将领导者-跟随者相对距离动力学映射到油门命令，将跟随器相对于视线方向的速度方向映射到气动控制面偏转，实现了导航与控制子系统之间的协同作用。控制设计采用基于动态表面控制的回步方法，将跟随器收敛到期望编队状态，通过Lypunov屏障函数确保跟随器的航向角在指定范围内。严格稳定性分析保证了所有误差状态的统一最终有界性，并在气动非线性存在的情况下满足严格约束。提出的柔性编队方案允许跟随器相对于领导者具有方向不匹配，以便通过在可接受编队集中切换相对位置来执行前瞻性重构，当领导者剧烈机动时。所提出的集成导航与控制法则仅依赖于相对信息和机载传感器，而不依赖于领导者机动信息，使其适用于GPS受限或非合作场景。最后，我们提供了仿真结果以验证我们方法的有效性和鲁棒性。 

---
