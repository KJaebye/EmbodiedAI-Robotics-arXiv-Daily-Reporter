# CLONE: Closed-Loop Whole-Body Humanoid Teleoperation for Long-Horizon Tasks 

**Title (ZH)**: CLONE: 闭环全身人形机器人力控远程操作长时任务 

**Authors**: Yixuan Li, Yutang Lin, Jieming Cui, Tengyu Liu, Wei Liang, Yixin Zhu, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08931)  

**Abstract**: Humanoid teleoperation plays a vital role in demonstrating and collecting data for complex humanoid-scene interactions. However, current teleoperation systems face critical limitations: they decouple upper- and lower-body control to maintain stability, restricting natural coordination, and operate open-loop without real-time position feedback, leading to accumulated drift. The fundamental challenge is achieving precise, coordinated whole-body teleoperation over extended durations while maintaining accurate global positioning. Here we show that an MoE-based teleoperation system, CLONE, with closed-loop error correction enables unprecedented whole-body teleoperation fidelity, maintaining minimal positional drift over long-range trajectories using only head and hand tracking from an MR headset. Unlike previous methods that either sacrifice coordination for stability or suffer from unbounded drift, CLONE learns diverse motion skills while preventing tracking error accumulation through real-time feedback, enabling complex coordinated movements such as ``picking up objects from the ground.'' These results establish a new milestone for whole-body humanoid teleoperation for long-horizon humanoid-scene interaction tasks. 

**Abstract (ZH)**: 基于MoE的人机遥控系统CLONE在长时间大范围全身人形场景遥控中的卓越表现 

---
# Human-Robot Teaming Field Deployments: A Comparison Between Verbal and Non-verbal Communication 

**Title (ZH)**: 人机协同现场部署：口头沟通与非口头沟通的比较 

**Authors**: Tauhid Tanjim, Promise Ekpo, Huajie Cao, Jonathan St. George, Kevin Ching, Hee Rin Lee, Angelique Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2506.08890)  

**Abstract**: Healthcare workers (HCWs) encounter challenges in hospitals, such as retrieving medical supplies quickly from crash carts, which could potentially result in medical errors and delays in patient care. Robotic crash carts (RCCs) have shown promise in assisting healthcare teams during medical tasks through guided object searches and task reminders. Limited exploration has been done to determine what communication modalities are most effective and least disruptive to patient care in real-world settings. To address this gap, we conducted a between-subjects experiment comparing the RCC's verbal and non-verbal communication of object search with a standard crash cart in resuscitation scenarios to understand the impact of robot communication on workload and attitudes toward using robots in the workplace. Our findings indicate that verbal communication significantly reduced mental demand and effort compared to visual cues and with a traditional crash cart. Although frustration levels were slightly higher during collaborations with the robot compared to a traditional cart, these research insights provide valuable implications for human-robot teamwork in high-stakes environments. 

**Abstract (ZH)**: 机器人监护车在急救场景中通过语音与非言语沟通方式辅助医疗任务对工作负荷及工作场所使用机器人态度的影响：一项实证研究 

---
# MOMAV: A highly symmetrical fully-actuated multirotor drone using optimizing control allocation 

**Title (ZH)**: MOMAV：一种采用优化控制分配的高度对称全驱动多旋翼无人机 

**Authors**: Marco Ruggia  

**Link**: [PDF](https://arxiv.org/pdf/2506.08868)  

**Abstract**: MOMAV (Marco's Omnidirectional Micro Aerial Vehicle) is a multirotor drone that is fully actuated, meaning it can control its orientation independently of its position. MOMAV is also highly symmetrical, making its flight efficiency largely unaffected by its current orientation. These characteristics are achieved by a novel drone design where six rotor arms align with the vertices of an octahedron, and where each arm can actively rotate along its long axis. Various standout features of MOMAV are presented: The high flight efficiency compared to arm configuration of other fully-actuated drones, the design of an original rotating arm assembly featuring slip-rings used to enable continuous arm rotation, and a novel control allocation algorithm based on sequential quadratic programming (SQP) used to calculate throttle and arm-angle setpoints in flight. Flight tests have shown that MOMAV is able to achieve remarkably low mean position/orientation errors of 6.6mm, 2.1° ({\sigma}: 3.0mm, 1.0°) when sweeping position setpoints, and 11.8mm, 3.3° ({\sigma}: 8.6mm, 2.0°) when sweeping orientation setpoints. 

**Abstract (ZH)**: 马科斯的全向微型飞行器（MOMAV）是一种全自主飞行的多旋翼无人机，能够独立控制姿态而不受位置影响。MOMAV具有高度对称性，其飞行效率在当前姿态变化时几乎不受影响。这些特性通过一种创新的无人机设计实现，该设计将六个旋翼臂排列成八面体的顶点，并且每根旋翼臂都可以沿其长轴主动旋转。MOMAV的一些突出特征包括与其它全自主飞行旋翼机相比的高飞行效率、一种新型旋转臂组件设计，该设计采用滑动环实现连续旋转，以及基于序列二次规划（SQP）的新型控制分配算法，用于飞行中计算油门和旋翼臂角度设定值。飞行测试表明，MOMAV在位置设定点扫描时可以实现显著低的平均位置/姿态偏差6.6毫米、2.1度（σ：3.0毫米、1.0度），在姿态设定点扫描时可以实现11.8毫米、3.3度（σ：8.6毫米、2.0度）的平均偏差。 

---
# Fast Estimation of Globally Optimal Independent Contact Regions for Robust Grasping and Manipulation 

**Title (ZH)**: 全局最优独立接触区域的快速估算：提高抓取与操作的鲁棒性 

**Authors**: Jonathan P. King, Harnoor Ahluwalia, Michael Zhang, Nancy S. Pollard  

**Link**: [PDF](https://arxiv.org/pdf/2506.08856)  

**Abstract**: This work presents a fast anytime algorithm for computing globally optimal independent contact regions (ICRs). ICRs are regions such that one contact within each region enables a valid grasp. Locations of ICRs can provide guidance for grasp and manipulation planning, learning, and policy transfer. However, ICRs for modern applications have been little explored, in part due to the expense of computing them, as they have a search space exponential in the number of contacts. We present a divide and conquer algorithm based on incremental n-dimensional Delaunay triangulation that produces results with bounded suboptimality in times sufficient for real-time planning. This paper presents the base algorithm for grasps where contacts lie within a plane. Our experiments show substantial benefits over competing grasp quality metrics and speedups of 100X and more for competing approaches to computing ICRs. We explore robustness of a policy guided by ICRs and outline a path to general 3D implementation. Code will be released on publication to facilitate further development and applications. 

**Abstract (ZH)**: 本工作提出一种快速的任意时序算法，用于计算全局最优独立接触区域（ICRs）。ICRs是这样一些区域：每个区域内的一个接触点能形成一个有效的握持。ICRs的位置可为抓取和操作规划、学习和策略转移提供指导。然而，由于计算ICRs的成本高昂，特别是在接触点数量增加时搜索空间呈指数增长，现代应用中的ICRs尚未得到充分探索。我们提出了一种基于增量n维Delaunay三角剖分的分而治之算法，能够在满足实时规划要求的时间内产生带有有界次优性的结果。本文提出的基算法适用于接触点位于平面内的抓取情况。我们的实验表明，该算法在抓取质量度量方面优于竞争对手，并且在计算ICRs的竞争方法上实现了100倍以上的加速。我们探讨了由ICRs引导的策略的鲁棒性，并概述了一条通向三维通用实现的方法。代码将在发表后公开，以促进进一步的发展和应用。 

---
# Deploying SICNav in the Field: Safe and Interactive Crowd Navigation using MPC and Bilevel Optimization 

**Title (ZH)**: 将SICNav应用于现实场景：基于MPC和双层优化的的安全互动人群导航 

**Authors**: Sepehr Samavi, Garvish Bhutani, Florian Shkurti, Angela P. Schoellig  

**Link**: [PDF](https://arxiv.org/pdf/2506.08851)  

**Abstract**: Safe and efficient navigation in crowded environments remains a critical challenge for robots that provide a variety of service tasks such as food delivery or autonomous wheelchair mobility. Classical robot crowd navigation methods decouple human motion prediction from robot motion planning, which neglects the closed-loop interactions between humans and robots. This lack of a model for human reactions to the robot plan (e.g. moving out of the way) can cause the robot to get stuck. Our proposed Safe and Interactive Crowd Navigation (SICNav) method is a bilevel Model Predictive Control (MPC) framework that combines prediction and planning into one optimization problem, explicitly modeling interactions among agents. In this paper, we present a systems overview of the crowd navigation platform we use to deploy SICNav in previously unseen indoor and outdoor environments. We provide a preliminary analysis of the system's operation over the course of nearly 7 km of autonomous navigation over two hours in both indoor and outdoor environments. 

**Abstract (ZH)**: 安全高效的拥挤环境导航仍然是为食物配送或自主轮椅移动等各类服务任务提供服务的机器人面临的关键挑战。经典的机器人人群导航方法将人类运动预测与机器人运动规划分离，忽略了人类与机器人之间的闭环交互。忽视了对机器人计划的人类反应模型（例如，主动避让）可能导致机器人陷入困境。我们提出的Safe and Interactive Crowd Navigation（SICNav）方法是一种二层模型预测控制（MPC）框架，将预测和规划结合为一个优化问题，明确地建模了代理之间的交互。在本文中，我们介绍了用于部署SICNav的 crowds navigation 平台的系统概述，该平台已在未见过的室内和室外环境中进行部署。我们提供了系统运行的初步分析，涵盖两小时内在室内和室外环境中近7公里的自主导航。 

---
# MoRE: Mixture of Residual Experts for Humanoid Lifelike Gaits Learning on Complex Terrains 

**Title (ZH)**: MoRE: 混合剩余专家模型用于复杂地形上的人形类生动力学学习 

**Authors**: Dewei Wang, Xinmiao Wang, Xinzhe Liu, Jiyuan Shi, Yingnan Zhao, Chenjia Bai, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.08840)  

**Abstract**: Humanoid robots have demonstrated robust locomotion capabilities using Reinforcement Learning (RL)-based approaches. Further, to obtain human-like behaviors, existing methods integrate human motion-tracking or motion prior in the RL framework. However, these methods are limited in flat terrains with proprioception only, restricting their abilities to traverse challenging terrains with human-like gaits. In this work, we propose a novel framework using a mixture of latent residual experts with multi-discriminators to train an RL policy, which is capable of traversing complex terrains in controllable lifelike gaits with exteroception. Our two-stage training pipeline first teaches the policy to traverse complex terrains using a depth camera, and then enables gait-commanded switching between human-like gait patterns. We also design gait rewards to adjust human-like behaviors like robot base height. Simulation and real-world experiments demonstrate that our framework exhibits exceptional performance in traversing complex terrains, and achieves seamless transitions between multiple human-like gait patterns. 

**Abstract (ZH)**: 基于混合潜在残差专家和多判别器的强化学习框架：具备外部感知的复杂地形可控仿人步态导航 

---
# FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency 

**Title (ZH)**: FreqPolicy: 基于频率一致性的一种高效流驱动视运动策略 

**Authors**: Yifei Su, Ning Liu, Dong Chen, Zhen Zhao, Kun Wu, Meng Li, Zhiyuan Xu, Zhengping Che, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08822)  

**Abstract**: Generative modeling-based visuomotor policies have been widely adopted in robotic manipulation attributed to their ability to model multimodal action distributions. However, the high inference cost of multi-step sampling limits their applicability in real-time robotic systems. To address this issue, existing approaches accelerate the sampling process in generative modeling-based visuomotor policies by adapting acceleration techniques originally developed for image generation. Despite this progress, a major distinction remains: image generation typically involves producing independent samples without temporal dependencies, whereas robotic manipulation involves generating time-series action trajectories that require continuity and temporal coherence. To effectively exploit temporal information in robotic manipulation, we propose FreqPolicy, a novel approach that first imposes frequency consistency constraints on flow-based visuomotor policies. Our work enables the action model to capture temporal structure effectively while supporting efficient, high-quality one-step action generation. We introduce a frequency consistency constraint that enforces alignment of frequency-domain action features across different timesteps along the flow, thereby promoting convergence of one-step action generation toward the target distribution. In addition, we design an adaptive consistency loss to capture structural temporal variations inherent in robotic manipulation tasks. We assess FreqPolicy on 53 tasks across 3 simulation benchmarks, proving its superiority over existing one-step action generators. We further integrate FreqPolicy into the vision-language-action (VLA) model and achieve acceleration without performance degradation on the 40 tasks of Libero. Besides, we show efficiency and effectiveness in real-world robotic scenarios with an inference frequency 93.5Hz. The code will be publicly available. 

**Abstract (ZH)**: 基于生成建模的感知运动策略在机器人 manipulation 中广泛应用，得益于其建模多模态动作分布的能力。然而，多步采样的高推理成本限制了其在实时机器人系统中的应用。为解决这一问题，现有方法通过适应最初为图像生成设计的加速技术，加快生成建模的感知运动策略的采样过程。尽管取得了进展，但仍存在一个重要区别：图像生成通常涉及生成独立样本而无需时间依赖性，而机器人 manipulation 则涉及生成需要连续性和时间一致性的时序动作轨迹。为了有效利用机器人 manipulation 中的时间信息，我们提出了 FreqPolicy，这是一种新颖的方法，首先在基于流的感知运动策略中施加频率一致性约束。我们的工作使动作模型能够有效地捕捉时间结构，同时支持高效、高质量的一步动作生成。我们引入了一个频率一致性约束，强制频率域动作特征在流的不同时间步长上对齐，从而促进一步动作生成向目标分布收敛。此外，我们设计了一种自适应一致性损失来捕捉机器人 manipulation 任务中内在的时间结构变化。我们在 3 个仿真基准上的 53 个任务中评估了 FreqPolicy，证明其优于现有的一步动作生成器。我们进一步将 FreqPolicy 集成到视觉-语言-动作（VLA）模型中，在 Libero 的 40 个任务上实现加速且不降低性能。此外，我们在每秒 93.5 次推理的实时机器人场景中展示了其效率和有效性。代码将公开。 

---
# Towards Biosignals-Free Autonomous Prosthetic Hand Control via Imitation Learning 

**Title (ZH)**: 基于imitation learning的无生物信号自主假手控制 

**Authors**: Kaijie Shi, Wanglong Lu, Hanli Zhao, Vinicius Prado da Fonseca, Ting Zou, Xianta Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08795)  

**Abstract**: Limb loss affects millions globally, impairing physical function and reducing quality of life. Most traditional surface electromyographic (sEMG) and semi-autonomous methods require users to generate myoelectric signals for each control, imposing physically and mentally taxing demands. This study aims to develop a fully autonomous control system that enables a prosthetic hand to automatically grasp and release objects of various shapes using only a camera attached to the wrist. By placing the hand near an object, the system will automatically execute grasping actions with a proper grip force in response to the hand's movements and the environment. To release the object being grasped, just naturally place the object close to the table and the system will automatically open the hand. Such a system would provide individuals with limb loss with a very easy-to-use prosthetic control interface and greatly reduce mental effort while using. To achieve this goal, we developed a teleoperation system to collect human demonstration data for training the prosthetic hand control model using imitation learning, which mimics the prosthetic hand actions from human. Through training the model using only a few objects' data from one single participant, we have shown that the imitation learning algorithm can achieve high success rates, generalizing to more individuals and unseen objects with a variation of weights. The demonstrations are available at \href{this https URL}{this https URL} 

**Abstract (ZH)**: 肢体损失影响全球数百万人，损害身体功能并降低生活质量。大多数传统的表面肌电图(sEMG)和半自主方法要求用户为每个控制生成肌电信号，这给身心带来了巨大的负担。本研究旨在开发一个完全自主的控制系统，使装有假手的用户能够仅通过手腕上的摄像头自动抓取和释放各种形状的物体。通过将手置于物体附近，系统将根据手部动作和环境自动执行适当的握持动作。要释放被抓取的物体，只需自然地将物体放在桌面上，系统将自动打开假手。这样的系统将为肢体损失的个人提供一个非常易于使用的假手控制界面，并大大减少使用过程中的精神努力。为了实现这一目标，我们开发了一个遥操作系统，通过模仿学习收集人类演示数据，以便训练假手控制模型。通过仅使用单个参与者几种物体的数据训练模型，我们已经证明模仿学习算法可以在不同个体和未见过的物体上实现高的成功率，并且具有不同的权重可实现泛化。演示数据可在 \href{this https URL}{this https URL} 获取。 

---
# Bayesian Inverse Physics for Neuro-Symbolic Robot Learning 

**Title (ZH)**: 基于贝叶斯逆物理的神经符号机器人学习 

**Authors**: Octavio Arriaga, Rebecca Adam, Melvin Laux, Lisa Gutzeit, Marco Ragni, Jan Peters, Frank Kirchner  

**Link**: [PDF](https://arxiv.org/pdf/2506.08756)  

**Abstract**: Real-world robotic applications, from autonomous exploration to assistive technologies, require adaptive, interpretable, and data-efficient learning paradigms. While deep learning architectures and foundation models have driven significant advances in diverse robotic applications, they remain limited in their ability to operate efficiently and reliably in unknown and dynamic environments. In this position paper, we critically assess these limitations and introduce a conceptual framework for combining data-driven learning with deliberate, structured reasoning. Specifically, we propose leveraging differentiable physics for efficient world modeling, Bayesian inference for uncertainty-aware decision-making, and meta-learning for rapid adaptation to new tasks. By embedding physical symbolic reasoning within neural models, robots could generalize beyond their training data, reason about novel situations, and continuously expand their knowledge. We argue that such hybrid neuro-symbolic architectures are essential for the next generation of autonomous systems, and to this end, we provide a research roadmap to guide and accelerate their development. 

**Abstract (ZH)**: 现实世界中的机器人应用，从自主探索到辅助技术，需要具备适应性、可解释性和数据高效性的学习范式。尽管深度学习架构和基础模型在多样化的机器人应用中取得了显著进展，但在未知和动态环境中高效可靠地运行仍存在局限性。在本文中，我们批判性地评估了这些局限性，并提出了将数据驱动学习与有意识的结构化推理相结合的概念框架。具体来说，我们建议利用可微物理模型进行高效的世界建模、利用贝叶斯推断进行不确定性感知决策，并利用元学习实现对新任务的快速适应。通过将物理符号推理嵌入神经模型中，机器人可以超越其训练数据进行泛化、推理关于新颖情况，并持续扩展知识。我们认为，这样的混合神经符号架构对于新一代自主系统是必不可少的，为此，我们提供了一条研究路线图以指导和加速其发展。 

---
# PhyBlock: A Progressive Benchmark for Physical Understanding and Planning via 3D Block Assembly 

**Title (ZH)**: PhyBlock：通过3D拼块组装的渐进式物理理解和规划基准 

**Authors**: Liang Ma, Jiajun Wen, Min Lin, Rongtao Xu, Xiwen Liang, Bingqian Lin, Jun Ma, Yongxin Wang, Ziming Wei, Haokun Lin, Mingfei Han, Meng Cao, Bokui Chen, Ivan Laptev, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08708)  

**Abstract**: While vision-language models (VLMs) have demonstrated promising capabilities in reasoning and planning for embodied agents, their ability to comprehend physical phenomena, particularly within structured 3D environments, remains severely limited. To close this gap, we introduce PhyBlock, a progressive benchmark designed to assess VLMs on physical understanding and planning through robotic 3D block assembly tasks. PhyBlock integrates a novel four-level cognitive hierarchy assembly task alongside targeted Visual Question Answering (VQA) samples, collectively aimed at evaluating progressive spatial reasoning and fundamental physical comprehension, including object properties, spatial relationships, and holistic scene understanding. PhyBlock includes 2600 block tasks (400 assembly tasks, 2200 VQA tasks) and evaluates models across three key dimensions: partial completion, failure diagnosis, and planning robustness. We benchmark 21 state-of-the-art VLMs, highlighting their strengths and limitations in physically grounded, multi-step planning. Our empirical findings indicate that the performance of VLMs exhibits pronounced limitations in high-level planning and reasoning capabilities, leading to a notable decline in performance for the growing complexity of the tasks. Error analysis reveals persistent difficulties in spatial orientation and dependency reasoning. Surprisingly, chain-of-thought prompting offers minimal improvements, suggesting spatial tasks heavily rely on intuitive model comprehension. We position PhyBlock as a unified testbed to advance embodied reasoning, bridging vision-language understanding and real-world physical problem-solving. 

**Abstract (ZH)**: PhyBlock：一种用于评估视觉-语言模型物理理解与规划能力的渐进式基准 

---
# ROS-related Robotic Systems Development with V-model-based Application of MeROS Metamodel 

**Title (ZH)**: 基于MeROS元模型的V模型应用的ROS相关机器人系统开发 

**Authors**: Tomasz Winiarski, Jan Kaniuka, Daniel Giełdowski, Jakub Ostrysz, Krystian Radlak, Dmytro Kushnir  

**Link**: [PDF](https://arxiv.org/pdf/2506.08706)  

**Abstract**: As robotic systems grow increasingly complex, heterogeneous, and safety-critical, the need for structured development methodologies becomes paramount. Although frameworks like the Robot Operating System (ROS) and Model-Based Systems Engineering (MBSE) offer foundational tools, they often lack integration when used together. This paper addresses that gap by aligning the widely recognized V-model development paradigm with the MeROS metamodel SysML-based modeling language tailored for ROS-based systems.
We propose a domain-specific methodology that bridges ROS-centric modelling with systems engineering practices. Our approach formalises the structure, behaviour, and validation processes of robotic systems using MeROS, while extending it with a generalized, adaptable V-model compatible with both ROS and ROS 2. Rather than prescribing a fixed procedure, the approach supports project-specific flexibility and reuse, offering guidance across all stages of development.
The approach is validated through a comprehensive case study on HeROS, a heterogeneous multi-robot platform comprising manipulators, mobile units, and dynamic test environments. This example illustrates how the MeROS-compatible V-model enhances traceability and system consistency while remaining accessible and extensible for future adaptation. The work contributes a structured, tool-agnostic foundation for developers and researchers seeking to apply MBSE practices in ROS-based projects. 

**Abstract (ZH)**: 随着机器人系统日益复杂、异构化和安全性关键性提高，结构化的开发方法学变得至关重要。虽然像Robot Operating System (ROS)和基于模型的系统工程（MBSE）这样的框架提供了基础工具，但它们在结合使用时往往缺乏集成。本文通过将广泛认可的V模型开发范式与基于SysML的MeROS元模型对齐，解决了这一问题，MeROS专门针对基于ROS的系统进行建模。我们提出了一种领域特定的方法，将以ROS为中心的建模与系统工程实践相结合。该方法使用MeROS形式化机器人系统的结构、行为和验证过程，并通过扩展，使其兼容ROS和ROS 2的通用和可扩展V模型。这种方法不规定固定流程，而是支持项目特定的灵活性和重用，在整个开发阶段提供指导。该方法通过一个全面的案例研究HeROS进行验证，HeROS是一个异构多机器人平台，包含操作臂、移动单元和动态测试环境。这一示例展示了MeROS兼容的V模型如何增强可追溯性和系统一致性，同时保持可访问性和未来扩展性。本研究为寻求在基于ROS的项目中应用MBSE实践的开发者和研究者提供了一个结构化、工具无关的基础。 

---
# Deep Reinforcement Learning-Based Motion Planning and PDE Control for Flexible Manipulators 

**Title (ZH)**: 基于深度强化学习的柔性 manipulator 运动规划与偏微分方程控制 

**Authors**: Amir Hossein Barjini, Seyed Adel Alizadeh Kolagar, Sadeq Yaqubi, Jouni Mattila  

**Link**: [PDF](https://arxiv.org/pdf/2506.08639)  

**Abstract**: This article presents a motion planning and control framework for flexible robotic manipulators, integrating deep reinforcement learning (DRL) with a nonlinear partial differential equation (PDE) controller. Unlike conventional approaches that focus solely on control, we demonstrate that the desired trajectory significantly influences endpoint vibrations. To address this, a DRL motion planner, trained using the soft actor-critic (SAC) algorithm, generates optimized trajectories that inherently minimize vibrations. The PDE nonlinear controller then computes the required torques to track the planned trajectory while ensuring closed-loop stability using Lyapunov analysis. The proposed methodology is validated through both simulations and real-world experiments, demonstrating superior vibration suppression and tracking accuracy compared to traditional methods. The results underscore the potential of combining learning-based motion planning with model-based control for enhancing the precision and stability of flexible robotic manipulators. 

**Abstract (ZH)**: 一种将深度强化学习与非线性偏微分方程控制器集成的柔性机器人 manipulator 运动规划与控制框架 

---
# Noise Analysis and Hierarchical Adaptive Body State Estimator For Biped Robot Walking With ESVC Foot 

**Title (ZH)**: 噪声分析及基于ESVC足的 biped机器人步行分层自适应身体状态估计器 

**Authors**: Boyang Chen, Xizhe Zang, Chao Song, Yue Zhang, Xuehe Zhang, Jie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.08578)  

**Abstract**: The ESVC(Ellipse-based Segmental Varying Curvature) foot, a robot foot design inspired by the rollover shape of the human foot, significantly enhances the energy efficiency of the robot walking gait. However, due to the tilt of the supporting leg, the error of the contact model are amplified, making robot state estimation more challenging. Therefore, this paper focuses on the noise analysis and state estimation for robot walking with the ESVC foot. First, through physical robot experiments, we investigate the effect of the ESVC foot on robot measurement noise and process noise. and a noise-time regression model using sliding window strategy is developed. Then, a hierarchical adaptive state estimator for biped robots with the ESVC foot is proposed. The state estimator consists of two stages: pre-estimation and post-estimation. In the pre-estimation stage, a data fusion-based estimation is employed to process the sensory data. During post-estimation, the acceleration of center of mass is first estimated, and then the noise covariance matrices are adjusted based on the regression model. Following that, an EKF(Extended Kalman Filter) based approach is applied to estimate the centroid state during robot walking. Physical experiments demonstrate that the proposed adaptive state estimator for biped robot walking with the ESVC foot not only provides higher precision than both EKF and Adaptive EKF, but also converges faster under varying noise conditions. 

**Abstract (ZH)**: 基于椭圆段变曲率(EVSC)脚的两足机器人姿态估计与噪声分析研究 

---
# Diffusion Models for Safety Validation of Autonomous Driving Systems 

**Title (ZH)**: 自主驾驶系统安全性验证的扩散模型 

**Authors**: Juanran Wang, Marc R. Schlichting, Harrison Delecki, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2506.08459)  

**Abstract**: Safety validation of autonomous driving systems is extremely challenging due to the high risks and costs of real-world testing as well as the rarity and diversity of potential failures. To address these challenges, we train a denoising diffusion model to generate potential failure cases of an autonomous vehicle given any initial traffic state. Experiments on a four-way intersection problem show that in a variety of scenarios, the diffusion model can generate realistic failure samples while capturing a wide variety of potential failures. Our model does not require any external training dataset, can perform training and inference with modest computing resources, and does not assume any prior knowledge of the system under test, with applicability to safety validation for traffic intersections. 

**Abstract (ZH)**: 自主驾驶系统安全验证由于现实世界测试的风险和成本高以及潜在故障的稀有性和多样性而极具挑战性。为应对这些挑战，我们训练一个去噪扩散模型，给定任意初始交通状态，生成该自主车辆的潜在故障案例。在四向交叉口问题上的实验表明，在多种场景下，扩散模型能够生成现实的故障样本，同时捕获广泛的潜在故障。我们的模型不需要任何外部训练数据集，可以使用有限的计算资源进行训练和推理，并且不需要测试系统的先验知识，适用于交通交叉口的安全验证。 

---
# TGRPO :Fine-tuning Vision-Language-Action Model via Trajectory-wise Group Relative Policy Optimization 

**Title (ZH)**: TGRPO：基于轨迹-wise 组相对策略优化的视觉-语言-动作模型微调 

**Authors**: Zengjue Chen, Runliang Niu, He Kong, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08440)  

**Abstract**: Recent advances in Vision-Language-Action (VLA) model have demonstrated strong generalization capabilities across diverse scenes, tasks, and robotic platforms when pretrained at large-scale datasets. However, these models still require task-specific fine-tuning in novel environments, a process that relies almost exclusively on supervised fine-tuning (SFT) using static trajectory datasets. Such approaches neither allow robot to interact with environment nor do they leverage feedback from live execution. Also, their success is critically dependent on the size and quality of the collected trajectories. Reinforcement learning (RL) offers a promising alternative by enabling closed-loop interaction and aligning learned policies directly with task objectives. In this work, we draw inspiration from the ideas of GRPO and propose the Trajectory-wise Group Relative Policy Optimization (TGRPO) method. By fusing step-level and trajectory-level advantage signals, this method improves GRPO's group-level advantage estimation, thereby making the algorithm more suitable for online reinforcement learning training of VLA. Experimental results on ten manipulation tasks from the libero-object benchmark demonstrate that TGRPO consistently outperforms various baseline methods, capable of generating more robust and efficient policies across multiple tested scenarios. Our source codes are available at: this https URL 

**Abstract (ZH)**: Recent Advances in Vision-Language-Action (VLA) Model: Trajectory-wise Group Relative Policy Optimization for Robotic Manipulation Tasks 

---
# Attention-based Learning for 3D Informative Path Planning 

**Title (ZH)**: 基于注意力的学习在3D信息性路径规划中的应用 

**Authors**: Rui Zhao, Xingjian Zhang, Yuhong Cao, Yizhuo Wang, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2506.08434)  

**Abstract**: In this work, we propose an attention-based deep reinforcement learning approach to address the adaptive informative path planning (IPP) problem in 3D space, where an aerial robot equipped with a downward-facing sensor must dynamically adjust its 3D position to balance sensing footprint and accuracy, and finally obtain a high-quality belief of an underlying field of interest over a given domain (e.g., presence of specific plants, hazardous gas, geological structures, etc.). In adaptive IPP tasks, the agent is tasked with maximizing information collected under time/distance constraints, continuously adapting its path based on newly acquired sensor data. To this end, we leverage attention mechanisms for their strong ability to capture global spatial dependencies across large action spaces, allowing the agent to learn an implicit estimation of environmental transitions. Our model builds a contextual belief representation over the entire domain, guiding sequential movement decisions that optimize both short- and long-term search objectives. Comparative evaluations against state-of-the-art planners demonstrate that our approach significantly reduces environmental uncertainty within constrained budgets, thus allowing the agent to effectively balance exploration and exploitation. We further show our model generalizes well to environments of varying sizes, highlighting its potential for many real-world applications. 

**Abstract (ZH)**: 基于注意力机制的深度强化学习在三维空间自适应信息路径规划问题中的应用 

---
# Periodic Bipedal Gait Learning Using Reward Composition Based on a Novel Gait Planner for Humanoid Robots 

**Title (ZH)**: 基于新型步态规划器的基于奖励组成的人形机器人双足周期步态学习 

**Authors**: Bolin Li, Linwei Sun, Xuecong Huang, Yuzhi Jiang, Lijun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08416)  

**Abstract**: This paper presents a periodic bipedal gait learning method using reward composition, integrated with a real-time gait planner for humanoid robots. First, we introduce a novel gait planner that incorporates dynamics to design the desired joint trajectory. In the gait design process, the 3D robot model is decoupled into two 2D models, which are then approximated as hybrid inverted pendulums (H-LIP) for trajectory planning. The gait planner operates in parallel in real time within the robot's learning environment. Second, based on this gait planner, we design three effective reward functions within a reinforcement learning framework, forming a reward composition to achieve periodic bipedal gait. This reward composition reduces the robot's learning time and enhances locomotion performance. Finally, a gait design example and performance comparison are presented to demonstrate the effectiveness of the proposed method. 

**Abstract (ZH)**: 本文提出了一种基于奖励组成的学习周期性双足步态方法，并结合实时步态规划器应用于类人机器人。首先，我们介绍了一种新的步态规划器，该规划器包含动力学设计来规划期望的关节轨迹。在步态设计过程中，3D 机器人模型被分解为两个2D模型，然后近似为混合倒摆（H-LIP）以进行轨迹规划。该步态规划器在机器人学习环境中实时并行运行。其次，基于此步态规划器，我们在强化学习框架内设计了三种有效的奖励函数，组成奖励函数以实现周期性双足步态。这种奖励函数组成减少了机器人学习时间并提升了运动性能。最后，给出了一个步态设计实例和性能对比，以展示所提方法的有效性。 

---
# Re4MPC: Reactive Nonlinear MPC for Multi-model Motion Planning via Deep Reinforcement Learning 

**Title (ZH)**: Re4MPC: 基于深度强化学习的多模型反应非线性模型预测控制运动规划 

**Authors**: Neşet Ünver Akmandor, Sarvesh Prajapati, Mark Zolotas, Taşkın Padır  

**Link**: [PDF](https://arxiv.org/pdf/2506.08344)  

**Abstract**: Traditional motion planning methods for robots with many degrees-of-freedom, such as mobile manipulators, are often computationally prohibitive for real-world settings. In this paper, we propose a novel multi-model motion planning pipeline, termed Re4MPC, which computes trajectories using Nonlinear Model Predictive Control (NMPC). Re4MPC generates trajectories in a computationally efficient manner by reactively selecting the model, cost, and constraints of the NMPC problem depending on the complexity of the task and robot state. The policy for this reactive decision-making is learned via a Deep Reinforcement Learning (DRL) framework. We introduce a mathematical formulation to integrate NMPC into this DRL framework. To validate our methodology and design choices, we evaluate DRL training and test outcomes in a physics-based simulation involving a mobile manipulator. Experimental results demonstrate that Re4MPC is more computationally efficient and achieves higher success rates in reaching end-effector goals than the NMPC baseline, which computes whole-body trajectories without our learning mechanism. 

**Abstract (ZH)**: 一种基于深度强化学习的高效多模型运动规划方法 

---
# HiBerNAC: Hierarchical Brain-emulated Robotic Neural Agent Collective for Disentangling Complex Manipulation 

**Title (ZH)**: HiBerNAC：分层次类脑机器人神经代理集体用于解缠复杂操作 

**Authors**: Hongjun Wu, Heng Zhang, Pengsong Zhang, Jin Wang, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08296)  

**Abstract**: Recent advances in multimodal vision-language-action (VLA) models have revolutionized traditional robot learning, enabling systems to interpret vision, language, and action in unified frameworks for complex task planning. However, mastering complex manipulation tasks remains an open challenge, constrained by limitations in persistent contextual memory, multi-agent coordination under uncertainty, and dynamic long-horizon planning across variable sequences. To address this challenge, we propose \textbf{HiBerNAC}, a \textbf{Hi}erarchical \textbf{B}rain-\textbf{e}mulated \textbf{r}obotic \textbf{N}eural \textbf{A}gent \textbf{C}ollective, inspired by breakthroughs in neuroscience, particularly in neural circuit mechanisms and hierarchical decision-making. Our framework combines: (1) multimodal VLA planning and reasoning with (2) neuro-inspired reflection and multi-agent mechanisms, specifically designed for complex robotic manipulation tasks. By leveraging neuro-inspired functional modules with decentralized multi-agent collaboration, our approach enables robust and enhanced real-time execution of complex manipulation tasks. In addition, the agentic system exhibits scalable collective intelligence via dynamic agent specialization, adapting its coordination strategy to variable task horizons and complexity. Through extensive experiments on complex manipulation tasks compared with state-of-the-art VLA models, we demonstrate that \textbf{HiBerNAC} reduces average long-horizon task completion time by 23\%, and achieves non-zero success rates (12\textendash 31\%) on multi-path tasks where prior state-of-the-art VLA models consistently fail. These results provide indicative evidence for bridging biological cognition and robotic learning mechanisms. 

**Abstract (ZH)**: Recent Advances in Multimodal Vision-Language-Action (VLA) Models Have Revolutionized Traditional Robot Learning, Enabling Systems to Interpret Vision, Language, and Action in Unified Frameworks for Complex Task Planning. However, Mastering Complex Manipulation Tasks Remains an Open Challenge, Constrained by Limitations in Persistent Contextual Memory, Multi-Agent Coordination Under Uncertainty, and Dynamic Long-Horizon Planning Across Variable Sequences. To Address This Challenge, We Propose HiBerNAC, a Hierarchical Brain-Emulated Robotic Neural Agent Collective, Inspired by Breakthroughs in Neuroscience, Particularly in Neural Circuit Mechanisms and Hierarchical Decision-Making. 

---
# TensorTouch: Calibration of Tactile Sensors for High Resolution Stress Tensor and Deformation for Dexterous Manipulation 

**Title (ZH)**: TensorTouch: 用于 Dexterous 操作的高分辨率应力张量和变形校准的触觉传感器标定 

**Authors**: Won Kyung Do, Matthew Strong, Aiden Swann, Boshu Lei, Monroe Kennedy III  

**Link**: [PDF](https://arxiv.org/pdf/2506.08291)  

**Abstract**: Advanced dexterous manipulation involving multiple simultaneous contacts across different surfaces, like pinching coins from ground or manipulating intertwined objects, remains challenging for robotic systems. Such tasks exceed the capabilities of vision and proprioception alone, requiring high-resolution tactile sensing with calibrated physical metrics. Raw optical tactile sensor images, while information-rich, lack interpretability and cross-sensor transferability, limiting their real-world utility. TensorTouch addresses this challenge by integrating finite element analysis with deep learning to extract comprehensive contact information from optical tactile sensors, including stress tensors, deformation fields, and force distributions at pixel-level resolution. The TensorTouch framework achieves sub-millimeter position accuracy and precise force estimation while supporting large sensor deformations crucial for manipulating soft objects. Experimental validation demonstrates 90% success in selectively grasping one of two strings based on detected motion, enabling new contact-rich manipulation capabilities previously inaccessible to robotic systems. 

**Abstract (ZH)**: 涉及多种同时接触不同表面的高级灵巧操作，如从地面夹取硬币或操作交织物体，仍是对机器人系统的一项挑战。此类任务超出了仅依赖视觉和本体感觉的能力范围，需要具备校准物理指标的高分辨率触觉感知。尽管光学触觉传感器图像信息丰富，但缺乏可解释性和跨传感器转移性，限制了其在实际中的应用。TensorTouch通过将有限元分析与深度学习结合，从光学触觉传感器中提取全面的接触信息，包括像素级分辨率的应力张量、变形场和力分布。TensorTouch框架实现了亚毫米级别的位置精度和精确的力估计，同时支持对于操纵软物体至关重要的大传感器变形。实验验证显示，在基于检测到的运动选择性抓取两条线之一方面达到了90%的成功率，为机器人系统提供了新的丰富的接触操作能力，这些能力以前是不可及的。 

---
# Ego-centric Learning of Communicative World Models for Autonomous Driving 

**Title (ZH)**: 基于ego-centric学习的沟通世界模型的自主驾驶 

**Authors**: Hang Wang, Dechen Gao, Junshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08149)  

**Abstract**: We study multi-agent reinforcement learning (MARL) for tasks in complex high-dimensional environments, such as autonomous driving. MARL is known to suffer from the \textit{partial observability} and \textit{non-stationarity} issues. To tackle these challenges, information sharing is often employed, which however faces major hurdles in practice, including overwhelming communication overhead and scalability concerns. By making use of generative AI embodied in world model together with its latent representation, we develop {\it CALL}, \underline{C}ommunic\underline{a}tive Wor\underline{l}d Mode\underline{l}, for MARL, where 1) each agent first learns its world model that encodes its state and intention into low-dimensional latent representation with smaller memory footprint, which can be shared with other agents of interest via lightweight communication; and 2) each agent carries out ego-centric learning while exploiting lightweight information sharing to enrich her world model, and then exploits its generalization capacity to improve prediction for better planning. We characterize the gain on the prediction accuracy from the information sharing and its impact on performance gap. Extensive experiments are carried out on the challenging local trajectory planning tasks in the CARLA platform to demonstrate the performance gains of using \textit{CALL}. 

**Abstract (ZH)**: 我们研究复杂高维度环境（如自主驾驶任务）下的多智能体强化学习（MARL），并针对其部分可观测性和非平稳性问题进行了研究。通过利用包含其潜在表示的世界模型生成AI，我们开发了CALL（CommUNative World Model），其中每个智能体首先学习其世界模型，将状态和意图编码为低维度的潜在表示，占用更少的内存，并可通过轻量级通信与感兴趣的其他智能体共享；每个智能体在利用轻量级信息共享丰富其世界模型的同时进行以自我为中心的学习，进而利用其泛化能力提高预测精度以更好地进行规划。我们量化了信息共享带来的预测准确性提升及其对性能差距的影响。我们在CARLA平台上进行了广泛的实验，以展示使用CALL所取得的性能增益。 

---
# Adaptive Per-Tree Canopy Volume Estimation Using Mobile LiDAR in Structured and Unstructured Orchards 

**Title (ZH)**: 基于移动LiDAR的结构化与非结构化果园树木冠体积自适应估测 

**Authors**: Ali Abedi, Fernando Cladera, Mohsen Farajijalal, Reza Ehsani  

**Link**: [PDF](https://arxiv.org/pdf/2506.08061)  

**Abstract**: We present a real-time system for per-tree canopy volume estimation using mobile LiDAR data collected during routine robotic navigation. Unlike prior approaches that rely on static scans or assume uniform orchard structures, our method adapts to varying field geometries via an integrated pipeline of LiDAR-inertial odometry, adaptive segmentation, and geometric reconstruction. We evaluate the system across two commercial orchards, one pistachio orchard with regular spacing and one almond orchard with dense, overlapping crowns. A hybrid clustering strategy combining DBSCAN and spectral clustering enables robust per-tree segmentation, achieving 93% success in pistachio and 80% in almond, with strong agreement to drone derived canopy volume estimates. This work advances scalable, non-intrusive tree monitoring for structurally diverse orchard environments. 

**Abstract (ZH)**: 基于移动LiDAR数据的实时单棵树冠体积估计算法：适应性分割与几何重建在商用果园中的应用 

---
# UAVs Meet Agentic AI: A Multidomain Survey of Autonomous Aerial Intelligence and Agentic UAVs 

**Title (ZH)**: UAVs遇上了自主性AI：自主 aerial 智能与自主性无人机的多领域综述 

**Authors**: Ranjan Sapkota, Konstantinos I. Roumeliotis, Manoj Karkee  

**Link**: [PDF](https://arxiv.org/pdf/2506.08045)  

**Abstract**: Agentic UAVs represent a new frontier in autonomous aerial intelligence, integrating perception, decision-making, memory, and collaborative planning to operate adaptively in complex, real-world environments. Driven by recent advances in Agentic AI, these systems surpass traditional UAVs by exhibiting goal-driven behavior, contextual reasoning, and interactive autonomy. We provide a comprehensive foundation for understanding the architectural components and enabling technologies that distinguish Agentic UAVs from traditional autonomous UAVs. Furthermore, a detailed comparative analysis highlights advancements in autonomy with AI agents, learning, and mission flexibility. This study explores seven high-impact application domains precision agriculture, construction & mining, disaster response, environmental monitoring, infrastructure inspection, logistics, security, and wildlife conservation, illustrating the broad societal value of agentic aerial intelligence. Furthermore, we identify key challenges in technical constraints, regulatory limitations, and data-model reliability, and we present emerging solutions across hardware innovation, learning architectures, and human-AI interaction. Finally, a future roadmap is proposed, outlining pathways toward self-evolving aerial ecosystems, system-level collaboration, and sustainable, equitable deployments. This survey establishes a foundational framework for the future development, deployment, and governance of agentic aerial systems (Agentic UAVs) across diverse societal and industrial domains. 

**Abstract (ZH)**: 自主代理无人机代表了自主空中智能的新前沿，结合了感知、决策、记忆和协同规划能力，以适应性方式在复杂的真实环境中运行。受近期自主代理人工智能进展的推动，这些系统通过表现出目标导向的行为、上下文推理和交互自主超越了传统无人机。我们提供了理解自主代理无人机与传统自主无人机区别的架构组件和使能技术的全面基础。此外，详细的比较分析突出了人工智能代理、学习和任务灵活性方面的自主性进步。本研究探讨了精确农业、建筑与采矿、救灾、环境监测、基础设施检查、物流、安全和野生动物保护等七个高影响应用领域，展示了自主代理空中智能的广泛社会价值。同时，我们识别出技术约束、监管限制和数据模型可靠性方面的关键挑战，并提出了涵盖硬件创新、学习架构和人类-人工智能交互的新兴解决方案。最后，提出了一个未来的路线图，概述了自主演化空中生态系统的途径、系统级协作，并实现了可持续和公平的应用部署。本综述为自主代理空中系统（自主代理无人机）在未来在多样化社会和工业领域的发展、部署和治理奠定了基础框架。 

---
# AI Magnetic Levitation (Maglev) Conveyor for Automated Assembly Production 

**Title (ZH)**: 基于AI磁悬浮（Maglev）输送机的自动化装配生产系统 

**Authors**: Ray Wai Man Kong  

**Link**: [PDF](https://arxiv.org/pdf/2506.08039)  

**Abstract**: Efficiency, speed, and precision are essential in modern manufacturing. AI Maglev Conveyor system, combining magnetic levitation (maglev) technology with artificial intelligence (AI), revolutionizes automated production processes. This system reduces maintenance costs and downtime by eliminating friction, enhancing operational efficiency. It transports goods swiftly with minimal energy consumption, optimizing resource use and supporting sustainability. AI integration enables real-time monitoring and adaptive control, allowing businesses to respond to production demand fluctuations and streamline supply chain operations.
The AI Maglev Conveyor offers smooth, silent operation, accommodating diverse product types and sizes for flexible manufacturing without extensive reconfiguration. AI algorithms optimize routing, reduce cycle times, and improve throughput, creating an agile production line adaptable to market changes.
This applied research paper introduces the Maglev Conveyor system, featuring an electromagnetic controller and multiple movers to enhance automation. It offers cost savings as an alternative to setups using six-axis robots or linear motors, with precise adjustments for robotic arm loading. Operating at high speeds minimizes treatment time for delicate components while maintaining precision. Its adaptable design accommodates various materials, facilitating integration of processing stations alongside electronic product assembly. Positioned between linear-axis and robotic systems in cost, the Maglev Conveyor is ideal for flat parts requiring minimal travel, transforming production efficiency across industries. It explores its technical advantages, flexibility, cost reductions, and overall benefits. 

**Abstract (ZH)**: AI磁浮传送系统：提高效率、速度和精度的自动化生产革命 

---
# VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning 

**Title (ZH)**: VIKI-R：通过强化学习协调具身多智能体合作 

**Authors**: Li Kang, Xiufeng Song, Heng Zhou, Yiran Qin, Jie Yang, Xiaohong Liu, Philip Torr, Lei Bai, Zhenfei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09049)  

**Abstract**: Coordinating multiple embodied agents in dynamic environments remains a core challenge in artificial intelligence, requiring both perception-driven reasoning and scalable cooperation strategies. While recent works have leveraged large language models (LLMs) for multi-agent planning, a few have begun to explore vision-language models (VLMs) for visual reasoning. However, these VLM-based approaches remain limited in their support for diverse embodiment types. In this work, we introduce VIKI-Bench, the first hierarchical benchmark tailored for embodied multi-agent cooperation, featuring three structured levels: agent activation, task planning, and trajectory perception. VIKI-Bench includes diverse robot embodiments, multi-view visual observations, and structured supervision signals to evaluate reasoning grounded in visual inputs. To demonstrate the utility of VIKI-Bench, we propose VIKI-R, a two-stage framework that fine-tunes a pretrained vision-language model (VLM) using Chain-of-Thought annotated demonstrations, followed by reinforcement learning under multi-level reward signals. Our extensive experiments show that VIKI-R significantly outperforms baselines method across all task levels. Furthermore, we show that reinforcement learning enables the emergence of compositional cooperation patterns among heterogeneous agents. Together, VIKI-Bench and VIKI-R offer a unified testbed and method for advancing multi-agent, visual-driven cooperation in embodied AI systems. 

**Abstract (ZH)**: 协调多个具身代理在动态环境中的合作仍然是人工智能的核心挑战，需要感知驱动的推理和可扩展的合作策略。虽然近期工作利用大型语言模型（LLMs）进行多智能体规划，但有少数研究开始探索视觉语言模型（VLMs）的视觉推理能力。然而，这些基于VLM的方法在支持多种具身类型方面仍然有限。在本工作中，我们引入了VIKI-Bench，这是首个针对具身多智能体合作的层级基准，包含三个结构化的层次：代理激活、任务规划和轨迹感知。VIKI-Bench 包含多样化的机器人具身类型、多视角视觉观测以及结构化的监督信号，以评估基于视觉输入的推理能力。为展示 VIKI-Bench 的实用性，我们提出了一种两阶段框架 VIKI-R，该框架利用具有因果链标注的演示样例对预训练的视觉语言模型（VLM）进行微调，随后在多层次奖励信号下进行强化学习。我们的广泛实验表明，VIKI-R 在所有任务层次上显著优于基线方法。此外，我们展示了强化学习促使异构智能体之间产生组合合作模式。总的来说，VIKI-Bench 和 VIKI-R 提供了一个统一的测试平台和方法，以促进具身AI系统的多智能体、视觉驱动的合作。 

---
# SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction 

**Title (ZH)**: SDTagNet: 利用文本标注的导航地图进行在线高精度地图构建 

**Authors**: Fabian Immel, Jan-Hendrik Pauls, Richard Fehler, Frank Bieder, Jonas Merkert, Christoph Stiller  

**Link**: [PDF](https://arxiv.org/pdf/2506.08997)  

**Abstract**: Autonomous vehicles rely on detailed and accurate environmental information to operate safely. High definition (HD) maps offer a promising solution, but their high maintenance cost poses a significant barrier to scalable deployment. This challenge is addressed by online HD map construction methods, which generate local HD maps from live sensor data. However, these methods are inherently limited by the short perception range of onboard sensors. To overcome this limitation and improve general performance, recent approaches have explored the use of standard definition (SD) maps as prior, which are significantly easier to maintain. We propose SDTagNet, the first online HD map construction method that fully utilizes the information of widely available SD maps, like OpenStreetMap, to enhance far range detection accuracy. Our approach introduces two key innovations. First, in contrast to previous work, we incorporate not only polyline SD map data with manually selected classes, but additional semantic information in the form of textual annotations. In this way, we enrich SD vector map tokens with NLP-derived features, eliminating the dependency on predefined specifications or exhaustive class taxonomies. Second, we introduce a point-level SD map encoder together with orthogonal element identifiers to uniformly integrate all types of map elements. Experiments on Argoverse 2 and nuScenes show that this boosts map perception performance by up to +5.9 mAP (+45%) w.r.t. map construction without priors and up to +3.2 mAP (+20%) w.r.t. previous approaches that already use SD map priors. Code is available at this https URL 

**Abstract (ZH)**: 自主驾驶车辆依赖于详细的accurate环境信息以确保安全运行。高精度(HD)地图提供了有希望的解决方案，但其高昂的维护成本成为广泛应用的瓶颈。通过在线HD地图构建方法，可以利用实时传感器数据生成局部HD地图，但这些方法受限于车载传感器较短的感知范围。为解决这一限制并提高整体性能，最近的研究探索了标准定义(SD)地图作为先验的可能性，SD地图易于维护得多。我们提出了SDTagNet，这是首个充分利用广泛可用的SD地图信息（例如OpenStreetMap）以增强远距离检测准确性的在线HD地图构建方法。我们的方法引入了两项关键创新。首先，与先前的工作不同，我们不仅利用带有人工选择类别的多段线SD地图数据，还附加了文本注释形式的语义信息，从而通过NLP提取特征丰富SD矢量地图标记，消除了对预定义规范或详尽类别分类的需求。其次，我们引入了一种点级别SD地图编码器，并结合正交元素标识符，以统一整合各类地图元素。实验结果表明，与不使用先验的HD地图构建方法相比，该方法的地图感知性能提高了最多5.9 mAP (+45%)；与已经使用SD地图先验的先前方法相比，提高了最多3.2 mAP (+20%)。代码可在此处访问。 

---
# Rethinking Range-View LiDAR Segmentation in Adverse Weather 

**Title (ZH)**: 重新思考不良天气条件下的范围视图LiDAR分割 

**Authors**: Longyu Yang, Ping Hu, Lu Zhang, Jun Liu, Yap-Peng Tan, Heng Tao Shen, Xiaofeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08979)  

**Abstract**: LiDAR segmentation has emerged as an important task to enrich multimedia experiences and analysis. Range-view-based methods have gained popularity due to their high computational efficiency and compatibility with real-time deployment. However, their generalized performance under adverse weather conditions remains underexplored, limiting their reliability in real-world environments. In this work, we identify and analyze the unique challenges that affect the generalization of range-view LiDAR segmentation in severe weather. To address these challenges, we propose a modular and lightweight framework that enhances robustness without altering the core architecture of existing models. Our method reformulates the initial stem block of standard range-view networks into two branches to process geometric attributes and reflectance intensity separately. Specifically, a Geometric Abnormality Suppression (GAS) module reduces the influence of weather-induced spatial noise, and a Reflectance Distortion Calibration (RDC) module corrects reflectance distortions through memory-guided adaptive instance normalization. The processed features are then fused and passed to the original segmentation pipeline. Extensive experiments on different benchmarks and baseline models demonstrate that our approach significantly improves generalization to adverse weather with minimal inference overhead, offering a practical and effective solution for real-world LiDAR segmentation. 

**Abstract (ZH)**: 基于范围视图的LiDAR分割在恶劣天气条件下的通用性分析与提升方法 

---
# Help or Hindrance: Understanding the Impact of Robot Communication in Action Teams 

**Title (ZH)**: 助益还是障碍：理解机器人沟通在行动团队中的影响 

**Authors**: Tauhid Tanjim, Jonathan St. George, Kevin Ching, Hee Rin Lee, Angelique Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2506.08892)  

**Abstract**: The human-robot interaction (HRI) field has recognized the importance of enabling robots to interact with teams. Human teams rely on effective communication for successful collaboration in time-sensitive environments. Robots can play a role in enhancing team coordination through real-time assistance. Despite significant progress in human-robot teaming research, there remains an essential gap in how robots can effectively communicate with action teams using multimodal interaction cues in time-sensitive environments. This study addresses this knowledge gap in an experimental in-lab study to investigate how multimodal robot communication in action teams affects workload and human perception of robots. We explore team collaboration in a medical training scenario where a robotic crash cart (RCC) provides verbal and non-verbal cues to help users remember to perform iterative tasks and search for supplies. Our findings show that verbal cues for object search tasks and visual cues for task reminders reduce team workload and increase perceived ease of use and perceived usefulness more effectively than a robot with no feedback. Our work contributes to multimodal interaction research in the HRI field, highlighting the need for more human-robot teaming research to understand best practices for integrating collaborative robots in time-sensitive environments such as in hospitals, search and rescue, and manufacturing applications. 

**Abstract (ZH)**: 人机交互（HRI）领域已认识到使机器人能够与团队互动的重要性。人类团队依赖有效的沟通在时间敏感环境中实现成功的合作。机器人可以通过实时协助在团队协调中发挥作用。尽管在人机团队合作研究方面取得了显著进展，但在时间敏感环境中，如何通过多模态交互提示有效与行动团队沟通的问题仍是一个重要缺口。本研究通过实验室内研究解决这一知识缺口，探讨多模态机器人通信如何影响工作负荷和人类对机器人的感知。我们探讨了在医疗培训场景中，机器人急救车（RCC）通过口头和非口头提示帮助用户记得执行迭代任务和寻找物资的方式。研究发现，在物体搜索任务中使用口头提示，在任务提醒中使用视觉提示，比没有反馈的机器人更能有效减少团队工作负荷并提高对机器人的使用便捷性和有用性的感知。我们的研究为HRI领域的多模态交互研究做出了贡献，强调了在医院、搜索与救援和制造等时间敏感环境中更好地整合协作机器人所需的人机团队合作研究的重要性。 

---
# Confidence Boosts Trust-Based Resilience in Cooperative Multi-Robot Systems 

**Title (ZH)**: 自信增强基于信任的协作多机器人系统韧性 

**Authors**: Luca Ballotta, Áron Vékássy, Stephanie Gil, Michal Yemini  

**Link**: [PDF](https://arxiv.org/pdf/2506.08807)  

**Abstract**: Wireless communication-based multi-robot systems open the door to cyberattacks that can disrupt safety and performance of collaborative robots. The physical channel supporting inter-robot communication offers an attractive opportunity to decouple the detection of malicious robots from task-relevant data exchange between legitimate robots. Yet, trustworthiness indications coming from physical channels are uncertain and must be handled with this in mind. In this paper, we propose a resilient protocol for multi-robot operation wherein a parameter {\lambda}t accounts for how confident a robot is about the legitimacy of nearby robots that the physical channel indicates. Analytical results prove that our protocol achieves resilient coordination with arbitrarily many malicious robots under mild assumptions. Tuning {\lambda}t allows a designer to trade between near-optimal inter-robot coordination and quick task execution; see Fig. 1. This is a fundamental performance tradeoff and must be carefully evaluated based on the task at hand. The effectiveness of our approach is numerically verified with experiments involving platoons of autonomous cars where some vehicles are maliciously spoofed. 

**Abstract (ZH)**: 基于无线通信的多机器人系统打开了通往干扰协作机器人安全性和性能的网络攻击的大门。支撑机器人间通信的物理信道提供了从合法机器人之间的任务相关数据交换中分离出恶意机器人检测的机会。然而，来自物理信道的信任指示具有不确定性，必须予以考虑。在本文中，我们提出了一种鲁棒的多机器人操作协议，其中参数λt表示机器人对其根据物理信道指示的附近机器人的合法性有多大的信心。理论结果证明，在温和假设下，我们的协议可以实现对任意数量恶意机器人的鲁棒协调。调整λt允许设计者在接近最优的机器人间协调和快速任务执行之间进行权衡；参见图1。这是基本的性能权衡，必须根据具体任务仔细评估。通过涉及恶意欺骗部分自主车辆的车队进行实验，验证了我们方法的有效性。 

---
# Communicating Through Avatars in Industry 5.0: A Focus Group Study on Human-Robot Collaboration 

**Title (ZH)**: 在 Industry 4.0 的背景下通过 avatar 进行沟通：人类与机器人协作的焦点小组研究 

**Authors**: Stina Klein, Pooja Prajod, Katharina Weitz, Matteo Lavit Nicora, Dimitra Tsovaltzi, Elisabeth André  

**Link**: [PDF](https://arxiv.org/pdf/2506.08805)  

**Abstract**: The integration of collaborative robots (cobots) in industrial settings raises concerns about worker well-being, particularly due to reduced social interactions. Avatars - designed to facilitate worker interactions and engagement - are promising solutions to enhance the human-robot collaboration (HRC) experience. However, real-world perspectives on avatar-supported HRC remain unexplored. To address this gap, we conducted a focus group study with employees from a German manufacturing company that uses cobots. Before the discussion, participants engaged with a scripted, industry-like HRC demo in a lab setting. This qualitative approach provided valuable insights into the avatar's potential roles, improvements to its behavior, and practical considerations for deploying them in industrial workcells. Our findings also emphasize the importance of personalized communication and task assistance. Although our study's limitations restrict its generalizability, it serves as an initial step in recognizing the potential of adaptive, context-aware avatar interactions in real-world industrial environments. 

**Abstract (ZH)**: 协作机器人（cobots）在工业环境中的集成引发了关于工人福祉的担忧，尤其是由于减少了社会互动。设计用于促进工人互动和参与的虚拟角色（avatar）是增强人类-机器人协作（HRC）体验的有前景的解决方案。然而，关于支持 avatar 的 HRC 的实际观点尚未被探索。为填补这一空白，我们对一家使用协作机器人的德国制造公司员工进行了焦点小组研究。在讨论前，参与者在实验室环境中参与了一个针对工业场景的 HRC 演示。通过这种质性研究方法，我们获得了有关 avatar 的潜在角色、其行为改进以及在工业工作站部署时的实际考虑的宝贵见解。研究结果还强调了个性化沟通和支持任务的重要性。尽管我们的研究存在局限性，限制了其普遍性，但它为识别适应性强、上下文感知的 avatar 交互在真实工业环境中的潜力提供了一个初步步骤。 

---
# Efficient Learning of Vehicle Controller Parameters via Multi-Fidelity Bayesian Optimization: From Simulation to Experiment 

**Title (ZH)**: 通过多保真度贝叶斯优化高效学习车辆控制器参数：从仿真到实验 

**Authors**: Yongpeng Zhao, Maik Pfefferkorn, Maximilian Templer, Rolf Findeisen  

**Link**: [PDF](https://arxiv.org/pdf/2506.08719)  

**Abstract**: Parameter tuning for vehicle controllers remains a costly and time-intensive challenge in automotive development. Traditional approaches rely on extensive real-world testing, making the process inefficient. We propose a multi-fidelity Bayesian optimization approach that efficiently learns optimal controller parameters by leveraging both low-fidelity simulation data and a very limited number of real-world experiments. Our approach significantly reduces the need for manual tuning and expensive field testing while maintaining the standard two-stage development workflow used in industry. The core contribution is the integration of an auto-regressive multi-fidelity Gaussian process model into Bayesian optimization, enabling knowledge transfer between different fidelity levels without requiring additional low-fidelity evaluations during real-world testing. We validate our approach through both simulation studies and realworld experiments. The results demonstrate that our method achieves high-quality controller performance with only very few real-world experiments, highlighting its potential as a practical and scalable solution for intelligent vehicle control tuning in industrial applications. 

**Abstract (ZH)**: 车辆控制器参数调优仍然是汽车开发中的一个 costly 和时间密集型挑战。传统方法依赖于广泛的实地测试，使过程效率低下。我们提出了一种多保真度贝叶斯优化方法，通过利用低保真度仿真数据和非常有限数量的真实世界试验来高效地学习最优控制器参数。该方法显著减少了手动调优和昂贵的实地测试需求，同时保持了工业中惯用的标准两阶段开发流程。核心贡献是将自回归多保真度高斯过程模型集成到贝叶斯优化中，能够在真实世界测试过程中无需额外的低保真度评估，实现不同保真度级别之间的知识迁移。我们通过仿真研究和实际试验验证了该方法。结果表明，该方法仅通过极少的真实世界试验就能实现高质量的控制器性能，突显了其在智能车辆控制调优方面作为实用和可扩展解决方案的潜力。 

---
# Modular Recurrence in Contextual MDPs for Universal Morphology Control 

**Title (ZH)**: 上下文MDP中的模块化递归用于通用形态控制 

**Authors**: Laurens Engwegen, Daan Brinks, Wendelin Böhmer  

**Link**: [PDF](https://arxiv.org/pdf/2506.08630)  

**Abstract**: A universal controller for any robot morphology would greatly improve computational and data efficiency. By utilizing contextual information about the properties of individual robots and exploiting their modular structure in the architecture of deep reinforcement learning agents, steps have been made towards multi-robot control. Generalization to new, unseen robots, however, remains a challenge. In this paper we hypothesize that the relevant contextual information is partially observable, but that it can be inferred through interactions for better generalization to contexts that are not seen during training. To this extent, we implement a modular recurrent architecture and evaluate its generalization performance on a large set of MuJoCo robots. The results show a substantial improved performance on robots with unseen dynamics, kinematics, and topologies, in four different environments. 

**Abstract (ZH)**: 一种适用于任何机器人形态的通用控制器将大幅提高计算和数据效率。通过利用单个机器人属性的上下文信息并利用其模块化结构来构建深度强化学习代理的架构，已在多机器人控制方面取得进展。然而，将该方法应用于未见过的新机器人环境仍然是一个挑战。本文假设相关的上下文信息部分可观测，但可以通过交互来推断，以提高在训练中未见上下文场景中的泛化性能。为此，我们实现了一种模块化递归架构，并在大量MuJoCo机器人上评估其泛化性能。结果表明，该方法在具有未见过的动力学、运动学和拓扑结构的四类不同环境中，机器人的性能有了显著提升。 

---
# Teaching Physical Awareness to LLMs through Sounds 

**Title (ZH)**: 通过声音教学物理感知给大型语言模型 

**Authors**: Weiguo Wang, Andy Nie, Wenrui Zhou, Yi Kai, Chengchen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08524)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in text and multimodal processing, yet they fundamentally lack physical awareness--understanding of real-world physical phenomena. In this work, we present ACORN, a framework that teaches LLMs physical awareness through sound, focusing on fundamental physical phenomena like the Doppler effect, multipath effect, and spatial relationships. To overcome data scarcity, ACORN introduce a physics-based simulator combining real-world sound sources with controlled physical channels to generate diverse training data. Using this simulator, we build AQA-PHY, a comprehensive Audio Question-Answer dataset, and propose an audio encoder that processes both magnitude and phase information. By connecting our audio encoder to state-of-the-art LLMs, we demonstrate reasonable results in both simulated and real-world tasks, such as line-of-sight detection, Doppler effect estimation, and Direction-of-Arrival estimation, paving the way for enabling LLMs to understand physical world. 

**Abstract (ZH)**: 大语言模型（LLMs）在文本和多模态处理方面展现了卓越的能力，但从根本上缺乏物理意识——对现实世界物理现象的理解。本文我们提出了ACORN框架，通过声音教导LLMs物理意识，重点关注多普勒效应、多路径效应和空间关系等基本物理现象。为克服数据稀缺问题，ACORN引入了一个基于物理的模拟器，结合现实世界的声音源和可控物理信道生成多样化的训练数据。利用该模拟器，我们构建了AQA-PHY综合音频问答数据集，并提出了一种处理幅度和相位信息的音频编码器。将我们的音频编码器连接到最先进的LLMs，我们展示了在模拟和真实世界任务中（如视线检测、多普勒效应估计和到达方向估计）合理的性能，为使LLMs理解物理世界开辟了路径。 

---
# How to Provably Improve Return Conditioned Supervised Learning? 

**Title (ZH)**: 如何证明性提高基于回报条件的监督学习？ 

**Authors**: Zhishuai Liu, Yu Yang, Ruhan Wang, Pan Xu, Dongruo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.08463)  

**Abstract**: In sequential decision-making problems, Return-Conditioned Supervised Learning (RCSL) has gained increasing recognition for its simplicity and stability in modern decision-making tasks. Unlike traditional offline reinforcement learning (RL) algorithms, RCSL frames policy learning as a supervised learning problem by taking both the state and return as input. This approach eliminates the instability often associated with temporal difference (TD) learning in offline RL. However, RCSL has been criticized for lacking the stitching property, meaning its performance is inherently limited by the quality of the policy used to generate the offline dataset. To address this limitation, we propose a principled and simple framework called Reinforced RCSL. The key innovation of our framework is the introduction of a concept we call the in-distribution optimal return-to-go. This mechanism leverages our policy to identify the best achievable in-dataset future return based on the current state, avoiding the need for complex return augmentation techniques. Our theoretical analysis demonstrates that Reinforced RCSL can consistently outperform the standard RCSL approach. Empirical results further validate our claims, showing significant performance improvements across a range of benchmarks. 

**Abstract (ZH)**: 在顺序决策问题中，基于返回条件的监督学习（RCSL）因其在现代决策任务中的简洁性和稳定性而得到了越来越多人的认可。与传统的离线强化学习（RL）算法不同，RCSL将策略学习框定为一个监督学习问题，通过输入状态和返回值来进行。这种方法消除了离线RL中与基于时差的（TD）学习相关的不稳定性。然而，RCSL因其缺乏缝合性质而受到批评，这意味着其性能受限于用于生成离线数据集的策略质量。为了克服这一局限，我们提出了一种称为强化RCSL的原则性和简单框架。我们框架的关键创新在于引入了一个我们称之为同分布最优未来回报的概念。该机制利用我们的策略来基于当前状态识别数据集中可实现的最佳未来回报，从而避免了复杂回报增强技术的需要。我们的理论分析表明，强化RCSL可以一致地优于标准的RCSL方法。实验结果进一步验证了我们的观点，显示了在多种基准上的显著性能提升。 

---
# Hybrid Reasoning for Perception, Explanation, and Autonomous Action in Manufacturing 

**Title (ZH)**: 制造领域中感知、解释与自主行动的混合推理 

**Authors**: Christos Margadji, Sebastian W. Pattinson  

**Link**: [PDF](https://arxiv.org/pdf/2506.08462)  

**Abstract**: Industrial processes must be robust and adaptable, as environments and tasks are often unpredictable, while operational errors remain costly and difficult to detect. AI-based control systems offer a path forward, yet typically depend on supervised learning with extensive labelled datasets, which limits their ability to generalize across variable and data-scarce industrial settings. Foundation models could enable broader reasoning and knowledge integration, but rarely deliver the quantitative precision demanded by engineering applications. Here, we introduceControl and Interpretation of Production via Hybrid Expertise and Reasoning (CIPHER): a vision-language-action (VLA) model framework aiming to replicate human-like reasoning for industrial control, instantiated in a commercial-grade 3D printer. It integrates a process expert, a regression model enabling quantitative characterization of system states required for engineering tasks. CIPHER also incorporates retrieval-augmented generation to access external expert knowledge and support physics-informed, chain-of-thought reasoning. This hybrid architecture exhibits strong generalization to out-of-distribution tasks. It interprets visual or textual inputs from process monitoring, explains its decisions, and autonomously generates precise machine instructions, without requiring explicit annotations. CIPHER thus lays the foundations for autonomous systems that act with precision, reason with context, and communicate decisions transparently, supporting safe and trusted deployment in industrial settings. 

**Abstract (ZH)**: 基于混合专业知识和推理的生产控制与解释（CIPHER）：一种工业控制的人类级推理 vision-language-action 模型框架 

---
# MOBODY: Model Based Off-Dynamics Offline Reinforcement Learning 

**Title (ZH)**: MOBODY：基于模型的离线动力学 offline 强化学习 

**Authors**: Yihong Guo, Yu Yang, Pan Xu, Anqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08460)  

**Abstract**: We study the off-dynamics offline reinforcement learning problem, where the goal is to learn a policy from offline datasets collected from source and target domains with mismatched transition. Existing off-dynamics offline RL methods typically either filter source transitions that resemble those of the target domain or apply reward augmentation to source data, both constrained by the limited transitions available from the target domain. As a result, the learned policy is unable to explore target domain beyond the offline datasets. We propose MOBODY, a Model-Based Off-Dynamics offline RL algorithm that addresses this limitation by enabling exploration of the target domain via learned dynamics. MOBODY generates new synthetic transitions in the target domain through model rollouts, which are used as data augmentation during offline policy learning. Unlike existing model-based methods that learn dynamics from a single domain, MOBODY tackles the challenge of mismatched dynamics by leveraging both source and target datasets. Directly merging these datasets can bias the learned model toward source dynamics. Instead, MOBODY learns target dynamics by discovering a shared latent representation of states and transitions across domains through representation learning. To stabilize training, MOBODY incorporates a behavior cloning loss that regularizes the policy. Specifically, we introduce a Q-weighted behavior cloning loss that regularizes the policy toward actions with high target-domain Q-values, rather than uniformly imitating all actions in the dataset. These Q-values are learned from an enhanced target dataset composed of offline target data, augmented source data, and rollout data from the learned target dynamics. We evaluate MOBODY on MuJoCo benchmarks and show that it significantly outperforms state-of-the-art baselines, with especially pronounced improvements in challenging scenarios. 

**Abstract (ZH)**: 基于模型的离线动力学离线强化学习算法MOBODY 

---
# DEKC: Data-Enable Control for Tethered Space Robot Deployment in the Presence of Uncertainty via Koopman Operator Theory 

**Title (ZH)**: DEKC: 数据驱动控制在不确定性环境下系留太空机器人部署的方法基于柯普曼算子理论 

**Authors**: Ao Jin, Qinyi Wang, Sijie Wen, Ya Liu, Ganghui Shen, Panfeng Huang, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08319)  

**Abstract**: This work focuses the deployment of tethered space robot in the presence of unknown uncertainty. A data-enable framework called DEKC which contains offline training part and online execution part is proposed to deploy tethered space robot in the presence of uncertainty. The main idea of this work is modeling the unknown uncertainty as a dynamical system, which enables high accuracy and convergence of capturing uncertainty. The core part of proposed framework is a proxy model of uncertainty, which is derived from data-driven Koopman theory and is separated with controller design. In the offline stage, the lifting functions associated with Koopman operator are parameterized with deep neural networks. Then by solving an optimization problem, the lifting functions are learned from sampling data. In the online execution stage, the proxy model cooperates the learned lifting functions obtained in the offline phase to capture the unknown uncertainty. Then the output of proxy model is compensated to the baseline controller such that the effect of uncertainty can be attenuated or even eliminated. Furthermore, considering some scenarios in which the performance of proxy model may weaken, a receding-horizon scheme is proposed to update the proxy model online. Finally, the extensive numerical simulations demonstrate the effectiveness of our proposed framework. The implementation of proposed DEKC framework is publicly available at this https URL. 

**Abstract (ZH)**: 本工作聚焦在未知不确定性环境下的系留空间机器人部署。提出了一种名为DEKC的数据使能框架，该框架包含离线训练部分和在线执行部分，以在不确定性环境下部署系留空间机器人。本文的主要思路是将未知不确定性建模为动力系统，从而实现不确定性捕捉的高精度和收敛性。所提框架的核心部分是不确定性代理模型，该模型源自数据驱动的科昂曼理论，并与控制设计分离。在离线阶段，与科昂曼算子关联的提升函数采用深度神经网络参数化。然后通过求解优化问题，从采样数据中学习提升函数。在在线执行阶段，代理模型与离线阶段学习到的提升函数合作，捕捉未知不确定性。然后将代理模型的输出补偿到基线控制器，以减轻或消除不确定性的影响。此外，考虑到代理模型在某些场景中性能可能减弱的情况，提出了一种前瞻式的方案在线更新代理模型。最后，广泛的数值仿真实验验证了所提框架的有效性。所提出的DEKC框架的实现可以在以下网址获取。 

---
# Scaling Laws of Motion Forecasting and Planning -- A Technical Report 

**Title (ZH)**: 运动预测与规划的标度律——技术报告 

**Authors**: Mustafa Baniodeh, Kratarth Goel, Scott Ettinger, Carlos Fuertes, Ari Seff, Tim Shen, Cole Gulino, Chenjie Yang, Ghassen Jerfel, Dokook Choe, Rui Wang, Vinutha Kallem, Sergio Casas, Rami Al-Rfou, Benjamin Sapp, Dragomir Anguelov  

**Link**: [PDF](https://arxiv.org/pdf/2506.08228)  

**Abstract**: We study the empirical scaling laws of a family of encoder-decoder autoregressive transformer models on the task of joint motion forecasting and planning in the autonomous driving domain. Using a 500 thousand hours driving dataset, we demonstrate that, similar to language modeling, model performance improves as a power-law function of the total compute budget, and we observe a strong correlation between model training loss and model evaluation metrics. Most interestingly, closed-loop metrics also improve with scaling, which has important implications for the suitability of open-loop metrics for model development and hill climbing. We also study the optimal scaling of the number of transformer parameters and the training data size for a training compute-optimal model. We find that as the training compute budget grows, optimal scaling requires increasing the model size 1.5x as fast as the dataset size. We also study inference-time compute scaling, where we observe that sampling and clustering the output of smaller models makes them competitive with larger models, up to a crossover point beyond which a larger models becomes more inference-compute efficient. Overall, our experimental results demonstrate that optimizing the training and inference-time scaling properties of motion forecasting and planning models is a key lever for improving their performance to address a wide variety of driving scenarios. Finally, we briefly study the utility of training on general logged driving data of other agents to improve the performance of the ego-agent, an important research area to address the scarcity of robotics data for large capacity models training. 

**Abstract (ZH)**: 我们研究了一类编码器-解码器自回归变压器模型在自主驾驶领域联合运动预测与规划任务中的经验标度律。使用50万小时的驾驶数据集，我们表明，类似于语言建模，模型性能随着总计算预算的幂律函数提高，并观察到模型训练损失与模型评估指标之间存在强烈的相关性。最有趣的是，闭环指标也随着标度提高，这对开放环指标在模型开发和爬坡中的适用性具有重要意义。我们还研究了训练计算最优模型的变压器参数数量和训练数据规模的最佳标度。我们发现，随着训练计算预算的增长，最优标度需要将模型大小以比数据集大小快1.5倍的速度增加。我们还研究了推理时的计算标度，观察到对较小模型的输出进行采样和聚类使其与较大模型竞争，直到交叉点之后，较大模型在推理计算效率上更具优势。总体而言，我们的实验结果表明，优化运动预测与规划模型的训练和推理时的标度特性是提高其性能以应对各种驾驶场景的关键杠杆。最后，我们简要研究了利用其他代理的一般记录驾驶数据进行训练以改进自身代理性能的研究领域，这对于解决大规模模型训练中机器人数据稀缺问题具有重要意义。 

---
# ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving 

**Title (ZH)**: ReCogDrive：端到端自动驾驶的强化认知框架 

**Authors**: Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Wenyu Liu, Xinggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08052)  

**Abstract**: Although end-to-end autonomous driving has made remarkable progress, its performance degrades significantly in rare and long-tail scenarios. Recent approaches attempt to address this challenge by leveraging the rich world knowledge of Vision-Language Models (VLMs), but these methods suffer from several limitations: (1) a significant domain gap between the pre-training data of VLMs and real-world driving data, (2) a dimensionality mismatch between the discrete language space and the continuous action space, and (3) imitation learning tends to capture the average behavior present in the dataset, which may be suboptimal even dangerous. In this paper, we propose ReCogDrive, an autonomous driving system that integrates VLMs with diffusion planner, which adopts a three-stage paradigm for training. In the first stage, we use a large-scale driving question-answering datasets to train the VLMs, mitigating the domain discrepancy between generic content and real-world driving scenarios. In the second stage, we employ a diffusion-based planner to perform imitation learning, mapping representations from the latent language space to continuous driving actions. Finally, we fine-tune the diffusion planner using reinforcement learning with NAVSIM non-reactive simulator, enabling the model to generate safer, more human-like driving trajectories. We evaluate our approach on the planning-oriented NAVSIM benchmark, achieving a PDMS of 89.6 and setting a new state-of-the-art that surpasses the previous vision-only SOTA by 5.6 PDMS. 

**Abstract (ZH)**: 尽管端到端自动驾驶取得了显著进展，但在罕见和长尾场景下的性能显著下降。近期方法尝试通过利用视觉语言模型（VLMs）丰富的世界知识来应对这一挑战，但这些方法存在几项局限性：（1）VLMs的预训练数据与真实世界驾驶数据之间存在显著Domain gap；（2）离散的语言空间与连续的动作空间之间存在维度不匹配；（3）模仿学习倾向于捕获数据集中存在的平均行为，这可能是次优的甚至是危险的。本文提出ReCogDrive，这是一种将VLMs与扩散规划器相结合的自动驾驶系统，采用三阶段训练 paradigm。在第一阶段，我们使用大规模的驾驶问答数据集训练VLMs，以缓解通用内容与真实世界驾驶场景之间的Domain discrepancy。在第二阶段，我们使用基于扩散的规划器进行模仿学习，将潜语言空间的表示映射到连续的驾驶动作。最后，我们使用基于强化学习的NAVSIM非反应式模拟器fine-tune扩散规划器，使模型能够生成更安全、更接近人类的驾驶轨迹。我们在以规划为导向的NAVSIM基准上评估了我们的方法，实现了89.6的PDMS，并且性能超过了之前的纯视觉SOTA，领先5.6 PDMS。 

---
# Towards Reliable AR-Guided Surgical Navigation: Interactive Deformation Modeling with Data-Driven Biomechanics and Prompts 

**Title (ZH)**: 基于数据驱动生物力学和提示的交互式变形建模以实现可靠的AR引导外科导航 

**Authors**: Zheng Han, Jun Zhou, Jialun Pei, Jing Qin, Yingfang Fan, Qi Dou  

**Link**: [PDF](https://arxiv.org/pdf/2506.08048)  

**Abstract**: In augmented reality (AR)-guided surgical navigation, preoperative organ models are superimposed onto the patient's intraoperative anatomy to visualize critical structures such as vessels and tumors. Accurate deformation modeling is essential to maintain the reliability of AR overlays by ensuring alignment between preoperative models and the dynamically changing anatomy. Although the finite element method (FEM) offers physically plausible modeling, its high computational cost limits intraoperative applicability. Moreover, existing algorithms often fail to handle large anatomical changes, such as those induced by pneumoperitoneum or ligament dissection, leading to inaccurate anatomical correspondences and compromised AR guidance. To address these challenges, we propose a data-driven biomechanics algorithm that preserves FEM-level accuracy while improving computational efficiency. In addition, we introduce a novel human-in-the-loop mechanism into the deformation modeling process. This enables surgeons to interactively provide prompts to correct anatomical misalignments, thereby incorporating clinical expertise and allowing the model to adapt dynamically to complex surgical scenarios. Experiments on a publicly available dataset demonstrate that our algorithm achieves a mean target registration error of 3.42 mm. Incorporating surgeon prompts through the interactive framework further reduces the error to 2.78 mm, surpassing state-of-the-art methods in volumetric accuracy. These results highlight the ability of our framework to deliver efficient and accurate deformation modeling while enhancing surgeon-algorithm collaboration, paving the way for safer and more reliable computer-assisted surgeries. 

**Abstract (ZH)**: 基于增强现实（AR）引导的外科导航中的预手术器官模型在术中解剖结构上叠加，以可视化关键结构如血管和肿瘤。准确的形变建模对于保持AR叠加的可靠性至关重要，确保预手术模型与动态变化的解剖结构之间的对齐。尽管有限元方法（FEM）提供了物理上合理的建模，但由于其高计算成本限制了术中的应用。此外，现有算法往往难以处理由腹腔充气或韧带分离引起的大型解剖变化，导致解剖对应不准确，从而损害AR引导效果。为解决这些挑战，我们提出了一种数据驱动的生物力学算法，能够在保持FEM级精度的同时提高计算效率。此外，我们引入了一种新的人工在环机制，将其纳入形变建模过程。这使外科医生能够交互地提供提示以纠正解剖偏差，从而结合临床专业知识，使模型能够动态适应复杂的手术场景。在公开数据集上的实验表明，我们的算法实现了平均目标注册误差3.42毫米。通过交互框架进一步结合外科医生提示将误差降至2.78毫米，超越了现有最先进的方法在体视精度方面的表现。这些结果突显了我们框架在提高形变建模效率和准确性的同时增强外科医生与算法合作的能力，为更安全和可靠的计算机辅助手术铺平了道路。 

---
# Neural-Augmented Kelvinlet: Real-Time Soft Tissue Deformation with Multiple Graspers 

**Title (ZH)**: 神经增强Kelvinlet：多爪器实时软组织变形算法 

**Authors**: Ashkan Shahbazi, Kyvia Pereira, Jon S. Heiselman, Elaheh Akbari, Annie C. Benson, Sepehr Seifi, Xinyuan Liu, Garrison L. Johnston, Erwin Terpstra, Anne Draaisma, Jan-Jaap Severes, Jie Ying Wu, Nabil Simaan, Michael L.Miga, Soheil Kolouri  

**Link**: [PDF](https://arxiv.org/pdf/2506.08043)  

**Abstract**: Fast and accurate simulation of soft tissue deformation is a critical factor for surgical robotics and medical training. In this paper, we introduce a novel physics-informed neural simulator that approximates soft tissue deformations in a realistic and real-time manner. Our framework integrates Kelvinlet-based priors into neural simulators, making it the first approach to leverage Kelvinlets for residual learning and regularization in data-driven soft tissue modeling. By incorporating large-scale Finite Element Method (FEM) simulations of both linear and nonlinear soft tissue responses, our method improves neural network predictions across diverse architectures, enhancing accuracy and physical consistency while maintaining low latency for real-time performance. We demonstrate the effectiveness of our approach by performing accurate surgical maneuvers that simulate the use of standard laparoscopic tissue grasping tools with high fidelity. These results establish Kelvinlet-augmented learning as a powerful and efficient strategy for real-time, physics-aware soft tissue simulation in surgical applications. 

**Abstract (ZH)**: 快速且准确的软组织变形模拟是手术机器人和医疗培训的关键因素。本文介绍了一种新颖的物理信息神经模拟器，能够在现实且实时的条件下近似软组织变形。我们的框架将Kelvinlet基础先验整合到神经模拟器中，使其成为首次利用Kelvinlets进行残差学习和数据驱动软组织建模正则化的研究方法。通过整合大规模的有限元方法(FEM)模拟，我们的方法在多种架构的神经网络预测中得到了改进，增强了准确性和物理一致性，同时保持了低延迟以实现实时性能。我们通过模拟标准腹腔镜组织抓取工具使用高保真的手术操作展示了我们方法的有效性。这些结果建立了Kelvinlet增强学习作为一种强大且高效的策略，用于手术应用中的实时、物理感知软组织模拟。 

---
