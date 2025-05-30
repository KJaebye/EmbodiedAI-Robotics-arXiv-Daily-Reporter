# HDVIO2.0: Wind and Disturbance Estimation with Hybrid Dynamics VIO 

**Title (ZH)**: HDVIO2.0: 基于混合动力学VIO的风速与干扰估计 

**Authors**: Giovanni Cioffi, Leonard Bauersfeld, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2504.00969)  

**Abstract**: Visual-inertial odometry (VIO) is widely used for state estimation in autonomous micro aerial vehicles using onboard sensors. Current methods improve VIO by incorporating a model of the translational vehicle dynamics, yet their performance degrades when faced with low-accuracy vehicle models or continuous external disturbances, like wind. Additionally, incorporating rotational dynamics in these models is computationally intractable when they are deployed in online applications, e.g., in a closed-loop control system. We present HDVIO2.0, which models full 6-DoF, translational and rotational, vehicle dynamics and tightly incorporates them into a VIO with minimal impact on the runtime. HDVIO2.0 builds upon the previous work, HDVIO, and addresses these challenges through a hybrid dynamics model combining a point-mass vehicle model with a learning-based component, with access to control commands and IMU history, to capture complex aerodynamic effects. The key idea behind modeling the rotational dynamics is to represent them with continuous-time functions. HDVIO2.0 leverages the divergence between the actual motion and the predicted motion from the hybrid dynamics model to estimate external forces as well as the robot state. Our system surpasses the performance of state-of-the-art methods in experiments using public and new drone dynamics datasets, as well as real-world flights in winds up to 25 km/h. Unlike existing approaches, we also show that accurate vehicle dynamics predictions are achievable without precise knowledge of the full vehicle state. 

**Abstract (ZH)**: 基于视觉-惯性里程计的全6自由度动力学模型（HDVIO2.0） 

---
# Time-optimal Convexified Reeds-Shepp Paths on a Sphere 

**Title (ZH)**: 球面上的时最优凸化Reeds-Shepp路径 

**Authors**: Sixu Li, Deepak Prakash Kumar, Swaroop Darbha, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.00966)  

**Abstract**: This article addresses time-optimal path planning for a vehicle capable of moving both forward and backward on a unit sphere with a unit maximum speed, and constrained by a maximum absolute turning rate $U_{max}$. The proposed formulation can be utilized for optimal attitude control of underactuated satellites, optimal motion planning for spherical rolling robots, and optimal path planning for mobile robots on spherical surfaces or uneven terrains. By utilizing Pontryagin's Maximum Principle and analyzing phase portraits, it is shown that for $U_{max}\geq1$, the optimal path connecting a given initial configuration to a desired terminal configuration falls within a sufficient list of 23 path types, each comprising at most 6 segments. These segments belong to the set $\{C,G,T\}$, where $C$ represents a tight turn with radius $r=\frac{1}{\sqrt{1+U_{max}^2}}$, $G$ represents a great circular arc, and $T$ represents a turn-in-place motion. Closed-form expressions for the angles of each path in the sufficient list are derived. The source code for solving the time-optimal path problem and visualization is publicly available at this https URL. 

**Abstract (ZH)**: 本文解决了在单元球面上既能前移又能后移、最大速度为1的车辆的时效最优路径规划问题，并且受到最大绝对转向率$U_{max}$的约束。提出的建模方法可用于未饱和卫星的姿态最优控制、球面滚动机器人最优运动规划以及球面或不平地形上移动机器人的最优路径规划。通过利用彭特里亚金最大原理并分析相位图象，证明了当$U_{max}\geq1$时，将给定初始配置连接到期望终端配置的最优路径属于23种路径类型中的某一种，每种类型最多由6段组成。这些路径段属于集合$\{C,G,T\}$，其中$C$表示半径为$r=\frac{1}{\sqrt{1+U_{max}^2}}$的紧转弯，$G$表示大圆弧，$T$表示原地转动。导出了每种路径中角度的显式表达式。关于时效最优路径问题的求解代码和可视化结果可以在下列链接公开获取。 

---
# Combined Aerial Cooperative Tethered Carrying and Path Planning for Quadrotors in Confined Environments 

**Title (ZH)**: 受限环境中四旋翼无人机的联合空中协同吊装与路径规划 

**Authors**: Marios-Nektarios Stamatopoulos, Panagiotis Koustoumpardis, Achilleas Santi Seisa, George Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2504.00926)  

**Abstract**: In this article, a novel combined aerial cooperative tethered carrying and path planning framework is introduced with a special focus on applications in confined environments. The proposed work is aiming towards solving the path planning problem for the formation of two quadrotors, while having a rope hanging below them and passing through or around obstacles. A novel composition mechanism is proposed, which simplifies the degrees of freedom of the combined aerial system and expresses the corresponding states in a compact form. Given the state of the composition, a dynamic body is generated that encapsulates the quadrotors-rope system and makes the procedure of collision checking between the system and the environment more efficient. By utilizing the above two abstractions, an RRT path planning scheme is implemented and a collision-free path for the formation is generated. This path is decomposed back to the quadrotors' desired positions that are fed to the Model Predictive Controller (MPC) for each one. The efficiency of the proposed framework is experimentally evaluated. 

**Abstract (ZH)**: 本文介绍了一种新型结合空中协同悬挂载荷和路径规划的框架，特别关注在受限环境中的应用。提出的这项工作旨在解决两个四旋翼在悬挂绳索并通过或绕过障碍物时的路径规划问题。提出了一种新颖的组合机制，简化了结合空中系统自由度，并以紧凑的形式表示相应的状态。根据组合状态，生成一个动态体来封装四旋翼-绳索系统，从而提高系统与环境之间碰撞检测的效率。利用上述两个抽象，实现了一种RRT路径规划方案，并生成了一条无碰撞路径，该路径被分解回每个四旋翼的目标位置，这些位置用于馈入模型预测控制（MPC）。所提出框架的效率通过实验进行了评估。 

---
# Context-Aware Human Behavior Prediction Using Multimodal Large Language Models: Challenges and Insights 

**Title (ZH)**: 基于上下文的多模态大型语言模型人体行为预测：挑战与见解 

**Authors**: Yuchen Liu, Lino Lerch, Luigi Palmieri, Andrey Rudenko, Sebastian Koch, Timo Ropinski, Marco Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2504.00839)  

**Abstract**: Predicting human behavior in shared environments is crucial for safe and efficient human-robot interaction. Traditional data-driven methods to that end are pre-trained on domain-specific datasets, activity types, and prediction horizons. In contrast, the recent breakthroughs in Large Language Models (LLMs) promise open-ended cross-domain generalization to describe various human activities and make predictions in any context. In particular, Multimodal LLMs (MLLMs) are able to integrate information from various sources, achieving more contextual awareness and improved scene understanding. The difficulty in applying general-purpose MLLMs directly for prediction stems from their limited capacity for processing large input sequences, sensitivity to prompt design, and expensive fine-tuning. In this paper, we present a systematic analysis of applying pre-trained MLLMs for context-aware human behavior prediction. To this end, we introduce a modular multimodal human activity prediction framework that allows us to benchmark various MLLMs, input variations, In-Context Learning (ICL), and autoregressive techniques. Our evaluation indicates that the best-performing framework configuration is able to reach 92.8% semantic similarity and 66.1% exact label accuracy in predicting human behaviors in the target frame. 

**Abstract (ZH)**: 预测共享环境中的人类行为对于安全高效的机器人交互至关重要。传统的数据驱动方法依赖于领域特定的数据集、活动类型和预测时间窗口进行预训练。相比之下，大型语言模型（LLMs）的 Recent 突破使其能够在多种领域之间进行开放性的泛化，描述各种人类活动并在任何上下文中进行预测。特别是，多模态 LLMs（MLLMs）能够集成多种来源的信息，从而实现更丰富的上下文感知和场景理解。由于通用 MLLMs 直接应用于预测时面临处理大量输入序列能力有限、对提示设计敏感以及昂贵的微调等挑战，本文对预训练 MLLMs 在上下文感知人类行为预测中的应用进行了系统性分析。为此，我们引入了一个模块化的多模态人类活动预测框架，允许我们对各种 MLLMs、输入变化、上下文学习（ICL）和自回归技术进行基准测试。我们的评估表明，表现最佳的框架配置能够在目标帧中预测人类行为的语义相似度达到 92.8%，精确标签准确率达到 66.1%。 

---
# Visual Environment-Interactive Planning for Embodied Complex-Question Answering 

**Title (ZH)**: 基于视觉环境交互规划的实体化复杂问题回答 

**Authors**: Ning Lan, Baoshan Ou, Xuemei Xie, Guangming Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00775)  

**Abstract**: This study focuses on Embodied Complex-Question Answering task, which means the embodied robot need to understand human questions with intricate structures and abstract semantics. The core of this task lies in making appropriate plans based on the perception of the visual environment. Existing methods often generate plans in a once-for-all manner, i.e., one-step planning. Such approach rely on large models, without sufficient understanding of the environment. Considering multi-step planning, the framework for formulating plans in a sequential manner is proposed in this paper. To ensure the ability of our framework to tackle complex questions, we create a structured semantic space, where hierarchical visual perception and chain expression of the question essence can achieve iterative interaction. This space makes sequential task planning possible. Within the framework, we first parse human natural language based on a visual hierarchical scene graph, which can clarify the intention of the question. Then, we incorporate external rules to make a plan for current step, weakening the reliance on large models. Every plan is generated based on feedback from visual perception, with multiple rounds of interaction until an answer is obtained. This approach enables continuous feedback and adjustment, allowing the robot to optimize its action strategy. To test our framework, we contribute a new dataset with more complex questions. Experimental results demonstrate that our approach performs excellently and stably on complex tasks. And also, the feasibility of our approach in real-world scenarios has been established, indicating its practical applicability. 

**Abstract (ZH)**: 本研究聚焦于具身复杂问题回答任务，即具身机器人需要理解具有复杂结构和抽象语义的人类问题。该任务的核心在于基于对视觉环境的感知制定合适的计划。现有方法通常采用一次性规划方式，即一步规划。此类方法依赖于大型模型，而未能充分理解环境。考虑多步规划，本文提出了顺序方式制定计划的框架。为了确保框架解决复杂问题的能力，我们构建了一个结构化的语义空间，在该空间中，层次视觉感知和问题核心的链式表达可以实现迭代交互，使顺序任务规划成为可能。在框架内，我们首先基于视觉层次场景图解析人类自然语言，以明确问题意图。然后，结合外部规则制定当前步骤的计划，减少对大型模型的依赖。每个计划都基于视觉感知的反馈生成，并通过多轮交互直至获得答案。这种方法允许持续反馈和调整，使得机器人能够优化其行动策略。为测试框架，我们贡献了一个包含更多复杂问题的新数据集。实验结果表明，该方法在复杂任务中表现出色且稳定，并且在真实场景中的可行性已得到验证，显示了其实用性。 

---
# Predictive Spray Switching for an Efficient Path Planning Pattern for Area Coverage 

**Title (ZH)**: 基于区域覆盖的高效路径规划模式的预测喷洒切换算法 

**Authors**: Mogens Plessen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00732)  

**Abstract**: This paper presents within an arable farming context a predictive logic for the on- and off-switching of a set of nozzles attached to a boom aligned along a working width and carried by a machinery with the purpose of applying spray along the working width while the machinery is traveling along a specific path planning pattern. Concatenation of multiple of those path patterns and corresponding concatenation of proposed switching logics enables nominal lossless spray application for area coverage tasks. Proposed predictive switching logic is compared to the common and state-of-the-art reactive switching logic for Boustrophedon-based path planning for area coverage. The trade-off between reduction in pathlength and increase in the number of required on- and off-switchings for proposed method is discussed. 

**Abstract (ZH)**: 本文在一个耕地环境中，提出了一种喷头切换的预测逻辑，该逻辑用于悬挂于特定作业宽度作业机具上的喷洒系统，在沿特定路径规划模式行进时，实现沿作业宽度喷洒。基于Boustrophedon路径规划的面积覆盖任务中，多个路径模式的串联及其相应的切换逻辑串联，能够实现名义上的无损失喷洒。所提出的预测切换逻辑与基于Boustrophedon路径规划的常见和先进反应式切换逻辑进行了比较。讨论了所提出方法路径长度减少与所需的开启和关闭切换次数增加之间的权衡。 

---
# Design and Validation of an Intention-Aware Probabilistic Framework for Trajectory Prediction: Integrating COLREGS, Grounding Hazards, and Planned Routes 

**Title (ZH)**: 基于意图感知的概率预测框架设计与验证：整合COLREGS、搁浅风险和计划航线 

**Authors**: Dhanika Mahipala, Trym Tengesdal, Børge Rokseth, Tor Arne Johansen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00731)  

**Abstract**: Collision avoidance capability is an essential component in an autonomous vessel navigation system. To this end, an accurate prediction of dynamic obstacle trajectories is vital. Traditional approaches to trajectory prediction face limitations in generalizability and often fail to account for the intentions of other vessels. While recent research has considered incorporating the intentions of dynamic obstacles, these efforts are typically based on the own-ship's interpretation of the situation. The current state-of-the-art in this area is a Dynamic Bayesian Network (DBN) model, which infers target vessel intentions by considering multiple underlying causes and allowing for different interpretations of the situation by different vessels. However, since its inception, there have not been any significant structural improvements to this model. In this paper, we propose enhancing the DBN model by incorporating considerations for grounding hazards and vessel waypoint information. The proposed model is validated using real vessel encounters extracted from historical Automatic Identification System (AIS) data. 

**Abstract (ZH)**: 基于差分障碍物轨迹预测的自主船舶避碰能力提升研究 

---
# Energy Weighted Learning Progress Guided Interleaved Multi-Task Learning 

**Title (ZH)**: 能源加权学习进度引导交错多任务学习 

**Authors**: Hanne Say, Suzan Ece Ada, Emre Ugur, Erhan Oztop  

**Link**: [PDF](https://arxiv.org/pdf/2504.00707)  

**Abstract**: Humans can continuously acquire new skills and knowledge by exploiting existing ones for improved learning, without forgetting them. Similarly, 'continual learning' in machine learning aims to learn new information while preserving the previously acquired knowledge. Existing research often overlooks the nature of human learning, where tasks are interleaved due to human choice or environmental constraints. So, almost never do humans master one task before switching to the next. To investigate to what extent human-like learning can benefit the learner, we propose a method that interleaves tasks based on their 'learning progress' and energy consumption. From a machine learning perspective, our approach can be seen as a multi-task learning system that balances learning performance with energy constraints while mimicking ecologically realistic human task learning. To assess the validity of our approach, we consider a robot learning setting in simulation, where the robot learns the effect of its actions in different contexts. The conducted experiments show that our proposed method achieves better performance than sequential task learning and reduces energy consumption for learning the tasks. 

**Abstract (ZH)**: 人类可以通过利用现有技能和知识来不断获取新的技能和知识，从而改善学习效果而不遗忘已有的知识。同样，机器学习中的“持续学习”旨在在保持之前获得的知识的同时学习新信息。现有研究往往忽视了人类学习的本质，即由于人类的选择或环境约束，任务往往是交错进行的。因此，人类很少在一个任务完全掌握后才转向下一个任务。为了探讨类人学习对学习者有多大益处，我们提出了一种根据“学习进展”和能量消耗交错任务的方法。从机器学习的角度来看，我们的方法可以被视为一种在同时平衡学习性能与能量约束的同时模拟生态上现实的人类任务学习过程的多任务学习系统。为了验证我们方法的有效性，我们在仿真中考虑了一个机器人学习的环境，该机器人在不同背景下学习其动作效果。实验结果表明，我们提出的方法在任务学习性能上优于顺序任务学习，并且降低了学习任务的能量消耗。 

---
# Auditory Localization and Assessment of Consequential Robot Sounds: A Multi-Method Study in Virtual Reality 

**Title (ZH)**: 基于虚拟现实的多方法研究：听觉定位与机器人后果性声音评估 

**Authors**: Marlene Wessels, Jorge de Heuvel, Leon Müller, Anna Luisa Maier, Maren Bennewitz, Johannes Kraus  

**Link**: [PDF](https://arxiv.org/pdf/2504.00697)  

**Abstract**: Mobile robots increasingly operate alongside humans but are often out of sight, so that humans need to rely on the sounds of the robots to recognize their presence. For successful human-robot interaction (HRI), it is therefore crucial to understand how humans perceive robots by their consequential sounds, i.e., operating noise. Prior research suggests that the sound of a quadruped Go1 is more detectable than that of a wheeled Turtlebot. This study builds on this and examines the human ability to localize consequential sounds of three robots (quadruped Go1, wheeled Turtlebot 2i, wheeled HSR) in Virtual Reality. In a within-subjects design, we assessed participants' localization performance for the robots with and without an acoustic vehicle alerting system (AVAS) for two velocities (0.3, 0.8 m/s) and two trajectories (head-on, radial). In each trial, participants were presented with the sound of a moving robot for 3~s and were tasked to point at its final position (localization task). Localization errors were measured as the absolute angular difference between the participants' estimated and the actual robot position. Results showed that the robot type significantly influenced the localization accuracy and precision, with the sound of the wheeled HSR (especially without AVAS) performing worst under all experimental conditions. Surprisingly, participants rated the HSR sound as more positive, less annoying, and more trustworthy than the Turtlebot and Go1 sound. This reveals a tension between subjective evaluation and objective auditory localization performance. Our findings highlight consequential robot sounds as a critical factor for designing intuitive and effective HRI, with implications for human-centered robot design and social navigation. 

**Abstract (ZH)**: 移动机器人越来越多地与人类协同工作，但常常处于人类视线之外，因此人类需要依赖机器人的声音来感知其存在。为了成功实现人机交互（HRI），理解人类通过伴随的声音（如运行噪声）感知机器人的方式至关重要。先前的研究表明，四足机器人Go1的声音比轮式机器人Turtlebot的声音更易被察觉。本研究在此基础上，探讨了人类在虚拟现实（Virtual Reality）中定位三种机器人（四足机器人Go1、轮式机器人Turtlebot 2i、轮式机器人HSR）伴随声音的能力。在单被试设计中，我们评估了参与者在有无声觉车辆警告系统（AVAS）的情况下，对于两种速度（0.3 m/s，0.8 m/s）和两种轨迹（正面、径向）的机器人定位表现。在每次试验中，参与者听到移动机器人的声音3~5秒，并被要求指出其最终位置（定位任务）。定位误差通过参与者估计位置与实际机器人位置的绝对角差来测量。结果表明，机器人类型显著影响定位准确性和精确度，尤其在没有AVAS的情况下，轮式机器人HSR的声音表现最差。有趣的是，参与者更倾向于评价HSR的声音更为积极、不那么烦人且更值得信赖，这揭示了主观评估与客观听觉定位性能之间的矛盾。本研究强调伴随声音对于设计直观有效的HRI的重要性，对于以人类为中心的机器人设计和社会导航具有重要意义。 

---
# Immersive Explainability: Visualizing Robot Navigation Decisions through XAI Semantic Scene Projections in Virtual Reality 

**Title (ZH)**: 沉浸式可解释性：通过虚拟现实中的XAI语义场景投影可视化机器人导航决策 

**Authors**: Jorge de Heuvel, Sebastian Müller, Marlene Wessels, Aftab Akhtar, Christian Bauckhage, Maren Bennewitz  

**Link**: [PDF](https://arxiv.org/pdf/2504.00682)  

**Abstract**: End-to-end robot policies achieve high performance through neural networks trained via reinforcement learning (RL). Yet, their black box nature and abstract reasoning pose challenges for human-robot interaction (HRI), because humans may experience difficulty in understanding and predicting the robot's navigation decisions, hindering trust development. We present a virtual reality (VR) interface that visualizes explainable AI (XAI) outputs and the robot's lidar perception to support intuitive interpretation of RL-based navigation behavior. By visually highlighting objects based on their attribution scores, the interface grounds abstract policy explanations in the scene context. This XAI visualization bridges the gap between obscure numerical XAI attribution scores and a human-centric semantic level of explanation. A within-subjects study with 24 participants evaluated the effectiveness of our interface for four visualization conditions combining XAI and lidar. Participants ranked scene objects across navigation scenarios based on their importance to the robot, followed by a questionnaire assessing subjective understanding and predictability. Results show that semantic projection of attributions significantly enhances non-expert users' objective understanding and subjective awareness of robot behavior. In addition, lidar visualization further improves perceived predictability, underscoring the value of integrating XAI and sensor for transparent, trustworthy HRI. 

**Abstract (ZH)**: 端到端机器人政策通过强化学习训练的神经网络实现高性能，但其黑箱性质和抽象推理给机器人与人类交互（HRI）带来挑战，因为人类可能难以理解并预测机器人的导航决策，阻碍了信任的建立。我们提出一个虚拟现实（VR）界面，可视化可解释人工智能（XAI）输出和机器人的激光雷达感知，以支持对基于强化学习（RL）导航行为的直观解释。通过根据注意力分数可视化突出显示对象，该界面将抽象的策略解释与场景上下文联系起来。这种XAI可视化填补了模糊的数值XAI注意力分数与以人类为中心的意义层面解释之间的差距。一项涉及24名参与者的被试内研究评估了在结合XAI和激光雷达的四种可视化条件下，该界面的有效性。参与者在导航场景中根据其对机器人的重要性对场景对象进行排序，随后填写问卷评估主观理解和可预测性。结果表明，意义投射的注意力显著增强了非专家用户对机器人行为的客观理解和主观意识。此外，激光雷达可视化进一步提高了感知的可预测性，强调了结合XAI和传感器以实现透明、可信赖的HRI的价值。 

---
# Optimal Control of Walkers with Parallel Actuation 

**Title (ZH)**: 平行驱动下步行者最优控制 

**Authors**: Ludovic de Matteis, Virgile Batto, Justin Carpentier, Nicolas Mansard  

**Link**: [PDF](https://arxiv.org/pdf/2504.00642)  

**Abstract**: Legged robots with closed-loop kinematic chains are increasingly prevalent due to their increased mobility and efficiency. Yet, most motion generation methods rely on serial-chain approximations, sidestepping their specific constraints and dynamics. This leads to suboptimal motions and limits the adaptability of these methods to diverse kinematic structures. We propose a comprehensive motion generation method that explicitly incorporates closed-loop kinematics and their associated constraints in an optimal control problem, integrating kinematic closure conditions and their analytical derivatives. This allows the solver to leverage the non-linear transmission effects inherent to closed-chain mechanisms, reducing peak actuator efforts and expanding their effective operating range. Unlike previous methods, our framework does not require serial approximations, enabling more accurate and efficient motion strategies. We also are able to generate the motion of more complex robots for which an approximate serial chain does not exist. We validate our approach through simulations and experiments, demonstrating superior performance in complex tasks such as rapid locomotion and stair negotiation. This method enhances the capabilities of current closed-loop robots and broadens the design space for future kinematic architectures. 

**Abstract (ZH)**: 具有闭环 kinematic 链的腿式机器人由于其增强的移动性和效率而日益普遍。然而，大多数运动生成方法依赖于串联链近似，绕过了它们特有的约束和动力学。这导致了次优运动，并限制了这些方法对多样化 kinematic 结构的适应性。我们提出了一种全面的运动生成方法，该方法在最优控制问题中明确地纳入了闭环 kinematic 链及其相关的约束条件，结合了 kinematic 闭合条件及其分析导数。这使求解器能够利用闭环机制固有的非线性传递效应，降低峰值执行器努力并扩大其有效操作范围。与先前的方法不同，我们的框架不需要串联近似，从而能够生成更准确和高效的运动策略。我们还能够为其中并不存在近似串联链的更复杂机器人生成运动。我们通过仿真和实验验证了我们的方法，在复杂的任务如快速运动和楼梯导航中表现出优越性能。此方法增强了当前闭环机器人的能力，并为未来的 kinematic 架构设计空间开辟了新的可能性。 

---
# Learning Bipedal Locomotion on Gear-Driven Humanoid Robot Using Foot-Mounted IMUs 

**Title (ZH)**: 基于脚部安装IMU的学习齿轮驱动人形机器人 bipedal 行走方法 

**Authors**: Sotaro Katayama, Yuta Koda, Norio Nagatsuka, Masaya Kinoshita  

**Link**: [PDF](https://arxiv.org/pdf/2504.00614)  

**Abstract**: Sim-to-real reinforcement learning (RL) for humanoid robots with high-gear ratio actuators remains challenging due to complex actuator dynamics and the absence of torque sensors. To address this, we propose a novel RL framework leveraging foot-mounted inertial measurement units (IMUs). Instead of pursuing detailed actuator modeling and system identification, we utilize foot-mounted IMU measurements to enhance rapid stabilization capabilities over challenging terrains. Additionally, we propose symmetric data augmentation dedicated to the proposed observation space and random network distillation to enhance bipedal locomotion learning over rough terrain. We validate our approach through hardware experiments on a miniature-sized humanoid EVAL-03 over a variety of environments. The experimental results demonstrate that our method improves rapid stabilization capabilities over non-rigid surfaces and sudden environmental transitions. 

**Abstract (ZH)**: 基于足部加速度计的高减速比腿式机器人Sim-to-real强化学习研究 

---
# Contextualized Autonomous Drone Navigation using LLMs Deployed in Edge-Cloud Computing 

**Title (ZH)**: 基于边缘-云计算部署的LLM驱动的上下文自主无人机导航 

**Authors**: Hongqian Chen, Yun Tang, Antonios Tsourdos, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.00607)  

**Abstract**: Autonomous navigation is usually trained offline in diverse scenarios and fine-tuned online subject to real-world experiences. However, the real world is dynamic and changeable, and many environmental encounters/effects are not accounted for in real-time due to difficulties in describing them within offline training data or hard to describe even in online scenarios. However, we know that the human operator can describe these dynamic environmental encounters through natural language, adding semantic context. The research is to deploy Large Language Models (LLMs) to perform real-time contextual code adjustment to autonomous navigation. The challenge not evaluated in literature is what LLMs are appropriate and where should these computationally heavy algorithms sit in the computation-communication edge-cloud computing architectures. In this paper, we evaluate how different LLMs can adjust both the navigation map parameters dynamically (e.g., contour map shaping) and also derive navigation task instruction sets. We then evaluate which LLMs are most suitable and where they should sit in future edge-cloud of 6G telecommunication architectures. 

**Abstract (ZH)**: 基于大语言模型的自主导航实时上下文代码调整研究 

---
# MRHaD: Mixed Reality-based Hand-Drawn Map Editing Interface for Mobile Robot Navigation 

**Title (ZH)**: 基于混合现实的手绘地图编辑界面 for 移动机器人导航 

**Authors**: Takumi Taki, Masato Kobayashi, Eduardo Iglesius, Naoya Chiba, Shizuka Shirai, Yuki Uranishi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00580)  

**Abstract**: Mobile robot navigation systems are increasingly relied upon in dynamic and complex environments, yet they often struggle with map inaccuracies and the resulting inefficient path planning. This paper presents MRHaD, a Mixed Reality-based Hand-drawn Map Editing Interface that enables intuitive, real-time map modifications through natural hand gestures. By integrating the MR head-mounted display with the robotic navigation system, operators can directly create hand-drawn restricted zones (HRZ), thereby bridging the gap between 2D map representations and the real-world environment. Comparative experiments against conventional 2D editing methods demonstrate that MRHaD significantly improves editing efficiency, map accuracy, and overall usability, contributing to safer and more efficient mobile robot operations. The proposed approach provides a robust technical foundation for advancing human-robot collaboration and establishing innovative interaction models that enhance the hybrid future of robotics and human society. For additional material, please check: this https URL 

**Abstract (ZH)**: 基于混合现实的手绘地图编辑界面MRHaD：一种直观的实时地图修改方法 

---
# Robust LiDAR-Camera Calibration with 2D Gaussian Splatting 

**Title (ZH)**: 鲁棒的 LiDAR-相机标定方法：基于2D高斯点云投影 

**Authors**: Shuyi Zhou, Shuxiang Xie, Ryoichi Ishikawa, Takeshi Oishi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00525)  

**Abstract**: LiDAR-camera systems have become increasingly popular in robotics recently. A critical and initial step in integrating the LiDAR and camera data is the calibration of the LiDAR-camera system. Most existing calibration methods rely on auxiliary target objects, which often involve complex manual operations, whereas targetless methods have yet to achieve practical effectiveness. Recognizing that 2D Gaussian Splatting (2DGS) can reconstruct geometric information from camera image sequences, we propose a calibration method that estimates LiDAR-camera extrinsic parameters using geometric constraints. The proposed method begins by reconstructing colorless 2DGS using LiDAR point clouds. Subsequently, we update the colors of the Gaussian splats by minimizing the photometric loss. The extrinsic parameters are optimized during this process. Additionally, we address the limitations of the photometric loss by incorporating the reprojection and triangulation losses, thereby enhancing the calibration robustness and accuracy. 

**Abstract (ZH)**: LiDAR-相机系统校准方法：基于几何约束的无靶标校准 

---
# Learning-Based Approximate Nonlinear Model Predictive Control Motion Cueing 

**Title (ZH)**: 基于学习的近似非线性模型预测控制运动模拟 

**Authors**: Camilo Gonzalez Arango, Houshyar Asadi, Mohammad Reza Chalak Qazani, Chee Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2504.00469)  

**Abstract**: Motion Cueing Algorithms (MCAs) encode the movement of simulated vehicles into movement that can be reproduced with a motion simulator to provide a realistic driving experience within the capabilities of the machine. This paper introduces a novel learning-based MCA for serial robot-based motion simulators. Building on the differentiable predictive control framework, the proposed method merges the advantages of Nonlinear Model Predictive Control (NMPC) - notably nonlinear constraint handling and accurate kinematic modeling - with the computational efficiency of machine learning. By shifting the computational burden to offline training, the new algorithm enables real-time operation at high control rates, thus overcoming the key challenge associated with NMPC-based motion cueing. The proposed MCA incorporates a nonlinear joint-space plant model and a policy network trained to mimic NMPC behavior while accounting for joint acceleration, velocity, and position limits. Simulation experiments across multiple motion cueing scenarios showed that the proposed algorithm performed on par with a state-of-the-art NMPC-based alternative in terms of motion cueing quality as quantified by the RMSE and correlation coefficient with respect to reference signals. However, the proposed algorithm was on average 400 times faster than the NMPC baseline. In addition, the algorithm successfully generalized to unseen operating conditions, including motion cueing scenarios on a different vehicle and real-time physics-based simulations. 

**Abstract (ZH)**: 基于学习的运动模拟器运动引导算法：一种串联机器人运动模拟器的新型学习导向方法 

---
# Egocentric Conformal Prediction for Safe and Efficient Navigation in Dynamic Cluttered Environments 

**Title (ZH)**: 基于自我中心可信预测的动态杂乱环境中安全高效导航 

**Authors**: Jaeuk Shin, Jungjin Lee, Insoon Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00447)  

**Abstract**: Conformal prediction (CP) has emerged as a powerful tool in robotics and control, thanks to its ability to calibrate complex, data-driven models with formal guarantees. However, in robot navigation tasks, existing CP-based methods often decouple prediction from control, evaluating models without considering whether prediction errors actually compromise safety. Consequently, ego-vehicles may become overly conservative or even immobilized when all potential trajectories appear infeasible. To address this issue, we propose a novel CP-based navigation framework that responds exclusively to safety-critical prediction errors. Our approach introduces egocentric score functions that quantify how much closer obstacles are to a candidate vehicle position than anticipated. These score functions are then integrated into a model predictive control scheme, wherein each candidate state is individually evaluated for safety. Combined with an adaptive CP mechanism, our framework dynamically adjusts to changes in obstacle motion without resorting to unnecessary conservatism. Theoretical analyses indicate that our method outperforms existing CP-based approaches in terms of cost-efficiency while maintaining the desired safety levels, as further validated through experiments on real-world datasets featuring densely populated pedestrian environments. 

**Abstract (ZH)**: 面向安全关键预测误差的配准预测导航框架 

---
# Indoor Drone Localization and Tracking Based on Acoustic Inertial Measurement 

**Title (ZH)**: 基于声学惯性测量的室内无人机定位与跟踪 

**Authors**: Yimiao Sun, Weiguo Wang, Luca Mottola, Zhang Jia, Ruijin Wang, Yuan He  

**Link**: [PDF](https://arxiv.org/pdf/2504.00445)  

**Abstract**: We present Acoustic Inertial Measurement (AIM), a one-of-a-kind technique for indoor drone localization and tracking. Indoor drone localization and tracking are arguably a crucial, yet unsolved challenge: in GPS-denied environments, existing approaches enjoy limited applicability, especially in Non-Line of Sight (NLoS), require extensive environment instrumentation, or demand considerable hardware/software changes on drones. In contrast, AIM exploits the acoustic characteristics of the drones to estimate their location and derive their motion, even in NLoS settings. We tame location estimation errors using a dedicated Kalman filter and the Interquartile Range rule (IQR) and demonstrate that AIM can support indoor spaces with arbitrary ranges and layouts. We implement AIM using an off-the-shelf microphone array and evaluate its performance with a commercial drone under varied settings. Results indicate that the mean localization error of AIM is 46% lower than that of commercial UWB-based systems in a complex 10m\times10m indoor scenario, where state-of-the-art infrared systems would not even work because of NLoS situations. When distributed microphone arrays are deployed, the mean error can be reduced to less than 0.5m in a 20m range, and even support spaces with arbitrary ranges and layouts. 

**Abstract (ZH)**: 声学惯性测量（AIM）：一种独特的室内无人机定位与跟踪技术 

---
# Think Small, Act Big: Primitive Prompt Learning for Lifelong Robot Manipulation 

**Title (ZH)**: think 小，做 大： lifelong 机器人操作 的原始提示学习 

**Authors**: Yuanqi Yao, Siao Liu, Haoming Song, Delin Qu, Qizhi Chen, Yan Ding, Bin Zhao, Zhigang Wang, Xuelong Li, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00420)  

**Abstract**: Building a lifelong robot that can effectively leverage prior knowledge for continuous skill acquisition remains significantly challenging. Despite the success of experience replay and parameter-efficient methods in alleviating catastrophic forgetting problem, naively applying these methods causes a failure to leverage the shared primitives between skills. To tackle these issues, we propose Primitive Prompt Learning (PPL), to achieve lifelong robot manipulation via reusable and extensible primitives. Within our two stage learning scheme, we first learn a set of primitive prompts to represent shared primitives through multi-skills pre-training stage, where motion-aware prompts are learned to capture semantic and motion shared primitives across different skills. Secondly, when acquiring new skills in lifelong span, new prompts are appended and optimized with frozen pretrained prompts, boosting the learning via knowledge transfer from old skills to new ones. For evaluation, we construct a large-scale skill dataset and conduct extensive experiments in both simulation and real-world tasks, demonstrating PPL's superior performance over state-of-the-art methods. 

**Abstract (ZH)**: 构建能够有效利用先验知识进行连续技能获取的终身机器人still remains significantly challenging. 

---
# Control Barrier Functions via Minkowski Operations for Safe Navigation among Polytopic Sets 

**Title (ZH)**: 基于Minkowski运算法则的多面体集安全导航的控制屏障函数方法 

**Authors**: Yi-Hsuan Chen, Shuo Liu, Wei Xiao, Calin Belta, Michael Otte  

**Link**: [PDF](https://arxiv.org/pdf/2504.00364)  

**Abstract**: Safely navigating around obstacles while respecting the dynamics, control, and geometry of the underlying system is a key challenge in robotics. Control Barrier Functions (CBFs) generate safe control policies by considering system dynamics and geometry when calculating safe forward-invariant sets. Existing CBF-based methods often rely on conservative shape approximations, like spheres or ellipsoids, which have explicit and differentiable distance functions. In this paper, we propose an optimization-defined CBF that directly considers the exact Signed Distance Function (SDF) between a polytopic robot and polytopic obstacles. Inspired by the Gilbert-Johnson-Keerthi (GJK) algorithm, we formulate both (i) minimum distance and (ii) penetration depth between polytopic sets as convex optimization problems in the space of Minkowski difference operations (the MD-space). Convenient geometric properties of the MD-space enable the derivatives of implicit SDF between two polytopes to be computed via differentiable optimization. We demonstrate the proposed framework in three scenarios including pure translation, initialization inside an unsafe set, and multi-obstacle avoidance. These three scenarios highlight the generation of a non-conservative maneuver, a recovery after starting in collision, and the consideration of multiple obstacles via pairwise CBF constraint, respectively. 

**Abstract (ZH)**: 基于优化定义的CBF方法：考虑多面体机器人与障碍物之间的精确 signed 距离函数 

---
# Safe Navigation in Dynamic Environments Using Data-Driven Koopman Operators and Conformal Prediction 

**Title (ZH)**: 使用数据驱动的Koopman算子和齐性预测在动态环境中的安全导航 

**Authors**: Kaier Liang, Guang Yang, Mingyu Cai, Cristian-Ioan Vasile  

**Link**: [PDF](https://arxiv.org/pdf/2504.00352)  

**Abstract**: We propose a novel framework for safe navigation in dynamic environments by integrating Koopman operator theory with conformal prediction. Our approach leverages data-driven Koopman approximation to learn nonlinear dynamics and employs conformal prediction to quantify uncertainty, providing statistical guarantees on approximation errors. This uncertainty is effectively incorporated into a Model Predictive Controller (MPC) formulation through constraint tightening, ensuring robust safety guarantees. We implement a layered control architecture with a reference generator providing waypoints for safe navigation. The effectiveness of our methods is validated in simulation. 

**Abstract (ZH)**: 我们提出了一种结合科赫曼算子理论与双曲预测的新颖框架，以实现动态环境下的安全导航。该方法利用数据驱动的科赫曼近似来学习非线性动力学，并采用双曲预测来量化不确定性，从而提供关于近似误差的统计保证。通过约束收紧将这种不确定性有效纳入模型预测控制器（MPC）的构架中，确保了 robust 的安全保证。我们实现了一种分层控制架构，其中参考生成器提供安全导航的航点。我们的方法在仿真实验中得到了验证。 

---
# Aligning Diffusion Model with Problem Constraints for Trajectory Optimization 

**Title (ZH)**: 基于问题约束的扩散模型轨迹优化对齐 

**Authors**: Anjian Li, Ryne Beeson  

**Link**: [PDF](https://arxiv.org/pdf/2504.00342)  

**Abstract**: Diffusion models have recently emerged as effective generative frameworks for trajectory optimization, capable of producing high-quality and diverse solutions. However, training these models in a purely data-driven manner without explicit incorporation of constraint information often leads to violations of critical constraints, such as goal-reaching, collision avoidance, and adherence to system dynamics. To address this limitation, we propose a novel approach that aligns diffusion models explicitly with problem-specific constraints, drawing insights from the Dynamic Data-driven Application Systems (DDDAS) framework. Our approach introduces a hybrid loss function that explicitly measures and penalizes constraint violations during training. Furthermore, by statistically analyzing how constraint violations evolve throughout the diffusion steps, we develop a re-weighting strategy that aligns predicted violations to ground truth statistics at each diffusion step. Evaluated on a tabletop manipulation and a two-car reach-avoid problem, our constraint-aligned diffusion model significantly reduces constraint violations compared to traditional diffusion models, while maintaining the quality of trajectory solutions. This approach is well-suited for integration into the DDDAS framework for efficient online trajectory adaptation as new environmental data becomes available. 

**Abstract (ZH)**: 基于约束对齐的扩散模型在轨迹优化中的应用 

---
# An Iterative Algorithm to Symbolically Derive Generalized n-Trailer Vehicle Kinematics 

**Title (ZH)**: 一种迭代算法用于符号推导n trailer车辆广义运动学 

**Authors**: Yuvraj Singh, Adithya Jayakumar, Giorgio Rizzoni  

**Link**: [PDF](https://arxiv.org/pdf/2504.00315)  

**Abstract**: Articulated multi-axle vehicles are interesting from a control-theoretic perspective due to their peculiar kinematic offtracking characteristics, instability modes, and singularities. Holonomic and nonholonomic constraints affecting the kinematic behavior is investigated in order to develop control-oriented kinematic models representative of these peculiarities. Then, the structure of these constraints is exploited to develop an iterative algorithm to symbolically derive yaw-plane kinematic models of generalized $n$-trailer articulated vehicles with an arbitrary number of multi-axle vehicle units. A formal proof is provided for the maximum number of kinematic controls admissible to a large-scale generalized articulated vehicle system, which leads to a generalized Ackermann steering law for $n$-trailer systems. Moreover, kinematic data collected from a test vehicle is used to validate the kinematic models and, to understand the rearward yaw rate amplification behavior of the vehicle pulling multiple simulated trailers. 

**Abstract (ZH)**: articulated 多轴车辆从控制理论的角度来看，由于其独特的动力学偏移特征、不稳定性模式和奇异性而令人感兴趣。研究影响其动力学行为的静止约束和非静止约束，以开发代表这些特性的控制导向动力学模型。然后，利用这些约束的结构，开发了一个迭代算法，以符号方式推导出n拖车 articulated 车辆的动力学模型，其中包含任意数量的多轴车辆单元。提供了对大型通用 articulated 车辆系统可允许的最大动力学控制数的正式证明，这导致了n拖车系统的通用 Ackermann 转向法则。此外，来自测试车辆的动力学数据被用来验证动力学模型，并理解车辆拖拉多个模拟拖车时的后向偏航率放大行为。 

---
# Co-design Optimization of Moving Parts for Compliance and Collision Avoidance 

**Title (ZH)**: 移动部件的协调优化设计以实现柔顺性与碰撞避免 

**Authors**: Amir M. Mirzendehdel, Morad Behandish  

**Link**: [PDF](https://arxiv.org/pdf/2504.00292)  

**Abstract**: Design requirements for moving parts in mechanical assemblies are typically specified in terms of interactions with other parts. Some are purely kinematic (e.g., pairwise collision avoidance) while others depend on physics and material properties (e.g., deformation under loads). Kinematic design methods and physics-based shape/topology optimization (SO/TO) deal separately with these requirements. They rarely talk to each other as the former uses set algebra and group theory while the latter requires discretizing and solving differential equations. Hence, optimizing a moving part based on physics typically relies on either neglecting or pruning kinematic constraints in advance, e.g., by restricting the design domain to a collision-free space using an unsweep operation. In this paper, we show that TO can be used to co-design two or more parts in relative motion to simultaneously satisfy physics-based criteria and collision avoidance. We restrict our attention to maximizing linear-elastic stiffness while penalizing collision measures aggregated in time. We couple the TO loops for two parts in relative motion so that the evolution of each part's shape is accounted for when penalizing collision for the other part. The collision measures are computed by a correlation functional that can be discretized by left- and right-multiplying the shape design variables by a pre-computed matrix that depends solely on the motion. This decoupling is key to making the computations scalable for TO iterations. We demonstrate the effectiveness of the approach with 2D and 3D examples. 

**Abstract (ZH)**: 机械装配中移动部件的设计要求通常用与其他部件的互作来指定。一些要求纯粹是机构学上的（如，成对碰撞避免），而另一些则依赖于物理和材料属性（如，在载荷下的变形）。机构学设计方法和基于物理的形状/拓扑优化（SO/TO）分别处理这些需求。前者使用集合代数和群论，后者则需要对微分方程进行离散化和求解。因此，基于物理优化一个移动部件通常要么提前忽略要么修剪机构约束。本文中，我们展示了可以通过同时满足基于物理的要求和碰撞避免来联合设计两个或多个相对运动的部件。我们将注意力集中在最大化线性弹性刚度上，并对时间累积的碰撞措施进行惩罚。我们将两个相对运动部件的优化循环耦合，使得一个部件形状的演变被考虑进去以惩罚另一个部件的碰撞。碰撞措施通过一个相关函数计算，该函数可以通过将形状设计变量左右乘以一个仅依赖于运动的先验计算矩阵来离散化。这种解耦是使计算在TO迭代中可扩展的关键。我们通过2D和3D示例展示了该方法的有效性。 

---
# Dynamics-aware Diffusion Models for Planning and Control 

**Title (ZH)**: 具备动力学意识的扩散模型在规划与控制中的应用 

**Authors**: Darshan Gadginmath, Fabio Pasqualetti  

**Link**: [PDF](https://arxiv.org/pdf/2504.00236)  

**Abstract**: This paper addresses the problem of generating dynamically admissible trajectories for control tasks using diffusion models, particularly in scenarios where the environment is complex and system dynamics are crucial for practical application. We propose a novel framework that integrates system dynamics directly into the diffusion model's denoising process through a sequential prediction and projection mechanism. This mechanism, aligned with the diffusion model's noising schedule, ensures generated trajectories are both consistent with expert demonstrations and adhere to underlying physical constraints. Notably, our approach can generate maximum likelihood trajectories and accurately recover trajectories generated by linear feedback controllers, even when explicit dynamics knowledge is unavailable. We validate the effectiveness of our method through experiments on standard control tasks and a complex non-convex optimal control problem involving waypoint tracking and collision avoidance, demonstrating its potential for efficient trajectory generation in practical applications. 

**Abstract (ZH)**: 本文探讨了使用扩散模型生成控制任务中动态容许轨迹的问题，特别是在环境复杂且系统动力学对于实际应用至关重要的场景中。我们提出了一种新的框架，通过顺序预测和投影机制将系统动力学直接整合到扩散模型的去噪过程中。该机制与扩散模型的加噪时间表相一致，确保生成的轨迹既符合专家演示，又遵守基本的物理约束。值得注意的是，即使在缺乏显式动力学知识的情况下，我们的方法也能生成最大似然轨迹，并且能够准确恢复由线性反馈控制器生成的轨迹。我们通过在标准控制任务和一个涉及航点跟踪和避障的复杂非凸最优控制问题上的实验验证了该方法的有效性，展示了其在实际应用场景中高效轨迹生成的潜力。 

---
# PneuDrive: An Embedded Pressure Control System and Modeling Toolkit for Large-Scale Soft Robots 

**Title (ZH)**: PneuDrive：一种用于大规模软机器人嵌入式压力控制系统及建模工具包 

**Authors**: Curtis C. Johnson, Daniel G. Cheney, Dallin L. Cordon, Marc D. Killpack  

**Link**: [PDF](https://arxiv.org/pdf/2504.00222)  

**Abstract**: In this paper, we present a modular pressure control system called PneuDrive that can be used for large-scale, pneumatically-actuated soft robots. The design is particularly suited for situations which require distributed pressure control and high flow rates. Up to four embedded pressure control modules can be daisy-chained together as peripherals on a robust RS-485 bus, enabling closed-loop control of up to 16 valves with pressures ranging from 0-100 psig (0-689 kPa) over distances of more than 10 meters. The system is configured as a C++ ROS node by default. However, independent of ROS, we provide a Python interface with a scripting API for added flexibility. We demonstrate our implementation of PneuDrive through various trajectory tracking experiments for a three-joint, continuum soft robot with 12 different pressure inputs. Finally, we present a modeling toolkit with implementations of three dynamic actuation models, all suitable for real-time simulation and control. We demonstrate the use of this toolkit in customizing each model with real-world data and evaluating the performance of each model. The results serve as a reference guide for choosing between several actuation models in a principled manner. A video summarizing our results can be found here: this https URL. 

**Abstract (ZH)**: 一种用于大规模 pneumatically-actuated 软机器人的模块化压力控制系统——PneuDrive及其应用 

---
# Enhancing Physical Human-Robot Interaction: Recognizing Digits via Intrinsic Robot Tactile Sensing 

**Title (ZH)**: 增强物理人机交互：通过内在机器人触觉感知识别数字 

**Authors**: Teresa Sinico, Giovanni Boschetti, Pedro Neto  

**Link**: [PDF](https://arxiv.org/pdf/2504.00167)  

**Abstract**: Physical human-robot interaction (pHRI) remains a key challenge for achieving intuitive and safe interaction with robots. Current advancements often rely on external tactile sensors as interface, which increase the complexity of robotic systems. In this study, we leverage the intrinsic tactile sensing capabilities of collaborative robots to recognize digits drawn by humans on an uninstrumented touchpad mounted to the robot's flange. We propose a dataset of robot joint torque signals along with corresponding end-effector (EEF) forces and moments, captured from the robot's integrated torque sensors in each joint, as users draw handwritten digits (0-9) on the touchpad. The pHRI-DIGI-TACT dataset was collected from different users to capture natural variations in handwriting. To enhance classification robustness, we developed a data augmentation technique to account for reversed and rotated digits inputs. A Bidirectional Long Short-Term Memory (Bi-LSTM) network, leveraging the spatiotemporal nature of the data, performs online digit classification with an overall accuracy of 94\% across various test scenarios, including those involving users who did not participate in training the system. This methodology is implemented on a real robot in a fruit delivery task, demonstrating its potential to assist individuals in everyday life. Dataset and video demonstrations are available at: this https URL. 

**Abstract (ZH)**: 基于机器人的物理人类-机器人交互 (pHRI)：利用协作机器人内嵌的触觉感知能力识别人类在未装备传感器的触控板上手写数字 

---
# SACA: A Scenario-Aware Collision Avoidance Framework for Autonomous Vehicles Integrating LLMs-Driven Reasoning 

**Title (ZH)**: 面向场景的自主车辆碰撞避免框架：结合LLM驱动的推理方法 

**Authors**: Shiyue Zhao, Junzhi Zhang, Neda Masoud, Heye Huang, Xingpeng Xia, Chengkun He  

**Link**: [PDF](https://arxiv.org/pdf/2504.00115)  

**Abstract**: Reliable collision avoidance under extreme situations remains a critical challenge for autonomous vehicles. While large language models (LLMs) offer promising reasoning capabilities, their application in safety-critical evasive maneuvers is limited by latency and robustness issues. Even so, LLMs stand out for their ability to weigh emotional, legal, and ethical factors, enabling socially responsible and context-aware collision avoidance. This paper proposes a scenario-aware collision avoidance (SACA) framework for extreme situations by integrating predictive scenario evaluation, data-driven reasoning, and scenario-preview-based deployment to improve collision avoidance decision-making. SACA consists of three key components. First, a predictive scenario analysis module utilizes obstacle reachability analysis and motion intention prediction to construct a comprehensive situational prompt. Second, an online reasoning module refines decision-making by leveraging prior collision avoidance knowledge and fine-tuning with scenario data. Third, an offline evaluation module assesses performance and stores scenarios in a memory bank. Additionally, A precomputed policy method improves deployability by previewing scenarios and retrieving or reasoning policies based on similarity and confidence levels. Real-vehicle tests show that, compared with baseline methods, SACA effectively reduces collision losses in extreme high-risk scenarios and lowers false triggering under complex conditions. Project page: this https URL. 

**Abstract (ZH)**: 可靠的极端情况下碰撞规避仍然是自动驾驶车辆面临的关键挑战。尽管大规模语言模型（LLMs）提供了有希望的推理能力，但其在安全关键的规避操作中的应用受限于延迟和鲁棒性问题。即便如此，LLMs 由于其在权衡情感、法律和伦理因素方面的能力，使得碰撞规避能够更加社会负责和情境意识。本文提出了一种情境感知碰撞规避（SACA）框架，通过融合预测情景评估、数据驱动推理和基于情景预览的部署，以改进碰撞规避决策制定。SACA 包含三个关键组件。首先，预测情景分析模块利用障碍可达性分析和运动意图预测来构建全面的情景提示。其次，在线推理模块利用先验碰撞规避知识和基于情景数据的微调来优化决策制定。第三，离线评估模块评估性能并将情景存储在记忆库中。此外，通过预计算策略方法改进可部署性，通过预览情景并基于相似性和置信度水平检索或推理策略。实车测试表明，与基线方法相比，SACA 在极端高风险情景中有效减少了碰撞损失，并在复杂条件下降低了误触发率。项目页面：this https URL。 

---
# Provably Stable Multi-Agent Routing with Bounded-Delay Adversaries in the Decision Loop 

**Title (ZH)**: 可验证稳定多Agent路由：决策循环中的有界延迟对手 

**Authors**: Roee M. Francos, Daniel Garces, Stephanie Gil  

**Link**: [PDF](https://arxiv.org/pdf/2504.00863)  

**Abstract**: In this work, we are interested in studying multi-agent routing settings, where adversarial agents are part of the assignment and decision loop, degrading the performance of the fleet by incurring bounded delays while servicing pickup-and-delivery requests. Specifically, we are interested in characterizing conditions on the fleet size and the proportion of adversarial agents for which a routing policy remains stable, where stability for a routing policy is achieved if the number of outstanding requests is uniformly bounded over time. To obtain this characterization, we first establish a threshold on the proportion of adversarial agents above which previously stable routing policies for fully cooperative fleets are provably unstable. We then derive a sufficient condition on the fleet size to recover stability given a maximum proportion of adversarial agents. We empirically validate our theoretical results on a case study on autonomous taxi routing, where we consider transportation requests from real San Francisco taxicab data. 

**Abstract (ZH)**: 在本工作中，我们关注包含敌对代理的多代理路径规划设置，敌对代理会影响路径规划政策的稳定性，通过引入有界延迟来降低车队的服务性能，特别是在执行取送请求时。具体来说，我们关注在何种车队规模和敌对代理占比条件下，路径规划政策仍能保持稳定，即路径规划政策稳定指的是时间上未完成请求的数量是均匀有界的。为了获得这一特性，我们首先确定了一个敌对代理占比的阈值，在此阈值以上，原本对完全合作车队稳定的路径规划政策是不稳定的。随后，我们推导了在给定最大敌对代理占比条件下，确保路径规划政策稳定所需的车队规模条件。我们通过基于真实旧金山出租车数据的案例研究，实证验证了理论结果。 

---
# UnIRe: Unsupervised Instance Decomposition for Dynamic Urban Scene Reconstruction 

**Title (ZH)**: UnIRe：无监督实例分解的城市动态场景重建 

**Authors**: Yunxuan Mao, Rong Xiong, Yue Wang, Yiyi Liao  

**Link**: [PDF](https://arxiv.org/pdf/2504.00763)  

**Abstract**: Reconstructing and decomposing dynamic urban scenes is crucial for autonomous driving, urban planning, and scene editing. However, existing methods fail to perform instance-aware decomposition without manual annotations, which is crucial for instance-level scene this http URL propose UnIRe, a 3D Gaussian Splatting (3DGS) based approach that decomposes a scene into a static background and individual dynamic instances using only RGB images and LiDAR point clouds. At its core, we introduce 4D superpoints, a novel representation that clusters multi-frame LiDAR points in 4D space, enabling unsupervised instance separation based on spatiotemporal correlations. These 4D superpoints serve as the foundation for our decomposed 4D initialization, i.e., providing spatial and temporal initialization to train a dynamic 3DGS for arbitrary dynamic classes without requiring bounding boxes or object this http URL, we introduce a smoothness regularization strategy in both 2D and 3D space, further improving the temporal this http URL on benchmark datasets show that our method outperforms existing methods in decomposed dynamic scene reconstruction while enabling accurate and flexible instance-level editing, making it a practical solution for real-world applications. 

**Abstract (ZH)**: 基于3D高斯散射的动态城市场景重建与分解对于自动驾驶、城市规划和场景编辑至关重要。然而，现有方法无法在无需手动标注的情况下进行实例感知分解，这对实例级场景编辑至关重要。本文提出UnIRe，一种基于3D高斯散射的方法，仅使用RGB图像和LiDAR点云将场景分解为静态背景和单独的动态实例。核心在于我们引入了4D超点，这是一种新型表示，能够在4D空间中聚类多帧LiDAR点，从而基于时空相关性实现无监督实例分离。这些4D超点作为我们分解的4D初始化的基础，为任意动态类训练动态3D高斯散射提供空间和时间初始化，而无需边界框或物体标注。在引入平滑正则化策略后，进一步提高了时间一致性。在基准数据集上的实验结果表明，我们的方法在动态场景重建分解方面优于现有方法，并且能够实现准确灵活的实例级编辑，是实际应用中的一个实用解决方案。 

---
# In-Context Learning for Zero-Shot Speed Estimation of BLDC motors 

**Title (ZH)**: 基于上下文学习的无监督BLDC电机速度估计 

**Authors**: Alessandro Colombo, Riccardo Busetto, Valentina Breschi, Marco Forgione, Dario Piga, Simone Formentin  

**Link**: [PDF](https://arxiv.org/pdf/2504.00673)  

**Abstract**: Accurate speed estimation in sensorless brushless DC motors is essential for high-performance control and monitoring, yet conventional model-based approaches struggle with system nonlinearities and parameter uncertainties. In this work, we propose an in-context learning framework leveraging transformer-based models to perform zero-shot speed estimation using only electrical measurements. By training the filter offline on simulated motor trajectories, we enable real-time inference on unseen real motors without retraining, eliminating the need for explicit system identification while retaining adaptability to varying operating conditions. Experimental results demonstrate that our method outperforms traditional Kalman filter-based estimators, especially in low-speed regimes that are crucial during motor startup. 

**Abstract (ZH)**: 基于变压器模型的上下文学习框架实现无传感器 Brushless DC 电机的精确速度估计 

---
# Interpreting and Improving Optimal Control Problems with Directional Corrections 

**Title (ZH)**: 解析并改进具有方向修正的最优控制问题 

**Authors**: Trevor Barron, Xiaojing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00415)  

**Abstract**: Many robotics tasks, such as path planning or trajectory optimization, are formulated as optimal control problems (OCPs). The key to obtaining high performance lies in the design of the OCP's objective function. In practice, the objective function consists of a set of individual components that must be carefully modeled and traded off such that the OCP has the desired solution. It is often challenging to balance multiple components to achieve the desired solution and to understand, when the solution is undesired, the impact of individual cost components. In this paper, we present a framework addressing these challenges based on the concept of directional corrections. Specifically, given the solution to an OCP that is deemed undesirable, and access to an expert providing the direction of change that would increase the desirability of the solution, our method analyzes the individual cost components for their "consistency" with the provided directional correction. This information can be used to improve the OCP formulation, e.g., by increasing the weight of consistent cost components, or reducing the weight of - or even redesigning - inconsistent cost components. We also show that our framework can automatically tune parameters of the OCP to achieve consistency with a set of corrections. 

**Abstract (ZH)**: 基于方向修正的高绩效优化控制问题设计框架 

---
# Skeletonization Quality Evaluation: Geometric Metrics for Point Cloud Analysis in Robotics 

**Title (ZH)**: 骨架化质量评价：机器人领域点云分析的几何度量 

**Authors**: Qingmeng Wen, Yu-Kun Lai, Ze Ji, Seyed Amir Tafrishi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00032)  

**Abstract**: Skeletonization is a powerful tool for shape analysis, rooted in the inherent instinct to understand an object's morphology. It has found applications across various domains, including robotics. Although skeletonization algorithms have been studied in recent years, their performance is rarely quantified with detailed numerical evaluations. This work focuses on defining and quantifying geometric properties to systematically score the skeletonization results of point cloud shapes across multiple aspects, including topological similarity, boundedness, centeredness, and smoothness. We introduce these representative metric definitions along with a numerical scoring framework to analyze skeletonization outcomes concerning point cloud data for different scenarios, from object manipulation to mobile robot navigation. Additionally, we provide an open-source tool to enable the research community to evaluate and refine their skeleton models. Finally, we assess the performance and sensitivity of the proposed geometric evaluation methods from various robotic applications. 

**Abstract (ZH)**: 骨架化是形状分析的强大工具，源自于理解对象形态的内在本能。它已在多个领域找到应用，包括机器人技术。尽管近年来对骨架化算法的研究不断增加，但其性能很少通过详细的数值评估来量化。本文集中在定义和量化几何特性，以便系统地评估点云形状骨架化结果在多个方面的表现，包括拓扑相似性、有界性、中心性和光滑性。我们提出了这些代表性的度量定义，并提供了一个数值评分框架，以分析涉及不同场景的点云数据骨架化结果，从对象操作到移动机器人导航。此外，我们提供了一个开源工具，以使研究社区能够评估和完善他们的骨架模型。最后，我们从各种机器人应用中评估所提出几何评估方法的性能和敏感性。 

---
# Enhance Vision-based Tactile Sensors via Dynamic Illumination and Image Fusion 

**Title (ZH)**: 基于动态光照和图像融合的视觉触觉传感器增强方法 

**Authors**: Artemii Redkin, Zdravko Dugonjic, Mike Lambeta, Roberto Calandra  

**Link**: [PDF](https://arxiv.org/pdf/2504.00017)  

**Abstract**: Vision-based tactile sensors use structured light to measure deformation in their elastomeric interface. Until now, vision-based tactile sensors such as DIGIT and GelSight have been using a single, static pattern of structured light tuned to the specific form factor of the sensor. In this work, we investigate the effectiveness of dynamic illumination patterns, in conjunction with image fusion techniques, to improve the quality of sensing of vision-based tactile sensors. Specifically, we propose to capture multiple measurements, each with a different illumination pattern, and then fuse them together to obtain a single, higher-quality measurement. Experimental results demonstrate that this type of dynamic illumination yields significant improvements in image contrast, sharpness, and background difference. This discovery opens the possibility of retroactively improving the sensing quality of existing vision-based tactile sensors with a simple software update, and for new hardware designs capable of fully exploiting dynamic illumination. 

**Abstract (ZH)**: 基于视觉的触觉传感器通过结构光动态照明模式与图像融合技术提高弹性界面变形测量的 effectiveness 

---
