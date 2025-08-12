# Verti-Arena: A Controllable and Standardized Indoor Testbed for Multi-Terrain Off-Road Autonomy 

**Title (ZH)**: Verti-Arena：一种可控且标准化的室内多地形越野自主测试平台 

**Authors**: Haiyue Chen, Aniket Datar, Tong Xu, Francesco Cancelliere, Harsh Rangwala, Madhan Balaji Rao, Daeun Song, David Eichinger, Xuesu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.08226)  

**Abstract**: Off-road navigation is an important capability for mobile robots deployed in environments that are inaccessible or dangerous to humans, such as disaster response or planetary exploration. Progress is limited due to the lack of a controllable and standardized real-world testbed for systematic data collection and validation. To fill this gap, we introduce Verti-Arena, a reconfigurable indoor facility designed specifically for off-road autonomy. By providing a repeatable benchmark environment, Verti-Arena supports reproducible experiments across a variety of vertically challenging terrains and provides precise ground truth measurements through onboard sensors and a motion capture system. Verti-Arena also supports consistent data collection and comparative evaluation of algorithms in off-road autonomy research. We also develop a web-based interface that enables research groups worldwide to remotely conduct standardized off-road autonomy experiments on Verti-Arena. 

**Abstract (ZH)**: 户外导航是部署在人类难以进入或危险环境中的移动机器人的一项重要能力，如灾害响应或行星探索。由于缺乏一个可控且标准化的实地测试平台来进行系统的数据收集和验证，进展受限。为填补这一空白，我们引入了Verti-Arena，一种专门为户外自主导航设计的可重新配置室内设施。通过提供一个可重复基准环境，Verti-Arena 支持在各种垂直挑战性地形上进行可重复的实验，并通过机载传感器和动作捕捉系统提供精确的地面真值测量。Verti-Arena 还支持户外自主导航研究中算法的一致数据收集和比较评估。我们还开发了一个网页界面，使世界各地的研究团队能够远程在Verti-Arena上进行标准化的户外自主导航实验。 

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
# Feedback Control of a Single-Tail Bioinspired 59-mg Swimmer 

**Title (ZH)**: 单尾生物启发型59毫克游泳器的反馈控制 

**Authors**: Conor K. Trygstad, Cody R. Longwell, Francisco M. F. R. Gonçalves, Elijah K. Blankenship, Néstor O. Pérez-Arancibia  

**Link**: [PDF](https://arxiv.org/pdf/2508.07566)  

**Abstract**: We present an evolved steerable version of the single-tail Fish-&-Ribbon-Inspired Small Swimming Harmonic roBot (FRISSHBot), a 59-mg biologically inspired swimmer, which is driven by a new shape-memory alloy (SMA)-based bimorph actuator. The new FRISSHBot is controllable in the two-dimensional (2D) space, which enabled the first demonstration of feedback-controlled trajectory tracking of a single-tail aquatic robot with onboard actuation at the subgram scale. These new capabilities are the result of a physics-informed design with an enlarged head and shortened tail relative to those of the original platform. Enhanced by its design, this new platform achieves forward swimming speeds of up to 13.6 mm/s (0.38 Bl/s), which is over four times that of the original platform. Furthermore, when following 2D references in closed loop, the tested FRISSHBot prototype attains forward swimming speeds of up to 9.1 mm/s, root-mean-square (RMS) tracking errors as low as 2.6 mm, turning rates of up to 13.1 °/s, and turning radii as small as 10 mm. 

**Abstract (ZH)**: 改进的可控二维空间单尾鱼- Ribbon启发小型游泳谐振机器人（FRISSHBot）及其反馈控制轨迹跟踪研究 

---
# A Learning-Based Framework for Collision-Free Motion Planning 

**Title (ZH)**: 基于学习的碰撞免费运动规划框架 

**Authors**: Mateus Salomão, Tianyü Ren, Alexander König  

**Link**: [PDF](https://arxiv.org/pdf/2508.07502)  

**Abstract**: This paper presents a learning-based extension to a Circular Field (CF)-based motion planner for efficient, collision-free trajectory generation in cluttered environments. The proposed approach overcomes the limitations of hand-tuned force field parameters by employing a deep neural network trained to infer optimal planner gains from a single depth image of the scene. The pipeline incorporates a CUDA-accelerated perception module, a predictive agent-based planning strategy, and a dataset generated through Bayesian optimization in simulation. The resulting framework enables real-time planning without manual parameter tuning and is validated both in simulation and on a Franka Emika Panda robot. Experimental results demonstrate successful task completion and improved generalization compared to classical planners. 

**Abstract (ZH)**: 基于学习的 Circular Field (CF) 基础运动规划器的高效、无碰撞轨迹生成扩展研究 

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
# DexFruit: Dexterous Manipulation and Gaussian Splatting Inspection of Fruit 

**Title (ZH)**: DexFruit: 柔顺操纵与水果的高斯散点图检测 

**Authors**: Aiden Swann, Alex Qiu, Matthew Strong, Angelina Zhang, Samuel Morstein, Kai Rayle, Monroe Kennedy III  

**Link**: [PDF](https://arxiv.org/pdf/2508.07118)  

**Abstract**: DexFruit is a robotic manipulation framework that enables gentle, autonomous handling of fragile fruit and precise evaluation of damage. Many fruits are fragile and prone to bruising, thus requiring humans to manually harvest them with care. In this work, we demonstrate by using optical tactile sensing, autonomous manipulation of fruit with minimal damage can be achieved. We show that our tactile informed diffusion policies outperform baselines in both reduced bruising and pick-and-place success rate across three fruits: strawberries, tomatoes, and blackberries. In addition, we introduce FruitSplat, a novel technique to represent and quantify visual damage in high-resolution 3D representation via 3D Gaussian Splatting (3DGS). Existing metrics for measuring damage lack quantitative rigor or require expensive equipment. With FruitSplat, we distill a 2D strawberry mask as well as a 2D bruise segmentation mask into the 3DGS representation. Furthermore, this representation is modular and general, compatible with any relevant 2D model. Overall, we demonstrate a 92% grasping policy success rate, up to a 20% reduction in visual bruising, and up to an 31% improvement in grasp success rate on challenging fruit compared to our baselines across our three tested fruits. We rigorously evaluate this result with over 630 trials. Please checkout our website at this https URL . 

**Abstract (ZH)**: DexFruit是一种机器人操作框架，能够实现对脆弱水果的轻柔自主处理并精确评估损伤。许多水果都很脆弱，容易受损，因此需要人类小心地手工采摘。在本文中，我们通过使用光学触觉传感技术，证明了可以实现最小损伤的水果自主操作。我们展示了我们的基于触觉的扩散策略在三种水果（草莓、番茄和黑莓）上在减少压痕和拾放成功率方面均优于基线方法。此外，我们引入了FruitSplat，这是一种用于通过3D高分辨率表示法（3DGS）表示和量化视觉损伤的新颖技术。现有的损伤测量标准缺乏定量严谨性或需要昂贵的设备。使用FruitSplat，我们将2D草莓掩码以及2D压痕分割掩码转化为3DGS表示。此外，此表示方法是模块化的和通用的，可以与任何相关联的2D模型兼容。总体而言，我们在三种测试水果上实现了92%的抓取策略成功率，视觉压痕最多减少20%，并且在具有挑战性的水果上的抓取成功率比基线方法提高了31%。我们通过超过630次试验严格评估了这一结果。请访问我们的网站：this https URL。 

---
# Model Predictive Control for Crowd Navigation via Learning-Based Trajectory Prediction 

**Title (ZH)**: 基于学习的轨迹预测的群体导航模型预测控制 

**Authors**: Mohamed Parvez Aslam, Bojan Derajic, Mohamed-Khalil Bouzidi, Sebastian Bernhard, Jan Oliver Ringert  

**Link**: [PDF](https://arxiv.org/pdf/2508.07079)  

**Abstract**: Safe navigation in pedestrian-rich environments remains a key challenge for autonomous robots. This work evaluates the integration of a deep learning-based Social-Implicit (SI) pedestrian trajectory predictor within a Model Predictive Control (MPC) framework on the physical Continental Corriere robot. Tested across varied pedestrian densities, the SI-MPC system is compared to a traditional Constant Velocity (CV) model in both open-loop prediction and closed-loop navigation. Results show that SI improves trajectory prediction - reducing errors by up to 76% in low-density settings - and enhances safety and motion smoothness in crowded scenes. Moreover, real-world deployment reveals discrepancies between open-loop metrics and closed-loop performance, as the SI model yields broader, more cautious predictions. These findings emphasize the importance of system-level evaluation and highlight the SI-MPC framework's promise for safer, more adaptive navigation in dynamic, human-populated environments. 

**Abstract (ZH)**: 基于深度学习的社会隐式模型在物理机器人上的安全导航研究：Model Predictive Control (MPC) 框架下社会隐式(SI)行人轨迹预测在丰富行人环境中的评估 

---
# From Data to Safe Mobile Robot Navigation: An Efficient and Modular Robust MPC Design Pipeline 

**Title (ZH)**: 从数据到安全移动机器人导航：一种高效且模块化的鲁棒MPC设计流程 

**Authors**: Dennis Benders, Johannes Köhler, Robert Babuška, Javier Alonso-Mora, Laura Ferranti  

**Link**: [PDF](https://arxiv.org/pdf/2508.07045)  

**Abstract**: Model predictive control (MPC) is a powerful strategy for planning and control in autonomous mobile robot navigation. However, ensuring safety in real-world deployments remains challenging due to the presence of disturbances and measurement noise. Existing approaches often rely on idealized assumptions, neglect the impact of noisy measurements, and simply heuristically guess unrealistic bounds. In this work, we present an efficient and modular robust MPC design pipeline that systematically addresses these limitations. The pipeline consists of an iterative procedure that leverages closed-loop experimental data to estimate disturbance bounds and synthesize a robust output-feedback MPC scheme. We provide the pipeline in the form of deterministic and reproducible code to synthesize the robust output-feedback MPC from data. We empirically demonstrate robust constraint satisfaction and recursive feasibility in quadrotor simulations using Gazebo. 

**Abstract (ZH)**: Model Predictive Control (MPC)是一种强大自主移动机器人导航规划与控制的策略。然而，由于存在扰动和测量噪声，在实际部署中确保安全性依然具有挑战性。现有方法往往依赖于理想化的假设，忽视了噪声测量的影响，并仅凭经验猜测不现实的边界值。在本文中，我们提出了一种高效且模块化的鲁棒MPC设计管道，系统地解决了这些限制。该管道包含一个迭代过程，利用闭环实验数据估计扰动边界并综合鲁棒输出反馈MPC方案。我们以确定性和可重复的代码形式提供了该管道，用于从数据中综合鲁棒输出反馈MPC方案。我们通过Gazebo在四旋翼飞行器仿真中 empirically 展示了鲁棒约束满足和递归可行性。 

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
# Robust-Sub-Gaussian Model Predictive Control for Safe Ultrasound-Image-Guided Robotic Spinal Surgery 

**Title (ZH)**: 鲁棒-亚高斯模型预测控制以实现安全的超声影像引导脊柱手术 

**Authors**: Yunke Ao, Manish Prajapat, Yarden As, Yassine Taoudi-Benchekroun, Fabio Carrillo, Hooman Esfandiari, Benjamin F. Grewe, Andreas Krause, Philipp Fürnstahl  

**Link**: [PDF](https://arxiv.org/pdf/2508.06744)  

**Abstract**: Safety-critical control using high-dimensional sensory feedback from optical data (e.g., images, point clouds) poses significant challenges in domains like autonomous driving and robotic surgery. Control can rely on low-dimensional states estimated from high-dimensional data. However, the estimation errors often follow complex, unknown distributions that standard probabilistic models fail to capture, making formal safety guarantees challenging. In this work, we introduce a novel characterization of these general estimation errors using sub-Gaussian noise with bounded mean. We develop a new technique for uncertainty propagation of proposed noise characterization in linear systems, which combines robust set-based methods with the propagation of sub-Gaussian variance proxies. We further develop a Model Predictive Control (MPC) framework that provides closed-loop safety guarantees for linear systems under the proposed noise assumption. We apply this MPC approach in an ultrasound-image-guided robotic spinal surgery pipeline, which contains deep-learning-based semantic segmentation, image-based registration, high-level optimization-based planning, and low-level robotic control. To validate the pipeline, we developed a realistic simulation environment integrating real human anatomy, robot dynamics, efficient ultrasound simulation, as well as in-vivo data of breathing motion and drilling force. Evaluation results in simulation demonstrate the potential of our approach for solving complex image-guided robotic surgery task while ensuring safety. 

**Abstract (ZH)**: 使用光学数据（例如图像、点云）的高维感知反馈进行安全关键控制在自动驾驶和机器人手术等领域面临重大挑战。控制可以依赖于从高维数据估计的低维状态。然而，估计错误通常遵循复杂且未知的分布，标准概率模型难以捕捉，从而使得正式的安全保证变得具有挑战性。在本文中，我们引入了一种新的方法来描述这些一般的估计误差，使用有界均值的亚高斯噪声。我们开发了一种新技术，用于线性系统中拟议噪声特征的不确定性传播方法，该方法结合了鲁棒集合方法与亚高斯方差代理的传播。我们进一步开发了一种模型预测控制（MPC）框架，在提出的噪声假定下为线性系统提供了闭环安全保证。我们将在深度学习基于语义分割、基于图像的配准、高层优化为基础的规划和低层机器人控制的超声图像引导下机器人脊柱手术流程中应用该种MPC方法。为了验证该流程，我们构建了一个现实的仿真环境，集成了真实的人体解剖、机器人动力学、高效的超声仿真以及实时呼吸运动和钻孔力的体内数据。仿真实验结果证明了该方法在确保安全的同时解决复杂成像引导机器人手术任务的潜力。 

---
# Improved Obstacle Avoidance for Autonomous Robots with ORCA-FLC 

**Title (ZH)**: 基于ORCA-FLC的自主机器人改进型障碍避让 

**Authors**: Justin London  

**Link**: [PDF](https://arxiv.org/pdf/2508.06722)  

**Abstract**: Obstacle avoidance enables autonomous agents and robots to operate safely and efficiently in dynamic and complex environments, reducing the risk of collisions and damage. For a robot or autonomous system to successfully navigate through obstacles, it must be able to detect such obstacles. While numerous collision avoidance algorithms like the dynamic window approach (DWA), timed elastic bands (TEB), and reciprocal velocity obstacles (RVO) have been proposed, they may lead to suboptimal paths due to fixed weights, be computationally expensive, or have limited adaptability to dynamic obstacles in multi-agent environments. Optimal reciprocal collision avoidance (ORCA), which improves on RVO, provides smoother trajectories and stronger collision avoidance guarantees. We propose ORCA-FL to improve on ORCA by using fuzzy logic controllers (FLCs) to better handle uncertainty and imprecision for obstacle avoidance in path planning. Numerous multi-agent experiments are conducted and it is shown that ORCA-FL can outperform ORCA in reducing the number of collision if the agent has a velocity that exceeds a certain threshold. In addition, a proposed algorithm for improving ORCA-FL using fuzzy Q reinforcement learning (FQL) is detailed for optimizing and tuning FLCs. 

**Abstract (ZH)**: 基于模糊逻辑的优化碰撞规避（ORCA-FL）：减少多自主系统碰撞的方法 

---
# Robust and Agile Quadrotor Flight via Adaptive Unwinding-Free Quaternion Sliding Mode Control 

**Title (ZH)**: 基于自适应无松弛四元数滑模控制的鲁棒且灵活的四旋翼飞行 

**Authors**: Amin Yazdanshenas, Reza Faieghi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06568)  

**Abstract**: This paper presents a new adaptive sliding mode control (SMC) framework for quadrotors that achieves robust and agile flight under tight computational constraints. The proposed controller addresses key limitations of prior SMC formulations, including (i) the slow convergence and almost-global stability of $\mathrm{SO(3)}$-based methods, (ii) the oversimplification of rotational dynamics in Euler-based controllers, (iii) the unwinding phenomenon in quaternion-based formulations, and (iv) the gain overgrowth problem in adaptive SMC schemes. Leveraging nonsmooth stability analysis, we provide rigorous global stability proofs for both the nonsmooth attitude sliding dynamics defined on $\mathbb{S}^3$ and the position sliding dynamics. Our controller is computationally efficient and runs reliably on a resource-constrained nano quadrotor, achieving 250 Hz and 500 Hz refresh rates for position and attitude control, respectively. In an extensive set of hardware experiments with over 130 flight trials, the proposed controller consistently outperforms three benchmark methods, demonstrating superior trajectory tracking accuracy and robustness with relatively low control effort. The controller enables aggressive maneuvers such as dynamic throw launches, flip maneuvers, and accelerations exceeding 3g, which is remarkable for a 32-gram nano quadrotor. These results highlight promising potential for real-world applications, particularly in scenarios requiring robust, high-performance flight control under significant external disturbances and tight computational constraints. 

**Abstract (ZH)**: 一种适用于小型四旋翼无人机的新型自适应滑模控制框架：在严格计算约束下的鲁棒敏捷飞行控制 

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
# Best-Effort Policies for Robust Markov Decision Processes 

**Title (ZH)**: 最佳努力策略对于稳健的马尔可夫决策过程 

**Authors**: Alessandro Abate, Thom Badings, Giuseppe De Giacomo, Francesco Fabiano  

**Link**: [PDF](https://arxiv.org/pdf/2508.07790)  

**Abstract**: We study the common generalization of Markov decision processes (MDPs) with sets of transition probabilities, known as robust MDPs (RMDPs). A standard goal in RMDPs is to compute a policy that maximizes the expected return under an adversarial choice of the transition probabilities. If the uncertainty in the probabilities is independent between the states, known as s-rectangularity, such optimal robust policies can be computed efficiently using robust value iteration. However, there might still be multiple optimal robust policies, which, while equivalent with respect to the worst-case, reflect different expected returns under non-adversarial choices of the transition probabilities. Hence, we propose a refined policy selection criterion for RMDPs, drawing inspiration from the notions of dominance and best-effort in game theory. Instead of seeking a policy that only maximizes the worst-case expected return, we additionally require the policy to achieve a maximal expected return under different (i.e., not fully adversarial) transition probabilities. We call such a policy an optimal robust best-effort (ORBE) policy. We prove that ORBE policies always exist, characterize their structure, and present an algorithm to compute them with a small overhead compared to standard robust value iteration. ORBE policies offer a principled tie-breaker among optimal robust policies. Numerical experiments show the feasibility of our approach. 

**Abstract (ZH)**: 我们研究带有转换概率集合的马尔可夫决策过程（MDPs）的通用化，称为鲁棒MDPs（RMDPs）。在RMDPs中，一个标准目标是计算在对手选择转换概率的情况下，能够最大化期望回报的策略。如果概率的不确定性在状态下是独立的，即s-矩形性，那么这样的最优鲁棒策略可以使用鲁棒值迭代高效地计算出来。然而，可能存在多个最优鲁棒策略，虽然从最坏情况来看是等价的，但在非对手选择的转换概率下，它们可能具有不同的期望回报。因此，我们提出了一种针对RMDPs的细化策略选择标准，借鉴了博弈论中的支配和尽力而为的概念。我们不仅仅寻求最大化最坏情况期望回报的策略，还要求该策略能够在不同的（即，非完全对手的）转换概率下实现最大期望回报。我们称这样的策略为最优鲁棒尽力而为（ORBE）策略。我们证明了ORBE策略总是存在的，刻画了它们的结构，并提出了一种与标准鲁棒值迭代相比具有较小开销的算法来计算它们。ORBE策略为最优鲁棒策略之间提供了一个原则性的决定标准。数值实验展示了我们方法的可行性。 

---
# UPP: Unified Path Planner with Adaptive Safety and Optimality 

**Title (ZH)**: 统一路径规划器：自适应安全与最优性结合 

**Authors**: Jatin Kumar Arora, Shubhendu Bhasin  

**Link**: [PDF](https://arxiv.org/pdf/2505.23197)  

**Abstract**: We are surrounded by robots helping us perform complex tasks. Robots have a wide range of applications, from industrial automation to personalized assistance. However, with great technological innovation come significant challenges. One of the major challenges in robotics is path planning. Despite advancements such as graph search, sampling, and potential field methods, most path planning algorithms focus either on optimality or on safety. Very little research addresses both simultaneously. We propose a Unified Path Planner (UPP) that uses modified heuristics and a dynamic safety cost function to balance safety and optimality. The level of safety can be adjusted via tunable parameters, trading off against computational complexity. We demonstrate the planner's performance in simulations, showing how parameter variation affects results. UPP is compared with various traditional and safe-optimal planning algorithms across different scenarios. We also validate it on a TurtleBot, where the robot successfully finds safe and sub-optimal paths. 

**Abstract (ZH)**: 我们周围充斥着帮助我们完成复杂任务的机器人。机器人有着广泛的应用范围，从工业自动化到个性化辅助。然而，随着技术的不断革新，随之而来的挑战也极为严峻。机器人领域的一个主要挑战是路径规划。尽管已经出现了诸如图搜索、采样和势场方法等进步，但大多数路径规划算法要么关注最优性，要么关注安全性。很少有研究同时兼顾两者。我们提出了一种统一路径规划器（UPP），它利用修改后的启发式方法和动态安全成本函数来平衡安全性和最优性。通过可调参数可以调整安全水平，这会牺牲计算复杂度。我们在仿真实验中展示了规划器的性能，展示了参数变化如何影响结果。UPP 在与各种传统及安全最优规划算法的对比中，在不同场景下得到了验证，并且在TurtleBot上也成功找到了安全但次优的路径。 

---
