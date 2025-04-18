# ViTa-Zero: Zero-shot Visuotactile Object 6D Pose Estimation 

**Title (ZH)**: ViTa-Zero: Zero-shot 视触觉对象6D姿态估计 

**Authors**: Hongyu Li, James Akl, Srinath Sridhar, Tye Brady, Taskin Padir  

**Link**: [PDF](https://arxiv.org/pdf/2504.13179)  

**Abstract**: Object 6D pose estimation is a critical challenge in robotics, particularly for manipulation tasks. While prior research combining visual and tactile (visuotactile) information has shown promise, these approaches often struggle with generalization due to the limited availability of visuotactile data. In this paper, we introduce ViTa-Zero, a zero-shot visuotactile pose estimation framework. Our key innovation lies in leveraging a visual model as its backbone and performing feasibility checking and test-time optimization based on physical constraints derived from tactile and proprioceptive observations. Specifically, we model the gripper-object interaction as a spring-mass system, where tactile sensors induce attractive forces, and proprioception generates repulsive forces. We validate our framework through experiments on a real-world robot setup, demonstrating its effectiveness across representative visual backbones and manipulation scenarios, including grasping, object picking, and bimanual handover. Compared to the visual models, our approach overcomes some drastic failure modes while tracking the in-hand object pose. In our experiments, our approach shows an average increase of 55% in AUC of ADD-S and 60% in ADD, along with an 80% lower position error compared to FoundationPose. 

**Abstract (ZH)**: 基于视觉和触觉的六自由度姿态估计在机器人领域的零样本框架 

---
# Force and Speed in a Soft Stewart Platform 

**Title (ZH)**: 软斯坦利平台的力与速度研究 

**Authors**: Jake Ketchum, James Avtges, Millicent Schlafly, Helena Young, Taekyoung Kim, Ryan L. Truby, Todd D. Murphey  

**Link**: [PDF](https://arxiv.org/pdf/2504.13127)  

**Abstract**: Many soft robots struggle to produce dynamic motions with fast, large displacements. We develop a parallel 6 degree-of-freedom (DoF) Stewart-Gough mechanism using Handed Shearing Auxetic (HSA) actuators. By using soft actuators, we are able to use one third as many mechatronic components as a rigid Stewart platform, while retaining a working payload of 2kg and an open-loop bandwidth greater than 16Hx. We show that the platform is capable of both precise tracing and dynamic disturbance rejection when controlling a ball and sliding puck using a Proportional Integral Derivative (PID) controller. We develop a machine-learning-based kinematics model and demonstrate a functional workspace of roughly 10cm in each translation direction and 28 degrees in each orientation. This 6DoF device has many of the characteristics associated with rigid components - power, speed, and total workspace - while capturing the advantages of soft mechanisms. 

**Abstract (ZH)**: 基于HSA执行器的并行6自由度斯特尔-戈奇机制：软执行器在精确跟踪和动态扰动拒绝中的应用及机器学习动力学模型 

---
# Imperative MPC: An End-to-End Self-Supervised Learning with Differentiable MPC for UAV Attitude Control 

**Title (ZH)**: imperative MPC：基于可微分MPC的端到端自监督学习无人机姿态控制 

**Authors**: Haonan He, Yuheng Qiu, Junyi Geng  

**Link**: [PDF](https://arxiv.org/pdf/2504.13088)  

**Abstract**: Modeling and control of nonlinear dynamics are critical in robotics, especially in scenarios with unpredictable external influences and complex dynamics. Traditional cascaded modular control pipelines often yield suboptimal performance due to conservative assumptions and tedious parameter tuning. Pure data-driven approaches promise robust performance but suffer from low sample efficiency, sim-to-real gaps, and reliance on extensive datasets. Hybrid methods combining learning-based and traditional model-based control in an end-to-end manner offer a promising alternative. This work presents a self-supervised learning framework combining learning-based inertial odometry (IO) module and differentiable model predictive control (d-MPC) for Unmanned Aerial Vehicle (UAV) attitude control. The IO denoises raw IMU measurements and predicts UAV attitudes, which are then optimized by MPC for control actions in a bi-level optimization (BLO) setup, where the inner MPC optimizes control actions and the upper level minimizes discrepancy between real-world and predicted performance. The framework is thus end-to-end and can be trained in a self-supervised manner. This approach combines the strength of learning-based perception with the interpretable model-based control. Results show the effectiveness even under strong wind. It can simultaneously enhance both the MPC parameter learning and IMU prediction performance. 

**Abstract (ZH)**: 基于自监督学习的集成学习导向航命周期控制框架用于UAV姿态控制 

---
# Krysalis Hand: A Lightweight, High-Payload, 18-DoF Anthropomorphic End-Effector for Robotic Learning and Dexterous Manipulation 

**Title (ZH)**: Krysalis 手部：一种轻量化、高负载、18 自由度的人类仿生末端执行器，用于机器人学习和灵巧操作 

**Authors**: Al Arsh Basheer, Justin Chang, Yuyang Chen, David Kim, Iman Soltani  

**Link**: [PDF](https://arxiv.org/pdf/2504.12967)  

**Abstract**: This paper presents the Krysalis Hand, a five-finger robotic end-effector that combines a lightweight design, high payload capacity, and a high number of degrees of freedom (DoF) to enable dexterous manipulation in both industrial and research settings. This design integrates the actuators within the hand while maintaining an anthropomorphic form. Each finger joint features a self-locking mechanism that allows the hand to sustain large external forces without active motor engagement. This approach shifts the payload limitation from the motor strength to the mechanical strength of the hand, allowing the use of smaller, more cost-effective motors. With 18 DoF and weighing only 790 grams, the Krysalis Hand delivers an active squeezing force of 10 N per finger and supports a passive payload capacity exceeding 10 lbs. These characteristics make Krysalis Hand one of the lightest, strongest, and most dexterous robotic end-effectors of its kind. Experimental evaluations validate its ability to perform intricate manipulation tasks and handle heavy payloads, underscoring its potential for industrial applications as well as academic research. All code related to the Krysalis Hand, including control and teleoperation, is available on the project GitHub repository: this https URL 

**Abstract (ZH)**: 本文介绍了Krysalis手部，这是一种结合轻量化设计、高负载能力和大量自由度（DoF）的五指仿人机器人末端执行器，能够在工业和研究环境中实现灵巧操作。该设计将驱动器集成在手部中，同时保持仿人形结构。每个手指关节配备了自锁机制，使手部能够在不主动参与电机驱动的情况下承受较大的外部力量。这种方法将负载限制从电机强度转移到手部的机械强度上，从而能够使用更小、更经济的电机。Krysalis手部拥有18个自由度，重量仅790克，每指提供10 N的主动捏力，并支持超过10磅的被动负载能力。这些特性使其成为此类中最轻、最坚固和最灵巧的机器人末端执行器之一。实验评估验证了其执行复杂操作任务和处理重负载的能力，凸显了其在工业应用和学术研究中的潜力。Krysalis手部的相关代码（包括控制和远程操作）均可在项目GitHub仓库中获得：this https URL 

---
# Taccel: Scaling Up Vision-based Tactile Robotics via High-performance GPU Simulation 

**Title (ZH)**: Taccel: 通过高性能GPU仿真扩展基于视觉的触觉机器人技术 

**Authors**: Yuyang Li, Wenxin Du, Chang Yu, Puhao Li, Zihang Zhao, Tengyu Liu, Chenfanfu Jiang, Yixin Zhu, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12908)  

**Abstract**: Tactile sensing is crucial for achieving human-level robotic capabilities in manipulation tasks. VBTSs have emerged as a promising solution, offering high spatial resolution and cost-effectiveness by sensing contact through camera-captured deformation patterns of elastic gel pads. However, these sensors' complex physical characteristics and visual signal processing requirements present unique challenges for robotic applications. The lack of efficient and accurate simulation tools for VBTS has significantly limited the scale and scope of tactile robotics research. Here we present Taccel, a high-performance simulation platform that integrates IPC and ABD to model robots, tactile sensors, and objects with both accuracy and unprecedented speed, achieving an 18-fold acceleration over real-time across thousands of parallel environments. Unlike previous simulators that operate at sub-real-time speeds with limited parallelization, Taccel provides precise physics simulation and realistic tactile signals while supporting flexible robot-sensor configurations through user-friendly APIs. Through extensive validation in object recognition, robotic grasping, and articulated object manipulation, we demonstrate precise simulation and successful sim-to-real transfer. These capabilities position Taccel as a powerful tool for scaling up tactile robotics research and development. By enabling large-scale simulation and experimentation with tactile sensing, Taccel accelerates the development of more capable robotic systems, potentially transforming how robots interact with and understand their physical environment. 

**Abstract (ZH)**: 触觉感知对于实现操纵任务中的人类级机器人能力至关重要。基于视觉的触觉传感器（VBTSs）因其通过弹性凝胶垫变形模式的相机捕获进行接触传感，提供了高空间分辨率和成本效益，而备受关注。然而，这些传感器复杂的物理特性和视觉信号处理要求为机器人应用带来了独特的挑战。缺乏高效的触觉机器人仿真工具显著限制了触觉机器人研究的规模和范围。本文介绍了一种高性能仿真平台Taccel，该平台结合了IPC和ABD，实现了机器人、触觉传感器和物体的高精度和前所未有的高速建模，相较于实时操作实现了18倍的速度提升，支持数千个并行环境。不同于之前运行在次实时速度且并行化有限的仿真的模拟器，Taccel提供了精确的物理模拟和-realistic触觉信号，同时通过用户友好的API支持灵活的机器人-传感器配置。通过在物体识别、机器人抓取和活动物体操纵方面的广泛验证，我们展示了精确的仿真和成功的仿真实验到现实世界的转移。这些能力使Taccel成为扩展触觉机器人研究和开发的强大工具。通过对触觉传感进行大规模仿真和实验，Taccel加速了更先进机器人系统的发展，有望改变机器人如何与和理解物理环境的方式。 

---
# Versatile, Robust, and Explosive Locomotion with Rigid and Articulated Compliant Quadrupeds 

**Title (ZH)**: 具有刚性与 articulated 柔顺四肢的多功能、 robust 和爆炸式运动的 quadruped 机器人 

**Authors**: Jiatao Ding, Peiyu Yang, Fabio Boekel, Jens Kober, Wei Pan, Matteo Saveriano, Cosimo Della Santina  

**Link**: [PDF](https://arxiv.org/pdf/2504.12854)  

**Abstract**: Achieving versatile and explosive motion with robustness against dynamic uncertainties is a challenging task. Introducing parallel compliance in quadrupedal design is deemed to enhance locomotion performance, which, however, makes the control task even harder. This work aims to address this challenge by proposing a general template model and establishing an efficient motion planning and control pipeline. To start, we propose a reduced-order template model-the dual-legged actuated spring-loaded inverted pendulum with trunk rotation-which explicitly models parallel compliance by decoupling spring effects from active motor actuation. With this template model, versatile acrobatic motions, such as pronking, froggy jumping, and hop-turn, are generated by a dual-layer trajectory optimization, where the singularity-free body rotation representation is taken into consideration. Integrated with a linear singularity-free tracking controller, enhanced quadrupedal locomotion is achieved. Comparisons with the existing template model reveal the improved accuracy and generalization of our model. Hardware experiments with a rigid quadruped and a newly designed compliant quadruped demonstrate that i) the template model enables generating versatile dynamic motion; ii) parallel elasticity enhances explosive motion. For example, the maximal pronking distance, hop-turn yaw angle, and froggy jumping distance increase at least by 25%, 15% and 25%, respectively; iii) parallel elasticity improves the robustness against dynamic uncertainties, including modelling errors and external disturbances. For example, the allowable support surface height variation increases by 100% for robust froggy jumping. 

**Abstract (ZH)**: 实现鲁棒性强的多功能和爆发性运动是一项挑战性任务。通过四足机器人设计引入并联柔顺性被认为能提升运动性能，然而这使得控制任务更复杂。本文通过提出一种通用的模板模型和建立一个高效的动力学规划与控制流水线来应对这一挑战。首先，我们提出一种简化版的模板模型——带有躯干旋转的双足主动簧载倒立摆模型，该模型通过解耦弹簧效应和主动电机驱动来明确建模并联柔顺性。基于该模板模型，通过双层轨迹优化生成多功能杂技动作，同时考虑到无奇点的体旋转表示。结合线性无奇点跟踪控制器，实现了增强的四足运动表现。与现有模板模型的对比证明了模型提高了精度和泛化能力。使用刚性四足机器人和新设计的柔性四足机器人的硬件实验表明：i) 模板模型能够生成多功能动态动作；ii) 并联弹性能够提升爆发性动作；例如，最大蹦跃距离、跳跃-转身航向角和青蛙跳跃距离分别至少增加25%、15%和25%；iii) 并联弹性提高了对动态不确定性（包括建模误差和外部干扰）的鲁棒性，例如，稳健青蛙跳跃时允许的支持表面高度变化范围增大了100%。 

---
# Biasing the Driving Style of an Artificial Race Driver for Online Time-Optimal Maneuver Planning 

**Title (ZH)**: 为在线最优机动规划调整人工赛车司机的驾驶风格偏差 

**Authors**: Sebastiano Taddei, Mattia Piccinini, Francesco Biral  

**Link**: [PDF](https://arxiv.org/pdf/2504.12744)  

**Abstract**: In this work, we present a novel approach to bias the driving style of an artificial race driver (ARD) for online time-optimal trajectory planning. Our method leverages a nonlinear model predictive control (MPC) framework that combines time minimization with exit speed maximization at the end of the planning horizon. We introduce a new MPC terminal cost formulation based on the trajectory planned in the previous MPC step, enabling ARD to adapt its driving style from early to late apex maneuvers in real-time. Our approach is computationally efficient, allowing for low replan times and long planning horizons. We validate our method through simulations, comparing the results against offline minimum-lap-time (MLT) optimal control and online minimum-time MPC solutions. The results demonstrate that our new terminal cost enables ARD to bias its driving style, and achieve online lap times close to the MLT solution and faster than the minimum-time MPC solution. Our approach paves the way for a better understanding of the reasons behind human drivers' choice of early or late apex maneuvers. 

**Abstract (ZH)**: 本研究提出了一种新颖的方法，用于为在线时间最优轨迹规划偏置人工赛车手（ARD）的驾驶风格。我们的方法利用了结合时间最小化和规划末期出口速度最大化的非线性模型预测控制（MPC）框架。我们引入了一种新的MPC终端成本公式，基于上一步MPC计划的轨迹，使ARD能够实时适应其驾驶风格，从早期到晚期的转向动作。该方法计算效率高，允许低重规划时间和长规划时间。我们通过仿真验证了该方法，将结果与离线最短圈时（MLT）最优控制和在线最短时间MPC解进行了比较。结果表明，我们的新终端成本使ARD能够偏置其驾驶风格，并实现接近MLT解的在线圈时，且快于最短时间MPC解。该方法为更好地理解人类驾驶员选择早期或晚期转向动作的原因铺平了道路。 

---
# B*: Efficient and Optimal Base Placement for Fixed-Base Manipulators 

**Title (ZH)**: B*: 固定基座 manipulator 有效且最优的基座位置规划 

**Authors**: Zihang Zhao, Leiyao Cui, Sirui Xie, Saiyao Zhang, Zhi Han, Lecheng Ruan, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12719)  

**Abstract**: B* is a novel optimization framework that addresses a critical challenge in fixed-base manipulator robotics: optimal base placement. Current methods rely on pre-computed kinematics databases generated through sampling to search for solutions. However, they face an inherent trade-off between solution optimality and computational efficiency when determining sampling resolution. To address these limitations, B* unifies multiple objectives without database dependence. The framework employs a two-layer hierarchical approach. The outer layer systematically manages terminal constraints through progressive tightening, particularly for base mobility, enabling feasible initialization and broad solution exploration. The inner layer addresses non-convexities in each outer-layer subproblem through sequential local linearization, converting the original problem into tractable sequential linear programming (SLP). Testing across multiple robot platforms demonstrates B*'s effectiveness. The framework achieves solution optimality five orders of magnitude better than sampling-based approaches while maintaining perfect success rates and reduced computational overhead. Operating directly in configuration space, B* enables simultaneous path planning with customizable optimization criteria. B* serves as a crucial initialization tool that bridges the gap between theoretical motion planning and practical deployment, where feasible trajectory existence is fundamental. 

**Abstract (ZH)**: B* 是一种新型优化框架，针对固定基座 manipulator 机器人中的关键挑战：最优基座定位进行了优化。当前方法依赖于通过采样生成的预计算运动学数据库来搜索解决方案。然而，它们在确定采样分辨率时面临解决方案最优性和计算效率之间的固有权衡。为了解决这些限制，B* 统一了多个目标，而不依赖于数据库。该框架采用两层层次结构方法。外层通过逐步收紧终端约束系统地管理终端约束，特别是对于基座的移动性，从而实现可行的初始化和广泛的解决方案探索。内层通过逐步局部线性化解决每一层子问题中的非凸性，将原始问题转化为可处理的顺序线性规划（SLP）。在多个机器人平台上进行的测试表明 B* 的有效性。该框架在保持完美成功率的同时，计算开销减少，并且解决方案最优性比基于采样的方法高出五个数量级。B* 直接在配置空间中操作，同时实现路径规划和可定制的优化标准。B* 作为理论运动规划与实际部署之间的重要初始化工具，其核心在于可行轨迹的存在至关重要。 

---
# A Genetic Approach to Gradient-Free Kinodynamic Planning in Uneven Terrains 

**Title (ZH)**: 基于遗传算法的无导数动力学规划在不平地形中 

**Authors**: Otobong Jerome, Alexandr Klimchik, Alexander Maloletov, Geesara Kulathunga  

**Link**: [PDF](https://arxiv.org/pdf/2504.12678)  

**Abstract**: This paper proposes a genetic algorithm-based kinodynamic planning algorithm (GAKD) for car-like vehicles navigating uneven terrains modeled as triangular meshes. The algorithm's distinct feature is trajectory optimization over a fixed-length receding horizon using a genetic algorithm with heuristic-based mutation, ensuring the vehicle's controls remain within its valid operational range. By addressing challenges posed by uneven terrain meshes, such as changing face normals, GAKD offers a practical solution for path planning in complex environments. Comparative evaluations against Model Predictive Path Integral (MPPI) and log-MPPI methods show that GAKD achieves up to 20 percent improvement in traversability cost while maintaining comparable path length. These results demonstrate GAKD's potential in improving vehicle navigation on challenging terrains. 

**Abstract (ZH)**: 基于遗传算法的 kinodynamic 规划算法 (GAKD)：汽车在三角网模型的不平地形导航轨迹优化 

---
# Autonomous Drone for Dynamic Smoke Plume Tracking 

**Title (ZH)**: 自主无人机动态烟柱追踪 

**Authors**: Srijan Kumar Pal, Shashank Sharma, Nikil Krishnakumar, Jiarong Hong  

**Link**: [PDF](https://arxiv.org/pdf/2504.12664)  

**Abstract**: This paper presents a novel autonomous drone-based smoke plume tracking system capable of navigating and tracking plumes in highly unsteady atmospheric conditions. The system integrates advanced hardware and software and a comprehensive simulation environment to ensure robust performance in controlled and real-world settings. The quadrotor, equipped with a high-resolution imaging system and an advanced onboard computing unit, performs precise maneuvers while accurately detecting and tracking dynamic smoke plumes under fluctuating conditions. Our software implements a two-phase flight operation, i.e., descending into the smoke plume upon detection and continuously monitoring the smoke movement during in-plume tracking. Leveraging Proportional Integral-Derivative (PID) control and a Proximal Policy Optimization based Deep Reinforcement Learning (DRL) controller enables adaptation to plume dynamics. Unreal Engine simulation evaluates performance under various smoke-wind scenarios, from steady flow to complex, unsteady fluctuations, showing that while the PID controller performs adequately in simpler scenarios, the DRL-based controller excels in more challenging environments. Field tests corroborate these findings. This system opens new possibilities for drone-based monitoring in areas like wildfire management and air quality assessment. The successful integration of DRL for real-time decision-making advances autonomous drone control for dynamic environments. 

**Abstract (ZH)**: 基于自主无人机的烟柱跟踪系统：在高度不稳定的气象条件下导航与跟踪 

---
# A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation 

**Title (ZH)**: A0: 一种考虑利用条件的分层模型 general robotic manipulation 

**Authors**: Rongtao Xu, Jian Zhang, Minghao Guo, Youpeng Wen, Haoting Yang, Min Lin, Jianzheng Huang, Zhe Li, Kaidong Zhang, Liqiong Wang, Yuxuan Kuang, Meng Cao, Feng Zheng, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12636)  

**Abstract**: Robotic manipulation faces critical challenges in understanding spatial affordances--the "where" and "how" of object interactions--essential for complex manipulation tasks like wiping a board or stacking objects. Existing methods, including modular-based and end-to-end approaches, often lack robust spatial reasoning capabilities. Unlike recent point-based and flow-based affordance methods that focus on dense spatial representations or trajectory modeling, we propose A0, a hierarchical affordance-aware diffusion model that decomposes manipulation tasks into high-level spatial affordance understanding and low-level action execution. A0 leverages the Embodiment-Agnostic Affordance Representation, which captures object-centric spatial affordances by predicting contact points and post-contact trajectories. A0 is pre-trained on 1 million contact points data and fine-tuned on annotated trajectories, enabling generalization across platforms. Key components include Position Offset Attention for motion-aware feature extraction and a Spatial Information Aggregation Layer for precise coordinate mapping. The model's output is executed by the action execution module. Experiments on multiple robotic systems (Franka, Kinova, Realman, and Dobot) demonstrate A0's superior performance in complex tasks, showcasing its efficiency, flexibility, and real-world applicability. 

**Abstract (ZH)**: 机器人操作在理解空间 afforded 性能方面面临关键挑战——这对于擦黑板或堆叠物体等复杂操作任务的“哪里”和“如何”至关重要。现有方法，包括模块化和端到端方法，通常缺乏 robust 的空间推理能力。有别于最近基于点和流的方法，这些方法集中在密集的空间表示或轨迹建模上，我们提出 A0，一种分层的认知操作扩散模型，将操作任务分解为高层次的空间 afforded 性理解与低层次的操作执行。A0 利用体无关的 afforded 性表示，通过预测接触点和接触后的轨迹来捕获以对象为中心的空间 afforded 性。A0 在 100 万接触点数据上进行了预训练，并在标注的轨迹上进行了微调，使其实现跨平台应用。关键组件包括位置偏移注意力机制，用于运动感知特征提取，以及空间信息聚合层，用于精确的坐标映射。模型的输出由操作执行模块执行。在多个机器人系统（Franka、Kinova、Realman 和 Dobot）上的实验表明，A0 在复杂任务中的性能优越，展示了其高效性、灵活性和实际应用性。 

---
# Graph-based Path Planning with Dynamic Obstacle Avoidance for Autonomous Parking 

**Title (ZH)**: 基于图的路径规划与动态障碍避障的自主停车 

**Authors**: Farhad Nawaz, Minjun Sung, Darshan Gadginmath, Jovin D'sa, Sangjae Bae, David Isele, Nadia Figueroa, Nikolai Matni, Faizan M. Tariq  

**Link**: [PDF](https://arxiv.org/pdf/2504.12616)  

**Abstract**: Safe and efficient path planning in parking scenarios presents a significant challenge due to the presence of cluttered environments filled with static and dynamic obstacles. To address this, we propose a novel and computationally efficient planning strategy that seamlessly integrates the predictions of dynamic obstacles into the planning process, ensuring the generation of collision-free paths. Our approach builds upon the conventional Hybrid A star algorithm by introducing a time-indexed variant that explicitly accounts for the predictions of dynamic obstacles during node exploration in the graph, thus enabling dynamic obstacle avoidance. We integrate the time-indexed Hybrid A star algorithm within an online planning framework to compute local paths at each planning step, guided by an adaptively chosen intermediate goal. The proposed method is validated in diverse parking scenarios, including perpendicular, angled, and parallel parking. Through simulations, we showcase our approach's potential in greatly improving the efficiency and safety when compared to the state of the art spline-based planning method for parking situations. 

**Abstract (ZH)**: 密 clustering 境环境下安全高效的道路规划面临着显著挑战，由于存在静态和动态障碍物。为了解决这一问题，我们提出了一种新颖且计算高效的规划策略，该策略将动态障碍物的预测无缝集成到规划过程中，确保生成无碰撞路径。我们的方法基于传统的Hybrid A*算法，并引入了一种时间索引变体，在图的节点探索过程中明确考虑动态障碍物的预测，从而实现动态障碍物的避让。我们将时间索引Hybrid A*算法集成到在线规划框架中，在每步规划时根据适应性选择的中间目标来计算局部路径。所提出的方法在垂直、斜角和平行停车等多种停车场景中得到了验证。通过仿真的方式，我们展示了该方法在效率和安全性方面较基于样条的最新规划方法的巨大改进潜力。 

---
# Practical Insights on Grasp Strategies for Mobile Manipulation in the Wild 

**Title (ZH)**: 移动操作中的野外抓取策略实用洞察 

**Authors**: Isabella Huang, Richard Cheng, Sangwoon Kim, Dan Kruse, Carolyn Matl, Lukas Kaul, JC Hancock, Shanmuga Harikumar, Mark Tjersland, James Borders, Dan Helmick  

**Link**: [PDF](https://arxiv.org/pdf/2504.12512)  

**Abstract**: Mobile manipulation robots are continuously advancing, with their grasping capabilities rapidly progressing. However, there are still significant gaps preventing state-of-the-art mobile manipulators from widespread real-world deployments, including their ability to reliably grasp items in unstructured environments. To help bridge this gap, we developed SHOPPER, a mobile manipulation robot platform designed to push the boundaries of reliable and generalizable grasp strategies. We develop these grasp strategies and deploy them in a real-world grocery store -- an exceptionally challenging setting chosen for its vast diversity of manipulable items, fixtures, and layouts. In this work, we present our detailed approach to designing general grasp strategies towards picking any item in a real grocery store. Additionally, we provide an in-depth analysis of our latest real-world field test, discussing key findings related to fundamental failure modes over hundreds of distinct pick attempts. Through our detailed analysis, we aim to offer valuable practical insights and identify key grasping challenges, which can guide the robotics community towards pressing open problems in the field. 

**Abstract (ZH)**: 移动 manipulation 机器人不断进步，其抓取能力迅速提升。然而，最先进的移动 manipulator 仍存在显著差距，限制了其在实际环境中的广泛应用，尤其是在不规则环境下的可靠抓取能力。为弥合这一差距，我们开发了 SHOPPER，一个旨在推动可靠且通用抓取策略边界的移动 manipulation 机器人平台。我们发展了这些抓取策略，并在真实的杂货店环境中部署——这是一个极具挑战性的环境，因其广泛的可操作物品、设备和布局多样性而被精心选择。在这项工作中，我们详细介绍了设计通用抓取策略的方法，以实现在真实杂货店中捡拾任何物品。此外，我们提供了我们最新实地测试的深入分析，讨论了数百次独立捡拾尝试中基本失败模式的关键发现。通过我们的详细分析，我们旨在提供有价值的实践洞察，并识别关键抓取挑战，从而指导机器人社区解决该领域迫切需要解决的开放问题。 

---
# Learning-based Delay Compensation for Enhanced Control of Assistive Soft Robots 

**Title (ZH)**: 基于学习的延迟补偿以增强辅助软机器人控制 

**Authors**: Adrià Mompó Alepuz, Dimitrios Papageorgiou, Silvia Tolu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12428)  

**Abstract**: Soft robots are increasingly used in healthcare, especially for assistive care, due to their inherent safety and adaptability. Controlling soft robots is challenging due to their nonlinear dynamics and the presence of time delays, especially in applications like a soft robotic arm for patient care. This paper presents a learning-based approach to approximate the nonlinear state predictor (Smith Predictor), aiming to improve tracking performance in a two-module soft robot arm with a short inherent input delay. The method uses Kernel Recursive Least Squares Tracker (KRLST) for online learning of the system dynamics and a Legendre Delay Network (LDN) to compress past input history for efficient delay compensation. Experimental results demonstrate significant improvement in tracking performance compared to a baseline model-based non-linear controller. Statistical analysis confirms the significance of the improvements. The method is computationally efficient and adaptable online, making it suitable for real-world scenarios and highlighting its potential for enabling safer and more accurate control of soft robots in assistive care applications. 

**Abstract (ZH)**: 软机器人在医疗领域的应用日益增多，尤其是在辅助护理方面，得益于其固有的安全性和适应性。控制软机器人颇具挑战性，主要由于其非线性动力学特性和存在的时延，尤其是在患者护理应用中，如软机器人臂。本文提出了一种基于学习的方法，以近似非线性状态预测器（Smith预测器），旨在提高具有较短固有时延的双模块软机器人臂的跟踪性能。该方法使用核递归最小二乘追踪器（KRLST）进行在线学习系统动力学，并使用勒让德延迟网络（LDN）压缩过去输入历史以实现高效的时延补偿。实验结果表明，与基于模型的非线性控制器基线相比，跟踪性能有了显著改善。统计分析证实了改进的显著性。该方法计算效率高且可在线适应，使其适用于现实场景，进一步凸显了其在辅助护理应用中实现更安全和更准确的软机器人控制的潜力。 

---
# Diffusion Based Robust LiDAR Place Recognition 

**Title (ZH)**: 基于扩散的鲁棒激光雷达地点识别 

**Authors**: Benjamin Krummenacher, Jonas Frey, Turcan Tuna, Olga Vysotska, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2504.12412)  

**Abstract**: Mobile robots on construction sites require accurate pose estimation to perform autonomous surveying and inspection missions. Localization in construction sites is a particularly challenging problem due to the presence of repetitive features such as flat plastered walls and perceptual aliasing due to apartments with similar layouts inter and intra floors. In this paper, we focus on the global re-positioning of a robot with respect to an accurate scanned mesh of the building solely using LiDAR data. In our approach, a neural network is trained on synthetic LiDAR point clouds generated by simulating a LiDAR in an accurate real-life large-scale mesh. We train a diffusion model with a PointNet++ backbone, which allows us to model multiple position candidates from a single LiDAR point cloud. The resulting model can successfully predict the global position of LiDAR in confined and complex sites despite the adverse effects of perceptual aliasing. The learned distribution of potential global positions can provide multi-modal position distribution. We evaluate our approach across five real-world datasets and show the place recognition accuracy of 77% +/-2m on average while outperforming baselines at a factor of 2 in mean error. 

**Abstract (ZH)**: 基于LiDAR数据的建筑工地机器人全局定位方法 

---
# AUTONAV: A Toolfor Autonomous Navigation of Robots 

**Title (ZH)**: AUTONAV：一种自主导航机器人工具 

**Authors**: Mir Md Sajid Sarwar, Sudip Samanta, Rajarshi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2504.12318)  

**Abstract**: We present a tool AUTONAV that automates the mapping, localization, and path-planning tasks for autonomous navigation of robots. The modular architecture allows easy integration of various algorithms for these tasks for comparison. We present the generated maps and path-plans by AUTONAV in indoor simulation scenarios. 

**Abstract (ZH)**: 我们提出了一种工具AUTONAV，该工具自动化了自主机器人导航的建图、定位和路径规划任务。模块化的架构允许轻松集成这些任务的各种算法以进行比较。我们在室内仿真场景中展示了AUTONAV生成的地图和路径规划。 

---
# Adaptive Task Space Non-Singular Terminal Super-Twisting Sliding Mode Control of a 7-DOF Robotic Manipulator 

**Title (ZH)**: 7-DOF机器人 manipulator 适应性任务空间非奇异终端超-twisting 滑模控制 

**Authors**: L. Wan, S. Smith, Y.-J. Pan, E. Witrant  

**Link**: [PDF](https://arxiv.org/pdf/2504.13056)  

**Abstract**: This paper presents a new task-space Non-singular Terminal Super-Twisting Sliding Mode (NT-STSM) controller with adaptive gains for robust trajectory tracking of a 7-DOF robotic manipulator. The proposed approach addresses the challenges of chattering, unknown disturbances, and rotational motion tracking, making it suited for high-DOF manipulators in dexterous manipulation tasks. A rigorous boundedness proof is provided, offering gain selection guidelines for practical implementation. Simulations and hardware experiments with external disturbances demonstrate the proposed controller's robust, accurate tracking with reduced control effort under unknown disturbances compared to other NT-STSM and conventional controllers. The results demonstrated that the proposed NT-STSM controller mitigates chattering and instability in complex motions, making it a viable solution for dexterous robotic manipulations and various industrial applications. 

**Abstract (ZH)**: 一种适用于7-DOF机器人 manipulator稳健轨迹跟踪的自适应增益非奇异终端超扭转滑模控制器 

---
# Acoustic Analysis of Uneven Blade Spacing and Toroidal Geometry for Reducing Propeller Annoyance 

**Title (ZH)**: 非均匀叶片间距和环形几何结构的 acoustic 分析以减少推进器恼人程度 

**Authors**: Nikhil Vijay, Will C. Forte, Ishan Gajjar, Sarvesh Patham, Syon Gupta, Sahil Shah, Prathamesh Trivedi, Rishit Arora  

**Link**: [PDF](https://arxiv.org/pdf/2504.12554)  

**Abstract**: Unmanned aerial vehicles (UAVs) are becoming more commonly used in populated areas, raising concerns about noise pollution generated from their propellers. This study investigates the acoustic performance of unconventional propeller designs, specifically toroidal and uneven-blade spaced propellers, for their potential in reducing psychoacoustic annoyance. Our experimental results show that these designs noticeably reduced acoustic characteristics associated with noise annoyance. 

**Abstract (ZH)**: 无人驾驶飞行器（UAV）在人口密集地区的应用日益增多，引起了对其推进器产生的噪声污染的关注。本研究探讨了非传统推进器设计，特别是环形和不等距叶片推进器的声学性能，评估其在降低心理声学烦恼方面的工作潜力。实验结果表明，这些设计显著降低了与噪声烦恼相关的声学特性。 

---
# Robust Visual Servoing under Human Supervision for Assembly Tasks 

**Title (ZH)**: 在人类监督下的鲁棒视觉伺服技术用于装配任务 

**Authors**: Victor Nan Fernandez-Ayala, Jorge Silva, Meng Guo, Dimos V. Dimarogonas  

**Link**: [PDF](https://arxiv.org/pdf/2504.12506)  

**Abstract**: We propose a framework enabling mobile manipulators to reliably complete pick-and-place tasks for assembling structures from construction blocks. The picking uses an eye-in-hand visual servoing controller for object tracking with Control Barrier Functions (CBFs) to ensure fiducial markers in the blocks remain visible. An additional robot with an eye-to-hand setup ensures precise placement, critical for structural stability. We integrate human-in-the-loop capabilities for flexibility and fault correction and analyze robustness to camera pose errors, proposing adapted barrier functions to handle them. Lastly, experiments validate the framework on 6-DoF mobile arms. 

**Abstract (ZH)**: 我们提出了一种框架，使移动 manipulator 能够可靠地完成基于构造块组装结构的取放任务。取件采用眼手视觉伺服控制器，并使用控制障碍函数（CBFs）确保构造块上的特征标记保持可见。另外一台具有眼手配置的机器人确保精确放置，这对于结构稳定性至关重要。我们整合了人为环路功能以提高灵活性和故障纠正能力，并分析了对于相机姿态误差的鲁棒性，提出了相应的障碍函数来处理这些问题。最后，实验在6-DoF移动臂上验证了该框架。 

---
