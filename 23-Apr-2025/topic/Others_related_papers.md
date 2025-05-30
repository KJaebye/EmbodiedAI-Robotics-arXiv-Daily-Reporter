# SAR4SLPs: An Asynchronous Survey of Speech-Language Pathologists' Perspectives on Socially Assistive Robots 

**Title (ZH)**: SAR4SLPs: 社交辅助机器人领域言语语言病理学家观点的异步综述 

**Authors**: Denielle Oliva, Abbie Olszewski, David Feil-Seifer  

**Link**: [PDF](https://arxiv.org/pdf/2504.16055)  

**Abstract**: Socially Assistive Robots (SARs) offer unique opportunities within speech language pathology (SLP) education and practice by supporting interactive interventions for children with communication disorders. This paper explores the implementation of SAR4SLPs (Socially Assistive Robots for Speech-Language Pathologists) to investigate aspects such as engagement, therapeutic strategy discipline, and consistent intervention support. We assessed the current application of technology to clinical and educational settings, especially with respect to how SLPs might use SAR in their therapeutic work. An asynchronous remote community (ARC) collaborated with a cohort of practicing SLPs to consider the feasibility, potential effectiveness, and anticipated challenges with implementing SARs in day-to-day interventions and as practice facilitators. We focus in particular on the expressive functionality of SARs, modeling a foundational strategy that SLPs employ across various intervention targets. This paper highlights clinician-driven insights and design implications for developing SARs that support specific treatment goals through collaborative and iterative design. 

**Abstract (ZH)**: 社会辅助机器人（SARs）在言语语言病理学（SLP）教育和实践中提供了独特的机会，通过支持针对沟通障碍儿童的互动干预措施。本文探讨了SAR4SLPs（社会辅助机器人用于言语语言病理学家）的实施，以研究诸如参与度、治疗策略一致性和持续干预支持等方面。我们评估了技术在临床和教育环境中的当前应用，特别是考虑到言语语言病理学家如何在治疗工作中使用SAR。异步远程社区（ARC）与一组实践中的言语语言病理学家合作，考虑在日常干预中实施SAR以及作为实践促进者的可行性和潜在挑战。本文特别关注SAR的表达功能，模拟了言语语言病理学家在各种干预目标中应用的基础策略。本文强调了临床驱动的观点和设计启示，以促进支持特定治疗目标的SAR的协作和迭代设计。 

---
# ad-trait: A Fast and Flexible Automatic Differentiation Library in Rust 

**Title (ZH)**: ad-trait: 一种快速灵活的自动微分库（Rust 语言实现） 

**Authors**: Chen Liang, Qian Wang, Andy Xu, Daniel Rakita  

**Link**: [PDF](https://arxiv.org/pdf/2504.15976)  

**Abstract**: The Rust programming language is an attractive choice for robotics and related fields, offering highly efficient and memory-safe code. However, a key limitation preventing its broader adoption in these domains is the lack of high-quality, well-supported Automatic Differentiation (AD)-a fundamental technique that enables convenient derivative computation by systematically accumulating data during function evaluation. In this work, we introduce ad-trait, a new Rust-based AD library. Our implementation overloads Rust's standard floating-point type with a flexible trait that can efficiently accumulate necessary information for derivative computation. The library supports both forward-mode and reverse-mode automatic differentiation, making it the first operator-overloading AD implementation in Rust to offer both options. Additionally, ad-trait leverages Rust's performance-oriented features, such as Single Instruction, Multiple Data acceleration in forward-mode AD, to enhance efficiency. Through benchmarking experiments, we show that our library is among the fastest AD implementations across several programming languages for computing derivatives. Moreover, it is already integrated into a Rust-based robotics library, where we showcase its ability to facilitate fast optimization procedures. We conclude with a discussion of the limitations and broader implications of our work. 

**Abstract (ZH)**: Rust编程语言在机器人技术及相关领域中的应用选择：一种高效且内存安全的自动求导库 

---
# RaSCL: Radar to Satellite Crossview Localization 

**Title (ZH)**: 雷达到卫星跨视角定位 

**Authors**: Blerim Abdullai, Tony Wang, Xinyuan Qiao, Florian Shkurti, Timothy D. Barfoot  

**Link**: [PDF](https://arxiv.org/pdf/2504.15899)  

**Abstract**: GNSS is unreliable, inaccurate, and insufficient in many real-time autonomous field applications. In this work, we present a GNSS-free global localization solution that contains a method of registering imaging radar on the ground with overhead RGB imagery, with joint optimization of relative poses from odometry and global poses from our overhead registration. Previous works have used various combinations of ground sensors and overhead imagery, and different feature extraction and matching methods. These include various handcrafted and deep-learning-based methods for extracting features from overhead imagery. Our work presents insights on extracting essential features from RGB overhead images for effective global localization against overhead imagery using only ground radar and a single georeferenced initial guess. We motivate our method by evaluating it on datasets in diverse geographic conditions and robotic platforms, including on an Unmanned Surface Vessel (USV) as well as urban and suburban driving datasets. 

**Abstract (ZH)**: GNSS在许多实时自主野外应用中不可靠、不准确且不足。在此工作中，我们提出了一种无需GNSS的全球定位解决方案，该方案包含一种将地面成像雷达与空中RGB图像进行注册的方法，并结合了来自传感器里程计的相对姿态优化和来自我们空中注册的绝对姿态优化。之前的研究所使用了各种地面传感器和空中图像的组合，以及不同的特征提取和匹配方法。这些方法包括用于从空中图像中提取特征的各种手工制作和基于深度学习的方法。我们的研究探讨了仅使用地面雷达和单个地理参考初始猜测，从RGB空中图像中提取关键特征以实现有效全球定位的方法。我们通过对在多种地理条件和机器人平台上的数据集（包括无人水面舰艇USV以及城市和郊区驾驶数据集）进行评估来激励我们的方法。 

---
# An Extended Horizon Tactical Decision-Making for Automated Driving Based on Monte Carlo Tree Search 

**Title (ZH)**: 基于蒙特卡洛树搜索的扩展视野战术决策方法在自动驾驶中的应用 

**Authors**: Karim Essalmi, Fernando Garrido, Fawzi Nashashibi  

**Link**: [PDF](https://arxiv.org/pdf/2504.15869)  

**Abstract**: This paper introduces COR-MCTS (Conservation of Resources - Monte Carlo Tree Search), a novel tactical decision-making approach for automated driving focusing on maneuver planning over extended horizons. Traditional decision-making algorithms are often constrained by fixed planning horizons, typically up to 6 seconds for classical approaches and 3 seconds for learning-based methods limiting their adaptability in particular dynamic driving scenarios. However, planning must be done well in advance in environments such as highways, roundabouts, and exits to ensure safe and efficient maneuvers. To address this challenge, we propose a hybrid method integrating Monte Carlo Tree Search (MCTS) with our prior utility-based framework, COR-MP (Conservation of Resources Model for Maneuver Planning). This combination enables long-term, real-time decision-making, significantly enhancing the ability to plan a sequence of maneuvers over extended horizons. Through simulations across diverse driving scenarios, we demonstrate that COR-MCTS effectively improves planning robustness and decision efficiency over extended horizons. 

**Abstract (ZH)**: COR-MCTS：资源保存-蒙特卡洛树搜索在长期内存机动规划中的新型自动化驾驶战术决策方法 

---
# Dynamic Intent Queries for Motion Transformer-based Trajectory Prediction 

**Title (ZH)**: 基于运动变换器的时间预测中的动态意图查询 

**Authors**: Tobias Demmler, Lennart Hartung, Andreas Tamke, Thao Dang, Alexander Hegai, Karsten Haug, Lars Mikelsons  

**Link**: [PDF](https://arxiv.org/pdf/2504.15766)  

**Abstract**: In autonomous driving, accurately predicting the movements of other traffic participants is crucial, as it significantly influences a vehicle's planning processes. Modern trajectory prediction models strive to interpret complex patterns and dependencies from agent and map data. The Motion Transformer (MTR) architecture and subsequent work define the most accurate methods in common benchmarks such as the Waymo Open Motion Benchmark. The MTR model employs pre-generated static intention points as initial goal points for trajectory prediction. However, the static nature of these points frequently leads to misalignment with map data in specific traffic scenarios, resulting in unfeasible or unrealistic goal points. Our research addresses this limitation by integrating scene-specific dynamic intention points into the MTR model. This adaptation of the MTR model was trained and evaluated on the Waymo Open Motion Dataset. Our findings demonstrate that incorporating dynamic intention points has a significant positive impact on trajectory prediction accuracy, especially for predictions over long time horizons. Furthermore, we analyze the impact on ground truth trajectories which are not compliant with the map data or are illegal maneuvers. 

**Abstract (ZH)**: 在自主驾驶中，准确预测其他交通参与者的运动至关重要，因为它显著影响车辆的规划过程。现代轨迹预测模型致力于从代理和地图数据中解析复杂的模式和依赖关系。Motion Transformer（MTR）架构及其后续工作在Waymo公开轨迹预测基准等常见基准中定义了最精确的方法。MTR模型使用预先生成的静态意向点作为轨迹预测的初始目标点。然而，这些点的静态性质在特定交通场景中往往会与地图数据产生对齐问题，导致不实际或不现实的目标点。我们的研究通过将场景特定的动态意向点集成到MTR模型中，以解决这一局限性。该MTR模型经过Waymo公开轨迹数据集的训练和评估，实验结果表明，纳入动态意向点显著提高了轨迹预测精度，尤其是对于长期预测的影响更为显著。此外，我们分析了不遵守地图数据或不合法的行为的真实轨迹的影响。 

---
# RiskNet: Interaction-Aware Risk Forecasting for Autonomous Driving in Long-Tail Scenarios 

**Title (ZH)**: RiskNet：长尾场景下考虑交互的风险预报在自动驾驶中的应用 

**Authors**: Qichao Liu, Heye Huang, Shiyue Zhao, Lei Shi, Soyoung Ahn, Xiaopeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.15541)  

**Abstract**: Ensuring the safety of autonomous vehicles (AVs) in long-tail scenarios remains a critical challenge, particularly under high uncertainty and complex multi-agent interactions. To address this, we propose RiskNet, an interaction-aware risk forecasting framework, which integrates deterministic risk modeling with probabilistic behavior prediction for comprehensive risk assessment. At its core, RiskNet employs a field-theoretic model that captures interactions among ego vehicle, surrounding agents, and infrastructure via interaction fields and force. This model supports multidimensional risk evaluation across diverse scenarios (highways, intersections, and roundabouts), and shows robustness under high-risk and long-tail settings. To capture the behavioral uncertainty, we incorporate a graph neural network (GNN)-based trajectory prediction module, which learns multi-modal future motion distributions. Coupled with the deterministic risk field, it enables dynamic, probabilistic risk inference across time, enabling proactive safety assessment under uncertainty. Evaluations on the highD, inD, and rounD datasets, spanning lane changes, turns, and complex merges, demonstrate that our method significantly outperforms traditional approaches (e.g., TTC, THW, RSS, NC Field) in terms of accuracy, responsiveness, and directional sensitivity, while maintaining strong generalization across scenarios. This framework supports real-time, scenario-adaptive risk forecasting and demonstrates strong generalization across uncertain driving environments. It offers a unified foundation for safety-critical decision-making in long-tail scenarios. 

**Abstract (ZH)**: 确保自动驾驶车辆在长尾场景下的安全性仍然是一个关键挑战，特别是在高不确定性和复杂多代理交互情境下。为此，我们提出RiskNet，一种交互感知的风险预测框架，该框架结合确定性风险建模与概率行为预测，进行全面风险评估。RiskNet的核心是一种场理论模型，通过交互场和力捕捉ego车辆、周围代理和基础设施之间的交互。该模型支持在不同场景（高速公路、交点和环岛）下的多维度风险评估，并在高风险和长尾情境下表现 robust。为了捕捉行为不确定性，我们引入了一种基于图神经网络（GNN）的轨迹预测模块，该模块学习多模态未来运动分布。结合确定性风险场，它能够实现随时间的动态、概率性风险推断，从而在不确定性下进行主动安全评估。在highD、inD和rounD数据集上的评估，涵盖了变道、转弯和复杂合并场景，证明了本方法在准确性、响应性和方向敏感性方面显著优于传统方法（如TTC、THW、RSS、NC Field），同时在不同场景下具有强大的泛化能力。该框架支持实时、场景自适应的风险预测，并在不确定驾驶环境中表现出强大的泛化能力。它为长尾场景下的安全关键决策提供了一个统一的基础。 

---
# Solving Multi-Agent Safe Optimal Control with Distributed Epigraph Form MARL 

**Title (ZH)**: 分布式上图形表示多agent安全最优控制 

**Authors**: Songyuan Zhang, Oswin So, Mitchell Black, Zachary Serlin, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15425)  

**Abstract**: Tasks for multi-robot systems often require the robots to collaborate and complete a team goal while maintaining safety. This problem is usually formalized as a constrained Markov decision process (CMDP), which targets minimizing a global cost and bringing the mean of constraint violation below a user-defined threshold. Inspired by real-world robotic applications, we define safety as zero constraint violation. While many safe multi-agent reinforcement learning (MARL) algorithms have been proposed to solve CMDPs, these algorithms suffer from unstable training in this setting. To tackle this, we use the epigraph form for constrained optimization to improve training stability and prove that the centralized epigraph form problem can be solved in a distributed fashion by each agent. This results in a novel centralized training distributed execution MARL algorithm named Def-MARL. Simulation experiments on 8 different tasks across 2 different simulators show that Def-MARL achieves the best overall performance, satisfies safety constraints, and maintains stable training. Real-world hardware experiments on Crazyflie quadcopters demonstrate the ability of Def-MARL to safely coordinate agents to complete complex collaborative tasks compared to other methods. 

**Abstract (ZH)**: 多机器人系统的任务往往要求机器人协作完成团队目标的同时保持安全。这个问题通常被形式化为约束马尔可夫决策过程（CMDP），其目标是最小化全局成本并将约束违背的均值保持在用户定义的阈值之下。受到现实机器人应用的启发，我们将安全定义为零约束违背。尽管已经提出了许多安全多智能体强化学习（MARL）算法来解决CMDP，但这些算法在这个设置中训练不稳定。为解决这一问题，我们使用约束优化的epigraph形式改进训练稳定性，并证明中心化epigraph形式的问题可以通过每个智能体分布式解决。这导致了一个名为Def-MARL的新颖中央训练分布式执行MARL算法。在两个不同模拟器上的8个不同任务的仿真实验表明，Def-MARL在总体性能、满足安全约束以及保持训练稳定性方面表现最佳。在Crazyflie四旋翼飞行器上的真实硬件实验表明，与其它方法相比，Def-MARL能够安全协调智能体完成复杂协作任务。 

---
# MRTA-Sim: A Modular Simulator for Multi-Robot Allocation, Planning, and Control in Open-World Environments 

**Title (ZH)**: MRTA-Sim: 一种适用于开放环境多机器人分配、规划与控制的模块化模拟器 

**Authors**: Victoria Marie Tuck, Hardik Parwana, Pei-Wei Chen, Georgios Fainekos, Bardh Hoxha, Hideki Okamoto, S. Shankar Sastry, Sanjit A. Seshia  

**Link**: [PDF](https://arxiv.org/pdf/2504.15418)  

**Abstract**: This paper introduces MRTA-Sim, a Python/ROS2/Gazebo simulator for testing approaches to Multi-Robot Task Allocation (MRTA) problems on simulated robots in complex, indoor environments. Grid-based approaches to MRTA problems can be too restrictive for use in complex, dynamic environments such in warehouses, department stores, hospitals, etc. However, approaches that operate in free-space often operate at a layer of abstraction above the control and planning layers of a robot and make an assumption on approximate travel time between points of interest in the system. These abstractions can neglect the impact of the tight space and multi-agent interactions on the quality of the solution. Therefore, MRTA solutions should be tested with the navigation stacks of the robots in mind, taking into account robot planning, conflict avoidance between robots, and human interaction and avoidance. This tool connects the allocation output of MRTA solvers to individual robot planning using the NAV2 stack and local, centralized multi-robot deconfliction using Control Barrier Function-Quadrtic Programs (CBF-QPs), creating a platform closer to real-world operation for more comprehensive testing of these approaches. The simulation architecture is modular so that users can swap out methods at different levels of the stack. We show the use of our system with a Satisfiability Modulo Theories (SMT)-based approach to dynamic MRTA on a fleet of indoor delivery robots. 

**Abstract (ZH)**: MRTA-Sim：一种用于复杂室内环境中基于Python/ROS2/Gazebo的多机器人任务分配测试模拟器 

---
# Nearly Optimal Nonlinear Safe Control with BaS-SDRE 

**Title (ZH)**: 几乎最优的非线性安全控制：BaS-SDRE方法 

**Authors**: Hassan Almubarak, Maitham F. AL-Sunni, Justin T. Dubbin, Nader Sadegh, John M. Dolan, Evangelos A. Theodorou  

**Link**: [PDF](https://arxiv.org/pdf/2504.15453)  

**Abstract**: The State-Dependent Riccati Equation (SDRE) approach has emerged as a systematic and effective means of designing nearly optimal nonlinear controllers. The Barrier States (BaS) embedding methodology was developed recently for safe multi-objective controls in which the safety condition is manifested as a state to be controlled along with other states of the system. The overall system, termed the safety embedded system, is highly nonlinear even if the original system is linear. This paper develops a nonlinear nearly optimal safe feedback control technique by combining the two strategies effectively. First, the BaS is derived in an extended linearization formulation to be subsequently used to form an extended safety embedded system. A new optimal control problem is formed thereafter, which is used to construct a safety embedded State-Dependent Riccati Equation, termed BaS-SDRE, whose solution approximates the solution of the optimal control problem's associated Hamilton-Jacobi-Bellman (HJB) equation. The BaS-SDRE is then solved online to synthesize the nearly optimal safe control. The proposed technique's efficacy is demonstrated on an unstable, constrained linear system that shows how the synthesized control reacts to nonlinearities near the unsafe region, a nonlinear flight control system with limited path angular velocity that exists due to structural and dynamic concerns, and a planar quadrotor system that navigates safely in a crowded environment. 

**Abstract (ZH)**: 基于Barrier States的State-Dependent Riccati方程方法在安全非线性控制设计中的应用 

---
# Safety Embedded Adaptive Control Using Barrier States 

**Title (ZH)**: 嵌入安全的自适应控制基于障碍状态 

**Authors**: Maitham F. AL-Sunni, Hassan Almubarak, John M. Dolan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15423)  

**Abstract**: In this work, we explore the application of barrier states (BaS) in the realm of safe nonlinear adaptive control. Our proposed framework derives barrier states for systems with parametric uncertainty, which are augmented into the uncertain dynamical model. We employ an adaptive nonlinear control strategy based on a control Lyapunov functions approach to design a stabilizing controller for the augmented system. The developed theory shows that the controller ensures safe control actions for the original system while meeting specified performance objectives. We validate the effectiveness of our approach through simulations on diverse systems, including a planar quadrotor subject to unknown drag forces and an adaptive cruise control system, for which we provide comparisons with existing methodologies. 

**Abstract (ZH)**: 本研究探讨了屏障状态在安全非线性自适应控制领域的应用。我们提出的框架为具有参数不确定性系统的屏障状态进行推导，并将其扩充到不确定动态模型中。基于控制李雅普诺夫函数的方法，我们设计了稳定控制器以控制扩充后的系统。所发展的理论表明，控制器能够确保对原始系统进行安全控制操作，同时满足指定的性能目标。通过在多种系统上进行仿真，包括受未知阻力影响的平面四旋翼飞行器和自适应巡航控制系统，验证了该方法的有效性，并提供了与现有方法的比较。 

---
# Vision6D: 3D-to-2D Interactive Visualization and Annotation Tool for 6D Pose Estimation 

**Title (ZH)**: Vision6D：用于6D姿态估计的3D到2D交互可视化和注释工具 

**Authors**: Yike Zhang, Eduardo Davalos, Jack Noble  

**Link**: [PDF](https://arxiv.org/pdf/2504.15329)  

**Abstract**: Accurate 6D pose estimation has gained more attention over the years for robotics-assisted tasks that require precise interaction with physical objects. This paper presents an interactive 3D-to-2D visualization and annotation tool to support the 6D pose estimation research community. To the best of our knowledge, the proposed work is the first tool that allows users to visualize and manipulate 3D objects interactively on a 2D real-world scene, along with a comprehensive user study. This system supports robust 6D camera pose annotation by providing both visual cues and spatial relationships to determine object position and orientation in various environments. The annotation feature in Vision6D is particularly helpful in scenarios where the transformation matrix between the camera and world objects is unknown, as it enables accurate annotation of these objects' poses using only the camera intrinsic matrix. This capability serves as a foundational step in developing and training advanced pose estimation models across various domains. We evaluate Vision6D's effectiveness by utilizing widely-used open-source pose estimation datasets Linemod and HANDAL through comparisons between the default ground-truth camera poses with manual annotations. A user study was performed to show that Vision6D generates accurate pose annotations via visual cues in an intuitive 3D user interface. This approach aims to bridge the gap between 2D scene projections and 3D scenes, offering an effective way for researchers and developers to solve 6D pose annotation related problems. The software is open-source and publicly available at this https URL. 

**Abstract (ZH)**: 准确的6D姿态估计在辅助机器人任务中日益受到关注，本论文介绍了一种交互式的3D到2D可视化和标注工具，以支持6D姿态估计研究社区。据我们所知，本工作中首次提出了允许用户在真实世界的2D场景中交互地可视化和操作3D对象的工具，并结合了全面的用户研究。该系统通过提供视觉提示和空间关系，支持在各种环境中稳健地标注6D相机姿态。Vision6D中的标注功能特别适用于相机与世界对象之间的转换矩阵未知的场景，通过仅使用相机内参矩阵即可准确标注这些对象的姿态。这一能力为跨各领域开发和训练高级姿态估计模型奠定了基础。我们通过与广泛使用的开源姿态估计数据集Linemod和HANDAL进行比较，评估了Vision6D的有效性。用户研究显示，Vision6D能够通过直观的3D用户界面中的视觉提示生成准确的姿态标注。该方法旨在弥合2D场景投影与3D场景之间的差距，为研究人员和开发人员提供解决6D姿态标注问题的有效途径。该软件开源并可在以下链接访问：this https URL。 

---
# Approximate matrices of systems of max-min fuzzy relational equations 

**Title (ZH)**: max-最小模糊关系方程的近似矩阵 

**Authors**: Ismaïl Baaj  

**Link**: [PDF](https://arxiv.org/pdf/2504.16042)  

**Abstract**: In this article, we address the inconsistency of a system of max-min fuzzy relational equations by minimally modifying the matrix governing the system in order to achieve consistency. Our method yields consistent systems that approximate the original inconsistent system in the following sense: the right-hand side vector of each consistent system is that of the inconsistent system, and the coefficients of the matrix governing each consistent system are obtained by modifying, exactly and minimally, the entries of the original matrix that must be corrected to achieve consistency, while leaving all other entries unchanged.
To obtain a consistent system that closely approximates the considered inconsistent system, we study the distance (in terms of a norm among $L_1$, $L_2$ or $L_\infty$) between the matrix of the inconsistent system and the set formed by the matrices of consistent systems that use the same right-hand side vector as the inconsistent system. We show that our method allows us to directly compute matrices of consistent systems that use the same right-hand side vector as the inconsistent system whose distance in terms of $L_\infty$ norm to the matrix of the inconsistent system is minimal (the computational costs are higher when using $L_1$ norm or $L_2$ norm). We also give an explicit analytical formula for computing this minimal $L_\infty$ distance. Finally, we translate our results for systems of min-max fuzzy relational equations and present some potential applications. 

**Abstract (ZH)**: 这篇文章通过最小修改调控系统的矩阵来解决max-min模糊关系方程系统的不一致性，以实现系统的一致性。我们的方法在如下意义上产生一致系统：每个一致系统的右侧向量与不一致系统的右侧向量相同，每个一致系统调控矩阵的系数是通过精确且最小地修改必须纠正的一致性条件的原矩阵元素获得的，而保持其他元素不变。

为了获得一个与考虑的不一致系统密切相关的一致系统，我们研究了不一致系统的矩阵与使用与不一致系统相同的右侧向量的一致系统矩阵集合之间的距离（以$L_1$、$L_2$或$L_\infty$范数的形式）。我们证明了我们的方法使得可以直接计算与不一致系统矩阵在$L_\infty$范数意义下距离最小的一致系统矩阵（使用$L_1$范数或$L_2$范数时计算成本更高）。我们还给出了计算此最小$L_\infty$距离的显式解析公式。最后，我们将结果应用于min-max模糊关系方程系统，并介绍一些潜在应用。 

---
# CARE: Compatibility-Aware Incentive Mechanisms for Federated Learning with Budgeted Requesters 

**Title (ZH)**: 兼容性 Awareness 在预算限制请求者参与的联邦学习激励机制中 

**Authors**: Xiang Liu, Hau Chan, Minming Li, Xianlong Zeng, Chenchen Fu, Weiwei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15847)  

**Abstract**: Federated learning (FL) is a promising approach that allows requesters (\eg, servers) to obtain local training models from workers (e.g., clients). Since workers are typically unwilling to provide training services/models freely and voluntarily, many incentive mechanisms in FL are designed to incentivize participation by offering monetary rewards from requesters. However, existing studies neglect two crucial aspects of real-world FL scenarios. First, workers can possess inherent incompatibility characteristics (e.g., communication channels and data sources), which can lead to degradation of FL efficiency (e.g., low communication efficiency and poor model generalization). Second, the requesters are budgeted, which limits the amount of workers they can hire for their tasks. In this paper, we investigate the scenario in FL where multiple budgeted requesters seek training services from incompatible workers with private training costs. We consider two settings: the cooperative budget setting where requesters cooperate to pool their budgets to improve their overall utility and the non-cooperative budget setting where each requester optimizes their utility within their own budgets. To address efficiency degradation caused by worker incompatibility, we develop novel compatibility-aware incentive mechanisms, CARE-CO and CARE-NO, for both settings to elicit true private costs and determine workers to hire for requesters and their rewards while satisfying requester budget constraints. Our mechanisms guarantee individual rationality, truthfulness, budget feasibility, and approximation performance. We conduct extensive experiments using real-world datasets to show that the proposed mechanisms significantly outperform existing baselines. 

**Abstract (ZH)**: 联邦学习中预算受限请求者与具有私有训练成本的不兼容工作者的激励机制研究 

---
# Generative AI for Research Data Processing: Lessons Learnt From Three Use Cases 

**Title (ZH)**: 生成式AI在科研数据处理中的应用：来自三个案例的研究心得 

**Authors**: Modhurita Mitra, Martine G. de Vos, Nicola Cortinovis, Dawa Ometto  

**Link**: [PDF](https://arxiv.org/pdf/2504.15829)  

**Abstract**: There has been enormous interest in generative AI since ChatGPT was launched in 2022. However, there are concerns about the accuracy and consistency of the outputs of generative AI. We have carried out an exploratory study on the application of this new technology in research data processing. We identified tasks for which rule-based or traditional machine learning approaches were difficult to apply, and then performed these tasks using generative AI.
We demonstrate the feasibility of using the generative AI model Claude 3 Opus in three research projects involving complex data processing tasks:
1) Information extraction: We extract plant species names from historical seedlists (catalogues of seeds) published by botanical gardens.
2) Natural language understanding: We extract certain data points (name of drug, name of health indication, relative effectiveness, cost-effectiveness, etc.) from documents published by Health Technology Assessment organisations in the EU.
3) Text classification: We assign industry codes to projects on the crowdfunding website Kickstarter.
We share the lessons we learnt from these use cases: How to determine if generative AI is an appropriate tool for a given data processing task, and if so, how to maximise the accuracy and consistency of the results obtained. 

**Abstract (ZH)**: 自2022年ChatGPT推出以来，生成式AI引起了巨大关注。然而，人们对生成式AI输出的准确性和一致性存在担忧。我们开展了探索性研究，探讨该新技术在科研数据分析中的应用。我们确定了传统基于规则或传统机器学习方法难以适用的任务，然后使用生成式AI完成这些任务。

我们在三个涉及复杂数据处理任务的科研项目中展示了使用生成式AI模型Claude 3 Opus的可行性：
1) 信息抽取：从植物园发布的种子列表（种子目录）中提取植物物种名称。
2) 自然语言理解：从欧盟卫生技术评估机构发布的文档中提取特定数据点（药物名称、健康影响名称、相对有效性、成本效益等）。
3) 文本分类：为在众筹网站Kickstarter上的项目分配行业代码。

我们分享了从这些案例中学到的教训：如何确定生成式AI是否是给定数据处理任务的合适工具，如果是，如何最大化结果的准确性和一致性。 

---
# Crisp complexity of fuzzy classifiers 

**Title (ZH)**: 模糊分类器的清晰复杂度 

**Authors**: Raquel Fernandez-Peralta, Javier Fumanal-Idocin, Javier Andreu-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2504.15791)  

**Abstract**: Rule-based systems are a very popular form of explainable AI, particularly in the fuzzy community, where fuzzy rules are widely used for control and classification problems. However, fuzzy rule-based classifiers struggle to reach bigger traction outside of fuzzy venues, because users sometimes do not know about fuzzy and because fuzzy partitions are not so easy to interpret in some situations. In this work, we propose a methodology to reduce fuzzy rule-based classifiers to crisp rule-based classifiers. We study different possible crisp descriptions and implement an algorithm to obtain them. Also, we analyze the complexity of the resulting crisp classifiers. We believe that our results can help both fuzzy and non-fuzzy practitioners understand better the way in which fuzzy rule bases partition the feature space and how easily one system can be translated to another and vice versa. Our complexity metric can also help to choose between different fuzzy classifiers based on what the equivalent crisp partitions look like. 

**Abstract (ZH)**: 基于规则的系统是一种非常流行的可解释人工智能形式，特别是在模糊社区，模糊规则广泛用于控制和分类问题。然而，模糊规则基于的分类器很难在模糊领域之外获得更大的关注度，因为有时用户不了解模糊概念，且在某些情况下模糊分区不易解释。本文提出了一种将模糊规则基于的分类器简化为清晰规则基于的分类器的方法。我们研究了不同的清晰描述形式，并实现了一个算法来获取它们。我们还分析了所得到的清晰分类器的复杂性。我们相信，我们的结果有助于模糊和非模糊从业人员更好地理解模糊规则基如何划分特征空间，以及一个系统如何容易地转换为另一个及其逆向过程。我们提出的复杂性度量也可以帮助根据等效清晰分区的形式来选择不同的模糊分类器。 

---
# TrustGeoGen: Scalable and Formal-Verified Data Engine for Trustworthy Multi-modal Geometric Problem Solving 

**Title (ZH)**: TrustGeoGen：可扩展且形式化验证的数据引擎，用于可信的多模态几何问题求解 

**Authors**: Daocheng Fu, Zijun Chen, Renqiu Xia, Qi Liu, Yuan Feng, Hongbin Zhou, Renrui Zhang, Shiyang Feng, Peng Gao, Junchi Yan, Botian Shi, Bo Zhang, Yu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15780)  

**Abstract**: Mathematical geometric problem solving (GPS) often requires effective integration of multimodal information and verifiable logical coherence. Despite the fast development of large language models in general problem solving, it remains unresolved regarding with both methodology and benchmarks, especially given the fact that exiting synthetic GPS benchmarks are often not self-verified and contain noise and self-contradicted information due to the illusion of LLMs. In this paper, we propose a scalable data engine called TrustGeoGen for problem generation, with formal verification to provide a principled benchmark, which we believe lays the foundation for the further development of methods for GPS. The engine synthesizes geometric data through four key innovations: 1) multimodal-aligned generation of diagrams, textual descriptions, and stepwise solutions; 2) formal verification ensuring rule-compliant reasoning paths; 3) a bootstrapping mechanism enabling complexity escalation via recursive state generation and 4) our devised GeoExplore series algorithms simultaneously produce multi-solution variants and self-reflective backtracking traces. By formal logical verification, TrustGeoGen produces GeoTrust-200K dataset with guaranteed modality integrity, along with GeoTrust-test testset. Experiments reveal the state-of-the-art models achieve only 49.17\% accuracy on GeoTrust-test, demonstrating its evaluation stringency. Crucially, models trained on GeoTrust achieve OOD generalization on GeoQA, significantly reducing logical inconsistencies relative to pseudo-label annotated by OpenAI-o1. Our code is available at this https URL 

**Abstract (ZH)**: 数学几何问题求解（GPS）通常要求有效整合多种模态信息和可验证的逻辑一致性。尽管大型语言模型在一般问题求解方面取得了快速进展，但在方法和基准方面仍存在未解决的问题，特别是在现有的合成GPS基准往往不具备自我验证性、含有噪声和矛盾信息的情况下。本文提出了一种可扩展的数据引擎TrustGeoGen用于问题生成，并通过形式验证提供一个规范化的基准，我们认为这为GPS方法的进一步发展奠定了基础。该引擎通过四项关键创新综合几何数据：1）多模态对齐的图示、文本描述和逐步解决方案生成；2）形式验证确保规则遵循的推理路径；3）一个递归状态生成的自增强机制；4）我们设计的GeoExplore系列算法同时生成多解变体和自我反思回溯轨迹。通过形式逻辑验证，TrustGeoGen生成了包含保证模态完整性的GeoTrust-200K数据集及其测试集GeoTrust-test。实验结果表明，最先进的模型在GeoTrust-test上仅能达到49.17%的准确率，证明了其评价的严格性。 crucial 地，基于GeoTrust训练的模型在GeoQA上的OOD泛化显著降低了逻辑不一致性，优于OpenAI-o1伪标签标注。我们的代码可在以下网址获取。 

---
# Exploring Inevitable Waypoints for Unsolvability Explanation in Hybrid Planning Problems 

**Title (ZH)**: 探索混合规划问题中不可解性解释的必然中间点 

**Authors**: Mir Md Sajid Sarwar, Rajarshi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2504.15668)  

**Abstract**: Explaining unsolvability of planning problems is of significant research interest in Explainable AI Planning. AI planning literature has reported several research efforts on generating explanations of solutions to planning problems. However, explaining the unsolvability of planning problems remains a largely open and understudied problem. A widely practiced approach to plan generation and automated problem solving, in general, is to decompose tasks into sub-problems that help progressively converge towards the goal. In this paper, we propose to adopt the same philosophy of sub-problem identification as a mechanism for analyzing and explaining unsolvability of planning problems in hybrid systems. In particular, for a given unsolvable planning problem, we propose to identify common waypoints, which are universal obstacles to plan existence; in other words, they appear on every plan from the source to the planning goal. This work envisions such waypoints as sub-problems of the planning problem and the unreachability of any of these waypoints as an explanation for the unsolvability of the original planning problem. We propose a novel method of waypoint identification by casting the problem as an instance of the longest common subsequence problem, a widely popular problem in computer science, typically considered as an illustrative example for the dynamic programming paradigm. Once the waypoints are identified, we perform symbolic reachability analysis on them to identify the earliest unreachable waypoint and report it as the explanation of unsolvability. We present experimental results on unsolvable planning problems in hybrid domains. 

**Abstract (ZH)**: 可解释的人工智能计划中解释规划问题不可解性的研究具有重要意义。尽管人工智能计划文献中已经报告了生成规划问题解的解释的研究努力，但解释规划问题不可解性仍是一个开放且研究不足的问题。一般用于计划生成和自动问题求解的一种广泛实践方法是将任务分解为有助于逐步靠近目标的子问题。在本文中，我们提议采用相同的问题识别哲学作为分析和解释混合系统中规划问题不可解性的机制。特别是，对于给定的不可解规划问题，我们提议识别常见的中间点，这些中间点是计划存在的普遍障碍，换句话说，它们出现在从起点到规划目标的每条计划路径上。我们提出了一种新颖的方法来识别中间点，将其问题重新表述为最长公共子序列问题的一个实例，这是计算机科学中广泛流行的问题，通常作为动态规划范式的示例问题。一旦识别出中间点，我们将对其执行符号可达性分析以识别最早不可达的中间点，并将其报告为不可解性的解释。我们展示了在混合域中不可解规划问题上的实验结果。 

---
# Improving Human-AI Coordination through Adversarial Training and Generative Models 

**Title (ZH)**: 通过对抗训练和生成模型提升人机协同效率 

**Authors**: Paresh Chaudhary, Yancheng Liang, Daphne Chen, Simon S. Du, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2504.15457)  

**Abstract**: Being able to cooperate with new people is an important component of many economically valuable AI tasks, from household robotics to autonomous driving. However, generalizing to novel humans requires training on data that captures the diversity of human behaviors. Adversarial training is one avenue for searching for such data and ensuring that agents are robust. However, it is difficult to apply in the cooperative setting because adversarial policies intentionally learn to sabotage the task instead of simulating valid cooperation partners. To address this challenge, we propose a novel strategy for overcoming self-sabotage that combines a pre-trained generative model to simulate valid cooperative agent policies with adversarial training to maximize regret. We call our method GOAT: Generative Online Adversarial Training. In this framework, the GOAT dynamically searches for and generates coordination strategies where the learning policy -- the Cooperator agent -- underperforms. GOAT enables better generalization by exposing the Cooperator to various challenging interaction scenarios. We maintain realistic coordination strategies by updating only the generative model's embedding while keeping its parameters frozen, thus avoiding adversarial exploitation. We evaluate GOAT with real human partners, and the results demonstrate state-of-the-art performance on the Overcooked benchmark, highlighting its effectiveness in generalizing to diverse human behaviors. 

**Abstract (ZH)**: 具备与新人合作的能力是许多具有经济价值的AI任务的重要组成部分，从家庭机器人到自主驾驶。然而，将合作对象推广到新型人类需要训练能够捕捉人类行为多样性的数据。对抗训练是寻找此类数据并确保智能体鲁棒性的途径之一。然而，在合作环境中应用对抗训练难度较大，因为对抗策略旨在破坏任务而不是模拟有效的合作伙伴。为应对这一挑战，我们提出了一种克服自我破坏的新策略，该策略结合了预训练生成模型来模拟有效的合作智能体策略，并使用对抗训练来最大化遗憾。我们称我们的方法为GOAT：生成对抗训练。在该框架中，GOAT动态地寻找并生成学习策略（合作者智能体）表现不佳的合作协调策略。GOAT通过使合作者智能体暴露于各种具有挑战性的交互场景中，从而实现更好的泛化。我们仅更新生成模型的嵌入而不冻结其参数，以防止对抗利用，并维持协调策略的现实性。我们使用真实人类合作伙伴评估GOAT，并且结果在Overcooked benchmark上展示了最先进的性能，突显了其在泛化到多样人类行为上的有效性。 

---
# AGI Is Coming... Right After AI Learns to Play Wordle 

**Title (ZH)**: AGI 将来临……在 AI 学会玩 Wordle 之后。 

**Authors**: Sarath Shekkizhar, Romain Cosentino  

**Link**: [PDF](https://arxiv.org/pdf/2504.15434)  

**Abstract**: This paper investigates multimodal agents, in particular, OpenAI's Computer-User Agent (CUA), trained to control and complete tasks through a standard computer interface, similar to humans. We evaluated the agent's performance on the New York Times Wordle game to elicit model behaviors and identify shortcomings. Our findings revealed a significant discrepancy in the model's ability to recognize colors correctly depending on the context. The model had a $5.36\%$ success rate over several hundred runs across a week of Wordle. Despite the immense enthusiasm surrounding AI agents and their potential to usher in Artificial General Intelligence (AGI), our findings reinforce the fact that even simple tasks present substantial challenges for today's frontier AI models. We conclude with a discussion of the potential underlying causes, implications for future development, and research directions to improve these AI systems. 

**Abstract (ZH)**: 本文研究了多模态代理，特别是由OpenAI训练的计算机-用户代理（CUA），该代理通过标准计算机接口控制和完成任务，类似于人类操作。我们评估了该代理在《纽约时报》Wordle游戏中的表现，以揭示其行为模式并识别其不足。研究发现表明，模型在正确识别颜色方面的能力在不同上下文中存在显著差异。该模型在Wordle游戏中数百次运行中的一周内成功率为5.36%。尽管人们对AI代理及其走向广泛人工智能（AGI）的潜力表现出极大的热情，但我们的发现强化了这样一个事实：即使是最简单的任务，对于当今的前沿AI模型来说仍然充满挑战。我们最后讨论了潜在的根本原因、对未来发展的影响以及改进这些AI系统的研究方向。 

---
# Reliable Classification with Conformal Learning and Interval-Type 2 Fuzzy Sets 

**Title (ZH)**: 可靠的分类：基于同调学习和区间型2模糊集的方法 

**Authors**: Javier Fumanal-Idocin, Javier Andreu-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2504.15360)  

**Abstract**: Classical machine learning classifiers tend to be overconfident can be unreliable outside of the laboratory benchmarks. Properly assessing the reliability of the output of the model per sample is instrumental for real-life scenarios where these systems are deployed. Because of this, different techniques have been employed to properly quantify the quality of prediction for a given model. These are most commonly Bayesian statistics and, more recently, conformal learning. Given a calibration set, conformal learning can produce outputs that are guaranteed to cover the target class with a desired significance level, and are more reliable than the standard confidence intervals used by Bayesian methods. In this work, we propose to use conformal learning with fuzzy rule-based systems in classification and show some metrics of their performance. Then, we discuss how the use of type 2 fuzzy sets can improve the quality of the output of the system compared to both fuzzy and crisp rules. Finally, we also discuss how the fine-tuning of the system can be adapted to improve the quality of the conformal prediction. 

**Abstract (ZH)**: 经典机器学习分类器往往在实验室基准之外表现出过度自信，可能导致不可靠性。正确评估每个样本模型输出的可靠性对于这些系统部署的真实场景至关重要。因此，已经采用了不同的技术来准确量化给定模型的预测质量。这些技术中最常见的是贝叶斯统计，近年来则是核验学习。通过给定校准集，核验学习可以生成保证覆盖目标类别的预测输出，并且比贝叶斯方法常用的标准置信区间更为可靠。在此工作中，我们提出将核验学习与模糊规则基于系统结合用于分类，并展示其性能指标。然后，我们讨论不同类型2模糊集如何改进系统输出质量，相较于模糊规则和模糊规则。最后，我们还讨论如何调整系统微调以提高核验预测的质量。 

---
# Vision language models are unreliable at trivial spatial cognition 

**Title (ZH)**: 视觉语言模型在简单的空间认知上不可靠。 

**Authors**: Sangeet Khemlani, Tyler Tran, Nathaniel Gyory, Anthony M. Harrison, Wallace E. Lawson, Ravenna Thielstrom, Hunter Thompson, Taaren Singh, J. Gregory Trafton  

**Link**: [PDF](https://arxiv.org/pdf/2504.16061)  

**Abstract**: Vision language models (VLMs) are designed to extract relevant visuospatial information from images. Some research suggests that VLMs can exhibit humanlike scene understanding, while other investigations reveal difficulties in their ability to process relational information. To achieve widespread applicability, VLMs must perform reliably, yielding comparable competence across a wide variety of related tasks. We sought to test how reliable these architectures are at engaging in trivial spatial cognition, e.g., recognizing whether one object is left of another in an uncluttered scene. We developed a benchmark dataset -- TableTest -- whose images depict 3D scenes of objects arranged on a table, and used it to evaluate state-of-the-art VLMs. Results show that performance could be degraded by minor variations of prompts that use logically equivalent descriptions. These analyses suggest limitations in how VLMs may reason about spatial relations in real-world applications. They also reveal novel opportunities for bolstering image caption corpora for more efficient training and testing. 

**Abstract (ZH)**: 视觉语言模型（VLMs）的设计目的是从图像中提取相关的视空间信息。一些研究表明VLMs可以表现出类似人类的场景理解能力，而其他研究则揭示了它们在处理关系信息方面的困难。为了实现广泛的应用性，VLMs必须可靠地工作，在一系列相关任务中展现出相当的绩效。我们试图测试这些架构在进行基本的空间认知（例如，在简洁场景中识别一个物体是否位于另一个物体左侧）方面的可靠性。我们开发了一个基准数据集——TableTest，该数据集包含桌子上摆放物体的3D场景图像，并使用该数据集评估了最先进的VLMs。结果表明，即使是逻辑等价的描述，提示语的小变化也可能导致性能下降。这些分析表明，VLMs在现实世界应用中如何推理空间关系存在局限性。它们还揭示了增强图像字幕语料库以提高训练和测试效率的新机会。 

---
# LongMamba: Enhancing Mamba's Long Context Capabilities via Training-Free Receptive Field Enlargement 

**Title (ZH)**: 长 '), 通过训练-free感受野扩展增强 Mamba 的长上下文能力 

**Authors**: Zhifan Ye, Kejing Xia, Yonggan Fu, Xin Dong, Jihoon Hong, Xiangchi Yuan, Shizhe Diao, Jan Kautz, Pavlo Molchanov, Yingyan Celine Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.16053)  

**Abstract**: State space models (SSMs) have emerged as an efficient alternative to Transformer models for language modeling, offering linear computational complexity and constant memory usage as context length increases. However, despite their efficiency in handling long contexts, recent studies have shown that SSMs, such as Mamba models, generally underperform compared to Transformers in long-context understanding tasks. To address this significant shortfall and achieve both efficient and accurate long-context understanding, we propose LongMamba, a training-free technique that significantly enhances the long-context capabilities of Mamba models. LongMamba builds on our discovery that the hidden channels in Mamba can be categorized into local and global channels based on their receptive field lengths, with global channels primarily responsible for long-context capability. These global channels can become the key bottleneck as the input context lengthens. Specifically, when input lengths largely exceed the training sequence length, global channels exhibit limitations in adaptively extend their receptive fields, leading to Mamba's poor long-context performance. The key idea of LongMamba is to mitigate the hidden state memory decay in these global channels by preventing the accumulation of unimportant tokens in their memory. This is achieved by first identifying critical tokens in the global channels and then applying token filtering to accumulate only those critical tokens. Through extensive benchmarking across synthetic and real-world long-context scenarios, LongMamba sets a new standard for Mamba's long-context performance, significantly extending its operational range without requiring additional training. Our code is available at this https URL. 

**Abstract (ZH)**: 基于状态空间模型（SSMs）已成为语言建模中Transformer模型的有效替代方案，提供随上下文长度增加而线性的计算复杂度和恒定的内存使用。然而，尽管在处理长上下文方面效率高，最近的研究表明，如Mamba模型在内的SSMs在长上下文理解任务中通常逊色于Transformer。为解决这一显著缺陷，实现既高效又准确的长上下文理解，我们提出了一种无需训练的技术LongMamba，显著增强了Mamba模型的长上下文能力。LongMamba基于我们发现Mamba模型中的隐藏通道可以根据感受野长度分为局部和全局通道，全局通道主要负责长上下文能力。当输入上下文长度增加时，全局通道可能会成为关键瓶颈，尤其是在输入长度远超训练序列长度时，全局通道表现出在自适应扩展感受野方面的局限，导致Mamba在长上下文性能上的不足。LongMamba的关键思想是通过阻止不重要 tokens 在全局通道中的累积来减轻隐藏状态记忆衰减，通过首先识别全局通道中的关键 tokens，然后应用 token 过滤只积累这些关键 tokens。通过在合成和真实世界长上下文场景中的广泛基准测试，LongMamba 设定了Mamba 长上下文性能的新标准，无需额外训练即可显著扩展其操作范围。我们的代码可在以下链接获取。 

---
# Muon Optimizer Accelerates Grokking 

**Title (ZH)**: Muon优化器加速Grokking 

**Authors**: Amund Tveit, Bjørn Remseth, Arve Skogvold  

**Link**: [PDF](https://arxiv.org/pdf/2504.16041)  

**Abstract**: This paper investigates the impact of different optimizers on the grokking phenomenon, where models exhibit delayed generalization. We conducted experiments across seven numerical tasks (primarily modular arithmetic) using a modern Transformer architecture. The experimental configuration systematically varied the optimizer (Muon vs. AdamW) and the softmax activation function (standard softmax, stablemax, and sparsemax) to assess their combined effect on learning dynamics. Our empirical evaluation reveals that the Muon optimizer, characterized by its use of spectral norm constraints and second-order information, significantly accelerates the onset of grokking compared to the widely used AdamW optimizer. Specifically, Muon reduced the mean grokking epoch from 153.09 to 102.89 across all configurations, a statistically significant difference (t = 5.0175, p = 6.33e-08). This suggests that the optimizer choice plays a crucial role in facilitating the transition from memorization to generalization. 

**Abstract (ZH)**: 本文探讨了不同优化器对grokking现象的影响，其中模型表现出延迟泛化。我们在七个数值任务（主要是模态算术）上使用了现代Transformer架构进行了实验。实验配置系统地改变了优化器（Muon与AdamW）和softmax激活函数（标准softmax、稳定softmax和稀疏softmax）以评估它们对学习动态的联合影响。我们的实证评估表明，通过使用谱范数约束和二阶信息的Muon优化器显著加速了与广泛使用的AdamW优化器相比的grokking的出现时期。具体而言，Muon将所有配置下的平均grokking时期从153.09降低到102.89，这具有统计学显著性差异（t = 5.0175，p = 6.33e-08）。这表明优化器的选择在促进从记忆到泛化的过渡中起着关键作用。 

---
# Trends in AI Supercomputers 

**Title (ZH)**: AI超级计算机的发展趋势 

**Authors**: Konstantin F. Pilz, James Sanders, Robi Rahman, Lennart Heim  

**Link**: [PDF](https://arxiv.org/pdf/2504.16026)  

**Abstract**: Frontier AI development relies on powerful AI supercomputers, yet analysis of these systems is limited. We create a dataset of 500 AI supercomputers from 2019 to 2025 and analyze key trends in performance, power needs, hardware cost, ownership, and global distribution. We find that the computational performance of AI supercomputers has doubled every nine months, while hardware acquisition cost and power needs both doubled every year. The leading system in March 2025, xAI's Colossus, used 200,000 AI chips, had a hardware cost of \$7B, and required 300 MW of power, as much as 250,000 households. As AI supercomputers evolved from tools for science to industrial machines, companies rapidly expanded their share of total AI supercomputer performance, while the share of governments and academia diminished. Globally, the United States accounts for about 75% of total performance in our dataset, with China in second place at 15%. If the observed trends continue, the leading AI supercomputer in 2030 will achieve $2\times10^{22}$ 16-bit FLOP/s, use two million AI chips, have a hardware cost of \$200 billion, and require 9 GW of power. Our analysis provides visibility into the AI supercomputer landscape, allowing policymakers to assess key AI trends like resource needs, ownership, and national competitiveness. 

**Abstract (ZH)**: 前沿AI发展依赖强大的AI超级计算机，但对其系统的分析却相对有限。我们创建了一个包含2019至2025年500台AI超级计算机的数据集，并分析了性能、功率需求、硬件成本、所有权和全球分布等方面的关键趋势。我们发现，AI超级计算机的计算性能每九个月翻一番，而硬件采购成本和功率需求则每年翻一番。2025年3月，排名首位的系统xAI的Colossus使用了200,000个AI芯片，硬件成本为70亿美元，需300兆瓦电力，相当于25万个家庭的用量。随着AI超级计算机从科研工具演变为工业机器，公司迅速扩大了它们在整体AI超级计算机性能中的份额，而政府和学术机构的份额则减少。在全球范围内，美国占据了我们数据集中总性能的约75%，中国位居第二，占比15%。若观察到的趋势持续下去，到2030年，领先的AI超级计算机将达到每秒2乘以10的22次方16位浮点运算，使用200万个AI芯片，硬件成本为2000亿美元，需9吉瓦电力。我们的分析为了解AI超级计算机的格局提供了视角，使政策制定者能够评估诸如资源需求、所有权和国家竞争力等关键AI趋势。 

---
# AlphaGrad: Non-Linear Gradient Normalization Optimizer 

**Title (ZH)**: AlphaGrad: 非线性梯度规范化优化器 

**Authors**: Soham Sane  

**Link**: [PDF](https://arxiv.org/pdf/2504.16020)  

**Abstract**: We introduce AlphaGrad, a memory-efficient, conditionally stateless optimizer addressing the memory overhead and hyperparameter complexity of adaptive methods like Adam. AlphaGrad enforces scale invariance via tensor-wise L2 gradient normalization followed by a smooth hyperbolic tangent transformation, $g' = \tanh(\alpha \cdot \tilde{g})$, controlled by a single steepness parameter $\alpha$. Our contributions include: (1) the AlphaGrad algorithm formulation; (2) a formal non-convex convergence analysis guaranteeing stationarity; (3) extensive empirical evaluation on diverse RL benchmarks (DQN, TD3, PPO). Compared to Adam, AlphaGrad demonstrates a highly context-dependent performance profile. While exhibiting instability in off-policy DQN, it provides enhanced training stability with competitive results in TD3 (requiring careful $\alpha$ tuning) and achieves substantially superior performance in on-policy PPO. These results underscore the critical importance of empirical $\alpha$ selection, revealing strong interactions between the optimizer's dynamics and the underlying RL algorithm. AlphaGrad presents a compelling alternative optimizer for memory-constrained scenarios and shows significant promise for on-policy learning regimes where its stability and efficiency advantages can be particularly impactful. 

**Abstract (ZH)**: AlphaGrad：一种内存高效且条件无状态的优化器及其在强化学习中的应用分析 

---
# How Private is Your Attention? Bridging Privacy with In-Context Learning 

**Title (ZH)**: 你的注意力有多私密？将隐私与上下文学习相结合 

**Authors**: Soham Bonnerjee, Zhen Wei, Yeon, Anna Asch, Sagnik Nandy, Promit Ghosal  

**Link**: [PDF](https://arxiv.org/pdf/2504.16000)  

**Abstract**: In-context learning (ICL)-the ability of transformer-based models to perform new tasks from examples provided at inference time-has emerged as a hallmark of modern language models. While recent works have investigated the mechanisms underlying ICL, its feasibility under formal privacy constraints remains largely unexplored. In this paper, we propose a differentially private pretraining algorithm for linear attention heads and present the first theoretical analysis of the privacy-accuracy trade-off for ICL in linear regression. Our results characterize the fundamental tension between optimization and privacy-induced noise, formally capturing behaviors observed in private training via iterative methods. Additionally, we show that our method is robust to adversarial perturbations of training prompts, unlike standard ridge regression. All theoretical findings are supported by extensive simulations across diverse settings. 

**Abstract (ZH)**: 基于上下文学习的差分隐私预训练算法及其在线性回归中的隐私-准确 trade-off 理论分析 

---
# OPUS-VFL: Incentivizing Optimal Privacy-Utility Tradeoffs in Vertical Federated Learning 

**Title (ZH)**: OPUS-VFL: 激励垂直联邦学习中的最优隐私-实用性权衡 

**Authors**: Sindhuja Madabushi, Ahmad Faraz Khan, Haider Ali, Jin-Hee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2504.15995)  

**Abstract**: Vertical Federated Learning (VFL) enables organizations with disjoint feature spaces but shared user bases to collaboratively train models without sharing raw data. However, existing VFL systems face critical limitations: they often lack effective incentive mechanisms, struggle to balance privacy-utility tradeoffs, and fail to accommodate clients with heterogeneous resource capabilities. These challenges hinder meaningful participation, degrade model performance, and limit practical deployment. To address these issues, we propose OPUS-VFL, an Optimal Privacy-Utility tradeoff Strategy for VFL. OPUS-VFL introduces a novel, privacy-aware incentive mechanism that rewards clients based on a principled combination of model contribution, privacy preservation, and resource investment. It employs a lightweight leave-one-out (LOO) strategy to quantify feature importance per client, and integrates an adaptive differential privacy mechanism that enables clients to dynamically calibrate noise levels to optimize their individual utility. Our framework is designed to be scalable, budget-balanced, and robust to inference and poisoning attacks. Extensive experiments on benchmark datasets (MNIST, CIFAR-10, and CIFAR-100) demonstrate that OPUS-VFL significantly outperforms state-of-the-art VFL baselines in both efficiency and robustness. It reduces label inference attack success rates by up to 20%, increases feature inference reconstruction error (MSE) by over 30%, and achieves up to 25% higher incentives for clients that contribute meaningfully while respecting privacy and cost constraints. These results highlight the practicality and innovation of OPUS-VFL as a secure, fair, and performance-driven solution for real-world VFL. 

**Abstract (ZH)**: OPUS-VFL：面向VFL的最优隱私-效用 TRADEOFF 策略 

---
# Bug Destiny Prediction in Large Open-Source Software Repositories through Sentiment Analysis and BERT Topic Modeling 

**Title (ZH)**: 基于情感分析和BERT主题建模的大规模开源软件仓库中的 Bug 目标预测 

**Authors**: Sophie C. Pope, Andrew Barovic, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15972)  

**Abstract**: This study explores a novel approach to predicting key bug-related outcomes, including the time to resolution, time to fix, and ultimate status of a bug, using data from the Bugzilla Eclipse Project. Specifically, we leverage features available before a bug is resolved to enhance predictive accuracy. Our methodology incorporates sentiment analysis to derive both an emotionality score and a sentiment classification (positive or negative). Additionally, we integrate the bug's priority level and its topic, extracted using a BERTopic model, as features for a Convolutional Neural Network (CNN) and a Multilayer Perceptron (MLP). Our findings indicate that the combination of BERTopic and sentiment analysis can improve certain model performance metrics. Furthermore, we observe that balancing model inputs enhances practical applicability, albeit at the cost of a significant reduction in accuracy in most cases. To address our primary objectives, predicting time-to-resolution, time-to-fix, and bug destiny, we employ both binary classification and exact time value predictions, allowing for a comparative evaluation of their predictive effectiveness. Results demonstrate that sentiment analysis serves as a valuable predictor of a bug's eventual outcome, particularly in determining whether it will be fixed. However, its utility is less pronounced when classifying bugs into more complex or unconventional outcome categories. 

**Abstract (ZH)**: 本研究利用来自Eclipse项目Bugzilla的数据，探索了一种新颖的方法来预测关键故障相关结果，包括故障解决时间、修复时间以及最终状态，并利用故障解决前可用的特征来提升预测准确性。我们的方法结合了情感分析以提取情感分数和情感分类（积极或消极），同时整合了使用BERTopic模型提取的故障优先级和主题作为卷积神经网络（CNN）和多层感知器（MLP）的特征。研究发现，BERTopic与情感分析的结合可以改善某些模型性能指标。此外，我们观察到，平衡模型输入可以提高实际应用性，尽管这通常会伴随着准确性的显著下降。为了实现主要目标，即预测故障解决时间、修复时间和故障命运，我们采用了二分类和精确时间值预测两种方法，以比较它们的预测效果。结果显示，情感分析是预测故障最终结果的一个有价值的指标，特别是对于确定故障是否会被修复有较大帮助。然而，在将故障归类为更复杂或非传统类别时，其效用则显得较低。 

---
# Universal Approximation with Softmax Attention 

**Title (ZH)**: softmax 注意机制的通用逼近能力 

**Authors**: Jerry Yao-Chieh Hu, Hude Liu, Hong-Yu Chen, Weimin Wu, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15956)  

**Abstract**: We prove that with linear transformations, both (i) two-layer self-attention and (ii) one-layer self-attention followed by a softmax function are universal approximators for continuous sequence-to-sequence functions on compact domains. Our main technique is a new interpolation-based method for analyzing attention's internal mechanism. This leads to our key insight: self-attention is able to approximate a generalized version of ReLU to arbitrary precision, and hence subsumes many known universal approximators. Building on these, we show that two-layer multi-head attention alone suffices as a sequence-to-sequence universal approximator. In contrast, prior works rely on feed-forward networks to establish universal approximation in Transformers. Furthermore, we extend our techniques to show that, (softmax-)attention-only layers are capable of approximating various statistical models in-context. We believe these techniques hold independent interest. 

**Abstract (ZH)**: 我们证明，在线性变换下，无论是（i）两层自注意力，还是（ii）一层自注意力后跟随softmax函数，都是紧域上连续序列到序列函数的通用逼近器。我们的主要技术是一种新的基于插值的方法来分析注意力的内部机制。这使得我们得出关键见解：自注意力能够以任意精度逼近广义ReLU函数，从而涵盖了多种已知的通用逼近器。在此基础上，我们展示两层多头注意力本身足以作为序列到序列的通用逼近器。相比之下，先前的工作依赖前馈网络来建立 Transformer 的通用逼近性。此外，我们将这些技术扩展，展示仅使用注意力层可以逼近各种上下文中的统计模型。我们认为这些技术具有独立的兴趣。 

---
# A Clinician-Friendly Platform for Ophthalmic Image Analysis Without Technical Barriers 

**Title (ZH)**: 适用于眼科图像分析的clinician-friendly平台，无技术壁垒 

**Authors**: Meng Wang, Tian Lin, Qingshan Hou, Aidi Lin, Jingcheng Wang, Qingsheng Peng, Truong X. Nguyen, Danqi Fang, Ke Zou, Ting Xu, Cancan Xue, Ten Cheer Quek, Qinkai Yu, Minxin Liu, Hui Zhou, Zixuan Xiao, Guiqin He, Huiyu Liang, Tingkun Shi, Man Chen, Linna Liu, Yuanyuan Peng, Lianyu Wang, Qiuming Hu, Junhong Chen, Zhenhua Zhang, Cheng Chen, Yitian Zhao, Dianbo Liu, Jianhua Wu, Xinjian Chen, Changqing Zhang, Triet Thanh Nguyen, Yanda Meng, Yalin Zheng, Yih Chung Tham, Carol Y. Cheung, Huazhu Fu, Haoyu Chen, Ching-Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.15928)  

**Abstract**: Artificial intelligence (AI) shows remarkable potential in medical imaging diagnostics, but current models typically require retraining when deployed across different clinical centers, limiting their widespread adoption. We introduce GlobeReady, a clinician-friendly AI platform that enables ocular disease diagnosis without retraining/fine-tuning or technical expertise. GlobeReady achieves high accuracy across imaging modalities: 93.9-98.5% for an 11-category fundus photo dataset and 87.2-92.7% for a 15-category OCT dataset. Through training-free local feature augmentation, it addresses domain shifts across centers and populations, reaching an average accuracy of 88.9% across five centers in China, 86.3% in Vietnam, and 90.2% in the UK. The built-in confidence-quantifiable diagnostic approach further boosted accuracy to 94.9-99.4% (fundus) and 88.2-96.2% (OCT), while identifying out-of-distribution cases at 86.3% (49 CFP categories) and 90.6% (13 OCT categories). Clinicians from multiple countries rated GlobeReady highly (average 4.6 out of 5) for its usability and clinical relevance. These results demonstrate GlobeReady's robust, scalable diagnostic capability and potential to support ophthalmic care without technical barriers. 

**Abstract (ZH)**: 人工智能（AI）在医疗影像诊断方面显示出巨大潜力，但当前模型在不同临床中心部署时通常需要重新训练，限制了其广泛应用。我们介绍了GlobeReady，一个用户友好的AI平台，可在无需重新训练/调整或技术 expertise 的情况下进行眼科疾病诊断。GlobeReady在不同影像模态上实现了高准确性：对于包含11个类别的基金图数据集达到了93.9-98.5%，对于包含15个类别的OCT数据集达到了87.2-92.7%。通过训练免费的本地特征增强，它解决了不同中心和人群之间的领域转移问题，在中国五个中心、越南和英国达到了平均88.9%、86.3%和90.2%的准确性。内置的可量化的诊断方法进一步提高了准确性，基金图为94.9-99.4%，OCT为88.2-96.2%，同时识别出不在分布情况下的病例，分别为86.3%（49个基金图类别）和90.6%（13个OCT类别）。来自多个地区的眼科医生高度评价了GlobeReady的易用性和临床相关性（平均评分为4.6分）。这些结果展示了GlobeReady稳健且可扩展的诊断能力，并且有可能在没有技术障碍的情况下支持眼科医疗。 

---
# New Recipe for Semi-supervised Community Detection: Clique Annealing under Crystallization Kinetics 

**Title (ZH)**: 新的半监督社区检测配方：结晶动力学下的团块退火 

**Authors**: Ling Cheng, Jiashu Pu, Ruicheng Liang, Qian Shao, Hezhe Qiao, Feida Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15927)  

**Abstract**: Semi-supervised community detection methods are widely used for identifying specific communities due to the label scarcity. Existing semi-supervised community detection methods typically involve two learning stages learning in both initial identification and subsequent adjustment, which often starts from an unreasonable community core candidate. Moreover, these methods encounter scalability issues because they depend on reinforcement learning and generative adversarial networks, leading to higher computational costs and restricting the selection of candidates. To address these limitations, we draw a parallel between crystallization kinetics and community detection to integrate the spontaneity of the annealing process into community detection. Specifically, we liken community detection to identifying a crystal subgrain (core) that expands into a complete grain (community) through a process similar to annealing. Based on this finding, we propose CLique ANNealing (CLANN), which applies kinetics concepts to community detection by integrating these principles into the optimization process to strengthen the consistency of the community core. Subsequently, a learning-free Transitive Annealer was employed to refine the first-stage candidates by merging neighboring cliques and repositioning the community core, enabling a spontaneous growth process that enhances scalability. Extensive experiments on \textbf{43} different network settings demonstrate that CLANN outperforms state-of-the-art methods across multiple real-world datasets, showcasing its exceptional efficacy and efficiency in community detection. 

**Abstract (ZH)**: 半监督社区检测方法通过晶化动力学增强社区核心一致性研究 

---
# Achieving Distributive Justice in Federated Learning via Uncertainty Quantification 

**Title (ZH)**: 通过不确定性量化实现联邦学习中的分配正义 

**Authors**: Alycia Carey, Xintao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15924)  

**Abstract**: Client-level fairness metrics for federated learning are used to ensure that all clients in a federation either: a) have similar final performance on their local data distributions (i.e., client parity), or b) obtain final performance on their local data distributions relative to their contribution to the federated learning process (i.e., contribution fairness). While a handful of works that propose either client-parity or contribution-based fairness metrics ground their definitions and decisions in social theories of equality -- such as distributive justice -- most works arbitrarily choose what notion of fairness to align with which makes it difficult for practitioners to choose which fairness metric aligns best with their fairness ethics. In this work, we propose UDJ-FL (Uncertainty-based Distributive Justice for Federated Learning), a flexible federated learning framework that can achieve multiple distributive justice-based client-level fairness metrics. Namely, by utilizing techniques inspired by fair resource allocation, in conjunction with performing aleatoric uncertainty-based client weighing, our UDJ-FL framework is able to achieve egalitarian, utilitarian, Rawls' difference principle, or desert-based client-level fairness. We empirically show the ability of UDJ-FL to achieve all four defined distributive justice-based client-level fairness metrics in addition to providing fairness equivalent to (or surpassing) other popular fair federated learning works. Further, we provide justification for why aleatoric uncertainty weighing is necessary to the construction of our UDJ-FL framework as well as derive theoretical guarantees for the generalization bounds of UDJ-FL. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 基于分布正义的联邦学习客户端级公平度量 

---
# Automated Bug Report Prioritization in Large Open-Source Projects 

**Title (ZH)**: 大型开源项目中的自动 bug 报告优先级分配 

**Authors**: Riley Pierson, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15912)  

**Abstract**: Large open-source projects receive a large number of issues (known as bugs), including software defect (i.e., bug) reports and new feature requests from their user and developer communities at a fast rate. The often limited project resources do not allow them to deal with all issues. Instead, they have to prioritize them according to the project's priorities and the issues' severities. In this paper, we propose a novel approach to automated bug prioritization based on the natural language text of the bug reports that are stored in the open bug repositories of the issue-tracking systems. We conduct topic modeling using a variant of LDA called TopicMiner-MTM and text classification with the BERT large language model to achieve a higher performance level compared to the state-of-the-art. Experimental results using an existing reference dataset containing 85,156 bug reports of the Eclipse Platform project indicate that we outperform existing approaches in terms of Accuracy, Precision, Recall, and F1-measure of the bug report priority prediction. 

**Abstract (ZH)**: 基于开放问题跟踪系统中 Bug 报告自然语言文本的新型自动化 Bug 优先级排序方法 

---
# GraphEdge: Dynamic Graph Partition and Task Scheduling for GNNs Computing in Edge Network 

**Title (ZH)**: GraphEdge: 动态图 partition 和任务调度以适应边缘网络中 GNNs 计算 

**Authors**: Wenjing Xiao, Chenglong Shi, Miaojiang Chen, Zhiquan Liu, Min Chen, H. Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.15905)  

**Abstract**: With the exponential growth of Internet of Things (IoT) devices, edge computing (EC) is gradually playing an important role in providing cost-effective services. However, existing approaches struggle to perform well in graph-structured scenarios where user data is correlated, such as traffic flow prediction and social relationship recommender systems. In particular, graph neural network (GNN)-based approaches lead to expensive server communication cost. To address this problem, we propose GraphEdge, an efficient GNN-based EC architecture. It considers the EC system of GNN tasks, where there are associations between users and it needs to take into account the task data of its neighbors when processing the tasks of a user. Specifically, the architecture first perceives the user topology and represents their data associations as a graph layout at each time step. Then the graph layout is optimized by calling our proposed hierarchical traversal graph cut algorithm (HiCut), which cuts the graph layout into multiple weakly associated subgraphs based on the aggregation characteristics of GNN, and the communication cost between different subgraphs during GNN inference is minimized. Finally, based on the optimized graph layout, our proposed deep reinforcement learning (DRL) based graph offloading algorithm (DRLGO) is executed to obtain the optimal offloading strategy for the tasks of users, the offloading strategy is subgraph-based, it tries to offload user tasks in a subgraph to the same edge server as possible while minimizing the task processing time and energy consumption of the EC system. Experimental results show the good effectiveness and dynamic adaptation of our proposed architecture and it also performs well even in dynamic scenarios. 

**Abstract (ZH)**: 基于图神经网络的边缘计算架构GraphEdge：在图结构场景中提高用户数据关联处理效率 

---
# Supporting Data-Frame Dynamics in AI-assisted Decision Making 

**Title (ZH)**: 支持数据框架动态性在AI辅助决策中的应用 

**Authors**: Chengbo Zheng, Tim Miller, Alina Bialkowski, H Peter Soyer, Monika Janda  

**Link**: [PDF](https://arxiv.org/pdf/2504.15894)  

**Abstract**: High stakes decision-making often requires a continuous interplay between evolving evidence and shifting hypotheses, a dynamic that is not well supported by current AI decision support systems. In this paper, we introduce a mixed-initiative framework for AI assisted decision making that is grounded in the data-frame theory of sensemaking and the evaluative AI paradigm. Our approach enables both humans and AI to collaboratively construct, validate, and adapt hypotheses. We demonstrate our framework with an AI-assisted skin cancer diagnosis prototype that leverages a concept bottleneck model to facilitate interpretable interactions and dynamic updates to diagnostic hypotheses. 

**Abstract (ZH)**: 高风险决策往往需要在不断演化的证据和变化的假设之间进行动态交互，而当前的AI决策支持系统在支持这种动态交互方面存在不足。本文 introduces一种基于数据框架的意义构建理论和评估型AI范式的混合初始化框架，以实现人机协作构建、验证和调整假设。我们通过一个利用概念瓶颈模型的AI辅助皮肤癌诊断原型来展示该框架，该原型促进了可解释的交互和诊断假设的动态更新。 

---
# MedNNS: Supernet-based Medical Task-Adaptive Neural Network Search 

**Title (ZH)**: MedNNS：基于超网络的医疗任务自适应神经网络搜索 

**Authors**: Lotfi Abdelkrim Mecharbat, Ibrahim Elmakky, Martin Takac, Mohammed Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2504.15865)  

**Abstract**: Deep learning (DL) has achieved remarkable progress in the field of medical imaging. However, adapting DL models to medical tasks remains a significant challenge, primarily due to two key factors: (1) architecture selection, as different tasks necessitate specialized model designs, and (2) weight initialization, which directly impacts the convergence speed and final performance of the models. Although transfer learning from ImageNet is a widely adopted strategy, its effectiveness is constrained by the substantial differences between natural and medical images. To address these challenges, we introduce Medical Neural Network Search (MedNNS), the first Neural Network Search framework for medical imaging applications. MedNNS jointly optimizes architecture selection and weight initialization by constructing a meta-space that encodes datasets and models based on how well they perform together. We build this space using a Supernetwork-based approach, expanding the model zoo size by 51x times over previous state-of-the-art (SOTA) methods. Moreover, we introduce rank loss and Fréchet Inception Distance (FID) loss into the construction of the space to capture inter-model and inter-dataset relationships, thereby achieving more accurate alignment in the meta-space. Experimental results across multiple datasets demonstrate that MedNNS significantly outperforms both ImageNet pre-trained DL models and SOTA Neural Architecture Search (NAS) methods, achieving an average accuracy improvement of 1.7% across datasets while converging substantially faster. The code and the processed meta-space is available at this https URL. 

**Abstract (ZH)**: 深度学习在医学影像领域的进展显著，但将深度学习模型适应医学任务仍然是一项重大挑战，主要归因于两个关键因素：(1) 架构选择，不同的任务需要专门的设计模型，(2) 权重初始化，这直接影响模型的收敛速度和最终性能。尽管从ImageNet迁移学习是一种广泛采用的策略，但其有效性受到自然图像和医学图像之间重大差异的限制。为了解决这些挑战，我们提出了Medical Neural Network Search (MedNNS)，这是第一个针对医学影像应用的神经网络搜索框架。MedNNS通过构建一个元空间来同时优化架构选择和权重初始化，该空间根据数据集和模型之间的性能关系来编码。我们采用SuperNetwork的方法构建此空间，将模型库的大小扩展了51倍，超过之前最先进的方法。此外，我们引入了排名损失和Fréchet Inception Distance (FID)损失来捕捉模型间和数据集间的关联，从而在元空间中实现更准确的对齐。跨多个数据集的实验结果表明，MedNNS在数据集上的平均准确率提高了1.7%，同时收敛速度大幅加快。代码和处理后的元空间可在此处访问。 

---
# DualOptim: Enhancing Efficacy and Stability in Machine Unlearning with Dual Optimizers 

**Title (ZH)**: DualOptim: 提升机器遗忘效果与稳定性的双重优化器 

**Authors**: Xuyang Zhong, Haochen Luo, Chen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15827)  

**Abstract**: Existing machine unlearning (MU) approaches exhibit significant sensitivity to hyperparameters, requiring meticulous tuning that limits practical deployment. In this work, we first empirically demonstrate the instability and suboptimal performance of existing popular MU methods when deployed in different scenarios. To address this issue, we propose Dual Optimizer (DualOptim), which incorporates adaptive learning rate and decoupled momentum factors. Empirical and theoretical evidence demonstrates that DualOptim contributes to effective and stable unlearning. Through extensive experiments, we show that DualOptim can significantly boost MU efficacy and stability across diverse tasks, including image classification, image generation, and large language models, making it a versatile approach to empower existing MU algorithms. 

**Abstract (ZH)**: 现有的机器卸载（MU）方法对超参数高度敏感，需要细致调整才能部署，限制了其实用性。在本文中，我们首先实验证明了现有流行的MU方法在不同场景下部署时的不稳定性和次优性能。为解决这一问题，我们提出了一种双优化器（DualOptim），它结合了自适应学习率和解耦动量因子。实证和理论证据表明，双优化器有助于实现有效的和稳定的卸载。通过广泛的实验，我们展示了双优化器可以显著提升MU的有效性和稳定性，适用于包括图像分类、图像生成和大规模语言模型等多种任务，使其成为增强现有MU算法的通用方法。 

---
# Human-Imperceptible Physical Adversarial Attack for NIR Face Recognition Models 

**Title (ZH)**: 不可感知的人体物理对抗攻击针对NIR面部识别模型 

**Authors**: Songyan Xie, Jinghang Wen, Encheng Su, Qiucheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15823)  

**Abstract**: Near-infrared (NIR) face recognition systems, which can operate effectively in low-light conditions or in the presence of makeup, exhibit vulnerabilities when subjected to physical adversarial attacks. To further demonstrate the potential risks in real-world applications, we design a novel, stealthy, and practical adversarial patch to attack NIR face recognition systems in a black-box setting. We achieved this by utilizing human-imperceptible infrared-absorbing ink to generate multiple patches with digitally optimized shapes and positions for infrared images. To address the optimization mismatch between digital and real-world NIR imaging, we develop a light reflection model for human skin to minimize pixel-level discrepancies by simulating NIR light reflection.
Compared to state-of-the-art (SOTA) physical attacks on NIR face recognition systems, the experimental results show that our method improves the attack success rate in both digital and physical domains, particularly maintaining effectiveness across various face postures. Notably, the proposed approach outperforms SOTA methods, achieving an average attack success rate of 82.46% in the physical domain across different models, compared to 64.18% for existing methods. The artifact is available at this https URL. 

**Abstract (ZH)**: 近红外(NIR)人脸识别系统在低光照条件或化妆情况下仍能有效运行，但在遭受物理对抗攻击时会存在漏洞。为了进一步在真实应用场景中展示潜在风险，我们设计了一种新颖、隐蔽且实用的对抗贴片，以在黑盒环境中攻击NIR人脸识别系统。我们通过使用人眼不可感知的红外吸收墨水，生成具有数字优化形状和位置的多个贴片，适用于红外图像。为了解决数字与现实世界NIR成像之间的优化不匹配问题，我们开发了一个人类皮肤的光反射模型，通过模拟红外光反射来最小化像素级差异。与最新的物理对抗攻击方法相比，实验结果显示，我们的方法在数字域和物理域均提高了攻击成功率，特别是在各种面部姿态下保持了有效性。值得注意的是，所提出的方法在不同模型的物理域中平均攻击成功率达到了82.46%，而现有方法仅为64.18%。该项目的实现代码可在此链接访问。 

---
# Fusing Reward and Dueling Feedback in Stochastic Bandits 

**Title (ZH)**: 融合奖励与 Dueling 反馈的随机Bandit问题 

**Authors**: Xuchuang Wang, Qirun Zeng, Jinhang Zuo, Xutong Liu, Mohammad Hajiesmaili, John C.S. Lui, Adam Wierman  

**Link**: [PDF](https://arxiv.org/pdf/2504.15812)  

**Abstract**: This paper investigates the fusion of absolute (reward) and relative (dueling) feedback in stochastic bandits, where both feedback types are gathered in each decision round. We derive a regret lower bound, demonstrating that an efficient algorithm may incur only the smaller among the reward and dueling-based regret for each individual arm. We propose two fusion approaches: (1) a simple elimination fusion algorithm that leverages both feedback types to explore all arms and unifies collected information by sharing a common candidate arm set, and (2) a decomposition fusion algorithm that selects the more effective feedback to explore the corresponding arms and randomly assigns one feedback type for exploration and the other for exploitation in each round. The elimination fusion experiences a suboptimal multiplicative term of the number of arms in regret due to the intrinsic suboptimality of dueling elimination. In contrast, the decomposition fusion achieves regret matching the lower bound up to a constant under a common assumption. Extensive experiments confirm the efficacy of our algorithms and theoretical results. 

**Abstract (ZH)**: 这篇论文探讨了在随机臂问题中绝对（奖励）反馈和相对（对弈）反馈的融合，每决策轮次收集两种类型的反馈。我们推导出了一个遗憾下界，表明高效算法为每个个体臂产生的遗憾可能是奖励或对弈基础上的较小者。我们提出了两种融合方法：（1）一种简单的消除融合算法，利用两种类型的反馈探索所有臂，并通过共享候选臂集统一收集的信息；（2）一种分解融合算法，根据反馈的有效性选择探索对应的臂，并在每轮次中随机分配一种反馈用于探索，另一种用于利分。消除融合因对弈消除的固有次优性而在遗憾中经历了臂数的次优乘性项。相比之下，分解融合在共同假设下实现了遗憾与下界在常数因子内的匹配。大量实验证明了我们算法和理论结果的有效性。 

---
# DAE-KAN: A Kolmogorov-Arnold Network Model for High-Index Differential-Algebraic Equations 

**Title (ZH)**: DAE-KAN：高阶微分代数方程的柯尔莫戈罗夫-阿诺尔德网络模型 

**Authors**: Kai Luo, Juan Tang, Mingchao Cai, Xiaoqing Zeng, Manqi Xie, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15806)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) have emerged as a promising alternative to Multi-Layer Perceptrons (MLPs) due to their superior function-fitting abilities in data-driven modeling. In this paper, we propose a novel framework, DAE-KAN, for solving high-index differential-algebraic equations (DAEs) by integrating KANs with Physics-Informed Neural Networks (PINNs). This framework not only preserves the ability of traditional PINNs to model complex systems governed by physical laws but also enhances their performance by leveraging the function-fitting strengths of KANs. Numerical experiments demonstrate that for DAE systems ranging from index-1 to index-3, DAE-KAN reduces the absolute errors of both differential and algebraic variables by 1 to 2 orders of magnitude compared to traditional PINNs. To assess the effectiveness of this approach, we analyze the drift-off error and find that both PINNs and DAE-KAN outperform classical numerical methods in controlling this phenomenon. Our results highlight the potential of neural network methods, particularly DAE-KAN, in solving high-index DAEs with substantial computational accuracy and generalization, offering a promising solution for challenging partial differential-algebraic equations. 

**Abstract (ZH)**: Kolmogorov-Arnold网络（KANs）作为多层感知机（MLPs）的有前景替代方案，由于其在数据驱动建模中的优越函数拟合能力而逐渐受到关注。本文提出了一种新的框架DAE-KAN，通过将KANs与物理感知神经网络（PINNs）结合起来，用于求解高索数微分代数方程（DAEs）。该框架不仅保留了传统PINNs模型复杂物理系统的能力，还通过利用KANs的函数拟合优势来提高其性能。数值实验表明，对于从索数1到索数3的DAE系统，相较于传统的PINNs，DAE-KAN可以将微分变量和代数变量的绝对误差降低1到2个数量级。通过分析偏差误差，我们发现无论是PINNs还是DAE-KAN均优于传统的数值方法。我们的结果突显了神经网络方法，特别是DAE-KAN，在解决具有高度计算准确性和泛化能力的高索数DAEs方面的潜力，为解决复杂的偏微分代数方程提供了一种有前景的解决方案。 

---
# Shannon invariants: A scalable approach to information decomposition 

**Title (ZH)**: 香农不变量：一种可扩展的信息分解方法 

**Authors**: Aaron J. Gutknecht, Fernando E. Rosas, David A. Ehrlich, Abdullah Makkeh, Pedro A. M. Mediano, Michael Wibral  

**Link**: [PDF](https://arxiv.org/pdf/2504.15779)  

**Abstract**: Distributed systems, such as biological and artificial neural networks, process information via complex interactions engaging multiple subsystems, resulting in high-order patterns with distinct properties across scales. Investigating how these systems process information remains challenging due to difficulties in defining appropriate multivariate metrics and ensuring their scalability to large systems. To address these challenges, we introduce a novel framework based on what we call "Shannon invariants" -- quantities that capture essential properties of high-order information processing in a way that depends only on the definition of entropy and can be efficiently calculated for large systems. Our theoretical results demonstrate how Shannon invariants can be used to resolve long-standing ambiguities regarding the interpretation of widely used multivariate information-theoretic measures. Moreover, our practical results reveal distinctive information-processing signatures of various deep learning architectures across layers, which lead to new insights into how these systems process information and how this evolves during training. Overall, our framework resolves fundamental limitations in analyzing high-order phenomena and offers broad opportunities for theoretical developments and empirical analyses. 

**Abstract (ZH)**: 分布系统，如生物和人工神经网络，通过多个子系统之间的复杂相互作用处理信息，产生具有不同尺度特性的高级模式。由于难以定义合适的多变量度量并确保其在大规模系统中的扩展性，研究这些系统如何处理信息仍然具有挑战性。为应对这些挑战，我们提出了一种基于我们称之为“香农不变量”的新型框架——这些量能够仅依赖于熵的定义来捕获高级信息处理的基本属性，并且能够高效地应用于大规模系统。我们的理论结果展示了如何使用香农不变量解决广泛使用的多变量信息论度量解释中的长期歧义。此外，我们的实证结果揭示了不同深度学习架构在各层之间的独特信息处理特征，这为理解这些系统如何处理信息及其在训练过程中如何演变提供了新的见解。总体而言，我们的框架解决了分析高级现象的基本局限性，并为理论发展和实证分析提供了广阔的空间。 

---
# Clifford Group Equivariant Diffusion Models for 3D Molecular Generation 

**Title (ZH)**: Clifford Group 等变扩散模型在三维分子生成中的应用 

**Authors**: Cong Liu, Sharvaree Vadgama, David Ruhe, Erik Bekkers, Patrick Forrè  

**Link**: [PDF](https://arxiv.org/pdf/2504.15773)  

**Abstract**: This paper explores leveraging the Clifford algebra's expressive power for $\E(n)$-equivariant diffusion models. We utilize the geometric products between Clifford multivectors and the rich geometric information encoded in Clifford subspaces in \emph{Clifford Diffusion Models} (CDMs). We extend the diffusion process beyond just Clifford one-vectors to incorporate all higher-grade multivector subspaces. The data is embedded in grade-$k$ subspaces, allowing us to apply latent diffusion across complete multivectors. This enables CDMs to capture the joint distribution across different subspaces of the algebra, incorporating richer geometric information through higher-order features. We provide empirical results for unconditional molecular generation on the QM9 dataset, showing that CDMs provide a promising avenue for generative modeling. 

**Abstract (ZH)**: 本文探索利用Clifford代数的表达能力构建$\E(n)$-对称扩散模型。我们利用Clifford多矢量间的几何积以及Clifford子空间中丰富的几何信息在Clifford扩散模型（CDMs）中加以利用。我们将扩散过程扩展到包含所有高阶多矢子空间，而不仅仅局限于Clifford一矢。数据被嵌入到阶-$k$子空间中，使得我们能够对完整的多矢量应用潜在扩散。这使得CDMs能够捕捉代数中不同子空间的联合分布，并通过高阶特征来包含更丰富的几何信息。我们在QM9数据集上的无条件分子生成实验结果表明，CDMs为生成模型提供了一个有前景的途径。 

---
# iMedic: Towards Smartphone-based Self-Auscultation Tool for AI-Powered Pediatric Respiratory Assessment 

**Title (ZH)**: iMedic: 基于智能手机的自我听诊工具，用于AI驱动的儿科呼吸评估 

**Authors**: Seung Gyu Jeong, Sung Woo Nam, Seong Kwan Jung, Seong-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.15743)  

**Abstract**: Respiratory auscultation is crucial for early detection of pediatric pneumonia, a condition that can quickly worsen without timely intervention. In areas with limited physician access, effective auscultation is challenging. We present a smartphone-based system that leverages built-in microphones and advanced deep learning algorithms to detect abnormal respiratory sounds indicative of pneumonia risk. Our end-to-end deep learning framework employs domain generalization to integrate a large electronic stethoscope dataset with a smaller smartphone-derived dataset, enabling robust feature learning for accurate respiratory assessments without expensive equipment. The accompanying mobile application guides caregivers in collecting high-quality lung sound samples and provides immediate feedback on potential pneumonia risks. User studies show strong classification performance and high acceptance, demonstrating the system's ability to facilitate proactive interventions and reduce preventable childhood pneumonia deaths. By seamlessly integrating into ubiquitous smartphones, this approach offers a promising avenue for more equitable and comprehensive remote pediatric care. 

**Abstract (ZH)**: 基于智能手机的内置麦克风和先进深度学习算法的呼吸音检测系统在儿科 pneumonia 早期检测中的应用：促进更具包容性和全面的远程儿科护理 

---
# Collaborative Split Federated Learning with Parallel Training and Aggregation 

**Title (ZH)**: 协作式分拆联邦学习与并行训练聚合 

**Authors**: Yiannis Papageorgiou, Yannis Thomas, Alexios Filippakopoulos, Ramin Khalili, Iordanis Koutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2504.15724)  

**Abstract**: Federated learning (FL) operates based on model exchanges between the server and the clients, and it suffers from significant client-side computation and communication burden. Split federated learning (SFL) arises a promising solution by splitting the model into two parts, that are trained sequentially: the clients train the first part of the model (client-side model) and transmit it to the server that trains the second (server-side model). Existing SFL schemes though still exhibit long training delays and significant communication overhead, especially when clients of different computing capability participate. Thus, we propose Collaborative-Split Federated Learning~(C-SFL), a novel scheme that splits the model into three parts, namely the model parts trained at the computationally weak clients, the ones trained at the computationally strong clients, and the ones at the server. Unlike existing works, C-SFL enables parallel training and aggregation of model's parts at the clients and at the server, resulting in reduced training delays and commmunication overhead while improving the model's accuracy. Experiments verify the multiple gains of C-SFL against the existing schemes. 

**Abstract (ZH)**: 协作分割联邦学习（C-SFL） 

---
# RePOPE: Impact of Annotation Errors on the POPE Benchmark 

**Title (ZH)**: RePOPE: 标注错误对POPE基准的影响 

**Authors**: Yannic Neuhaus, Matthias Hein  

**Link**: [PDF](https://arxiv.org/pdf/2504.15707)  

**Abstract**: Since data annotation is costly, benchmark datasets often incorporate labels from established image datasets. In this work, we assess the impact of label errors in MSCOCO on the frequently used object hallucination benchmark POPE. We re-annotate the benchmark images and identify an imbalance in annotation errors across different subsets. Evaluating multiple models on the revised labels, which we denote as RePOPE, we observe notable shifts in model rankings, highlighting the impact of label quality. Code and data are available at this https URL . 

**Abstract (ZH)**: 自标注数据成本高昂，基准数据集通常会采用已有的图像标注数据。在本文中，我们评估MSCOCO数据集中的标签错误对常用的目标 hallucination 基准POPE的影响。我们重新标注基准图像，并识别出不同子集间的标注错误不平衡现象。在使用修订后的标签（我们称之为RePOPE）评估多个模型后，我们观察到模型排名出现了显著变化，突显了标签质量的重要性。代码和数据可在以下链接获取。 

---
# FADEL: Uncertainty-aware Fake Audio Detection with Evidential Deep Learning 

**Title (ZH)**: FADEL：基于证据深度学习的不确定性感知假音频检测 

**Authors**: Ju Yeon Kang, Ji Won Yoon, Semin Kim, Min Hyun Han, Nam Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.15663)  

**Abstract**: Recently, fake audio detection has gained significant attention, as advancements in speech synthesis and voice conversion have increased the vulnerability of automatic speaker verification (ASV) systems to spoofing attacks. A key challenge in this task is generalizing models to detect unseen, out-of-distribution (OOD) attacks. Although existing approaches have shown promising results, they inherently suffer from overconfidence issues due to the usage of softmax for classification, which can produce unreliable predictions when encountering unpredictable spoofing attempts. To deal with this limitation, we propose a novel framework called fake audio detection with evidential learning (FADEL). By modeling class probabilities with a Dirichlet distribution, FADEL incorporates model uncertainty into its predictions, thereby leading to more robust performance in OOD scenarios. Experimental results on the ASVspoof2019 Logical Access (LA) and ASVspoof2021 LA datasets indicate that the proposed method significantly improves the performance of baseline models. Furthermore, we demonstrate the validity of uncertainty estimation by analyzing a strong correlation between average uncertainty and equal error rate (EER) across different spoofing algorithms. 

**Abstract (ZH)**: 近期，随着语音合成和声纹转换技术的进步，虚假音频检测受到了广泛关注，这使自动说话人验证（ASV）系统更容易受到欺骗攻击。这一任务中的一个关键挑战是使模型能够检测未见过的分布外（OOD）攻击。尽管现有方法展示了有前途的结果，但由于分类时使用了softmax，它们本质上会遭受过度自信问题，当遇到不可预测的欺骗尝试时会生成不可靠的预测。为解决这一局限，我们提出了一种名为证据学习虚假音频检测（FADEL）的新框架。通过使用狄利克雷分布建模类概率，FADEL 将模型不确定性纳入其预测中，从而在分布外场景中实现了更稳健的性能。在 ASVspoof2019 逻辑访问（LA）和 ASVspoof2021 逻辑访问（LA）数据集上的实验结果表明，所提出的方法显著改善了基线模型的性能。此外，我们通过分析平均不确定性与等错误率（EER）在不同欺骗算法之间的强相关性，验证了不确定性估计的有效性。 

---
# MetaMolGen: A Neural Graph Motif Generation Model for De Novo Molecular Design 

**Title (ZH)**: MetaMolGen：一种用于从头分子设计的神经图形动机生成模型 

**Authors**: Zimo Yan, Jie Zhang, Zheng Xie, Chang Liu, Yizhen Liu, Yiping Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.15587)  

**Abstract**: Molecular generation plays an important role in drug discovery and materials science, especially in data-scarce scenarios where traditional generative models often struggle to achieve satisfactory conditional generalization. To address this challenge, we propose MetaMolGen, a first-order meta-learning-based molecular generator designed for few-shot and property-conditioned molecular generation. MetaMolGen standardizes the distribution of graph motifs by mapping them to a normalized latent space, and employs a lightweight autoregressive sequence model to generate SMILES sequences that faithfully reflect the underlying molecular structure. In addition, it supports conditional generation of molecules with target properties through a learnable property projector integrated into the generative this http URL results demonstrate that MetaMolGen consistently generates valid and diverse SMILES sequences under low-data regimes, outperforming conventional baselines. This highlights its advantage in fast adaptation and efficient conditional generation for practical molecular design. 

**Abstract (ZH)**: 分子生成在药物发现和材料科学中发挥着重要作用，尤其是在数据稀缺场景中，传统生成模型往往难以实现满意的条件泛化。为 Address这一挑战，我们提出了一种名为MetaMolGen的一阶元学习分子生成器，该生成器适用于少样本和性质条件下的分子生成。MetaMolGen通过将图模式映射到标准化的潜在空间来标准化图模式的分布，并采用轻量级自回归序列模型生成能够忠实反映分子结构的SMILES序列。此外，它通过集成可学习的性质投影器支持具有目标性质的分子的条件生成。实验结果表明，在低数据条件下，MetaMolGen能够一致地生成有效且多样的SMILES序列，并优于常规基线。这突显了其在实际分子设计中的快速适应能力和高效条件生成的优势。 

---
# Do It For Me vs. Do It With Me: Investigating User Perceptions of Different Paradigms of Automation in Copilots for Feature-Rich Software 

**Title (ZH)**: 为我做 vs. 与我一起做：探究用户对功能丰富软件副驾驶中不同自动化范式的感知 

**Authors**: Anjali Khurana, Xiaotian Su, April Yi Wang, Parmit K Chilana  

**Link**: [PDF](https://arxiv.org/pdf/2504.15549)  

**Abstract**: Large Language Model (LLM)-based in-application assistants, or copilots, can automate software tasks, but users often prefer learning by doing, raising questions about the optimal level of automation for an effective user experience. We investigated two automation paradigms by designing and implementing a fully automated copilot (AutoCopilot) and a semi-automated copilot (GuidedCopilot) that automates trivial steps while offering step-by-step visual guidance. In a user study (N=20) across data analysis and visual design tasks, GuidedCopilot outperformed AutoCopilot in user control, software utility, and learnability, especially for exploratory and creative tasks, while AutoCopilot saved time for simpler visual tasks. A follow-up design exploration (N=10) enhanced GuidedCopilot with task-and state-aware features, including in-context preview clips and adaptive instructions. Our findings highlight the critical role of user control and tailored guidance in designing the next generation of copilots that enhance productivity, support diverse skill levels, and foster deeper software engagement. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的应用内辅助程序（或副驾）可以自动执行软件任务，但用户通常更喜欢边做边学，这引发了关于有效用户体验的最佳自动化水平的问题。我们通过设计和实现一个全自动副驾（AutoCopilot）和一个半自动副驾（GuidedCopilot）来研究两种自动化范式：GuidedCopilot在用户控制、软件实用性及易学习性方面优于AutoCopilot，尤其是在探索性和创造性任务中；而AutoCopilot在简单视觉任务中节省时间。后续的设计探索（N=10）增强了GuidedCopilot的功能，包括上下文相关预览片段和自适应指令。我们的研究结果强调了用户控制和定制化指导在设计下一代能够提升生产力、支持多样技能水平并促进更深层次软件参与度的副驾中的关键作用。 

---
# Transport f divergences 

**Title (ZH)**: 传输散度 

**Authors**: Wuchen Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.15515)  

**Abstract**: We define a class of divergences to measure differences between probability density functions in one-dimensional sample space. The construction is based on the convex function with the Jacobi operator of mapping function that pushforwards one density to the other. We call these information measures {\em transport $f$-divergences}. We present several properties of transport $f$-divergences, including invariances, convexities, variational formulations, and Taylor expansions in terms of mapping functions. Examples of transport $f$-divergences in generative models are provided. 

**Abstract (ZH)**: 我们定义了一类测度一维样本空间中概率密度函数之间差异的发散度。这种构造基于将一种密度映推至另一种的映射函数的Jacobi算子与凸函数相结合。我们将这类信息测度称为{\em 运输 $f$-发散度}。我们提出了运输 $f$-发散度的若干性质，包括不变性、凸性、变分形式以及与映射函数相关的泰勒展开。提供了生成模型中运输 $f$-发散度的例子。 

---
# Guillotine: Hypervisors for Isolating Malicious AIs 

**Title (ZH)**: Guillotine: 用于隔离恶意AI的虚拟机管理程序 

**Authors**: James Mickens, Sarah Radway, Ravi Netravali  

**Link**: [PDF](https://arxiv.org/pdf/2504.15499)  

**Abstract**: As AI models become more embedded in critical sectors like finance, healthcare, and the military, their inscrutable behavior poses ever-greater risks to society. To mitigate this risk, we propose Guillotine, a hypervisor architecture for sandboxing powerful AI models -- models that, by accident or malice, can generate existential threats to humanity. Although Guillotine borrows some well-known virtualization techniques, Guillotine must also introduce fundamentally new isolation mechanisms to handle the unique threat model posed by existential-risk AIs. For example, a rogue AI may try to introspect upon hypervisor software or the underlying hardware substrate to enable later subversion of that control plane; thus, a Guillotine hypervisor requires careful co-design of the hypervisor software and the CPUs, RAM, NIC, and storage devices that support the hypervisor software, to thwart side channel leakage and more generally eliminate mechanisms for AI to exploit reflection-based vulnerabilities. Beyond such isolation at the software, network, and microarchitectural layers, a Guillotine hypervisor must also provide physical fail-safes more commonly associated with nuclear power plants, avionic platforms, and other types of mission critical systems. Physical fail-safes, e.g., involving electromechanical disconnection of network cables, or the flooding of a datacenter which holds a rogue AI, provide defense in depth if software, network, and microarchitectural isolation is compromised and a rogue AI must be temporarily shut down or permanently destroyed. 

**Abstract (ZH)**: Guillotine：一种用于限制存在风险人工智能模型的虚拟化架构 

---
# Scalable APT Malware Classification via Parallel Feature Extraction and GPU-Accelerated Learning 

**Title (ZH)**: 基于并行特征提取和GPU加速学习的大规模APT恶意软件分类 

**Authors**: Noah Subedar, Taeui Kim, Saathwick Venkataramalingam  

**Link**: [PDF](https://arxiv.org/pdf/2504.15497)  

**Abstract**: This paper presents an underlying framework for both automating and accelerating malware classification, more specifically, mapping malicious executables to known Advanced Persistent Threat (APT) groups. The main feature of this analysis is the assembly-level instructions present in executables which are also known as opcodes. The collection of such opcodes on many malicious samples is a lengthy process; hence, open-source reverse engineering tools are used in tandem with scripts that leverage parallel computing to analyze multiple files at once. Traditional and deep learning models are applied to create models capable of classifying malware samples. One-gram and two-gram datasets are constructed and used to train models such as SVM, KNN, and Decision Tree; however, they struggle to provide adequate results without relying on metadata to support n-gram sequences. The computational limitations of such models are overcome with convolutional neural networks (CNNs) and heavily accelerated using graphical compute unit (GPU) resources. 

**Abstract (ZH)**: 本文提出了一种自动化和加速恶意软件分类的基础框架，更具体地说，是将恶意执行文件映射到已知的高级持久性威胁（APT）组。该分析的主要特征是存在于执行文件中的汇编级指令，也称为Opcode。从大量恶意样本中收集此类Opcode是一个耗时的过程；因此，使用开源逆向工程工具结合利用并行计算的脚本，可以一次性分析多个文件。传统和深度学习模型被应用于创建能够分类恶意软件样本的模型。构建了一gram和二gram数据集并用于训练SVM、KNN和决策树模型；然而，它们在不依赖元数据支持n-gram序列的情况下难以提供满意的成果。通过使用卷积神经网络（CNN）克服了这些模型的计算限制，并通过图形计算单元（GPU）资源极大地加速了计算。 

---
# On the Boolean Network Theory of Datalog$^\neg$ 

**Title (ZH)**: Datalog$^\neg$的布尔网络理论 

**Authors**: Van-Giang Trinh, Belaid Benhamou, Sylvain Soliman, François Fages  

**Link**: [PDF](https://arxiv.org/pdf/2504.15417)  

**Abstract**: Datalog$^\neg$ is a central formalism used in a variety of domains ranging from deductive databases and abstract argumentation frameworks to answer set programming. Its model theory is the finite counterpart of the logical semantics developed for normal logic programs, mainly based on the notions of Clark's completion and two-valued or three-valued canonical models including supported, stable, regular and well-founded models. In this paper we establish a formal link between Datalog$^\neg$ and Boolean network theory, which was initially introduced by Stuart Kaufman and René Thomas to reason about gene regulatory networks. We use previous results from Boolean network theory to prove that in the absence of odd cycles in a Datalog$^\neg$ program, the regular models coincide with the stable models, which entails the existence of stable models, and in the absence of even cycles, we show the uniqueness of stable partial models, which entails the uniqueness of regular models. These results on regular models have been claimed by You and Yuan in 1994 for normal logic programs but we show problems in their definition of well-founded stratification and in their proofs that we can fix for negative normal logic programs only. We also give upper bounds on the numbers of stable partial, regular, and stable models of a Datalog$^\neg$ program using the cardinality of a feedback vertex set in its atom dependency graph. Interestingly, our connection to Boolean network theory also points us to the notion of trap spaces for Datalog$^\neg$ programs. We relate the notions of supported or stable trap spaces to the other semantics of Datalog$^\neg$, and show the equivalence between subset-minimal stable trap spaces and regular models. 

**Abstract (ZH)**: Datalog\(^-\)与布尔网络理论的正式关联 

---
# Bayesian Federated Learning for Continual Training 

**Title (ZH)**: 贝叶斯联邦学习在连续训练中的应用 

**Authors**: Usevalad Milasheuski, Luca Barbieri, Sanaz Kianoush, Monica Nicoli, Stefano Savazzi  

**Link**: [PDF](https://arxiv.org/pdf/2504.15328)  

**Abstract**: Bayesian Federated Learning (BFL) enables uncertainty quantification and robust adaptation in distributed learning. In contrast to the frequentist approach, it estimates the posterior distribution of a global model, offering insights into model reliability. However, current BFL methods neglect continual learning challenges in dynamic environments where data distributions shift over time. We propose a continual BFL framework applied to human sensing with radar data collected over several days. Using Stochastic Gradient Langevin Dynamics (SGLD), our approach sequentially updates the model, leveraging past posteriors to construct the prior for the new tasks. We assess the accuracy, the expected calibration error (ECE) and the convergence speed of our approach against several baselines. Results highlight the effectiveness of continual Bayesian updates in preserving knowledge and adapting to evolving data. 

**Abstract (ZH)**: 贝叶斯联邦学习（BFL）在分布式学习中实现不确定性量化和鲁棒适应。与 frequentist 方法不同，它估计全局模型的后验分布，提供模型可靠性的见解。然而，当前的 BFL 方法忽视了动态环境中数据分布随时间变化的持续学习挑战。我们提出了一种适用于雷达数据收集多天的人体传感的持续 BFL 框架。利用随机梯度 Langevin 动力学（SGLD），我们的方法顺序更新模型，利用过去的后验分布构建新任务的先验。我们评估了我们的方法在准确度、预期校准误差（ECE）和收敛速度方面的表现，并与多种基线进行了比较。结果显示，持续的贝叶斯更新在保持知识和适应变化的数据方面具有有效性。 

---
# Significativity Indices for Agreement Values 

**Title (ZH)**: 协变量显著性指标 

**Authors**: Alberto Casagrande, Francesco Fabris, Rossano Girometti, Roberto Pagliarini  

**Link**: [PDF](https://arxiv.org/pdf/2504.15325)  

**Abstract**: Agreement measures, such as Cohen's kappa or intraclass correlation, gauge the matching between two or more classifiers. They are used in a wide range of contexts from medicine, where they evaluate the effectiveness of medical treatments and clinical trials, to artificial intelligence, where they can quantify the approximation due to the reduction of a classifier. The consistency of different classifiers to a golden standard can be compared simply by using the order induced by their agreement measure with respect to the golden standard itself. Nevertheless, labelling an approach as good or bad exclusively by using the value of an agreement measure requires a scale or a significativity index. Some quality scales have been proposed in the literature for Cohen's kappa, but they are mainly naive, and their boundaries are arbitrary. This work proposes a general approach to evaluate the significativity of any agreement value between two classifiers and introduces two significativity indices: one dealing with finite data sets, the other one handling classification probability distributions. Moreover, this manuscript considers the computational issues of evaluating such indices and identifies some efficient algorithms to evaluate them. 

**Abstract (ZH)**: 一致性度量，如科恩κ系数或内クラス相关系数，衡量两个或多个分类器之间的匹配程度。它们广泛应用于医学领域评估医疗治疗和临床试验的有效性，以及人工智能领域量化分类器减少后的接近程度。不同分类器与金标准的一致性可以通过各自的金标准诱导顺序进行简单比较。然而，仅根据一致性度量值将一种方法评估为良好或不良需要一个尺度或显著性指数。文献中为科恩κ系数提出了一些质量尺度，但它们主要是朴素的，其界限是任意的。本文提出了一种通用方法来评估任意两个分类器之间一致性值的显著性，并引入了两个显著性指标：一个适用于有限数据集，另一个处理分类概率分布。此外，本文考虑了评估这些指标的计算问题，并识别了一些高效的算法来评估它们。 

---
# A Graph Based Raman Spectral Processing Technique for Exosome Classification 

**Title (ZH)**: 基于图的拉曼光谱处理技术用于外泌体分类 

**Authors**: Vuong M. Ngo, Edward Bolger, Stan Goodwin, John O'Sullivan, Dinh Viet Cuong, Mark Roantree  

**Link**: [PDF](https://arxiv.org/pdf/2504.15324)  

**Abstract**: Exosomes are small vesicles crucial for cell signaling and disease biomarkers. Due to their complexity, an "omics" approach is preferable to individual biomarkers. While Raman spectroscopy is effective for exosome analysis, it requires high sample concentrations and has limited sensitivity to lipids and proteins. Surface-enhanced Raman spectroscopy helps overcome these challenges. In this study, we leverage Neo4j graph databases to organize 3,045 Raman spectra of exosomes, enhancing data generalization. To further refine spectral analysis, we introduce a novel spectral filtering process that integrates the PageRank Filter with optimal Dimensionality Reduction. This method improves feature selection, resulting in superior classification performance. Specifically, the Extra Trees model, using our spectral processing approach, achieves 0.76 and 0.857 accuracy in classifying hyperglycemic, hypoglycemic, and normal exosome samples based on Raman spectra and surface, respectively, with group 10-fold cross-validation. Our results show that graph-based spectral filtering combined with optimal dimensionality reduction significantly improves classification accuracy by reducing noise while preserving key biomarker signals. This novel framework enhances Raman-based exosome analysis, expanding its potential for biomedical applications, disease diagnostics, and biomarker discovery. 

**Abstract (ZH)**: 基于图数据库的光谱过滤与最优降维结合的exoRNA分析方法 

---
# HyperFlow: Gradient-Free Emulation of Few-Shot Fine-Tuning 

**Title (ZH)**: HyperFlow：无梯度Few-Shot微调仿真 

**Authors**: Donggyun Kim, Chanwoo Kim, Seunghoon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2504.15323)  

**Abstract**: While test-time fine-tuning is beneficial in few-shot learning, the need for multiple backpropagation steps can be prohibitively expensive in real-time or low-resource scenarios. To address this limitation, we propose an approach that emulates gradient descent without computing gradients, enabling efficient test-time adaptation. Specifically, we formulate gradient descent as an Euler discretization of an ordinary differential equation (ODE) and train an auxiliary network to predict the task-conditional drift using only the few-shot support set. The adaptation then reduces to a simple numerical integration (e.g., via the Euler method), which requires only a few forward passes of the auxiliary network -- no gradients or forward passes of the target model are needed. In experiments on cross-domain few-shot classification using the Meta-Dataset and CDFSL benchmarks, our method significantly improves out-of-domain performance over the non-fine-tuned baseline while incurring only 6\% of the memory cost and 0.02\% of the computation time of standard fine-tuning, thus establishing a practical middle ground between direct transfer and fully fine-tuned approaches. 

**Abstract (ZH)**: 虽然测试时微调在少样本学习中有益，但在实时或资源受限场景中进行多次反向传播步骤可能代价高昂。为了解决这一局限性，我们提出了一种无需计算梯度即可模拟梯度下降的方法，从而实现高效的测试时自适应。具体来说，我们将梯度下降公式化为常微分方程（ODE）的欧拉离散化形式，并训练一个辅助网络仅使用少样本支持集来预测任务条件下的漂移。随后的自适应过程简化为简单的数值积分（例如，通过欧拉方法），仅需辅助网络的几次前向传播——无需计算梯度或目标模型的前向传播。在使用Meta-Dataset和CDFSL基准进行跨域少样本分类实验中，我们的方法在不损失域外性能的前提下，内存成本仅占标准微调的6%，计算时间仅为标准微调的0.02%，从而在直接转移和完全微调方法之间建立了一个实用的中间立场。 

---
# How to systematically develop an effective AI-based bias correction model? 

**Title (ZH)**: 如何系统性地开发一种有效的AI偏差矫正模型？ 

**Authors**: Xiao Zhou, Yuze Sun, Jie Wu, Xiaomeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15322)  

**Abstract**: This study introduces ReSA-ConvLSTM, an artificial intelligence (AI) framework for systematic bias correction in numerical weather prediction (NWP). We propose three innovations by integrating dynamic climatological normalization, ConvLSTM with temporal causality constraints, and residual self-attention mechanisms. The model establishes a physics-aware nonlinear mapping between ECMWF forecasts and ERA5 reanalysis data. Using 41 years (1981-2021) of global atmospheric data, the framework reduces systematic biases in 2-m air temperature (T2m), 10-m winds (U10/V10), and sea-level pressure (SLP), achieving up to 20% RMSE reduction over 1-7 day forecasts compared to operational ECMWF outputs. The lightweight architecture (10.6M parameters) enables efficient generalization to multiple variables and downstream applications, reducing retraining time by 85% for cross-variable correction while improving ocean model skill through bias-corrected boundary conditions. The ablation experiments demonstrate that our innovations significantly improve the model's correction performance, suggesting that incorporating variable characteristics into the model helps enhance forecasting skills. 

**Abstract (ZH)**: 本研究介绍了一种用于数值天气预测系统偏差校正的人工智能（AI）框架ReSA-ConvLSTM。我们通过集成动态气候归一化、具有时间因果约束的ConvLSTM和剩余自注意力机制提出了三项创新。该模型在ECMWF预报和ERA5再分析数据之间建立了物理意识的非线性映射。利用1981-2021年41年的全球大气数据，该框架减少了2米气温（T2m）、10米风速（U10/V10）和海平面气压（SLP）的系统偏差，与ECMWF运行输出相比，在1-7天预报中最多可降低20%的RMSE。轻量级架构（10.6M参数）使其能够高效地应用于多个变量和下游应用，交叉变量校正值的重新训练时间减少85%，同时通过偏差校正的边界条件提高了海洋模型技能。消融实验表明，我们的创新显著提高了模型的校正性能，表明将变量特性纳入模型有助于增强预报技能。 

---
# Diffusion-Driven Inertial Generated Data for Smartphone Location Classification 

**Title (ZH)**: 由扩散驱动惯性生成的数据在智能手机位置分类中的应用 

**Authors**: Noa Cohen, Rotem Dror, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2504.15315)  

**Abstract**: Despite the crucial role of inertial measurements in motion tracking and navigation systems, the time-consuming and resource-intensive nature of collecting extensive inertial data has hindered the development of robust machine learning models in this field. In recent years, diffusion models have emerged as a revolutionary class of generative models, reshaping the landscape of artificial data generation. These models surpass generative adversarial networks and other state-of-the-art approaches to complex tasks. In this work, we propose diffusion-driven specific force-generated data for smartphone location recognition. We provide a comprehensive evaluation methodology by comparing synthetic and real recorded specific force data across multiple metrics. Our results demonstrate that our diffusion-based generative model successfully captures the distinctive characteristics of specific force signals across different smartphone placement conditions. Thus, by creating diverse, realistic synthetic data, we can reduce the burden of extensive data collection while providing high-quality training data for machine learning models. 

**Abstract (ZH)**: 尽管惯性测量在运动跟踪和导航系统中起着关键作用，但由于收集大量惯性数据耗时且资源密集，限制了该领域稳健机器学习模型的发展。近年来，扩散模型作为一种生成模型的新兴类别，正在重塑人工数据生成的格局。这些模型在复杂任务上超越了生成对抗网络和其他先进方法。在本文中，我们提出了一种基于扩散模型的特定加速度数据生成方法，用于智能手机位置识别。我们通过多个指标比较合成和实际记录的特定加速度数据，提供了全面的评估方法。结果表明，我们的基于扩散模型的生成模型成功捕捉了不同智能手机放置条件下特定加速度信号的独特特征。从而，通过生成多样化的现实合成数据，可以减轻大量数据收集的负担，同时为机器学习模型提供高质量的训练数据。 

---
# RINN: One Sample Radio Frequency Imaging based on Physics Informed Neural Network 

**Title (ZH)**: 物理导向神经网络下的单样本射频成像 

**Authors**: Fei Shang, Haohua Du, Dawei Yan, Panlong Yang, Xiang-Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.15311)  

**Abstract**: Due to its ability to work in non-line-of-sight and low-light environments, radio frequency (RF) imaging technology is expected to bring new possibilities for embodied intelligence and multimodal sensing. However, widely used RF devices (such as Wi-Fi) often struggle to provide high-precision electromagnetic measurements and large-scale datasets, hindering the application of RF imaging technology. In this paper, we combine the ideas of PINN to design the RINN network, using physical constraints instead of true value comparison constraints and adapting it with the characteristics of ubiquitous RF signals, allowing the RINN network to achieve RF imaging using only one sample without phase and with amplitude noise. Our numerical evaluation results show that compared with 5 classic algorithms based on phase data for imaging results, RINN's imaging results based on phaseless data are good, with indicators such as RRMSE (0.11) performing similarly well. RINN provides new possibilities for the universal development of radio frequency imaging technology. 

**Abstract (ZH)**: 基于物理约束的无相位RF成像网络(RINN)：一种新的射频成像技术发展方向 

---
# Power Transformer Health Index and Life Span Assessment: A Comprehensive Review of Conventional and Machine Learning based Approaches 

**Title (ZH)**: 电力变压器健康指数和寿命评估：传统方法与机器学习方法的综合评审 

**Authors**: Syeda Tahreem Zahra, Syed Kashif Imdad, Sohail Khan, Sohail Khalid, Nauman Anwar Baig  

**Link**: [PDF](https://arxiv.org/pdf/2504.15310)  

**Abstract**: Power transformers play a critical role within the electrical power system, making their health assessment and the prediction of their remaining lifespan paramount for the purpose of ensuring efficient operation and facilitating effective maintenance planning. This paper undertakes a comprehensive examination of existent literature, with a primary focus on both conventional and cutting-edge techniques employed within this domain. The merits and demerits of recent methodologies and techniques are subjected to meticulous scrutiny and explication. Furthermore, this paper expounds upon intelligent fault diagnosis methodologies and delves into the most widely utilized intelligent algorithms for the assessment of transformer conditions. Diverse Artificial Intelligence (AI) approaches, including Artificial Neural Networks (ANN) and Convolutional Neural Network (CNN), Support Vector Machine (SVM), Random Forest (RF), Genetic Algorithm (GA), and Particle Swarm Optimization (PSO), are elucidated offering pragmatic solutions for enhancing the performance of transformer fault diagnosis. The amalgamation of multiple AI methodologies and the exploration of timeseries analysis further contribute to the augmentation of diagnostic precision and the early detection of faults in transformers. By furnishing a comprehensive panorama of AI applications in the field of transformer fault diagnosis, this study lays the groundwork for future research endeavors and the progression of this critical area of study. 

**Abstract (ZH)**: 电力变压器在电力系统中扮演着关键角色，因此对其健康状况评估和剩余寿命预测至关重要，以确保系统高效运行并促进有效维护规划。本文综述了该领域的现有文献，主要关注传统和前沿技术，并详细探讨了这些方法和技术的优缺点。此外，本文还阐述了智能故障诊断方法，并深入探讨了最常用的智能算法以评估变压器状况。文中详细解释了包括人工神经网络（ANN）、卷积神经网络（CNN）、支持向量机（SVM）、随机森林（RF）、遗传算法（GA）和粒子 swarm 优化（PSO）等多种人工智能方法，提供了提高变压器故障诊断性能的实用解决方案。结合多种人工智能方法的综合应用和时间序列分析的探索，进一步提高了诊断精度并促进了变压器早期故障检测。通过全面概述变压器故障诊断领域中人工智能的应用，本文为未来的研究奠定了基础，并推动了该关键领域的进步。 

---
# A biologically Inspired Trust Model for Open Multi-Agent Systems that is Resilient to Rapid Performance Fluctuations 

**Title (ZH)**: 受生物启发的 resilient 快速性能波动抵御型开放多智能体系统信任模型 

**Authors**: Zoi Lygizou, Dimitris Kalles  

**Link**: [PDF](https://arxiv.org/pdf/2504.15301)  

**Abstract**: Trust management provides an alternative solution for securing open, dynamic, and distributed multi-agent systems, where conventional cryptographic methods prove to be impractical. However, existing trust models face challenges related to agent mobility, changing behaviors, and the cold start problem. To address these issues we introduced a biologically inspired trust model in which trustees assess their own capabilities and store trust data locally. This design improves mobility support, reduces communication overhead, resists disinformation, and preserves privacy. Despite these advantages, prior evaluations revealed limitations of our model in adapting to provider population changes and continuous performance fluctuations. This study proposes a novel algorithm, incorporating a self-classification mechanism for providers to detect performance drops potentially harmful for the service consumers. Simulation results demonstrate that the new algorithm outperforms its original version and FIRE, a well-known trust and reputation model, particularly in handling dynamic trustee behavior. While FIRE remains competitive under extreme environmental changes, the proposed algorithm demonstrates greater adaptability across various conditions. In contrast to existing trust modeling research, this study conducts a comprehensive evaluation of our model using widely recognized trust model criteria, assessing its resilience against common trust-related attacks while identifying strengths, weaknesses, and potential countermeasures. Finally, several key directions for future research are proposed. 

**Abstract (ZH)**: 生物启发的信任管理提供了一种替代方案，用于保障开放、动态和分布式多代理人系统，而传统的加密方法在此场景下变得不切实际。然而，现有的信任模型面临着代理移动性、行为变化以及冷启动问题的挑战。为应对这些问题，我们引入了一种生物启发的信任模型，其中信任者评估自身能力并本地存储信任数据。该设计提高了移动性支持，减少了通信开销，抵御了误导信息，并保护了隐私。尽管具有这些优点，之前的评估显示，该模型在适应提供商群体变化以及持续性能波动方面存在限制。本研究提出了一种新的算法，结合了提供者自我分类机制以检测可能对服务消费者有害的服务性能下降。仿真结果表明，新算法在处理动态信任者行为方面优于原有版本和FIRE，这是一种广为人知的信任和声誉模型。尽管FIRE在极端环境变化下仍具有竞争力，但所提算法在各种条件下表现出更大的适应性。与现有的信任建模研究不同，本研究全面评估了我们模型，使用广为人接受的信任模型标准，评估其对常见信任相关攻击的抵抗力，同时识别其优势、弱点及潜在对策。最后，提出了若干对未来研究的关键方向。 

---
# Scalability Optimization in Cloud-Based AI Inference Services: Strategies for Real-Time Load Balancing and Automated Scaling 

**Title (ZH)**: 基于云的AI推理服务的可扩展性优化：实时负载均衡与自动扩展策略 

**Authors**: Yihong Jin, Ze Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15296)  

**Abstract**: The rapid expansion of AI inference services in the cloud necessitates a robust scalability solution to manage dynamic workloads and maintain high performance. This study proposes a comprehensive scalability optimization framework for cloud AI inference services, focusing on real-time load balancing and autoscaling strategies. The proposed model is a hybrid approach that combines reinforcement learning for adaptive load distribution and deep neural networks for accurate demand forecasting. This multi-layered approach enables the system to anticipate workload fluctuations and proactively adjust resources, ensuring maximum resource utilisation and minimising latency. Furthermore, the incorporation of a decentralised decision-making process within the model serves to enhance fault tolerance and reduce response time in scaling operations. Experimental results demonstrate that the proposed model enhances load balancing efficiency by 35\ and reduces response delay by 28\, thereby exhibiting a substantial optimization effect in comparison with conventional scalability solutions. 

**Abstract (ZH)**: 云AI推理服务的快速扩展 necessitates 一个 robust 可扩展性解决方案 以管理动态工作负载并保持高性能。本研究提出了一种全面的可扩展性优化框架，专注于实时负载平衡和自动扩展策略。所提出的方法是一种混合方法，结合了强化学习以实现自适应负载分配和深度神经网络以实现准确的需求预测。多层方法使系统能够预见工作负载波动并主动调整资源，确保最大资源利用率并最小化延迟。此外，模型中的去中心化决策过程提高了容错性并减少了缩放操作的响应时间。实验结果表明，所提模型通过提高 35\% 的负载平衡效率并减少 28\% 的响应延迟，相比传统可扩展性解决方案表现出显著的优化效果。 

---
