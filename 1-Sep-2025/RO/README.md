# Can a mobile robot learn from a pedestrian model to prevent the sidewalk salsa? 

**Title (ZH)**: 移动机器人能否从行人模型中学习以防止人行道桑巴舞？ 

**Authors**: Olger Siebinga, David Abbink  

**Link**: [PDF](https://arxiv.org/pdf/2508.21690)  

**Abstract**: Pedestrians approaching each other on a sidewalk sometimes end up in an awkward interaction known as the "sidewalk salsa": they both (repeatedly) deviate to the same side to avoid a collision. This provides an interesting use case to study interactions between pedestrians and mobile robots because, in the vast majority of cases, this phenomenon is avoided through a negotiation based on implicit communication. Understanding how it goes wrong and how pedestrians end up in the sidewalk salsa will therefore provide insight into the implicit communication. This understanding can be used to design safe and acceptable robotic behaviour. In a previous attempt to gain this understanding, a model of pedestrian behaviour based on the Communication-Enabled Interaction (CEI) framework was developed that can replicate the sidewalk salsa. However, it is unclear how to leverage this model in robotic planning and decision-making since it violates the assumptions of game theory, a much-used framework in planning and decision-making. Here, we present a proof-of-concept for an approach where a Reinforcement Learning (RL) agent leverages the model to learn how to interact with pedestrians. The results show that a basic RL agent successfully learned to interact with the CEI model. Furthermore, a risk-averse RL agent that had access to the perceived risk of the CEI model learned how to effectively communicate its intention through its motion and thereby substantially lowered the perceived risk, and displayed effort by the modelled pedestrian. These results show this is a promising approach and encourage further exploration. 

**Abstract (ZH)**: 基于行人行为模型的强化学习方法探究：从侧面瓦萨尔到安全机器人交互 

---
# Robust Convex Model Predictive Control with collision avoidance guarantees for robot manipulators 

**Title (ZH)**: 具有碰撞避免保证的鲁棒凸模型预测控制机器人 manipulator 

**Authors**: Bernhard Wullt, Johannes Köhler, Per Mattsson, Mikeal Norrlöf, Thomas B. Schön  

**Link**: [PDF](https://arxiv.org/pdf/2508.21677)  

**Abstract**: Industrial manipulators are normally operated in cluttered environments, making safe motion planning important. Furthermore, the presence of model-uncertainties make safe motion planning more difficult. Therefore, in practice the speed is limited in order to reduce the effect of disturbances. There is a need for control methods that can guarantee safe motions that can be executed fast. We address this need by suggesting a novel model predictive control (MPC) solution for manipulators, where our two main components are a robust tube MPC and a corridor planning algorithm to obtain collision-free motion. Our solution results in a convex MPC, which we can solve fast, making our method practically useful. We demonstrate the efficacy of our method in a simulated environment with a 6 DOF industrial robot operating in cluttered environments with uncertainties in model parameters. We outperform benchmark methods, both in terms of being able to work under higher levels of model uncertainties, while also yielding faster motion. 

**Abstract (ZH)**: 工业 manipulator 在复杂环境中的安全运动规划至关重要，模型不确定性进一步增加了这一挑战。为保证快速执行的安全运动，我们提出了一种新型的模型预测控制（MPC）方法，该方法结合了鲁棒管型 MPC 和走廊规划算法以获得无碰撞路径。我们的解决方案形成了一个凸型 MPC，可以通过快速求解，使该方法具有实际应用价值。我们在具有模型参数不确定性且环境复杂的 6 自由度工业机器人仿真实验中展示了该方法的有效性，并在应对更高水平的模型不确定性方面优于基准方法，同时实现了更快的运动。 

---
# The Rosario Dataset v2: Multimodal Dataset for Agricultural Robotics 

**Title (ZH)**: Rosario 数据集 v2：农业机器人多模态数据集 

**Authors**: Nicolas Soncini, Javier Cremona, Erica Vidal, Maximiliano García, Gastón Castro, Taihú Pire  

**Link**: [PDF](https://arxiv.org/pdf/2508.21635)  

**Abstract**: We present a multi-modal dataset collected in a soybean crop field, comprising over two hours of recorded data from sensors such as stereo infrared camera, color camera, accelerometer, gyroscope, magnetometer, GNSS (Single Point Positioning, Real-Time Kinematic and Post-Processed Kinematic), and wheel odometry. This dataset captures key challenges inherent to robotics in agricultural environments, including variations in natural lighting, motion blur, rough terrain, and long, perceptually aliased sequences. By addressing these complexities, the dataset aims to support the development and benchmarking of advanced algorithms for localization, mapping, perception, and navigation in agricultural robotics. The platform and data collection system is designed to meet the key requirements for evaluating multi-modal SLAM systems, including hardware synchronization of sensors, 6-DOF ground truth and loops on long trajectories.
We run multimodal state-of-the art SLAM methods on the dataset, showcasing the existing limitations in their application on agricultural settings. The dataset and utilities to work with it are released on this https URL. 

**Abstract (ZH)**: 我们呈现了一个在大豆田地中收集的多模态数据集，包含超过两小时的传感器记录数据，这些传感器包括立体红外相机、彩色相机、加速度计、陀螺仪、磁力计、GNSS（单点定位、实时动态和后处理动态）、以及车轮里程计。该数据集捕捉了农业环境中机器人技术固有的关键挑战，包括自然光照变化、运动模糊、崎岖地形以及长时间的知觉 alias 序列。通过解决这些复杂性，该数据集旨在支持用于定位、制图、感知和导航的农业机器人高级算法的研发和基准测试。该平台和数据采集系统设计用于评估多模态 SLAM 系统的关键需求，包括传感器的硬件同步、6-DOF 地面真实值和长轨迹上的闭环。

我们在该数据集上运行了多模态最先进的 SLAM 方法，展示了其在农业环境应用中的现有局限性。数据集及其相关工具在此 https://链接 释放。 

---
# Learning Agile Gate Traversal via Analytical Optimal Policy Gradient 

**Title (ZH)**: 基于分析最优策略梯度的敏捷门电路遍历学习 

**Authors**: Tianchen Sun, Bingheng Wang, Longbin Tang, Yichao Gao, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.21592)  

**Abstract**: Traversing narrow gates presents a significant challenge and has become a standard benchmark for evaluating agile and precise quadrotor flight. Traditional modularized autonomous flight stacks require extensive design and parameter tuning, while end-to-end reinforcement learning (RL) methods often suffer from low sample efficiency and limited interpretability. In this work, we present a novel hybrid framework that adaptively fine-tunes model predictive control (MPC) parameters online using outputs from a neural network (NN) trained offline. The NN jointly predicts a reference pose and cost-function weights, conditioned on the coordinates of the gate corners and the current drone state. To achieve efficient training, we derive analytical policy gradients not only for the MPC module but also for an optimization-based gate traversal detection module. Furthermore, we introduce a new formulation of the attitude tracking error that admits a simplified representation, facilitating effective learning with bounded gradients. Hardware experiments demonstrate that our method enables fast and accurate quadrotor traversal through narrow gates in confined environments. It achieves several orders of magnitude improvement in sample efficiency compared to naive end-to-end RL approaches. 

**Abstract (ZH)**: 穿越狭窄通道对四旋翼飞行器构成显著挑战，并已成为评估灵活精准四旋翼飞行性能的标准基准。传统的模块化自主飞行栈需要 extensive 设计和参数调整，而端到端的强化学习方法往往样本效率低且可解释性有限。在本工作中，我们提出了一种新颖的混合框架，该框架在线自适应地通过神经网络（NN）的输出微调模型预测控制（MPC）参数，该神经网络预先训练以根据通道角坐标和当前无人机状态预测参考姿态和成本函数权重。为了实现高效训练，我们不仅为 MPC 模块推导了分析性的策略梯度，还为基于优化的目标通道穿越检测模块推导了分析性的策略梯度。此外，我们引入了一种新的姿态跟踪误差表示法，便于有效学习并限定了梯度。硬件实验结果表明，本方法使四旋翼飞行器能够在受限环境中快速准确地穿越狭窄通道，相比朴素的端到端 RL 方法，样本效率提高了几个数量级。 

---
# Estimated Informed Anytime Search for Sampling-Based Planning via Adaptive Sampler 

**Title (ZH)**: 基于自适应采样的估计知情任意时间搜索采样基于规划算法 

**Authors**: Liding Zhang, Kuanqi Cai, Yu Zhang, Zhenshan Bing, Chaoqun Wang, Fan Wu, Sami Haddadin, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2508.21549)  

**Abstract**: Path planning in robotics often involves solving continuously valued, high-dimensional problems. Popular informed approaches include graph-based searches, such as A*, and sampling-based methods, such as Informed RRT*, which utilize informed set and anytime strategies to expedite path optimization incrementally. Informed sampling-based planners define informed sets as subsets of the problem domain based on the current best solution cost. However, when no solution is found, these planners re-sample and explore the entire configuration space, which is time-consuming and computationally expensive. This article introduces Multi-Informed Trees (MIT*), a novel planner that constructs estimated informed sets based on prior admissible solution costs before finding the initial solution, thereby accelerating the initial convergence rate. Moreover, MIT* employs an adaptive sampler that dynamically adjusts the sampling strategy based on the exploration process. Furthermore, MIT* utilizes length-related adaptive sparse collision checks to guide lazy reverse search. These features enhance path cost efficiency and computation times while ensuring high success rates in confined scenarios. Through a series of simulations and real-world experiments, it is confirmed that MIT* outperforms existing single-query, sampling-based planners for problems in R^4 to R^16 and has been successfully applied to real-world robot manipulation tasks. A video showcasing our experimental results is available at: this https URL 

**Abstract (ZH)**: 机器人路径规划经常涉及到连续值的高维问题。流行的启发式方法包括基于图的搜索，如A*，以及基于采样的方法，如Informed RRT*，这些方法利用启发式集合和即时策略来逐步加速路径优化。启发式基于采样规划器根据当前最佳解的成本定义启发式集合。然而，当未找到解时，这些规划器需要重新采样并探索整个配置空间，这既耗时又计算成本高。本文介绍了一种新的规划器——多启发式树（MIT*），它在找到初始解之前基于先前可接纳解的成本估计启发式集合，从而加速初始收敛率。此外，MIT*采用了一种自适应采样器，根据探索过程动态调整采样策略。此外，MIT*利用与长度相关的自适应稀疏碰撞检查来引导懒惰的反向搜索。这些功能在确保在受限场景中的高成功率的同时，提高了路径成本效率和计算时间。通过一系列仿真和现实世界的实验，验证了MIT*在从R^4到R^16的问题中优于现有的单查询采样基于规划器，并成功应用于实际的机器人操作任务。我们的实验结果演示视频可在以下链接查看：this https URL。 

---
# Few-Shot Neuro-Symbolic Imitation Learning for Long-Horizon Planning and Acting 

**Title (ZH)**: 少量样本神经符号模仿学习：长时规划与执行 

**Authors**: Pierrick Lorang, Hong Lu, Johannes Huemer, Patrik Zips, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2508.21501)  

**Abstract**: Imitation learning enables intelligent systems to acquire complex behaviors with minimal supervision. However, existing methods often focus on short-horizon skills, require large datasets, and struggle to solve long-horizon tasks or generalize across task variations and distribution shifts. We propose a novel neuro-symbolic framework that jointly learns continuous control policies and symbolic domain abstractions from a few skill demonstrations. Our method abstracts high-level task structures into a graph, discovers symbolic rules via an Answer Set Programming solver, and trains low-level controllers using diffusion policy imitation learning. A high-level oracle filters task-relevant information to focus each controller on a minimal observation and action space. Our graph-based neuro-symbolic framework enables capturing complex state transitions, including non-spatial and temporal relations, that data-driven learning or clustering techniques often fail to discover in limited demonstration datasets. We validate our approach in six domains that involve four robotic arms, Stacking, Kitchen, Assembly, and Towers of Hanoi environments, and a distinct Automated Forklift domain with two environments. The results demonstrate high data efficiency with as few as five skill demonstrations, strong zero- and few-shot generalizations, and interpretable decision making. 

**Abstract (ZH)**: 模仿学习使智能系统能够在最少监督的情况下获得复杂行为。然而，现有方法往往专注于短期技能，需要大量数据集，并且难以解决长期任务或在任务变化和分布偏移的情况下泛化。我们提出了一种新的神经符号框架，该框架可以从少数技能示范中联合学习连续控制策略和符号领域抽象。该方法将高层任务结构抽象为图，并通过Answer Set Programming求解器发现符号规则，使用弥散策略模仿学习训练低层控制器。高层先验过滤任务相关信息，使每个控制器专注于最小的观测和行动空间。基于图的神经符号框架能够捕捉复杂的状态转换，包括非空间关系和时间关系，而数据驱动的学习或聚类技术在有限的示范数据集中往往难以发现这些关系。我们在涉及四个机器人手臂、堆积、厨房、装配和汉诺塔环境以及一个独特的自动叉车领域的六个领域中验证了我们的方法。结果表明，即使只有五个技能示范，也能实现高数据效率、强大的零样本和少样本泛化以及可解释的决策制定。 

---
# Assessing Human Cooperation for Enhancing Social Robot Navigation 

**Title (ZH)**: 评估人类合作以提高社会机器人导航能力 

**Authors**: Hariharan Arunachalam, Phani Teja Singamaneni, Rachid Alami  

**Link**: [PDF](https://arxiv.org/pdf/2508.21455)  

**Abstract**: Socially aware robot navigation is a planning paradigm where the robot navigates in human environments and tries to adhere to social constraints while interacting with the humans in the scene. These navigation strategies were further improved using human prediction models, where the robot takes the potential future trajectory of humans while computing its own. Though these strategies significantly improve the robot's behavior, it faces difficulties from time to time when the human behaves in an unexpected manner. This happens as the robot fails to understand human intentions and cooperativeness, and the human does not have a clear idea of what the robot is planning to do. In this paper, we aim to address this gap through effective communication at an appropriate time based on a geometric analysis of the context and human cooperativeness in head-on crossing scenarios. We provide an assessment methodology and propose some evaluation metrics that could distinguish a cooperative human from a non-cooperative one. Further, we also show how geometric reasoning can be used to generate appropriate verbal responses or robot actions. 

**Abstract (ZH)**: 基于社会意识的机器人导航是一种规划范式，机器人在人类环境中导航并尝试在与场景中的人类互动时遵守社会约束。通过使用人类预测模型进一步改进了这些导航策略，机器人在计算自身路径时会考虑到未来人类的潜在轨迹。尽管这些策略显著提高了机器人的行为表现，但它在人类表现出意外行为时有时会遇到困难。这种情况的发生是因为机器人无法理解人类的意图和合作性，而人类也不清楚机器人计划做什么。本文旨在通过基于几何分析的上下文和人类合作性进行有效沟通来解决这一问题，特别是在迎面 crossing 情景下。我们提供了一种评估方法并提出了一些评价指标，以区分合作的人类和不合作的人类。此外，我们还展示了如何通过几何推理生成适当的口头回应或机器人动作。 

---
# RoboInspector: Unveiling the Unreliability of Policy Code for LLM-enabled Robotic Manipulation 

**Title (ZH)**: RoboInspector: 揭示基于LLM的机器人操作中策略代码的不可靠性 

**Authors**: Chenduo Ying, Linkang Du, Peng Cheng, Yuanchao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21378)  

**Abstract**: Large language models (LLMs) demonstrate remarkable capabilities in reasoning and code generation, enabling robotic manipulation to be initiated with just a single instruction. The LLM carries out various tasks by generating policy code required to control the robot. Despite advances in LLMs, achieving reliable policy code generation remains a significant challenge due to the diverse requirements of real-world tasks and the inherent complexity of user instructions. In practice, different users may provide distinct instructions to drive the robot for the same task, which may cause the unreliability of policy code generation. To bridge this gap, we design RoboInspector, a pipeline to unveil and characterize the unreliability of the policy code for LLM-enabled robotic manipulation from two perspectives: the complexity of the manipulation task and the granularity of the instruction. We perform comprehensive experiments with 168 distinct combinations of tasks, instructions, and LLMs in two prominent frameworks. The RoboInspector identifies four main unreliable behaviors that lead to manipulation failure. We provide a detailed characterization of these behaviors and their underlying causes, giving insight for practical development to reduce unreliability. Furthermore, we introduce a refinement approach guided by failure policy code feedback that improves the reliability of policy code generation by up to 35% in LLM-enabled robotic manipulation, evaluated in both simulation and real-world environments. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推理和代码生成方面的卓越能力使机器人操作只需一个指令即可启动。LLMs通过生成控制机器人的策略代码来执行各种任务。尽管在LLMs方面取得了进展，但由于实际任务的多样性和用户指令的内在复杂性，实现可靠的策略代码生成仍然是一项重大挑战。实践中，不同用户可能为同一任务提供不同的指令，这可能导致策略代码生成的不可靠性。为弥补这一差距，我们设计了RoboInspector，这是一种管道，从操作任务的复杂性和指令的粒度两个视角揭示和表征LLM赋能机器人操作中策略代码的不可靠性。我们在两个主流框架中进行了全面实验，涉及168种不同的任务、指令和LLM组合。RoboInspector识别出四种主要的不可靠行为，导致操作失败。我们详细描述了这些行为及其根本原因，为实际开发降低不可靠性提供了见解。此外，我们提出了一种由失败策略代码反馈引导的改进方法，在LLM赋能的机器人操作中提高了策略代码生成的可靠性最高可达35%，并在模拟和实际环境中进行了评估。 

---
# Dynamics-Compliant Trajectory Diffusion for Super-Nominal Payload Manipulation 

**Title (ZH)**: 符合动力学要求的超额定载荷轨迹扩散 

**Authors**: Anuj Pasricha, Joewie Koh, Jay Vakil, Alessandro Roncone  

**Link**: [PDF](https://arxiv.org/pdf/2508.21375)  

**Abstract**: Nominal payload ratings for articulated robots are typically derived from worst-case configurations, resulting in uniform payload constraints across the entire workspace. This conservative approach severely underutilizes the robot's inherent capabilities -- our analysis demonstrates that manipulators can safely handle payloads well above nominal capacity across broad regions of their workspace while staying within joint angle, velocity, acceleration, and torque limits. To address this gap between assumed and actual capability, we propose a novel trajectory generation approach using denoising diffusion models that explicitly incorporates payload constraints into the planning process. Unlike traditional sampling-based methods that rely on inefficient trial-and-error, optimization-based methods that are prohibitively slow, or kinodynamic planners that struggle with problem dimensionality, our approach generates dynamically feasible joint-space trajectories in constant time that can be directly executed on physical hardware without post-processing. Experimental validation on a 7 DoF Franka Emika Panda robot demonstrates that up to 67.6% of the workspace remains accessible even with payloads exceeding 3 times the nominal capacity. This expanded operational envelope highlights the importance of a more nuanced consideration of payload dynamics in motion planning algorithms. 

**Abstract (ZH)**: articulated 机器人的名义载荷评级通常基于最坏情况配置得出，导致整个工作空间内的载荷限制均匀分布。这种保守的方法严重低估了机器人的固有能力——我们的分析表明，当处于关节角度、速度、加速度和扭矩限制范围内时，操作器可以在其工作空间的广大区域中安全地处理远超名义容量的载荷。为了弥补假设能力和实际能力之间的差距，我们提出了一种使用去噪扩散模型的新型轨迹生成方法，该方法在规划过程中明确包含了载荷限制。不同于依赖低效试错的基于采样的方法、难以实现优化的优化方法，或在处理问题维度时挣扎的运动动力学规划器，我们的方法可以在常数时间内生成动态可行的关节空间轨迹，可以直接在物理硬件上执行而不需要后处理。在7自由度的Franka Emika Panda机器人上的实验验证表明，即使载荷超过名义容量的3倍，仍有高达67.6%的工作空间保持可访问。这种扩展的工作空间范围突显了在运动规划算法中更细致地考虑载荷动力学的重要性。 

---
# Multi-Modal Model Predictive Path Integral Control for Collision Avoidance 

**Title (ZH)**: 多模态模型预测路径积分控制以实现避障 

**Authors**: Alberto Bertipaglia, Dariu M. Gavrila, Barys Shyrokau  

**Link**: [PDF](https://arxiv.org/pdf/2508.21364)  

**Abstract**: This paper proposes a novel approach to motion planning and decision-making for automated vehicles, using a multi-modal Model Predictive Path Integral control algorithm. The method samples with Sobol sequences around the prior input and incorporates analytical solutions for collision avoidance. By leveraging multiple modes, the multi-modal control algorithm explores diverse trajectories, such as manoeuvring around obstacles or stopping safely before them, mitigating the risk of sub-optimal solutions. A non-linear single-track vehicle model with a Fiala tyre serves as the prediction model, and tyre force constraints within the friction circle are enforced to ensure vehicle stability during evasive manoeuvres. The optimised steering angle and longitudinal acceleration are computed to generate a collision-free trajectory and to control the vehicle. In a high-fidelity simulation environment, we demonstrate that the proposed algorithm can successfully avoid obstacles, keeping the vehicle stable while driving a double lane change manoeuvre on high and low-friction road surfaces and occlusion scenarios with moving obstacles, outperforming a standard Model Predictive Path Integral approach. 

**Abstract (ZH)**: 本文提出了一种用于自动车辆运动规划和决策的新方法，采用多模式模型预测路径积分控制算法。该方法利用Sobol序列对先验输入进行采样，并结合碰撞避免的解析解。通过利用多种模式，多模式控制算法探索多样化的轨迹，如绕过障碍物或在障碍物前安全停车，从而减轻次优解的风险。采用带有Fiala轮胎的非线性单通道车辆模型作为预测模型，在摩擦圆内的轮胎力约束确保了在避让机动过程中车辆的稳定性。优化的转向角和纵向加速度被计算以生成无碰撞轨迹并控制车辆。在高保真仿真环境中，证明了所提出的算法能够成功避开障碍物，在高摩擦和低摩擦路面以及移动障碍物遮挡场景下进行双车道变换机动时保持车辆稳定，且表现优于标准的模型预测路径积分方法。 

---
# Robust Real-Time Coordination of CAVs: A Distributed Optimization Framework under Uncertainty 

**Title (ZH)**: 具有不确定性条件下分布式优化框架的 robust 实时协调 of CAVs 

**Authors**: Haojie Bai, Yang Wang, Cong Guo, Xiongwei Zhao, Hai Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21322)  

**Abstract**: Achieving both safety guarantees and real-time performance in cooperative vehicle coordination remains a fundamental challenge, particularly in dynamic and uncertain environments. This paper presents a novel coordination framework that resolves this challenge through three key innovations: 1) direct control of vehicles' trajectory distributions during coordination, formulated as a robust cooperative planning problem with adaptive enhanced safety constraints, ensuring a specified level of safety regarding the uncertainty of the interactive trajectory, 2) a fully parallel ADMM-based distributed trajectory negotiation (ADMM-DTN) algorithm that efficiently solves the optimization problem while allowing configurable negotiation rounds to balance solution quality and computational resources, and 3) an interactive attention mechanism that selectively focuses on critical interactive participants to further enhance computational efficiency. Both simulation results and practical experiments demonstrate that our framework achieves significant advantages in safety (reducing collision rates by up to 40.79\% in various scenarios) and real-time performance compared to state-of-the-art methods, while maintaining strong scalability with increasing vehicle numbers. The proposed interactive attention mechanism further reduces the computational demand by 14.1\%. The framework's effectiveness is further validated through real-world experiments with unexpected dynamic obstacles, demonstrating robust coordination in complex environments. The experiment demo could be found at this https URL. 

**Abstract (ZH)**: 在合作车辆协调中同时实现安全保证和实时性能仍然是一个基础挑战，尤其是在动态和不确定的环境中。本文提出了一种新颖的协调框架，通过三项关键技术解决这一挑战：1) 在协调过程中直接控制车辆的轨迹分布，将其建模为具有自适应增强安全约束的鲁棒协同规划问题，确保在交互轨迹不确定性方面的特定安全水平；2) 一种全并行的基于ADMM的分布式轨迹协商（ADMM-DTN）算法，能够在保证求解质量的同时灵活配置协商轮次，平衡解决方案质量与计算资源；3) 一种交互式注意力机制，能够有选择地关注关键的交互参与者，进一步提高计算效率。仿真结果和实际试验表明，与现有最佳方法相比，本框架在安全性（在多种场景下降低碰撞率高达40.79%）和实时性能上具有显著优势，且具有良好的可扩展性，随着车辆数量的增加保持高效。所提出的交互式注意力机制进一步降低了14.1%的计算需求。通过实际试验验证了该框架在具有意外动态障碍物的复杂环境中的鲁棒协调能力。完整的实验演示可以访问以下链接：this https URL。 

---
# Observability-driven Assignment of Heterogeneous Sensors for Multi-Target Tracking 

**Title (ZH)**: 可观测性驱动的异构传感器多目标跟踪分配 

**Authors**: Seyed Ali Rakhshan, Mehdi Golestani, He Kong  

**Link**: [PDF](https://arxiv.org/pdf/2508.21309)  

**Abstract**: This paper addresses the challenge of assigning heterogeneous sensors (i.e., robots with varying sensing capabilities) for multi-target tracking. We classify robots into two categories: (1) sufficient sensing robots, equipped with range and bearing sensors, capable of independently tracking targets, and (2) limited sensing robots, which are equipped with only range or bearing sensors and need to at least form a pair to collaboratively track a target. Our objective is to optimize tracking quality by minimizing uncertainty in target state estimation through efficient robot-to-target assignment. By leveraging matroid theory, we propose a greedy assignment algorithm that dynamically allocates robots to targets to maximize tracking quality. The algorithm guarantees constant-factor approximation bounds of 1/3 for arbitrary tracking quality functions and 1/2 for submodular functions, while maintaining polynomial-time complexity. Extensive simulations demonstrate the algorithm's effectiveness in accurately estimating and tracking targets over extended periods. Furthermore, numerical results confirm that the algorithm's performance is close to that of the optimal assignment, highlighting its robustness and practical applicability. 

**Abstract (ZH)**: 本文讨论了为多目标跟踪分配异构传感器（即具有不同感知能力的机器人）的挑战。我们将机器人分为两类：（1）足夠感知能力的机器人，配备有距离和方位传感器，能够独立跟踪目标；（2）受限感知能力的机器人，仅配备距离或方位传感器，至少需要成对协作才能跟踪目标。我们的目标是通过高效地分配机器人到目标来最小化目标状态估计的不确定性，从而优化跟踪质量。利用 matroid 理论，我们提出了一种贪婪分配算法，该算法动态地将机器人分配给目标以最大化跟踪质量。该算法对于任意跟踪质量函数提供了恒定因子逼近界的1/3，并且对于亚模函数提供了1/2的逼近界，同时保持多项式时间复杂性。 extensive 模拟表明，该算法在长期准确估计和跟踪目标方面非常有效。此外，数值结果证实该算法的性能接近最优分配，突显了其鲁棒性和实用性。 

---
# Learning to Assemble the Soma Cube with Legal-Action Masked DQN and Safe ZYZ Regrasp on a Doosan M0609 

**Title (ZH)**: 使用合法动作掩蔽DQN和Safe ZYZ 重新抓取学习构建Soma立方体 

**Authors**: Jaehong Oh, Seungjun Jung, Sawoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.21272)  

**Abstract**: This paper presents the first comprehensive application of legal-action masked Deep Q-Networks with safe ZYZ regrasp strategies to an underactuated gripper-equipped 6-DOF collaborative robot for autonomous Soma cube assembly learning. Our approach represents the first systematic integration of constraint-aware reinforcement learning with singularity-safe motion planning on a Doosan M0609 collaborative robot. We address critical challenges in robotic manipulation: combinatorial action space explosion, unsafe motion planning, and systematic assembly strategy learning. Our system integrates a legal-action masked DQN with hierarchical architecture that decomposes Q-function estimation into orientation and position components, reducing computational complexity from $O(3,132)$ to $O(116) + O(27)$ while maintaining solution completeness. The robot-friendly reward function encourages ground-first, vertically accessible assembly sequences aligned with manipulation constraints. Curriculum learning across three progressive difficulty levels (2-piece, 3-piece, 7-piece) achieves remarkable training efficiency: 100\% success rate for Level 1 within 500 episodes, 92.9\% for Level 2, and 39.9\% for Level 3 over 105,300 total training episodes. 

**Abstract (ZH)**: 这篇论文提出了将法律行动掩码深度Q网络与安全的ZYZ重抓策略综合应用于一个装备有6自由度未饱和夹持器的协作机器人，以进行自适应Soma立方体组装学习的方法。我们的方法代表了将约束感知强化学习与奇异点安全运动规划系统性集成到Doosan M0609协作机器人上的首次尝试。我们解决了机器人操作中的关键挑战：组合动作空间爆炸、不安全的运动规划和系统化组装策略学习。该系统结合了带分层架构的法律行动掩码DQN，将Q函数估计分解为姿态和位置组件，将计算复杂度从$O(3,132)$降低到$O(116) + O(27)$，同时保持解的完整性。针对现有的机器人友好型奖励函数鼓励先地再垂直组装序列，该序列符合操作约束。跨三个逐步提高难度级别（2块、3块、7块）的分级课程学习实现了显著的训练效率：在一级中500个episode内100%成功，二级中92.9%，三级中39.9%（总计105,300个训练episode）。 

---
# Mini Autonomous Car Driving based on 3D Convolutional Neural Networks 

**Title (ZH)**: 基于3D卷积神经网络的微型自主驾车技术 

**Authors**: Pablo Moraes, Monica Rodriguez, Kristofer S. Kappel, Hiago Sodre, Santiago Fernandez, Igor Nunes, Bruna Guterres, Ricardo Grando  

**Link**: [PDF](https://arxiv.org/pdf/2508.21271)  

**Abstract**: Autonomous driving applications have become increasingly relevant in the automotive industry due to their potential to enhance vehicle safety, efficiency, and user experience, thereby meeting the growing demand for sophisticated driving assistance features. However, the development of reliable and trustworthy autonomous systems poses challenges such as high complexity, prolonged training periods, and intrinsic levels of uncertainty. Mini Autonomous Cars (MACs) are used as a practical testbed, enabling validation of autonomous control methodologies on small-scale setups. This simplified and cost-effective environment facilitates rapid evaluation and comparison of machine learning models, which is particularly useful for algorithms requiring online training. To address these challenges, this work presents a methodology based on RGB-D information and three-dimensional convolutional neural networks (3D CNNs) for MAC autonomous driving in simulated environments. We evaluate the proposed approach against recurrent neural networks (RNNs), with architectures trained and tested on two simulated tracks with distinct environmental features. Performance was assessed using task completion success, lap-time metrics, and driving consistency. Results highlight how architectural modifications and track complexity influence the models' generalization capability and vehicle control performance. The proposed 3D CNN demonstrated promising results when compared with RNNs. 

**Abstract (ZH)**: 自主驾驶应用由于其在提升车辆安全、效率和用户体验方面的潜力，在汽车行业中越来越受到关注，从而满足了对复杂驾驶辅助功能的日益增长的需求。然而，可靠且可信赖的自主系统的开发面临着高复杂性、漫长的训练周期以及固有的不确定性等挑战。微型自主车（MACs）被用作实际的试验平台，用于在小型设置中验证自主控制方法。这种简化且经济高效的环境促进了机器学习模型的快速评估和比较，尤其适用于需要在线训练的算法。为应对这些挑战，本研究提出了一种基于RGB-D信息和三维卷积神经网络（3D CNN）的方法，用于仿真实验环境中的MAC自主驾驶。我们用两种具有不同环境特征的仿真实 Tracks 来训练和测试基于RNN的方法，并评估了提出的3D CNN方法的表现，通过任务完成成功率、圈速指标和驾驶一致性进行评估。研究结果表明，架构修改和赛道复杂性影响模型的泛化能力和车辆控制性能。与RNN相比，提出的3D CNN方法显示出有前景的结果。 

---
# Remarks on stochastic cloning and delayed-state filtering 

**Title (ZH)**: 关于随机克隆和延迟状态滤波的注记 

**Authors**: Tara Mina, Lindsey Marinello, John Christian  

**Link**: [PDF](https://arxiv.org/pdf/2508.21260)  

**Abstract**: Many estimation problems in robotics and navigation involve measurements that depend on prior states. A prominent example is odometry, which measures the relative change between states over time. Accurately handling these delayed-state measurements requires capturing their correlations with prior state estimates, and a widely used approach is stochastic cloning (SC), which augments the state vector to account for these correlations.
This work revisits a long-established but often overlooked alternative--the delayed-state Kalman filter--and demonstrates that a properly derived filter yields exactly the same state and covariance update as SC, without requiring state augmentation. Moreover, the generalized Kalman filter formulation provides computational advantages, while also reducing memory requirements for higher-dimensional states.
Our findings clarify a common misconception that Kalman filter variants are inherently unable to handle correlated delayed-state measurements, demonstrating that an alternative formulation achieves the same results more efficiently. 

**Abstract (ZH)**: 许多机器人学和导航中的估计问题涉及依赖于先验状态的测量。一个典型的例子是里程计，它测量的是时间上状态之间的相对变化。准确处理这些延迟状态的测量要求捕获它们与先验状态估计值之间的相关性，广泛应用的方法是随机克隆（SC），该方法通过扩展状态向量来考虑这些相关性。
本研究重新审视了一种长时间以来被忽视但仍然有效的替代方法——延迟状态卡尔曼滤波器——并证明了一个正确推导的滤波器可以达到与SC相同的状态和协方差更新，而无需扩展状态向量。此外，广义卡尔曼滤波器的构建形式提供了计算上的优势，并且对于高维状态还能降低内存需求。
我们的研究澄清了一个常见的误解，即卡尔曼滤波器变体本质上无法处理相关延迟状态测量，展示了另一种形式的滤波器可以更高效地达到相同的结果。 

---
# Uncertainty-Aware Ankle Exoskeleton Control 

**Title (ZH)**: 不确定性感知踝关节外骨骼控制 

**Authors**: Fatima Mumtaza Tourk, Bishoy Galoaa, Sanat Shajan, Aaron J. Young, Michael Everett, Max K. Shepherd  

**Link**: [PDF](https://arxiv.org/pdf/2508.21221)  

**Abstract**: Lower limb exoskeletons show promise to assist human movement, but their utility is limited by controllers designed for discrete, predefined actions in controlled environments, restricting their real-world applicability. We present an uncertainty-aware control framework that enables ankle exoskeletons to operate safely across diverse scenarios by automatically disengaging when encountering unfamiliar movements. Our approach uses an uncertainty estimator to classify movements as similar (in-distribution) or different (out-of-distribution) relative to actions in the training set. We evaluated three architectures (model ensembles, autoencoders, and generative adversarial networks) on an offline dataset and tested the strongest performing architecture (ensemble of gait phase estimators) online. The online test demonstrated the ability of our uncertainty estimator to turn assistance on and off as the user transitioned between in-distribution and out-of-distribution tasks (F1: 89.2). This new framework provides a path for exoskeletons to safely and autonomously support human movement in unstructured, everyday environments. 

**Abstract (ZH)**: 不确定aware控制框架使踝关节外骨骼能够在遇到未知动作时安全地适应多种场景 

---
# Multi-robot Path Planning and Scheduling via Model Predictive Optimal Transport (MPC-OT) 

**Title (ZH)**: 基于模型预测最优传输（MPC-OT）的多机器人路径规划与调度 

**Authors**: Usman A. Khan, Mouhacine Benosman, Wenliang Liu, Federico Pecora, Joseph W. Durham  

**Link**: [PDF](https://arxiv.org/pdf/2508.21205)  

**Abstract**: In this paper, we propose a novel methodology for path planning and scheduling for multi-robot navigation that is based on optimal transport theory and model predictive control. We consider a setup where $N$ robots are tasked to navigate to $M$ targets in a common space with obstacles. Mapping robots to targets first and then planning paths can result in overlapping paths that lead to deadlocks. We derive a strategy based on optimal transport that not only provides minimum cost paths from robots to targets but also guarantees non-overlapping trajectories. We achieve this by discretizing the space of interest into $K$ cells and by imposing a ${K\times K}$ cost structure that describes the cost of transitioning from one cell to another. Optimal transport then provides \textit{optimal and non-overlapping} cell transitions for the robots to reach the targets that can be readily deployed without any scheduling considerations. The proposed solution requires $\unicode{x1D4AA}(K^3\log K)$ computations in the worst-case and $\unicode{x1D4AA}(K^2\log K)$ for well-behaved problems. To further accommodate potentially overlapping trajectories (unavoidable in certain situations) as well as robot dynamics, we show that a temporal structure can be integrated into optimal transport with the help of \textit{replans} and \textit{model predictive control}. 

**Abstract (ZH)**: 基于最优传输理论和模型预测控制的多机器人导航路径规划与调度新方法 

---
# Observer Design for Optical Flow-Based Visual-Inertial Odometry with Almost-Global Convergence 

**Title (ZH)**: 基于光学流动的视觉-惯性里程计的观察者设计与几乎全局收敛分析 

**Authors**: Tarek Bouazza, Soulaimane Berkane, Minh-Duc Hua, Tarek Hamel  

**Link**: [PDF](https://arxiv.org/pdf/2508.21163)  

**Abstract**: This paper presents a novel cascaded observer architecture that combines optical flow and IMU measurements to perform continuous monocular visual-inertial odometry (VIO). The proposed solution estimates body-frame velocity and gravity direction simultaneously by fusing velocity direction information from optical flow measurements with gyro and accelerometer data. This fusion is achieved using a globally exponentially stable Riccati observer, which operates under persistently exciting translational motion conditions. The estimated gravity direction in the body frame is then employed, along with an optional magnetometer measurement, to design a complementary observer on $\mathbf{SO}(3)$ for attitude estimation. The resulting interconnected observer architecture is shown to be almost globally asymptotically stable. To extract the velocity direction from sparse optical flow data, a gradient descent algorithm is developed to solve a constrained minimization problem on the unit sphere. The effectiveness of the proposed algorithms is validated through simulation results. 

**Abstract (ZH)**: 这篇论文提出了一种新颖的级联观测器架构，结合了光学流和IMU测量来执行单目视觉惯性里程计（VIO）。所提出的方法通过融合光学流测量的速度方向信息与陀螺仪和加速度计数据，同时估计体-frame速度和重力方向。这种融合是通过在持续激发的平移运动条件下使用全局指数稳定的Riccati观测器实现的。在体-frame中估计的重力方向随后被用于与可选的磁力计测量结合，以在$\mathbf{SO}(3)$上设计一个补充观测器进行姿态估计。所得的互联观测器架构被证明是几乎全局渐近稳定的。为了从稀疏的光学流数据中提取速度方向，开发了一种梯度下降算法来求解单位球上的约束最小化问题。所提出算法的有效性通过仿真结果得到了验证。 

---
# EmbodiedOneVision: Interleaved Vision-Text-Action Pretraining for General Robot Control 

**Title (ZH)**: EmbodiedOneVision:交替进行的视觉-文本-动作预训练以实现通用机器人控制 

**Authors**: Delin Qu, Haoming Song, Qizhi Chen, Zhaoqing Chen, Xianqiang Gao, Xinyi Ye, Qi Lv, Modi Shi, Guanghui Ren, Cheng Ruan, Maoqing Yao, Haoran Yang, Jiacheng Bao, Bin Zhao, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21112)  

**Abstract**: The human ability to seamlessly perform multimodal reasoning and physical interaction in the open world is a core goal for general-purpose embodied intelligent systems. Recent vision-language-action (VLA) models, which are co-trained on large-scale robot and visual-text data, have demonstrated notable progress in general robot control. However, they still fail to achieve human-level flexibility in interleaved reasoning and interaction. In this work, introduce EO-Robotics, consists of EO-1 model and EO-Data1.5M dataset. EO-1 is a unified embodied foundation model that achieves superior performance in multimodal embodied reasoning and robot control through interleaved vision-text-action pre-training. The development of EO-1 is based on two key pillars: (i) a unified architecture that processes multimodal inputs indiscriminately (image, text, video, and action), and (ii) a massive, high-quality multimodal embodied reasoning dataset, EO-Data1.5M, which contains over 1.5 million samples with emphasis on interleaved vision-text-action comprehension. EO-1 is trained through synergies between auto-regressive decoding and flow matching denoising on EO-Data1.5M, enabling seamless robot action generation and multimodal embodied reasoning. Extensive experiments demonstrate the effectiveness of interleaved vision-text-action learning for open-world understanding and generalization, validated through a variety of long-horizon, dexterous manipulation tasks across multiple embodiments. This paper details the architecture of EO-1, the data construction strategy of EO-Data1.5M, and the training methodology, offering valuable insights for developing advanced embodied foundation models. 

**Abstract (ZH)**: 在开放世界中无缝进行多模态推理和物理交互的人类能力是通用体怔智能系统的核心目标。近期的视知觉-语言-行动（VLA）模型，在大规模机器人和视觉文本数据上联合训练，已在通用机器人控制方面取得了显著进展。然而，它们仍未实现与人类水平相媲美的交错推理和交互灵活性。在本工作中，我们引入了EO-Robotics，包括EO-1模型和EO-Data1.5M数据集。EO-1是一个统一的体怔基础模型，通过交错的视知觉-行动预训练，在多模态体怔推理和机器人控制方面表现出色。EO-1的发展基于两个关键支柱：（i）一个不区分地处理多模态输入（图像、文本、视频和行动）的统一架构，以及（ii）一个大规模的高质量多模态体怔推理数据集EO-Data1.5M，包含了超过150万样本，重点关注交错的视知觉-文本-行动理解。EO-1通过与EO-Data1.5M的协同自回归解码和流动匹配去噪训练，实现了无缝的机器人动作生成和多模态体怔推理。广泛的实验验证了交错视知觉-文本-行动学习在开放世界理解与泛化中的有效性，通过多种长期任务和灵巧操作任务进行了验证。本文详细介绍了EO-1的架构、EO-Data1.5M的数据构建策略和训练方法，为开发高级体怔基础模型提供了宝贵的见解。 

---
# Tree-Guided Diffusion Planner 

**Title (ZH)**: 树引导扩散规划者 

**Authors**: Hyeonseong Jeon, Cheolhong Min, Jaesik Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.21800)  

**Abstract**: Planning with pretrained diffusion models has emerged as a promising approach for solving test-time guided control problems. However, standard gradient guidance typically performs optimally under convex and differentiable reward landscapes, showing substantially reduced effectiveness in real-world scenarios involving non-convex objectives, non-differentiable constraints, and multi-reward structures. Furthermore, recent supervised planning approaches require task-specific training or value estimators, which limits test-time flexibility and zero-shot generalization. We propose a Tree-guided Diffusion Planner (TDP), a zero-shot test-time planning framework that balances exploration and exploitation through structured trajectory generation. We frame test-time planning as a tree search problem using a bi-level sampling process: (1) diverse parent trajectories are produced via training-free particle guidance to encourage broad exploration, and (2) sub-trajectories are refined through fast conditional denoising guided by task objectives. TDP addresses the limitations of gradient guidance by exploring diverse trajectory regions and harnessing gradient information across this expanded solution space using only pretrained models and test-time reward signals. We evaluate TDP on three diverse tasks: maze gold-picking, robot arm block manipulation, and AntMaze multi-goal exploration. TDP consistently outperforms state-of-the-art approaches on all tasks. The project page can be found at: this http URL. 

**Abstract (ZH)**: 基于树引导的扩散计划器：一种零样本测试时规划框架 

---
# A-MHA*: Anytime Multi-Heuristic A* 

**Title (ZH)**: A-MHA*: 任意可选多启发式A* 

**Authors**: Ramkumar Natarajan, Muhammad Suhail Saleem, William Xiao, Sandip Aine, Howie Choset, Maxim Likhachev  

**Link**: [PDF](https://arxiv.org/pdf/2508.21637)  

**Abstract**: Designing good heuristic functions for graph search requires adequate domain knowledge. It is often easy to design heuristics that perform well and correlate with the underlying true cost-to-go values in certain parts of the search space but these may not be admissible throughout the domain thereby affecting the optimality guarantees of the search. Bounded suboptimal search using several such partially good but inadmissible heuristics was developed in Multi-Heuristic A* (MHA*). Although MHA* leverages multiple inadmissible heuristics to potentially generate a faster suboptimal solution, the original version does not improve the solution over time. It is a one shot algorithm that requires careful setting of inflation factors to obtain a desired one time solution. In this work, we tackle this issue by extending MHA* to an anytime version that finds a feasible suboptimal solution quickly and continually improves it until time runs out. Our work is inspired from the Anytime Repairing A* (ARA*) algorithm. We prove that our precise adaptation of ARA* concepts in the MHA* framework preserves the original suboptimal and completeness guarantees and enhances MHA* to perform in an anytime fashion. Furthermore, we report the performance of A-MHA* in 3-D path planning domain and sliding tiles puzzle and compare against MHA* and other anytime algorithms. 

**Abstract (ZH)**: 设计良好的图搜索启发式函数需要足够的领域知识。虽然容易设计出在搜索空间某些部分表现良好且与真实代价到终点值相关联的启发式函数，但这些启发式函数可能在整个领域内不具备可接受性，从而影响搜索的最优性保证。Multi-Heuristic A* (MHA*) 开发了一种利用多个部分良好但不可接受的启发式函数进行有界限次优搜索的方法。尽管 MHA* 利用了多个启发式函数以潜在地生成更快的次优解，但原始版本不能随着时间改善解的质量。它是一个一次性算法，需要仔细设置放大因子以获得所需的单次解。在本工作中，我们通过将 MHA* 扩展为一个可以快速找到可行的次优解并在时间耗尽前不断改进的任意时间版本解决这一问题。我们的工作灵感来自于 Anytime Repairing A* (ARA*) 算法。我们证明了将 ARA* 概念精确适应 MHA* 架构保持了原始的次优性和完备性保证，并使 MHA* 能够以任意时间的方式运行。此外，我们在三维路径规划领域和滑动拼图中报告了 A-MHA* 的性能，并与 MHA* 和其他任意时间算法进行了对比。 

---
# Complete Gaussian Splats from a Single Image with Denoising Diffusion Models 

**Title (ZH)**: 使用去噪扩散模型从单张图像生成完整高斯体素 

**Authors**: Ziwei Liao, Mohamed Sayed, Steven L. Waslander, Sara Vicente, Daniyar Turmukhambetov, Michael Firman  

**Link**: [PDF](https://arxiv.org/pdf/2508.21542)  

**Abstract**: Gaussian splatting typically requires dense observations of the scene and can fail to reconstruct occluded and unobserved areas. We propose a latent diffusion model to reconstruct a complete 3D scene with Gaussian splats, including the occluded parts, from only a single image during inference. Completing the unobserved surfaces of a scene is challenging due to the ambiguity of the plausible surfaces. Conventional methods use a regression-based formulation to predict a single "mode" for occluded and out-of-frustum surfaces, leading to blurriness, implausibility, and failure to capture multiple possible explanations. Thus, they often address this problem partially, focusing either on objects isolated from the background, reconstructing only visible surfaces, or failing to extrapolate far from the input views. In contrast, we propose a generative formulation to learn a distribution of 3D representations of Gaussian splats conditioned on a single input image. To address the lack of ground-truth training data, we propose a Variational AutoReconstructor to learn a latent space only from 2D images in a self-supervised manner, over which a diffusion model is trained. Our method generates faithful reconstructions and diverse samples with the ability to complete the occluded surfaces for high-quality 360-degree renderings. 

**Abstract (ZH)**: 基于拉普拉斯扩散模型的单图像完整3D场景重建 

---
# Cooperative Sensing Enhanced UAV Path-Following and Obstacle Avoidance with Variable Formation 

**Title (ZH)**: 基于可变编队合作感知的无人机路径跟踪与避障技术 

**Authors**: Changheng Wang, Zhiqing Wei, Wangjun Jiang, Haoyue Jiang, Zhiyong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.21316)  

**Abstract**: The high mobility of unmanned aerial vehicles (UAVs) enables them to be used in various civilian fields, such as rescue and cargo transport. Path-following is a crucial way to perform these tasks while sensing and collision avoidance are essential for safe flight. In this paper, we investigate how to efficiently and accurately achieve path-following, obstacle sensing and avoidance subtasks, as well as their conflict-free fusion scheduling. Firstly, a high precision deep reinforcement learning (DRL)-based UAV formation path-following model is developed, and the reward function with adaptive weights is designed from the perspective of distance and velocity errors. Then, we use integrated sensing and communication (ISAC) signals to detect the obstacle and derive the Cramer-Rao lower bound (CRLB) for obstacle sensing by information-level fusion, based on which we propose the variable formation enhanced obstacle position estimation (VFEO) algorithm. In addition, an online obstacle avoidance scheme without pretraining is designed to solve the sparse reward. Finally, with the aid of null space based (NSB) behavioral method, we present a hierarchical subtasks fusion strategy. Simulation results demonstrate the effectiveness and superiority of the subtask algorithms and the hierarchical fusion strategy. 

**Abstract (ZH)**: 无人驾驶航空车辆（UAVs）的高机动性使其能够在救援和货物运输等众多民用领域应用，路径跟踪是执行这些任务的关键方式，而感测和避障是确保安全飞行的必要条件。本文探讨了如何高效准确地实现路径跟踪、障碍感测与避障子任务及其冲突-free融合调度。首先，提出了一种基于深度强化学习（DRL）的高精度UAV编队路径跟踪模型，并从距离和速度误差的角度设计了自适应权重的奖励函数。然后，利用综合感知与通信（ISAC）信号进行障碍检测，并基于信息级融合推导出障碍感测的克拉默- Rao下界（CRLB），在此基础上提出了变形成编队增强障碍位置估计（VFEO）算法。此外，设计了一种在线避障方案以解决稀疏奖励问题。最后，借助基于 null 空间方法（NSB）的行为方法，提出了分层次子任务融合策略。仿真实验结果表明了子任务算法和分层次融合策略的有效性和优越性。 

---
# Detecting Domain Shifts in Myoelectric Activations: Challenges and Opportunities in Stream Learning 

**Title (ZH)**: 检测肌电激活中的领域转移：流学习中的挑战与机遇 

**Authors**: Yibin Sun, Nick Lim, Guilherme Weigert Cassales, Heitor Murilo Gomes, Bernhard Pfahringer, Albert Bifet, Anany Dwivedi  

**Link**: [PDF](https://arxiv.org/pdf/2508.21278)  

**Abstract**: Detecting domain shifts in myoelectric activations poses a significant challenge due to the inherent non-stationarity of electromyography (EMG) signals. This paper explores the detection of domain shifts using data stream (DS) learning techniques, focusing on the DB6 dataset from the Ninapro database. We define domains as distinct time-series segments based on different subjects and recording sessions, applying Kernel Principal Component Analysis (KPCA) with a cosine kernel to pre-process and highlight these shifts. By evaluating multiple drift detection methods such as CUSUM, Page-Hinckley, and ADWIN, we reveal the limitations of current techniques in achieving high performance for real-time domain shift detection in EMG signals. Our results underscore the potential of streaming-based approaches for maintaining stable EMG decoding models, while highlighting areas for further research to enhance robustness and accuracy in real-world scenarios. 

**Abstract (ZH)**: 基于数据流学习方法在肌电激活域迁移检测中的挑战与探索：以Ninapro数据库的DB6数据集为例 

---
# GENNAV: Polygon Mask Generation for Generalized Referring Navigable Regions 

**Title (ZH)**: GENNAV: 通用引用可导航区域的多边形掩码生成 

**Authors**: Kei Katsumata, Yui Iioka, Naoki Hosomi, Teruhisa Misu, Kentaro Yamada, Komei Sugiura  

**Link**: [PDF](https://arxiv.org/pdf/2508.21102)  

**Abstract**: We focus on the task of identifying the location of target regions from a natural language instruction and a front camera image captured by a mobility. This task is challenging because it requires both existence prediction and segmentation, particularly for stuff-type target regions with ambiguous boundaries. Existing methods often underperform in handling stuff-type target regions, in addition to absent or multiple targets. To overcome these limitations, we propose GENNAV, which predicts target existence and generates segmentation masks for multiple stuff-type target regions. To evaluate GENNAV, we constructed a novel benchmark called GRiN-Drive, which includes three distinct types of samples: no-target, single-target, and multi-target. GENNAV achieved superior performance over baseline methods on standard evaluation metrics. Furthermore, we conducted real-world experiments with four automobiles operated in five geographically distinct urban areas to validate its zero-shot transfer performance. In these experiments, GENNAV outperformed baseline methods and demonstrated its robustness across diverse real-world environments. The project page is available at this https URL. 

**Abstract (ZH)**: 我们专注于从自然语言指令和由移动设备捕捉的前视摄像头图像中识别目标区域的位置。这项任务具有挑战性，因为它需要同时进行存在预测和分割，特别是对于边界模糊的目标区域。现有方法在处理此类目标区域时常常表现不佳，尤其是在处理不存在或多个目标的情况下。为了克服这些限制，我们提出了GENNAV，它预测目标的存在并为多个类型的目标区域生成分割掩码。为了评估GENNAV，我们构建了一个名为GRiN-Drive的新基准，其中包括三种不同的样本类型：无目标、单目标和多目标。GENNAV在标准评估指标上优于基线方法。此外，我们在五个城市地区进行的四辆汽车的实际实验中验证了其零样本迁移性能，在这些实验中，GENNAV优于基线方法并展示了其在不同现实环境中的鲁棒性。项目页面可访问此链接。 

---
# 2COOOL: 2nd Workshop on the Challenge Of Out-Of-Label Hazards in Autonomous Driving 

**Title (ZH)**: 2COOOL: 第二届关于自主驾驶领域未标注风险挑战研讨会 

**Authors**: Ali K. AlShami, Ryan Rabinowitz, Maged Shoman, Jianwu Fang, Lukas Picek, Shao-Yuan Lo, Steve Cruz, Khang Nhut Lam, Nachiket Kamod, Lei-Lei Li, Jugal Kalita, Terrance E. Boult  

**Link**: [PDF](https://arxiv.org/pdf/2508.21080)  

**Abstract**: As the computer vision community advances autonomous driving algorithms, integrating vision-based insights with sensor data remains essential for improving perception, decision making, planning, prediction, simulation, and control. Yet we must ask: Why don't we have entirely safe self-driving cars yet? A key part of the answer lies in addressing novel scenarios, one of the most critical barriers to real-world deployment. Our 2COOOL workshop provides a dedicated forum for researchers and industry experts to push the state of the art in novelty handling, including out-of-distribution hazard detection, vision-language models for hazard understanding, new benchmarking and methodologies, and safe autonomous driving practices. The 2nd Workshop on the Challenge of Out-of-Label Hazards in Autonomous Driving (2COOOL) will be held at the International Conference on Computer Vision (ICCV) 2025 in Honolulu, Hawaii, on October 19, 2025. We aim to inspire the development of new algorithms and systems for hazard avoidance, drawing on ideas from anomaly detection, open-set recognition, open-vocabulary modeling, domain adaptation, and related fields. Building on the success of its inaugural edition at the Winter Conference on Applications of Computer Vision (WACV) 2025, the workshop will feature a mix of academic and industry participation. 

**Abstract (ZH)**: 面向自主驾驶领域的标签外危害应对挑战研讨会（2COOOL） 

---
