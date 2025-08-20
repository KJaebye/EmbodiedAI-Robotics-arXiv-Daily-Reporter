# The Social Context of Human-Robot Interactions 

**Title (ZH)**: 人类与机器人互动的社会背景 

**Authors**: Sydney Thompson, Kate Candon, Marynel Vázquez  

**Link**: [PDF](https://arxiv.org/pdf/2508.13982)  

**Abstract**: The Human-Robot Interaction (HRI) community often highlights the social context of an interaction as a key consideration when designing, implementing, and evaluating robot behavior. Unfortunately, researchers use the term "social context" in varied ways. This can lead to miscommunication, making it challenging to draw connections between related work on understanding and modeling the social contexts of human-robot interactions. To address this gap, we survey the HRI literature for existing definitions and uses of the term "social context". Then, we propose a conceptual model for describing the social context of a human-robot interaction. We apply this model to existing work, and we discuss a range of attributes of social contexts that can help researchers plan for interactions, develop behavior models for robots, and gain insights after interactions have taken place. We conclude with a discussion of open research questions in relation to understanding and modeling the social contexts of human-robot interactions. 

**Abstract (ZH)**: 人机交互（HRI）社区常强调在设计、实现和评估机器人行为时，社交背景是关键考虑因素。然而，研究人员对“社交背景”一词的使用方式各异，这可能导致沟通不畅，使得难以在理解与建模人机交互的社交背景方面建立相关研究之间的联系。为解决这一问题，我们回顾了HRI文献中对“社交背景”这一术语的现有定义和使用方式，并提出了一种描述人机交互社交背景的概念模型。我们将该模型应用于现有研究，并讨论了一系列有助于研究人员规划交互、为机器人开发行为模型以及在交互发生后获得见解的社交背景属性。最后，我们讨论了在理解与建模人机交互的社交背景方面的开放性研究问题。 

---
# Incremental Generalized Hybrid A* 

**Title (ZH)**: 增量广义混合A*算法 

**Authors**: Sidharth Talia, Oren Salzman, Siddhartha Srinivasa  

**Link**: [PDF](https://arxiv.org/pdf/2508.13392)  

**Abstract**: We address the problem of efficiently organizing search over very large trees, which arises in many applications ranging from autonomous driving to aerial vehicles. Here, we are motivated by off-road autonomy, where real-time planning is essential. Classical approaches use graphs of motion primitives and exploit dominance to mitigate the curse of dimensionality and prune expansions efficiently. However, for complex dynamics, repeatedly solving two-point boundary-value problems makes graph construction too slow for fast kinodynamic planning. Hybrid A* (HA*) addressed this challenge by searching over a tree of motion primitives and introducing approximate pruning using a grid-based dominance check. However, choosing the grid resolution is difficult: too coarse risks failure, while too fine leads to excessive expansions and slow planning. We propose Incremental Generalized Hybrid A* (IGHA*), an anytime tree-search framework that dynamically organizes vertex expansions without rigid pruning. IGHA* provably matches or outperforms HA*. For both on-road kinematic and off-road kinodynamic planning queries for a car-like robot, variants of IGHA* use 6x fewer expansions to the best solution compared to an optimized version of HA*. In simulated off-road experiments in a high fidelity simulator, IGHA* outperforms HA*M when both are used in the loop with a model predictive controller. We demonstrate real-time performance both in simulation and on a small-scale off-road vehicle, enabling fast, robust planning under complex dynamics. Code: this https URL 

**Abstract (ZH)**: 高效组织大规模树结构搜索的问题及其应用：增量通用混合A*在离路面自主导航中的表现 

---
# ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor Plans 

**Title (ZH)**: ResPlan: 一个包含17,000个住宅平面图的大型向量-图数据集 

**Authors**: Mohamed Abouagour, Eleftherios Garyfallidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.14006)  

**Abstract**: We introduce ResPlan, a large-scale dataset of 17,000 detailed, structurally rich, and realistic residential floor plans, created to advance spatial AI research. Each plan includes precise annotations of architectural elements (walls, doors, windows, balconies) and functional spaces (such as kitchens, bedrooms, and bathrooms). ResPlan addresses key limitations of existing datasets such as RPLAN (Wu et al., 2019) and MSD (van Engelenburg et al., 2024) by offering enhanced visual fidelity and greater structural diversity, reflecting realistic and non-idealized residential layouts. Designed as a versatile, general-purpose resource, ResPlan supports a wide range of applications including robotics, reinforcement learning, generative AI, virtual and augmented reality, simulations, and game development. Plans are provided in both geometric and graph-based formats, enabling direct integration into simulation engines and fast 3D conversion. A key contribution is an open-source pipeline for geometry cleaning, alignment, and annotation refinement. Additionally, ResPlan includes structured representations of room connectivity, supporting graph-based spatial reasoning tasks. Finally, we present comparative analyses with existing benchmarks and outline several open benchmark tasks enabled by ResPlan. Ultimately, ResPlan offers a significant advance in scale, realism, and usability, providing a robust foundation for developing and benchmarking next-generation spatial intelligence systems. 

**Abstract (ZH)**: ResPlan：一个包含17,000个详细、结构丰富且逼真的住宅平面图的大规模数据集 

---
# A Screw Approach to the Approximation of the Local Geometry of the Configuration Space and of the set of Configurations of Certain Rank of Lower Pair Linkages 

**Title (ZH)**: 螺杆方法在连续刚体体系局部几何结构及其特定秩的配置集逼近中的应用 

**Authors**: Andreas Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2508.13802)  

**Abstract**: A motion of a mechanism is a curve in its configuration space (c-space). Singularities of the c-space are kinematic singularities of the mechanism. Any mobility analysis of a particular mechanism amounts to investigating the c-space geometry at a given configuration. A higher-order analysis is necessary to determine the finite mobility. To this end, past research lead to approaches using higher-order time derivatives of loop closure constraints assuming (implicitly) that all possible motions are smooth. This continuity assumption limits the generality of these methods. In this paper an approach to the higher-order local mobility analysis of lower pair multi-loop linkages is presented. This is based on a higher-order Taylor series expansion of the geometric constraint mapping, for which a recursive algebraic expression in terms of joint screws is presented. An exhaustive local analysis includes analysis of the set of constraint singularities (configurations where the constraint Jacobian has certain corank). A local approximation of the set of configurations with certain rank is presented, along with an explicit expression for the differentials of Jacobian minors in terms of instantaneous joint screws. The c-space and the set of points of certain corank are therewith locally approximated by an algebraic variety determined algebraically from the mechanism's screw system. Results are shown for a simple planar 4-bar linkage, which exhibits a bifurcation singularity, and for a planar three-loop linkage exhibiting a cusp in c-space. The latter cannot be treated by the higher-order local analysis methods proposed in the literature. 

**Abstract (ZH)**: 一种机制的运动是其配置空间（c-空间）中的曲线。c-空间的奇异点是机制的运动奇异点。对某一特定机制的自由度分析等同于研究给定配置下的c-空间几何。要确定有限自由度，需要进行更高阶分析。过去的研究通过假设（隐含地）所有可能运动都平滑来进行更高阶时间导数的环闭约束分析。这一连续性假设限制了这些方法的普遍性。本文提出了低副多环杆机构更高阶局部自由度分析的方法，基于几何约束映射的更高阶泰勒级数展开，给出了基于关节楔形的递归代数表达式。完整的局部分析包括约束奇异点集（约束雅可比矩阵具有特定秩亏度的配置）的分析。给出了具有特定秩的配置集的局部近似表示，以及雅可比子式微分的显式表达式，与瞬时关节楔形相关。c-空间和特定秩亏度点的集合通过机制楔形系统的代数方法确定的代数簇进行局部近似。结果展示了具有分支奇异点的简单平面4杆机构以及具有c-空间尖点的平面三环杆机构。后者无法被文献中提出的局部高阶分析方法处理。 

---
# AutoMPC: A Code Generator for MPC-based Automated Driving 

**Title (ZH)**: AutoMPC: 一种基于MPC的自动行驶代码生成器 

**Authors**: Georg Schildbach, Jasper Pflughaupt  

**Link**: [PDF](https://arxiv.org/pdf/2508.13656)  

**Abstract**: Model Predictive Control (MPC) is a powerful technique to control nonlinear, multi-input multi-output systems subject to input and state constraints. It is now a standard tool for trajectory tracking control of automated vehicles. As such it has been used in many research and development projects. However, MPC faces several challenges to be integrated into industrial production vehicles. The most important ones are its high computational demands and the complexity of implementation. The software packages AutoMPC aims to address both of these challenges. It builds on a robustified version of an active set algorithm for Nonlinear MPC. The algorithm is embedded into a framework for vehicle trajectory tracking, which makes it easy to used, yet highly customizable. Automatic code generation transforms the selections into a standalone, computationally efficient C-code file with static memory allocation. As such it can be readily deployed on a wide range of embedded platforms, e.g., based on Matlab/Simulink or Robot Operating System (ROS). Compared to a previous version of the code, the vehicle model and the numerical integration method can be manually specified, besides basic algorithm parameters. All of this information and all specifications are directly baked into the generated C-code. The algorithm is suitable driving scenarios at low or high speeds, even drifting, and supports direction changes. Multiple simulation scenarios show the versatility and effectiveness of the AutoMPC code, with the guarantee of a feasible solution, a high degree of robustness, and computational efficiency. 

**Abstract (ZH)**: 模型预测控制（MPC）是用于控制受输入和状态约束的非线性多输入多输出系统的强大技术，现已成为自动驾驶车辆轨迹跟踪控制的标准工具。尽管如此，MPC 集成到工业生产车辆中仍面临诸多挑战，其中最重要的包括其高计算需求和复杂的实现方式。软件包AutoMPC旨在解决这两个问题。它基于鲁棒化的非线性MPC活性集算法版本，并将其嵌入到车辆轨迹跟踪框架中，使其易于使用且高度可定制。自动代码生成将选择转换为独立的、计算效率高的C代码文件，具有静态内存分配。因此，它可以便捷地部署到各种嵌入式平台，例如基于Matlab/Simulink或Robot Operating System (ROS)。与之前版本的代码相比，不仅可以手动指定车辆模型和数值积分方法，还可以指定基本算法参数。所有这些信息和所有规范都直接嵌入到生成的C代码中。该算法适用于低速或高速驾驶场景，甚至漂移，并支持方向变化。多个仿真场景展示了AutoMPC代码的多样性和有效性，并保证了可行解、高鲁棒性和计算效率。 

---
# Observed Control -- Linearly Scalable Nonlinear Model Predictive Control with Adaptive Horizons 

**Title (ZH)**: 观测控制——具有自适应 horizons 的线性可扩展非线性模型预测控制 

**Authors**: Eugene T. Hamzezadeh, Andrew J. Petruska  

**Link**: [PDF](https://arxiv.org/pdf/2508.13339)  

**Abstract**: This work highlights the duality between state estimation methods and model predictive control. A predictive controller, observed control, is presented that uses this duality to efficiently compute control actions with linear time-horizon length scalability. The proposed algorithms provide exceptional computational efficiency, adaptive time horizon lengths, and early optimization termination criteria. The use of Kalman smoothers as the backend optimization framework provides for a straightforward implementation supported by strong theoretical guarantees. Additionally, a formulation is presented that separates linear model predictive control into purely reactive and anticipatory components, enabling any-time any-horizon observed control while ensuring controller stability for short time horizons. Finally, numerical case studies confirm that nonlinear filter extensions, i.e., the extended Kalman filter and unscented Kalman filter, effectively extend observed control to nonlinear systems and objectives. 

**Abstract (ZH)**: 这项工作强调了状态估计方法与模型预测控制之间的二重性。提出了一种预测控制器，称为观察控制，该控制器利用这种二重性高效地计算控制动作，并具有线性时间范围长度可扩展性。所提出的算法提供了出色的计算效率、自适应的时间范围长度以及早期优化终止标准。用卡尔曼平滑器作为后端优化框架，提供了一种直接实现方式，并具有强大的理论保证。此外，提出了一种将线性模型预测控制分解为纯粹的反应性和预见性组件的公式，从而实现任何时间和任何时间范围的观察控制，同时确保控制器在短时间范围内的稳定性。最后，数值案例研究证实，非线性滤波器扩展，即扩展卡尔曼滤波器和无迹卡尔曼滤波器，有效地将观察控制扩展到非线性系统和目标。 

---
# A Biased Random Key Genetic Algorithm for Solving the Longest Run Subsequence Problem 

**Title (ZH)**: 带有偏置随机密钥的遗传算法求解最长连续子序列问题 

**Authors**: Christian Blum, Pedro Pinacho-Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2508.14020)  

**Abstract**: The longest run subsequence (LRS) problem is an NP-hard combinatorial optimization problem belonging to the class of subsequence problems from bioinformatics. In particular, the problem plays a role in genome reassembly. In this paper, we present a solution to the LRS problem using a Biased Random Key Genetic Algorithm (BRKGA). Our approach places particular focus on the computational efficiency of evaluating individuals, which involves converting vectors of gray values into valid solutions to the problem. For comparison purposes, a Max-Min Ant System is developed and implemented. This is in addition to the application of the integer linear programming solver CPLEX for solving all considered problem instances. The computation results show that the proposed BRKGA is currently a state-of-the-art technique for the LRS problem. Nevertheless, the results also show that there is room for improvement, especially in the context of input strings based on large alphabet sizes. 

**Abstract (ZH)**: 最长运行子序列问题（LRS）是属于生物信息学子序列问题类别的NP难组合优化问题，特别在基因组重构中起着重要作用。本文提出了一种使用有偏随机键遗传算法（BRKGA）解决LRS问题的方法，重点在于计算效率的评估个体过程，即将灰度值向量转换为问题的有效解。为了进行对比，我们开发并实现了最大最小蚁群系统（Max-Min Ant System），同时还应用了整数线性规划求解器CPLEX求解所有考虑的问题实例。计算结果表明，提出的BRKGA目前是解决LRS问题的先进方法。然而，结果也表明，在基于大字母表的输入字符串的背景下，仍有一定的改进空间。 

---
# Quantifier Instantiations: To Mimic or To Revolt? 

**Title (ZH)**: 量词实例化：模仿还是反叛？ 

**Authors**: Jan Jakubův, Mikoláš Janota  

**Link**: [PDF](https://arxiv.org/pdf/2508.13811)  

**Abstract**: Quantified formulas pose a significant challenge for Satisfiability Modulo Theories (SMT) solvers due to their inherent undecidability. Existing instantiation techniques, such as e-matching, syntax-guided, model-based, conflict-based, and enumerative methods, often complement each other. This paper introduces a novel instantiation approach that dynamically learns from these techniques during solving. By treating observed instantiations as samples from a latent language, we use probabilistic context-free grammars to generate new, similar terms. Our method not only mimics successful past instantiations but also explores diversity by optionally inverting learned term probabilities, aiming to balance exploitation and exploration in quantifier reasoning. 

**Abstract (ZH)**: 量化公式的存在使得理论饱和可满足性（SMT）求解器面临显著挑战，这归因于其固有的不可判定性。现有的实例化技术，如e-matching、语法引导、基于模型、冲突驱动和枚举方法，常常相互补充。本文提出了一种新颖的实例化方法，在求解过程中动态学习这些技术。通过将观察到的实例化视作潜在语言的样本，我们使用概率上下文无关文法生成新的、类似的项。我们的方法不仅模仿成功的过去实例化，还通过可选地反转学习到的项概率来探索多样性，旨在量化推理中利用和探索间的平衡。 

---
# The DeepLog Neurosymbolic Machine 

**Title (ZH)**: 深度日志神经符号机器 

**Authors**: Vincent Derkinderen, Robin Manhaeve, Rik Adriaensen, Lucas Van Praet, Lennert De Smet, Giuseppe Marra, Luc De Raedt  

**Link**: [PDF](https://arxiv.org/pdf/2508.13697)  

**Abstract**: We contribute a theoretical and operational framework for neurosymbolic AI called DeepLog. DeepLog introduces building blocks and primitives for neurosymbolic AI that make abstraction of commonly used representations and computational mechanisms used in neurosymbolic AI. DeepLog can represent and emulate a wide range of neurosymbolic systems. It consists of two key components. The first is the DeepLog language for specifying neurosymbolic models and inference tasks. This language consists of an annotated neural extension of grounded first-order logic, and makes abstraction of the type of logic, e.g. boolean, fuzzy or probabilistic, and whether logic is used in the architecture or in the loss function. The second DeepLog component is situated at the computational level and uses extended algebraic circuits as computational graphs. Together these two components are to be considered as a neurosymbolic abstract machine, with the DeepLog language as the intermediate level of abstraction and the circuits level as the computational one. DeepLog is implemented in software, relies on the latest insights in implementing algebraic circuits on GPUs, and is declarative in that it is easy to obtain different neurosymbolic models by making different choices for the underlying algebraic structures and logics. The generality and efficiency of the DeepLog neurosymbolic machine is demonstrated through an experimental comparison between 1) different fuzzy and probabilistic logics, 2) between using logic in the architecture or in the loss function, and 3) between a standalone CPU-based implementation of a neurosymbolic AI system and a DeepLog GPU-based one. 

**Abstract (ZH)**: 我们提出了一种名为DeepLog的神经符号.Symbolic人工智能的理论与操作框架。DeepLog引入了构建.神经符号.Symbolic人工智能的基础构建模块和.原语，用于表示.和推演常见的表示.表示.抽象 e.在神经符号.Symbolic人工智能中.中的 e e表示使用的表示...e.机制。DeepLog.可以能够表示. e e和和 e e各种 e e e广的 e e e e e e e e e e e e神经 e.符号. e符号 e e e和 e e e e e e系统 e e e.系统。 e e e Deep E e 由 由由 e e e 由  e e e e  e e由  e e e  e e e e  e e e  e e  e e e e  e e e  e e e  e e  e e e e  e e e e  e  e e e e e e e e e e e e e e e e e e e e e e e e e  e  e e e  e e e  e e e e e e e e e e e  e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e工作作风。 �_Equals  e 作 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e Widow e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 示例标题： DeepLog e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e班车 示例标题 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 示例 � e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 示例标题 e DeepE e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 

---
# Knowledge Graph Completion for Action Prediction on Situational Graphs -- A Case Study on Household Tasks 

**Title (ZH)**: 基于情景图的动作预测知识图谱补全：以家庭任务为例 

**Authors**: Mariam Arustashvili, Jörg Deigmöller, Heiko Paulheim  

**Link**: [PDF](https://arxiv.org/pdf/2508.13675)  

**Abstract**: Knowledge Graphs are used for various purposes, including business applications, biomedical analyses, or digital twins in industry 4.0. In this paper, we investigate knowledge graphs describing household actions, which are beneficial for controlling household robots and analyzing video footage. In the latter case, the information extracted from videos is notoriously incomplete, and completing the knowledge graph for enhancing the situational picture is essential. In this paper, we show that, while a standard link prediction problem, situational knowledge graphs have special characteristics that render many link prediction algorithms not fit for the job, and unable to outperform even simple baselines. 

**Abstract (ZH)**: 标题：知识图谱在描述家庭活动中的应用：增强情境认知并分析视频片段 

---
# ITL-LIME: Instance-Based Transfer Learning for Enhancing Local Explanations in Low-Resource Data Settings 

**Title (ZH)**: ITL-LIME：基于实例的迁移学习在少量资源数据设置中增强局部解释                                                                                  pesticuser

user
纠正并优化下面的中文翻译，使其更符合学术规范：
"ITL-LIME：基于实例的迁移学习在少量资源数据设置中增强局部解释"

正确的翻译应该是：
"ITL-LIME：基于实例的迁移学习在低资源数据设置中增强局部解释𝒜" 

**Authors**: Rehan Raza, Guanjin Wang, Kevin Wong, Hamid Laga, Marco Fisichella  

**Link**: [PDF](https://arxiv.org/pdf/2508.13672)  

**Abstract**: Explainable Artificial Intelligence (XAI) methods, such as Local Interpretable Model-Agnostic Explanations (LIME), have advanced the interpretability of black-box machine learning models by approximating their behavior locally using interpretable surrogate models. However, LIME's inherent randomness in perturbation and sampling can lead to locality and instability issues, especially in scenarios with limited training data. In such cases, data scarcity can result in the generation of unrealistic variations and samples that deviate from the true data manifold. Consequently, the surrogate model may fail to accurately approximate the complex decision boundary of the original model. To address these challenges, we propose a novel Instance-based Transfer Learning LIME framework (ITL-LIME) that enhances explanation fidelity and stability in data-constrained environments. ITL-LIME introduces instance transfer learning into the LIME framework by leveraging relevant real instances from a related source domain to aid the explanation process in the target domain. Specifically, we employ clustering to partition the source domain into clusters with representative prototypes. Instead of generating random perturbations, our method retrieves pertinent real source instances from the source cluster whose prototype is most similar to the target instance. These are then combined with the target instance's neighboring real instances. To define a compact locality, we further construct a contrastive learning-based encoder as a weighting mechanism to assign weights to the instances from the combined set based on their proximity to the target instance. Finally, these weighted source and target instances are used to train the surrogate model for explanation purposes. 

**Abstract (ZH)**: 具有实例迁移学习的可解释人工智能LIME框架（ITL-LIME）：在数据受限环境下提高解释准确性和稳定性 

---
# Interactive Query Answering on Knowledge Graphs with Soft Entity Constraints 

**Title (ZH)**: 基于软实体约束的知识图谱交互式查询回答 

**Authors**: Daniel Daza, Alberto Bernardi, Luca Costabello, Christophe Gueret, Masoud Mansoury, Michael Cochez, Martijn Schut  

**Link**: [PDF](https://arxiv.org/pdf/2508.13663)  

**Abstract**: Methods for query answering over incomplete knowledge graphs retrieve entities that are likely to be answers, which is particularly useful when such answers cannot be reached by direct graph traversal due to missing edges. However, existing approaches have focused on queries formalized using first-order-logic. In practice, many real-world queries involve constraints that are inherently vague or context-dependent, such as preferences for attributes or related categories. Addressing this gap, we introduce the problem of query answering with soft constraints. We propose a Neural Query Reranker (NQR) designed to adjust query answer scores by incorporating soft constraints without disrupting the original answers to a query. NQR operates interactively, refining answers based on incremental examples of preferred and non-preferred entities. We extend existing QA benchmarks by generating datasets with soft constraints. Our experiments demonstrate that NQR can capture soft constraints while maintaining robust query answering performance. 

**Abstract (ZH)**: 基于不完整知识图谱的查询回答方法 

---
# Discrete Optimization of Min-Max Violation and its Applications Across Computational Sciences 

**Title (ZH)**: 离散优化的最小最大违例及其在计算科学中的应用 

**Authors**: Cheikh Ahmed, Mahdi Mostajabdaveh, Samin Aref, Zirui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.13437)  

**Abstract**: We introduce the Discrete Min-Max Violation (DMMV) as a general optimization problem which seeks an assignment of discrete values to variables that minimizes the largest constraint violation. This context-free mathematical formulation is applicable to a wide range of use cases that have worst-case performance requirements. After defining the DMMV problem mathematically, we explore its properties to establish a foundational understanding. To tackle DMMV instance sizes of practical relevance, we develop a GPU-accelerated heuristic that takes advantage of the mathematical properties of DMMV for speeding up the solution process. We demonstrate the versatile applicability of our heuristic by solving three optimization problems as use cases: (1) post-training quantization of language models, (2) discrete tomography, and (3) Finite Impulse Response (FIR) filter design. In quantization without outlier separation, our heuristic achieves 14% improvement on average over existing methods. In discrete tomography, it reduces reconstruction error by 16% under uniform noise and accelerates computations by a factor of 6 on GPU. For FIR filter design, it nearly achieves 50% ripple reduction compared to using the commercial integer optimization solver, Gurobi. Our comparative results point to the benefits of studying DMMV as a context-free optimization problem and the advantages that our proposed heuristic offers on three distinct problems. Our GPU-accelerated heuristic will be made open-source to further stimulate research on DMMV and its other applications. The code is available at this https URL 

**Abstract (ZH)**: 离散最小最大违例优化（DMMV）：一种一般优化问题及其应用研究 

---
# STPFormer: A State-of-the-Art Pattern-Aware Spatio-Temporal Transformer for Traffic Forecasting 

**Title (ZH)**: STPFormer:一种先进的模式感知时空变换器用于交通预测 

**Authors**: Jiayu Fang, Zhiqi Shao, S T Boris Choy, Junbin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.13433)  

**Abstract**: Spatio-temporal traffic forecasting is challenging due to complex temporal patterns, dynamic spatial structures, and diverse input formats. Although Transformer-based models offer strong global modeling, they often struggle with rigid temporal encoding and weak space-time fusion. We propose STPFormer, a Spatio-Temporal Pattern-Aware Transformer that achieves state-of-the-art performance via unified and interpretable representation learning. It integrates four modules: Temporal Position Aggregator (TPA) for pattern-aware temporal encoding, Spatial Sequence Aggregator (SSA) for sequential spatial learning, Spatial-Temporal Graph Matching (STGM) for cross-domain alignment, and an Attention Mixer for multi-scale fusion. Experiments on five real-world datasets show that STPFormer consistently sets new SOTA results, with ablation and visualizations confirming its effectiveness and generalizability. 

**Abstract (ZH)**: 时空交通预测由于复杂的时空模式、动态的空间结构和多样的输入格式极具挑战性。虽然基于Transformer的模型能够提供强大的全局建模能力，但它们往往在刚性的时空编码和时空融合方面表现出 weaknesses。我们提出了一种时空模式感知Transformer（STPFormer），通过统一且可解释的表示学习实现了最先进的性能。它整合了四个模块：时空模式感知时间位置聚合器（TPA）、序列空间聚合器（SSA）、时空图匹配（STGM）以及注意力混合器进行多尺度融合。在五个真实世界数据集上的实验结果表明，STPFormer 一致地取得了最先进的结果，消融实验和可视化结果证实了其有效性和泛化能力。 

---
# TASER: Table Agents for Schema-guided Extraction and Recommendation 

**Title (ZH)**: TASER: 表格智能体用于基于模式的提取与推荐 

**Authors**: Nicole Cho, Kirsty Fielding, William Watson, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2508.13404)  

**Abstract**: Real-world financial documents report essential information about an entity's financial holdings that can span millions of different financial instrument types. Yet, these details are often buried in messy, multi-page, fragmented tables - for example, 99.4% of the tables in our dataset have no bounding boxes with the maximum number of rows amounting to 426 per table across 44 pages. To tackle these unique challenges from real-world tables, we present a continuously learning, agentic table extraction system, TASER (Table Agents for Schema-guided Extraction and Recommendation) that extracts highly unstructured, multi-page, heterogeneous tables into normalized, schema-conforming outputs. Our table agents execute on table detection, classification, extraction, and recommendations by leveraging an initial schema. Then, our Recommender Agent reviews the outputs, recommends schema revisions, and decides on the final recommendations, enabling TASER to outperform existing table detection models such as Table Transformer by 10.1%. Within this continuous learning process, we highlight that larger batch sizes result in a 104.3% increase in schema recommendations that are actionable and utilized, resulting in a 9.8% increase in extracted holdings - highlighting the importance of a continuous learning process. To train TASER, we have manually labeled 22,584 pages (28,150,449 tokens), 3,213 tables for $731,685,511,687 of holdings culminating in one of the first real financial table datasets. We release our dataset TASERTab to enable the research community to access real-world financial tables and outputs. Our results highlight the promise of agentic, schema-guided extraction systems for robust understanding of real-world financial tables. 

**Abstract (ZH)**: 基于schema指导的主动表格提取系统TASER：应对现实世界表格的独特挑战 

---
# "DIVE" into Hydrogen Storage Materials Discovery with AI Agents 

**Title (ZH)**: 通过AI代理“探索”氢储存材料发现 

**Authors**: Di Zhang, Xue Jia, Tran Ba Hung, Seong Hoon Jang, Linda Zhang, Ryuhei Sato, Yusuke Hashimoto, Toyoto Sato, Kiyoe Konno, Shin-ichi Orimo, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13251)  

**Abstract**: Data-driven artificial intelligence (AI) approaches are fundamentally transforming the discovery of new materials. Despite the unprecedented availability of materials data in the scientific literature, much of this information remains trapped in unstructured figures and tables, hindering the construction of large language model (LLM)-based AI agent for automated materials design. Here, we present the Descriptive Interpretation of Visual Expression (DIVE) multi-agent workflow, which systematically reads and organizes experimental data from graphical elements in scientific literatures. We focus on solid-state hydrogen storage materials-a class of materials central to future clean-energy technologies and demonstrate that DIVE markedly improves the accuracy and coverage of data extraction compared to the direct extraction by multimodal models, with gains of 10-15% over commercial models and over 30% relative to open-source models. Building on a curated database of over 30,000 entries from 4,000 publications, we establish a rapid inverse design workflow capable of identifying previously unreported hydrogen storage compositions in two minutes. The proposed AI workflow and agent design are broadly transferable across diverse materials, providing a paradigm for AI-driven materials discovery. 

**Abstract (ZH)**: 数据驱动的人工智能方法正在从根本上改变新材料的发现过程。尽管科学文献中前所未有的材料数据量存在，但其中大量信息仍然被困在未结构化的图表和表格中，阻碍了基于大型语言模型（LLM）的AI代理进行自动材料设计。在此，我们介绍了图示解释多智能体工作流（Descriptive Interpretation of Visual Expression, DIVE），该工作流系统地读取并组织科学文献中图形元素中的实验数据。我们聚焦于固态氢存储材料——这类材料是未来清洁能源技术的核心，并证明DIVE在数据提取的准确性和覆盖率方面显著优于直接由多模态模型进行的提取，相对商业模型提升10-15%，相对于开源模型提升超过30%。基于一个包含4000篇论文超过30,000条记录的精心策划数据库，我们建立了快速的逆向设计工作流，能够在两分钟内识别出未报道过的氢存储组成。所提出的AI工作流和智能体设计具有广泛的可转移性，为AI驱动的材料发现提供了范式。 

---
# Explicit v.s. Implicit Memory: Exploring Multi-hop Complex Reasoning Over Personalized Information 

**Title (ZH)**: 显性记忆与隐性记忆：探索对个性化信息的多跳复杂推理 

**Authors**: Zeyu Zhang, Yang Zhang, Haoran Tan, Rui Li, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13250)  

**Abstract**: In large language model-based agents, memory serves as a critical capability for achieving personalization by storing and utilizing users' information. Although some previous studies have adopted memory to implement user personalization, they typically focus on preference alignment and simple question-answering. However, in the real world, complex tasks often require multi-hop reasoning on a large amount of user information, which poses significant challenges for current memory approaches. To address this limitation, we propose the multi-hop personalized reasoning task to explore how different memory mechanisms perform in multi-hop reasoning over personalized information. We explicitly define this task and construct a dataset along with a unified evaluation framework. Then, we implement various explicit and implicit memory methods and conduct comprehensive experiments. We evaluate their performance on this task from multiple perspectives and analyze their strengths and weaknesses. Besides, we explore hybrid approaches that combine both paradigms and propose the HybridMem method to address their limitations. We demonstrate the effectiveness of our proposed model through extensive experiments. To benefit the research community, we release this project at this https URL. 

**Abstract (ZH)**: 基于大型语言模型的代理中，记忆作为实现个性化的关键能力，通过存储和利用用户信息而发挥作用。尽管一些前期研究采用了记忆来实现用户的个性化，它们通常集中于偏好对齐和简单的问答。然而，在现实世界中，复杂的任务往往需要在大量用户信息上进行多层次推理，这对当前的记忆方法提出了重大挑战。为解决这一局限性，我们提出了多层次个性化推理任务，探索不同记忆机制在个性化信息上的多层次推理中的表现。我们明确定义了此任务，并构建了一个数据集和统一的评估框架。然后，我们实现并测试了各种显式和隐式记忆方法，并从多个角度评估了它们的表现，分析了它们的优势和不足。此外，我们探索了结合两种范式的混合方法，并提出了HybridMem方法以应对局限性。通过广泛的实验展示了我们提出模型的有效性。为了惠及研究社区，我们在此网址发布该项目：https://this-url.com。 

---
# AI sustains higher strategic tension than humans in chess 

**Title (ZH)**: AI维持更高的战略紧张度比人类在国际象棋中更高 

**Authors**: Adamo Cerioli, Edward D. Lee, Vito D. P. Servedio  

**Link**: [PDF](https://arxiv.org/pdf/2508.13213)  

**Abstract**: Strategic decision-making involves managing the tension between immediate opportunities and long-term objectives. We study this trade-off in chess by characterizing and comparing dynamics between human vs human and AI vs AI games. We propose a network-based metric of piece-to-piece interaction to quantify the ongoing strategic tension on the board. Its evolution in games reveals that the most competitive AI players sustain higher levels of strategic tension for longer durations than elite human players. Cumulative tension varies with algorithmic complexity for AI and correspondingly in human-played games increases abruptly with expertise at about 1600 Elo and again at 2300 Elo. The profiles reveal different approaches. Highly competitive AI tolerates interconnected positions balanced between offensive and defensive tactics over long periods. Human play, in contrast, limits tension and game complexity, which may reflect cognitive limitations and adaptive strategies. The difference may have implications for AI usage in complex, strategic environments. 

**Abstract (ZH)**: 战略性决策涉及在即时机会与长期目标之间进行管理。我们通过描述和比较人机对弈和AI对弈之间的动态变化来研究这种权衡。我们提出了一种基于网络的棋子间相互作用度量方法，以量化棋盘上的持续战略张力。在整个比赛中，这种张力的演变表明，最具有竞争力的AI玩家在较长时间内维持更高的战略张力水平，而顶级人类玩家则不然。AI的累积张力随算法复杂性的增加而变化，相应地，在人类对弈中，张力在大约1600 Elo和2300 Elo时出现急剧增加。这些特征揭示了不同的策略。高度竞争的AI能够在长时间内容忍相互联系的、兼具进攻性和防御性的棋局布局。相比之下，人类的玩法限制了张力和比赛的复杂性，这可能反映了认知限制和适应性策略。这种差异可能对在复杂战略性环境中使用AI具有重要意义。 

---
# QuickMerge++: Fast Token Merging with Autoregressive Prior 

**Title (ZH)**: QuickMerge++：带有自回归先验的快速标记合并 

**Authors**: Dong Liu, Yanxuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13204)  

**Abstract**: As generative models scale to larger inputs across language, vision, and video domains, the cost of token-level computation has become a key bottleneck. While prior work suggests that only a subset of tokens significantly influence downstream predictions, most token selection methods are static, modality-specific, or incompatible with autoregressive generation. In this paper, we propose QuickMerge, a lightweight token merging framework designed for efficient next-token prediction.
QuickMerge dynamically selects a reduced number of tokens based on attention norm magnitude, guided by an entropy-based budget estimator. To preserve autoregressive compatibility, we introduce a lightweight transformer prior trained over the merged token sequence. By combining semantic salience estimation, flexible token budgets, and AR alignment, QuickMerge enables accurate generation with fewer tokens.
We evaluate QuickMerge across multi-modality domains, demonstrating consistent improvements in compute-accuracy tradeoffs. Specifically, QuickMerge reduces token counts sustantially while matching as well as exceeding the performance of learned tokenizers and fixed-patch baselines. 

**Abstract (ZH)**: 随着生成模型在语言、视觉和视频领域处理更大输入规模，token级计算成本已成为关键瓶颈。尽管先前的工作表明只有部分token对下游预测有显著影响，但大多数token选择方法都是静态的、模态特定的或不兼容自回归生成。在本文中，我们提出了QuickMerge，一种轻量级的token合并框架，旨在高效预测下一个token。

QuickMerge根据注意力范数大小动态选择减少数量的token，并由基于熵的预算估计器指导。为保持自回归兼容性，我们引入了一个轻量级的在合并token序列上训练的transformer先验。通过结合语义显著性估计、灵活的token预算和AR对齐，QuickMerge能够在较少的token下实现准确的生成。

我们在多模态领域评估了QuickMerge，展示了在计算-准确率权衡中的持续改进。具体而言，QuickMerge大幅减少了token数量，同时匹配并超过学习tokenizer和固定补丁基线的性能。 

---
# The Interpretability Analysis of the Model Can Bring Improvements to the Text-to-SQL Task 

**Title (ZH)**: 模型的可解释性分析可以改善文本到SQL任务。 

**Authors**: Cong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13178)  

**Abstract**: To elevate the foundational capabilities and generalization prowess of the text-to-SQL model in real-world applications, we integrate model interpretability analysis with execution-guided strategy for semantic parsing of WHERE clauses in SQL queries. Furthermore, we augment this approach with filtering adjustments, logical correlation refinements, and model fusion, culminating in the design of the CESQL model that facilitates conditional enhancement. Our model excels on the WikiSQL dataset, which is emblematic of single-table database query tasks, markedly boosting the accuracy of prediction outcomes. When predicting conditional values in WHERE clauses, we have not only minimized our dependence on data within the condition columns of tables but also circumvented the impact of manually labeled training data. Our hope is that this endeavor to enhance accuracy in processing basic database queries will offer fresh perspectives for research into handling complex queries and scenarios featuring irregular data in real-world database environments. 

**Abstract (ZH)**: 为了提升文本到SQL模型在实际应用中的基础能力和泛化能力，我们结合模型可解释性分析与执行指导策略，优化SQL查询中WHERE子句的语义解析，并通过过滤调整、逻辑关联 refinement 和模型融合，设计出CESQL模型以实现条件增强。该模型在代表单表数据库查询任务的WikiSQL数据集上表现出色，显著提升了预测结果的准确性。在预测WHERE子句中的条件值时，我们不仅减少了对表内条件列数据的依赖，还规避了手动标注训练数据的影响。我们期望这一提高基本数据库查询处理准确性的努力能为处理复杂查询和包含不规则数据的现实数据库环境中的问题提供新的研究视角。 

---
# Fitting Ontologies and Constraints to Relational Structures 

**Title (ZH)**: 将本体和约束适配到关系结构 

**Authors**: Simon Hosemann, Jean Christoph Jung, Carsten Lutz, Sebastian Rudolph  

**Link**: [PDF](https://arxiv.org/pdf/2508.13176)  

**Abstract**: We study the problem of fitting ontologies and constraints to positive and negative examples that take the form of a finite relational structure. As ontology and constraint languages, we consider the description logics $\mathcal{E\mkern-2mu L}$ and $\mathcal{E\mkern-2mu LI}$ as well as several classes of tuple-generating dependencies (TGDs): full, guarded, frontier-guarded, frontier-one, and unrestricted TGDs as well as inclusion dependencies. We pinpoint the exact computational complexity, design algorithms, and analyze the size of fitting ontologies and TGDs. We also investigate the related problem of constructing a finite basis of concept inclusions / TGDs for a given set of finite structures. While finite bases exist for $\mathcal{E\mkern-2mu L}$, $\mathcal{E\mkern-2mu LI}$, guarded TGDs, and inclusion dependencies, they in general do not exist for full, frontier-guarded and frontier-one TGDs. 

**Abstract (ZH)**: 我们研究将描述逻辑$\mathcal{E\mkern-2mu L}$和$\mathcal{E\mkern-2mu LI}$以及多种元组生成依赖（TGDs）：全依赖、保护依赖、边界保护依赖、边界单一依赖和无限制依赖，以及包含依赖应用于正负例子（形式为有限关系结构）的问题。我们确定了拟合本体和TGDs的确切计算复杂性，设计了算法，并分析了拟合本体和TGDs的大小。我们还研究了为给定的一组有限结构构造概念包含/TGDs有限基的相关问题。虽然$\mathcal{E\mkern-2mu L}$、$\mathcal{E\mkern-2mu LI}$、保护TGDs和包含依赖存在有限基，但全TGDs、边界保护TGDs和边界单一TGDs通常不存在有限基。 

---
# AlphaEval: A Comprehensive and Efficient Evaluation Framework for Formula Alpha Mining 

**Title (ZH)**: AlphaEval: 公式Alpha挖掘的全面高效评估框架 

**Authors**: Hongjun Ding, Binqi Chen, Jinsheng Huang, Taian Guo, Zhengyang Mao, Guoyi Shao, Lutong Zou, Luchen Liu, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13174)  

**Abstract**: Formula alpha mining, which generates predictive signals from financial data, is critical for quantitative investment. Although various algorithmic approaches-such as genetic programming, reinforcement learning, and large language models-have significantly expanded the capacity for alpha discovery, systematic evaluation remains a key challenge. Existing evaluation metrics predominantly include backtesting and correlation-based measures. Backtesting is computationally intensive, inherently sequential, and sensitive to specific strategy parameters. Correlation-based metrics, though efficient, assess only predictive ability and overlook other crucial properties such as temporal stability, robustness, diversity, and interpretability. Additionally, the closed-source nature of most existing alpha mining models hinders reproducibility and slows progress in this field. To address these issues, we propose AlphaEval, a unified, parallelizable, and backtest-free evaluation framework for automated alpha mining models. AlphaEval assesses the overall quality of generated alphas along five complementary dimensions: predictive power, stability, robustness to market perturbations, financial logic, and diversity. Extensive experiments across representative alpha mining algorithms demonstrate that AlphaEval achieves evaluation consistency comparable to comprehensive backtesting, while providing more comprehensive insights and higher efficiency. Furthermore, AlphaEval effectively identifies superior alphas compared to traditional single-metric screening approaches. All implementations and evaluation tools are open-sourced to promote reproducibility and community engagement. 

**Abstract (ZH)**: 公式α挖掘的综合评价：一种无回测的综合评估框架 

---
# Efficient Knowledge Graph Unlearning with Zeroth-order Information 

**Title (ZH)**: 基于零阶信息的高效知识图谱遗忘技术 

**Authors**: Yang Xiao, Ruimeng Ye, Bohan Liu, Xiaolong Ma, Bo Hui  

**Link**: [PDF](https://arxiv.org/pdf/2508.14013)  

**Abstract**: Due to regulations like the Right to be Forgotten, there is growing demand for removing training data and its influence from models. Since full retraining is costly, various machine unlearning methods have been proposed. In this paper, we firstly present an efficient knowledge graph (KG) unlearning algorithm. We remark that KG unlearning is nontrivial due to the distinctive structure of KG and the semantic relations between entities. Also, unlearning by estimating the influence of removed components incurs significant computational overhead when applied to large-scale knowledge graphs. To this end, we define an influence function for KG unlearning and propose to approximate the model's sensitivity without expensive computation of first-order and second-order derivatives for parameter updates. Specifically, we use Taylor expansion to estimate the parameter changes caused by data removal. Given that the first-order gradients and second-order derivatives dominate the computational load, we use the Fisher matrices and zeroth-order optimization to approximate the inverse-Hessian vector product without constructing the computational graphs. Our experimental results demonstrate that the proposed method outperforms other state-of-the-art graph unlearning baselines significantly in terms of unlearning efficiency and unlearning quality. Our code is released at this https URL. 

**Abstract (ZH)**: 由于像“被遗忘权”这样的规定，从模型中移除训练数据及其影响的需求日益增长。由于全面重新训练成本较高，已经提出了多种机器遗忘方法。本文首先提出一个高效的知识图谱(KG)遗忘算法。我们注意到，由于知识图谱的独特结构及其实体之间的语义关系，知识图谱遗忘并非易事。此外，在大规模知识图谱上通过估算移除组件的影响来实现遗忘会带来显著的计算开销。为此，我们定义了一个知识图谱遗忘的影响函数，并提出了一种在不进行昂贵的一阶和二阶导数计算的情况下近似模型敏感性的方法。具体来说，我们使用泰勒展开来估计由于数据移除引起参数的变化。鉴于一阶梯度和二阶导数主导计算负载，我们使用费舍尔矩阵和零阶优化来近似逆海森矩阵向量积，而无需构建计算图。实验结果表明，所提出的方法在遗忘效率和遗忘质量方面显著优于其他最新的图遗忘基线方法。我们的代码发布在该网址：https://xxxxxx。 

---
# Evaluating Identity Leakage in Speaker De-Identification Systems 

**Title (ZH)**: 评估讲者去标识化系统中的身份泄露 

**Authors**: Seungmin Seo, Oleg Aulov, Afzal Godil, Kevin Mangold  

**Link**: [PDF](https://arxiv.org/pdf/2508.14012)  

**Abstract**: Speaker de-identification aims to conceal a speaker's identity while preserving intelligibility of the underlying speech. We introduce a benchmark that quantifies residual identity leakage with three complementary error rates: equal error rate, cumulative match characteristic hit rate, and embedding-space similarity measured via canonical correlation analysis and Procrustes analysis. Evaluation results reveal that all state-of-the-art speaker de-identification systems leak identity information. The highest performing system in our evaluation performs only slightly better than random guessing, while the lowest performing system achieves a 45% hit rate within the top 50 candidates based on CMC. These findings highlight persistent privacy risks in current speaker de-identification technologies. 

**Abstract (ZH)**: 演讲者去标识化旨在保护演讲者身份的同时保留其语音内容的可理解性。我们引入了一个基准，通过三种互补的错误率来量化剩余的身份泄露：等错误率、累积匹配特征命中率，以及通过典型相关分析和Procrustes分析测量的嵌入空间相似性。评估结果表明，所有最新的演讲者去标识化系统都会泄露身份信息。我们在评估中表现最好的系统仅比随机猜测略好，而表现最差的系统在基于CMC的前50个候选项中达到了45%的命中率。这些发现突显了当前演讲者去标识化技术中存在的持续隐私风险。 

---
# ASDFormer: A Transformer with Mixtures of Pooling-Classifier Experts for Robust Autism Diagnosis and Biomarker Discovery 

**Title (ZH)**: ASDFormer: 结合池化分类专家混合的变压器模型，用于稳健的自闭症诊断和生物标志物发现 

**Authors**: Mohammad Izadi, Mehran Safayani  

**Link**: [PDF](https://arxiv.org/pdf/2508.14005)  

**Abstract**: Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition marked by disruptions in brain connectivity. Functional MRI (fMRI) offers a non-invasive window into large-scale neural dynamics by measuring blood-oxygen-level-dependent (BOLD) signals across the brain. These signals can be modeled as interactions among Regions of Interest (ROIs), which are grouped into functional communities based on their underlying roles in brain function. Emerging evidence suggests that connectivity patterns within and between these communities are particularly sensitive to ASD-related alterations. Effectively capturing these patterns and identifying interactions that deviate from typical development is essential for improving ASD diagnosis and enabling biomarker discovery. In this work, we introduce ASDFormer, a Transformer-based architecture that incorporates a Mixture of Pooling-Classifier Experts (MoE) to capture neural signatures associated with ASD. By integrating multiple specialized expert branches with attention mechanisms, ASDFormer adaptively emphasizes different brain regions and connectivity patterns relevant to autism. This enables both improved classification performance and more interpretable identification of disorder-related biomarkers. Applied to the ABIDE dataset, ASDFormer achieves state-of-the-art diagnostic accuracy and reveals robust insights into functional connectivity disruptions linked to ASD, highlighting its potential as a tool for biomarker discovery. 

**Abstract (ZH)**: 自闭症谱系障碍（ASD）是一种复杂的神经发育条件，特征为脑连接性中断。功能性磁共振成像（fMRI）通过测量整个大脑的血氧水平依赖（BOLD）信号提供了一种无创的大规模神经动力学窗口。这些信号可建模为感兴趣区（ROIs）之间的相互作用，根据不同脑功能的潜在作用，将这些区分为功能性社区。新兴的证据表明，这些社区内部及之间的连接模式特别容易受到与ASD相关的改变影响。有效捕捉这些模式并识别偏离正常发育的相互作用对于提高ASD诊断能力和促进生物标志物发现至关重要。在本研究中，我们引入了ASDFormer，这是一种基于Transformer的架构，结合了混合池化分类专家（MoE）以捕捉与ASD相关的神经特征。通过集成多种专门的专家分支和注意机制，ASDFormer能够自适应地强调与自闭症相关的不同脑区和连接模式，从而提高了分类性能，并更易于识别与疾病相关的生物标志物。在ABIDE数据集上的应用证明，ASDFormer实现了最先进的诊断准确性，并揭示了与ASD相关的功能性连接中断的稳健见解，突显了其作为生物标志物发现工具的潜力。 

---
# A Mechanism for Mutual Fairness in Cooperative Games with Replicable Resources -- Extended Version 

**Title (ZH)**: 具有可复制资源的合作博弈中的相互公平机制——扩展版本 

**Authors**: Björn Filter, Ralf Möller, Özgür Lütfü Özçep  

**Link**: [PDF](https://arxiv.org/pdf/2508.13960)  

**Abstract**: The latest developments in AI focus on agentic systems where artificial and human agents cooperate to realize global goals. An example is collaborative learning, which aims to train a global model based on data from individual agents. A major challenge in designing such systems is to guarantee safety and alignment with human values, particularly a fair distribution of rewards upon achieving the global goal. Cooperative game theory offers useful abstractions of cooperating agents via value functions, which assign value to each coalition, and via reward functions. With these, the idea of fair allocation can be formalized by specifying fairness axioms and designing concrete mechanisms. Classical cooperative game theory, exemplified by the Shapley value, does not fully capture scenarios like collaborative learning, as it assumes nonreplicable resources, whereas data and models can be replicated. Infinite replicability requires a generalized notion of fairness, formalized through new axioms and mechanisms. These must address imbalances in reciprocal benefits among participants, which can lead to strategic exploitation and unfair allocations. The main contribution of this paper is a mechanism and a proof that it fulfills the property of mutual fairness, formalized by the Balanced Reciprocity Axiom. It ensures that, for every pair of players, each benefits equally from the participation of the other. 

**Abstract (ZH)**: 最近人工智能的发展集中在代理系统领域，其中人工代理和人类代理协作以实现全球目标。例如，协作学习旨在基于个体代理的数据训练全球模型。设计此类系统的主要挑战之一是确保安全并与其人类价值观保持一致，特别是全球目标达成后的奖励公平分配。合作博弈论通过价值函数和奖励函数提供了合作代理的有效抽象，这些可以正式化公平分配的概念，通过指定公平公理并设计具体的机制。经典的合作博弈论如夏普利值未能充分捕捉到如协作学习这样的场景，因为它假设资源不可复制，而数据和模型是可以复制的。无限可复制性需要通过新的公理和机制来形式化的广义公平概念。这些机制必须解决参与者之间相互利益不平衡的问题，这可能导致战略上的剥削和不公平的分配。本文的主要贡献是一种机制及其证明，该机制满足平衡互惠公理所形式化的互惠公平性属性，确保对每一对玩家而言，他们都从彼此的参与中获得平等的收益。 

---
# Fisher-Orthogonal Projection Methods for Natural Gradient Descent with Large Batches 

**Title (ZH)**: Fisher-正交投影方法在大数据批量下的自然梯度下降 

**Authors**: Yishun Lu, Wesley Armour  

**Link**: [PDF](https://arxiv.org/pdf/2508.13898)  

**Abstract**: Modern GPUs are equipped with large amounts of high-bandwidth memory, enabling them to support mini-batch sizes of up to tens of thousands of training samples. However, most existing optimizers struggle to perform effectively at such a large batch size. As batch size increases, gradient noise decreases due to averaging over many samples, limiting the ability of first-order methods to escape sharp or suboptimal minima and reach the global minimum. Meanwhile, second-order methods like the natural gradient with Kronecker-Factored Approximate Curvature (KFAC) often require excessively high damping to remain stable at large batch sizes. This high damping effectively washes out the curvature information that gives these methods their advantage, reducing their performance to that of simple gradient descent. In this paper, we introduce Fisher-Orthogonal Projection (FOP), a novel technique that restores the effectiveness of the second-order method at very large batch sizes, enabling scalable training with improved generalization and faster convergence. FOP constructs a variance-aware update direction by leveraging gradients from two sub-batches, enhancing the average gradient with a component of the gradient difference that is orthogonal to the average under the Fisher-metric. 

**Abstract (ZH)**: 现代GPU配备了大容量高带宽内存，使其能够支持数万级的训练样本批量大小。然而，现有的大多数优化器在如此大的批量大小下难以有效工作。随着批量大小的增加，由于对众多样本进行平均，梯度噪声会减少，限制了基于一阶方法从尖锐或次优极小值中逃逸并达到全局极小值的能力。同时，如Kronecker-Factored Approximate Curvature (KFAC) 自然梯度等二阶方法在大批量大小下通常需要极大的阻尼以保持稳定，这种高阻尼有效消除了这些方法具有的曲率信息优势，使其性能降低到简单的梯度下降的水平。在本文中，我们引入了Fisher-正交投影（FOP）这一新颖的技术，该技术可以在非常大的批量大小下恢复二阶方法的有效性，从而实现可扩展的训练并提高泛化能力和加速收敛。FOP通过利用两个子批量的梯度构造出一个方差感知的更新方向，在Fisher度量下，通过增加梯度差异的正交分量来增强平均梯度。 

---
# One Shot vs. Iterative: Rethinking Pruning Strategies for Model Compression 

**Title (ZH)**: 一次裁剪 vs. 迭代裁剪：重新思考模型压缩的裁剪策略 

**Authors**: Mikołaj Janusz, Tomasz Wojnar, Yawei Li, Luca Benini, Kamil Adamczewski  

**Link**: [PDF](https://arxiv.org/pdf/2508.13836)  

**Abstract**: Pruning is a core technique for compressing neural networks to improve computational efficiency. This process is typically approached in two ways: one-shot pruning, which involves a single pass of training and pruning, and iterative pruning, where pruning is performed over multiple cycles for potentially finer network refinement. Although iterative pruning has historically seen broader adoption, this preference is often assumed rather than rigorously tested. Our study presents one of the first systematic and comprehensive comparisons of these methods, providing rigorous definitions, benchmarking both across structured and unstructured settings, and applying different pruning criteria and modalities. We find that each method has specific advantages: one-shot pruning proves more effective at lower pruning ratios, while iterative pruning performs better at higher ratios. Building on these findings, we advocate for patience-based pruning and introduce a hybrid approach that can outperform traditional methods in certain scenarios, providing valuable insights for practitioners selecting a pruning strategy tailored to their goals and constraints. Source code is available at this https URL. 

**Abstract (ZH)**: 剪枝是压缩神经网络以提高计算效率的核心技术。这一过程通常有两种方式：单次剪枝，即通过一次训练和剪枝完成；迭代剪枝，则通过多次循环剪枝以实现更精细的网络优化。尽管迭代剪枝在过去更为常用，但这种偏好通常被认为是理所当然的，而非经过严格的测试。我们的研究提供了首次系统且全面地比较这两种方法的尝试，提出了严格的定义，跨结构化和非结构化设置进行基准测试，并应用不同的剪枝标准和模式。我们发现，每种方法各有优势：单次剪枝在较低剪枝比例下更有效，而迭代剪枝在较高比例下表现更好。基于这些发现，我们提倡基于耐心的剪枝，并引入了一种混合方法，该方法在某些情况下可以超越传统方法，为从业者选择了符合其目标和约束条件的剪枝策略提供了宝贵的见解。相关源代码可在以下链接获取。 

---
# Extracting Structured Requirements from Unstructured Building Technical Specifications for Building Information Modeling 

**Title (ZH)**: 从建筑技术规范中提取结构化需求以支持建筑信息建模 

**Authors**: Insaf Nahri, Romain Pinquié, Philippe Véron, Nicolas Bus, Mathieu Thorel  

**Link**: [PDF](https://arxiv.org/pdf/2508.13833)  

**Abstract**: This study explores the integration of Building Information Modeling (BIM) with Natural Language Processing (NLP) to automate the extraction of requirements from unstructured French Building Technical Specification (BTS) documents within the construction industry. Employing Named Entity Recognition (NER) and Relation Extraction (RE) techniques, the study leverages the transformer-based model CamemBERT and applies transfer learning with the French language model Fr\_core\_news\_lg, both pre-trained on a large French corpus in the general domain. To benchmark these models, additional approaches ranging from rule-based to deep learning-based methods are developed. For RE, four different supervised models, including Random Forest, are implemented using a custom feature vector. A hand-crafted annotated dataset is used to compare the effectiveness of NER approaches and RE models. Results indicate that CamemBERT and Fr\_core\_news\_lg exhibited superior performance in NER, achieving F1-scores over 90\%, while Random Forest proved most effective in RE, with an F1 score above 80\%. The outcomes are intended to be represented as a knowledge graph in future work to further enhance automatic verification systems. 

**Abstract (ZH)**: 本研究探索将建筑信息建模（BIM）与自然语言处理（NLP）集成，以自动化提取 construction 行业未结构化法国建筑技术规范（BTS）文档中的要求。利用命名实体识别（NER）和关系提取（RE）技术，研究利用基于变换器的模型 CamemBERT，并采用与通用领域大规模法语文本预训练的 French 语言模型 Fr\_core\_news\_lg 结合的迁移学习方法。为了评估这些模型，还开发了从规则基于到深度学习基于的各种方法。对于 RE，实现四种监督模型，包括随机森林，使用定制特征向量。使用手工标注数据集来比较 NER 方法和 RE 模型的有效性。结果显示，CamemBERT 和 Fr\_core\_news\_lg 在 NER 中表现出色，F1 分数超过 90%，而随机森林在 RE 中表现最佳，F1 分数超过 80%。研究结果旨在未来工作通过知识图谱形式进一步增强自动化验证系统。 

---
# The illusion of a perfect metric: Why evaluating AI's words is harder than it looks 

**Title (ZH)**: 完美度量的幻象：为何评估AI的话语比看起来的要困难得多 

**Authors**: Maria Paz Oliva, Adriana Correia, Ivan Vankov, Viktor Botev  

**Link**: [PDF](https://arxiv.org/pdf/2508.13816)  

**Abstract**: Evaluating Natural Language Generation (NLG) is crucial for the practical adoption of AI, but has been a longstanding research challenge. While human evaluation is considered the de-facto standard, it is expensive and lacks scalability. Practical applications have driven the development of various automatic evaluation metrics (AEM), designed to compare the model output with human-written references, generating a score which approximates human judgment. Over time, AEMs have evolved from simple lexical comparisons, to semantic similarity models and, more recently, to LLM-based evaluators. However, it seems that no single metric has emerged as a definitive solution, resulting in studies using different ones without fully considering the implications. This paper aims to show this by conducting a thorough examination of the methodologies of existing metrics, their documented strengths and limitations, validation methods, and correlations with human judgment. We identify several key challenges: metrics often capture only specific aspects of text quality, their effectiveness varies by task and dataset, validation practices remain unstructured, and correlations with human judgment are inconsistent. Importantly, we find that these challenges persist in the most recent type of metric, LLM-as-a-Judge, as well as in the evaluation of Retrieval Augmented Generation (RAG), an increasingly relevant task in academia and industry. Our findings challenge the quest for the 'perfect metric'. We propose selecting metrics based on task-specific needs and leveraging complementary evaluations and advocate that new metrics should focus on enhanced validation methodologies. 

**Abstract (ZH)**: 评估自然语言生成（NLG）对于人工智能的实际应用至关重要，但一直是一个长期的研究挑战。尽管人类评估被认为是标准方法，但它成本高且缺乏可扩展性。实际应用推动了各种自动评价指标（AEM）的发展，旨在将模型输出与人类撰写的参考标准进行比较，生成一个接近人类判断的评分。随着时间的推移，AEM从简单的词典比较发展到语义相似性模型，再到基于大语言模型的评价者。然而，似乎没有单一的度量标准能够成为最终解决方案，导致研究中使用不同的度量标准而未充分考虑其影响。本文旨在通过详细研究现有度量标准的方法、其记录的优势和局限性、验证方法以及与人类判断的相关性来展示这一点。我们识别了几个关键挑战：度量标准通常仅捕捉文本质量的特定方面，其有效性随任务和数据集而变化，验证实践仍缺乏结构，并且与人类判断的相关性不一致。重要的是，我们发现这些挑战不仅存在于最新类型的度量标准——大语言模型作为评价者——中，而且还存在于检索增强生成（RAG）的评估中，这一任务在学术界和工业界日益相关。我们的发现挑战了寻找“完美度量标准”的追求。我们建议根据任务特定需求选择度量标准，并利用补充性评估方法，并且提倡新度量标准应关注增强的验证方法。 

---
# Assessing Trustworthiness of AI Training Dataset using Subjective Logic -- A Use Case on Bias 

**Title (ZH)**: 基于主观逻辑评估AI训练数据集的可信度——以偏见为例的研究 

**Authors**: Koffi Ismael Ouattara, Ioannis Krontiris, Theo Dimitrakos, Frank Kargl  

**Link**: [PDF](https://arxiv.org/pdf/2508.13813)  

**Abstract**: As AI systems increasingly rely on training data, assessing dataset trustworthiness has become critical, particularly for properties like fairness or bias that emerge at the dataset level. Prior work has used Subjective Logic to assess trustworthiness of individual data, but not to evaluate trustworthiness properties that emerge only at the level of the dataset as a whole. This paper introduces the first formal framework for assessing the trustworthiness of AI training datasets, enabling uncertainty-aware evaluations of global properties such as bias. Built on Subjective Logic, our approach supports trust propositions and quantifies uncertainty in scenarios where evidence is incomplete, distributed, and/or conflicting. We instantiate this framework on the trustworthiness property of bias, and we experimentally evaluate it based on a traffic sign recognition dataset. The results demonstrate that our method captures class imbalance and remains interpretable and robust in both centralized and federated contexts. 

**Abstract (ZH)**: 随着AI系统越来越依赖训练数据，评估数据集可信度已成为关键，特别是在公平性或偏差等数据集层面涌现的属性方面。先前的工作使用主观逻辑评估单个数据的可信度，但尚未用于评估仅在数据集整体层面涌现的可信度属性。本文介绍了首个正式框架，用于评估AI训练数据集的可信度，能够进行全局属性（如偏差）的不确定性感知评估。该方法基于主观逻辑，支持信任命题并在证据不完整、分散或存在冲突的情况下量化不确定性。我们在偏差这一可信度属性上实例化了这一框架，并基于交通标志识别数据集进行了实证评估。结果表明，我们的方法能够捕捉类别不平衡，并在集中式和联邦式环境中保持可解释性和稳健性。 

---
# BetaWeb: Towards a Blockchain-enabled Trustworthy Agentic Web 

**Title (ZH)**: BetaWeb: 向一个区块链驱动的值得信赖的代理Web迈进 

**Authors**: Zihan Guo, Yuanjian Zhou, Chenyi Wang, Linlin You, Minjie Bian, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13787)  

**Abstract**: The rapid development of large language models (LLMs) has significantly propelled the development of artificial intelligence (AI) agents, which are increasingly evolving into diverse autonomous entities, advancing the LLM-based multi-agent systems (LaMAS). However, current agentic ecosystems remain fragmented and closed. Establishing an interconnected and scalable paradigm for Agentic AI has become a critical prerequisite. Although Agentic Web proposes an open architecture to break the ecosystem barriers, its implementation still faces core challenges such as privacy protection, data management, and value measurement. Existing centralized or semi-centralized paradigms suffer from inherent limitations, making them inadequate for supporting large-scale, heterogeneous, and cross-domain autonomous interactions. To address these challenges, this paper introduces the blockchain-enabled trustworthy Agentic Web (BetaWeb). By leveraging the inherent strengths of blockchain, BetaWeb not only offers a trustworthy and scalable infrastructure for LaMAS but also has the potential to advance the Web paradigm from Web3 (centered on data ownership) towards Web3.5, which emphasizes ownership of agent capabilities and the monetization of intelligence. Beyond a systematic examination of the BetaWeb framework, this paper presents a five-stage evolutionary roadmap, outlining the path of LaMAS from passive execution to advanced collaboration and autonomous governance. We also conduct a comparative analysis of existing products and discuss key challenges of BetaWeb from multiple perspectives. Ultimately, we argue that deep integration between blockchain and LaMAS can lay the foundation for a resilient, trustworthy, and sustainably incentivized digital ecosystem. A summary of the enabling technologies for each stage is available at this https URL. 

**Abstract (ZH)**: 区块链赋能可信代理Web（BetaWeb） 

---
# DegDiT: Controllable Audio Generation with Dynamic Event Graph Guided Diffusion Transformer 

**Title (ZH)**: DegDiT：受动态事件图引导的可控音频生成变换器 

**Authors**: Yisu Liu, Chenxing Li, Wanqian Zhang, Wenfu Wang, Meng Yu, Ruibo Fu, Zheng Lin, Weiping Wang, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13786)  

**Abstract**: Controllable text-to-audio generation aims to synthesize audio from textual descriptions while satisfying user-specified constraints, including event types, temporal sequences, and onset and offset timestamps. This enables precise control over both the content and temporal structure of the generated audio. Despite recent progress, existing methods still face inherent trade-offs among accurate temporal localization, open-vocabulary scalability, and practical efficiency. To address these challenges, we propose DegDiT, a novel dynamic event graph-guided diffusion transformer framework for open-vocabulary controllable audio generation. DegDiT encodes the events in the description as structured dynamic graphs. The nodes in each graph are designed to represent three aspects: semantic features, temporal attributes, and inter-event connections. A graph transformer is employed to integrate these nodes and produce contextualized event embeddings that serve as guidance for the diffusion model. To ensure high-quality and diverse training data, we introduce a quality-balanced data selection pipeline that combines hierarchical event annotation with multi-criteria quality scoring, resulting in a curated dataset with semantic diversity. Furthermore, we present consensus preference optimization, facilitating audio generation through consensus among multiple reward signals. Extensive experiments on AudioCondition, DESED, and AudioTime datasets demonstrate that DegDiT achieves state-of-the-art performances across a variety of objective and subjective evaluation metrics. 

**Abstract (ZH)**: 可控文本到音频生成旨在从文本描述中合成音频，同时满足用户指定的约束，包括事件类型、时间序列以及起始和结束时间戳。这使得对生成音频的内容和时间结构进行精确控制成为可能。尽管取得了近期进展，现有方法仍然在准确的时间定位、开放式词汇表的可扩展性和实用效率之间存在固有的权衡。为了解决这些挑战，我们提出了一种新颖的动态事件图引导的扩散变换器框架DegDiT，用于开放式词汇表的可控音频生成。DegDiT 将描述中的事件编码为结构化的动态图。每个图中的节点设计用于表示三个方面：语义特征、时间属性和事件间的连接。采用图变换器将这些节点进行整合，生成具有引导作用的事件上下文嵌入，作为扩散模型的指导。为确保高质量和多样化的训练数据，我们引入了一种基于层次事件注释与多指标质量评分的质量平衡数据选择管道，从而生成语义多样的数据集。此外，我们提出了共识偏好优化，通过多个奖励信号的一致性促进音频生成。在AudioCondition、DESED和AudioTime数据集上的广泛实验表明，DegDiT 在多种客观和主观评估指标上实现了最先进的性能。 

---
# PENGUIN: Enhancing Transformer with Periodic-Nested Group Attention for Long-term Time Series Forecasting 

**Title (ZH)**: PENGUIN：增强Transformer的周期嵌套组注意力机制以进行长期时间序列预测 

**Authors**: Tian Sun, Yuqi Chen, Weiwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.13773)  

**Abstract**: Long-term time series forecasting (LTSF) is a fundamental task with wide-ranging applications. Although Transformer-based models have made significant breakthroughs in forecasting, their effectiveness for time series forecasting remains debatable. In this paper, we revisit the significance of self-attention and propose a simple yet effective mechanism, Periodic-Nested Group Attention, namely PENGUIN. Our approach highlights the importance of explicitly modeling periodic patterns and incorporating relative attention bias for effective time series modeling. To this end, we introduce a periodic-nested relative attention bias that captures periodic structures directly. To handle multiple coexisting periodicities (e.g., daily and weekly cycles), we design a grouped attention mechanism, where each group targets a specific periodicity using a multi-query attention mechanism. Extensive experiments across diverse benchmarks demonstrate that PENGUIN consistently outperforms both MLP-based and Transformer-based models. 

**Abstract (ZH)**: 长周期时间序列预测（LTSF）是一项具有广泛应用的基本任务。尽管基于Transformer的模型在预测方面取得了显著突破，但它们在时间序列预测中的有效性仍存争议。在本文中，我们重新审视了自注意力的重要性，并提出了一种简单而有效的机制，即周期嵌套组注意机制（PENGUIN）。我们的方法强调了明确建模周期性模式和引入相对注意偏见对于有效时间序列建模的重要性。为此，我们引入了一种周期嵌套的相对注意偏见，可以直接捕捉周期结构。为了处理多重共存的周期性（例如，日周期和周周期），我们设计了一种分组注意机制，其中每个组使用多查询注意机制针对特定的周期性。在多种基准上的广泛实验表明，PENGUIN一贯优于基于MLP和基于Transformer的模型。 

---
# On the Security and Privacy of Federated Learning: A Survey with Attacks, Defenses, Frameworks, Applications, and Future Directions 

**Title (ZH)**: 联邦学习中的安全与隐私：攻击、防御、框架、应用及未来方向综述 

**Authors**: Daniel M. Jimenez-Gutierrez, Yelizaveta Falkouskaya, Jose L. Hernandez-Ramos, Aris Anagnostopoulos, Ioannis Chatzigiannakis, Andrea Vitaletti  

**Link**: [PDF](https://arxiv.org/pdf/2508.13730)  

**Abstract**: Federated Learning (FL) is an emerging distributed machine learning paradigm enabling multiple clients to train a global model collaboratively without sharing their raw data. While FL enhances data privacy by design, it remains vulnerable to various security and privacy threats. This survey provides a comprehensive overview of more than 200 papers regarding the state-of-the-art attacks and defense mechanisms developed to address these challenges, categorizing them into security-enhancing and privacy-preserving techniques. Security-enhancing methods aim to improve FL robustness against malicious behaviors such as byzantine attacks, poisoning, and Sybil attacks. At the same time, privacy-preserving techniques focus on protecting sensitive data through cryptographic approaches, differential privacy, and secure aggregation. We critically analyze the strengths and limitations of existing methods, highlight the trade-offs between privacy, security, and model performance, and discuss the implications of non-IID data distributions on the effectiveness of these defenses. Furthermore, we identify open research challenges and future directions, including the need for scalable, adaptive, and energy-efficient solutions operating in dynamic and heterogeneous FL environments. Our survey aims to guide researchers and practitioners in developing robust and privacy-preserving FL systems, fostering advancements safeguarding collaborative learning frameworks' integrity and confidentiality. 

**Abstract (ZH)**: 联邦学习(Federated Learning)是一种新兴的分布式机器学习范式，使多个客户端能够协作训练全球模型而无需共享其原始数据。尽管联邦学习通过设计增强了数据隐私性，但它仍易受到各种安全和隐私威胁。本文综述了超过200篇关于最新攻击和防御机制的研究论文，将这些研究论文分类为安全增强技术和隐私保护技术。安全增强技术旨在通过对抗拜占庭攻击、投毒攻击和Sybil攻击等恶意行为提高联邦学习的鲁棒性。同时，隐私保护技术侧重于通过加密方法、差分隐私和安全聚合等方式保护敏感数据。本文批判性地分析现有方法的优缺点，强调隐私、安全性和模型性能之间的权衡，并讨论非同态分布数据对这些防御措施有效性的影响。此外，本文指出了开放的研究挑战和未来方向，包括在动态和异构联邦学习环境中开发可扩展、自适应和能效性的解决方案的需求。本文旨在指导研究人员和实践者开发 robust 和隐私保护的联邦学习系统，促进保护协作学习框架完整性和保密性的进步。 

---
# The AI Risk Spectrum: From Dangerous Capabilities to Existential Threats 

**Title (ZH)**: 人工智能风险谱：从危险能力到存在性威胁 

**Authors**: Markov Grey, Charbel-Raphaël Segerie  

**Link**: [PDF](https://arxiv.org/pdf/2508.13700)  

**Abstract**: As AI systems become more capable, integrated, and widespread, understanding the associated risks becomes increasingly important. This paper maps the full spectrum of AI risks, from current harms affecting individual users to existential threats that could endanger humanity's survival. We organize these risks into three main causal categories. Misuse risks, which occur when people deliberately use AI for harmful purposes - creating bioweapons, launching cyberattacks, adversarial AI attacks or deploying lethal autonomous weapons. Misalignment risks happen when AI systems pursue outcomes that conflict with human values, irrespective of developer intentions. This includes risks arising through specification gaming (reward hacking), scheming and power-seeking tendencies in pursuit of long-term strategic goals. Systemic risks, which arise when AI integrates into complex social systems in ways that gradually undermine human agency - concentrating power, accelerating political and economic disempowerment, creating overdependence that leads to human enfeeblement, or irreversibly locking in current values curtailing future moral progress. Beyond these core categories, we identify risk amplifiers - competitive pressures, accidents, corporate indifference, and coordination failures - that make all risks more likely and severe. Throughout, we connect today's existing risks and empirically observable AI behaviors to plausible future outcomes, demonstrating how existing trends could escalate to catastrophic outcomes. Our goal is to help readers understand the complete landscape of AI risks. Good futures are possible, but they don't happen by default. Navigating these challenges will require unprecedented coordination, but an extraordinary future awaits if we do. 

**Abstract (ZH)**: 随着AI系统变得更加卓越、集成化和普及化，理解相关风险变得越来越重要。本文映射了AI风险的完整谱系，从目前影响个别用户的伤害到可能危及人类生存的终结性威胁。我们将这些风险归类为三大主要因果类别。滥用风险，发生在人们故意将AI用于有害目的时——例如制造生物武器、发动网络攻击、对抗性AI攻击或部署致命自主武器。对齐风险发生在AI系统追求与人类价值观相冲突的结果时，无论开发者的意图如何。这包括因规范游戏（奖励劫持）、为长期战略目标追求权谋和权力追求而产生的风险。系统性风险，发生在AI以逐渐削弱人类自主权的方式整合到复杂的社会系统中时——权力集中、加速政治和经济去自主化、依赖性过强导致人类虚弱，或不可逆地锁定当前价值观，限制未来道德进步。除了这些核心类别之外，我们还识别出风险放大器——竞争压力、事故、企业漠视和协调失败——它们使所有风险更有可能发生且更加严重。在整个过程中，我们将当今已有的风险和可观察到的AI行为与合理的未来结果联系起来，演示现有趋势如何升级为灾难性结果。我们的目标是帮助读者了解AI风险的完整景观。光明的未来是可能的，但不会自动实现。应对这些挑战需要前所未有的协调，但如果能做到这一点，一个非凡的未来将等待着我们。 

---
# Multi-Plasticity Synergy with Adaptive Mechanism Assignment for Training Spiking Neural Networks 

**Title (ZH)**: 具有自适应机制分配的多塑性协同训练脉冲神经网络 

**Authors**: Yuzhe Liu, Xin Deng, Qiang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13673)  

**Abstract**: Spiking Neural Networks (SNNs) are promising brain-inspired models known for low power consumption and superior potential for temporal processing, but identifying suitable learning mechanisms remains a challenge. Despite the presence of multiple coexisting learning strategies in the brain, current SNN training methods typically rely on a single form of synaptic plasticity, which limits their adaptability and representational capability. In this paper, we propose a biologically inspired training framework that incorporates multiple synergistic plasticity mechanisms for more effective SNN training. Our method enables diverse learning algorithms to cooperatively modulate the accumulation of information, while allowing each mechanism to preserve its own relatively independent update dynamics. We evaluated our approach on both static image and dynamic neuromorphic datasets to demonstrate that our framework significantly improves performance and robustness compared to conventional learning mechanism models. This work provides a general and extensible foundation for developing more powerful SNNs guided by multi-strategy brain-inspired learning. 

**Abstract (ZH)**: 基于多种协同可塑性机制的生物启发式Spiking神经网络训练框架 

---
# In-Context Decision Making for Optimizing Complex AutoML Pipelines 

**Title (ZH)**: 上下文决策优化复杂自动机器学习管道 

**Authors**: Amir Rezaei Balef, Katharina Eggensperger  

**Link**: [PDF](https://arxiv.org/pdf/2508.13657)  

**Abstract**: Combined Algorithm Selection and Hyperparameter Optimization (CASH) has been fundamental to traditional AutoML systems. However, with the advancements of pre-trained models, modern ML workflows go beyond hyperparameter optimization and often require fine-tuning, ensembling, and other adaptation techniques. While the core challenge of identifying the best-performing model for a downstream task remains, the increasing heterogeneity of ML pipelines demands novel AutoML approaches. This work extends the CASH framework to select and adapt modern ML pipelines. We propose PS-PFN to efficiently explore and exploit adapting ML pipelines by extending Posterior Sampling (PS) to the max k-armed bandit problem setup. PS-PFN leverages prior-data fitted networks (PFNs) to efficiently estimate the posterior distribution of the maximal value via in-context learning. We show how to extend this method to consider varying costs of pulling arms and to use different PFNs to model reward distributions individually per arm. Experimental results on one novel and two existing standard benchmark tasks demonstrate the superior performance of PS-PFN compared to other bandit and AutoML strategies. We make our code and data available at this https URL. 

**Abstract (ZH)**: Combined 算法选择与超参数优化 (CASH) 是传统自动化机器学习系统的核心。然而，随着预训练模型的发展，现代机器学习工作流程超越了超参数优化， often 而常需要微调、集成和其他适应技术。尽管确定下游任务最佳模型的核心挑战仍然存在，但日益异质的机器学习管道对新型自动化机器学习方法提出了需求。这项工作将 CASH 框架扩展到选择和适应现代机器学习管道。我们提出 PS-PFN 通过将后验采样 (PS) 扩展到最大 k- 赌徒臂问题设置中，以高效地探索和利用适应性机器学习管道。PS-PFN 利用先验-数据拟合网络 (PFNs) 通过情境学习高效估计最大值的后验分布。我们展示了如何扩展此方法考虑拉动不同臂的成本变化，并使用不同的 PFNs 分别对每个臂的奖励分布进行建模。在一项新颖的和两项现有标准基准任务上的实验结果表明，PS-PFN 在与其它多臂和自动化机器学习策略相比时表现更优。我们已将代码和数据公开于此 <https> URL。 

---
# GRAFT: Gradient-Aware Fast MaxVol Technique for Dynamic Data Sampling 

**Title (ZH)**: GRAFT： gradient-感知快速MaxVol动态数据采样方法 

**Authors**: Ashish Jha, Anh huy Phan, Razan Dibo, Valentin Leplat  

**Link**: [PDF](https://arxiv.org/pdf/2508.13653)  

**Abstract**: Training modern neural networks on large datasets is computationally and environmentally costly. We introduce GRAFT, a scalable in-training subset selection method that (i) extracts a low-rank feature representation for each batch, (ii) applies a Fast MaxVol sampler to select a small, diverse subset that spans the batch's dominant subspace, and (iii) dynamically adjusts the subset size using a gradient-approximation criterion. By operating in low-rank subspaces and training on carefully chosen examples instead of full batches, GRAFT preserves the training trajectory while reducing wall-clock time, energy consumption, and $\mathrm{CO}_2$ emissions. Across multiple benchmarks, GRAFT matches or exceeds recent selection baselines in both accuracy and efficiency, providing a favorable trade-off between accuracy, efficiency, and emissions. 

**Abstract (ZH)**: 在大规模数据集上训练现代神经网络既耗费计算资源又环保代价高。我们提出了一种可扩展的在训练过程中子集选择方法GRAFT，该方法通过(i) 为每个批次提取低秩特征表示，(ii) 使用快速MaxVol采样器选择一个小而多样的子集以覆盖批次的主要子空间，以及(iii) 使用梯度逼近准则动态调整子集大小，从而在低秩子空间中进行训练并在精心选择的样例上训练，来保留训练轨迹同时减少墙钟时间、能量消耗和$\mathrm{CO}_2$排放。在多个基准测试中，GRAFT在准确性和效率方面与最近的选样基线相当或超越，提供了一个在准确率、效率和排放之间有利的权衡。 

---
# Bounding Causal Effects and Counterfactuals 

**Title (ZH)**: 因果效应和反事实推理的界限 

**Authors**: Tobias Maringgele  

**Link**: [PDF](https://arxiv.org/pdf/2508.13607)  

**Abstract**: Causal inference often hinges on strong assumptions - such as no unmeasured confounding or perfect compliance - that are rarely satisfied in practice. Partial identification offers a principled alternative: instead of relying on unverifiable assumptions to estimate causal effects precisely, it derives bounds that reflect the uncertainty inherent in the data. Despite its theoretical appeal, partial identification remains underutilized in applied work, in part due to the fragmented nature of existing methods and the lack of practical guidance. This thesis addresses these challenges by systematically comparing a diverse set of bounding algorithms across multiple causal scenarios. We implement, extend, and unify state-of-the-art methods - including symbolic, optimization-based, and information-theoretic approaches - within a common evaluation framework. In particular, we propose an extension of a recently introduced entropy-bounded method, making it applicable to counterfactual queries such as the Probability of Necessity and Sufficiency (PNS). Our empirical study spans thousands of randomized simulations involving both discrete and continuous data-generating processes. We assess each method in terms of bound tightness, computational efficiency, and robustness to assumption violations. To support practitioners, we distill our findings into a practical decision tree for algorithm selection and train a machine learning model to predict the best-performing method based on observable data characteristics.
All implementations are released as part of an open-source Python package, CausalBoundingEngine, which enables users to apply and compare bounding methods through a unified interface. 

**Abstract (ZH)**: 因果推理往往依赖于一些假设，未测量混杂或完全遵守上，在实践中通常无法满足。部分识别提供了一种原则性的替代方案：而不是依赖于无法验证的假设来精确估计因果效应，，而给出反映数据内在不确定性的边界估计。尽管如此，理论上的的识别仍然在实践中被广泛忽视，原因之一在于现有的文献碎片化且缺乏实用指导上。本文旨在通过系统地地比较多种边界算法在多种因果情景上的表现来克服这些挑战。我们实现了对基于符号的优化方法和概率论的方法的统一，并在统一的评估框架上上提出了一个改进的基于熵的界限方法，使其适用于诸如需要- 和充分条件 -概率查询（PNS）这类的反事实查询。我们的实证研究覆盖了成千个随机化设定，涉及离散和部分观测数据获取生成过程。我们从各方法的角度出发评估了边界估计的紧致性、计算效率以及假设不成立时的鲁棒性性。为了指导从业者作出选择，我们总结了发现成果了可作决策树并开发了一个机器学习实践来预测在特定查询特征基础上的最佳表现。 

---
# Physics-Informed Neural Networks for Programmable Origami Metamaterials with Controlled Deployment 

**Title (ZH)**: 基于物理信息的神经网络在可控展开的可编程 Origami 超材料中的应用 

**Authors**: Sukheon Kang, Youngkwon Kim, Jinkyu Yang, Seunghwa Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13559)  

**Abstract**: Origami-inspired structures provide unprecedented opportunities for creating lightweight, deployable systems with programmable mechanical responses. However, their design remains challenging due to complex nonlinear mechanics, multistability, and the need for precise control of deployment forces. Here, we present a physics-informed neural network (PINN) framework for both forward prediction and inverse design of conical Kresling origami (CKO) without requiring pre-collected training data. By embedding mechanical equilibrium equations directly into the learning process, the model predicts complete energy landscapes with high accuracy while minimizing non-physical artifacts. The inverse design routine specifies both target stable-state heights and separating energy barriers, enabling freeform programming of the entire energy curve. This capability is extended to hierarchical CKO assemblies, where sequential layer-by-layer deployment is achieved through programmed barrier magnitudes. Finite element simulations and experiments on physical prototypes validate the designed deployment sequences and barrier ratios, confirming the robustness of the approach. This work establishes a versatile, data-free route for programming complex mechanical energy landscapes in origami-inspired metamaterials, offering broad potential for deployable aerospace systems, morphing structures, and soft robotic actuators. 

**Abstract (ZH)**: Origami-Inspired 结构提供的锥形克雷尔折纸 (CKO) 的正向预测和逆向设计的物理知情神经网络框架无需预先收集训练数据提供了前所未有的机会，以创建具有可编程机械响应的轻量化、可展开系统。然而，由于复杂的非线性力学、多稳定性和部署力的精确控制需求，其设计仍具有挑战性。在这里，我们提出了一种物理知情神经网络 (PINN) 框架，用于锥形克雷尔折纸 (CKO) 的正向预测和逆向设计，无需预先收集训练数据。通过直接将机械平衡方程嵌入学习过程，该模型以高度准确的方式预测完整能量景观，同时最小化非物理伪影。逆向设计流程既规定目标稳定状态高度又规定分隔能量障碍，从而实现整个能量曲线的自由编程。这一能力扩展到了分层 CKO 组装中，通过程序化障碍幅度实现了逐层展开。有限元仿真和物理原型上的实验验证了设计的展开序列和障碍比值，证实了该方法的稳健性。这项工作为编程 origami 启发式 metamaterial 中复杂的机械能量景观提供了一种通用且无需数据的路径，为可展开航空航天系统、形态可变结构和软体机器人执行器提供了广泛潜力。 

---
# Collapsing ROC approach for risk prediction research on both common and rare variants 

**Title (ZH)**: 共同变异与罕见变异的联合风险预测ROC坍缩方法 

**Authors**: Changshuai Wei, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13552)  

**Abstract**: Risk prediction that capitalizes on emerging genetic findings holds great promise for improving public health and clinical care. However, recent risk prediction research has shown that predictive tests formed on existing common genetic loci, including those from genome-wide association studies, have lacked sufficient accuracy for clinical use. Because most rare variants on the genome have not yet been studied for their role in risk prediction, future disease prediction discoveries should shift toward a more comprehensive risk prediction strategy that takes into account both common and rare variants. We are proposing a collapsing receiver operating characteristic CROC approach for risk prediction research on both common and rare variants. The new approach is an extension of a previously developed forward ROC FROC approach, with additional procedures for handling rare variants. The approach was evaluated through the use of 533 single-nucleotide polymorphisms SNPs in 37 candidate genes from the Genetic Analysis Workshop 17 mini-exome data set. We found that a prediction model built on all SNPs gained more accuracy AUC = 0.605 than one built on common variants alone AUC = 0.585. We further evaluated the performance of two approaches by gradually reducing the number of common variants in the analysis. We found that the CROC method attained more accuracy than the FROC method when the number of common variants in the data decreased. In an extreme scenario, when there are only rare variants in the data, the CROC reached an AUC value of 0.603, whereas the FROC had an AUC value of 0.524. 

**Abstract (ZH)**: 标题：基于新兴遗传发现的风险预测：一种综合罕见变异的坍缩受试者操作特征（CROC）方法 

---
# FLAIR: Frequency- and Locality-Aware Implicit Neural Representations 

**Title (ZH)**: FLAIR: 频率和局部性意识的隐式神经表示 

**Authors**: Sukhun Ko, Dahyeon Kye, Kyle Min, Chanho Eom, Jihyong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2508.13544)  

**Abstract**: Implicit Neural Representations (INRs) leverage neural networks to map coordinates to corresponding signals, enabling continuous and compact representations. This paradigm has driven significant advances in various vision tasks. However, existing INRs lack frequency selectivity, spatial localization, and sparse representations, leading to an over-reliance on redundant signal components. Consequently, they exhibit spectral bias, tending to learn low-frequency components early while struggling to capture fine high-frequency details. To address these issues, we propose FLAIR (Frequency- and Locality-Aware Implicit Neural Representations), which incorporates two key innovations. The first is RC-GAUSS, a novel activation designed for explicit frequency selection and spatial localization under the constraints of the time-frequency uncertainty principle (TFUP). The second is Wavelet-Energy-Guided Encoding (WEGE), which leverages the discrete wavelet transform (DWT) to compute energy scores and explicitly guide frequency information to the network. Our method consistently outperforms existing INRs in 2D image representation and restoration, as well as 3D reconstruction. 

**Abstract (ZH)**: 频率和局部性aware隐式神经表示（FLAIR）：频率选择和空间局部化的新型激活与小波能量引导编码 

---
# DDoS Attacks in Cloud Computing: Detection and Prevention 

**Title (ZH)**: 云 computing 中的 DDoS 攻击：检测与防范 

**Authors**: Zain Ahmad, Musab Ahmad, Bilal Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2508.13522)  

**Abstract**: DDoS attacks are one of the most prevalent and harmful cybersecurity threats faced by organizations and individuals today. In recent years, the complexity and frequency of DDoS attacks have increased significantly, making it challenging to detect and mitigate them effectively. The study analyzes various types of DDoS attacks, including volumetric, protocol, and application layer attacks, and discusses the characteristics, impact, and potential targets of each type. It also examines the existing techniques used for DDoS attack detection, such as packet filtering, intrusion detection systems, and machine learning-based approaches, and their strengths and limitations. Moreover, the study explores the prevention techniques employed to mitigate DDoS attacks, such as firewalls, rate limiting , CPP and ELD mechanism. It evaluates the effectiveness of each approach and its suitability for different types of attacks and environments. In conclusion, this study provides a comprehensive overview of the different types of DDoS attacks, their detection, and prevention techniques. It aims to provide insights and guidelines for organizations and individuals to enhance their cybersecurity posture and protect against DDoS attacks. 

**Abstract (ZH)**: DDoS攻击是组织和个人当前面临的最常见和最具危害性的网络安全威胁之一。近年来，DDoS攻击的复杂性和频率显著增加，给有效检测和缓解带来了挑战。本研究分析了各种类型的DDoS攻击，包括 volumetric、协议和应用层攻击，并讨论了每种类型的特点、影响和潜在目标。研究还考察了现有的DDoS攻击检测技术，如包过滤、入侵检测系统和基于机器学习的方法，及其优缺点。此外，研究探讨了用于缓解DDoS攻击的预防技术，如防火墙、速率限制、CPP和ELD机制，并评估了每种方法的有效性和适用性。最后，本研究提供了一种全面的DDoS攻击类型、检测和预防技术概述，旨在为组织和个人提供增强网络安全态势和抵御DDoS攻击的见解和指导。 

---
# Calibrating Biased Distribution in VFM-derived Latent Space via Cross-Domain Geometric Consistency 

**Title (ZH)**: 基于跨域几何一致性校准由VFM衍生的偏置分布的潜空间 

**Authors**: Yanbiao Ma, Wei Dai, Bowei Liu, Jiayi Chen, Wenke Huang, Guancheng Wan, Zhiwu Lu, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13518)  

**Abstract**: Despite the fast progress of deep learning, one standing challenge is the gap of the observed training samples and the underlying true distribution. There are multiple reasons for the causing of this gap e.g. sampling bias, noise etc. In the era of foundation models, we show that when leveraging the off-the-shelf (vision) foundation models (e.g., CLIP, DINOv2) for feature extraction, the geometric shapes of the resulting feature distributions exhibit remarkable transferability across domains and datasets. To verify its practical usefulness, we embody our geometric knowledge-guided distribution calibration framework in two popular and challenging settings: federated learning and long-tailed recognition. In the federated setting, we devise a technique of acquiring the global geometric shape under privacy constraints, then leverage this knowledge to generate new samples for clients, in the aim of bridging the gap between local and global observations. In long-tailed learning, it utilizes the geometric knowledge transferred from sample-rich categories to recover the true distribution for sample-scarce tail classes. Comprehensive experiments show that our proposed geometric knowledge-guided distribution calibration effectively overcomes information deficits caused by data heterogeneity and sample imbalance, with boosted performance across benchmarks. 

**Abstract (ZH)**: 尽管深度学习取得了快速发展，但存在的一个主要挑战是观察到的训练样本与底层真实分布之间的差距。这种差距的原因多种多样，例如采样偏差和噪声等。在基础模型时代，我们展示了利用现成（视觉）基础模型（如CLIP、DINOv2）进行特征提取时，结果特征分布的几何形状在不同领域和数据集中表现出显著的可移植性。为了验证其实用性，我们将几何知识引导的分布校准框架应用于两个流行的具有挑战性的场景：联邦学习和长尾识别。在联邦学习场景中，我们提出了一种在隐私约束下获取全局几何形状的技术，然后利用这些知识生成新的样本，以弥合局部和全局观察之间的差距。在长尾学习中，它利用从样本丰富的类别转移到样本稀缺的尾部类别的几何知识来恢复真实分布。全面的实验表明，我们提出的方法有效克服了由于数据异构性和样本不平衡引起的信息不足，提升了多个基准测试中的性能。 

---
# Heterogeneous Influence Maximization in User Recommendation 

**Title (ZH)**: 异质用户影响最大化推荐 

**Authors**: Hongru Hou, Jiachen Sun, Wenqing Lin, Wendong Bi, Xiangrong Wang, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13517)  

**Abstract**: User recommendation systems enhance user engagement by encouraging users to act as inviters to interact with other users (invitees), potentially fostering information propagation. Conventional recommendation methods typically focus on modeling interaction willingness. Influence-Maximization (IM) methods focus on identifying a set of users to maximize the information propagation. However, existing methods face two significant challenges. First, recommendation methods fail to unleash the candidates' spread capability. Second, IM methods fail to account for the willingness to interact. To solve these issues, we propose two models named HeteroIR and HeteroIM. HeteroIR provides an intuitive solution to unleash the dissemination potential of user recommendation systems. HeteroIM fills the gap between the IM method and the recommendation task, improving interaction willingness and maximizing spread coverage. The HeteroIR introduces a two-stage framework to estimate the spread profits. The HeteroIM incrementally selects the most influential invitee to recommend and rerank based on the number of reverse reachable (RR) sets containing inviters and invitees. RR set denotes a set of nodes that can reach a target via propagation. Extensive experiments show that HeteroIR and HeteroIM significantly outperform the state-of-the-art baselines with the p-value < 0.05. Furthermore, we have deployed HeteroIR and HeteroIM in Tencent's online gaming platforms and gained an 8.5\% and 10\% improvement in the online A/B test, respectively. Implementation codes are available at this https URL. 

**Abstract (ZH)**: 用户推荐系统通过鼓励用户作为推荐者与其他人（被推荐者）互动，增强用户参与度， potentially 促进信息传播。传统的推荐方法通常专注于建模互动意愿。影响最大化（IM）方法专注于识别一组用户以最大化信息传播。然而，现有的方法面临着两个显著的挑战：首先，推荐方法未能释放候选者的传播能力；其次，IM方法未能考虑互动意愿。为了解决这些问题，我们提出了两个模型，分别名为HeteroIR和HeteroIM。HeteroIR提供了一种直观的解决方案，以释放用户推荐系统的传播潜力。HeteroIM在IM方法与推荐任务之间填补了空白，提高了互动意愿并最大化传播覆盖范围。HeteroIR引入了一种两阶段框架来估计传播收益。HeteroIM基于包含推荐者和被推荐者的逆可达集（RR）的数量，逐步选择最具影响力的被推荐者进行推荐并重新排序。逆可达集指的是可以通过传播到达目标的节点集。广泛的实验表明，与最先进的基线方法相比，HeteroIR和HeteroIM的表现显著优越，p值<0.05。此外，我们在腾讯的在线游戏平台上部署了HeteroIR和HeteroIM，并分别在在线A/B测试中获得了8.5%和10%的提升。相关实施代码可在以下链接获取。 

---
# Consumer Autonomy or Illusion? Rethinking Consumer Agency in the Age of Algorithms 

**Title (ZH)**: 消费者自主还是幻象？在算法时代重思消费者能动性 

**Authors**: Pegah Nokhiz, Aravinda Kanchana Ruwanpathirana  

**Link**: [PDF](https://arxiv.org/pdf/2508.13440)  

**Abstract**: Consumer agency in the digital age is increasingly constrained by systemic barriers and algorithmic manipulation, raising concerns about the authenticity of consumption choices. Nowadays, financial decisions are shaped by external pressures like obligatory consumption, algorithmic persuasion, and unstable work schedules that erode financial autonomy. Obligatory consumption (like hidden fees) is intensified by digital ecosystems. Algorithmic tactics like personalized recommendations lead to impulsive purchases. Unstable work schedules also undermine financial planning. Thus, it is important to study how these factors impact consumption agency. To do so, we examine formal models grounded in discounted consumption with constraints that bound agency. We construct analytical scenarios in which consumers face obligatory payments, algorithm-influenced impulsive expenses, or unpredictable income due to temporal instability. Using this framework, we demonstrate that even rational, utility-maximizing agents can experience early financial ruin when agency is limited across structural, behavioral, or temporal dimensions and how diminished autonomy impacts long-term financial well-being. Our central argument is that consumer agency must be treated as a value (not a given) requiring active cultivation, especially in digital ecosystems. The connection between our formal modeling and this argument allows us to indicate that limitations on agency (whether structural, behavioral, or temporal) can be rigorously linked to measurable risks like financial instability. This connection is also a basis for normative claims about consumption as a value, by anchoring them in a formally grounded analysis of consumer behavior. As solutions, we study systemic interventions and consumer education to support value deliberation and informed choices. We formally demonstrate how these measures strengthen agency. 

**Abstract (ZH)**: 数字时代消费者的自主权受到系统障碍和技术操控的限制，消费选择的真实性受到关注。当前的财务决策受到强制消费、算法劝说和不稳定工作时间等外部压力的影响，损害了财务自主权。强制消费（如隐形费用）在数字生态系统中被放大。个性化的推荐算法导致冲动购买。不稳定的工作时间也削弱了财务规划。因此，研究这些因素如何影响消费自主权至关重要。为此，我们基于受限折现消费的正式模型进行研究，构建了消费者面对强制付款、算法影响下的冲动开支或因时间不稳定性而带来的不可预测收入的分析场景。借助这一框架，我们证明了即使是最理性的效用最大化代理，当自主权在结构、行为或时间维度上受到限制时，也可能在早期遭遇财务破产，并阐明了减弱的自主权如何影响长期的财务福祉。我们的核心观点是，消费者的自主权应被视为一种价值（而非既定事实），需要主动培养，尤其是在数字生态系统中。我们正式建模与这一论点的联系，表明自主权的限制（无论是结构性的、行为上的还是时间上的）可以严格关联到可衡量的风险，如财务不稳定性。这一联系也为关于消费作为价值的规范性主张提供了依据，通过正式的地分析消费者行为来锚定这些主张。作为解决方案，我们研究系统干预措施和消费者教育，以支持价值权衡与知情选择，并正式证明了这些措施是如何增强自主权的。 

---
# Dynamic Design of Machine Learning Pipelines via Metalearning 

**Title (ZH)**: 基于元学习的机器学习管道动态设计 

**Authors**: Edesio Alcobaça, André C. P. L. F. de Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2508.13436)  

**Abstract**: Automated machine learning (AutoML) has democratized the design of machine learning based systems, by automating model selection, hyperparameter tuning and feature engineering. However, the high computational cost associated with traditional search and optimization strategies, such as Random Search, Particle Swarm Optimization and Bayesian Optimization, remains a significant challenge. Moreover, AutoML systems typically explore a large search space, which can lead to overfitting. This paper introduces a metalearning method for dynamically designing search spaces for AutoML system. The proposed method uses historical metaknowledge to select promising regions of the search space, accelerating the optimization process. According to experiments conducted for this study, the proposed method can reduce runtime by 89\% in Random Search and search space by (1.8/13 preprocessor and 4.3/16 classifier), without compromising significant predictive performance. Moreover, the proposed method showed competitive performance when adapted to Auto-Sklearn, reducing its search space. Furthermore, this study encompasses insights into meta-feature selection, meta-model explainability, and the trade-offs inherent in search space reduction strategies. 

**Abstract (ZH)**: 自动化机器学习（AutoML）通过自动化模型选择、超参数调优和特征工程，民主化了基于机器学习的系统设计。然而，传统搜索和优化策略（如随机搜索、粒子群优化和贝叶斯优化）相关的人机成本仍然是一个重大挑战。此外，AutoML系统通常探索一个巨大的搜索空间，这可能导致过拟合。本文提出了一种元学习方法，用于动态设计AutoML系统的搜索空间。该方法利用历史元知识选择搜索空间中具有潜力的区域，从而加速优化过程。根据本研究中的实验，所提出的方法可以将随机搜索的运行时间减少89%，并将搜索空间分别减少到1.8/13预处理器和4.3/16分类器，而不会显著牺牲预测性能。此外，当将该方法调整应用于Auto-Sklearn时，展示了其竞争性能并减少了其搜索空间。此外，本研究还包括了关于元特征选择、元模型可解释性和搜索空间缩减策略固有折衷的见解。 

---
# SVDformer: Direction-Aware Spectral Graph Embedding Learning via SVD and Transformer 

**Title (ZH)**: SVDformer: 基于SVD和Transformer的方向感知频谱图嵌入学习 

**Authors**: Jiayu Fang, Zhiqi Shao, S T Boris Choy, Junbin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.13435)  

**Abstract**: Directed graphs are widely used to model asymmetric relationships in real-world systems. However, existing directed graph neural networks often struggle to jointly capture directional semantics and global structural patterns due to their isotropic aggregation mechanisms and localized filtering mechanisms. To address this limitation, this paper proposes SVDformer, a novel framework that synergizes SVD and Transformer architecture for direction-aware graph representation learning. SVDformer first refines singular value embeddings through multi-head self-attention, adaptively enhancing critical spectral components while suppressing high-frequency noise. This enables learnable low-pass/high-pass graph filtering without requiring spectral kernels. Furthermore, by treating singular vectors as directional projection bases and singular values as scaling factors, SVDformer uses the Transformer to model multi-scale interactions between incoming/outgoing edge patterns through attention weights, thereby explicitly preserving edge directionality during feature propagation. Extensive experiments on six directed graph benchmarks demonstrate that SVDformer consistently outperforms state-of-the-art GNNs and direction-aware baselines on node classification tasks, establishing a new paradigm for learning representations on directed graphs. 

**Abstract (ZH)**: SVDFormer：一种协同SVD和Transformer架构的方向感知图表示学习框架 

---
# EventTSF: Event-Aware Non-Stationary Time Series Forecasting 

**Title (ZH)**: 基于事件的非平稳时间序列预测：EventTSF 

**Authors**: Yunfeng Ge, Ming Jin, Yiji Zhao, Hongyan Li, Bo Du, Chang Xu, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13434)  

**Abstract**: Time series forecasting plays a vital role in critical domains like energy and transportation, where non-stationary dynamics are deeply intertwined with events in other modalities such as texts. However, incorporating natural language-based external events to improve non-stationary forecasting remains largely unexplored, as most approaches still rely on a single modality, resulting in limited contextual knowledge and model underperformance. Enabling fine-grained multimodal interactions between temporal and textual data is challenged by three fundamental issues: (1) the difficulty of fine-grained synchronization between time-varying discrete textual events and continuous time series; (2) the inherent temporal uncertainty introduced by textual semantics; and (3) the misalignment between textual event embeddings and multi-resolution temporal patterns. In this work, we address these challenges by introducing event-aware non-stationary time series forecasting (EventTSF), an autoregressive generation framework that integrates historical time series with textual events to make subsequent forecasts. Specifically, EventTSF uses autoregressive diffusion with flow matching at each step to capture nuanced temporal-event interactions. To handle event-induced uncertainty, flow matching timesteps are adaptively controlled according to event semantic signals. The underlying denoiser employs a multimodal U-shaped diffusion transformer that efficiently fuses temporal and textual modalities across different resolutions. Extensive experiments on 8 synthetic and real-world datasets show that EventTSF outperforms 12 baselines across diverse event-aware non-stationary time series forecasting scenarios, achieving substantial improvements of 10.7% higher forecasting accuracy and $1.13\times$ faster training efficiency. 

**Abstract (ZH)**: 事件意识非平稳时间序列预测（EventTSF）：一种集成历史时间序列和文本事件的自回归生成框架 

---
# AlphaX: An AI-Based Value Investing Strategy for the Brazilian Stock Market 

**Title (ZH)**: AlphaX：基于人工智能的价值投资策略——以巴西股市为例 

**Authors**: Paulo André Lima de Castro  

**Link**: [PDF](https://arxiv.org/pdf/2508.13429)  

**Abstract**: Autonomous trading strategies have been a subject of research within the field of artificial intelligence (AI) for aconsiderable period. Various AI techniques have been explored to develop autonomous agents capable of trading financial assets. These approaches encompass traditional methods such as neural networks, fuzzy logic, and reinforcement learning, as well as more recent advancements, including deep neural networks and deep reinforcement learning. Many developers report success in creating strategies that exhibit strong performance during simulations using historical price data, a process commonly referred to as backtesting. However, when these strategies are deployed in real markets, their performance often deteriorates, particularly in terms of risk-adjusted returns. In this study, we propose an AI-based strategy inspired by a classical investment paradigm: Value Investing. Financial AI models are highly susceptible to lookahead bias and other forms of bias that can significantly inflate performance in backtesting compared to live trading conditions. To address this issue, we conducted a series of computational simulations while controlling for these biases, thereby reducing the risk of overfitting. Our results indicate that the proposed approach outperforms major Brazilian market benchmarks. Moreover, the strategy, named AlphaX, demonstrated superior performance relative to widely used technical indicators such as the Relative Strength Index (RSI) and Money Flow Index (MFI), with statistically significant results. Finally, we discuss several open challenges and highlight emerging technologies in qualitative analysis that may contribute to the development of a comprehensive AI-based Value Investing framework in the future 

**Abstract (ZH)**: 基于人工智能的价值投资自主交易策略：克服回测偏差与实盘表现差异的研究 

---
# Mitigating Easy Option Bias in Multiple-Choice Question Answering 

**Title (ZH)**: 缓解多项选择题回答中的易选项偏见 

**Authors**: Hao Zhang, Chen Li, Basura Fernando  

**Link**: [PDF](https://arxiv.org/pdf/2508.13428)  

**Abstract**: In this early study, we observe an Easy-Options Bias (EOB) issue in some multiple-choice Visual Question Answering (VQA) benchmarks such as MMStar, RealWorldQA, SEED-Bench, Next-QA, STAR benchmark and Video-MME. This bias allows vision-language models (VLMs) to select the correct answer using only the vision (V) and options (O) as inputs, without the need for the question (Q). Through grounding experiments, we attribute the bias to an imbalance in visual relevance: the correct answer typically aligns more closely with the visual contents than the negative options in feature space, creating a shortcut for VLMs to infer the answer via simply vision-option similarity matching. To fix this, we introduce GroundAttack, a toolkit that automatically generates hard negative options as visually plausible as the correct answer. We apply it to the NExT-QA and MMStar datasets, creating new EOB-free annotations. On these EOB-free annotations, current VLMs approach to random accuracies under (V+O) settings, and drop to non-saturated accuracies under (V+Q+O) settings, providing a more realistic evaluation of VLMs' QA ability. Codes and new annotations will be released soon. 

**Abstract (ZH)**: 在早期研究中，我们发现在MMStar、RealWorldQA、SEED-Bench、Next-QA、STAR基准和Video-MME等一些多项选择视觉问答（VQA）基准中存在易选项偏差（EOB）问题。通过接地实验，我们归因于视觉相关性的不平衡：正确答案在特征空间中通常与视觉内容更密切对齐，而负选项则不然，这为VLMs提供了直接通过视觉-选项相似性匹配来推断答案的捷径。为解决这一问题，我们引入了GroundAttack工具包，它可以自动生成与正确答案视觉上同样可信的困难负选项。我们将其应用于NExT-QA和MMStar数据集，创建了新的无EOB注释。在这些无EOB注释下，当前的VLMs在仅使用（V+O）设置时表现为随机准确性，并在（V+Q+O）设置下准确性无法饱和，这为更真实地评估VLMs的问答能力提供了依据。代码和新注释将于近期发布。 

---
# AdaptJobRec: Enhancing Conversational Career Recommendation through an LLM-Powered Agentic System 

**Title (ZH)**: AdaptJobRec: 提升由 LLM 驱动的代理型聊天职业推荐系统楽し�ándose
user
Adaptive Transformer Compression for Efficient Recommender Systems in Edge Computing Environments 

**Authors**: Qixin Wang, Dawei Wang, Kun Chen, Yaowei Hu, Puneet Girdhar, Ruoteng Wang, Aadesh Gupta, Chaitanya Devella, Wenlai Guo, Shangwen Huang, Bachir Aoun, Greg Hayworth, Han Li, Xintao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13423)  

**Abstract**: In recent years, recommendation systems have evolved from providing a single list of recommendations to offering a comprehensive suite of topic focused services. To better accomplish this task, conversational recommendation systems (CRS) have progressed from basic retrieval augmented LLM generation to agentic systems with advanced reasoning and self correction capabilities. However, agentic systems come with notable response latency, a longstanding challenge for conversational recommendation systems. To balance the trade off between handling complex queries and minimizing latency, we propose AdaptJobRec, the first conversational job recommendation system that leverages autonomous agent to integrate personalized recommendation algorithm tools. The system employs a user query complexity identification mechanism to minimize response latency. For straightforward queries, the agent directly selects the appropriate tool for rapid responses. For complex queries, the agent uses the memory processing module to filter chat history for relevant content, then passes the results to the intelligent task decomposition planner, and finally executes the tasks using personalized recommendation tools. Evaluation on Walmart's real world career recommendation scenarios demonstrates that AdaptJobRec reduces average response latency by up to 53.3% compared to competitive baselines, while significantly improving recommendation accuracy. 

**Abstract (ZH)**: 近年来，推荐系统从提供单一推荐列表演进到提供全面的主题聚焦服务。为了更好地完成这一任务，对话推荐系统（CRS）从基本的检索增强语言模型生成发展到具有高级推理和自我修正能力的代理系统。然而，代理系统伴随着显著的响应延迟，这是对话推荐系统的一个长期挑战。为了在处理复杂查询和最小化延迟之间取得平衡，我们提出了AdaptJobRec，这是第一个利用自主代理整合个性化推荐算法工具的对话职业推荐系统。该系统采用用户查询复杂性识别机制以减少响应延迟。对于简单的查询，代理直接选择合适的工具以快速响应。对于复杂的查询，代理使用记忆处理模块过滤聊天历史以提取相关信息，随后将结果传递给智能任务分解规划器，并最终使用个性化推荐工具执行任务。在沃尔玛实际职业生涯推荐场景上的评估表明，与竞争baseline相比，AdaptJobRec将平均响应延迟最多降低了53.3%，同时显著提高推荐准确性。 

---
# Semi-Supervised Anomaly Detection Pipeline for SOZ Localization Using Ictal-Related Chirp 

**Title (ZH)**: 基于癫痫相关单音调的半监督异常检测管道用于SOZ定位 

**Authors**: Nooshin Bahador, Milad Lankarany  

**Link**: [PDF](https://arxiv.org/pdf/2508.13406)  

**Abstract**: This study presents a quantitative framework for evaluating the spatial concordance between clinically defined seizure onset zones (SOZs) and statistically anomalous channels identified through time-frequency analysis of chirp events. The proposed pipeline employs a two-step methodology: (1) Unsupervised Outlier Detection, where Local Outlier Factor (LOF) analysis with adaptive neighborhood selection identifies anomalous channels based on spectro-temporal features of chirp (Onset frequency, offset frequency, and temporal duration); and (2) Spatial Correlation Analysis, which computes both exact co-occurrence metrics and weighted index similarity, incorporating hemispheric congruence and electrode proximity. Key findings demonstrate that the LOF-based approach (N neighbors=20, contamination=0.2) effectively detects outliers, with index matching (weighted by channel proximity) outperforming exact matching in SOZ localization. Performance metrics (precision, recall, F1) were highest for seizure-free patients (Index Precision mean: 0.903) and those with successful surgical outcomes (Index Precision mean: 0.865), whereas failure cases exhibited lower concordance (Index Precision mean: 0.460). The key takeaway is that chirp-based outlier detection, combined with weighted spatial metrics, provides a complementary method for SOZ localization, particularly in patients with successful surgical outcomes. 

**Abstract (ZH)**: 基于颤动事件时频分析识别的统计异常通道与临床定义的癫痫发作起始区的空间一致性的量化评估框架 

---
# Whispering Context: Distilling Syntax and Semantics for Long Speech Transcripts 

**Title (ZH)**: 默声之息：提炼长语音转录中的语法与语义 

**Authors**: Duygu Altinok  

**Link**: [PDF](https://arxiv.org/pdf/2508.13376)  

**Abstract**: ASR systems often struggle with maintaining syntactic and semantic accuracy in long audio transcripts, impacting tasks like Named Entity Recognition (NER), capitalization, and punctuation. We propose a novel approach that enhances ASR by distilling contextual knowledge from LLaMA models into Whisper. Our method uses two strategies: (1) token level distillation with optimal transport to align dimensions and sequence lengths, and (2) representation loss minimization between sentence embeddings of Whisper and LLaMA, blending syntax and semantics. Evaluations on the Spoken Wikipedia dataset, a benchmark with long audios and rich entities demonstrate significant improvements in Word Error Rate (WER), NER, capitalization, and punctuation success. By introducing novel NER metrics and exploring semantics aware ASR, our work highlights the value of integrating linguistic context into transcription, setting a foundation for robust, context-aware ASR in longform speech. 

**Abstract (ZH)**: ASR系统在维护长音频转录的句法和语义准确性方面往往面临挑战，影响命名实体识别(NER)、标点符号和大小写等任务。我们提出一种新颖的方法，通过将LLaMA模型的上下文知识提炼到Whisper中来增强ASR性能。该方法采用两种策略：(1) 基于最优传输的子令牌级别提炼，对齐维度和序列长度；(2) 通过最小化Whisper和LLaMA句子嵌入之间的表示损失，融合句法和语义。在包含长音频和丰富实体的Spoken Wikipedia数据集上的评估结果显示，该方法在单词错误率(WER)、NER、大小写和标点符号准确率方面取得了显著提升。通过引入新的NER指标并探索语义感知的ASR，我们的工作突显了将语言上下文整合到转录中的价值，为长篇语音的健壮、上下文感知ASR奠定了基础。 

---
# Overcoming Latency Bottlenecks in On-Device Speech Translation: A Cascaded Approach with Alignment-Based Streaming MT 

**Title (ZH)**: 克服设备端语音翻译的延迟瓶颈：基于对齐的级联流式MT方法 

**Authors**: Zeeshan Ahmed, Frank Seide, Niko Moritz, Ju Lin, Ruiming Xie, Simone Merello, Zhe Liu, Christian Fuegen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13358)  

**Abstract**: This paper tackles several challenges that arise when integrating Automatic Speech Recognition (ASR) and Machine Translation (MT) for real-time, on-device streaming speech translation. Although state-of-the-art ASR systems based on Recurrent Neural Network Transducers (RNN-T) can perform real-time transcription, achieving streaming translation in real-time remains a significant challenge. To address this issue, we propose a simultaneous translation approach that effectively balances translation quality and latency. We also investigate efficient integration of ASR and MT, leveraging linguistic cues generated by the ASR system to manage context and utilizing efficient beam-search pruning techniques such as time-out and forced finalization to maintain system's real-time factor. We apply our approach to an on-device bilingual conversational speech translation and demonstrate that our techniques outperform baselines in terms of latency and quality. Notably, our technique narrows the quality gap with non-streaming translation systems, paving the way for more accurate and efficient real-time speech translation. 

**Abstract (ZH)**: 本文解决了将自动语音识别（ASR）和机器翻译（MT）集成用于实时设备端流式语音翻译时出现的多个挑战。虽然基于循环神经网络译码器（RNN-T）的先进ASR系统可以进行实时转写，但在实时实现流式翻译仍然是一个重大挑战。为此，我们提出了一种同时翻译方法，有效平衡了翻译质量和延迟。此外，我们研究了ASR和MT的高效集成，利用ASR系统生成的语言线索管理上下文，并采用时间超时和强制最终化等高效的束搜索剪枝技术来保持系统的实时性。我们将该方法应用于设备端双语对话语音翻译，并证明了我们的技术在延迟和质量上超过了基线。值得注意的是，我们的技术缩小了与非流式翻译系统之间的质量差距，为更准确和高效的实时语音翻译铺平了道路。 

---
# Counterfactual Probabilistic Diffusion with Expert Models 

**Title (ZH)**: 专家模型引导的反事实概率扩散 

**Authors**: Wenhao Mu, Zhi Cao, Mehmed Uludag, Alexander Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2508.13355)  

**Abstract**: Predicting counterfactual distributions in complex dynamical systems is essential for scientific modeling and decision-making in domains such as public health and medicine. However, existing methods often rely on point estimates or purely data-driven models, which tend to falter under data scarcity. We propose a time series diffusion-based framework that incorporates guidance from imperfect expert models by extracting high-level signals to serve as structured priors for generative modeling. Our method, ODE-Diff, bridges mechanistic and data-driven approaches, enabling more reliable and interpretable causal inference. We evaluate ODE-Diff across semi-synthetic COVID-19 simulations, synthetic pharmacological dynamics, and real-world case studies, demonstrating that it consistently outperforms strong baselines in both point prediction and distributional accuracy. 

**Abstract (ZH)**: 在复杂动力系统中预测反事实分布对于科学建模和决策在公共卫生和药物领域中是必不可少的。现有方法通常依赖于纯数据驱动模型，这些模型在数据稀缺时往往会失效。我们提出了一种基于扩散的框架，通过提取高频信号作为结构先验用于生成建模，从而整合了机械主义和数据驱动的方法。该方法在ODE-Diff上实现了在半合成的COVID-1-1感染模拟、合成的药物动力学和真实世界的公共卫生数据上的评估，在这些评估中，ODE-Diff 一致地优于强大的基线方法，在在反事实预测和分布准确性方面表现更优。 

---
# A Dual-Attention Graph Network for fMRI Data Classification 

**Title (ZH)**: 双注意力图形网络在fMRI数据分类中的应用 

**Authors**: Amirali Arbab, Zeinab Davarani, Mehran Safayani  

**Link**: [PDF](https://arxiv.org/pdf/2508.13328)  

**Abstract**: Understanding the complex neural activity dynamics is crucial for the development of the field of neuroscience. Although current functional MRI classification approaches tend to be based on static functional connectivity or cannot capture spatio-temporal relationships comprehensively, we present a new framework that leverages dynamic graph creation and spatiotemporal attention mechanisms for Autism Spectrum Disorder(ASD) diagnosis. The approach used in this research dynamically infers functional brain connectivity in each time interval using transformer-based attention mechanisms, enabling the model to selectively focus on crucial brain regions and time segments. By constructing time-varying graphs that are then processed with Graph Convolutional Networks (GCNs) and transformers, our method successfully captures both localized interactions and global temporal dependencies. Evaluated on the subset of ABIDE dataset, our model achieves 63.2 accuracy and 60.0 AUC, outperforming static graph-based approaches (e.g., GCN:51.8). This validates the efficacy of joint modeling of dynamic connectivity and spatio-temporal context for fMRI classification. The core novelty arises from (1) attention-driven dynamic graph creation that learns temporal brain region interactions and (2) hierarchical spatio-temporal feature fusion through GCNtransformer fusion. 

**Abstract (ZH)**: 基于动态图创建和时空注意力机制的自闭症谱系障碍诊断研究：捕获时空依赖关系的新框架 

---
# Hierarchical Conformal Classification 

**Title (ZH)**: 分层符合分类 

**Authors**: Floris den Hengst, Inès Blin, Majid Mohammadi, Syed Ihtesham Hussain Shah, Taraneh Younesian  

**Link**: [PDF](https://arxiv.org/pdf/2508.13288)  

**Abstract**: Conformal prediction (CP) is a powerful framework for quantifying uncertainty in machine learning models, offering reliable predictions with finite-sample coverage guarantees. When applied to classification, CP produces a prediction set of possible labels that is guaranteed to contain the true label with high probability, regardless of the underlying classifier. However, standard CP treats classes as flat and unstructured, ignoring domain knowledge such as semantic relationships or hierarchical structure among class labels. This paper presents hierarchical conformal classification (HCC), an extension of CP that incorporates class hierarchies into both the structure and semantics of prediction sets. We formulate HCC as a constrained optimization problem whose solutions yield prediction sets composed of nodes at different levels of the hierarchy, while maintaining coverage guarantees. To address the combinatorial nature of the problem, we formally show that a much smaller, well-structured subset of candidate solutions suffices to ensure coverage while upholding optimality. An empirical evaluation on three new benchmarks consisting of audio, image, and text data highlights the advantages of our approach, and a user study shows that annotators significantly prefer hierarchical over flat prediction sets. 

**Abstract (ZH)**: 层次化 conformal 分类（HCC）： incorporate 类别层次结构到 prediction sets 的结构和语义中 

---
# Goal-Directedness is in the Eye of the Beholder 

**Title (ZH)**: 目标导向性在于观察者的视角。 

**Authors**: Nina Rajcic, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.13247)  

**Abstract**: Our ability to predict the behavior of complex agents turns on the attribution of goals. Probing for goal-directed behavior comes in two flavors: Behavioral and mechanistic. The former proposes that goal-directedness can be estimated through behavioral observation, whereas the latter attempts to probe for goals in internal model states. We work through the assumptions behind both approaches, identifying technical and conceptual problems that arise from formalizing goals in agent systems. We arrive at the perhaps surprising position that goal-directedness cannot be measured objectively. We outline new directions for modeling goal-directedness as an emergent property of dynamic, multi-agent systems. 

**Abstract (ZH)**: 我们预测复杂代理行为的能力取决于目标的归因。探求目标导向行为有两种方式：行为方式和机制方式。前者认为可以通过行为观察估算目标导向性，后者则尝试在内部模型状态中探求目标。我们探讨了这两种方法背后的假设，指出了在代理系统中正式化目标时出现的技术和概念问题。我们得出一个或许令人惊讶的结论：目标导向性无法客观测量。我们概述了将目标导向性建模为动态多代理系统 emergent 属性的新方向。 

---
# Uncertainty-Aware Learning Policy for Reliable Pulmonary Nodule Detection on Chest X-Ray 

**Title (ZH)**: 面向胸片中肺结节检测的不确定性aware学习策略 

**Authors**: Hyeonjin Choi, Jinse Kim, Dong-yeon Yoo, Ju-sung Sun, Jung-won Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.13236)  

**Abstract**: Early detection and rapid intervention of lung cancer are crucial. Nonetheless, ensuring an accurate diagnosis is challenging, as physicians' ability to interpret chest X-rays varies significantly depending on their experience and degree of fatigue. Although medical AI has been rapidly advancing to assist in diagnosis, physicians' trust in such systems remains limited, preventing widespread clinical adoption. This skepticism fundamentally stems from concerns about its diagnostic uncertainty. In clinical diagnosis, physicians utilize extensive background knowledge and clinical experience. In contrast, medical AI primarily relies on repetitive learning of the target lesion to generate diagnoses based solely on that data. In other words, medical AI does not possess sufficient knowledge to render a diagnosis, leading to diagnostic uncertainty. Thus, this study suggests an Uncertainty-Aware Learning Policy that can address the issue of knowledge deficiency by learning the physicians' background knowledge alongside the Chest X-ray lesion information. We used 2,517 lesion-free images and 656 nodule images, all obtained from Ajou University Hospital. The proposed model attained 92% (IoU 0.2 / FPPI 2) with a 10% enhancement in sensitivity compared to the baseline model while also decreasing entropy as a measure of uncertainty by 0.2. 

**Abstract (ZH)**: 早期检测与迅速干预肺癌至关重要。然而，确保准确诊断极具挑战性，因为医生解读胸部X光的能力因经验程度和疲劳程度而异。尽管医学AI已迅速发展以辅助诊断，但医生对其系统的信任程度有限，阻碍了其在临床中的广泛应用。这种怀疑从根本上源于对诊断不确定性的担忧。在临床诊断中，医生利用广泛的背景知识和临床经验。相比之下，医学AI主要依靠重复学习目标病灶来生成诊断，仅基于那组数据。换句话说，医学AI缺乏足够的知识进行诊断，导致诊断不确定性。因此，本研究提出一种awareness of uncertainty学习策略，该策略通过同时学习医生的背景知识和胸部X光病灶信息，以解决知识不足的问题。我们使用了2,517张无病灶图像和656张结节图像，所有数据均来自 Ajou大学医院。所提出的模型在IoU为0.2和FPPI为2的情况下达到了92%的检测率，与基线模型相比，灵敏度提高了10%，同时通过减少不确定性度量（熵）0.2来降低不确定性。 

---
# The Role of AI in Facilitating Interdisciplinary Collaboration: Evidence from AlphaFold 

**Title (ZH)**: AI在促进跨学科合作中的作用：来自AlphaFold的证据 

**Authors**: Naixuan Zhao, Chunli Wei, Xinyan Zhang, Jiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13234)  

**Abstract**: The acceleration of artificial intelligence (AI) in science is recognized and many scholars have begun to explore its role in interdisciplinary collaboration. However, the mechanisms and extent of this impact are still unclear. This study, using AlphaFold's impact on structural biologists, examines how AI technologies influence interdisciplinary collaborative patterns. By analyzing 1,247 AlphaFold-related papers and 7,700 authors from Scopus, we employ bibliometric analysis and causal inference to compare interdisciplinary collaboration between AlphaFold adopters and non-adopters. Contrary to the widespread belief that AI facilitates interdisciplinary collaboration, our findings show that AlphaFold increased structural biology-computer science collaborations by just 0.48%, with no measurable effect on other disciplines. Specifically, AI creates interdisciplinary collaboration demands with specific disciplines due to its technical characteristics, but this demand is weakened by technological democratization and other factors. These findings demonstrate that artificial intelligence (AI) alone has limited efficacy in bridging disciplinary divides or fostering meaningful interdisciplinary collaboration. 

**Abstract (ZH)**: 人工智能（AI）在科学中的加速应用及其对跨学科合作的影响：以AlphaFold为例的研究 

---
# Deep Graph Neural Point Process For Learning Temporal Interactive Networks 

**Title (ZH)**: 深度图神经点过程学习时序交互网络 

**Authors**: Su Chen, Xiaohua Qi, Xixun Lin, Yanmin Shang, Xiaolin Xu, Yangxi Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13219)  

**Abstract**: Learning temporal interaction networks(TIN) is previously regarded as a coarse-grained multi-sequence prediction problem, ignoring the network topology structure influence. This paper addresses this limitation and a Deep Graph Neural Point Process(DGNPP) model for TIN is proposed. DGNPP consists of two key modules: the Node Aggregation Layer and the Self Attentive Layer. The Node Aggregation Layer captures topological structures to generate static representation for users and items, while the Self Attentive Layer dynamically updates embeddings over time. By incorporating both dynamic and static embeddings into the event intensity function and optimizing the model via maximum likelihood estimation, DGNPP predicts events and occurrence time effectively. Experimental evaluations on three public datasets demonstrate that DGNPP achieves superior performance in event prediction and time prediction tasks with high efficiency, significantly outperforming baseline models and effectively mitigating the limitations of prior approaches. 

**Abstract (ZH)**: 学习时序交互网络（TIN） previously被视为粗粒度的多序列预测问题，忽略了网络拓扑结构的影响。本文解决了这一局限，并提出了一种深度图神经点过程（DGNPP）模型用于TIN。DGNPP由两个关键模块组成：节点聚合层和自我注意层。节点聚合层捕获拓扑结构以生成用户和项目的静态表示，而自我注意层则动态更新时间上的嵌入表示。通过将动态和静态嵌入整合到事件强度函数中，并通过最大似然估计优化模型，DGNPP能够有效预测事件及其发生时间。实验评估表明，DGNPP在事件预测和时间预测任务中表现出色且效率高，显著优于基准模型，并有效缓解了先前方法的局限性。 

---
# Research on Conversational Recommender System Considering Consumer Types 

**Title (ZH)**: 考虑消费者类型的对话推荐系统研究 

**Authors**: Yaying Luo, Hui Fang, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.13209)  

**Abstract**: Conversational Recommender Systems (CRS) provide personalized services through multi-turn interactions, yet most existing methods overlook users' heterogeneous decision-making styles and knowledge levels, which constrains both accuracy and efficiency. To address this gap, we propose CT-CRS (Consumer Type-Enhanced Conversational Recommender System), a framework that integrates consumer type modeling into dialogue recommendation. Based on consumer type theory, we define four user categories--dependent, efficient, cautious, and expert--derived from two dimensions: decision-making style (maximizers vs. satisficers) and knowledge level (high vs. low). CT-CRS employs interaction histories and fine-tunes the large language model to automatically infer user types in real time, avoiding reliance on static questionnaires. We incorporate user types into state representation and design a type-adaptive policy that dynamically adjusts recommendation granularity, diversity, and attribute query complexity. To further optimize the dialogue policy, we adopt Inverse Reinforcement Learning (IRL), enabling the agent to approximate expert-like strategies conditioned on consumer type. Experiments on LastFM, Amazon-Book, and Yelp show that CTCRS improves recommendation success rate and reduces interaction turns compared to strong baselines. Ablation studies confirm that both consumer type modeling and IRL contribute significantly to performance gains. These results demonstrate that CT-CRS offers a scalable and interpretable solution for enhancing CRS personalization through the integration of psychological modeling and advanced policy optimization. 

**Abstract (ZH)**: 面向消费者类型的对话推荐系统（CT-CRS）：结合心理建模的个性化优化 

---
# Utilizing the RAIN method and Graph SAGE Model to Identify Effective Drug Combinations for Gastric Neoplasm Treatment 

**Title (ZH)**: 利用RAIN方法和Graph SAGE模型识别胃神经内分泌肿瘤的有效药物组合 

**Authors**: S. Z. Pirasteh, Ali A. Kiaei, Mahnaz Bush, Sabra Moghadam, Raha Aghaei, Behnaz Sadeghigol  

**Link**: [PDF](https://arxiv.org/pdf/2508.13207)  

**Abstract**: Background: Gastric neoplasm, primarily adenocarcinoma, is an aggressive cancer with high mortality, often diagnosed late, leading to complications like metastasis. Effective drug combinations are vital to address disease heterogeneity, enhance efficacy, reduce resistance, and improve patient outcomes. Methods: The RAIN method integrated Graph SAGE to propose drug combinations, using a graph model with p-value-weighted edges connecting drugs, genes, and proteins. NLP and systematic literature review (PubMed, Scopus, etc.) validated proposed drugs, followed by network meta-analysis to assess efficacy, implemented in Python. Results: Oxaliplatin, fluorouracil, and trastuzumab were identified as effective, supported by 61 studies. Fluorouracil alone had a p-value of 0.0229, improving to 0.0099 with trastuzumab, and 0.0069 for the triple combination, indicating superior efficacy. Conclusion: The RAIN method, combining AI and network meta-analysis, effectively identifies optimal drug combinations for gastric neoplasm, offering a promising strategy to enhance treatment outcomes and guide health policy. 

**Abstract (ZH)**: 背景：胃恶性肿瘤主要是腺癌，是一种具有高死亡率的侵袭性癌症，常常在晚期诊断，导致转移等并发症。有效的药物组合对于应对疾病异质性、增强疗效、减少抗药性并改善患者预后至关重要。方法：RAIN方法结合Graph SAGE提出药物组合，使用一个连接药物、基因和蛋白质的图模型，并通过p值加权的边进行连接。通过自然语言处理和系统文献回顾（PubMed、Scopus等）验证提出的药物，随后通过网络meta分析评估疗效，全部在Python中实施。结果：奥沙利铂、氟尿嘧啶和曲妥珠单抗被识别为有效的药物组合，有61项研究支持。单独使用氟尿嘧啶的p值为0.0229，加入曲妥珠单抗后降至0.0099，而三联组合的p值为0.0069，表明其有效性更优。结论：RAIN方法结合AI和网络meta分析，有效地识别出胃恶性肿瘤的最佳药物组合，为提高治疗效果和指导卫生政策提供了有前景的策略。 

---
# The Rise of Generative AI for Metal-Organic Framework Design and Synthesis 

**Title (ZH)**: 金属有机框架设计与合成中生成式AI的崛起 

**Authors**: Chenru Duan, Aditya Nandy, Shyam Chand Pal, Xin Yang, Wenhao Gao, Yuanqi Du, Hendrik Kraß, Yeonghun Kang, Varinia Bernales, Zuyang Ye, Tristan Pyle, Ray Yang, Zeqi Gu, Philippe Schwaller, Shengqian Ma, Shijing Sun, Alán Aspuru-Guzik, Seyed Mohamad Moosavi, Robert Wexler, Zhiling Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.13197)  

**Abstract**: Advances in generative artificial intelligence are transforming how metal-organic frameworks (MOFs) are designed and discovered. This Perspective introduces the shift from laborious enumeration of MOF candidates to generative approaches that can autonomously propose and synthesize in the laboratory new porous reticular structures on demand. We outline the progress of employing deep learning models, such as variational autoencoders, diffusion models, and large language model-based agents, that are fueled by the growing amount of available data from the MOF community and suggest novel crystalline materials designs. These generative tools can be combined with high-throughput computational screening and even automated experiments to form accelerated, closed-loop discovery pipelines. The result is a new paradigm for reticular chemistry in which AI algorithms more efficiently direct the search for high-performance MOF materials for clean air and energy applications. Finally, we highlight remaining challenges such as synthetic feasibility, dataset diversity, and the need for further integration of domain knowledge. 

**Abstract (ZH)**: 生成式人工智能的进步正在变革金属有机框架（MOFs）的设计与发现方式。本文概览了从耗时的MOF候选物枚举方法向能够自主提出并在实验室合成新多孔骨架结构的方法的转变。我们概述了利用变分自编码器、扩散模型和基于大型语言模型的代理等深度学习模型的应用进展，这些模型得益于越来越多的来自MOF社区的数据，并提出新型晶体材料设计。这些生成工具可以与高通量计算筛选和自动化实验相结合，形成加速的闭环发现流程。结果，这为晶态化学提供了一个新的范式，在此范式中，AI算法更有效地指导高性能MOF材料在清洁空气和能源应用中的搜索。最后，我们指出了剩余的挑战，如合成可行性、数据集多样性以及需要进一步整合专业知识。 

---
# Contextual Attention-Based Multimodal Fusion of LLM and CNN for Sentiment Analysis 

**Title (ZH)**: 基于上下文注意力的LLM和CNN多模态融合情感分析 

**Authors**: Meriem Zerkouk, Miloud Mihoubi, Belkacem Chikhaoui  

**Link**: [PDF](https://arxiv.org/pdf/2508.13196)  

**Abstract**: This paper introduces a novel approach for multimodal sentiment analysis on social media, particularly in the context of natural disasters, where understanding public sentiment is crucial for effective crisis management. Unlike conventional methods that process text and image modalities separately, our approach seamlessly integrates Convolutional Neural Network (CNN) based image analysis with Large Language Model (LLM) based text processing, leveraging Generative Pre-trained Transformer (GPT) and prompt engineering to extract sentiment relevant features from the CrisisMMD dataset. To effectively model intermodal relationships, we introduce a contextual attention mechanism within the fusion process. Leveraging contextual-attention layers, this mechanism effectively captures intermodality interactions, enhancing the model's comprehension of complex relationships between textual and visual data. The deep neural network architecture of our model learns from these fused features, leading to improved accuracy compared to existing baselines. Experimental results demonstrate significant advancements in classifying social media data into informative and noninformative categories across various natural disasters. Our model achieves a notable 2.43% increase in accuracy and 5.18% in F1-score, highlighting its efficacy in processing complex multimodal data. Beyond quantitative metrics, our approach provides deeper insight into the sentiments expressed during crises. The practical implications extend to real time disaster management, where enhanced sentiment analysis can optimize the accuracy of emergency interventions. By bridging the gap between multimodal analysis, LLM powered text understanding, and disaster response, our work presents a promising direction for Artificial Intelligence (AI) driven crisis management solutions. Keywords: 

**Abstract (ZH)**: 一种针对自然灾害情境下的社交媒体多模态情感分析的新型方法：基于上下文注意力机制的图像分析与语言模型文本处理融合 

---
# Preference Models assume Proportional Hazards of Utilities 

**Title (ZH)**: 偏好模型假设效用的比例危害。 

**Authors**: Chirag Nagpal  

**Link**: [PDF](https://arxiv.org/pdf/2508.13189)  

**Abstract**: Approaches for estimating preferences from human annotated data typically involves inducing a distribution over a ranked list of choices such as the Plackett-Luce model. Indeed, modern AI alignment tools such as Reward Modelling and Direct Preference Optimization are based on the statistical assumptions posed by the Plackett-Luce model. In this paper, I will connect the Plackett-Luce model to another classical and well known statistical model, the Cox Proportional Hazards model and attempt to shed some light on the implications of the connection therein. 

**Abstract (ZH)**: 基于人类标注数据估计偏好方法通常涉及诱导一个排序选择列表上的分布，如Plackett-Luce模型。事实上，现代AI对齐工具，如奖励建模和直接偏好优化，正是基于Plackett-Luce模型的统计假设。在本文中，我将连接Plackett-Luce模型与另一个经典且广为人知的统计模型——Cox比例风险模型，并尝试探讨其中连接的含义。 

---
# Toward an African Agenda for AI Safety 

**Title (ZH)**: 面向非洲的AI安全议程 

**Authors**: Samuel T. Segun, Rachel Adams, Ana Florido, Scott Timcke, Jonathan Shock, Leah Junck, Fola Adeleke, Nicolas Grossman, Ayantola Alayande, Jerry John Kponyo, Matthew Smith, Dickson Marfo Fosu, Prince Dawson Tetteh, Juliet Arthur, Stephanie Kasaon, Odilile Ayodele, Laetitia Badolo, Paul Plantinga, Michael Gastrow, Sumaya Nur Adan, Joanna Wiaterek, Cecil Abungu, Kojo Apeagyei, Luise Eder, Tegawende Bissyande  

**Link**: [PDF](https://arxiv.org/pdf/2508.13179)  

**Abstract**: This paper maps Africa's distinctive AI risk profile, from deepfake fuelled electoral interference and data colonial dependency to compute scarcity, labour disruption and disproportionate exposure to climate driven environmental costs. While major benefits are promised to accrue, the availability, development and adoption of AI also mean that African people and countries face particular AI safety risks, from large scale labour market disruptions to the nefarious use of AI to manipulate public opinion. To date, African perspectives have not been meaningfully integrated into global debates and processes regarding AI safety, leaving African stakeholders with limited influence over the emerging global AI safety governance agenda. While there are Computer Incident Response Teams on the continent, none hosts a dedicated AI Safety Institute or office. We propose a five-point action plan centred on (i) a policy approach that foregrounds the protection of the human rights of those most vulnerable to experiencing the harmful socio-economic effects of AI; (ii) the establishment of an African AI Safety Institute; (iii) promote public AI literacy and awareness; (iv) development of early warning system with inclusive benchmark suites for 25+ African languages; and (v) an annual AU-level AI Safety & Security Forum. 

**Abstract (ZH)**: 这篇论文映射了非洲独特的AI风险画像，从深度造假选举干预和数据殖民依赖，到计算资源稀缺、劳动力市场扰乱以及对由气候驱动的环境成本的不成比例暴露。虽然AI带来的好处受到期待，但AI的可用性、开发和采用也意味着非洲人民和国家面临特定的AI安全风险，从大规模劳动力市场扰乱到利用AI manipulate公共意见的恶意行为。迄今为止，非洲视角尚未被有意义地纳入关于AI安全的全球辩论和进程中，导致非洲利益相关者在正在形成的全球AI安全治理议程中影响力有限。虽然非洲大陆上有计算机应急响应团队，但没有专门的AI安全研究所或办公室。我们提出一个五点行动计划，重点在于（i）一种以保护最易遭受AI有害社会经济影响的人类权利为中心的政策方法；（ii）建立非洲AI安全研究所；（iii）促进公众AI素养和意识；（iv）开发适用于25种以上非洲语言的早期预警系统和包容性基准测试套件；以及（v）每年举办一次非洲联盟层面的AI安全与安全论坛。 

---
# Sustainable AI Training via Hardware-Software Co-Design on NVIDIA, AMD, and Emerging GPU Architectures 

**Title (ZH)**: 基于 NVIDIA、AMD 及新兴 GPU 架构的硬件-软件协同设计可持续 AI 训练 

**Authors**: Yashasvi Makin, Rahul Maliakkal  

**Link**: [PDF](https://arxiv.org/pdf/2508.13163)  

**Abstract**: In particular, large-scale deep learning and artificial intelligence model training uses a lot of computational power and energy, so it poses serious sustainability issues. The fast rise in model complexity has resulted in exponential increases in energy consumption, increasing the demand for techniques maximizing computational efficiency and lowering environmental impact. This work explores environmentally driven performance optimization methods especially intended for advanced GPU architectures from NVIDIA, AMD, and other emerging GPU architectures. Our main focus is on investigating hardware-software co-design techniques meant to significantly increase memory-level and kernel-level operations, so improving performance-per-watt measures. Our thorough research encompasses evaluations of specialized tensor and matrix cores, advanced memory optimization methods, and creative integration approaches that taken together result in notable energy efficiency increases. We also discuss important software-level optimizations that augment hardware capability including mixed-precision arithmetic, advanced energy-aware scheduling algorithms, and compiler-driven kernel enhancements. Moreover, we methodically point out important research gaps and suggest future directions necessary to create really sustainable artificial intelligence systems. This paper emphasizes how major increases in training efficiency can be obtained by co-design of hardware and software, so lowering the environmental impact of artificial intelligence without compromising performance. To back up our analysis, we use real-world case studies from top companies like Meta, Google, Amazon, and others that show how these sustainable AI training methods are used in the real world. 

**Abstract (ZH)**: 大规模深度学习和人工智能模型训练消耗大量计算资源和能源，导致严重的可持续性问题。模型复杂度的快速提高导致能源消耗呈指数级增长，增加了提高计算效率和降低环境影响的技术需求。本研究探索了特别针对NVIDIA、AMD及其他新兴GPU架构的环境驱动型性能优化方法，重点关注硬件-软件协同设计技术以显著提高内存级和内核级操作，从而提升单位瓦特性能。我们的深入研究涵盖了专用张量和矩阵核的评估、高级内存优化方法以及创新的集成方法，这些方法结合起来能够显著提高能效。我们还讨论了在硬件能力基础上的软件级优化技术，包括混合精度算术、高级能效调度算法和编译器驱动的内核增强。此外，我们系统地指出现有研究中的重要空白，并建议未来发展方向，以创建真正可持续的人工智能系统。本文强调了通过硬件和软件协同设计，可以在不牺牲性能的情况下降低人工智能的环境影响，从而获得大幅提高训练效率的方式。为了支持我们的分析，我们使用来自Meta、Google、Amazon等顶级公司的实际案例研究，展示了这些可持续的AI训练方法在实际中的应用。 

---
# Piano: A Multi-Constraint Pin Assignment-Aware Floorplanner 

**Title (ZH)**: 钢琴：一种多约束针脚分配感知的布局规划器 

**Authors**: Zhexuan Xu, Kexin Zhou, Jie Wang, Zijie Geng, Siyuan Xu, Shixiong Kai, Mingxuan Yuan, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13161)  

**Abstract**: Floorplanning is a critical step in VLSI physical design, increasingly complicated by modern constraints such as fixed-outline requirements, whitespace removal, and the presence of pre-placed modules. In addition, the assignment of pins on module boundaries significantly impacts the performance of subsequent stages, including detailed placement and routing. However, traditional floorplanners often overlook pin assignment with modern constraints during the floorplanning stage. In this work, we introduce Piano, a floorplanning framework that simultaneously optimizes module placement and pin assignment under multiple constraints. Specifically, we construct a graph based on the geometric relationships among modules and their netlist connections, then iteratively search for shortest paths to determine pin assignments. This graph-based method also enables accurate evaluation of feedthrough and unplaced pins, thereby guiding overall layout quality. To further improve the design, we adopt a whitespace removal strategy and employ three local optimizers to enhance layout metrics under multi-constraint scenarios. Experimental results on widely used benchmark circuits demonstrate that Piano achieves an average 6.81% reduction in HPWL, a 13.39% decrease in feedthrough wirelength, a 16.36% reduction in the number of feedthrough modules, and a 21.21% drop in unplaced pins, while maintaining zero whitespace. 

**Abstract (ZH)**: 基于多约束条件下的模块放置与引脚分配优化框架Piano 

---
# Image2Net: Datasets, Benchmark and Hybrid Framework to Convert Analog Circuit Diagrams into Netlists 

**Title (ZH)**: Image2Net: 数据集、基准和混合框架，用于将模拟电路图转换为网表 

**Authors**: Haohang Xu, Chengjie Liu, Qihang Wang, Wenhao Huang, Yongjian Xu, Weiyu Chen, Anlan Peng, Zhijun Li, Bo Li, Lei Qi, Jun Yang, Yuan Du, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.13157)  

**Abstract**: Large Language Model (LLM) exhibits great potential in designing of analog integrated circuits (IC) because of its excellence in abstraction and generalization for knowledge. However, further development of LLM-based analog ICs heavily relies on textual description of analog ICs, while existing analog ICs are mostly illustrated in image-based circuit diagrams rather than text-based netlists. Converting circuit diagrams to netlists help LLMs to enrich the knowledge of analog IC. Nevertheless, previously proposed conversion frameworks face challenges in further application because of limited support of image styles and circuit elements. Up to now, it still remains a challenging task to effectively convert complex circuit diagrams into netlists. To this end, this paper constructs and opensources a new dataset with rich styles of circuit diagrams as well as balanced distribution of simple and complex analog ICs. And a hybrid framework, named Image2Net, is proposed for practical conversion from circuit diagrams to netlists. The netlist edit distance (NED) is also introduced to precisely assess the difference between the converted netlists and ground truth. Based on our benchmark, Image2Net achieves 80.77\% successful rate, which is 34.62\%-45.19\% higher than previous works. Specifically, the proposed work shows 0.116 averaged NED, which is 62.1\%-69.6\% lower than state-of-the-arts. 

**Abstract (ZH)**: 基于图像到网表转换的新数据集及Image2Net混合框架：复杂电路图到网表的有效转换 

---
# Preliminary suggestions for rigorous GPAI model evaluations 

**Title (ZH)**: 初步建议：严格的GPAI模型评估 

**Authors**: Patricia Paskov, Michael J. Byun, Kevin Wei, Toby Webster  

**Link**: [PDF](https://arxiv.org/pdf/2508.00875)  

**Abstract**: This document presents a preliminary compilation of general-purpose AI (GPAI) evaluation practices that may promote internal validity, external validity and reproducibility. It includes suggestions for human uplift studies and benchmark evaluations, as well as cross-cutting suggestions that may apply to many different evaluation types. Suggestions are organised across four stages in the evaluation life cycle: design, implementation, execution and documentation. Drawing from established practices in machine learning, statistics, psychology, economics, biology and other fields recognised to have important lessons for AI evaluation, these suggestions seek to contribute to the conversation on the nascent and evolving field of the science of GPAI evaluations. The intended audience of this document includes providers of GPAI models presenting systemic risk (GPAISR), for whom the EU AI Act lays out specific evaluation requirements; third-party evaluators; policymakers assessing the rigour of evaluations; and academic researchers developing or conducting GPAI evaluations. 

**Abstract (ZH)**: 本文档提出了促进一般用途人工智能（GPAI）内部有效性、外部有效性和可再现性的初步综合评价实践。它包括人类提升研究和基准评估的建议，以及可应用于多种评价类型的跨学科建议。这些建议按照评价生命周期的四个阶段（设计、实施、执行和文档）进行组织。本文档借鉴了机器学习、统计学、心理学、经济学、生物学及其他领域公认具有重要评价教训的实践，旨在为新兴且不断发展中的GPAI评价科学领域的讨论做出贡献。本文档的预期读者包括提供可能产生系统性风险的一般用途人工智能（GPAI）模型的供应商（欧盟人工智能法案为此类供应商列出了具体评价要求）、第三方评价者、评估评价严谨性的政策制定者以及进行或开发GPAI评价的学术研究人员。 

---
