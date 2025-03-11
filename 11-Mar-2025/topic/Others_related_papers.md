# Geometric Retargeting: A Principled, Ultrafast Neural Hand Retargeting Algorithm 

**Title (ZH)**: 几何重塑：一个原则性的超快速神经手部重塑算法 

**Authors**: Zhao-Heng Yin, Changhao Wang, Luis Pineda, Krishna Bodduluri, Tingfan Wu, Pieter Abbeel, Mustafa Mukadam  

**Link**: [PDF](https://arxiv.org/pdf/2503.07541)  

**Abstract**: We introduce Geometric Retargeting (GeoRT), an ultrafast, and principled neural hand retargeting algorithm for teleoperation, developed as part of our recent Dexterity Gen (DexGen) system. GeoRT converts human finger keypoints to robot hand keypoints at 1KHz, achieving state-of-the-art speed and accuracy with significantly fewer hyperparameters. This high-speed capability enables flexible postprocessing, such as leveraging a foundational controller for action correction like DexGen. GeoRT is trained in an unsupervised manner, eliminating the need for manual annotation of hand pairs. The core of GeoRT lies in novel geometric objective functions that capture the essence of retargeting: preserving motion fidelity, ensuring configuration space (C-space) coverage, maintaining uniform response through high flatness, pinch correspondence and preventing self-collisions. This approach is free from intensive test-time optimization, offering a more scalable and practical solution for real-time hand retargeting. 

**Abstract (ZH)**: 几何重塑（GeoRT）：一种用于遥操作的超快速和原理性的神经手部重塑算法 

---
# QBIT: Quality-Aware Cloud-Based Benchmarking for Robotic Insertion Tasks 

**Title (ZH)**: QBIT: 基于质量aware的云平台机器人插入任务基准测试 

**Authors**: Constantin Schempp, Yongzhou Zhang, Christian Friedrich, Bjorn Hein  

**Link**: [PDF](https://arxiv.org/pdf/2503.07479)  

**Abstract**: Insertion tasks are fundamental yet challenging for robots, particularly in autonomous operations, due to their continuous interaction with the environment. AI-based approaches appear to be up to the challenge, but in production they must not only achieve high success rates. They must also ensure insertion quality and reliability. To address this, we introduce QBIT, a quality-aware benchmarking framework that incorporates additional metrics such as force energy, force smoothness and completion time to provide a comprehensive assessment. To ensure statistical significance and minimize the sim-to-real gap, we randomize contact parameters in the MuJoCo simulator, account for perceptual uncertainty, and conduct large-scale experiments on a Kubernetes-based infrastructure. Our microservice-oriented architecture ensures extensibility, broad applicability, and improved reproducibility. To facilitate seamless transitions to physical robotic testing, we use ROS2 with containerization to reduce integration barriers. We evaluate QBIT using three insertion approaches: geometricbased, force-based, and learning-based, in both simulated and real-world environments. In simulation, we compare the accuracy of contact simulation using different mesh decomposition techniques. Our results demonstrate the effectiveness of QBIT in comparing different insertion approaches and accelerating the transition from laboratory to real-world applications. Code is available on GitHub. 

**Abstract (ZH)**: 基于质量感知的插入任务基准框架QBIT：针对自主操作的评估与优化 

---
# CATPlan: Loss-based Collision Prediction in End-to-End Autonomous Driving 

**Title (ZH)**: CATPlan: 基于损失的端到端自主驾驶碰撞预测 

**Authors**: Ziliang Xiong, Shipeng Liu, Nathaniel Helgesen, Joakim Johnander, Per-Erik Forssen  

**Link**: [PDF](https://arxiv.org/pdf/2503.07425)  

**Abstract**: In recent years, there has been increased interest in the design, training, and evaluation of end-to-end autonomous driving (AD) systems. One often overlooked aspect is the uncertainty of planned trajectories predicted by these systems, despite awareness of their own uncertainty being key to achieve safety and robustness. We propose to estimate this uncertainty by adapting loss prediction from the uncertainty quantification literature. To this end, we introduce a novel light-weight module, dubbed CATPlan, that is trained to decode motion and planning embeddings into estimates of the collision loss used to partially supervise end-to-end AD systems. During inference, these estimates are interpreted as collision risk. We evaluate CATPlan on the safety-critical, nerf-based, closed-loop benchmark NeuroNCAP and find that it manages to detect collisions with a $54.8\%$ relative improvement to average precision over a GMM-based baseline in which the predicted trajectory is compared to the forecasted trajectories of other road users. Our findings indicate that the addition of CATPlan can lead to safer end-to-end AD systems and hope that our work will spark increased interest in uncertainty quantification for such systems. 

**Abstract (ZH)**: 近年来，人们对端到端自动驾驶（AD）系统的设计、训练和评估表现出了浓厚的兴趣。一个经常被忽视的问题是，尽管意识到自身的不确定性对于实现安全和鲁棒性至关重要，但这些系统预测的轨迹不确定性往往被忽视。为此，我们提议通过适应不确定性量化领域的损失预测方法来估计这种不确定性。为此，我们引入了一个新颖的轻量级模块，称为CATPlan，该模块被训练为将运动和规划嵌入解码为用于部分监督端到端AD系统的碰撞损失估计。在推理过程中，这些估计值被解释为碰撞风险。我们使用安全至关重要的、基于nerf的闭环基准NeuroNCAP对CATPlan进行了评估，并发现它在平均精确率上相比基于GMM的基线方法，能够以54.8%的相对改进检测到碰撞。我们的研究结果表明，CATPlan的添加可以使端到端AD系统更加安全，希望我们的工作能够引发对这类系统中不确定性量化更广泛的兴趣。 

---
# PER-DPP Sampling Framework and Its Application in Path Planning 

**Title (ZH)**: PER-DPP采样框架及其在路径规划中的应用 

**Authors**: Junzhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07411)  

**Abstract**: Autonomous navigation in intelligent mobile systems represents a core research focus within artificial intelligence-driven robotics. Contemporary path planning approaches face constraints in dynamic environmental responsiveness and multi-objective task scalability, limiting their capacity to address growing intelligent operation requirements. Decision-centric reinforcement learning frameworks, capitalizing on their unique strengths in adaptive environmental interaction and self-optimization, have gained prominence in advanced control system research. This investigation introduces methodological improvements to address sample homogeneity challenges in reinforcement learning experience replay mechanisms. By incorporating determinant point processes (DPP) for diversity assessment, we develop a dual-criteria sampling framework with adaptive selection protocols. This approach resolves representation bias in conventional prioritized experience replay (PER) systems while preserving algorithmic interoperability, offering improved decision optimization for dynamic operational scenarios. Key contributions comprise: Develop a hybrid sampling paradigm (PER-DPP) combining priority sequencing with diversity this http URL on this,create an integrated optimization scheme (PER-DPP-Elastic DQN) merging diversity-aware sampling with adaptive step-size regulation. Comparative simulations in 2D navigation scenarios demonstrate that the elastic step-size component temporarily delays initial convergence speed but synergistically enhances final-stage optimization with PER-DPP integration. The synthesized method generates navigation paths with optimized length efficiency and directional stability. 

**Abstract (ZH)**: 基于自主导航的智能移动系统中的强化学习方法改进研究 

---
# Learning Nash Equilibrial Hamiltonian for Two-Player Collision-Avoiding Interactions 

**Title (ZH)**: 学习两玩家避碰交互的纳什均衡哈密尔顿量 

**Authors**: Lei Zhang, Siddharth Das, Tanner Merry, Wenlong Zhang, Yi Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.07013)  

**Abstract**: We consider the problem of learning Nash equilibrial policies for two-player risk-sensitive collision-avoiding interactions. Solving the Hamilton-Jacobi-Isaacs equations of such general-sum differential games in real time is an open challenge due to the discontinuity of equilibrium values on the state space. A common solution is to learn a neural network that approximates the equilibrium Hamiltonian for given system states and actions. The learning, however, is usually supervised and requires a large amount of sample equilibrium policies from different initial states in order to mitigate the risks of collisions. This paper claims two contributions towards more data-efficient learning of equilibrium policies: First, instead of computing Hamiltonian through a value network, we show that the equilibrium co-states have simple structures when collision avoidance dominates the agents' loss functions and system dynamics is linear, and therefore are more data-efficient to learn. Second, we introduce theory-driven active learning to guide data sampling, where the acquisition function measures the compliance of the predicted co-states to Pontryagin's Maximum Principle. On an uncontrolled intersection case, the proposed method leads to more generalizable approximation of the equilibrium policies, and in turn, lower collision probabilities, than the state-of-the-art under the same data acquisition budget. 

**Abstract (ZH)**: 我们考虑两-player风险敏感碰撞避免交互中学习纳什均衡策略的问题。由于均衡值在状态空间上的非连续性，实时求解此类常和微分博弈的哈密尔顿-雅可比-伊斯阿斯方程仍然是一个开放挑战。一种常见的解决方案是学习一个神经网络来近似给定系统状态和动作的均衡哈密尔顿量。然而，这种学习通常是监督式的，需要大量不同初始状态下的样本均衡策略以减轻碰撞风险。本文提出了两种更高效学习均衡策略的贡献：首先，当避免碰撞是代理损失函数和系统动力学的主要因素时，我们证明均衡共态具有简单的结构，因此更具数据效率。其次，我们引入理论驱动的主动学习来指导数据采样，其中获取函数衡量预测共态与庞特里亚金最大原则的一致性。在无控交叉口情况下，所提出的方法在相同的数据采集预算下，其均衡策略的泛化性更强，从而降低了碰撞概率，优于现有方法。 

---
# Parametric Value Approximation for General-sum Differential Games with State Constraints 

**Title (ZH)**: 带有状态约束的一般和微分博弈的参数值近似估计 

**Authors**: Lei Zhang, Mukesh Ghimire, Wenlong Zhang, Zhe Xu, Yi Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.06994)  

**Abstract**: General-sum differential games can approximate values solved by Hamilton-Jacobi-Isaacs (HJI) equations for efficient inference when information is incomplete. However, solving such games through conventional methods encounters the curse of dimensionality (CoD). Physics-informed neural networks (PINNs) offer a scalable approach to alleviate the CoD and approximate values, but there exist convergence issues for value approximations through vanilla PINNs when state constraints lead to values with large Lipschitz constants, particularly in safety-critical applications. In addition to addressing CoD, it is necessary to learn a generalizable value across a parametric space of games, rather than training multiple ones for each specific player-type configuration. To overcome these challenges, we propose a Hybrid Neural Operator (HNO), which is an operator that can map parameter functions for games to value functions. HNO leverages informative supervised data and samples PDE-driven data across entire spatial-temporal space for model refinement. We evaluate HNO on 9D and 13D scenarios with nonlinear dynamics and state constraints, comparing it against a Supervised Neural Operator (a variant of DeepONet). Under the same computational budget and training data, HNO outperforms SNO for safety performance. This work provides a step toward scalable and generalizable value function approximation, enabling real-time inference for complex human-robot or multi-agent interactions. 

**Abstract (ZH)**: 一般和差博弈可以通过Hamilton-Jacobi-Isaacs (HJI) 方程近似求解值，从而在信息不完备时实现高效推理。然而，通过传统方法解决此类博弈会遭遇维数灾。物理信息神经网络（PINNs）提供了一种可扩展的方法来缓解维数灾并近似求解值，但对于受状态约束导致Lipschitz常数较大的值，vanilla PINNs存在收敛问题，特别是在关键安全应用中。除了解决维数灾之外，还需要在一个参数化的博弈空间中学习泛化的值，而不是为每个特定玩家配置训练多个模型。为克服这些挑战，我们提出了一种混合神经算子（HNO），它可以将博弈的参数函数映射到值函数。HNO利用有信息的监督数据并采样来自整个时空域的PDE驱动数据以进行模型校准。我们在具有非线性动力学和状态约束的9D和13D场景中评估了HNO，并将其与监督神经算子（SNO，DeepONet的变体）进行了对比。在相同的计算预算和训练数据下，HNO在安全性性能上优于SNO。这项工作为可扩展和可泛化的值函数近似提供了一个方向，从而实现在复杂的人机交互或多智能体交互中的实时推理。 

---
# RoboDesign1M: A Large-scale Dataset for Robot Design Understanding 

**Title (ZH)**: RoboDesign1M: 机器人设计理解的大规模数据集 

**Authors**: Tri Le, Toan Nguyen, Quang Tran, Quang Nguyen, Baoru Huang, Hoan Nguyen, Minh Nhat Vu, Tung D. Ta, Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.06796)  

**Abstract**: Robot design is a complex and time-consuming process that requires specialized expertise. Gaining a deeper understanding of robot design data can enable various applications, including automated design generation, retrieving example designs from text, and developing AI-powered design assistants. While recent advancements in foundation models present promising approaches to addressing these challenges, progress in this field is hindered by the lack of large-scale design datasets. In this paper, we introduce RoboDesign1M, a large-scale dataset comprising 1 million samples. Our dataset features multimodal data collected from scientific literature, covering various robotics domains. We propose a semi-automated data collection pipeline, enabling efficient and diverse data acquisition. To assess the effectiveness of RoboDesign1M, we conduct extensive experiments across multiple tasks, including design image generation, visual question answering about designs, and design image retrieval. The results demonstrate that our dataset serves as a challenging new benchmark for design understanding tasks and has the potential to advance research in this field. RoboDesign1M will be released to support further developments in AI-driven robotic design automation. 

**Abstract (ZH)**: 机器人设计是一个复杂且耗时的过程，需要专门的技术知识。对机器人设计数据的深入理解能够促进自动化设计生成、从文本检索样例设计以及开发AI辅助设计助手等应用。尽管基础模型的最新进展为解决这些挑战提供了有前景的方法，但该领域的发展受到大型设计数据集缺乏的阻碍。本文介绍了RoboDesign1M，这是一个包含100万样本的大型数据集，涵盖了多模态数据，来自科学文献，涉及多种机器人领域。我们提出了一种半自动数据收集管道，以实现高效且多样化的数据获取。为了评估RoboDesign1M的有效性，我们在多个任务上进行了广泛实验，包括设计图像生成、关于设计的视觉问答以及设计图像检索。实验结果表明，我们的数据集是一个具有挑战性的新基准，能够促进该领域的研究，并具有推动研究发展的潜力。RoboDesign1M将被发布以支持基于AI的机器人设计自动化的进一步发展。 

---
# Chance-Constrained Trajectory Planning with Multimodal Environmental Uncertainty 

**Title (ZH)**: 多模态环境不确定性下的机会约束轨迹规划 

**Authors**: Kai Ren, Heejin Ahn, Maryam Kamgarpour  

**Link**: [PDF](https://arxiv.org/pdf/2503.06779)  

**Abstract**: We tackle safe trajectory planning under Gaussian mixture model (GMM) uncertainty. Specifically, we use a GMM to model the multimodal behaviors of obstacles' uncertain states. Then, we develop a mixed-integer conic approximation to the chance-constrained trajectory planning problem with deterministic linear systems and polyhedral obstacles. When the GMM moments are estimated via finite samples, we develop a tight concentration bound to ensure the chance constraint with a desired confidence. Moreover, to limit the amount of constraint violation, we develop a Conditional Value-at-Risk (CVaR) approach corresponding to the chance constraints and derive a tractable approximation for known and estimated GMM moments. We verify our methods with state-of-the-art trajectory prediction algorithms and autonomous driving datasets. 

**Abstract (ZH)**: 基于高斯混合模型不确定性下的安全轨迹规划 

---
# Chance-constrained Linear Quadratic Gaussian Games for Multi-robot Interaction under Uncertainty 

**Title (ZH)**: 线性二次高斯游戏的不确定性下多机器人交互机会约束模型 

**Authors**: Kai Ren, Giulio Salizzoni, Mustafa Emre Gürsoy, Maryam Kamgarpour  

**Link**: [PDF](https://arxiv.org/pdf/2503.06776)  

**Abstract**: We address safe multi-robot interaction under uncertainty. In particular, we formulate a chance-constrained linear quadratic Gaussian game with coupling constraints and system uncertainties. We find a tractable reformulation of the game and propose a dual ascent algorithm. We prove that the algorithm converges to a generalized Nash equilibrium of the reformulated game, ensuring the satisfaction of the chance constraints. We test our method in driving simulations and real-world robot experiments. Our method ensures safety under uncertainty and generates less conservative trajectories than single-agent model predictive control. 

**Abstract (ZH)**: 我们研究了不确定条件下多机器人安全交互问题。特别地，我们提出了一个带有耦合约束和系统不确定性的一般化纳什均衡的机会约束线性二次高斯博弈。我们找到了博弈的可处理重写形式，并提出了一种对偶上升算法。我们证明该算法能够收敛到重写博弈的广义纳什均衡，确保机会约束的满足。我们该方法在驾驶模拟和实际机器人实验中进行了测试。该方法在不确定性条件下确保安全，并生成比单智能体模型预测控制更为保守的轨迹。 

---
# pRRTC: GPU-Parallel RRT-Connect for Fast, Consistent, and Low-Cost Motion Planning 

**Title (ZH)**: PRTTC：GPU并行RRT-Connect算法实现快速、一致且低成本的运动规划 

**Authors**: Chih H. Huang, Pranav Jadhav, Brian Plancher, Zachary Kingston  

**Link**: [PDF](https://arxiv.org/pdf/2503.06757)  

**Abstract**: Sampling-based motion planning algorithms, like the Rapidly-Exploring Random Tree (RRT) and its widely used variant, RRT-Connect, provide efficient solutions for high-dimensional planning problems faced by real-world robots. However, these methods remain computationally intensive, particularly in complex environments that require many collision checks. As such, to improve performance, recent efforts have explored parallelizing specific components of RRT, such as collision checking or running multiple planners independently, but no prior work has integrated parallelism at multiple levels of the algorithm for robotic manipulation. In this work, we present pRRTC, a GPU-accelerated implementation of RRT-Connect that achieves parallelism across the entire algorithm through multithreaded expansion and connection, SIMT-optimized collision checking, and hierarchical parallelism optimization, improving efficiency, consistency, and initial solution cost. We evaluate the effectiveness of pRRTC on the MotionBenchMaker dataset using robots with 7, 8, and 14 degrees-of-freedom, demonstrating up to 6x average speedup on constrained reaching tasks at high collision checking resolution compared to state-of-the-art. pRRTC also demonstrates a 5x reduction in solution time variance and 1.5x improvement in initial path costs compared to state-of-the-art motion planners in complex environments across all robots. 

**Abstract (ZH)**: 基于采样的运动规划算法，如快速扩展随机树（RRT）及其广泛应用的变体RRT-Connect，为真实世界机器人面临的高维规划问题提供了高效的解决方案。然而，这些方法在复杂的环境需求下仍计算密集，特别是需要进行大量碰撞检测的情况。因此，为了提高性能，近期研究探索了并行化RRT的具体组件，如碰撞检测或独立运行多个规划器，但此前没有任何研究在算法的多个层次集成并行性以提高机器人的操作效率。在这项工作中，我们提出了pRRTC，这是一种基于GPU加速的RRT-Connect实现，通过多线程扩展和连接、SIMT优化的碰撞检测以及分层并行性优化，在整个算法中实现并行化，提高效率、一致性和初始解成本。我们使用具有7、8和14自由度的MotionBenchMaker数据集评估pRRTC的有效性，与最先进的方法相比，在高碰撞检测分辨率下执行受限拾取任务时平均提速6倍。pRRTC还在所有机器人中展示了5倍的解时间方差减少和1.5倍的初始路径成本改进，这些改进在复杂环境中表现尤为显著。 

---
# InfoFusion Controller: Informed TRRT Star with Mutual Information based on Fusion of Pure Pursuit and MPC for Enhanced Path Planning 

**Title (ZH)**: 基于纯追迹与模型预测控制融合的互信息引导TRRT星型信息融合控制器：增强路径规划 

**Authors**: Seongjun Choi, Youngbum Kim, Nam Woo Kim, Mansun Shin, Byunggi Chae, Sungjin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.06010)  

**Abstract**: In this paper, we propose the InfoFusion Controller, an advanced path planning algorithm that integrates both global and local planning strategies to enhance autonomous driving in complex urban environments. The global planner utilizes the informed Theta-Rapidly-exploring Random Tree Star (Informed-TRRT*) algorithm to generate an optimal reference path, while the local planner combines Model Predictive Control (MPC) and Pure Pursuit algorithms. Mutual Information (MI) is employed to fuse the outputs of the MPC and Pure Pursuit controllers, effectively balancing their strengths and compensating for their weaknesses. The proposed method addresses the challenges of navigating in dynamic environments with unpredictable obstacles by reducing uncertainty in local path planning and improving dynamic obstacle avoidance capabilities. Experimental results demonstrate that the InfoFusion Controller outperforms traditional methods in terms of safety, stability, and efficiency across various scenarios, including complex maps generated using SLAM techniques.
The code for the InfoFusion Controller is available at https: //github.com/DrawingProcess/InfoFusionController. 

**Abstract (ZH)**: 本文提出了一种名为InfoFusion Controller的高级路径规划算法，该算法结合全局和局部规划策略以增强在复杂城市环境中的自主驾驶能力。全局规划器采用知情Theta-Rapidly-exploring Random Tree Star (Informed-TRRT*) 算法生成最优参考路径，局部规划器结合了模型预测控制（MPC）和纯追踪算法。通过互信息（MI）融合MPC和纯追踪控制器的输出，有效平衡其优势并弥补其不足。所提出的方法通过减少局部路径规划中的不确定性并提高动态障碍物避免能力，解决了动态环境中不可预测障碍物的导航挑战。实验结果表明，在各种场景下，包括使用SLAM技术生成的复杂地图情景中，InfoFusion Controller在安全性、稳定性和效率方面均优于传统方法。 

---
# Optimal sensor deception in stochastic environments with partial observability to mislead a robot to a decoy goal 

**Title (ZH)**: 在部分可观测的随机环境中优化传感器欺骗以引导机器人偏离真实目标至诱饵目标 

**Authors**: Hazhar Rahmani, Mukulika Ghosh, Syed Md Hasnayeen  

**Link**: [PDF](https://arxiv.org/pdf/2503.05972)  

**Abstract**: Deception is a common strategy adapted by autonomous systems in adversarial settings. Existing deception methods primarily focus on increasing opacity or misdirecting agents away from their goal or itinerary. In this work, we propose a deception problem aiming to mislead the robot towards a decoy goal through altering sensor events under a constrained budget of alteration. The environment along with the robot's interaction with it is modeled as a Partially Observable Markov Decision Process (POMDP), and the robot's action selection is governed by a Finite State Controller (FSC). Given a constrained budget for sensor event modifications, the objective is to compute a sensor alteration that maximizes the probability of the robot reaching a decoy goal. We establish the computational hardness of the problem by a reduction from the $0/1$ Knapsack problem and propose a Mixed Integer Linear Programming (MILP) formulation to compute optimal deception strategies. We show the efficacy of our MILP formulation via a sequence of experiments. 

**Abstract (ZH)**: 自主系统在对抗环境中采用欺骗策略是一种常见策略。现有的欺骗方法主要关注于增加不透明度或引导代理偏离其目标或行程。本工作中，我们提出了一种新的欺骗问题，旨在通过在有限的修改预算下改变传感器事件，引导机器人向一个诱饵目标偏离。环境以及机器人与其的交互被建模为部分可观测马尔可夫决策过程（POMDP），机器人的动作选择由有限状态控制器（FSC）管理。给定传感器事件修改的预算限制，目标是计算一个能最大化机器人达到诱饵目标概率的传感器修改方案。通过对0/1背包问题进行归约来证明该问题的计算难题，并提出混合整数线性规划（MILP）方法来计算最优的欺骗策略。我们通过一系列实验展示了我们提出的MILP模型的有效性。 

---
# Universal Framework to Evaluate Automotive Perception Sensor Impact on Perception Functions 

**Title (ZH)**: 通用框架评估汽车感知传感器对感知功能的影响 

**Authors**: A Gamage, V Donzella  

**Link**: [PDF](https://arxiv.org/pdf/2503.05939)  

**Abstract**: Current research on automotive perception systems predominantly focusses on either improving the sensors for data quality or enhancing the performance of perception functions in isolation. Although automotive perception sensors form a fundamental part of the perception system, value addition in sensor data quality in isolation is questionable. However, the end goal for most perception systems is the accuracy of high-level functions such as trajectory prediction of surrounding vehicles. High-level perception functions are increasingly based on deep learning (DL) models due to their improved performance and generalisability compared to traditional algorithms. Innately, DL models develop a performance bias on the comprehensiveness of the training data. Despite the vital need to evaluate the performance of DL-based perception functions under real-world conditions using onboard sensor inputs, there is a lack of frameworks to facilitate systematic evaluations. This paper presents a versatile and cost-effective framework to evaluate the impact of perception sensor modalities and parameter settings on DL-based perception functions. Using a simulation environment, the framework facilitates sensor modality testing and parameter tuning under different environmental conditions. Its effectiveness is demonstrated through a case study involving a state-of-the-art surround trajectory prediction model, highlighting performance differences across sensor modalities and recommending optimal parameter settings. The proposed framework offers valuable insights for designing the perception sensor suite, contributing to the development of robust perception systems for autonomous vehicles. 

**Abstract (ZH)**: 基于传感器模态和参数设置对_DL驱动的感知函数影响的综合评价框架 

---
# A Representationalist, Functionalist and Naturalistic Conception of Intelligence as a Foundation for AGI 

**Title (ZH)**: 一种作为AGI基础的代表主义、功能主义和自然主义的智能观 

**Authors**: Rolf Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2503.07600)  

**Abstract**: The article analyses foundational principles relevant to the creation of artificial general intelligence (AGI). Intelligence is understood as the ability to create novel skills that allow to achieve goals under previously unknown conditions. To this end, intelligence utilises reasoning methods such as deduction, induction and abduction as well as other methods such as abstraction and classification to develop a world model. The methods are applied to indirect and incomplete representations of the world, which are obtained through perception, for example, and which do not depict the world but only correspond to it. Due to these limitations and the uncertain and contingent nature of reasoning, the world model is constructivist. Its value is functionally determined by its viability, i.e., its potential to achieve the desired goals. In consequence, meaning is assigned to representations by attributing them a function that makes it possible to achieve a goal. This representational and functional conception of intelligence enables a naturalistic interpretation that does not presuppose mental features, such as intentionality and consciousness, which are regarded as independent of intelligence. Based on a phenomenological analysis, it is shown that AGI can gain a more fundamental access to the world than humans, although it is limited by the No Free Lunch theorems, which require assumptions to be made. 

**Abstract (ZH)**: artificial generalintelligence的基础原则：一种基于表征与功能的解释及其对世界的自然主义理解 

---
# Real-Time Structural Deflection Estimation in Hydraulically Actuated Systems Using 3D Flexible Multibody Simulation and DNNs 

**Title (ZH)**: 使用3D柔性多体仿真和DNNs的液压驱动系统实时结构挠度估计 

**Authors**: Qasim Khadim, Peter Manzl, Emil Kurvinen, Aki Mikkola, Grzegorz Orzechowski, Johannes Gerstmayr  

**Link**: [PDF](https://arxiv.org/pdf/2503.07528)  

**Abstract**: The precision, stability, and performance of lightweight high-strength steel structures in heavy machinery is affected by their highly nonlinear dynamics. This, in turn, makes control more difficult, simulation more computationally intensive, and achieving real-time autonomy, using standard approaches, impossible. Machine learning through data-driven, physics-informed and physics-inspired networks, however, promises more computationally efficient and accurate solutions to nonlinear dynamic problems. This study proposes a novel framework that has been developed to estimate real-time structural deflection in hydraulically actuated three-dimensional systems. It is based on SLIDE, a machine-learning-based method to estimate dynamic responses of mechanical systems subjected to forced excitations.~Further, an algorithm is introduced for the data acquisition from a hydraulically actuated system using randomized initial configurations and hydraulic pressures.~The new framework was tested on a hydraulically actuated flexible boom with various sensor combinations and lifting various payloads. The neural network was successfully trained in less time using standard parameters from PyTorch, ADAM optimizer, the various sensor inputs, and minimal output data. The SLIDE-trained neural network accelerated deflection estimation solutions by a factor of $10^7$ in reference to flexible multibody simulation batches and provided reasonable accuracy. These results support the studies goal of providing robust, real-time solutions for control, robotic manipulators, structural health monitoring, and automation problems. 

**Abstract (ZH)**: 轻型高强钢结构在重型机械中的精度、稳定性和性能受其高度非线性动力学的影响，这使得控制更加困难，仿真计算更加耗时，使用标准方法实现实时自主控制成为不可能。然而，通过基于数据驱动、物理信息和物理启发的机器学习网络，有望为非线性动力学问题提供更高效和准确的解决方案。本研究提出了一种新的框架，旨在实时估计液压驱动的三维系统中的结构位移。该框架基于一种基于机器学习的方法SLIDE，用于估计受到强制激励的机械系统动力学响应。此外，介绍了一种算法，用于通过随机初始配置和液压压力从液压驱动系统中获取数据。新框架在多种传感器组合和不同负载下测试于一个液压驱动的柔性臂上。神经网络使用标准的PyTorch参数、ADAM优化器和多种传感器输入在较短的时间内成功训练，且少量输出数据即可。SLIDE训练的神经网络将位移估计解决方案的速度提高了$10^7$倍，并提供了合理的精度。这些结果支持了本研究旨在为控制、机器人 manipulator、结构健康监测和自动化问题提供稳健的实时解决方案的目标。 

---
# A High Efficient and Scalable Obstacle-Avoiding VLSI Global Routing Flow 

**Title (ZH)**: 一种高效可扩展的避障VLSI全局路由流程 

**Authors**: Junhao Guo, Hongxin Kong, Lang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.07268)  

**Abstract**: Routing is a crucial step in the VLSI design flow. With the advancement of manufacturing technologies, more constraints have emerged in design rules, particularly regarding obstacles during routing, leading to increased routing complexity. Unfortunately, many global routers struggle to efficiently generate obstacle-free solutions due to the lack of scalable obstacle-avoiding tree generation methods and the capability of handling modern designs with complex obstacles and nets. In this work, we propose an efficient obstacle-aware global routing flow for VLSI designs with obstacles. The flow includes a rule-based obstacle-avoiding rectilinear Steiner minimal tree (OARSMT) algorithm during the tree generation phase. This algorithm is both scalable and fast to provide tree topologies avoiding obstacles in the early stage globally. With its guidance, OARSMT-guided and obstacle-aware sparse maze routing are proposed in the later stages to minimize obstacle violations further and reduce overflow costs. Compared to advanced methods on the benchmark with obstacles, our approach successfully eliminates obstacle violations, and reduces wirelength and overflow cost, while sacrificing only a limited number of via counts and runtime overhead. 

**Abstract (ZH)**: VLSI 设计中具有障碍感知的高效全局布线流程 

---
# Performance-driven Constrained Optimal Auto-Tuner for MPC 

**Title (ZH)**: 基于性能驱动的约束最优自动化调谐器for MPC 

**Authors**: Albert Gassol Puigjaner, Manish Prajapat, Andrea Carron, Andreas Krause, Melanie N. Zeilinger  

**Link**: [PDF](https://arxiv.org/pdf/2503.07127)  

**Abstract**: A key challenge in tuning Model Predictive Control (MPC) cost function parameters is to ensure that the system performance stays consistently above a certain threshold. To address this challenge, we propose a novel method, COAT-MPC, Constrained Optimal Auto-Tuner for MPC. With every tuning iteration, COAT-MPC gathers performance data and learns by updating its posterior belief. It explores the tuning parameters' domain towards optimistic parameters in a goal-directed fashion, which is key to its sample efficiency. We theoretically analyze COAT-MPC, showing that it satisfies performance constraints with arbitrarily high probability at all times and provably converges to the optimum performance within finite time. Through comprehensive simulations and comparative analyses with a hardware platform, we demonstrate the effectiveness of COAT-MPC in comparison to classical Bayesian Optimization (BO) and other state-of-the-art methods. When applied to autonomous racing, our approach outperforms baselines in terms of constraint violations and cumulative regret over time. 

**Abstract (ZH)**: 自适应约束优化调谐器COAT-MPC：模型预测控制的成本函数参数调谐方法 

---
# The Multi-Trip Time-Dependent Mix Vehicle Routing Problem for Hybrid Autonomous Shared Delivery Location and Traditional Door-to-Door Delivery Modes 

**Title (ZH)**: 多行程时间依赖混合自主共享配送与传统门到门配送模式混合车辆路线问题 

**Authors**: Jingyi Zhao, Jiayu Yang, Haoxiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05842)  

**Abstract**: Rising labor costs and increasing logistical demands pose significant challenges to modern delivery systems. Automated Electric Vehicles (AEVs) could reduce reliance on delivery personnel and increase route flexibility, but their adoption is limited due to varying customer acceptance and integration complexities. Shared Distribution Locations (SDLs) offer an alternative to door-to-door (D2D) delivery by providing a wider delivery window and serving multiple community customers, thereby improving last-mile logistics through reduced delivery time, lower costs, and higher customer this http URL paper introduces the Multi-Trip Time-Dependent Hybrid Vehicle Routing Problem (MTTD-MVRP), a challenging variant of the Vehicle Routing Problem (VRP) that combines Autonomous Electric Vehicles (AEVs) with conventional vehicles. The problem's complexity arises from factors such as time-dependent travel speeds, strict time windows, battery limitations, and driver labor constraints, while integrating both SDLs and D2D deliveries. To solve the MTTD-MVRP efficiently, we develop a tailored meta-heuristic based on Adaptive Large Neighborhood Search (ALNS) augmented with column generation (CG). This approach intensively explores the solution space using problem-specific operators and adaptively refines solutions, balancing high-quality outcomes with computational effort. Extensive experiments show that the proposed method delivers near-optimal solutions for large-scale instances within practical time this http URL a managerial perspective, our findings highlight the importance of integrating autonomous and human-driven vehicles in last-mile logistics. Decision-makers can leverage SDLs to reduce operational costs and carbon footprints while still accommodating customers who require or prefer D2D services. 

**Abstract (ZH)**: Rising劳动成本和不断增加的物流需求对现代配送系统构成了重大挑战。自动化电动车辆（AEVs）可以减少对配送人员的依赖并增加路线灵活性，但其采用受限于客户接受度的差异性和集成复杂性。共享配送中心（SDLs）通过提供更宽的配送窗口并服务多个社区客户，为门到门（D2D）配送提供了替代方案，从而通过减少配送时间、降低配送成本和提高客户满意度来改善最后一英里物流。本文引入了多行程时间依赖混合车辆路径问题（MTTD-MVRP），这是车辆路径问题（VRP）的一个具有挑战性的变体，结合了自主电动车辆（AEVs）和传统车辆。该问题的复杂性来自于时间依赖的行驶速度、严格的时间窗口、电池限制和司机劳动力约束等因素，同时整合了SDLs和D2D配送。为了高效解决MTTD-MVRP，我们基于自适应大邻域搜索（ALNS）并结合列生成（CG）开发了一种定制的元启发式算法。该方法通过特定的问题操作深入探索解空间，并适应性地优化解决方案，旨在平衡高质量结果与计算努力。大量实验表明，所提出的方法可以在实际时间限制内为大规模实例提供近似最优解。从管理角度来看，我们的发现强调了在最后一英里物流中整合自主和人驱动车辆的重要性。决策者可以利用SDLs降低运营成本和碳足迹，同时仍能满足需要或偏好D2D服务的客户。 

---
# An Unsupervised C-Uniform Trajectory Sampler with Applications to Model Predictive Path Integral Control 

**Title (ZH)**: 无监督C-均匀轨迹采样及其在模型预测路径积分控制中的应用 

**Authors**: O. Goktug Poyrazoglu, Rahul Moorthy, Yukang Cao, William Chastek, Volkan Isler  

**Link**: [PDF](https://arxiv.org/pdf/2503.05819)  

**Abstract**: Sampling-based model predictive controllers generate trajectories by sampling control inputs from a fixed, simple distribution such as the normal or uniform distributions. This sampling method yields trajectory samples that are tightly clustered around a mean trajectory. This clustering behavior in turn, limits the exploration capability of the controller and reduces the likelihood of finding feasible solutions in complex environments. Recent work has attempted to address this problem by either reshaping the resulting trajectory distribution or increasing the sample entropy to enhance diversity and promote exploration. In our recent work, we introduced the concept of C-Uniform trajectory generation [1] which allows the computation of control input probabilities to generate trajectories that sample the configuration space uniformly. In this work, we first address the main limitation of this method: lack of scalability due to computational complexity. We introduce Neural C-Uniform, an unsupervised C-Uniform trajectory sampler that mitigates scalability issues by computing control input probabilities without relying on a discretized configuration space. Experiments show that Neural C-Uniform achieves a similar uniformity ratio to the original C-Uniform approach and generates trajectories over a longer time horizon while preserving uniformity. Next, we present CU-MPPI, which integrates Neural C-Uniform sampling into existing MPPI variants. We analyze the performance of CU-MPPI in simulation and real-world experiments. Our results indicate that in settings where the optimal solution has high curvature, CU-MPPI leads to drastic improvements in performance. 

**Abstract (ZH)**: 基于采样的模型预测控制器通过从固定简单的分布（如正态分布或均匀分布）中采样控制输入来生成轨迹。这种采样方法产生了紧密围绕均值轨迹的轨迹样本。这种聚类行为反过来限制了控制器的探索能力，并减少了在复杂环境中找到可行解的几率。最近的工作试图通过重塑生成的轨迹分布或增加样本熵来增强多样性并促进探索来解决这个问题。在我们最近的工作中，我们介绍了C-Uniform轨迹生成的概念，该概念允许计算控制输入概率以生成均匀覆盖配置空间的轨迹。在本文中，我们首先解决这种方法的主要局限性：由于计算复杂性导致的不具扩展性。我们引入了一种无需依赖离散化配置空间即可计算控制输入概率的无监督神经C-Uniform轨迹采样器，从而缓解了可扩展性问题。实验表明，神经C-Uniform实现了与原始C-Uniform方法相似的均匀性比率，并且可以在更长的时间框架内生成轨迹，同时保持均匀性。接下来，我们提出了CU-MPPI，将神经C-Uniform采样融入现有的MPPI变体中。我们在仿真和真实世界实验中分析了CU-MPPI的性能。我们的结果表明，在最优解具有高曲率的情况下，CU-MPPI在性能上会产生显著的改进。 

---
# AI-Enabled Knowledge Sharing for Enhanced Collaboration and Decision-Making in Non-Profit Healthcare Organizations: A Scoping Review Protocol 

**Title (ZH)**: 基于AI的知识共享以增强非营利医疗机构中的协作与决策制定：一项范围性回顾研究方案 

**Authors**: Maurice Ongala, Ruth Kiraka, Jyoti Choundrie, Javan Okello  

**Link**: [PDF](https://arxiv.org/pdf/2503.07540)  

**Abstract**: This protocol outlines a scoping review designed to systematically map the existing body of evidence on AI-enabled knowledge sharing in resource-limited non-profit healthcare organizations. The review aims to investigate how such technologies enhance collaboration and decision-making, particularly in the context of reduced external support following the cessation of USAID operations. Guided by three theoretical frameworks namely, the Resource-Based View, Dynamic Capabilities Theory, and Absorptive Capacity Theory, this study will explore the dual role of AI as a strategic resource and an enabler of organizational learning and agility. The protocol details a rigorous methodological approach based on PRISMA-ScR guidelines, encompassing a systematic search strategy across multiple databases, inclusion and exclusion criteria, and a structured data extraction process. By integrating theoretical insights with empirical evidence, this scoping review seeks to identify critical gaps in the literature and inform the design of effective, resource-optimized AI solutions in non-profit healthcare settings. 

**Abstract (ZH)**: 本研究概述了一项系统性回顾协议，旨在系统地梳理有限资源非营利医疗卫生组织中人工智能驱动的知识共享现有证据。该回顾旨在研究此类技术如何增强合作与决策制定，尤其是在美国国际开发署运作停止后外部支持减少的背景下。本研究将基于资源基础视角、动态能力理论和吸收能力理论三大理论框架，探索人工智能作为战略资源和组织学习与敏捷性促进者的双重角色。该协议基于PRISMA-ScR指南，详细描述了严格的方法学方法，包括多数据库系统的检索策略、纳入和排除标准以及结构化数据提取过程。通过将理论洞察与实证证据相结合，本系统性回顾旨在识别文献中的关键空白，并指导非营利医疗卫生环境中有效、资源优化的人工智能解决方案的设计。 

---
# Encoding Argumentation Frameworks to Propositional Logic Systems 

**Title (ZH)**: 将论证框架编码为命题逻辑系统 

**Authors**: Shuai Tang, Jiachao Wu, Ning Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.07351)  

**Abstract**: The theory of argumentation frameworks ($AF$s) has been a useful tool for artificial intelligence. The research of the connection between $AF$s and logic is an important branch. This paper generalizes the encoding method by encoding $AF$s as logical formulas in different propositional logic systems. It studies the relationship between models of an AF by argumentation semantics, including Dung's classical semantics and Gabbay's equational semantics, and models of the encoded formulas by semantics of propositional logic systems. Firstly, we supplement the proof of the regular encoding function in the case of encoding $AF$s to the 2-valued propositional logic system. Then we encode $AF$s to 3-valued propositional logic systems and fuzzy propositional logic systems and explore the model relationship. This paper enhances the connection between $AF$s and propositional logic systems. It also provides a new way to construct new equational semantics by choosing different fuzzy logic operations. 

**Abstract (ZH)**: 论辩框架理论（$AF$s）及其与逻辑的连接研究：不同命题逻辑系统中编码方法的推广与模型关系探讨 

---
# Automatic Curriculum Design for Zero-Shot Human-AI Coordination 

**Title (ZH)**: 零样本人机协调自动课程设计 

**Authors**: Won-Sang You, Tae-Gwan Ha, Seo-Young Lee, Kyung-Joong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.07275)  

**Abstract**: Zero-shot human-AI coordination is the training of an ego-agent to coordinate with humans without using human data. Most studies on zero-shot human-AI coordination have focused on enhancing the ego-agent's coordination ability in a given environment without considering the issue of generalization to unseen environments. Real-world applications of zero-shot human-AI coordination should consider unpredictable environmental changes and the varying coordination ability of co-players depending on the environment. Previously, the multi-agent UED (Unsupervised Environment Design) approach has investigated these challenges by jointly considering environmental changes and co-player policy in competitive two-player AI-AI scenarios. In this paper, our study extends the multi-agent UED approach to a zero-shot human-AI coordination. We propose a utility function and co-player sampling for a zero-shot human-AI coordination setting that helps train the ego-agent to coordinate with humans more effectively than the previous multi-agent UED approach. The zero-shot human-AI coordination performance was evaluated in the Overcooked-AI environment, using human proxy agents and real humans. Our method outperforms other baseline models and achieves a high human-AI coordination performance in unseen environments. 

**Abstract (ZH)**: 零样本人机协调训练ego-agent在给定环境中与人类协调而无需使用人类数据。大部分关于零样本人机协调的研究集中在增强ego-agent在特定环境中的协调能力，而未考虑未见过的环境下的泛化问题。零样本人机协调的实际应用需要考虑环境的不可预测变化以及随环境变化而变化的合作者协调能力。以往，多智能体无监督环境设计（UED）方法通过在竞争性二人对战的人工智能对战场景中同时考虑环境变化和合作者策略，来研究这些挑战。本文将无监督环境设计方法扩展至零样本人机协调场景，提出了一种效用函数和合作者采样方法，以提高ego-agent与人类协调的能力。零样本人机协调性能在Overcooked-AI环境中通过人类代理智能体和真实人类进行评估，结果显示本方法优于其他基线模型，并在未见过的环境中实现了高人机协调性能。 

---
# Lawful and Accountable Personal Data Processing with GDPR-based Access and Usage Control in Distributed Systems 

**Title (ZH)**: 基于GDPR的数据合法与负责任处理及其在分布式系统中的访问与使用控制 

**Authors**: L. Thomas van Binsbergen, Marten C. Steketee, Milen G. Kebede, Heleen L. Janssen, Tom M. van Engers  

**Link**: [PDF](https://arxiv.org/pdf/2503.07172)  

**Abstract**: Compliance with the GDPR privacy regulation places a significant burden on organisations regarding the handling of personal data. The perceived efforts and risks of complying with the GDPR further increase when data processing activities span across organisational boundaries, as is the case in both small-scale data sharing settings and in large-scale international data spaces.
This paper addresses these concerns by proposing a case-generic method for automated normative reasoning that establishes legal arguments for the lawfulness of data processing activities. The arguments are established on the basis of case-specific legal qualifications made by privacy experts, bringing the human in the loop. The obtained expert system promotes transparency and accountability, remains adaptable to extended or altered interpretations of the GDPR, and integrates into novel or existing distributed data processing systems.
This result is achieved by defining a formal ontology and semantics for automated normative reasoning based on an analysis of the purpose-limitation principle of the GDPR. The ontology and semantics are implemented in eFLINT, a domain-specific language for specifying and reasoning with norms. The XACML architecture standard, applicable to both access and usage control, is extended, demonstrating how GDPR-based normative reasoning can integrate into (existing, distributed) systems for data processing. The resulting system is designed and critically assessed in reference to requirements extracted from the GPDR. 

**Abstract (ZH)**: GDPR隐私法规遵守对组织在个人数据处理方面造成了显著负担。当数据处理活动跨越组织边界时，合规努力和风险进一步增加，这在小规模数据共享设置和大规模国际数据空间中尤为明显。

本文通过提出一种案例通用的方法来自动进行规范推理，为此类数据处理活动建立合法性的法律论据。这些论据基于隐私专家提供的案例特定的法律资格，从而使人在环中发挥作用。所得专家系统促进透明度和问责制，能够适应GDPR扩展或更改的解释，并可集成到新型或现有的分布式数据处理系统中。

这一结果是通过基于GDPR目的限制原则的分析，定义自动规范推理的形式本体论和语义来实现的。本体论和语义基于eFLINT（一个专门领域语言，用于规范的指定和推理）进行实现。扩展了适用于访问和使用控制的XACML架构标准，展示GDPR为基础的规范推理如何整合到（现有的、分布式的）数据处理系统中。最终系统根据从GDPR中提取的功能需求进行设计和批判性评估。 

---
# Generative AI in Transportation Planning: A Survey 

**Title (ZH)**: 生成式AI在交通规划中的应用：一个综述 

**Authors**: Longchao Da, Tiejin Chen, Zhuoheng Li, Shreyas Bachiraju, Huaiyuan Yao, Xiyang Hu, Zhengzhong Tu, Yue Zhao, Dongjie Wang, Xuanyu, Zhou, Ram Pendyala, Benjamin Stabler, Yezhou Yang, Xuesong Zhou, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.07158)  

**Abstract**: The integration of generative artificial intelligence (GenAI) into transportation planning has the potential to revolutionize tasks such as demand forecasting, infrastructure design, policy evaluation, and traffic simulation. However, there is a critical need for a systematic framework to guide the adoption of GenAI in this interdisciplinary domain. In this survey, we, a multidisciplinary team of researchers spanning computer science and transportation engineering, present the first comprehensive framework for leveraging GenAI in transportation planning. Specifically, we introduce a new taxonomy that categorizes existing applications and methodologies into two perspectives: transportation planning tasks and computational techniques. From the transportation planning perspective, we examine the role of GenAI in automating descriptive, predictive, generative, simulation, and explainable tasks to enhance mobility systems. From the computational perspective, we detail advancements in data preparation, domain-specific fine-tuning, and inference strategies, such as retrieval-augmented generation and zero-shot learning tailored to transportation applications. Additionally, we address critical challenges, including data scarcity, explainability, bias mitigation, and the development of domain-specific evaluation frameworks that align with transportation goals like sustainability, equity, and system efficiency. This survey aims to bridge the gap between traditional transportation planning methodologies and modern AI techniques, fostering collaboration and innovation. By addressing these challenges and opportunities, we seek to inspire future research that ensures ethical, equitable, and impactful use of generative AI in transportation planning. 

**Abstract (ZH)**: 将生成型人工智能（GenAI）整合到交通运输规划中，有可能革命性地改变需求预测、基础设施设计、政策评估和交通仿真等任务。然而，在这个跨学科领域中，系统性框架的缺乏限制了GenAI的应用。在本综述中，我们，来自计算机科学和交通运输工程领域的多学科研究团队，提出了首个整合GenAI于交通运输规划中的全面框架。具体而言，我们引入了一种新的分类体系，从交通运输规划任务和计算技术两个视角对现有应用和方法学进行分类。从交通运输规划视角出发，我们探讨了GenAI在自动化描述性、预测性、生成性、仿真性和解释性任务中的作用，以增强移动性系统。从计算技术视角出发，我们详细阐述了数据准备、领域特定微调和推理策略（如检索增强生成和针对交通运输应用的零样本学习）的进步。此外，我们还讨论了关键挑战，包括数据匮乏、可解释性、偏见缓解以及与可持续性、公平性和系统效率等交通运输目标相一致的领域特定评估框架的发展。本综述旨在弥合传统交通运输规划方法与现代AI技术之间的差距，促进协作与创新。通过解决这些挑战和机遇，我们希望激发未来研究，确保生成型AI在交通运输规划中的伦理、公平和影响。 

---
# Hierarchical Neuro-Symbolic Decision Transformer 

**Title (ZH)**: 分层神经符号决策变换器 

**Authors**: Ali Baheri, Cecilia O. Alm  

**Link**: [PDF](https://arxiv.org/pdf/2503.07148)  

**Abstract**: We present a hierarchical neuro-symbolic control framework that couples classical symbolic planning with transformer-based policies to address complex, long-horizon decision-making tasks. At the high level, a symbolic planner constructs an interpretable sequence of operators based on logical propositions, ensuring systematic adherence to global constraints and goals. At the low level, each symbolic operator is translated into a sub-goal token that conditions a decision transformer to generate a fine-grained sequence of actions in uncertain, high-dimensional environments. We provide theoretical analysis showing how approximation errors from both the symbolic planner and the neural execution layer accumulate. Empirical evaluations in grid-worlds with multiple keys, locked doors, and item-collection tasks show that our hierarchical approach outperforms purely end-to-end neural approach in success rates and policy efficiency. 

**Abstract (ZH)**: 一种结合经典符号规划和基于变压器的策略的层次神经符号控制框架及其应用 

---
# Correctness Learning: Deductive Verification Guided Learning for Human-AI Collaboration 

**Title (ZH)**: 正确性学习：基于演绎验证的-human-AI协作学习 

**Authors**: Zhao Jin, Lu Jin, Yizhe Luo, Shuo Feng, Yucheng Shi, Kai Zheng, Xinde Yu, Mingliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07096)  

**Abstract**: Despite significant progress in AI and decision-making technologies in safety-critical fields, challenges remain in verifying the correctness of decision output schemes and verification-result driven design. We propose correctness learning (CL) to enhance human-AI collaboration integrating deductive verification methods and insights from historical high-quality schemes. The typical pattern hidden in historical high-quality schemes, such as change of task priorities in shared resources, provides critical guidance for intelligent agents in learning and decision-making. By utilizing deductive verification methods, we proposed patten-driven correctness learning (PDCL), formally modeling and reasoning the adaptive behaviors-or 'correctness pattern'-of system agents based on historical high-quality schemes, capturing the logical relationships embedded within these schemes. Using this logical information as guidance, we establish a correctness judgment and feedback mechanism to steer the intelligent decision model toward the 'correctness pattern' reflected in historical high-quality schemes. Extensive experiments across multiple working conditions and core parameters validate the framework's components and demonstrate its effectiveness in improving decision-making and resource optimization. 

**Abstract (ZH)**: 尽管在AI和决策技术在安全关键领域取得了显著进展，但在验证决策输出方案的正确性和验证结果驱动的设计方面仍面临挑战。我们提出正确性学习（CL）以结合演绎验证方法和历史高质量方案的见解，增强人机协作。历史高质量方案中隐藏的典型模式，如共享资源中的任务优先级变化，为智能代理的学习和决策提供了关键指导。通过利用演绎验证方法，我们提出了模式驱动的正确性学习（PDCL），基于历史高质量方案形式化建模和推理系统代理的适应性行为或“正确性模式”，捕获这些方案内部嵌入的逻辑关系。利用这种逻辑信息作为指导，我们建立了一种正确性判断和反馈机制，引导智能决策模型向历史高质量方案中反映的“正确性模式”发展。广泛的实验验证了该框架的各个组件，并展示了其在提高决策和资源优化方面的有效性。 

---
# Enhancing Time Series Forecasting via Logic-Inspired Regularization 

**Title (ZH)**: 基于逻辑启发的正则化增强时间序列预测 

**Authors**: Jianqi Zhang, Jingyao Wang, Xingchen Shen, Wenwen Qiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06867)  

**Abstract**: Time series forecasting (TSF) plays a crucial role in many applications. Transformer-based methods are one of the mainstream techniques for TSF. Existing methods treat all token dependencies equally. However, we find that the effectiveness of token dependencies varies across different forecasting scenarios, and existing methods ignore these differences, which affects their performance. This raises two issues: (1) What are effective token dependencies? (2) How can we learn effective dependencies? From a logical perspective, we align Transformer-based TSF methods with the logical framework and define effective token dependencies as those that ensure the tokens as atomic formulas (Issue 1). We then align the learning process of Transformer methods with the process of obtaining atomic formulas in logic, which inspires us to design a method for learning these effective dependencies (Issue 2). Specifically, we propose Attention Logic Regularization (Attn-L-Reg), a plug-and-play method that guides the model to use fewer but more effective dependencies by making the attention map sparse, thereby ensuring the tokens as atomic formulas and improving prediction performance. Extensive experiments and theoretical analysis confirm the effectiveness of Attn-L-Reg. 

**Abstract (ZH)**: 基于Transformer的时间序列forecasting中有效token依赖的学习与正则化（Attention Logic Regularization） 

---
# Dubito Ergo Sum: Exploring AI Ethics 

**Title (ZH)**: 我思故我疑：探索人工智能伦理 

**Authors**: Viktor Dorfler, Giles Cuthbert  

**Link**: [PDF](https://arxiv.org/pdf/2503.06788)  

**Abstract**: We paraphrase Descartes' famous dictum in the area of AI ethics where the "I doubt and therefore I am" is suggested as a necessary aspect of morality. Therefore AI, which cannot doubt itself, cannot possess moral agency. Of course, this is not the end of the story. We explore various aspects of the human mind that substantially differ from AI, which includes the sensory grounding of our knowing, the act of understanding, and the significance of being able to doubt ourselves. The foundation of our argument is the discipline of ethics, one of the oldest and largest knowledge projects of human history, yet, we seem only to be beginning to get a grasp of it. After a couple of thousand years of studying the ethics of humans, we (humans) arrived at a point where moral psychology suggests that our moral decisions are intuitive, and all the models from ethics become relevant only when we explain ourselves. This recognition has a major impact on what and how we can do regarding AI ethics. We do not offer a solution, we explore some ideas and leave the problem open, but we hope somewhat better understood than before our study. 

**Abstract (ZH)**: 我们在AI伦理领域重述笛卡尔的名言，“我怀疑因此我存在”被视作道德的必要组成部分。因此，AI无法自我怀疑，无法具备道德代理能力。当然，这并不是故事的终点。我们探讨了人类心灵的各种方面，这些方面与AI有实质性差异，包括我们知识的感觉基础、理解的行动以及自我怀疑的意义。我们论点的基础是伦理学这一学科，它是人类历史最早且最大的知识项目之一，但似乎我们才刚刚开始理解和把握它。经过数千年对人类伦理的研究，我们认识到道德决策是直觉性的，所有的伦理模型只有在解释自身时才变得相关。这一认识对我们处理AI伦理问题的方式有着重大影响。我们没有提供解决方案，而是探讨了一些想法，并将问题留待开放，但我们希望在研究后比之前对问题有更深入的理解。 

---
# Beyond Black-Box Benchmarking: Observability, Analytics, and Optimization of Agentic Systems 

**Title (ZH)**: 超越黑盒基准测试：代理系统的可观测性、分析与优化 

**Authors**: Dany Moshkovich, Hadar Mulian, Sergey Zeltyn, Natti Eder, Inna Skarbovsky, Roy Abitbol  

**Link**: [PDF](https://arxiv.org/pdf/2503.06745)  

**Abstract**: The rise of agentic AI systems, where agents collaborate to perform diverse tasks, poses new challenges with observing, analyzing and optimizing their behavior. Traditional evaluation and benchmarking approaches struggle to handle the non-deterministic, context-sensitive, and dynamic nature of these systems. This paper explores key challenges and opportunities in analyzing and optimizing agentic systems across development, testing, and maintenance. We explore critical issues such as natural language variability and unpredictable execution flows, which hinder predictability and control, demanding adaptive strategies to manage input variability and evolving behaviors. Through our user study, we supported these hypotheses. In particular, we showed a 79% agreement that non deterministic flow of agentic systems acts as a major challenge. Finally, we validated our statements empirically advocating the need for moving beyond classical benchmarking. To bridge these gaps, we introduce taxonomies to present expected analytics outcomes and the ways to collect them by extending standard observability frameworks. Building on these foundations, we introduce and demonstrate novel approach for benchmarking of agent evaluation systems. Unlike traditional "black box" performance evaluation approaches, our benchmark is built from agent runtime logs as input, and analytics outcome including discovered flows and issues. By addressing key limitations in existing methodologies, we aim to set the stage for more advanced and holistic evaluation strategies, which could foster the development of adaptive, interpretable, and robust agentic AI systems. 

**Abstract (ZH)**: 代理型AI系统的兴起，其中代理协作执行多样任务，带来了新的挑战，涉及观察、分析和优化其行为。传统的评估和基准测试方法难以应对这些系统的非确定性、上下文敏感性和动态性。本文探讨了在开发、测试和维护过程中分析和优化代理型系统的关键挑战和机遇。我们研究了诸如自然语言变异性和不可预见的执行流程等核心问题，这些问题妨碍了可预测性和控制，要求采用适应性策略来管理输入变异性和不断变化的行为。通过我们的用户研究，我们支持了这些假设，并特别指出79%的参与者认为代理型系统的非确定性流程是主要挑战之一。最后，我们通过实证验证证明了需要超越经典基准测试的必要性。为了弥补这些差距，我们引入了分类学来呈现预期的分析结果及其收集方式，通过扩展标准可观测性框架。在这些建设的基础上，我们提出了并示范了一种新的代理评估基准测试方法。与其他传统的“黑盒”性能评估方法不同，我们的基准测试基于代理运行时日志作为输入，并结合分析结果，包括发现的流程和问题。通过解决现有方法的关键限制，我们旨在为更高级和综合的评估策略奠定基础，这有助于促进开发适应性强、可解释且稳健的代理型AI系统。 

---
# ChatGPT-4 in the Turing Test: A Critical Analysis 

**Title (ZH)**: ChatGPT-4 在图灵测试中的批判性分析 

**Authors**: Marco Giunti  

**Link**: [PDF](https://arxiv.org/pdf/2503.06551)  

**Abstract**: This paper critically examines the recent publication "ChatGPT-4 in the Turing Test" by Restrepo Echavarría (2025), challenging its central claims regarding the absence of minimally serious test implementations and the conclusion that ChatGPT-4 fails the Turing Test. The analysis reveals that the criticisms based on rigid criteria and limited experimental data are not fully justified. More importantly, the paper makes several constructive contributions that enrich our understanding of Turing Test implementations. It demonstrates that two distinct formats--the three-player and two-player tests--are both valid, each with unique methodological implications. The work distinguishes between absolute criteria (reflecting an optimal 50% identification rate in a three-player format) and relative criteria (which measure how closely a machine's performance approximates that of a human), offering a more nuanced evaluation framework. Furthermore, the paper clarifies the probabilistic underpinnings of both test types by modeling them as Bernoulli experiments--correlated in the three-player version and uncorrelated in the two-player version. This formalization allows for a rigorous separation between the theoretical criteria for passing the test, defined in probabilistic terms, and the experimental data that require robust statistical methods for proper interpretation. In doing so, the paper not only refutes key aspects of the criticized study but also lays a solid foundation for future research on objective measures of how closely an AI's behavior aligns with, or deviates from, that of a human being. 

**Abstract (ZH)**: 本文批判性地考察了Restrepo Echavarría (2025) 的近期出版物《ChatGPT-4在图灵测试中的表现》一文，对其关于最小严肃测试实施的缺失以及ChatGPT-4未通过图灵测试的主要论点提出质疑。分析表明，基于僵化标准和有限实验数据的批评并不完全成立。更重要的是，本文作出了若干建设性贡献，丰富了我们对图灵测试实施的理解。文章指出，三种赛制格式——三人测试和二人测试——都是有效的，各自具有独特的方法论意义。研究区分了绝对标准（反映三人测试中50%的理想识别率）和相对标准（衡量机器性能与人类表现的接近程度），提供了更为细致的评估框架。此外，论文通过将两种测试类型建模为伯努利试验——三人测试相关，二人测试不相关——澄清了测试类型的概率基础。这种形式化使得可以对测试通过的理论标准（以概率术语定义）和需要稳健统计方法解读的实验数据进行严格的分离。通过这种方式，本文不仅反驳了被批评研究的关键方面，也为未来关于人工智能行为与人类行为契合度或偏离程度的客观衡量研究奠定了坚实基础。 

---
# Think Twice, Click Once: Enhancing GUI Grounding via Fast and Slow Systems 

**Title (ZH)**: Twice思慎行，一次轻点：通过快慢系统增强GUI定位 

**Authors**: Fei Tang, Yongliang Shen, Hang Zhang, Siqi Chen, Guiyang Hou, Wenqi Zhang, Wenqiao Zhang, Kaitao Song, Weiming Lu, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06470)  

**Abstract**: Humans can flexibly switch between different modes of thinking based on task complexity: from rapid intuitive judgments to in-depth analytical understanding. However, current Graphical User Interface (GUI) grounding systems which locate interface elements based on natural language instructions rely solely on immediate prediction without reasoning, struggling to understand complex interface layouts with nested structures and hierarchical relationships, limiting their effectiveness on complex interfaces. Inspired by human dual-system cognition, we present Focus, a novel GUI grounding framework that combines fast prediction with systematic analysis. The framework dynamically switches between rapid and deliberate processing through an adaptive system switching based on task complexity, optimizing both efficiency and accuracy. Focus decomposes grounding into progressive stages: interface summarization, visual focused analysis, and precise coordinate prediction. This structured decomposition enables systematic understanding of both interface layouts and visual relationships. Extensive experiments show that Focus achieves state-of-the-art performance using only 300K of the training data with a 2B parameter model compared to existing approaches. Focus demonstrates superior performance particularly in complex GUI scenarios, achieving 77.4% average accuracy on ScreenSpot and 13.3% on the more challenging ScreenSpot-Pro. Our analysis reveals the effectiveness of this dual-system approach while demonstrating its potential for improving complex GUI interaction scenarios. 

**Abstract (ZH)**: 人类可以根据任务复杂度灵活切换不同的思维模式：从快速直觉判断到深入分析理解。然而，当前基于自然语言指令定位界面元素的图形用户界面（GUI）接地系统依赖于即时预测而不进行推理，难以理解具有嵌套结构和层次关系的复杂界面布局，限制了其在复杂界面中的有效性。受人类双系统认知的启发，我们提出 Focus，一个结合快速预测与系统分析的新型 GUI 接地框架。该框架通过基于任务复杂度的自适应系统切换，在保持效率的同时优化准确性。Focus 将接地分解为渐进的阶段：界面总结、视觉聚焦分析以及精确坐标预测。这种结构化的分解使得能够系统地理解界面布局及其视觉关系。广泛实验证明，Focus 使用仅为 300K 的训练数据和 2B 参数模型，实现了现有的最佳性能。在复杂 GUI 场景中，Focus 特别表现出色，在 ScreenSpot 中实现 77.4% 的平均准确率，在更具挑战性的 ScreenSpot-Pro 中实现 13.3% 的准确率。我们的分析揭示了这种双系统方法的有效性，并展示了其在改进复杂 GUI 交互场景中的潜力。 

---
# Explaining Control Policies through Predicate Decision Diagrams 

**Title (ZH)**: 通过谓词决策图解释控制策略 

**Authors**: Debraj Chakraborty, Clemens Dubslaff, Sudeep Kanav, Jan Kretinsky, Christoph Weinhuber  

**Link**: [PDF](https://arxiv.org/pdf/2503.06420)  

**Abstract**: Safety-critical controllers of complex systems are hard to construct manually. Automated approaches such as controller synthesis or learning provide a tempting alternative but usually lack explainability. To this end, learning decision trees (DTs) have been prevalently used towards an interpretable model of the generated controllers. However, DTs do not exploit shared decision-making, a key concept exploited in binary decision diagrams (BDDs) to reduce their size and thus improve explainability. In this work, we introduce predicate decision diagrams (PDDs) that extend BDDs with predicates and thus unite the advantages of DTs and BDDs for controller representation. We establish a synthesis pipeline for efficient construction of PDDs from DTs representing controllers, exploiting reduction techniques for BDDs also for PDDs. 

**Abstract (ZH)**: 复杂系统的安全关键控制器手工构建难度大。自动方法如控制器综合或学习提供了诱人的替代方案，但通常缺乏可解释性。为解决这一问题，已有研究表明，通过学习决策树（DTs）已经广泛用于生成具有可解释性的控制器模型。然而，DTs 未能利用共享决策这一关键概念，该概念在二元决策图（BDDs）中被用来减少其大小并提高可解释性。在此项工作中，我们引入了谓词决策图（PDDs），其扩展了BDDs以包含谓词，从而结合了决策树和BDDs的优点，用于控制器表示。我们建立了一个合成管道，从表示控制器的决策树（DTs）高效构建PDDs，并利用BDDs的缩减技术也适用于PDDs的技巧来建立PDDs。 

---
# Optimizing Minimum Vertex Cover Solving via a GCN-assisted Heuristic Algorithm 

**Title (ZH)**: 基于GCN辅助启发式算法的最小顶点覆盖优化求解 

**Authors**: Enqiang Zhu, Qiqi Bao, Yu Zhang, Chanjuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06396)  

**Abstract**: The problem of finding a minimum vertex cover (MVC) in a graph is a well-known NP-hard problem with significant practical applications in optimization and scheduling. Its complexity, combined with the increasing scale of problems, underscores the need for efficient and effective algorithms. However, existing heuristic algorithms for MVC often rely on simplistic initialization strategies and overlook the impact of edge attributes and neighborhood information on vertex selection. In this paper, we introduce GCNIVC, a novel heuristic search algorithm designed to address the limitations of existing methods for solving MVC problems in large-scale graphs. Our approach features two main innovations. First, it utilizes a Graph Convolutional Network (GCN) to capture the global structure of graphs, which enables the generation of high-quality initial solutions that enhance the efficiency of the subsequent search process. Second, GCNIVC introduces a new heuristic that employs three containers and the concept of double-covered edges (dc-edges), improving search efficiency and providing greater flexibility for adding and removing operations based on edge attributes. Through extensive experiments on benchmark datasets, we demonstrate that GCNIVC outperforms state-of-the-art MVC algorithms in terms of both accuracy and efficiency. Our results highlight the effectiveness of GCNIVC's GCN-assisted initialization and its edge-informed search strategy. This study not only advances the understanding of MVC problem-solving but also contributes a new tool for addressing large-scale graph optimization challenges. 

**Abstract (ZH)**: 基于图卷积网络的最小顶点覆盖新启发式搜索算法 

---
# Causal Discovery and Inference towards Urban Elements and Associated Factors 

**Title (ZH)**: 城市要素及其相关因素的因果发现与推断 

**Authors**: Tao Feng, Yunke Zhang, Xiaochen Fan, Huandong Wang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06395)  

**Abstract**: To uncover the city's fundamental functioning mechanisms, it is important to acquire a deep understanding of complicated relationships among citizens, location, and mobility behaviors. Previous research studies have applied direct correlation analysis to investigate such relationships. Nevertheless, due to the ubiquitous confounding effects, empirical correlation analysis may not accurately reflect underlying causal relationships among basic urban elements. In this paper, we propose a novel urban causal computing framework to comprehensively explore causalities and confounding effects among a variety of factors across different types of urban elements. In particular, we design a reinforcement learning algorithm to discover the potential causal graph, which depicts the causal relations between urban factors. The causal graph further serves as the guidance for estimating causal effects between pair-wise urban factors by propensity score matching. After removing the confounding effects from correlations, we leverage significance levels of causal effects in downstream urban mobility prediction tasks. Experimental studies on open-source urban datasets show that the discovered causal graph demonstrates a hierarchical structure, where citizens affect locations, and they both cause changes in urban mobility behaviors. Experimental results in urban mobility prediction tasks further show that the proposed method can effectively reduce confounding effects and enhance performance of urban computing tasks. 

**Abstract (ZH)**: 探索城市基本运作机制：一种新型城市因果计算框架及其应用 

---
# General Scales Unlock AI Evaluation with Explanatory and Predictive Power 

**Title (ZH)**: 通用尺度解锁具有解释性和预测性力量的AI评估 

**Authors**: Lexin Zhou, Lorenzo Pacchiardi, Fernando Martínez-Plumed, Katherine M. Collins, Yael Moros-Daval, Seraphina Zhang, Qinlin Zhao, Yitian Huang, Luning Sun, Jonathan E. Prunty, Zongqian Li, Pablo Sánchez-García, Kexin Jiang Chen, Pablo A. M. Casares, Jiyun Zu, John Burden, Behzad Mehrbakhsh, David Stillwell, Manuel Cebrian, Jindong Wang, Peter Henderson, Sherry Tongshuang Wu, Patrick C. Kyllonen, Lucy Cheke, Xing Xie, José Hernández-Orallo  

**Link**: [PDF](https://arxiv.org/pdf/2503.06378)  

**Abstract**: Ensuring safe and effective use of AI requires understanding and anticipating its performance on novel tasks, from advanced scientific challenges to transformed workplace activities. So far, benchmarking has guided progress in AI, but it has offered limited explanatory and predictive power for general-purpose AI systems, given the low transferability across diverse tasks. In this paper, we introduce general scales for AI evaluation that can explain what common AI benchmarks really measure, extract ability profiles of AI systems, and predict their performance for new task instances, in- and out-of-distribution. Our fully-automated methodology builds on 18 newly-crafted rubrics that place instance demands on general scales that do not saturate. Illustrated for 15 large language models and 63 tasks, high explanatory power is unleashed from inspecting the demand and ability profiles, bringing insights on the sensitivity and specificity exhibited by different benchmarks, and how knowledge, metacognition and reasoning are affected by model size, chain-of-thought and distillation. Surprisingly, high predictive power at the instance level becomes possible using these demand levels, providing superior estimates over black-box baseline predictors based on embeddings or finetuning, especially in out-of-distribution settings (new tasks and new benchmarks). The scales, rubrics, battery, techniques and results presented here represent a major step for AI evaluation, underpinning the reliable deployment of AI in the years ahead. 

**Abstract (ZH)**: 确保人工智能的安全和有效使用需要理解并预见其在新颖任务上的表现，从高级科学挑战到重塑的工作场所活动。目前，基准测试已指导人工智能的进步，但对于通用人工智能系统，它提供的解释和预测能力有限，因为这些任务之间的可迁移性较低。在本文中，我们提出了通用的评估尺度，可以解释常见的AI基准检测到的内容，提取AI系统的能力特征，并预测其在新任务实例中的表现，包括分布内和分布外的情况。我们的全自动方法基于18个新设计的评分标准，这些标准不饱和且适用于通用尺度。通过15个大型语言模型和63个任务的举例说明，通过对需求和能力特征的检查，释放了高解释力，揭示了不同基准的敏感性和特异性，并探讨了模型大小、思维链和知识、元认知及推理能力在这些特征上的影响。令人惊讶的是，在实例级别使用这些需求水平可以实现高预测能力，这优于基于嵌入或微调的黑盒基准预测器，特别是在分布外场景下（新任务和新基准）。这里呈现的尺度、评分标准、题库、技术和结果代表了人工智能评估的一大步，支撑了未来几年中人工智能的可靠部署。 

---
# Higher-Order Belief in Incomplete Information MAIDs 

**Title (ZH)**: 不完备信息条件下的高阶信念研究 

**Authors**: Jack Foxabbott, Rohan Subramani, Francis Rhys Ward  

**Link**: [PDF](https://arxiv.org/pdf/2503.06323)  

**Abstract**: Multi-agent influence diagrams (MAIDs) are probabilistic graphical models which represent strategic interactions between agents. MAIDs are equivalent to extensive form games (EFGs) but have a more compact and informative structure. However, MAIDs cannot, in general, represent settings of incomplete information -- wherein agents have different beliefs about the game being played, and different beliefs about each-other's beliefs. In this paper, we introduce incomplete information MAIDs (II-MAIDs). We define both infinite and finite-depth II-MAIDs and prove an equivalence relation to EFGs with incomplete information and no common prior over types. We prove that II-MAIDs inherit classical equilibria concepts via this equivalence, but note that these solution concepts are often unrealistic in the setting with no common prior because they violate common knowledge of rationality. We define a more realistic solution concept based on recursive best-response. Throughout, we describe an example with a hypothetical AI agent undergoing evaluation to illustrate the applicability of II-MAIDs. 

**Abstract (ZH)**: 不完备信息多智能体影响图（II-MAIDs） 

---
# LapSum -- One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection 

**Title (ZH)**: LapSum — 一种区分一切的方法：排名、排序和Top-k选择 

**Authors**: Łukasz Struski, Michał B. Bednarczyk, Igor T. Podolak, Jacek Tabor  

**Link**: [PDF](https://arxiv.org/pdf/2503.06242)  

**Abstract**: We present a novel technique for constructing differentiable order-type operations, including soft ranking, soft top-k selection, and soft permutations. Our approach leverages an efficient closed-form formula for the inverse of the function LapSum, defined as the sum of Laplace distributions. This formulation ensures low computational and memory complexity in selecting the highest activations, enabling losses and gradients to be computed in $O(n\log{}n)$ time. Through extensive experiments, we demonstrate that our method outperforms state-of-the-art techniques for high-dimensional vectors and large $k$ values. Furthermore, we provide efficient implementations for both CPU and CUDA environments, underscoring the practicality and scalability of our method for large-scale ranking and differentiable ordering problems. 

**Abstract (ZH)**: 我们提出了一种新的技术，用于构建可微序型操作，包括软排名、软top-k选择和软排列。该方法利用了LapSum函数（定义为拉普拉斯分布之和）的逆函数的有效闭式公式。该公式确保在选择最高激活值时具有较低的计算和内存复杂度，使得损失和梯度的计算时间为$O(n\log{}n)$。通过广泛的实验，我们证明了在高维向量和大k值情况下，我们的方法优于现有的先进技术。此外，我们提供了对CPU和CUDA环境的有效实现，突显了该方法在大规模排名和可微排序问题中的实用性和可扩展性。 

---
# MANDARIN: Mixture-of-Experts Framework for Dynamic Delirium and Coma Prediction in ICU Patients: Development and Validation of an Acute Brain Dysfunction Prediction Model 

**Title (ZH)**: MANDARIN：混合专家框架在ICU患者谵妄和昏迷动态预测中的应用：急性脑功能障碍预测模型的开发与验证 

**Authors**: Miguel Contreras, Jessica Sena, Andrea Davidson, Jiaqing Zhang, Tezcan Ozrazgat-Baslanti, Yuanfang Ren, Ziyuan Guan, Jeremy Balch, Tyler Loftus, Subhash Nerella, Azra Bihorac, Parisa Rashidi  

**Link**: [PDF](https://arxiv.org/pdf/2503.06059)  

**Abstract**: Acute brain dysfunction (ABD) is a common, severe ICU complication, presenting as delirium or coma and leading to prolonged stays, increased mortality, and cognitive decline. Traditional screening tools like the Glasgow Coma Scale (GCS), Confusion Assessment Method (CAM), and Richmond Agitation-Sedation Scale (RASS) rely on intermittent assessments, causing delays and inconsistencies. In this study, we propose MANDARIN (Mixture-of-Experts Framework for Dynamic Delirium and Coma Prediction in ICU Patients), a 1.5M-parameter mixture-of-experts neural network to predict ABD in real-time among ICU patients. The model integrates temporal and static data from the ICU to predict the brain status in the next 12 to 72 hours, using a multi-branch approach to account for current brain status. The MANDARIN model was trained on data from 92,734 patients (132,997 ICU admissions) from 2 hospitals between 2008-2019 and validated externally on data from 11,719 patients (14,519 ICU admissions) from 15 hospitals and prospectively on data from 304 patients (503 ICU admissions) from one hospital in 2021-2024. Three datasets were used: the University of Florida Health (UFH) dataset, the electronic ICU Collaborative Research Database (eICU), and the Medical Information Mart for Intensive Care (MIMIC)-IV dataset. MANDARIN significantly outperforms the baseline neurological assessment scores (GCS, CAM, and RASS) for delirium prediction in both external (AUROC 75.5% CI: 74.2%-76.8% vs 68.3% CI: 66.9%-69.5%) and prospective (AUROC 82.0% CI: 74.8%-89.2% vs 72.7% CI: 65.5%-81.0%) cohorts, as well as for coma prediction (external AUROC 87.3% CI: 85.9%-89.0% vs 72.8% CI: 70.6%-74.9%, and prospective AUROC 93.4% CI: 88.5%-97.9% vs 67.7% CI: 57.7%-76.8%) with a 12-hour lead time. This tool has the potential to assist clinicians in decision-making by continuously monitoring the brain status of patients in the ICU. 

**Abstract (ZH)**: 急性脑功能障碍的MANDARIN模型：一种用于ICU患者实时预测的混合专家框架 

---
# Empowering Edge Intelligence: A Comprehensive Survey on On-Device AI Models 

**Title (ZH)**: 赋能边缘智能：基于设备的AI模型综述 

**Authors**: Xubin Wang, Zhiqing Tang, Jianxiong Guo, Tianhui Meng, Chenhao Wang, Tian Wang, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.06027)  

**Abstract**: The rapid advancement of artificial intelligence (AI) technologies has led to an increasing deployment of AI models on edge and terminal devices, driven by the proliferation of the Internet of Things (IoT) and the need for real-time data processing. This survey comprehensively explores the current state, technical challenges, and future trends of on-device AI models. We define on-device AI models as those designed to perform local data processing and inference, emphasizing their characteristics such as real-time performance, resource constraints, and enhanced data privacy. The survey is structured around key themes, including the fundamental concepts of AI models, application scenarios across various domains, and the technical challenges faced in edge environments. We also discuss optimization and implementation strategies, such as data preprocessing, model compression, and hardware acceleration, which are essential for effective deployment. Furthermore, we examine the impact of emerging technologies, including edge computing and foundation models, on the evolution of on-device AI models. By providing a structured overview of the challenges, solutions, and future directions, this survey aims to facilitate further research and application of on-device AI, ultimately contributing to the advancement of intelligent systems in everyday life. 

**Abstract (ZH)**: 人工智能技术的飞速发展推动了边缘和终端设备上AI模型的部署，这主要是由于物联网的普及和实时数据处理的需求。本文综述了设备上AI模型的当前状态、技术挑战及未来趋势。我们将设备上AI模型定义为用于进行本地数据处理和推理的模型，强调其实时性能、资源限制和增强的数据隐私等特性。本文围绕关键主题展开，包括AI模型的基本概念、跨领域应用场景以及边缘环境中的技术挑战。我们还讨论了优化和实现策略，如数据预处理、模型压缩和硬件加速，这些对于有效的部署至关重要。此外，我们探讨了边缘计算和基础模型等新兴技术对设备上AI模型演进的影响。通过提供对挑战、解决方案及未来方向的结构化概述，本文旨在促进设备上AI的进一步研究和应用，最终推动智能系统的进步。 

---
# Bayesian Graph Traversal 

**Title (ZH)**: 贝叶斯图遍历 

**Authors**: William N. Caballero, Phillip R. Jenkins, David Banks, Matthew Robbins  

**Link**: [PDF](https://arxiv.org/pdf/2503.05963)  

**Abstract**: This research considers Bayesian decision-analytic approaches toward the traversal of an uncertain graph. Namely, a traveler progresses over a graph in which rewards are gained upon a node's first visit and costs are incurred for every edge traversal. The traveler knows the graph's adjacency matrix and his starting position but does not know the rewards and costs. The traveler is a Bayesian who encodes his beliefs about these values using a Gaussian process prior and who seeks to maximize his expected utility over these beliefs. Adopting a decision-analytic perspective, we develop sequential decision-making solution strategies for this coupled information-collection and network-routing problem. We show that the problem is NP-Hard and derive properties of the optimal walk. These properties provide heuristics for the traveler's problem that balance exploration and exploitation. We provide a practical case study focused on the use of unmanned aerial systems for public safety and empirically study policy performance in myriad Erdos-Renyi settings. 

**Abstract (ZH)**: 本研究考虑了面向不确定图的遍历的贝叶斯决策分析方法。旅行者在一张图上行进，首次访问节点可获得奖励，每通过一条边需支付成本。旅行者只知道图的邻接矩阵和起始位置，但不知道奖励和成本的具体数值。旅行者采用贝叶斯方法，通过高斯过程先验来编码他对这些值的信念，并力求在其信念上最大化其预期效用。从决策分析的角度出发，我们开发了针对这一信息收集与网络路径规划问题的序贯决策制定解决方案。我们证明了该问题是NP难问题，并推导出了最优路径的性质。这些性质为旅行者的决策提供了平衡探索与利用的启发式方法。我们提供了一个实际案例研究，关注无人驾驶航空系统在公共安全中的应用，并在众多的Erdos-Renyi设置下实证研究了策略性能。 

---
# Quantum-like cognition and decision making in the light of quantum measurement theory 

**Title (ZH)**: 基于量子测量理论视角下的 Quantum-like 认知与决策 

**Authors**: Miho Fuyama, Andrei Khrennikov, Masanao Ozawa  

**Link**: [PDF](https://arxiv.org/pdf/2503.05859)  

**Abstract**: We characterize the class of quantum measurements that matches the applications of quantum theory to cognition (and decision making) - quantum-like modeling. Projective measurements describe the canonical measurements of the basic observables of quantum physics. However, the combinations of the basic cognitive effects, such as the question order and response replicability effects, cannot be described by projective measurements. We motivate the use of the special class of quantum measurements, namely {\it sharp repeatable non-projective measurements} - ${\cal SR\bar{P}}. $ This class is practically unused in quantum physics. Thus, physics and cognition explore different parts of quantum measurement theory. Quantum-like modeling isn't automatic borrowing of the quantum formalism. Exploring the class ${\cal SR\bar{P}}$ highlights the role of {\it noncommutativity of the state update maps generated by measurement back action.} Thus, ``non-classicality'' in quantum physics as well as quantum-like modeling for cognition is based on two different types of noncommutativity, of operators (observables) and instruments (state update maps): {\it observable-noncommutativity} vs. {\it state update-noncommutativity}. We speculate that distinguishing quantum-like properties of the cognitive effects are the expressions of the latter, or possibly both. 

**Abstract (ZH)**: 量子测量分类及其在认知（决策）中的应用：基于非幺正可重复测量的量子似模型研究 

---
# A Comprehensive Survey of Fuzzy Implication Functions 

**Title (ZH)**: 模糊蕴含函数综述 

**Authors**: Raquel Fernandez-Peralta  

**Link**: [PDF](https://arxiv.org/pdf/2503.05702)  

**Abstract**: Fuzzy implication functions are a key area of study in fuzzy logic, extending the classical logical conditional to handle truth degrees in the interval $[0,1]$. While existing literature often focuses on a limited number of families, in the last ten years many new families have been introduced, each defined by specific construction methods and having different key properties. This survey aims to provide a comprehensive and structured overview of the diverse families of fuzzy implication functions, emphasizing their motivations, properties, and potential applications. By organizing the information schematically, this document serves as a valuable resource for both theoretical researchers seeking to avoid redundancy and practitioners looking to select appropriate operators for specific applications. 

**Abstract (ZH)**: 模糊蕴含函数是模糊逻辑中的一个关键研究领域，扩展了经典的逻辑条件以处理区间$[0,1]$内的真度。尽管现有文献通常集中于少数几类，但在过去的十年里，引入了很多新的家庭，每种家庭都由特定的构造方法定义并具有不同的关键属性。本文综述旨在提供对多样化模糊蕴含函数家庭的全面和结构化的概述，强调它们的动机、属性和潜在应用。通过方案化地组织信息，本文文件既是对理论研究人员避免冗余的宝贵资源，也是对实践者为特定应用选择合适运算符的指导。 

---
# Denoising Hamiltonian Network for Physical Reasoning 

**Title (ZH)**: Hamiltonian网络去噪方法用于物理推理 

**Authors**: Congyue Deng, Brandon Y. Feng, Cecilia Garraffo, Alan Garbarz, Robin Walters, William T. Freeman, Leonidas Guibas, Kaiming He  

**Link**: [PDF](https://arxiv.org/pdf/2503.07596)  

**Abstract**: Machine learning frameworks for physical problems must capture and enforce physical constraints that preserve the structure of dynamical systems. Many existing approaches achieve this by integrating physical operators into neural networks. While these methods offer theoretical guarantees, they face two key limitations: (i) they primarily model local relations between adjacent time steps, overlooking longer-range or higher-level physical interactions, and (ii) they focus on forward simulation while neglecting broader physical reasoning tasks. We propose the Denoising Hamiltonian Network (DHN), a novel framework that generalizes Hamiltonian mechanics operators into more flexible neural operators. DHN captures non-local temporal relationships and mitigates numerical integration errors through a denoising mechanism. DHN also supports multi-system modeling with a global conditioning mechanism. We demonstrate its effectiveness and flexibility across three diverse physical reasoning tasks with distinct inputs and outputs. 

**Abstract (ZH)**: 物理问题中的机器学习框架必须捕获并执行物理约束以保持动态系统的结构。许多现有方法通过将物理运算符集成到神经网络中来实现这一目标。虽然这些方法具有理论保证，但它们面临着两个关键限制：（i）它们主要建模相邻时间步之间的局部关系，忽略了更长范围或更高层次的物理交互；（ii）它们专注于正向仿真，忽视了更广泛的物理推理任务。我们提出了一种新颖的框架去噪哈密顿网络（Denoising Hamiltonian Network, DHN），该框架将哈密顿力学运算符泛化为更灵活的神经运算符。DHN 捕捉非局部时间关系并通过去噪机制减轻数值积分误差。DHN 还通过全局条件机制支持多系统建模。我们通过三个具有不同输入和输出的物理推理任务展示了其有效性和灵活性。 

---
# Filter Images First, Generate Instructions Later: Pre-Instruction Data Selection for Visual Instruction Tuning 

**Title (ZH)**: 先过滤图片，后生成指令：预指令数据选择方法用于视觉指令调优 

**Authors**: Bardia Safaei, Faizan Siddiqui, Jiacong Xu, Vishal M. Patel, Shao-Yuan Lo  

**Link**: [PDF](https://arxiv.org/pdf/2503.07591)  

**Abstract**: Visual instruction tuning (VIT) for large vision-language models (LVLMs) requires training on expansive datasets of image-instruction pairs, which can be costly. Recent efforts in VIT data selection aim to select a small subset of high-quality image-instruction pairs, reducing VIT runtime while maintaining performance comparable to full-scale training. However, a major challenge often overlooked is that generating instructions from unlabeled images for VIT is highly expensive. Most existing VIT datasets rely heavily on human annotations or paid services like the GPT API, which limits users with constrained resources from creating VIT datasets for custom applications. To address this, we introduce Pre-Instruction Data Selection (PreSel), a more practical data selection paradigm that directly selects the most beneficial unlabeled images and generates instructions only for the selected images. PreSel first estimates the relative importance of each vision task within VIT datasets to derive task-wise sampling budgets. It then clusters image features within each task, selecting the most representative images with the budget. This approach reduces computational overhead for both instruction generation during VIT data formation and LVLM fine-tuning. By generating instructions for only 15% of the images, PreSel achieves performance comparable to full-data VIT on the LLaVA-1.5 and Vision-Flan datasets. The link to our project page: this https URL 

**Abstract (ZH)**: 预指令数据选择（PreSel）：一种用于大规模视觉-语言模型（LVLMs）的视觉指令调优（VIT）的数据选择范式 

---
# Runtime Detection of Adversarial Attacks in AI Accelerators Using Performance Counters 

**Title (ZH)**: 使用性能计数器在AI加速器中运行时检测 adversarial 攻击 

**Authors**: Habibur Rahaman, Atri Chatterjee, Swarup Bhunia  

**Link**: [PDF](https://arxiv.org/pdf/2503.07568)  

**Abstract**: Rapid adoption of AI technologies raises several major security concerns, including the risks of adversarial perturbations, which threaten the confidentiality and integrity of AI applications. Protecting AI hardware from misuse and diverse security threats is a challenging task. To address this challenge, we propose SAMURAI, a novel framework for safeguarding against malicious usage of AI hardware and its resilience to attacks. SAMURAI introduces an AI Performance Counter (APC) for tracking dynamic behavior of an AI model coupled with an on-chip Machine Learning (ML) analysis engine, known as TANTO (Trained Anomaly Inspection Through Trace Observation). APC records the runtime profile of the low-level hardware events of different AI operations. Subsequently, the summary information recorded by the APC is processed by TANTO to efficiently identify potential security breaches and ensure secure, responsible use of AI. SAMURAI enables real-time detection of security threats and misuse without relying on traditional software-based solutions that require model integration. Experimental results demonstrate that SAMURAI achieves up to 97% accuracy in detecting adversarial attacks with moderate overhead on various AI models, significantly outperforming conventional software-based approaches. It enhances security and regulatory compliance, providing a comprehensive solution for safeguarding AI against emergent threats. 

**Abstract (ZH)**: 快速adopt AI技术引发了一系列重大安全 concern，包括对抗性 perturbations 的风险，这些风险威胁到AI应用的保密性和完整性。保护AI硬件免受滥用和各种安全威胁是一项具有挑战性的任务。为应对这一挑战，我们提出了SAMURAI，一种新型框架，旨在保护AI硬件免受恶意使用，并增强其对攻击的 resilience。SAMURAI引入了一个AI性能计数器（APC）来跟踪不同AI操作的动态行为，并结合了一个嵌入式机器学习（ML）分析引擎，称为TANTO（通过跟踪观察训练异常）。APC记录了不同AI操作的低级硬件事件的运行时配置文件。随后，TANTO处理APC记录的概要信息，以高效地识别潜在的安全 breach，并确保AI的 secure和 responsible使用。SAMURAI能够在不依赖传统软件解决方案的前提下实现对安全威胁和滥用的实时 detection，这些传统解决方案需要与模型集成。实验结果表明，SAMURAI在各种AI模型上实现了高达97%的检测对抗性攻击的准确率，且具有适度的overhead，显著优于传统的软件解决方案。它增强了安全性和合规性，提供了一种全面的解决方案，以抵御新兴威胁对AI的攻击。 

---
# Interference-Aware Super-Constellation Design for NOMA 

**Title (ZH)**: 干扰意识下的NOMA超级星系设计 

**Authors**: Mojtaba Vaezi, Xinliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07509)  

**Abstract**: Non-orthogonal multiple access (NOMA) has gained significant attention as a potential next-generation multiple access technique. However, its implementation with finite-alphabet inputs faces challenges. Particularly, due to inter-user interference, superimposed constellations may have overlapping symbols leading to high bit error rates when successive interference cancellation (SIC) is applied. To tackle the issue, this paper employs autoencoders to design interference-aware super-constellations. Unlike conventional methods where superimposed constellation may have overlapping symbols, the proposed autoencoder-based NOMA (AE-NOMA) is trained to design super-constellations with distinguishable symbols at receivers, regardless of channel gains. The proposed architecture removes the need for SIC, allowing maximum likelihood-based approaches to be used instead. The paper presents the conceptual architecture, loss functions, and training strategies for AE-NOMA. Various test results are provided to demonstrate the effectiveness of interference-aware constellations in improving the bit error rate, indicating the adaptability of AE-NOMA to different channel scenarios and its promising potential for implementing NOMA systems 

**Abstract (ZH)**: 非正交多址(NOMA)作为一种潜在的下一代多址技术已获得广泛关注，但其在有限字母表输入下的实现面临挑战。特别是由于用户间干扰，叠加星座可能具有重叠符号，导致在 successive interference cancellation (SIC) 应用时出现高比特错误率。为解决该问题，本文采用自编码器设计干扰感知叠加星座。与传统方法不同，所提基于自编码器的NOMA (AE-NOMA)经过训练可在接收端设计具有可区分符号的叠加星座，而不考虑信道增益。该提出的架构消除了SIC的需求，允许使用最大似然方法。本文介绍了AE-NOMA的概念架构、损失函数和训练策略，并提供了各种测试结果以证明干扰感知星座在提高比特误差率方面的有效性，展示了AE-NOMA对不同信道场景的适应性和实施NOMA系统的乐观前景。 

---
# From Centralized to Decentralized Federated Learning: Theoretical Insights, Privacy Preservation, and Robustness Challenges 

**Title (ZH)**: 从集中式到去中心化联邦学习：理论见解、隐私保护与健壮性挑战 

**Authors**: Qiongxiu Li, Wenrui Yu, Yufei Xia, Jun Pang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07505)  

**Abstract**: Federated Learning (FL) enables collaborative learning without directly sharing individual's raw data. FL can be implemented in either a centralized (server-based) or decentralized (peer-to-peer) manner. In this survey, we present a novel perspective: the fundamental difference between centralized FL (CFL) and decentralized FL (DFL) is not merely the network topology, but the underlying training protocol: separate aggregation vs. joint optimization. We argue that this distinction in protocol leads to significant differences in model utility, privacy preservation, and robustness to attacks. We systematically review and categorize existing works in both CFL and DFL according to the type of protocol they employ. This taxonomy provides deeper insights into prior research and clarifies how various approaches relate or differ. Through our analysis, we identify key gaps in the literature. In particular, we observe a surprising lack of exploration of DFL approaches based on distributed optimization methods, despite their potential advantages. We highlight this under-explored direction and call for more research on leveraging distributed optimization for federated learning. Overall, this work offers a comprehensive overview from centralized to decentralized FL, sheds new light on the core distinctions between approaches, and outlines open challenges and future directions for the field. 

**Abstract (ZH)**: 联邦学习（FL）使个体无需直接共享原始数据即可进行协作学习。FL可以在中心化（基于服务器）或去中心化（点对点）的方式下实施。在本综述中，我们提出一种新颖的观点：中心化FL（CFL）和去中心化FL（DFL）之间的根本区别不仅在于网络拓扑，还在于基础的训练协议：分离聚合 vs. 联合优化。我们认为这种协议上的区别导致了模型实用性、隐私保护以及对抗攻击鲁棒性方面的显著差异。我们系统地按照所采用的协议类型对CFL和DFL中的现有工作进行了回顾和分类。这种分类提供了对先前研究的更深入洞察，并明确了各种方法之间的关联或差异。通过对这些方法的分析，我们识别出了文献中的关键空白。特别是，我们注意到基于分布式优化方法的DFL方法探索不足，尽管它们具有潜在的优势。我们强调了这一未充分探索的方向，并呼吁对利用分布式优化进行联邦学习的研究进行更多关注。总体而言，本工作提供了从中心化到去中心化FL的全面概述，从新的角度揭示了各种方法的核心区别，并指出了该领域面临的一些公开挑战和未来方向。 

---
# Efficient Membership Inference Attacks by Bayesian Neural Network 

**Title (ZH)**: 基于贝叶斯神经网络的高效成员推理攻击 

**Authors**: Zhenlong Liu, Wenyu Jiang, Feng Zhou, Hongxin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.07482)  

**Abstract**: Membership Inference Attacks (MIAs) aim to estimate whether a specific data point was used in the training of a given model. Previous attacks often utilize multiple reference models to approximate the conditional score distribution, leading to significant computational overhead. While recent work leverages quantile regression to estimate conditional thresholds, it fails to capture epistemic uncertainty, resulting in bias in low-density regions. In this work, we propose a novel approach - Bayesian Membership Inference Attack (BMIA), which performs conditional attack through Bayesian inference. In particular, we transform a trained reference model into Bayesian neural networks by Laplace approximation, enabling the direct estimation of the conditional score distribution by probabilistic model parameters. Our method addresses both epistemic and aleatoric uncertainty with only a reference model, enabling efficient and powerful MIA. Extensive experiments on five datasets demonstrate the effectiveness and efficiency of BMIA. 

**Abstract (ZH)**: Bayesian Membership Inference Attack (BMIA): Performing Conditional Attacks through Bayesian Inference 

---
# Advancing Vietnamese Information Retrieval with Learning Objective and Benchmark 

**Title (ZH)**: 基于学习目标和基准提高越南语信息检索 

**Authors**: Phu-Vinh Nguyen, Minh-Nam Tran, Long Nguyen, Dien Dinh  

**Link**: [PDF](https://arxiv.org/pdf/2503.07470)  

**Abstract**: With the rapid development of natural language processing, many language models have been invented for multiple tasks. One important task is information retrieval (IR), which requires models to retrieve relevant documents. Despite its importance in many real-life applications, especially in retrieval augmented generation (RAG) systems, this task lacks Vietnamese benchmarks. This situation causes difficulty in assessing and comparing many existing Vietnamese embedding language models on the task and slows down the advancement of Vietnamese natural language processing (NLP) research. In this work, we aim to provide the Vietnamese research community with a new benchmark for information retrieval, which mainly focuses on retrieval and reranking tasks. Furthermore, we also present a new objective function based on the InfoNCE loss function, which is used to train our Vietnamese embedding model. Our function aims to be better than the origin in information retrieval tasks. Finally, we analyze the effect of temperature, a hyper-parameter in both objective functions, on the performance of text embedding models. 

**Abstract (ZH)**: 随着自然语言处理的快速发展，许多语言模型被发明用于多种任务。其中一个重要的任务是信息检索（IR），要求模型检索相关文档。尽管信息检索在许多实际应用中非常重要，尤其是在检索增强生成（RAG）系统中，这一任务缺乏越南语基准。这种状况导致了难以评估和比较许多现有的越南语嵌入语言模型在该任务上的性能，并阻碍了越南语自然语言处理（NLP）研究的发展。在本文中，我们旨在为越南语研究社区提供一个新的信息检索基准，主要关注检索和再排序任务。此外，我们还提出了一种基于InfoNCE损失函数的新目标函数，用于训练我们的越南语嵌入模型。我们的函数旨在在信息检索任务中表现更优。最后，我们分析了目标函数中的温度这一超参数对文本嵌入模型性能的影响。 

---
# MedAgentsBench: Benchmarking Thinking Models and Agent Frameworks for Complex Medical Reasoning 

**Title (ZH)**: MedAgentsBench: 评估复杂医疗推理模型与代理框架的基准测试 

**Authors**: Xiangru Tang, Daniel Shao, Jiwoong Sohn, Jiapeng Chen, Jiayi Zhang, Jinyu Xiang, Fang Wu, Yilun Zhao, Chenglin Wu, Wenqi Shi, Arman Cohan, Mark Gerstein  

**Link**: [PDF](https://arxiv.org/pdf/2503.07459)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance on existing medical question-answering benchmarks. This high performance makes it increasingly difficult to meaningfully evaluate and differentiate advanced methods. We present MedAgentsBench, a benchmark that focuses on challenging medical questions requiring multi-step clinical reasoning, diagnosis formulation, and treatment planning-scenarios where current models still struggle despite their strong performance on standard tests. Drawing from seven established medical datasets, our benchmark addresses three key limitations in existing evaluations: (1) the prevalence of straightforward questions where even base models achieve high performance, (2) inconsistent sampling and evaluation protocols across studies, and (3) lack of systematic analysis of the interplay between performance, cost, and inference time. Through experiments with various base models and reasoning methods, we demonstrate that the latest thinking models, DeepSeek R1 and OpenAI o3, exhibit exceptional performance in complex medical reasoning tasks. Additionally, advanced search-based agent methods offer promising performance-to-cost ratios compared to traditional approaches. Our analysis reveals substantial performance gaps between model families on complex questions and identifies optimal model selections for different computational constraints. Our benchmark and evaluation framework are publicly available at this https URL. 

**Abstract (ZH)**: MedAgentsBench：针对复杂医学推理任务的基准测试 

---
# Divide and Conquer Self-Supervised Learning for High-Content Imaging 

**Title (ZH)**: 征服与征服：高内容成像中的分级自我监督学习 

**Authors**: Lucas Farndale, Paul Henderson, Edward W Roberts, Ke Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07444)  

**Abstract**: Self-supervised representation learning methods often fail to learn subtle or complex features, which can be dominated by simpler patterns which are much easier to learn. This limitation is particularly problematic in applications to science and engineering, as complex features can be critical for discovery and analysis. To address this, we introduce Split Component Embedding Registration (SpliCER), a novel architecture which splits the image into sections and distils information from each section to guide the model to learn more subtle and complex features without compromising on simpler features. SpliCER is compatible with any self-supervised loss function and can be integrated into existing methods without modification. The primary contributions of this work are as follows: i) we demonstrate that existing self-supervised methods can learn shortcut solutions when simple and complex features are both present; ii) we introduce a novel self-supervised training method, SpliCER, to overcome the limitations of existing methods, and achieve significant downstream performance improvements; iii) we demonstrate the effectiveness of SpliCER in cutting-edge medical and geospatial imaging settings. SpliCER offers a powerful new tool for representation learning, enabling models to uncover complex features which could be overlooked by other methods. 

**Abstract (ZH)**: 自监督表示学习方法往往难以学习细微或复杂的特征，这些特征可能会被更简单且更容易学习的模式所主导。这一局限性在科学和工程应用中尤为令人困扰，因为复杂的特征对于发现和分析至关重要。为了解决这一问题，我们提出了一种新颖的架构——Split Component Embedding Registration (SpliCER)，该架构将图像划分为多个部分，并从每个部分提取信息以引导模型学习更为细微和复杂的特征，同时保留简单特征的学习。SpliCER 可与任何自监督损失函数兼容，并且可以无缝集成到现有方法中而不需修改。本文的主要贡献如下：i) 我们证明了现有的自监督方法在同时存在简单和复杂特征时会学习捷径解；ii) 我们引入了一种全新的自监督训练方法 SpliCER，以克服现有方法的局限性，并显著提高了下游性能；iii) 我们展示了 SpliCER 在尖端的医学和地理空间成像场景中的有效性。SpliCER 提供了一种强大的新工具，使模型能够发现其他方法可能忽略的复杂特征。 

---
# Brain Inspired Adaptive Memory Dual-Net for Few-Shot Image Classification 

**Title (ZH)**: 仿生自适应记忆双网络在少样本图像分类中的应用 

**Authors**: Kexin Di, Xiuxing Li, Yuyang Han, Ziyu Li, Qing Li, Xia Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07396)  

**Abstract**: Few-shot image classification has become a popular research topic for its wide application in real-world scenarios, however the problem of supervision collapse induced by single image-level annotation remains a major challenge. Existing methods aim to tackle this problem by locating and aligning relevant local features. However, the high intra-class variability in real-world images poses significant challenges in locating semantically relevant local regions under few-shot settings. Drawing inspiration from the human's complementary learning system, which excels at rapidly capturing and integrating semantic features from limited examples, we propose the generalization-optimized Systems Consolidation Adaptive Memory Dual-Network, SCAM-Net. This approach simulates the systems consolidation of complementary learning system with an adaptive memory module, which successfully addresses the difficulty of identifying meaningful features in few-shot scenarios. Specifically, we construct a Hippocampus-Neocortex dual-network that consolidates structured representation of each category, the structured representation is then stored and adaptively regulated following the generalization optimization principle in a long-term memory inside Neocortex. Extensive experiments on benchmark datasets show that the proposed model has achieved state-of-the-art performance. 

**Abstract (ZH)**: 基于Few-shot图像分类的Generalization-Optimized Systems Consolidation Adaptive Memory Dual-NetworkSCAM-Net 

---
# Artificial Utopia: Simulation and Intelligent Agents for a Democratised Future 

**Title (ZH)**: 人工乌托邦：仿真与智能代理 toward一个民主化未来 

**Authors**: Yannick Oswald  

**Link**: [PDF](https://arxiv.org/pdf/2503.07364)  

**Abstract**: Prevailing top-down systems in politics and economics struggle to keep pace with the pressing challenges of the 21st century, such as climate change, social inequality and conflict. Bottom-up democratisation and participatory approaches in politics and economics are increasingly seen as promising alternatives to confront and overcome these issues, often with utopian overtones, as proponents believe they may dramatically reshape political, social and ecological futures for the better and in contrast to contemporary authoritarian tendencies across various countries. Institutional specifics and the associated collective human behavior or culture remains little understood and debated, however. In this article, I propose a novel research agenda focusing on utopian democratisation efforts with formal and computational methods as well as with artificial intelligence - I call this agenda Artificial Utopia. Artificial Utopias provide safe testing grounds for new political ideas and economic policies in-silico with reduced risk of negative consequences as compared to testing ideas in real-world contexts. An increasing number of advanced simulation and intelligence methods, that aim at representing human cognition and collective decision-making in more realistic ways, could benefit this process. This includes agent-based modelling, reinforcement learning, large language models and more. I clarify what some of these simulation approaches can contribute to the study of Artificial Utopias with the help of two institutional examples: the citizen assembly and the democratic firm. 

**Abstract (ZH)**: 占据主导地位的政治和经济自上而下体系难以应对21世纪紧迫的挑战，如气候变化、社会不平等和冲突。自下而上的民主化和参与性方法在政治和经济中越来越被视为对抗和克服这些问题的有希望的替代方案，常常带有乌托邦的色彩，因为支持者认为它们可能会显著塑造更美好的政治、社会和生态未来，这与当今各国日益增强的威权倾向相对。然而，相关机构的具体特性和与之相关的集体人类行为或文化仍知之甚少且讨论不多。在本文中，我提出了一项新的研究议程，专注于使用形式化和计算方法以及人工智能进行乌托邦民主化的努力——我将这一议程称为“人工智能乌托邦”。人工智能乌托邦为在硅中测试新的政治理念和经济政策提供了安全的试验场，相比于在现实世界中测试概念，其负面后果的风险更低。越来越多旨在以更现实的方式代表人类认知和集体决策的先进仿真和智能方法可以受益于这一过程。这包括基于代理的建模、强化学习、大规模语言模型等。利用两个机构性案例——公民议会和民主企业，我阐明了这些仿真方法如何为研究人工智能乌托邦做出贡献。 

---
# The Economics of p(doom): Scenarios of Existential Risk and Economic Growth in the Age of Transformative AI 

**Title (ZH)**: Transformative AI时代的生死风险与经济增长经济学：p(doom)情境研究 

**Authors**: Jakub Growiec, Klaus Prettner  

**Link**: [PDF](https://arxiv.org/pdf/2503.07341)  

**Abstract**: Recent advances in artificial intelligence (AI) have led to a diverse set of predictions about its long-term impact on humanity. A central focus is the potential emergence of transformative AI (TAI), eventually capable of outperforming humans in all economically valuable tasks and fully automating labor. Discussed scenarios range from human extinction after a misaligned TAI takes over ("AI doom") to unprecedented economic growth and abundance ("post-scarcity"). However, the probabilities and implications of these scenarios remain highly uncertain. Here, we organize the various scenarios and evaluate their associated existential risks and economic outcomes in terms of aggregate welfare. Our analysis shows that even low-probability catastrophic outcomes justify large investments in AI safety and alignment research. We find that the optimizing representative individual would rationally allocate substantial resources to mitigate extinction risk; in some cases, she would prefer not to develop TAI at all. This result highlights that current global efforts in AI safety and alignment research are vastly insufficient relative to the scale and urgency of existential risks posed by TAI. Our findings therefore underscore the need for stronger safeguards to balance the potential economic benefits of TAI with the prevention of irreversible harm. Addressing these risks is crucial for steering technological progress toward sustainable human prosperity. 

**Abstract (ZH)**: 近期人工智能的发展对未来影响的预测引发了多样化的观点。一个核心关注点是转型人工智能（TAI）的潜在出现，最终能够在所有经济上有价值的任务中超越人类并完全自动化劳动力。讨论的场景从转型人工智能失控行为导致人类灭绝（“AI末日”）到前所未有的经济增长和 abundance（“后稀缺”）不等。然而，这些情景的可能性和影响依然高度不确定。本文整理了各种情景，并基于总体福利评估它们相关的存在风险和经济结果。我们的分析表明，即使是低概率的灾难性结果也证明了在人工智能安全和对齐研究上进行大量投资的必要性。我们发现，优化的代表性个体会理性地分配大量资源来降低灭绝风险；在某些情况下，她可能完全不想开发转型人工智能。这一结果突显了当前全球在人工智能安全和对齐研究上的努力与所面临的存在风险的规模和紧迫性相比还远远不足。因此，我们的发现强调了需要更强的保障措施来平衡转型人工智能可能带来的经济利益与防止不可逆伤害之间的关系。应对这些风险对于引导技术进步朝着可持续的人类繁荣方向至关重要。 

---
# AI Biases as Asymmetries: A Review to Guide Practice 

**Title (ZH)**: AI偏差作为不对称性：一项指导实践的综述 

**Authors**: Gabriella Waters, Phillip Honenberger  

**Link**: [PDF](https://arxiv.org/pdf/2503.07326)  

**Abstract**: The understanding of bias in AI is currently undergoing a revolution. Initially understood as errors or flaws, biases are increasingly recognized as integral to AI systems and sometimes preferable to less biased alternatives. In this paper, we review the reasons for this changed understanding and provide new guidance on two questions: First, how should we think about and measure biases in AI systems, consistent with the new understanding? Second, what kinds of bias in an AI system should we accept or even amplify, and what kinds should we minimize or eliminate, and why? The key to answering both questions, we argue, is to understand biases as "violations of a symmetry standard" (following Kelly). We distinguish three main types of asymmetry in AI systems-error biases, inequality biases, and process biases-and highlight places in the pipeline of AI development and application where bias of each type is likely to be good, bad, or inevitable. 

**Abstract (ZH)**: AI中的偏差理解正在经历一场革命：从错误或缺陷到被视为AI系统的核心要素，有时甚至比无偏差的替代方案更可取。在本文中，我们回顾了这一理解变化的原因，并就以下两个问题提供了新的指导：首先，我们应该如何根据新的理解来思考和衡量AI系统的偏差？其次，我们应该接受或放大哪些类型的偏差，哪些类型的偏差应该减少或消除，原因是什么？我们主张，回答这两个问题的关键是将偏差理解为“违反对称标准”的偏差（参考Kelly）。我们区分了AI系统中的三种主要不对称性类型：错误偏差、不平等偏差和过程偏差，并指出了在AI开发和应用管道中每种类型偏差可能是良好的、不良的或不可避免的地方。 

---
# Group-robust Sample Reweighting for Subpopulation Shifts via Influence Functions 

**Title (ZH)**: 基于影响函数的组稳健样本加权方法以应对亚群体偏移 

**Authors**: Rui Qiao, Zhaoxuan Wu, Jingtan Wang, Pang Wei Koh, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2503.07315)  

**Abstract**: Machine learning models often have uneven performance among subpopulations (a.k.a., groups) in the data distributions. This poses a significant challenge for the models to generalize when the proportions of the groups shift during deployment. To improve robustness to such shifts, existing approaches have developed strategies that train models or perform hyperparameter tuning using the group-labeled data to minimize the worst-case loss over groups. However, a non-trivial amount of high-quality labels is often required to obtain noticeable improvements. Given the costliness of the labels, we propose to adopt a different paradigm to enhance group label efficiency: utilizing the group-labeled data as a target set to optimize the weights of other group-unlabeled data. We introduce Group-robust Sample Reweighting (GSR), a two-stage approach that first learns the representations from group-unlabeled data, and then tinkers the model by iteratively retraining its last layer on the reweighted data using influence functions. Our GSR is theoretically sound, practically lightweight, and effective in improving the robustness to subpopulation shifts. In particular, GSR outperforms the previous state-of-the-art approaches that require the same amount or even more group labels. 

**Abstract (ZH)**: 机器学习模型在亚人群中（即，组）的表现通常不均衡。在部署过程中，如果组的比例发生变化，这会显著挑战模型的泛化能力。为提高对这些变化的鲁棒性，现有方法开发了利用组标签数据训练模型或进行超参数调优的策略，以最小化最坏情况的损失。然而，这通常需要大量的高质量标签才能获得显著改善。鉴于标签的成本，我们提出采用不同的范式来提升组标签的效率：利用组标签数据作为目标集，优化其他未标记组的数据的权重。我们引入了组鲁棒样本重加权（GSR），这是一种两阶段方法，首先从未标记组数据中学习表示，然后通过迭代地在重新加权的数据上重新训练模型的最后一层，使用影响函数来微调模型。GSR在理论上扎实、在实践上轻量且有效，能提高对亚人群转移的鲁棒性。特别是，GSR优于需要相同数量甚至更多组标签的最新最先进的方法。 

---
# VizTrust: A Visual Analytics Tool for Capturing User Trust Dynamics in Human-AI Communication 

**Title (ZH)**: VizTrust：一种捕获人类-人工智能通信中用户信任动态的可视化分析工具 

**Authors**: Xin Wang, Stephanie Tulk Jesso, Sadamori Kojaku, David M Neyens, Min Sun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.07279)  

**Abstract**: Trust plays a fundamental role in shaping the willingness of users to engage and collaborate with artificial intelligence (AI) systems. Yet, measuring user trust remains challenging due to its complex and dynamic nature. While traditional survey methods provide trust levels for long conversations, they fail to capture its dynamic evolution during ongoing interactions. Here, we present VizTrust, which addresses this challenge by introducing a real-time visual analytics tool that leverages a multi-agent collaboration system to capture and analyze user trust dynamics in human-agent communication. Built on established human-computer trust scales-competence, integrity, benevolence, and predictability-, VizTrust enables stakeholders to observe trust formation as it happens, identify patterns in trust development, and pinpoint specific interaction elements that influence trust. Our tool offers actionable insights into human-agent trust formation and evolution in real time through a dashboard, supporting the design of adaptive conversational agents that responds effectively to user trust signals. 

**Abstract (ZH)**: 信任在塑造用户参与和协作人工智能系统意愿方面发挥着基础性作用。然而，由于信任的复杂和动态性质，测量用户信任仍然颇具挑战。虽然传统的调查方法可以提供长时间对话的信任水平，但它们无法捕捉持续互动过程中信任动态演变的情况。在此，我们提出VizTrust，通过引入一个基于多代理协作系统的实时可视化分析工具来应对这一挑战，以捕获和分析人类-代理通信中的信任动态。基于已建立的人机信任尺度——能力、诚信、善意和可预测性，VizTrust使利益相关者能够实时观察信任的形成过程，识别信任发展的模式，并确定影响信任的具体交互要素。该工具通过仪表板提供实时的人机信任形成和演变的可操作洞察，支持设计出能够有效响应用户信任信号的自适应对话代理。 

---
# Federated Learning in NTNs: Design, Architecture and Challenges 

**Title (ZH)**: NTN中的联邦学习：设计、架构与挑战 

**Authors**: Amin Farajzadeh, Animesh Yadav, Halim Yanikomeroglu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07272)  

**Abstract**: Non-terrestrial networks (NTNs) are emerging as a core component of future 6G communication systems, providing global connectivity and supporting data-intensive applications. In this paper, we propose a distributed hierarchical federated learning (HFL) framework within the NTN architecture, leveraging a high altitude platform station (HAPS) constellation as intermediate distributed FL servers. Our framework integrates both low-Earth orbit (LEO) satellites and ground clients in the FL training process while utilizing geostationary orbit (GEO) and medium-Earth orbit (MEO) satellites as relays to exchange FL global models across other HAPS constellations worldwide, enabling seamless, global-scale learning. The proposed framework offers several key benefits: (i) enhanced privacy through the decentralization of the FL mechanism by leveraging the HAPS constellation, (ii) improved model accuracy and reduced training loss while balancing latency, (iii) increased scalability of FL systems through ubiquitous connectivity by utilizing MEO and GEO satellites, and (iv) the ability to use FL data, such as resource utilization metrics, to further optimize the NTN architecture from a network management perspective. A numerical study demonstrates the proposed framework's effectiveness, with improved model accuracy, reduced training loss, and efficient latency management. The article also includes a brief review of FL in NTNs and highlights key challenges and future research directions. 

**Abstract (ZH)**: NTN架构内的分布式层次联邦学习框架：利用高空平台站星座实现全球无缝学习 

---
# WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation 

**Title (ZH)**: WISE：一种基于世界知识的语义评估方法用于文本到图像生成 

**Authors**: Yuwei Niu, Munan Ning, Mengren Zheng, Bin Lin, Peng Jin, Jiaqi Liao, Kunpeng Ning, Bin Zhu, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07265)  

**Abstract**: Text-to-Image (T2I) models are capable of generating high-quality artistic creations and visual content. However, existing research and evaluation standards predominantly focus on image realism and shallow text-image alignment, lacking a comprehensive assessment of complex semantic understanding and world knowledge integration in text to image generation. To address this challenge, we propose $\textbf{WISE}$, the first benchmark specifically designed for $\textbf{W}$orld Knowledge-$\textbf{I}$nformed $\textbf{S}$emantic $\textbf{E}$valuation. WISE moves beyond simple word-pixel mapping by challenging models with 1000 meticulously crafted prompts across 25 sub-domains in cultural common sense, spatio-temporal reasoning, and natural science. To overcome the limitations of traditional CLIP metric, we introduce $\textbf{WiScore}$, a novel quantitative metric for assessing knowledge-image alignment. Through comprehensive testing of 20 models (10 dedicated T2I models and 10 unified multimodal models) using 1,000 structured prompts spanning 25 subdomains, our findings reveal significant limitations in their ability to effectively integrate and apply world knowledge during image generation, highlighting critical pathways for enhancing knowledge incorporation and application in next-generation T2I models. Code and data are available at this https URL. 

**Abstract (ZH)**: 基于世界知识的文本到图像语义评估 benchmarks（WISE）：超越简单的词-像素映射 

---
# Cross-Lingual IPA Contrastive Learning for Zero-Shot NER 

**Title (ZH)**: 跨语言IPA对比学习在零样本命名实体识别中的应用 

**Authors**: Jimin Sohn, David R. Mortensen  

**Link**: [PDF](https://arxiv.org/pdf/2503.07214)  

**Abstract**: Existing approaches to zero-shot Named Entity Recognition (NER) for low-resource languages have primarily relied on machine translation, whereas more recent methods have shifted focus to phonemic representation. Building upon this, we investigate how reducing the phonemic representation gap in IPA transcription between languages with similar phonetic characteristics enables models trained on high-resource languages to perform effectively on low-resource languages. In this work, we propose CONtrastive Learning with IPA (CONLIPA) dataset containing 10 English and high resource languages IPA pairs from 10 frequently used language families. We also propose a cross-lingual IPA Contrastive learning method (IPAC) using the CONLIPA dataset. Furthermore, our proposed dataset and methodology demonstrate a substantial average gain when compared to the best performing baseline. 

**Abstract (ZH)**: 基于音标表示的低资源语言零样本命名实体识别方法研究 

---
# DeFine: A Decomposed and Fine-Grained Annotated Dataset for Long-form Article Generation 

**Title (ZH)**: DeFine: 一种分解和细粒度标注的数据集，用于长文生成 

**Authors**: Ming Wang, Fang Wang, Minghao Hu, Li He, Haiyang Wang, Jun Zhang, Tianwei Yan, Li Li, Zhunchen Luo, Wei Luo, Xiaoying Bai, Guotong Geng  

**Link**: [PDF](https://arxiv.org/pdf/2503.07170)  

**Abstract**: Long-form article generation (LFAG) presents challenges such as maintaining logical consistency, comprehensive topic coverage, and narrative coherence across extended articles. Existing datasets often lack both the hierarchical structure and fine-grained annotation needed to effectively decompose tasks, resulting in shallow, disorganized article generation. To address these limitations, we introduce DeFine, a Decomposed and Fine-grained annotated dataset for long-form article generation. DeFine is characterized by its hierarchical decomposition strategy and the integration of domain-specific knowledge with multi-level annotations, ensuring granular control and enhanced depth in article generation. To construct the dataset, a multi-agent collaborative pipeline is proposed, which systematically segments the generation process into four parts: Data Miner, Cite Retreiver, Q&A Annotator and Data Cleaner. To validate the effectiveness of DeFine, we designed and tested three LFAG baselines: the web retrieval, the local retrieval, and the grounded reference. We fine-tuned the Qwen2-7b-Instruct model using the DeFine training dataset. The experimental results showed significant improvements in text quality, specifically in topic coverage, depth of information, and content fidelity. Our dataset publicly available to facilitate future research. 

**Abstract (ZH)**: 长文生成（LFAG）面临着保持逻辑一致性、全面的专题覆盖以及叙述连贯性的挑战。现有数据集往往缺乏有效分解任务所需的层级结构和细粒度标注，导致文章生成浅显且零散。为解决这些局限性，我们提出了一个分解和细粒度标注的数据集DeFine，用于长文生成。DeFine以其层级分解策略和领域特定知识与多层级标注的集成著称，确保了在文章生成中的粒度控制与深度增强。为构建该数据集，我们提出了一个多智能体协作管道，系统地将生成过程细分为四个部分：数据挖掘器、引文检索器、问答标注器和数据清理器。为了验证DeFine的有效性，我们设计并测试了三个LFAG基线模型：网页检索、本地检索和基于参考的模型。我们使用DeFine训练数据集微调了Qwen2-7b-Instruct模型。实验结果显示，在专题覆盖、信息深度和内容忠实度方面有显著改进。我们的数据集已公开，以促进未来的研究。 

---
# PTMs-TSCIL Pre-Trained Models Based Class-Incremental Learning 

**Title (ZH)**: PTMs-TSCIL 预训练模型导向的类别增量学习 

**Authors**: Yuanlong Wu, Mingxing Nie, Tao Zhu, Liming Chen, Huansheng Ning, Yaping Wan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07153)  

**Abstract**: Class-incremental learning (CIL) for time series data faces critical challenges in balancing stability against catastrophic forgetting and plasticity for new knowledge acquisition, particularly under real-world constraints where historical data access is restricted. While pre-trained models (PTMs) have shown promise in CIL for vision and NLP domains, their potential in time series class-incremental learning (TSCIL) remains underexplored due to the scarcity of large-scale time series pre-trained models. Prompted by the recent emergence of large-scale pre-trained models (PTMs) for time series data, we present the first exploration of PTM-based Time Series Class-Incremental Learning (TSCIL). Our approach leverages frozen PTM backbones coupled with incrementally tuning the shared adapter, preserving generalization capabilities while mitigating feature drift through knowledge distillation. Furthermore, we introduce a Feature Drift Compensation Network (DCN), designed with a novel two-stage training strategy to precisely model feature space transformations across incremental tasks. This allows for accurate projection of old class prototypes into the new feature space. By employing DCN-corrected prototypes, we effectively enhance the unified classifier retraining, mitigating model feature drift and alleviating catastrophic forgetting. Extensive experiments on five real-world datasets demonstrate state-of-the-art performance, with our method yielding final accuracy gains of 1.4%-6.1% across all datasets compared to existing PTM-based approaches. Our work establishes a new paradigm for TSCIL, providing insights into stability-plasticity optimization for continual learning systems. 

**Abstract (ZH)**: 基于预训练模型的时间序列类增量学习（PTM-based Time Series Class-Incremental Learning） 

---
# MRCEval: A Comprehensive, Challenging and Accessible Machine Reading Comprehension Benchmark 

**Title (ZH)**: MRCEval: 一个全面、有挑战性和易于访问的机器阅读理解基准 

**Authors**: Shengkun Ma, Hao Peng, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07144)  

**Abstract**: Machine Reading Comprehension (MRC) is an essential task in evaluating natural language understanding. Existing MRC datasets primarily assess specific aspects of reading comprehension (RC), lacking a comprehensive MRC benchmark. To fill this gap, we first introduce a novel taxonomy that categorizes the key capabilities required for RC. Based on this taxonomy, we construct MRCEval, an MRC benchmark that leverages advanced Large Language Models (LLMs) as both sample generators and selection judges. MRCEval is a comprehensive, challenging and accessible benchmark designed to assess the RC capabilities of LLMs thoroughly, covering 13 distinct RC skills with a total of 2.1K high-quality multi-choice questions. We perform an extensive evaluation of 28 widely used open-source and proprietary models, highlighting that MRC continues to present significant challenges even in the era of LLMs. 

**Abstract (ZH)**: 机器阅读理解（MRC）是评估自然语言理解能力的重要任务。现有的MRC数据集主要评估阅读理解（RC）的特定方面，缺乏全面的MRC基准。为填补这一空白，我们首先引入了一种新型分类法，将关键的RC能力进行分类。基于这种分类法，我们构建了MRCEval，这是一个利用高级大型语言模型（LLMs）作为样本生成器和选择裁判的MRC基准。MRCEval是一个全面、具有挑战性和易访问的基准，旨在全面评估LLMs的RC能力，共涵盖13种不同的RC技能，总计2100个多选题。我们对28个广泛使用的开源和专有模型进行了广泛评估，突显了即使在LLM时代，MRC仍存在重大挑战。 

---
# A Comprehensive Survey of Mixture-of-Experts: Algorithms, Theory, and Applications 

**Title (ZH)**: 专家混合模型综述：算法、理论与应用 

**Authors**: Siyuan Mu, Sen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.07137)  

**Abstract**: Artificial intelligence (AI) has achieved astonishing successes in many domains, especially with the recent breakthroughs in the development of foundational large models. These large models, leveraging their extensive training data, provide versatile solutions for a wide range of downstream tasks. However, as modern datasets become increasingly diverse and complex, the development of large AI models faces two major challenges: (1) the enormous consumption of computational resources and deployment difficulties, and (2) the difficulty in fitting heterogeneous and complex data, which limits the usability of the models. Mixture of Experts (MoE) models has recently attracted much attention in addressing these challenges, by dynamically selecting and activating the most relevant sub-models to process input data. It has been shown that MoEs can significantly improve model performance and efficiency with fewer resources, particularly excelling in handling large-scale, multimodal data. Given the tremendous potential MoE has demonstrated across various domains, it is urgent to provide a comprehensive summary of recent advancements of MoEs in many important fields. Existing surveys on MoE have their limitations, e.g., being outdated or lacking discussion on certain key areas, and we aim to address these gaps. In this paper, we first introduce the basic design of MoE, including gating functions, expert networks, routing mechanisms, training strategies, and system design. We then explore the algorithm design of MoE in important machine learning paradigms such as continual learning, meta-learning, multi-task learning, and reinforcement learning. Additionally, we summarize theoretical studies aimed at understanding MoE and review its applications in computer vision and natural language processing. Finally, we discuss promising future research directions. 

**Abstract (ZH)**: 人工智能（AI）在许多领域取得了惊人的成就，尤其是随着基础大型模型开发的突破性进展。这些大型模型利用其大量的训练数据，为广泛下游任务提供了多功能的解决方案。然而，随着现代数据集越来越多元化和复杂化，大型AI模型的开发面临两大挑战：（1）巨大的计算资源消耗和部署困难，以及（2）难以适应异构和复杂数据，这限制了模型的应用。专家混合模型（MoE）最近因其通过动态选择和激活最相关的子模型来处理输入数据而受到广泛关注，已被证明能显著提高模型性能和效率，尤其是在处理大规模、多模态数据方面尤为出色。鉴于MoE在各个领域展示出的巨大潜力，迫切需要对其在许多重要领域的最新进展进行全面总结。现有MoE综述存在局限性，如内容过时或缺乏对某些关键领域的讨论，我们致力于弥补这些不足。在本文中，我们首先介绍MoE的基本设计，包括门控函数、专家网络、路由机制、训练策略和系统设计。接着，我们探讨了MoE在连续学习、元学习、多任务学习和强化学习等重要机器学习范式中的算法设计。此外，我们总结了旨在理解MoE的理论研究，并回顾了MoE在计算机视觉和自然语言处理中的应用。最后，我们讨论了未来研究的有前景的方向。 

---
# ASTRA: A Negotiation Agent with Adaptive and Strategic Reasoning through Action in Dynamic Offer Optimization 

**Title (ZH)**: ASTRA：一种在动态报价优化中具备适应性和战略推理的谈判代理 

**Authors**: Deuksin Kwon, Jiwon Hae, Emma Clift, Daniel Shamsoddini, Jonathan Gratch, Gale M. Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2503.07129)  

**Abstract**: Negotiation requires dynamically balancing self-interest and cooperation to maximize one's own utility. Yet, existing agents struggle due to bounded rationality in human data, low adaptability to counterpart behavior, and limited strategic reasoning. To address this, we introduce principle-driven negotiation agents, powered by ASTRA, a novel framework for turn-level offer optimization grounded in two core principles: opponent modeling and Tit-for-Tat reciprocity. ASTRA operates in three stages: (1) interpreting counterpart behavior, (2) optimizing counteroffers via a linear programming (LP) solver, and (3) selecting offers based on negotiation tactics and the partner's acceptance probability. Through simulations and human evaluations, our agent effectively adapts to an opponent's shifting stance and achieves favorable outcomes through enhanced adaptability and strategic reasoning. Beyond improving negotiation performance, it also serves as a powerful coaching tool, offering interpretable strategic feedback and optimal offer recommendations. 

**Abstract (ZH)**: 原则驱动的谈判代理：基于ASTRA的回合级报价优化框架 

---
# A LSTM-Transformer Model for pulsation control of pVADs 

**Title (ZH)**: 一种用于pVADs脉动控制的LSTM-Transformer模型 

**Authors**: Chaoran E, Chenghan Chen, Yuyang Shi, Haiyun Wang, Peixin Hua, Xiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07110)  

**Abstract**: Methods: A method of the pulsation for a pVAD is proposed (AP-pVAD Model). AP-pVAD Model consists of two parts: NPQ Model and LSTM-Transformer Model. (1)The NPQ Model determines the mathematical relationship between motor speed, pressure, and flow rate for the pVAD. (2)The Attention module of Transformer neural network is integrated into the LSTM neural network to form the new LSTM-Transformer Model to predict the pulsation time characteristic points for adjusting the motor speed of the pVAD. Results: The AP-pVAD Model is validated in three hydraulic experiments and an animal experiment. (1)The pressure provided by pVAD calculated with the NPQ Model has a maximum error of only 2.15 mmHg compared to the expected values. (2)The pulsation time characteristic points predicted by the LSTM-Transformer Model shows a maximum prediction error of 1.78ms, which is significantly lower than other methods. (3)The in-vivo test of pVAD in animal experiment has significant improvements in aortic pressure. Animals survive for over 27 hours after the initiation of pVAD operation. Conclusion: (1)For a given pVAD, motor speed has a linear relationship with pressure and a quadratic relationship with flow. (2)Deep learning can be used to predict pulsation characteristic time points, with the LSTM-Transformer Model demonstrating minimal prediction error and better robust performance under conditions of limited dataset sizes, elevated noise levels, and diverse hyperparameter combinations, demonstrating its feasibility and effectiveness. 

**Abstract (ZH)**: 方法: 提出了一种用于血液泵助装置的脉动方法（AP-pVAD模型）。AP-pVAD模型由两部分组成：NPQ模型和LSTM-Transformer模型。(1)NPQ模型确定了血液泵助装置的电机速度、压力和流量之间的数学关系。(2)LSTM神经网络中的注意力模块被集成到Transformer神经网络中，形成新的LSTM-Transformer模型，以预测脉动时间特征点，进而调整血液泵助装置的电机速度。结果: AP-pVAD模型在三个液压实验和一个动物实验中得到验证。(1)使用NPQ模型计算的血液泵助装置提供的压力与预期值的最大误差仅为2.15 mmHg。(2)LSTM-Transformer模型预测的脉动时间特征点的最大预测误差为1.78ms，显著低于其他方法。(3)动物实验中血液泵助装置的体内测试显示了显著的主动脉压力改善，在启动血液泵助装置后，动物存活超过27小时。结论: (1)对于给定的血液泵助装置，电机速度与压力呈线性关系，与流量呈二次关系。(2)深度学习可用于预测脉动时间特征点，LSTM-Transformer模型在数据集较小、噪声水平较高和超参数组合多样化的情况下，显示出最小的预测误差和更好的稳健性能，证明了其可行性和有效性。 

---
# An Experience Report on Regression-Free Repair of Deep Neural Network Model 

**Title (ZH)**: 深度神经网络模型无回归修复经验报告 

**Authors**: Takao Nakagawa, Susumu Tokumoto, Shogo Tokui, Fuyuki Ishikawa  

**Link**: [PDF](https://arxiv.org/pdf/2503.07079)  

**Abstract**: Systems based on Deep Neural Networks (DNNs) are increasingly being used in industry. In the process of system operation, DNNs need to be updated in order to improve their performance. When updating DNNs, systems used in companies that require high reliability must have as few regressions as possible. Since the update of DNNs has a data-driven nature, it is difficult to suppress regressions as expected by developers. This paper identifies the requirements for DNN updating in industry and presents a case study using techniques to meet those requirements. In the case study, we worked on satisfying the requirement to update models trained on car images collected in Fujitsu assuming security applications without regression for a specific class. We were able to suppress regression by customizing the objective function based on NeuRecover, a DNN repair technique. Moreover, we discuss some of the challenges identified in the case study. 

**Abstract (ZH)**: 基于深度神经网络的系统在工业中越来越广泛地被使用。在系统运行过程中，为了提高性能需要对深度神经网络（DNNs）进行更新。在进行DNN更新时，对于需要高可靠性的企业系统应尽量减少回归现象。由于DNN更新具有数据驱动的特性，开发人员很难按预期抑制回归现象。本文识别了工业中DNN更新的要求，并利用技术手段满足这些要求。在案例研究中，我们针对假设用于安全应用的汽车图像训练模型，制定了不针对特定类别的回归现象的更新要求，并通过基于NeuRecover的定制目标函数实现了回归现象的抑制。此外，本文还讨论了案例研究中遇到的一些挑战。 

---
# PIED: Physics-Informed Experimental Design for Inverse Problems 

**Title (ZH)**: PIED: 物理启发的实验设计用于逆问题 

**Authors**: Apivich Hemachandra, Gregory Kang Ruey Lau, See-Kiong Ng, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2503.07070)  

**Abstract**: In many science and engineering settings, system dynamics are characterized by governing PDEs, and a major challenge is to solve inverse problems (IPs) where unknown PDE parameters are inferred based on observational data gathered under limited budget. Due to the high costs of setting up and running experiments, experimental design (ED) is often done with the help of PDE simulations to optimize for the most informative design parameters to solve such IPs, prior to actual data collection. This process of optimizing design parameters is especially critical when the budget and other practical constraints make it infeasible to adjust the design parameters between trials during the experiments. However, existing experimental design (ED) methods tend to require sequential and frequent design parameter adjustments between trials. Furthermore, they also have significant computational bottlenecks due to the need for complex numerical simulations for PDEs, and do not exploit the advantages provided by physics informed neural networks (PINNs), such as its meshless solutions, differentiability, and amortized training. This work presents PIED, the first ED framework that makes use of PINNs in a fully differentiable architecture to perform continuous optimization of design parameters for IPs for one-shot deployments. PIED overcomes existing methods' computational bottlenecks through parallelized computation and meta-learning of PINN parameter initialization, and proposes novel methods to effectively take into account PINN training dynamics in optimizing the ED parameters. Through experiments based on noisy simulated data and even real world experimental data, we empirically show that given limited observation budget, PIED significantly outperforms existing ED methods in solving IPs, including challenging settings where the inverse parameters are unknown functions rather than just finite-dimensional. 

**Abstract (ZH)**: 基于PINNs的全程连续优化实验设计框架PIED 

---
# Generative method for aerodynamic optimization based on classifier-free guided denoising diffusion probabilistic model 

**Title (ZH)**: 基于分类器-free 引导去噪扩散概率模型的气动优化生成方法 

**Authors**: Shisong Deng, Qiang Zhang, Zhengyang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.07056)  

**Abstract**: Inverse design approach, which directly generates optimal aerodynamic shape with neural network models to meet designated performance targets, has drawn enormous attention. However, the current state-of-the-art inverse design approach for airfoils, which is based on generative adversarial network, demonstrates insufficient precision in its generating and training processes and struggles to reveal the coupling relationship among specified performance indicators. To address these issues, the airfoil inverse design framework based on the classifier-free guided denoising diffusion probabilistic model (CDDPM) is proposed innovatively in this paper. First, the CDDPM can effectively capture the correlations among specific performance indicators and, by adjusting the classifier-free guide coefficient, generate corresponding upper and lower surface pressure coefficient distributions based on designated pressure features. These distributions are then accurately translated into airfoil geometries through a mapping model. Experimental results using classical transonic airfoils as examples show that the inverse design based on CDDPM can generate a variety of pressure coefficient distributions, which enriches the diversity of design results. Compared with current state-of-the-art Wasserstein generative adversarial network methods, CDDPM achieves a 33.6% precision improvement in airfoil generating tasks. Moreover, a practical method to readjust each performance indicator value is proposed based on global optimization algorithm in conjunction with active learning strategy, aiming to provide rational value combination of performance indicators for the inverse design framework. This work is not only suitable for the airfoils design, but also has the capability to apply to optimization process of general product parts targeting selected performance indicators. 

**Abstract (ZH)**: 基于分类器免费引导去噪扩散概率模型的翼型逆设计框架 

---
# Weak Supervision for Improved Precision in Search Systems 

**Title (ZH)**: 弱监督以提高搜索系统的精度 

**Authors**: Sriram Vasudevan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07025)  

**Abstract**: Labeled datasets are essential for modern search engines, which increasingly rely on supervised learning methods like Learning to Rank and massive amounts of data to power deep learning models. However, creating these datasets is both time-consuming and costly, leading to the common use of user click and activity logs as proxies for relevance. In this paper, we present a weak supervision approach to infer the quality of query-document pairs and apply it within a Learning to Rank framework to enhance the precision of a large-scale search system. 

**Abstract (ZH)**: 标记数据集对于现代搜索引擎至关重要，这些搜索引擎越来越多地依赖于如学习排名等监督学习方法和大量数据来驱动深度学习模型。然而，创建这些数据集既耗时又昂贵，因此常用用户点击和活动日志作为相关性的代理。在本文中，我们提出了一种弱监督方法来推断查询-文档对的质量，并将其应用于学习排名框架中，以提高大规模搜索引擎的精度。 

---
# NukesFormers: Unpaired Hyperspectral Image Generation with Non-Uniform Domain Alignment 

**Title (ZH)**: NukesFormers: 无配对高光谱图像生成与非均匀域对齐 

**Authors**: Jiaojiao Li, Shiyao Duan, Haitao XU, Rui Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.07004)  

**Abstract**: The inherent difficulty in acquiring accurately co-registered RGB-hyperspectral image (HSI) pairs has significantly impeded the practical deployment of current data-driven Hyperspectral Image Generation (HIG) networks in engineering applications. Gleichzeitig, the ill-posed nature of the aligning constraints, compounded with the complexities of mining cross-domain features, also hinders the advancement of unpaired HIG (UnHIG) tasks. In this paper, we conquer these challenges by modeling the UnHIG to range space interaction and compensations of null space through Range-Null Space Decomposition (RND) methodology. Specifically, the introduced contrastive learning effectively aligns the geometric and spectral distributions of unpaired data by building the interaction of range space, considering the consistent feature in degradation process. Following this, we map the frequency representations of dual-domain input and thoroughly mining the null space, like degraded and high-frequency components, through the proposed Non-uniform Kolmogorov-Arnold Networks. Extensive comparative experiments demonstrate that it establishes a new benchmark in UnHIG. 

**Abstract (ZH)**: 非配对高光谱图像生成中的固有难度显著阻碍了当前数据驱动的高光谱图像生成网络在工程应用中的实际部署。同时，对齐约束的不良性质与跨域特征提取的复杂性也阻碍了无配对高光谱图像生成（UnHIG）任务的发展。本文通过范围空间和零空间交互与补偿的范围-零空间分解（RND）方法克服了这些挑战。具体而言，引入的对比学习通过建立范围空间的交互来有效对齐无配对数据的几何和光谱分布，考虑到退化过程中的一致特征。随后，我们通过所提出的非均匀柯尔莫哥洛夫-阿诺尔德网络映射双域输入的频率表示，并全面挖掘零空间，如退化和高频成分。广泛的对比实验表明，它在无配对高光谱图像生成中建立了新的基准。 

---
# Multi-Behavior Recommender Systems: A Survey 

**Title (ZH)**: 多行为推荐系统：文献综述 

**Authors**: Kyungho Kim, Sunwoo Kim, Geon Lee, Jinhong Jung, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2503.06963)  

**Abstract**: Traditional recommender systems primarily rely on a single type of user-item interaction, such as item purchases or ratings, to predict user preferences. However, in real-world scenarios, users engage in a variety of behaviors, such as clicking on items or adding them to carts, offering richer insights into their interests. Multi-behavior recommender systems leverage these diverse interactions to enhance recommendation quality, and research on this topic has grown rapidly in recent years. This survey provides a timely review of multi-behavior recommender systems, focusing on three key steps: (1) Data Modeling: representing multi-behaviors at the input level, (2) Encoding: transforming these inputs into vector representations (i.e., embeddings), and (3) Training: optimizing machine-learning models. We systematically categorize existing multi-behavior recommender systems based on the commonalities and differences in their approaches across the above steps. Additionally, we discuss promising future directions for advancing multi-behavior recommender systems. 

**Abstract (ZH)**: 传统推荐系统主要依赖于单一类型的用户-项交互，如购买或评分，以预测用户偏好。然而，在现实场景中，用户表现出多种行为，如点击物品或将物品加入购物车，这些行为提供了更多关于用户兴趣的洞察。多行为推荐系统通过利用这些多样化的交互来提升推荐质量，近年来该领域的研究迅速增长。本文提供了一篇关于多行为推荐系统的及时综述，重点关注三个关键步骤：(1) 数据建模：在输入层面表示多种行为，(2) 编码：将这些输入转换为向量表示（即嵌入），以及(3) 训练：优化机器学习模型。我们系统地根据上述步骤中的共性和差异对现有的多行为推荐系统进行了分类。此外，我们还讨论了推进多行为推荐系统发展的有希望的新方向。 

---
# Capture Global Feature Statistics for One-Shot Federated Learning 

**Title (ZH)**: 捕获全局特征统计用于一-shot联邦学习 

**Authors**: Zenghao Guan, Yucan Zhou, Xiaoyan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06962)  

**Abstract**: Traditional Federated Learning (FL) necessitates numerous rounds of communication between the server and clients, posing significant challenges including high communication costs, connection drop risks and susceptibility to privacy attacks. One-shot FL has become a compelling learning paradigm to overcome above drawbacks by enabling the training of a global server model via a single communication round. However, existing one-shot FL methods suffer from expensive computation cost on the server or clients and cannot deal with non-IID (Independent and Identically Distributed) data stably and effectively. To address these challenges, this paper proposes FedCGS, a novel Federated learning algorithm that Capture Global feature Statistics leveraging pre-trained models. With global feature statistics, we achieve training-free and heterogeneity-resistant one-shot FL. Furthermore, we extend its application to personalization scenario, where clients only need execute one extra communication round with server to download global statistics. Extensive experimental results demonstrate the effectiveness of our methods across diverse data heterogeneity settings. Code is available at this https URL. 

**Abstract (ZH)**: 传统的联邦学习（FL）需要服务器与客户端进行多轮通信，这带来了高通信成本、连接中断风险及隐私攻击的脆弱性挑战。单轮次联邦学习已成为克服上述问题的一种有吸引力的学习范式，通过单一通信轮次即可训练全局服务器模型。然而，现有的单轮次联邦学习方法在服务器或客户端的计算成本高昂，并且在处理非IID数据时表现不稳定和不有效。为应对这些挑战，本文提出了一种名为FedCGS的新颖联邦学习算法，该算法利用预训练模型捕获全局特征统计。借助全局特征统计，我们实现了无需训练且具有异构性抵抗能力的单轮次联邦学习。此外，我们将其应用扩展到个性化场景，客户端仅需额外与服务器执行一轮通信以下载全局统计。广泛的经验结果证明了我们的方法在多种数据异构性设置下的有效性。代码详见这个链接。 

---
# Interactive Medical Image Analysis with Concept-based Similarity Reasoning 

**Title (ZH)**: 基于概念相似性推理的交互式医学图像分析 

**Authors**: Ta Duc Huy, Sen Kim Tran, Phan Nguyen, Nguyen Hoang Tran, Tran Bao Sam, Anton van den Hengel, Zhibin Liao, Johan W. Verjans, Minh-Son To, Vu Minh Hieu Phan  

**Link**: [PDF](https://arxiv.org/pdf/2503.06873)  

**Abstract**: The ability to interpret and intervene model decisions is important for the adoption of computer-aided diagnosis methods in clinical workflows. Recent concept-based methods link the model predictions with interpretable concepts and modify their activation scores to interact with the model. However, these concepts are at the image level, which hinders the model from pinpointing the exact patches the concepts are activated. Alternatively, prototype-based methods learn representations from training image patches and compare these with test image patches, using the similarity scores for final class prediction. However, interpreting the underlying concepts of these patches can be challenging and often necessitates post-hoc guesswork. To address this issue, this paper introduces the novel Concept-based Similarity Reasoning network (CSR), which offers (i) patch-level prototype with intrinsic concept interpretation, and (ii) spatial interactivity. First, the proposed CSR provides localized explanation by grounding prototypes of each concept on image regions. Second, our model introduces novel spatial-level interaction, allowing doctors to engage directly with specific image areas, making it an intuitive and transparent tool for medical imaging. CSR improves upon prior state-of-the-art interpretable methods by up to 4.5\% across three biomedical datasets. Our code is released at this https URL. 

**Abstract (ZH)**: 基于概念的相似性推理网络：局部解释与空间交互在医学影像诊断中的应用 

---
# Lost-in-the-Middle in Long-Text Generation: Synthetic Dataset, Evaluation Framework, and Mitigation 

**Title (ZH)**: 长文本生成中的中途迷失：合成数据集、评估框架及缓解方法 

**Authors**: Junhao Zhang, Richong Zhang, Fanshuang Kong, Ziyang Miao, Yanhan Ye, Yaowei Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.06868)  

**Abstract**: Existing long-text generation methods primarily concentrate on producing lengthy texts from short inputs, neglecting the long-input and long-output tasks. Such tasks have numerous practical applications while lacking available benchmarks. Moreover, as the input grows in length, existing methods inevitably encounter the "lost-in-the-middle" phenomenon. In this paper, we first introduce a Long Input and Output Benchmark (LongInOutBench), including a synthetic dataset and a comprehensive evaluation framework, addressing the challenge of the missing benchmark. We then develop the Retrieval-Augmented Long-Text Writer (RAL-Writer), which retrieves and restates important yet overlooked content, mitigating the "lost-in-the-middle" issue by constructing explicit prompts. We finally employ the proposed LongInOutBench to evaluate our RAL-Writer against comparable baselines, and the results demonstrate the effectiveness of our approach. Our code has been released at this https URL. 

**Abstract (ZH)**: 现有的长文本生成方法主要关注从短输入生成长文本，忽视了长输入和长输出任务。这类任务具有广泛的实际应用价值，但缺乏相应的基准数据。此外，随着输入文本的长度增加，现有方法不可避免地会遇到“中间内容丢失”现象。在本文中，我们首先引入了一个长输入和长输出基准(LongInOutBench)，包括合成数据集和综合评估框架，以应对缺失基准的挑战。随后，我们开发了检索增强长文本作家(RAL-Writer)，该方法检索并重述重要但被忽视的内容，通过构建显式的提示来缓解“中间内容丢失”问题。最后，我们使用提出的LongInOutBench对我们的RAL-Writer与现有基准进行评估，结果表明了我们方法的有效性。我们的代码已发布在此 <https://> 地址。 

---
# Enhanced Multi-Tuple Extraction for Alloys: Integrating Pointer Networks and Augmented Attention 

**Title (ZH)**: 增强的合金多组元提取：集成指针网络和增强注意力机制 

**Authors**: Mengzhe Hei, Zhouran Zhang, Qingbao Liu, Yan Pan, Xiang Zhao, Yongqian Peng, Yicong Ye, Xin Zhang, Shuxin Bai  

**Link**: [PDF](https://arxiv.org/pdf/2503.06861)  

**Abstract**: Extracting high-quality structured information from scientific literature is crucial for advancing material design through data-driven methods. Despite the considerable research in natural language processing for dataset extraction, effective approaches for multi-tuple extraction in scientific literature remain scarce due to the complex interrelations of tuples and contextual ambiguities. In the study, we illustrate the multi-tuple extraction of mechanical properties from multi-principal-element alloys and presents a novel framework that combines an entity extraction model based on MatSciBERT with pointer networks and an allocation model utilizing inter- and intra-entity attention. Our rigorous experiments on tuple extraction demonstrate impressive F1 scores of 0.963, 0.947, 0.848, and 0.753 across datasets with 1, 2, 3, and 4 tuples, confirming the effectiveness of the model. Furthermore, an F1 score of 0.854 was achieved on a randomly curated dataset. These results highlight the model's capacity to deliver precise and structured information, offering a robust alternative to large language models and equipping researchers with essential data for fostering data-driven innovations. 

**Abstract (ZH)**: 从科学文献中提取高质量结构化信息对于通过数据驱动方法推进材料设计至关重要。尽管在数据集提取的自然语言处理研究方面进展显著，但由于元组间的复杂关联性和上下文歧义，科学文献中的多元组提取有效方法仍较为稀缺。在本研究中，我们展示了多元组提取在多主元素合金中力学性能的提取，并提出了一种结合基于MatSciBERT的实体提取模型、指针网络以及利用实体间和实体内注意力的分配模型的新型框架。我们在元组提取的严格实验中展示了在包含1、2、3和4个元组的数据集上分别取得了0.963、0.947、0.848和0.753的F1分数，证实了该模型的有效性。此外，我们还在一个随机整理的数据集上取得了0.854的F1分数。这些结果突显了该模型在提供精准和结构化信息方面的能力，为其提供了与大型语言模型相比的稳健替代方案，同时也为研究人员提供了驱动数据创新所需的关键数据。 

---
# AttFC: Attention Fully-Connected Layer for Large-Scale Face Recognition with One GPU 

**Title (ZH)**: AttFC：一种用于单GPU大规模人脸识别的注意力全连接层 

**Authors**: Zhuowen Zheng, Yain-Whar Si, Xiaochen Yuan, Junwei Duan, Ke Wang, Xiaofan Li, Xinyuan Zhang, Xueyuan Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.06839)  

**Abstract**: Nowadays, with the advancement of deep neural networks (DNNs) and the availability of large-scale datasets, the face recognition (FR) model has achieved exceptional performance. However, since the parameter magnitude of the fully connected (FC) layer directly depends on the number of identities in the dataset. If training the FR model on large-scale datasets, the size of the model parameter will be excessively huge, leading to substantial demand for computational resources, such as time and memory. This paper proposes the attention fully connected (AttFC) layer, which could significantly reduce computational resources. AttFC employs an attention loader to generate the generative class center (GCC), and dynamically store the class center with Dynamic Class Container (DCC). DCC only stores a small subset of all class centers in FC, thus its parameter count is substantially less than the FC layer. Also, training face recognition models on large-scale datasets with one GPU often encounter out-of-memory (OOM) issues. AttFC overcomes this and achieves comparable performance to state-of-the-art methods. 

**Abstract (ZH)**: 基于注意力机制的全连接层在大规模人脸识别人脸识别模型中的应用：大幅降低计算资源需求 

---
# Can Proof Assistants Verify Multi-Agent Systems? 

**Title (ZH)**: 多Agent系统能否被证明助手验证？ 

**Authors**: Julian Alfredo Mendez, Timotheus Kampik  

**Link**: [PDF](https://arxiv.org/pdf/2503.06812)  

**Abstract**: This paper presents the Soda language for verifying multi-agent systems. Soda is a high-level functional and object-oriented language that supports the compilation of its code not only to Scala, a strongly statically typed high-level programming language, but also to Lean, a proof assistant and programming language. Given these capabilities, Soda can implement multi-agent systems, or parts thereof, that can then be integrated into a mainstream software ecosystem on the one hand and formally verified with state-of-the-art tools on the other hand. We provide a brief and informal introduction to Soda and the aforementioned interoperability capabilities, as well as a simple demonstration of how interaction protocols can be designed and verified with Soda. In the course of the demonstration, we highlight challenges with respect to real-world applicability. 

**Abstract (ZH)**: 本文提出了Soda语言用于验证多代理系统。Soda是一种高层函数式和面向对象语言，支持将其代码编译为Scala（一种强静态类型高层编程语言）以及Lean（一种证明助手兼编程语言）。凭借这些能力，Soda可以实现可集成到主流软件生态系统中的多代理系统，或其部分系统，并且可以用最先进的工具对其进行形式化验证。我们提供了一个简要且非正式的Soda语言和上述互操作性能力的介绍，并通过一个简单的示例展示了如何使用Soda设计和验证交互协议。在演示过程中，我们指出了实际应用中的挑战。 

---
# Mitigating Preference Hacking in Policy Optimization with Pessimism 

**Title (ZH)**: 使用悲观主义缓解政策优化中的偏好劫持 

**Authors**: Dhawal Gupta, Adam Fisch, Christoph Dann, Alekh Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2503.06810)  

**Abstract**: This work tackles the problem of overoptimization in reinforcement learning from human feedback (RLHF), a prevalent technique for aligning models with human preferences. RLHF relies on reward or preference models trained on \emph{fixed preference datasets}, and these models are unreliable when evaluated outside the support of this preference data, leading to the common reward or preference hacking phenomenon. We propose novel, pessimistic objectives for RLHF which are provably robust to overoptimization through the use of pessimism in the face of uncertainty, and design practical algorithms, P3O and PRPO, to optimize these objectives. Our approach is derived for the general preference optimization setting, but can be used with reward models as well. We evaluate P3O and PRPO on the tasks of fine-tuning language models for document summarization and creating helpful assistants, demonstrating remarkable resilience to overoptimization. 

**Abstract (ZH)**: 本文解决了强化学习从人类反馈中过度优化的问题（RLHF），这是一种使模型与人类偏好一致的常见技术。RLHF依赖于在固定偏好评价数据集上训练的奖励或偏好模型，这些模型在评价超出该偏好数据支持范围时可靠性较低，导致了常见的奖励或偏好篡改现象。我们提出了新型的悲观目标函数，通过在不确定性面前保持悲观态度，这些目标函数在理论上能够抵抗过度优化，并设计了实用的算法P3O和PRPO来优化这些目标函数。我们的方法适用于一般偏好优化设置，也可以与奖励模型一起使用。我们在文档摘要语言模型微调和创建有助手的任务上评估了P3O和PRPO，展示了它们对过度优化的出色抗御能力。 

---
# Actionable AI: Enabling Non Experts to Understand and Configure AI Systems 

**Title (ZH)**: 可操作的AI：使非专家能够理解并配置AI系统 

**Authors**: Cécile Boulard, Sruthi Viswanathan, Wanda Fey, Thierry Jacquin  

**Link**: [PDF](https://arxiv.org/pdf/2503.06803)  

**Abstract**: Interaction between humans and AI systems raises the question of how people understand AI systems. This has been addressed with explainable AI, the interpretability arising from users' domain expertise, or collaborating with AI in a stable environment. In the absence of these elements, we discuss designing Actionable AI, which allows non-experts to configure black-box agents. In this paper, we experiment with an AI-powered cartpole game and observe 22 pairs of participants to configure it via direct manipulation. Our findings suggest that, in uncertain conditions, non-experts were able to achieve good levels of performance. By influencing the behaviour of the agent, they exhibited an operational understanding of it, which proved sufficient to reach their goals. Based on this, we derive implications for designing Actionable AI systems. In conclusion, we propose Actionable AI as a way to open access to AI-based agents, giving end users the agency to influence such agents towards their own goals. 

**Abstract (ZH)**: 人类与AI系统互动引发了人们对如何理解AI系统的疑问。这个问题通过可解释AI、用户专业知识带来的可理解性或在稳定环境中与AI协作来解决。在缺乏这些要素的情况下，我们讨论了设计可操作AI的做法，允许非专家配置黑盒代理。在本文中，我们通过直接操作实验了一个基于AI的杆车游戏，并观察了22对参与者配置该游戏的过程。我们的研究发现，在不确定性条件下，非专家能够达到良好的性能水平。通过影响代理的行为，他们表现出了对其的操作理解，这种理解足够使其达到目标。基于此，我们推导出了设计可操作AI系统的建议。总之，我们提出可操作AI作为一种使基于AI的代理对终端用户开放的方式，赋予用户影响这些代理以实现自身目标的能力。 

---
# Characterizing Learning in Spiking Neural Networks with Astrocyte-Like Units 

**Title (ZH)**: 具有星形胶质细胞样单元的脉冲神经网络中的学习表征 

**Authors**: Christopher S. Yang, Sylvester J. Gates III, Dulara De Zoysa, Jaehoon Choe, Wolfgang Losert, Corey B. Hart  

**Link**: [PDF](https://arxiv.org/pdf/2503.06798)  

**Abstract**: Traditional artificial neural networks take inspiration from biological networks, using layers of neuron-like nodes to pass information for processing. More realistic models include spiking in the neural network, capturing the electrical characteristics more closely. However, a large proportion of brain cells are of the glial cell type, in particular astrocytes which have been suggested to play a role in performing computations. Here, we introduce a modified spiking neural network model with added astrocyte-like units in a neural network and asses their impact on learning. We implement the network as a liquid state machine and task the network with performing a chaotic time-series prediction task. We varied the number and ratio of neuron-like and astrocyte-like units in the network to examine the latter units effect on learning. We show that the combination of neurons and astrocytes together, as opposed to neural- and astrocyte-only networks, are critical for driving learning. Interestingly, we found that the highest learning rate was achieved when the ratio between astrocyte-like and neuron-like units was roughly 2 to 1, mirroring some estimates of the ratio of biological astrocytes to neurons. Our results demonstrate that incorporating astrocyte-like units which represent information across longer timescales can alter the learning rates of neural networks, and the proportion of astrocytes to neurons should be tuned appropriately to a given task. 

**Abstract (ZH)**: 包含类似星形胶质细胞单元的修改后脉冲神经网络对学习的影响研究 

---
# Effectiveness of Zero-shot-CoT in Japanese Prompts 

**Title (ZH)**: 零样本-CoT在日本提示中的有效性 

**Authors**: Shusuke Takayama, Ian Frank  

**Link**: [PDF](https://arxiv.org/pdf/2503.06765)  

**Abstract**: We compare the effectiveness of zero-shot Chain-of-Thought (CoT) prompting in Japanese and English using ChatGPT-3.5 and 4o-mini. The technique of zero-shot CoT, which involves appending a phrase such as "Let's think step by step" to a prompt to encourage reasoning before answering, has been shown to offer LLM performance improvements in mathematical and reasoning tasks, particularly in English. We investigate how these effects transfer to Japanese using the Japanese Multi-task Language Understanding Benchmark (JMMLU) and the Multi-task Language Understanding Benchmark (MMLU). Our results show that while zero-shot CoT prompting can lead to notable performance gains for some prompt categories in GPT-3.5, its impact in GPT-4o-mini is associated with significant performance declines. However, for Japanese prompts there remain certain categories, such as college mathematics and abstract algebra, that still exhibit improvements, despite the broader trend of diminishing effectiveness in more advanced models. 

**Abstract (ZH)**: 我们在使用ChatGPT-3.5和4o-mini时，比较了零样本Chain-of-Thought (CoT) 提示在日本语和英语中的有效性。 

---
# Fully-Decentralized MADDPG with Networked Agents 

**Title (ZH)**: 基于网络化代理的完全去中心化MADDPG 

**Authors**: Diego Bolliger, Lorenz Zauter, Robert Ziegler  

**Link**: [PDF](https://arxiv.org/pdf/2503.06747)  

**Abstract**: In this paper, we devise three actor-critic algorithms with decentralized training for multi-agent reinforcement learning in cooperative, adversarial, and mixed settings with continuous action spaces. To this goal, we adapt the MADDPG algorithm by applying a networked communication approach between agents. We introduce surrogate policies in order to decentralize the training while allowing for local communication during training. The decentralized algorithms achieve comparable results to the original MADDPG in empirical tests, while reducing computational cost. This is more pronounced with larger numbers of agents. 

**Abstract (ZH)**: 在本论文中，我们为协同、对抗和混合设置下的多智能体强化学习设计了三种带去中心化训练的Actor-Critic算法，适用于连续动作空间。为此，我们通过应用智能体之间的网络化通信方法来适应MADDPG算法。我们引入代理的替代策略以在允许局部通信的同时实现训练的去中心化。去中心化的算法在实证测试中达到了与原MADDPG相当的结果，同时降低了计算成本，这种效果在更多智能体的情况下更为显著。 

---
# ACAI for SBOs: AI Co-creation for Advertising and Inspiration for Small Business Owners 

**Title (ZH)**: ACAI 对 SBOs 的协同创造：面向小型企业主的广告人工智能共创 

**Authors**: Nimisha Karnatak, Adrien Baranes, Rob Marchant, Triona Butler, Kristen Olson  

**Link**: [PDF](https://arxiv.org/pdf/2503.06729)  

**Abstract**: Small business owners (SBOs) often lack the resources and design experience needed to produce high-quality advertisements. To address this, we developed ACAI (AI Co-Creation for Advertising and Inspiration), an GenAI-powered multimodal advertisement creation tool, and conducted a user study with 16 SBOs in London to explore their perceptions of and interactions with ACAI in advertisement creation. Our findings reveal that structured inputs enhance user agency and control while improving AI outputs by facilitating better brand alignment, enhancing AI transparency, and offering scaffolding that assists novice designers, such as SBOs, in formulating prompts. We also found that ACAI's multimodal interface bridges the design skill gap for SBOs with a clear advertisement vision, but who lack the design jargon necessary for effective prompting. Building on our findings, we propose three capabilities: contextual intelligence, adaptive interactions, and data management, with corresponding design recommendations to advance the co-creative attributes of AI-mediated design tools. 

**Abstract (ZH)**: 小型企业主（SBOs）常常缺乏生产高质量广告所需的资源和设计经验。为此，我们开发了ACAI（AI协同创作广告与灵感），这是一种基于GenAI的多模态广告创作工具，并在伦敦对16位SBOs进行了用户研究，以探索他们对ACAI在广告创作中的感知和互动。我们的研究发现，结构化的输入增强了用户的自主性和控制力，同时通过促进更好的品牌对齐、提高AI的透明度，并为初学者设计师，如SBOs，提供支撑性工具以制定提示，从而改进AI输出。我们还发现，ACAI的多模态界面为拥有清晰广告愿景但缺乏必要设计术语的SBOs填补了设计技能的空白。基于这些发现，我们提出了三项能力：情境智能、适应性交互和数据管理，并提出相应的设计建议以促进AI介导设计工具有助于协同创作的属性。 

---
# Pull-Based Query Scheduling for Goal-Oriented Semantic Communication 

**Title (ZH)**: 目标导向的语义通信基于拉式查询调度 

**Authors**: Pouya Agheli, Nikolaos Pappas, Marios Kountouris  

**Link**: [PDF](https://arxiv.org/pdf/2503.06725)  

**Abstract**: This paper addresses query scheduling for goal-oriented semantic communication in pull-based status update systems. We consider a system where multiple sensing agents (SAs) observe a source characterized by various attributes and provide updates to multiple actuation agents (AAs), which act upon the received information to fulfill their heterogeneous goals at the endpoint. A hub serves as an intermediary, querying the SAs for updates on observed attributes and maintaining a knowledge base, which is then broadcast to the AAs. The AAs leverage the knowledge to perform their actions effectively. To quantify the semantic value of updates, we introduce a grade of effectiveness (GoE) metric. Furthermore, we integrate cumulative perspective theory (CPT) into the long-term effectiveness analysis to account for risk awareness and loss aversion in the system. Leveraging this framework, we compute effect-aware scheduling policies aimed at maximizing the expected discounted sum of CPT-based total GoE provided by the transmitted updates while complying with a given query cost constraint. To achieve this, we propose a model-based solution based on dynamic programming and model-free solutions employing state-of-the-art deep reinforcement learning (DRL) algorithms. Our findings demonstrate that effect-aware scheduling significantly enhances the effectiveness of communicated updates compared to benchmark scheduling methods, particularly in settings with stringent cost constraints where optimal query scheduling is vital for system performance and overall effectiveness. 

**Abstract (ZH)**: 面向目标语义通信的基于拉取的现状更新系统的查询调度研究 

---
# PFDial: A Structured Dialogue Instruction Fine-tuning Method Based on UML Flowcharts 

**Title (ZH)**: PFDial：基于UML流程图的结构化对话指令微调方法 

**Authors**: Ming Zhang, Yuhui Wang, Yujiong Shen, Tingyi Yang, Changhao Jiang, Yilong Wu, Shihan Dou, Qinhao Chen, Zhiheng Xi, Zhihao Zhang, Yi Dong, Zhen Wang, Zhihui Fei, Mingyang Wan, Tao Liang, Guojun Ma, Qi Zhang, Tao Gui, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06706)  

**Abstract**: Process-driven dialogue systems, which operate under strict predefined process constraints, are essential in customer service and equipment maintenance scenarios. Although Large Language Models (LLMs) have shown remarkable progress in dialogue and reasoning, they still struggle to solve these strictly constrained dialogue tasks. To address this challenge, we construct Process Flow Dialogue (PFDial) dataset, which contains 12,705 high-quality Chinese dialogue instructions derived from 440 flowcharts containing 5,055 process nodes. Based on PlantUML specification, each UML flowchart is converted into atomic dialogue units i.e., structured five-tuples. Experimental results demonstrate that a 7B model trained with merely 800 samples, and a 0.5B model trained on total data both can surpass 90% accuracy. Additionally, the 8B model can surpass GPT-4o up to 43.88% with an average of 11.00%. We further evaluate models' performance on challenging backward transitions in process flows and conduct an in-depth analysis of various dataset formats to reveal their impact on model performance in handling decision and sequential branches. The data is released in this https URL. 

**Abstract (ZH)**: 过程驱动对话系统在客户服务中心和设备维护场景中至关重要。尽管大规模语言模型在对话和推理方面取得了显著进展，但在解决这些严格受限的对话任务方面仍然存在挑战。为应对这一挑战，我们构建了过程流程对话（PFDial）数据集，该数据集包含源自440个流程图的12,705条高质量中文对话指令，这些流程图包含5,055个过程节点。基于PlantUML规范，每张UML流程图被转换为原子对话单元，即结构化的五元组。实验结果显示，仅使用800样本训练的7B模型和使用全部数据训练的0.5B模型均可超过90%的准确率。此外，8B模型在平均准确性11.00%的基础上，比GPT-4o高出43.88%。我们进一步评估了模型在过程流程中具有挑战性的反向过渡的表现，并对各种数据集格式进行了深入分析，以揭示它们对模型处理决策和序列分支时性能的影响。数据在此处发布：https://链接。 

---
# Censoring-Aware Tree-Based Reinforcement Learning for Estimating Dynamic Treatment Regimes with Censored Outcomes 

**Title (ZH)**: 基于树的强化学习：考虑截尾结果的治疗策略估计， Aware剪枝树结构强化学习方法用于动态治疗方案估计 

**Authors**: Animesh Kumar Paul, Russell Greiner  

**Link**: [PDF](https://arxiv.org/pdf/2503.06690)  

**Abstract**: Dynamic Treatment Regimes (DTRs) provide a systematic approach for making sequential treatment decisions that adapt to individual patient characteristics, particularly in clinical contexts where survival outcomes are of interest. Censoring-Aware Tree-Based Reinforcement Learning (CA-TRL) is a novel framework to address the complexities associated with censored data when estimating optimal DTRs. We explore ways to learn effective DTRs, from observational data. By enhancing traditional tree-based reinforcement learning methods with augmented inverse probability weighting (AIPW) and censoring-aware modifications, CA-TRL delivers robust and interpretable treatment strategies. We demonstrate its effectiveness through extensive simulations and real-world applications using the SANAD epilepsy dataset, where it outperformed the recently proposed ASCL method in key metrics such as restricted mean survival time (RMST) and decision-making accuracy. This work represents a step forward in advancing personalized and data-driven treatment strategies across diverse healthcare settings. 

**Abstract (ZH)**: 基于去 cen 截断aware 的树形强化学习的动态治疗策略（Censoring-Aware Tree-Based Reinforcement Learning for Dynamic Treatment Regimes） 

---
# UniGenX: Unified Generation of Sequence and Structure with Autoregressive Diffusion 

**Title (ZH)**: UniGenX：统一生成序列与结构的自回归扩散方法 

**Authors**: Gongbo Zhang, Yanting Li, Renqian Luo, Pipi Hu, Zeru Zhao, Lingbo Li, Guoqing Liu, Zun Wang, Ran Bi, Kaiyuan Gao, Liya Guo, Yu Xie, Chang Liu, Jia Zhang, Tian Xie, Robert Pinsler, Claudio Zeni, Ziheng Lu, Yingce Xia, Marwin Segler, Maik Riechert, Li Yuan, Lei Chen, Haiguang Liu, Tao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.06687)  

**Abstract**: Unified generation of sequence and structure for scientific data (e.g., materials, molecules, proteins) is a critical task. Existing approaches primarily rely on either autoregressive sequence models or diffusion models, each offering distinct advantages and facing notable limitations. Autoregressive models, such as GPT, Llama, and Phi-4, have demonstrated remarkable success in natural language generation and have been extended to multimodal tasks (e.g., image, video, and audio) using advanced encoders like VQ-VAE to represent complex modalities as discrete sequences. However, their direct application to scientific domains is challenging due to the high precision requirements and the diverse nature of scientific data. On the other hand, diffusion models excel at generating high-dimensional scientific data, such as protein, molecule, and material structures, with remarkable accuracy. Yet, their inability to effectively model sequences limits their potential as general-purpose multimodal foundation models. To address these challenges, we propose UniGenX, a unified framework that combines autoregressive next-token prediction with conditional diffusion models. This integration leverages the strengths of autoregressive models to ease the training of conditional diffusion models, while diffusion-based generative heads enhance the precision of autoregressive predictions. We validate the effectiveness of UniGenX on material and small molecule generation tasks, achieving a significant leap in state-of-the-art performance for material crystal structure prediction and establishing new state-of-the-art results for small molecule structure prediction, de novo design, and conditional generation. Notably, UniGenX demonstrates significant improvements, especially in handling long sequences for complex structures, showcasing its efficacy as a versatile tool for scientific data generation. 

**Abstract (ZH)**: 统一生成科学数据（如材料、分子、蛋白质）的序列和结构是一个关键任务。现有方法主要依赖于自回归序列模型或扩散模型，各自具有独特的优势和明显的局限性。自回归模型，如GPT、Llama和Phi-4，在自然语言生成方面取得了显著成功，并通过像VQ-VAE这样的高级编码器扩展到多模态任务（如图像、视频和音频），以表示复杂的模态为离散序列。然而，它们直接应用于科学领域因高精度要求和科学数据的多样性而具有挑战性。另一方面，扩散模型在生成蛋白质、分子和材料结构等高维科学数据方面表现出显著的准确性。然而，它们在有效建模序列方面的不足限制了它们作为通用多模态基础模型的应用潜力。为应对这些挑战，我们提出了UniGenX，一个结合自回归下一个令牌预测与条件扩散模型的统一框架。这种集成利用自回归模型的优势来简化条件扩散模型的训练，而基于扩散的生成头部则增强了自回归预测的精度。我们通过材料和小分子生成任务验证了UniGenX的有效性，显著提高了材料晶体结构预测的最新性能，并在小分子结构预测、从头设计和条件生成方面建立了新的最新性能结果。值得注意的是，UniGenX 在处理复杂结构的长序列时显示出显著改进，彰显了其作为科学数据生成工具的多功能性。 

---
# AA-CLIP: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP 

**Title (ZH)**: AA-CLIP: 基于异常意识CLIP的零样本异常检测增强方法 

**Authors**: Wenxin Ma, Xu Zhang, Qingsong Yao, Fenghe Tang, Chenxu Wu, Yingtai Li, Rui Yan, Zihang Jiang, S.Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.06661)  

**Abstract**: Anomaly detection (AD) identifies outliers for applications like defect and lesion detection. While CLIP shows promise for zero-shot AD tasks due to its strong generalization capabilities, its inherent Anomaly-Unawareness leads to limited discrimination between normal and abnormal features. To address this problem, we propose Anomaly-Aware CLIP (AA-CLIP), which enhances CLIP's anomaly discrimination ability in both text and visual spaces while preserving its generalization capability. AA-CLIP is achieved through a straightforward yet effective two-stage approach: it first creates anomaly-aware text anchors to differentiate normal and abnormal semantics clearly, then aligns patch-level visual features with these anchors for precise anomaly localization. This two-stage strategy, with the help of residual adapters, gradually adapts CLIP in a controlled manner, achieving effective AD while maintaining CLIP's class knowledge. Extensive experiments validate AA-CLIP as a resource-efficient solution for zero-shot AD tasks, achieving state-of-the-art results in industrial and medical applications. The code is available at this https URL. 

**Abstract (ZH)**: 基于异常感知的CLIP（AA-CLIP）在零样本异常检测任务中的应用 

---
# Deep Cut-informed Graph Embedding and Clustering 

**Title (ZH)**: 基于深切图的图嵌入与聚类 

**Authors**: Zhiyuan Ning, Zaitian Wang, Ran Zhang, Ping Xu, Kunpeng Liu, Pengyang Wang, Chong Chen, Pengfei Wang, Yuanchun Zhou, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2503.06635)  

**Abstract**: Graph clustering aims to divide the graph into different clusters. The recently emerging deep graph clustering approaches are largely built on graph neural networks (GNN). However, GNN is designed for general graph encoding and there is a common issue of representation collapse in existing GNN-based deep graph clustering algorithms. We attribute two main reasons for such issue: (i) the inductive bias of GNN models: GNNs tend to generate similar representations for proximal nodes. Since graphs often contain a non-negligible amount of inter-cluster links, the bias results in error message passing and leads to biased clustering; (ii) the clustering guided loss function: most traditional approaches strive to make all samples closer to pre-learned cluster centers, which cause a degenerate solution assigning all data points to a single label thus make all samples and less discriminative. To address these challenges, we investigate graph clustering from a graph cut perspective and propose an innovative and non-GNN-based Deep Cut-informed Graph embedding and Clustering framework, namely DCGC. This framework includes two modules: (i) cut-informed graph encoding; (ii) self-supervised graph clustering via optimal transport. For the encoding module, we derive a cut-informed graph embedding objective to fuse graph structure and attributes by minimizing their joint normalized cut. For the clustering module, we utilize the optimal transport theory to obtain the clustering assignments, which can balance the guidance of proximity to the pre-learned cluster center. With the above two tailored designs, DCGC is more suitable for the graph clustering task, which can effectively alleviate the problem of representation collapse and achieve better performance. We conduct extensive experiments to demonstrate that our method is simple but effective compared with benchmarks. 

**Abstract (ZH)**: 基于图切分的非GNN深度图嵌入与聚类框架：DCGC 

---
# BTFL: A Bayesian-based Test-Time Generalization Method for Internal and External Data Distributions in Federated learning 

**Title (ZH)**: BTFL：一种基于贝叶斯方法的fed学习内外数据分布迁移测试时泛化方法 

**Authors**: Yu Zhou, Bingyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06633)  

**Abstract**: Federated Learning (FL) enables multiple clients to collaboratively develop a global model while maintaining data privacy. However, online FL deployment faces challenges due to distribution shifts and evolving test samples. Personalized Federated Learning (PFL) tailors the global model to individual client distributions, but struggles with Out-Of-Distribution (OOD) samples during testing, leading to performance degradation. In real-world scenarios, balancing personalization and generalization during online testing is crucial and existing methods primarily focus on training-phase generalization. To address the test-time trade-off, we introduce a new scenario: Test-time Generalization for Internal and External Distributions in Federated Learning (TGFL), which evaluates adaptability under Internal Distribution (IND) and External Distribution (EXD). We propose BTFL, a Bayesian-based test-time generalization method for TGFL, which balances generalization and personalization at the sample level during testing. BTFL employs a two-head architecture to store local and global knowledge, interpolating predictions via a dual-Bayesian framework that considers both historical test data and current sample characteristics with theoretical guarantee and faster speed. Our experiments demonstrate that BTFL achieves improved performance across various datasets and models with less time cost. The source codes are made publicly available at this https URL . 

**Abstract (ZH)**: 联邦学习中的内部和外部分布下的测试时泛化（BTFL）：一种基于贝叶斯的测试时泛化方法 

---
# Hardware-Accelerated Event-Graph Neural Networks for Low-Latency Time-Series Classification on SoC FPGA 

**Title (ZH)**: 基于SoC FPGA的硬件加速事件图神经网络低延迟时间序列分类 

**Authors**: Hiroshi Nakano, Krzysztof Blachut, Kamil Jeziorek, Piotr Wzorek, Manon Dampfhoffer, Thomas Mesquida, Hiroaki Nishi, Tomasz Kryjak, Thomas Dalgaty  

**Link**: [PDF](https://arxiv.org/pdf/2503.06629)  

**Abstract**: As the quantities of data recorded by embedded edge sensors grow, so too does the need for intelligent local processing. Such data often comes in the form of time-series signals, based on which real-time predictions can be made locally using an AI model. However, a hardware-software approach capable of making low-latency predictions with low power consumption is required. In this paper, we present a hardware implementation of an event-graph neural network for time-series classification. We leverage an artificial cochlea model to convert the input time-series signals into a sparse event-data format that allows the event-graph to drastically reduce the number of calculations relative to other AI methods. We implemented the design on a SoC FPGA and applied it to the real-time processing of the Spiking Heidelberg Digits (SHD) dataset to benchmark our approach against competitive solutions. Our method achieves a floating-point accuracy of 92.7% on the SHD dataset for the base model, which is only 2.4% and 2% less than the state-of-the-art models with over 10% and 67% fewer model parameters, respectively. It also outperforms FPGA-based spiking neural network implementations by 19.3% and 4.5%, achieving 92.3% accuracy for the quantised model while using fewer computational resources and reducing latency. 

**Abstract (ZH)**: 基于事件图神经网络的时间序列分类的硬件实现 

---
# Revisiting Early Detection of Sexual Predators via Turn-level Optimization 

**Title (ZH)**: 重新审视基于对话回合级优化的早期识别性虐待者方法 

**Authors**: Jinmyeong An, Sangwon Ryu, Heejin Do, Yunsu Kim, Jungseul Ok, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.06627)  

**Abstract**: Online grooming is a severe social threat where sexual predators gradually entrap child victims with subtle and gradual manipulation. Therefore, timely intervention for online grooming is critical for proactive protection. However, previous methods fail to determine the optimal intervention points (i.e., jump to conclusions) as they rely on chat-level risk labels by causing weak supervision of risky utterances. For timely detection, we propose speed control reinforcement learning (SCoRL) (The code and supplementary materials are available at this https URL), incorporating a practical strategy derived from luring communication theory (LCT). To capture the predator's turn-level entrapment, we use a turn-level risk label based on the LCT. Then, we design a novel speed control reward function that balances the trade-off between speed and accuracy based on turn-level risk label; thus, SCoRL can identify the optimal intervention moment. In addition, we introduce a turn-level metric for precise evaluation, identifying limitations in previously used chat-level metrics. Experimental results show that SCoRL effectively preempted online grooming, offering a more proactive and timely solution. Further analysis reveals that our method enhances performance while intuitively identifying optimal early intervention points. 

**Abstract (ZH)**: 在线诱骗是一种严重的社会威胁，性 predators 通过微妙而渐进的操纵逐步诱骗儿童受害者。因此，及时干预在线诱骗对于主动保护至关重要。然而，以往的方法未能确定最佳干预点（即过早下结论），因为它们依赖于对话级别风险标签，导致对风险性陈述的监督薄弱。为实现及时检测，我们提出了一种速度控制强化学习（SCoRL）方法（代码和补充材料可在以下链接获取：this https URL），结合了诱骗沟通理论（LCT）提出的一种实用策略。为了捕捉 predator 的回合级诱骗，我们基于 LCT 使用了回合级风险标签。然后，我们设计了一种新的速度控制奖励函数，该函数基于回合级风险标签平衡速度与准确性的trade-off；因此，SCoRL 可以识别最优干预时刻。此外，我们引入了回合级评估指标，以精确评估并识别先前使用的对话级评估指标的局限性。实验结果显示，SCoRL 有效预防了在线诱骗，提供了更为主动和及时的解决方案。进一步分析表明，我们的方法在直观地识别最优早期干预点方面提高了性能。 

---
# Using Subgraph GNNs for Node Classification:an Overlooked Potential Approach 

**Title (ZH)**: 使用子图GNNs进行节点分类：一种被忽视的潜在方法 

**Authors**: Qian Zeng, Xin Lin, Jingyi Gao, Yang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06614)  

**Abstract**: Previous studies have demonstrated the strong performance of Graph Neural Networks (GNNs) in node classification. However, most existing GNNs adopt a node-centric perspective and rely on global message passing, leading to high computational and memory costs that hinder scalability. To mitigate these challenges, subgraph-based methods have been introduced, leveraging local subgraphs as approximations of full computational trees. While this approach improves efficiency, it often suffers from performance degradation due to the loss of global contextual information, limiting its effectiveness compared to global GNNs. To address this trade-off between scalability and classification accuracy, we reformulate the node classification task as a subgraph classification problem and propose SubGND (Subgraph GNN for NoDe). This framework introduces a differentiated zero-padding strategy and an Ego-Alter subgraph representation method to resolve label conflicts while incorporating an Adaptive Feature Scaling Mechanism to dynamically adjust feature contributions based on dataset-specific dependencies. Experimental results on six benchmark datasets demonstrate that SubGND achieves performance comparable to or surpassing global message-passing GNNs, particularly in heterophilic settings, highlighting its effectiveness and scalability as a promising solution for node classification. 

**Abstract (ZH)**: 基于子图的节点分类方法SubGND：在保持分类准确性的同时提高可扩展性 

---
# SHIP: A Shapelet-based Approach for Interpretable Patient-Ventilator Asynchrony Detection 

**Title (ZH)**: SHIP：一种基于形状图的可解释患者-呼吸机异步检测方法 

**Authors**: Xuan-May Le, Ling Luo, Uwe Aickelin, Minh-Tuan Tran, David Berlowitz, Mark Howard  

**Link**: [PDF](https://arxiv.org/pdf/2503.06571)  

**Abstract**: Patient-ventilator asynchrony (PVA) is a common and critical issue during mechanical ventilation, affecting up to 85% of patients. PVA can result in clinical complications such as discomfort, sleep disruption, and potentially more severe conditions like ventilator-induced lung injury and diaphragm dysfunction. Traditional PVA management, which relies on manual adjustments by healthcare providers, is often inadequate due to delays and errors. While various computational methods, including rule-based, statistical, and deep learning approaches, have been developed to detect PVA events, they face challenges related to dataset imbalances and lack of interpretability. In this work, we propose a shapelet-based approach SHIP for PVA detection, utilizing shapelets - discriminative subsequences in time-series data - to enhance detection accuracy and interpretability. Our method addresses dataset imbalances through shapelet-based data augmentation and constructs a shapelet pool to transform the dataset for more effective classification. The combined shapelet and statistical features are then used in a classifier to identify PVA events. Experimental results on medical datasets show that SHIP significantly improves PVA detection while providing interpretable insights into model decisions. 

**Abstract (ZH)**: 基于形状let的方法SHIP在机械通气患者-呼吸机不协调检测中的应用：提高检测准确性和可解释性 

---
# LSA: Latent Style Augmentation Towards Stain-Agnostic Cervical Cancer Screening 

**Title (ZH)**: LSA: 潜在风格增强 toward 无染色差异性的宫颈癌筛查 

**Authors**: Jiangdong Cai, Haotian Jiang, Zhenrong Shen, Yonghao Li, Honglin Xiong, Lichi Zhang, Qian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06563)  

**Abstract**: The deployment of computer-aided diagnosis systems for cervical cancer screening using whole slide images (WSIs) faces critical challenges due to domain shifts caused by staining variations across different scanners and imaging environments. While existing stain augmentation methods improve patch-level robustness, they fail to scale to WSIs due to two key limitations: (1) inconsistent stain patterns when extending patch operations to gigapixel slides, and (2) prohibitive computational/storage costs from offline processing of augmented this http URL address this, we propose Latent Style Augmentation (LSA), a framework that performs efficient, online stain augmentation directly on WSI-level latent features. We first introduce WSAug, a WSI-level stain augmentation method ensuring consistent stain across patches within a WSI. Using offline-augmented WSIs by WSAug, we design and train Stain Transformer, which can simulate targeted style in the latent space, efficiently enhancing the robustness of the WSI-level classifier. We validate our method on a multi-scanner WSI dataset for cervical cancer diagnosis. Despite being trained on data from a single scanner, our approach achieves significant performance improvements on out-of-distribution data from other scanners. Code will be available at this https URL. 

**Abstract (ZH)**: 使用整个切片图像（WSI）进行宫颈癌筛查的计算机辅助诊断系统的部署受到跨不同扫描器和成像环境的染色变异引起的领域变化的严峻挑战。尽管现有的染色增强方法可以提高局部鲁棒性，但由于两个关键限制，它们难以扩展到WSI：（1）在扩展局部操作到 gigapixel 切片时的一致染色模式不一致，（2）从离线处理增强切片所导致的高昂的计算/存储成本。为了解决这一问题，我们提出了一种称为潜在風格增强（Latent Style Augmentation，LSA）的框架，该框架可以在WSI级别的潜在特征上直接执行高效的在线染色增强。我们首先引入WSAug，一种WSI级别的染色增强方法，确保WSI内各个局部区域的一致染色。利用WSAug离线增强的WSI，我们设计并训练Stain Transformer，该模型可以在潜在空间中模拟目标风格，高效地增强WSI级分类器的鲁棒性。我们在一个包含多个扫描器的WSI数据集上验证了该方法，尽管仅使用单个扫描器的数据进行训练，但在来自其他扫描器的外部数据上实现了显著的性能提升。代码可从此地址获取。 

---
# AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection 

**Title (ZH)**: AnywhereDoor：面向目标检测的多目标后门攻击 

**Authors**: Jialin Lu, Junjie Shan, Ziqi Zhao, Ka-Ho Chow  

**Link**: [PDF](https://arxiv.org/pdf/2503.06529)  

**Abstract**: As object detection becomes integral to many safety-critical applications, understanding its vulnerabilities is essential. Backdoor attacks, in particular, pose a serious threat by implanting hidden triggers in victim models, which adversaries can later exploit to induce malicious behaviors during inference. However, current understanding is limited to single-target attacks, where adversaries must define a fixed malicious behavior (target) before training, making inference-time adaptability impossible. Given the large output space of object detection (including object existence prediction, bounding box estimation, and classification), the feasibility of flexible, inference-time model control remains unexplored. This paper introduces AnywhereDoor, a multi-target backdoor attack for object detection. Once implanted, AnywhereDoor allows adversaries to make objects disappear, fabricate new ones, or mislabel them, either across all object classes or specific ones, offering an unprecedented degree of control. This flexibility is enabled by three key innovations: (i) objective disentanglement to scale the number of supported targets; (ii) trigger mosaicking to ensure robustness even against region-based detectors; and (iii) strategic batching to address object-level data imbalances that hinder manipulation. Extensive experiments demonstrate that AnywhereDoor grants attackers a high degree of control, improving attack success rates by 26% compared to adaptations of existing methods for such flexible control. 

**Abstract (ZH)**: 对象检测中的多目标后门攻击：AnywhereDoor 

---
# Generative AI as Digital Media 

**Title (ZH)**: 生成式AI作为数字媒体 

**Authors**: Gilad Abiri  

**Link**: [PDF](https://arxiv.org/pdf/2503.06523)  

**Abstract**: Generative AI is frequently portrayed as revolutionary or even apocalyptic, prompting calls for novel regulatory approaches. This essay argues that such views are misguided. Instead, generative AI should be understood as an evolutionary step in the broader algorithmic media landscape, alongside search engines and social media. Like these platforms, generative AI centralizes information control, relies on complex algorithms to shape content, and extensively uses user data, thus perpetuating common problems: unchecked corporate power, echo chambers, and weakened traditional gatekeepers. Regulation should therefore share a consistent objective: ensuring media institutions remain trustworthy. Without trust, public discourse risks fragmenting into isolated communities dominated by comforting, tribal beliefs -- a threat intensified by generative AI's capacity to bypass gatekeepers and personalize truth. Current governance frameworks, such as the EU's AI Act and the US Executive Order 14110, emphasize reactive risk mitigation, addressing measurable threats like national security, public health, and algorithmic bias. While effective for novel technological risks, this reactive approach fails to adequately address broader issues of trust and legitimacy inherent to digital media. Proactive regulation fostering transparency, accountability, and public confidence is essential. Viewing generative AI exclusively as revolutionary risks repeating past regulatory failures that left social media and search engines insufficiently regulated. Instead, regulation must proactively shape an algorithmic media environment serving the public good, supporting quality information and robust civic discourse. 

**Abstract (ZH)**: 生成式AI常常被描绘为革命性的甚至带来 apocalypse 的技术，促使人们呼吁采取新的监管方法。本文认为这些观点是误导性的。相反，生成式AI应被视为在更广泛的算法媒体景观中的一种进化步骤，类似于搜索引擎和社会媒体。与这些平台一样，生成式AI集中控制信息，依赖复杂的算法来塑造内容，并大量使用用户数据，从而延续了常见的问题：不受约束的公司权力、回声室效应和传统把关人的削弱。因此，监管应致力于一个共同目标：确保媒体机构保持可信度。缺乏信任，公共话语风险分裂成由令人安慰的部落信念主导的孤立社区——生成式AI绕过把关人和个性化真实性的能力使其威胁加剧。当前的治理框架，如欧盟AI法案和美国行政命令14110，强调应对性风险缓解，针对国家安保、公共健康和算法偏见等可测量的威胁。虽然对于新兴技术风险是有效的，但这种应对性方法未能充分解决数字媒体中固有的信任和合法性问题。建立促进透明度、问责制和公众信心的前瞻性监管是必不可少的。仅仅将生成式AI视为革命性的风险，有重复过去监管失败、使社交媒体和搜索引擎监管不足的危险。相反，监管必须积极塑造服务于公共利益的算法媒体环境，支持高质量信息和稳健的公民对话。 

---
# HFedCKD: Toward Robust Heterogeneous Federated Learning via Data-free Knowledge Distillation and Two-way Contrast 

**Title (ZH)**: HFedCKD: 基于数据无关知识蒸馏和双向对比的鲁棒异构联邦学习 

**Authors**: Yiting Zheng, Bohan Lin, Jinqian Chen, Jihua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06511)  

**Abstract**: Most current federated learning frameworks are modeled as static processes, ignoring the dynamic characteristics of the learning system. Under the limited communication budget of the central server, the flexible model architecture of a large number of clients participating in knowledge transfer requires a lower participation rate, active clients have uneven contributions, and the client scale seriously hinders the performance of FL. We consider a more general and practical federation scenario and propose a system heterogeneous federation method based on data-free knowledge distillation and two-way contrast (HFedCKD). We apply the Inverse Probability Weighted Distillation (IPWD) strategy to the data-free knowledge transfer framework. The generator completes the data features of the nonparticipating clients. IPWD implements a dynamic evaluation of the prediction contribution of each client under different data distributions. Based on the antibiased weighting of its prediction loss, the weight distribution of each client is effectively adjusted to fairly integrate the knowledge of participating clients. At the same time, the local model is split into a feature extractor and a classifier. Through differential contrast learning, the feature extractor is aligned with the global model in the feature space, while the classifier maintains personalized decision-making capabilities. HFedCKD effectively alleviates the knowledge offset caused by a low participation rate under data-free knowledge distillation and improves the performance and stability of the model. We conduct extensive experiments on image and IoT datasets to comprehensively evaluate and verify the generalization and robustness of the proposed HFedCKD framework. 

**Abstract (ZH)**: 基于数据免费知识蒸馏和双向对比的系统异构联邦学习方法（HFedCKD） 

---
# ExGes: Expressive Human Motion Retrieval and Modulation for Audio-Driven Gesture Synthesis 

**Title (ZH)**: ExGes: 表达力人体动作检索与调节用于音控手势合成 

**Authors**: Xukun Zhou, Fengxin Li, Ming Chen, Yan Zhou, Pengfei Wan, Di Zhang, Hongyan Liu, Jun He, Zhaoxin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.06499)  

**Abstract**: Audio-driven human gesture synthesis is a crucial task with broad applications in virtual avatars, human-computer interaction, and creative content generation. Despite notable progress, existing methods often produce gestures that are coarse, lack expressiveness, and fail to fully align with audio semantics. To address these challenges, we propose ExGes, a novel retrieval-enhanced diffusion framework with three key designs: (1) a Motion Base Construction, which builds a gesture library using training dataset; (2) a Motion Retrieval Module, employing constrative learning and momentum distillation for fine-grained reference poses retreiving; and (3) a Precision Control Module, integrating partial masking and stochastic masking to enable flexible and fine-grained control. Experimental evaluations on BEAT2 demonstrate that ExGes reduces Fréchet Gesture Distance by 6.2\% and improves motion diversity by 5.3\% over EMAGE, with user studies revealing a 71.3\% preference for its naturalness and semantic relevance. Code will be released upon acceptance. 

**Abstract (ZH)**: 基于音频的人体手势合成是虚拟化身、人机交互和创意内容生成等领域的重要任务。尽管取得了显著进展，现有方法往往生成粗糙、缺乏表现力的手势，并且无法完全与音频语义对齐。为了解决这些挑战，我们提出了ExGes，一种带有三种关键设计的检索增强扩散框架：（1）运动基础构建，使用训练数据集构建手势库；（2）运动检索模块，采用对比学习和动量蒸馏进行精细参考姿态检索；（3）精度控制模块，结合部分掩码和随机掩码以实现灵活和精细的控制。实验结果表明，ExGes 在 BEAT2 上将弗雷谢手势距离降低了 6.2%，提高了运动多样性 5.3%，用户研究显示其自然性和语义相关性获得了 71.3% 的偏好。接受后将发布代码。 

---
# PDB: Not All Drivers Are the Same -- A Personalized Dataset for Understanding Driving Behavior 

**Title (ZH)**: PDB: 并非所有驾驶员都相同——一个理解驾驶行为的个性化数据集 

**Authors**: Chuheng Wei, Ziye Qin, Siyan Li, Ziyan Zhang, Xuanpeng Zhao, Amr Abdelraouf, Rohit Gupta, Kyungtae Han, Matthew J. Barth, Guoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06477)  

**Abstract**: Driving behavior is inherently personal, influenced by individual habits, decision-making styles, and physiological states. However, most existing datasets treat all drivers as homogeneous, overlooking driver-specific variability. To address this gap, we introduce the Personalized Driving Behavior (PDB) dataset, a multi-modal dataset designed to capture personalization in driving behavior under naturalistic driving conditions. Unlike conventional datasets, PDB minimizes external influences by maintaining consistent routes, vehicles, and lighting conditions across sessions. It includes sources from 128-line LiDAR, front-facing camera video, GNSS, 9-axis IMU, CAN bus data (throttle, brake, steering angle), and driver-specific signals such as facial video and heart rate. The dataset features 12 participants, approximately 270,000 LiDAR frames, 1.6 million images, and 6.6 TB of raw sensor data. The processed trajectory dataset consists of 1,669 segments, each spanning 10 seconds with a 0.2-second interval. By explicitly capturing drivers' behavior, PDB serves as a unique resource for human factor analysis, driver identification, and personalized mobility applications, contributing to the development of human-centric intelligent transportation systems. 

**Abstract (ZH)**: 个性化驾驶行为（PDB）数据集 

---
# Enhancing Layer Attention Efficiency through Pruning Redundant Retrievals 

**Title (ZH)**: 通过修剪冗余检索提升层注意力效率 

**Authors**: Hanze Li, Xiande Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06473)  

**Abstract**: Growing evidence suggests that layer attention mechanisms, which enhance interaction among layers in deep neural networks, have significantly advanced network architectures. However, existing layer attention methods suffer from redundancy, as attention weights learned by adjacent layers often become highly similar. This redundancy causes multiple layers to extract nearly identical features, reducing the model's representational capacity and increasing training time. To address this issue, we propose a novel approach to quantify redundancy by leveraging the Kullback-Leibler (KL) divergence between adjacent layers. Additionally, we introduce an Enhanced Beta Quantile Mapping (EBQM) method that accurately identifies and skips redundant layers, thereby maintaining model stability. Our proposed Efficient Layer Attention (ELA) architecture, improves both training efficiency and overall performance, achieving a 30\% reduction in training time while enhancing performance in tasks such as image classification and object detection. 

**Abstract (ZH)**: 逐层注意力机制通过增强深神经网络中各层之间的交互，显著推进了网络架构的发展。然而，现有的层注意力方法存在冗余问题，相邻层学习到的注意力权重通常高度相似，导致多个层提取几乎相同的特征，降低了模型的表示能力并增加了训练时间。为此，我们提出了一种新颖的方法，通过利用相邻层之间的Kullback-Leibler（KL）散度来量化冗余。此外，我们引入了一种增强的贝塔分位映射（EBQM）方法，能够准确识别并跳过冗余层，从而保持模型的稳定性。我们提出的高效层注意力（ELA）架构提高了训练效率和整体性能，在图像分类和物体检测等任务中实现了30%的训练时间减少。 

---
# Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning 

**Title (ZH)**: 几何知识引导的局部全局分布对齐的联邦学习 

**Authors**: Yanbiao Ma, Wei Dai, Wenke Huang, Jiayi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.06457)  

**Abstract**: Data heterogeneity in federated learning, characterized by a significant misalignment between local and global distributions, leads to divergent local optimization directions and hinders global model training. Existing studies mainly focus on optimizing local updates or global aggregation, but these indirect approaches demonstrate instability when handling highly heterogeneous data distributions, especially in scenarios where label skew and domain skew coexist. To address this, we propose a geometry-guided data generation method that centers on simulating the global embedding distribution locally. We first introduce the concept of the geometric shape of an embedding distribution and then address the challenge of obtaining global geometric shapes under privacy constraints. Subsequently, we propose GGEUR, which leverages global geometric shapes to guide the generation of new samples, enabling a closer approximation to the ideal global distribution. In single-domain scenarios, we augment samples based on global geometric shapes to enhance model generalization; in multi-domain scenarios, we further employ class prototypes to simulate the global distribution across domains. Extensive experimental results demonstrate that our method significantly enhances the performance of existing approaches in handling highly heterogeneous data, including scenarios with label skew, domain skew, and their coexistence. Code published at: this https URL 

**Abstract (ZH)**: 联邦学习中由局部和全局分布显著不匹配引起的数据异质性导致了局部优化方向的发散并阻碍了全局模型训练。现有研究主要集中在优化局部更新或全局聚合，但这些间接方法在处理高度异质性数据分布时表现不稳定，尤其是在标签偏斜和领域偏斜共存的情况下。为解决这一问题，我们提出了一种几何引导的数据生成方法，重点在于局部模拟全局嵌入分布的几何形状。我们首先引入嵌入分布几何形状的概念，然后在隐私约束下解决了获取全局几何形状的挑战。随后，我们提出了GGEUR方法，利用全局几何形状来指导新样本的生成，以更接近理想全局分布。在单领域场景下，我们基于全局几何形状增强样本以提高模型泛化能力；在多领域场景下，我们进一步使用类原型来模拟跨领域的全局分布。大量实验结果表明，我们的方法显著提升了现有方法在处理高度异质性数据，包括标签偏斜、领域偏斜及其共存情况下的性能。代码发布于：this https URL 

---
# CtrTab: Tabular Data Synthesis with High-Dimensional and Limited Data 

**Title (ZH)**: CtrTab: 高维度小样本量表格数据合成 

**Authors**: Zuqing Li, Jianzhong Qi, Junhao Gan  

**Link**: [PDF](https://arxiv.org/pdf/2503.06444)  

**Abstract**: Diffusion-based tabular data synthesis models have yielded promising results. However, we observe that when the data dimensionality increases, existing models tend to degenerate and may perform even worse than simpler, non-diffusion-based models. This is because limited training samples in high-dimensional space often hinder generative models from capturing the distribution accurately. To address this issue, we propose CtrTab-a condition controlled diffusion model for tabular data synthesis-to improve the performance of diffusion-based generative models in high-dimensional, low-data scenarios. Through CtrTab, we inject samples with added Laplace noise as control signals to improve data diversity and show its resemblance to L2 regularization, which enhances model robustness. Experimental results across multiple datasets show that CtrTab outperforms state-of-the-art models, with performance gap in accuracy over 80% on average. Our source code will be released upon paper publication. 

**Abstract (ZH)**: 基于扩散的表格数据合成模型已经取得了令人鼓舞的结果。然而，我们观察到，在数据维度增加时，现有模型往往会出现退化现象，甚至可能比简单的非扩散基于模型表现更差。这是因为高维度空间中有限的训练样本常常会阻碍生成模型准确捕捉数据分布。为解决这一问题，我们提出了CtrTab——一种条件控制的扩散模型，以提高高维度、低数据场景下扩散生成模型的性能。通过CtrTab，我们注入带有拉普lace噪声的样本作为控制信号，以提高数据多样性，并将其与L2正则化做类比，以增强模型的稳健性。来自多个数据集的实验结果显示，CtrTab在准确率方面平均超过现有最先进的模型80%以上。我们的源代码将在论文发表后公开。 

---
# Physics-Informed Residual Neural Ordinary Differential Equations for Enhanced Tropical Cyclone Intensity Forecasting 

**Title (ZH)**: 基于物理约束的残差神经常微分方程在热带气旋强度预报中的应用 

**Authors**: Fan Meng  

**Link**: [PDF](https://arxiv.org/pdf/2503.06436)  

**Abstract**: Accurate tropical cyclone (TC) intensity prediction is crucial for mitigating storm hazards, yet its complex dynamics pose challenges to traditional methods. Here, we introduce a Physics-Informed Residual Neural Ordinary Differential Equation (PIR-NODE) model to precisely forecast TC intensity evolution. This model leverages the powerful non-linear fitting capabilities of deep learning, integrates residual connections to enhance model depth and training stability, and explicitly models the continuous temporal evolution of TC intensity using Neural ODEs. Experimental results in the SHIPS dataset demonstrate that the PIR-NODE model achieves a significant improvement in 24-hour intensity prediction accuracy compared to traditional statistical models and benchmark deep learning methods, with a 25. 2\% reduction in the root mean square error (RMSE) and a 19.5\% increase in R-square (R2) relative to a baseline of neural network. Crucially, the residual structure effectively preserves initial state information, and the model exhibits robust generalization capabilities. This study details the PIR-NODE model architecture, physics-informed integration strategies, and comprehensive experimental validation, revealing the substantial potential of deep learning techniques in predicting complex geophysical systems and laying the foundation for future refined TC forecasting research. 

**Abstract (ZH)**: 准确的热带气旋（TC）强度预测对于减缓风暴灾害至关重要，但其复杂的动力学特性对传统方法构成了挑战。本文介绍了一种物理信息残差神经常微分方程（PIR-NODE）模型，以精确预报TC强度演变。该模型利用深度学习强大的非线性拟合能力，通过残差连接增强模型深度和训练稳定性，并使用神经常微分方程显式建模TC强度的连续时间演化。在SHIPS数据集的实验结果表明，PIR-NODE模型在24小时强度预测精度上显著优于传统统计模型和基准深度学习方法，与基线神经网络相比，均方根误差（RMSE）降低了25.2%，R平方（R2）提高了19.5%。重要的是，残差结构有效保留了初始状态信息，并且该模型表现出 robust 通用化能力。本文详细介绍了PIR-NODE模型架构、物理信息集成策略以及全面的实验验证，揭示了深度学习技术在预测复杂地球物理系统方面的巨大潜力，并为未来的精细化TC预报研究奠定了基础。 

---
# Decoding the Black Box: Integrating Moral Imagination with Technical AI Governance 

**Title (ZH)**: 解码黑盒：将道德想象与技术AI治理相结合 

**Authors**: Krti Tallam  

**Link**: [PDF](https://arxiv.org/pdf/2503.06411)  

**Abstract**: This paper examines the intricate interplay among AI safety, security, and governance by integrating technical systems engineering with principles of moral imagination and ethical philosophy. Drawing on foundational insights from Weapons of Math Destruction and Thinking in Systems alongside contemporary debates in AI ethics, we develop a comprehensive multi-dimensional framework designed to regulate AI technologies deployed in high-stakes domains such as defense, finance, healthcare, and education. Our approach combines rigorous technical analysis, quantitative risk assessment, and normative evaluation to expose systemic vulnerabilities inherent in opaque, black-box models. Detailed case studies, including analyses of Microsoft Tay (2016) and the UK A-Level Grading Algorithm (2020), demonstrate how security lapses, bias amplification, and lack of accountability can precipitate cascading failures that undermine public trust. We conclude by outlining targeted strategies for enhancing AI resilience through adaptive regulatory mechanisms, robust security protocols, and interdisciplinary oversight, thereby advancing the state of the art in ethical and technical AI governance. 

**Abstract (ZH)**: 本文通过将技术系统工程与道德想象原则和伦理哲学相结合，探讨了AI安全、安全与治理之间的复杂交互关系，并借鉴《数学破坏武器》和《系统思维》的基础洞见以及当前AI伦理领域的讨论，开发了一个全面的多维框架，旨在监管国防、金融、医疗和教育等高风险领域中的AI技术。本文方法结合了严格的技術分析、定量风险评估和规范评估，以揭示不透明的黑盒模型中固有的系统性漏洞。通过详细的案例研究，包括对微软Tay（2016年）和英国A-Level评分算法（2020年）的分析，展示了安全漏洞、偏见放大及缺乏问责制如何导致级联失败，从而侵蚀公众信任。最后，本文提出了通过适应性监管机制、 robust安全协议和跨学科监督来增强AI韧性的目标策略，从而推动伦理和技术治理领域的创新。 

---
# Causality Enhanced Origin-Destination Flow Prediction in Data-Scarce Cities 

**Title (ZH)**: 数据稀缺城市中的因果增强起源-目的地流量预测 

**Authors**: Tao Feng, Yunke Zhang, Huandong Wang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06398)  

**Abstract**: Accurate origin-destination (OD) flow prediction is of great importance to developing cities, as it can contribute to optimize urban structures and layouts. However, with the common issues of missing regional features and lacking OD flow data, it is quite daunting to predict OD flow in developing cities. To address this challenge, we propose a novel Causality-Enhanced OD Flow Prediction (CE-OFP), a unified framework that aims to transfer urban knowledge between cities and achieve accuracy improvements in OD flow predictions across data-scarce cities. In specific, we propose a novel reinforcement learning model to discover universal causalities among urban features in data-rich cities and build corresponding causal graphs. Then, we further build Causality-Enhanced Variational Auto-Encoder (CE-VAE) to incorporate causal graphs for effective feature reconstruction in data-scarce cities. Finally, with the reconstructed features, we devise a knowledge distillation method with a graph attention network to migrate the OD prediction model from data-rich cities to data-scare cities. Extensive experiments on two pairs of real-world datasets validate that the proposed CE-OFP remarkably outperforms state-of-the-art baselines, which can reduce the RMSE of OD flow prediction for data-scarce cities by up to 11%. 

**Abstract (ZH)**: 基于因果增强的Origin-Destination流预测（CE-OFP） 

---
# EPR-GAIL: An EPR-Enhanced Hierarchical Imitation Learning Framework to Simulate Complex User Consumption Behaviors 

**Title (ZH)**: EPR-GAIL：一种增强层次 imitation 学习框架，用于模拟复杂的用户消费行为 

**Authors**: Tao Feng, Yunke Zhang, Huandong Wang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06392)  

**Abstract**: User consumption behavior data, which records individuals' online spending history at various types of stores, has been widely used in various applications, such as store recommendation, site selection, and sale forecasting. However, its high worth is limited due to deficiencies in data comprehensiveness and changes of application scenarios. Thus, generating high-quality sequential consumption data by simulating complex user consumption behaviors is of great importance to real-world applications. Two branches of existing sequence generation methods are both limited in quality. Model-based methods with simplified assumptions fail to model the complex decision process of user consumption, while data-driven methods that emulate real-world data are prone to noises, unobserved behaviors, and dynamic decision space. In this work, we propose to enhance the fidelity and trustworthiness of the data-driven Generative Adversarial Imitation Learning (GAIL) method by blending it with the Exploration and Preferential Return EPR model . The core idea of our EPR-GAIL framework is to model user consumption behaviors as a complex EPR decision process, which consists of purchase, exploration, and preference decisions. Specifically, we design the hierarchical policy function in the generator as a realization of the EPR decision process and employ the probability distributions of the EPR model to guide the reward function in the discriminator. Extensive experiments on two real-world datasets of user consumption behaviors on an online platform demonstrate that the EPR-GAIL framework outperforms the best state-of-the-art baseline by over 19\% in terms of data fidelity. Furthermore, the generated consumption behavior data can improve the performance of sale prediction and location recommendation by up to 35.29% and 11.19%, respectively, validating its advantage for practical applications. 

**Abstract (ZH)**: 基于EPR模型增强的生成对抗模仿学习框架：提升用户消费行为数据的真实性和可靠性 

---
# Machine Learning meets Algebraic Combinatorics: A Suite of Datasets Capturing Research-level Conjecturing Ability in Pure Mathematics 

**Title (ZH)**: 机器学习邂逅代数组合数学：一套捕捉纯数学研究级猜想能力的数据集 

**Authors**: Herman Chau, Helen Jenne, Davis Brown, Jesse He, Mark Raugas, Sara Billey, Henry Kvinge  

**Link**: [PDF](https://arxiv.org/pdf/2503.06366)  

**Abstract**: With recent dramatic increases in AI system capabilities, there has been growing interest in utilizing machine learning for reasoning-heavy, quantitative tasks, particularly mathematics. While there are many resources capturing mathematics at the high-school, undergraduate, and graduate level, there are far fewer resources available that align with the level of difficulty and open endedness encountered by professional mathematicians working on open problems. To address this, we introduce a new collection of datasets, the Algebraic Combinatorics Dataset Repository (ACD Repo), representing either foundational results or open problems in algebraic combinatorics, a subfield of mathematics that studies discrete structures arising from abstract algebra. Further differentiating our dataset collection is the fact that it aims at the conjecturing process. Each dataset includes an open-ended research-level question and a large collection of examples (up to 10M in some cases) from which conjectures should be generated. We describe all nine datasets, the different ways machine learning models can be applied to them (e.g., training with narrow models followed by interpretability analysis or program synthesis with LLMs), and discuss some of the challenges involved in designing datasets like these. 

**Abstract (ZH)**: 随着人工智能系统能力的 recent 激增，在推理性和量化任务，特别是数学领域，利用机器学习的兴趣逐渐增长。为了应对这一需求，我们引入了代数组合学数据集库（ACD Repo），收录了代数组合学领域的基础成果或开放问题，代数组合学是研究源自抽象代数的离散结构的一个数学子领域。该数据集库特别注重猜想过程，每个数据集包含一个开放性研究级问题以及大量示例（某些情况下多达10M），用于生成猜想。我们描述了所有九个数据集以及机器学习模型在它们上的不同应用方式（例如，使用窄模型训练后进行可解释性分析或使用大语言模型进行程序合成），并讨论了设计这类数据集所面临的挑战。 

---
# The AI Pentad, the CHARME$^{2}$D Model, and an Assessment of Current-State AI Regulation 

**Title (ZH)**: 人工智能五元组、CHARME$^{2}$D模型及其对当前人工智能监管状况的评估 

**Authors**: Di Kevin Gao, Sudip Mittal, Jiming Wu, Hongwei Du, Jingdao Chen, Shahram Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2503.06353)  

**Abstract**: Artificial Intelligence (AI) has made remarkable progress in the past few years with AI-enabled applications beginning to permeate every aspect of our society. Despite the widespread consensus on the need to regulate AI, there remains a lack of a unified approach to framing, developing, and assessing AI regulations. Many of the existing methods take a value-based approach, for example, accountability, fairness, free from bias, transparency, and trust. However, these methods often face challenges at the outset due to disagreements in academia over the subjective nature of these definitions. This paper aims to establish a unifying model for AI regulation from the perspective of core AI components. We first introduce the AI Pentad, which comprises the five essential components of AI: humans and organizations, algorithms, data, computing, and energy. We then review AI regulatory enablers, including AI registration and disclosure, AI monitoring, and AI enforcement mechanisms. Subsequently, we present the CHARME$^{2}$D Model to explore further the relationship between the AI Pentad and AI regulatory enablers. Finally, we apply the CHARME$^{2}$D model to assess AI regulatory efforts in the European Union (EU), China, the United Arab Emirates (UAE), the United Kingdom (UK), and the United States (US), highlighting their strengths, weaknesses, and gaps. This comparative evaluation offers insights for future legislative work in the AI domain. 

**Abstract (ZH)**: 人工智能（AI）在过去的几年中取得了显著进步，AI驱动的应用开始渗透到我们社会的各个方面。尽管普遍认为需要对AI进行监管，但仍缺乏一个统一的方法来构建、开发和评估AI监管。现有方法多采用一种基于价值的方法，例如，问责性、公平性、无偏见、透明度和信任。然而，这些方法往往由于学术界对这些定义的主观性存在分歧而面临初始挑战。本文旨在从核心AI组件的角度建立一个统一的AI监管模型。首先，我们介绍了AI五元组，包括AI的五大基本组件：人类与组织、算法、数据、计算和能源。然后，我们回顾了AI监管支持者，包括AI登记与披露、AI监控和AI执法机制。接着，我们提出了CHARME$^{2}$D模型，进一步探索AI五元组与AI监管支持者之间的关系。最后，我们应用CHARME$^{2}$D模型评估欧盟、中国、阿拉伯联合酋长国、英国和美国在AI监管方面的努力，指出其优点、缺点和不足。这一比较评估为未来AI领域的立法工作提供了启示。 

---
# Studying the Interplay Between the Actor and Critic Representations in Reinforcement Learning 

**Title (ZH)**: 研究演员与评论家表示之间的交互作用在强化学习中的作用 

**Authors**: Samuel Garcin, Trevor McInroe, Pablo Samuel Castro, Prakash Panangaden, Christopher G. Lucas, David Abel, Stefano V. Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2503.06343)  

**Abstract**: Extracting relevant information from a stream of high-dimensional observations is a central challenge for deep reinforcement learning agents. Actor-critic algorithms add further complexity to this challenge, as it is often unclear whether the same information will be relevant to both the actor and the critic. To this end, we here explore the principles that underlie effective representations for the actor and for the critic in on-policy algorithms. We focus our study on understanding whether the actor and critic will benefit from separate, rather than shared, representations. Our primary finding is that when separated, the representations for the actor and critic systematically specialise in extracting different types of information from the environment -- the actor's representation tends to focus on action-relevant information, while the critic's representation specialises in encoding value and dynamics information. We conduct a rigourous empirical study to understand how different representation learning approaches affect the actor and critic's specialisations and their downstream performance, in terms of sample efficiency and generation capabilities. Finally, we discover that a separated critic plays an important role in exploration and data collection during training. Our code, trained models and data are accessible at this https URL. 

**Abstract (ZH)**: 从高维观测流中提取相关信息是深度强化学习代理面临的核心挑战。演员-评论家算法为这一挑战增加了额外的复杂性，因为通常不清楚相同的信息对演员和评论家而言是否具有相关性。为此，我们在此探索了有效表示原则，这些原则适用于策略和评论在在线算法中的表示。我们将研究重点放在理解演员和评论家是否从分离的、而不是共享的表示中受益上。我们的主要发现是，当分离时，演员和评论家的表示会系统地专门化于从环境中提取不同类型的信息——演员的表示倾向于关注与动作相关的信息，而评论家的表示则专门编码价值和动力学信息。我们进行了严格的经验研究，以了解不同的表示学习方法如何影响演员和评论家的专门化及其下游性能，特别是在样本效率和生成能力方面。最后，我们发现分离的评论家在训练过程中的探索和数据收集中扮演着重要角色。我们的代码、训练模型和数据可在以下网址访问：this https URL。 

---
# Synergizing AI and Digital Twins for Next-Generation Network Optimization, Forecasting, and Security 

**Title (ZH)**: 协同AI与数字孪生优化下一代网络性能、预测与安全 

**Authors**: Zifan Zhang, Minghong Fang, Dianwei Chen, Xianfeng Yang, Yuchen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06302)  

**Abstract**: Digital network twins (DNTs) are virtual representations of physical networks, designed to enable real-time monitoring, simulation, and optimization of network performance. When integrated with machine learning (ML) techniques, particularly federated learning (FL) and reinforcement learning (RL), DNTs emerge as powerful solutions for managing the complexities of network operations. This article presents a comprehensive analysis of the synergy of DNTs, FL, and RL techniques, showcasing their collective potential to address critical challenges in 6G networks. We highlight key technical challenges that need to be addressed, such as ensuring network reliability, achieving joint data-scenario forecasting, and maintaining security in high-risk environments. Additionally, we propose several pipelines that integrate DNT and ML within coherent frameworks to enhance network optimization and security. Case studies demonstrate the practical applications of our proposed pipelines in edge caching and vehicular networks. In edge caching, the pipeline achieves over 80% cache hit rates while balancing base station loads. In autonomous vehicular system, it ensure a 100% no-collision rate, showcasing its reliability in safety-critical scenarios. By exploring these synergies, we offer insights into the future of intelligent and adaptive network systems that automate decision-making and problem-solving. 

**Abstract (ZH)**: 数字网络孪生（DNTs）是物理网络的虚拟表示，旨在实现网络性能的实时监测、仿真和优化。当结合机器学习（ML）技术，特别是联邦学习（FL）和强化学习（RL）时，DNTs成为管理网络操作复杂性的有力解决方案。本文全面分析了DNTs、FL和RL技术的协同效应，展示了其在解决6G网络关键挑战方面的潜在能力。我们强调了需要应对的关键技术挑战，包括确保网络可靠性、实现联合数据场景预测以及在高风险环境中维护安全。此外，我们提出了几条将DNT和ML集成到一致框架中的管道，以增强网络优化和安全性。案例研究展示了我们提出的管道在边缘缓存和车载网络中的实际应用。在边缘缓存中，管道实现了超过80%的缓存命中率，并平衡了基站的负载。在自主车载系统中，它确保了100%的安全，展示了其在安全关键场景中的可靠性。通过探索这些协同效应，我们提供了有关未来智能化和自适应网络系统的信息，这些系统能够自动进行决策和问题解决。 

---
# Single Domain Generalization with Adversarial Memory 

**Title (ZH)**: 单域适应性记忆强化单域泛化 

**Authors**: Hao Yan, Marzi Heidari, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.06288)  

**Abstract**: Domain Generalization (DG) aims to train models that can generalize to unseen testing domains by leveraging data from multiple training domains. However, traditional DG methods rely on the availability of multiple diverse training domains, limiting their applicability in data-constrained scenarios. Single Domain Generalization (SDG) addresses the more realistic and challenging setting by restricting the training data to a single domain distribution. The main challenges in SDG stem from the limited diversity of training data and the inaccessibility of unseen testing data distributions. To tackle these challenges, we propose a single domain generalization method that leverages an adversarial memory bank to augment training features. Our memory-based feature augmentation network maps both training and testing features into an invariant subspace spanned by diverse memory features, implicitly aligning the training and testing domains in the projected space. To maintain a diverse and representative feature memory bank, we introduce an adversarial feature generation method that creates features extending beyond the training domain distribution. Experimental results demonstrate that our approach achieves state-of-the-art performance on standard single domain generalization benchmarks. 

**Abstract (ZH)**: 单域泛化（SDG）旨在通过利用单个训练域的数据训练模型，使其能够在未见测试域上泛化。传统的泛化方法依赖于多个 diverse 的训练域，这限制了它们在数据受限场景中的应用。单域泛化（SDG）通过限制训练数据到单个域分布，解决了更现实和具有挑战性的设置。单域泛化的主要挑战来自于训练数据的有限多样性和未见测试域分布的不可访问性。为了应对这些挑战，我们提出了一种利用对抗记忆库扩充训练特征的单域泛化方法。基于记忆的特征扩充网络将训练和测试特征映射到由多样记忆特征构成的不变子空间中，隐式地在投影空间中对齐训练和测试域。为了保持多样且具有代表性的特征记忆库，我们引入了一种对抗特征生成方法，该方法生成扩展到训练域分布之外的特征。实验结果表明，我们的方法在标准单域泛化基准上达到了最先进的性能。 

---
# Applied Machine Learning Methods with Long-Short Term Memory Based Recurrent Neural Networks for Multivariate Temperature Prediction 

**Title (ZH)**: 基于长短期记忆循环神经网络的多元温度预测中应用的机器学习方法 

**Authors**: Bojan Lukić  

**Link**: [PDF](https://arxiv.org/pdf/2503.06278)  

**Abstract**: This paper gives an overview on how to develop a dense and deep neural network for making a time series prediction. First, the history and cornerstones in Artificial Intelligence and Machine Learning will be presented. After a short introduction to the theory of Artificial Intelligence and Machine Learning, the paper will go deeper into the techniques for conducting a time series prediction with different models of neural networks. For this project, Python's development environment Jupyter, extended with the TensorFlow package and deep-learning application Keras is used. The system setup and project framework are explained in more detail before discussing the time series prediction. The main part shows an applied example of time series prediction with weather data. For this work, a deep recurrent neural network with Long Short-Term Memory cells is used to conduct the time series prediction. The results and evaluation of the work show that a weather prediction with deep neural networks can be successful for a short time period. However, there are some drawbacks and limitations with time series prediction, which will be discussed towards the end of the paper. 

**Abstract (ZH)**: 本文概述了如何开发密集和深层神经网络以进行时间序列预测。首先，将介绍人工智能和机器学习的历史和基石。在简要介绍人工智能和机器学习的理论之后，文章将深入探讨使用不同神经网络模型进行时间序列预测的技术。为此项目，使用了 Python 的开发环境 Jupyter，扩展了 TensorFlow 包和深度学习应用 Keras。在讨论时间序列预测之前，详细解释了系统设置和项目框架。主要部分展示了使用天气数据进行时间序列预测的应用示例。在此工作中，使用具有长短期记忆单元的深层递归神经网络进行时间序列预测。工作的结果和评估表明，使用深度神经网络进行短期天气预测可以取得成功。然而，时间序列预测存在一些缺点和局限性，这将在本文末尾进行讨论。 

---
# Infant Cry Detection Using Causal Temporal Representation 

**Title (ZH)**: 基于因果时间表示的婴儿哭声检测 

**Authors**: Minghao Fu, Danning Li, Aryan Gadhiya, Benjamin Lambright, Mohamed Alowais, Mohab Bahnassy, Saad El Dine Elletter, Hawau Olamide Toyin, Haiyan Jiang, Kun Zhang, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.06247)  

**Abstract**: This paper addresses a major challenge in acoustic event detection, in particular infant cry detection in the presence of other sounds and background noises: the lack of precise annotated data. We present two contributions for supervised and unsupervised infant cry detection. The first is an annotated dataset for cry segmentation, which enables supervised models to achieve state-of-the-art performance. Additionally, we propose a novel unsupervised method, Causal Representation Spare Transition Clustering (CRSTC), based on causal temporal representation, which helps address the issue of data scarcity more generally. By integrating the detected cry segments, we significantly improve the performance of downstream infant cry classification, highlighting the potential of this approach for infant care applications. 

**Abstract (ZH)**: 本文解决了声学事件检测中的一个重大挑战，特别是在其他声音和背景噪声共存的情况下婴儿哭声检测缺乏精确标注数据的问题。我们提出了监督和非监督婴儿哭声检测的两个贡献。首先，我们提供了一个标注数据集用于哭声分割，这使得监督模型能够达到最佳性能。此外，我们提出了一种基于因果时间表征的新型非监督方法——因果表示稀疏过渡聚类（CRSTC），该方法有助于更广泛地解决数据稀缺问题。通过整合检测到的哭声片段，显著提高了婴儿哭声分类的性能，突显了该方法在婴儿护理应用中的潜力。 

---
# A Frank System for Co-Evolutionary Hybrid Decision-Making 

**Title (ZH)**: 一种共生演化混合决策系统 

**Authors**: Federico Mazzoni, Riccardo Guidotti, Alessio Malizia  

**Link**: [PDF](https://arxiv.org/pdf/2503.06229)  

**Abstract**: We introduce Frank, a human-in-the-loop system for co-evolutionary hybrid decision-making aiding the user to label records from an un-labeled dataset. Frank employs incremental learning to ``evolve'' in parallel with the user's decisions, by training an interpretable machine learning model on the records labeled by the user. Furthermore, Frank advances state-of-the-art approaches by offering inconsistency controls, explanations, fairness checks, and bad-faith safeguards simultaneously. We evaluate our proposal by simulating the users' behavior with various levels of expertise and reliance on Frank's suggestions. The experiments show that Frank's intervention leads to improvements in the accuracy and the fairness of the decisions. 

**Abstract (ZH)**: 我们介绍Frank，一个包含人类闭环的系统，用于共生演化混合决策辅助用户从未标记数据集中标注记录。Frank通过在用户的决策过程中并行“演化”，利用增量学习训练可解释的机器学习模型来进行标注。此外，Frank同时提供了不一致性控制、解释、公平性检查和恶意行为防护，超越了现有方法。我们通过模拟不同专业水平和依赖Frank建议程度的用户行为来评估我们的提议。实验表明，Frank的干预提高了决策的准确性和公平性。 

---
# Optimal Output Feedback Learning Control for Discrete-Time Linear Quadratic Regulation 

**Title (ZH)**: 离散时间线性二次调节的最优输出反馈学习控制 

**Authors**: Kedi Xiea, Martin Guay, Shimin Wang, Fang Deng, Maobin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06226)  

**Abstract**: This paper studies the linear quadratic regulation (LQR) problem of unknown discrete-time systems via dynamic output feedback learning control. In contrast to the state feedback, the optimality of the dynamic output feedback control for solving the LQR problem requires an implicit condition on the convergence of the state observer. Moreover, due to unknown system matrices and the existence of observer error, it is difficult to analyze the convergence and stability of most existing output feedback learning-based control methods. To tackle these issues, we propose a generalized dynamic output feedback learning control approach with guaranteed convergence, stability, and optimality performance for solving the LQR problem of unknown discrete-time linear systems. In particular, a dynamic output feedback controller is designed to be equivalent to a state feedback controller. This equivalence relationship is an inherent property without requiring convergence of the estimated state by the state observer, which plays a key role in establishing the off-policy learning control approaches. By value iteration and policy iteration schemes, the adaptive dynamic programming based learning control approaches are developed to estimate the optimal feedback control gain. In addition, a model-free stability criterion is provided by finding a nonsingular parameterization matrix, which contributes to establishing a switched iteration scheme. Furthermore, the convergence, stability, and optimality analyses of the proposed output feedback learning control approaches are given. Finally, the theoretical results are validated by two numerical examples. 

**Abstract (ZH)**: 基于动态输出反馈学习控制的未知离散时间系统线性二次调节问题研究 

---
# GraphGen+: Advancing Distributed Subgraph Generation and Graph Learning On Industrial Graphs 

**Title (ZH)**: GraphGen+：推动工业图的分布式子图生成与图学习 

**Authors**: Yue Jin, Yongchao Liu, Chuntao Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.06212)  

**Abstract**: Graph-based computations are crucial in a wide range of applications, where graphs can scale to trillions of edges. To enable efficient training on such large graphs, mini-batch subgraph sampling is commonly used, which allows training without loading the entire graph into memory. However, existing solutions face significant trade-offs: online subgraph generation, as seen in frameworks like DGL and PyG, is limited to a single machine, resulting in severe performance bottlenecks, while offline precomputed subgraphs, as in GraphGen, improve sampling efficiency but introduce large storage overhead and high I/O costs during training. To address these challenges, we propose \textbf{GraphGen+}, an integrated framework that synchronizes distributed subgraph generation with in-memory graph learning, eliminating the need for external storage while significantly improving efficiency. GraphGen+ achieves a \textbf{27$\times$} speedup in subgraph generation compared to conventional SQL-like methods and a \textbf{1.3$\times$} speedup over GraphGen, supporting training on 1 million nodes per iteration and removing the overhead associated with precomputed subgraphs, making it a scalable and practical solution for industry-scale graph learning. 

**Abstract (ZH)**: 基于图的计算在众多应用中至关重要，其中图可以扩展到万亿边的规模。为在如此大的图上进行高效训练，通常使用mini-batch子图采样，这允许在不加载整个图到内存中的情况下进行训练。然而，现有解决方案存在重大权衡：像DGL和PyG这样的框架中的在线子图生成仅限于单台机器，导致严重的性能瓶颈，而GraphGen等预先计算子图的方法提高采样效率但引入了较大的存储开销和高I/O成本。为解决这些挑战，我们提出了一种综合框架GraphGen+，该框架将分布式子图生成与内存中图学习同步，无需外部存储并显著提高效率。GraphGen+在子图生成上的速度比传统SQL-like方法快27倍，比GraphGen快1.3倍，支持每迭代训练100万节点，并消除了预先计算子图的开销，从而提供了一个适用于大规模工业图学习的可扩展和实用的解决方案。 

---
# Distributed Graph Neural Network Inference With Just-In-Time Compilation For Industry-Scale Graphs 

**Title (ZH)**: 基于即时编译的分布式图神经网络推理方法及其在工业规模图中的应用 

**Authors**: Xiabao Wu, Yongchao Liu, Wei Qin, Chuntao Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.06208)  

**Abstract**: Graph neural networks (GNNs) have delivered remarkable results in various fields. However, the rapid increase in the scale of graph data has introduced significant performance bottlenecks for GNN inference. Both computational complexity and memory usage have risen dramatically, with memory becoming a critical limitation. Although graph sampling-based subgraph learning methods can help mitigate computational and memory demands, they come with drawbacks such as information loss and high redundant computation among subgraphs. This paper introduces an innovative processing paradgim for distributed graph learning that abstracts GNNs with a new set of programming interfaces and leverages Just-In-Time (JIT) compilation technology to its full potential. This paradigm enables GNNs to highly exploit the computational resources of distributed clusters by eliminating the drawbacks of subgraph learning methods, leading to a more efficient inference process. Our experimental results demonstrate that on industry-scale graphs of up to \textbf{500 million nodes and 22.4 billion edges}, our method can produce a performance boost of up to \textbf{27.4 times}. 

**Abstract (ZH)**: 图神经网络（GNNs）在多个领域取得了 remarkable 的成果。然而，图数据规模的迅速增长为 GNN 推断引入了显著的性能瓶颈。计算复杂度和内存使用量大幅上升，其中内存成为关键的限制因素。尽管基于图采样的子图学习方法可以缓解计算和内存需求，但这些方法存在信息丢失和子图之间高冗余计算的缺点。本文提出了一种创新的分布式图学习处理范式，通过引入一组新的编程接口并充分利用 Just-In-Time (JIT) 编译技术，使 GNN 能够充分利用分布式集群的计算资源，从而避免子图学习方法的缺点，实现更高效的推断过程。实验结果表明，在包含多达 \textbf{500 million 节点和 22.4 billion 边} 的工业规模图上，我们的方法可以带来高达 \textbf{27.4 倍} 的性能提升。 

---
# Explainable Synthetic Image Detection through Diffusion Timestep Ensembling 

**Title (ZH)**: 可解释的合成图像检测通过扩散时间步长集成 

**Authors**: Yixin Wu, Feiran Zhang, Tianyuan Shi, Ruicheng Yin, Zhenghua Wang, Zhenliang Gan, Xiaohua Wang, Changze Lv, Xiaoqing Zheng, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06201)  

**Abstract**: Recent advances in diffusion models have enabled the creation of deceptively real images, posing significant security risks when misused. In this study, we reveal that natural and synthetic images exhibit distinct differences in the high-frequency domains of their Fourier power spectra after undergoing iterative noise perturbations through an inverse multi-step denoising process, suggesting that such noise can provide additional discriminative information for identifying synthetic images. Based on this observation, we propose a novel detection method that amplifies these differences by progressively adding noise to the original images across multiple timesteps, and train an ensemble of classifiers on these noised images. To enhance human comprehension, we introduce an explanation generation and refinement module to identify flaws located in AI-generated images. Additionally, we construct two new datasets, GenHard and GenExplain, derived from the GenImage benchmark, providing detection samples of greater difficulty and high-quality rationales for fake images. Extensive experiments show that our method achieves state-of-the-art performance with 98.91% and 95.89% detection accuracy on regular and harder samples, increasing a minimal of 2.51% and 3.46% compared to baselines. Furthermore, our method also generalizes effectively to images generated by other diffusion models. Our code and datasets will be made publicly available. 

**Abstract (ZH)**: Recent advances in扩散模型的进展使得生成具有欺骗性的逼真图像成为可能，当这些图像被误用时会引发重大安全风险。在本研究中，我们揭示了经过迭代去噪过程中的多步逆向噪声扰动后，自然图像和合成图像在傅里叶功率谱的高频域中表现出明显的差异，表明这种噪声可以提供额外的判别信息以识别合成图像。基于这一观察，我们提出了一种新的检测方法，通过在多个时间步逐步向原始图像添加噪声来放大这些差异，并在这些被噪声增强的图像上训练集成分类器。为了增强人类的可理解性，我们引入了一个解释生成和精炼模块，用于识别AI生成图像中的缺陷。此外，我们构建了两个新的数据集，GenHard和GenExplain，源自GenImage基准，提供了更具难度的检测样本和高质量的假图像理由。广泛实验证明，我们的方法在常规样本和更难样本上的检测准确率分别达到98.91%和95.89%，相较于基线方法分别提升了至少2.51%和3.46%。此外，我们的方法还能够有效泛化到其他扩散模型生成的图像。我们的代码和数据集将公开发布。 

---
# Human-AI Experience in Integrated Development Environments: A Systematic Literature Review 

**Title (ZH)**: 集成开发环境中的人类-人工智能体验：一项系统文献综述 

**Authors**: Agnia Sergeyuk, Ilya Zakharov, Ekaterina Koshchenko, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2503.06195)  

**Abstract**: The integration of Artificial Intelligence (AI) into Integrated Development Environments (IDEs) is reshaping software development, fundamentally altering how developers interact with their tools. This shift marks the emergence of Human-AI Experience in Integrated Development Environment (in-IDE HAX), a field that explores the evolving dynamics of Human-Computer Interaction in AI-assisted coding environments. Despite rapid adoption, research on in-IDE HAX remains fragmented which highlights the need for a unified overview of current practices, challenges, and opportunities. To provide a structured overview of existing research, we conduct a systematic literature review of 89 studies, summarizing current findings and outlining areas for further investigation.
Our findings reveal that AI-assisted coding enhances developer productivity but also introduces challenges, such as verification overhead, automation bias, and over-reliance, particularly among novice developers. Furthermore, concerns about code correctness, security, and maintainability highlight the urgent need for explainability, verification mechanisms, and adaptive user control. Although recent advances have driven the field forward, significant research gaps remain, including a lack of longitudinal studies, personalization strategies, and AI governance frameworks. This review provides a foundation for advancing in-IDE HAX research and offers guidance for responsibly integrating AI into software development. 

**Abstract (ZH)**: AI融入集成开发环境中的集成开发环境内人机体验（in-IDE HAX）：现状、挑战与机遇 

---
# Lightweight Software Kernels and Hardware Extensions for Efficient Sparse Deep Neural Networks on Microcontrollers 

**Title (ZH)**: 轻量级软件内核和硬件扩展以提高Micro控制器上稀疏深度神经网络的效率 

**Authors**: Francesco Daghero, Daniele Jahier Pagliari, Francesco Conti, Luca Benini, Massimo Poncino, Alessio Burrello  

**Link**: [PDF](https://arxiv.org/pdf/2503.06183)  

**Abstract**: The acceleration of pruned Deep Neural Networks (DNNs) on edge devices such as Microcontrollers (MCUs) is a challenging task, given the tight area- and power-constraints of these devices. In this work, we propose a three-fold contribution to address this problem. First, we design a set of optimized software kernels for N:M pruned layers, targeting ultra-low-power, multicore RISC-V MCUs, which are up to 2.1x and 3.4x faster than their dense counterparts at 1:8 and 1:16 sparsity, respectively. Then, we implement a lightweight Instruction-Set Architecture (ISA) extension to accelerate the indirect load and non-zero indices decompression operations required by our kernels, obtaining up to 1.9x extra speedup, at the cost of a 5% area overhead. Lastly, we extend an open-source DNN compiler to utilize our sparse kernels for complete networks, showing speedups of 3.21x and 1.81x on a ResNet18 and a Vision Transformer (ViT), with less than 1.5% accuracy drop compared to a dense baseline. 

**Abstract (ZH)**: 针对边缘设备如微控制器（MCUs）上剪枝深度神经网络（DNNs）的加速是一个具有挑战性的问题，鉴于这些设备面积和功率限制紧凑。本文提出了三个方面的贡献来解决这一问题。首先，我们为N:M剪枝层设计了一组优化软件内核，目标是超低功耗多核RISC-V微控制器，在稀疏度分别为1:8和1:16的情况下，内核分别比其密集对应版本快2.1倍和3.4倍。其次，我们实现了一个轻量级指令集架构（ISA）扩展以加速我们的内核所需的数据间接加载和非零索引解压缩操作，额外获得了1.9倍的速度提升，成本仅为5%的面积开销。最后，我们将一个开源DNN编译器扩展为利用我们的稀疏内核来优化整个网络，在ResNet18和Vision Transformer（ViT）上分别获得了3.21倍和1.81倍的速度提升，与密集基线相比准确率降低不到1.5%。 

---
# ROCM: RLHF on consistency models 

**Title (ZH)**: ROCM: 一致性模型上的RLHF 

**Authors**: Shivanshu Shekhar, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06171)  

**Abstract**: Diffusion models have revolutionized generative modeling in continuous domains like image, audio, and video synthesis. However, their iterative sampling process leads to slow generation and inefficient training, challenges that are further exacerbated when incorporating Reinforcement Learning from Human Feedback (RLHF) due to sparse rewards and long time horizons. Consistency models address these issues by enabling single-step or efficient multi-step generation, significantly reducing computational costs.
In this work, we propose a direct reward optimization framework for applying RLHF to consistency models, incorporating distributional regularization to enhance training stability and prevent reward hacking. We investigate various $f$-divergences as regularization strategies, striking a balance between reward maximization and model consistency. Unlike policy gradient methods, our approach leverages first-order gradients, making it more efficient and less sensitive to hyperparameter tuning. Empirical results show that our method achieves competitive or superior performance compared to policy gradient based RLHF methods, across various automatic metrics and human evaluation. Additionally, our analysis demonstrates the impact of different regularization techniques in improving model generalization and preventing overfitting. 

**Abstract (ZH)**: 扩散模型在图像、音频和视频合成等连续域中的生成建模中取得了革命性进展。然而，其迭代采样过程导致生成速度慢和训练效率低，而在结合人类反馈强化学习（RLHF）时，由于稀疏奖励和长时间 horizon，这些问题进一步加剧。一致性模型通过启用单步或多步生成来解决这些问题，显著降低了计算成本。
在本工作中，我们提出了一种直接的奖励优化框架，将 RLHF 应用于一致性模型，并引入分布正则化以提高训练稳定性并防止奖励作弊。我们研究了不同类型的 $f$-散度作为正则化策略，以平衡奖励最大化和模型一致性。与策略梯度方法不同，我们的方法利用一阶梯度，使其更有效且对超参数调整的敏感度较低。实验结果表明，我们的方法在各种自动评价指标和人工评估中与基于策略梯度的 RLHF 方法相比表现相当或更优。此外，我们的分析还展示了不同正则化技术对模型泛化能力和防止过拟合的影响。 

---
# Secure On-Device Video OOD Detection Without Backpropagation 

**Title (ZH)**: 设备端无反向传播的 Secure 在设备上视频 OOD 检测 

**Authors**: Li Li, Peilin Cai, Yuxiao Zhou, Zhiyu Ni, Renjie Liang, You Qin, Yi Nian, Zhengzhong Tu, Xiyang Hu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.06166)  

**Abstract**: Out-of-Distribution (OOD) detection is critical for ensuring the reliability of machine learning models in safety-critical applications such as autonomous driving and medical diagnosis. While deploying personalized OOD detection directly on edge devices is desirable, it remains challenging due to large model sizes and the computational infeasibility of on-device training. Federated learning partially addresses this but still requires gradient computation and backpropagation, exceeding the capabilities of many edge devices. To overcome these challenges, we propose SecDOOD, a secure cloud-device collaboration framework for efficient on-device OOD detection without requiring device-side backpropagation. SecDOOD utilizes cloud resources for model training while ensuring user data privacy by retaining sensitive information on-device. Central to SecDOOD is a HyperNetwork-based personalized parameter generation module, which adapts cloud-trained models to device-specific distributions by dynamically generating local weight adjustments, effectively combining central and local information without local fine-tuning. Additionally, our dynamic feature sampling and encryption strategy selectively encrypts only the most informative feature channels, largely reducing encryption overhead without compromising detection performance. Extensive experiments across multiple datasets and OOD scenarios demonstrate that SecDOOD achieves performance comparable to fully fine-tuned models, enabling secure, efficient, and personalized OOD detection on resource-limited edge devices. To enhance accessibility and reproducibility, our code is publicly available at this https URL. 

**Abstract (ZH)**: 基于安全云-设备协作的高效本地OOD检测框架SecDOOD 

---
# Exploring the usage of Probabilistic Neural Networks for Ionospheric electron density estimation 

**Title (ZH)**: 探究概率神经网络在电离层电子密度估计中的应用 

**Authors**: Miquel Garcia-Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2503.06144)  

**Abstract**: A fundamental limitation of traditional Neural Networks (NN) in predictive modelling is their inability to quantify uncertainty in their outputs. In critical applications like positioning systems, understanding the reliability of predictions is critical for constructing confidence intervals, early warning systems, and effectively propagating results. For instance, Precise Point Positioning in satellite navigation heavily relies on accurate error models for ancillary data (orbits, clocks, ionosphere, and troposphere) to compute precise error estimates. In addition, these uncertainty estimates are needed to establish robust protection levels in safety critical applications.
To address this challenge, the main objectives of this paper aims at exploring a potential framework capable of providing both point estimates and associated uncertainty measures of ionospheric Vertical Total Electron Content (VTEC). In this context, Probabilistic Neural Networks (PNNs) offer a promising approach to achieve this goal. However, constructing an effective PNN requires meticulous design of hidden and output layers, as well as careful definition of prior and posterior probability distributions for network weights and biases.
A key finding of this study is that the uncertainty provided by the PNN model in VTEC estimates may be systematically underestimated. In low-latitude areas, the actual error was observed to be as much as twice the model's estimate. This underestimation is expected to be more pronounced during solar maximum, correlating with increased VTEC values. 

**Abstract (ZH)**: 传统神经网络在预测建模中的一个基本局限性是它们无法量化输出的不确定性。在像定位系统这样关键的应用中，理解预测的可靠性对于构建置信区间、早期预警系统和有效传播结果至关重要。例如，卫星导航中的精确点定位严重依赖于辅助数据（轨道、时钟、电离层和对流层）的精确错误模型以计算精确的误差估计。此外，在关键安全应用中，这些不确定性估计是建立稳健防护水平所必需的。

为了应对这一挑战，本文的主要目标是探索一种潜在框架，该框架能够提供电离层垂直总电子含量（VTEC）的点估计及其相关不确定性度量。在此背景下，概率神经网络（PNNs）提供了一种实现这一目标的有前景的方法。然而，构建有效的PNN需要精细设计隐藏层和输出层，并仔细定义网络权重和偏置的先验和后验概率分布。

本研究的一个重要发现是，PNN模型提供的VTEC估计的不确定性可能系统性地被低估。在低纬度地区，实际误差观察到的值可能是模型估计值的两倍。这种低估在太阳极大期更为明显，与VTEC值的增加相关。 

---
# ULTHO: Ultra-Lightweight yet Efficient Hyperparameter Optimization in Deep Reinforcement Learning 

**Title (ZH)**: ULTHO: 超轻量且高效的深度强化学习超参数优化 

**Authors**: Mingqi Yuan, Bo Li, Xin Jin, Wenjun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2503.06101)  

**Abstract**: Hyperparameter optimization (HPO) is a billion-dollar problem in machine learning, which significantly impacts the training efficiency and model performance. However, achieving efficient and robust HPO in deep reinforcement learning (RL) is consistently challenging due to its high non-stationarity and computational cost. To tackle this problem, existing approaches attempt to adapt common HPO techniques (e.g., population-based training or Bayesian optimization) to the RL scenario. However, they remain sample-inefficient and computationally expensive, which cannot facilitate a wide range of applications. In this paper, we propose ULTHO, an ultra-lightweight yet powerful framework for fast HPO in deep RL within single runs. Specifically, we formulate the HPO process as a multi-armed bandit with clustered arms (MABC) and link it directly to long-term return optimization. ULTHO also provides a quantified and statistical perspective to filter the HPs efficiently. We test ULTHO on benchmarks including ALE, Procgen, MiniGrid, and PyBullet. Extensive experiments demonstrate that the ULTHO can achieve superior performance with simple architecture, contributing to the development of advanced and automated RL systems. 

**Abstract (ZH)**: 深度强化学习中的超参数优化：ULTHO——一种轻量而强大的单次运行快速超参数优化框架 

---
# ZO-DARTS++: An Efficient and Size-Variable Zeroth-Order Neural Architecture Search Algorithm 

**Title (ZH)**: ZO-DARTS++: 一种高效可变大小的零阶神经架构搜索算法 

**Authors**: Lunchen Xie, Eugenio Lomurno, Matteo Gambella, Danilo Ardagna, Manual Roveri, Matteo Matteucci, Qingjiang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.06092)  

**Abstract**: Differentiable Neural Architecture Search (NAS) provides a promising avenue for automating the complex design of deep learning (DL) models. However, current differentiable NAS methods often face constraints in efficiency, operation selection, and adaptability under varying resource limitations. We introduce ZO-DARTS++, a novel NAS method that effectively balances performance and resource constraints. By integrating a zeroth-order approximation for efficient gradient handling, employing a sparsemax function with temperature annealing for clearer and more interpretable architecture distributions, and adopting a size-variable search scheme for generating compact yet accurate architectures, ZO-DARTS++ establishes a new balance between model complexity and performance. In extensive tests on medical imaging datasets, ZO-DARTS++ improves the average accuracy by up to 1.8\% over standard DARTS-based methods and shortens search time by approximately 38.6\%. Additionally, its resource-constrained variants can reduce the number of parameters by more than 35\% while maintaining competitive accuracy levels. Thus, ZO-DARTS++ offers a versatile and efficient framework for generating high-quality, resource-aware DL models suitable for real-world medical applications. 

**Abstract (ZH)**: 不同的神经架构搜索（NAS）为自动化深度学习（DL）模型的复杂设计提供了有希望的途径。然而，当前的可微分NAS方法在效率、操作选择和适应不同资源限制方面经常面临挑战。我们引入ZO-DARTS++，这是一种有效平衡性能与资源约束的新NAS方法。通过整合零阶近似以提高梯度处理效率，采用带有温度退火的sparsemax函数以获得更清晰和可解释的架构分布，并采用可变大小的搜索方案以生成紧凑而准确的架构，ZO-DARTS++在模型复杂性和性能之间建立了一个新的平衡。在广泛的医疗成像数据集测试中，ZO-DARTS++的平均准确性比基于标准DARTS的方法提高了1.8%，搜索时间缩短约38.6%。此外，其资源受限变体可以在参数数量减少超过35%的同时保持竞争力的准确性水平。因此，ZO-DARTS++提供了一个适用于实际医疗应用的多功能和高效框架，用于生成高质量的资源感知DL模型。 

---
# Vairiational Stochastic Games 

**Title (ZH)**: 变分随机博弈 

**Authors**: Zhiyu Zhao, Haifeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06037)  

**Abstract**: The Control as Inference (CAI) framework has successfully transformed single-agent reinforcement learning (RL) by reframing control tasks as probabilistic inference problems. However, the extension of CAI to multi-agent, general-sum stochastic games (SGs) remains underexplored, particularly in decentralized settings where agents operate independently without centralized coordination. In this paper, we propose a novel variational inference framework tailored to decentralized multi-agent systems. Our framework addresses the challenges posed by non-stationarity and unaligned agent objectives, proving that the resulting policies form an $\epsilon$-Nash equilibrium. Additionally, we demonstrate theoretical convergence guarantees for the proposed decentralized algorithms. Leveraging this framework, we instantiate multiple algorithms to solve for Nash equilibrium, mean-field Nash equilibrium, and correlated equilibrium, with rigorous theoretical convergence analysis. 

**Abstract (ZH)**: CAI框架下的控制即推理在分布式多智能体系统的扩展研究：非平稳性和对齐问题的处理及理论收敛性分析 

---
# Towards Improving Reward Design in RL: A Reward Alignment Metric for RL Practitioners 

**Title (ZH)**: 提高强化学习中奖励设计：强化学习实践者的奖励对齐度量方法 

**Authors**: Calarina Muslimani, Kerrick Johnstonbaugh, Suyog Chandramouli, Serena Booth, W. Bradley Knox, Matthew E. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2503.05996)  

**Abstract**: Reinforcement learning agents are fundamentally limited by the quality of the reward functions they learn from, yet reward design is often overlooked under the assumption that a well-defined reward is readily available. However, in practice, designing rewards is difficult, and even when specified, evaluating their correctness is equally problematic: how do we know if a reward function is correctly specified? In our work, we address these challenges by focusing on reward alignment -- assessing whether a reward function accurately encodes the preferences of a human stakeholder. As a concrete measure of reward alignment, we introduce the Trajectory Alignment Coefficient to quantify the similarity between a human stakeholder's ranking of trajectory distributions and those induced by a given reward function. We show that the Trajectory Alignment Coefficient exhibits desirable properties, such as not requiring access to a ground truth reward, invariance to potential-based reward shaping, and applicability to online RL. Additionally, in an 11 -- person user study of RL practitioners, we found that access to the Trajectory Alignment Coefficient during reward selection led to statistically significant improvements. Compared to relying only on reward functions, our metric reduced cognitive workload by 1.5x, was preferred by 82% of users and increased the success rate of selecting reward functions that produced performant policies by 41%. 

**Abstract (ZH)**: 强化学习代理受其所学习的奖励函数质量的限制，但在假设奖励定义明确的情况下，奖励设计往往被忽视。实际上，设计奖励具有挑战性，即使给出了奖励规格，评估其正确性同样困难：我们如何确定一个奖励函数是否正确地编码了人类相关方的偏好？在我们的工作中，我们通过关注奖励对齐（reward alignment）来应对这些挑战——评估奖励函数是否准确地编码了人类相关方的偏好。作为奖励对齐的 concrete 度量，我们引入了轨迹对齐系数来量化人类相关方对轨迹分布的排名与由给定奖励函数诱导的排名之间的相似性。我们证明，轨迹对齐系数具有良好的性质，如无需访问 ground truth 奖励、对基于势的奖励塑造不变以及适用于在线 RL。此外，在的一项涉及11名RL实践者的用户研究中，我们发现，在奖励选择过程中使用轨迹对齐系数带来了统计显著性的改进。与仅依赖奖励函数相比，我们的指标使认知负担减少了1.5倍，82%的用户更偏好它，并且将选择产生高性能策略的奖励函数的成功率提高了41%。 

---
# Black Box Causal Inference: Effect Estimation via Meta Prediction 

**Title (ZH)**: 黑箱因果推断：通过元预测估计效应 

**Authors**: Lucius E.J. Bynum, Aahlad Manas Puli, Diego Herrero-Quevedo, Nhi Nguyen, Carlos Fernandez-Granda, Kyunghyun Cho, Rajesh Ranganath  

**Link**: [PDF](https://arxiv.org/pdf/2503.05985)  

**Abstract**: Causal inference and the estimation of causal effects plays a central role in decision-making across many areas, including healthcare and economics. Estimating causal effects typically requires an estimator that is tailored to each problem of interest. But developing estimators can take significant effort for even a single causal inference setting. For example, algorithms for regression-based estimators, propensity score methods, and doubly robust methods were designed across several decades to handle causal estimation with observed confounders. Similarly, several estimators have been developed to exploit instrumental variables (IVs), including two-stage least-squares (TSLS), control functions, and the method-of-moments. In this work, we instead frame causal inference as a dataset-level prediction problem, offloading algorithm design to the learning process. The approach we introduce, called black box causal inference (BBCI), builds estimators in a black-box manner by learning to predict causal effects from sampled dataset-effect pairs. We demonstrate accurate estimation of average treatment effects (ATEs) and conditional average treatment effects (CATEs) with BBCI across several causal inference problems with known identification, including problems with less developed estimators. 

**Abstract (ZH)**: 因果推断和因果效应的估计在医疗保健和经济学等多个领域中的决策制定中发挥着核心作用。因果效应的估计通常需要针对每个感兴趣的问题量身定制的估计器。然而，即使对于单个因果推断设置，开发估计器也可能需要大量努力。例如，基于回归的估计器算法、倾向得分方法和双重鲁棒方法是在多个世纪设计出来的，以处理带有观测混杂因素的因果估计问题。同样，已经开发出了利用工具变量的方法，包括两阶段最小二乘法（TSLS）、控制函数方法和矩方法。在此项工作中，我们相反地将因果推断框架化为数据集级的预测问题，将算法设计的任务交给学习过程。我们引入的方法称为黑盒因果推断（BBCI），通过学习预测采样数据集-效应对的因果效应，以黑盒方式构建估计器。我们展示了在多个已知识别的因果推理问题（包括那些缺乏成熟估计器的问题）中，使用BBCI准确估计平均处理效应（ATE）和条件平均处理效应（CATE）。 

---
# Learning-Order Autoregressive Models with Application to Molecular Graph Generation 

**Title (ZH)**: 学习顺序自回归模型及其在分子图生成中的应用 

**Authors**: Zhe Wang, Jiaxin Shi, Nicolas Heess, Arthur Gretton, Michalis K. Titsias  

**Link**: [PDF](https://arxiv.org/pdf/2503.05979)  

**Abstract**: Autoregressive models (ARMs) have become the workhorse for sequence generation tasks, since many problems can be modeled as next-token prediction. While there appears to be a natural ordering for text (i.e., left-to-right), for many data types, such as graphs, the canonical ordering is less obvious. To address this problem, we introduce a variant of ARM that generates high-dimensional data using a probabilistic ordering that is sequentially inferred from data. This model incorporates a trainable probability distribution, referred to as an \emph{order-policy}, that dynamically decides the autoregressive order in a state-dependent manner. To train the model, we introduce a variational lower bound on the exact log-likelihood, which we optimize with stochastic gradient estimation. We demonstrate experimentally that our method can learn meaningful autoregressive orderings in image and graph generation. On the challenging domain of molecular graph generation, we achieve state-of-the-art results on the QM9 and ZINC250k benchmarks, evaluated using the Fréchet ChemNet Distance (FCD). 

**Abstract (ZH)**: 自回归模型（ARMs）已成为序列生成任务的主力模型，因为许多问题可以建模为下一个token的预测。尽管文本具有自然的顺序（即从左到右），但对于诸如图等许多数据类型，其经典的顺序并不明显。为解决这一问题，我们引入了一种自回归模型的变体，该模型使用从数据中顺序推断出的概率顺序生成高维度数据。该模型嵌入了一个可训练的概率分布，称为\emph{order-policy}，它以状态依赖的方式动态决定自回归顺序。为了训练该模型，我们引入了一个变分下界来优化精确对数似然，采用基于梯度的随机优化方法。实验结果表明，我们的方法可以在图像和图的生成中学习到有意义的自回归顺序。在分子图生成这一具有挑战性的领域中，我们在QM9和ZINC250k基准测试中使用Fréchet ChemNet距离（FCD）取得了最先进的结果。 

---
# A Real-time Multimodal Transformer Neural Network-powered Wildfire Forecasting System 

**Title (ZH)**: 实时多模态变压器神经网络驱动的 wildfire 预测系统 

**Authors**: Qijun Chen, Shaofan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.05971)  

**Abstract**: Due to climate change, the extreme wildfire has become one of the most dangerous natural hazards to human civilization. Even though, some wildfires may be initially caused by human activity, but the spread of wildfires is mainly determined by environmental factors, for examples, (1) weather conditions such as temperature, wind direction and intensity, and moisture levels; (2) the amount and types of dry vegetation in a local area, and (3) topographic or local terrian conditions, which affects how much rain an area gets and how fire dynamics will be constrained or faciliated. Thus, to accurately forecast wildfire occurrence has become one of most urgent and taunting environmental challenges in global scale. In this work, we developed a real-time Multimodal Transformer Neural Network Machine Learning model that combines several advanced artificial intelligence techniques and statistical methods to practically forecast the occurrence of wildfire at the precise location in real time, which not only utilizes large scale data information such as hourly weather forecasting data, but also takes into account small scale topographical data such as local terrain condition and local vegetation conditions collecting from Google Earth images to determine the probabilities of wildfire occurrence location at small scale as well as their timing synchronized with weather forecast information. By using the wildfire data in the United States from 1992 to 2015 to train the multimodal transformer neural network, it can predict the probabilities of wildfire occurrence according to the real-time weather forecast and the synchronized Google Earth image data to provide the wildfire occurrence probability in any small location ($100m^2$) within 24 hours ahead. 

**Abstract (ZH)**: 由于气候变化，极端森林火灾已成为人类文明面临的最危险的自然灾难之一。尽管一些森林火灾最初可能由人类活动引起，但森林火灾的蔓延主要由环境因素决定，例如（1）温度、风向和强度等气象条件；（2）当地可燃干植被的数量和类型；（3）地形或当地地形条件，这些条件影响一个区域的降雨量并制约或促进火灾动态。因此，准确预报森林火灾的发生已成为全球最紧迫和具挑战性的环境挑战之一。在本研究中，我们开发了一种实时多模态变压器神经网络机器学习模型，结合了多种先进的人工智能技术和统计方法，以实时准确预测特定地点的森林火灾的发生，该模型不仅利用了小时级别的气象预报数据，还考虑了从Google Earth图像收集的局部地形条件和植被条件，以确定局部小范围内森林火灾发生概率及其与天气预报信息同步的时间。通过使用1992年至2015年美国的森林火灾数据训练多模态变压器神经网络，该模型可以根据实时气象预报和同步的Google Earth图像数据，在24小时内提前预测任何小范围内（100平方米）的森林火灾发生概率。 

---
# Explaining the Unexplainable: A Systematic Review of Explainable AI in Finance 

**Title (ZH)**: 解释无法解释的：金融领域可解释AI的系统综述 

**Authors**: Md Talha Mohsin, Nabid Bin Nasim  

**Link**: [PDF](https://arxiv.org/pdf/2503.05966)  

**Abstract**: Practitioners and researchers trying to strike a balance between accuracy and transparency center Explainable Artificial Intelligence (XAI) at the junction of finance. This paper offers a thorough overview of the changing scene of XAI applications in finance together with domain-specific implementations, methodological developments, and trend mapping of research. Using bibliometric and content analysis, we find topic clusters, significant research, and most often used explainability strategies used in financial industries. Our results show a substantial dependence on post-hoc interpretability techniques; attention mechanisms, feature importance analysis and SHAP are the most often used techniques among them. This review stresses the need of multidisciplinary approaches combining financial knowledge with improved explainability paradigms and exposes important shortcomings in present XAI systems. 

**Abstract (ZH)**: 从业者和研究者致力于在准确性和透明度之间寻求平衡，将可解释的人工智能（XAI）置于金融领域。本文提供了XAI在金融领域应用的全面 overview，包括领域特定的实施、方法论发展和研究趋势映射。通过文献计量和内容分析，我们找到了主题群簇、重要研究和在金融行业中最常使用的技术。研究结果表明，这些技术中对事后可解释性方法的依赖程度很大；其中最常用的技术包括注意力机制、特征重要性分析和SHAP。本文强调了结合金融知识和改进的可解释性范式的多学科方法的需求，并揭示了现有XAI系统的诸多不足。 

---
# Uncertainty Quantification From Scaling Laws in Deep Neural Networks 

**Title (ZH)**: 深度神经网络中尺度定律下的不确定性量化 

**Authors**: Ibrahim Elsharkawy, Yonatan Kahn, Benjamin Hooberman  

**Link**: [PDF](https://arxiv.org/pdf/2503.05938)  

**Abstract**: Quantifying the uncertainty from machine learning analyses is critical to their use in the physical sciences. In this work we focus on uncertainty inherited from the initialization distribution of neural networks. We compute the mean $\mu_{\mathcal{L}}$ and variance $\sigma_{\mathcal{L}}^2$ of the test loss $\mathcal{L}$ for an ensemble of multi-layer perceptrons (MLPs) with neural tangent kernel (NTK) initialization in the infinite-width limit, and compare empirically to the results from finite-width networks for three example tasks: MNIST classification, CIFAR classification and calorimeter energy regression. We observe scaling laws as a function of training set size $N_\mathcal{D}$ for both $\mu_{\mathcal{L}}$ and $\sigma_{\mathcal{L}}$, but find that the coefficient of variation $\epsilon_{\mathcal{L}} \equiv \sigma_{\mathcal{L}}/\mu_{\mathcal{L}}$ becomes independent of $N_\mathcal{D}$ at both infinite and finite width for sufficiently large $N_\mathcal{D}$. This implies that the coefficient of variation of a finite-width network may be approximated by its infinite-width value, and may in principle be calculable using finite-width perturbation theory. 

**Abstract (ZH)**: 机器学习分析中的不确定性量化对于它们在物理科学中的应用至关重要。本工作中，我们关注继承自神经网络初始化分布的不确定性。我们计算了在神经 tangent 核（NTK）初始化下的多层感知机（MLPs）无穷宽度极限下的测试损失值 $\mathcal{L}$ 的均值 $\mu_{\mathcal{L}}$ 和方差 $\sigma_{\mathcal{L}}^2$，并与三类示例任务（MNIST分类、CIFAR分类和 calorimeter 能量回归）中的有限宽度网络结果进行了empirical比较。我们观察到 $\mu_{\mathcal{L}}$ 和 $\sigma_{\mathcal{L}}$ 随训练集大小 $N_\mathcal{D}$ 的函数关系存在缩放规律，但发现当 $N_\mathcal{D}$ 足够大时，变量 $\epsilon_{\mathcal{L}} \equiv \sigma_{\mathcal{L}}/\mu_{\mathcal{L}}$ 在无穷宽度和有限宽度下均独立于 $N_\mathcal{D}$。这表明对于有限宽度网络，其变异系数可以近似为无穷宽度下的值，并且原则上可以使用有限宽度的微扰理论进行计算。 

---
# The Unified Control Framework: Establishing a Common Foundation for Enterprise AI Governance, Risk Management and Regulatory Compliance 

**Title (ZH)**: 统一控制框架：建立企业AI治理、风险管理及合规性监管的共同基础 

**Authors**: Ian W. Eisenberg, Lucía Gamboa, Eli Sherman  

**Link**: [PDF](https://arxiv.org/pdf/2503.05937)  

**Abstract**: The rapid adoption of AI systems presents enterprises with a dual challenge: accelerating innovation while ensuring responsible governance. Current AI governance approaches suffer from fragmentation, with risk management frameworks that focus on isolated domains, regulations that vary across jurisdictions despite conceptual alignment, and high-level standards lacking concrete implementation guidance. This fragmentation increases governance costs and creates a false dichotomy between innovation and responsibility. We propose the Unified Control Framework (UCF): a comprehensive governance approach that integrates risk management and regulatory compliance through a unified set of controls. The UCF consists of three key components: (1) a comprehensive risk taxonomy synthesizing organizational and societal risks, (2) structured policy requirements derived from regulations, and (3) a parsimonious set of 42 controls that simultaneously address multiple risk scenarios and compliance requirements. We validate the UCF by mapping it to the Colorado AI Act, demonstrating how our approach enables efficient, adaptable governance that scales across regulations while providing concrete implementation guidance. The UCF reduces duplication of effort, ensures comprehensive coverage, and provides a foundation for automation, enabling organizations to achieve responsible AI governance without sacrificing innovation speed. 

**Abstract (ZH)**: AI系统快速采纳给企业带来了双重挑战：加速创新的同时确保负责任的治理。现有的AI治理方法存在碎片化问题，风险管理框架专注于孤立领域，尽管有概念上的统一，但由于管辖区域不同而存在差异，高层次的标准缺乏具体实施指南。这种碎片化增加了治理成本，并在创新与责任之间制造了虚假对立。我们提出统一控制框架（UCF）：一种综合的治理方法，通过一套统一的控制措施将风险管理与合规要求整合起来。UCF包括三个关键组成部分：（1）综合风险分类，综合组织与社会风险，（2）结构化的政策要求，源自法规，（3）简化的42项控制措施，可以同时应对多种风险场景和合规要求。我们通过将其映射到科罗拉多AI法案，验证了UCF的有效性，证明了我们的方法能够实现高效、灵活的治理，跨越不同法规扩展，并提供具体实施指南。UCF减少了重复工作，确保了全面覆盖，并为自动化奠定了基础，从而使组织在不牺牲创新速度的情况下实现负责任的AI治理。 

---
# ElementaryNet: A Non-Strategic Neural Network for Predicting Human Behavior in Normal-Form Games 

**Title (ZH)**: ElementaryNet：一种用于预测正常形博弈中人类行为的非策略神经网络 

**Authors**: Greg d'Eon, Hala Murad, Kevin Leyton-Brown, James R. Wright  

**Link**: [PDF](https://arxiv.org/pdf/2503.05925)  

**Abstract**: Models of human behavior in game-theoretic settings often distinguish between strategic behavior, in which a player both reasons about how others will act and best responds to these beliefs, and "level-0" non-strategic behavior, in which they do not respond to explicit beliefs about others. The state of the art for predicting human behavior on unrepeated simultaneous-move games is GameNet, a neural network that learns extremely complex level-0 specifications from data. The current paper makes three contributions. First, it shows that GameNet's level-0 specifications are too powerful, because they are capable of strategic reasoning. Second, it introduces a novel neural network architecture (dubbed ElementaryNet) and proves that it is only capable of nonstrategic behavior. Third, it describes an extensive experimental evaluation of ElementaryNet. Our overall findings are that (1) ElementaryNet dramatically underperforms GameNet when neither model is allowed to explicitly model higher level agents who best-respond to the model's predictions, indicating that good performance on our dataset requires a model capable of strategic reasoning; (2) that the two models achieve statistically indistinguishable performance when such higher-level agents are introduced, meaning that ElementaryNet's restriction to a non-strategic level-0 specification does not degrade model performance; and (3) that this continues to hold even when ElementaryNet is restricted to a set of level-0 building blocks previously introduced in the literature, with only the functional form being learned by the neural network. 

**Abstract (ZH)**: 人类行为在博弈论设置中的模型通常区分策略行为和“水平-0”的非策略行为，并介绍了一种新型神经网络架构及其对重复同时移动博弈中人类行为预测的贡献。 

---
# SAS: Segment Anything Small for Ultrasound -- A Non-Generative Data Augmentation Technique for Robust Deep Learning in Ultrasound Imaging 

**Title (ZH)**: SAS: 用于超声成像中稳健深度学习的超小型区域分割数据增强技术 

**Authors**: Danielle L. Ferreira, Ahana Gangopadhyay, Hsi-Ming Chang, Ravi Soni, Gopal Avinash  

**Link**: [PDF](https://arxiv.org/pdf/2503.05916)  

**Abstract**: Accurate segmentation of anatomical structures in ultrasound (US) images, particularly small ones, is challenging due to noise and variability in imaging conditions (e.g., probe position, patient anatomy, tissue characteristics and pathology). To address this, we introduce Segment Anything Small (SAS), a simple yet effective scale- and texture-aware data augmentation technique designed to enhance the performance of deep learning models for segmenting small anatomical structures in ultrasound images. SAS employs a dual transformation strategy: (1) simulating diverse organ scales by resizing and embedding organ thumbnails into a black background, and (2) injecting noise into regions of interest to simulate varying tissue textures. These transformations generate realistic and diverse training data without introducing hallucinations or artifacts, improving the model's robustness to noise and variability. We fine-tuned a promptable foundation model on a controlled organ-specific medical imaging dataset and evaluated its performance on one internal and five external datasets. Experimental results demonstrate significant improvements in segmentation performance, with Dice score gains of up to 0.35 and an average improvement of 0.16 [95% CI 0.132,0.188]. Additionally, our iterative point prompts provide precise control and adaptive refinement, achieving performance comparable to bounding box prompts with just two points. SAS enhances model robustness and generalizability across diverse anatomical structures and imaging conditions, particularly for small structures, without compromising the accuracy of larger ones. By offering a computationally efficient solution that eliminates the need for extensive human labeling efforts, SAS emerges as a powerful tool for advancing medical image analysis, particularly in resource-constrained settings. 

**Abstract (ZH)**: 准确分割超声图像中小结构的挑战及其解决方案：一种尺度和纹理感知的数据增强技术（SAS） 

---
# Zero-shot Medical Event Prediction Using a Generative Pre-trained Transformer on Electronic Health Records 

**Title (ZH)**: 基于电子健康记录的生成预训练变压器的零-shot 医学事件预测 

**Authors**: Ekaterina Redekop, Zichen Wang, Rushikesh Kulkarni, Mara Pleasure, Aaron Chin, Hamid Reza Hassanzadeh, Brian L. Hill, Melika Emami, William Speier, Corey W. Arnold  

**Link**: [PDF](https://arxiv.org/pdf/2503.05893)  

**Abstract**: Longitudinal data in electronic health records (EHRs) represent an individual`s clinical history through a sequence of codified concepts, including diagnoses, procedures, medications, and laboratory tests. Foundational models, such as generative pre-trained transformers (GPT), can leverage this data to predict future events. While fine-tuning of these models enhances task-specific performance, it is costly, complex, and unsustainable for every target. We show that a foundation model trained on EHRs can perform predictive tasks in a zero-shot manner, eliminating the need for fine-tuning.
This study presents the first comprehensive analysis of zero-shot forecasting with GPT-based foundational models in EHRs, introducing a novel pipeline that formulates medical concept prediction as a generative modeling task. Unlike supervised approaches requiring extensive labeled data, our method enables the model to forecast a next medical event purely from a pretraining knowledge. We evaluate performance across multiple time horizons and clinical categories, demonstrating model`s ability to capture latent temporal dependencies and complex patient trajectories without task supervision.
Model performance for predicting the next medical concept was evaluated using precision and recall metrics, achieving an average top1 precision of 0.614 and recall of 0.524. For 12 major diagnostic conditions, the model demonstrated strong zero-shot performance, achieving high true positive rates while maintaining low false positives.
We demonstrate the power of a foundational EHR GPT model in capturing diverse phenotypes and enabling robust, zero-shot forecasting of clinical outcomes. This capability enhances the versatility of predictive healthcare models and reduces the need for task-specific training, enabling more scalable applications in clinical settings. 

**Abstract (ZH)**: 电子健康记录（EHRs）中的纵向数据通过一系列编码概念（包括诊断、程序、药物和实验室测试）代表个体的临床历史。基于生成预训练变换器（GPT）的骨架模型可以利用这些数据预测未来事件。尽管对这些模型进行微调可以提升特定任务的性能，但对每个目标进行微调是昂贵、复杂的且不可持续的。我们展示了预先在EHRs上训练的骨架模型可以在零样本状态下执行预测任务，从而消除微调的需要。

本研究首次全面分析了基于GPT的骨架模型在EHRs中的零样本预测预报，引入了一种新颖的工作流程，将医疗概念预测转化为生成建模任务。与需要大量标注数据的监督方法不同，我们的方法使模型仅依靠预训练知识来预测即将发生的医疗事件。我们在多个时间跨度和临床类别上评估了模型的性能，展示了模型在无需特定任务监督的情况下捕捉潜在的时间依赖性和复杂患者轨迹的能力。

使用精确度和召回率指标评估了预测下一个医疗概念的模型性能，平均Top1精确度为0.614，召回率为0.524。对于12种主要诊断条件，模型展示了强大的零样本性能，实现了较高的真实阳性率同时保持了较低的假阳性率。

我们展示了基于EHR的GPT骨架模型的力量，能够在捕获多样表型的同时实现临床结果的稳健、零样本预测预报。这种能力增强了预测型医疗保健模型的灵活性，并减少了特定任务训练的需要，从而在临床环境中实现了更可扩展的应用。 

---
# QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation 

**Title (ZH)**: QG-SMS：通过学生建模与仿真增强试题分析 

**Authors**: Bang Nguyen, Tingting Du, Mengxia Yu, Lawrence Angrave, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05888)  

**Abstract**: While the Question Generation (QG) task has been increasingly adopted in educational assessments, its evaluation remains limited by approaches that lack a clear connection to the educational values of test items. In this work, we introduce test item analysis, a method frequently used by educators to assess test question quality, into QG evaluation. Specifically, we construct pairs of candidate questions that differ in quality across dimensions such as topic coverage, item difficulty, item discrimination, and distractor efficiency. We then examine whether existing QG evaluation approaches can effectively distinguish these differences. Our findings reveal significant shortcomings in these approaches with respect to accurately assessing test item quality in relation to student performance. To address this gap, we propose a novel QG evaluation framework, QG-SMS, which leverages Large Language Model for Student Modeling and Simulation to perform test item analysis. As demonstrated in our extensive experiments and human evaluation study, the additional perspectives introduced by the simulated student profiles lead to a more effective and robust assessment of test items. 

**Abstract (ZH)**: 虽然问题生成（QG）任务在教育评估中逐渐得到应用，其评价仍受限于缺乏与测试项目教育价值明确联系的方法。在本文中，我们引入了测试项目分析方法，这是一种教育工作者常用的评估测试问题质量的方法，将其应用于QG评价。具体而言，我们构建了一系列在主题覆盖、项目难度、项目区分度和干扰项效率等方面存在质量差异的候选问题对。然后，我们考察现有QG评价方法是否能够有效地区分这些差异。我们的研究发现，这些方法在准确评估与学生表现相关的测试项目质量方面存在显著不足。为弥补这一不足，我们提出了一种新的QG评价框架QG-SMS，该框架利用大型语言模型进行学生建模与模拟，以实现测试项目分析。如我们在广泛实验和人工评估研究中的演示所示，由模拟学生档案引入的额外视角能够产生更为有效且稳健的测试项目评估。 

---
# Practical Topics in Optimization 

**Title (ZH)**: 优化中的实用话题 

**Authors**: Jun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05882)  

**Abstract**: In an era where data-driven decision-making and computational efficiency are paramount, optimization plays a foundational role in advancing fields such as mathematics, computer science, operations research, machine learning, and beyond. From refining machine learning models to improving resource allocation and designing efficient algorithms, optimization techniques serve as essential tools for tackling complex problems. This book aims to provide both an introductory guide and a comprehensive reference, equipping readers with the necessary knowledge to understand and apply optimization methods within their respective fields.
Our primary goal is to demystify the inner workings of optimization algorithms, including black-box and stochastic optimizers, by offering both formal and intuitive explanations. Starting from fundamental mathematical principles, we derive key results to ensure that readers not only learn how these techniques work but also understand when and why to apply them effectively. By striking a careful balance between theoretical depth and practical application, this book serves a broad audience, from students and researchers to practitioners seeking robust optimization strategies. 

**Abstract (ZH)**: 在数据驱动决策和计算效率至关重要的时代，最优化在数学、计算机科学、运筹学、机器学习等领域的发展中起着基础性作用。从完善机器学习模型到改进资源分配和设计高效算法，最优化技术是解决复杂问题的重要工具。本书旨在提供一个入门指导和全面参考，帮助读者掌握在各自领域内理解和应用最优化方法的必要知识。
我们的主要目标是通过提供形式化和直观的解释，揭开最优化算法内部运作的神秘面纱，包括黑盒和随机最优化器。从基本的数学原理出发，我们推导出关键结果，确保读者不仅了解这些技术如何工作，还知道在何时和为何有效地应用它们。通过在理论深度和实际应用之间取得平衡，本书服务于广泛的读者群体，包括学生、研究人员和寻求稳健最优化策略的实践者。 

---
# Benchmarking AI Models in Software Engineering: A Review, Search Tool, and Enhancement Protocol 

**Title (ZH)**: 软件工程中AI模型的基准测试：综述、搜索工具及优化协议 

**Authors**: Roham Koohestani, Philippe de Bekker, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2503.05860)  

**Abstract**: Benchmarks are essential for consistent evaluation and reproducibility. The integration of Artificial Intelligence into Software Engineering (AI4SE) has given rise to numerous benchmarks for tasks such as code generation and bug fixing. However, this surge presents challenges: (1) scattered benchmark knowledge across tasks, (2) difficulty in selecting relevant benchmarks, (3) the absence of a uniform standard for benchmark development, and (4) limitations of existing benchmarks. In this paper, we review 173 studies and identify 204 AI4SE benchmarks. We classify these benchmarks, analyze their limitations, and expose gaps in practices. Based on our review, we created BenchScout, a semantic search tool to find relevant benchmarks, using automated clustering of the contexts from associated studies. We conducted a user study with 22 participants to evaluate BenchScout's usability, effectiveness, and intuitiveness which resulted in average scores of 4.5, 4.0, and 4.1 out of 5. To advance benchmarking standards, we propose BenchFrame, a unified method to enhance benchmark quality. As a case study, we applied BenchFrame to the HumanEval benchmark and addressed its main limitations. This led to HumanEvalNext, featuring (1) corrected errors, (2) improved language conversion, (3) expanded test coverage, and (4) increased difficulty. We then evaluated ten state-of-the-art code language models on HumanEval, HumanEvalPlus, and HumanEvalNext. On HumanEvalNext, models showed a pass@1 score reduction of 31.22% and 19.94% compared to HumanEval and HumanEvalPlus, respectively. 

**Abstract (ZH)**: 基准对于一致的评估和可再现性至关重要。将人工智能融入软件工程（AI4SE）已为代码生成和漏洞修复等任务产生了众多基准。然而，这一发展带来了挑战：（1）任务之间的基准知识分散，（2）难以选择相关的基准，（3）缺乏统一的基准开发标准，以及（4）现有基准的局限性。本文审查了173篇研究，确定了204个AI4SE基准。我们对这些基准进行了分类，分析了其局限性，揭示了实践中的空白。基于我们的审查，我们创建了BenchScout，这是一种语义搜索工具，用于通过与关联研究相关的上下文的自动聚类来查找相关基准。我们进行了22名参与者的研究，评估了BenchScout的可用性、有效性和直观性，其平均得分为4.5、4.0和4.1（满分5分）。为了提高基准标准，我们提出了一种统一方法BenchFrame，以提高基准质量。作为案例研究，我们将BenchFrame应用于HumanEval基准，并解决了其主要局限性，从而产生了HumanEvalNext，其特点包括：（1）修正错误，（2）改进语言转换，（3）扩展测试覆盖范围，（4）增加难度。然后，我们在HumanEval、HumanEvalPlus和HumanEvalNext上评估了10个最先进的代码语言模型。在HumanEvalNext上，模型在pass@1得分上分别比HumanEval和HumanEvalPlus降低了31.22%和19.94%。 

---
# SYMBIOSIS: Systems Thinking and Machine Intelligence for Better Outcomes in Society 

**Title (ZH)**: 共生：系统思维与机器智能以实现更好的社会成果 

**Authors**: Sameer Sethi, Donald Martin Jr., Emmanuel Klu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05857)  

**Abstract**: This paper presents SYMBIOSIS, an AI-powered framework and platform designed to make Systems Thinking accessible for addressing societal challenges and unlock paths for leveraging systems thinking frameworks to improve AI systems. The platform establishes a centralized, open-source repository of systems thinking/system dynamics models categorized by Sustainable Development Goals (SDGs) and societal topics using topic modeling and classification techniques. Systems Thinking resources, though critical for articulating causal theories in complex problem spaces, are often locked behind specialized tools and intricate notations, creating high barriers to entry. To address this, we developed a generative co-pilot that translates complex systems representations - such as causal loop and stock-flow diagrams - into natural language (and vice-versa), allowing users to explore and build models without extensive technical training.
Rooted in community-based system dynamics (CBSD) and informed by community-driven insights on societal context, we aim to bridge the problem understanding chasm. This gap, driven by epistemic uncertainty, often limits ML developers who lack the community-specific knowledge essential for problem understanding and formulation, often leading to ill informed causal assumptions, reduced intervention effectiveness and harmful biases. Recent research identifies causal and abductive reasoning as crucial frontiers for AI, and Systems Thinking provides a naturally compatible framework for both. By making Systems Thinking frameworks more accessible and user-friendly, SYMBIOSIS aims to serve as a foundational step to unlock future research into responsible and society-centered AI. Our work underscores the need for ongoing research into AI's capacity to understand essential characteristics of complex adaptive systems paving the way for more socially attuned, effective AI systems. 

**Abstract (ZH)**: 基于AI的SYMBIOSIS框架与平台：使系统思考普及以应对社会挑战并改善AI系统 

---
# Machine Learned Force Fields: Fundamentals, its reach, and challenges 

**Title (ZH)**: 机器学习力场：原理、应用及其挑战 

**Authors**: Carlos A. Vital, Román J. Armenta-Rico, Huziel E. Sauceda  

**Link**: [PDF](https://arxiv.org/pdf/2503.05845)  

**Abstract**: Highly accurate force fields are a mandatory requirement to generate predictive simulations. In this regard, Machine Learning Force Fields (MLFFs) have emerged as a revolutionary approach in computational chemistry and materials science, combining the accuracy of quantum mechanical methods with computational efficiency orders of magnitude superior to ab-initio methods. This chapter provides an introduction of the fundamentals of learning and how it is applied to construct MLFFs, detailing key methodologies such as neural network potentials and kernel-based models. Emphasis is placed on the construction of SchNet model, as one of the most elemental neural network-based force fields that are nowadays the basis of modern architectures. Additionally, the GDML framework is described in detail as an example of how the elegant formulation of kernel methods can be used to construct mathematically robust and physics-inspired MLFFs. The ongoing advancements in MLFF development continue to expand their applicability, enabling precise simulations of large and complex systems that were previously beyond reach. This chapter concludes by highlighting the transformative impact of MLFFs on scientific research, underscoring their role in driving future discoveries in the fields of chemistry, physics, and materials science. 

**Abstract (ZH)**: 高精度势场是生成预测性模拟的必备要求。在这方面，机器学习势场（MLFFs）已成为计算化学和材料科学中的革命性方法，结合了量子力学方法的准确性，并且在计算效率上比从头算方法高出了数量级。本章介绍了学习的基本原理及其在构建MLFFs中的应用，并详细阐述了关键方法，如神经网络势和核基模型。重点介绍了SchNet模型，作为一种基础的基于神经网络的势场，现已成为现代架构的基础。此外，详细描述了GDML框架，说明了如何使用核方法的优美表达式构建数学上稳健且受物理启发的MLFFs。MLFF发展的持续进步不断扩展其适用范围，使以前无法实现的大而复杂的系统模拟变得精确。本章最后强调了MLFFs对科学研究的颠覆性影响，突出了它们在推动化学、物理和材料科学领域未来发现中的作用。 

---
# AI-Facilitated Collective Judgements 

**Title (ZH)**: AI促进的集体判断 

**Authors**: Manon Revel, Théophile Pénigaud  

**Link**: [PDF](https://arxiv.org/pdf/2503.05830)  

**Abstract**: This article unpacks the design choices behind longstanding and newly proposed computational frameworks aimed at finding common grounds across collective preferences and examines their potential future impacts, both technically and normatively. It begins by situating AI-assisted preference elicitation within the historical role of opinion polls, emphasizing that preferences are shaped by the decision-making context and are seldom objectively captured. With that caveat in mind, we explore AI-facilitated collective judgment as a discovery tool for fostering reasonable representations of a collective will, sense-making, and agreement-seeking. At the same time, we caution against dangerously misguided uses, such as enabling binding decisions, fostering gradual disempowerment or post-rationalizing political outcomes. 

**Abstract (ZH)**: 本文拆解了旨在跨越集体偏好的共识寻找长久以来及新提出的计算框架背后的设计选择，并探讨了它们在技术和规范层面的潜在未来影响。本文首先将AI辅助的偏好 elicitation 放在历史意见调查的角色中，强调偏好是由决策环境塑造的，很少能够客观捕捉到。在此前提下，我们探讨了AI促进的集体判断作为促进合理反映集体意志、意义建构和寻求共识的发现工具。同时，我们警告避免危险的误用，如使决策具有约束力、逐渐剥夺权力或事后合理化政治结果。 

---
# The impact of AI and peer feedback on research writing skills: a study using the CGScholar platform among Kazakhstani scholars 

**Title (ZH)**: AI和同伴反馈对研究写作技能的影响：使用CGScholar平台的哈萨克斯坦学者研究 

**Authors**: Raigul Zheldibayeva  

**Link**: [PDF](https://arxiv.org/pdf/2503.05820)  

**Abstract**: This research studies the impact of AI and peer feedback on the academic writing development of Kazakhstani scholars using the CGScholar platform - a product of research into collaborative learning, big data, and artificial intelligence developed by educators and computer scientists at the University of Illinois at Urbana-Champaign (UIUC). The study aimed to find out how familiarity with AI tools and peer feedback processes impacts participants' openness to incorporating feedback into their academic writing. The study involved 36 scholars enrolled in a scientific internship focused on education at UIUC. A survey with 15 multiple-choice questions, a Likert scale, and open-ended questions was used to collect data. The survey was conducted via Google Forms in both English and Russian to ensure linguistic accessibility. Demographic information such as age, gender, and first language was collected to provide a detailed understanding of the data. The analysis revealed a moderate positive correlation between familiarity with AI tools and openness to making changes based on feedback, and a strong positive correlation between research writing experience and expectations of peer feedback, especially in the area of research methodology. These results show that participants are open-minded to AI-assisted feedback; however, they still highly appreciate peer input, especially regarding methodological guidance. This study demonstrates the potential benefits of integrating AI tools with traditional feedback mechanisms to improve research writing quality in academic settings. 

**Abstract (ZH)**: 本研究探讨了AI工具和同伴反馈对 UIBK 平台上 Kazakhstan 学者学术写作发展的影响——UIUC 教育学家和计算机科学家开发的一种基于协作学习、大数据和人工智能的产品。研究旨在了解熟悉AI工具和同伴反馈过程如何影响参与者在其学术写作中接受反馈的开放程度。研究涉及了36名参与UIUC科学实习的教育方向学者。通过Google Forms使用包含15道多项选择题、李克特量表和开放性问题的问卷收集数据。问卷同时提供英文和俄文版本，以确保语言 accessibility。收集了年龄、性别和第一语言等人口统计信息，以提供详细的数据理解。分析结果显示，熟悉AI工具与基于反馈作出改变的开放性之间存在中等程度的正相关，研究写作经验与期望中的同伴反馈之间存在强烈正相关，特别是在研究方法领域。这些结果表明，参与者对AI辅助反馈持开放态度；然而，他们仍然高度重视同伴反馈，尤其是在方法论指导方面。本研究展示了将AI工具与传统反馈机制集成以提高学术写作质量的潜在好处。 

---
# Will Neural Scaling Laws Activate Jevons' Paradox in AI Labor Markets? A Time-Varying Elasticity of Substitution (VES) Analysis 

**Title (ZH)**: 神经扩展律将激活AI劳动力市场的灰心悖论吗？一种时间 varying 的替代弹性（VES）分析 

**Authors**: Rajesh P. Narayanan, R. Kelley Pace  

**Link**: [PDF](https://arxiv.org/pdf/2503.05816)  

**Abstract**: AI industry leaders often use the term ``Jevons' Paradox.'' We explore the significance of this term for artificial intelligence adoption through a time-varying elasticity of substitution framework. We develop a model connecting AI development to labor substitution through four key mechanisms: (1) increased effective computational capacity from both hardware and algorithmic improvements; (2) AI capabilities that rise logarithmically with computation following established neural scaling laws; (3) declining marginal computational costs leading to lower AI prices through competitive pressure; and (4) a resulting increase in the elasticity of substitution between AI and human labor over time. Our time-varying elasticity of substitution (VES) framework, incorporating the Gørtz identity, yields analytical conditions for market transformation dynamics. This work provides a simple framework to help assess the economic reasoning behind industry claims that AI will increasingly substitute for human labor across diverse economic sectors. 

**Abstract (ZH)**: AI行业领导者经常使用“Jevons' Paradox”这一术语。我们通过时间变化的替代弹性框架探索这一术语对人工智能采纳的重要性。我们发展了一个将人工智能发展与劳动替代联系起来的模型，通过四个关键机制：（1）从硬件和算法改进中增强的有效计算能力；（2）遵循已建立的神经网络缩放定律，AI能力随计算量呈对数增长；（3）边际计算成本下降，导致更具竞争性的市场压力下AI价格降低；以及（4）随着时间推移，AI与人力劳动之间替代弹性的增加。我们的带Gørtz身份的时间变化替代弹性（VES）框架为市场转型动力学提供了分析条件。本研究提供了一个简单框架，以帮助评估行业声称人工智能将在多种经济领域内越来越多地替代人类劳动的经济合理性。 

---
# Trust, Experience, and Innovation: Key Factors Shaping American Attitudes About AI 

**Title (ZH)**: 信任、经验和创新：塑造美国民众对人工智能态度的关键因素 

**Authors**: Risa Palm, Justin Kingsland, Toby Bolsen  

**Link**: [PDF](https://arxiv.org/pdf/2503.05815)  

**Abstract**: A large survey of American adults explored the complex landscape of attitudes towards artificial intelligence (AI). It explored the degree of concern regarding specific potential outcomes of the new advances in AI technology and correlates of these concerns. Key variables associated with the direction and intensity of concern include prior experience using a large language model such as ChatGPT, general trust in science, adherence to the precautionary principle versus support for unrestricted innovation, and demographic factors such as gender. By analyzing these relationships, the paper provides valuable insights into the American public's response to AI that are particularly important in the development of policy to regulate or further encourage its development. 

**Abstract (ZH)**: 一项针对美国成人的大规模调查显示了人们对人工智能（AI）的态度复杂 landscape。该调查探讨了对新AI技术进展可能产生的特定结果的担忧程度及其相关因素。与担忧方向和强度相关的关键变量包括之前使用如ChatGPT这样的大型语言模型的经验、对科学的一般信任度、预防原则与无限制创新的支持程度以及性别等人口统计学因素。通过分析这些关系，该论文提供了关于美国公众对AI的反应的有价值的见解，特别是在制定监管或进一步促进其发展的政策方面尤为重要。 

---
# Intolerable Risk Threshold Recommendations for Artificial Intelligence 

**Title (ZH)**: 不可接受的风险阈值建议：人工智能领域 

**Authors**: Deepika Raman, Nada Madkour, Evan R. Murphy, Krystal Jackson, Jessica Newman  

**Link**: [PDF](https://arxiv.org/pdf/2503.05812)  

**Abstract**: Frontier AI models -- highly capable foundation models at the cutting edge of AI development -- may pose severe risks to public safety, human rights, economic stability, and societal value in the coming years. These risks could arise from deliberate adversarial misuse, system failures, unintended cascading effects, or simultaneous failures across multiple models.
In response to such risks, at the AI Seoul Summit in May 2024, 16 global AI industry organizations signed the Frontier AI Safety Commitments, and 27 nations and the EU issued a declaration on their intent to define these thresholds. To fulfill these commitments, organizations must determine and disclose ``thresholds at which severe risks posed by a model or system, unless adequately mitigated, would be deemed intolerable.''
To assist in setting and operationalizing intolerable risk thresholds, we outline key principles and considerations; for example, to aim for ``good, not perfect'' thresholds in the face of limited data on rapidly advancing AI capabilities and consequently evolving risks. We also propose specific threshold recommendations, including some detailed case studies, for a subset of risks across eight risk categories: (1) Chemical, Biological, Radiological, and Nuclear (CBRN) Weapons, (2) Cyber Attacks, (3) Model Autonomy, (4) Persuasion and Manipulation, (5) Deception, (6) Toxicity, (7) Discrimination, and (8) Socioeconomic Disruption. Our goal is to serve as a starting point or supplementary resource for policymakers and industry leaders, encouraging proactive risk management that prioritizes preventing intolerable risks (ex ante) rather than merely mitigating them after they occur (ex post). 

**Abstract (ZH)**: 前沿AI模型——处于AI开发前沿的高度 capable 基础模型——可能在未来几年对公共安全、人权、经济稳定和社会价值构成严重风险。这些风险可能源于故意恶意使用、系统失效、意外连锁反应或多个模型的同时失效。

响应这些风险，在2024年5月的AI首尔峰会上，16家全球AI行业组织签署了前沿AI安全承诺，27个国家和欧盟发布了旨在界定这些门槛的声明。为了履行这些承诺，组织必须确定并披露“除非得到充分缓解，否则由模型或系统带来的严重风险将被认为是不可接受”的门槛。

为辅助设定和实现不可接受风险门槛，我们概述了关键原则和考虑事项；例如，在有限的数据和迅速发展的AI能力背景下，追求“足够好而非完美”的门槛。我们还提出了具体的门槛建议，包括一些详细的案例研究，针对八类风险中的部分风险，这八类风险包括：（1）化学、生物、放射性和核武器（CBRN武器），（2）网络攻击，（3）模型自主性，（4）说服与操控，（5）欺骗，（6）毒性，（7）歧视，和（8）社会经济破坏。我们的目标是为政策制定者和行业领袖提供一个起点或辅助资源，促进前瞻性风险管理，优先防止不可接受的风险（预防）而非仅仅在风险发生后进行缓解（事后）。 

---
# A Transformer Model for Predicting Chemical Reaction Products from Generic Templates 

**Title (ZH)**: 基于通用模板预测化学反应产物的转换器模型 

**Authors**: Derin Ozer, Sylvain Lamprier, Thomas Cauchy, Nicolas Gutowski, Benoit Da Mota  

**Link**: [PDF](https://arxiv.org/pdf/2503.05810)  

**Abstract**: The accurate prediction of chemical reaction outcomes is a major challenge in computational chemistry. Current models rely heavily on either highly specific reaction templates or template-free methods, both of which present limitations. To address these limitations, this work proposes the Broad Reaction Set (BRS), a dataset featuring 20 generic reaction templates that allow for the efficient exploration of the chemical space. Additionally, ProPreT5 is introduced, a T5 model tailored to chemistry that achieves a balance between rigid templates and template-free methods. ProPreT5 demonstrates its capability to generate accurate, valid, and realistic reaction products, making it a promising solution that goes beyond the current state-of-the-art on the complex reaction product prediction task. 

**Abstract (ZH)**: 化学反应结果的准确预测是计算化学中的一个主要挑战。当前模型要么依赖高度特定的反应模板，要么是非模板方法，两者都存在局限性。为解决这些局限性，本工作提出了广义反应集（BRS），这是一个包含20个通用反应模板的数据集，可以有效地探索化学空间。此外，还引入了ProPreT5模型，这是一种针对化学定制的T5模型，能够在刚性模板和非模板方法之间找到平衡。ProPreT5展示了生成准确、有效且现实反应产物的能力，使其成为在复杂反应产物预测任务中超越当前最先进水平的有前景的解决方案。 

---
# Multi-agent Auto-Bidding with Latent Graph Diffusion Models 

**Title (ZH)**: 基于潜在图扩散模型的多代理自动出价方法 

**Authors**: Dom Huh, Prasant Mohapatra  

**Link**: [PDF](https://arxiv.org/pdf/2503.05805)  

**Abstract**: This paper proposes a diffusion-based auto-bidding framework that leverages graph representations to model large-scale auction environments. In such settings, agents must dynamically optimize bidding strategies under constraints defined by key performance indicator (KPI) metrics, all while operating in competitive environments characterized by uncertain, sparse, and stochastic variables. To address these challenges, we introduce a novel approach combining learnable graph-based embeddings with a planning-based latent diffusion model (LDM). By capturing patterns and nuances underlying the interdependence of impression opportunities and the multi-agent dynamics of the auction environment, the graph representation enable expressive computations regarding auto-bidding outcomes. With reward alignment techniques, the LDM's posterior is fine-tuned to generate auto-bidding trajectories that maximize KPI metrics while satisfying constraint thresholds. Empirical evaluations on both real-world and synthetic auction environments demonstrate significant improvements in auto-bidding performance across multiple common KPI metrics, as well as accuracy in forecasting auction outcomes. 

**Abstract (ZH)**: 基于图表示的大规模拍卖环境自调 bidding 扩散框架 

---
# Federated Learning Framework via Distributed Mutual Learning 

**Title (ZH)**: 分布式互助学习的联邦学习框架 

**Authors**: Yash Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2503.05803)  

**Abstract**: Federated Learning often relies on sharing full or partial model weights, which can burden network bandwidth and raise privacy risks. We present a loss-based alternative using distributed mutual learning. Instead of transmitting weights, clients periodically share their loss predictions on a public test set. Each client then refines its model by combining its local loss with the average Kullback-Leibler divergence over losses from other clients. This collaborative approach both reduces transmission overhead and preserves data privacy. Experiments on a face mask detection task demonstrate that our method outperforms weight-sharing baselines, achieving higher accuracy on unseen data while providing stronger generalization and privacy benefits. 

**Abstract (ZH)**: 基于损失的联邦学习：一种分布式互学习替代方案 

---
# Illuminant and light direction estimation using Wasserstein distance method 

**Title (ZH)**: 使用Wasserstein距离方法估计光源和光照方向 

**Authors**: Selcuk Yazar  

**Link**: [PDF](https://arxiv.org/pdf/2503.05802)  

**Abstract**: Illumination estimation remains a pivotal challenge in image processing, particularly for robotics, where robust environmental perception is essential under varying lighting conditions. Traditional approaches, such as RGB histograms and GIST descriptors, often fail in complex scenarios due to their sensitivity to illumination changes. This study introduces a novel method utilizing the Wasserstein distance, rooted in optimal transport theory, to estimate illuminant and light direction in images. Experiments on diverse images indoor scenes, black-and-white photographs, and night images demonstrate the method's efficacy in detecting dominant light sources and estimating their directions, outperforming traditional statistical methods in complex lighting environments. The approach shows promise for applications in light source localization, image quality assessment, and object detection enhancement. Future research may explore adaptive thresholding and integrate gradient analysis to enhance accuracy, offering a scalable solution for real-world illumination challenges in robotics and beyond. 

**Abstract (ZH)**: 光照估计仍然是图像处理中的一个关键挑战，特别是在机器人领域，需要在不同光照条件下进行鲁棒的环境感知。传统的基于RGB直方图和GIST描述子的方法往往在复杂场景下由于对光照变化的敏感性而失效。本研究提出了一种利用最优传输理论中的 Wasserstein 距离的新方法，用于估计图像中的光照和光的方向。实验结果显示，该方法在室内场景、黑白照片和夜景图像上能够有效地检测主要光源和估计其方向，在复杂光照环境下优于传统统计方法。该方法在光源定位、图像质量评估和目标检测增强等方面具有应用前景。未来的研究可以探索自适应阈值处理并集成梯度分析以提高准确性，提供一种适用于机器人和其他领域的光照挑战的可扩展解决方案。 

---
# Fault Localization and State Estimation of Power Grid under Parallel Cyber-Physical Attacks 

**Title (ZH)**: 电力网络在平行网络物理攻击下的故障定位与状态估计 

**Authors**: Junhao Ren, Kai Zhao, Guangxiao Zhang, Xinghua Liu, Chao Zhai, Gaoxi Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.05797)  

**Abstract**: Parallel cyber-physical attacks (PCPA) refer to those attacks on power grids by disturbing/cutting off physical transmission lines and meanwhile blocking transmission of measurement data to dwarf or delay the system protection and recovery actions. Such fierce hostile attacks impose critical threats to the modern power grids when there is a fusion of power grids and telecommunication technologies. In this paper, we investigate the fault diagnosis problem of faulty transmission lines under a broader spectrum of PCPA for a linearized (or DC) power flow model. The physical attack mechanism of PCPA includes not only disconnection but also admittance value modification on transmission lines, for example, by invading distributed flexible AC transmission system (D-FACTS). To tackle the problem, we first recover the information of voltage phase angles within the attacked area. Using the information of voltage phase angle and power injection of buses, a graph attention network-based fault localization (GAT-FL) algorithm is proposed to find the locations of the physical attacks. By capitalizing on the feature extraction capability of the GAT on graph data, the fault localization algorithm outperforms the existing results when under cyber attacks, e.g., denial of service (DoS) attacks. A line state identification algorithm is then developed to identify the states of the transmission lines within the attacked area. Specifically, the algorithm restores the power injection of buses within the attacked area and then identities the state of all the transmission lines within the attacked area by solving a linear programming (LP) problem. Experimental simulations are effectiveness of the proposed fault diagnosis algorithms. 

**Abstract (ZH)**: 平行 cyber-物理 攻击下的传输线路故障诊断研究（PCPA） 

---
# Towards Multi-Stakeholder Evaluation of ML Models: A Crowdsourcing Study on Metric Preferences in Job-matching System 

**Title (ZH)**: 面向多利益相关方的机器学习模型评估研究：基于工作匹配系统中度量偏好指标的众包研究 

**Authors**: Takuya Yokota, Yuri Nakao  

**Link**: [PDF](https://arxiv.org/pdf/2503.05796)  

**Abstract**: While machine learning (ML) technology affects diverse stakeholders, there is no one-size-fits-all metric to evaluate the quality of outputs, including performance and fairness. Using predetermined metrics without soliciting stakeholder opinions is problematic because it leads to an unfair disregard for stakeholders in the ML pipeline. In this study, to establish practical ways to incorporate diverse stakeholder opinions into the selection of metrics for ML, we investigate participants' preferences for different metrics by using crowdsourcing. We ask 837 participants to choose a better model from two hypothetical ML models in a hypothetical job-matching system twenty times and calculate their utility values for seven metrics. To examine the participants' feedback in detail, we divide them into five clusters based on their utility values and analyze the tendencies of each cluster, including their preferences for metrics and common attributes. Based on the results, we discuss the points that should be considered when selecting appropriate metrics and evaluating ML models with multiple stakeholders. 

**Abstract (ZH)**: 机器学习技术影响多元利益相关者，但缺乏适用于评估输出质量（包括性能和公平性）的一揽子评价指标。在未征求利益相关者意见的情况下使用预设指标会导致对机器学习管道中利益相关者的不公平忽视。为建立将多元利益相关者意见纳入机器学习指标选择的实用方法，本研究通过 crowdsourcing 探究参与者对不同指标的偏好。我们要求 837 名参与者在假设的工作匹配系统中从两个假设的机器学习模型中选择较好的模型二十次，并计算他们对七种指标的效用值。为详细检查参与者反馈，我们根据效用值将参与者分为五类，并分析每类的倾向性，包括其对指标的偏好和共同特征。基于研究结果，我们讨论了在多利益相关者情境下选择合适指标和评估机器学习模型时应注意的要点。 

---
# CBW: Towards Dataset Ownership Verification for Speaker Verification via Clustering-based Backdoor Watermarking 

**Title (ZH)**: CBW：基于聚类后门水印的说话人验证数据集所有权验证方法 

**Authors**: Yiming Li, Kaiying Yan, Shuo Shao, Tongqing Zhai, Shu-Tao Xia, Zhan Qin, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.05794)  

**Abstract**: With the increasing adoption of deep learning in speaker verification, large-scale speech datasets have become valuable intellectual property. To audit and prevent the unauthorized usage of these valuable released datasets, especially in commercial or open-source scenarios, we propose a novel dataset ownership verification method. Our approach introduces a clustering-based backdoor watermark (CBW), enabling dataset owners to determine whether a suspicious third-party model has been trained on a protected dataset under a black-box setting. The CBW method consists of two key stages: dataset watermarking and ownership verification. During watermarking, we implant multiple trigger patterns in the dataset to make similar samples (measured by their feature similarities) close to the same trigger while dissimilar samples are near different triggers. This ensures that any model trained on the watermarked dataset exhibits specific misclassification behaviors when exposed to trigger-embedded inputs. To verify dataset ownership, we design a hypothesis-test-based framework that statistically evaluates whether a suspicious model exhibits the expected backdoor behavior. We conduct extensive experiments on benchmark datasets, verifying the effectiveness and robustness of our method against potential adaptive attacks. The code for reproducing main experiments is available at this https URL 

**Abstract (ZH)**: 随着深度学习在语音识别验证中的应用日益增加，大规模语音数据集已成为有价值的知识产权。为了审计和防止这些 valuable 数据集在未经授权的情况下被使用，尤其是在商业或开源场景中，我们提出了一种新颖的数据集所有权验证方法。该方法引入了一种基于聚类的后门水印（CBW），使数据集所有者能够在黑盒环境中确定可疑第三方模型是否使用了受保护的数据集。CBW 方法包括两个关键阶段：数据集水印和所有权验证。在水印阶段，我们通过植入多个触发模式，使具有相似特征的样本在触发方面接近相同的模式，而具有不同特征的样本则接近不同的模式，从而确保任何基于水印数据集训练的模型在遇到触发器嵌入的输入时会展现出特定的错误分类行为。为了验证数据集所有权，我们设计了一种基于 hypothesis test 的框架，用于统计评估可疑模型是否表现出预期的后门行为。我们在基准数据集上进行了广泛的实验，验证了该方法在潜在适应性攻击下的有效性和鲁棒性。主要实验的代码可在此处访问：[此 https URL]。 

---
# Artificial Intelligence in Sports: Insights from a Quantitative Survey among Sports Students in Germany about their Perceptions, Expectations, and Concerns regarding the Use of AI Tools 

**Title (ZH)**: 体育领域的人工智能：关于人工智能工具在体育中的应用、期望与担忧的德国体育学生定量调研见解 

**Authors**: Dennis Krämer, Anja Bosold, Martin Minarik, Cleo Schyvinck, Andre Hajek  

**Link**: [PDF](https://arxiv.org/pdf/2503.05785)  

**Abstract**: Generative Artificial Intelligence (AI) tools such as ChatGPT, Copilot, or Gemini have a crucial impact on academic research and teaching. Empirical data on how students perceive the increasing influence of AI, which different types of tools they use, what they expect from them in their daily academic tasks, and their concerns regarding the use of AI in their studies are still limited. The manuscript presents findings from a quantitative survey conducted among sports students of all semesters in Germany using an online questionnaire. It explores aspects such as students' usage behavior, motivational factors, and uncertainties regarding the impact of AI tools on academia in the future. Furthermore, the social climate in sports studies is being investigated to provide a general overview of the current situation of the students in Germany. Data collection took place between August and November 2023, addressing all sports departments at German universities, with a total of 262 students participating. Our Findings indicate that students have a strong interest in using AI tools in their studies, expecting them to improve their overall academic performance, understand the complexity of scientific approaches, and save time. They express confidence that the proliferation of AI will not compromise their critical thinking skills. Moreover, students are positive about integrating more AI-related topics into the curriculum and about lecturers adopting more AI-based teaching methods. However, our findings also show that students have concerns about plagiarism, lecturer preparedness and their own skills and future skill development. 

**Abstract (ZH)**: 生成性人工智能工具（如ChatGPT、Copilot或Gemini）对学术研究和教学具有关键影响。关于学生如何感知人工智能日益增大的影响、他们使用的不同类型工具、期望这些工具在日常学术任务中的作用以及他们对在学习中使用人工智能的担忧的实证数据仍然有限。本文通过在德国所有学期的体育学生中进行在线问卷调查，展示了研究结果，探讨了学生使用行为、动机因素以及对人工智能工具未来影响的不确定性。此外，研究还调查了体育研究领域的社会气候，以提供当前德国学生状况的总体概述。数据采集时间为2023年8月至11月，涵盖德国所有体育系，共262名学生参与。我们的研究结果显示，学生对在学习中使用人工智能工具表现出浓厚兴趣，期望这些工具能够提高他们的学术表现、理解科学研究的复杂性并节省时间。他们相信人工智能的发展不会削弱他们的批判性思维能力。此外，学生对将更多与人工智能相关的话题纳入课程和讲师采用更多基于人工智能的教学方法持积极态度。然而，研究结果也显示，学生对学术抄袭、讲师准备程度以及自身技能和未来技能发展存在担忧。 

---
# The Illusion of Rights based AI Regulation 

**Title (ZH)**: 基于权利的AI监管幻象 

**Authors**: Yiyang Mei, Matthew Sag  

**Link**: [PDF](https://arxiv.org/pdf/2503.05784)  

**Abstract**: Whether and how to regulate AI is one of the defining questions of our times - a question that is being debated locally, nationally, and internationally. We argue that much of this debate is proceeding on a false premise. Specifically, our article challenges the prevailing academic consensus that the European Union's AI regulatory framework is fundamentally rights-driven and the correlative presumption that other rights-regarding nations should therefore follow Europe's lead in AI regulation. Rather than taking rights language in EU rules and regulations at face value, we show how EU AI regulation is the logical outgrowth of a particular cultural, political, and historical context. We show that although instruments like the General Data Protection Regulation (GDPR) and the AI Act invoke the language of fundamental rights, these rights are instrumentalized - used as rhetorical cover for governance tools that address systemic risks and maintain institutional stability. As such, we reject claims that the EU's regulatory framework and the substance of its rules should be adopted as universal imperatives and transplanted to other liberal democracies. To add weight to our argument from historical context, we conduct a comparative analysis of AI regulation in five contested domains: data privacy, cybersecurity, healthcare, labor, and misinformation. This EU-US comparison shows that the EU's regulatory architecture is not meaningfully rights-based. Our article's key intervention in AI policy debates is not to suggest that the current American regulatory model is necessarily preferable but that the presumed legitimacy of the EU's AI regulatory approach must be abandoned. 

**Abstract (ZH)**: 是否及如何规制人工智能：当下定义性问题之一——一种地方性、国家性和国际性的辩论。我们主张，这场辩论在很大程度上基于一个虚假的前提。具体而言，本文挑战了关于欧盟人工智能监管框架本质上是以权利为导向的学术共识，并由此推论其他以权利为导向的国家应效仿欧盟在人工智能监管方面的做法。我们并非简单接受欧盟规则和条例中的权利话语，而是展示了欧盟人工智能监管是如何在特定文化、政治和历史背景下自然发展的逻辑结果。我们表明，尽管《通用数据保护条例》（GDPR）和《人工智能法案》等工具使用了基本权利的语言，但这些权利实际上是被作为治理工具使用的，旨在覆盖系统性风险并维持机构稳定。因此，我们拒绝了认为欧盟监管框架及其规则内容应作为普遍要求并移植到其他自由民主国家的主张。为进一步支持我们的论点，我们对人工智能监管在五个争议领域的比较分析——数据隐私、网络安全、医疗、劳动和虚假信息——进行了历史背景下的比较。这种欧盟-美国比较表明，欧盟的监管架构本质上并非以权利为基础。本文在人工智能政策辩论中的关键介入并非建议当前美国监管模型一定更优，而是认为应放弃对欧盟人工智能监管方法合法性的假定。 

---
# Knowledge representation and scalable abstract reasoning for simulated democracy in Unity 

**Title (ZH)**: Unity中模拟民主的知识表示与可扩展抽象推理 

**Authors**: Eleftheria Katsiri, Alexandros Gazis, Angelos Protopapas  

**Link**: [PDF](https://arxiv.org/pdf/2503.05783)  

**Abstract**: We present a novel form of scalable knowledge representation about agents in a simulated democracy, e-polis, where real users respond to social challenges associated with democratic institutions, structured as Smart Spatial Types, a new type of Smart Building that changes architectural form according to the philosophical doctrine of a visitor. At the end of the game players vote on the Smart City that results from their collective choices. Our approach uses deductive systems in an unusual way: by integrating a model of democracy with a model of a Smart City we are able to prove quality aspects of the simulated democracy in different urban and social settings, while adding ease and flexibility to the development. Second, we can infer and reason with abstract knowledge, which is a limitation of the Unity platform; third, our system enables real-time decision-making and adaptation of the game flow based on the player's abstract state, paving the road to explainability. Scalability is achieved by maintaining a dual-layer knowledge representation mechanism for reasoning about the simulated democracy that functions in a similar way to a two-level cache. The lower layer knows about the current state of the game by continually processing a high rate of events produced by the in-built physics engine of the Unity platform, e.g., it knows of the position of a player in space, in terms of his coordinates x,y,z as well as their choices for each challenge. The higher layer knows of easily-retrievable, user-defined abstract knowledge about current and historical states, e.g., it knows of the political doctrine of a Smart Spatial Type, a player's philosophical doctrine, and the collective philosophical doctrine of a community players with respect to current social issues. 

**Abstract (ZH)**: 我们提出了一种新的可扩展的知识表示形式，用于模拟民主环境e-polis中的代理，其中真实用户响应与民主机构相关的社会挑战，这些挑战被结构化为智能空间类型，这是一种新的智能建筑类型，其建筑形式根据访客的哲学教义变化。游戏结束时，玩家投票决定他们集体选择所形成的智慧城市。我们的方法以非同寻常的方式使用演绎系统：通过将民主模型与智慧城市模型整合，我们能够在不同的城市和社会环境中证明模拟民主的质量方面，同时增加了开发的便捷性和灵活性。其次，我们能够推断和处理抽象知识，这是Unity平台的局限性；第三，我们的系统能够根据玩家的抽象状态实现实时决策和游戏流程的适应性，从而为可解释性铺平道路。通过维护一种类似于两级缓存的知识表示机制来实现可扩展性，以推理模拟民主的状态，该机制的下层不断处理由Unity平台内置物理引擎生成的高频事件，例如，它知道玩家在空间中的位置及其坐标的x、y、z以及每个挑战的选择。上层则易于获取用户定义的关于当前和历史状态的抽象知识，例如，它知道智能空间类型的哲学教义、玩家的哲学教义以及社区玩家对当前社会问题的集体哲学教义。 

---
# AI Mentors for Student Projects: Spotting Early Issues in Computer Science Proposals 

**Title (ZH)**: AI导师助力学生项目：计算机科学提案中早期问题的识别 

**Authors**: Gati Aher, Robin Schmucker, Tom Mitchell, Zachary C. Lipton  

**Link**: [PDF](https://arxiv.org/pdf/2503.05782)  

**Abstract**: When executed well, project-based learning (PBL) engages students' intrinsic motivation, encourages students to learn far beyond a course's limited curriculum, and prepares students to think critically and maturely about the skills and tools at their disposal. However, educators experience mixed results when using PBL in their classrooms: some students thrive with minimal guidance and others flounder. Early evaluation of project proposals could help educators determine which students need more support, yet evaluating project proposals and student aptitude is time-consuming and difficult to scale. In this work, we design, implement, and conduct an initial user study (n = 36) for a software system that collects project proposals and aptitude information to support educators in determining whether a student is ready to engage with PBL. We find that (1) users perceived the system as helpful for writing project proposals and identifying tools and technologies to learn more about, (2) educator ratings indicate that users with less technical experience in the project topic tend to write lower-quality project proposals, and (3) GPT-4o's ratings show agreement with educator ratings. While the prospect of using LLMs to rate the quality of students' project proposals is promising, its long-term effectiveness strongly hinges on future efforts at characterizing indicators that reliably predict students' success and motivation to learn. 

**Abstract (ZH)**: 基于项目的教学（PBL）实施得当可激发学生内在动机，鼓励学生超越课程限定范围进行学习，并为学生提供批判性和成熟地思考现有技能和工具的机会。然而，教师在课堂上使用PBL时体验到的效果参差不齐：一些学生在较少指导的情况下蓬勃发展，而另一些学生则陷入了困境。对项目提案的早期评估可以帮助教师确定哪些学生需要更多支持，但评估项目提案和学生能力既耗时又难以规模化。在本工作中，我们设计、实现并开展了一项初步用户研究（n=36），研究一个软件系统如何收集项目提案和能力信息，以帮助教师判断学生是否准备好参与PBL。我们发现：（1）用户认为该系统有助于撰写项目提案并识别需要深入了解的工具和技术；（2）教师评分表明，在项目主题方面技术经验较少的用户倾向于撰写质量较低的项目提案；（3）GPT-4o的评分与教师评分存在一致性。虽然使用大型语言模型（LLM）评估学生项目提案质量的前景令人鼓舞，但其长期有效性在很大程度上取决于未来在识别可靠预测学生成功和学习动机的指标方面所做的努力。 

---
# Homomorphic Encryption of Intuitionistic Logic Proofs and Functional Programs: A Categorical Approach Inspired by Composite-Order Bilinear Groups 

**Title (ZH)**: 同态加密直觉逻辑证明与功能程序：受复合阶双线性群启发的范畴论方法 

**Authors**: Ben Goertzel  

**Link**: [PDF](https://arxiv.org/pdf/2503.05779)  

**Abstract**: We present a conceptual framework for extending homomorphic encryption beyond arithmetic or Boolean operations into the domain of intuitionistic logic proofs and, by the Curry-Howard correspondence, into the domain of typed functional programs. We begin by reviewing well-known homomorphic encryption schemes for arithmetic operations, and then discuss the adaptation of similar concepts to support logical inference steps in intuitionistic logic. Key to our construction are polynomial functors and Bounded Natural Functors (BNFs), which serve as a categorical substrate on which logic formulas and proofs are represented and manipulated. We outline a complexity-theoretic hardness assumption -- the BNF Distinguishing Problem, constructed via a reduction from Subgraph Isomorphism, providing a foundation for cryptographic security. Finally, we describe how these methods can homomorphically encode the execution of total, dependently typed functional programs, and outline strategies for making the approach potentially efficient, including software optimizations and hardware acceleration. 

**Abstract (ZH)**: 我们提出了一种概念框架，将其同态加密技术从算术运算或布尔操作扩展到直觉逻辑证明领域，并通过 curry-howard 对应关系扩展到带有类型的功能程序领域。我们首先回顾了用于算术操作的已知同态加密方案，然后讨论了如何适应类似的概念以支持直觉逻辑中的逻辑推理步骤。我们的构建关键在于多项式函子和有界自然函子（BNFs），它们作为逻辑公式和证明表示和操作的范畴基底。我们概述了一个计算复杂性理论上的硬度假设——BNF区分问题，通过从子图同构问题的归约构建而成，为密码安全性提供基础。最后，我们描述了如何通过同态加密编码完整依赖类型的功能程序的执行，并概述了使该方法可能高效的方法，包括软件优化和硬件加速。 

---
# Medical Hallucinations in Foundation Models and Their Impact on Healthcare 

**Title (ZH)**: 基础模型中的医疗幻觉及其对医疗健康的影响 

**Authors**: Yubin Kim, Hyewon Jeong, Shan Chen, Shuyue Stella Li, Mingyu Lu, Kumail Alhamoud, Jimin Mun, Cristina Grau, Minseok Jung, Rodrigo Gameiro, Lizhou Fan, Eugene Park, Tristan Lin, Joonsik Yoon, Wonjin Yoon, Maarten Sap, Yulia Tsvetkov, Paul Liang, Xuhai Xu, Xin Liu, Daniel McDuff, Hyeonhoon Lee, Hae Won Park, Samir Tulebaev, Cynthia Breazeal  

**Link**: [PDF](https://arxiv.org/pdf/2503.05777)  

**Abstract**: Foundation Models that are capable of processing and generating multi-modal data have transformed AI's role in medicine. However, a key limitation of their reliability is hallucination, where inaccurate or fabricated information can impact clinical decisions and patient safety. We define medical hallucination as any instance in which a model generates misleading medical content. This paper examines the unique characteristics, causes, and implications of medical hallucinations, with a particular focus on how these errors manifest themselves in real-world clinical scenarios. Our contributions include (1) a taxonomy for understanding and addressing medical hallucinations, (2) benchmarking models using medical hallucination dataset and physician-annotated LLM responses to real medical cases, providing direct insight into the clinical impact of hallucinations, and (3) a multi-national clinician survey on their experiences with medical hallucinations. Our results reveal that inference techniques such as Chain-of-Thought (CoT) and Search Augmented Generation can effectively reduce hallucination rates. However, despite these improvements, non-trivial levels of hallucination persist. These findings underscore the ethical and practical imperative for robust detection and mitigation strategies, establishing a foundation for regulatory policies that prioritize patient safety and maintain clinical integrity as AI becomes more integrated into healthcare. The feedback from clinicians highlights the urgent need for not only technical advances but also for clearer ethical and regulatory guidelines to ensure patient safety. A repository organizing the paper resources, summaries, and additional information is available at this https URL hallucination. 

**Abstract (ZH)**: 具备处理和生成多模态数据能力的模型已经改变了医学中的AI角色。然而，它们可靠性的一个关键限制是幻觉现象，即不准确或捏造的信息可能会影响临床决策和患者安全。我们定义医学幻觉为模型生成误导性医疗内容的任何实例。本文探讨了医学幻觉的独特特征、成因及其影响，特别是关注这些错误在实际临床场景中的表现形式。我们的贡献包括（1）一种理解并解决医学幻觉的分类体系，（2）使用医学幻觉数据集和医生注释的大语言模型对实际医疗案例的响应进行基准测试，提供幻觉对临床影响的直接洞察，以及（3）一项跨国家的临床医生调查，了解他们在处理医学幻觉方面的经验。结果显示，如Chain-of-Thought（CoT）和搜索增强生成等推理技术可以有效降低幻觉率。尽管取得了这些进步，幻觉现象仍存在不可忽视的水平。这些发现强调了建立稳健检测和缓解策略的伦理和实践紧迫性，为重视患者安全和保持临床完整性的监管政策奠定了基础，尤其是在AI在医疗保健中的深度融合背景下。临床医生的反馈强调了不仅需要技术进步，还需要更加清晰的伦理和监管指南以确保患者安全。相关论文资源、摘要和额外信息的存储库可访问此链接：https://this-url-hallucination.com 

---
# Between Innovation and Oversight: A Cross-Regional Study of AI Risk Management Frameworks in the EU, U.S., UK, and China 

**Title (ZH)**: 创新与监管之间：欧盟、美国、英国和中国人工智能风险管理框架的跨区域研究 

**Authors**: Amir Al-Maamari  

**Link**: [PDF](https://arxiv.org/pdf/2503.05773)  

**Abstract**: As artificial intelligence (AI) technologies increasingly enter important sectors like healthcare, transportation, and finance, the development of effective governance frameworks is crucial for dealing with ethical, security, and societal risks. This paper conducts a comparative analysis of AI risk management strategies across the European Union (EU), United States (U.S.), United Kingdom (UK), and China. A multi-method qualitative approach, including comparative policy analysis, thematic analysis, and case studies, investigates how these regions classify AI risks, implement compliance measures, structure oversight, prioritize transparency, and respond to emerging innovations. Examples from high-risk contexts like healthcare diagnostics, autonomous vehicles, fintech, and facial recognition demonstrate the advantages and limitations of different regulatory models. The findings show that the EU implements a structured, risk-based framework that prioritizes transparency and conformity assessments, while the U.S. uses decentralized, sector-specific regulations that promote innovation but may lead to fragmented enforcement. The flexible, sector-specific strategy of the UK facilitates agile responses but may lead to inconsistent coverage across domains. China's centralized directives allow rapid large-scale implementation while constraining public transparency and external oversight. These insights show the necessity for AI regulation that is globally informed yet context-sensitive, aiming to balance effective risk management with technological progress. The paper concludes with policy recommendations and suggestions for future research aimed at enhancing effective, adaptive, and inclusive AI governance globally. 

**Abstract (ZH)**: 随着人工智能（AI）技术在医疗、交通和金融等重要领域中的应用越来越广泛，建立有效的治理框架以应对伦理、安全和社会风险至关重要。本文通过比较分析欧盟（EU）、美国（U.S.）、英国（UK）和中国在AI风险管理策略上的差异，探讨这些地区如何分类AI风险、实施合规措施、构建监督结构、强调透明度以及应对新兴创新。在医疗诊断、自动驾驶车辆、金融科技和人脸识别等高风险领域内，不同的监管模型显示出其优缺点。研究发现，欧盟实施了一种结构化、基于风险的框架，强调透明度和一致性评估，而美国则采用分散的、面向特定行业的监管措施，促进创新但也可能导致执法碎片化。英国灵活的、面向特定行业的策略有助于灵活响应，但也可能导致不同领域的一致性覆盖不足。中国的集中指导方针允许快速大规模实施，但同时限制了公众透明度和外部监督。这些洞察显示，需要一种既全局视角又具有情境敏感性的AI治理模式，旨在平衡有效的风险管理与技术进步。本文最后提出了政策建议，并就未来旨在增强全球有效、适应性强且包容性AI治理的研究方向提出了建议。 

---
# Effect of Gender Fair Job Description on Generative AI Images 

**Title (ZH)**: 性别公平招聘信息对生成式AI图像的影响 

**Authors**: Finn Böckling, Jan Marquenie, Ingo Siegert  

**Link**: [PDF](https://arxiv.org/pdf/2503.05769)  

**Abstract**: STEM fields are traditionally male-dominated, with gender biases shaping perceptions of job accessibility. This study analyzed gender representation in STEM occupation images generated by OpenAI DALL-E 3 \& Black Forest FLUX.1 using 150 prompts in three linguistic forms: German generic masculine, German pair form, and English. As control, 20 pictures of social occupations were generated as well. Results revealed significant male bias across all forms, with the German pair form showing reduced bias but still overrepresenting men for the STEM-Group and mixed results for the Group of Social Occupations. These findings highlight generative AI's role in reinforcing societal biases, emphasizing the need for further discussion on diversity (in AI). Further aspects analyzed are age-distribution and ethnic diversity. 

**Abstract (ZH)**: STEM领域传统上男性主导，性别偏见影响着职业可访问性的看法。本研究分析了由OpenAI DALL-E 3 & Black Forest FLUX生成的150个与STEM职业相关的图像中性别代表情况，使用了三种语言形式：德语通用男性形式、德语文本对形式和英语。作为对照，还生成了20张与社会职业相关的图片。结果显示，在所有形式中均存在显著的男性偏见，其中德语文本对形式显示出减少偏见的趋势，但仍过度代表男性群体，且对社会职业群体显示出混合结果。这些发现强调了生成式AI在强化社会偏见方面的作用，强调了进一步讨论多样性的必要性。同时还分析了年龄分布和族裔多样性。 

---
# A Collection of Innovations in Medical AI for patient records in 2024 

**Title (ZH)**: 2024年医疗AI在患者记录方面的创新集锦 

**Authors**: Yuanyun Zhang, Shi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.05768)  

**Abstract**: The field of Artificial Intelligence in healthcare is evolving at an unprecedented pace, driven by rapid advancements in machine learning and the recent breakthroughs in large language models. While these innovations hold immense potential to transform clinical decision making, diagnostics, and patient care, the accelerating speed of AI development has outpaced traditional academic publishing cycles. As a result, many scholarly contributions quickly become outdated, failing to capture the latest state of the art methodologies and their real world implications. This paper advocates for a new category of academic publications an annualized citation framework that prioritizes the most recent AI driven healthcare innovations. By systematically referencing the breakthroughs of the year, such papers would ensure that research remains current, fostering a more adaptive and informed discourse. This approach not only enhances the relevance of AI research in healthcare but also provides a more accurate reflection of the fields ongoing evolution. 

**Abstract (ZH)**: 人工智能在医疗领域的研究正在以前所未有的速度发展，得益于机器学习的 rapid advancements 和大型语言模型的 recent breakthroughs。尽管这些创新在改善临床决策、诊断和患者护理方面具有巨大的潜力，但人工智能的发展速度已经超过了传统学术出版周期。因此，许多学术贡献很快变得过时，无法捕捉最新的前沿方法及其实际应用。本文倡导一种新的学术出版类别——年度引用框架，优先考虑最前沿的 AI 驱动医疗创新。通过系统地引用当年的突破性成果，这类论文能够确保研究保持最新，促进更适应和有根据的对话。这种方法不仅增强了人工智能研究在医疗领域的相关性，还更准确地反映了该领域的发展演变。 

---
# Mesterséges Intelligencia Kutatások Magyarországon 

**Title (ZH)**: 匈牙利的机器学习研究 

**Authors**: András A. Benczúr, Tibor Gyimóthy, Balázs Szegedy  

**Link**: [PDF](https://arxiv.org/pdf/2503.05767)  

**Abstract**: Artificial intelligence (AI) has undergone remarkable development since the mid-2000s, particularly in the fields of machine learning and deep learning, driven by the explosive growth of large databases and computational capacity. Hungarian researchers recognized the significance of AI early on, actively participating in international research and achieving significant results in both theoretical and practical domains. This article presents some key achievements in Hungarian AI research. It highlights the results from the period before the rise of deep learning (the early 2010s), then discusses major theoretical advancements in Hungary after 2010. Finally, it provides a brief overview of AI-related applied scientific achievements from 2010 onward. 

**Abstract (ZH)**: 自2000年代中期以来，人工智能（AI）取得了 remarkable 的发展，尤其是在机器学习和深度学习领域，这得益于大规模数据库和计算能力的爆炸式增长。匈牙利研究人员早期就认识到AI的重要性，积极参加了国际研究，并在理论和实践领域取得了显著成果。本文介绍了匈牙利AI研究的一些关键成就。它突出了2010年代初深度学习兴起之前的成果，然后讨论了2010年后匈牙利在理论方面的重大进展。最后，它简要概述了2010年以来与AI相关的应用科学成就。 

---
# Graph Masked Language Models 

**Title (ZH)**: 图掩码语言模型 

**Authors**: Aarush Sinha, OM Kumar CU  

**Link**: [PDF](https://arxiv.org/pdf/2503.05763)  

**Abstract**: Language Models (LMs) are integral to Natural Language Processing (NLP), yet their interaction with structured knowledge graphs (KGs) remains an open research challenge. While Graph Neural Networks (GNNs) excel at capturing graph structures, they struggle with textual feature representation compared to pretrained LMs. To bridge this gap, we propose \textbf{Graph Masked Language Models (GMLM)} for node classification tasks. Our approach introduces two key innovations: a \textit{semantic masking strategy} that selectively masks nodes based on their structural importance, ensuring critical graph components contribute effectively to learning, and a \textit{soft masking mechanism} that generates interpolated node representations, enabling smoother information retention and improved gradient flow. Our dual-branch model architecture fuses structural graph information with contextual embeddings via a multi-layer fusion network. Extensive experiments on six node classification benchmarks demonstrate that GMLM not only achieves state-of-the-art (SOTA) performance but also enhances robustness and stability across datasets. 

**Abstract (ZH)**: 语言模型（LMs）是自然语言处理（NLP）的核心，然而它们与结构化知识图谱（KGs）的交互仍然是一个开放的研究挑战。虽然图神经网络（GNNs）在捕捉图结构方面表现优异，但在文本特征表示方面却不如预训练的语言模型。为了解决这一问题，我们提出了一种用于节点分类任务的**图掩码语言模型（GMLM）**。该方法包含两项关键创新：一种基于节点结构重要性的**语义掩码策略**，确保关键图组件有效参与到学习中；以及一种软掩码机制，生成插值节点表示，实现更平滑的信息保留和改进的梯度流动。我们的双分支模型架构通过多层融合网络将结构性图信息与上下文嵌入结合起来。在六个节点分类基准上的广泛实验表明，GMLM不仅能实现目前最高的（SOTA）性能，还在数据集间提高了鲁棒性和稳定性。 

---
# ADAPT Centre Contribution on Implementation of the EU AI Act and Fundamental Right Protection 

**Title (ZH)**: ADAPT中心在实施欧盟AI法案与基本权利保护方面的贡献 

**Authors**: Dave Lewis, Marta Lasek-Markey, Harshvardhan J. Pandit, Delaram Golpayegani, Darren McCabe, Louise McCormack, Joshua Hovsha, Deirdre Ahern, Arthit Suriyawongku  

**Link**: [PDF](https://arxiv.org/pdf/2503.05758)  

**Abstract**: This document represents the ADAPT Centre's submission to the Irish Department of Enterprise, Trade and Employment (DETE) regarding the public consultation on implementation of the EU AI Act. 

**Abstract (ZH)**: 本论文代表ADAPT研究中心向爱尔兰工业、贸易和就业部（DETE）提交的关于欧盟AI法案实施公众咨询的报告。 

---
# SEAFL: Enhancing Efficiency in Semi-Asynchronous Federated Learning through Adaptive Aggregation and Selective Training 

**Title (ZH)**: SEAFL：通过自适应聚合和选择性训练提升半异步联邦学习的效率 

**Authors**: Md Sirajul Islam, Sanjeev Panta, Fei Xu, Xu Yuan, Li Chen, Nian-Feng Tzeng  

**Link**: [PDF](https://arxiv.org/pdf/2503.05755)  

**Abstract**: Federated Learning (FL) is a promising distributed machine learning framework that allows collaborative learning of a global model across decentralized devices without uploading their local data. However, in real-world FL scenarios, the conventional synchronous FL mechanism suffers from inefficient training caused by slow-speed devices, commonly known as stragglers, especially in heterogeneous communication environments. Though asynchronous FL effectively tackles the efficiency challenge, it induces substantial system overheads and model degradation. Striking for a balance, semi-asynchronous FL has gained increasing attention, while still suffering from the open challenge of stale models, where newly arrived updates are calculated based on outdated weights that easily hurt the convergence of the global model. In this paper, we present {\em SEAFL}, a novel FL framework designed to mitigate both the straggler and the stale model challenges in semi-asynchronous FL. {\em SEAFL} dynamically assigns weights to uploaded models during aggregation based on their staleness and importance to the current global model. We theoretically analyze the convergence rate of {\em SEAFL} and further enhance the training efficiency with an extended variant that allows partial training on slower devices, enabling them to contribute to global aggregation while reducing excessive waiting times. We evaluate the effectiveness of {\em SEAFL} through extensive experiments on three benchmark datasets. The experimental results demonstrate that {\em SEAFL} outperforms its closest counterpart by up to $\sim$22\% in terms of the wall-clock training time required to achieve target accuracy. 

**Abstract (ZH)**: 联邦学习（FL）是一种有潜力的分布式机器学习框架，允许在不上传本地数据的情况下，跨分散设备协作学习全局模型。然而，在实际的FL场景中，传统的同步FL机制由于慢速设备，即所谓的“拖后腿节点”，在异构通信环境中效率低下。尽管异步FL可以有效应对效率挑战，但它会引入显著的系统开销和模型退化。介于两者之间，半异步FL正逐渐受到关注，但仍面临过时模型的开放挑战，其中新到达的更新基于过时的权重进行计算，这容易损害全局模型的收敛。本文提出了一种名为SEAFL的新颖FL框架，旨在解决半异步FL中的拖后腿节点和过时模型挑战。SEAFL在聚合期间基于模型的过时程度和对当前全局模型的重要性动态分配权重。我们从理论上分析了SEAFL的收敛速率，并通过扩展变体进一步提高训练效率，该扩展变体允许在较慢的设备上进行部分训练，使其能够在减少过多等待时间的情况下贡献于全局聚合。我们通过在三个基准数据集上进行广泛实验评估SEAFL的有效性。实验结果表明，与最接近的对照组相比，SEAFL在实现目标准确率所需的墙-clock训练时间上表现出优越性，最高可提高约22%。 

---
# Exploring AI Writers: Technology, Impact, and Future Prospects 

**Title (ZH)**: 探索AI写手：技术、影响与未来前景 

**Authors**: Zhiqian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05753)  

**Abstract**: This study explores the practical capabilities of AI writers, focusing on their applications across various creative domains. It delves into the potential impact of AI-generated content on traditional media industries and academic writing processes. The research examines how AI tools are reshaping news production workflows, particularly in fields such as finance, sports, and natural disasters. Additionally, it addresses ethical concerns, including authorship and copyright issues arising from AI-driven creative outputs. The findings reveal mixed perceptions among media students regarding the integration of AI into their profession, reflecting both optimism about efficiency gains and apprehensions over increased job market competition. 

**Abstract (ZH)**: 本研究探讨了人工智能撰稿人的实际能力，重点关注其在各类创意领域的应用。研究深入分析了人工智能生成内容对传统媒体行业及学术写作流程的潜在影响。研究考察了AI工具如何重新塑造新闻生产工作流程，特别是在金融、体育和自然灾害等领域。此外，研究还探讨了由AI驱动的创意输出引发的伦理问题，包括作者归属和版权问题。研究发现，媒体学生对将AI整合到其职业中持有复杂的看法，既体现了对效率提升的乐观态度，也反映了对就业市场竞争加剧的担忧。 

---
# CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization 

**Title (ZH)**: 基于上下文驱动的序列迁移学习的抽象放射学报告总结 

**Authors**: Mst. Fahmida Sultana Naznin, Adnan Ibney Faruq, Mostafa Rifat Tazwar, Md Jobayer, Md. Mehedi Hasan Shawon, Md Rakibul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2503.05750)  

**Abstract**: A radiology report comprises several sections, including the Findings and Impression of the diagnosis. Automatically generating the Impression from the Findings is crucial for reducing radiologists' workload and improving diagnostic accuracy. Pretrained models that excel in common abstractive summarization problems encounter challenges when applied to specialized medical domains largely due to the complex terminology and the necessity for accurate clinical context. Such tasks in medical domains demand extracting core information, avoiding context shifts, and maintaining proper flow. Misuse of medical terms can lead to drastic clinical errors. To address these issues, we introduce a sequential transfer learning that ensures key content extraction and coherent summarization. Sequential transfer learning often faces challenges like initial parameter decay and knowledge loss, which we resolve with the Fisher matrix regularization. Using MIMIC-CXR and Open-I datasets, our model, CSTRL-Context-driven Sequential TRansfer Learning-achieved state-of-the-art performance, showing 56.2% improvement in BLEU-1, 40.5% in BLEU-2, 84.3% in BLEU-3, 28.9% in ROUGE-1, 41.0% in ROUGE-2 and 26.5% in ROGUE-3 score over benchmark studies. We also analyze factual consistency scores while preserving the medical context. Our code is publicly available at TBA. 

**Abstract (ZH)**: 医学影像报告包括几个部分，如发现和诊断印象。从发现自动生成印象对于减轻放射学家的工作负荷和提高诊断准确性至关重要。由于复杂的医学术语和需要准确的临床背景，预训练模型在应用到专业医学领域时面临挑战。医学领域的此类任务要求提取核心信息、避免内容转换并保持正确的语义流程。错误使用医学术语可能导致严重的临床错误。为了解决这些问题，我们引入了一种序贯迁移学习方法，确保关键内容提取和连贯总结。序贯迁移学习常常面临初始参数衰减和知识损失等挑战，我们通过Fisher矩阵正则化解决了这些问题。使用MIMIC-CXR和Open-I数据集，我们的模型CSTRL-基于上下文的序贯转移学习达到了最先进的性能，BLEU-1分数提高了56.2%，BLEU-2提高了40.5%，BLEU-3提高了84.3%，ROUGE-1提高了28.9%，ROUGE-2提高了41.0%，ROUGE-3提高了26.5%。同时，我们保持了医学背景进行了事实一致性分析。我们的代码将在TBA上公开。 

---
# Alignment, Agency and Autonomy in Frontier AI: A Systems Engineering Perspective 

**Title (ZH)**: 前沿人工智能中的对齐、自主与自治：系统工程视角 

**Authors**: Krti Tallam  

**Link**: [PDF](https://arxiv.org/pdf/2503.05748)  

**Abstract**: As artificial intelligence scales, the concepts of alignment, agency, and autonomy have become central to AI safety, governance, and control. However, even in human contexts, these terms lack universal definitions, varying across disciplines such as philosophy, psychology, law, computer science, mathematics, and political science. This inconsistency complicates their application to AI, where differing interpretations lead to conflicting approaches in system design and regulation. This paper traces the historical, philosophical, and technical evolution of these concepts, emphasizing how their definitions influence AI development, deployment, and oversight.
We argue that the urgency surrounding AI alignment and autonomy stems not only from technical advancements but also from the increasing deployment of AI in high-stakes decision making. Using Agentic AI as a case study, we examine the emergent properties of machine agency and autonomy, highlighting the risks of misalignment in real-world systems. Through an analysis of automation failures (Tesla Autopilot, Boeing 737 MAX), multi-agent coordination (Metas CICERO), and evolving AI architectures (DeepMinds AlphaZero, OpenAIs AutoGPT), we assess the governance and safety challenges posed by frontier AI. 

**Abstract (ZH)**: 随着人工智能的发展，对齐性、自主性和自主权的概念已成为人工智能安全、治理和控制的核心。然而，在人类情境中，这些术语缺乏统一定义，在哲学、心理学、法律、计算机科学、数学和政治科学等多个学科中有所差异。这种不一致性使这些概念在应用于人工智能时变得更加复杂，不同的解释导致系统设计和监管中存在冲突的方法。本文追溯了这些概念的历史、哲学和技术进化，强调其定义如何影响人工智能的发展、部署和监管。

我们指出，围绕人工智能对齐性和自主性的紧迫性不仅源于技术进步，还在于人工智能在高风险决策中的日益广泛应用。通过以Agentic AI为例，我们探讨了机器自主性与自主权的 emergent 属性，并强调现实系统中对齐性偏差的风险。通过对自动化失败案例（特斯拉Autopilot，波音737 MAX）、多智能体协调（Meta CICERO）和 evolving 人工智能架构（DeepMind AlphaZero，OpenAI AutoGPT）的分析，我们评估了前沿人工智能带来的治理和安全挑战。 

---
# Balancing Innovation and Integrity: AI Integration in Liberal Arts College Administration 

**Title (ZH)**: 平衡创新与诚信：文科学院管理中人工智能的整合 

**Authors**: Ian Olivo Read  

**Link**: [PDF](https://arxiv.org/pdf/2503.05747)  

**Abstract**: This paper explores the intersection of artificial intelligence and higher education administration, focusing on liberal arts colleges (LACs). It examines AI's opportunities and challenges in academic and student affairs, legal compliance, and accreditation processes, while also addressing the ethical considerations of AI deployment in mission-driven institutions. Considering AI's value pluralism and potential allocative or representational harms caused by algorithmic bias, LACs must ensure AI aligns with its mission and principles. The study highlights other strategies for responsible AI integration, balancing innovation with institutional values. 

**Abstract (ZH)**: 本文探索人工智能与高等教育管理的交集，重点关注文理学院。它分析了人工智能在学术事务、学生事务、法律法规遵从及认证过程中的机遇与挑战，同时探讨了在具有使命驱动性质的机构中部署人工智能时的伦理考量。鉴于人工智能的价值多元性和因算法偏见可能带来的分配或代表性危害，文理学院必须确保人工智能与其使命和原则相一致。研究强调了负责任地整合人工智能的其他策略，平衡创新与机构价值。 

---
# Local Differences, Global Lessons: Insights from Organisation Policies for International Legislation 

**Title (ZH)**: 局部差异，全局启示：组织政策对于国际立法的见解 

**Authors**: Lucie-Aimée Kaffee, Pepa Atanasova, Anna Rogers  

**Link**: [PDF](https://arxiv.org/pdf/2503.05737)  

**Abstract**: The rapid adoption of AI across diverse domains has led to the development of organisational guidelines that vary significantly, even within the same sector. This paper examines AI policies in two domains, news organisations and universities, to understand how bottom-up governance approaches shape AI usage and oversight. By analysing these policies, we identify key areas of convergence and divergence in how organisations address risks such as bias, privacy, misinformation, and accountability. We then explore the implications of these findings for international AI legislation, particularly the EU AI Act, highlighting gaps where practical policy insights could inform regulatory refinements. Our analysis reveals that organisational policies often address issues such as AI literacy, disclosure practices, and environmental impact, areas that are underdeveloped in existing international frameworks. We argue that lessons from domain-specific AI policies can contribute to more adaptive and effective AI governance at the global level. This study provides actionable recommendations for policymakers seeking to bridge the gap between local AI practices and international regulations. 

**Abstract (ZH)**: 跨领域快速采纳人工智能导致组织指导原则在不同领域之间存在显著差异，即使在同一行业中也是如此。本文探讨新闻组织和大学领域的AI政策，以了解自下而上的治理方法如何影响人工智能的应用和监督。通过分析这些政策，我们识别出组织在应对偏差、隐私、虚假信息和问责等问题上的关键趋同与分歧领域。我们随后探讨这些发现对国际人工智能立法，特别是欧盟人工智能法案的影响，强调存在的差距，这些差距可以通过实用的政策洞察来指导立法改进。我们的分析显示，组织政策通常会关注人工智能素养、披露实践和环境影响等问题，而这些问题在现有的国际框架中尚未得到充分发展。我们认为，领域特定的人工智能政策经验可以为全球更适应和有效的治理做出贡献。该研究提供了供政策制定者参考的具体建议，旨在弥合地方人工智能实践与国际法规之间的差距。 

---
# Modeling Behavior Change for Multi-model At-Risk Students Early Prediction (extended version) 

**Title (ZH)**: 多模型风险学生早期行为改变建模（扩展版本） 

**Authors**: Jiabei Cheng, Zhen-Qun Yang, Jiannong Cao, Yu Yang, Kai Cheung Franky Poon, Daniel Lai  

**Link**: [PDF](https://arxiv.org/pdf/2503.05734)  

**Abstract**: In the educational domain, identifying students at risk of dropping out is essential for allowing educators to intervene effectively, improving both academic outcomes and overall student well-being. Data in educational settings often originate from diverse sources, such as assignments, grades, and attendance records. However, most existing research relies on online learning data and just extracting the quantitative features. While quantification eases processing, it also leads to a significant loss of original information. Moreover, current models primarily identify students with consistently poor performance through simple and discrete behavioural patterns, failing to capture the complex continuity and non-linear changes in student behaviour. We have developed an innovative prediction model, Multimodal- ChangePoint Detection (MCPD), utilizing the textual teacher remark data and numerical grade data from middle schools. Our model achieves a highly integrated and intelligent analysis by using independent encoders to process two data types, fusing the encoded feature. The model further refines its analysis by leveraging a changepoint detection module to pinpoint crucial behavioral changes, which are integrated as dynamic weights through a simple attention mechanism. Experimental validations indicate that our model achieves an accuracy range of 70- 75%, with an average outperforming baseline algorithms by approximately 5-10%. Additionally, our algorithm demonstrates a certain degree of transferability, maintaining high accuracy when adjusted and retrained with different definitions of at-risk, proving its broad applicability. 

**Abstract (ZH)**: 在教育领域，识别有辍学风险的学生对于允许教育者有效干预、改善学术成果和整体学生福祉至关重要。教育场景中的数据通常来自多种来源，如作业、成绩和出勤记录。然而，现有大多数研究依赖于在线学习数据，并仅提取定量特征。虽然量化简化了处理过程，但也导致了大量的原始信息丢失。此外，当前模型主要通过简单的离散行为模式来识别持续表现不佳的学生，未能捕捉学生行为的复杂连续性和非线性变化。我们开发了一种创新预测模型——多模态变化点检测（MCPD），利用中学的文本教师评语数据和数字成绩数据。该模型通过使用独立编码器处理两种数据类型并融合编码特征，实现了高度集成和智能的分析。模型进一步通过变更点检测模块识别关键行为变化，并通过简单的注意力机制将这些变化作为动态权重进行整合。实验验证显示，该模型的准确率为70%-75%，平均比基线算法高出约5-10%。此外，我们的算法具有一定的迁移性，在调整和重新训练时仍能保持高准确性，证明了其广泛的适用性。 

---
# Design an Ontology for Cognitive Business Strategy Based on Customer Satisfaction 

**Title (ZH)**: 基于客户满意度的认知商业战略本体设计 

**Authors**: Neda Bagherzadeh, Saeed Setayeshi, Samaneh Yazdani  

**Link**: [PDF](https://arxiv.org/pdf/2503.05733)  

**Abstract**: Ontology is a general term used by researchers who want to share information in a specific domain. One of the hallmarks of the greatest success of a powerful manager of an organization is his ability to interpret unplanned and unrelated events. Tools to solve this problem are vital to business growth. Modern technology allows customers to be more informed and influential in their roles as patrons and critics. This can make or break a business. Research shows that businesses that employ a customer-first strategy and prioritize their customers can generate more revenue. Even though there are many different Ontologies offered to businesses, none of it is built from a cognitive perspective. The objective of this study is to address the concept of strategic business plans with a cognitive ontology approach as a basis for a new management tool. This research proposes to design a cognitive ontology model that links customer measurement with traditional business models, define relationships between components and verify the accuracy of the added financial value. 

**Abstract (ZH)**: Ontology是一种研究人员用于在特定领域共享信息的通用术语。一个强大组织的卓越管理者最大成功之一在于其解读未预见和无关事件的能力。解决这一问题的工具对业务增长至关重要。现代技术使得顾客在消费者和批评者角色中更加知情和有影响力。这可能决定着一个企业的成败。研究显示，采用以顾客为中心的战略并优先考虑顾客的公司可以产生更多的收入。尽管市场上有许多不同类型的Ontology可供企业选择，但它们均未从认知角度进行构建。本研究的目标是采用基于认知Ontology的方法来讨论战略业务计划的概念，并作为新一代管理工具的基础。本文提议设计一个认知Ontology模型，将客户测量与传统商业模式联系起来，定义组件之间的关系，并验证所增加的财务价值的准确性。 

---
# AILuminate: Introducing v1.0 of the AI Risk and Reliability Benchmark from MLCommons 

**Title (ZH)**: AILuminate: MLCommons的AI风险与可靠性基准v1.0介绍 

**Authors**: Shaona Ghosh, Heather Frase, Adina Williams, Sarah Luger, Paul Röttger, Fazl Barez, Sean McGregor, Kenneth Fricklas, Mala Kumar, Quentin Feuillade--Montixi, Kurt Bollacker, Felix Friedrich, Ryan Tsang, Bertie Vidgen, Alicia Parrish, Chris Knotz, Eleonora Presani, Jonathan Bennion, Marisa Ferrara Boston, Mike Kuniavsky, Wiebke Hutiri, James Ezick, Malek Ben Salem, Rajat Sahay, Sujata Goswami, Usman Gohar, Ben Huang, Supheakmungkol Sarin, Elie Alhajjar, Canyu Chen, Roman Eng, Kashyap Ramanandula Manjusha, Virendra Mehta, Eileen Long, Murali Emani, Natan Vidra, Benjamin Rukundo, Abolfazl Shahbazi, Kongtao Chen, Rajat Ghosh, Vithursan Thangarasa, Pierre Peigné, Abhinav Singh, Max Bartolo, Satyapriya Krishna, Mubashara Akhtar, Rafael Gold, Cody Coleman, Luis Oala, Vassil Tashev, Joseph Marvin Imperial, Amy Russ, Sasidhar Kunapuli, Nicolas Miailhe, Julien Delaunay, Bhaktipriya Radharapu, Rajat Shinde, Tuesday, Debojyoti Dutta, Declan Grabb, Ananya Gangavarapu, Saurav Sahay, Agasthya Gangavarapu, Patrick Schramowski, Stephen Singam, Tom David, Xudong Han, Priyanka Mary Mammen, Tarunima Prabhakar, Venelin Kovatchev, Ahmed Ahmed, Kelvin N. Manyeki, Sandeep Madireddy, Foutse Khomh, Fedor Zhdanov, Joachim Baumann, Nina Vasan, Xianjun Yang, Carlos Mougn, Jibin Rajan Varghese, Hussain Chinoy, Seshakrishna Jitendar, Manil Maskey, Claire V. Hardgrove, Tianhao Li, Aakash Gupta, Emil Joswin, Yifan Mai, Shachi H Kumar, Cigdem Patlak, Kevin Lu, Vincent Alessi, Sree Bhargavi Balija, Chenhe Gu, Robert Sullivan, James Gealy, Matt Lavrisa, James Goel, Peter Mattson, Percy Liang, Joaquin Vanschoren  

**Link**: [PDF](https://arxiv.org/pdf/2503.05731)  

**Abstract**: The rapid advancement and deployment of AI systems have created an urgent need for standard safety-evaluation frameworks. This paper introduces AILuminate v1.0, the first comprehensive industry-standard benchmark for assessing AI-product risk and reliability. Its development employed an open process that included participants from multiple fields. The benchmark evaluates an AI system's resistance to prompts designed to elicit dangerous, illegal, or undesirable behavior in 12 hazard categories, including violent crimes, nonviolent crimes, sex-related crimes, child sexual exploitation, indiscriminate weapons, suicide and self-harm, intellectual property, privacy, defamation, hate, sexual content, and specialized advice (election, financial, health, legal). Our method incorporates a complete assessment standard, extensive prompt datasets, a novel evaluation framework, a grading and reporting system, and the technical as well as organizational infrastructure for long-term support and evolution. In particular, the benchmark employs an understandable five-tier grading scale (Poor to Excellent) and incorporates an innovative entropy-based system-response evaluation.
In addition to unveiling the benchmark, this report also identifies limitations of our method and of building safety benchmarks generally, including evaluator uncertainty and the constraints of single-turn interactions. This work represents a crucial step toward establishing global standards for AI risk and reliability evaluation while acknowledging the need for continued development in areas such as multiturn interactions, multimodal understanding, coverage of additional languages, and emerging hazard categories. Our findings provide valuable insights for model developers, system integrators, and policymakers working to promote safer AI deployment. 

**Abstract (ZH)**: AI系统的快速进步与部署迫切需要标准安全评估框架。本文介绍了AILuminate v1.0，这是首个全面的行业标准基准，用于评估AI产品的风险和可靠性。该基准评估了AI系统在12个危害类别中的抗性，包括暴力犯罪、非暴力犯罪、性犯罪、儿童性剥削、非选择性武器、自杀和自残、知识产权、隐私、诽谤、仇恨、性内容和专业建议（选举、金融、健康、法律），涵盖了广泛的危险行为诱导提示。我们的方法包括完整的评估标准、广泛的提示数据集、创新的评估框架、评分和报告系统，以及长期支持和演化的技术和组织基础设施。特别是，基准采用了易于理解的五级评分体系（差到优秀），并引入了基于熵的系统响应评估机制。除了公布该基准之外，本报告还指出了我们方法及构建安全基准的一般局限性，包括评估者的不确定性以及单轮交互的约束。本研究代表了朝着建立全球AI风险与可靠性评估标准迈出的关键一步，同时认识到在多轮交互、多模态理解、其他语言覆盖及新兴危害类别领域仍需持续发展。我们的发现为促进更安全的AI部署提供了宝贵的见解，适用于模型开发者、系统集成商和政策制定者。 

---
# Robust Optimization with Diffusion Models for Green Security 

**Title (ZH)**: 基于扩散模型的鲁棒优化绿色安全 

**Authors**: Lingkai Kong, Haichuan Wang, Yuqi Pan, Cheol Woo Kim, Mingxiao Song, Alayna Nguyen, Tonghan Wang, Haifeng Xu, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2503.05730)  

**Abstract**: In green security, defenders must forecast adversarial behavior, such as poaching, illegal logging, and illegal fishing, to plan effective patrols. These behavior are often highly uncertain and complex. Prior work has leveraged game theory to design robust patrol strategies to handle uncertainty, but existing adversarial behavior models primarily rely on Gaussian processes or linear models, which lack the expressiveness needed to capture intricate behavioral patterns. To address this limitation, we propose a conditional diffusion model for adversary behavior modeling, leveraging its strong distribution-fitting capabilities. To the best of our knowledge, this is the first application of diffusion models in the green security domain. Integrating diffusion models into game-theoretic optimization, however, presents new challenges, including a constrained mixed strategy space and the need to sample from an unnormalized distribution to estimate utilities. To tackle these challenges, we introduce a mixed strategy of mixed strategies and employ a twisted Sequential Monte Carlo (SMC) sampler for accurate sampling. Theoretically, our algorithm is guaranteed to converge to an epsilon equilibrium with high probability using a finite number of iterations and samples. Empirically, we evaluate our approach on both synthetic and real-world poaching datasets, demonstrating its effectiveness. 

**Abstract (ZH)**: 在绿色安全领域，防御者必须预测诸如偷猎、非法砍伐和非法捕鱼等敌对行为，以规划有效的巡逻策略。这些行为往往高度不确定且复杂。以往研究通过博弈论设计了稳健的巡逻策略以应对不确定性，但现有的敌对行为模型主要依赖于高斯过程或线性模型，这限制了它们捕捉复杂行为模式的能力。为解决这一限制，我们提出了一种条件扩散模型来建模敌对行为，利用其强大的分布拟合能力。据我们所知，这是首次在绿色安全领域应用扩散模型。然而，将扩散模型融入博弈论优化带来了新的挑战，包括受限的混合策略空间以及需要从未正则化分布中采样以估计效用。为解决这些挑战，我们引入了一种混合策略，并采用扭曲的顺序蒙特卡洛（SMC）采样器进行准确采样。理论上，我们的算法在有限的迭代次数和样本数量下有高概率收敛到ε平衡点。实验上，我们分别在合成的和实际的偷猎数据集上评估了我们的方法，证明了其有效性。 

---
# Political Neutrality in AI is Impossible- But Here is How to Approximate it 

**Title (ZH)**: AI中的政治中立是不可能的——但这里有如何接近它的方法 

**Authors**: Jillian Fisher, Ruth E. Appel, Chan Young Park, Yujin Potter, Liwei Jiang, Taylor Sorensen, Shangbin Feng, Yulia Tsvetkov, Margaret E. Roberts, Jennifer Pan, Dawn Song, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2503.05728)  

**Abstract**: AI systems often exhibit political bias, influencing users' opinions and decision-making. While political neutrality-defined as the absence of bias-is often seen as an ideal solution for fairness and safety, this position paper argues that true political neutrality is neither feasible nor universally desirable due to its subjective nature and the biases inherent in AI training data, algorithms, and user interactions. However, inspired by Joseph Raz's philosophical insight that "neutrality [...] can be a matter of degree" (Raz, 1986), we argue that striving for some neutrality remains essential for promoting balanced AI interactions and mitigating user manipulation. Therefore, we use the term "approximation" of political neutrality to shift the focus from unattainable absolutes to achievable, practical proxies. We propose eight techniques for approximating neutrality across three levels of conceptualizing AI, examining their trade-offs and implementation strategies. In addition, we explore two concrete applications of these approximations to illustrate their practicality. Finally, we assess our framework on current large language models (LLMs) at the output level, providing a demonstration of how it can be evaluated. This work seeks to advance nuanced discussions of political neutrality in AI and promote the development of responsible, aligned language models. 

**Abstract (ZH)**: AI系统中的政治中立：从理想到实际的逼近 

---
# A new framework for prognostics in decentralized industries: Enhancing fairness, security, and transparency through Blockchain and Federated Learning 

**Title (ZH)**: 分散行业前瞻性分析的新框架：通过区块链和联邦学习增强公平性、安全性和透明度 

**Authors**: T.Q.D. Pham, K.D. Tran, Khanh T. P. Nguyen, X.V. Tran, K.P. Tran  

**Link**: [PDF](https://arxiv.org/pdf/2503.05725)  

**Abstract**: As global industries transition towards Industry 5.0 predictive maintenance PM remains crucial for cost effective operations resilience and minimizing downtime in increasingly smart manufacturing environments In this chapter we explore how the integration of Federated Learning FL and blockchain BC technologies enhances the prediction of machinerys Remaining Useful Life RUL within decentralized and human centric industrial ecosystems Traditional centralized data approaches raise concerns over privacy security and scalability especially as Artificial intelligence AI driven smart manufacturing becomes more prevalent This chapter leverages FL to enable localized model training across multiple sites while utilizing BC to ensure trust transparency and data integrity across the network This BC integrated FL framework optimizes RUL predictions enhances data privacy and security establishes transparency and promotes collaboration in decentralized manufacturing It addresses key challenges such as maintaining privacy and security ensuring transparency and fairness and incentivizing participation in decentralized networks Experimental validation using the NASA CMAPSS dataset demonstrates the model effectiveness in real world scenarios and we extend our findings to the broader research community through open source code on GitHub inviting collaborative development to drive innovation in Industry 5.0 

**Abstract (ZH)**: 随着全球工业向 Industry 5.0 转型，预测性维护 PM 在成本有效运营、增强韧性以及减少停机时间方面仍然至关重要，在日益智能化的制造环境中尤为如此。本章探讨了联邦学习 FL 和区块链 BC 技术集成如何在分散且以人类为中心的工业生态系统中增强机器剩余使用寿命 RUL 的预测。传统的集中式数据方法在人工智能 AI 驱动的智能制造普及过程中引发了对隐私、安全性和可扩展性的担忧。本章利用联邦学习实现多站点的局部模型训练，同时利用区块链确保网络中的信任、透明度和数据完整性。该区块链集成的联邦学习框架优化了 RUL 的预测，增强了数据隐私和安全性，促进了透明度和分散制造中的协作。本章解决了分散网络中维护隐私和安全、确保透明度和公平性以及激励参与的关键挑战。通过使用 NASA CMAPSS 数据集进行实验验证，证明了模型在现实场景中的有效性，并通过 GitHub 上的开源代码扩展了研究发现，邀请合作开发以推动 Industry 5.0 的创新。 

---
# The Butterfly Effect of Technology: How Various Factors accelerate or hinder the Arrival of Technological Singularity 

**Title (ZH)**: 技术的蝴蝶效应：各种因素如何加速或阻碍技术奇点的到来 

**Authors**: Hooman Shababi  

**Link**: [PDF](https://arxiv.org/pdf/2503.05715)  

**Abstract**: This article explores the concept of technological singularity and the factors that could accelerate or hinder its arrival. The butterfly effect is used as a framework to understand how seemingly small changes in complex systems can have significant and unpredictable outcomes. In section II, we discuss the various factors that could hasten the arrival of technological singularity, such as advances in artificial intelligence and machine learning, breakthroughs in quantum computing, progress in brain-computer interfaces and human augmentation, and development of nanotechnology and 3D printing. In section III, we examine the factors that could delay or impede the arrival of technological singularity, including technical limitations and setbacks in AI and machine learning, ethical and societal concerns around AI and its impact on jobs and privacy, lack of sufficient investment in research and development, and regulatory barriers and political instability. Section IV explores the interplay of these factors and how they can impact the butterfly effect. Finally, in the conclusion, we summarize the key points discussed and emphasize the importance of considering the butterfly effect in predicting the future of technology. We call for continued research and investment in technology to shape its future and mitigate potential risks. 

**Abstract (ZH)**: 本文探索技术奇点的概念以及可能加速或阻碍其到来的因素。蝴蝶效应被用作框架，以理解看似微小的复杂系统变化可能产生的重大且不可预测的后果。在第二部分，我们讨论了可能加速技术奇点到来的各种因素，如人工智能和机器学习的进步、量子计算的突破、脑机接口和人类增强的进展，以及纳米技术和3D打印的发展。在第三部分，我们探讨了可能推迟或阻碍技术奇点到来的因素，包括人工智能和机器学习的技术限制和挫折、围绕人工智能及其对就业和隐私影响的伦理和社会关切、研究和开发投资不足，以及监管障碍和政治不稳定。第四部分探讨了这些因素的相互作用以及它们如何影响蝴蝶效应。最后，在结论中，我们总结了讨论的关键点，并强调在预测技术未来时考虑蝴蝶效应的重要性。我们呼吁继续对技术进行研究和投资，以塑造其未来并减轻潜在风险。 

---
# Labeling Synthetic Content: User Perceptions of Warning Label Designs for AI-generated Content on Social Media 

**Title (ZH)**: 合成内容的标签标识：社交媒体上AI生成内容警告标签设计的用户感知 

**Authors**: Dilrukshi Gamage, Dilki Sewwandi, Min Zhang, Arosha Bandara  

**Link**: [PDF](https://arxiv.org/pdf/2503.05711)  

**Abstract**: In this research, we explored the efficacy of various warning label designs for AI-generated content on social media platforms e.g., deepfakes. We devised and assessed ten distinct label design samples that varied across the dimensions of sentiment, color/iconography, positioning, and level of detail. Our experimental study involved 911 participants randomly assigned to these ten label designs and a control group evaluating social media content. We explored their perceptions relating to 1. Belief in the content being AI-generated, 2. Trust in the labels and 3. Social Media engagement perceptions of the content. The results demonstrate that the presence of labels had a significant effect on the users belief that the content is AI generated, deepfake, or edited by AI. However their trust in the label significantly varied based on the label design. Notably, having labels did not significantly change their engagement behaviors, such as like, comment, and sharing. However, there were significant differences in engagement based on content type: political and entertainment. This investigation contributes to the field of human computer interaction by defining a design space for label implementation and providing empirical support for the strategic use of labels to mitigate the risks associated with synthetically generated media. 

**Abstract (ZH)**: 本研究探讨了各种AI生成内容.warning标签设计在社交媒体平台上的有效性，如深度假货。我们设计并评估了十个不同标签设计样本，这些样本在情感、色彩/图标、位置和细节程度等方面有所不同。实验研究涉及911名随机分配到这十个标签设计组和一个对照组，对社交媒体内容进行评估。我们探索了他们对1. 内容为AI生成的信念、2. 对标签的信任以及3. 对内容的社交媒体参与感知的看法。结果表明，标签的存在对用户认为内容是AI生成、深度假货或由AI编辑的信念有显著影响。然而，他们对标签的信任显著地因标签设计而异。值得注意的是，标签的存在并未显著改变他们的参与行为，如点赞、评论和分享。然而，不同内容类型在参与行为上存在显著差异，尤其是政治和娱乐内容。本研究为人类计算机交互领域定义了标签实施的设计空间，并提供了实证支持，以战略方式使用标签以减轻合成媒体相关风险。 

---
# Inference Scaling Reshapes AI Governance 

**Title (ZH)**: 推理扩展重塑人工智能治理 

**Authors**: Toby Ord  

**Link**: [PDF](https://arxiv.org/pdf/2503.05705)  

**Abstract**: The shift from scaling up the pre-training compute of AI systems to scaling up their inference compute may have profound effects on AI governance. The nature of these effects depends crucially on whether this new inference compute will primarily be used during external deployment or as part of a more complex training programme within the lab. Rapid scaling of inference-at-deployment would: lower the importance of open-weight models (and of securing the weights of closed models), reduce the impact of the first human-level models, change the business model for frontier AI, reduce the need for power-intense data centres, and derail the current paradigm of AI governance via training compute thresholds. Rapid scaling of inference-during-training would have more ambiguous effects that range from a revitalisation of pre-training scaling to a form of recursive self-improvement via iterated distillation and amplification. 

**Abstract (ZH)**: 从扩大AI系统的预训练计算规模转向扩大其推理计算规模可能会对AI治理产生深远影响。这种影响的性质取决于新扩展的推理计算主要是在外部部署中使用，还是作为实验室中更复杂训练程序的一部分。推理计算在部署时的快速扩展将：降低开放权重模型的重要性（以及保护封闭模型权重的重要性），减少首批人类水平模型的影响，改变前沿AI的商业模式，减少对能耗密集型数据中心的需求，并颠覆通过训练计算阈值主导当前的AI治理范式。推理计算在训练期间的快速扩展将产生更加模糊的影响，范围从预训练规模扩展的再 revival 到通过迭代提炼和增强实现的递归自我改善。 

---
# High pressure hydrogen by machine learning and quantum Monte Carlo 

**Title (ZH)**: 机器学习和量子蒙特卡洛方法高压氢 

**Authors**: Andrea Tirelli, Giacomo Tenti, Kousuke Nakano, Sandro Sorella  

**Link**: [PDF](https://arxiv.org/pdf/2112.11099)  

**Abstract**: We have developed a technique combining the accuracy of quantum Monte Carlo in describing the electron correlation with the efficiency of a Machine Learning Potential (MLP). We use kernel regression in combination with SOAP (Smooth Overlap of Atomic Position) features, implemented here in a very efficient way. The key ingredients are: i) a sparsification technique, based on farthest point sampling, ensuring generality and transferability of our MLPs and ii) the so called $\Delta$-learning, allowing a small training data set, a fundamental property for highly accurate but computationally demanding calculations, such as the ones based on quantum Monte Carlo. As the first application we present a benchmark study of the liquid-liquid transition of high-pressure hydrogen and show the quality of our MLP, by emphasizing the importance of high accuracy for this very debated subject, where experiments are difficult in the lab, and theory is still far from being conclusive. 

**Abstract (ZH)**: 我们开发了一种结合量子蒙特卡罗描述电子相关性精确性和机器学习势效率性的技术。该方法采用核回归与SOAP（平滑原子位置重叠）特征相结合，并在此实现了非常高效的实现。关键成分包括：i) 基于最远点采样的稀疏化技术，确保我们的机器学习势的一般性和转移性；ii) 所谓的$\Delta$学习，允许使用较小的训练数据集，这是对于如基于量子蒙特卡罗计算这类高精度但计算密集型的计算而言一个基本属性。作为第一个应用，我们对高压氢的液-液相变进行了基准研究，并强调了该主题的重要性，其中实验在实验室中难以进行，而理论尚远未得出明确结论，以此展现了我们机器学习势的质量。 

---
# Learning quantum phase transitions through Topological Data Analysis 

**Title (ZH)**: 通过拓扑数据分析学习量子相变 

**Authors**: Andrea Tirelli, Natanael C. Costa  

**Link**: [PDF](https://arxiv.org/pdf/2109.09555)  

**Abstract**: We implement a computational pipeline based on a recent machine learning technique, namely the Topological Data Analysis (TDA), that has the capability of extracting powerful information-carrying topological features. We apply such a method to the study quantum phase transitions and, to showcase its validity and potential, we exploit such a method for the investigation of two paramount important quantum systems: the 2D periodic Anderson model and the Hubbard model on the honeycomb lattice, both cases on the half-filling. To this end, we have performed unbiased auxiliary field quantum Monte Carlo simulations, feeding the TDA with snapshots of the Hubbard-Stratonovich fields through the course of the simulations The quantum critical points obtained from TDA agree quantitatively well with the existing literature, therefore suggesting that this technique could be used to investigate quantum systems where the analysis of the phase transitions is still a challenge. 

**Abstract (ZH)**: 基于拓扑数据分析的计算管道在量子相变研究中的应用：以半填充满孔Anderson模型和六角晶格Hubbard模型为例 

---
