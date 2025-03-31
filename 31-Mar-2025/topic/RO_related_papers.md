# Next-Best-Trajectory Planning of Robot Manipulators for Effective Observation and Exploration 

**Title (ZH)**: 机器人 manipulator 的 Next-Best 轨迹规划以实现有效观察与探索 

**Authors**: Heiko Renz, Maximilian Krämer, Frank Hoffmann, Torsten Bertram  

**Link**: [PDF](https://arxiv.org/pdf/2503.22588)  

**Abstract**: Visual observation of objects is essential for many robotic applications, such as object reconstruction and manipulation, navigation, and scene understanding. Machine learning algorithms constitute the state-of-the-art in many fields but require vast data sets, which are costly and time-intensive to collect. Automated strategies for observation and exploration are crucial to enhance the efficiency of data gathering. Therefore, a novel strategy utilizing the Next-Best-Trajectory principle is developed for a robot manipulator operating in dynamic environments. Local trajectories are generated to maximize the information gained from observations along the path while avoiding collisions. We employ a voxel map for environment modeling and utilize raycasting from perspectives around a point of interest to estimate the information gain. A global ergodic trajectory planner provides an optional reference trajectory to the local planner, improving exploration and helping to avoid local minima. To enhance computational efficiency, raycasting for estimating the information gain in the environment is executed in parallel on the graphics processing unit. Benchmark results confirm the efficiency of the parallelization, while real-world experiments demonstrate the strategy's effectiveness. 

**Abstract (ZH)**: 利用Next-Best-Trajectory原则的机器人 manipulator 在动态环境中的观测与探索新策略 

---
# Task Hierarchical Control via Null-Space Projection and Path Integral Approach 

**Title (ZH)**: 基于零空间投影和路径积分方法的任务层级控制 

**Authors**: Apurva Patil, Riku Funada, Takashi Tanaka, Luis Sentis  

**Link**: [PDF](https://arxiv.org/pdf/2503.22574)  

**Abstract**: This paper addresses the problem of hierarchical task control, where a robotic system must perform multiple subtasks with varying levels of priority. A commonly used approach for hierarchical control is the null-space projection technique, which ensures that higher-priority tasks are executed without interference from lower-priority ones. While effective, the state-of-the-art implementations of this method rely on low-level controllers, such as PID controllers, which can be prone to suboptimal solutions in complex tasks. This paper presents a novel framework for hierarchical task control, integrating the null-space projection technique with the path integral control method. Our approach leverages Monte Carlo simulations for real-time computation of optimal control inputs, allowing for the seamless integration of simpler PID-like controllers with a more sophisticated optimal control technique. Through simulation studies, we demonstrate the effectiveness of this combined approach, showing how it overcomes the limitations of traditional 

**Abstract (ZH)**: 本文探讨了一种层次化任务控制问题，其中机器人系统需要执行具有不同优先级的多个子任务。层次化控制中常用的>`;
user
把下面的论文内容或标题翻译成中文，要符合学术规范：Path Integral Control: An Effective Solution for Hierarchical Task Control of Robotic Systems 

---
# A Centralized Planning and Distributed Execution Method for Shape Filling with Homogeneous Mobile Robots 

**Title (ZH)**: 集中规划与分布执行的 homogeneous 移动机器人形状填充方法 

**Authors**: Shuqing Liu, Rong Su, Karl H.Johansson  

**Link**: [PDF](https://arxiv.org/pdf/2503.22522)  

**Abstract**: Nature has inspired humans in different ways. The formation behavior of animals can perform tasks that exceed individual capability. For example, army ants could transverse gaps by forming bridges, and fishes could group up to protect themselves from predators. The pattern formation task is essential in a multiagent robotic system because it usually serves as the initial configuration of downstream tasks, such as collective manipulation and adaptation to various environments. The formation of complex shapes, especially hollow shapes, remains an open question. Traditional approaches either require global coordinates for each robot or are prone to failure when attempting to close the hole due to accumulated localization errors. Inspired by the ribbon idea introduced in the additive self-assembly algorithm by the Kilobot team, we develop a two-stage algorithm that does not require global coordinates information and effectively forms shapes with holes. In this paper, we investigate the partitioning of the shape using ribbons in a hexagonal lattice setting and propose the add-subtract algorithm based on the movement sequence induced by the ribbon structure. This advancement opens the door to tasks requiring complex pattern formations, such as the assembly of nanobots for medical applications involving intricate structures and the deployment of robots along the boundaries of areas of interest. We also provide simulation results on complex shapes, an analysis of the robustness as well as a proof of correctness of the proposed algorithm. 

**Abstract (ZH)**: 自然界的启发：基于绳索结构的无全局坐标两阶段算法及其在复杂形状形成中的应用 

---
# Collapse and Collision Aware Grasping for Cluttered Shelf Picking 

**Title (ZH)**: 考虑坍塌与碰撞的杂乱货架拾取中的抓取策略 

**Authors**: Abhinav Pathak, Rajkumar Muthusamy  

**Link**: [PDF](https://arxiv.org/pdf/2503.22427)  

**Abstract**: Efficient and safe retrieval of stacked objects in warehouse environments is a significant challenge due to complex spatial dependencies and structural inter-dependencies. Traditional vision-based methods excel at object localization but often lack the physical reasoning required to predict the consequences of extraction, leading to unintended collisions and collapses. This paper proposes a collapse and collision aware grasp planner that integrates dynamic physics simulations for robotic decision-making. Using a single image and depth map, an approximate 3D representation of the scene is reconstructed in a simulation environment, enabling the robot to evaluate different retrieval strategies before execution. Two approaches 1) heuristic-based and 2) physics-based are proposed for both single-box extraction and shelf clearance tasks. Extensive real-world experiments on structured and unstructured box stacks, along with validation using datasets from existing databases, show that our physics-aware method significantly improves efficiency and success rates compared to baseline heuristics. 

**Abstract (ZH)**: 在仓库环境中高效且安全地检索堆叠物体是一项重大挑战，由于复杂的空间依赖性和结构依赖性。基于传统视觉的方法在物体定位方面表现出色，但往往缺乏对物体提取后可能产生的物理后果进行预测的能力，导致意外碰撞和倒塌。本文提出了一种注意倒塌和碰撞的抓取规划器，将动态物理仿真集成到机器人决策中。仅使用单张图像和深度图，在仿真环境中重建场景的近似3D表示，从而使机器人能够在执行前评估不同的检索策略。提出了两种方法，1) 基于启发式和2) 基于物理的方法，用于单箱提取和货架清理任务。在结构化和非结构化箱堆的广泛真实世界实验中，以及使用现有数据库的数据集进行验证表明，我们的物理意识方法在效率和成功率方面显著优于基线启发式方法。 

---
# Grasping a Handful: Sequential Multi-Object Dexterous Grasp Generation 

**Title (ZH)**: 抓取一捧：序贯多对象灵巧抓取生成 

**Authors**: Haofei Lu, Yifei Dong, Zehang Weng, Jens Lundell, Danica Kragic  

**Link**: [PDF](https://arxiv.org/pdf/2503.22370)  

**Abstract**: We introduce the sequential multi-object robotic grasp sampling algorithm SeqGrasp that can robustly synthesize stable grasps on diverse objects using the robotic hand's partial Degrees of Freedom (DoF). We use SeqGrasp to construct the large-scale Allegro Hand sequential grasping dataset SeqDataset and use it for training the diffusion-based sequential grasp generator SeqDiffuser. We experimentally evaluate SeqGrasp and SeqDiffuser against the state-of-the-art non-sequential multi-object grasp generation method MultiGrasp in simulation and on a real robot. The experimental results demonstrate that SeqGrasp and SeqDiffuser reach an 8.71%-43.33% higher grasp success rate than MultiGrasp. Furthermore, SeqDiffuser is approximately 1000 times faster at generating grasps than SeqGrasp and MultiGrasp. 

**Abstract (ZH)**: 我们介绍了一种Sequential Multi-Object Robotic Grasp Sampling算法SeqGrasp，该算法可以通过机器人手部的部分自由度（DoF）稳健地合成各种物体的稳定抓取。我们使用SeqGrasp构建了大规模的Sequential Grasping数据集SeqDataset，并用于训练基于扩散的Sequential抓取生成模型SeqDiffuser。我们在模拟和真实机器人上将SeqGrasp和SeqDiffuser与最先进的非序列多对象抓取生成方法MultiGrasp进行了实验评估。实验结果表明，SeqGrasp和SeqDiffuser的抓取成功率比MultiGrasp高8.71%-43.33%。此外，SeqDiffuser生成抓取的速度比SeqGrasp和MultiGrasp快约1000倍。 

---
# Robust simultaneous UWB-anchor calibration and robot localization for emergency situations 

**Title (ZH)**: 鲁棒的同时超宽带锚点校准与机器人定位方法在紧急情况下的应用 

**Authors**: Xinghua Liu, Ming Cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.22272)  

**Abstract**: In this work, we propose a factor graph optimization (FGO) framework to simultaneously solve the calibration problem for Ultra-WideBand (UWB) anchors and the robot localization problem. Calibrating UWB anchors manually can be time-consuming and even impossible in emergencies or those situations without special calibration tools. Therefore, automatic estimation of the anchor positions becomes a necessity. The proposed method enables the creation of a soft sensor providing the position information of the anchors in a UWB network. This soft sensor requires only UWB and LiDAR measurements measured from a moving robot. The proposed FGO framework is suitable for the calibration of an extendable large UWB network. Moreover, the anchor calibration problem and robot localization problem can be solved simultaneously, which saves time for UWB network deployment. The proposed framework also helps to avoid artificial errors in the UWB-anchor position estimation and improves the accuracy and robustness of the robot-pose. The experimental results of the robot localization using LiDAR and a UWB network in a 3D environment are discussed, demonstrating the performance of the proposed method. More specifically, the anchor calibration problem with four anchors and the robot localization problem can be solved simultaneously and automatically within 30 seconds by the proposed framework. The supplementary video and codes can be accessed via this https URL. 

**Abstract (ZH)**: 基于因子图优化的超宽带锚点校准与机器人定位一体化框架 

---
# Bimanual Regrasp Planning and Control for Eliminating Object Pose Uncertainty 

**Title (ZH)**: 双手重新抓取计划与控制以消除物体姿态不确定性 

**Authors**: Ryuta Nagahama, Weiwei Wan, Zhengtao Hu, Kensuke Harada  

**Link**: [PDF](https://arxiv.org/pdf/2503.22240)  

**Abstract**: Precisely grasping an object is a challenging task due to pose uncertainties. Conventional methods have used cameras and fixtures to reduce object uncertainty. They are effective but require intensive preparation, such as designing jigs based on the object geometry and calibrating cameras with high-precision tools fabricated using lasers. In this study, we propose a method to reduce the uncertainty of the position and orientation of a grasped object without using a fixture or a camera. Our method is based on the concept that the flat finger pads of a parallel gripper can reduce uncertainty along its opening/closing direction through flat surface contact. Three orthogonal grasps by parallel grippers with flat finger pads collectively constrain an object's position and orientation to a unique state. Guided by the concepts, we develop a regrasp planning and admittance control approach that sequentially finds and leverages three orthogonal grasps of two robotic arms to eliminate uncertainties in the object pose. We evaluated the proposed method on different initial object uncertainties and verified that the method has satisfactory repeatability accuracy. It outperforms an AR marker detection method implemented using cameras and laser jet printers under standard laboratory conditions. 

**Abstract (ZH)**: 精确抓取对象是一项具有挑战性的任务，由于姿态不确定性的存在。传统方法使用相机和固定装置来减少对象的不确定性，这些方法虽然有效，但需要大量的准备工作，如根据对象的几何形状设计夹具，并使用激光加工的高精度工具对相机进行校准。在本研究中，我们提出了一种在不使用固定装置或相机的情况下减少被抓取对象的位置和姿态不确定性的方法。该方法基于并指夹爪的平finger pads可以在平表面接触时减少沿开口/闭合方向的不确定性。并指夹爪配备平finger pads进行三次相互垂直的抓取共同约束对象的位置和姿态到一个独特状态。根据上述理念，我们开发了一种重新抓取规划和顺应控制方法，该方法顺序地找到并利用两个机械臂的三次相互垂直的抓取，以消除对象姿态的不确定性。我们对不同初始对象不确定性进行了评估，并验证了该方法具有满意的重复精度。在标准实验室条件下，该方法优于使用相机和激光喷墨打印机实现的AR标记检测方法。 

---
# IKSel: Selecting Good Seed Joint Values for Fast Numerical Inverse Kinematics Iterations 

**Title (ZH)**: IKSel: 选择良好的起始关节点值以实现快速数值逆运动学迭代 

**Authors**: Xinyi Yuan, Weiwei Wan, Kensuke Harada  

**Link**: [PDF](https://arxiv.org/pdf/2503.22234)  

**Abstract**: This paper revisits the numerical inverse kinematics (IK) problem, leveraging modern computational resources and refining the seed selection process to develop a solver that is competitive with analytical-based methods. The proposed seed selection strategy consists of three key stages: (1) utilizing a K-Dimensional Tree (KDTree) to identify seed candidates based on workspace proximity, (2) sorting candidates by joint space adjustment and attempting numerical iterations with the one requiring minimal adjustment, and (3) re-selecting the most distant joint configurations for new attempts in case of failures. The joint space adjustment-based seed selection increases the likelihood of rapid convergence, while the re-attempt strategy effectively helps circumvent local minima and joint limit constraints. Comparison results with both traditional numerical solvers and learning-based methods demonstrate the strengths of the proposed approach in terms of success rate, time efficiency, and accuracy. Additionally, we conduct detailed ablation studies to analyze the effects of various parameters and solver settings, providing practical insights for customization and optimization. The proposed method consistently exhibits high success rates and computational efficiency. It is suitable for time-sensitive applications. 

**Abstract (ZH)**: 本文重新审视了数值逆运动学（IK）问题，利用现代计算资源并精炼种子选择过程，开发了一种与基于分析的方法相竞争的求解器。所提出的一种种子选择策略包含三个关键阶段：（1）利用K-Dimensional Tree（K-D树）基于工作空间接近性识别种子候选；（2）根据关节空间调整对候选种子进行排序，并尝试对调整最小的关节进行数值迭代；（3）在失败情况下，重新选择最远离的关节配置进行新的尝试。基于关节空间调整的种子选择增加了快速收敛的可能性，而重新尝试策略有效帮助避免局部最小值和关节限制约束。与传统数值求解器和基于学习的方法的比较结果表明，所提出方法在成功率、时间效率和准确性方面的优势。此外，我们进行了详细的成功率和求解器设置消融研究，为自定义和优化提供了实用见解。所提出的方法一致地表现出高成功率和计算效率，适用于时间敏感的应用。 

---
# Bayesian Inferential Motion Planning Using Heavy-Tailed Distributions 

**Title (ZH)**: 基于重尾分布的贝叶斯推断motion planning 

**Authors**: Ali Vaziri, Iman Askari, Huazhen Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22030)  

**Abstract**: Robots rely on motion planning to navigate safely and efficiently while performing various tasks. In this paper, we investigate motion planning through Bayesian inference, where motion plans are inferred based on planning objectives and constraints. However, existing Bayesian motion planning methods often struggle to explore low-probability regions of the planning space, where high-quality plans may reside. To address this limitation, we propose the use of heavy-tailed distributions -- specifically, Student's-$t$ distributions -- to enhance probabilistic inferential search for motion plans. We develop a novel sequential single-pass smoothing approach that integrates Student's-$t$ distribution with Monte Carlo sampling. A special case of this approach is ensemble Kalman smoothing, which depends on short-tailed Gaussian distributions. We validate the proposed approach through simulations in autonomous vehicle motion planning, demonstrating its superior performance in planning, sampling efficiency, and constraint satisfaction compared to ensemble Kalman smoothing. While focused on motion planning, this work points to the broader potential of heavy-tailed distributions in enhancing probabilistic decision-making in robotics. 

**Abstract (ZH)**: 机器人依靠运动规划在执行各种任务时安全高效地导航。本文通过贝叶斯推断研究运动规划，其中运动计划根据规划目标和约束进行推断。然而，现有的贝叶斯运动规划方法往往难以探索规划空间中的低概率区域，而高质量的计划可能就位于这些区域。为解决这一局限性，我们提出使用重尾分布——具体而言是学生-t分布——来增强运动计划的概率推断搜索。我们开发了一种新颖的顺序单遍平滑方法，将学生-t分布与蒙特卡洛采样结合起来。这种方法的一个特例是集成卡尔曼平滑，它依赖于短尾正态分布。我们通过自主车辆运动规划的仿真实验证明了所提出方法在规划、采样效率和约束满足方面的优势，相较于集成卡尔曼平滑。尽管该工作专注于运动规划，但它指出了重尾分布在增强机器人领域概率决策方面的更广泛潜力。 

---
# Beyond Omakase: Designing Shared Control for Navigation Robots with Blind People 

**Title (ZH)**: 超越 Omakase：为视障人士设计导航机器人共享控制方案 

**Authors**: Rie Kamikubo, Seita Kayukawa, Yuka Kaniwa, Allan Wang, Hernisa Kacorri, Hironobu Takagi, Chieko Asakawa  

**Link**: [PDF](https://arxiv.org/pdf/2503.21997)  

**Abstract**: Autonomous navigation robots can increase the independence of blind people but often limit user control, following what is called in Japanese an "omakase" approach where decisions are left to the robot. This research investigates ways to enhance user control in social robot navigation, based on two studies conducted with blind participants. The first study, involving structured interviews (N=14), identified crowded spaces as key areas with significant social challenges. The second study (N=13) explored navigation tasks with an autonomous robot in these environments and identified design strategies across different modes of autonomy. Participants preferred an active role, termed the "boss" mode, where they managed crowd interactions, while the "monitor" mode helped them assess the environment, negotiate movements, and interact with the robot. These findings highlight the importance of shared control and user involvement for blind users, offering valuable insights for designing future social navigation robots. 

**Abstract (ZH)**: 自主导航机器人可以增加盲人独立性，但 often limit user control，采取日本所说的“omakase”方式，由机器人自行作出决策。本研究基于与盲人参与者进行的两项研究，探索增强社会机器人导航中用户控制的方法。第一项研究（N=14）通过结构化访谈识别出拥挤空间为关键的社会挑战区域。第二项研究（N=13）探讨了在这些环境中使用自主机器人完成导航任务，并提出了不同类型自主模式下的设计策略。参与者偏好一种主动模式，称为“老板”模式，他们在该模式下管理人群互动，而“监控”模式则帮助他们评估环境、规划移动路线并与机器人互动。这些发现突出了盲人用户中共同控制和用户参与的重要性，为设计未来社会导航机器人提供了宝贵见解。 

---
