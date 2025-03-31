# Empirical Analysis of Sim-and-Real Cotraining Of Diffusion Policies For Planar Pushing from Pixels 

**Title (ZH)**: 像素级平面上推任务中模拟与现实协同训练扩散政策的实证分析 

**Authors**: Adam Wei, Abhinav Agarwal, Boyuan Chen, Rohan Bosworth, Nicholas Pfaff, Russ Tedrake  

**Link**: [PDF](https://arxiv.org/pdf/2503.22634)  

**Abstract**: In imitation learning for robotics, cotraining with demonstration data generated both in simulation and on real hardware has emerged as a powerful recipe to overcome the sim2real gap. This work seeks to elucidate basic principles of this sim-and-real cotraining to help inform simulation design, sim-and-real dataset creation, and policy training. Focusing narrowly on the canonical task of planar pushing from camera inputs enabled us to be thorough in our study. These experiments confirm that cotraining with simulated data \emph{can} dramatically improve performance in real, especially when real data is limited. Performance gains scale with simulated data, but eventually plateau; real-world data increases this performance ceiling. The results also suggest that reducing the domain gap in physics may be more important than visual fidelity for non-prehensile manipulation tasks. Perhaps surprisingly, having some visual domain gap actually helps the cotrained policy -- binary probes reveal that high-performing policies learn to distinguish simulated domains from real. We conclude by investigating this nuance and mechanisms that facilitate positive transfer between sim-and-real. In total, our experiments span over 40 real-world policies (evaluated on 800+ trials) and 200 simulated policies (evaluated on 40,000+ trials). 

**Abstract (ZH)**: 在机器人领域，使用在仿真和实际硬件中生成的演示数据进行协同训练，已成为克服仿真到现实差距的有效方法。本研究旨在阐明这种仿真实践协同训练的基本原理，以帮助指导仿真设计、仿真实际数据集的创建以及策略训练。我们将研究聚焦于平面推物这一典型任务，特别是在摄像头输入的情况下，以确保研究的彻底性。实验结果表明，使用仿真实例数据进行协同训练可以显著提高实际环境中的性能，尤其是在真实数据有限的情况下。随着仿真实例数据量的增加，性能提升会逐渐减缓，而实际数据则进一步提高了性能天花板。研究结果还表明，对于非接触式 manipulation 任务，物理域差距的减少可能比视觉保真度更为重要。令人意外的是，存在一定程度的视觉域差距实际上有助于协同训练策略的表现——二元探针显示，高性能策略能够学会区分仿真实例域和实际域。最后，我们探讨了这一细微之处及其促进仿真到现实正向迁移的机制。我们的实验包括了超过40个实际世界的策略（在800多次试验中评估）和超过200个仿真的策略（在40000多次试验中评估）。 

---
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
# SafeCast: Risk-Responsive Motion Forecasting for Autonomous Vehicles 

**Title (ZH)**: SafeCast：响应风险的自主车辆运动预测 

**Authors**: Haicheng Liao, Hanlin Kong, Bin Rao, Bonan Wang, Chengyue Wang, Guyang Yu, Yuming Huang, Ruru Tang, Chengzhong Xu, Zhenning Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.22541)  

**Abstract**: Accurate motion forecasting is essential for the safety and reliability of autonomous driving (AD) systems. While existing methods have made significant progress, they often overlook explicit safety constraints and struggle to capture the complex interactions among traffic agents, environmental factors, and motion dynamics. To address these challenges, we present SafeCast, a risk-responsive motion forecasting model that integrates safety-aware decision-making with uncertainty-aware adaptability. SafeCast is the first to incorporate the Responsibility-Sensitive Safety (RSS) framework into motion forecasting, encoding interpretable safety rules--such as safe distances and collision avoidance--based on traffic norms and physical principles. To further enhance robustness, we introduce the Graph Uncertainty Feature (GUF), a graph-based module that injects learnable noise into Graph Attention Networks, capturing real-world uncertainties and enhancing generalization across diverse scenarios. We evaluate SafeCast on four real-world benchmark datasets--Next Generation Simulation (NGSIM), Highway Drone (HighD), ApolloScape, and the Macao Connected Autonomous Driving (MoCAD)--covering highway, urban, and mixed-autonomy traffic environments. Our model achieves state-of-the-art (SOTA) accuracy while maintaining a lightweight architecture and low inference latency, underscoring its potential for real-time deployment in safety-critical AD systems. 

**Abstract (ZH)**: 准确的运动预测对于自动驾驶（AD）系统的安全性和可靠性至关重要。尽管现有方法取得了显著进展，但它们往往忽视了明确的安全约束，难以捕捉交通代理、环境因素和运动动力学之间的复杂交互。为应对这些挑战，我们提出SafeCast，一种风险响应型运动预测模型，将安全意识决策与不确定性意识适应性相结合。SafeCast是首次将责任敏感安全（RSS）框架融入运动预测中，基于交通规范和物理原理编码可解释的安全规则，如安全距离和碰撞避免规则。为进一步增强鲁棒性，我们引入了基于图的不确定性特征（GUF）模块，该模块通过图注意力网络注入可学习的噪声，捕捉真实世界的不确定性，增强在不同场景下的泛化能力。我们在Next Generation Simulation（NGSIM）、Highway Drone（HighD）、ApolloScape和Macao Connected Autonomous Driving（MoCAD）四个真实世界基准数据集上评估SafeCast，涵盖高速公路、城市和混合自主交通环境。我们的模型在保持轻量级架构和低推理延迟的同时实现了最先进的（SOTA）精度，表明其在安全关键的AD系统中的实时部署潜力。 

---
# Robust Offline Imitation Learning Through State-level Trajectory Stitching 

**Title (ZH)**: 基于状态级轨迹拼接的鲁棒离线 imitation 学习 

**Authors**: Shuze Wang, Yunpeng Mei, Hongjie Cao, Yetian Yuan, Gang Wang, Jian Sun, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22524)  

**Abstract**: Imitation learning (IL) has proven effective for enabling robots to acquire visuomotor skills through expert demonstrations. However, traditional IL methods are limited by their reliance on high-quality, often scarce, expert data, and suffer from covariate shift. To address these challenges, recent advances in offline IL have incorporated suboptimal, unlabeled datasets into the training. In this paper, we propose a novel approach to enhance policy learning from mixed-quality offline datasets by leveraging task-relevant trajectory fragments and rich environmental dynamics. Specifically, we introduce a state-based search framework that stitches state-action pairs from imperfect demonstrations, generating more diverse and informative training trajectories. Experimental results on standard IL benchmarks and real-world robotic tasks showcase that our proposed method significantly improves both generalization and performance. 

**Abstract (ZH)**: 基于任务相关轨迹片段和丰富环境动力学的混合质量离线 imitation 学习方法 

---
# A Centralized Planning and Distributed Execution Method for Shape Filling with Homogeneous Mobile Robots 

**Title (ZH)**: 集中规划与分布执行的 homogeneous 移动机器人形状填充方法 

**Authors**: Shuqing Liu, Rong Su, Karl H.Johansson  

**Link**: [PDF](https://arxiv.org/pdf/2503.22522)  

**Abstract**: Nature has inspired humans in different ways. The formation behavior of animals can perform tasks that exceed individual capability. For example, army ants could transverse gaps by forming bridges, and fishes could group up to protect themselves from predators. The pattern formation task is essential in a multiagent robotic system because it usually serves as the initial configuration of downstream tasks, such as collective manipulation and adaptation to various environments. The formation of complex shapes, especially hollow shapes, remains an open question. Traditional approaches either require global coordinates for each robot or are prone to failure when attempting to close the hole due to accumulated localization errors. Inspired by the ribbon idea introduced in the additive self-assembly algorithm by the Kilobot team, we develop a two-stage algorithm that does not require global coordinates information and effectively forms shapes with holes. In this paper, we investigate the partitioning of the shape using ribbons in a hexagonal lattice setting and propose the add-subtract algorithm based on the movement sequence induced by the ribbon structure. This advancement opens the door to tasks requiring complex pattern formations, such as the assembly of nanobots for medical applications involving intricate structures and the deployment of robots along the boundaries of areas of interest. We also provide simulation results on complex shapes, an analysis of the robustness as well as a proof of correctness of the proposed algorithm. 

**Abstract (ZH)**: 自然界的启发：基于绳索结构的无全局坐标两阶段算法及其在复杂形状形成中的应用 

---
# Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments 

**Title (ZH)**: 情景梦师：向量化的潜空间扩散模型用于生成驾驶模拟环境 

**Authors**: Luke Rowe, Roger Girgis, Anthony Gosselin, Liam Paull, Christopher Pal, Felix Heide  

**Link**: [PDF](https://arxiv.org/pdf/2503.22496)  

**Abstract**: We introduce Scenario Dreamer, a fully data-driven generative simulator for autonomous vehicle planning that generates both the initial traffic scene - comprising a lane graph and agent bounding boxes - and closed-loop agent behaviours. Existing methods for generating driving simulation environments encode the initial traffic scene as a rasterized image and, as such, require parameter-heavy networks that perform unnecessary computation due to many empty pixels in the rasterized scene. Moreover, we find that existing methods that employ rule-based agent behaviours lack diversity and realism. Scenario Dreamer instead employs a novel vectorized latent diffusion model for initial scene generation that directly operates on the vectorized scene elements and an autoregressive Transformer for data-driven agent behaviour simulation. Scenario Dreamer additionally supports scene extrapolation via diffusion inpainting, enabling the generation of unbounded simulation environments. Extensive experiments show that Scenario Dreamer outperforms existing generative simulators in realism and efficiency: the vectorized scene-generation base model achieves superior generation quality with around 2x fewer parameters, 6x lower generation latency, and 10x fewer GPU training hours compared to the strongest baseline. We confirm its practical utility by showing that reinforcement learning planning agents are more challenged in Scenario Dreamer environments than traditional non-generative simulation environments, especially on long and adversarial driving environments. 

**Abstract (ZH)**: 情景梦者：一种全数据驱动的自主车辆规划生成模拟器 

---
# Control of Humanoid Robots with Parallel Mechanisms using Kinematic Actuation Models 

**Title (ZH)**: 基于并行机构动力学模型的人形机器人控制 

**Authors**: Victor Lutz, Ludovic de Matteïs, Virgile Batto, Nicolas Mansard  

**Link**: [PDF](https://arxiv.org/pdf/2503.22459)  

**Abstract**: Inspired by the mechanical design of Cassie, several recently released humanoid robots are using actuator configuration in which the motor is displaced from the joint location to optimize the leg inertia. This in turn induces a non linearity in the reduction ratio of the transmission which is often neglected when computing the robot motion (e.g. by trajectory optimization or reinforcement learning) and only accounted for at control time. This paper proposes an analytical method to efficiently handle this non-linearity. Using this actuation model, we demonstrate that we can leverage the dynamic abilities of the non-linear transmission while only modeling the inertia of the main serial chain of the leg, without approximating the motor capabilities nor the joint range. Based on analytical inverse kinematics, our method does not need any numerical routines dedicated to the closed-kinematics actuation, hence leading to very efficient computations. Our study focuses on two mechanisms widely used in recent humanoid robots; the four bar knee linkage as well as a parallel 2 DoF ankle mechanism. We integrate these models inside optimization based (DDP) and learning (PPO) control approaches. A comparison of our model against a simplified model that completely neglects closed chains is then shown in simulation. 

**Abstract (ZH)**: 基于卡西机器人机械设计的类人机器人腿部驱动配置及其非线性传动比的解析处理方法 

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
# FLAM: Foundation Model-Based Body Stabilization for Humanoid Locomotion and Manipulation 

**Title (ZH)**: FLAM: 基于基础模型的人形运动和操作中的姿态稳定化 

**Authors**: Xianqi Zhang, Hongliang Wei, Wenrui Wang, Xingtao Wang, Xiaopeng Fan, Debin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.22249)  

**Abstract**: Humanoid robots have attracted significant attention in recent years. Reinforcement Learning (RL) is one of the main ways to control the whole body of humanoid robots. RL enables agents to complete tasks by learning from environment interactions, guided by task rewards. However, existing RL methods rarely explicitly consider the impact of body stability on humanoid locomotion and manipulation. Achieving high performance in whole-body control remains a challenge for RL methods that rely solely on task rewards. In this paper, we propose a Foundation model-based method for humanoid Locomotion And Manipulation (FLAM for short). FLAM integrates a stabilizing reward function with a basic policy. The stabilizing reward function is designed to encourage the robot to learn stable postures, thereby accelerating the learning process and facilitating task completion. Specifically, the robot pose is first mapped to the 3D virtual human model. Then, the human pose is stabilized and reconstructed through a human motion reconstruction model. Finally, the pose before and after reconstruction is used to compute the stabilizing reward. By combining this stabilizing reward with the task reward, FLAM effectively guides policy learning. Experimental results on a humanoid robot benchmark demonstrate that FLAM outperforms state-of-the-art RL methods, highlighting its effectiveness in improving stability and overall performance. 

**Abstract (ZH)**: 基于基础模型的人形机器人步行与 manipulation 方法（FLAM） 

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
# 3D Acetabular Surface Reconstruction from 2D Pre-operative X-ray Images using SRVF Elastic Registration and Deformation Graph 

**Title (ZH)**: 基于SRVF弹性注册与变形图的术前X-ray图像三维髋臼表面重建 

**Authors**: Shuai Zhang, Jinliang Wang, Sujith Konandetails, Xu Wang, Danail Stoyanov, Evangelos B.Mazomenos  

**Link**: [PDF](https://arxiv.org/pdf/2503.22177)  

**Abstract**: Accurate and reliable selection of the appropriate acetabular cup size is crucial for restoring joint biomechanics in total hip arthroplasty (THA). This paper proposes a novel framework that integrates square-root velocity function (SRVF)-based elastic shape registration technique with an embedded deformation (ED) graph approach to reconstruct the 3D articular surface of the acetabulum by fusing multiple views of 2D pre-operative pelvic X-ray images and a hemispherical surface model. The SRVF-based elastic registration establishes 2D-3D correspondences between the parametric hemispherical model and X-ray images, and the ED framework incorporates the SRVF-derived correspondences as constraints to optimize the 3D acetabular surface reconstruction using nonlinear least-squares optimization. Validations using both simulation and real patient datasets are performed to demonstrate the robustness and the potential clinical value of the proposed algorithm. The reconstruction result can assist surgeons in selecting the correct acetabular cup on the first attempt in primary THA, minimising the need for revision surgery. 

**Abstract (ZH)**: 基于平方根速度函数的弹性形状注册与嵌入变形图结合的股骨头臼杯大小精确选择框架：融合二维术前骨盆X射线多视角图像和半球面表面模型重构三维髋臼关节表面 

---
# Cooperative Hybrid Multi-Agent Pathfinding Based on Shared Exploration Maps 

**Title (ZH)**: 基于共享探索地图的合作混合多智能体路径规划 

**Authors**: Ning Liu, Sen Shen, Xiangrui Kong, Hongtao Zhang, Thomas Bräunl  

**Link**: [PDF](https://arxiv.org/pdf/2503.22162)  

**Abstract**: Multi-Agent Pathfinding is used in areas including multi-robot formations, warehouse logistics, and intelligent vehicles. However, many environments are incomplete or frequently change, making it difficult for standard centralized planning or pure reinforcement learning to maintain both global solution quality and local flexibility. This paper introduces a hybrid framework that integrates D* Lite global search with multi-agent reinforcement learning, using a switching mechanism and a freeze-prevention strategy to handle dynamic conditions and crowded settings. We evaluate the framework in the discrete POGEMA environment and compare it with baseline methods. Experimental outcomes indicate that the proposed framework substantially improves success rate, collision rate, and path efficiency. The model is further tested on the EyeSim platform, where it maintains feasible Pathfinding under frequent changes and large-scale robot deployments. 

**Abstract (ZH)**: 多智能体路径规划在多机器人编队、仓库物流和智能车辆等领域被广泛应用。然而，许多环境不完整或频繁变化，使得标准的集中规划或纯强化学习难以同时保持全局解的质量和局部灵活性。本文提出了一种将D* Lite全局搜索与多智能体强化学习相结合的混合框架，通过切换机制和防冻结策略处理动态条件和密集环境。我们在离散POGEMA环境中评估了该框架，并与基线方法进行比较。实验结果表明，所提出的框架显著提高了成功率、碰撞率和路径效率。该模型进一步在EyeSim平台上测试，能够在频繁变化和大规模机器人部署的环境中保持有效的路径规划。 

---
# REMAC: Self-Reflective and Self-Evolving Multi-Agent Collaboration for Long-Horizon Robot Manipulation 

**Title (ZH)**: REMAC: 自我反思与自我进化的多agent协作长时程机器人操作 

**Authors**: Puzhen Yuan, Angyuan Ma, Yunchao Yao, Huaxiu Yao, Masayoshi Tomizuka, Mingyu Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.22122)  

**Abstract**: Vision-language models (VLMs) have demonstrated remarkable capabilities in robotic planning, particularly for long-horizon tasks that require a holistic understanding of the environment for task decomposition. Existing methods typically rely on prior environmental knowledge or carefully designed task-specific prompts, making them struggle with dynamic scene changes or unexpected task conditions, e.g., a robot attempting to put a carrot in the microwave but finds the door was closed. Such challenges underscore two critical issues: adaptability and efficiency. To address them, in this work, we propose an adaptive multi-agent planning framework, termed REMAC, that enables efficient, scene-agnostic multi-robot long-horizon task planning and execution through continuous reflection and self-evolution. REMAC incorporates two key modules: a self-reflection module performing pre-condition and post-condition checks in the loop to evaluate progress and refine plans, and a self-evolvement module dynamically adapting plans based on scene-specific reasoning. It offers several appealing benefits: 1) Robots can initially explore and reason about the environment without complex prompt design. 2) Robots can keep reflecting on potential planning errors and adapting the plan based on task-specific insights. 3) After iterations, a robot can call another one to coordinate tasks in parallel, maximizing the task execution efficiency. To validate REMAC's effectiveness, we build a multi-agent environment for long-horizon robot manipulation and navigation based on RoboCasa, featuring 4 task categories with 27 task styles and 50+ different objects. Based on it, we further benchmark state-of-the-art reasoning models, including DeepSeek-R1, o3-mini, QwQ, and Grok3, demonstrating REMAC's superiority by boosting average success rates by 40% and execution efficiency by 52.7% over the single robot baseline. 

**Abstract (ZH)**: 基于视觉-语言模型的自适应多-agent规划框架REMAC：在动态场景下的长时任务执行与协同规划 

---
# Bayesian Inferential Motion Planning Using Heavy-Tailed Distributions 

**Title (ZH)**: 基于重尾分布的贝叶斯推断motion planning 

**Authors**: Ali Vaziri, Iman Askari, Huazhen Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22030)  

**Abstract**: Robots rely on motion planning to navigate safely and efficiently while performing various tasks. In this paper, we investigate motion planning through Bayesian inference, where motion plans are inferred based on planning objectives and constraints. However, existing Bayesian motion planning methods often struggle to explore low-probability regions of the planning space, where high-quality plans may reside. To address this limitation, we propose the use of heavy-tailed distributions -- specifically, Student's-$t$ distributions -- to enhance probabilistic inferential search for motion plans. We develop a novel sequential single-pass smoothing approach that integrates Student's-$t$ distribution with Monte Carlo sampling. A special case of this approach is ensemble Kalman smoothing, which depends on short-tailed Gaussian distributions. We validate the proposed approach through simulations in autonomous vehicle motion planning, demonstrating its superior performance in planning, sampling efficiency, and constraint satisfaction compared to ensemble Kalman smoothing. While focused on motion planning, this work points to the broader potential of heavy-tailed distributions in enhancing probabilistic decision-making in robotics. 

**Abstract (ZH)**: 机器人依靠运动规划在执行各种任务时安全高效地导航。本文通过贝叶斯推断研究运动规划，其中运动计划根据规划目标和约束进行推断。然而，现有的贝叶斯运动规划方法往往难以探索规划空间中的低概率区域，而高质量的计划可能就位于这些区域。为解决这一局限性，我们提出使用重尾分布——具体而言是学生-t分布——来增强运动计划的概率推断搜索。我们开发了一种新颖的顺序单遍平滑方法，将学生-t分布与蒙特卡洛采样结合起来。这种方法的一个特例是集成卡尔曼平滑，它依赖于短尾正态分布。我们通过自主车辆运动规划的仿真实验证明了所提出方法在规划、采样效率和约束满足方面的优势，相较于集成卡尔曼平滑。尽管该工作专注于运动规划，但它指出了重尾分布在增强机器人领域概率决策方面的更广泛潜力。 

---
# Bresa: Bio-inspired Reflexive Safe Reinforcement Learning for Contact-Rich Robotic Tasks 

**Title (ZH)**: Bresa：基于生物启发的反应式安全强化学习方法，适用于高接触机器人任务 

**Authors**: Heng Zhang, Gokhan Solak, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2503.21989)  

**Abstract**: Ensuring safety in reinforcement learning (RL)-based robotic systems is a critical challenge, especially in contact-rich tasks within unstructured environments. While the state-of-the-art safe RL approaches mitigate risks through safe exploration or high-level recovery mechanisms, they often overlook low-level execution safety, where reflexive responses to potential hazards are crucial. Similarly, variable impedance control (VIC) enhances safety by adjusting the robot's mechanical response, yet lacks a systematic way to adapt parameters, such as stiffness and damping throughout the task. In this paper, we propose Bresa, a Bio-inspired Reflexive Hierarchical Safe RL method inspired by biological reflexes. Our method decouples task learning from safety learning, incorporating a safety critic network that evaluates action risks and operates at a higher frequency than the task solver. Unlike existing recovery-based methods, our safety critic functions at a low-level control layer, allowing real-time intervention when unsafe conditions arise. The task-solving RL policy, running at a lower frequency, focuses on high-level planning (decision-making), while the safety critic ensures instantaneous safety corrections. We validate Bresa on multiple tasks including a contact-rich robotic task, demonstrating its reflexive ability to enhance safety, and adaptability in unforeseen dynamic environments. Our results show that Bresa outperforms the baseline, providing a robust and reflexive safety mechanism that bridges the gap between high-level planning and low-level execution. Real-world experiments and supplementary material are available at project website this https URL. 

**Abstract (ZH)**: 确保基于强化学习（RL）的机器人系统安全是关键挑战，尤其在无结构环境中进行接触丰富的任务时。现有的先进安全RL方法通过安全探索或高层恢复机制减轻风险，但往往忽略了低层级执行安全，而在潜在危害面前的反射性响应至关重要。同样，阻抗调节控制（VIC）通过调整机器人机械响应来增强安全，但缺乏系统的方法来适应参数，如刚度和阻尼。在本文中，我们提出了Bresa，一种受生物反射启发的反射性层次安全RL方法。该方法将任务学习与安全学习分离，引入了一个安全评判网络来评估动作风险，其运行频率高于任务解决器。与现有的基于恢复的方法不同，我们的安全评判在网络控制层工作，可以在不安全条件出现时实时干预。任务解决的RL策略以较低的频率运行，专注于高层规划（决策制定），而安全评判确保即时的安全修正。我们通过包括接触丰富的机器人任务在内的多项任务验证了Bresa，展示了其反射性增强安全能力和在未预见的动态环境中的适应性。我们的结果表明，Bresa优于基线，提供了将高层规划与低层执行联系起来的稳健且反射性的安全机制。实验证明和补充材料可在项目网站上获取：this https URL。 

---
# Pretrained Bayesian Non-parametric Knowledge Prior in Robotic Long-Horizon Reinforcement Learning 

**Title (ZH)**: 预训练贝叶斯非参数先验知识在机器人长时 horizon 强化学习中的应用 

**Authors**: Yuan Meng, Xiangtong Yao, Kejia Chen, Yansong Wu, Liding Zhang, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.21975)  

**Abstract**: Reinforcement learning (RL) methods typically learn new tasks from scratch, often disregarding prior knowledge that could accelerate the learning process. While some methods incorporate previously learned skills, they usually rely on a fixed structure, such as a single Gaussian distribution, to define skill priors. This rigid assumption can restrict the diversity and flexibility of skills, particularly in complex, long-horizon tasks. In this work, we introduce a method that models potential primitive skill motions as having non-parametric properties with an unknown number of underlying features. We utilize a Bayesian non-parametric model, specifically Dirichlet Process Mixtures, enhanced with birth and merge heuristics, to pre-train a skill prior that effectively captures the diverse nature of skills. Additionally, the learned skills are explicitly trackable within the prior space, enhancing interpretability and control. By integrating this flexible skill prior into an RL framework, our approach surpasses existing methods in long-horizon manipulation tasks, enabling more efficient skill transfer and task success in complex environments. Our findings show that a richer, non-parametric representation of skill priors significantly improves both the learning and execution of challenging robotic tasks. All data, code, and videos are available at this https URL. 

**Abstract (ZH)**: 强化学习方法通常从头学习新任务，往往忽视先前的知识，这可能会加快学习过程。虽然有一些方法整合了之前学习的技能，但它们通常依赖于固定结构，如单一高斯分布，来定义技能先验。这种刚性假设可能会限制技能的多样性和灵活性，特别是在复杂、长期的任务中。在本文中，我们提出了一种方法，将潜在的原始技能运动建模为具有非参数性质且潜在特征数量未知的运动。我们利用Dirichlet过程混合模型进行先验训练，并结合出生和合并启发式算法，以有效地捕捉技能的多样性。此外，学习到的技能在先验空间中明确可追踪，增强了可解释性和控制能力。通过将这种灵活的技能先验整合到RL框架中，我们的方法在长期操作任务中超越了现有方法，使在复杂环境中的技能转移和任务成功更高效。我们的研究表明，更丰富、非参数化的技能先验表示显著提高了挑战性机器人任务的 learning 和执行。所有数据、代码和视频均可通过以下链接访问：this https URL。 

---
# Data-Agnostic Robotic Long-Horizon Manipulation with Vision-Language-Guided Closed-Loop Feedback 

**Title (ZH)**: 基于视觉-语言引导的闭环反馈的无数据长时_horizon机械臂操作 

**Authors**: Yuan Meng, Xiangtong Yao, Haihui Ye, Yirui Zhou, Shengqiang Zhang, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.21969)  

**Abstract**: Recent advances in language-conditioned robotic manipulation have leveraged imitation and reinforcement learning to enable robots to execute tasks from human commands. However, these methods often suffer from limited generalization, adaptability, and the lack of large-scale specialized datasets, unlike data-rich domains such as computer vision, making long-horizon task execution challenging. To address these gaps, we introduce DAHLIA, a data-agnostic framework for language-conditioned long-horizon robotic manipulation, leveraging large language models (LLMs) for real-time task planning and execution. DAHLIA employs a dual-tunnel architecture, where an LLM-powered planner collaborates with co-planners to decompose tasks and generate executable plans, while a reporter LLM provides closed-loop feedback, enabling adaptive re-planning and ensuring task recovery from potential failures. Moreover, DAHLIA integrates chain-of-thought (CoT) in task reasoning and temporal abstraction for efficient action execution, enhancing traceability and robustness. Our framework demonstrates state-of-the-art performance across diverse long-horizon tasks, achieving strong generalization in both simulated and real-world scenarios. Videos and code are available at this https URL. 

**Abstract (ZH)**: 语言条件下的机器人 manipulatiion 最近进展：一种利用大型语言模型的数据agnostic框架（DAHLIA）实现长时 horizon 任务执行 

---
# ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning 

**Title (ZH)**: ManipTrans: 通过残差学习高效实现灵巧双手操作转移 

**Authors**: Kailin Li, Puhao Li, Tengyu Liu, Yuyang Li, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21860)  

**Abstract**: Human hands play a central role in interacting, motivating increasing research in dexterous robotic manipulation. Data-driven embodied AI algorithms demand precise, large-scale, human-like manipulation sequences, which are challenging to obtain with conventional reinforcement learning or real-world teleoperation. To address this, we introduce ManipTrans, a novel two-stage method for efficiently transferring human bimanual skills to dexterous robotic hands in simulation. ManipTrans first pre-trains a generalist trajectory imitator to mimic hand motion, then fine-tunes a specific residual module under interaction constraints, enabling efficient learning and accurate execution of complex bimanual tasks. Experiments show that ManipTrans surpasses state-of-the-art methods in success rate, fidelity, and efficiency. Leveraging ManipTrans, we transfer multiple hand-object datasets to robotic hands, creating DexManipNet, a large-scale dataset featuring previously unexplored tasks like pen capping and bottle unscrewing. DexManipNet comprises 3.3K episodes of robotic manipulation and is easily extensible, facilitating further policy training for dexterous hands and enabling real-world deployments. 

**Abstract (ZH)**: 人类双手在交互中发挥核心作用，推动了灵巧机器人操纵研究的增加。基于数据的实体AI算法需要精确、大规模、类似人类的操纵序列，这给传统强化学习或真实世界远程操作带来了挑战。为了解决这一问题，我们提出了ManipTrans，一种新型的两阶段方法，用于高效地将人类双手技能转移到仿真中的灵巧机器人手中。ManipTrans 首先预训练一个通用轨迹模仿器来模仿手部运动，然后在交互约束下微调一个特定的残差模块，从而实现复杂双手任务的高效学习和精确执行。实验表明，ManipTrans 在成功率、保真度和效率方面超越了现有最先进的方法。利用ManipTrans，我们将多个手-物体数据集转移到机器人手中，创建了DexManipNet，这是一个大规模数据集，包含了诸如笔盖帽和瓶盖拧开等之前未探索的任务。该数据集包含了3300多个机器人操纵回合，并且易于扩展，有利于进一步训练灵巧手的策略并使其在真实世界中部署。 

---
# Scaling Laws of Scientific Discovery with AI and Robot Scientists 

**Title (ZH)**: AI和机器人科学家在科学发现中的规模律 

**Authors**: Pengsong Zhang, Heng Zhang, Huazhe Xu, Renjun Xu, Zhenting Wang, Cong Wang, Animesh Garg, Zhibin Li, Arash Ajoudani, Xinyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22444)  

**Abstract**: The rapid evolution of scientific inquiry highlights an urgent need for groundbreaking methodologies that transcend the limitations of traditional research. Conventional approaches, bogged down by manual processes and siloed expertise, struggle to keep pace with the demands of modern discovery. We envision an autonomous generalist scientist (AGS) system-a fusion of agentic AI and embodied robotics-that redefines the research lifecycle. This system promises to autonomously navigate physical and digital realms, weaving together insights from disparate disciplines with unprecedented efficiency. By embedding advanced AI and robot technologies into every phase-from hypothesis formulation to peer-ready manuscripts-AGS could slash the time and resources needed for scientific research in diverse field. We foresee a future where scientific discovery follows new scaling laws, driven by the proliferation and sophistication of such systems. As these autonomous agents and robots adapt to extreme environments and leverage a growing reservoir of knowledge, they could spark a paradigm shift, pushing the boundaries of what's possible and ushering in an era of relentless innovation. 

**Abstract (ZH)**: 快速发展的科学研究突显了超越传统研究局限的颠覆性方法论的迫切需求。常规方法因手工流程和孤岛式专业知识而受困，难以跟上现代发现的需求。我们设想一种自主通才科学家（AGS）系统——结合代理AI和 embodiment robotics的融合——重新定义研究生命周期。该系统承诺能够自主穿梭于物理和数字领域，以前所未有的效率整合来自不同学科的见解。通过将先进的AI和机器人技术嵌入研究的每一阶段——从假设制定到可同行评审的手稿——AGS有望大幅减少在不同领域进行科学研究所需的时间和资源。我们预见一个未来，在这种系统普及和复杂性提高的推动下，科学发现将遵循新的扩展定律。随着这些自主代理和机器人适应极端环境并利用日益增多的知识库，它们可能引发范式转变，推动可能的边界，引领持续创新的时代。 

---
# CRLLK: Constrained Reinforcement Learning for Lane Keeping in Autonomous Driving 

**Title (ZH)**: CRLLK: 受约束的强化学习在自动驾驶中的车道保持 

**Authors**: Xinwei Gao, Arambam James Singh, Gangadhar Royyuru, Michael Yuhas, Arvind Easwaran  

**Link**: [PDF](https://arxiv.org/pdf/2503.22248)  

**Abstract**: Lane keeping in autonomous driving systems requires scenario-specific weight tuning for different objectives. We formulate lane-keeping as a constrained reinforcement learning problem, where weight coefficients are automatically learned along with the policy, eliminating the need for scenario-specific tuning. Empirically, our approach outperforms traditional RL in efficiency and reliability. Additionally, real-world demonstrations validate its practical value for real-world autonomous driving. 

**Abstract (ZH)**: 自主驾驶系统中的车道保持需要针对不同目标进行场景特定的权重调优。我们将车道保持建模为一个受约束的强化学习问题，在此问题中，权重系数与策略一起自动学习，消除了场景特定的调优需求。实验结果表明，与传统强化学习相比，我们的方法在效率和可靠性方面表现更优。此外，实际应用场景的演示验证了其在实际自主驾驶中的实用价值。 

---
# Deep Depth Estimation from Thermal Image: Dataset, Benchmark, and Challenges 

**Title (ZH)**: 基于热成像的深度估计：数据集、基准和挑战 

**Authors**: Ukcheol Shin, Jinsun Park  

**Link**: [PDF](https://arxiv.org/pdf/2503.22060)  

**Abstract**: Achieving robust and accurate spatial perception under adverse weather and lighting conditions is crucial for the high-level autonomy of self-driving vehicles and robots. However, existing perception algorithms relying on the visible spectrum are highly affected by weather and lighting conditions. A long-wave infrared camera (i.e., thermal imaging camera) can be a potential solution to achieve high-level robustness. However, the absence of large-scale datasets and standardized benchmarks remains a significant bottleneck to progress in active research for robust visual perception from thermal images. To this end, this manuscript provides a large-scale Multi-Spectral Stereo (MS$^2$) dataset that consists of stereo RGB, stereo NIR, stereo thermal, stereo LiDAR data, and GNSS/IMU information along with semi-dense depth ground truth. MS$^2$ dataset includes 162K synchronized multi-modal data pairs captured across diverse locations (e.g., urban city, residential area, campus, and high-way road) at different times (e.g., morning, daytime, and nighttime) and under various weather conditions (e.g., clear-sky, cloudy, and rainy). Secondly, we conduct a thorough evaluation of monocular and stereo depth estimation networks across RGB, NIR, and thermal modalities to establish standardized benchmark results on MS$^2$ depth test sets (e.g., day, night, and rainy). Lastly, we provide in-depth analyses and discuss the challenges revealed by the benchmark results, such as the performance variability for each modality under adverse conditions, domain shift between different sensor modalities, and potential research direction for thermal perception. Our dataset and source code are publicly available at this https URL and this https URL. 

**Abstract (ZH)**: 在恶劣天气和光照条件下的稳健准确的空间知觉对于自动驾驶车辆和机器人的高级自主至关重要。然而，依赖可见光谱的现有感知算法在受到天气和光照条件的影响下表现较差。长波红外相机（即热成像相机）可能是实现高度稳健性的潜在解决方案。然而，大型数据集和标准化基准的缺乏仍然是稳健热成像感知研究中的重要瓶颈。为此，本文提供了一个大规模的多光谱立体（MS$^2$）数据集，该数据集包含立体RGB、立体近红外、立体热成像、立体LiDAR数据以及GNSS/IMU信息，并附有半密集深度地面真值。MS$^2$数据集包括在不同地点（例如城市、住宅区、校园、高速公路）和不同时段（例如早晨、白天、夜间）以及各种天气条件下（例如晴朗、多云、雨天）同步采集的162,000多组多模态数据对。其次，我们在RGB、近红外和热成像模态下对单目和立体深度估计网络进行了全面评估，以在MS$^2$深度测试集（例如白天、夜晚和雨天）上确立标准化基准结果。最后，我们进行了深入分析，并讨论了基准结果揭示的挑战，例如在不利条件下每个模态的性能变异性、不同传感器模态之间的领域转移以及热感知研究潜在的研究方向。我们的数据集和源代码可在以下网址获取：this https URL 和 this https URL。 

---
# CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models 

**Title (ZH)**: CoT-VLA：视觉链式思维推理在视觉语言行动模型中的应用 

**Authors**: Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, Tsung-Yi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.22020)  

**Abstract**: Vision-language-action models (VLAs) have shown potential in leveraging pretrained vision-language models and diverse robot demonstrations for learning generalizable sensorimotor control. While this paradigm effectively utilizes large-scale data from both robotic and non-robotic sources, current VLAs primarily focus on direct input--output mappings, lacking the intermediate reasoning steps crucial for complex manipulation tasks. As a result, existing VLAs lack temporal planning or reasoning capabilities. In this paper, we introduce a method that incorporates explicit visual chain-of-thought (CoT) reasoning into vision-language-action models (VLAs) by predicting future image frames autoregressively as visual goals before generating a short action sequence to achieve these goals. We introduce CoT-VLA, a state-of-the-art 7B VLA that can understand and generate visual and action tokens. Our experimental results demonstrate that CoT-VLA achieves strong performance, outperforming the state-of-the-art VLA model by 17% in real-world manipulation tasks and 6% in simulation benchmarks. Project website: this https URL 

**Abstract (ZH)**: 视觉-语言-行动模型（VLAs）展示了通过利用预训练的视觉-语言模型和多样化的机器人示范来学习可泛化的传感器运动控制的潜力。尽管这一范式有效利用了来自机器人和非机器人来源的大规模数据，目前的VLAs主要专注于直接的输入-输出映射，而缺乏完成复杂操作任务所必需的中间推理步骤。因此，现有的VLAs缺乏时间规划或推理能力。在本文中，我们引入了一种方法，通过预测未来图像帧作为视觉目标，然后生成短暂的行动序列以实现这些目标，将显式的视觉链式推理（CoT）整合到视觉-语言-行动模型（VLAs）中。我们提出了CoT-VLA，这是一种最先进的7B VLA，能够理解和生成视觉和行动标记。我们的实验结果表明，CoT-VLA表现出色，在真实世界的操作任务中比最先进的VLAS模型高出17%，在仿真基准测试中高出6%。项目网址：这个 https URL。 

---
# Beyond Omakase: Designing Shared Control for Navigation Robots with Blind People 

**Title (ZH)**: 超越 Omakase：为视障人士设计导航机器人共享控制方案 

**Authors**: Rie Kamikubo, Seita Kayukawa, Yuka Kaniwa, Allan Wang, Hernisa Kacorri, Hironobu Takagi, Chieko Asakawa  

**Link**: [PDF](https://arxiv.org/pdf/2503.21997)  

**Abstract**: Autonomous navigation robots can increase the independence of blind people but often limit user control, following what is called in Japanese an "omakase" approach where decisions are left to the robot. This research investigates ways to enhance user control in social robot navigation, based on two studies conducted with blind participants. The first study, involving structured interviews (N=14), identified crowded spaces as key areas with significant social challenges. The second study (N=13) explored navigation tasks with an autonomous robot in these environments and identified design strategies across different modes of autonomy. Participants preferred an active role, termed the "boss" mode, where they managed crowd interactions, while the "monitor" mode helped them assess the environment, negotiate movements, and interact with the robot. These findings highlight the importance of shared control and user involvement for blind users, offering valuable insights for designing future social navigation robots. 

**Abstract (ZH)**: 自主导航机器人可以增加盲人独立性，但 often limit user control，采取日本所说的“omakase”方式，由机器人自行作出决策。本研究基于与盲人参与者进行的两项研究，探索增强社会机器人导航中用户控制的方法。第一项研究（N=14）通过结构化访谈识别出拥挤空间为关键的社会挑战区域。第二项研究（N=13）探讨了在这些环境中使用自主机器人完成导航任务，并提出了不同类型自主模式下的设计策略。参与者偏好一种主动模式，称为“老板”模式，他们在该模式下管理人群互动，而“监控”模式则帮助他们评估环境、规划移动路线并与机器人互动。这些发现突出了盲人用户中共同控制和用户参与的重要性，为设计未来社会导航机器人提供了宝贵见解。 

---
# Threshold Adaptation in Spiking Networks Enables Shortest Path Finding and Place Disambiguation 

**Title (ZH)**: 阈值自适应在脉冲神经网络中实现最短路径寻找和位置模糊区分 

**Authors**: Robin Dietrich, Tobias Fischer, Nicolai Waniek, Nico Reeb, Michael Milford, Alois Knoll, Adam D. Hines  

**Link**: [PDF](https://arxiv.org/pdf/2503.21795)  

**Abstract**: Efficient spatial navigation is a hallmark of the mammalian brain, inspiring the development of neuromorphic systems that mimic biological principles. Despite progress, implementing key operations like back-tracing and handling ambiguity in bio-inspired spiking neural networks remains an open challenge. This work proposes a mechanism for activity back-tracing in arbitrary, uni-directional spiking neuron graphs. We extend the existing replay mechanism of the spiking hierarchical temporal memory (S-HTM) by our spike timing-dependent threshold adaptation (STDTA), which enables us to perform path planning in networks of spiking neurons. We further present an ambiguity dependent threshold adaptation (ADTA) for identifying places in an environment with less ambiguity, enhancing the localization estimate of an agent. Combined, these methods enable efficient identification of the shortest path to an unambiguous target. Our experiments show that a network trained on sequences reliably computes shortest paths with fewer replays than the steps required to reach the target. We further show that we can identify places with reduced ambiguity in multiple, similar environments. These contributions advance the practical application of biologically inspired sequential learning algorithms like the S-HTM towards neuromorphic localization and navigation. 

**Abstract (ZH)**: 高效的空间导航是哺乳动物大脑的 hallmark，启发了模仿生物原则的神经形态系统的开发。尽管取得了进展，但在生物启发的脉冲神经网络中实现回溯操作和处理模糊性仍然是一个开放挑战。本工作提出了一种机制，用于任意单向脉冲神经元图中的活动回溯。我们通过脉冲时序依赖阈值适应（STDTA）扩展了现有的脉冲多层次时间记忆（S-HTM）的回放机制，从而能够在脉冲神经元网络中进行路径规划。我们还提出了依赖模糊性的阈值适应（ADTA），用于识别环境中的模糊性较低的地点，增强代理的定位估计。结合这些方法，能够高效地识别到明确目标的最短路径。实验结果表明，训练在网络序列上可以比到达目标所需的步数更少地回放计算出最短路径。此外，我们还展示了可以在多个相似环境中识别模糊性较低的地点。这些贡献推动了像S-HTM这样的生物启发序贯学习算法在神经形态定位和导航中的实际应用。 

---
