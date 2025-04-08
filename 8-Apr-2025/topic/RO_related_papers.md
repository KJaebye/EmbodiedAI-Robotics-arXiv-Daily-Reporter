# Segmented Trajectory Optimization for Autonomous Parking in Unstructured Environments 

**Title (ZH)**: 非结构化环境中自主泊车分段轨迹优化 

**Authors**: Hang Yu, Renjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.05041)  

**Abstract**: This paper presents a Segmented Trajectory Optimization (STO) method for autonomous parking, which refines an initial trajectory into a dynamically feasible and collision-free one using an iterative SQP-based approach. STO maintains the maneuver strategy of the high-level global planner while allowing curvature discontinuities at switching points to improve maneuver efficiency. To ensure safety, a convex corridor is constructed via GJK-accelerated ellipse shrinking and expansion, serving as safety constraints in each iteration. Numerical simulations in perpendicular and reverse-angled parking scenarios demonstrate that STO enhances maneuver efficiency while ensuring safety. Moreover, computational performance confirms its practicality for real-world applications. 

**Abstract (ZH)**: 基于迭代SQP方法的分段轨迹优化在自动驾驶泊车中的应用 

---
# CONCERT: a Modular Reconfigurable Robot for Construction 

**Title (ZH)**: CONCERT：一种可重构模块化建筑机器人 

**Authors**: Luca Rossini, Edoardo Romiti, Arturo Laurenzi, Francesco Ruscelli, Marco Ruzzon, Luca Covizzi, Lorenzo Baccelliere, Stefano Carrozzo, Michael Terzer, Marco Magri, Carlo Morganti, Maolin Lei, Liana Bertoni, Diego Vedelago, Corrado Burchielli, Stefano Cordasco, Luca Muratore, Andrea Giusti, Nikos Tsagarakis  

**Link**: [PDF](https://arxiv.org/pdf/2504.04998)  

**Abstract**: This paper presents CONCERT, a fully reconfigurable modular collaborative robot (cobot) for multiple on-site operations in a construction site. CONCERT has been designed to support human activities in construction sites by leveraging two main characteristics: high-power density motors and modularity. In this way, the robot is able to perform a wide range of highly demanding tasks by acting as a co-worker of the human operator or by autonomously executing them following user instructions. Most of its versatility comes from the possibility of rapidly changing its kinematic structure by adding or removing passive or active modules. In this way, the robot can be set up in a vast set of morphologies, consequently changing its workspace and capabilities depending on the task to be executed. In the same way, distal end-effectors can be replaced for the execution of different operations. This paper also includes a full description of the software pipeline employed to automatically discover and deploy the robot morphology. Specifically, depending on the modules installed, the robot updates the kinematic, dynamic, and geometric parameters, taking into account the information embedded in each module. In this way, we demonstrate how the robot can be fully reassembled and made operational in less than ten minutes. We validated the CONCERT robot across different use cases, including drilling, sanding, plastering, and collaborative transportation with obstacle avoidance, all performed in a real construction site scenario. We demonstrated the robot's adaptivity and performance in multiple scenarios characterized by different requirements in terms of power and workspace. CONCERT has been designed and built by the Humanoid and Human-Centered Mechatronics Laboratory (HHCM) at the Istituto Italiano di Tecnologia in the context of the European Project Horizon 2020 CONCERT. 

**Abstract (ZH)**: CONCERT：一种用于施工现场多任务操作的全可重构模块化协作机器人 

---
# BayesCPF: Enabling Collective Perception in Robot Swarms with Degrading Sensors 

**Title (ZH)**: BayesCPF：在传感器退化的条件下使机器人 Swarm 实现集体感知 

**Authors**: Khai Yi Chin, Carlo Pinciroli  

**Link**: [PDF](https://arxiv.org/pdf/2504.04774)  

**Abstract**: The collective perception problem -- where a group of robots perceives its surroundings and comes to a consensus on an environmental state -- is a fundamental problem in swarm robotics. Past works studying collective perception use either an entire robot swarm with perfect sensing or a swarm with only a handful of malfunctioning members. A related study proposed an algorithm that does account for an entire swarm of unreliable robots but assumes that the sensor faults are known and remain constant over time. To that end, we build on that study by proposing the Bayes Collective Perception Filter (BayesCPF) that enables robots with continuously degrading sensors to accurately estimate the fill ratio -- the rate at which an environmental feature occurs. Our main contribution is the Extended Kalman Filter within the BayesCPF, which helps swarm robots calibrate for their time-varying sensor degradation. We validate our method across different degradation models, initial conditions, and environments in simulated and physical experiments. Our findings show that, regardless of degradation model assumptions, fill ratio estimation using the BayesCPF is competitive to the case if the true sensor accuracy is known, especially when assumptions regarding the model and initial sensor accuracy levels are preserved. 

**Abstract (ZH)**: 集体感知问题——一群机器人感知其周围环境并就环境状态达成一致的过程是群集机器人领域的一个基本问题。现有的集体感知研究要么假设整个机器人群具备完美的传感能力，要么仅考虑少量功能失效的成员。相关研究提出了一种算法来处理整个由不可靠机器人组成的群集，但假设传感器故障是已知且恒定的。在此基础上，我们提出了一种称为贝叶斯集体感知滤波器（BayesCPF）的方法，该方法能够使具有持续退化传感器的机器人准确估计填充率——即环境特征出现的速率。我们的主要贡献是BayesCPF中的扩展Kalman滤波器，它帮助群集机器人校准其时变传感器退化情况。我们在模拟和物理实验中对不同退化模型、初始条件和环境进行了验证。研究结果表明，无论假设何种退化模型，使用BayesCPF进行填充率估计都能在已知真实传感器精度的情况下保持竞争力，尤其是在模型和初始传感器精度水平的假设保持一致的情况下。 

---
# Extended URDF: Accounting for parallel mechanism in robot description 

**Title (ZH)**: 扩展的URDF：考虑并联机构的机器人描述 

**Authors**: Virgile Batto, Ludovic de Matteïs, Nicolas Mansard  

**Link**: [PDF](https://arxiv.org/pdf/2504.04767)  

**Abstract**: Robotic designs played an important role in recent advances by providing powerful robots with complex mechanics. Many recent systems rely on parallel actuation to provide lighter limbs and allow more complex motion. However, these emerging architectures fall outside the scope of most used description formats, leading to difficulties when designing, storing, and sharing the models of these systems. This paper introduces an extension to the widely used Unified Robot Description Format (URDF) to support closed-loop kinematic structures. Our approach relies on augmenting URDF with minimal additional information to allow more efficient modeling of complex robotic systems while maintaining compatibility with existing design and simulation frameworks. This method sets the basic requirement for a description format to handle parallel mechanisms efficiently. We demonstrate the applicability of our approach by providing an open-source collection of parallel robots, along with tools for generating and parsing this extended description format. The proposed extension simplifies robot modeling, reduces redundancy, and improves usability for advanced robotic applications. 

**Abstract (ZH)**: 机器人设计在 recent advances 中发挥了重要作用，通过提供具有复杂机械结构的强大机器人。许多近期系统依赖并行驱动以实现更轻的肢体和更复杂的操作。然而，这些新兴架构超出了大多数使用描述格式的范围，导致在设计、存储和分享这些系统的模型时遇到困难。本文介绍了对广泛使用的统一机器人描述格式（URDF）的一个扩展，以支持闭环运动学结构。我们的方法通过在URDF中添加最少的额外信息来提高复杂机器人系统建模效率，同时保持与现有设计和仿真框架的兼容性。该方法为高效处理并行机构设定了基本要求。我们通过提供一个开源的并行机器人集合以及生成和解析此扩展描述格式的工具，展示了我们方法的应用性。提出的扩展简化了机器人建模，减少了冗余，并提高了高级机器人应用的易用性。 

---
# B4P: Simultaneous Grasp and Motion Planning for Object Placement via Parallelized Bidirectional Forests and Path Repair 

**Title (ZH)**: B4P: 同时进行物体放置时的抓取与运动规划方法_via_并行双向森林及路径修复_ 

**Authors**: Benjamin H. Leebron, Kejia Ren, Yiting Chen, Kaiyu Hang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04598)  

**Abstract**: Robot pick and place systems have traditionally decoupled grasp, placement, and motion planning to build sequential optimization pipelines with the assumption that the individual components will be able to work together. However, this separation introduces sub-optimality, as grasp choices may limit or even prohibit feasible motions for a robot to reach the target placement pose, particularly in cluttered environments with narrow passages. To this end, we propose a forest-based planning framework to simultaneously find grasp configurations and feasible robot motions that explicitly satisfy downstream placement configurations paired with the selected grasps. Our proposed framework leverages a bidirectional sampling-based approach to build a start forest, rooted at the feasible grasp regions, and a goal forest, rooted at the feasible placement regions, to facilitate the search through randomly explored motions that connect valid pairs of grasp and placement trees. We demonstrate that the framework's inherent parallelism enables superlinear speedup, making it scalable for applications for redundant robot arms (e.g., 7 Degrees of Freedom) to work efficiently in highly cluttered environments. Extensive experiments in simulation demonstrate the robustness and efficiency of the proposed framework in comparison with multiple baselines under diverse scenarios. 

**Abstract (ZH)**: 基于森林的规划框架：同时寻找满足选握态与下游放置态的握持配置和可行机器人运动 

---
# DexSinGrasp: Learning a Unified Policy for Dexterous Object Singulation and Grasping in Cluttered Environments 

**Title (ZH)**: DexSinGrasp: 学习在拥挤环境中进行灵巧物体分拣和抓取的统一策略 

**Authors**: Lixin Xu, Zixuan Liu, Zhewei Gui, Jingxiang Guo, Zeyu Jiang, Zhixuan Xu, Chongkai Gao, Lin Shao  

**Link**: [PDF](https://arxiv.org/pdf/2504.04516)  

**Abstract**: Grasping objects in cluttered environments remains a fundamental yet challenging problem in robotic manipulation. While prior works have explored learning-based synergies between pushing and grasping for two-fingered grippers, few have leveraged the high degrees of freedom (DoF) in dexterous hands to perform efficient singulation for grasping in cluttered settings. In this work, we introduce DexSinGrasp, a unified policy for dexterous object singulation and grasping. DexSinGrasp enables high-dexterity object singulation to facilitate grasping, significantly improving efficiency and effectiveness in cluttered environments. We incorporate clutter arrangement curriculum learning to enhance success rates and generalization across diverse clutter conditions, while policy distillation enables a deployable vision-based grasping strategy. To evaluate our approach, we introduce a set of cluttered grasping tasks with varying object arrangements and occlusion levels. Experimental results show that our method outperforms baselines in both efficiency and grasping success rate, particularly in dense clutter. Codes, appendix, and videos are available on our project website this https URL. 

**Abstract (ZH)**: 在杂乱环境中的物体抓取仍然是机器人操作中的一个基础但具有挑战性的问题。尽管先前的工作探索了推拿和抓取之间基于学习的协同作用以适用于两指夹持器，但很少有工作利用灵巧手的高自由度来高效地实现杂乱环境中的物体分离与抓取。在本文中，我们引入了DexSinGrasp，一种统一的灵巧物体分离与抓取策略。DexSinGrasp通过高灵巧度的物体分离来促进抓取，显著提高了杂乱环境中的效率和有效性。我们通过杂乱排列课程学习来增强成功率，并在多种杂乱条件下实现泛化，同时策略蒸馏使基于视觉的抓取策略可部署。为了评估我们的方法，我们引入了一组具有不同物体排列和遮挡程度的抓取任务。实验结果表明，我们的方法在效率和抓取成功率方面均优于基线方法，特别是在密集杂乱环境中表现尤为突出。代码、附录和视频可在我们的项目网站上获得：this https URL。 

---
# A Self-Supervised Learning Approach with Differentiable Optimization for UAV Trajectory Planning 

**Title (ZH)**: 基于可微优化的自监督学习方法在无人机轨迹规划中的应用 

**Authors**: Yufei Jiang, Yuanzhu Zhan, Harsh Vardhan Gupta, Chinmay Borde, Junyi Geng  

**Link**: [PDF](https://arxiv.org/pdf/2504.04289)  

**Abstract**: While Unmanned Aerial Vehicles (UAVs) have gained significant traction across various fields, path planning in 3D environments remains a critical challenge, particularly under size, weight, and power (SWAP) constraints. Traditional modular planning systems often introduce latency and suboptimal performance due to limited information sharing and local minima issues. End-to-end learning approaches streamline the pipeline by mapping sensory observations directly to actions but require large-scale datasets, face significant sim-to-real gaps, or lack dynamical feasibility. In this paper, we propose a self-supervised UAV trajectory planning pipeline that integrates a learning-based depth perception with differentiable trajectory optimization. A 3D cost map guides UAV behavior without expert demonstrations or human labels. Additionally, we incorporate a neural network-based time allocation strategy to improve the efficiency and optimality. The system thus combines robust learning-based perception with reliable physics-based optimization for improved generalizability and interpretability. Both simulation and real-world experiments validate our approach across various environments, demonstrating its effectiveness and robustness. Our method achieves a 31.33% improvement in position tracking error and 49.37% reduction in control effort compared to the state-of-the-art. 

**Abstract (ZH)**: 基于自我监督的无人机轨迹规划管道：结合基于学习的距离感知和可微轨迹优化 

---
# ORCA: An Open-Source, Reliable, Cost-Effective, Anthropomorphic Robotic Hand for Uninterrupted Dexterous Task Learning 

**Title (ZH)**: ORCA：一种开放源代码、可靠、低成本的人类仿生机器人手，用于不间断的灵巧任务学习 

**Authors**: Clemens C. Christoph, Maximilian Eberlein, Filippos Katsimalis, Arturo Roberti, Aristotelis Sympetheros, Michel R. Vogt, Davide Liconti, Chenyu Yang, Barnabas Gavin Cangan, Ronan J. Hinchet, Robert K. Katzschmann  

**Link**: [PDF](https://arxiv.org/pdf/2504.04259)  

**Abstract**: General-purpose robots should possess humanlike dexterity and agility to perform tasks with the same versatility as us. A human-like form factor further enables the use of vast datasets of human-hand interactions. However, the primary bottleneck in dexterous manipulation lies not only in software but arguably even more in hardware. Robotic hands that approach human capabilities are often prohibitively expensive, bulky, or require enterprise-level maintenance, limiting their accessibility for broader research and practical applications. What if the research community could get started with reliable dexterous hands within a day? We present the open-source ORCA hand, a reliable and anthropomorphic 17-DoF tendon-driven robotic hand with integrated tactile sensors, fully assembled in less than eight hours and built for a material cost below 2,000 CHF. We showcase ORCA's key design features such as popping joints, auto-calibration, and tensioning systems that significantly reduce complexity while increasing reliability, accuracy, and robustness. We benchmark the ORCA hand across a variety of tasks, ranging from teleoperation and imitation learning to zero-shot sim-to-real reinforcement learning. Furthermore, we demonstrate its durability, withstanding more than 10,000 continuous operation cycles - equivalent to approximately 20 hours - without hardware failure, the only constraint being the duration of the experiment itself. All design files, source code, and documentation will be available at this https URL. 

**Abstract (ZH)**: 通用机器人应具备人类般的灵巧性和敏捷性，以便像人类一样执行多样化任务。类似人类的手形进一步使人类手部互动的大量数据集的利用成为可能。然而，灵巧操作的主要瓶颈不仅在于软件，也许甚至更多地在于硬件。接近人类能力的手部机器人往往价格昂贵、笨重，或者需要企业级维护，限制了其在更广泛研究和实际应用中的 accessibility。如果研究社区能在一天内开始使用可靠的灵巧手部，该有多好？我们介绍了开源ORCA手部，这是一种可靠且仿人化的17自由度肌腱驱动的灵巧手，集成了触觉传感器，并可在不到八小时内完全组装完成，材料成本低于2000瑞士法郎。我们展示了ORCA的关键设计功能，如弹出关节、自校准系统和张力系统，这些功能大幅降低了复杂性，同时提高了可靠性和精度。我们跨多种任务benchmark了ORCA手部，从遥控操作和模仿学习到零样本模拟到现实的强化学习。此外，我们展示了其耐用性，在超过10,000次连续操作循环（约20小时）中未发生硬件故障，唯一的限制是实验本身的持续时间。所有设计文件、源代码和文档将在以下网址获得。 

---
# An Optimized Density-Based Lane Keeping System for A Cost-Efficient Autonomous Vehicle Platform: AurigaBot V1 

**Title (ZH)**: 一种面向经济型自主车辆平台的优化密度基车道保持系统：AurigaBot V1 

**Authors**: Farbod Younesi, Milad Rabiei, Soroush Keivanfard, Mohsen Sharifi, Marzieh Ghayour Najafabadi, Bahar Moadeli, Arshia Jafari, Mohammad Hossein Moaiyeri  

**Link**: [PDF](https://arxiv.org/pdf/2504.04217)  

**Abstract**: The development of self-driving cars has garnered significant attention from researchers, universities, and industries worldwide. Autonomous vehicles integrate numerous subsystems, including lane tracking, object detection, and vehicle control, which require thorough testing and validation. Scaled-down vehicles offer a cost-effective and accessible platform for experimentation, providing researchers with opportunities to optimize algorithms under constraints of limited computational power. This paper presents a four-wheeled autonomous vehicle platform designed to facilitate research and prototyping in autonomous driving. Key contributions include (1) a novel density-based clustering approach utilizing histogram statistics for landmark tracking, (2) a lateral controller, and (3) the integration of these innovations into a cohesive platform. Additionally, the paper explores object detection through systematic dataset augmentation and introduces an autonomous parking procedure. The results demonstrate the platform's effectiveness in achieving reliable lane tracking under varying lighting conditions, smooth trajectory following, and consistent object detection performance. Though developed for small-scale vehicles, these modular solutions are adaptable for full-scale autonomous systems, offering a versatile and cost-efficient framework for advancing research and industry applications. 

**Abstract (ZH)**: 自动驾驶汽车的发展已引起全球研究人员、大学和工业界的广泛关注。自动驾驶车辆集成了多个子系统，包括车道跟踪、物体检测和车辆控制，这些系统需要进行全面的测试和验证。缩小规模的车辆提供了一种经济且易于访问的实验平台，使研究人员能够在计算资源有限的情况下优化算法。本文介绍了用于促进自动驾驶研究和原型设计的四轮自动驾驶车辆平台。主要贡献包括：（1）一种基于直方图统计的新型基于密度的聚类方法用于地标跟踪，（2）一种横向控制器，以及（3）将这些创新整合到一个协调平台中。此外，本文还探讨了通过系统数据集扩增进行物体检测，并介绍了自动驾驶泊车程序。实验结果表明，该平台在不同光照条件下的车道跟踪可靠性、平滑轨迹跟踪以及一致的物体检测性能方面具有有效性。尽管是为小型车辆开发的，但这些模块化解决方案也可适应全尺寸自动驾驶系统，提供了一个灵活且成本效益高的研究和工业应用框架。 

---
# Passive Luminescent Bellows Mechanism 

**Title (ZH)**: 被动发光 bellows 机制 

**Authors**: Naoto Kikuta, Issei Onda, Kazuki Abe, Masahiro Watanabe, Kenjiro Tadakuma  

**Link**: [PDF](https://arxiv.org/pdf/2504.04194)  

**Abstract**: The use of robots in disaster sites has rapidly expanded, with soft robots attracting particular interest due to their flexibility and adaptability. They can navigate through narrow spaces and debris, facilitating efficient and safe operations. However, low visibility in such environments remains a challenge. This study aims to enhance the visibility of soft robots by developing and evaluating a passive luminescent exible actuator activated by a black light. Using Ecoex mixed with phosphorescent powder, we fabricated an actuator and confirmed its fluorescence phosphorescence and deformation ability. Furthermore the effects of the mixing ratio on optical and mechanical properties were assessed. 

**Abstract (ZH)**: 软体机器人在灾害现场的应用拓展及其被动发光柔性执行机构的研究：黑光激活下的荧光磷光和变形能力评估 

---
# Bistable SMA-driven engine for pulse-jet locomotion in soft aquatic robots 

**Title (ZH)**: 双稳态SMC驱动发动机在软水下机器人中的脉冲喷气驱动推进 

**Authors**: Graziella Bedenik, Antonio Morales, Supun Pieris, Barbara da Silva, John W. Kurelek, Melissa Greeff, Matthew Robertson  

**Link**: [PDF](https://arxiv.org/pdf/2504.03988)  

**Abstract**: This paper presents the design and experimental validation of a bio-inspired soft aquatic robot, the DilBot, which uses a bistable shape memory alloy-driven engine for pulse-jet locomotion. Drawing inspiration from the efficient swimming mechanisms of box jellyfish, the DilBot incorporates antagonistic shape memory alloy springs encapsulated in silicone insulation to achieve high-power propulsion. The innovative bistable mechanism allows continuous swimming cycles by storing and releasing energy in a controlled manner. Through free-swimming experiments and force characterization tests, we evaluated the DilBot's performance, achieving a peak speed of 158 mm/s and generating a maximum thrust of 5.59 N. This work demonstrates a novel approach to enhancing the efficiency of shape memory alloy actuators in aquatic environments. It presents a promising pathway for future applications in underwater environmental monitoring using robotic swarms. 

**Abstract (ZH)**: 基于 bistable 形状记忆合金驱动的仿生软水下机器人 DilBot 的设计与实验验证 

---
# Deep Learning-Enhanced Robotic Subretinal Injection with Real-Time Retinal Motion Compensation 

**Title (ZH)**: 基于深度学习的实时视网膜运动补偿视网膜下注射机器人增强技术 

**Authors**: Tianle Wu, Mojtaba Esfandiari, Peiyao Zhang, Russell H. Taylor, Peter Gehlbach, Iulian Iordachita  

**Link**: [PDF](https://arxiv.org/pdf/2504.03939)  

**Abstract**: Subretinal injection is a critical procedure for delivering therapeutic agents to treat retinal diseases such as age-related macular degeneration (AMD). However, retinal motion caused by physiological factors such as respiration and heartbeat significantly impacts precise needle positioning, increasing the risk of retinal pigment epithelium (RPE) damage. This paper presents a fully autonomous robotic subretinal injection system that integrates intraoperative optical coherence tomography (iOCT) imaging and deep learning-based motion prediction to synchronize needle motion with retinal displacement. A Long Short-Term Memory (LSTM) neural network is used to predict internal limiting membrane (ILM) motion, outperforming a Fast Fourier Transform (FFT)-based baseline model. Additionally, a real-time registration framework aligns the needle tip position with the robot's coordinate frame. Then, a dynamic proportional speed control strategy ensures smooth and adaptive needle insertion. Experimental validation in both simulation and ex vivo open-sky porcine eyes demonstrates precise motion synchronization and successful subretinal injections. The experiment achieves a mean tracking error below 16.4 {\mu}m in pre-insertion phases. These results show the potential of AI-driven robotic assistance to improve the safety and accuracy of retinal microsurgery. 

**Abstract (ZH)**: 基于光相干断层成像和深度学习的自主视网膜注射机器人系统：运动预测与精准运动同步 

---
# Energy Efficient Planning for Repetitive Heterogeneous Tasks in Precision Agriculture 

**Title (ZH)**: 精准农业中重复异构任务的能源高效规划 

**Authors**: Shuangyu Xie, Ken Goldberg, Dezhen Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.03938)  

**Abstract**: Robotic weed removal in precision agriculture introduces a repetitive heterogeneous task planning (RHTP) challenge for a mobile manipulator. RHTP has two unique characteristics: 1) an observe-first-and-manipulate-later (OFML) temporal constraint that forces a unique ordering of two different tasks for each target and 2) energy savings from efficient task collocation to minimize unnecessary movements. RHTP can be framed as a stochastic renewal process. According to the Renewal Reward Theorem, the expected energy usage per task cycle is the long-run average. Traditional task and motion planning focuses on feasibility rather than optimality due to the unknown object and obstacle position prior to execution. However, the known target/obstacle distribution in precision agriculture allows minimizing the expected energy usage. For each instance in this renewal process, we first compute task space partition, a novel data structure that computes all possibilities of task multiplexing and its probabilities with robot reachability. Then we propose a region-based set-coverage problem to formulate the RHTP as a mixed-integer nonlinear programming. We have implemented and solved RHTP using Branch-and-Bound solver. Compared to a baseline in simulations based on real field data, the results suggest a significant improvement in path length, number of robot stops, overall energy usage, and number of replans. 

**Abstract (ZH)**: 精准农业中机器人除草任务规划引入了一种重复异构任务规划（RHTP）挑战，为移动 manipulator 提出新的任务规划问题。RHTP 具有两大特性：1) 观察先行和操作后续（OFML）的时间约束，要求每个目标任务按特定顺序执行；2) 通过高效的任务共位减少不必要的移动从而节省能量。RHTP 可以作为随机更新过程进行建模。依据更新报酬定理，每轮任务的预期能耗是长期平均值。传统任务与运动规划主要关注可行性而非最优性，因为执行前目标和障碍物的位置未知。然而，精准农业中已知的目标/障碍物分布允许最大限度地减少预期能耗。对于这个随机更新过程中的每一个实例，首先计算任务空间分区，这是一种新的数据结构，计算所有任务复用的可能性及其概率，并结合机器人的可达性。然后提出基于区域的集合覆盖问题，将RHTP形式化为混合整数非线性规划问题。我们使用分支定界求解器实施和解决了RHTP。与基于实际农田数据的基线在仿真中的比较结果表明，在路径长度、机器人停顿次数、总体能耗和重新规划次数方面有显著改进。 

---
# Modeling of AUV Dynamics with Limited Resources: Efficient Online Learning Using Uncertainty 

**Title (ZH)**: 基于有限资源的自主 underwater 车辆动力学建模：利用不确定性进行高效在线学习 

**Authors**: Michal Tešnar, Bilal Wehbe, Matias Valdenegro-Toro  

**Link**: [PDF](https://arxiv.org/pdf/2504.04583)  

**Abstract**: Machine learning proves effective in constructing dynamics models from data, especially for underwater vehicles. Continuous refinement of these models using incoming data streams, however, often requires storage of an overwhelming amount of redundant data. This work investigates the use of uncertainty in the selection of data points to rehearse in online learning when storage capacity is constrained. The models are learned using an ensemble of multilayer perceptrons as they perform well at predicting epistemic uncertainty. We present three novel approaches: the Threshold method, which excludes samples with uncertainty below a specified threshold, the Greedy method, designed to maximize uncertainty among the stored points, and Threshold-Greedy, which combines the previous two approaches. The methods are assessed on data collected by an underwater vehicle Dagon. Comparison with baselines reveals that the Threshold exhibits enhanced stability throughout the learning process and also yields a model with the least cumulative testing loss. We also conducted detailed analyses on the impact of model parameters and storage size on the performance of the models, as well as a comparison of three different uncertainty estimation methods. 

**Abstract (ZH)**: 机器学习在从数据构建水下车辆动力学模型中证明有效，然而，使用有限存储能力下的连续数据流细化这些模型往往需要存储大量冗余数据。本工作研究在存储能力受限时，在在线学习中使用不确定性选择数据点进行重新训练的方法。模型使用多层感知机集成进行学习，因为它们在预测认识不确定性方面表现良好。我们提出了三种新颖的方法：阈值方法（排除低于指定阈值不确定性的样本）、贪心方法（旨在最大化存储点中的不确定性），以及结合前两者的方法（阈值-贪心）。这些方法在由水下车辆Dagon采集的数据上进行评估。与基线方法的比较表明，阈值方法在整个学习过程中表现出增强的稳定性，并且产生的模型具有最小的累积测试损失。我们还详细分析了模型参数和存储大小对模型性能的影响，并比较了三种不同的不确定性估计方法。 

---
# Risk-Aware Robot Control in Dynamic Environments Using Belief Control Barrier Functions 

**Title (ZH)**: 使用信念控制屏障函数的动态环境下具风险意识的机器人控制 

**Authors**: Shaohang Han, Matti Vahs, Jana Tumova  

**Link**: [PDF](https://arxiv.org/pdf/2504.04097)  

**Abstract**: Ensuring safety for autonomous robots operating in dynamic environments can be challenging due to factors such as unmodeled dynamics, noisy sensor measurements, and partial observability. To account for these limitations, it is common to maintain a belief distribution over the true state. This belief could be a non-parametric, sample-based representation to capture uncertainty more flexibly. In this paper, we propose a novel form of Belief Control Barrier Functions (BCBFs) specifically designed to ensure safety in dynamic environments under stochastic dynamics and a sample-based belief about the environment state. Our approach incorporates provable concentration bounds on tail risk measures into BCBFs, effectively addressing possible multimodal and skewed belief distributions represented by samples. Moreover, the proposed method demonstrates robustness against distributional shifts up to a predefined bound. We validate the effectiveness and real-time performance (approximately 1kHz) of the proposed method through two simulated underwater robotic applications: object tracking and dynamic collision avoidance. 

**Abstract (ZH)**: 确保自主机器人在动态环境中操作的安全性因未建模的动力学、嘈杂的传感器测量和部分可观测性等因素而具有挑战性。为了应对这些限制，通常需要维护对真实状态的信念分布。这种信念可以是非参数的样本基表示，以更灵活地捕捉不确定性。在本文中，我们提出了一种新型的信念控制屏障函数（Belief Control Barrier Functions，BCBFs），专门设计用于在随机动力学和基于样本的环境状态信念下确保动态环境中的安全性。我们的方法将可证明的尾部风险度量的集中界引入到BCBFs中，有效地解决了由样本表示的可能的多模态和偏斜信念分布。此外，提出的方法在预定义的界内表现出对分布偏移的鲁棒性。我们通过两个模拟的水下机器人应用（物体跟踪和动态避碰）验证了所提出方法的有效性和实时性能（约1kHz）。 

---
# A Geometric Approach For Pose and Velocity Estimation Using IMU and Inertial/Body-Frame Measurements 

**Title (ZH)**: 基于IMU和体帧测量的几何方法用于姿态和速度估计 

**Authors**: Sifeddine Benahmed, Soulaimane Berkane, Tarek Hamel  

**Link**: [PDF](https://arxiv.org/pdf/2504.03764)  

**Abstract**: This paper addresses accurate pose estimation (position, velocity, and orientation) for a rigid body using a combination of generic inertial-frame and/or body-frame measurements along with an Inertial Measurement Unit (IMU). By embedding the original state space, $\so \times \R^3 \times \R^3$, within the higher-dimensional Lie group $\sefive$, we reformulate the vehicle dynamics and outputs within a structured, geometric framework. In particular, this embedding enables a decoupling of the resulting geometric error dynamics: the translational error dynamics follow a structure similar to the error dynamics of a continuous-time Kalman filter, which allows for a time-varying gain design using the Riccati equation. Under the condition of uniform observability, we establish that the proposed observer design on $\sefive$ guarantees almost global asymptotic stability. We validate the approach in simulations for two practical scenarios: stereo-aided inertial navigation systems (INS) and GPS-aided INS. The proposed method significantly simplifies the design of nonlinear geometric observers for INS, providing a generalized and robust approach to state estimation. 

**Abstract (ZH)**: 基于广义惯性框架和/或体框架测量及惯性测量单元的刚体精确姿态估计 

---
# Optimizing UAV Aerial Base Station Flights Using DRL-based Proximal Policy Optimization 

**Title (ZH)**: 基于近端策略优化的深度 reinforcement 学习优化无人机高空基站飞行 

**Authors**: Mario Rico Ibanez, Azim Akhtarshenas, David Lopez-Perez, Giovanni Geraci  

**Link**: [PDF](https://arxiv.org/pdf/2504.03961)  

**Abstract**: Unmanned aerial vehicle (UAV)-based base stations offer a promising solution in emergencies where the rapid deployment of cutting-edge networks is crucial for maximizing life-saving potential. Optimizing the strategic positioning of these UAVs is essential for enhancing communication efficiency. This paper introduces an automated reinforcement learning approach that enables UAVs to dynamically interact with their environment and determine optimal configurations. By leveraging the radio signal sensing capabilities of communication networks, our method provides a more realistic perspective, utilizing state-of-the-art algorithm -- proximal policy optimization -- to learn and generalize positioning strategies across diverse user equipment (UE) movement patterns. We evaluate our approach across various UE mobility scenarios, including static, random, linear, circular, and mixed hotspot movements. The numerical results demonstrate the algorithm's adaptability and effectiveness in maintaining comprehensive coverage across all movement patterns. 

**Abstract (ZH)**: 基于无人机的基站（UAV基站）在紧急情况下提供了一种有希望的解决方案，因为快速部署尖端网络对于最大化生命救援潜力至关重要。优化这些无人机的战略性位置对于提高通信效率至关重要。本文介绍了一种自动强化学习方法，使无人机能够动态与环境交互并确定最优配置。借助通信网络的无线电信号感知能力，我们的方法提供了更现实的视角，并利用最新的算法——近端策略优化——来学习和泛化适用于不同用户设备（UE）运动模式的定位策略。我们在包括静态、随机、直线、圆周和混合热点移动在内的各种UE移动场景下评估了我们的方法。数值结果表明，该算法在所有运动模式下具有适应性和有效性，能够维持全面的覆盖范围。 

---
