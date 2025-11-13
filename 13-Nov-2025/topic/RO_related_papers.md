# SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation 

**Title (ZH)**: SpatialActor: 探索解耦的时空表示以实现鲁棒的机器人 manipulation 

**Authors**: Hao Shi, Bin Xie, Yingfei Liu, Yang Yue, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.09555)  

**Abstract**: Robotic manipulation requires precise spatial understanding to interact with objects in the real world. Point-based methods suffer from sparse sampling, leading to the loss of fine-grained semantics. Image-based methods typically feed RGB and depth into 2D backbones pre-trained on 3D auxiliary tasks, but their entangled semantics and geometry are sensitive to inherent depth noise in real-world that disrupts semantic understanding. Moreover, these methods focus on high-level geometry while overlooking low-level spatial cues essential for precise interaction. We propose SpatialActor, a disentangled framework for robust robotic manipulation that explicitly decouples semantics and geometry. The Semantic-guided Geometric Module adaptively fuses two complementary geometry from noisy depth and semantic-guided expert priors. Also, a Spatial Transformer leverages low-level spatial cues for accurate 2D-3D mapping and enables interaction among spatial features. We evaluate SpatialActor on multiple simulation and real-world scenarios across 50+ tasks. It achieves state-of-the-art performance with 87.4% on RLBench and improves by 13.9% to 19.4% under varying noisy conditions, showing strong robustness. Moreover, it significantly enhances few-shot generalization to new tasks and maintains robustness under various spatial perturbations. Project Page: this https URL 

**Abstract (ZH)**: 基于空间理解的机器人操作需要精确的空间认知来与真实世界的物体交互。基于点的方法由于稀疏采样而损失了细粒度语义。基于图像的方法通常将RGB和深度输入预训练于三维辅助任务的二维骨干网络，但它们的纠缠语义和几何结构对真实世界固有的深度噪声敏感，这干扰了语义理解。此外，这些方法关注高级几何结构，而忽略了对精确交互至关重要的低级空间线索。我们提出SpatialActor，一种松散耦合语义和几何的鲁棒机器人操作框架。语义引导几何模块自适应融合来自噪声深度和语义引导专家先验的互补几何。此外，空间变换器利用低级空间线索进行准确的二维-三维映射，并在空间特征之间实现交互。我们在50多个任务的多个仿真和真实世界场景中评估SpatialActor，其在RLBench上的性能达到最佳，为87.4%，在不同噪声条件下的性能提升13.9%至19.4%，显示出较强的鲁棒性。此外，它显著增强了基于少量样本的新任务泛化能力，同时在各种空间扰动下保持鲁棒性。项目页面：this https URL 

---
# LODESTAR: Degeneracy-Aware LiDAR-Inertial Odometry with Adaptive Schmidt-Kalman Filter and Data Exploitation 

**Title (ZH)**: LODESTAR：aware退化现象的LiDAR-惯性里程计，基于自适应Schmidt-Kalman滤波器和数据利用 

**Authors**: Eungchang Mason Lee, Kevin Christiansen Marsim, Hyun Myung  

**Link**: [PDF](https://arxiv.org/pdf/2511.09142)  

**Abstract**: LiDAR-inertial odometry (LIO) has been widely used in robotics due to its high accuracy. However, its performance degrades in degenerate environments, such as long corridors and high-altitude flights, where LiDAR measurements are imbalanced or sparse, leading to ill-posed state estimation. In this letter, we present LODESTAR, a novel LIO method that addresses these degeneracies through two key modules: degeneracy-aware adaptive Schmidt-Kalman filter (DA-ASKF) and degeneracy-aware data exploitation (DA-DE). DA-ASKF employs a sliding window to utilize past states and measurements as additional constraints. Specifically, it introduces degeneracy-aware sliding modes that adaptively classify states as active or fixed based on their degeneracy level. Using Schmidt-Kalman update, it partially optimizes active states while preserving fixed states. These fixed states influence the update of active states via their covariances, serving as reference anchors--akin to a lodestar. Additionally, DA-DE prunes less-informative measurements from active states and selectively exploits measurements from fixed states, based on their localizability contribution and the condition number of the Jacobian matrix. Consequently, DA-ASKF enables degeneracy-aware constrained optimization and mitigates measurement sparsity, while DA-DE addresses measurement imbalance. Experimental results show that LODESTAR outperforms existing LiDAR-based odometry methods and degeneracy-aware modules in terms of accuracy and robustness under various degenerate conditions. 

**Abstract (ZH)**: LODESTAR：一种针对退化环境的新型LiDAR-惯性里程计方法 

---
# Decoupling Torque and Stiffness: A Unified Modeling and Control Framework for Antagonistic Artificial Muscles 

**Title (ZH)**: 解耦扭矩与刚度：对抗性人工肌肉的统一建模与控制框架 

**Authors**: Amirhossein Kazemipour, Robert K. Katzschmann  

**Link**: [PDF](https://arxiv.org/pdf/2511.09104)  

**Abstract**: Antagonistic soft actuators built from artificial muscles (PAMs, HASELs, DEAs) promise plant-level torque-stiffness decoupling, yet existing controllers for soft muscles struggle to maintain independent control through dynamic contact transients. We present a unified framework enabling independent torque and stiffness commands in real-time for diverse soft actuator types. Our unified force law captures diverse soft muscle physics in a single model with sub-ms computation, while our cascaded controller with analytical inverse dynamics maintains decoupling despite model errors and disturbances. Using co-contraction/bias coordinates, the controller independently modulates torque via bias and stiffness via co-contraction-replicating biological impedance strategies. Simulation-based validation through contact experiments demonstrates maintained independence: 200x faster settling on soft surfaces, 81% force reduction on rigid surfaces, and stable interaction vs 22-54% stability for fixed policies. This framework provides a foundation for enabling musculoskeletal antagonistic systems to execute adaptive impedance control for safe human-robot interaction. 

**Abstract (ZH)**: 基于人工肌肉（PAMs、HASELs、DEAs）的对抗性软执行机构有望实现植物级扭矩-刚度解耦，而现有软肌肉控制器难以在动态接触瞬态过程中保持独立控制。我们提出了一种统一框架，能够在实时环境中为不同类型的软执行机构提供独立的扭矩和刚度命令。我们的统一力律在一个模型中捕捉到各种软肌肉的物理特性，并实现了毫秒级的计算；而我们的级联控制器结合分析逆动力学方法，即使在模型误差和干扰存在的情况下也能保持解耦。通过使用共收缩/偏置坐标，控制器通过偏置独立调节扭矩，通过共收缩模仿生物阻抗策略独立调节刚度。基于接触实验的仿真实验验证了维持独立性：软表面快速收敛200倍，刚表面力减少81%，与固定策略相比稳定交互稳定性提高22-54%。该框架为实现适应性阻抗控制以实现安全的人机交互提供了基础。 

---
# A Quantum Tunneling and Bio-Phototactic Driven Enhanced Dwarf Mongoose Optimizer for UAV Trajectory Planning and Engineering Problem 

**Title (ZH)**: 一种基于量子隧道效应和光生物趋化性的迷你.za狐蚁优化器在无人机航迹规划及工程问题中的增强应用 

**Authors**: Mingyang Yu, Haorui Yang, Kangning An, Xinjian Wei, Xiaoxuan Xu, Jing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.09020)  

**Abstract**: With the widespread adoption of unmanned aerial vehicles (UAV), effective path planning has become increasingly important. Although traditional search methods have been extensively applied, metaheuristic algorithms have gained popularity due to their efficiency and problem-specific heuristics. However, challenges such as premature convergence and lack of solution diversity still hinder their performance in complex scenarios. To address these issues, this paper proposes an Enhanced Multi-Strategy Dwarf Mongoose Optimization (EDMO) algorithm, tailored for three-dimensional UAV trajectory planning in dynamic and obstacle-rich environments. EDMO integrates three novel strategies: (1) a Dynamic Quantum Tunneling Optimization Strategy (DQTOS) to enable particles to probabilistically escape local optima; (2) a Bio-phototactic Dynamic Focusing Search Strategy (BDFSS) inspired by microbial phototaxis for adaptive local refinement; and (3) an Orthogonal Lens Opposition-Based Learning (OLOBL) strategy to enhance global exploration through structured dimensional recombination. EDMO is benchmarked on 39 standard test functions from CEC2017 and CEC2020, outperforming 14 advanced algorithms in convergence speed, robustness, and optimization accuracy. Furthermore, real-world validations on UAV three-dimensional path planning and three engineering design tasks confirm its practical applicability and effectiveness in field robotics missions requiring intelligent, adaptive, and time-efficient planning. 

**Abstract (ZH)**: 基于多策略增强矮獴优化算法的三维无人机路径规划 

---
# A Shared Control Framework for Mobile Robots with Planning-Level Intention Prediction 

**Title (ZH)**: 基于计划层级意图预测的移动机器人共享控制框架 

**Authors**: Jinyu Zhang, Lijun Han, Feng Jian, Lingxi Zhang, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08912)  

**Abstract**: In mobile robot shared control, effectively understanding human motion intention is critical for seamless human-robot collaboration. This paper presents a novel shared control framework featuring planning-level intention prediction. A path replanning algorithm is designed to adjust the robot's desired trajectory according to inferred human intentions. To represent future motion intentions, we introduce the concept of an intention domain, which serves as a constraint for path replanning. The intention-domain prediction and path replanning problems are jointly formulated as a Markov Decision Process and solved through deep reinforcement learning. In addition, a Voronoi-based human trajectory generation algorithm is developed, allowing the model to be trained entirely in simulation without human participation or demonstration data. Extensive simulations and real-world user studies demonstrate that the proposed method significantly reduces operator workload and enhances safety, without compromising task efficiency compared with existing assistive teleoperation approaches. 

**Abstract (ZH)**: 移动机器人共享控制中，有效理解人类运动意图对于实现无缝的人机协作至关重要。本文提出了一种新的共享控制框架，该框架强调意图预测在计划级别上的应用。设计了一条路径重规划算法，根据推断的人类意图调整机器人的期望轨迹。为了表示未来的运动意图，引入了意图域的概念，作为路径重规划的约束。意图域预测和路径重规划问题被联合形式化为马尔可夫决策过程，并通过深度强化学习求解。此外，开发了一种基于Voronoi的人类轨迹生成算法，使得模型可以在完全离线仿真的情况下进行训练，不需要人类的参与或示例数据。广泛的仿真实验和现实世界用户研究表明，所提出的方法显著减少了操作员的工作负担，提高了安全性，且与现有的辅助遥操作方法相比并未牺牲任务效率。 

---
# MirrorLimb: Implementing hand pose acquisition and robot teleoperation based on RealMirror 

**Title (ZH)**: 镜像四肢：基于RealMirror实现手部姿态获取与机器人遥操作 

**Authors**: Cong Tai, Hansheng Wu, Haixu Long, Zhengbin Long, Zhaoyu Zheng, Haodong Xiang, Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2511.08865)  

**Abstract**: In this work, we present a PICO-based robot remote operating framework that enables low-cost, real-time acquisition of hand motion and pose data, outperforming mainstream visual tracking and motion capture solutions in terms of cost-effectiveness. The framework is natively compatible with the RealMirror ecosystem, offering ready-to-use functionality for stable and precise robotic trajectory recording within the Isaac simulation environment, thereby facilitating the construction of Vision-Language-Action (VLA) datasets. Additionally, the system supports real-time teleoperation of a variety of end-effector-equipped robots, including dexterous hands and robotic grippers. This work aims to lower the technical barriers in the study of upper-limb robotic manipulation, thereby accelerating advancements in VLA-related research. 

**Abstract (ZH)**: 基于PICO的低成本实时手部运动与姿态数据获取机器人远程操作框架：实现稳定的精确机器人 trajectories 记录以构建视觉-语言-动作数据集，并支持各种末端执行器机器人实时远程操作，从而降低上肢机器人操作研究的技术门槛，促进相关研究进展。 

---
# XPRESS: X-Band Radar Place Recognition via Elliptical Scan Shaping 

**Title (ZH)**: XPRESS：基于椭圆扫描成型的X波段雷达Place Recognition 

**Authors**: Hyesu Jang, Wooseong Yang, Ayoung Kim, Dongje Lee, Hanguen Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.08863)  

**Abstract**: X-band radar serves as the primary sensor on maritime vessels, however, its application in autonomous navigation has been limited due to low sensor resolution and insufficient information content. To enable X-band radar-only autonomous navigation in maritime environments, this paper proposes a place recognition algorithm specifically tailored for X-band radar, incorporating an object density-based rule for efficient candidate selection and intentional degradation of radar detections to achieve robust retrieval performance. The proposed algorithm was evaluated on both public maritime radar datasets and our own collected dataset, and its performance was compared against state-of-the-art radar place recognition methods. An ablation study was conducted to assess the algorithm's performance sensitivity with respect to key parameters. 

**Abstract (ZH)**: X波段雷达在海事船舶中的应用主要依赖其作为主要传感器，但由于传感器分辨率低和信息量不足，其在自主导航中的应用受到限制。为使X波段雷达能够在海事环境中实现自主导航，本文提出了一种专门针对X波段雷达的场所识别算法，该算法结合了基于物体密度的选择规则，并故意降级雷达检测以实现稳健的检索性能。该算法在公共海事雷达数据集和我们收集的数据集上进行了评估，并将其性能与最先进的雷达场所识别方法进行了比较。进行了消融研究以评估算法对关键参数的性能敏感性。 

---
# Low-cost Multi-agent Fleet for Acoustic Cooperative Localization Research 

**Title (ZH)**: 低成本多agent舰队在声学协同定位研究 

**Authors**: Nelson Durrant, Braden Meyers, Matthew McMurray, Clayton Smith, Brighton Anderson, Tristan Hodgins, Kalliyan Velasco, Joshua G. Mangelson  

**Link**: [PDF](https://arxiv.org/pdf/2511.08822)  

**Abstract**: Real-world underwater testing for multi-agent autonomy presents substantial financial and engineering challenges. In this work, we introduce the Configurable Underwater Group of Autonomous Robots (CoUGARs) as a low-cost, configurable autonomous-underwater-vehicle (AUV) platform for multi-agent autonomy research. The base design costs less than $3,000 USD (as of May 2025) and is based on commercially-available and 3D-printed parts, enabling quick customization for various sensor payloads and configurations. Our current expanded model is equipped with a doppler velocity log (DVL) and ultra-short-baseline (USBL) acoustic array/transducer to support research on acoustic-based cooperative localization. State estimation, navigation, and acoustic communications software has been developed and deployed using a containerized software stack and is tightly integrated with the HoloOcean simulator. The system was tested both in simulation and via in-situ field trials in Utah lakes and reservoirs. 

**Abstract (ZH)**: 面向多自主-agent自主性的实地水下测试面临着重大的财务和工程挑战。在此项工作中，我们介绍了可配置水下自主机器人组（CoUGARs）作为一种低成本且可配置的自主 underwater 机器人（AUV）平台，用于多自主-agent自主性研究。基础设计成本低于3000美元（截至2025年5月），基于商用和3D打印部件，能够快速定制各种传感器载荷和配置。我们当前扩展的模型配备了多普勒速度 log（DVL）和超短基线（USBL）声学阵列/换能器，以支持基于声波的协同定位研究。已经开发并部署了状态估计、导航和声学通信软件，并使用容器化软件栈进行集成，并与HoloOcean模拟器紧密集成。该系统在模拟中进行了测试，并在犹他州的湖泊和水库中进行了现场试验。 

---
# Dual-Arm Whole-Body Motion Planning: Leveraging Overlapping Kinematic Chains 

**Title (ZH)**: 双臂全身运动规划：利用重叠运动学链 

**Authors**: Richard Cheng, Peter Werner, Carolyn Matl  

**Link**: [PDF](https://arxiv.org/pdf/2511.08778)  

**Abstract**: High degree-of-freedom dual-arm robots are becoming increasingly common due to their morphology enabling them to operate effectively in human environments. However, motion planning in real-time within unknown, changing environments remains a challenge for such robots due to the high dimensionality of the configuration space and the complex collision-avoidance constraints that must be obeyed. In this work, we propose a novel way to alleviate the curse of dimensionality by leveraging the structure imposed by shared joints (e.g. torso joints) in a dual-arm robot. First, we build two dynamic roadmaps (DRM) for each kinematic chain (i.e. left arm + torso, right arm + torso) with specific structure induced by the shared joints. Then, we show that we can leverage this structure to efficiently search through the composition of the two roadmaps and largely sidestep the curse of dimensionality. Finally, we run several experiments in a real-world grocery store with this motion planner on a 19 DoF mobile manipulation robot executing a grocery fulfillment task, achieving 0.4s average planning times with 99.9% success rate across more than 2000 motion plans. 

**Abstract (ZH)**: 具有高自由度的双臂机器人由于其形态能够在人类环境中有效操作而越来越多地被使用。然而，由于配置空间的高维性和必须遵守的复杂碰撞避免约束，在未知且不断变化的环境中进行实时运动规划仍然是一个挑战。在本工作中，我们提出了一种新颖的方法，通过利用双臂机器人中共享关节（例如躯干关节）所施加的结构来缓解维数灾。首先，我们为每个运动链（即左臂+躯干，右臂+躯干）构建两个动态 roadmap (DRM)，并引入由共享关节引起的特定结构。然后，我们表明可以利用这种结构高效地搜索两个 roadmap 的组合，并大大避免维数灾。最后，我们在这个具有19个自由度的移动操作机器人上对该运动规划器进行了多次实验，在现实世界的超市中执行杂货补货任务，实现了超过2000个运动计划的平均规划时间为0.4秒，成功率高达99.9%。 

---
# CENIC: Convex Error-controlled Numerical Integration for Contact 

**Title (ZH)**: CENIC: 凸误差控制数值积分方法用于接触模拟 

**Authors**: Vince Kurtz, Alejandro Castro  

**Link**: [PDF](https://arxiv.org/pdf/2511.08771)  

**Abstract**: State-of-the-art robotics simulators operate in discrete time. This requires users to choose a time step, which is both critical and challenging: large steps can produce non-physical artifacts, while small steps force the simulation to run slowly. Continuous-time error-controlled integration avoids such issues by automatically adjusting the time step to achieve a desired accuracy. But existing error-controlled integrators struggle with the stiff dynamics of contact, and cannot meet the speed and scalability requirements of modern robotics workflows. We introduce CENIC, a new continuous-time integrator that brings together recent advances in convex time-stepping and error-controlled integration, inheriting benefits from both continuous integration and discrete time-stepping. CENIC runs at fast real-time rates comparable to discrete-time robotics simulators like MuJoCo, Drake and Isaac Sim, while also providing guarantees on accuracy and convergence. 

**Abstract (ZH)**: 一种新的连续时间误差控制积分器：CENIC 

---
