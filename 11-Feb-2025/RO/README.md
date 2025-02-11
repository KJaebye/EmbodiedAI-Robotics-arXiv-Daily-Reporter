# Infinite-Horizon Value Function Approximation for Model Predictive Control 

**Title (ZH)**: 无限 horizons 价值函数逼近用于模型预测控制 

**Authors**: Armand Jordana, Sébastien Kleff, Arthur Haffemayer, Joaquim Ortiz-Haro, Justin Carpentier, Nicolas Mansard, Ludovic Righetti  

**Link**: [PDF](https://arxiv.org/pdf/2502.06760)  

**Abstract**: Model Predictive Control has emerged as a popular tool for robots to generate complex motions. However, the real-time requirement has limited the use of hard constraints and large preview horizons, which are necessary to ensure safety and stability. In practice, practitioners have to carefully design cost functions that can imitate an infinite horizon formulation, which is tedious and often results in local minima. In this work, we study how to approximate the infinite horizon value function of constrained optimal control problems with neural networks using value iteration and trajectory optimization. Furthermore, we demonstrate how using this value function approximation as a terminal cost provides global stability to the model predictive controller. The approach is validated on two toy problems and a real-world scenario with online obstacle avoidance on an industrial manipulator where the value function is conditioned to the goal and obstacle. 

**Abstract (ZH)**: 我们研究了如何使用神经网络通过值迭代和轨迹优化来逼近约束最优控制问题的无界 horizons 价值函数，并进一步展示了将此价值函数逼近作为终端代价如何为模型预测控制器提供全局稳定性。该方法在两个玩具问题和一个具有在线障碍避免的实际工业 manipulator 场景中得到了验证，其中价值函数根据目标和障碍进行条件化。 

---
# AgilePilot: DRL-Based Drone Agent for Real-Time Motion Planning in Dynamic Environments by Leveraging Object Detection 

**Title (ZH)**: AgilePilot: 基于DRL的无人机代理在动态环境中进行实时运动规划的方法，借助物体检测 

**Authors**: Roohan Ahmed Khan, Valerii Serpiva, Demetros Aschalew, Aleksey Fedoseev, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2502.06725)  

**Abstract**: Autonomous drone navigation in dynamic environments remains a critical challenge, especially when dealing with unpredictable scenarios including fast-moving objects with rapidly changing goal positions. While traditional planners and classical optimisation methods have been extensively used to address this dynamic problem, they often face real-time, unpredictable changes that ultimately leads to sub-optimal performance in terms of adaptiveness and real-time decision making. In this work, we propose a novel motion planner, AgilePilot, based on Deep Reinforcement Learning (DRL) that is trained in dynamic conditions, coupled with real-time Computer Vision (CV) for object detections during flight. The training-to-deployment framework bridges the Sim2Real gap, leveraging sophisticated reward structures that promotes both safety and agility depending upon environment conditions. The system can rapidly adapt to changing environments, while achieving a maximum speed of 3.0 m/s in real-world scenarios. In comparison, our approach outperforms classical algorithms such as Artificial Potential Field (APF) based motion planner by 3 times, both in performance and tracking accuracy of dynamic targets by using velocity predictions while exhibiting 90% success rate in 75 conducted experiments. This work highlights the effectiveness of DRL in tackling real-time dynamic navigation challenges, offering intelligent safety and agility. 

**Abstract (ZH)**: 基于深度强化学习的自主无人机在动态环境下的敏捷导航方法 

---
# HetSwarm: Cooperative Navigation of Heterogeneous Swarm in Dynamic and Dense Environments through Impedance-based Guidance 

**Title (ZH)**: HetSwarm：基于阻抗引导的异构 swarm 在动态密集环境中的协同导航 

**Authors**: Malaika Zafar, Roohan Ahmed Khan, Aleksey Fedoseev, Kumar Katyayan Jaiswal, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2502.06722)  

**Abstract**: With the growing demand for efficient logistics and warehouse management, unmanned aerial vehicles (UAVs) are emerging as a valuable complement to automated guided vehicles (AGVs). UAVs enhance efficiency by navigating dense environments and operating at varying altitudes. However, their limited flight time, battery life, and payload capacity necessitate a supporting ground station. To address these challenges, we propose HetSwarm, a heterogeneous multi-robot system that combines a UAV and a mobile ground robot for collaborative navigation in cluttered and dynamic conditions. Our approach employs an artificial potential field (APF)-based path planner for the UAV, allowing it to dynamically adjust its trajectory in real time. The ground robot follows this path while maintaining connectivity through impedance links, ensuring stable coordination. Additionally, the ground robot establishes temporal impedance links with low-height ground obstacles to avoid local collisions, as these obstacles do not interfere with the UAV's flight. Experimental validation of HetSwarm in diverse environmental conditions demonstrated a 90% success rate across 30 test cases. The ground robot exhibited an average deviation of 45 cm near obstacles, confirming effective collision avoidance. Extensive simulations in the Gym PyBullet environment further validated the robustness of our system for real-world applications, demonstrating its potential for dynamic, real-time task execution in cluttered environments. 

**Abstract (ZH)**: 基于异构多机器人系统的UAV与移动地面机器人协同导航研究 

---
# Discovery of skill switching criteria for learning agile quadruped locomotion 

**Title (ZH)**: 发现学习敏捷四足运动中的技能切换准则 

**Authors**: Wanming Yu, Fernando Acero, Vassil Atanassov, Chuanyu Yang, Ioannis Havoutis, Dimitrios Kanoulas, Zhibin Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.06676)  

**Abstract**: This paper develops a hierarchical learning and optimization framework that can learn and achieve well-coordinated multi-skill locomotion. The learned multi-skill policy can switch between skills automatically and naturally in tracking arbitrarily positioned goals and recover from failures promptly. The proposed framework is composed of a deep reinforcement learning process and an optimization process. First, the contact pattern is incorporated into the reward terms for learning different types of gaits as separate policies without the need for any other references. Then, a higher level policy is learned to generate weights for individual policies to compose multi-skill locomotion in a goal-tracking task setting. Skills are automatically and naturally switched according to the distance to the goal. The proper distances for skill switching are incorporated in reward calculation for learning the high level policy and updated by an outer optimization loop as learning progresses. We first demonstrated successful multi-skill locomotion in comprehensive tasks on a simulated Unitree A1 quadruped robot. We also deployed the learned policy in the real world showcasing trotting, bounding, galloping, and their natural transitions as the goal position changes. Moreover, the learned policy can react to unexpected failures at any time, perform prompt recovery, and resume locomotion successfully. Compared to discrete switch between single skills which failed to transition to galloping in the real world, our proposed approach achieves all the learned agile skills, with smoother and more continuous skill transitions. 

**Abstract (ZH)**: 本文开发了一种分层学习与优化框架，能够学习和实现协调的多技能移动。学习到的多技能策略可以在追踪任意位置的目标时自动且自然地切换技能，并能迅速从失败中恢复。该提出的框架由深度 reinforcement 学习过程和优化过程组成。首先，通过将接触模式纳入奖励项中，无需任何其他参考，即可分别学习不同类型的步伐作为单独策略。然后，学习一个高层策略来生成各单独策略的权重，以在目标追踪任务设置中组合多技能移动。根据与目标的距离自动且自然地切换技能。适当的技能切换距离被纳入奖励计算中，以学习高层策略，并随着学习进程由外部优化循环更新。我们首先在模拟的 Unite A1 四足机器人上展示了多技能移动的全面任务。我们还在现实世界中部署了学习到的策略，展示了随着目标位置变化而进行的典型的摆动、跳跃、驰骋及其自然过渡。此外，学习到的策略可以对任何意外故障迅速作出反应，执行及时的恢复，并成功继续移动。与仅在现实世界中进行离散切换的单一技能，无法过渡到驰骋相比，我们的方法实现了所有学习到的敏捷技能，并且技能过渡更加平滑和连续。 

---
# Predictive Red Teaming: Breaking Policies Without Breaking Robots 

**Title (ZH)**: 预测性红队演练：在不破坏机器人的情况下突破政策 

**Authors**: Anirudha Majumdar, Mohit Sharma, Dmitry Kalashnikov, Sumeet Singh, Pierre Sermanet, Vikas Sindhwani  

**Link**: [PDF](https://arxiv.org/pdf/2502.06575)  

**Abstract**: Visuomotor policies trained via imitation learning are capable of performing challenging manipulation tasks, but are often extremely brittle to lighting, visual distractors, and object locations. These vulnerabilities can depend unpredictably on the specifics of training, and are challenging to expose without time-consuming and expensive hardware evaluations. We propose the problem of predictive red teaming: discovering vulnerabilities of a policy with respect to environmental factors, and predicting the corresponding performance degradation without hardware evaluations in off-nominal scenarios. In order to achieve this, we develop RoboART: an automated red teaming (ART) pipeline that (1) modifies nominal observations using generative image editing to vary different environmental factors, and (2) predicts performance under each variation using a policy-specific anomaly detector executed on edited observations. Experiments across 500+ hardware trials in twelve off-nominal conditions for visuomotor diffusion policies demonstrate that RoboART predicts performance degradation with high accuracy (less than 0.19 average difference between predicted and real success rates). We also demonstrate how predictive red teaming enables targeted data collection: fine-tuning with data collected under conditions predicted to be adverse boosts baseline performance by 2-7x. 

**Abstract (ZH)**: 视觉运动政策通过模仿学习训练后能够执行复杂的操作任务，但常对光照、视觉干扰和物体位置极其脆弱。这些脆弱性可能依赖于训练的具体情况而不可预测，且在无需硬件评估的情况下难以暴露。我们提出了预测性红队攻击的问题：发现政策在环境因素方面的脆弱性，并在非标准情况下预测相应的性能降级。为了实现这一目标，我们开发了RoboART：一个自动化的红队攻击（ART）流水线，该流水线通过使用生成图像编辑来修改名义观察，以变化不同的环境因素；并通过在编辑过的观察上执行特定于策略的异常检测器来预测每种变化下的性能。在十二种非标准条件下超过500次硬件试验中，针对视觉运动扩散政策的实验结果表明，RoboART能够以高精度（预测成功率与实际成功率的平均差异小于0.19）预测性能降级。我们还展示了预测性红队攻击如何使数据收集更加有针对性：在预测为不利条件下的数据收集进行微调可以将基线性能提高2-7倍。 

---
# SIREN: Semantic, Initialization-Free Registration of Multi-Robot Gaussian Splatting Maps 

**Title (ZH)**: SIREN: 具有语义初始化自由的多机器人高斯斑点图注册 

**Authors**: Ola Shorinwa, Jiankai Sun, Mac Schwager, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2502.06519)  

**Abstract**: We present SIREN for registration of multi-robot Gaussian Splatting (GSplat) maps, with zero access to camera poses, images, and inter-map transforms for initialization or fusion of local submaps. To realize these capabilities, SIREN harnesses the versatility and robustness of semantics in three critical ways to derive a rigorous registration pipeline for multi-robot GSplat maps. First, SIREN utilizes semantics to identify feature-rich regions of the local maps where the registration problem is better posed, eliminating the need for any initialization which is generally required in prior work. Second, SIREN identifies candidate correspondences between Gaussians in the local maps using robust semantic features, constituting the foundation for robust geometric optimization, coarsely aligning 3D Gaussian primitives extracted from the local maps. Third, this key step enables subsequent photometric refinement of the transformation between the submaps, where SIREN leverages novel-view synthesis in GSplat maps along with a semantics-based image filter to compute a high-accuracy non-rigid transformation for the generation of a high-fidelity fused map. We demonstrate the superior performance of SIREN compared to competing baselines across a range of real-world datasets, and in particular, across the most widely-used robot hardware platforms, including a manipulator, drone, and quadruped. In our experiments, SIREN achieves about 90x smaller rotation errors, 300x smaller translation errors, and 44x smaller scale errors in the most challenging scenes, where competing methods struggle. We will release the code and provide a link to the project page after the review process. 

**Abstract (ZH)**: 宋iren：用于多机器人Gauss斑点图注册的语义驱动方法，无需相机姿态、图像或地图间变换的初始化或局部子图融合 

---
# Inflatable Kirigami Crawlers 

**Title (ZH)**: kirigami 扩展爬行器 

**Authors**: Burcu Seyidoğlu, Aida Parvaresh, Bahman Taherkhani, Ahmad Rafsanjani  

**Link**: [PDF](https://arxiv.org/pdf/2502.06466)  

**Abstract**: Kirigami offers unique opportunities for guided morphing by leveraging the geometry of the cuts. This work presents inflatable kirigami crawlers created by introducing cut patterns into heat-sealable textiles to achieve locomotion upon cyclic pneumatic actuation. Inflating traditional air pouches results in symmetric bulging and contraction. In inflated kirigami actuators, the accumulated compressive forces uniformly break the symmetry, enhance contraction compared to simple air pouches by two folds, and trigger local rotation of the sealed edges that overlap and self-assemble into an architected surface with emerging scale-like features. As a result, the inflatable kirigami actuators exhibit a uniform, controlled contraction with asymmetric localized out-of-plane deformations. This process allows us to harness the geometric and material nonlinearities to imbue inflatable textile-based kirigami actuators with predictable locomotive functionalities. We thoroughly characterized the programmed deformations of these actuators and their impact on friction. We found that the kirigami actuators exhibit directional anisotropic friction properties when inflated, having higher friction coefficients against the direction of the movement, enabling them to move across surfaces with varying roughness. We further enhanced the functionality of inflatable kirigami actuators by introducing multiple channels and segments to create functional soft robotic prototypes with versatile locomotion capabilities. 

**Abstract (ZH)**: 剪纸几何结构驱动的可充气剪纸爬行器及其定向黏着性能与功能性软机器人原型研究 

---
# SIGMA: Sheaf-Informed Geometric Multi-Agent Pathfinding 

**Title (ZH)**: SIGMA：Sheaf-Informed几何多agent路径规划 

**Authors**: Shuhao Liao, Weihang Xia, Yuhong Cao, Weiheng Dai, Chengyang He, Wenjun Wu, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2502.06440)  

**Abstract**: The Multi-Agent Path Finding (MAPF) problem aims to determine the shortest and collision-free paths for multiple agents in a known, potentially obstacle-ridden environment. It is the core challenge for robotic deployments in large-scale logistics and transportation. Decentralized learning-based approaches have shown great potential for addressing the MAPF problems, offering more reactive and scalable solutions. However, existing learning-based MAPF methods usually rely on agents making decisions based on a limited field of view (FOV), resulting in short-sighted policies and inefficient cooperation in complex scenarios. There, a critical challenge is to achieve consensus on potential movements between agents based on limited observations and communications. To tackle this challenge, we introduce a new framework that applies sheaf theory to decentralized deep reinforcement learning, enabling agents to learn geometric cross-dependencies between each other through local consensus and utilize them for tightly cooperative decision-making. In particular, sheaf theory provides a mathematical proof of conditions for achieving global consensus through local observation. Inspired by this, we incorporate a neural network to approximately model the consensus in latent space based on sheaf theory and train it through self-supervised learning. During the task, in addition to normal features for MAPF as in previous works, each agent distributedly reasons about a learned consensus feature, leading to efficient cooperation on pathfinding and collision avoidance. As a result, our proposed method demonstrates significant improvements over state-of-the-art learning-based MAPF planners, especially in relatively large and complex scenarios, demonstrating its superiority over baselines in various simulations and real-world robot experiments. 

**Abstract (ZH)**: 多智能体路径规划问题旨在确定在已知且可能存在障碍物环境中的多个智能体的最短且无碰撞路径。它是大规模物流和交通机器人部署的核心挑战。基于去中心化的学习方法显示出解决多智能体路径规划问题的巨大潜力，提供了更具反应性和可扩展性的解决方案。然而，现有的基于学习的多智能体路径规划方法通常依赖于智能体基于有限视野（FOV）进行决策，导致在复杂场景中出现短视的政策和低效的合作。在这种情况下，一个关键挑战是基于有限的观察和通信实现智能体间潜在动作的一致性。为了应对这一挑战，我们引入了一个新的框架，该框架将层流理论应用于去中心化的深度强化学习，使智能体能够通过局部共识学习几何交叉依赖关系，并利用这些关系进行紧密的合作决策。特别是，层流理论提供了通过局部观察实现全局一致性所需的数学证明。受此启发，我们引入一种神经网络，在层流理论的基础上近似模型潜在空间中的共识，并通过自我监督学习训练该网络。在执行任务期间，除了解析传统的MAPF特征外，每个智能体还分布式地推理一种学习到的共识特征，从而在路径规划和碰撞避免方面实现高效的协作。因此，我们提出的方法在相对较大型和复杂场景中展示了显著优于现有最佳学习方法的效果，各种仿真和实际机器人实验表明其超越基准方法的优越性。 

---
# Occ-LLM: Enhancing Autonomous Driving with Occupancy-Based Large Language Models 

**Title (ZH)**: 基于占用率的大语言模型增强自动驾驶：Occ-LLM 

**Authors**: Tianshuo Xu, Hao Lu, Xu Yan, Yingjie Cai, Bingbing Liu, Yingcong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.06419)  

**Abstract**: Large Language Models (LLMs) have made substantial advancements in the field of robotic and autonomous driving. This study presents the first Occupancy-based Large Language Model (Occ-LLM), which represents a pioneering effort to integrate LLMs with an important representation. To effectively encode occupancy as input for the LLM and address the category imbalances associated with occupancy, we propose Motion Separation Variational Autoencoder (MS-VAE). This innovative approach utilizes prior knowledge to distinguish dynamic objects from static scenes before inputting them into a tailored Variational Autoencoder (VAE). This separation enhances the model's capacity to concentrate on dynamic trajectories while effectively reconstructing static scenes. The efficacy of Occ-LLM has been validated across key tasks, including 4D occupancy forecasting, self-ego planning, and occupancy-based scene question answering. Comprehensive evaluations demonstrate that Occ-LLM significantly surpasses existing state-of-the-art methodologies, achieving gains of about 6\% in Intersection over Union (IoU) and 4\% in mean Intersection over Union (mIoU) for the task of 4D occupancy forecasting. These findings highlight the transformative potential of Occ-LLM in reshaping current paradigms within robotic and autonomous driving. 

**Abstract (ZH)**: 大型语言模型（LLMs）在机器人和自主驾驶领域取得了重大进展。本研究提出了首个基于 occupancy 的大型语言模型（Occ-LLM），这是将 LLMs 与重要表示形式集成的开创性努力。为了有效将 occupancy 作为输入编码到 LLM 中并解决与 occupancy 相关的类别不平衡问题，我们提出了运动分离变分自编码器（MS-VAE）。这种创新方法利用先验知识，在输入到定制的变分自编码器（VAE）之前，将动态对象与静态场景区分开来。这种分离增强了模型专注于动态轨迹并有效重建静态场景的能力。Occ-LLM 的有效性已经在 4D 占有率预测、自我自我规划以及基于占有的场景问答等关键任务中得到验证。综合评估表明，Occ-LLM 显著优于现有最先进的方法，在 4D 占有率预测任务中，IoU 和 mIoU 分别提高了约 6% 和 4%。这些发现强调了 Occ-LLM 在重塑机器人和自主驾驶领域现有范式方面的变革潜力。 

---
# Proprioceptive Origami Manipulator 

**Title (ZH)**: proprioceptive 赋范的
Origami 纸艺的
Manipulator 控制器

赋范纸艺控制器 

**Authors**: Aida Parvaresh, Arman Goshtasbi, Jonathan Andres Tirado Rosero, Ahmad Rafsanjani  

**Link**: [PDF](https://arxiv.org/pdf/2502.06362)  

**Abstract**: Origami offers a versatile framework for designing morphable structures and soft robots by exploiting the geometry of folds. Tubular origami structures can act as continuum manipulators that balance flexibility and strength. However, precise control of such manipulators often requires reliance on vision-based systems that limit their application in complex and cluttered environments. Here, we propose a proprioceptive tendon-driven origami manipulator without compromising its flexibility. Using conductive threads as actuating tendons, we multiplex them with proprioceptive sensing capabilities. The change in the active length of the tendons is reflected in their effective resistance, which can be measured with a simple circuit. We correlated the change in the resistance to the lengths of the tendons. We input this information into a forward kinematic model to reconstruct the manipulator configuration and end-effector position. This platform provides a foundation for the closed-loop control of continuum origami manipulators while preserving their inherent flexibility. 

**Abstract (ZH)**: Origami offers a versatile framework for designing morphable structures and soft robots by exploiting the geometry of folds. Tubular origami structures can act as continuum manipulators that balance flexibility and strength. However, precise control of such manipulators often requires reliance on vision-based systems that limit their application in complex and cluttered environments. Here, we propose a proprioceptive tendon-driven origami manipulator without compromising its flexibility. Using conductive threads as actuating tendons, we multiplex them with proprioceptive sensing capabilities. The change in the active length of the tendons is reflected in their effective resistance, which can be measured with a simple circuit. We correlated the change in the resistance to the lengths of the tendons. We input this information into a forward kinematic model to reconstruct the manipulator configuration and end-effector position. This platform provides a foundation for the closed-loop control of continuum origami manipulators while preserving their inherent flexibility. 

 proprioceptive腱驱动折纸 manipulator及其闭环控制的柔性框架 

---
# Weld n'Cut: Automated fabrication of inflatable fabric actuators 

**Title (ZH)**: 焊切：自动制造充气织物执行器 

**Authors**: Arman Goshtasbi, Burcu Seyidoğlu, Saravana Prashanth Murali Babu, Aida Parvaresh, Cao Danh Do, Ahmad Rafsanjani  

**Link**: [PDF](https://arxiv.org/pdf/2502.06361)  

**Abstract**: Lightweight, durable textile-based inflatable soft actuators are widely used in soft robotics, particularly for wearable robots in rehabilitation and in enhancing human performance in demanding jobs. Fabricating these actuators typically involves multiple steps: heat-sealable fabrics are fused with a heat press, and non-stick masking layers define internal chambers. These layers must be carefully removed post-fabrication, often making the process labor-intensive and prone to errors. To address these challenges and improve the accuracy and performance of inflatable actuators, we introduce the Weld n'Cut platform-an open-source, automated manufacturing process that combines ultrasonic welding for fusing textile layers with an oscillating knife for precise cuts, enabling the creation of complex inflatable structures. We demonstrate the machine's performance across various materials and designs with arbitrarily complex geometries. 

**Abstract (ZH)**: 轻质耐用纺织基可充气软执行器广泛应用于软机器人技术，特别是在康复穿戴机器人和提高严苛工作中的人类性能方面。我们介绍了Weld n'Cut平台——一种开源的自动化制造过程，该过程结合了超声焊接以融合纺织层和振刀以进行精确切割，从而能够创建复杂的可充气结构。我们展示了该机器在各种材料和设计上的性能，这些设计具有任意复杂的几何形状。 

---
# Occlusion-Aware Contingency Safety-Critical Planning for Autonomous Vehicles 

**Title (ZH)**: 面向遮挡感知的应急安全关键规划方法研究 

**Authors**: Lei Zheng, Rui Yang, Minzhe Zheng, Zengqi Peng, Michael Yu Wang, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.06359)  

**Abstract**: Ensuring safe driving while maintaining travel efficiency for autonomous vehicles in dynamic and occluded environments is a critical challenge. This paper proposes an occlusion-aware contingency safety-critical planning approach for real-time autonomous driving in such environments. Leveraging reachability analysis for risk assessment, forward reachable sets of occluded phantom vehicles are computed to quantify dynamic velocity boundaries. These velocity boundaries are incorporated into a biconvex nonlinear programming (NLP) formulation, enabling simultaneous optimization of exploration and fallback trajectories within a receding horizon planning framework. To facilitate real-time optimization and ensure coordination between trajectories, we employ the consensus alternating direction method of multipliers (ADMM) to decompose the biconvex NLP problem into low-dimensional convex subproblems. The effectiveness of the proposed approach is validated through simulation studies and real-world experiments in occluded intersections. Experimental results demonstrate enhanced safety and improved travel efficiency, enabling real-time safe trajectory generation in dynamic occluded intersections under varying obstacle conditions. A video showcasing the experimental results is available at this https URL. 

**Abstract (ZH)**: 确保在动态和遮挡环境中的自动驾驶车辆安全驾驶并保持行驶效率是一个关键挑战。本文提出了一种基于遮挡感知的应急安全关键规划方法，用于此类环境下的实时自动驾驶。利用可达性分析进行风险评估，计算遮挡虚拟车辆的前向可达集以量化动态速度边界。将这些速度边界整合到一种双凸非线性规划（NLP）公式中，可在滑行期预测框架中同时优化探索和应急路径。为实现实时优化并确保路径之间的协调，我们采用共识交替方向乘子法（ADMM）将双凸NLP问题分解为低维凸子问题。通过仿真研究和遮挡交叉口的真实世界实验验证了所提出方法的有效性。实验结果表明，该方法能够提高安全性并改善行驶效率，在不同障碍物条件下实现动态遮挡交叉口的实时安全路径生成。实验结果视频可通过此链接查看：https://xxxxx。 

---
# Occupancy-SLAM: An Efficient and Robust Algorithm for Simultaneously Optimizing Robot Poses and Occupancy Map 

**Title (ZH)**: occupancy-SLAM：一种高效且 robust 的同时优化机器人姿态和占用地图算法 

**Authors**: Yingyu Wang, Liang Zhao, Shoudong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06292)  

**Abstract**: Joint optimization of poses and features has been extensively studied and demonstrated to yield more accurate results in feature-based SLAM problems. However, research on jointly optimizing poses and non-feature-based maps remains limited. Occupancy maps are widely used non-feature-based environment representations because they effectively classify spaces into obstacles, free areas, and unknown regions, providing robots with spatial information for various tasks. In this paper, we propose Occupancy-SLAM, a novel optimization-based SLAM method that enables the joint optimization of robot trajectory and the occupancy map through a parameterized map representation. The key novelty lies in optimizing both robot poses and occupancy values at different cell vertices simultaneously, a significant departure from existing methods where the robot poses need to be optimized first before the map can be estimated. Evaluations using simulations and practical 2D laser datasets demonstrate that the proposed approach can robustly obtain more accurate robot trajectories and occupancy maps than state-of-the-art techniques with comparable computational time. Preliminary results in the 3D case further confirm the potential of the proposed method in practical 3D applications, achieving more accurate results than existing methods. 

**Abstract (ZH)**: 基于 occupancy 地图的联合优化 SLAM 方法 

---
# CT-UIO: Continuous-Time UWB-Inertial-Odometer Localization Using Non-Uniform B-spline with Fewer Anchors 

**Title (ZH)**: CT-UIO: 基于非均匀B样条的连续时间UWB-惯性里程计定位方法（使用较少的锚点） 

**Authors**: Jian Sun, Wei Sun, Genwei Zhang, Kailun Yang, Song Li, Xiangqi Meng, Na Deng, Chongbin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2502.06287)  

**Abstract**: Ultra-wideband (UWB) based positioning with fewer anchors has attracted significant research interest in recent years, especially under energy-constrained conditions. However, most existing methods rely on discrete-time representations and smoothness priors to infer a robot's motion states, which often struggle with ensuring multi-sensor data synchronization. In this paper, we present an efficient UWB-Inertial-odometer localization system, utilizing a non-uniform B-spline framework with fewer anchors. Unlike traditional uniform B-spline-based continuous-time methods, we introduce an adaptive knot-span adjustment strategy for non-uniform continuous-time trajectory representation. This is accomplished by adjusting control points dynamically based on movement speed. To enable efficient fusion of IMU and odometer data, we propose an improved Extended Kalman Filter (EKF) with innovation-based adaptive estimation to provide short-term accurate motion prior. Furthermore, to address the challenge of achieving a fully observable UWB localization system under few-anchor conditions, the Virtual Anchor (VA) generation method based on multiple hypotheses is proposed. At the backend, we propose a CT-UIO factor graph with an adaptive sliding window for global trajectory estimation. Comprehensive experiments conducted on corridor and exhibition hall datasets validate the proposed system's high precision and robust performance. The codebase and datasets of this work will be open-sourced at this https URL. 

**Abstract (ZH)**: 基于较少锚点的非均匀B-样条框架的超宽带(UWB)惯性里程计定位系统 

---
# Interaction-aware Conformal Prediction for Crowd Navigation 

**Title (ZH)**: 基于交互的齐性预测 crowdsourcing 导航 

**Authors**: Zhe Huang, Tianchen Ji, Heling Zhang, Fatemeh Cheraghi Pouria, Katherine Driggs-Campbell, Roy Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.06221)  

**Abstract**: During crowd navigation, robot motion plan needs to consider human motion uncertainty, and the human motion uncertainty is dependent on the robot motion plan. We introduce Interaction-aware Conformal Prediction (ICP) to alternate uncertainty-aware robot motion planning and decision-dependent human motion uncertainty quantification. ICP is composed of a trajectory predictor to predict human trajectories, a model predictive controller to plan robot motion with confidence interval radii added for probabilistic safety, a human simulator to collect human trajectory calibration dataset conditioned on the planned robot motion, and a conformal prediction module to quantify trajectory prediction error on the decision-dependent calibration dataset. Crowd navigation simulation experiments show that ICP strikes a good balance of performance among navigation efficiency, social awareness, and uncertainty quantification compared to previous works. ICP generalizes well to navigation tasks under various crowd densities. The fast runtime and efficient memory usage make ICP practical for real-world applications. Code is available at this https URL. 

**Abstract (ZH)**: 基于交互感知同轨预测的不确定性-aware机器人导航与人类行为不确定性量化 

---
# Improved Extrinsic Calibration of Acoustic Cameras via Batch Optimization 

**Title (ZH)**: 基于批量优化的声摄像机外部标定改进方法 

**Authors**: Zhi Li, Jiang Wang, Xiaoyang Li, He Kong  

**Link**: [PDF](https://arxiv.org/pdf/2502.06196)  

**Abstract**: Acoustic cameras have found many applications in practice. Accurate and reliable extrinsic calibration of the microphone array and visual sensors within acoustic cameras is crucial for fusing visual and auditory measurements. Existing calibration methods either require prior knowledge of the microphone array geometry or rely on grid search which suffers from slow iteration speed or poor convergence. To overcome these limitations, in this paper, we propose an automatic calibration technique using a calibration board with both visual and acoustic markers to identify each microphone position in the camera frame. We formulate the extrinsic calibration problem (between microphones and the visual sensor) as a nonlinear least squares problem and employ a batch optimization strategy to solve the associated problem. Extensive numerical simulations and realworld experiments show that the proposed method improves both the accuracy and robustness of extrinsic parameter calibration for acoustic cameras, in comparison to existing methods. To benefit the community, we open-source all the codes and data at this https URL. 

**Abstract (ZH)**: 声学相机在实践中找到了许多应用。声学相机中麦克风阵列和视觉传感器的精确可靠的外部校准对于融合视觉和听觉测量至关重要。现有的校准方法要么需要麦克风阵列几何结构的先验知识，要么依赖于网格搜索，这会导致迭代速度慢或收敛效果差。为克服这些限制，本文提出了一种使用带有视觉和声学标记的校准板的自动校准技术，以在相机框架中识别每个麦克风的位置。我们将外部校准问题（麦克风与视觉传感器之间）表述为非线性最小二乘问题，并采用批量优化策略来解决相关问题。广泛的数值模拟和实地实验表明，所提出的方法在声学相机的外部参数校准的精确性和鲁棒性方面均优于现有方法。为了惠及社区，我们在此处开放了所有代码和数据。 

---
# Portable, High-Frequency, and High-Voltage Control Circuits for Untethered Miniature Robots Driven by Dielectric Elastomer Actuators 

**Title (ZH)**: 便携式、高频和高压控制电路用于由介电弹性体执行器驱动的无缆微型机器人 

**Authors**: Qi Shao, Xin-Jun Liu, Huichan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.06166)  

**Abstract**: In this work, we propose a high-voltage, high-frequency control circuit for the untethered applications of dielectric elastomer actuators (DEAs). The circuit board leverages low-voltage resistive components connected in series to control voltages of up to 1.8 kV within a compact size, suitable for frequencies ranging from 0 to 1 kHz. A single-channel control board weighs only 2.5 g. We tested the performance of the control circuit under different load conditions and power supplies. Based on this control circuit, along with a commercial miniature high-voltage power converter, we construct an untethered crawling robot driven by a cylindrical DEA. The 42-g untethered robots successfully obtained crawling locomotion on a bench and within a pipeline at a driving frequency of 15 Hz, while simultaneously transmitting real-time video data via an onboard camera and antenna. Our work provides a practical way to use low-voltage control electronics to achieve the untethered driving of DEAs, and therefore portable and wearable devices. 

**Abstract (ZH)**: 一种用于介电弹性体执行器无绳应用的高压高频控制电路 

---
# Reward-Based Collision-Free Algorithm for Trajectory Planning of Autonomous Robots 

**Title (ZH)**: 基于奖励的碰撞 avoidance 路径规划算法 

**Authors**: Jose D. Hoyos, Tianyu Zhou, Zehui Lu, Shaoshuai Mou  

**Link**: [PDF](https://arxiv.org/pdf/2502.06149)  

**Abstract**: This paper introduces a new mission planning algorithm for autonomous robots that enables the reward-based selection of an optimal waypoint sequence from a predefined set. The algorithm computes a feasible trajectory and corresponding control inputs for a robot to navigate between waypoints while avoiding obstacles, maximizing the total reward, and adhering to constraints on state, input and its derivatives, mission time window, and maximum distance. This also solves a generalized prize-collecting traveling salesman problem. The proposed algorithm employs a new genetic algorithm that evolves solution candidates toward the optimal solution based on a fitness function and crossover. During fitness evaluation, a penalty method enforces constraints, and the differential flatness property with clothoid curves efficiently penalizes infeasible trajectories. The Euler spiral method showed promising results for trajectory parameterization compared to minimum snap and jerk polynomials. Due to the discrete exploration space, crossover is performed using a dynamic time-warping-based method and extended convex combination with projection. A mutation step enhances exploration. Results demonstrate the algorithm's ability to find the optimal waypoint sequence, fulfill constraints, avoid infeasible waypoints, and prioritize high-reward ones. Simulations and experiments with a ground vehicle, quadrotor, and quadruped are presented, complemented by benchmarking and a time-complexity analysis. 

**Abstract (ZH)**: 本文介绍了一种自主机器人任务规划算法，该算法能够基于奖励从预定义的集合中选择最优航点序列。该算法计算出机器人在避免障碍、最大化总奖励并遵守状态、输入及其导数、任务时间窗口和最大距离约束的前提下从一个航点导航到另一个航点的可行轨迹和相应的控制输入。该算法还解决了广义的收集奖品旅行商问题。所提出的算法采用了一种新的遗传算法，该算法基于适应度函数和交叉操作，通过进化解决方案候选人来朝向最优解。在适应度评估过程中，采用惩罚方法强制执行约束，并利用逐次曲线的微分平坦性性质高效地惩罚不可行轨迹。欧拉螺旋方法在轨迹参数化方面的表现优于最小拍和抖动多项式。由于探索空间的离散性，交叉操作通过动态时间扭曲方法和扩展凸组合与投影方法实现。变异步骤增强了探索性。实验结果表明，该算法能够找到最优航点序列，满足约束条件，避免不可行航点，并优先选择高奖励航点。并进行了地面车辆、四旋翼无人机和四足机器人仿真和实验，同时进行了基准测试和时间复杂性分析。 

---
# Mixed Reality Outperforms Virtual Reality for Remote Error Resolution in Pick-and-Place Tasks 

**Title (ZH)**: 混合现实优于虚拟现实的远程拾取放置任务错误解决性能 

**Authors**: Advay Kumar, Stephanie Simangunsong, Pamela Carreno-Medrano, Akansel Cosgun  

**Link**: [PDF](https://arxiv.org/pdf/2502.06141)  

**Abstract**: This study evaluates the performance and usability of Mixed Reality (MR), Virtual Reality (VR), and camera stream interfaces for remote error resolution tasks, such as correcting warehouse packaging errors. Specifically, we consider a scenario where a robotic arm halts after detecting an error, requiring a remote operator to intervene and resolve it via pick-and-place actions. Twenty-one participants performed simulated pick-and-place tasks using each interface. A linear mixed model (LMM) analysis of task resolution time, usability scores (SUS), and mental workload scores (NASA-TLX) showed that the MR interface outperformed both VR and camera interfaces. MR enabled significantly faster task completion, was rated higher in usability, and was perceived to be less cognitively demanding. Notably, the MR interface, which projected a virtual robot onto a physical table, provided superior spatial understanding and physical reference cues. Post-study surveys further confirmed participants' preference for MR over other interfaces. 

**Abstract (ZH)**: 本研究评估了混合现实（MR）、虚拟现实（VR）和摄像头流接口在远程错误解决任务中的性能和易用性，例如纠正仓库包装错误。具体而言，我们考虑了机器人手臂在检测到错误后停止工作，需要远程操作员通过抓取和放置操作介入并解决问题的场景。二十一名参与者使用每种接口完成了模拟的抓取和放置任务。线性混合模型（LMM）分析任务解决时间、易用性评分（SUS）和心理负荷评分（NASA-TLX）显示，MR接口在性能和可用性方面均优于VR和摄像头接口。MR使任务完成显著加快，易用性评分较高，并被认为认知负担较小。值得注意的是，能够将虚拟机器人投射到物理桌面上的MR接口提供了更好的空间理解和物理参考提示。研究后的调查进一步证实了参与者更偏好MR接口。 

---
# Real-Time LiDAR Point Cloud Compression and Transmission for Resource-constrained Robots 

**Title (ZH)**: 基于资源受限机器人实时LiDAR点云压缩与传输 

**Authors**: Yuhao Cao, Yu Wang, Haoyao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.06123)  

**Abstract**: LiDARs are widely used in autonomous robots due to their ability to provide accurate environment structural information. However, the large size of point clouds poses challenges in terms of data storage and transmission. In this paper, we propose a novel point cloud compression and transmission framework for resource-constrained robotic applications, called RCPCC. We iteratively fit the surface of point clouds with a similar range value and eliminate redundancy through their spatial relationships. Then, we use Shape-adaptive DCT (SA-DCT) to transform the unfit points and reduce the data volume by quantizing the transformed coefficients. We design an adaptive bitrate control strategy based on QoE as the optimization goal to control the quality of the transmitted point cloud. Experiments show that our framework achieves compression rates of 40$\times$ to 80$\times$ while maintaining high accuracy for downstream applications. our method significantly outperforms other baselines in terms of accuracy when the compression rate exceeds 70$\times$. Furthermore, in situations of reduced communication bandwidth, our adaptive bitrate control strategy demonstrates significant QoE improvements. The code will be available at this https URL. 

**Abstract (ZH)**: 基于资源约束的机器人应用的新型点云压缩与传输框架RCPCC 

---
# Towards Bio-inspired Heuristically Accelerated Reinforcement Learning for Adaptive Underwater Multi-Agents Behaviour 

**Title (ZH)**: 面向生物启发的启发式加速 reinforcement learning 在适应性水下多智能体行为中的研究 

**Authors**: Antoine Vivien, Thomas Chaffre, Matthew Stephenson, Eva Artusi, Paulo Santos, Benoit Clement, Karl Sammut  

**Link**: [PDF](https://arxiv.org/pdf/2502.06113)  

**Abstract**: This paper describes the problem of coordination of an autonomous Multi-Agent System which aims to solve the coverage planning problem in a complex environment. The considered applications are the detection and identification of objects of interest while covering an area. These tasks, which are highly relevant for space applications, are also of interest among various domains including the underwater context, which is the focus of this study. In this context, coverage planning is traditionally modelled as a Markov Decision Process where a coordinated MAS, a swarm of heterogeneous autonomous underwater vehicles, is required to survey an area and search for objects. This MDP is associated with several challenges: environment uncertainties, communication constraints, and an ensemble of hazards, including time-varying and unpredictable changes in the underwater environment. MARL algorithms can solve highly non-linear problems using deep neural networks and display great scalability against an increased number of agents. Nevertheless, most of the current results in the underwater domain are limited to simulation due to the high learning time of MARL algorithms. For this reason, a novel strategy is introduced to accelerate this convergence rate by incorporating biologically inspired heuristics to guide the policy during training. The PSO method, which is inspired by the behaviour of a group of animals, is selected as a heuristic. It allows the policy to explore the highest quality regions of the action and state spaces, from the beginning of the training, optimizing the exploration/exploitation trade-off. The resulting agent requires fewer interactions to reach optimal performance. The method is applied to the MSAC algorithm and evaluated for a 2D covering area mission in a continuous control environment. 

**Abstract (ZH)**: 基于生物启发的策略加速多自主-agent 系统覆盖规划学习方法 

---
# CDM: Contact Diffusion Model for Multi-Contact Point Localization 

**Title (ZH)**: CDM：接触扩散模型多接触点定位 

**Authors**: Seo Wook Han, Min Jun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.06109)  

**Abstract**: In this paper, we propose a Contact Diffusion Model (CDM), a novel learning-based approach for multi-contact point localization. We consider a robot equipped with joint torque sensors and a force/torque sensor at the base. By leveraging a diffusion model, CDM addresses the singularity where multiple pairs of contact points and forces produce identical sensor measurements. We formulate CDM to be conditioned on past model outputs to account for the time-dependent characteristics of the multi-contact scenarios. Moreover, to effectively address the complex shape of the robot surfaces, we incorporate the signed distance field in the denoising process. Consequently, CDM can localize contacts at arbitrary locations with high accuracy. Simulation and real-world experiments demonstrate the effectiveness of the proposed method. In particular, CDM operates at 15.97ms and, in the real world, achieves an error of 0.44cm in single-contact scenarios and 1.24cm in dual-contact scenarios. 

**Abstract (ZH)**: 基于接触扩散模型的多接触点定位方法 

---
# Motion Control in Multi-Rotor Aerial Robots Using Deep Reinforcement Learning 

**Title (ZH)**: 多旋翼飞行机器人基于深度强化学习的运动控制 

**Authors**: Gaurav Shetty, Mahya Ramezani, Hamed Habibi, Holger Voos, Jose Luis Sanchez-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2502.05996)  

**Abstract**: This paper investigates the application of Deep Reinforcement (DRL) Learning to address motion control challenges in drones for additive manufacturing (AM). Drone-based additive manufacturing promises flexible and autonomous material deposition in large-scale or hazardous environments. However, achieving robust real-time control of a multi-rotor aerial robot under varying payloads and potential disturbances remains challenging. Traditional controllers like PID often require frequent parameter re-tuning, limiting their applicability in dynamic scenarios. We propose a DRL framework that learns adaptable control policies for multi-rotor drones performing waypoint navigation in AM tasks. We compare Deep Deterministic Policy Gradient (DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3) within a curriculum learning scheme designed to handle increasing complexity. Our experiments show TD3 consistently balances training stability, accuracy, and success, particularly when mass variability is introduced. These findings provide a scalable path toward robust, autonomous drone control in additive manufacturing. 

**Abstract (ZH)**: 基于深度强化学习的多旋翼无人机在增材制造中的运动控制应用研究 

---
# Mechanic Modeling and Nonlinear Optimal Control of Actively Articulated Suspension of Mobile Heavy-Duty Manipulators 

**Title (ZH)**: 移动重型 manipulator 活动关节悬挂的力学建模与非线性最优控制 

**Authors**: Alvaro Paz, Jouni Mattila  

**Link**: [PDF](https://arxiv.org/pdf/2502.05972)  

**Abstract**: This paper presents the analytic modeling of mobile heavy-duty manipulators with actively articulated suspension and its optimal control to maximize its static and dynamic stabilization. By adopting the screw theory formalism, we consider the suspension mechanism as a rigid multibody composed of two closed kinematic chains. This mechanical modeling allows us to compute the spatial inertial parameters of the whole platform as a function of the suspension's linear actuators through the articulated-body inertia method. Our solution enhances the computation accuracy of the wheels' reaction normal forces by providing an exact solution for the center of mass and inertia tensor of the mobile manipulator. Moreover, these inertial parameters and the normal forces are used to define metrics of both static and dynamic stability of the mobile manipulator and formulate a nonlinear programming problem that optimizes such metrics to generate an optimal stability motion that prevents the platform's overturning, such optimal position of the actuator is tracked with a state-feedback hydraulic valve control. We demonstrate our method's efficiency in terms of C++ computational speed, accuracy and performance improvement by simulating a 7 degrees-of-freedom heavy-duty parallel-serial mobile manipulator with four wheels and actively articulated suspension. 

**Abstract (ZH)**: 基于主动 articulated 悬挂的移动重型的操作机的分析建模及其最优控制以最大化其静态和动态稳定性的研究 

---
# Sustainable Adaptation for Autonomous Driving with the Mixture of Progressive Experts Networ 

**Title (ZH)**: 基于渐进专家网络混合的自主驾驶可持续适应方法 

**Authors**: Yixin Cui, Shuo Yang, Chi Wan, Xincheng Li, Jiaming Xing, Yuanjian Zhang, Yanjun Huang, Hong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05943)  

**Abstract**: Learning-based autonomous driving methods require continuous acquisition of domain knowledge to adapt to diverse driving scenarios. However, due to the inherent challenges of long-tailed data distribution, current approaches still face limitations in complex and dynamic driving environments, particularly when encountering new scenarios and data. This underscores the necessity for enhanced continual learning capabilities to improve system adaptability. To address these challenges, the paper introduces a dynamic progressive optimization framework that facilitates adaptation to variations in dynamic environments, achieved by integrating reinforcement learning and supervised learning for data aggregation. Building on this framework, we propose the Mixture of Progressive Experts (MoPE) network. The proposed method selectively activates multiple expert models based on the distinct characteristics of each task and progressively refines the network architecture to facilitate adaptation to new tasks. Simulation results show that the MoPE model outperforms behavior cloning methods, achieving up to a 7.3% performance improvement in intricate urban road environments. 

**Abstract (ZH)**: 基于学习的自动驾驶方法需要不断获取领域知识以适应多样的驾驶场景。然而，由于长尾数据分布固有的挑战，当前方法在复杂和动态的驾驶环境中仍然面临局限，特别是在遇到新场景和数据时。这突显了增强持续学习能力的必要性，以提高系统的适应性。为应对这些挑战，本文引入了一个动态渐进优化框架，通过结合强化学习和监督学习进行数据聚合，以适应动态环境中的变化。在此框架基础上，我们提出了混合渐进专家网络（MoPE）。所提出的方法根据每个任务的特定特性有选择地激活多个专家模型，并逐步优化网络架构，以促进对新任务的适应。仿真实验结果表明，MoPE模型优于行为克隆方法，在复杂的城市道路环境中性能提升最高可达7.3%。 

---
# Energy-Efficient Autonomous Aerial Navigation with Dynamic Vision Sensors: A Physics-Guided Neuromorphic Approach 

**Title (ZH)**: 基于物理指导的类神经形态方法：动态视觉传感器的能效自主 aerial 导航 

**Authors**: Sourav Sanyal, Amogh Joshi, Manish Nagaraj, Rohan Kumar Manna, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2502.05938)  

**Abstract**: Vision-based object tracking is a critical component for achieving autonomous aerial navigation, particularly for obstacle avoidance. Neuromorphic Dynamic Vision Sensors (DVS) or event cameras, inspired by biological vision, offer a promising alternative to conventional frame-based cameras. These cameras can detect changes in intensity asynchronously, even in challenging lighting conditions, with a high dynamic range and resistance to motion blur. Spiking neural networks (SNNs) are increasingly used to process these event-based signals efficiently and asynchronously. Meanwhile, physics-based artificial intelligence (AI) provides a means to incorporate system-level knowledge into neural networks via physical modeling. This enhances robustness, energy efficiency, and provides symbolic explainability. In this work, we present a neuromorphic navigation framework for autonomous drone navigation. The focus is on detecting and navigating through moving gates while avoiding collisions. We use event cameras for detecting moving objects through a shallow SNN architecture in an unsupervised manner. This is combined with a lightweight energy-aware physics-guided neural network (PgNN) trained with depth inputs to predict optimal flight times, generating near-minimum energy paths. The system is implemented in the Gazebo simulator and integrates a sensor-fused vision-to-planning neuro-symbolic framework built with the Robot Operating System (ROS) middleware. This work highlights the future potential of integrating event-based vision with physics-guided planning for energy-efficient autonomous navigation, particularly for low-latency decision-making. 

**Abstract (ZH)**: 基于视觉的物体跟踪是实现自主空中导航的关键组件，特别是在障碍物避免中。神经形态动态视觉传感器（DVS）或事件相机受生物视觉启发，提供了与传统帧基相机的有前途的替代方案。这些相机即使在具有挑战性的光照条件下，也能异步检测到强度变化，具有高动态范围和抗运动模糊性。突触神经网络（SNNs）被越来越多地用于高效、异步地处理这些基于事件的信号。同时，基于物理的人工智能提供了一种通过物理建模将系统级知识纳入神经网络的方法。这增强了鲁棒性、能量效率，并提供了象征性的可解释性。在本工作中，我们提出了一种神经形态导航框架，用于自主无人机导航。重点在于检测并穿越移动门框，同时避免碰撞。我们通过浅层SNN架构在无监督方式下使用事件相机检测移动物体。这与使用深度输入训练的轻量级能量感知物理引导神经网络（PgNN）结合，用于预测最优飞行时间，生成接近最小能量路径。该系统在Gazebo模拟器中实现，并与基于Robot Operating System（ROS）中间件构建的传感器融合视觉到规划神经符号框架集成。本工作突显了事件视觉与物理引导规划在未来实现高效自主导航中的潜在价值，特别是在低延迟决策中。 

---
# Adaptive Grasping of Moving Objects in Dense Clutter via Global-to-Local Detection and Static-to-Dynamic Planning 

**Title (ZH)**: 基于全局到局部检测与静态到动态规划的密集杂乱环境中移动物体的适应性抓取 

**Authors**: Hao Chen, Takuya Kiyokawa, Weiwei Wan, Kensuke Harada  

**Link**: [PDF](https://arxiv.org/pdf/2502.05916)  

**Abstract**: Robotic grasping is facing a variety of real-world uncertainties caused by non-static object states, unknown object properties, and cluttered object arrangements. The difficulty of grasping increases with the presence of more uncertainties, where commonly used learning-based approaches struggle to perform consistently across varying conditions. In this study, we integrate the idea of similarity matching to tackle the challenge of grasping novel objects that are simultaneously in motion and densely cluttered using a single RGBD camera, where multiple uncertainties coexist. We achieve this by shifting visual detection from global to local states and operating grasp planning from static to dynamic scenes. Notably, we introduce optimization methods to enhance planning efficiency for this time-sensitive task. Our proposed system can adapt to various object types, arrangements and movement speeds without the need for extensive training, as demonstrated by real-world experiments. 

**Abstract (ZH)**: 基于相似性匹配的单目RGBD相机在动kening紧密布置的待抓取物上的抓取规划 

---
# EvoAgent: Agent Autonomous Evolution with Continual World Model for Long-Horizon Tasks 

**Title (ZH)**: EvoAgent：基于持续世界模型的智能体自主进化方法用于长时_horizon任务 

**Authors**: Tongtong Feng, Xin Wang, Zekai Zhou, Ren Wang, Yuwei Zhan, Guangyao Li, Qing Li, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.05907)  

**Abstract**: Completing Long-Horizon (LH) tasks in open-ended worlds is an important yet difficult problem for embodied agents. Existing approaches suffer from two key challenges: (1) they heavily rely on experiences obtained from human-created data or curricula, lacking the ability to continuously update multimodal experiences, and (2) they may encounter catastrophic forgetting issues when faced with new tasks, lacking the ability to continuously update world knowledge. To solve these challenges, this paper presents EvoAgent, an autonomous-evolving agent with a continual World Model (WM), which can autonomously complete various LH tasks across environments through self-planning, self-control, and self-reflection, without human intervention. Our proposed EvoAgent contains three modules, i.e., i) the memory-driven planner which uses an LLM along with the WM and interaction memory, to convert LH tasks into executable sub-tasks; ii) the WM-guided action controller which leverages WM to generate low-level actions and incorporates a self-verification mechanism to update multimodal experiences; iii) the experience-inspired reflector which implements a two-stage curriculum learning algorithm to select experiences for task-adaptive WM updates. Moreover, we develop a continual World Model for EvoAgent, which can continuously update the multimodal experience pool and world knowledge through closed-loop dynamics. We conducted extensive experiments on Minecraft, compared with existing methods, EvoAgent can achieve an average success rate improvement of 105% and reduce ineffective actions by more than 6x. 

**Abstract (ZH)**: 开放环境中长时 horizon（LH）任务的自主完成是沉浸式代理面临的重要但困难的问题。现有方法面临两个关键挑战：（1）它们高度依赖于人类创建的数据或课程学习的经验，缺乏不断更新多模态经验的能力；（2）在面对新任务时可能会遇到灾难性遗忘的问题，缺乏不断更新世界知识的能力。为了解决这些挑战，本文提出了一种自主进化代理 EvoAgent，该代理配备持续的世界模型（WM），可以通过自我规划、自我控制和自我反思，无需人类干预，便能跨环境自主完成各种 LH 任务。我们提出的 EvoAgent 包含三个模块，即：i）以记忆驱动的规划器，使用语言模型（LLM）与世界模型和交互记忆结合，将 LH 任务转换为可执行的子任务；ii）由世界模型引导的动作控制器，利用世界模型生成低级动作，并结合自我验证机制更新多模态经验；iii）经验启发的反思器，实施两阶段的课程学习算法，为任务适配的世界模型更新选择经验。此外，我们为 EvoAgent 开发了一种持续的世界模型，可以利用闭环动态不断更新多模态经验池和世界知识。在 Minecraft 上进行的广泛实验表明，与现有方法相比，EvoAgent 可以实现平均成功率提高 105%，并将无效动作减少超过 6 倍。 

---
# DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control 

**Title (ZH)**: DexVLA：配备插件扩散专家的视觉-语言模型用于通用机器人控制 

**Authors**: Junjie Wen, Yichen Zhu, Jinming Li, Zhibin Tang, Chaomin Shen, Feifei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.05855)  

**Abstract**: Enabling robots to perform diverse tasks across varied environments is a central challenge in robot learning. While vision-language-action (VLA) models have shown promise for generalizable robot skills, realizing their full potential requires addressing limitations in action representation and efficient training. Current VLA models often focus on scaling the vision-language model (VLM) component, while the action space representation remains a critical bottleneck. This paper introduces DexVLA, a novel framework designed to enhance the efficiency and generalization capabilities of VLAs for complex, long-horizon tasks across diverse robot embodiments. DexVLA features a novel diffusion-based action expert, scaled to one billion parameters, designed for cross-embodiment learning. A novel embodiment curriculum learning strategy facilitates efficient training: (1) pre-training the diffusion expert that is separable from the VLA on cross-embodiment data, (2) aligning the VLA model to specific embodiments, and (3) post-training for rapid adaptation to new tasks. We conduct comprehensive experiments across multiple embodiments, including single-arm, bimanual, and dexterous hand, demonstrating DexVLA's adaptability to challenging tasks without task-specific adaptation, its ability to learn dexterous skills on novel embodiments with limited data, and its capacity to complete complex, long-horizon tasks using only direct language prompting, such as laundry folding. In all settings, our method demonstrates superior performance compared to state-of-the-art models like Octo, OpenVLA, and Diffusion Policy. 

**Abstract (ZH)**: 使机器人能够在多样化的环境中执行多种任务是机器人学习中的一个核心挑战。尽管视觉-语言-行动（VLA）模型在通用机器人技能方面显示出潜力，但要充分发挥其潜力，需要解决行动表示和高效训练的限制。当前的VLA模型通常侧重于扩展视觉-语言模型（VLM）组件，而行动空间表示仍然是一个关键瓶颈。本文介绍了一种名为DexVLA的新框架，旨在增强VLA在多样机器人实体中的高效性和泛化能力，用于复杂、长时程的任务。DexVLA特征是一种新型的基于扩散的动作专家，参数规模高达十亿，设计用于跨实体学习。一种新的实体课程学习策略促进了高效训练：（1）在跨实体数据上预先训练与VLA分离的动作专家，（2）将VLA模型与特定实体对齐，（3）在最终训练中快速适应新任务。我们在多种实体上进行了全面的实验，包括单臂、双臂和灵巧手，展示了DexVLA在具有挑战性任务中的适应性，能够在有限数据下学习新实体上的灵巧技能，并仅通过直接的语言提示完成复杂的长期任务，如衣物折叠。在所有设置中，我们的方法在与Octo、OpenVLA和Diffusion Policy等最先进模型的性能比较中表现出色。 

---
# DreamFLEX: Learning Fault-Aware Quadrupedal Locomotion Controller for Anomaly Situation in Rough Terrains 

**Title (ZH)**: DreamFLEX：学习故障感知的四足步行控制器以应对粗糙地形中的异常情况 

**Authors**: Seunghyun Lee, I Made Aswin Nahrendra, Dongkyu Lee, Byeongho Yu, Minho Oh, Hyun Myung  

**Link**: [PDF](https://arxiv.org/pdf/2502.05817)  

**Abstract**: Recent advances in quadrupedal robots have demonstrated impressive agility and the ability to traverse diverse terrains. However, hardware issues, such as motor overheating or joint locking, may occur during long-distance walking or traversing through rough terrains leading to locomotion failures. Although several studies have proposed fault-tolerant control methods for quadrupedal robots, there are still challenges in traversing unstructured terrains. In this paper, we propose DreamFLEX, a robust fault-tolerant locomotion controller that enables a quadrupedal robot to traverse complex environments even under joint failure conditions. DreamFLEX integrates an explicit failure estimation and modulation network that jointly estimates the robot's joint fault vector and utilizes this information to adapt the locomotion pattern to faulty conditions in real-time, enabling quadrupedal robots to maintain stability and performance in rough terrains. Experimental results demonstrate that DreamFLEX outperforms existing methods in both simulation and real-world scenarios, effectively managing hardware failures while maintaining robust locomotion performance. 

**Abstract (ZH)**: Recent Advances in Quadrupedal Robots: DreamFLEX——一种鲁棒的故障 tolerant 行走控制器及其在关节故障条件下的应用 

---
# AToM: Adaptive Theory-of-Mind-Based Human Motion Prediction in Long-Term Human-Robot Interactions 

**Title (ZH)**: AToM：基于自适应心智理论的人motion预测在长时间人机交互中的应用 

**Authors**: Yuwen Liao, Muqing Cao, Xinhang Xu, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.05792)  

**Abstract**: Humans learn from observations and experiences to adjust their behaviours towards better performance. Interacting with such dynamic humans is challenging, as the robot needs to predict the humans accurately for safe and efficient operations. Long-term interactions with dynamic humans have not been extensively studied by prior works. We propose an adaptive human prediction model based on the Theory-of-Mind (ToM), a fundamental social-cognitive ability that enables humans to infer others' behaviours and intentions. We formulate the human internal belief about others using a game-theoretic model, which predicts the future motions of all agents in a navigation scenario. To estimate an evolving belief, we use an Unscented Kalman Filter to update the behavioural parameters in the human internal model. Our formulation provides unique interpretability to dynamic human behaviours by inferring how the human predicts the robot. We demonstrate through long-term experiments in both simulations and real-world settings that our prediction effectively promotes safety and efficiency in downstream robot planning. Code will be available at this https URL. 

**Abstract (ZH)**: 人类通过观察和经验调整行为以获得更好的表现。与这样的动态人类交互具有挑战性，因为机器人需要准确预测人类行为以确保安全和高效的操作。先前的研究尚未广泛探讨长期与动态人类的交互。我们提出了一种基于理论思维（ToM）的自适应人类预测模型，理论思维是一种基础的社会认知能力，使人类能够推断他人的行为和意图。我们使用博弈论模型来表达人类对他人内部信念，并预测导航场景中所有代理的未来运动。为估计不断 evolving 的信念，我们使用无迹卡尔曼滤波器来更新人类内部模型中的行为参数。我们的建模为动态人类行为提供了独特的可解释性，通过推断人类如何预测机器人。我们在仿真和真实-world 设置中通过长期实验演示，我们的预测有效促进了下游机器人规划的安全性和效率。代码将在此处 https:// 提供。 

---
# Implicit Communication of Contextual Information in Human-Robot Collaboration 

**Title (ZH)**: 人类与机器人协作中的隐式背景信息沟通 

**Authors**: Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05775)  

**Abstract**: Implicit communication is crucial in human-robot collaboration (HRC), where contextual information, such as intentions, is conveyed as implicatures, forming a natural part of human interaction. However, enabling robots to appropriately use implicit communication in cooperative tasks remains challenging. My research addresses this through three phases: first, exploring the impact of linguistic implicatures on collaborative tasks; second, examining how robots' implicit cues for backchanneling and proactive communication affect team performance and perception, and how they should adapt to human teammates; and finally, designing and evaluating a multi-LLM robotics system that learns from human implicit communication. This research aims to enhance the natural communication abilities of robots and facilitate their integration into daily collaborative activities. 

**Abstract (ZH)**: 隐含沟通在人机协作中的作用至关重要，其中上下文信息，如意图，作为 implicatures 传递，成为人类交互自然的一部分。然而，使机器人在协作任务中恰当地使用隐含沟通仍然具有挑战性。我的研究通过三个阶段来解决这一问题：首先，探索语言 implicatures 对协作任务的影响；其次，研究机器人在回话填补和主动沟通中隐含信号如何影响团队绩效和感知，并探讨它们如何适应人类队友；最后，设计并评估一个从人类隐含沟通中学习的多语言模型机器人系统。这项研究旨在增强机器人的自然沟通能力，并促进它们融入日常协作活动中。 

---
# PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map 

**Title (ZH)**: PINGS：基于点的隐式神经映射中的高斯散点与距离场相结合 

**Authors**: Yue Pan, Xingguang Zhong, Liren Jin, Louis Wiesmann, Marija Popović, Jens Behley, Cyrill Stachniss  

**Link**: [PDF](https://arxiv.org/pdf/2502.05752)  

**Abstract**: Robots require high-fidelity reconstructions of their environment for effective operation. Such scene representations should be both, geometrically accurate and photorealistic to support downstream tasks. While this can be achieved by building distance fields from range sensors and radiance fields from cameras, the scalable incremental mapping of both fields consistently and at the same time with high quality remains challenging. In this paper, we propose a novel map representation that unifies a continuous signed distance field and a Gaussian splatting radiance field within an elastic and compact point-based implicit neural map. By enforcing geometric consistency between these fields, we achieve mutual improvements by exploiting both modalities. We devise a LiDAR-visual SLAM system called PINGS using the proposed map representation and evaluate it on several challenging large-scale datasets. Experimental results demonstrate that PINGS can incrementally build globally consistent distance and radiance fields encoded with a compact set of neural points. Compared to the state-of-the-art methods, PINGS achieves superior photometric and geometric rendering at novel views by leveraging the constraints from the distance field. Furthermore, by utilizing dense photometric cues and multi-view consistency from the radiance field, PINGS produces more accurate distance fields, leading to improved odometry estimation and mesh reconstruction. 

**Abstract (ZH)**: 机器人需要对其环境进行高保真重建以实现有效操作。这样的场景表示既要几何上准确，又要具备照片现实性，以支持下游任务。虽然可以通过构建来自距离传感器的距离场和来自相机的辐射场来实现这一点，但同时以高质量的方式进行可扩展的增量映射仍具有挑战性。在本文中，我们提出了一种新的地图表示法，该表示法将连续的带符号距离场和高斯采样辐射场统一在一种弹性且紧凑的基于点的隐式神经地图中。通过在这些领域之间强求几何一致性，我们通过利用这两种模态的优势实现了相互改进。我们使用所提出的地图表示法开发了一种基于激光雷达和视觉的SLAM系统PINGS，并在多个具有挑战性的大规模数据集上对其进行评估。实验结果表明，PINGS能够通过紧凑的神经点集增量构建全局一致的距离和辐射场。与现有方法相比，PINGS通过利用距离场的约束条件，在新视角下实现了更优的 photometric 和几何渲染。此外，PINGS 利用辐射场中的密集 photometric 提示和多视角一致性，生成更准确的距离场，从而提高运动估计和网格重建的精度。 

---
# Hierarchical Equivariant Policy via Frame Transf 

**Title (ZH)**: 基于框架变换的分层等变策略 

**Authors**: Haibo Zhao, Dian Wang, Yizhe Zhu, Xupeng Zhu, Owen Howell, Linfeng Zhao, Yaoyao Qian, Robin Walters, Robert Platt  

**Link**: [PDF](https://arxiv.org/pdf/2502.05728)  

**Abstract**: Recent advances in hierarchical policy learning highlight the advantages of decomposing systems into high-level and low-level agents, enabling efficient long-horizon reasoning and precise fine-grained control. However, the interface between these hierarchy levels remains underexplored, and existing hierarchical methods often ignore domain symmetry, resulting in the need for extensive demonstrations to achieve robust performance. To address these issues, we propose Hierarchical Equivariant Policy (HEP), a novel hierarchical policy framework. We propose a frame transfer interface for hierarchical policy learning, which uses the high-level agent's output as a coordinate frame for the low-level agent, providing a strong inductive bias while retaining flexibility. Additionally, we integrate domain symmetries into both levels and theoretically demonstrate the system's overall equivariance. HEP achieves state-of-the-art performance in complex robotic manipulation tasks, demonstrating significant improvements in both simulation and real-world settings. 

**Abstract (ZH)**: 最近在层次性策略学习方面的进展突显了将系统分解为高层和低层代理的优势，使得高效的长期推理和精确的细粒度控制成为可能。然而，这些层次之间的接口仍缺乏探索，现有的层次方法经常忽视领域对称性，导致需要大量的演示以实现稳健的性能。为了解决这些问题，我们提出了一种新型的层次性等变策略（HEP）框架。我们提出了一种框架转移接口，该接口使用高层代理的输出作为低层代理的坐标系，提供了强烈的归纳偏差同时保持灵活性。此外，我们还将领域对称性集成到两个层次中，并从理论上证明了系统的整体等变性。HEP在复杂的机器人操作任务中实现了最先进的性能，表明其在仿真和真实世界设置中均取得了显著改进。 

---
# Implicit Physics-aware Policy for Dynamic Manipulation of Rigid Objects via Soft Body Tools 

**Title (ZH)**: 基于刚体对象动态操作的软体工具驱动隐式物理意识策略 

**Authors**: Zixing Wang, Ahmed H. Qureshi  

**Link**: [PDF](https://arxiv.org/pdf/2502.05696)  

**Abstract**: Recent advancements in robot tool use have unlocked their usage for novel tasks, yet the predominant focus is on rigid-body tools, while the investigation of soft-body tools and their dynamic interaction with rigid bodies remains unexplored. This paper takes a pioneering step towards dynamic one-shot soft tool use for manipulating rigid objects, a challenging problem posed by complex interactions and unobservable physical properties. To address these problems, we propose the Implicit Physics-aware (IPA) policy, designed to facilitate effective soft tool use across various environmental configurations. The IPA policy conducts system identification to implicitly identify physics information and predict goal-conditioned, one-shot actions accordingly. We validate our approach through a challenging task, i.e., transporting rigid objects using soft tools such as ropes to distant target positions in a single attempt under unknown environment physics parameters. Our experimental results indicate the effectiveness of our method in efficiently identifying physical properties, accurately predicting actions, and smoothly generalizing to real-world environments. The related video is available at: this https URL 

**Abstract (ZH)**: Recent advancements in机器人工具使用方面的 recent advancements in 机器人工具使用方面的进展解锁了它们在新任务中的应用，但主要集中在刚体工具上，而关于软体工具及其与刚体的动态交互的研究尚未探索。本文在利用软体工具动态操控刚体这一具有复杂交互和不可观测物理性质挑战性问题上取得了先驱性进展。为了应对这些问题，我们提出了隐式物理意识（IPA）策略，旨在在各种环境配置下促进有效的软体工具使用。IPA策略通过隐式识别物理信息来制定相应的目标导向、一次执行的动作。我们通过一个具有挑战性的任务——在未知环境物理参数条件下，使用软体工具（如绳子）一次性将刚体物体运输到远程目标位置——验证了我们的方法。我们的实验结果表明，该方法在高效识别物理性质、准确预测动作以及平滑地应用于真实环境方面的有效性。相关视频可参见：this https URL。 

---
# Vertical Vibratory Transport of Grasped Parts Using Impacts 

**Title (ZH)**: 抓取部件的垂直振动运输撞击方法 

**Authors**: C. L. Yako, Jérôme Nowak, Shenli Yuan, Kenneth Salisbury  

**Link**: [PDF](https://arxiv.org/pdf/2502.05693)  

**Abstract**: In this paper, we use impact-induced acceleration in conjunction with periodic stick-slip to successfully and quickly transport parts vertically against gravity. We show analytically that vertical vibratory transport is more difficult than its horizontal counterpart, and provide guidelines for achieving optimal vertical vibratory transport of a part. Namely, such a system must be capable of quickly realizing high accelerations, as well as supply normal forces at least several times that required for static equilibrium. We also show that for a given maximum acceleration, there is an optimal normal force for transport. To test our analytical guidelines, we built a vibrating surface using flexures and a voice coil actuator that can accelerate a magnetic ram into various materials to generate impacts. The surface was used to transport a part against gravity. Experimentally obtained motion tracking data confirmed the theoretical model. A series of grasping tests with a vibrating-surface equipped parallel jaw gripper confirmed the design guidelines. 

**Abstract (ZH)**: 在本文中，我们利用冲击引起的加速度结合周期性滑移粘着成功快速地实现了零件垂直方向上的反重力运输。我们从理论上证明了垂直振动运输比水平运输更困难，并提供了实现最优垂直振动运输的指南。具体来说，这样的系统必须能够快速实现高加速度，并且能够提供至少是静力平衡所需正常力几倍的正常力。我们还证明，在给定的最大加速度下，存在一个最佳的正常力用于运输。为了验证我们的理论指南，我们使用柔性铰链和声音线圈执行器构建了一个振动表面，并通过使一个磁性撞针进入各种材料来生成冲击，该表面被用于反重力运输零件。实验获得的运动跟踪数据证实了理论模型。配备振动表面的并指夹持器的一系列夹持测试进一步验证了设计指南。 

---
# Surprise Potential as a Measure of Interactivity in Driving Scenarios 

**Title (ZH)**: Surprise潜力作为驾驶场景中互动性的度量 

**Authors**: Wenhao Ding, Sushant Veer, Karen Leung, Yulong Cao, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2502.05677)  

**Abstract**: Validating the safety and performance of an autonomous vehicle (AV) requires benchmarking on real-world driving logs. However, typical driving logs contain mostly uneventful scenarios with minimal interactions between road users. Identifying interactive scenarios in real-world driving logs enables the curation of datasets that amplify critical signals and provide a more accurate assessment of an AV's performance. In this paper, we present a novel metric that identifies interactive scenarios by measuring an AV's surprise potential on others. First, we identify three dimensions of the design space to describe a family of surprise potential measures. Second, we exhaustively evaluate and compare different instantiations of the surprise potential measure within this design space on the nuScenes dataset. To determine how well a surprise potential measure correctly identifies an interactive scenario, we use a reward model learned from human preferences to assess alignment with human intuition. Our proposed surprise potential, arising from this exhaustive comparative study, achieves a correlation of more than 0.82 with the human-aligned reward function, outperforming existing approaches. Lastly, we validate motion planners on curated interactive scenarios to demonstrate downstream applications. 

**Abstract (ZH)**: 验证自动驾驶汽车（AV）的安全性和性能需要在实际驾驶日志中进行基准测试。然而，典型的驾驶日志主要包含无事件场景，路用户之间的互动很少。在实际驾驶日志中识别互动场景能够创建更能放大关键信号的数据集，从而提供对AV性能更准确的评估。本文提出了一种新的度量标准，通过衡量AV对其他方的惊喜潜力来识别互动场景。首先，我们定义了设计空间的三个维度以描述惊喜潜力度量的一组度量。其次，我们在nuScenes数据集上全面评估和比较设计空间内不同实例化的惊喜潜力度量。为了确定惊喜潜力度量在多大程度上正确识别互动场景，我们使用从人类偏好中学习到的奖励模型来评估其与人类直觉的契合度。我们提出的一种全面比较研究中产生的惊喜潜力，与人类对齐的奖励函数的相关性超过0.82，优于现有方法。最后，我们在精心策划的互动场景中验证运动规划器，以展示下游应用。 

---
# Online Controller Synthesis for Robot Collision Avoidance: A Case Study 

**Title (ZH)**: 机器人碰撞避免的在线控制器合成：一个案例研究 

**Authors**: Yuheng Fan, Wang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.05667)  

**Abstract**: The inherent uncertainty of dynamic environments poses significant challenges for modeling robot behavior, particularly in tasks such as collision avoidance. This paper presents an online controller synthesis framework tailored for robots equipped with deep learning-based perception components, with a focus on addressing distribution shifts. Our approach integrates periodic monitoring and repair mechanisms for the deep neural network perception component, followed by uncertainty reassessment. These uncertainty evaluations are injected into a parametric discrete-time markov chain, enabling the synthesis of robust controllers via probabilistic model checking. To ensure high system availability during the repair process, we propose a dual-component configuration that seamlessly transitions between operational states. Through a case study on robot collision avoidance, we demonstrate the efficacy of our method, showcasing substantial performance improvements over baseline approaches. This work provides a comprehensive and scalable solution for enhancing the safety and reliability of autonomous systems operating in uncertain environments. 

**Abstract (ZH)**: 动态环境中的固有不确定性给机器人行为建模带来了显著挑战，尤其是在避碰任务中。本文提出了一种针对配备基于深度学习感知组件的机器人在线控制器综合框架，重点关注解决分布偏移问题。该方法将周期性监控和修复机制整合到深度神经网络感知组件中，随后进行不确定性重新评估。这些不确定性评估被注入到参数离散时间马尔可夫链中，从而通过概率模型检查合成鲁棒控制器。为了在修复过程中确保系统的高可用性，我们提出了一种双组件配置，可在操作状态之间无缝切换。通过机器人避碰案例研究，我们展示了该方法的有效性，展示了与基线方法相比显著的性能改进。本文为在不确定环境中运行的自主系统提高了安全性和可靠性提供了全面且可扩展的解决方案。 

---
# Generating Physically Realistic and Directable Human Motions from Multi-Modal Inputs 

**Title (ZH)**: 从多模态输入生成物理上真实且可导向的人类动作 

**Authors**: Aayam Shrestha, Pan Liu, German Ros, Kai Yuan, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2502.05641)  

**Abstract**: This work focuses on generating realistic, physically-based human behaviors from multi-modal inputs, which may only partially specify the desired motion. For example, the input may come from a VR controller providing arm motion and body velocity, partial key-point animation, computer vision applied to videos, or even higher-level motion goals. This requires a versatile low-level humanoid controller that can handle such sparse, under-specified guidance, seamlessly switch between skills, and recover from failures. Current approaches for learning humanoid controllers from demonstration data capture some of these characteristics, but none achieve them all. To this end, we introduce the Masked Humanoid Controller (MHC), a novel approach that applies multi-objective imitation learning on augmented and selectively masked motion demonstrations. The training methodology results in an MHC that exhibits the key capabilities of catch-up to out-of-sync input commands, combining elements from multiple motion sequences, and completing unspecified parts of motions from sparse multimodal input. We demonstrate these key capabilities for an MHC learned over a dataset of 87 diverse skills and showcase different multi-modal use cases, including integration with planning frameworks to highlight MHC's ability to solve new user-defined tasks without any finetuning. 

**Abstract (ZH)**: 基于多模态输入生成现实物理驱动的人类行为的研究：引入掩码 humanoid 控制器（MHC） 

---
# Data efficient Robotic Object Throwing with Model-Based Reinforcement Learning 

**Title (ZH)**: 基于模型的强化学习在数据高效机器人物体投掷中的应用 

**Authors**: Niccolò Turcato, Giulio Giacomuzzo, Matteo Terreran, Davide Allegro, Ruggero Carli, Alberto Dalla Libera  

**Link**: [PDF](https://arxiv.org/pdf/2502.05595)  

**Abstract**: Pick-and-place (PnP) operations, featuring object grasping and trajectory planning, are fundamental in industrial robotics applications. Despite many advancements in the field, PnP is limited by workspace constraints, reducing flexibility. Pick-and-throw (PnT) is a promising alternative where the robot throws objects to target locations, leveraging extrinsic resources like gravity to improve efficiency and expand the workspace. However, PnT execution is complex, requiring precise coordination of high-speed movements and object dynamics. Solutions to the PnT problem are categorized into analytical and learning-based approaches. Analytical methods focus on system modeling and trajectory generation but are time-consuming and offer limited generalization. Learning-based solutions, in particular Model-Free Reinforcement Learning (MFRL), offer automation and adaptability but require extensive interaction time. This paper introduces a Model-Based Reinforcement Learning (MBRL) framework, MC-PILOT, which combines data-driven modeling with policy optimization for efficient and accurate PnT tasks. MC-PILOT accounts for model uncertainties and release errors, demonstrating superior performance in simulations and real-world tests with a Franka Emika Panda manipulator. The proposed approach generalizes rapidly to new targets, offering advantages over analytical and Model-Free methods. 

**Abstract (ZH)**: Pick-and-Throw (PnT) 操作结合目标抓取和轨迹规划，在工业机器人应用中具有前景，但执行复杂，需要精确协调高速运动和物体动力学。本文介绍了一种基于模型的强化学习（MBRL）框架 MC-PILOT，该框架结合数据驱动建模与策略优化，以高效准确地执行 PnT 任务。MC-PILOT 考虑了模型不确定性及释放误差，在使用 Franka Emika Panda 操作器的仿真和现实世界测试中均表现出优越性能。该方法能快速泛化到新目标，优于分析法和无模型强化学习方法。 

---
# Towards Learning Scalable Agile Dynamic Motion Planning for Robosoccer Teams with Policy Optimization 

**Title (ZH)**: 面向 robosoccer 队伍的可扩展敏捷动态运动规划的学习方法研究（基于策略优化） 

**Authors**: Brandon Ho, Batuhan Altundas, Matthew Gombolay  

**Link**: [PDF](https://arxiv.org/pdf/2502.05526)  

**Abstract**: In fast-paced, ever-changing environments, dynamic Motion Planning for Multi-Agent Systems in the presence of obstacles is a universal and unsolved problem. Be it from path planning around obstacles to the movement of robotic arms, or in planning navigation of robot teams in settings such as Robosoccer, dynamic motion planning is needed to avoid collisions while reaching the targeted destination when multiple agents occupy the same area. In continuous domains where the world changes quickly, existing classical Motion Planning algorithms such as RRT* and A* become computationally expensive to rerun at every time step. Many variations of classical and well-formulated non-learning path-planning methods have been proposed to solve this universal problem but fall short due to their limitations of speed, smoothness, optimally, etc. Deep Learning models overcome their challenges due to their ability to adapt to varying environments based on past experience. However, current learning motion planning models use discretized environments, do not account for heterogeneous agents or replanning, and build up to improve the classical motion planners' efficiency, leading to issues with scalability. To prevent collisions between heterogenous team members and collision to obstacles while trying to reach the target location, we present a learning-based dynamic navigation model and show our model working on a simple environment in the concept of a simple Robosoccer Game. 

**Abstract (ZH)**: 在瞬息万变的环境中，多Agent系统在障碍物存在下的动态运动规划是一个普遍且未解决的问题。无论是路径规划绕过障碍物，还是在如Robosoccer设定下的机器人团队导航中，动态运动规划都需避免碰撞并到达目标位置。在世界快速变化的连续域中，现有经典运动规划算法如RRT*和A*在每一步重新运行时变得计算成本高昂。尽管提出了许多经典和非学习路径规划方法的变体来解决这一普遍问题，但由于速度、平滑性和最优性等方面的限制，它们仍存在不足。基于过往经验适应不同环境的深度学习模型可以克服这些挑战。然而，当前的学习运动规划模型使用离散化环境，未考虑异质Agent或重新规划，旨在提高经典运动规划器的效率，导致可扩展性问题。为防止异质团队成员之间的碰撞以及与障碍物的碰撞，在尝试达到目标位置时，我们提出了一种基于学习的动态导航模型，并展示了该模型在简单Robosoccer游戏环境中的应用。 

---
# Vision-Ultrasound Robotic System based on Deep Learning for Gas and Arc Hazard Detection in Manufacturing 

**Title (ZH)**: 基于深度学习的视觉-超声机器人系统及其在制造业气体和电弧危害检测中的应用 

**Authors**: Jin-Hee Lee, Dahyun Nam, Robin Inho Kee, YoungKey Kim, Seok-Jun Buu  

**Link**: [PDF](https://arxiv.org/pdf/2502.05500)  

**Abstract**: Gas leaks and arc discharges present significant risks in industrial environments, requiring robust detection systems to ensure safety and operational efficiency. Inspired by human protocols that combine visual identification with acoustic verification, this study proposes a deep learning-based robotic system for autonomously detecting and classifying gas leaks and arc discharges in manufacturing settings. The system is designed to execute all experimental tasks entirely onboard the robot. Utilizing a 112-channel acoustic camera operating at a 96 kHz sampling rate to capture ultrasonic frequencies, the system processes real-world datasets recorded in diverse industrial scenarios. These datasets include multiple gas leak configurations (e.g., pinhole, open end) and partial discharge types (Corona, Surface, Floating) under varying environmental noise conditions. Proposed system integrates visual detection and a beamforming-enhanced acoustic analysis pipeline. Signals are transformed using STFT and refined through Gamma Correction, enabling robust feature extraction. An Inception-inspired CNN further classifies hazards, achieving 99% gas leak detection accuracy. The system not only detects individual hazard sources but also enhances classification reliability by fusing multi-modal data from both vision and acoustic sensors. When tested in reverberation and noise-augmented environments, the system outperformed conventional models by up to 44%p, with experimental tasks meticulously designed to ensure fairness and reproducibility. Additionally, the system is optimized for real-time deployment, maintaining an inference time of 2.1 seconds on a mobile robotic platform. By emulating human-like inspection protocols and integrating vision with acoustic modalities, this study presents an effective solution for industrial automation, significantly improving safety and operational reliability. 

**Abstract (ZH)**: 基于深度学习的机器人系统在制造业环境中自主检测和分类气体泄漏和电弧放电 

---
# Lie-algebra Adaptive Tracking Control for Rigid Body Dynamics 

**Title (ZH)**: 刚体动力学的李代数自适应跟踪控制 

**Authors**: Jiawei Tang, Shilei Li, Ling Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.05491)  

**Abstract**: Adaptive tracking control for rigid body dynamics is of critical importance in control and robotics, particularly for addressing uncertainties or variations in system model parameters. However, most existing adaptive control methods are designed for systems with states in vector spaces, often neglecting the manifold constraints inherent to robotic systems. In this work, we propose a novel Lie-algebra-based adaptive control method that leverages the intrinsic relationship between the special Euclidean group and its associated Lie algebra. By transforming the state space from the group manifold to a vector space, we derive a linear error dynamics model that decouples model parameters from the system state. This formulation enables the development of an adaptive optimal control method that is both geometrically consistent and computationally efficient. Extensive simulations demonstrate the effectiveness and efficiency of the proposed method. We have made our source code publicly available to the community to support further research and collaboration. 

**Abstract (ZH)**: 基于李代数的 rigid 体动力学自适应跟踪控制方法在控制与机器人领域至关重要，特别是在处理系统模型参数的不确定性或变化时。然而，现有的大多数自适应控制方法针对的是状态位于向量空间中的系统，往往忽略了机器人系统固有的流形约束。在本工作中，我们提出了一种新的基于李代数的自适应控制方法，该方法利用特殊欧几里得群与其关联的李代数之间的内在关系。通过将状态空间从流形转换到向量空间，我们推导出一个解耦模型参数与系统状态的线性误差动态模型。该表述形式使得能够开发出既几何上一致又计算效率高的自适应最优控制方法。广泛的仿真实验证实了所提出方法的有效性和高效性。我们已将源代码开源以支持进一步的研究和合作。 

---
# HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation 

**Title (ZH)**: HAMSTER: 开阔场景下基于层级动作模型的机器人操作方法 

**Authors**: Yi Li, Yuquan Deng, Jesse Zhang, Joel Jang, Marius Memme, Raymond Yu, Caelan Reed Garrett, Fabio Ramos, Dieter Fox, Anqi Li, Abhishek Gupta, Ankit Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2502.05485)  

**Abstract**: Large foundation models have shown strong open-world generalization to complex problems in vision and language, but similar levels of generalization have yet to be achieved in robotics. One fundamental challenge is the lack of robotic data, which are typically obtained through expensive on-robot operation. A promising remedy is to leverage cheaper, off-domain data such as action-free videos, hand-drawn sketches or simulation data. In this work, we posit that hierarchical vision-language-action (VLA) models can be more effective in utilizing off-domain data than standard monolithic VLA models that directly finetune vision-language models (VLMs) to predict actions. In particular, we study a class of hierarchical VLA models, where the high-level VLM is finetuned to produce a coarse 2D path indicating the desired robot end-effector trajectory given an RGB image and a task description. The intermediate 2D path prediction is then served as guidance to the low-level, 3D-aware control policy capable of precise manipulation. Doing so alleviates the high-level VLM from fine-grained action prediction, while reducing the low-level policy's burden on complex task-level reasoning. We show that, with the hierarchical design, the high-level VLM can transfer across significant domain gaps between the off-domain finetuning data and real-robot testing scenarios, including differences on embodiments, dynamics, visual appearances and task semantics, etc. In the real-robot experiments, we observe an average of 20% improvement in success rate across seven different axes of generalization over OpenVLA, representing a 50% relative gain. Visual results are provided at: this https URL 

**Abstract (ZH)**: 大型基础模型在视觉和语言领域展示了强大的开放世界泛化能力，但在机器人领域尚未达到类似水平。一个基本挑战是对机器人数据的缺乏，这些数据通常通过昂贵的机器人操作获得。一种有前景的解决方法是利用更便宜的离域数据，如无动作视频、手绘草图或仿真数据。在本文中，我们提出层次视觉-语言-动作（VLA）模型比直接微调视觉-语言模型（VLMs）以预测动作的标准单一VLA模型更能有效地利用离域数据。特别是，我们研究了一类层次VLA模型，其中高层VLM微调以生成粗略的二维路径，表示给定RGB图像和任务描述时期望的机器人末端执行器轨迹。中间的二维路径预测则作为指导，用于低层、三维感知控制策略，该策略能够进行精确操作。这样做可以减轻高层VLM的细粒度动作预测负担，同时减少低层策略在复杂任务级推理上的负担。我们展示了，在层次设计下，高层VLM可以跨越显著的领域差距，将在离域数据微调和真实机器人测试场景之间，包括实体、动力学、视觉外观和任务语义等方面的差异进行泛化。在真实机器人实验中，我们观察到与OpenVLA在七个不同泛化轴上的成功率平均提高20%，代表了50%的相对增幅。视觉结果请参见：this https URL 

---
# Model Validity in Observers: When to Increase the Complexity of Your Model? 

**Title (ZH)**: 观察者中模型有效性的问题：何时增加模型的复杂度？ 

**Authors**: Agapius Bou Ghosn, Philip Polack, Arnaud de La Fortelle  

**Link**: [PDF](https://arxiv.org/pdf/2502.05479)  

**Abstract**: Model validity is key to the accurate and safe behavior of autonomous vehicles. Using invalid vehicle models in the different plan and control vehicle frameworks puts the stability of the vehicle, and thus its safety at stake. In this work, we analyze the validity of several popular vehicle models used in the literature with respect to a real vehicle and we prove that serious accuracy issues are encountered beyond a specific lateral acceleration point. We set a clear lateral acceleration domain in which the used models are an accurate representation of the behavior of the vehicle. We then target the necessity of using learned methods to model the vehicle's behavior. The effects of model validity on state observers are investigated. The performance of model-based observers is compared to learning-based ones. Overall, the presented work emphasizes the validity of vehicle models and presents clear operational domains in which models could be used safely. 

**Abstract (ZH)**: 模型的有效性是自动驾驶车辆准确和安全行为的关键。使用无效的车辆模型会影响不同的规划和控制框架的稳定性，从而影响其安全性。在本工作中，我们分析了几种在文献中使用的流行车辆模型与实际车辆的一致性，并证明在特定侧向加速度点之后会出现严重的准确度问题。我们设定了一个明确的侧向加速度域，在此域内所使用的模型能准确地反映车辆的行为。然后我们强调了使用学习方法来建模车辆行为的必要性。研究了模型有效性对状态观测器的影响。比较了基于模型的观测器与基于学习的观测器的性能。总体而言，本工作强调了车辆模型的有效性，并提出了模型可以安全使用的明确操作域。 

---
# Motion Planning of Nonholonomic Cooperative Mobile Manipulators 

**Title (ZH)**: 非holonomic协同移动 manipulator 的运动规划 

**Authors**: Keshab Patra, Arpita Sinha, Anirban Guha  

**Link**: [PDF](https://arxiv.org/pdf/2502.05462)  

**Abstract**: We propose a real-time implementable motion planning technique for cooperative object transportation by nonholonomic mobile manipulator robots (MMRs) in an environment with static and dynamic obstacles. The proposed motion planning technique works in two steps. A novel visibility vertices-based path planning algorithm computes a global piece-wise linear path between the start and the goal location in the presence of static obstacles offline. It defines the static obstacle free space around the path with a set of convex polygons for the online motion planner. We employ a Nonliner Model Predictive Control (NMPC) based online motion planning technique for nonholonomic MMRs that jointly plans for the mobile base and the manipulators arm. It efficiently utilizes the locomotion capability of the mobile base and the manipulation capability of the arm. The motion planner plans feasible motion for the MMRs and generates trajectory for object transportation considering the kinodynamic constraints and the static and dynamic obstacles. The efficiency of our approach is validated by numerical simulation and hardware experiments in varied environments. 

**Abstract (ZH)**: 一种面向非完整移动机械手机器人（MMRs）的具有静态和动态障碍物环境下的实时可实现的合作物体运输运动规划技术 

---
# Temporal Representation Alignment: Successor Features Enable Emergent Compositionality in Robot Instruction Following Temporal Representation Alignment 

**Title (ZH)**: 时间表表示对齐：后续特征使机器人指令跟随能力具备 emergent 组合性 

**Authors**: Vivek Myers, Bill Chunyuan Zheng, Anca Dragan, Kuan Fang, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2502.05454)  

**Abstract**: Effective task representations should facilitate compositionality, such that after learning a variety of basic tasks, an agent can perform compound tasks consisting of multiple steps simply by composing the representations of the constituent steps together. While this is conceptually simple and appealing, it is not clear how to automatically learn representations that enable this sort of compositionality. We show that learning to associate the representations of current and future states with a temporal alignment loss can improve compositional generalization, even in the absence of any explicit subtask planning or reinforcement learning. We evaluate our approach across diverse robotic manipulation tasks as well as in simulation, showing substantial improvements for tasks specified with either language or goal images. 

**Abstract (ZH)**: 有效的任务表示应该促进组合性，使得在学习了多种基本任务后，代理可以通过将构成步骤的表示组合起来，简单地执行由多个步骤组成的复合任务。虽然这一概念上很简单且具有吸引力，但不清楚如何自动学习支持这种组合性的表示。我们展示了通过与时间对齐损失关联当前状态和未来状态的表示可以改善组合性泛化，即使在没有任何显式的子任务规划或强化学习的情况下。我们在多种机器人操控任务以及仿真中评估了该方法，对于使用语言或目标图像指定的任务，均显示出显著的改进。 

---
# ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy 

**Title (ZH)**: ConRFT: 一种通过一致性策略进行VLA模型强化微调的方法 

**Authors**: Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.05450)  

**Abstract**: Vision-Language-Action (VLA) models have shown substantial potential in real-world robotic manipulation. However, fine-tuning these models through supervised learning struggles to achieve robust performance due to limited, inconsistent demonstrations, especially in contact-rich environments. In this paper, we propose a reinforced fine-tuning approach for VLA models, named ConRFT, which consists of offline and online fine-tuning with a unified consistency-based training objective, to address these challenges. In the offline stage, our method integrates behavior cloning and Q-learning to effectively extract policy from a small set of demonstrations and stabilize value estimating. In the online stage, the VLA model is further fine-tuned via consistency policy, with human interventions to ensure safe exploration and high sample efficiency. We evaluate our approach on eight diverse real-world manipulation tasks. It achieves an average success rate of 96.3% within 45-90 minutes of online fine-tuning, outperforming prior supervised methods with a 144% improvement in success rate and 1.9x shorter episode length. This work highlights the potential of integrating reinforcement learning to enhance the performance of VLA models for real-world robotic applications. 

**Abstract (ZH)**: 基于强化学习的Vision-Language-Action (VLA) 模型增强微调方法：ConRFT及其在真实机器人操作任务中的应用 

---
# Non-cooperative Stochastic Target Encirclement by Anti-synchronization Control via Range-only Measurement 

**Title (ZH)**: 通过范围测量实现的非合作随机目标圈闭控制异步同步控制 

**Authors**: Fen Liu, Shenghai Yuan, Wei Meng, Rong Su, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.05440)  

**Abstract**: This paper investigates the stochastic moving target encirclement problem in a realistic setting. In contrast to typical assumptions in related works, the target in our work is non-cooperative and capable of escaping the circle containment by boosting its speed to maximum for a short duration. Considering the extreme environment, such as GPS denial, weight limit, and lack of ground guidance, two agents can only rely on their onboard single-modality perception tools to measure the distances to the target. The distance measurement allows for creating a position estimator by providing a target position-dependent variable. Furthermore, the construction of the unique distributed anti-synchronization controller (DASC) can guarantee that the two agents track and encircle the target swiftly. The convergence of the estimator and controller is rigorously evaluated using the Lyapunov technique. A real-world UAV-based experiment is conducted to illustrate the performance of the proposed methodology in addition to a simulated Matlab numerical sample. Our video demonstration can be found in the URL this https URL. 

**Abstract (ZH)**: 本文在实际场景中研究随机移动目标的包围问题。与相关工作中典型的假设不同，我们的目标是不合作的，并且能够在短时间内通过提高速度来突破包围圈。考虑到极端环境，如GPS拒绝服务、重量限制和缺乏地面引导，两个代理只能依赖其机载单一模态感知工具来测量到目标的距离。距离测量允许通过目标位置依赖变量创建位置估计器。此外，独特分布的反同步控制器（DASC）的构建可以保证两个代理能够迅速跟踪并包围目标。通过Lyapunov技术严格评估估计器和控制器的收敛性。除了MATLAB模拟数值样本外，我们还进行了基于UAV的实际实验以展示所提出方法的性能。我们的视频演示可以在以下链接找到：<https://this-is-the-url.com>。 

---
# Demonstrating CavePI: Autonomous Exploration of Underwater Caves by Semantic Guidance 

**Title (ZH)**: 展示CavePI：基于语义指导的水下洞穴自主探索 

**Authors**: Alankrit Gupta, Adnan Abdullah, Xianyao Li, Vaishnav Ramesh, Ioannis Rekleitis, Md Jahidul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2502.05384)  

**Abstract**: Enabling autonomous robots to safely and efficiently navigate, explore, and map underwater caves is of significant importance to water resource management, hydrogeology, archaeology, and marine robotics. In this work, we demonstrate the system design and algorithmic integration of a visual servoing framework for semantically guided autonomous underwater cave exploration. We present the hardware and edge-AI design considerations to deploy this framework on a novel AUV (Autonomous Underwater Vehicle) named CavePI. The guided navigation is driven by a computationally light yet robust deep visual perception module, delivering a rich semantic understanding of the environment. Subsequently, a robust control mechanism enables CavePI to track the semantic guides and navigate within complex cave structures. We evaluate the system through field experiments in natural underwater caves and spring-water sites and further validate its ROS (Robot Operating System)-based digital twin in a simulation environment. Our results highlight how these integrated design choices facilitate reliable navigation under feature-deprived, GPS-denied, and low-visibility conditions. 

**Abstract (ZH)**: 实现自主水下机器人在水下洞穴中安全高效地导航、探索和建图对于水资源管理、水文地质学、考古学和水下机器人技术具有重要意义。本文展示了基于语义引导的自主水下洞穴探索视觉伺服框架的系统设计与算法集成。我们介绍了将该框架部署在新型AUV（自主水下机器人）CavePI上的硬件和边缘AI设计考虑因素。由轻量级且可靠的深度视觉感知模块驱动的语义引导导航，提供了丰富的环境语义理解。接着，一个可靠的控制机制使CavePI能够跟踪语义引导并在复杂的洞穴结构中导航。我们通过在自然水下洞穴和温泉水区进行实地试验评估了该系统，并在仿真环境中验证了其基于ROS的数字孪生。我们的结果突显了在特征匮乏、GPS受限和低能见度条件下这些集成设计选择如何促进可靠导航。 

---
# Towards Wearable Interfaces for Robotic Caregiving 

**Title (ZH)**: 面向机器人护理的可穿戴界面研究 

**Authors**: Akhil Padmanabha, Carmel Majidi, Zackory Erickson  

**Link**: [PDF](https://arxiv.org/pdf/2502.05343)  

**Abstract**: Physically assistive robots in home environments can enhance the autonomy of individuals with impairments, allowing them to regain the ability to conduct self-care and household tasks. Individuals with physical limitations may find existing interfaces challenging to use, highlighting the need for novel interfaces that can effectively support them. In this work, we present insights on the design and evaluation of an active control wearable interface named HAT, Head-Worn Assistive Teleoperation. To tackle challenges in user workload while using such interfaces, we propose and evaluate a shared control algorithm named Driver Assistance. Finally, we introduce the concept of passive control, in which wearable interfaces detect implicit human signals to inform and guide robotic actions during caregiving tasks, with the aim of reducing user workload while potentially preserving the feeling of control. 

**Abstract (ZH)**: 家庭环境中具备物理辅助功能的机器人可以增强有身体障碍个体的自主性，使他们能够恢复自我照顾和家务活动的能力。身体有限制的个体可能发现现有接口难以使用，这凸显了需要有效支持他们的新型接口的需求。在此项工作中，我们介绍了名为HAT（Head-Worn Assistive Teleoperation）的主动控制可穿戴接口的设计与评估。为解决使用此类接口时的用户工作负荷问题，我们提出了并评估了名为Driver Assistance的联合控制算法。最后，我们引入了被动控制的概念，在这种概念下，可穿戴接口通过检测隐含的人类信号来告知和引导机器人动作，旨在降低用户工作负荷的同时，可能保留用户对系统的控制感。 

---
# Learning the Geometric Mechanics of Robot Motion Using Gaussian Mixtures 

**Title (ZH)**: 使用高斯混合模型学习机器人运动的几何力学 

**Authors**: Ruizhen Hu, Shai Revzen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05309)  

**Abstract**: Data-driven models of robot motion constructed using principles from Geometric Mechanics have been shown to produce useful predictions of robot motion for a variety of robots. For robots with a useful number of DoF, these geometric mechanics models can only be constructed in the neighborhood of a gait. Here we show how Gaussian Mixture Models (GMM) can be used as a form of manifold learning that learns the structure of the Geometric Mechanics "motility map" and demonstrate: [i] a sizable improvement in prediction quality when compared to the previously published methods; [ii] a method that can be applied to any motion dataset and not only periodic gait data; [iii] a way to pre-process the data-set to facilitate extrapolation in places where the motility map is known to be linear. Our results can be applied anywhere a data-driven geometric motion model might be useful. 

**Abstract (ZH)**: 使用几何力学原理构建的数据驱动的机器人运动模型已经在多种机器人上展示了有用的运动预测。对于具有实用自由度的机器人，这些几何力学模型只能在步态的邻域内构建。我们展示了高斯混合模型（GMM）可以作为一种流形学习方法来学习几何力学“运动图”的结构，并证明了以下几点：[i] 与先前发表的方法相比，预测质量有显著提高；[ii] 该方法可以应用于任何运动数据集，而不仅仅是周期步态数据；[iii] 一种数据预处理方法，以在已知运动图线性的地方促进外推。我们的结果可以应用于任何需要数据驱动几何运动模型的地方。 

---
# Switch-based Independent Antagonist Actuation with a Single Motor for a Soft Exosuit 

**Title (ZH)**: 基于开关的独立拮抗驱动软外骨骼单电机独立对抗驱动 

**Authors**: Atharva Vadeyar, Rejin John Varghese, Etienne Burdet, Dario Farina  

**Link**: [PDF](https://arxiv.org/pdf/2502.05290)  

**Abstract**: The use of a cable-driven soft exosuit poses challenges with regards to the mechanical design of the actuation system, particularly when used for actuation along multiple degrees of freedom (DoF). The simplest general solution requires the use of two actuators to be capable of inducing movement along one DoF. However, this solution is not practical for the development of multi-joint exosuits. Reducing the number of actuators is a critical need in multi-DoF exosuits. We propose a switch-based mechanism to control an antagonist pair of cables such that it can actuate along any cable path geometry. The results showed that 298.24ms was needed for switching between cables. While this latency is relatively large, it can reduced in the future by a better choice of the motor used for actuation. 

**Abstract (ZH)**: 基于电缆驱动的软外骨骼系统在多自由度 actuuator 设计中的挑战及解决方案 

---
# RobotMover: Learning to Move Large Objects by Imitating the Dynamic Chain 

**Title (ZH)**: RobotMover: 学习移动大型物体的动态链模拟方法 

**Authors**: Tianyu Li, Joanne Truong, Jimmy Yang, Alexander Clegg, Akshara Rai, Sehoon Ha, Xavier Puig  

**Link**: [PDF](https://arxiv.org/pdf/2502.05271)  

**Abstract**: Moving large objects, such as furniture, is a critical capability for robots operating in human environments. This task presents significant challenges due to two key factors: the need to synchronize whole-body movements to prevent collisions between the robot and the object, and the under-actuated dynamics arising from the substantial size and weight of the objects. These challenges also complicate performing these tasks via teleoperation. In this work, we introduce \method, a generalizable learning framework that leverages human-object interaction demonstrations to enable robots to perform large object manipulation tasks. Central to our approach is the Dynamic Chain, a novel representation that abstracts human-object interactions so that they can be retargeted to robotic morphologies. The Dynamic Chain is a spatial descriptor connecting the human and object root position via a chain of nodes, which encode the position and velocity of different interaction keypoints. We train policies in simulation using Dynamic-Chain-based imitation rewards and domain randomization, enabling zero-shot transfer to real-world settings without fine-tuning. Our approach outperforms both learning-based methods and teleoperation baselines across six evaluation metrics when tested on three distinct object types, both in simulation and on physical hardware. Furthermore, we successfully apply the learned policies to real-world tasks, such as moving a trash cart and rearranging chairs. 

**Abstract (ZH)**: 移动大型物体，例如家具，是机器人在人类环境中操作的一项关键能力。由于两个关键因素的存在，这项任务面临着巨大的挑战：需要同步全身运动以防止机器人与物体发生碰撞，以及由于物体的庞大尺寸和重量而产生的欠驱动动态。这些挑战也使得通过遥控操作执行这些任务变得更加复杂。在本工作中，我们引入了\method，这是一种泛化学习框架，利用人类-物体交互演示来使机器人能够执行大型物体操作任务。我们方法的核心是动态链，这是一种新颖的表现形式，可将人类-物体交互抽象化并重新定标到机器人的形态学上。动态链是一种空问描述符，通过一串节点连接人类和物体的根位置，这些节点编码了不同交互关节点的位置和速度。我们在仿真中使用基于动态链的模仿奖励和领域随机化进行策略训练，使策略能够在无需微调的情况下实现零样本迁移至真实环境。当分别在三种不同类型的物体上进行六项评估指标的测试时，我们的方法在仿真和物理硬件上均优于基于学习的方法和遥控操作基准方法。此外，我们成功将学到的策略应用于实际任务，如移动垃圾车和重新安排椅子。 

---
# Robotouille: An Asynchronous Planning Benchmark for LLM Agents 

**Title (ZH)**: Robotouille: 一个用于LLM代理的异步规划基准测试 

**Authors**: Gonzalo Gonzalez-Pumariega, Leong Su Yean, Neha Sunkara, Sanjiban Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2502.05227)  

**Abstract**: Effective asynchronous planning, or the ability to efficiently reason and plan over states and actions that must happen in parallel or sequentially, is essential for agents that must account for time delays, reason over diverse long-horizon tasks, and collaborate with other agents. While large language model (LLM) agents show promise in high-level task planning, current benchmarks focus primarily on short-horizon tasks and do not evaluate such asynchronous planning capabilities. We introduce Robotouille, a challenging benchmark environment designed to test LLM agents' ability to handle long-horizon asynchronous scenarios. Our synchronous and asynchronous datasets capture increasingly complex planning challenges that go beyond existing benchmarks, requiring agents to manage overlapping tasks and interruptions. Our results show that ReAct (gpt4-o) achieves 47% on synchronous tasks but only 11% on asynchronous tasks, highlighting significant room for improvement. We further analyze failure modes, demonstrating the need for LLM agents to better incorporate long-horizon feedback and self-audit their reasoning during task execution. Code is available at this https URL. 

**Abstract (ZH)**: 有效的异步规划能力，即高效地处理必须并行或顺序发生的状态和行动的能力，对于必须考虑时间延迟、处理多样化的长期任务以及与其他代理合作的智能体来说是必不可少的。虽然大型语言模型（LLM）代理在高层次任务规划方面显示出潜力，但当前的基准测试主要集中在短期任务上，并未评估这种异步规划能力。我们介绍了Robotouille，一个具有挑战性的基准环境，旨在测试LLM代理处理长期异步场景的能力。我们的同步和异步数据集捕捉到了超出现有基准的日益复杂的规划挑战，要求代理管理重叠任务和干扰。我们的结果显示，ReAct（gpt4-o）在同步任务中的得分为47%，但在异步任务中的得分仅为11%，强调了显著的改进空间。我们进一步分析了失败模式，展示了LLM代理需要更好地纳入长期反馈并在任务执行期间自我审核其推理的必要性。代码可在以下链接获取。 

---
# Rough Stochastic Pontryagin Maximum Principle and an Indirect Shooting Method 

**Title (ZH)**: 不连续随机庞特里亚金最大原理及间接射击方法 

**Authors**: Thomas Lew  

**Link**: [PDF](https://arxiv.org/pdf/2502.06726)  

**Abstract**: We derive first-order Pontryagin optimality conditions for stochastic optimal control with deterministic controls for systems modeled by rough differential equations (RDE) driven by Gaussian rough paths. This Pontryagin Maximum Principle (PMP) applies to systems following stochastic differential equations (SDE) driven by Brownian motion, yet it does not rely on forward-backward SDEs and involves the same Hamiltonian as the deterministic PMP. The proof consists of first deriving various integrable error bounds for solutions to nonlinear and linear RDEs by leveraging recent results on Gaussian rough paths. The PMP then follows using standard techniques based on needle-like variations. As an application, we propose the first indirect shooting method for nonlinear stochastic optimal control and show that it converges 10x faster than a direct method on a stabilization task. 

**Abstract (ZH)**: 我们推导了基于高斯粗糙道路驱动的粗糙微分方程（RDE）系统的随机最优控制的一阶庞特里亚金最优性条件。该庞特里亚金最大原理（PMP）适用于由布朗运动驱动的随机微分方程（SDE）系统，且不依赖于前向后向SDE，并且涉及与确定性PMP相同的哈密尔顿量。证明过程首先通过利用近期关于高斯粗糙道路的结果，推导出非线性和线性RDE解的各种可积误差界。随后使用基于细针变化的标准技术得出PMP。作为应用，我们提出了首个非线性随机最优控制的间接射击方法，并展示了其在稳定化任务上比直接方法快10倍的收敛速度。 

---
# An Automated Machine Learning Framework for Surgical Suturing Action Detection under Class Imbalance 

**Title (ZH)**: 面向类别不平衡的外科缝合动作检测的自动化机器学习框架 

**Authors**: Baobing Zhang, Paul Sullivan, Benjie Tang, Ghulam Nabi, Mustafa Suphi Erden  

**Link**: [PDF](https://arxiv.org/pdf/2502.06407)  

**Abstract**: In laparoscopy surgical training and evaluation, real-time detection of surgical actions with interpretable outputs is crucial for automated and real-time instructional feedback and skill development. Such capability would enable development of machine guided training systems. This paper presents a rapid deployment approach utilizing automated machine learning methods, based on surgical action data collected from both experienced and trainee surgeons. The proposed approach effectively tackles the challenge of highly imbalanced class distributions, ensuring robust predictions across varying skill levels of surgeons. Additionally, our method partially incorporates model transparency, addressing the reliability requirements in medical applications. Compared to deep learning approaches, traditional machine learning models not only facilitate efficient rapid deployment but also offer significant advantages in interpretability. Through experiments, this study demonstrates the potential of this approach to provide quick, reliable and effective real-time detection in surgical training environments 

**Abstract (ZH)**: 在腹腔镜手术培训与评估中，实时检测手术动作并产生可解释的输出对于自动化和实时教学反馈及技能发展至关重要。这种能力将使指导性训练系统的开发成为可能。本文提出了一种基于自动机器学习方法的快速部署方法，利用从经验丰富的和实习外科医生处收集的手术动作数据。所提出的方法有效解决了类分布高度不平衡的挑战，确保了在不同水平外科医生的预测具有鲁棒性。此外，该方法部分实现了模型透明性，以应对医疗应用中的可靠性要求。与深度学习方法相比，传统机器学习模型不仅促进了高效快速部署，还提供了显著的可解释性优势。通过实验，本文证明了该方法在手术训练环境中提供快速、可靠和有效的实时检测潜力。 

---
# Accelerating Outlier-robust Rotation Estimation by Stereographic Projection 

**Title (ZH)**: 基于立体投影加速抗离群点旋转估计 

**Authors**: Taosi Xu, Yinlong Liu, Xianbo Wang, Zhi-Xin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06337)  

**Abstract**: Rotation estimation plays a fundamental role in many computer vision and robot tasks. However, efficiently estimating rotation in large inputs containing numerous outliers (i.e., mismatches) and noise is a recognized challenge. Many robust rotation estimation methods have been designed to address this challenge. Unfortunately, existing methods are often inapplicable due to their long computation time and the risk of local optima. In this paper, we propose an efficient and robust rotation estimation method. Specifically, our method first investigates geometric constraints involving only the rotation axis. Then, it uses stereographic projection and spatial voting techniques to identify the rotation axis and angle. Furthermore, our method efficiently obtains the optimal rotation estimation and can estimate multiple rotations simultaneously. To verify the feasibility of our method, we conduct comparative experiments using both synthetic and real-world data. The results show that, with GPU assistance, our method can solve large-scale ($10^6$ points) and severely corrupted (90\% outlier rate) rotation estimation problems within 0.07 seconds, with an angular error of only 0.01 degrees, which is superior to existing methods in terms of accuracy and efficiency. 

**Abstract (ZH)**: 旋转估计在许多计算机视觉和机器人任务中起着基础性作用。然而，在包含大量离群点（即错误匹配）和噪声的大输入中高效地估计旋转是一个公认的挑战。许多稳健的旋转估计方法已经被设计出来以应对这一挑战。不幸的是，现有方法往往由于计算时间过长和容易陷入局部最优而不可用。在本文中，我们提出了一种高效且稳健的旋转估计方法。具体而言，该方法首先探讨仅涉及旋转轴的几何约束。然后，它使用立体投影和空间投票技术来识别旋转轴和角度。此外，该方法能有效地获得最佳旋转估计，并能同时估计多个旋转。为了验证该方法的可行性，我们在合成和真实数据上进行了对比实验。结果显示，在GPU辅助下，该方法可以在0.07秒内解决包含100万点的大规模和严重污染（90%离群点比率）的旋转估计问题，并且角度误差仅为0.01度，无论在准确性和效率方面都优于现有方法。 

---
# Calibration of Multiple Asynchronous Microphone Arrays using Hybrid TDOA 

**Title (ZH)**: 异步麦克风阵列基于混合TDOA的校准 

**Authors**: Chengjie Zhang, Wenda Pan, Xinyang Han, He Kong  

**Link**: [PDF](https://arxiv.org/pdf/2502.06195)  

**Abstract**: Accurate calibration of acoustic sensing systems made of multiple asynchronous microphone arrays is essential for satisfactory performance in sound source localization and tracking. State-of-the-art calibration methods for this type of system rely on the time difference of arrival and direction of arrival measurements among the microphone arrays (denoted as TDOA-M and DOA, respectively). In this paper, to enhance calibration accuracy, we propose to incorporate the time difference of arrival measurements between adjacent sound events (TDOAS) with respect to the microphone arrays. More specifically, we propose a two-stage calibration approach, including an initial value estimation (IVE) procedure and the final joint optimization step. The IVE stage first initializes all parameters except for microphone array orientations, using hybrid TDOA (i.e., TDOAM and TDOA-S), odometer data from a moving robot carrying a speaker, and DOA. Subsequently, microphone orientations are estimated through the iterative closest point method. The final joint optimization step estimates multiple microphone array locations, orientations, time offsets, clock drift rates, and sound source locations simultaneously. Both simulation and experiment results show that for scenarios with low or moderate TDOA noise levels, our approach outperforms existing methods in terms of accuracy. All code and data are available at this https URL. 

**Abstract (ZH)**: 多异步麦克风阵列的声学传感系统精确校准对于声音源定位和跟踪的满意性能至关重要。本文为提高校准精度，提出了一种结合相邻声事件到达时间差测量（TDOAS）的两阶段校准方法，包括初步值估计（IVE）程序和最终联合优化步骤。 

---
# Redefining Robot Generalization Through Interactive Intelligence 

**Title (ZH)**: 通过交互智能重新定义机器人泛化能力 

**Authors**: Sharmita Dey  

**Link**: [PDF](https://arxiv.org/pdf/2502.05963)  

**Abstract**: Recent advances in large-scale machine learning have produced high-capacity foundation models capable of adapting to a broad array of downstream tasks. While such models hold great promise for robotics, the prevailing paradigm still portrays robots as single, autonomous decision-makers, performing tasks like manipulation and navigation, with limited human involvement. However, a large class of real-world robotic systems, including wearable robotics (e.g., prostheses, orthoses, exoskeletons), teleoperation, and neural interfaces, are semiautonomous, and require ongoing interactive coordination with human partners, challenging single-agent assumptions. In this position paper, we argue that robot foundation models must evolve to an interactive multi-agent perspective in order to handle the complexities of real-time human-robot co-adaptation. We propose a generalizable, neuroscience-inspired architecture encompassing four modules: (1) a multimodal sensing module informed by sensorimotor integration principles, (2) an ad-hoc teamwork model reminiscent of joint-action frameworks in cognitive science, (3) a predictive world belief model grounded in internal model theories of motor control, and (4) a memory/feedback mechanism that echoes concepts of Hebbian and reinforcement-based plasticity. Although illustrated through the lens of cyborg systems, where wearable devices and human physiology are inseparably intertwined, the proposed framework is broadly applicable to robots operating in semi-autonomous or interactive contexts. By moving beyond single-agent designs, our position emphasizes how foundation models in robotics can achieve a more robust, personalized, and anticipatory level of performance. 

**Abstract (ZH)**: 近期大规模机器学习的进展产生了高容量的基础模型，能够适应广泛的下游任务。尽管这类模型在机器人领域具有巨大潜力，现有的主流 paradigm仍视机器人为主动的单个决策者，进行操作和导航等任务，且人类的介入有限。然而，包括可穿戴机器人（例如假肢、矫形器、外骨骼）、遥控操作和神经接口在内的许多实际机器人系统是半自主的，需要与人类伙伴持续互动协调，这挑战了单智能体假设。在本文中，我们argue机器人基础模型必须进化到互动多智能体的视角，以处理实时人机共适应的复杂性。我们提出了一种通用的、受神经科学启发的架构，包含四个模块：（1）由感觉运动整合原理指导的多模态传感模块，（2）借鉴认知科学中联合行动框架的一种临时性团队模型，（3）扎根于运动控制内部模型理论的预测世界信念模块，以及（4）一种回声海氏和强化学习可塑性的记忆/反馈机制。尽管通过半机械人系统这一视角进行说明，其中穿戴设备和人类生理不可分割地交织在一起，所提出的方法框架在半自主或互动性操作的机器人中具有广泛的适用性。通过超越单智能体设计，我们的立场强调了在机器人领域，基础模型可以实现更加稳健、个性化和前瞻性水平的表现。 

---
# Skill Expansion and Composition in Parameter Space 

**Title (ZH)**: 参数空间中的技能扩展与组合 

**Authors**: Tenglong Liu, Jianxiong Li, Yinan Zheng, Haoyi Niu, Yixing Lan, Xin Xu, Xianyuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2502.05932)  

**Abstract**: Humans excel at reusing prior knowledge to address new challenges and developing skills while solving problems. This paradigm becomes increasingly popular in the development of autonomous agents, as it develops systems that can self-evolve in response to new challenges like human beings. However, previous methods suffer from limited training efficiency when expanding new skills and fail to fully leverage prior knowledge to facilitate new task learning. In this paper, we propose Parametric Skill Expansion and Composition (PSEC), a new framework designed to iteratively evolve the agents' capabilities and efficiently address new challenges by maintaining a manageable skill library. This library can progressively integrate skill primitives as plug-and-play Low-Rank Adaptation (LoRA) modules in parameter-efficient finetuning, facilitating efficient and flexible skill expansion. This structure also enables the direct skill compositions in parameter space by merging LoRA modules that encode different skills, leveraging shared information across skills to effectively program new skills. Based on this, we propose a context-aware module to dynamically activate different skills to collaboratively handle new tasks. Empowering diverse applications including multi-objective composition, dynamics shift, and continual policy shift, the results on D4RL, DSRL benchmarks, and the DeepMind Control Suite show that PSEC exhibits superior capacity to leverage prior knowledge to efficiently tackle new challenges, as well as expand its skill libraries to evolve the capabilities. Project website: this https URL. 

**Abstract (ZH)**: 人类擅长利用先验知识应对新挑战并发展解决问题所需的新技能。这一 paradigm 在自主代理系统的开发中越来越受欢迎，因为它可以开发出能够自我进化以应对新挑战的系统，类似于人类的做法。然而，先前的方法在扩展新技能时训练效率有限，并且未能充分利用先验知识来促进新任务的学习。本文提出了参数化技能扩展与组合（PSEC）框架，旨在通过维护一个可管理的技能库来逐步进化代理的技能，从而有效应对新挑战。该库可以通过参数高效微调逐步集成作为插件式低秩适应（LoRA）模块的技能原语，从而促进高效的技能扩展。此外，该结构还允许在参数空间中直接组合技能，通过合并编码不同技能的LoRA模块，利用技能间的共享信息来有效编程新技能。基于此，我们提出了一个上下文感知模块，以动态激活不同技能，协同处理新任务。PSEC在D4RL、DSRL基准和DeepMind控制套件上的实验结果表明，它具有更强的能力，可以利用先验知识高效应对新挑战，扩展技能库以进化能力。项目网站：https://this-link-is-intended-to-be-inserted-by-the-author. 

---
# Kalman Filter-Based Distributed Gaussian Process for Unknown Scalar Field Estimation in Wireless Sensor Networks 

**Title (ZH)**: 基于卡尔曼滤波的分布式高斯过程未知标量场估计在无线传感器网络中 

**Authors**: Jaemin Seo, Geunsik Bae, Hyondong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2502.05802)  

**Abstract**: In this letter, we propose an online scalar field estimation algorithm of unknown environments using a distributed Gaussian process (DGP) framework in wireless sensor networks (WSNs). While the kernel-based Gaussian process (GP) has been widely employed for estimating unknown scalar fields, its centralized nature is not well-suited for handling a large amount of data from WSNs. To overcome the limitations of the kernel-based GP, recent advancements in GP research focus on approximating kernel functions as products of E-dimensional nonlinear basis functions, which can handle large WSNs more efficiently in a distributed manner. However, this approach requires a large number of basis functions for accurate approximation, leading to increased computational and communication complexities. To address these complexity issues, the paper proposes a distributed GP framework by incorporating a Kalman filter scheme (termed as K-DGP), which scales linearly with the number of nonlinear basis functions. Moreover, we propose a new consensus protocol designed to handle the unique data transmission requirement residing in the proposed K-DGP framework. This protocol preserves the inherent elements in the form of a certain column in the nonlinear function matrix of the communicated message; it enables wireless sensors to cooperatively estimate the environment and reach the global consensus through distributed learning with faster convergence than the widely-used average consensus protocol. Simulation results demonstrate rapid consensus convergence and outstanding estimation accuracy achieved by the proposed K-DGP algorithm. The scalability and efficiency of the proposed approach are further demonstrated by online dynamic environment estimation using WSNs. 

**Abstract (ZH)**: 基于分布式高斯过程的无线传感器网络中未知环境在线标量场估计算法 

---
# Low-Rank Agent-Specific Adaptation (LoRASA) for Multi-Agent Policy Learning 

**Title (ZH)**: 低秩个体特定适应（LoRASA）多智能体策略学习 

**Authors**: Beining Zhang, Aditya Kapoor, Mingfei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.05573)  

**Abstract**: Multi-agent reinforcement learning (MARL) often relies on \emph{parameter sharing (PS)} to scale efficiently. However, purely shared policies can stifle each agent's unique specialization, reducing overall performance in heterogeneous environments. We propose \textbf{Low-Rank Agent-Specific Adaptation (LoRASA)}, a novel approach that treats each agent's policy as a specialized ``task'' fine-tuned from a shared backbone. Drawing inspiration from parameter-efficient transfer methods, LoRASA appends small, low-rank adaptation matrices to each layer of the shared policy, naturally inducing \emph{parameter-space sparsity} that promotes both specialization and scalability. We evaluate LoRASA on challenging benchmarks including the StarCraft Multi-Agent Challenge (SMAC) and Multi-Agent MuJoCo (MAMuJoCo), implementing it atop widely used algorithms such as MAPPO and A2PO. Across diverse tasks, LoRASA matches or outperforms existing baselines \emph{while reducing memory and computational overhead}. Ablation studies on adapter rank, placement, and timing validate the method's flexibility and efficiency. Our results suggest LoRASA's potential to establish a new norm for MARL policy parameterization: combining a shared foundation for coordination with low-rank agent-specific refinements for individual specialization. 

**Abstract (ZH)**: 低秩代理特定适应（LoRASA）：一种新颖的多代理 reinforcement 学习方法 

---
# Vision-in-the-loop Simulation for Deep Monocular Pose Estimation of UAV in Ocean Environment 

**Title (ZH)**: 基于视觉的环路仿真方法在海洋环境中无人机单目姿态估计中应用 

**Authors**: Maneesha Wickramasuriya, Beomyeol Yu, Taeyoung Lee, Murray Snyder  

**Link**: [PDF](https://arxiv.org/pdf/2502.05409)  

**Abstract**: This paper proposes a vision-in-the-loop simulation environment for deep monocular pose estimation of a UAV operating in an ocean environment. Recently, a deep neural network with a transformer architecture has been successfully trained to estimate the pose of a UAV relative to the flight deck of a research vessel, overcoming several limitations of GPS-based approaches. However, validating the deep pose estimation scheme in an actual ocean environment poses significant challenges due to the limited availability of research vessels and the associated operational costs. To address these issues, we present a photo-realistic 3D virtual environment leveraging recent advancements in Gaussian splatting, a novel technique that represents 3D scenes by modeling image pixels as Gaussian distributions in 3D space, creating a lightweight and high-quality visual model from multiple viewpoints. This approach enables the creation of a virtual environment integrating multiple real-world images collected in situ. The resulting simulation enables the indoor testing of flight maneuvers while verifying all aspects of flight software, hardware, and the deep monocular pose estimation scheme. This approach provides a cost-effective solution for testing and validating the autonomous flight of shipboard UAVs, specifically focusing on vision-based control and estimation algorithms. 

**Abstract (ZH)**: 基于视觉的无人机海洋环境单目姿态估计视景环模拟环境 

---
# NextBestPath: Efficient 3D Mapping of Unseen Environments 

**Title (ZH)**: NextBestPath: 效率高的未见环境三维建图 

**Authors**: Shiyao Li, Antoine Guédon, Clémentin Boittiaux, Shizhe Chen, Vincent Lepetit  

**Link**: [PDF](https://arxiv.org/pdf/2502.05378)  

**Abstract**: This work addresses the problem of active 3D mapping, where an agent must find an efficient trajectory to exhaustively reconstruct a new scene. Previous approaches mainly predict the next best view near the agent's location, which is prone to getting stuck in local areas. Additionally, existing indoor datasets are insufficient due to limited geometric complexity and inaccurate ground truth meshes. To overcome these limitations, we introduce a novel dataset AiMDoom with a map generator for the Doom video game, enabling to better benchmark active 3D mapping in diverse indoor environments. Moreover, we propose a new method we call next-best-path (NBP), which predicts long-term goals rather than focusing solely on short-sighted views. The model jointly predicts accumulated surface coverage gains for long-term goals and obstacle maps, allowing it to efficiently plan optimal paths with a unified model. By leveraging online data collection, data augmentation and curriculum learning, NBP significantly outperforms state-of-the-art methods on both the existing MP3D dataset and our AiMDoom dataset, achieving more efficient mapping in indoor environments of varying complexity. 

**Abstract (ZH)**: 这种工作解决了主动3D建图的问题，其中代理剂必须找到一条高效的路径来全面重建一个新的场景。此前的方法主要预测代理剂位置附近的最优视角，容易陷入局部区域。此外，现有的室内数据集由于几何复杂度有限且ground truth网格不准确而不足。为克服这些限制，我们引入了一个名为AiMDoom的新数据集及其映射生成器，用于Doom视频游戏，以更好地在多样化的室内环境中基准测试主动3D建图。此外，我们提出了一种新的方法，称为最优路径（Next-Best-Path，NBP），该方法预测长期目标而非仅为短视视角。该模型联合预测长期目标的累积表面覆盖增益和障碍物地图，使其能够使用统一模型高效地规划最优路径。通过利用在线数据采集、数据增强和级联学习，NBP在现有的MP3D数据集和我们的AiMDoom数据集上均显著优于现有方法，在不同复杂度的室内环境中实现了更高效的建图。 

---
