# Dex1B: Learning with 1B Demonstrations for Dexterous Manipulation 

**Title (ZH)**: Dex1B: 使用1亿示范样本进行灵巧操作学习 

**Authors**: Jianglong Ye, Keyi Wang, Chengjing Yuan, Ruihan Yang, Yiquan Li, Jiyue Zhu, Yuzhe Qin, Xueyan Zou, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17198)  

**Abstract**: Generating large-scale demonstrations for dexterous hand manipulation remains challenging, and several approaches have been proposed in recent years to address this. Among them, generative models have emerged as a promising paradigm, enabling the efficient creation of diverse and physically plausible demonstrations. In this paper, we introduce Dex1B, a large-scale, diverse, and high-quality demonstration dataset produced with generative models. The dataset contains one billion demonstrations for two fundamental tasks: grasping and articulation. To construct it, we propose a generative model that integrates geometric constraints to improve feasibility and applies additional conditions to enhance diversity. We validate the model on both established and newly introduced simulation benchmarks, where it significantly outperforms prior state-of-the-art methods. Furthermore, we demonstrate its effectiveness and robustness through real-world robot experiments. Our project page is at this https URL 

**Abstract (ZH)**: 生成灵巧手操作的大规模演示仍然具有挑战性，近年来提出了一些方法来应对这一挑战。其中，生成模型作为一种有前景的范式 emerged as a promising paradigm，使得高效创建多样且物理上合理的演示成为可能。在本文中，我们介绍了 Dex1B，这是一个使用生成模型构建的大规模、多样且高质量的演示数据集。该数据集包含了一百亿个演示，用于两个基础任务：抓取和动作。为构建该数据集，我们提出了一种生成模型，该模型整合了几何约束以提高可行性，并应用额外条件以增强多样性。我们在多个常用和新引入的仿真基准上验证了该模型，其性能显著优于先前的最佳方法。此外，我们通过实际的机器人实验展示了其有效性和鲁棒性。项目页面链接为：这个 https URL。 

---
# Judo: A User-Friendly Open-Source Package for Sampling-Based Model Predictive Control 

**Title (ZH)**: judo：一个用户友好的基于采样的模型预测控制开源软件包 

**Authors**: Albert H. Li, Brandon Hung, Aaron D. Ames, Jiuguang Wang, Simon Le Cleac'h, Preston Culbertson  

**Link**: [PDF](https://arxiv.org/pdf/2506.17184)  

**Abstract**: Recent advancements in parallel simulation and successful robotic applications are spurring a resurgence in sampling-based model predictive control. To build on this progress, however, the robotics community needs common tooling for prototyping, evaluating, and deploying sampling-based controllers. We introduce Judo, a software package designed to address this need. To facilitate rapid prototyping and evaluation, Judo provides robust implementations of common sampling-based MPC algorithms and standardized benchmark tasks. It further emphasizes usability with simple but extensible interfaces for controller and task definitions, asynchronous execution for straightforward simulation-to-hardware transfer, and a highly customizable interactive GUI for tuning controllers interactively. While written in Python, the software leverages MuJoCo as its physics backend to achieve real-time performance, which we validate across both consumer and server-grade hardware. Code at this https URL. 

**Abstract (ZH)**: 最近并行仿真技术的进展及其在机器人领域的成功应用正推动基于采样模型预测控制的复兴。为了在此基础上进一步发展，机器人社区需要一套通用工具来原型设计、评估和部署基于采样模型预测控制器。为此，我们引入了Judo，一个旨在解决这一需求的软件包。为了促进快速原型设计和评估，Judo 提供了常见的基于采样模型预测控制算法的 robust 实现和标准化基准任务。它还强调易用性，通过简单但可扩展的控制器和任务定义接口、异步执行以实现仿真到硬件的简便转移，以及高度可自定义的交互式 GUI 来交互调整控制器。虽然用 Python 编写，但该软件利用 MuJoCo 作为物理后端以实现实时性能，并已在消费级和服务器级硬件上进行了验证。相关代码请参见此 https URL。 

---
# Monocular One-Shot Metric-Depth Alignment for RGB-Based Robot Grasping 

**Title (ZH)**: 基于RGB的一次成像度量深度对齐方法用于机器人抓取 

**Authors**: Teng Guo, Baichuan Huang, Jingjin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17110)  

**Abstract**: Accurate 6D object pose estimation is a prerequisite for successfully completing robotic prehensile and non-prehensile manipulation tasks. At present, 6D pose estimation for robotic manipulation generally relies on depth sensors based on, e.g., structured light, time-of-flight, and stereo-vision, which can be expensive, produce noisy output (as compared with RGB cameras), and fail to handle transparent objects. On the other hand, state-of-the-art monocular depth estimation models (MDEMs) provide only affine-invariant depths up to an unknown scale and shift. Metric MDEMs achieve some successful zero-shot results on public datasets, but fail to generalize. We propose a novel framework, Monocular One-shot Metric-depth Alignment (MOMA), to recover metric depth from a single RGB image, through a one-shot adaptation building on MDEM techniques. MOMA performs scale-rotation-shift alignments during camera calibration, guided by sparse ground-truth depth points, enabling accurate depth estimation without additional data collection or model retraining on the testing setup. MOMA supports fine-tuning the MDEM on transparent objects, demonstrating strong generalization capabilities. Real-world experiments on tabletop 2-finger grasping and suction-based bin-picking applications show MOMA achieves high success rates in diverse tasks, confirming its effectiveness. 

**Abstract (ZH)**: 单目一次-shot métric深度对齐（MOMA）：从单张RGB图像恢复 métric深度 

---
# Learning Accurate Whole-body Throwing with High-frequency Residual Policy and Pullback Tube Acceleration 

**Title (ZH)**: 基于高频残差策略和拉回管加速的学习精确全身投掷方法 

**Authors**: Yuntao Ma, Yang Liu, Kaixian Qu, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2506.16986)  

**Abstract**: Throwing is a fundamental skill that enables robots to manipulate objects in ways that extend beyond the reach of their arms. We present a control framework that combines learning and model-based control for prehensile whole-body throwing with legged mobile manipulators. Our framework consists of three components: a nominal tracking policy for the end-effector, a high-frequency residual policy to enhance tracking accuracy, and an optimization-based module to improve end-effector acceleration control. The proposed controller achieved the average of 0.28 m landing error when throwing at targets located 6 m away. Furthermore, in a comparative study with university students, the system achieved a velocity tracking error of 0.398 m/s and a success rate of 56.8%, hitting small targets randomly placed at distances of 3-5 m while throwing at a specified speed of 6 m/s. In contrast, humans have a success rate of only 15.2%. This work provides an early demonstration of prehensile throwing with quantified accuracy on hardware, contributing to progress in dynamic whole-body manipulation. 

**Abstract (ZH)**: 基于学习与模型导向控制的腿式操作机器人预抓握全身投掷控制框架 

---
# SDDiff: Boost Radar Perception via Spatial-Doppler Diffusion 

**Title (ZH)**: SDDiff: 通过空间-多普勒扩散增强雷达感知 

**Authors**: Shengpeng Wang, Xin Luo, Yulong Xie, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16936)  

**Abstract**: Point cloud extraction (PCE) and ego velocity estimation (EVE) are key capabilities gaining attention in 3D radar perception. However, existing work typically treats these two tasks independently, which may neglect the interplay between radar's spatial and Doppler domain features, potentially introducing additional bias. In this paper, we observe an underlying correlation between 3D points and ego velocity, which offers reciprocal benefits for PCE and EVE. To fully unlock such inspiring potential, we take the first step to design a Spatial-Doppler Diffusion (SDDiff) model for simultaneously dense PCE and accurate EVE. To seamlessly tailor it to radar perception, SDDiff improves the conventional latent diffusion process in three major aspects. First, we introduce a representation that embodies both spatial occupancy and Doppler features. Second, we design a directional diffusion with radar priors to streamline the sampling. Third, we propose Iterative Doppler Refinement to enhance the model's adaptability to density variations and ghosting effects. Extensive evaluations show that SDDiff significantly outperforms state-of-the-art baselines by achieving 59% higher in EVE accuracy, 4X greater in valid generation density while boosting PCE effectiveness and reliability. 

**Abstract (ZH)**: 基于空时扩散的点云提取与 ego 速度估计 

---
# Orbital Collision: An Indigenously Developed Web-based Space Situational Awareness Platform 

**Title (ZH)**: 轨道碰撞：一款本土开发的基于Web的空间态势感知平台 

**Authors**: Partha Chowdhury, Harsha M, Ayush Gupta, Sanat K Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2506.16892)  

**Abstract**: This work presents an indigenous web based platform Orbital Collision (OrCo), created by the Space Systems Laboratory at IIIT Delhi, to enhance Space Situational Awareness (SSA) by predicting collision probabilities of space objects using Two Line Elements (TLE) data. The work highlights the growing challenges of congestion in the Earth's orbital environment, mainly due to space debris and defunct satellites, which increase collision risks. It employs several methods for propagating orbital uncertainty and calculating the collision probability. The performance of the platform is evaluated through accuracy assessments and efficiency metrics, in order to improve the tracking of space objects and ensure the safety of the satellite in congested space. 

**Abstract (ZH)**: 本研究介绍由印度信息技术学院DELHI空间系统实验室创建的本民族Web基于平台Orbital Collision (OrCo)，通过使用Two Line Elements (TLE)数据预测空间物体的碰撞概率以增强太空态势感知（SSA）。该研究强调了地球轨道环境日益严重的拥堵问题，主要由于空间碎片和失效卫星增加了碰撞风险。平台采用了多种方法传播轨道不确定性并计算碰撞概率。通过准确性和效率指标评估平台性能，以提高对空间物体的跟踪并确保拥挤太空中卫星的安全。 

---
# Learning Dexterous Object Handover 

**Title (ZH)**: 学习灵巧的物体传递 

**Authors**: Daniel Frau-Alfaro, Julio Castaño-Amoros, Santiago Puente, Pablo Gil, Roberto Calandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.16822)  

**Abstract**: Object handover is an important skill that we use daily when interacting with other humans. To deploy robots in collaborative setting, like houses, being able to receive and handing over objects safely and efficiently becomes a crucial skill. In this work, we demonstrate the use of Reinforcement Learning (RL) for dexterous object handover between two multi-finger hands. Key to this task is the use of a novel reward function based on dual quaternions to minimize the rotation distance, which outperforms other rotation representations such as Euler and rotation matrices. The robustness of the trained policy is experimentally evaluated by testing w.r.t. objects that are not included in the training distribution, and perturbations during the handover process. The results demonstrate that the trained policy successfully perform this task, achieving a total success rate of 94% in the best-case scenario after 100 experiments, thereby showing the robustness of our policy with novel objects. In addition, the best-case performance of the policy decreases by only 13.8% when the other robot moves during the handover, proving that our policy is also robust to this type of perturbation, which is common in real-world object handovers. 

**Abstract (ZH)**: 基于鲁棒性的双四元数强化学习在多指手之间进行精细物体交接的研究 

---
# A Scalable Post-Processing Pipeline for Large-Scale Free-Space Multi-Agent Path Planning with PiBT 

**Title (ZH)**: 一种用于大规模自由空间多agent路径规划的可扩展后处理管道PiBT 

**Authors**: Arjo Chakravarty, Michael X. Grey, M. A. Viraj J. Muthugala, Mohan Rajesh Elara  

**Link**: [PDF](https://arxiv.org/pdf/2506.16748)  

**Abstract**: Free-space multi-agent path planning remains challenging at large scales. Most existing methods either offer optimality guarantees but do not scale beyond a few dozen agents, or rely on grid-world assumptions that do not generalize well to continuous space. In this work, we propose a hybrid, rule-based planning framework that combines Priority Inheritance with Backtracking (PiBT) with a novel safety-aware path smoothing method. Our approach extends PiBT to 8-connected grids and selectively applies string-pulling based smoothing while preserving collision safety through local interaction awareness and a fallback collision resolution step based on Safe Interval Path Planning (SIPP). This design allows us to reduce overall path lengths while maintaining real-time performance. We demonstrate that our method can scale to over 500 agents in large free-space environments, outperforming existing any-angle and optimal methods in terms of runtime, while producing near-optimal trajectories in sparse domains. Our results suggest this framework is a promising building block for scalable, real-time multi-agent navigation in robotics systems operating beyond grid constraints. 

**Abstract (ZH)**: 自由空间多 Agents 路径规划在大规模场景下仍具挑战性。大多数现有方法要么提供了最优性保证但不能扩展到数十个以上的 Agents，要么依赖于网格世界的假设，不能很好地泛化到连续空间。在本文中，我们提出了一种基于优先级继承与回溯（PiBT）的混合规则规划框架，并结合了一种新型的安全意识路径平滑方法。我们的方法将 PiBT 扩展到 8 连通网格上，并在保持碰撞安全的前提下，通过局部交互意识和基于 Safe Interval Path Planning (SIPP) 的回退碰撞解决步骤，选择性地应用字符串拉伸基于的平滑方法。这种设计使我们在保持实时性能的同时，减少整体路径长度。我们证明，我们的方法可以扩展到超过 500 个 Agents 的大型自由空间环境，在运行时性能方面优于现有的任意角度和最优方法，同时在稀疏领域生成接近最优的轨迹。实验结果表明，该框架是构建适用于超越网格约束的大规模、实时多 Agents 导航的有前景的基本组件。 

---
# DRARL: Disengagement-Reason-Augmented Reinforcement Learning for Efficient Improvement of Autonomous Driving Policy 

**Title (ZH)**: DRARL: 回退原因增强的 reinforcement 学习以提高自主驾驶策略效率 

**Authors**: Weitao Zhou, Bo Zhang, Zhong Cao, Xiang Li, Qian Cheng, Chunyang Liu, Yaqin Zhang, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16720)  

**Abstract**: With the increasing presence of automated vehicles on open roads under driver supervision, disengagement cases are becoming more prevalent. While some data-driven planning systems attempt to directly utilize these disengagement cases for policy improvement, the inherent scarcity of disengagement data (often occurring as a single instances) restricts training effectiveness. Furthermore, some disengagement data should be excluded since the disengagement may not always come from the failure of driving policies, e.g. the driver may casually intervene for a while. To this end, this work proposes disengagement-reason-augmented reinforcement learning (DRARL), which enhances driving policy improvement process according to the reason of disengagement cases. Specifically, the reason of disengagement is identified by a out-of-distribution (OOD) state estimation model. When the reason doesn't exist, the case will be identified as a casual disengagement case, which doesn't require additional policy adjustment. Otherwise, the policy can be updated under a reason-augmented imagination environment, improving the policy performance of disengagement cases with similar reasons. The method is evaluated using real-world disengagement cases collected by autonomous driving robotaxi. Experimental results demonstrate that the method accurately identifies policy-related disengagement reasons, allowing the agent to handle both original and semantically similar cases through reason-augmented training. Furthermore, the approach prevents the agent from becoming overly conservative after policy adjustments. Overall, this work provides an efficient way to improve driving policy performance with disengagement cases. 

**Abstract (ZH)**: 基于脱离原因增强的强化学习方法（DRARL）以提升驾驶策略性能 

---
# Experimental Setup and Software Pipeline to Evaluate Optimization based Autonomous Multi-Robot Search Algorithms 

**Title (ZH)**: 基于优化的自主多机器人搜索算法评估的实验设置与软件管道 

**Authors**: Aditya Bhatt, Mary Katherine Corra, Franklin Merlo, Prajit KrisshnaKumar, Souma Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2506.16710)  

**Abstract**: Signal source localization has been a problem of interest in the multi-robot systems domain given its applications in search \& rescue and hazard localization in various industrial and outdoor settings. A variety of multi-robot search algorithms exist that usually formulate and solve the associated autonomous motion planning problem as a heuristic model-free or belief model-based optimization process. Most of these algorithms however remains tested only in simulation, thereby losing the opportunity to generate knowledge about how such algorithms would compare/contrast in a real physical setting in terms of search performance and real-time computing performance. To address this gap, this paper presents a new lab-scale physical setup and associated open-source software pipeline to evaluate and benchmark multi-robot search algorithms. The presented physical setup innovatively uses an acoustic source (that is safe and inexpensive) and small ground robots (e-pucks) operating in a standard motion-capture environment. This setup can be easily recreated and used by most robotics researchers. The acoustic source also presents interesting uncertainty in terms of its noise-to-signal ratio, which is useful to assess sim-to-real gaps. The overall software pipeline is designed to readily interface with any multi-robot search algorithm with minimal effort and is executable in parallel asynchronous form. This pipeline includes a framework for distributed implementation of multi-robot or swarm search algorithms, integrated with a ROS (Robotics Operating System)-based software stack for motion capture supported localization. The utility of this novel setup is demonstrated by using it to evaluate two state-of-the-art multi-robot search algorithms, based on swarm optimization and batch-Bayesian Optimization (called Bayes-Swarm), as well as a random walk baseline. 

**Abstract (ZH)**: 多机器人信号源定位的实验室物理设置及其开源软件管道的研究 

---
# VLM-Empowered Multi-Mode System for Efficient and Safe Planetary Navigation 

**Title (ZH)**: VLM赋能的多模式系统及其在高效与安全行星导航中的应用 

**Authors**: Sinuo Cheng, Ruyi Zhou, Wenhao Feng, Huaiguang Yang, Haibo Gao, Zongquan Deng, Liang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.16703)  

**Abstract**: The increasingly complex and diverse planetary exploration environment requires more adaptable and flexible rover navigation strategy. In this study, we propose a VLM-empowered multi-mode system to achieve efficient while safe autonomous navigation for planetary rovers. Vision-Language Model (VLM) is used to parse scene information by image inputs to achieve a human-level understanding of terrain complexity. Based on the complexity classification, the system switches to the most suitable navigation mode, composing of perception, mapping and planning modules designed for different terrain types, to traverse the terrain ahead before reaching the next waypoint. By integrating the local navigation system with a map server and a global waypoint generation module, the rover is equipped to handle long-distance navigation tasks in complex scenarios. The navigation system is evaluated in various simulation environments. Compared to the single-mode conservative navigation method, our multi-mode system is able to bootstrap the time and energy efficiency in a long-distance traversal with varied type of obstacles, enhancing efficiency by 79.5%, while maintaining its avoidance capabilities against terrain hazards to guarantee rover safety. More system information is shown at this https URL. 

**Abstract (ZH)**: 行星探索环境日益复杂多变，需要更加适应灵活的火星车导航策略。本研究提出一种基于VLM的多模式系统，以实现高效安全的自主导航。视觉语言模型（VLM）通过图像输入解析场景信息，实现对地形复杂性的类人类理解。基于复杂性分类，系统切换到最适合的导航模式，该模式由设计适用于不同地形类型的感知、制图和规划模块组成，以便在到达下一个航点前穿越前方地形。通过整合局部导航系统、地图服务器和全球航点生成模块，火星车能够处理复杂场景下的远程导航任务。导航系统在多种仿真环境中进行了评估。与单一模式保守导航方法相比，本多模式系统能够在不同类型障碍物存在的远程穿越中提高79.5%的时间和能量效率，同时保持对地形危险的规避能力，确保火星车的安全。更多系统信息详见此链接：[更多系统信息链接]。 

---
# Compliant Residual DAgger: Improving Real-World Contact-Rich Manipulation with Human Corrections 

**Title (ZH)**: compliant residual DAgger: 通过人类纠正提高实际场景中接触丰富的 manipulation 技能 

**Authors**: Xiaomeng Xu, Yifan Hou, Zeyi Liu, Shuran Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.16685)  

**Abstract**: We address key challenges in Dataset Aggregation (DAgger) for real-world contact-rich manipulation: how to collect informative human correction data and how to effectively update policies with this new data. We introduce Compliant Residual DAgger (CR-DAgger), which contains two novel components: 1) a Compliant Intervention Interface that leverages compliance control, allowing humans to provide gentle, accurate delta action corrections without interrupting the ongoing robot policy execution; and 2) a Compliant Residual Policy formulation that learns from human corrections while incorporating force feedback and force control. Our system significantly enhances performance on precise contact-rich manipulation tasks using minimal correction data, improving base policy success rates by over 50\% on two challenging tasks (book flipping and belt assembly) while outperforming both retraining-from-scratch and finetuning approaches. Through extensive real-world experiments, we provide practical guidance for implementing effective DAgger in real-world robot learning tasks. Result videos are available at: this https URL 

**Abstract (ZH)**: 我们研究了现实世界中接触丰富的操作数据集聚合（Dataset Aggregation, DAgger）的关键挑战：如何收集有用的人类更正数据，以及如何有效利用这些新数据更新策略。我们引入了Compliant Residual DAgger (CR-DAgger)，其包含两个新颖的组成部分：1) 一种顺应性干预接口，利用顺应性控制，使人类能够在不中断机器人策略执行的情况下提供温柔准确的增量动作更正；2) 一种顺应性残差策略形式化，该形式化从人类更正中学习，同时结合力反馈和力控制。我们的系统使用极少的更正数据显著提升了精确接触丰富操作任务的表现，相对于两个具有挑战性的任务（书本翻转和带子装配），基线策略的成功率提高了超过50%，并且优于从头训练和微调的方法。通过广泛的现实世界实验，我们提供了在实际机器人学习任务中实施有效DAgger的实用指导。结果视频可在以下链接查看：this https URL 

---
# CodeDiffuser: Attention-Enhanced Diffusion Policy via VLM-Generated Code for Instruction Ambiguity 

**Title (ZH)**: CodeDiffuser: 通过VLM生成代码增强注意力的扩散策略以解决指令歧义 

**Authors**: Guang Yin, Yitong Li, Yixuan Wang, Dale McConachie, Paarth Shah, Kunimatsu Hashimoto, Huan Zhang, Katherine Liu, Yunzhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16652)  

**Abstract**: Natural language instructions for robotic manipulation tasks often exhibit ambiguity and vagueness. For instance, the instruction "Hang a mug on the mug tree" may involve multiple valid actions if there are several mugs and branches to choose from. Existing language-conditioned policies typically rely on end-to-end models that jointly handle high-level semantic understanding and low-level action generation, which can result in suboptimal performance due to their lack of modularity and interpretability. To address these challenges, we introduce a novel robotic manipulation framework that can accomplish tasks specified by potentially ambiguous natural language. This framework employs a Vision-Language Model (VLM) to interpret abstract concepts in natural language instructions and generates task-specific code - an interpretable and executable intermediate representation. The generated code interfaces with the perception module to produce 3D attention maps that highlight task-relevant regions by integrating spatial and semantic information, effectively resolving ambiguities in instructions. Through extensive experiments, we identify key limitations of current imitation learning methods, such as poor adaptation to language and environmental variations. We show that our approach excels across challenging manipulation tasks involving language ambiguity, contact-rich manipulation, and multi-object interactions. 

**Abstract (ZH)**: 自然语言指令在机器人操作任务中的表达往往具有歧义性和模糊性。例如，“将茶杯挂在茶杯树上”这一指令在存在多个茶杯和分支的情况下，可能涉及多种有效的操作方式。现有的基于语言的策略通常依赖于端到端模型，这些模型能够同时处理高层次语义理解和低层次动作生成，但由于缺乏模块化和可解释性，可能会导致性能不佳。为应对这些挑战，我们提出了一种新的机器人操作框架，能够完成由可能含糊不清的自然语言指定的任务。该框架利用视觉-语言模型（VLM）解释自然语言指令中的抽象概念，并生成特定任务的代码——一种具有可解释性和可执行性的中间表示。生成的代码与感知模块接口，通过整合空间和语义信息生成3D注意图，突出显示任务相关的区域，从而有效解决指令中的歧义性。通过广泛实验证明，我们的方法在涉及语言歧义、丰富接触操作及多物体交互等具有挑战性的操作任务中表现优异。 

---
# See What I Mean? Expressiveness and Clarity in Robot Display Design 

**Title (ZH)**: 你看得明白吗？机器人展示设计中的表达力与清晰度研究 

**Authors**: Matthew Ebisu, Hang Yu, Reuben Aronson, Elaine Short  

**Link**: [PDF](https://arxiv.org/pdf/2506.16643)  

**Abstract**: Nonverbal visual symbols and displays play an important role in communication when humans and robots work collaboratively. However, few studies have investigated how different types of non-verbal cues affect objective task performance, especially in a dynamic environment that requires real time decision-making. In this work, we designed a collaborative navigation task where the user and the robot only had partial information about the map on each end and thus the users were forced to communicate with a robot to complete the task. We conducted our study in a public space and recruited 37 participants who randomly passed by our setup. Each participant collaborated with a robot utilizing either animated anthropomorphic eyes and animated icons, or static anthropomorphic eyes and static icons. We found that participants that interacted with a robot with animated displays reported the greatest level of trust and satisfaction; that participants interpreted static icons the best; and that participants with a robot with static eyes had the highest completion success. These results suggest that while animation can foster trust with robots, human-robot communication can be optimized by the addition of familiar static icons that may be easier for users to interpret. We published our code, designed symbols, and collected results online at: this https URL. 

**Abstract (ZH)**: 非言语视觉符号和显示在人机协作沟通中扮演重要角色，但在动态环境下的实时决策中，不同类型的非言语线索如何影响客观任务绩效的研究较少。在本研究中，我们设计了一个协作导航任务，用户和机器人仅掌握地图的部分信息，从而迫使用户与机器人进行沟通以完成任务。我们在公共场所进行了研究，招募了37名随机路过该设置的参与者。每位参与者与机器人协作，使用的信号方式为带有动画拟人眼睛和动画图标，或带有静态拟人眼睛和静态图标。我们发现，与具有动画显示的机器人互动的参与者报告了最高的信任和满意度；参与者对静态图标理解最好；与具有静态眼睛的机器人合作的参与者完成任务的成功率最高。这些结果表明，虽然动画可以促进对机器人的信任，但通过添加易于用户理解的熟悉的静态图标，可以优化人机沟通。我们在网上发布了我们的代码、设计的符号和收集的结果：https://yourlinkhere。 

---
# History-Augmented Vision-Language Models for Frontier-Based Zero-Shot Object Navigation 

**Title (ZH)**: 基于历史的视觉-语言模型在前沿导向的零-shot 物体导航中的应用 

**Authors**: Mobin Habibpour, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2506.16623)  

**Abstract**: Object Goal Navigation (ObjectNav) challenges robots to find objects in unseen environments, demanding sophisticated reasoning. While Vision-Language Models (VLMs) show potential, current ObjectNav methods often employ them superficially, primarily using vision-language embeddings for object-scene similarity checks rather than leveraging deeper reasoning. This limits contextual understanding and leads to practical issues like repetitive navigation behaviors. This paper introduces a novel zero-shot ObjectNav framework that pioneers the use of dynamic, history-aware prompting to more deeply integrate VLM reasoning into frontier-based exploration. Our core innovation lies in providing the VLM with action history context, enabling it to generate semantic guidance scores for navigation actions while actively avoiding decision loops. We also introduce a VLM-assisted waypoint generation mechanism for refining the final approach to detected objects. Evaluated on the HM3D dataset within Habitat, our approach achieves a 46% Success Rate (SR) and 24.8% Success weighted by Path Length (SPL). These results are comparable to state-of-the-art zero-shot methods, demonstrating the significant potential of our history-augmented VLM prompting strategy for more robust and context-aware robotic navigation. 

**Abstract (ZH)**: 基于对象的导航（ObjectNav）挑战机器人在未见环境中寻找对象，要求复杂的推理能力。尽管视觉-语言模型（VLMs）显示出潜力，当前的ObjectNav方法往往浅层次地利用它们，主要使用视觉-语言嵌入进行物体-场景相似性检查，而非充分利用更深层次的推理。这限制了上下文理解，并导致反复的导航行为。本文提出了一种新颖的零样本ObjectNav框架，该框架开创性地使用动态、历史感知提示，更深入地将VLM推理整合到前沿探索中。我们的核心创新在于为VLM提供动作历史上下文，使其能够生成导航行动的语义指导分数，同时积极避免决策循环。我们还引入了VLM辅助的航点生成机制，以细化对检测到的对象的最终接近。在Habitat的HM3D数据集上进行评估，我们的方法实现了46%的成功率（SR）和24.8%的成功路径长度加权成功率（SPL）。这些结果与最先进的零样本方法相当，展示了我们增强历史信息的VLM提示策略在更加稳健和上下文感知的机器人导航方面的巨大潜力。 

---
# DRIVE Through the Unpredictability:From a Protocol Investigating Slip to a Metric Estimating Command Uncertainty 

**Title (ZH)**: DRIVE通过不确定性：从一个研究滑动的协议到一个命令不确定性估计度量 

**Authors**: Nicolas Samson, William Larrivée-Hardy, William Dubois, Élie Roy-Brouard, Edith Brotherton, Dominic Baril, Julien Lépine, François Pomerleau  

**Link**: [PDF](https://arxiv.org/pdf/2506.16593)  

**Abstract**: Off-road autonomous navigation is a challenging task as it is mainly dependent on the accuracy of the motion model. Motion model performances are limited by their ability to predict the interaction between the terrain and the UGV, which an onboard sensor can not directly measure. In this work, we propose using the DRIVE protocol to standardize the collection of data for system identification and characterization of the slip state space. We validated this protocol by acquiring a dataset with two platforms (from 75 kg to 470 kg) on six terrains (i.e., asphalt, grass, gravel, ice, mud, sand) for a total of 4.9 hours and 14.7 km. Using this data, we evaluate the DRIVE protocol's ability to explore the velocity command space and identify the reachable velocities for terrain-robot interactions. We investigated the transfer function between the command velocity space and the resulting steady-state slip for an SSMR. An unpredictability metric is proposed to estimate command uncertainty and help assess risk likelihood and severity in deployment. Finally, we share our lessons learned on running system identification on large UGV to help the community. 

**Abstract (ZH)**: 离线自主导航是一个具有挑战性的任务，因为它主要依赖于运动模型的准确性。运动模型性能受限于它们预测地形与UGV之间相互作用的能力，而这种相互作用是车载传感器无法直接测量的。在这项工作中，我们提出使用DRIVE协议来标准化系统识别和描述滑动状态空间的数据收集。通过在六种不同地形（即沥青、草地、碎石、冰、泥地、沙地）上使用两种平台（质量从75公斤到470公斤）收集数据，总时长为4.9小时，行驶距离14.7公里，验证了该协议。利用这些数据，我们评估了DRIVE协议探索速度命令空间和识别地形-机器人相互作用可达到的速度范围的能力。我们研究了SSMR的速度命令空间与最终稳定滑动之间的传递函数。提出了一个不确定性度量指标，以估计命令不确定性并帮助评估部署中的风险可能性和严重程度。最后，我们分享了在大型UGV上进行系统识别过程中获取的经验教训，以帮助社区。 

---
# Reimagination with Test-time Observation Interventions: Distractor-Robust World Model Predictions for Visual Model Predictive Control 

**Title (ZH)**: 基于测试时观察干预的重塑：视觉模型预测控制中的干扰物鲁棒世界模型预测 

**Authors**: Yuxin Chen, Jianglan Wei, Chenfeng Xu, Boyi Li, Masayoshi Tomizuka, Andrea Bajcsy, Ran Tian  

**Link**: [PDF](https://arxiv.org/pdf/2506.16565)  

**Abstract**: World models enable robots to "imagine" future observations given current observations and planned actions, and have been increasingly adopted as generalized dynamics models to facilitate robot learning. Despite their promise, these models remain brittle when encountering novel visual distractors such as objects and background elements rarely seen during training. Specifically, novel distractors can corrupt action outcome predictions, causing downstream failures when robots rely on the world model imaginations for planning or action verification. In this work, we propose Reimagination with Observation Intervention (ReOI), a simple yet effective test-time strategy that enables world models to predict more reliable action outcomes in open-world scenarios where novel and unanticipated visual distractors are inevitable. Given the current robot observation, ReOI first detects visual distractors by identifying which elements of the scene degrade in physically implausible ways during world model prediction. Then, it modifies the current observation to remove these distractors and bring the observation closer to the training distribution. Finally, ReOI "reimagines" future outcomes with the modified observation and reintroduces the distractors post-hoc to preserve visual consistency for downstream planning and verification. We validate our approach on a suite of robotic manipulation tasks in the context of action verification, where the verifier needs to select desired action plans based on predictions from a world model. Our results show that ReOI is robust to both in-distribution and out-of-distribution visual distractors. Notably, it improves task success rates by up to 3x in the presence of novel distractors, significantly outperforming action verification that relies on world model predictions without imagination interventions. 

**Abstract (ZH)**: Reimagination with Observation Interventionmissive 

---
# An Optimization-Augmented Control Framework for Single and Coordinated Multi-Arm Robotic Manipulation 

**Title (ZH)**: 单臂和协调多臂机器人操作的优化增强控制框架 

**Authors**: Melih Özcan, Ozgur S. Oguz  

**Link**: [PDF](https://arxiv.org/pdf/2506.16555)  

**Abstract**: Robotic manipulation demands precise control over both contact forces and motion trajectories. While force control is essential for achieving compliant interaction and high-frequency adaptation, it is limited to operations in close proximity to the manipulated object and often fails to maintain stable orientation during extended motion sequences. Conversely, optimization-based motion planning excels in generating collision-free trajectories over the robot's configuration space but struggles with dynamic interactions where contact forces play a crucial role. To address these limitations, we propose a multi-modal control framework that combines force control and optimization-augmented motion planning to tackle complex robotic manipulation tasks in a sequential manner, enabling seamless switching between control modes based on task requirements. Our approach decomposes complex tasks into subtasks, each dynamically assigned to one of three control modes: Pure optimization for global motion planning, pure force control for precise interaction, or hybrid control for tasks requiring simultaneous trajectory tracking and force regulation. This framework is particularly advantageous for bimanual and multi-arm manipulation, where synchronous motion and coordination among arms are essential while considering both the manipulated object and environmental constraints. We demonstrate the versatility of our method through a range of long-horizon manipulation tasks, including single-arm, bimanual, and multi-arm applications, highlighting its ability to handle both free-space motion and contact-rich manipulation with robustness and precision. 

**Abstract (ZH)**: 机器人操作需要对接触力和运动轨迹进行精确控制。虽然力控制对于实现柔顺交互和高频适应至关重要，但它受限于与操作对象的近距离操作，并且在长时间的运动序列中往往难以维持稳定的方位。相反，基于优化的运动规划在生成机器人配置空间中的无碰撞轨迹方面表现出色，但在涉及接触力起关键作用的动态交互中却力不从心。为了解决这些限制，我们提出了一种多模态控制框架，该框架结合了力控制和优化增强的运动规划，以顺序方式应对复杂的机器人操作任务，并根据任务需求无缝切换控制模式。该方法将复杂的任务分解为子任务，并动态分配给三种控制模式之一：纯粹的优化用于全局运动规划、纯粹的力控制用于精确交互，或混合控制用于需要同时进行轨迹跟踪和力调节的任务。该框架特别适用于双臂和多臂操作，其中同步运动和手臂之间的协调至关重要，同时需要考虑操作对象和环境约束。通过一系列长时程操作任务，包括单臂、双臂和多臂应用，我们的方法展示了其在自由空间运动和丰富的接触操作中具有鲁棒性和精确性的 versatility。 

---
# BIDA: A Bi-level Interaction Decision-making Algorithm for Autonomous Vehicles in Dynamic Traffic Scenarios 

**Title (ZH)**: BIDA：动态交通场景中自主车辆的多层次交互决策算法 

**Authors**: Liyang Yu, Tianyi Wang, Junfeng Jiao, Fengwu Shan, Hongqing Chu, Bingzhao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16546)  

**Abstract**: In complex real-world traffic environments, autonomous vehicles (AVs) need to interact with other traffic participants while making real-time and safety-critical decisions accordingly. The unpredictability of human behaviors poses significant challenges, particularly in dynamic scenarios, such as multi-lane highways and unsignalized T-intersections. To address this gap, we design a bi-level interaction decision-making algorithm (BIDA) that integrates interactive Monte Carlo tree search (MCTS) with deep reinforcement learning (DRL), aiming to enhance interaction rationality, efficiency and safety of AVs in dynamic key traffic scenarios. Specifically, we adopt three types of DRL algorithms to construct a reliable value network and policy network, which guide the online deduction process of interactive MCTS by assisting in value update and node selection. Then, a dynamic trajectory planner and a trajectory tracking controller are designed and implemented in CARLA to ensure smooth execution of planned maneuvers. Experimental evaluations demonstrate that our BIDA not only enhances interactive deduction and reduces computational costs, but also outperforms other latest benchmarks, which exhibits superior safety, efficiency and interaction rationality under varying traffic conditions. 

**Abstract (ZH)**: 在复杂现实交通环境中，自动驾驶车辆（AVs）需要在进行实时和安全关键决策的同时与其它交通参与者互动。人类行为的不可预测性在动态场景下，如多车道高速公路和无信号T形交叉路口等，提出了重大挑战。为应对这一挑战，我们设计了一种双层互动决策算法（BIDA），将交互蒙特卡洛树搜索（MCTS）与深度强化学习（DRL）相结合，旨在提升自动驾驶车辆在动态关键交通场景中的互动合理性、效率和安全性。具体而言，我们采用了三种类型的DRL算法构建了可靠的值网络和策略网络，通过协助价值更新和节点选择来引导交互MCTS的在线推断过程。然后，在CARLA中设计并实现了动态轨迹规划器和轨迹跟踪控制器，以确保计划机动动作的平滑执行。实验评估表明，我们的BIDA不仅提高了互动推断的合理性和降低了计算成本，还在各种交通条件下超越了其他最新基准，显示出优越的安全性、效率和互动合理性。 

---
# Agile, Autonomous Spacecraft Constellations with Disruption Tolerant Networking to Monitor Precipitation and Urban Floods 

**Title (ZH)**: 具有中断 tolerant 网络的敏捷自主卫星星座及其在监测降水和城市洪涝中的应用 

**Authors**: Sreeja Roy-Singh, Alan P. Li, Vinay Ravindra, Roderick Lammers, Marc Sanchez Net  

**Link**: [PDF](https://arxiv.org/pdf/2506.16537)  

**Abstract**: Fully re-orientable small spacecraft are now supported by commercial technologies, allowing them to point their instruments in any direction and capture images, with short notice. When combined with improved onboard processing, and implemented on a constellation of inter-communicable satellites, this intelligent agility can significantly increase responsiveness to transient or evolving phenomena. We demonstrate a ground-based and onboard algorithmic framework that combines orbital mechanics, attitude control, inter-satellite communication, intelligent prediction and planning to schedule the time-varying, re-orientation of agile, small satellites in a constellation. Planner intelligence is improved by updating the predictive value of future space-time observations based on shared observations of evolving episodic precipitation and urban flood forecasts. Reliable inter-satellite communication within a fast, dynamic constellation topology is modeled in the physical, access control and network layer. We apply the framework on a representative 24-satellite constellation observing 5 global regions. Results show appropriately low latency in information exchange (average within 1/3rd available time for implicit consensus), enabling the onboard scheduler to observe ~7% more flood magnitude than a ground-based implementation. Both onboard and offline versions performed ~98% better than constellations without agility. 

**Abstract (ZH)**: 完全可重新定向的小型航天器现在由商业技术支持，允许它们在任何方向指向仪器并快速捕获图像。通过结合改进的机载处理，并在通信卫星星座中实施，这种智能敏捷性可以显著提高对瞬态或演变现象的响应性。我们展示了结合轨道力学、姿态控制、卫星间通信、智能预测和规划的地面和机载算法框架，以协调星座中敏捷小型卫星的时间变化重新定向。通过基于共享的演化 episodic 降水和城市洪水预报更新未来时空观测的预测值来提高规划的智能性。在物理层、访问控制层和网络层中建模了快速动态星座拓扑内的可靠卫星间通信。我们在一个代表性的24颗卫星星座上应用了该框架，该星座观察5个全球区域。结果显示适当较低的延迟（平均在隐式共识可用时间的三分之一以内）使得机载调度器能够检测到比地面实现多约7%的洪水规模。无论是机载版本还是离线版本都比没有敏捷性的星座高出约98%。 

---
# eCAV: An Edge-Assisted Evaluation Platform for Connected Autonomous Vehicles 

**Title (ZH)**: 基于边缘辅助的Connected Autonomous Vehicles评估平台:eCAV 

**Authors**: Tyler Landle, Jordan Rapp, Dean Blank, Chandramouli Amarnath, Abhijit Chatterjee, Alex Daglis, Umakishore Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2506.16535)  

**Abstract**: As autonomous vehicles edge closer to widespread adoption, enhancing road safety through collision avoidance and minimization of collateral damage becomes imperative. Vehicle-to-everything (V2X) technologies, which include vehicle-to-vehicle (V2V), vehicle-to-infrastructure (V2I), and vehicle-to-cloud (V2C), are being proposed as mechanisms to achieve this safety improvement.
Simulation-based testing is crucial for early-stage evaluation of Connected Autonomous Vehicle (CAV) control systems, offering a safer and more cost-effective alternative to real-world tests. However, simulating large 3D environments with many complex single- and multi-vehicle sensors and controllers is computationally intensive. There is currently no evaluation framework that can effectively evaluate realistic scenarios involving large numbers of autonomous vehicles.
We propose eCAV -- an efficient, modular, and scalable evaluation platform to facilitate both functional validation of algorithmic approaches to increasing road safety, as well as performance prediction of algorithms of various V2X technologies, including a futuristic Vehicle-to-Edge control plane and correspondingly designed control algorithms. eCAV can model up to 256 vehicles running individual control algorithms without perception enabled, which is $8\times$ more vehicles than what is possible with state-of-the-art alternatives. %faster than state-of-the-art alternatives that can simulate $8\times$ fewer vehicles. With perception enabled, eCAV simulates up to 64 vehicles with a step time under 800ms, which is $4\times$ more and $1.5\times$ faster than the state-of-the-art OpenCDA framework. 

**Abstract (ZH)**: 随着自动驾驶车辆向广泛应用接近，通过碰撞避免和减少副损伤来增强道路安全变得至关重要。车辆到万物（V2X）技术，包括车辆到车辆（V2V）、车辆到基础设施（V2I）和车辆到云（V2C），被提出作为实现这一安全提升的机制。

基于仿真的测试对于联网自动驾驶车辆（CAV）控制系统的早期评估至关重要， Offering a更安全和更具成本效益的替代方案，智能替代真实的道路测试。然而，模拟包含许多复杂单、多车辆传感器和控制器的大型三维环境计算量巨大。目前尚不存在有效的评估框架来评估大量自动驾驶车辆的现实场景。

我们提出eCAV——一种高效的、模块化的和可扩展的评估平台，以促进对增加道路安全的算法方法的功能验证，以及各种V2X技术的算法性能预测，包括未来的车辆到边缘控制平面及其相应设计的控制算法。eCAV在未启用感知的情况下可模拟最多256辆车单独运行控制算法，比最先进的替代方案多8倍的车辆。当启用感知时，eCAV可模拟最多64辆车，每步时间低于800毫秒，比最先进的OpenCDA框架多4倍、快1.5倍。 

---
# Grounding Language Models with Semantic Digital Twins for Robotic Planning 

**Title (ZH)**: 基于语义数字孪生的语言模型在机器人规划中的应用 

**Authors**: Mehreen Naeem, Andrew Melnik, Michael Beetz  

**Link**: [PDF](https://arxiv.org/pdf/2506.16493)  

**Abstract**: We introduce a novel framework that integrates Semantic Digital Twins (SDTs) with Large Language Models (LLMs) to enable adaptive and goal-driven robotic task execution in dynamic environments. The system decomposes natural language instructions into structured action triplets, which are grounded in contextual environmental data provided by the SDT. This semantic grounding allows the robot to interpret object affordances and interaction rules, enabling action planning and real-time adaptability. In case of execution failures, the LLM utilizes error feedback and SDT insights to generate recovery strategies and iteratively revise the action plan. We evaluate our approach using tasks from the ALFRED benchmark, demonstrating robust performance across various household scenarios. The proposed framework effectively combines high-level reasoning with semantic environment understanding, achieving reliable task completion in the face of uncertainty and failure. 

**Abstract (ZH)**: 我们将一种将语义数字孪生体（SDTs）与大型语言模型（LLMs）集成的新框架应用于动态环境中的自适应和目标驱动的机器人任务执行。该系统将自然语言指令分解为结构化的动作三元组，这些动作三元组基于SDT提供的上下文环境数据进行接地。这种语义接地使机器人能够解释物体的功能和交互规则，从而实现动作规划和实时适应。在执行失败时，LLM利用错误反馈和SDT的见解来生成恢复策略，并迭代地修正动作计划。我们使用ALFRED基准的任务对这种方法进行评估，展示了其在各种家庭场景中的稳健性能。所提出框架有效地结合了高层推理与语义环境理解，能够在不确定性与失败面前可靠地完成任务。 

---
# Human2LocoMan: Learning Versatile Quadrupedal Manipulation with Human Pretraining 

**Title (ZH)**: Human2LocoMan: 基于人类预训练的多功能四足操控学习 

**Authors**: Yaru Niu, Yunzhe Zhang, Mingyang Yu, Changyi Lin, Chenhao Li, Yikai Wang, Yuxiang Yang, Wenhao Yu, Tingnan Zhang, Bingqing Chen, Jonathan Francis, Zhenzhen Li, Jie Tan, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16475)  

**Abstract**: Quadrupedal robots have demonstrated impressive locomotion capabilities in complex environments, but equipping them with autonomous versatile manipulation skills in a scalable way remains a significant challenge. In this work, we introduce a cross-embodiment imitation learning system for quadrupedal manipulation, leveraging data collected from both humans and LocoMan, a quadruped equipped with multiple manipulation modes. Specifically, we develop a teleoperation and data collection pipeline, which unifies and modularizes the observation and action spaces of the human and the robot. To effectively leverage the collected data, we propose an efficient modularized architecture that supports co-training and pretraining on structured modality-aligned data across different embodiments. Additionally, we construct the first manipulation dataset for the LocoMan robot, covering various household tasks in both unimanual and bimanual modes, supplemented by a corresponding human dataset. We validate our system on six real-world manipulation tasks, where it achieves an average success rate improvement of 41.9% overall and 79.7% under out-of-distribution (OOD) settings compared to the baseline. Pretraining with human data contributes a 38.6% success rate improvement overall and 82.7% under OOD settings, enabling consistently better performance with only half the amount of robot data. Our code, hardware, and data are open-sourced at: this https URL. 

**Abstract (ZH)**: 四足机器人在复杂环境中的搬运能力已表现出色，但以可扩展的方式为其配备自主的多用途操作技能仍是一项重大挑战。本文介绍了一种跨体态模仿学习系统，利用来自人类和LocoMan（一种配备多种操作模式的四足机器人）的数据。具体来说，我们开发了一种远程操作和数据采集管道，统一并模块化了人类和机器人之间的观察和动作空间。为了有效利用收集到的数据，我们提出了一个高效的模块化架构，支持在不同体态的结构模态对齐数据上进行协同训练和预训练。此外，我们构建了LocoMan机器人首个操作数据集，涵盖单手和双手模式下的各种家务任务，并提供了相应的手动数据集。我们在六项真实世界的操作任务中验证了该系统，结果显示整体成功率提升了41.9%，在未见分布设置下的成功率提升了79.7%。使用人类数据进行预训练整体提升了38.6%的成功率，在未见分布设置下的成功率提升了82.7%，仅使用一半机器人数据即可实现持续更好的性能。代码、硬件和数据已开源：this https URL。 

---
# Full-Pose Tracking via Robust Control for Over-Actuated Multirotors 

**Title (ZH)**: 过驱动多旋翼飞行器的鲁棒控制全姿态跟踪 

**Authors**: Mohamad Hachem, Clément Roos, Thierry Miquel, Murat Bronz  

**Link**: [PDF](https://arxiv.org/pdf/2506.16427)  

**Abstract**: This paper presents a robust cascaded control architecture for over-actuated multirotors. It extends the Incremental Nonlinear Dynamic Inversion (INDI) control combined with structured H_inf control, initially proposed for under-actuated multirotors, to a broader range of multirotor configurations. To achieve precise and robust attitude and position tracking, we employ a weighted least-squares geometric guidance control allocation method, formulated as a quadratic optimization problem, enabling full-pose tracking. The proposed approach effectively addresses key challenges, such as preventing infeasible pose references and enhancing robustness against disturbances, as well as considering multirotor's actual physical limitations. Numerical simulations with an over-actuated hexacopter validate the method's effectiveness, demonstrating its adaptability to diverse mission scenarios and its potential for real-world aerial applications. 

**Abstract (ZH)**: 本文提出了一种适用于过度驱动多旋翼者的稳健级联控制架构。该架构将初始用于欠驱动多旋翼者的增量非线性动态逆控制（INDI）与结构化H_inf控制相结合，扩展到更广泛的多旋翼配置。为了实现精确且稳健的姿态和位置跟踪，我们采用了一种加权最小二乘几何引导控制分配方法，将其形式化为二次优化问题，从而实现全方位姿态跟踪。所提出的方法有效应对了诸如预防不可行姿态参考和增强对干扰的鲁棒性等关键挑战，并考虑了多旋翼的实际物理限制。数值 simulations 与一个过度驱动的六旋翼机的仿真验证了该方法的有效性，展示了其在多样任务场景下的适应性及其在实际空中应用中的潜在价值。 

---
# CSC-MPPI: A Novel Constrained MPPI Framework with DBSCAN for Reliable Obstacle Avoidance 

**Title (ZH)**: CSC-MPPI：一种基于DBSCAN的新型约束MPPI框架以实现可靠的障碍 avoidance 

**Authors**: Leesai Park, Keunwoo Jang, Sanghyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.16386)  

**Abstract**: This paper proposes Constrained Sampling Cluster Model Predictive Path Integral (CSC-MPPI), a novel constrained formulation of MPPI designed to enhance trajectory optimization while enforcing strict constraints on system states and control inputs. Traditional MPPI, which relies on a probabilistic sampling process, often struggles with constraint satisfaction and generates suboptimal trajectories due to the weighted averaging of sampled trajectories. To address these limitations, the proposed framework integrates a primal-dual gradient-based approach and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to steer sampled input trajectories into feasible regions while mitigating risks associated with weighted averaging. First, to ensure that sampled trajectories remain within the feasible region, the primal-dual gradient method is applied to iteratively shift sampled inputs while enforcing state and control constraints. Then, DBSCAN groups the sampled trajectories, enabling the selection of representative control inputs within each cluster. Finally, among the representative control inputs, the one with the lowest cost is chosen as the optimal action. As a result, CSC-MPPI guarantees constraint satisfaction, improves trajectory selection, and enhances robustness in complex environments. Simulation and real-world experiments demonstrate that CSC-MPPI outperforms traditional MPPI in obstacle avoidance, achieving improved reliability and efficiency. The experimental videos are available at this https URL 

**Abstract (ZH)**: 受限采样聚类模型预测路径积分（CSC-MPPI）：一种强化轨迹优化并严格约束系统状态和控制输入的新型方法 

---
# Comparison between External and Internal Single Stage Planetary gearbox actuators for legged robots 

**Title (ZH)**: 外部与内部单级行星齿轮减速器执行机构在腿式机器人中的对比 

**Authors**: Aman Singh, Deepak Kapa, Prasham Chedda, Shishir N.Y. Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2506.16356)  

**Abstract**: Legged robots, such as quadrupeds and humanoids, require high-performance actuators for efficient locomotion. Quasi-Direct-Drive (QDD) actuators with single-stage planetary gearboxes offer low inertia, high efficiency, and transparency. Among planetary gearbox architectures, Internal (ISSPG) and External Single-Stage Planetary Gearbox (ESSPG) are the two predominant designs. While ISSPG is often preferred for its compactness and high torque density at certain gear ratios, no objective comparison between the two architectures exists. Additionally, existing designs rely on heuristics rather than systematic optimization. This paper presents a design framework for optimally selecting actuator parameters based on given performance requirements and motor specifications. Using this framework, we generate and analyze various optimized gearbox designs for both architectures. Our results demonstrate that for the T-motor U12, ISSPG is the superior choice within the lower gear ratio range of 5:1 to 7:1, offering a lighter design. However, for gear ratios exceeding 7:1, ISSPG becomes infeasible, making ESSPG the better option in the 7:1 to 11:1 range. To validate our approach, we designed and optimized two actuators for manufacturing: an ISSPG with a 6.0:1 gear ratio and an ESSPG with a 7.2:1 gear ratio. Their respective masses closely align with our optimization model predictions, confirming the effectiveness of our methodology. 

**Abstract (ZH)**: 基于给定性能要求和电机规格优化选择传动器参数的设计框架：内部单级行星齿轮箱与外部单级行星齿轮箱的对比分析 

---
# Goal-conditioned Hierarchical Reinforcement Learning for Sample-efficient and Safe Autonomous Driving at Intersections 

**Title (ZH)**: 面向交叉口高效安全自主驾驶的条件导向分层强化学习 

**Authors**: Yiou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16336)  

**Abstract**: Reinforcement learning (RL) exhibits remarkable potential in addressing autonomous driving tasks. However, it is difficult to train a sample-efficient and safe policy in complex scenarios. In this article, we propose a novel hierarchical reinforcement learning (HRL) framework with a goal-conditioned collision prediction (GCCP) module. In the hierarchical structure, the GCCP module predicts collision risks according to different potential subgoals of the ego vehicle. A high-level decision-maker choose the best safe subgoal. A low-level motion-planner interacts with the environment according to the subgoal. Compared to traditional RL methods, our algorithm is more sample-efficient, since its hierarchical structure allows reusing the policies of subgoals across similar tasks for various navigation scenarios. In additional, the GCCP module's ability to predict both the ego vehicle's and surrounding vehicles' future actions according to different subgoals, ensures the safety of the ego vehicle throughout the decision-making process. Experimental results demonstrate that the proposed method converges to an optimal policy faster and achieves higher safety than traditional RL methods. 

**Abstract (ZH)**: 强化学习（RL）在应对自动驾驶任务方面展现了显著潜力。然而，在复杂场景下训练高效且安全的策略颇具挑战。本文提出了一种新颖的分层强化学习（HRL）框架，包含目标引导的碰撞预测（GCCP）模块。在分层次结构中，GCCP模块根据自主车辆的不同潜在子目标预测碰撞风险，高层次决策制定者选择最佳安全子目标，低层次运动规划者根据子目标与环境交互。与传统的RL方法相比，我们的算法更具样本效率，因为其分层结构允许在相似任务中重用子目标的策略，以适应各种导航场景。此外，GCCP模块根据不同子目标预测自主车辆及其周围车辆的未来行动，确保在决策过程中自主车辆的安全性。实验结果表明，所提出的方法比传统RL方法更快地收敛到最优策略，并且安全性更高。 

---
# M-Predictive Spliner: Enabling Spatiotemporal Multi-Opponent Overtaking for Autonomous Racing 

**Title (ZH)**: M-预测型样条: 使能自主赛车时空多对手超越预测底盘 

**Authors**: Nadine Imholz, Maurice Brunner, Nicolas Baumann, Edoardo Ghignone, Michele Magno  

**Link**: [PDF](https://arxiv.org/pdf/2506.16301)  

**Abstract**: Unrestricted multi-agent racing presents a significant research challenge, requiring decision-making at the limits of a robot's operational capabilities. While previous approaches have either ignored spatiotemporal information in the decision-making process or been restricted to single-opponent scenarios, this work enables arbitrary multi-opponent head-to-head racing while considering the opponents' future intent. The proposed method employs a KF-based multi-opponent tracker to effectively perform opponent ReID by associating them across observations. Simultaneously, spatial and velocity GPR is performed on all observed opponent trajectories, providing predictive information to compute the overtaking maneuvers. This approach has been experimentally validated on a physical 1:10 scale autonomous racing car, achieving an overtaking success rate of up to 91.65% and demonstrating an average 10.13%-point improvement in safety at the same speed as the previous SotA. These results highlight its potential for high-performance autonomous racing. 

**Abstract (ZH)**: 不受限制的多agent竞速研究提出了一个重大的研究挑战，需要在机器人的操作极限范围内进行决策。虽然以前的方法要么在决策过程中忽略了时空信息，要么仅限于单对手场景，本工作则允许在考虑对手未来意图的情况下进行任意多对手一对一竞速。所提出的方法使用基于卡尔曼滤波的多对手跟踪器，通过跨观测关联对手来有效执行对手重识别。同时，对所有观测到的对手轨迹进行空间和速度广义回归分析，提供预测信息以计算超车动作。该方法已在物理1:10比例的自主竞速车上进行实验验证，超车成功率高达91.65%，并在相同速度下平均提高了10.13%的安全性。这些结果突显了其在高性能自主竞速中的潜力。 

---
# CapsDT: Diffusion-Transformer for Capsule Robot Manipulation 

**Title (ZH)**: CapsDT：用于胶囊机器人 manipulation 的扩散-变换器 

**Authors**: Xiting He, Mingwu Su, Xinqi Jiang, Long Bai, Jiewen Lai, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.16263)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as a prominent research area, showcasing significant potential across a variety of applications. However, their performance in endoscopy robotics, particularly endoscopy capsule robots that perform actions within the digestive system, remains unexplored. The integration of VLA models into endoscopy robots allows more intuitive and efficient interactions between human operators and medical devices, improving both diagnostic accuracy and treatment outcomes. In this work, we design CapsDT, a Diffusion Transformer model for capsule robot manipulation in the stomach. By processing interleaved visual inputs, and textual instructions, CapsDT can infer corresponding robotic control signals to facilitate endoscopy tasks. In addition, we developed a capsule endoscopy robot system, a capsule robot controlled by a robotic arm-held magnet, addressing different levels of four endoscopy tasks and creating corresponding capsule robot datasets within the stomach simulator. Comprehensive evaluations on various robotic tasks indicate that CapsDT can serve as a robust vision-language generalist, achieving state-of-the-art performance in various levels of endoscopy tasks while achieving a 26.25% success rate in real-world simulation manipulation. 

**Abstract (ZH)**: Vision-Language-Action (VLA)模型已成为一个重要的研究领域，展示了在多种应用中的巨大潜力。然而，这些模型在内窥镜机器人领域，特别是在消化系统内执行操作的胶囊内窥镜机器人中的性能尚未得到探索。将VLA模型集成到内窥镜机器人中可以实现人类操作者和医疗设备之间更直观和高效的交互，从而提高诊断准确性和治疗效果。在此工作中，我们设计了CapsDT，这是一种用于胃部胶囊机器人操作的扩散变压器模型，通过处理交错的视觉输入和文本指令，CapsDT可以推断出相应的机器人控制信号以辅助内窥镜任务。此外，我们开发了一种胶囊内窥镜机器人系统，该系统通过手持机械臂的磁铁控制胶囊机器人，并在不同的四级内窥镜任务中进行操作，同时在胃部模拟器中创建相应的胶囊机器人数据集。对各种机器人任务的全面评估表明，CapsDT可以作为强大的视觉-语言通用模型，实现各种程度内窥镜任务的先进性能，在实际模拟操作中的成功率达到了26.25%。 

---
# Probabilistic Collision Risk Estimation for Pedestrian Navigation 

**Title (ZH)**: 行人导航中的概率碰撞风险估计 

**Authors**: Amine Tourki, Paul Prevel, Nils Einecke, Tim Puphal, Alexandre Alahi  

**Link**: [PDF](https://arxiv.org/pdf/2506.16219)  

**Abstract**: Intelligent devices for supporting persons with vision impairment are becoming more widespread, but they are lacking behind the advancements in intelligent driver assistant system. To make a first step forward, this work discusses the integration of the risk model technology, previously used in autonomous driving and advanced driver assistance systems, into an assistance device for persons with vision impairment. The risk model computes a probabilistic collision risk given object trajectories which has previously been shown to give better indications of an object's collision potential compared to distance or time-to-contact measures in vehicle scenarios. In this work, we show that the risk model is also superior in warning persons with vision impairment about dangerous objects. Our experiments demonstrate that the warning accuracy of the risk model is 67% while both distance and time-to-contact measures reach only 51% accuracy for real-world data. 

**Abstract (ZH)**: 智能视力障碍辅助设备正变得越来越普遍，但它们落后于智能驾驶辅助系统的进步。为了迈出第一步，本文讨论了将 previously 用于自动驾驶和先进驾驶辅助系统的风险模型技术整合到视力障碍人员的辅助设备中。风险模型计算基于物体轨迹的概率碰撞风险，研究表明在车辆场景中这种测量方法比距离或接触时间指标能更准确地指示物体的碰撞潜力。在这项工作中，我们展示了风险模型在警告视力障碍人员关于危险物体方面也更为优越。我们的实验表明，风险模型的警告准确率为67%，而距离和接触时间指标在实际数据中的准确率仅为51%。 

---
# ControlVLA: Few-shot Object-centric Adaptation for Pre-trained Vision-Language-Action Models 

**Title (ZH)**: ControlVLA：预训练视觉-语言-行动模型的少样本对象中心适应方法 

**Authors**: Puhao Li, Yingying Wu, Ziheng Xi, Wanlin Li, Yuzhe Huang, Zhiyuan Zhang, Yinghan Chen, Jianan Wang, Song-Chun Zhu, Tengyu Liu, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16211)  

**Abstract**: Learning real-world robotic manipulation is challenging, particularly when limited demonstrations are available. Existing methods for few-shot manipulation often rely on simulation-augmented data or pre-built modules like grasping and pose estimation, which struggle with sim-to-real gaps and lack extensibility. While large-scale imitation pre-training shows promise, adapting these general-purpose policies to specific tasks in data-scarce settings remains unexplored. To achieve this, we propose ControlVLA, a novel framework that bridges pre-trained VLA models with object-centric representations via a ControlNet-style architecture for efficient fine-tuning. Specifically, to introduce object-centric conditions without overwriting prior knowledge, ControlVLA zero-initializes a set of projection layers, allowing them to gradually adapt the pre-trained manipulation policies. In real-world experiments across 6 diverse tasks, including pouring cubes and folding clothes, our method achieves a 76.7% success rate while requiring only 10-20 demonstrations -- a significant improvement over traditional approaches that require more than 100 demonstrations to achieve comparable success. Additional experiments highlight ControlVLA's extensibility to long-horizon tasks and robustness to unseen objects and backgrounds. 

**Abstract (ZH)**: 学习现实世界中的机器人 manipulation 挑战重重，尤其在示例有限的情况下。现有的少样本 manipulation 方法往往依赖于模拟增强的数据或预构建的模块（如抓取和姿态估计），这些方法难以解决模拟与现实之间的差距，并且缺乏可扩展性。尽管大规模模仿预训练前景广阔，但在数据稀缺的情况下将这些通用策略适应特定任务仍待探索。为此，我们提出了 ControlVLA，这是一种新的框架，通过 ControlNet 风格的架构将预训练的 VLA 模型与物体中心表示相结合，以实现高效的微调。具体而言，为了引入物体中心条件而不覆盖先前知识，ControlVLA 将一系列投影层初始化为零，使其能够逐步适应预训练的 manipulation 策略。在包括倒立方体和折叠衣物在内的 6 个不同任务的现实世界实验中，我们的方法在仅需要 10-20 个示例的情况下实现了 76.7% 的成功率，这远优于传统方法需要超过 100 个示例才能达到类似成功率的情形。额外的实验还突显了 ControlVLA 在长时任务上的可扩展性和对未见过的物体和背景的鲁棒性。 

---
# FlowRAM: Grounding Flow Matching Policy with Region-Aware Mamba Framework for Robotic Manipulation 

**Title (ZH)**: FlowRAM：基于区域意识Mamba框架的流动匹配策略接地方法 

**Authors**: Sen Wang, Le Wang, Sanping Zhou, Jingyi Tian, Jiayi Li, Haowen Sun, Wei Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16201)  

**Abstract**: Robotic manipulation in high-precision tasks is essential for numerous industrial and real-world applications where accuracy and speed are required. Yet current diffusion-based policy learning methods generally suffer from low computational efficiency due to the iterative denoising process during inference. Moreover, these methods do not fully explore the potential of generative models for enhancing information exploration in 3D environments. In response, we propose FlowRAM, a novel framework that leverages generative models to achieve region-aware perception, enabling efficient multimodal information processing. Specifically, we devise a Dynamic Radius Schedule, which allows adaptive perception, facilitating transitions from global scene comprehension to fine-grained geometric details. Furthermore, we integrate state space models to integrate multimodal information, while preserving linear computational complexity. In addition, we employ conditional flow matching to learn action poses by regressing deterministic vector fields, simplifying the learning process while maintaining performance. We verify the effectiveness of the FlowRAM in the RLBench, an established manipulation benchmark, and achieve state-of-the-art performance. The results demonstrate that FlowRAM achieves a remarkable improvement, particularly in high-precision tasks, where it outperforms previous methods by 12.0% in average success rate. Additionally, FlowRAM is able to generate physically plausible actions for a variety of real-world tasks in less than 4 time steps, significantly increasing inference speed. 

**Abstract (ZH)**: 基于流”的生成模型在高精度任务中的机器人操作：实现区域感知的高效多模态信息处理 

---
# Single-Microphone-Based Sound Source Localization for Mobile Robots in Reverberant Environments 

**Title (ZH)**: 基于单麦克风的移动机器人在混响环境中的声源定位 

**Authors**: Jiang Wang, Runwu Shi, Benjamin Yen, He Kong, Kazuhiro Nakadai  

**Link**: [PDF](https://arxiv.org/pdf/2506.16173)  

**Abstract**: Accurately estimating sound source positions is crucial for robot audition. However, existing sound source localization methods typically rely on a microphone array with at least two spatially preconfigured microphones. This requirement hinders the applicability of microphone-based robot audition systems and technologies. To alleviate these challenges, we propose an online sound source localization method that uses a single microphone mounted on a mobile robot in reverberant environments. Specifically, we develop a lightweight neural network model with only 43k parameters to perform real-time distance estimation by extracting temporal information from reverberant signals. The estimated distances are then processed using an extended Kalman filter to achieve online sound source localization. To the best of our knowledge, this is the first work to achieve online sound source localization using a single microphone on a moving robot, a gap that we aim to fill in this work. Extensive experiments demonstrate the effectiveness and merits of our approach. To benefit the broader research community, we have open-sourced our code at this https URL. 

**Abstract (ZH)**: 准确估计声源位置对于机器人听觉至关重要。然而，现有的声源定位方法通常依赖于包含至少两个空间预配置麦克风的麦克风阵列。这一要求限制了基于麦克风的机器人听觉系统和技术的适用性。为了解决这些挑战，我们提出了一种在线声源定位方法，该方法在回声环境中使用安装在移动机器人上的单个麦克风进行声源定位。具体而言，我们开发了一个仅包含43K参数的轻量级神经网络模型，通过从回声信号中提取时域信息来实时估计距离。然后使用扩展的卡尔曼滤波器处理这些估计的距离，以实现在线声源定位。据我们所知，这是首次使用移动机器人上的单个麦克风实现在线声源定位的工作，我们希望填补这一空白。大量实验表明了我们方法的有效性和优点。为了惠及更广泛的科研界，我们在以下网址开源了我们的代码：this https URL。 

---
# From Theory to Practice: Identifying the Optimal Approach for Offset Point Tracking in the Context of Agricultural Robotics 

**Title (ZH)**: 从理论到实践：在农业机器人背景下确定最优偏移点跟踪方法的研究 

**Authors**: Stephane Ngnepiepaye Wembe, Vincent Rousseau, Johann Laconte, Roland Lenain  

**Link**: [PDF](https://arxiv.org/pdf/2506.16143)  

**Abstract**: Modern agriculture faces escalating challenges: increasing demand for food, labor shortages, and the urgent need to reduce environmental impact. Agricultural robotics has emerged as a promising response to these pressures, enabling the automation of precise and suitable field operations. In particular, robots equipped with implements for tasks such as weeding or sowing must interact delicately and accurately with the crops and soil. Unlike robots in other domains, these agricultural platforms typically use rigidly mounted implements, where the implement's position is more critical than the robot's center in determining task success. Yet, most control strategies in the literature focus on the vehicle body, often neglecting the acctual working point of the system. This is particularly important when considering new agriculture practices where crops row are not necessary straights. This paper presents a predictive control strategy targeting the implement's reference point. The method improves tracking performance by anticipating the motion of the implement, which, due to its offset from the vehicle's center of rotation, is prone to overshooting during turns if not properly accounted for. 

**Abstract (ZH)**: 现代农业面临着 escalating 挑战：不断增长的食品需求、劳动力短缺以及迫切需要减少环境影响。农业机器人作为应对这些压力的前景广泛的方法之一，能够实现精确和适宜的田间操作自动化。特别是，用于除草或播种等任务的机器人必须与作物和土壤进行细致而准确的交互。与其它领域中的机器人不同，这些农业平台通常使用刚性安装的工具，工具的位置比机器人的中心更为关键，决定了任务的成功。然而，文献中的大多数控制策略侧重于车辆主体，往往忽视了系统的实际工作点。当考虑新的农业实践时，尤为重要的是作物行不一定必须是直线的。本文提出了一种针对工具参考点的预测控制策略，该方法通过预测工具的运动来提高跟踪性能，因为工具相对于车辆旋转中心的位置偏移，在转弯时如果没有适当考虑，容易导致过冲。 

---
# Investigating Lagrangian Neural Networks for Infinite Horizon Planning in Quadrupedal Locomotion 

**Title (ZH)**: 基于拉格朗日神经网络的四足行走无限_horizon规划研究 

**Authors**: Prakrut Kotecha, Aditya Shirwatkar, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2506.16079)  

**Abstract**: Lagrangian Neural Networks (LNNs) present a principled and interpretable framework for learning the system dynamics by utilizing inductive biases. While traditional dynamics models struggle with compounding errors over long horizons, LNNs intrinsically preserve the physical laws governing any system, enabling accurate and stable predictions essential for sustainable locomotion. This work evaluates LNNs for infinite horizon planning in quadrupedal robots through four dynamics models: (1) full-order forward dynamics (FD) training and inference, (2) diagonalized representation of Mass Matrix in full order FD, (3) full-order inverse dynamics (ID) training with FD inference, (4) reduced-order modeling via torso centre-of-mass (CoM) dynamics. Experiments demonstrate that LNNs bring improvements in sample efficiency (10x) and superior prediction accuracy (up to 2-10x) compared to baseline methods. Notably, the diagonalization approach of LNNs reduces computational complexity while retaining some interpretability, enabling real-time receding horizon control. These findings highlight the advantages of LNNs in capturing the underlying structure of system dynamics in quadrupeds, leading to improved performance and efficiency in locomotion planning and control. Additionally, our approach achieves a higher control frequency than previous LNN methods, demonstrating its potential for real-world deployment on quadrupeds. 

**Abstract (ZH)**: Lagrangian 神经网络 (LNNs) 为通过利用归纳偏置学习系统动力学提供了一个原则性和可解释性的框架。尽管传统动力模型在长期预测中面临着累积误差的问题，LNNs 本质上能保持任何系统所遵循的物理法则，从而实现对可持续运动至关重要的准确且稳定的预测。本文通过四种动力学模型评估 LNNs 在四足机器人无限 horizon 规划中的应用：(1) 完全套数前向动力学 (FD) 训练和推理，(2) 完全套数质量矩阵对角化表示，(3) 完全套数逆动力学 (ID) 训练与 FD 推理，(4) 通过躯干质心 (CoM) 动力学的降维建模。实验结果表明，与基准方法相比，LNNs 在样本效率 (10 倍) 和预测精度 (最多 2-10 倍) 上带来了改进。值得注意的是，LNNs 的对角化方法降低了计算复杂度并保持了一定的可解释性，使其能够实现实时后退预测视野控制。这些发现突显了 LNNs 在捕获四足动物系统动力学底层结构方面的优势，从而提高了运动规划和控制的性能和效率。此外，我们的方法实现的控制频率高于之前的 LNN 方法，展示了其在四足动物的实际部署中的潜力。 

---
# Noise Fusion-based Distillation Learning for Anomaly Detection in Complex Industrial Environments 

**Title (ZH)**: 基于噪声融合的精炼学习方法在复杂工业环境中的异常检测 

**Authors**: Jiawen Yu, Jieji Ren, Yang Chang, Qiaojun Yu, Xuan Tong, Boyang Wang, Yan Song, You Li, Xinji Mai, Wenqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16050)  

**Abstract**: Anomaly detection and localization in automated industrial manufacturing can significantly enhance production efficiency and product quality. Existing methods are capable of detecting surface defects in pre-defined or controlled imaging environments. However, accurately detecting workpiece defects in complex and unstructured industrial environments with varying views, poses and illumination remains challenging. We propose a novel anomaly detection and localization method specifically designed to handle inputs with perturbative patterns. Our approach introduces a new framework based on a collaborative distillation heterogeneous teacher network (HetNet), an adaptive local-global feature fusion module, and a local multivariate Gaussian noise generation module. HetNet can learn to model the complex feature distribution of normal patterns using limited information about local disruptive changes. We conducted extensive experiments on mainstream benchmarks. HetNet demonstrates superior performance with approximately 10% improvement across all evaluation metrics on MSC-AD under industrial conditions, while achieving state-of-the-art results on other datasets, validating its resilience to environmental fluctuations and its capability to enhance the reliability of industrial anomaly detection systems across diverse scenarios. Tests in real-world environments further confirm that HetNet can be effectively integrated into production lines to achieve robust and real-time anomaly detection. Codes, images and videos are published on the project website at: this https URL 

**Abstract (ZH)**: 自动化工业制造中的异常检测和定位可以显著提高生产效率和产品质量。现有的方法能够在预定义或受控的成像环境中检测表面缺陷。然而，在复杂且非结构化的工业环境中，面对不同视角、姿态和照明条件导致的缺陷检测仍然具有挑战性。我们提出了一种新型的异常检测和定位方法，专门用于处理具有扰动模式的输入。我们的方法引入了一种基于协作式蒸馏异构教师网络（HetNet）的新框架、自适应局部-全局特征融合模块以及局部多元高斯噪声生成模块。HetNet能够利用有限的局部扰动变化信息学习建模正常模式的复杂特征分布。我们在主流基准上进行了广泛的实验。在工业条件下，HetNet在MSC-AD上的所有评估指标上表现出约10%的性能提升，而在其他数据集上则达到了最先进的结果，验证了其对环境波动的抗性和在各种场景下增强工业异常检测系统的可靠性。在真实环境中的测试进一步证实HetNet可以有效集成到生产线上，实现稳健且实时的异常检测。相关代码、图像和视频已在项目网站上发布：this https URL。 

---
# DualTHOR: A Dual-Arm Humanoid Simulation Platform for Contingency-Aware Planning 

**Title (ZH)**: DualTHOR：一种面向 contingencies 意识规划的双臂仿人模拟平台 

**Authors**: Boyu Li, Siyuan He, Hang Xu, Haoqi Yuan, Yu Zang, Liwei Hu, Junpeng Yue, Zhenxiong Jiang, Pengbo Hu, Börje F. Karlsson, Yehui Tang, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16012)  

**Abstract**: Developing embodied agents capable of performing complex interactive tasks in real-world scenarios remains a fundamental challenge in embodied AI. Although recent advances in simulation platforms have greatly enhanced task diversity to train embodied Vision Language Models (VLMs), most platforms rely on simplified robot morphologies and bypass the stochastic nature of low-level execution, which limits their transferability to real-world robots. To address these issues, we present a physics-based simulation platform DualTHOR for complex dual-arm humanoid robots, built upon an extended version of AI2-THOR. Our simulator includes real-world robot assets, a task suite for dual-arm collaboration, and inverse kinematics solvers for humanoid robots. We also introduce a contingency mechanism that incorporates potential failures through physics-based low-level execution, bridging the gap to real-world scenarios. Our simulator enables a more comprehensive evaluation of the robustness and generalization of VLMs in household environments. Extensive evaluations reveal that current VLMs struggle with dual-arm coordination and exhibit limited robustness in realistic environments with contingencies, highlighting the importance of using our simulator to develop more capable VLMs for embodied tasks. The code is available at this https URL. 

**Abstract (ZH)**: 在现实世界场景中开发能够执行复杂交互任务的具身代理仍然是具身AI领域的基本挑战。尽管近期在模拟平台方面的进展极大地增强了训练具身视觉语言模型（VLMs）的任务多样性，但大多数平台依赖于简化的人形机器人形态，并规避了低级执行的随机性质，这限制了它们向实际机器人转移的能力。为了解决这些问题，我们提出了一个基于物理的模拟平台DualTHOR，该平台适用于复杂双臂人形机器人，基于AI2-THOR的扩展版本。我们的模拟器包括真实机器人资产、双臂协作任务套件以及人形机器人逆运动学求解器。我们还引入了一种应急机制，通过基于物理的低级执行来整合潜在的失败情况，从而填补与现实世界场景之间的差距。我们的模拟器使得更具身环境中的VLMs的鲁棒性和泛化能力评估更加全面。广泛评估表明，当前的VLMs难以应对双臂协调，并且在具有应急情况的现实环境中表现出有限的鲁棒性，强调了使用我们的模拟器开发更加能够胜任的VLMs的重要性。代码可在该网址获取。 

---
# A Low-Cost Portable Lidar-based Mobile Mapping System on an Android Smartphone 

**Title (ZH)**: 一种基于Android智能手机的低成本便携式LiDAR移动 Mapping 系统 

**Authors**: Jianzhu Huai, Yuxin Shao, Yujia Zhang, Alper Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.15983)  

**Abstract**: The rapid advancement of the metaverse, digital twins, and robotics underscores the demand for low-cost, portable mapping systems for reality capture. Current mobile solutions, such as the Leica BLK2Go and lidar-equipped smartphones, either come at a high cost or are limited in range and accuracy. Leveraging the proliferation and technological evolution of mobile devices alongside recent advancements in lidar technology, we introduce a novel, low-cost, portable mobile mapping system. Our system integrates a lidar unit, an Android smartphone, and an RTK-GNSS stick. Running on the Android platform, it features lidar-inertial odometry built with the NDK, and logs data from the lidar, wide-angle camera, IMU, and GNSS. With a total bill of materials (BOM) cost under 2,000 USD and a weight of about 1 kilogram, the system achieves a good balance between affordability and portability. We detail the system design, multisensor calibration, synchronization, and evaluate its performance for tracking and mapping. To further contribute to the community, the system's design and software are made open source at: this https URL 

**Abstract (ZH)**: 虚拟现实、数字孪生和机器人技术的快速发展突显了对低成本、便携式现实捕捉测绘系统的 Demand。当前的移动解决方案，如 Leica BLK2Go 和配备激光雷达的智能手机，要么成本高昂，要么在范围和精度上有限制。借助移动设备的普及和技术演变以及近年来激光雷达技术的进步，我们引入了一种新型低成本便携式移动测绘系统。该系统集成了激光雷达单元、Android智能手机和RTK-GNSS天线杆。基于Android平台，该系统使用NDK构建了激光雷达-惯性 odometry，并记录来自激光雷达、广角相机、IMU和GNSS的数据。该系统的材料成本低于2000美元，重量约为1千克，实现了成本效益和便携性的良好平衡。我们详细介绍了系统设计、多传感器标定、同步及其在追踪和测绘中的性能评估。为进一步贡献社区，该系统的硬件设计和软件已在以下链接开源：this https URL。 

---
# ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation 

**Title (ZH)**: ViTacFormer: 学习跨模态表示以实现视觉-触觉灵巧操作 

**Authors**: Liang Heng, Haoran Geng, Kaifeng Zhang, Pieter Abbeel, Jitendra Malik  

**Link**: [PDF](https://arxiv.org/pdf/2506.15953)  

**Abstract**: Dexterous manipulation is a cornerstone capability for robotic systems aiming to interact with the physical world in a human-like manner. Although vision-based methods have advanced rapidly, tactile sensing remains crucial for fine-grained control, particularly in unstructured or visually occluded settings. We present ViTacFormer, a representation-learning approach that couples a cross-attention encoder to fuse high-resolution vision and touch with an autoregressive tactile prediction head that anticipates future contact signals. Building on this architecture, we devise an easy-to-challenging curriculum that steadily refines the visual-tactile latent space, boosting both accuracy and robustness. The learned cross-modal representation drives imitation learning for multi-fingered hands, enabling precise and adaptive manipulation. Across a suite of challenging real-world benchmarks, our method achieves approximately 50% higher success rates than prior state-of-the-art systems. To our knowledge, it is also the first to autonomously complete long-horizon dexterous manipulation tasks that demand highly precise control with an anthropomorphic hand, successfully executing up to 11 sequential stages and sustaining continuous operation for 2.5 minutes. 

**Abstract (ZH)**: 灵巧操作是旨在以类人方式与物理世界互动的机器人系统的一个核心能力。尽管基于视觉的方法已经迅速发展，但在精细控制方面，特别是在无结构或视觉遮挡的环境中，触觉感知仍然至关重要。我们提出了一种名为ViTacFormer的表示学习方法，该方法结合了跨注意力编码器以融合高分辨率视觉和触觉信息，并配备了一个自回归触觉预测头部，可以预见未来的接触信号。在此架构基础上，我们设计了一种从易到难的课程学习，逐步优化视觉-触觉潜在空间，从而提升准确性和鲁棒性。学习到的跨模态表示驱动多指手的模仿学习，使其能够实现精确和适应性的操作。在一系列具有挑战性的现实世界基准测试中，我们的方法实现了比之前的最先进的系统约50%更高的成功率。据我们所知，这也是第一个能够自主完成需要具有高度精确控制的长时间 horizon 灵巧操作任务的人形手，成功执行了多达11个连续阶段，并持续运行2.5分钟。 

---
# KARL: Kalman-Filter Assisted Reinforcement Learner for Dynamic Object Tracking and Grasping 

**Title (ZH)**: KARL：卡尔曼滤波辅助强化学习者用于动态物体跟踪与抓取 

**Authors**: Kowndinya Boyalakuntla, Abdeslam Boularias, Jingjin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15945)  

**Abstract**: We present Kalman-filter Assisted Reinforcement Learner (KARL) for dynamic object tracking and grasping over eye-on-hand (EoH) systems, significantly expanding such systems capabilities in challenging, realistic environments. In comparison to the previous state-of-the-art, KARL (1) incorporates a novel six-stage RL curriculum that doubles the system's motion range, thereby greatly enhancing the system's grasping performance, (2) integrates a robust Kalman filter layer between the perception and reinforcement learning (RL) control modules, enabling the system to maintain an uncertain but continuous 6D pose estimate even when the target object temporarily exits the camera's field-of-view or undergoes rapid, unpredictable motion, and (3) introduces mechanisms to allow retries to gracefully recover from unavoidable policy execution failures. Extensive evaluations conducted in both simulation and real-world experiments qualitatively and quantitatively corroborate KARL's advantage over earlier systems, achieving higher grasp success rates and faster robot execution speed. Source code and supplementary materials for KARL will be made available at: this https URL. 

**Abstract (ZH)**: 基于卡尔曼滤波辅助强化学习的动态物体跟踪与抓取系统（KARL）：扩展眼随手系统在挑战性现实环境中的能力 

---
# Learning from Planned Data to Improve Robotic Pick-and-Place Planning Efficiency 

**Title (ZH)**: 基于计划数据的学习以提高机器人上下料规划效率 

**Authors**: Liang Qin, Weiwei Wan, Jun Takahashi, Ryo Negishi, Masaki Matsushita, Kensuke Harada  

**Link**: [PDF](https://arxiv.org/pdf/2506.15920)  

**Abstract**: This work proposes a learning method to accelerate robotic pick-and-place planning by predicting shared grasps. Shared grasps are defined as grasp poses feasible to both the initial and goal object configurations in a pick-and-place task. Traditional analytical methods for solving shared grasps evaluate grasp candidates separately, leading to substantial computational overhead as the candidate set grows. To overcome the limitation, we introduce an Energy-Based Model (EBM) that predicts shared grasps by combining the energies of feasible grasps at both object poses. This formulation enables early identification of promising candidates and significantly reduces the search space. Experiments show that our method improves grasp selection performance, offers higher data efficiency, and generalizes well to unseen grasps and similarly shaped objects. 

**Abstract (ZH)**: 本研究提出了一种学习方法，通过预测共享握持方式来加速机器人取放规划。共享握持方式被定义为在取放任务中初始和目标对象配置都可行的握持姿态。传统分析方法求解共享握持方式时，分别评估每个握持候选方案，导致随着候选方案集的增加，计算开销显著增大。为克服这一限制，我们引入了一种能量模型（EBM），通过结合两种对象姿态下可行握持的能量来预测共享握持方式。该表述能在早期识别出有前景的候选方案，并显著缩小搜索空间。实验结果显示，本方法提高了握持选择性能，具有更高的数据效率，并能很好地泛化到未见过的握持方式和形状相似的物体。 

---
# Advancing Autonomous Racing: A Comprehensive Survey of the RoboRacer (F1TENTH) Platform 

**Title (ZH)**: 自动驾驶赛车进展：RoboRacer（F1TENTH）平台综述 

**Authors**: Israel Charles, Hossein Maghsoumi, Yaser Fallah  

**Link**: [PDF](https://arxiv.org/pdf/2506.15899)  

**Abstract**: The RoboRacer (F1TENTH) platform has emerged as a leading testbed for advancing autonomous driving research, offering a scalable, cost-effective, and community-driven environment for experimentation. This paper presents a comprehensive survey of the platform, analyzing its modular hardware and software architecture, diverse research applications, and role in autonomous systems education. We examine critical aspects such as bridging the simulation-to-reality (Sim2Real) gap, integration with simulation environments, and the availability of standardized datasets and benchmarks. Furthermore, the survey highlights advancements in perception, planning, and control algorithms, as well as insights from global competitions and collaborative research efforts. By consolidating these contributions, this study positions RoboRacer as a versatile framework for accelerating innovation and bridging the gap between theoretical research and real-world deployment. The findings underscore the platform's significance in driving forward developments in autonomous racing and robotics. 

**Abstract (ZH)**: RoboRacer (F1TENTH) 平台：一个促进自主驾驶研究的领先测试床，及其在自主系统教育中的作用综述 

---
# Challenges and Research Directions from the Operational Use of a Machine Learning Damage Assessment System via Small Uncrewed Aerial Systems at Hurricanes Debby and Helene 

**Title (ZH)**: 小型无人航空系统在飓风德贝和海伦中运用的机器学习损伤评估系统的操作使用挑战及研究方向 

**Authors**: Thomas Manzini, Priyankari Perali, Robin R. Murphy, David Merrick  

**Link**: [PDF](https://arxiv.org/pdf/2506.15890)  

**Abstract**: This paper details four principal challenges encountered with machine learning (ML) damage assessment using small uncrewed aerial systems (sUAS) at Hurricanes Debby and Helene that prevented, degraded, or delayed the delivery of data products during operations and suggests three research directions for future real-world deployments. The presence of these challenges is not surprising given that a review of the literature considering both datasets and proposed ML models suggests this is the first sUAS-based ML system for disaster damage assessment actually deployed as a part of real-world operations. The sUAS-based ML system was applied by the State of Florida to Hurricanes Helene (2 orthomosaics, 3.0 gigapixels collected over 2 sorties by a Wintra WingtraOne sUAS) and Debby (1 orthomosaic, 0.59 gigapixels collected via 1 sortie by a Wintra WingtraOne sUAS) in Florida. The same model was applied to crewed aerial imagery of inland flood damage resulting from post-tropical remnants of Hurricane Debby in Pennsylvania (436 orthophotos, 136.5 gigapixels), providing further insights into the advantages and limitations of sUAS for disaster response. The four challenges (variationin spatial resolution of input imagery, spatial misalignment between imagery and geospatial data, wireless connectivity, and data product format) lead to three recommendations that specify research needed to improve ML model capabilities to accommodate the wide variation of potential spatial resolutions used in practice, handle spatial misalignment, and minimize the dependency on wireless connectivity. These recommendations are expected to improve the effective operational use of sUAS and sUAS-based ML damage assessment systems for disaster response. 

**Abstract (ZH)**: 本文详细分析了在飓风黛比和海伦期间使用小型无人 aerial 系统（sUAS）进行机器学习（ML）损伤评估时遇到的四个主要挑战，这些挑战妨碍、降低了或延迟了数据产品的交付，并提出了三个未来实际部署中的研究方向。鉴于文献审查表明，这是首个在实际灾害响应操作中部署的基于sUAS的ML系统，这些挑战的存在并不令人惊讶。基于sUAS的ML系统分别应用于佛罗里达州的飓风海伦（2张正射影像，采集数据量3.0吉像素，通过两次飞行任务由Wintra WingtraOne sUAS完成）和黛比（1张正射影像，0.59吉像素，通过一次飞行任务由Wintra WingtraOne sUAS采集），以及宾夕法尼亚州由后热带残余的黛比引起的内陆洪水灾害的有照侦察（436张正射影像，136.5吉像素）。这进一步揭示了sUAS在灾害响应中的优势与局限性。四个挑战（输入影像的空间分辨率变异、影像与地理空间数据的空间错位、无线连接以及数据产品格式）导致三个建议，这些建议详细规定了需进行的研究以改进ML模型的能力，以适应实际操作中使用的广泛空间分辨率变异、处理空间错位，并减少对无线连接的依赖。这些建议有望提高sUAS及其基于sUAS的ML损伤评估系统在灾害响应中的有效操作使用。 

---
# A Small-Scale Robot for Autonomous Driving: Design, Challenges, and Best Practices 

**Title (ZH)**: 小型自主驾驶机器人：设计、挑战及最佳实践 

**Authors**: Hossein Maghsoumi, Yaser Fallah  

**Link**: [PDF](https://arxiv.org/pdf/2506.15870)  

**Abstract**: Small-scale autonomous vehicle platforms provide a cost-effective environment for developing and testing advanced driving systems. However, specific configurations within this scale are underrepresented, limiting full awareness of their potential. This paper focuses on a one-sixth-scale setup, offering a high-level overview of its design, hardware and software integration, and typical challenges encountered during development. We discuss methods for addressing mechanical and electronic issues common to this scale and propose guidelines for improving reliability and performance. By sharing these insights, we aim to expand the utility of small-scale vehicles for testing autonomous driving algorithms and to encourage further research in this domain. 

**Abstract (ZH)**: 小型自主车辆平台为高级驾驶系统的研究与测试提供了一种经济高效的环境。然而，该规模下的特定配置研究不足，限制了其潜在价值的全面认识。本文专注于六分之一规模的搭建，从设计、硬件与软件集成以及开发过程中遇到的典型挑战等方面进行高层次概述。我们讨论了针对该规模常见机械和电气问题的解决方法，并提出了提高可靠性和性能的建议。通过分享这些见解，我们旨在扩大小型车辆在测试自主驾驶算法方面的应用范围，并鼓励在该领域进行进一步研究。 

---
# CooperRisk: A Driving Risk Quantification Pipeline with Multi-Agent Cooperative Perception and Prediction 

**Title (ZH)**: CooperRisk: 基于多智能体协同感知与预测的驾驶风险量化管道 

**Authors**: Mingyue Lei, Zewei Zhou, Hongchen Li, Jia Hu, Jiaqi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.15868)  

**Abstract**: Risk quantification is a critical component of safe autonomous driving, however, constrained by the limited perception range and occlusion of single-vehicle systems in complex and dense scenarios. Vehicle-to-everything (V2X) paradigm has been a promising solution to sharing complementary perception information, nevertheless, how to ensure the risk interpretability while understanding multi-agent interaction with V2X remains an open question. In this paper, we introduce the first V2X-enabled risk quantification pipeline, CooperRisk, to fuse perception information from multiple agents and quantify the scenario driving risk in future multiple timestamps. The risk is represented as a scenario risk map to ensure interpretability based on risk severity and exposure, and the multi-agent interaction is captured by the learning-based cooperative prediction model. We carefully design a risk-oriented transformer-based prediction model with multi-modality and multi-agent considerations. It aims to ensure scene-consistent future behaviors of multiple agents and avoid conflicting predictions that could lead to overly conservative risk quantification and cause the ego vehicle to become overly hesitant to drive. Then, the temporal risk maps could serve to guide a model predictive control planner. We evaluate the CooperRisk pipeline in a real-world V2X dataset V2XPnP, and the experiments demonstrate its superior performance in risk quantification, showing a 44.35% decrease in conflict rate between the ego vehicle and background traffic participants. 

**Abstract (ZH)**: 基于V2X的风险量化管道CooperRisk：多代理交互的可解释风险评估与预测 

---
# Improving Robotic Manipulation: Techniques for Object Pose Estimation, Accommodating Positional Uncertainty, and Disassembly Tasks from Examples 

**Title (ZH)**: 改进机器人操作:基于例证的物体姿态估计、位置不确定性处理及拆解任务方法研究 

**Authors**: Viral Rasik Galaiya  

**Link**: [PDF](https://arxiv.org/pdf/2506.15865)  

**Abstract**: To use robots in more unstructured environments, we have to accommodate for more complexities. Robotic systems need more awareness of the environment to adapt to uncertainty and variability. Although cameras have been predominantly used in robotic tasks, the limitations that come with them, such as occlusion, visibility and breadth of information, have diverted some focus to tactile sensing. In this thesis, we explore the use of tactile sensing to determine the pose of the object using the temporal features. We then use reinforcement learning with tactile collisions to reduce the number of attempts required to grasp an object resulting from positional uncertainty from camera estimates. Finally, we use information provided by these tactile sensors to a reinforcement learning agent to determine the trajectory to take to remove an object from a restricted passage while reducing training time by pertaining from human examples. 

**Abstract (ZH)**: 为了在更多非结构化环境中使用机器人，我们需要应对更多的复杂性。机器人系统需要增加对环境的意识以适应不确定性与变化性。虽然相机在机器人任务中被广泛使用，但它们带来的遮挡、可见范围和信息广度的限制促使一些研究转向触觉感知。在本文中，我们探索使用触觉感知通过时间特征来确定物体的姿态。然后，我们利用基于触觉碰撞的强化学习来减少因相机估计位置不确定性而导致的抓取物体的尝试次数。最后，我们利用这些触觉传感器提供的信息来为强化学习代理确定轨迹，以从狭窄通道中移除物体，同时通过避免人类示例来减少训练时间。 

---
# Semantic and Feature Guided Uncertainty Quantification of Visual Localization for Autonomous Vehicles 

**Title (ZH)**: 基于语义和特征引导的不确定性量化在自动驾驶车辆视觉定位中的应用 

**Authors**: Qiyuan Wu, Mark Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2506.15851)  

**Abstract**: The uncertainty quantification of sensor measurements coupled with deep learning networks is crucial for many robotics systems, especially for safety-critical applications such as self-driving cars. This paper develops an uncertainty quantification approach in the context of visual localization for autonomous driving, where locations are selected based on images. Key to our approach is to learn the measurement uncertainty using light-weight sensor error model, which maps both image feature and semantic information to 2-dimensional error distribution. Our approach enables uncertainty estimation conditioned on the specific context of the matched image pair, implicitly capturing other critical, unannotated factors (e.g., city vs highway, dynamic vs static scenes, winter vs summer) in a latent manner. We demonstrate the accuracy of our uncertainty prediction framework using the Ithaca365 dataset, which includes variations in lighting and weather (sunny, night, snowy). Both the uncertainty quantification of the sensor+network is evaluated, along with Bayesian localization filters using unique sensor gating method. Results show that the measurement error does not follow a Gaussian distribution with poor weather and lighting conditions, and is better predicted by our Gaussian Mixture model. 

**Abstract (ZH)**: 传感器测量与深度学习网络结合的不确定性量化对于许多机器人系统至关重要，尤其是在自动驾驶等安全关键应用中。本文在视觉定位的自主驾驶背景下开发了一种不确定性量化方法，其中位置基于图像选择。我们的方法的关键在于使用轻量化传感器误差模型学习测量不确定性，该模型将图像特征和语义信息映射到二维误差分布。我们的方法能够在匹配图像对的具体上下文中进行不确定性估计，隐含地捕捉其他关键但未标注的因素（例如城市 vs 高速公路、动态 vs 静态场景、冬季 vs 夏季）。我们使用Ithaca365数据集展示了我们的不确定性预测框架的准确性，该数据集包含不同光照和天气条件（晴天、夜晚、雪天）。我们评估了传感器+网络的不确定性量化以及使用唯一传感器门控方法的贝叶斯定位滤波器。结果表明，在恶劣天气和光照条件下，测量误差不遵循正态分布，并且我们的混合高斯模型能够更好地预测这些误差。 

---
# PRISM-Loc: a Lightweight Long-range LiDAR Localization in Urban Environments with Topological Maps 

**Title (ZH)**: PRISM-Loc：基于拓扑地图的城市环境轻量级远距离LiDAR定位 

**Authors**: Kirill Muravyev, Vasily Yuryev, Oleg Bulichev, Dmitry Yudin, Konstantin Yakovlev  

**Link**: [PDF](https://arxiv.org/pdf/2506.15849)  

**Abstract**: Localization in the environment is one of the crucial tasks of navigation of a mobile robot or a self-driving vehicle. For long-range routes, performing localization within a dense global lidar map in real time may be difficult, and the creation of such a map may require much memory. To this end, leveraging topological maps may be useful. In this work, we propose PRISM-Loc -- a topological map-based approach for localization in large environments. The proposed approach leverages a twofold localization pipeline, which consists of global place recognition and estimation of the local pose inside the found location. For local pose estimation, we introduce an original lidar scan matching algorithm, which is based on 2D features and point-based optimization. We evaluate the proposed method on the ITLP-Campus dataset on a 3 km route, and compare it against the state-of-the-art metric map-based and place recognition-based competitors. The results of the experiments show that the proposed method outperforms its competitors both quality-wise and computationally-wise. 

**Abstract (ZH)**: 基于拓扑图的大型环境定位方法PRISM-Loc 

---
# SafeMimic: Towards Safe and Autonomous Human-to-Robot Imitation for Mobile Manipulation 

**Title (ZH)**: SafeMimic: 通往自主安全的人机imitation移动 manipulation的途径 

**Authors**: Arpit Bahety, Arnav Balaji, Ben Abbatematteo, Roberto Martín-Martín  

**Link**: [PDF](https://arxiv.org/pdf/2506.15847)  

**Abstract**: For robots to become efficient helpers in the home, they must learn to perform new mobile manipulation tasks simply by watching humans perform them. Learning from a single video demonstration from a human is challenging as the robot needs to first extract from the demo what needs to be done and how, translate the strategy from a third to a first-person perspective, and then adapt it to be successful with its own morphology. Furthermore, to mitigate the dependency on costly human monitoring, this learning process should be performed in a safe and autonomous manner. We present SafeMimic, a framework to learn new mobile manipulation skills safely and autonomously from a single third-person human video. Given an initial human video demonstration of a multi-step mobile manipulation task, SafeMimic first parses the video into segments, inferring both the semantic changes caused and the motions the human executed to achieve them and translating them to an egocentric reference. Then, it adapts the behavior to the robot's own morphology by sampling candidate actions around the human ones, and verifying them for safety before execution in a receding horizon fashion using an ensemble of safety Q-functions trained in simulation. When safe forward progression is not possible, SafeMimic backtracks to previous states and attempts a different sequence of actions, adapting both the trajectory and the grasping modes when required for its morphology. As a result, SafeMimic yields a strategy that succeeds in the demonstrated behavior and learns task-specific actions that reduce exploration in future attempts. Our experiments show that our method allows robots to safely and efficiently learn multi-step mobile manipulation behaviors from a single human demonstration, from different users, and in different environments, with improvements over state-of-the-art baselines across seven tasks 

**Abstract (ZH)**: 机器人要在家庭中成为高效的助手，必须学会通过观看人类表演的新移动操作任务来执行这些任务。仅通过一个人类单个视频示范进行学习是具有挑战性的，因为机器人需要首先从示范中提取需要执行的内容及其方式，然后将其策略从第三人称视角转换到第一人称视角，并适应其自身形态以获得成功。此外，为了减少对昂贵的人类监控的依赖，这个学习过程应该以安全和自主的方式进行。我们提出了SafeMimic框架，用于从单个第三人称人类视频中安全、自主地学习新的移动操作技能。给定一个涉及多步骤移动操作任务的初始人类视频示范，SafeMimic首先将视频分解为段落，推断出人类执行的动作及其引起的语义变化，并将其转换为以自我为中心的参考。然后，通过在人类动作周围采样候选动作，并通过一个在仿真中训练的集合安全Q函数以回退视野的方式验证其安全性来适应机器人的形态。当无法安全地向前推进时，SafeMimic会回溯到先前状态，并尝试不同的动作序列，在必要时适应其形态的轨迹和抓取模式。因此，SafeMimic生成了一种策略，使演示的行为得以成功，并在未来的尝试中学习任务特定的动作以减少探索。我们的实验表明，我们的方法允许机器人从单个人类示范中安全高效地学习多步骤移动操作行为，这不仅适用于不同的用户，还适用于不同的环境，并且在七个任务上优于最先进的基准方法。 

---
# Context Matters! Relaxing Goals with LLMs for Feasible 3D Scene Planning 

**Title (ZH)**: 背景至关重要！借助LLMs实现可行的3D场景规划 

**Authors**: Emanuele Musumeci, Michele Brienza, Francesco Argenziano, Vincenzo Suriani, Daniele Nardi, Domenico D. Bloisi  

**Link**: [PDF](https://arxiv.org/pdf/2506.15828)  

**Abstract**: Classical planning in AI and Robotics addresses complex tasks by shifting from imperative to declarative approaches (e.g., PDDL). However, these methods often fail in real scenarios due to limited robot perception and the need to ground perceptions to planning predicates. This often results in heavily hard-coded behaviors that struggle to adapt, even with scenarios where goals can be achieved through relaxed planning. Meanwhile, Large Language Models (LLMs) lead to planning systems that leverage commonsense reasoning but often at the cost of generating unfeasible and/or unsafe plans. To address these limitations, we present an approach integrating classical planning with LLMs, leveraging their ability to extract commonsense knowledge and ground actions. We propose a hierarchical formulation that enables robots to make unfeasible tasks tractable by defining functionally equivalent goals through gradual relaxation. This mechanism supports partial achievement of the intended objective, suited to the agent's specific context. Our method demonstrates its ability to adapt and execute tasks effectively within environments modeled using 3D Scene Graphs through comprehensive qualitative and quantitative evaluations. We also show how this method succeeds in complex scenarios where other benchmark methods are more likely to fail. Code, dataset, and additional material are released to the community. 

**Abstract (ZH)**: 经典的AI与机器人规划方法通过从命令式转向声明式方法（例如PDDL）来处理复杂任务，但在实际场景中往往因机器人感知能力有限及需将感知与规划谓词对接而失败，这常常导致行为高度硬编码化，难以适应变化。与此同时，大规模语言模型（LLMs）能够利用常识推理来构建规划系统，但往往会产生不可行的和/或不安全的计划。为解决这些限制，我们提出了一种结合经典规划与LLMs的方法，利用其提取常识知识和对接动作的能力。我们提出了一种分层公式化方法，通过逐步放松定义功能等价的目标使不可行的任务变得可处理。该机制支持根据代理特定环境部分达成预定目标。我们的方法通过使用3D场景图建模环境的全面定性和定量评估展示了其适应性和任务执行能力，并且在其他基准方法更有可能失败的复杂场景中成功。已向社区发布代码、数据集及其他相关材料。 

---
# Steering Your Diffusion Policy with Latent Space Reinforcement Learning 

**Title (ZH)**: 使用潜空间强化学习引导你的扩散策略 

**Authors**: Andrew Wagenmaker, Mitsuhiko Nakamoto, Yunchu Zhang, Seohong Park, Waleed Yagoub, Anusha Nagabandi, Abhishek Gupta, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2506.15799)  

**Abstract**: Robotic control policies learned from human demonstrations have achieved impressive results in many real-world applications. However, in scenarios where initial performance is not satisfactory, as is often the case in novel open-world settings, such behavioral cloning (BC)-learned policies typically require collecting additional human demonstrations to further improve their behavior -- an expensive and time-consuming process. In contrast, reinforcement learning (RL) holds the promise of enabling autonomous online policy improvement, but often falls short of achieving this due to the large number of samples it typically requires. In this work we take steps towards enabling fast autonomous adaptation of BC-trained policies via efficient real-world RL. Focusing in particular on diffusion policies -- a state-of-the-art BC methodology -- we propose diffusion steering via reinforcement learning (DSRL): adapting the BC policy by running RL over its latent-noise space. We show that DSRL is highly sample efficient, requires only black-box access to the BC policy, and enables effective real-world autonomous policy improvement. Furthermore, DSRL avoids many of the challenges associated with finetuning diffusion policies, obviating the need to modify the weights of the base policy at all. We demonstrate DSRL on simulated benchmarks, real-world robotic tasks, and for adapting pretrained generalist policies, illustrating its sample efficiency and effective performance at real-world policy improvement. 

**Abstract (ZH)**: 基于强化学习的快速自主适应行为克隆训练策略 

---
# Robust control for multi-legged elongate robots in noisy environments 

**Title (ZH)**: 多腿长形机器人在噪声环境中的鲁棒控制 

**Authors**: Baxi Chong, Juntao He, Daniel Irvine, Tianyu Wang, Esteban Flores, Daniel Soto, Jianfeng Lin, Zhaochen Xu, Vincent R Nienhusser, Grigoriy Blekherman, Daniel I. Goldman  

**Link**: [PDF](https://arxiv.org/pdf/2506.15788)  

**Abstract**: Modern two and four legged robots exhibit impressive mobility on complex terrain, largely attributed to advancement in learning algorithms. However, these systems often rely on high-bandwidth sensing and onboard computation to perceive/respond to terrain uncertainties. Further, current locomotion strategies typically require extensive robot-specific training, limiting their generalizability across platforms. Building on our prior research connecting robot-environment interaction and communication theory, we develop a new paradigm to construct robust and simply controlled multi-legged elongate robots (MERs) capable of operating effectively in cluttered, unstructured environments. In this framework, each leg-ground contact is thought of as a basic active contact (bac), akin to bits in signal transmission. Reliable locomotion can be achieved in open-loop on "noisy" landscapes via sufficient redundancy in bacs. In such situations, robustness is achieved through passive mechanical responses. We term such processes as those displaying mechanical intelligence (MI) and analogize these processes to forward error correction (FEC) in signal transmission. To augment MI, we develop feedback control schemes, which we refer to as computational intelligence (CI) and such processes analogize automatic repeat request (ARQ) in signal transmission. Integration of these analogies between locomotion and communication theory allow analysis, design, and prediction of embodied intelligence control schemes (integrating MI and CI) in MERs, showing effective and reliable performance (approximately half body lengths per cycle) on complex landscapes with terrain "noise" over twice the robot's height. Our work provides a foundation for systematic development of MER control, paving the way for terrain-agnostic, agile, and resilient robotic systems capable of operating in extreme environments. 

**Abstract (ZH)**: 基于机械智能和计算智能的多足延展机器人在复杂地形上的稳健控制新范式 

---
# Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation 

**Title (ZH)**: 交替自回归运动与场景生成的长期交通仿真 

**Authors**: Xiuyu Yang, Shuhan Tan, Philipp Krähenbühl  

**Link**: [PDF](https://arxiv.org/pdf/2506.17213)  

**Abstract**: An ideal traffic simulator replicates the realistic long-term point-to-point trip that a self-driving system experiences during deployment. Prior models and benchmarks focus on closed-loop motion simulation for initial agents in a scene. This is problematic for long-term simulation. Agents enter and exit the scene as the ego vehicle enters new regions. We propose InfGen, a unified next-token prediction model that performs interleaved closed-loop motion simulation and scene generation. InfGen automatically switches between closed-loop motion simulation and scene generation mode. It enables stable long-term rollout simulation. InfGen performs at the state-of-the-art in short-term (9s) traffic simulation, and significantly outperforms all other methods in long-term (30s) simulation. The code and model of InfGen will be released at this https URL 

**Abstract (ZH)**: 一种理想的交通模拟器能够重现自动驾驶系统部署过程中点对点的真实长周期行程。现有模型和基准主要集中在场景中初始代理的闭环运动模拟。这不利于长周期模拟。随着ego车辆进入新的区域，代理会进入和退出场景。我们提出了一种InfGen统一的下一个token预测模型，该模型同时进行交替的闭环运动模拟和场景生成。InfGen能够在闭环运动模拟和场景生成模式之间自动切换，从而实现稳定的长周期滚动模拟。InfGen在短期内（9秒）的交通模拟中达到最先进的性能，并在长期内（30秒）的模拟中显著优于所有其他方法。InfGen的代码和模型将在以下链接发布：这个 https URL。 

---
# Part$^{2}$GS: Part-aware Modeling of Articulated Objects using 3D Gaussian Splatting 

**Title (ZH)**: Part$^{2}$GS: 使用3D高斯点表示的关节部位建模 

**Authors**: Tianjiao Yu, Vedant Shah, Muntasir Wahed, Ying Shen, Kiet A. Nguyen, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17212)  

**Abstract**: Articulated objects are common in the real world, yet modeling their structure and motion remains a challenging task for 3D reconstruction methods. In this work, we introduce Part$^{2}$GS, a novel framework for modeling articulated digital twins of multi-part objects with high-fidelity geometry and physically consistent articulation. Part$^{2}$GS leverages a part-aware 3D Gaussian representation that encodes articulated components with learnable attributes, enabling structured, disentangled transformations that preserve high-fidelity geometry. To ensure physically consistent motion, we propose a motion-aware canonical representation guided by physics-based constraints, including contact enforcement, velocity consistency, and vector-field alignment. Furthermore, we introduce a field of repel points to prevent part collisions and maintain stable articulation paths, significantly improving motion coherence over baselines. Extensive evaluations on both synthetic and real-world datasets show that Part$^{2}$GS consistently outperforms state-of-the-art methods by up to 10$\times$ in Chamfer Distance for movable parts. 

**Abstract (ZH)**: articulated对象在现实世界中普遍存在，但对其进行结构和运动建模仍然是3D重建方法的一项挑战。本工作中，我们引入了Part$^{2}$GS，这是一种新的框架，用于建模多部件对象的高保真几何和物理一致的articulated数字双胞胎。Part$^{2}$GS利用了部件感知的3D高斯表示，其中编码可学习的articulated组件属性，实现结构化、去纠缠的变换，保持高保真几何。为了确保物理一致的运动，我们提出了一种由基于物理约束指导的运动感知标准表示，包括接触约束、速度一致性和矢量场对齐。此外，我们引入了一种斥力点场，以防止部件碰撞并保持稳定的艺术化路径，显著提高了运动连贯性。在合成和真实世界数据集上的广泛评估表明，Part$^{2}$GS在可移动部件的chamfer距离上一致地优于最先进的方法，最高可提高10倍。 

---
# RGBTrack: Fast, Robust Depth-Free 6D Pose Estimation and Tracking 

**Title (ZH)**: RGBTrack：快速可靠的无深度6D姿态估计与跟踪 

**Authors**: Teng Guo, Jingjin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17119)  

**Abstract**: We introduce a robust framework, RGBTrack, for real-time 6D pose estimation and tracking that operates solely on RGB data, thereby eliminating the need for depth input for such dynamic and precise object pose tracking tasks. Building on the FoundationPose architecture, we devise a novel binary search strategy combined with a render-and-compare mechanism to efficiently infer depth and generate robust pose hypotheses from true-scale CAD models. To maintain stable tracking in dynamic scenarios, including rapid movements and occlusions, RGBTrack integrates state-of-the-art 2D object tracking (XMem) with a Kalman filter and a state machine for proactive object pose recovery. In addition, RGBTrack's scale recovery module dynamically adapts CAD models of unknown scale using an initial depth estimate, enabling seamless integration with modern generative reconstruction techniques. Extensive evaluations on benchmark datasets demonstrate that RGBTrack's novel depth-free approach achieves competitive accuracy and real-time performance, making it a promising practical solution candidate for application areas including robotics, augmented reality, and computer vision.
The source code for our implementation will be made publicly available at this https URL. 

**Abstract (ZH)**: 我们提出了一种鲁棒框架RGBTrack，用于仅基于RGB数据进行实时6D姿态估计与跟踪，从而消除动态且精确物体姿态跟踪任务中对深度输入的需求。该框架基于FoundationPose架构，设计了一种新颖的二分搜索策略结合渲染与比对机制，以高效地从真实尺度的CAD模型中推断深度并生成稳健的姿态假设。为了在包括快速运动和遮挡在内的动态场景中保持稳定的跟踪，RGBTrack将最先进的2D物体跟踪（XMem）与卡尔曼滤波器和状态机集成，以实现前瞻性的物体姿态恢复。此外，RGBTrack的尺度恢复模块能够根据初始深度估计动态适应未知尺度的CAD模型，使其能够无缝集成现代生成重建技术。在基准数据集上的广泛评估表明，RGBTrack的新颖无深度方法在精度和实时性能方面具有竞争力，使其成为机器人技术、增强现实和计算机视觉等领域具有潜力的实用解决方案候选者。源代码将在该网址公开发布：https://example.com。 

---
# Multimodal Fused Learning for Solving the Generalized Traveling Salesman Problem in Robotic Task Planning 

**Title (ZH)**: 多模态融合学习在机器人任务规划中解决广义旅行商问题 

**Authors**: Jiaqi Chen, Mingfeng Fan, Xuefeng Zhang, Jingsong Liang, Yuhong Cao, Guohua Wu, Guillaume Adrien Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2506.16931)  

**Abstract**: Effective and efficient task planning is essential for mobile robots, especially in applications like warehouse retrieval and environmental monitoring. These tasks often involve selecting one location from each of several target clusters, forming a Generalized Traveling Salesman Problem (GTSP) that remains challenging to solve both accurately and efficiently. To address this, we propose a Multimodal Fused Learning (MMFL) framework that leverages both graph and image-based representations to capture complementary aspects of the problem, and learns a policy capable of generating high-quality task planning schemes in real time. Specifically, we first introduce a coordinate-based image builder that transforms GTSP instances into spatially informative representations. We then design an adaptive resolution scaling strategy to enhance adaptability across different problem scales, and develop a multimodal fusion module with dedicated bottlenecks that enables effective integration of geometric and spatial features. Extensive experiments show that our MMFL approach significantly outperforms state-of-the-art methods across various GTSP instances while maintaining the computational efficiency required for real-time robotic applications. Physical robot tests further validate its practical effectiveness in real-world scenarios. 

**Abstract (ZH)**: 有效的多模态融合学习框架对于移动机器人任务规划至关重要，特别是在仓库检索和环境监测等应用中。这些任务通常涉及从多个目标簇中选择一个位置，形成一个通用旅行商问题（GTSP），这一问题在准确性和效率上都极具挑战性。为此，我们提出了一种多模态融合学习（MMFL）框架，利用图和图像表示来捕捉问题的互补方面，并学习一种能够实时生成高质量任务规划方案的策略。具体而言，我们首先介绍了一种基于坐标的图像构建器，将GTSP问题实例转换为空间信息丰富的表示。然后设计了一种自适应分辨率缩放策略以增强不同问题规模下的适应性，并开发了一种具有专用瓶颈的多模态融合模块，以实现几何和空间特征的有效集成。广泛实验表明，我们的MMFL方法在各种GTSP实例中显著优于现有方法，同时保持了实时机器人应用所需的计算效率。实体机器人测试进一步验证了其在实际场景中的有效性和实用性。 

---
# ROS 2 Agnocast: Supporting Unsized Message Types for True Zero-Copy Publish/Subscribe IPC 

**Title (ZH)**: ROS 2 Agnocast: 支持无大小消息类型的真正零拷贝发布/订阅IPC 

**Authors**: Takahiro Ishikawa-Aso, Shinpei Kato  

**Link**: [PDF](https://arxiv.org/pdf/2506.16882)  

**Abstract**: Robot applications, comprising independent components that mutually publish/subscribe messages, are built on inter-process communication (IPC) middleware such as Robot Operating System 2 (ROS 2). In large-scale ROS 2 systems like autonomous driving platforms, true zero-copy communication -- eliminating serialization and deserialization -- is crucial for efficiency and real-time performance. However, existing true zero-copy middleware solutions lack widespread adoption as they fail to meet three essential requirements: 1) Support for all ROS 2 message types including unsized ones; 2) Minimal modifications to existing application code; 3) Selective implementation of zero-copy communication between specific nodes while maintaining conventional communication mechanisms for other inter-node communications including inter-host node communications. This first requirement is critical, as production-grade ROS 2 projects like Autoware rely heavily on unsized message types throughout their codebase to handle diverse use cases (e.g., various sensors), and depend on the broader ROS 2 ecosystem, where unsized message types are pervasive in libraries. The remaining requirements facilitate seamless integration with existing projects. While IceOryx middleware, a practical true zero-copy solution, meets all but the first requirement, other studies achieving the first requirement fail to satisfy the remaining criteria. This paper presents Agnocast, a true zero-copy IPC framework applicable to ROS 2 C++ on Linux that fulfills all these requirements. Our evaluation demonstrates that Agnocast maintains constant IPC overhead regardless of message size, even for unsized message types. In Autoware PointCloud Preprocessing, Agnocast achieves a 16% improvement in average response time and a 25% improvement in worst-case response time. 

**Abstract (ZH)**: 基于ROS 2的真正零拷贝应用：Agnocast框架 

---
# Vision-Based Multirotor Control for Spherical Target Tracking: A Bearing-Angle Approach 

**Title (ZH)**: 基于视觉的多旋翼飞行器球形目标追踪控制：一种方位角方法 

**Authors**: Marcelo Jacinto, Rita Cunha  

**Link**: [PDF](https://arxiv.org/pdf/2506.16870)  

**Abstract**: This work addresses the problem of designing a visual servo controller for a multirotor vehicle, with the end goal of tracking a moving spherical target with unknown radius. To address this problem, we first transform two bearing measurements provided by a camera sensor into a bearing-angle pair. We then use this information to derive the system's dynamics in a new set of coordinates, where the angle measurement is used to quantify a relative distance to the target. Building on this system representation, we design an adaptive nonlinear control algorithm that takes advantage of the properties of the new system geometry and assumes that the target follows a constant acceleration model. Simulation results illustrate the performance of the proposed control algorithm. 

**Abstract (ZH)**: 本研究解决了设计多旋翼飞行器视觉伺服控制器的问题，目的是跟踪一个未知半径的移动球形目标。为此，我们首先将相机传感器提供的两个方位角测量值转换为一个方位角-角度对。然后，利用这一信息在新的坐标系中推导系统的动力学模型，其中的角度测量值用于量化相对于目标的距离。基于这种系统表示，我们设计了一种利用新系统几何特性且假定目标遵循恒定加速度模型的自适应非线性控制算法。仿真结果展示了所提出的控制算法的性能。 

---
# Camera Calibration via Circular Patterns: A Comprehensive Framework with Measurement Uncertainty and Unbiased Projection Model 

**Title (ZH)**: 基于圆特征的相机标定：包含测量不确定性及无偏投影模型的综合框架 

**Authors**: Chaehyeon Song, Dongjae Lee, Jongwoo Lim, Ayoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.16842)  

**Abstract**: Camera calibration using planar targets has been widely favored, and two types of control points have been mainly considered as measurements: the corners of the checkerboard and the centroid of circles. Since a centroid is derived from numerous pixels, the circular pattern provides more precise measurements than the checkerboard. However, the existing projection model of circle centroids is biased under lens distortion, resulting in low performance. To surmount this limitation, we propose an unbiased projection model of the circular pattern and demonstrate its superior accuracy compared to the checkerboard. Complementing this, we introduce uncertainty into circular patterns to enhance calibration robustness and completeness. Defining centroid uncertainty improves the performance of calibration components, including pattern detection, optimization, and evaluation metrics. We also provide guidelines for performing good camera calibration based on the evaluation metric. The core concept of this approach is to model the boundary points of a two-dimensional shape as a Markov random field, considering its connectivity. The shape distribution is propagated to the centroid uncertainty through an appropriate shape representation based on the Green theorem. Consequently, the resulting framework achieves marked gains in calibration accuracy and robustness. The complete source code and demonstration video are available at this https URL. 

**Abstract (ZH)**: 使用平面靶标进行相机校准已被广泛青睐，主要考虑的两种测量控制点是棋盘格的角点和圆的质心。由于质心是从众多像素中得出的，因此圆的模式提供了比棋盘格更精确的测量。然而现有的圆心投影模型在镜头畸变下存在偏差，导致性能较低。为了克服这一限制，我们提出了一个无偏的圆的投影模型，并证明其在标定精度上优于棋盘格。此外，我们引入圆中的不确定性来增强标定的稳健性和完整性。定义质心不确定性可以提高包括图案检测、优化和评估指标在内的标定组件的性能。我们还基于评估指标提供了进行良好相机校准的准则。该方法的核心概念是将二维形状的边界点建模为马尔可夫随机场，考虑其连通性。形状分布通过基于格林定理的适当形状表示传播到质心不确定性中。因此，该框架实现了显著的标定精度和稳健性的提升。完整的源代码和演示视频可在此处访问：this https URL。 

---
# AnyTraverse: An off-road traversability framework with VLM and human operator in the loop 

**Title (ZH)**: AnyTraverse: 带有VLM和人工操作员介入的离路可通过性框架 

**Authors**: Sattwik Sahu, Agamdeep Singh, Karthik Nambiar, Srikanth Saripalli, P.B. Sujit  

**Link**: [PDF](https://arxiv.org/pdf/2506.16826)  

**Abstract**: Off-road traversability segmentation enables autonomous navigation with applications in search-and-rescue, military operations, wildlife exploration, and agriculture. Current frameworks struggle due to significant variations in unstructured environments and uncertain scene changes, and are not adaptive to be used for different robot types. We present AnyTraverse, a framework combining natural language-based prompts with human-operator assistance to determine navigable regions for diverse robotic vehicles. The system segments scenes for a given set of prompts and calls the operator only when encountering previously unexplored scenery or unknown class not part of the prompt in its region-of-interest, thus reducing active supervision load while adapting to varying outdoor scenes. Our zero-shot learning approach eliminates the need for extensive data collection or retraining. Our experimental validation includes testing on RELLIS-3D, Freiburg Forest, and RUGD datasets and demonstrate real-world deployment on multiple robot platforms. The results show that AnyTraverse performs better than GA-NAV and Off-seg while offering a vehicle-agnostic approach to off-road traversability that balances automation with targeted human supervision. 

**Abstract (ZH)**: 基于自然语言提示和人工辅助的离路通行性分割使能自主导航在搜救、军事操作、野生动物探索和农业中的应用 

---
# Off-Policy Actor-Critic for Adversarial Observation Robustness: Virtual Alternative Training via Symmetric Policy Evaluation 

**Title (ZH)**: 基于 off-policy 行为者-评论者的方法：通过对称策略评估实现虚拟替代训练以增强对抗性观测鲁棒性 

**Authors**: Kosuke Nakanishi, Akihiro Kubo, Yuji Yasui, Shin Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2506.16753)  

**Abstract**: Recently, robust reinforcement learning (RL) methods designed to handle adversarial input observations have received significant attention, motivated by RL's inherent vulnerabilities. While existing approaches have demonstrated reasonable success, addressing worst-case scenarios over long time horizons requires both minimizing the agent's cumulative rewards for adversaries and training agents to counteract them through alternating learning. However, this process introduces mutual dependencies between the agent and the adversary, making interactions with the environment inefficient and hindering the development of off-policy methods. In this work, we propose a novel off-policy method that eliminates the need for additional environmental interactions by reformulating adversarial learning as a soft-constrained optimization problem. Our approach is theoretically supported by the symmetric property of policy evaluation between the agent and the adversary. The implementation is available at this https URL. 

**Abstract (ZH)**: 近期，针对对抗性输入观测的鲁棒强化学习方法受到了广泛关注，这源于强化学习固有的脆弱性。尽管现有方法已取得不错的效果，但在长时间尺度上解决最坏情况场景需要同时最小化智能体对敌手的累积奖励，并通过交替学习来训练智能体对抗敌手。然而，这一过程引入了智能体与敌手之间的相互依赖，使得与环境的交互变得低效，并阻碍了脱策学习方法的发展。在这项工作中，我们提出了一种新型脱策学习方法，通过将对抗学习重新表述为软约束优化问题来消除额外环境交互的需要。我们的方法通过智能体和敌手之间策略评估的对称性得到了理论支持。详细实现可参见this https URL。 

---
# IsoNet: Causal Analysis of Multimodal Transformers for Neuromuscular Gesture Classification 

**Title (ZH)**: IsoNet: 多模态变换器的神经肌肉手势分类因果分析 

**Authors**: Eion Tyacke, Kunal Gupta, Jay Patel, Rui Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16744)  

**Abstract**: Hand gestures are a primary output of the human motor system, yet the decoding of their neuromuscular signatures remains a bottleneck for basic neuroscience and assistive technologies such as prosthetics. Traditional human-machine interface pipelines rely on a single biosignal modality, but multimodal fusion can exploit complementary information from sensors. We systematically compare linear and attention-based fusion strategies across three architectures: a Multimodal MLP, a Multimodal Transformer, and a Hierarchical Transformer, evaluating performance on scenarios with unimodal and multimodal inputs. Experiments use two publicly available datasets: NinaPro DB2 (sEMG and accelerometer) and HD-sEMG 65-Gesture (high-density sEMG and force). Across both datasets, the Hierarchical Transformer with attention-based fusion consistently achieved the highest accuracy, surpassing the multimodal and best single-modality linear-fusion MLP baseline by over 10% on NinaPro DB2 and 3.7% on HD-sEMG. To investigate how modalities interact, we introduce an Isolation Network that selectively silences unimodal or cross-modal attention pathways, quantifying each group of token interactions' contribution to downstream decisions. Ablations reveal that cross-modal interactions contribute approximately 30% of the decision signal across transformer layers, highlighting the importance of attention-driven fusion in harnessing complementary modality information. Together, these findings reveal when and how multimodal fusion would enhance biosignal classification and also provides mechanistic insights of human muscle activities. The study would be beneficial in the design of sensor arrays for neurorobotic systems. 

**Abstract (ZH)**: 多模态融合在解码手部手势神经肌电信号中的应用：从单模态到基于注意力的多模态变压器架构的系统比较 

---
# PPTP: Performance-Guided Physiological Signal-Based Trust Prediction in Human-Robot Collaboration 

**Title (ZH)**: 基于生理信号的性能指导型人类与机器人协作信任预测 

**Authors**: Hao Guo, Wei Fan, Shaohui Liu, Feng Jiang, Chunzhi Yi  

**Link**: [PDF](https://arxiv.org/pdf/2506.16677)  

**Abstract**: Trust prediction is a key issue in human-robot collaboration, especially in construction scenarios where maintaining appropriate trust calibration is critical for safety and efficiency. This paper introduces the Performance-guided Physiological signal-based Trust Prediction (PPTP), a novel framework designed to improve trust assessment. We designed a human-robot construction scenario with three difficulty levels to induce different trust states. Our approach integrates synchronized multimodal physiological signals (ECG, GSR, and EMG) with collaboration performance evaluation to predict human trust levels. Individual physiological signals are processed using collaboration performance information as guiding cues, leveraging the standardized nature of collaboration performance to compensate for individual variations in physiological responses. Extensive experiments demonstrate the efficacy of our cross-modality fusion method in significantly improving trust classification performance. Our model achieves over 81% accuracy in three-level trust classification, outperforming the best baseline method by 6.7%, and notably reaches 74.3% accuracy in high-resolution seven-level classification, which is a first in trust prediction research. Ablation experiments further validate the superiority of physiological signal processing guided by collaboration performance assessment. 

**Abstract (ZH)**: 基于绩效引导的生理信号信任预测（PPTP）：一种提高信任评估的方法 

---
# IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks 

**Title (ZH)**: IS-Bench: 评估驱动日常家庭任务的VLM引导式代理的互动安全性 

**Authors**: Xiaoya Lu, Zeren Chen, Xuhao Hu, Yijin Zhou, Weichen Zhang, Dongrui Liu, Lu Sheng, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16402)  

**Abstract**: Flawed planning from VLM-driven embodied agents poses significant safety hazards, hindering their deployment in real-world household tasks. However, existing static, non-interactive evaluation paradigms fail to adequately assess risks within these interactive environments, since they cannot simulate dynamic risks that emerge from an agent's actions and rely on unreliable post-hoc evaluations that ignore unsafe intermediate steps. To bridge this critical gap, we propose evaluating an agent's interactive safety: its ability to perceive emergent risks and execute mitigation steps in the correct procedural order. We thus present IS-Bench, the first multi-modal benchmark designed for interactive safety, featuring 161 challenging scenarios with 388 unique safety risks instantiated in a high-fidelity simulator. Crucially, it facilitates a novel process-oriented evaluation that verifies whether risk mitigation actions are performed before/after specific risk-prone steps. Extensive experiments on leading VLMs, including the GPT-4o and Gemini-2.5 series, reveal that current agents lack interactive safety awareness, and that while safety-aware Chain-of-Thought can improve performance, it often compromises task completion. By highlighting these critical limitations, IS-Bench provides a foundation for developing safer and more reliable embodied AI systems. 

**Abstract (ZH)**: 基于VLM驱动的实体代理的规划缺陷引发了重大的安全风险，阻碍了它们在真实家庭任务中的部署。现有的静态、非互动评估范式无法充分评估这些互动环境中出现的风险，因为它们无法模拟由代理行为引发的动态风险，并依赖于忽略不安全中间步骤的不可靠事后评估。为弥补这一关键缺口，我们提出评估实体代理的互动安全性：其识别新兴风险并按正确流程执行缓解步骤的能力。我们因此介绍了IS-Bench，这是首个针对互动安全设计的多模态基准，包含161个具有388种独特安全风险的高保真模拟器中的挑战性场景。 crucially，它支持一种新颖的过程导向评估，验证风险缓解行动是否在特定风险易发步骤之前/之后执行。对领先的VLM的广泛实验，包括GPT-4o和Gemini-2.5系列，表明当前的代理缺乏互动安全性意识，虽然安全意识的推理链可以提高性能，但往往会牺牲任务完成度。通过突出这些关键局限性，IS-Bench为开发更安全、更可靠的实体AI系统提供了基础。 

---
# Dense 3D Displacement Estimation for Landslide Monitoring via Fusion of TLS Point Clouds and Embedded RGB Images 

**Title (ZH)**: 基于TLS点云和嵌入RGB图像融合的滑坡监测密集3D位移估计 

**Authors**: Zhaoyi Wang, Jemil Avers Butt, Shengyu Huang, Tomislav Medic, Andreas Wieser  

**Link**: [PDF](https://arxiv.org/pdf/2506.16265)  

**Abstract**: Landslide monitoring is essential for understanding geohazards and mitigating associated risks. However, existing point cloud-based methods typically rely on either geometric or radiometric information and often yield sparse or non-3D displacement estimates. In this paper, we propose a hierarchical partition-based coarse-to-fine approach that fuses 3D point clouds and co-registered RGB images to estimate dense 3D displacement vector fields. We construct patch-level matches using both 3D geometry and 2D image features. These matches are refined via geometric consistency checks, followed by rigid transformation estimation per match. Experimental results on two real-world landslide datasets demonstrate that our method produces 3D displacement estimates with high spatial coverage (79% and 97%) and high accuracy. Deviations in displacement magnitude with respect to external measurements (total station or GNSS observations) are 0.15 m and 0.25 m on the two datasets, respectively, and only 0.07 m and 0.20 m compared to manually derived references. These values are below the average scan resolutions (0.08 m and 0.30 m). Our method outperforms the state-of-the-art method F2S3 in spatial coverage while maintaining comparable accuracy. Our approach offers a practical and adaptable solution for TLS-based landslide monitoring and is extensible to other types of point clouds and monitoring tasks. Our example data and source code are publicly available at this https URL. 

**Abstract (ZH)**: 基于层次分区的粗细结合方法：融合三维点云和共注册RGB图像进行密集三维位移场估计 

---
# Human-Centered Shared Autonomy for Motor Planning, Learning, and Control Applications 

**Title (ZH)**: 以人为本的共享自治在运动规划、学习与控制中的应用 

**Authors**: MH Farhadi, Ali Rabiee, Sima Ghafoori, Anna Cetera, Wei Xu, Reza Abiri  

**Link**: [PDF](https://arxiv.org/pdf/2506.16044)  

**Abstract**: With recent advancements in AI and computational tools, intelligent paradigms have emerged to enhance fields like shared autonomy and human-machine teaming in healthcare. Advanced AI algorithms (e.g., reinforcement learning) can autonomously make decisions to achieve planning and motion goals. However, in healthcare, where human intent is crucial, fully independent machine decisions may not be ideal. This chapter presents a comprehensive review of human-centered shared autonomy AI frameworks, focusing on upper limb biosignal-based machine interfaces and associated motor control systems, including computer cursors, robotic arms, and planar platforms. We examine motor planning, learning (rehabilitation), and control, covering conceptual foundations of human-machine teaming in reach-and-grasp tasks and analyzing both theoretical and practical implementations. Each section explores how human and machine inputs can be blended for shared autonomy in healthcare applications. Topics include human factors, biosignal processing for intent detection, shared autonomy in brain-computer interfaces (BCI), rehabilitation, assistive robotics, and Large Language Models (LLMs) as the next frontier. We propose adaptive shared autonomy AI as a high-performance paradigm for collaborative human-AI systems, identify key implementation challenges, and outline future directions, particularly regarding AI reasoning agents. This analysis aims to bridge neuroscientific insights with robotics to create more intuitive, effective, and ethical human-machine teaming frameworks. 

**Abstract (ZH)**: 近年来，随着人工智能和计算工具的进步，智能 paradigms 已在增强医疗健康领域的共享自主性和人机协同方面崭露头角。高级 AI 算法（例如强化学习）可以自主作出决策以实现规划和运动目标。然而，在医疗健康领域，由于人类意图至关重要，完全独立的机器决策可能并不理想。本章综述了以人类为中心的共享自主性 AI 框架，重点关注基于上肢生物信号的机器界面及相关运动控制系统，包括计算机光标、机器人手臂和平面平台。本文探讨了运动规划、学习（康复）和控制，涵盖接近-抓取任务中的人机协同概念基础，并分析了理论和实践实现。每个部分都探讨了如何在医疗应用中将人类和机器输入结合起来实现共享自主性。主题包括人因工程、生物信号处理以探测意图、脑机接口（BCI）中的共享自主性、康复、辅助机器人以及大语言模型（LLMs）作为下一个前沿领域。我们提出了适应性共享自主性 AI 作为协作人-机系统高性能范式，确定了关键实现挑战，并概述了未来方向，特别是关于 AI 推理代理。本分析旨在将神经科学洞察与机器人技术相结合，创建更具直观性、有效性与伦理性的医疗人机协同框架。 

---
# EndoMUST: Monocular Depth Estimation for Robotic Endoscopy via End-to-end Multi-step Self-supervised Training 

**Title (ZH)**: EndoMUST：基于端到端多步自监督训练的单目深度估计在机器人内窥镜中的应用 

**Authors**: Liangjing Shao, Linxin Bai, Chenkang Du, Xinrong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.16017)  

**Abstract**: Monocular depth estimation and ego-motion estimation are significant tasks for scene perception and navigation in stable, accurate and efficient robot-assisted endoscopy. To tackle lighting variations and sparse textures in endoscopic scenes, multiple techniques including optical flow, appearance flow and intrinsic image decomposition have been introduced into the existing methods. However, the effective training strategy for multiple modules are still critical to deal with both illumination issues and information interference for self-supervised depth estimation in endoscopy. Therefore, a novel framework with multistep efficient finetuning is proposed in this work. In each epoch of end-to-end training, the process is divided into three steps, including optical flow registration, multiscale image decomposition and multiple transformation alignments. At each step, only the related networks are trained without interference of irrelevant information. Based on parameter-efficient finetuning on the foundation model, the proposed method achieves state-of-the-art performance on self-supervised depth estimation on SCARED dataset and zero-shot depth estimation on Hamlyn dataset, with 4\%$\sim$10\% lower error. The evaluation code of this work has been published on this https URL. 

**Abstract (ZH)**: 单目深度估计和自我运动估计是稳定、准确、高效机器人辅助内窥镜场景感知与导航的重要任务。为此，在处理内窥镜场景中的光照变化和稀疏纹理时，已将光流、外观流和固有图像分解等多种技术引入现有方法。然而，如何有效训练多个模块以解决照明问题和信息干扰对于自监督深度估计仍至关重要。因此，本文提出了一种具有多步骤高效微调的新框架。在端到端训练的每个周期中，过程分为三步：光流注册、多尺度图像分解和多种变换对齐。在每一步中，只训练相关的网络以避免无关信息的干扰。基于对基础模型的参数高效微调，所提出的方法在SCARED数据集上的自监督深度估计和Hamlyn数据集上的零样本深度估计中均取得了最佳性能，误差降低了4%~10%。本文的评估代码已发布在https://...。 

---
# Quantum Artificial Intelligence for Secure Autonomous Vehicle Navigation: An Architectural Proposal 

**Title (ZH)**: 量子人工智能在安全自主车辆导航中的架构提案 

**Authors**: Hemanth Kannamarlapudi, Sowmya Chintalapudi  

**Link**: [PDF](https://arxiv.org/pdf/2506.16000)  

**Abstract**: Navigation is a very crucial aspect of autonomous vehicle ecosystem which heavily relies on collecting and processing large amounts of data in various states and taking a confident and safe decision to define the next vehicle maneuver. In this paper, we propose a novel architecture based on Quantum Artificial Intelligence by enabling quantum and AI at various levels of navigation decision making and communication process in Autonomous vehicles : Quantum Neural Networks for multimodal sensor fusion, Nav-Q for Quantum reinforcement learning for navigation policy optimization and finally post-quantum cryptographic protocols for secure communication. Quantum neural networks uses quantum amplitude encoding to fuse data from various sensors like LiDAR, radar, camera, GPS and weather etc., This approach gives a unified quantum state representation between heterogeneous sensor modalities. Nav-Q module processes the fused quantum states through variational quantum circuits to learn optimal navigation policies under swift dynamic and complex conditions. Finally, post quantum cryptographic protocols are used to secure communication channels for both within vehicle communication and V2X (Vehicle to Everything) communications and thus secures the autonomous vehicle communication from both classical and quantum security threats. Thus, the proposed framework addresses fundamental challenges in autonomous vehicles navigation by providing quantum performance and future proof security. Index Terms Quantum Computing, Autonomous Vehicles, Sensor Fusion 

**Abstract (ZH)**: 量子人工智能驱动的自动驾驶导航新型架构：量子神经网络赋能多模传感器融合、Nav-Q量子强化学习优化导航策略及后量子加密协议保障安全通信 

---
# Adversarial Attacks and Detection in Visual Place Recognition for Safer Robot Navigation 

**Title (ZH)**: 视觉地点识别中的对抗性攻击与检测以实现更安全的机器人导航 

**Authors**: Connor Malone, Owen Claxton, Iman Shames, Michael Milford  

**Link**: [PDF](https://arxiv.org/pdf/2506.15988)  

**Abstract**: Stand-alone Visual Place Recognition (VPR) systems have little defence against a well-designed adversarial attack, which can lead to disastrous consequences when deployed for robot navigation. This paper extensively analyzes the effect of four adversarial attacks common in other perception tasks and four novel VPR-specific attacks on VPR localization performance. We then propose how to close the loop between VPR, an Adversarial Attack Detector (AAD), and active navigation decisions by demonstrating the performance benefit of simulated AADs in a novel experiment paradigm -- which we detail for the robotics community to use as a system framework. In the proposed experiment paradigm, we see the addition of AADs across a range of detection accuracies can improve performance over baseline; demonstrating a significant improvement -- such as a ~50% reduction in the mean along-track localization error -- can be achieved with True Positive and False Positive detection rates of only 75% and up to 25% respectively. We examine a variety of metrics including: Along-Track Error, Percentage of Time Attacked, Percentage of Time in an `Unsafe' State, and Longest Continuous Time Under Attack. Expanding further on these results, we provide the first investigation into the efficacy of the Fast Gradient Sign Method (FGSM) adversarial attack for VPR. The analysis in this work highlights the need for AADs in real-world systems for trustworthy navigation, and informs quantitative requirements for system design. 

**Abstract (ZH)**: 独立视觉位置识别(VPR)系统对精心设计的对抗攻击缺乏防御能力，这可能导致灾难性的后果，特别是在机器人导航中部署时。本文广泛分析了四种常见于其他感知任务的对抗攻击和四种新型VPR特定攻击对VPR定位性能的影响。随后，我们提出了一种通过展示模拟对抗攻击检测器(AAD)在一种新型实验范式中的性能优势（我们对此范式进行了详细说明，供机器人社区作为系统框架使用）来闭环VPR和主动导航决策的方法。在所提出的实验范式中，我们发现不同检测准确度的AADs的添加可以提高性能；仅通过True Positive和False Positive检测率分别为75%和25%即可实现显著的性能提升，如沿航向定位误差降低约50%。我们考察了包含以下指标在内的多种指标：沿航向误差、受攻击时间百分比、不可用状态时间百分比以及连续受攻击时间最长。进一步扩展这些结果，我们首次探讨了快速梯度符号方法(FGSM)对抗攻击在VPR中的效用。本文的分析强调了在实际系统中使用对抗攻击检测器以实现值得信赖的导航的必要性，并提供了系统设计的量化要求。 

---
# Contactless Precision Steering of Particles in a Fluid inside a Cube with Rotating Walls 

**Title (ZH)**: 腔内旋转壁对流体中颗粒的无接触精确转向 

**Authors**: Lucas Amoudruz, Petr Karnakov, Petros Koumoutsakos  

**Link**: [PDF](https://arxiv.org/pdf/2506.15958)  

**Abstract**: Contactless manipulation of small objects is essential for biomedical and chemical applications, such as cell analysis, assisted fertilisation, and precision chemistry. Established methods, including optical, acoustic, and magnetic tweezers, are now complemented by flow control techniques that use flow-induced motion to enable precise and versatile manipulation. However, trapping multiple particles in fluid remains a challenge. This study introduces a novel control algorithm capable of steering multiple particles in flow. The system uses rotating disks to generate flow fields that transport particles to precise locations. Disk rotations are governed by a feedback control policy based on the Optimising a Discrete Loss (ODIL) framework, which combines fluid dynamics equations with path objectives into a single loss function. Our experiments, conducted in both simulations and with the physical device, demonstrate the capability of the approach to transport two beads simultaneously to predefined locations, advancing robust contactless particle manipulation for biomedical applications. 

**Abstract (ZH)**: 接触less操控小型物体对于生物医学和化学应用至关重要，如细胞分析、辅助受精和精密化学。已有方法，包括光学、声学和磁力镊子，现在被流动控制技术所补充，后者通过流动诱导的运动来实现精确和多功能的操控。然而，流体中捕获多个颗粒仍然是一项挑战。本研究介绍了一种新型控制算法，能够引导流动中的多个颗粒。该系统使用旋转盘生成流场，将颗粒输送到精确位置。盘的旋转由基于优化离散损失（ODIL）框架的反馈控制策略指导，该策略将流体力学方程与路径目标结合成单一的损失函数。我们在模拟和物理设备实验中演示了该方法同时将两个微珠运送到预定义位置的能力，从而推进了生物医学应用中稳健的接触less颗粒操控。 

---
# Optimal Navigation in Microfluidics via the Optimization of a Discrete Loss 

**Title (ZH)**: 通过离散损失优化实现的微流控最优导航 

**Authors**: Petr Karnakov, Lucas Amoudruz, Petros Koumoutsakos  

**Link**: [PDF](https://arxiv.org/pdf/2506.15902)  

**Abstract**: Optimal path planning and control of microscopic devices navigating in fluid environments is essential for applications ranging from targeted drug delivery to environmental monitoring. These tasks are challenging due to the complexity of microdevice-flow interactions. We introduce a closed-loop control method that optimizes a discrete loss (ODIL) in terms of dynamics and path objectives. In comparison with reinforcement learning, ODIL is more robust, up to three orders faster, and excels in high-dimensional action/state spaces, making it a powerful tool for navigating complex flow environments. 

**Abstract (ZH)**: 微观设备在流体环境中导航的最优路径规划与控制对于从靶向药物递送到环境监测等应用至关重要。由于微设备-流体相互作用的复杂性，这些任务具有挑战性。我们提出了一种闭环控制方法，基于动力学和路径目标优化离散损失（ODIL）。与强化学习相比，ODIL更稳健，速度快三个数量级，并且在高维动作/状态空间中表现出色，使其成为导航复杂流场环境的有力工具。 

---
