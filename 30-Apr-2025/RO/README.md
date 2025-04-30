# XPG-RL: Reinforcement Learning with Explainable Priority Guidance for Efficiency-Boosted Mechanical Search 

**Title (ZH)**: XPG-RL: 具有可解释优先级指导的增强学习以提高机械搜索效率 

**Authors**: Yiting Zhang, Shichen Li, Elena Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2504.20969)  

**Abstract**: Mechanical search (MS) in cluttered environments remains a significant challenge for autonomous manipulators, requiring long-horizon planning and robust state estimation under occlusions and partial observability. In this work, we introduce XPG-RL, a reinforcement learning framework that enables agents to efficiently perform MS tasks through explainable, priority-guided decision-making based on raw sensory inputs. XPG-RL integrates a task-driven action prioritization mechanism with a learned context-aware switching strategy that dynamically selects from a discrete set of action primitives such as target grasping, occlusion removal, and viewpoint adjustment. Within this strategy, a policy is optimized to output adaptive threshold values that govern the discrete selection among action primitives. The perception module fuses RGB-D inputs with semantic and geometric features to produce a structured scene representation for downstream decision-making. Extensive experiments in both simulation and real-world settings demonstrate that XPG-RL consistently outperforms baseline methods in task success rates and motion efficiency, achieving up to 4.5$\times$ higher efficiency in long-horizon tasks. These results underscore the benefits of integrating domain knowledge with learnable decision-making policies for robust and efficient robotic manipulation. 

**Abstract (ZH)**: 基于可解释优先级引导的感知与学习框架（XPG-RL）在复杂环境中的机械搜索任务 

---
# Opinion-Driven Decision-Making for Multi-Robot Navigation through Narrow Corridors 

**Title (ZH)**: 基于意见驱动的多机器人在狭窄走廊导航的决策方法 

**Authors**: Norah K. Alghamdi, Shinkyu Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.20947)  

**Abstract**: We propose an opinion-driven navigation framework for multi-robot traversal through a narrow corridor. Our approach leverages a multi-agent decision-making model known as the Nonlinear Opinion Dynamics (NOD) to address the narrow corridor passage problem, formulated as a multi-robot navigation game. By integrating the NOD model with a multi-robot path planning algorithm, we demonstrate that the framework effectively reduces the likelihood of deadlocks during corridor traversal. To ensure scalability with an increasing number of robots, we introduce a game reduction technique that enables efficient coordination in larger groups. Extensive simulation studies are conducted to validate the effectiveness of the proposed approach. 

**Abstract (ZH)**: 基于意见驱动的多robot在狭窄走廊导航框架 

---
# Bayesian Optimization-based Tire Parameter and Uncertainty Estimation for Real-World Data 

**Title (ZH)**: 基于贝叶斯优化的轮胎参数及不确定性估计方法研究 

**Authors**: Sven Goblirsch, Benedikt Ruhland, Johannes Betz, Markus Lienkamp  

**Link**: [PDF](https://arxiv.org/pdf/2504.20863)  

**Abstract**: This work presents a methodology to estimate tire parameters and their uncertainty using a Bayesian optimization approach. The literature mainly considers the estimation of tire parameters but lacks an evaluation of the parameter identification quality and the required slip ratios for an adequate model fit. Therefore, we examine the use of Stochastical Variational Inference as a methodology to estimate both - the parameters and their uncertainties. We evaluate the method compared to a state-of-the-art Nelder-Mead algorithm for theoretical and real-world application. The theoretical study considers parameter fitting at different slip ratios to evaluate the required excitation for an adequate fitting of each parameter. The results are compared to a sensitivity analysis for a Pacejka Magic Formula tire model. We show the application of the algorithm on real-world data acquired during the Abu Dhabi Autonomous Racing League and highlight the uncertainties in identifying the curvature and shape parameters due to insufficient excitation. The gathered insights can help assess the acquired data's limitations and instead utilize standardized parameters until higher slip ratios are captured. We show that our proposed method can be used to assess the mean values and the uncertainties of tire model parameters in real-world conditions and derive actions for the tire modeling based on our simulative study. 

**Abstract (ZH)**: 本研究提出了一种使用贝叶斯优化方法估计轮胎参数及其不确定性的方法。文献中主要考虑了轮胎参数的估计，但缺乏对参数识别质量的评估以及适应模型所需的适当滑移比。因此，我们探讨了使用随机变分推断作为同时估计参数及其不确定性的方法。我们将该方法与当前最先进的Nelder-Mead算法进行了比较，用于理论和实际应用场景。理论研究考虑了在不同滑移比下的参数拟合，以评估每个参数适当地拟合所需的激励。结果与Pacejka魔术公式轮胎模型的敏感性分析进行了比较。我们展示了该算法在阿布扎比自主赛车联赛获取的真实数据中的应用，并突出了因激励不足而难以识别曲率和形状参数的不确定性。所得见解有助于评估所获取数据的局限性，并在此基础上使用标准化参数直到捕捉到更高的滑移比。我们证明了本方法可以在实际条件下评估轮胎模型参数的均值和不确定，并根据模拟研究推导出轮胎建模的行动方案。 

---
# SoccerDiffusion: Toward Learning End-to-End Humanoid Robot Soccer from Gameplay Recordings 

**Title (ZH)**: SoccerDiffusion: 从 gameplay 录像中学习端到端的人形机器人足球 

**Authors**: Florian Vahl, Jörn Griepenburg, Jan Gutsche, Jasper Güldenstein, Jianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20808)  

**Abstract**: This paper introduces SoccerDiffusion, a transformer-based diffusion model designed to learn end-to-end control policies for humanoid robot soccer directly from real-world gameplay recordings. Using data collected from RoboCup competitions, the model predicts joint command trajectories from multi-modal sensor inputs, including vision, proprioception, and game state. We employ a distillation technique to enable real-time inference on embedded platforms that reduces the multi-step diffusion process to a single step. Our results demonstrate the model's ability to replicate complex motion behaviors such as walking, kicking, and fall recovery both in simulation and on physical robots. Although high-level tactical behavior remains limited, this work provides a robust foundation for subsequent reinforcement learning or preference optimization methods. We release the dataset, pretrained models, and code under: this https URL 

**Abstract (ZH)**: 基于变换器的扩散模型SoccerDiffusion及其在 humanoid 机器人足球中的端到端控制策略学习 

---
# Confidence-based Intent Prediction for Teleoperation in Bimanual Robotic Suturing 

**Title (ZH)**: 基于置信度的双臂机器人缝合远程操作意图预测 

**Authors**: Zhaoyang Jacopo Hu, Haozheng Xu, Sion Kim, Yanan Li, Ferdinando Rodriguez y Baena, Etienne Burdet  

**Link**: [PDF](https://arxiv.org/pdf/2504.20761)  

**Abstract**: Robotic-assisted procedures offer enhanced precision, but while fully autonomous systems are limited in task knowledge, difficulties in modeling unstructured environments, and generalisation abilities, fully manual teleoperated systems also face challenges such as delay, stability, and reduced sensory information. To address these, we developed an interactive control strategy that assists the human operator by predicting their motion plan at both high and low levels. At the high level, a surgeme recognition system is employed through a Transformer-based real-time gesture classification model to dynamically adapt to the operator's actions, while at the low level, a Confidence-based Intention Assimilation Controller adjusts robot actions based on user intent and shared control paradigms. The system is built around a robotic suturing task, supported by sensors that capture the kinematics of the robot and task dynamics. Experiments across users with varying skill levels demonstrated the effectiveness of the proposed approach, showing statistically significant improvements in task completion time and user satisfaction compared to traditional teleoperation. 

**Abstract (ZH)**: 辅助机器人手术Procedure提供增强的精确度，但全自主系统在任务知识、环境建模能力和泛化能力方面受限，而全手动遥操作系统则面临延迟、稳定性和减少的感官信息等问题。为解决这些问题，我们开发了一种交互控制策略，该策略通过高、低层预测操作员的动作计划来辅助人类操作员。在高层，通过基于Transformer的实时手势分类模型实现手术元识别系统，以动态适应操作员的动作；在低层，基于信心程度的意图同化控制器根据用户意图和共享控制范式调整机器人动作。该系统基于一个机器人缝合任务构建，支持捕捉机器人和任务动力学的传感器。针对不同技能水平的用户进行的实验证明了所提方法的有效性，相比传统遥操作，显示出统计显著的任务完成时间和用户满意度的改善。 

---
# A Survey on Event-based Optical Marker Systems 

**Title (ZH)**: 基于事件的光学标记系统综述 

**Authors**: Nafiseh Jabbari Tofighi, Maxime Robic, Fabio Morbidi, Pascal Vasseur  

**Link**: [PDF](https://arxiv.org/pdf/2504.20736)  

**Abstract**: The advent of event-based cameras, with their low latency, high dynamic range, and reduced power consumption, marked a significant change in robotic vision and machine perception. In particular, the combination of these neuromorphic sensors with widely-available passive or active optical markers (e.g. AprilTags, arrays of blinking LEDs), has recently opened up a wide field of possibilities. This survey paper provides a comprehensive review on Event-Based Optical Marker Systems (EBOMS). We analyze the basic principles and technologies on which these systems are based, with a special focus on their asynchronous operation and robustness against adverse lighting conditions. We also describe the most relevant applications of EBOMS, including object detection and tracking, pose estimation, and optical communication. The article concludes with a discussion of possible future research directions in this rapidly-emerging and multidisciplinary field. 

**Abstract (ZH)**: 基于事件的光学标记系统：原理、技术与应用综述 

---
# Learning a General Model: Folding Clothing with Topological Dynamics 

**Title (ZH)**: 学习通用模型：基于拓扑动力学的衣物折叠 

**Authors**: Yiming Liu, Lijun Han, Enlin Gu, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20720)  

**Abstract**: The high degrees of freedom and complex structure of garments present significant challenges for clothing manipulation. In this paper, we propose a general topological dynamics model to fold complex clothing. By utilizing the visible folding structure as the topological skeleton, we design a novel topological graph to represent the clothing state. This topological graph is low-dimensional and applied for complex clothing in various folding states. It indicates the constraints of clothing and enables predictions regarding clothing movement. To extract graphs from self-occlusion, we apply semantic segmentation to analyze the occlusion relationships and decompose the clothing structure. The decomposed structure is then combined with keypoint detection to generate the topological graph. To analyze the behavior of the topological graph, we employ an improved Graph Neural Network (GNN) to learn the general dynamics. The GNN model can predict the deformation of clothing and is employed to calculate the deformation Jacobi matrix for control. Experiments using jackets validate the algorithm's effectiveness to recognize and fold complex clothing with self-occlusion. 

**Abstract (ZH)**: 高自由度和复杂结构的服装对人体形操作提出显著挑战。本文提出了一种通用拓扑动力学模型以折叠复杂服装。通过利用可见折叠结构作为拓扑骨架，我们设计了一种新型拓扑图来表示服装状态。该拓扑图低维度且适用于各种折叠状态的复杂服装，可以表明服装的约束并使服装运动预测成为可能。为了从自遮挡中提取图形，我们应用语义分割来分析遮挡关系并分解服装结构。分解后的结构随后与关键点检测结合以生成拓扑图。为了分析拓扑图的行为，我们采用改进的图神经网络（GNN）来学习一般动力学。GNN模型可以预测服装变形并用于计算变形雅可比矩阵以进行控制。使用夹克进行的实验验证了该算法在识别和折叠具有自遮挡的复杂服装时的有效性。 

---
# Identifying Uncertainty in Self-Adaptive Robotics with Large Language Models 

**Title (ZH)**: 使用大型语言模型识别自适应机器人中的不确定性 

**Authors**: Hassan Sartaj, Jalil Boudjadar, Mirgita Frasheri, Shaukat Ali, Peter Gorm Larsen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20684)  

**Abstract**: Future self-adaptive robots are expected to operate in highly dynamic environments while effectively managing uncertainties. However, identifying the sources and impacts of uncertainties in such robotic systems and defining appropriate mitigation strategies is challenging due to the inherent complexity of self-adaptive robots and the lack of comprehensive knowledge about the various factors influencing uncertainty. Hence, practitioners often rely on intuition and past experiences from similar systems to address uncertainties. In this article, we evaluate the potential of large language models (LLMs) in enabling a systematic and automated approach to identify uncertainties in self-adaptive robotics throughout the software engineering lifecycle. For this evaluation, we analyzed 10 advanced LLMs with varying capabilities across four industrial-sized robotics case studies, gathering the practitioners' perspectives on the LLM-generated responses related to uncertainties. Results showed that practitioners agreed with 63-88% of the LLM responses and expressed strong interest in the practicality of LLMs for this purpose. 

**Abstract (ZH)**: 未来自适应机器人预计能够在高度动态环境中运作，同时有效管理不确定性。然而，识别此类机器人系统中的不确定性来源及其影响，并定义适当的缓解策略非常具有挑战性，因为自适应机器人本身固有的复杂性以及对其各种影响不确定性因素的了解不足。因此，实践者常常依赖直觉和来自类似系统的过往经验来应对不确定性。在本文中，我们评估了大型语言模型（LLMs）在软件开发生命周期中系统化和自动化识别自适应机器人中不确定性方面的潜力。为此，我们分析了四个工业规模的机器人案例研究中的10种具有不同能力的先进LLM，并收集了实践者对LLM生成的关于不确定性的响应的意见。结果显示，实践者同意63-88%的LLM响应，并对该LLMs在该领域的实用性表现出强烈兴趣。 

---
# Multi-Sensor Fusion for Quadruped Robot State Estimation using Invariant Filtering and Smoothing 

**Title (ZH)**: 基于不变滤波和平滑的四足机器人状态估计多传感器融合 

**Authors**: Ylenia Nisticò, Hajun Kim, João Carlos Virgolino Soares, Geoff Fink, Hae-Won Park, Claudio Semini  

**Link**: [PDF](https://arxiv.org/pdf/2504.20615)  

**Abstract**: This letter introduces two multi-sensor state estimation frameworks for quadruped robots, built on the Invariant Extended Kalman Filter (InEKF) and Invariant Smoother (IS). The proposed methods, named E-InEKF and E-IS, fuse kinematics, IMU, LiDAR, and GPS data to mitigate position drift, particularly along the z-axis, a common issue in proprioceptive-based approaches. We derived observation models that satisfy group-affine properties to integrate LiDAR odometry and GPS into InEKF and IS. LiDAR odometry is incorporated using Iterative Closest Point (ICP) registration on a parallel thread, preserving the computational efficiency of proprioceptive-based state estimation. We evaluate E-InEKF and E-IS with and without exteroceptive sensors, benchmarking them against LiDAR-based odometry methods in indoor and outdoor experiments using the KAIST HOUND2 robot. Our methods achieve lower Relative Position Errors (RPE) and significantly reduce Absolute Trajectory Error (ATE), with improvements of up to 28% indoors and 40% outdoors compared to LIO-SAM and FAST-LIO2. Additionally, we compare E-InEKF and E-IS in terms of computational efficiency and accuracy. 

**Abstract (ZH)**: 基于不变扩展卡尔曼滤波器和不变平滑器的四肢机器人多传感器状态估计框架 

---
# Hydra: Marker-Free RGB-D Hand-Eye Calibration 

**Title (ZH)**: Hydra: 无标记点的RGB-D 手眼标定 

**Authors**: Martin Huber, Huanyu Tian, Christopher E. Mower, Lucas-Raphael Müller, Sébastien Ourselin, Christos Bergeles, Tom Vercauteren  

**Link**: [PDF](https://arxiv.org/pdf/2504.20584)  

**Abstract**: This work presents an RGB-D imaging-based approach to marker-free hand-eye calibration using a novel implementation of the iterative closest point (ICP) algorithm with a robust point-to-plane (PTP) objective formulated on a Lie algebra. Its applicability is demonstrated through comprehensive experiments using three well known serial manipulators and two RGB-D cameras. With only three randomly chosen robot configurations, our approach achieves approximately 90% successful calibrations, demonstrating 2-3x higher convergence rates to the global optimum compared to both marker-based and marker-free baselines. We also report 2 orders of magnitude faster convergence time (0.8 +/- 0.4 s) for 9 robot configurations over other marker-free methods. Our method exhibits significantly improved accuracy (5 mm in task space) over classical approaches (7 mm in task space) whilst being marker-free. The benchmarking dataset and code are open sourced under Apache 2.0 License, and a ROS 2 integration with robot abstraction is provided to facilitate deployment. 

**Abstract (ZH)**: 基于RGB-D成像的无标记手眼标定方法：结合Lie代数上鲁棒的点到平面目标函数的迭代最近点算法新实现 

---
# PRISM: Projection-based Reward Integration for Scene-Aware Real-to-Sim-to-Real Transfer with Few Demonstrations 

**Title (ZH)**: PRISM: 基于投影的奖励整合方法在少量示例下的场景感知实域到仿真域再到实域的迁移学习 

**Authors**: Haowen Sun, Han Wang, Chengzhong Ma, Shaolong Zhang, Jiawei Ye, Xingyu Chen, Xuguang Lan  

**Link**: [PDF](https://arxiv.org/pdf/2504.20520)  

**Abstract**: Learning from few demonstrations to develop policies robust to variations in robot initial positions and object poses is a problem of significant practical interest in robotics. Compared to imitation learning, which often struggles to generalize from limited samples, reinforcement learning (RL) can autonomously explore to obtain robust behaviors. Training RL agents through direct interaction with the real world is often impractical and unsafe, while building simulation environments requires extensive manual effort, such as designing scenes and crafting task-specific reward functions. To address these challenges, we propose an integrated real-to-sim-to-real pipeline that constructs simulation environments based on expert demonstrations by identifying scene objects from images and retrieving their corresponding 3D models from existing libraries. We introduce a projection-based reward model for RL policy training that is supervised by a vision-language model (VLM) using human-guided object projection relationships as prompts, with the policy further fine-tuned using expert demonstrations. In general, our work focuses on the construction of simulation environments and RL-based policy training, ultimately enabling the deployment of reliable robotic control policies in real-world scenarios. 

**Abstract (ZH)**: 从有限示范中学习以开发鲁棒性强于机器人初始位置和物体姿态变化的政策是机器人技术中一个重要实用问题。与难以从有限样本来泛化的模仿学习相比，强化学习（RL）能够自主探索以获得鲁棒性行为。直接通过与真实世界交互训练RL代理常常不切实际且不安全，而构建仿真环境则需要大量手工努力，例如设计场景和构建特定任务的奖励函数。为应对这些挑战，我们提出了一种集成的从现实到仿真再到现实的工作流程，该流程基于专家示范识别场景物体并从现有库中检索其对应的3D模型来构建仿真环境。我们引入了一种基于投影的奖励模型用于RL策略训练，该模型由视觉-语言模型（VLM）监督，并使用人类指导的对象投影关系作为提示，策略进一步通过专家示范进行微调。总体而言，我们的工作集中在仿真环境的构建和基于RL的策略训练，最终能够在实际场景中部署可靠的机器人控制策略。 

---
# SPARK Hand: Scooping-Pinching Adaptive Robotic Hand with Kempe Mechanism for Vertical Passive Grasp in Environmental Constraints 

**Title (ZH)**: SPARK 手爪：基于 Kempe 机构的适应性抓取手爪，适用于环境约束下的垂直被动抓取 

**Authors**: Jiaqi Yin, Tianyi Bi, Wenzeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20506)  

**Abstract**: This paper presents the SPARK finger, an innovative passive adaptive robotic finger capable of executing both parallel pinching and scooping grasps. The SPARK finger incorporates a multi-link mechanism with Kempe linkages to achieve a vertical linear fingertip trajectory. Furthermore, a parallelogram linkage ensures the fingertip maintains a fixed orientation relative to the base, facilitating precise and stable manipulation. By integrating these mechanisms with elastic elements, the design enables effective interaction with surfaces, such as tabletops, to handle challenging objects. The finger employs a passive switching mechanism that facilitates seamless transitions between pinching and scooping modes, adapting automatically to various object shapes and environmental constraints without additional actuators. To demonstrate its versatility, the SPARK Hand, equipped with two SPARK fingers, has been developed. This system exhibits enhanced grasping performance and stability for objects of diverse sizes and shapes, particularly thin and flat objects that are traditionally challenging for conventional grippers. Experimental results validate the effectiveness of the SPARK design, highlighting its potential for robotic manipulation in constrained and dynamic environments. 

**Abstract (ZH)**: SPARK指尖：一种创新的被动自适应机械手指，兼具平行夹持和舀取抓握能力 

---
# Combining Quality of Service and System Health Metrics in MAPE-K based ROS Systems through Behavior Trees 

**Title (ZH)**: 基于行为树结合服务质量与系统健康指标的MAPE-K机制在ROS系统中的应用 

**Authors**: Andreas Wiedholz, Rafael Paintner, Julian Gleißner, Alwin Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2504.20477)  

**Abstract**: In recent years, the field of robotics has witnessed a significant shift from operating in structured environments to handling dynamic and unpredictable settings. To tackle these challenges, methodologies from the field of self-adaptive systems enabling these systems to react to unforeseen circumstances during runtime have been applied. The Monitoring-Analysis- Planning-Execution over Knowledge (MAPE-K) feedback loop model is a popular approach, often implemented in a managing subsystem, responsible for monitoring and adapting a managed subsystem. This work explores the implementation of the MAPE- K feedback loop based on Behavior Trees (BTs) within the Robot Operating System 2 (ROS2) framework. By delineating the managed and managing subsystems, our approach enhances the flexibility and adaptability of ROS-based systems, ensuring they not only meet Quality-of-Service (QoS), but also system health metric requirements, namely availability of ROS nodes and communication channels. Our implementation allows for the application of the method to new managed subsystems without needing custom BT nodes as the desired behavior can be configured within a specific rule set. We demonstrate the effectiveness of our method through various experiments on a system showcasing an aerial perception use case. By evaluating different failure cases, we show both an increased perception quality and a higher system availability. Our code is open source 

**Abstract (ZH)**: 近年来，机器人领域见证了从操作结构化环境向处理动态和不可预测环境的转变。为了应对这些挑战，来自自适应系统领域的方法被应用，使这些系统能够在运行时对不可预见的情况作出反应。MAPE-K反馈循环模型是一种流行的方法，通常在管理子系统中实现，负责监控和适应被管理的子系统。本文探讨了基于行为树（Behavior Trees，BTs）在Robot Operating System 2（ROS2）框架中实现MAPE-K反馈循环的方法。通过界定被管理子系统和管理子系统，我们的方法增强了基于ROS的系统的灵活性和适应性，确保它们不仅满足服务质量（QoS）要求，还满足系统健康指标要求，即ROS节点和通信通道的可用性。我们的实现允许将该方法应用于新的被管理子系统，无需为所需行为创建自定义的行为树节点，只需配置特定的规则集即可。通过在展示空中感知应用案例的系统上进行各种实验，我们展示了该方法的有效性。通过评估不同的故障情况，我们表明了感知质量的提高和系统可用性的增加。我们的代码是开源的。标题：基于行为树在ROS2框架中实现MAPE-K反馈循环的方法 

---
# SAS-Prompt: Large Language Models as Numerical Optimizers for Robot Self-Improvement 

**Title (ZH)**: SAS-Prompt: 大型语言模型作为数值优化器实现机器人自我提升 

**Authors**: Heni Ben Amor, Laura Graesser, Atil Iscen, David D'Ambrosio, Saminda Abeyruwan, Alex Bewley, Yifan Zhou, Kamalesh Kalirathinam, Swaroop Mishra, Pannag Sanketi  

**Link**: [PDF](https://arxiv.org/pdf/2504.20459)  

**Abstract**: We demonstrate the ability of large language models (LLMs) to perform iterative self-improvement of robot policies. An important insight of this paper is that LLMs have a built-in ability to perform (stochastic) numerical optimization and that this property can be leveraged for explainable robot policy search. Based on this insight, we introduce the SAS Prompt (Summarize, Analyze, Synthesize) -- a single prompt that enables iterative learning and adaptation of robot behavior by combining the LLM's ability to retrieve, reason and optimize over previous robot traces in order to synthesize new, unseen behavior. Our approach can be regarded as an early example of a new family of explainable policy search methods that are entirely implemented within an LLM. We evaluate our approach both in simulation and on a real-robot table tennis task. Project website: this http URL 

**Abstract (ZH)**: 我们展示了大型语言模型（LLMs）执行机器人策略迭代自我优化的能力。本文的一个重要见解是，LLMs 内置了进行（随机的）数值优化的能力，这一特性可以用于可解释的机器人策略搜索。基于这一见解，我们引入了 SAS 提示（Summarize, Analyze, Synthesize）——一个单一的提示，通过结合LLM检索、推理和优化之前机器人行为的能力来综合新的未见过的行为，以实现迭代的学习和适应。我们的方法可以被视为一种新的可解释策略搜索方法的早期范例，这些方法完全在LLM中实现。我们在仿真和真实的乒乓球机器人任务中评估了我们的方法。项目网址：this http URL 

---
# LPVIMO-SAM: Tightly-coupled LiDAR/Polarization Vision/Inertial/Magnetometer/Optical Flow Odometry via Smoothing and Mapping 

**Title (ZH)**: LPVIMO-SAM: 结合平滑与制图的紧耦合激光雷达/偏振视觉/惯性/磁计/光学流bundle调整与建图算法 

**Authors**: Derui Shan, Peng Guo, Wenshuo Li, Du Tao  

**Link**: [PDF](https://arxiv.org/pdf/2504.20380)  

**Abstract**: We propose a tightly-coupled LiDAR/Polarization Vision/Inertial/Magnetometer/Optical Flow Odometry via Smoothing and Mapping (LPVIMO-SAM) framework, which integrates LiDAR, polarization vision, inertial measurement unit, magnetometer, and optical flow in a tightly-coupled fusion. This framework enables high-precision and highly robust real-time state estimation and map construction in challenging environments, such as LiDAR-degraded, low-texture regions, and feature-scarce areas. The LPVIMO-SAM comprises two subsystems: a Polarized Vision-Inertial System and a LiDAR/Inertial/Magnetometer/Optical Flow System. The polarized vision enhances the robustness of the Visual/Inertial odometry in low-feature and low-texture scenarios by extracting the polarization information of the scene. The magnetometer acquires the heading angle, and the optical flow obtains the speed and height to reduce the accumulated error. A magnetometer heading prior factor, an optical flow speed observation factor, and a height observation factor are designed to eliminate the cumulative errors of the LiDAR/Inertial odometry through factor graph optimization. Meanwhile, the LPVIMO-SAM can maintain stable positioning even when one of the two subsystems fails, further expanding its applicability in LiDAR-degraded, low-texture, and low-feature environments. Code is available on this https URL. 

**Abstract (ZH)**: 基于平滑与制图的紧耦合LiDAR/偏振视觉/惯性/磁强计/光流里程计(LPVIMO-SAM)框架 

---
# PRISM-DP: Spatial Pose-based Observations for Diffusion-Policies via Segmentation, Mesh Generation, and Pose Tracking 

**Title (ZH)**: PRISM-DP: 基于空间姿态的观察方法以通过分割、网格生成和姿态跟踪实现扩散策略 

**Authors**: Xiatao Sun, Yinxing Chen, Daniel Rakita  

**Link**: [PDF](https://arxiv.org/pdf/2504.20359)  

**Abstract**: Diffusion-based visuomotor policies generate robot motions by learning to denoise action-space trajectories conditioned on observations. These observations are commonly streams of RGB images, whose high dimensionality includes substantial task-irrelevant information, requiring large models to extract relevant patterns. In contrast, using more structured observations, such as the spatial poses (positions and orientations) of key objects over time, enables training more compact policies that can recognize relevant patterns with fewer parameters. However, obtaining accurate object poses in open-set, real-world environments remains challenging. For instance, it is impractical to assume that all relevant objects are equipped with markers, and recent learning-based 6D pose estimation and tracking methods often depend on pre-scanned object meshes, requiring manual reconstruction. In this work, we propose PRISM-DP, an approach that leverages segmentation, mesh generation, pose estimation, and pose tracking models to enable compact diffusion policy learning directly from the spatial poses of task-relevant objects. Crucially, because PRISM-DP uses a mesh generation model, it eliminates the need for manual mesh processing or creation, improving scalability and usability in open-set, real-world environments. Experiments across a range of tasks in both simulation and real-world settings show that PRISM-DP outperforms high-dimensional image-based diffusion policies and achieves performance comparable to policies trained with ground-truth state information. We conclude with a discussion of the broader implications and limitations of our approach. 

**Abstract (ZH)**: 基于扩散的视听运动策略通过学习条件于观察的行动空间轨迹去噪来生成机器人运动。这些观察通常是RGB图像流，其高维度中包含大量与任务无关的信息，需要大型模型来提取相关模式。相比之下，使用更结构化的观察，如随时间变化的关键对象的空间姿态（位置和方向），能够训练更紧凑的策略，并用较少的参数识别相关模式。然而，在开放集的实际环境中获得准确的对象姿态仍然具有挑战性。例如，并非所有相关对象都配备标记，基于学习的6D姿态估计算法通常依赖于预扫描的对象网格，需要手动重建。在本工作中，我们提出了一种名为PRISM-DP的方法，该方法利用分割、网格生成、姿态估计和姿态跟踪模型，直接从任务相关对象的空间姿态中进行紧凑扩散策略学习。关键的是，由于PRISM-DP使用了网格生成模型，它消除了手动网格处理或创建的需求，提高了在开放集的实际环境中的可扩展性和易用性。在仿真和实际环境中的多种任务实验中，PRISM-DP优于高维度图像基扩散策略，并实现了与基于真实状态信息训练的策略相当的性能。本文最后讨论了我们方法的更广泛影响和局限性。 

---
# DRO: Doppler-Aware Direct Radar Odometry 

**Title (ZH)**: DRO：多普勒感知直接雷达里程计 

**Authors**: Cedric Le Gentil, Leonardo Brizi, Daniil Lisus, Xinyuan Qiao, Giorgio Grisetti, Timothy D. Barfoot  

**Link**: [PDF](https://arxiv.org/pdf/2504.20339)  

**Abstract**: A renaissance in radar-based sensing for mobile robotic applications is underway. Compared to cameras or lidars, millimetre-wave radars have the ability to `see' through thin walls, vegetation, and adversarial weather conditions such as heavy rain, fog, snow, and dust. In this paper, we propose a novel SE(2) odometry approach for spinning frequency-modulated continuous-wave radars. Our method performs scan-to-local-map registration of the incoming radar data in a direct manner using all the radar intensity information without the need for feature or point cloud extraction. The method performs locally continuous trajectory estimation and accounts for both motion and Doppler distortion of the radar scans. If the radar possesses a specific frequency modulation pattern that makes radial Doppler velocities observable, an additional Doppler-based constraint is formulated to improve the velocity estimate and enable odometry in geometrically feature-deprived scenarios (e.g., featureless tunnels). Our method has been validated on over 250km of on-road data sourced from public datasets (Boreas and MulRan) and collected using our automotive platform. With the aid of a gyroscope, it outperforms state-of-the-art methods and achieves an average relative translation error of 0.26% on the Boreas leaderboard. When using data with the appropriate Doppler-enabling frequency modulation pattern, the translation error is reduced to 0.18% in similar environments. We also benchmarked our algorithm using 1.5 hours of data collected with a mobile robot in off-road environments with various levels of structure to demonstrate its versatility. Our real-time implementation is publicly available: this https URL. 

**Abstract (ZH)**: 基于雷达的移动机器人应用传感复兴正在进行中。与摄像头或激光雷达相比，毫米波雷达能够在透过薄墙、植被以及恶劣天气（如大雨、雾、雪和尘埃）的情况下“看”到目标。本文提出了一种针对旋转频率调制连续波雷达的新型SE(2)里程计方法。该方法直接利用所有雷达强度信息进行进来雷达数据与局部地图的配准，无需提取特征或点云。方法实现了局部连续轨迹估计，并同时考虑了雷达扫描的运动和多普勒失真。如果雷达具有特定的频率调制模式使其径向多普勒速度可观测，将额外引入多普勒约束以改进速度估计，并在几何特征缺乏的场景中（如无特征隧道）实现里程计。该方法已在公共数据集（Boreas和MulRan）的超过250公里路面上数据以及使用我们的汽车平台收集的数据上进行了验证。借助陀螺仪，该方法在Boreas排行榜上优于现有最佳方法，平均相对位移误差为0.26%。在具有适当多普勒使能频率调制模式的数据中，相同环境下的位移误差可降低至0.18%。我们还在不同结构水平的野外环境中用移动机器人收集了1.5小时的数据，以此来验证其通用性。我们的实时实现已公开：this https URL。 

---
# NMPC-based Unified Posture Manipulation and Thrust Vectoring for Agile and Fault-Tolerant Flight of a Morphing Aerial Robot 

**Title (ZH)**: 基于NMPC的形态变化航空机器人的一体化姿态操控与推力矢量控制敏捷及容错飞行方法 

**Authors**: Shashwat Pandya  

**Link**: [PDF](https://arxiv.org/pdf/2504.20326)  

**Abstract**: This thesis presents a unified control framework for agile and fault-tolerant flight of the Multi-Modal Mobility Morphobot (M4) in aerial mode. The M4 robot is capable of transitioning between ground and aerial locomotion. The articulated legs enable more dynamic maneuvers than a standard quadrotor platform. A nonlinear model predictive control (NMPC) approach is developed to simultaneously plan posture manipulation and thrust vectoring actions, allowing the robot to execute sharp turns and dynamic flight trajectories. The framework integrates an agile and fault-tolerant control logic that enables precise tracking under aggressive maneuvers while compensating for actuator failures, ensuring continued operation without significant performance degradation. Simulation results validate the effectiveness of the proposed method, demonstrating accurate trajectory tracking and robust recovery from faults, contributing to resilient autonomous flight in complex environments. 

**Abstract (ZH)**: 本论文提出了一种统一的控制框架，用于Multi-Modal Mobility Morphobot (M4) 无人机模式下的敏捷和容错飞行控制。M4 机器人能够在地面和空中运动间转换。其 articulated 腿使得动作更为动态，超过了标准四旋翼平台。开发了一种非线性模型预测控制（NMPC）方法，同时规划姿态操作和推力矢量动作，使机器人能够执行快速转弯和动态飞行轨迹。该框架整合了敏捷和容错控制逻辑，能够在剧烈机动下实现精确跟踪，并补偿执行器故障，确保在不显著性能下降的情况下持续运行。仿真结果验证了所提出方法的有效性，展示了精确的轨迹跟踪和从故障中稳健恢复的能力，从而为复杂环境下的可靠自主飞行做出了贡献。 

---
# System Identification of Thrust and Torque Characteristics for a Bipedal Robot with Integrated Propulsion 

**Title (ZH)**: 双足机器人集成推进系统推力和扭矩特性的系统识别 

**Authors**: Thomas Cahill  

**Link**: [PDF](https://arxiv.org/pdf/2504.20313)  

**Abstract**: Bipedal robots represent a remarkable and sophisticated class of robotics, designed to emulate human form and movement. Their development marks a significant milestone in the field. However, even the most advanced bipedal robots face challenges related to terrain variation, obstacle negotiation, payload management, weight distribution, and recovering from stumbles. These challenges can be mitigated by incorporating thrusters, which enhance stability on uneven terrain, facilitate obstacle avoidance, and improve recovery after stumbling. Harpy is a bipedal robot equipped with six joints and two thrusters, serving as a hardware platform for implementing and testing advanced control algorithms. This thesis focuses on characterizing Harpy's hardware to improve the system's overall robustness, controllability, and predictability. It also examines simulation results for predicting thrust in propeller-based mechanisms, the integration of thrusters into the Harpy platform and associated testing, as well as an exploration of motor torque characterization methods and their application to hardware in relation to closed-loop force-based impedance control. 

**Abstract (ZH)**: 双足机器人代表一类优异而复杂的机器人，设计旨在模拟人类形态和运动。它们的发展标志着领域内的一个重要里程碑。然而，即使是最先进的双足机器人仍然面临地形变化、障碍物规避、载荷管理、重量分布以及摔倒后恢复等挑战。通过集成推进器，可以减轻这些挑战，推进器能提高不规则地形上的稳定性、帮助规避障碍物，并改善摔倒后的恢复能力。Harpy是一种配备了六个关节和两个推进器的双足机器人，作为实现和测试高级控制算法的硬件平台。本文 focuses于表征Harpy的硬件，以提高系统的整体稳健性、可控性和可预测性。同时，本文还探讨了基于推进器机制的推力预测仿真结果、将推进器集成到Harpy平台及其相关测试，并探索电动机扭矩表征方法及其在闭环力基阻抗控制中的应用。 

---
# Deformable Multibody Modeling for Model Predictive Control in Legged Locomotion with Embodied Compliance 

**Title (ZH)**: 基于体业态 compliant 态控制的腿足运动中可变形多体建模 

**Authors**: Keran Ye, Konstantinos Karydis  

**Link**: [PDF](https://arxiv.org/pdf/2504.20301)  

**Abstract**: The paper presents a method to stabilize dynamic gait for a legged robot with embodied compliance. Our approach introduces a unified description for rigid and compliant bodies to approximate their deformation and a formulation for deformable multibody systems. We develop the centroidal composite predictive deformed inertia (CCPDI) tensor of a deformable multibody system and show how to integrate it with the standard-of-practice model predictive controller (MPC). Simulation shows that the resultant control framework can stabilize trot stepping on a quadrupedal robot with both rigid and compliant spines under the same MPC configurations. Compared to standard MPC, the developed CCPDI-enabled MPC distributes the ground reactive forces closer to the heuristics for body balance, and it is thus more likely to stabilize the gaits of the compliant robot. A parametric study shows that our method preserves some level of robustness within a suitable envelope of key parameter values. 

**Abstract (ZH)**: 一种基于体质性顺应性的Legged机器人动态步态稳定方法 

---
# GenGrid: A Generalised Distributed Experimental Environmental Grid for Swarm Robotics 

**Title (ZH)**: GenGrid: 一种通用的分布式实验环境网格在 swarm 机器人中的应用 

**Authors**: Pranav Kedia, Madhav Rao  

**Link**: [PDF](https://arxiv.org/pdf/2504.20071)  

**Abstract**: GenGrid is a novel comprehensive open-source, distributed platform intended for conducting extensive swarm robotic experiments. The modular platform is designed to run swarm robotics experiments that are compatible with different types of mobile robots ranging from Colias, Kilobot, and E puck. The platform offers programmable control over the experimental setup and its parameters and acts as a tool to collect swarm robot data, including localization, sensory feedback, messaging, and interaction. GenGrid is designed as a modular grid of attachable computing nodes that offers bidirectional communication between the robotic agent and grid nodes and within grids. The paper describes the hardware and software architecture design of the GenGrid system. Further, it discusses some common experimental studies covering multi-robot and swarm robotics to showcase the platform's use. GenGrid of 25 homogeneous cells with identical sensing and communication characteristics with a footprint of 37.5 cm X 37.5 cm, exhibits multiple capabilities with minimal resources. The open-source hardware platform is handy for running swarm experiments, including robot hopping based on multiple gradients, collective transport, shepherding, continuous pheromone deposition, and subsequent evaporation. The low-cost, modular, and open-source platform is significant in the swarm robotics research community, which is currently driven by commercial platforms that allow minimal modifications. 

**Abstract (ZH)**: GenGrid是一种新型综合开源分布式平台，旨在进行广泛的 swarm 机器人实验。 

---
# TesserAct: Learning 4D Embodied World Models 

**Title (ZH)**: TesserAct: 学习四维实体世界模型 

**Authors**: Haoyu Zhen, Qiao Sun, Hongxin Zhang, Junyan Li, Siyuan Zhou, Yilun Du, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2504.20995)  

**Abstract**: This paper presents an effective approach for learning novel 4D embodied world models, which predict the dynamic evolution of 3D scenes over time in response to an embodied agent's actions, providing both spatial and temporal consistency. We propose to learn a 4D world model by training on RGB-DN (RGB, Depth, and Normal) videos. This not only surpasses traditional 2D models by incorporating detailed shape, configuration, and temporal changes into their predictions, but also allows us to effectively learn accurate inverse dynamic models for an embodied agent. Specifically, we first extend existing robotic manipulation video datasets with depth and normal information leveraging off-the-shelf models. Next, we fine-tune a video generation model on this annotated dataset, which jointly predicts RGB-DN (RGB, Depth, and Normal) for each frame. We then present an algorithm to directly convert generated RGB, Depth, and Normal videos into a high-quality 4D scene of the world. Our method ensures temporal and spatial coherence in 4D scene predictions from embodied scenarios, enables novel view synthesis for embodied environments, and facilitates policy learning that significantly outperforms those derived from prior video-based world models. 

**Abstract (ZH)**: 本文提出了一种有效的方法，用于学习新颖的4D具身世界模型，这些模型能够预测在具身代理行动响应下3D场景随时间的动态演变，同时提供时空一致性。我们提出通过训练RGB-DN（RGB、深度和法线）视频来学习4D世界模型。这种方法不仅超越了传统的2D模型，通过其预测中包含了详细的形状、配置和时间变化，还使我们能够有效地学习具身代理的精确逆动力学模型。具体而言，我们首先利用现成的模型扩展现有的具身操作视频数据集，使其包含深度和法线信息。接着，我们在该标注数据集上微调视频生成模型，该模型联合预测每一帧的RGB-DN（RGB、深度和法线）。然后，我们提出了一种算法，可以直接将生成的RGB、深度和法线视频转换成高质量的4D世界场景。本方法确保了具身场景中4D场景预测的时间和空间一致性，支持具身环境的新视角合成，并促进了显著优于先前基于视频的世界模型所导出策略的学习。 

---
# Scenario-based Compositional Verification of Autonomous Systems with Neural Perception 

**Title (ZH)**: 基于场景的自主系统神经感知组成验证 

**Authors**: Christopher Watson, Rajeev Alur, Divya Gopinath, Ravi Mangal, Corina S. Pasareanu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20942)  

**Abstract**: Recent advances in deep learning have enabled the development of autonomous systems that use deep neural networks for perception. Formal verification of these systems is challenging due to the size and complexity of the perception DNNs as well as hard-to-quantify, changing environment conditions. To address these challenges, we propose a probabilistic verification framework for autonomous systems based on the following key concepts: (1) Scenario-based Modeling: We decompose the task (e.g., car navigation) into a composition of scenarios, each representing a different environment condition. (2) Probabilistic Abstractions: For each scenario, we build a compact abstraction of perception based on the DNN's performance on an offline dataset that represents the scenario's environment condition. (3) Symbolic Reasoning and Acceleration: The abstractions enable efficient compositional verification of the autonomous system via symbolic reasoning and a novel acceleration proof rule that bounds the error probability of the system under arbitrary variations of environment conditions. We illustrate our approach on two case studies: an experimental autonomous system that guides airplanes on taxiways using high-dimensional perception DNNs and a simulation model of an F1Tenth autonomous car using LiDAR observations. 

**Abstract (ZH)**: Recent Advances in Deep Learning Have Enabled the Development of Autonomous Systems That Use Deep Neural Networks for Perception: A Probabilistic Verification Framework Based on Scenario-Based Modeling, Probabilistic Abstractions, and Symbolic Reasoning 

---
# The Mean of Multi-Object Trajectories 

**Title (ZH)**: 多目标轨迹的均值 

**Authors**: Tran Thien Dat Nguyen, Ba Tuong Vo, Ba-Ngu Vo, Hoa Van Nguyen, Changbeom Shim  

**Link**: [PDF](https://arxiv.org/pdf/2504.20391)  

**Abstract**: This paper introduces the concept of a mean for trajectories and multi-object trajectories--sets or multi-sets of trajectories--along with algorithms for computing them. Specifically, we use the Fréchet mean, and metrics based on the optimal sub-pattern assignment (OSPA) construct, to extend the notion of average from vectors to trajectories and multi-object trajectories. Further, we develop efficient algorithms to compute these means using greedy search and Gibbs sampling. Using distributed multi-object tracking as an application, we demonstrate that the Fréchet mean approach to multi-object trajectory consensus significantly outperforms state-of-the-art distributed multi-object tracking methods. 

**Abstract (ZH)**: 本文介绍了轨迹及多对象轨迹——集合或多重集合的均值概念，并提出了计算这些均值的算法。具体而言，我们使用Fréchet均值和基于最优子模式分配（OSPA）构造的距离度量，将平均的概念从向量扩展到轨迹及多对象轨迹。进一步，我们开发了使用贪婪搜索和吉布斯采样的高效算法来计算这些均值。通过分布式多对象跟踪的应用，我们展示了Fréchet均值方法在多对象轨迹一致性方面的性能显著优于现有的分布式多对象跟踪方法。 

---
# GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting 

**Title (ZH)**: GSFeatLoc: 基于3D高斯绘制特征对应的空间定位 

**Authors**: Jongwon Lee, Timothy Bretl  

**Link**: [PDF](https://arxiv.org/pdf/2504.20379)  

**Abstract**: In this paper, we present a method for localizing a query image with respect to a precomputed 3D Gaussian Splatting (3DGS) scene representation. First, the method uses 3DGS to render a synthetic RGBD image at some initial pose estimate. Second, it establishes 2D-2D correspondences between the query image and this synthetic image. Third, it uses the depth map to lift the 2D-2D correspondences to 2D-3D correspondences and solves a perspective-n-point (PnP) problem to produce a final pose estimate. Results from evaluation across three existing datasets with 38 scenes and over 2,700 test images show that our method significantly reduces both inference time (by over two orders of magnitude, from more than 10 seconds to as fast as 0.1 seconds) and estimation error compared to baseline methods that use photometric loss minimization. Results also show that our method tolerates large errors in the initial pose estimate of up to 55° in rotation and 1.1 units in translation (normalized by scene scale), achieving final pose errors of less than 5° in rotation and 0.05 units in translation on 90% of images from the Synthetic NeRF and Mip-NeRF360 datasets and on 42% of images from the more challenging Tanks and Temples dataset. 

**Abstract (ZH)**: 本文提出了一种基于预先计算的3D高斯点云表示的查询图像定位方法。首先，该方法使用3D高斯点云渲染初始姿态估计下的合成RGBD图像。其次，它在查询图像与合成图像之间建立2D-2D对应关系。然后，利用深度图将2D-2D对应关系提升为2D-3D对应关系，并求解透视n点问题（PnP问题）以生成最终的姿态估计。在三个现有数据集上的评估结果显示，与使用 photometric损失最小化的基线方法相比，该方法显著减少了推理时间（减少了两个数量级以上，从超过10秒加速到0.1秒以内）并降低了估计误差。结果还显示，该方法可以容忍初始姿态估计的大误差，旋转误差高达55°，平移误差高达1.1个单位（以场景尺度归一化），在90%的Synthetic NeRF和Mip-NeRF360数据集图像以及42%的更具挑战性的Tanks and Temples数据集图像上，最终的姿态误差分别小于5°旋转和0.05个单位平移。 

---
# Improving trajectory continuity in drone-based crowd monitoring using a set of minimal-cost techniques and deep discriminative correlation filters 

**Title (ZH)**: 基于一组最小成本技术与深度区分性相关滤波器提高无人机人群监控轨迹连续性 

**Authors**: Bartosz Ptak, Marek Kraft  

**Link**: [PDF](https://arxiv.org/pdf/2504.20234)  

**Abstract**: Drone-based crowd monitoring is the key technology for applications in surveillance, public safety, and event management. However, maintaining tracking continuity and consistency remains a significant challenge. Traditional detection-assignment tracking methods struggle with false positives, false negatives, and frequent identity switches, leading to degraded counting accuracy and making in-depth analysis impossible. This paper introduces a point-oriented online tracking algorithm that improves trajectory continuity and counting reliability in drone-based crowd monitoring. Our method builds on the Simple Online and Real-time Tracking (SORT) framework, replacing the original bounding-box assignment with a point-distance metric. The algorithm is enhanced with three cost-effective techniques: camera motion compensation, altitude-aware assignment, and classification-based trajectory validation. Further, Deep Discriminative Correlation Filters (DDCF) that re-use spatial feature maps from localisation algorithms for increased computational efficiency through neural network resource sharing are integrated to refine object tracking by reducing noise and handling missed detections. The proposed method is evaluated on the DroneCrowd and newly shared UP-COUNT-TRACK datasets, demonstrating substantial improvements in tracking metrics, reducing counting errors to 23% and 15%, respectively. The results also indicate a significant reduction of identity switches while maintaining high tracking accuracy, outperforming baseline online trackers and even an offline greedy optimisation method. 

**Abstract (ZH)**: 基于无人机的人群监测中点导向的在线跟踪算法：提高轨迹连续性和计数可靠性 

---
