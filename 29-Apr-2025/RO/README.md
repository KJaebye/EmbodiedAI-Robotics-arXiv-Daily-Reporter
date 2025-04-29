# Kinodynamic Trajectory Following with STELA: Simultaneous Trajectory Estimation & Local Adaptation 

**Title (ZH)**: 基于STELA的 kino-dynamic 轨迹跟随：轨迹估计与局部适应的同时进行 

**Authors**: Edgar Granados, Sumanth Tangirala, Kostas E. Bekris  

**Link**: [PDF](https://arxiv.org/pdf/2504.20009)  

**Abstract**: State estimation and control are often addressed separately, leading to unsafe execution due to sensing noise, execution errors, and discrepancies between the planning model and reality. Simultaneous control and trajectory estimation using probabilistic graphical models has been proposed as a unified solution to these challenges. Previous work, however, relies heavily on appropriate Gaussian priors and is limited to holonomic robots with linear time-varying models. The current research extends graphical optimization methods to vehicles with arbitrary dynamical models via Simultaneous Trajectory Estimation and Local Adaptation (STELA). The overall approach initializes feasible trajectories using a kinodynamic, sampling-based motion planner. Then, it simultaneously: (i) estimates the past trajectory based on noisy observations, and (ii) adapts the controls to be executed to minimize deviations from the planned, feasible trajectory, while avoiding collisions. The proposed factor graph representation of trajectories in STELA can be applied for any dynamical system given access to first or second-order state update equations, and introduces the duration of execution between two states in the trajectory discretization as an optimization variable. These features provide both generalization and flexibility in trajectory following. In addition to targeting computational efficiency, the proposed strategy performs incremental updates of the factor graph using the iSAM algorithm and introduces a time-window mechanism. This mechanism allows the factor graph to be dynamically updated to operate over a limited history and forward horizon of the planned trajectory. This enables online updates of controls at a minimum of 10Hz. Experiments demonstrate that STELA achieves at least comparable performance to previous frameworks on idealized vehicles with linear dynamics.[...] 

**Abstract (ZH)**: 同时轨迹估计与局部适应的控制与轨迹估计方法 

---
# Socially-Aware Autonomous Driving: Inferring Yielding Intentions for Safer Interactions 

**Title (ZH)**: 社交意识自主驾驶：推断让行意图以实现更安全的交互 

**Authors**: Jing Wang, Yan Jin, Hamid Taghavifar, Fei Ding, Chongfeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2504.20004)  

**Abstract**: Since the emergence of autonomous driving technology, it has advanced rapidly over the past decade. It is becoming increasingly likely that autonomous vehicles (AVs) would soon coexist with human-driven vehicles (HVs) on the roads. Currently, safety and reliable decision-making remain significant challenges, particularly when AVs are navigating lane changes and interacting with surrounding HVs. Therefore, precise estimation of the intentions of surrounding HVs can assist AVs in making more reliable and safe lane change decision-making. This involves not only understanding their current behaviors but also predicting their future motions without any direct communication. However, distinguishing between the passing and yielding intentions of surrounding HVs still remains ambiguous. To address the challenge, we propose a social intention estimation algorithm rooted in Directed Acyclic Graph (DAG), coupled with a decision-making framework employing Deep Reinforcement Learning (DRL) algorithms. To evaluate the method's performance, the proposed framework can be tested and applied in a lane-changing scenario within a simulated environment. Furthermore, the experiment results demonstrate how our approach enhances the ability of AVs to navigate lane changes safely and efficiently on roads. 

**Abstract (ZH)**: 自从自主驾驶技术的出现，过去十年间该技术已快速发展。随着自主车辆（AVs）在未来不久可能与人工驾驶车辆（HVs）共同行驶在道路上，当前安全性和可靠的决策仍然是重大挑战，尤其是在AVs进行车道变换并与周围HV互动时。因此，精确估计周围HV的意图可以帮助AVs做出更可靠和安全的车道变换决策。这不仅涉及理解其当前行为，还涉及在没有直接通信的情况下预测其未来运动。然而，区分周围HV的通过意图和礼让意图仍然存在模糊性。为应对这一挑战，我们提出了一种基于有向无环图（DAG）的社会意图估计算法，并结合了一种采用深度强化学习（DRL）算法的决策框架。通过在模拟环境中测试和应用该框架，在车道变换场景中评估该方法的性能。实验结果进一步证明了我们方法如何增强AVs在道路上安全高效地进行车道变换的能力。 

---
# HJRNO: Hamilton-Jacobi Reachability with Neural Operators 

**Title (ZH)**: HJRNO: Hamilton-Jacobi Reachability with Neural Operators 

**Authors**: Yankai Li, Mo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.19989)  

**Abstract**: Ensuring the safety of autonomous systems under uncertainty is a critical challenge. Hamilton-Jacobi reachability (HJR) analysis is a widely used method for guaranteeing safety under worst-case disturbances. Traditional HJR methods provide safety guarantees but suffer from the curse of dimensionality, limiting their scalability to high-dimensional systems or varying environmental conditions. In this work, we propose HJRNO, a neural operator-based framework for solving backward reachable tubes (BRTs) efficiently and accurately. By leveraging the Fourier Neural Operator (FNO), HJRNO learns a mapping between value functions, enabling fast inference with strong generalization across different obstacle shapes, system configurations, and hyperparameters. We demonstrate that HJRNO achieves low error on random obstacle scenarios and generalizes effectively across varying system dynamics. These results suggest that HJRNO offers a promising foundation model approach for scalable, real-time safety analysis in autonomous systems. 

**Abstract (ZH)**: 确保自主系统在不确定条件下的安全性是一个关键挑战。哈密尔顿-雅各比可达性（HJR）分析是一种广泛用于在最坏情况干扰下提供安全性保证的方法。传统HJR方法虽能提供安全性保证，但受到维数灾的限制，限制了其在高维系统或变化环境条件下的可扩展性。在这项工作中，我们提出了HJRNO，一种基于神经算子的框架，用于高效准确地求解后向可达管（BRTs）。通过利用傅里叶神经算子（FNO），HJRNO学习值函数之间的映射，使得具有强泛化能力的快速推理成为可能，适用于不同的障碍形状、系统配置和超参数。我们展示了HJRNO在随机障碍情境下具有低误差，并且在不同系统动力学条件下具有良好的泛化能力。这些结果表明，HJRNO为自主系统中可扩展的实时安全性分析提供了有前景的基础模型方法。 

---
# Real-Time Imitation of Human Head Motions, Blinks and Emotions by Nao Robot: A Closed-Loop Approach 

**Title (ZH)**: Nao机器人基于闭合回路的实时模仿人类头部运动、眨眼和情绪方法 

**Authors**: Keyhan Rayati, Amirhossein Feizi, Alireza Beigy, Pourya Shahverdi, Mehdi Tale Masouleh, Ahmad Kalhor  

**Link**: [PDF](https://arxiv.org/pdf/2504.19985)  

**Abstract**: This paper introduces a novel approach for enabling real-time imitation of human head motion by a Nao robot, with a primary focus on elevating human-robot interactions. By using the robust capabilities of the MediaPipe as a computer vision library and the DeepFace as an emotion recognition library, this research endeavors to capture the subtleties of human head motion, including blink actions and emotional expressions, and seamlessly incorporate these indicators into the robot's responses. The result is a comprehensive framework which facilitates precise head imitation within human-robot interactions, utilizing a closed-loop approach that involves gathering real-time feedback from the robot's imitation performance. This feedback loop ensures a high degree of accuracy in modeling head motion, as evidenced by an impressive R2 score of 96.3 for pitch and 98.9 for yaw. Notably, the proposed approach holds promise in improving communication for children with autism, offering them a valuable tool for more effective interaction. In essence, proposed work explores the integration of real-time head imitation and real-time emotion recognition to enhance human-robot interactions, with potential benefits for individuals with unique communication needs. 

**Abstract (ZH)**: 一种基于MediaPipe和DeepFace实现实时人类头部运动仿真的新方法：提升人机交互质量 

---
# Feelbert: A Feedback Linearization-based Embedded Real-Time Quadrupedal Locomotion Framework 

**Title (ZH)**: Feelbert: 一种基于反馈线性化的嵌入式实时四足行走框架 

**Authors**: Aristide Emanuele Casucci, Federico Nesti, Mauro Marinoni, Giorgio Buttazzo  

**Link**: [PDF](https://arxiv.org/pdf/2504.19965)  

**Abstract**: Quadruped robots have become quite popular for their ability to adapt their locomotion to generic uneven terrains. For this reason, over time, several frameworks for quadrupedal locomotion have been proposed, but with little attention to ensuring a predictable timing behavior of the controller.
To address this issue, this work presents \NAME, a modular control framework for quadrupedal locomotion suitable for execution on an embedded system under hard real-time execution constraints. It leverages the feedback linearization control technique to obtain a closed-form control law for the body, valid for all configurations of the robot. The control law was derived after defining an appropriate rigid body model that uses the accelerations of the feet as control variables, instead of the estimated contact forces. This work also provides a novel algorithm to compute footholds and gait temporal parameters using the concept of imaginary wheels, and a heuristic algorithm to select the best gait schedule for the current velocity commands.
The proposed framework is developed entirely in C++, with no dependencies on third-party libraries and no dynamic memory allocation, to ensure predictability and real-time performance. Its implementation allows \NAME\ to be both compiled and executed on an embedded system for critical applications, as well as integrated into larger systems such as Robot Operating System 2 (ROS 2). For this reason, \NAME\ has been tested in both scenarios, demonstrating satisfactory results both in terms of reference tracking and temporal predictability, whether integrated into ROS 2 or compiled as a standalone application on a Raspberry Pi 5. 

**Abstract (ZH)**: 四足机器人因其能够适应通用不平地形的运动能力而变得非常流行。因此，随着时间的推移，已经提出了多种四足运动框架，但缺乏对控制器可预测的时间行为的保证。
为了解决这个问题，本工作提出了一种名为\NAME的模块化控制框架，适用于具有严格实时执行约束的嵌入式系统。该框架利用反馈线性化控制技术，获得适用于机器人所有配置的有效闭环控制律。控制律在定义了适当的刚体模型之后得出，该模型使用脚的加速度作为控制变量，而不是估计的接触力。此外，本工作还提供了一种使用想象滚轮的概念来计算 footholds 和步态时间参数的新算法，并提供了一种启发式算法来为当前速度命令选择最佳步态调度。
本框架完全用 C++ 编写，不依赖第三方库，且不进行动态内存分配，以确保可预测性和实时性能。其实现允许 \NAME 在关键应用中编译并执行在嵌入式系统上，同时也能集成到更大的系统中，如 Robot Operating System 2（ROS 2）。因此，\NAME 在这两种场景下都进行了测试，无论是在 ROS 2 中集成还是在 Raspberry Pi 5 上作为独立应用程序进行编译，都能在参考跟踪和时间可预测性方面展示出令人满意的结果。 

---
# Tendon-Actuated Concentric Tube Endonasal Robot (TACTER) 

**Title (ZH)**: 经鼻 concentric 管 tendon 驱动机器人 (TACTER) 

**Authors**: Kent K. Yamamoto, Tanner J. Zachem, Pejman Kheradmand, Patrick Zheng, Jihad Abdelgadir, Jared Laurance Bailey, Kaelyn Pieter, Patrick J. Codd, Yash Chitalia  

**Link**: [PDF](https://arxiv.org/pdf/2504.19948)  

**Abstract**: Endoscopic endonasal approaches (EEA) have become more prevalent for minimally invasive skull base and sinus surgeries. However, rigid scopes and tools significantly decrease the surgeon's ability to operate in tight anatomical spaces and avoid critical structures such as the internal carotid artery and cranial nerves. This paper proposes a novel tendon-actuated concentric tube endonasal robot (TACTER) design in which two tendon-actuated robots are concentric to each other, resulting in an outer and inner robot that can bend independently. The outer robot is a unidirectionally asymmetric notch (UAN) nickel-titanium robot, and the inner robot is a 3D-printed bidirectional robot, with a nickel-titanium bending member. In addition, the inner robot can translate axially within the outer robot, allowing the tool to traverse through structures while bending, thereby executing follow-the-leader motion. A Cosserat-rod based mechanical model is proposed that uses tendon tension of both tendon-actuated robots and the relative translation between the robots as inputs and predicts the TACTER tip position for varying input parameters. The model is validated with experiments, and a human cadaver experiment is presented to demonstrate maneuverability from the nostril to the sphenoid sinus. This work presents the first tendon-actuated concentric tube (TACT) dexterous robotic tool capable of performing follow-the-leader motion within natural nasal orifices to cover workspaces typically required for a successful EEA. 

**Abstract (ZH)**: 经鼻内窥镜 approaches (EEA)在颅底和鼻窦手术中的微创手术中逐渐普及。然而，刚性内窥镜和工具显著降低了外科医生在狭窄解剖空间操作的能力，并且难以避开关键结构如颈内动脉和颅神经。本文提出了一种新型的腱驱 concentric 管内窥镜机器人 (TACTER) 设计，该设计中两个腱驱机器人同心布置，形成可独立弯曲的外机器人和内机器人。外机器人是一个单向不对称槽 (UAN) 镍钛合金机器人，内机器人是一个三维打印的双向机器人，装备有镍钛合金弯曲部件。此外，内机器人可以在外机器人中进行轴向移动，使得器械在弯曲过程中能够穿越结构，执行跟随运动。基于 Cosserat 杆的机械模型被提出，该模型使用两个 tendon-actuated 机器人以及两个机器人之间的相对平移作为输入，预测 TACTER 作用点的位置。实验验证了该模型，并通过人类尸体实验展示了从鼻孔到蝶窦的操作灵活性。本文介绍了首款能够在自然鼻孔内执行跟随运动的腱驱 concentric 管内窥镜机器人，以覆盖EEA通常所需的工作空间。 

---
# NORA: A Small Open-Sourced Generalist Vision Language Action Model for Embodied Tasks 

**Title (ZH)**: NORA：一个小型开源通用视觉语言行动模型适用于体态任务 

**Authors**: Chia-Yu Hung, Qi Sun, Pengfei Hong, Amir Zadeh, Chuan Li, U-Xuan Tan, Navonil Majumder, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2504.19854)  

**Abstract**: Existing Visual-Language-Action (VLA) models have shown promising performance in zero-shot scenarios, demonstrating impressive task execution and reasoning capabilities. However, a significant challenge arises from the limitations of visual encoding, which can result in failures during tasks such as object grasping. Moreover, these models typically suffer from high computational overhead due to their large sizes, often exceeding 7B parameters. While these models excel in reasoning and task planning, the substantial computational overhead they incur makes them impractical for real-time robotic environments, where speed and efficiency are paramount. To address the limitations of existing VLA models, we propose NORA, a 3B-parameter model designed to reduce computational overhead while maintaining strong task performance. NORA adopts the Qwen-2.5-VL-3B multimodal model as its backbone, leveraging its superior visual-semantic understanding to enhance visual reasoning and action grounding. Additionally, our \model{} is trained on 970k real-world robot demonstrations and equipped with the FAST+ tokenizer for efficient action sequence generation. Experimental results demonstrate that NORA outperforms existing large-scale VLA models, achieving better task performance with significantly reduced computational overhead, making it a more practical solution for real-time robotic autonomy. 

**Abstract (ZH)**: 现有视觉-语言-动作（VLA）模型在零样本场景中展现出了令人鼓舞的性能，展示了强大的任务执行和推理能力。然而，视觉编码的限制给诸如物体抓取等任务带来了显著挑战。此外，这些模型通常因为庞大的尺寸而面临高计算开销的问题，参数往往超过7B。尽管这些模型在推理和任务规划方面表现出色，但其高昂的计算开销使其在需要速度和效率的实时机器人环境中无法实现。为解决现有VLA模型的限制，我们提出NORA，这是一种3B参数模型，旨在减少计算开销同时保持强大的任务性能。NORA采用Qwen-2.5-VL-3B多模态模型作为骨干，利用其卓越的视觉语义理解来增强视觉推理和动作定位。此外，我们的模型基于970k真实的机器人演示数据进行训练，并配备了FAST+分词器以实现高效的动作序列生成。实验结果表明，NORA在计算开销大幅减少的情况下，仍能超越现有大规模VLA模型，实现更好的任务性能，使其成为实时机器人自主控制的更实用解决方案。 

---
# Do You Know the Way? Human-in-the-Loop Understanding for Fast Traversability Estimation in Mobile Robotics 

**Title (ZH)**: 你知道路在何方？移动机器人快速可通过性评估的人机循环理解方法 

**Authors**: Andre Schreiber, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2504.19851)  

**Abstract**: The increasing use of robots in unstructured environments necessitates the development of effective perception and navigation strategies to enable field robots to successfully perform their tasks. In particular, it is key for such robots to understand where in their environment they can and cannot travel -- a task known as traversability estimation. However, existing geometric approaches to traversability estimation may fail to capture nuanced representations of traversability, whereas vision-based approaches typically either involve manually annotating a large number of images or require robot experience. In addition, existing methods can struggle to address domain shifts as they typically do not learn during deployment. To this end, we propose a human-in-the-loop (HiL) method for traversability estimation that prompts a human for annotations as-needed. Our method uses a foundation model to enable rapid learning on new annotations and to provide accurate predictions even when trained on a small number of quickly-provided HiL annotations. We extensively validate our method in simulation and on real-world data, and demonstrate that it can provide state-of-the-art traversability prediction performance. 

**Abstract (ZH)**: 在无结构环境中超密集机器人应用促使开发有效的感知与导航策略，以使田地机器人成功完成任务。特别是，此类机器人需要理解其环境中的可通行与不可通行区域——这一任务称为通行性估计。然而，现有的几何方法可能难以捕捉通行性的细微表现，而基于视觉的方法通常需要手动标注大量图像或依靠机器人的经验。此外，现有方法在部署过程中通常无法学习，难以应对领域转移。为此，我们提出了一种基于人类在环（HiL）的通行性估计方法，该方法需要时即请人类进行标注。我们的方法利用基础模型快速学习新的标注，并在仅使用少量快速提供的HiL标注训练时仍能够提供准确的预测。我们通过仿真和实际数据进行了广泛验证，并证明该方法可以提供最先进的通行性预测性能。 

---
# Human-Centered AI and Autonomy in Robotics: Insights from a Bibliometric Study 

**Title (ZH)**: 以人为本的AI与机器人自主性：基于文献计量学的研究见解 

**Authors**: Simona Casini, Pietro Ducange, Francesco Marcelloni, Lorenzo Pollini  

**Link**: [PDF](https://arxiv.org/pdf/2504.19848)  

**Abstract**: The development of autonomous robotic systems offers significant potential for performing complex tasks with precision and consistency. Recent advances in Artificial Intelligence (AI) have enabled more capable intelligent automation systems, addressing increasingly complex challenges. However, this progress raises questions about human roles in such systems. Human-Centered AI (HCAI) aims to balance human control and automation, ensuring performance enhancement while maintaining creativity, mastery, and responsibility. For real-world applications, autonomous robots must balance task performance with reliability, safety, and trustworthiness. Integrating HCAI principles enhances human-robot collaboration and ensures responsible operation.
This paper presents a bibliometric analysis of intelligent autonomous robotic systems, utilizing SciMAT and VOSViewer to examine data from the Scopus database. The findings highlight academic trends, emerging topics, and AI's role in self-adaptive robotic behaviour, with an emphasis on HCAI architecture. These insights are then projected onto the IBM MAPE-K architecture, with the goal of identifying how these research results map into actual robotic autonomous systems development efforts for real-world scenarios. 

**Abstract (ZH)**: 自主机器人系统的发展为精确和一致地执行复杂任务提供了重要潜力。近期人工智能（AI）的进步使智能自动化系统更加 capable，以应对愈加复杂的挑战。然而，这种进步引发了关于此类系统中人类角色的问题。以人为中心的人工智能（HCAI）旨在平衡人类控制与自动化，确保性能提升的同时保持创造力、专业性和责任感。在实际应用中，自主机器人必须平衡任务性能、可靠性和可信度。结合HCAI原则可以增强人机协作并确保负责任的操作。

本文利用SciMAT和VOSViewer对Scopus数据库的数据进行文献计量分析，研究智能自主机器人系统的学术趋势、新兴主题以及AI在自适应机器人行为中的作用，重点在于HCAI架构。随后，这些见解被投射到IBM MAPE-K架构上，旨在识别这些研究结果如何映射到实际的自主机器人系统开发努力中的现实场景。 

---
# Automated Generation of Precedence Graphs in Digital Value Chains for Automotive Production 

**Title (ZH)**: 汽车生产中数字价值链中 precedence 图的自动生成 

**Authors**: Cornelius Hake, Christian Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2504.19835)  

**Abstract**: This study examines the digital value chain in automotive manufacturing, focusing on the identification, software flashing, customization, and commissioning of electronic control units in vehicle networks. A novel precedence graph design is proposed to optimize this process chain using an automated scheduling algorithm that employs mixed integer linear programming techniques. The results show significant improvements in key metrics. The algorithm reduces the number of production stations equipped with expensive hardware and software to execute digital value chain processes, while increasing capacity utilization through efficient scheduling and reduced idle time. Task parallelization is optimized, resulting in streamlined workflows and increased throughput. Compared to the traditional method, the automated approach has reduced preparation time by 50% and reduced scheduling activities, as it now takes two minutes to create the precedence graph. The flexibility of the algorithm's constraints allows for vehicle-specific configurations while maintaining high responsiveness, eliminating backup stations and facilitating the integration of new topologies. Automated scheduling significantly outperforms manual methods in efficiency, functionality, and adaptability. 

**Abstract (ZH)**: 本研究考察了汽车制造中的数字化价值链，着重于电动控制单元在车辆网络中的识别、软件烧录、定制和调试过程。提出了一种新颖的优先级图设计，通过使用混合整数线性规划技术的自动化调度算法来优化这一过程链。结果显示，在关键指标上取得了显著改进。该算法减少了用于执行数字化价值链过程的昂贵硬件和软件的生产站数量，通过高效的调度和减少闲置时间从而提高产能利用率。任务并行化得到了优化，实现了流畅的工作流程和更高的吞吐量。与传统方法相比，自动化方法将准备时间减少了50%，创建优先级图只需两分钟。算法的约束条件具有灵活性，允许针对特定车型进行配置，同时保持高响应性，消除了备用站并促进了新型拓扑结构的集成。自动化调度在效率、功能性和适应性方面显著优于手动方法。 

---
# On Solving the Dynamics of Constrained Rigid Multi-Body Systems with Kinematic Loops 

**Title (ZH)**: 关于解决具有 Kinematic 循环的约束刚体多体系统动力学问题的研究 

**Authors**: Vassilios Tsounis, Ruben Grandia, Moritz Bächer  

**Link**: [PDF](https://arxiv.org/pdf/2504.19771)  

**Abstract**: This technical report provides an in-depth evaluation of both established and state-of-the-art methods for simulating constrained rigid multi-body systems with hard-contact dynamics, using formulations of Nonlinear Complementarity Problems (NCPs). We are particularly interest in examining the simulation of highly coupled mechanical systems with multitudes of closed-loop bilateral kinematic joint constraints in the presence of additional unilateral constraints such as joint limits and frictional contacts with restitutive impacts. This work thus presents an up-to-date literature survey of the relevant fields, as well as an in-depth description of the approaches used for the formulation and solving of the numerical time-integration problem in a maximal coordinate setting. More specifically, our focus lies on a version of the overall problem that decomposes it into the forward dynamics problem followed by a time-integration using the states of the bodies and the constraint reactions rendered by the former. We then proceed to elaborate on the formulations used to model frictional contact dynamics and define a set of solvers that are representative of those currently employed in the majority of the established physics engines. A key aspect of this work is the definition of a benchmarking framework that we propose as a means to both qualitatively and quantitatively evaluate the performance envelopes of the set of solvers on a diverse set of challenging simulation scenarios. We thus present an extensive set of experiments that aim at highlighting the absolute and relative performance of all solvers on particular problems of interest as well as aggravatingly over the complete set defined in the suite. 

**Abstract (ZH)**: 本技术报告对使用非线性互补问题（NCPs）表述的约束刚体多体系统硬接触动力学模拟方法进行了深入评估，包括传统方法和当前最先进的方法。我们特别关注含有大量闭环双边运动学关节约束以及额外的单向约束（如关节限位和具有恢复性碰撞的摩擦接触）的强耦合机械系统的模拟。本工作因此提供了一个相关领域的最新文献综述，并对最大坐标设置下数值时间积分问题的建模和求解方法进行了深入描述。特别是，我们关注的是将整体问题分解为前向动力学问题，随后使用先前得到的身体状态和约束反作用力进行时间积分的方法。报告进一步详细描述了用于建模摩擦接触动力学的公式，并定义了一套代表当前广泛使用的物理引擎中所采用的各种求解器的集合。本研究的一个关键方面是定义了一个基准测试框架，作为评估所研究求解器性能边界的手段，无论是定性还是定量评价。因此，报告呈现了一系列全面的实验，旨在突出所有求解器在特定问题上的绝对和相对性能，并在整个测试套件中进行放大。 

---
# UTTG_ A Universal Teleoperation Approach via Online Trajectory Generation 

**Title (ZH)**: UTTG_基于在线轨迹生成的通用远程操作方法 

**Authors**: Shengjian Fang, Yixuan Zhou, Yu Zheng, Pengyu Jiang, Siyuan Liu, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19736)  

**Abstract**: Teleoperation is crucial for hazardous environment operations and serves as a key tool for collecting expert demonstrations in robot learning. However, existing methods face robotic hardware dependency and control frequency mismatches between teleoperation devices and robotic platforms. Our approach automatically extracts kinematic parameters from unified robot description format (URDF) files, and enables pluggable deployment across diverse robots through uniform interfaces. The proposed interpolation algorithm bridges the frequency gap between low-rate human inputs and high-frequency robotic control commands through online continuous trajectory generation, \n{while requiring no access to the closed, bottom-level control loop}. To enhance trajectory smoothness, we introduce a minimum-stretch spline that optimizes the motion quality. The system further provides precision and rapid modes to accommodate different task requirements. Experiments across various robotic platforms including dual-arm ones demonstrate generality and smooth operation performance of our methods. The code is developed in C++ with python interface, and available at this https URL. 

**Abstract (ZH)**: 遥操作对于危险环境作业至关重要，并且是机器人学习中收集专家演示的关键工具。然而，现有方法存在对机器人硬件的依赖以及遥操作设备与机器人平台之间控制频率不匹配的问题。我们的方法自动从统一机器人描述格式（URDF）文件中提取运动参数，并通过统一接口实现跨不同机器人的插件化部署。提出的插值算法通过在线连续轨迹生成，解决了低频人类输入与高频机器人控制命令之间的频率差距问题，且不需要访问关闭的底层控制回路。为了提升轨迹平滑性，我们引入了一种最小拉伸样条线来优化运动质量。系统还提供了精确模式和快速模式以适应不同的任务需求。来自各种机器人平台的实验证明了我们方法的普适性和平滑操作性能。代码使用C++编写，并通过Python接口提供，可在以下链接获取。 

---
# Hector UI: A Flexible Human-Robot User Interface for (Semi-)Autonomous Rescue and Inspection Robots 

**Title (ZH)**: Hector UI: 一种灵活的半自主救援与检测机器人用户界面 

**Authors**: Stefan Fabian, Oskar von Stryk  

**Link**: [PDF](https://arxiv.org/pdf/2504.19728)  

**Abstract**: The remote human operator's user interface (UI) is an important link to make the robot an efficient extension of the operator's perception and action. In rescue applications, several studies have investigated the design of operator interfaces based on observations during major robotics competitions or field deployments. Based on this research, guidelines for good interface design were empirically identified. The investigations on the UIs of teams participating in competitions are often based on external observations during UI application, which may miss some relevant requirements for UI flexibility. In this work, we present an open-source and flexibly configurable user interface based on established guidelines and its exemplary use for wheeled, tracked, and walking robots. We explain the design decisions and cover the insights we have gained during its highly successful applications in multiple robotics competitions and evaluations. The presented UI can also be adapted for other robots with little effort and is available as open source. 

**Abstract (ZH)**: 远程人类操作者的用户界面是使机器人成为操作者感知和行动高效延伸的重要环节。在救援应用中，多项研究基于重要机器人竞赛或现场部署期间的观察，调查了操作员界面的设计。基于这些研究，良好的界面设计原则被实证识别出来。对于参与竞赛团队的UI研究通常基于UI应用期间的外部观察，可能会遗漏一些UI灵活性的相关要求。在本工作中，我们提出了一种基于已确立原则的开源且可灵活配置的用户界面，并展示了其在履带式、轨道式和步行机器人上的典型应用。我们解释了设计决策，并涵盖了在其在多个机器人竞赛和评估中的广泛应用中获得的见解。所展示的UI也可以轻松适应其他机器人，并作为开源软件提供。 

---
# QuickGrasp: Lightweight Antipodal Grasp Planning with Point Clouds 

**Title (ZH)**: QuickGrasp: 基于点云的轻量级反平行夹持规划 

**Authors**: Navin Sriram Ravie, Keerthi Vasan M, Asokan Thondiyath, Bijo Sebastian  

**Link**: [PDF](https://arxiv.org/pdf/2504.19716)  

**Abstract**: Grasping has been a long-standing challenge in facilitating the final interface between a robot and the environment. As environments and tasks become complicated, the need to embed higher intelligence to infer from the surroundings and act on them has become necessary. Although most methods utilize techniques to estimate grasp pose by treating the problem via pure sampling-based approaches in the six-degree-of-freedom space or as a learning problem, they usually fail in real-life settings owing to poor generalization across domains. In addition, the time taken to generate the grasp plan and the lack of repeatability, owing to sampling inefficiency and the probabilistic nature of existing grasp planning approaches, severely limits their application in real-world tasks. This paper presents a lightweight analytical approach towards robotic grasp planning, particularly antipodal grasps, with little to no sampling in the six-degree-of-freedom space. The proposed grasp planning algorithm is formulated as an optimization problem towards estimating grasp points on the object surface instead of directly estimating the end-effector pose. To this extent, a soft-region-growing algorithm is presented for effective plane segmentation, even in the case of curved surfaces. An optimization-based quality metric is then used for the evaluation of grasp points to ensure indirect force closure. The proposed grasp framework is compared with the existing state-of-the-art grasp planning approach, Grasp pose detection (GPD), as a baseline over multiple simulated objects. The effectiveness of the proposed approach in comparison to GPD is also evaluated in a real-world setting using image and point-cloud data, with the planned grasps being executed using a ROBOTIQ gripper and UR5 manipulator. 

**Abstract (ZH)**: 基于最少六自由度空间采样的轻量级抗反 grasp 规划方法 

---
# Tensegrity-based Robot Leg Design with Variable Stiffness 

**Title (ZH)**: 基于张拉整体原理的 VARIABLE STIFFNESS 机器人腿设计 

**Authors**: Erik Mortensen, Jan Petrs, Alexander Dittrich, Dario Floreano  

**Link**: [PDF](https://arxiv.org/pdf/2504.19685)  

**Abstract**: Animals can finely modulate their leg stiffness to interact with complex terrains and absorb sudden shocks. In feats like leaping and sprinting, animals demonstrate a sophisticated interplay of opposing muscle pairs that actively modulate joint stiffness, while tendons and ligaments act as biological springs storing and releasing energy. Although legged robots have achieved notable progress in robust locomotion, they still lack the refined adaptability inherent in animal motor control. Integrating mechanisms that allow active control of leg stiffness presents a pathway towards more resilient robotic systems. This paper proposes a novel mechanical design to integrate compliancy into robot legs based on tensegrity - a structural principle that combines flexible cables and rigid elements to balance tension and compression. Tensegrity structures naturally allow for passive compliance, making them well-suited for absorbing impacts and adapting to diverse terrains. Our design features a robot leg with tensegrity joints and a mechanism to control the joint's rotational stiffness by modulating the tension of the cable actuation system. We demonstrate that the robot leg can reduce the impact forces of sudden shocks by at least 34.7 % and achieve a similar leg flexion under a load difference of 10.26 N by adjusting its stiffness configuration. The results indicate that tensegrity-based leg designs harbors potential towards more resilient and adaptable legged robots. 

**Abstract (ZH)**: 动物能够精细调节腿部刚度以应对复杂的地形并吸收突然的冲击。在跳跃和短跑等动作中，动物展示了对立肌肉群的复杂互动，主动调节关节刚度，而肌腱和韧带则作为生物弹簧储存和释放能量。尽管-legged robots在稳健移动方面取得了显著进展，但仍缺乏动物运动控制中固有的精细适应性。通过集成允许主动控制腿部刚度的机制，可以朝着更具韧性的机器人系统发展。本文提出了一种基于张拉整体原理的新型机械设计，将顺应性整合到机器人腿部中——张拉整体原理将柔性缆线和刚性元件结合在一起，以平衡拉力和压力。张拉整体结构天然地允许被动顺应性，使其适用于吸收冲击和适应各种地形。该设计包括一个具有张拉整体关节的机器人腿和一个通过调节缆线驱动系统张力来控制关节旋转刚度的机制。实验结果表明，该机器人腿可以通过调整刚度配置将突然冲击的冲击力降低至少34.7%，并在负载差为10.26 N的情况下实现相似的腿部弯曲。研究结果表明，基于张拉整体原理的腿部设计具有朝着更具韧性和适应性的腿式机器人发展的潜力。 

---
# GPA-RAM: Grasp-Pretraining Augmented Robotic Attention Mamba for Spatial Task Learning 

**Title (ZH)**: GPA-RAM: 抓取预训练增强的机器人注意力Mamba空间任务学习 

**Authors**: Juyi Sheng, Yangjun Liu, Sheng Xu, Zhixin Yang, Mengyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.19683)  

**Abstract**: Most existing robot manipulation methods prioritize task learning by enhancing perception through complex deep network architectures. However, they face challenges in real-time collision-free planning. Hence, Robotic Attention Mamba (RAM) is designed for refined planning. Specifically, by integrating Mamba and parallel single-view attention, RAM aligns multi-view vision and task-related language features, ensuring efficient fine-grained task planning with linear complexity and robust real-time performance. Nevertheless, it has the potential for further improvement in high-precision grasping and manipulation. Thus, Grasp-Pretraining Augmentation (GPA) is devised, with a grasp pose feature extractor pretrained utilizing object grasp poses directly inherited from whole-task demonstrations. Subsequently, the extracted grasp features are fused with the spatially aligned planning features from RAM through attention-based Pre-trained Location Fusion, preserving high-resolution grasping cues overshadowed by an overemphasis on global planning. To summarize, we propose Grasp-Pretraining Augmented Robotic Attention Mamba (GPA-RAM), dividing spatial task learning into RAM for planning skill learning and GPA for grasping skill learning. GPA-RAM demonstrates superior performance across three robot systems with distinct camera configurations in simulation and the real world. Compared with previous state-of-the-art methods, it improves the absolute success rate by 8.2% (from 79.3% to 87.5%) on the RLBench multi-task benchmark and 40\% (from 16% to 56%), 12% (from 86% to 98%) on the ALOHA bimanual manipulation tasks, while delivering notably faster inference. Furthermore, experimental results demonstrate that both RAM and GPA enhance task learning, with GPA proving robust to different architectures of pretrained grasp pose feature extractors. The website is: this https URL\_RAM\_website/. 

**Abstract (ZH)**: 基于抓取先-training增强的精细化机器人注意力规划方法（GPA-RAM） 

---
# Transformation & Translation Occupancy Grid Mapping: 2-Dimensional Deep Learning Refined SLAM 

**Title (ZH)**: 基于转换与翻译 occupancy 网格映射的二维深度学习增强SLAM 

**Authors**: Leon Davies, Baihua Li, Mohamad Saada, Simon Sølvsten, Qinggang Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.19654)  

**Abstract**: SLAM (Simultaneous Localisation and Mapping) is a crucial component for robotic systems, providing a map of an environment, the current location and previous trajectory of a robot. While 3D LiDAR SLAM has received notable improvements in recent years, 2D SLAM lags behind. Gradual drifts in odometry and pose estimation inaccuracies hinder modern 2D LiDAR-odometry algorithms in large complex environments. Dynamic robotic motion coupled with inherent estimation based SLAM processes introduce noise and errors, degrading map quality. Occupancy Grid Mapping (OGM) produces results that are often noisy and unclear. This is due to the fact that evidence based mapping represents maps according to uncertain observations. This is why OGMs are so popular in exploration or navigation tasks. However, this also limits OGMs' effectiveness for specific mapping based tasks such as floor plan creation in complex scenes. To address this, we propose our novel Transformation and Translation Occupancy Grid Mapping (TT-OGM). We adapt and enable accurate and robust pose estimation techniques from 3D SLAM to the world of 2D and mitigate errors to improve map quality using Generative Adversarial Networks (GANs). We introduce a novel data generation method via deep reinforcement learning (DRL) to build datasets large enough for training a GAN for SLAM error correction. We demonstrate our SLAM in real-time on data collected at Loughborough University. We also prove its generalisability on a variety of large complex environments on a collection of large scale well-known 2D occupancy maps. Our novel approach enables the creation of high quality OGMs in complex scenes, far surpassing the capabilities of current SLAM algorithms in terms of quality, accuracy and reliability. 

**Abstract (ZH)**: 基于转换与平移的 occupancy 栅格地图 (TT-OGM): 结合3D SLAM的精准_pose估计与生成对抗网络的SLAM误差校正 

---
# GAN-SLAM: Real-Time GAN Aided Floor Plan Creation Through SLAM 

**Title (ZH)**: GAN-SLAM：通过SLAM实现的实时GAN辅助平面图创建 

**Authors**: Leon Davies, Baihua Li, Mohamad Saada, Simon Sølvsten, Qinggang Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.19653)  

**Abstract**: SLAM is a fundamental component of modern autonomous systems, providing robots and their operators with a deeper understanding of their environment. SLAM systems often encounter challenges due to the dynamic nature of robotic motion, leading to inaccuracies in mapping quality, particularly in 2D representations such as Occupancy Grid Maps. These errors can significantly degrade map quality, hindering the effectiveness of specific downstream tasks such as floor plan creation. To address this challenge, we introduce our novel 'GAN-SLAM', a new SLAM approach that leverages Generative Adversarial Networks to clean and complete occupancy grids during the SLAM process, reducing the impact of noise and inaccuracies introduced on the output map. We adapt and integrate accurate pose estimation techniques typically used for 3D SLAM into a 2D form. This enables the quality improvement 3D LiDAR-odometry has seen in recent years to be effective for 2D representations. Our results demonstrate substantial improvements in map fidelity and quality, with minimal noise and errors, affirming the effectiveness of GAN-SLAM for real-world mapping applications within large-scale complex environments. We validate our approach on real-world data operating in real-time, and on famous examples of 2D maps. The improved quality of the output map enables new downstream tasks, such as floor plan drafting, further enhancing the capabilities of autonomous systems. Our novel approach to SLAM offers a significant step forward in the field, improving the usability for SLAM in mapping-based tasks, and offers insight into the usage of GANs for OGM error correction. 

**Abstract (ZH)**: 基于生成对抗网络的SLAM（GAN-SLAM）：提高二维地图质量的新方法 

---
# Robot Motion Planning using One-Step Diffusion with Noise-Optimized Approximate Motions 

**Title (ZH)**: 基于噪声优化近似运动的一步扩散的机器人运动规划 

**Authors**: Tomoharu Aizu, Takeru Oba, Yuki Kondo, Norimichi Ukita  

**Link**: [PDF](https://arxiv.org/pdf/2504.19652)  

**Abstract**: This paper proposes an image-based robot motion planning method using a one-step diffusion model. While the diffusion model allows for high-quality motion generation, its computational cost is too expensive to control a robot in real time. To achieve high quality and efficiency simultaneously, our one-step diffusion model takes an approximately generated motion, which is predicted directly from input images. This approximate motion is optimized by additive noise provided by our novel noise optimizer. Unlike general isotropic noise, our noise optimizer adjusts noise anisotropically depending on the uncertainty of each motion element. Our experimental results demonstrate that our method outperforms state-of-the-art methods while maintaining its efficiency by one-step diffusion. 

**Abstract (ZH)**: 基于图像的一步扩散模型机器人运动规划方法 

---
# ARMOR: Adaptive Meshing with Reinforcement Optimization for Real-time 3D Monitoring in Unexposed Scenes 

**Title (ZH)**: ARMOR：适用于未暴露场景的自适应网格生成与强化优化实时3D监控 

**Authors**: Yizhe Zhang, Jianping Li, Xin Zhao, Fuxun Liang, Zhen Dong, Bisheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19624)  

**Abstract**: Unexposed environments, such as lava tubes, mines, and tunnels, are among the most complex yet strategically significant domains for scientific exploration and infrastructure development. Accurate and real-time 3D meshing of these environments is essential for applications including automated structural assessment, robotic-assisted inspection, and safety monitoring. Implicit neural Signed Distance Fields (SDFs) have shown promising capabilities in online meshing; however, existing methods often suffer from large projection errors and rely on fixed reconstruction parameters, limiting their adaptability to complex and unstructured underground environments such as tunnels, caves, and lava tubes. To address these challenges, this paper proposes ARMOR, a scene-adaptive and reinforcement learning-based framework for real-time 3D meshing in unexposed environments. The proposed method was validated across more than 3,000 meters of underground environments, including engineered tunnels, natural caves, and lava tubes. Experimental results demonstrate that ARMOR achieves superior performance in real-time mesh reconstruction, reducing geometric error by 3.96\% compared to state-of-the-art baselines, while maintaining real-time efficiency. The method exhibits improved robustness, accuracy, and adaptability, indicating its potential for advanced 3D monitoring and mapping in challenging unexposed scenarios. The project page can be found at: this https URL 

**Abstract (ZH)**: 未暴露环境的实时3D网状构建：ARMOR——一种场景自适应和基于强化学习的框架 

---
# Adaptive Locomotion on Mud through Proprioceptive Sensing of Substrate Properties 

**Title (ZH)**: 通过本体感觉 substrate 属性的适配性运动 

**Authors**: Shipeng Liu, Jiaze Tang, Siyuan Meng, Feifei Qian  

**Link**: [PDF](https://arxiv.org/pdf/2504.19607)  

**Abstract**: Muddy terrains present significant challenges for terrestrial robots, as subtle changes in composition and water content can lead to large variations in substrate strength and force responses, causing the robot to slip or get stuck. This paper presents a method to estimate mud properties using proprioceptive sensing, enabling a flipper-driven robot to adapt its locomotion through muddy substrates of varying strength. First, we characterize mud reaction forces through actuator current and position signals from a statically mounted robotic flipper. We use the measured force to determine key coefficients that characterize intrinsic mud properties. The proprioceptively estimated coefficients match closely with measurements from a lab-grade load cell, validating the effectiveness of the proposed method. Next, we extend the method to a locomoting robot to estimate mud properties online as it crawls across different mud mixtures. Experimental data reveal that mud reaction forces depend sensitively on robot motion, requiring joint analysis of robot movement with proprioceptive force to determine mud properties correctly. Lastly, we deploy this method in a flipper-driven robot moving across muddy substrates of varying strengths, and demonstrate that the proposed method allows the robot to use the estimated mud properties to adapt its locomotion strategy, and successfully avoid locomotion failures. Our findings highlight the potential of proprioception-based terrain sensing to enhance robot mobility in complex, deformable natural environments, paving the way for more robust field exploration capabilities. 

**Abstract (ZH)**: 软泥地形给地面机器人带来了显著挑战，因为组成成分和水分含量的细微变化会导致基质强度和力响应产生较大变化，从而使机器人出现打滑或卡住的情况。本文提出了一种利用本体感觉估计泥地属性的方法，使鳍驱动机器人能够适应不同强度泥地地形的运动。首先，我们通过静置安装的机器人鳍片的执行器电流和位置信号来表征泥地的反应力，使用测量的力来确定描述内在泥地属性的关键系数。通过本体感觉估计得到的系数与实验室级载荷细胞测量值高度一致，验证了该方法的有效性。接下来，我们将该方法扩展到行进机器人，以在线估计机器人爬行过程中不同混合泥地的属性。实验数据表明，泥地的反应力对机器人运动高度敏感，需要结合机器人运动和本体感觉力的综合分析来正确确定泥地的属性。最后，我们将在不同强度泥地地形上移动的鳍驱动机器人中部署该方法，并证明所提出的方法使机器人能够利用估计的泥地属性来调整其运动策略，从而成功避免运动失败。我们的研究结果突显了基于本体感觉的地形感知在复杂可变形自然环境中的潜力，为更可靠的野外探索能力铺平了道路。 

---
# A Time-dependent Risk-aware distributed Multi-Agent Path Finder based on A* 

**Title (ZH)**: 基于A*的时间依赖风险意识分布式多Agent路径规划器 

**Authors**: S Nordström, Y Bai, B Lindqvist, G Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2504.19593)  

**Abstract**: Multi-Agent Path-Finding (MAPF) focuses on the collaborative planning of paths for multiple agents within shared spaces, aiming for collision-free navigation. Conventional planning methods often overlook the presence of other agents, which can result in conflicts. In response, this article introduces the A$^*_+$T algorithm, a distributed approach that improves coordination among agents by anticipating their positions based on their movement speeds. The algorithm also considers dynamic obstacles, assessing potential collisions with respect to observed speeds and trajectories, thereby facilitating collision-free path planning in environments populated by other agents and moving objects. It incorporates a risk layer surrounding both dynamic and static entities, enhancing its utility in real-world applications. Each agent functions autonomously while being mindful of the paths chosen by others, effectively addressing the complexities inherent in multi-agent situations. The performance of A$^*_+$T has been rigorously tested in the Gazebo simulation environment and benchmarked against established approaches such as CBS, ECBS, and SIPP. Furthermore, the algorithm has shown competence in single-agent experiments, with results demonstrating its effectiveness in managing dynamic obstacles and affirming its practical relevance across various scenarios. 

**Abstract (ZH)**: 多代理路径寻找(Multi-Agent Path-Finding)专注于在共享空间内多个代理的协作路径规划，旨在实现无碰撞导航。传统的规划方法往往忽视其他代理的存在，可能导致冲突。为应对这一挑战，本文引入了A$^*_+$T算法，这是一种分布式方法，通过预测代理的运动位置来改善多代理之间的协调，同时考虑动态障碍物，评估潜在碰撞，从而在包含其他代理和移动物体的环境中实现无碰撞路径规划。该算法在动态和静态实体周围引入了一个风险层，增强了其实用性。每个代理在自主运行的同时，也考虑了其他代理选择的路径，有效解决了多代理情况下的复杂性。A$^*_+$T算法已在Gazebo仿真环境中严格测试，并与CBS、ECBS和SIPP等现有方法进行了对比。此外，该算法在单代理实验中也表现出色，结果证明其在管理动态障碍物方面的有效性，并且在各种场景中具有实际的相关性。 

---
# ARTEMIS: Autoregressive End-to-End Trajectory Planning with Mixture of Experts for Autonomous Driving 

**Title (ZH)**: ARTEMIS：基于专家混合的自回归端到端轨迹规划自主驾驶 

**Authors**: Renju Feng, Ning Xi, Duanfeng Chu, Rukang Wang, Zejian Deng, Anzheng Wang, Liping Lu, Jinxiang Wang, Yanjun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19580)  

**Abstract**: This paper presents ARTEMIS, an end-to-end autonomous driving framework that combines autoregressive trajectory planning with Mixture-of-Experts (MoE). Traditional modular methods suffer from error propagation, while existing end-to-end models typically employ static one-shot inference paradigms that inadequately capture the dynamic changes of the environment. ARTEMIS takes a different method by generating trajectory waypoints sequentially, preserves critical temporal dependencies while dynamically routing scene-specific queries to specialized expert networks. It effectively relieves trajectory quality degradation issues encountered when guidance information is ambiguous, and overcomes the inherent representational limitations of singular network architectures when processing diverse driving scenarios. Additionally, we use a lightweight batch reallocation strategy that significantly improves the training speed of the Mixture-of-Experts model. Through experiments on the NAVSIM dataset, ARTEMIS exhibits superior competitive performance, achieving 87.0 PDMS and 83.1 EPDMS with ResNet-34 backbone, demonstrates state-of-the-art performance on multiple metrics. 

**Abstract (ZH)**: ARTEMIS：一种结合自回归轨迹规划与Mixture-of-Experts的端到端自主驾驶框架 

---
# Smart Placement, Faster Robots -- A Comparison of Algorithms for Robot Base-Pose Optimization 

**Title (ZH)**: 智能定位，更快的机器人——机器人基座姿态优化算法对比 

**Authors**: Matthias Mayer, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2504.19577)  

**Abstract**: Robotic automation is a key technology that increases the efficiency and flexibility of manufacturing processes. However, one of the challenges in deploying robots in novel environments is finding the optimal base pose for the robot, which affects its reachability and deployment cost. Yet, the existing research for automatically optimizing the base pose of robots has not been compared. We address this problem by optimizing the base pose of industrial robots with Bayesian optimization, exhaustive search, genetic algorithms, and stochastic gradient descent and find that all algorithms can reduce the cycle time for various evaluated tasks in synthetic and real-world environments. Stochastic gradient descent shows superior performance with regard to success rate solving over 90% of our real-world tasks, while genetic algorithms show the lowest final costs. All benchmarks and implemented methods are available as baselines against which novel approaches can be compared. 

**Abstract (ZH)**: 机器人自动化是提高制造过程效率和灵活性的关键技术。然而，在部署机器人到新型环境中时，找到最优基座姿态以影响其可达性和部署成本的一个挑战是现有的自动优化机器人基座姿态的研究尚未进行比较。我们通过使用 Bayesian 优化、穷举搜索、遗传算法和随机梯度下降来优化工业机器人的基座姿态，并发现所有算法都能在合成和实际环境中减少各种评估任务的周期时间。随机梯度下降在解决超过90%的实际任务成功率方面表现出色，而遗传算法具有最低的最终成本。所有基准和实现方法均可作为新型方法的比较基准。 

---
# Video-Based Detection and Analysis of Errors in Robotic Surgical Training 

**Title (ZH)**: 基于视频的机器人手术培训中错误的检测与分析 

**Authors**: Hanna Kossowsky Lev, Yarden Sharon, Alex Geftler, Ilana Nisky  

**Link**: [PDF](https://arxiv.org/pdf/2504.19571)  

**Abstract**: Robot-assisted minimally invasive surgeries offer many advantages but require complex motor tasks that take surgeons years to master. There is currently a lack of knowledge on how surgeons acquire these robotic surgical skills. To help bridge this gap, we previously followed surgical residents learning complex surgical training dry-lab tasks on a surgical robot over six months. Errors are an important measure for self-training and for skill evaluation, but unlike in virtual simulations, in dry-lab training, errors are difficult to monitor automatically. Here, we analyzed the errors in the ring tower transfer task, in which surgical residents moved a ring along a curved wire as quickly and accurately as possible. We developed an image-processing algorithm to detect collision errors and achieved detection accuracy of ~95%. Using the detected errors and task completion time, we found that the surgical residents decreased their completion time and number of errors over the six months. This analysis provides a framework for detecting collision errors in similar surgical training tasks and sheds light on the learning process of the surgical residents. 

**Abstract (ZH)**: 机器人辅助微创手术提供了许多优势，但要求外科医生掌握复杂的运动任务，这需要数年时间。目前尚缺乏关于外科医生如何获得这些机器人手术技能的知识。为了填补这一空白，我们之前在六个月内跟踪了外科住院医师在手术机器人上学习复杂手术培训Dry Lab任务的过程。错误是自我训练和技能评估的重要指标，但在Dry Lab培训中，与虚拟模拟不同，错误难以自动监测。在此，我们分析了环塔楼转移任务中的错误，在该任务中，外科住院医师尽可能快速和准确地沿弯曲线移动环。我们开发了一种图像处理算法来检测碰撞错误，并实现了约95%的检测准确性。利用检测到的错误和任务完成时间，我们发现外科住院医师在六个月内减少了完成时间和错误数量。这一分析为检测类似手术培训任务中的碰撞错误提供了框架，并揭示了外科住院医师的学习过程。 

---
# Simultaneous Pick and Place Detection by Combining SE(3) Diffusion Models with Differential Kinematics 

**Title (ZH)**: 结合SE(3)扩散模型与差分运动学的同时拾放检测 

**Authors**: Tianyi Ko, Takuya Ikeda, Koichi Nishiwaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.19502)  

**Abstract**: Grasp detection methods typically target the detection of a set of free-floating hand poses that can grasp the object. However, not all of the detected grasp poses are executable due to physical constraints. Even though it is straightforward to filter invalid grasp poses in the post-process, such a two-staged approach is computationally inefficient, especially when the constraint is hard. In this work, we propose an approach to take the following two constraints into account during the grasp detection stage, namely, (i) the picked object must be able to be placed with a predefined configuration without in-hand manipulation (ii) it must be reachable by the robot under the joint limit and collision-avoidance constraints for both pick and place cases. Our key idea is to train an SE(3) grasp diffusion network to estimate the noise in the form of spatial velocity, and constrain the denoising process by a multi-target differential inverse kinematics with an inequality constraint, so that the states are guaranteed to be reachable and placement can be performed without collision. In addition to an improved success ratio, we experimentally confirmed that our approach is more efficient and consistent in computation time compared to a naive two-stage approach. 

**Abstract (ZH)**: 基于运动学约束的抓取检测方法 

---
# Motion Generation for Food Topping Challenge 2024: Serving Salmon Roe Bowl and Picking Fried Chicken 

**Title (ZH)**: 2024食品配料挑战中的运动生成：提供鲑鱼子碗和捡起炸鸡 

**Authors**: Koki Inami, Masashi Konosu, Koki Yamane, Nozomu Masuya, Yunhan Li, Yu-Han Shu, Hiroshi Sato, Shinnosuke Homma, Sho Sakaino  

**Link**: [PDF](https://arxiv.org/pdf/2504.19498)  

**Abstract**: Although robots have been introduced in many industries, food production robots are yet to be widely employed because the food industry requires not only delicate movements to handle food but also complex movements that adapt to the environment. Force control is important for handling delicate objects such as food. In addition, achieving complex movements is possible by making robot motions based on human teachings. Four-channel bilateral control is proposed, which enables the simultaneous teaching of position and force information. Moreover, methods have been developed to reproduce motions obtained through human teachings and generate adaptive motions using learning. We demonstrated the effectiveness of these methods for food handling tasks in the Food Topping Challenge at the 2024 IEEE International Conference on Robotics and Automation (ICRA 2024). For the task of serving salmon roe on rice, we achieved the best performance because of the high reproducibility and quick motion of the proposed method. Further, for the task of picking fried chicken, we successfully picked the most pieces of fried chicken among all participating teams. This paper describes the implementation and performance of these methods. 

**Abstract (ZH)**: 尽管机器人已在许多行业中应用，但由于食品行业需要精细操作以处理食物并适应复杂环境，食品生产机器人尚未广泛应用。力控制对于处理如食物等精细物体至关重要。此外，通过基于人类教学的机器人运动，可以实现复杂运动。本文提出了四通道双边控制方法，能够同时教授位置和力信息。同时，已经开发出再现通过人类教学获得的运动并利用学习生成适应性运动的方法。我们在2024年IEEE机器人与自动化国际会议（ICRA 2024）的食品配料挑战赛中展示了这些方法的有效性。对于在日本寿司饭上摆放鲑鱼子的任务，由于所提出方法的高再现性和快速运动，我们取得了最佳性能。此外，对于鸡块抓取任务，我们成功地在所有参赛队伍中抓取了最多的鸡块。本文描述了这些方法的实现及其性能。 

---
# Bearing-Only Tracking and Circumnavigation of a Fast Time-Varied Velocity Target Utilising an LSTM 

**Title (ZH)**: 基于LSTM的快速时变速度目标仅 Bearings 跟踪与环绕控制 

**Authors**: Mitchell Torok, Mohammad Deghat, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.19463)  

**Abstract**: Bearing-only tracking, localisation, and circumnavigation is a problem in which a single or a group of agents attempts to track a target while circumnavigating it at a fixed distance using only bearing measurements. While previous studies have addressed scenarios involving stationary targets or those moving with an unknown constant velocity, the challenge of accurately tracking a target moving with a time-varying velocity remains open. This paper presents an approach utilising a Long Short-Term Memory (LSTM) based estimator for predicting the target's position and velocity. We also introduce a corresponding control strategy. When evaluated against previously proposed estimation and circumnavigation approaches, our approach demonstrates significantly lower control and estimation errors across various time-varying velocity scenarios. Additionally, we illustrate the effectiveness of the proposed method in tracking targets with a double integrator nonholonomic system dynamics that mimic real-world systems. 

**Abstract (ZH)**: 仅凭航向角的目标跟踪、定位和环航是一个问题，即单个或多个代理试图使用仅有的航向测量数据，在固定距离上跟踪目标并沿其环航。尽管先前的研究已经处理了静止目标或未知恒定速度移动目标的情形，但准确跟踪具有时间变化速度的目标的挑战仍然存在。本文提出了一种利用基于长短期记忆（LSTM）的估计算法来预测目标的位置和速度的方法，并介绍了相应的控制策略。当与先前提出的估计和环航方法进行对比评估时，本文提出的方法在各种时间变化速度场景中表现出显著较低的控制和估计误差。此外，本文还展示了在模拟真实系统动力学的双积分非完整系统动力学下，提出的方法在跟踪目标方面的有效性。 

---
# An End-to-End Framework for Optimizing Foot Trajectory and Force in Dry Adhesion Legged Wall-Climbing Robots 

**Title (ZH)**: 一种优化足轨迹和干黏附壁爬机器人接触力的端到端框架 

**Authors**: Jichun Xiao, Jiawei Nie, Lina Hao, Zhi Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.19448)  

**Abstract**: Foot trajectory planning for dry adhesion legged climbing robots presents challenges, as the phases of foot detachment, swing, and adhesion significantly influence the adhesion and detachment forces essential for stable climbing. To tackle this, an end-to-end foot trajectory and force optimization framework (FTFOF) is proposed, which optimizes foot adhesion and detachment forces through trajectory adjustments. This framework accepts general foot trajectory constraints and user-defined parameters as input, ultimately producing an optimal single foot trajectory. It integrates three-segment $C^2$ continuous Bezier curves, tailored to various foot structures, enabling the generation of effective climbing trajectories. A dilate-based GRU predictive model establishes the relationship between foot trajectories and the corresponding foot forces. Multi-objective optimization algorithms, combined with a redundancy hierarchical strategy, identify the most suitable foot trajectory for specific tasks, thereby ensuring optimal performance across detachment force, adhesion force and vibration amplitude. Experimental validation on the quadruped climbing robot MST-M3F showed that, compared to commonly used trajectories in existing legged climbing robots, the proposed framework achieved reductions in maximum detachment force by 28 \%, vibration amplitude by 82 \%, which ensures the stable climbing of dry adhesion legged climbing robots. 

**Abstract (ZH)**: 基于干粘附的腿足攀爬机器人足轨迹规划与力优化框架 

---
# GSFF-SLAM: 3D Semantic Gaussian Splatting SLAM via Feature Field 

**Title (ZH)**: GSFF-SLAM: 3D语义高斯点云SLAM via 特征场 

**Authors**: Zuxing Lu, Xin Yuan, Shaowen Yang, Jingyu Liu, Jiawei Wang, Changyin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.19409)  

**Abstract**: Semantic-aware 3D scene reconstruction is essential for autonomous robots to perform complex interactions. Semantic SLAM, an online approach, integrates pose tracking, geometric reconstruction, and semantic mapping into a unified framework, shows significant potential. However, existing systems, which rely on 2D ground truth priors for supervision, are often limited by the sparsity and noise of these signals in real-world environments. To address this challenge, we propose GSFF-SLAM, a novel dense semantic SLAM system based on 3D Gaussian Splatting that leverages feature fields to achieve joint rendering of appearance, geometry, and N-dimensional semantic features. By independently optimizing feature gradients, our method supports semantic reconstruction using various forms of 2D priors, particularly sparse and noisy signals. Experimental results demonstrate that our approach outperforms previous methods in both tracking accuracy and photorealistic rendering quality. When utilizing 2D ground truth priors, GSFF-SLAM achieves state-of-the-art semantic segmentation performance with 95.03\% mIoU, while achieving up to 2.9$\times$ speedup with only marginal performance degradation. 

**Abstract (ZH)**: 基于3D高斯点扩散的语义SLAM系统：实现联合建模外观、几何和N维语义特征 

---
# Follow Everything: A Leader-Following and Obstacle Avoidance Framework with Goal-Aware Adaptation 

**Title (ZH)**: 跟随一切：一种带有目标意识适应性的领导者跟随与避障框架 

**Authors**: Qianyi Zhang, Shijian Ma, Boyi Liu, Jingtai Liu, Jianhao Jiao, Dimitrios Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2504.19399)  

**Abstract**: Robust and flexible leader-following is a critical capability for robots to integrate into human society. While existing methods struggle to generalize to leaders of arbitrary form and often fail when the leader temporarily leaves the robot's field of view, this work introduces a unified framework addressing both challenges. First, traditional detection models are replaced with a segmentation model, allowing the leader to be anything. To enhance recognition robustness, a distance frame buffer is implemented that stores leader embeddings at multiple distances, accounting for the unique characteristics of leader-following tasks. Second, a goal-aware adaptation mechanism is designed to govern robot planning states based on the leader's visibility and motion, complemented by a graph-based planner that generates candidate trajectories for each state, ensuring efficient following with obstacle avoidance. Simulations and real-world experiments with a legged robot follower and various leaders (human, ground robot, UAV, legged robot, stop sign) in both indoor and outdoor environments show competitive improvements in follow success rate, reduced visual loss duration, lower collision rate, and decreased leader-follower distance. 

**Abstract (ZH)**: 鲁棒且灵活的跟随能力是机器人融入人类社会的关键能力。现有方法难以泛化到任意形式的领导者，并且当领导者暂时离开机器人的视野范围时往往无法有效应对，本研究提出了一种统一框架以应对上述挑战。首先，传统的检测模型被分割模型所取代，使得领导者可以是任意物体。为了增强识别的鲁棒性，实现了一个距离帧缓冲区，该缓冲区存储了不同距离的领导者嵌入，以考虑跟随任务的独特特性。其次，设计了一个目标感知的自适应机制，根据领导者的可见性和运动来管理机器人的规划状态，配合基于图的规划器为每个状态生成候选轨迹，确保高效跟随的同时实现避障。在室内和室外环境中，使用四足机器人跟随者和多种领导者（人类、地面机器人、UAV、四足机器人、停止标志）进行的仿真和实地实验显示，在跟随成功率、视觉丧失时间、碰撞率和领导者跟随者距离方面均取得了显著改进。 

---
# PolyTouch: A Robust Multi-Modal Tactile Sensor for Contact-rich Manipulation Using Tactile-Diffusion Policies 

**Title (ZH)**: PolyTouch: 一种用于接触丰富操作的鲁棒多模态触觉传感器及触觉扩散策略 

**Authors**: Jialiang Zhao, Naveen Kuppuswamy, Siyuan Feng, Benjamin Burchfiel, Edward Adelson  

**Link**: [PDF](https://arxiv.org/pdf/2504.19341)  

**Abstract**: Achieving robust dexterous manipulation in unstructured domestic environments remains a significant challenge in robotics. Even with state-of-the-art robot learning methods, haptic-oblivious control strategies (i.e. those relying only on external vision and/or proprioception) often fall short due to occlusions, visual complexities, and the need for precise contact interaction control. To address these limitations, we introduce PolyTouch, a novel robot finger that integrates camera-based tactile sensing, acoustic sensing, and peripheral visual sensing into a single design that is compact and durable. PolyTouch provides high-resolution tactile feedback across multiple temporal scales, which is essential for efficiently learning complex manipulation tasks. Experiments demonstrate an at least 20-fold increase in lifespan over commercial tactile sensors, with a design that is both easy to manufacture and scalable. We then use this multi-modal tactile feedback along with visuo-proprioceptive observations to synthesize a tactile-diffusion policy from human demonstrations; the resulting contact-aware control policy significantly outperforms haptic-oblivious policies in multiple contact-aware manipulation policies. This paper highlights how effectively integrating multi-modal contact sensing can hasten the development of effective contact-aware manipulation policies, paving the way for more reliable and versatile domestic robots. More information can be found at this https URL 

**Abstract (ZH)**: 在无结构家庭环境中实现稳健的灵巧 manipulation 仍然是机器人技术中的一个重大挑战。即使在最先进的机器人学习方法中，仅依赖外部视觉和/或本体感觉的触觉无感知控制策略（即，由于遮挡、视觉复杂性和精确接触交互控制的需要）往往仍会遇到障碍。为了解决这些限制，我们引入了 PolyTouch，这是一种集成了基于相机的触觉传感、声学传感和周视视觉传感的新型机器人手指，设计紧凑且坚固。PolyTouch 在多个时间尺度上提供了高分辨率的触觉反馈，这对于高效学习复杂的 manipulation 任务至关重要。实验表明，PolyTouch 的寿命至少比商用触觉传感器高 20 倍，且设计易于制造且可扩展。然后，我们利用这种多模态触觉反馈以及视觉-本体感受观察来从人类演示中合成触觉扩散策略；生成的接触感知控制策略在多种接触感知的 manipulation 策略中显著优于触觉无感知策略。本文突显了如何有效地集成多模态接触传感可以加速接触感知 manipulation 策略的发展，为更可靠和多功能的家庭机器人铺平道路。更多信息请访问 <https://github.com/alibaba/PolyTouch>。 

---
# Learned Perceptive Forward Dynamics Model for Safe and Platform-aware Robotic Navigation 

**Title (ZH)**: 学习感知前向动力学模型以实现安全且平台感知的机器人导航 

**Authors**: Pascal Roth, Jonas Frey, Cesar Cadena, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2504.19322)  

**Abstract**: Ensuring safe navigation in complex environments requires accurate real-time traversability assessment and understanding of environmental interactions relative to the robot`s capabilities. Traditional methods, which assume simplified dynamics, often require designing and tuning cost functions to safely guide paths or actions toward the goal. This process is tedious, environment-dependent, and not this http URL overcome these issues, we propose a novel learned perceptive Forward Dynamics Model (FDM) that predicts the robot`s future state conditioned on the surrounding geometry and history of proprioceptive measurements, proposing a more scalable, safer, and heuristic-free solution. The FDM is trained on multiple years of simulated navigation experience, including high-risk maneuvers, and real-world interactions to incorporate the full system dynamics beyond rigid body simulation. We integrate our perceptive FDM into a zero-shot Model Predictive Path Integral (MPPI) planning framework, leveraging the learned mapping between actions, future states, and failure probability. This allows for optimizing a simplified cost function, eliminating the need for extensive cost-tuning to ensure safety. On the legged robot ANYmal, the proposed perceptive FDM improves the position estimation by on average 41% over competitive baselines, which translates into a 27% higher navigation success rate in rough simulation environments. Moreover, we demonstrate effective sim-to-real transfer and showcase the benefit of training on synthetic and real data. Code and models are made publicly available under this https URL. 

**Abstract (ZH)**: 确保在复杂环境中的安全导航需要进行准确的实时通行性评估及理解环境交互相对于机器人能力的关联。传统的假设简化动力学的方法通常需要设计和调整成本函数以安全地指导路径或动作朝向目标。这一过程繁琐，且依赖于环境，我们提出了一种新的基于学习的感知前向动力学模型（FDM），该模型可根据周围的几何结构和本体感知测量的历史预测机器人的未来状态，提供一种更为可扩展、更安全且无需启发式的解决方案。该FDM基于多年模拟导航经验进行训练，包括高风险机动和真实世界交互，以涵盖超出刚体模拟的整个系统动力学。我们将感知FDM整合到零样本模型预测路径积分（MPPI）规划框架中，利用动作、未来状态和故障概率之间的学习映射进行优化，从而消除广泛成本调优以确保安全的需要。在腿式机器人ANYmal上，提出的感知FDM在位置估计上平均改善了41%，在粗糙的模拟环境中将导航成功率提高了27%。此外，我们展示了有效的仿真实验转移，并展示了在合成和真实数据上进行训练的好处。代码和模型已在此处公开。 

---
# Unscented Particle Filter for Visual-inertial Navigation using IMU and Landmark Measurements 

**Title (ZH)**: 基于IMU和地标测量的无迹粒子滤波视觉-惯性导航 

**Authors**: Khashayar Ghanizadegan, Hashim A. Hashim  

**Link**: [PDF](https://arxiv.org/pdf/2504.19318)  

**Abstract**: This paper introduces a geometric Quaternion-based Unscented Particle Filter for Visual-Inertial Navigation (QUPF-VIN) specifically designed for a vehicle operating with six degrees of freedom (6 DoF). The proposed QUPF-VIN technique is quaternion-based capturing the inherently nonlinear nature of true navigation kinematics. The filter fuses data from a low-cost inertial measurement unit (IMU) and landmark observations obtained via a vision sensor. The QUPF-VIN is implemented in discrete form to ensure seamless integration with onboard inertial sensing systems. Designed for robustness in GPS-denied environments, the proposed method has been validated through experiments with real-world dataset involving an unmanned aerial vehicle (UAV) equipped with a 6-axis IMU and a stereo camera, operating with 6 DoF. The numerical results demonstrate that the QUPF-VIN provides superior tracking accuracy compared to ground truth data. Additionally, a comparative analysis with a standard Kalman filter-based navigation technique further highlights the enhanced performance of the QUPF-VIN. 

**Abstract (ZH)**: 基于几何四元数的无迹粒子滤波视觉惯性导航（QUPF-VIN）技术及其在六自由度车辆中的应用 

---
# Quantitative evaluation of brain-inspired vision sensors in high-speed robotic perception 

**Title (ZH)**: 基于大脑启发的视觉传感器在高速机器人感知中的定量评价 

**Authors**: Taoyi Wang, Lijian Wang, Yihan Lin, Mingtao Ou, Yuguo Chen, Xinglong Ji, Rong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.19253)  

**Abstract**: Perception systems in robotics encounter significant challenges in high-speed and dynamic conditions when relying on traditional cameras, where motion blur can compromise spatial feature integrity and task performance. Brain-inspired vision sensors (BVS) have recently gained attention as an alternative, offering high temporal resolution with reduced bandwidth and power requirements. Here, we present the first quantitative evaluation framework for two representative classes of BVSs in variable-speed robotic sensing, including event-based vision sensors (EVS) that detect asynchronous temporal contrasts, and the primitive-based sensor Tianmouc that employs a complementary mechanism to encode both spatiotemporal changes and intensity. A unified testing protocol is established, including crosssensor calibrations, standardized testing platforms, and quality metrics to address differences in data modality. From an imaging standpoint, we evaluate the effects of sensor non-idealities, such as motion-induced distortion, on the capture of structural information. For functional benchmarking, we examine task performance in corner detection and motion estimation under different rotational speeds. Results indicate that EVS performs well in highspeed, sparse scenarios and in modestly fast, complex scenes, but exhibits performance limitations in high-speed, cluttered settings due to pixel-level bandwidth variations and event rate saturation. In comparison, Tianmouc demonstrates consistent performance across sparse and complex scenarios at various speeds, supported by its global, precise, high-speed spatiotemporal gradient samplings. These findings offer valuable insights into the applicationdependent suitability of BVS technologies and support further advancement in this area. 

**Abstract (ZH)**: 基于脑启发视觉传感器在可变速度机器人感知中的量化评估框架 

---
# Efficient COLREGs-Compliant Collision Avoidance using Turning Circle-based Control Barrier Function 

**Title (ZH)**: 基于转向圈基控制约束函数的有效遵 Compliance 航行规则避碰控制 

**Authors**: Changyu Lee, Jinwook Park, Jinwhan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.19247)  

**Abstract**: This paper proposes a computationally efficient collision avoidance algorithm using turning circle-based control barrier functions (CBFs) that comply with international regulations for preventing collisions at sea (COLREGs). Conventional CBFs often lack explicit consideration of turning capabilities and avoidance direction, which are key elements in developing a COLREGs-compliant collision avoidance algorithm. To overcome these limitations, we introduce two CBFs derived from left and right turning circles. These functions establish safety conditions based on the proximity between the traffic ships and the centers of the turning circles, effectively determining both avoidance directions and turning capabilities. The proposed method formulates a quadratic programming problem with the CBFs as constraints, ensuring safe navigation without relying on computationally intensive trajectory optimization. This approach significantly reduces computational effort while maintaining performance comparable to model predictive control-based methods. Simulation results validate the effectiveness of the proposed algorithm in enabling COLREGs-compliant, safe navigation, demonstrating its potential for reliable and efficient operation in complex maritime environments. 

**Abstract (ZH)**: 本文提出了一种基于转向圆控制障碍函数（CBFs）的计算高效避碰算法，并符合国际海上避碰规则（COLREGs）。传统CBFs往往未能明确考虑转向能力和避碰方向，这是制定符合COLREGs的避碰算法的关键要素。为克服这些限制，我们引入了源自左转和右转转向圆的两种CBFs。这些函数基于航分区船舶与转向圆中心之间的接近程度，有效地确定了避碰方向和转向能力。所提出的方法将CBFs作为约束形式ulating一个二次规划问题，确保安全航行而不依赖于计算 intensive的轨迹优化。该方法显著减少了计算负担，同时保持与基于模型预测控制的方法相当的性能。仿真结果验证了所提出算法在实现COLREGs合规、安全航行方面的有效性，展示了其在复杂 maritime环境下可靠和高效操作的潜力。 

---
# Robotic Trail Maker Platform for Rehabilitation in Neurological Conditions: Clinical Use Cases 

**Title (ZH)**: 神经科康复中的机器人路径铺设平台：临床案例 

**Authors**: Srikar Annamraju, Harris Nisar, Dayu Xia, Shankar A. Deka, Anne Horowitz, Nadica Miljković, Dušan M. Stipanović  

**Link**: [PDF](https://arxiv.org/pdf/2504.19230)  

**Abstract**: Patients with neurological conditions require rehabilitation to restore their motor, visual, and cognitive abilities. To meet the shortage of therapists and reduce their workload, a robotic rehabilitation platform involving the clinical trail making test is proposed. Therapists can create custom trails for each patient and the patient can trace the trails using a robotic device. The platform can track the performance of the patient and use these data to provide dynamic assistance through the robot to the patient interface. Therefore, the proposed platform not only functions as an evaluation platform, but also trains the patient in recovery. The developed platform has been validated at a rehabilitation center, with therapists and patients operating the device. It was found that patients performed poorly while using the platform compared to healthy subjects and that the assistance provided also improved performance amongst patients. Statistical analysis demonstrated that the speed of the patients was significantly enhanced with the robotic assistance. Further, neural networks are trained to classify between patients and healthy subjects and to forecast their movements using the data collected. 

**Abstract (ZH)**: 神经条件患者需要康复以恢复其运动、视觉和认知能力。为缓解治疗师短缺和减轻其工作负担，提出了一种涉及临床连线绘制测试的机器人康复平台。治疗师可以为每位患者创建定制化 trails，患者使用机器人设备进行跟踪。该平台可以跟踪患者的性能，并利用这些数据通过机器人向患者界面提供动态辅助。因此，所提出的平台不仅作为评估平台使用，还用于训练患者的康复。所开发的平台已在康复中心得到验证，由治疗师和患者操作设备。研究发现，与健康对照组相比，患者在使用平台时表现较差，但提供的辅助也改善了患者的性能。统计分析表明，机器人辅助显著提高了患者的运动速度。此外，使用收集到的数据训练神经网络以区分患者和健康个体，并预测其运动。 

---
# NANO-SLAM : Natural Gradient Gaussian Approximation for Vehicle SLAM 

**Title (ZH)**: NANO-SLAM：车辆SLAM的自然梯度高斯近似 

**Authors**: Tianyi Zhang, Wenhan Cao, Chang Liu, Feihong Zhang, Wei Wu, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.19195)  

**Abstract**: Accurate localization is a challenging task for autonomous vehicles, particularly in GPS-denied environments such as urban canyons and tunnels. In these scenarios, simultaneous localization and mapping (SLAM) offers a more robust alternative to GPS-based positioning, enabling vehicles to determine their position using onboard sensors and surrounding environment's landmarks. Among various vehicle SLAM approaches, Rao-Blackwellized particle filter (RBPF) stands out as one of the most widely adopted methods due to its efficient solution with logarithmic complexity relative to the map size. RBPF approximates the posterior distribution of the vehicle pose using a set of Monte Carlo particles through two main steps: sampling and importance weighting. The key to effective sampling lies in solving a distribution that closely approximates the posterior, known as the sampling distribution, to accelerate convergence. Existing methods typically derive this distribution via linearization, which introduces significant approximation errors due to the inherent nonlinearity of the system. To address this limitation, we propose a novel vehicle SLAM method called \textit{N}atural Gr\textit{a}dient Gaussia\textit{n} Appr\textit{o}ximation (NANO)-SLAM, which avoids linearization errors by modeling the sampling distribution as the solution to an optimization problem over Gaussian parameters and solving it using natural gradient descent. This approach improves the accuracy of the sampling distribution and consequently enhances localization performance. Experimental results on the long-distance Sydney Victoria Park vehicle SLAM dataset show that NANO-SLAM achieves over 50\% improvement in localization accuracy compared to the most widely used vehicle SLAM algorithms, with minimal additional computational cost. 

**Abstract (ZH)**: 自然梯度高斯近似自主车辆SLAM（NANO-SLAM） 

---
# VTire: A Bimodal Visuotactile Tire with High-Resolution Sensing Capability 

**Title (ZH)**: VTire：一种高分辨率感测能力的双模态视触轮胎 

**Authors**: Shoujie Li, Jianle Xu, Tong Wu, Yang Yang, Yanbo Chen, Xueqian Wang, Wenbo Ding, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19194)  

**Abstract**: Developing smart tires with high sensing capability is significant for improving the moving stability and environmental adaptability of wheeled robots and vehicles. However, due to the classical manufacturing design, it is always challenging for tires to infer external information precisely. To this end, this paper introduces a bimodal sensing tire, which can simultaneously capture tactile and visual data. By leveraging the emerging visuotactile techniques, the proposed smart tire can realize various functions, including terrain recognition, ground crack detection, load sensing, and tire damage detection. Besides, we optimize the material and structure of the tire to ensure its outstanding elasticity, toughness, hardness, and transparency. In terms of algorithms, a transformer-based multimodal classification algorithm, a load detection method based on finite element analysis, and a contact segmentation algorithm have been developed. Furthermore, we construct an intelligent mobile platform to validate the system's effectiveness and develop visual and tactile datasets in complex terrains. The experimental results show that our multimodal terrain sensing algorithm can achieve a classification accuracy of 99.2\%, a tire damage detection accuracy of 97\%, a 98\% success rate in object search, and the ability to withstand tire loading weights exceeding 35 kg. In addition, we open-source our algorithms, hardware, and datasets at this https URL. 

**Abstract (ZH)**: 开发高感知能力的智能轮胎对于提高轮式机器人和车辆的移动稳定性和环境适应性具有重要意义。然而，由于传统的制造设计，轮胎精确推断外部信息一直颇具挑战。为此，本文介绍了一种双模态感知轮胎，可以同时获取触觉和视觉数据。通过利用新兴的视触觉技术，所提出的智能轮胎可以实现地形识别、地面裂缝检测、负载感测和轮胎损伤检测等多种功能。此外，我们优化了轮胎的材料和结构，确保其具备出色的弹性、韧性、硬度和透明度。在算法方面，开发了一种基于变换器的多模态分类算法、基于有限元分析的负载检测方法和接触分割算法。我们还构建了一个智能移动平台来验证系统的有效性，并在复杂地形中开发了视觉和触觉数据集。实验结果表明，我们的多模态地形感知算法可以达到99.2%的分类准确率、97%的轮胎损伤检测准确率、98%的目标搜索成功率，并能够承受超过35公斤的轮胎负载。此外，我们在此httpsURL开放了我们的算法、硬件和数据集。 

---
# Making Physical Objects with Generative AI and Robotic Assembly: Considering Fabrication Constraints, Sustainability, Time, Functionality, and Accessibility 

**Title (ZH)**: 使用生成式人工智能和机器人装配制作物理对象：考虑加工约束、可持续性、时间、功能和可访问性 

**Authors**: Alexander Htet Kyaw, Se Hwan Jeon, Miana Smith, Neil Gershenfeld  

**Link**: [PDF](https://arxiv.org/pdf/2504.19131)  

**Abstract**: 3D generative AI enables rapid and accessible creation of 3D models from text or image inputs. However, translating these outputs into physical objects remains a challenge due to the constraints in the physical world. Recent studies have focused on improving the capabilities of 3D generative AI to produce fabricable outputs, with 3D printing as the main fabrication method. However, this workshop paper calls for a broader perspective by considering how fabrication methods align with the capabilities of 3D generative AI. As a case study, we present a novel system using discrete robotic assembly and 3D generative AI to make physical objects. Through this work, we identified five key aspects to consider in a physical making process based on the capabilities of 3D generative AI. 1) Fabrication Constraints: Current text-to-3D models can generate a wide range of 3D designs, requiring fabrication methods that can adapt to the variability of generative AI outputs. 2) Time: While generative AI can generate 3D models in seconds, fabricating physical objects can take hours or even days. Faster production could enable a closer iterative design loop between humans and AI in the making process. 3) Sustainability: Although text-to-3D models can generate thousands of models in the digital world, extending this capability to the real world would be resource-intensive, unsustainable and irresponsible. 4) Functionality: Unlike digital outputs from 3D generative AI models, the fabrication method plays a crucial role in the usability of physical objects. 5) Accessibility: While generative AI simplifies 3D model creation, the need for fabrication equipment can limit participation, making AI-assisted creation less inclusive. These five key aspects provide a framework for assessing how well a physical making process aligns with the capabilities of 3D generative AI and values in the world. 

**Abstract (ZH)**: 3D生成AI使从文本或图像输入快速便捷地创建3D模型成为可能，但由于物理世界的约束，将这些输出转化为物理对象仍具挑战性。最近的研究集中在提高3D生成AI的能力以生产可制造的输出，3D打印为主要的制造方法。然而，本文研讨会论文呼吁从更广泛的角度考虑制造方法与3D生成AI能力的契合性。作为案例研究，我们提出了一种使用离散机器人装配和3D生成AI生成物理对象的新系统。通过这项工作，我们基于3D生成AI的能力，确定了物理制作过程中的五个关键方面：1) 制造约束；2) 时间；3) 可持续性；4) 功能性；5) 可及性。这些五个关键方面为评估物理制作过程与3D生成AI能力及世界价值观的契合度提供了框架。 

---
# MISO: Multiresolution Submap Optimization for Efficient Globally Consistent Neural Implicit Reconstruction 

**Title (ZH)**: MISO：多分辨率子地图优化以实现高效的全局一致神经隐式重建 

**Authors**: Yulun Tian, Hanwen Cao, Sunghwan Kim, Nikolay Atanasov  

**Link**: [PDF](https://arxiv.org/pdf/2504.19104)  

**Abstract**: Neural implicit representations have had a significant impact on simultaneous localization and mapping (SLAM) by enabling robots to build continuous, differentiable, and high-fidelity 3D maps from sensor data. However, as the scale and complexity of the environment increase, neural SLAM approaches face renewed challenges in the back-end optimization process to keep up with runtime requirements and maintain global consistency. We introduce MISO, a hierarchical optimization approach that leverages multiresolution submaps to achieve efficient and scalable neural implicit reconstruction. For local SLAM within each submap, we develop a hierarchical optimization scheme with learned initialization that substantially reduces the time needed to optimize the implicit submap features. To correct estimation drift globally, we develop a hierarchical method to align and fuse the multiresolution submaps, leading to substantial acceleration by avoiding the need to decode the full scene geometry. MISO significantly improves computational efficiency and estimation accuracy of neural signed distance function (SDF) SLAM on large-scale real-world benchmarks. 

**Abstract (ZH)**: 神经隐式表示对同时定位与mapping（SLAM）产生了显著影响，使机器人能够从传感器数据中构建连续、可微分和高保真的3D地图。然而，随着环境规模和复杂性的增加，神经SLAM方法在后端优化过程中面临着新的挑战，以满足实时要求并保持全局一致性。我们提出了MISO，一种利用多分辨率子地图的分层优化方法，以实现高效的可扩展神经隐式重建。对于每个子地图内的局部SLAM，我们开发了一种包含学习初始化的分层优化方案，极大地减少了优化隐式子地图特征所需的时间。为了全局纠正估计漂移，我们开发了一种分层方法来对齐和融合多分辨率子地图，从而通过避免解码整个场景几何结构来实现显著加速。MISO在大型真实世界基准上的神经符号距离函数（SDF）SLAM计算效率和估计准确性得到了显著提升。 

---
# Geometric Gait Optimization for Kinodynamic Systems Using a Lie Group Integrator 

**Title (ZH)**: 几何步态优化在李群积分器下的动力学系统中应用 

**Authors**: Yanhao Yang, Ross L. Hatton  

**Link**: [PDF](https://arxiv.org/pdf/2504.19072)  

**Abstract**: This paper presents a gait optimization and motion planning framework for a class of locomoting systems with mixed kinematic and dynamic properties. Using Lagrangian reduction and differential geometry, we derive a general dynamic model that incorporates second-order dynamics and nonholonomic constraints, applicable to kinodynamic systems such as wheeled robots with nonholonomic constraints as well as swimming robots with nonisotropic fluid-added inertia and hydrodynamic drag. Building on Lie group integrators and group symmetries, we develop a variational gait optimization method for kinodynamic systems. By integrating multiple gaits and their transitions, we construct comprehensive motion plans that enable a wide range of motions for these systems. We evaluate our framework on three representative examples: roller racer, snakeboard, and swimmer. Simulation and hardware experiments demonstrate diverse motions, including acceleration, steady-state maintenance, gait transitions, and turning. The results highlight the effectiveness of the proposed method and its potential for generalization to other biological and robotic locomoting systems. 

**Abstract (ZH)**: 一种混合动力学与动力学属性的运动系统步态优化与运动规划框架 

---
# Efficient Control Allocation and 3D Trajectory Tracking of a Highly Manoeuvrable Under-actuated Bio-inspired AUV 

**Title (ZH)**: 高效控制分配与 Highly Manoeuvrable Under-actuated Bio-inspired AUV 的三维轨迹跟踪 

**Authors**: Walid Remmas, Christian Meurer, Yuya Hamamatsu, Ahmed Chemori, Maarja Kruusmaa  

**Link**: [PDF](https://arxiv.org/pdf/2504.19049)  

**Abstract**: Fin actuators can be used for for both thrust generation and vectoring. Therefore, fin-driven autonomous underwater vehicles (AUVs) can achieve high maneuverability with a smaller number of actuators, but their control is challenging. This study proposes an analytic control allocation method for underactuated Autonomous Underwater Vehicles (AUVs). By integrating an adaptive hybrid feedback controller, we enable an AUV with 4 actuators to move in 6 degrees of freedom (DOF) in simulation and up to 5-DOF in real-world experiments. The proposed method outperformed state-of-the-art control allocation techniques in 6-DOF trajectory tracking simulations, exhibiting centimeter-scale accuracy and higher energy and computational efficiency. Real-world pool experiments confirmed the method's robustness and efficacy in tracking complex 3D trajectories, with significant computational efficiency gains 0.007 (ms) vs. 22.28 (ms). Our method offers a balance between performance, energy efficiency, and computational efficiency, showcasing a potential avenue for more effective tracking of a large number of DOF for under-actuated underwater robots. 

**Abstract (ZH)**: 基于鳍驱动的欠驱动自治水下车辆的解析控制分配方法 

---
# An SE(3) Noise Model for Range-Azimuth-Elevation Sensors 

**Title (ZH)**: SE(3)噪声模型用于距离-方位-仰角传感器 

**Authors**: Thomas Hitchcox, James Richard Forbes  

**Link**: [PDF](https://arxiv.org/pdf/2504.19009)  

**Abstract**: Scan matching is a widely used technique in state estimation. Point-cloud alignment, one of the most popular methods for scan matching, is a weighted least-squares problem in which the weights are determined from the inverse covariance of the measured points. An inaccurate representation of the covariance will affect the weighting of the least-squares problem. For example, if ellipsoidal covariance bounds are used to approximate the curved, "banana-shaped" noise characteristics of many scanning sensors, the weighting in the least-squares problem may be overconfident. Additionally, sensor-to-vehicle extrinsic uncertainty and odometry uncertainty during submap formation are two sources of uncertainty that are often overlooked in scan matching applications, also likely contributing to overconfidence on the scan matching estimate. This paper attempts to address these issues by developing a model for range-azimuth-elevation sensors on matrix Lie groups. The model allows for the seamless incorporation of extrinsic and odometry uncertainty. Illustrative results are shown both for a simulated example and for a real point-cloud submap collected with an underwater laser scanner. 

**Abstract (ZH)**: 扫描匹配是一种广泛应用于状态估计的技术。点云对齐，作为扫描匹配中最流行的方法之一，是一个加权最小二乘问题，其中权重由测量点的逆协方差决定。协方差的不准确表示将影响最小二乘问题中的权重。例如，如果使用椭球协方差边界来近似许多扫描传感器的弯曲的“香蕉形”噪声特性，最小二乘问题中的权重可能会过于自信。另外，传感器到车辆的外参不确定性以及子地图构建期间的里程计不确定性也是扫描匹配应用中常被忽视的两种不确定性来源，也可能导致对扫描匹配估计过于自信。本文通过在矩阵李群上开发范围-方位-仰角传感器模型试图解决这些问题，该模型允许无缝地整合外参和里程计不确定性。文中展示了模拟示例和使用水下激光扫描器收集的实际点云子地图的示例结果。 

---
# A biconvex method for minimum-time motion planning through sequences of convex sets 

**Title (ZH)**: 双凸方法在凸集序列通过下的最小时间运动规划 

**Authors**: Tobia Marcucci, Mathew Halm, Will Yang, Dongchan Lee, Andrew D. Marchese  

**Link**: [PDF](https://arxiv.org/pdf/2504.18978)  

**Abstract**: We consider the problem of designing a smooth trajectory that traverses a sequence of convex sets in minimum time, while satisfying given velocity and acceleration constraints. This problem is naturally formulated as a nonconvex program. To solve it, we propose a biconvex method that quickly produces an initial trajectory and iteratively refines it by solving two convex subproblems in alternation. This method is guaranteed to converge, returns a feasible trajectory even if stopped early, and does not require the selection of any line-search or trust-region parameter. Exhaustive experiments show that our method finds high-quality trajectories in a fraction of the time of state-of-the-art solvers for nonconvex optimization. In addition, it achieves runtimes comparable to industry-standard waypoint-based motion planners, while consistently designing lower-duration trajectories than existing optimization-based planners. 

**Abstract (ZH)**: 我们考虑设计一条平滑轨迹，使其在满足给定的速度和加速度约束条件下，以最小时间穿越一系列凸集。该问题自然地形式化为一个非凸规划问题。为了解决这个问题，我们提出了一种双凸方法，该方法能够快速生成初始轨迹，并通过交替求解两个凸子问题来逐步优化它。该方法能得到收敛保证，即使提前停止也能返回可行轨迹，且无需选择任何线搜索或信任区域参数。详尽的实验表明，我们的方法在非凸优化中最先进的求解器所需时间的一小部分内就能找到高质量的轨迹。此外，它在运行时间上与基于航点的工业标准轨迹规划器相当，但始终能够设计出比现有基于优化的轨迹规划器更短时间的轨迹。 

---
# Generative AI in Embodied Systems: System-Level Analysis of Performance, Efficiency and Scalability 

**Title (ZH)**: 生成式AI在具身系统中的应用：性能、效率和扩展性系统的分析 

**Authors**: Zishen Wan, Jiayi Qian, Yuhang Du, Jason Jabbour, Yilun Du, Yang Katie Zhao, Arijit Raychowdhury, Tushar Krishna, Vijay Janapa Reddi  

**Link**: [PDF](https://arxiv.org/pdf/2504.18945)  

**Abstract**: Embodied systems, where generative autonomous agents engage with the physical world through integrated perception, cognition, action, and advanced reasoning powered by large language models (LLMs), hold immense potential for addressing complex, long-horizon, multi-objective tasks in real-world environments. However, deploying these systems remains challenging due to prolonged runtime latency, limited scalability, and heightened sensitivity, leading to significant system inefficiencies.
In this paper, we aim to understand the workload characteristics of embodied agent systems and explore optimization solutions. We systematically categorize these systems into four paradigms and conduct benchmarking studies to evaluate their task performance and system efficiency across various modules, agent scales, and embodied tasks. Our benchmarking studies uncover critical challenges, such as prolonged planning and communication latency, redundant agent interactions, complex low-level control mechanisms, memory inconsistencies, exploding prompt lengths, sensitivity to self-correction and execution, sharp declines in success rates, and reduced collaboration efficiency as agent numbers increase. Leveraging these profiling insights, we suggest system optimization strategies to improve the performance, efficiency, and scalability of embodied agents across different paradigms. This paper presents the first system-level analysis of embodied AI agents, and explores opportunities for advancing future embodied system design. 

**Abstract (ZH)**: 具身系统中的生成自主代理通过集成感知、认知、行动和由大规模语言模型支持的高级推理与物理世界互动，具有应对真实环境中的复杂、长周期、多目标任务的巨大潜力。然而，由于持续的运行时延、有限的可扩展性和增强的敏感性，部署这些系统仍然具有挑战性，导致系统效率低下。
本文旨在理解具身代理系统的工作负载特征并探索优化方案。我们系统地将这些系统划分为四种范式，并通过基准测试研究评估其在不同模块、代理规模和具身任务中的任务性能和系统效率。我们的基准测试研究揭示了关键挑战，如持续的规划和通信延迟、冗余的代理交互、复杂的低级控制机制、内存不一致性、提示长度爆炸性增长、对自我校正和执行的敏感性、成功率的急剧下降以及随着代理数量增加的合作效率降低。基于这些分析结果，我们提出了系统的优化策略，以提高不同范式下具身代理的性能、效率和可扩展性。本文首次对具身AI代理进行了系统级分析，并探讨了推进未来具身系统设计的机会。 

---
# Demonstrating DVS: Dynamic Virtual-Real Simulation Platform for Mobile Robotic Tasks 

**Title (ZH)**: 基于移动机器人任务的动态虚拟-现实仿真平台Demonstrating DVS: Dynamic Virtual-Real Simulation Platform for Mobile Robotic Tasks 

**Authors**: Zijie Zheng, Zeshun Li, Yunpeng Wang, Qinghongbing Xie, Long Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2504.18944)  

**Abstract**: With the development of embodied artificial intelligence, robotic research has increasingly focused on complex tasks. Existing simulation platforms, however, are often limited to idealized environments, simple task scenarios and lack data interoperability. This restricts task decomposition and multi-task learning. Additionally, current simulation platforms face challenges in dynamic pedestrian modeling, scene editability, and synchronization between virtual and real assets. These limitations hinder real world robot deployment and feedback. To address these challenges, we propose DVS (Dynamic Virtual-Real Simulation Platform), a platform for dynamic virtual-real synchronization in mobile robotic tasks. DVS integrates a random pedestrian behavior modeling plugin and large-scale, customizable indoor scenes for generating annotated training datasets. It features an optical motion capture system, synchronizing object poses and coordinates between virtual and real world to support dynamic task benchmarking. Experimental validation shows that DVS supports tasks such as pedestrian trajectory prediction, robot path planning, and robotic arm grasping, with potential for both simulation and real world deployment. In this way, DVS represents more than just a versatile robotic platform; it paves the way for research in human intervention in robot execution tasks and real-time feedback algorithms in virtual-real fusion environments. More information about the simulation platform is available on this https URL. 

**Abstract (ZH)**: 随着体态人工智能的发展，机器人研究越来越多地关注复杂任务。然而，现有的仿真平台往往局限于理想化的环境、简单的任务场景以及数据互操作性不足的问题，这限制了任务分解和多任务学习。此外，当前的仿真平台在动态行人建模、场景可编辑性和虚拟与现实资产之间的同步方面也面临挑战。这些限制妨碍了现实世界中机器人部署和反馈。为了解决这些挑战，我们提出DVS（动态虚拟-现实仿真平台），一个用于移动机器人任务动态虚拟-现实同步的平台。DVS集成了随机行人行为建模插件和可定制的大规模室内场景，用于生成标注的训练数据集。它配备了光学动作捕捉系统，实现虚拟和现实世界中物体姿态和坐标的同步，以支持动态任务基准测试。实验验证显示，DVS支持行人轨迹预测、机器人路径规划和机械臂抓取等任务，具有在仿真和现实世界中部署的潜力。通过这种方式，DVS不仅代表了一个多功能的机器人平台，还为在虚拟-现实融合环境中进行人类干预机器人执行任务的研究以及实时反馈算法的研究铺平了道路。了解更多关于仿真平台的信息，请访问此 [链接]。 

---
# Advanced Longitudinal Control and Collision Avoidance for High-Risk Edge Cases in Autonomous Driving 

**Title (ZH)**: 高级纵向控制与碰撞避免技术在自动驾驶高风险边缘情况中的应用 

**Authors**: Dianwei Chen, Yaobang Gong, Xianfeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18931)  

**Abstract**: Advanced Driver Assistance Systems (ADAS) and Advanced Driving Systems (ADS) are key to improving road safety, yet most existing implementations focus primarily on the vehicle ahead, neglecting the behavior of following vehicles. This shortfall often leads to chain reaction collisions in high speed, densely spaced traffic particularly when a middle vehicle suddenly brakes and trailing vehicles cannot respond in time. To address this critical gap, we propose a novel longitudinal control and collision avoidance algorithm that integrates adaptive cruising with emergency braking. Leveraging deep reinforcement learning, our method simultaneously accounts for both leading and following vehicles. Through a data preprocessing framework that calibrates real-world sensor data, we enhance the robustness and reliability of the training process, ensuring the learned policy can handle diverse driving conditions. In simulated high risk scenarios (e.g., emergency braking in dense traffic), the algorithm effectively prevents potential pile up collisions, even in situations involving heavy duty vehicles. Furthermore, in typical highway scenarios where three vehicles decelerate, the proposed DRL approach achieves a 99% success rate far surpassing the standard Federal Highway Administration speed concepts guide, which reaches only 36.77% success under the same conditions. 

**Abstract (ZH)**: 先进的驾驶辅助系统（ADAS）和高级驾驶系统（ADS）对于提高道路交通安全至关重要，但现有大多数实现主要关注前方车辆，忽视了跟随车辆的行为。这一不足往往导致在高密度、高速交通中，当中间车辆突然制动而后续车辆无法及时响应时发生连锁碰撞。为弥补这一关键缺口，我们提出了一种新颖的纵向控制与碰撞避免算法，将自适应巡航与紧急制动相结合。利用深度强化学习，我们的方法同时考虑领车和随车的行为。通过一个数据预处理框架来校准现实世界传感器数据，我们增强了训练过程的鲁棒性和可靠性，确保学习到的策略能够应对多种驾驶条件。在模拟高风险场景（如密集交通中的紧急制动）中，该算法有效防止了潜在的堆积碰撞，即使涉及重型车辆。此外，在典型高速公路上三辆车减速的场景中，所提出的DRL方法的成功率达到99%，远超美国联邦公路管理局速度概念指南在同一条件下的36.77%的成功率。 

---
# RoboVerse: Towards a Unified Platform, Dataset and Benchmark for Scalable and Generalizable Robot Learning 

**Title (ZH)**: RoboVerse: 向统一平台、数据集和基准测试方向实现可扩展和泛化的机器人学习 

**Authors**: Haoran Geng, Feishi Wang, Songlin Wei, Yuyang Li, Bangjun Wang, Boshi An, Charlie Tianyue Cheng, Haozhe Lou, Peihao Li, Yen-Jen Wang, Yutong Liang, Dylan Goetting, Chaoyi Xu, Haozhe Chen, Yuxi Qian, Yiran Geng, Jiageng Mao, Weikang Wan, Mingtong Zhang, Jiangran Lyu, Siheng Zhao, Jiazhao Zhang, Jialiang Zhang, Chengyang Zhao, Haoran Lu, Yufei Ding, Ran Gong, Yuran Wang, Yuxuan Kuang, Ruihai Wu, Baoxiong Jia, Carlo Sferrazza, Hao Dong, Siyuan Huang, Yue Wang, Jitendra Malik, Pieter Abbeel  

**Link**: [PDF](https://arxiv.org/pdf/2504.18904)  

**Abstract**: Data scaling and standardized evaluation benchmarks have driven significant advances in natural language processing and computer vision. However, robotics faces unique challenges in scaling data and establishing evaluation protocols. Collecting real-world data is resource-intensive and inefficient, while benchmarking in real-world scenarios remains highly complex. Synthetic data and simulation offer promising alternatives, yet existing efforts often fall short in data quality, diversity, and benchmark standardization. To address these challenges, we introduce RoboVerse, a comprehensive framework comprising a simulation platform, a synthetic dataset, and unified benchmarks. Our simulation platform supports multiple simulators and robotic embodiments, enabling seamless transitions between different environments. The synthetic dataset, featuring high-fidelity physics and photorealistic rendering, is constructed through multiple approaches. Additionally, we propose unified benchmarks for imitation learning and reinforcement learning, enabling evaluation across different levels of generalization. At the core of the simulation platform is MetaSim, an infrastructure that abstracts diverse simulation environments into a universal interface. It restructures existing simulation environments into a simulator-agnostic configuration system, as well as an API aligning different simulator functionalities, such as launching simulation environments, loading assets with initial states, stepping the physics engine, etc. This abstraction ensures interoperability and extensibility. Comprehensive experiments demonstrate that RoboVerse enhances the performance of imitation learning, reinforcement learning, world model learning, and sim-to-real transfer. These results validate the reliability of our dataset and benchmarks, establishing RoboVerse as a robust solution for advancing robot learning. 

**Abstract (ZH)**: 基于数据缩放和标准化评估基准，自然语言处理和计算机视觉取得了显著进展。然而，机器人技术在数据缩放和建立评估标准方面面临独特挑战。收集真实世界数据资源密集且效率低下，而在真实世界场景中的基准测试依然非常复杂。合成数据和模拟提供了有希望的替代方案，但现有努力往往在数据质量和多样性以及基准测试标准化方面不尽如人意。为应对这些挑战，我们引入了RoboVerse，这是一个包含模拟平台、合成数据集和统一基准的全面框架。我们的模拟平台支持多种模拟器和机器人实体，使得在不同环境之间实现无缝过渡成为可能。合成数据集包含高保真物理和光realistic渲染，通过多种方法构建而成。此外，我们还提出了统一的模仿学习和强化学习基准，使得在不同泛化水平上进行评估成为可能。模拟平台的核心是MetaSim基础设施，它将多种多样的模拟环境抽象为一个通用界面。该基础设施重构现有模拟环境为一种与模拟器无关的配置系统，并提供一种API来对齐不同模拟器的功能，如启动模拟环境、加载初始状态的资源、推进物理引擎等。这种抽象确保了互操作性和可扩展性。综合实验表明，RoboVerse提升了模仿学习、强化学习、世界模型学习和模拟到现实应用的性能。这些结果验证了我们数据集和基准的可靠性，将RoboVerse确立为推动机器人学习发展的稳健解决方案。 

---
# Hierarchical Temporal Logic Task and Motion Planning for Multi-Robot Systems 

**Title (ZH)**: 多机器人系统分层时序逻辑任务与运动规划 

**Authors**: Zhongqi Wei, Xusheng Luo, Changliu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18899)  

**Abstract**: Task and motion planning (TAMP) for multi-robot systems, which integrates discrete task planning with continuous motion planning, remains a challenging problem in robotics. Existing TAMP approaches often struggle to scale effectively for multi-robot systems with complex specifications, leading to infeasible solutions and prolonged computation times. This work addresses the TAMP problem in multi-robot settings where tasks are specified using expressive hierarchical temporal logic and task assignments are not pre-determined. Our approach leverages the efficiency of hierarchical temporal logic specifications for task-level planning and the optimization-based graph of convex sets method for motion-level planning, integrating them within a product graph framework. At the task level, we convert hierarchical temporal logic specifications into a single graph, embedding task allocation within its edges. At the motion level, we represent the feasible motions of multiple robots through convex sets in the configuration space, guided by a sampling-based motion planner. This formulation allows us to define the TAMP problem as a shortest path search within the product graph, where efficient convex optimization techniques can be applied. We prove that our approach is both sound and complete under mild assumptions. Additionally, we extend our framework to cooperative pick-and-place tasks involving object handovers between robots. We evaluate our method across various high-dimensional multi-robot scenarios, including simulated and real-world environments with quadrupeds, robotic arms, and automated conveyor systems. Our results show that our approach outperforms existing methods in execution time and solution optimality while effectively scaling with task complexity. 

**Abstract (ZH)**: 多机器人系统中基于表达性层次时序逻辑的任务与运动规划（TAMP）及其应用 

---
# Diffeomorphic Obstacle Avoidance for Contractive Dynamical Systems via Implicit Representations 

**Title (ZH)**: 基于隐式表示的收敛动力系统可变形障碍避让 

**Authors**: Ken-Joel Simmoteit, Philipp Schillinger, Leonel Rozo  

**Link**: [PDF](https://arxiv.org/pdf/2504.18860)  

**Abstract**: Ensuring safety and robustness of robot skills is becoming crucial as robots are required to perform increasingly complex and dynamic tasks. The former is essential when performing tasks in cluttered environments, while the latter is relevant to overcome unseen task situations. This paper addresses the challenge of ensuring both safety and robustness in dynamic robot skills learned from demonstrations. Specifically, we build on neural contractive dynamical systems to provide robust extrapolation of the learned skills, while designing a full-body obstacle avoidance strategy that preserves contraction stability via diffeomorphic transforms. This is particularly crucial in complex environments where implicit scene representations, such as Signed Distance Fields (SDFs), are necessary. To this end, our framework called Signed Distance Field Diffeomorphic Transform, leverages SDFs and flow-based diffeomorphisms to achieve contraction-preserving obstacle avoidance. We thoroughly evaluate our framework on synthetic datasets and several real-world robotic tasks in a kitchen environment. Our results show that our approach locally adapts the learned contractive vector field while staying close to the learned dynamics and without introducing highly-curved motion paths, thus outperforming several state-of-the-art methods. 

**Abstract (ZH)**: 确保机器人技能的安全性和鲁棒性正变得越来越重要，尤其是在机器人需要执行日益复杂和动态的任务时。前者在进行 cluttered 环境中的任务时至关重要，而后者则与克服未知任务情境相关。本文探讨了在示例中学习的动态机器人技能中确保同时具备安全性和鲁棒性的挑战。具体来说，我们基于神经收敛动力学系统提供学习技能的稳健外推，并设计了一种全身障碍物规避策略，通过差分同胚变换保持收缩稳定性。这对于复杂环境中尤为重要，尤其是当需要使用隐式场景表示（如符号距离场 SDF）时。为此，我们提出了一种名为符号距离场差分同胚变换的框架，利用 SDF 和基于流的差分同胚来实现保持收缩的障碍物规避。我们通过合成数据集和厨房环境中的多个真实世界机器人任务全面评估了该框架，结果显示，我们的方法在局部适应所学的收敛向量场的同时保持接近所学的动力学，并避免了引入高度弯曲的运动路径，从而优于几种最先进的方法。 

---
# Imitation Learning for Autonomous Driving: Insights from Real-World Testing 

**Title (ZH)**: 自动驾驶领域的模仿学习：来自实际测试的见解 

**Authors**: Hidayet Ersin Dursun, Yusuf Güven, Tufan Kumbasar  

**Link**: [PDF](https://arxiv.org/pdf/2504.18847)  

**Abstract**: This work focuses on the design of a deep learning-based autonomous driving system deployed and tested on the real-world MIT Racecar to assess its effectiveness in driving scenarios. The Deep Neural Network (DNN) translates raw image inputs into real-time steering commands in an end-to-end learning fashion, following the imitation learning framework. The key design challenge is to ensure that DNN predictions are accurate and fast enough, at a high sampling frequency, and result in smooth vehicle operation under different operating conditions. In this study, we design and compare various DNNs, to identify the most effective approach for real-time autonomous driving. In designing the DNNs, we adopted an incremental design approach that involved enhancing the model capacity and dataset to address the challenges of real-world driving scenarios. We designed a PD system, CNN, CNN-LSTM, and CNN-NODE, and evaluated their performance on the real-world MIT Racecar. While the PD system handled basic lane following, it struggled with sharp turns and lighting variations. The CNN improved steering but lacked temporal awareness, which the CNN-LSTM addressed as it resulted in smooth driving performance. The CNN-NODE performed similarly to the CNN-LSTM in handling driving dynamics, yet with slightly better driving performance. The findings of this research highlight the importance of iterative design processes in developing robust DNNs for autonomous driving applications. The experimental video is available at this https URL. 

**Abstract (ZH)**: 基于深度学习的自主驾驶系统设计与在MIT Racecar上的实际测试及其在驾驶场景中的有效性评估 

---
# A Microgravity Simulation Experimental Platform For Small Space Robots In Orbit 

**Title (ZH)**: 轨道小型太空机器人微重力模拟实验平台 

**Authors**: Hang Luo, Nanlin Zhou, Haoxiang Zhang, Kai Han, Ning Zhao, Zhiyuan Yang, Jian Qi, Sikai Zhao, Jie Zhao, Yanhe Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18842)  

**Abstract**: This study describes the development and validation of a novel microgravity experimental platform that is mainly applied to small robots such as modular self-reconfigurable robots. This platform mainly consists of an air supply system, a microporous platform and glass. By supplying air to the microporous platform to form an air film, the influence of the weight of the air foot and the ventilation hose of traditional air-float platforms on microgravity experiments is solved. The contribution of this work is to provide a platform with less external interference for microgravity simulation experiments on small robots. 

**Abstract (ZH)**: 本研究描述了一种新型微重力实验平台的发展与验证，该平台主要用于模块化自重构机器人等小型机器人。该平台主要由气动供应系统、微孔平台和玻璃构成。通过向微孔平台供气形成气膜，解决了传统气浮平台的气脚重量和通风管对微重力实验的影响。本研究的贡献在于为小型机器人微重力模拟实验提供了一个外部干扰更少的平台。 

---
# Swarming in the Wild: A Distributed Communication-less Lloyd-based Algorithm dealing with Uncertainties 

**Title (ZH)**: 野外集群：一种处理不确定性无需通信的分布式Lloyd算法 

**Authors**: Manuel Boldrer, Vit Kratky, Viktor Walter, Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2504.18840)  

**Abstract**: In this work, we present a distributed algorithm for swarming in complex environments that operates with no communication, no a priori information about the environment, and using only onboard sensing and computation capabilities. We provide sufficient conditions to guarantee that each robot reaches its goal region in a finite time, avoiding collisions with obstacles and other robots without exceeding a desired maximum distance from a predefined set of neighbors (flocking constraint). In addition, we show how the proposed algorithm can deal with tracking errors and onboard sensing errors without violating safety and proximity constraints, still providing the conditions for having convergence towards the goal region. To validate the approach, we provide experiments in the field. We tested our algorithm in GNSS-denied environments i.e., a dense forest, where fully autonomous aerial robots swarmed safely to the desired destinations, by relying only on onboard sensors, i.e., without a communication network. This work marks the initial deployment of a fully distributed system where there is no communication between the robots, nor reliance on any global localization system, which at the same time it ensures safety and convergence towards the goal within such complex environments. 

**Abstract (ZH)**: 一种在复杂环境中无需通信的分布式群集算法及其应用 

---
# Aerial Robots Persistent Monitoring and Target Detection: Deployment and Assessment in the Field 

**Title (ZH)**: 空中机器人持续监测与目标检测：现场部署与评估 

**Authors**: Manuel Boldrer, Vit Kratky, Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2504.18832)  

**Abstract**: In this manuscript, we present a distributed algorithm for multi-robot persistent monitoring and target detection. In particular, we propose a novel solution that effectively integrates the Time-inverted Kuramoto model, three-dimensional Lissajous curves, and Model Predictive Control. We focus on the implementation of this algorithm on aerial robots, addressing the practical challenges involved in deploying our approach under real-world conditions. Our method ensures an effective and robust solution that maintains operational efficiency even in the presence of what we define as type I and type II failures. Type I failures refer to short-time disruptions, such as tracking errors and communication delays, while type II failures account for long-time disruptions, including malicious attacks, severe communication failures, and battery depletion. Our approach guarantees persistent monitoring and target detection despite these challenges. Furthermore, we validate our method with extensive field experiments involving up to eleven aerial robots, demonstrating the effectiveness, resilience, and scalability of our solution. 

**Abstract (ZH)**: 本论文提出了一种分布式多_robot持久监测与目标检测算法。特别地，我们提出了一种新颖的解决方案，有效地整合了时间倒置库拉莫模型、三维利萨茹曲线和模型预测控制。我们重点讨论了在实际条件下部署该方法所面临的实际挑战，并确保我们的方法即使在我们定义的类型I和类型II故障存在的情况下也能提供有效且 robust 的解决方案。类型I故障指的是短时间中断，如跟踪误差和通信延迟，而类型II故障涉及长时间中断，包括恶意攻击、严重通信故障和电池耗尽。我们的方法能够在这些挑战下保证持久监测和目标检测。此外，我们通过涉及多达十一架飞行机器人的大量野外试验验证了该方法，展示了我们解决方案的有效性、韧性和可扩展性。 

---
# Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy 

**Title (ZH)**: dexonomy: 合成抓持分类学中的所有灵巧握持类型 

**Authors**: Jiayi Chen, Yubin Ke, Lin Peng, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18829)  

**Abstract**: Generalizable dexterous grasping with suitable grasp types is a fundamental skill for intelligent robots. Developing such skills requires a large-scale and high-quality dataset that covers numerous grasp types (i.e., at least those categorized by the GRASP taxonomy), but collecting such data is extremely challenging. Existing automatic grasp synthesis methods are often limited to specific grasp types or object categories, hindering scalability. This work proposes an efficient pipeline capable of synthesizing contact-rich, penetration-free, and physically plausible grasps for any grasp type, object, and articulated hand. Starting from a single human-annotated template for each hand and grasp type, our pipeline tackles the complicated synthesis problem with two stages: optimize the object to fit the hand template first, and then locally refine the hand to fit the object in simulation. To validate the synthesized grasps, we introduce a contact-aware control strategy that allows the hand to apply the appropriate force at each contact point to the object. Those validated grasps can also be used as new grasp templates to facilitate future synthesis. Experiments show that our method significantly outperforms previous type-unaware grasp synthesis baselines in simulation. Using our algorithm, we construct a dataset containing 10.7k objects and 9.5M grasps, covering 31 grasp types in the GRASP taxonomy. Finally, we train a type-conditional generative model that successfully performs the desired grasp type from single-view object point clouds, achieving an 82.3% success rate in real-world experiments. Project page: this https URL. 

**Abstract (ZH)**: 通用可转移的 Dexterous 抓取技能对于智能机器人来说是一项基本技能。开发此类技能需要大规模且高质量的数据集，涵盖多种抓取类型（即按照 GRASP 分类法分类的类型），但收集此类数据极具挑战性。现有的自动抓取合成方法通常局限于特定的抓取类型或对象类别，阻碍了其扩展性。本工作提出了一种高效的工作流程，能够为任何抓取类型、对象和 articulated 手合成接触丰富、无穿刺且物理上可实现的抓取。从每个手和抓取类型的单一个人注释模板开始，本工作流程通过两个阶段解决复杂的合成问题：首先优化对象以匹配手模板，然后在模拟中局部优化手以适应对象。为了验证合成的抓取，我们引入了一种接触感知的控制策略，允许手在每个接触点上施加适当的力到对象上。这些验证过的抓取也可以作为新的抓取模板，以促进未来的合成。实验表明，与之前的无抓取类型感知的抓取合成基准相比，本方法在模拟中表现显著优越。使用我们的算法，我们构建了一个包含 10,700 个对象和 9.5 百万抓取的数据集，涵盖了 GRASP 分类法中的 31 种抓取类型。最后，我们训练了一个基于类型的生成模型，可以从单视图对象点云中成功地执行所需的抓取类型，在真实世界实验中成功率达到 82.3%。项目页面：this https URL。 

---
# STDArm: Transferring Visuomotor Policies From Static Data Training to Dynamic Robot Manipulation 

**Title (ZH)**: STDArm: 从静态数据训练向动态机器人操作转移视觉运动策略 

**Authors**: Yifan Duan, Heng Li, Yilong Wu, Wenhao Yu, Xinran Zhang, Yedong Shen, Jianmin Ji, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18792)  

**Abstract**: Recent advances in mobile robotic platforms like quadruped robots and drones have spurred a demand for deploying visuomotor policies in increasingly dynamic environments. However, the collection of high-quality training data, the impact of platform motion and processing delays, and limited onboard computing resources pose significant barriers to existing solutions. In this work, we present STDArm, a system that directly transfers policies trained under static conditions to dynamic platforms without extensive modifications.
The core of STDArm is a real-time action correction framework consisting of: (1) an action manager to boost control frequency and maintain temporal consistency, (2) a stabilizer with a lightweight prediction network to compensate for motion disturbances, and (3) an online latency estimation module for calibrating system parameters. In this way, STDArm achieves centimeter-level precision in mobile manipulation tasks.
We conduct comprehensive evaluations of the proposed STDArm on two types of robotic arms, four types of mobile platforms, and three tasks. Experimental results indicate that the STDArm enables real-time compensation for platform motion disturbances while preserving the original policy's manipulation capabilities, achieving centimeter-level operational precision during robot motion. 

**Abstract (ZH)**: 近期，四足机器人和无人机等移动机器人平台的进步推动了在动态环境中部署视听运动策略的需求。然而，高质量训练数据的采集、平台运动和处理延迟的影响以及有限的车载计算资源构成了现有解决方案的重要障碍。本文介绍了一种名为STDArm的系统，该系统能够在无需进行大量修改的情况下，将静态条件下训练的策略直接转移到动态平台上。STDArm的核心是一个实时动作修正框架，包括：（1）动作管理器以提高控制频率并保持时间一致性，（2）用于补偿运动干扰的轻量级预测稳定器，以及（3）在线延时估计模块以校准系统参数。通过这种方式，STDArm在移动操作任务中实现了厘米级的精度。我们在两种类型的机器人手臂、四种类型的移动平台和三种任务上进行了全面评估。实验结果表明，STDArm能够在保持原始策略操作能力的同时，实时补偿平台运动干扰，实现机器人运动中的厘米级操作精度。 

---
# Coherence-based Approximate Derivatives via Web of Affine Spaces Optimization 

**Title (ZH)**: 基于一致性的近似导数通过仿射空间网络优化 

**Authors**: Daniel Rakita, Chen Liang, Qian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18790)  

**Abstract**: Computing derivatives is a crucial subroutine in computer science and related fields as it provides a local characterization of a function's steepest directions of ascent or descent. In this work, we recognize that derivatives are often not computed in isolation; conversely, it is quite common to compute a \textit{sequence} of derivatives, each one somewhat related to the last. Thus, we propose accelerating derivative computation by reusing information from previous, related calculations-a general strategy known as \textit{coherence}. We introduce the first instantiation of this strategy through a novel approach called the Web of Affine Spaces (WASP) Optimization. This approach provides an accurate approximation of a function's derivative object (i.e. gradient, Jacobian matrix, etc.) at the current input within a sequence. Each derivative within the sequence only requires a small number of forward passes through the function (typically two), regardless of the number of function inputs and outputs. We demonstrate the efficacy of our approach through several numerical experiments, comparing it with alternative derivative computation methods on benchmark functions. We show that our method significantly improves the performance of derivative computation on small to medium-sized functions, i.e., functions with approximately fewer than 500 combined inputs and outputs. Furthermore, we show that this method can be effectively applied in a robotics optimization context. We conclude with a discussion of the limitations and implications of our work. Open-source code, visual explanations, and videos are located at the paper website: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 计算导数是计算机科学及相关领域中的一个关键子程序，它提供了函数最陡上升或下降方向的局部表征。在这项工作中，我们认识到导数通常不是独立计算的；相反，计算导数序列的情况相当常见，每一项都与前一项有一定的关联。因此，我们提出通过重用之前相关计算中的信息来加速导数计算——这一一般策略称为“一致性”。我们通过一种新型方法——仿射空间网（WASP）优化——实现这一策略的第一种具体实例。这种方法在序列中的当前输入处提供了函数导数对象（即梯度、雅可比矩阵等）的精确近似。序列中的每项导数只需少量（通常是两次）函数前向传递，而不受函数输入和输出数量的影响。我们通过多项数值实验来展示该方法的有效性，将它与其他导数计算方法在基准函数上进行比较。我们证明，对于小型到中型函数（即大约合并输入和输出小于500的函数），我们的方法显著提高了导数计算的性能。此外，我们展示了该方法在机器人优化上下文中的应用效果。最后，我们讨论了该工作的局限性和意义。开源代码、可视化解释和视频可在论文网站：\[this https URL\]找到。 

---
# Design, Contact Modeling, and Collision-inclusive Planning of a Dual-stiffness Aerial RoboT (DART) 

**Title (ZH)**: 设计、接触建模及碰撞包容规划的双刚度空中机器人(DART) 

**Authors**: Yogesh Kumar, Karishma Patnaik, Wenlong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18780)  

**Abstract**: Collision-resilient quadrotors have gained significant attention given their potential for operating in cluttered environments and leveraging impacts to perform agile maneuvers. However, existing designs are typically single-mode: either safeguarded by propeller guards that prevent deformation or deformable but lacking rigidity, which is crucial for stable flight in open environments. This paper introduces DART, a Dual-stiffness Aerial RoboT, that adapts its post-collision response
by either engaging a locking mechanism for a rigid mode or disengaging it for a flexible mode, respectively. Comprehensive characterization tests highlight the significant difference in post collision responses between its rigid and flexible modes, with the rigid mode offering seven times higher stiffness compared to the flexible mode. To understand and harness the collision dynamics, we propose a novel collision response prediction model based on the linear complementarity system theory. We demonstrate the accuracy of predicting collision forces for both the rigid and flexible modes of DART. Experimental results confirm the accuracy of the model and underscore its potential to advance collision-inclusive trajectory planning in aerial robotics. 

**Abstract (ZH)**: 双刚度空中机器人 DART：基于碰撞后响应适应的新型设计 

---
# Vysics: Object Reconstruction Under Occlusion by Fusing Vision and Contact-Rich Physics 

**Title (ZH)**: Vysics: 在接触丰富物理融合下的物体遮挡重建 

**Authors**: Bibit Bianchini, Minghan Zhu, Mengti Sun, Bowen Jiang, Camillo J. Taylor, Michael Posa  

**Link**: [PDF](https://arxiv.org/pdf/2504.18719)  

**Abstract**: We introduce Vysics, a vision-and-physics framework for a robot to build an expressive geometry and dynamics model of a single rigid body, using a seconds-long RGBD video and the robot's proprioception. While the computer vision community has built powerful visual 3D perception algorithms, cluttered environments with heavy occlusions can limit the visibility of objects of interest. However, observed motion of partially occluded objects can imply physical interactions took place, such as contact with a robot or the environment. These inferred contacts can supplement the visible geometry with "physible geometry," which best explains the observed object motion through physics. Vysics uses a vision-based tracking and reconstruction method, BundleSDF, to estimate the trajectory and the visible geometry from an RGBD video, and an odometry-based model learning method, Physics Learning Library (PLL), to infer the "physible" geometry from the trajectory through implicit contact dynamics optimization. The visible and "physible" geometries jointly factor into optimizing a signed distance function (SDF) to represent the object shape. Vysics does not require pretraining, nor tactile or force sensors. Compared with vision-only methods, Vysics yields object models with higher geometric accuracy and better dynamics prediction in experiments where the object interacts with the robot and the environment under heavy occlusion. Project page: this https URL 

**Abstract (ZH)**: 视觉与物理框架Vysics：基于秒级RGBD视频和机器人本体感受构建单个刚体的表达几何与动力学模型 

---
# Certifiably-Correct Mapping for Safe Navigation Despite Odometry Drift 

**Title (ZH)**: 可认证正确的映射以实现即使在里程计漂移情况下的安全导航 

**Authors**: Devansh R. Agrawal, Taekyung Kim, Rajiv Govindjee, Trushant Adeshara, Jiangbo Yu, Anurekha Ravikumar, Dimitra Panagou  

**Link**: [PDF](https://arxiv.org/pdf/2504.18713)  

**Abstract**: Accurate perception, state estimation and mapping are essential for safe robotic navigation as planners and controllers rely on these components for safety-critical decisions. However, existing mapping approaches often assume perfect pose estimates, an unrealistic assumption that can lead to incorrect obstacle maps and therefore collisions. This paper introduces a framework for certifiably-correct mapping that ensures that the obstacle map correctly classifies obstacle-free regions despite the odometry drift in vision-based localization systems (VIO}/SLAM). By deflating the safe region based on the incremental odometry error at each timestep, we ensure that the map remains accurate and reliable locally around the robot, even as the overall odometry error with respect to the inertial frame grows unbounded.
Our contributions include two approaches to modify popular obstacle mapping paradigms, (I) Safe Flight Corridors, and (II) Signed Distance Fields. We formally prove the correctness of both methods, and describe how they integrate with existing planning and control modules. Simulations using the Replica dataset highlight the efficacy of our methods compared to state-of-the-art techniques. Real-world experiments with a robotic rover show that, while baseline methods result in collisions with previously mapped obstacles, the proposed framework enables the rover to safely stop before potential collisions. 

**Abstract (ZH)**: 准确感知、状态估计与建图对于安全的机器人导航至关重要，因为规划器和控制器依赖这些组件来做安全关键决策。然而，现有建图方法往往假设完美的姿态估计，这是一个不现实的假设，可能导致错误的障碍物地图并因此引发碰撞。本文介绍了一种确认证确的建图框架，确保即使在基于视觉定位系统（VIO/SLAM）的姿态漂移情况下，障碍物地图也能正确分类无障碍区域。通过基于每个时间步的增量姿态误差进行区域收缩，我们确保地图在机器人周围保持局部准确可靠，即使全局姿态误差相对于惯性框架无限增长也是如此。我们的贡献包括两类修改流行障碍物建图方法的途径，（I）安全飞行走廊，（II）符号距离场。我们形式化证明了两种方法的正确性，并描述了它们如何与现有的规划和控制模块集成。复制品数据集的仿真结果显示了我们方法的有效性，优于现有最先进的技术。在机器人漫游车的实际实验中表明，尽管基线方法会导致与先前映射的障碍物相撞，但提出的框架使漫游车能够在潜在碰撞前安全停止。 

---
# Decentralized Fusion of 3D Extended Object Tracking based on a B-Spline Shape Model 

**Title (ZH)**: 基于B-Spline形状模型的分布式三维扩展目标跟踪融合 

**Authors**: Longfei Han, Klaus Kefferpütz, Jürgen Beyerer  

**Link**: [PDF](https://arxiv.org/pdf/2504.18708)  

**Abstract**: Extended Object Tracking (EOT) exploits the high resolution of modern sensors for detailed environmental perception. Combined with decentralized fusion, it contributes to a more scalable and robust perception system. This paper investigates the decentralized fusion of 3D EOT using a B-spline curve based model. The spline curve is used to represent the side-view profile, which is then extruded with a width to form a 3D shape. We use covariance intersection (CI) for the decentralized fusion and discuss the challenge of applying it to EOT. We further evaluate the tracking result of the decentralized fusion with simulated and real datasets of traffic scenarios. We show that the CI-based fusion can significantly improve the tracking performance for sensors with unfavorable perspective. 

**Abstract (ZH)**: 基于B-样条曲线模型的分布式融合扩展目标跟踪 

---
# Robust Push Recovery on Bipedal Robots: Leveraging Multi-Domain Hybrid Systems with Reduced-Order Model Predictive Control 

**Title (ZH)**: 双足机器人稳健的推送恢复：基于降阶模型预测控制的多领域混合系统方法 

**Authors**: Min Dai, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2504.18698)  

**Abstract**: In this paper, we present a novel control framework to achieve robust push recovery on bipedal robots while locomoting. The key contribution is the unification of hybrid system models of locomotion with a reduced-order model predictive controller determining: foot placement, step timing, and ankle control. The proposed reduced-order model is an augmented Linear Inverted Pendulum model with zero moment point coordinates; this is integrated within a model predictive control framework for robust stabilization under external disturbances. By explicitly leveraging the hybrid dynamics of locomotion, our approach significantly improves stability and robustness across varying walking heights, speeds, step durations, and is effective for both flat-footed and more complex multi-domain heel-to-toe walking patterns. The framework is validated with high-fidelity simulation on Cassie, a 3D underactuated robot, showcasing real-time feasibility and substantially improved stability. The results demonstrate the robustness of the proposed method in dynamic environments. 

**Abstract (ZH)**: 本文提出了一种新颖的控制框架，以实现具有运动能力的双足机器人在行进过程中稳健的推倒恢复。关键贡献是将行进的混合系统模型与降低阶数的模型预测控制器结合，该控制器决定：脚部放置、步态时间以及踝关节控制。提出的降低阶数模型为带零力点坐标的增广线性倒 pendulum 模型；该模型整合到一种模型预测控制框架中，以在外部干扰下实现鲁棒的稳定化。通过明确利用行进的混合动力学特性，我们的方法显著提高了在不同行走高度、速度和步长时间下的稳定性，并且对于平足行走模式和更复杂的多领域后跟到脚尖行走模式都有效。该框架通过在高保真模拟中使用 Cassie（一种 3D 欠驱动机器人）进行验证，展示了实时可行性和显著改善的稳定性。结果表明，所提出方法在动态环境中的鲁棒性。 

---
# Learning-Based Modeling of Soft Actuators Using Euler Spiral-Inspired Curvature 

**Title (ZH)**: 基于欧拉螺旋启发的曲率学习建模方法用于软执行器 

**Authors**: Yu Mei, Shangyuan Yuan, Xinda Qi, Preston Fairchild, Xiaobo Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.18692)  

**Abstract**: Soft robots, distinguished by their inherent compliance and continuum structures, present unique modeling challenges, especially when subjected to significant external loads such as gravity and payloads. In this study, we introduce an innovative data-driven modeling framework leveraging an Euler spiral-inspired shape representations to accurately describe the complex shapes of soft continuum actuators. Based on this representation, we develop neural network-based forward and inverse models to effectively capture the nonlinear behavior of a fiber-reinforced pneumatic bending actuator. Our forward model accurately predicts the actuator's deformation given inputs of pressure and payload, while the inverse model reliably estimates payloads from observed actuator shapes and known pressure inputs. Comprehensive experimental validation demonstrates the effectiveness and accuracy of our proposed approach. Notably, the augmented Euler spiral-based forward model achieves low average positional prediction errors of 3.38%, 2.19%, and 1.93% of the actuator length at the one-third, two-thirds, and tip positions, respectively. Furthermore, the inverse model demonstrates precision of estimating payloads with an average error as low as 0.72% across the tested range. These results underscore the potential of our method to significantly enhance the accuracy and predictive capabilities of modeling frameworks for soft robotic systems. 

**Abstract (ZH)**: 软体机器人因其固有的顺应性和连续结构，在承受重力和载荷等显著外部负载时，呈现出独特的建模挑战。本文引入了一种基于欧拉螺旋启发式的数据驱动建模框架，以准确描述软连续执行器的复杂形状。在此基础上，我们开发了基于神经网络的正向和逆向模型，有效捕捉了纤维增强气动弯曲执行器的非线性行为。我们的正向模型能够根据压力和载荷输入准确预测执行器的变形，而逆向模型可以从观测到的执行器形状和已知的压力输入中可靠地估计载荷。全面的实验验证证明了所提方法的有效性和准确性。值得注意的是，增强的欧拉螺旋基于的正向模型分别在执行器长度的三分之一、二分之一和尖端位置实现了平均位置预测误差低至3.38%、2.19%和1.93%。此外，逆向模型在测试范围内估计载荷的精度平均误差低至0.72%。这些结果强调了本文方法在提升软体机器人系统建模框架的准确性和预测能力方面的潜在价值。 

---
# Collaborative Object Transportation in Space via Impact Interactions 

**Title (ZH)**: 空间中基于碰撞交互的协同对象运输 

**Authors**: Joris Verhagen, Jana Tumova  

**Link**: [PDF](https://arxiv.org/pdf/2504.18667)  

**Abstract**: We present a planning and control approach for collaborative transportation of objects in space by a team of robots. Object and robots in microgravity environments are not subject to friction but are instead free floating. This property is key to how we approach the transportation problem: the passive objects are controlled by impact interactions with the controlled robots. In particular, given a high-level Signal Temporal Logic (STL) specification of the transportation task, we synthesize motion plans for the robots to maximize the specification satisfaction in terms of spatial STL robustness. Given that the physical impact interactions are complex and hard to model precisely, we also present an alternative formulation maximizing the permissible uncertainty in a simplified kinematic impact model. We define the full planning and control stack required to solve the object transportation problem; an offline planner, an online replanner, and a low-level model-predictive control scheme for each of the robots. We show the method in a high-fidelity simulator for a variety of scenarios and present experimental validation of 2-robot, 1-object scenarios on a freeflyer platform. 

**Abstract (ZH)**: 一种用于空间机器人团队协作运输物体的规划与控制方法 

---
# M2R2: MulitModal Robotic Representation for Temporal Action Segmentation 

**Title (ZH)**: M2R2: 多模态机器人表示用于时间动作分割 

**Authors**: Daniel Sliwowski, Dongheui Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.18662)  

**Abstract**: Temporal action segmentation (TAS) has long been a key area of research in both robotics and computer vision. In robotics, algorithms have primarily focused on leveraging proprioceptive information to determine skill boundaries, with recent approaches in surgical robotics incorporating vision. In contrast, computer vision typically relies on exteroceptive sensors, such as cameras. Existing multimodal TAS models in robotics integrate feature fusion within the model, making it difficult to reuse learned features across different models. Meanwhile, pretrained vision-only feature extractors commonly used in computer vision struggle in scenarios with limited object visibility. In this work, we address these challenges by proposing M2R2, a multimodal feature extractor tailored for TAS, which combines information from both proprioceptive and exteroceptive sensors. We introduce a novel pretraining strategy that enables the reuse of learned features across multiple TAS models. Our method achieves state-of-the-art performance on the REASSEMBLE dataset, a challenging multimodal robotic assembly dataset, outperforming existing robotic action segmentation models by 46.6%. Additionally, we conduct an extensive ablation study to evaluate the contribution of different modalities in robotic TAS tasks. 

**Abstract (ZH)**: 多模态动作分割（Multimodal Temporal Action Segmentation, TAS）一直是机器人学和计算机视觉领域的关键研究方向。在机器人学中，算法主要侧重于利用本体感觉信息确定技能边界，而最近的外科机器人技术则开始结合视觉信息。相比之下，计算机视觉通常依赖于外部感受器，如摄像头。现有的机器人多模态TAS模型在模型内部融合特征，使得这些特征难以在不同模型之间复用。同时，计算机视觉中常用的预训练的仅基于视觉特征提取器在物体可见度有限的场景中表现不佳。在本工作中，我们提出了一种名为M2R2的多模态特征提取器，专门用于TAS，结合了本体感觉和外部感受器的信息。我们引入了一种新的预训练策略，使学习到的特征能够在多个TAS模型之间复用。我们的方法在REASSEMBLE数据集上取得了最先进的性能，该数据集是一个具有挑战性的多模态机器人装配数据集，相比现有的机器人动作分割模型提高了46.6%的性能。此外，我们还进行了详尽的消融研究，以评估不同模态在机器人TAS任务中的贡献。 

---
# DriVerse: Navigation World Model for Driving Simulation via Multimodal Trajectory Prompting and Motion Alignment 

**Title (ZH)**: DriVerse: 驾驶模拟的多模态轨迹提示与运动对齐导航世界模型 

**Authors**: Xiaofan Li, Chenming Wu, Zhao Yang, Zhihao Xu, Dingkang Liang, Yumeng Zhang, Ji Wan, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18576)  

**Abstract**: This paper presents DriVerse, a generative model for simulating navigation-driven driving scenes from a single image and a future trajectory. Previous autonomous driving world models either directly feed the trajectory or discrete control signals into the generation pipeline, leading to poor alignment between the control inputs and the implicit features of the 2D base generative model, which results in low-fidelity video outputs. Some methods use coarse textual commands or discrete vehicle control signals, which lack the precision to guide fine-grained, trajectory-specific video generation, making them unsuitable for evaluating actual autonomous driving algorithms. DriVerse introduces explicit trajectory guidance in two complementary forms: it tokenizes trajectories into textual prompts using a predefined trend vocabulary for seamless language integration, and converts 3D trajectories into 2D spatial motion priors to enhance control over static content within the driving scene. To better handle dynamic objects, we further introduce a lightweight motion alignment module, which focuses on the inter-frame consistency of dynamic pixels, significantly enhancing the temporal coherence of moving elements over long sequences. With minimal training and no need for additional data, DriVerse outperforms specialized models on future video generation tasks across both the nuScenes and Waymo datasets. The code and models will be released to the public. 

**Abstract (ZH)**: 本文介绍了DriVerse，一种从单张图像和未来轨迹生成由导航驱动的驾驶场景的生成模型。之前的自主驾驶世界模型要么直接将轨迹或离散控制信号输入生成管道，要么使用粗糙的文本命令或离散的车辆控制信号，这导致控制输入与2D基础生成模型的隐式特征对齐较差，从而产生低保真度的视频输出。一些方法使用粗略的文本命令或离散的车辆控制信号，缺乏对细粒度、轨迹特定视频生成的精度指导，使其不适合评估实际的自主驾驶算法。DriVerse通过两种互补的方式引入了显式的轨迹指导：使用预定义的趋势词汇对轨迹进行标记化，以便无缝的语言整合，并将3D轨迹转换为2D空间运动先验，以增强对驾驶场景中静态内容的控制。为了更好地处理动态对象，我们进一步引入了一个轻量级的运动对齐模块，该模块专注于帧间动态像素的连续性，显著增强了长时间序列中移动元素的时序一致性。DriVerse在nuScenes和Waymo数据集上的未来视频生成任务中表现出色，训练量小且无需额外数据，代码和模型将对公众开放。 

---
# MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion 

**Title (ZH)**: MP-SfM: 单目表面先验用于稳健的结构从运动重建 

**Authors**: Zador Pataki, Paul-Edouard Sarlin, Johannes L. Schönberger, Marc Pollefeys  

**Link**: [PDF](https://arxiv.org/pdf/2504.20040)  

**Abstract**: While Structure-from-Motion (SfM) has seen much progress over the years, state-of-the-art systems are prone to failure when facing extreme viewpoint changes in low-overlap, low-parallax or high-symmetry scenarios. Because capturing images that avoid these pitfalls is challenging, this severely limits the wider use of SfM, especially by non-expert users. We overcome these limitations by augmenting the classical SfM paradigm with monocular depth and normal priors inferred by deep neural networks. Thanks to a tight integration of monocular and multi-view constraints, our approach significantly outperforms existing ones under extreme viewpoint changes, while maintaining strong performance in standard conditions. We also show that monocular priors can help reject faulty associations due to symmetries, which is a long-standing problem for SfM. This makes our approach the first capable of reliably reconstructing challenging indoor environments from few images. Through principled uncertainty propagation, it is robust to errors in the priors, can handle priors inferred by different models with little tuning, and will thus easily benefit from future progress in monocular depth and normal estimation. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 尽管结构从运动（SfM）技术在过去几年取得了显著进步，最新的系统在低重叠、低视角差或高对称场景下的极端视角变化下容易失败。由于避免这些难题捕捉图像具有挑战性，这严重限制了SfM的广泛应用，尤其是对非专家用户。我们通过结合经典的SfM paradigm与由深度神经网络推断出的单目深度和法线先验，克服了这些限制。得益于单目和多视图约束的紧密整合，我们的方法在极端视角变化下显著优于现有方法，同时在标准条件下保持了强大的性能。我们还展示了单目先验可以帮助拒绝由于对称性引起的错误关联，这是SfM长期以来的一个难题。这使得我们的方法成为首个能够可靠地从少量图像重建具有挑战性的室内环境的方法。通过原理上的不确定性传播，该方法对于先验中的错误具有鲁棒性，能够处理由不同模型推断出的先验，且不需要大量调整，因此将容易从未来的单目深度和法线估计进展中受益。我们的代码已在以下网址公开：this https URL。 

---
# Modelling of Underwater Vehicles using Physics-Informed Neural Networks with Control 

**Title (ZH)**: 使用物理约束神经网络建模的水下车辆控制 

**Authors**: Abdelhakim Amer, David Felsager, Yury Brodskiy, Andriy Sarabakha  

**Link**: [PDF](https://arxiv.org/pdf/2504.20019)  

**Abstract**: Physics-informed neural networks (PINNs) integrate physical laws with data-driven models to improve generalization and sample efficiency. This work introduces an open-source implementation of the Physics-Informed Neural Network with Control (PINC) framework, designed to model the dynamics of an underwater vehicle. Using initial states, control actions, and time inputs, PINC extends PINNs to enable physically consistent transitions beyond the training domain. Various PINC configurations are tested, including differing loss functions, gradient-weighting schemes, and hyperparameters. Validation on a simulated underwater vehicle demonstrates more accurate long-horizon predictions compared to a non-physics-informed baseline 

**Abstract (ZH)**: 物理约束神经网络（PINC）框架的开源实现及其在水下车辆动力学建模中的应用 

---
# Mesh-Learner: Texturing Mesh with Spherical Harmonics 

**Title (ZH)**: 网状学习者：基于球谐函数的网格纹理化 

**Authors**: Yunfei Wan, Jianheng Liu, Jiarong Lin, Fu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19938)  

**Abstract**: In this paper, we present a 3D reconstruction and rendering framework termed Mesh-Learner that is natively compatible with traditional rasterization pipelines. It integrates mesh and spherical harmonic (SH) texture (i.e., texture filled with SH coefficients) into the learning process to learn each mesh s view-dependent radiance end-to-end. Images are rendered by interpolating surrounding SH Texels at each pixel s sampling point using a novel interpolation method. Conversely, gradients from each pixel are back-propagated to the related SH Texels in SH textures. Mesh-Learner exploits graphic features of rasterization pipeline (texture sampling, deferred rendering) to render, which makes Mesh-Learner naturally compatible with tools (e.g., Blender) and tasks (e.g., 3D reconstruction, scene rendering, reinforcement learning for robotics) that are based on rasterization pipelines. Our system can train vast, unlimited scenes because we transfer only the SH textures within the frustum to the GPU for training. At other times, the SH textures are stored in CPU RAM, which results in moderate GPU memory usage. The rendering results on interpolation and extrapolation sequences in the Replica and FAST-LIVO2 datasets achieve state-of-the-art performance compared to existing state-of-the-art methods (e.g., 3D Gaussian Splatting and M2-Mapping). To benefit the society, the code will be available at this https URL. 

**Abstract (ZH)**: 一种与传统栅格化管道本征兼容的3D重建与渲染框架：Mesh-Learner 

---
# Category-Level and Open-Set Object Pose Estimation for Robotics 

**Title (ZH)**: 机器人领域类别级和开放集物体姿态估计 

**Authors**: Peter Hönig, Matthias Hirschmanner, Markus Vincze  

**Link**: [PDF](https://arxiv.org/pdf/2504.19572)  

**Abstract**: Object pose estimation enables a variety of tasks in computer vision and robotics, including scene understanding and robotic grasping. The complexity of a pose estimation task depends on the unknown variables related to the target object. While instance-level methods already excel for opaque and Lambertian objects, category-level and open-set methods, where texture, shape, and size are partially or entirely unknown, still struggle with these basic material properties. Since texture is unknown in these scenarios, it cannot be used for disambiguating object symmetries, another core challenge of 6D object pose estimation. The complexity of estimating 6D poses with such a manifold of unknowns led to various datasets, accuracy metrics, and algorithmic solutions. This paper compares datasets, accuracy metrics, and algorithms for solving 6D pose estimation on the category-level. Based on this comparison, we analyze how to bridge category-level and open-set object pose estimation to reach generalization and provide actionable recommendations. 

**Abstract (ZH)**: 类别级6D对象姿态估计数据集、准确度指标和算法比较：如何弥合类别级与开放集对象姿态估计差距以实现泛化并提供可操作建议 

---
# Stability Enhancement in Reinforcement Learning via Adaptive Control Lyapunov Function 

**Title (ZH)**: 基于自适应控制李亚普诺夫函数的强化学习稳定性增强 

**Authors**: Donghe Chen, Han Wang, Lin Cheng, Shengping Gong  

**Link**: [PDF](https://arxiv.org/pdf/2504.19473)  

**Abstract**: Reinforcement Learning (RL) has shown promise in control tasks but faces significant challenges in real-world applications, primarily due to the absence of safety guarantees during the learning process. Existing methods often struggle with ensuring safe exploration, leading to potential system failures and restricting applications primarily to simulated environments. Traditional approaches such as reward shaping and constrained policy optimization can fail to guarantee safety during initial learning stages, while model-based methods using Control Lyapunov Functions (CLFs) or Control Barrier Functions (CBFs) may hinder efficient exploration and performance. To address these limitations, this paper introduces Soft Actor-Critic with Control Lyapunov Function (SAC-CLF), a framework that enhances stability and safety through three key innovations: (1) a task-specific CLF design method for safe and optimal performance; (2) dynamic adjustment of constraints to maintain robustness under unmodeled dynamics; and (3) improved control input smoothness while ensuring safety. Experimental results on a classical nonlinear system and satellite attitude control demonstrate the effectiveness of SAC-CLF in overcoming the shortcomings of existing methods. 

**Abstract (ZH)**: 基于控制李apunov函数的软演员-评论家（SAC-CLF）：一种增强稳定性和安全性的强化学习框架 

---
# OPAL: Visibility-aware LiDAR-to-OpenStreetMap Place Recognition via Adaptive Radial Fusion 

**Title (ZH)**: OPAL：基于可见性aware的LiDAR到OpenStreetMap的地物识别 via 自适应径向融合 

**Authors**: Shuhao Kang, Martin Y. Liao, Yan Xia, Olaf Wysocki, Boris Jutzi, Daniel Cremers  

**Link**: [PDF](https://arxiv.org/pdf/2504.19258)  

**Abstract**: LiDAR place recognition is a critical capability for autonomous navigation and cross-modal localization in large-scale outdoor environments. Existing approaches predominantly depend on pre-built 3D dense maps or aerial imagery, which impose significant storage overhead and lack real-time adaptability. In this paper, we propose OPAL, a novel network for LiDAR place recognition that leverages OpenStreetMap as a lightweight and up-to-date prior. Our key innovation lies in bridging the domain disparity between sparse LiDAR scans and structured OSM data through two carefully designed components: a cross-modal visibility mask that identifies maximal observable regions from both modalities to guide feature learning, and an adaptive radial fusion module that dynamically consolidates multiscale radial features into discriminative global descriptors. Extensive experiments on the augmented KITTI and KITTI-360 datasets demonstrate OPAL's superiority, achieving 15.98% higher recall at @1m threshold for top-1 retrieved matches while operating at 12x faster inference speeds compared to state-of-the-art approaches. Code and datasets are publicly available at: this https URL . 

**Abstract (ZH)**: 基于OpenStreetMap的LiDAR场所识别网络OPAL 

---
# LM-MCVT: A Lightweight Multi-modal Multi-view Convolutional-Vision Transformer Approach for 3D Object Recognition 

**Title (ZH)**: LM-MCVT：一种轻量级多模态多视图卷积-视觉变换器方法用于三维物体识别 

**Authors**: Songsong Xiong, Hamidreza Kasaei  

**Link**: [PDF](https://arxiv.org/pdf/2504.19256)  

**Abstract**: In human-centered environments such as restaurants, homes, and warehouses, robots often face challenges in accurately recognizing 3D objects. These challenges stem from the complexity and variability of these environments, including diverse object shapes. In this paper, we propose a novel Lightweight Multi-modal Multi-view Convolutional-Vision Transformer network (LM-MCVT) to enhance 3D object recognition in robotic applications. Our approach leverages the Globally Entropy-based Embeddings Fusion (GEEF) method to integrate multi-views efficiently. The LM-MCVT architecture incorporates pre- and mid-level convolutional encoders and local and global transformers to enhance feature extraction and recognition accuracy. We evaluate our method on the synthetic ModelNet40 dataset and achieve a recognition accuracy of 95.6% using a four-view setup, surpassing existing state-of-the-art methods. To further validate its effectiveness, we conduct 5-fold cross-validation on the real-world OmniObject3D dataset using the same configuration. Results consistently show superior performance, demonstrating the method's robustness in 3D object recognition across synthetic and real-world 3D data. 

**Abstract (ZH)**: 面向餐厅、家庭和仓库等以人为中心环境中的机器人3D物体识别挑战：一种轻量级多模态多视图卷积-视觉变换器网络（LM-MCVT）的研究 

---
# Trajectory Planning with Model Predictive Control for Obstacle Avoidance Considering Prediction Uncertainty 

**Title (ZH)**: 考虑预测不确定性的模型预测控制避障轨迹规划 

**Authors**: Eric Schöneberg, Michael Schröder, Daniel Görges, Hans D. Schotten  

**Link**: [PDF](https://arxiv.org/pdf/2504.19193)  

**Abstract**: This paper introduces a novel trajectory planner for autonomous robots, specifically designed to enhance navigation by incorporating dynamic obstacle avoidance within the Robot Operating System 2 (ROS2) and Navigation 2 (Nav2) framework. The proposed method utilizes Model Predictive Control (MPC) with a focus on handling the uncertainties associated with the movement prediction of dynamic obstacles. Unlike existing Nav2 trajectory planners which primarily deal with static obstacles or react to the current position of dynamic obstacles, this planner predicts future obstacle positions using a stochastic Vector Auto-Regressive Model (VAR). The obstacles' future positions are represented by probability distributions, and collision avoidance is achieved through constraints based on the Mahalanobis distance, ensuring the robot avoids regions where obstacles are likely to be. This approach considers the robot's kinodynamic constraints, enabling it to track a reference path while adapting to real-time changes in the environment. The paper details the implementation, including obstacle prediction, tracking, and the construction of feasible sets for MPC. Simulation results in a Gazebo environment demonstrate the effectiveness of this method in scenarios where robots must navigate around each other, showing improved collision avoidance capabilities. 

**Abstract (ZH)**: 基于ROS2和Nav2框架的具有动态障碍物避障功能的新型轨迹规划器研究 

---
# LRFusionPR: A Polar BEV-Based LiDAR-Radar Fusion Network for Place Recognition 

**Title (ZH)**: LRFusionPR: 一种基于极坐标BEV的LiDAR-雷达融合网络用于场所识别 

**Authors**: Zhangshuo Qi, Luqi Cheng, Zijie Zhou, Guangming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2504.19186)  

**Abstract**: In autonomous driving, place recognition is critical for global localization in GPS-denied environments. LiDAR and radar-based place recognition methods have garnered increasing attention, as LiDAR provides precise ranging, whereas radar excels in adverse weather resilience. However, effectively leveraging LiDAR-radar fusion for place recognition remains challenging. The noisy and sparse nature of radar data limits its potential to further improve recognition accuracy. In addition, heterogeneous radar configurations complicate the development of unified cross-modality fusion frameworks. In this paper, we propose LRFusionPR, which improves recognition accuracy and robustness by fusing LiDAR with either single-chip or scanning radar. Technically, a dual-branch network is proposed to fuse different modalities within the unified polar coordinate bird's eye view (BEV) representation. In the fusion branch, cross-attention is utilized to perform cross-modality feature interactions. The knowledge from the fusion branch is simultaneously transferred to the distillation branch, which takes radar as its only input to further improve the robustness. Ultimately, the descriptors from both branches are concatenated, producing the multimodal global descriptor for place retrieval. Extensive evaluations on multiple datasets demonstrate that our LRFusionPR achieves accurate place recognition, while maintaining robustness under varying weather conditions. Our open-source code will be released at this https URL. 

**Abstract (ZH)**: 基于LiDAR和雷达融合的自主驾驶场景识别方法 

---
# Towards Latency-Aware 3D Streaming Perception for Autonomous Driving 

**Title (ZH)**: 面向延迟感知的3D流式自动驾驶感知 

**Authors**: Jiaqi Peng, Tai Wang, Jiangmiao Pang, Yuan Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.19115)  

**Abstract**: Although existing 3D perception algorithms have demonstrated significant improvements in performance, their deployment on edge devices continues to encounter critical challenges due to substantial runtime latency. We propose a new benchmark tailored for online evaluation by considering runtime latency. Based on the benchmark, we build a Latency-Aware 3D Streaming Perception (LASP) framework that addresses the latency issue through two primary components: 1) latency-aware history integration, which extends query propagation into a continuous process, ensuring the integration of historical feature regardless of varying latency; 2) latency-aware predictive detection, a module that compensates the detection results with the predicted trajectory and the posterior accessed latency. By incorporating the latency-aware mechanism, our method shows generalization across various latency levels, achieving an online performance that closely aligns with 80\% of its offline evaluation on the Jetson AGX Orin without any acceleration techniques. 

**Abstract (ZH)**: 虽然现有的3D感知算法在性能上已经取得了显著进步，但它们在边缘设备上的部署仍然面临由于运行时延迟巨大的挑战。我们提出了一种新的基准，旨在考虑运行时延迟进行在线评估。基于该基准，我们构建了一种latency-aware 3D流式感知（LASP）框架，该框架通过两个主要组成部分来解决延迟问题：1) latency-aware历史集成，将查询传播扩展为连续过程，确保在各种延迟情况下历史特征的集成；2) latency-aware预测检测模块，该模块利用预测轨迹和后验访问的延迟来补偿检测结果。通过引入latency-aware机制，我们的方法在各种延迟水平下表现出通用性，在Jetson AGX Orin上实现的在线性能与80%的离线评估结果相匹配，无需任何加速技术。 

---
# Snake locomotion learning search 

**Title (ZH)**: 蛇形运动学习搜索 

**Authors**: Sheng-Xue He  

**Link**: [PDF](https://arxiv.org/pdf/2504.19114)  

**Abstract**: This research introduces a novel heuristic algorithm known as the Snake Locomotion Learning Search algorithm (SLLS) designed to address optimization problems. The SLLS draws inspiration from the locomotion patterns observed in snakes, particularly serpentine and caterpillar locomotion. We leverage these two modes of snake locomotion to devise two distinct search mechanisms within the SLLS. In our quest to mimic a snake's natural adaptation to its surroundings, we incorporate a learning efficiency component generated from the Sigmoid function. This helps strike a balance between exploration and exploitation capabilities throughout the SLLS computation process. The efficacy and effectiveness of this innovative algorithm are demonstrated through its application to 60 standard benchmark optimization problems and seven well-known engineering optimization problems. The performance analysis reveals that in most cases, the SLLS outperforms other algorithms, and even in the remaining scenarios, it exhibits robust performance. This conforms to the No Free Lunch Theorem, affirming that the SLLS stands as a valuable heuristic algorithm with significant potential for effectively addressing specific optimization challenges. 

**Abstract (ZH)**: Snake Locomotion Learning Search算法（SLLS）在优化问题中的应用研究 

---
# Learning to Drive from a World Model 

**Title (ZH)**: 从世界模型学习驾驶 

**Authors**: Mitchell Goff, Greg Hogan, George Hotz, Armand du Parc Locmaria, Kacper Raczy, Harald Schäfer, Adeeb Shihadeh, Weixing Zhang, Yassine Yousfi  

**Link**: [PDF](https://arxiv.org/pdf/2504.19077)  

**Abstract**: Most self-driving systems rely on hand-coded perception outputs and engineered driving rules. Learning directly from human driving data with an end-to-end method can allow for a training architecture that is simpler and scales well with compute and data.
In this work, we propose an end-to-end training architecture that uses real driving data to train a driving policy in an on-policy simulator. We show two different methods of simulation, one with reprojective simulation and one with a learned world model. We show that both methods can be used to train a policy that learns driving behavior without any hand-coded driving rules. We evaluate the performance of these policies in a closed-loop simulation and when deployed in a real-world advanced driver-assistance system. 

**Abstract (ZH)**: 一种使用实际驾驶数据训练驾驶策略的端到端训练架构：基于策略模拟的方法和 Learned 世界模型的实现 

---
# Deep Learning-Based Multi-Modal Fusion for Robust Robot Perception and Navigation 

**Title (ZH)**: 基于深度学习的多模态融合方法用于鲁棒机器人感知与导航 

**Authors**: Delun Lai, Yeyubei Zhang, Yunchong Liu, Chaojie Li, Huadong Mo  

**Link**: [PDF](https://arxiv.org/pdf/2504.19002)  

**Abstract**: This paper introduces a novel deep learning-based multimodal fusion architecture aimed at enhancing the perception capabilities of autonomous navigation robots in complex environments. By utilizing innovative feature extraction modules, adaptive fusion strategies, and time-series modeling mechanisms, the system effectively integrates RGB images and LiDAR data. The key contributions of this work are as follows: a. the design of a lightweight feature extraction network to enhance feature representation; b. the development of an adaptive weighted cross-modal fusion strategy to improve system robustness; and c. the incorporation of time-series information modeling to boost dynamic scene perception accuracy. Experimental results on the KITTI dataset demonstrate that the proposed approach increases navigation and positioning accuracy by 3.5% and 2.2%, respectively, while maintaining real-time performance. This work provides a novel solution for autonomous robot navigation in complex environments. 

**Abstract (ZH)**: 基于深度学习的多模态融合架构在复杂环境自主导航机器人感知能力增强的研究 

---
# WLTCL: Wide Field-of-View 3-D LiDAR Truck Compartment Automatic Localization System 

**Title (ZH)**: WLTCL: 广视野3D LiDAR货箱自动定位系统 

**Authors**: Guodong Sun, Mingjing Li, Dingjie Liu, Mingxuan Liu, Bo Wu, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18870)  

**Abstract**: As an essential component of logistics automation, the automated loading system is becoming a critical technology for enhancing operational efficiency and safety. Precise automatic positioning of the truck compartment, which serves as the loading area, is the primary step in automated loading. However, existing methods have difficulty adapting to truck compartments of various sizes, do not establish a unified coordinate system for LiDAR and mobile manipulators, and often exhibit reliability issues in cluttered environments. To address these limitations, our study focuses on achieving precise automatic positioning of key points in large, medium, and small fence-style truck compartments in cluttered scenarios. We propose an innovative wide field-of-view 3-D LiDAR vehicle compartment automatic localization system. For vehicles of various sizes, this system leverages the LiDAR to generate high-density point clouds within an extensive field-of-view range. By incorporating parking area constraints, our vehicle point cloud segmentation method more effectively segments vehicle point clouds within the scene. Our compartment key point positioning algorithm utilizes the geometric features of the compartments to accurately locate the corner points, providing stackable spatial regions. Extensive experiments on our collected data and public datasets demonstrate that this system offers reliable positioning accuracy and reduced computational resource consumption, leading to its application and promotion in relevant fields. 

**Abstract (ZH)**: 物流自动化中必不可少的自动加载系统已成为提升操作效率和安全性的关键技术。卡车货箱的精确自动定位作为装载区域的第一步，至关重要。然而，现有的方法难以适应不同尺寸的卡车货箱，未能为LiDAR和移动 manipulator建立统一的坐标系统，且在复杂环境中可靠性差。为弥补这些局限性，本研究专注于在复杂场景中实现大、中、小型围栏式卡车货箱的关键点的精确自动定位。我们提出了一种创新的广角3D LiDAR车辆货箱自动定位系统。对于不同尺寸的车辆，该系统利用LiDAR在大范围视场内生成高密度点云。通过结合停车区域约束，我们的车辆点云分割方法更有效地在场景中分割车辆点云。我们提出的货箱关键点定位算法利用货箱的几何特征准确定位角点，提供堆叠的空间区域。在我们收集的数据集和公开数据集上的 extensive 实验显示，该系统提供了可靠的定位精度并减少了计算资源消耗，从而在相关领域得到了应用和推广。 

---
# Hierarchical Reinforcement Learning in Multi-Goal Spatial Navigation with Autonomous Mobile Robots 

**Title (ZH)**: 多目标空间导航中基于层次的强化学习方法研究（自主移动机器人） 

**Authors**: Brendon Johnson, Alfredo Weitzenfeld  

**Link**: [PDF](https://arxiv.org/pdf/2504.18794)  

**Abstract**: Hierarchical reinforcement learning (HRL) is hypothesized to be able to take advantage of the inherent hierarchy in robot learning tasks with sparse reward schemes, in contrast to more traditional reinforcement learning algorithms. In this research, hierarchical reinforcement learning is evaluated and contrasted with standard reinforcement learning in complex navigation tasks. We evaluate unique characteristics of HRL, including their ability to create sub-goals and the termination function. We constructed experiments to test the differences between PPO and HRL, different ways of creating sub-goals, manual vs automatic sub-goal creation, and the effects of the frequency of termination on performance. These experiments highlight the advantages of HRL and how it achieves these advantages. 

**Abstract (ZH)**: 分层强化学习（HRL）被假设为能够在稀疏奖励方案下利用机器人学习任务中的固有层次结构，相较于传统的强化学习算法。在本研究中，分层强化学习在复杂导航任务中与标准强化学习进行了评估和对比。我们评估了分层强化学习的独特特性，包括其创建子目标和终止函数的能力。我们构建了实验来测试PPO与HRL之间的差异、不同的子目标创建方式、手动与自动子目标创建方式以及终止频率对性能的影响。这些实验突显了分层强化学习的优势及其如何实现这些优势。 

---
# SORT3D: Spatial Object-centric Reasoning Toolbox for Zero-Shot 3D Grounding Using Large Language Models 

**Title (ZH)**: SORT3D: 空间对象中心推理工具箱——基于大型语言模型的零样本3D 地基推理 

**Authors**: Nader Zantout, Haochen Zhang, Pujith Kachana, Jinkai Qiu, Ji Zhang, Wenshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18684)  

**Abstract**: Interpreting object-referential language and grounding objects in 3D with spatial relations and attributes is essential for robots operating alongside humans. However, this task is often challenging due to the diversity of scenes, large number of fine-grained objects, and complex free-form nature of language references. Furthermore, in the 3D domain, obtaining large amounts of natural language training data is difficult. Thus, it is important for methods to learn from little data and zero-shot generalize to new environments. To address these challenges, we propose SORT3D, an approach that utilizes rich object attributes from 2D data and merges a heuristics-based spatial reasoning toolbox with the ability of large language models (LLMs) to perform sequential reasoning. Importantly, our method does not require text-to-3D data for training and can be applied zero-shot to unseen environments. We show that SORT3D achieves state-of-the-art performance on complex view-dependent grounding tasks on two benchmarks. We also implement the pipeline to run real-time on an autonomous vehicle and demonstrate that our approach can be used for object-goal navigation on previously unseen real-world environments. All source code for the system pipeline is publicly released at this https URL . 

**Abstract (ZH)**: 基于空间关系和属性将物体参考语言解释为3D中的物体对于与人类一同操作的机器人至关重要。然而，由于场景的多样性、精细物体的大量存在以及语言参考的复杂自由形态，这一任务往往极具挑战性。此外，在3D领域，获得大量自然语言训练数据是困难的。因此，对于方法而言，从少量数据中学习并在新环境中进行零样本泛化非常重要。为应对这些挑战，我们提出SORT3D，这是一种利用2D数据丰富的物体属性并通过启发式的空间推理工具箱与大型语言模型的顺序推理能力相结合的方法。重要的是，我们的方法不需要文本到3D数据的训练，并且可以针对未见环境进行零样本应用。我们展示SORT3D在两个基准上的复杂视角依赖 grounding 任务中达到了最先进的性能。我们还实现了一整套流水线在自主车辆上实时运行，并展示了我们的方法可以在未见的真实世界环境中用于物体目标导航。该系统的全部源代码已在此 https URL 公开发布。 

---
