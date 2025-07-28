# ReCoDe: Reinforcement Learning-based Dynamic Constraint Design for Multi-Agent Coordination 

**Title (ZH)**: ReCoDe: 基于强化学习的多agent协调动态约束设计 

**Authors**: Michael Amir, Guang Yang, Zhan Gao, Keisuke Okumura, Heedo Woo, Amanda Prorok  

**Link**: [PDF](https://arxiv.org/pdf/2507.19151)  

**Abstract**: Constraint-based optimization is a cornerstone of robotics, enabling the design of controllers that reliably encode task and safety requirements such as collision avoidance or formation adherence. However, handcrafted constraints can fail in multi-agent settings that demand complex coordination. We introduce ReCoDe--Reinforcement-based Constraint Design--a decentralized, hybrid framework that merges the reliability of optimization-based controllers with the adaptability of multi-agent reinforcement learning. Rather than discarding expert controllers, ReCoDe improves them by learning additional, dynamic constraints that capture subtler behaviors, for example, by constraining agent movements to prevent congestion in cluttered scenarios. Through local communication, agents collectively constrain their allowed actions to coordinate more effectively under changing conditions. In this work, we focus on applications of ReCoDe to multi-agent navigation tasks requiring intricate, context-based movements and consensus, where we show that it outperforms purely handcrafted controllers, other hybrid approaches, and standard MARL baselines. We give empirical (real robot) and theoretical evidence that retaining a user-defined controller, even when it is imperfect, is more efficient than learning from scratch, especially because ReCoDe can dynamically change the degree to which it relies on this controller. 

**Abstract (ZH)**: 基于强化学习的约束设计：ReCoDe——一种去中心化的混合框架 

---
# Diverse and Adaptive Behavior Curriculum for Autonomous Driving: A Student-Teacher Framework with Multi-Agent RL 

**Title (ZH)**: 自主驾驶的多样化和适应性行为课程：基于多Agent强化学习的学生-教师框架 

**Authors**: Ahmed Abouelazm, Johannes Ratz, Philip Schörner, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2507.19146)  

**Abstract**: Autonomous driving faces challenges in navigating complex real-world traffic, requiring safe handling of both common and critical scenarios. Reinforcement learning (RL), a prominent method in end-to-end driving, enables agents to learn through trial and error in simulation. However, RL training often relies on rule-based traffic scenarios, limiting generalization. Additionally, current scenario generation methods focus heavily on critical scenarios, neglecting a balance with routine driving behaviors. Curriculum learning, which progressively trains agents on increasingly complex tasks, is a promising approach to improving the robustness and coverage of RL driving policies. However, existing research mainly emphasizes manually designed curricula, focusing on scenery and actor placement rather than traffic behavior dynamics. This work introduces a novel student-teacher framework for automatic curriculum learning. The teacher, a graph-based multi-agent RL component, adaptively generates traffic behaviors across diverse difficulty levels. An adaptive mechanism adjusts task difficulty based on student performance, ensuring exposure to behaviors ranging from common to critical. The student, though exchangeable, is realized as a deep RL agent with partial observability, reflecting real-world perception constraints. Results demonstrate the teacher's ability to generate diverse traffic behaviors. The student, trained with automatic curricula, outperformed agents trained on rule-based traffic, achieving higher rewards and exhibiting balanced, assertive driving. 

**Abstract (ZH)**: 自主驾驶在导航复杂实际交通环境中面临挑战，需要安全处理常见和关键场景。强化学习（RL）作为一种端到端驾驶的主流方法，使代理能够在模拟中通过试错学习。然而，RL训练往往依赖于基于规则的交通场景，限制了泛化能力。此外，当前的场景生成方法主要集中在关键场景上，忽略了与常规驾驶行为的平衡。层次学习作为一种逐步训练代理执行越来越复杂任务的方法，是提高RL驾驶策略的稳健性和覆盖面的有前途的方法。然而，现有研究主要强调手动设计的课程，关注于场景和行为者的布局而非交通行为动力学。本文介绍了一种用于自动层次学习的新颖学生-教师框架。教师作为一个基于图的多代理RL组件，能够自适应地生成不同难度级别的交通行为。一种自适应机制根据学生的表现调整任务难度，确保学生接触从常见到关键的各种行为。学生，虽然可以替换，但实现为具有部分可观测性的深度RL代理，反映了现实世界的感知约束。结果表明，教师能够生成多样化的交通行为。接受自动课程训练的学生，在奖励和表现平衡性方面优于基于规则的交通场景训练的代理。 

---
# Monocular Vision-Based Swarm Robot Localization Using Equilateral Triangular Formations 

**Title (ZH)**: 基于单目视觉和等边三角形 formations 的群机器人定位 

**Authors**: Taewon Kang, Ji-Wook Kwon, Il Bae, Jin Hyo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.19100)  

**Abstract**: Localization of mobile robots is crucial for deploying robots in real-world applications such as search and rescue missions. This work aims to develop an accurate localization system applicable to swarm robots equipped only with low-cost monocular vision sensors and visual markers. The system is designed to operate in fully open spaces, without landmarks or support from positioning infrastructures. To achieve this, we propose a localization method based on equilateral triangular formations. By leveraging the geometric properties of equilateral triangles, the accurate two-dimensional position of each participating robot is estimated using one-dimensional lateral distance information between robots, which can be reliably and accurately obtained with a low-cost monocular vision sensor. Experimental and simulation results demonstrate that, as travel time increases, the positioning error of the proposed method becomes significantly smaller than that of a conventional dead-reckoning system, another low-cost localization approach applicable to open environments. 

**Abstract (ZH)**: 移动机器人定位是将其部署到实际应用如搜索与救援任务中的关键。本文旨在开发一种适用于仅装备低成本单目视觉传感器和视觉标记的群体机器人的精确定位系统。该系统设计用于开放空间工作，无需地标或位置基础设施的支持。为实现这一目标，我们提出了一种基于等边三角形编队的定位方法。通过利用等边三角形的几何特性，使用机器人间的一维侧向距离信息，可以准确估计每个参与机器人的二维位置，而这种一维侧向距离信息可通过低成本单目视觉传感器可靠且准确地获取。实验和仿真结果表明，随着行程时间的增加，所提出方法的定位误差显著小于传统里程计系统，这是另一种适用于开放环境的低成本定位方法。 

---
# Frequency Response Data-Driven Disturbance Observer Design for Flexible Joint Robots 

**Title (ZH)**: 柔性关节机器人基于频域数据的扰动观测器设计 

**Authors**: Deokjin Lee, Junho Song, Alireza Karimi, Sehoon Oh  

**Link**: [PDF](https://arxiv.org/pdf/2507.18979)  

**Abstract**: Motion control of flexible joint robots (FJR) is challenged by inherent flexibility and configuration-dependent variations in system dynamics. While disturbance observers (DOB) can enhance system robustness, their performance is often limited by the elasticity of the joints and the variations in system parameters, which leads to a conservative design of the DOB. This paper presents a novel frequency response function (FRF)-based optimization method aimed at improving DOB performance, even in the presence of flexibility and system variability. The proposed method maximizes control bandwidth and effectively suppresses vibrations, thus enhancing overall system performance. Closed-loop stability is rigorously proven using the Nyquist stability criterion. Experimental validation on a FJR demonstrates that the proposed approach significantly improves robustness and motion performance, even under conditions of joint flexibility and system variation. 

**Abstract (ZH)**: 基于频率响应函数优化的柔性关节机器人扰动观测器性能改进研究 

---
# Assessing the Reliability and Validity of a Balance Mat for Measuring Postural Stability: A Combined Robot-Human Approach 

**Title (ZH)**: 基于机器人-人类联合方法评估平衡垫测量姿势稳定性可靠性和有效性的研究 

**Authors**: Abishek Shrestha, Damith Herath, Angie Fearon, Maryam Ghahramani  

**Link**: [PDF](https://arxiv.org/pdf/2507.18943)  

**Abstract**: Postural sway assessment is important for detecting balance problems and identifying people at risk of falls. Force plates (FP) are considered the gold standard postural sway assessment method in laboratory conditions, but their lack of portability and requirement of high-level expertise limit their widespread usage. This study evaluates the reliability and validity of a novel Balance Mat (BM) device, a low-cost portable alternative that uses optical fibre technology. The research includes two studies: a robot study and a human study. In the robot study, a UR10 robotic arm was used to obtain controlled sway patterns to assess the reliability and sensitivity of the BM. In the human study, 51 healthy young participants performed balance tasks on the BM in combination with an FP to evaluate the BM's validity. Sway metrics such as sway mean, sway absolute mean, sway root mean square (RMS), sway path, sway range, and sway velocity were calculated from both BM and FP and compared. Reliability was evaluated using the intra-class correlation coefficient (ICC), where values greater than 0.9 were considered excellent and values between 0.75 and 0.9 were considered good. Results from the robot study demonstrated good to excellent ICC values in both single and double-leg stances. The human study showed moderate to strong correlations for sway path and range. Using Bland-Altman plots for agreement analysis revealed proportional bias between the BM and the FP where the BM overestimated sway metrics compared to the FP. Calibration was used to improve the agreement between the devices. The device demonstrated consistent sway measurement across varied stance conditions, establishing both reliability and validity following appropriate calibration. 

**Abstract (ZH)**: 姿势摇摆评估对于检测平衡问题和识别跌倒风险人群很重要。实验室条件下，力plate (FP) 是姿势摇摆评估的金标准方法，但由于其缺乏便携性和对高技术水平的要求限制了其广泛应用。本研究评估了一种新型平衡垫(BM)设备的可靠性和有效性，这是一种低成本便携式替代品，使用了光纤技术。研究包括两个部分：机器人研究和人类研究。在机器人研究中，使用UR10机器人臂获得受控的摇摆模式，以评估BM的可靠性和敏感性。在人类研究中，51名健康的年轻参与者在BM和FP组合下进行平衡任务，以评估BM的有效性。计算了来自BM和FP的摇摆指标包括摇摆均值、绝对均值、均方根(RMS)、摇摆路径、摇摆范围和摇摆速度，并进行了比较。可靠性的评估使用了内氏相关系数(ICC)，其中大于0.9的值被认为是优秀的，介于0.75和0.9之间的值被认为是良好的。机器人研究的结果显示，在单腿和双腿站立状态下，BM的ICC值表现出良好到优秀的水平。人类研究结果显示，BM和FP在摇摆路径和范围上的相关性为中等到较强。通过Bland-Altman图进行一致性分析，显示出BM相对于FP在摇摆指标上存在比例偏移，在适当校准后，该设备在各种站立条件下表现出一致的摇摆测量结果，建立了其可靠性和有效性。 

---
