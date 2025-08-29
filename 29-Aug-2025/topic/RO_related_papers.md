# Rapid Mismatch Estimation via Neural Network Informed Variational Inference 

**Title (ZH)**: 基于神经网络引导的变分推断的快速不匹配估计 

**Authors**: Mateusz Jaszczuk, Nadia Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2508.21007)  

**Abstract**: With robots increasingly operating in human-centric environments, ensuring soft and safe physical interactions, whether with humans, surroundings, or other machines, is essential. While compliant hardware can facilitate such interactions, this work focuses on impedance controllers that allow torque-controlled robots to safely and passively respond to contact while accurately executing tasks. From inverse dynamics to quadratic programming-based controllers, the effectiveness of these methods relies on accurate dynamics models of the robot and the object it manipulates. Any model mismatch results in task failures and unsafe behaviors. Thus, we introduce Rapid Mismatch Estimation (RME), an adaptive, controller-agnostic, probabilistic framework that estimates end-effector dynamics mismatches online, without relying on external force-torque sensors. From the robot's proprioceptive feedback, a Neural Network Model Mismatch Estimator generates a prior for a Variational Inference solver, which rapidly converges to the unknown parameters while quantifying uncertainty. With a real 7-DoF manipulator driven by a state-of-the-art passive impedance controller, RME adapts to sudden changes in mass and center of mass at the end-effector in $\sim400$ ms, in static and dynamic settings. We demonstrate RME in a collaborative scenario where a human attaches an unknown basket to the robot's end-effector and dynamically adds/removes heavy items, showcasing fast and safe adaptation to changing dynamics during physical interaction without any external sensory system. 

**Abstract (ZH)**: 随着机器人越来越多地在以人类为中心的环境中操作，确保与其进行软而安全的物理交互（无论是与人类、环境还是其他机器的交互）至关重要。虽然顺应性硬件可以促进这种交互，但本文的重点在于允许扭矩控制机器人安全且被动地响应接触的同时准确执行任务的阻抗控制器。从逆动力学到基于二次规划的控制器，这些方法的有效性依赖于对机器人及其操作对象的精确动力学模型。任何模型不匹配都会导致任务失败和不安全的行为。因此，我们提出了快速不匹配估计（RME）方法，这是一种自适应的、控制器无关的概率框架，能够在不依赖外部力-扭矩传感器的情况下在线估计末端执行器动力学不匹配。从机器人的本体感受反馈出发，神经网络模型不匹配估计器生成变分推断求解器的先验信息，该解算器能够快速收敛到未知参数的同时量化不确定性。我们使用一个由先进被动阻抗控制器驱动的真实7自由度 manipulator，RME 在大约400毫秒内适应末端执行器处突然的质量和质心变化，无论是静态还是动态情况下均能实现这一点。我们在一个协作场景中展示了 RME，其中人类将一个未知篮子附加到机器人的末端执行器上，并动态添加/移除重物，证明了在物理交互过程中能够快速且安全地适应不断变化的动力学，无需任何外部感知系统。 

---
# UltraTac: Integrated Ultrasound-Augmented Visuotactile Sensor for Enhanced Robotic Perception 

**Title (ZH)**: UltraTac：集成超声增强的视触觉传感器以提高机器人感知能力 

**Authors**: Junhao Gong, Kit-Wa Sou, Shoujie Li, Changqing Guo, Yan Huang, Chuqiao Lyu, Ziwu Song, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2508.20982)  

**Abstract**: Visuotactile sensors provide high-resolution tactile information but are incapable of perceiving the material features of objects. We present UltraTac, an integrated sensor that combines visuotactile imaging with ultrasound sensing through a coaxial optoacoustic architecture. The design shares structural components and achieves consistent sensing regions for both modalities. Additionally, we incorporate acoustic matching into the traditional visuotactile sensor structure, enabling integration of the ultrasound sensing modality without compromising visuotactile performance. Through tactile feedback, we dynamically adjust the operating state of the ultrasound module to achieve flexible functional coordination. Systematic experiments demonstrate three key capabilities: proximity sensing in the 3-8 cm range ($R^2=0.90$), material classification (average accuracy: 99.20%), and texture-material dual-mode object recognition achieving 92.11% accuracy on a 15-class task. Finally, we integrate the sensor into a robotic manipulation system to concurrently detect container surface patterns and internal content, which verifies its potential for advanced human-machine interaction and precise robotic manipulation. 

**Abstract (ZH)**: 基于共轴光学声学架构的综合触觉-视觉传感器UltraTac：接近感知、材料分类与纹理-材料双重模式物体识别 

---
# Scaling Fabric-Based Piezoresistive Sensor Arrays for Whole-Body Tactile Sensing 

**Title (ZH)**: 基于织物的压阻式传感器阵列放大技术及其在全身触觉感知中的应用 

**Authors**: Curtis C. Johnson, Daniel Webb, David Hill, Marc D. Killpack  

**Link**: [PDF](https://arxiv.org/pdf/2508.20959)  

**Abstract**: Scaling tactile sensing for robust whole-body manipulation is a significant challenge, often limited by wiring complexity, data throughput, and system reliability. This paper presents a complete architecture designed to overcome these barriers. Our approach pairs open-source, fabric-based sensors with custom readout electronics that reduce signal crosstalk to less than 3.3% through hardware-based mitigation. Critically, we introduce a novel, daisy-chained SPI bus topology that avoids the practical limitations of common wireless protocols and the prohibitive wiring complexity of USB hub-based systems. This architecture streams synchronized data from over 8,000 taxels across 1 square meter of sensing area at update rates exceeding 50 FPS, confirming its suitability for real-time control. We validate the system's efficacy in a whole-body grasping task where, without feedback, the robot's open-loop trajectory results in an uncontrolled application of force that slowly crushes a deformable cardboard box. With real-time tactile feedback, the robot transforms this motion into a gentle, stable grasp, successfully manipulating the object without causing structural damage. This work provides a robust and well-characterized platform to enable future research in advanced whole-body control and physical human-robot interaction. 

**Abstract (ZH)**: 扩展触觉传感以实现稳健的全身操作是一个重大挑战，常受到线缆复杂性、数据吞吐量和系统可靠性的限制。本文提出了一种完整的架构以克服这些障碍。我们的方法将开源的织物基传感器与定制的读出电子设备配对，通过硬件基础的缓解措施将信号串扰降至少于3.3%。关键的是，我们引入了一种新型的级联SPI总线拓扑结构，避免了常见无线协议的实用性限制，并克服了基于USB集线器系统的繁琐线缆复杂性。该架构在超过一平方米的传感区域内，以超过50 FPS的更新率同步传输来自超过8,000个触点的数据，证实其适用于实时控制。我们在一项全身抓取任务中验证了该系统的有效性：在没有反馈的情况下，机器人的开环轨迹会导致对可变形纸板箱的无控施力，逐渐将其压扁；而在实时触觉反馈下，机器人将此运动转化为一种柔和、稳定的抓取，成功操纵物体而不会造成结构性损坏。该工作提供了一个稳健且性能良好的平台，以促进先进全身控制和物理人机交互的未来研究。 

---
# Deep Fuzzy Optimization for Batch-Size and Nearest Neighbors in Optimal Robot Motion Planning 

**Title (ZH)**: 基于模糊优化的批量大小和最近邻在最优机器人运动规划中的应用 

**Authors**: Liding Zhang, Qiyang Zong, Yu Zhang, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2508.20884)  

**Abstract**: Efficient motion planning algorithms are essential in robotics. Optimizing essential parameters, such as batch size and nearest neighbor selection in sampling-based methods, can enhance performance in the planning process. However, existing approaches often lack environmental adaptability. Inspired by the method of the deep fuzzy neural networks, this work introduces Learning-based Informed Trees (LIT*), a sampling-based deep fuzzy learning-based planner that dynamically adjusts batch size and nearest neighbor parameters to obstacle distributions in the configuration spaces. By encoding both global and local ratios via valid and invalid states, LIT* differentiates between obstacle-sparse and obstacle-dense regions, leading to lower-cost paths and reduced computation time. Experimental results in high-dimensional spaces demonstrate that LIT* achieves faster convergence and improved solution quality. It outperforms state-of-the-art single-query, sampling-based planners in environments ranging from R^8 to R^14 and is successfully validated on a dual-arm robot manipulation task. A video showcasing our experimental results is available at: this https URL 

**Abstract (ZH)**: 基于学习的知情树（LIT*）：一种用于配置空间中动态调整batch大小和最近邻参数的采样基于深度模糊学习规划算法 

---
# Model-Free Hovering and Source Seeking via Extremum Seeking Control: Experimental Demonstration 

**Title (ZH)**: 模型无关的悬浮与源搜索通过极值搜索控制：实验演示 

**Authors**: Ahmed A. Elgohary, Rohan Palanikumar, Sameh A. Eisa  

**Link**: [PDF](https://arxiv.org/pdf/2508.20836)  

**Abstract**: In a recent effort, we successfully proposed a categorically novel approach to mimic the phenomenoa of hovering and source seeking by flapping insects and hummingbirds using a new extremum seeking control (ESC) approach. Said ESC approach was shown capable of characterizing the physics of hovering and source seeking by flapping systems, providing at the same time uniquely novel opportunity for a model-free, real-time biomimicry control design. In this paper, we experimentally test and verify, for the first time in the literature, the potential of ESC in flapping robots to achieve model-free, real-time controlled hovering and source seeking. The results of this paper, while being restricted to 1D, confirm the premise of introducing ESC as a natural control method and biomimicry mechanism to the field of flapping flight and robotics. 

**Abstract (ZH)**: 一种新的 extremum seeking 控制方法在仿生扑翼飞行机器人悬停和源寻求中的实验验证与理论确认 

---
# A Soft Fabric-Based Thermal Haptic Device for VR and Teleoperation 

**Title (ZH)**: 基于软织物的热触觉装置：适用于VR和远程操作 

**Authors**: Rui Chen, Domenico Chiaradia, Antonio Frisoli, Daniele Leonardis  

**Link**: [PDF](https://arxiv.org/pdf/2508.20831)  

**Abstract**: This paper presents a novel fabric-based thermal-haptic interface for virtual reality and teleoperation. It integrates pneumatic actuation and conductive fabric with an innovative ultra-lightweight design, achieving only 2~g for each finger unit. By embedding heating elements within textile pneumatic chambers, the system delivers modulated pressure and thermal stimuli to fingerpads through a fully soft, wearable interface.
Comprehensive characterization demonstrates rapid thermal modulation with heating rates up to 3$^{\circ}$C/s, enabling dynamic thermal feedback for virtual or teleoperation interactions. The pneumatic subsystem generates forces up to 8.93~N at 50~kPa, while optimization of fingerpad-actuator clearance enhances cooling efficiency with minimal force reduction. Experimental validation conducted with two different user studies shows high temperature identification accuracy (0.98 overall) across three thermal levels, and significant manipulation improvements in a virtual pick-and-place tasks. Results show enhanced success rates (88.5\% to 96.4\%, p = 0.029) and improved force control precision (p = 0.013) when haptic feedback is enabled, validating the effectiveness of the integrated thermal-haptic approach for advanced human-machine interaction applications. 

**Abstract (ZH)**: 基于织物的新型热触觉界面：适用于虚拟现实和远程操作的新设计 

---
# SimShear: Sim-to-Real Shear-based Tactile Servoing 

**Title (ZH)**: SimShear: 基于剪切的模拟到现实的触觉伺服控制 

**Authors**: Kipp McAdam Freud, Yijiong Lin, Nathan F. Lepora  

**Link**: [PDF](https://arxiv.org/pdf/2508.20561)  

**Abstract**: We present SimShear, a sim-to-real pipeline for tactile control that enables the use of shear information without explicitly modeling shear dynamics in simulation. Shear, arising from lateral movements across contact surfaces, is critical for tasks involving dynamic object interactions but remains challenging to simulate. To address this, we introduce shPix2pix, a shear-conditioned U-Net GAN that transforms simulated tactile images absent of shear, together with a vector encoding shear information, into realistic equivalents with shear deformations. This method outperforms baseline pix2pix approaches in simulating tactile images and in pose/shear prediction. We apply SimShear to two control tasks using a pair of low-cost desktop robotic arms equipped with a vision-based tactile sensor: (i) a tactile tracking task, where a follower arm tracks a surface moved by a leader arm, and (ii) a collaborative co-lifting task, where both arms jointly hold an object while the leader follows a prescribed trajectory. Our method maintains contact errors within 1 to 2 mm across varied trajectories where shear sensing is essential, validating the feasibility of sim-to-real shear modeling with rigid-body simulators and opening new directions for simulation in tactile robotics. 

**Abstract (ZH)**: SimShear：一种无需显式建模摩擦动力学的摩擦信息仿真到现实的管道 

---
# Learning Fast, Tool aware Collision Avoidance for Collaborative Robots 

**Title (ZH)**: 学习快速的工具感知碰撞避免方法——协作机器人应用 

**Authors**: Joonho Lee, Yunho Kim, Seokjoon Kim, Quan Nguyen, Youngjin Heo  

**Link**: [PDF](https://arxiv.org/pdf/2508.20457)  

**Abstract**: Ensuring safe and efficient operation of collaborative robots in human environments is challenging, especially in dynamic settings where both obstacle motion and tasks change over time. Current robot controllers typically assume full visibility and fixed tools, which can lead to collisions or overly conservative behavior. In our work, we introduce a tool-aware collision avoidance system that adjusts in real time to different tool sizes and modes of tool-environment interaction. Using a learned perception model, our system filters out robot and tool components from the point cloud, reasons about occluded area, and predicts collision under partial observability. We then use a control policy trained via constrained reinforcement learning to produce smooth avoidance maneuvers in under 10 milliseconds. In simulated and real-world tests, our approach outperforms traditional approaches (APF, MPPI) in dynamic environments, while maintaining sub-millimeter accuracy. Moreover, our system operates with approximately 60% lower computational cost compared to a state-of-the-art GPU-based planner. Our approach provides modular, efficient, and effective collision avoidance for robots operating in dynamic environments. We integrate our method into a collaborative robot application and demonstrate its practical use for safe and responsive operation. 

**Abstract (ZH)**: 确保协作机器人在人类环境中的安全高效运行具有挑战性，特别是在动态环境中，障碍物和任务会随时间发生变化。当前的机器人控制器通常假设完全可见性和固定工具，这可能导致碰撞或过于保守的行为。在我们的工作中，我们引入了一种工具感知的碰撞避免系统，该系统能够实时调整不同的工具尺寸和工具-环境互动模式。借助学习到的感知模型，我们的系统过滤掉机器人和工具组件，推理出被遮挡的区域，并在部分可观测性下预测碰撞。然后，我们使用通过约束强化学习训练的控制策略，在不到10毫秒的时间内生成平滑的避免动作。在模拟和现实世界的测试中，我们的方法在动态环境中比传统方法（如APF、MPPI）表现出更优的性能，同时保持亚毫米级的精度。此外，与最先进的基于GPU的规划器相比，我们的系统计算成本大约降低了60%。我们的方法为动态环境中的机器人提供了模块化、高效和有效的碰撞避免。我们将该方法集成到协作机器人应用中，并展示了其在安全和响应性操作中的实际应用。 

---
# Regulation-Aware Game-Theoretic Motion Planning for Autonomous Racing 

**Title (ZH)**: 基于博弈论的自主赛车运动规划，考虑监管要求 

**Authors**: Francesco Prignoli, Francesco Borrelli, Paolo Falcone, Mark Pustilnik  

**Link**: [PDF](https://arxiv.org/pdf/2508.20203)  

**Abstract**: This paper presents a regulation-aware motion planning framework for autonomous racing scenarios. Each agent solves a Regulation-Compliant Model Predictive Control problem, where racing rules - such as right-of-way and collision avoidance responsibilities - are encoded using Mixed Logical Dynamical constraints. We formalize the interaction between vehicles as a Generalized Nash Equilibrium Problem (GNEP) and approximate its solution using an Iterative Best Response scheme. Building on this, we introduce the Regulation-Aware Game-Theoretic Planner (RA-GTP), in which the attacker reasons over the defender's regulation-constrained behavior. This game-theoretic layer enables the generation of overtaking strategies that are both safe and non-conservative. Simulation results demonstrate that the RA-GTP outperforms baseline methods that assume non-interacting or rule-agnostic opponent models, leading to more effective maneuvers while consistently maintaining compliance with racing regulations. 

**Abstract (ZH)**: 基于规则意识的自主赛车运动规划框架 

---
# Train-Once Plan-Anywhere Kinodynamic Motion Planning via Diffusion Trees 

**Title (ZH)**: 一次训练，随处规划：基于扩散树的kinodynamic运动规划 

**Authors**: Yaniv Hassidof, Tom Jurgenson, Kiril Solovey  

**Link**: [PDF](https://arxiv.org/pdf/2508.21001)  

**Abstract**: Kinodynamic motion planning is concerned with computing collision-free trajectories while abiding by the robot's dynamic constraints. This critical problem is often tackled using sampling-based planners (SBPs) that explore the robot's high-dimensional state space by constructing a search tree via action propagations. Although SBPs can offer global guarantees on completeness and solution quality, their performance is often hindered by slow exploration due to uninformed action sampling. Learning-based approaches can yield significantly faster runtimes, yet they fail to generalize to out-of-distribution (OOD) scenarios and lack critical guarantees, e.g., safety, thus limiting their deployment on physical robots. We present Diffusion Tree (DiTree): a \emph{provably-generalizable} framework leveraging diffusion policies (DPs) as informed samplers to efficiently guide state-space search within SBPs. DiTree combines DP's ability to model complex distributions of expert trajectories, conditioned on local observations, with the completeness of SBPs to yield \emph{provably-safe} solutions within a few action propagation iterations for complex dynamical systems. We demonstrate DiTree's power with an implementation combining the popular RRT planner with a DP action sampler trained on a \emph{single environment}. In comprehensive evaluations on OOD scenarios, % DiTree has comparable runtimes to a standalone DP (3x faster than classical SBPs), while improving the average success rate over DP and SBPs. DiTree is on average 3x faster than classical SBPs, and outperforms all other approaches by achieving roughly 30\% higher success rate. Project webpage: this https URL. 

**Abstract (ZH)**: 基于扩散策略的扩散树（DiTree）：一种可证明泛化的动态规划框架 

---
