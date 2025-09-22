# Modeling Elastic-Body Dynamics of Fish Swimming Using a Variational Framework 

**Title (ZH)**: 使用变分框架建模鱼类游泳的弹性体动力学 

**Authors**: Zhiheng Chen, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16145)  

**Abstract**: Fish-inspired aquatic robots are gaining increasing attention in research communities due to their high swimming speeds and efficient propulsion enabled by flexible bodies that generate undulatory motions. To support the design optimizations and control of such systems, accurate, interpretable, and computationally tractable modeling of the underlying swimming dynamics is indispensable. In this letter, we present a full-body dynamics model for fish swimming, rigorously derived from Hamilton's principle. The model captures the continuously distributed elasticity of a deformable fish body undergoing large deformations and incorporates fluid-structure coupling effects, enabling self-propelled motion without prescribing kinematics. A preliminary parameter study explores the influence of actuation frequency and body stiffness on swimming speed and cost of transport (COT). Simulation results indicate that swimming speed and energy efficiency exhibit opposing trends with tail-beat frequency and that both body stiffness and body length have distinct optimal values. These findings provide insights into biological swimming mechanisms and inform the design of high-performance soft robotic swimmers. 

**Abstract (ZH)**: 鱼启发的水下机器人的全身体动力学模型：从哈密尔顿原理严格推导，探究驱动频率和身体刚度对游泳速度和运能成本的影响 

---
# Efficient Detection of Objects Near a Robot Manipulator via Miniature Time-of-Flight Sensors 

**Title (ZH)**: 基于微型飞行时间传感器的机器人 manipulator 近处物体高效检测方法 

**Authors**: Carter Sifferman, Mohit Gupta, Michael Gleicher  

**Link**: [PDF](https://arxiv.org/pdf/2509.16122)  

**Abstract**: We provide a method for detecting and localizing objects near a robot arm using arm-mounted miniature time-of-flight sensors. A key challenge when using arm-mounted sensors is differentiating between the robot itself and external objects in sensor measurements. To address this challenge, we propose a computationally lightweight method which utilizes the raw time-of-flight information captured by many off-the-shelf, low-resolution time-of-flight sensor. We build an empirical model of expected sensor measurements in the presence of the robot alone, and use this model at runtime to detect objects in proximity to the robot. In addition to avoiding robot self-detections in common sensor configurations, the proposed method enables extra flexibility in sensor placement, unlocking configurations which achieve more efficient coverage of a radius around the robot arm. Our method can detect small objects near the arm and localize the position of objects along the length of a robot link to reasonable precision. We evaluate the performance of the method with respect to object type, location, and ambient light level, and identify limiting factors on performance inherent in the measurement principle. The proposed method has potential applications in collision avoidance and in facilitating safe human-robot interaction. 

**Abstract (ZH)**: 使用安装在机器人臂上的微型飞行时间传感器检测和定位附近物体的方法 

---
# Real-Time Planning and Control with a Vortex Particle Model for Fixed-Wing UAVs in Unsteady Flows 

**Title (ZH)**: 基于旋涡粒子模型的固定翼无人机在非定常流场中的实时规划与控制 

**Authors**: Ashwin Gupta, Kevin Wolfe, Gino Perrotta, Joseph Moore  

**Link**: [PDF](https://arxiv.org/pdf/2509.16079)  

**Abstract**: Unsteady aerodynamic effects can have a profound impact on aerial vehicle flight performance, especially during agile maneuvers and in complex aerodynamic environments. In this paper, we present a real-time planning and control approach capable of reasoning about unsteady aerodynamics. Our approach relies on a lightweight vortex particle model, parallelized to allow GPU acceleration, and a sampling-based policy optimization strategy capable of leveraging the vortex particle model for predictive reasoning. We demonstrate, through both simulation and hardware experiments, that by replanning with our unsteady aerodynamics model, we can improve the performance of aggressive post-stall maneuvers in the presence of unsteady environmental flow disturbances. 

**Abstract (ZH)**: 不稳态气动力效应对空中车辆飞行性能有着深远影响，特别是在快速机动和复杂气动力环境中。本文提出了一种实时规划与控制方法，能够对不稳态气动力进行推理。该方法依赖于轻量级涡旋粒子模型，该模型已并行化以允许GPU加速，并采用基于采样的策略优化方法，能够利用涡旋粒子模型进行预测推理。通过仿真和硬件实验，我们证明了通过利用不稳态气动力模型进行重新规划，可以在不稳态环境气流扰动下提高飞机激进后失速机动的性能。 

---
# Learning Safety for Obstacle Avoidance via Control Barrier Functions 

**Title (ZH)**: 基于控制屏障函数的避障安全学习 

**Authors**: Shuo Liu, Zhe Huang, Calin A. Belta  

**Link**: [PDF](https://arxiv.org/pdf/2509.16037)  

**Abstract**: Obstacle avoidance is central to safe navigation, especially for robots with arbitrary and nonconvex geometries operating in cluttered environments. Existing Control Barrier Function (CBF) approaches often rely on analytic clearance computations, which are infeasible for complex geometries, or on polytopic approximations, which become intractable when robot configurations are unknown. To address these limitations, this paper trains a residual neural network on a large dataset of robot-obstacle configurations to enable fast and tractable clearance prediction, even at unseen configurations. The predicted clearance defines the radius of a Local Safety Ball (LSB), which ensures continuous-time collision-free navigation. The LSB boundary is encoded as a Discrete-Time High-Order CBF (DHOCBF), whose constraints are incorporated into a nonlinear optimization framework. To improve feasibility, a novel relaxation technique is applied. The resulting framework ensure that the robot's rigid-body motion between consecutive time steps remains collision-free, effectively bridging discrete-time control and continuous-time safety. We show that the proposed method handles arbitrary, including nonconvex, robot geometries and generates collision-free, dynamically feasible trajectories in cluttered environments. Experiments demonstrate millisecond-level solve times and high prediction accuracy, highlighting both safety and efficiency beyond existing CBF-based methods. 

**Abstract (ZH)**: 障碍物避免对于安全导航至关重要，尤其是对于在复杂环境中有任意和非凸几何结构的机器人。现有的控制障碍函数（CBF）方法往往依赖于解析的清除计算，这在复杂几何结构情况下是不可行的，或者依赖于多面体近似，当机器人姿态未知时会变得不可行。为了解决这些局限，本文在一个大规模的机器人-障碍物配置数据集上训练了一个残差神经网络，以实现即使是未见过的配置也能快速和有效地进行清除预测。预测的清除定义了一个局部安全球（LSB）的半径，该球确保连续时间内的无碰撞导航。LSB的边界被编码为离散时间高阶控制障碍函数（DHOCBF），其约束被整合到非线性优化框架中。为了提高可行性，提出了一种新的松弛技术。由此形成的框架确保机器人在连续时间步之间的刚体运动始终保持无碰撞，有效地在离散时间和连续时间安全之间架起桥梁。我们展示了所提出的方法能够处理任意形状，包括非凸形状的机器人几何结构，并在复杂环境中生成无碰撞且动力学可行的轨迹。实验表明，该方法具有毫秒级的求解时间和高预测精度，突显了其在安全性和效率方面超越现有CBF基方法的优势。 

---
# A Matter of Height: The Impact of a Robotic Object on Human Compliance 

**Title (ZH)**: 高度问题：机器人物体对人类遵从性的影响 

**Authors**: Michael Faber, Andrey Grishko, Julian Waksberg, David Pardo, Tomer Leivy, Yuval Hazan, Emanuel Talmansky, Benny Megidish, Hadas Erel  

**Link**: [PDF](https://arxiv.org/pdf/2509.16032)  

**Abstract**: Robots come in various forms and have different characteristics that may shape the interaction with them. In human-human interactions, height is a characteristic that shapes human dynamics, with taller people typically perceived as more persuasive. In this work, we aspired to evaluate if the same impact replicates in a human-robot interaction and specifically with a highly non-humanoid robotic object. The robot was designed with modules that could be easily added or removed, allowing us to change its height without altering other design features. To test the impact of the robot's height, we evaluated participants' compliance with its request to volunteer to perform a tedious task. In the experiment, participants performed a cognitive task on a computer, which was framed as the main experiment. When done, they were informed that the experiment was completed. While waiting to receive their credits, the robotic object, designed as a mobile robotic service table, entered the room, carrying a tablet that invited participants to complete a 300-question questionnaire voluntarily. We compared participants' compliance in two conditions: A Short robot composed of two modules and 95cm in height and a Tall robot consisting of three modules and 132cm in height. Our findings revealed higher compliance with the Short robot's request, demonstrating an opposite pattern to human dynamics. We conclude that while height has a substantial social impact on human-robot interactions, it follows a unique pattern of influence. Our findings suggest that designers cannot simply adopt and implement elements from human social dynamics to robots without testing them first. 

**Abstract (ZH)**: 机器人呈现多种形态，具有不同的特性，这些特性可能会影响与之的互动。在人与人之间的互动中，身高是一种特性，会影响人际关系动态，通常更高的个体被认为更具说服力。在这项工作中，我们旨在评估这种情况在人与机器人互动中是否同样有效，特别是对于一个高度非人形的机器人对象。机器人设计有可轻松添加或移除的模块，使我们能够改变其高度而不改变其他设计特征。为测试机器人高度的影响，我们评估了参与者对其请求他们自愿完成一项枯燥任务的遵从性。在实验中，参与者在计算机上完成了一个认知任务，这被视为主要实验。完成后，他们被告知实验已经结束。在等待领奖金时，被设计成移动服务机器人的机器人对象，携带平板电脑进入房间，邀请参与者自愿完成一份包含300个问题的问卷。我们比较了在两种条件下参与者的遵从性：一种是高95厘米由两个模块组成的短机器人，另一种是高132厘米由三个模块组成的高机器人。我们的研究发现，参与者对短机器人的请求遵从性更高，显示出与人类互动动态相反的模式。我们得出结论，尽管身高在人机互动中对社会影响很大，但其影响模式是独特的。我们的研究结果表明，设计师不能简单地借鉴和应用人类社会动态的元素到机器人中，而无需首先进行测试。 

---
# An MPC framework for efficient navigation of mobile robots in cluttered environments 

**Title (ZH)**: 基于模型预测控制的移动机器人在复杂环境中的高效导航框架 

**Authors**: Johannes Köhler, Daniel Zhang, Raffaele Soloperto, Andrea Carron, Melanie Zeilinger  

**Link**: [PDF](https://arxiv.org/pdf/2509.15917)  

**Abstract**: We present a model predictive control (MPC) framework for efficient navigation of mobile robots in cluttered environments. The proposed approach integrates a finite-segment shortest path planner into the finite-horizon trajectory optimization of the MPC. This formulation ensures convergence to dynamically selected targets and guarantees collision avoidance, even under general nonlinear dynamics and cluttered environments. The approach is validated through hardware experiments on a small ground robot, where a human operator dynamically assigns target locations. The robot successfully navigated through complex environments and reached new targets within 2-3 seconds. 

**Abstract (ZH)**: 我们提出了一种模型预测控制（MPC）框架，用于_cluttered环境中小型地面机器人的高效导航。该方法将有限段最短路径规划器集成到MPC的有限时间轨迹优化中，确保在一般非线性动力学和_cluttered环境中动态选择目标并实现碰撞避免。该方法通过硬件实验得到了验证，其中人工操作员动态分配目标位置，机器人能在2-3秒内成功导航通过复杂环境并到达新目标。 

---
# Improving Robotic Manipulation with Efficient Geometry-Aware Vision Encoder 

**Title (ZH)**: 提高基于高效几何aware视觉编码器的机器人操作性能 

**Authors**: An Dinh Vuong, Minh Nhat Vu, Ian Reid  

**Link**: [PDF](https://arxiv.org/pdf/2509.15880)  

**Abstract**: Existing RGB-based imitation learning approaches typically employ traditional vision encoders such as ResNet or ViT, which lack explicit 3D reasoning capabilities. Recent geometry-grounded vision models, such as VGGT~\cite{wang2025vggt}, provide robust spatial understanding and are promising candidates to address this limitation. This work investigates the integration of geometry-aware visual representations into robotic manipulation. Our results suggest that incorporating the geometry-aware vision encoder into imitation learning frameworks, including ACT and DP, yields up to 6.5% improvement over standard vision encoders in success rate across single- and bi-manual manipulation tasks in both simulation and real-world settings. Despite these benefits, most geometry-grounded models require high computational cost, limiting their deployment in practical robotic systems. To address this challenge, we propose eVGGT, an efficient geometry-aware encoder distilled from VGGT. eVGGT is nearly 9 times faster and 5 times smaller than VGGT, while preserving strong 3D reasoning capabilities. Code and pretrained models will be released to facilitate further research in geometry-aware robotics. 

**Abstract (ZH)**: 基于几何信息的视觉表示在机器人操作中的应用研究 

---
# High-Bandwidth Tactile-Reactive Control for Grasp Adjustment 

**Title (ZH)**: 高频触觉反应控制以调整抓取 

**Authors**: Yonghyeon Lee, Tzu-Yuan Lin, Alexander Alexiev, Sangbae Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.15876)  

**Abstract**: Vision-only grasping systems are fundamentally constrained by calibration errors, sensor noise, and grasp pose prediction inaccuracies, leading to unavoidable contact uncertainty in the final stage of grasping. High-bandwidth tactile feedback, when paired with a well-designed tactile-reactive controller, can significantly improve robustness in the presence of perception errors. This paper contributes to controller design by proposing a purely tactile-feedback grasp-adjustment algorithm. The proposed controller requires neither prior knowledge of the object's geometry nor an accurate grasp pose, and is capable of refining a grasp even when starting from a crude, imprecise initial configuration and uncertain contact points. Through simulation studies and real-world experiments on a 15-DoF arm-hand system (featuring an 8-DoF hand) equipped with fingertip tactile sensors operating at 200 Hz, we demonstrate that our tactile-reactive grasping framework effectively improves grasp stability. 

**Abstract (ZH)**: 基于触觉反馈的手抓调整算法能够显著提高抓取的 robustness，特别是在感知错误存在的情况下。本文提出了一种纯触觉反馈的抓取调整算法，无需对象几何形状的先验知识或精确的抓取姿态，即使从粗糙且不精确的初始配置和不确定的接触点开始，也能改进抓取。通过在配备8自由度手并带有指尖触觉传感器（采样率200 Hz）的15自由度臂手系统上的模拟实验和现实世界实验，证明了所提出的触觉反应式抓取框架有效提高了抓取稳定性。 

---
# Coordinated Multi-Drone Last-mile Delivery: Learning Strategies for Energy-aware and Timely Operations 

**Title (ZH)**: 协调多无人机最后一英里交付：面向能量感知和及时运营的学习策略 

**Authors**: Chuhao Qin, Arun Narayanan, Evangelos Pournaras  

**Link**: [PDF](https://arxiv.org/pdf/2509.15830)  

**Abstract**: Drones have recently emerged as a faster, safer, and cost-efficient way for last-mile deliveries of parcels, particularly for urgent medical deliveries highlighted during the pandemic. This paper addresses a new challenge of multi-parcel delivery with a swarm of energy-aware drones, accounting for time-sensitive customer requirements. Each drone plans an optimal multi-parcel route within its battery-restricted flight range to minimize delivery delays and reduce energy consumption. The problem is tackled by decomposing it into three sub-problems: (1) optimizing depot locations and service areas using K-means clustering; (2) determining the optimal flight range for drones through reinforcement learning; and (3) planning and selecting multi-parcel delivery routes via a new optimized plan selection approach. To integrate these solutions and enhance long-term efficiency, we propose a novel algorithm leveraging actor-critic-based multi-agent deep reinforcement learning. Extensive experimentation using realistic delivery datasets demonstrate an exceptional performance of the proposed algorithm. We provide new insights into economic efficiency (minimize energy consumption), rapid operations (reduce delivery delays and overall execution time), and strategic guidance on depot deployment for practical logistics applications. 

**Abstract (ZH)**: 无人机在多包裹交付中的能效挑战与解决方案：基于强化学习的最优路径规划与多无人机集群部署 

---
# Miniature soft robot with magnetically reprogrammable surgical functions 

**Title (ZH)**: 磁性可重新编程外科功能的微型软机器人 

**Authors**: Chelsea Shan Xian Ng, Yu Xuan Yeoh, Nicholas Yong Wei Foo, Keerthana Radhakrishnan, Guo Zhan Lum  

**Link**: [PDF](https://arxiv.org/pdf/2509.15610)  

**Abstract**: Miniature robots are untethered actuators, which have significant potential to make existing minimally invasive surgery considerably safer and painless, and enable unprecedented treatments because they are much smaller and dexterous than existing surgical robots. Of the miniature robots, the magnetically actuated ones are the most functional and dexterous. However, existing magnetic miniature robots are currently impractical for surgery because they are either restricted to possessing at most two on-board functionalities or having limited five degrees-of-freedom (DOF) locomotion. Some of these actuators are also only operational under specialized environments where actuation from strong external magnets must be at very close proximity (< 4 cm away). Here we present a millimeter-scale soft robot where its magnetization profile can be reprogrammed upon command to perform five surgical functionalities: drug-dispensing, cutting through biological tissues (simulated with gelatin), gripping, storing (biological) samples and remote heating. By possessing full six-DOF motions, including the sixth-DOF rotation about its net magnetic moment, our soft robot can also roll and two-anchor crawl across challenging unstructured environments, which are impassable by its five-DOF counterparts. Because our actuating magnetic fields are relatively uniform and weak (at most 65 mT and 1.5 T/m), such fields can theoretically penetrate through biological tissues harmlessly and allow our soft robot to remain controllable within the depths of the human body. We envision that this work marks a major milestone for the advancement of soft actuators, and towards revolutionizing minimally invasive treatments with untethered miniature robots that have unprecedented functionalities. 

**Abstract (ZH)**: 毫米级软磁驱动机器人可在命令下重编程磁化轮廓，执行五种手术功能：药物释放、切割生物组织、抓取、存储生物样本和远程加热。通过具备六自由度运动，包括其净磁矩的第六自由度旋转，该软机器人还可以在结构复杂的环境中进行滚动和双锚点爬行，这是其五自由度同类无法实现的。由于我们的驱动磁场相对均匀且较弱（最大65 mT和1.5 T/m），这些磁场理论上可以无害地穿透生物组织，使软机器人能够在人体内部深处保持可控。我们展望这项工作是软驱动器发展的重要里程碑，并将推动以具有前所未有的功能的无绳微型机器人革新微创治疗方法。 

---
# ORB: Operating Room Bot, Automating Operating Room Logistics through Mobile Manipulation 

**Title (ZH)**: ORB：手术室机器人，通过移动操作自动化手术室物流 

**Authors**: Jinkai Qiu, Yungjun Kim, Gaurav Sethia, Tanmay Agarwal, Siddharth Ghodasara, Zackory Erickson, Jeffrey Ichnowski  

**Link**: [PDF](https://arxiv.org/pdf/2509.15600)  

**Abstract**: Efficiently delivering items to an ongoing surgery in a hospital operating room can be a matter of life or death. In modern hospital settings, delivery robots have successfully transported bulk items between rooms and floors. However, automating item-level operating room logistics presents unique challenges in perception, efficiency, and maintaining sterility. We propose the Operating Room Bot (ORB), a robot framework to automate logistics tasks in hospital operating rooms (OR). ORB leverages a robust, hierarchical behavior tree (BT) architecture to integrate diverse functionalities of object recognition, scene interpretation, and GPU-accelerated motion planning. The contributions of this paper include: (1) a modular software architecture facilitating robust mobile manipulation through behavior trees; (2) a novel real-time object recognition pipeline integrating YOLOv7, Segment Anything Model 2 (SAM2), and Grounded DINO; (3) the adaptation of the cuRobo parallelized trajectory optimization framework to real-time, collision-free mobile manipulation; and (4) empirical validation demonstrating an 80% success rate in OR supply retrieval and a 96% success rate in restocking operations. These contributions establish ORB as a reliable and adaptable system for autonomous OR logistics. 

**Abstract (ZH)**: 高效地将物品送达医院手术室可以关乎生死。在现代医院环境中，送货机器人已成功实现房间间和楼层间的批量物品运输。然而，自动化手术室内的物品级物流面临独特的感知、效率和保持无菌性的挑战。我们提出手术室机器人（ORB），一种用于医院手术室（OR）内自动化物流任务的机器人框架。ORB 利用稳健的分层行为树（BT）架构整合了物体识别、场景解释和GPU加速运动规划等多种功能。本文的贡献包括：（1）一种模块化的软件架构，通过行为树实现稳健的移动操作；（2）一种结合YOLOv7、Segment Anything Model 2（SAM2）和Grounded DINO的新颖实时物体识别流水线；（3）将cuRobo并行轨迹优化框架适应于实时、无碰撞的移动操作；（4）实验证明ORB在OR物资检索中成功率达到80%，在补货操作中成功率达到96%。这些贡献确立了ORB作为可靠且适应性强的自主OR物流系统的地位。 

---
# Momentum-constrained Hybrid Heuristic Trajectory Optimization Framework with Residual-enhanced DRL for Visually Impaired Scenarios 

**Title (ZH)**: 基于动量约束的混合 heuristic 轨迹优化框架及残差增强的深度强化学习在视力障碍场景中的应用 

**Authors**: Yuting Zeng, Zhiwen Zheng, You Zhou, JiaLing Xiao, Yongbin Yu, Manping Fan, Bo Gong, Liyong Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.15582)  

**Abstract**: This paper proposes a momentum-constrained hybrid heuristic trajectory optimization framework (MHHTOF) tailored for assistive navigation in visually impaired scenarios, integrating trajectory sampling generation, optimization and evaluation with residual-enhanced deep reinforcement learning (DRL). In the first stage, heuristic trajectory sampling cluster (HTSC) is generated in the Frenet coordinate system using third-order interpolation with fifth-order polynomials and momentum-constrained trajectory optimization (MTO) constraints to ensure smoothness and feasibility. After first stage cost evaluation, the second stage leverages a residual-enhanced actor-critic network with LSTM-based temporal feature modeling to adaptively refine trajectory selection in the Cartesian coordinate system. A dual-stage cost modeling mechanism (DCMM) with weight transfer aligns semantic priorities across stages, supporting human-centered optimization. Experimental results demonstrate that the proposed LSTM-ResB-PPO achieves significantly faster convergence, attaining stable policy performance in approximately half the training iterations required by the PPO baseline, while simultaneously enhancing both reward outcomes and training stability. Compared to baseline method, the selected model reduces average cost and cost variance by 30.3% and 53.3%, and lowers ego and obstacle risks by over 77%. These findings validate the framework's effectiveness in enhancing robustness, safety, and real-time feasibility in complex assistive planning tasks. 

**Abstract (ZH)**: 基于动量约束的混合启发式轨迹优化框架（MHHTOF）在视障导航中的应用：结合残差增强的深度强化学习 

---
# Online Slip Detection and Friction Coefficient Estimation for Autonomous Racing 

**Title (ZH)**: 在线打滑检测与摩擦系数估计在自主赛车中的应用 

**Authors**: Christopher Oeltjen, Carson Sobolewski, Saleh Faghfoorian, Lorant Domokos, Giancarlo Vidal, Ivan Ruchkin  

**Link**: [PDF](https://arxiv.org/pdf/2509.15423)  

**Abstract**: Accurate knowledge of the tire-road friction coefficient (TRFC) is essential for vehicle safety, stability, and performance, especially in autonomous racing, where vehicles often operate at the friction limit. However, TRFC cannot be directly measured with standard sensors, and existing estimation methods either depend on vehicle or tire models with uncertain parameters or require large training datasets. In this paper, we present a lightweight approach for online slip detection and TRFC estimation. Our approach relies solely on IMU and LiDAR measurements and the control actions, without special dynamical or tire models, parameter identification, or training data. Slip events are detected in real time by comparing commanded and measured motions, and the TRFC is then estimated directly from observed accelerations under no-slip conditions. Experiments with a 1:10-scale autonomous racing car across different friction levels demonstrate that the proposed approach achieves accurate and consistent slip detections and friction coefficients, with results closely matching ground-truth measurements. These findings highlight the potential of our simple, deployable, and computationally efficient approach for real-time slip monitoring and friction coefficient estimation in autonomous driving. 

**Abstract (ZH)**: 基于IMU和LiDAR的在线侧滑检测与摩擦系数估算 

---
# Sym2Real: Symbolic Dynamics with Residual Learning for Data-Efficient Adaptive Control 

**Title (ZH)**: Sym2Real: 基于残差学习的符号动力学在数据高效自适应控制中的应用 

**Authors**: Easop Lee, Samuel A. Moore, Boyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.15412)  

**Abstract**: We present Sym2Real, a fully data-driven framework that provides a principled way to train low-level adaptive controllers in a highly data-efficient manner. Using only about 10 trajectories, we achieve robust control of both a quadrotor and a racecar in the real world, without expert knowledge or simulation tuning. Our approach achieves this data efficiency by bringing symbolic regression to real-world robotics while addressing key challenges that prevent its direct application, including noise sensitivity and model degradation that lead to unsafe control. Our key observation is that the underlying physics is often shared for a system regardless of internal or external changes. Hence, we strategically combine low-fidelity simulation data with targeted real-world residual learning. Through experimental validation on quadrotor and racecar platforms, we demonstrate consistent data-efficient adaptation across six out-of-distribution sim2sim scenarios and successful sim2real transfer across five real-world conditions. More information and videos can be found at at this http URL 

**Abstract (ZH)**: Sym2Real: 一个数据驱动的框架，以高效数据方式训练低级自适应控制器 

---
# All-Electric Heavy-Duty Robotic Manipulator: Actuator Configuration Optimization and Sensorless Control 

**Title (ZH)**: 全电驱动重型机器人 manipulator: 执行器配置优化与无传感器控制 

**Authors**: Mohammad Bahari, Amir Hossein Barjini, Pauli Mustalahti, Jouni Mattila  

**Link**: [PDF](https://arxiv.org/pdf/2509.15778)  

**Abstract**: This paper presents a unified framework that integrates modeling, optimization, and sensorless control of an all-electric heavy-duty robotic manipulator (HDRM) driven by electromechanical linear actuators (EMLAs). An EMLA model is formulated to capture motor electromechanics and direction-dependent transmission efficiencies, while a mathematical model of the HDRM, incorporating both kinematics and dynamics, is established to generate joint-space motion profiles for prescribed TCP trajectories. A safety-ensured trajectory generator, tailored to this model, maps Cartesian goals to joint space while enforcing joint-limit and velocity margins. Based on the resulting force and velocity demands, a multi-objective Non-dominated Sorting Genetic Algorithm II (NSGA-II) is employed to select the optimal EMLA configuration. To accelerate this optimization, a deep neural network, trained with EMLA parameters, is embedded in the optimization process to predict steady-state actuator efficiency from trajectory profiles. For the chosen EMLA design, a physics-informed Kriging surrogate, anchored to the analytic model and refined with experimental data, learns residuals of EMLA outputs to support force and velocity sensorless control. The actuator model is further embedded in a hierarchical virtual decomposition control (VDC) framework that outputs voltage commands. Experimental validation on a one-degree-of-freedom EMLA testbed confirms accurate trajectory tracking and effective sensorless control under varying loads. 

**Abstract (ZH)**: 一种集成了电动机械线性执行器(EMLA)驱动的全电重型机器人 manipulator (HDRM) 的建模、优化和无传感器控制的统一框架 

---
