# Dynamic Modeling and Efficient Data-Driven Optimal Control for Micro Autonomous Surface Vehicles 

**Title (ZH)**: 微自主水面 vehicle 的动力学建模与高效数据驱动最优控制 

**Authors**: Zhiheng Chen, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06882)  

**Abstract**: Micro Autonomous Surface Vehicles (MicroASVs) offer significant potential for operations in confined or shallow waters and swarm robotics applications. However, achieving precise and robust control at such small scales remains highly challenging, mainly due to the complexity of modeling nonlinear hydrodynamic forces and the increased sensitivity to self-motion effects and environmental disturbances, including waves and boundary effects in confined spaces. This paper presents a physics-driven dynamics model for an over-actuated MicroASV and introduces a data-driven optimal control framework that leverages a weak formulation-based online model learning method. Our approach continuously refines the physics-driven model in real time, enabling adaptive control that adjusts to changing system parameters. Simulation results demonstrate that the proposed method substantially enhances trajectory tracking accuracy and robustness, even under unknown payloads and external disturbances. These findings highlight the potential of data-driven online learning-based optimal control to improve MicroASV performance, paving the way for more reliable and precise autonomous surface vehicle operations. 

**Abstract (ZH)**: 基于物理的微自主水面车辆动力学模型及数据驱动最优控制框架研究 

---
# CRISP - Compliant ROS2 Controllers for Learning-Based Manipulation Policies and Teleoperation 

**Title (ZH)**: CRISP - 符合ROS2标准的基于学习的 manipulation 策略及遥操作控制器 

**Authors**: Daniel San José Pro, Oliver Hausdörfer, Ralf Römer, Maximilian Dösch, Martin Schuck, Angela P. Schöllig  

**Link**: [PDF](https://arxiv.org/pdf/2509.06819)  

**Abstract**: Learning-based controllers, such as diffusion policies and vision-language action models, often generate low-frequency or discontinuous robot state changes. Achieving smooth reference tracking requires a low-level controller that converts high-level targets commands into joint torques, enabling compliant behavior during contact interactions. We present CRISP, a lightweight C++ implementation of compliant Cartesian and joint-space controllers for the ROS2 control standard, designed for seamless integration with high-level learning-based policies as well as teleoperation. The controllers are compatible with any manipulator that exposes a joint-torque interface. Through our Python and Gymnasium interfaces, CRISP provides a unified pipeline for recording data from hardware and simulation and deploying high-level learning-based policies seamlessly, facilitating rapid experimentation. The system has been validated on hardware with the Franka Robotics FR3 and in simulation with the Kuka IIWA14 and Kinova Gen3. Designed for rapid integration, flexible deployment, and real-time performance, our implementation provides a unified pipeline for data collection and policy execution, lowering the barrier to applying learning-based methods on ROS2-compatible manipulators. Detailed documentation is available at the project website - this https URL. 

**Abstract (ZH)**: 基于学习的控制器，如扩散策略和视觉-语言动作模型，通常生成低频或不连续的机器人状态变化。实现平滑的参考跟踪需要一个低级控制器，将高层目标命令转换为关节扭矩，从而在接触交互中实现柔顺行为。我们介绍了CRISP，这是一种轻量级的C++实现，用于ROS2控制标准下的柔顺笛卡尔空间和关节空间控制器，旨在与高层学习基础策略以及远程操作无缝集成。该控制器兼容任何暴露关节扭矩接口的 manipulator。通过我们的Python和Gymnasium接口，CRISP提供了一个统一的数据采集和高效率学习策略部署管道，便于快速实验。该系统已在Franka Robotics FR3硬件上以及Kuka IIWA14和Kinova Gen3仿真环境中进行验证。为实现快速集成、灵活部署和实时性能，我们的实现提供了一个统一的数据收集和策略执行管道，降低了在ROS2兼容的 manipulator 上应用学习方法的门槛。详细文档可在项目网站获取 - 这 https URL。 

---
# Safe Robust Predictive Control-based Motion Planning of Automated Surface Vessels in Inland Waterways 

**Title (ZH)**: 基于安全鲁棒预测控制的内河水域自主水面船舶运动规划 

**Authors**: Sajad Ahmadi, Hossein Nejatbakhsh Esfahani, Javad Mohammadpour Velni  

**Link**: [PDF](https://arxiv.org/pdf/2509.06687)  

**Abstract**: Deploying self-navigating surface vessels in inland waterways offers a sustainable alternative to reduce road traffic congestion and emissions. However, navigating confined waterways presents unique challenges, including narrow channels, higher traffic density, and hydrodynamic disturbances. Existing methods for autonomous vessel navigation often lack the robustness or precision required for such environments. This paper presents a new motion planning approach for Automated Surface Vessels (ASVs) using Robust Model Predictive Control (RMPC) combined with Control Barrier Functions (CBFs). By incorporating channel borders and obstacles as safety constraints within the control design framework, the proposed method ensures both collision avoidance and robust navigation on complex waterways. Simulation results demonstrate the efficacy of the proposed method in safely guiding ASVs under realistic conditions, highlighting its improved safety and adaptability compared to the state-of-the-art. 

**Abstract (ZH)**: 在内河部署自主航行水面船舶提供了一种减少道路拥堵和排放的可持续替代方案。然而，在狭窄水道中航行 presents unique challenges，包括狭窄航道、更高交通密度和水动力干扰。现有船舶自主导航方法往往无法应对这些环境所需的鲁棒性和精确度。本文提出了一种新的运动规划方法，用于自主水面船舶（ASVs）的鲁棒模型预测控制（RMPC）结合控制障碍函数（CBFs），通过在控制设计框架中同时考虑航道边界和障碍物作为安全约束，该方法确保了复杂水道中的碰撞避免和稳健导航。仿真结果表明，在实际条件下，所提出的方法能够安全地引导ASVs，并突出其与现有技术相比的优势，包括更高的安全性和适应性。 

---
# An Adaptive Coverage Control Approach for Multiple Autonomous Off-road Vehicles in Dynamic Agricultural Fields 

**Title (ZH)**: 多自主 Off-road 车辆在动态农业田地中的自适应覆盖控制方法 

**Authors**: Sajad Ahmadi, Mohammadreza Davoodi, Javad Mohammadpour Velni  

**Link**: [PDF](https://arxiv.org/pdf/2509.06682)  

**Abstract**: This paper presents an adaptive coverage control method for a fleet of off-road and Unmanned Ground Vehicles (UGVs) operating in dynamic (time-varying) agricultural environments. Traditional coverage control approaches often assume static conditions, making them unsuitable for real-world farming scenarios where obstacles, such as moving machinery and uneven terrains, create continuous challenges. To address this, we propose a real-time path planning framework that integrates Unmanned Aerial Vehicles (UAVs) for obstacle detection and terrain assessment, allowing UGVs to dynamically adjust their coverage paths. The environment is modeled as a weighted directed graph, where the edge weights are continuously updated based on the UAV observations to reflect obstacle motion and terrain variations. The proposed approach incorporates Voronoi-based partitioning, adaptive edge weight assignment, and cost-based path optimization to enhance navigation efficiency. Simulation results demonstrate the effectiveness of the proposed method in improving path planning, reducing traversal costs, and maintaining robust coverage in the presence of dynamic obstacles and muddy terrains. 

**Abstract (ZH)**: 本文提出了一种适应性覆盖控制方法，用于在动态农业环境中操作的无远程地面车辆（UGVs）队列。传统的覆盖控制方法通常假设静态条件，这使得它们不适合包含移动机械和不平地形等障碍物的真实农业场景。为此，我们提出了一种结合无人驾驶航空 vehicles (UAVs) 的实时路径规划框架，用于障碍检测和地形评估，使UGVs能够动态调整其覆盖路径。环境被建模为加权有向图，其中边权重根据UAV观察结果连续更新以反映障碍物运动和地形变化。所提出的方法结合了基于Voronoi的分区、自适应边权重分配和基于成本的路径优化，以提高导航效率。仿真实验结果表明，所提出的方法在动态障碍物和泥泞地形等动态环境下的路径规划、降低通行成本和保持稳健覆盖方面的有效性。 

---
# A Robust Approach for LiDAR-Inertial Odometry Without Sensor-Specific Modeling 

**Title (ZH)**: 一种无需专用传感器建模的鲁棒LiDAR-惯性里程计方法 

**Authors**: Meher V.R. Malladi, Tiziano Guadagnino, Luca Lobefaro, Cyrill Stachniss  

**Link**: [PDF](https://arxiv.org/pdf/2509.06593)  

**Abstract**: Accurate odometry is a critical component in a robotic navigation stack, and subsequent modules such as planning and control often rely on an estimate of the robot's motion. Sensor-based odometry approaches should be robust across sensor types and deployable in different target domains, from solid-state LiDARs mounted on cars in urban-driving scenarios to spinning LiDARs on handheld packages used in unstructured natural environments. In this paper, we propose a robust LiDAR-inertial odometry system that does not rely on sensor-specific modeling. Sensor fusion techniques for LiDAR and inertial measurement unit (IMU) data typically integrate IMU data iteratively in a Kalman filter or use pre-integration in a factor graph framework, combined with LiDAR scan matching often exploiting some form of feature extraction. We propose an alternative strategy that only requires a simplified motion model for IMU integration and directly registers LiDAR scans in a scan-to-map approach. Our approach allows us to impose a novel regularization on the LiDAR registration, improving the overall odometry performance. We detail extensive experiments on a number of datasets covering a wide array of commonly used robotic sensors and platforms. We show that our approach works with the exact same configuration in all these scenarios, demonstrating its robustness. We have open-sourced our implementation so that the community can build further on our work and use it in their navigation stacks. 

**Abstract (ZH)**: 基于激光雷达-惯性测速的鲁棒系统及其应用 

---
# Event Driven CBBA with Reduced Communication 

**Title (ZH)**: 基于事件驱动的通信减少CBBA算法 

**Authors**: Vinita Sao, Tu Dac Ho, Sujoy Bhore, P.B. Sujit  

**Link**: [PDF](https://arxiv.org/pdf/2509.06481)  

**Abstract**: In various scenarios such as multi-drone surveillance and search-and-rescue operations, deploying multiple robots is essential to accomplish multiple tasks at once. Due to the limited communication range of these vehicles, a decentralised task allocation algorithm is crucial for effective task distribution among robots. The consensus-based bundle algorithm (CBBA) has been promising for multi-robot operation, offering theoretical guarantees. However, CBBA demands continuous communication, leading to potential congestion and packet loss that can hinder performance. In this study, we introduce an event-driven communication mechanism designed to address these communication challenges while maintaining the convergence and performance bounds of CBBA. We demonstrate theoretically that the solution quality matches that of CBBA and validate the approach with Monte-Carlo simulations across varying targets, agents, and bundles. Results indicate that the proposed algorithm (ED-CBBA) can reduce message transmissions by up to 52%. 

**Abstract (ZH)**: 基于事件驱动的通信机制CBBA在多机器人操作中的应用研究 

---
# Grasp-MPC: Closed-Loop Visual Grasping via Value-Guided Model Predictive Control 

**Title (ZH)**: 抓取-模型预测控制：基于价值引导的闭环视觉抓取 

**Authors**: Jun Yamada, Adithyavairavan Murali, Ajay Mandlekar, Clemens Eppner, Ingmar Posner, Balakumar Sundaralingam  

**Link**: [PDF](https://arxiv.org/pdf/2509.06201)  

**Abstract**: Grasping of diverse objects in unstructured environments remains a significant challenge. Open-loop grasping methods, effective in controlled settings, struggle in cluttered environments. Grasp prediction errors and object pose changes during grasping are the main causes of failure. In contrast, closed-loop methods address these challenges in simplified settings (e.g., single object on a table) on a limited set of objects, with no path to generalization. We propose Grasp-MPC, a closed-loop 6-DoF vision-based grasping policy designed for robust and reactive grasping of novel objects in cluttered environments. Grasp-MPC incorporates a value function, trained on visual observations from a large-scale synthetic dataset of 2 million grasp trajectories that include successful and failed attempts. We deploy this learned value function in an MPC framework in combination with other cost terms that encourage collision avoidance and smooth execution. We evaluate Grasp-MPC on FetchBench and real-world settings across diverse environments. Grasp-MPC improves grasp success rates by up to 32.6% in simulation and 33.3% in real-world noisy conditions, outperforming open-loop, diffusion policy, transformer policy, and IQL approaches. Videos and more at this http URL. 

**Abstract (ZH)**: 在杂乱环境中抓取多种物体仍然是一个重大挑战。开环抓取方法在受控环境中有效，但在杂乱环境中表现不佳。抓取过程中的抓取预测误差和物体姿态变化是失败的主要原因。相比之下，闭环方法可以在简化环境中（例如，桌上单一物体）解决这些问题，但在少量物体上进行，并没有普适性的途径。我们提出了一种名为Grasp-MPC的闭环六自由度基于视觉的抓取策略，旨在在杂乱环境中稳健且反应迅速地抓取新型物体。Grasp-MPC结合了一种在包含成功和失败尝试的大规模合成数据集（200万次抓取轨迹）上的视觉观察训练的价值函数。我们在基于MPC框架中部署了这个学习到的价值函数，并与其他鼓励碰撞避免和平滑执行的代价项结合使用。我们在FetchBench和各种真实环境中评估了Grasp-MPC。在仿真环境中，Grasp-MPC的抓取成功率提高了多达32.6%，在实际嘈杂环境中提高了33.3%，优于开环、扩散策略、变压器策略和IQL方法。更多信息和视频请点击此链接。 

---
# A Hybrid TDMA/CSMA Protocol for Time-Sensitive Traffic in Robot Applications 

**Title (ZH)**: 一种用于机器人应用中时间敏感交通的混合TDMA/CSMA协议 

**Authors**: Shiqi Xu, Lihao Zhang, Yuyang Du, Qun Yang, Soung Chang Liew  

**Link**: [PDF](https://arxiv.org/pdf/2509.06119)  

**Abstract**: Recent progress in robotics has underscored the demand for real-time control in applications such as manufacturing, healthcare, and autonomous systems, where the timely delivery of mission-critical commands under heterogeneous robotic traffic is paramount for operational efficacy and safety. In these scenarios, mission-critical traffic follows a strict deadline-constrained communication pattern: commands must arrive within defined QoS deadlines, otherwise late arrivals can degrade performance or destabilize control this http URL this work, we demonstrate on a real-time SDR platform that CSMA, widely adopted in robotic communications,suffers severe degradation under high robot traffic loads, with contention-induced collisions and delays disrupting the on-time arrival of mission-critical packets. To address this problem, we propose an IEEE 802.11-compatible hybrid TDMA/CSMA protocol that combines TDMA's deterministic slot scheduling with CSMA's adaptability for heterogeneous robot this http URL protocol achieves collision-free, low-latency mission-critical command delivery and IEEE 802.11 compatibility through the synergistic integration of sub-microsecond PTP-based slot synchronization-essential for establishing precise timing for TDMA, a three-session superframe with dynamic TDMA allocation for structured and adaptable traffic management,and beacon-NAV protection to preemptively secure these critical communication sessions from interference. Emulation experiments on real-time SDR testbed and Robot Operating System (ROS) simulation show that the proposed protocol reduces missed-deadline errors by 93% compared to the CSMA baseline. In high-speed robot path-tracking ROS simulations, the protocol lowers Root Mean Square (RMS) trajectory error by up to 90% compared with a CSMA baseline, all while maintaining throughput for non-critical traffic within +-2%. 

**Abstract (ZH)**: 近期机器人领域的进展凸显了在制造、医疗和自主系统等应用中对实时控制的需求，特别是在异构机器人流量下的及时交付关键任务命令对于操作有效性和安全至关重要。在这种场景下，关键任务流量遵循严格的截止时间约束通信模式：命令必须在定义的QoS截止时间内到达，否则迟到可能会降低性能或导致控制失稳。在这项工作中，我们基于真实时间SDR平台演示了在高机器人流量负载下，广泛应用于机器人通信的CSMA表现出严重的性能下降，内容冲突引发的碰撞和延迟破坏了关键任务数据包的及时到达。为了解决这一问题，我们提出了一种兼容IEEE 802.11的混合TDMA/CSMA协议，该协议结合了TDMA的确定性时隙调度与CSMA的异构机器人环境下的适应性。该协议通过亚微秒级PTP基时隙同步的协同整合实现无碰撞、低延迟的关键任务命令交付，并通过动态TDMA分配的三会话超帧和基于标志-NAV保护机制来确保关键通信会话免受干扰，从而实现了兼容IEEE 802.11、无冲突和低延迟的关键任务命令交付。仿真实验结果表明，与CSMA基线相比，所提出的协议将错过截止时间的错误率降低了93%。在高速机器人路径跟踪的ROS仿真中，与CSMA基线相比，该协议将均方根轨迹误差降低了高达90%，同时非关键任务的吞吐量保持在±2%以内。 

---
# Hybrid A* Path Planning with Multi-Modal Motion Extension for Four-Wheel Steering Mobile Robots 

**Title (ZH)**: 基于多模态运动扩展的四轮转向移动机器人工斤 breadcrumb导航规划 

**Authors**: Runjiao Bao, Lin Zhang, Tianwei Niu, Haoyu Yuan, Shoukun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06115)  

**Abstract**: Four-wheel independent steering (4WIS) systems provide mobile robots with a rich set of motion modes, such as Ackermann steering, lateral steering, and parallel movement, offering superior maneuverability in constrained environments. However, existing path planning methods generally assume a single kinematic model and thus fail to fully exploit the multi-modal capabilities of 4WIS platforms. To address this limitation, we propose an extended Hybrid A* framework that operates in a four-dimensional state space incorporating both spatial states and motion modes. Within this framework, we design multi-modal Reeds-Shepp curves tailored to the distinct kinematic constraints of each motion mode, develop an enhanced heuristic function that accounts for mode-switching costs, and introduce a terminal connection strategy with intelligent mode selection to ensure smooth transitions between different steering patterns. The proposed planner enables seamless integration of multiple motion modalities within a single path, significantly improving flexibility and adaptability in complex environments. Results demonstrate significantly improved planning performance for 4WIS robots in complex environments. 

**Abstract (ZH)**: 四轮独立转向（4WIS）系统为移动机器人提供了丰富的运动模式，如Ackermann转向、侧向转向和并行运动，使其在受限环境中具有卓越的机动性。然而，现有的路径规划方法通常假设单一的运动学模型，因而未能充分利用4WIS平台的多模态能力。为解决这一局限性，我们提出了一种扩展的混合A*框架，该框架在四维状态空间中运行，同时考虑空间状态和运动模式。在此框架内，我们设计了适应每种运动模式独特运动学约束的多模态Reeds-Shepp曲线，开发了考虑模式切换成本的增强启发式函数，并引入了带智能模式选择的终端连接策略，以确保不同转向模式之间的平滑过渡。所提出的规划器能够在单条路径中无缝集成多种运动模态，显著提高复杂环境中路径规划的灵活性和适应性。结果表明，该规划器显著改善了4WIS机器人的路径规划性能。 

---
# Energy-Efficient Path Planning with Multi-Location Object Pickup for Mobile Robots on Uneven Terrain 

**Title (ZH)**: 不规则地形上带多地点物体拾取的移动机器人能量高效路径规划 

**Authors**: Faiza Babakano, Ahmed Fahmin, Bojie Shen, Muhammad Aamir Cheema, Isma Farah Siddiqui  

**Link**: [PDF](https://arxiv.org/pdf/2509.06061)  

**Abstract**: Autonomous Mobile Robots (AMRs) operate on battery power, making energy efficiency a critical consideration, particularly in outdoor environments where terrain variations affect energy consumption. While prior research has primarily focused on computing energy-efficient paths from a source to a destination, these approaches often overlook practical scenarios where a robot needs to pick up an object en route - an action that can significantly impact energy consumption due to changes in payload. This paper introduces the Object-Pickup Minimum Energy Path Problem (OMEPP), which addresses energy-efficient route planning for AMRs required to pick up an object from one of many possible locations and deliver it to a destination. To address OMEPP, we first introduce a baseline algorithm that employs the Z star algorithm, a variant of A star tailored for energy-efficient routing, to iteratively visit each pickup point. While this approach guarantees optimality, it suffers from high computational cost due to repeated searches at each pickup location. To mitigate this inefficiency, we propose a concurrent PCPD search that manages multiple Z star searches simultaneously across all pickup points. Central to our solution is the Payload-Constrained Path Database (PCPD), an extension of the Compressed Path Database (CPD) that incorporates payload constraints. We demonstrate that PCPD significantly reduces branching factors during search, improving overall performance. Although the concurrent PCPD search may produce slightly suboptimal solutions, extensive experiments on real-world datasets show it achieves near-optimal performance while being one to two orders of magnitude faster than the baseline algorithm. 

**Abstract (ZH)**: 自主移动机器人对象拾取最低能耗路径问题（Object-Pickup Minimum Energy Path Problem, OMEPP） 

---
# Robotic Manipulation Framework Based on Semantic Keypoints for Packing Shoes of Different Sizes, Shapes, and Softness 

**Title (ZH)**: 基于语义关键点的机器人 manipulation 框架：用于不同大小、形状和柔软度的鞋子包装 

**Authors**: Yi Dong, Yangjun Liu, Jinjun Duan, Yang Li, Zhendong Dai  

**Link**: [PDF](https://arxiv.org/pdf/2509.06048)  

**Abstract**: With the rapid development of the warehousing and logistics industries, the packing of goods has gradually attracted the attention of academia and industry. The packing of footwear products is a typical representative paired-item packing task involving irregular shapes and deformable objects. Although studies on shoe packing have been conducted, different initial states due to the irregular shapes of shoes and standard packing placement poses have not been considered. This study proposes a robotic manipulation framework, including a perception module, reorientation planners, and a packing planner, that can complete the packing of pairs of shoes in any initial state. First, to adapt to the large intraclass variations due to the state, shape, and deformation of the shoe, we propose a vision module based on semantic keypoints, which can also infer more information such as size, state, pose, and manipulation points by combining geometric features. Subsequently, we not only proposed primitive-based reorientation methods for different states of a single deformable shoe but also proposed a fast reorientation method for the top state using box edge contact and gravity, which further improved the efficiency of reorientation. Finally, based on the perception module and reorientation methods, we propose a task planner for shoe pair packing in any initial state to provide an optimal packing strategy. Real-world experiments were conducted to verify the robustness of the reorientation methods and the effectiveness of the packing strategy for various types of shoes. In this study, we highlight the potential of semantic keypoint representation methods, introduce new perspectives on the reorientation of 3D deformable objects and multi-object manipulation, and provide a reference for paired object packing. 

**Abstract (ZH)**: 仓储和物流行业迅速发展背景下，货物包装逐渐引起学术界和工业界的关注。鞋类产品包装是涉及不规则形状和可变形物体的典型成对物品包装任务。虽然已有针对鞋类产品包装的研究，但由于鞋类形状不规则和标准包装放置姿态的不同初始状态尚未被考虑。本研究提出了一种机器人操作框架，包括感知模块、重新定位规划器和包装规划器，可以在任意初始状态下完成鞋类成对物品的包装。首先，为了适应由于鞋类状态、形状和变形导致的大量类内变化，我们提出了一种基于语义关键点的视觉模块，该模块还可以结合几何特征推断更多信息，如尺寸、状态、姿态和操作点。随后，我们不仅提出了适用于单个可变形鞋类不同姿态的原始重新定位方法，还提出了一种利用盒子边缘接触和重力的快速重新定位方法，从而进一步提高了重新定位的效率。最后，基于感知模块和重新定位方法，我们提出了适用于任意初始状态的鞋类成对物品包装任务规划器，以提供最优包装策略。实际实验验证了重新定位方法的鲁棒性和适用于各种鞋类的包装策略的有效性。本研究突显了语义关键点表示方法的潜力，提出了关于三维可变形物体重新定位和多物体操作的新视角，并为成对物体包装提供了参考。 

---
# Scenario-based Decision-making Using Game Theory for Interactive Autonomous Driving: A Survey 

**Title (ZH)**: 基于场景的博弈论交互自主驾驶决策研究综述 

**Authors**: Zhihao Lin, Zhen Tian  

**Link**: [PDF](https://arxiv.org/pdf/2509.05777)  

**Abstract**: Game-based interactive driving simulations have emerged as versatile platforms for advancing decision-making algorithms in road transport mobility. While these environments offer safe, scalable, and engaging settings for testing driving strategies, ensuring both realism and robust performance amid dynamic and diverse scenarios remains a significant challenge. Recently, the integration of game-based techniques with advanced learning frameworks has enabled the development of adaptive decision-making models that effectively manage the complexities inherent in varied driving conditions. These models outperform traditional simulation methods, especially when addressing scenario-specific challenges, ranging from obstacle avoidance on highways and precise maneuvering during on-ramp merging to navigation in roundabouts, unsignalized intersections, and even the high-speed demands of autonomous racing. Despite numerous innovations in game-based interactive driving, a systematic review comparing these approaches across different scenarios is still missing. This survey provides a comprehensive evaluation of game-based interactive driving methods by summarizing recent advancements and inherent roadway features in each scenario. Furthermore, the reviewed algorithms are critically assessed based on their adaptation of the standard game model and an analysis of their specific mechanisms to understand their impact on decision-making performance. Finally, the survey discusses the limitations of current approaches and outlines promising directions for future research. 

**Abstract (ZH)**: 基于游戏的交互式驾驶仿真已成为道路交通移动性决策算法发展的多功能平台。虽然这些环境提供了安全、可扩展且富有吸引力的测试驾驶策略的场所，但在动态和多样化场景中确保真实性和稳健性能仍然是一个重要挑战。近年来，将基于游戏的技术与先进学习框架相结合，已能够开发出有效管理各种驾驶条件固有复杂性的自适应决策模型。这些模型在处理具体场景挑战方面优于传统仿真方法，包括但不限于在高速公路上避免障碍物、入匝道精确操控行为、环岛导航、无信号交叉口通行以及高速自主竞速的需求。尽管在基于游戏的交互式驾驶方面取得了众多创新，但缺乏对不同场景下这些方法的系统性综述。本文综述提供了一种全面评估基于游戏的交互式驾驶方法的方式，总结了每种场景下的最新进展和道路特性。此外，根据标准游戏模型的适应性和其特定机制对所评估算法进行了批判性分析，以了解其对决策性能的影响。最后，本文讨论了当前方法的局限性，并指出了未来研究的有希望的方向。 

---
# Super-LIO: A Robust and Efficient LiDAR-Inertial Odometry System with a Compact Mapping Strategy 

**Title (ZH)**: 超LIO：一种具有紧凑mapping策略的鲁棒高效LiDAR-惯性里程计系统 

**Authors**: Liansheng Wang, Xinke Zhang, Chenhui Li, Dongjiao He, Yihan Pan, Jianjun Yi  

**Link**: [PDF](https://arxiv.org/pdf/2509.05723)  

**Abstract**: LiDAR-Inertial Odometry (LIO) is a foundational technique for autonomous systems, yet its deployment on resource-constrained platforms remains challenging due to computational and memory limitations. We propose Super-LIO, a robust LIO system that demands both high performance and accuracy, ideal for applications such as aerial robots and mobile autonomous systems. At the core of Super-LIO is a compact octo-voxel-based map structure, termed OctVox, that limits each voxel to eight fused subvoxels, enabling strict point density control and incremental denoising during map updates. This design enables a simple yet efficient and accurate map structure, which can be easily integrated into existing LIO frameworks. Additionally, Super-LIO designs a heuristic-guided KNN strategy (HKNN) that accelerates the correspondence search by leveraging spatial locality, further reducing runtime overhead. We evaluated the proposed system using four publicly available datasets and several self-collected datasets, totaling more than 30 sequences. Extensive testing on both X86 and ARM platforms confirms that Super-LIO offers superior efficiency and robustness, while maintaining competitive accuracy. Super-LIO processes each frame approximately 73% faster than SOTA, while consuming less CPU resources. The system is fully open-source and plug-and-play compatible with a wide range of LiDAR sensors and platforms. The implementation is available at: this https URL 

**Abstract (ZH)**: 基于LiDAR-惯性里程计的Super-LIO：一种高性能高精度的自主系统关键技 

---
# A*-PRM: A Dynamic Weight-Based Probabilistic Roadmap Algorithm 

**Title (ZH)**: A*-PRM: 一种基于动态权重的概率路网算法 

**Authors**: Siyuan Wang, Shuyi Zhang, Zhen Tian, Yuheng Yao, Gongsen Wang, Yu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.05701)  

**Abstract**: Robot path planning is a fundamental challenge in enhancing the environmental adaptability of autonomous navigation systems. This paper presents a hybrid path planning algorithm, A-star PRM, which incorporates dynamic weights. By embedding the Manhattan distance heuristic of the A-star algorithm into the random sampling process of PRM, the algorithm achieves a balanced optimization of path quality and computational efficiency. The approach uses a hierarchical sampling strategy and a dynamic connection mechanism, greatly improving adaptability to complex obstacle distributions. Experiments show that under a baseline configuration with one thousand sampled vertices, the path length of A-star PRM is 1073.23 plus or minus 14.8 meters and is 42.3 percent shorter than that of PRM with p value less than 0.01. With high-density sampling using three thousand vertices, the path length is reduced by 0.94 percent, 1036.61 meters compared with 1046.42 meters, while the increase in computational time is cut to about one tenth of the PRM increase, 71 percent compared with 785 percent. These results confirm the comprehensive advantages of A-star PRM in path quality, stability, and computational efficiency. Compared with existing hybrid algorithms, the proposed method shows clear benefits, especially in narrow channels and scenarios with dynamic obstacles. 

**Abstract (ZH)**: 基于动态权重的A-star PRM混合路径规划算法 

---
# Sharing but Not Caring: Similar Outcomes for Shared Control and Switching Control in Telepresence-Robot Navigation 

**Title (ZH)**: 共享而非关切：远程机器人导航中共享控制与切换控制具有相似的效果 

**Authors**: Juho Kalliokoski, Evan G. Center, Steven M. LaValle, Timo Ojala, Basak Sakcak  

**Link**: [PDF](https://arxiv.org/pdf/2509.05672)  

**Abstract**: Telepresence robots enable users to interact with remote environments, but efficient and intuitive navigation remains a challenge. In this work, we developed and evaluated a shared control method, in which the robot navigates autonomously while allowing users to affect the path generation to better suit their needs. We compared this with control switching, where users toggle between direct and automated control. We hypothesized that shared control would maintain efficiency comparable to control switching while potentially reducing user workload. The results of two consecutive user studies (each with final sample of n=20) showed that shared control does not degrade navigation efficiency, but did not show a significant reduction in task load compared to control switching. Further research is needed to explore the underlying factors that influence user preference and performance in these control systems. 

**Abstract (ZH)**: 远程存在机器人使用户能够与远程环境交互，但高效直观的导航仍具挑战性。在本项工作中，我们开发并评估了一种共享控制方法，该方法使机器人自主导航，同时允许用户影响路径生成以更好地满足其需求。我们将其与控制切换进行了比较，在控制切换中，用户可以在直接控制和自动化控制之间切换。我们假设共享控制能在保持与控制切换相当的导航效率的同时，可能减少用户的工作负荷。连续进行的两项用户研究（每项研究最终样本量均为n=20）的结果表明，共享控制并未损害导航效率，但并未显示出与控制切换相比显著降低任务负荷。需要进一步研究以探索这些控制系统的用户偏好和性能影响因素。 

---
# TeleopLab: Accessible and Intuitive Teleoperation of a Robotic Manipulator for Remote Labs 

**Title (ZH)**: TeleopLab: 便捷直观的远程操控机器人 manipulator 的实验平台 

**Authors**: Ziling Chen, Yeo Jung Yoon, Rolando Bautista-Montesano, Zhen Zhao, Ajay Mandlekar, John Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05547)  

**Abstract**: Teleoperation offers a promising solution for enabling hands-on learning in remote education, particularly in environments requiring interaction with real-world equipment. However, such remote experiences can be costly or non-intuitive. To address these challenges, we present TeleopLab, a mobile device teleoperation system that allows students to control a robotic arm and operate lab equipment. TeleopLab comprises a robotic arm, an adaptive gripper, cameras, lab equipment for a diverse range of applications, a user interface accessible through smartphones, and video call software. We conducted a user study, focusing on task performance, students' perspectives toward the system, usability, and workload assessment. Our results demonstrate a 46.1% reduction in task completion time as users gained familiarity with the system. Quantitative feedback highlighted improvements in students' perspectives after using the system, while NASA TLX and SUS assessments indicated a manageable workload of 38.2 and a positive usability of 73.8. TeleopLab successfully bridges the gap between physical labs and remote education, offering a scalable and effective platform for remote STEM learning. 

**Abstract (ZH)**: 远程教育中实物操作学习的电信操作提供了一种有前景的解决方案，特别是在需要与实物设备互动的环境中。然而，这样的远程体验可能成本高昂或不够直观。为了解决这些挑战，我们提出了TeleopLab，一种移动设备电信操作系统，允许学生控制机器人手臂并操作实验设备。TeleopLab 包括一个机器人手臂、一个自适应夹爪、摄像头、适用于各种应用的实验设备、可通过智能手机访问的用户界面以及视频通话软件。我们进行了一项用户研究，重点在于任务性能、学生对系统的看法、易用性和工作量评估。结果显示，在用户熟悉系统后，任务完成时间减少了46.1%。定量反馈表明，使用该系统后学生的态度有所改善，而NASA TLX 和SUS 评估显示其工作量为38.2，易用性为73.8。TeleopLab 成功地缓解了物理实验室与远程教育之间的差距，提供了一个可扩展且有效的远程STEM学习平台。 

---
# Microrobot Vascular Parkour: Analytic Geometry-based Path Planning with Real-time Dynamic Obstacle Avoidance 

**Title (ZH)**: 微机器人血管越障：基于解析几何的路径规划与实时动态避障 

**Authors**: Yanda Yang, Max Sokolich, Fatma Ceren Kirmizitas, Sambeeta Das, Andreas A. Malikopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2509.05500)  

**Abstract**: Autonomous microrobots in blood vessels could enable minimally invasive therapies, but navigation is challenged by dense, moving obstacles. We propose a real-time path planning framework that couples an analytic geometry global planner (AGP) with two reactive local escape controllers, one based on rules and one based on reinforcement learning, to handle sudden moving obstacles. Using real-time imaging, the system estimates the positions of the microrobot, obstacles, and targets and computes collision-free motions. In simulation, AGP yields shorter paths and faster planning than weighted A* (WA*), particle swarm optimization (PSO), and rapidly exploring random trees (RRT), while maintaining feasibility and determinism. We extend AGP from 2D to 3D without loss of speed. In both simulations and experiments, the combined global planner and local controllers reliably avoid moving obstacles and reach targets. The average planning time is 40 ms per frame, compatible with 25 fps image acquisition and real-time closed-loop control. These results advance autonomous microrobot navigation and targeted drug delivery in vascular environments. 

**Abstract (ZH)**: 自主微机器人在血管中的导航可实现微创治疗，但面临密集移动障碍的挑战。我们提出了一种实时路径规划框架，将解析几何全局规划器（AGP）与基于规则和强化学习的两个反应式局部逃生控制器结合，以应对突发的移动障碍。利用实时成像，系统估计微型机器人、障碍物和目标的位置，并计算无碰撞运动。在模拟中，AGP在路径长度和规划速度上优于加权A*（WA*）、粒子 swarm 优化（PSO）和快速扩展随机树（RRT），同时保持可行性和确定性。我们将AGP从2D扩展到3D而不损失速度。在模拟和实验中，结合的全局规划器和局部控制器可靠地避免了移动障碍并到达目标。平均规划时间为每帧40毫秒，与25 fps图像采集和实时闭环控制兼容。这些结果推动了血管环境中自主微机器人导航和靶向药物输送的发展。 

---
# HapMorph: A Pneumatic Framework for Multi-Dimensional Haptic Property Rendering 

**Title (ZH)**: HapMorph: 一种用于多维触觉属性渲染的气动框架 

**Authors**: Rui Chen, Domenico Chiaradia, Antonio Frisoli, Daniele Leonardis  

**Link**: [PDF](https://arxiv.org/pdf/2509.05433)  

**Abstract**: Haptic interfaces that can simultaneously modulate multiple physical properties remain a fundamental challenge in human-robot interaction. Existing systems typically allow the rendering of either geometric features or mechanical properties, but rarely both, within wearable form factors. Here, we introduce HapMorph, a pneumatic framework that enables continuous, simultaneous modulation of object size and stiffness through antagonistic fabric-based pneumatic actuators (AFPAs). We implemented a HapMorph protoytpe designed for hands interaction achieving size variation from 50 to 104 mm, stiffness modulation up to 4.7 N/mm and mass of the wearable parts of just 21 g. Through systematic characterization, we demonstrate decoupled control of size and stiffness properties via dual-chamber pressure regulation. Human perception studies with 10 participants reveal that users can distinguish nine discrete states across three size categories and three stiffness levels with 89.4% accuracy and 6.7 s average response time. We further demonstrate extended architectures that combine AFPAs with complementary pneumatic structures to enable shape or geometry morphing with concurrent stiffness control. Our results establish antagonistic pneumatic principle as a pathway toward next-generation haptic interfaces, capable of multi-dimensiona rendering properties within practical wearable constraints. 

**Abstract (ZH)**: 能够同时调节数个物理属性的触觉接口在人机交互中仍是一大基本挑战。现有系统通常只能在可穿戴形式因素内渲染几何特征或机械属性中的一项，但很少两者兼顾。本文介绍了HapMorph，一种基于 antagonistic fabric-based 气动执行器（AFPAs）的气动框架，能够在保持可穿戴性的同时，连续且同步地调变物体尺寸和刚度。我们设计了一个适用于手部交互的HapMorph原型，尺寸范围从50毫米到104毫米，最大刚度调节达到每毫米4.7牛顿，并且可穿戴部分的质量仅为21克。通过系统表征，我们展示了通过双腔室压力调节实现尺寸和刚度属性的独立控制。10名参与者的感知研究表明，用户能够以89.4%的准确率区分三个尺寸类别中的九种不同状态，并且平均响应时间为6.7秒。此外，我们展示了结合AFPAs和互补气动结构的扩展架构，能够实现同时控制刚度的形状或几何形态变化。我们的研究结果表明了对抗式气动原理是实现具备在实际可穿戴限制内多维属性渲染能力的下一代触觉接口的一条途径。 

---
# RoboBallet: Planning for Multi-Robot Reaching with Graph Neural Networks and Reinforcement Learning 

**Title (ZH)**: RoboBallet: 多机器人抓取规划的图神经网络与强化学习方法 

**Authors**: Matthew Lai, Keegan Go, Zhibin Li, Torsten Kroger, Stefan Schaal, Kelsey Allen, Jonathan Scholz  

**Link**: [PDF](https://arxiv.org/pdf/2509.05397)  

**Abstract**: Modern robotic manufacturing requires collision-free coordination of multiple robots to complete numerous tasks in shared, obstacle-rich workspaces. Although individual tasks may be simple in isolation, automated joint task allocation, scheduling, and motion planning under spatio-temporal constraints remain computationally intractable for classical methods at real-world scales. Existing multi-arm systems deployed in the industry rely on human intuition and experience to design feasible trajectories manually in a labor-intensive process. To address this challenge, we propose a reinforcement learning (RL) framework to achieve automated task and motion planning, tested in an obstacle-rich environment with eight robots performing 40 reaching tasks in a shared workspace, where any robot can perform any task in any order. Our approach builds on a graph neural network (GNN) policy trained via RL on procedurally-generated environments with diverse obstacle layouts, robot configurations, and task distributions. It employs a graph representation of scenes and a graph policy neural network trained through reinforcement learning to generate trajectories of multiple robots, jointly solving the sub-problems of task allocation, scheduling, and motion planning. Trained on large randomly generated task sets in simulation, our policy generalizes zero-shot to unseen settings with varying robot placements, obstacle geometries, and task poses. We further demonstrate that the high-speed capability of our solution enables its use in workcell layout optimization, improving solution times. The speed and scalability of our planner also open the door to new capabilities such as fault-tolerant planning and online perception-based re-planning, where rapid adaptation to dynamic task sets is required. 

**Abstract (ZH)**: 现代机器人制造需要在共享且障碍丰富的 workspace 中实现多个机器人无碰撞的协调作业以完成多项任务。现有的行业多臂系统依赖人工经验和直观设计可行轨迹，过程 labor-intensive。为应对这一挑战，我们提出一种基于强化学习（RL）的框架实现自动的任务和运动规划，该框架在包含八个机器人执行40项抓取任务的障碍丰富环境中进行测试。我们的方法基于通过强化学习在生成式环境中训练的图神经网络（GNN）策略。该方法利用场景的图表示和通过强化学习训练的图策略神经网络来生成多个机器人的轨迹，联合解决子问题：任务分配、调度和运动规划。在大规模随机生成的任务集仿真中训练，我们的策略在不同机器人布局、障碍几何和任务姿态的未见过的场景中实现了零样本泛化。进一步证明，我们的解决方案的高速能力使其适用于工位布局优化，提高了解决问题的时间。我们规划器的速度和可扩展性还开启了新的能力，如容错规划和基于在线感知的重规划，其中快速适应动态任务集合是必要的。 

---
# Evaluating Magic Leap 2 Tool Tracking for AR Sensor Guidance in Industrial Inspections 

**Title (ZH)**: 评估Magic Leap 2工具跟踪在工业检测中作为AR传感器引导的应用 

**Authors**: Christian Masuhr, Julian Koch, Thorsten Schüppstuhl  

**Link**: [PDF](https://arxiv.org/pdf/2509.05391)  

**Abstract**: Rigorous evaluation of commercial Augmented Reality (AR) hardware is crucial, yet public benchmarks for tool tracking on modern Head-Mounted Displays (HMDs) are limited. This paper addresses this gap by systematically assessing the Magic Leap 2 (ML2) controllers tracking performance. Using a robotic arm for repeatable motion (EN ISO 9283) and an optical tracking system as ground truth, our protocol evaluates static and dynamic performance under various conditions, including realistic paths from a hydrogen leak inspection use case. The results provide a quantitative baseline of the ML2 controller's accuracy and repeatability and present a robust, transferable evaluation methodology. The findings provide a basis to assess the controllers suitability for the inspection use case and similar industrial sensor-based AR guidance tasks. 

**Abstract (ZH)**: 商业增强现实（AR）硬件的严谨评估至关重要，但现代头戴式显示器（HMD）工具跟踪的公开基准有限。本文通过系统评估Magic Leap 2（ML2）控制器的跟踪性能来弥补这一缺口。使用机器人臂进行可重复运动（EN ISO 9283）和光学跟踪系统作为参考，我们的协议在多种条件下评估其静态和动态性能，包括氢泄漏检查使用案例中的真实路径。结果提供了ML2控制器准确性和重复性的量化基准，并提出了一种稳健且可转移的评估方法。研究结果为评估控制器在检查使用案例及其他类似工业基于传感器的AR指导任务中的适用性提供了基础。 

---
# Online Clustering of Seafloor Imagery for Interpretation during Long-Term AUV Operations 

**Title (ZH)**: 长期内海床成像的在线聚类 Interpretation during Long-Term AUV Operations 

**Authors**: Cailei Liang, Adrian Bodenmann, Sam Fenton, Blair Thornton  

**Link**: [PDF](https://arxiv.org/pdf/2509.06678)  

**Abstract**: As long-endurance and seafloor-resident AUVs become more capable, there is an increasing need for extended, real-time interpretation of seafloor imagery to enable adaptive missions and optimise communication efficiency. Although offline image analysis methods are well established, they rely on access to complete datasets and human-labelled examples to manage the strong influence of environmental and operational conditions on seafloor image appearance-requirements that cannot be met in real-time settings. To address this, we introduce an online clustering framework (OCF) capable of interpreting seafloor imagery without supervision, which is designed to operate in real-time on continuous data streams in a scalable, adaptive, and self-consistent manner. The method enables the efficient review and consolidation of common patterns across the entire data history in constant time by identifying and maintaining a set of representative samples that capture the evolving feature distribution, supporting dynamic cluster merging and splitting without reprocessing the full image history. We evaluate the framework on three diverse seafloor image datasets, analysing the impact of different representative sampling strategies on both clustering accuracy and computational cost. The OCF achieves the highest average F1 score of 0.68 across the three datasets among all comparative online clustering approaches, with a standard deviation of 3% across three distinct survey trajectories, demonstrating its superior clustering capability and robustness to trajectory variation. In addition, it maintains consistently lower and bounded computational time as the data volume increases. These properties are beneficial for generating survey data summaries and supporting informative path planning in long-term, persistent autonomous marine exploration. 

**Abstract (ZH)**: 随着长时间和海底驻留自主 underwater 车辆（AUV）的能力增强，对实时解释海底图像进行扩展和实时解释的需求不断增加，以支持适应性任务并优化通信效率。尽管离线图像分析方法已经成熟，但它们依赖于完整数据集的访问和带有环境和操作条件标签的数据示例来管理对海底图像外观的强烈影响，这在实时环境中难以满足。为此，我们提出了一种在线聚类框架（OCF），该框架能够在不监督的情况下解释海底图像，并设计为能够以可扩展、适应性和自一致的方式实时处理连续数据流。该方法通过识别并维护一组代表性样本来高效地审查和合并整个数据历史中的共现模式，这些样本捕捉到特征分布的变化，并支持在无需重新处理完整图像历史的情况下进行动态聚类合并和分裂。我们在三个不同的海底图像数据集上评估了该框架，分析了不同类型代表性样本策略对聚类准确性和计算成本的影响。OCF 在所有比较的在线聚类方法中实现了最高的平均 F1 分数 0.68，并且在三条不同的调查轨迹上标准偏差为 3%，显示出它在聚类能力和轨道变化稳健性方面的优越性。此外，随着数据量的增加，它能够保持一致并受到良好控制的计算时间。这些属性对于生成长期持续自主海洋探索的调查数据摘要并支持信息性路径规划是十分有益的。 

---
# Advancing Resource Extraction Systems in Martian Volcanic Terrain: Rover Design, Power Consumption and Hazard Analysis 

**Title (ZH)**: 在火星火山地形中推进资源开采系统：漫游车设计、能耗分析与风险评估 

**Authors**: Divij Gupta, Arkajit Aich  

**Link**: [PDF](https://arxiv.org/pdf/2509.06103)  

**Abstract**: This study proposes a schematic plan for in-situ resource utilization (ISRU) in Martian volcanic terrains. The work investigated the complexity of volcanic terrains and Martian environmental hazards and suggested comprehensive engineering strategies to overcome the odds and establish a successful mining program in Martian volcanic regions. Slope stabilization methods - such as terracing and anchored drilling rigs - with terrain-adaptive rovers capable of autonomous operations on steep unstable slopes has been suggested as feasible solutions to navigate the complex geological terrains of Martian volcanoes. The mid range rover design with a mass of approximately 2.1 t, proposed here for mining operations, incorporates a six-wheel rocker-bogie suspension, anchoring-enabled drilling arm, dust-mitigation solar arrays, and advanced sensing systems for hazard detection and navigation. A comparative analysis regarding choice of roads and rails for building transport infrastructure has also been performed. We have also looked into the energy requirement of the rover to work under extreme environmental conditions of Mars and suggested a combination of solar and nuclear power to account for the huge energy requirements of sustained operations on Mars. The results demonstrate that mission success in these environments depends on integrating mechanical resilience, environmental adaptability, and operational autonomy, enabling sustainable access to resources in one of Mars' most geologically challenging settings. 

**Abstract (ZH)**: 本研究提出了一种在火星火山地形中就地资源利用（ISRU）的方案。工作调查了火山地形的复杂性以及火星环境风险，并建议了全面的工程策略以克服这些挑战并在火星火山区域建立成功的采矿计划。提议了诸如梯田化和固定式钻探平台等边坡稳定方法，以及具备自主操作能力的地形适应性轮式机器人，以在陡峭不稳定的火山地形中导航。为采矿操作设计的中型漫游者重约2.1吨，配备了六轮摆臂悬挂系统、可锚固的钻探臂、防尘光伏阵列和先进的传感系统，用于危险检测和导航。还进行了关于道路和铁轨选择以构建运输基础设施的竞争性分析。此外，还探讨了漫游者在火星极端环境条件下工作的能源需求，并建议结合太阳能和核能来满足火星长时间运营的巨大能源需求。结果表明，这些环境中的任务成功取决于将机械韧性、环境适应性和操作自主性相结合，以实现对火星地质挑战性环境中资源的可持续利用。 

---
# Deep Reactive Policy: Learning Reactive Manipulator Motion Planning for Dynamic Environments 

**Title (ZH)**: 深度反应性策略：学习动态环境下的反应性操纵器运动规划 

**Authors**: Jiahui Yang, Jason Jingzhou Liu, Yulong Li, Youssef Khaky, Kenneth Shaw, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2509.06953)  

**Abstract**: Generating collision-free motion in dynamic, partially observable environments is a fundamental challenge for robotic manipulators. Classical motion planners can compute globally optimal trajectories but require full environment knowledge and are typically too slow for dynamic scenes. Neural motion policies offer a promising alternative by operating in closed-loop directly on raw sensory inputs but often struggle to generalize in complex or dynamic settings. We propose Deep Reactive Policy (DRP), a visuo-motor neural motion policy designed for reactive motion generation in diverse dynamic environments, operating directly on point cloud sensory input. At its core is IMPACT, a transformer-based neural motion policy pretrained on 10 million generated expert trajectories across diverse simulation scenarios. We further improve IMPACT's static obstacle avoidance through iterative student-teacher finetuning. We additionally enhance the policy's dynamic obstacle avoidance at inference time using DCP-RMP, a locally reactive goal-proposal module. We evaluate DRP on challenging tasks featuring cluttered scenes, dynamic moving obstacles, and goal obstructions. DRP achieves strong generalization, outperforming prior classical and neural methods in success rate across both simulated and real-world settings. Video results and code available at this https URL 

**Abstract (ZH)**: 在动态部分可观测环境中生成无碰撞运动路径是机器人操作臂面临的基本挑战。经典运动规划器可以计算全局最优轨迹，但需要完全了解环境，并且通常在动态场景下速度过慢。神经运动策略通过直接在原始感官输入上闭环运行提供了一种有前景的替代方案，但在复杂或动态环境中常常难以泛化。我们提出了Deep Reactive Policy (DRP) 神经运动策略，专门设计用于在多变的动态环境中实时生成运动，直接处理点云感知输入。其核心是IMPACT，一种基于转换器的预训练神经运动策略，在多样化的模拟场景中生成了100万条专家轨迹。我们进一步通过迭代的学生-教师微调方法改进了IMPACT的静态障碍物回避能力。此外，在推理阶段，我们使用DCP-RMP局部反应性目标提议模块增强策略的动态障碍物回避能力。我们对包含杂乱场景、动态移动障碍物和目标障碍的任务评估了DRP，DRP表现出强大的泛化能力，在模拟和真实世界设置中的成功率均优于先前的经典和神经方法。视频结果和代码可在以下链接查看：this https URL。 

---
