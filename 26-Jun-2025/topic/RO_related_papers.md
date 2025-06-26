# EANS: Reducing Energy Consumption for UAV with an Environmental Adaptive Navigation Strategy 

**Title (ZH)**: EANS: 一种环境自适应导航策略降低无人机能耗 

**Authors**: Tian Liu, Han Liu, Boyang Li, Long Chen, Kai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20485)  

**Abstract**: Unmanned Aerial Vehicles (UAVS) are limited by the onboard energy. Refinement of the navigation strategy directly affects both the flight velocity and the trajectory based on the adjustment of key parameters in the UAVS pipeline, thus reducing energy consumption. However, existing techniques tend to adopt static and conservative strategies in dynamic scenarios, leading to inefficient energy reduction. Dynamically adjusting the navigation strategy requires overcoming the challenges including the task pipeline interdependencies, the environmental-strategy correlations, and the selecting parameters. To solve the aforementioned problems, this paper proposes a method to dynamically adjust the navigation strategy of the UAVS by analyzing its dynamic characteristics and the temporal characteristics of the autonomous navigation pipeline, thereby reducing UAVS energy consumption in response to environmental changes. We compare our method with the baseline through hardware-in-the-loop (HIL) simulation and real-world experiments, showing our method 3.2X and 2.6X improvements in mission time, 2.4X and 1.6X improvements in energy, respectively. 

**Abstract (ZH)**: 无人驾驶飞行器（UAV）受载荷能量限制。导航策略的优化直接影响飞行速度和轨迹，通过调整UAV管道中的关键参数来实现，从而减少能量消耗。然而，现有技术在动态场景中倾向于采用静态和保守策略，导致能量减少效率低下。动态调整导航策略需要克服任务管道相互依赖性、环境-策略相关性以及参数选择等挑战。为解决上述问题，本文提出了一种方法，通过分析UAV自主导航管道的动力学特性和时间特性来动态调整其导航策略，从而在环境变化时减少UAV的能量消耗。通过硬件在环（HIL）模拟和实际实验将我们的方法与基线进行比较，结果显示，分别在任务时间上提高了3.2倍、能源上提高了2.4倍和1.6倍。 

---
# Learn to Position -- A Novel Meta Method for Robotic Positioning 

**Title (ZH)**: 学习定位——一种新型元学习方法用于机器人定位 

**Authors**: Dongkun Wang, Junkai Zhao, Yunfei Teng, Jieyang Peng, Wenjing Xue, Xiaoming Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.20445)  

**Abstract**: Absolute positioning accuracy is a vital specification for robots. Achieving high position precision can be challenging due to the presence of various sources of errors. Meanwhile, accurately depicting these errors is difficult due to their stochastic nature. Vision-based methods are commonly integrated to guide robotic positioning, but their performance can be highly impacted by inevitable occlusions or adverse lighting conditions. Drawing on the aforementioned considerations, a vision-free, model-agnostic meta-method for compensating robotic position errors is proposed, which maximizes the probability of accurate robotic position via interactive feedback. Meanwhile, the proposed method endows the robot with the capability to learn and adapt to various position errors, which is inspired by the human's instinct for grasping under uncertainties. Furthermore, it is a self-learning and self-adaptive method able to accelerate the robotic positioning process as more examples are incorporated and learned. Empirical studies validate the effectiveness of the proposed method. As of the writing of this paper, the proposed meta search method has already been implemented in a robotic-based assembly line for odd-form electronic components. 

**Abstract (ZH)**: 基于交互反馈的无视觉模型agnostic元方法及其在机器人定位误差补偿中的应用 

---
# Enhanced Robotic Navigation in Deformable Environments using Learning from Demonstration and Dynamic Modulation 

**Title (ZH)**: 基于演示学习和动态调节的可变形环境增强机器人导航 

**Authors**: Lingyun Chen, Xinrui Zhao, Marcos P. S. Campanha, Alexander Wegener, Abdeldjallil Naceri, Abdalla Swikir, Sami Haddadin  

**Link**: [PDF](https://arxiv.org/pdf/2506.20376)  

**Abstract**: This paper presents a novel approach for robot navigation in environments containing deformable obstacles. By integrating Learning from Demonstration (LfD) with Dynamical Systems (DS), we enable adaptive and efficient navigation in complex environments where obstacles consist of both soft and hard regions. We introduce a dynamic modulation matrix within the DS framework, allowing the system to distinguish between traversable soft regions and impassable hard areas in real-time, ensuring safe and flexible trajectory planning. We validate our method through extensive simulations and robot experiments, demonstrating its ability to navigate deformable environments. Additionally, the approach provides control over both trajectory and velocity when interacting with deformable objects, including at intersections, while maintaining adherence to the original DS trajectory and dynamically adapting to obstacles for smooth and reliable navigation. 

**Abstract (ZH)**: 本文提出了一种新型机器人在包含可变形障碍物环境中的导航方法。通过将学习从演示（LfD）与动力学系统（DS）集成，使机器人能够在由软硬区域组成的复杂环境中实现适应性和高效的导航。在动力学系统框架中引入了动态调节矩阵，使系统能够实时区分可通行的软区域和不可通行的硬区域，从而确保安全和灵活的轨迹规划。通过广泛的仿真实验和机器人实验验证了该方法，展示了其在可变形环境中的导航能力。此外，该方法在与可变形物体交互时，包括在交叉口处，提供了对轨迹和速度的控制，同时保持对原始DS轨迹的遵从性，并动态适应障碍物以实现平滑可靠的导航。 

---
# Finding the Easy Way Through -- the Probabilistic Gap Planner for Social Robot Navigation 

**Title (ZH)**: 寻找简便之道——社会机器人导航的概率性间隙规划者 

**Authors**: Malte Probst, Raphael Wenzel, Tim Puphal, Monica Dasi, Nico A. Steinhardt, Sango Matsuzaki, Misa Komuro  

**Link**: [PDF](https://arxiv.org/pdf/2506.20320)  

**Abstract**: In Social Robot Navigation, autonomous agents need to resolve many sequential interactions with other agents. State-of-the art planners can efficiently resolve the next, imminent interaction cooperatively and do not focus on longer planning horizons. This makes it hard to maneuver scenarios where the agent needs to select a good strategy to find gaps or channels in the crowd. We propose to decompose trajectory planning into two separate steps: Conflict avoidance for finding good, macroscopic trajectories, and cooperative collision avoidance (CCA) for resolving the next interaction optimally. We propose the Probabilistic Gap Planner (PGP) as a conflict avoidance planner. PGP modifies an established probabilistic collision risk model to include a general assumption of cooperativity. PGP biases the short-term CCA planner to head towards gaps in the crowd. In extensive simulations with crowds of varying density, we show that using PGP in addition to state-of-the-art CCA planners improves the agents' performance: On average, agents keep more space to others, create less tension, and cause fewer collisions. This typically comes at the expense of slightly longer paths. PGP runs in real-time on WaPOCHI mobile robot by Honda R&D. 

**Abstract (ZH)**: 社会机器人导航中，自主代理需要解决与其它代理的许多序贯互动。最先进的规划者可以有效地协同解决即将发生的互动，但不聚焦于更长时间的规划展望。这使得在代理需要选择策略以在人群中共找到合适的空隙或通道时难以应对。我们提出将轨迹规划分解为两个单独的步骤：冲突避免以找到好的宏观轨迹，以及协同碰撞避免（CCA）以最优地解决下一个互动。我们提出了概率性间隙规划器（PGP）作为冲突避免规划器。PGP修改了现有的概率碰撞风险模型，使其包含普遍的协同假设。PGP使短期CCA规划器倾向于朝向人群中的空隙前进。在广泛且密度变化的群体仿真中，我们展示了使用PGP与最先进的CCA规划器相结合可以提高代理的表现：平均而言，代理与他人保持更大的空间，产生的紧张情绪更少，碰撞次数也更少。这通常会略微增加路径长度。PGP在本田研发的WaPOCHI移动机器人上实时运行。 

---
# Near Time-Optimal Hybrid Motion Planning for Timber Cranes 

**Title (ZH)**: 近时效最优混合运动规划木材起重机 

**Authors**: Marc-Philip Ecker, Bernhard Bischof, Minh Nhat Vu, Christoph Fröhlich, Tobias Glück, Wolfgang Kemmetmüller  

**Link**: [PDF](https://arxiv.org/pdf/2506.20314)  

**Abstract**: Efficient, collision-free motion planning is essential for automating large-scale manipulators like timber cranes. They come with unique challenges such as hydraulic actuation constraints and passive joints-factors that are seldom addressed by current motion planning methods. This paper introduces a novel approach for time-optimal, collision-free hybrid motion planning for a hydraulically actuated timber crane with passive joints. We enhance the via-point-based stochastic trajectory optimization (VP-STO) algorithm to include pump flow rate constraints and develop a novel collision cost formulation to improve robustness. The effectiveness of the enhanced VP-STO as an optimal single-query global planner is validated by comparison with an informed RRT* algorithm using a time-optimal path parameterization (TOPP). The overall hybrid motion planning is formed by combination with a gradient-based local planner that is designed to follow the global planner's reference and to systematically consider the passive joint dynamics for both collision avoidance and sway damping. 

**Abstract (ZH)**: 高效的、无碰撞路径规划对于自动化大规模操纵器如木材起重机至关重要。它们面临着独特的挑战，如液压驱动约束和被动关节因素，这些问题当前的路径规划方法很少予以考虑。本文提出了一种新颖的方法，用于液压驱动木材起重机（具有被动关节）的时最优、无碰撞混合路径规划。我们增强了基于途径点的随机轨迹优化（VP-STO）算法，使其包括泵流量约束，并开发了一种新的碰撞代价公式来提高鲁棒性。通过与基于启发式的RRT*算法（使用时最优路径参数化TOPP）进行比较，验证了增强的VP-STO作为最优单查询全局规划器的有效性。整体混合路径规划结合了一个基于梯度的局部规划器，该规划器设计用于跟踪全局规划器的参考，并系统地考虑被动关节动力学，以实现避碰和减摇。 

---
# Real-Time Obstacle Avoidance Algorithms for Unmanned Aerial and Ground Vehicles 

**Title (ZH)**: 实时障碍避让算法研究：无人机与地面车辆 

**Authors**: Jingwen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.20311)  

**Abstract**: The growing use of mobile robots in sectors such as automotive, agriculture, and rescue operations reflects progress in robotics and autonomy. In unmanned aerial vehicles (UAVs), most research emphasizes visual SLAM, sensor fusion, and path planning. However, applying UAVs to search and rescue missions in disaster zones remains underexplored, especially for autonomous navigation.
This report develops methods for real-time and secure UAV maneuvering in complex 3D environments, crucial during forest fires. Building upon past research, it focuses on designing navigation algorithms for unfamiliar and hazardous environments, aiming to improve rescue efficiency and safety through UAV-based early warning and rapid response.
The work unfolds in phases. First, a 2D fusion navigation strategy is explored, initially for mobile robots, enabling safe movement in dynamic settings. This sets the stage for advanced features such as adaptive obstacle handling and decision-making enhancements. Next, a novel 3D reactive navigation strategy is introduced for collision-free movement in forest fire simulations, addressing the unique challenges of UAV operations in such scenarios.
Finally, the report proposes a unified control approach that integrates UAVs and unmanned ground vehicles (UGVs) for coordinated rescue missions in forest environments. Each phase presents challenges, proposes control models, and validates them with mathematical and simulation-based evidence. The study offers practical value and academic insights for improving the role of UAVs in natural disaster rescue operations. 

**Abstract (ZH)**: 移动机器人在汽车、农业和救援操作领域的日益广泛应用反映了机器人技术和自主性的进步。在无人驾驶飞行器（UAVs）中，大多数研究集中于视觉SLAM、传感器融合和路径规划。然而，将UAV应用于灾害区搜救任务的自主导航研究仍相对较少。

本报告开发了在复杂3D环境中实时和安全的UAV机动方法，对于森林火灾等场景至关重要。在此前研究的基础上，本报告专注于设计适用于未知和危险环境的导航算法，旨在通过基于UAV的早期预警和快速响应提高救援效率和安全性。

这项工作分为几个阶段。首先，探索了一种2D融合导航策略，最初应用于移动机器人，以实现动态环境中的安全移动。这为引入高级功能，如自适应障碍物处理和决策增强奠定了基础。其次，提出了一种新颖的3D反应式导航策略，在森林火灾模拟中实现无碰撞移动，解决了无人机在这种场景下操作的独特挑战。

最后，报告提出了一种统一的控制方法，将无人机和无人驾驶地面车辆（UGVs）整合起来，为森林环境中的协同救援任务提供支持。每个阶段都提出了挑战，提出了控制模型，并通过数学和仿真证据进行了验证。本研究为改善无人机在自然灾害救援中的作用提供了实际价值和学术见解。 

---
# Generating and Customizing Robotic Arm Trajectories using Neural Networks 

**Title (ZH)**: 使用神经网络生成和定制机器人臂轨迹 

**Authors**: Andrej Lúčny, Matilde Antonj, Carlo Mazzola, Hana Hornáčková, Igor Farkaš  

**Link**: [PDF](https://arxiv.org/pdf/2506.20259)  

**Abstract**: We introduce a neural network approach for generating and customizing the trajectory of a robotic arm, that guarantees precision and repeatability. To highlight the potential of this novel method, we describe the design and implementation of the technique and show its application in an experimental setting of cognitive robotics. In this scenario, the NICO robot was characterized by the ability to point to specific points in space with precise linear movements, increasing the predictability of the robotic action during its interaction with humans. To achieve this goal, the neural network computes the forward kinematics of the robot arm. By integrating it with a generator of joint angles, another neural network was developed and trained on an artificial dataset created from suitable start and end poses of the robotic arm. Through the computation of angular velocities, the robot was characterized by its ability to perform the movement, and the quality of its action was evaluated in terms of shape and accuracy. Thanks to its broad applicability, our approach successfully generates precise trajectories that could be customized in their shape and adapted to different settings. 

**Abstract (ZH)**: 一种确保精确性和可重复性的机器人臂轨迹生成与定制的神经网络方法：应用与评估 

---
# Consensus-Driven Uncertainty for Robotic Grasping based on RGB Perception 

**Title (ZH)**: 基于RGB感知的共识驱动不确定性机器人抓取 

**Authors**: Eric C. Joyce, Qianwen Zhao, Nathaniel Burgdorfer, Long Wang, Philippos Mordohai  

**Link**: [PDF](https://arxiv.org/pdf/2506.20045)  

**Abstract**: Deep object pose estimators are notoriously overconfident. A grasping agent that both estimates the 6-DoF pose of a target object and predicts the uncertainty of its own estimate could avoid task failure by choosing not to act under high uncertainty. Even though object pose estimation improves and uncertainty quantification research continues to make strides, few studies have connected them to the downstream task of robotic grasping. We propose a method for training lightweight, deep networks to predict whether a grasp guided by an image-based pose estimate will succeed before that grasp is attempted. We generate training data for our networks via object pose estimation on real images and simulated grasping. We also find that, despite high object variability in grasping trials, networks benefit from training on all objects jointly, suggesting that a diverse variety of objects can nevertheless contribute to the same goal. 

**Abstract (ZH)**: 深度物体姿态估计器 notoriously 过于自信。一种同时估计目标物体6-自由度姿态并预测自身估计不确定性的方法可以避免在高不确定性下执行任务而导致的任务失败。尽管物体姿态估计进步显著且不确定性量化研究不断取得进展，但很少有研究将它们与后续的机器人抓取任务联系起来。我们提出了一种训练轻量级深度网络的方法，在抓取尝试之前预测由基于图像的姿态估计引导的抓取是否成功。我们通过对真实图像进行物体姿态估计和模拟抓取生成网络的训练数据。我们还发现，尽管抓取试验中物体存在高变异性，但网络从共同训练所有物体中受益，这表明多样化的物体仍然可以为同一个目标做出贡献。 

---
# Task Allocation of UAVs for Monitoring Missions via Hardware-in-the-Loop Simulation and Experimental Validation 

**Title (ZH)**: 基于硬件在环仿真与实验验证的无人机监测任务分配 

**Authors**: Hamza Chakraa, François Guérin, Edouard Leclercq, Dimitri Lefebvre  

**Link**: [PDF](https://arxiv.org/pdf/2506.20626)  

**Abstract**: This study addresses the optimisation of task allocation for Unmanned Aerial Vehicles (UAVs) within industrial monitoring missions. The proposed methodology integrates a Genetic Algorithms (GA) with a 2-Opt local search technique to obtain a high-quality solution. Our approach was experimentally validated in an industrial zone to demonstrate its efficacy in real-world scenarios. Also, a Hardware-in-the-loop (HIL) simulator for the UAVs team is introduced. Moreover, insights about the correlation between the theoretical cost function and the actual battery consumption and time of flight are deeply analysed. Results show that the considered costs for the optimisation part of the problem closely correlate with real-world data, confirming the practicality of the proposed approach. 

**Abstract (ZH)**: 本研究针对工业监测任务中无人机任务分配的优化进行了研究。提出的算法将遗传算法与2-Opt局部搜索技术集成以获得高质量的解决方案。该方法在工业区进行了实验验证，以展示其实用性。此外，还引入了无人机团队的硬件在环（HIL）仿真器。同时，深入分析了理论成本函数与实际电池消耗和飞行时间之间的关联。研究结果表明，考虑的成本与实际数据高度相关，验证了所提方法的实用性。 

---
