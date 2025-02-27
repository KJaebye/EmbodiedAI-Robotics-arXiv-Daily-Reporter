# ARENA: Adaptive Risk-aware and Energy-efficient NAvigation for Multi-Objective 3D Infrastructure Inspection with a UAV 

**Title (ZH)**: ARENA：自适应风险意识和能效导航的多目标3D基础设施无人机检测 

**Authors**: David-Alexandre Poissant, Alexis Lussier Desbiens, François Ferland, Louis Petit  

**Link**: [PDF](https://arxiv.org/pdf/2502.19401)  

**Abstract**: Autonomous robotic inspection missions require balancing multiple conflicting objectives while navigating near costly obstacles. Current multi-objective path planning (MOPP) methods struggle to adapt to evolving risks like localization errors, weather, battery state, and communication issues. This letter presents an Adaptive Risk-aware and Energy-efficient NAvigation (ARENA) MOPP approach for UAVs in complex 3D environments. Our method enables online trajectory adaptation by optimizing safety, time, and energy using 4D NURBS representation and a genetic-based algorithm to generate the Pareto front. A novel risk-aware voting algorithm ensures adaptivity. Simulations and real-world tests demonstrate the planner's ability to produce diverse, optimized trajectories covering 95% or more of the range defined by single-objective benchmarks and its ability to estimate power consumption with a mean error representing 14% of the full power range. The ARENA framework enhances UAV autonomy and reliability in critical, evolving 3D missions. 

**Abstract (ZH)**: 自主巡检机器人任务要求在接近昂贵障碍物时权衡多个相互冲突的目标。当前的多目标路径规划方法难以适应如定位误差、天气、电池状态和通信问题等不断变化的风险。本信提出了一种适应风险感知和能量高效导航（ARENA）的多目标路径规划方法，适用于具有复杂3D环境的无人机。该方法通过使用4D NURBS表示和基于遗传的算法优化安全、时间与能量，并生成帕累托前沿，实现了在线轨迹适应。一种新颖的风险感知投票算法确保了系统的适应性。仿真和实地测试展示了该规划器生成多样化、优化轨迹的能力，覆盖单目标基准定义范围的95%以上，并且能够以接近14%的最大功率范围误差估计功率消耗。ARENA框架提升了无人机在关键、动态3D任务中的自主性和可靠性。 

---
# Surface-Based Manipulation 

**Title (ZH)**: 基于表面的操作 

**Authors**: Ziqiao Wang, Serhat Demirtas, Fabio Zuliani, Jamie Paik  

**Link**: [PDF](https://arxiv.org/pdf/2502.19389)  

**Abstract**: Intelligence lies not only in the brain but in the body. The shape of our bodies can influence how we think and interact with the physical world. In robotics research, interacting with the physical world is crucial as it allows robots to manipulate objects in various real-life scenarios. Conventional robotic manipulation strategies mainly rely on finger-shaped end effectors. However, achieving stable grasps on fragile, deformable, irregularly shaped, or slippery objects is challenging due to difficulties in establishing stable force or geometric constraints.
Here, we present surface-based manipulation strategies that diverge from classical grasping approaches, using with flat surfaces as minimalist end-effectors. By changing the position and orientation of these surfaces, objects can be translated, rotated and even flipped across the surface using closed-loop control strategies. Since this method does not rely on stable grasp, it can adapt to objects of various shapes, sizes, and stiffness levels, even enabling the manipulation the shape of deformable objects. Our results provide a new perspective for solving complex manipulation problems. 

**Abstract (ZH)**: 智能不仅存在于大脑中，也存在于身体中。我们的身体形状可以影响我们的思维和与物理世界的互动。在机器人研究中，与物理世界的交互至关重要，因为它使机器人能够在各种现实情境中操作物体。传统的机器人操作策略主要依赖于手指状的末端执行器。然而，对脆弱、可变形、形状不规则或滑溜的物体实现稳定的抓握极具挑战性，原因在于难以建立稳定的力或几何约束。

在这里，我们提出了基于表面的操作策略，这些策略偏离了传统的抓取方法，使用平坦表面作为简约的末端执行器。通过改变这些表面的位置和方向，可以利用闭环控制策略将物体在表面进行平移、旋转甚至翻转。由于这种方法无需依赖稳定的抓握，因此它可以适应各种形状、大小和刚度级别的物体，甚至能够操作可变形物体的形状。我们的研究结果为解决复杂操作问题提供了新的视角。 

---
# Leg Exoskeleton Odometry using a Limited FOV Depth Sensor 

**Title (ZH)**: 基于有限视野深度传感器的下肢外骨骼里程计 

**Authors**: Fabio Elnecave Xavier, Matis Viozelange, Guillaume Burger, Marine Pétriaux, Jean-Emmanuel Deschaud, François Goulette  

**Link**: [PDF](https://arxiv.org/pdf/2502.19237)  

**Abstract**: For leg exoskeletons to operate effectively in real-world environments, they must be able to perceive and understand the terrain around them. However, unlike other legged robots, exoskeletons face specific constraints on where depth sensors can be mounted due to the presence of a human user. These constraints lead to a limited Field Of View (FOV) and greater sensor motion, making odometry particularly challenging. To address this, we propose a novel odometry algorithm that integrates proprioceptive data from the exoskeleton with point clouds from a depth camera to produce accurate elevation maps despite these limitations. Our method builds on an extended Kalman filter (EKF) to fuse kinematic and inertial measurements, while incorporating a tailored iterative closest point (ICP) algorithm to register new point clouds with the elevation map. Experimental validation with a leg exoskeleton demonstrates that our approach reduces drift and enhances the quality of elevation maps compared to a purely proprioceptive baseline, while also outperforming a more traditional point cloud map-based variant. 

**Abstract (ZH)**: 基于深度相机点云与 proprioceptive 数据的新型腿部外骨骼 odometer 算法 

---
# CPG-Based Manipulation with Multi-Module Origami Robot Surface 

**Title (ZH)**: 基于CPG的多模块Origami机器人表面 manipulation 

**Authors**: Yuhao Jiang, Serge El Asmar, Ziqiao Wang, Serhat Demirtas, Jamie Paik  

**Link**: [PDF](https://arxiv.org/pdf/2502.19218)  

**Abstract**: Robotic manipulators often face challenges in handling objects of different sizes and materials, limiting their effectiveness in practical applications. This issue is particularly pronounced when manipulating meter-scale objects or those with varying stiffness, as traditional gripping techniques and strategies frequently prove inadequate. In this letter, we introduce a novel surface-based multi-module robotic manipulation framework that utilizes a Central Pattern Generator (CPG)-based motion generator, combined with a simulation-based optimization method to determine the optimal manipulation parameters for a multi-module origami robotic surface (Ori-Pixel). This approach allows for the manipulation of objects ranging from centimeters to meters in size, with varying stiffness and shape. The optimized CPG parameters are tested through both dynamic simulations and a series of prototype experiments involving a wide range of objects differing in size, weight, shape, and material, demonstrating robust manipulation capabilities. 

**Abstract (ZH)**: 基于表面的多模块机器人 manipulation 框架：用于不同大小和刚度物体的优化 Central Pattern Generator 参数研究 

---
# Increasing the Task Flexibility of Heavy-Duty Manipulators Using Visual 6D Pose Estimation of Objects 

**Title (ZH)**: 使用物体6D姿态估计提高重型 manipulator 作业灵活性 

**Authors**: Petri Mäkinen, Pauli Mustalahti, Tuomo Kivelä, Jouni Mattila  

**Link**: [PDF](https://arxiv.org/pdf/2502.19169)  

**Abstract**: Recent advances in visual 6D pose estimation of objects using deep neural networks have enabled novel ways of vision-based control for heavy-duty robotic applications. In this study, we present a pipeline for the precise tool positioning of heavy-duty, long-reach (HDLR) manipulators using advanced machine vision. A camera is utilized in the so-called eye-in-hand configuration to estimate directly the poses of a tool and a target object of interest (OOI). Based on the pose error between the tool and the target, along with motion-based calibration between the camera and the robot, precise tool positioning can be reliably achieved using conventional robotic modeling and control methods prevalent in the industry. The proposed methodology comprises orientation and position alignment based on the visually estimated OOI poses, whereas camera-to-robot calibration is conducted based on motion utilizing visual SLAM. The methods seek to avert the inaccuracies resulting from rigid-body--based kinematics of structurally flexible HDLR manipulators via image-based algorithms. To train deep neural networks for OOI pose estimation, only synthetic data are utilized. The methods are validated in a real-world setting using an HDLR manipulator with a 5 m reach. The experimental results demonstrate that an image-based average tool positioning error of less than 2 mm along the non-depth axes is achieved, which facilitates a new way to increase the task flexibility and automation level of non-rigid HDLR manipulators. 

**Abstract (ZH)**: 基于深度神经网络的物体视化6D姿态估计在重型长臂 manipulator 精确工具定位中的应用研究 

---
# RL-OGM-Parking: Lidar OGM-Based Hybrid Reinforcement Learning Planner for Autonomous Parking 

**Title (ZH)**: 基于激光雷达OGM的混合强化学习自主泊车规划方法 

**Authors**: Zhitao Wang, Zhe Chen, Mingyang Jiang, Tong Qin, Ming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18846)  

**Abstract**: Autonomous parking has become a critical application in automatic driving research and development. Parking operations often suffer from limited space and complex environments, requiring accurate perception and precise maneuvering. Traditional rule-based parking algorithms struggle to adapt to diverse and unpredictable conditions, while learning-based algorithms lack consistent and stable performance in various scenarios. Therefore, a hybrid approach is necessary that combines the stability of rule-based methods and the generalizability of learning-based methods. Recently, reinforcement learning (RL) based policy has shown robust capability in planning tasks. However, the simulation-to-reality (sim-to-real) transfer gap seriously blocks the real-world deployment. To address these problems, we employ a hybrid policy, consisting of a rule-based Reeds-Shepp (RS) planner and a learning-based reinforcement learning (RL) planner. A real-time LiDAR-based Occupancy Grid Map (OGM) representation is adopted to bridge the sim-to-real gap, leading the hybrid policy can be applied to real-world systems seamlessly. We conducted extensive experiments both in the simulation environment and real-world scenarios, and the result demonstrates that the proposed method outperforms pure rule-based and learning-based methods. The real-world experiment further validates the feasibility and efficiency of the proposed method. 

**Abstract (ZH)**: 自主泊车已成为自动驾驶研究与开发中的关键应用。泊车操作常受限于有限的空间和复杂的环境，需要准确的感知和精确的操作。传统的基于规则的泊车算法难以适应多变且不可预测的条件，而基于学习的算法在各种场景中缺乏一致且稳定的表现。因此，有必要结合基于规则方法的稳定性和基于学习方法的一般性，采用一种混合策略。最近，基于强化学习（RL）的策略在规划任务中显示出强大的能力。然而，模拟到现实（sim-to-real）的转移差距严重阻碍了其实用性部署。为解决这些问题，我们采用了一种混合策略，该策略结合了基于规则的Reeds-Shepp（RS）规划器和基于学习的强化学习（RL）规划器。采用实时LiDAR基于的占用网格地图（OGM）表示来弥合模拟到现实的差距，使得混合策略能够无缝应用于现实系统。我们在模拟环境和真实场景中进行了广泛的实验，结果表明所提出的方法优于纯基于规则和基于学习的方法。进一步的真实场景实验验证了所提出方法的可行性和效率。标题：

一种结合Reeds-Shepp规划器和强化学习规划器的混合自主泊车方法 

---
# Interpretable Data-Driven Ship Dynamics Model: Enhancing Physics-Based Motion Prediction with Parameter Optimization 

**Title (ZH)**: 可解释的数据驱动船舶动力学模型：基于参数优化的物理基础运动预测增强 

**Authors**: Papandreou Christos, Mathioudakis Michail, Stouraitis Theodoros, Iatropoulos Petros, Nikitakis Antonios, Stavros Paschalakis, Konstantinos Kyriakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.18696)  

**Abstract**: The deployment of autonomous navigation systems on ships necessitates accurate motion prediction models tailored to individual vessels. Traditional physics-based models, while grounded in hydrodynamic principles, often fail to account for ship-specific behaviors under real-world conditions. Conversely, purely data-driven models offer specificity but lack interpretability and robustness in edge cases. This study proposes a data-driven physics-based model that integrates physics-based equations with data-driven parameter optimization, leveraging the strengths of both approaches to ensure interpretability and adaptability. The model incorporates physics-based components such as 3-DoF dynamics, rudder, and propeller forces, while parameters such as resistance curve and rudder coefficients are optimized using synthetic data. By embedding domain knowledge into the parameter optimization process, the fitted model maintains physical consistency. Validation of the approach is realized with two container ships by comparing, both qualitatively and quantitatively, predictions against ground-truth trajectories. The results demonstrate significant improvements, in predictive accuracy and reliability, of the data-driven physics-based models over baseline physics-based models tuned with traditional marine engineering practices. The fitted models capture ship-specific behaviors in diverse conditions with their predictions being, 51.6% (ship A) and 57.8% (ship B) more accurate, 72.36% (ship A) and 89.67% (ship B) more consistent. 

**Abstract (ZH)**: 自主导航系统在船舶上的部署需要针对单艘船舶定制的准确运动预测模型。传统基于物理的模型虽然基于水动力原理，但在实际条件下往往无法准确反映船舶特定行为。相比之下，纯数据驱动的模型虽然具有特定性，但在边缘情况下缺乏可解释性和鲁棒性。本研究提出了一种结合基于物理的方程与数据驱动参数优化的基于物理的数据驱动模型，利用两者的优点确保模型的可解释性和适应性。该模型整合了基于物理的组件如3-DoF动力学、舵力和推进力，而阻力曲线参数和舵系数则通过合成数据进行优化。通过将领域知识嵌入参数优化过程，拟合模型保持了物理一致性。通过将两种集装箱船的实际轨迹与预测结果进行定性和定量比较，验证了该方法的有效性。结果表明，基于物理的数据驱动模型在预测准确性和可靠性方面显著优于传统海洋工程实践中调优的传统基于物理的模型。拟合模型在不同条件下捕捉到船特有的行为，预测准确性和一致性分别提高了51.6%（船A）和57.8%（船B），72.36%（船A）和89.67%（船B）。 

---
# Learning Autonomy: Off-Road Navigation Enhanced by Human Input 

**Title (ZH)**: 自主学习：由人类输入增强的离线导航 

**Authors**: Akhil Nagariya, Dimitar Filev, Srikanth Saripalli, Gaurav Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2502.18760)  

**Abstract**: In the area of autonomous driving, navigating off-road terrains presents a unique set of challenges, from unpredictable surfaces like grass and dirt to unexpected obstacles such as bushes and puddles. In this work, we present a novel learning-based local planner that addresses these challenges by directly capturing human driving nuances from real-world demonstrations using only a monocular camera. The key features of our planner are its ability to navigate in challenging off-road environments with various terrain types and its fast learning capabilities. By utilizing minimal human demonstration data (5-10 mins), it quickly learns to navigate in a wide array of off-road conditions. The local planner significantly reduces the real world data required to learn human driving preferences. This allows the planner to apply learned behaviors to real-world scenarios without the need for manual fine-tuning, demonstrating quick adjustment and adaptability in off-road autonomous driving technology. 

**Abstract (ZH)**: 自主驾驶领域中的非铺装地形导航面临独特的挑战，包括不可预测的地面如草地和泥土，以及意外的障碍物如灌木和水坑。本文提出了一种新颖的学习型局部规划器，通过仅使用单目摄像头实时捕获真实驾驶示范中的人类驾驶细节，来应对这些挑战。该规划器的关键特性在于其能够在多种地形类型的挑战性非铺装环境中导航，并且具有快速学习能力。借助少量的人工示范数据（5-10分钟），它能够迅速学会在各种非铺装条件下导航。该局部规划器大大减少了学习人类驾驶偏好的所需真实世界数据量，使得规划器能够将学到的行为应用到真实世界场景中，而无需手动微调，从而展示了在非铺装自主驾驶技术中的快速调整和适应能力。 

---
