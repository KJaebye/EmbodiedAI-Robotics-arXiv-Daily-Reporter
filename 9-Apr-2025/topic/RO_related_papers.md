# Underwater Robotic Simulators Review for Autonomous System Development 

**Title (ZH)**: 水下机器人仿真器综述：自主系统开发 

**Authors**: Sara Aldhaheri, Yang Hu, Yongchang Xie, Peng Wu, Dimitrios Kanoulas, Yuanchang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06245)  

**Abstract**: The increasing complexity of underwater robotic systems has led to a surge in simulation platforms designed to support perception, planning, and control tasks in marine environments. However, selecting the most appropriate underwater robotic simulator (URS) remains a challenge due to wide variations in fidelity, extensibility, and task suitability. This paper presents a comprehensive review and comparative analysis of five state-of-the-art, ROS-compatible, open-source URSs: Stonefish, DAVE, HoloOcean, MARUS, and UNav-Sim. Each simulator is evaluated across multiple criteria including sensor fidelity, environmental realism, sim-to-real capabilities, and research impact. We evaluate them across architectural design, sensor and physics modeling, task capabilities, and research impact. Additionally, we discuss ongoing challenges in sim-to-real transfer and highlight the need for standardization and benchmarking in the field. Our findings aim to guide practitioners in selecting effective simulation environments and inform future development of more robust and transferable URSs. 

**Abstract (ZH)**: 随着水下机器人系统复杂性的增加，设计用于支持海洋环境中感知、规划和控制任务的模拟平台的需求急剧上升。然而，选择最合适的水下机器人模拟器（URS）仍然具有挑战性，因为这些模拟器在保真度、可拓展性和任务适用性方面存在广泛差异。本文对五个最先进的、兼容ROS的开源URS——Stonefish、DAVE、HoloOcean、MARUS和UNav-Sim进行全面回顾和比较分析。各模拟器在传感器保真度、环境现实度、模拟到现实的能力和研究影响等多个方面进行评估。此外，我们讨论了模拟到现实转移中面临的持续挑战，并强调了该领域标准化和基准测试的必要性。我们的研究结果旨在指导实践者选择有效的模拟环境，并为未来开发更稳健和可转移的URS提供信息。 

---
# Exploring Adversarial Obstacle Attacks in Search-based Path Planning for Autonomous Mobile Robots 

**Title (ZH)**: 基于搜索的路径规划中对抗障碍攻击的探索 

**Authors**: Adrian Szvoren, Jianwei Liu, Dimitrios Kanoulas, Nilufer Tuptuk  

**Link**: [PDF](https://arxiv.org/pdf/2504.06154)  

**Abstract**: Path planning algorithms, such as the search-based A*, are a critical component of autonomous mobile robotics, enabling robots to navigate from a starting point to a destination efficiently and safely. We investigated the resilience of the A* algorithm in the face of potential adversarial interventions known as obstacle attacks. The adversary's goal is to delay the robot's timely arrival at its destination by introducing obstacles along its original path.
We developed malicious software to execute the attacks and conducted experiments to assess their impact, both in simulation using TurtleBot in Gazebo and in real-world deployment with the Unitree Go1 robot. In simulation, the attacks resulted in an average delay of 36\%, with the most significant delays occurring in scenarios where the robot was forced to take substantially longer alternative paths. In real-world experiments, the delays were even more pronounced, with all attacks successfully rerouting the robot and causing measurable disruptions. These results highlight that the algorithm's robustness is not solely an attribute of its design but is significantly influenced by the operational environment. For example, in constrained environments like tunnels, the delays were maximized due to the limited availability of alternative routes. 

**Abstract (ZH)**: 基于搜索的A*路径规划算法在面对敌对障碍攻击时的鲁棒性研究：从起点到目的地的高效安全导航算法在遭遇敌对障碍攻击时的鲁棒性研究 

---
# A ROS2-based software library for inverse dynamics computation 

**Title (ZH)**: 基于ROS2的逆动力学计算软件库 

**Authors**: Vincenzo Petrone, Enrico Ferrentino, Pasquale Chiacchio  

**Link**: [PDF](https://arxiv.org/pdf/2504.06106)  

**Abstract**: Inverse dynamics computation is a critical component in robot control, planning and simulation, enabling the calculation of joint torques required to achieve a desired motion. This paper presents a ROS2-based software library designed to solve the inverse dynamics problem for robotic systems. The library is built around an abstract class with three concrete implementations: one for simulated robots and two for real UR10 and Franka robots. This contribution aims to provide a flexible, extensible, robot-agnostic solution to inverse dynamics, suitable for both simulation and real-world scenarios involving planning and control applications. The related software is available at this https URL. 

**Abstract (ZH)**: 基于ROS2的机器人逆动力学计算软件库的设计与实现 

---
# Robust Statistics vs. Machine Learning vs. Bayesian Inference: Insights into Handling Faulty GNSS Measurements in Field Robotics 

**Title (ZH)**: 稳健统计学 vs. 机器学习 vs. 贝叶斯推断：处理现场机器人中故障GNSS测量的见解 

**Authors**: Haoming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06015)  

**Abstract**: This paper presents research findings on handling faulty measurements (i.e., outliers) of global navigation satellite systems (GNSS) for robot localization under adverse signal conditions in field applications, where raw GNSS data are frequently corrupted due to environmental interference such as multipath, signal blockage, or non-line-of-sight conditions. In this context, we investigate three strategies applied specifically to GNSS pseudorange observations: robust statistics for error mitigation, machine learning for faulty measurement prediction, and Bayesian inference for noise distribution approximation. Since previous studies have provided limited insight into the theoretical foundations and practical evaluations of these three methodologies within a unified problem statement (i.e., state estimation using ranging sensors), we conduct extensive experiments using real-world sensor data collected in diverse urban environments. Our goal is to examine both established techniques and newly proposed methods, thereby advancing the understanding of how to handle faulty range measurements, such as GNSS, for robust, long-term robot localization. In addition to presenting successful results, this work highlights critical observations and open questions to motivate future research in robust state estimation. 

**Abstract (ZH)**: 本文研究了在恶劣信号条件下处理全球导航卫星系统（GNSS）故障测量（即离群值）以实现机器人定位的方法，探讨了适用于GNSS伪距观测的三种策略：稳健统计法用于误差缓解、机器学习法用于故障测量预测以及贝叶斯推断法用于噪声分布逼近。为了统一地探讨这些方法在基于范围传感器的状态估计中的理论基础和实践评估，本文使用多种城市环境中采集的真实传感器数据进行了广泛的实验。本文旨在研究既有的技术和新提出的方法，旨在增进对如何处理故障范围测量（如GNSS测量）以实现鲁棒的长期机器人定位的理解。除展示成功结果外，本文还强调了关键观察结果和开放问题，以激发鲁棒状态估计的未来研究。 

---
# Learning-enhanced electronic skin for tactile sensing on deformable surface based on electrical impedance tomography 

**Title (ZH)**: 基于电气阻抗断层成像的增强学习电子皮肤在变形表面的触觉感知 

**Authors**: Huazhi Dong, Xiaopeng Wu, Delin Hu, Zhe Liu, Francesco Giorgio-Serchi, Yunjie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05987)  

**Abstract**: Electrical Impedance Tomography (EIT)-based tactile sensors offer cost-effective and scalable solutions for robotic sensing, especially promising for soft robots. However a major issue of EIT-based tactile sensors when applied in highly deformable objects is their performance degradation due to surface deformations. This limitation stems from their inherent sensitivity to strain, which is particularly exacerbated in soft bodies, thus requiring dedicated data interpretation to disentangle the parameter being measured and the signal deriving from shape changes. This has largely limited their practical implementations. This paper presents a machine learning-assisted tactile sensing approach to address this challenge by tracking surface deformations and segregating this contribution in the signal readout during tactile sensing. We first capture the deformations of the target object, followed by tactile reconstruction using a deep learning model specifically designed to process and fuse EIT data and deformation information. Validations using numerical simulations achieved high correlation coefficients (0.9660 - 0.9999), peak signal-to-noise ratios (28.7221 - 55.5264 dB) and low relative image errors (0.0107 - 0.0805). Experimental validations, using a hydrogel-based EIT e-skin under various deformation scenarios, further demonstrated the effectiveness of the proposed approach in real-world settings. The findings could underpin enhanced tactile interaction in soft and highly deformable robotic applications. 

**Abstract (ZH)**: 基于电气阻抗成像(EIT)的触觉传感器：基于机器学习的触觉感知方法解决高度可变形物体表面变形导致的性能退化问题 

---
# Adaptive RISE Control for Dual-Arm Unmanned Aerial Manipulator Systems with Deep Neural Networks 

**Title (ZH)**: 基于深度神经网络的双臂无人机操作系统自适应RISE控制 

**Authors**: Yang Wang, Hai Yu, Shizhen Wu, Zhichao Yang, Jianda Han, Yongchun Fang, Xiao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05985)  

**Abstract**: The unmanned aerial manipulator system, consisting of a multirotor UAV (unmanned aerial vehicle) and a manipulator, has attracted considerable interest from researchers. Nevertheless, the operation of a dual-arm manipulator poses a dynamic challenge, as the CoM (center of mass) of the system changes with manipulator movement, potentially impacting the multirotor UAV. Additionally, unmodeled effects, parameter uncertainties, and external disturbances can significantly degrade control performance, leading to unforeseen dangers. To tackle these issues, this paper proposes a nonlinear adaptive RISE (robust integral of the sign of the error) controller based on DNN (deep neural network). The first step involves establishing the kinematic and dynamic model of the dual-arm aerial manipulator. Subsequently, the adaptive RISE controller is proposed with a DNN feedforward term to effectively address both internal and external challenges. By employing Lyapunov techniques, the asymptotic convergence of the tracking error signals are guaranteed rigorously. Notably, this paper marks a pioneering effort by presenting the first DNN-based adaptive RISE controller design accompanied by a comprehensive stability analysis. To validate the practicality and robustness of the proposed control approach, several groups of actual hardware experiments are conducted. The results confirm the efficacy of the developed methodology in handling real-world scenarios, thereby offering valuable insights into the performance of the dual-arm aerial manipulator system. 

**Abstract (ZH)**: 基于深度神经网络的非线性自适应RISE控制方法：双臂无人机 manipulator 系统的轨迹跟踪控制 

---
# A Corrector-aided Look-ahead Distance-based Guidance for Reference Path Following with an Efficient Midcourse Guidance Strategy 

**Title (ZH)**: 基于校正辅助前瞻距离导向的参考路径跟随及高效中程导向策略 

**Authors**: Reva Dhillon, Agni Ravi Deepa, Hrishav Das, Subham Basak, Satadal Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2504.05975)  

**Abstract**: Efficient path-following is crucial in most of the applications of autonomous vehicles (UxV). Among various guidance strategies presented in literature, look-ahead distance ($L_1$)-based guidance method has received significant attention due to its ease in implementation and ability to maintain a low cross-track error while following simpler reference paths and generate bounded lateral acceleration commands. However, the constant value of $L_1$ becomes problematic when the UxV is far away from the reference path and also produce higher cross-track error while following complex reference paths having high variation in radius of curvature. To address these challenges, the notion of look-ahead distance is leveraged in a novel way to develop a two-phase guidance strategy. Initially, when the UxV is far from the reference path, an optimized $L_1$ selection strategy is developed to guide the UxV toward the reference path in order to maintain minimal lateral acceleration command. Once the vehicle reaches a close vicinity of the reference path, a novel notion of corrector point is incorporated in the constant $L_1$-based guidance scheme to generate the lateral acceleration command that effectively reduces the root mean square of the cross-track error thereafter. Simulation results demonstrate that this proposed corrector point and look-ahead point pair-based guidance strategy along with the developed midcourse guidance scheme outperforms the conventional constant $L_1$ guidance scheme both in terms of feasibility and measures of effectiveness like cross-track error and lateral acceleration requirements. 

**Abstract (ZH)**: 基于前瞻距离的两阶段导航策略在自主车辆中的高效路径跟踪 

---
# Collision-free landing of multiple UAVs on moving ground vehicles using time-varying control barrier functions 

**Title (ZH)**: 基于时间varying控制障碍函数的多无人机在移动地面车辆上无碰撞着陆 

**Authors**: Viswa Narayanan Sankaranarayanan, Akshit Saradagi, Sumeet Satpute, George Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2504.05939)  

**Abstract**: In this article, we present a centralized approach for the control of multiple unmanned aerial vehicles (UAVs) for landing on moving unmanned ground vehicles (UGVs) using control barrier functions (CBFs). The proposed control framework employs two kinds of CBFs to impose safety constraints on the UAVs' motion. The first class of CBFs (LCBF) is a three-dimensional exponentially decaying function centered above the landing platform, designed to safely and precisely land UAVs on the UGVs. The second set is a spherical CBF (SCBF), defined between every pair of UAVs, which avoids collisions between them. The LCBF is time-varying and adapts to the motions of the UGVs. In the proposed CBF approach, the control input from the UAV's nominal tracking controller designed to reach the landing platform is filtered to choose a minimally-deviating control input that ensures safety (as defined by the CBFs). As the control inputs of every UAV are shared in establishing multiple CBF constraints, we prove that the control inputs are shared without conflict in rendering the safe sets forward invariant. The performance of the control framework is validated through a simulated scenario involving three UAVs landing on three moving targets. 

**Abstract (ZH)**: 本文提出了一种集中式控制方法，使用控制障碍函数（CBFs）使多个无人机（UAVs）能够在移动地面车辆（UGVs）上着陆。提出的控制框架使用两种类型的CBFs来对无人机的运动施加安全约束。第一类CBF（LCBF）是一种三维指数衰减函数，中心位于着陆平台上方，旨在安全且精确地使无人机着陆在UGVs上。第二类是球形CBF（SCBF），定义在每对无人机之间，避免它们之间的碰撞。LCBF是时间变化的，并适应UGVs的运动。在提出的CBF方法中，从无人机名义跟踪控制器设计的用于到达着陆平台的控制输入被过滤，以选择一个最小偏离的安全控制输入。由于每架无人机的控制输入用于建立多个CBF约束，我们证明这些控制输入在提供安全集前不变时可以共享而无冲突。通过涉及三架无人机在三个移动目标上着陆的仿真场景验证了控制框架的性能。 

---
# Accelerated Reeds-Shepp and Under-Specified Reeds-Shepp Algorithms for Mobile Robot Path Planning 

**Title (ZH)**: 加速Reeds-Shepp和欠定Reeds-Shepp算法在移动机器人路径规划中的应用 

**Authors**: Ibrahim Ibrahim, Wilm Decré, Jan Swevers  

**Link**: [PDF](https://arxiv.org/pdf/2504.05921)  

**Abstract**: In this study, we present a simple and intuitive method for accelerating optimal Reeds-Shepp path computation. Our approach uses geometrical reasoning to analyze the behavior of optimal paths, resulting in a new partitioning of the state space and a further reduction in the minimal set of viable paths. We revisit and reimplement classic methodologies from the literature, which lack contemporary open-source implementations, to serve as benchmarks for evaluating our method. Additionally, we address the under-specified Reeds-Shepp planning problem where the final orientation is unspecified. We perform exhaustive experiments to validate our solutions. Compared to the modern C++ implementation of the original Reeds-Shepp solution in the Open Motion Planning Library, our method demonstrates a 15x speedup, while classic methods achieve a 5.79x speedup. Both approaches exhibit machine-precision differences in path lengths compared to the original solution. We release our proposed C++ implementations for both the accelerated and under-specified Reeds-Shepp problems as open-source code. 

**Abstract (ZH)**: 本研究提出一种简单直观的方法来加速最优Reeds-Shepp路径计算。我们的方法通过几何推理分析最优路径的行为，从而获得新的状态空间分区，并进一步减少可行路径的最小集合。我们回顾并重新实现了文献中缺乏现代开源实现的经典方法，作为评估我们方法的基准。此外，我们解决了一个未完全指定的Reeds-Shepp规划问题，其中最终方向未指定。我们进行了详尽的实验来验证我们的解决方案。与Open Motion Planning Library中现代C++实现的原始Reeds-Shepp解决方案相比，我们的方法显示出15倍的速度提升，而经典方法则实现了5.79倍的速度提升。两种方法在路径长度上与原始解决方案相比都存在机器精度上的差异。我们开源发布了我们为加速和未完全指定的Reeds-Shepp问题提出的C++实现。 

---
# Jointly-optimized Trajectory Generation and Camera Control for 3D Coverage Planning 

**Title (ZH)**: jointly优化的轨迹生成与相机控制应用于3D覆盖规划 

**Authors**: Savvas Papaioannou, Panayiotis Kolios, Theocharis Theocharides, Christos G. Panayiotou, Marios M. Polycarpou  

**Link**: [PDF](https://arxiv.org/pdf/2504.05887)  

**Abstract**: This work proposes a jointly optimized trajectory generation and camera control approach, enabling an autonomous agent, such as an unmanned aerial vehicle (UAV) operating in 3D environments, to plan and execute coverage trajectories that maximally cover the surface area of a 3D object of interest. Specifically, the UAV's kinematic and camera control inputs are jointly optimized over a rolling planning horizon to achieve complete 3D coverage of the object. The proposed controller incorporates ray-tracing into the planning process to simulate the propagation of light rays, thereby determining the visible parts of the object through the UAV's camera. This integration enables the generation of precise look-ahead coverage trajectories. The coverage planning problem is formulated as a rolling finite-horizon optimal control problem and solved using mixed-integer programming techniques. Extensive real-world and synthetic experiments validate the performance of the proposed approach. 

**Abstract (ZH)**: 本研究提出了一种联合优化的轨迹生成和相机控制方法，使得像无人驾驶飞行器（UAV）这样的自主代理能够在3D环境中规划和执行最大限度覆盖目标3D物体表面的轨迹。具体而言，通过在滚动规划 horizons 上联合优化UAV的动力学和相机控制输入，实现对物体的全面3D覆盖。所提出的控制器将射线追踪集成到规划过程中，用于模拟光线的传播，从而确定通过UAV相机可见的物体部分。这种集成使得能够生成精确的前瞻覆盖轨迹。将覆盖规划问题表述为滚动有限时间最优控制问题，并使用混合整数规划技术求解。大量实际和合成实验验证了所提出方法的性能。 

---
# Rolling Horizon Coverage Control with Collaborative Autonomous Agents 

**Title (ZH)**: 滚动时间窗覆盖控制与协作自主代理 

**Authors**: Savvas Papaioannou, Panayiotis Kolios, Theocharis Theocharides, Christos G. Panayiotou, Marios M. Polycarpou  

**Link**: [PDF](https://arxiv.org/pdf/2504.05883)  

**Abstract**: This work proposes a coverage controller that enables an aerial team of distributed autonomous agents to collaboratively generate non-myopic coverage plans over a rolling finite horizon, aiming to cover specific points on the surface area of a 3D object of interest. The collaborative coverage problem, formulated, as a distributed model predictive control problem, optimizes the agents' motion and camera control inputs, while considering inter-agent constraints aiming at reducing work redundancy. The proposed coverage controller integrates constraints based on light-path propagation techniques to predict the parts of the object's surface that are visible with regard to the agents' future anticipated states. This work also demonstrates how complex, non-linear visibility assessment constraints can be converted into logical expressions that are embedded as binary constraints into a mixed-integer optimization framework. The proposed approach has been demonstrated through simulations and practical applications for inspecting buildings with unmanned aerial vehicles (UAVs). 

**Abstract (ZH)**: 本工作提出了一种覆盖控制器，使分布式自主代理组成的空中团队能够在滚动有限时间内协作生成非短视的覆盖计划，旨在覆盖感兴趣三维对象表面的特定点。该协作覆盖问题被形式化为分布式模型预测控制问题，优化代理的运动和相机控制输入，同时考虑到代理间的约束以减少工作冗余。所提出的覆盖控制器整合了基于光路传播技术的约束，以预测代理未来状态下可见的对象表面部分。本工作还展示了如何将复杂的非线性可见性评估约束转换为逻辑表达式，并嵌入到混合整数优化框架中。所提出的方法已通过无人机对建筑物进行检查的仿真和实际应用进行了验证。 

---
# SAP-CoPE: Social-Aware Planning using Cooperative Pose Estimation with Infrastructure Sensor Nodes 

**Title (ZH)**: 基于协作姿态估计和基础设施传感器节点的社会感知规划方法(SAP-CoPE) 

**Authors**: Minghao Ning, Yufeng Yang, Shucheng Huang, Jiaming Zhong, Keqi Shu, Chen Sun, Ehsan Hashemi, Amir Khajepour  

**Link**: [PDF](https://arxiv.org/pdf/2504.05727)  

**Abstract**: Autonomous driving systems must operate safely in human-populated indoor environments, where challenges such as limited perception and occlusion sensitivity arise when relying solely on onboard sensors. These factors generate difficulties in the accurate recognition of human intentions and the generation of comfortable, socially aware trajectories. To address these issues, we propose SAP-CoPE, a social-aware planning framework that integrates cooperative infrastructure with a novel 3D human pose estimation method and a model predictive control-based controller. This real-time framework formulates an optimization problem that accounts for uncertainty propagation in the camera projection matrix while ensuring human joint coherence. The proposed method is adaptable to single- or multi-camera configurations and can incorporate sparse LiDAR point-cloud data. To enhance safety and comfort in human environments, we integrate a human personal space field based on human pose into a model predictive controller, enabling the system to navigate while avoiding discomfort zones. Extensive evaluations in both simulated and real-world settings demonstrate the effectiveness of our approach in generating socially aware trajectories for autonomous systems. 

**Abstract (ZH)**: 自主驾驶系统必须在包含人群的室内环境中安全运行，依赖车载传感器时面临感知受限和遮挡敏感等问题，这给精确识别人类意图和生成舒适、社会意识强的轨迹带来了困难。为应对这些问题，我们提出了一种结合协作基础设施的SAP-CoPE社会意识规划框架，该框架集成了一种新颖的3D人体姿态估计方法和基于模型预测控制的控制器。该实时框架在确保人体关节一致性的同时，考虑了相机投影矩阵中不确定性的传播。该方法适用于单摄像头或多摄像头配置，并可整合稀疏的LiDAR点云数据。为了在人类环境中增强安全性和舒适性，我们在模型预测控制器中结合了基于人体姿态的人类个人空间场，使系统能够在避免不适区域的同时导航。在模拟和实地环境中的广泛评估表明，我们的方法能够为自主系统生成社会意识强的轨迹。 

---
# Experimental Evaluation of Precise Placement of the Hollow Object with Asymmetric Pivot Manipulation 

**Title (ZH)**: 不对称支点操控下空心物体精准定位的实验评价 

**Authors**: Jinseong Park, Jeong-Jung Kim, Doo-Yeol Koh  

**Link**: [PDF](https://arxiv.org/pdf/2504.05665)  

**Abstract**: In this paper, we present asymmetric pivot manipulation for picking up rigid hollow objects to achieve a hole grasp. The pivot motion, executed by a position-controlled robotic arm, enables the gripper to effectively grasp hollow objects placed horizontally such that one gripper finger is positioned inside the object's hole, while the other contacts its outer surface along the length. Hole grasp is widely employed by humans to manipulate hollow objects, facilitating precise placement and enabling efficient subsequent operations, such as tightly packing objects into trays or accurately inserting them into narrow machine slots in manufacturing processes. Asymmetric pivoting for hole grasping is applicable to hollow objects of various sizes and hole shapes, including bottles, cups, and ducts. We investigate the variable parameters that satisfy the force balance conditions for successful grasping configurations. Our method can be implemented using a commercially available parallel-jaw gripper installed directly on a robot arm without modification. Experimental verification confirmed that hole grasp can be achieved using our proposed asymmetric pivot manipulation for various hollow objects, demonstrating a high success rate. Two use cases, namely aligning and feeding hollow cylindrical objects, were experimentally demonstrated on the testbed to clearly showcase the advantages of the hole grasp approach. 

**Abstract (ZH)**: 本文介绍了不对称枢轴操作方法，用于拾取刚性空心物体以实现孔抓取。通过位置控制机器人手臂执行的枢轴运动，使得夹爪能够有效地抓取水平放置的空心物体，其中一个夹爪手指定位在物体的孔内，另一个则沿物体外表面接触。孔抓取方法广泛应用于人类抓取空心物体的操作中，有助于精确放置，并使得后续操作，如紧密装填托盘或制造业中将物体准确插入狭窄机槽变得高效。不同大小和孔型的空心物体（如瓶子、杯子和管路）均适用于不对称绕孔抓取。我们研究了满足抓取配置中力平衡条件的可变参数。该方法可以通过直接安装在机器人臂上的商品化并指夹爪实现，而无需进行修改。实验验证证实了使用本文提出的方法可以成功地对多种空心物体实现孔抓取，并且成功率较高。通过测试床分别演示了对中和填充空心圆柱形物体的两个应用案例，以清晰展示孔抓取方法的优势。 

---
# Adaptive Multirobot Virtual Structure Control using Dual Quaternions 

**Title (ZH)**: 基于双四元数的自适应多机器人虚拟结构控制 

**Authors**: Juan Giribet, Alejandro Ghersin, Ignacio Mas  

**Link**: [PDF](https://arxiv.org/pdf/2504.05560)  

**Abstract**: A dual quaternion-based control strategy for formation flying of small UAV groups is proposed. Through the definition of a virtual structure, the coordinated control of formation's position, orientation, and shape parameters is enabled. This abstraction simplifies formation management, allowing a low-level controller to compute commands for individual UAVs. The controller is divided into a pose control module and a geometry-based adaptive strategy, providing efficient and precise task execution. Simulation and experimental results validate the approach. 

**Abstract (ZH)**: 基于双四元数的形成飞行控制策略用于小型无人机群：通过虚拟结构定义实现 formations 的位置、姿态和形状参数的协调控制，简化了形成管理，使低级控制器能够为个别无人机计算命令。控制器分为姿态控制模块和基于几何的自适应策略，提供高效的精确任务执行。仿真实验结果验证了该方法。 

---
# SPARK-Remote: A Cost-Effective System for Remote Bimanual Robot Teleoperation 

**Title (ZH)**: SPARK-远程：一种经济高效的远程双臂机器人遥控系统 

**Authors**: Adam Imdieke, Karthik Desingh  

**Link**: [PDF](https://arxiv.org/pdf/2504.05488)  

**Abstract**: Robot teleoperation enables human control over robotic systems in environments where full autonomy is challenging. Recent advancements in low-cost teleoperation devices and VR/AR technologies have expanded accessibility, particularly for bimanual robot manipulators. However, transitioning from in-person to remote teleoperation presents challenges in task performance. We introduce SPARK, a kinematically scaled, low-cost teleoperation system for operating bimanual robots. Its effectiveness is compared to existing technologies like the 3D SpaceMouse and VR/AR controllers. We further extend SPARK to SPARK-Remote, integrating sensor-based force feedback using haptic gloves and a force controller for remote teleoperation. We evaluate SPARK and SPARK-Remote variants on 5 bimanual manipulation tasks which feature operational properties - positional precision, rotational precision, large movements in the workspace, and bimanual collaboration - to test the effective teleoperation modes. Our findings offer insights into improving low-cost teleoperation interfaces for real-world applications. For supplementary materials, additional experiments, and qualitative results, visit the project webpage: this https URL 

**Abstract (ZH)**: 机器人遥操作使人能够在完全自主控制具有挑战的环境中操控机器人系统。低成本遥操作系统及VR/AR技术的进步扩展了遥操作的可访问性，特别是对于双臂机器人操作器。然而，从现场操作过渡到远程操作在任务执行方面提出了挑战。我们引入了SPARK，一种用于操作双臂机器人的 cinématically 缩放且低成本的遥操作系统，并将其与现有的3D SpaceMouse和VR/AR控制器进行比较。我们进一步将SPARK扩展为SPARK-Remote，集成基于传感器的力反馈，使用触觉手套和力控制器进行远程操作。我们评估了SPARK和SPARK-Remote变体在5个具有操作特性（位置精确度、旋转精确度、工作空间中的大范围移动和双臂协作）的双臂操作任务上的表现，以测试有效的遥操作模式。我们的发现提供了有关改善低成本遥操作系统架构，使其适用于实际应用的见解。更多信息、补充材料和定性结果，请访问项目网页：this https URL 

---
# Development and Experimental Evaluation of a Vibration-Based Adhesion System for Miniature Wall-Climbing Robots 

**Title (ZH)**: 基于振动的粘附系统的小型攀壁机器人开发与实验评估 

**Authors**: Siqian Li, Jung-Che Chang, Xi Wang, Xin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.05351)  

**Abstract**: In recent years, miniature wall-climbing robots have attracted widespread attention due to their significant potential in equipment inspection and in-situ repair applications. Traditional wall-climbing systems typically rely on electromagnetic, electrostatic, vacuum suction, or van der Waals forces for controllable adhesion. However, these conventional methods impose limitations when striving for both a compact design and high-speed mobility. This paper proposes a novel Vibration-Based Adhesion (VBA) technique, which utilizes a flexible disk vibrating near a surface to generate a strong and controllable attractive force without direct contact. By employing an electric motor as the vibration source, the constructed VBA system was experimentally evaluated, achieving an adhesion-to-weight ratio exceeding 51 times. The experimental results demonstrate that this adhesion mechanism not only provides a high normal force but also maintains minimal shear force, making it particularly suitable for high-speed movement and heavy load applications in miniature wall-climbing robots. 

**Abstract (ZH)**: 近年来，小型攀墙机器人由于其在设备检查和原位修复应用中的巨大潜力而引起了广泛关注。传统的攀墙系统通常依赖电磁、静电、真空吸附或范德华力实现可控黏附。然而，这些传统方法在追求紧凑设计和高速移动时存在局限性。本文提出了一种基于振动的黏附（Vibration-Based Adhesion, VBA）新技术，利用接近表面的柔性盘振动生成强可控的吸引力，无需直接接触。通过使用电动机作为振动源，构建的VBA系统进行了实验评估，实现黏附质量比超过51倍。实验结果表明，该黏附机制不仅提供了高法向力，而且保持了最小的剪切力，特别适合小型攀墙机器人的高速移动和重载应用。 

---
# Optimized Path Planning for Logistics Robots Using Ant Colony Algorithm under Multiple Constraints 

**Title (ZH)**: 多约束条件下物流机器人路径规划的蚁群优化算法 

**Authors**: Haopeng Zhao, Zhichao Ma, Lipeng Liu, Yang Wang, Zheyu Zhang, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05339)  

**Abstract**: With the rapid development of the logistics industry, the path planning of logistics vehicles has become increasingly complex, requiring consideration of multiple constraints such as time windows, task sequencing, and motion smoothness. Traditional path planning methods often struggle to balance these competing demands efficiently. In this paper, we propose a path planning technique based on the Ant Colony Optimization (ACO) algorithm to address these challenges. The proposed method optimizes key performance metrics, including path length, task completion time, turning counts, and motion smoothness, to ensure efficient and practical route planning for logistics vehicles. Experimental results demonstrate that the ACO-based approach outperforms traditional methods in terms of both efficiency and adaptability. This study provides a robust solution for logistics vehicle path planning, offering significant potential for real-world applications in dynamic and constrained environments. 

**Abstract (ZH)**: 基于蚁群优化算法的物流车辆路径规划技术 

---
# Ultrasound-Guided Robotic Blood Drawing and In Vivo Studies on Submillimetre Vessels of Rats 

**Title (ZH)**: 基于超声引导的机器人采血及小鼠毫米级血管的在 vivo 研究 

**Authors**: Shuaiqi Jing, Tianliang Yao, Ke Zhang, Di Wu, Qiulin Wang, Zixi Chen, Ke Chen, Peng Qi  

**Link**: [PDF](https://arxiv.org/pdf/2504.05329)  

**Abstract**: Billions of vascular access procedures are performed annually worldwide, serving as a crucial first step in various clinical diagnostic and therapeutic procedures. For pediatric or elderly individuals, whose vessels are small in size (typically 2 to 3 mm in diameter for adults and less than 1 mm in children), vascular access can be highly challenging. This study presents an image-guided robotic system aimed at enhancing the accuracy of difficult vascular access procedures. The system integrates a 6-DoF robotic arm with a 3-DoF end-effector, ensuring precise navigation and needle insertion. Multi-modal imaging and sensing technologies have been utilized to endow the medical robot with precision and safety, while ultrasound imaging guidance is specifically evaluated in this study. To evaluate in vivo vascular access in submillimeter vessels, we conducted ultrasound-guided robotic blood drawing on the tail veins (with a diameter of 0.7 plus or minus 0.2 mm) of 40 rats. The results demonstrate that the system achieved a first-attempt success rate of 95 percent. The high first-attempt success rate in intravenous vascular access, even with small blood vessels, demonstrates the system's effectiveness in performing these procedures. This capability reduces the risk of failed attempts, minimizes patient discomfort, and enhances clinical efficiency. 

**Abstract (ZH)**: billions of 血管通路操作 annually 全球进行，作为各种临床诊断和治疗操作的关键第一步。对于儿科或老年人患者，其血管较小（成人通常为2至3毫米直径，儿童则小于1毫米），血管通路操作具有挑战性。本研究提出了一种图像引导的机器人系统，旨在提高困难血管通路操作的准确性。该系统集成了6自由度机械臂和3自由度末端执行器，确保精确导航和针头插入。多模态成像和感知技术被用于赋予医疗机器人精确性和安全性，本研究特别评估了超声成像引导。为了在毫米级血管中进行在 vivo 血管通路操作，我们在40只大鼠的尾巴静脉（直径为0.7±0.2毫米）上进行了超声引导下的机器人采血。结果显示，该系统的一次性成功率达到了95%。即使在小血管静脉内进行静脉血管通路操作时，较高的一次性成功率证明了该系统的有效性能，减少了失败风险，减轻了患者的不适，并提高了临床效率。 

---
# Addressing Relative Degree Issues in Control Barrier Function Synthesis with Physics-Informed Neural Networks 

**Title (ZH)**: 基于物理先验神经网络的控制屏障函数合成中相对度问题的解决方案 

**Authors**: Lukas Brunke, Siqi Zhou, Francesco D'Orazio, Angela P. Schoellig  

**Link**: [PDF](https://arxiv.org/pdf/2504.06242)  

**Abstract**: In robotics, control barrier function (CBF)-based safety filters are commonly used to enforce state constraints. A critical challenge arises when the relative degree of the CBF varies across the state space. This variability can create regions within the safe set where the control input becomes unconstrained. When implemented as a safety filter, this may result in chattering near the safety boundary and ultimately compromise system safety. To address this issue, we propose a novel approach for CBF synthesis by formulating it as solving a set of boundary value problems. The solutions to the boundary value problems are determined using physics-informed neural networks (PINNs). Our approach ensures that the synthesized CBFs maintain a constant relative degree across the set of admissible states, thereby preventing unconstrained control scenarios. We illustrate the approach in simulation and further verify it through real-world quadrotor experiments, demonstrating its effectiveness in preserving desired system safety properties. 

**Abstract (ZH)**: 基于控制障碍函数的安全滤波在机器人领域的状态约束 enforcement 中的应用：通过边值问题形式化合成方法的研究 

---
# Channel State Information Analysis for Jamming Attack Detection in Static and Dynamic UAV Networks -- An Experimental Study 

**Title (ZH)**: 静态和动态无人机网络中Jamming攻击检测的信道状态信息分析——一项实验研究 

**Authors**: Pavlo Mykytyn, Ronald Chitauro, Zoya Dyka, Peter Langendoerfer  

**Link**: [PDF](https://arxiv.org/pdf/2504.05832)  

**Abstract**: Networks built on the IEEE 802.11 standard have experienced rapid growth in the last decade. Their field of application is vast, including smart home applications, Internet of Things (IoT), and short-range high throughput static and dynamic inter-vehicular communication networks. Within such networks, Channel State Information (CSI) provides a detailed view of the state of the communication channel and represents the combined effects of multipath propagation, scattering, phase shift, fading, and power decay. In this work, we investigate the problem of jamming attack detection in static and dynamic vehicular networks. We utilize ESP32-S3 modules to set up a communication network between an Unmanned Aerial Vehicle (UAV) and a Ground Control Station (GCS), to experimentally test the combined effects of a constant jammer on recorded CSI parameters, and the feasibility of jamming detection through CSI analysis in static and dynamic communication scenarios. 

**Abstract (ZH)**: 基于IEEE 802.11标准的网络在过去十年中经历了快速增长，其应用领域广泛，包括智能家居应用、物联网（IoT）以及短距离高吞吐量的静态和动态车载通信网络。在这些网络中，信道状态信息（CSI）提供了一个详细的通信信道状态视图，并代表了多径传播、散射、相位变化、衰落和功率衰减的综合影响。本文研究了在静态和动态车载网络中检测干扰攻击的问题。我们利用ESP32-S3模块设置了一个无人飞行器（UAV）与地面控制站（GCS）之间的通信网络，通过实验测试恒定干扰对记录的CSI参数的影响，并探讨通过CSI分析在静态和动态通信场景中检测干扰的可能性。 

---
# BC-ADMM: An Efficient Non-convex Constrained Optimizer with Robotic Applications 

**Title (ZH)**: BC-ADMM: 一种高效非凸约束优化器及其在机器人领域的应用 

**Authors**: Zherong Pan, Kui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05465)  

**Abstract**: Non-convex constrained optimizations are ubiquitous in robotic applications such as multi-agent navigation, UAV trajectory optimization, and soft robot simulation. For this problem class, conventional optimizers suffer from small step sizes and slow convergence. We propose BC-ADMM, a variant of Alternating Direction Method of Multiplier (ADMM), that can solve a class of non-convex constrained optimizations with biconvex constraint relaxation. Our algorithm allows larger step sizes by breaking the problem into small-scale sub-problems that can be easily solved in parallel. We show that our method has both theoretical convergence speed guarantees and practical convergence guarantees in the asymptotic sense. Through numerical experiments in a row of four robotic applications, we show that BC-ADMM has faster convergence than conventional gradient descent and Newton's method in terms of wall clock time. 

**Abstract (ZH)**: 非凸约束优化在机器人应用中普遍存在，如多agent导航、UAV轨迹优化和软机器人仿真。针对这一问题类别，传统的优化器存在步长小和收敛慢的问题。我们提出了一种交替方向乘子法（ADMM）的变体BC-ADMM，它可以解决一类通过双凸约束松弛处理的非凸约束优化问题。该算法通过将问题分解为可并行解决的小规模子问题，允许使用更大的步长。我们证明了该方法在理论和实际意义上都具有收敛速度保证。通过在四类机器人应用中的数值实验，我们展示了BC-ADMM在墙钟时间上比常规梯度下降法和牛顿法具有更快的收敛速度。 

---
