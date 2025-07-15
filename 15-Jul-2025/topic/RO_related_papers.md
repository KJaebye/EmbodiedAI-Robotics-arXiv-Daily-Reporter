# MP1: Mean Flow Tames Policy Learning in 1-step for Robotic Manipulation 

**Title (ZH)**: MP1: 平均流约束单一时间步策略学习以实现机器人操作 

**Authors**: Juyi Sheng, Ziyi Wang, Peiming Li, Mengyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.10543)  

**Abstract**: In robot manipulation, robot learning has become a prevailing approach. However, generative models within this field face a fundamental trade-off between the slow, iterative sampling of diffusion models and the architectural constraints of faster Flow-based methods, which often rely on explicit consistency losses. To address these limitations, we introduce MP1, which pairs 3D point-cloud inputs with the MeanFlow paradigm to generate action trajectories in one network function evaluation (1-NFE). By directly learning the interval-averaged velocity via the MeanFlow Identity, our policy avoids any additional consistency constraints. This formulation eliminates numerical ODE-solver errors during inference, yielding more precise trajectories. MP1 further incorporates CFG for improved trajectory controllability while retaining 1-NFE inference without reintroducing structural constraints. Because subtle scene-context variations are critical for robot learning, especially in few-shot learning, we introduce a lightweight Dispersive Loss that repels state embeddings during training, boosting generalization without slowing inference. We validate our method on the Adroit and Meta-World benchmarks, as well as in real-world scenarios. Experimental results show MP1 achieves superior average task success rates, outperforming DP3 by 10.2% and FlowPolicy by 7.3%. Its average inference time is only 6.8 ms-19x faster than DP3 and nearly 2x faster than FlowPolicy. Our code is available at this https URL. 

**Abstract (ZH)**: 机器人操作中的机器人学习已成为主流方法。然而，该领域内的生成模型面临着扩散模型迭代采样缓慢与快速Flow-based方法的架构约束之间的根本权衡，后者通常依赖于显式的一致性损失。为了解决这些局限性，我们引入了MP1，它将3D点云输入与MeanFlow范式配对，在一次网络函数评估（1-NFE）中生成动作轨迹。通过直接学习MeanFlow Identity的平均间隔速度，我们的策略避免了任何额外的一致性约束。这种形式在推断过程中消除了数值ODE求解器错误，提供了更精确的轨迹。MP1进一步结合了CFG以提高轨迹可控性，同时保持1-NFE推断而不引入新的结构约束。由于微妙的场景上下文变化对机器人学习至关重要，尤其是在少样本学习中，我们引入了轻量级的散射损失，以在训练期间排斥状态嵌入，从而提升泛化能力而不影响推断速度。我们在Adroit和Meta-World基准测试以及实际应用中验证了我们的方法。实验结果表明，MP1实现了更高的平均任务成功率，分别比DP3高出10.2%、比FlowPolicy高出7.3%。其平均推断时间仅为6.8毫秒，比DP3快19倍，比FlowPolicy快近2倍。我们提供的代码可在以下链接访问。 

---
# Polygonal Obstacle Avoidance Combining Model Predictive Control and Fuzzy Logic 

**Title (ZH)**: 结合模型预测控制与模糊逻辑的多边形障碍避障方法 

**Authors**: Michael Schröder, Eric Schöneberg, Daniel Görges, Hans D. Schotten  

**Link**: [PDF](https://arxiv.org/pdf/2507.10310)  

**Abstract**: In practice, navigation of mobile robots in confined environments is often done using a spatially discrete cost-map to represent obstacles. Path following is a typical use case for model predictive control (MPC), but formulating constraints for obstacle avoidance is challenging in this case. Typically the cost and constraints of an MPC problem are defined as closed-form functions and typical solvers work best with continuously differentiable functions. This is contrary to spatially discrete occupancy grid maps, in which a grid's value defines the cost associated with occupancy. This paper presents a way to overcome this compatibility issue by re-formulating occupancy grid maps to continuously differentiable functions to be embedded into the MPC scheme as constraints. Each obstacle is defined as a polygon -- an intersection of half-spaces. Any half-space is a linear inequality representing one edge of a polygon. Using AND and OR operators, the combined set of all obstacles and therefore the obstacle avoidance constraints can be described. The key contribution of this paper is the use of fuzzy logic to re-formulate such constraints that include logical operators as inequality constraints which are compatible with standard MPC formulation. The resulting MPC-based trajectory planner is successfully tested in simulation. This concept is also applicable outside of navigation tasks to implement logical or verbal constraints in MPC. 

**Abstract (ZH)**: 实践中的移动机器人在受限环境中的导航常使用空间离散的成本图来表示障碍物。将占用网格地图重新形式化为连续可微函数以嵌入到模型预测控制方案中作为约束，克服兼容性问题 

---
# REACT: Real-time Entanglement-Aware Coverage Path Planning for Tethered Underwater Vehicles 

**Title (ZH)**: 实时时效缠结感知覆盖路径规划算法 for 绞紧式水下车辆 

**Authors**: Abdelhakim Amer, Mohit Mehindratta, Yury Brodskiy, Bilal Wehbe, Erdal Kayacan  

**Link**: [PDF](https://arxiv.org/pdf/2507.10204)  

**Abstract**: Inspection of complex underwater structures with tethered underwater vehicles is often hindered by the risk of tether entanglement. We propose REACT (real-time entanglement-aware coverage path planning for tethered underwater vehicles), a framework designed to overcome this limitation. REACT comprises a fast geometry-based tether model using the signed distance field (SDF) map for accurate, real-time simulation of taut tether configurations around arbitrary structures in 3D. This model enables an efficient online replanning strategy by enforcing a maximum tether length constraint, thereby actively preventing entanglement. By integrating REACT into a coverage path planning framework, we achieve safe and optimal inspection paths, previously challenging due to tether constraints. The complete REACT framework's efficacy is validated in a pipe inspection scenario, demonstrating safe, entanglement-free navigation and full-coverage inspection. Simulation results show that REACT achieves complete coverage while maintaining tether constraints and completing the total mission 20% faster than conventional planners, despite a longer inspection time due to proactive avoidance of entanglement that eliminates extensive post-mission disentanglement. Real-world experiments confirm these benefits, where REACT completes the full mission, while the baseline planner fails due to physical tether entanglement. 

**Abstract (ZH)**: 基于 tether 能量知觉的 tethered 水下车辆实时绕开纠缠的覆盖率规划框架 REACT 

---
# Ariel Explores: Vision-based underwater exploration and inspection via generalist drone-level autonomy 

**Title (ZH)**: Ariel Explores：基于视觉的水下探索与检查通过通用型无人机级自主性 

**Authors**: Mohit Singh, Mihir Dharmadhikari, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2507.10003)  

**Abstract**: This work presents a vision-based underwater exploration and inspection autonomy solution integrated into Ariel, a custom vision-driven underwater robot. Ariel carries a $5$ camera and IMU based sensing suite, enabling a refraction-aware multi-camera visual-inertial state estimation method aided by a learning-based proprioceptive robot velocity prediction method that enhances robustness against visual degradation. Furthermore, our previously developed and extensively field-verified autonomous exploration and general visual inspection solution is integrated on Ariel, providing aerial drone-level autonomy underwater. The proposed system is field-tested in a submarine dry dock in Trondheim under challenging visual conditions. The field demonstration shows the robustness of the state estimation solution and the generalizability of the path planning techniques across robot embodiments. 

**Abstract (ZH)**: 基于视觉的水下探索与检测自主解决方案：集成于 Ariel 自定义视觉驱动水下机器人中的实现 

---
# Customize Harmonic Potential Fields via Hybrid Optimization over Homotopic Paths 

**Title (ZH)**: 基于同伦路径上的混合优化定制谐波势场 

**Authors**: Shuaikang Wang, Tiecheng Guo, Meng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.09858)  

**Abstract**: Safe navigation within a workspace is a fundamental skill for autonomous robots to accomplish more complex tasks. Harmonic potentials are artificial potential fields that are analytical, globally convergent and provably free of local minima. Thus, it has been widely used for generating safe and reliable robot navigation control policies. However, most existing methods do not allow customization of the harmonic potential fields nor the resulting paths, particularly regarding their topological properties. In this paper, we propose a novel method that automatically finds homotopy classes of paths that can be generated by valid harmonic potential fields. The considered complex workspaces can be as general as forest worlds consisting of numerous overlapping star-obstacles. The method is based on a hybrid optimization algorithm that searches over homotopy classes, selects the structure of each tree-of-stars within the forest, and optimizes over the continuous weight parameters for each purged tree via the projected gradient descent. The key insight is to transform the forest world to the unbounded point world via proper diffeomorphic transformations. It not only facilitates a simpler design of the multi-directional D-signature between non-homotopic paths, but also retain the safety and convergence properties. Extensive simulations and hardware experiments are conducted for non-trivial scenarios, where the navigation potentials are customized for desired homotopic properties. Project page: this https URL. 

**Abstract (ZH)**: 自主机器人在工作空间内的安全导航是一项基本技能，使其能够完成更复杂的任务。调和势是由分析方法生成的全局收敛且可证明无局部极小值的人工势场。因此，它广泛用于生成安全可靠的机器人导航控制策略。然而，现有大多数方法不允许可定制的调和势场及其生成的路径，特别是关于其拓扑性质。本文提出了一种新型方法，该方法能自动找到由有效调和势场生成的路径同伦类。所考虑的复杂工作空间可以是包括众多重叠星形障碍物的森林世界。该方法基于混合优化算法，搜索同伦类、选择森林中每个星形树的结构，并通过投影梯度下降优化每个清除的树的连续权重参数。关键见解是通过适当的微分同胚变换将森林世界转换为无界点世界。这不仅简化了非同伦路径之间多方向D-签名的设计，还保留了安全性和收敛性。在非平凡场景中进行了广泛的仿真和硬件实验，其中导航势场根据所需的同伦性质进行定制。项目页面：这个 https URL。 

---
# AdvGrasp: Adversarial Attacks on Robotic Grasping from a Physical Perspective 

**Title (ZH)**: AdvGrasp: 从物理视角展开的机器人抓取 adversarial 攻击 

**Authors**: Xiaofei Wang, Mingliang Han, Tianyu Hao, Cegang Li, Yunbo Zhao, Keke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09857)  

**Abstract**: Adversarial attacks on robotic grasping provide valuable insights into evaluating and improving the robustness of these systems. Unlike studies that focus solely on neural network predictions while overlooking the physical principles of grasping, this paper introduces AdvGrasp, a framework for adversarial attacks on robotic grasping from a physical perspective. Specifically, AdvGrasp targets two core aspects: lift capability, which evaluates the ability to lift objects against gravity, and grasp stability, which assesses resistance to external disturbances. By deforming the object's shape to increase gravitational torque and reduce stability margin in the wrench space, our method systematically degrades these two key grasping metrics, generating adversarial objects that compromise grasp performance. Extensive experiments across diverse scenarios validate the effectiveness of AdvGrasp, while real-world validations demonstrate its robustness and practical applicability 

**Abstract (ZH)**: 针对机器人抓取的对抗攻击为评估和提高这些系统的鲁棒性提供了宝贵见解。从物理角度出发，本文引入AdvGrasp框架，针对抓取的两个核心方面进行对抗攻击：提升能力（评估对抗重力提起物体的能力）和抓取稳定性（评估对外部干扰的抵抗能力）。通过变形物体形状以增加重力扭矩并减少稳定性裕度，我们的方法系统地降低了这两个关键抓取指标，生成对抗物体以破坏抓取性能。广泛实验跨不同场景验证了AdvGrasp的有效性，而实际验证则展示了其鲁棒性和实际应用价值。 

---
# IteraOptiRacing: A Unified Planning-Control Framework for Real-time Autonomous Racing for Iterative Optimal Performance 

**Title (ZH)**: IteraOptiRacing：一种用于迭代最优性能的实时自主赛车规划-控制框架 

**Authors**: Yifan Zeng, Yihan Li, Suiyi He, Koushil Sreenath, Jun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2507.09714)  

**Abstract**: This paper presents a unified planning-control strategy for competing with other racing cars called IteraOptiRacing in autonomous racing environments. This unified strategy is proposed based on Iterative Linear Quadratic Regulator for Iterative Tasks (i2LQR), which can improve lap time performance in the presence of surrounding racing obstacles. By iteratively using the ego car's historical data, both obstacle avoidance for multiple moving cars and time cost optimization are considered in this unified strategy, resulting in collision-free and time-optimal generated trajectories. The algorithm's constant low computation burden and suitability for parallel computing enable real-time operation in competitive racing scenarios. To validate its performance, simulations in a high-fidelity simulator are conducted with multiple randomly generated dynamic agents on the track. Results show that the proposed strategy outperforms existing methods across all randomly generated autonomous racing scenarios, enabling enhanced maneuvering for the ego racing car. 

**Abstract (ZH)**: 本论文提出了一种统一的规划-控制策略，用于在自主赛车环境中与称为IteraOptiRacing的其他赛车竞争。该统一策略基于迭代线性二次调节器（迭代任务版，i2LQR），能够在存在周围赛车障碍物的情况下提高圈速性能。通过迭代使用 ego 车的历史数据，该统一策略同时考虑了多辆移动车辆的避障和时间成本优化，生成了无碰撞且时间最优的轨迹。算法具有恒定的低计算负担，并适用于并行计算，可在竞争性赛车场景中实现实时运行。为了验证其性能，在高保真模拟器中使用赛道上多个随机生成的动态代理进行了仿真。结果显示，提出的策略在所有随机生成的自主赛车场景中均优于现有方法，提升了ego赛车的机动性能。 

---
# On the Importance of Neural Membrane Potential Leakage for LIDAR-based Robot Obstacle Avoidance using Spiking Neural Networks 

**Title (ZH)**: 基于视觉脉冲神经网络的LIDAR机器人避障中神经膜电位泄漏的重要性 

**Authors**: Zainab Ali, Lujayn Al-Amir, Ali Safa  

**Link**: [PDF](https://arxiv.org/pdf/2507.09538)  

**Abstract**: Using neuromorphic computing for robotics applications has gained much attention in recent year due to the remarkable ability of Spiking Neural Networks (SNNs) for high-precision yet low memory and compute complexity inference when implemented in neuromorphic hardware. This ability makes SNNs well-suited for autonomous robot applications (such as in drones and rovers) where battery resources and payload are typically limited. Within this context, this paper studies the use of SNNs for performing direct robot navigation and obstacle avoidance from LIDAR data. A custom robot platform equipped with a LIDAR is set up for collecting a labeled dataset of LIDAR sensing data together with the human-operated robot control commands used for obstacle avoidance. Crucially, this paper provides what is, to the best of our knowledge, a first focused study about the importance of neuron membrane leakage on the SNN precision when processing LIDAR data for obstacle avoidance. It is shown that by carefully tuning the membrane potential leakage constant of the spiking Leaky Integrate-and-Fire (LIF) neurons used within our SNN, it is possible to achieve on-par robot control precision compared to the use of a non-spiking Convolutional Neural Network (CNN). Finally, the LIDAR dataset collected during this work is released as open-source with the hope of benefiting future research. 

**Abstract (ZH)**: 使用神经形态计算进行基于神经脉冲网络的机器人应用取得了广泛关注，得益于其在神经形态硬件实现时对高精度但低存储和计算复杂度推断的能力。这种能力使神经脉冲网络（SNNs）非常适合电池资源和载荷通常受限的自主机器人应用（如无人机和火星车）。在此背景下，本文研究了使用SNNs直接从LIDAR数据执行机器人导航和避障的方法。搭建了一个配备了LIDAR的自定义机器人平台，采集了带标注的LIDAR感知数据以及用于避障的人工操作机器人控制命令。本文还首次集中在我们所知的范围内，探讨了神经元膜泄漏在处理LIDAR数据进行避障时对SNN精度的重要性。结果显示，通过仔细调整用于我们SNN中的脉冲泄漏整化放电（LIF）神经元的膜电位泄漏常数，可以实现与非脉冲卷积神经网络（CNN）相当的机器人控制精度。最后，本工作中收集的LIDAR数据集已开源，旨在为未来的研究提供支持。 

---
# mmE-Loc: Facilitating Accurate Drone Landing with Ultra-High-Frequency Localization 

**Title (ZH)**: mmE-Loc: 促进超高频定位准确的无人机着陆辅助技术 

**Authors**: Haoyang Wang, Jingao Xu, Xinyu Luo, Ting Zhang, Xuecheng Chen, Ruiyang Duan, Jialong Chen, Yunhao Liu, Jianfeng Zheng, Weijie Hong, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09469)  

**Abstract**: For precise, efficient, and safe drone landings, ground platforms should real-time, accurately locate descending drones and guide them to designated spots. While mmWave sensing combined with cameras improves localization accuracy, lower sampling frequency of traditional frame cameras compared to mmWave radar creates bottlenecks in system throughput. In this work, we upgrade traditional frame camera with event camera, a novel sensor that harmonizes in sampling frequency with mmWave radar within ground platform setup, and introduce mmE-Loc, a high-precision, low-latency ground localization system designed for precise drone landings. To fully exploit the \textit{temporal consistency} and \textit{spatial complementarity} between these two modalities, we propose two innovative modules: \textit{(i)} the Consistency-instructed Collaborative Tracking module, which further leverages the drone's physical knowledge of periodic micro-motions and structure for accurate measurements extraction, and \textit{(ii)} the Graph-informed Adaptive Joint Optimization module, which integrates drone motion information for efficient sensor fusion and drone localization. Real-world experiments conducted in landing scenarios with a drone delivery company demonstrate that mmE-Loc significantly outperforms state-of-the-art methods in both accuracy and latency. 

**Abstract (ZH)**: 基于事件摄像头的高精度低延迟地面定位系统mmE-Loc及其在无人机精确降落中的应用 

---
# Unmanned Aerial Vehicle (UAV) Data-Driven Modeling Software with Integrated 9-Axis IMUGPS Sensor Fusion and Data Filtering Algorithm 

**Title (ZH)**: 基于9轴IMU-GPS传感器融合与数据过滤算法的无人驾驶航空车辆（UAV）数据驱动建模软件 

**Authors**: Azfar Azdi Arfakhsyad, Aufa Nasywa Rahman, Larasati Kinanti, Ahmad Ataka Awwalur Rizqi, Hannan Nur Muhammad  

**Link**: [PDF](https://arxiv.org/pdf/2507.09464)  

**Abstract**: Unmanned Aerial Vehicles (UAV) have emerged as versatile platforms, driving the demand for accurate modeling to support developmental testing. This paper proposes data-driven modeling software for UAV. Emphasizes the utilization of cost-effective sensors to obtain orientation and location data subsequently processed through the application of data filtering algorithms and sensor fusion techniques to improve the data quality to make a precise model visualization on the software. UAV's orientation is obtained using processed Inertial Measurement Unit (IMU) data and represented using Quaternion Representation to avoid the gimbal lock problem. The UAV's location is determined by combining data from the Global Positioning System (GPS), which provides stable geographic coordinates but slower data update frequency, and the accelerometer, which has higher data update frequency but integrating it to get position data is unstable due to its accumulative error. By combining data from these two sensors, the software is able to calculate and continuously update the UAV's real-time position during its flight operations. The result shows that the software effectively renders UAV orientation and position with high degree of accuracy and fluidity 

**Abstract (ZH)**: 无人驾驶飞行器（UAV）作为多功能平台的出现，推动了对其准确建模的需求以支持开发测试。本文提出了一种基于数据驱动的UAV建模软件。强调使用经济有效的传感器获取姿态和位置数据，随后通过应用数据滤波算法和传感器融合技术来提高数据质量，从而在软件上实现精确的模型可视化。UAV的姿态使用处理后的惯性测量单元（IMU）数据获取，并用四元数表示以避免万向锁问题。UAV的位置通过结合全球定位系统（GPS）提供的稳定地理坐标但数据更新频率较低的数据和加速度计提供的高数据更新频率但因累积误差导致位置数据不稳定的数据显示。通过结合这两种传感器的数据，软件能够在UAV飞行操作中计算并连续更新其实时位置。结果表明，该软件有效地以高精度和流畅性渲染UAV的姿态和位置。 

---
# Influence of Static and Dynamic Downwash Interactions on Multi-Quadrotor Systems 

**Title (ZH)**: 静态和动态下洗气流交互作用对多旋翼系统的影响 

**Authors**: Anoop Kiran, Nora Ayanian, Kenneth Breuer  

**Link**: [PDF](https://arxiv.org/pdf/2507.09463)  

**Abstract**: Flying multiple quadrotors in close proximity presents a significant challenge due to complex aerodynamic interactions, particularly downwash effects that are known to destabilize vehicles and degrade performance. Traditionally, multi-quadrotor systems rely on conservative strategies, such as collision avoidance zones around the robot volume, to circumvent this effect. This restricts their capabilities by requiring a large volume for the operation of a multi-quadrotor system, limiting their applicability in dense environments. This work provides a comprehensive, data-driven analysis of the downwash effect, with a focus on characterizing, analyzing, and understanding forces, moments, and velocities in both single and multi-quadrotor configurations. We use measurements of forces and torques to characterize vehicle interactions, and particle image velocimetry (PIV) to quantify the spatial features of the downwash wake for a single quadrotor and an interacting pair of quadrotors. This data can be used to inform physics-based strategies for coordination, leverage downwash for optimized formations, expand the envelope of operation, and improve the robustness of multi-quadrotor control. 

**Abstract (ZH)**: 在紧密 proximity 内操纵多架旋翼机由于存在复杂的气动相互作用，尤其是已知会导致飞行器失稳并降低性能的下洗效应，构成了重大挑战。传统多旋翼系统依赖于保守策略，如在机器人体积周围设立避碰区域，以规避这种效应。这限制了它们的能力，要求多旋翼系统操作占用较大的空间，从而限制了它们在密集环境中的应用。本研究提供了下洗效应的全面数据驱动分析，重点关注单旋翼机和多旋翼机配置下的力、力矩和速度的特征化、分析和理解。我们利用力和力矩的测量结果来表征飞行器间的相互作用，并使用粒子图像 velocimetry (PIV) 来量化单旋翼机和相互作用的旋翼机对下洗尾流的空间特征。这些数据可用于指导基于物理策略的协调、利用下洗效应优化编队飞行、扩展操作包线并提高多旋翼机控制的鲁棒性。 

---
# Real-Time Adaptive Motion Planning via Point Cloud-Guided, Energy-Based Diffusion and Potential Fields 

**Title (ZH)**: 基于点云导向的能量扩散与势场的实时自适应运动规划 

**Authors**: Wondmgezahu Teshome, Kian Behzad, Octavia Camps, Michael Everett, Milad Siami, Mario Sznaier  

**Link**: [PDF](https://arxiv.org/pdf/2507.09383)  

**Abstract**: Motivated by the problem of pursuit-evasion, we present a motion planning framework that combines energy-based diffusion models with artificial potential fields for robust real time trajectory generation in complex environments. Our approach processes obstacle information directly from point clouds, enabling efficient planning without requiring complete geometric representations. The framework employs classifier-free guidance training and integrates local potential fields during sampling to enhance obstacle avoidance. In dynamic scenarios, the system generates initial trajectories using the diffusion model and continuously refines them through potential field-based adaptation, demonstrating effective performance in pursuit-evasion scenarios with partial pursuer observability. 

**Abstract (ZH)**: 基于追逐逃脱问题的能源驱动扩散模型与人工势场结合的运动规划框架：复杂环境下的鲁棒实时轨迹生成 

---
# C-ZUPT: Stationarity-Aided Aerial Hovering 

**Title (ZH)**: C-ZUPT: 站姿辅助空中悬停 

**Authors**: Daniel Engelsman, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2507.09344)  

**Abstract**: Autonomous systems across diverse domains have underscored the need for drift-resilient state estimation. Although satellite-based positioning and cameras are widely used, they often suffer from limited availability in many environments. As a result, positioning must rely solely on inertial sensors, leading to rapid accuracy degradation over time due to sensor biases and noise. To counteract this, alternative update sources-referred to as information aiding-serve as anchors of certainty. Among these, the zero-velocity update (ZUPT) is particularly effective in providing accurate corrections during stationary intervals, though it is restricted to surface-bound platforms. This work introduces a controlled ZUPT (C-ZUPT) approach for aerial navigation and control, independent of surface contact. By defining an uncertainty threshold, C-ZUPT identifies quasi-static equilibria to deliver precise velocity updates to the estimation filter. Extensive validation confirms that these opportunistic, high-quality updates significantly reduce inertial drift and control effort. As a result, C-ZUPT mitigates filter divergence and enhances navigation stability, enabling more energy-efficient hovering and substantially extending sustained flight-key advantages for resource-constrained aerial systems. 

**Abstract (ZH)**: 跨领域自主系统强调了抗漂移状态估计的必要性。尽管卫星定位和摄像头广泛应用，但在许多环境中availability受到限制。因此，定位不得不依赖惯性传感器，但由于传感器偏差和噪声的影响，随着时间的推移会导致快速准确度下降。为应对这一问题，提供替代更新来源的方法——称为信息辅助——作为确定性的锚点。其中，零速度更新（ZUPT）特别有效，能在静止间隔提供准确的修正，但仅限于地面固定平台。本文介绍了无地面接触的可控零速度更新（C-ZUPT）方法，用于空中导航和控制。通过定义不确定性阈值，C-ZUPT识别准静态平衡，为估计滤波器提供精确的速度更新。广泛的验证表明，这些机会性的高质量更新显著减少了惯性漂移并降低了控制努力。结果，C-ZUPT规避了滤波器发散问题，提高了导航稳定性，使更节能的悬停成为可能，并显著延长了持续飞行时间——对于资源受限的空中系统而言，这些都是关键优势。 

---
# Unified Linear Parametric Map Modeling and Perception-aware Trajectory Planning for Mobile Robotics 

**Title (ZH)**: 统一的线性参数地图建模与感知驱动的轨迹规划 for 移动机器人 

**Authors**: Hongyu Nie, Xingyu Li, Xu Liu, Zhaotong Tan, Sen Mei, Wenbo Su  

**Link**: [PDF](https://arxiv.org/pdf/2507.09340)  

**Abstract**: Autonomous navigation in mobile robots, reliant on perception and planning, faces major hurdles in large-scale, complex environments. These include heavy computational burdens for mapping, sensor occlusion failures for UAVs, and traversal challenges on irregular terrain for UGVs, all compounded by a lack of perception-aware strategies. To address these challenges, we introduce Random Mapping and Random Projection (RMRP). This method constructs a lightweight linear parametric map by first mapping data to a high-dimensional space, followed by a sparse random projection for dimensionality reduction. Our novel Residual Energy Preservation Theorem provides theoretical guarantees for this process, ensuring critical geometric properties are preserved. Based on this map, we propose the RPATR (Robust Perception-Aware Trajectory Planner) framework. For UAVs, our method unifies grid and Euclidean Signed Distance Field (ESDF) maps. The front-end uses an analytical occupancy gradient to refine initial paths for safety and smoothness, while the back-end uses a closed-form ESDF for trajectory optimization. Leveraging the trained RMRP model's generalization, the planner predicts unobserved areas for proactive navigation. For UGVs, the model characterizes terrain and provides closed-form gradients, enabling online planning to circumvent large holes. Validated in diverse scenarios, our framework demonstrates superior mapping performance in time, memory, and accuracy, and enables computationally efficient, safe navigation for high-speed UAVs and UGVs. The code will be released to foster community collaboration. 

**Abstract (ZH)**: 自主移动机器人在大规模复杂环境中的自主导航，依赖于感知和规划，面临重大挑战。这些挑战包括测绘计算负担沉重、无人机传感器遮挡失败以及地面机器人在不规则地形上的行进困难，所有这些问题都进一步加剧了缺乏感知导向策略的情况。为应对这些挑战，我们提出了一种名为随机映射和随机投影（RMRP）的方法。该方法通过先将数据映射到高维空间，再通过稀疏随机投影进行维数约简来构建轻量级的线性参数化地图。我们提出的残差能量保持定理为这一过程提供了理论保证，确保关键的几何特性得以保留。基于此地图，我们提出了鲁棒感知导向轨迹规划框架（RPATR）。对于无人机，我们的方法统一了网格地图和欧几里得符号距离场（ESDF）地图。前端使用分析占里梯度来细化初始路径以确保安全性和平滑性，而后端则使用闭式解的ESDF来进行轨迹优化。利用训练好的RMRP模型的泛化能力，规划器可以预测未观测区域，实现前瞻性导航。对于地面机器人，该模型描述地形并提供闭式梯度，使在线规划能够绕过大孔洞。在多种场景下验证，该框架在时间、内存和准确性方面表现出优越的绘图性能，并使高速无人机和地面机器人能够高效、安全地导航。代码将对外开放以促进社区合作。 

---
# Behavioral Exploration: Learning to Explore via In-Context Adaptation 

**Title (ZH)**: 行为探索:通过上下文适配进行探索学习 

**Authors**: Andrew Wagenmaker, Zhiyuan Zhou, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2507.09041)  

**Abstract**: Developing autonomous agents that quickly explore an environment and adapt their behavior online is a canonical challenge in robotics and machine learning. While humans are able to achieve such fast online exploration and adaptation, often acquiring new information and skills in only a handful of interactions, existing algorithmic approaches tend to rely on random exploration and slow, gradient-based behavior updates. How can we endow autonomous agents with such capabilities on par with humans? Taking inspiration from recent progress on both in-context learning and large-scale behavioral cloning, in this work we propose behavioral exploration: training agents to internalize what it means to explore and adapt in-context over the space of ``expert'' behaviors. To achieve this, given access to a dataset of expert demonstrations, we train a long-context generative model to predict expert actions conditioned on a context of past observations and a measure of how ``exploratory'' the expert's behaviors are relative to this context. This enables the model to not only mimic the behavior of an expert, but also, by feeding its past history of interactions into its context, to select different expert behaviors than what have been previously selected, thereby allowing for fast online adaptation and targeted, ``expert-like'' exploration. We demonstrate the effectiveness of our method in both simulated locomotion and manipulation settings, as well as on real-world robotic manipulation tasks, illustrating its ability to learn adaptive, exploratory behavior. 

**Abstract (ZH)**: 开发能够在短时间内探索环境并在线调整其行为的自主代理是机器人学和机器学习中的经典挑战。如何赋予自主代理与人类相当的这些能力？受到上下文学习和大规模行为克隆近期进展的启发，本文提出行为探索：训练代理在“专家”行为空间中理解和内化探索和适应的含义。为此，利用专家演示的数据集，训练一个长上下文生成模型，预测在过去的观察和行为相对于这些观察的探索性程度条件下的专家行为。这使得模型不仅能模拟专家的行为，还能通过将过去的交互历史反馈到上下文中，选择不同的专家行为，从而实现快速在线适应和有针对性的、“专家级”的探索。我们在模拟的运动和操作设置以及现实世界的机器人操作任务中展示了该方法的有效性，证明了其学习适应性、探索性行为的能力。 

---
