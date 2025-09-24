# Imitation-Guided Bimanual Planning for Stable Manipulation under Changing External Forces 

**Title (ZH)**: 模仿引导的双臂规划以应对外部力变化的稳定操作 

**Authors**: Kuanqi Cai, Chunfeng Wang, Zeqi Li, Haowen Yao, Weinan Chen, Luis Figueredo, Aude Billard, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2509.19261)  

**Abstract**: Robotic manipulation in dynamic environments often requires seamless transitions between different grasp types to maintain stability and efficiency. However, achieving smooth and adaptive grasp transitions remains a challenge, particularly when dealing with external forces and complex motion constraints. Existing grasp transition strategies often fail to account for varying external forces and do not optimize motion performance effectively. In this work, we propose an Imitation-Guided Bimanual Planning Framework that integrates efficient grasp transition strategies and motion performance optimization to enhance stability and dexterity in robotic manipulation. Our approach introduces Strategies for Sampling Stable Intersections in Grasp Manifolds for seamless transitions between uni-manual and bi-manual grasps, reducing computational costs and regrasping inefficiencies. Additionally, a Hierarchical Dual-Stage Motion Architecture combines an Imitation Learning-based Global Path Generator with a Quadratic Programming-driven Local Planner to ensure real-time motion feasibility, obstacle avoidance, and superior manipulability. The proposed method is evaluated through a series of force-intensive tasks, demonstrating significant improvements in grasp transition efficiency and motion performance. A video demonstrating our simulation results can be viewed at \href{this https URL}{\textcolor{blue}{this https URL}}. 

**Abstract (ZH)**: 动态环境下的机器人操作往往需要在不同抓取类型之间实现无缝过渡，以维持稳定性和效率。然而，实现平滑且适应性的抓取过渡仍是一项挑战，特别是在处理外部力和复杂运动约束时。现有的抓取过渡策略往往未能考虑到变化的外部力，也没有有效优化运动性能。在本工作中，我们提出了一种模仿引导的双臂规划框架，将有效的抓取过渡策略和运动性能优化相结合，以增强机器人操作中的稳定性和灵巧性。我们的方法引入了在抓取流形中采样稳定交点的策略，以实现单手抓取和双手抓取之间的无缝过渡，从而减少计算成本和重新抓取的无效性。此外，层次化的双阶段运动架构结合了基于模仿学习的全局路径生成器和二次规划驱动的局部规划器，以确保实时运动可行性、避开障碍物以及卓越的操作性能。提出的方​​法通过一系列力密集型任务进行评估，显示出在抓取过渡效率和运动性能方面的显著改进。我们的仿真结果演示视频可以在 \href{this https URL}{这个网址} 查看。 

---
# Proactive-reactive detection and mitigation of intermittent faults in robot swarms 

**Title (ZH)**: proactive-反应式检测与缓解机器人 swarm 中的间歇性故障 

**Authors**: Sinan Oğuz, Emanuele Garone, Marco Dorigo, Mary Katherine Heinrich  

**Link**: [PDF](https://arxiv.org/pdf/2509.19246)  

**Abstract**: Intermittent faults are transient errors that sporadically appear and disappear. Although intermittent faults pose substantial challenges to reliability and coordination, existing studies of fault tolerance in robot swarms focus instead on permanent faults. One reason for this is that intermittent faults are prohibitively difficult to detect in the fully self-organized ad-hoc networks typical of robot swarms, as their network topologies are transient and often unpredictable. However, in the recently introduced self-organizing nervous systems (SoNS) approach, robot swarms are able to self-organize persistent network structures for the first time, easing the problem of detecting intermittent faults. To address intermittent faults in robot swarms that have persistent networks, we propose a novel proactive-reactive strategy to detection and mitigation, based on self-organized backup layers and distributed consensus in a multiplex network. Proactively, the robots self-organize dynamic backup paths before faults occur, adapting to changes in the primary network topology and the robots' relative positions. Reactively, robots use one-shot likelihood ratio tests to compare information received along different paths in the multiplex network, enabling early fault detection. Upon detection, communication is temporarily rerouted in a self-organized way, until the detected fault resolves. We validate the approach in representative scenarios of faulty positional data occurring during formation control, demonstrating that intermittent faults are prevented from disrupting convergence to desired formations, with high fault detection accuracy and low rates of false positives. 

**Abstract (ZH)**: 间歇性故障是偶尔出现并消失的瞬态错误，尽管间歇性故障给可靠性和协调带来了重大挑战，现有机器人 swarm 故障容错研究主要集中在永久性故障上。其中一个原因是间歇性故障在典型由机器人 swarm 构成的完全自组织即兴网络中难以检测，因为这些网络拓扑是瞬态且往往不可预测的。然而，在最近引入的自组织神经系统（SoNS）方法中，机器人 swarm 首次能够自我组织持久的网络结构，从而减轻了检测间歇性故障的问题。为了应对具有持久网络结构的机器人 swarm 中的间歇性故障，我们提出了一种新颖的主动-被动检测与缓解策略，基于多层网络中的自组织备份层和分布式一致意见。主动地，机器人在故障发生前自组织动态备份路径，适应主网络拓扑和机器人相对位置的变化。被动地，机器人使用一次似然比检验来比较多层网络中不同路径接收到的信息，实现早期故障检测。检测到故障后，通信暂时以自组织方式重路由，直到故障被解决。我们在故障位置数据导致队形控制失效的代表性场景中验证了该方法，证明了间歇性故障不会破坏对期望队形的收敛，具有高故障检测准确性和低误报率。 

---
# SlicerROS2: A Research and Development Module for Image-Guided Robotic Interventions 

**Title (ZH)**: SlicerROS2: 一种图像引导机器人干预研究与开发模块 

**Authors**: Laura Connolly, Aravind S. Kumar, Kapi Ketan Mehta, Lidia Al-Zogbi, Peter Kazanzides, Parvin Mousavi, Gabor Fichtinger, Axel Krieger, Junichi Tokuda, Russell H. Taylor, Simon Leonard, Anton Deguet  

**Link**: [PDF](https://arxiv.org/pdf/2509.19076)  

**Abstract**: Image-guided robotic interventions involve the use of medical imaging in tandem with robotics. SlicerROS2 is a software module that combines 3D Slicer and robot operating system (ROS) in pursuit of a standard integration approach for medical robotics research. The first release of SlicerROS2 demonstrated the feasibility of using the C++ API from 3D Slicer and ROS to load and visualize robots in real time. Since this initial release, we've rewritten and redesigned the module to offer greater modularity, access to low-level features, access to 3D Slicer's Python API, and better data transfer protocols. In this paper, we introduce this new design as well as four applications that leverage the core functionalities of SlicerROS2 in realistic image-guided robotics scenarios. 

**Abstract (ZH)**: 基于图像的机器人干预涉及将医学成像与机器人技术结合使用。SlicerROS2 是一个软件模块，结合了3D Slicer和机器人操作系统（ROS），旨在提供医学机器人研究的标准集成方法。SlicerROS2 的首次发布展示了使用3D Slicer和ROS的C++ API 实时加载和可视化机器人的可行性。在此初步发布的基础上，我们重新设计了该模块，使其更具模块性，提供低级功能访问，3D Slicer的Python API 访问，并改进了数据传输协议。本文介绍了这一新设计以及四种利用SlicerROS2核心功能的应用程序，适用于实际的基于图像的机器人场景。 

---
# TacEva: A Performance Evaluation Framework For Vision-Based Tactile Sensors 

**Title (ZH)**: TacEva：基于视觉的触觉传感器性能评估框架 

**Authors**: Qingzheng Cong, Steven Oh, Wen Fan, Shan Luo, Kaspar Althoefer, Dandan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19037)  

**Abstract**: Vision-Based Tactile Sensors (VBTSs) are widely used in robotic tasks because of the high spatial resolution they offer and their relatively low manufacturing costs. However, variations in their sensing mechanisms, structural dimension, and other parameters lead to significant performance disparities between existing VBTSs. This makes it challenging to optimize them for specific tasks, as both the initial choice and subsequent fine-tuning are hindered by the lack of standardized metrics. To address this issue, TacEva is introduced as a comprehensive evaluation framework for the quantitative analysis of VBTS performance. The framework defines a set of performance metrics that capture key characteristics in typical application scenarios. For each metric, a structured experimental pipeline is designed to ensure consistent and repeatable quantification. The framework is applied to multiple VBTSs with distinct sensing mechanisms, and the results demonstrate its ability to provide a thorough evaluation of each design and quantitative indicators for each performance dimension. This enables researchers to pre-select the most appropriate VBTS on a task by task basis, while also offering performance-guided insights into the optimization of VBTS design. A list of existing VBTS evaluation methods and additional evaluations can be found on our website: this https URL 

**Abstract (ZH)**: 基于视觉的触觉传感器（VBTSs）的综合评估框架：针对典型应用场景的定量分析 

---
# Towards Robust LiDAR Localization: Deep Learning-based Uncertainty Estimation 

**Title (ZH)**: 基于深度学习的不确定性估计的鲁棒LiDAR定位 

**Authors**: Minoo Dolatabadi, Fardin Ayar, Ehsan Javanmardi, Manabu Tsukada, Mahdi Javanmardi  

**Link**: [PDF](https://arxiv.org/pdf/2509.18954)  

**Abstract**: LiDAR-based localization and SLAM often rely on iterative matching algorithms, particularly the Iterative Closest Point (ICP) algorithm, to align sensor data with pre-existing maps or previous scans. However, ICP is prone to errors in featureless environments and dynamic scenes, leading to inaccurate pose estimation. Accurately predicting the uncertainty associated with ICP is crucial for robust state estimation but remains challenging, as existing approaches often rely on handcrafted models or simplified assumptions. Moreover, a few deep learning-based methods for localizability estimation either depend on a pre-built map, which may not always be available, or provide a binary classification of localizable versus non-localizable, which fails to properly model uncertainty. In this work, we propose a data-driven framework that leverages deep learning to estimate the registration error covariance of ICP before matching, even in the absence of a reference map. By associating each LiDAR scan with a reliable 6-DoF error covariance estimate, our method enables seamless integration of ICP within Kalman filtering, enhancing localization accuracy and robustness. Extensive experiments on the KITTI dataset demonstrate the effectiveness of our approach, showing that it accurately predicts covariance and, when applied to localization using a pre-built map or SLAM, reduces localization errors and improves robustness. 

**Abstract (ZH)**: 基于LiDAR的定位与SLAM往往依赖于迭代配准算法，特别是ICP算法，将传感器数据与已有地图或先前扫描进行对齐。然而，ICP在缺乏特征环境和动态场景中容易出错，导致姿态估计不准确。准确预测ICP相关的不确定性对于鲁棒的状态估计至关重要，但现有方法往往依赖于手工制作的模型或简化假设，仍然具有挑战性。此外，一些基于深度学习的可定位性估计方法要么依赖于先建好的地图，这可能并不总是可用的，要么仅二元分类可定位与不可定位，无法很好地建模不确定性。本文提出了一种数据驱动框架，利用深度学习在匹配前估计ICP的配准误差协方差，即使在没有参考地图的情况下也是如此。通过将每个LiDAR扫描与可靠的6-DoF误差协方差估计关联，我们的方法能够无缝地将ICP集成到卡尔曼滤波中，从而提升定位的准确性和鲁棒性。在KITTI数据集上的广泛实验表明，我们的方法能够准确预测协方差，并在使用先建好的地图或SLAM进行定位时，减少定位误差并提升鲁棒性。 

---
# DexSkin: High-Coverage Conformable Robotic Skin for Learning Contact-Rich Manipulation 

**Title (ZH)**: DexSkin: 高覆盖率顺应式机器人皮肤用于学习接触丰富操作 

**Authors**: Suzannah Wistreich, Baiyu Shi, Stephen Tian, Samuel Clarke, Michael Nath, Chengyi Xu, Zhenan Bao, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18830)  

**Abstract**: Human skin provides a rich tactile sensing stream, localizing intentional and unintentional contact events over a large and contoured region. Replicating these tactile sensing capabilities for dexterous robotic manipulation systems remains a longstanding challenge. In this work, we take a step towards this goal by introducing DexSkin. DexSkin is a soft, conformable capacitive electronic skin that enables sensitive, localized, and calibratable tactile sensing, and can be tailored to varying geometries. We demonstrate its efficacy for learning downstream robotic manipulation by sensorizing a pair of parallel jaw gripper fingers, providing tactile coverage across almost the entire finger surfaces. We empirically evaluate DexSkin's capabilities in learning challenging manipulation tasks that require sensing coverage across the entire surface of the fingers, such as reorienting objects in hand and wrapping elastic bands around boxes, in a learning-from-demonstration framework. We then show that, critically for data-driven approaches, DexSkin can be calibrated to enable model transfer across sensor instances, and demonstrate its applicability to online reinforcement learning on real robots. Our results highlight DexSkin's suitability and practicality for learning real-world, contact-rich manipulation. Please see our project webpage for videos and visualizations: this https URL. 

**Abstract (ZH)**: 人类皮肤提供了一条丰富的触觉传感流，能够在大面积且形状复杂的区域定位有意和无意的接触事件。为 Dexterous 机器人 manipulation 系统复制这些触觉传感能力仍然是一个长期的挑战。在本文中，我们朝着这一目标迈出了一步，介绍了一种名为 DexSkin 的软性可调节电容式电子皮肤。DexSkin 具备灵敏、局部化和可标定的触觉传感功能，并可根据不同的几何形状进行定制。我们通过对并指夹爪手指进行传感化处理，展示了其在几乎整个手指表面提供触觉覆盖方面的有效性。我们还在演示学习框架中，通过 DexSkin 的能力来学习需要覆盖手指整个表面的传感覆盖的复杂 manipulation 任务，例如在手中重新定向物体和在盒子上缠绕弹性带子。我们还展示了，对于数据驱动的方法，DexSkin 可以被标定以实现传感器实例之间的模型迁移，并展示了其在真实机器人上的在线强化学习中的适用性。我们的结果强调了 DexSkin 在学习真实世界、接触丰富的 manipulation 任务方面的适用性和实用性。请参见我们的项目网页，观看相关视频和可视化内容：this https URL。 

---
# Query-Centric Diffusion Policy for Generalizable Robotic Assembly 

**Title (ZH)**: 以查询为中心的扩散策略及其在通用机器人装配中的应用 

**Authors**: Ziyi Xu, Haohong Lin, Shiqi Liu, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18686)  

**Abstract**: The robotic assembly task poses a key challenge in building generalist robots due to the intrinsic complexity of part interactions and the sensitivity to noise perturbations in contact-rich settings. The assembly agent is typically designed in a hierarchical manner: high-level multi-part reasoning and low-level precise control. However, implementing such a hierarchical policy is challenging in practice due to the mismatch between high-level skill queries and low-level execution. To address this, we propose the Query-centric Diffusion Policy (QDP), a hierarchical framework that bridges high-level planning and low-level control by utilizing queries comprising objects, contact points, and skill information. QDP introduces a query-centric mechanism that identifies task-relevant components and uses them to guide low-level policies, leveraging point cloud observations to improve the policy's robustness. We conduct comprehensive experiments on the FurnitureBench in both simulation and real-world settings, demonstrating improved performance in skill precision and long-horizon success rate. In the challenging insertion and screwing tasks, QDP improves the skill-wise success rate by over 50% compared to baselines without structured queries. 

**Abstract (ZH)**: 基于查询的扩散策略（QDP）：桥接高层规划与低层控制的层次框架 

---
# Number Adaptive Formation Flight Planning via Affine Deformable Guidance in Narrow Environments 

**Title (ZH)**: 窄环境中方形可变形引导的自适应编队飞行规划 

**Authors**: Yuan Zhou, Jialiang Hou, Guangtong Xu, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18636)  

**Abstract**: Formation maintenance with varying number of drones in narrow environments hinders the convergence of planning to the desired configurations. To address this challenge, this paper proposes a formation planning method guided by Deformable Virtual Structures (DVS) with continuous spatiotemporal transformation. Firstly, to satisfy swarm safety distance and preserve formation shape filling integrity for irregular formation geometries, we employ Lloyd algorithm for uniform $\underline{PA}$rtitioning and Hungarian algorithm for $\underline{AS}$signment (PAAS) in DVS. Subsequently, a spatiotemporal trajectory involving DVS is planned using primitive-based path search and nonlinear trajectory optimization. The DVS trajectory achieves adaptive transitions with respect to a varying number of drones while ensuring adaptability to narrow environments through affine transformation. Finally, each agent conducts distributed trajectory planning guided by desired spatiotemporal positions within the DVS, while incorporating collision avoidance and dynamic feasibility requirements. Our method enables up to 15\% of swarm numbers to join or leave in cluttered environments while rapidly restoring the desired formation shape in simulation. Compared to cutting-edge formation planning method, we demonstrate rapid formation recovery capacity and environmental adaptability. Real-world experiments validate the effectiveness and resilience of our formation planning method. 

**Abstract (ZH)**: 狭窄环境中有变化无人机数量的编队维持會阻碍规划收敛到所需配置。为此，本文提出了一种由可变形虚拟结构（DVS）引导并在时空上持续变换的编队规划方法。首先，为了满足群体安全距离并保持不规则编队几何形状的整体完整性，我们采用Lloyd算法进行均匀PA分割和匈牙利算法进行AS分配（PAAS）以在DVS中实现。随后，基于.primitive.路径搜索和非线性轨迹优化，规划涉及DVS的时空轨迹。DVS轨迹能够针对变化的无人机数量实现自适应过渡，并通过仿射变换确保适应狭窄环境。最后，每个代理根据DVS内的期望时空位置进行分布式轨迹规划，同时包含碰撞规避和动态可行性要求。我们的方法在杂乱环境中使多达15%的群体数量加入或离开，并在仿真中快速恢复所需的编队形状。与最新的编队规划方法相比，我们展示了快速的编队恢复能力和环境适应性。实地实验验证了我们编队规划方法的有效性和鲁棒性。 

---
# Spatial Envelope MPC: High Performance Driving without a Reference 

**Title (ZH)**: 空间包络 MPC：无需参考轨迹的高性能驾驶 

**Authors**: Siyuan Yu, Congkai Shen, Yufei Xi, James Dallas, Michael Thompson, John Subosits, Hiroshi Yasuda, Tulga Ersal  

**Link**: [PDF](https://arxiv.org/pdf/2509.18506)  

**Abstract**: This paper presents a novel envelope based model predictive control (MPC) framework designed to enable autonomous vehicles to handle high performance driving across a wide range of scenarios without a predefined reference. In high performance autonomous driving, safe operation at the vehicle's dynamic limits requires a real time planning and control framework capable of accounting for key vehicle dynamics and environmental constraints when following a predefined reference trajectory is suboptimal or even infeasible. State of the art planning and control frameworks, however, are predominantly reference based, which limits their performance in such situations. To address this gap, this work first introduces a computationally efficient vehicle dynamics model tailored for optimization based control and a continuously differentiable mathematical formulation that accurately captures the entire drivable envelope. This novel model and formulation allow for the direct integration of dynamic feasibility and safety constraints into a unified planning and control framework, thereby removing the necessity for predefined references. The challenge of envelope planning, which refers to maximally approximating the safe drivable area, is tackled by combining reinforcement learning with optimization techniques. The framework is validated through both simulations and real world experiments, demonstrating its high performance across a variety of tasks, including racing, emergency collision avoidance and off road navigation. These results highlight the framework's scalability and broad applicability across a diverse set of scenarios. 

**Abstract (ZH)**: 基于包线的新型模型预测控制框架：无需预定义参考的高性能自主驾驶 

---
# Assistive Decision-Making for Right of Way Navigation at Uncontrolled Intersections 

**Title (ZH)**: 辅助决策在未控制交叉口通行优先导航中的应用 

**Authors**: Navya Tiwari, Joseph Vazhaeparampil, Victoria Preston  

**Link**: [PDF](https://arxiv.org/pdf/2509.18407)  

**Abstract**: Uncontrolled intersections account for a significant fraction of roadway crashes due to ambiguous right-of-way rules, occlusions, and unpredictable driver behavior. While autonomous vehicle research has explored uncertainty-aware decision making, few systems exist to retrofit human-operated vehicles with assistive navigation support. We present a driver-assist framework for right-of-way reasoning at uncontrolled intersections, formulated as a Partially Observable Markov Decision Process (POMDP). Using a custom simulation testbed with stochastic traffic agents, pedestrians, occlusions, and adversarial scenarios, we evaluate four decision-making approaches: a deterministic finite state machine (FSM), and three probabilistic planners: QMDP, POMCP, and DESPOT. Results show that probabilistic planners outperform the rule-based baseline, achieving up to 97.5 percent collision-free navigation under partial observability, with POMCP prioritizing safety and DESPOT balancing efficiency and runtime feasibility. Our findings highlight the importance of uncertainty-aware planning for driver assistance and motivate future integration of sensor fusion and environment perception modules for real-time deployment in realistic traffic environments. 

**Abstract (ZH)**: 不受控制的交叉口由于模糊的优先通行规则、遮挡和不可预测的驾驶行为，占了相当大的道路事故比例。尽管自动驾驶车辆研究探索了不确定性aware决策制定，却鲜有系统能够为人为操作的车辆提供辅助导航支持。我们提出了一种用于不受控制交叉口优先通行权推理的驾驶员辅助框架，该框架被形式化为部分可观测马尔可夫决策过程（POMDP）。使用包含随机交通代理、行人的自定义仿真测试床以及对抗性场景，我们评估了四种决策方法：确定性有限状态机（FSM），以及三种概率性规划器：QMDP、POMCP和DESPOT。结果表明，概率性规划器优于基于规则的基础方法，在部分可观测情况下实现了高达97.5%的无碰撞导航，其中POMCP侧重于安全，DESPOT则在效率和运行时可行性之间取得平衡。我们的研究结果强调了不确定性aware规划对于驾驶员辅助的重要性，并促进了将传感器融合和环境感知模块在未来实时部署到真实交通环境中的发展。 

---
# Semantic-Aware Particle Filter for Reliable Vineyard Robot Localisation 

**Title (ZH)**: 面向语义的粒子滤波在可靠酿酒葡萄园机器人定位中的应用 

**Authors**: Rajitha de Silva, Jonathan Cox, James R. Heselden, Marija Popovic, Cesar Cadena, Riccardo Polvara  

**Link**: [PDF](https://arxiv.org/pdf/2509.18342)  

**Abstract**: Accurate localisation is critical for mobile robots in structured outdoor environments, yet LiDAR-based methods often fail in vineyards due to repetitive row geometry and perceptual aliasing. We propose a semantic particle filter that incorporates stable object-level detections, specifically vine trunks and support poles into the likelihood estimation process. Detected landmarks are projected into a birds eye view and fused with LiDAR scans to generate semantic observations. A key innovation is the use of semantic walls, which connect adjacent landmarks into pseudo-structural constraints that mitigate row aliasing. To maintain global consistency in headland regions where semantics are sparse, we introduce a noisy GPS prior that adaptively supports the filter. Experiments in a real vineyard demonstrate that our approach maintains localisation within the correct row, recovers from deviations where AMCL fails, and outperforms vision-based SLAM methods such as RTAB-Map. 

**Abstract (ZH)**: 基于语义的粒子滤波在结构化户外环境中正确定位葡萄园移动机器人，克服重复行几何和感知混叠的问题 

---
# Haptic Communication in Human-Human and Human-Robot Co-Manipulation 

**Title (ZH)**: 人类与人类及人类与机器人协同操作中的触觉通信 

**Authors**: Katherine H. Allen, Chris Rogers, Elaine S. Short  

**Link**: [PDF](https://arxiv.org/pdf/2509.18327)  

**Abstract**: When a human dyad jointly manipulates an object, they must communicate about their intended motion plans. Some of that collaboration is achieved through the motion of the manipulated object itself, which we call "haptic communication." In this work, we captured the motion of human-human dyads moving an object together with one participant leading a motion plan about which the follower is uninformed. We then captured the same human participants manipulating the same object with a robot collaborator. By tracking the motion of the shared object using a low-cost IMU, we can directly compare human-human shared manipulation to the motion of those same participants interacting with the robot. Intra-study and post-study questionnaires provided participant feedback on the collaborations, indicating that the human-human collaborations are significantly more fluent, and analysis of the IMU data indicates that it captures objective differences in the motion profiles of the conditions. The differences in objective and subjective measures of accuracy and fluency between the human-human and human-robot trials motivate future research into improving robot assistants for physical tasks by enabling them to send and receive anthropomorphic haptic signals. 

**Abstract (ZH)**: 当一个人类双人组共同操作物体时，他们必须沟通他们的运动计划。部分合作通过所操作物体本身的运动实现，我们称之为“触觉通信”。在本研究中，我们记录了一名参与者主导运动计划而另一名跟随者对此未知的人类双人组共同操作物体的运动。接着，我们让相同的参与者与机器合作者共同操作相同的物体。通过使用低成本IMU追踪共享物体的运动，我们可以直接比较人类双人组共享操作与参与者与机器人互动时物体运动之间的差异。研究中的问卷和后续问卷提供了参与者对合作的反馈，表明人类双人组的合作更为流畅，IMU数据的分析显示它捕捉到了条件间运动特征的客观差异。人类双人组和人类-机器人试次在准确性和流畅性方面的客观与主观差异激励未来研究以使机器人能够发送和接收类人触觉信号，从而改进其在物理任务中的辅助能力。 

---
# A Fast Initialization Method for Neural Network Controllers: A Case Study of Image-based Visual Servoing Control for the multicopter Interception 

**Title (ZH)**: 基于图像视觉伺服控制的多旋翼拦截中神经网络控制的快速初始化方法：案例研究 

**Authors**: Chenxu Ke, Congling Tian, Kaichen Xu, Ye Li, Lingcong Bao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19110)  

**Abstract**: Reinforcement learning-based controller design methods often require substantial data in the initial training phase. Moreover, the training process tends to exhibit strong randomness and slow convergence. It often requires considerable time or high computational resources. Another class of learning-based method incorporates Lyapunov stability theory to obtain a control policy with stability guarantees. However, these methods generally require an initially stable neural network control policy at the beginning of training. Evidently, a stable neural network controller can not only serve as an initial policy for reinforcement learning, allowing the training to focus on improving controller performance, but also act as an initial state for learning-based Lyapunov control methods. Although stable controllers can be designed using traditional control theory, designers still need to have a great deal of control design knowledge to address increasingly complicated control problems. The proposed neural network rapid initialization method in this paper achieves the initial training of the neural network control policy by constructing datasets that conform to the stability conditions based on the system model. Furthermore, using the image-based visual servoing control for multicopter interception as a case study, simulations and experiments were conducted to validate the effectiveness and practical performance of the proposed method. In the experiment, the trained control policy attains a final interception velocity of 15 m/s. 

**Abstract (ZH)**: 基于强化学习的控制器设计方法常需要大量的初始训练数据，且训练过程往往表现出较强的随机性和缓慢的收敛性，需要消耗大量时间和计算资源。另一类基于学习的方法通过引入李雅普诺夫稳定性理论来获得具有稳定性的控制策略。然而，这些方法通常需要在训练之初具备一个稳定的神经网络控制策略。显然，一个稳定的神经网络控制器不仅可以用作强化学习的初始策略，使训练专注于提高控制器性能，还可以作为基于学习的李雅普诺夫控制方法的初始状态。尽管传统的控制理论可以设计出稳定的控制器，但设计者仍需具备大量的控制设计知识以应对日益复杂的控制问题。本文提出的神经网络快速初始化方法通过基于系统模型构建满足稳定性条件的数据集，实现了神经网络控制策略的初始训练。此外，以多旋翼拦截基于图像的视觉伺服控制为例，进行了仿真和实验来验证该方法的有效性和实际性能。在实验中，训练得到的控制策略获得最终拦截速度为15 m/s。 

---
# Guaranteed Robust Nonlinear MPC via Disturbance Feedback 

**Title (ZH)**: 通过干扰反馈确保的鲁棒非线性MPC 

**Authors**: Antoine P. Leeman, Johannes Köhler, Melanie N. Zeilinger  

**Link**: [PDF](https://arxiv.org/pdf/2509.18760)  

**Abstract**: Robots must satisfy safety-critical state and input constraints despite disturbances and model mismatch. We introduce a robust model predictive control (RMPC) formulation that is fast, scalable, and compatible with real-time implementation. Our formulation guarantees robust constraint satisfaction, input-to-state stability (ISS) and recursive feasibility. The key idea is to decompose the uncertain nonlinear system into (i) a nominal nonlinear dynamic model, (ii) disturbance-feedback controllers, and (iii) bounds on the model error. These components are optimized jointly using sequential convex programming. The resulting convex subproblems are solved efficiently using a recent disturbance-feedback MPC solver. The approach is validated across multiple dynamics, including a rocket-landing problem with steerable thrust. An open-source implementation is available at this https URL. 

**Abstract (ZH)**: 机器人必须在干扰和模型不匹配的情况下满足安全关键的状态和输入约束。我们介绍了一种快速、可扩展且适用于实时实现的鲁棒模型预测控制（RMPC） formulation。该 formulation 保证了鲁棒的约束满足、输入状态稳定性（ISS）和递归可行性。关键思想是将不确定的非线性系统分解为（i）名义非线性动态模型，（ii）干扰反馈控制器，和（iii）模型误差的界。这些组成部分是通过序列凸规划联合优化的。由此产生的凸子问题通过最近的干扰反馈 MPC 解算器高效求解。该方法跨多种动力学进行了验证，包括具有可转向推力的火箭着陆问题。开源实现可在以下链接获得：this https URL。 

---
# Dual Iterative Learning Control for Multiple-Input Multiple-Output Dynamics with Validation in Robotic Systems 

**Title (ZH)**: 基于验证的多输入多输出动力学的双迭代学习控制 

**Authors**: Jan-Hendrik Ewering, Alessandro Papa, Simon F.G. Ehlers, Thomas Seel, Michael Meindl  

**Link**: [PDF](https://arxiv.org/pdf/2509.18723)  

**Abstract**: Solving motion tasks autonomously and accurately is a core ability for intelligent real-world systems. To achieve genuine autonomy across multiple systems and tasks, key challenges include coping with unknown dynamics and overcoming the need for manual parameter tuning, which is especially crucial in complex Multiple-Input Multiple-Output (MIMO) systems.
This paper presents MIMO Dual Iterative Learning Control (DILC), a novel data-driven iterative learning scheme for simultaneous tracking control and model learning, without requiring any prior system knowledge or manual parameter tuning. The method is designed for repetitive MIMO systems and integrates seamlessly with established iterative learning control methods. We provide monotonic convergence conditions for both reference tracking error and model error in linear time-invariant systems.
The DILC scheme -- rapidly and autonomously -- solves various motion tasks in high-fidelity simulations of an industrial robot and in multiple nonlinear real-world MIMO systems, without requiring model knowledge or manually tuning the algorithm. In our experiments, many reference tracking tasks are solved within 10-20 trials, and even complex motions are learned in less than 100 iterations. We believe that, because of its rapid and autonomous learning capabilities, DILC has the potential to serve as an efficient building block within complex learning frameworks for intelligent real-world systems. 

**Abstract (ZH)**: 自主准确地解决运动任务是智能现实系统的一项核心能力。为了在多个系统和任务中实现真正的自主性，关键挑战包括应对未知动态和克服手动参数调谐的需要，尤其是在复杂的多输入多输出（MIMO）系统中。

本文提出了一种新颖的数据驱动迭代学习控制方案MIMO双迭代学习控制（DILC），该方案无需任何先验系统知识或手动参数调谐，同时实现了跟踪控制和模型学习。该方法适用于重复的MIMO系统，并可无缝集成到现有的迭代学习控制方法中。我们为线性时不变系统提供了参考跟踪误差和模型误差的单调收敛条件。

DILC方案在高保真工业机器人模拟和多个非线性实际MIMO系统中快速自主地解决各种运动任务，无需模型知识或手动调谐算法。在我们的实验中，许多参考跟踪任务在10-20次迭代内得到解决，甚至复杂的运动在不到100次迭代内也被学习。我们相信，由于其快速自主的学习能力，DILC有望成为用于智能现实系统复杂学习框架的有效构建块。 

---
