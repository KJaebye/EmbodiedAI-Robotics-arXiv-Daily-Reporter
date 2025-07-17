# Design and Development of an Automated Contact Angle Tester (ACAT) for Surface Wettability Measurement 

**Title (ZH)**: 自动接触角测试仪（ACAT）的设计与开发：表面湿润性测量 

**Authors**: Connor Burgess, Kyle Douin, Amir Kordijazi  

**Link**: [PDF](https://arxiv.org/pdf/2507.12431)  

**Abstract**: The Automated Contact Angle Tester (ACAT) is a fully integrated robotic work cell developed to automate the measurement of surface wettability on 3D-printed materials. Designed for precision, repeatability, and safety, ACAT addresses the limitations of manual contact angle testing by combining programmable robotics, precise liquid dispensing, and a modular software-hardware architecture. The system is composed of three core subsystems: (1) an electrical system including power, control, and safety circuits compliant with industrial standards such as NEC 70, NFPA 79, and UL 508A; (2) a software control system based on a Raspberry Pi and Python, featuring fault detection, GPIO logic, and operator interfaces; and (3) a mechanical system that includes a 3-axis Cartesian robot, pneumatic actuation, and a precision liquid dispenser enclosed within a safety-certified frame. The ACAT enables high-throughput, automated surface characterization and provides a robust platform for future integration into smart manufacturing and materials discovery workflows. This paper details the design methodology, implementation strategies, and system integration required to develop the ACAT platform. 

**Abstract (ZH)**: 自动化接触角测试仪（ACAT）：一种用于自动化3D打印材料表面润湿性测量的集成机器人工作站 

---
# Robust Route Planning for Sidewalk Delivery Robots 

**Title (ZH)**: 侧路边缘配送机器人鲁棒路径规划 

**Authors**: Xing Tong, Michele D. Simoni  

**Link**: [PDF](https://arxiv.org/pdf/2507.12067)  

**Abstract**: Sidewalk delivery robots are a promising solution for urban freight distribution, reducing congestion compared to trucks and providing a safer, higher-capacity alternative to drones. However, unreliable travel times on sidewalks due to pedestrian density, obstacles, and varying infrastructure conditions can significantly affect their efficiency. This study addresses the robust route planning problem for sidewalk robots, explicitly accounting for travel time uncertainty due to varying sidewalk conditions. Optimization is integrated with simulation to reproduce the effect of obstacles and pedestrian flows and generate realistic travel times. The study investigates three different approaches to derive uncertainty sets, including budgeted, ellipsoidal, and support vector clustering (SVC)-based methods, along with a distributionally robust method to solve the shortest path (SP) problem. A realistic case study reproducing pedestrian patterns in Stockholm's city center is used to evaluate the efficiency of robust routing across various robot designs and environmental conditions. The results show that, when compared to a conventional SP, robust routing significantly enhances operational reliability under variable sidewalk conditions. The Ellipsoidal and DRSP approaches outperform the other methods, yielding the most efficient paths in terms of average and worst-case delay. Sensitivity analyses reveal that robust approaches consistently outperform the conventional SP, particularly for sidewalk delivery robots that are wider, slower, and have more conservative navigation behaviors. These benefits are even more pronounced in adverse weather conditions and high pedestrian congestion scenarios. 

**Abstract (ZH)**: 人行道配送机器人在城市货运分配中的稳健路径规划研究：考虑不同人行道条件下的 travel 时间不确定性 

---
# NemeSys: An Online Underwater Explorer with Goal-Driven Adaptive Autonomy 

**Title (ZH)**: NemeSys: 一种基于目标导向自适应自主性的在线水下探索器 

**Authors**: Adnan Abdullah, Alankrit Gupta, Vaishnav Ramesh, Shivali Patel, Md Jahidul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2507.11889)  

**Abstract**: Adaptive mission control and dynamic parameter reconfiguration are essential for autonomous underwater vehicles (AUVs) operating in GPS-denied, communication-limited marine environments. However, most current AUV platforms execute static, pre-programmed missions or rely on tethered connections and high-latency acoustic channels for mid-mission updates, significantly limiting their adaptability and responsiveness. In this paper, we introduce NemeSys, a novel AUV system designed to support real-time mission reconfiguration through compact optical and magnetoelectric (OME) signaling facilitated by floating buoys. We present the full system design, control architecture, and a semantic mission encoding framework that enables interactive exploration and task adaptation via low-bandwidth communication. The proposed system is validated through analytical modeling, controlled experimental evaluations, and open-water trials. Results confirm the feasibility of online mission adaptation and semantic task updates, highlighting NemeSys as an online AUV platform for goal-driven adaptive autonomy in dynamic and uncertain underwater environments. 

**Abstract (ZH)**: 自适应任务控制和动态参数重构对于在GPS受限、通信受限的海洋环境中运行的自主水下 vehicle (AUVs) 至关重要。然而，当前大多数 AUV 平台执行静态、预先编程的任务，或者依赖于有缆连接和高延迟声学信道的中途更新，这严重限制了它们的适应性和响应性。在本文中，我们介绍了 NemeSys，这是一种新型 AUV 系统，设计用于通过浮标辅助的紧凑光学和磁电 (OME) 信号进行实时任务重构。我们介绍了整个系统设计、控制架构以及一种语义任务编码框架，该框架能够通过低带宽通信实现互动探索和任务适应。所提出的系统通过分析建模、受控实验评估和开放水域试验进行了验证。结果证实了在线任务适应和语义任务更新的可行性，突显了 NemeSys 作为目标驱动的自适应自主 AUV 平台在动态和不确定的水下环境中的在线应用。 

---
# A Fast Method for Planning All Optimal Homotopic Configurations for Tethered Robots and Its Extended Applications 

**Title (ZH)**: 一种快速规划 tethered 机器人所有最优同伦配置的方法及其扩展应用 

**Authors**: Jinyuan Liu, Minglei Fu, Ling Shi, Chenguang Yang, Wenan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11880)  

**Abstract**: Tethered robots play a pivotal role in specialized environments such as disaster response and underground exploration, where their stable power supply and reliable communication offer unparalleled advantages. However, their motion planning is severely constrained by tether length limitations and entanglement risks, posing significant challenges to achieving optimal path planning. To address these challenges, this study introduces CDT-TCS (Convex Dissection Topology-based Tethered Configuration Search), a novel algorithm that leverages CDT Encoding as a homotopy invariant to represent topological states of paths. By integrating algebraic topology with geometric optimization, CDT-TCS efficiently computes the complete set of optimal feasible configurations for tethered robots at all positions in 2D environments through a single computation. Building on this foundation, we further propose three application-specific algorithms: i) CDT-TPP for optimal tethered path planning, ii) CDT-TMV for multi-goal visiting with tether constraints, iii) CDT-UTPP for distance-optimal path planning of untethered robots. All theoretical results and propositions underlying these algorithms are rigorously proven and thoroughly discussed in this paper. Extensive simulations demonstrate that the proposed algorithms significantly outperform state-of-the-art methods in their respective problem domains. Furthermore, real-world experiments on robotic platforms validate the practicality and engineering value of the proposed framework. 

**Abstract (ZH)**: 拴系机器人在灾害响应和地下探索等专业环境中扮演着重要角色，它们稳定的电源供应和可靠的通信提供了无与伦比的优势。然而，它们的运动规划受到了牵引线长度限制和缠绕风险的严重制约，给最优路径规划带来了重大挑战。为应对这些挑战，本研究引入了一种新的算法CDT-TCS（基于凸剖分拓扑的拴系配置搜索），该算法利用CDT编码作为同伦不变量来表示路径的拓扑状态。通过将代数拓扑与几何优化相结合，CDT-TCS能够在单次计算中高效地确定2D环境中所有位置下拴系机器人的完整最优可行配置集。在此基础上，进一步提出了三种特定应用算法：i) CDT-TPP（最优拴系路径规划），ii) CDT-TMV（带拴系约束的多目标访问），iii) CDT-UTPP（无牵引距离最优路径规划）。本研究中所有这些算法的理论结果和命题均得到了严格证明并进行了详细讨论。大量仿真实验表明，所提出算法在各自的问题领域显著优于现有最先进的方法。此外，基于实际机器人平台的实验进一步验证了该框架的实用性和工程价值。 

---
# Towards Autonomous Riding: A Review of Perception, Planning, and Control in Intelligent Two-Wheelers 

**Title (ZH)**: 面向自主骑行：智能两轮车感知、规划与控制综述 

**Authors**: Mohammed Hassanin, Mohammad Abu Alsheikh, Carlos C. N. Kuhn, Damith Herath, Dinh Thai Hoang, Ibrahim Radwan  

**Link**: [PDF](https://arxiv.org/pdf/2507.11852)  

**Abstract**: The rapid adoption of micromobility solutions, particularly two-wheeled vehicles like e-scooters and e-bikes, has created an urgent need for reliable autonomous riding (AR) technologies. While autonomous driving (AD) systems have matured significantly, AR presents unique challenges due to the inherent instability of two-wheeled platforms, limited size, limited power, and unpredictable environments, which pose very serious concerns about road users' safety. This review provides a comprehensive analysis of AR systems by systematically examining their core components, perception, planning, and control, through the lens of AD technologies. We identify critical gaps in current AR research, including a lack of comprehensive perception systems for various AR tasks, limited industry and government support for such developments, and insufficient attention from the research community. The review analyses the gaps of AR from the perspective of AD to highlight promising research directions, such as multimodal sensor techniques for lightweight platforms and edge deep learning architectures. By synthesising insights from AD research with the specific requirements of AR, this review aims to accelerate the development of safe, efficient, and scalable autonomous riding systems for future urban mobility. 

**Abstract (ZH)**: 微移动性解决方案的快速采用，特别是电动滑板车和电动自行车等两轮车辆，迫切需要可靠的自主骑行（AR）技术。虽然自动驾驶（AD）系统已显著成熟，但由于两轮平台的固有不稳定性和有限的空间、功率以及不可预测的环境等因素，AR面临着独特的挑战，这给道路使用者的安全带来了严重关切。本综述通过自动驾驶技术的视角系统分析了AR系统的核心组件、感知、规划和控制等方面，指出了当前AR研究中的关键空白，包括缺乏全面的感知系统、行业和政府对该领域的支持有限以及研究社区对此关注不足。本综述从自动驾驶的角度分析AR的不足之处，以突出多模式传感器技术和边缘深度学习架构等有前景的研究方向。通过将自动驾驶研究的洞见与AR的具体需求相结合，本综旨在加速安全、高效和可扩展的自主骑行系统的开发，以为未来的城市 mobility 提供支持。 

---
# CoNav Chair: Development and Evaluation of a Shared Control based Wheelchair for the Built Environment 

**Title (ZH)**: CoNav轮椅：基于共享控制的建筑物环境轮椅的研发与评估 

**Authors**: Yifan Xu, Qianwei Wang, Jordan Lillie, Vineet Kamat, Carol Menassa, Clive D'Souza  

**Link**: [PDF](https://arxiv.org/pdf/2507.11716)  

**Abstract**: As the global population of people with disabilities (PWD) continues to grow, so will the need for mobility solutions that promote independent living and social integration. Wheelchairs are vital for the mobility of PWD in both indoor and outdoor environments. The current SOTA in powered wheelchairs is based on either manually controlled or fully autonomous modes of operation, offering limited flexibility and often proving difficult to navigate in spatially constrained environments. Moreover, research on robotic wheelchairs has focused predominantly on complete autonomy or improved manual control; approaches that can compromise efficiency and user trust. To overcome these challenges, this paper introduces the CoNav Chair, a smart wheelchair based on the Robot Operating System (ROS) and featuring shared control navigation and obstacle avoidance capabilities that are intended to enhance navigational efficiency, safety, and ease of use for the user. The paper outlines the CoNav Chair's design and presents a preliminary usability evaluation comparing three distinct navigation modes, namely, manual, shared, and fully autonomous, conducted with 21 healthy, unimpaired participants traversing an indoor building environment. Study findings indicated that the shared control navigation framework had significantly fewer collisions and performed comparably, if not superior to the autonomous and manual modes, on task completion time, trajectory length, and smoothness; and was perceived as being safer and more efficient based on user reported subjective assessments of usability. Overall, the CoNav system demonstrated acceptable safety and performance, laying the foundation for subsequent usability testing with end users, namely, PWDs who rely on a powered wheelchair for mobility. 

**Abstract (ZH)**: 随着全球残疾人（PWD）人口的不断增长，对促进独立生活和社会融合的移动解决方案的需求也将不断增加。轮椅对于残疾人室内和室外环境中的移动至关重要。目前的最先进动力轮椅技术基于手动控制或完全自主操作模式，灵活性有限，往往在空间受限的环境中难以导航。此外，关于机器人轮椅的研究主要集中在完全自主或改进的手动控制上；这些方法可能会影响效率和用户的信任。为克服这些挑战，本文介绍了基于Robot Operating System (ROS)的CoNav Chair，这是一种具备共控制导航和障碍物避免能力的智能轮椅，旨在提高用户导航效率、安全性和易用性。论文概述了CoNav Chair的设计，并进行了初步的可用性评估，比较了三种不同的导航模式——手动、共控制和完全自主，共21名健康无缺陷的参与者在室内建筑环境中进行了评估。研究发现表明，共控制导航框架的碰撞次数明显较少，并在任务完成时间、路径长度和流畅性方面表现与自主和手动模式相当甚至更优；并且根据用户主观评估的可用性报告，共控制模式被认为更加安全和高效。总体而言，CoNav系统展示了可接受的安全性和性能，为后续与最终用户即依赖动力轮椅的残疾人进行的可用性测试奠定了基础。 

---
