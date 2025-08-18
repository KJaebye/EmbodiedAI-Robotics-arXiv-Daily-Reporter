# Investigating Sensors and Methods in Grasp State Classification in Agricultural Manipulation 

**Title (ZH)**: 研究农业 manipulation 中抓取状态分类的传感器与方法 

**Authors**: Benjamin Walt, Jordan Westphal, Girish Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11588)  

**Abstract**: Effective and efficient agricultural manipulation and harvesting depend on accurately understanding the current state of the grasp. The agricultural environment presents unique challenges due to its complexity, clutter, and occlusion. Additionally, fruit is physically attached to the plant, requiring precise separation during harvesting. Selecting appropriate sensors and modeling techniques is critical for obtaining reliable feedback and correctly identifying grasp states. This work investigates a set of key sensors, namely inertial measurement units (IMUs), infrared (IR) reflectance, tension, tactile sensors, and RGB cameras, integrated into a compliant gripper to classify grasp states. We evaluate the individual contribution of each sensor and compare the performance of two widely used classification models: Random Forest and Long Short-Term Memory (LSTM) networks. Our results demonstrate that a Random Forest classifier, trained in a controlled lab environment and tested on real cherry tomato plants, achieved 100% accuracy in identifying slip, grasp failure, and successful picks, marking a substantial improvement over baseline performance. Furthermore, we identify a minimal viable sensor combination, namely IMU and tension sensors that effectively classifies grasp states. This classifier enables the planning of corrective actions based on real-time feedback, thereby enhancing the efficiency and reliability of fruit harvesting operations. 

**Abstract (ZH)**: 有效的农业操作和收获依赖于对当前握持状态的准确理解。农业环境因其复杂性、混乱和遮挡而提出独特的挑战。此外，水果物理上附着在植物上，要求在收获过程中进行精确分离。选择适当的传感器和建模技术对于获取可靠反馈和正确识别握持状态至关重要。本研究探讨了一组关键传感器，即惯性测量单元（IMUs）、红外（IR）反射率、张力、触觉传感器和RGB相机，集成到顺应性夹爪中以分类握持状态。我们评估了每个传感器的个体贡献，并比较了两种广泛使用的分类模型：随机森林和长短期记忆（LSTM）网络的性能。我们的结果显示，随机森林分类器在可控实验室环境中训练，在实际樱桃番茄植株上测试，实现了对滑落、握持失败和成功收获的100%准确识别，显著优于基线性能。此外，我们确定了一种最小可行的传感器组合，即IMU和张力传感器，能够有效分类握持状态。该分类器使得基于实时反馈规划纠正措施成为可能，从而提高了水果收获操作的效率和可靠性。 

---
# Towards Fully Onboard State Estimation and Trajectory Tracking for UAVs with Suspended Payloads 

**Title (ZH)**: 面向悬挂载荷的无人机的完全载荷状态下状态估计与轨迹跟踪研究 

**Authors**: Martin Jiroušek, Tomáš Báča, Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2508.11547)  

**Abstract**: This paper addresses the problem of tracking the position of a cable-suspended payload carried by an unmanned aerial vehicle, with a focus on real-world deployment and minimal hardware requirements. In contrast to many existing approaches that rely on motion-capture systems, additional onboard cameras, or instrumented payloads, we propose a framework that uses only standard onboard sensors--specifically, real-time kinematic global navigation satellite system measurements and data from the onboard inertial measurement unit--to estimate and control the payload's position. The system models the full coupled dynamics of the aerial vehicle and payload, and integrates a linear Kalman filter for state estimation, a model predictive contouring control planner, and an incremental model predictive controller. The control architecture is designed to remain effective despite sensing limitations and estimation uncertainty. Extensive simulations demonstrate that the proposed system achieves performance comparable to control based on ground-truth measurements, with only minor degradation (< 6%). The system also shows strong robustness to variations in payload parameters. Field experiments further validate the framework, confirming its practical applicability and reliable performance in outdoor environments using only off-the-shelf aerial vehicle hardware. 

**Abstract (ZH)**: 本文解决了由无人驾驶航空器承载的悬挂电缆载荷位置跟踪问题，着重于实际部署并与最少硬件要求相结合。与许多现有的依赖于运动捕捉系统、附加机载摄像头或配备传感器的载荷的方法不同，我们提出了一种仅使用标准机载传感器（具体而言是实时动态全球定位系统测量和机载惯性测量单元的数据）来估计和控制载荷位置的框架。该系统模型涵盖了空中飞行器和载荷的完整耦合动态，集成了线性卡尔曼滤波器进行状态估计、模型预测包络控制规划器以及增量模型预测控制器。控制架构设计旨在即使在存在传感限制和估计不确定性的情况下仍能保持有效性。大量仿真表明，所提出系统在基于地面真实测量的控制性能方面具有可比性，仅轻微降解 (< 6%)。该系统还表现出对载荷参数变化的强大鲁棒性。现场实验进一步验证了该框架，在仅使用即插即用的飞行器硬件的户外环境中证实了其实用可行性和可靠性能。 

---
# A Comparative Study of Floating-Base Space Parameterizations for Agile Whole-Body Motion Planning 

**Title (ZH)**: 浮基空间参数化对比研究：敏捷全身运动规划 

**Authors**: Evangelos Tsiatsianas, Chairi Kiourt, Konstantinos Chatzilygeroudis  

**Link**: [PDF](https://arxiv.org/pdf/2508.11520)  

**Abstract**: Automatically generating agile whole-body motions for legged and humanoid robots remains a fundamental challenge in robotics. While numerous trajectory optimization approaches have been proposed, there is no clear guideline on how the choice of floating-base space parameterization affects performance, especially for agile behaviors involving complex contact dynamics. In this paper, we present a comparative study of different parameterizations for direct transcription-based trajectory optimization of agile motions in legged systems. We systematically evaluate several common choices under identical optimization settings to ensure a fair comparison. Furthermore, we introduce a novel formulation based on the tangent space of SE(3) for representing the robot's floating-base pose, which, to our knowledge, has not received attention from the literature. This approach enables the use of mature off-the-shelf numerical solvers without requiring specialized manifold optimization techniques. We hope that our experiments and analysis will provide meaningful insights for selecting the appropriate floating-based representation for agile whole-body motion generation. 

**Abstract (ZH)**: 自动生成腿式和类人机器人灵活全身运动仍然是机器人领域的一项基本挑战。尽管提出了众多轨迹优化方法，但关于浮动基空间参数化选择如何影响性能的指导性原则尚不明确，尤其对于涉及复杂接触动力学的灵活行为。本文对直接转录法轨迹优化中灵活运动的不同参数化进行了比较研究。我们系统地在相同的优化设置下评估了几种常见的选择，以确保公平比较。此外，我们引入了一种基于SE(3)切空间的新形式，用于表示机器人的浮动基姿态，据我们所知，这种表示方法尚未受到文献关注。这种方法允许使用成熟的现成数值求解器，而无需专门的流形优化技术。我们希望我们的实验和分析能够为选择合适的浮动基表示以生成灵活的全身运动提供有价值的见解。 

---
# Sim2Dust: Mastering Dynamic Waypoint Tracking on Granular Media 

**Title (ZH)**: Sim2Dust: 在颗粒介质中掌握动态途经点跟踪技能 

**Authors**: Andrej Orsula, Matthieu Geist, Miguel Olivares-Mendez, Carol Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2508.11503)  

**Abstract**: Reliable autonomous navigation across the unstructured terrains of distant planetary surfaces is a critical enabler for future space exploration. However, the deployment of learning-based controllers is hindered by the inherent sim-to-real gap, particularly for the complex dynamics of wheel interactions with granular media. This work presents a complete sim-to-real framework for developing and validating robust control policies for dynamic waypoint tracking on such challenging surfaces. We leverage massively parallel simulation to train reinforcement learning agents across a vast distribution of procedurally generated environments with randomized physics. These policies are then transferred zero-shot to a physical wheeled rover operating in a lunar-analogue facility. Our experiments systematically compare multiple reinforcement learning algorithms and action smoothing filters to identify the most effective combinations for real-world deployment. Crucially, we provide strong empirical evidence that agents trained with procedural diversity achieve superior zero-shot performance compared to those trained on static scenarios. We also analyze the trade-offs of fine-tuning with high-fidelity particle physics, which offers minor gains in low-speed precision at a significant computational cost. Together, these contributions establish a validated workflow for creating reliable learning-based navigation systems, marking a critical step towards deploying autonomous robots in the final frontier. 

**Abstract (ZH)**: 可靠的自主导航穿越遥远行星表面的未结构化地形是未来空间探索的关键使能器。然而，基于学习的控制器控制器控制器控制器的部署受到固有的仿真到到到现实差距的阻碍，尤其是对于复杂的动力学交互，特别是在与颗粒介质的交互中。本文提出了一种完整的从仿真到到现实的框架，用于在如此具有挑战性的表面上开发和验证动态航路点跟踪的稳健控制策略。该框架基于大量并行仿真来训练强化学习代理，并在使用程序生成环境的分布上随机物理参数。然后，这些策略在月球类比设施中的实际轮式 rover 上进行了零样本迁移。我们系统研究了多种强化学习算法和动作平滑滤波器，以确定最适合实际部署的组合。 crucially 我们提供了强有力的实证证据，证明在程序多样性上上 场景上 上训练的代理 Zero-shot �態現表现优于仅在少数特定场景上 培训的代理。我们还 还分析了与高保真度 physics 在上的微调的权衡，这提供了在高速度精度上的的微小收益，但需要大量计算资源。我们的贡献一起建立了一种验证的流程框架，用于创建可靠的的基于学习的系统，朝着部署自主机器人在新兴领域迈出了关键一步。 

---
# Swarm-in-Blocks: Simplifying Drone Swarm Programming with Block-Based Language 

**Title (ZH)**: 块中群集：基于模块化语言简化无人机群集编程 

**Authors**: Agnes Bressan de Almeida, Joao Aires Correa Fernandes Marsicano  

**Link**: [PDF](https://arxiv.org/pdf/2508.11498)  

**Abstract**: Swarm in Blocks, originally developed for CopterHack 2022, is a high-level interface that simplifies drone swarm programming using a block-based language. Building on the Clover platform, this tool enables users to create functionalities like loops and conditional structures by assembling code blocks. In 2023, we introduced Swarm in Blocks 2.0, further refining the platform to address the complexities of swarm management in a user-friendly way. As drone swarm applications grow in areas like delivery, agriculture, and surveillance, the challenge of managing them, especially for beginners, has also increased. The Atena team developed this interface to make swarm handling accessible without requiring extensive knowledge of ROS or programming. The block-based approach not only simplifies swarm control but also expands educational opportunities in programming. 

**Abstract (ZH)**: 块中 swarm：一种简化无人机 swarm 编程的积木式接口 

---
# i2Nav-Robot: A Large-Scale Indoor-Outdoor Robot Dataset for Multi-Sensor Fusion Navigation and Mapping 

**Title (ZH)**: i2Nav-Robot: 一种用于多传感器融合导航与建图的室内-室外机器人数据集 

**Authors**: Hailiang Tang, Tisheng Zhang, Liqiang Wang, Xin Ding, Man Yuan, Zhiyu Xiang, Jujin Chen, Yuhan Bian, Shuangyan Liu, Yuqing Wang, Guan Wang, Xiaoji Niu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11485)  

**Abstract**: Accurate and reliable navigation is crucial for autonomous unmanned ground vehicle (UGV). However, current UGV datasets fall short in meeting the demands for advancing navigation and mapping techniques due to limitations in sensor configuration, time synchronization, ground truth, and scenario diversity. To address these challenges, we present i2Nav-Robot, a large-scale dataset designed for multi-sensor fusion navigation and mapping in indoor-outdoor environments. We integrate multi-modal sensors, including the newest front-view and 360-degree solid-state LiDARs, 4-dimensional (4D) radar, stereo cameras, odometer, global navigation satellite system (GNSS) receiver, and inertial measurement units (IMU) on an omnidirectional wheeled robot. Accurate timestamps are obtained through both online hardware synchronization and offline calibration for all sensors. The dataset comprises ten larger-scale sequences covering diverse UGV operating scenarios, such as outdoor streets, and indoor parking lots, with a total length of about 17060 meters. High-frequency ground truth, with centimeter-level accuracy for position, is derived from post-processing integrated navigation methods using a navigation-grade IMU. The proposed i2Nav-Robot dataset is evaluated by more than ten open-sourced multi-sensor fusion systems, and it has proven to have superior data quality. 

**Abstract (ZH)**: 面向室内-室外环境的多传感器融合导航与mapping的i2Nav-Robot大型数据集 

---
# Open, Reproducible and Trustworthy Robot-Based Experiments with Virtual Labs and Digital-Twin-Based Execution Tracing 

**Title (ZH)**: 基于虚拟实验室和基于数字孪生的执行追踪的开放、可再现和可信赖的机器人实验 

**Authors**: Benjamin Alt, Mareike Picklum, Sorin Arion, Franklin Kenghagho Kenfack, Michael Beetz  

**Link**: [PDF](https://arxiv.org/pdf/2508.11406)  

**Abstract**: We envision a future in which autonomous robots conduct scientific experiments in ways that are not only precise and repeatable, but also open, trustworthy, and transparent. To realize this vision, we present two key contributions: a semantic execution tracing framework that logs sensor data together with semantically annotated robot belief states, ensuring that automated experimentation is transparent and replicable; and the AICOR Virtual Research Building (VRB), a cloud-based platform for sharing, replicating, and validating robot task executions at scale. Together, these tools enable reproducible, robot-driven science by integrating deterministic execution, semantic memory, and open knowledge representation, laying the foundation for autonomous systems to participate in scientific discovery. 

**Abstract (ZH)**: 我们设想一个未来，在自主机器人将以不仅精确可 

---
# An Exploratory Study on Crack Detection in Concrete through Human-Robot Collaboration 

**Title (ZH)**: 通过人机协作的混凝土裂缝检测探索性研究 

**Authors**: Junyeon Kim, Tianshu Ruan, Cesar Alan Contreras, Manolis Chiou  

**Link**: [PDF](https://arxiv.org/pdf/2508.11404)  

**Abstract**: Structural inspection in nuclear facilities is vital for maintaining operational safety and integrity. Traditional methods of manual inspection pose significant challenges, including safety risks, high cognitive demands, and potential inaccuracies due to human limitations. Recent advancements in Artificial Intelligence (AI) and robotic technologies have opened new possibilities for safer, more efficient, and accurate inspection methodologies. Specifically, Human-Robot Collaboration (HRC), leveraging robotic platforms equipped with advanced detection algorithms, promises significant improvements in inspection outcomes and reductions in human workload. This study explores the effectiveness of AI-assisted visual crack detection integrated into a mobile Jackal robot platform. The experiment results indicate that HRC enhances inspection accuracy and reduces operator workload, resulting in potential superior performance outcomes compared to traditional manual methods. 

**Abstract (ZH)**: 核设施结构检查对于维持运营安全和完整性至关重要。传统的手动检查方法面临显著挑战，包括安全风险、高认知负荷以及由于人类限制可能导致的不准确性。最近在人工智能（AI）和机器人技术方面的进步为更安全、更高效和更准确的检查方法打开了新的可能性。特别是通过利用装有先进检测算法的机器人平台实现的人机协作（HRC），有望在检查结果和减少人力工作量方面取得显著改善。本研究探讨了将AI辅助的裂纹检测集成到移动Jackal机器人平台中的有效性。实验结果表明，HRC能够提高检查准确性并减轻操作员的工作负担，从而可能在性能方面超过传统的手动方法。 

---
# Pedestrian Dead Reckoning using Invariant Extended Kalman Filter 

**Title (ZH)**: 行人无迹卡尔曼滤波定位 

**Authors**: Jingran Zhang, Zhengzhang Yan, Yiming Chen, Zeqiang He, Jiahao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11396)  

**Abstract**: This paper presents a cost-effective inertial pedestrian dead reckoning method for the bipedal robot in the GPS-denied environment. Each time when the inertial measurement unit (IMU) is on the stance foot, a stationary pseudo-measurement can be executed to provide innovation to the IMU measurement based prediction. The matrix Lie group based theoretical development of the adopted invariant extended Kalman filter (InEKF) is set forth for tutorial purpose. Three experiments are conducted to compare between InEKF and standard EKF, including motion capture benchmark experiment, large-scale multi-floor walking experiment, and bipedal robot experiment, as an effort to show our method's feasibility in real-world robot system. In addition, a sensitivity analysis is included to show that InEKF is much easier to tune than EKF. 

**Abstract (ZH)**: 本论文提出了一种适用于GPS受限环境的双足机器人低成本惯性步行 dead reckoning 方法。每次惯性测量单元(IMU)位于支撑脚时，可以执行一个静止伪测量以提供基于IMU测量的预测创新。本文为了教学目的，提供了基于矩阵李群的采用不变扩展卡尔曼滤波器(InEKF)的理论发展。进行了三项实验，将InEKF与标准卡尔曼滤波器(EKF)进行比较，包括运动捕捉基准实验、大规模多层行走实验和双足机器人实验，以展示本方法在实际机器人系统中的可行性。此外，还进行了敏感性分析，表明InEKF比EKF更容易调优。 

---
# Tactile Robotics: An Outlook 

**Title (ZH)**: 触觉机器人：前景展望 

**Authors**: Shan Luo, Nathan F. Lepora, Wenzhen Yuan, Kaspar Althoefer, Gordon Cheng, Ravinder Dahiya  

**Link**: [PDF](https://arxiv.org/pdf/2508.11261)  

**Abstract**: Robotics research has long sought to give robots the ability to perceive the physical world through touch in an analogous manner to many biological systems. Developing such tactile capabilities is important for numerous emerging applications that require robots to co-exist and interact closely with humans. Consequently, there has been growing interest in tactile sensing, leading to the development of various technologies, including piezoresistive and piezoelectric sensors, capacitive sensors, magnetic sensors, and optical tactile sensors. These diverse approaches utilise different transduction methods and materials to equip robots with distributed sensing capabilities, enabling more effective physical interactions. These advances have been supported in recent years by simulation tools that generate large-scale tactile datasets to support sensor designs and algorithms to interpret and improve the utility of tactile data. The integration of tactile sensing with other modalities, such as vision, as well as with action strategies for active tactile perception highlights the growing scope of this field. To further the transformative progress in tactile robotics, a holistic approach is essential. In this outlook article, we examine several challenges associated with the current state of the art in tactile robotics and explore potential solutions to inspire innovations across multiple domains, including manufacturing, healthcare, recycling and agriculture. 

**Abstract (ZH)**: 机器人研究长期致力于赋予机器人通过触觉感知物理世界的能力，以此类比许多生物系统。开发此类触觉能力对于众多需要机器人与人类密切共存和交互的应用来说非常重要。因此，对触觉感知的兴趣不断增加，引领了各种技术的发展，包括压阻式和压电式传感器、电容式传感器、磁性传感器和光学触觉传感器。这些多样的方法利用不同的转换机制和材料，为机器人配备了分布式感知能力，从而使物理交互更加有效。近年来，通过生成大规模触觉数据集的支持，模拟工具和解释并改进触觉数据的算法的进步为这一领域的发展提供了支持。将触觉感知与其他模态，如视觉，以及用于主动触觉感知的动作策略的整合，突显了该领域的日益广泛的范围。为了进一步推动触觉机器人领域的变革性进展，需要一个全面的方法。在这篇展望文章中，我们探讨了当前触觉机器人技术面临的几个挑战，并探索可能的解决方案，以激发制造、医疗、回收和农业等多个领域的创新。 

---
# Robust Online Calibration for UWB-Aided Visual-Inertial Navigation with Bias Correction 

**Title (ZH)**: Robust Online Calibration for UWB-Aided Visual-Inertial Navigation with Bias Correction 

**Authors**: Yizhi Zhou, Jie Xu, Jiawei Xia, Zechen Hu, Weizi Li, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10999)  

**Abstract**: This paper presents a novel robust online calibration framework for Ultra-Wideband (UWB) anchors in UWB-aided Visual-Inertial Navigation Systems (VINS). Accurate anchor positioning, a process known as calibration, is crucial for integrating UWB ranging measurements into state estimation. While several prior works have demonstrated satisfactory results by using robot-aided systems to autonomously calibrate UWB systems, there are still some limitations: 1) these approaches assume accurate robot localization during the initialization step, ignoring localization errors that can compromise calibration robustness, and 2) the calibration results are highly sensitive to the initial guess of the UWB anchors' positions, reducing the practical applicability of these methods in real-world scenarios. Our approach addresses these challenges by explicitly incorporating the impact of robot localization uncertainties into the calibration process, ensuring robust initialization. To further enhance the robustness of the calibration results against initialization errors, we propose a tightly-coupled Schmidt Kalman Filter (SKF)-based online refinement method, making the system suitable for practical applications. Simulations and real-world experiments validate the improved accuracy and robustness of our approach. 

**Abstract (ZH)**: 基于UWB辅助视觉-惯性导航系统的稳健在线校准框架：UWB锚点的校准研究 

---
# Developing and Validating a High-Throughput Robotic System for the Accelerated Development of Porous Membranes 

**Title (ZH)**: 开发并验证一种高通量机器人系统，用于加速构建多孔膜的研究 

**Authors**: Hongchen Wang, Sima Zeinali Danalou, Jiahao Zhu, Kenneth Sulimro, Chaewon Lim, Smita Basak, Aimee Tai, Usan Siriwardana, Jason Hattrick-Simpers, Jay Werber  

**Link**: [PDF](https://arxiv.org/pdf/2508.10973)  

**Abstract**: The development of porous polymeric membranes remains a labor-intensive process, often requiring extensive trial and error to identify optimal fabrication parameters. In this study, we present a fully automated platform for membrane fabrication and characterization via nonsolvent-induced phase separation (NIPS). The system integrates automated solution preparation, blade casting, controlled immersion, and compression testing, allowing precise control over fabrication parameters such as polymer concentration and ambient humidity. The modular design allows parallel processing and reproducible handling of samples, reducing experimental time and increasing consistency. Compression testing is introduced as a sensitive mechanical characterization method for estimating membrane stiffness and as a proxy to infer porosity and intra-sample uniformity through automated analysis of stress-strain curves. As a proof of concept to demonstrate the effectiveness of the system, NIPS was carried out with polysulfone, the green solvent PolarClean, and water as the polymer, solvent, and nonsolvent, respectively. Experiments conducted with the automated system reproduced expected effects of polymer concentration and ambient humidity on membrane properties, namely increased stiffness and uniformity with increasing polymer concentration and humidity variations in pore morphology and mechanical response. The developed automated platform supports high-throughput experimentation and is well-suited for integration into self-driving laboratory workflows, offering a scalable and reproducible foundation for data-driven optimization of porous polymeric membranes through NIPS. 

**Abstract (ZH)**: 基于非溶剂诱导相分离的膜制备与表征自动化平台 

---
# ReachVox: Clutter-free Reachability Visualization for Robot Motion Planning in Virtual Reality 

**Title (ZH)**: ReachVox: 无干扰可达性可视化在虚拟现实中的机器人运动规划中应用 

**Authors**: Steffen Hauck, Diar Abdlkarim, John Dudley, Per Ola Kristensson, Eyal Ofek, Jens Grubert  

**Link**: [PDF](https://arxiv.org/pdf/2508.11426)  

**Abstract**: Human-Robot-Collaboration can enhance workflows by leveraging the mutual strengths of human operators and robots. Planning and understanding robot movements remain major challenges in this domain. This problem is prevalent in dynamic environments that might need constant robot motion path adaptation. In this paper, we investigate whether a minimalistic encoding of the reachability of a point near an object of interest, which we call ReachVox, can aid the collaboration between a remote operator and a robotic arm in VR. Through a user study (n=20), we indicate the strength of the visualization relative to a point-based reachability check-up. 

**Abstract (ZH)**: 人类-机器人协作可以通过利用人类操作者和机器人互 supplement的优势来增强工作流程。在这一领域，规划和理解机器人动作仍然是主要挑战。这个问题在可能需要不断适应机器人运动路径的动态环境中尤为普遍。在本文中，我们探讨了一种简易表示目标物体附近点可达性的编码方法（称为ReachVox）是否能在虚拟现实环境中辅助远程操作者与机器人手臂的合作。通过一项用户研究（n=20），我们表明可视化相对于基于点的可达性检查的优势。 

---
# Optimizing ROS 2 Communication for Wireless Robotic Systems 

**Title (ZH)**: 优化ROS无线机器人系统中的通信 

**Authors**: Sanghoon Lee, Taehun Kim, Jiyeong Chae, Kyung-Joon Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.11366)  

**Abstract**: Wireless transmission of large payloads, such as high-resolution images and LiDAR point clouds, is a major bottleneck in ROS 2, the leading open-source robotics middleware. The default Data Distribution Service (DDS) communication stack in ROS 2 exhibits significant performance degradation over lossy wireless links. Despite the widespread use of ROS 2, the underlying causes of these wireless communication challenges remain unexplored. In this paper, we present the first in-depth network-layer analysis of ROS 2's DDS stack under wireless conditions with large payloads. We identify the following three key issues: excessive IP fragmentation, inefficient retransmission timing, and congestive buffer bursts. To address these issues, we propose a lightweight and fully compatible DDS optimization framework that tunes communication parameters based on link and payload characteristics. Our solution can be seamlessly applied through the standard ROS 2 application interface via simple XML-based QoS configuration, requiring no protocol modifications, no additional components, and virtually no integration efforts. Extensive experiments across various wireless scenarios demonstrate that our framework successfully delivers large payloads in conditions where existing DDS modes fail, while maintaining low end-to-end latency. 

**Abstract (ZH)**: 无线传输大规模载荷，如高分辨率图像和LiDAR点云，是ROS 2中主要的瓶颈。ROS 2主流的开放式机器人中间件中的默认数据分布服务（DDS）通信堆栈在无线链路上表现出显著的性能退化。尽管ROS 2被广泛使用，但这些无线通信挑战的根本原因尚未被充分探讨。在本文中，我们首次在无线条件下对ROS 2的DDS堆栈进行深入的网络层分析，研究大规模载荷的情况。我们识别出以下三个关键问题：过度的IP分片、无效的重传时间设置以及拥塞缓冲区突发。为了解决这些问题，我们提出了一种轻量级且完全兼容的DDS优化框架，该框架根据不同链路和载荷特性调整通信参数。我们的解决方案可以通过标准ROS 2应用接口无缝应用，只需简单的XML基QoS配置，无需修改协议、无需额外组件，且几乎不需要集成工作。广泛的实验结果表明，我们的框架在现有DDS模式失败的情况下成功地传输了大规模载荷，同时维持了低端到端延迟。 

---
