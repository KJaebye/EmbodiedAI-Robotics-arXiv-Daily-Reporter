# DexWrist: A Robotic Wrist for Constrained and Dynamic Manipulation 

**Title (ZH)**: DexWrist: 一种用于受限和动态操作的机器人手腕 

**Authors**: Martin Peticco, Gabriella Ulloa, John Marangola, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2507.01008)  

**Abstract**: We present the DexWrist, a compliant robotic wrist designed to advance robotic manipulation in highly-constrained environments, enable dynamic tasks, and speed up data collection. DexWrist is designed to be close to the functional capabilities of the human wrist and achieves mechanical compliance and a greater workspace as compared to existing robotic wrist designs. The DexWrist can supercharge policy learning by (i) enabling faster teleoperation and therefore making data collection more scalable; (ii) completing tasks in fewer steps which reduces trajectory lengths and therefore can ease policy learning; (iii) DexWrist is designed to be torque transparent with easily simulatable kinematics for simulated data collection; and (iv) most importantly expands the workspace of manipulation for approaching highly cluttered scenes and tasks. More details about the wrist can be found at: this http URL. 

**Abstract (ZH)**: 我们提出DexWrist，一种符合人体工程学的机器人手腕，旨在促进在高度受限环境中的人机 manipulation，实现动态任务，并加速数据收集。DexWrist 设计接近人类手腕的功能能力，并在机械柔顺性和工作空间方面优于现有机器人手腕设计。DexWrist 可以通过以下方式增强策略学习：(i) 使远程操作更快，从而使数据收集更具可扩展性；(ii) 以更少的步骤完成任务，减少轨迹长度，从而简化策略学习；(iii) 设计为扭矩透明，并具有易于模拟的动力学，便于模拟数据收集；(iv) 最重要的是，DexWrist 扩展了接近高度杂乱场景和任务的 manipulation 工作空间。有关手腕的更多详细信息，请参阅：this http URL。 

---
# Parallel Transmission Aware Co-Design: Enhancing Manipulator Performance Through Actuation-Space Optimization 

**Title (ZH)**: 面向并行传输协同设计：通过执行空间优化提升 manipulator 性能 

**Authors**: Rohit Kumar, Melya Boukheddimi, Dennis Mronga, Shivesh Kumar, Frank Kirchner  

**Link**: [PDF](https://arxiv.org/pdf/2507.00644)  

**Abstract**: In robotics, structural design and behavior optimization have long been considered separate processes, resulting in the development of systems with limited capabilities. Recently, co-design methods have gained popularity, where bi-level formulations are used to simultaneously optimize the robot design and behavior for specific tasks. However, most implementations assume a serial or tree-type model of the robot, overlooking the fact that many robot platforms incorporate parallel mechanisms. In this paper, we present a novel co-design approach that explicitly incorporates parallel coupling constraints into the dynamic model of the robot. In this framework, an outer optimization loop focuses on the design parameters, in our case the transmission ratios of a parallel belt-driven manipulator, which map the desired torques from the joint space to the actuation space. An inner loop performs trajectory optimization in the actuation space, thus exploiting the entire dynamic range of the manipulator. We compare the proposed method with a conventional co-design approach based on a simplified tree-type model. By taking advantage of the actuation space representation, our approach leads to a significant increase in dynamic payload capacity compared to the conventional co-design implementation. 

**Abstract (ZH)**: 机器人结构设计与行为优化的并行耦合协同设计方法 

---
# Stable Tracking of Eye Gaze Direction During Ophthalmic Surgery 

**Title (ZH)**: 眼科学手术中稳定追踪眼球注视方向 

**Authors**: Tinghe Hong, Shenlin Cai, Boyang Li, Kai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00635)  

**Abstract**: Ophthalmic surgical robots offer superior stability and precision by reducing the natural hand tremors of human surgeons, enabling delicate operations in confined surgical spaces. Despite the advancements in developing vision- and force-based control methods for surgical robots, preoperative navigation remains heavily reliant on manual operation, limiting the consistency and increasing the uncertainty. Existing eye gaze estimation techniques in the surgery, whether traditional or deep learning-based, face challenges including dependence on additional sensors, occlusion issues in surgical environments, and the requirement for facial detection. To address these limitations, this study proposes an innovative eye localization and tracking method that combines machine learning with traditional algorithms, eliminating the requirements of landmarks and maintaining stable iris detection and gaze estimation under varying lighting and shadow conditions. Extensive real-world experiment results show that our proposed method has an average estimation error of 0.58 degrees for eye orientation estimation and 2.08-degree average control error for the robotic arm's movement based on the calculated orientation. 

**Abstract (ZH)**: 眼科手术机器人通过减少人类外科医生的自然手震，提供卓越的稳定性和精确度，使在狭小手术空间中进行精细操作成为可能。尽管在开发基于视觉和力的控制方法方面取得了进步，但预手术导航仍然高度依赖手动操作，限制了一致性并增加了不确定性。现有的手术中眼球运动估计技术，无论是传统方法还是深度学习方法，都面临额外传感器依赖、手术环境中的遮挡问题以及面部检测的需要等挑战。为了解决这些局限性，本研究提出了一种结合机器学习与传统算法的创新眼球定位和跟踪方法，该方法无需特征点，能够在不同光照和阴影条件下保持稳定的虹膜检测和眼球运动估计。广泛的实际实验结果表明，所提出的方法在眼球方向估计中的平均误差为0.58度，在基于计算方向的机器人臂运动控制中的平均误差为2.08度。 

---
# Generation of Indoor Open Street Maps for Robot Navigation from CAD Files 

**Title (ZH)**: 基于CAD文件生成室内开放街道地图以供机器人导航 

**Authors**: Jiajie Zhang, Shenrui Wu, Xu Ma, Sören Schwertfeger  

**Link**: [PDF](https://arxiv.org/pdf/2507.00552)  

**Abstract**: The deployment of autonomous mobile robots is predicated on the availability of environmental maps, yet conventional generation via SLAM (Simultaneous Localization and Mapping) suffers from significant limitations in time, labor, and robustness, particularly in dynamic, large-scale indoor environments where map obsolescence can lead to critical localization failures. To address these challenges, this paper presents a complete and automated system for converting architectural Computer-Aided Design (CAD) files into a hierarchical topometric OpenStreetMap (OSM) representation, tailored for robust life-long robot navigation. Our core methodology involves a multi-stage pipeline that first isolates key structural layers from the raw CAD data and then employs an AreaGraph-based topological segmentation to partition the building layout into a hierarchical graph of navigable spaces. This process yields a comprehensive and semantically rich map, further enhanced by automatically associating textual labels from the CAD source and cohesively merging multiple building floors into a unified, topologically-correct model. By leveraging the permanent structural information inherent in CAD files, our system circumvents the inefficiencies and fragility of SLAM, offering a practical and scalable solution for deploying robots in complex indoor spaces. The software is encapsulated within an intuitive Graphical User Interface (GUI) to facilitate practical use. The code and dataset are available at this https URL. 

**Abstract (ZH)**: 基于自主移动机器人在动态大规模室内环境中的应用，其部署依赖于可用的环境地图，而传统的通过SLAM生成地图在时间、劳动和鲁棒性方面存在显著限制。为解决这些挑战，本文提出了一种完整且自动化的系统，将建筑计算机辅助设计（CAD）文件转换为适用于鲁棒 lifelong 机器人导航的分层拓扑OpenStreetMap（OSM）表示。该核心方法涉及一个多阶段管道，首先从原始CAD数据中隔离关键的结构层，然后采用基于AreaGraph的拓扑分割将建筑布局划分为分层图中的可导航空间。此过程生成了一个全面且语义丰富的地图，并通过自动关联CAD源中的文本标签和统一合并多层建筑结构，进一步增强了这一模型。通过利用CAD文件中固有的永久结构信息，我们的系统规避了SLAM的低效性和脆弱性，提供了一种适用于复杂室内空间的实用且可扩展的机器人部署解决方案。该软件封装在直观的图形用户界面（GUI）中，以促进实际使用。相关代码和数据集可从以下链接获得。 

---
# Edge Computing and its Application in Robotics: A Survey 

**Title (ZH)**: 边缘计算及其在机器人领域的应用：一个综述 

**Authors**: Nazish Tahir, Ramviyas Parasuraman  

**Link**: [PDF](https://arxiv.org/pdf/2507.00523)  

**Abstract**: The Edge computing paradigm has gained prominence in both academic and industry circles in recent years. By implementing edge computing facilities and services in robotics, it becomes a key enabler in the deployment of artificial intelligence applications to robots. Time-sensitive robotics applications benefit from the reduced latency, mobility, and location awareness provided by the edge computing paradigm, which enables real-time data processing and intelligence at the network's edge. While the advantages of integrating edge computing into robotics are numerous, there has been no recent survey that comprehensively examines these benefits. This paper aims to bridge that gap by highlighting important work in the domain of edge robotics, examining recent advancements, and offering deeper insight into the challenges and motivations behind both current and emerging solutions. In particular, this article provides a comprehensive evaluation of recent developments in edge robotics, with an emphasis on fundamental applications, providing in-depth analysis of the key motivations, challenges, and future directions in this rapidly evolving domain. It also explores the importance of edge computing in real-world robotics scenarios where rapid response times are critical. Finally, the paper outlines various open research challenges in the field of edge robotics. 

**Abstract (ZH)**: 边缘计算范式在学术和工业领域中日益凸显，通过在机器人中实施边缘计算设施和服务，它成为将人工智能应用部署到机器人中的关键使能器。时间敏感的机器人应用得益于边缘计算范式提供的减少延迟、提高移动性和位置感知性，这使得网络边缘处能实现实时数据处理和智能。尽管将边缘计算整合到机器人中有诸多优势，但近年来尚未有任何综述全面探讨这些好处。本文旨在通过突出边缘机器人领域的关键工作、考察最近的进步并深入探讨当前及新兴解决方案背后的挑战和动机来填补这一空白。特别是，本文对边缘机器人领域的最近发展进行了全方位评估，重点关注基础应用，并对关键动机、挑战及这一迅速发展的领域中的未来方向进行了深入分析。文章还探讨了在需要快速响应时间的现实机器人场景中边缘计算的重要性。最后，论文概述了边缘机器人领域的各种开放研究挑战。 

---
# A Miniature High-Resolution Tension Sensor Based on a Photo-Reflector for Robotic Hands and Grippers 

**Title (ZH)**: 基于光电反射器的高分辨率微型张力传感器及其在机器人手和手指中的应用 

**Authors**: Hyun-Bin Kim, Kyung-Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.00464)  

**Abstract**: This paper presents a miniature tension sensor using a photo-reflector, designed for compact tendon-driven grippers and robotic hands. The proposed sensor has a small form factor of 13~mm x 7~mm x 6.5~mm and is capable of measuring tensile forces up to 200~N. A symmetric elastomer structure incorporating fillets and flexure hinges is designed based on Timoshenko beam theory and verified via FEM analysis, enabling improved sensitivity and mechanical durability while minimizing torsional deformation. The sensor utilizes a compact photo-reflector (VCNT2020) to measure displacement in the near-field region, eliminating the need for light-absorbing materials or geometric modifications required in photo-interrupter-based designs. A 16-bit analog-to-digital converter (ADC) and CAN-FD (Flexible Data-rate) communication enable efficient signal acquisition with up to 5~kHz sampling rate. Calibration experiments demonstrate a resolution of 9.9~mN (corresponding to over 14-bit accuracy) and a root mean square error (RMSE) of 0.455~N. Force control experiments using a twisted string actuator and PI control yield RMSEs as low as 0.073~N. Compared to previous research using photo-interrupter, the proposed method achieves more than tenfold improvement in resolution while also reducing nonlinearity and hysteresis. The design is mechanically simple, lightweight, easy to assemble, and suitable for integration into robotic and prosthetic systems requiring high-resolution force feedback. 

**Abstract (ZH)**: 基于光反射器的微型张力传感器及其在紧凑型肌腱驱动手指和机器人手中的应用 

---
# Novel Pigeon-inspired 3D Obstacle Detection and Avoidance Maneuver for Multi-UAV Systems 

**Title (ZH)**: 基于鸽子启发的多无人机系统3D障碍检测与规避 maneuvers 研究 

**Authors**: Reza Ahmadvand, Sarah Safura Sharif, Yaser Mike Banad  

**Link**: [PDF](https://arxiv.org/pdf/2507.00443)  

**Abstract**: Recent advances in multi-agent systems manipulation have demonstrated a rising demand for the implementation of multi-UAV systems in urban areas, which are always subjected to the presence of static and dynamic obstacles. Inspired by the collective behavior of tilapia fish and pigeons, the focus of the presented research is on the introduction of a nature-inspired collision-free formation control for a multi-UAV system, considering the obstacle avoidance maneuvers. The developed framework in this study utilizes a semi-distributed control approach, in which, based on a probabilistic Lloyd's algorithm, a centralized guidance algorithm works for optimal positioning of the UAVs, while a distributed control approach has been used for the intervehicle collision and obstacle avoidance. Further, the presented framework has been extended to the 3D space with a novel definition of 3D maneuvers. Finally, the presented framework has been applied to multi-UAV systems in 2D and 3D scenarios, and the obtained results demonstrated the validity of the presented method in dynamic environments with stationary and moving obstacles. 

**Abstract (ZH)**: 最近多智能体系统方面的进展展示了在城市区域实施多无人机系统的需求，这些区域往往存在静态和动态障碍物。借鉴鲱鱼和鸽子的集体行为，本文的研究重点在于引入一种受自然启发的无碰撞编队控制方法，同时考虑避障 maneuvers。本研究开发的框架采用半分布式控制方法，在此方法中，基于概率Lloyd算法的集中指导算法负责无人机的最优定位，而分布控制方法用于车辆间的碰撞和障碍物避免。进一步，本文框架扩展到了三维空间，并引入了三维 maneuvers 的新定义。最后，该框架应用于二维和三维场景中的多无人机系统，并获得的结果证实了在有静态和移动障碍物的动态环境中该方法的有效性。 

---
# RoboEval: Where Robotic Manipulation Meets Structured and Scalable Evaluation 

**Title (ZH)**: RoboEval：机器人Manipulation的结构化和可扩展评估 

**Authors**: Yi Ru Wang, Carter Ung, Grant Tannert, Jiafei Duan, Josephine Li, Amy Le, Rishabh Oswal, Markus Grotz, Wilbert Pumacay, Yuquan Deng, Ranjay Krishna, Dieter Fox, Siddhartha Srinivasa  

**Link**: [PDF](https://arxiv.org/pdf/2507.00435)  

**Abstract**: We present RoboEval, a simulation benchmark and structured evaluation framework designed to reveal the limitations of current bimanual manipulation policies. While prior benchmarks report only binary task success, we show that such metrics often conceal critical weaknesses in policy behavior -- such as poor coordination, slipping during grasping, or asymmetric arm usage. RoboEval introduces a suite of tiered, semantically grounded tasks decomposed into skill-specific stages, with variations that systematically challenge spatial, physical, and coordination capabilities. Tasks are paired with fine-grained diagnostic metrics and 3000+ human demonstrations to support imitation learning. Our experiments reveal that policies with similar success rates diverge in how tasks are executed -- some struggle with alignment, others with temporally consistent bimanual control. We find that behavioral metrics correlate with success in over half of task-metric pairs, and remain informative even when binary success saturates. By pinpointing when and how policies fail, RoboEval enables a deeper, more actionable understanding of robotic manipulation -- and highlights the need for evaluation tools that go beyond success alone. 

**Abstract (ZH)**: RoboEval：一种用于揭示当前双臂操作策略限制的模拟基准和结构化评估框架 

---
# Novel Design of 3D Printed Tumbling Microrobots for in vivo Targeted Drug Delivery 

**Title (ZH)**: 3D打印旋转微机器人新型设计及其体内靶向药物递送应用 

**Authors**: Aaron C. Davis, Siting Zhang, Adalyn Meeks, Diya Sakhrani, Luis Carlos Sanjuan Acosta, D. Ethan Kelley, Emma Caldwell, Luis Solorio, Craig J. Goergen, David J. Cappelleri  

**Link**: [PDF](https://arxiv.org/pdf/2507.00166)  

**Abstract**: This paper presents innovative designs for 3D-printed tumbling microrobots, specifically engineered for targeted in vivo drug delivery applications. The microrobot designs, created using stereolithography 3D printing technologies, incorporate permanent micro-magnets to enable actuation via a rotating magnetic field actuator system. The experimental framework encompasses a series of locomotion characterization tests to evaluate microrobot performance under various conditions. Testing variables include variations in microrobot geometries, actuation frequencies, and environmental conditions, such as dry and wet environments, and temperature changes. The paper outlines designs for three drug loading methods, along with comprehensive assessments thermal drug release using a focused ultrasound system, as well as biocompatibility tests. Animal model testing involves tissue phantoms and in vivo rat models, ensuring a thorough evaluation of the microrobots' performance and compatibility. The results highlight the robustness and adaptability of the proposed microrobot designs, showcasing the potential for efficient and targeted in vivo drug delivery. This novel approach addresses current limitations in existing tumbling microrobot designs and paves the way for advancements in targeted drug delivery within the large intestine. 

**Abstract (ZH)**: 本论文介绍了用于靶向体内药物传递应用的3D打印滚动微机器人创新设计。采用源自立体光刻3D打印技术的微机器人设计集成了永久微磁体，可通过旋转磁场激励系统实现操作。实验框架包括一系列运动特性测试，以评估微机器人在不同条件下的性能。测试变量包括微机器人几何形状的变化、激励频率以及环境条件，如干燥和湿润环境以及温度变化。论文概述了三种药物装载方法的设计，并进行了详细的聚焦超声系统下的热药物释放评估以及生物相容性测试。动物模型测试使用组织模拟物和活体大鼠模型，确保对微机器人的性能和兼容性进行全面评估。结果突显了所提出微机器人设计的 robustness 和适应性，展示了其在体内高效和靶向药物传递方面的潜力。该新颖方法解决了现有滚动微机器人设计的局限性，为进一步推动大肠靶向药物传递技术的发展铺平了道路。 

---
