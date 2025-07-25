# ReactEMG: Zero-Shot, Low-Latency Intent Detection via sEMG 

**Title (ZH)**: ReactEMG: 零样本、低延迟意图检测基于sEMG 

**Authors**: Runsheng Wang, Xinyue Zhu, Ava Chen, Jingxi Xu, Lauren Winterbottom, Dawn M. Nilsen, Joel Stein, Matei Ciocarlie  

**Link**: [PDF](https://arxiv.org/pdf/2506.19815)  

**Abstract**: Surface electromyography (sEMG) signals show promise for effective human-computer interfaces, particularly in rehabilitation and prosthetics. However, challenges remain in developing systems that respond quickly and reliably to user intent, across different subjects and without requiring time-consuming calibration. In this work, we propose a framework for EMG-based intent detection that addresses these challenges. Unlike traditional gesture recognition models that wait until a gesture is completed before classifying it, our approach uses a segmentation strategy to assign intent labels at every timestep as the gesture unfolds. We introduce a novel masked modeling strategy that aligns muscle activations with their corresponding user intents, enabling rapid onset detection and stable tracking of ongoing gestures. In evaluations against baseline methods, considering both accuracy and stability for device control, our approach surpasses state-of-the-art performance in zero-shot transfer conditions, demonstrating its potential for wearable robotics and next-generation prosthetic systems. Our project page is available at: this https URL 

**Abstract (ZH)**: 表面肌电图(sEMG)信号在人机接口中显示出有效的前景，尤其是在康复和假肢领域。然而，在开发能快速可靠地响应用户意图的系统方面仍面临挑战，特别是在不同受试者之间无需长时间校准的情况下。在这项工作中，我们提出了一种基于肌电图的意图检测框架，以应对这些挑战。与传统的手势识别模型在手势完成后再进行分类不同，我们的方法使用分割策略，在手势展开的每个时间步骤都分配意图标签。我们引入了一种新颖的掩码建模策略，将肌肉激活与相应的用户意图对齐，从而实现快速起始检测并稳定追踪正在进行的手势。在基线方法的评估中，考虑设备控制的准确性和稳定性，我们的方法在零样本迁移条件下超越了现有最佳性能，展示了其在可穿戴机器人和下一代假肢系统中的潜力。更多详情请参见我们的项目页面：this https URL 

---
# Estimating Spatially-Dependent GPS Errors Using a Swarm of Robots 

**Title (ZH)**: 使用机器人群估计空间相关的GPS误差 

**Authors**: Praneeth Somisetty, Robert Griffin, Victor M. Baez, Miguel F. Arevalo-Castiblanco, Aaron T. Becker, Jason M. O'Kane  

**Link**: [PDF](https://arxiv.org/pdf/2506.19712)  

**Abstract**: External factors, including urban canyons and adversarial interference, can lead to Global Positioning System (GPS) inaccuracies that vary as a function of the position in the environment. This study addresses the challenge of estimating a static, spatially-varying error function using a team of robots. We introduce a State Bias Estimation Algorithm (SBE) whose purpose is to estimate the GPS biases. The central idea is to use sensed estimates of the range and bearing to the other robots in the team to estimate changes in bias across the environment. A set of drones moves in a 2D environment, each sampling data from GPS, range, and bearing sensors. The biases calculated by the SBE at estimated positions are used to train a Gaussian Process Regression (GPR) model. We use a Sparse Gaussian process-based Informative Path Planning (IPP) algorithm that identifies high-value regions of the environment for data collection. The swarm plans paths that maximize information gain in each iteration, further refining their understanding of the environment's positional bias landscape. We evaluated SBE and IPP in simulation and compared the IPP methodology to an open-loop strategy. 

**Abstract (ZH)**: 外部因素，包括城市峡谷效应和对抗性干扰，会导致全球定位系统（GPS）的不准确性，这种不准确性随着环境位置的变化而变化。本研究旨在利用机器人团队解决估计静态、空间变化误差函数的挑战。我们介绍了一种状态偏差估计算法（SBE），其目的是估计GPS偏差。核心思想是利用团队中其他机器人到自身的距离和方位角感应估计值，来估计环境中偏差的变化。一组无人机在二维环境中移动，每个无人机从GPS、距离和方位传感器采集数据。由SBE在估计位置计算的偏差用于训练高斯过程回归（GPR）模型。我们使用基于稀疏高斯过程的信息路径规划（IPP）算法来识别环境中的高价值区域以采集数据。根据每次迭代中获取的最大信息增益规划路径，进一步精确定义环境位置偏差景观。我们通过仿真评估了SBE和IPP，并将IPP方法与开环策略进行了比较。 

---
# UniTac-NV: A Unified Tactile Representation For Non-Vision-Based Tactile Sensors 

**Title (ZH)**: UniTac-NV: 一种基于非视觉触觉传感器的统一触觉表示 

**Authors**: Jian Hou, Xin Zhou, Qihan Yang, Adam J. Spiers  

**Link**: [PDF](https://arxiv.org/pdf/2506.19699)  

**Abstract**: Generalizable algorithms for tactile sensing remain underexplored, primarily due to the diversity of sensor modalities. Recently, many methods for cross-sensor transfer between optical (vision-based) tactile sensors have been investigated, yet little work focus on non-optical tactile sensors. To address this gap, we propose an encoder-decoder architecture to unify tactile data across non-vision-based sensors. By leveraging sensor-specific encoders, the framework creates a latent space that is sensor-agnostic, enabling cross-sensor data transfer with low errors and direct use in downstream applications. We leverage this network to unify tactile data from two commercial tactile sensors: the Xela uSkin uSPa 46 and the Contactile PapillArray. Both were mounted on a UR5e robotic arm, performing force-controlled pressing sequences against distinct object shapes (circular, square, and hexagonal prisms) and two materials (rigid PLA and flexible TPU). Another more complex unseen object was also included to investigate the model's generalization capabilities. We show that alignment in latent space can be implicitly learned from joint autoencoder training with matching contacts collected via different sensors. We further demonstrate the practical utility of our approach through contact geometry estimation, where downstream models trained on one sensor's latent representation can be directly applied to another without retraining. 

**Abstract (ZH)**: 可泛化的触觉感知算法研究：基于非视觉传感器的统一表示与跨传感器转移 

---
# A Verification Methodology for Safety Assurance of Robotic Autonomous Systems 

**Title (ZH)**: 一种用于保障机器人自主系统安全性的验证方法学 

**Authors**: Mustafa Adam, David A. Anisi, Pedro Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2506.19622)  

**Abstract**: Autonomous robots deployed in shared human environments, such as agricultural settings, require rigorous safety assurance to meet both functional reliability and regulatory compliance. These systems must operate in dynamic, unstructured environments, interact safely with humans, and respond effectively to a wide range of potential hazards. This paper presents a verification workflow for the safety assurance of an autonomous agricultural robot, covering the entire development life-cycle, from concept study and design to runtime verification. The outlined methodology begins with a systematic hazard analysis and risk assessment to identify potential risks and derive corresponding safety requirements. A formal model of the safety controller is then developed to capture its behaviour and verify that the controller satisfies the specified safety properties with respect to these requirements. The proposed approach is demonstrated on a field robot operating in an agricultural setting. The results show that the methodology can be effectively used to verify safety-critical properties and facilitate the early identification of design issues, contributing to the development of safer robots and autonomous systems. 

**Abstract (ZH)**: 自主机器人部署在共享人类环境中的农业设置等场景下，需要严格的安全性保证以满足功能性可靠性和监管合规性要求。这些系统必须在动态、非结构化的环境中操作，安全地与人类互动，并有效地应对各种潜在的危害。本文提出了一种自主农业机器人安全保证的验证工作流，涵盖从概念研究和设计到运行时验证的整个开发生命周期。该概述的方法从系统化的危害分析和风险评估开始，以识别潜在风险并推导出相应的安全要求。然后开发形式化的安全控制器模型以捕捉其行为，并验证控制器是否满足这些要求所规定的安全属性。所提出的的方法在一种田间操作的农业机器人中进行了演示。结果表明，该方法可以有效地用于验证安全关键属性，并有助于早期识别设计问题，从而促进更安全的机器人和自主系统的开发。 

---
# Probabilistic modelling and safety assurance of an agriculture robot providing light-treatment 

**Title (ZH)**: 农业机器人提供光照处理的概率建模与安全保证 

**Authors**: Mustafa Adam, Kangfeng Ye, David A. Anisi, Ana Cavalcanti, Jim Woodcock, Robert Morris  

**Link**: [PDF](https://arxiv.org/pdf/2506.19620)  

**Abstract**: Continued adoption of agricultural robots postulates the farmer's trust in the reliability, robustness and safety of the new technology. This motivates our work on safety assurance of agricultural robots, particularly their ability to detect, track and avoid obstacles and humans. This paper considers a probabilistic modelling and risk analysis framework for use in the early development phases. Starting off with hazard identification and a risk assessment matrix, the behaviour of the mobile robot platform, sensor and perception system, and any humans present are captured using three state machines. An auto-generated probabilistic model is then solved and analysed using the probabilistic model checker PRISM. The result provides unique insight into fundamental development and engineering aspects by quantifying the effect of the risk mitigation actions and risk reduction associated with distinct design concepts. These include implications of adopting a higher performance and more expensive Object Detection System or opting for a more elaborate warning system to increase human awareness. Although this paper mainly focuses on the initial concept-development phase, the proposed safety assurance framework can also be used during implementation, and subsequent deployment and operation phases. 

**Abstract (ZH)**: 农业机器人持续采用预示农民对其可靠性和安全性信托，这促使我们关注农业机器人安全性保障，特别是其检测、跟踪和避开障碍物及人类的能力。本文提出一种概率建模和风险管理框架，适用于早期开发阶段。通过危害识别和风险评估矩阵，利用三个状态机捕捉移动机器人平台、传感器和感知系统，以及任何现场人类的行为。然后自动生成的概率模型通过PRISM概率模型检查器求解和分析。结果通过量化风险缓解措施和不同设计概念相关的风险降低，为基本开发和工程方面提供独特的见解。这些包括采用性能更高、成本更高的目标检测系统或选择更复杂的警告系统以提升人类意识的涵义。尽管本文主要集中在初始概念开发阶段，但提出的安全保障框架也可用于实施、后续部署和运行阶段。 

---
# Soft Robotic Delivery of Coiled Anchors for Cardiac Interventions 

**Title (ZH)**: 心脏介入手术用螺旋锚的软体机器人递送技术 

**Authors**: Leonardo Zamora Yanez, Jacob Rogatinsky, Dominic Recco, Sang-Yoep Lee, Grace Matthews, Andrew P. Sabelhaus, Tommaso Ranzani  

**Link**: [PDF](https://arxiv.org/pdf/2506.19602)  

**Abstract**: Trans-catheter cardiac intervention has become an increasingly available option for high-risk patients without the complications of open heart surgery. However, current catheterbased platforms suffer from a lack of dexterity, force application, and compliance required to perform complex intracardiac procedures. An exemplary task that would significantly ease minimally invasive intracardiac procedures is the implantation of anchor coils, which can be used to fix and implant various devices in the beating heart. We introduce a robotic platform capable of delivering anchor coils. We develop a kineto-statics model of the robotic platform and demonstrate low positional error. We leverage the passive compliance and high force output of the actuator in a multi-anchor delivery procedure against a motile in-vitro simulator with millimeter level accuracy. 

**Abstract (ZH)**: 经导管心脏干预已成为一种越来越可供高风险患者选择的选项，且避免了开胸手术的并发症。然而，当前的基于导管的平台在执行复杂的心内操作时缺乏所需的柔顺性、力应用和顺应性。一个能够显著简化微创心内操作的范例任务是植入锚线圈，这些锚线圈可用于在跳动的心脏中定位和植入各种设备。我们介绍了一个能够输送锚线圈的机器人平台。我们建立了该机器人平台的运动静力学模型，并展示了低位置误差。我们利用执行器的被动顺应性和高力输出，针对一个移动的体外模拟器进行了多锚输送操作，实现了毫米级的精确度。 

---
# Robotics Under Construction: Challenges on Job Sites 

**Title (ZH)**: 机器人技术在建设现场：面临的挑战 

**Authors**: Haruki Uchiito, Akhilesh Bhat, Koji Kusaka, Xiaoya Zhang, Hiraku Kinjo, Honoka Uehara, Motoki Koyama, Shinji Natsume  

**Link**: [PDF](https://arxiv.org/pdf/2506.19597)  

**Abstract**: As labor shortages and productivity stagnation increasingly challenge the construction industry, automation has become essential for sustainable infrastructure development. This paper presents an autonomous payload transportation system as an initial step toward fully unmanned construction sites. Our system, based on the CD110R-3 crawler carrier, integrates autonomous navigation, fleet management, and GNSS-based localization to facilitate material transport in construction site environments. While the current system does not yet incorporate dynamic environment adaptation algorithms, we have begun fundamental investigations into external-sensor based perception and mapping system. Preliminary results highlight the potential challenges, including navigation in evolving terrain, environmental perception under construction-specific conditions, and sensor placement optimization for improving autonomy and efficiency. Looking forward, we envision a construction ecosystem where collaborative autonomous agents dynamically adapt to site conditions, optimizing workflow and reducing human intervention. This paper provides foundational insights into the future of robotics-driven construction automation and identifies critical areas for further technological development. 

**Abstract (ZH)**: 随着劳动力短缺和生产率停滞日益挑战建筑行业，自动化已成为可持续基础设施发展的关键。本文提出了一个自主载荷运输系统，作为迈向完全无人施工场地的第一步。我们的系统基于CD110R-3履带载体，集成了自主导航、车队管理以及基于GNSS的定位技术，以促进施工场地环境下的物料运输。虽然当前系统尚未集成动态环境适应算法，但已经开始对外部传感器为基础的感知和定位系统进行基本研究。初步结果显示，主要挑战包括在动态地形中的导航、在施工特定条件下进行环境感知，以及优化传感器布局以提高自主性和效率。展望未来，我们设想一个协作自主代理动态适应场地条件的施工生态系统，优化工作流程并减少人为干预。本文为机器人驱动的建筑自动化未来提供了基础见解，并指出需要进一步技术发展的关键领域。 

---
# Ground-Effect-Aware Modeling and Control for Multicopters 

**Title (ZH)**: 地面效应对多旋翼飞行器的建模与控制 

**Authors**: Tiankai Yang, Kaixin Chai, Jialin Ji, Yuze Wu, Chao Xu, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.19424)  

**Abstract**: The ground effect on multicopters introduces several challenges, such as control errors caused by additional lift, oscillations that may occur during near-ground flight due to external torques, and the influence of ground airflow on models such as the rotor drag and the mixing matrix. This article collects and analyzes the dynamics data of near-ground multicopter flight through various methods, including force measurement platforms and real-world flights. For the first time, we summarize the mathematical model of the external torque of multicopters under ground effect. The influence of ground airflow on rotor drag and the mixing matrix is also verified through adequate experimentation and analysis. Through simplification and derivation, the differential flatness of the multicopter's dynamic model under ground effect is confirmed. To mitigate the influence of these disturbance models on control, we propose a control method that combines dynamic inverse and disturbance models, ensuring consistent control effectiveness at both high and low altitudes. In this method, the additional thrust and variations in rotor drag under ground effect are both considered and compensated through feedforward models. The leveling torque of ground effect can be equivalently represented as variations in the center of gravity and the moment of inertia. In this way, the leveling torque does not explicitly appear in the dynamic model. The final experimental results show that the method proposed in this paper reduces the control error (RMSE) by \textbf{45.3\%}. Please check the supplementary material at: this https URL. 

**Abstract (ZH)**: 地面效应对多旋翼飞行器的影响及其控制方法研究：通过地面气流对旋翼阻力和混合矩阵的影响的分析与验证 

---
# A Survey on Soft Robot Adaptability: Implementations, Applications, and Prospects 

**Title (ZH)**: 软体机器人适应性综述：实现、应用与前景 

**Authors**: Zixi Chen, Di Wu, Qinghua Guan, David Hardman, Federico Renda, Josie Hughes, Thomas George Thuruthel, Cosimo Della Santina, Barbara Mazzolai, Huichan Zhao, Cesare Stefanini  

**Link**: [PDF](https://arxiv.org/pdf/2506.19397)  

**Abstract**: Soft robots, compared to rigid robots, possess inherent advantages, including higher degrees of freedom, compliance, and enhanced safety, which have contributed to their increasing application across various fields. Among these benefits, adaptability is particularly noteworthy. In this paper, adaptability in soft robots is categorized into external and internal adaptability. External adaptability refers to the robot's ability to adjust, either passively or actively, to variations in environments, object properties, geometries, and task dynamics. Internal adaptability refers to the robot's ability to cope with internal variations, such as manufacturing tolerances or material aging, and to generalize control strategies across different robots. As the field of soft robotics continues to evolve, the significance of adaptability has become increasingly pronounced. In this review, we summarize various approaches to enhancing the adaptability of soft robots, including design, sensing, and control strategies. Additionally, we assess the impact of adaptability on applications such as surgery, wearable devices, locomotion, and manipulation. We also discuss the limitations of soft robotics adaptability and prospective directions for future research. By analyzing adaptability through the lenses of implementation, application, and challenges, this paper aims to provide a comprehensive understanding of this essential characteristic in soft robotics and its implications for diverse applications. 

**Abstract (ZH)**: 软体机器人相较于刚性机器人具有更高的自由度、顺应性和增强的安全性等固有优势，这些优势使其在多个领域中的应用日益增多。在这其中，适应性尤为重要。本文将软体机器人的适应性分为外部适应性和内部适应性。外部适应性指的是机器人能够被动或主动地适应环境变化、对象性质、几何形态和任务动态的变化。内部适应性指的是机器人能够应对制造公差或材料老化等内部变化，并在不同机器人上泛化控制策略的能力。随着软体机器人领域的不断发展，适应性的重要性日益凸显。在本文的综述中，我们总结了提高软体机器人适应性的各种方法，包括设计、传感和控制策略。此外，我们还评估了适应性对诸如手术、可穿戴设备、运动和操纵等领域应用的影响。我们还讨论了软体机器人适应性的局限性及未来研究的潜在方向。通过从实现、应用和挑战三个视角分析适应性，本文旨在提供对这一软体机器人关键特性及其对多种应用的潜在影响的全面理解。 

---
# The MOTIF Hand: A Robotic Hand for Multimodal Observations with Thermal, Inertial, and Force Sensors 

**Title (ZH)**: MOTIF 手部：一种配备热感、惯性及力敏传感器的多功能观察机器人手 

**Authors**: Hanyang Zhou, Haozhe Lou, Wenhao Liu, Enyu Zhao, Yue Wang, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2506.19201)  

**Abstract**: Advancing dexterous manipulation with multi-fingered robotic hands requires rich sensory capabilities, while existing designs lack onboard thermal and torque sensing. In this work, we propose the MOTIF hand, a novel multimodal and versatile robotic hand that extends the LEAP hand by integrating: (i) dense tactile information across the fingers, (ii) a depth sensor, (iii) a thermal camera, (iv), IMU sensors, and (v) a visual sensor. The MOTIF hand is designed to be relatively low-cost (under 4000 USD) and easily reproducible. We validate our hand design through experiments that leverage its multimodal sensing for two representative tasks. First, we integrate thermal sensing into 3D reconstruction to guide temperature-aware, safe grasping. Second, we show how our hand can distinguish objects with identical appearance but different masses - a capability beyond methods that use vision only. 

**Abstract (ZH)**: 增强多指灵巧 manipulative 能力需要丰富的感知能力，而现有设计缺乏在手内进行热和扭矩感知的能力。本文提出 MOTIF 手，一种通过整合（i）手指上密集的触觉信息，（ii）深度传感器，（iii）热成像相机，（iv）惯性测量单元传感器，以及（v）视觉传感器，扩展 LEAP 手的新颖的多模态和多功能机械手。MOTIF 手旨在较低成本（低于4000美元）且易于复现。我们通过利用其多模态感知进行的实验验证了我们的手设计，完成两项代表性任务。首先，我们将热感知整合到3D重建中，以指导温度感知的安全抓取。其次，我们展示了我们的手如何区分外观相同但质量不同的物体——这超出了仅使用视觉方法的能力。 

---
# Analysis and experiments of the dissipative Twistcar: direction reversal and asymptotic approximations 

**Title (ZH)**: 耗散Twistcar的方向反转及渐近逼近分析与实验 

**Authors**: Rom Levy, Ari Dantus, Zitao Yu, Yizhar Or  

**Link**: [PDF](https://arxiv.org/pdf/2506.19112)  

**Abstract**: Underactuated wheeled vehicles are commonly studied as nonholonomic systems with periodic actuation. Twistcar is a classical example inspired by a riding toy, which has been analyzed using a planar model of a dynamical system with nonholonomic constraints. Most of the previous analyses did not account for energy dissipation due to friction. In this work, we study a theoretical two-link model of the Twistcar while incorporating dissipation due to rolling resistance. We obtain asymptotic expressions for the system's small-amplitude steady-state periodic dynamics, which reveals the possibility of reversing the direction of motion upon varying the geometric and mass properties of the vehicle. Next, we design and construct a robotic prototype of the Twistcar whose center-of-mass position can be shifted by adding and removing a massive block, enabling demonstration of the Twistcar's direction reversal phenomenon. We also conduct parameter fitting for the frictional resistance in order to improve agreement with experiments. 

**Abstract (ZH)**: 欠驱动轮式车辆通常被研究为具有周期性激励的非完整系统。Twistcar 是一个源于一种儿童玩具的经典例子，它已经被用平面动力学系统的非完整约束模型进行分析。之前的大多数分析都没有考虑到摩擦引起的能量损耗。在本文中，我们考虑滚动阻力引起的耗散，研究了一个Twistcar的两环节理论模型，并获得了系统的小振幅稳态周期动力学的渐近表达式，揭示了通过改变车辆的几何和质量属性可以逆转运动方向的可能性。接下来，我们设计并构建了一个能够通过添加和移除重块改变质心位置的Twistcar机器人原型，以便演示Twistcar的运动方向反转现象。我们还进行了摩擦阻力的参数拟合，以提高与实验的吻合度。 

---
# FORTE: Tactile Force and Slip Sensing on Compliant Fingers for Delicate Manipulation 

**Title (ZH)**: FORTE：具有良好触觉力感知和打滑检测的柔顺手指精细操作技术 

**Authors**: Siqi Shang, Mingyo Seo, Yuke Zhu, Lilly Chin  

**Link**: [PDF](https://arxiv.org/pdf/2506.18960)  

**Abstract**: Handling delicate and fragile objects remains a major challenge for robotic manipulation, especially for rigid parallel grippers. While the simplicity and versatility of parallel grippers have led to widespread adoption, these grippers are limited by their heavy reliance on visual feedback. Tactile sensing and soft robotics can add responsiveness and compliance. However, existing methods typically involve high integration complexity or suffer from slow response times. In this work, we introduce FORTE, a tactile sensing system embedded in compliant gripper fingers. FORTE uses 3D-printed fin-ray grippers with internal air channels to provide low-latency force and slip feedback. FORTE applies just enough force to grasp objects without damaging them, while remaining easy to fabricate and integrate. We find that FORTE can accurately estimate grasping forces from 0-8 N with an average error of 0.2 N, and detect slip events within 100 ms of occurring. We demonstrate FORTE's ability to grasp a wide range of slippery, fragile, and deformable objects. In particular, FORTE grasps fragile objects like raspberries and potato chips with a 98.6% success rate, and achieves 93% accuracy in detecting slip events. These results highlight FORTE's potential as a robust and practical solution for enabling delicate robotic manipulation. Project page: this https URL 

**Abstract (ZH)**: Handling Delicate and Fragile Objects Remains a Major Challenge for Robotic Manipulation, Especially for Rigid Parallel Grippers. Tactile Sensing and Soft Robotics Can Add Responsiveness and Compliance, but Existing Methods Usually Involve High Integration Complexity or Suffer from Slow Response Times. In This Work, We Introduce FORTE, a Tactile Sensing System Embedded in Compliant Gripper Fingers. 

---
# Experimental Assessment of Neural 3D Reconstruction for Small UAV-based Applications 

**Title (ZH)**: 基于小型无人机的神经网络3D重构实验评估 

**Authors**: Genís Castillo Gómez-Raya, Álmos Veres-Vitályos, Filip Lemic, Pablo Royo, Mario Montagud, Sergi Fernández, Sergi Abadal, Xavier Costa-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2506.19491)  

**Abstract**: The increasing miniaturization of Unmanned Aerial Vehicles (UAVs) has expanded their deployment potential to indoor and hard-to-reach areas. However, this trend introduces distinct challenges, particularly in terms of flight dynamics and power consumption, which limit the UAVs' autonomy and mission capabilities. This paper presents a novel approach to overcoming these limitations by integrating Neural 3D Reconstruction (N3DR) with small UAV systems for fine-grained 3-Dimensional (3D) digital reconstruction of small static objects. Specifically, we design, implement, and evaluate an N3DR-based pipeline that leverages advanced models, i.e., Instant-ngp, Nerfacto, and Splatfacto, to improve the quality of 3D reconstructions using images of the object captured by a fleet of small UAVs. We assess the performance of the considered models using various imagery and pointcloud metrics, comparing them against the baseline Structure from Motion (SfM) algorithm. The experimental results demonstrate that the N3DR-enhanced pipeline significantly improves reconstruction quality, making it feasible for small UAVs to support high-precision 3D mapping and anomaly detection in constrained environments. In more general terms, our results highlight the potential of N3DR in advancing the capabilities of miniaturized UAV systems. 

**Abstract (ZH)**: 小型无人机系统结合神经三维重建的小尺度静态物体精细三维数字重建方法与评估 

---
