# Palpation Alters Auditory Pain Expressions with Gender-Specific Variations in Robopatients 

**Title (ZH)**: palpation 调整了 Robopatients 中具有性别特异性变化的听觉疼痛表达。 

**Authors**: Chapa Sirithunge, Yue Xie, Saitarun Nadipineni, Fumiya Iida, Thilina Dulantha Lalitharatne  

**Link**: [PDF](https://arxiv.org/pdf/2506.11906)  

**Abstract**: Diagnostic errors remain a major cause of preventable deaths, particularly in resource-limited regions. Medical training simulators, including robopatients, play a vital role in reducing these errors by mimicking real patients for procedural training such as palpation. However, generating multimodal feedback, especially auditory pain expressions, remains challenging due to the complex relationship between palpation behavior and sound. The high-dimensional nature of pain sounds makes exploration challenging with conventional methods. This study introduces a novel experimental paradigm for pain expressivity in robopatients where they dynamically generate auditory pain expressions in response to palpation force, by co-optimizing human feedback using machine learning. Using Proximal Policy Optimization (PPO), a reinforcement learning (RL) technique optimized for continuous adaptation, our robot iteratively refines pain sounds based on real-time human feedback. This robot initializes randomized pain responses to palpation forces, and the RL agent learns to adjust these sounds to align with human preferences. The results demonstrated that the system adapts to an individual's palpation forces and sound preferences and captures a broad spectrum of pain intensity, from mild discomfort to acute distress, through RL-guided exploration of the auditory pain space. The study further showed that pain sound perception exhibits saturation at lower forces with gender specific thresholds. These findings highlight the system's potential to enhance abdominal palpation training by offering a controllable and immersive simulation platform. 

**Abstract (ZH)**: 诊断错误仍然是导致可预防死亡的主要原因，特别是在资源有限的地区。医学培训模拟器，包括仿真人患者，在通过模拟真实患者进行程序训练（如触诊）以减少这些错误方面发挥着重要作用。然而，生成多模态反馈，尤其是听觉疼痛表达，由于触诊行为与声音之间的复杂关系，依然具有挑战性。疼痛声音的高维性质使得使用传统方法进行探索变得困难。本研究介绍了一种新的实验范式，用于仿真人患者的疼痛表达性，在此范式下，仿真人患者根据触诊力动态生成听觉疼痛表达，并通过机器学习优化人类反馈进行协同优化。使用近端策略优化（PPO），一种针对连续适应优化的强化学习（RL）技术，我们的机器人根据实时人类反馈逐步完善疼痛声音。该机器人初始化随机化的疼痛响应，而RL代理学习调整这些声音以与人类偏好对齐。研究结果表明，该系统能够适应个体的触诊力和声音偏好，并通过RL引导的听觉疼痛空间探索，捕捉从轻微不适到急性痛苦的广泛疼痛强度谱。本研究还表明，疼痛声音感知在较低的力下显示出饱和现象，存在性别特定的阈值。这些发现突显了该系统通过提供可控且沉浸式的模拟平台，增强腹部触诊训练的潜力。 

---
# Auditory-Tactile Congruence for Synthesis of Adaptive Pain Expressions in RoboPatients 

**Title (ZH)**: 听触觉一致性在合成适应回应疼痛表达中的应用研究 

**Authors**: Saitarun Nadipineni, Chapa Sirithunge, Yue Xie, Fumiya Iida, Thilina Dulantha Lalitharatne  

**Link**: [PDF](https://arxiv.org/pdf/2506.11827)  

**Abstract**: Misdiagnosis can lead to delayed treatments and harm. Robotic patients offer a controlled way to train and evaluate clinicians in rare, subtle, or complex cases, reducing diagnostic errors. We present RoboPatient, a medical robotic simulator aimed at multimodal pain synthesis based on haptic and auditory feedback during palpation-based training scenarios. The robopatient functions as an adaptive intermediary, capable of synthesizing plausible pain expressions vocal and facial in response to tactile stimuli generated during palpation. Using an abdominal phantom, robopatient captures and processes haptic input via an internal palpation-to-pain mapping model. To evaluate perceptual congruence between palpation and the corresponding auditory output, we conducted a study involving 7680 trials across 20 participants, where they evaluated pain intensity through sound. Results show that amplitude and pitch significantly influence agreement with the robot's pain expressions, irrespective of pain sounds. Stronger palpation forces elicited stronger agreement, aligning with psychophysical patterns. The study revealed two key dimensions: pitch and amplitude are central to how people perceive pain sounds, with pitch being the most influential cue. These acoustic features shape how well the sound matches the applied force during palpation, impacting perceived realism. This approach lays the groundwork for high-fidelity robotic patients in clinical education and diagnostic simulation. 

**Abstract (ZH)**: 基于触觉和听觉反馈的多模态疼痛合成医学机器人模拟器：RoboPatient 

---
# Robot Context Protocol (RCP): A Runtime-Agnostic Interface for Agent-Aware Robot Control 

**Title (ZH)**: 基于代理感知的机器人控制的运行时无关接口：Robot Context Protocol (RCP) 

**Authors**: Lambert Lee, Joshua Lau  

**Link**: [PDF](https://arxiv.org/pdf/2506.11650)  

**Abstract**: The Robot Context Protocol (RCP) is a lightweight, middleware-agnostic communication protocol designed to simplify the complexity of robotic systems and enable seamless interaction between robots, users, and autonomous agents. RCP provides a unified and semantically meaningful interface that decouples client-facing operations from backend implementations, supporting a wide range of deployment environments including physical robots, cloud-based orchestrators, and simulated platforms. Built on HTTP and WebSocket transport layers, the protocol defines a schema-driven message format with structured operations such as read, write, execute, and subscribe. It integrates features such as runtime introspection, asynchronous feedback, multi-tenant namespace isolation, and strict type validation to ensure robustness, scalability, and security. The architecture, message structure, interface model, and adapter-based backend integration strategy of RCP are described, along with deployment practices and applicability across industries including manufacturing, logistics, and healthcare. RCP enables intelligent, resilient, and safe robotic operations in complex, multi-agent ecosystems. 

**Abstract (ZH)**: 基于上下文的机器人协议（RCP）是一种轻量级、中间件无关的通信协议，旨在简化机器人系统的复杂性，并促进机器人、用户和自主代理之间的无缝交互。RCP 提供了一个统一且语义上有意义的接口，将面向客户端的操作与后端实现解耦，支持包括物理机器人、基于云的编排器和模拟平台在内的广泛部署环境。该协议基于 HTTP 和 WebSocket 传输层，定义了一种以结构化操作（如读取、写入、执行和订阅）为基础的模式驱动消息格式。RCP 集成运行时反思、异步反馈、多租户命名空间隔离和严格的类型验证等功能，以确保其稳健性、可扩展性和安全性。文章描述了 RCP 的架构、消息结构、接口模型以及基于适配器的后端集成策略，并探讨了其在制造业、物流业和医疗保健等行业的应用实践，RCP 使智能、稳健和安全的机器人操作在复杂的多代理生态系统中成为可能。 

---
# Construction of a Multiple-DOF Under-actuated Gripper with Force-Sensing via Deep Learning 

**Title (ZH)**: 基于深度学习的多自由度欠驱动力感掌握豹构建 

**Authors**: Jihao Li, Keqi Zhu, Guodong Lu, I-Ming Chen, Huixu Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.11570)  

**Abstract**: We present a novel under-actuated gripper with two 3-joint fingers, which realizes force feedback control by the deep learning technique- Long Short-Term Memory (LSTM) model, without any force sensor. First, a five-linkage mechanism stacked by double four-linkages is designed as a finger to automatically achieve the transformation between parallel and enveloping grasping modes. This enables the creation of a low-cost under-actuated gripper comprising a single actuator and two 3-phalange fingers. Second, we devise theoretical models of kinematics and power transmission based on the proposed gripper, accurately obtaining fingertip positions and contact forces. Through coupling and decoupling of five-linkage mechanisms, the proposed gripper offers the expected capabilities of grasping payload/force/stability and objects with large dimension ranges. Third, to realize the force control, an LSTM model is proposed to determine the grasping mode for synthesizing force-feedback control policies that exploit contact sensing after outlining the uncertainty of currents using a statistical method. Finally, a series of experiments are implemented to measure quantitative indicators, such as the payload, grasping force, force sensing, grasping stability and the dimension ranges of objects to be grasped. Additionally, the grasping performance of the proposed gripper is verified experimentally to guarantee the high versatility and robustness of the proposed gripper. 

**Abstract (ZH)**: 一种基于长短期记忆模型的两指三关节未驱动夹爪及其力反馈控制方法 

---
# Control Architecture and Design for a Multi-robotic Visual Servoing System in Automated Manufacturing Environment 

**Title (ZH)**: 多机器人视觉伺服系统在自动化制造环境中的控制架构与设计 

**Authors**: Rongfei Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.11387)  

**Abstract**: The use of robotic technology has drastically increased in manufacturing in the 21st century. But by utilizing their sensory cues, humans still outperform machines, especially in micro scale manufacturing, which requires high-precision robot manipulators. These sensory cues naturally compensate for high levels of uncertainties that exist in the manufacturing environment. Uncertainties in performing manufacturing tasks may come from measurement noise, model inaccuracy, joint compliance (e.g., elasticity), etc. Although advanced metrology sensors and high precision microprocessors, which are utilized in modern robots, have compensated for many structural and dynamic errors in robot positioning, a well-designed control algorithm still works as a comparable and cheaper alternative to reduce uncertainties in automated manufacturing. Our work illustrates that a multi-robot control system that simulates the positioning process for fastening and unfastening applications can reduce various uncertainties, which may occur in this process, to a great extent. In addition, most research papers in visual servoing mainly focus on developing control and observation architectures in various scenarios, but few have discussed the importance of the camera's location in the configuration. In a manufacturing environment, the quality of camera estimations may vary significantly from one observation location to another, as the combined effects of environmental conditions result in different noise levels of a single image shot at different locations. Therefore, in this paper, we also propose a novel algorithm for the camera's moving policy so that it explores the camera workspace and searches for the optimal location where the image noise level is minimized. 

**Abstract (ZH)**: 机器人技术在21世纪制造业中的应用大幅增加。但通过利用其传感提示，人类在微观规模制造中仍然表现优于机器，特别是在需要高精度机器人操作的制造任务中。这些传感提示自然能够补偿制造环境中存在的高不确定因素。执行制造任务时的不确定性可能来源于测量噪声、模型不准确、关节顺应性（例如弹性）等。尽管现代机器人利用先进的计量传感器和高精度微处理器已经补偿了许多结构和动态定位误差，但精心设计的控制算法仍然作为一种性价比更高的替代方案，用于减少自动化制造过程中的不确定性。我们的研究展示了一种模拟固定和拆卸应用的多机器人控制系统，可以大大减少这一过程中可能发生的各种不确定性。此外，大多数视觉伺服领域的研究主要集中在不同场景下的控制和观察架构开发，但很少讨论相机位置在配置中的重要性。在制造环境中，不同观测位置处的相机估计质量可能会有很大差异，因为环境条件的综合影响导致不同位置拍摄的单张图像的噪声水平不同。因此，在本文中，我们还提出了一种新的相机运动策略算法，使其能够探索相机的工作空间并寻找图像噪声水平最小的最佳位置。 

---
# Robotic System for Chemical Experiment Automation with Dual Demonstration of End-effector and Jig Operations 

**Title (ZH)**: 化学实验自动化机器人系统：末端执行器和治具操作的双重示示例演示 

**Authors**: Hikaru Sasaki, Naoto Komeno, Takumi Hachimine, Kei Takahashi, Yu-ya Ohnishi, Tetsunori Sugawara, Araki Wakiuchi, Miho Hatanaka, Tomoyuki Miyao, Hiroharu Ajiro, Mikiya Fujii, Takamitsu Matsubara  

**Link**: [PDF](https://arxiv.org/pdf/2506.11384)  

**Abstract**: While robotic automation has demonstrated remarkable performance, such as executing hundreds of experiments continuously over several days, it is challenging to design a program that synchronizes the robot's movements with the experimental jigs to conduct an experiment. We propose a concept that enables the automation of experiments by utilizing dual demonstrations of robot motions and jig operations by chemists in an experimental environment constructed to be controlled by a robot. To verify this concept, we developed a chemical-experiment-automation system consisting of jigs to assist the robot in experiments, a motion-demonstration interface, a jig-control interface, and a mobile manipulator. We validate the concept through polymer-synthesis experiments, focusing on critical liquid-handling tasks such as pipetting and dilution. The experimental results indicate high reproducibility of the demonstrated motions and robust task-success rates. This comprehensive concept not only simplifies the robot programming process for chemists but also provides a flexible and efficient solution to accommodate a wide range of experimental conditions, contributing significantly to the field of chemical experiment automation. 

**Abstract (ZH)**: 利用化学家在实验环境中示教的双示范实现化学实验自动化 

---
# Robust Optimal Task Planning to Maximize Battery Life 

**Title (ZH)**: 鲁棒最优任务规划以最大化电池寿命 

**Authors**: Jiachen Li, Chu Jian, Feiyang Zhao, Shihao Li, Wei Li, Dongmei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11264)  

**Abstract**: This paper proposes a control-oriented optimization platform for autonomous mobile robots (AMRs), focusing on extending battery life while ensuring task completion. The requirement of fast AMR task planning while maintaining minimum battery state of charge, thus maximizing the battery life, renders a bilinear optimization problem. McCormick envelop technique is proposed to linearize the bilinear term. A novel planning algorithm with relaxed constraints is also developed to handle parameter uncertainties robustly with high efficiency ensured. Simulation results are provided to demonstrate the utility of the proposed methods in reducing battery degradation while satisfying task completion requirements. 

**Abstract (ZH)**: 面向自主移动机器人（AMR）的控制导向优化平台：基于快速任务规划与电池寿命最大化的要求，通过McCormick包络技术线性化bilinear项，并提出一种新型规划算法以处理参数不确定性，同时确保高效性，仿真结果验证了所提出方法在减少电池退化的同时满足任务完成要求的有效性。 

---
# A Step-by-Step Guide to Creating a Robust Autonomous Drone Testing Pipeline 

**Title (ZH)**: 创建 robust 自主无人机测试管道的逐步指南 

**Authors**: Yupeng Jiang, Yao Deng, Sebastian Schroder, Linfeng Liang, Suhaas Gambhir, Alice James, Avishkar Seth, James Pirrie, Yihao Zhang, Xi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11400)  

**Abstract**: Autonomous drones are rapidly reshaping industries ranging from aerial delivery and infrastructure inspection to environmental monitoring and disaster response. Ensuring the safety, reliability, and efficiency of these systems is paramount as they transition from research prototypes to mission-critical platforms. This paper presents a step-by-step guide to establishing a robust autonomous drone testing pipeline, covering each critical stage: Software-in-the-Loop (SIL) Simulation Testing, Hardware-in-the-Loop (HIL) Testing, Controlled Real-World Testing, and In-Field Testing. Using practical examples, including the marker-based autonomous landing system, we demonstrate how to systematically verify drone system behaviors, identify integration issues, and optimize performance. Furthermore, we highlight emerging trends shaping the future of drone testing, including the integration of Neurosymbolic and LLMs, creating co-simulation environments, and Digital Twin-enabled simulation-based testing techniques. By following this pipeline, developers and researchers can achieve comprehensive validation, minimize deployment risks, and prepare autonomous drones for safe and reliable real-world operations. 

**Abstract (ZH)**: 自主无人机正迅速重塑从空中配送、基础设施检测到环境监测和灾害响应等各行各业。随着它们从研究原型过渡到 mission-critical 平台，确保这些系统的安全性、可靠性和效率是至关重要的。本文提供了一种构建 robust 自主无人机测试管道的逐步指南，涵盖每个关键阶段：软件在环（SIL）仿真测试、硬件在环（HIL）测试、受控实地测试和现场测试。通过实用示例，包括基于标记的自主着陆系统，我们展示了如何系统地验证无人机系统行为、识别集成问题并优化性能。此外，我们还强调了塑造无人机测试未来趋势的发展方向，包括神经符号和大语言模型的集成、创建协同仿真环境以及基于数字孪生的仿真测试技术。通过遵循此管道，开发者和研究人员可以实现全面验证、最小化部署风险，并为自主无人机的安全可靠实地操作做好准备。 

---
# 15,500 Seconds: Lean UAV Classification Leveraging PEFT and Pre-Trained Networks 

**Title (ZH)**: 15,500秒：基于PEFT和预训练网络的轻量级无人机分类 

**Authors**: Andrew P. Berg, Qian Zhang, Mia Y. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11049)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) pose an escalating security concerns as the market for consumer and military UAVs grows. This paper address the critical data scarcity challenges in deep UAV audio classification. We build upon our previous work expanding novel approaches such as: parameter efficient fine-tuning, data augmentation, and pre-trained networks. We achieve performance upwards of 95\% validation accuracy with EfficientNet-B0. 

**Abstract (ZH)**: 无人航空器（UAVs）随着消费级和军用无人机市场的扩大，带来了日益加剧的安全 concerns。本文解决了深度无人机音频分类中的关键数据稀缺挑战。我们在此前工作的基础上，扩展了新型方法，包括参数高效微调、数据增强和预训练网络。我们使用EfficientNet-B0实现了超过95%的验证准确率。 

---
