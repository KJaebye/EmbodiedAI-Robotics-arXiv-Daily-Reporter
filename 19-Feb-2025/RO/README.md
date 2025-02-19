# SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation 

**Title (ZH)**: SoFar: 语言导向的空间导航连接空间推理与物体操作 

**Authors**: Zekun Qi, Wenyao Zhang, Yufei Ding, Runpei Dong, Xinqiang Yu, Jingwen Li, Lingyun Xu, Baoyu Li, Xialin He, Guofan Fan, Jiazhao Zhang, Jiawei He, Jiayuan Gu, Xin Jin, Kaisheng Ma, Zhizheng Zhang, He Wang, Li Yi  

**Link**: [PDF](https://arxiv.org/pdf/2502.13143)  

**Abstract**: Spatial intelligence is a critical component of embodied AI, promoting robots to understand and interact with their environments. While recent advances have enhanced the ability of VLMs to perceive object locations and positional relationships, they still lack the capability to precisely understand object orientations-a key requirement for tasks involving fine-grained manipulations. Addressing this limitation not only requires geometric reasoning but also an expressive and intuitive way to represent orientation. In this context, we propose that natural language offers a more flexible representation space than canonical frames, making it particularly suitable for instruction-following robotic systems. In this paper, we introduce the concept of semantic orientation, which defines object orientations using natural language in a reference-frame-free manner (e.g., the ''plug-in'' direction of a USB or the ''handle'' direction of a knife). To support this, we construct OrienText300K, a large-scale dataset of 3D models annotated with semantic orientations that link geometric understanding to functional semantics. By integrating semantic orientation into a VLM system, we enable robots to generate manipulation actions with both positional and orientational constraints. Extensive experiments in simulation and real world demonstrate that our approach significantly enhances robotic manipulation capabilities, e.g., 48.7% accuracy on Open6DOR and 74.9% accuracy on SIMPLER. 

**Abstract (ZH)**: 空间智能是体态人工智能的关键组成部分，促进机器人理解并与其环境互动。尽管近期进展增强了视觉语言模型感知物体位置和位置关系的能力，但它们仍然缺乏准确理解物体方向的能力——这对涉及精细操作的任务至关重要。为解决这一限制，不仅需要几何推理，还需要一种表达清晰且直观的方向表示方法。在这个背景下，我们认为自然语言提供了比经典坐标系更灵活的表示空间，使其特别适合于遵循指令的机器人系统。在本文中，我们引入了语义方向的概念，通过自然语言以参考框架无关的方式定义物体方向（例如，USB的“插孔”方向或刀子的“把手”方向）。为此，我们构建了OrienText300K，这是一个大规模的包含语义方向标注的3D模型数据集，将几何理解与功能语义联系起来。通过将语义方向集成到视觉语言模型系统中，使机器人能够生成既包含位置约束也包含方向约束的操纵动作。在模拟和实际环境中的实验表明，我们的方法显著提高了机器人的操纵能力，例如，在Open6DOR上的准确率为48.7%，在SIMPLER上的准确率为74.9%。 

---
# Pre-training Auto-regressive Robotic Models with 4D Representations 

**Title (ZH)**: 预训练自回归机器人模型的4D表示 

**Authors**: Dantong Niu, Yuvan Sharma, Haoru Xue, Giscard Biamby, Junyi Zhang, Ziteng Ji, Trevor Darrell, Roei Herzig  

**Link**: [PDF](https://arxiv.org/pdf/2502.13142)  

**Abstract**: Foundation models pre-trained on massive unlabeled datasets have revolutionized natural language and computer vision, exhibiting remarkable generalization capabilities, thus highlighting the importance of pre-training. Yet, efforts in robotics have struggled to achieve similar success, limited by either the need for costly robotic annotations or the lack of representations that effectively model the physical world. In this paper, we introduce ARM4R, an Auto-regressive Robotic Model that leverages low-level 4D Representations learned from human video data to yield a better pre-trained robotic model. Specifically, we focus on utilizing 3D point tracking representations from videos derived by lifting 2D representations into 3D space via monocular depth estimation across time. These 4D representations maintain a shared geometric structure between the points and robot state representations up to a linear transformation, enabling efficient transfer learning from human video data to low-level robotic control. Our experiments show that ARM4R can transfer efficiently from human video data to robotics and consistently improves performance on tasks across various robot environments and configurations. 

**Abstract (ZH)**: 一种利用低级4D表示自动回归机器人模型（ARM4R） 

---
# RHINO: Learning Real-Time Humanoid-Human-Object Interaction from Human Demonstrations 

**Title (ZH)**: RHINO：从人类示范学习实时人体-人体-物体交互 

**Authors**: Jingxiao Chen, Xinyao Li, Jiahang Cao, Zhengbang Zhu, Wentao Dong, Minghuan Liu, Ying Wen, Yong Yu, Liqing Zhang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13134)  

**Abstract**: Humanoid robots have shown success in locomotion and manipulation. Despite these basic abilities, humanoids are still required to quickly understand human instructions and react based on human interaction signals to become valuable assistants in human daily life. Unfortunately, most existing works only focus on multi-stage interactions, treating each task separately, and neglecting real-time feedback. In this work, we aim to empower humanoid robots with real-time reaction abilities to achieve various tasks, allowing human to interrupt robots at any time, and making robots respond to humans immediately. To support such abilities, we propose a general humanoid-human-object interaction framework, named RHINO, i.e., Real-time Humanoid-human Interaction and Object manipulation. RHINO provides a unified view of reactive motion, instruction-based manipulation, and safety concerns, over multiple human signal modalities, such as languages, images, and motions. RHINO is a hierarchical learning framework, enabling humanoids to learn reaction skills from human-human-object demonstrations and teleoperation data. In particular, it decouples the interaction process into two levels: 1) a high-level planner inferring human intentions from real-time human behaviors; and 2) a low-level controller achieving reactive motion behaviors and object manipulation skills based on the predicted intentions. We evaluate the proposed framework on a real humanoid robot and demonstrate its effectiveness, flexibility, and safety in various scenarios. 

**Abstract (ZH)**: 实时人形机器人-human物体交互框架RHINO：实时反应能力与人机互动 

---
# HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit 

**Title (ZH)**: HOMIE：类人抓举操作同构外骨骼座舱 

**Authors**: Qingwei Ben, Feiyu Jia, Jia Zeng, Junting Dong, Dahua Lin, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13013)  

**Abstract**: Current humanoid teleoperation systems either lack reliable low-level control policies, or struggle to acquire accurate whole-body control commands, making it difficult to teleoperate humanoids for loco-manipulation tasks. To solve these issues, we propose HOMIE, a novel humanoid teleoperation cockpit integrates a humanoid loco-manipulation policy and a low-cost exoskeleton-based hardware system. The policy enables humanoid robots to walk and squat to specific heights while accommodating arbitrary upper-body poses. This is achieved through our novel reinforcement learning-based training framework that incorporates upper-body pose curriculum, height-tracking reward, and symmetry utilization, without relying on any motion priors. Complementing the policy, the hardware system integrates isomorphic exoskeleton arms, a pair of motion-sensing gloves, and a pedal, allowing a single operator to achieve full control of the humanoid robot. Our experiments show our cockpit facilitates more stable, rapid, and precise humanoid loco-manipulation teleoperation, accelerating task completion and eliminating retargeting errors compared to inverse kinematics-based methods. We also validate the effectiveness of the data collected by our cockpit for imitation learning. Our project is fully open-sourced, demos and code can be found in this https URL. 

**Abstract (ZH)**: 当前的人形远程操作系统要么缺乏可靠的低级控制策略，要么难以获得精确的全身控制指令，使得远程操作人形机器人进行移动操作任务变得困难。为了解决这些问题，我们提出HOMIE，这是一种新型的人形远程操作座舱，集成了人形移动操作策略和低成本外骨骼硬件系统。该策略使人形机器人能够行走和蹲做到特定高度，同时适应任意上半身姿态。这通过我们的基于强化学习的新颖训练框架实现，该框架包含了上半身姿态课程、高度跟踪奖励和对称利用，无需依赖任何-motion先验知识。与策略配套，硬件系统集成了同构外骨骼臂、一副运动感应手套和一个踏板，允许单一操作员对人形机器人实现全控。我们的实验表明，我们的座舱促进了更稳定、更快速和更精确的人形移动操作远程操作，加速了任务完成并消除了与基于逆向动力学的方法相比的重新目标化误差。我们还验证了由我们的座舱收集的数据在模仿学习中的有效性。我们的项目完全开源，演示和代码可在如下链接找到：这个https URL。 

---
# D3-ARM: High-Dynamic, Dexterous and Fully Decoupled Cable-driven Robotic Arm 

**Title (ZH)**: D3-ARM: 高动态、灵巧且完全解耦的电缆驱动机器人臂 

**Authors**: Hong Luo, Jianle Xu, Shoujie Li, Huayue Liang, Yanbo Chen, Chongkun Xia, Xueqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12963)  

**Abstract**: Cable transmission enables motors of robotic arm to operate lightweight and low-inertia joints remotely in various environments, but it also creates issues with motion coupling and cable routing that can reduce arm's control precision and performance. In this paper, we present a novel motion decoupling mechanism with low-friction to align the cables and efficiently transmit the motor's power. By arranging these mechanisms at the joints, we fabricate a fully decoupled and lightweight cable-driven robotic arm called D3-Arm with all the electrical components be placed at the base. Its 776 mm length moving part boasts six degrees of freedom (DOF) and only 1.6 kg weights. To address the issue of cable slack, a cable-pretension mechanism is integrated to enhance the stability of long-distance cable transmission. Through a series of comprehensive tests, D3-Arm demonstrated 1.29 mm average positioning error and 2.0 kg payload capacity, proving the practicality of the proposed decoupling mechanisms in cable-driven robotic arm. 

**Abstract (ZH)**: 基于缆索传动的运动解耦机制与低惯量轻量化机器人手臂设计 

---
# RobotIQ: Empowering Mobile Robots with Human-Level Planning for Real-World Execution 

**Title (ZH)**: RobotIQ：赋予移动机器人类人级别的规划能力以实现实际执行 

**Authors**: Emmanuel K. Raptis, Athanasios Ch. Kapoutsis, Elias B. Kosmatopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.12862)  

**Abstract**: This paper introduces RobotIQ, a framework that empowers mobile robots with human-level planning capabilities, enabling seamless communication via natural language instructions through any Large Language Model. The proposed framework is designed in the ROS architecture and aims to bridge the gap between humans and robots, enabling robots to comprehend and execute user-expressed text or voice commands. Our research encompasses a wide spectrum of robotic tasks, ranging from fundamental logical, mathematical, and learning reasoning for transferring knowledge in domains like navigation, manipulation, and object localization, enabling the application of learned behaviors from simulated environments to real-world operations. All encapsulated within a modular crafted robot library suite of API-wise control functions, RobotIQ offers a fully functional AI-ROS-based toolset that allows researchers to design and develop their own robotic actions tailored to specific applications and robot configurations. The effectiveness of the proposed system was tested and validated both in simulated and real-world experiments focusing on a home service scenario that included an assistive application designed for elderly people. RobotIQ with an open-source, easy-to-use, and adaptable robotic library suite for any robot can be found at this https URL. 

**Abstract (ZH)**: 本文介绍了RobotIQ，这是一种框架，使移动机器人具备类似于人类的规划能力，通过任何大型语言模型实现自然语言指令的无缝通信。该提出的框架基于ROS架构设计，旨在弥合人类与机器人之间的差距，使机器人能够理解并执行用户表达的文本或语音命令。我们的研究涵盖了一系列机器人任务，包括导航、操作和物体定位等领域的基本逻辑、数学和学习推理，以便将模拟环境中学到的行为应用于现实世界操作。RobotIQ通过模块化的机器人库API控制函数封装，提供了一个功能齐全的基于AI-ROS的工具集，允许研究人员设计和开发针对特定应用和机器人配置的机器人动作。在家庭服务场景中，该系统包括一个辅助应用程序，旨在为老年人提供服务。RobotIQ附带一个开源、使用简便且可适应任何机器人的机器人库套件，可访问此链接：https://github.com/robotiq/RobotIQ。 

---
# InstructRobot: A Model-Free Framework for Mapping Natural Language Instructions into Robot Motion 

**Title (ZH)**: InstructRobot: 一种无需模型的自然语言指令到机器人运动映射框架 

**Authors**: Iury Cleveston, Alana C. Santana, Paula D. P. Costa, Ricardo R. Gudwin, Alexandre S. Simões, Esther L. Colombini  

**Link**: [PDF](https://arxiv.org/pdf/2502.12861)  

**Abstract**: The ability to communicate with robots using natural language is a significant step forward in human-robot interaction. However, accurately translating verbal commands into physical actions is promising, but still presents challenges. Current approaches require large datasets to train the models and are limited to robots with a maximum of 6 degrees of freedom. To address these issues, we propose a framework called InstructRobot that maps natural language instructions into robot motion without requiring the construction of large datasets or prior knowledge of the robot's kinematics model. InstructRobot employs a reinforcement learning algorithm that enables joint learning of language representations and inverse kinematics model, simplifying the entire learning process. The proposed framework is validated using a complex robot with 26 revolute joints in object manipulation tasks, demonstrating its robustness and adaptability in realistic environments. The framework can be applied to any task or domain where datasets are scarce and difficult to create, making it an intuitive and accessible solution to the challenges of training robots using linguistic communication. Open source code for the InstructRobot framework and experiments can be accessed at this https URL. 

**Abstract (ZH)**: 使用自然语言与机器人交流的能力是人类-机器人交互领域的一项重要进展。然而，将口头指令准确地翻译成物理动作尽管充满希望，但仍面临挑战。现有方法需要大量数据集来训练模型，并且限制在具有最多6个自由度的机器人上。为了解决这些问题，我们提出了一种名为InstructRobot的框架，该框架将自然语言指令映射为机器人运动，而无需构建大量数据集或预先了解机器人的运动学模型。InstructRobot采用了强化学习算法，能够联合学习语言表示和逆运动学模型，简化了整个学习过程。该提出的框架通过使用具有26个转动关节的复杂机器人在物体操作任务中进行了验证，展示了其在现实环境中的稳健性和适应性。该框架可以应用于数据集稀缺且难以创建的任务或领域，使其成为训练使用语言交流的机器人的一种直观且易用的解决方案。InstructRobot框架的开源代码和实验可以在以下网址访问：this https URL。 

---
# Applications of Stretch Reflex for the Upper Limb of Musculoskeletal Humanoids: Protective Behavior, Postural Stability, and Active Induction 

**Title (ZH)**: 上肢Musculoskeletal人形机器人伸展反射的应用：保护性行为、姿势稳定性和主动诱发 

**Authors**: Kento Kawaharazuka, Yuya Koga, Kei Tsuzuki, Moritaka Onitsuka, Yuki Asano, Kei Okada, Koji Kawasaki, Masayuki Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2502.12811)  

**Abstract**: The musculoskeletal humanoid has various biomimetic benefits, and it is important that we can embed and evaluate human reflexes in the actual robot. Although stretch reflex has been implemented in lower limbs of musculoskeletal humanoids, we apply it to the upper limb to discover its useful applications. We consider the implementation of stretch reflex in the actual robot, its active/passive applications, and the change in behavior according to the difference of parameters. 

**Abstract (ZH)**: 具有运动学模仿优势的人形机器人中，我们必须能够在实际机器人中嵌入并评估人体反射。尽管伸展反射已经在人形机器人下肢中实现，但我们将其应用于上肢以发现其有用的应用。我们考虑在实际机器人中实现伸展反射、其主动/被动应用以及参数差异导致行为变化的情况。 

---
# Exceeding the Maximum Speed Limit of the Joint Angle for the Redundant Tendon-driven Structures of Musculoskeletal Humanoids 

**Title (ZH)**: 超越冗余肌腱驱动类人机器人关节角的最大速度限制 

**Authors**: Kento Kawaharazuka, Yuya Koga, Kei Tsuzuki, Moritaka Onitsuka, Yuki Asano, Kei Okada, Koji Kawasaki, Masayuki Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2502.12808)  

**Abstract**: The musculoskeletal humanoid has various biomimetic benefits, and the redundant muscle arrangement is one of its most important characteristics. This redundancy can achieve fail-safe redundant actuation and variable stiffness control. However, there is a problem that the maximum joint angle velocity is limited by the slowest muscle among the redundant muscles. In this study, we propose two methods that can exceed the limited maximum joint angle velocity, and verify the effectiveness with actual robot experiments. 

**Abstract (ZH)**: 具有冗余肌肉排列的肌骨仿人机器人具有多种生物模拟优势，但最大关节角速度受限于最慢的冗余肌肉。为克服这一限制，本研究提出两种方法，并通过实际机器人实验验证了其有效性。 

---
# Design Optimization of Musculoskeletal Humanoids with Maximization of Redundancy to Compensate for Muscle Rupture 

**Title (ZH)**: 具备肌肉断裂补偿能力的最大冗余度设计优化的人体机器人类体设计优化 

**Authors**: Kento Kawaharazuka, Yasunori Toshimitsu, Manabu Nishiura, Yuya Koga, Yusuke Omura, Yuki Asano, Kei Okada, Koji Kawasaki, Masayuki Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2502.12803)  

**Abstract**: Musculoskeletal humanoids have various biomimetic advantages, and the redundant muscle arrangement allowing for variable stiffness control is one of the most important. In this study, we focus on one feature of the redundancy, which enables the humanoid to keep moving even if one of its muscles breaks, an advantage that has not been dealt with in many studies. In order to make the most of this advantage, the design of muscle arrangement is optimized by considering the maximization of minimum available torque that can be exerted when one muscle breaks. This method is applied to the elbow of a musculoskeletal humanoid Musashi with simulations, the design policy is extracted from the optimization results, and its effectiveness is confirmed with the actual robot. 

**Abstract (ZH)**: 具有冗余肌肉排列的肌骨骼人形机器人具有多种生物模仿优势，其中允许变量刚度控制的多余肌肉排列尤为关键。本研究聚焦于冗余的一种特性，即使一根肌肉断裂，机器人仍能继续移动，这一点在许多研究中尚未得到充分探讨。为了充分利用这一优势，通过最大化单根肌肉断裂时仍可利用的最小可用扭矩来优化肌肉排列设计。该方法应用到肌骨骼人形机器人Musashi的肘部，并通过仿真提取设计策略，最终通过实际机器人验证其有效性。 

---
# Responsive Noise-Relaying Diffusion Policy: Responsive and Efficient Visuomotor Control 

**Title (ZH)**: 响应性噪声传递扩散策略：响应性强且高效的感觉运动控制 

**Authors**: Zhuoqun Chen, Xiu Yuan, Tongzhou Mu, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2502.12724)  

**Abstract**: Imitation learning is an efficient method for teaching robots a variety of tasks. Diffusion Policy, which uses a conditional denoising diffusion process to generate actions, has demonstrated superior performance, particularly in learning from multi-modal demonstrates. However, it relies on executing multiple actions to retain performance and prevent mode bouncing, which limits its responsiveness, as actions are not conditioned on the most recent observations. To address this, we introduce Responsive Noise-Relaying Diffusion Policy (RNR-DP), which maintains a noise-relaying buffer with progressively increasing noise levels and employs a sequential denoising mechanism that generates immediate, noise-free actions at the head of the sequence, while appending noisy actions at the tail. This ensures that actions are responsive and conditioned on the latest observations, while maintaining motion consistency through the noise-relaying buffer. This design enables the handling of tasks requiring responsive control, and accelerates action generation by reusing denoising steps. Experiments on response-sensitive tasks demonstrate that, compared to Diffusion Policy, ours achieves 18% improvement in success rate. Further evaluation on regular tasks demonstrates that RNR-DP also exceeds the best acceleration method by 6.9%, highlighting its computational efficiency advantage in scenarios where responsiveness is less critical. 

**Abstract (ZH)**: 响应式噪声传递扩散策略在模仿学习中的应用研究 

---
# Soft Arm-Motor Thrust Characterization for a Pneumatically Actuated Soft Morphing Quadrotor 

**Title (ZH)**: 气动驱动软形态变换四旋翼无人机的软臂-电机推力特性研究 

**Authors**: Vidya Sumathy, Jakub Haluska, George Nikolokopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.12716)  

**Abstract**: In this work, an experimental characterization of the configuration space of a soft, pneumatically actuated morphing quadrotor is presented, with a focus on precise thrust characterization of its flexible arms, considering the effect of downwash. Unlike traditional quadrotors, the soft drone has pneumatically actuated arms, introducing complex, nonlinear interactions between motor thrust and arm deformation, which make precise control challenging. The silicone arms are actuated using differential pressure to achieve flexibility and thus have a variable workspace compared to their fixed counter-parts. The deflection of the soft arms during compression and expansion is controlled throughout the flight. However, in real time, the downwash from the motor attached at the tip of the soft arm generates a significant and random disturbance on the arm. This disturbance affects both the desired deflection of the arm and the overall stability of the system. To address this factor, an experimental characterization of the effect of downwash on the deflection angle of the arm is conducted. 

**Abstract (ZH)**: 一种软气动驱动四旋翼飞行器柔臂配置空间的实验characterization及其精确推力characterization，考虑下洗流的影响 

---
# SATA: Safe and Adaptive Torque-Based Locomotion Policies Inspired by Animal Learning 

**Title (ZH)**: SATA：受动物学习启发的安全自适应扭矩基运动策略 

**Authors**: Peizhuo Li, Hongyi Li, Ge Sun, Jin Cheng, Xinrong Yang, Guillaume Bellegarda, Milad Shafiee, Yuhong Cao, Auke Ijspeert, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2502.12674)  

**Abstract**: Despite recent advances in learning-based controllers for legged robots, deployments in human-centric environments remain limited by safety concerns. Most of these approaches use position-based control, where policies output target joint angles that must be processed by a low-level controller (e.g., PD or impedance controllers) to compute joint torques. Although impressive results have been achieved in controlled real-world scenarios, these methods often struggle with compliance and adaptability when encountering environments or disturbances unseen during training, potentially resulting in extreme or unsafe behaviors. Inspired by how animals achieve smooth and adaptive movements by controlling muscle extension and contraction, torque-based policies offer a promising alternative by enabling precise and direct control of the actuators in torque space. In principle, this approach facilitates more effective interactions with the environment, resulting in safer and more adaptable behaviors. However, challenges such as a highly nonlinear state space and inefficient exploration during training have hindered their broader adoption. To address these limitations, we propose SATA, a bio-inspired framework that mimics key biomechanical principles and adaptive learning mechanisms observed in animal locomotion. Our approach effectively addresses the inherent challenges of learning torque-based policies by significantly improving early-stage exploration, leading to high-performance final policies. Remarkably, our method achieves zero-shot sim-to-real transfer. Our experimental results indicate that SATA demonstrates remarkable compliance and safety, even in challenging environments such as soft/slippery terrain or narrow passages, and under significant external disturbances, highlighting its potential for practical deployments in human-centric and safety-critical scenarios. 

**Abstract (ZH)**: 基于扭矩的学习控制方法在人本环境中应用的研究 

---
# LiMo-Calib: On-Site Fast LiDAR-Motor Calibration for Quadruped Robot-Based Panoramic 3D Sensing System 

**Title (ZH)**: LiMo-Calib：基于四足机器人导向全景3D感知系统中的现场快速LiDAR-电机校准 

**Authors**: Jianping Li, Zhongyuan Liu, Xinhang Xu, Jinxin Liu, Shenghai Yuan, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.12655)  

**Abstract**: Conventional single LiDAR systems are inherently constrained by their limited field of view (FoV), leading to blind spots and incomplete environmental awareness, particularly on robotic platforms with strict payload limitations. Integrating a motorized LiDAR offers a practical solution by significantly expanding the sensor's FoV and enabling adaptive panoramic 3D sensing. However, the high-frequency vibrations of the quadruped robot introduce calibration challenges, causing variations in the LiDAR-motor transformation that degrade sensing accuracy. Existing calibration methods that use artificial targets or dense feature extraction lack feasibility for on-site applications and real-time implementation. To overcome these limitations, we propose LiMo-Calib, an efficient on-site calibration method that eliminates the need for external targets by leveraging geometric features directly from raw LiDAR scans. LiMo-Calib optimizes feature selection based on normal distribution to accelerate convergence while maintaining accuracy and incorporates a reweighting mechanism that evaluates local plane fitting quality to enhance robustness. We integrate and validate the proposed method on a motorized LiDAR system mounted on a quadruped robot, demonstrating significant improvements in calibration efficiency and 3D sensing accuracy, making LiMo-Calib well-suited for real-world robotic applications. The demo video is available at: this https URL 

**Abstract (ZH)**: 传统的单线激光雷达系统受限于其有限的视场角（FoV），导致盲区和环境感知不完整，特别是在有严格载重限制的机器人平台上。通过整合电机驱动的激光雷达，可以显著扩展传感器的FoV，并实现自适应全景3D感知，提供一种实用的解决方案。然而，四足机器人的高频振动引入了校准挑战，导致激光雷达-电机变换的偏差，降低了感知精度。现有的使用人工目标或密集特征提取的校准方法在实地应用和实时实施中缺乏可行性。为克服这些限制，我们提出了一种高效的实地校准方法LiMo-Calib，通过直接利用原始激光雷达扫描中的几何特征来消除对外部目标的需求。LiMo-Calib基于正态分布优化特征选择以加速收敛同时保持准确性，并引入了一种重新加权机制，通过评估局部平面拟合质量来增强鲁棒性。我们在一个安装在四足机器人上的电机驱动激光雷达系统上集成了并验证了该方法，展示了校准效率和3D感知精度的显著提升，使LiMo-Calib适用于实际的机器人应用。演示视频可访问：this https URL 

---
# Learning-based Dynamic Robot-to-Human Handover 

**Title (ZH)**: 基于学习的动态机器人到人类的手递交接 

**Authors**: Hyeonseong Kim, Chanwoo Kim, Matthew Pan, Kyungjae Lee, Sungjoon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12602)  

**Abstract**: This paper presents a novel learning-based approach to dynamic robot-to-human handover, addressing the challenges of delivering objects to a moving receiver. We hypothesize that dynamic handover, where the robot adjusts to the receiver's movements, results in more efficient and comfortable interaction compared to static handover, where the receiver is assumed to be stationary. To validate this, we developed a nonparametric method for generating continuous handover motion, conditioned on the receiver's movements, and trained the model using a dataset of 1,000 human-to-human handover demonstrations. We integrated preference learning for improved handover effectiveness and applied impedance control to ensure user safety and adaptiveness. The approach was evaluated in both simulation and real-world settings, with user studies demonstrating that dynamic handover significantly reduces handover time and improves user comfort compared to static methods. Videos and demonstrations of our approach are available at this https URL . 

**Abstract (ZH)**: 基于学习的动态机器人到人类递物方法：面向运动接收者的高效舒适递物 

---
# Learning a High-quality Robotic Wiping Policy Using Systematic Reward Analysis and Visual-Language Model Based Curriculum 

**Title (ZH)**: 基于系统奖励分析和视觉-语言模型引导课程的学习高质量机器人擦拭策略 

**Authors**: Yihong Liu, Dongyeop Kang, Sehoon Ha  

**Link**: [PDF](https://arxiv.org/pdf/2502.12599)  

**Abstract**: Autonomous robotic wiping is an important task in various industries, ranging from industrial manufacturing to sanitization in healthcare. Deep reinforcement learning (Deep RL) has emerged as a promising algorithm, however, it often suffers from a high demand for repetitive reward engineering. Instead of relying on manual tuning, we first analyze the convergence of quality-critical robotic wiping, which requires both high-quality wiping and fast task completion, to show the poor convergence of the problem and propose a new bounded reward formulation to make the problem feasible. Then, we further improve the learning process by proposing a novel visual-language model (VLM) based curriculum, which actively monitors the progress and suggests hyperparameter tuning. We demonstrate that the combined method can find a desirable wiping policy on surfaces with various curvatures, frictions, and waypoints, which cannot be learned with the baseline formulation. The demo of this project can be found at: this https URL. 

**Abstract (ZH)**: 自主机器人擦拭是一项重要任务，应用于从工业制造到医疗消毒的各个领域。深度强化学习（Deep RL）已成为一种有前景的算法，然而它通常会遭受重复奖励工程需求高的困扰。我们首先分析关键质量要求的机器人擦拭的收敛性，该要求既需要高质量擦拭又需要快速任务完成，展示了该问题的不良收敛性，并提出了一种新的有界奖励形式来使问题可行。然后，我们通过提出一种基于视觉语言模型（VLM）的新颖课程学习方法进一步改进学习过程，该方法积极监控进度并建议超参数调整。我们证明，结合方法能够在具有各种曲率、摩擦力和航点的表面上找到一个理想的擦拭策略，这是基线形式无法学习到的。该项目的演示可以在以下链接找到：this https URL。 

---
# Design and Implementation of a Dual Uncrewed Surface Vessel Platform for Bathymetry Research under High-flow Conditions 

**Title (ZH)**: 高流速条件下双无人驾驶水面船只平台的设计与实现 

**Authors**: Dinesh Kumar, Amin Ghorbanpour, Kin Yen, Iman Soltani  

**Link**: [PDF](https://arxiv.org/pdf/2502.12539)  

**Abstract**: Bathymetry, the study of underwater topography, relies on sonar mapping of submerged structures. These measurements, critical for infrastructure health monitoring, often require expensive instrumentation. The high financial risk associated with sensor damage or vessel loss creates a reluctance to deploy uncrewed surface vessels (USVs) for bathymetry. However, the crewed-boat bathymetry operations, are costly, pose hazards to personnel, and frequently fail to achieve the stable conditions necessary for bathymetry data collection, especially under high currents. Further research is essential to advance autonomous control, navigation, and data processing technologies, with a particular focus on bathymetry. There is a notable lack of accessible hardware platforms that allow for integrated research in both bathymetry-focused autonomous control and navigation, as well as data evaluation and processing. This paper addresses this gap through the design and implementation of two complementary USV systems tailored for uncrewed bathymetry research. This includes a low-cost USV for Navigation And Control research (NAC-USV) and a second, high-end USV equipped with a high-resolution multi-beam sonar and the associated hardware for Bathymetry data quality Evaluation and Post-processing research (BEP-USV). The NAC-USV facilitates the investigation of autonomous, fail-safe navigation and control, emphasizing the stability requirements for high-quality bathymetry data collection while minimizing the risk to equipment. The BEP-USV, which mirrors the NAC-USV hardware, is then used for additional control validation and in-depth exploration of bathymetry data evaluation and post-processing methodologies. We detail the design and implementation of both systems, and open source the design. Furthermore, we demonstrate the system's effectiveness in a range of operational scenarios. 

**Abstract (ZH)**: underwater地形测量依赖于 submerged结构的声纳mapping。这些测量对于基础设施健康监测至关重要，但通常需要昂贵的仪器。与传感器损坏或船只损失相关的高经济风险导致对无人水面舰艇（USVs）进行地形测量的犹豫。然而，有人船进行的地形测量操作成本高、人员有危险，并且常常无法达到地形测量数据采集所需的稳定条件，尤其是在强流条件下。进一步的研究对于推进自主控制、导航和数据处理技术至关重要，特别是对于地形测量。目前存在一个明显的差距，即能够集成研究焦点于地形测量的自主控制和导航，以及数据评估和处理的可访问硬件平台较少。本文通过设计和实现两个针对无人地形测量研究的互补USV系统来填补这一空白。这包括一个用于导航和控制研究的低成本USV（NAC-USV），以及一个配备高分辨率多波束声纳和相关硬件的高性能USV，用于地形测量数据质量评估和后处理研究（BEP-USV）。NAC-USV促进对自主、安全导航和控制的调查，强调高质量地形测量数据采集所需的稳定性要求，同时最大限度地减少对设备的风险。随后，BEP-USV使用与NAC-USV相同的硬件进行额外的控制验证，并深入探索地形测量数据评估和后处理方法。我们详细说明了两个系统的设计和实现，并开源了设计。此外，我们展示了该系统在多种操作场景中的有效性。 

---
# GSCE: A Prompt Framework with Enhanced Reasoning for Reliable LLM-driven Drone Control 

**Title (ZH)**: GSCE: 一种增强推理的提示框架，用于可靠的大规模语言模型驱动的无人机控制 

**Authors**: Wenhao Wang, Yanyan Li, Long Jiao, Jiawei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12531)  

**Abstract**: The integration of Large Language Models (LLMs) into robotic control, including drones, has the potential to revolutionize autonomous systems. Research studies have demonstrated that LLMs can be leveraged to support robotic operations. However, when facing tasks with complex reasoning, concerns and challenges are raised about the reliability of solutions produced by LLMs. In this paper, we propose a prompt framework with enhanced reasoning to enable reliable LLM-driven control for drones. Our framework consists of novel technical components designed using Guidelines, Skill APIs, Constraints, and Examples, namely GSCE. GSCE is featured by its reliable and constraint-compliant code generation. We performed thorough experiments using GSCE for the control of drones with a wide level of task complexities. Our experiment results demonstrate that GSCE can significantly improve task success rates and completeness compared to baseline approaches, highlighting its potential for reliable LLM-driven autonomous drone systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在机器人控制中的整合，包括无人机，有望彻底改变自主系统。研究表明，LLMs可以用于支持机器人操作。然而，在面对复杂推理任务时，关于由LLMs生成的解决方案可靠性的担忧和挑战被提出。本文提出了一种增强推理的提示框架，以实现可靠的LLM驱动无人机控制。该框架包含使用指南、技能API、约束和示例设计的新技术组件，即GSCE。GSCE的特点是其可靠且符合约束的代码生成。我们使用GSCE对不同类型复杂度的无人机控制任务进行了全面实验。实验结果表明，与基线方法相比，GSCE可以显著提高任务的成功率和完整性，突显了其在可靠LLM驱动的自主无人机系统中的潜力。 

---
# Memory-updated-based Framework for 100% Reliable Flexible Flat Cables Insertion 

**Title (ZH)**: 基于内存更新的100%可靠柔性扁 cable 插接口技术框架 

**Authors**: Zhengrong Ling, Xiong Yang, Dong Guo, Hongyuan Chang, Tieshan Zhang, Ruijia Zhang, Yajing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12514)  

**Abstract**: Automatic assembly lines have increasingly replaced human labor in various tasks; however, the automation of Flexible Flat Cable (FFC) insertion remains unrealized due to its high requirement for effective feedback and dynamic operation, limiting approximately 11% of global industrial capacity. Despite lots of approaches, like vision-based tactile sensors and reinforcement learning, having been proposed, the implementation of human-like high-reliable insertion (i.e., with a 100% success rate in completed insertion) remains a big challenge. Drawing inspiration from human behavior in FFC insertion, which involves sensing three-dimensional forces, translating them into physical concepts, and continuously improving estimates, we propose a novel framework. This framework includes a sensing module for collecting three-dimensional tactile data, a perception module for interpreting this data into meaningful physical signals, and a memory module based on Bayesian theory for reliability estimation and control. This strategy enables the robot to accurately assess its physical state and generate reliable status estimations and corrective actions. Experimental results demonstrate that the robot using this framework can detect alignment errors of 0.5 mm with an accuracy of 97.92% and then achieve a 100% success rate in all completed tests after a few iterations. This work addresses the challenges of unreliable perception and control in complex insertion tasks, highlighting the path toward the development of fully automated production lines. 

**Abstract (ZH)**: 自动装配线已在各种任务中逐渐取代了人力劳动；然而，由于柔性扁平电缆（FFC）插入对有效反馈和动态操作有高度要求，FFC插入的自动化尚未实现，限制了大约11%的全球工业产能。尽管提出了许多方法，如基于视觉的触觉传感器和强化学习，但实现类似人类的高可靠性插入（即完成插入的100%成功率）仍然是一个巨大挑战。借鉴人类在FFC插入过程中的行为，包括感知三维力、将其转化为物理概念并不断改进估计，我们提出了一种新型框架。该框架包括一个用于收集三维触觉数据的感知模块、一个用于将这些数据解释为有意义的物理信号的认知模块以及基于贝叶斯理论的记忆模块，用于可靠性估计和控制。这种策略使机器人能够准确评估其物理状态并生成可靠的状况估计和纠正措施。实验结果表明，使用该框架的机器人在几次迭代后能够检测到0.5 mm的对准误差，并且在所有完成的测试中实现100%的成功率。本工作解决了复杂插入任务中不可靠感知和控制的挑战，指出了全自动化生产线开发的方向。 

---
# USPilot: An Embodied Robotic Assistant Ultrasound System with Large Language Model Enhanced Graph Planner 

**Title (ZH)**: USPilot: 一种增强图规划的大语言模型驱动的 embodied 超声机器人助手系统 

**Authors**: Mingcong Chen, Siqi Fan, Guanglin Cao, Hongbin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12498)  

**Abstract**: In the era of Large Language Models (LLMs), embodied artificial intelligence presents transformative opportunities for robotic manipulation tasks. Ultrasound imaging, a widely used and cost-effective medical diagnostic procedure, faces challenges due to the global shortage of professional sonographers. To address this issue, we propose USPilot, an embodied robotic assistant ultrasound system powered by an LLM-based framework to enable autonomous ultrasound acquisition. USPilot is designed to function as a virtual sonographer, capable of responding to patients' ultrasound-related queries and performing ultrasound scans based on user intent. By fine-tuning the LLM, USPilot demonstrates a deep understanding of ultrasound-specific questions and tasks. Furthermore, USPilot incorporates an LLM-enhanced Graph Neural Network (GNN) to manage ultrasound robotic APIs and serve as a task planner. Experimental results show that the LLM-enhanced GNN achieves unprecedented accuracy in task planning on public datasets. Additionally, the system demonstrates significant potential in autonomously understanding and executing ultrasound procedures. These advancements bring us closer to achieving autonomous and potentially unmanned robotic ultrasound systems, addressing critical resource gaps in medical imaging. 

**Abstract (ZH)**: 在大型语言模型时代，具身人工智能为机器人操作任务提供了变革性的机会。超声成像作为一种广泛使用且成本效益高的医疗诊断程序，由于专业超声技师的全球短缺而面临挑战。为解决这一问题，我们提出了USPilot，一种基于大型语言模型框架的动力具身机器人辅助超声系统，用于实现自动超声成像。USPilot被设计为一种虚拟超声技师，能够响应患者的超声相关查询，并根据用户意图执行超声扫描。通过微调大型语言模型，USPilot展示了对超声特定问题和任务的深刻理解。此外，USPilot还集成了增强的图形神经网络（GNN），用于管理和作为任务规划器处理超声机器人API。实验结果表明，增强的GNN在公共数据集上的任务规划中达到了前所未有的准确性。此外，该系统在自主理解和执行超声程序方面显示出显著的潜力。这些进步使我们更接近实现自主的、可能是无人驾驶的机器人超声系统，解决医学成像中的关键资源缺口。 

---
# Multi-vision-based Picking Point Localisation of Target Fruit for Harvesting Robots 

**Title (ZH)**: 基于多视角的目标水果采收机器人采摘点定位 

**Authors**: C. Beldek, A. Dunn, J. Cunningham, E. Sariyildiz, S. L. Phung, G.Alici  

**Link**: [PDF](https://arxiv.org/pdf/2502.12406)  

**Abstract**: This paper presents multi-vision-based localisation strategies for harvesting robots. Identifying picking points accurately is essential for robotic harvesting because insecure grasping can lead to economic loss through fruit damage and dropping. In this study, two multi-vision-based localisation methods, namely the analytical approach and model-based algorithms, were employed. The actual geometric centre points of fruits were collected using a motion capture system (mocap), and two different surface points Cfix and Ceih were extracted using two Red-Green-Blue-Depth (RGB-D) cameras. First, the picking points of the target fruit were detected using analytical methods. Second, various primary and ensemble learning methods were employed to predict the geometric centre of target fruits by taking surface points as input. Adaboost regression, the most successful model-based localisation algorithm, achieved 88.8% harvesting accuracy with a Mean Euclidean Distance (MED) of 4.40 mm, while the analytical approach reached 81.4% picking success with a MED of 14.25 mm, both demonstrating better performance than the single-camera, which had a picking success rate of 77.7% with a MED of 24.02 mm. To evaluate the effect of picking point accuracy in collecting fruits, a series of robotic harvesting experiments were performed utilising a collaborative robot (cobot). It is shown that multi-vision systems can improve picking point localisation, resulting in higher success rates of picking in robotic harvesting. 

**Abstract (ZH)**: 基于多视图的采摘机器人定位策略及其应用 

---
# Sensing-based Robustness Challenges in Agricultural Robotic Harvesting 

**Title (ZH)**: 基于传感的农业机器人收获稳健性挑战 

**Authors**: C. Beldek, J. Cunningham, M.Aydin, E. Sariyildiz, S. L. Phung, G.Alici  

**Link**: [PDF](https://arxiv.org/pdf/2502.12403)  

**Abstract**: This paper presents the challenges agricultural robotic harvesters face in detecting and localising fruits under various environmental disturbances. In controlled laboratory settings, both the traditional HSV (Hue Saturation Value) transformation and the YOLOv8 (You Only Look Once) deep learning model were employed. However, only YOLOv8 was utilised in outdoor experiments, as the HSV transformation was not capable of accurately drawing fruit contours. Experiments include ten distinct fruit patterns with six apples and six oranges. A grid structure for homography (perspective) transformation was employed to convert detected midpoints into 3D world coordinates. The experiments evaluated detection and localisation under varying lighting and background disturbances, revealing accurate performance indoors, but significant challenges outdoors. Our results show that indoor experiments using YOLOv8 achieved 100% detection accuracy, while outdoor conditions decreased performance, with an average accuracy of 69.15% for YOLOv8 under direct sunlight. The study demonstrates that real-world applications reveal significant limitations due to changing lighting, background disturbances, and colour and shape variability. These findings underscore the need for further refinement of algorithms and sensors to enhance the robustness of robotic harvesters for agricultural use. 

**Abstract (ZH)**: 本文探讨了农业机器人收获机在各种环境干扰下检测和定位水果所面临的挑战。在受控实验室环境中，采用了传统的HSV变换和YOLOv8深度学习模型。但在户外实验中，仅使用了YOLOv8，因为HSV变换无法准确勾勒出水果轮廓。实验包括十种不同的水果模式，共六颗苹果和六颗橙子。采用网格结构进行霍夫变换（透视变换），将检测到的中点转换为3D世界坐标。实验在不同光照和背景干扰条件下评估了检测和定位的性能，室内环境下表现准确，但室外环境下存在显著挑战。实验结果显示，使用YOLOv8的室内实验实现了100%的检测准确性，但在直射阳光下的室外条件下，YOLOv8的平均检测准确率为69.15%。研究表明，实际应用中由于光照变化、背景干扰及颜色和形状变化，显示出显著的局限性。这些发现强调了进一步改进算法和传感器的必要性，以提高农业用机器人收获器的鲁棒性。 

---
# Soft Robotics for Search and Rescue: Advancements, Challenges, and Future Directions 

**Title (ZH)**: 软体机器人在搜索与救援中的应用：进展、挑战及未来方向 

**Authors**: Abhishek Sebastian  

**Link**: [PDF](https://arxiv.org/pdf/2502.12373)  

**Abstract**: Soft robotics has emerged as a transformative technology in Search and Rescue (SAR) operations, addressing challenges in navigating complex, hazardous environments that often limit traditional rigid robots. This paper critically examines advancements in soft robotic technologies tailored for SAR applications, focusing on their unique capabilities in adaptability, safety, and efficiency. By leveraging bio-inspired designs, flexible materials, and advanced locomotion mechanisms, such as crawling, rolling, and shape morphing, soft robots demonstrate exceptional potential in disaster scenarios. However, significant barriers persist, including material durability, power inefficiency, sensor integration, and control complexity. This comprehensive review highlights the current state of soft robotics in SAR, discusses simulation methodologies and hardware validations, and introduces performance metrics essential for their evaluation. By bridging the gap between theoretical advancements and practical deployment, this study underscores the potential of soft robotic systems to revolutionize SAR missions and advocates for continued interdisciplinary innovation to overcome existing limitations. 

**Abstract (ZH)**: 软体机器人技术在搜索与救援（SAR）操作中的 emergence 作为一种变革性技术，解决了传统刚性机器人在导航复杂及危险环境中的局限性。本文从适应性、安全性和效率的角度， critically 评析了为 SAR 应用量身定制的软体机器人技术 advancements，通过借鉴生物启发式设计、柔性材料以及爬行、滚动和形态变化等先进运动机制，软体机器人在灾难场景中展现出非凡的潜力。然而，仍存在一些重大障碍，包括材料耐久性、能源效率、传感器集成和控制复杂性等问题。本文对软体机器人在 SAR 中的当前状态进行了全面回顾，讨论了仿真方法和硬件验证，并介绍了评估其性能的基本指标。通过在理论进步与实际应用之间的接轨，本研究强调了软体机器人系统在变革 SAR 任务方面的发展潜力，并呼吁推动跨学科创新以克服现有限制。 

---
# IMLE Policy: Fast and Sample Efficient Visuomotor Policy Learning via Implicit Maximum Likelihood Estimation 

**Title (ZH)**: IMLE策略：通过隐式最大似然估计实现快速和样本高效的动作视觉策略学习 

**Authors**: Krishan Rana, Robert Lee, David Pershouse, Niko Suenderhauf  

**Link**: [PDF](https://arxiv.org/pdf/2502.12371)  

**Abstract**: Recent advances in imitation learning, particularly using generative modelling techniques like diffusion, have enabled policies to capture complex multi-modal action distributions. However, these methods often require large datasets and multiple inference steps for action generation, posing challenges in robotics where the cost for data collection is high and computation resources are limited. To address this, we introduce IMLE Policy, a novel behaviour cloning approach based on Implicit Maximum Likelihood Estimation (IMLE). IMLE Policy excels in low-data regimes, effectively learning from minimal demonstrations and requiring 38\% less data on average to match the performance of baseline methods in learning complex multi-modal behaviours. Its simple generator-based architecture enables single-step action generation, improving inference speed by 97.3\% compared to Diffusion Policy, while outperforming single-step Flow Matching. We validate our approach across diverse manipulation tasks in simulated and real-world environments, showcasing its ability to capture complex behaviours under data constraints. Videos and code are provided on our project page: this https URL. 

**Abstract (ZH)**: Recent Advances in Imitation Learning Using Generative Modeling Techniques like Diffusion Have Enabled Policies to Capture Complex Multi-Modal Action Distributions, but IMLE Policy Addresses Challenges in Robotics with Implicit Maximum Likelihood Estimation in Low-Data Regimes 

---
# Hovering Flight of Soft-Actuated Insect-Scale Micro Aerial Vehicles using Deep Reinforcement Learning 

**Title (ZH)**: 软驱动昆虫尺度微型飞行器的悬浮飞行基于深度强化学习 

**Authors**: Yi-Hsuan Hsiao, Wei-Tung Chen, Yun-Sheng Chang, Pulkit Agrawal, YuFeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12355)  

**Abstract**: Soft-actuated insect-scale micro aerial vehicles (IMAVs) pose unique challenges for designing robust and computationally efficient controllers. At the millimeter scale, fast robot dynamics ($\sim$ms), together with system delay, model uncertainty, and external disturbances significantly affect flight performances. Here, we design a deep reinforcement learning (RL) controller that addresses system delay and uncertainties. To initialize this neural network (NN) controller, we propose a modified behavior cloning (BC) approach with state-action re-matching to account for delay and domain-randomized expert demonstration to tackle uncertainty. Then we apply proximal policy optimization (PPO) to fine-tune the policy during RL, enhancing performance and smoothing commands. In simulations, our modified BC substantially increases the mean reward compared to baseline BC; and RL with PPO improves flight quality and reduces command fluctuations. We deploy this controller on two different insect-scale aerial robots that weigh 720 mg and 850 mg, respectively. The robots demonstrate multiple successful zero-shot hovering flights, with the longest lasting 50 seconds and root-mean-square errors of 1.34 cm in lateral direction and 0.05 cm in altitude, marking the first end-to-end deep RL-based flight on soft-driven IMAVs. 

**Abstract (ZH)**: 软驱动昆虫尺度微型飞行器（IMAVs）的软化执行控制器设计提出独特挑战：面向鲁棒性和计算效率的控制设计。在毫米尺度上，快速机器人动力学（≈ms）、系统延迟、模型不确定性以及外部干扰严重影响飞行性能。在这里，我们设计了一种深度强化学习（RL）控制器，以应对系统延迟和不确定性。为了初始化这个神经网络（NN）控制器，我们提出了带有状态-动作重新匹配的修改行为 cloning（BC）方法，并利用域随机化专家演示来应对不确定性。然后我们使用近端策略优化（PPO）在RL过程中精细调整策略，提升性能并平滑命令。在仿真中，我们修改后的BC相比于基线BC显著提高了平均奖励；而使用PPO的RL进一步提升了飞行质量并减少了命令波动。我们将此控制器部署在两个不同重量的昆虫尺度微型飞行器上，分别为720 mg和850 mg。这些飞行器展现了多次成功的一次性悬停飞行，最长持续50秒，侧向方向的均方根误差为1.34 cm，海拔方向的均方根误差为0.05 cm，标志着软驱动IMAVs端到端深度RL控制的首次实现。 

---
# Improving Grip Stability Using Passive Compliant Microspine Arrays for Soft Robots in Unstructured Terrain 

**Title (ZH)**: 基于非结构化地形软机器人用被动 compliant 微钩阵列提升握持稳定性 

**Authors**: Lauren Ervin, Harish Bezawada, Vishesh Vikas  

**Link**: [PDF](https://arxiv.org/pdf/2502.12347)  

**Abstract**: Microspine grippers are small spines commonly found on insect legs that reinforce surface interaction by engaging with asperities to increase shear force and traction. An array of such microspines, when integrated into the limbs or undercarriage of a robot, can provide the ability to maneuver uneven terrains, traverse inclines, and even climb walls. Conformability and adaptability of soft robots makes them ideal candidates for these applications involving traversal of complex, unstructured terrains. However, there remains a real-life realization gap for soft locomotors pertaining to their transition from controlled lab environment to the field by improving grip stability through effective integration of microspines. We propose a passive, compliant microspine stacked array design to enhance the locomotion capabilities of mobile soft robots, in our case, ones that are motor tendon actuated. We offer a standardized microspine array integration method with effective soft-compliant stiffness integration, and reduced complexity resulting from a single actuator passively controlling them. The presented design utilizes a two-row, stacked microspine array configuration that offers additional gripping capabilities on extremely steep/irregular surfaces from the top row while not hindering the effectiveness of the more frequently active bottom row. We explore different configurations of the microspine array to account for changing surface topologies and enable independent, adaptable gripping of asperities per microspine. Field test experiments are conducted on various rough surfaces including concrete, brick, compact sand, and tree roots with three robots consisting of a baseline without microspines compared against two robots with different combinations of microspine arrays. Tracking results indicate that the inclusion of microspine arrays increases planar displacement on average by 15 and 8 times. 

**Abstract (ZH)**: 微刺爪是一种常见的昆虫腿部小刺，通过与表面不平度啮合以增加切向力和抓地力。将一系列微刺爪整合到机器人的肢体或底盘中，可以增强其在不平地形上移动、跨越斜坡，甚至攀墙的能力。软体机器人的柔韧性和适应性使其成为涉及复杂、非结构化地形穿越的理想选择。然而，软体行者的实际应用仍然存在从受控实验室环境过渡到野外的实现差距，特别是在通过有效整合微刺爪提高抓地稳定性方面。我们提出了一种被动的、柔性的微刺爪堆叠阵列设计，以增强机动作软体机器人的移动能力，特别是对于由肌腱驱动的机器。我们提供了一种标准化的微刺爪阵列整合方法，具有有效的软性柔顺刚度整合，以及由于单一驱动器的被动控制所降低的复杂性。所提出的设计采用两行堆叠的微刺爪阵列配置，上行行的微刺爪提供了在极其陡峭/不规则表面上的额外抓握能力，而不妨碍下行行微刺爪更频繁的抓握效果。我们探索了不同配置的微刺爪阵列，以适应变化的表面拓扑，并使每个微刺爪独立地适应抓握表面不平度。在各种粗糙表面上（包括混凝土、砖块、密砂和树根）进行了实地测试，使用三台机器人进行实验，包括无微刺爪的基线机器人和配备不同微刺爪阵列组合的两台机器人。跟踪结果显示，加入微刺爪阵列平均增加了平面位移15倍和8倍。 

---
# X-IL: Exploring the Design Space of Imitation Learning Policies 

**Title (ZH)**: X-IL：探索模仿学习策略的设计空间 

**Authors**: Xiaogang Jia, Atalay Donat, Xi Huang, Xuan Zhao, Denis Blessing, Hongyi Zhou, Hanyi Zhang, Han A. Wang, Qian Wang, Rudolf Lioutikov, Gerhard Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2502.12330)  

**Abstract**: Designing modern imitation learning (IL) policies requires making numerous decisions, including the selection of feature encoding, architecture, policy representation, and more. As the field rapidly advances, the range of available options continues to grow, creating a vast and largely unexplored design space for IL policies. In this work, we present X-IL, an accessible open-source framework designed to systematically explore this design space. The framework's modular design enables seamless swapping of policy components, such as backbones (e.g., Transformer, Mamba, xLSTM) and policy optimization techniques (e.g., Score-matching, Flow-matching). This flexibility facilitates comprehensive experimentation and has led to the discovery of novel policy configurations that outperform existing methods on recent robot learning benchmarks. Our experiments demonstrate not only significant performance gains but also provide valuable insights into the strengths and weaknesses of various design choices. This study serves as both a practical reference for practitioners and a foundation for guiding future research in imitation learning. 

**Abstract (ZH)**: 设计现代模仿学习（IL）策略需要做出众多决策，包括特征编码、架构、策略表示的选择等。随着该领域的快速发展，可供选择的范围不断扩大，为IL策略创造了广阔的、尚未充分探索的设计空间。在本文中，我们介绍了一种开源框架X-IL，旨在系统地探索这一设计空间。该框架模块化的设计使得可以无缝地替换策略组件，如骨干网络（例如，Transformer、Mamba、xLSTM）和策略优化技术（例如，Score-matching、Flow-matching）。这种灵活性促进了全面的实验，并发现了优于现有方法的新颖策略配置。我们的实验证明了显著的性能提升，并提供了关于各种设计选择的优缺点的重要见解。这项研究不仅为实践者提供了一个实用的参考，也为指导未来模仿学习的研究奠定了基础。 

---
# Towards Fusing Point Cloud and Visual Representations for Imitation Learning 

**Title (ZH)**: 面向点云和视觉表示融合的imitation learning研究 

**Authors**: Atalay Donat, Xiaogang Jia, Xi Huang, Aleksandar Taranovic, Denis Blessing, Ge Li, Hongyi Zhou, Hanyi Zhang, Rudolf Lioutikov, Gerhard Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2502.12320)  

**Abstract**: Learning for manipulation requires using policies that have access to rich sensory information such as point clouds or RGB images. Point clouds efficiently capture geometric structures, making them essential for manipulation tasks in imitation learning. In contrast, RGB images provide rich texture and semantic information that can be crucial for certain tasks. Existing approaches for fusing both modalities assign 2D image features to point clouds. However, such approaches often lose global contextual information from the original images. In this work, we propose a novel imitation learning method that effectively combines the strengths of both point cloud and RGB modalities. Our method conditions the point-cloud encoder on global and local image tokens using adaptive layer norm conditioning, leveraging the beneficial properties of both modalities. Through extensive experiments on the challenging RoboCasa benchmark, we demonstrate the limitations of relying on either modality alone and show that our method achieves state-of-the-art performance across all tasks. 

**Abstract (ZH)**: 学习用于操作任务需要使用能够访问丰富感官信息（如点云或RGB图像）的策略。点云有效地捕获几何结构，使其成为模仿学习中操作任务的关键。相比之下，RGB图像提供了丰富的纹理和语义信息，对于某些任务至关重要。现有的方法通过将2D图像特征分配给点云来融合这两种模态，但这些方法往往会从原始图像中丢失全局上下文信息。在本工作中，我们提出了一种新颖的模仿学习方法，该方法有效地结合了点云和RGB模态的优势。我们的方法通过自适应层规范条件使得点云编码器依赖于全局和局部图像标记，利用这两种模态的有益特性。通过在具有挑战性的RoboCasa基准上的广泛实验，我们展示了仅依赖任一模态的局限性，并证明了我们的方法在所有任务中都达到了最先进的性能。 

---
# RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning 

**Title (ZH)**: RAD: 通过大规模3DGS为基础的强化学习训练端到端驾驶策略 

**Authors**: Hao Gao, Shaoyu Chen, Bo Jiang, Bencheng Liao, Yiang Shi, Xiaoyang Guo, Yuechuan Pu, Haoran Yin, Xiangyu Li, Xinbang Zhang, Ying Zhang, Wenyu Liu, Qian Zhang, Xinggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13144)  

**Abstract**: Existing end-to-end autonomous driving (AD) algorithms typically follow the Imitation Learning (IL) paradigm, which faces challenges such as causal confusion and the open-loop gap. In this work, we establish a 3DGS-based closed-loop Reinforcement Learning (RL) training paradigm. By leveraging 3DGS techniques, we construct a photorealistic digital replica of the real physical world, enabling the AD policy to extensively explore the state space and learn to handle out-of-distribution scenarios through large-scale trial and error. To enhance safety, we design specialized rewards that guide the policy to effectively respond to safety-critical events and understand real-world causal relationships. For better alignment with human driving behavior, IL is incorporated into RL training as a regularization term. We introduce a closed-loop evaluation benchmark consisting of diverse, previously unseen 3DGS environments. Compared to IL-based methods, RAD achieves stronger performance in most closed-loop metrics, especially 3x lower collision rate. Abundant closed-loop results are presented at this https URL. 

**Abstract (ZH)**: 基于3DGS的闭环强化学习训练 paradigm 在自主驾驶中的应用：克服imitation learning的挑战并实现更强的安全性和泛化能力 

---
# Magma: A Foundation Model for Multimodal AI Agents 

**Title (ZH)**: Magma：多模态AI代理的基础模型 

**Authors**: Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, Yuquan Deng, Lars Liden, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.13130)  

**Abstract**: We present Magma, a foundation model that serves multimodal AI agentic tasks in both the digital and physical worlds. Magma is a significant extension of vision-language (VL) models in that it not only retains the VL understanding ability (verbal intelligence) of the latter, but is also equipped with the ability to plan and act in the visual-spatial world (spatial-temporal intelligence) and complete agentic tasks ranging from UI navigation to robot manipulation. To endow the agentic capabilities, Magma is pretrained on large amounts of heterogeneous datasets spanning from images, videos to robotics data, where the actionable visual objects (e.g., clickable buttons in GUI) in images are labeled by Set-of-Mark (SoM) for action grounding, and the object movements (e.g., the trace of human hands or robotic arms) in videos are labeled by Trace-of-Mark (ToM) for action planning. Extensive experiments show that SoM and ToM reach great synergy and facilitate the acquisition of spatial-temporal intelligence for our Magma model, which is fundamental to a wide range of tasks as shown in Fig.1. In particular, Magma creates new state-of-the-art results on UI navigation and robotic manipulation tasks, outperforming previous models that are specifically tailored to these tasks. On image and video-related multimodal tasks, Magma also compares favorably to popular large multimodal models that are trained on much larger datasets. We make our model and code public for reproducibility at this https URL. 

**Abstract (ZH)**: 我们提出Magma，一个服务于数字世界和物理世界多模态人工智能自主任务的基座模型。Magma是对视觉语言（VL）模型的重大扩展，不仅保留了后者在语言理解方面的能力（言语智能），还具备在视觉空间世界中规划和行动的能力（时空智能），能够完成从UI导航到机器人操作等一系列自主任务。为了赋予这些自主能力，Magma在图像、视频以及机器人数据等多种异构数据集上进行预训练，其中图像中的可操作视觉对象（例如GUI中的可点击按钮）通过Set-of-Mark（SoM）进行标注以实现动作定位，视频中的物体运动（例如人类手部或机器人手臂的轨迹）通过Trace-of-Mark（ToM）进行标注以实现动作规划。广泛的实验表明，SoM和ToM达到了很好的协同效应，促进了Magma模型时空智能的获取，这对其它多种任务至关重要，如图1所示。特别是，Magma在UI导航和机器人操作任务上创造了新的最先进技术结果，超越了专为这些任务设计的先前模型。在图像和视频相关的多模态任务上，Magma也优于在更大数据集上训练的流行多模态模型。我们将在以下网址公开我们的模型和代码以确保可再现性：[此 https URL]。 

---
# ExoKit: A Toolkit for Rapid Prototyping of Interactions for Arm-based Exoskeletons 

**Title (ZH)**: ExoKit：基于臂部外骨骼交互的快速原型开发工具-kit 

**Authors**: Marie Muehlhaus, Alexander Liggesmeyer, Jürgen Steimle  

**Link**: [PDF](https://arxiv.org/pdf/2502.12747)  

**Abstract**: Exoskeletons open up a unique interaction space that seamlessly integrates users' body movements with robotic actuation. Despite its potential, human-exoskeleton interaction remains an underexplored area in HCI, largely due to the lack of accessible prototyping tools that enable designers to easily develop exoskeleton designs and customized interactive behaviors. We present ExoKit, a do-it-yourself toolkit for rapid prototyping of low-fidelity, functional exoskeletons targeted at novice roboticists. ExoKit includes modular hardware components for sensing and actuating shoulder and elbow joints, which are easy to fabricate and (re)configure for customized functionality and wearability. To simplify the programming of interactive behaviors, we propose functional abstractions that encapsulate high-level human-exoskeleton interactions. These can be readily accessed either through ExoKit's command-line or graphical user interface, a Processing library, or microcontroller firmware, each targeted at different experience levels. Findings from implemented application cases and two usage studies demonstrate the versatility and accessibility of ExoKit for early-stage interaction design. 

**Abstract (ZH)**: 外骨骼开辟了一个独特的交互空间，能够无缝整合用户的肢体运动与机器人驱动。尽管具有潜在价值，但人类-外骨骼交互在HCI领域仍然是一个未充分探索的领域，主要原因是缺乏易于使用的原型工具，使得设计师难以轻松开发外骨骼设计及其定制交互行为。我们介绍了ExoKit，这是一个面向新手机器人工程师的快速原型制作工具包，用于快速制作低保真度功能外骨骼。ExoKit包括用于感知和驱动肩关节和肘关节的模块化硬件组件，这些组件易于制造和重新配置，以实现定制的功能性和穿戴性。为了简化交互行为的编程，我们提出了功能抽象，可以封装高层的人类-外骨骼交互。这些功能抽象可以通过ExoKit的命令行界面或图形用户界面、Processing库或微控制器固件访问，针对不同的经验水平。通过实现的应用案例和两次使用研究的发现，表明ExoKit可以为早期交互设计提供灵活性和易用性。 

---
# Introducing ROADS: A Systematic Comparison of Remote Control Interaction Concepts for Automated Vehicles at Road Works 

**Title (ZH)**: 引入ROADS：针对道路施工的自动驾驶车辆远程控制交互概念系统比较 

**Authors**: Mark Colley, Jonathan Westhauser, Jonas Andersson, Alexander G. Mirnig, Enrico Rukzio  

**Link**: [PDF](https://arxiv.org/pdf/2502.12680)  

**Abstract**: As vehicle automation technology continues to mature, there is a necessity for robust remote monitoring and intervention features. These are essential for intervening during vehicle malfunctions, challenging road conditions, or in areas that are difficult to navigate. This evolution in the role of the human operator - from a constant driver to an intermittent teleoperator - necessitates the development of suitable interaction interfaces. While some interfaces were suggested, a comparative study is missing. We designed, implemented, and evaluated three interaction concepts (path planning, trajectory guidance, and waypoint guidance) with up to four concurrent requests of automated vehicles in a within-subjects study with N=23 participants. The results showed a clear preference for the path planning concept. It also led to the highest usability but lower satisfaction. With trajectory guidance, the fewest requests were resolved. The study's findings contribute to the ongoing development of HMIs focused on the remote assistance of automated vehicles. 

**Abstract (ZH)**: 随着车辆自动化技术的不断成熟， robust 的远程监控和干预功能变得必不可少。这些功能对于在车辆故障、复杂道路条件或难以导航的区域进行干预至关重要。这一角色转变——从不间断的驾驶员到间歇的远程操作员——促使开发合适的交互界面。虽然已有一些界面建议，但缺乏对比研究。我们设计、实现并评估了三种交互概念（路径规划、轨迹引导和航点引导），并在包含23名参与者的单项研究中处理多达四个并发的自动化车辆请求。研究结果显示，路径规划概念最受欢迎，使用便利性最高，但满意度较低。在轨迹引导中，解决的请求最少。研究发现为专注于自动化车辆远程协助的人机界面的进一步发展做出了贡献。 

---
# Predicate Hierarchies Improve Few-Shot State Classification 

**Title (ZH)**: 谓词层级结构改善少样本状态分类 

**Authors**: Emily Jin, Joy Hsu, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12481)  

**Abstract**: State classification of objects and their relations is core to many long-horizon tasks, particularly in robot planning and manipulation. However, the combinatorial explosion of possible object-predicate combinations, coupled with the need to adapt to novel real-world environments, makes it a desideratum for state classification models to generalize to novel queries with few examples. To this end, we propose PHIER, which leverages predicate hierarchies to generalize effectively in few-shot scenarios. PHIER uses an object-centric scene encoder, self-supervised losses that infer semantic relations between predicates, and a hyperbolic distance metric that captures hierarchical structure; it learns a structured latent space of image-predicate pairs that guides reasoning over state classification queries. We evaluate PHIER in the CALVIN and BEHAVIOR robotic environments and show that PHIER significantly outperforms existing methods in few-shot, out-of-distribution state classification, and demonstrates strong zero- and few-shot generalization from simulated to real-world tasks. Our results demonstrate that leveraging predicate hierarchies improves performance on state classification tasks with limited data. 

**Abstract (ZH)**: 基于谓词层次结构的状态分类及其关系分类对于长期任务至关重要，特别是在机器人规划和操作中。为实现这一目标，我们提出了PHIER，它利用谓词层次结构在少样本场景中有效泛化。PHIER采用以对象为中心的场景编码器、自监督损失来推断谓词之间的语义关系、以及捕捉层次结构的超曲面距离度量；它学习具有图像-谓词对结构化潜在空间，以指导状态分类查询的推理。我们在CALVIN和BEHAVIOR机器人环境中评估了PHIER，结果表明PHIER在少样本、分布外状态分类中显著优于现有方法，并展示了从模拟环境到真实世界的任务中的强零样本和少样本泛化能力。我们的结果表明，利用谓词层次结构可以提高在数据有限的情况下状态分类任务的表现。 

---
# AI-Augmented Metamorphic Testing for Comprehensive Validation of Autonomous Vehicles 

**Title (ZH)**: 基于AI增强的元变换测试对于自主车辆全面验证的研究 

**Authors**: Tony Zhang, Burak Kantarci, Umair Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2502.12208)  

**Abstract**: Self-driving cars have the potential to revolutionize transportation, but ensuring their safety remains a significant challenge. These systems must navigate a variety of unexpected scenarios on the road, and their complexity poses substantial difficulties for thorough testing. Conventional testing methodologies face critical limitations, including the oracle problem determining whether the systems behavior is correct and the inability to exhaustively recreate a range of situations a self-driving car may encounter. While Metamorphic Testing (MT) offers a partial solution to these challenges, its application is often limited by simplistic modifications to test scenarios. In this position paper, we propose enhancing MT by integrating AI-driven image generation tools, such as Stable Diffusion, to improve testing methodologies. These tools can generate nuanced variations of driving scenarios within the operational design domain (ODD)for example, altering weather conditions, modifying environmental elements, or adjusting lane markings while preserving the critical features necessary for system evaluation. This approach enables reproducible testing, efficient reuse of test criteria, and comprehensive evaluation of a self-driving systems performance across diverse scenarios, thereby addressing key gaps in current testing practices. 

**Abstract (ZH)**: 自动驾驶汽车有潜力革新交通方式，但确保其安全仍面临重大挑战。这些系统必须应对道路上的各种意外场景，其复杂性给全面测试带来了巨大困难。传统测试方法存在关键限制，包括确定系统行为是否正确的奥里卡问题以及无法彻底重现自动驾驶汽车可能遇到的各种情况。尽管元型测试（MT）为解决这些挑战提供了一部分解决方案，但其应用往往受限于对测试场景的简单修改。在本文中，我们提议通过集成如Stable Diffusion等AI驱动的图像生成工具来增强MT，以改进测试方法。这些工具可以在操作设计域（ODD）中生成驾驶场景的精细变体，例如改变天气条件、修改环境元素或调整车道标记，同时保留系统评估所需的关键特征。这种方法能够实现可重复的测试、高效地重用测试标准并全面评估自动驾驶系统在不同场景中的性能，从而填补当前测试实践的关键缺口。 

---
# AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors 

**Title (ZH)**: AnyTouch: 学习多传感器视触觉统一静态-动态表示 

**Authors**: Ruoxuan Feng, Jiangyu Hu, Wenke Xia, Tianci Gao, Ao Shen, Yuhao Sun, Bin Fang, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12191)  

**Abstract**: Visuo-tactile sensors aim to emulate human tactile perception, enabling robots to precisely understand and manipulate objects. Over time, numerous meticulously designed visuo-tactile sensors have been integrated into robotic systems, aiding in completing various tasks. However, the distinct data characteristics of these low-standardized visuo-tactile sensors hinder the establishment of a powerful tactile perception system. We consider that the key to addressing this issue lies in learning unified multi-sensor representations, thereby integrating the sensors and promoting tactile knowledge transfer between them. To achieve unified representation of this nature, we introduce TacQuad, an aligned multi-modal multi-sensor tactile dataset from four different visuo-tactile sensors, which enables the explicit integration of various sensors. Recognizing that humans perceive the physical environment by acquiring diverse tactile information such as texture and pressure changes, we further propose to learn unified multi-sensor representations from both static and dynamic perspectives. By integrating tactile images and videos, we present AnyTouch, a unified static-dynamic multi-sensor representation learning framework with a multi-level structure, aimed at both enhancing comprehensive perceptual abilities and enabling effective cross-sensor transfer. This multi-level architecture captures pixel-level details from tactile data via masked modeling and enhances perception and transferability by learning semantic-level sensor-agnostic features through multi-modal alignment and cross-sensor matching. We provide a comprehensive analysis of multi-sensor transferability, and validate our method on various datasets and in the real-world pouring task. Experimental results show that our method outperforms existing methods, exhibits outstanding static and dynamic perception capabilities across various sensors. 

**Abstract (ZH)**: 视觉-触觉传感器旨在模拟人类的触觉感知，使机器人能够精确地理解和操作物体。随着时间的推移，众多精心设计的视觉-触觉传感器被集成到机器人系统中，以帮助完成各种任务。然而，这些低标准化的视觉-触觉传感器的特殊数据特征阻碍了强大的触觉感知系统的建立。我们认为解决这一问题的关键在于学习统一的多传感器表示，从而整合传感器并促进它们之间的触觉知识转移。为了实现这种统一表示，我们引入了TacQuad，一个来自四种不同视觉-触觉传感器的对齐多模态多传感器触觉数据集，它允许显式地集成各种传感器。鉴于人类通过获取诸如纹理和压力变化等多种触觉信息来感知物理环境，我们进一步提出从静态和动态两个方面学习统一的多传感器表示。通过整合触觉图像和视频，我们提出了AnyTouch，一个具有多级结构的统一静态-动态多传感器表示学习框架，旨在提高综合感知能力和促进有效的跨传感器转移。该多级架构通过掩蔽建模捕捉触觉数据的像素级细节，并通过多模态对齐和跨传感器匹配学习语义级传感器无损特征，从而增强感知能力和转移性。我们全面分析了多传感器转移能力，并在多种数据集以及现实世界的倒水任务中验证了我们的方法。实验结果表明，我们的方法优于现有方法，在各种传感器上的静态和动态感知能力都表现出色。 

---
