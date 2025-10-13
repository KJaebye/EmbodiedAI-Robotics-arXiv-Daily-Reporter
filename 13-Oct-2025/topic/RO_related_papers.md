# Autonomous Soft Robotic Guidewire Navigation via Imitation Learning 

**Title (ZH)**: 自主软机器人导丝导航通过类比学习 

**Authors**: Noah Barnes, Ji Woong Kim, Lingyun Di, Hannah Qu, Anuruddha Bhattacharjee, Miroslaw Janowski, Dheeraj Gandhi, Bailey Felix, Shaopeng Jiang, Olivia Young, Mark Fuge, Ryan D. Sochol, Jeremy D. Brown, Axel Krieger  

**Link**: [PDF](https://arxiv.org/pdf/2510.09497)  

**Abstract**: In endovascular surgery, endovascular interventionists push a thin tube called a catheter, guided by a thin wire to a treatment site inside the patient's blood vessels to treat various conditions such as blood clots, aneurysms, and malformations. Guidewires with robotic tips can enhance maneuverability, but they present challenges in modeling and control. Automation of soft robotic guidewire navigation has the potential to overcome these challenges, increasing the precision and safety of endovascular navigation. In other surgical domains, end-to-end imitation learning has shown promising results. Thus, we develop a transformer-based imitation learning framework with goal conditioning, relative action outputs, and automatic contrast dye injections to enable generalizable soft robotic guidewire navigation in an aneurysm targeting task. We train the model on 36 different modular bifurcated geometries, generating 647 total demonstrations under simulated fluoroscopy, and evaluate it on three previously unseen vascular geometries. The model can autonomously drive the tip of the robot to the aneurysm location with a success rate of 83% on the unseen geometries, outperforming several baselines. In addition, we present ablation and baseline studies to evaluate the effectiveness of each design and data collection choice. Project website: this https URL 

**Abstract (ZH)**: 血管内手术中，血管内介入医生通过一根由细线引导的细管（导管）推送至患者血管内的治疗部位，以治疗血栓、动脉瘤和畸形等多种疾病。具有机械臂末端执行器的导丝可以增强操作灵活性，但同时也带来了建模和控制上的挑战。软体机器人导丝导航的自动化有可能克服这些挑战，提高血管内导航的精确性和安全性。在其他手术领域，端到端的模仿学习已显示出积极的结果。因此，我们开发了一种基于变换器的模仿学习框架，该框架具有目标条件、相对动作输出和自动对比度造影剂注射功能，以实现针对动脉瘤目标任务的一般化软体机器人导丝导航。我们使用36种不同的模块化分叉几何结构训练模型，在模拟透视成像下生成了共计647个演示，并在三种以前未见过的血管几何结构上进行评估。该模型能够在未见过的几何结构上自主将机器人导丝的末端驱动到动脉瘤位置，成功率高达83%，优于多个基线方法。此外，我们还进行了消融和基线研究，以评估每个设计和数据收集选择的有效性。项目网站：this https URL 

---
# Glovity: Learning Dexterous Contact-Rich Manipulation via Spatial Wrench Feedback Teleoperation System 

**Title (ZH)**: Glovity: 通过空间力矩反馈遥操作系统学习多接触灵活操作 

**Authors**: Yuyang Gao, Haofei Ma, Pai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09229)  

**Abstract**: We present Glovity, a novel, low-cost wearable teleoperation system that integrates a spatial wrench (force-torque) feedback device with a haptic glove featuring fingertip Hall sensor calibration, enabling feedback-rich dexterous manipulation. Glovity addresses key challenges in contact-rich tasks by providing intuitive wrench and tactile feedback, while overcoming embodiment gaps through precise retargeting. User studies demonstrate significant improvements: wrench feedback boosts success rates in book-flipping tasks from 48% to 78% and reduces completion time by 25%, while fingertip calibration enhances thin-object grasping success significantly compared to commercial glove. Furthermore, incorporating wrench signals into imitation learning (via DP-R3M) achieves high success rate in novel contact-rich scenarios, such as adaptive page flipping and force-aware handovers. All hardware designs, software will be open-sourced. Project website: this https URL 

**Abstract (ZH)**: Glovity：一种低成本的集成空间 wrench 反馈装置的穿戴式远程操作系统，配备经过指端霍尔传感器校准的触感手套，实现丰富的灵巧操控反馈。 

---
# PLEXUS Hand: Lightweight Four-Motor Prosthetic Hand Enabling Precision-Lateral Dexterous Manipulation 

**Title (ZH)**: PLEXUS 手: 轻量化四电机假手实现精确-侧向灵巧操作 

**Authors**: Yuki Kuroda, Tomoya Takahashi, Cristian C Beltran-Hernandez, Masashi Hamaya, Kazutoshi Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2510.09209)  

**Abstract**: Electric prosthetic hands should be lightweight to decrease the burden on the user, shaped like human hands for cosmetic purposes, and have motors inside to protect them from damage and dirt. In addition to the ability to perform daily activities, these features are essential for everyday use of the hand. In-hand manipulation is necessary to perform daily activities such as transitioning between different postures, particularly through rotational movements, such as reorienting cards before slot insertion and operating tools such as screwdrivers. However, currently used electric prosthetic hands only achieve static grasp postures, and existing manipulation approaches require either many motors, which makes the prosthesis heavy for daily use in the hand, or complex mechanisms that demand a large internal space and force external motor placement, complicating attachment and exposing the components to damage. Alternatively, we combine a single-axis thumb and optimized thumb positioning to achieve basic posture and in-hand manipulation, that is, the reorientation between precision and lateral grasps, using only four motors in a lightweight (311 g) prosthetic hand. Experimental validation using primitive objects of various widths (5-30 mm) and shapes (cylinders and prisms) resulted in success rates of 90-100% for reorientation tasks. The hand performed seal stamping and USB device insertion, as well as rotation to operate a screwdriver. 

**Abstract (ZH)**: 电假手应当轻便以减轻使用者的负担，外观接近人类手掌以满足美化需求，并内含电机以防止损坏和脏污。除了能够执行日常活动外，这些特点对于手的日常使用至关重要。内部操纵对于执行日常活动如在不同姿势之间转换，尤其是通过旋转运动，比如在插入插槽前重新定位卡片和操作螺丝刀等工具是必要的。然而，目前使用的电假手只能实现静态握持姿势，现有的操纵方法要么需要多台电机，这使得假手在日常使用中变得沉重，要么需要复杂的机构，需要较大的内部空间并要求外部电机定位，从而复杂化安装过程并使组件暴露在外受损伤。相反，我们通过采用单轴拇指和优化的拇指定位，仅使用四个电机在轻量化（311克）的假手中实现了基本的姿势和内部操纵，即在精密握和侧向握之间的重新定位。实验验证使用不同宽度（5-30毫米）和形状（圆柱和棱柱）的原始物体，重新定位任务的成功率为90-100%。该手可以进行密封印章、USB设备插入，并且可以旋转操作螺丝刀。 

---
# Flow-Opt: Scalable Centralized Multi-Robot Trajectory Optimization with Flow Matching and Differentiable Optimization 

**Title (ZH)**: Flow-Opt: 基于流匹配和可微优化的可扩展集中式多机器人轨迹优化 

**Authors**: Simon Idoko, Arun Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.09204)  

**Abstract**: Centralized trajectory optimization in the joint space of multiple robots allows access to a larger feasible space that can result in smoother trajectories, especially while planning in tight spaces. Unfortunately, it is often computationally intractable beyond a very small swarm size. In this paper, we propose Flow-Opt, a learning-based approach towards improving the computational tractability of centralized multi-robot trajectory optimization. Specifically, we reduce the problem to first learning a generative model to sample different candidate trajectories and then using a learned Safety-Filter(SF) to ensure fast inference-time constraint satisfaction. We propose a flow-matching model with a diffusion transformer (DiT) augmented with permutation invariant robot position and map encoders as the generative model. We develop a custom solver for our SF and equip it with a neural network that predicts context-specific initialization. The initialization network is trained in a self-supervised manner, taking advantage of the differentiability of the SF solver. We advance the state-of-the-art in the following respects. First, we show that we can generate trajectories of tens of robots in cluttered environments in a few tens of milliseconds. This is several times faster than existing centralized optimization approaches. Moreover, our approach also generates smoother trajectories orders of magnitude faster than competing baselines based on diffusion models. Second, each component of our approach can be batched, allowing us to solve a few tens of problem instances in a fraction of a second. We believe this is a first such result; no existing approach provides such capabilities. Finally, our approach can generate a diverse set of trajectories between a given set of start and goal locations, which can capture different collision-avoidance behaviors. 

**Abstract (ZH)**: 基于流匹配的学习方法Flow-Opt在多机器人联合空间中的集中轨迹优化计算可逾性改进 

---
# Robust Visual Teach-and-Repeat Navigation with Flexible Topo-metric Graph Map Representation 

**Title (ZH)**: 鲁棒的视觉指导重复导航与灵活的拓扑-度量图测绘表示 

**Authors**: Jikai Wang, Yunqi Cheng, Kezhi Wang, Zonghai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.09089)  

**Abstract**: Visual Teach-and-Repeat Navigation is a direct solution for mobile robot to be deployed in unknown environments. However, robust trajectory repeat navigation still remains challenged due to environmental changing and dynamic objects. In this paper, we propose a novel visual teach-and-repeat navigation system, which consists of a flexible map representation, robust map matching and a map-less local navigation module. During the teaching process, the recorded keyframes are formulated as a topo-metric graph and each node can be further extended to save new observations. Such representation also alleviates the requirement of globally consistent mapping. To enhance the place recognition performance during repeating process, instead of using frame-to-frame matching, we firstly implement keyframe clustering to aggregate similar connected keyframes into local map and perform place recognition based on visual frame-tolocal map matching strategy. To promote the local goal persistent tracking performance, a long-term goal management algorithm is constructed, which can avoid the robot getting lost due to environmental changes or obstacle occlusion. To achieve the goal without map, a local trajectory-control candidate optimization algorithm is proposed. Extensively experiments are conducted on our mobile platform. The results demonstrate that our system is superior to the baselines in terms of robustness and effectiveness. 

**Abstract (ZH)**: 视觉教学与重复导航是一种直接解决方案，用于在未知环境中部署移动机器人。然而，由于环境变化和动态物体的存在，稳健的轨迹重复导航仍面临挑战。本文提出了一种新颖的视觉教学与重复导航系统，该系统由灵活的地图表示、稳健的地图匹配和无地图的局部导航模块组成。在教学过程中，记录的关键帧被形式化为拓扑图，每个节点可以进一步扩展以保存新观测值。这种表示也有助于减轻全局一致映射的需求。为了在重复过程中增强位置识别性能，我们首先实现关键帧聚类，将相似的关键帧聚合成局部地图，并基于视觉帧与局部地图匹配策略进行位置识别。为了促进局部目标持久跟踪性能，构建了一个长期目标管理算法，该算法可以避免由于环境变化或障碍物遮挡导致的机器人迷路。为了在无地图的情况下实现目标，提出了一种局部轨迹控制候选优化算法。在我们移动平台上进行了广泛实验。结果表明，我们的系统在稳健性和有效性方面优于基线系统。 

---
# Direct Data-Driven Predictive Control for a Three-dimensional Cable-Driven Soft Robotic Arm 

**Title (ZH)**: 直接数据驱动预测控制 for 三维缆索驱动软机器人臂 

**Authors**: Cheng Ouyang, Moeen Ul Islam, Dong Chen, Kaixiang Zhang, Zhaojian Li, Xiaobo Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.08953)  

**Abstract**: Soft robots offer significant advantages in safety and adaptability, yet achieving precise and dynamic control remains a major challenge due to their inherently complex and nonlinear dynamics. Recently, Data-enabled Predictive Control (DeePC) has emerged as a promising model-free approach that bypasses explicit system identification by directly leveraging input-output data. While DeePC has shown success in other domains, its application to soft robots remains underexplored, particularly for three-dimensional (3D) soft robotic systems. This paper addresses this gap by developing and experimentally validating an effective DeePC framework on a 3D, cable-driven soft arm. Specifically, we design and fabricate a soft robotic arm with a thick tubing backbone for stability, a dense silicone body with large cavities for strength and flexibility, and rigid endcaps for secure termination. Using this platform, we implement DeePC with singular value decomposition (SVD)-based dimension reduction for two key control tasks: fixed-point regulation and trajectory tracking in 3D space. Comparative experiments with a baseline model-based controller demonstrate DeePC's superior accuracy, robustness, and adaptability, highlighting its potential as a practical solution for dynamic control of soft robots. 

**Abstract (ZH)**: 软体机器人在安全性和适应性方面提供了显著优势，但由于其固有的复杂非线性动力学，实现精确和动态控制仍是一项重大挑战。最近，数据驱动预测控制（DeePC）作为一种无需进行显式系统辨识的有希望的方法，通过直接利用输入输出数据得到了发展。尽管DeePC在其他领域已显示出成功，但其在软体机器人中的应用仍待探索，尤其是在三维（3D）软体机器人系统中。本文通过在3D线缆驱动软臂平台上开发并实验证实了有效的DeePC框架，解决了这一问题。具体来说，我们设计并制作了一款具有稳定性的厚管骨架、具有强大柔性的多腔硅胶体和安全终接的刚性端帽的软体机器人手臂。利用该平台，我们实现了基于奇异值分解（SVD）的降维DeePC，用于两个关键控制任务：定点调节和三维空间中的轨迹跟踪。与基于模型的基线控制器的对比实验表明，DeePC在精确性、鲁棒性和适应性方面均有优越表现，突显了其作为软体机器人动态控制实用解决方案的潜力。 

---
# Online IMU-odometer Calibration using GNSS Measurements for Autonomous Ground Vehicle Localization 

**Title (ZH)**: 基于GNSS测测量的在线IMU-里程计标定方法及其在自主地面车辆定位中的应用 

**Authors**: Baoshan Song, Xiao Xia, Penggao Yan, Yihan Zhong, Weisong Wen, Li-Ta Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08880)  

**Abstract**: Accurate calibration of intrinsic (odometer scaling factors) and extrinsic parameters (IMU-odometer translation and rotation) is essential for autonomous ground vehicle localization. Existing GNSS-aided approaches often rely on positioning results or raw measurements without ambiguity resolution, and their observability properties remain underexplored. This paper proposes a tightly coupled online calibration method that fuses IMU, odometer, and raw GNSS measurements (pseudo-range, carrier-phase, and Doppler) within an extendable factor graph optimization (FGO) framework, incorporating outlier mitigation and ambiguity resolution. Observability analysis reveals that two horizontal translation and three rotation parameters are observable under general motion, while vertical translation remains unobservable. Simulation and real-world experiments demonstrate superior calibration and localization performance over state-of-the-art loosely coupled methods. Specifically, the IMU-odometer positioning using our calibrated parameters achieves the absolute maximum error of 17.75 m while the one of LC method is 61.51 m, achieving up to 71.14 percent improvement. To foster further research, we also release the first open-source dataset that combines IMU, 2D odometer, and raw GNSS measurements from both rover and base stations. 

**Abstract (ZH)**: 精确校准固有参数（里程计缩放因子）和外在参数（IMU-里程计平移和旋转）对于自主地面车辆定位至关重要。现有基于GNSS辅助的方法往往依赖于定位结果或原始测量值而缺乏模糊性解析，其可观测性特性尚未充分探索。本文提出了一种紧耦合的在线校准方法，将IMU、里程计和原始GNSS测量（伪距离、载波相位和多普勒）融合在可扩展的因子图优化（FGO）框架内，同时整合了离群值剔除和模糊性解析。可观测性分析表明，在一般运动情况下，可以观测到两个水平平移和三个旋转参数，而垂直平移参数不可观测。仿真和现实世界实验表明，与最先进的松耦合方法相比，该校准方法和定位性能更优秀。具体而言，使用校准参数的IMU-里程计定位的绝对最大误差为17.75米，而松耦合方法（LC方法）的误差为61.51米，提高了71.14%。为促进进一步研究，我们还发布了第一个综合IMU、2D里程计和原始GNSS测量数据的开源数据集，数据来自移动站和基准站。 

---
# Point and Go: Intuitive Reference Frame Reallocation in Mode Switching for Assistive Robotics 

**Title (ZH)**: 点即移：辅助机器人模式切换中的直观参考坐标重分配 

**Authors**: A. Wang, C. Jiang, M. Przystupa, J. Valentine, M. Jagersand  

**Link**: [PDF](https://arxiv.org/pdf/2510.08753)  

**Abstract**: Operating high degree of freedom robots can be difficult for users of wheelchair mounted robotic manipulators. Mode switching in Cartesian space has several drawbacks such as unintuitive control reference frames, separate translation and orientation control, and limited movement capabilities that hinder performance. We propose Point and Go mode switching, which reallocates the Cartesian mode switching reference frames into a more intuitive action space comprised of new translation and rotation modes. We use a novel sweeping motion to point the gripper, which defines the new translation axis along the robot base frame's horizontal plane. This creates an intuitive `point and go' translation mode that allows the user to easily perform complex, human-like movements without switching control modes. The system's rotation mode combines position control with a refined end-effector oriented frame that provides precise and consistent robot actions in various end-effector poses. We verified its effectiveness through initial experiments, followed by a three-task user study that compared our method to Cartesian mode switching and a state of the art learning method. Results show that Point and Go mode switching reduced completion times by 31\%, pauses by 41\%, and mode switches by 33\%, while receiving significantly favorable responses in user surveys. 

**Abstract (ZH)**: 基于指定点动作的轮椅载 manipulator 高自由度机器人模式切换方法 

---
# Differential Analysis of Pseudo Haptic Feedback: Novel Comparative Study of Visual and Auditory Cue Integration for Psychophysical Evaluation 

**Title (ZH)**: 伪触觉反馈的差异分析：视觉和听觉提示整合的新型比较研究用于心理物理评估 

**Authors**: Nishant Gautam, Somya Sharma, Peter Corcoran, Kaspar Althoefer  

**Link**: [PDF](https://arxiv.org/pdf/2510.09570)  

**Abstract**: Pseudo-haptics exploit carefully crafted visual or auditory cues to trick the brain into "feeling" forces that are never physically applied, offering a low-cost alternative to traditional haptic hardware. Here, we present a comparative psychophysical study that quantifies how visual and auditory stimuli combine to evoke pseudo-haptic pressure sensations on a commodity tablet. Using a Unity-based Rollball game, participants (n = 4) guided a virtual ball across three textured terrains while their finger forces were captured in real time with a Robotous RFT40 force-torque sensor. Each terrain was paired with a distinct rolling-sound profile spanning 440 Hz - 4.7 kHz, 440 Hz - 13.1 kHz, or 440 Hz - 8.9 kHz; crevice collisions triggered additional "knocking" bursts to heighten realism. Average tactile forces increased systematically with cue intensity: 0.40 N, 0.79 N and 0.88 N for visual-only trials and 0.41 N, 0.81 N and 0.90 N for audio-only trials on Terrains 1-3, respectively. Higher audio frequencies and denser visual textures both elicited stronger muscle activation, and their combination further reduced the force needed to perceive surface changes, confirming multisensory integration. These results demonstrate that consumer-grade isometric devices can reliably induce and measure graded pseudo-haptic feedback without specialized actuators, opening a path toward affordable rehabilitation tools, training simulators and assistive interfaces. 

**Abstract (ZH)**: 伪触觉利用精心设计的视觉或听觉提示欺骗大脑感受到从未实际施加的力，提供了一种传统触觉硬件的低成本替代方案。在此，我们进行了一项比较的心理物理研究，量化了视觉和听觉刺激如何在商用平板电脑上引发伪触觉压力感。通过一个基于Unity的Rollball游戏，参与者（n=4）引导虚拟球跨越三种不同的地形，同时使用Robotous RFT40力扭矩传感器实时捕捉手指力。每种地形配有一套独特的滚动声音谱，频率范围分别为440 Hz - 4.7 kHz、440 Hz - 13.1 kHz或440 Hz - 8.9 kHz；裂缝碰撞触发额外的“敲击”爆发以增强真实性。平均触觉力随提示强度系统增加：地形1-3的视觉仅试验分别为0.40 N、0.79 N和0.88 N，听觉仅试验分别为0.41 N、0.81 N和0.90 N。更高的音频频率和更密集的视觉纹理均引发了更强的肌肉激活，并且它们的结合进一步减少了感知表面变化所需的力，证实了多感官整合。研究结果表明，消费级等向性设备在没有专用执行器的情况下可以可靠地诱导和测量分级的伪触觉反馈，为经济实惠的康复工具、培训模拟器和辅助界面提供了途径。 

---
# Toggling stiffness via multistability 

**Title (ZH)**: 多稳态实现切换刚度 

**Authors**: Hugo de Souza Oliveira, Michele Curatolo, Renate Sachse, Edoardo Milana  

**Link**: [PDF](https://arxiv.org/pdf/2510.09511)  

**Abstract**: Mechanical metamaterials enable unconventional and programmable mechanical responses through structural design rather than material composition. In this work, we introduce a multistable mechanical metamaterial that exhibits a toggleable stiffness effect, where the effective shear stiffness switches discretely between stable configurations. The mechanical analysis of surrogate beam models of the unit cell reveal that this behavior originates from the rotation transmitted by the support beams to the curved beam, which governs the balance between bending and axial deformation. The stiffness ratio between the two states of the unit cell can be tuned by varying the slenderness of the support beams or by incorporating localized hinges that modulate rotational transfer. Experiments on 3D-printed prototypes validate the numerical predictions, confirming consistent stiffness toggling across different geometries. Finally, we demonstrate a monolithic soft clutch that leverages this effect to achieve programmable, stepwise stiffness modulation. This work establishes a design strategy for toggleable stiffness using multistable metamaterials, paving the way for adaptive, lightweight, and autonomous systems in soft robotics and smart structures. 

**Abstract (ZH)**: 机械 metamaterials 通过结构设计而非材料组成实现非传统和可编程的机械响应。在本工作中，我们介绍了一种多稳定机械 metamaterial，其表现出可切换的刚度效果，其中有效剪切刚度在稳定配置之间离散切换。单元结构的代理梁模型的机械分析表明，这种行为源于支撑梁传递给曲梁的旋转，这决定了弯曲变形和轴向变形之间的平衡。通过改变支撑梁的细长比或引入局部铰链来调节旋转传递，可以调整单元的刚度比。3D 打印原型实验验证了数值预测，确认了不同几何结构中的刚度切换一致性。最后，我们展示了利用此效应实现可编程步进刚度调节的单片软离合器。本工作确立了使用多稳定 metamaterial 实现可切换刚度的设计策略，为软机器人和智能结构的自适应、轻量化和自主系统开辟了道路。 

---
# Scalable Multi-Agent Path Finding using Collision-Aware Dynamic Alert Mask and a Hybrid Execution Strategy 

**Title (ZH)**: 基于碰撞意识动态警示掩码和混合执行策略的可扩展多智能体路径规划 

**Authors**: Bharath Muppasani, Ritirupa Dey, Biplav Srivastava, Vignesh Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2510.09469)  

**Abstract**: Multi-agent pathfinding (MAPF) remains a critical problem in robotics and autonomous systems, where agents must navigate shared spaces efficiently while avoiding conflicts. Traditional centralized algorithms that have global information, such as Conflict-Based Search (CBS), provide high-quality solutions but become computationally expensive in large-scale scenarios due to the combinatorial explosion of conflicts that need resolution. Conversely, distributed approaches that have local information, particularly learning-based methods, offer better scalability by operating with relaxed information availability, yet often at the cost of solution quality. To address these limitations, we propose a hybrid framework that combines decentralized path planning with a lightweight centralized coordinator. Our framework leverages reinforcement learning (RL) for decentralized planning, enabling agents to adapt their planning based on minimal, targeted alerts--such as static conflict-cell flags or brief conflict tracks--that are dynamically shared information from the central coordinator for effective conflict resolution. We empirically study the effect of the information available to an agent on its planning performance. Our approach reduces the inter-agent information sharing compared to fully centralized and distributed methods, while still consistently finding feasible, collision-free solutions--even in large-scale scenarios having higher agent counts. 

**Abstract (ZH)**: 多智能体路径规划（MAPF）仍然是机器人技术和自主系统中的一个关键问题，其中智能体必须在共享空间中高效导航并避免冲突。传统的集中式算法虽然具有全局信息，如冲突基础搜索（CBS），能够提供高质量的解决方案，但在大型场景中因需要解决的冲突组合爆炸而变得计算耗费巨大。相反，特别是基于学习的方法，具有局部信息的分布式方法通过以较宽松的信息可用性来操作，提供了更好的扩展性，但往往以解决方案质量为代价。为了解决这些局限性，我们提出了一种结合去中心化路径规划和轻量级集中协调器的混合框架。该框架利用强化学习（RL）进行去中心化规划，使智能体可以根据集中协调器动态共享的最小化和针对性的警报（例如静态冲突单元标志或简短冲突轨迹）进行适应性规划，从而有效解决冲突。我们实证研究了可用信息对智能体规划性能的影响。与完全集中式和分布式方法相比，我们的方法减少了智能体之间的信息共享，但仍能一致地找到可行且无碰撞的解决方案，即使在智能体数量较多的大型场景中也是如此。 

---
# Visual Anomaly Detection for Reliable Robotic Implantation of Flexible Microelectrode Array 

**Title (ZH)**: 柔性微电极阵列可靠植入的视觉异常检测 

**Authors**: Yitong Chen, Xinyao Xu, Ping Zhu, Xinyong Han, Fangbo Qin, Shan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09071)  

**Abstract**: Flexible microelectrode (FME) implantation into brain cortex is challenging due to the deformable fiber-like structure of FME probe and the interaction with critical bio-tissue. To ensure reliability and safety, the implantation process should be monitored carefully. This paper develops an image-based anomaly detection framework based on the microscopic cameras of the robotic FME implantation system. The unified framework is utilized at four checkpoints to check the micro-needle, FME probe, hooking result, and implantation point, respectively. Exploiting the existing object localization results, the aligned regions of interest (ROIs) are extracted from raw image and input to a pretrained vision transformer (ViT). Considering the task specifications, we propose a progressive granularity patch feature sampling method to address the sensitivity-tolerance trade-off issue at different locations. Moreover, we select a part of feature channels with higher signal-to-noise ratios from the raw general ViT features, to provide better descriptors for each specific scene. The effectiveness of the proposed methods is validated with the image datasets collected from our implantation system. 

**Abstract (ZH)**: 基于机器人柔性微电极植入系统微观摄像头的图像异常检测框架 

---
