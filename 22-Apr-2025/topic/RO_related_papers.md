# Cascade IPG Observer for Underwater Robot State Estimation 

**Title (ZH)**: 级联IPG观测器用于水下机器人状态估计 

**Authors**: Kaustubh Joshi, Tianchen Liu, Nikhil Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2504.15235)  

**Abstract**: This paper presents a novel cascade nonlinear observer framework for inertial state estimation. It tackles the problem of intermediate state estimation when external localization is unavailable or in the event of a sensor outage. The proposed observer comprises two nonlinear observers based on a recently developed iteratively preconditioned gradient descent (IPG) algorithm. It takes the inputs via an IMU preintegration model where the first observer is a quaternion-based IPG. The output for the first observer is the input for the second observer, estimating the velocity and, consequently, the position. The proposed observer is validated on a public underwater dataset and a real-world experiment using our robot platform. The estimation is compared with an extended Kalman filter (EKF) and an invariant extended Kalman filter (InEKF). Results demonstrate that our method outperforms these methods regarding better positional accuracy and lower variance. 

**Abstract (ZH)**: 本文提出了一种新型级联非线性观测量化框架用于惯性状态估计。该框架解决了在外置定位不可用或传感器故障时的中间状态估计问题。所提出的观测量化器包括两个基于 recently developed 迭代预条件梯度下降 (IPG) 算法的非线性观测量化器。观测量化器通过陀螺仪数据预积分模型接收输入，其中第一个观测量化器是基于四元数的 IPG。第一个观测量化器的输出作为第二个观测量化器的输入，估计速度并进而估计位置。所提出的观测量化器在公共水下数据集和使用我们机器人平台的真实世界实验中进行了验证。估计结果与扩展卡尔曼滤波器 (EKF) 和不变扩展卡尔曼滤波器 (InEKF) 进行了比较。结果表明，我们的方法在位置准确性方面优于这两种方法，并且具有更低的变异。 

---
# Immersive Teleoperation Framework for Locomanipulation Tasks 

**Title (ZH)**: 沉浸式远程操作框架用于局部操作任务 

**Authors**: Takuya Boehringer, Jonathan Embley-Riches, Karim Hammoud, Valerio Modugno, Dimitrios Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2504.15229)  

**Abstract**: Recent advancements in robotic loco-manipulation have leveraged Virtual Reality (VR) to enhance the precision and immersiveness of teleoperation systems, significantly outperforming traditional methods reliant on 2D camera feeds and joystick controls. Despite these advancements, challenges remain, particularly concerning user experience across different setups. This paper introduces a novel VR-based teleoperation framework designed for a robotic manipulator integrated onto a mobile platform. Central to our approach is the application of Gaussian splatting, a technique that abstracts the manipulable scene into a VR environment, thereby enabling more intuitive and immersive interactions. Users can navigate and manipulate within the virtual scene as if interacting with a real robot, enhancing both the engagement and efficacy of teleoperation tasks. An extensive user study validates our approach, demonstrating significant usability and efficiency improvements. Two-thirds (66%) of participants completed tasks faster, achieving an average time reduction of 43%. Additionally, 93% preferred the Gaussian Splat interface overall, with unanimous (100%) recommendations for future use, highlighting improvements in precision, responsiveness, and situational awareness. Finally, we demonstrate the effectiveness of our framework through real-world experiments in two distinct application scenarios, showcasing the practical capabilities and versatility of the Splat-based VR interface. 

**Abstract (ZH)**: 近期机器人移动操作领域的进展通过虚拟现实（VR）提高了远程操作系统的精确度和沉浸感，显著优于依赖2D相机馈送和操纵杆控制的传统方法。尽管取得了这些进展，仍存在挑战，尤其是在不同配置下的用户体验方面。本文提出了一种新型基于VR的远程操作框架，适用于集成在移动平台上的机器人 manipulator。我们方法的核心是应用高斯点积技术，将可操作场景抽象为VR环境，从而实现更具直观性和沉浸感的交互。用户可以像操作真实机器人一样导航和操控虚拟场景，增强远程操作任务的参与度和有效性。广泛用户的实验验证了我们方法的有效性，显示出显著的可用性和效率提升。66%的参与者完成任务的速度更快，平均时间减少了43%。此外，93%的参与者整体上更偏好高斯点积界面，并一致推荐用于未来使用，突显了在精确度、响应性和情境意识方面的改进。最后，我们通过两个不同应用场景的实际实验展示了基于点积的VR界面的有效性和灵活性。 

---
# A Genetic Fuzzy-Enabled Framework on Robotic Manipulation for In-Space Servicing 

**Title (ZH)**: 基于遗传模糊系统的太空服务机器人操作框架 

**Authors**: Nathan Steffen, Wilhelm Louw, Nicholas Ernest, Timothy Arnett, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15226)  

**Abstract**: Automation of robotic systems for servicing in cislunar space is becoming extremely important as the number of satellites in orbit increases. Safety is critical in performing satellite maintenance, so the control techniques utilized must be trusted in addition to being highly efficient. In this work, Genetic Fuzzy Trees are combined with the widely used LQR control scheme via Thales' TrUE AI Toolkit to create a trusted and efficient controller for a two-degree-of-freedom planar robotic manipulator that would theoretically be used to perform satellite maintenance. It was found that Genetic Fuzzy-LQR is 18.5% more performant than optimal LQR on average, and that it is incredibly robust to uncertainty. 

**Abstract (ZH)**: 基于Genetic Fuzzy Trees的LQR控制方案在cislunar空间卫星维护机器人系统中的应用研究 

---
# Automatic Generation of Aerobatic Flight in Complex Environments via Diffusion Models 

**Title (ZH)**: 通过扩散模型在复杂环境下自动生成空中飞行机动动作 

**Authors**: Yuhang Zhong, Anke Zhao, Tianyue Wu, Tingrui Zhang, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15138)  

**Abstract**: Performing striking aerobatic flight in complex environments demands manual designs of key maneuvers in advance, which is intricate and time-consuming as the horizon of the trajectory performed becomes long. This paper presents a novel framework that leverages diffusion models to automate and scale up aerobatic trajectory generation. Our key innovation is the decomposition of complex maneuvers into aerobatic primitives, which are short frame sequences that act as building blocks, featuring critical aerobatic behaviors for tractable trajectory synthesis. The model learns aerobatic primitives using historical trajectory observations as dynamic priors to ensure motion continuity, with additional conditional inputs (target waypoints and optional action constraints) integrated to enable user-editable trajectory generation. During model inference, classifier guidance is incorporated with batch sampling to achieve obstacle avoidance. Additionally, the generated outcomes are refined through post-processing with spatial-temporal trajectory optimization to ensure dynamical feasibility. Extensive simulations and real-world experiments have validated the key component designs of our method, demonstrating its feasibility for deploying on real drones to achieve long-horizon aerobatic flight. 

**Abstract (ZH)**: 利用扩散模型自动扩展空中特技轨迹生成的新框架：在复杂环境中执行令人印象深刻的有氧飞行需要提前手动设计关键机动，随着飞行轨迹范围变得广泛，这一过程既复杂又耗时。本文提出了一种利用扩散模型自动扩展和放大空中特技轨迹生成的新框架。我们的核心创新是将复杂的机动分解为空中特技基本功，这些基本功是作为构建块的短帧序列，具备关键的空中特技行为，以实现可处理的轨迹合成。该模型通过历史轨迹观察动态先验学习空中特技基本功，并通过额外的条件输入（目标航点和可选的动作约束）实现用户可编辑的轨迹生成。在模型推理过程中，通过批量采样结合分类器指导实现障碍物避免。此外，通过时空轨迹优化后处理生成的结果，以确保动力学可行性。本方法的关键组件设计经过广泛的仿真和实地实验验证，证明其可用于实际无人机上的高航程空中特技飞行。 

---
# Robust Planning and Control of Omnidirectional MRAVs for Aerial Communications in Wireless Networks 

**Title (ZH)**: omnidirectional MRAVs在无线网络中空中通信的稳健规划与控制 

**Authors**: Giuseppe Silano, Daniel Bonilla Licea, Hajar El Hammouti, Mounir Ghogho, and Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2504.15089)  

**Abstract**: A new class of Multi-Rotor Aerial Vehicles (MRAVs), known as omnidirectional MRAVs (o-MRAVs), has gained attention for their ability to independently control 3D position and orientation. This capability enhances robust planning and control in aerial communication networks, enabling more adaptive trajectory planning and precise antenna alignment without additional mechanical components. These features are particularly valuable in uncertain environments, where disturbances such as wind and interference affect communication stability. This paper examines o-MRAVs in the context of robust aerial network planning, comparing them with the more common under-actuated MRAVs (u-MRAVs). Key applications, including physical layer security, optical communications, and network densification, are highlighted, demonstrating the potential of o-MRAVs to improve reliability and efficiency in dynamic communication scenarios. 

**Abstract (ZH)**: 一种新的全方位多旋翼航空车辆（o-MRAVs）类：在 robust 悬空网络规划中的应用及其优势 

---
# Never too Cocky to Cooperate: An FIM and RL-based USV-AUV Collaborative System for Underwater Tasks in Extreme Sea Conditions 

**Title (ZH)**: 始终保持谦逊的合作：一种基于FIM和RL的USV-AUV协作系统，用于极端海况下的水下任务 

**Authors**: Jingzehua Xu, Guanwen Xie, Jiwei Tang, Yimian Ding, Weiyi Liu, Shuai Zhang, Yi Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14894)  

**Abstract**: This paper develops a novel unmanned surface vehicle (USV)-autonomous underwater vehicle (AUV) collaborative system designed to enhance underwater task performance in extreme sea conditions. The system integrates a dual strategy: (1) high-precision multi-AUV localization enabled by Fisher information matrix-optimized USV path planning, and (2) reinforcement learning-based cooperative planning and control method for multi-AUV task execution. Extensive experimental evaluations in the underwater data collection task demonstrate the system's operational feasibility, with quantitative results showing significant performance improvements over baseline methods. The proposed system exhibits robust coordination capabilities between USV and AUVs while maintaining stability in extreme sea conditions. To facilitate reproducibility and community advancement, we provide an open-source simulation toolkit available at: this https URL . 

**Abstract (ZH)**: 本文开发了一种新颖的自主水面车辆（USV）-自主水下 vehicle（AUV）协作系统，旨在在极端海况下增强水下任务性能。该系统集成了一种双策略：（1）由Fisher信息矩阵优化的USV路径规划实现的高精度多AUV定位；（2）基于强化学习的多AUV任务执行协作规划与控制方法。在水下数据采集任务中的广泛实验评估证明了该系统的操作可行性，并且定量结果表明与基准方法相比有显著性能改进。所提系统在极端海况下展示了USV与AUV之间的稳健协调能力，同时保持稳定性。为了促进可复制性和社区发展，我们提供了一个开源仿真工具包，可通过以下链接访问： this https URL 。 

---
# Safe Autonomous Environmental Contact for Soft Robots using Control Barrier Functions 

**Title (ZH)**: 软体机器人安全自主环境接触控制策略研究 

**Authors**: Akua K. Dickson, Juan C. Pacheco Garcia, Meredith L. Anderson, Ran Jing, Sarah Alizadeh-Shabdiz, Audrey X. Wang, Charles DeLorey, Zach J. Patterson, Andrew P. Sabelhaus  

**Link**: [PDF](https://arxiv.org/pdf/2504.14755)  

**Abstract**: Robots built from soft materials will inherently apply lower environmental forces than their rigid counterparts, and therefore may be more suitable in sensitive settings with unintended contact. However, these robots' applied forces result from both their design and their control system in closed-loop, and therefore, ensuring bounds on these forces requires controller synthesis for safety as well. This article introduces the first feedback controller for a soft manipulator that formally meets a safety specification with respect to environmental contact. In our proof-of-concept setting, the robot's environment has known geometry and is deformable with a known elastic modulus. Our approach maps a bound on applied forces to a safe set of positions of the robot's tip via predicted deformations of the environment. Then, a quadratic program with Control Barrier Functions in its constraints is used to supervise a nominal feedback signal, verifiably maintaining the robot's tip within this safe set. Hardware experiments on a multi-segment soft pneumatic robot demonstrate that the proposed framework successfully constrains its environmental contact forces. This framework represents a fundamental shift in perspective on control and safety for soft robots, defining and implementing a formally verifiable logic specification on their pose and contact forces. 

**Abstract (ZH)**: 软材料构建的机器人相较于刚性机器人会施加更低的环境作用力，因此在可能产生无意接触的敏感环境中可能更为适用。然而，这些机器人施加的作用力来源于其设计及闭环控制系统的控制，因此确保这些作用力的边界需要通过控制器合成来保障安全性。本文引入了首个正式满足环境接触安全性规范的软 manipulator 的反馈控制器。在我们的概念验证设定中，机器人环境具有已知的几何结构，并且具有已知的弹性模量的变形能力。我们的方法将施加力的边界映射到机器人末端安全位置集合，通过预测环境变形实现。然后，通过具有控制约束功能的二次规划来监督名义反馈信号，验证性地保持机器人末端在安全集合内。硬件实验表明，所提出框架成功地约束了其环境接触力。该框架代表了软机器人控制与安全的基本视角转变，定义并实现了一个形式化可验证的姿态与接触力逻辑规范。 

---
# A Modularized Design Approach for GelSight Family of Vision-based Tactile Sensors 

**Title (ZH)**: 基于视觉触觉传感器GelSight家族的模块化设计方法 

**Authors**: Arpit Agarwal, Mohammad Amin Mirzaee, Xiping Sun, Wenzhen Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.14739)  

**Abstract**: GelSight family of vision-based tactile sensors has proven to be effective for multiple robot perception and manipulation tasks. These sensors are based on an internal optical system and an embedded camera to capture the deformation of the soft sensor surface, inferring the high-resolution geometry of the objects in contact. However, customizing the sensors for different robot hands requires a tedious trial-and-error process to re-design the optical system. In this paper, we formulate the GelSight sensor design process as a systematic and objective-driven design problem and perform the design optimization with a physically accurate optical simulation. The method is based on modularizing and parameterizing the sensor's optical components and designing four generalizable objective functions to evaluate the sensor. We implement the method with an interactive and easy-to-use toolbox called OptiSense Studio. With the toolbox, non-sensor experts can quickly optimize their sensor design in both forward and inverse ways following our predefined modules and steps. We demonstrate our system with four different GelSight sensors by quickly optimizing their initial design in simulation and transferring it to the real sensors. 

**Abstract (ZH)**: 基于视觉的GelSight家族触觉传感器在多个机器人感知与 manipulation 任务中证明非常有效。这些传感器基于内部光学系统和嵌入式摄像头以捕获软传感器表面的变形，并推断出接触物体的高分辨率几何结构。然而，针对不同机器人手部定制传感器需要一个繁琐的试错过程来重新设计光学系统。在本文中，我们将GelSight传感器的设计过程转化为一个系统化和目标驱动的设计问题，并利用物理准确的光学模拟进行设计优化。该方法基于模块化和参数化传感器的光学组件，并设计了四个可泛化的目标函数来评估传感器。我们通过一个交互式且易于使用的工具箱OptiSense Studio实施了该方法。利用该工具箱，非传感器专家可以按照我们预定义的模块和步骤，快速从前向和逆向两个方面优化传感器设计。我们通过快速优化四个不同GelSight传感器的初始设计并在仿真实验中进行验证来展示我们的系统，并将优化结果转移到实际传感器上。 

---
# BiDexHand: Design and Evaluation of an Open-Source 16-DoF Biomimetic Dexterous Hand 

**Title (ZH)**: BiDexHand：一种开源16自由度仿生灵巧手的设计与评估 

**Authors**: Zhengyang Kris Weng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14712)  

**Abstract**: Achieving human-level dexterity in robotic hands remains a fundamental challenge for enabling versatile manipulation across diverse applications. This extended abstract presents BiDexHand, a cable-driven biomimetic robotic hand that combines human-like dexterity with accessible and efficient mechanical design. The robotic hand features 16 independently actuated degrees of freedom and 5 mechanically coupled joints through novel phalange designs that replicate natural finger motion. Performance validation demonstrated success across all 33 grasp types in the GRASP Taxonomy, 9 of 11 positions in the Kapandji thumb opposition test, a measured fingertip force of 2.14\,N, and the capability to lift a 10\,lb weight. As an open-source platform supporting multiple control modes including vision-based teleoperation, BiDexHand aims to democratize access to advanced manipulation capabilities for the broader robotics research community. 

**Abstract (ZH)**: 实现人类级别的灵巧性在机器人手中仍然是跨多种应用实现通用操作的 fundamental 挑战。本扩展摘要介绍 BiDexHand，一种集成了人类灵巧性和可访问高效机械设计的电缆驱动仿生机器人手。该机器人手具备 16 个独立驱动的自由度和 5 个机械耦合关节，通过新颖的指节设计模仿天然手指运动。性能验证显示，其在 GRASP 分类法中的 33 种抓取类型、Kapandji 小指对指测试中的 9 个位置、测得的指尖力量为 2.14 N，以及举起 10 磅重量的能力。作为支持多种控制模式（包括基于视觉的远程操作）的开源平台，BiDexHand 力求为更广泛的机器人研究社区提供高级操作能力的平权访问。 

---
# Latent Representations for Visual Proprioception in Inexpensive Robots 

**Title (ZH)**: 廉价机器人中的视觉本体感受表示 

**Authors**: Sahara Sheikholeslami, Ladislau Bölöni  

**Link**: [PDF](https://arxiv.org/pdf/2504.14634)  

**Abstract**: Robotic manipulation requires explicit or implicit knowledge of the robot's joint positions. Precise proprioception is standard in high-quality industrial robots but is often unavailable in inexpensive robots operating in unstructured environments. In this paper, we ask: to what extent can a fast, single-pass regression architecture perform visual proprioception from a single external camera image, available even in the simplest manipulation settings? We explore several latent representations, including CNNs, VAEs, ViTs, and bags of uncalibrated fiducial markers, using fine-tuning techniques adapted to the limited data available. We evaluate the achievable accuracy through experiments on an inexpensive 6-DoF robot. 

**Abstract (ZH)**: 基于单个外部摄像头图像的快速单次通过回归架构能否实现视觉本体感觉：在简单的操作设置中，这种架构能在多大程度上从单一外部相机图像中进行视觉本体感觉估计？ 

---
# K2MUSE: A human lower limb multimodal dataset under diverse conditions for facilitating rehabilitation robotics 

**Title (ZH)**: K2MUSE: 在多样化条件下的人类下肢多模态数据集，用于促进康复机器人技术 

**Authors**: Jiwei Li, Bi Zhang, Xiaowei Tan, Wanxin Chen, Zhaoyuan Liu, Juanjuan Zhang, Weiguang Huo, Jian Huang, Lianqing Liu, Xingang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.14602)  

**Abstract**: The natural interaction and control performance of lower limb rehabilitation robots are closely linked to biomechanical information from various human locomotion activities. Multidimensional human motion data significantly deepen the understanding of the complex mechanisms governing neuromuscular alterations, thereby facilitating the development and application of rehabilitation robots in multifaceted real-world environments. However, currently available lower limb datasets are inadequate for supplying the essential multimodal data and large-scale gait samples necessary for effective data-driven approaches, and they neglect the significant effects of acquisition interference in real this http URL fill this gap, we present the K2MUSE dataset, which includes a comprehensive collection of multimodal data, comprising kinematic, kinetic, amplitude-mode ultrasound (AUS), and surface electromyography (sEMG) measurements. The proposed dataset includes lower limb multimodal data from 30 able-bodied participants walking under different inclines (0$^\circ$, $\pm$5$^\circ$, and $\pm$10$^\circ$), various speeds (0.5 m/s, 1.0 m/s, and 1.5 m/s), and different nonideal acquisition conditions (muscle fatigue, electrode shifts, and inter-day differences). The kinematic and ground reaction force data were collected via a Vicon motion capture system and an instrumented treadmill with embedded force plates, whereas the sEMG and AUS data were synchronously recorded for thirteen muscles on the bilateral lower limbs. This dataset offers a new resource for designing control frameworks for rehabilitation robots and conducting biomechanical analyses of lower limb locomotion. The dataset is available at this https URL. 

**Abstract (ZH)**: 下肢康复机器人的人机自然交互与控制性能与多种人体运动活动的生物力学信息密切相关。多维人体运动数据加深了对调控神经肌肉改变复杂机制的理解，进而促进了康复机器人在多种现实环境中的开发与应用。然而，目前可用的下肢数据集无法提供有效数据驱动方法所需的关键多模态数据和大规模步态样本，并且忽略了实际获取干扰的影响。为填补这一空白，我们提出了K2MUSE数据集，该数据集包含全面的多模态数据，包括运动学、动力学、振幅模式超声波（AUS）和表层肌电图（sEMG）测量。拟议的数据集包括30名健康参与者在不同坡度（0°、±5°和±10°）、不同速度（0.5 m/s、1.0 m/s和1.5 m/s）和不同非理想采集条件（肌肉疲劳、电极移位和日间差异）下行走的下肢多模态数据。运动学和地面反作用力数据是通过Vicon运动捕捉系统和内置力板的仪器跑步机收集的，而sEMG和AUS数据则同步记录了双侧下肢的十三块肌肉。该数据集为设计康复机器人控制框架和下肢运动的生物力学分析提供了新的资源。数据集可在以下链接获取：[此链接]。 

---
# Haptic-based Complementary Filter for Rigid Body Rotations 

**Title (ZH)**: 基于触觉的补充滤波器用于刚体旋转 

**Authors**: Amit Kumar, Domenico Campolo, Ravi N. Banavar  

**Link**: [PDF](https://arxiv.org/pdf/2504.14570)  

**Abstract**: The non-commutative nature of 3D rotations poses well-known challenges in generalizing planar problems to three-dimensional ones, even more so in contact-rich tasks where haptic information (i.e., forces/torques) is involved. In this sense, not all learning-based algorithms that are currently available generalize to 3D orientation estimation. Non-linear filters defined on $\mathbf{\mathbb{SO}(3)}$ are widely used with inertial measurement sensors; however, none of them have been used with haptic measurements. This paper presents a unique complementary filtering framework that interprets the geometric shape of objects in the form of superquadrics, exploits the symmetry of $\mathbf{\mathbb{SO}(3)}$, and uses force and vision sensors as measurements to provide an estimate of orientation. The framework's robustness and almost global stability are substantiated by a set of experiments on a dual-arm robotic setup. 

**Abstract (ZH)**: 非交换的三维旋转性质给平面问题推广到三维问题带来了已知的挑战，尤其是在涉及触觉信息（即力/力矩）的丰富触觉任务中。因此，并非所有当前可用的基于学习的算法都能推广到三维姿态估计。定义在$\mathbf{\mathbb{SO}(3)}$上的非线性滤波器通常与惯性测量传感器结合使用；然而，尚未有研究将此类滤波器与触觉测量结合使用。本文提出了一种独特的互补滤波框架，该框架以超 Quadrics 的几何形状来解释物体的形状，利用 $\mathbf{\mathbb{SO}(3)}$ 的对称性，并使用力和视觉传感器作为测量来提供姿态估计。该框架的鲁棒性和几乎全局稳定性通过在双臂机器人平台上的实验得以验证。 

---
# RadarTrack: Enhancing Ego-Vehicle Speed Estimation with Single-chip mmWave Radar 

**Title (ZH)**: RadarTrack: 采用单芯片毫米波雷达提升 ego-车辆速度估计 

**Authors**: Argha Sen, Soham Chakraborty, Soham Tripathy, Sandip Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2504.14495)  

**Abstract**: In this work, we introduce RadarTrack, an innovative ego-speed estimation framework utilizing a single-chip millimeter-wave (mmWave) radar to deliver robust speed estimation for mobile platforms. Unlike previous methods that depend on cross-modal learning and computationally intensive Deep Neural Networks (DNNs), RadarTrack utilizes a novel phase-based speed estimation approach. This method effectively overcomes the limitations of conventional ego-speed estimation approaches which rely on doppler measurements and static surrondings. RadarTrack is designed for low-latency operation on embedded platforms, making it suitable for real-time applications where speed and efficiency are critical. Our key contributions include the introduction of a novel phase-based speed estimation technique solely based on signal processing and the implementation of a real-time prototype validated through extensive real-world evaluations. By providing a reliable and lightweight solution for ego-speed estimation, RadarTrack holds significant potential for a wide range of applications, including micro-robotics, augmented reality, and autonomous navigation. 

**Abstract (ZH)**: RadarTrack：一种利用单芯片毫米波雷达的创新自我速度估计框架 

---
# Collision Induced Binding and Transport of Shape Changing Robot Pairs 

**Title (ZH)**: 形状变化的机器人对的碰撞诱导结合与传输 

**Authors**: Akash Vardhan, Ram Avinery, Hosain Bagheri, Velin Kojohourav, Shengkai Li, Hridesh Kedia, Tianyu Wang, Daniel Soto, Kurt Wiesenfeld, Daniel I. Goldman  

**Link**: [PDF](https://arxiv.org/pdf/2504.14170)  

**Abstract**: We report in experiment and simulation the spontaneous formation of dynamically bound pairs of shape changing robots undergoing locally repulsive collisions. These physical `gliders' robustly emerge from an ensemble of individually undulating three-link two-motor robots and can remain bound for hundreds of undulations and travel for multiple robot dimensions. Gliders occur in two distinct binding symmetries and form over a wide range of angular oscillation extent. This parameter sets the maximal concavity which influences formation probability and translation characteristics. Analysis of dynamics in simulation reveals the mechanism of effective dynamical attraction -- a result of the emergent interplay of appropriately oriented and timed repulsive interactions. Tactile sensing stabilizes the short-lived conformation via concavity modulation. 

**Abstract (ZH)**: 我们报告了实验和模拟中形状可变机器人在局部排斥碰撞下自发形成的动态结合对的现象。这些物理“滑行器”稳健地从三链双电机独立蜿蜒运动的机器人集群中涌现，并可保持结合数百次蜿蜒运动并行进多个机器人尺寸。滑行器具有两种不同的结合对称性，并可在广泛的角振幅范围内形成。该参数确定了最大凹度，影响形成概率和转换特性。仿真动力学分析揭示了有效的动态吸引力机制——这是适当定向和时间协调的排斥相互作用的涌现效应的结果。触觉感知通过凹度调节来稳定短暂的构型。 

---
# Enhanced UAV Navigation Systems through Sensor Fusion with Trident Quaternions 

**Title (ZH)**: 通过三叉四元数的传感器融合增强无人机导航系统 

**Authors**: Sebastian Incicco, Juan Ignacio Giribet, Leonardo Colombo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14133)  

**Abstract**: This paper presents an integrated navigation algorithm based on trident quaternions, an extension of dual quaternions. The proposed methodology provides an efficient approach for achieving precise and robust navigation by leveraging the advantages of trident quaternions. The performance of the navigation system was validated through experimental tests using a multi-rotor UAV equipped with two navigation computers: one executing the proposed algorithm and the other running a commercial autopilot, which was used as a reference. 

**Abstract (ZH)**: 基于三叉四元数的综合导航算法研究：扩展双四元数的应用 

---
# Infrared Vision Systems for Emergency Vehicle Driver Assistance in Low-Visibility Conditions 

**Title (ZH)**: 低能见度条件下应急车辆驾驶员辅助的红外 vision 系统 

**Authors**: M-Mahdi Naddaf-Sh, Andrew Lee, Kin Yen, Eemon Amini, Iman Soltani  

**Link**: [PDF](https://arxiv.org/pdf/2504.14078)  

**Abstract**: This study investigates the potential of infrared (IR) camera technology to enhance driver safety for emergency vehicles operating in low-visibility conditions, particularly at night and in dense fog. Such environments significantly increase the risk of collisions, especially for tow trucks and snowplows that must remain operational in challenging conditions. Conventional driver assistance systems often struggle under these conditions due to limited visibility. In contrast, IR cameras, which detect the thermal signatures of obstacles, offer a promising alternative. The evaluation combines controlled laboratory experiments, real-world field tests, and surveys of emergency vehicle operators. In addition to assessing detection performance, the study examines the feasibility of retrofitting existing Department of Transportation (DoT) fleets with cost-effective IR-based driver assistance systems. Results underscore the utility of IR technology in enhancing driver awareness and provide data-driven recommendations for scalable deployment across legacy emergency vehicle fleets. 

**Abstract (ZH)**: 本研究探讨红外（IR）相机技术在低能见度条件下，特别是夜间和大雾环境中，增强应急车辆驾驶员安全的潜力。此类环境显著增加了碰撞风险，尤其是对于必须在恶劣条件下保持运行的拖车和除雪车。传统驾驶辅助系统在这些条件下往往由于能见度有限而效果不佳。相比之下，红外相机通过检测障碍物的热特征，提供了一种有前景的替代方案。评估结合了受控实验室实验、实地测试以及对应急车辆操作员的调查。除了评估检测性能外，该研究还检查了将成本效益高的红外基驾驶辅助系统安装到现有交通部（DoT）车队中的可行性。研究结果强调了红外技术在增强驾驶员意识方面的实用性，并提供了针对现有应急车辆车队的可扩展部署的数据驱动建议。 

---
# Knitting Robots: A Deep Learning Approach for Reverse-Engineering Fabric Patterns 

**Title (ZH)**: 织布机器人：一种用于逆向工程织物图案的深度学习方法 

**Authors**: Haoliang Sheng, Songpu Cai, Xingyu Zheng, Meng Cheng Lau  

**Link**: [PDF](https://arxiv.org/pdf/2504.14007)  

**Abstract**: Knitting, a cornerstone of textile manufacturing, is uniquely challenging to automate, particularly in terms of converting fabric designs into precise, machine-readable instructions. This research bridges the gap between textile production and robotic automation by proposing a novel deep learning-based pipeline for reverse knitting to integrate vision-based robotic systems into textile manufacturing. The pipeline employs a two-stage architecture, enabling robots to first identify front labels before inferring complete labels, ensuring accurate, scalable pattern generation. By incorporating diverse yarn structures, including single-yarn (sj) and multi-yarn (mj) patterns, this study demonstrates how our system can adapt to varying material complexities. Critical challenges in robotic textile manipulation, such as label imbalance, underrepresented stitch types, and the need for fine-grained control, are addressed by leveraging specialized deep-learning architectures. This work establishes a foundation for fully automated robotic knitting systems, enabling customizable, flexible production processes that integrate perception, planning, and actuation, thereby advancing textile manufacturing through intelligent robotic automation. 

**Abstract (ZH)**: 针织，纺织制造的基石，特别在将织物设计转化为精确的机器可读指令方面极具挑战性。本研究通过提出一种基于深度学习的创新管道，弥补了纺织生产和机器人自动化之间的差距，旨在将基于视觉的机器人系统集成到纺织制造中。该管道采用两阶段架构，使机器人首先识别正面标签，然后推断完整标签，确保准确的、可扩展的图案生成。通过纳入包括单纱 (sj) 和多纱 (mj) 模式在内的各种纱线结构，本研究展示了系统如何适应不同的材料复杂性。通过利用专门的深度学习架构，本文解决了机器人纺织操作中的关键挑战，如标签不平衡、代表性不足的针法类型以及细粒度控制的需求。这项工作为全自动机器人针织系统奠定了基础，使其能够实现可定制的、灵活的生产过程，整合感知、规划和执行，从而通过智能机器人自动化推动纺织制造的发展。 

---
# Coordinating Spinal and Limb Dynamics for Enhanced Sprawling Robot Mobility 

**Title (ZH)**: 增强 sprawling 机器人移动性的脊椎与肢体动力学协调 

**Authors**: Merve Atasever, Ali Okhovat, Azhang Nazaripouya, John Nisbet, Omer Kurkutlu, Jyotirmoy V. Deshmukh, Yasemin Ozkan Aydin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14103)  

**Abstract**: Among vertebrates, salamanders, with their unique ability to transition between walking and swimming gaits, highlight the role of spinal mobility in locomotion. A flexible spine enables undulation of the body through a wavelike motion along the spine, aiding navigation over uneven terrains and obstacles. Yet environmental uncertainties, such as surface irregularities and variations in friction, can significantly disrupt body-limb coordination and cause discrepancies between predictions from mathematical models and real-world outcomes. Addressing this challenge requires the development of sophisticated control strategies capable of dynamically adapting to uncertain conditions while maintaining efficient locomotion. Deep reinforcement learning (DRL) offers a promising framework for handling non-deterministic environments and enabling robotic systems to adapt effectively and perform robustly under challenging conditions. In this study, we comparatively examine learning-based control strategies and biologically inspired gait design methods on a salamander-like robot. 

**Abstract (ZH)**: 在脊椎动物中，蝾螈因其独特的行走与游泳姿态转换能力，突显了脊柱灵活性在运动中的作用。灵活的脊柱通过脊柱上的波浪状运动实现身体的波浪式摆动，有助于在不平地形和障碍物上的导航。然而，环境不确定性，如表面不规则性和摩擦力变化，会显著干扰身体与肢体的协调，导致数学模型预测与实际结果之间存在差异。解决这一挑战需要开发出能够动态适应不确定条件并保持高效运动的复杂控制策略。深度强化学习（DRL）为处理非确定性环境并使机器人系统在挑战性条件下有效适应和稳健运行提供了有前景的框架。在这项研究中，我们比较了基于学习的控制策略和生物启发的步态设计方法在蝾螈类机器人上的应用。 

---
