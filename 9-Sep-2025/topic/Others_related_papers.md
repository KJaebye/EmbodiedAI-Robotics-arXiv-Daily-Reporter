# LiHRA: A LiDAR-Based HRI Dataset for Automated Risk Monitoring Methods 

**Title (ZH)**: 基于激光雷达的HRI数据集LiHRA：自动化风险监测方法 

**Authors**: Frederik Plahl, Georgios Katranis, Ilshat Mamaev, Andrey Morozov  

**Link**: [PDF](https://arxiv.org/pdf/2509.06597)  

**Abstract**: We present LiHRA, a novel dataset designed to facilitate the development of automated, learning-based, or classical risk monitoring (RM) methods for Human-Robot Interaction (HRI) scenarios. The growing prevalence of collaborative robots in industrial environments has increased the need for reliable safety systems. However, the lack of high-quality datasets that capture realistic human-robot interactions, including potentially dangerous events, slows development. LiHRA addresses this challenge by providing a comprehensive, multi-modal dataset combining 3D LiDAR point clouds, human body keypoints, and robot joint states, capturing the complete spatial and dynamic context of human-robot collaboration. This combination of modalities allows for precise tracking of human movement, robot actions, and environmental conditions, enabling accurate RM during collaborative tasks. The LiHRA dataset covers six representative HRI scenarios involving collaborative and coexistent tasks, object handovers, and surface polishing, with safe and hazardous versions of each scenario. In total, the data set includes 4,431 labeled point clouds recorded at 10 Hz, providing a rich resource for training and benchmarking classical and AI-driven RM algorithms. Finally, to demonstrate LiHRA's utility, we introduce an RM method that quantifies the risk level in each scenario over time. This method leverages contextual information, including robot states and the dynamic model of the robot. With its combination of high-resolution LiDAR data, precise human tracking, robot state data, and realistic collision events, LiHRA offers an essential foundation for future research into real-time RM and adaptive safety strategies in human-robot workspaces. 

**Abstract (ZH)**: LiHRA：一种用于人类-机器人交互风险监测的数据集 

---
# Co-Located VR with Hybrid SLAM-based HMD Tracking and Motion Capture Synchronization 

**Title (ZH)**: 基于混合SLAM的HMD跟踪与运动捕捉同步的共存虚拟现实 

**Authors**: Carlos A. Pinheiro de Sousa, Niklas Gröne, Mathias Günther, Oliver Deussen  

**Link**: [PDF](https://arxiv.org/pdf/2509.06582)  

**Abstract**: We introduce a multi-user VR co-location framework that synchronizes users within a shared virtual environment aligned to physical space. Our approach combines a motion capture system with SLAM-based inside-out tracking to deliver smooth, high-framerate, low-latency performance. Previous methods either rely on continuous external tracking, which introduces latency and jitter, or on one-time calibration, which cannot correct drift over time. In contrast, our approach combines the responsiveness of local HMD SLAM tracking with the flexibility to realign to an external source when needed. It also supports real-time pose sharing across devices, ensuring consistent spatial alignment and engagement between users. Our evaluation demonstrates that our framework achieves the spatial accuracy required for natural multi-user interaction while offering improved comfort, scalability, and robustness over existing co-located VR solutions. 

**Abstract (ZH)**: 我们介绍了一种多用户VR共处框架，该框架通过将用户同步到与物理空间对齐的共享虚拟环境中来进行同步。该方法结合了运动捕捉系统与基于SLAM的内部跟踪技术，提供了平滑、高帧率、低延迟的性能。以往的方法要么依赖连续的外部跟踪，这会引入延迟和抖动，要么依赖一次性校准，这种校准无法随着时间纠正漂移。相比之下，我们的方法结合了本地HMD SLAM跟踪的响应性与在需要时重新校准到外部源的灵活性。它还支持设备间的实时姿态共享，确保用户之间的一致的空间对齐和参与度。我们的评估表明，该框架在实现自然多用户交互所需的空间精度的同时，提供了比现有共处VR解决方案更好的舒适性、可扩展性和鲁棒性。 

---
# Safety Meets Speed: Accelerated Neural MPC with Safety Guarantees and No Retraining 

**Title (ZH)**: 安全与速度并重：具备安全性保证且无需重新训练的加速神经MPC 

**Authors**: Kaikai Wang, Tianxun Li, Liang Xu, Qinglei Hu, Keyou You  

**Link**: [PDF](https://arxiv.org/pdf/2509.06404)  

**Abstract**: While Model Predictive Control (MPC) enforces safety via constraints, its real-time execution can exceed embedded compute budgets. We propose a Barrier-integrated Adaptive Neural Model Predictive Control (BAN-MPC) framework that synergizes neural networks' fast computation with MPC's constraint-handling capability. To ensure strict safety, we replace traditional Euclidean distance with Control Barrier Functions (CBFs) for collision avoidance. We integrate an offline-learned neural value function into the optimization objective of a Short-horizon MPC, substantially reducing online computational complexity. Additionally, we use a second neural network to learn the sensitivity of the value function to system parameters, and adaptively adjust the neural value function based on this neural sensitivity when model parameters change, eliminating the need for retraining and reducing offline computation costs. The hardware in-the-loop (HIL) experiments on Jetson Nano show that BAN-MPC solves 200 times faster than traditional MPC, enabling collision-free navigation with control error below 5\% under model parameter variations within 15\%, making it an effective embedded MPC alternative. 

**Abstract (ZH)**: Barrier-integrated Adaptive Neural Model Predictive Control (BAN-MPC) Framework 

---
# Adaptive Evolution Factor Risk Ellipse Framework for Reliable and Safe Autonomous Driving 

**Title (ZH)**: 自适应进化因子风险椭圆框架以实现可靠和安全的自动驾驶 

**Authors**: Fujiang Yuan, Zhen Tian, Yangfan He, Guojian Zou, Chunhong Yuan, Yanhong Peng, Zhihao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06375)  

**Abstract**: In recent years, ensuring safety, efficiency, and comfort in interactive autonomous driving has become a critical challenge. Traditional model-based techniques, such as game-theoretic methods and robust control, are often overly conservative or computationally intensive. Conversely, learning-based approaches typically require extensive training data and frequently exhibit limited interpretability and generalizability. Simpler strategies, such as Risk Potential Fields (RPF), provide lightweight alternatives with minimal data demands but are inherently static and struggle to adapt effectively to dynamic traffic conditions. To overcome these limitations, we propose the Evolutionary Risk Potential Field (ERPF), a novel approach that dynamically updates risk assessments in dynamical scenarios based on historical obstacle proximity data. We introduce a Risk-Ellipse construct that combines longitudinal reach and lateral uncertainty into a unified spatial temporal collision envelope. Additionally, we define an adaptive Evolution Factor metric, computed through sigmoid normalization of Time to Collision (TTC) and Time-Window-of-Hazard (TWH), which dynamically adjusts the dimensions of the ellipse axes in real time. This adaptive risk metric is integrated seamlessly into a Model Predictive Control (MPC) framework, enabling autonomous vehicles to proactively address complex interactive driving scenarios in terms of uncertain driving of surrounding vehicles. Comprehensive comparative experiments demonstrate that our ERPF-MPC approach consistently achieves smoother trajectories, higher average speeds, and collision-free navigation, offering a robust and adaptive solution suitable for complex interactive driving environments. 

**Abstract (ZH)**: 基于进化的风险椭球场（ERPF）在动态场景中动态更新风险评估的模型预测控制方法 

---
# DCReg: Decoupled Characterization for Efficient Degenerate LiDAR Registration 

**Title (ZH)**: DCReg: 解耦特征表征以实现高效的退化LiDAR注册 

**Authors**: Xiangcheng Hu, Xieyuanli Chen, Mingkai Jia, Jin Wu, Ping Tan, Steven L. Waslander  

**Link**: [PDF](https://arxiv.org/pdf/2509.06285)  

**Abstract**: LiDAR point cloud registration is fundamental to robotic perception and navigation. However, in geometrically degenerate or narrow environments, registration problems become ill-conditioned, leading to unstable solutions and degraded accuracy. While existing approaches attempt to handle these issues, they fail to address the core challenge: accurately detection, interpret, and resolve this ill-conditioning, leading to missed detections or corrupted solutions. In this study, we introduce DCReg, a principled framework that systematically addresses the ill-conditioned registration problems through three integrated innovations. First, DCReg achieves reliable ill-conditioning detection by employing a Schur complement decomposition to the hessian matrix. This technique decouples the registration problem into clean rotational and translational subspaces, eliminating coupling effects that mask degeneracy patterns in conventional analyses. Second, within these cleanly subspaces, we develop quantitative characterization techniques that establish explicit mappings between mathematical eigenspaces and physical motion directions, providing actionable insights about which specific motions lack constraints. Finally, leveraging this clean subspace, we design a targeted mitigation strategy: a novel preconditioner that selectively stabilizes only the identified ill-conditioned directions while preserving all well-constrained information in observable space. This enables efficient and robust optimization via the Preconditioned Conjugate Gradient method with a single physical interpretable parameter. Extensive experiments demonstrate DCReg achieves at least 20% - 50% improvement in localization accuracy and 5-100 times speedup over state-of-the-art methods across diverse environments. Our implementation will be available at this https URL. 

**Abstract (ZH)**: LiDAR点云注册是机器人感知与导航的基础。然而，在几何退化或狭窄环境中，注册问题变得病态，导致不稳定解和降低的精度。尽管现有方法尝试解决这些问题，但未能从根本上解决核心挑战：准确检测、解释和解决这种病态性，导致误检或污染的解。在本研究中，我们介绍了DCReg，这是一种系统解决病态注册问题的原则性框架，通过三项集成创新。首先，DCReg通过Hessian矩阵的舒尔补分解可靠地检测病态性。该技术将注册问题分解为干净的旋转和平移子空间，消除传统分析中掩盖病态模式的耦合效应。其次，在这些干净的子空间内，我们开发了定量表征技术，建立了数学特征空间与物理运动方向之间的显式映射，提供了具体哪些运动缺乏约束的可操作见解。最后，利用这个干净子空间，我们设计了一种针对性的缓解策略：一种新颖的预处理因子，仅稳定已识别的病态方向，同时保留所有在可观测空间中的良好约束信息。这使得通过预处理共轭梯度方法进行高效的鲁棒优化成为可能，并且只需要一个物理可解释的参数。广泛实验表明，在各种环境中，DCReg相对于现有最佳方法在定位精度上至少提高了20%-50%，并加快了5-100倍的速度。我们的实现将可在以下链接获取。 

---
# eKalibr-Inertial: Continuous-Time Spatiotemporal Calibration for Event-Based Visual-Inertial Systems 

**Title (ZH)**: eKalibr-Inertial：事件驱动视觉-惯性系统的连续时空标定 

**Authors**: Shuolong Chen, Xingxing Li, Liu Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.05923)  

**Abstract**: The bioinspired event camera, distinguished by its exceptional temporal resolution, high dynamic range, and low power consumption, has been extensively studied in recent years for motion estimation, robotic perception, and object detection. In ego-motion estimation, the visual-inertial setup is commonly adopted due to complementary characteristics between sensors (e.g., scale perception and low drift). For optimal event-based visual-inertial fusion, accurate spatiotemporal (extrinsic and temporal) calibration is required. In this work, we present eKalibr-Inertial, an accurate spatiotemporal calibrator for event-based visual-inertial systems, utilizing the widely used circle grid board. Building upon the grid pattern recognition and tracking methods in eKalibr and eKalibr-Stereo, the proposed method starts with a rigorous and efficient initialization, where all parameters in the estimator would be accurately recovered. Subsequently, a continuous-time-based batch optimization is conducted to refine the initialized parameters toward better states. The results of extensive real-world experiments show that eKalibr-Inertial can achieve accurate event-based visual-inertial spatiotemporal calibration. The implementation of eKalibr-Inertial is open-sourced at (this https URL) to benefit the research community. 

**Abstract (ZH)**: 受生物启发的事件相机由于其卓越的时间分辨率、高动态范围和低功耗，在近年来的运动估计、机器人感知和物体检测领域得到了广泛研究。在自我运动估计中，由于传感器之间的互补特性（如尺度感知和低漂移），通常采用视觉-惯性配置。为了实现最优的基于事件的视觉-惯性融合，需要进行精确的空间-时间（外在和时间）校准。本文提出了eKalibr-Inertial，这是一种基于广泛使用的格子网格板的精确空间-时间校准器，该方法在eKalibr和eKalibr-Stereo的网格模式识别和跟踪方法的基础上，通过严格的高效初始化阶段，准确恢复所有估计算法的参数，随后进行基于连续时间的批量优化，进一步优化初始化参数。广泛的实验证明，eKalibr-Inertial可以实现精确的基于事件的视觉-惯性空间-时间校准。eKalibr-Inertial的实现已开源（this https URL），以服务于研究社区。 

---
# INF-3DP: Implicit Neural Fields for Collision-Free Multi-Axis 3D Printing 

**Title (ZH)**: INF-3DP: 隐式神经场在碰撞自由多轴3D打印中的应用 

**Authors**: Jiasheng Qu, Zhuo Huang, Dezhao Guo, Hailin Sun, Aoran Lyu, Chengkai Dai, Yeung Yam, Guoxin Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05345)  

**Abstract**: We introduce a general, scalable computational framework for multi-axis 3D printing based on implicit neural fields (INFs) that unifies all stages of toolpath generation and global collision-free motion planning. In our pipeline, input models are represented as signed distance fields, with fabrication objectives such as support-free printing, surface finish quality, and extrusion control being directly encoded in the optimization of an implicit guidance field. This unified approach enables toolpath optimization across both surface and interior domains, allowing shell and infill paths to be generated via implicit field interpolation. The printing sequence and multi-axis motion are then jointly optimized over a continuous quaternion field. Our continuous formulation constructs the evolving printing object as a time-varying SDF, supporting differentiable global collision handling throughout INF-based motion planning. Compared to explicit-representation-based methods, INF-3DP achieves up to two orders of magnitude speedup and significantly reduces waypoint-to-surface error. We validate our framework on diverse, complex models and demonstrate its efficiency with physical fabrication experiments using a robot-assisted multi-axis system. 

**Abstract (ZH)**: 基于隐神经场的通用可扩展多轴3D打印计算框架 

---
# VehicleWorld: A Highly Integrated Multi-Device Environment for Intelligent Vehicle Interaction 

**Title (ZH)**: VehicleWorld: 一种高度集成的多设备智能车辆交互环境 

**Authors**: Jie Yang, Jiajun Chen, Zhangyue Yin, Shuo Chen, Yuxin Wang, Yiran Guo, Yuan Li, Yining Zheng, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06736)  

**Abstract**: Intelligent vehicle cockpits present unique challenges for API Agents, requiring coordination across tightly-coupled subsystems that exceed typical task environments' complexity. Traditional Function Calling (FC) approaches operate statelessly, requiring multiple exploratory calls to build environmental awareness before execution, leading to inefficiency and limited error recovery. We introduce VehicleWorld, the first comprehensive environment for the automotive domain, featuring 30 modules, 250 APIs, and 680 properties with fully executable implementations that provide real-time state information during agent execution. This environment enables precise evaluation of vehicle agent behaviors across diverse, challenging scenarios. Through systematic analysis, we discovered that direct state prediction outperforms function calling for environmental control. Building on this insight, we propose State-based Function Call (SFC), a novel approach that maintains explicit system state awareness and implements direct state transitions to achieve target conditions. Experimental results demonstrate that SFC significantly outperforms traditional FC approaches, achieving superior execution accuracy and reduced latency. We have made all implementation code publicly available on Github this https URL. 

**Abstract (ZH)**: 智能车辆仪表板为API代理带来了独特挑战，需要协调紧密耦合的子系统，其复杂性远超典型任务环境。传统函数调用（FC）方法以无状态方式操作，要求在执行前进行多次探索性调用以建立环境意识，导致效率低下且错误恢复能力有限。我们引入了VehicleWorld，这是汽车领域首个全面的环境，包含30个模块、250个API和680个属性，提供了完全可执行的实现，并在代理执行过程中提供实时状态信息。该环境使得能够在多种具有挑战性的场景中精确评估车辆代理行为。通过系统分析，我们发现直接状态预测在环境控制方面优于函数调用。基于这一见解，我们提出了基于状态的函数调用（SFC）这一新颖方法，该方法保持明确的系统状态意识，并直接实现状态转换以达到目标条件。实验结果表明，SFC在执行准确性和降低延迟方面显著优于传统FC方法。我们已将所有实现代码公开发布在GitHub上：https://github.com/XXXXX。 

---
# MAPF-HD: Multi-Agent Path Finding in High-Density Environments 

**Title (ZH)**: MAPF-HD: 高密度环境中的多agent路径规划 

**Authors**: Hiroya Makino, Seigo Ito  

**Link**: [PDF](https://arxiv.org/pdf/2509.06374)  

**Abstract**: Multi-agent path finding (MAPF) involves planning efficient paths for multiple agents to move simultaneously while avoiding collisions. In typical warehouse environments, agents are often sparsely distributed along aisles. However, increasing the agent density can improve space efficiency. When the agent density is high, we must optimize the paths not only for goal-assigned agents but also for those obstructing them. This study proposes a novel MAPF framework for high-density environments (MAPF-HD). Several studies have explored MAPF in similar settings using integer linear programming (ILP). However, ILP-based methods require substantial computation time to optimize all agent paths simultaneously. Even in small grid-based environments with fewer than $100$ cells, these computations can incur tens to hundreds of seconds. These high computational costs render these methods impractical for large-scale applications such as automated warehouses and valet parking. To address these limitations, we introduce the phased null-agent swapping (PHANS) method. PHANS employs a heuristic approach to incrementally swap positions between agents and empty vertices. This method solves the MAPF-HD problem within seconds to tens of seconds, even in large environments containing more than $700$ cells. The proposed method can potentially improve efficiency in various real-world applications such as warehouse logistics, traffic management, or crowd control. Code is available at this https URL. 

**Abstract (ZH)**: 高密度环境下的多_agent路径规划（MAPF-HD） 

---
# Programming tension in 3D printed networks inspired by spiderwebs 

**Title (ZH)**: 受蜘蛛网启发的3D打印网络中的编程张力 

**Authors**: Thijs Masmeijer, Caleb Swain, Jeff Hill, Ed Habtour  

**Link**: [PDF](https://arxiv.org/pdf/2509.05855)  

**Abstract**: Each element in tensioned structural networks -- such as tensegrity, architectural fabrics, or medical braces/meshes -- requires a specific tension level to achieve and maintain the desired shape, stability, and compliance. These structures are challenging to manufacture, 3D print, or assemble because flattening the network during fabrication introduces multiplicative inaccuracies in the network's final tension gradients. This study overcomes this challenge by offering a fabrication algorithm for direct 3D printing of such networks with programmed tension gradients, an approach analogous to the spinning of spiderwebs. The algorithm: (i) defines the desired network and prescribes its tension gradients using the force density method; (ii) converts the network into an unstretched counterpart by numerically optimizing vertex locations toward target element lengths and converting straight elements into arcs to resolve any remaining error; and (iii) decomposes the network into printable toolpaths; Optional additional steps are: (iv) flattening curved 2D networks or 3D networks to ensure 3D printing compatibility; and (v) automatically resolving any unwanted crossings introduced by the flattening process. The proposed method is experimentally validated using 2D unit cells of viscoelastic filaments, where accurate tension gradients are achieved with an average element strain error of less than 1.0\%. The method remains effective for networks with element minimum length and maximum stress of 5.8 mm and 7.3 MPa, respectively. The method is used to demonstrate the fabrication of three complex cases: a flat spiderweb, a curved mesh, and a tensegrity system. The programmable tension gradient algorithm can be utilized to produce compact, integrated cable networks, enabling novel applications such as moment-exerting structures in medical braces and splints. 

**Abstract (ZH)**: 基于编程张力梯度的紧张结构网络直接3D打印算法 

---
# InterAct: A Large-Scale Dataset of Dynamic, Expressive and Interactive Activities between Two People in Daily Scenarios 

**Title (ZH)**: InterAct: 日常场景中两个人之间动态、-expressionistic 和互动活动的大规模数据集 

**Authors**: Leo Ho, Yinghao Huang, Dafei Qin, Mingyi Shi, Wangpok Tse, Wei Liu, Junichi Yamagishi, Taku Komura  

**Link**: [PDF](https://arxiv.org/pdf/2509.05747)  

**Abstract**: We address the problem of accurate capture of interactive behaviors between two people in daily scenarios. Most previous works either only consider one person or solely focus on conversational gestures of two people, assuming the body orientation and/or position of each actor are constant or barely change over each interaction. In contrast, we propose to simultaneously model two people's activities, and target objective-driven, dynamic, and semantically consistent interactions which often span longer duration and cover bigger space. To this end, we capture a new multi-modal dataset dubbed InterAct, which is composed of 241 motion sequences where two people perform a realistic and coherent scenario for one minute or longer over a complete interaction. For each sequence, two actors are assigned different roles and emotion labels, and collaborate to finish one task or conduct a common interaction activity. The audios, body motions, and facial expressions of both persons are captured. InterAct contains diverse and complex motions of individuals and interesting and relatively long-term interaction patterns barely seen before. We also demonstrate a simple yet effective diffusion-based method that estimates interactive face expressions and body motions of two people from speech inputs. Our method regresses the body motions in a hierarchical manner, and we also propose a novel fine-tuning mechanism to improve the lip accuracy of facial expressions. To facilitate further research, the data and code is made available at this https URL . 

**Abstract (ZH)**: 我们解决了一日常生活中两个人之间互动行为准确捕捉的问题。大多数先前的工作要么仅考虑一人，要么仅聚焦于两人对话手势，假设每个表演者的身体朝向和/or位置在每次互动中保持不变或仅微有变化。相反，我们提出同时建模两个人的活动，并针对具有目标驱动、动态且语义一致的互动，这些互动往往持续时间更长、覆盖范围更广。为此，我们捕获了一个名为InterAct的新多模态数据集，该数据集由241个运动序列组成，两个演员在一分钟或更长时间内完成一个现实且连贯的互动场景。对于每个序列，两位演员分配不同的角色和情绪标签，并协作完成一项任务或进行共同互动活动。两位演员的音频、身体运动和面部表情都被捕捉下来。InterAct包含个体多样且复杂的运动以及前所未见的有趣且相对长期的互动模式。我们还展示了从语音输入估计两人互动面部表情和身体运动的一种简单有效的扩散方法。该方法按层次回归身体运动，并提出了一种新的微调机制以提高面部表情的唇部准确性。为了促进进一步研究，数据和代码可通过此链接获取。 

---
# LiDAR-BIND-T: Improving SLAM with Temporally Consistent Cross-Modal LiDAR Reconstruction 

**Title (ZH)**: LiDAR-BIND-T：通过具有时间一致性的跨模态LiDAR重建改进SLAM 

**Authors**: Niels Balemans, Ali Anwar, Jan Steckel, Siegfried Mercelis  

**Link**: [PDF](https://arxiv.org/pdf/2509.05728)  

**Abstract**: This paper extends LiDAR-BIND, a modular multi-modal fusion framework that binds heterogeneous sensors (radar, sonar) to a LiDAR-defined latent space, with mechanisms that explicitly enforce temporal consistency. We introduce three contributions: (i) temporal embedding similarity that aligns consecutive latents, (ii) a motion-aligned transformation loss that matches displacement between predictions and ground truth LiDAR, and (iii) windows temporal fusion using a specialised temporal module. We further update the model architecture to better preserve spatial structure. Evaluations on radar/sonar-to-LiDAR translation demonstrate improved temporal and spatial coherence, yielding lower absolute trajectory error and better occupancy map accuracy in Cartographer-based SLAM (Simultaneous Localisation and Mapping). We propose different metrics based on the Fréchet Video Motion Distance (FVMD) and a correlation-peak distance metric providing practical temporal quality indicators to evaluate SLAM performance. The proposed temporal LiDAR-BIND, or LiDAR-BIND-T, maintains plug-and-play modality fusion while substantially enhancing temporal stability, resulting in improved robustness and performance for downstream SLAM. 

**Abstract (ZH)**: 本文扩展了LiDAR-BIND框架，该框架将雷达、声纳等异构传感器绑定到由LiDAR定义的潜空间中，并通过明确的机制确保时间一致性。我们提出了三项贡献：（i）时间嵌入相似性以对齐连续的潜变量，（ii）运动对齐变换损失以匹配预测与真实LiDAR之间的位移，以及（iii）使用专门的时间模块进行窗内时间融合。我们进一步更新了模型架构以更好地保持空间结构。在基于LiDAR/声纳到LiDAR的翻译评估中，展示了更好的时态和空间一致性，降低了绝对轨迹误差，并在Cartographer基于的同时定位与地图构建（SLAM）中提高了占用地图的准确性。我们提出了基于Fréchet视频运动距离（FVMD）和相关峰距离度量的不同评估指标，以提供实际的时间质量指标来评估SLAM性能。所提出的时态LiDAR-BIND（或LiDAR-BIND-T）在保持即插即用模态融合的同时显著增强了时间稳定性，从而提高了下游SLAM的鲁棒性和性能。 

---
# Cumplimiento del Reglamento (UE) 2024/1689 en robótica y sistemas autónomos: una revisión sistemática de la literatura 

**Title (ZH)**: 欧盟条例2024/1689在机器人和自主系统中的实施：文献综述 

**Authors**: Yoana Pita Lorenzo  

**Link**: [PDF](https://arxiv.org/pdf/2509.05380)  

**Abstract**: This systematic literature review analyzes the current state of compliance with Regulation (EU) 2024/1689 in autonomous robotic systems, focusing on cybersecurity frameworks and methodologies. Using the PRISMA protocol, 22 studies were selected from 243 initial records across IEEE Xplore, ACM DL, Scopus, and Web of Science. Findings reveal partial regulatory alignment: while progress has been made in risk management and encrypted communications, significant gaps persist in explainability modules, real-time human oversight, and knowledge base traceability. Only 40% of reviewed solutions explicitly address transparency requirements, and 30% implement failure intervention mechanisms. The study concludes that modular approaches integrating risk, supervision, and continuous auditing are essential to meet the AI Act mandates in autonomous robotics. 

**Abstract (ZH)**: This systematic literature review analyzes the current state of compliance with Regulation (EU) 2024/1689 in autonomous robotic systems, focusing on cybersecurity frameworks and methodologies。该系统文献综述分析了欧盟条例（EU）2024/1689 在自主机器人系统中的当前合规状况，重点关注网络安全框架和方法学。 

---
# Anticipatory Fall Detection in Humans with Hybrid Directed Graph Neural Networks and Long Short-Term Memory 

**Title (ZH)**: 基于混合有向图神经网络和长短期记忆的前瞻性跌倒检测在人类中 

**Authors**: Younggeol Cho, Gokhan Solak, Olivia Nocentini, Marta Lorenzini, Andrea Fortuna, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2509.05337)  

**Abstract**: Detecting and preventing falls in humans is a critical component of assistive robotic systems. While significant progress has been made in detecting falls, the prediction of falls before they happen, and analysis of the transient state between stability and an impending fall remain unexplored. In this paper, we propose a anticipatory fall detection method that utilizes a hybrid model combining Dynamic Graph Neural Networks (DGNN) with Long Short-Term Memory (LSTM) networks that decoupled the motion prediction and gait classification tasks to anticipate falls with high accuracy. Our approach employs real-time skeletal features extracted from video sequences as input for the proposed model. The DGNN acts as a classifier, distinguishing between three gait states: stable, transient, and fall. The LSTM-based network then predicts human movement in subsequent time steps, enabling early detection of falls. The proposed model was trained and validated using the OUMVLP-Pose and URFD datasets, demonstrating superior performance in terms of prediction error and recognition accuracy compared to models relying solely on DGNN and models from literature. The results indicate that decoupling prediction and classification improves performance compared to addressing the unified problem using only the DGNN. Furthermore, our method allows for the monitoring of the transient state, offering valuable insights that could enhance the functionality of advanced assistance systems. 

**Abstract (ZH)**: 基于动态图神经网络和长短期记忆网络的预见性跌倒检测方法 

---
# Paper2Agent: Reimagining Research Papers As Interactive and Reliable AI Agents 

**Title (ZH)**: Paper2Agent: 重塑科研论文为互动可靠的人工智能代理 

**Authors**: Jiacheng Miao, Joe R. Davis, Jonathan K. Pritchard, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2509.06917)  

**Abstract**: We introduce Paper2Agent, an automated framework that converts research papers into AI agents. Paper2Agent transforms research output from passive artifacts into active systems that can accelerate downstream use, adoption, and discovery. Conventional research papers require readers to invest substantial effort to understand and adapt a paper's code, data, and methods to their own work, creating barriers to dissemination and reuse. Paper2Agent addresses this challenge by automatically converting a paper into an AI agent that acts as a knowledgeable research assistant. It systematically analyzes the paper and the associated codebase using multiple agents to construct a Model Context Protocol (MCP) server, then iteratively generates and runs tests to refine and robustify the resulting MCP. These paper MCPs can then be flexibly connected to a chat agent (e.g. Claude Code) to carry out complex scientific queries through natural language while invoking tools and workflows from the original paper. We demonstrate Paper2Agent's effectiveness in creating reliable and capable paper agents through in-depth case studies. Paper2Agent created an agent that leverages AlphaGenome to interpret genomic variants and agents based on ScanPy and TISSUE to carry out single-cell and spatial transcriptomics analyses. We validate that these paper agents can reproduce the original paper's results and can correctly carry out novel user queries. By turning static papers into dynamic, interactive AI agents, Paper2Agent introduces a new paradigm for knowledge dissemination and a foundation for the collaborative ecosystem of AI co-scientists. 

**Abstract (ZH)**: Paper2Agent：一种将研究论文转换为AI代理的自动化框架 

---
# Test-Time Scaling in Reasoning Models Is Not Effective for Knowledge-Intensive Tasks Yet 

**Title (ZH)**: 测试时缩放在推理模型中尚未证明对知识密集型任务有效 

**Authors**: James Xu Zhao, Bryan Hooi, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06861)  

**Abstract**: Test-time scaling increases inference-time computation by allowing models to generate long reasoning chains, and has shown strong performance across many domains. However, in this work, we show that this approach is not yet effective for knowledge-intensive tasks, where high factual accuracy and low hallucination rates are essential. We conduct a comprehensive evaluation of test-time scaling using 12 reasoning models on two knowledge-intensive benchmarks. Our results reveal that increasing test-time computation does not consistently improve accuracy and, in many cases, it even leads to more hallucinations. We then analyze how extended reasoning affects hallucination behavior. We find that reduced hallucinations often result from the model choosing to abstain after thinking more, rather than from improved factual recall. Conversely, for some models, longer reasoning encourages attempts on previously unanswered questions, many of which result in hallucinations. Case studies show that extended reasoning can induce confirmation bias, leading to overconfident hallucinations. Despite these limitations, we observe that compared to non-thinking, enabling thinking remains beneficial. Code and data are available at this https URL 

**Abstract (ZH)**: 测试时扩展会增加推理时的计算量，允许模型生成长的推理链，并在许多领域显示出强大的性能。然而，在这项工作中，我们表明，在高事实准确性和低幻觉率至关重要的知识密集型任务中，这一方法尚未有效。我们使用12个推理模型在两个知识密集型基准上全面评估了测试时扩展。我们的结果表明，增加测试时的计算并不一致地提高准确性，在许多情况下甚至会导致更多的幻觉。然后我们分析了扩展推理如何影响幻觉行为。我们发现，幻觉减少通常是因为模型在思考更多后选择避免给出答案，而不是因为事实检索的改进。相反，对于一些模型，较长的推理会鼓励尝试更多之前未回答的问题，许多问题最终导致幻觉。案例研究表明，扩展推理可能导致确认偏见，从而导致过度自信的幻觉。尽管存在这些局限性，我们观察到，与不进行思考相比，使模型能够思考仍然有益。代码和数据可在以下链接获得。 

---
# MAS-Bench: A Unified Benchmark for Shortcut-Augmented Hybrid Mobile GUI Agents 

**Title (ZH)**: MAS-Bench: 一键增强混合移动GUI代理的统一基准 

**Authors**: Pengxiang Zhao, Guangyi Liu, Yaozhen Liang, Weiqing He, Zhengxi Lu, Yuehao Huang, Yaxuan Guo, Kexin Zhang, Hao Wang, Liang Liu, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06477)  

**Abstract**: To enhance the efficiency of GUI agents on various platforms like smartphones and computers, a hybrid paradigm that combines flexible GUI operations with efficient shortcuts (e.g., API, deep links) is emerging as a promising direction. However, a framework for systematically benchmarking these hybrid agents is still underexplored. To take the first step in bridging this gap, we introduce MAS-Bench, a benchmark that pioneers the evaluation of GUI-shortcut hybrid agents with a specific focus on the mobile domain. Beyond merely using predefined shortcuts, MAS-Bench assesses an agent's capability to autonomously generate shortcuts by discovering and creating reusable, low-cost workflows. It features 139 complex tasks across 11 real-world applications, a knowledge base of 88 predefined shortcuts (APIs, deep-links, RPA scripts), and 7 evaluation metrics. The tasks are designed to be solvable via GUI-only operations, but can be significantly accelerated by intelligently embedding shortcuts. Experiments show that hybrid agents achieve significantly higher success rates and efficiency than their GUI-only counterparts. This result also demonstrates the effectiveness of our method for evaluating an agent's shortcut generation capabilities. MAS-Bench fills a critical evaluation gap, providing a foundational platform for future advancements in creating more efficient and robust intelligent agents. 

**Abstract (ZH)**: MAS-Bench：面向移动领域的GUI-shortcut混合代理评估基准 

---
# HyFedRAG: A Federated Retrieval-Augmented Generation Framework for Heterogeneous and Privacy-Sensitive Data 

**Title (ZH)**: HyFedRAG：一种用于异构和隐私敏感数据的联邦检索增强生成框架 

**Authors**: Cheng Qian, Hainan Zhang, Yongxin Tong, Hong-Wei Zheng, Zhiming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06444)  

**Abstract**: Centralized RAG pipelines struggle with heterogeneous and privacy-sensitive data, especially in distributed healthcare settings where patient data spans SQL, knowledge graphs, and clinical notes. Clinicians face difficulties retrieving rare disease cases due to privacy constraints and the limitations of traditional cloud-based RAG systems in handling diverse formats and edge devices. To address this, we introduce HyFedRAG, a unified and efficient Federated RAG framework tailored for Hybrid data modalities. By leveraging an edge-cloud collaborative mechanism, HyFedRAG enables RAG to operate across diverse data sources while preserving data privacy. Our key contributions are: (1) We design an edge-cloud collaborative RAG framework built on Flower, which supports querying structured SQL data, semi-structured knowledge graphs, and unstructured documents. The edge-side LLMs convert diverse data into standardized privacy-preserving representations, and the server-side LLMs integrates them for global reasoning and generation. (2) We integrate lightweight local retrievers with privacy-aware LLMs and provide three anonymization tools that enable each client to produce semantically rich, de-identified summaries for global inference across devices. (3) To optimize response latency and reduce redundant computation, we design a three-tier caching strategy consisting of local cache, intermediate representation cache, and cloud inference cache. Experimental results on PMC-Patients demonstrate that HyFedRAG outperforms existing baselines in terms of retrieval quality, generation consistency, and system efficiency. Our framework offers a scalable and privacy-compliant solution for RAG over structural-heterogeneous data, unlocking the potential of LLMs in sensitive and diverse data environments. 

**Abstract (ZH)**: 集中式的RAG管道在处理异构和隐私敏感数据时存在困难，特别是在患者数据横跨SQL、知识图谱和临床笔记的分布式医疗环境中。临床医生因隐私限制和传统基于云的RAG系统在处理多样格式和边缘设备时的局限性，难以检索罕见疾病案例。为此，我们提出了HyFedRAG，这是一种针对混合数据模态统一且高效的联邦RAG框架。通过利用边缘-云协作机制，HyFedRAG能够在保护数据隐私的同时跨多种数据来源运行RAG。我们的主要贡献包括：（1）我们设计了一种基于Flower的边缘-云协作RAG框架，支持查询结构化SQL数据、半结构化知识图谱和非结构化文档。边缘侧的LLM将多种数据转换为标准化的隐私保护表示，服务器侧的LLM则将它们结合进行全局推理和生成。（2）我们集成了轻量级的本地检索器并与隐私保护的LLM集成，并提供了三种匿名化工具，使每个客户端能够在设备间生成语义丰富的去标识化摘要以供全局推理。（3）为了优化响应延迟并减少冗余计算，我们设计了一种三级缓存策略，包括本地缓存、中间表示缓存和云推理缓存。实验证明，HyFedRAG在检索质量、生成一致性和系统效率方面优于现有基线。我们的框架为在结构异构数据上实现RAG提供了可扩展且符合隐私要求的解决方案，并在敏感和多样化数据环境中释放了LLM的潜力。 

---
# A data-driven discretized CS:GO simulation environment to facilitate strategic multi-agent planning research 

**Title (ZH)**: 基于数据驱动的 discretized CS:GO 模拟环境，以促进战略性多Agent规划研究 

**Authors**: Yunzhe Wang, Volkan Ustun, Chris McGroarty  

**Link**: [PDF](https://arxiv.org/pdf/2509.06355)  

**Abstract**: Modern simulation environments for complex multi-agent interactions must balance high-fidelity detail with computational efficiency. We present DECOY, a novel multi-agent simulator that abstracts strategic, long-horizon planning in 3D terrains into high-level discretized simulation while preserving low-level environmental fidelity. Using Counter-Strike: Global Offensive (CS:GO) as a testbed, our framework accurately simulates gameplay using only movement decisions as tactical positioning -- without explicitly modeling low-level mechanics such as aiming and shooting. Central to our approach is a waypoint system that simplifies and discretizes continuous states and actions, paired with neural predictive and generative models trained on real CS:GO tournament data to reconstruct event outcomes. Extensive evaluations show that replays generated from human data in DECOY closely match those observed in the original game. Our publicly available simulation environment provides a valuable tool for advancing research in strategic multi-agent planning and behavior generation. 

**Abstract (ZH)**: 现代复杂多智能体交互的仿真环境需平衡高保真细节与计算效率。我们提出了DECOY多智能体仿真器，将其在3D地形上的战略、长时规划抽象为高层次离散化仿真，同时保留低层级环境保真度。使用《反恐精英：全球进攻》（CS:GO）作为实验平台，我们的框架仅通过移动决策进行战术位置模拟，即可准确再现 gameplay，无需明确建模低层级机制如瞄准和射击。我们方法的核心在于一个路径点系统，该系统简化并离散化了连续状态和动作，并配以基于真实CS:GO锦标赛数据训练的神经预测和生成模型，以重建事件结果。广泛评估表明，DECOY生成的重放与原始游戏中的观察结果高度一致。我们公开提供的仿真环境为战略多智能体规划和行为生成的研究提供了有价值的工具。 

---
# Proof2Silicon: Prompt Repair for Verified Code and Hardware Generation via Reinforcement Learning 

**Title (ZH)**: Proof2Silicon: 通过强化学习进行验证代码和硬件生成的提示修复 

**Authors**: Manvi Jha, Jiaxin Wan, Deming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.06239)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in automated code generation but frequently produce code that fails formal verification, an essential requirement for hardware and safety-critical domains. To overcome this fundamental limitation, we previously proposed PREFACE, a model-agnostic framework based on reinforcement learning (RL) that iteratively repairs the prompts provided to frozen LLMs, systematically steering them toward generating formally verifiable Dafny code without costly fine-tuning. This work presents Proof2Silicon, a novel end-to-end synthesis framework that embeds the previously proposed PREFACE flow to enable the generation of correctness-by-construction hardware directly from natural language specifications. Proof2Silicon operates by: (1) leveraging PREFACE's verifier-driven RL agent to optimize prompt generation iteratively, ensuring Dafny code correctness; (2) automatically translating verified Dafny programs into synthesizable high-level C using Dafny's Python backend and PyLog; and (3) employing Vivado HLS to produce RTL implementations. Evaluated rigorously on a challenging 100-task benchmark, PREFACE's RL-guided prompt optimization consistently improved Dafny verification success rates across diverse LLMs by up to 21%. Crucially, Proof2Silicon achieved an end-to-end hardware synthesis success rate of up to 72%, generating RTL designs through Vivado HLS synthesis flows. These results demonstrate a robust, scalable, and automated pipeline for LLM-driven, formally verified hardware synthesis, bridging natural-language specification and silicon realization. 

**Abstract (ZH)**: Large Language Models (LLMs)在自动生成代码方面的表现令人印象深刻，但经常生成无法进行形式验证的代码，而形式验证是硬件和安全关键领域的一项基本要求。为克服这一根本性限制，我们之前提出了PREFACE，这是一种基于强化学习的模型无关框架，该框架通过迭代修复冻结的LLM的提示，有系统地引导其生成形式可验证的Dafny代码，而无需昂贵的微调。本文介绍了一种名为Proof2Silicon的新颖端到端综合框架，该框架嵌入了PREFACE流程，能够直接从自然语言规范生成构造正确的硬件。Proof2Silicon通过以下步骤工作：(1)利用PREFACE的验证驱动RL代理优化提示生成，确保Dafny代码的正确性；(2)使用Dafny的Python后端和PyLog自动将验证通过的Dafny程序翻译为可综合的高层C代码；(3)使用Vivado HLS生成寄存器传输级实现。在一项具有挑战性的100任务基准测试中，PREFACE的RL引导提示优化一致地提高了各种LLM的Dafny验证成功率多达21%。关键的是，Proof2Silicon实现了从端到端硬件综合的成功率高达72%，通过Vivado HLS综合流程生成了寄存器传输级设计。这些结果表明，PREFACE为LLM驱动的形式验证硬件综合提供了一个稳健、可扩展和自动化的管道，将自然语言规范与硅实现连接起来。 

---
# Reverse-Engineered Reasoning for Open-Ended Generation 

**Title (ZH)**: 逆向工程推理以实现开放生成 

**Authors**: Haozhe Wang, Haoran Que, Qixin Xu, Minghao Liu, Wangchunshu Zhou, Jiazhan Feng, Wanjun Zhong, Wei Ye, Tong Yang, Wenhao Huang, Ge Zhang, Fangzhen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06160)  

**Abstract**: While the ``deep reasoning'' paradigm has spurred significant advances in verifiable domains like mathematics, its application to open-ended, creative generation remains a critical challenge. The two dominant methods for instilling reasoning -- reinforcement learning (RL) and instruction distillation -- falter in this area; RL struggles with the absence of clear reward signals and high-quality reward models, while distillation is prohibitively expensive and capped by the teacher model's capabilities. To overcome these limitations, we introduce REverse-Engineered Reasoning (REER), a new paradigm that fundamentally shifts the approach. Instead of building a reasoning process ``forwards'' through trial-and-error or imitation, REER works ``backwards'' from known-good solutions to computationally discover the latent, step-by-step deep reasoning process that could have produced them. Using this scalable, gradient-free approach, we curate and open-source DeepWriting-20K, a large-scale dataset of 20,000 deep reasoning trajectories for open-ended tasks. Our model, DeepWriter-8B, trained on this data, not only surpasses strong open-source baselines but also achieves performance competitive with, and at times superior to, leading proprietary models like GPT-4o and Claude 3.5. 

**Abstract (ZH)**: 虽然“深度推理”范式在可验证领域如数学取得了显著进展，其在开放性和创造性生成任务中的应用仍然是一个关键挑战。传统的两种增强推理的方法——强化学习（RL）和指令蒸馏——在这个领域遇到了困难；RL因缺乏清晰的奖励信号和高质量的奖励模型而举步维艰，而蒸馏则由于成本高昂且受到教师模型能力的限制而显得力不从心。为克服这些局限，我们提出了反向工程推理（REER）这一新的范式，从根本上改变了方法论。REER不再通过试错或模仿来构建推理过程，而是从已知良好的解决方案出发，向后发现能够生成这些解决方案的潜在的、步骤化的深度推理过程。借助这一可扩展且无梯度的方法，我们精选并开源了包含20,000个开放任务深度推理轨迹的DeepWriting-20K大数据集。基于此数据训练的DeepWriter-8B模型不仅超越了强大的开源基线，还在某些方面优于领先的专业模型，如GPT-4o和Claude 3.5。 

---
# Decision-Focused Learning Enhanced by Automated Feature Engineering for Energy Storage Optimisation 

**Title (ZH)**: 基于自动特征工程的决策导向学习促进能量存储优化 

**Authors**: Nasser Alkhulaifi, Ismail Gokay Dogan, Timothy R. Cargan, Alexander L. Bowler, Direnc Pekaslan, Nicholas J. Watson, Isaac Triguero  

**Link**: [PDF](https://arxiv.org/pdf/2509.05772)  

**Abstract**: Decision-making under uncertainty in energy management is complicated by unknown parameters hindering optimal strategies, particularly in Battery Energy Storage System (BESS) operations. Predict-Then-Optimise (PTO) approaches treat forecasting and optimisation as separate processes, allowing prediction errors to cascade into suboptimal decisions as models minimise forecasting errors rather than optimising downstream tasks. The emerging Decision-Focused Learning (DFL) methods overcome this limitation by integrating prediction and optimisation; however, they are relatively new and have been tested primarily on synthetic datasets or small-scale problems, with limited evidence of their practical viability. Real-world BESS applications present additional challenges, including greater variability and data scarcity due to collection constraints and operational limitations. Because of these challenges, this work leverages Automated Feature Engineering (AFE) to extract richer representations and improve the nascent approach of DFL. We propose an AFE-DFL framework suitable for small datasets that forecasts electricity prices and demand while optimising BESS operations to minimise costs. We validate its effectiveness on a novel real-world UK property dataset. The evaluation compares DFL methods against PTO, with and without AFE. The results show that, on average, DFL yields lower operating costs than PTO and adding AFE further improves the performance of DFL methods by 22.9-56.5% compared to the same models without AFE. These findings provide empirical evidence for DFL's practical viability in real-world settings, indicating that domain-specific AFE enhances DFL and reduces reliance on domain expertise for BESS optimisation, yielding economic benefits with broader implications for energy management systems facing similar challenges. 

**Abstract (ZH)**: 基于不确定性的能源管理决策受到未知参数的困扰，尤其是在电池储能系统（BESS）操作中。预测然后优化（PTO）方法将预测和优化视为独立的过程，这会使得预测误差积累成次优决策。新兴的决策导向学习（DFL）方法通过将预测和优化整合起来克服了这一局限，但它们相对新颖，主要在合成数据集或小型问题上进行了测试，实际可行性证据有限。实际的BESS应用增加了更多挑战，包括由于数据收集限制和运营限制导致的大得多的变异性及数据稀缺性。鉴于这些挑战，本工作利用自动特征工程（AFE）提取更丰富的表示，以改进DFL的初步方法。我们提出了一种适用于小型数据集的AFE-DFL框架，该框架既预测电价和需求，又优化BESS操作以最小化成本。我们在一个新型的英国房产实际数据集上验证了其有效性。评价包括将DFL方法与PTO方法（有和无AFE）进行比较。结果显示，DFL平均降低了运营成本，而添加AFE进一步提高了DFL方法的性能，相比无AFE模型提高了22.9%-56.5%。这些发现为DFL的实际可行性提供了实证证据，表明领域特定的AFE可以增强DFL，并减少对领域专业知识的依赖，对于面临类似挑战的能源管理系统具有经济利益和更广泛的含义。 

---
# MSRFormer: Road Network Representation Learning using Multi-scale Feature Fusion of Heterogeneous Spatial Interactions 

**Title (ZH)**: MSRFormer：利用异质空间交互多尺度特征融合的路网表示学习 

**Authors**: Jian Yang, Jiahui Wu, Li Fang, Hongchao Fan, Bianying Zhang, Huijie Zhao, Guangyi Yang, Rui Xin, Xiong You  

**Link**: [PDF](https://arxiv.org/pdf/2509.05685)  

**Abstract**: Transforming road network data into vector representations using deep learning has proven effective for road network analysis. However, urban road networks' heterogeneous and hierarchical nature poses challenges for accurate representation learning. Graph neural networks, which aggregate features from neighboring nodes, often struggle due to their homogeneity assumption and focus on a single structural scale. To address these issues, this paper presents MSRFormer, a novel road network representation learning framework that integrates multi-scale spatial interactions by addressing their flow heterogeneity and long-distance dependencies. It uses spatial flow convolution to extract small-scale features from large trajectory datasets, and identifies scale-dependent spatial interaction regions to capture the spatial structure of road networks and flow heterogeneity. By employing a graph transformer, MSRFormer effectively captures complex spatial dependencies across multiple scales. The spatial interaction features are fused using residual connections, which are fed to a contrastive learning algorithm to derive the final road network representation. Validation on two real-world datasets demonstrates that MSRFormer outperforms baseline methods in two road network analysis tasks. The performance gains of MSRFormer suggest the traffic-related task benefits more from incorporating trajectory data, also resulting in greater improvements in complex road network structures with up to 16% improvements compared to the most competitive baseline method. This research provides a practical framework for developing task-agnostic road network representation models and highlights distinct association patterns of the interplay between scale effects and flow heterogeneity of spatial interactions. 

**Abstract (ZH)**: 使用深度学习将道路网络数据转换为向量表示在道路网络分析中已被证明有效。然而，城市道路网络的异构性和层次性对准确的表示学习提出了挑战。为应对这些问题，本文提出MSRFormer，一种通过解决空间流异构性和长距离依赖性的多尺度空间交互综合框架。它利用空间流卷积从大规模轨迹数据集中提取小尺度特征，并识别尺度相关的空间交互区域以捕获道路网络的空间结构和流异构性。通过采用图变换器，MSRFormer有效捕捉了多尺度的空间依赖关系。通过残差连接融合空间交互特征，并馈送到对比学习算法以获得最终的道路网络表示。在两个真实世界数据集上的验证表明，MSRFormer在两种道路网络分析任务中优于基准方法。MSRFormer的性能提升表明，与交通相关的任务可以从结合轨迹数据中获益更多，特别是在复杂道路网络结构中可获得高达16%的性能提升，超越最竞争的基准方法。该研究提供了一种适用于多种任务的道路网络表示模型的实际框架，并突出显示了尺度效应和空间交互流异构性交互的不同关联模式。 

---
# TreeGPT: A Novel Hybrid Architecture for Abstract Syntax Tree Processing with Global Parent-Child Aggregation 

**Title (ZH)**: TreeGPT：一种新型全局父节点-子节点聚合的混合架构用于抽象语法树处理 

**Authors**: Zixi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05550)  

**Abstract**: We introduce TreeGPT, a novel neural architecture that combines transformer-based attention mechanisms with global parent-child aggregation for processing Abstract Syntax Trees (ASTs) in neural program synthesis tasks. Unlike traditional approaches that rely solely on sequential processing or graph neural networks, TreeGPT employs a hybrid design that leverages both self-attention for capturing local dependencies and a specialized Tree Feed-Forward Network (TreeFFN) for modeling hierarchical tree structures through iterative message passing.
The core innovation lies in our Global Parent-Child Aggregation mechanism, formalized as: $$h_i^{(t+1)} = \sigma \Big( h_i^{(0)} + W_{pc} \sum_{(p,c) \in E_i} f(h_p^{(t)}, h_c^{(t)}) + b \Big)$$ where $h_i^{(t)}$ represents the hidden state of node $i$ at iteration $t$, $E_i$ denotes all parent-child edges involving node $i$, and $f(h_p, h_c)$ is an edge aggregation function. This formulation enables each node to progressively aggregate information from the entire tree structure through $T$ iterations.
Our architecture integrates optional enhancements including gated aggregation with learnable edge weights, residual connections for gradient stability, and bidirectional propagation for capturing both bottom-up and top-down dependencies. We evaluate TreeGPT on the ARC Prize 2025 dataset, a challenging visual reasoning benchmark requiring abstract pattern recognition and rule inference. Experimental results demonstrate that TreeGPT achieves 96\% accuracy, significantly outperforming transformer baselines (1.3\%), large-scale models like Grok-4 (15.9\%), and specialized program synthesis methods like SOAR (52\%) while using only 1.5M parameters. Our comprehensive ablation study reveals that edge projection is the most critical component, with the combination of edge projection and gating achieving optimal performance. 

**Abstract (ZH)**: TreeGPT：结合全局父节点-子节点聚合机制的变压器架构用于神经程序合成任务中的抽象语法树处理 

---
# SynDelay: A Synthetic Dataset for Delivery Delay Prediction 

**Title (ZH)**: SynDelay: 用于配送延迟预测的合成数据集 

**Authors**: Liming Xu, Yunbo Long, Alexandra Brintrup  

**Link**: [PDF](https://arxiv.org/pdf/2509.05325)  

**Abstract**: Artificial intelligence (AI) is transforming supply chain management, yet progress in predictive tasks -- such as delivery delay prediction -- remains constrained by the scarcity of high-quality, openly available datasets. Existing datasets are often proprietary, small, or inconsistently maintained, hindering reproducibility and benchmarking. We present SynDelay, a synthetic dataset designed for delivery delay prediction. Generated using an advanced generative model trained on real-world data, SynDelay preserves realistic delivery patterns while ensuring privacy. Although not entirely free of noise or inconsistencies, it provides a challenging and practical testbed for advancing predictive modelling. To support adoption, we provide baseline results and evaluation metrics as initial benchmarks, serving as reference points rather than state-of-the-art claims. SynDelay is publicly available through the Supply Chain Data Hub, an open initiative promoting dataset sharing and benchmarking in supply chain AI. We encourage the community to contribute datasets, models, and evaluation practices to advance research in this area. All code is openly accessible at this https URL. 

**Abstract (ZH)**: 人工智能（AI）正在变革供应链管理，但在预测任务（如交付延迟预测）方面，由于高质量、公开可用的数据集匮乏，进展仍然受限。现有数据集往往具有专有性、规模小或维护不一致，阻碍了可再现性和基准测试。我们提出了SynDelay，这是一个用于交付延迟预测的合成数据集。该数据集使用基于实际数据训练的高级生成模型生成，保留了现实的交付模式并确保了隐私性。尽管不完全无噪声和无一致性，SynDelay 仍提供了一个具有挑战性和实用性的测试平台，用于推进预测建模。为了促进采用，我们提供了基准结果和评估指标作为初始基准，作为参考点而非前沿claim。SynDelay 通过供应链数据联盟公开提供，这是一个促进供应链AI领域数据集共享和基准测试的开放倡议。我们鼓励社区贡献数据集、模型和评估实践，以促进该领域的研究。所有代码均可在该网址访问。 

---
# Attention of a Kiss: Exploring Attention Maps in Video Diffusion for XAIxArts 

**Title (ZH)**: 一顿热吻的关注：探索视频扩散中的注意力图在XAIxArts中的应用 

**Authors**: Adam Cole, Mick Grierson  

**Link**: [PDF](https://arxiv.org/pdf/2509.05323)  

**Abstract**: This paper presents an artistic and technical investigation into the attention mechanisms of video diffusion transformers. Inspired by early video artists who manipulated analog video signals to create new visual aesthetics, this study proposes a method for extracting and visualizing cross-attention maps in generative video models. Built on the open-source Wan model, our tool provides an interpretable window into the temporal and spatial behavior of attention in text-to-video generation. Through exploratory probes and an artistic case study, we examine the potential of attention maps as both analytical tools and raw artistic material. This work contributes to the growing field of Explainable AI for the Arts (XAIxArts), inviting artists to reclaim the inner workings of AI as a creative medium. 

**Abstract (ZH)**: 本文对视频扩散变换器中的注意力机制进行了艺术和技术研究。受早期视频艺术家通过操控模拟视频信号创造新视觉美学的启发，本研究提出了在生成视频模型中提取和可视化交叉注意力图的方法。基于开源的Wan模型，我们的工具为文本到视频生成中注意力的时空行为提供了可解释的窗口。通过探索性探针和艺术案例研究，我们探讨了注意力图作为分析工具和原始艺术材料的潜力。本文为艺术领域的可解释人工智能（XAIxArts）领域的发展做出了贡献，邀请艺术家重新掌握人工智能内部机制作为创作媒介的权利。 

---
# Neuro-Symbolic AI for Cybersecurity: State of the Art, Challenges, and Opportunities 

**Title (ZH)**: 神经符号人工智能在网络安全领域的现状、挑战与机遇 

**Authors**: Safayat Bin Hakim, Muhammad Adil, Alvaro Velasquez, Shouhuai Xu, Houbing Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.06921)  

**Abstract**: Traditional Artificial Intelligence (AI) approaches in cybersecurity exhibit fundamental limitations: inadequate conceptual grounding leading to non-robustness against novel attacks; limited instructibility impeding analyst-guided adaptation; and misalignment with cybersecurity objectives. Neuro-Symbolic (NeSy) AI has emerged with the potential to revolutionize cybersecurity AI. However, there is no systematic understanding of this emerging approach. These hybrid systems address critical cybersecurity challenges by combining neural pattern recognition with symbolic reasoning, enabling enhanced threat understanding while introducing concerning autonomous offensive capabilities that reshape threat landscapes. In this survey, we systematically characterize this field by analyzing 127 publications spanning 2019-July 2025. We introduce a Grounding-Instructibility-Alignment (G-I-A) framework to evaluate these systems, focusing on both cyber defense and cyber offense across network security, malware analysis, and cyber operations. Our analysis shows advantages of multi-agent NeSy architectures and identifies critical implementation challenges including standardization gaps, computational complexity, and human-AI collaboration requirements that constrain deployment. We show that causal reasoning integration is the most transformative advancement, enabling proactive defense beyond correlation-based approaches. Our findings highlight dual-use implications where autonomous systems demonstrate substantial capabilities in zero-day exploitation while achieving significant cost reductions, altering threat dynamics. We provide insights and future research directions, emphasizing the urgent need for community-driven standardization frameworks and responsible development practices that ensure advancement serves defensive cybersecurity objectives while maintaining societal alignment. 

**Abstract (ZH)**: 传统人工智能在网络安全中的根本局限性包括概念基础不足导致对新型攻击的不 robust 性；可指导性有限阻碍分析师引导的适应性；以及与网络安全目标的不一致。神经符号人工智能（NeSy AI）有可能彻底改变网络安全人工智能，但对其这一新兴方法的理解尚未系统化。这些混合系统通过结合神经模式识别与符号推理，以增强对威胁的理解并引入令人关切的自主攻击能力来重新定义威胁景观，从而应对关键的网络安全挑战。在本文综述中，我们通过分析2019年至2025年间的127篇出版物，系统化地 characterizes 这个领域。我们引入了一个扎根-可指导性-一致性的框架（G-I-A框架）来评估这些系统，重点关注网络安全性、恶意软件分析和网络操作中的网络防御和网络攻击。我们的分析表明，多代理神经符号架构具有优势，并确定了关键实施挑战，如标准差距、计算复杂性和人机协作要求，这些挑战限制了部署。我们表明，因果推理整合是最具变革性的进步，能够超越相关性方法实现主动防御。我们的发现指出，自主系统在零日攻击利用方面表现出色，同时显著降低成本，改变威胁动态。我们提供了见解并为未来研究方向提出了建议，强调了社区驱动的标准框架和负责任的发展实践的迫切需求，以确保技术进步服务于防御性网络安全目标的同时保持社会共识。 

---
# Tackling the Noisy Elephant in the Room: Label Noise-robust Out-of-Distribution Detection via Loss Correction and Low-rank Decomposition 

**Title (ZH)**: 处理房间里的嘈杂大象：基于损失纠正和低秩分解的标签噪声鲁棒异分布检测 

**Authors**: Tarhib Al Azad, Shahana Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2509.06918)  

**Abstract**: Robust out-of-distribution (OOD) detection is an indispensable component of modern artificial intelligence (AI) systems, especially in safety-critical applications where models must identify inputs from unfamiliar classes not seen during training. While OOD detection has been extensively studied in the machine learning literature--with both post hoc and training-based approaches--its effectiveness under noisy training labels remains underexplored. Recent studies suggest that label noise can significantly degrade OOD performance, yet principled solutions to this issue are lacking. In this work, we demonstrate that directly combining existing label noise-robust methods with OOD detection strategies is insufficient to address this critical challenge. To overcome this, we propose a robust OOD detection framework that integrates loss correction techniques from the noisy label learning literature with low-rank and sparse decomposition methods from signal processing. Extensive experiments on both synthetic and real-world datasets demonstrate that our method significantly outperforms the state-of-the-art OOD detection techniques, particularly under severe noisy label settings. 

**Abstract (ZH)**: 稳健的离域分布（OOD）检测是现代人工智能（AI）系统不可或缺的组成部分，尤其是在涉及模型必须识别训练期间未见过的陌生类别的安全关键应用中。虽然在机器学习文献中对OOD检测进行了广泛研究，包括事后方法和训练基于方法，但在噪声训练标签下的有效性却未得到充分探索。近期研究表明，标签噪声可以显著劣化OOD性能，但针对这一问题的原理性解决方案仍然不足。在本文中，我们证明了直接将现有的标签噪声鲁棒方法与OOD检测策略结合是不足以解决这一关键挑战的。为克服这一问题，我们提出了一种将有噪标签学习文献中的损失校正技术与信号处理领域的低秩和稀疏分解方法相结合的稳健OOD检测框架。在合成数据集和真实世界数据集上的广泛实验表明，我们的方法在严重噪声标签设置下显著优于最先进的OOD检测技术。 

---
# UNH at CheckThat! 2025: Fine-tuning Vs Prompting in Claim Extraction 

**Title (ZH)**: UNH在CheckThat! 2025：细调 vs 填充文本在声明提取中的对比研究 

**Authors**: Joe Wilder, Nikhil Kadapala, Benji Xu, Mohammed Alsaadi, Aiden Parsons, Mitchell Rogers, Palash Agarwal, Adam Hassick, Laura Dietz  

**Link**: [PDF](https://arxiv.org/pdf/2509.06883)  

**Abstract**: We participate in CheckThat! Task 2 English and explore various methods of prompting and in-context learning, including few-shot prompting and fine-tuning with different LLM families, with the goal of extracting check-worthy claims from social media passages. Our best METEOR score is achieved by fine-tuning a FLAN-T5 model. However, we observe that higher-quality claims can sometimes be extracted using other methods, even when their METEOR scores are lower. 

**Abstract (ZH)**: 我们参与CheckThat! 任务2英语部分，探索各种提示方法和上下文学习方法，包括少量示例提示和不同大规模语言模型家族的微调，并旨在从社交媒体段落中提取值得验证的声明。我们的最好METEOR分数是由微调FLAN-T5模型获得的。然而，我们观察到，在某些情况下，即使METEOR分数较低，其他方法也可能提取出更高质量的声明。 

---
# AxelSMOTE: An Agent-Based Oversampling Algorithm for Imbalanced Classification 

**Title (ZH)**: 基于代理的过采样算法：AxelSMOTE 用于不平衡分类 

**Authors**: Sukumar Kishanthan, Asela Hevapathige  

**Link**: [PDF](https://arxiv.org/pdf/2509.06875)  

**Abstract**: Class imbalance in machine learning poses a significant challenge, as skewed datasets often hinder performance on minority classes. Traditional oversampling techniques, which are commonly used to alleviate class imbalance, have several drawbacks: they treat features independently, lack similarity-based controls, limit sample diversity, and fail to manage synthetic variety effectively. To overcome these issues, we introduce AxelSMOTE, an innovative agent-based approach that views data instances as autonomous agents engaging in complex interactions. Based on Axelrod's cultural dissemination model, AxelSMOTE implements four key innovations: (1) trait-based feature grouping to preserve correlations; (2) a similarity-based probabilistic exchange mechanism for meaningful interactions; (3) Beta distribution blending for realistic interpolation; and (4) controlled diversity injection to avoid overfitting. Experiments on eight imbalanced datasets demonstrate that AxelSMOTE outperforms state-of-the-art sampling methods while maintaining computational efficiency. 

**Abstract (ZH)**: 机器学习中的类别不平衡构成了一个重大挑战，因为偏斜的数据集通常会妨碍少数类别的性能。传统过采样技术虽然常被用于缓解类别不平衡，但也存在几个缺点：它们独立处理特征，缺乏基于相似性的控制，限制了样本多样性，并且无法有效地管理合成样本的多样性。为克服这些问题，我们提出了一种名为AxelSMOTE的新颖基于代理的方法，该方法将数据实例视作参与复杂交互的自主代理。基于Axelrod的文化传播模型，AxelSMOTE实现了四个关键创新：（1）基于特征的特质分组以保留相关性；（2）一种基于相似性的概率交换机制以实现有意义的交互；（3）使用Beta分布混合进行现实插值；（4）控制多样性注入以避免过拟合。在八个不平衡数据集上的实验表明，AxelSMOTE在保持计算效率的同时，优于最先进的采样方法。 

---
# floq: Training Critics via Flow-Matching for Scaling Compute in Value-Based RL 

**Title (ZH)**: FloQ: 通过流动匹配训练批评家以扩展值ベースRL的计算资源 

**Authors**: Bhavya Agrawalla, Michal Nauman, Khush Agarwal, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.06863)  

**Abstract**: A hallmark of modern large-scale machine learning techniques is the use of training objectives that provide dense supervision to intermediate computations, such as teacher forcing the next token in language models or denoising step-by-step in diffusion models. This enables models to learn complex functions in a generalizable manner. Motivated by this observation, we investigate the benefits of iterative computation for temporal difference (TD) methods in reinforcement learning (RL). Typically they represent value functions in a monolithic fashion, without iterative compute. We introduce floq (flow-matching Q-functions), an approach that parameterizes the Q-function using a velocity field and trains it using techniques from flow-matching, typically used in generative modeling. This velocity field underneath the flow is trained using a TD-learning objective, which bootstraps from values produced by a target velocity field, computed by running multiple steps of numerical integration. Crucially, floq allows for more fine-grained control and scaling of the Q-function capacity than monolithic architectures, by appropriately setting the number of integration steps. Across a suite of challenging offline RL benchmarks and online fine-tuning tasks, floq improves performance by nearly 1.8x. floq scales capacity far better than standard TD-learning architectures, highlighting the potential of iterative computation for value learning. 

**Abstract (ZH)**: 现代大规模机器学习技术的一个标志是使用提供密集监督的训练目标，例如在语言模型中通过教师强迫下一个token，在扩散模型中通过逐步去噪。这使得模型能够以可泛化的方式学习复杂的函数。受这一观察的启发，我们研究了迭代计算在强化学习（RL）中的时间差分（TD）方法中的益处。通常，它们以整体的方式表示值函数，不包含迭代计算。我们提出了floq（流匹配Q函数），这是一种使用速度场参数化Q函数并利用流匹配技术（通常用于生成建模）进行训练的方法。该速度场通过运行多次数值积分计算的目标速度场生成的值进行引导，采用TD学习目标进行训练。关键的是，floq通过适当设置积分步数，提供了比整体架构更精细的Q函数容量控制。在一系列具有挑战性的离线RL基准测试和在线微调任务中，floq将性能提高了近1.8倍。floq在Q函数容量扩展方面远远优于标准的TD学习架构，突显了迭代计算在值学习中的潜力。 

---
# Long-Range Graph Wavelet Networks 

**Title (ZH)**: 长距离图小波网络 

**Authors**: Filippo Guerranti, Fabrizio Forte, Simon Geisler, Stephan Günnemann  

**Link**: [PDF](https://arxiv.org/pdf/2509.06743)  

**Abstract**: Modeling long-range interactions, the propagation of information across distant parts of a graph, is a central challenge in graph machine learning. Graph wavelets, inspired by multi-resolution signal processing, provide a principled way to capture both local and global structures. However, existing wavelet-based graph neural networks rely on finite-order polynomial approximations, which limit their receptive fields and hinder long-range propagation. We propose Long-Range Graph Wavelet Networks (LR-GWN), which decompose wavelet filters into complementary local and global components. Local aggregation is handled with efficient low-order polynomials, while long-range interactions are captured through a flexible spectral domain parameterization. This hybrid design unifies short- and long-distance information flow within a principled wavelet framework. Experiments show that LR-GWN achieves state-of-the-art performance among wavelet-based methods on long-range benchmarks, while remaining competitive on short-range datasets. 

**Abstract (ZH)**: 长距离图波动网络（LR-GWN）在图机器学习中的应用：结合局部和全局成分实现高效长距离信息传播 

---
# MRI-Based Brain Tumor Detection through an Explainable EfficientNetV2 and MLP-Mixer-Attention Architecture 

**Title (ZH)**: 基于MRI的脑肿瘤检测：可解释的EfficientNetV2和MLP-Mixer-Attention架构 

**Authors**: Mustafa Yurdakul, Şakir Taşdemir  

**Link**: [PDF](https://arxiv.org/pdf/2509.06713)  

**Abstract**: Brain tumors are serious health problems that require early diagnosis due to their high mortality rates. Diagnosing tumors by examining Magnetic Resonance Imaging (MRI) images is a process that requires expertise and is prone to error. Therefore, the need for automated diagnosis systems is increasing day by day. In this context, a robust and explainable Deep Learning (DL) model for the classification of brain tumors is proposed. In this study, a publicly available Figshare dataset containing 3,064 T1-weighted contrast-enhanced brain MRI images of three tumor types was used. First, the classification performance of nine well-known CNN architectures was evaluated to determine the most effective backbone. Among these, EfficientNetV2 demonstrated the best performance and was selected as the backbone for further development. Subsequently, an attention-based MLP-Mixer architecture was integrated into EfficientNetV2 to enhance its classification capability. The performance of the final model was comprehensively compared with basic CNNs and the methods in the literature. Additionally, Grad-CAM visualization was used to interpret and validate the decision-making process of the proposed model. The proposed model's performance was evaluated using the five-fold cross-validation method. The proposed model demonstrated superior performance with 99.50% accuracy, 99.47% precision, 99.52% recall and 99.49% F1 score. The results obtained show that the model outperforms the studies in the literature. Moreover, Grad-CAM visualizations demonstrate that the model effectively focuses on relevant regions of MRI images, thus improving interpretability and clinical reliability. A robust deep learning model for clinical decision support systems has been obtained by combining EfficientNetV2 and attention-based MLP-Mixer, providing high accuracy and interpretability in brain tumor classification. 

**Abstract (ZH)**: 基于EfficientNetV2和注意力机制MLP-Mixer的脑肿瘤分类 robust deep learning模型 

---
# Barycentric Neural Networks and Length-Weighted Persistent Entropy Loss: A Green Geometric and Topological Framework for Function Approximation 

**Title (ZH)**: 重心神经网络和长度加权持久熵损失：一种绿色几何与拓扑函数逼近框架 

**Authors**: Victor Toscano-Duran, Rocio Gonzalez-Diaz, Miguel A. Gutiérrez-Naranjo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06694)  

**Abstract**: While it is well-established that artificial neural networks are \emph{universal approximators} for continuous functions on compact domains, many modern approaches rely on deep or overparameterized architectures that incur high computational costs. In this paper, a new type of \emph{small shallow} neural network, called the \emph{Barycentric Neural Network} ($\BNN$), is proposed, which leverages a fixed set of \emph{base points} and their \emph{barycentric coordinates} to define both its structure and its parameters. We demonstrate that our $\BNN$ enables the exact representation of \emph{continuous piecewise linear functions} ($\CPLF$s), ensuring strict continuity across segments. Since any continuous function over a compact domain can be approximated arbitrarily well by $\CPLF$s, the $\BNN$ naturally emerges as a flexible and interpretable tool for \emph{function approximation}. Beyond the use of this representation, the main contribution of the paper is the introduction of a new variant of \emph{persistent entropy}, a topological feature that is stable and scale invariant, called the \emph{length-weighted persistent entropy} ($\LWPE$), which is weighted by the lifetime of topological features. Our framework, which combines the $\BNN$ with a loss function based on our $\LWPE$, aims to provide flexible and geometrically interpretable approximations of nonlinear continuous functions in resource-constrained settings, such as those with limited base points for $\BNN$ design and few training epochs. Instead of optimizing internal weights, our approach directly \emph{optimizes the base points that define the $\BNN$}. Experimental results show that our approach achieves \emph{superior and faster approximation performance} compared to classical loss functions such as MSE, RMSE, MAE, and log-cosh. 

**Abstract (ZH)**: 一种基于基点和重心坐标的小浅层神经网络及其在受限资源环境下的非线性连续函数近似方法 

---
# TrajAware: Graph Cross-Attention and Trajectory-Aware for Generalisable VANETs under Partial Observations 

**Title (ZH)**: TrajAware: 图交叉注意力和路径aware性用于部分观测下的通用车联网 

**Authors**: Xiaolu Fu, Ziyuan Bao, Eiman Kanjo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06665)  

**Abstract**: Vehicular ad hoc networks (VANETs) are a crucial component of intelligent transportation systems; however, routing remains challenging due to dynamic topologies, incomplete observations, and the limited resources of edge devices. Existing reinforcement learning (RL) approaches often assume fixed graph structures and require retraining when network conditions change, making them unsuitable for deployment on constrained hardware. We present TrajAware, an RL-based framework designed for edge AI deployment in VANETs. TrajAware integrates three components: (i) action space pruning, which reduces redundant neighbour options while preserving two-hop reachability, alleviating the curse of dimensionality; (ii) graph cross-attention, which maps pruned neighbours to the global graph context, producing features that generalise across diverse network sizes; and (iii) trajectory-aware prediction, which uses historical routes and junction information to estimate real-time positions under partial observations. We evaluate TrajAware in the open-source SUMO simulator using real-world city maps with a leave-one-city-out setup. Results show that TrajAware achieves near-shortest paths and high delivery ratios while maintaining efficiency suitable for constrained edge devices, outperforming state-of-the-art baselines in both full and partial observation scenarios. 

**Abstract (ZH)**: 基于轨迹感知的VANET边缘AI路由框架TrajAware 

---
# AnalysisGNN: Unified Music Analysis with Graph Neural Networks 

**Title (ZH)**: AnalysisGNN: 基于图神经网络的统一音乐分析 

**Authors**: Emmanouil Karystinaios, Johannes Hentschel, Markus Neuwirth, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.06654)  

**Abstract**: Recent years have seen a boom in computational approaches to music analysis, yet each one is typically tailored to a specific analytical domain. In this work, we introduce AnalysisGNN, a novel graph neural network framework that leverages a data-shuffling strategy with a custom weighted multi-task loss and logit fusion between task-specific classifiers to integrate heterogeneously annotated symbolic datasets for comprehensive score analysis. We further integrate a Non-Chord-Tone prediction module, which identifies and excludes passing and non-functional notes from all tasks, thereby improving the consistency of label signals. Experimental evaluations demonstrate that AnalysisGNN achieves performance comparable to traditional static-dataset approaches, while showing increased resilience to domain shifts and annotation inconsistencies across multiple heterogeneous corpora. 

**Abstract (ZH)**: 近年来，音乐分析的计算方法取得了蓬勃发展，但每一种方法通常仅针对特定的分析领域。在此工作中，我们提出了一种新型的图神经网络框架——AnalysisGNN，该框架利用数据打乱策略和自定义加权多任务损失以及任务特定分类器之间的 logits 融合，以综合集成异构标注的符号数据集，实现全面的乐谱分析。我们进一步整合了一个非和弦音预测模块，该模块能够识别并排除所有任务中的经过音和非功能音，从而提高标签信号的一致性。实验评估表明，AnalysisGNN 在多项异构数据集上实现了与传统静态数据集方法相当的性能，并且在领域转换和标注不一致性方面显示出更高的鲁棒性。 

---
# The First Voice Timbre Attribute Detection Challenge 

**Title (ZH)**: 首次声音音色属性检测挑战 

**Authors**: Liping Chen, Jinghao He, Zhengyan Sheng, Kong Aik Lee, Zhen-Hua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2509.06635)  

**Abstract**: The first voice timbre attribute detection challenge is featured in a special session at NCMMSC 2025. It focuses on the explainability of voice timbre and compares the intensity of two speech utterances in a specified timbre descriptor dimension. The evaluation was conducted on the VCTK-RVA dataset. Participants developed their systems and submitted their outputs to the organizer, who evaluated the performance and sent feedback to them. Six teams submitted their outputs, with five providing descriptions of their methodologies. 

**Abstract (ZH)**: 首届声音音色属性检测挑战赛在2025年NCMMSC会议的特别研讨会中举办。该挑战关注声音音色的可解释性，并在指定的音色描述维度上比较两个语音陈述的强度。评估基于VCTK-RVA数据集进行。参与者开发了系统并提交了输出，组织者评估了性能并给予了反馈。六支队伍提交了输出，其中五支提供了其方法论的描述。 

---
# BEAM: Brainwave Empathy Assessment Model for Early Childhood 

**Title (ZH)**: BEAM: 脑电同理心评估模型 for 早期 childhood 

**Authors**: Chen Xie, Gaofeng Wu, Kaidong Wang, Zihao Zhu, Xiaoshu Luo, Yan Liang, Feiyu Quan, Ruoxi Wu, Xianghui Huang, Han Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06620)  

**Abstract**: Empathy in young children is crucial for their social and emotional development, yet predicting it remains challenging. Traditional methods often only rely on self-reports or observer-based labeling, which are susceptible to bias and fail to objectively capture the process of empathy formation. EEG offers an objective alternative; however, current approaches primarily extract static patterns, neglecting temporal dynamics. To overcome these limitations, we propose a novel deep learning framework, the Brainwave Empathy Assessment Model (BEAM), to predict empathy levels in children aged 4-6 years. BEAM leverages multi-view EEG signals to capture both cognitive and emotional dimensions of empathy. The framework comprises three key components: 1) a LaBraM-based encoder for effective spatio-temporal feature extraction, 2) a feature fusion module to integrate complementary information from multi-view signals, and 3) a contrastive learning module to enhance class separation. Validated on the CBCP dataset, BEAM outperforms state-of-the-art methods across multiple metrics, demonstrating its potential for objective empathy assessment and providing a preliminary insight into early interventions in children's prosocial development. 

**Abstract (ZH)**: 幼儿期共情对于其社会和情感发展至关重要，但对其预测仍然具有挑战性。传统的预测方法通常仅依赖自我报告或观察者基于的标签，这些方法容易引入偏差且无法客观捕捉共情形成的过程。虽然脑电图（EEG）可以提供一种客观替代方案，但当前方法主要提取静态模式，忽略了时间动态性。为克服这些局限性，我们提出了一种新的深度学习框架——脑波共情评估模型（BEAM），以预测4-6岁儿童的共情水平。BEAM 利用多视角的脑电信号来捕捉共情的认知和情感维度。该框架包括三个关键组成部分：1）基于LabraM的编码器以有效地进行时空特征提取，2）特征融合模块以整合多视角信号中的互补信息，3）对比学习模块以增强类别分离。BEAM 在CBCP 数据集上验证，其在多个指标上优于现有方法，展示了其在客观共情评估方面的潜力，并为儿童亲社会发展早期干预提供了初步见解。 

---
# Integrated Detection and Tracking Based on Radar Range-Doppler Feature 

**Title (ZH)**: 基于雷达距离-多普勒特征的综合检测与跟踪 

**Authors**: Chenyu Zhang, Yuanhang Wu, Xiaoxi Ma, Wei Yi  

**Link**: [PDF](https://arxiv.org/pdf/2509.06569)  

**Abstract**: Detection and tracking are the basic tasks of radar systems. Current joint detection tracking methods, which focus on dynamically adjusting detection thresholds from tracking results, still present challenges in fully utilizing the potential of radar signals. These are mainly reflected in the limited capacity of the constant false-alarm rate model to accurately represent information, the insufficient depiction of complex scenes, and the limited information acquired by the tracker. We introduce the Integrated Detection and Tracking based on radar feature (InDT) method, which comprises a network architecture for radar signal detection and a tracker that leverages detection assistance. The InDT detector extracts feature information from each Range-Doppler (RD) matrix and then returns the target position through the feature enhancement module and the detection head. The InDT tracker adaptively updates the measurement noise covariance of the Kalman filter based on detection confidence. The similarity of target RD features is measured by cosine distance, which enhances the data association process by combining location and feature information. Finally, the efficacy of the proposed method was validated through testing on both simulated data and publicly available datasets. 

**Abstract (ZH)**: 基于雷达特征的综合检测与跟踪方法（InDT） 

---
# Contrastive Self-Supervised Network Intrusion Detection using Augmented Negative Pairs 

**Title (ZH)**: 对比自监督网络入侵检测使用增强负样本对 

**Authors**: Jack Wilkie, Hanan Hindy, Christos Tachtatzis, Robert Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2509.06550)  

**Abstract**: Network intrusion detection remains a critical challenge in cybersecurity. While supervised machine learning models achieve state-of-the-art performance, their reliance on large labelled datasets makes them impractical for many real-world applications. Anomaly detection methods, which train exclusively on benign traffic to identify malicious activity, suffer from high false positive rates, limiting their usability. Recently, self-supervised learning techniques have demonstrated improved performance with lower false positive rates by learning discriminative latent representations of benign traffic. In particular, contrastive self-supervised models achieve this by minimizing the distance between similar (positive) views of benign traffic while maximizing it between dissimilar (negative) views. Existing approaches generate positive views through data augmentation and treat other samples as negative. In contrast, this work introduces Contrastive Learning using Augmented Negative pairs (CLAN), a novel paradigm for network intrusion detection where augmented samples are treated as negative views - representing potentially malicious distributions - while other benign samples serve as positive views. This approach enhances both classification accuracy and inference efficiency after pretraining on benign traffic. Experimental evaluation on the Lycos2017 dataset demonstrates that the proposed method surpasses existing self-supervised and anomaly detection techniques in a binary classification task. Furthermore, when fine-tuned on a limited labelled dataset, the proposed approach achieves superior multi-class classification performance compared to existing self-supervised models. 

**Abstract (ZH)**: 网络入侵检测仍然是网络安全中的一个关键挑战。虽然监督机器学习模型取得了最先进的性能，但它们对大型标注数据集的依赖使其在许多实际应用中 impractical。仅通过良性流量进行训练以识别恶意活动的异常检测方法遭受高误报率的限制，这限制了它们的应用。最近，自我监督学习技术通过学习良性流量的判别潜在表示，展现了改进的性能和较低的误报率。特别是对比自我监督模型通过最小化类似（正面）良性流量视图之间的距离，同时最大化不相似（负面）视图之间的距离来实现这一点。现有方法通过数据增强生成正面视图，并将其他样本视为负面。相比之下，本工作引入了使用增强负样本对的对比学习（CLAN），这是一种新颖的网络入侵检测范式，在良性流量上预训练后，增强样本被视作负面视图——代表潜在恶意流量分布，而其他良性样本作为正面视图。此方法在预训练后提高了分类准确性和推理效率。在 Lycos2017 数据集上的实验评估表明，所提出的方法在二分类任务中超过了现有的自我监督和异常检测技术。此外，当在有限的标注数据集上进行微调时，所提出的方法在多分类任务中表现优于现有自我监督模型。 

---
# Signal-Based Malware Classification Using 1D CNNs 

**Title (ZH)**: 基于信号的恶意软件分类方法研究：采用1D CNNs 

**Authors**: Jack Wilkie, Hanan Hindy, Ivan Andonovic, Christos Tachtatzis, Robert Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2509.06548)  

**Abstract**: Malware classification is a contemporary and ongoing challenge in cyber-security: modern obfuscation techniques are able to evade traditional static analysis, while dynamic analysis is too resource intensive to be deployed at a large scale. One prominent line of research addresses these limitations by converting malware binaries into 2D images by heuristically reshaping them into a 2D grid before resizing using Lanczos resampling. These images can then be classified based on their textural information using computer vision approaches. While this approach can detect obfuscated malware more effectively than static analysis, the process of converting files into 2D images results in significant information loss due to both quantisation noise, caused by rounding to integer pixel values, and the introduction of 2D dependencies which do not exist in the original data. This loss of signal limits the classification performance of the downstream model. This work addresses these weaknesses by instead resizing the files into 1D signals which avoids the need for heuristic reshaping, and additionally these signals do not suffer from quantisation noise due to being stored in a floating-point format. It is shown that existing 2D CNN architectures can be readily adapted to classify these 1D signals for improved performance. Furthermore, a bespoke 1D convolutional neural network, based on the ResNet architecture and squeeze-and-excitation layers, was developed to classify these signals and evaluated on the MalNet dataset. It was found to achieve state-of-the-art performance on binary, type, and family level classification with F1 scores of 0.874, 0.503, and 0.507, respectively, paving the way for future models to operate on the proposed signal modality. 

**Abstract (ZH)**: 恶意软件分类是网络安全领域的一个当代和持续性挑战：现代混淆技术能够规避传统静态分析，而动态分析因资源密集型而不易大规模部署。一条突出的研究路线通过启发式地将恶意软件二进制文件重塑为2D网格，然后使用兰契兹重采样进行调整，将文件转换成2D图像。随后可以利用计算机视觉方法根据纹理信息对这些图像进行分类。尽管这种方法比静态分析更能检测被混淆的恶意软件，但将文件转换成2D图像的过程会导致因量化噪声（由舍入到整数像素值引起）和引入不存在于原始数据中的2D依赖关系而导致的信息显著丢失，而这限制了后续模型的分类性能。本研究通过将文件直接调整为1D信号来解决这些弱点，避免了启发式重塑的需要，而且由于以浮点格式存储，这些信号不会遭受量化噪声的影响。研究表明，现有的2D CNN架构可以轻松适配来分类这些1D信号从而获得更好的性能。此外，基于ResNet架构和挤压-激励层开发了一种专门的1D卷积神经网络，并在MalNet数据集上进行了评估，实现了二进制、类型和家族水平分类的最佳性能，F1得分分别为0.874、0.503和0.507，为未来的模型提供了一种新的信号模态操作的途径。 

---
# Learning Optimal Defender Strategies for CAGE-2 using a POMDP Model 

**Title (ZH)**: 基于POMDP模型学习CAGE-2的最佳防御策略 

**Authors**: Duc Huy Le, Rolf Stadler  

**Link**: [PDF](https://arxiv.org/pdf/2509.06539)  

**Abstract**: CAGE-2 is an accepted benchmark for learning and evaluating defender strategies against cyberattacks. It reflects a scenario where a defender agent protects an IT infrastructure against various attacks. Many defender methods for CAGE-2 have been proposed in the literature. In this paper, we construct a formal model for CAGE-2 using the framework of Partially Observable Markov Decision Process (POMDP). Based on this model, we define an optimal defender strategy for CAGE-2 and introduce a method to efficiently learn this strategy. Our method, called BF-PPO, is based on PPO, and it uses particle filter to mitigate the computational complexity due to the large state space of the CAGE-2 model. We evaluate our method in the CAGE-2 CybORG environment and compare its performance with that of CARDIFF, the highest ranked method on the CAGE-2 leaderboard. We find that our method outperforms CARDIFF regarding the learned defender strategy and the required training time. 

**Abstract (ZH)**: CAGE-2是学习和评估网络防御策略的公认的基准，它反映了防御代理保护IT基础设施免受各种攻击的场景。文献中提出了许多针对CAGE-2的防御方法。在本文中，我们使用部分可观测马尔可夫决策过程（POMDP）框架构建了CAGE-2的正式模型，并在此基础上定义了CAGE-2的最优防御策略，并介绍了一种高效学习该策略的方法。我们的方法称为BF-PPO，基于PPO，并使用粒子滤波来缓解由于CAGE-2模型状态空间庞大而导致的计算复杂性。我们在CAGE-2 CybORG环境中评估了该方法，并将其性能与CAGE-2榜单上最高排名的CARDIFF方法进行了比较。我们发现，我们的方法在学习到的防御策略和所需的训练时间方面优于CARDIFF。 

---
# On the Reproducibility of "FairCLIP: Harnessing Fairness in Vision-Language Learning'' 

**Title (ZH)**: “FairCLIP： Harnessing Fairness in Vision-Language Learning” 的再现性研究 

**Authors**: Hua Chang Bakker, Stan Fris, Angela Madelon Bernardy, Stan Deutekom  

**Link**: [PDF](https://arxiv.org/pdf/2509.06535)  

**Abstract**: We investigated the reproducibility of FairCLIP, proposed by Luo et al. (2024), for improving the group fairness of CLIP (Radford et al., 2021) by minimizing image-text similarity score disparities across sensitive groups using the Sinkhorn distance. The experimental setup of Luo et al. (2024) was reproduced to primarily investigate the research findings for FairCLIP. The model description by Luo et al. (2024) was found to differ from the original implementation. Therefore, a new implementation, A-FairCLIP, is introduced to examine specific design choices. Furthermore, FairCLIP+ is proposed to extend the FairCLIP objective to include multiple attributes. Additionally, the impact of the distance minimization on FairCLIP's fairness and performance was explored. In alignment with the original authors, CLIP was found to be biased towards certain demographics when applied to zero-shot glaucoma classification using medical scans and clinical notes from the Harvard-FairVLMed dataset. However, the experimental results on two datasets do not support their claim that FairCLIP improves the performance and fairness of CLIP. Although the regularization objective reduces Sinkhorn distances, both the official implementation and the aligned implementation, A-FairCLIP, were not found to improve performance nor fairness in zero-shot glaucoma classification. 

**Abstract (ZH)**: 我们调查了由Luo等（2024）提出的FairCLIP在通过最小化敏感群体间的图像-文本相似性得分差异来提高CLIP（Radford等，2021）的分组公平性方面的可再现性，使用Sinkhorn距离。我们复制了Luo等（2024）的研究设置，主要研究FairCLIP的研究发现。发现Luo等（2024）的模型描述与原始实现不同，因此提出了一种新的实现A-FairCLIP，以检查特定的设计选择。此外，我们提出了FairCLIP+，将其公平性目标扩展到包括多个属性。我们还探讨了距离最小化对FairCLIP的公平性和性能的影响。与原始作者一致，我们发现当使用哈佛-FairVLMed数据集中的医疗影像和临床笔记进行零样本青光眼分类时，CLIP偏向于某些人口统计学特征。然而，两个数据集的实验结果并不支持FairCLIP改善CLIP的性能和公平性的主张。尽管正则化目标减少了Sinkhorn距离，官方实现和对齐实现A-FairCLIP均未发现对零样本青光眼分类的性能和公平性有任何改进。 

---
# DyC-STG: Dynamic Causal Spatio-Temporal Graph Network for Real-time Data Credibility Analysis in IoT 

**Title (ZH)**: DyC-STG：动态因果时空图网络在物联网实时数据可信性分析中的应用 

**Authors**: Guanjie Cheng, Boyi Li, Peihan Wu, Feiyi Chen, Xinkui Zhao, Mengying Zhu, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06483)  

**Abstract**: The wide spreading of Internet of Things (IoT) sensors generates vast spatio-temporal data streams, but ensuring data credibility is a critical yet unsolved challenge for applications like smart homes. While spatio-temporal graph (STG) models are a leading paradigm for such data, they often fall short in dynamic, human-centric environments due to two fundamental limitations: (1) their reliance on static graph topologies, which fail to capture physical, event-driven dynamics, and (2) their tendency to confuse spurious correlations with true causality, undermining robustness in human-centric environments. To address these gaps, we propose the Dynamic Causal Spatio-Temporal Graph Network (DyC-STG), a novel framework designed for real-time data credibility analysis in IoT. Our framework features two synergistic contributions: an event-driven dynamic graph module that adapts the graph topology in real-time to reflect physical state changes, and a causal reasoning module to distill causally-aware representations by strictly enforcing temporal precedence. To facilitate the research in this domain we release two new real-world datasets. Comprehensive experiments show that DyC-STG establishes a new state-of-the-art, outperforming the strongest baselines by 1.4 percentage points and achieving an F1-Score of up to 0.930. 

**Abstract (ZH)**: 物联网传感器的广泛普及生成了大量的时空数据流，但在如智能家庭等应用中确保数据的可信度仍是一个关键且未解决的挑战。虽然时空图（STG）模型是处理此类数据的主要范式，但在动态的人本环境中，它们常常受限于两个根本性的局限性：（1）依赖静态的图形拓扑，这无法捕捉物理和事件驱动的动力学，（2）倾向于将虚假的相关性误认为真正的原因，从而在人本环境中削弱了鲁棒性。为了解决这些不足，我们提出了一种名为动态因果时空图网络（DyC-STG）的新颖框架，专门用于物联网中的实时数据可信度分析。该框架包含两个协同贡献：一种事件驱动的动态图模块，能够实时调整图形拓扑以反映物理状态变化，以及一种因果推理模块，通过严格遵守时间顺序来提取因果感知的表示。为促进该领域的研究，我们发布了两个新的现实世界数据集。全面的实验表明，DyC-STG 达到了新的最佳水平，在最强基线的基础上提高了 1.4 个百分点，并实现了高达 0.930 的 F1 分数。 

---
# Explained, yet misunderstood: How AI Literacy shapes HR Managers' interpretation of User Interfaces in Recruiting Recommender Systems 

**Title (ZH)**: 解释 yet 误解：AI 文盲如何影响人力资源经理对招聘推荐系统用户界面的解读 

**Authors**: Yannick Kalff, Katharina Simbeck  

**Link**: [PDF](https://arxiv.org/pdf/2509.06475)  

**Abstract**: AI-based recommender systems increasingly influence recruitment decisions. Thus, transparency and responsible adoption in Human Resource Management (HRM) are critical. This study examines how HR managers' AI literacy influences their subjective perception and objective understanding of explainable AI (XAI) elements in recruiting recommender dashboards. In an online experiment, 410 German-based HR managers compared baseline dashboards to versions enriched with three XAI styles: important features, counterfactuals, and model criteria. Our results show that the dashboards used in practice do not explain AI results and even keep AI elements opaque. However, while adding XAI features improves subjective perceptions of helpfulness and trust among users with moderate or high AI literacy, it does not increase their objective understanding. It may even reduce accurate understanding, especially with complex explanations. Only overlays of important features significantly aided the interpretations of high-literacy users. Our findings highlight that the benefits of XAI in recruitment depend on users' AI literacy, emphasizing the need for tailored explanation strategies and targeted literacy training in HRM to ensure fair, transparent, and effective adoption of AI. 

**Abstract (ZH)**: 基于AI的推荐系统日益影响招聘决策。因此，人力资源管理中透明性和负责任的采用至关重要。本研究探讨了人力资源经理的AI素养如何影响他们对招聘推荐仪表板中可解释AI（XAI）元素的主观感知和客观理解。通过一项在线实验，410名德国人力资源经理将基线仪表板与增加了三种XAI风格（关键特征、反事实和模型标准）的版本进行了比较。结果显示，实践中使用的仪表板并未解释AI结果，甚至使AI元素变得不透明。然而，尽管增加了XAI功能可以改善中等或高水平AI素养用户对帮助性和可信度的主观感知，但并没有提高他们的客观理解能力。甚至可能降低准确理解，尤其是在复杂解释的情况下。只有关键特征的覆盖层显著帮助高水平素养用户进行解释。我们的研究结果强调，招聘中的XAI益处取决于用户AI素养，强调了在人力资源管理中制定针对性解释策略和能力培训的必要性，以确保AI的公平、透明和有效采用。 

---
# Several Performance Bounds on Decentralized Online Optimization are Highly Conservative and Potentially Misleading 

**Title (ZH)**: 几个去中心化在线优化的性能上限可能极具保守性且可能存在误导性 

**Authors**: Erwan Meunier, Julien M. Hendrickx  

**Link**: [PDF](https://arxiv.org/pdf/2509.06466)  

**Abstract**: We analyze Decentralized Online Optimization algorithms using the Performance Estimation Problem approach which allows, to automatically compute exact worst-case performance of optimization algorithms. Our analysis shows that several available performance guarantees are very conservative, sometimes by multiple orders of magnitude, and can lead to misguided choices of algorithm. Moreover, at least in terms of worst-case performance, some algorithms appear not to benefit from inter-agent communications for a significant period of time. We show how to improve classical methods by tuning their step-sizes, and find that we can save up to 20% on their actual worst-case performance regret. 

**Abstract (ZH)**: 我们使用性能估计问题方法分析去中心化在线优化算法，该方法允许自动计算优化算法的精确最坏情况性能。我们的分析表明，多种可用的性能保证非常保守，有时相差几个数量级，并可能导致对算法的选择产生误导。此外，至少从最坏情况性能的角度来看，某些算法似乎在相当长的一段时间内并未从代理间的通信中受益。我们展示了如何通过调整步长改进经典方法，并发现可以将其实际最坏情况性能后悔值节省多达20%。 

---
# HECATE: An ECS-based Framework for Teaching and Developing Multi-Agent Systems 

**Title (ZH)**: HECATE：基于ECS的多Agent系统教学与开发框架 

**Authors**: Arthur Casals, Anarosa A. F. Brandão  

**Link**: [PDF](https://arxiv.org/pdf/2509.06431)  

**Abstract**: This paper introduces HECATE, a novel framework based on the Entity-Component-System (ECS) architectural pattern that bridges the gap between distributed systems engineering and MAS development. HECATE is built using the Entity-Component-System architectural pattern, leveraging data-oriented design to implement multiagent systems. This approach involves engineering multiagent systems (MAS) from a distributed systems (DS) perspective, integrating agent concepts directly into the DS domain. This approach simplifies MAS development by (i) reducing the need for specialized agent knowledge and (ii) leveraging familiar DS patterns and standards to minimize the agent-specific knowledge required for engineering MAS. We present the framework's architecture, core components, and implementation approach, demonstrating how it supports different agent models. 

**Abstract (ZH)**: 基于实体-组件-系统架构模式的HECATE框架：分布式系统工程与多Agent系统开发的桥梁 

---
# CAPMix: Robust Time Series Anomaly Detection Based on Abnormal Assumptions with Dual-Space Mixup 

**Title (ZH)**: CAPMix：基于异常假设的双空间混合时间序列异常检测 

**Authors**: Xudong Mou, Rui Wang, Tiejun Wang, Renyu Yang, Shiru Chen, Jie Sun, Tianyu Wo, Xudong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06419)  

**Abstract**: Time series anomaly detection (TSAD) is a vital yet challenging task, particularly in scenarios where labeled anomalies are scarce and temporal dependencies are complex. Recent anomaly assumption (AA) approaches alleviate the lack of anomalies by injecting synthetic samples and training discriminative models. Despite promising results, these methods often suffer from two fundamental limitations: patchy generation, where scattered anomaly knowledge leads to overly simplistic or incoherent anomaly injection, and Anomaly Shift, where synthetic anomalies either resemble normal data too closely or diverge unrealistically from real anomalies, thereby distorting classification boundaries. In this paper, we propose CAPMix, a controllable anomaly augmentation framework that addresses both issues. First, we design a CutAddPaste mechanism to inject diverse and complex anomalies in a targeted manner, avoiding patchy generation. Second, we introduce a label revision strategy to adaptively refine anomaly labels, reducing the risk of anomaly shift. Finally, we employ dual-space mixup within a temporal convolutional network to enforce smoother and more robust decision boundaries. Extensive experiments on five benchmark datasets, including AIOps, UCR, SWaT, WADI, and ESA, demonstrate that CAPMix achieves significant improvements over state-of-the-art baselines, with enhanced robustness against contaminated training data. The code is available at this https URL. 

**Abstract (ZH)**: 可控异常增强框架CAPMix：解决异常生成片段化和异常偏移问题 

---
# Index-Preserving Lightweight Token Pruning for Efficient Document Understanding in Vision-Language Models 

**Title (ZH)**: 基于索引保留的轻量级 token 裁剪以实现高效的视觉-语言模型文档理解 

**Authors**: Jaemin Son, Sujin Choi, Inyong Yun  

**Link**: [PDF](https://arxiv.org/pdf/2509.06415)  

**Abstract**: Recent progress in vision-language models (VLMs) has led to impressive results in document understanding tasks, but their high computational demands remain a challenge. To mitigate the compute burdens, we propose a lightweight token pruning framework that filters out non-informative background regions from document images prior to VLM processing. A binary patch-level classifier removes non-text areas, and a max-pooling refinement step recovers fragmented text regions to enhance spatial coherence. Experiments on real-world document datasets demonstrate that our approach substantially lowers computational costs, while maintaining comparable accuracy. 

**Abstract (ZH)**: Recent进展在视觉-语言模型中的最新进展已经在文档理解任务中取得了令人印象深刻的成果，但它们的高计算需求仍然是一个挑战。为了缓解计算负担，我们提出了一种轻量级的令牌剪枝框架，在视觉-语言模型处理之前，过滤掉文档图像中的非信息性背景区域。二进制patches级分类器移除非文本区域，最大池化精炼步骤恢复片段化的文本区域以增强空间一致性。实验结果表明，我们的方法显著降低了计算成本，同时保持了相当的准确性。 

---
# Beyond the Pre-Service Horizon: Infusing In-Service Behavior for Improved Financial Risk Forecasting 

**Title (ZH)**: 超越入职视野：注入在职行为以提高财务风险预测 

**Authors**: Senhao Liu, Zhiyu Guo, Zhiyuan Ji, Yueguo Chen, Yateng Tang, Yunhai Wang, Xuehao Zheng, Xiang Ao  

**Link**: [PDF](https://arxiv.org/pdf/2509.06385)  

**Abstract**: Typical financial risk management involves distinct phases for pre-service risk assessment and in-service default detection, often modeled separately. This paper proposes a novel framework, Multi-Granularity Knowledge Distillation (abbreviated as MGKD), aimed at improving pre-service risk prediction through the integration of in-service user behavior data. MGKD follows the idea of knowledge distillation, where the teacher model, trained on historical in-service data, guides the student model, which is trained on pre-service data. By using soft labels derived from in-service data, the teacher model helps the student model improve its risk prediction prior to service activation. Meanwhile, a multi-granularity distillation strategy is introduced, including coarse-grained, fine-grained, and self-distillation, to align the representations and predictions of the teacher and student models. This approach not only reinforces the representation of default cases but also enables the transfer of key behavioral patterns associated with defaulters from the teacher to the student model, thereby improving the overall performance of pre-service risk assessment. Moreover, we adopt a re-weighting strategy to mitigate the model's bias towards the minority class. Experimental results on large-scale real-world datasets from Tencent Mobile Payment demonstrate the effectiveness of our proposed approach in both offline and online scenarios. 

**Abstract (ZH)**: 面向服务的多粒度知识蒸馏框架：通过集成在服务中用户行为数据以改进预服务风险预测 

---
# MRD-LiNet: A Novel Lightweight Hybrid CNN with Gradient-Guided Unlearning for Improved Drought Stress Identification 

**Title (ZH)**: MRD-LiNet: 一种带有梯度导向遗忘的新型轻量级混合CNN及其在改善旱灾 stress 识别中的应用 

**Authors**: Aswini Kumar Patra, Lingaraj Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06367)  

**Abstract**: Drought stress is a major threat to global crop productivity, making its early and precise detection essential for sustainable agricultural management. Traditional approaches, though useful, are often time-consuming and labor-intensive, which has motivated the adoption of deep learning methods. In recent years, Convolutional Neural Network (CNN) and Vision Transformer architectures have been widely explored for drought stress identification; however, these models generally rely on a large number of trainable parameters, restricting their use in resource-limited and real-time agricultural settings. To address this challenge, we propose a novel lightweight hybrid CNN framework inspired by ResNet, DenseNet, and MobileNet architectures. The framework achieves a remarkable 15-fold reduction in trainable parameters compared to conventional CNN and Vision Transformer models, while maintaining competitive accuracy. In addition, we introduce a machine unlearning mechanism based on a gradient norm-based influence function, which enables targeted removal of specific training data influence, thereby improving model adaptability. The method was evaluated on an aerial image dataset of potato fields with expert-annotated healthy and drought-stressed regions. Experimental results show that our framework achieves high accuracy while substantially lowering computational costs. These findings highlight its potential as a practical, scalable, and adaptive solution for drought stress monitoring in precision agriculture, particularly under resource-constrained conditions. 

**Abstract (ZH)**: 干旱压力是全球作物产量的主要威胁，其早期和精准检测对于可持续农业管理至关重要。传统方法虽然有用，但往往耗时且劳动密集，这促使了深度学习方法的应用。近年来，卷积神经网络（CNN）和视觉变换器架构广泛用于干旱压力识别；然而，这些模型通常依赖大量的可训练参数，限制了其在资源受限和实时农业生产环境中的应用。为解决这一挑战，我们提出了一种受ResNet、DenseNet和MobileNet架构启发的新型轻量级混合CNN框架。该框架与传统CNN和视觉变换器模型相比，实现了可训练参数15倍的减少，同时保持了竞争力的准确性。此外，我们引入了一种基于梯度范数影响函数的机器遗忘机制，能够针对性地去除特定训练数据的影响，从而提高模型的适应性。该方法在具有专家注释的健康和干旱胁迫区域的马铃薯田空中图像数据集上进行了评估。实验结果表明，我们的框架在显著降低计算成本的同时实现了高精度。这些发现突显了其在资源受限条件下精准农业中作为实用、可扩展和适应性干旱胁迫监测解决方案的潜力。 

---
# PL-CA: A Parametric Legal Case Augmentation Framework 

**Title (ZH)**: PL-CA: 一种参数化法律案例扩充框架 

**Authors**: Ao Chang, Yubo Chen, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.06356)  

**Abstract**: Conventional RAG is considered one of the most effective methods for addressing model knowledge insufficiency and hallucination, particularly in the judicial domain that requires high levels of knowledge rigor, logical consistency, and content integrity. However, the conventional RAG method only injects retrieved documents directly into the model's context, which severely constrains models due to their limited context windows and introduces additional computational overhead through excessively long contexts, thereby disrupting models' attention and degrading performance on downstream tasks. Moreover, many existing benchmarks lack expert annotation and focus solely on individual downstream tasks while real-world legal scenarios consist of multiple mixed legal tasks, indicating conventional benchmarks' inadequacy for reflecting models' true capabilities. To address these limitations, we propose PL-CA, which introduces a parametric RAG (P-RAG) framework to perform data augmentation on corpus knowledge and encode this legal knowledge into parametric vectors, and then integrates this parametric knowledge into the LLM's feed-forward networks (FFN) via LoRA, thereby alleviating models' context pressure. Additionally, we also construct a multi-task legal dataset comprising more than 2000 training and test instances, which are all expert-annotated and manually verified. We conduct our experiments on our dataset, and the experimental results demonstrate that our method reduces the overhead associated with excessively long contexts while maintaining competitive performance on downstream tasks compared to conventional RAG. Our code and dataset are provided in the appendix. 

**Abstract (ZH)**: 基于参数化RAG的法律知识增强方法：PL-CA 

---
# Statistical Inference for Misspecified Contextual Bandits 

**Title (ZH)**: 错定性上下文_bandits的统计推断 

**Authors**: Yongyi Guo, Ziping Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06287)  

**Abstract**: Contextual bandit algorithms have transformed modern experimentation by enabling real-time adaptation for personalized treatment and efficient use of data. Yet these advantages create challenges for statistical inference due to adaptivity. A fundamental property that supports valid inference is policy convergence, meaning that action-selection probabilities converge in probability given the context. Convergence ensures replicability of adaptive experiments and stability of online algorithms. In this paper, we highlight a previously overlooked issue: widely used algorithms such as LinUCB may fail to converge when the reward model is misspecified, and such non-convergence creates fundamental obstacles for statistical inference. This issue is practically important, as misspecified models -- such as linear approximations of complex dynamic system -- are often employed in real-world adaptive experiments to balance bias and variance.
Motivated by this insight, we propose and analyze a broad class of algorithms that are guaranteed to converge even under model misspecification. Building on this guarantee, we develop a general inference framework based on an inverse-probability-weighted Z-estimator (IPW-Z) and establish its asymptotic normality with a consistent variance estimator. Simulation studies confirm that the proposed method provides robust and data-efficient confidence intervals, and can outperform existing approaches that exist only in the special case of offline policy evaluation. Taken together, our results underscore the importance of designing adaptive algorithms with built-in convergence guarantees to enable stable experimentation and valid statistical inference in practice. 

**Abstract (ZH)**: 上下文臂算法通过实现实时个性化治疗和高效数据利用，已经转变了现代实验。然而，这些优势也带来了统计推断中的挑战，因为它们是适应性的。支持有效推断的一个基本性质是策略收敛，即在给定上下文的情况下动作选择概率收敛。收敛确保了适应性实验的可重复性以及在线算法的稳定性。本文强调了一个之前被忽视的问题：广泛使用的算法如LinUCB，在奖励模型错误指定的情况下可能无法收敛，这种不收敛为统计推断创造了根本障碍。这一问题在实践中具有重要意义，因为许多实际的适应性实验中，如为了权衡偏差和方差而使用复杂的动态系统的线性近似模型时，错误指定的模型通常会被采用。

基于这一见解，我们提出并分析了一类保证在模型错误指定情况下仍然能够收敛的算法。基于这一保证，我们开发了一种通用的推断框架，该框架基于逆概率加权Z估计器（IPW-Z），并建立了其渐近正态性以及一致方差估计器。模拟研究证实，所提出的方法提供了稳健且数据高效的置信区间，并且在仅在离线策略评估中的特殊情况中才有现有方法的表现优于该方法。总的来说，我们的结果强调了设计内置收敛保证的适应性算法的重要性，以实现稳定的实验和有效的统计推断。 

---
# UrbanMIMOMap: A Ray-Traced MIMO CSI Dataset with Precoding-Aware Maps and Benchmarks 

**Title (ZH)**: UrbanMIMOMap：一种包含预编码aware图和基准的射线 tracing MIMO CSI数据集 

**Authors**: Honggang Jia, Xiucheng Wang, Nan Cheng, Ruijin Sun, Changle Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.06270)  

**Abstract**: Sixth generation (6G) systems require environment-aware communication, driven by native artificial intelligence (AI) and integrated sensing and communication (ISAC). Radio maps (RMs), providing spatially continuous channel information, are key enablers. However, generating high-fidelity RM ground truth via electromagnetic (EM) simulations is computationally intensive, motivating machine learning (ML)-based RM construction. The effectiveness of these data-driven methods depends on large-scale, high-quality training data. Current public datasets often focus on single-input single-output (SISO) and limited information, such as path loss, which is insufficient for advanced multi-input multi-output (MIMO) systems requiring detailed channel state information (CSI). To address this gap, this paper presents UrbanMIMOMap, a novel large-scale urban MIMO CSI dataset generated using high-precision ray tracing. UrbanMIMOMap offers comprehensive complex CSI matrices across a dense spatial grid, going beyond traditional path loss data. This rich CSI is vital for constructing high-fidelity RMs and serves as a fundamental resource for data-driven RM generation, including deep learning. We demonstrate the dataset's utility through baseline performance evaluations of representative ML methods for RM construction. This work provides a crucial dataset and reference for research in high-precision RM generation, MIMO spatial performance, and ML for 6G environment awareness. The code and data for this work are available at: this https URL. 

**Abstract (ZH)**: 第六代（6G）系统需要环境感知通信，受原生人工智能（AI）和集成传感与通信（ISAC）驱动。射频地图（RMs），提供连续空间信道信息，是关键使能器。然而，通过电磁（EM）模拟生成高保真RM地面真实值计算密集，促使基于机器学习（ML）的RM构建。这些数据驱动方法的有效性取决于大规模高质量的训练数据。当前的公共数据集通常侧重于单输入单输出（SISO）和有限的信息，如路径损耗，这不足以支持需要详细信道状态信息（CSI）的先进多输入多输出（MIMO）系统。为填补这一空白，本文提出了UrbanMIMOMap，这是一种使用高精度射线追踪生成的大规模城市MIMO CSI数据集。UrbanMIMOMap提供了一系列密集空间网格上的全面复杂CSI矩阵，超越了传统的路径损耗数据。这种丰富的CSI对于构建高保真射频地图至关重要，并作为数据驱动射频地图生成的基础资源，包括深度学习。我们通过代表性的ML方法对射频地图构建的基本性能进行评估，展示了该数据集的应用价值。该工作为高精度射频地图生成、MIMO空间性能以及6G环境感知中的机器学习提供了关键数据集和参考。相关代码和数据可在以下链接获取：this https URL。 

---
# On Synthesis of Timed Regular Expressions 

**Title (ZH)**: 定时正则表达式的合成 

**Authors**: Ziran Wang, Jie An, Naijun Zhan, Miaomiao Zhang, Zhenya Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06262)  

**Abstract**: Timed regular expressions serve as a formalism for specifying real-time behaviors of Cyber-Physical Systems. In this paper, we consider the synthesis of timed regular expressions, focusing on generating a timed regular expression consistent with a given set of system behaviors including positive and negative examples, i.e., accepting all positive examples and rejecting all negative examples. We first prove the decidability of the synthesis problem through an exploration of simple timed regular expressions. Subsequently, we propose our method of generating a consistent timed regular expression with minimal length, which unfolds in two steps. The first step is to enumerate and prune candidate parametric timed regular expressions. In the second step, we encode the requirement that a candidate generated by the first step is consistent with the given set into a Satisfiability Modulo Theories (SMT) formula, which is consequently solved to determine a solution to parametric time constraints. Finally, we evaluate our approach on benchmarks, including randomly generated behaviors from target timed models and a case study. 

**Abstract (ZH)**: 定时正规表达式用于描述 Cyber-Physical 系统的实时行为。本文考虑定时正规表达式的合成问题，重点是生成一个与给定的行为集（包括正例和反例）一致的定时正规表达式，即接受所有正例并拒绝所有反例。我们首先通过探索简单的定时正规表达式证明合成问题的可决定性。随后，我们提出了一种生成最短一致定时正规表达式的方法，该方法分为两步。第一步是枚举并修剪候选的参数化定时正规表达式。第二步是将第一步生成的候选表达式与给定集一致的要求编码为满意度模理论（SMT）公式，进而求解以确定参数时间约束的解。最后，我们在基准测试上评估了我们的方法，包括来自目标定时模型的随机生成行为和一个案例研究。 

---
# Distillation of CNN Ensemble Results for Enhanced Long-Term Prediction of the ENSO Phenomenon 

**Title (ZH)**: CNN集合结果的蒸馏以增强ENSO现象的长期预测能力 

**Authors**: Saghar Ganji, Mohammad Naisipour, Alireza Hassani, Arash Adib  

**Link**: [PDF](https://arxiv.org/pdf/2509.06227)  

**Abstract**: The accurate long-term forecasting of the El Nino Southern Oscillation (ENSO) is still one of the biggest challenges in climate science. While it is true that short-to medium-range performance has been improved significantly using the advances in deep learning, statistical dynamical hybrids, most operational systems still use the simple mean of all ensemble members, implicitly assuming equal skill across members. In this study, we demonstrate, through a strictly a-posteriori evaluation , for any large enough ensemble of ENSO forecasts, there is a subset of members whose skill is substantially higher than that of the ensemble mean. Using a state-of-the-art ENSO forecast system cross-validated against the 1986-2017 observed Nino3.4 index, we identify two Top-5 subsets one ranked on lowest Root Mean Square Error (RMSE) and another on highest Pearson correlation. Generally across all leads, these outstanding members show higher correlation and lower RMSE, with the advantage rising enormously with lead time. Whereas at short leads (1 month) raises the mean correlation by about +0.02 (+1.7%) and lowers the RMSE by around 0.14 °C or by 23.3% compared to the All-40 mean, at extreme leads (23 months) the correlation is raised by +0.43 (+172%) and RMSE by 0.18 °C or by 22.5% decrease. The enhancements are largest during crucial ENSO transition periods such as SON and DJF, when accurate amplitude and phase forecasting is of greatest socio-economic benefit, and furthermore season-dependent e.g., mid-year months such as JJA and MJJ have incredibly large RMSE reductions. This study provides a solid foundation for further investigations to identify reliable clues for detecting high-quality ensemble members, thereby enhancing forecasting skill. 

**Abstract (ZH)**: 准确长期预测厄尔尼诺南方 oscillation (ENSO) 仍然是气候科学中的一个重大挑战。通过严格的事后评估，我们展示，在任何足够大的 ENSO 预测 Ensemble 中，存在一个技能明显高于总体 Ensemble 平均值的子集。基于 1986-2017 年 Nino3.4 指数与最先进的 ENSO 预测系统交叉验证，我们确定了两个顶级子集，一个基于最低均方根误差 (RMSE)，另一个基于最高皮尔逊相关系数。总的来说，在所有预测时长中，这些杰出成员显示出更高的相关性和更低的 RMSE，而该优势随预测时长增加而显著增大。在短期预测（1 个月）中，顶级成员将平均相关性提高了约 +0.02 (+1.7%)，降低 RMSE 约 0.14 °C 或 23.3%，而在极端长期预测（23 个月）中，相关性提高了约 +0.43 (+172%)，降低 RMSE 约 0.18 °C 或 22.5%。这些改进在关键的ENSO 转换期（例如SON 和 DJF）尤为显著，那时准确的振幅和相位预测具有最大的社会经济效益，并且这种改进具有季节依赖性，如中期年份中的 JJA 和 MJJ 月份的 RMSE 减少尤为显著。本研究为后续研究识别可靠线索以检测高质量的 Ensemble 成员奠定了坚实基础，进而提高预测技能。 

---
# The Efficiency Frontier: Classical Shadows versus Quantum Footage 

**Title (ZH)**: 经典概影与量子剪影的效率边界 

**Authors**: Shuowei Ma, Junyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06218)  

**Abstract**: Interfacing quantum and classical processors is an important subroutine in full-stack quantum algorithms. The so-called "classical shadow" method efficiently extracts essential classical information from quantum states, enabling the prediction of many properties of a quantum system from only a few measurements. However, for a small number of highly non-local observables, or when classical post-processing power is limited, the classical shadow method is not always the most efficient choice. Here, we address this issue quantitatively by performing a full-stack resource analysis that compares classical shadows with ``quantum footage," which refers to direct quantum measurement. Under certain assumptions, our analysis illustrates a boundary of download efficiency between classical shadows and quantum footage. For observables expressed as linear combinations of Pauli matrices, the classical shadow method outperforms direct measurement when the number of observables is large and the Pauli weight is small. For observables in the form of large Hermitian sparse matrices, the classical shadow method shows an advantage when the number of observables, the sparsity of the matrix, and the number of qubits fall within a certain range. The key parameters influencing this behavior include the number of qubits $n$, observables $M$, sparsity $k$, Pauli weight $w$, accuracy requirement $\epsilon$, and failure tolerance $\delta$. We also compare the resource consumption of the two methods on different types of quantum computers and identify break-even points where the classical shadow method becomes more efficient, which vary depending on the hardware. This paper opens a new avenue for quantitatively designing optimal strategies for hybrid quantum-classical tomography and provides practical insights for selecting the most suitable quantum measurement approach in real-world applications. 

**Abstract (ZH)**: 量子和经典处理器接口是全栈量子算法中的一个重要子程序。所谓的“经典阴影”方法高效地从量子态中提取关键的经典信息，从而仅通过少量测量就能够预测量子系统中的许多性质。然而，对于高度非局域的观测量较少或经典后处理能力有限的情况，“经典阴影”方法并不总是最有效的选择。通过进行全面栈资源分析，我们将“经典阴影”方法与“量子影像”进行了比较，后者是指直接量子测量。在某些假设下，我们的分析展示了经典阴影方法与量子影像之间下载效率的边界。对于用保罗伊矩阵线性组合表示的观测量，当观测量的数量较大且保罗伊权重较小时，“经典阴影”方法优于直接测量。对于用大型厄密稀疏矩阵表示的观测量，在观测量的数量、矩阵的稀疏性和量子比特数处于一定范围内的条件下，“经典阴影”方法显示出优势。影响这一行为的关键参数包括量子比特数$n$、观测量$M$、稀疏性$k$、保罗伊权重$w$、准确度要求$\epsilon$和失败容忍度$\delta$。我们还比较了两种方法在不同类型量子计算机上的资源消耗，并确定了“经典阴影”方法变得更有效的转折点，这些点取决于硬件的不同。本文为定量设计混合量子-经典成像的最佳策略开辟了新途径，并为实际应用中选择最适合的量子测量方法提供了实用见解。 

---
# Agentic Software Engineering: Foundational Pillars and a Research Roadmap 

**Title (ZH)**: 代理软件工程：基础支柱与研究路线图 

**Authors**: Ahmed E. Hassan, Hao Li, Dayi Lin, Bram Adams, Tse-Hsun Chen, Yutaro Kashiwa, Dong Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06216)  

**Abstract**: Agentic Software Engineering (SE 3.0) represents a new era where intelligent agents are tasked not with simple code generation, but with achieving complex, goal-oriented SE objectives. To harness these new capabilities while ensuring trustworthiness, we must recognize a fundamental duality within the SE field in the Agentic SE era, comprising two symbiotic modalities: SE for Humans and SE for Agents. This duality demands a radical reimagining of the foundational pillars of SE (actors, processes, tools, and artifacts) which manifest differently across each modality. We propose two purpose-built workbenches to support this vision. The Agent Command Environment (ACE) serves as a command center where humans orchestrate and mentor agent teams, handling outputs such as Merge-Readiness Packs (MRPs) and Consultation Request Packs (CRPs). The Agent Execution Environment (AEE) is a digital workspace where agents perform tasks while invoking human expertise when facing ambiguity or complex trade-offs. This bi-directional partnership, which supports agent-initiated human callbacks and handovers, gives rise to new, structured engineering activities (i.e., processes) that redefine human-AI collaboration, elevating the practice from agentic coding to true agentic software engineering. This paper presents the Structured Agentic Software Engineering (SASE) vision, outlining several of the foundational pillars for the future of SE. The paper culminates in a research roadmap that identifies a few key challenges and opportunities while briefly discussing the resulting impact of this future on SE education. Our goal is not to offer a definitive solution, but to provide a conceptual scaffold with structured vocabulary to catalyze a community-wide dialogue, pushing the SE community to think beyond its classic, human-centric tenets toward a disciplined, scalable, and trustworthy agentic future. 

**Abstract (ZH)**: 代理软件工程（SE 3.0）代表了一个新时代，在这个时代，智能代理的任务不仅仅是简单的代码生成，而是实现复杂的、目标导向的软件工程目标。要在利用这些新能力的同时确保可靠性，我们必须认识到代理软件工程时代软件工程领域的根本二元性，包括两种共生的方式：为人类的软件工程和为代理的软件工程。这种二元性要求对软件工程的基础支柱（参与者、过程、工具和制品）进行根本性的重新思考，这些支柱在每种方式中以不同的方式体现出来。我们提出了两个定制工作台来支持这一愿景。代理命令环境（ACE）作为指令中心，人类在这里编排并指导代理团队，并处理合并就绪包（MRP）和咨询请求包（CRP）等多种输出。代理执行环境（AEE）是一个数字工作空间，在这里代理执行任务，当遇到模糊性或复杂权衡时，调用人类的专业知识。这种双向伙伴关系，支持代理发起的人类回调和接力，产生了新的结构化工程活动（即过程），重新定义了人类与人工智能的合作，使实践从代理编程提升为真正的代理软件工程。本文提出了结构化代理软件工程（SASE）的愿景，概述了未来软件工程的一些基本支柱，并列出了几个关键挑战和机遇，简要讨论了这一未来对软件工程教育的影响。我们的目标不是提供一种终极解决方案，而是提供一种结构化的概念框架和词汇，以促进整个社区的对话，促使软件工程社区超越其传统的、以人类为中心的原则，朝着有纪律、可扩展和可靠的代理未来迈进。 

---
# Language Bias in Information Retrieval: The Nature of the Beast and Mitigation Methods 

**Title (ZH)**: 语言偏见在信息检索中的表现及其缓解方法 

**Authors**: Jinrui Yang, Fan Jiang, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06195)  

**Abstract**: Language fairness in multilingual information retrieval (MLIR) systems is crucial for ensuring equitable access to information across diverse languages. This paper sheds light on the issue, based on the assumption that queries in different languages, but with identical semantics, should yield equivalent ranking lists when retrieving on the same multilingual documents. We evaluate the degree of fairness using both traditional retrieval methods, and a DPR neural ranker based on mBERT and XLM-R. Additionally, we introduce `LaKDA', a novel loss designed to mitigate language biases in neural MLIR approaches. Our analysis exposes intrinsic language biases in current MLIR technologies, with notable disparities across the retrieval methods, and the effectiveness of LaKDA in enhancing language fairness. 

**Abstract (ZH)**: 多语言信息检索系统中的语言公平性对于确保不同语言用户获得平等的信息访问权至关重要。本文基于不同语言但语义相同的查询在检索相同的多语言文档时应产生等效的排名列表这一假设，探讨了这一问题。我们使用传统的检索方法和基于mBERT和XLM-R的DPR神经排名器来评估公平性的程度，并引入了`LaKDA`这一新型损失函数，以减轻神经多语言信息检索方法中的语言偏见。我们的分析揭示了当前多语言信息检索技术中存在的内在语言偏见，以及`LaKDA`在提升语言公平性方面的有效性。 

---
# AI Governance in Higher Education: A course design exploring regulatory, ethical and practical considerations 

**Title (ZH)**: 高等教育中的AI治理：一门课程设计探索监管、伦理与实践考量 

**Authors**: Zsolt Almási, Hannah Bleher, Johannes Bleher, Rozanne Tuesday Flores, Guo Xuanyang, Paweł Pujszo, Raphaël Weuts  

**Link**: [PDF](https://arxiv.org/pdf/2509.06176)  

**Abstract**: As artificial intelligence (AI) systems permeate critical sectors, the need for professionals who can address ethical, legal and governance challenges has become urgent. Current AI ethics education remains fragmented, often siloed by discipline and disconnected from practice. This paper synthesizes literature and regulatory developments to propose a modular, interdisciplinary curriculum that integrates technical foundations with ethics, law and policy. We highlight recurring operational failures in AI - bias, misspecified objectives, generalization errors, misuse and governance breakdowns - and link them to pedagogical strategies for teaching AI governance. Drawing on perspectives from the EU, China and international frameworks, we outline a semester plan that emphasizes integrated ethics, stakeholder engagement and experiential learning. The curriculum aims to prepare students to diagnose risks, navigate regulation and engage diverse stakeholders, fostering adaptive and ethically grounded professionals for responsible AI governance. 

**Abstract (ZH)**: 人工智能系统渗透关键领域后对伦理、法律与治理挑战的专业人才需求迫在眉睫：模块化跨学科课程设计以整合技术基础与伦理、法律及政策的研究与教学 

---
# Tracking daily paths in home contexts with RSSI fingerprinting based on UWB through deep learning models 

**Title (ZH)**: 基于UWB的RSSI指纹识别结合深度学习模型在家庭场景中跟踪日常路径 

**Authors**: Aurora Polo-Rodríguez, Juan Carlos Valera, Jesús Peral, David Gil, Javier Medina-Quero  

**Link**: [PDF](https://arxiv.org/pdf/2509.06161)  

**Abstract**: The field of human activity recognition has evolved significantly, driven largely by advancements in Internet of Things (IoT) device technology, particularly in personal devices. This study investigates the use of ultra-wideband (UWB) technology for tracking inhabitant paths in home environments using deep learning models. UWB technology estimates user locations via time-of-flight and time-difference-of-arrival methods, which are significantly affected by the presence of walls and obstacles in real environments, reducing their precision. To address these challenges, we propose a fingerprinting-based approach utilizing received signal strength indicator (RSSI) data collected from inhabitants in two flats (60 m2 and 100 m2) while performing daily activities. We compare the performance of convolutional neural network (CNN), long short-term memory (LSTM), and hybrid CNN+LSTM models, as well as the use of Bluetooth technology. Additionally, we evaluate the impact of the type and duration of the temporal window (future, past, or a combination of both). Our results demonstrate a mean absolute error close to 50 cm, highlighting the superiority of the hybrid model in providing accurate location estimates, thus facilitating its application in daily human activity recognition in residential settings. 

**Abstract (ZH)**: 基于超宽带技术的深学习方法在家庭环境居民路径跟踪中的应用研究 

---
# Software Dependencies 2.0: An Empirical Study of Reuse and Integration of Pre-Trained Models in Open-Source Projects 

**Title (ZH)**: 软件依赖关系 2.0：开源项目中预训练模型的重用与集成实证研究 

**Authors**: Jerin Yasmin, Wenxin Jiang, James C. Davis, Yuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2509.06085)  

**Abstract**: Pre-trained models (PTMs) are machine learning models that have been trained in advance, often on large-scale data, and can be reused for new tasks, thereby reducing the need for costly training from scratch. Their widespread adoption introduces a new class of software dependency, which we term Software Dependencies 2.0, extending beyond conventional libraries to learned behaviors embodied in trained models and their associated artifacts. The integration of PTMs as software dependencies in real projects remains unclear, potentially threatening maintainability and reliability of modern software systems that increasingly rely on them. Objective: In this study, we investigate Software Dependencies 2.0 in open-source software (OSS) projects by examining the reuse of PTMs, with a focus on how developers manage and integrate these models. Specifically, we seek to understand: (1) how OSS projects structure and document their PTM dependencies; (2) what stages and organizational patterns emerge in the reuse pipelines of PTMs within these projects; and (3) the interactions among PTMs and other learned components across pipeline stages. We conduct a mixed-methods analysis of a statistically significant random sample of 401 GitHub repositories from the PeaTMOSS dataset (28,575 repositories reusing PTMs from Hugging Face and PyTorch Hub). We quantitatively examine PTM reuse by identifying patterns and qualitatively investigate how developers integrate and manage these models in practice. 

**Abstract (ZH)**: 预训练模型（PTMs）是预先在大型数据集上训练的机器学习模型，可以重新用于新任务，从而减少从头开始训练的成本。其广泛应用引入了一类新的软件依赖关系，我们称之为软件依赖关系2.0，超越了传统的库，涵盖了嵌入在训练模型及其相关制品中的学习行为。PTMs在实际项目中作为软件依赖的集成仍不明确，可能威胁现代软件系统的可维护性和可靠性，这些系统越来越多地依赖于它们。目的：在本研究中，我们通过检查PTM的再利用情况，研究开源软件（OSS）项目的软件依赖关系2.0，重点关注开发人员如何管理和集成这些模型。具体而言，我们旨在理解：（1）OSS项目如何结构化和文档化其PTM依赖关系；（2）这些项目中PTM再利用管道中出现的阶段和组织模式；以及（3）在管道阶段中，PTMs与其他学习组件之间的交互。我们对PeaTMOSS数据集中的401个GitHub仓库（28,575个仓库从Hugging Face和PyTorch Hub再利用PTMs）进行了统计上有意义的随机样本混合方法分析。我们定量检查PTM的再利用情况，通过识别模式，并定性研究开发人员在实践中如何集成和管理这些模型。 

---
# ARIES: Relation Assessment and Model Recommendation for Deep Time Series Forecasting 

**Title (ZH)**: ARIES：深度时间序列预测中的关系评估与模型推荐 

**Authors**: Fei Wang, Yujie Li, Zezhi Shao, Chengqing Yu, Yisong Fu, Zhulin An, Yongjun Xu, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06060)  

**Abstract**: Recent advancements in deep learning models for time series forecasting have been significant. These models often leverage fundamental time series properties such as seasonality and non-stationarity, which may suggest an intrinsic link between model performance and data properties. However, existing benchmark datasets fail to offer diverse and well-defined temporal patterns, restricting the systematic evaluation of such connections. Additionally, there is no effective model recommendation approach, leading to high time and cost expenditures when testing different architectures across different downstream applications. For those reasons, we propose ARIES, a framework for assessing relation between time series properties and modeling strategies, and for recommending deep forcasting models for realistic time series. First, we construct a synthetic dataset with multiple distinct patterns, and design a comprehensive system to compute the properties of time series. Next, we conduct an extensive benchmarking of over 50 forecasting models, and establish the relationship between time series properties and modeling strategies. Our experimental results reveal a clear correlation. Based on these findings, we propose the first deep forecasting model recommender, capable of providing interpretable suggestions for real-world time series. In summary, ARIES is the first study to establish the relations between the properties of time series data and modeling strategies, while also implementing a model recommendation system. The code is available at: this https URL. 

**Abstract (ZH)**: Recent advancements in deep learning models for time series forecasting have been significant. These models often leverage fundamental time series properties such as seasonality and non-stationarity, which may suggest an intrinsic link between model performance and data properties. However, existing benchmark datasets fail to offer diverse and well-defined temporal patterns, restricting the systematic evaluation of such connections. Additionally, there is no effective model recommendation approach, leading to high time and cost expenditures when testing different architectures across different downstream applications. For those reasons, we propose ARIES, a framework for assessing the relation between time series properties and modeling strategies, and for recommending deep forecasting models for realistic time series. First, we construct a synthetic dataset with multiple distinct patterns, and design a comprehensive system to compute the properties of time series. Next, we conduct an extensive benchmarking of over 50 forecasting models, and establish the relationship between time series properties and modeling strategies. Our experimental results reveal a clear correlation. Based on these findings, we propose the first deep forecasting model recommender, capable of providing interpretable suggestions for real-world time series. In summary, ARIES is the first study to establish the relations between the properties of time series data and modeling strategies, while also implementing a model recommendation system. The code is available at: this https URL. 

---
# BranchGRPO: Stable and Efficient GRPO with Structured Branching in Diffusion Models 

**Title (ZH)**: BranchGRPO：具有结构化分支的稳定高效GRPO在扩散模型中 

**Authors**: Yuming Li, Yikai Wang, Yuying Zhu, Zhongyu Zhao, Ming Lu, Qi She, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06040)  

**Abstract**: Recent advancements in aligning image and video generative models via GRPO have achieved remarkable gains in enhancing human preference alignment. However, these methods still face high computational costs from on-policy rollouts and excessive SDE sampling steps, as well as training instability due to sparse rewards. In this paper, we propose BranchGRPO, a novel method that introduces a branch sampling policy updating the SDE sampling process. By sharing computation across common prefixes and pruning low-reward paths and redundant depths, BranchGRPO substantially lowers the per-update compute cost while maintaining or improving exploration diversity. This work makes three main contributions: (1) a branch sampling scheme that reduces rollout and training cost; (2) a tree-based advantage estimator incorporating dense process-level rewards; and (3) pruning strategies exploiting path and depth redundancy to accelerate convergence and boost performance. Experiments on image and video preference alignment show that BranchGRPO improves alignment scores by 16% over strong baselines, while cutting training time by 50%. 

**Abstract (ZH)**: 基于GRPO的分支采样方法在图像和视频生成模型对齐中的Recent Advancements and Contributions 

---
# DreamAudio: Customized Text-to-Audio Generation with Diffusion Models 

**Title (ZH)**: DreamAudio: 定制化文本到语音生成方法Based on扩散模型 

**Authors**: Yi Yuan, Xubo Liu, Haohe Liu, Xiyuan Kang, Zhuo Chen, Yuxuan Wang, Mark D. Plumbley, Wenwu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06027)  

**Abstract**: With the development of large-scale diffusion-based and language-modeling-based generative models, impressive progress has been achieved in text-to-audio generation. Despite producing high-quality outputs, existing text-to-audio models mainly aim to generate semantically aligned sound and fall short on precisely controlling fine-grained acoustic characteristics of specific sounds. As a result, users that need specific sound content may find it challenging to generate the desired audio clips. In this paper, we present DreamAudio for customized text-to-audio generation (CTTA). Specifically, we introduce a new framework that is designed to enable the model to identify auditory information from user-provided reference concepts for audio generation. Given a few reference audio samples containing personalized audio events, our system can generate new audio samples that include these specific events. In addition, two types of datasets are developed for training and testing the customized systems. The experiments show that the proposed model, DreamAudio, generates audio samples that are highly consistent with the customized audio features and aligned well with the input text prompts. Furthermore, DreamAudio offers comparable performance in general text-to-audio tasks. We also provide a human-involved dataset containing audio events from real-world CTTA cases as the benchmark for customized generation tasks. 

**Abstract (ZH)**: 基于定制文本到音频生成的DreamAudio 

---
# DCMI: A Differential Calibration Membership Inference Attack Against Retrieval-Augmented Generation 

**Title (ZH)**: DCMI: 一种针对检索增强生成的差分校准成员推断攻击 

**Authors**: Xinyu Gao, Xiangtao Meng, Yingkai Dong, Zheng Li, Shanqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06026)  

**Abstract**: While Retrieval-Augmented Generation (RAG) effectively reduces hallucinations by integrating external knowledge bases, it introduces vulnerabilities to membership inference attacks (MIAs), particularly in systems handling sensitive data. Existing MIAs targeting RAG's external databases often rely on model responses but ignore the interference of non-member-retrieved documents on RAG outputs, limiting their effectiveness. To address this, we propose DCMI, a differential calibration MIA that mitigates the negative impact of non-member-retrieved documents. Specifically, DCMI leverages the sensitivity gap between member and non-member retrieved documents under query perturbation. It generates perturbed queries for calibration to isolate the contribution of member-retrieved documents while minimizing the interference from non-member-retrieved documents. Experiments under progressively relaxed assumptions show that DCMI consistently outperforms baselines--for example, achieving 97.42% AUC and 94.35% Accuracy against the RAG system with Flan-T5, exceeding the MBA baseline by over 40%. Furthermore, on real-world RAG platforms such as Dify and MaxKB, DCMI maintains a 10%-20% advantage over the baseline. These results highlight significant privacy risks in RAG systems and emphasize the need for stronger protection mechanisms. We appeal to the community's consideration of deeper investigations, like ours, against the data leakage risks in rapidly evolving RAG systems. Our code is available at this https URL. 

**Abstract (ZH)**: DCMI：针对检索增强生成系统的差异校准会员推理攻击 

---
# Unified Interaction Foundational Model (UIFM) for Predicting Complex User and System Behavior 

**Title (ZH)**: 统一交互基础模型（UIFM）用于预测复杂用户和系统行为 

**Authors**: Vignesh Ethiraj, Subhash Talluri  

**Link**: [PDF](https://arxiv.org/pdf/2509.06025)  

**Abstract**: A central goal of artificial intelligence is to build systems that can understand and predict complex, evolving sequences of events. However, current foundation models, designed for natural language, fail to grasp the holistic nature of structured interactions found in domains like telecommunications, e-commerce and finance. By serializing events into text, they disassemble them into semantically fragmented parts, losing critical context. In this work, we introduce the Unified Interaction Foundation Model (UIFM), a foundation model engineered for genuine behavioral understanding. At its core is the principle of composite tokenization, where each multi-attribute event is treated as a single, semantically coherent unit. This allows UIFM to learn the underlying "grammar" of user behavior, perceiving entire interactions rather than a disconnected stream of data points. We demonstrate that this architecture is not just more accurate, but represents a fundamental step towards creating more adaptable and intelligent predictive systems. 

**Abstract (ZH)**: 统一交互基础模型：一种用于真实行为理解的基础模型 

---
# Khana: A Comprehensive Indian Cuisine Dataset 

**Title (ZH)**: Khana: 一套全面的印度 Cuisine 数据集 

**Authors**: Omkar Prabhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06006)  

**Abstract**: As global interest in diverse culinary experiences grows, food image models are essential for improving food-related applications by enabling accurate food recognition, recipe suggestions, dietary tracking, and automated meal planning. Despite the abundance of food datasets, a noticeable gap remains in capturing the nuances of Indian cuisine due to its vast regional diversity, complex preparations, and the lack of comprehensive labeled datasets that cover its full breadth. Through this exploration, we uncover Khana, a new benchmark dataset for food image classification, segmentation, and retrieval of dishes from Indian cuisine. Khana fills the gap by establishing a taxonomy of Indian cuisine and offering around 131K images in the dataset spread across 80 labels, each with a resolution of 500x500 pixels. This paper describes the dataset creation process and evaluates state-of-the-art models on classification, segmentation, and retrieval as baselines. Khana bridges the gap between research and development by providing a comprehensive and challenging benchmark for researchers while also serving as a valuable resource for developers creating real-world applications that leverage the rich tapestry of Indian cuisine. Webpage: this https URL 

**Abstract (ZH)**: 随着全球对多元化美食体验的兴趣增长，食品图像模型对于通过准确的食物识别、食谱建议、饮食跟踪和自动化餐食规划来改进相关应用至关重要。尽管存在大量的食品数据集，但由于印度菜的地域多样性、复杂的料理方式以及缺乏涵盖其全部范围的综合标注数据集，仍存在明显的差距。通过这一探索，我们发现了Khana，一个用于印度菜菜品分类、分割和检索的新基准数据集。Khana通过建立印度菜的分类体系，并提供约131K张图像（每个标签80个，分辨率为500x500像素），填补了这一空白。本文描述了数据集的创建过程，并在分类、分割和检索任务上评估了最先进的模型，作为基线。Khana通过提供全面且具有挑战性的基准，架起了研究与开发之间的桥梁，同时也为开发人员创建利用丰富多样的印度菜美食的应用程序提供了宝贵的资源。网页地址：这个 https URL 

---
# Operationalising AI Regulatory Sandboxes under the EU AI Act: The Triple Challenge of Capacity, Coordination and Attractiveness to Providers 

**Title (ZH)**: 欧盟AI法案下AI监管沙箱的三重挑战：能力、协调与服务提供者的吸引力 

**Authors**: Deirdre Ahern  

**Link**: [PDF](https://arxiv.org/pdf/2509.05985)  

**Abstract**: The EU AI Act provides a rulebook for all AI systems being put on the market or into service in the European Union. This article investigates the requirement under the AI Act that Member States establish national AI regulatory sandboxes for testing and validation of innovative AI systems under regulatory supervision to assist with fostering innovation and complying with regulatory requirements. Against the backdrop of the EU objective that AI regulatory sandboxes would both foster innovation and assist with compliance, considerable challenges are identified for Member States around capacity-building and design of regulatory sandboxes. While Member States are early movers in laying the ground for national AI regulatory sandboxes, the article contends that there is a risk that differing approaches being taken by individual national sandboxes could jeopardise a uniform interpretation of the AI Act and its application in practice. This could motivate innovators to play sandbox arbitrage. The article therefore argues that the European Commission and the AI Board need to act decisively in developing rules and guidance to ensure a cohesive, coordinated approach in national AI regulatory sandboxes. With sandbox participation being voluntary, the possibility that AI regulatory sandboxes may prove unattractive to innovators on their compliance journey is also explored. Confidentiality concerns, the inability to relax legal rules during the sandbox, and the inability of sandboxes to deliver a presumption of conformity with the AI Act are identified as pertinent concerns for innovators contemplating applying to AI regulatory sandboxes as compared with other direct compliance routes provided to them through application of harmonised standards and conformity assessment procedures. 

**Abstract (ZH)**: 欧盟AI法案为欧盟市场上的所有AI系统提供了操作规范。本文 investigate 欧盟AI法案中要求成员国建立国家级AI监管沙箱以测试和验证创新AI系统并在监管监督下助于促进创新和合规的要求。在欧盟旨在通过监管沙箱促进创新和助于合规的背景下，成员国在能力建设和监管沙箱设计方面面临着重大挑战。尽管成员国在为国家级AI监管沙箱奠定基础方面处于领先地位，但文章认为，个别国家级沙箱采取的不同方法可能会危及AI法案的统一解释和实际应用，这可能会激励创新者进行沙箱套利。因此，文章认为，欧盟委员会和AI委员会需要果断行动，制定规则和指导，确保国家级AI监管沙箱的一致性和协调性。鉴于沙箱参与是自愿的，文章还探讨了AI监管沙箱可能对创新者的合规之路缺乏吸引力的可能性。此外，还指出了创新者在考虑申请AI监管沙箱与通过采纳协调标准和一致性评估程序提供给他们的其他直接合规途径相比所面临的相关性隐私担忧、无法在沙箱期间放松法律规定以及沙箱无法提供AI法案符合性推定等关键问题。 

---
# TSPC: A Two-Stage Phoneme-Centric Architecture for code-switching Vietnamese-English Speech Recognition 

**Title (ZH)**: TSPC：一种两阶段基于音素的代码切换越南英语语音识别架构 

**Authors**: Minh N. H. Nguyen, Anh Nguyen Tran, Dung Truong Dinh, Nam Van Vo  

**Link**: [PDF](https://arxiv.org/pdf/2509.05983)  

**Abstract**: Code-switching (CS) presents a significant challenge for general Auto-Speech Recognition (ASR) systems. Existing methods often fail to capture the subtle phonological shifts inherent in CS scenarios. The challenge is particularly difficult for language pairs like Vietnamese and English, where both distinct phonological features and the ambiguity arising from similar sound recognition are present. In this paper, we propose a novel architecture for Vietnamese-English CS ASR, a Two-Stage Phoneme-Centric model (TSPC). The TSPC employs a phoneme-centric approach, built upon an extended Vietnamese phoneme set as an intermediate representation to facilitate mixed-lingual modeling. Experimental results demonstrate that TSPC consistently outperforms existing baselines, including PhoWhisper-base, in Vietnamese-English CS ASR, achieving a significantly lower word error rate of 20.8\% with reduced training resources. Furthermore, the phonetic-based two-stage architecture enables phoneme adaptation and language conversion to enhance ASR performance in complex CS Vietnamese-English ASR scenarios. 

**Abstract (ZH)**: 越南语-英语代码转换自动语音识别中的 Two-Stage 音位中心模型 

---
# ConstStyle: Robust Domain Generalization with Unified Style Transformation 

**Title (ZH)**: ConstStyle: 一致风格转换下的稳健领域泛化 

**Authors**: Nam Duong Tran, Nam Nguyen Phuong, Hieu H. Pham, Phi Le Nguyen, My T. Thai  

**Link**: [PDF](https://arxiv.org/pdf/2509.05975)  

**Abstract**: Deep neural networks often suffer performance drops when test data distribution differs from training data. Domain Generalization (DG) aims to address this by focusing on domain-invariant features or augmenting data for greater diversity. However, these methods often struggle with limited training domains or significant gaps between seen (training) and unseen (test) domains. To enhance DG robustness, we hypothesize that it is essential for the model to be trained on data from domains that closely resemble unseen test domains-an inherently difficult task due to the absence of prior knowledge about the unseen domains. Accordingly, we propose ConstStyle, a novel approach that leverages a unified domain to capture domain-invariant features and bridge the domain gap with theoretical analysis. During training, all samples are mapped onto this unified domain, optimized for seen domains. During testing, unseen domain samples are projected similarly before predictions. By aligning both training and testing data within this unified domain, ConstStyle effectively reduces the impact of domain shifts, even with large domain gaps or few seen domains. Extensive experiments demonstrate that ConstStyle consistently outperforms existing methods across diverse scenarios. Notably, when only a limited number of seen domains are available, ConstStyle can boost accuracy up to 19.82\% compared to the next best approach. 

**Abstract (ZH)**: 深度神经网络在测试数据分布与训练数据分布不同时往往会性能下降。域泛化（Domain Generalization, DG）旨在通过关注域不变特征或增强数据多样性来解决这一问题。然而，这些方法在面对训练域有限或训练域与测试域之间差距显著时常常表现不佳。为增强域泛化的鲁棒性，我们假设模型需要在其训练过程中使用与未知测试域相近的域数据，这是一个由于缺乏未知域先验知识而本身颇具挑战的任务。据此，我们提出了一种名为ConstStyle的新型方法，通过利用统一域来捕获域不变特征，并利用理论分析缩小域间差距。在训练过程中，所有样本都被映射到这个统一域中，优化针对已见域进行。在测试过程中，未知域的样本被以类似方式投影后再进行预测。通过将训练和测试数据均映射到此统一域中，ConstStyle有效地减少了域转移的影响，即使在域间差距大或已见域有限的情况下也是如此。大量实验证明，ConstStyle在各种场景中均能显著超过现有方法。特别地，在仅有有限数量已见域可用的情况下，ConstStyle相比第二优方法能将准确率提升高达19.82%。 

---
# Meta-training of diffractive meta-neural networks for super-resolution direction of arrival estimation 

**Title (ZH)**: 基于分类衍射元神经网络的超分辨到达角估计元训练 

**Authors**: Songtao Yang, Sheng Gao, Chu Wu, Zejia Zhao, Haiou Zhang, Xing Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05926)  

**Abstract**: Diffractive neural networks leverage the high-dimensional characteristics of electromagnetic (EM) fields for high-throughput computing. However, the existing architectures face challenges in integrating large-scale multidimensional metasurfaces with precise network training and haven't utilized multidimensional EM field coding scheme for super-resolution sensing. Here, we propose diffractive meta-neural networks (DMNNs) for accurate EM field modulation through metasurfaces, which enable multidimensional multiplexing and coding for multi-task learning and high-throughput super-resolution direction of arrival estimation. DMNN integrates pre-trained mini-metanets to characterize the amplitude and phase responses of meta-atoms across different polarizations and frequencies, with structure parameters inversely designed using the gradient-based meta-training. For wide-field super-resolution angle estimation, the system simultaneously resolves azimuthal and elevational angles through x and y-polarization channels, while the interleaving of frequency-multiplexed angular intervals generates spectral-encoded optical super-oscillations to achieve full-angle high-resolution estimation. Post-processing lightweight electronic neural networks further enhance the performance. Experimental results validate that a three-layer DMNN operating at 27 GHz, 29 GHz, and 31 GHz achieves $\sim7\times$ Rayleigh diffraction-limited angular resolution (0.5$^\circ$), a mean absolute error of 0.048$^\circ$ for two incoherent targets within a $\pm 11.5^\circ$ field of view, and an angular estimation throughput an order of magnitude higher (1917) than that of existing methods. The proposed architecture advances high-dimensional photonic computing systems by utilizing inherent high-parallelism and all-optical coding methods for ultra-high-resolution, high-throughput applications. 

**Abstract (ZH)**: 衍射神经网络利用电磁场的高维特性进行高速计算。然而，现有的架构在集成大规模多维元表面及精确的网络训练方面面临挑战，并未利用多维电磁场编码方案进行超分辨率传感。这里，我们提出衍射元神经网络（DMNN）以通过元表面实现精确的电磁场调制，从而实现多维复用和编码，支持多任务学习和高速超分辨率到达角估计。DMNN集成了预先训练的小型元网络，用于表征不同极化和频率下元原子的幅度和相位响应，并通过基于梯度的元训练逆向设计结构参数。对于宽场超分辨率角度估计，系统通过x和y极化通道同时解算方位角和仰角，而频率多路复用的 angular 间隔的交错排列生成谱编码光的超振荡，以实现全方位高分辨率估计。后续处理的轻量级电子神经网络进一步提升性能。实验结果验证了在27 GHz、29 GHz和31 GHz工作的三层DMNN可实现约7倍瑞利衍射极限角度分辨率（0.5°）、视场为±11.5°内的两个非相干目标的绝对误差均值为0.048°，以及比现有方法高一个数量级的角度估计吞吐量（1917）。所提出的架构通过利用固有的高并行性和全光编码方法，推动了高维光子计算系统的应用，实现了超高分辨率和高速处理。 

---
# Quantum spatial best-arm identification via quantum walks 

**Title (ZH)**: 量子行走最佳臂识别的空间寻优方法 

**Authors**: Tomoki Yamagami, Etsuo Segawa, Takatomo Mihana, André Röhm, Atsushi Uchida, Ryoichi Horisaki  

**Link**: [PDF](https://arxiv.org/pdf/2509.05890)  

**Abstract**: Quantum reinforcement learning has emerged as a framework combining quantum computation with sequential decision-making, and applications to the multi-armed bandit (MAB) problem have been reported. The graph bandit problem extends the MAB setting by introducing spatial constraints, yet quantum approaches remain limited. We propose a quantum algorithm for best-arm identification in graph bandits, termed Quantum Spatial Best-Arm Identification (QSBAI). The method employs quantum walks to encode superpositions over graph-constrained actions, extending amplitude amplification and generalizing the Quantum BAI algorithm via Szegedy's walk framework. This establishes a link between Grover-type search and reinforcement learning tasks with structural restrictions. We analyze complete and bipartite graphs, deriving the maximal success probability of identifying the best arm and the time step at which it is achieved. Our results highlight the potential of quantum walks to accelerate exploration in constrained environments and extend the applicability of quantum algorithms for decision-making. 

**Abstract (ZH)**: 量子强化学习已成为结合量子计算与序列决策的一种框架，并已被应用于多臂bandit问题。图bandit问题通过引入空间约束扩展了MAB设置，但量子方法仍然有限。我们提出了一种用于图bandit中最佳臂识别的量子算法，称为量子空间最佳臂识别（QSBAI）。该方法利用量子行走来编码受图约束的动作的超位置，并通过Szegedy的量子步行框架扩展了振幅放大和量子BAI算法。这建立了Grover型搜索与具有结构限制的强化学习任务之间的联系。我们分析了完全图和二部图，得出了识别最佳臂的最大成功概率及其实现的时间步。结果突显了量子行走在约束环境中加速探索的潜力，并扩展了量子算法在决策中的应用。 

---
# Uncertainty Quantification in Probabilistic Machine Learning Models: Theory, Methods, and Insights 

**Title (ZH)**: 概率机器学习模型中的不确定性量化：理论、方法与见解 

**Authors**: Marzieh Ajirak, Anand Ravishankar, Petar M. Djuric  

**Link**: [PDF](https://arxiv.org/pdf/2509.05877)  

**Abstract**: Uncertainty Quantification (UQ) is essential in probabilistic machine learning models, particularly for assessing the reliability of predictions. In this paper, we present a systematic framework for estimating both epistemic and aleatoric uncertainty in probabilistic models. We focus on Gaussian Process Latent Variable Models and employ scalable Random Fourier Features-based Gaussian Processes to approximate predictive distributions efficiently. We derive a theoretical formulation for UQ, propose a Monte Carlo sampling-based estimation method, and conduct experiments to evaluate the impact of uncertainty estimation. Our results provide insights into the sources of predictive uncertainty and illustrate the effectiveness of our approach in quantifying the confidence in the predictions. 

**Abstract (ZH)**: 不确定性量化（UQ）在概率机器学习模型中至关重要，特别用于评估预测的可靠性。本文提出了一种系统框架，用于估算概率模型中的命题性和偶然性不确定性。我们专注于高斯过程潜在变量模型，并采用可扩展的随机傅里叶特征基于的高斯过程来高效地逼近预测分布。我们为不确定性量化推导出理论公式，提出了一种基于蒙特卡洛采样的估算方法，并进行了实验以评估不确定性估算的影响。我们的结果为预测不确定性来源提供了见解，并展示了我们方法在量化预测置信度方面的有效性。 

---
# Learning to Construct Knowledge through Sparse Reference Selection with Reinforcement Learning 

**Title (ZH)**: 通过稀疏引用选择强化学习驱动的知识构建 

**Authors**: Shao-An Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05874)  

**Abstract**: The rapid expansion of scientific literature makes it increasingly difficult to acquire new knowledge, particularly in specialized domains where reasoning is complex, full-text access is restricted, and target references are sparse among a large set of candidates. We present a Deep Reinforcement Learning framework for sparse reference selection that emulates human knowledge construction, prioritizing which papers to read under limited time and cost. Evaluated on drug--gene relation discovery with access restricted to titles and abstracts, our approach demonstrates that both humans and machines can construct knowledge effectively from partial information. 

**Abstract (ZH)**: 基于深度强化学习的稀疏参考选择框架：在受限时间和成本条件下模拟人类知识构建过程，以发现药物-基因关系为例 

---
# ZhiFangDanTai: Fine-tuning Graph-based Retrieval-Augmented Generation Model for Traditional Chinese Medicine Formula 

**Title (ZH)**: 质方丹台：基于图的检索增强生成模型的微调研究（用于中药方剂） 

**Authors**: ZiXuan Zhang, Bowen Hao, Yingjie Li, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05867)  

**Abstract**: Traditional Chinese Medicine (TCM) formulas play a significant role in treating epidemics and complex diseases. Existing models for TCM utilize traditional algorithms or deep learning techniques to analyze formula relationships, yet lack comprehensive results, such as complete formula compositions and detailed explanations. Although recent efforts have used TCM instruction datasets to fine-tune Large Language Models (LLMs) for explainable formula generation, existing datasets lack sufficient details, such as the roles of the formula's sovereign, minister, assistant, courier; efficacy; contraindications; tongue and pulse diagnosis-limiting the depth of model outputs. To address these challenges, we propose ZhiFangDanTai, a framework combining Graph-based Retrieval-Augmented Generation (GraphRAG) with LLM fine-tuning. ZhiFangDanTai uses GraphRAG to retrieve and synthesize structured TCM knowledge into concise summaries, while also constructing an enhanced instruction dataset to improve LLMs' ability to integrate retrieved information. Furthermore, we provide novel theoretical proofs demonstrating that integrating GraphRAG with fine-tuning techniques can reduce generalization error and hallucination rates in the TCM formula task. Experimental results on both collected and clinical datasets demonstrate that ZhiFangDanTai achieves significant improvements over state-of-the-art models. Our model is open-sourced at this https URL. 

**Abstract (ZH)**: 中医方剂图基检索增强生成框架ZhiFangDanTai 

---
# GenAI on Wall Street -- Opportunities and Risk Controls 

**Title (ZH)**: GenAI在华尔街——机遇与风险控制 

**Authors**: Jackie Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.05841)  

**Abstract**: We give an overview on the emerging applications of GenAI in the financial industry, especially within investment banks. Inherent to these exciting opportunities is a new realm of risks that must be managed properly. By heeding both the Yin and Yang sides of GenAI, we can accelerate its organic growth while safeguarding the entire financial industry during this nascent era of AI. 

**Abstract (ZH)**: GenAI在金融行业，尤其是投资银行中的新兴应用：把握机遇，管理风险，促进行业发展 

---
# time2time: Causal Intervention in Hidden States to Simulate Rare Events in Time Series Foundation Models 

**Title (ZH)**: 时间到时间：在隐藏状态中进行因果干预以模拟时间序列基础模型中的罕见事件 

**Authors**: Debdeep Sanyal, Aaryan Nagpal, Dhruv Kumar, Murari Mandal, Saurabh Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2509.05801)  

**Abstract**: While transformer-based foundation models excel at forecasting routine patterns, two questions remain: do they internalize semantic concepts such as market regimes, or merely fit curves? And can their internal representations be leveraged to simulate rare, high-stakes events such as market crashes? To investigate this, we introduce activation transplantation, a causal intervention that manipulates hidden states by imposing the statistical moments of one event (e.g., a historical crash) onto another (e.g., a calm period) during the forward pass. This procedure deterministically steers forecasts: injecting crash semantics induces downturn predictions, while injecting calm semantics suppresses crashes and restores stability. Beyond binary control, we find that models encode a graded notion of event severity, with the latent vector norm directly correlating with the magnitude of systemic shocks. Validated across two architecturally distinct TSFMs, Toto (decoder only) and Chronos (encoder-decoder), our results demonstrate that steerable, semantically grounded representations are a robust property of large time series transformers. Our findings provide evidence for a latent concept space that governs model predictions, shifting interpretability from post-hoc attribution to direct causal intervention, and enabling semantic "what-if" analysis for strategic stress-testing. 

**Abstract (ZH)**: 基于变换器的基础模型在预测常规模式方面表现出色，但仍存在两个问题：它们是否会内部化市场制度等语义概念，还是仅仅拟合曲线？它们的内部表示能否用来模拟市场崩盘等罕见的高风险事件？为探究这一问题，我们引入了一种称为激活移植的因果干预方法，在前向传递过程中通过将一个事件（如历史崩盘）的统计特征施加到另一个事件（如平静时期）上来操纵隐藏状态。这种方法在确定性地引导预测：注入崩盘语义会引发衰退预测，而注入平静语义则会抑制崩盘并恢复稳定。除了二元控制，我们发现模型编码了事件严重性的分级概念，潜在向量的范数直接与系统性冲击的幅度相关。在两种具有不同架构的时间序列变换器（Toto 和 Chronos）中得到验证，我们的结果表明，可引导且具语义基础的表示是大规模时间序列变换器的稳健属性。我们的研究提供了证据，表明存在一个潜在的概念空间控制着模型的预测，将解释性从事后归因转变为直接因果干预，并使战略压力测试中的语义“假设情境”分析成为可能。 

---
# Hybrid Fourier Neural Operator-Plasma Fluid Model for Fast and Accurate Multiscale Simulations of High Power Microwave Breakdown 

**Title (ZH)**: 混合傅里叶神经运算子等离子体流体模型：快速准确的高功率微波击穿多尺度仿真 

**Authors**: Kalp Pandya, Pratik Ghosh, Ajeya Mandikal, Shivam Gandha, Bhaskar Chaudhury  

**Link**: [PDF](https://arxiv.org/pdf/2509.05799)  

**Abstract**: Modeling and simulation of High Power Microwave (HPM) breakdown, a multiscale phenomenon, is computationally expensive and requires solving Maxwell's equations (EM solver) coupled with a plasma continuity equation (plasma solver). In this work, we present a hybrid modeling approach that combines the accuracy of a differential equation-based plasma fluid solver with the computational efficiency of FNO (Fourier Neural Operator) based EM solver. Trained on data from an in-house FDTD-based plasma-fluid solver, the FNO replaces computationally expensive EM field updates, while the plasma solver governs the dynamic plasma response. The hybrid model is validated on microwave streamer formation, due to diffusion ionization mechanism, in a 2D scenario for unseen incident electric fields corresponding to entirely new plasma streamer simulations not included in model training, showing excellent agreement with FDTD based fluid simulations in terms of streamer shape, velocity, and temporal evolution. This hybrid FNO based strategy delivers significant acceleration of the order of 60X compared to traditional simulations for the specified problem size and offers an efficient alternative for computationally demanding multiscale and multiphysics simulations involved in HPM breakdown. Our work also demonstrate how such hybrid pipelines can be used to seamlessly to integrate existing C-based simulation codes with Python-based machine learning frameworks for simulations of plasma science and engineering problems. 

**Abstract (ZH)**: 基于FNO的混合模型方法在高功率微波（HPM）击穿多尺度现象建模与仿真中的应用研究 

---
# DCV-ROOD Evaluation Framework: Dual Cross-Validation for Robust Out-of-Distribution Detection 

**Title (ZH)**: DCV-ROOD评估框架：稳健的离分布检测双重交叉验证 

**Authors**: Arantxa Urrea-Castaño, Nicolás Segura-Kunsagi, Juan Luis Suárez-Díaz, Rosana Montes, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2509.05778)  

**Abstract**: Out-of-distribution (OOD) detection plays a key role in enhancing the robustness of artificial intelligence systems by identifying inputs that differ significantly from the training distribution, thereby preventing unreliable predictions and enabling appropriate fallback mechanisms. Developing reliable OOD detection methods is a significant challenge, and rigorous evaluation of these techniques is essential for ensuring their effectiveness, as it allows researchers to assess their performance under diverse conditions and to identify potential limitations or failure modes. Cross-validation (CV) has proven to be a highly effective tool for providing a reasonable estimate of the performance of a learning algorithm. Although OOD scenarios exhibit particular characteristics, an appropriate adaptation of CV can lead to a suitable evaluation framework for this setting. This work proposes a dual CV framework for robust evaluation of OOD detection models, aimed at improving the reliability of their assessment. The proposed evaluation framework aims to effectively integrate in-distribution (ID) and OOD data while accounting for their differing characteristics. To achieve this, ID data are partitioned using a conventional approach, whereas OOD data are divided by grouping samples based on their classes. Furthermore, we analyze the context of data with class hierarchy to propose a data splitting that considers the entire class hierarchy to obtain fair ID-OOD partitions to apply the proposed evaluation framework. This framework is called Dual Cross-Validation for Robust Out-of-Distribution Detection (DCV-ROOD). To test the validity of the evaluation framework, we selected a set of state-of-the-art OOD detection methods, both with and without outlier exposure. The results show that the method achieves very fast convergence to the true performance. 

**Abstract (ZH)**: 双交叉验证用于稳健的异常分布检测评估（DCV-ROOD） 

---
# Real-E: A Foundation Benchmark for Advancing Robust and Generalizable Electricity Forecasting 

**Title (ZH)**: Real-E：促进鲁棒性和泛化能力电能预测的基础基准 

**Authors**: Chen Shao, Yue Wang, Zhenyi Zhu, Zhanbo Huang, Sebastian Pütz, Benjamin Schäfer, Tobais Käfer, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2509.05768)  

**Abstract**: Energy forecasting is vital for grid reliability and operational efficiency. Although recent advances in time series forecasting have led to progress, existing benchmarks remain limited in spatial and temporal scope and lack multi-energy features. This raises concerns about their reliability and applicability in real-world deployment. To address this, we present the Real-E dataset, covering over 74 power stations across 30+ European countries over a 10-year span with rich metadata. Using Real- E, we conduct an extensive data analysis and benchmark over 20 baselines across various model types. We introduce a new metric to quantify shifts in correlation structures and show that existing methods struggle on our dataset, which exhibits more complex and non-stationary correlation dynamics. Our findings highlight key limitations of current methods and offer a strong empirical basis for building more robust forecasting models 

**Abstract (ZH)**: 能源预测对于电网可靠性和运营效率至关重要。尽管近期时间序列预测的进展取得了进步，现有的基准在空间和时间范围上仍有限制，并且缺乏多能源特征。这引起了对其在实际部署中的可靠性和适用性的担忧。为应对这一挑战，我们介绍了Real-E数据集，该数据集涵盖了来自30多个欧洲国家的超过74个发电站，时间跨度为10年，并附有丰富的元数据。使用Real-E，我们进行了广泛的数据分析，并在各种模型类型中超过20种基线方法上进行了基准测试。我们引入了一个新的度量标准来量化相关结构的转变，并展示了现有方法在我们数据集上表现不佳，该数据集表现出更复杂和非稳定的相关动态。我们的研究结果突显了当前方法的关键局限性，并为构建更稳健的预测模型提供了强有力的实证基础。 

---
# Tell-Tale Watermarks for Explanatory Reasoning in Synthetic Media Forensics 

**Title (ZH)**: Tell-Tale水印在合成媒体鉴伪中的解释性推理中应用 

**Authors**: Ching-Chun Chang, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2509.05753)  

**Abstract**: The rise of synthetic media has blurred the boundary between reality and fabrication under the evolving power of artificial intelligence, fueling an infodemic that erodes public trust in cyberspace. For digital imagery, a multitude of editing applications further complicates the forensic analysis, including semantic edits that alter content, photometric adjustments that recalibrate colour characteristics, and geometric projections that reshape viewpoints. Collectively, these transformations manipulate and control perceptual interpretation of digital imagery. This susceptibility calls for forensic enquiry into reconstructing the chain of events, thereby revealing deeper evidential insight into the presence or absence of criminal intent. This study seeks to address an inverse problem of tracing the underlying generation chain that gives rise to the observed synthetic media. A tell-tale watermarking system is developed for explanatory reasoning over the nature and extent of transformations across the lifecycle of synthetic media. Tell-tale watermarks are tailored to different classes of transformations, responding in a manner that is neither strictly robust nor fragile but instead interpretable. These watermarks function as reference clues that evolve under the same transformation dynamics as the carrier media, leaving interpretable traces when subjected to transformations. Explanatory reasoning is then performed to infer the most plausible account across the combinatorial parameter space of composite transformations. Experimental evaluations demonstrate the validity of tell-tale watermarking with respect to fidelity, synchronicity and traceability. 

**Abstract (ZH)**: 合成媒体的兴起模糊了现实与伪造之间的界限，在人工智能不断演进的力量下引发了信息疫情，侵蚀了网络空间中的公众信任。对于数字图像而言，众多编辑应用进一步复杂化了法医分析，包括语义编辑、光度调整和几何投影等变换，这些变换操纵和控制着数字图像的知觉解释。这种易受操控性要求进行法医调查以重建事件链，从而揭示犯罪意图存在的证据。本研究旨在解决合成媒体生成链条追溯的逆问题。开发了一种告示水印系统，用于解释性推理不同类型的变换在整个合成媒体生命周期中的范围和性质。这些告示水印针对不同的变换类别进行定制，既不具备严格鲁棒性也不具备脆弱性，而是具备可解释性。这些水印作为参考线索，在跟随载体媒体的变换动态演化后，在受到变换时留下可解释的痕迹。通过解释性推理，在组合变换参数空间中推断出最有可能的情况。实验评估证明了告示水印在保真度、同步性和可追溯性方面的有效性。 

---
# Offline vs. Online Learning in Model-based RL: Lessons for Data Collection Strategies 

**Title (ZH)**: 基于模型的强化学习中离线学习与在线学习的比较：数据收集策略的启示 

**Authors**: Jiaqi Chen, Ji Shi, Cansu Sancaktar, Jonas Frey, Georg Martius  

**Link**: [PDF](https://arxiv.org/pdf/2509.05735)  

**Abstract**: Data collection is crucial for learning robust world models in model-based reinforcement learning. The most prevalent strategies are to actively collect trajectories by interacting with the environment during online training or training on offline datasets. At first glance, the nature of learning task-agnostic environment dynamics makes world models a good candidate for effective offline training. However, the effects of online vs. offline data on world models and thus on the resulting task performance have not been thoroughly studied in the literature. In this work, we investigate both paradigms in model-based settings, conducting experiments on 31 different environments. First, we showcase that online agents outperform their offline counterparts. We identify a key challenge behind performance degradation of offline agents: encountering Out-Of-Distribution states at test time. This issue arises because, without the self-correction mechanism in online agents, offline datasets with limited state space coverage induce a mismatch between the agent's imagination and real rollouts, compromising policy training. We demonstrate that this issue can be mitigated by allowing for additional online interactions in a fixed or adaptive schedule, restoring the performance of online training with limited interaction data. We also showcase that incorporating exploration data helps mitigate the performance degradation of offline agents. Based on our insights, we recommend adding exploration data when collecting large datasets, as current efforts predominantly focus on expert data alone. 

**Abstract (ZH)**: 基于模型的强化学习中世界模型的数据收集至关重要：在线数据与离线数据的影响研究 

---
# A Survey of the State-of-the-Art in Conversational Question Answering Systems 

**Title (ZH)**: 当前对话式问答系统综述 

**Authors**: Manoj Madushanka Perera, Adnan Mahmood, Kasun Eranda Wijethilake, Fahmida Islam, Maryam Tahermazandarani, Quan Z. Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.05716)  

**Abstract**: Conversational Question Answering (ConvQA) systems have emerged as a pivotal area within Natural Language Processing (NLP) by driving advancements that enable machines to engage in dynamic and context-aware conversations. These capabilities are increasingly being applied across various domains, i.e., customer support, education, legal, and healthcare where maintaining a coherent and relevant conversation is essential. Building on recent advancements, this survey provides a comprehensive analysis of the state-of-the-art in ConvQA. This survey begins by examining the core components of ConvQA systems, i.e., history selection, question understanding, and answer prediction, highlighting their interplay in ensuring coherence and relevance in multi-turn conversations. It further investigates the use of advanced machine learning techniques, including but not limited to, reinforcement learning, contrastive learning, and transfer learning to improve ConvQA accuracy and efficiency. The pivotal role of large language models, i.e., RoBERTa, GPT-4, Gemini 2.0 Flash, Mistral 7B, and LLaMA 3, is also explored, thereby showcasing their impact through data scalability and architectural advancements. Additionally, this survey presents a comprehensive analysis of key ConvQA datasets and concludes by outlining open research directions. Overall, this work offers a comprehensive overview of the ConvQA landscape and provides valuable insights to guide future advancements in the field. 

**Abstract (ZH)**: 基于对话的问答（ConvQA）系统已成为自然语言处理（NLP）中一个关键领域，通过推动使机器能够进行动态和上下文相关的对话。这些能力正越来越多地被应用到客户支持、教育、法律和医疗等领域，其中保持连贯和相关对话至关重要。本综述基于近期进展，对 ConvQA 的最新状态进行了全面分析。本综述首先审视了 ConvQA 系统的核心组件，即历史选择、问题理解与答案预测，并强调了它们如何相互作用以确保多轮对话的连贯性和相关性。随后进一步探讨了使用强化学习、对比学习和迁移学习等高级机器学习技术以提高 ConvQA 的准确性和效率。同时，还探讨了大型语言模型，如 RoBERTa、GPT-4、Gemini 2.0 Flash、Mistral 7B 和 LLaMA 3 的关键作用，并展示了它们通过数据规模扩展和架构进展所产生的重要影响。此外，本综述还对关键的 ConvQA 数据集进行了全面分析，并概述了未来研究方向。总体而言，本研究工作提供了 ConvQA 场景的全面概述，并提供了对未来领域发展的宝贵见解。 

---
# Revealing the Numeracy Gap: An Empirical Investigation of Text Embedding Models 

**Title (ZH)**: 揭示 numeracy 隙口：文本嵌入模型的实证研究 

**Authors**: Ningyuan Deng, Hanyu Duan, Yixuan Tang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05691)  

**Abstract**: Text embedding models are widely used in natural language processing applications. However, their capability is often benchmarked on tasks that do not require understanding nuanced numerical information in text. As a result, it remains unclear whether current embedding models can precisely encode numerical content, such as numbers, into embeddings. This question is critical because embedding models are increasingly applied in domains where numbers matter, such as finance and healthcare. For example, Company X's market share grew by 2\% should be interpreted very differently from Company X's market share grew by 20\%, even though both indicate growth in market share. This study aims to examine whether text embedding models can capture such nuances. Using synthetic data in a financial context, we evaluate 13 widely used text embedding models and find that they generally struggle to capture numerical details accurately. Our further analyses provide deeper insights into embedding numeracy, informing future research to strengthen embedding model-based NLP systems with improved capacity for handling numerical content. 

**Abstract (ZH)**: 文本嵌入模型在自然语言处理应用中广泛应用，但其能力往往通过不需理解文本中细微数值信息的任务进行评估。这使得人们不清楚当前的嵌入模型是否能够精确地将数字内容，如数字，编码到嵌入式表示中。由于嵌入模型在涉及数字的重要领域，如金融和医疗保健中被越来越多地应用，这个问题尤为重要。例如，Company X的市场份额增长了2%和增长了20%应被非常不同地解释，尽管两者都表明市场份额的增长。本研究旨在探讨文本嵌入模型是否能够捕捉到这种细微差别。使用金融背景下的人工合成数据，我们评估了13种广泛使用的文本嵌入模型，并发现它们在准确捕捉数值细节方面普遍存在问题。进一步分析提供了对嵌入数值能力的更深入见解，为未来研究加强基于嵌入模型的自然语言处理系统的处理数值内容的能力提供了指导。 

---
# SEASONED: Semantic-Enhanced Self-Counterfactual Explainable Detection of Adversarial Exploiter Contracts 

**Title (ZH)**: SEASONED: 语义增强自反事实可解释 adversarial合约检测 

**Authors**: Xng Ai, Shudan Lin, Zecheng Li, Kai Zhou, Bixin Li, Bin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.05681)  

**Abstract**: Decentralized Finance (DeFi) attacks have resulted in significant losses, often orchestrated through Adversarial Exploiter Contracts (AECs) that exploit vulnerabilities in victim smart contracts. To proactively identify such threats, this paper targets the explainable detection of AECs.
Existing detection methods struggle to capture semantic dependencies and lack interpretability, limiting their effectiveness and leaving critical knowledge gaps in AEC analysis. To address these challenges, we introduce SEASONED, an effective, self-explanatory, and robust framework for AEC detection.
SEASONED extracts semantic information from contract bytecode to construct a semantic relation graph (SRG), and employs a self-counterfactual explainable detector (SCFED) to classify SRGs and generate explanations that highlight the core attack logic. SCFED further enhances robustness, generalizability, and data efficiency by extracting representative information from these explanations. Both theoretical analysis and experimental results demonstrate the effectiveness of SEASONED, which showcases outstanding detection performance, robustness, generalizability, and data efficiency learning ability. To support further research, we also release a new dataset of 359 AECs. 

**Abstract (ZH)**: 去中心化金融(DeFi)攻击导致了重大损失，often orchestrated through 对抗性 exploit者合约(AECs)，这些合约利用了受害智能合约中的漏洞。为了提前识别此类威胁，本文旨在解释性检测AECs。
现有的检测方法难以捕捉语义依赖关系并且缺乏可解释性，限制了其有效性并留下了AEC分析中的关键知识缺口。为了应对这些挑战，我们提出了SEASONED，这是一种有效、自我解释且稳健的AEC检测框架。
SEASONED 从合约字节码中提取语义信息以构建语义关系图(SRG)，并采用自我反事实可解释检测器(SCFED)对SRGs进行分类并生成突出核心攻击逻辑的解释。SCFED 进一步通过从这些解释中提取代表性信息增强了稳健性、通用性和数据效率。理论分析和实验结果均证明了SEASONED的有效性，展示了其出色的检测性能、稳健性、通用性和高效的数据学习能力。为支持进一步研究，我们还发布了包含359个AEC的新数据集。 

---
# GraMFedDHAR: Graph Based Multimodal Differentially Private Federated HAR 

**Title (ZH)**: 基于图的多模态差异隐私联邦健康行为识别 

**Authors**: Labani Halder, Tanmay Sen, Sarbani Palit  

**Link**: [PDF](https://arxiv.org/pdf/2509.05671)  

**Abstract**: Human Activity Recognition (HAR) using multimodal sensor data remains challenging due to noisy or incomplete measurements, scarcity of labeled examples, and privacy concerns. Traditional centralized deep learning approaches are often constrained by infrastructure availability, network latency, and data sharing restrictions. While federated learning (FL) addresses privacy by training models locally and sharing only model parameters, it still has to tackle issues arising from the use of heterogeneous multimodal data and differential privacy requirements. In this article, a Graph-based Multimodal Federated Learning framework, GraMFedDHAR, is proposed for HAR tasks. Diverse sensor streams such as a pressure mat, depth camera, and multiple accelerometers are modeled as modality-specific graphs, processed through residual Graph Convolutional Neural Networks (GCNs), and fused via attention-based weighting rather than simple concatenation. The fused embeddings enable robust activity classification, while differential privacy safeguards data during federated aggregation. Experimental results show that the proposed MultiModalGCN model outperforms the baseline MultiModalFFN, with up to 2 percent higher accuracy in non-DP settings in both centralized and federated paradigms. More importantly, significant improvements are observed under differential privacy constraints: MultiModalGCN consistently surpasses MultiModalFFN, with performance gaps ranging from 7 to 13 percent depending on the privacy budget and setting. These results highlight the robustness of graph-based modeling in multimodal learning, where GNNs prove more resilient to the performance degradation introduced by DP noise. 

**Abstract (ZH)**: 基于图的多模态联邦学习框架GraMFedDHAR用于人体活动识别 

---
# OptiProxy-NAS: Optimization Proxy based End-to-End Neural Architecture Search 

**Title (ZH)**: OptiProxy-NAS: 基于优化代理的端到端神经架构搜索 

**Authors**: Bo Lyu, Yu Cui, Tuo Shi, Ke Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05656)  

**Abstract**: Neural architecture search (NAS) is a hard computationally expensive optimization problem with a discrete, vast, and spiky search space. One of the key research efforts dedicated to this space focuses on accelerating NAS via certain proxy evaluations of neural architectures. Different from the prevalent predictor-based methods using surrogate models and differentiable architecture search via supernetworks, we propose an optimization proxy to streamline the NAS as an end-to-end optimization framework, named OptiProxy-NAS. In particular, using a proxy representation, the NAS space is reformulated to be continuous, differentiable, and smooth. Thereby, any differentiable optimization method can be applied to the gradient-based search of the relaxed architecture parameters. Our comprehensive experiments on $12$ NAS tasks of $4$ search spaces across three different domains including computer vision, natural language processing, and resource-constrained NAS fully demonstrate the superior search results and efficiency. Further experiments on low-fidelity scenarios verify the flexibility. 

**Abstract (ZH)**: 神经架构搜索（NAS）是一个计算上昂贵的离散、庞大且不连续的优化问题。致力于这一空间的关键研究工作之一是通过某些代理评估加速NAS。不同于基于预测器的方法使用替代模型和通过超网络进行可微分的架构搜索，我们提出了一种优化代理，将其命名为OptiProxy-NAS，以将NAS直接作为端到端的优化框架进行优化。特别是，使用代理表示，NAS空间被重新表述为连续、可微分且平滑的。因此，任何可微分的优化方法都可以应用于对放松的架构参数的梯度搜索。在三个不同领域（计算机视觉、自然语言处理和资源受限的NAS）的四个搜索空间下的12个NAS任务上的全面实验充分证明了其优越的搜索结果和效率。进一步在低保真场景下的实验验证了其灵活性。 

---
# Orchestrator: Active Inference for Multi-Agent Systems in Long-Horizon Tasks 

**Title (ZH)**: orchestrator: 长时_horizon 任务中多智能体系统的主动推断 

**Authors**: Lukas Beckenbauer, Johannes-Lucas Loewe, Ge Zheng, Alexandra Brintrup  

**Link**: [PDF](https://arxiv.org/pdf/2509.05651)  

**Abstract**: Complex, non-linear tasks challenge LLM-enhanced multi-agent systems (MAS) due to partial observability and suboptimal coordination. We propose Orchestrator, a novel MAS framework that leverages attention-inspired self-emergent coordination and reflective benchmarking to optimize global task performance. Orchestrator introduces a monitoring mechanism to track agent-environment dynamics, using active inference benchmarks to optimize system behavior. By tracking agent-to-agent and agent-to-environment interaction, Orchestrator mitigates the effects of partial observability and enables agents to approximate global task solutions more efficiently. We evaluate the framework on a series of maze puzzles of increasing complexity, demonstrating its effectiveness in enhancing coordination and performance in dynamic, non-linear environments with long-horizon objectives. 

**Abstract (ZH)**: 复杂的非线性任务挑战了基于LLM增强的多智能体系统（MAS）的能力，由于半可观测性和次优协调。我们提出了一种名为Orchestrator的新型MAS框架，该框架利用注意力启发式的自我涌现协调和反思型基准测试来优化全局任务性能。Orchestrator引入了一种监控机制来跟踪智能体-环境动态，并使用主动推断基准测试来优化系统行为。通过跟踪智能体间的交互和智能体与环境的交互，Orchestrator缓解了半可观测性的影响，使智能体能够更高效地逼近全局任务解决方案。我们在一系列复杂度递增的迷宫谜题上评估了该框架，证明了其在具有长期目标的动态、非线性环境中的协调性和性能提升效果。 

---
# Self-supervised Learning for Hyperspectral Images of Trees 

**Title (ZH)**: 树冠_hyper spectral图像的自监督学习 

**Authors**: Moqsadur Rahman, Saurav Kumar, Santosh S. Palmate, M. Shahriar Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2509.05630)  

**Abstract**: Aerial remote sensing using multispectral and RGB imagers has provided a critical impetus to precision agriculture. Analysis of the hyperspectral images with limited or no labels is challenging. This paper focuses on self-supervised learning to create neural network embeddings reflecting vegetation properties of trees from aerial hyperspectral images of crop fields. Experimental results demonstrate that a constructed tree representation, using a vegetation property-related embedding space, performs better in downstream machine learning tasks compared to the direct use of hyperspectral vegetation properties as tree representations. 

**Abstract (ZH)**: 基于多光谱和RGB成像的航空遥感为精准农业提供了关键动力。分析标注有限或无标注的高光谱图像具有挑战性。本文专注于自监督学习以从农田的航空高光谱图像中创建反映树木植被属性的神经网络嵌入。实验结果表明，使用与植被属性相关的嵌入空间构建的树木表示，在下游机器学习任务中表现优于直接使用高光谱植被属性作为树木表示。 

---
# Natural Language-Programming Language Software Traceability Link Recovery Needs More than Textual Similarity 

**Title (ZH)**: 自然语言编程语言软件可追溯性链接的恢复需要超出文本相似性。 

**Authors**: Zhiyuan Zou, Bangchao Wang, Peng Liang, Tingting Bi, Huan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05585)  

**Abstract**: In the field of software traceability link recovery (TLR), textual similarity has long been regarded as the core criterion. However, in tasks involving natural language and programming language (NL-PL) artifacts, relying solely on textual similarity is limited by their semantic gap. To this end, we conducted a large-scale empirical evaluation across various types of TLR tasks, revealing the limitations of textual similarity in NL-PL scenarios. To address these limitations, we propose an approach that incorporates multiple domain-specific auxiliary strategies, identified through empirical analysis, into two models: the Heterogeneous Graph Transformer (HGT) via edge types and the prompt-based Gemini 2.5 Pro via additional input information. We then evaluated our approach using the widely studied requirements-to-code TLR task, a representative case of NL-PL TLR. Experimental results show that both the multi-strategy HGT and Gemini 2.5 Pro models outperformed their original counterparts without strategy integration. Furthermore, compared to the current state-of-the-art method HGNNLink, the multi-strategy HGT and Gemini 2.5 Pro models achieved average F1-score improvements of 3.68% and 8.84%, respectively, across twelve open-source projects, demonstrating the effectiveness of multi-strategy integration in enhancing overall model performance for the requirements-code TLR task. 

**Abstract (ZH)**: 在软件追溯链接恢复（TLR）领域，文本相似性长期以来一直被视为核心标准。但在涉及自然语言和编程语言（NL-PL）制品的任务中，仅依赖文本相似性受到语义差距的限制。为此，我们在多种类型的TLR任务中进行了大规模的实证评估，揭示了文本相似性在NL-PL情景下的局限性。为应对这些局限性，我们提出了一种方法，将通过实证分析识别出的多种领域特定辅助策略整合到两种模型中：通过边类型实现的异质图变换器（HGT）和基于提示的Gemini 2.5 Pro，后者通过附加输入信息。我们使用广泛研究的需求到代码的TLR任务对这种方法进行了评估，这是一个典型的NL-PL TLR案例。实验结果表明，多策略HGT和Gemini 2.5 Pro模型在没有策略整合的情况下均优于其原来的版本。此外，与当前最先进的方法HGNNLink相比，多策略HGT和Gemini 2.5 Pro模型在12个开源项目中分别实现了平均F1分数提升3.68%和8.84%，证实了多策略整合在提高需求到代码TLR任务的整体模型性能方面的有效性。 

---
# MambaLite-Micro: Memory-Optimized Mamba Inference on MCUs 

**Title (ZH)**: MambaLite-Micro：MCUs上优化内存的Mamba推断 

**Authors**: Hongjun Xu, Junxi Xia, Weisi Yang, Yueyuan Sui, Stephen Xia  

**Link**: [PDF](https://arxiv.org/pdf/2509.05488)  

**Abstract**: Deploying Mamba models on microcontrollers (MCUs) remains challenging due to limited memory, the lack of native operator support, and the absence of embedded-friendly toolchains. We present, to our knowledge, the first deployment of a Mamba-based neural architecture on a resource-constrained MCU, a fully C-based runtime-free inference engine: MambaLite-Micro. Our pipeline maps a trained PyTorch Mamba model to on-device execution by (1) exporting model weights into a lightweight format, and (2) implementing a handcrafted Mamba layer and supporting operators in C with operator fusion and memory layout optimization. MambaLite-Micro eliminates large intermediate tensors, reducing 83.0% peak memory, while maintaining an average numerical error of only 1.7x10-5 relative to the PyTorch Mamba implementation. When evaluated on keyword spotting(KWS) and human activity recognition (HAR) tasks, MambaLite-Micro achieved 100% consistency with the PyTorch baselines, fully preserving classification accuracy. We further validated portability by deploying on both ESP32S3 and STM32H7 microcontrollers, demonstrating consistent operation across heterogeneous embedded platforms and paving the way for bringing advanced sequence models like Mamba to real-world resource-constrained applications. 

**Abstract (ZH)**: 将基于Mamba的神经架构部署在微控制器（MCUs）上仍然具有挑战性，受限于有限的内存、缺乏内置操作支持和缺失的嵌入式友好工具链。我们提出了迄今为止首个在资源受限的MCU上部署基于Mamba的神经架构的方法：一个完全基于C语言的无运行时推理引擎MambaLite-Micro。我们的流程通过（1）将模型权重导出为轻量级格式，（2）在C语言中手工实现Mamba层及其支持的操作，并通过操作融合和内存布局优化来实现。MambaLite-Micro消除了大量中间张量，峰值内存减少了83.0%，同时相对PyTorch Mamba实现的平均数值误差仅为1.7x10^-5。在关键词识别（KWS）和人体活动识别（HAR）任务上评估时，MambaLite-Micro与PyTorch基准保持了100%的一致性，完全保留了分类准确率。此外，我们进一步验证了其可移植性，分别部署在ESP32S3和STM32H7微控制器上，展示了其在异构嵌入式平台上的稳定运行，并为将如Mamba等高级序列模型引入实际资源受限的应用铺平了道路。 

---
# PLanTS: Periodicity-aware Latent-state Representation Learning for Multivariate Time Series 

**Title (ZH)**: PLanTS: 具有期hythm意识的多变量时间序列潜在状态表示学习 

**Authors**: Jia Wang, Xiao Wang, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05478)  

**Abstract**: Multivariate time series (MTS) are ubiquitous in domains such as healthcare, climate science, and industrial monitoring, but their high dimensionality, limited labeled data, and non-stationary nature pose significant challenges for conventional machine learning methods. While recent self-supervised learning (SSL) approaches mitigate label scarcity by data augmentations or time point-based contrastive strategy, they neglect the intrinsic periodic structure of MTS and fail to capture the dynamic evolution of latent states. We propose PLanTS, a periodicity-aware self-supervised learning framework that explicitly models irregular latent states and their transitions. We first designed a period-aware multi-granularity patching mechanism and a generalized contrastive loss to preserve both instance-level and state-level similarities across multiple temporal resolutions. To further capture temporal dynamics, we design a next-transition prediction pretext task that encourages representations to encode predictive information about future state evolution. We evaluate PLanTS across a wide range of downstream tasks-including multi-class and multi-label classification, forecasting, trajectory tracking and anomaly detection. PLanTS consistently improves the representation quality over existing SSL methods and demonstrates superior runtime efficiency compared to DTW-based methods. 

**Abstract (ZH)**: 周期性意识自监督学习框架PLanTS： explicit建模不规则潜状态及其转换 

---
# From Vision to Validation: A Theory- and Data-Driven Construction of a GCC-Specific AI Adoption Index 

**Title (ZH)**: 从视觉到验证：一种基于理论和数据的GCC特定AI adoption指数构建方法 

**Authors**: Mohammad Rashed Albous, Anwaar AlKandari, Abdel Latef Anouze  

**Link**: [PDF](https://arxiv.org/pdf/2509.05474)  

**Abstract**: Artificial intelligence (AI) is rapidly transforming public-sector processes worldwide, yet standardized measures rarely address the unique drivers, governance models, and cultural nuances of the Gulf Cooperation Council (GCC) countries. This study employs a theory-driven foundation derived from an in-depth analysis of literature review and six National AI Strategies (NASs), coupled with a data-driven approach that utilizes a survey of 203 mid- and senior-level government employees and advanced statistical techniques (K-Means clustering, Principal Component Analysis, and Partial Least Squares Structural Equation Modeling). By combining policy insights with empirical evidence, the research develops and validates a novel AI Adoption Index specifically tailored to the GCC public sector. Findings indicate that robust infrastructure and clear policy mandates exert the strongest influence on successful AI implementations, overshadowing organizational readiness in early adoption stages. The combined model explains 70% of the variance in AI outcomes, suggesting that resource-rich environments and top-down policy directives can drive rapid but uneven technology uptake. By consolidating key dimensions (Infrastructure & Resources, Organizational Readiness, and Policy & Regulatory Environment) into a single composite index, this study provides a holistic yet context-sensitive tool for benchmarking AI maturity. The index offers actionable guidance for policymakers seeking to harmonize large-scale deployments with ethical and regulatory standards. Beyond advancing academic discourse, these insights inform more strategic allocation of resources, cross-country cooperation, and capacity-building initiatives, thereby supporting sustained AI-driven transformation in the GCC region and beyond. 

**Abstract (ZH)**: 人工智能（AI）正快速改变全球公共部门的过程，然而标准化的衡量标准很少考虑海湾合作委员会（GCC）国家的独特驱动力、治理模式和文化差异。本研究基于深入的文献综述和六个国家人工智能战略（NASs）的理论驱动基础，并结合利用203名中高层政府员工的调查数据和高级统计技术（K-Means聚类、主成分分析和偏最小二乘结构方程建模）的数据驱动方法。通过将政策见解与实证证据相结合，该研究开发并验证了一个专门针对GCC公共部门的人工智能采用指数。研究发现，坚固的基础设施和明确的政策指令对成功的人工智能实施具有最强的影响，早期采用阶段组织准备度的影响较小。该综合模型解释了70%的人工智能结果的差异，表明资源丰富的环境和自上而下的政策指令可以推动快速但不均衡的技术采用。通过将关键维度（基础设施与资源、组织准备度和政策与监管环境）整合为单一复合指数，本研究提供了一个既全面又具有情境敏感性的工具，用于评估人工智能成熟度。该指数为寻求实现大规模部署与伦理和监管标准和谐一致的政策制定者提供了可操作的指导。这些见解不仅推动了学术讨论，还为跨国有序资源分配、合作及能力提升倡议提供了信息，从而支持GCC地区乃至更广泛的地区的人工智能驱动型持续转型。 

---
# Newton to Einstein: Axiom-Based Discovery via Game Design 

**Title (ZH)**: 牛顿到爱因斯坦：基于公理的游戏设计驱动发现 

**Authors**: Pingchuan Ma, Benjamin Tod Jones, Tsun-Hsuan Wang, Minghao Guo, Michal Piotr Lipiec, Chuang Gan, Wojciech Matusik  

**Link**: [PDF](https://arxiv.org/pdf/2509.05448)  

**Abstract**: This position paper argues that machine learning for scientific discovery should shift from inductive pattern recognition to axiom-based reasoning. We propose a game design framework in which scientific inquiry is recast as a rule-evolving system: agents operate within environments governed by axioms and modify them to explain outlier observations. Unlike conventional ML approaches that operate within fixed assumptions, our method enables the discovery of new theoretical structures through systematic rule adaptation. We demonstrate the feasibility of this approach through preliminary experiments in logic-based games, showing that agents can evolve axioms that solve previously unsolvable problems. This framework offers a foundation for building machine learning systems capable of creative, interpretable, and theory-driven discovery. 

**Abstract (ZH)**: 机器学习在科学研究中的应用应从归纳模式识别转向基于公理的推理：一种基于规则演化系统的游戏设计框架 

---
# No Translation Needed: Forecasting Quality from Fertility and Metadata 

**Title (ZH)**: 从生育率和元数据预测质量 

**Authors**: Jessica M. Lundin, Ada Zhang, David Adelani, Cody Carroll  

**Link**: [PDF](https://arxiv.org/pdf/2509.05425)  

**Abstract**: We show that translation quality can be predicted with surprising accuracy \textit{without ever running the translation system itself}. Using only a handful of features, token fertility ratios, token counts, and basic linguistic metadata (language family, script, and region), we can forecast ChrF scores for GPT-4o translations across 203 languages in the FLORES-200 benchmark. Gradient boosting models achieve favorable performance ($R^{2}=0.66$ for XX$\rightarrow$English and $R^{2}=0.72$ for English$\rightarrow$XX). Feature importance analyses reveal that typological factors dominate predictions into English, while fertility plays a larger role for translations into diverse target languages. These findings suggest that translation quality is shaped by both token-level fertility and broader linguistic typology, offering new insights for multilingual evaluation and quality estimation. 

**Abstract (ZH)**: 我们展示了在从未运行翻译系统的情况下，翻译质量可以用惊人的准确性进行预测。仅使用少量特征，如标记生育率、标记计数以及基本的语言元数据（语言家族、书写系统和地区），我们可以在FLORES-200基准中的203种语言中预测GPT-4o翻译的ChrF评分。梯度提升模型在XX到英语的预测中表现出色（$R^{2}=0.66$），在英语到XX的预测中表现更好（$R^{2}=0.72$）。特征重要性分析表明，类型学因素在翻译成英语时占主导地位，而生育率对多种目标语言的翻译起着更大的作用。这些发现表明，翻译质量受到标记水平生育率和更广泛语言类型学的影响，为多语言评估和质量估计提供了新的见解。 

---
# Universality of physical neural networks with multivariate nonlinearity 

**Title (ZH)**: 多元非线性下的物理神经网络普遍性 

**Authors**: Benjamin Savinson, David J. Norris, Siddhartha Mishra, Samuel Lanthaler  

**Link**: [PDF](https://arxiv.org/pdf/2509.05420)  

**Abstract**: The enormous energy demand of artificial intelligence is driving the development of alternative hardware for deep learning. Physical neural networks try to exploit physical systems to perform machine learning more efficiently. In particular, optical systems can calculate with light using negligible energy. While their computational capabilities were long limited by the linearity of optical materials, nonlinear computations have recently been demonstrated through modified input encoding. Despite this breakthrough, our inability to determine if physical neural networks can learn arbitrary relationships between data -- a key requirement for deep learning known as universality -- hinders further progress. Here we present a fundamental theorem that establishes a universality condition for physical neural networks. It provides a powerful mathematical criterion that imposes device constraints, detailing how inputs should be encoded in the tunable parameters of the physical system. Based on this result, we propose a scalable architecture using free-space optics that is provably universal and achieves high accuracy on image classification tasks. Further, by combining the theorem with temporal multiplexing, we present a route to potentially huge effective system sizes in highly practical but poorly scalable on-chip photonic devices. Our theorem and scaling methods apply beyond optical systems and inform the design of a wide class of universal, energy-efficient physical neural networks, justifying further efforts in their development. 

**Abstract (ZH)**: 物理神经网络的通用性条件及其实现方法 

---
# Graph Connectionist Temporal Classification for Phoneme Recognition 

**Title (ZH)**: 图连接主义时序分类在音素识别中的应用 

**Authors**: Henry Grafé, Hugo Van hamme  

**Link**: [PDF](https://arxiv.org/pdf/2509.05399)  

**Abstract**: Automatic Phoneme Recognition (APR) systems are often trained using pseudo phoneme-level annotations generated from text through Grapheme-to-Phoneme (G2P) systems. These G2P systems frequently output multiple possible pronunciations per word, but the standard Connectionist Temporal Classification (CTC) loss cannot account for such ambiguity during training. In this work, we adapt Graph Temporal Classification (GTC) to the APR setting. GTC enables training from a graph of alternative phoneme sequences, allowing the model to consider multiple pronunciations per word as valid supervision. Our experiments on English and Dutch data sets show that incorporating multiple pronunciations per word into the training loss consistently improves phoneme error rates compared to a baseline trained with CTC. These results suggest that integrating pronunciation variation into the loss function is a promising strategy for training APR systems from noisy G2P-based supervision. 

**Abstract (ZH)**: 自动音位识别（APR）系统常常使用通过图形到音位（G2P）系统从文本生成的伪音位级注释进行训练。这些G2P系统经常为每个单词输出多个可能的发音，但标准的连接主义时序分类（CTC）损失在训练过程中无法处理这种不确定性。在本工作中，我们将图形时序分类（GTC）应用于APR设置。GTC允许从多个音位序列的图形中进行训练，使模型能够将每个单词的多种发音视为有效的监督。我们的实验结果显示，在英语和荷兰数据集上，将每个单词的多种发音纳入训练损失中，相对于使用CTC训练的基准模型，可以一致地提高音位错误率。这些结果表明，在损失函数中整合发音变异是有希望的策略，用于从嘈杂的G2P基础监督中训练APR系统。 

---
# Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate 

**Title (ZH)**: 谈不一定免费：理解多agent辩论中的故障模式 

**Authors**: Andrea Wynn, Harsh Satija, Gillian Hadfield  

**Link**: [PDF](https://arxiv.org/pdf/2509.05396)  

**Abstract**: While multi-agent debate has been proposed as a promising strategy for improving AI reasoning ability, we find that debate can sometimes be harmful rather than helpful. The prior work has exclusively focused on debates within homogeneous groups of agents, whereas we explore how diversity in model capabilities influences the dynamics and outcomes of multi-agent interactions. Through a series of experiments, we demonstrate that debate can lead to a decrease in accuracy over time -- even in settings where stronger (i.e., more capable) models outnumber their weaker counterparts. Our analysis reveals that models frequently shift from correct to incorrect answers in response to peer reasoning, favoring agreement over challenging flawed reasoning. These results highlight important failure modes in the exchange of reasons during multi-agent debate, suggesting that naive applications of debate may cause performance degradation when agents are neither incentivized nor adequately equipped to resist persuasive but incorrect reasoning. 

**Abstract (ZH)**: 多智能体辩论虽然被提出作为一种提高AI推理能力的有前景策略，但我们发现辩论有时可能是有害的而非有益的。先前的工作仅专注于同质智能体群体内的辩论，而我们探讨了模型能力多样性的存在如何影响多智能体互动的动力学和结果。通过一系列实验，我们证明了即使在更强（即更具备能力）的模型多于较弱模型的情况下，辩论也会导致准确性的下降。我们的分析表明，模型经常因同伴推理而从正确答案变为错误答案，偏好一致而非挑战错误推理。这些结果强调了多智能体辩论期间理由交换中的重要失灵模式，表明在代理缺乏适当激励或充分准备以抗拒有 persuasiveness但错误的推理时，简单的辩论应用可能导致性能下降。 

---
# Reverse Browser: Vector-Image-to-Code Generator 

**Title (ZH)**: 反向浏览器：向量图像到代码生成器 

**Authors**: Zoltan Toth-Czifra  

**Link**: [PDF](https://arxiv.org/pdf/2509.05394)  

**Abstract**: Automating the conversion of user interface design into code (image-to-code or image-to-UI) is an active area of software engineering research. However, the state-of-the-art solutions do not achieve high fidelity to the original design, as evidenced by benchmarks. In this work, I approach the problem differently: I use vector images instead of bitmaps as model input. I create several large datasets for training machine learning models. I evaluate the available array of Image Quality Assessment (IQA) algorithms and introduce a new, multi-scale metric. I then train a large open-weights model and discuss its limitations. 

**Abstract (ZH)**: 自动化用户界面设计到代码的转换（图像到代码或图像到UI）是软件工程研究的一个活跃领域。然而，现有解决方案在保留原始设计的保真度方面不尽如人意，这在基准测试中已有体现。在此工作中，我从不同的角度解决问题：使用矢量图像而不是位图作为模型输入。我创建了多个大型数据集以训练机器学习模型。我评估了现有的多种图像质量评估（IQA）算法，并引入了一个新的多尺度度量标准。随后，我训练了一个大型开放权重模型，并讨论了其局限性。 

---
# Inferring Prerequisite Knowledge Concepts in Educational Knowledge Graphs: A Multi-criteria Approach 

**Title (ZH)**: 教育知识图谱中先决知识概念的推断：一种多准则方法 

**Authors**: Rawaa Alatrash, Mohamed Amine Chatti, Nasha Wibowo, Qurat Ul Ain  

**Link**: [PDF](https://arxiv.org/pdf/2509.05393)  

**Abstract**: Educational Knowledge Graphs (EduKGs) organize various learning entities and their relationships to support structured and adaptive learning. Prerequisite relationships (PRs) are critical in EduKGs for defining the logical order in which concepts should be learned. However, the current EduKG in the MOOC platform CourseMapper lacks explicit PR links, and manually annotating them is time-consuming and inconsistent. To address this, we propose an unsupervised method for automatically inferring concept PRs without relying on labeled data. We define ten criteria based on document-based, Wikipedia hyperlink-based, graph-based, and text-based features, and combine them using a voting algorithm to robustly capture PRs in educational content. Experiments on benchmark datasets show that our approach achieves higher precision than existing methods while maintaining scalability and adaptability, thus providing reliable support for sequence-aware learning in CourseMapper. 

**Abstract (ZH)**: 教育知识图谱（EduKGs）组织各种学习实体及其关系以支持结构化和适应性学习。先决条件关系（PRs）在EduKGs中对于定义概念的学习逻辑顺序至关重要。然而，MOOC平台CourseMapper中的当前EduKG缺乏明确的PR链接，手动标注它们耗费时间且不一致。为解决这一问题，我们提出了一种无需依赖标注数据的无监督方法，以自动推断概念的PRs。我们基于文档、Wikipedia超链接、图和文本定义了十项标准，并借助投票算法将这些标准结合，以稳健地捕获教育内容中的PRs。实验表明，我们的方法在基准数据集上实现了更高的精度，同时保持了可扩展性和适应性，从而为CourseMapper中的序列感知学习提供了可靠的支撑。 

---
# An Optimized Pipeline for Automatic Educational Knowledge Graph Construction 

**Title (ZH)**: 一种自动教育知识图谱构建的优化管道 

**Authors**: Qurat Ul Ain, Mohamed Amine Chatti, Jean Qussa, Amr Shakhshir, Rawaa Alatrash, Shoeb Joarder  

**Link**: [PDF](https://arxiv.org/pdf/2509.05392)  

**Abstract**: The automatic construction of Educational Knowledge Graphs (EduKGs) is essential for domain knowledge modeling by extracting meaningful representations from learning materials. Despite growing interest, identifying a scalable and reliable approach for automatic EduKG generation remains a challenge. In an attempt to develop a unified and robust pipeline for automatic EduKG construction, in this study we propose a pipeline for automatic EduKG construction from PDF learning materials. The process begins with generating slide-level EduKGs from individual pages/slides, which are then merged to form a comprehensive EduKG representing the entire learning material. We evaluate the accuracy of the EduKG generated from the proposed pipeline in our MOOC platform, CourseMapper. The observed accuracy, while indicative of partial success, is relatively low particularly in the educational context, where the reliability of knowledge representations is critical for supporting meaningful learning. To address this, we introduce targeted optimizations across multiple pipeline components. The optimized pipeline achieves a 17.5% improvement in accuracy and a tenfold increase in processing efficiency. Our approach offers a holistic, scalable and end-to-end pipeline for automatic EduKG construction, adaptable to diverse educational contexts, and supports improved semantic representation of learning content. 

**Abstract (ZH)**: 自动构建教育知识图谱（EduKG）对于从学习材料中提取有意义的表示以进行领域知识建模至关重要。尽管兴趣 Growing，但自动 EduKG 生成的可扩展和可靠方法仍然是一个挑战。为了开发一个统一且稳健的自动 EduKG 构建管道，本研究提出了一种从 PDF 学习材料自动构建 EduKG 的管道。该过程始于从单页/幻灯片生成幻灯片级别的 EduKG，然后合并形成代表整个学习材料的全面 EduKG。我们在 MOOC 平台 CourseMapper 中评估了所提管道生成的 EduKG 的准确性。观察到的准确性虽然表明部分成功，但在教育情境中较低，知识表示的可靠性对于支持有意义的学习至关重要。为解决这一问题，我们针对管道多个组件引入了目标优化。优化后的管道在准确性和处理效率上分别提高了 17.5% 和十倍。我们的方法提供了一个全面、可扩展且端到端的自动 EduKG 构建管道，适应多种教育情境，并支持学习内容的改进语义表示。 

---
# Augmented Structure Preserving Neural Networks for cell biomechanics 

**Title (ZH)**: 增强结构保持神经网络在细胞生物力学中的应用 

**Authors**: Juan Olalla-Pombo, Alberto Badías, Miguel Ángel Sanz-Gómez, José María Benítez, Francisco Javier Montáns  

**Link**: [PDF](https://arxiv.org/pdf/2509.05388)  

**Abstract**: Cell biomechanics involve a great number of complex phenomena that are fundamental to the evolution of life itself and other associated processes, ranging from the very early stages of embryo-genesis to the maintenance of damaged structures or the growth of tumors. Given the importance of such phenomena, increasing research has been dedicated to their understanding, but the many interactions between them and their influence on the decisions of cells as a collective network or cluster remain unclear. We present a new approach that combines Structure Preserving Neural Networks, which study cell movements as a purely mechanical system, with other Machine Learning tools (Artificial Neural Networks), which allow taking into consideration environmental factors that can be directly deduced from an experiment with Computer Vision techniques. This new model, tested on simulated and real cell migration cases, predicts complete cell trajectories following a roll-out policy with a high level of accuracy. This work also includes a mitosis event prediction model based on Neural Networks architectures which makes use of the same observed features. 

**Abstract (ZH)**: 细胞生物力学涉及生命进化及其相关过程中众多复杂的现象，从胚胎发生早期阶段到损伤结构的维持或肿瘤的生长都包括在内。鉴于这些现象的重要性，越来越多的研究致力于理解它们，但它们之间的复杂交互作用及其对细胞集体网络或群体决策的影响仍然不甚清楚。我们提出了一种新的方法，将保持结构神经网络与其它机器学习工具（人工神经网络）相结合，以考虑通过计算机视觉技术直接从实验中得出的环境因素。该新模型已在模拟和实际细胞迁移案例中得到测试，能够以高精度预测完整细胞轨迹，并采用神经网络架构提出了一种基于观察特征的有丝分裂事件预测模型。 

---
# User Privacy and Large Language Models: An Analysis of Frontier Developers' Privacy Policies 

**Title (ZH)**: 用户隐私与大型语言模型：前沿开发者隐私政策分析 

**Authors**: Jennifer King, Kevin Klyman, Emily Capstick, Tiffany Saade, Victoria Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2509.05382)  

**Abstract**: Hundreds of millions of people now regularly interact with large language models via chatbots. Model developers are eager to acquire new sources of high-quality training data as they race to improve model capabilities and win market share. This paper analyzes the privacy policies of six U.S. frontier AI developers to understand how they use their users' chats to train models. Drawing primarily on the California Consumer Privacy Act, we develop a novel qualitative coding schema that we apply to each developer's relevant privacy policies to compare data collection and use practices across the six companies. We find that all six developers appear to employ their users' chat data to train and improve their models by default, and that some retain this data indefinitely. Developers may collect and train on personal information disclosed in chats, including sensitive information such as biometric and health data, as well as files uploaded by users. Four of the six companies we examined appear to include children's chat data for model training, as well as customer data from other products. On the whole, developers' privacy policies often lack essential information about their practices, highlighting the need for greater transparency and accountability. We address the implications of users' lack of consent for the use of their chat data for model training, data security issues arising from indefinite chat data retention, and training on children's chat data. We conclude by providing recommendations to policymakers and developers to address the data privacy challenges posed by LLM-powered chatbots. 

**Abstract (ZH)**: 数百万人现在经常通过聊天机器人与大型语言模型互动。模型开发者急于获取新的高品质训练数据来源，以提升模型能力并赢得市场份额。本文分析了六家美国领先AI开发者的隐私政策，以了解他们如何使用用户聊天数据来训练模型。主要基于加利福尼亚消费者隐私法，我们发展了一种新的定性编码方案，并将其应用于每家开发者的相关隐私政策，以比较六家公司的数据收集和使用实践。我们发现，所有六家开发者似乎默认使用用户的聊天数据来训练和改进他们的模型，并且有些开发者会无限期保留这些数据。开发者可能收集并用于训练在聊天中披露的个人信息，包括生物识别和健康等敏感信息，以及用户上传的文件。我们研究的六家公司中有四家似乎包括儿童聊天数据用于模型训练，以及其他产品的客户数据。总体而言，开发者们的隐私政策往往缺乏关于其实践的必要信息，突显了提高透明度和问责制的必要性。我们探讨了用户对使用其聊天数据进行模型训练缺乏同意的含义，以及无限期保留聊天数据所带来的数据安全问题，以及使用儿童聊天数据进行训练的问题。最后，我们为政策制定者和开发者提供了应对基于大语言模型的聊天机器人带来的数据隐私挑战的建议。 

---
# ThreatGPT: An Agentic AI Framework for Enhancing Public Safety through Threat Modeling 

**Title (ZH)**: 威胁GPT：一个赋能型AI框架，通过威胁建模提升公共安全 

**Authors**: Sharif Noor Zisad, Ragib Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2509.05379)  

**Abstract**: As our cities and communities become smarter, the systems that keep us safe, such as traffic control centers, emergency response networks, and public transportation, also become more complex. With this complexity comes a greater risk of security threats that can affect not just machines but real people's lives. To address this challenge, we present ThreatGPT, an agentic Artificial Intelligence (AI) assistant built to help people whether they are engineers, safety officers, or policy makers to understand and analyze threats in public safety systems. Instead of requiring deep cybersecurity expertise, it allows users to simply describe the components of a system they are concerned about, such as login systems, data storage, or communication networks. Then, with the click of a button, users can choose how they want the system to be analyzed by using popular frameworks such as STRIDE, MITRE ATT&CK, CVE reports, NIST, or CISA. ThreatGPT is unique because it does not just provide threat information, but rather it acts like a knowledgeable partner. Using few-shot learning, the AI learns from examples and generates relevant smart threat models. It can highlight what might go wrong, how attackers could take advantage, and what can be done to prevent harm. Whether securing a city's infrastructure or a local health service, this tool adapts to users' needs. In simple terms, ThreatGPT brings together AI and human judgment to make our public systems safer. It is designed not just to analyze threats, but to empower people to understand and act on them, faster, smarter, and with more confidence. 

**Abstract (ZH)**: 随着我们的城市和社区变得更加智能化，保障我们安全的系统，如交通控制中心、应急响应网络和公共交通系统，也变得愈加复杂。随之而来的复杂性带来了更多的安全威胁风险，这些威胁不仅影响机器，还可能影响人们的真实生活。为应对这一挑战，我们提出 ThreatGPT，这是一种代理型人工智能（AI）助理，旨在帮助工程师、安全官员或政策制定者理解和分析公共安全系统中的威胁。它不需要用户具备深厚的网络安全专业知识，用户只需描述他们关心的系统组件，如登录系统、数据存储或通信网络。然后，只需点击按钮，用户就可以使用STRIDE、MITRE ATT&CK、CVE报告、NIST或CISA等流行框架来选择他们希望系统如何被分析。ThreatGPT的独特之处在于，它不仅提供威胁信息，更像是一个有知识的合作伙伴。通过少量示例学习，AI学习并生成相关的智能威胁模型，指出可能出现的问题、攻击者可能利用的方法以及可以预防的措施。无论是保护城市的基础设施还是本地健康服务，该工具都能适应用户的需求。简而言之，ThreatGPT将AI和人类判断结合在一起，使我们的公共系统更加安全。它不仅用于分析威胁，还旨在赋能人们更快、更智能地理解和应对这些威胁。 

---
# Privacy Preservation and Identity Tracing Prevention in AI-Driven Eye Tracking for Interactive Learning Environments 

**Title (ZH)**: 基于人工智能驱动的眼动追踪的交互学习环境中隐私保护与身份追踪预防 

**Authors**: Abdul Rehman, Are Dæhlen, Ilona Heldal, Jerry Chun-wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05376)  

**Abstract**: Eye-tracking technology can aid in understanding neurodevelopmental disorders and tracing a person's identity. However, this technology poses a significant risk to privacy, as it captures sensitive information about individuals and increases the likelihood that data can be traced back to them. This paper proposes a human-centered framework designed to prevent identity backtracking while preserving the pedagogical benefits of AI-powered eye tracking in interactive learning environments. We explore how real-time data anonymization, ethical design principles, and regulatory compliance (such as GDPR) can be integrated to build trust and transparency. We first demonstrate the potential for backtracking student IDs and diagnoses in various scenarios using serious game-based eye-tracking data. We then provide a two-stage privacy-preserving framework that prevents participants from being tracked while still enabling diagnostic classification. The first phase covers four scenarios: I) Predicting disorder diagnoses based on different game levels. II) Predicting student IDs based on different game levels. III) Predicting student IDs based on randomized data. IV) Utilizing K-Means for out-of-sample data. In the second phase, we present a two-stage framework that preserves privacy. We also employ Federated Learning (FL) across multiple clients, incorporating a secure identity management system with dummy IDs and administrator-only access controls. In the first phase, the proposed framework achieved 99.3% accuracy for scenario 1, 63% accuracy for scenario 2, and 99.7% accuracy for scenario 3, successfully identifying and assigning a new student ID in scenario 4. In phase 2, we effectively prevented backtracking and established a secure identity management system with dummy IDs and administrator-only access controls, achieving an overall accuracy of 99.40%. 

**Abstract (ZH)**: 眼动追踪技术可以辅助理解神经发育障碍并追踪个人身份，但也带来了重大的隐私风险。本文提出了一种以人为核心的设计框架，旨在防止身份追溯的同时保留基于人工智能的眼动追踪技术在交互学习环境中的教学益处。我们探讨了如何通过实时数据匿名化、伦理设计原则及合规性（如GDPR）来构建信任和透明度。首先，我们通过基于严肃游戏的眼动追踪数据展示了在各种场景下学生身份和诊断回溯的潜在可能性。然后，我们提供了一种两阶段的隐私保护框架，防止参与者被跟踪的同时仍能实现诊断分类。第一阶段涵盖了四种情景：I）基于不同游戏级别预测障碍诊断；II）基于不同游戏级别预测学生身份；III）基于随机化数据预测学生身份；IV）利用K-Means处理外样数据。在第二阶段，我们提出了一种两阶段框架来保护隐私，并结合多方学习（FL）和一个包含虚拟身份管理和管理员专属访问控制的加密身份管理系统。第一阶段提出的框架在情景1中实现了99.3%的准确率，在情景2中实现了63%的准确率，在情景3中实现了99.7%的准确率，并成功在情景4中识别并分配了一个新的学生身份。在第二阶段，我们有效地防止了身份回溯，并建立了一个包含虚拟身份管理和管理员专属访问控制的加密身份管理系统，整体准确率为99.40%。 

---
# Prototyping an AI-powered Tool for Energy Efficiency in New Zealand Homes 

**Title (ZH)**: 基于AI的高效能住宅工具原型设计：以新西兰住宅为例 

**Authors**: Abdollah Baghaei Daemei  

**Link**: [PDF](https://arxiv.org/pdf/2509.05364)  

**Abstract**: Residential buildings contribute significantly to energy use, health outcomes, and carbon emissions. In New Zealand, housing quality has historically been poor, with inadequate insulation and inefficient heating contributing to widespread energy hardship. Recent reforms, including the Warmer Kiwi Homes program, Healthy Homes Standards, and H1 Building Code upgrades, have delivered health and comfort improvements, yet challenges persist. Many retrofits remain partial, data on household performance are limited, and decision-making support for homeowners is fragmented. This study presents the design and evaluation of an AI-powered decision-support tool for residential energy efficiency in New Zealand. The prototype, developed using Python and Streamlit, integrates data ingestion, anomaly detection, baseline modeling, and scenario simulation (e.g., LED retrofits, insulation upgrades) into a modular dashboard. Fifteen domain experts, including building scientists, consultants, and policy practitioners, tested the tool through semi-structured interviews. Results show strong usability (M = 4.3), high value of scenario outputs (M = 4.5), and positive perceptions of its potential to complement subsidy programs and regulatory frameworks. The tool demonstrates how AI can translate national policies into personalized, household-level guidance, bridging the gap between funding, standards, and practical decision-making. Its significance lies in offering a replicable framework for reducing energy hardship, improving health outcomes, and supporting climate goals. Future development should focus on carbon metrics, tariff modeling, integration with national datasets, and longitudinal trials to assess real-world adoption. 

**Abstract (ZH)**: 住宅建筑对能源使用、健康结果和碳排放产生了显著影响。在新西兰，住房质量 historical 上较差，缺乏足够的保温和低效的供暖设施，导致广泛的能源负担。最近的改革，包括“温暖新西兰家园”计划、健康家园标准和H1建筑规范升级，已经带来了健康和舒适度的改善，但仍面临挑战。许多翻新仍然只是部分完成，家庭性能数据有限，对房主的决策支持也支离破碎。本研究提出了一个基于人工智能的决策支持工具，用于新西兰住宅能源效率的设计和评估。该原型使用Python和Streamlit开发，整合了数据摄入、异常检测、基线建模和场景模拟（例如LED翻新、保温升级）等功能，形成模块化的仪表板。十五位领域专家，包括建筑科学家、咨询顾问和政策从业者，通过半结构化访谈测试了该工具。结果显示，该工具具有良好的可用性（平均分4.3）、场景输出的高度价值（平均分4.5），并且被认为有可能补充补贴计划和监管框架。该工具展示了如何将国家政策转化为个人化的家庭指导，跨越了从资金、标准到实际决策的鸿沟。其意义在于提供了一个可复制的框架，用于减少能源负担、改善健康结果和支持气候目标。未来开发应关注碳指标、电价建模、与国家数据集的整合以及纵向试验，以评估其在实际中的应用。 

---
# Governing AI R&D: A Legal Framework for Constraining Dangerous AI 

**Title (ZH)**: 治理AI研发：约束危险人工智能的法律框架 

**Authors**: Alex Mark, Aaron Scher  

**Link**: [PDF](https://arxiv.org/pdf/2509.05361)  

**Abstract**: As AI advances, governing its development may become paramount to public safety. Lawmakers may seek to restrict the development and release of AI models or of AI research itself. These governance actions could trigger legal challenges that invalidate the actions, so lawmakers should consider these challenges ahead of time. We investigate three classes of potential litigation risk for AI regulation in the U.S.: the First Amendment, administrative law, and the Fourteenth Amendment. We discuss existing precedent that is likely to apply to AI, which legal challenges are likely to arise, and how lawmakers might preemptively address them. Effective AI regulation is possible, but it requires careful implementation to avoid these legal challenges. 

**Abstract (ZH)**: 随着人工智能的发展，对其发展的治理可能对公共安全至关重要。立法者可能会寻求限制AI模型的发展和发布，或限制AI研究本身。这些治理行动可能会引发法律挑战，从而使这些行动无效，因此立法者应在采取行动前考虑这些挑战。我们研究了美国AI治理潜在诉讼风险的三类：第一修正案、行政法和第十四修正案。我们讨论了可能适用于AI的现有先例，可能出现的法律挑战以及立法者如何事先应对这些挑战。有效的AI治理是可能的，但需要谨慎实施以避免这些法律挑战。 

---
# An Empirical Analysis of Discrete Unit Representations in Speech Language Modeling Pre-training 

**Title (ZH)**: 离散单位表示在语音语言模型预训练中的实证分析 

**Authors**: Yanis Labrak, Richard Dufour, Mickaël Rouvier  

**Link**: [PDF](https://arxiv.org/pdf/2509.05359)  

**Abstract**: This paper investigates discrete unit representations in Speech Language Models (SLMs), focusing on optimizing speech modeling during continual pre-training. In this paper, we systematically examine how model architecture, data representation, and training robustness influence the pre-training stage in which we adapt existing pre-trained language models to the speech modality. Our experiments highlight the role of speech encoders and clustering granularity across different model scales, showing how optimal discretization strategies vary with model capacity. By examining cluster distribution and phonemic alignments, we investigate the effective use of discrete vocabulary, uncovering both linguistic and paralinguistic patterns. Additionally, we explore the impact of clustering data selection on model robustness, highlighting the importance of domain matching between discretization training and target applications. 

**Abstract (ZH)**: 本文研究了语音语言模型中离散单位表示，聚焦于优化连续预训练中的语音建模。本文系统地考察了模型架构、数据表示和训练鲁棒性对预训练阶段的影响，该阶段涉及将现有预训练语言模型适配至语音模态。我们的实验强调了不同模型规模下的语音编码器和聚类粒度的作用，展示了最优离散化策略随模型容量的变化。通过研究聚类分布和音素对齐，我们探讨了离散词汇的有效应用，揭示了语言和副语言模式。此外，我们还探索了聚类数据选择对模型鲁棒性的影响，强调了离散化训练与目标应用领域匹配的重要性。 

---
# Comparative Evaluation of Hard and Soft Clustering for Precise Brain Tumor Segmentation in MR Imaging 

**Title (ZH)**: 硬聚类与软聚类在MR成像精准脑肿瘤分割中的比较评价 

**Authors**: Dibya Jyoti Bora, Mrinal Kanti Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2509.05340)  

**Abstract**: Segmentation of brain tumors from Magnetic Resonance Imaging (MRI) remains a pivotal challenge in medical image analysis due to the heterogeneous nature of tumor morphology and intensity distributions. Accurate delineation of tumor boundaries is critical for clinical decision-making, radiotherapy planning, and longitudinal disease monitoring. In this study, we perform a comprehensive comparative analysis of two major clustering paradigms applied in MRI tumor segmentation: hard clustering, exemplified by the K-Means algorithm, and soft clustering, represented by Fuzzy C-Means (FCM). While K-Means assigns each pixel strictly to a single cluster, FCM introduces partial memberships, meaning each pixel can belong to multiple clusters with varying degrees of association. Experimental validation was performed using the BraTS2020 dataset, incorporating pre-processing through Gaussian filtering and Contrast Limited Adaptive Histogram Equalization (CLAHE). Evaluation metrics included the Dice Similarity Coefficient (DSC) and processing time, which collectively demonstrated that K-Means achieved superior speed with an average runtime of 0.3s per image, whereas FCM attained higher segmentation accuracy with an average DSC of 0.67 compared to 0.43 for K-Means, albeit at a higher computational cost (1.3s per image). These results highlight the inherent trade-off between computational efficiency and boundary precision. 

**Abstract (ZH)**: 磁共振成像（MRI）中脑肿瘤分割仍然是医学图像分析中的主要挑战，由于肿瘤形态和强度分布的异质性。准确划定肿瘤边界对于临床决策、放疗计划和纵向疾病监测至关重要。在本研究中，我们对应用于MRI肿瘤分割的两大主要聚类范式进行了全面比较分析：硬聚类，以K-均值算法为代表；软聚类，以模糊C均值（FCM）为代表。尽管K-均值将每个像素严格分配给单个聚类，FCM引入了部分隶属度，使得每个像素可以以不同程度归属于多个聚类。实验验证使用了BraTS2020数据集，并进行了高斯滤波和对比限制定制直方图均衡化（CLAHE）预处理。评价指标包括 Dice 相似性系数（DSC）和处理时间，结果显示，K-均值在平均运行时间为每幅图像0.3秒的情况下实现了更好的速度，而FCM在平均DSC为0.67的情况下达到了更高的分割准确度，尽管每幅图像的运算时间为1.3秒，具有更高的计算成本。这些结果突显了计算效率和边界精度之间的固有权衡。 

---
# Optical Music Recognition of Jazz Lead Sheets 

**Title (ZH)**: 爵士乐乐谱的光学音乐识别 

**Authors**: Juan Carlos Martinez-Sevilla, Francesco Foscarin, Patricia Garcia-Iasci, David Rizo, Jorge Calvo-Zaragoza, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.05329)  

**Abstract**: In this paper, we address the challenge of Optical Music Recognition (OMR) for handwritten jazz lead sheets, a widely used musical score type that encodes melody and chords. The task is challenging due to the presence of chords, a score component not handled by existing OMR systems, and the high variability and quality issues associated with handwritten images. Our contribution is two-fold. We present a novel dataset consisting of 293 handwritten jazz lead sheets of 163 unique pieces, amounting to 2021 total staves aligned with Humdrum **kern and MusicXML ground truth scores. We also supply synthetic score images generated from the ground truth. The second contribution is the development of an OMR model for jazz lead sheets. We discuss specific tokenisation choices related to our kind of data, and the advantages of using synthetic scores and pretrained models. We publicly release all code, data, and models. 

**Abstract (ZH)**: 本文针对手写爵士乐lead sheets的光学音乐识别（OMR）挑战进行了研究，这是一种广泛使用的乐谱类型，用于编码旋律和和弦。由于存在和弦这一现有OMR系统未处理的乐谱组件，以及手写图像的高变异性与质量 issues，使得任务极具挑战性。我们的贡献主要有两方面：一是提供了一个新颖的数据集，包含293份手写爵士乐lead sheets，共计163首独特的乐谱，共有2021个staff，并与Humdrum **kern和MusicXML参考乐谱对齐；二是开发了一种适用于爵士乐lead sheets的OMR模型。讨论了与我们数据类型相关的一些特定标记化选择，以及使用合成乐谱和预训练模型的优势。所有代码、数据和模型均已公开发布。 

---
# Zero-Knowledge Proofs in Sublinear Space 

**Title (ZH)**: 子线性空间中的零知识证明 

**Authors**: Logan Nye  

**Link**: [PDF](https://arxiv.org/pdf/2509.05326)  

**Abstract**: Modern zero-knowledge proof (ZKP) systems, essential for privacy and verifiable computation, suffer from a fundamental limitation: the prover typically uses memory that scales linearly with the computation's trace length T, making them impractical for resource-constrained devices and prohibitively expensive for large-scale tasks. This paper overcomes this barrier by constructing, to our knowledge, the first sublinear-space ZKP prover. Our core contribution is an equivalence that reframes proof generation as an instance of the classic Tree Evaluation problem. Leveraging a recent space-efficient tree-evaluation algorithm, we design a streaming prover that assembles the proof without ever materializing the full execution trace. The approach reduces prover memory from linear in T to O(sqrt(T)) (up to O(log T) lower-order terms) while preserving proof size, verifier time, and the transcript/security guarantees of the underlying system. This enables a shift from specialized, server-bound proving to on-device proving, opening applications in decentralized systems, on-device machine learning, and privacy-preserving technologies. 

**Abstract (ZH)**: 本篇论文克服了现代零知识证明（ZKP）系统的基本限制，构建了首个亚线性空间的ZKP证明者。 

---
# VILOD: A Visual Interactive Labeling Tool for Object Detection 

**Title (ZH)**: VILOD: 一种用于目标检测的可视交互标注工具 

**Authors**: Isac Holm  

**Link**: [PDF](https://arxiv.org/pdf/2509.05317)  

**Abstract**: The advancement of Object Detection (OD) using Deep Learning (DL) is often hindered by the significant challenge of acquiring large, accurately labeled datasets, a process that is time-consuming and expensive. While techniques like Active Learning (AL) can reduce annotation effort by intelligently querying informative samples, they often lack transparency, limit the strategic insight of human experts, and may overlook informative samples not aligned with an employed query strategy. To mitigate these issues, Human-in-the-Loop (HITL) approaches integrating human intelligence and intuition throughout the machine learning life-cycle have gained traction. Leveraging Visual Analytics (VA), effective interfaces can be created to facilitate this human-AI collaboration. This thesis explores the intersection of these fields by developing and investigating "VILOD: A Visual Interactive Labeling tool for Object Detection". VILOD utilizes components such as a t-SNE projection of image features, together with uncertainty heatmaps and model state views. Enabling users to explore data, interpret model states, AL suggestions, and implement diverse sample selection strategies within an iterative HITL workflow for OD. An empirical investigation using comparative use cases demonstrated how VILOD, through its interactive visualizations, facilitates the implementation of distinct labeling strategies by making the model's state and dataset characteristics more interpretable (RQ1). The study showed that different visually-guided labeling strategies employed within VILOD result in competitive OD performance trajectories compared to an automated uncertainty sampling AL baseline (RQ2). This work contributes a novel tool and empirical insight into making the HITL-AL workflow for OD annotation more transparent, manageable, and potentially more effective. 

**Abstract (ZH)**: 基于深度学习的对象检测进展往往受到获取大量准确标注数据集的显著挑战的阻碍，这一过程耗时且昂贵。虽然主动学习等技术可以通过智能化查询信息性样本来减少标注工作量，但这些技术往往缺乏透明性，限制了人类专家的战略洞察力，并可能忽略与所采用查询策略不一致的信息性样本。为了缓解这些问题，结合人类智能和直觉的人在回路（HITL）方法在机器学习生命周期中的应用逐渐受到关注。利用可视分析（VA），可以创建有效的界面以促进人类-AI合作。本文通过开发和研究"VILOD：一种用于对象检测的可视交互标注工具"来探讨这些领域的交叉。VILOD利用了如t-SNE图像特征投影、不确定性热图和模型状态视图等组件。它使用户能够在迭代的HITL工作流程中探索数据、解释模型状态、主动学习建议并实施多样化的样本选择策略。通过比较使用案例的实验研究结果表明，VILOD通过其交互式可视化使对象检测标注工作流程中的模型状态和数据集特征更具可解释性（RQ1）。研究结果显示，VILOD中采用的不同视觉引导的标注策略在对象检测性能轨迹上与自动不确定抽样主动学习基线具有竞争力（RQ2）。本文贡献了一种新颖的工具和实证见解，以使对象检测标注的人机在环-主动学习工作流程更具透明性、可管理性和潜在的有效性。 

---
# Towards Log Analysis with AI Agents: Cowrie Case Study 

**Title (ZH)**: 基于AI代理的日志分析：Cowrie案例研究 

**Authors**: Enis Karaarslan, Esin Güler, Efe Emir Yüce, Cagatay Coban  

**Link**: [PDF](https://arxiv.org/pdf/2509.05306)  

**Abstract**: The scarcity of real-world attack data significantly hinders progress in cybersecurity research and education. Although honeypots like Cowrie effectively collect live threat intelligence, they generate overwhelming volumes of unstructured and heterogeneous logs, rendering manual analysis impractical. As a first step in our project on secure and efficient AI automation, this study explores the use of AI agents for automated log analysis. We present a lightweight and automated approach to process Cowrie honeypot logs. Our approach leverages AI agents to intelligently parse, summarize, and extract insights from raw data, while also considering the security implications of deploying such an autonomous system. Preliminary results demonstrate the pipeline's effectiveness in reducing manual effort and identifying attack patterns, paving the way for more advanced autonomous cybersecurity analysis in future work. 

**Abstract (ZH)**: 实物攻击数据的稀缺性显著妨碍了网络安全研究和教育的进步。尽管像Cowrie这样的蜜罐能够有效收集实时威胁情报，但它们生成的大量未结构化和异构日志使得手动分析变得不切实际。作为我们项目中安全高效AI自动化的第一步，本研究探索了使用AI代理进行自动日志分析的方法。我们提出了一种轻量级的自动化方法来处理Cowrie蜜罐日志。该方法利用AI代理智能地解析、总结和从原始数据中提取洞察，同时考虑部署此类自主系统的安全影响。初步结果显示，该管道在减少手动工作量并识别攻击模式方面具有有效性，为进一步的自主网络安全分析奠定了基础。 

---
# Multi-IaC-Eval: Benchmarking Cloud Infrastructure as Code Across Multiple Formats 

**Title (ZH)**: Multi-IaC-Eval：多种格式下云基础设施即代码的基准测试 

**Authors**: Sam Davidson, Li Sun, Bhavana Bhasker, Laurent Callot, Anoop Deoras  

**Link**: [PDF](https://arxiv.org/pdf/2509.05303)  

**Abstract**: Infrastructure as Code (IaC) is fundamental to modern cloud computing, enabling teams to define and manage infrastructure through machine-readable configuration files. However, different cloud service providers utilize diverse IaC formats. The lack of a standardized format requires cloud architects to be proficient in multiple IaC languages, adding complexity to cloud deployment. While Large Language Models (LLMs) show promise in automating IaC creation and maintenance, progress has been limited by the lack of comprehensive benchmarks across multiple IaC formats. We present Multi-IaC-Bench, a novel benchmark dataset for evaluating LLM-based IaC generation and mutation across AWS CloudFormation, Terraform, and Cloud Development Kit (CDK) formats. The dataset consists of triplets containing initial IaC templates, natural language modification requests, and corresponding updated templates, created through a synthetic data generation pipeline with rigorous validation. We evaluate several state-of-the-art LLMs on Multi-IaC-Bench, demonstrating that while modern LLMs can achieve high success rates (>95%) in generating syntactically valid IaC across formats, significant challenges remain in semantic alignment and handling complex infrastructure patterns. Our ablation studies highlight the importance of prompt engineering and retry mechanisms in successful IaC generation. We release Multi-IaC-Bench to facilitate further research in AI-assisted infrastructure management and establish standardized evaluation metrics for this crucial domain. 

**Abstract (ZH)**: 基于代码的基础设施（IaC）是现代云计算的基础，使团队能够通过机器可读的配置文件定义和管理基础设施。然而，不同的云服务提供商使用多种多样的IaC格式。缺乏标准格式需要云架构师精通多种IaC语言，从而增加了云部署的复杂性。虽然大型语言模型（LLMs）在自动化IaC创建和维护方面展现出潜力，但由于缺乏跨多种IaC格式的全面基准，进展受限。我们提出了一个多IaC基准（Multi-IaC-Bench），这是一个新型基准数据集，用于评估基于LLM的IaC生成和变异，涵盖AWS CloudFormation、Terraform和Cloud Development Kit（CDK）格式。该数据集由包含初始IaC模板、自然语言修改请求以及通过严格的合成数据生成管道创建的对应更新模板的 triplet 组成。我们使用多IaC基准对几种最先进的LLM进行评估，结果显示，现代LLM可以实现高成功率（>95%）的跨格式语法有效的IaC生成，但在语义对齐和处理复杂基础设施模式方面仍面临重大挑战。我们的消融研究突显了提示工程和重试机制在成功IaC生成中的重要性。我们发布了多IaC基准，以促进进一步的AI辅助基础设施管理研究，并建立这一关键领域的标准化评估指标。 

---
# Sesame: Opening the door to protein pockets 

**Title (ZH)**: 芝麻：开启蛋白质口袋的大门 

**Authors**: Raúl Miñán, Carles Perez-Lopez, Javier Iglesias, Álvaro Ciudad, Alexis Molina  

**Link**: [PDF](https://arxiv.org/pdf/2509.05302)  

**Abstract**: Molecular docking is a cornerstone of drug discovery, relying on high-resolution ligand-bound structures to achieve accurate predictions. However, obtaining these structures is often costly and time-intensive, limiting their availability. In contrast, ligand-free structures are more accessible but suffer from reduced docking performance due to pocket geometries being less suited for ligand accommodation in apo structures. Traditional methods for artificially inducing these conformations, such as molecular dynamics simulations, are computationally expensive. In this work, we introduce Sesame, a generative model designed to predict this conformational change efficiently. By generating geometries better suited for ligand accommodation at a fraction of the computational cost, Sesame aims to provide a scalable solution for improving virtual screening workflows. 

**Abstract (ZH)**: 分子对接是药物发现的基础，依赖于高分辨率的配体结合结构以实现准确的预测。然而，获取这些结构往往成本高且耗时，限制了它们的可用性。相比之下，配体自由结构更容易获取，但由于活性位点几何结构不适合配体容纳，导致对接性能降低。传统通过分子动力学模拟等人工诱导这些构象变化的方法计算成本高昂。在这项工作中，我们引入了Sesame，一个生成模型，旨在高效预测这种构象变化。通过以较低的计算成本生成更适合配体容纳的几何结构，Sesame旨在为改进虚拟筛选工作流程提供一种可扩展的解决方案。 

---
# Nonnegative matrix factorization and the principle of the common cause 

**Title (ZH)**: 非负矩阵分解与共同原因原则 

**Authors**: E. Khalafyan, A. E. Allahverdyan, A. Hovhannisyan  

**Link**: [PDF](https://arxiv.org/pdf/2509.03652)  

**Abstract**: Nonnegative matrix factorization (NMF) is a known unsupervised data-reduction method. The principle of the common cause (PCC) is a basic methodological approach in probabilistic causality, which seeks an independent mixture model for the joint probability of two dependent random variables. It turns out that these two concepts are closely related. This relationship is explored reciprocally for several datasets of gray-scale images, which are conveniently mapped into probability models. On one hand, PCC provides a predictability tool that leads to a robust estimation of the effective rank of NMF. Unlike other estimates (e.g., those based on the Bayesian Information Criteria), our estimate of the rank is stable against weak noise. We show that NMF implemented around this rank produces features (basis images) that are also stable against noise and against seeds of local optimization, thereby effectively resolving the NMF nonidentifiability problem. On the other hand, NMF provides an interesting possibility of implementing PCC in an approximate way, where larger and positively correlated joint probabilities tend to be explained better via the independent mixture model. We work out a clustering method, where data points with the same common cause are grouped into the same cluster. We also show how NMF can be employed for data denoising. 

**Abstract (ZH)**: 非负矩阵分解（NMF）与基本原因原则（PCC）的密切关系及其应用 

---
