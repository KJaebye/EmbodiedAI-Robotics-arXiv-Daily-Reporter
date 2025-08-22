# Neural Robot Dynamics 

**Title (ZH)**: 神经机器人动力学 

**Authors**: Jie Xu, Eric Heiden, Iretiayo Akinola, Dieter Fox, Miles Macklin, Yashraj Narang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15755)  

**Abstract**: Accurate and efficient simulation of modern robots remains challenging due to their high degrees of freedom and intricate mechanisms. Neural simulators have emerged as a promising alternative to traditional analytical simulators, capable of efficiently predicting complex dynamics and adapting to real-world data; however, existing neural simulators typically require application-specific training and fail to generalize to novel tasks and/or environments, primarily due to inadequate representations of the global state. In this work, we address the problem of learning generalizable neural simulators for robots that are structured as articulated rigid bodies. We propose NeRD (Neural Robot Dynamics), learned robot-specific dynamics models for predicting future states for articulated rigid bodies under contact constraints. NeRD uniquely replaces the low-level dynamics and contact solvers in an analytical simulator and employs a robot-centric and spatially-invariant simulation state representation. We integrate the learned NeRD models as an interchangeable backend solver within a state-of-the-art robotics simulator. We conduct extensive experiments to show that the NeRD simulators are stable and accurate over a thousand simulation steps; generalize across tasks and environment configurations; enable policy learning exclusively in a neural engine; and, unlike most classical simulators, can be fine-tuned from real-world data to bridge the gap between simulation and reality. 

**Abstract (ZH)**: 现代机器人高效准确的模拟依然颇具挑战，由于其高自由度和复杂的机械结构。神经模拟器作为一种替代传统分析模拟器的有前景的选择，能够在预测复杂动力学和适应现实世界数据方面高效工作；然而，现有的神经模拟器通常需要针对特定应用进行训练，并且难以泛化到新的任务和/or环境，主要是由于对全局状态的表示不足。本工作中，我们针对由刚性连接的刚体结构组成的机器人，解决了学习可泛化的神经模拟器的问题。我们提出了NeRD（Neural Robot Dynamics），用于在存在接触约束的情况下预测刚性连接刚体未来状态的机器人特定动力学模型。NeRD独特地替代了分析模拟器中的低层级动力学和接触求解器，并采用以机器人为中心且空间不变的模拟状态表示。我们将学习到的NeRD模型集成到最新机器人模拟器的可互换后端求解器中。通过广泛的实验，我们展示了NeRD模拟器在上千个模拟步骤中稳定且准确；能够跨越任务和环境配置进行泛化；能够在神经引擎中独立试策学习；并且，与大多数经典模拟器不同，可以从现实世界数据进行微调，以弥合模拟与现实之间的差距。 

---
# Understanding and Utilizing Dynamic Coupling in Free-Floating Space Manipulators for On-Orbit Servicing 

**Title (ZH)**: 理解与利用自由浮动空间操作器中的动态耦合进行在轨服务 

**Authors**: Gargi Das, Daegyun Choi, Donghoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.15732)  

**Abstract**: This study proposes a dynamic coupling-informed trajectory optimization algorithm for free-floating space manipulator systems (SMSs). Dynamic coupling between the base and the manipulator arms plays a critical role in influencing the system's behavior. While prior research has predominantly focused on minimizing this coupling, often overlooking its potential advantages, this work investigates how dynamic coupling can instead be leveraged to improve trajectory planning. Singular value decomposition (SVD) of the dynamic coupling matrix is employed to identify the dominant components governing coupling behavior. A quantitative metric is then formulated to characterize the strength and directionality of the coupling and is incorporated into a trajectory optimization framework. To assess the feasibility of the optimized trajectory, a sliding mode control-based tracking controller is designed to generate the required joint torque inputs. Simulation results demonstrate that explicitly accounting for dynamic coupling in trajectory planning enables more informed and potentially more efficient operation, offering new directions for the control of free-floating SMSs. 

**Abstract (ZH)**: 本文提出了一种动态耦合指导下的自由浮动空间 manipulator 系统轨迹优化算法。基座与 manipulator 臂之间的动态耦合对系统行为起着关键作用。尽管以往研究主要侧重于减小这种耦合，而忽视了其潜在优势，本研究探讨了如何利用动态耦合来改进轨迹规划。通过动态耦合矩阵的奇异值分解 (SVD) 来识别主导耦合行为的组件。然后，构建一个量化指标来表征耦合的强度和方向性，并将其纳入轨迹优化框架。为了评估优化轨迹的可行性，设计了一种滑模控制跟踪控制器来生成所需的关节扭矩输入。仿真结果表明，在轨迹规划中显式考虑动态耦合能够实现更加明智且可能更高效的运行，为自由浮动空间 manipulator 系统的控制提供了新的方向。 

---
# Exploiting Policy Idling for Dexterous Manipulation 

**Title (ZH)**: 利用策略空闲时间进行灵巧操作 

**Authors**: Annie S. Chen, Philemon Brakel, Antonia Bronars, Annie Xie, Sandy Huang, Oliver Groth, Maria Bauza, Markus Wulfmeier, Nicolas Heess, Dushyant Rao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15669)  

**Abstract**: Learning-based methods for dexterous manipulation have made notable progress in recent years. However, learned policies often still lack reliability and exhibit limited robustness to important factors of variation. One failure pattern that can be observed across many settings is that policies idle, i.e. they cease to move beyond a small region of states when they reach certain states. This policy idling is often a reflection of the training data. For instance, it can occur when the data contains small actions in areas where the robot needs to perform high-precision motions, e.g., when preparing to grasp an object or object insertion. Prior works have tried to mitigate this phenomenon e.g. by filtering the training data or modifying the control frequency. However, these approaches can negatively impact policy performance in other ways. As an alternative, we investigate how to leverage the detectability of idling behavior to inform exploration and policy improvement. Our approach, Pause-Induced Perturbations (PIP), applies perturbations at detected idling states, thus helping it to escape problematic basins of attraction. On a range of challenging simulated dual-arm tasks, we find that this simple approach can already noticeably improve test-time performance, with no additional supervision or training. Furthermore, since the robot tends to idle at critical points in a movement, we also find that learning from the resulting episodes leads to better iterative policy improvement compared to prior approaches. Our perturbation strategy also leads to a 15-35% improvement in absolute success rate on a real-world insertion task that requires complex multi-finger manipulation. 

**Abstract (ZH)**: 基于学习的灵巧操作方法在近年来取得了显著进展。然而，学习得到的策略在可靠性和对重要变化因素的鲁棒性方面仍然存在不足。一种常见的失败模式是策略在达到某些状态时会停滞，即它们会停止在状态空间的一个小区域内移动。这种策略停滞往往是训练数据的反映。例如，当数据中包含机器人需要进行高精度动作的区域中的小动作时，可能会发生这种情况，比如准备抓取物体或插入物体。先前的工作尝试通过过滤训练数据或修改控制频率来减轻这种现象的负面影响，但这些方法可能以其他方式负面影响策略的表现。作为替代方案，我们研究如何利用检测到的停滞行为可检测性来指导探索和策略改进。我们的方法，暂停引发扰动（PIP），在检测到的停滞状态下应用扰动，从而帮助其逃离不利的吸引子。在一系列具有挑战性的模拟双臂任务中，我们发现这种简单方法可以显著提高测试时的表现，无需额外的监督或训练。此外，由于机器人倾向于在动作的关键点处停滞，我们还发现从生成的 episode 中学习导致了与先前方法相比更好的迭代策略改进。我们提出的扰动策略在一项需要复杂多指操作的现实插入任务中绝对成功率提高了15-35%。 

---
# Mind and Motion Aligned: A Joint Evaluation IsaacSim Benchmark for Task Planning and Low-Level Policies in Mobile Manipulation 

**Title (ZH)**: 思维与动作统一：移动 manipulator 任务规划与低级策略联合评估的 IsaacSim 基准 

**Authors**: Nikita Kachaev, Andrei Spiridonov, Andrey Gorodetsky, Kirill Muravyev, Nikita Oskolkov, Aditya Narendra, Vlad Shakhuro, Dmitry Makarov, Aleksandr I. Panov, Polina Fedotova, Alexey K. Kovalev  

**Link**: [PDF](https://arxiv.org/pdf/2508.15663)  

**Abstract**: Benchmarks are crucial for evaluating progress in robotics and embodied AI. However, a significant gap exists between benchmarks designed for high-level language instruction following, which often assume perfect low-level execution, and those for low-level robot control, which rely on simple, one-step commands. This disconnect prevents a comprehensive evaluation of integrated systems where both task planning and physical execution are critical. To address this, we propose Kitchen-R, a novel benchmark that unifies the evaluation of task planning and low-level control within a simulated kitchen environment. Built as a digital twin using the Isaac Sim simulator and featuring more than 500 complex language instructions, Kitchen-R supports a mobile manipulator robot. We provide baseline methods for our benchmark, including a task-planning strategy based on a vision-language model and a low-level control policy based on diffusion policy. We also provide a trajectory collection system. Our benchmark offers a flexible framework for three evaluation modes: independent assessment of the planning module, independent assessment of the control policy, and, crucially, an integrated evaluation of the whole system. Kitchen-R bridges a key gap in embodied AI research, enabling more holistic and realistic benchmarking of language-guided robotic agents. 

**Abstract (ZH)**: Kitchen-R：一种统一任务规划与低级控制的厨房环境基准 

---
# LLM-Driven Self-Refinement for Embodied Drone Task Planning 

**Title (ZH)**: 基于LLM的自主精炼在身临其境的无人机任务规划中 

**Authors**: Deyu Zhang, Xicheng Zhang, Jiahao Li, Tingting Long, Xunhua Dai, Yongjian Fu, Jinrui Zhang, Ju Ren, Yaoxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15501)  

**Abstract**: We introduce SRDrone, a novel system designed for self-refinement task planning in industrial-grade embodied drones. SRDrone incorporates two key technical contributions: First, it employs a continuous state evaluation methodology to robustly and accurately determine task outcomes and provide explanatory feedback. This approach supersedes conventional reliance on single-frame final-state assessment for continuous, dynamic drone operations. Second, SRDrone implements a hierarchical Behavior Tree (BT) modification model. This model integrates multi-level BT plan analysis with a constrained strategy space to enable structured reflective learning from experience. Experimental results demonstrate that SRDrone achieves a 44.87% improvement in Success Rate (SR) over baseline methods. Furthermore, real-world deployment utilizing an experience base optimized through iterative self-refinement attains a 96.25% SR. By embedding adaptive task refinement capabilities within an industrial-grade BT planning framework, SRDrone effectively integrates the general reasoning intelligence of Large Language Models (LLMs) with the stringent physical execution constraints inherent to embodied drones. Code is available at this https URL. 

**Abstract (ZH)**: 我们介绍SRDrone，一种用于工业级实体无人机自完善任务规划的新型系统。SRDrone包含两项关键技术贡献：首先，它采用连续状态评估方法，以稳健且准确地确定任务结果并提供解释性反馈。这种方法取代了传统的一帧终态评估方法，适用于连续动态无人机操作。其次，SRDrone实现了一种分层行为树（BT）修改模型。该模型结合多级BT计划分析和约束策略空间，以实现结构化的经验反思学习。实验证明，SRDrone在成功率（SR）上相比基线方法提高了44.87%。此外，通过迭代自完善优化的经验基部署可实现96.25%的SR。通过在工业级BT规划框架中嵌入自适应任务完善能力，SRDrone有效地将大型语言模型（LLMs）的通用推理智能与实体无人机固有的严格物理执行约束相结合。代码可在以下链接获取。 

---
# Lang2Lift: A Framework for Language-Guided Pallet Detection and Pose Estimation Integrated in Autonomous Outdoor Forklift Operation 

**Title (ZH)**: Lang2Lift: 一种语言引导的托盘检测与姿态估计算法框架集成在自主室外仓储叉车操作中 

**Authors**: Huy Hoang Nguyen, Johannes Huemer, Markus Murschitz, Tobias Glueck, Minh Nhat Vu, Andreas Kugi  

**Link**: [PDF](https://arxiv.org/pdf/2508.15427)  

**Abstract**: The logistics and construction industries face persistent challenges in automating pallet handling, especially in outdoor environments with variable payloads, inconsistencies in pallet quality and dimensions, and unstructured surroundings. In this paper, we tackle automation of a critical step in pallet transport: the pallet pick-up operation. Our work is motivated by labor shortages, safety concerns, and inefficiencies in manually locating and retrieving pallets under such conditions. We present Lang2Lift, a framework that leverages foundation models for natural language-guided pallet detection and 6D pose estimation, enabling operators to specify targets through intuitive commands such as "pick up the steel beam pallet near the crane." The perception pipeline integrates Florence-2 and SAM-2 for language-grounded segmentation with FoundationPose for robust pose estimation in cluttered, multi-pallet outdoor scenes under variable lighting. The resulting poses feed into a motion planning module for fully autonomous forklift operation. We validate Lang2Lift on the ADAPT autonomous forklift platform, achieving 0.76 mIoU pallet segmentation accuracy on a real-world test dataset. Timing and error analysis demonstrate the system's robustness and confirm its feasibility for deployment in operational logistics and construction environments. Video demonstrations are available at this https URL 

**Abstract (ZH)**: 物流和建筑业在自动托盘处理方面面临持续挑战，尤其是在具有变化负荷、托盘质量和尺寸不一致以及无结构环境的户外环境中。本文探讨了托盘运输中的关键步骤——托盘拾取操作的自动化。我们的研究动机源于劳动力短缺、安全问题以及在这种条件下手动定位和获取托盘的效率低下。我们提出了一种名为Lang2Lift的框架，该框架利用基础模型实现基于自然语言的托盘检测和6D姿态估计，使得操作员可以通过直观的指令（如“使用起重机附近的钢梁托盘”）指定目标。感知管道结合使用Florence-2和SAM-2进行语言指导的分割，并利用FoundationPose实现复杂多托盘户外场景中具有鲁棒性的姿态估计。估计的姿态输入到运动规划模块，实现全自动叉车操作。我们在ADAPT全自动叉车平台上验证了Lang2Lift，实测数据集的托盘分割准确率达到0.76 mIoU。时间分析和误差分析表明该系统的鲁棒性和在运营物流和建筑环境中的部署可行性。更多信息请参阅此链接。 

---
# Sensing, Social, and Motion Intelligence in Embodied Navigation: A Comprehensive Survey 

**Title (ZH)**: 具身导航中的传感、社会与运动智能综述 

**Authors**: Chaoran Xiong, Yulong Huang, Fangwen Yu, Changhao Chen, Yue Wang, Songpengchen Xia, Ling Pei  

**Link**: [PDF](https://arxiv.org/pdf/2508.15354)  

**Abstract**: Embodied navigation (EN) advances traditional navigation by enabling robots to perform complex egocentric tasks through sensing, social, and motion intelligence. In contrast to classic methodologies that rely on explicit localization and pre-defined maps, EN leverages egocentric perception and human-like interaction strategies. This survey introduces a comprehensive EN formulation structured into five stages: Transition, Observation, Fusion, Reward-policy construction, and Action (TOFRA). The TOFRA framework serves to synthesize the current state of the art, provide a critical review of relevant platforms and evaluation metrics, and identify critical open research challenges. A list of studies is available at this https URL. 

**Abstract (ZH)**: 嵌入式导航（EN）通过感知、社会交互和运动智能使机器人能够执行复杂的第一人称任务，从而推动了传统导航的进步。与依赖显式定位和预制地图的经典方法不同，EN利用第一人称感知和类人的交互策略。本文综述将嵌入式导航结构化为五个阶段：转换、观察、融合、奖励策略构建和行动（TOFRA）。TOFRA框架旨在综合当前的研究成果，提供相关平台和评价指标的批判性评审，并识别关键的开放研究挑战。相关研究列表可在以下链接获取：this https URL。 

---
# Mag-Match: Magnetic Vector Field Features for Map Matching and Registration 

**Title (ZH)**: 磁匹配：磁场矢量场特征在地图匹配和配准中的应用 

**Authors**: William McDonald, Cedric Le Gentil, Jennifer Wakulicz, Teresa Vidal-Calleja  

**Link**: [PDF](https://arxiv.org/pdf/2508.15300)  

**Abstract**: Map matching and registration are essential tasks in robotics for localisation and integration of multi-session or multi-robot data. Traditional methods rely on cameras or LiDARs to capture visual or geometric information but struggle in challenging conditions like smoke or dust. Magnetometers, on the other hand, detect magnetic fields, revealing features invisible to other sensors and remaining robust in such environments. In this paper, we introduce Mag-Match, a novel method for extracting and describing features in 3D magnetic vector field maps to register different maps of the same area. Our feature descriptor, based on higher-order derivatives of magnetic field maps, is invariant to global orientation, eliminating the need for gravity-aligned mapping. To obtain these higher-order derivatives map-wide given point-wise magnetometer data, we leverage a physics-informed Gaussian Process to perform efficient and recursive probabilistic inference of both the magnetic field and its derivatives. We evaluate Mag-Match in simulated and real-world experiments against a SIFT-based approach, demonstrating accurate map-to-map, robot-to-map, and robot-to-robot transformations - even without initial gravitational alignment. 

**Abstract (ZH)**: 磁匹配和注册是机器人学中用于局部化和多会话或多机器人数据整合的重要任务。传统方法依赖于摄像头或激光雷达捕获视觉或几何信息，但在烟雾或灰尘等挑战性条件下表现不佳。相比之下，磁强计检测磁场，揭示其他传感器看不见的特征，并且能够在恶劣环境中保持稳健性。本文介绍了Mag-Match，一种通过提取和描述3D磁场矢量场图中的特征来注册相同区域不同地图的新型方法。我们的特征描述符基于磁场图的高阶导数，具有全局方向的不变性，消除了重力对齐映射的需要。为了从局部磁强计数据获得整个区域的高阶导数图，我们利用物理启发的高斯过程进行高效且递归的概率推断，以获取磁场及其导数。在模拟和实际实验中，Mag-Match与基于SIFT的方法进行比较，展示了准确的地图间、机器人间以及机器人到地图的变换，即使没有初始重力对齐。 

---
# Survey of Vision-Language-Action Models for Embodied Manipulation 

**Title (ZH)**: 视觉-语言-动作模型综述：面向嵌体操作的任务 

**Authors**: Haoran Li, Yuhui Chen, Wenbo Cui, Weiheng Liu, Kai Liu, Mingcai Zhou, Zhengtao Zhang, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15201)  

**Abstract**: Embodied intelligence systems, which enhance agent capabilities through continuous environment interactions, have garnered significant attention from both academia and industry. Vision-Language-Action models, inspired by advancements in large foundation models, serve as universal robotic control frameworks that substantially improve agent-environment interaction capabilities in embodied intelligence systems. This expansion has broadened application scenarios for embodied AI robots. This survey comprehensively reviews VLA models for embodied manipulation. Firstly, it chronicles the developmental trajectory of VLA architectures. Subsequently, we conduct a detailed analysis of current research across 5 critical dimensions: VLA model structures, training datasets, pre-training methods, post-training methods, and model evaluation. Finally, we synthesize key challenges in VLA development and real-world deployment, while outlining promising future research directions. 

**Abstract (ZH)**: 具身智能系统通过连续环境交互增强代理能力，已引起学术界和工业界的广泛关注。受大规模基础模型进展的启发，视觉-语言-动作模型作为通用的机器人控制框架，在具身智能系统中显著提升了代理-环境交互能力，拓展了具身AI机器人应用 scenarios。本文全面回顾了视觉-语言-动作模型在具身操控中的应用。首先，梳理了视觉-语言-动作架构的发展轨迹。随后，我们在五个关键维度上详细分析了当前研究：视觉-语言-动作模型结构、训练数据集、预训练方法、后训练方法以及模型评估。最后，总结了视觉-语言-动作模型开发和实际部署中的关键挑战，并指出了有前景的未来研究方向。 

---
# Hardware Implementation of a Zero-Prior-Knowledge Approach to Lifelong Learning in Kinematic Control of Tendon-Driven Quadrupeds 

**Title (ZH)**: 基于腱驱四足机器人运动控制的零先验知识 lifelong 学习硬件实现 

**Authors**: Hesam Azadjou, Suraj Chakravarthi Raja, Ali Marjaninejad, Francisco J. Valero-Cuevas  

**Link**: [PDF](https://arxiv.org/pdf/2508.15160)  

**Abstract**: Like mammals, robots must rapidly learn to control their bodies and interact with their environment despite incomplete knowledge of their body structure and surroundings. They must also adapt to continuous changes in both. This work presents a bio-inspired learning algorithm, General-to-Particular (G2P), applied to a tendon-driven quadruped robotic system developed and fabricated in-house. Our quadruped robot undergoes an initial five-minute phase of generalized motor babbling, followed by 15 refinement trials (each lasting 20 seconds) to achieve specific cyclical movements. This process mirrors the exploration-exploitation paradigm observed in mammals. With each refinement, the robot progressively improves upon its initial "good enough" solution. Our results serve as a proof-of-concept, demonstrating the hardware-in-the-loop system's ability to learn the control of a tendon-driven quadruped with redundancies in just a few minutes to achieve functional and adaptive cyclical non-convex movements. By advancing autonomous control in robotic locomotion, our approach paves the way for robots capable of dynamically adjusting to new environments, ensuring sustained adaptability and performance. 

**Abstract (ZH)**: 类哺乳动物的机器人必须在不完全了解自身结构和环境的情况下，迅速学会控制身体并与其环境交互，同时还需要适应两者持续的变化。本工作提出了一种受生物启发的学习算法——从一般到具体（General-to-Particular, G2P），应用于自主研发的肌腱驱动四足机器人系统。我们的四足机器人首先经历一个初始的五分钟通用运动babbling阶段，随后进行15次细化试验（每次20秒），以实现特定的周期性运动。这一过程反映了哺乳动物观察到的探索-利用范式。每次细化后，机器人都会逐步完善其初始的“足够好”的解决方案。我们的结果提供了概念验证，展示了闭环硬件系统仅在几分钟内就能学习控制具有冗余性的肌腱驱动四足机器人，并实现功能性、适应性的非凸周期性运动。通过在机器人行走自主控制方面的进展，本方法为能够动态适应新环境的机器人铺平了道路，确保持续的适应性和性能。 

---
# Decentralized Vision-Based Autonomous Aerial Wildlife Monitoring 

**Title (ZH)**: 基于视觉的去中心化自主航空野生动物监测 

**Authors**: Makram Chahine, William Yang, Alaa Maalouf, Justin Siriska, Ninad Jadhav, Daniel Vogt, Stephanie Gil, Robert Wood, Daniela Rus  

**Link**: [PDF](https://arxiv.org/pdf/2508.15038)  

**Abstract**: Wildlife field operations demand efficient parallel deployment methods to identify and interact with specific individuals, enabling simultaneous collective behavioral analysis, and health and safety interventions. Previous robotics solutions approach the problem from the herd perspective, or are manually operated and limited in scale. We propose a decentralized vision-based multi-quadrotor system for wildlife monitoring that is scalable, low-bandwidth, and sensor-minimal (single onboard RGB camera). Our approach enables robust identification and tracking of large species in their natural habitat. We develop novel vision-based coordination and tracking algorithms designed for dynamic, unstructured environments without reliance on centralized communication or control. We validate our system through real-world experiments, demonstrating reliable deployment in diverse field conditions. 

**Abstract (ZH)**: 野生动物野外作业需要高效的并行部署方法，以识别和互动特定个体，实现同时进行群体行为分析及健康安全干预。之前的机器人解决方案多从群体角度出发，或者手动操作且规模有限。我们提出了一种去中心化的基于视觉的多旋翼无人机系统，用于野生动物监测，该系统具有可扩展性、低带宽和传感器最少（单个机载RGB摄像机）。我们的方法能够在自然栖息地中稳健地识别和跟踪大型物种。我们开发了一种新型基于视觉的协调与跟踪算法，适用于动态且结构不规则的环境，无需依赖集中式通信或控制。我们通过实地实验验证了该系统，在多样化的野外条件下表现出可靠的部署能力。 

---
# In-Context Iterative Policy Improvement for Dynamic Manipulation 

**Title (ZH)**: 基于上下文的迭代策略改进方法用于动态操作 

**Authors**: Mark Van der Merwe, Devesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2508.15021)  

**Abstract**: Attention-based architectures trained on internet-scale language data have demonstrated state of the art reasoning ability for various language-based tasks, such as logic problems and textual reasoning. Additionally, these Large Language Models (LLMs) have exhibited the ability to perform few-shot prediction via in-context learning, in which input-output examples provided in the prompt are generalized to new inputs. This ability furthermore extends beyond standard language tasks, enabling few-shot learning for general patterns. In this work, we consider the application of in-context learning with pre-trained language models for dynamic manipulation. Dynamic manipulation introduces several crucial challenges, including increased dimensionality, complex dynamics, and partial observability. To address this, we take an iterative approach, and formulate our in-context learning problem to predict adjustments to a parametric policy based on previous interactions. We show across several tasks in simulation and on a physical robot that utilizing in-context learning outperforms alternative methods in the low data regime. Video summary of this work and experiments can be found this https URL. 

**Abstract (ZH)**: 基于互联网规模语言数据训练的注意力机制架构展现了各种语言任务中优异的推理能力，如逻辑问题和文本推理。此外，这些大型语言模型（LLMs）还展示了通过上下文学习进行少样本预测的能力，在这种学习方式中，提示中的输入-输出示例可以泛化到新的输入。这一能力进一步超越了标准的语言任务，使少样本学习适用于更广泛的模式。在这项工作中，我们考虑使用预训练语言模型进行动态操作的应用。动态操作引入了若干关键挑战，包括维度增加、复杂动力学和部分可观测性。为应对这些挑战，我们采取迭代方法，并将上下文学习问题形式化为基于先前交互预测参数化策略调整的问题。我们在模拟和物理机器人上进行的多项任务中展示了利用上下文学习在数据稀缺条件下优于其他方法的结果。有关此项工作和实验的视频总结，请访问以下链接：this https URL。 

---
# GraspQP: Differentiable Optimization of Force Closure for Diverse and Robust Dexterous Grasping 

**Title (ZH)**: GraspQP: 力闭合的可微优化方法实现多样且稳健的灵巧抓取 

**Authors**: René Zurbrügg, Andrei Cramariuc, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2508.15002)  

**Abstract**: Dexterous robotic hands enable versatile interactions due to the flexibility and adaptability of multi-fingered designs, allowing for a wide range of task-specific grasp configurations in diverse environments. However, to fully exploit the capabilities of dexterous hands, access to diverse and high-quality grasp data is essential -- whether for developing grasp prediction models from point clouds, training manipulation policies, or supporting high-level task planning with broader action options. Existing approaches for dataset generation typically rely on sampling-based algorithms or simplified force-closure analysis, which tend to converge to power grasps and often exhibit limited diversity. In this work, we propose a method to synthesize large-scale, diverse, and physically feasible grasps that extend beyond simple power grasps to include refined manipulations, such as pinches and tri-finger precision grasps. We introduce a rigorous, differentiable energy formulation of force closure, implicitly defined through a Quadratic Program (QP). Additionally, we present an adjusted optimization method (MALA*) that improves performance by dynamically rejecting gradient steps based on the distribution of energy values across all samples. We extensively evaluate our approach and demonstrate significant improvements in both grasp diversity and the stability of final grasp predictions. Finally, we provide a new, large-scale grasp dataset for 5,700 objects from DexGraspNet, comprising five different grippers and three distinct grasp types.
Dataset and Code:this https URL 

**Abstract (ZH)**: 灵巧机械手的手指灵活性和多指设计的适应性使其能够进行多样的交互，从而在各种环境中实现多种任务特定的抓持配置。然而，为了充分利用灵巧手的 capabilities，获取多样且高质量的抓持数据是必不可少的——无论是开发从点云预测抓持模型，训练操作策略，还是为高级任务规划提供更多操作选项。现有的数据集生成方法通常依赖于基于采样的算法或简化的力量闭合分析，这些方法往往会收敛于功率抓持，并表现出有限的多样性。在这项工作中，我们提出了一种合成大规模、多样且物理上可行的抓持的数据方法，这种方法不仅包括简单的功率抓持，还扩展到包括精细操作，如捏握和三指精度抓持。我们引入了一种严格的、可通过二次规划（QP）隐式定义的能量形式的力量闭合差分方法。此外，我们提出了一种调整的优化方法（MALA*），该方法通过基于所有样本的能量值分布动态拒绝梯度步骤来提高性能。我们广泛评估了我们的方法，并展示了在抓持多样性和最终抓持预测稳定性方面的显著改进。最后，我们提供了来自DexGraspNet的5,700个物体的新大规模抓持数据集，包含五种不同的夹爪和三种不同的抓取类型。 Dataset and Code:https://... 

---
# A Vision-Based Shared-Control Teleoperation Scheme for Controlling the Robotic Arm of a Four-Legged Robot 

**Title (ZH)**: 基于视觉的四足机器人臂共享控制远程操作方案 

**Authors**: Murilo Vinicius da Silva, Matheus Hipolito Carvalho, Juliano Negri, Thiago Segreto, Gustavo J. G. Lahr, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2508.14994)  

**Abstract**: In hazardous and remote environments, robotic systems perform critical tasks demanding improved safety and efficiency. Among these, quadruped robots with manipulator arms offer mobility and versatility for complex operations. However, teleoperating quadruped robots is challenging due to the lack of integrated obstacle detection and intuitive control methods for the robotic arm, increasing collision risks in confined or dynamically changing workspaces. Teleoperation via joysticks or pads can be non-intuitive and demands a high level of expertise due to its complexity, culminating in a high cognitive load on the operator. To address this challenge, a teleoperation approach that directly maps human arm movements to the robotic manipulator offers a simpler and more accessible solution. This work proposes an intuitive remote control by leveraging a vision-based pose estimation pipeline that utilizes an external camera with a machine learning-based model to detect the operator's wrist position. The system maps these wrist movements into robotic arm commands to control the robot's arm in real-time. A trajectory planner ensures safe teleoperation by detecting and preventing collisions with both obstacles and the robotic arm itself. The system was validated on the real robot, demonstrating robust performance in real-time control. This teleoperation approach provides a cost-effective solution for industrial applications where safety, precision, and ease of use are paramount, ensuring reliable and intuitive robotic control in high-risk environments. 

**Abstract (ZH)**: 在恶劣和偏远环境中，机器人系统执行需要改进安全性和效率的关键任务。其中，具有 manipulator 臂的四足机器人提供复杂操作所需的机动性和多功能性。然而，远程操作四足机器人由于缺乏集成的障碍检测和直观的控制方法而具有挑战性，增加了在受限或动态变化的工作空间中发生碰撞的风险。使用操纵杆或按键进行远程操控可能缺乏直观性，并且由于其复杂性要求高度的专业技能，从而给操作者带来较高的认知负荷。为解决这一挑战，一种直接将人类手臂运动映射到机器人 manipulator 的远程操控方法提供了一种更简单、更易操作的解决方案。本研究提出了一种直观的远程控制方法，利用基于视觉的姿态估计流水线，借助外部摄像头和基于机器学习的模型来检测操作者的腕部位置。系统将这些手腕运动映射为机器人手臂命令，以实现实时控制。轨迹规划器通过检测并与障碍物及机器人手臂本身发生碰撞来进行安全远程操控。该系统已在真实机器人上进行了验证，显示了在实时控制中表现出的稳健性能。这种远程操控方法为工业应用提供了一种经济有效的解决方案，其中安全、精度和易用性至关重要，确保在高风险环境中实现可靠且直观的机器人控制。 

---
# Open-Universe Assistance Games 

**Title (ZH)**: 开放宇宙辅助博弈 

**Authors**: Rachel Ma, Jingyi Qu, Andreea Bobu, Dylan Hadfield-Menell  

**Link**: [PDF](https://arxiv.org/pdf/2508.15119)  

**Abstract**: Embodied AI agents must infer and act in an interpretable way on diverse human goals and preferences that are not predefined. To formalize this setting, we introduce Open-Universe Assistance Games (OU-AGs), a framework where the agent must reason over an unbounded and evolving space of possible goals. In this context, we introduce GOOD (GOals from Open-ended Dialogue), a data-efficient, online method that extracts goals in the form of natural language during an interaction with a human, and infers a distribution over natural language goals. GOOD prompts an LLM to simulate users with different complex intents, using its responses to perform probabilistic inference over candidate goals. This approach enables rich goal representations and uncertainty estimation without requiring large offline datasets. We evaluate GOOD in a text-based grocery shopping domain and in a text-operated simulated household robotics environment (AI2Thor), using synthetic user profiles. Our method outperforms a baseline without explicit goal tracking, as confirmed by both LLM-based and human evaluations. 

**Abstract (ZH)**: 开放宇宙辅助游戏：基于开放对话的目标抽取方法 

---
# Discrete VHCs for Propeller Motion of a Devil-Stick using purely Impulsive Inputs 

**Title (ZH)**: 离散化VHCs在devil-stick桨动中的纯冲量输入研究 

**Authors**: Aakash Khandelwal, Ranjan Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2508.15040)  

**Abstract**: The control problem of realizing propeller motion of a devil-stick in the vertical plane using impulsive forces applied normal to the stick is considered. This problem is an example of underactuated robotic juggling and has not been considered in the literature before. Inspired by virtual holonomic constraints, the concept of discrete virtual holonomic constraints (DVHC) is introduced for the first time to solve this orbital stabilization problem. At the discrete instants when impulsive inputs are applied, the location of the center-of-mass of the devil-stick is specified in terms of its orientation angle. This yields the discrete zero dynamics (DZD), which provides conditions for stable propeller motion. In the limiting case, when the rotation angle between successive applications of impulsive inputs is chosen to be arbitrarily small, the problem reduces to that of propeller motion under continuous forcing. A controller that enforces the DVHC, and an orbit stabilizing controller based on the impulse controlled Poincaré map approach are presented. The efficacy of the approach to trajectory design and stabilization is validated through simulations. 

**Abstract (ZH)**: 使用作用于魔棍上的脉冲力在垂直平面内实现魔棍旋翼运动的控制问题：基于离散虚拟约束的概念解决轨道稳定问题 

---
# You Only Pose Once: A Minimalist's Detection Transformer for Monocular RGB Category-level 9D Multi-Object Pose Estimation 

**Title (ZH)**: 一次姿态检测： minimalist 的单目 RGB 多物体 9D 类别级姿态估计检测变压器 

**Authors**: Hakjin Lee, Junghoon Seo, Jaehoon Sim  

**Link**: [PDF](https://arxiv.org/pdf/2508.14965)  

**Abstract**: Accurately recovering the full 9-DoF pose of unseen instances within specific categories from a single RGB image remains a core challenge for robotics and automation. Most existing solutions still rely on pseudo-depth, CAD models, or multi-stage cascades that separate 2D detection from pose estimation. Motivated by the need for a simpler, RGB-only alternative that learns directly at the category level, we revisit a longstanding question: Can object detection and 9-DoF pose estimation be unified with high performance, without any additional data? We show that they can with our method, YOPO, a single-stage, query-based framework that treats category-level 9-DoF estimation as a natural extension of 2D detection. YOPO augments a transformer detector with a lightweight pose head, a bounding-box-conditioned translation module, and a 6D-aware Hungarian matching cost. The model is trained end-to-end only with RGB images and category-level pose labels. Despite its minimalist design, YOPO sets a new state of the art on three benchmarks. On the REAL275 dataset, it achieves 79.6% $\rm{IoU}_{50}$ and 54.1% under the $10^\circ$$10{\rm{cm}}$ metric, surpassing prior RGB-only methods and closing much of the gap to RGB-D systems. The code, models, and additional qualitative results can be found on our project. 

**Abstract (ZH)**: 从单张RGB图像中准确恢复未见过的特定类别实例的全9-自由度姿态仍然是机器人技术和自动化领域的核心挑战。现有的大多数解决方案仍然依赖于伪深度、CAD模型或多阶段级联方法，将2D检测与姿态估计分离。受需要一种更简单、仅基于RGB且可在类别级别进行学习的替代方案的启发，我们重新审视了一个长期存在的问题：是否可以在没有任何额外数据的情况下，同时实现高性能的对象检测和9-自由度姿态估计？我们证明了可以通过我们的方法YOPO实现这一目标，这是一种单阶段、基于查询的框架，将类别级别的9-自由度估计视为2D检测的自然扩展。YOPO通过一个轻量级姿态头、一个基于边界框的平移模块和一个6D感知匈牙利匹配成本来增强变压器检测器。该模型仅使用RGB图像和类别级别的姿态标签进行端到端训练。尽管设计简洁，YOPO在三个基准测试上取得了新的最佳性能。在REAL275数据集上，它实现了79.6%的$\rm{IoU}_{50}$和54.1%的$10^\circ10\rm{cm}$指标，超越了之前仅基于RGB的方法，并显著缩小了与RGB-D系统之间的差距。更多代码、模型和额外的定性结果可在我们的项目中找到。 

---
# Learning to Drive Ethically: Embedding Moral Reasoning into Autonomous Driving 

**Title (ZH)**: 学习如何道德驾驶：将道德推理嵌入自动驾驶 

**Authors**: Dianzhao Li, Ostap Okhrin  

**Link**: [PDF](https://arxiv.org/pdf/2508.14926)  

**Abstract**: Autonomous vehicles hold great promise for reducing traffic fatalities and improving transportation efficiency, yet their widespread adoption hinges on embedding robust ethical reasoning into routine and emergency maneuvers. Here, we present a hierarchical Safe Reinforcement Learning (Safe RL) framework that explicitly integrates moral considerations with standard driving objectives. At the decision level, a Safe RL agent is trained using a composite ethical risk cost, combining collision probability and harm severity, to generate high-level motion targets. A dynamic Prioritized Experience Replay mechanism amplifies learning from rare but critical, high-risk events. At the execution level, polynomial path planning coupled with Proportional-Integral-Derivative (PID) and Stanley controllers translates these targets into smooth, feasible trajectories, ensuring both accuracy and comfort. We train and validate our approach on rich, real-world traffic datasets encompassing diverse vehicles, cyclists, and pedestrians, and demonstrate that it outperforms baseline methods in reducing ethical risk and maintaining driving performance. To our knowledge, this is the first study of ethical decision-making for autonomous vehicles via Safe RL in real-world scenarios. Our results highlight the potential of combining formal control theory and data-driven learning to advance ethically accountable autonomy in complex, human-mixed traffic environments. 

**Abstract (ZH)**: 自主驾驶车辆在降低交通死亡事故和提高运输效率方面持有巨大潜力，但其广泛应用依赖于在常规和应急操作中嵌入 robust 的伦理推理。本文提出了一种分层安全强化学习（Safe RL）框架，明确整合了道德考量与标准驾驶目标。在决策层面，通过结合碰撞概率和损害严重性的复合伦理风险成本训练安全RL代理，生成高阶运动目标。动态优先经验重放机制增强了对罕见但关键的高风险事件的学习。在执行层面，结合多项式路径规划与比例积分微分（PID）和斯坦利控制器，将这些目标转化为平滑、可行的轨迹，确保精确性和舒适性。我们在包含多种车辆、自行车和行人的丰富真实世界交通数据集上训练和验证了该方法，并证明其在减少伦理风险和保持驾驶性能方面优于基线方法。据我们所知，这是首次在真实场景中通过安全强化学习进行自主车辆伦理决策的研究。我们的结果突显了将形式控制理论与数据驱动学习相结合，在复杂、人混杂的交通环境中推进负责任自主性的潜力。 

---
