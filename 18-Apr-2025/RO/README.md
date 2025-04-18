# ViTa-Zero: Zero-shot Visuotactile Object 6D Pose Estimation 

**Title (ZH)**: ViTa-Zero: Zero-shot 视触觉对象6D姿态估计 

**Authors**: Hongyu Li, James Akl, Srinath Sridhar, Tye Brady, Taskin Padir  

**Link**: [PDF](https://arxiv.org/pdf/2504.13179)  

**Abstract**: Object 6D pose estimation is a critical challenge in robotics, particularly for manipulation tasks. While prior research combining visual and tactile (visuotactile) information has shown promise, these approaches often struggle with generalization due to the limited availability of visuotactile data. In this paper, we introduce ViTa-Zero, a zero-shot visuotactile pose estimation framework. Our key innovation lies in leveraging a visual model as its backbone and performing feasibility checking and test-time optimization based on physical constraints derived from tactile and proprioceptive observations. Specifically, we model the gripper-object interaction as a spring-mass system, where tactile sensors induce attractive forces, and proprioception generates repulsive forces. We validate our framework through experiments on a real-world robot setup, demonstrating its effectiveness across representative visual backbones and manipulation scenarios, including grasping, object picking, and bimanual handover. Compared to the visual models, our approach overcomes some drastic failure modes while tracking the in-hand object pose. In our experiments, our approach shows an average increase of 55% in AUC of ADD-S and 60% in ADD, along with an 80% lower position error compared to FoundationPose. 

**Abstract (ZH)**: 基于视觉和触觉的六自由度姿态估计在机器人领域的零样本框架 

---
# Novel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot Manipulation 

**Title (ZH)**: 基于高斯点云的新示例生成实现稳健的一次性操作 Manipulation 

**Authors**: Sizhe Yang, Wenye Yu, Jia Zeng, Jun Lv, Kerui Ren, Cewu Lu, Dahua Lin, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13175)  

**Abstract**: Visuomotor policies learned from teleoperated demonstrations face challenges such as lengthy data collection, high costs, and limited data diversity. Existing approaches address these issues by augmenting image observations in RGB space or employing Real-to-Sim-to-Real pipelines based on physical simulators. However, the former is constrained to 2D data augmentation, while the latter suffers from imprecise physical simulation caused by inaccurate geometric reconstruction. This paper introduces RoboSplat, a novel method that generates diverse, visually realistic demonstrations by directly manipulating 3D Gaussians. Specifically, we reconstruct the scene through 3D Gaussian Splatting (3DGS), directly edit the reconstructed scene, and augment data across six types of generalization with five techniques: 3D Gaussian replacement for varying object types, scene appearance, and robot embodiments; equivariant transformations for different object poses; visual attribute editing for various lighting conditions; novel view synthesis for new camera perspectives; and 3D content generation for diverse object types. Comprehensive real-world experiments demonstrate that RoboSplat significantly enhances the generalization of visuomotor policies under diverse disturbances. Notably, while policies trained on hundreds of real-world demonstrations with additional 2D data augmentation achieve an average success rate of 57.2%, RoboSplat attains 87.8% in one-shot settings across six types of generalization in the real world. 

**Abstract (ZH)**: RoboSplat：通过直接操纵3D高斯分布生成多样且视觉上逼真的演示以增强感知运动策略的泛化能力 

---
# A New Semidefinite Relaxation for Linear and Piecewise-Affine Optimal Control with Time Scaling 

**Title (ZH)**: 一种新的半定松弛方法，用于具有时间缩放的线性与分段线性最优控制 

**Authors**: Lujie Yang, Tobia Marcucci, Pablo A. Parrilo, Russ Tedrake  

**Link**: [PDF](https://arxiv.org/pdf/2504.13170)  

**Abstract**: We introduce a semidefinite relaxation for optimal control of linear systems with time scaling. These problems are inherently nonconvex, since the system dynamics involves bilinear products between the discretization time step and the system state and controls. The proposed relaxation is closely related to the standard second-order semidefinite relaxation for quadratic constraints, but we carefully select a subset of the possible bilinear terms and apply a change of variables to achieve empirically tight relaxations while keeping the computational load light. We further extend our method to handle piecewise-affine (PWA) systems by formulating the PWA optimal-control problem as a shortest-path problem in a graph of convex sets (GCS). In this GCS, different paths represent different mode sequences for the PWA system, and the convex sets model the relaxed dynamics within each mode. By combining a tight convex relaxation of the GCS problem with our semidefinite relaxation with time scaling, we can solve PWA optimal-control problems through a single semidefinite program. 

**Abstract (ZH)**: 时间尺度下的线性系统最优控制的半定 Relaxation 方法及其在分段线性 affine 系统中的应用 

---
# RUKA: Rethinking the Design of Humanoid Hands with Learning 

**Title (ZH)**: RUKA: 重新思考类人手的设计以学习 

**Authors**: Anya Zorin, Irmak Guzey, Billy Yan, Aadhithya Iyer, Lisa Kondrich, Nikhil X. Bhattasali, Lerrel Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2504.13165)  

**Abstract**: Dexterous manipulation is a fundamental capability for robotic systems, yet progress has been limited by hardware trade-offs between precision, compactness, strength, and affordability. Existing control methods impose compromises on hand designs and applications. However, learning-based approaches present opportunities to rethink these trade-offs, particularly to address challenges with tendon-driven actuation and low-cost materials. This work presents RUKA, a tendon-driven humanoid hand that is compact, affordable, and capable. Made from 3D-printed parts and off-the-shelf components, RUKA has 5 fingers with 15 underactuated degrees of freedom enabling diverse human-like grasps. Its tendon-driven actuation allows powerful grasping in a compact, human-sized form factor. To address control challenges, we learn joint-to-actuator and fingertip-to-actuator models from motion-capture data collected by the MANUS glove, leveraging the hand's morphological accuracy. Extensive evaluations demonstrate RUKA's superior reachability, durability, and strength compared to other robotic hands. Teleoperation tasks further showcase RUKA's dexterous movements. The open-source design and assembly instructions of RUKA, code, and data are available at this https URL. 

**Abstract (ZH)**: Dexterous Manipulation是机器人系统的一项基本能力，但由于精确度、紧凑性、力量和成本之间的硬件权衡限制了进展。现有的控制方法在手的设计和应用方面强制执行妥协。然而，基于学习的方法提供了重新思考这些权衡的机会，特别是在解决基于肌腱驱动的执行和低成本材料的挑战方面。本项工作介绍了RUKA，一种紧凑、经济实惠且功能强大的肌腱驱动类人手。RUKA采用3D打印部件和现成组件制作，拥有5个手指和15个欠驱动自由度，可实现多种类似人类的抓取。其肌腱驱动的执行机构允许在一个紧凑的人类尺寸的封装中进行强大的抓取。为了解决控制方面的挑战，我们利用MANUS手套捕获的运动捕捉数据，学习关节到执行器和指尖到执行器的模型，利用手部的形态准确性。广泛的评估表明，与其它机器人手相比，RUKA在可达性、耐用性和强度方面表现出优越性。远程操作任务进一步展示了RUKA的灵巧动作。RUKA的开源设计和组装说明、代码和数据可在以下链接获得。 

---
# Long Range Navigator (LRN): Extending robot planning horizons beyond metric maps 

**Title (ZH)**: 长距离导航器（LRN）：将机器人规划视野扩展至超越度量地图之外 

**Authors**: Matt Schmittle, Rohan Baijal, Nathan Hatch, Rosario Scalise, Mateo Guaman Castro, Sidharth Talia, Khimya Khetarpal, Byron Boots, Siddhartha Srinivasa  

**Link**: [PDF](https://arxiv.org/pdf/2504.13149)  

**Abstract**: A robot navigating an outdoor environment with no prior knowledge of the space must rely on its local sensing to perceive its surroundings and plan. This can come in the form of a local metric map or local policy with some fixed horizon. Beyond that, there is a fog of unknown space marked with some fixed cost. A limited planning horizon can often result in myopic decisions leading the robot off course or worse, into very difficult terrain. Ideally, we would like the robot to have full knowledge that can be orders of magnitude larger than a local cost map. In practice, this is intractable due to sparse sensing information and often computationally expensive. In this work, we make a key observation that long-range navigation only necessitates identifying good frontier directions for planning instead of full map knowledge. To this end, we propose Long Range Navigator (LRN), that learns an intermediate affordance representation mapping high-dimensional camera images to `affordable' frontiers for planning, and then optimizing for maximum alignment with the desired goal. LRN notably is trained entirely on unlabeled ego-centric videos making it easy to scale and adapt to new platforms. Through extensive off-road experiments on Spot and a Big Vehicle, we find that augmenting existing navigation stacks with LRN reduces human interventions at test-time and leads to faster decision making indicating the relevance of LRN. this https URL 

**Abstract (ZH)**: 一种在无先验环境知识的情况下导航室外环境的机器人必须依赖于局部感知来感知其周围环境并规划路径。这可以表现为局部度量地图或具有固定预测 horizon 的局部策略。超出这一范围，存在未知的“模糊”区域，标记有一定的固定成本。固定的规划 horizon 往往会导致短视的决策，使机器人偏离航线，甚至进入非常困难的地形。理想情况下，我们希望机器人具备全面的知识，这种知识可能是局部成本图的成千上万倍。然而，由于稀疏的感知信息和通常计算上的昂贵，这往往是不可行的。在此工作中，我们关键地观察到，远程导航只需要识别出适合规划的前沿方向，而非全面的地图知识。为此，我们提出了Long Range Navigator (LRN)，它学习一个中间的操作潜能表示，将高维相机图像映射到可用于规划的“可负担”的前沿，并优化与目标的最大对齐度。LRN 特别是通过全未标记的自视点视频进行训练，使其易于扩展并适应新的平台。通过在 Spot 和一辆大型车辆上的离路面实验，我们发现，将 LRN 与现有的导航堆栈结合使用，在测试时减少了人工干预，并导致更快的决策，表明了 LRN 的相关性。 

---
# Force and Speed in a Soft Stewart Platform 

**Title (ZH)**: 软斯坦利平台的力与速度研究 

**Authors**: Jake Ketchum, James Avtges, Millicent Schlafly, Helena Young, Taekyoung Kim, Ryan L. Truby, Todd D. Murphey  

**Link**: [PDF](https://arxiv.org/pdf/2504.13127)  

**Abstract**: Many soft robots struggle to produce dynamic motions with fast, large displacements. We develop a parallel 6 degree-of-freedom (DoF) Stewart-Gough mechanism using Handed Shearing Auxetic (HSA) actuators. By using soft actuators, we are able to use one third as many mechatronic components as a rigid Stewart platform, while retaining a working payload of 2kg and an open-loop bandwidth greater than 16Hx. We show that the platform is capable of both precise tracing and dynamic disturbance rejection when controlling a ball and sliding puck using a Proportional Integral Derivative (PID) controller. We develop a machine-learning-based kinematics model and demonstrate a functional workspace of roughly 10cm in each translation direction and 28 degrees in each orientation. This 6DoF device has many of the characteristics associated with rigid components - power, speed, and total workspace - while capturing the advantages of soft mechanisms. 

**Abstract (ZH)**: 基于HSA执行器的并行6自由度斯特尔-戈奇机制：软执行器在精确跟踪和动态扰动拒绝中的应用及机器学习动力学模型 

---
# Imperative MPC: An End-to-End Self-Supervised Learning with Differentiable MPC for UAV Attitude Control 

**Title (ZH)**: imperative MPC：基于可微分MPC的端到端自监督学习无人机姿态控制 

**Authors**: Haonan He, Yuheng Qiu, Junyi Geng  

**Link**: [PDF](https://arxiv.org/pdf/2504.13088)  

**Abstract**: Modeling and control of nonlinear dynamics are critical in robotics, especially in scenarios with unpredictable external influences and complex dynamics. Traditional cascaded modular control pipelines often yield suboptimal performance due to conservative assumptions and tedious parameter tuning. Pure data-driven approaches promise robust performance but suffer from low sample efficiency, sim-to-real gaps, and reliance on extensive datasets. Hybrid methods combining learning-based and traditional model-based control in an end-to-end manner offer a promising alternative. This work presents a self-supervised learning framework combining learning-based inertial odometry (IO) module and differentiable model predictive control (d-MPC) for Unmanned Aerial Vehicle (UAV) attitude control. The IO denoises raw IMU measurements and predicts UAV attitudes, which are then optimized by MPC for control actions in a bi-level optimization (BLO) setup, where the inner MPC optimizes control actions and the upper level minimizes discrepancy between real-world and predicted performance. The framework is thus end-to-end and can be trained in a self-supervised manner. This approach combines the strength of learning-based perception with the interpretable model-based control. Results show the effectiveness even under strong wind. It can simultaneously enhance both the MPC parameter learning and IMU prediction performance. 

**Abstract (ZH)**: 基于自监督学习的集成学习导向航命周期控制框架用于UAV姿态控制 

---
# RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins 

**Title (ZH)**: RoboTwin: 双臂机器人基准测试与生成式数字孪生 

**Authors**: Yao Mu, Tianxing Chen, Zanxin Chen, Shijia Peng, Zhiqian Lan, Zeyu Gao, Zhixuan Liang, Qiaojun Yu, Yude Zou, Mingkun Xu, Lunkai Lin, Zhiqiang Xie, Mingyu Ding, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.13059)  

**Abstract**: In the rapidly advancing field of robotics, dual-arm coordination and complex object manipulation are essential capabilities for developing advanced autonomous systems. However, the scarcity of diverse, high-quality demonstration data and real-world-aligned evaluation benchmarks severely limits such development. To address this, we introduce RoboTwin, a generative digital twin framework that uses 3D generative foundation models and large language models to produce diverse expert datasets and provide a real-world-aligned evaluation platform for dual-arm robotic tasks. Specifically, RoboTwin creates varied digital twins of objects from single 2D images, generating realistic and interactive scenarios. It also introduces a spatial relation-aware code generation framework that combines object annotations with large language models to break down tasks, determine spatial constraints, and generate precise robotic movement code. Our framework offers a comprehensive benchmark with both simulated and real-world data, enabling standardized evaluation and better alignment between simulated training and real-world performance. We validated our approach using the open-source COBOT Magic Robot platform. Policies pre-trained on RoboTwin-generated data and fine-tuned with limited real-world samples demonstrate significant potential for enhancing dual-arm robotic manipulation systems by improving success rates by over 70% for single-arm tasks and over 40% for dual-arm tasks compared to models trained solely on real-world data. 

**Abstract (ZH)**: 机器人领域中双臂协调和复杂物体操作是开发高级自主系统的关键能力。然而，缺乏多样性和高质量的示范数据以及与现实世界对齐的评估基准严重限制了这些能力的发展。为了解决这一问题，我们介绍了一种生成式数字孪生框架RoboTwin，该框架使用3D生成基础模型和大型语言模型来生成多样化的专家数据集，并提供一个与现实世界对齐的评估平台，用于双臂机器人任务。具体而言，RoboTwin从单张2D图像中创建多样化的数字孪生对象，生成真实且互动的场景。它还引入了一种空间关系感知的代码生成框架，将对象标注与大型语言模型相结合，分解任务、确定空间约束并生成精确的机器人运动代码。我们的框架提供了一个综合基准，包含模拟和真实世界数据，使标准化评价成为可能，并更好地实现了模拟训练与实际性能的对齐。我们使用开源COBOT Magic Robot平台验证了这种方法。在RoboTwin生成数据上预训练并在有限的真实世界样本上微调的策略，在单臂任务成功率提高超过70%、双臂任务成功率提高超过40%方面显示出显著潜力，这与仅在真实世界数据上训练的模型相比。 

---
# Krysalis Hand: A Lightweight, High-Payload, 18-DoF Anthropomorphic End-Effector for Robotic Learning and Dexterous Manipulation 

**Title (ZH)**: Krysalis 手部：一种轻量化、高负载、18 自由度的人类仿生末端执行器，用于机器人学习和灵巧操作 

**Authors**: Al Arsh Basheer, Justin Chang, Yuyang Chen, David Kim, Iman Soltani  

**Link**: [PDF](https://arxiv.org/pdf/2504.12967)  

**Abstract**: This paper presents the Krysalis Hand, a five-finger robotic end-effector that combines a lightweight design, high payload capacity, and a high number of degrees of freedom (DoF) to enable dexterous manipulation in both industrial and research settings. This design integrates the actuators within the hand while maintaining an anthropomorphic form. Each finger joint features a self-locking mechanism that allows the hand to sustain large external forces without active motor engagement. This approach shifts the payload limitation from the motor strength to the mechanical strength of the hand, allowing the use of smaller, more cost-effective motors. With 18 DoF and weighing only 790 grams, the Krysalis Hand delivers an active squeezing force of 10 N per finger and supports a passive payload capacity exceeding 10 lbs. These characteristics make Krysalis Hand one of the lightest, strongest, and most dexterous robotic end-effectors of its kind. Experimental evaluations validate its ability to perform intricate manipulation tasks and handle heavy payloads, underscoring its potential for industrial applications as well as academic research. All code related to the Krysalis Hand, including control and teleoperation, is available on the project GitHub repository: this https URL 

**Abstract (ZH)**: 本文介绍了Krysalis手部，这是一种结合轻量化设计、高负载能力和大量自由度（DoF）的五指仿人机器人末端执行器，能够在工业和研究环境中实现灵巧操作。该设计将驱动器集成在手部中，同时保持仿人形结构。每个手指关节配备了自锁机制，使手部能够在不主动参与电机驱动的情况下承受较大的外部力量。这种方法将负载限制从电机强度转移到手部的机械强度上，从而能够使用更小、更经济的电机。Krysalis手部拥有18个自由度，重量仅790克，每指提供10 N的主动捏力，并支持超过10磅的被动负载能力。这些特性使其成为此类中最轻、最坚固和最灵巧的机器人末端执行器之一。实验评估验证了其执行复杂操作任务和处理重负载的能力，凸显了其在工业应用和学术研究中的潜力。Krysalis手部的相关代码（包括控制和远程操作）均可在项目GitHub仓库中获得：this https URL 

---
# Taccel: Scaling Up Vision-based Tactile Robotics via High-performance GPU Simulation 

**Title (ZH)**: Taccel: 通过高性能GPU仿真扩展基于视觉的触觉机器人技术 

**Authors**: Yuyang Li, Wenxin Du, Chang Yu, Puhao Li, Zihang Zhao, Tengyu Liu, Chenfanfu Jiang, Yixin Zhu, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12908)  

**Abstract**: Tactile sensing is crucial for achieving human-level robotic capabilities in manipulation tasks. VBTSs have emerged as a promising solution, offering high spatial resolution and cost-effectiveness by sensing contact through camera-captured deformation patterns of elastic gel pads. However, these sensors' complex physical characteristics and visual signal processing requirements present unique challenges for robotic applications. The lack of efficient and accurate simulation tools for VBTS has significantly limited the scale and scope of tactile robotics research. Here we present Taccel, a high-performance simulation platform that integrates IPC and ABD to model robots, tactile sensors, and objects with both accuracy and unprecedented speed, achieving an 18-fold acceleration over real-time across thousands of parallel environments. Unlike previous simulators that operate at sub-real-time speeds with limited parallelization, Taccel provides precise physics simulation and realistic tactile signals while supporting flexible robot-sensor configurations through user-friendly APIs. Through extensive validation in object recognition, robotic grasping, and articulated object manipulation, we demonstrate precise simulation and successful sim-to-real transfer. These capabilities position Taccel as a powerful tool for scaling up tactile robotics research and development. By enabling large-scale simulation and experimentation with tactile sensing, Taccel accelerates the development of more capable robotic systems, potentially transforming how robots interact with and understand their physical environment. 

**Abstract (ZH)**: 触觉感知对于实现操纵任务中的人类级机器人能力至关重要。基于视觉的触觉传感器（VBTSs）因其通过弹性凝胶垫变形模式的相机捕获进行接触传感，提供了高空间分辨率和成本效益，而备受关注。然而，这些传感器复杂的物理特性和视觉信号处理要求为机器人应用带来了独特的挑战。缺乏高效的触觉机器人仿真工具显著限制了触觉机器人研究的规模和范围。本文介绍了一种高性能仿真平台Taccel，该平台结合了IPC和ABD，实现了机器人、触觉传感器和物体的高精度和前所未有的高速建模，相较于实时操作实现了18倍的速度提升，支持数千个并行环境。不同于之前运行在次实时速度且并行化有限的仿真的模拟器，Taccel提供了精确的物理模拟和-realistic触觉信号，同时通过用户友好的API支持灵活的机器人-传感器配置。通过在物体识别、机器人抓取和活动物体操纵方面的广泛验证，我们展示了精确的仿真和成功的仿真实验到现实世界的转移。这些能力使Taccel成为扩展触觉机器人研究和开发的强大工具。通过对触觉传感进行大规模仿真和实验，Taccel加速了更先进机器人系统的发展，有望改变机器人如何与和理解物理环境的方式。 

---
# Versatile, Robust, and Explosive Locomotion with Rigid and Articulated Compliant Quadrupeds 

**Title (ZH)**: 具有刚性与 articulated 柔顺四肢的多功能、 robust 和爆炸式运动的 quadruped 机器人 

**Authors**: Jiatao Ding, Peiyu Yang, Fabio Boekel, Jens Kober, Wei Pan, Matteo Saveriano, Cosimo Della Santina  

**Link**: [PDF](https://arxiv.org/pdf/2504.12854)  

**Abstract**: Achieving versatile and explosive motion with robustness against dynamic uncertainties is a challenging task. Introducing parallel compliance in quadrupedal design is deemed to enhance locomotion performance, which, however, makes the control task even harder. This work aims to address this challenge by proposing a general template model and establishing an efficient motion planning and control pipeline. To start, we propose a reduced-order template model-the dual-legged actuated spring-loaded inverted pendulum with trunk rotation-which explicitly models parallel compliance by decoupling spring effects from active motor actuation. With this template model, versatile acrobatic motions, such as pronking, froggy jumping, and hop-turn, are generated by a dual-layer trajectory optimization, where the singularity-free body rotation representation is taken into consideration. Integrated with a linear singularity-free tracking controller, enhanced quadrupedal locomotion is achieved. Comparisons with the existing template model reveal the improved accuracy and generalization of our model. Hardware experiments with a rigid quadruped and a newly designed compliant quadruped demonstrate that i) the template model enables generating versatile dynamic motion; ii) parallel elasticity enhances explosive motion. For example, the maximal pronking distance, hop-turn yaw angle, and froggy jumping distance increase at least by 25%, 15% and 25%, respectively; iii) parallel elasticity improves the robustness against dynamic uncertainties, including modelling errors and external disturbances. For example, the allowable support surface height variation increases by 100% for robust froggy jumping. 

**Abstract (ZH)**: 实现鲁棒性强的多功能和爆发性运动是一项挑战性任务。通过四足机器人设计引入并联柔顺性被认为能提升运动性能，然而这使得控制任务更复杂。本文通过提出一种通用的模板模型和建立一个高效的动力学规划与控制流水线来应对这一挑战。首先，我们提出一种简化版的模板模型——带有躯干旋转的双足主动簧载倒立摆模型，该模型通过解耦弹簧效应和主动电机驱动来明确建模并联柔顺性。基于该模板模型，通过双层轨迹优化生成多功能杂技动作，同时考虑到无奇点的体旋转表示。结合线性无奇点跟踪控制器，实现了增强的四足运动表现。与现有模板模型的对比证明了模型提高了精度和泛化能力。使用刚性四足机器人和新设计的柔性四足机器人的硬件实验表明：i) 模板模型能够生成多功能动态动作；ii) 并联弹性能够提升爆发性动作；例如，最大蹦跃距离、跳跃-转身航向角和青蛙跳跃距离分别至少增加25%、15%和25%；iii) 并联弹性提高了对动态不确定性（包括建模误差和外部干扰）的鲁棒性，例如，稳健青蛙跳跃时允许的支持表面高度变化范围增大了100%。 

---
# UncAD: Towards Safe End-to-end Autonomous Driving via Online Map Uncertainty 

**Title (ZH)**: UncAD：通过在线地图不确定性实现安全端到端自动驾驶 

**Authors**: Pengxuan Yang, Yupeng Zheng, Qichao Zhang, Kefei Zhu, Zebin Xing, Qiao Lin, Yun-Fu Liu, Zhiguo Su, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12826)  

**Abstract**: End-to-end autonomous driving aims to produce planning trajectories from raw sensors directly. Currently, most approaches integrate perception, prediction, and planning modules into a fully differentiable network, promising great scalability. However, these methods typically rely on deterministic modeling of online maps in the perception module for guiding or constraining vehicle planning, which may incorporate erroneous perception information and further compromise planning safety. To address this issue, we delve into the importance of online map uncertainty for enhancing autonomous driving safety and propose a novel paradigm named UncAD. Specifically, UncAD first estimates the uncertainty of the online map in the perception module. It then leverages the uncertainty to guide motion prediction and planning modules to produce multi-modal trajectories. Finally, to achieve safer autonomous driving, UncAD proposes an uncertainty-collision-aware planning selection strategy according to the online map uncertainty to evaluate and select the best trajectory. In this study, we incorporate UncAD into various state-of-the-art (SOTA) end-to-end methods. Experiments on the nuScenes dataset show that integrating UncAD, with only a 1.9% increase in parameters, can reduce collision rates by up to 26% and drivable area conflict rate by up to 42%. Codes, pre-trained models, and demo videos can be accessed at this https URL. 

**Abstract (ZH)**: 从传感器直接生成自主驾驶规划轨迹的端到端方法：考虑在线地图不确定性以增强自主驾驶安全 

---
# Explainable Scene Understanding with Qualitative Representations and Graph Neural Networks 

**Title (ZH)**: 可解释的场景理解：基于定性表示与图神经网络的方法 

**Authors**: Nassim Belmecheri, Arnaud Gotlieb, Nadjib Lazaar, Helge Spieker  

**Link**: [PDF](https://arxiv.org/pdf/2504.12817)  

**Abstract**: This paper investigates the integration of graph neural networks (GNNs) with Qualitative Explainable Graphs (QXGs) for scene understanding in automated driving. Scene understanding is the basis for any further reactive or proactive decision-making. Scene understanding and related reasoning is inherently an explanation task: why is another traffic participant doing something, what or who caused their actions? While previous work demonstrated QXGs' effectiveness using shallow machine learning models, these approaches were limited to analysing single relation chains between object pairs, disregarding the broader scene context. We propose a novel GNN architecture that processes entire graph structures to identify relevant objects in traffic scenes. We evaluate our method on the nuScenes dataset enriched with DriveLM's human-annotated relevance labels. Experimental results show that our GNN-based approach achieves superior performance compared to baseline methods. The model effectively handles the inherent class imbalance in relevant object identification tasks while considering the complete spatial-temporal relationships between all objects in the scene. Our work demonstrates the potential of combining qualitative representations with deep learning approaches for explainable scene understanding in autonomous driving systems. 

**Abstract (ZH)**: 基于图神经网络的质化可解释图在自动驾驶场景理解中的集成研究 

---
# Approaching Current Challenges in Developing a Software Stack for Fully Autonomous Driving 

**Title (ZH)**: 克服开发全自动驾驶软件栈面临的当前挑战 

**Authors**: Simon Sagmeister, Simon Hoffmann, Tobias Betz, Dominic Ebner, Daniel Esser, Markus Lienkamp  

**Link**: [PDF](https://arxiv.org/pdf/2504.12813)  

**Abstract**: Autonomous driving is a complex undertaking. A common approach is to break down the driving task into individual subtasks through modularization. These sub-modules are usually developed and published separately. However, if these individually developed algorithms have to be combined again to form a full-stack autonomous driving software, this poses particular challenges. Drawing upon our practical experience in developing the software of TUM Autonomous Motorsport, we have identified and derived these challenges in developing an autonomous driving software stack within a scientific environment. We do not focus on the specific challenges of individual algorithms but on the general difficulties that arise when deploying research algorithms on real-world test vehicles. To overcome these challenges, we introduce strategies that have been effective in our development approach. We additionally provide open-source implementations that enable these concepts on GitHub. As a result, this paper's contributions will simplify future full-stack autonomous driving projects, which are essential for a thorough evaluation of the individual algorithms. 

**Abstract (ZH)**: 自主驾驶是一项复杂的任务。一种常见的方法是通过模块化将驾驶任务分解为个体子任务。这些子模块通常分别开发和发布。然而，如果这些单独开发的算法需要重新组合以形成完整的自主驾驶软件堆栈，这将带来特定的挑战。基于我们为TUM Autonomous Motorsport开发软件的实际经验，我们在这篇论文中识别并分析了在科学环境中开发自主驾驶软件堆栈时遇到的一般困难。我们不聚焦于个别算法的具体挑战，而是关注部署研究算法到真实世界测试车辆时出现的一般困难。为了克服这些挑战，我们提出了在开发过程中有效的策略，并在GitHub上提供了开源实现，以促进这些概念的应用。因此，本文的贡献将简化未来的完整堆栈自主驾驶项目，这对于全面评估个别算法至关重要。 

---
# Trajectory Adaptation using Large Language Models 

**Title (ZH)**: 使用大规模语言模型进行轨迹适应 

**Authors**: Anurag Maurya, Tashmoy Ghosh, Ravi Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2504.12755)  

**Abstract**: Adapting robot trajectories based on human instructions as per new situations is essential for achieving more intuitive and scalable human-robot interactions. This work proposes a flexible language-based framework to adapt generic robotic trajectories produced by off-the-shelf motion planners like RRT, A-star, etc, or learned from human demonstrations. We utilize pre-trained LLMs to adapt trajectory waypoints by generating code as a policy for dense robot manipulation, enabling more complex and flexible instructions than current methods. This approach allows us to incorporate a broader range of commands, including numerical inputs. Compared to state-of-the-art feature-based sequence-to-sequence models which require training, our method does not require task-specific training and offers greater interpretability and more effective feedback mechanisms. We validate our approach through simulation experiments on the robotic manipulator, aerial vehicle, and ground robot in the Pybullet and Gazebo simulation environments, demonstrating that LLMs can successfully adapt trajectories to complex human instructions. 

**Abstract (ZH)**: 基于人类指令适应机器人轨迹对于实现更直观和可扩展的人机交互至关重要。本工作提出了一种灵活的语言框架，用于适应通用机器人轨迹，这些轨迹由RRT、A-Star等商用运动规划器生成，或从人类演示中学习。我们利用预训练的大语言模型（LLM）生成代码作为策略，以适应轨迹Waypoints，从而实现更复杂的灵活指令，这比现有方法更具优势。该方法允许我们纳入更广泛的命令，包括数值输入。与需要训练的最新基于特征的序列到序列模型相比，我们的方法无需特定任务的训练，提供了更高的可解释性和更有效的反馈机制。我们通过在Pybullet和Gazebo模拟环境中对 manipulator、无人机和地面机器人进行仿真实验，验证了该方法的有效性，展示了大语言模型能够成功适应复杂的人类指令。 

---
# Biasing the Driving Style of an Artificial Race Driver for Online Time-Optimal Maneuver Planning 

**Title (ZH)**: 为在线最优机动规划调整人工赛车司机的驾驶风格偏差 

**Authors**: Sebastiano Taddei, Mattia Piccinini, Francesco Biral  

**Link**: [PDF](https://arxiv.org/pdf/2504.12744)  

**Abstract**: In this work, we present a novel approach to bias the driving style of an artificial race driver (ARD) for online time-optimal trajectory planning. Our method leverages a nonlinear model predictive control (MPC) framework that combines time minimization with exit speed maximization at the end of the planning horizon. We introduce a new MPC terminal cost formulation based on the trajectory planned in the previous MPC step, enabling ARD to adapt its driving style from early to late apex maneuvers in real-time. Our approach is computationally efficient, allowing for low replan times and long planning horizons. We validate our method through simulations, comparing the results against offline minimum-lap-time (MLT) optimal control and online minimum-time MPC solutions. The results demonstrate that our new terminal cost enables ARD to bias its driving style, and achieve online lap times close to the MLT solution and faster than the minimum-time MPC solution. Our approach paves the way for a better understanding of the reasons behind human drivers' choice of early or late apex maneuvers. 

**Abstract (ZH)**: 本研究提出了一种新颖的方法，用于为在线时间最优轨迹规划偏置人工赛车手（ARD）的驾驶风格。我们的方法利用了结合时间最小化和规划末期出口速度最大化的非线性模型预测控制（MPC）框架。我们引入了一种新的MPC终端成本公式，基于上一步MPC计划的轨迹，使ARD能够实时适应其驾驶风格，从早期到晚期的转向动作。该方法计算效率高，允许低重规划时间和长规划时间。我们通过仿真验证了该方法，将结果与离线最短圈时（MLT）最优控制和在线最短时间MPC解进行了比较。结果表明，我们的新终端成本使ARD能够偏置其驾驶风格，并实现接近MLT解的在线圈时，且快于最短时间MPC解。该方法为更好地理解人类驾驶员选择早期或晚期转向动作的原因铺平了道路。 

---
# B*: Efficient and Optimal Base Placement for Fixed-Base Manipulators 

**Title (ZH)**: B*: 固定基座 manipulator 有效且最优的基座位置规划 

**Authors**: Zihang Zhao, Leiyao Cui, Sirui Xie, Saiyao Zhang, Zhi Han, Lecheng Ruan, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12719)  

**Abstract**: B* is a novel optimization framework that addresses a critical challenge in fixed-base manipulator robotics: optimal base placement. Current methods rely on pre-computed kinematics databases generated through sampling to search for solutions. However, they face an inherent trade-off between solution optimality and computational efficiency when determining sampling resolution. To address these limitations, B* unifies multiple objectives without database dependence. The framework employs a two-layer hierarchical approach. The outer layer systematically manages terminal constraints through progressive tightening, particularly for base mobility, enabling feasible initialization and broad solution exploration. The inner layer addresses non-convexities in each outer-layer subproblem through sequential local linearization, converting the original problem into tractable sequential linear programming (SLP). Testing across multiple robot platforms demonstrates B*'s effectiveness. The framework achieves solution optimality five orders of magnitude better than sampling-based approaches while maintaining perfect success rates and reduced computational overhead. Operating directly in configuration space, B* enables simultaneous path planning with customizable optimization criteria. B* serves as a crucial initialization tool that bridges the gap between theoretical motion planning and practical deployment, where feasible trajectory existence is fundamental. 

**Abstract (ZH)**: B* 是一种新型优化框架，针对固定基座 manipulator 机器人中的关键挑战：最优基座定位进行了优化。当前方法依赖于通过采样生成的预计算运动学数据库来搜索解决方案。然而，它们在确定采样分辨率时面临解决方案最优性和计算效率之间的固有权衡。为了解决这些限制，B* 统一了多个目标，而不依赖于数据库。该框架采用两层层次结构方法。外层通过逐步收紧终端约束系统地管理终端约束，特别是对于基座的移动性，从而实现可行的初始化和广泛的解决方案探索。内层通过逐步局部线性化解决每一层子问题中的非凸性，将原始问题转化为可处理的顺序线性规划（SLP）。在多个机器人平台上进行的测试表明 B* 的有效性。该框架在保持完美成功率的同时，计算开销减少，并且解决方案最优性比基于采样的方法高出五个数量级。B* 直接在配置空间中操作，同时实现路径规划和可定制的优化标准。B* 作为理论运动规划与实际部署之间的重要初始化工具，其核心在于可行轨迹的存在至关重要。 

---
# Embodied Neuromorphic Control Applied on a 7-DOF Robotic Manipulator 

**Title (ZH)**: 具身神经形态控制在7-自由度机器人 manipulator 上的应用 

**Authors**: Ziqi Wang, Jingyue Zhao, Jichao Yang, Yaohua Wang, Xun Xiao, Yuan Li, Chao Xiao, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12702)  

**Abstract**: The development of artificial intelligence towards real-time interaction with the environment is a key aspect of embodied intelligence and robotics. Inverse dynamics is a fundamental robotics problem, which maps from joint space to torque space of robotic systems. Traditional methods for solving it rely on direct physical modeling of robots which is difficult or even impossible due to nonlinearity and external disturbance. Recently, data-based model-learning algorithms are adopted to address this issue. However, they often require manual parameter tuning and high computational costs. Neuromorphic computing is inherently suitable to process spatiotemporal features in robot motion control at extremely low costs. However, current research is still in its infancy: existing works control only low-degree-of-freedom systems and lack performance quantification and comparison. In this paper, we propose a neuromorphic control framework to control 7 degree-of-freedom robotic manipulators. We use Spiking Neural Network to leverage the spatiotemporal continuity of the motion data to improve control accuracy, and eliminate manual parameters tuning. We validated the algorithm on two robotic platforms, which reduces torque prediction error by at least 60% and performs a target position tracking task successfully. This work advances embodied neuromorphic control by one step forward from proof of concept to applications in complex real-world tasks. 

**Abstract (ZH)**: 人工智能向实时环境交互的发展是本体智能和机器人技术的关键aspect。逆动力学是机器人技术中的一个基础问题，它将关节空间映射到机器人系统的扭矩空间。传统的方法依赖于对机器人的直接物理建模，但由于非线性和外部干扰，这通常是困难的或甚至不可能的。近年来，基于数据的模型学习算法被采用来解决这个问题。然而，它们通常需要手动参数调整并且计算成本高。神经形态计算本质上适合以极低的成本处理机器人运动控制中的时空特征。然而，当前的研究仍处于初级阶段：现有的工作只控制低自由度系统，并缺乏性能量化和比较。在本文中，我们提出了一种神经形态控制框架来控制7自由度的机器人 manipulator。我们使用脉冲神经网络利用运动数据的时空连续性以提高控制精度并消除手动参数调整。我们在两个机器人平台上验证了该算法，降低了至少60%的扭矩预测误差并成功执行了目标位置跟踪任务。这项工作将本体神经形态控制从概念证明推进到复杂实际任务的应用中。 

---
# A Genetic Approach to Gradient-Free Kinodynamic Planning in Uneven Terrains 

**Title (ZH)**: 基于遗传算法的无导数动力学规划在不平地形中 

**Authors**: Otobong Jerome, Alexandr Klimchik, Alexander Maloletov, Geesara Kulathunga  

**Link**: [PDF](https://arxiv.org/pdf/2504.12678)  

**Abstract**: This paper proposes a genetic algorithm-based kinodynamic planning algorithm (GAKD) for car-like vehicles navigating uneven terrains modeled as triangular meshes. The algorithm's distinct feature is trajectory optimization over a fixed-length receding horizon using a genetic algorithm with heuristic-based mutation, ensuring the vehicle's controls remain within its valid operational range. By addressing challenges posed by uneven terrain meshes, such as changing face normals, GAKD offers a practical solution for path planning in complex environments. Comparative evaluations against Model Predictive Path Integral (MPPI) and log-MPPI methods show that GAKD achieves up to 20 percent improvement in traversability cost while maintaining comparable path length. These results demonstrate GAKD's potential in improving vehicle navigation on challenging terrains. 

**Abstract (ZH)**: 基于遗传算法的 kinodynamic 规划算法 (GAKD)：汽车在三角网模型的不平地形导航轨迹优化 

---
# Autonomous Drone for Dynamic Smoke Plume Tracking 

**Title (ZH)**: 自主无人机动态烟柱追踪 

**Authors**: Srijan Kumar Pal, Shashank Sharma, Nikil Krishnakumar, Jiarong Hong  

**Link**: [PDF](https://arxiv.org/pdf/2504.12664)  

**Abstract**: This paper presents a novel autonomous drone-based smoke plume tracking system capable of navigating and tracking plumes in highly unsteady atmospheric conditions. The system integrates advanced hardware and software and a comprehensive simulation environment to ensure robust performance in controlled and real-world settings. The quadrotor, equipped with a high-resolution imaging system and an advanced onboard computing unit, performs precise maneuvers while accurately detecting and tracking dynamic smoke plumes under fluctuating conditions. Our software implements a two-phase flight operation, i.e., descending into the smoke plume upon detection and continuously monitoring the smoke movement during in-plume tracking. Leveraging Proportional Integral-Derivative (PID) control and a Proximal Policy Optimization based Deep Reinforcement Learning (DRL) controller enables adaptation to plume dynamics. Unreal Engine simulation evaluates performance under various smoke-wind scenarios, from steady flow to complex, unsteady fluctuations, showing that while the PID controller performs adequately in simpler scenarios, the DRL-based controller excels in more challenging environments. Field tests corroborate these findings. This system opens new possibilities for drone-based monitoring in areas like wildfire management and air quality assessment. The successful integration of DRL for real-time decision-making advances autonomous drone control for dynamic environments. 

**Abstract (ZH)**: 基于自主无人机的烟柱跟踪系统：在高度不稳定的气象条件下导航与跟踪 

---
# A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation 

**Title (ZH)**: A0: 一种考虑利用条件的分层模型 general robotic manipulation 

**Authors**: Rongtao Xu, Jian Zhang, Minghao Guo, Youpeng Wen, Haoting Yang, Min Lin, Jianzheng Huang, Zhe Li, Kaidong Zhang, Liqiong Wang, Yuxuan Kuang, Meng Cao, Feng Zheng, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12636)  

**Abstract**: Robotic manipulation faces critical challenges in understanding spatial affordances--the "where" and "how" of object interactions--essential for complex manipulation tasks like wiping a board or stacking objects. Existing methods, including modular-based and end-to-end approaches, often lack robust spatial reasoning capabilities. Unlike recent point-based and flow-based affordance methods that focus on dense spatial representations or trajectory modeling, we propose A0, a hierarchical affordance-aware diffusion model that decomposes manipulation tasks into high-level spatial affordance understanding and low-level action execution. A0 leverages the Embodiment-Agnostic Affordance Representation, which captures object-centric spatial affordances by predicting contact points and post-contact trajectories. A0 is pre-trained on 1 million contact points data and fine-tuned on annotated trajectories, enabling generalization across platforms. Key components include Position Offset Attention for motion-aware feature extraction and a Spatial Information Aggregation Layer for precise coordinate mapping. The model's output is executed by the action execution module. Experiments on multiple robotic systems (Franka, Kinova, Realman, and Dobot) demonstrate A0's superior performance in complex tasks, showcasing its efficiency, flexibility, and real-world applicability. 

**Abstract (ZH)**: 机器人操作在理解空间 afforded 性能方面面临关键挑战——这对于擦黑板或堆叠物体等复杂操作任务的“哪里”和“如何”至关重要。现有方法，包括模块化和端到端方法，通常缺乏 robust 的空间推理能力。有别于最近基于点和流的方法，这些方法集中在密集的空间表示或轨迹建模上，我们提出 A0，一种分层的认知操作扩散模型，将操作任务分解为高层次的空间 afforded 性理解与低层次的操作执行。A0 利用体无关的 afforded 性表示，通过预测接触点和接触后的轨迹来捕获以对象为中心的空间 afforded 性。A0 在 100 万接触点数据上进行了预训练，并在标注的轨迹上进行了微调，使其实现跨平台应用。关键组件包括位置偏移注意力机制，用于运动感知特征提取，以及空间信息聚合层，用于精确的坐标映射。模型的输出由操作执行模块执行。在多个机器人系统（Franka、Kinova、Realman 和 Dobot）上的实验表明，A0 在复杂任务中的性能优越，展示了其高效性、灵活性和实际应用性。 

---
# Graph-based Path Planning with Dynamic Obstacle Avoidance for Autonomous Parking 

**Title (ZH)**: 基于图的路径规划与动态障碍避障的自主停车 

**Authors**: Farhad Nawaz, Minjun Sung, Darshan Gadginmath, Jovin D'sa, Sangjae Bae, David Isele, Nadia Figueroa, Nikolai Matni, Faizan M. Tariq  

**Link**: [PDF](https://arxiv.org/pdf/2504.12616)  

**Abstract**: Safe and efficient path planning in parking scenarios presents a significant challenge due to the presence of cluttered environments filled with static and dynamic obstacles. To address this, we propose a novel and computationally efficient planning strategy that seamlessly integrates the predictions of dynamic obstacles into the planning process, ensuring the generation of collision-free paths. Our approach builds upon the conventional Hybrid A star algorithm by introducing a time-indexed variant that explicitly accounts for the predictions of dynamic obstacles during node exploration in the graph, thus enabling dynamic obstacle avoidance. We integrate the time-indexed Hybrid A star algorithm within an online planning framework to compute local paths at each planning step, guided by an adaptively chosen intermediate goal. The proposed method is validated in diverse parking scenarios, including perpendicular, angled, and parallel parking. Through simulations, we showcase our approach's potential in greatly improving the efficiency and safety when compared to the state of the art spline-based planning method for parking situations. 

**Abstract (ZH)**: 密 clustering 境环境下安全高效的道路规划面临着显著挑战，由于存在静态和动态障碍物。为了解决这一问题，我们提出了一种新颖且计算高效的规划策略，该策略将动态障碍物的预测无缝集成到规划过程中，确保生成无碰撞路径。我们的方法基于传统的Hybrid A*算法，并引入了一种时间索引变体，在图的节点探索过程中明确考虑动态障碍物的预测，从而实现动态障碍物的避让。我们将时间索引Hybrid A*算法集成到在线规划框架中，在每步规划时根据适应性选择的中间目标来计算局部路径。所提出的方法在垂直、斜角和平行停车等多种停车场景中得到了验证。通过仿真的方式，我们展示了该方法在效率和安全性方面较基于样条的最新规划方法的巨大改进潜力。 

---
# Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration 

**Title (ZH)**: 使用一次人类示范的从模拟到现实的RL跨人类-机器人体态 gaps 方法 

**Authors**: Tyler Ga Wei Lum, Olivia Y. Lee, C. Karen Liu, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2504.12609)  

**Abstract**: Teaching robots dexterous manipulation skills often requires collecting hundreds of demonstrations using wearables or teleoperation, a process that is challenging to scale. Videos of human-object interactions are easier to collect and scale, but leveraging them directly for robot learning is difficult due to the lack of explicit action labels from videos and morphological differences between robot and human hands. We propose Human2Sim2Robot, a novel real-to-sim-to-real framework for training dexterous manipulation policies using only one RGB-D video of a human demonstrating a task. Our method utilizes reinforcement learning (RL) in simulation to cross the human-robot embodiment gap without relying on wearables, teleoperation, or large-scale data collection typically necessary for imitation learning methods. From the demonstration, we extract two task-specific components: (1) the object pose trajectory to define an object-centric, embodiment-agnostic reward function, and (2) the pre-manipulation hand pose to initialize and guide exploration during RL training. We found that these two components are highly effective for learning the desired task, eliminating the need for task-specific reward shaping and tuning. We demonstrate that Human2Sim2Robot outperforms object-aware open-loop trajectory replay by 55% and imitation learning with data augmentation by 68% across grasping, non-prehensile manipulation, and multi-step tasks. Project Site: this https URL 

**Abstract (ZH)**: 将人类的灵巧操作技能传授给机器人通常需要收集 hundreds of 示范，这使用可穿戴设备或远程操作来完成，但这一过程难以扩展。人类与物体的交互视频更容易收集和扩展，但在不依赖于可穿戴设备、远程操作或通常需要的大量数据收集的情况下，直接利用这些视频进行机器人学习是非常困难的，因为机器人和人类的手部形态差异以及视频中缺乏明确的动作标签。我们提出了一种名为 Human2Sim2Robot 的新颖的从真实世界到模拟再到现实的框架，仅使用一条 RGB-D 视频即可训练灵巧操作策略。该方法利用模拟环境中的强化学习（RL）跨越了人类与机器人形态的鸿沟，而无需依赖可穿戴设备、远程操作或传统的数据收集。从演示中，我们提取了两个任务特定的组件：(1) 物体姿态轨迹，用于定义一个基于物体而非具体形态的、不受表现形式影响的奖励函数，以及 (2) 操作前的手部姿态，用于初始化和指导 RL 训练期间的探索。我们发现这两个组件对于学习所需的任务非常有效，消除了任务特定奖励塑造和调优的需要。实验结果表明，Human2Sim2Robot 在抓取、非抓握操作和多步骤任务上分别比对象感知的开环轨迹重放性能高出55%，比具有数据增强的模仿学习高出68%。项目地址：this https URL 

---
# Practical Insights on Grasp Strategies for Mobile Manipulation in the Wild 

**Title (ZH)**: 移动操作中的野外抓取策略实用洞察 

**Authors**: Isabella Huang, Richard Cheng, Sangwoon Kim, Dan Kruse, Carolyn Matl, Lukas Kaul, JC Hancock, Shanmuga Harikumar, Mark Tjersland, James Borders, Dan Helmick  

**Link**: [PDF](https://arxiv.org/pdf/2504.12512)  

**Abstract**: Mobile manipulation robots are continuously advancing, with their grasping capabilities rapidly progressing. However, there are still significant gaps preventing state-of-the-art mobile manipulators from widespread real-world deployments, including their ability to reliably grasp items in unstructured environments. To help bridge this gap, we developed SHOPPER, a mobile manipulation robot platform designed to push the boundaries of reliable and generalizable grasp strategies. We develop these grasp strategies and deploy them in a real-world grocery store -- an exceptionally challenging setting chosen for its vast diversity of manipulable items, fixtures, and layouts. In this work, we present our detailed approach to designing general grasp strategies towards picking any item in a real grocery store. Additionally, we provide an in-depth analysis of our latest real-world field test, discussing key findings related to fundamental failure modes over hundreds of distinct pick attempts. Through our detailed analysis, we aim to offer valuable practical insights and identify key grasping challenges, which can guide the robotics community towards pressing open problems in the field. 

**Abstract (ZH)**: 移动 manipulation 机器人不断进步，其抓取能力迅速提升。然而，最先进的移动 manipulator 仍存在显著差距，限制了其在实际环境中的广泛应用，尤其是在不规则环境下的可靠抓取能力。为弥合这一差距，我们开发了 SHOPPER，一个旨在推动可靠且通用抓取策略边界的移动 manipulation 机器人平台。我们发展了这些抓取策略，并在真实的杂货店环境中部署——这是一个极具挑战性的环境，因其广泛的可操作物品、设备和布局多样性而被精心选择。在这项工作中，我们详细介绍了设计通用抓取策略的方法，以实现在真实杂货店中捡拾任何物品。此外，我们提供了我们最新实地测试的深入分析，讨论了数百次独立捡拾尝试中基本失败模式的关键发现。通过我们的详细分析，我们旨在提供有价值的实践洞察，并识别关键抓取挑战，从而指导机器人社区解决该领域迫切需要解决的开放问题。 

---
# Learning Transferable Friction Models and LuGre Identification via Physics Informed Neural Networks 

**Title (ZH)**: 基于物理约束神经网络的学习转移摩擦模型及LuGre识别 

**Authors**: Asutay Ozmen, João P. Hespanha, Katie Byl  

**Link**: [PDF](https://arxiv.org/pdf/2504.12441)  

**Abstract**: Accurately modeling friction in robotics remains a core challenge, as robotics simulators like Mujoco and PyBullet use simplified friction models or heuristics to balance computational efficiency with accuracy, where these simplifications and approximations can lead to substantial differences between simulated and physical performance. In this paper, we present a physics-informed friction estimation framework that enables the integration of well-established friction models with learnable components-requiring only minimal, generic measurement data. Our approach enforces physical consistency yet retains the flexibility to adapt to real-world complexities. We demonstrate, on an underactuated and nonlinear system, that the learned friction models, trained solely on small and noisy datasets, accurately simulate dynamic friction properties and reduce the sim-to-real gap. Crucially, we show that our approach enables the learned models to be transferable to systems they are not trained on. This ability to generalize across multiple systems streamlines friction modeling for complex, underactuated tasks, offering a scalable and interpretable path toward bridging the sim-to-real gap in robotics and control. 

**Abstract (ZH)**: 准确建模机器人中的摩擦仍然是一个核心挑战，因为像Mujoco和PyBullet这样的机器人模拟器使用简化或启发式的摩擦模型来平衡计算效率与准确性，这些简化和近似可能导致模拟和物理性能之间的巨大差异。在本文中，我们提出了一种物理信息摩擦估计框架，该框架能够将成熟的摩擦模型与可学习组件集成起来，只需使用少量的通用测量数据。我们的方法保证了物理一致性，同时保持了适应现实世界复杂性的灵活性。我们在一个欠驱动且非线性系统上展示了所学的摩擦模型，仅在少量且噪声数据集上进行训练，能够准确模拟动态摩擦性能并减少模拟到现实的差距。关键的是，我们展示了我们的方法使得所学模型能够迁移到未经过训练的系统上。这种跨多个系统泛化的能力简化了复杂欠驱动任务中的摩擦建模，提供了一种可扩展且可解释的途径，以缩小机器人和控制中的模拟到现实差距。 

---
# Learning-based Delay Compensation for Enhanced Control of Assistive Soft Robots 

**Title (ZH)**: 基于学习的延迟补偿以增强辅助软机器人控制 

**Authors**: Adrià Mompó Alepuz, Dimitrios Papageorgiou, Silvia Tolu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12428)  

**Abstract**: Soft robots are increasingly used in healthcare, especially for assistive care, due to their inherent safety and adaptability. Controlling soft robots is challenging due to their nonlinear dynamics and the presence of time delays, especially in applications like a soft robotic arm for patient care. This paper presents a learning-based approach to approximate the nonlinear state predictor (Smith Predictor), aiming to improve tracking performance in a two-module soft robot arm with a short inherent input delay. The method uses Kernel Recursive Least Squares Tracker (KRLST) for online learning of the system dynamics and a Legendre Delay Network (LDN) to compress past input history for efficient delay compensation. Experimental results demonstrate significant improvement in tracking performance compared to a baseline model-based non-linear controller. Statistical analysis confirms the significance of the improvements. The method is computationally efficient and adaptable online, making it suitable for real-world scenarios and highlighting its potential for enabling safer and more accurate control of soft robots in assistive care applications. 

**Abstract (ZH)**: 软机器人在医疗领域的应用日益增多，尤其是在辅助护理方面，得益于其固有的安全性和适应性。控制软机器人颇具挑战性，主要由于其非线性动力学特性和存在的时延，尤其是在患者护理应用中，如软机器人臂。本文提出了一种基于学习的方法，以近似非线性状态预测器（Smith预测器），旨在提高具有较短固有时延的双模块软机器人臂的跟踪性能。该方法使用核递归最小二乘追踪器（KRLST）进行在线学习系统动力学，并使用勒让德延迟网络（LDN）压缩过去输入历史以实现高效的时延补偿。实验结果表明，与基于模型的非线性控制器基线相比，跟踪性能有了显著改善。统计分析证实了改进的显著性。该方法计算效率高且可在线适应，使其适用于现实场景，进一步凸显了其在辅助护理应用中实现更安全和更准确的软机器人控制的潜力。 

---
# Diffusion Based Robust LiDAR Place Recognition 

**Title (ZH)**: 基于扩散的鲁棒激光雷达地点识别 

**Authors**: Benjamin Krummenacher, Jonas Frey, Turcan Tuna, Olga Vysotska, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2504.12412)  

**Abstract**: Mobile robots on construction sites require accurate pose estimation to perform autonomous surveying and inspection missions. Localization in construction sites is a particularly challenging problem due to the presence of repetitive features such as flat plastered walls and perceptual aliasing due to apartments with similar layouts inter and intra floors. In this paper, we focus on the global re-positioning of a robot with respect to an accurate scanned mesh of the building solely using LiDAR data. In our approach, a neural network is trained on synthetic LiDAR point clouds generated by simulating a LiDAR in an accurate real-life large-scale mesh. We train a diffusion model with a PointNet++ backbone, which allows us to model multiple position candidates from a single LiDAR point cloud. The resulting model can successfully predict the global position of LiDAR in confined and complex sites despite the adverse effects of perceptual aliasing. The learned distribution of potential global positions can provide multi-modal position distribution. We evaluate our approach across five real-world datasets and show the place recognition accuracy of 77% +/-2m on average while outperforming baselines at a factor of 2 in mean error. 

**Abstract (ZH)**: 基于LiDAR数据的建筑工地机器人全局定位方法 

---
# AUTONAV: A Toolfor Autonomous Navigation of Robots 

**Title (ZH)**: AUTONAV：一种自主导航机器人工具 

**Authors**: Mir Md Sajid Sarwar, Sudip Samanta, Rajarshi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2504.12318)  

**Abstract**: We present a tool AUTONAV that automates the mapping, localization, and path-planning tasks for autonomous navigation of robots. The modular architecture allows easy integration of various algorithms for these tasks for comparison. We present the generated maps and path-plans by AUTONAV in indoor simulation scenarios. 

**Abstract (ZH)**: 我们介绍了一种工具AUTONAV，用于自动化机器人自主导航的映射、定位和路径规划任务。模块化的架构允许这些任务的各种算法的易用集成以便于比较。我们在室内模拟场景中展示了AUTONAV生成的地图和路径规划结果。 

---
# Uncertainty-Aware Trajectory Prediction via Rule-Regularized Heteroscedastic Deep Classification 

**Title (ZH)**: 基于规则正则化异方差深度分类的不确定性意识轨迹预测 

**Authors**: Kumar Manas, Christian Schlauch, Adrian Paschke, Christian Wirth, Nadja Klein  

**Link**: [PDF](https://arxiv.org/pdf/2504.13111)  

**Abstract**: Deep learning-based trajectory prediction models have demonstrated promising capabilities in capturing complex interactions. However, their out-of-distribution generalization remains a significant challenge, particularly due to unbalanced data and a lack of enough data and diversity to ensure robustness and calibration. To address this, we propose SHIFT (Spectral Heteroscedastic Informed Forecasting for Trajectories), a novel framework that uniquely combines well-calibrated uncertainty modeling with informative priors derived through automated rule extraction. SHIFT reformulates trajectory prediction as a classification task and employs heteroscedastic spectral-normalized Gaussian processes to effectively disentangle epistemic and aleatoric uncertainties. We learn informative priors from training labels, which are automatically generated from natural language driving rules, such as stop rules and drivability constraints, using a retrieval-augmented generation framework powered by a large language model. Extensive evaluations over the nuScenes dataset, including challenging low-data and cross-location scenarios, demonstrate that SHIFT outperforms state-of-the-art methods, achieving substantial gains in uncertainty calibration and displacement metrics. In particular, our model excels in complex scenarios, such as intersections, where uncertainty is inherently higher. Project page: this https URL. 

**Abstract (ZH)**: 基于深度学习的轨迹预测模型在捕捉复杂交互方面展现了 promising 能力，但它们的 out-of-distribution 通用性仍然是一个重大挑战，尤其是由于数据不平衡和缺乏足够的数据和多样性来确保 robustness 和 calibration。为了解决这个问题，我们提出了一种名为 SHIFT（Spectral Heteroscedastic Informed Forecasting for Trajectories）的新颖框架，该框架独特地结合了校准良好的不确定性建模和通过自动规则提取获得的信息先验。SHIFT 将轨迹预测重新定义为分类任务，并采用异方差谱规范化高斯过程来有效分离 epistemic 和 aleatoric 不确定性。我们通过一种使用大型语言模型驱动的检索增强生成框架从训练标签中学习信息先验，这些标签是从自然语言驾驶规则（如停止规则和可行驶性约束）自动生成的。在 nuScenes 数据集上的广泛评估，包括具有挑战性的低数据和跨地点场景，表明 SHIFT 在不确定性校准和位移指标方面优于最先进的方法，特别是在复杂的交叉路口等场景中表现出色。项目页面: this https URL。 

---
# Adaptive Task Space Non-Singular Terminal Super-Twisting Sliding Mode Control of a 7-DOF Robotic Manipulator 

**Title (ZH)**: 7-DOF机器人 manipulator 适应性任务空间非奇异终端超-twisting 滑模控制 

**Authors**: L. Wan, S. Smith, Y.-J. Pan, E. Witrant  

**Link**: [PDF](https://arxiv.org/pdf/2504.13056)  

**Abstract**: This paper presents a new task-space Non-singular Terminal Super-Twisting Sliding Mode (NT-STSM) controller with adaptive gains for robust trajectory tracking of a 7-DOF robotic manipulator. The proposed approach addresses the challenges of chattering, unknown disturbances, and rotational motion tracking, making it suited for high-DOF manipulators in dexterous manipulation tasks. A rigorous boundedness proof is provided, offering gain selection guidelines for practical implementation. Simulations and hardware experiments with external disturbances demonstrate the proposed controller's robust, accurate tracking with reduced control effort under unknown disturbances compared to other NT-STSM and conventional controllers. The results demonstrated that the proposed NT-STSM controller mitigates chattering and instability in complex motions, making it a viable solution for dexterous robotic manipulations and various industrial applications. 

**Abstract (ZH)**: 一种适用于7-DOF机器人 manipulator稳健轨迹跟踪的自适应增益非奇异终端超扭转滑模控制器 

---
# 3D-PNAS: 3D Industrial Surface Anomaly Synthesis with Perlin Noise 

**Title (ZH)**: 3D-PNAS：基于Perlin噪声的3D工业表面异常合成 

**Authors**: Yifeng Cheng, Juan Du  

**Link**: [PDF](https://arxiv.org/pdf/2504.12856)  

**Abstract**: Large pretrained vision foundation models have shown significant potential in various vision tasks. However, for industrial anomaly detection, the scarcity of real defect samples poses a critical challenge in leveraging these models. While 2D anomaly generation has significantly advanced with established generative models, the adoption of 3D sensors in industrial manufacturing has made leveraging 3D data for surface quality inspection an emerging trend. In contrast to 2D techniques, 3D anomaly generation remains largely unexplored, limiting the potential of 3D data in industrial quality inspection. To address this gap, we propose a novel yet simple 3D anomaly generation method, 3D-PNAS, based on Perlin noise and surface parameterization. Our method generates realistic 3D surface anomalies by projecting the point cloud onto a 2D plane, sampling multi-scale noise values from a Perlin noise field, and perturbing the point cloud along its normal direction. Through comprehensive visualization experiments, we demonstrate how key parameters - including noise scale, perturbation strength, and octaves, provide fine-grained control over the generated anomalies, enabling the creation of diverse defect patterns from pronounced deformations to subtle surface variations. Additionally, our cross-category experiments show that the method produces consistent yet geometrically plausible anomalies across different object types, adapting to their specific surface characteristics. We also provide a comprehensive codebase and visualization toolkit to facilitate future research. 

**Abstract (ZH)**: 基于皮恩林噪声和曲面参数化的3D异常生成方法3D-PNAS：实现工业表面质量检测中的3D数据利用 

---
# Acoustic Analysis of Uneven Blade Spacing and Toroidal Geometry for Reducing Propeller Annoyance 

**Title (ZH)**: 非均匀叶片间距和环形几何结构的 acoustic 分析以减少推进器恼人程度 

**Authors**: Nikhil Vijay, Will C. Forte, Ishan Gajjar, Sarvesh Patham, Syon Gupta, Sahil Shah, Prathamesh Trivedi, Rishit Arora  

**Link**: [PDF](https://arxiv.org/pdf/2504.12554)  

**Abstract**: Unmanned aerial vehicles (UAVs) are becoming more commonly used in populated areas, raising concerns about noise pollution generated from their propellers. This study investigates the acoustic performance of unconventional propeller designs, specifically toroidal and uneven-blade spaced propellers, for their potential in reducing psychoacoustic annoyance. Our experimental results show that these designs noticeably reduced acoustic characteristics associated with noise annoyance. 

**Abstract (ZH)**: 无人驾驶飞行器（UAV）在人口密集地区的应用日益增多，引起了对其推进器产生的噪声污染的关注。本研究探讨了非传统推进器设计，特别是环形和不等距叶片推进器的声学性能，评估其在降低心理声学烦恼方面的工作潜力。实验结果表明，这些设计显著降低了与噪声烦恼相关的声学特性。 

---
# UniPhys: Unified Planner and Controller with Diffusion for Flexible Physics-Based Character Control 

**Title (ZH)**: UniPhys：基于扩散的统一规划与控制器，实现灵活的物理_basis角色控制 

**Authors**: Yan Wu, Korrawe Karunratanakul, Zhengyi Luo, Siyu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12540)  

**Abstract**: Generating natural and physically plausible character motion remains challenging, particularly for long-horizon control with diverse guidance signals. While prior work combines high-level diffusion-based motion planners with low-level physics controllers, these systems suffer from domain gaps that degrade motion quality and require task-specific fine-tuning. To tackle this problem, we introduce UniPhys, a diffusion-based behavior cloning framework that unifies motion planning and control into a single model. UniPhys enables flexible, expressive character motion conditioned on multi-modal inputs such as text, trajectories, and goals. To address accumulated prediction errors over long sequences, UniPhys is trained with the Diffusion Forcing paradigm, learning to denoise noisy motion histories and handle discrepancies introduced by the physics simulator. This design allows UniPhys to robustly generate physically plausible, long-horizon motions. Through guided sampling, UniPhys generalizes to a wide range of control signals, including unseen ones, without requiring task-specific fine-tuning. Experiments show that UniPhys outperforms prior methods in motion naturalness, generalization, and robustness across diverse control tasks. 

**Abstract (ZH)**: 基于扩散的行为克隆框架UniPhys：统一运动规划与控制生成自然且物理合理的角色运动仍然具有挑战性，尤其是在具有多样指导信号的长期控制中。尽管现有工作结合了高层的基于扩散的运动规划器与低层的物理控制器，但这些系统存在领域差距，这会降低运动质量并需要特定任务的微调。为解决这一问题，我们引入了UniPhys，这是一种基于扩散的行为克隆框架，将运动规划与控制统一到一个模型中。UniPhys能够根据多模态输入（如文本、轨迹和目标）生成灵活且具表达性的角色运动。为了解决长时间序列中累积的预测误差，UniPhys采用扩散强迫范式进行训练，学习去除噪声的运动历史并处理物理模拟器引入的不一致性。这种设计使UniPhys能够稳健地生成物理合理的长期运动。通过引导采样，UniPhys能够在无需特定任务微调的情况下泛化到广泛的控制信号，包括未见过的信号。实验结果表明，UniPhys在运动自然性、泛化能力和跨多种控制任务的鲁棒性方面优于之前的方法。 

---
# Robust Visual Servoing under Human Supervision for Assembly Tasks 

**Title (ZH)**: 在人类监督下的鲁棒视觉伺服技术用于装配任务 

**Authors**: Victor Nan Fernandez-Ayala, Jorge Silva, Meng Guo, Dimos V. Dimarogonas  

**Link**: [PDF](https://arxiv.org/pdf/2504.12506)  

**Abstract**: We propose a framework enabling mobile manipulators to reliably complete pick-and-place tasks for assembling structures from construction blocks. The picking uses an eye-in-hand visual servoing controller for object tracking with Control Barrier Functions (CBFs) to ensure fiducial markers in the blocks remain visible. An additional robot with an eye-to-hand setup ensures precise placement, critical for structural stability. We integrate human-in-the-loop capabilities for flexibility and fault correction and analyze robustness to camera pose errors, proposing adapted barrier functions to handle them. Lastly, experiments validate the framework on 6-DoF mobile arms. 

**Abstract (ZH)**: 我们提出了一种框架，使移动 manipulator 能够可靠地完成基于构造块组装结构的取放任务。取件采用眼手视觉伺服控制器，并使用控制障碍函数（CBFs）确保构造块上的特征标记保持可见。另外一台具有眼手配置的机器人确保精确放置，这对于结构稳定性至关重要。我们整合了人为环路功能以提高灵活性和故障纠正能力，并分析了对于相机姿态误差的鲁棒性，提出了相应的障碍函数来处理这些问题。最后，实验在6-DoF移动臂上验证了该框架。 

---
