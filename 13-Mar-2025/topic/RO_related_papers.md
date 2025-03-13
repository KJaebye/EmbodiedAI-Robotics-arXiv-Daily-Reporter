# Action-Aware Pro-Active Safe Exploration for Mobile Robot Mapping 

**Title (ZH)**: 基于动作意识的主动安全探索方法在移动机器人建图中的应用 

**Authors**: Aykut İşleyen, René van de Molengraft, Ömür Arslan  

**Link**: [PDF](https://arxiv.org/pdf/2503.09515)  

**Abstract**: Safe autonomous exploration of unknown environments is an essential skill for mobile robots to effectively and adaptively perform environmental mapping for diverse critical tasks. Due to its simplicity, most existing exploration methods rely on the standard frontier-based exploration strategy, which directs a robot to the boundary between the known safe and the unknown unexplored spaces to acquire new information about the environment. This typically follows a recurrent persistent planning strategy, first selecting an informative frontier viewpoint, then moving the robot toward the selected viewpoint until reaching it, and repeating these steps until termination. However, exploration with persistent planning may lack adaptivity to continuously updated maps, whereas highly adaptive exploration with online planning often suffers from high computational costs and potential issues with livelocks. In this paper, as an alternative to less-adaptive persistent planning and costly online planning, we introduce a new proactive preventive replanning strategy for effective exploration using the immediately available actionable information at a viewpoint to avoid redundant, uninformative last-mile exploration motion. We also use the actionable information of a viewpoint as a systematic termination criterion for exploration. To close the gap between perception and action, we perform safe and informative path planning that minimizes the risk of collision with detected obstacles and the distance to unexplored regions, and we apply action-aware viewpoint selection with maximal information utility per total navigation cost. We demonstrate the effectiveness of our action-aware proactive exploration method in numerical simulations and hardware experiments. 

**Abstract (ZH)**: 安全自主探索未知环境是移动机器人有效适应性执行环境建图以完成多样关键任务的一项基本技能。大多数现有探索方法依赖于标准的前沿基探索策略，该策略指导机器人前往已知安全区域与未知未探索区域的边界，以获取有关环境的新信息。这种探索通常遵循循环坚持规划策略，首先选择一个有信息价值的前沿视角，然后将机器人移动到选定的视角，直到到达，然后重复这些步骤直到终止。然而，坚持规划的探索缺乏对持续更新地图的适应性，而在线规划的高适应性探索往往面临较高的计算成本和潜在死锁问题。在本文中，作为少适应性坚持规划和高成本在线规划的替代方案，我们引入了一种基于即时可用可操作信息的主动预防性重规划策略，以有效探索并避免冗余、无信息的最后阶段探索运动。我们还使用视点的可操作信息作为探索的系统终止标准。为了弥合感知与行动之间的差距，我们执行了安全且信息丰富的路径规划，以最小化与检测到的障碍物碰撞的风险和到未探索区域的距离，并应用了具有最大信息效用的感知行动视点选择，以最小化总导航成本。我们通过数值仿真和硬件实验展示了我们感知行动的主动探索方法的有效性。 

---
# Neural-Augmented Incremental Nonlinear Dynamic Inversion for Quadrotors with Payload Adaptation 

**Title (ZH)**: 基于神经增强增量非线性动态反转的载荷自适应四旋翼控制 

**Authors**: Eckart Cobo-Briesewitz, Khaled Wahba, Wolfgang Hönig  

**Link**: [PDF](https://arxiv.org/pdf/2503.09441)  

**Abstract**: The increasing complexity of multirotor applications has led to the need of more accurate flight controllers that can reliably predict all forces acting on the robot. Traditional flight controllers model a large part of the forces but do not take so called residual forces into account. A reason for this is that accurately computing the residual forces can be computationally expensive. Incremental Nonlinear Dynamic Inversion (INDI) is a method that computes the difference between different sensor measurements in order to estimate these residual forces. The main issue with INDI is it's reliance on special sensor measurements which can be very noisy. Recent work has also shown that residual forces can be predicted using learning-based methods. In this work, we demonstrate that a learning algorithm can predict a smoother version of INDI outputs without requiring additional sensor measurements. In addition, we introduce a new method that combines learning based predictions with INDI. We also adapt the two approaches to work on quadrotors carrying a slung-type payload. The results show that using a neural network to predict residual forces can outperform INDI while using the combination of neural network and INDI can yield even better results than each method individually. 

**Abstract (ZH)**: 多旋翼应用复杂性的增加促使需要更准确的飞行控制器以可靠地预测作用于机器人上的所有力。传统的飞行控制器建模了大部分力，但没有考虑所谓的残余力。这主要是因为准确计算残余力可能是计算上昂贵的。增量非线性动态逆（INDI）是一种方法，通过计算不同传感器测量值之间的差异来估计这些残余力。INDI的主要问题是依赖于特殊的传感器测量，这些测量可能会非常嘈杂。最近的研究还表明，可以使用基于学习的方法来预测残余力。在这项工作中，我们证明了一种学习算法可以在不需要额外传感器测量的情况下预测INDI输出的平滑版本。此外，我们提出了一种新方法，将基于学习的预测与INDI相结合。我们还将两种方法适应于悬挂载荷的四旋翼。结果显示，使用神经网络预测残余力可以优于INDI，而将神经网络与INDI结合使用则可以比各自的方法获得更好的结果。 

---
# AI-based Framework for Robust Model-Based Connector Mating in Robotic Wire Harness Installation 

**Title (ZH)**: 基于AI的鲁棒模型导向连接器对接框架在机器人线束安装中的应用 

**Authors**: Claudius Kienle, Benjamin Alt, Finn Schneider, Tobias Pertlwieser, Rainer Jäkel, Rania Rayyes  

**Link**: [PDF](https://arxiv.org/pdf/2503.09409)  

**Abstract**: Despite the widespread adoption of industrial robots in automotive assembly, wire harness installation remains a largely manual process, as it requires precise and flexible manipulation. To address this challenge, we design a novel AI-based framework that automates cable connector mating by integrating force control with deep visuotactile learning. Our system optimizes search-and-insertion strategies using first-order optimization over a multimodal transformer architecture trained on visual, tactile, and proprioceptive data. Additionally, we design a novel automated data collection and optimization pipeline that minimizes the need for machine learning expertise. The framework optimizes robot programs that run natively on standard industrial controllers, permitting human experts to audit and certify them. Experimental validations on a center console assembly task demonstrate significant improvements in cycle times and robustness compared to conventional robot programming approaches. Videos are available under this https URL. 

**Abstract (ZH)**: 尽管工业机器人在汽车装配中得到了广泛应用，线束安装过程仍主要依赖手动操作，因为它需要精确且灵活的操作。为应对这一挑战，我们设计了一种基于AI的新框架，通过将力控制与深度跨模态学习相结合，自动实现电缆接头对接。该系统使用多模态变换器架构对视觉、触觉和本体感受数据进行训练，并通过一阶优化方法优化搜索与插入策略。此外，我们还设计了一种新的自动化数据收集和优化管道，以减少对机器学习专业知识的需求。该框架优化的机器人程序可以在标准工业控制器上本地运行，使人类专家能够审核和认证这些程序。在中央控制台装配任务上的实验验证表明，与传统的机器人编程方法相比，该框架在循环时间和鲁棒性方面取得了显著改进。更多视频请访问此链接。 

---
# Robust Self-Reconfiguration for Fault-Tolerant Control of Modular Aerial Robot Systems 

**Title (ZH)**: 模块化空中机器人系统容错控制的鲁棒自重构方法 

**Authors**: Rui Huang, Siyu Tang, Zhiqian Cai, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.09376)  

**Abstract**: Modular Aerial Robotic Systems (MARS) consist of multiple drone units assembled into a single, integrated rigid flying platform. With inherent redundancy, MARS can self-reconfigure into different configurations to mitigate rotor or unit failures and maintain stable flight. However, existing works on MARS self-reconfiguration often overlook the practical controllability of intermediate structures formed during the reassembly process, which limits their applicability. In this paper, we address this gap by considering the control-constrained dynamic model of MARS and proposing a robust and efficient self-reconstruction algorithm that maximizes the controllability margin at each intermediate stage. Specifically, we develop algorithms to compute optimal, controllable disassembly and assembly sequences, enabling robust self-reconfiguration. Finally, we validate our method in several challenging fault-tolerant self-reconfiguration scenarios, demonstrating significant improvements in both controllability and trajectory tracking while reducing the number of assembly steps. The videos and source code of this work are available at this https URL 

**Abstract (ZH)**: 模块化 aerial 机器人系统（MARS）由多个无人机单元组装成一个单一的集成刚性飞行平台。具有固有的冗余性，MARS 可以自我重构为不同的配置以减轻旋翼或单元故障并保持稳定的飞行。然而，现有的 MARS 自我重构工作往往忽略了重组过程中形成的中间结构的可操作性，这限制了其应用范围。在本文中，我们通过考虑 MARS 的控制约束动态模型，并提出一种 robust 和高效的自我重构算法来填补这一空白，该算法在每个中间阶段最大化可操作性裕度。具体而言，我们开发了算法来计算最优且可操作的拆卸和组装序列，从而实现 robust 自我重构。最后，我们在几个具有挑战性的容错自我重构场景中验证了我们的方法，证明了在可操作性和轨迹跟踪方面有显著改进，同时减少了组装步骤的数量。本工作的视频和源代码可从以下网址获取。 

---
# Robust Fault-Tolerant Control and Agile Trajectory Planning for Modular Aerial Robotic Systems 

**Title (ZH)**: 模块化空中机器人系统的鲁棒容错控制与敏捷轨迹规划 

**Authors**: Rui Huang, Zhenyu Zhang, Siyu Tang, Zhiqian Cai, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.09351)  

**Abstract**: Modular Aerial Robotic Systems (MARS) consist of multiple drone units that can self-reconfigure to adapt to various mission requirements and fault conditions. However, existing fault-tolerant control methods exhibit significant oscillations during docking and separation, impacting system stability. To address this issue, we propose a novel fault-tolerant control reallocation method that adapts to arbitrary number of modular robots and their assembly formations. The algorithm redistributes the expected collective force and torque required for MARS to individual unit according to their moment arm relative to the center of MARS mass. Furthermore, We propose an agile trajectory planning method for MARS of arbitrary configurations, which is collision-avoiding and dynamically feasible. Our work represents the first comprehensive approach to enable fault-tolerant and collision avoidance flight for MARS. We validate our method through extensive simulations, demonstrating improved fault tolerance, enhanced trajectory tracking accuracy, and greater robustness in cluttered environments. The videos and source code of this work are available at this https URL 

**Abstract (ZH)**: 模块化空中机器人系统（MARS）由多个能够自重构以适应各种任务要求和故障状态的无人机单元组成。然而，现有的容错控制方法在对接和分离过程中表现出显著的振荡，影响系统稳定性。为解决这一问题，我们提出了一种新的容错控制重新分配方法，该方法适用于任意数量的模块化机器人及其组装形态。该算法根据各单元相对于MARS质心的力臂重新分配MARS所需的预期集体力和力矩。此外，我们还提出了一种适用于任意配置的MARS的敏捷轨迹规划方法，该方法具有避碰能力和动态可行性。我们的工作代表了首次全面的方法，以实现MARS的容错和碰撞避让飞行。我们通过广泛的仿真验证了该方法，展示了改进的容错性能、提高的轨迹跟踪精度以及在复杂环境中的更大鲁棒性。该工作的视频和源代码可在以下链接获取：this https URL 

---
# Rethinking Bimanual Robotic Manipulation: Learning with Decoupled Interaction Framework 

**Title (ZH)**: 重新思考双臂机器人操作：基于解耦互动框架的学习 

**Authors**: Jian-Jian Jiang, Xiao-Ming Wu, Yi-Xiang He, Ling-An Zeng, Yi-Lin Wei, Dandan Zhang, Wei-Shi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.09186)  

**Abstract**: Bimanual robotic manipulation is an emerging and critical topic in the robotics community. Previous works primarily rely on integrated control models that take the perceptions and states of both arms as inputs to directly predict their actions. However, we think bimanual manipulation involves not only coordinated tasks but also various uncoordinated tasks that do not require explicit cooperation during execution, such as grasping objects with the closest hand, which integrated control frameworks ignore to consider due to their enforced cooperation in the early inputs. In this paper, we propose a novel decoupled interaction framework that considers the characteristics of different tasks in bimanual manipulation. The key insight of our framework is to assign an independent model to each arm to enhance the learning of uncoordinated tasks, while introducing a selective interaction module that adaptively learns weights from its own arm to improve the learning of coordinated tasks. Extensive experiments on seven tasks in the RoboTwin dataset demonstrate that: (1) Our framework achieves outstanding performance, with a 23.5% boost over the SOTA method. (2) Our framework is flexible and can be seamlessly integrated into existing methods. (3) Our framework can be effectively extended to multi-agent manipulation tasks, achieving a 28% boost over the integrated control SOTA. (4) The performance boost stems from the decoupled design itself, surpassing the SOTA by 16.5% in success rate with only 1/6 of the model size. 

**Abstract (ZH)**: 双臂机器人操作是机器人领域的一个新兴且关键课题。 

---
# Predictor-Based Time Delay Control of A Hex-Jet Unmanned Aerial Vehicle 

**Title (ZH)**: 基于预测的时间延迟控制六旋翼无人机 

**Authors**: Junning Liang, Haowen Zheng, Yuying Zhang, Yongzhuo Gao, Wei Dong, Ximin Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2503.09148)  

**Abstract**: Turbojet-powered VTOL UAVs have garnered increased attention in heavy-load transport and emergency services, due to their superior power density and thrust-to-weight ratio compared to existing electronic propulsion systems. The main challenge with jet-powered UAVs lies in the complexity of thrust vectoring mechanical systems, which aim to mitigate the slow dynamics of the turbojet. In this letter, we introduce a novel turbojet-powered UAV platform named Hex-Jet. Our concept integrates thrust vectoring and differential thrust for comprehensive attitude control. This approach notably simplifies the thrust vectoring mechanism. We utilize a predictor-based time delay control method based on the frequency domain model in our Hex-Jet controller design to mitigate the delay in roll attitude control caused by turbojet dynamics. Our comparative studies provide valuable insights for the UAV community, and flight tests on the scaled prototype demonstrate the successful implementation and verification of the proposed predictor-based time delay control technique. 

**Abstract (ZH)**: 基于涡喷发动机的垂直起降无人机在重型负载运输和紧急服务领域引起了广泛关注，由于其比现有电动推进系统具有更高的功率密度和推重比。喷气动力无人机的主要挑战在于推进矢量机机械系统的复杂性，旨在缓解涡喷发动机的缓慢动态特性。本文介绍了一种新型涡喷发动机动力垂直起降无人机平台Hex-Jet，该平台将推进矢量控制与差动推力集成，全面实现姿态控制。该方法显著简化了推进矢量控制机制。我们使用基于频域模型的预测式时间延迟控制方法，在Hex-Jet控制器设计中减轻了由涡喷发动机动态特性引起的滚转姿态控制的延迟。我们的比较研究为无人机社区提供了有价值的见解，并对缩小比例原型机的飞行测试证明了所提出的预测式时间延迟控制技术的成功实施和验证。 

---
# Tacchi 2.0: A Low Computational Cost and Comprehensive Dynamic Contact Simulator for Vision-based Tactile Sensors 

**Title (ZH)**: Tacchi 2.0：一种低计算成本和综合动态接触模拟器，用于基于视觉的触觉传感器 

**Authors**: Yuhao Sun, Shixin Zhang, Wenzhuang Li, Jie Zhao, Jianhua Shan, Zirong Shen, Zixi Chen, Fuchun Sun, Di Guo, Bin Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.09100)  

**Abstract**: With the development of robotics technology, some tactile sensors, such as vision-based sensors, have been applied to contact-rich robotics tasks. However, the durability of vision-based tactile sensors significantly increases the cost of tactile information acquisition. Utilizing simulation to generate tactile data has emerged as a reliable approach to address this issue. While data-driven methods for tactile data generation lack robustness, finite element methods (FEM) based approaches require significant computational costs. To address these issues, we integrated a pinhole camera model into the low computational cost vision-based tactile simulator Tacchi that used the Material Point Method (MPM) as the simulated method, completing the simulation of marker motion images. We upgraded Tacchi and introduced Tacchi 2.0. This simulator can simulate tactile images, marked motion images, and joint images under different motion states like pressing, slipping, and rotating. Experimental results demonstrate the reliability of our method and its robustness across various vision-based tactile sensors. 

**Abstract (ZH)**: 随着机器人技术的发展，一些触觉传感器，如基于视觉的传感器，已被应用于接触密集型机器人任务。然而，基于视觉的触觉传感器的耐用性显著增加了触觉信息获取的成本。利用仿真生成触觉数据已成为解决这一问题的可靠方法。虽然基于数据驱动的方法在触觉数据生成中缺乏稳定性，但基于有限元方法（FEM）的方法需要巨大的计算成本。为了解决这些问题，我们将针孔相机模型集成到使用物质点方法（MPM）进行仿真的低计算成本视觉触觉模拟器Tacchi中，完成了标记运动图像的模拟。我们升级了Tacchi并推出了Tacchi 2.0。该模拟器可以在不同的运动状态下（如按压、滑动和旋转）模拟触觉图像、标记运动图像和关节图像。实验结果证明了我们方法的可靠性和在各种视觉触觉传感器上的稳定性。 

---
# Sequential Multi-Object Grasping with One Dexterous Hand 

**Title (ZH)**: 单灵巧手的序列多对象抓取 

**Authors**: Sicheng He, Zeyu Shangguan, Kuanning Wang, Yongchong Gu, Yuqian Fu, Yanwei Fu, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2503.09078)  

**Abstract**: Sequentially grasping multiple objects with multi-fingered hands is common in daily life, where humans can fully leverage the dexterity of their hands to enclose multiple objects. However, the diversity of object geometries and the complex contact interactions required for high-DOF hands to grasp one object while enclosing another make sequential multi-object grasping challenging for robots. In this paper, we propose SeqMultiGrasp, a system for sequentially grasping objects with a four-fingered Allegro Hand. We focus on sequentially grasping two objects, ensuring that the hand fully encloses one object before lifting it and then grasps the second object without dropping the first. Our system first synthesizes single-object grasp candidates, where each grasp is constrained to use only a subset of the hand's links. These grasps are then validated in a physics simulator to ensure stability and feasibility. Next, we merge the validated single-object grasp poses to construct multi-object grasp configurations. For real-world deployment, we train a diffusion model conditioned on point clouds to propose grasp poses, followed by a heuristic-based execution strategy. We test our system using $8 \times 8$ object combinations in simulation and $6 \times 3$ object combinations in real. Our diffusion-based grasp model obtains an average success rate of 65.8% over 1600 simulation trials and 56.7% over 90 real-world trials, suggesting that it is a promising approach for sequential multi-object grasping with multi-fingered hands. Supplementary material is available on our project website: this https URL. 

**Abstract (ZH)**: 基于四指 Allegro 手的序列多物 grasping 系统：SeqMultiGrasp 

---
# Feasibility-aware Imitation Learning from Observations through a Hand-mounted Demonstration Interface 

**Title (ZH)**: 基于手部穿戴示范接口的观察模仿学习的可行性意识化研究 

**Authors**: Kei Takahashi, Hikaru Sasaki, Takamitsu Matsubara  

**Link**: [PDF](https://arxiv.org/pdf/2503.09018)  

**Abstract**: Imitation learning through a demonstration interface is expected to learn policies for robot automation from intuitive human demonstrations. However, due to the differences in human and robot movement characteristics, a human expert might unintentionally demonstrate an action that the robot cannot execute. We propose feasibility-aware behavior cloning from observation (FABCO). In the FABCO framework, the feasibility of each demonstration is assessed using the robot's pre-trained forward and inverse dynamics models. This feasibility information is provided as visual feedback to the demonstrators, encouraging them to refine their demonstrations. During policy learning, estimated feasibility serves as a weight for the demonstration data, improving both the data efficiency and the robustness of the learned policy. We experimentally validated FABCO's effectiveness by applying it to a pipette insertion task involving a pipette and a vial. Four participants assessed the impact of the feasibility feedback and the weighted policy learning in FABCO. Additionally, we used the NASA Task Load Index (NASA-TLX) to evaluate the workload induced by demonstrations with visual feedback. 

**Abstract (ZH)**: 通过演示界面进行模仿学习有望从直观的人类演示中学习用于机器人自动化的策略。然而，由于人类和机器人运动特征的差异，人类专家可能会无意中演示机器人无法执行的动作。我们提出了基于观察的可行性感知行为克隆（FABCO）。在FABCO框架中，使用机器人预训练的正向和逆向动力学模型来评估每项演示的可行性。该可行性信息作为视觉反馈提供给演示者，鼓励他们改进演示。在策略学习过程中，估计的可行性作为演示数据的权重，提高了学习策略的数据效率和鲁棒性。我们通过将FABCO应用于涉及移液管和瓶的移液任务，实验验证了其有效性。四位参与者评估了可行性反馈和带有加权策略学习的FABCO的影响。此外，我们使用NASA任务负荷指数（NASA-TLX）评估了带有视觉反馈的演示引起的负荷。 

---
# TetraGrip: Sensor-Driven Multi-Suction Reactive Object Manipulation in Cluttered Scenes 

**Title (ZH)**: TetraGrip: 基于传感器的多吸附反应式物体 manipulation 在杂乱场景中的应用 

**Authors**: Paolo Torrado, Joshua Levin, Markus Grotz, Joshua Smith  

**Link**: [PDF](https://arxiv.org/pdf/2503.08978)  

**Abstract**: Warehouse robotic systems equipped with vacuum grippers must reliably grasp a diverse range of objects from densely packed shelves. However, these environments present significant challenges, including occlusions, diverse object orientations, stacked and obstructed items, and surfaces that are difficult to suction. We introduce \tetra, a novel vacuum-based grasping strategy featuring four suction cups mounted on linear actuators. Each actuator is equipped with an optical time-of-flight (ToF) proximity sensor, enabling reactive grasping.
We evaluate \tetra in a warehouse-style setting, demonstrating its ability to manipulate objects in stacked and obstructed configurations. Our results show that our RL-based policy improves picking success in stacked-object scenarios by 22.86\% compared to a single-suction gripper. Additionally, we demonstrate that TetraGrip can successfully grasp objects in scenarios where a single-suction gripper fails due to physical limitations, specifically in two cases: (1) picking an object occluded by another object and (2) retrieving an object in a complex scenario. These findings highlight the advantages of multi-actuated, suction-based grasping in unstructured warehouse environments. The project website is available at: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 装备有真空吸盘的仓库机器人系统必须可靠地从密集排列的货架上抓取多种多样的物体。然而，这些环境带来了显著的挑战，包括遮挡、多样的物体朝向、堆叠和阻挡的物品，以及难以吸盘吸附的表面。我们引入了\tetra，一种基于真空的新型抓取策略，采用线性执行器上安装的四个吸盘。每个执行器配备了一个光学飞行时间（ToF）距离传感器，使得抓取具有反应性。 

---
# Geometric Data-Driven Multi-Jet Locomotion Inspired by Salps 

**Title (ZH)**: 几何数据驱动的多喷流运动受沙虱启发 

**Authors**: Yanhao Yang, Nina L. Hecht, Yousef Salaman-Maclara, Nathan Justus, Zachary A. Thomas, Farhan Rozaidi, Ross L. Hatton  

**Link**: [PDF](https://arxiv.org/pdf/2503.08817)  

**Abstract**: Salps are marine animals consisting of chains of jellyfish-like units. Their capacity for effective underwater undulatory locomotion through coordinating multi-jet propulsion has aroused significant interest in the field of robotics and inspired extensive research including design, modeling, and control. In this paper, we conduct a comprehensive analysis of the locomotion of salp-like systems using the robotic platform "LandSalp" based on geometric mechanics, including mechanism design, dynamic modeling, system identification, and motion planning and control. Our work takes a step toward a better understanding of salps' underwater locomotion and provides a clear path for extending these insights to more complex and capable underwater robotic systems. Furthermore, this study illustrates the effectiveness of geometric mechanics in bio-inspired robots for efficient data-driven locomotion modeling, demonstrated by learning the dynamics of LandSalp from only 3 minutes of experimental data. Lastly, we extend the geometric mechanics principles to multi-jet propulsion systems with stability considerations and validate the theory through experiments on the LandSalp hardware. 

**Abstract (ZH)**: 珊懑是由一系列类似水母的单元组成的marine动物，它们通过协调多喷射推进实现有效的水下波浪式运动，这一特性在机器人学领域引起了广泛关注，并激发了大量关于设计、建模和控制的研究。在本文中，我们基于几何力学对“LandSalp”机器人平台上的类似珊懑系统的运动进行了全面分析，包括机构设计、动态建模、系统辨识以及运动规划与控制。我们的工作为进一步理解珊懑的水下运动提供了新的见解，并为将这些见解扩展到更加复杂和高性能的水下机器人系统指明了路径。此外，本研究展示了几何力学在生物启发机器人中实现高效数据驱动运动建模的有效性，仅通过3分钟的实验数据就学习到了LandSalp的动力学特性。最后，我们将几何力学原理扩展到具有稳定性考虑的多喷射推进系统，并通过在LandSalp硬件上的实验验证了理论。 

---
# Accurate Control under Voltage Drop for Rotor Drones 

**Title (ZH)**: 准确控制下的电压降对旋翼无人机的影响 

**Authors**: Yuhang Liu, Jindou Jia, Zihan Yang, Kexin Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.09017)  

**Abstract**: This letter proposes an anti-disturbance control scheme for rotor drones to counteract voltage drop (VD) disturbance caused by voltage drop of the battery, which is a common case for long-time flight or aggressive maneuvers. Firstly, the refined dynamics of rotor drones considering VD disturbance are presented. Based on the dynamics, a voltage drop observer (VDO) is developed to accurately estimate the VD disturbance by decoupling the disturbance and state information of the drone, reducing the conservativeness of conventional disturbance observers. Subsequently, the control scheme integrates the VDO within the translational loop and a fixed-time sliding mode observer (SMO) within the rotational loop, enabling it to address force and torque disturbances caused by voltage drop of the battery. Sufficient real flight experiments are conducted to demonstrate the effectiveness of the proposed control scheme under VD disturbance. 

**Abstract (ZH)**: 本论文提出了一种反干扰控制方案，用于抵消电池电压下降（VD）引起的电压下降干扰，该干扰常见于长时间飞行或激烈机动。首先，考虑VD干扰的旋翼无人机精细化动力学模型被呈现。基于该动力学模型，开发了一种电压下降观察器（VDO），通过解耦无人机的扰动和状态信息来精确估计VD干扰，从而减少传统扰动观察器的保守性。随后，控制方案将VDO集成到平移回路中，并将固定时间滑模观察器（SMO）集成到旋转回路中，使其能够处理由电池电压下降引起的力和力矩干扰。进行了充分的实飞实验以证明在VD干扰下所提控制方案的有效性。 

---
# Real-time simulation enabled navigation control of magnetic soft continuum robots in confined lumens 

**Title (ZH)**: 受限管道中基于实时模拟的磁软连续机器人导航控制 

**Authors**: Dezhong Tong, Zhuonan Hao, Jiyu Li, Boxi Sun, Mingchao Liu, Liu Wang, Weicheng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08864)  

**Abstract**: Magnetic soft continuum robots (MSCRs) have emerged as a promising technology for minimally invasive interventions, offering enhanced dexterity and remote-controlled navigation in confined lumens. Unlike conventional guidewires with pre-shaped tips, MSCRs feature a magnetic tip that actively bends under applied magnetic fields. Despite extensive studies in modeling and simulation, achieving real-time navigation control of MSCRs in confined lumens remains a significant challenge. The primary reasons are due to robot-lumen contact interactions and computational limitations in modeling MSCR nonlinear behavior under magnetic actuation. Existing approaches, such as Finite Element Method (FEM) simulations and energy-minimization techniques, suffer from high computational costs and oversimplified contact interactions, making them impractical for real-world applications. In this work, we develop a real-time simulation and navigation control framework that integrates hard-magnetic elastic rod theory, formulated within the Discrete Differential Geometry (DDG) framework, with an order-reduced contact handling strategy. Our approach captures large deformations and complex interactions while maintaining computational efficiency. Next, the navigation control problem is formulated as an inverse design task, where optimal magnetic fields are computed in real time by minimizing the constrained forces and enhancing navigation accuracy. We validate the proposed framework through comprehensive numerical simulations and experimental studies, demonstrating its robustness, efficiency, and accuracy. The results show that our method significantly reduces computational costs while maintaining high-fidelity modeling, making it feasible for real-time deployment in clinical settings. 

**Abstract (ZH)**: 基于磁性的软连续体机器人（MSCRs）已成为一种有前景的微创介入技术，能够在狭小的腔道中提供增强的操作灵活性和远程控制导航。与具有预成型尖端的常规导丝不同，MSCRs配备了一个在施加磁场下主动弯曲的磁性尖端。尽管在建模和仿真方面进行了大量研究，但在狭小腔道中实现MSCRs的实时导航控制仍然是一项重大挑战。主要原因在于机器人-腔道接触交互和在磁场驱动下建模MSCRs非线性行为的计算限制。现有的方法，如有限元方法（FEM）仿真和能量最小化技术，由于计算成本高和接触交互的简化，使其在实际应用中不可行。在本工作中，我们开发了一种实时仿真和导航控制框架，该框架结合了在离散微分几何（DDG）框架内制定的硬磁弹性杆理论，并采用了降阶接触处理策略。我们的方法能够捕捉到大面积变形和复杂交互，同时保持计算效率。然后，将导航控制问题形式化为逆设计任务，通过在实时计算约束力最小化和提升导航精度来确定最优磁场。我们通过全面的数值仿真和实验研究验证了所提出的框架，证明了其鲁棒性、效率和准确性。结果显示，我们的方法显著降低了计算成本，同时保持了高保真建模，使其在临床设置中实时部署成为可能。 

---
# Keypoint Semantic Integration for Improved Feature Matching in Outdoor Agricultural Environments 

**Title (ZH)**: 户外农业环境中超連結语义关键点集成以改进特征匹配 

**Authors**: Rajitha de Silva, Jonathan Cox, Marija Popovic, Cesar Cadena, Cyrill Stachniss, Riccardo Polvara  

**Link**: [PDF](https://arxiv.org/pdf/2503.08843)  

**Abstract**: Robust robot navigation in outdoor environments requires accurate perception systems capable of handling visual challenges such as repetitive structures and changing appearances. Visual feature matching is crucial to vision-based pipelines but remains particularly challenging in natural outdoor settings due to perceptual aliasing. We address this issue in vineyards, where repetitive vine trunks and other natural elements generate ambiguous descriptors that hinder reliable feature matching. We hypothesise that semantic information tied to keypoint positions can alleviate perceptual aliasing by enhancing keypoint descriptor distinctiveness. To this end, we introduce a keypoint semantic integration technique that improves the descriptors in semantically meaningful regions within the image, enabling more accurate differentiation even among visually similar local features. We validate this approach in two vineyard perception tasks: (i) relative pose estimation and (ii) visual localisation. Across all tested keypoint types and descriptors, our method improves matching accuracy by 12.6%, demonstrating its effectiveness over multiple months in challenging vineyard conditions. 

**Abstract (ZH)**: 户外环境中的鲁棒机器人导航需要能够处理重复结构和变化外观等视觉挑战的准确感知系统。基于视觉的特征匹配在自然户外环境中尤为关键，但由于感知混叠问题，仍然面临巨大挑战。在葡萄园中，重复的葡萄藤主干和其他自然元素会产生模糊的描述符，阻碍可靠的特征匹配。我们假设与关键点位置相关的语义信息能够通过增强描述符的独特性来缓解感知混叠。为此，我们提出了一种关键点语义集成技术，能够在图像中的语义有意义的区域改进描述符，即使在视觉上相似的局部特征之间也能实现更准确的区分。我们在两个葡萄园感知任务中验证了这种方法：（i）相对姿态估计和（ii）视觉定位。在所有测试的关键点类型和描述符中，我们的方法将匹配准确性提高12.6%，证明了其在挑战性葡萄园条件下的有效性。 

---
