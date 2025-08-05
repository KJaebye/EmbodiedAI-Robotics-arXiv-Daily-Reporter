# Manip4Care: Robotic Manipulation of Human Limbs for Solving Assistive Tasks 

**Title (ZH)**: Manip4Care: 人为肢体的机器人操作以解决辅助任务 

**Authors**: Yubin Koh, Ahmed H. Qureshi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02649)  

**Abstract**: Enabling robots to grasp and reposition human limbs can significantly enhance their ability to provide assistive care to individuals with severe mobility impairments, particularly in tasks such as robot-assisted bed bathing and dressing. However, existing assistive robotics solutions often assume that the human remains static or quasi-static, limiting their effectiveness. To address this issue, we present Manip4Care, a modular simulation pipeline that enables robotic manipulators to grasp and reposition human limbs effectively. Our approach features a physics simulator equipped with built-in techniques for grasping and repositioning while considering biomechanical and collision avoidance constraints. Our grasping method employs antipodal sampling with force closure to grasp limbs, and our repositioning system utilizes the Model Predictive Path Integral (MPPI) and vector-field-based control method to generate motion trajectories under collision avoidance and biomechanical constraints. We evaluate this approach across various limb manipulation tasks in both supine and sitting positions and compare outcomes for different age groups with differing shoulder joint limits. Additionally, we demonstrate our approach for limb manipulation using a real-world mannequin and further showcase its effectiveness in bed bathing tasks. 

**Abstract (ZH)**: 使机器人能够抓握和重新定位人类肢体可以显著增强它们为严重行动障碍个体提供辅助护理的能力，特别是在辅助沐浴和 dressing 等任务中。然而，现有的辅助机器人解决方案通常假设人类保持静止或准静止状态，限制了其有效性。为了解决这一问题，我们提出了 Manip4Care，一个模块化的仿真流水线，使机器人操作器能够有效抓握和重新定位人类肢体。我们的方法配备了一个具有抓握和重新定位内置技术的物理学仿真器，并考虑了生物力学和碰撞避免约束。我们的抓握方法采用了反握采样与力闭合技术来抓取肢体，而我们的重新定位系统则利用模型预测路径积分（MPPI）和基于向量场的控制方法，在避免碰撞和生物力学约束条件下生成运动轨迹。我们在仰卧和坐姿位置下的多种肢体操作任务中评估了这种方法，并对不同年龄组和不同的肩关节限制进行了比较。此外，我们使用实物人形模特展示了肢体操作的方法，并进一步展示了其在辅助沐浴任务中的有效性。 

---
# Vision-based Navigation of Unmanned Aerial Vehicles in Orchards: An Imitation Learning Approach 

**Title (ZH)**: 基于视觉的果园无人无人机导航：一种模仿学习方法 

**Authors**: Peng Wei, Prabhash Ragbir, Stavros G. Vougioukas, Zhaodan Kong  

**Link**: [PDF](https://arxiv.org/pdf/2508.02617)  

**Abstract**: Autonomous unmanned aerial vehicle (UAV) navigation in orchards presents significant challenges due to obstacles and GPS-deprived environments. In this work, we introduce a learning-based approach to achieve vision-based navigation of UAVs within orchard rows. Our method employs a variational autoencoder (VAE)-based controller, trained with an intervention-based learning framework that allows the UAV to learn a visuomotor policy from human experience. We validate our approach in real orchard environments with a custom-built quadrotor platform. Field experiments demonstrate that after only a few iterations of training, the proposed VAE-based controller can autonomously navigate the UAV based on a front-mounted camera stream. The controller exhibits strong obstacle avoidance performance, achieves longer flying distances with less human assistance, and outperforms existing algorithms. Furthermore, we show that the policy generalizes effectively to novel environments and maintains competitive performance across varying conditions and speeds. This research not only advances UAV autonomy but also holds significant potential for precision agriculture, improving efficiency in orchard monitoring and management. 

**Abstract (ZH)**: 基于学习的视觉导航在果园中自主无人机飞行面临显著挑战，因为空中障碍和GPS受限环境。本文提出了一种基于学习的方法，实现无人机在果园行间基于视觉的自主导航。该方法采用基于变分自编码器（VAE）的控制器，并通过基于干预的学习框架进行训练，使无人机能够从人类经验中学习视觉运动策略。我们在自建的四旋翼平台上在真实果园环境中验证了该方法。实地实验表明，在几次训练迭代后，提出的基于VAE的控制器能够基于前向安装的摄像头流自主导航无人机，并表现出强大的避障性能，在较少的人工干预下实现了更远的飞行距离，且优于现有算法。此外，我们还展示了该策略在新颖环境中的有效推广，并在不同条件和速度下保持竞争力。这项研究不仅推动了无人机自主性的发展，还对精准农业具有重要意义，有助于提高果园监测和管理的效率。 

---
# Periodic robust robotic rock chop via virtual model control 

**Title (ZH)**: 周期性鲁棒岩石切割的虚拟模型控制 

**Authors**: Yi Zhang, Fumiya Iida, Fulvio Forni  

**Link**: [PDF](https://arxiv.org/pdf/2508.02604)  

**Abstract**: Robotic cutting is a challenging contact-rich manipulation task where the robot must simultaneously negotiate unknown object mechanics, large contact forces, and precise motion requirements. We introduce a new virtual-model control scheme that enables knife rocking motion for robot manipulators, without pre-planned trajectories or precise information of the environment. Motion is generated through interconnection with virtual mechanisms, given by virtual springs, dampers, and masses arranged in a suitable way. Through analysis and experiments, we demonstrate that the controlled robot behavior settles into a periodic motion. Experiments with a Franka manipulator demonstrate robust cuts with five different vegetables, and sub-millimeter slice accuracy from 1 mm to 6 mm at nearly one cut per second. The same controller survives changes in knife shape and cutting board height, and adaptation to a different humanoid manipulator, demonstrating robustness and platform independence. 

**Abstract (ZH)**: 机器人切削是一项接触丰富的操作任务，其中机器人必须同时应对未知物体的机械特性、大的接触力以及精确的运动要求。我们提出了一种新的虚拟模型控制方案，使机器人 manipulator 能够实现刀具摇动运动，无需预先规划轨迹或精确的环境信息。运动通过与虚拟弹簧、阻尼器和质量的相互连接生成。通过分析和实验，我们证明控制的机器人行为会稳定在周期性运动中。使用 Franka maniuplators 的实验展示了对五种不同蔬菜的稳健切割，并实现了从 1 mm 到 6 mm 的亚毫米级切片精度，几乎每秒一次切割。相同的控制器能够适应刀具形状的变化、切割板高度的变化，并适用于不同的类人 manipulator，显示了其稳健性和平台独立性。 

---
# An RGB-D Camera-Based Multi-Small Flying Anchors Control for Wire-Driven Robots Connecting to the Environment 

**Title (ZH)**: 基于RGB-D摄像机的多小型悬挂锚点控制方法及其在环境连接的线驱动机器人中的应用 

**Authors**: Shintaro Inoue, Kento Kawaharazuka, Keita Yoneda, Sota Yuzaki, Yuta Sahara, Temma Suzuki, Kei Okada  

**Link**: [PDF](https://arxiv.org/pdf/2508.02544)  

**Abstract**: In order to expand the operational range and payload capacity of robots, wire-driven robots that leverage the external environment have been proposed. It can exert forces and operate in spaces far beyond those dictated by its own structural limits. However, for practical use, robots must autonomously attach multiple wires to the environment based on environmental recognition-an operation so difficult that many wire-driven robots remain restricted to specialized, pre-designed environments. Here, in this study, we propose a robot that autonomously connects multiple wires to the environment by employing a multi-small flying anchor system, as well as an RGB-D camera-based control and environmental recognition method. Each flying anchor is a drone with an anchoring mechanism at the wire tip, allowing the robot to attach wires by flying into position. Using the robot's RGB-D camera to identify suitable attachment points and a flying anchor position, the system can connect wires in environments that are not specially prepared, and can also attach multiple wires simultaneously. Through this approach, a wire-driven robot can autonomously attach its wires to the environment, thereby realizing the benefits of wire-driven operation at any location. 

**Abstract (ZH)**: 基于多小型飞行锚系统和RGB-D相机控制与环境识别的环境自主连接多条牵引线的机器人 

---
# Multi-Class Human/Object Detection on Robot Manipulators using Proprioceptive Sensing 

**Title (ZH)**: 基于本体感觉的机器人 manipulator 多类人类/物体检测 

**Authors**: Justin Hehli, Marco Heiniger, Maryam Rezayati, Hans Wernher van de Venn  

**Link**: [PDF](https://arxiv.org/pdf/2508.02425)  

**Abstract**: In physical human-robot collaboration (pHRC) settings, humans and robots collaborate directly in shared environments. Robots must analyze interactions with objects to ensure safety and facilitate meaningful workflows. One critical aspect is human/object detection, where the contacted object is identified. Past research introduced binary machine learning classifiers to distinguish between soft and hard objects. This study improves upon those results by evaluating three-class human/object detection models, offering more detailed contact analysis. A dataset was collected using the Franka Emika Panda robot manipulator, exploring preprocessing strategies for time-series analysis. Models including LSTM, GRU, and Transformers were trained on these datasets. The best-performing model achieved 91.11\% accuracy during real-time testing, demonstrating the feasibility of multi-class detection models. Additionally, a comparison of preprocessing strategies suggests a sliding window approach is optimal for this task. 

**Abstract (ZH)**: 在物理人机协作（pHRC）环境中的人类/机器人协作中，人类和机器人在共享环境中直接协作。机器人必须分析与物体的交互以确保安全并促进有意义的工作流程。一个关键方面是人类/物体检测，其中需要识别被接触的物体。以往研究引入了二元机器学习分类器来区分软物体和硬物体。本研究在此基础上通过评估三类人类/物体检测模型来改进以往成果，提供更详细的接触分析。使用Franka Emika Panda 机器人操作器收集数据，探索时间序列分析的预处理策略。在这些数据集上训练了包括LSTM、GRU和变换器在内的模型。最佳模型在实时测试中达到91.11%的准确率，证明了多类检测模型的可行性。此外，预处理策略比较表明滑动窗口方法在这种任务中是最优的。 

---
# Adaptive Lattice-based Motion Planning 

**Title (ZH)**: 自适应格子基运动规划 

**Authors**: Abhishek Dhar, Sarthak Mishra, Spandan Roy, Daniel Axehill  

**Link**: [PDF](https://arxiv.org/pdf/2508.02350)  

**Abstract**: This paper proposes an adaptive lattice-based motion planning solution to address the problem of generating feasible trajectories for systems, represented by a linearly parameterizable non-linear model operating within a cluttered environment. The system model is considered to have uncertain model parameters. The key idea here is to utilize input/output data online to update the model set containing the uncertain system parameter, as well as a dynamic estimated parameter of the model, so that the associated model estimation error reduces over time. This in turn improves the quality of the motion primitives generated by the lattice-based motion planner using a nominal estimated model selected on the basis of suitable criteria. The motion primitives are also equipped with tubes to account for the model mismatch between the nominal estimated model and the true system model, to guarantee collision-free overall motion. The tubes are of uniform size, which is directly proportional to the size of the model set containing the uncertain system parameter. The adaptive learning module guarantees a reduction in the diameter of the model set as well as in the parameter estimation error between the dynamic estimated parameter and the true system parameter. This directly implies a reduction in the size of the implemented tubes and guarantees that the utilized motion primitives go arbitrarily close to the resolution-optimal motion primitives associated with the true model of the system, thus significantly improving the overall motion planning performance over time. The efficiency of the motion planner is demonstrated by a suitable simulation example that considers a drone model represented by Euler-Lagrange dynamics containing uncertain parameters and operating within a cluttered environment. 

**Abstract (ZH)**: 基于自适应格形的运动规划方案：解决具不确定参数的非线性模型在复杂环境下的可行轨迹生成问题 

---
# Framework for Robust Motion Planning of Tethered Multi-Robot Systems in Marine Environments 

**Title (ZH)**: tethered多机器人系统在海洋环境中的鲁棒运动规划框架 

**Authors**: Markus Buchholz, Ignacio Carlucho, Zebin Huang, Michele Grimaldi, Pierre Nicolay, Sumer Tuncay, Yvan R. Petillot  

**Link**: [PDF](https://arxiv.org/pdf/2508.02287)  

**Abstract**: This paper introduces CoralGuide, a novel framework designed for path planning and trajectory optimization for tethered multi-robot systems. We focus on marine robotics, which commonly have tethered configurations of an Autonomous Surface Vehicle (ASV) and an Autonomous Underwater Vehicle (AUV). CoralGuide provides safe navigation in marine environments by enhancing the A* algorithm with specialized heuristics tailored for tethered ASV-AUV systems. Our method integrates catenary curve modelling for tether management and employs Bezier curve interpolation for smoother trajectory planning, ensuring efficient and synchronized operations without compromising safety. Through simulations and real-world experiments, we have validated CoralGuides effectiveness in improving path planning and trajectory optimization, demonstrating its potential to significantly enhance operational capabilities in marine research and infrastructure inspection. 

**Abstract (ZH)**: CoralGuide：一种用于 tethered 多机器人系统路径规划与轨迹优化的新框架 

---
# Tethered Multi-Robot Systems in Marine Environments 

**Title (ZH)**: 海洋环境中的 tethered 多机器人系统 

**Authors**: Markus Buchholz, Ignacio Carlucho, Michele Grimaldi, Yvan R. Petillot  

**Link**: [PDF](https://arxiv.org/pdf/2508.02264)  

**Abstract**: This paper introduces a novel simulation framework for evaluating motion control in tethered multi-robot systems within dynamic marine environments. Specifically, it focuses on the coordinated operation of an Autonomous Underwater Vehicle (AUV) and an Autonomous Surface Vehicle(ASV). The framework leverages GazeboSim, enhanced with realistic marine environment plugins and ArduPilots SoftwareIn-The-Loop (SITL) mode, to provide a high-fidelity simulation platform. A detailed tether model, combining catenary equations and physical simulation, is integrated to accurately represent the dynamic interactions between the vehicles and the environment. This setup facilitates the development and testing of advanced control strategies under realistic conditions, demonstrating the frameworks capability to analyze complex tether interactions and their impact on system performance. 

**Abstract (ZH)**: 本文介绍了一种用于评估缆绳约束多机器人系统在动态海洋环境中的运动控制的新模拟框架，具体聚焦于自主水下车辆(AUV)与自主水面车辆(ASV)的协调操作。该框架借助增强现实海洋环境插件的GazeboSim及ArduPilots软件在环(SITL)模式，提供了一个高保真度的模拟平台。结合缆绳方程和物理模拟的详细缆绳模型被集成进来，以准确表示车辆与环境之间的动态交互。该设置促进了在实际条件下先进控制策略的开发与测试，展示了该框架分析复杂缆绳交互及其对系统性能影响的能力。 

---
# TacMan-Turbo: Proactive Tactile Control for Robust and Efficient Articulated Object Manipulation 

**Title (ZH)**: TacMan-Turbo: 主动触觉控制以实现 robust 和 efficient 的刚性对象操控 

**Authors**: Zihang Zhao, Zhenghao Qi, Yuyang Li, Leiyao Cui, Zhi Han, Lecheng Ruan, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02204)  

**Abstract**: Adept manipulation of articulated objects is essential for robots to operate successfully in human environments. Such manipulation requires both effectiveness -- reliable operation despite uncertain object structures -- and efficiency -- swift execution with minimal redundant steps and smooth actions. Existing approaches struggle to achieve both objectives simultaneously: methods relying on predefined kinematic models lack effectiveness when encountering structural variations, while tactile-informed approaches achieve robust manipulation without kinematic priors but compromise efficiency through reactive, step-by-step exploration-compensation cycles. This paper introduces TacMan-Turbo, a novel proactive tactile control framework for articulated object manipulation that resolves this fundamental trade-off. Unlike previous approaches that treat tactile contact deviations merely as error signals requiring compensation, our method interprets these deviations as rich sources of local kinematic information. This new perspective enables our controller to predict optimal future interactions and make proactive adjustments, significantly enhancing manipulation efficiency. In comprehensive evaluations across 200 diverse simulated articulated objects and real-world experiments, our approach maintains a 100% success rate while significantly outperforming the previous tactile-informed method in time efficiency, action efficiency, and trajectory smoothness (all p-values < 0.0001). These results demonstrate that the long-standing trade-off between effectiveness and efficiency in articulated object manipulation can be successfully resolved without relying on prior kinematic knowledge. 

**Abstract (ZH)**: 灵巧操作 articulated 物体是机器人在人类环境中成功操作的关键。这种操作既需要有效性——在遇到结构不确定性时能够可靠运行——也需要效率——快速执行且步骤最少、动作流畅。现有方法难以同时实现这两个目标：依赖预定义动力学模型的方法在遇到结构变化时有效性不足，而基于触觉的信息的方法不依赖动力学先验从而实现稳健操作，但通过反应性的、逐步的探索—补偿循环降低了效率。本文介绍了一种新的前瞻触觉控制框架 TacMan-Turbo，解决了这种基本的权衡问题。与以往方法仅将触觉接触偏差视作需要补偿的误差信号不同，我们的方法将这些偏差视为丰富的局部动力学信息来源。这种新的视角使我们的控制器能够预测最优的未来交互并采取积极的调整，显著提高了操作效率。在针对 200 种不同模拟 articulated 物体以及真实世界实验的全面评估中，我们的方法保持了 100% 的成功率，在时间效率、动作效率和轨迹流畅度方面显著优于之前的触觉信息方法（所有 p 值 < 0.0001）。这些结果表明，articulated 物体操作中的长期有效性与效率权衡可以通过不依赖先验动力学知识的方法成功解决。 

---
# Towards High Precision: An Adaptive Self-Supervised Learning Framework for Force-Based Verification 

**Title (ZH)**: 面向高精度：基于自适应自监督学习框架的力基验证方法 

**Authors**: Zebin Duan, Frederik Hagelskjær, Aljaz Kramberger, Juan Heredia and, Norbert Krüger  

**Link**: [PDF](https://arxiv.org/pdf/2508.02153)  

**Abstract**: The automation of robotic tasks requires high precision and adaptability, particularly in force-based operations such as insertions. Traditional learning-based approaches either rely on static datasets, which limit their ability to generalize, or require frequent manual intervention to maintain good performances. As a result, ensuring long-term reliability without human supervision remains a significant challenge. To address this, we propose an adaptive self-supervised learning framework for insertion classification that continuously improves its precision over time. The framework operates in real-time, incrementally refining its classification decisions by integrating newly acquired force data. Unlike conventional methods, it does not rely on pre-collected datasets but instead evolves dynamically with each task execution. Through real-world experiments, we demonstrate how the system progressively reduces execution time while maintaining near-perfect precision as more samples are processed. This adaptability ensures long-term reliability in force-based robotic tasks while minimizing the need for manual intervention. 

**Abstract (ZH)**: 基于力的操作的机器人任务自动化需要高精度和适应性，传统的基于学习的方法要么依赖于静态数据集，限制了其泛化能力，要么需要频繁的手动干预以保持性能。因此，在无需人工监督的情况下确保长期可靠性仍然是一个重大挑战。为此，我们提出了一种适应性自我监督学习框架，用于插入分类，该框架能够随着时间的推移持续提高其精度。该框架实时运行，通过集成新获取的力数据，逐步精炼其分类决策。与传统方法不同，它不依赖于预先收集的数据集，而是随着每次任务执行动态演变。通过实际实验，我们展示了该系统如何在处理更多样本时逐步减少执行时间，并保持接近完美的精度。这种适应性确保了力基机器人任务的长期可靠性，同时最大限度地减少了手动干预的需要。 

---
# Design and Control of an Actively Morphing Quadrotor with Vertically Foldable Arms 

**Title (ZH)**: 可主动变形且臂可竖折的四旋翼无人机的设计与控制 

**Authors**: Tingyu Yeh, Mengxin Xu, Lijun Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.02022)  

**Abstract**: In this work, we propose a novel quadrotor design capable of folding its arms vertically to grasp objects and navigate through narrow spaces. The transformation is controlled actively by a central servomotor, gears, and racks. The arms connect the motor bases to the central frame, forming a parallelogram structure that ensures the propellers maintain a constant orientation during morphing. In its stretched state, the quadrotor resembles a conventional design, and when contracted, it functions as a gripper with grasping components emerging from the motor bases. To mitigate disturbances during transforming and grasping payloads, we employ an adaptive sliding mode controller with a disturbance observer. After fully folded, the quadrotor frame shrinks to 67% of its original size. The control performance and versatility of the morphing quadrotor are validated through real-world experiments. 

**Abstract (ZH)**: 本研究提出了一种新颖的四旋翼设计，能够在垂直折叠其臂部以抓取物体并穿梭于狭窄空间。转换由中央伺服电机、齿轮和齿条主动控制。臂部将电机底座连接到中央框架，形成一个确保推进器在变形过程中保持恒定方向的平行四边形结构。在拉伸状态下，该四旋翼机类似于常规设计；当收缩时，它变为具有从电机底座伸出的抓取部件的夹持器。为减轻转换和抓取载荷时的干扰，我们采用带有干扰观测器的自适应滑模控制器。完全折叠后，四旋翼机框架缩小至原尺寸的67%。通过实际实验验证了变形四旋翼机的控制性能和多功能性。 

---
# From Photons to Physics: Autonomous Indoor Drones and the Future of Objective Property Assessment 

**Title (ZH)**: 从光子到物理：自主室内无人机与客观属性评估的未来 

**Authors**: Petteri Teikari, Mike Jarrell, Irene Bandera Moreno, Harri Pesola  

**Link**: [PDF](https://arxiv.org/pdf/2508.01965)  

**Abstract**: The convergence of autonomous indoor drones with physics-aware sensing technologies promises to transform property assessment from subjective visual inspection to objective, quantitative measurement. This comprehensive review examines the technical foundations enabling this paradigm shift across four critical domains: (1) platform architectures optimized for indoor navigation, where weight constraints drive innovations in heterogeneous computing, collision-tolerant design, and hierarchical control systems; (2) advanced sensing modalities that extend perception beyond human vision, including hyperspectral imaging for material identification, polarimetric sensing for surface characterization, and computational imaging with metaphotonics enabling radical miniaturization; (3) intelligent autonomy through active reconstruction algorithms, where drones equipped with 3D Gaussian Splatting make strategic decisions about viewpoint selection to maximize information gain within battery constraints; and (4) integration pathways with existing property workflows, including Building Information Modeling (BIM) systems and industry standards like Uniform Appraisal Dataset (UAD) 3.6. 

**Abstract (ZH)**: 自主室内无人机结合物理感知技术的收敛有望将房产评估从主观的视觉检查转变为客观的定量测量。本文综述了促成这一范式转变的技术基础，涵盖了四个关键领域：(1) 优化室内导航的平台架构，其中重量限制推动了异构计算、防碰撞设计和分层控制系统的发展；(2) 超越人类视觉的先进感知模式，包括用于材料识别的超光谱成像、用于表面表征的偏振光 sensing 以及通过元光子学实现的计算成像以实现根本性的微型化；(3) 通过主动重建算法实现的智能自主性，其中配备三维正态斑图化算法的无人机在电池限制条件下做出关于视点选择的策略性决策以最大化信息增益；以及 (4) 与现有房产工作流程的集成途径，包括建筑信息建模（BIM）系统和行业标准如统一评估数据集（UAD）3.6。 

---
# Exploring Stiffness Gradient Effects in Magnetically Induced Metamorphic Materials via Continuum Simulation and Validation 

**Title (ZH)**: 通过连续模拟与验证探索磁场诱导 metamorphic 材料中的刚度梯度效应 

**Authors**: Wentao Shi, Yang Yang, Yiming Huang, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.01810)  

**Abstract**: Magnetic soft continuum robots are capable of bending with remote control in confined space environments, and they have been applied in various bioengineering contexts. As one type of ferromagnetic soft continuums, the Magnetically Induced Metamorphic Materials (MIMMs)-based continuum (MC) exhibits similar bending behaviors. Based on the characteristics of its base material, MC is flexible in modifying unit stiffness and convenient in molding fabrication. However, recent studies on magnetic continuum robots have primarily focused on one or two design parameters, limiting the development of a comprehensive magnetic continuum bending model. In this work, we constructed graded-stiffness MCs (GMCs) and developed a numerical model for GMCs' bending performance, incorporating four key parameters that determine their performance. The simulated bending results were validated with real bending experiments in four different categories: varying magnetic field, cross-section, unit stiffness, and unit length. The graded-stiffness design strategy applied to GMCs prevents sharp bending at the fixed end and results in a more circular curvature. We also trained an expansion model for GMCs' bending performance that is highly efficient and accurate compared to the simulation process. An extensive library of bending prediction for GMCs was built using the trained model. 

**Abstract (ZH)**: 磁诱导 metamorphic 材料为基础的渐变刚度连续体机器人能够在受限空间中远程控制弯曲，并已应用于多种生物工程领域。在利用基材特性可调节单一刚度和方便成型加工的基础上，磁连续体机器人的近期研究主要集中在一两个设计参数上，限制了综合磁连续体弯曲模型的发展。在这项工作中，我们构建了渐变刚度连续体（GMC），并发展了适用于 GMC 的弯曲性能数值模型，综合了影响性能的四个关键参数。通过四种不同类别（磁场强度、截面形状、单一刚度和单一长度）的模拟弯曲实验和实际弯曲实验验证了模拟结果。渐变刚度设计策略应用于 GMC 避免了固定端的尖锐弯曲，产生了更圆的曲率。此外，我们还训练了一个高效的高精度扩展模型，用于预测 GMC 的弯曲性能，并构建了基于该模型的大量弯曲预测库。 

---
# Learning to Perform Low-Contact Autonomous Nasotracheal Intubation by Recurrent Action-Confidence Chunking with Transformer 

**Title (ZH)**: 学习通过递归动作-信心片段方法运用变压器进行低接触自主鼻.RESET插管 

**Authors**: Yu Tian, Ruoyi Hao, Yiming Huang, Dihong Xie, Catherine Po Ling Chan, Jason Ying Kuen Chan, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.01808)  

**Abstract**: Nasotracheal intubation (NTI) is critical for establishing artificial airways in clinical anesthesia and critical care. Current manual methods face significant challenges, including cross-infection, especially during respiratory infection care, and insufficient control of endoluminal contact forces, increasing the risk of mucosal injuries. While existing studies have focused on automated endoscopic insertion, the automation of NTI remains unexplored despite its unique challenges: Nasotracheal tubes exhibit greater diameter and rigidity than standard endoscopes, substantially increasing insertion complexity and patient risks. We propose a novel autonomous NTI system with two key components to address these challenges. First, an autonomous NTI system is developed, incorporating a prosthesis embedded with force sensors, allowing for safety assessment and data filtering. Then, the Recurrent Action-Confidence Chunking with Transformer (RACCT) model is developed to handle complex tube-tissue interactions and partial visual observations. Experimental results demonstrate that the RACCT model outperforms the ACT model in all aspects and achieves a 66% reduction in average peak insertion force compared to manual operations while maintaining equivalent success rates. This validates the system's potential for reducing infection risks and improving procedural safety. 

**Abstract (ZH)**: 鼻咽气管插管（NTI）是临床麻醉和重症监护中建立人工气道的关键技术。现有的手动方法面临显著挑战，包括交叉感染，尤其是在呼吸道感染护理中，以及内腔接触力控制不足，增加黏膜损伤的风险。尽管已有研究集中在自动化内镜插入上，但鼻咽气管插管的自动化仍未得到探索，其独特挑战包括鼻咽气管导管直径和刚度大于标准内窥镜，显著增加插入复杂性和患者风险。我们提出了一种新的自主鼻咽气管插管系统，其中包括两个关键组件以应对这些挑战。首先，开发了一种自主鼻咽气管插管系统，该系统包含一个内置力传感器的植入物，以实现安全性评估和数据过滤。然后，开发了基于变换器的循环动作-置信度片段模型（RACCT），以处理复杂管-组织相互作用和不完整视觉观察。实验结果表明，RACCT模型在各方面都优于ACT模型，与手动操作相比，平均峰值插入力降低了66%，同时保持了同等的成功率。这验证了该系统在降低感染风险和提高操作安全方面的潜在价值。 

---
# Energy-Predictive Planning for Optimizing Drone Service Delivery 

**Title (ZH)**: 基于能量预测的无人机服务交付优化规划 

**Authors**: Guanting Ren, Babar Shahzaad, Balsam Alkouz, Abdallah Lakhdari, Athman Bouguettaya  

**Link**: [PDF](https://arxiv.org/pdf/2508.01671)  

**Abstract**: We propose a novel Energy-Predictive Drone Service (EPDS) framework for efficient package delivery within a skyway network. The EPDS framework incorporates a formal modeling of an EPDS and an adaptive bidirectional Long Short-Term Memory (Bi-LSTM) machine learning model. This model predicts the energy status and stochastic arrival times of other drones operating in the same skyway network. Leveraging these predictions, we develop a heuristic optimization approach for composite drone services. This approach identifies the most time-efficient and energy-efficient skyway path and recharging schedule for each drone in the network. We conduct extensive experiments using a real-world drone flight dataset to evaluate the performance of the proposed framework. 

**Abstract (ZH)**: 一种新型能量预测无人机服务（EPDS）框架：在天道网络中实现高效的包裹交付 

---
# Decentralized Aerial Manipulation of a Cable-Suspended Load using Multi-Agent Reinforcement Learning 

**Title (ZH)**: 基于多智能体强化学习的分布式缆索悬挂负载空中操控 

**Authors**: Jack Zeng, Andreu Matoses Gimenez, Eugene Vinitsky, Javier Alonso-Mora, Sihao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.01522)  

**Abstract**: This paper presents the first decentralized method to enable real-world 6-DoF manipulation of a cable-suspended load using a team of Micro-Aerial Vehicles (MAVs). Our method leverages multi-agent reinforcement learning (MARL) to train an outer-loop control policy for each MAV. Unlike state-of-the-art controllers that utilize a centralized scheme, our policy does not require global states, inter-MAV communications, nor neighboring MAV information. Instead, agents communicate implicitly through load pose observations alone, which enables high scalability and flexibility. It also significantly reduces computing costs during inference time, enabling onboard deployment of the policy. In addition, we introduce a new action space design for the MAVs using linear acceleration and body rates. This choice, combined with a robust low-level controller, enables reliable sim-to-real transfer despite significant uncertainties caused by cable tension during dynamic 3D motion. We validate our method in various real-world experiments, including full-pose control under load model uncertainties, showing setpoint tracking performance comparable to the state-of-the-art centralized method. We also demonstrate cooperation amongst agents with heterogeneous control policies, and robustness to the complete in-flight loss of one MAV. Videos of experiments: this https URL 

**Abstract (ZH)**: 本文提出了首款用于通过微空中车辆（MAVs）团队实现真实世界中6-自由度操纵悬吊负载的去中心化方法。我们的方法利用多智能体强化学习（MARL）为每架MAV训练一个外环控制策略。与现有利用中心化方案的控制器不同，我们的策略不要求全局状态、智能体间通信或邻近智能体的信息。取而代之的是，智能体仅通过负载姿态观测进行隐式通信，这使得系统具有高可扩展性和灵活性。此外，这种方法还显著降低了推理时的计算成本，使策略能够实现搭载部署。另外，我们为MAVs引入了一种新的动作空间设计，使用线性加速度和体速率。这种选择，结合一个鲁棒的低层控制器，使得即使在动态3D运动中由于缆绳张力引起的显著不确定性时，也能可靠地实现仿真到现实的转移。我们在各种实际实验中验证了我们的方法，包括在负载模型不确定性下的全姿态控制，展示了与现有最佳中心化方法相当的定值跟踪性能。我们还展示了具有不同控制策略的智能体之间的合作以及面对其中一架MAV完全空中故障时的鲁棒性。实验视频：this https URL 

---
# Physically-based Lighting Augmentation for Robotic Manipulation 

**Title (ZH)**: 基于物理的照明增强方法在机器人操作中的应用 

**Authors**: Shutong Jin, Lezhong Wang, Ben Temming, Florian T. Pokorny  

**Link**: [PDF](https://arxiv.org/pdf/2508.01442)  

**Abstract**: Despite advances in data augmentation, policies trained via imitation learning still struggle to generalize across environmental variations such as lighting changes. To address this, we propose the first framework that leverages physically-based inverse rendering for lighting augmentation on real-world human demonstrations. Specifically, inverse rendering decomposes the first frame in each demonstration into geometric (surface normal, depth) and material (albedo, roughness, metallic) properties, which are then used to render appearance changes under different lighting. To ensure consistent augmentation across each demonstration, we fine-tune Stable Video Diffusion on robot execution videos for temporal lighting propagation. We evaluate our framework by measuring the structural and temporal consistency of the augmented sequences, and by assessing its effectiveness in reducing the behavior cloning generalization gap (40.1%) on a 7-DoF robot across 6 lighting conditions using 720 real-world evaluations. We further showcase three downstream applications enabled by the proposed framework. 

**Abstract (ZH)**: 尽管在数据增强方面取得了进展，通过imitation learning训练的策略仍然难以在光照变化等环境变化中泛化。为了解决这个问题，我们提出了第一个利用基于物理的逆渲染进行光照增强的框架，以增强现实人类示范。具体而言，逆渲染将每个示范的第一帧分解为几何（法线、深度）和材料（反射率、粗糙度、金属度）属性，并利用这些属性在不同光照条件下渲染外观变化。为了确保每个示范中增强的连贯性，我们在机器人执行视频上微调了Stable Video Diffusion以实现时间光照传播。我们通过测量增强序列的结构和时间一致性，并通过在6种光照条件下使用720个真实场景评估，在7-DoF机器人上减少行为克隆泛化差距（40.1%）来评估该框架的有效性。我们进一步展示了该框架使三个下游应用成为可能。 

---
# Design of Q8bot: A Miniature, Low-Cost, Dynamic Quadruped Built with Zero Wires 

**Title (ZH)**: Q8bot的设计：一款无线、低成本、动态四足机器人 

**Authors**: Yufeng Wu, Dennis Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01149)  

**Abstract**: This paper introduces Q8bot, an open-source, miniature quadruped designed for robotics research and education. We present the robot's novel zero-wire design methodology, which leads to its superior form factor, robustness, replicability, and high performance. With a size and weight similar to a modern smartphone, this standalone robot can walk for over an hour on a single battery charge and survive meter-high drops with simple repairs. Its 300-dollar bill of materials includes minimal off-the-shelf components, readily available custom electronics from online vendors, and structural parts that can be manufactured on hobbyist 3D printers. A preliminary user assembly study confirms that Q8bot can be easily replicated, with an average assembly time of under one hour by a single person. With heuristic open-loop control, Q8bot achieves a stable walking speed of 5.4 body lengths per second and a turning speed of 5 radians per second, along with other dynamic movements such as jumping and climbing moderate slopes. 

**Abstract (ZH)**: 本论文介绍了Q8bot，一种面向机器人研究与教育的开源微型四足机器人。我们介绍了该机器人独特的无缆设计方法论，这使其具有优异的外形、 sturdy性和可复制性以及高性能。这款单体机器人大小和重量类似现代智能手机，单次电池充电后可行走超过一个小时，并且简单修复后可以从一米高的地方跌落存活。其300美元的物料清单包括少量标准组件、易于获取的定制电子元件以及可以使用爱好者级3D打印机制造的结构部件。初步用户组装研究证实，Q8bot 可以轻松复制，单人组装时间平均小于一小时。通过启发式的开环控制，Q8bot 达到了每秒5.4个身长的稳定行走速度和每秒5弧度的转向速度，以及其他动态动作，如跳跃和攀爬较陡斜坡。 

---
# Improving Drone Racing Performance Through Iterative Learning MPC 

**Title (ZH)**: 通过迭代学习模型预测控制提高无人机竞速 performance 

**Authors**: Haocheng Zhao, Niklas Schlüter, Lukas Brunke, Angela P. Schoellig  

**Link**: [PDF](https://arxiv.org/pdf/2508.01103)  

**Abstract**: Autonomous drone racing presents a challenging control problem, requiring real-time decision-making and robust handling of nonlinear system dynamics. While iterative learning model predictive control~(LMPC) offers a promising framework for iterative performance improvement, its direct application to drone racing faces challenges like real-time compatibility or the trade-off between time-optimal and safe traversal. In this paper, we enhance LMPC with three key innovations:~(1) an adaptive cost function that dynamically weights time-optimal tracking against centerline adherence,~(2)~a shifted local safe set to prevent excessive shortcutting and enable more robust iterative updates, and~(3) a Cartesian-based formulation that accommodates safety constraints without the singularities or integration errors associated with Frenet-frame transformations. Results from extensive simulation and real-world experiments demonstrate that our improved algorithm can optimize initial trajectories generated by a wide range of controllers with varying levels of tuning for a maximum improvement in lap time by 60.85\%. Even applied to the most aggressively tuned state-of-the-art model-based controller, MPCC++, on a real drone, a 6.05\% improvement is still achieved. Overall, the proposed method pushes the drone toward faster traversal and avoids collisions in simulation and real-world experiments, making it a practical solution to improve the peak performance of drone racing. 

**Abstract (ZH)**: 自主无人机竞速 presents a challenging control problem, requiring real-time decision-making and robust handling of nonlinear system dynamics. While iterative learning model predictive control (LMPC) offers a promising framework for iterative performance improvement, its direct application to drone racing faces challenges like real-time compatibility or the trade-off between time-optimal and safe traversal. In this paper, we enhance LMPC with three key innovations: (1) an adaptive cost function that dynamically weights time-optimal tracking against centerline adherence, (2) a shifted local safe set to prevent excessive shortcutting and enable more robust iterative updates, and (3) a Cartesian-based formulation that accommodates safety constraints without the singularities or integration errors associated with Frenet-frame transformations. Results from extensive simulation and real-world experiments demonstrate that our improved algorithm can optimize initial trajectories generated by a wide range of controllers with varying levels of tuning for a maximum improvement in lap time by 60.85%. Even applied to the most aggressively tuned state-of-the-art model-based controller, MPCC++, on a real drone, a 6.05% improvement is still achieved. Overall, the proposed method pushes the drone toward faster traversal and avoids collisions in simulation and real-world experiments, making it a practical solution to improve the peak performance of drone racing. 

---
# Learning Pivoting Manipulation with Force and Vision Feedback Using Optimization-based Demonstrations 

**Title (ZH)**: 基于优化演示的力与视觉反馈驱动的翻转操作学习 

**Authors**: Yuki Shirai, Kei Ota, Devesh K. Jha, Diego Romeres  

**Link**: [PDF](https://arxiv.org/pdf/2508.01082)  

**Abstract**: Non-prehensile manipulation is challenging due to complex contact interactions between objects, the environment, and robots. Model-based approaches can efficiently generate complex trajectories of robots and objects under contact constraints. However, they tend to be sensitive to model inaccuracies and require access to privileged information (e.g., object mass, size, pose), making them less suitable for novel objects. In contrast, learning-based approaches are typically more robust to modeling errors but require large amounts of data. In this paper, we bridge these two approaches to propose a framework for learning closed-loop pivoting manipulation. By leveraging computationally efficient Contact-Implicit Trajectory Optimization (CITO), we design demonstration-guided deep Reinforcement Learning (RL), leading to sample-efficient learning. We also present a sim-to-real transfer approach using a privileged training strategy, enabling the robot to perform pivoting manipulation using only proprioception, vision, and force sensing without access to privileged information. Our method is evaluated on several pivoting tasks, demonstrating that it can successfully perform sim-to-real transfer. 

**Abstract (ZH)**: 基于接触的非抓取操作由于对象、环境和机器人之间的复杂接触交互而具有挑战性。模型驱动的方法可以在接触约束下高效生成机器人和物体的复杂轨迹。然而，它们往往对模型不准确性敏感，并需要访问特权信息（如物体的质量、尺寸、姿态），使其不适用于新型物体。相比之下，基于学习的方法通常对建模错误更具有鲁棒性，但需要大量的数据。本文我们结合这两种方法，提出了一个学习闭环 pivot 操作的框架。通过利用计算效率高的接触隐式轨迹优化（CITO），我们设计了演示引导的深度强化学习（RL），实现了样本高效学习。我们还提出了一种从仿真到现实的转移方法，并采用特权训练策略，使机器人仅通过 proprioception、视觉和力感知就能执行 pivot 操作，而无需访问特权信息。我们的方法在几个 pivot 任务上进行了评估，展示了其成功实现从仿真到现实的转移能力。 

---
# Service Discovery-Based Hybrid Network Middleware for Efficient Communication in Distributed Robotic Systems 

**Title (ZH)**: 基于服务发现的混合网络中间件：分布式机器人系统中的高效通信 

**Authors**: Shiyao Sang, Yinggang Ling  

**Link**: [PDF](https://arxiv.org/pdf/2508.00947)  

**Abstract**: Robotic middleware is fundamental to ensuring reliable communication among system components and is crucial for intelligent robotics, autonomous vehicles, and smart manufacturing. However, existing robotic middleware often struggles to meet the diverse communication demands, optimize data transmission efficiency, and maintain scheduling determinism between Orin computing units in large-scale L4 autonomous vehicle deployments. This paper presents RIMAOS2C, a service discovery-based hybrid network communication middleware designed to tackle these challenges. By leveraging multi-level service discovery multicast, RIMAOS2C supports a wide variety of communication modes, including multiple cross-chip Ethernet protocols and PCIe communication capabilities. Its core mechanism, the Message Bridge, optimizes data flow forwarding and employs shared memory for centralized message distribution, reducing message redundancy and minimizing transmission delay uncertainty. Tested on L4 vehicles and Jetson Orin domain controllers, RIMAOS2C leverages TCP-based ZeroMQ to overcome the large-message transmission bottleneck in native CyberRT. In scenarios with two cross-chip subscribers, it eliminates message redundancy and improves large-data transmission efficiency by 36 to 40 percent while reducing callback latency variation by 42 to 906 percent. This research advances the communication capabilities of robotic operating systems and proposes a novel approach to optimizing communication in distributed computing architectures for autonomous driving. 

**Abstract (ZH)**: 基于服务发现的混合网络通信中间件RIMAOS2C 

---
# Uncertainty-Aware Perception-Based Control for Autonomous Racing 

**Title (ZH)**: 基于感知的、考虑不确定性自主赛车控制 

**Authors**: Jelena Trisovic, Andrea Carron, Melanie N. Zeilinger  

**Link**: [PDF](https://arxiv.org/pdf/2508.02494)  

**Abstract**: Autonomous systems operating in unknown environments often rely heavily on visual sensor data, yet making safe and informed control decisions based on these measurements remains a significant challenge. To facilitate the integration of perception and control in autonomous vehicles, we propose a novel perception-based control approach that incorporates road estimation, quantification of its uncertainty, and uncertainty-aware control based on this estimate. At the core of our method is a parametric road curvature model, optimized using visual measurements of the road through a constrained nonlinear optimization problem. This process ensures adherence to constraints on both model parameters and curvature. By leveraging the Frenet frame formulation, we embed the estimated track curvature into the system dynamics, allowing the controller to explicitly account for perception uncertainty and enhancing robustness to estimation errors based on visual input. We validate our approach in a simulated environment, using a high-fidelity 3D rendering engine, and demonstrate its effectiveness in achieving reliable and uncertainty-aware control for autonomous racing. 

**Abstract (ZH)**: 自主系统在未知环境中运作时常依赖视觉传感器数据，然而基于这些测量数据做出安全、可靠的控制决策仍然是一个重大挑战。为了促进自主车辆中感知与控制的整合，我们提出了一种新颖的基于感知的控制方法，该方法结合了道路估计、估计的不确定性量化以及基于此估计的不确定性感知控制。该方法的核心是一种子模型参数化的道路曲率模型，通过对视觉测量的道路进行约束非线性优化问题求解来优化。这一过程确保了模型参数和曲率的约束满足。通过利用弗朗et帧公式，我们将估计的道路曲率嵌入系统动力学中，使得控制器能够明确考虑感知的不确定性，并基于视觉输入的估计误差提高系统的鲁棒性。我们在一个高保真3D渲染引擎模拟环境中验证了该方法，并展示了其在自主赛车中的有效性和不确定性感知控制能力。 

---
# An Event-based Fast Intensity Reconstruction Scheme for UAV Real-time Perception 

**Title (ZH)**: 基于事件的快速强度重建方案用于UAV实时感知 

**Authors**: Xin Dong, Yiwei Zhang, Yangjie Cui, Jinwu Xiang, Daochun Li, Zhan Tu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02238)  

**Abstract**: Event cameras offer significant advantages, including a wide dynamic range, high temporal resolution, and immunity to motion blur, making them highly promising for addressing challenging visual conditions. Extracting and utilizing effective information from asynchronous event streams is essential for the onboard implementation of event cameras. In this paper, we propose a streamlined event-based intensity reconstruction scheme, event-based single integration (ESI), to address such implementation challenges. This method guarantees the portability of conventional frame-based vision methods to event-based scenarios and maintains the intrinsic advantages of event cameras. The ESI approach reconstructs intensity images by performing a single integration of the event streams combined with an enhanced decay algorithm. Such a method enables real-time intensity reconstruction at a high frame rate, typically 100 FPS. Furthermore, the relatively low computation load of ESI fits onboard implementation suitably, such as in UAV-based visual tracking scenarios. Extensive experiments have been conducted to evaluate the performance comparison of ESI and state-of-the-art algorithms. Compared to state-of-the-art algorithms, ESI demonstrates remarkable runtime efficiency improvements, superior reconstruction quality, and a high frame rate. As a result, ESI enhances UAV onboard perception significantly under visual adversary surroundings. In-flight tests, ESI demonstrates effective performance for UAV onboard visual tracking under extremely low illumination conditions(2-10lux), whereas other comparative algorithms fail due to insufficient frame rate, poor image quality, or limited real-time performance. 

**Abstract (ZH)**: 事件相机因其宽动态范围、高时间分辨率和运动模糊免疫等显著优势，成为应对复杂视觉条件的理想选择。从异步事件流中提取并利用有效信息对于事件相机的机载实现至关重要。本文提出了一种简化的事件驱动强度重建方案——事件驱动单积分(ESI)，以应对这些实现挑战。该方法确保了传统基于帧的视觉方法在事件驱动场景中的可移植性，并保持事件相机的固有优势。ESI通过结合改进的衰减算法对事件流进行单次积分，从而实现了实时高帧率（通常为100 FPS）强度重建。此外，ESI较低的计算负载使其适合于机载实现，如无人机视觉跟踪场景。通过广泛实验，ESI在与最新算法性能对比中展示了显著的运行时效率提升、卓越的重建质量和高帧率。因此，ESI在视觉对抗环境下显著增强了无人机的机载感知能力。飞行测试表明，在极低光照条件下（2-10lux），ESI表现出有效的无人机机载视觉跟踪性能，而其他比较算法因帧率不足、图像质量差或实时性能受限而失败。 

---
# mmWave Radar-Based Non-Line-of-Sight Pedestrian Localization at T-Junctions Utilizing Road Layout Extraction via Camera 

**Title (ZH)**: 基于毫米波雷达的利用摄像头提取道路布局实现T字路口非视距行人定位 

**Authors**: Byeonggyu Park, Hee-Yeun Kim, Byonghyok Choi, Hansang Cho, Byungkwan Kim, Soomok Lee, Mingu Jeon, Seong-Woo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.02348)  

**Abstract**: Pedestrians Localization in Non-Line-of-Sight (NLoS) regions within urban environments poses a significant challenge for autonomous driving systems. While mmWave radar has demonstrated potential for detecting objects in such scenarios, the 2D radar point cloud (PCD) data is susceptible to distortions caused by multipath reflections, making accurate spatial inference difficult. Additionally, although camera images provide high-resolution visual information, they lack depth perception and cannot directly observe objects in NLoS regions. In this paper, we propose a novel framework that interprets radar PCD through road layout inferred from camera for localization of NLoS pedestrians. The proposed method leverages visual information from the camera to interpret 2D radar PCD, enabling spatial scene reconstruction. The effectiveness of the proposed approach is validated through experiments conducted using a radar-camera system mounted on a real vehicle. The localization performance is evaluated using a dataset collected in outdoor NLoS driving environments, demonstrating the practical applicability of the method. 

**Abstract (ZH)**: 在城市环境中非视距(NLoS)区域中的行人定位对自主驾驶系统构成了重大挑战。尽管毫米波雷达在这样的场景下检测物体方面展示了潜力，但2D雷达点云(PCD)数据易受多路径反射引起的失真影响，导致空间推断困难。此外，虽然相机图像提供了高分辨率的视觉信息，但缺乏深度感知能力，无法直接观察NLoS区域中的物体。在本文中，我们提出了一种新颖的框架，通过相机推断的道路布局来解释雷达PCD，以实现NLoS行人定位。所提方法利用相机的视觉信息解释2D雷达PCD，从而实现空间场景重建。通过在真实车辆上安装的雷达-相机系统进行的实验验证了所提方法的有效性。通过在室外NLoS驾驶环境中收集的数据集评估定位性能，证明了该方法的实际适用性。 

---
# ChairPose: Pressure-based Chair Morphology Grounded Sitting Pose Estimation through Simulation-Assisted Training 

**Title (ZH)**: ChairPose: 基于压力的椅子形态引导的坐姿估计通过模拟辅助训练 

**Authors**: Lala Shakti Swarup Ray, Vitor Fortes Rey, Bo Zhou, Paul Lukowicz, Sungho Suh  

**Link**: [PDF](https://arxiv.org/pdf/2508.01850)  

**Abstract**: Prolonged seated activity is increasingly common in modern environments, raising concerns around musculoskeletal health, ergonomics, and the design of responsive interactive systems. Existing posture sensing methods such as vision-based or wearable approaches face limitations including occlusion, privacy concerns, user discomfort, and restricted deployment flexibility. We introduce ChairPose, the first full body, wearable free seated pose estimation system that relies solely on pressure sensing and operates independently of chair geometry. ChairPose employs a two stage generative model trained on pressure maps captured from a thin, chair agnostic sensing mattress. Unlike prior approaches, our method explicitly incorporates chair morphology into the inference process, enabling accurate, occlusion free, and privacy preserving pose estimation. To support generalization across diverse users and chairs, we introduce a physics driven data augmentation pipeline that simulates realistic variations in posture and seating conditions. Evaluated across eight users and four distinct chairs, ChairPose achieves a mean per joint position error of 89.4 mm when both the user and the chair are unseen, demonstrating robust generalization to novel real world generalizability. ChairPose expands the design space for posture aware interactive systems, with potential applications in ergonomics, healthcare, and adaptive user interfaces. 

**Abstract (ZH)**: 长时间静坐活动在现代环境中越来越普遍，引发了对肌肉骨骼健康、人机工程学和响应式交互系统设计的关注。现有的基于视觉或穿戴式姿势传感方法面临遮挡、隐私问题、用户不适和部署灵活性受限等限制。我们引入了ChairPose，这是一种首款基于全身、穿戴式压力传感的独立于椅子几何结构的坐姿估测系统。ChairPose采用一种基于压力图的两阶段生成模型，这些压力图由一种薄且对椅子无依赖性的传感床垫捕获。与先前的方法不同，我们的方法在推断过程中明确考虑了椅子的形态特征，从而实现了准确、无遮挡且保护隐私的姿势估测。为了支持对多样化用户和椅子的泛化能力，我们引入了一种基于物理驱动的数据增强pipeline，模拟了姿势和座椅条件的现实变化。在八名用户和四把不同椅子上进行评估，当用户和椅子均未在训练集中出现时，ChairPose 的每关节位置误差的平均值为89.4毫米，展示了其在新型真实世界环境下的鲁棒泛化能力。ChairPose 扩展了姿势感知交互系统的设计空间，具有在人机工程学、医疗保健和自适应用户界面方面的潜在应用。 

---
