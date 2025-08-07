# Reliable and Real-Time Highway Trajectory Planning via Hybrid Learning-Optimization Frameworks 

**Title (ZH)**: 基于混合学习-优化框架的可靠实时高速公路轨迹规划 

**Authors**: Yujia Lu, Chong Wei, Lu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.04436)  

**Abstract**: Autonomous highway driving presents a high collision risk due to fast-changing environments and limited reaction time, necessitating reliable and efficient trajectory planning. This paper proposes a hybrid trajectory planning framework that integrates the adaptability of learning-based methods with the formal safety guarantees of optimization-based approaches. The framework features a two-layer architecture: an upper layer employing a graph neural network (GNN) trained on real-world highway data to predict human-like longitudinal velocity profiles, and a lower layer utilizing path optimization formulated as a mixed-integer quadratic programming (MIQP) problem. The primary contribution is the lower-layer path optimization model, which introduces a linear approximation of discretized vehicle geometry to substantially reduce computational complexity, while enforcing strict spatiotemporal non-overlapping constraints to formally guarantee collision avoidance throughout the planning horizon. Experimental results demonstrate that the planner generates highly smooth, collision-free trajectories in complex real-world emergency scenarios, achieving success rates exceeding 97% with average planning times of 54 ms, thereby confirming real-time capability. 

**Abstract (ZH)**: 基于学习方法的适应性与优化方法的形式安全保证相结合的自驾车高速道路规划框架 

---
# Incorporating Stochastic Models of Controller Behavior into Kinodynamic Efficiently Adaptive State Lattices for Mobile Robot Motion Planning in Off-Road Environments 

**Title (ZH)**: 将控制器行为的随机模型整合进离线高效自适应状态格网中以优化离路环境下的移动机器人运动规划 

**Authors**: Eric R. Damm, Eli S. Lancaster, Felix A. Sanchez, Kiana Bronder, Jason M. Gregory, Thomas M. Howard  

**Link**: [PDF](https://arxiv.org/pdf/2508.04384)  

**Abstract**: Mobile robot motion planners rely on theoretical models to predict how the robot will move through the world. However, when deployed on a physical robot, these models are subject to errors due to real-world physics and uncertainty in how the lower-level controller follows the planned trajectory. In this work, we address this problem by presenting three methods of incorporating stochastic controller behavior into the recombinant search space of the Kinodynamic Efficiently Adaptive State Lattice (KEASL) planner. To demonstrate this work, we analyze the results of experiments performed on a Clearpath Robotics Warthog Unmanned Ground Vehicle (UGV) in an off-road, unstructured environment using two different perception algorithms, and performed an ablation study using a full spectrum of simulated environment map complexities. Analysis of the data found that incorporating stochastic controller sampling into KEASL leads to more conservative trajectories that decrease predicted collision likelihood when compared to KEASL without sampling. When compared to baseline planning with expanded obstacle footprints, the predicted likelihood of collisions becomes more comparable, but reduces the planning success rate for baseline search. 

**Abstract (ZH)**: 移动机器人运动规划器依赖于理论模型来预测机器人如何在环境中移动。然而，在实际部署到物理机器人后，这些模型会因现实世界物理条件和底层控制器跟随计划轨迹的不确定性而出现误差。在这项工作中，我们通过将随机控制器行为纳入基于Kinodynamic Efficiently Adaptive State Lattice（KEASL）规划器的重组搜索空间中，来解决这一问题，并提出了三种方法。为了验证这一工作，我们在Clearpath Robotics Warthog无人地面车辆（UGV）上进行户外、非结构化环境下的实验，使用了两种不同的感知算法，并进行了全谱模拟环境地图复杂度的消融研究。数据分析显示，将随机控制器采样纳入KEASL中可以产生更为保守的轨迹，从而降低预测碰撞的可能性。与基线规划扩展障碍物足迹相比，预测的碰撞可能性更加接近，但降低了基线搜索的成功率。 

---
# Improving Tactile Gesture Recognition with Optical Flow 

**Title (ZH)**: 基于光学流提高触觉手势识别性能 

**Authors**: Shaohong Zhong, Alessandro Albini, Giammarco Caroleo, Giorgio Cannata, Perla Maiolino  

**Link**: [PDF](https://arxiv.org/pdf/2508.04338)  

**Abstract**: Tactile gesture recognition systems play a crucial role in Human-Robot Interaction (HRI) by enabling intuitive communication between humans and robots. The literature mainly addresses this problem by applying machine learning techniques to classify sequences of tactile images encoding the pressure distribution generated when executing the gestures. However, some gestures can be hard to differentiate based on the information provided by tactile images alone. In this paper, we present a simple yet effective way to improve the accuracy of a gesture recognition classifier. Our approach focuses solely on processing the tactile images used as input by the classifier. In particular, we propose to explicitly highlight the dynamics of the contact in the tactile image by computing the dense optical flow. This additional information makes it easier to distinguish between gestures that produce similar tactile images but exhibit different contact dynamics. We validate the proposed approach in a tactile gesture recognition task, showing that a classifier trained on tactile images augmented with optical flow information achieved a 9% improvement in gesture classification accuracy compared to one trained on standard tactile images. 

**Abstract (ZH)**: 触觉手势识别系统在人机交互（HRI）中发挥着关键作用，通过使人类和机器人之间的通信直观化。文献主要通过应用机器学习技术来对编码执行手势时产生的压力分布的触觉图像序列进行分类来解决这一问题。然而，仅凭触觉图像的信息，某些手势可能会难以区分。在本文中，我们提出了一种简单而有效的方法，以提高手势识别分类器的准确率。我们的方法仅专注于处理用于输入分类器的触觉图像。具体来说，我们建议通过计算密集型光学流来显式地突出触觉图像中的接触动态。这种额外的信息使得更容易区分产生相似触觉图像但接触动态不同的手势。我们在触觉手势识别任务中验证了所提出的方法，结果显示，与仅使用标准触觉图像训练的分类器相比，使用包含光学流信息的触觉图像训练的分类器在手势分类准确率上提高了9%。 

---
# Industrial Robot Motion Planning with GPUs: Integration of cuRobo for Extended DOF Systems 

**Title (ZH)**: 基于GPU的工业机器人运动规划：cuRobo在扩展自由度系统中的集成 

**Authors**: Luai Abuelsamen, Harsh Rana, Ho-Wei Lu, Wenhan Tang, Swati Priyadarshini, Gabriel Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2508.04146)  

**Abstract**: Efficient motion planning remains a key challenge in industrial robotics, especially for multi-axis systems operating in complex environments. This paper addresses that challenge by integrating GPU-accelerated motion planning through NVIDIA's cuRobo library into Vention's modular automation platform. By leveraging accurate CAD-based digital twins and real-time parallel optimization, our system enables rapid trajectory generation and dynamic collision avoidance for pick-and-place tasks. We demonstrate this capability on robots equipped with additional degrees of freedom, including a 7th-axis gantry, and benchmark performance across various scenarios. The results show significant improvements in planning speed and robustness, highlighting the potential of GPU-based planning pipelines for scalable, adaptable deployment in modern industrial workflows. 

**Abstract (ZH)**: 高效的运动规划仍然是工业机器人领域的关键挑战，特别是在复杂环境中操作的多轴系统中。本文通过将NVIDIA cuRobo库的GPU加速运动规划技术集成到Vention的模块化自动化平台上，解决了这一挑战。通过利用基于准确的CAD数字孪生和实时并行优化，我们的系统能够快速生成轨迹并实现动态碰撞避免，以支持拾取和放置任务。我们在具有额外自由度的机器人上展示了这种能力，包括7轴龙门架，并在各种场景下进行了性能基准测试。结果表明，在规划速度和鲁棒性方面有显著改进，突显了基于GPU的规划流水线在现代工业工作流程中可扩展和灵活部署的潜力。 

---
# Optimization of sliding control parameters for a 3-dof robot arm using genetic algorithm (GA) 

**Title (ZH)**: 使用遗传算法（GA）优化三自由度机器人手臂滑动控制参数 

**Authors**: Vu Ngoc Son, Pham Van Cuong, Dao Thi My Linh, Le Tieu Nien  

**Link**: [PDF](https://arxiv.org/pdf/2508.04009)  

**Abstract**: This paper presents a method for optimizing the sliding mode control (SMC) parameter for a robot manipulator applying a genetic algorithm (GA). The objective of the SMC is to achieve precise and consistent tracking of the trajectory of the robot manipulator under uncertain and disturbed conditions. However, the system effectiveness and robustness depend on the choice of the SMC parameters, which is a difficult and crucial task. To solve this problem, a genetic algorithm is used to locate the optimal values of these parameters that gratify the capability criteria. The proposed method is efficient compared with the conventional SMC and Fuzzy-SMC. The simulation results show that the genetic algorithm with SMC can achieve better tracking capability and reduce the chattering effect. 

**Abstract (ZH)**: 基于遗传算法优化机器人 manipulator 滑模控制参数的方法 

---
# Position-Based Flocking for Robust Alignment 

**Title (ZH)**: 基于位置的群集同步方法 

**Authors**: Hossein B. Jond  

**Link**: [PDF](https://arxiv.org/pdf/2508.04378)  

**Abstract**: This paper presents a position-based flocking model for interacting agents, balancing cohesion-separation and alignment to achieve stable collective motion. The model modifies a velocity-based approach by approximating velocity differences using initial and current positions, introducing a threshold weight to ensure sustained alignment. Simulations with 50 agents in 2D demonstrate that the position-based model produces stronger alignment and more rigid and compact formations compared to the velocity-based model. The alignment metric and separation distances highlight the efficacy of the proposed model in achieving robust flocking behavior. The model's use of positions ensures robust alignment, with applications in robotics and collective dynamics. 

**Abstract (ZH)**: 基于位置的交互代理群集移动模型：平衡凝聚力与分离以及对齐以实现稳定的集体运动 

---
# From MAS to MARS: Coordination Failures and Reasoning Trade-offs in Hierarchical Multi-Agent Robotic Systems within a Healthcare Scenario 

**Title (ZH)**: 从MAS到MARS：基于医疗保健场景的分层多Agent机器人系统中的协调失败与推理权衡 

**Authors**: Yuanchen Bai, Zijian Ding, Shaoyue Wen, Xiang Chang, Angelique Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2508.04691)  

**Abstract**: Multi-agent robotic systems (MARS) build upon multi-agent systems by integrating physical and task-related constraints, increasing the complexity of action execution and agent coordination. However, despite the availability of advanced multi-agent frameworks, their real-world deployment on robots remains limited, hindering the advancement of MARS research in practice. To bridge this gap, we conducted two studies to investigate performance trade-offs of hierarchical multi-agent frameworks in a simulated real-world multi-robot healthcare scenario. In Study 1, using CrewAI, we iteratively refine the system's knowledge base, to systematically identify and categorize coordination failures (e.g., tool access violations, lack of timely handling of failure reports) not resolvable by providing contextual knowledge alone. In Study 2, using AutoGen, we evaluate a redesigned bidirectional communication structure and further measure the trade-offs between reasoning and non-reasoning models operating within the same robotic team setting. Drawing from our empirical findings, we emphasize the tension between autonomy and stability and the importance of edge-case testing to improve system reliability and safety for future real-world deployment. Supplementary materials, including codes, task agent setup, trace outputs, and annotated examples of coordination failures and reasoning behaviors, are available at: this https URL. 

**Abstract (ZH)**: 多代理 robotic 系统（MARS）通过整合物理和任务相关的约束，增加了行为执行和代理协调的复杂性。然而，尽管有先进的多代理框架，其在机器人上的实际部署仍然受到限制，阻碍了 MARS 研究的实际进展。为了弥合这一差距，我们开展了两项研究，以调查分级多代理框架在模拟的多机器人医疗场景中的性能权衡。在研究 1 中，使用 CrewAI 逐步细化系统的知识库，系统识别和分类由于仅提供上下文知识无法解决的协作失败（例如，工具访问违规、故障报告未及时处理）。在研究 2 中，使用 AutoGen 评估重新设计的双向通信结构，并进一步衡量工作于同一机器人团队中的推理和非推理模型之间的权衡。从我们的实证研究中，我们强调自主性与稳定性的张力以及边缘案例测试的重要性，以提高系统的可靠性和安全性，以便未来实际部署。附加材料，包括代码、任务代理设置、跟踪输出以及协作失败和推理行为的标注示例等，可在以下网址获取：this https URL。 

---
