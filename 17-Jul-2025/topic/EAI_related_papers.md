# Regrasp Maps for Sequential Manipulation Planning 

**Title (ZH)**: 序列操作规划的重抓取地图 

**Authors**: Svetlana Levit, Marc Toussaint  

**Link**: [PDF](https://arxiv.org/pdf/2507.12407)  

**Abstract**: We consider manipulation problems in constrained and cluttered settings, which require several regrasps at unknown locations. We propose to inform an optimization-based task and motion planning (TAMP) solver with possible regrasp areas and grasp sequences to speed up the search. Our main idea is to use a state space abstraction, a regrasp map, capturing the combinations of available grasps in different parts of the configuration space, and allowing us to provide the solver with guesses for the mode switches and additional constraints for the object placements. By interleaving the creation of regrasp maps, their adaptation based on failed refinements, and solving TAMP (sub)problems, we are able to provide a robust search method for challenging regrasp manipulation problems. 

**Abstract (ZH)**: 我们在受约束和拥挤环境中考虑夹持操作问题，这些环境需要在未知位置进行多次重新夹持。我们提出了一种使用可能的重新夹持区域和夹持序列来引导基于优化的任务和运动规划（TAMP）求解器以加速搜索的方法。我们的主要思想是使用状态空间抽象——重新夹持地图，该地图捕捉配置空间不同部分可用夹持方式的组合，从而使求解器能够提供模式切换的猜测和物体放置的附加约束。通过交错生成重新夹持地图、根据失败的细化进行适应以及求解TAMP（子）问题，我们能够提供对具有挑战性的重新夹持操作问题 robust 的搜索方法。 

---
# Assessing the Value of Visual Input: A Benchmark of Multimodal Large Language Models for Robotic Path Planning 

**Title (ZH)**: 视觉输入价值评估：面向机器人路径规划的多模态大型语言模型基准研究 

**Authors**: Jacinto Colan, Ana Davila, Yasuhisa Hasegawa  

**Link**: [PDF](https://arxiv.org/pdf/2507.12391)  

**Abstract**: Large Language Models (LLMs) show potential for enhancing robotic path planning. This paper assesses visual input's utility for multimodal LLMs in such tasks via a comprehensive benchmark. We evaluated 15 multimodal LLMs on generating valid and optimal paths in 2D grid environments, simulating simplified robotic planning, comparing text-only versus text-plus-visual inputs across varying model sizes and grid complexities. Our results indicate moderate success rates on simpler small grids, where visual input or few-shot text prompting offered some benefits. However, performance significantly degraded on larger grids, highlighting a scalability challenge. While larger models generally achieved higher average success, the visual modality was not universally dominant over well-structured text for these multimodal systems, and successful paths on simpler grids were generally of high quality. These results indicate current limitations in robust spatial reasoning, constraint adherence, and scalable multimodal integration, identifying areas for future LLM development in robotic path planning. 

**Abstract (ZH)**: 大型语言模型（LLMs）在增强机器人路径规划方面显示出潜力。本文通过全面的基准测试评估了多模态LLMs在路径规划任务中利用视觉输入的效果。我们在2D网格环境中评估了15个多模态LLMs生成有效和最优路径的能力，模拟简化版的机器人规划过程，比较了仅文本输入与结合视觉输入的情况，考察了不同模型大小和网格复杂度的表现。结果显示，在较小的简单网格上，视觉输入或少量提示文本能够提供一定程度的好处，但在较大网格上性能显著下降，突显了可扩展性的挑战。尽管大型模型通常实现了更高的平均成功率，但视觉模态在这些多模态系统中并非普遍优于结构良好的文本，并且在小网格上成功路径通常质量较高。这些结果表明当前在稳健的空间推理、约束遵守以及可扩展的多模态集成方面的局限性，指出了未来LLM在机器人路径规划方面的开发方向。 

---
# Next-Gen Museum Guides: Autonomous Navigation and Visitor Interaction with an Agentic Robot 

**Title (ZH)**: 下一代博物馆导览机器人：自主导航与有能动性的游客互动 

**Authors**: Luca Garello, Francesca Cocchella, Alessandra Sciutti, Manuel Catalano, Francesco Rea  

**Link**: [PDF](https://arxiv.org/pdf/2507.12273)  

**Abstract**: Autonomous robots are increasingly being tested into public spaces to enhance user experiences, particularly in cultural and educational settings. This paper presents the design, implementation, and evaluation of the autonomous museum guide robot Alter-Ego equipped with advanced navigation and interactive capabilities. The robot leverages state-of-the-art Large Language Models (LLMs) to provide real-time, context aware question-and-answer (Q&A) interactions, allowing visitors to engage in conversations about exhibits. It also employs robust simultaneous localization and mapping (SLAM) techniques, enabling seamless navigation through museum spaces and route adaptation based on user requests. The system was tested in a real museum environment with 34 participants, combining qualitative analysis of visitor-robot conversations and quantitative analysis of pre and post interaction surveys. Results showed that the robot was generally well-received and contributed to an engaging museum experience, despite some limitations in comprehension and responsiveness. This study sheds light on HRI in cultural spaces, highlighting not only the potential of AI-driven robotics to support accessibility and knowledge acquisition, but also the current limitations and challenges of deploying such technologies in complex, real-world environments. 

**Abstract (ZH)**: 自主机器人在公共空间中的测试及其在博物馆导览中的设计、实现与评估：基于先进导航和交互能力的Alter-Ego自主博物馆导游机器人的研究 

---
# Probabilistic Safety Verification for an Autonomous Ground Vehicle: A Situation Coverage Grid Approach 

**Title (ZH)**: 自主地面车辆的概率安全性验证：一种情境覆盖率网格方法 

**Authors**: Nawshin Mannan Proma, Gricel Vázquez, Sepeedeh Shahbeigi, Arjun Badyal, Victoria Hodge  

**Link**: [PDF](https://arxiv.org/pdf/2507.12158)  

**Abstract**: As industrial autonomous ground vehicles are increasingly deployed in safety-critical environments, ensuring their safe operation under diverse conditions is paramount. This paper presents a novel approach for their safety verification based on systematic situation extraction, probabilistic modelling and verification. We build upon the concept of a situation coverage grid, which exhaustively enumerates environmental configurations relevant to the vehicle's operation. This grid is augmented with quantitative probabilistic data collected from situation-based system testing, capturing probabilistic transitions between situations. We then generate a probabilistic model that encodes the dynamics of both normal and unsafe system behaviour. Safety properties extracted from hazard analysis and formalised in temporal logic are verified through probabilistic model checking against this model. The results demonstrate that our approach effectively identifies high-risk situations, provides quantitative safety guarantees, and supports compliance with regulatory standards, thereby contributing to the robust deployment of autonomous systems. 

**Abstract (ZH)**: 基于系统情况提取、概率建模与验证的工业自主地面车辆安全性验证新方法 

---
# Leveraging Sidewalk Robots for Walkability-Related Analyses 

**Title (ZH)**: 利用人行道机器人进行无障碍性相关分析 

**Authors**: Xing Tong, Michele D. Simoni, Kaj Munhoz Arfvidsson, Jonas Mårtensson  

**Link**: [PDF](https://arxiv.org/pdf/2507.12148)  

**Abstract**: Walkability is a key component of sustainable urban development, while collecting detailed data on its related features remains challenging due to the high costs and limited scalability of traditional methods. Sidewalk delivery robots, increasingly deployed in urban environments, offer a promising solution to these limitations. This paper explores how these robots can serve as mobile data collection platforms, capturing sidewalk-level features related to walkability in a scalable, automated, and real-time manner. A sensor-equipped robot was deployed on a sidewalk network at KTH in Stockholm, completing 101 trips covering 900 segments. From the collected data, different typologies of features are derived, including robot trip characteristics (e.g., speed, duration), sidewalk conditions (e.g., width, surface unevenness), and sidewalk utilization (e.g., pedestrian density). Their walkability-related implications were investigated with a series of analyses. The results demonstrate that pedestrian movement patterns are strongly influenced by sidewalk characteristics, with higher density, reduced width, and surface irregularity associated with slower and more variable trajectories. Notably, robot speed closely mirrors pedestrian behavior, highlighting its potential as a proxy for assessing pedestrian dynamics. The proposed framework enables continuous monitoring of sidewalk conditions and pedestrian behavior, contributing to the development of more walkable, inclusive, and responsive urban environments. 

**Abstract (ZH)**: 步行性是可持续城市发展的重要组成部分，而收集其相关特征的详细数据由于传统方法成本高且难以扩展现状仍具有挑战性。在城市环境中越来越多部署的步行机器人提供了这些限制的 promising 解决方案。本文探讨了这些机器人如何作为移动数据收集平台，以可扩展、自动化和实时的方式捕获与步行性相关的街头特征。一台配备传感器的机器人在斯德哥尔摩皇家理工学院的街头网络上完成了101次行程，覆盖了900个路段。从收集的数据中，提取了不同类型的特征，包括机器人行程特征（如速度、持续时间）、街道条件（如宽度、表面不平整性）和街道使用情况（如行人密度）。一系列分析探讨了这些特征与步行性相关的含义。结果显示，行人移动模式受街头特征强烈影响，高密度、狭窄宽度和表面不平整性与更慢且更不稳定的轨迹相关。值得注意的是，机器人速度与行人行为高度一致，凸显了其作为评估行人动态代理的潜力。所提出的方法框架能够持续监测街头条件和行人行为，有助于开发更具步行性、包容性和响应性的城市环境。 

---
# Tree-SLAM: semantic object SLAM for efficient mapping of individual trees in orchards 

**Title (ZH)**: 树SLAM：具有高效单株树木映射的语义对象SLAM 

**Authors**: David Rapado-Rincon, Gert Kootstra  

**Link**: [PDF](https://arxiv.org/pdf/2507.12093)  

**Abstract**: Accurate mapping of individual trees is an important component for precision agriculture in orchards, as it allows autonomous robots to perform tasks like targeted operations or individual tree monitoring. However, creating these maps is challenging because GPS signals are often unreliable under dense tree canopies. Furthermore, standard Simultaneous Localization and Mapping (SLAM) approaches struggle in orchards because the repetitive appearance of trees can confuse the system, leading to mapping errors. To address this, we introduce Tree-SLAM, a semantic SLAM approach tailored for creating maps of individual trees in orchards. Utilizing RGB-D images, our method detects tree trunks with an instance segmentation model, estimates their location and re-identifies them using a cascade-graph-based data association algorithm. These re-identified trunks serve as landmarks in a factor graph framework that integrates noisy GPS signals, odometry, and trunk observations. The system produces maps of individual trees with a geo-localization error as low as 18 cm, which is less than 20\% of the planting distance. The proposed method was validated on diverse datasets from apple and pear orchards across different seasons, demonstrating high mapping accuracy and robustness in scenarios with unreliable GPS signals. 

**Abstract (ZH)**: 果园单个树木的精准建图是精准耕作的重要组成部分，可以使得自主机器人执行如定向操作或单个树木监测等任务。然而，创建这些地图颇具挑战性，因为GPS信号在密集树冠下往往不可靠。此外，标准的同步定位与建图（SLAM）方法在果园中表现不佳，因为树木的重复外观容易使系统产生混淆，导致建图错误。为解决这一问题，我们提出了Tree-SLAM，这是一种针对果园中单个树木建图的语义SLAM方法。利用RGB-D图像，我们的方法通过实例分割模型检测树干，使用基于cascade-graph的数据关联算法估计其位置并重新识别它们。这些重新识别的树干作为地标，在结合噪声GPS信号、里程计和树干观测的因子图框架中发挥作用。系统产生的单个树木地图的地理定位误差低至18厘米，小于种植间距的20%。所提出的方法在不同季节来自苹果和梨园的多种数据集上得到了验证，展现了在GPS信号不可靠场景中的高建图准确性和 robustness。 

---
# Robust Planning for Autonomous Vehicles with Diffusion-Based Failure Samplers 

**Title (ZH)**: 基于扩散型故障采样的鲁棒自主车辆规划 

**Authors**: Juanran Wang, Marc R. Schlichting, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2507.11991)  

**Abstract**: High-risk traffic zones such as intersections are a major cause of collisions. This study leverages deep generative models to enhance the safety of autonomous vehicles in an intersection context. We train a 1000-step denoising diffusion probabilistic model to generate collision-causing sensor noise sequences for an autonomous vehicle navigating a four-way intersection based on the current relative position and velocity of an intruder. Using the generative adversarial architecture, the 1000-step model is distilled into a single-step denoising diffusion model which demonstrates fast inference speed while maintaining similar sampling quality. We demonstrate one possible application of the single-step model in building a robust planner for the autonomous vehicle. The planner uses the single-step model to efficiently sample potential failure cases based on the currently measured traffic state to inform its decision-making. Through simulation experiments, the robust planner demonstrates significantly lower failure rate and delay rate compared with the baseline Intelligent Driver Model controller. 

**Abstract (ZH)**: 高风险交通区域（如交叉口）是碰撞的主要原因。本研究利用深度生成模型在交叉口场景中提高自动驾驶车辆的安全性。我们基于当前侵入物的相对位置和速度，训练一个1000步去噪扩散概率模型，生成导致碰撞的传感器噪声序列，用于自动驾驶车辆在四向交叉口的行驶。通过生成对抗性架构，1000步模型被精简为一个单步去噪扩散模型，该模型在保持类似采样质量的同时，展现出更快的推理速度。我们展示了一步模型的一种可能应用，即构建一个 robust 的规划器，该规划器利用一步模型根据当前测量的交通状态高效地采样潜在故障案例，以指导其决策。通过仿真实验，robust 规划器显示出显著更低的故障率和延迟率，优于基准智能驾驶员模型控制器。 

---
# A Multi-Level Similarity Approach for Single-View Object Grasping: Matching, Planning, and Fine-Tuning 

**Title (ZH)**: 单视图物体抓取的多层级相似性方法：匹配、规划与精细调整 

**Authors**: Hao Chen, Takuya Kiyokawa, Zhengtao Hu, Weiwei Wan, Kensuke Harada  

**Link**: [PDF](https://arxiv.org/pdf/2507.11938)  

**Abstract**: Grasping unknown objects from a single view has remained a challenging topic in robotics due to the uncertainty of partial observation. Recent advances in large-scale models have led to benchmark solutions such as GraspNet-1Billion. However, such learning-based approaches still face a critical limitation in performance robustness for their sensitivity to sensing noise and environmental changes. To address this bottleneck in achieving highly generalized grasping, we abandon the traditional learning framework and introduce a new perspective: similarity matching, where similar known objects are utilized to guide the grasping of unknown target objects. We newly propose a method that robustly achieves unknown-object grasping from a single viewpoint through three key steps: 1) Leverage the visual features of the observed object to perform similarity matching with an existing database containing various object models, identifying potential candidates with high similarity; 2) Use the candidate models with pre-existing grasping knowledge to plan imitative grasps for the unknown target object; 3) Optimize the grasp quality through a local fine-tuning process. To address the uncertainty caused by partial and noisy observation, we propose a multi-level similarity matching framework that integrates semantic, geometric, and dimensional features for comprehensive evaluation. Especially, we introduce a novel point cloud geometric descriptor, the C-FPFH descriptor, which facilitates accurate similarity assessment between partial point clouds of observed objects and complete point clouds of database models. In addition, we incorporate the use of large language models, introduce the semi-oriented bounding box, and develop a novel point cloud registration approach based on plane detection to enhance matching accuracy under single-view conditions. Videos are available at this https URL. 

**Abstract (ZH)**: 基于单视图抓取未知物体：通过相似性匹配实现高度泛化的未知物体抓取 

---
# Hybrid Conformal Prediction-based Risk-Aware Model Predictive Planning in Dense, Uncertain Environments 

**Title (ZH)**: 基于混合保角预测的风险意识模型预测规划在密集、不确定环境中的应用 

**Authors**: Jeongyong Yang, KwangBin Lee, SooJean Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.11920)  

**Abstract**: Real-time path planning in dense, uncertain environments remains a challenging problem, as predicting the future motions of numerous dynamic obstacles is computationally burdensome and unrealistic. To address this, we introduce Hybrid Prediction-based Risk-Aware Planning (HyPRAP), a prediction-based risk-aware path-planning framework which uses a hybrid combination of models to predict local obstacle movement. HyPRAP uses a novel Prediction-based Collision Risk Index (P-CRI) to evaluate the risk posed by each obstacle, enabling the selective use of predictors based on whether the agent prioritizes high predictive accuracy or low computational prediction overhead. This selective routing enables the agent to focus on high-risk obstacles while ignoring or simplifying low-risk ones, making it suitable for environments with a large number of obstacles. Moreover, HyPRAP incorporates uncertainty quantification through hybrid conformal prediction by deriving confidence bounds simultaneously achieved by multiple predictions across different models. Theoretical analysis demonstrates that HyPRAP effectively balances safety and computational efficiency by leveraging the diversity of prediction models. Extensive simulations validate these insights for more general settings, confirming that HyPRAP performs better compared to single predictor methods, and P-CRI performs better over naive proximity-based risk assessment. 

**Abstract (ZH)**: 基于混合预测的风险感知实时路径规划 

---
# The Developments and Challenges towards Dexterous and Embodied Robotic Manipulation: A Survey 

**Title (ZH)**: 面向灵巧操控与 embodied 机器人 manipulation 的发展与挑战：一篇综述 

**Authors**: Gaofeng Li, Ruize Wang, Peisen Xu, Qi Ye, Jiming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.11840)  

**Abstract**: Achieving human-like dexterous robotic manipulation remains a central goal and a pivotal challenge in robotics. The development of Artificial Intelligence (AI) has allowed rapid progress in robotic manipulation. This survey summarizes the evolution of robotic manipulation from mechanical programming to embodied intelligence, alongside the transition from simple grippers to multi-fingered dexterous hands, outlining key characteristics and main challenges. Focusing on the current stage of embodied dexterous manipulation, we highlight recent advances in two critical areas: dexterous manipulation data collection (via simulation, human demonstrations, and teleoperation) and skill-learning frameworks (imitation and reinforcement learning). Then, based on the overview of the existing data collection paradigm and learning framework, three key challenges restricting the development of dexterous robotic manipulation are summarized and discussed. 

**Abstract (ZH)**: 实现类人灵巧机器人操作仍是以机器人学为核心并构成关键挑战的目标。随着人工智能的发展，机器人操作取得了快速进步。本文综述了从机械编程到具身智能的机器人操作演变，以及从单一手指夹持器到多指灵巧手的过渡，概述了关键特征和主要挑战。聚焦于当前的具身灵巧操作阶段，我们强调了两个关键领域的最新进展：灵巧操作数据收集（通过仿真、人类示范和遥控操作）以及技能学习框架（模仿学习和强化学习）。在此基础上，根据现有的数据收集范式和学习框架的综述，总结并讨论了三个限制灵巧机器人操作发展的关键挑战。 

---
# Generating Actionable Robot Knowledge Bases by Combining 3D Scene Graphs with Robot Ontologies 

**Title (ZH)**: 基于结合3D场景图与机器人本体的知识生成可操作的机器人知识库 

**Authors**: Giang Nguyen, Mihai Pomarlan, Sascha Jongebloed, Nils Leusmann, Minh Nhat Vu, Michael Beetz  

**Link**: [PDF](https://arxiv.org/pdf/2507.11770)  

**Abstract**: In robotics, the effective integration of environmental data into actionable knowledge remains a significant challenge due to the variety and incompatibility of data formats commonly used in scene descriptions, such as MJCF, URDF, and SDF. This paper presents a novel approach that addresses these challenges by developing a unified scene graph model that standardizes these varied formats into the Universal Scene Description (USD) format. This standardization facilitates the integration of these scene graphs with robot ontologies through semantic reporting, enabling the translation of complex environmental data into actionable knowledge essential for cognitive robotic control. We evaluated our approach by converting procedural 3D environments into USD format, which is then annotated semantically and translated into a knowledge graph to effectively answer competency questions, demonstrating its utility for real-time robotic decision-making. Additionally, we developed a web-based visualization tool to support the semantic mapping process, providing users with an intuitive interface to manage the 3D environment. 

**Abstract (ZH)**: 在机器人技术中，将场景描述中常见的MJCF、URDF和SDF等多样且不兼容的数据格式有效整合为可操作的知识仍然是一项重大挑战。本文提出了一种新颖的方法，通过开发一个统一的场景图模型，将这些多样化的格式标准化为通用场景描述(USD)格式，从而解决了这些挑战。这种标准化促进了场景图与机器人本体的结合，通过语义报告实现了复杂环境数据向认知机器人控制所需可操作知识的翻译。我们通过将过程化3D环境转换为USD格式进行评估，随后对其进行语义标注并转化为知识图谱，以有效回答能力问题，展示了其在实时机器人决策中的实用性。此外，我们开发了一个基于网络的可视化工具，以支持语义映射过程，为用户提供了一个直观的界面来管理3D环境。 

---
# Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics 

**Title (ZH)**: 运动中的远见：结合奖励启发式方法强化轨迹预测 

**Authors**: Muleilan Pei, Shaoshuai Shi, Xuesong Chen, Xu Liu, Shaojie Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.12083)  

**Abstract**: Motion forecasting for on-road traffic agents presents both a significant challenge and a critical necessity for ensuring safety in autonomous driving systems. In contrast to most existing data-driven approaches that directly predict future trajectories, we rethink this task from a planning perspective, advocating a "First Reasoning, Then Forecasting" strategy that explicitly incorporates behavior intentions as spatial guidance for trajectory prediction. To achieve this, we introduce an interpretable, reward-driven intention reasoner grounded in a novel query-centric Inverse Reinforcement Learning (IRL) scheme. Our method first encodes traffic agents and scene elements into a unified vectorized representation, then aggregates contextual features through a query-centric paradigm. This enables the derivation of a reward distribution, a compact yet informative representation of the target agent's behavior within the given scene context via IRL. Guided by this reward heuristic, we perform policy rollouts to reason about multiple plausible intentions, providing valuable priors for subsequent trajectory generation. Finally, we develop a hierarchical DETR-like decoder integrated with bidirectional selective state space models to produce accurate future trajectories along with their associated probabilities. Extensive experiments on the large-scale Argoverse and nuScenes motion forecasting datasets demonstrate that our approach significantly enhances trajectory prediction confidence, achieving highly competitive performance relative to state-of-the-art methods. 

**Abstract (ZH)**: 基于规划视角的道路交通代理运动预测：一种“先推理后预测”的策略 

---
# Online Training and Pruning of Deep Reinforcement Learning Networks 

**Title (ZH)**: 在线训练与裁剪深度强化学习网络 

**Authors**: Valentin Frank Ingmar Guenter, Athanasios Sideris  

**Link**: [PDF](https://arxiv.org/pdf/2507.11975)  

**Abstract**: Scaling deep neural networks (NN) of reinforcement learning (RL) algorithms has been shown to enhance performance when feature extraction networks are used but the gained performance comes at the significant expense of increased computational and memory complexity. Neural network pruning methods have successfully addressed this challenge in supervised learning. However, their application to RL is underexplored. We propose an approach to integrate simultaneous training and pruning within advanced RL methods, in particular to RL algorithms enhanced by the Online Feature Extractor Network (OFENet). Our networks (XiNet) are trained to solve stochastic optimization problems over the RL networks' weights and the parameters of variational Bernoulli distributions for 0/1 Random Variables $\xi$ scaling each unit in the networks. The stochastic problem formulation induces regularization terms that promote convergence of the variational parameters to 0 when a unit contributes little to the performance. In this case, the corresponding structure is rendered permanently inactive and pruned from its network. We propose a cost-aware, sparsity-promoting regularization scheme, tailored to the DenseNet architecture of OFENets expressing the parameter complexity of involved networks in terms of the parameters of the RVs in these networks. Then, when matching this cost with the regularization terms, the many hyperparameters associated with them are automatically selected, effectively combining the RL objectives and network compression. We evaluate our method on continuous control benchmarks (MuJoCo) and the Soft Actor-Critic RL agent, demonstrating that OFENets can be pruned considerably with minimal loss in performance. Furthermore, our results confirm that pruning large networks during training produces more efficient and higher performing RL agents rather than training smaller networks from scratch. 

**Abstract (ZH)**: 加强深度神经网络（NN）在强化学习（RL）算法中的应用已被证明能提升性能，但在使用特征提取网络时，这种性能提升伴随着计算和内存复杂度的显著增加。神经网络裁剪方法已经在监督学习中成功地解决了这一挑战。然而，它们在RL中的应用尚未得到充分探索。我们提出了一种在高级RL方法中结合同时训练和裁剪的方法，特别适用于通过在线特征提取网络（OFENet）增强的RL算法。我们的网络（XiNet）被训练以解决RL网络权重和0/1随机变量ξ的变分伯努利分布参数下的随机优化问题，这些随机变量用于缩放网络中的每个单元。随机问题的形式化推导出正则化项，当一个单元对性能贡献很少时，促进变分参数收敛于0。在这种情况下，相应的结构被永久性地禁用并从其网络中裁剪。我们提出了一种成本意识的、促进稀疏性的正则化方案，针对OFENets的DenseNet架构，通过网络中的随机变量参数表达参与网络的参数复杂性。然后，当匹配成本与正则化项时，与它们相关的许多超参数会自动选择，从而有效地结合了RL目标和网络压缩。我们在连续控制基准（MuJoCo）和Soft Actor-Critic RL代理上评估了该方法，表明可以显著裁剪OFENets并几乎不损失性能。此外，我们的结果证实，在训练过程中裁剪大网络会生成更高效和性能更高的RL代理，而不是从头训练较小的网络。 

---
# Emergent Heterogeneous Swarm Control Through Hebbian Learning 

**Title (ZH)**: 通过 Hebbsian 学习实现 Emergent 异质 swarm 控制 

**Authors**: Fuda van Diggelen, Tugay Alperen Karagüzel, Andres Garcia Rincon, A.E. Eiben, Dario Floreano, Eliseo Ferrante  

**Link**: [PDF](https://arxiv.org/pdf/2507.11566)  

**Abstract**: In this paper, we introduce Hebbian learning as a novel method for swarm robotics, enabling the automatic emergence of heterogeneity. Hebbian learning presents a biologically inspired form of neural adaptation that solely relies on local information. By doing so, we resolve several major challenges for learning heterogeneous control: 1) Hebbian learning removes the complexity of attributing emergent phenomena to single agents through local learning rules, thus circumventing the micro-macro problem; 2) uniform Hebbian learning rules across all swarm members limit the number of parameters needed, mitigating the curse of dimensionality with scaling swarm sizes; and 3) evolving Hebbian learning rules based on swarm-level behaviour minimises the need for extensive prior knowledge typically required for optimising heterogeneous swarms. This work demonstrates that with Hebbian learning heterogeneity naturally emerges, resulting in swarm-level behavioural switching and in significantly improved swarm capabilities. It also demonstrates how the evolution of Hebbian learning rules can be a valid alternative to Multi Agent Reinforcement Learning in standard benchmarking tasks. 

**Abstract (ZH)**: 在本文中，我们介绍了一种新的群机器人学习方法——Hebbian学习，以自动实现异质性的自然涌现。Hebbian学习提供了一种生物启发式的神经适应形式，仅依赖局部信息。通过这种方式，我们解决了学习异质控制的几个主要挑战：1) Hebbian学习通过局部学习规则消除了将涌现现象归因于单一代理的复杂性，从而规避了微观-宏观问题；2) 所有群成员统一的Hebbian学习规则限制了所需参数的数量，随着群规模的扩大缓解了维度灾难；3) 基于群层面行为演变Hebbian学习规则减少了对大量先验知识的依赖，通常这些知识对于优化异质群而言是必要的。本研究证明，通过Hebbian学习，异质性自然涌现，导致群级别的行为切换，并显著提高了群的性能。此外，本研究还展示了Hebbian学习规则的演变可以作为一种有效的替代方法，与标准基准任务中的多代理强化学习竞争。 

---
# Understanding visual attention beehind bee-inspired UAV navigation 

**Title (ZH)**: 理解基于蜂群启发的无人飞行器导航中的视觉注意力机制 

**Authors**: Pranav Rajbhandari, Abhi Veda, Matthew Garratt, Mandayam Srinivasan, Sridhar Ravi  

**Link**: [PDF](https://arxiv.org/pdf/2507.11992)  

**Abstract**: Bio-inspired design is often used in autonomous UAV navigation due to the capacity of biological systems for flight and obstacle avoidance despite limited sensory and computational capabilities. In particular, honeybees mainly use the sensory input of optic flow, the apparent motion of objects in their visual field, to navigate cluttered environments. In our work, we train a Reinforcement Learning agent to navigate a tunnel with obstacles using only optic flow as sensory input. We inspect the attention patterns of trained agents to determine the regions of optic flow on which they primarily base their motor decisions. We find that agents trained in this way pay most attention to regions of discontinuity in optic flow, as well as regions with large optic flow magnitude. The trained agents appear to navigate a cluttered tunnel by avoiding the obstacles that produce large optic flow, while maintaining a centered position in their environment, which resembles the behavior seen in flying insects. This pattern persists across independently trained agents, which suggests that this could be a good strategy for developing a simple explicit control law for physical UAVs. 

**Abstract (ZH)**: 生物启发设计在自主无人机导航中的应用：基于光学流的隧道导航研究成果 

---
# EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos 

**Title (ZH)**: 自视角语言行动模型：从自视角人类视频中学习Vision-Language-Action模型 

**Authors**: Ruihan Yang, Qinxi Yu, Yecheng Wu, Rui Yan, Borui Li, An-Chieh Cheng, Xueyan Zou, Yunhao Fang, Hongxu Yin, Sifei Liu, Song Han, Yao Lu, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12440)  

**Abstract**: Real robot data collection for imitation learning has led to significant advancements in robotic manipulation. However, the requirement for robot hardware in the process fundamentally constrains the scale of the data. In this paper, we explore training Vision-Language-Action (VLA) models using egocentric human videos. The benefit of using human videos is not only for their scale but more importantly for the richness of scenes and tasks. With a VLA trained on human video that predicts human wrist and hand actions, we can perform Inverse Kinematics and retargeting to convert the human actions to robot actions. We fine-tune the model using a few robot manipulation demonstrations to obtain the robot policy, namely EgoVLA. We propose a simulation benchmark called Isaac Humanoid Manipulation Benchmark, where we design diverse bimanual manipulation tasks with demonstrations. We fine-tune and evaluate EgoVLA with Isaac Humanoid Manipulation Benchmark and show significant improvements over baselines and ablate the importance of human data. Videos can be found on our website: this https URL 

**Abstract (ZH)**: 基于人类视角视频训练Vision-Language-Action模型以实现仿学习和机器人操作 

---
# Quantum Machine Learning in Multi-Qubit Phase-Space Part I: Foundations 

**Title (ZH)**: 多量子位相空间中的量子机器学习 第一部分：基础理论 

**Authors**: Timothy Heightman, Edward Jiang, Ruth Mora-Soto, Maciej Lewenstein, Marcin Płodzień  

**Link**: [PDF](https://arxiv.org/pdf/2507.12117)  

**Abstract**: Quantum machine learning (QML) seeks to exploit the intrinsic properties of quantum mechanical systems, including superposition, coherence, and quantum entanglement for classical data processing. However, due to the exponential growth of the Hilbert space, QML faces practical limits in classical simulations with the state-vector representation of quantum system. On the other hand, phase-space methods offer an alternative by encoding quantum states as quasi-probability functions. Building on prior work in qubit phase-space and the Stratonovich-Weyl (SW) correspondence, we construct a closed, composable dynamical formalism for one- and many-qubit systems in phase-space. This formalism replaces the operator algebra of the Pauli group with function dynamics on symplectic manifolds, and recasts the curse of dimensionality in terms of harmonic support on a domain that scales linearly with the number of qubits. It opens a new route for QML based on variational modelling over phase-space. 

**Abstract (ZH)**: 量子机器学习（QML）aimed at利用量子力学系统的固有性质，如叠加、相干性和量子纠缠进行经典数据处理。然而，由于希尔伯特空间的指数增长，QML在使用量子系统的态矢量表示进行经典模拟时面临实操限制。另一方面，相空间方法通过将量子态编码为拟概率函数提供了另一种选择。在先前关于量子比特相空间和Stratonovich-Weyl (SW) 对应工作的基础上，我们构建了一种封闭且可组合的动力学形式主义，适用于单量子比特和多量子比特系统。该形式主义用辛流形上的函数动力学取代了Pauli群的算子代数，并将维度灾难重新定义为谐波支持在随量子比特数目线性增长的空间域上的问题。它为基于相空间变分建模的量子机器学习开辟了一条新途径。 

---
