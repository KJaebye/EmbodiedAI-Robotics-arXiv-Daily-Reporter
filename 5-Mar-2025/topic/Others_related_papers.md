# Integral Forms in Matrix Lie Groups 

**Title (ZH)**: 矩阵李群中的整形式 

**Authors**: Timothy D Barfoot  

**Link**: [PDF](https://arxiv.org/pdf/2503.02820)  

**Abstract**: Matrix Lie groups provide a language for describing motion in such fields as robotics, computer vision, and graphics. When using these tools, we are often faced with turning infinite-series expressions into more compact finite series (e.g., the Euler-Rodriques formula), which can sometimes be onerous. In this paper, we identify some useful integral forms in matrix Lie group expressions that offer a more streamlined pathway for computing compact analytic results. Moreover, we present some recursive structures in these integral forms that show many of these expressions are interrelated. Key to our approach is that we are able to apply the minimal polynomial for a Lie algebra quite early in the process to keep expressions compact throughout the derivations. With the series approach, the minimal polynomial is usually applied at the end, making it hard to recognize common analytic expressions in the result. We show that our integral method can reproduce several series-derived results from the literature. 

**Abstract (ZH)**: 矩阵李群为描述机器人学、计算机视觉和图形学中的运动提供了一种语言。在使用这些工具时，我们常常需要将无限级数表达式转化为更紧凑的有限级数（例如欧拉-罗德里奇公式），这有时会带来不便。本文中，我们识别了一些在矩阵李群表达式中有用的积分形式，这些形式为计算紧凑的解析结果提供了更简洁的途径。此外，我们展示了这些积分形式中的递归结构，表明许多这些表达式是相互关联的。我们方法的关键在于能够尽早应用李代数的最小多项式，从而在整个推导过程中保持表达式的紧凑性。使用级数方法时，通常在最后才应用最小多项式，从而难以在结果中识别出常见的解析表达式。我们展示了我们的积分方法可以重现文献中由级数方法得到的多个结果。 

---
# Scalable Multi-Robot Task Allocation and Coordination under Signal Temporal Logic Specifications 

**Title (ZH)**: 基于信号时序逻辑规范的大规模多机器人任务分配与协调 

**Authors**: Wenliang Liu, Nathalie Majcherczyk, Federico Pecora  

**Link**: [PDF](https://arxiv.org/pdf/2503.02719)  

**Abstract**: Motion planning with simple objectives, such as collision-avoidance and goal-reaching, can be solved efficiently using modern planners. However, the complexity of the allowed tasks for these planners is limited. On the other hand, signal temporal logic (STL) can specify complex requirements, but STL-based motion planning and control algorithms often face scalability issues, especially in large multi-robot systems with complex dynamics. In this paper, we propose an algorithm that leverages the best of the two worlds. We first use a single-robot motion planner to efficiently generate a set of alternative reference paths for each robot. Then coordination requirements are specified using STL, which is defined over the assignment of paths and robots' progress along those paths. We use a Mixed Integer Linear Program (MILP) to compute task assignments and robot progress targets over time such that the STL specification is satisfied. Finally, a local controller is used to track the target progress. Simulations demonstrate that our method can handle tasks with complex constraints and scales to large multi-robot teams and intricate task allocation scenarios. 

**Abstract (ZH)**: 基于单机器人运动规划和信号时序逻辑的混合算法在复杂约束与大规模多机器人系统中的应用 

---
# Introspective Loop Closure for SLAM with 4D Imaging Radar 

**Title (ZH)**: 基于4D成像雷达的introspective环回闭合SLAM 

**Authors**: Maximilian Hilger, Vladimír Kubelka, Daniel Adolfsson, Ralf Becker, Henrik Andreasson, Achim J. Lilienthal  

**Link**: [PDF](https://arxiv.org/pdf/2503.02383)  

**Abstract**: Simultaneous Localization and Mapping (SLAM) allows mobile robots to navigate without external positioning systems or pre-existing maps. Radar is emerging as a valuable sensing tool, especially in vision-obstructed environments, as it is less affected by particles than lidars or cameras. Modern 4D imaging radars provide three-dimensional geometric information and relative velocity measurements, but they bring challenges, such as a small field of view and sparse, noisy point clouds. Detecting loop closures in SLAM is critical for reducing trajectory drift and maintaining map accuracy. However, the directional nature of 4D radar data makes identifying loop closures, especially from reverse viewpoints, difficult due to limited scan overlap. This article explores using 4D radar for loop closure in SLAM, focusing on similar and opposing viewpoints. We generate submaps for a denser environment representation and use introspective measures to reject false detections in feature-degenerate environments. Our experiments show accurate loop closure detection in geometrically diverse settings for both similar and opposing viewpoints, improving trajectory estimation with up to 82 % improvement in ATE and rejecting false positives in self-similar environments. 

**Abstract (ZH)**: 4D雷达在SLAM中同时定位与建图中的循环闭合检测 

---
# JPDS-NN: Reinforcement Learning-Based Dynamic Task Allocation for Agricultural Vehicle Routing Optimization 

**Title (ZH)**: JPDS-NN：基于强化学习的农业生产车辆动态任务分配与路径优化 

**Authors**: Yixuan Fan, Haotian Xu, Mengqiao Liu, Qing Zhuo, Tao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02369)  

**Abstract**: The Entrance Dependent Vehicle Routing Problem (EDVRP) is a variant of the Vehicle Routing Problem (VRP) where the scale of cities influences routing outcomes, necessitating consideration of their entrances. This paper addresses EDVRP in agriculture, focusing on multi-parameter vehicle planning for irregularly shaped fields. To address the limitations of traditional methods, such as heuristic approaches, which often overlook field geometry and entrance constraints, we propose a Joint Probability Distribution Sampling Neural Network (JPDS-NN) to effectively solve the EDVRP. The network uses an encoder-decoder architecture with graph transformers and attention mechanisms to model routing as a Markov Decision Process, and is trained via reinforcement learning for efficient and rapid end-to-end planning. Experimental results indicate that JPDS-NN reduces travel distances by 48.4-65.4%, lowers fuel consumption by 14.0-17.6%, and computes two orders of magnitude faster than baseline methods, while demonstrating 15-25% superior performance in dynamic arrangement scenarios. Ablation studies validate the necessity of cross-attention and pre-training. The framework enables scalable, intelligent routing for large-scale farming under dynamic constraints. 

**Abstract (ZH)**: 基于入口依赖的车辆路由问题（EDVRP）在农业中的多参数车辆规划：一种联合概率分布采样神经网络（JPDS-NN）方法 

---
# Uncertainty Representation in a SOTIF-Related Use Case with Dempster-Shafer Theory for LiDAR Sensor-Based Object Detection 

**Title (ZH)**: 基于 Dempster-Shafer 理论的 LiDAR 传感器目标检测中不确定性的表示在SOTIF相关应用场景中 

**Authors**: Milin Patel, Rolf Jung  

**Link**: [PDF](https://arxiv.org/pdf/2503.02087)  

**Abstract**: Uncertainty in LiDAR sensor-based object detection arises from environmental variability and sensor performance limitations. Representing these uncertainties is essential for ensuring the Safety of the Intended Functionality (SOTIF), which focuses on preventing hazards in automated driving scenarios. This paper presents a systematic approach to identifying, classifying, and representing uncertainties in LiDAR-based object detection within a SOTIF-related scenario. Dempster-Shafer Theory (DST) is employed to construct a Frame of Discernment (FoD) to represent detection outcomes. Conditional Basic Probability Assignments (BPAs) are applied based on dependencies among identified uncertainty sources. Yager's Rule of Combination is used to resolve conflicting evidence from multiple sources, providing a structured framework to evaluate uncertainties' effects on detection accuracy. The study applies variance-based sensitivity analysis (VBSA) to quantify and prioritize uncertainties, detailing their specific impact on detection performance. 

**Abstract (ZH)**: LiDAR传感器基于的目标检测中的不确定性源自环境变化和传感器性能限制。在确保功能安全（SOTIF）的场景中，这些不确定性的表示至关重要，SOTIF关注于防止自动驾驶场景中的危险。本文提出了一种系统化的方法，以在与SOTIF相关的情景中识别、分类和表示LiDAR基于的目标检测中的不确定性。文中采用Dempster-Shafer理论（DST）构建区分框架（FoD）来表示检测结果。基于已识别的不确定性来源之间的依赖性，应用条件基本概率分配（BPAs）。使用Yager的证据合成规则来解决来自多个来源的冲突证据，提供了一个结构化的框架，以评估不确定性对检测精度的影响。研究应用方差为基础的敏感性分析（VBSA）来量化和优先排序不确定性，并详细说明了它们对检测性能的具体影响。 

---
# Active Alignments of Lens Systems with Reinforcement Learning 

**Title (ZH)**: 基于强化学习的镜系主动对齐方法 

**Authors**: Matthias Burkhardt, Tobias Schmähling, Michael Layh, Tobias Windisch  

**Link**: [PDF](https://arxiv.org/pdf/2503.02075)  

**Abstract**: Aligning a lens system relative to an imager is a critical challenge in camera manufacturing. While optimal alignment can be mathematically computed under ideal conditions, real-world deviations caused by manufacturing tolerances often render this approach impractical. Measuring these tolerances can be costly or even infeasible, and neglecting them may result in suboptimal alignments. We propose a reinforcement learning (RL) approach that learns exclusively in the pixel space of the sensor output, eliminating the need to develop expert-designed alignment concepts. We conduct an extensive benchmark study and show that our approach surpasses other methods in speed, precision, and robustness. We further introduce relign, a realistic, freely explorable, open-source simulation utilizing physically based rendering that models optical systems with non-deterministic manufacturing tolerances and noise in robotic alignment movement. It provides an interface to popular machine learning frameworks, enabling seamless experimentation and development. Our work highlights the potential of RL in a manufacturing environment to enhance efficiency of optical alignments while minimizing the need for manual intervention. 

**Abstract (ZH)**: 基于像素空间的学习：一种强化学习方法在相机制造中的光学对准应用 

---
# Constraint-Based Modeling of Dynamic Entities in 3D Scene Graphs for Robust SLAM 

**Title (ZH)**: 基于约束的3D场景图中动态实体建模及其在鲁棒SLAM中的应用 

**Authors**: Marco Giberna, Muhammad Shaheer, Hriday Bavle, Jose Andres Millan-Romera, Jose Luis Sanchez-Lopez, Holger Voos  

**Link**: [PDF](https://arxiv.org/pdf/2503.02050)  

**Abstract**: Autonomous robots depend crucially on their ability to perceive and process information from dynamic, ever-changing environments. Traditional simultaneous localization and mapping (SLAM) approaches struggle to maintain consistent scene representations because of numerous moving objects, often treating dynamic elements as outliers rather than explicitly modeling them in the scene representation. In this paper, we present a novel hierarchical 3D scene graph-based SLAM framework that addresses the challenge of modeling and estimating the pose of dynamic objects and agents. We use fiducial markers to detect dynamic entities and to extract their attributes while improving keyframe selection and implementing new capabilities for dynamic entity mapping. We maintain a hierarchical representation where dynamic objects are registered in the SLAM graph and are constrained with robot keyframes and the floor level of the building with our novel entity-keyframe constraints and intra-entity constraints. By combining semantic and geometric constraints between dynamic entities and the environment, our system jointly optimizes the SLAM graph to estimate the pose of the robot and various dynamic agents and objects while maintaining an accurate map. Experimental evaluation demonstrates that our approach achieves a 27.57% reduction in pose estimation error compared to traditional methods and enables higher-level reasoning about scene dynamics. 

**Abstract (ZH)**: 自主机器人依赖于其对动态、不断变化环境中的信息进行感知和处理的能力。传统的同步定位与 Mapping（SLAM）方法难以维持一致的场景表示，因为存在大量移动物体，通常将动态元素视为离群值，而不是在场景表示中显式建模。在本文中，我们提出了一种新颖的分层 3D 场景图基于的 SLAM 框架，以解决动态对象和代理建模和姿态估计的挑战。我们使用标记物检测动态实体并提取其属性，同时改进关键帧选择并实施动态实体映射的新能力。我们维护一个分层表示，其中动态对象被注册在 SLAM 图中，并通过我们新颖的实体-关键帧约束和内部实体约束与机器人关键帧及建筑楼层进行约束。通过结合动态实体与其环境之间的语义和几何约束，我们的系统联合优化 SLAM 图以估计机器人的姿态及各种动态代理和物体的姿态，同时保持准确的映射。实验评估表明，与传统方法相比，我们的方法在姿态估计误差上降低了 27.57%，并能够实现关于场景动态的高级推理。 

---
# Pretrained Embeddings as a Behavior Specification Mechanism 

**Title (ZH)**: 预训练嵌入作为行为规范机制 

**Authors**: Parv Kapoor, Abigail Hammer, Ashish Kapoor, Karen Leung, Eunsuk Kang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02012)  

**Abstract**: We propose an approach to formally specifying the behavioral properties of systems that rely on a perception model for interactions with the physical world. The key idea is to introduce embeddings -- mathematical representations of a real-world concept -- as a first-class construct in a specification language, where properties are expressed in terms of distances between a pair of ideal and observed embeddings. To realize this approach, we propose a new type of temporal logic called Embedding Temporal Logic (ETL), and describe how it can be used to express a wider range of properties about AI-enabled systems than previously possible. We demonstrate the applicability of ETL through a preliminary evaluation involving planning tasks in robots that are driven by foundation models; the results are promising, showing that embedding-based specifications can be used to steer a system towards desirable behaviors. 

**Abstract (ZH)**: 我们提出了一种形式化规定依赖于感知模型的系统行为属性的方法。关键想法是在规范语言中引入嵌入式表示——现实世界概念的数学表示，通过表达理想嵌入和观测嵌入之间的距离来描述系统的性质。为了实现这一方法，我们提出了一种新的时序逻辑类型，称为嵌入时序逻辑(ETL)，并描述了如何使用ETL表达关于AI使能系统的更广泛属性。我们通过涉及由基础模型驱动的机器人执行规划任务的初步评估展示了ETL的适用性，结果表明，基于嵌入的规范可以引导系统朝向期望的行为。 

---
# Uncertainty Comes for Free: Human-in-the-Loop Policies with Diffusion Models 

**Title (ZH)**: 不确定性免费到来：具有扩散模型的人机交互策略 

**Authors**: Zhanpeng He, Yifeng Cao, Matei Ciocarlie  

**Link**: [PDF](https://arxiv.org/pdf/2503.01876)  

**Abstract**: Human-in-the-loop (HitL) robot deployment has gained significant attention in both academia and industry as a semi-autonomous paradigm that enables human operators to intervene and adjust robot behaviors at deployment time, improving success rates. However, continuous human monitoring and intervention can be highly labor-intensive and impractical when deploying a large number of robots. To address this limitation, we propose a method that allows diffusion policies to actively seek human assistance only when necessary, reducing reliance on constant human oversight. To achieve this, we leverage the generative process of diffusion policies to compute an uncertainty-based metric based on which the autonomous agent can decide to request operator assistance at deployment time, without requiring any operator interaction during training. Additionally, we show that the same method can be used for efficient data collection for fine-tuning diffusion policies in order to improve their autonomous performance. Experimental results from simulated and real-world environments demonstrate that our approach enhances policy performance during deployment for a variety of scenarios. 

**Abstract (ZH)**: 循环人类在环中的机器人部署：一种在必要时主动寻求人类协助的方法 

---
# Data Augmentation for Instruction Following Policies via Trajectory Segmentation 

**Title (ZH)**: 基于轨迹分割的指令跟随策略数据增强方法 

**Authors**: Niklas Höpner, Ilaria Tiddi, Herke van Hoof  

**Link**: [PDF](https://arxiv.org/pdf/2503.01871)  

**Abstract**: The scalability of instructable agents in robotics or gaming is often hindered by limited data that pairs instructions with agent trajectories. However, large datasets of unannotated trajectories containing sequences of various agent behaviour (play trajectories) are often available. In a semi-supervised setup, we explore methods to extract labelled segments from play trajectories. The goal is to augment a small annotated dataset of instruction-trajectory pairs to improve the performance of an instruction-following policy trained downstream via imitation learning. Assuming little variation in segment length, recent video segmentation methods can effectively extract labelled segments. To address the constraint of segment length, we propose Play Segmentation (PS), a probabilistic model that finds maximum likely segmentations of extended subsegments, while only being trained on individual instruction segments. Our results in a game environment and a simulated robotic gripper setting underscore the importance of segmentation; randomly sampled segments diminish performance, while incorporating labelled segments from PS improves policy performance to the level of a policy trained on twice the amount of labelled data. 

**Abstract (ZH)**: 指令可执行代理在机器人学或游戏中的可扩展性常受限于指令与代理轨迹配对数据的有限性。然而，通常有大量的未标注轨迹数据包含各种代理行为序列（游戏轨迹）。在半监督设置中，我们探索从游戏轨迹中提取标注片段的方法。目标是扩展一个小型标注数据集，以通过模仿学习增强指令跟随策略的性能。假设片段长度变化不大，近期的视频分割方法可以有效提取标注片段。为了解决片段长度的限制，我们提出了Play Segmentation (PS)，一种概率模型，能够在仅使用个体指令片段进行训练的情况下找到最有可能的片段划分。我们的实验结果在游戏环境和模拟的机器人夹持器设置中强调了分割的重要性；随机抽取的片段降低了性能，而结合PS生成的标注片段可以提升策略性能，使其与在两倍标注数据上训练的策略性能相当。 

---
# Interaction-Aware Model Predictive Decision-Making for Socially-Compliant Autonomous Driving in Mixed Urban Traffic Scenarios 

**Title (ZH)**: 面向社会合规的混合城市交通场景中交互aware模型预测决策控制 

**Authors**: Balint Varga, Thomas Brand, Marcus Schmitz, Ehsan Hashemi  

**Link**: [PDF](https://arxiv.org/pdf/2503.01852)  

**Abstract**: This paper presents the experimental validation of an interaction-aware model predictive decision-making (IAMPDM) approach in the course of a simulator study. The proposed IAMPDM uses a model of the pedestrian, which simultaneously predicts their future trajectories and characterizes the interaction between the pedestrian and the automated vehicle. The main benefit of the proposed concept and the experiment is that the interaction between the pedestrian and the socially compliant autonomous vehicle leads to smoother traffic. Furthermore, the experiment features a novel human-in-the-decision-loop aspect, meaning that the test subjects have no expected behavior or defined sequence of their actions, better imitating real traffic scenarios. Results show that intention-aware decision-making algorithms are more effective in realistic conditions and contribute to smoother traffic flow than state-of-the-art solutions. Furthermore, the findings emphasize the crucial impact of intention-aware decision-making on autonomous vehicle performance in urban areas and the need for further research. 

**Abstract (ZH)**: 本文提出了一种交互感知模型预测决策（IAMPDM）方法在模拟器研究中的实验验证。所提出的IAMPDM使用了一个行人的模型，该模型能够同时预测行人的未来轨迹并表征行人与自动驾驶车辆之间的交互。该研究的主要益处在于，行人与社会适应性的自动驾驶车辆的交互导致了更顺畅的交通流动。此外，实验还包含了一个新颖的人在决策环中的方面，即测试对象没有预设的行为或定义好的行动顺序，更好地模拟了实际的交通场景。结果表明，意图感知的决策算法在现实条件下比现有的解决方案更为有效，并有助于更顺畅的交通流动。此外，研究结果强调了意图感知决策对城市区域自动驾驶车辆性能的至关重要影响，以及需要进一步研究。 

---
# Bringing Comparative Cognition To Computers 

**Title (ZH)**: 将比较认知带入计算机 

**Authors**: Konstantinos Voudouris, Lucy G. Cheke, Eric Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2503.02882)  

**Abstract**: Researchers are increasingly subjecting artificial intelligence systems to psychological testing. But to rigorously compare their cognitive capacities with humans and other animals, we must avoid both over- and under-stating our similarities and differences. By embracing a comparative approach, we can integrate AI cognition research into the broader cognitive sciences. 

**Abstract (ZH)**: 研究人员 increasingly 将人工智能系统置于心理学测试之下。但为了严格比较其认知能力与人类和其他动物的相似性和差异性，我们必须避免夸大或低估这些相似性和差异性。通过采用比较方法，我们可以将人工智能认知研究纳入更广泛的认知科学领域。 

---
# Evaluation of Architectural Synthesis Using Generative AI 

**Title (ZH)**: 使用生成式人工智能进行建筑合成评估 

**Authors**: Jingfei Huang, Alexandros Haridis  

**Link**: [PDF](https://arxiv.org/pdf/2503.02861)  

**Abstract**: Recent advancements in multimodal Generative AI have the potential to democratize specialized architectural tasks, such as interpreting technical drawings and creating 3D CAD models, which traditionally require expert knowledge. This paper presents a comparative evaluation of two systems: GPT-4o and Claude 3.5, in the task of architectural 3D synthesis. We conduct a case study on two buildings from Palladio's Four Books of Architecture (1965): Villa Rotonda and Palazzo Porto. High-level architectural models and drawings of these buildings were prepared, inspired by Palladio's original texts and drawings. Through sequential text and image prompting, we assess the systems' abilities in (1) interpreting 2D and 3D representations of buildings from drawings, (2) encoding the buildings into a CAD software script, and (3) self-improving based on outputs. While both systems successfully generate individual parts, they struggle to accurately assemble these parts into the desired spatial relationships, with Claude 3.5 demonstrating better performance, particularly in self-correcting its output. This study contributes to ongoing research on benchmarking the strengths and weaknesses of off-the-shelf AI systems in performing intelligent human tasks that require discipline-specific knowledge. The findings highlight the potential of language-enabled AI systems to act as collaborative technical assistants in the architectural design process. 

**Abstract (ZH)**: Recent advancements in multimodal Generative AI有潜力使专门的建筑任务民主化，如解读技术图纸和创建3D CAD模型，这些任务 traditionally需要专家知识。本文对GPT-4o和Claude 3.5两个系统在建筑3D合成任务中的表现进行了比较评估。我们以帕拉第奥《四本建筑著作》（1965年）中的Villa Rotonda和Palazzo Porto两座建筑为例进行了案例研究。根据帕拉第奥原始文本和图纸的启发，我们准备了这些建筑的高层建筑模型和图纸。通过顺序的文字和图像提示，我们评估了这些系统在（1）从图纸解读建筑的2D和3D表示，（2）将建筑编码到CAD软件脚本中，以及（3）基于输出自我改进方面的能力。尽管两个系统都能生成独立的部分，但在准确组装这些部分以达到预期的空间关系方面存在困难，Claude 3.5在自我纠正输出方面表现更好。本研究为评估现成AI系统在执行需要学科特定知识的智能人类任务方面的强项和弱点的持续研究做出了贡献。研究结果突显了语言驱动的AI系统在建筑设计过程中的合作技术助手潜力。 

---
# Prime Convolutional Model: Breaking the Ground for Theoretical Explainability 

**Title (ZH)**: 首要卷积模型：理论可解释性的基石 

**Authors**: Francesco Panelli, Doaa Almhaithawi, Tania Cerquitelli, Alessandro Bellini  

**Link**: [PDF](https://arxiv.org/pdf/2503.02773)  

**Abstract**: In this paper, we propose a new theoretical approach to Explainable AI. Following the Scientific Method, this approach consists in formulating on the basis of empirical evidence, a mathematical model to explain and predict the behaviors of Neural Networks. We apply the method to a case study created in a controlled environment, which we call Prime Convolutional Model (p-Conv for short). p-Conv operates on a dataset consisting of the first one million natural numbers and is trained to identify the congruence classes modulo a given integer $m$. Its architecture uses a convolutional-type neural network that contextually processes a sequence of $B$ consecutive numbers to each input. We take an empirical approach and exploit p-Conv to identify the congruence classes of numbers in a validation set using different values for $m$ and $B$. The results show that the different behaviors of p-Conv (i.e., whether it can perform the task or not) can be modeled mathematically in terms of $m$ and $B$. The inferred mathematical model reveals interesting patterns able to explain when and why p-Conv succeeds in performing task and, if not, which error pattern it follows. 

**Abstract (ZH)**: 本文提出了一种新的理论方法来解释AI。该方法基于科学方法，通过对神经网络行为进行基于实证证据的数学建模来进行解释和预测。我们将这种方法应用于一个在受控环境中创建的案例如“质数卷积模型”（简称为p-Conv）。p-Conv处理由前一百万个自然数组成的数据集，并被训练识别给定整数$m$模下的同余类。其架构使用了一种卷积型神经网络，能够对每个输入的$B$个连续数字进行上下文处理。我们采用实证方法，利用p-Conv识别验证集中的数字的同余类，使用不同的$m$和$B$值。结果表明，p-Conv的不同行为（即它能否执行任务）可以用$B$和$m$的数学模型来建模。从推导出的数学模型中，我们可以发现能够解释p-Conv何时以及为何成功执行任务，如果不成功，则遵循何种错误模式的有趣模式。 

---
# Seeding for Success: Skill and Stochasticity in Tabletop Games 

**Title (ZH)**: seeding for Success: 技能与随机性在桌游中的作用 

**Authors**: James Goodman, Diego Perez-Liebana, Simon Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2503.02686)  

**Abstract**: Games often incorporate random elements in the form of dice or shuffled card decks. This randomness is a key contributor to the player experience and the variety of game situations encountered. There is a tension between a level of randomness that makes the game interesting and contributes to the player enjoyment of a game, and a level at which the outcome itself is effectively random and the game becomes dull. The optimal level for a game will depend on the design goals and target audience. We introduce a new technique to quantify the level of randomness in game outcome and use it to compare 15 tabletop games and disentangle the different contributions to the overall randomness from specific parts of some games. We further explore the interaction between game randomness and player skill, and how this innate randomness can affect error analysis in common game experiments. 

**Abstract (ZH)**: 游戏常常通过骰子或洗牌的牌组等随机元素来融入不确定性。这种不确定性是玩家体验和游戏中遇到的各种情况的关键因素。随机性的适宜水平在于既能使游戏有趣并增加玩家的愉悦感，又不至于使得游戏结果完全随机而变得乏味。游戏的最佳随机性水平将取决于设计目标和目标受众。我们提出了一种新的技术来量化游戏结果中的随机性，并利用它来比较15款桌面游戏，并从一些游戏中特定部分的角度解构整体随机性。进一步探讨了游戏随机性与玩家技能之间的互动，以及这种固有的随机性如何影响常见游戏实验中的误差分析。 

---
# AutoEval: A Practical Framework for Autonomous Evaluation of Mobile Agents 

**Title (ZH)**: AutoEval：移动代理自主评估的实用框架 

**Authors**: Jiahui Sun, Zhichao Hua, Yubin Xia  

**Link**: [PDF](https://arxiv.org/pdf/2503.02403)  

**Abstract**: Accurate and systematic evaluation of mobile agents can significantly advance their development and real-world applicability. However, existing benchmarks for mobile agents lack practicality and scalability due to the extensive manual effort required to define task reward signals and implement corresponding evaluation codes. To this end, we propose AutoEval, an autonomous agent evaluation framework that tests a mobile agent without any manual effort. First, we design a Structured Substate Representation to describe the UI state changes while agent execution, such that task reward signals can be automatically generated. Second, we utilize a Judge System that can autonomously evaluate agents' performance given the automatically generated task reward signals. By providing only a task description, our framework evaluates agents with fine-grained performance feedback to that task without any extra manual effort. We implement a prototype of our framework and validate the automatically generated task reward signals, finding over 93% coverage to human-annotated reward signals. Moreover, to prove the effectiveness of our autonomous Judge System, we manually verify its judge results and demonstrate that it achieves 94% accuracy. Finally, we evaluate the state-of-the-art mobile agents using our framework, providing detailed insights into their performance characteristics and limitations. 

**Abstract (ZH)**: 移动代理的自动准确与系统评价可显著促进其发展与实际应用。然而，现有移动代理基准缺乏实用性和可扩展性，因为定义任务奖励信号和实现相应的评价代码需要大量的手工 effort。为此，我们提出 AutoEval，一种无需任何手工努力即可测试移动代理的自主代理评估框架。首先，我们设计了一种结构化子状态表示法，用于描述代理执行过程中的 UI 状态变化，以便能够自动生成任务奖励信号。其次，我们利用一个裁判系统，在给定自动生成的任务奖励信号的情况下，可自主评估代理的性能。仅提供任务描述，我们的框架即可在不进行额外手工努力的情况下，对代理进行细粒度性能反馈评价。我们实现了该框架的一个原型，并验证了自动生成的任务奖励信号，发现其覆盖率超过 93%。此外，为了证明我们自主裁判系统的有效性，我们手工验证了其裁判结果，并证明其准确率达到 94%。最后，我们使用该框架评估最先进的移动代理，提供了对其性能特征和限制的详细见解。 

---
# Enhancing the Product Quality of the Injection Process Using eXplainable Artificial Intelligence 

**Title (ZH)**: 使用可解释的人工智能提升注射成型过程的产品质量 

**Authors**: Jisoo Hong, Yongmin Hong, Jung-Woo Baek, Sung-Woo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02338)  

**Abstract**: The injection molding process is a traditional technique for making products in various industries such as electronics and automobiles via solidifying liquid resin into certain molds. Although the process is not related to creating the main part of engines or semiconductors, this manufacturing methodology sets the final form of the products. Re-cently, research has continued to reduce the defect rate of the injection molding process. This study proposes an optimal injection molding process control system to reduce the defect rate of injection molding products with XAI (eXplainable Artificial Intelligence) ap-proaches. Boosting algorithms (XGBoost and LightGBM) are used as tree-based classifiers for predicting whether each product is normal or defective. The main features to control the process for improving the product are extracted by SHapley Additive exPlanations, while the individual conditional expectation analyzes the optimal control range of these extracted features. To validate the methodology presented in this work, the actual injection molding AI manufacturing dataset provided by KAMP (Korea AI Manufacturing Platform) is employed for the case study. The results reveal that the defect rate decreases from 1.00% (Original defect rate) to 0.21% with XGBoost and 0.13% with LightGBM, respectively. 

**Abstract (ZH)**: 采用XAI方法优化注塑成型工艺控制系统的研究 

---
# KGCompiler: Deep Learning Compilation Optimization for Knowledge Graph Complex Logical Query Answering 

**Title (ZH)**: KGCompiler: 深度学习编译优化以解答知识图谱复杂逻辑查询 

**Authors**: Hongyu Lin, Haoran Luo, Hanghang Cao, Yang Liu, Shihao Gao, Kaichun Yao, Libo Zhang, Mingjie Xing, Yanjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02172)  

**Abstract**: Complex Logical Query Answering (CLQA) involves intricate multi-hop logical reasoning over large-scale and potentially incomplete Knowledge Graphs (KGs). Although existing CLQA algorithms achieve high accuracy in answering such queries, their reasoning time and memory usage scale significantly with the number of First-Order Logic (FOL) operators involved, creating serious challenges for practical deployment. In addition, current research primarily focuses on algorithm-level optimizations for CLQA tasks, often overlooking compiler-level optimizations, which can offer greater generality and scalability. To address these limitations, we introduce a Knowledge Graph Compiler, namely KGCompiler, the first deep learning compiler specifically designed for CLQA tasks. By incorporating KG-specific optimizations proposed in this paper, KGCompiler enhances the reasoning performance of CLQA algorithms without requiring additional manual modifications to their implementations. At the same time, it significantly reduces memory usage. Extensive experiments demonstrate that KGCompiler accelerates CLQA algorithms by factors ranging from 1.04x to 8.26x, with an average speedup of 3.71x. We also provide an interface to enable hands-on experience with KGCompiler. 

**Abstract (ZH)**: 知识图谱复合逻辑查询编译器（KGCompiler）：针对复合逻辑查询回答任务的深度学习编译器 

---
# EPEE: Towards Efficient and Effective Foundation Models in Biomedicine 

**Title (ZH)**: EPEE: 向生物医学领域高效且有效的基础模型迈进 

**Authors**: Zaifu Zhan, Shuang Zhou, Huixue Zhou, Zirui Liu, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02053)  

**Abstract**: Foundation models, including language models, e.g., GPT, and vision models, e.g., CLIP, have significantly advanced numerous biomedical tasks. Despite these advancements, the high inference latency and the "overthinking" issues in model inference impair the efficiency and effectiveness of foundation models, thus limiting their application in real-time clinical settings. To address these challenges, we proposed EPEE (Entropy- and Patience-based Early Exiting), a novel hybrid strategy designed to improve the inference efficiency of foundation models. The core idea was to leverage the strengths of entropy-based and patience-based early exiting methods to overcome their respective weaknesses. To evaluate EPEE, we conducted experiments on three core biomedical tasks-classification, relation extraction, and event extraction-using four foundation models (BERT, ALBERT, GPT-2, and ViT) across twelve datasets, including clinical notes and medical images. The results showed that EPEE significantly reduced inference time while maintaining or improving accuracy, demonstrating its adaptability to diverse datasets and tasks. EPEE addressed critical barriers to deploying foundation models in healthcare by balancing efficiency and effectiveness. It potentially provided a practical solution for real-time clinical decision-making with foundation models, supporting reliable and efficient workflows. 

**Abstract (ZH)**: 基于熵和耐心早期退出的混合策略EPEE：提升基础模型推理效率的方法 

---
# Neural Manifolds and Cognitive Consistency: A New Approach to Memory Consolidation in Artificial Systems 

**Title (ZH)**: 神经流形与认知一致性：人工系统中记忆巩固的新方法 

**Authors**: Phuong-Nam Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01867)  

**Abstract**: We introduce a novel mathematical framework that unifies neural population dynamics, hippocampal sharp wave-ripple (SpWR) generation, and cognitive consistency constraints inspired by Heider's theory. Our model leverages low-dimensional manifold representations to capture structured neural drift and incorporates a balance energy function to enforce coherent synaptic interactions, effectively simulating the memory consolidation processes observed in biological systems. Simulation results demonstrate that our approach not only reproduces key features of SpWR events but also enhances network interpretability. This work paves the way for scalable neuromorphic architectures that bridge neuroscience and artificial intelligence, offering more robust and adaptive learning mechanisms for future intelligent systems. 

**Abstract (ZH)**: 我们提出了一种新的数学框架，该框架统一了神经群体动力学、海马尖波-成簇波（SpWR）生成以及受Heider理论启发的认知一致性约束。该模型利用低维流形表示来捕获结构化的神经漂移，并引入平衡能量函数以强制执行一致的突触交互，有效地模拟了生物系统中观察到的记忆巩固过程。仿真结果表明，我们的方法不仅能够重现SpWR事件的关键特征，还能增强网络的可解释性。这项工作为将神经科学与人工智能结合起来的可扩展神经形态架构铺平了道路，提供了更稳健和适应性强的学习机制，以供未来的智能系统使用。 

---
# Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024 

**Title (ZH)**: Deepfake-Eval-2024：2024年传播的多模态野生深度合成基准 

**Authors**: Nuria Alina Chandra, Ryan Murtfeldt, Lin Qiu, Arnab Karmakar, Hannah Lee, Emmanuel Tanumihardja, Kevin Farhat, Ben Caffee, Sejin Paik, Changyeon Lee, Jongwook Choi, Aerin Kim, Oren Etzioni  

**Link**: [PDF](https://arxiv.org/pdf/2503.02857)  

**Abstract**: In the age of increasingly realistic generative AI, robust deepfake detection is essential for mitigating fraud and disinformation. While many deepfake detectors report high accuracy on academic datasets, we show that these academic benchmarks are out of date and not representative of recent deepfakes. We introduce Deepfake-Eval-2024, a new deepfake detection benchmark consisting of in-the-wild deepfakes collected from social media and deepfake detection platform users in 2024. Deepfake-Eval-2024 consists of 44 hours of videos, 56.5 hours of audio, and 1,975 images, encompassing the latest manipulation technologies. The benchmark contains diverse media content from 88 different websites in 52 different languages. We find that the performance of open-source state-of-the-art deepfake detection models drops precipitously when evaluated on Deepfake-Eval-2024, with AUC decreasing by 50% for video, 48% for audio, and 45% for image models compared to previous benchmarks. We also evaluate commercial deepfake detection models and models finetuned on Deepfake-Eval-2024, and find that they have superior performance to off-the-shelf open-source models, but they do not yet reach the accuracy of human deepfake forensic analysts. The dataset is available at this https URL. 

**Abstract (ZH)**: 在生成式AI日益逼真化的时代，稳健的深度合成检测对于减轻欺诈和虚假信息至关重要。虽然许多深度合成检测器在学术数据集上报告了高准确率，但我们发现这些学术基准已经过时且不具有代表性。我们引入了Deepfake-Eval-2024，这是一个新的深度合成检测基准，包括从社交媒体和2024年深度合成检测平台用户收集的“野生”深度合成媒体。Deepfake-Eval-2024包含44小时的视频、56.5小时的音频和1,975张图像，涵盖了最新的合成技术。基准数据集包含来自52种语言88个不同网站的多样媒体内容。我们发现，当在Deepfake-Eval-2024上评估时，开源的最先进的深度合成检测模型性能急剧下降，视频模型的AUC下降50%，音频模型下降48%，图像模型下降45%。我们还评估了商用深度合成检测模型和在Deepfake-Eval-2024上微调的模型，发现它们的性能优于现成的开源模型，但尚未达到人类深度合成法医分析师的准确度。数据集可在以下链接获取。 

---
# SeqFusion: Sequential Fusion of Pre-Trained Models for Zero-Shot Time-Series Forecasting 

**Title (ZH)**: SeqFusion: 预训练模型的序列融合在零样本时间序列预测中的应用 

**Authors**: Ting-Ji Huang, Xu-Yang Chen, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.02836)  

**Abstract**: Unlike traditional time-series forecasting methods that require extensive in-task data for training, zero-shot forecasting can directly predict future values given a target time series without additional training data. Current zero-shot approaches primarily rely on pre-trained generalized models, with their performance often depending on the variety and relevance of the pre-training data, which can raise privacy concerns. Instead of collecting diverse pre-training data, we introduce SeqFusion in this work, a novel framework that collects and fuses diverse pre-trained models (PTMs) sequentially for zero-shot forecasting. Based on the specific temporal characteristics of the target time series, SeqFusion selects the most suitable PTMs from a batch of pre-collected PTMs, performs sequential predictions, and fuses all the predictions while using minimal data to protect privacy. Each of these PTMs specializes in different temporal patterns and forecasting tasks, allowing SeqFusion to select by measuring distances in a shared representation space of the target time series with each PTM. Experiments demonstrate that SeqFusion achieves competitive accuracy in zero-shot forecasting compared to state-of-the-art methods. 

**Abstract (ZH)**: 不同于传统的时间序列预测方法需要大量的在任务数据进行训练，零样本预测可以直接根据目标时间序列预测未来值而无需额外的训练数据。当前的零样本方法主要依赖于预训练的一般模型，其性能往往取决于预训练数据的多样性和相关性，这可能会引起隐私问题。本工作中，我们引入SeqFusion框架，该框架可以通过序列化地收集和融合预训练模型（PTMs）来进行零样本预测，基于目标时间序列的特定时序特征，SeqFusion从预收集的PTMs中选择最合适的模型进行序列化预测，并在使用 minimal 数据保护隐私的同时融合所有预测。每个PTMs专注于不同的时序模式和预测任务，SeqFusion通过在目标时间序列与每个PTM的共享表示空间中测量距离来选择最合适的模型。实验表明，SeqFusion在零样本预测中的准确性与最先进的方法相当。 

---
# Do Not Trust Licenses You See -- Dataset Compliance Requires Massive-Scale AI-Powered Lifecycle Tracing 

**Title (ZH)**: 不要轻信你看到的许可——数据集合规需要大规模AI驱动的生命周期追踪 

**Authors**: Jaekyeom Kim, Sungryull Sohn, Gerrard Jeongwon Jo, Jihoon Choi, Kyunghoon Bae, Hwayoung Lee, Yongmin Park, Honglak Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.02784)  

**Abstract**: This paper argues that a dataset's legal risk cannot be accurately assessed by its license terms alone; instead, tracking dataset redistribution and its full lifecycle is essential. However, this process is too complex for legal experts to handle manually at scale. Tracking dataset provenance, verifying redistribution rights, and assessing evolving legal risks across multiple stages require a level of precision and efficiency that exceeds human capabilities. Addressing this challenge effectively demands AI agents that can systematically trace dataset redistribution, analyze compliance, and identify legal risks. We develop an automated data compliance system called NEXUS and show that AI can perform these tasks with higher accuracy, efficiency, and cost-effectiveness than human experts. Our massive legal analysis of 17,429 unique entities and 8,072 license terms using this approach reveals the discrepancies in legal rights between the original datasets before redistribution and their redistributed subsets, underscoring the necessity of the data lifecycle-aware compliance. For instance, we find that out of 2,852 datasets with commercially viable individual license terms, only 605 (21%) are legally permissible for commercialization. This work sets a new standard for AI data governance, advocating for a framework that systematically examines the entire lifecycle of dataset redistribution to ensure transparent, legal, and responsible dataset management. 

**Abstract (ZH)**: 本文argues数据集的法律风险不能仅通过其许可条款来准确评估；而是需要跟踪数据集的再分配及其完整生命周期。然而，这一过程对法律专家来说在大规模处理时太过复杂。追踪数据集的来源、验证再分配权利并在多个阶段评估不断变化的法律风险需要超出人类能力的精度和效率。有效应对这一挑战需要能够系统追踪数据集再分配、分析合规性并识别法律风险的AI代理。我们开发了一种名为NEXUS的自动化数据合规系统，并证明AI能够比人类专家以更高的准确率、效率和成本效益执行这些任务。采用这种方法对17,429个独特实体和8,072个许可条款进行大规模法律分析揭示了再分配前的数据集与其再分配子集之间的法律权利差异，强调了数据生命周期意识合规的重要性。例如，我们发现，在2,852个具有商业可行许可条款的数据集中，只有605个（21%）合法可用于商业化。这项工作为AI数据治理设定了新的标准，倡导一种系统地检查数据集再分配全流程的框架，以确保透明、合法和负责任的数据管理。 

---
# Improving Oil Slick Trajectory Simulations with Bayesian Optimization 

**Title (ZH)**: 使用贝叶斯优化改进油污轨迹仿真 

**Authors**: Gabriele Accarino, Marco M. De Carlo, Igor Atake, Donatello Elia, Anusha L. Dissanayake, Antonio Augusto Sepp Neves, Juan Peña Ibañez, Italo Epicoco, Paola Nassisi, Sandro Fiore, Giovanni Coppini  

**Link**: [PDF](https://arxiv.org/pdf/2503.02749)  

**Abstract**: Accurate simulations of oil spill trajectories are essential for supporting practitioners' response and mitigating environmental and socioeconomic impacts. Numerical models, such as MEDSLIK-II, simulate advection, dispersion, and transformation processes of oil particles. However, simulations heavily rely on accurate parameter tuning, still based on expert knowledge and manual calibration. To overcome these limitations, we integrate the MEDSLIK-II numerical oil spill model with a Bayesian optimization framework to iteratively estimate the best physical parameter configuration that yields simulation closer to satellite observations of the slick. We focus on key parameters, such as horizontal diffusivity and drift factor, maximizing the Fraction Skill Score (FSS) as a measure of spatio-temporal overlap between simulated and observed oil distributions. We validate the framework for the Baniyas oil incident that occurred in Syria between August 23 and September 4, 2021, which released over 12,000 $m^3$ of oil. We show that, on average, the proposed approach systematically improves the FSS from 5.82% to 11.07% compared to control simulations initialized with default parameters. The optimization results in consistent improvement across multiple time steps, particularly during periods of increased drift variability, demonstrating the robustness of our method in dynamic environmental conditions. 

**Abstract (ZH)**: 准确模拟油污轨迹对于支持应急响应和减轻环境及经济影响至关重要。我们结合MEDSLIK-II数值油污模型与贝叶斯优化框架，迭代估计最优物理参数配置，使模拟结果更接近于油污的卫星观测数据。我们专注于关键参数，如水平扩散系数和漂流因子，以分数技能评分（FSS）作为模拟和观测油污时空分布重叠的度量。我们对2021年8月23日至9月4日在叙利亚发生的巴尼雅斯油污事件进行了验证，该事件导致超过12,000立方米的原油泄漏。结果显示，与使用默认参数初始化的控制模拟相比，所提出的方法平均提高了FSS，从5.82%提高到11.07%。优化结果在多个时间步长中表现出一致的改进，特别是在漂移可变性增加的时期，证明了该方法在动态环境条件下的鲁棒性。 

---
# Generative Tools for Graphical Assets: Empirical Guidelines based on Game Designers' and Developers' Preferences 

**Title (ZH)**: 图形资产生成工具：基于游戏设计师和开发者偏好的实证指南 

**Authors**: Kaisei Fukaya, Damon Daylamani-Zad, Harry Agius  

**Link**: [PDF](https://arxiv.org/pdf/2503.02703)  

**Abstract**: Graphical assets play an important role in the design and development of games. There is potential in the use of generative tools, to aid in creating graphical assets, thus improving game design and development pipelines. However, there is little research to address how the generative methods can fit into the wider pipeline. We conducted a user study with 16 game designers and developers to examine their preferences regarding generative tools for graphical assets. The findings highlight that early design stage is preferred by all participants (mean values above 0.67 and p < .001 for early stages). Designers and developers prefer to use such tools for creating large amounts of variations at the cost of quality as they can improve the quality of the artefacts once they generate a suitable asset (mean value 0.17 where 1 is high quality, p < .001). They also strongly (mean value .78, p < .001) raised the need for better integration of such tools in existing design and development environments and the need for the outputs to be in common data formats, to be manipulatable and integrate smoothly into existing environments (mean 3.5 out of 5, p = .004). The study also highlights the requirement for further emphasis on the needs of the users to incorporate these tools effectively in existing pipelines. Informed by these results, we provide a set of guidelines for creating tools that meet the expectations and needs of game designers and developers. 

**Abstract (ZH)**: 图形资产在游戏设计与开发中的作用日益重要。生成性工具的应用有望提升图形资产的创建，进而改进游戏设计与开发流程。然而，关于生成性方法如何融入更广泛流程的研究较为不足。我们通过一项包含16名游戏设计师与开发者的用户研究，探索了他们对生成性工具的偏好。研究发现，所有参与者均偏好于在早期设计阶段使用这些工具（早期阶段均值高于0.67，p < .001）。设计师和开发者倾向于利用这些工具生成大量变体，尽管代价可能是质量的牺牲，但一旦生成合适资产，它们能够提升最终产出的质量（质量均值0.17，1表示高质量，p < .001）。研究还显示，参与者强烈要求更好地将此类工具整合到现有设计与开发环境中，并希望输出可采用常见数据格式、可操作且能平滑整合到现有环境（均值3.5，p = .004）。另外，研究进一步强调了需更关注用户需求，以有效将这些工具整合到现有流程中。根据这些结果，我们提出了设计满足游戏设计师和开发人员期望与需求工具的指导原则。 

---
# YARE-GAN: Yet Another Resting State EEG-GAN 

**Title (ZH)**: YARE-GAN：又一休息状态EEG-GAN 

**Authors**: Yeganeh Farahzadi, Morteza Ansarinia, Zoltan Kekecs  

**Link**: [PDF](https://arxiv.org/pdf/2503.02636)  

**Abstract**: Generative Adversarial Networks (GANs) have shown promise in synthesising realistic neural data, yet their potential for unsupervised representation learning in resting-state EEG remains under explored. In this study, we implement a Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate multi-channel resting-state EEG data and assess the quality of the synthesised signals through both visual and feature-based evaluations. Our results indicate that the model effectively captures the statistical and spectral characteristics of real EEG data, although challenges remain in replicating high-frequency oscillations in the frontal region. Additionally, we demonstrate that the Critic's learned representations can be fine-tuned for age group classification, achieving an out-of-sample accuracy, significantly better than a shuffled-label baseline. These findings suggest that generative models can serve not only as EEG data generators but also as unsupervised feature extractors, reducing the need for manual feature engineering. This study highlights the potential of GAN-based unsupervised learning for EEG analysis, suggesting avenues for more data-efficient deep learning applications in neuroscience. 

**Abstract (ZH)**: 生成对抗网络（GANs）在合成真实的神经数据方面展现了前景，但在静息态EEG的无监督表示学习方面仍待探索。在本研究中，我们采用带梯度惩罚的Wasserstein GAN（WGAN-GP）生成多通道静息态EEG数据，并通过视觉评估和特征评估来评估合成信号的质量。研究结果表明，模型有效地捕捉了真实EEG数据的统计和频谱特性，尽管在前额区域高频率振荡的复制上仍面临挑战。此外，我们展示了判别器学习到的表示可以微调用于年龄分组分类，其外样本准确性显著优于随机标签基线。这些发现表明，生成模型不仅可以作为EEG数据生成器，还可以作为无监督特征提取器，减少手动特征工程的需求。本研究突显了基于GAN的无监督学习在EEG分析中的潜在价值，为神经科学中的更高效深度学习应用指出了方向。 

---
# Reflection on Data Storytelling Tools in the Generative AI Era from the Human-AI Collaboration Perspective 

**Title (ZH)**: 从人机协作视角反思生成式AI时代的数据 storytelling 工具 

**Authors**: Haotian Li, Yun Wang, Huamin Qu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02631)  

**Abstract**: Human-AI collaborative tools attract attentions from the data storytelling community to lower the barrier of expertise and streamline the workflow. The recent advance in large-scale generative AI techniques, e.g., large language models (LLMs) and text-to-image models, has the potential to enhance data storytelling with their power in visual and narration generation. After two years since these techniques were publicly available, it is important to reflect our progress of applying them and have an outlook for future opportunities. To achieve the goal, we compare the collaboration patterns of the latest tools with those of earlier ones using a dedicated framework for understanding human-AI collaboration in data storytelling. Through comparison, we identify persistent collaboration patterns, e.g., human-creator + AI-assistant, and emerging ones, e.g., AI-creator + human-reviewer. The benefits of these AI techniques and other implications to human-AI collaboration are also revealed. We further propose future directions to hopefully ignite innovations. 

**Abstract (ZH)**: 大规模生成AI技术的进步吸引了数据讲故事社区的关注，以降低专业门槛并简化工作流程。近年来，大型语言模型（LLMs）和文本转图像模型等技术的发展为数据讲故事提供了强大的视觉和叙述生成能力。自这些技术公开以来两年时间里，反思其应用进展并对未来机遇进行展望变得尤为重要。为实现这一目标，我们使用一个专门的框架比较了最新工具与早期工具的人机协作模式。通过比较，我们识别出持续的人机协作模式，如人类创作+AI助理，以及新兴模式，如AI创作+人类审阅者。这些AI技术的优势及其他对人机协作的影响也得到了揭示。我们进一步提出未来方向，以期激发创新。 

---
# Federated nnU-Net for Privacy-Preserving Medical Image Segmentation 

**Title (ZH)**: 联邦nnU-Netfor隐私保护医疗图像分割 

**Authors**: Grzegorz Skorupko, Fotios Avgoustidis, Carlos Martín-Isla, Lidia Garrucho, Dimitri A. Kessler, Esmeralda Ruiz Pujadas, Oliver Díaz, Maciej Bobowicz, Katarzyna Gwoździewicz, Xavier Bargalló, Paulius Jaruševičius, Kaisar Kushibar, Karim Lekadir  

**Link**: [PDF](https://arxiv.org/pdf/2503.02549)  

**Abstract**: The nnU-Net framework has played a crucial role in medical image segmentation and has become the gold standard in multitudes of applications targeting different diseases, organs, and modalities. However, so far it has been used primarily in a centralized approach where the data collected from hospitals are stored in one center and used to train the nnU-Net. This centralized approach has various limitations, such as leakage of sensitive patient information and violation of patient privacy. Federated learning is one of the approaches to train a segmentation model in a decentralized manner that helps preserve patient privacy. In this paper, we propose FednnU-Net, a federated learning extension of nnU-Net. We introduce two novel federated learning methods to the nnU-Net framework - Federated Fingerprint Extraction (FFE) and Asymmetric Federated Averaging (AsymFedAvg) - and experimentally show their consistent performance for breast, cardiac and fetal segmentation using 6 datasets representing samples from 18 institutions. Additionally, to further promote research and deployment of decentralized training in privacy constrained institutions, we make our plug-n-play framework public. The source-code is available at this https URL . 

**Abstract (ZH)**: nnU-Net框架在医学图像分割中发挥了关键作用，并已成为针对不同疾病、器官和模态的众多应用中的黄金标准。然而，迄今为止，它主要在集中式方法中使用，其中医院收集的数据存储在一个中心，并用于训练nnU-Net。集中式方法存在各种限制，如敏感患者信息泄露和患者隐私侵犯。联邦学习是一种在分布式方式下训练分割模型的方法，有助于保护患者隐私。本文提出FednnU-Net，这是nnU-Net的联邦学习扩展。我们向nnU-Net框架引入了两种新的联邦学习方法——联邦指纹提取（FFE）和非对称联邦平均（AsymFedAvg），并通过6个数据集（代表来自18个机构的样本）实验证明了它们在乳腺、心脏和胎儿分割任务中的一致性能。此外，为了进一步促进在限制隐私的研究和部署去中心化训练，我们公开了我们的即插即用框架。源代码可从以下链接获取。 

---
# LTL Verification of Memoryful Neural Agents 

**Title (ZH)**: 内存型神经代理的LTL验证 

**Authors**: Mehran Hosseini, Alessio Lomuscio, Nicola Paoletti  

**Link**: [PDF](https://arxiv.org/pdf/2503.02512)  

**Abstract**: We present a framework for verifying Memoryful Neural Multi-Agent Systems (MN-MAS) against full Linear Temporal Logic (LTL) specifications. In MN-MAS, agents interact with a non-deterministic, partially observable environment. Examples of MN-MAS include multi-agent systems based on feed-forward and recurrent neural networks or state-space models. Different from previous approaches, we support the verification of both bounded and unbounded LTL specifications. We leverage well-established bounded model checking techniques, including lasso search and invariant synthesis, to reduce the verification problem to that of constraint solving. To solve these constraints, we develop efficient methods based on bound propagation, mixed-integer linear programming, and adaptive splitting. We evaluate the effectiveness of our algorithms in single and multi-agent environments from the Gymnasium and PettingZoo libraries, verifying unbounded specifications for the first time and improving the verification time for bounded specifications by an order of magnitude compared to the SoA. 

**Abstract (ZH)**: 我们提出了一种框架，用于验证含有内存的神经多智能体系统（MN-MAS） against 全局线性时序逻辑（LTL）规范。我们支持有界和无界LTL规范的验证。我们利用成熟的有界模型检查技术，包括lasso搜索和不变式的合成，将验证问题转化为约束求解问题。为了解这些约束，我们基于边界传播、混合整数线性规划和自适应分裂开发了高效的方法。我们在来自Gymnasium和PettingZoo库的单智能体和多智能体环境中评估了算法的有效性，首次验证了无界规范，并将有界规范的验证时间提高了数量级，超过当前最佳方法（SoA）。 

---
# Union of Experts: Adapting Hierarchical Routing to Equivalently Decomposed Transformer 

**Title (ZH)**: 专家联盟：将分层路由适应等效分解的变压器模型 

**Authors**: Yujiao Yang, Jing Lian, Linhui Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02495)  

**Abstract**: Mixture-of-Experts (MoE) enhances model performance while maintaining computational efficiency, making it well-suited for large-scale applications. However, expert in exist MoE paradigm works as an individual, thereby lacking high-quality expert interactions. Moreover, they have not been effectively extended to attention block, which constrains further efficiency improvements. To tackle these issues, we propose Union-of-Experts (UoE), which decomposes transformer into an equitant group of experts, and then implement dynamic routing on input data and experts. Our approach advances MoE design with three key innovations: (1) We conducted equitant expert decomposition on both MLP blocks and attention blocks based on matrix partition in tensor parallelism. (2) We developed two routing paradigms: patch wise data selection and expert selection, to apply routing across different levels. (3) We design the architecture of UoE model, including Selective Multi-Head Attention (SMHA) and Union-of-MLP-Experts (UoME). (4) We develop parallel implementation of UoE's routing and computation operation, and optimize efficiency based on the hardware processing analysis. The experiments demonstrate that the model employed with UoE surpass Full Attention, state-of-art MoEs and efficient transformers in several tasks across image and natural language domains. The source codes are available at this https URL. 

**Abstract (ZH)**: 专家集合（UoE）增强了模型性能的同时保持了计算效率，使其适合大规模应用。然而，现有专家集合（MoE）范式中的专家独立工作，缺乏高质量的专家交互。此外，它们未有效扩展到注意块中，限制了进一步的效率提升。为了应对这些问题，我们提出了专家集合（UoE），将Transformer分解为等效的专家组，并在输入数据和专家之间实施动态路由。我们的方法通过三个关键创新促进了MoE设计：（1）基于张量并行中的矩阵分割，在MLP块和注意块上进行了等效专家分解。（2）开发了两种路由范式：基于块的数据选择和专家选择，以在不同级别应用路由。（3）设计了UoE模型的架构，包括选择性多头注意（SMHA）和专家集合MLP（UoME）。（4）开发了UoE路由和计算操作的并行实现，并基于硬件处理分析进行了效率优化。实验结果表明，使用UoE的模型在图像和自然语言处理任务中优于全注意、最先进的MoE模型和高效Transformer模型。源代码可在以下网址获取。 

---
# ERetinex: Event Camera Meets Retinex Theory for Low-Light Image Enhancement 

**Title (ZH)**: ERetinex：事件相机与Retinex理论在低光照图像增强中的结合 

**Authors**: Xuejian Guo, Zhiqiang Tian, Yuehang Wang, Siqi Li, Yu Jiang, Shaoyi Du, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02484)  

**Abstract**: Low-light image enhancement aims to restore the under-exposure image captured in dark scenarios. Under such scenarios, traditional frame-based cameras may fail to capture the structure and color information due to the exposure time limitation. Event cameras are bio-inspired vision sensors that respond to pixel-wise brightness changes asynchronously. Event cameras' high dynamic range is pivotal for visual perception in extreme low-light scenarios, surpassing traditional cameras and enabling applications in challenging dark environments. In this paper, inspired by the success of the retinex theory for traditional frame-based low-light image restoration, we introduce the first methods that combine the retinex theory with event cameras and propose a novel retinex-based low-light image restoration framework named ERetinex. Among our contributions, the first is developing a new approach that leverages the high temporal resolution data from event cameras with traditional image information to estimate scene illumination accurately. This method outperforms traditional image-only techniques, especially in low-light environments, by providing more precise lighting information. Additionally, we propose an effective fusion strategy that combines the high dynamic range data from event cameras with the color information of traditional images to enhance image quality. Through this fusion, we can generate clearer and more detail-rich images, maintaining the integrity of visual information even under extreme lighting conditions. The experimental results indicate that our proposed method outperforms state-of-the-art (SOTA) methods, achieving a gain of 1.0613 dB in PSNR while reducing FLOPS by \textbf{84.28}\%. 

**Abstract (ZH)**: 低光图像增强旨在恢复在黑暗场景中拍摄的欠曝光图像。在这种场景下，传统的帧基相机可能会由于曝光时间限制而无法捕捉到结构和颜色信息。事件摄像头是受生物启发的视觉传感器，能够异步响应像素级别的亮度变化。事件摄像头的高动态范围在极端低光场景中的视觉感知中至关重要，超过了传统相机，并在具有挑战性的黑暗环境中实现了应用。在本文中，受传统帧基低光图像恢复中Retinex理论成功的启发，我们首次引入了将Retinex理论与事件摄像头相结合的方法，并提出了一种新的基于Retinex的低光图像增强框架，命名为ERetinex。在我们的贡献中，首先开发了一种新的方法，该方法利用事件摄像头的高时间分辨率数据与传统图像信息相结合，以精确估计场景光照。该方法在低光环境下优于仅使用图像的技术，通过提供更精确的照明信息。此外，我们提出了一种有效的融合策略，将事件摄像头的高动态范围数据与传统图像的颜色信息相结合，以增强图像质量。通过这种融合，可以生成更清晰且细节更丰富的图像，即使在极端光照条件下也能保持视觉信息的完整性。实验结果表明，我们提出的方法优于当前最先进的方法，在PSNR上提高了1.0613 dB，同时FLOPS减少了84.28%。 

---
# Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations 

**Title (ZH)**: 稀疏遇密集：级联稀疏-密集表示的一体化生成推荐 

**Authors**: Yuhao Yang, Zhi Ji, Zhaopeng Li, Yi Li, Zhonglin Mo, Yue Ding, Kai Chen, Zijian Zhang, Jie Li, Shuanglong Li, Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02453)  

**Abstract**: Generative models have recently gained attention in recommendation systems by directly predicting item identifiers from user interaction sequences. However, existing methods suffer from significant information loss due to the separation of stages such as quantization and sequence modeling, hindering their ability to achieve the modeling precision and accuracy of sequential dense retrieval techniques. Integrating generative and dense retrieval methods remains a critical challenge. To address this, we introduce the Cascaded Organized Bi-Represented generAtive retrieval (COBRA) framework, which innovatively integrates sparse semantic IDs and dense vectors through a cascading process. Our method alternates between generating these representations by first generating sparse IDs, which serve as conditions to aid in the generation of dense vectors. End-to-end training enables dynamic refinement of dense representations, capturing both semantic insights and collaborative signals from user-item interactions. During inference, COBRA employs a coarse-to-fine strategy, starting with sparse ID generation and refining them into dense vectors via the generative model. We further propose BeamFusion, an innovative approach combining beam search with nearest neighbor scores to enhance inference flexibility and recommendation diversity. Extensive experiments on public datasets and offline tests validate our method's robustness. Online A/B tests on a real-world advertising platform with over 200 million daily users demonstrate substantial improvements in key metrics, highlighting COBRA's practical advantages. 

**Abstract (ZH)**: 生成模型Recently在推荐系统中通过直接从用户交互序列预测项目标识符获得了关注。然而，现有方法由于量化和序列建模阶段的分离而导致了显著的信息损失，阻碍了它们实现与序列密集检索技术相当的建模精确度和准确性。将生成方法和密集检索方法的集成仍然是一项关键挑战。为此，我们引入了级联组织双表示生成检索（COBRA）框架，该框架创新地通过级联过程将稀疏语义ID和稠密向量结合起来。我们的方法交替生成这些表示，首先生成稀疏ID，它们作为条件以辅助稠密向量的生成。端到端的训练使稠密表示能够动态优化，捕捉用户项目交互中的语义洞察和协作信号。在推理过程中，COBRA采用粗到细策略，首先生成稀疏ID，再通过生成模型将它们细化成稠密向量。我们进一步提出了BeamFusion，这是一种结合了束搜索和最近邻评分的创新方法，以增强推理灵活性和推荐多样性。广泛的公开数据集实验和离线测试验证了我们方法的鲁棒性。在线A/B测试在具有超过2亿日活跃用户的实际广告平台上也证明了其显著改进，突显了COBRA的实用优势。 

---
# VisAgent: Narrative-Preserving Story Visualization Framework 

**Title (ZH)**: VisAgent: 故事叙述保真的叙事可视化框架 

**Authors**: Seungkwon Kim, GyuTae Park, Sangyeon Kim, Seung-Hun Nam  

**Link**: [PDF](https://arxiv.org/pdf/2503.02399)  

**Abstract**: Story visualization is the transformation of narrative elements into image sequences. While existing research has primarily focused on visual contextual coherence, the deeper narrative essence of stories often remains overlooked. This limitation hinders the practical application of these approaches, as generated images frequently fail to capture the intended meaning and nuances of the narrative fully. To address these challenges, we propose VisAgent, a training-free multi-agent framework designed to comprehend and visualize pivotal scenes within a given story. By considering story distillation, semantic consistency, and contextual coherence, VisAgent employs an agentic workflow. In this workflow, multiple specialized agents collaborate to: (i) refine layered prompts based on the narrative structure and (ii) seamlessly integrate \gt{generated} elements, including refined prompts, scene elements, and subject placement, into the final image. The empirically validated effectiveness confirms the framework's suitability for practical story visualization applications. 

**Abstract (ZH)**: 故事可视化是将叙事元素转化为图像序列的过程。尽管现有研究主要集中在视觉上下文连贯性上，但故事深层次的叙事本质往往被忽视。这一局限性阻碍了这些方法的实际应用，因为生成的图像通常无法充分捕捉叙事的意图意义和细微之处。为应对这些挑战，我们提出了一种无需训练的多智能体框架VisAgent，旨在理解并可视化给定故事中的关键场景。VisAgent通过考虑故事提炼、语义一致性及上下文连贯性，采用智能工作流。在此工作流中，多个专业智能体协作：（i）根据叙事结构细化分层提示，以及（ii）无缝集成生成的元素，包括细化提示、场景元素和主体布局，最终整合到图像中。经实验证明的有效性证实了该框架适用于实际故事可视化应用。 

---
# A Binary Classification Social Network Dataset for Graph Machine Learning 

**Title (ZH)**: 二元分类社交网络数据集用于图机器学习 

**Authors**: Adnan Ali, Jinglong Li, Huanhuan Chen, AlMotasem Bellah Al Ajlouni  

**Link**: [PDF](https://arxiv.org/pdf/2503.02397)  

**Abstract**: Social networks have a vast range of applications with graphs. The available benchmark datasets are citation, co-occurrence, e-commerce networks, etc, with classes ranging from 3 to 15. However, there is no benchmark classification social network dataset for graph machine learning. This paper fills the gap and presents the Binary Classification Social Network Dataset (\textit{BiSND}), designed for graph machine learning applications to predict binary classes. We present the BiSND in \textit{tabular and graph} formats to verify its robustness across classical and advanced machine learning. We employ a diverse set of classifiers, including four traditional machine learning algorithms (Decision Trees, K-Nearest Neighbour, Random Forest, XGBoost), one Deep Neural Network (multi-layer perceptrons), one Graph Neural Network (Graph Convolutional Network), and three state-of-the-art Graph Contrastive Learning methods (BGRL, GRACE, DAENS). Our findings reveal that BiSND is suitable for classification tasks, with F1-scores ranging from 67.66 to 70.15, indicating promising avenues for future enhancements. 

**Abstract (ZH)**: 社交网络具有广泛的应用范围，其中涉及图的数据。现有的基准数据集包括引用网络、共现网络、电子商务网络等，其类别从3到15不等。然而，还没有适用于图机器学习的基准分类社交网络数据集。本文填补了这一空白，提出了二分类社交网络数据集（\textit{BiSND}），旨在为图机器学习应用中的二分类预测任务提供支持。我们以表格式和图格式展示了\textit{BiSND}，以验证其在经典和高级机器学习方法中的 robustness。我们使用了多样化的分类器，包括四种传统的机器学习算法（决策树、K-最近邻、随机森林、XGBoost）、一个深度神经网络（多层感知机）、一个图神经网络（图卷积网络）以及三种最新的图对比学习方法（BGRL、GRACE、DAENS）。我们的研究结果表明，\textit{BiSND} 适用于分类任务，F1分数范围从67.66到70.15，这为未来的研究提供了有希望的途径。 

---
# GRADEO: Towards Human-Like Evaluation for Text-to-Video Generation via Multi-Step Reasoning 

**Title (ZH)**: GRADEO: 向量人类评价的文本到视频生成多步推理方法 

**Authors**: Zhun Mou, Bin Xia, Zhengchao Huang, Wenming Yang, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.02341)  

**Abstract**: Recent great advances in video generation models have demonstrated their potential to produce high-quality videos, bringing challenges to effective evaluation. Unlike human evaluation, existing automated evaluation metrics lack high-level semantic understanding and reasoning capabilities for video, thus making them infeasible and unexplainable. To fill this gap, we curate GRADEO-Instruct, a multi-dimensional T2V evaluation instruction tuning dataset, including 3.3k videos from over 10 existing video generation models and multi-step reasoning assessments converted by 16k human annotations. We then introduce GRADEO, one of the first specifically designed video evaluation models, which grades AI-generated videos for explainable scores and assessments through multi-step reasoning. Experiments show that our method aligns better with human evaluations than existing methods. Furthermore, our benchmarking reveals that current video generation models struggle to produce content that aligns with human reasoning and complex real-world scenarios. The models, datasets, and codes will be released soon. 

**Abstract (ZH)**: 近期在视频生成模型方面的重大进展展示了其产生高质量视频的潜力，但也带来了有效评估的挑战。现有自动评估指标缺乏对视频的高层语义理解和推理能力，因此使其不可行且难以解释。为了解决这一问题，我们编纂了GRADEO-Instruct多维度T2V评估指令调优数据集，包含来自超过10个现有视频生成模型的3300个视频和由16000个人工注释转换而来的多步推理评估。我们随后引入了GRADEO，这是首个专门设计的视频评估模型之一，能够通过多步推理为可解释的评分和评估打分AI生成的视频。实验结果显示，我们的方法比现有方法更符合人类评估。此外，我们的基准测试表明，当前的视频生成模型在产生符合人类推理和复杂现实场景的内容方面存在困难。相关模型、数据集和代码将很快发布。 

---
# BiasICL: In-Context Learning and Demographic Biases of Vision Language Models 

**Title (ZH)**: BiasICL: 在上下文学习与视觉语言模型的群体偏差 

**Authors**: Sonnet Xu, Joseph Janizek, Yixing Jiang, Roxana Daneshjou  

**Link**: [PDF](https://arxiv.org/pdf/2503.02334)  

**Abstract**: Vision language models (VLMs) show promise in medical diagnosis, but their performance across demographic subgroups when using in-context learning (ICL) remains poorly understood. We examine how the demographic composition of demonstration examples affects VLM performance in two medical imaging tasks: skin lesion malignancy prediction and pneumothorax detection from chest radiographs. Our analysis reveals that ICL influences model predictions through multiple mechanisms: (1) ICL allows VLMs to learn subgroup-specific disease base rates from prompts and (2) ICL leads VLMs to make predictions that perform differently across demographic groups, even after controlling for subgroup-specific disease base rates. Our empirical results inform best-practices for prompting current VLMs (specifically examining demographic subgroup performance, and matching base rates of labels to target distribution at a bulk level and within subgroups), while also suggesting next steps for improving our theoretical understanding of these models. 

**Abstract (ZH)**: 视觉语言模型（VLMs）在医学诊断中展现出潜力，但在使用上下文学习（ICL）时，其在不同人口子组中的性能仍然知之甚少。我们探讨了示范示例的人口组成如何影响VLM在两项医学影像任务中的表现：皮肤病变恶性预测和胸部X光片中气胸检测。我们的分析揭示了ICL通过多种机制影响模型预测：（1）ICL使VLM能够从提示中学习子组特定的疾病基率；（2）ICL导致VLM在不同人口组中的预测表现不同，即使在控制了子组特定的疾病基率后也是如此。我们的实证结果为当前VLM的启提示最佳实践提供了指导（特别关注人口子组的表现，并在总体层面和子组内部将标签基率与目标分布相匹配），同时也指出了需要进一步研究以改进我们对这些模型的理论理解的方向。 

---
# Examining the Mental Health Impact of Misinformation on Social Media Using a Hybrid Transformer-Based Approach 

**Title (ZH)**: 使用混合变压器方法探究社交媒体上 misinformation 对心理健康的影响 

**Authors**: Sarvesh Arora, Sarthak Arora, Deepika Kumar, Vallari Agrawal, Vedika Gupta, Dipit Vasdev  

**Link**: [PDF](https://arxiv.org/pdf/2503.02333)  

**Abstract**: Social media has significantly reshaped interpersonal communication, fostering connectivity while also enabling the proliferation of misinformation. The unchecked spread of false narratives has profound effects on mental health, contributing to increased stress, anxiety, and misinformation-driven paranoia. This study presents a hybrid transformer-based approach using a RoBERTa-LSTM classifier to detect misinformation, assess its impact on mental health, and classify disorders linked to misinformation exposure. The proposed models demonstrate accuracy rates of 98.4, 87.8, and 77.3 in detecting misinformation, mental health implications, and disorder classification, respectively. Furthermore, Pearson's Chi-Squared Test for Independence (p-value = 0.003871) validates the direct correlation between misinformation and deteriorating mental well-being. This study underscores the urgent need for better misinformation management strategies to mitigate its psychological repercussions. Future research could explore broader datasets incorporating linguistic, demographic, and cultural variables to deepen the understanding of misinformation-induced mental health distress. 

**Abstract (ZH)**: 社交媒体显著重塑了人际沟通，促进了连接性的同时也使得虚假信息的传播无约束。未经约束的虚假叙事蔓延对心理健康产生了深远影响，增加了压力、焦虑和由虚假信息驱动的猜疑。本研究提出了一种基于混合变换器的 approaching，使用 RoBERTa-LSTM 分类器来检测虚假信息、评估其对心理健康的影响以及分类与虚假信息暴露相关的障碍。所提出的模型在检测虚假信息、心理健康影响和障碍分类方面的准确率分别为 98.4%、87.8% 和 77.3%。此外，皮尔森独立性卡方检验（p值 = 0.003871）证实了虚假信息与心理健康恶化之间的直接关联。本研究强调了迫切需要更好的虚假信息管理策略以减轻其心理影响。未来的研究可以探索包含语言学、人口统计和文化变量的更广泛的数据库，以深化对虚假信息引发的心理健康困扰的理解。 

---
# Target Return Optimizer for Multi-Game Decision Transformer 

**Title (ZH)**: 多游戏决策变换器的目标回报优化器 

**Authors**: Kensuke Tatematsu, Akifumi Wachi  

**Link**: [PDF](https://arxiv.org/pdf/2503.02311)  

**Abstract**: Achieving autonomous agents with robust generalization capabilities across diverse games and tasks remains one of the ultimate goals in AI research. Recent advancements in transformer-based offline reinforcement learning, exemplified by the MultiGame Decision Transformer [Lee et al., 2022], have shown remarkable performance across various games or tasks. However, these approaches depend heavily on human expertise, presenting substantial challenges for practical deployment, particularly in scenarios with limited prior game-specific knowledge. In this paper, we propose an algorithm called Multi-Game Target Return Optimizer (MTRO) to autonomously determine game-specific target returns within the Multi-Game Decision Transformer framework using solely offline datasets. MTRO addresses the existing limitations by automating the target return configuration process, leveraging environmental reward information extracted from offline datasets. Notably, MTRO does not require additional training, enabling seamless integration into existing Multi-Game Decision Transformer architectures. Our experimental evaluations on Atari games demonstrate that MTRO enhances the performance of RL policies across a wide array of games, underscoring its potential to advance the field of autonomous agent development. 

**Abstract (ZH)**: 实现跨多种游戏和任务具有稳健泛化能力的自主代理仍是AI研究中的终极目标。Recent advancements in transformer-based offline reinforcement learning, exemplified by the MultiGame Decision Transformer [Lee et al., 2022], have shown remarkable performance across various games or tasks.然而，这些方法高度依赖于人类专业知识，为实际部署带来了重大挑战，尤其是在有限的游戏特定先验知识的情景下。本文提出了一种称为Multi-Game Target Return Optimizer (MTRO)的算法，该算法在Multi-Game Decision Transformer框架中仅使用离线数据集自主确定游戏特定的目标回报。MTRO通过利用从离线数据集中提取的环境奖励信息自动化目标回报配置过程，解决了现有方法的局限性。值得注意的是，MTRO无需额外训练，可以无缝集成到现有的Multi-Game Decision Transformer架构中。我们在 Atari 游戏上的实验评估表明，MTRO 改善了强化学习策略在多种游戏中的性能，突显了其在自主代理开发领域进步的潜力。 

---
# Flexible Prefrontal Control over Hippocampal Episodic Memory for Goal-Directed Generalization 

**Title (ZH)**: 前额叶对目标引导性泛化的 hippocampal 事件记忆的灵活控制 

**Authors**: Yicong Zheng, Nora Wolf, Charan Ranganath, Randall C. O'Reilly, Kevin L. McKee  

**Link**: [PDF](https://arxiv.org/pdf/2503.02303)  

**Abstract**: Many tasks require flexibly modifying perception and behavior based on current goals. Humans can retrieve episodic memories from days to years ago, using them to contextualize and generalize behaviors across novel but structurally related situations. The brain's ability to control episodic memories based on task demands is often attributed to interactions between the prefrontal cortex (PFC) and hippocampus (HPC). We propose a reinforcement learning model that incorporates a PFC-HPC interaction mechanism for goal-directed generalization. In our model, the PFC learns to generate query-key representations to encode and retrieve goal-relevant episodic memories, modulating HPC memories top-down based on current task demands. Moreover, the PFC adapts its encoding and retrieval strategies dynamically when faced with multiple goals presented in a blocked, rather than interleaved, manner. Our results show that: (1) combining working memory with selectively retrieved episodic memory allows transfer of decisions among similar environments or situations, (2) top-down control from PFC over HPC improves learning of arbitrary structural associations between events for generalization to novel environments compared to a bottom-up sensory-driven approach, and (3) the PFC encodes generalizable representations during both encoding and retrieval of goal-relevant memories, whereas the HPC exhibits event-specific representations. Together, these findings highlight the importance of goal-directed prefrontal control over hippocampal episodic memory for decision-making in novel situations and suggest a computational mechanism by which PFC-HPC interactions enable flexible behavior. 

**Abstract (ZH)**: 基于目标控制情景相关性的强化学习模型：前额皮层-海马体交互机制 

---
# Experience Replay with Random Reshuffling 

**Title (ZH)**: 随机重排的经验重演 

**Authors**: Yasuhiro Fujita  

**Link**: [PDF](https://arxiv.org/pdf/2503.02269)  

**Abstract**: Experience replay is a key component in reinforcement learning for stabilizing learning and improving sample efficiency. Its typical implementation samples transitions with replacement from a replay buffer. In contrast, in supervised learning with a fixed dataset, it is a common practice to shuffle the dataset every epoch and consume data sequentially, which is called random reshuffling (RR). RR enjoys theoretically better convergence properties and has been shown to outperform with-replacement sampling empirically. To leverage the benefits of RR in reinforcement learning, we propose sampling methods that extend RR to experience replay, both in uniform and prioritized settings. We evaluate our sampling methods on Atari benchmarks, demonstrating their effectiveness in deep reinforcement learning. 

**Abstract (ZH)**: 经验回放是强化学习中稳定学习和提高样本效率的关键组件。其典型的实现方式是从回放缓冲区中带替换地采样过渡。相比之下，在监督学习中，每轮使用固定的数据集时，常见的做法是每轮重新洗牌数据集并顺序消费数据，这种方法称为随机重新洗牌（RR）。理论上，RR 具有更好的收敛性质，并且实验表明它优于带替换采样。为了在强化学习中利用 RR 的益处，我们提出了一种采样方法，将 RR 扩展到经验回放中，既适用于均匀采样也适用于优先级采样。我们在 Atari 基准上评估了我们的采样方法，展示了其在深度强化学习中的有效性。 

---
# REAct: Rational Exponential Activation for Better Learning and Generalization in PINNs 

**Title (ZH)**: REAct: 基于理性指数激活以提高物理 informer 网络的学习能力和泛化能力 

**Authors**: Sourav Mishra, Shreya Hallikeri, Suresh Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2503.02267)  

**Abstract**: Physics-Informed Neural Networks (PINNs) offer a promising approach to simulating physical systems. Still, their application is limited by optimization challenges, mainly due to the lack of activation functions that generalize well across several physical systems. Existing activation functions often lack such flexibility and generalization power. To address this issue, we introduce Rational Exponential Activation (REAct), a generalized form of tanh consisting of four learnable shape parameters. Experiments show that REAct outperforms many standard and benchmark activations, achieving an MSE three orders of magnitude lower than tanh on heat problems and generalizing well to finer grids and points beyond the training domain. It also excels at function approximation tasks and improves noise rejection in inverse problems, leading to more accurate parameter estimates across varying noise levels. 

**Abstract (ZH)**: 基于物理的神经网络（PINNs）提供了一种模拟物理系统的有前途的方法。然而，其应用受限于优化挑战，主要原因是缺乏能够在多种物理系统中泛化的激活函数。现有的激活函数通常缺乏这种灵活性和泛化能力。为了解决这一问题，我们引入了一种tanh的广义形式——理性指数激活（REAct），它包含四个可学习的形状参数。实验结果显示，REAct在热问题上的均方误差比tanh低三个数量级，并且能够在细网格和训练域外的点上很好地泛化。此外，REAct在函数逼近任务中表现出色，提高了逆问题中的噪声抵制能力，从而在不同噪声水平下获得更准确的参数估计。 

---
# Deficient Excitation in Parameter Learning 

**Title (ZH)**: 参数学习中的激发不足 

**Authors**: Ganghui Cao, Shimin Wang, Martin Guay, Jinzhi Wang, Zhisheng Duan, Marios M. Polycarpou  

**Link**: [PDF](https://arxiv.org/pdf/2503.02235)  

**Abstract**: This paper investigates parameter learning problems under deficient excitation (DE). The DE condition is a rank-deficient, and therefore, a more general evolution of the well-known persistent excitation condition. Under the DE condition, a proposed online algorithm is able to calculate the identifiable and non-identifiable subspaces, and finally give an optimal parameter estimate in the sense of least squares. In particular, the learning error within the identifiable subspace exponentially converges to zero in the noise-free case, even without persistent excitation. The DE condition also provides a new perspective for solving distributed parameter learning problems, where the challenge is posed by local regressors that are often insufficiently excited. To improve knowledge of the unknown parameters, a cooperative learning protocol is proposed for a group of estimators that collect measured information under complementary DE conditions. This protocol allows each local estimator to operate locally in its identifiable subspace, and reach a consensus with neighbours in its non-identifiable subspace. As a result, the task of estimating unknown parameters can be achieved in a distributed way using cooperative local estimators. Application examples in system identification are given to demonstrate the effectiveness of the theoretical results developed in this paper. 

**Abstract (ZH)**: 本文探讨在 deficient excitation (DE) 条件下的参数学习问题。DE 条件是一种秩亏条件，因而是一种已知的持久激励条件的更通用形式。在 DE 条件下，提出了一种在线算法，能够计算可识别子空间和不可识别子空间，并最终给出最小二乘意义下的最优参数估计。特别地，在无噪声情况下，可识别子空间内的学习误差指数地收敛于零，即使没有持久激励。DE 条件还为解决分布式参数学习问题提供了新的视角，这些建模器面临的挑战在于局部 regressors 通常激励不足。为了提高未知参数的知识，提出了一种合作学习协议，用于一组在互补 DE 条件下收集测量信息的估计器。该协议允许每个局部估计器在其可识别子空间内进行局部操作，并在不可识别子空间内与邻居达成一致。结果表明，可以通过合作局部估计器以分布式方式实现未知参数的估计。本文给出了系统识别的应用实例，以证明所发展的理论结果的有效性。 

---
# Discrete Differential Evolution Particle Swarm Optimization Algorithm for Energy Saving Flexible Job Shop Scheduling Problem Considering Machine Multi States 

**Title (ZH)**: 考虑机器多状态的节能柔性作业 shop排程问题的离散差分进化粒子群优化算法 

**Authors**: Da Wang, Yu Zhang, Kai Zhang, Junqing Li, Dengwang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02180)  

**Abstract**: As the continuous deepening of low-carbon emission reduction policies, the manufacturing industries urgently need sensible energy-saving scheduling schemes to achieve the balance between improving production efficiency and reducing energy consumption. In energy-saving scheduling, reasonable machine states-switching is a key point to achieve expected goals, i.e., whether the machines need to switch speed between different operations, and whether the machines need to add extra setup time between different jobs. Regarding this matter, this work proposes a novel machine multi states-based energy saving flexible job scheduling problem (EFJSP-M), which simultaneously takes into account machine multi speeds and setup time. To address the proposed EFJSP-M, a kind of discrete differential evolution particle swarm optimization algorithm (D-DEPSO) is designed. In specific, D-DEPSO includes a hybrid initialization strategy to improve the initial population performance, an updating mechanism embedded with differential evolution operators to enhance population diversity, and a critical path variable neighborhood search strategy to expand the solution space. At last, based on datasets DPs and MKs, the experiment results compared with five state-of-the-art algorithms demonstrate the feasible of EFJSP-M and the superior of D-DEPSO. 

**Abstract (ZH)**: 随着低碳减排政策的不断深化，制造业迫切需要 sensible 能源节约调度方案以实现提高生产效率与降低能耗之间的平衡。在能源节约调度中，合理的机器状态切换是实现预期目标的关键，即是否需要在不同工序间切换机器速度，以及是否需要在不同任务间增加额外的切换时间。针对此问题，本文提出了一种新型的基于多状态的节能柔性作业调度问题 (EFJSP-M)，同时考虑了机器多速性和切换时间。为解决提出的 EFJSP-M，设计了一种离散微分进化粒子群优化算法 (D-DEPSO)。具体而言，D-DEPSO 包括混合初始化策略以提高初始种群性能，嵌入差异进化算子的更新机制以增强种群多样性，以及关键路径变邻域搜索策略以扩展解空间。最后，基于 DPs 和 MKs 数据集，与五种先进算法的实验结果对比证明了 EFJSP-M 的可行性和 D-DEPSO 的优越性。 

---
# MobRFFI: Non-cooperative Device Re-identification for Mobility Intelligence 

**Title (ZH)**: MobRFFI: 不合作设备识别以提高移动智能 

**Authors**: Stepan Mazokha, Fanchen Bao, George Sklivanitis, Jason O. Hallstrom  

**Link**: [PDF](https://arxiv.org/pdf/2503.02156)  

**Abstract**: WiFi-based mobility monitoring in urban environments can provide valuable insights into pedestrian and vehicle movements. However, MAC address randomization introduces a significant obstacle in accurately estimating congestion levels and path trajectories. To this end, we consider radio frequency fingerprinting and re-identification for attributing WiFi traffic to emitting devices without the use of MAC addresses.
We present MobRFFI, an AI-based device fingerprinting and re-identification framework for WiFi networks that leverages an encoder deep learning model to extract unique features based on WiFi chipset hardware impairments. It is entirely independent of frame type. When evaluated on the WiFi fingerprinting dataset WiSig, our approach achieves 94% and 100% device accuracy in multi-day and single-day re-identification scenarios, respectively.
We also collect a novel dataset, MobRFFI, for granular multi-receiver WiFi device fingerprinting evaluation. Using the dataset, we demonstrate that the combination of fingerprints from multiple receivers boosts re-identification performance from 81% to 100% on a single-day scenario and from 41% to 100% on a multi-day scenario. 

**Abstract (ZH)**: 基于WiFi的移动监控在城市环境中可以提供行人和车辆移动的宝贵见解。然而，MAC地址随机化给准确估计拥堵水平和路径轨迹带来重大障碍。为此，我们考虑使用射频指纹识别和再识别，以不依赖MAC地址的方式将WiFi流量归因于发出设备。
MobRFFI：一种基于AI的WiFi网络设备指纹识别和再识别框架，利用编码深度学习模型根据WiFi芯片组硬件缺陷提取独特的特征。该框架完全不依赖于帧类型。在WiSig WiFi指纹识别数据集上评估，我们的方法在多天和单日再识别场景中分别实现了94%和100%的设备准确性。
我们还收集了一个新的数据集MobRFFI，用于细粒度多接收器WiFi设备指纹识别评估。利用该数据集，我们展示了多接收器指纹组合在单日场景中将再识别性能从81%提升到100%，在多天场景中提升从41%到100%。 

---
# AugFL: Augmenting Federated Learning with Pretrained Models 

**Title (ZH)**: AugFL：利用预训练模型增强联邦学习 

**Authors**: Sheng Yue, Zerui Qin, Yongheng Deng, Ju Ren, Yaoxue Zhang, Junshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02154)  

**Abstract**: Federated Learning (FL) has garnered widespread interest in recent years. However, owing to strict privacy policies or limited storage capacities of training participants such as IoT devices, its effective deployment is often impeded by the scarcity of training data in practical decentralized learning environments. In this paper, we study enhancing FL with the aid of (large) pre-trained models (PMs), that encapsulate wealthy general/domain-agnostic knowledge, to alleviate the data requirement in conducting FL from scratch. Specifically, we consider a networked FL system formed by a central server and distributed clients. First, we formulate the PM-aided personalized FL as a regularization-based federated meta-learning problem, where clients join forces to learn a meta-model with knowledge transferred from a private PM stored at the server. Then, we develop an inexact-ADMM-based algorithm, AugFL, to optimize the problem with no need to expose the PM or incur additional computational costs to local clients. Further, we establish theoretical guarantees for AugFL in terms of communication complexity, adaptation performance, and the benefit of knowledge transfer in general non-convex cases. Extensive experiments corroborate the efficacy and superiority of AugFL over existing baselines. 

**Abstract (ZH)**: 联邦学习（FL）近年来引起了广泛兴趣。然而，由于严格的隐私政策或训练参与者如物联网设备的有限存储容量，其在实际去中心化学习环境中有效部署往往受限于训练数据的稀缺性。本文研究了通过辅助机制（如大型）预训练模型（PMs）来增强FL，以减少从头进行FL时的数据需求。具体而言，我们考虑由中央服务器和分布式客户端组成的一个网络化FL系统。首先，我们将PM辅助个性化FL形式化为基于正则化联邦元学习问题，客户端共同努力通过存储在服务器上的私人PM转移知识来学习一个元模型。然后，我们开发了一个基于不精确ADMM的算法AugFL，该算法优化该问题而不需暴露PM或给本地客户端增加额外的计算成本。此外，我们在一般非凸情况下为AugFL建立了通信复杂度、适应性能和知识转移收益的理论保证。广泛的实验验证了AugFL的有效性和优越性。 

---
# Elliptic Loss Regularization 

**Title (ZH)**: 椭圆损失正则化 

**Authors**: Ali Hasan, Haoming Yang, Yuting Ng, Vahid Tarokh  

**Link**: [PDF](https://arxiv.org/pdf/2503.02138)  

**Abstract**: Regularizing neural networks is important for anticipating model behavior in regions of the data space that are not well represented. In this work, we propose a regularization technique for enforcing a level of smoothness in the mapping between the data input space and the loss value. We specify the level of regularity by requiring that the loss of the network satisfies an elliptic operator over the data domain. To do this, we modify the usual empirical risk minimization objective such that we instead minimize a new objective that satisfies an elliptic operator over points within the domain. This allows us to use existing theory on elliptic operators to anticipate the behavior of the error for points outside the training set. We propose a tractable computational method that approximates the behavior of the elliptic operator while being computationally efficient. Finally, we analyze the properties of the proposed regularization to understand the performance on common problems of distribution shift and group imbalance. Numerical experiments confirm the utility of the proposed regularization technique. 

**Abstract (ZH)**: 强制神经网络在数据空间中未充分代表的区域具有一致的行为对于预测模型行为很重要。本文提出了一种正则化技术，以确保数据输入空间与损失值之间的映射具有一定程度的平滑性。通过要求网络的损失满足数据域上的椭圆算子来指定这种正则性。为此，我们修改了通常的经验风险最小化目标，使其而是最小化一个满足数据域内点上的椭圆算子的新目标。这使得我们可以利用现有的椭圆算子理论来预测训练集外点的误差行为。我们提出了一种计算上可行的方法来近似椭圆算子的行为，同时保持计算效率。最后，我们分析了所提正则化技术的性质，以了解其在常见分布转移和组不平衡问题上的性能。数值实验证实了所提正则化技术的有效性。 

---
# Forgetting Transformer: Softmax Attention with a Forget Gate 

**Title (ZH)**: 遗忘变换器：带有遗忘门的softmax注意机制 

**Authors**: Zhixuan Lin, Evgenii Nikishin, Xu Owen He, Aaron Courville  

**Link**: [PDF](https://arxiv.org/pdf/2503.02130)  

**Abstract**: An essential component of modern recurrent sequence models is the forget gate. While Transformers do not have an explicit recurrent form, we show that a forget gate can be naturally incorporated into Transformers by down-weighting the unnormalized attention scores in a data-dependent way. We name this attention mechanism the Forgetting Attention and the resulting model the Forgetting Transformer (FoX). We show that FoX outperforms the Transformer on long-context language modeling, length extrapolation, and short-context downstream tasks, while performing on par with the Transformer on long-context downstream tasks. Moreover, it is compatible with the FlashAttention algorithm and does not require any positional embeddings. Several analyses, including the needle-in-the-haystack test, show that FoX also retains the Transformer's superior long-context capabilities over recurrent sequence models such as Mamba-2, HGRN2, and DeltaNet. We also introduce a "Pro" block design that incorporates some common architectural components in recurrent sequence models and find it significantly improves the performance of both FoX and the Transformer. Our code is available at this https URL. 

**Abstract (ZH)**: 现代递归序列模型的一个基本组件是忘门。虽然变压器没有显式的递归形式，但我们展示了可以通过数据依赖的方式降低未归一化的注意力分数来自然地将忘门融入到变压器中。我们称这种注意力机制为忘门注意力，并将相应的模型命名为忘门变压器（FoX）。我们展示了FoX在长上下文语言建模、长度外推以及短上下文下游任务上优于变压器，而在长上下文下游任务上与变压器性能相当。此外，FoX 与 FlashAttention 算法兼容，无需任何位置嵌入。包括针扎 haystack 测试在内的一些分析表明，FoX 也保留了变压器在与 Mamba-2、HGRN2 和 DeltaNet 这种递归序列模型相比时优于长上下文能力。我们还引入了一种“Pro”块设计，将一些常见的递归序列模型架构组件整合其中，并发现它显著提高了FoX和变压器的性能。我们的代码可在以下链接获取。 

---
# A Near Complete Nonasymptotic Generalization Theory For Multilayer Neural Networks: Beyond the Bias-Variance Tradeoff 

**Title (ZH)**: 近完全非渐近通用化理论：超越偏差-方差权衡的多层神经网络 

**Authors**: Hao Yu, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.02129)  

**Abstract**: We propose a first near complete (that will make explicit sense in the main text) nonasymptotic generalization theory for multilayer neural networks with arbitrary Lipschitz activations and general Lipschitz loss functions (with some very mild conditions). In particular, it doens't require the boundness of loss function, as commonly assumed in the literature. Our theory goes beyond the bias-variance tradeoff, aligned with phenomenon typically encountered in deep learning. It is therefore sharp different with other existing nonasymptotic generalization error bounds for neural networks. More explicitly, we propose an explicit generalization error upper bound for multilayer neural networks with arbitrary Lipschitz activations $\sigma$ with $\sigma(0)=0$ and broad enough Lipschitz loss functions, without requiring either the width, depth or other hyperparameters of the neural network approaching infinity, a specific neural network architect (e.g. sparsity, boundness of some norms), a particular activation function, a particular optimization algorithm or boundness of the loss function, and with taking the approximation error into consideration. General Lipschitz activation can also be accommodated into our framework. A feature of our theory is that it also considers approximation errors. Furthermore, we show the near minimax optimality of our theory for multilayer ReLU networks for regression problems. Notably, our upper bound exhibits the famous double descent phenomenon for such networks, which is the most distinguished characteristic compared with other existing results. This work emphasizes a view that many classical results should be improved to embrace the unintuitive characteristics of deep learning to get a better understanding of it. 

**Abstract (ZH)**: 我们提出了一种接近完备的非渐近泛化理论，适用于具有任意Lipschitz激活函数和广义Lipschitz损失函数的多层神经网络（在主文中将明确阐述）。特别地，该理论不要求损失函数有界，这不同于文献中的常见假设。该理论超越了偏差-方差权衡，与在深度学习中通常遇到的现象相一致。因此，它与现有的非渐近神经网络泛化误差边界有显著不同。更具体地说，我们提出了一个关于具有任意Lipschitz激活函数$\sigma(\sigma(0)=0)$和足够广义的Lipschitz损失函数的多层神经网络的显式泛化误差上界，而不需要神经网络的宽度、深度或其他超参数趋于无穷大，也不需要特定的神经网络结构（如稀疏性、某些范数有界性）、特定的激活函数、特定的优化算法或损失函数有界性，同时考虑了逼近误差。我们理论框架也可以容纳广义Lipschitz激活函数。我们的理论的一个特点是考虑了逼近误差。此外，我们证明了对于回归问题，我们对于多层ReLU网络的理论近乎最小最大最优，并且我们的上界表现出著名的双下降现象，这是与其他现有结果最显著的区别。这项工作强调了一种观点，即许多经典结果需要改进，以便纳入深度学习的直觉之外的特性，从而更好地理解它。 

---
# Parabolic Continual Learning 

**Title (ZH)**: 抛物线连续学习 

**Authors**: Haoming Yang, Ali Hasan, Vahid Tarokh  

**Link**: [PDF](https://arxiv.org/pdf/2503.02117)  

**Abstract**: Regularizing continual learning techniques is important for anticipating algorithmic behavior under new realizations of data. We introduce a new approach to continual learning by imposing the properties of a parabolic partial differential equation (PDE) to regularize the expected behavior of the loss over time. This class of parabolic PDEs has a number of favorable properties that allow us to analyze the error incurred through forgetting and the error induced through generalization. Specifically, we do this through imposing boundary conditions where the boundary is given by a memory buffer. By using the memory buffer as a boundary, we can enforce long term dependencies by bounding the expected error by the boundary loss. Finally, we illustrate the empirical performance of the method on a series of continual learning tasks. 

**Abstract (ZH)**: 通过施加抛物型偏微分方程（PDE）的性质来正则化持续学习技术，以预测在新数据实现下的算法行为至关重要。我们提出了一种新的持续学习方法，通过对损失在时间上的预期行为施加抛物型偏微分方程（PDE）的性质来进行正则化。此类抛物型偏微分方程具有许多有利特性，允许我们分析由于遗忘引起的误差和由于泛化引起的误差。具体而言，我们通过施加由记忆缓冲区给出的边界条件来实现这一点。通过将记忆缓冲区作为边界，我们可以通过边界损失来约束预期误差，从而确保长期依赖性。最后，我们在一系列持续学习任务中展示了该方法的实证性能。 

---
# Correlation to Causation: A Causal Deep Learning Framework for Arctic Sea Ice Prediction 

**Title (ZH)**: 因果关联：北极海冰预测的因果深度学习框架 

**Authors**: Emam Hossain, Muhammad Hasan Ferdous, Jianwu Wang, Aneesh Subramanian, Md Osman Gani  

**Link**: [PDF](https://arxiv.org/pdf/2503.02093)  

**Abstract**: Traditional machine learning and deep learning techniques rely on correlation-based learning, often failing to distinguish spurious associations from true causal relationships, which limits robustness, interpretability, and generalizability. To address these challenges, we propose a causality-driven deep learning framework that integrates Multivariate Granger Causality (MVGC) and PCMCI+ causal discovery algorithms with a hybrid deep learning architecture. Using 43 years (1979-2021) of daily and monthly Arctic Sea Ice Extent (SIE) and ocean-atmospheric datasets, our approach identifies causally significant factors, prioritizes features with direct influence, reduces feature overhead, and improves computational efficiency. Experiments demonstrate that integrating causal features enhances the deep learning model's predictive accuracy and interpretability across multiple lead times. Beyond SIE prediction, the proposed framework offers a scalable solution for dynamic, high-dimensional systems, advancing both theoretical understanding and practical applications in predictive modeling. 

**Abstract (ZH)**: 传统的机器学习和深度学习技术依赖于基于相关性的学习，往往难以区分虚假关联和真正的因果关系，这限制了模型的稳健性、可解释性和泛化能力。为了解决这些问题，我们提出了一种因果驱动的深度学习框架，该框架结合了多元格兰杰因果性（MVGC）和PCMCI+因果发现算法，并采用混合深度学习架构。通过1979-2021年43年的日度和月度北极海冰Extent (SIE) 和海洋-大气数据集，我们的方法识别出具有因果意义的因素，优先考虑具有直接影响的特征，减少特征过载，并提高计算效率。实验结果表明，集成因果特征提升了深度学习模型在多个预见时长的预测准确性和可解释性。该框架不仅适用于SIE预测，还提供了动态高维系统的可扩展解决方案，推进了预测建模的理论理解和实际应用。 

---
# Survey Perspective: The Role of Explainable AI in Threat Intelligence 

**Title (ZH)**: 解释性人工智能在威胁情报中的作用调查视角 

**Authors**: Nidhi Rastogi, Devang Dhanuka, Amulya Saxena, Pranjal Mairal, Le Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.02065)  

**Abstract**: The increasing reliance on AI-based security tools in Security Operations Centers (SOCs) has transformed threat detection and response, yet analysts frequently struggle with alert overload, false positives, and lack of contextual relevance. The inability to effectively analyze AI-generated security alerts lead to inefficiencies in incident response and reduces trust in automated decision-making. In this paper, we show results and analysis of our investigation of how SOC analysts navigate AI-based alerts, their challenges with current security tools, and how explainability (XAI) integrated into their security workflows has the potential to become an effective decision support. In this vein, we conducted an industry survey. Using the survey responses, we analyze how security analysts' process, retrieve, and prioritize alerts. Our findings indicate that most analysts have not yet adopted XAI-integrated tools, but they express high interest in attack attribution, confidence scores, and feature contribution explanations to improve interpretability, and triage efficiency. Based on our findings, we also propose practical design recommendations for XAI-enhanced security alert systems, enabling AI-based cybersecurity solutions to be more transparent, interpretable, and actionable. 

**Abstract (ZH)**: 基于AI的安全工具在安全运营中心的应用日益增多：分析师面临的挑战及解释性人工智能的潜在价值分析 

---
# Dynamic Search for Inference-Time Alignment in Diffusion Models 

**Title (ZH)**: 差分模型推断时动态搜索对齐方法 

**Authors**: Xiner Li, Masatoshi Uehara, Xingyu Su, Gabriele Scalia, Tommaso Biancalani, Aviv Regev, Sergey Levine, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.02039)  

**Abstract**: Diffusion models have shown promising generative capabilities across diverse domains, yet aligning their outputs with desired reward functions remains a challenge, particularly in cases where reward functions are non-differentiable. Some gradient-free guidance methods have been developed, but they often struggle to achieve optimal inference-time alignment. In this work, we newly frame inference-time alignment in diffusion as a search problem and propose Dynamic Search for Diffusion (DSearch), which subsamples from denoising processes and approximates intermediate node rewards. It also dynamically adjusts beam width and tree expansion to efficiently explore high-reward generations. To refine intermediate decisions, DSearch incorporates adaptive scheduling based on noise levels and a lookahead heuristic function. We validate DSearch across multiple domains, including biological sequence design, molecular optimization, and image generation, demonstrating superior reward optimization compared to existing approaches. 

**Abstract (ZH)**: 在不同领域中，扩散模型展示了强大的生成能力，但在将模型输出与期望的奖励函数对齐时仍面临挑战，特别是在奖励函数非可微的情况下。一些无梯度引导方法已被开发，但它们在实现最佳推理时对齐方面往往表现不佳。在本文中，我们将扩散模型的推理时对齐重新定义为一个搜索问题，并提出了Dynamic Search for Diffusion (DSearch)，该方法从去噪过程中抽样并近似中间节点奖励。此外，DSearch动态调整搜索宽度和树扩展，以高效探索高奖励生成。为了细化中间决策，DSearch结合了基于噪声水平的自适应调度和前瞻启发式函数。我们在生物序列设计、分子优化和图像生成等多个领域对DSearch进行了验证，展示了与现有方法相比的优越奖励优化效果。 

---
# TactStyle: Generating Tactile Textures with Generative AI for Digital Fabrication 

**Title (ZH)**: TactStyle：使用生成式人工智能生成触觉纹理以应用于数字 fabrication 

**Authors**: Faraz Faruqi, Maxine Perroni-Scharf, Jaskaran Singh Walia, Yunyi Zhu, Shuyue Feng, Donald Degraen, Stefanie Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2503.02007)  

**Abstract**: Recent work in Generative AI enables the stylization of 3D models based on image prompts. However, these methods do not incorporate tactile information, leading to designs that lack the expected tactile properties. We present TactStyle, a system that allows creators to stylize 3D models with images while incorporating the expected tactile properties. TactStyle accomplishes this using a modified image-generation model fine-tuned to generate heightfields for given surface textures. By optimizing 3D model surfaces to embody a generated texture, TactStyle creates models that match the desired style and replicate the tactile experience. We utilize a large-scale dataset of textures to train our texture generation model. In a psychophysical experiment, we evaluate the tactile qualities of a set of 3D-printed original textures and TactStyle's generated textures. Our results show that TactStyle successfully generates a wide range of tactile features from a single image input, enabling a novel approach to haptic design. 

**Abstract (ZH)**: 基于图像提示的生成AI Recent Work使3D模型的风格化成为可能，但这些方法没有纳入触觉信息，导致设计缺乏预期的触觉特性。我们提出了TactStyle系统，该系统允许创作人员在使用图像风格化3D模型的同时，纳入预期的触觉特性。TactStyle通过调整 fine-tuned 以生成给定表面纹理的高度场的图像生成模型来实现这一点。通过优化3D模型表面以体现生成的纹理，TactStyle 创建了符合期望样式且复制触觉体验的模型。我们利用大规模纹理数据集来训练我们的纹理生成模型。在一项心理物理实验中，我们评估了一组3D打印原始纹理和TactStyle生成纹理的触觉质量。结果显示，TactStyle 成功地从单个图像输入生成了广泛的触觉特性，提供了一种新颖的触觉设计方法。 

---
# Proportionality in Thumbs Up and Down Voting 

**Title (ZH)**: 拇指点赞与反对投票的比例性 

**Authors**: Sonja Kraiczy, Georgios Papasotiropoulos, Grzegorz Pierczyński, Piotr Skowron  

**Link**: [PDF](https://arxiv.org/pdf/2503.01985)  

**Abstract**: Consider the decision-making setting where agents elect a panel by expressing both positive and negative preferences. Prominently, in constitutional AI, citizens democratically select a slate of ethical preferences on which a foundation model is to be trained. There, in practice, agents may both approve and disapprove of different ethical principles. Proportionality has been well-studied in computational social choice for approval ballots, but its meaning remains unclear when negative sentiments are also considered. In this work, we propose two conceptually distinct approaches to interpret proportionality in the presence of up and down votes. The first approach treats the satisfaction from electing candidates and the impact of vetoing them as comparable, leading to combined proportionality guarantees. The second approach considers veto power separately, introducing guarantees distinct from traditional proportionality. We formalize axioms for each perspective and examine their satisfiability by suitable adaptations of Phragmén's rule, Proportional Approval Voting rule and the Method of Equal Shares. 

**Abstract (ZH)**: 考虑代理通过表达正负偏好来选举委员会的决策设置。在宪法AI中，公民民主地选择一组伦理偏好，作为基础模型的训练依据。在此实践中，代理可能既批准又反对不同的伦理原则。对于有赞同和反对票的批准票，相同比例性在计算社会选择中已被广泛研究，但当也考虑负面情感时，其含义仍然不清楚。在本文中，我们提出了两种概念上不同的方法来解释在有赞成和反对票情况下相同比例性的含义。第一种方法将选举候选人的满意度与其否决投票的影响视为可比的，从而提供综合比例性保证。第二种方法单独考虑否决权，引入不同于传统比例性的保证。我们为每种视角形式化了公理，并通过Phragmén规则、比例性批准投票规则和均等份额方法的适当改编来检查它们的可满足性。 

---
# Mathematical Foundation of Interpretable Equivariant Surrogate Models 

**Title (ZH)**: 可解释等变替代模型的数学基础 

**Authors**: Jacopo Joy Colombini, Filippo Bonchi, Francesco Giannini, Fosca Giannotti, Roberto Pellungrini, Patrizio Frosini  

**Link**: [PDF](https://arxiv.org/pdf/2503.01942)  

**Abstract**: This paper introduces a rigorous mathematical framework for neural network explainability, and more broadly for the explainability of equivariant operators called Group Equivariant Operators (GEOs) based on Group Equivariant Non-Expansive Operators (GENEOs) transformations. The central concept involves quantifying the distance between GEOs by measuring the non-commutativity of specific diagrams. Additionally, the paper proposes a definition of interpretability of GEOs according to a complexity measure that can be defined according to each user preferences. Moreover, we explore the formal properties of this framework and show how it can be applied in classical machine learning scenarios, like image classification with convolutional neural networks. 

**Abstract (ZH)**: 基于Group Equivariant Non-Expansive Operators（GENEOs）变换的Group Equivariant Operators（GEOs）可解释性严格数学框架研究 

---
# Task Scheduling & Forgetting in Multi-Task Reinforcement Learning 

**Title (ZH)**: 多任务强化学习中的任务调度与遗忘 

**Authors**: Marc Speckmann, Theresa Eimer  

**Link**: [PDF](https://arxiv.org/pdf/2503.01941)  

**Abstract**: Reinforcement learning (RL) agents can forget tasks they have previously been trained on. There is a rich body of work on such forgetting effects in humans. Therefore we look for commonalities in the forgetting behavior of humans and RL agents across tasks and test the viability of forgetting prevention measures from learning theory in RL. We find that in many cases, RL agents exhibit forgetting curves similar to those of humans. Methods like Leitner or SuperMemo have been shown to be effective at counteracting human forgetting, but we demonstrate they do not transfer as well to RL. We identify a likely cause: asymmetrical learning and retention patterns between tasks that cannot be captured by retention-based or performance-based curriculum strategies. 

**Abstract (ZH)**: 强化学习代理可能会忘记之前训练的任务。人类在任务中的遗忘效应已有丰富的研究。因此，我们在不同任务中寻找人类和强化学习代理遗忘行为的共同点，并测试学习理论中的遗忘预防措施在强化学习中的有效性。我们发现，在许多情况下，强化学习代理的遗忘曲线与人类相似。Leitner或SuperMemo等方法已被证明对人类遗忘有有效的对抗作用，但我们表明这些方法在转移到强化学习中并不如预期有效。我们确定了可能的原因：任务之间不对称的学习和保持模式，这些模式不能被基于保持或基于性能的课程策略捕获。 

---
# Synthetic Tabular Data Detection In the Wild 

**Title (ZH)**: 合成表格数据的野生环境检测 

**Authors**: G. Charbel N. Kindji, Elisa Fromont, Lina Maria Rojas-Barahona, Tanguy Urvoy  

**Link**: [PDF](https://arxiv.org/pdf/2503.01937)  

**Abstract**: Detecting synthetic tabular data is essential to prevent the distribution of false or manipulated datasets that could compromise data-driven decision-making. This study explores whether synthetic tabular data can be reliably identified across different tables. This challenge is unique to tabular data, where structures (such as number of columns, data types, and formats) can vary widely from one table to another. We propose four table-agnostic detectors combined with simple preprocessing schemes that we evaluate on six evaluation protocols, with different levels of ''wildness''. Our results show that cross-table learning on a restricted set of tables is possible even with naive preprocessing schemes. They confirm however that cross-table transfer (i.e. deployment on a table that has not been seen before) is challenging. This suggests that sophisticated encoding schemes are required to handle this problem. 

**Abstract (ZH)**: 检测合成表格数据对于防止分发虚假或操纵的数据集、确保数据驱动决策的安全至关重要。本研究探讨了合成表格数据是否能在不同表格间可靠地被识别。这一挑战性问题特别适用于表格数据，因为各张表格的结构（如列数、数据类型和格式）差异很大。我们提出了一种结合简单预处理方案的四款跨表通用检测器，并在六个不同“野度”的评估协议上进行了评估。结果显示，即使使用简单的预处理方案，对有限几张表格的跨表学习也是可行的。然而，这些结果也证实了跨表迁移（即在未见过的表格上部署）的挑战性。这表明，为了解决这一问题，需要采用复杂的编码方案。 

---
# Decision-Focused Fine-Tuning of Time Series Foundation Models for Dispatchable Feeder Optimization 

**Title (ZH)**: 基于决策的时间序列基础模型可调度配电线优化的微调方法 

**Authors**: Maximilian Beichter, Nils Friederich, Janik Pinter, Dorina Werling, Kaleb Phipps, Sebastian Beichter, Oliver Neumann, Ralf Mikut, Veit Hagenmeyer, Benedikt Heidrich  

**Link**: [PDF](https://arxiv.org/pdf/2503.01936)  

**Abstract**: Time series foundation models provide a universal solution for generating forecasts to support optimization problems in energy systems. Those foundation models are typically trained in a prediction-focused manner to maximize forecast quality. In contrast, decision-focused learning directly improves the resulting value of the forecast in downstream optimization rather than merely maximizing forecasting quality. The practical integration of forecast values into forecasting models is challenging, particularly when addressing complex applications with diverse instances, such as buildings. This becomes even more complicated when instances possess specific characteristics that require instance-specific, tailored predictions to increase the forecast value. To tackle this challenge, we use decision-focused fine-tuning within time series foundation models to offer a scalable and efficient solution for decision-focused learning applied to the dispatchable feeder optimization problem. To obtain more robust predictions for scarce building data, we use Moirai as a state-of-the-art foundation model, which offers robust and generalized results with few-shot parameter-efficient fine-tuning. Comparing the decision-focused fine-tuned Moirai with a state-of-the-art classical prediction-focused fine-tuning Morai, we observe an improvement of 9.45% in average total daily costs. 

**Abstract (ZH)**: 时间序列基础模型为能源系统中的优化问题提供了一种通用的预测解决方案。与预测导向的学习相比，决策导向的学习直接通过改进预测值在下游优化中的结果来提高决策质量，而不是仅追求最佳预测质量。将预测值集成到预测模型中以满足复杂且多样化应用的需求（如建筑物）具有挑战性。当实例具有特定特征需要实例特定的预测以提高预测值时，这一挑战更为复杂。为解决这一挑战，我们利用时间序列基础模型中的决策导向微调来提供一种可扩展且高效的解决方案，应用于调度支路优化问题的决策导向学习。为了获得更 robust 的预测结果，我们使用 Moirai 作为最先进的基础模型，该模型通过少量样本高效的参数微调提供 robust 和通用的结果。在平均总日成本方面，与最先进的经典预测导向微调模型 Morai 相比，决策导向微调的 Moirai 显示出 9.45% 的改进。 

---
# Adversarial Generative Flow Network for Solving Vehicle Routing Problems 

**Title (ZH)**: 对抗生成流网络解决车辆路线问题 

**Authors**: Ni Zhang, Jingfeng Yang, Zhiguang Cao, Xu Chi  

**Link**: [PDF](https://arxiv.org/pdf/2503.01931)  

**Abstract**: Recent research into solving vehicle routing problems (VRPs) has gained significant traction, particularly through the application of deep (reinforcement) learning for end-to-end solution construction. However, many current construction-based neural solvers predominantly utilize Transformer architectures, which can face scalability challenges and struggle to produce diverse solutions. To address these limitations, we introduce a novel framework beyond Transformer-based approaches, i.e., Adversarial Generative Flow Networks (AGFN). This framework integrates the generative flow network (GFlowNet)-a probabilistic model inherently adept at generating diverse solutions (routes)-with a complementary model for discriminating (or evaluating) the solutions. These models are trained alternately in an adversarial manner to improve the overall solution quality, followed by a proposed hybrid decoding method to construct the solution. We apply the AGFN framework to solve the capacitated vehicle routing problem (CVRP) and travelling salesman problem (TSP), and our experimental results demonstrate that AGFN surpasses the popular construction-based neural solvers, showcasing strong generalization capabilities on synthetic and real-world benchmark instances. 

**Abstract (ZH)**: 基于对抗生成流网络的车辆路由问题求解研究 

---
# QCS-ADME: Quantum Circuit Search for Drug Property Prediction with Imbalanced Data and Regression Adaptation 

**Title (ZH)**: QCS-ADME：量子电路搜索在不均衡数据和回归适应下的药物性质预测 

**Authors**: Kangyu Zheng, Tianfan Fu, Zhiding Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01927)  

**Abstract**: The biomedical field is beginning to explore the use of quantum machine learning (QML) for tasks traditionally handled by classical machine learning, especially in predicting ADME (absorption, distribution, metabolism, and excretion) properties, which are essential in drug evaluation. However, ADME tasks pose unique challenges for existing quantum computing systems (QCS) frameworks, as they involve both classification with unbalanced dataset and regression problems. These dual requirements make it necessary to adapt and refine current QCS frameworks to effectively address the complexities of ADME predictions. We propose a novel training-free scoring mechanism to evaluate QML circuit performance on imbalanced classification and regression tasks. Our mechanism demonstrates significant correlation between scoring metrics and test performance on imbalanced classification tasks. Additionally, we develop methods to quantify continuous similarity relationships between quantum states, enabling performance prediction for regression tasks. This represents the first comprehensive approach to searching and evaluating QCS circuits specifically for regression applications. Validation on representative ADME tasks-one imbalanced classification and one regression-demonstrates moderate positive correlation between our scoring metrics and circuit performance, significantly outperforming baseline scoring methods that show negligible correlation. 

**Abstract (ZH)**: 生物医药领域开始探索使用量子机器学习（QML）来处理传统由经典机器学习处理的任务，特别是在预测ADME（吸收、分布、代谢、排泄）属性方面，这些属性对于药物评价至关重要。然而，ADME任务为现有的量子计算系统框架带来了独特的挑战，因为它们同时涉及不平衡数据集的分类问题和回归问题。这些双重要求使得适应和改进现有的量子计算系统框架变得必要，以有效应对ADME预测的复杂性。我们提出了一种新的无需训练的评分机制，用于评估QML电路在不平衡分类和回归任务中的性能。我们的机制在不平衡分类任务中显示了评分指标与测试性能之间显著的相关性。此外，我们开发了量化量子态之间连续相似关系的方法，从而能够预测回归任务的性能。这代表了第一个专门针对回归应用搜索和评估量子计算系统电路的全面方法。代表性的ADME任务（一个不平衡分类和一个回归任务）的验证显示，我们的评分指标与电路性能之间存在适度的正相关，显著优于基准评分方法，后者显示几乎不存在相关性。 

---
# TAET: Two-Stage Adversarial Equalization Training on Long-Tailed Distributions 

**Title (ZH)**: TAET：两阶段对抗均衡训练在长尾分布上的应用 

**Authors**: Wang YuHang, Junkang Guo, Aolei Liu, Kaihao Wang, Zaitong Wu, Zhenyu Liu, Wenfei Yin, Jian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01924)  

**Abstract**: Adversarial robustness is a critical challenge in deploying deep neural networks for real-world applications. While adversarial training is a widely recognized defense strategy, most existing studies focus on balanced datasets, overlooking the prevalence of long-tailed distributions in real-world data, which significantly complicates robustness. This paper provides a comprehensive analysis of adversarial training under long-tailed distributions and identifies limitations in the current state-of-the-art method, AT-BSL, in achieving robust performance under such conditions. To address these challenges, we propose a novel training framework, TAET, which integrates an initial stabilization phase followed by a stratified equalization adversarial training phase. Additionally, prior work on long-tailed robustness has largely ignored the crucial evaluation metric of balanced accuracy. To bridge this gap, we introduce the concept of balanced robustness, a comprehensive metric tailored for assessing robustness under long-tailed distributions. Extensive experiments demonstrate that our method surpasses existing advanced defenses, achieving significant improvements in both memory and computational efficiency. This work represents a substantial advancement in addressing robustness challenges in real-world applications. Our code is available at: this https URL. 

**Abstract (ZH)**: adversarial稳健性是将深度神经网络应用于实际应用中的一个关键挑战。虽然已有研究表明对抗训练是一种广为人知的防御策略，但大多数现有研究集中在均衡数据集上，忽视了实际数据中长尾分布的普遍性，这极大地增加了稳健性的复杂性。本文对长尾分布下的对抗训练进行了全面分析，并指出现有最先进的方法AT-BSL在实现此类条件下的稳健性能时存在局限性。为应对这些挑战，我们提出了一种新颖的训练框架TAET，该框架包括一个初始稳定阶段，随后是分层等化对抗训练阶段。此外，关于长尾稳健性的现有研究很大程度上忽略了平衡准确率这一关键评估指标。为弥补这一不足，我们引入了平衡稳健性的概念，这是一种专门用于评估长尾分布下稳健性的综合指标。广泛的实验表明，我们的方法在内存和计算效率上均显著优于现有先进的防御方法。这项工作代表了在实际应用中解决稳健性挑战的一个重要进展。我们的代码可在以下链接获取：this https URL。 

---
# Reinforcement learning with combinatorial actions for coupled restless bandits 

**Title (ZH)**: 组合动作强化学习在耦合不安定bandits中的应用 

**Authors**: Lily Xu, Bryan Wilder, Elias B. Khalil, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2503.01919)  

**Abstract**: Reinforcement learning (RL) has increasingly been applied to solve real-world planning problems, with progress in handling large state spaces and time horizons. However, a key bottleneck in many domains is that RL methods cannot accommodate large, combinatorially structured action spaces. In such settings, even representing the set of feasible actions at a single step may require a complex discrete optimization formulation. We leverage recent advances in embedding trained neural networks into optimization problems to propose SEQUOIA, an RL algorithm that directly optimizes for long-term reward over the feasible action space. Our approach embeds a Q-network into a mixed-integer program to select a combinatorial action in each timestep. Here, we focus on planning over restless bandits, a class of planning problems which capture many real-world examples of sequential decision making. We introduce coRMAB, a broader class of restless bandits with combinatorial actions that cannot be decoupled across the arms of the restless bandit, requiring direct solving over the joint, exponentially large action space. We empirically validate SEQUOIA on four novel restless bandit problems with combinatorial constraints: multiple interventions, path constraints, bipartite matching, and capacity constraints. Our approach significantly outperforms existing methods -- which cannot address sequential planning and combinatorial selection simultaneously -- by an average of 26.4% on these difficult instances. 

**Abstract (ZH)**: 强化学习（RL）越来越被应用于解决实际规划问题，尤其是在处理大型状态空间和时间 horizon 方面取得了进展。然而，在许多领域中，强化学习方法无法容纳大规模的组合结构动作空间。在这种情况下，即使在单个时间步表示可行动作集也可能需要复杂的离散优化建模。我们利用最新将训练好的神经网络嵌入到优化问题中的进展，提出了一种名为 SEQUOIA 的强化学习算法，该算法可以直接优化长期奖励，目标是可行动作空间。我们的方法将 Q 网络嵌入混合整数规划中，在每个时间步选择一个组合动作。我们关注的是不朽-bedenech（restless bandit）类规划问题，这类问题涵盖了大量现实世界中的序贯决策实例。我们引入了带组合动作的共RMAB（coRMAB），这是一种更广泛的不朽-bedenech 类问题，其中的动作不能在不朽-bedenech 的各个臂之间解耦，需要直接解决联合的、指数级大的动作空间。我们在四个带有组合约束的新型不朽-bedenech 问题上实证验证了 SEQUOIA：多干预、路径约束、二分匹配和容量约束。我们的方法在这些困难实例中平均表现比现有方法（无法同时解决序贯规划和组合选择问题）提高了 26.4%。 

---
# Conceptual Contrastive Edits in Textual and Vision-Language Retrieval 

**Title (ZH)**: 文本和跨模态检索中的概念对比编辑 

**Authors**: Maria Lymperaiou, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2503.01914)  

**Abstract**: As deep learning models grow in complexity, achieving model-agnostic interpretability becomes increasingly vital. In this work, we employ post-hoc conceptual contrastive edits to expose noteworthy patterns and biases imprinted in representations of retrieval models. We systematically design optimal and controllable contrastive interventions targeting various parts of speech, and effectively apply them to explain both linguistic and visiolinguistic pre-trained models in a black-box manner. Additionally, we introduce a novel metric to assess the per-word impact of contrastive interventions on model outcomes, providing a comprehensive evaluation of each intervention's effectiveness. 

**Abstract (ZH)**: 随着深度学习模型变得日益复杂，实现模型通用可解释性变得越来越重要。在本文中，我们采用事后概念对比编辑方法揭示检索模型表示中突出的模式和偏见。我们系统地设计了针对各种词性的最优且可控的对比干预措施，并以黑盒方式有效地应用于解释语言和语意图像预训练模型。此外，我们引入了一个新的度量标准来评估对比干预措施对模型结果的单词影响，从而全面评估每种干预措施的效果。 

---
# dyAb: Flow Matching for Flexible Antibody Design with AlphaFold-driven Pre-binding Antigen 

**Title (ZH)**: dyAb: 基于AlphaFold驱动的预结合抗原的流式抗体设计 

**Authors**: Cheng Tan, Yijie Zhang, Zhangyang Gao, Yufei Huang, Haitao Lin, Lirong Wu, Fandi Wu, Mathieu Blanchette, Stan. Z. Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.01910)  

**Abstract**: The development of therapeutic antibodies heavily relies on accurate predictions of how antigens will interact with antibodies. Existing computational methods in antibody design often overlook crucial conformational changes that antigens undergo during the binding process, significantly impacting the reliability of the resulting antibodies. To bridge this gap, we introduce dyAb, a flexible framework that incorporates AlphaFold2-driven predictions to model pre-binding antigen structures and specifically addresses the dynamic nature of antigen conformation changes. Our dyAb model leverages a unique combination of coarse-grained interface alignment and fine-grained flow matching techniques to simulate the interaction dynamics and structural evolution of the antigen-antibody complex, providing a realistic representation of the binding process. Extensive experiments show that dyAb significantly outperforms existing models in antibody design involving changing antigen conformations. These results highlight dyAb's potential to streamline the design process for therapeutic antibodies, promising more efficient development cycles and improved outcomes in clinical applications. 

**Abstract (ZH)**: 基于抗原构象变化预测的治疗性抗体开发 

---
# Attend or Perish: Benchmarking Attention in Algorithmic Reasoning 

**Title (ZH)**: Attendance or Perish: 评估算法推理中的注意力机制 

**Authors**: Michal Spiegel, Michal Štefánik, Marek Kadlčík, Josef Kuchař  

**Link**: [PDF](https://arxiv.org/pdf/2503.01909)  

**Abstract**: Can transformers learn to perform algorithmic tasks reliably across previously unseen input/output domains? While pre-trained language models show solid accuracy on benchmarks incorporating algorithmic reasoning, assessing the reliability of these results necessitates an ability to cleanse models' functional capabilities from memorization. In this paper, we propose an algorithmic benchmark comprising six tasks of infinite input domains where we can also disentangle and trace the correct, robust algorithm necessary for the task. This allows us to assess (i) models' ability to extrapolate to unseen types of inputs, including new lengths, value ranges or input domains, but also (ii) to assess the robustness of the functional mechanism in recent models through the lens of their attention maps. We make the implementation of all our tasks and interoperability methods publicly available at this https URL . 

**Abstract (ZH)**: Transformers能否可靠地执行跨未见过的输入/输出领域的算法任务？虽然预训练语言模型在包含算法推理的基准测试中表现出色，但评估这些结果的可靠性需要能够清除模型的功能能力中的记忆现象。在本文中，我们提出了一种算法基准，包括六个具有无限输入域的任务，我们可以在其中拆分并追踪完成任务所需的正确且稳健的算法。这使我们能够评估（i）模型能否将任务扩展到未见过的输入类型，包括新的长度、值范围或输入域，以及（ii）通过注意力图评估最近模型的功能机制的鲁棒性。我们在<这个链接>公开了所有任务的实现和互操作方法。 

---
# Learning to Chain Operations by Routing Information Through a Global Workspace 

**Title (ZH)**: 通过全局工作空间传输信息以学习串联操作 

**Authors**: Hugo Chateau-Laurent, Rufin VanRullen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01906)  

**Abstract**: We present a model inspired by the Global Workspace Theory that integrates specialized modules to perform a sequential reasoning task. A controller selectively routes information between modules through the workspace using a gating mechanism. This approach allows the model to chain operations by iteratively broadcasting information between specialized domains, mimicking System-2 reasoning. We evaluate the model's performance on a simple addition task, where two addends must be summed. The task can be solved by routing information sequentially through an Input module, an Increment module (multiple times), and finally an Output module. We consider two implementations of this system with increasing complexity. First, using hand-designed modules operating on one-hot digit representations, the controller (a LSTM recurrent network) learns to select the appropriate modules (input, increment, output) in the appropriate sequence. Second, we replace the hand-designed modules with learned representation modules for MNIST images and an increment module trained on the task objectives; here again, the controller learns the appropriate sequential module selection to solve the task. Finally, we show that the Global Workspace model, while having fewer parameters, outperforms LSTMs and Transformers when tested on unseen addition operations (both interpolations and extrapolations of addition operations seen during training). Our results highlight the potential of architectures inspired by the Global Workspace Theory to enhance deep learning's reasoning capabilities. 

**Abstract (ZH)**: 基于全局工作区理论的序列推理模型研究 

---
# PaCA: Partial Connection Adaptation for Efficient Fine-Tuning 

**Title (ZH)**: PaCA: 部分连接适应性调整以实现高效的微调 

**Authors**: Sunghyeon Woo, Sol Namkung, Sunwoo Lee, Inho Jeong, Beomseok Kim, Dongsuk Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2503.01905)  

**Abstract**: Prior parameter-efficient fine-tuning (PEFT) algorithms reduce memory usage and computational costs of fine-tuning large neural network models by training only a few additional adapter parameters, rather than the entire model. However, the reduction in computational costs due to PEFT does not necessarily translate to a reduction in training time; although the computational costs of the adapter layers are much smaller than the pretrained layers, it is well known that those two types of layers are processed sequentially on GPUs, resulting in significant latency overhead. LoRA and its variants merge low-rank adapter matrices with pretrained weights during inference to avoid latency overhead, but during training, the pretrained weights remain frozen while the adapter matrices are continuously updated, preventing such merging. To mitigate this issue, we propose Partial Connection Adaptation (PaCA), which fine-tunes randomly selected partial connections within the pretrained weights instead of introducing adapter layers in the model. PaCA not only enhances training speed by eliminating the time overhead due to the sequential processing of the adapter and pretrained layers but also reduces activation memory since only partial activations, rather than full activations, need to be stored for gradient computation. Compared to LoRA, PaCA reduces training time by 22% and total memory usage by 16%, while maintaining comparable accuracy across various fine-tuning scenarios, such as fine-tuning on the MMLU dataset and instruction tuning on the Oasst1 dataset. PaCA can also be combined with quantization, enabling the fine-tuning of large models such as LLaMA3.1-70B. In addition, PaCA enables training with 23% longer sequence and improves throughput by 16% on both NVIDIA A100 GPU and INTEL Gaudi2 HPU compared to LoRA. The code is available at this https URL. 

**Abstract (ZH)**: 部分连接适应（PaCA）：一种减少训练时间与内存使用的新方法 

---
# Continual Learning-Aided Super-Resolution Scheme for Channel Reconstruction and Generalization in OFDM Systems 

**Title (ZH)**: 基于连续学习的OFDM系统信道重建与泛化超分辨率方案 

**Authors**: Jianqiao Chen, Nan Ma, Wenkai Liu, Xiaodong Xu, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01897)  

**Abstract**: Channel reconstruction and generalization capability are of equal importance for developing channel estimation schemes within deep learning (DL) framework. In this paper, we exploit a novel DL-based scheme for efficient OFDM channel estimation where the neural networks for channel reconstruction and generalization are respectively designed. For the former, we propose a dual-attention-aided super-resolution neural network (DA-SRNN) to map the channels at pilot positions to the whole time-frequency channels. Specifically, the channel-spatial attention mechanism is first introduced to sequentially infer attention maps along two separate dimensions corresponding to two types of underlying channel correlations, and then the lightweight SR module is developed for efficient channel reconstruction. For the latter, we introduce continual learning (CL)-aided training strategies to make the neural network adapt to different channel distributions. Specifically, the elastic weight consolidation (EWC) is introduced as the regularization term in regard to loss function of channel reconstruction, which can constrain the direction and space of updating the important weights of neural networks among different channel distributions. Meanwhile, the corresponding training process is provided in detail. By evaluating under 3rd Generation Partnership Project (3GPP) channel models, numerical results verify the superiority of the proposed channel estimation scheme with significantly improved channel reconstruction and generalization performance over counterparts. 

**Abstract (ZH)**: 基于深度学习的信道估计方案中，信道重建和泛化能力具有同等重要性。本文提出了一种新颖的基于深度学习的OFDM信道估计算法，在该算法中分别设计了适用于信道重建和泛化的神经网络。对于信道重建，我们提出了一种双注意力辅助超分辨率神经网络（DA-SRNN），用于将导频位置的信道映射到整个时频信道。具体来说，首先引入了信道-空间注意力机制，以逐步推断出与两种不同信道相关性类型对应的注意力图，然后开发了轻量级的超分辨率模块以实现高效的信道重建。对于泛化能力，我们引入了连续学习（CL）辅助训练策略，使得神经网络能够适应不同的信道分布。具体而言，我们引入了弹性权重聚合（EWC）作为重构损失函数的正则化项，可以约束不同信道分布下重要权重更新的方向和空间。同时，详细描述了相应的训练过程。通过在3GPP信道模型下的评估，数值结果验证了所提信道估计算法在信道重建和泛化性能上的优越性。 

---
# Evaluating System 1 vs. 2 Reasoning Approaches for Zero-Shot Time-Series Forecasting: A Benchmark and Insights 

**Title (ZH)**: 评估系统1 vs. 系统2推理方法在零样本时间序列预测中的性能：基准与见解 

**Authors**: Haoxin Liu, Zhiyuan Zhao, Shiduo Li, B. Aditya Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2503.01895)  

**Abstract**: Reasoning ability is crucial for solving challenging tasks. With the advancement of foundation models, such as the emergence of large language models (LLMs), a wide range of reasoning strategies has been proposed, including test-time enhancements, such as Chain-ofThought, and post-training optimizations, as used in DeepSeek-R1. While these reasoning strategies have demonstrated effectiveness across various challenging language or vision tasks, their applicability and impact on time-series forecasting (TSF), particularly the challenging zero-shot TSF, remain largely unexplored. In particular, it is unclear whether zero-shot TSF benefits from reasoning and, if so, what types of reasoning strategies are most effective. To bridge this gap, we propose ReC4TS, the first benchmark that systematically evaluates the effectiveness of popular reasoning strategies when applied to zero-shot TSF tasks. ReC4TS conducts comprehensive evaluations across datasets spanning eight domains, covering both unimodal and multimodal with short-term and longterm forecasting tasks. More importantly, ReC4TS provides key insights: (1) Self-consistency emerges as the most effective test-time reasoning strategy; (2) Group-relative policy optimization emerges as a more suitable approach for incentivizing reasoning ability during post-training; (3) Multimodal TSF benefits more from reasoning strategies compared to unimodal TSF. Beyond these insights, ReC4TS establishes two pioneering starting blocks to support future zero-shot TSF reasoning research: (1) A novel dataset, TimeThinking, containing forecasting samples annotated with reasoning trajectories from multiple advanced LLMs, and (2) A new and simple test-time scaling-law validated on foundational TSF models enabled by self-consistency reasoning strategy. All data and code are publicly accessible at: this https URL 

**Abstract (ZH)**: ReC4TS：系统评估推理策略在零样本时间序列预测中的有效性 

---
# LIVS: A Pluralistic Alignment Dataset for Inclusive Public Spaces 

**Title (ZH)**: LIVS: 包容性公共空间多样共识数据集 

**Authors**: Rashid Mushkani, Shravan Nayak, Hugo Berard, Allison Cohen, Shin Koseki, Hadrien Bertrand  

**Link**: [PDF](https://arxiv.org/pdf/2503.01894)  

**Abstract**: We introduce the Local Intersectional Visual Spaces (LIVS) dataset, a benchmark for multi-criteria alignment of text-to-image (T2I) models in inclusive urban planning. Developed through a two-year participatory process with 30 community organizations, LIVS encodes diverse spatial preferences across 634 initial concepts, consolidated into six core criteria: Accessibility, Safety, Comfort, Invitingness, Inclusivity, and Diversity, through 37,710 pairwise comparisons. Using Direct Preference Optimization (DPO) to fine-tune Stable Diffusion XL, we observed a measurable increase in alignment with community preferences, though a significant proportion of neutral ratings highlights the complexity of modeling intersectional needs. Additionally, as annotation volume increases, accuracy shifts further toward the DPO-tuned model, suggesting that larger-scale preference data enhances fine-tuning effectiveness. LIVS underscores the necessity of integrating context-specific, stakeholder-driven criteria into generative modeling and provides a resource for evaluating AI alignment methodologies across diverse socio-spatial contexts. 

**Abstract (ZH)**: 我们介绍了局部交叉视域数据集（LIVS），这是一个用于包容性城市规划中多准则文本到图像（T2I）模型对齐的基准数据集。通过与30个社区组织进行为期两年的参与式过程开发，LIVS 编码了634个初始概念中的多样化空间偏好，并通过37,710对两两比较将其整合为六项核心标准：可达性、安全性、舒适性、亲和性、包容性和多样性。利用直接偏好优化（DPO）对Stable Diffusion XL进行微调，我们观察到与社区偏好对齐的可测量增加，尽管大量的中间评分突显了建模交叉需求的复杂性。此外，随着注释量的增加，准确性更加倾向于DPO微调模型，表明大规模偏好数据增强了微调效果。LIVS 强调了将具体情境和利益相关者驱动的标准整合到生成建模中的必要性，并提供了一个跨多元社会空间环境评估AI对齐方法论的资源。 

---
# Enhancing Transformer with GNN Structural Knowledge via Distillation: A Novel Approach 

**Title (ZH)**: 基于蒸馏的变换器增强新方法：通过图神经网络结构知识 

**Authors**: Zhihua Duan, Jialin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01888)  

**Abstract**: Integrating the structural inductive biases of Graph Neural Networks (GNNs) with the global contextual modeling capabilities of Transformers represents a pivotal challenge in graph representation learning. While GNNs excel at capturing localized topological patterns through message-passing mechanisms, their inherent limitations in modeling long-range dependencies and parallelizability hinder their deployment in large-scale scenarios. Conversely, Transformers leverage self-attention mechanisms to achieve global receptive fields but struggle to inherit the intrinsic graph structural priors of GNNs. This paper proposes a novel knowledge distillation framework that systematically transfers multiscale structural knowledge from GNN teacher models to Transformer student models, offering a new perspective on addressing the critical challenges in cross-architectural distillation. The framework effectively bridges the architectural gap between GNNs and Transformers through micro-macro distillation losses and multiscale feature alignment. This work establishes a new paradigm for inheriting graph structural biases in Transformer architectures, with broad application prospects. 

**Abstract (ZH)**: 将图神经网络（GNNs）的结构诱导偏置与变换器的全局上下文建模能力集成是图表示学习中的一个关键挑战。尽管GNNs通过消息传递机制擅长捕捉局部拓扑模式，但它们在建模长程依赖关系和并行化方面的固有局限性限制了其在大规模场景中的部署。相反，变换器通过自注意力机制实现了全局的感受野但难以继承GNNs的内在图结构先验。本文提出了一种新的知识蒸馏框架，系统地将多尺度结构知识从GNN教师模型转移到变换器学生模型，为跨架构蒸馏的关键挑战提供了新的视角。该框架通过微宏观蒸馏损失和多尺度特征对齐有效地弥合了GNNs和变换器之间的架构差距。这项工作为在变换器架构中继承图结构偏置建立了新的范式，具有广泛的应用前景。 

---
# Advanced Deep Learning Techniques for Analyzing Earnings Call Transcripts: Methodologies and Applications 

**Title (ZH)**: 基于财务报告电话会议文本分析的先进深度学习技术：方法与应用 

**Authors**: Umair Zakir, Evan Daykin, Amssatou Diagne, Jacob Faile  

**Link**: [PDF](https://arxiv.org/pdf/2503.01886)  

**Abstract**: This study presents a comparative analysis of deep learning methodologies such as BERT, FinBERT and ULMFiT for sentiment analysis of earnings call transcripts. The objective is to investigate how Natural Language Processing (NLP) can be leveraged to extract sentiment from large-scale financial transcripts, thereby aiding in more informed investment decisions and risk management strategies. We examine the strengths and limitations of each model in the context of financial sentiment analysis, focusing on data preprocessing requirements, computational efficiency, and model optimization. Through rigorous experimentation, we evaluate their performance using key metrics, including accuracy, precision, recall, and F1-score. Furthermore, we discuss potential enhancements to improve the effectiveness of these models in financial text analysis, providing insights into their applicability for real-world financial decision-making. 

**Abstract (ZH)**: 本研究对BERT、FinBERT和ULMFiT等深度学习方法在收益电话会议转录文本情感分析中的应用进行比较分析，旨在探讨自然语言处理（NLP）如何通过提取大规模财务转录文中的情感信息，辅助更明智的投资决策和风险管理策略。我们考察了每种模型在金融情感分析中的优势与局限性，重点包括数据预处理要求、计算效率和模型优化。通过严格的实证研究，我们使用准确率、精确率、召回率和F1分数等关键指标评估其性能。此外，我们讨论了改进这些模型在金融文本分析中有效性的潜在方法，提供了它们在实际金融决策中应用的洞察。 

---
# Learning Policy Committees for Effective Personalization in MDPs with Diverse Tasks 

**Title (ZH)**: 学习策略委员会以实现MDPs中多样化任务的有效个性化 

**Authors**: Luise Ge, Michael Lanier, Anindya Sarkar, Bengisu Guresti, Yevgeniy Vorobeychik, Chongjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01885)  

**Abstract**: Many dynamic decision problems, such as robotic control, involve a series of tasks, many of which are unknown at training time. Typical approaches for these problems, such as multi-task and meta reinforcement learning, do not generalize well when the tasks are diverse. On the other hand, approaches that aim to tackle task diversity, such as using task embedding as policy context and task clustering, typically lack performance guarantees and require a large number of training tasks. To address these challenges, we propose a novel approach for learning a policy committee that includes at least one near-optimal policy with high probability for tasks encountered during execution. While we show that this problem is in general inapproximable, we present two practical algorithmic solutions. The first yields provable approximation and task sample complexity guarantees when tasks are low-dimensional (the best we can do due to inapproximability), whereas the second is a general and practical gradient-based approach. In addition, we provide a provable sample complexity bound for few-shot learning. Our experiments on MuJoCo and Meta-World show that the proposed approach outperforms state-of-the-art multi-task, meta-, and task clustering baselines in training, generalization, and few-shot learning, often by a large margin. 

**Abstract (ZH)**: 一种学习执行过程中遇到的任务至少包含一个高概率近最优策略的政策委员会的新方法：克服多样性挑战 

---
# Contextual Quantum Neural Networks for Stock Price Prediction 

**Title (ZH)**: 基于上下文的量子神经网络股票价格预测 

**Authors**: Sharan Mourya, Hannes Leipold, Bibhas Adhikari  

**Link**: [PDF](https://arxiv.org/pdf/2503.01884)  

**Abstract**: In this paper, we apply quantum machine learning (QML) to predict the stock prices of multiple assets using a contextual quantum neural network. Our approach captures recent trends to predict future stock price distributions, moving beyond traditional models that focus on entire historical data, enhancing adaptability and precision. Utilizing the principles of quantum superposition, we introduce a new training technique called the quantum batch gradient update (QBGU), which accelerates the standard stochastic gradient descent (SGD) in quantum applications and improves convergence. Consequently, we propose a quantum multi-task learning (QMTL) architecture, specifically, the share-and-specify ansatz, that integrates task-specific operators controlled by quantum labels, enabling the simultaneous and efficient training of multiple assets on the same quantum circuit as well as enabling efficient portfolio representation with logarithmic overhead in the number of qubits. This architecture represents the first of its kind in quantum finance, offering superior predictive power and computational efficiency for multi-asset stock price forecasting. Through extensive experimentation on S\&P 500 data for Apple, Google, Microsoft, and Amazon stocks, we demonstrate that our approach not only outperforms quantum single-task learning (QSTL) models but also effectively captures inter-asset correlations, leading to enhanced prediction accuracy. Our findings highlight the transformative potential of QML in financial applications, paving the way for more advanced, resource-efficient quantum algorithms in stock price prediction and other complex financial modeling tasks. 

**Abstract (ZH)**: 基于上下文的量子神经网络在多重资产股票价格预测中的量子机器学习应用 

---
# Learning Surrogates for Offline Black-Box Optimization via Gradient Matching 

**Title (ZH)**: 基于梯度匹配的 Offline 黑盒优化代理学习 

**Authors**: Minh Hoang, Azza Fadhel, Aryan Deshwal, Janardhan Rao Doppa, Trong Nghia Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01883)  

**Abstract**: Offline design optimization problem arises in numerous science and engineering applications including material and chemical design, where expensive online experimentation necessitates the use of in silico surrogate functions to predict and maximize the target objective over candidate designs. Although these surrogates can be learned from offline data, their predictions are often inaccurate outside the offline data regime. This challenge raises a fundamental question about the impact of imperfect surrogate model on the performance gap between its optima and the true optima, and to what extent the performance loss can be mitigated. Although prior work developed methods to improve the robustness of surrogate models and their associated optimization processes, a provably quantifiable relationship between an imperfect surrogate and the corresponding performance gap, as well as whether prior methods directly address it, remain elusive. To shed light on this important question, we present a theoretical framework to understand offline black-box optimization, by explicitly bounding the optimization quality based on how well the surrogate matches the latent gradient field that underlines the offline data. Inspired by our theoretical analysis, we propose a principled black-box gradient matching algorithm to create effective surrogate models for offline optimization, improving over prior approaches on various real-world benchmarks. 

**Abstract (ZH)**: 离线设计优化问题在材料和化学设计等科学与工程应用中普遍存在，其中昂贵的在线实验需要使用计算仿真的代理函数来预测并最大化目标性能。尽管这些代理函数可以基于离线数据进行学习，但它们在离线数据范围之外的预测往往不够准确。这一挑战引发了关于不完善的代理模型对其最优解与真实最优解之间的性能差距的影响以及性能损失可减少程度的基本问题。尽管已有研究发展了改进代理模型稳健性及其优化过程的方法，但不完善代理模型与相应性能差距之间的可证明可量化关系，以及先前方法是否直接解决该问题依然不清楚。为了揭示这一重要问题，我们提出了一种理论框架来理解离线黑盒优化问题，并通过明确界定制约代理函数与潜在梯度场的一致性来评估优化质量。受到理论分析的启发，我们提出了一个原则性的黑盒梯度匹配算法来创建有效的代理模型，该算法在多种实际基准测试上优于先前的方法。 

---
# Mapping representations in Reinforcement Learning via Semantic Alignment for Zero-Shot Stitching 

**Title (ZH)**: 通过语义对齐映射强化学习中的表示用于零样本缝合 

**Authors**: Antonio Pio Ricciardi, Valentino Maiorca, Luca Moschella, Riccardo Marin, Emanuele Rodolà  

**Link**: [PDF](https://arxiv.org/pdf/2503.01881)  

**Abstract**: Deep Reinforcement Learning (RL) models often fail to generalize when even small changes occur in the environment's observations or task requirements. Addressing these shifts typically requires costly retraining, limiting the reusability of learned policies. In this paper, we build on recent work in semantic alignment to propose a zero-shot method for mapping between latent spaces across different agents trained on different visual and task variations. Specifically, we learn a transformation that maps embeddings from one agent's encoder to another agent's encoder without further fine-tuning. Our approach relies on a small set of "anchor" observations that are semantically aligned, which we use to estimate an affine or orthogonal transform. Once the transformation is found, an existing controller trained for one domain can interpret embeddings from a different (existing) encoder in a zero-shot fashion, skipping additional trainings. We empirically demonstrate that our framework preserves high performance under visual and task domain shifts. We empirically demonstrate zero-shot stitching performance on the CarRacing environment with changing background and task. By allowing modular re-assembly of existing policies, it paves the way for more robust, compositional RL in dynamically changing environments. 

**Abstract (ZH)**: 深度强化学习模型在环境观察或任务需求发生微小变化时往往难以泛化，通常需要昂贵的重新训练才能应对这些变化，限制了学习策略的可重用性。在本文中，我们基于近期在语义对齐方面的研究工作，提出了一种零shot方法，用于在训练于不同视觉和任务变异上的不同智能体之间映射潜在空间。具体来说，我们学习一个变换，该变换将一个智能体编码器的嵌入映射到另一个智能体编码器，而无需进一步微调。我们的方法依赖于一组“锚点”观察，它们在语义上是齐的，我们使用这些观察来估计仿射或正交变换。一旦找到变换，一个为一个领域训练的现有控制器可以以零shot的方式解释另一个（现有）编码器的嵌入，从而跳过额外的训练。我们实证结果显示，我们的框架在视觉和任务领域变化下能够保持高性能。我们展示了在背景和任务变化的CarRacing环境中实现零shot拼接性能。通过允许模块化重组现有策略，我们的方法为动态变化环境中的更稳健和组合式强化学习铺平了道路。 

---
# District Vitality Index Using Machine Learning Methods for Urban Planners 

**Title (ZH)**: 基于机器学习方法的城市活力指数研究 

**Authors**: Sylvain Marcoux, Jean-Sébastien Dessureault  

**Link**: [PDF](https://arxiv.org/pdf/2503.01878)  

**Abstract**: City leaders face critical decisions regarding budget allocation and investment priorities. How can they identify which city districts require revitalization? To address this challenge, a Current Vitality Index and a Long-Term Vitality Index are proposed. These indexes are based on a carefully curated set of indicators. Missing data is handled using K-Nearest Neighbors imputation, while Random Forest is employed to identify the most reliable and significant features. Additionally, k-means clustering is utilized to generate meaningful data groupings for enhanced monitoring of Long-Term Vitality. Current vitality is visualized through an interactive map, while Long-Term Vitality is tracked over 15 years with predictions made using Multilayer Perceptron or Linear Regression. The results, approved by urban planners, are already promising and helpful, with the potential for further improvement as more data becomes available. This paper proposes leveraging machine learning methods to optimize urban planning and enhance citizens' quality of life. 

**Abstract (ZH)**: 城市领导者面临关于预算分配和投资优先级的关键决策。如何确定哪些城市区域需要 revitalization? 为应对这一挑战，本文提出了当前活力指数和长期活力指数。这些指数基于精心挑选的一系列指标。缺失数据使用K-最近邻插补，而随机森林被用于识别最可靠和显著的特征。此外，K-means聚类用于生成有意义的数据组，以增强对长期活力的监控。当前活力通过互动地图可视化，而长期活力则通过15年的跟踪和使用多层感知器或线性回归做出的预测来跟踪。结果得到了城市规划者的批准，初步结果显示出希望并具有提升潜力，随着更多数据的可用，仍有进一步改进的空间。本文提出利用机器学习方法来优化城市规划并提升市民生活质量。 

---
# Time-MQA: Time Series Multi-Task Question Answering with Context Enhancement 

**Title (ZH)**: 时间序列多任务问答/context增强 

**Authors**: Yaxuan Kong, Yiyuan Yang, Yoontae Hwang, Wenjie Du, Stefan Zohren, Zhangyang Wang, Ming Jin, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01875)  

**Abstract**: Time series data are foundational in finance, healthcare, and energy domains. However, most existing methods and datasets remain focused on a narrow spectrum of tasks, such as forecasting or anomaly detection. To bridge this gap, we introduce Time Series Multi-Task Question Answering (Time-MQA), a unified framework that enables natural language queries across multiple time series tasks - numerical analytical tasks and open-ended question answering with reasoning. Central to Time-MQA is the TSQA dataset, a large-scale dataset containing $\sim$200k question-answer pairs derived from diverse time series spanning environment, traffic, etc. This comprehensive resource covers various time series lengths and promotes robust model development. We further demonstrate how continually pre-training large language models (Mistral 7B, Llama-3 8B, and Qwen-2.5 7B) on the TSQA dataset enhanced time series reasoning capabilities, moving beyond mere numeric tasks and enabling more advanced and intuitive interactions with temporal data. The complete TSQA dataset, models, executable codes, user study questionnaires for evaluation, and results have all been open-sourced. 

**Abstract (ZH)**: 时间序列多任务问答：跨多个时间序列任务的自然语言查询与推理 

---
# CABS: Conflict-Aware and Balanced Sparsification for Enhancing Model Merging 

**Title (ZH)**: 冲突感知和平衡稀疏化方法以增强模型融合 

**Authors**: Zongzhen Yang, Binhang Qi, Hailong Sun, Wenrui Long, Ruobing Zhao, Xiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01874)  

**Abstract**: Model merging based on task vectors, i.e., the parameter differences between fine-tuned models and a shared base model, provides an efficient way to integrate multiple task-specific models into a multitask model without retraining. Recent works have endeavored to address the conflicts between task vectors, one of the significant challenges faced by model merging, through sparsification; however, two issues significantly limit their performance: high parameter overlap and unbalanced weight distribution. To address these issues, we propose a simple, yet effective framework called CABS (Conflict-Aware and Balanced Sparsification), consisting of Conflict-Aware Sparsification (CA) and Balanced Sparsification (BS). CA can reduce parameter overlap by applying masks during sequential pruning, ensuring that each task vector retains distinct, non-overlapping parameters. BS leverages $n$: $m$ pruning to preserve critical weights while maintaining an even distribution across layers. Our comprehensive experiments demonstrate that CABS outperforms state-of-the-art methods across diverse tasks and model sizes. 

**Abstract (ZH)**: 基于任务向量的模型合并，即微调模型与共享基模型参数差异的合并，提供了一种在不重新训练的情况下将多个任务特定模型集成到多任务模型中的高效方式。为了应对模型合并中任务向量之间的冲突这一重大挑战，近期研究尝试通过稀疏化进行解决；然而，高参数重叠和权重分布不平衡显著限制了其性能。为了解决这些问题，我们提出了一种名为CABS（冲突意识和平衡稀疏化）的简单而有效的框架，包括冲突意识稀疏化（CA）和平衡稀疏化（BS）。CA通过在顺序剪枝过程中应用掩码来减少参数重叠，确保每个任务向量保留独特的非重叠参数。BS利用$n:m$剪枝来保留关键权重，并在各层中维持权重分布的均衡。我们的全面实验表明，CABS在多种任务和模型规模上优于现有方法。 

---
# FairGen: Controlling Sensitive Attributes for Fair Generations in Diffusion Models via Adaptive Latent Guidance 

**Title (ZH)**: FairGen：通过自适应潜在引导控制敏感属性在扩散模型中生成公正的内容 

**Authors**: Mintong Kang, Vinayshekhar Bannihatti Kumar, Shamik Roy, Abhishek Kumar, Sopan Khosla, Balakrishnan Murali Narayanaswamy, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2503.01872)  

**Abstract**: Text-to-image diffusion models often exhibit biases toward specific demographic groups, such as generating more males than females when prompted to generate images of engineers, raising ethical concerns and limiting their adoption. In this paper, we tackle the challenge of mitigating generation bias towards any target attribute value (e.g., "male" for "gender") in diffusion models while preserving generation quality. We propose FairGen, an adaptive latent guidance mechanism which controls the generation distribution during inference. In FairGen, a latent guidance module dynamically adjusts the diffusion process to enforce specific attributes, while a memory module tracks the generation statistics and steers latent guidance to align with the targeted fair distribution of the attribute values. Further, given the limitations of existing datasets in comprehensively assessing bias in diffusion models, we introduce a holistic bias evaluation benchmark HBE, covering diverse domains and incorporating complex prompts across various applications. Extensive evaluations on HBE and Stable Bias datasets demonstrate that FairGen outperforms existing bias mitigation approaches, achieving substantial bias reduction (e.g., 68.5% gender bias reduction on Stable Diffusion 2). Ablation studies highlight FairGen's ability to flexibly and precisely control generation distribution at any user-specified granularity, ensuring adaptive and targeted bias mitigation. 

**Abstract (ZH)**: 面向任意目标属性值的文本到图像扩散模型公平生成机制及全面偏倚评估基准HBE 

---
# Vision Language Models in Medicine 

**Title (ZH)**: 医学中的视觉语言模型 

**Authors**: Beria Chingnabe Kalpelbe, Angel Gabriel Adaambiik, Wei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01863)  

**Abstract**: With the advent of Vision-Language Models (VLMs), medical artificial intelligence (AI) has experienced significant technological progress and paradigm shifts. This survey provides an extensive review of recent advancements in Medical Vision-Language Models (Med-VLMs), which integrate visual and textual data to enhance healthcare outcomes. We discuss the foundational technology behind Med-VLMs, illustrating how general models are adapted for complex medical tasks, and examine their applications in healthcare. The transformative impact of Med-VLMs on clinical practice, education, and patient care is highlighted, alongside challenges such as data scarcity, narrow task generalization, interpretability issues, and ethical concerns like fairness, accountability, and privacy. These limitations are exacerbated by uneven dataset distribution, computational demands, and regulatory hurdles. Rigorous evaluation methods and robust regulatory frameworks are essential for safe integration into healthcare workflows. Future directions include leveraging large-scale, diverse datasets, improving cross-modal generalization, and enhancing interpretability. Innovations like federated learning, lightweight architectures, and Electronic Health Record (EHR) integration are explored as pathways to democratize access and improve clinical relevance. This review aims to provide a comprehensive understanding of Med-VLMs' strengths and limitations, fostering their ethical and balanced adoption in healthcare. 

**Abstract (ZH)**: 随视觉语言模型（VLMs）的出现，医疗人工智能（AI）经历了显著的技术进步和范式转变。本文综述了近期在医学视觉语言模型（Med-VLMs）方面的进展，这些模型将视觉和文本数据结合起来以提升医疗保健结果。文中讨论了Med-VLMs的基础技术，展示了通用模型如何适应复杂的医疗任务，并探讨了它们在医疗保健中的应用。Med-VLMs在临床实践、教育和患者护理中的变革性影响及其面临的挑战，如数据稀缺性、任务泛化不足、可解释性问题以及公平性、可问责性和隐私等伦理问题得到了强调。这些限制因数据集分布不均、计算需求和监管障碍而加剧。严格的评估方法和健全的监管框架是实现Med-VLMs安全集成到医疗工作流程中的关键。未来方向包括利用大规模多样化的数据集、提高跨模态泛化能力和增强可解释性。探索联邦学习、轻量级架构和电子健康记录（EHR）集成等创新作为使医学生物语言模型民主化和临床相关性提升的途径。本文旨在提供Med-VLMs的强项和局限性的全面理解，促进其在医疗保健中的道德和平衡采用。 

---
# Towards Enterprise-Ready Computer Using Generalist Agent 

**Title (ZH)**: 面向企业应用的通用型智能体 

**Authors**: Sami Marreed, Alon Oved, Avi Yaeli, Segev Shlomov, Ido Levy, Aviad Sela, Asaf Adi, Nir Mashkif  

**Link**: [PDF](https://arxiv.org/pdf/2503.01861)  

**Abstract**: This paper presents our ongoing work toward developing an enterprise-ready Computer Using Generalist Agent (CUGA) system. Our research highlights the evolutionary nature of building agentic systems suitable for enterprise environments. By integrating state-of-the-art agentic AI techniques with a systematic approach to iterative evaluation, analysis, and refinement, we have achieved rapid and cost-effective performance gains, notably reaching a new state-of-the-art performance on the WebArena benchmark. We detail our development roadmap, the methodology and tools that facilitated rapid learning from failures and continuous system refinement, and discuss key lessons learned and future challenges for enterprise adoption. 

**Abstract (ZH)**: 本文介绍了我们正在开发的企业级通用智能体（CUGA）系统的持续工作。我们的研究突显了构建适合企业环境的代理系统的发展性质。通过将最先进的代理AI技术与系统化的迭代评估、分析和改进方法相结合，我们实现了快速且成本效益高的性能提升，特别是在WebArena基准测试中达到了新的前沿性能。我们详细介绍了我们的开发路线图、促进快速从失败中学习并持续系统改进的方法和工具，以及讨论了企业采用的关键经验教训和未来挑战。 

---
# Optimizing Retrieval-Augmented Generation of Medical Content for Spaced Repetition Learning 

**Title (ZH)**: 优化基于检索增强生成的医学内容间隔重复学习检索算法 

**Authors**: Jeremi I. Kaczmarek, Jakub Pokrywka, Krzysztof Biedalak, Grzegorz Kurzyp, Łukasz Grzybowski  

**Link**: [PDF](https://arxiv.org/pdf/2503.01859)  

**Abstract**: Advances in Large Language Models revolutionized medical education by enabling scalable and efficient learning solutions. This paper presents a pipeline employing Retrieval-Augmented Generation (RAG) system to prepare comments generation for Poland's State Specialization Examination (PES) based on verified resources. The system integrates these generated comments and source documents with a spaced repetition learning algorithm to enhance knowledge retention while minimizing cognitive overload. By employing a refined retrieval system, query rephraser, and an advanced reranker, our modified RAG solution promotes accuracy more than efficiency. Rigorous evaluation by medical annotators demonstrates improvements in key metrics such as document relevance, credibility, and logical coherence of generated content, proven by a series of experiments presented in the paper. This study highlights the potential of RAG systems to provide scalable, high-quality, and individualized educational resources, addressing non-English speaking users. 

**Abstract (ZH)**: 大型语言模型的发展 revolutionized 医学教育，通过提供可扩展和高效的學習解決方案。本文提出了一种基于验证资源的管线，采用检索增强生成（RAG）系统为波兰国家专门化考试（PES）准备评论生成。该系统将生成的评论和源文档与间隔重复学习算法结合，以提高知识保留并减轻认知负担。通过采用改进的检索系统、查询重写器和高级重排序器，我们修改后的RAG解决方案更注重准确性而非效率。医学注释者的严格评估表明，在文档相关性、可信度和生成内容的逻辑连贯性等关键指标上取得了改进，这由论文中介绍的一系列实验所证明。本研究强调了RAG系统在提供可扩展、高质量和个性化教育资源方面的潜力，以满足非英语母语用户的需求。 

---
# A Review of Artificial Intelligence Impacting Statistical Process Monitoring and Future Directions 

**Title (ZH)**: 人工智能影响统计过程监控的综述及未来方向 

**Authors**: Shing I Chang, Parviz Ghafariasl  

**Link**: [PDF](https://arxiv.org/pdf/2503.01858)  

**Abstract**: It has been 100 years since statistical process control (SPC) or statistical process monitoring (SPM) was first introduced for production processes and later applied to service, healthcare, and other industries. The techniques applied to SPM applications are mostly statistically oriented. Recent advances in Artificial Intelligence (AI) have reinvigorated the imagination of adopting AI for SPM applications. This manuscript begins with a concise review of the historical development of the statistically based SPM methods. Next, this manuscript explores AI and Machine Learning (ML) algorithms and methods applied in various SPM applications, addressing quality characteristics of univariate, multivariate, profile, and image. These AI methods can be classified into the following categories: classification, pattern recognition, time series applications, and generative AI. Specifically, different kinds of neural networks, such as artificial neural networks (ANN), convolutional neural networks (CNN), recurrent neural networks (RNN), and generative adversarial networks (GAN), are among the most implemented AI methods impacting SPM. Finally, this manuscript outlines a couple of future directions that harness the potential of the Large Multimodal Model (LMM) for advancing SPM research and applications in complex systems. The ultimate objective is to transform statistical process monitoring (SPM) into smart process control (SMPC), where corrective actions are autonomously implemented to either prevent quality issues or restore process performance. 

**Abstract (ZH)**: 统计过程控制（SPC）或统计过程监控（SPM）自首次应用于生产过程至今已有一百周年，并后被应用于服务、医疗及其他行业。用于SPM应用的技术大多基于统计方法。近年来，人工智能（AI）的发展重新激发了采用AI进行SPM应用的想象。本文首先简要回顾了基于统计方法的SPM方法的发展历史。接着，本文探讨了应用于各种SPM应用的AI和机器学习（ML）算法与方法，涉及单一变量、多变量、特性和图像的质量特性。这些AI方法可以分为分类、模式识别、时间序列应用和生成性AI等类别。特别是，各种类型的神经网络，如人工神经网络（ANN）、卷积神经网络（CNN）、循环神经网络（RNN）和生成对抗网络（GAN），是影响SPM应用最多的AI方法之一。最后，本文概述了利用大型多模态模型（LMM）潜力以推动SPM研究和复杂系统应用的若干未来方向。最终目标是将统计过程监控（SPM）转变为智能过程控制（SMPC），其中自动实施纠正措施以预防质量问题或恢复过程性能。 

---
