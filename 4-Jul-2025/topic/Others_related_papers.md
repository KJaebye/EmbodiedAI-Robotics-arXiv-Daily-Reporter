# Optimizing Start Locations in Ergodic Search for Disaster Response 

**Title (ZH)**: 优化灾害响应中遍历搜索的起始位置 

**Authors**: Ananya Rao, Alyssa Hargis, David Wettergreen, Howie Choset  

**Link**: [PDF](https://arxiv.org/pdf/2507.02708)  

**Abstract**: In disaster response scenarios, deploying robotic teams effectively is crucial for improving situational awareness and enhancing search and rescue operations. The use of robots in search and rescue has been studied but the question of where to start robot deployments has not been addressed. This work addresses the problem of optimally selecting starting locations for robots with heterogeneous capabilities by formulating a joint optimization problem. To determine start locations, this work adds a constraint to the ergodic optimization framework whose minimum assigns robots to start locations. This becomes a little more challenging when the robots are heterogeneous (equipped with different sensing and motion modalities) because not all robots start at the same location, and a more complex adaptation of the aforementioned constraint is applied. Our method assumes access to potential starting locations, which can be obtained from expert knowledge or aerial imagery. We experimentally evaluate the efficacy of our joint optimization approach by comparing it to baseline methods that use fixed starting locations for all robots. Our experimental results show significant gains in coverage performance, with average improvements of 35.98% on synthetic data and 31.91% on real-world data for homogeneous and heterogeneous teams, in terms of the ergodic metric. 

**Abstract (ZH)**: 基于灾害响应场景中异构机器人团队起始部署位置的联合优化研究 

---
# MISC: Minimal Intervention Shared Control with Guaranteed Safety under Non-Convex Constraints 

**Title (ZH)**: MISC: 最小干预共享控制并在非凸约束下保证安全 

**Authors**: Shivam Chaubey, Francesco Verdoja, Shankar Deka, Ville Kyrki  

**Link**: [PDF](https://arxiv.org/pdf/2507.02438)  

**Abstract**: Shared control combines human intention with autonomous decision-making, from low-level safety overrides to high-level task guidance, enabling systems that adapt to users while ensuring safety and performance. This enhances task effectiveness and user experience across domains such as assistive robotics, teleoperation, and autonomous driving. However, existing shared control methods, based on e.g. Model Predictive Control, Control Barrier Functions, or learning-based control, struggle with feasibility, scalability, or safety guarantees, particularly since the user input is unpredictable.
To address these challenges, we propose an assistive controller framework based on Constrained Optimal Control Problem that incorporates an offline-computed Control Invariant Set, enabling online computation of control actions that ensure feasibility, strict constraint satisfaction, and minimal override of user intent. Moreover, the framework can accommodate structured class of non-convex constraints, which are common in real-world scenarios. We validate the approach through a large-scale user study with 66 participants--one of the most extensive in shared control research--using a computer game environment to assess task load, trust, and perceived control, in addition to performance. The results show consistent improvements across all these aspects without compromising safety and user intent. 

**Abstract (ZH)**: 共享控制结合了人类意图与自主决策，在从低级安全 Override 到高级任务指导的各个层面，使系统能够在确保安全和性能的同时适应用户，从而在辅助机器人、远程操作和自动驾驶等领域提升任务效果和用户体验。为了解决现有共享控制方法（如基于模型预测控制、控制障碍函数或基于学习的控制）在可行性、可扩展性或安全性保证方面遇到的挑战，特别是由于用户输入的不可预测性，我们提出了一种基于约束最优控制问题的辅助控制器框架，该框架结合了离线计算的控制不变集，能够在在线计算控制动作时确保可行性、严格约束满足，并最小化对用户意图的干预。此外，该框架能够处理在现实场景中常见的结构化非凸约束。通过一项包含66名参与者的大型用户研究（目前共享控制研究中规模最大之一），并在计算机游戏环境中评估任务负荷、信任度和感知控制，以及性能，结果表明，在不牺牲安全性和用户意图的情况下，该方法在所有这些方面都表现出一致的改进。 

---
# DigiT4TAF -- Bridging Physical and Digital Worlds for Future Transportation Systems 

**Title (ZH)**: DigiT4TAF -- 跨越物理与数字世界的未来交通运输系统桥梁 

**Authors**: Maximilian Zipfl, Pascal Zwick, Patrick Schulz, Marc Rene Zofka, Albert Schotschneider, Helen Gremmelmaier, Nikolai Polley, Ferdinand Mütsch, Kevin Simon, Fabian Gottselig, Michael Frey, Sergio Marschall, Akim Stark, Maximilian Müller, Marek Wehmer, Mihai Kocsis, Dominic Waldenmayer, Florian Schnepf, Erik Heinrich, Sabrina Pletz, Matthias Kölle, Karin Langbein-Euchner, Alexander Viehl, Raoul Zöllner, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2507.02400)  

**Abstract**: In the future, mobility will be strongly shaped by the increasing use of digitalization. Not only will individual road users be highly interconnected, but also the road and associated infrastructure. At that point, a Digital Twin becomes particularly appealing because, unlike a basic simulation, it offers a continuous, bilateral connection linking the real and virtual environments. This paper describes the digital reconstruction used to develop the Digital Twin of the Test Area Autonomous Driving-Baden-Württemberg (TAF-BW), Germany. The TAF-BW offers a variety of different road sections, from high-traffic urban intersections and tunnels to multilane motorways. The test area is equipped with a comprehensive Vehicle-to-Everything (V2X) communication infrastructure and multiple intelligent intersections equipped with camera sensors to facilitate real-time traffic flow monitoring. The generation of authentic data as input for the Digital Twin was achieved by extracting object lists at the intersections. This process was facilitated by the combined utilization of camera images from the intelligent infrastructure and LiDAR sensors mounted on a test vehicle. Using a unified interface, recordings from real-world detections of traffic participants can be resimulated. Additionally, the simulation framework's design and the reconstruction process is discussed. The resulting framework is made publicly available for download and utilization at: this https URL The demonstration uses two case studies to illustrate the application of the digital twin and its interfaces: the analysis of traffic signal systems to optimize traffic flow and the simulation of security-related scenarios in the communications sector. 

**Abstract (ZH)**: 未来，移动性将强烈受到数字化程度增加的影响。不仅个体道路使用者将高度互联，道路及其相关基础设施也将实现互联。在这个阶段，数字孪生变得尤为吸引人，因为它不仅可以提供基本模拟，还能实现现实环境与虚拟环境之间的持续双向连接。本文描述了用于开发德国巴登-符腾堡自动驾驶测试区域（TAF-BW）数字孪生的数字重构方法。TAF-BW提供了从高流量城市交叉口和隧道到多车道高速公路等各种道路路段。测试区域配备了全面的车对外界通信（V2X）基础设施和多个配备摄像头传感器的智能交叉口，以实现实时交通流量监控。通过提取智能基础设施的摄像头图像列表和测试车辆上的LiDAR传感器数据，实现了真实数据的生成，用作数字孪生的输入。此外，还讨论了模拟框架的设计及其重构过程。该框架已公开供下载和使用：[此链接](this https URL)。通过两个案例研究展示了数字孪生及其界面的应用：交通信号系统的分析以优化交通流量，以及在通讯领域中的安全相关场景的模拟。 

---
# Path Planning using a One-shot-sampling Skeleton Map 

**Title (ZH)**: 使用单次采样骨架图进行路径规划 

**Authors**: Gabriel O. Flores-Aquino, Octavio Gutierrez-Frias, Juan Irving Vasquez  

**Link**: [PDF](https://arxiv.org/pdf/2507.02328)  

**Abstract**: Path planning algorithms aim to compute a collision-free path, and many works focus on finding the optimal distance path. However, for some applications, a more suitable approach is to balance response time, safety of the paths, and path length. In this context, a skeleton map is a useful tool in graph-based schemes, as it provides an intrinsic representation of free configuration space. However, skeletonization algorithms are very resource-intensive, being primarily oriented towards image processing tasks. We propose an efficient path-planning methodology that finds safe paths within an acceptable processing time. This methodology leverages a Deep Denoising Auto-Encoder (DDAE) based on U-Net architecture to compute a skeletonized version of the navigation map, which we refer to as SkelUnet. The SkelUnet network facilitates exploration of the entire workspace through one-shot sampling (OSS), as opposed to the iterative process used by exact algorithms or the probabilistic sampling process. SkelUnet is trained and tested on a dataset consisting of 12,500 bi-dimensional dungeon maps. The motion planning methodology is evaluated in a simulation environment for an Unmanned Aerial Vehicle (UAV) using 250 previously unseen maps, and assessed with various navigation metrics to quantify the navigability of the computed paths. The results demonstrate that using SkelUnet to construct a roadmap offers significant advantages, such as connecting all regions of free workspace, providing safer paths, and reducing processing times. These characteristics make this method particularly suitable for mobile service robots in structured environments. 

**Abstract (ZH)**: 基于路径规划的高效方法：利用SkelUnet构建骨架地图 

---
# Effective Explanations for Belief-Desire-Intention Robots: When and What to Explain 

**Title (ZH)**: 信念-欲望-意图机器人有效的解释：何时以及解释什么 

**Authors**: Cong Wang, Roberto Calandra, Verena Klös  

**Link**: [PDF](https://arxiv.org/pdf/2507.02016)  

**Abstract**: When robots perform complex and context-dependent tasks in our daily lives, deviations from expectations can confuse users. Explanations of the robot's reasoning process can help users to understand the robot intentions. However, when to provide explanations and what they contain are important to avoid user annoyance. We have investigated user preferences for explanation demand and content for a robot that helps with daily cleaning tasks in a kitchen. Our results show that users want explanations in surprising situations and prefer concise explanations that clearly state the intention behind the confusing action and the contextual factors that were relevant to this decision. Based on these findings, we propose two algorithms to identify surprising actions and to construct effective explanations for Belief-Desire-Intention (BDI) robots. Our algorithms can be easily integrated in the BDI reasoning process and pave the way for better human-robot interaction with context- and user-specific explanations. 

**Abstract (ZH)**: 当机器人在日常生活中执行复杂且依赖于上下文的任务时，偏离预期可能会使用户感到困惑。解释机器人的推理过程有助于用户理解机器人的意图。然而，何时提供解释以及解释的内容对于避免用户烦恼至关重要。我们研究了一种在厨房中协助日常清洁任务的机器人的解释需求和内容偏好。研究结果表明，用户希望在意外情况下获得解释，并偏好简洁明了的解释，这些解释清晰地陈述了令人困惑的行为背后的意图以及与此决策相关的上下文因素。基于这些发现，我们提出了两种算法来识别令人惊讶的行为并为Belief-Desire-Intention（BDI）机器人构建有效的解释。我们的算法可以很容易地集成到BDI推理过程中，并为具有上下文和用户特定解释的更好人机交互铺平道路。 

---
# Red grape detection with accelerated artificial neural networks in the FPGA's programmable logic 

**Title (ZH)**: 基于FPGA可编程逻辑的加速人工神经网络红葡萄检测 

**Authors**: Sandro Costa Magalhães, Marco Almeida, Filipe Neves dos Santos, António Paulo Moreira, Jorge Dias  

**Link**: [PDF](https://arxiv.org/pdf/2507.02443)  

**Abstract**: Robots usually slow down for canning to detect objects while moving. Additionally, the robot's camera is configured with a low framerate to track the velocity of the detection algorithms. This would be constrained while executing tasks and exploring, making robots increase the task execution time. AMD has developed the Vitis-AI framework to deploy detection algorithms into FPGAs. However, this tool does not fully use the FPGAs' PL. In this work, we use the FINN architecture to deploy three ANNs, MobileNet v1 with 4-bit quantisation, CNV with 2-bit quantisation, and CNV with 1-bit quantisation (BNN), inside an FPGA's PL. The models were trained on the RG2C dataset. This is a self-acquired dataset released in open access. MobileNet v1 performed better, reaching a success rate of 98 % and an inference speed of 6611 FPS. In this work, we proved that we can use FPGAs to speed up ANNs and make them suitable for attention mechanisms. 

**Abstract (ZH)**: 机器人通常在搬运过程中减慢速度以检测物体，同时，机器人的相机配置为低帧率以跟踪检测算法的velocity。这在执行任务和探索时会受限，使机器人增加任务执行时间。AMD开发了Vitis-AI框架以将检测算法部署到FPGAs中，但该工具并未充分利用FPGAs的PL。在本文中，我们使用FINN架构将三个ANN，MobileNet v1（4位量化）、CNV（2位量化）和CNV（1位量化，BNN）部署到FPGA的PL中。这些模型在RG2C数据集上进行了训练。该数据集是自获取的数据集，现已开放获取。MobileNet v1表现最佳，成功率达到98%，推理速度为6611 FPS。在本文中，我们证明了可以使用FPGAs加速ANN并使其适合注意力机制。 

---
# Establishing Best Practices for Building Rigorous Agentic Benchmarks 

**Title (ZH)**: 建立严谨代理基准的最佳实践 

**Authors**: Yuxuan Zhu, Tengjun Jin, Yada Pruksachatkun, Andy Zhang, Shu Liu, Sasha Cui, Sayash Kapoor, Shayne Longpre, Kevin Meng, Rebecca Weiss, Fazl Barez, Rahul Gupta, Jwala Dhamala, Jacob Merizian, Mario Giulianelli, Harry Coppock, Cozmin Ududec, Jasjeet Sekhon, Jacob Steinhardt, Antony Kellerman, Sarah Schwettmann, Matei Zaharia, Ion Stoica, Percy Liang, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02825)  

**Abstract**: Benchmarks are essential for quantitatively tracking progress in AI. As AI agents become increasingly capable, researchers and practitioners have introduced agentic benchmarks to evaluate agents on complex, real-world tasks. These benchmarks typically measure agent capabilities by evaluating task outcomes via specific reward designs. However, we show that many agentic benchmarks have issues task setup or reward design. For example, SWE-bench Verified uses insufficient test cases, while TAU-bench counts empty responses as successful. Such issues can lead to under- or overestimation agents' performance by up to 100% in relative terms. To make agentic evaluation rigorous, we introduce the Agentic Benchmark Checklist (ABC), a set of guidelines that we synthesized from our benchmark-building experience, a survey of best practices, and previously reported issues. When applied to CVE-Bench, a benchmark with a particularly complex evaluation design, ABC reduces the performance overestimation by 33%. 

**Abstract (ZH)**: 基准对于定量跟踪AI进展至关重要。随着AI代理能力的不断提升，研究人员和实践者引入了代理基准来评估代理在复杂的真实世界任务上的表现。这些基准通常通过特定的奖励设计来评估代理能力。然而，我们发现许多代理基准存在任务设置或奖励设计的问题。例如，SWE-bench Verified使用了不足的测试案例，而TAU-bench将空响应视为成功。这些问题可能导致代理性能的低估或高估，幅度最高可达100%。为了使代理评估更加严谨，我们引入了代理基准检查清单（ABC），这是从我们的基准构建经验、最佳实践调查以及之前报告的问题中综合得出的一套指南。应用于特别是复杂评估设计的CVE-Bench时，ABC将性能高估减少了33%。 

---
# Bourbaki: Self-Generated and Goal-Conditioned MDPs for Theorem Proving 

**Title (ZH)**: Bourbaki: 自生自控的定理证明MDP模型 

**Authors**: Matthieu Zimmer, Xiaotong Ji, Rasul Tutunov, Anthony Bordg, Jun Wang, Haitham Bou Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2507.02726)  

**Abstract**: Reasoning remains a challenging task for large language models (LLMs), especially within the logically constrained environment of automated theorem proving (ATP), due to sparse rewards and the vast scale of proofs. These challenges are amplified in benchmarks like PutnamBench, which contains university-level problems requiring complex, multi-step reasoning. To address this, we introduce self-generated goal-conditioned MDPs (sG-MDPs), a new framework in which agents generate and pursue their subgoals based on the evolving proof state. Given this more structured generation of goals, the resulting problem becomes more amenable to search. We then apply Monte Carlo Tree Search (MCTS)-like algorithms to solve the sG-MDP, instantiating our approach in Bourbaki (7B), a modular system that can ensemble multiple 7B LLMs for subgoal generation and tactic synthesis. On PutnamBench, Bourbaki (7B) solves 26 problems, achieving new state-of-the-art results with models at this scale. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动定理证明（ATP）的逻辑受限环境中进行推理仍然是一项具有挑战性的任务，主要原因是稀疏的奖励和证明规模庞大。在包含需进行复杂多步推理的大学水平问题的基准测试PutnamBench中，这些挑战被进一步放大。为应对这一挑战，我们引入了一种新的自生成目标条件MDP（sG-MDP）框架，在该框架中，代理可以根据证明状态的演变生成并追求其次级目标。借助这种更结构化的目标生成，问题变得更易于搜索。随后，我们使用类似于蒙特卡洛树搜索（MCTS）的算法来求解sG-MDP，并将该方法实例化为Bourbaki（7B），这是一个模块化的系统，可以集成多个7B LLM进行次级目标生成和策略合成。在PutnamBench上，Bourbaki（7B）解决了26个问题，实现了这一规模模型的新最佳结果。 

---
# Time-critical and confidence-based abstraction dropping methods 

**Title (ZH)**: 基于时间敏感性和置信度的抽象丢弃方法 

**Authors**: Robin Schmöcker, Lennart Kampmann, Alexander Dockhorn  

**Link**: [PDF](https://arxiv.org/pdf/2507.02703)  

**Abstract**: One paradigm of Monte Carlo Tree Search (MCTS) improvements is to build and use state and/or action abstractions during the tree search. Non-exact abstractions, however, introduce an approximation error making convergence to the optimal action in the abstract space impossible. Hence, as proposed as a component of Elastic Monte Carlo Tree Search by Xu et al., abstraction algorithms should eventually drop the abstraction. In this paper, we propose two novel abstraction dropping schemes, namely OGA-IAAD and OGA-CAD which can yield clear performance improvements whilst being safe in the sense that the dropping never causes any notable performance degradations contrary to Xu's dropping method. OGA-IAAD is designed for time critical settings while OGA-CAD is designed to improve the MCTS performance with the same number of iterations. 

**Abstract (ZH)**: 一种 Monte Carlo Tree Search (MCTS) 改进的范式是在树搜索过程中构建和使用状态和/or 动作抽象。然而，非精确抽象引入了近似误差，使得在抽象空间中达到最优动作变得不可能。因此，正如 Xu 等人提出的 Elastic Monte Carlo Tree Search 的一部分，抽象算法最终应丢弃抽象。在本文中，我们提出了两种新颖的抽象丢弃方案，即 OGA-IAAD 和 OGA-CAD，这些方案既能够带来明显的性能提升，又确保了丢弃操作不会引起任何显著的性能退化，与 Xu 的丢弃方法相比更安全。OGA-IAAD 适用于时间关键的环境，而 OGA-CAD 则旨在在相同迭代次数的情况下提高 MCTS 的性能。 

---
# Detection of Disengagement from Voluntary Quizzes: An Explainable Machine Learning Approach in Higher Distance Education 

**Title (ZH)**: 志愿 quizzes 参与度脱落的检测：高等教育远程教育中的可解释机器学习方法 

**Authors**: Behnam Parsaeifard, Christof Imhof, Tansu Pancar, Ioan-Sorin Comsa, Martin Hlosta, Nicole Bergamin, Per Bergamin  

**Link**: [PDF](https://arxiv.org/pdf/2507.02681)  

**Abstract**: Students disengaging from their tasks can have serious long-term consequences, including academic drop-out. This is particularly relevant for students in distance education. One way to measure the level of disengagement in distance education is to observe participation in non-mandatory exercises in different online courses. In this paper, we detect student disengagement in the non-mandatory quizzes of 42 courses in four semesters from a distance-based university. We carefully identified the most informative student log data that could be extracted and processed from Moodle. Then, eight machine learning algorithms were trained and compared to obtain the highest possible prediction accuracy. Using the SHAP method, we developed an explainable machine learning framework that allows practitioners to better understand the decisions of the trained algorithm. The experimental results show a balanced accuracy of 91\%, where about 85\% of disengaged students were correctly detected. On top of the highly predictive performance and explainable framework, we provide a discussion on how to design a timely intervention to minimise disengagement from voluntary tasks in online learning. 

**Abstract (ZH)**: 学生从其任务中脱离可能导致严重的长期后果，包括学业退学。这对基于距离教育的学生尤其相关。测量距离教育中学生脱离程度的一种方法是观察不同在线课程中非强制性练习的参与情况。本文通过对一所基于距离教育的大学四个学期中的42门课程中非强制性测验的学生日志数据进行检测，仔细识别了可以从Moodle提取和处理的最具信息量的学生日志数据。然后训练并比较了八种机器学习算法，以获得最高的预测准确性。使用SHAP方法，我们开发了一个可解释的机器学习框架，使从业者能够更好地理解训练模型的决策。实验结果表明，平衡准确率为91%，其中约85%的脱离学生被正确检测出来。除了高度预测性能和可解释框架，我们还讨论了如何设计及时干预以最大限度地减少在线学习中自愿任务的脱离。 

---
# Decoupled Planning and Execution: A Hierarchical Reasoning Framework for Deep Search 

**Title (ZH)**: 解耦规划与执行：深度搜索的层级推理框架 

**Authors**: Jiajie Jin, Xiaoxi Li, Guanting Dong, Yuyao Zhang, Yutao Zhu, Yang Zhao, Hongjin Qian, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02652)  

**Abstract**: Complex information needs in real-world search scenarios demand deep reasoning and knowledge synthesis across diverse sources, which traditional retrieval-augmented generation (RAG) pipelines struggle to address effectively. Current reasoning-based approaches suffer from a fundamental limitation: they use a single model to handle both high-level planning and detailed execution, leading to inefficient reasoning and limited scalability. In this paper, we introduce HiRA, a hierarchical framework that separates strategic planning from specialized execution. Our approach decomposes complex search tasks into focused subtasks, assigns each subtask to domain-specific agents equipped with external tools and reasoning capabilities, and coordinates the results through a structured integration mechanism. This separation prevents execution details from disrupting high-level reasoning while enabling the system to leverage specialized expertise for different types of information processing. Experiments on four complex, cross-modal deep search benchmarks demonstrate that HiRA significantly outperforms state-of-the-art RAG and agent-based systems. Our results show improvements in both answer quality and system efficiency, highlighting the effectiveness of decoupled planning and execution for multi-step information seeking tasks. Our code is available at this https URL. 

**Abstract (ZH)**: 现实世界搜索场景中的复杂信息需求要求在多样来源中进行深入推理和知识合成，而传统的检索增强生成（RAG）管道在这方面难以有效应对。当前基于推理的方法存在一个根本性限制：它们使用单一模型处理高层次规划和详细执行，导致推理效率低下并限制了可扩展性。本文提出了一种分层框架HiRA，该框架将战略规划与专门执行分离。我们的方法将复杂的搜索任务分解为专注于子任务，将每个子任务分配给配备了外部工具和推理能力的领域特定代理，并通过结构化的集成机制协调结果。这种分离防止了执行细节干扰高层次推理，同时使系统能够利用不同类型的专门知识进行信息处理。在四个复杂的跨模态深度搜索基准上的实验表明，HiRA显著优于现有的RAG和基于代理的系统。我们的结果展示了答案质量和系统效率的改进，突显了脱耦规划和执行在多步信息查找任务中的有效性。代码可在以下网址获得：this https URL。 

---
# Responsibility Gap and Diffusion in Sequential Decision-Making Mechanisms 

**Title (ZH)**: 责任缺口与序列决策机制中的扩散 

**Authors**: Junli Jiang, Pavel Naumov  

**Link**: [PDF](https://arxiv.org/pdf/2507.02582)  

**Abstract**: Responsibility has long been a subject of study in law and philosophy. More recently, it became a focus of AI literature. The article investigates the computational complexity of two important properties of responsibility in collective decision-making: diffusion and gap. It shows that the sets of diffusion-free and gap-free decision-making mechanisms are $\Pi_2$-complete and $\Pi_3$-complete, respectively. At the same time, the intersection of these classes is $\Pi_2$-complete. 

**Abstract (ZH)**: 责任长期以来一直是法律和哲学研究的课题。近年来，它成为人工智能文献中的一个焦点。本文探讨了集体决策中责任的两个重要性质——扩散和间隙的计算复杂性。研究表明，无扩散和无间隙的决策机制集分别是$\Pi_2$-完全和$\Pi_3$-完全的。同时，这些类别的交集也是$\Pi_2$-完全的。 

---
# AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench 

**Title (ZH)**: AI 研究代理用于机器学习：在 MLE-bench 中的搜索、探索与泛化 

**Authors**: Edan Toledo, Karen Hambardzumyan, Martin Josifoski, Rishi Hazra, Nicolas Baldwin, Alexis Audran-Reiss, Michael Kuchnik, Despoina Magka, Minqi Jiang, Alisia Maria Lupidi, Andrei Lupu, Roberta Raileanu, Kelvin Niu, Tatiana Shavrina, Jean-Christophe Gagnon-Audet, Michael Shvartsman, Shagun Sodhani, Alexander H. Miller, Abhishek Charnalia, Derek Dunfield, Carole-Jean Wu, Pontus Stenetorp, Nicola Cancedda, Jakob Nicolaus Foerster, Yoram Bachrach  

**Link**: [PDF](https://arxiv.org/pdf/2507.02554)  

**Abstract**: AI research agents are demonstrating great potential to accelerate scientific progress by automating the design, implementation, and training of machine learning models. We focus on methods for improving agents' performance on MLE-bench, a challenging benchmark where agents compete in Kaggle competitions to solve real-world machine learning problems. We formalize AI research agents as search policies that navigate a space of candidate solutions, iteratively modifying them using operators. By designing and systematically varying different operator sets and search policies (Greedy, MCTS, Evolutionary), we show that their interplay is critical for achieving high performance. Our best pairing of search strategy and operator set achieves a state-of-the-art result on MLE-bench lite, increasing the success rate of achieving a Kaggle medal from 39.6% to 47.7%. Our investigation underscores the importance of jointly considering the search strategy, operator design, and evaluation methodology in advancing automated machine learning. 

**Abstract (ZH)**: AI研究代理通过自动化机器学习模型的设计、实现和训练展现了加速科学进步的巨大潜力。我们聚焦于提高代理在MLE-bench上的性能，这是个具有挑战性的基准，代理们在Kaggle竞赛中竞相对解决实际机器学习问题。我们将AI研究代理形式化为在候选解空间中导航的搜索策略，并通过迭代使用操作符对其进行修改。通过设计并系统地变化不同的操作符集合和搜索策略（贪婪、Monte Carlo树搜索、演化算法），我们表明它们之间的交互对于实现高性能至关重要。我们最佳的搜索策略与操作符集合组合在MLE-bench lite上达到了最先进的结果，将获得Kaggle奖牌的成功率从39.6%提高到47.7%。我们的研究强调了在推进自动化机器学习过程中同时考虑搜索策略、操作符设计和评估方法的重要性。 

---
# Clarifying Before Reasoning: A Coq Prover with Structural Context 

**Title (ZH)**: 明确结构上下文之前的推理：一个 Coq 自动化证明器 

**Authors**: Yanzhen Lu, Hanbin Yang, Xiaodie Wang, Ge Zhang, Biao Li, Chenxu Fu, Chao Li, Yang Yuan, Andrew Chi-Chih Yao  

**Link**: [PDF](https://arxiv.org/pdf/2507.02541)  

**Abstract**: In this work, we investigate whether improving task clarity can enhance reasoning ability of large language models, focusing on theorem proving in Coq. We introduce a concept-level metric to evaluate task clarity and show that adding structured semantic context to the standard input used by modern LLMs, leads to a 1.85$\times$ improvement in clarity score (44.5\%~$\rightarrow$~82.3\%). Using the general-purpose model \texttt{DeepSeek-V3}, our approach leads to a 2.1$\times$ improvement in proof success (21.8\%~$\rightarrow$~45.8\%) and outperforms the previous state-of-the-art \texttt{Graph2Tac} (33.2\%). We evaluate this on 1,386 theorems randomly sampled from 15 standard Coq packages, following the same evaluation protocol as \texttt{Graph2Tac}. Furthermore, fine-tuning smaller models on our structured data can achieve even higher performance (48.6\%). Our method uses selective concept unfolding to enrich task descriptions, and employs a Planner--Executor architecture. These findings highlight the value of structured task representations in bridging the gap between understanding and reasoning. 

**Abstract (ZH)**: 本研究探讨提升任务清晰度是否能增强大规模语言模型的推理能力，以Coq中的定理证明为例。引入了一种基于概念的评估指标来衡量任务清晰度，并展示了通过为现代大型语言模型的标准输入添加结构化的语义上下文，清晰度评分提高了1.85倍（从44.5%增加到82.3%）。使用通用模型DeepSeek-V3，我们的方法在证明成功上提高了2.1倍（从21.8%增加到45.8%），超越了先前的最先进方法Graph2Tac（33.2%）。我们在15个标准Coq包中随机抽取的1,386个定理上进行了评估，遵循与Graph2Tac相同的评估协议。此外，对我们的结构化数据进行微调的小模型可以实现更高的性能（48.6%）。我们的方法通过选择性概念展开来丰富任务描述，并采用规划器-执行器架构。这些发现突显了结构化任务表示在理解和推理差距中的价值。 

---
# The Gauss-Markov Adjunction: Categorical Semantics of Residuals in Supervised Learning 

**Title (ZH)**: 高斯-马尔可夫增益：监督学习中余差的范畴语义 

**Authors**: Moto Kamiura  

**Link**: [PDF](https://arxiv.org/pdf/2507.02442)  

**Abstract**: Enhancing the intelligibility and interpretability of machine learning is a crucial task in responding to the demand for Explicability as an AI principle, and in promoting the better social implementation of AI. The aim of our research is to contribute to this improvement by reformulating machine learning models through the lens of category theory, thereby developing a semantic framework for structuring and understanding AI systems. Our categorical modeling in this paper clarifies and formalizes the structural interplay between residuals and parameters in supervised learning. The present paper focuses on the multiple linear regression model, which represents the most basic form of supervised learning. By defining two concrete categories corresponding to parameters and data, along with an adjoint pair of functors between them, we introduce our categorical formulation of supervised learning. We show that the essential structure of this framework is captured by what we call the Gauss-Markov Adjunction. Within this setting, the dual flow of information can be explicitly described as a correspondence between variations in parameters and residuals. The ordinary least squares estimator for the parameters and the minimum residual are related via the preservation of limits by the right adjoint functor. Furthermore, we position this formulation as an instance of extended denotational semantics for supervised learning, and propose applying a semantic perspective developed in theoretical computer science as a formal foundation for Explicability in AI. 

**Abstract (ZH)**: 增强机器学习的可解释性和可理解性是响应明确性作为人工智能原则需求的关键任务，也是促进人工智能更好地社会实施的关键。我们的研究旨在通过范畴论的视角重新构建机器学习模型，从而发展一种语义框架来结构化和理解AI系统。本文中的范畴论建模阐明并形式化了监督学习中残差与参数的结构性相互作用。本文集中于多元线性回归模型，这是监督学习的最基本形式。通过定义与参数和数据对应的两个具体范畴，并引入它们之间的伴随对 Functors，我们提出了监督学习的范畴论表述。我们表明，这种框架的基本结构由我们称为高斯-马尔可夫伴随所捕获。在此设置下，信息的对偶传递可显式描述为参数变化与残差之间的对应关系。参数的普通最小二乘估计量和最小残差通过右伴随 Functors 保持极限。此外，我们将这一表述定位为监督学习扩展语义解释的一个实例，并提出应用来自理论计算机科学发展的语义视角作为人工智能明确性的形式基础。 

---
# Iterated belief revision: from postulates to abilities 

**Title (ZH)**: 迭代信念修订：从公理到能力 

**Authors**: Paolo Liberatore  

**Link**: [PDF](https://arxiv.org/pdf/2507.02319)  

**Abstract**: The belief revision field is opulent in new proposals and indigent in analyses of existing approaches. Much work hinge on postulates, employed as syntactic characterizations: some revision mechanism is equivalent to some properties. Postulates constraint specific revision instances: certain revisions update certain beliefs in a certain way. As an example, if the revision is consistent with the current beliefs, it is incorporated with no other change. A postulate like this tells what revisions must do and neglect what they can do. Can they reach a certain state of beliefs? Can they reach all possible states of beliefs? Can they reach all possible states of beliefs from no previous belief? Can they reach a dogmatic state of beliefs, where everything not believed is impossible? Can they make two conditions equally believed? An application where every possible state of beliefs is sensible requires each state of beliefs to be reachable. An application where conditions may be equally believed requires such a belief state to be reachable. An application where beliefs may become dogmatic requires a way to make them dogmatic. Such doxastic states need to be reached in a way or another. Not in specific way, as dictated by a typical belief revision postulate. This is an ability, not a constraint: the ability of being plastic, equating, dogmatic. Amnesic, correcting, believer, damascan, learnable are other abilities. Each revision mechanism owns some of these abilities and lacks the others: lexicographic, natural, restrained, very radical, full meet, radical, severe, moderate severe, deep severe, plain severe and deep severe revisions, each of these revisions is proved to possess certain abilities. 

**Abstract (ZH)**: 信念修订领域在新提案方面丰富多样，但在现有方法的分析方面却资源匮乏。许多工作依赖于公理，这些公理作为句法特征被使用：某些修订机制等同于某些性质。公理约束特定的修订实例：某些修订以某种方式更新某些信念。例如，如果修订与当前信念一致，则不会发生其他更改而被采纳。这样的公理表明修订必须执行什么操作，而忽略它们可以执行的操作。它们是否能够达到某种信念状态？它们是否能够达到所有可能的信念状态？它们是否能够从没有先前信念的状态达到所有可能的信念状态？它们是否能够达到一种绝对主义的信念状态，即所有未被相信的东西都是不可能的？它们是否能够使两种条件被同等相信？一个能够达到每种可能信念状态的应用要求每种信念状态都是可达到的。一个允许条件被同等相信的应用需要能够达到这样一种信念状态。一个允许信念变得绝对主义的应用需要有一种使它们变得绝对主义的方式。这样的知觉状态需要以某种方式被达到。这并不是特定方式的约束，而是灵活性、等价性、绝对主义、健忘性、校正性、信念者的特性：列克斯诺夫、自然的、受限制的、非常激进的、全交集的、激进的、严厉的、中等严厉的、深严厉的和深层严厉的修订，每种修订都被证明具有某些特性。 

---
# Dilution, Diffusion and Symbiosis in Spatial Prisoner's Dilemma with Reinforcement Learning 

**Title (ZH)**: 空间囚徒困境中强化学习的稀释、扩散与共生 

**Authors**: Gustavo C. Mangold, Heitor C. M. Fernandes, Mendeli H. Vainstein  

**Link**: [PDF](https://arxiv.org/pdf/2507.02211)  

**Abstract**: Recent studies in the spatial prisoner's dilemma games with reinforcement learning have shown that static agents can learn to cooperate through a diverse sort of mechanisms, including noise injection, different types of learning algorithms and neighbours' payoff this http URL this work, using an independent multi-agent Q-learning algorithm, we study the effects of dilution and mobility in the spatial version of the prisoner's dilemma. Within this setting, different possible actions for the algorithm are defined, connecting with previous results on the classical, non-reinforcement learning spatial prisoner's dilemma, showcasing the versatility of the algorithm in modeling different game-theoretical scenarios and the benchmarking potential of this this http URL a result, a range of effects is observed, including evidence that games with fixed update rules can be qualitatively equivalent to those with learned ones, as well as the emergence of a symbiotic mutualistic effect between populations that forms when multiple actions are defined. 

**Abstract (ZH)**: 近期的研究表明，在具有强化学习的空间囚徒困境游戏中，静态代理可以通过多种机制学会合作，包括噪声注入、不同的学习算法以及邻居的收益。在此项工作中，我们利用独立的多代理Q学习算法研究了稀释和移动性对空间囚徒困境的影响。在这一框架下，我们定义了算法的不同可能行动，从而与经典的空间囚徒困境（非强化学习）的先前结果相连接，展示了该算法在 modeling 不同博弈论场景中的多功能性及作为基准测试的潜力。研究结果观察到了一系列效应，包括固定更新规则的游戏可以与学习更新规则的游戏在定性上等价的证据，以及当定义多种行动时，群体之间出现共生互惠效应。 

---
# The Illusion of Fairness: Auditing Fairness Interventions with Audit Studies 

**Title (ZH)**: 公平幻象：审计研究审视公平干预措施 

**Authors**: Disa Sariola, Patrick Button, Aron Culotta, Nicholas Mattei  

**Link**: [PDF](https://arxiv.org/pdf/2507.02152)  

**Abstract**: Artificial intelligence systems, especially those using machine learning, are being deployed in domains from hiring to loan issuance in order to automate these complex decisions. Judging both the effectiveness and fairness of these AI systems, and their human decision making counterpart, is a complex and important topic studied across both computational and social sciences. Within machine learning, a common way to address bias in downstream classifiers is to resample the training data to offset disparities. For example, if hiring rates vary by some protected class, then one may equalize the rate within the training set to alleviate bias in the resulting classifier. While simple and seemingly effective, these methods have typically only been evaluated using data obtained through convenience samples, introducing selection bias and label bias into metrics. Within the social sciences, psychology, public health, and medicine, audit studies, in which fictitious ``testers'' (e.g., resumes, emails, patient actors) are sent to subjects (e.g., job openings, businesses, doctors) in randomized control trials, provide high quality data that support rigorous estimates of discrimination. In this paper, we investigate how data from audit studies can be used to improve our ability to both train and evaluate automated hiring algorithms. We find that such data reveals cases where the common fairness intervention method of equalizing base rates across classes appears to achieve parity using traditional measures, but in fact has roughly 10% disparity when measured appropriately. We additionally introduce interventions based on individual treatment effect estimation methods that further reduce algorithmic discrimination using this data. 

**Abstract (ZH)**: 人工智能系统，尤其是在招聘到贷款发放等领域使用的机器学习系统，正被用于自动化这些复杂的决策过程。评估这些AI系统及其人工决策的效用和公平性是一个跨计算和社会科学的重要研究课题。在机器学习领域，通过重新采样训练数据来缓解下游分类器中的偏差是一种常见方法。例如，如果某保护类别的招聘率存在差异，可以通过在训练集中平衡这些率来减轻结果分类器中的偏差。尽管这种方法简单且看似有效，但通常仅使用便利样本获得的数据进行评估，这引入了选择偏差和标签偏差。在社会科学、心理学、公共卫生和医学领域，通过向随机控制试验中的受试者（例如，招聘信息、企业、医生）发送虚构的“测试者”（如简历、电子邮件、病人演员）来进行审计研究提供了高质量的数据，支持对歧视现象进行严格的估计。在本文中，我们研究了审计研究数据如何被用来提高训练和评估自动化招聘算法的能力。我们发现，这种方法在传统的衡量标准下似乎实现了平等，但实际上在适当测量时仍有大约10%的偏差。我们还介绍了基于个体治疗效应估计的干预措施，利用这些数据进一步减少算法中的歧视性。 

---
# What Neuroscience Can Teach AI About Learning in Continuously Changing Environments 

**Title (ZH)**: 神经科学能向AI学习连续变化环境下的学习机制传授什么 

**Authors**: Daniel Durstewitz, Bruno Averbeck, Georgia Koppe  

**Link**: [PDF](https://arxiv.org/pdf/2507.02103)  

**Abstract**: Modern AI models, such as large language models, are usually trained once on a huge corpus of data, potentially fine-tuned for a specific task, and then deployed with fixed parameters. Their training is costly, slow, and gradual, requiring billions of repetitions. In stark contrast, animals continuously adapt to the ever-changing contingencies in their environments. This is particularly important for social species, where behavioral policies and reward outcomes may frequently change in interaction with peers. The underlying computational processes are often marked by rapid shifts in an animal's behaviour and rather sudden transitions in neuronal population activity. Such computational capacities are of growing importance for AI systems operating in the real world, like those guiding robots or autonomous vehicles, or for agentic AI interacting with humans online. Can AI learn from neuroscience? This Perspective explores this question, integrating the literature on continual and in-context learning in AI with the neuroscience of learning on behavioral tasks with shifting rules, reward probabilities, or outcomes. We will outline an agenda for how specifically insights from neuroscience may inform current developments in AI in this area, and - vice versa - what neuroscience may learn from AI, contributing to the evolving field of NeuroAI. 

**Abstract (ZH)**: 现代AI模型，如大型语言模型，通常在大量数据上一次性训练，可能针对特定任务进行微调，然后部署固定参数。它们的训练过程昂贵、缓慢且渐进，需要数以亿计的重复。与此形成鲜明对比的是，动物能够不断适应其环境中的不断变化的条件。这对于社会物种尤为重要，因为它们的社交行为策略和奖励结果可能会频繁地根据与同伴的互动而改变。这些背后的计算过程往往伴随着动物行为的快速转变以及神经元群体活动的突然过渡。此类计算能力对于在真实世界中运行的AI系统至关重要，如指导机器人或自动驾驶汽车的系统，或与人类在线交互的代理AI。神经科学能为AI学习提供什么启示？本视角探讨了这一问题，将AI领域持续学习和上下文学习的文献与行为任务中规则变化、奖励概率或结果变化的神经科学学习文献相结合。我们将概述一个议程，说明具体而言，神经科学的见解如何影响该领域当前的AI发展，反之亦然，AI又能为神经科学提供什么贡献，从而推动神经科学与AI交叉领域的发展。 

---
# HCVR: A Hybrid Approach with Correlation-aware Voting Rules for Feature Selection 

**Title (ZH)**: HCVR：一种基于相关性感知投票规则的混合特征选择方法 

**Authors**: Nikita Bhedasgaonkar, Rushikesh K. Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02073)  

**Abstract**: In this paper, we propose HCVR (Hybrid approach with Correlation-aware Voting Rules), a lightweight rule-based feature selection method that combines Parameter-to-Parameter (P2P) and Parameter-to-Target (P2T) correlations to eliminate redundant features and retain relevant ones. This method is a hybrid of non-iterative and iterative filtering approaches for dimensionality reduction. It is a greedy method, which works by backward elimination, eliminating possibly multiple features at every step. The rules contribute to voting for features, and a decision to keep or discard is made by majority voting. The rules make use of correlation thresholds between every pair of features, and between features and the target. We provide the results from the application of HCVR to the SPAMBASE dataset. The results showed improvement performance as compared to traditional non-iterative (CFS, mRMR and MI) and iterative (RFE, SFS and Genetic Algorithm) techniques. The effectiveness was assessed based on the performance of different classifiers after applying filtering. 

**Abstract (ZH)**: 基于相关性意识投票规则的混合特征选择方法HCVR 

---
# Answer Matching Outperforms Multiple Choice for Language Model Evaluation 

**Title (ZH)**: 语言模型评估中，答案匹配优于多项选择 

**Authors**: Nikhil Chandak, Shashwat Goel, Ameya Prabhu, Moritz Hardt, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2507.02856)  

**Abstract**: Multiple choice benchmarks have long been the workhorse of language model evaluation because grading multiple choice is objective and easy to automate. However, we show multiple choice questions from popular benchmarks can often be answered without even seeing the question. These shortcuts arise from a fundamental limitation of discriminative evaluation not shared by evaluations of the model's free-form, generative answers. Until recently, there appeared to be no viable, scalable alternative to multiple choice--but, we show that this has changed. We consider generative evaluation via what we call answer matching: Give the candidate model the question without the options, have it generate a free-form response, then use a modern language model with the reference answer to determine if the response matches the reference. To compare the validity of different evaluation strategies, we annotate MMLU-Pro and GPQA-Diamond to obtain human grading data, and measure the agreement of each evaluation approach. We find answer matching using recent models--even small ones--achieves near-perfect agreement, in the range of inter-annotator agreement. In contrast, both multiple choice evaluation and using LLM-as-a-judge without reference answers aligns poorly with human grading. Improving evaluations via answer matching is not merely a conceptual concern: the rankings of several models change significantly when evaluating their free-form responses with answer matching. In light of these findings, we discuss how to move the evaluation ecosystem from multiple choice to answer matching. 

**Abstract (ZH)**: 多项选择基准长期以来一直是语言模型评估的主要工具，因为多项选择评分客观且易于自动化。然而，我们展示出流行基准中的多项选择问题往往可以在看不到问题的情况下被回答。这些捷径源于区分性评估的固有局限性，这种局限性不适用于模型生成答案的评估。直到最近，似乎没有可行且可扩展的替代方案——但我们展示出这种情况已经改变。我们通过我们称之为答案匹配的生成评估方法来考虑：让候选模型不提供选项只回答问题，然后使用现代语言模型和参考答案来判断其响应是否匹配。为了对比不同评估策略的有效性，我们对MMLU-Pro和GPQA-Diamond进行标注以获得人类评分数据，并测量每种评估方法的一致性。我们发现，即使使用较小的模型进行答案匹配也能实现近乎完美的一致性，范围与注释者间的一致性相近。相反，多项选择评估和不使用参考答案的语言模型作为评判者的方法与人类评分的契合度较低。通过答案匹配改进评估不仅仅是一个概念性问题：在使用答案匹配评估其生成答案时，某些模型的排名发生了显著变化。鉴于这些发现，我们讨论如何将评估生态系统从多项选择转移到答案匹配。 

---
# Subtyping in DHOL -- Extended preprint 

**Title (ZH)**: DHOL中的子类型系统——扩展预印本 

**Authors**: Colin Rothgang, Florian Rabe  

**Link**: [PDF](https://arxiv.org/pdf/2507.02855)  

**Abstract**: The recently introduced dependent typed higher-order logic (DHOL) offers an interesting compromise between expressiveness and automation support. It sacrifices the decidability of its type system in order to significantly extend its expressiveness over standard HOL. Yet it retains strong automated theorem proving support via a sound and complete translation to HOL.
We leverage this design to extend DHOL with refinement and quotient types. Both of these are commonly requested by practitioners but rarely provided by automated theorem provers. This is because they inherently require undecidable typing and thus are very difficult to retrofit to decidable type systems. But with DHOL already doing the heavy lifting, adding them is not only possible but elegant and simple.
Concretely, we add refinement and quotient types as special cases of subtyping. This turns the associated canonical inclusion resp. projection maps into identity maps and thus avoids costly changes in representation. We present the syntax, semantics, and translation to HOL for the extended language, including the proofs of soundness and completeness. 

**Abstract (ZH)**: 最近引入的依赖类型高阶逻辑（DHOL）在表达能力和自动化支持之间提供了有趣的权衡。它牺牲了类型系统的可判定性，以显著扩展其相对于标准 HOL 的表达能力。然而，它仍然通过一种 sound 和 complete 的转换保持了强大的自动化定理证明支持。我们利用这一设计，将细化和商类型扩展到 DHOL。这两者通常是由实践者们请求的，但自动化定理证明器很少提供。这是因为它们本质上要求类型判断不可判定，因此很难适应可判定的类型系统。但是，由于 DHOL 已经处理了大部分难题，添加它们不仅可行而且简洁而优雅。具体来说，我们将细化和商类型作为子类型的一种特殊情形加入，将相关的标型插入映射和投影映射转化为恒等映射，从而避免了昂贵的表示更改。我们展示了扩展语言的形式语法、语义以及到 HOL 的转换，并包括完整性和正确性的证明。 

---
# DNN-Based Precoding in RIS-Aided mmWave MIMO Systems With Practical Phase Shift 

**Title (ZH)**: 基于DNN的RIS辅助毫米波MIMO系统中考虑实际相移的预编码技术 

**Authors**: Po-Heng Chou, Ching-Wen Chen, Wan-Jen Huang, Walid Saad, Yu Tsao, Ronald Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02824)  

**Abstract**: In this paper, the precoding design is investigated for maximizing the throughput of millimeter wave (mmWave) multiple-input multiple-output (MIMO) systems with obstructed direct communication paths. In particular, a reconfigurable intelligent surface (RIS) is employed to enhance MIMO transmissions, considering mmWave characteristics related to line-of-sight (LoS) and multipath effects. The traditional exhaustive search (ES) for optimal codewords in the continuous phase shift is computationally intensive and time-consuming. To reduce computational complexity, permuted discrete Fourier transform (DFT) vectors are used for finding codebook design, incorporating amplitude responses for practical or ideal RIS systems. However, even if the discrete phase shift is adopted in the ES, it results in significant computation and is time-consuming. Instead, the trained deep neural network (DNN) is developed to facilitate faster codeword selection. Simulation results show that the DNN maintains sub-optimal spectral efficiency even as the distance between the end-user and the RIS has variations in the testing phase. These results highlight the potential of DNN in advancing RIS-aided systems. 

**Abstract (ZH)**: 毫米波多输入多输出系统中具有阻挡直接通信路径的波束形成设计研究——考虑可重构智能表面的帮助 

---
# Multi-agent Auditory Scene Analysis 

**Title (ZH)**: 多智能体声场景分析 

**Authors**: Caleb Rascon, Luis Gato-Diaz, Eduardo García-Alarcón  

**Link**: [PDF](https://arxiv.org/pdf/2507.02755)  

**Abstract**: Auditory scene analysis (ASA) aims to retrieve information from the acoustic environment, by carrying out three main tasks: sound source location, separation, and classification. These tasks are traditionally executed with a linear data flow, where the sound sources are first located; then, using their location, each source is separated into its own audio stream; from each of which, information is extracted that is relevant to the application scenario (audio event detection, speaker identification, emotion classification, etc.). However, running these tasks linearly increases the overall response time, while making the last tasks (separation and classification) highly sensitive to errors of the first task (location). A considerable amount of effort and computational complexity has been employed in the state-of-the-art to develop techniques that are the least error-prone possible. However, doing so gives rise to an ASA system that is non-viable in many applications that require a small computational footprint and a low response time, such as bioacoustics, hearing-aid design, search and rescue, human-robot interaction, etc. To this effect, in this work, a multi-agent approach is proposed to carry out ASA where the tasks are run in parallel, with feedback loops between them to compensate for local errors, such as: using the quality of the separation output to correct the location error; and using the classification result to reduce the localization's sensitivity towards interferences. The result is a multi-agent auditory scene analysis (MASA) system that is robust against local errors, without a considerable increase in complexity, and with a low response time. The complete proposed MASA system is provided as a framework that uses open-source tools for sound acquisition and reproduction (JACK) and inter-agent communication (ROS2), allowing users to add their own agents. 

**Abstract (ZH)**: 基于多代理的 auditory 场景分析（Multi-agent Based Auditory Scene Analysis, MASA） 

---
# Synthesizable by Design: A Retrosynthesis-Guided Framework for Molecular Analog Generation 

**Title (ZH)**: 设计合乎合成需求：一种 retrosynthesis 引导的分子模拟生成框架 

**Authors**: Shuan Chen, Gunwook Nam, Yousung Jung  

**Link**: [PDF](https://arxiv.org/pdf/2507.02752)  

**Abstract**: The disconnect between AI-generated molecules with desirable properties and their synthetic feasibility remains a critical bottleneck in computational drug and material discovery. While generative AI has accelerated the proposal of candidate molecules, many of these structures prove challenging or impossible to synthesize using established chemical reactions. Here, we introduce SynTwins, a novel retrosynthesis-guided molecular analog design framework that designs synthetically accessible molecular analogs by emulating expert chemist strategies through a three-step process: retrosynthesis, similar building block searching, and virtual synthesis. In comparative evaluations, SynTwins demonstrates superior performance in generating synthetically accessible analogs compared to state-of-the-art machine learning models while maintaining high structural similarity to original target molecules. Furthermore, when integrated with existing molecule optimization frameworks, our hybrid approach produces synthetically feasible molecules with property profiles comparable to unconstrained molecule generators, yet its synthesizability ensured. Our comprehensive benchmarking across diverse molecular datasets demonstrates that SynTwins effectively bridges the gap between computational design and experimental synthesis, providing a practical solution for accelerating the discovery of synthesizable molecules with desired properties for a wide range of applications. 

**Abstract (ZH)**: 生成具有良好性质的分子与其实用合成性之间的差距仍然是计算药物和材料发现中的一个关键瓶颈。尽管生成式AI加速了候选分子的提出，但许多这些结构仍然难以或不可能通过已知的化学反应进行合成。为此，我们提出了一种新的反合成反应导向的分子类比设计框架SynTwins，该框架通过模拟专家化学家的策略，在三个步骤过程中设计合成上可行的分子类比物：反合成反应、相似构建基元搜索和虚拟合成。在对比评估中，SynTwins在生成合成上可行的类比物方面展示了优于当前最先进的机器学习模型的能力，同时保持与原始目标分子的高结构相似性。此外，当与现有的分子优化框架集成时，我们的混合方法能够产生具有与无约束分子生成器相当的性质谱但使其合成化的分子。我们对各种分子数据集的全面基准测试表明，SynTwins有效地弥合了计算设计与实验合成之间的差距，提供了一种加速发现具有所需性质的可合成分子的实用解决方案，适用于多种应用。 

---
# APT: Adaptive Personalized Training for Diffusion Models with Limited Data 

**Title (ZH)**: 自适应个性化训练：基于有限数据的扩散模型训练方法 

**Authors**: JungWoo Chae, Jiyoon Kim, JaeWoong Choi, Kyungyul Kim, Sangheum Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02687)  

**Abstract**: Personalizing diffusion models using limited data presents significant challenges, including overfitting, loss of prior knowledge, and degradation of text alignment. Overfitting leads to shifts in the noise prediction distribution, disrupting the denoising trajectory and causing the model to lose semantic coherence. In this paper, we propose Adaptive Personalized Training (APT), a novel framework that mitigates overfitting by employing adaptive training strategies and regularizing the model's internal representations during fine-tuning. APT consists of three key components: (1) Adaptive Training Adjustment, which introduces an overfitting indicator to detect the degree of overfitting at each time step bin and applies adaptive data augmentation and adaptive loss weighting based on this indicator; (2)Representation Stabilization, which regularizes the mean and variance of intermediate feature maps to prevent excessive shifts in noise prediction; and (3) Attention Alignment for Prior Knowledge Preservation, which aligns the cross-attention maps of the fine-tuned model with those of the pretrained model to maintain prior knowledge and semantic coherence. Through extensive experiments, we demonstrate that APT effectively mitigates overfitting, preserves prior knowledge, and outperforms existing methods in generating high-quality, diverse images with limited reference data. 

**Abstract (ZH)**: 利用有限数据个性化扩散模型面临的挑战及其克服方法：Adaptive Personalized Training (APT) mitigates overfitting, preserves prior knowledge, and generates high-quality, diverse images with limited reference data. 

---
# Solving the Hubbard model with Neural Quantum States 

**Title (ZH)**: 用神经量子态求解霍尔ब模型 

**Authors**: Yuntian Gu, Wenrui Li, Heng Lin, Bo Zhan, Ruichen Li, Yifei Huang, Di He, Yantao Wu, Tao Xiang, Mingpu Qin, Liwei Wang, Dingshun Lv  

**Link**: [PDF](https://arxiv.org/pdf/2507.02644)  

**Abstract**: The rapid development of neural quantum states (NQS) has established it as a promising framework for studying quantum many-body systems. In this work, by leveraging the cutting-edge transformer-based architectures and developing highly efficient optimization algorithms, we achieve the state-of-the-art results for the doped two-dimensional (2D) Hubbard model, arguably the minimum model for high-Tc superconductivity. Interestingly, we find different attention heads in the NQS ansatz can directly encode correlations at different scales, making it capable of capturing long-range correlations and entanglements in strongly correlated systems. With these advances, we establish the half-filled stripe in the ground state of 2D Hubbard model with the next nearest neighboring hoppings, consistent with experimental observations in cuprates. Our work establishes NQS as a powerful tool for solving challenging many-fermions systems. 

**Abstract (ZH)**: 神经量子态（NQS）的快速发展已经使其成为研究量子多体系统有前途的框架。通过利用先进的基于变换器的架构并开发高效的优化算法，我们在此工作中实现了掺杂二维（2D）Hubbard模型的最新成果，这可能是高温超导性中最基本的模型。有趣的是，我们发现NQS ansatz中的不同注意力头可以直接编码不同尺度的关联，使其能够捕获强关联系统中的长程关联和纠缠。借助这些进展，我们建立了具有最邻近和次邻近跃迁的二维Hubbard模型的半填充条纹相，这与 cuprates 中的实验观察结果一致。我们的工作确立了NQS作为解决具有挑战性的多重费米子系统工具的地位。 

---
# De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks 

**Title (ZH)**: De-AntiFake: 重新思考对抗语音 cloning 攻击的保护性扰动 

**Authors**: Wei Fan, Kejiang Chen, Chang Liu, Weiming Zhang, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02606)  

**Abstract**: The rapid advancement of speech generation models has heightened privacy and security concerns related to voice cloning (VC). Recent studies have investigated disrupting unauthorized voice cloning by introducing adversarial perturbations. However, determined attackers can mitigate these protective perturbations and successfully execute VC. In this study, we conduct the first systematic evaluation of these protective perturbations against VC under realistic threat models that include perturbation purification. Our findings reveal that while existing purification methods can neutralize a considerable portion of the protective perturbations, they still lead to distortions in the feature space of VC models, which degrades the performance of VC. From this perspective, we propose a novel two-stage purification method: (1) Purify the perturbed speech; (2) Refine it using phoneme guidance to align it with the clean speech distribution. Experimental results demonstrate that our method outperforms state-of-the-art purification methods in disrupting VC defenses. Our study reveals the limitations of adversarial perturbation-based VC defenses and underscores the urgent need for more robust solutions to mitigate the security and privacy risks posed by VC. The code and audio samples are available at this https URL. 

**Abstract (ZH)**: 快速发展的语音生成模型加剧了语音克隆（VC）相关的隐私和安全担忧。近期研究通过引入对抗性扰动来干扰未经授权的语音克隆。然而，有决心的攻击者能够削弱这些防护扰动并成功执行VC。本研究首次在包括扰动净化在内的现实威胁模型下系统评估这些防护扰动的有效性。研究发现，虽然现有的净化方法能够中和相当一部分防护扰动，但仍会导致VC模型特征空间的失真，从而降低其性能。基于此，我们提出了一种新型的两阶段净化方法：（1）净化被扰动的语音；（2）利用音素指导进一步优化以与干净语音分布对齐。实验结果表明，我们的方法在干扰VC防御方面优于现有的最先进的净化方法。本研究揭示了基于对抗性扰动的VC防护措施的局限性，并强调了更稳健解决方案的迫切需求，以减轻VC带来的安全和隐私风险。更多代码和音频样本请访问此网址。 

---
# AC-Refiner: Efficient Arithmetic Circuit Optimization Using Conditional Diffusion Models 

**Title (ZH)**: AC-精炼器：基于条件扩散模型的高效算术电路优化 

**Authors**: Chenhao Xue, Kezhi Li, Jiaxing Zhang, Yi Ren, Zhengyuan Shi, Chen Zhang, Yibo Lin, Lining Zhang, Qiang Xu, Guangyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.02598)  

**Abstract**: Arithmetic circuits, such as adders and multipliers, are fundamental components of digital systems, directly impacting the performance, power efficiency, and area footprint. However, optimizing these circuits remains challenging due to the vast design space and complex physical constraints. While recent deep learning-based approaches have shown promise, they struggle to consistently explore high-potential design variants, limiting their optimization efficiency. To address this challenge, we propose AC-Refiner, a novel arithmetic circuit optimization framework leveraging conditional diffusion models. Our key insight is to reframe arithmetic circuit synthesis as a conditional image generation task. By carefully conditioning the denoising diffusion process on target quality-of-results (QoRs), AC-Refiner consistently produces high-quality circuit designs. Furthermore, the explored designs are used to fine-tune the diffusion model, which focuses the exploration near the Pareto frontier. Experimental results demonstrate that AC-Refiner generates designs with superior Pareto optimality, outperforming state-of-the-art baselines. The performance gain is further validated by integrating AC-Refiner into practical applications. 

**Abstract (ZH)**: 算术电路（如加法器和乘法器）是数字系统的基本组件，直接影响性能、功耗效率和面积占用。然而，由于设计空间庞大和复杂的物理约束，优化这些电路仍然具有挑战性。虽然基于深度学习的方法显示出潜力，但它们在一致探索高潜力设计变体方面存在困难，限制了其优化效率。为应对这一挑战，我们提出了一种名为AC-Refiner的新型算术电路优化框架，利用条件扩散模型。我们的核心洞察是将算术电路合成重新定义为一个条件图像生成任务。通过仔细调整去噪扩散过程以适应目标质量成果（QoRs），AC-Refiner能够一致地生成高质量的电路设计。此外，探索出的设计被用于微调扩散模型，从而将探索聚焦在帕累托前沿附近。实验结果表明，AC-Refiner生成的设计具有更优的帕累托最优性，优于现有的最先进的基线。性能的提升进一步通过将其整合到实际应用中得到验证。 

---
# Position: A Theory of Deep Learning Must Include Compositional Sparsity 

**Title (ZH)**: 位置：一个深度学习理论必须包括组合稀疏性 

**Authors**: David A. Danhofer, Davide D'Ascenzo, Rafael Dubach, Tomaso Poggio  

**Link**: [PDF](https://arxiv.org/pdf/2507.02550)  

**Abstract**: Overparametrized Deep Neural Networks (DNNs) have demonstrated remarkable success in a wide variety of domains too high-dimensional for classical shallow networks subject to the curse of dimensionality. However, open questions about fundamental principles, that govern the learning dynamics of DNNs, remain. In this position paper we argue that it is the ability of DNNs to exploit the compositionally sparse structure of the target function driving their success. As such, DNNs can leverage the property that most practically relevant functions can be composed from a small set of constituent functions, each of which relies only on a low-dimensional subset of all inputs. We show that this property is shared by all efficiently Turing-computable functions and is therefore highly likely present in all current learning problems. While some promising theoretical insights on questions concerned with approximation and generalization exist in the setting of compositionally sparse functions, several important questions on the learnability and optimization of DNNs remain. Completing the picture of the role of compositional sparsity in deep learning is essential to a comprehensive theory of artificial, and even general, intelligence. 

**Abstract (ZH)**: 过参数化的深度神经网络(DNNs)在高维领域展示了 remarkable 成功，这些领域对经典的受限浅层网络来说由于维数灾提供了巨大挑战。然而，关于治理 DNNs 学习动力学的基本原理，仍有许多待解答的问题。在本文中，我们认为 DNNs 的成功得益于其利用目标函数的组合稀疏结构的能力。因此，DNNs 可以利用这样一个性质：大多数实际相关函数都能够由少量的组成函数通过组合产生，每个组成函数只依赖于所有输入中的低维子集。我们表明，这一性质适用于所有高效图灵可计算函数，因此很可能是当今所有学习问题中的常见特征。虽然在组合稀疏函数的背景下关于逼近和泛化的几个理论上有趣的洞察已经存在，但关于 DNNs 可学习性和优化方面的一些重要问题仍然没有解决。完整地描绘组合稀疏性在深度学习中的作用对于全面的人工智能理论，甚至通用人工智能理论至关重要。 

---
# IndianBailJudgments-1200: A Multi-Attribute Dataset for Legal NLP on Indian Bail Orders 

**Title (ZH)**: 印度保释判决数据集-1200：用于印度保释命令的法律自然语言处理的多属性数据集 

**Authors**: Sneha Deshmukh, Prathmesh Kamble  

**Link**: [PDF](https://arxiv.org/pdf/2507.02506)  

**Abstract**: Legal NLP remains underdeveloped in regions like India due to the scarcity of structured datasets. We introduce IndianBailJudgments-1200, a new benchmark dataset comprising 1200 Indian court judgments on bail decisions, annotated across 20+ attributes including bail outcome, IPC sections, crime type, and legal reasoning. Annotations were generated using a prompt-engineered GPT-4o pipeline and verified for consistency. This resource supports a wide range of legal NLP tasks such as outcome prediction, summarization, and fairness analysis, and is the first publicly available dataset focused specifically on Indian bail jurisprudence. 

**Abstract (ZH)**: 印度保释判决-1200：一种新的标注数据集及其在法律NLP任务中的应用 

---
# S2FGL: Spatial Spectral Federated Graph Learning 

**Title (ZH)**: S2FGL：空间频谱联邦图学习 

**Authors**: Zihan Tan, Suyuan Huang, Guancheng Wan, Wenke Huang, He Li, Mang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.02409)  

**Abstract**: Federated Graph Learning (FGL) combines the privacy-preserving capabilities of federated learning (FL) with the strong graph modeling capability of Graph Neural Networks (GNNs). Current research addresses subgraph-FL only from the structural perspective, neglecting the propagation of graph signals on spatial and spectral domains of the structure. From a spatial perspective, subgraph-FL introduces edge disconnections between clients, leading to disruptions in label signals and a degradation in the class knowledge of the global GNN. From a spectral perspective, spectral heterogeneity causes inconsistencies in signal frequencies across subgraphs, which makes local GNNs overfit the local signal propagation schemes. As a result, spectral client drifts occur, undermining global generalizability. To tackle the challenges, we propose a global knowledge repository to mitigate label signal disruption and a frequency alignment to address spectral client drifts. The combination of spatial and spectral strategies forms our framework S2FGL. Extensive experiments on multiple datasets demonstrate the superiority of S2FGL. The code is available at this https URL. 

**Abstract (ZH)**: 联邦图学习（FGL）结合了联邦学习（FL）的隐私保护能力与图 Neural Networks（GNNs）的强图建模能力。当前研究仅从结构视角探讨了子图-FL，忽略了结构的空间域和频域中图信号的传播。从空间视角来看，子图-FL引入了客户端之间的边断开，导致标签信号中断并降低了全局GNN的类别知识。从频域视角来看，频域异质性导致子图中信号频率的一致性问题，使得局部GNN过度拟合局部信号传播方案。因此，频域客户端漂移发生，损害了全局泛化能力。为应对挑战，我们提出了一种全局知识库以减轻标签信号中断，并提出频率对齐以解决频域客户端漂移问题。结合空间和频域策略形成了我们的框架S2FGL。在多个数据集上的广泛实验展示了S2FGL的优势。代码可在如下链接获取。 

---
# Wildlife Target Re-Identification Using Self-supervised Learning in Non-Urban Settings 

**Title (ZH)**: 非城区环境下基于自监督学习的野生动物目标重识别 

**Authors**: Mufhumudzi Muthivhi, Terence L. van Zyl  

**Link**: [PDF](https://arxiv.org/pdf/2507.02403)  

**Abstract**: Wildlife re-identification aims to match individuals of the same species across different observations. Current state-of-the-art (SOTA) models rely on class labels to train supervised models for individual classification. This dependence on annotated data has driven the curation of numerous large-scale wildlife datasets. This study investigates self-supervised learning Self-Supervised Learning (SSL) for wildlife re-identification. We automatically extract two distinct views of an individual using temporal image pairs from camera trap data without supervision. The image pairs train a self-supervised model from a potentially endless stream of video data. We evaluate the learnt representations against supervised features on open-world scenarios and transfer learning in various wildlife downstream tasks. The analysis of the experimental results shows that self-supervised models are more robust even with limited data. Moreover, self-supervised features outperform supervision across all downstream tasks. The code is available here this https URL. 

**Abstract (ZH)**: 野生动物重识别旨在跨不同观察记录匹配同一物种的个体。当前最先进的（SOTA）模型依赖类别标签来训练监督模型进行个体分类。这种对标注数据的依赖促使构建了大量大规模的野生动物数据集。本研究探讨了在野生动物重识别中使用自我监督学习（Self-Supervised Learning，SSL）的方法。我们使用相机陷阱数据中的时间图像对自动提取个体的两种不同视图，无需监督即可训练自我监督模型，并从潜在无尽的视频流数据中获取训练。我们在开放世界场景和各种野生动物下游任务中评估学习表示与监督特征的表现。实验结果分析表明，即使数据有限，自我监督模型也更具鲁棒性。此外，自我监督特征在所有下游任务中均优于监督特征。代码可在这里获取：https://。 

---
# VeFIA: An Efficient Inference Auditing Framework for Vertical Federated Collaborative Software 

**Title (ZH)**: VeFIA：一种高效的垂直联邦协作软件推断审计框架 

**Authors**: Chung-ju Huang, Ziqi Zhang, Yinggui Wang, Binghui Wang, Tao Wei, Leye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02376)  

**Abstract**: Vertical Federated Learning (VFL) is a distributed AI software deployment mechanism for cross-silo collaboration without accessing participants' data. However, existing VFL work lacks a mechanism to audit the execution correctness of the inference software of the data party. To address this problem, we design a Vertical Federated Inference Auditing (VeFIA) framework. VeFIA helps the task party to audit whether the data party's inference software is executed as expected during large-scale inference without leaking the data privacy of the data party or introducing additional latency to the inference system. The core of VeFIA is that the task party can use the inference results from a framework with Trusted Execution Environments (TEE) and the coordinator to validate the correctness of the data party's computation results. VeFIA guarantees that, as long as the abnormal inference exceeds 5.4%, the task party can detect execution anomalies in the inference software with a probability of 99.99%, without incurring any additional online inference latency. VeFIA's random sampling validation achieves 100% positive predictive value, negative predictive value, and true positive rate in detecting abnormal inference. To the best of our knowledge, this is the first paper to discuss the correctness of inference software execution in VFL. 

**Abstract (ZH)**: 垂直联邦推理审计（VeFIA）框架 

---
# Two-Steps Neural Networks for an Automated Cerebrovascular Landmark Detection 

**Title (ZH)**: 两阶段神经网络用于自动化颅内血管标志点检测 

**Authors**: Rafic Nader, Vincent L'Allinec, Romain Bourcier, Florent Autrusseau  

**Link**: [PDF](https://arxiv.org/pdf/2507.02349)  

**Abstract**: Intracranial aneurysms (ICA) commonly occur in specific segments of the Circle of Willis (CoW), primarily, onto thirteen major arterial bifurcations. An accurate detection of these critical landmarks is necessary for a prompt and efficient diagnosis. We introduce a fully automated landmark detection approach for CoW bifurcations using a two-step neural networks process. Initially, an object detection network identifies regions of interest (ROIs) proximal to the landmark locations. Subsequently, a modified U-Net with deep supervision is exploited to accurately locate the bifurcations. This two-step method reduces various problems, such as the missed detections caused by two landmarks being close to each other and having similar visual characteristics, especially when processing the complete MRA Time-of-Flight (TOF). Additionally, it accounts for the anatomical variability of the CoW, which affects the number of detectable landmarks per scan. We assessed the effectiveness of our approach using two cerebral MRA datasets: our In-House dataset which had varying numbers of landmarks, and a public dataset with standardized landmark configuration. Our experimental results demonstrate that our method achieves the highest level of performance on a bifurcation detection task. 

**Abstract (ZH)**: Willis圆各分支关键解剖标志的自动检测方法：基于两步神经网络的过程 

---
# HelixDesign-Antibody: A Scalable Production-Grade Platform for Antibody Design Built on HelixFold3 

**Title (ZH)**: HelixDesign-抗体：基于HelixFold3构建的可扩展的生产级抗体设计平台 

**Authors**: Jie Gao, Jing Hu, Shanzhuo Zhang, Kunrui Zhu, Sheng Qian, Yueyang Huang, Xiaonan Zhang, Xiaomin Fang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02345)  

**Abstract**: Antibody engineering is essential for developing therapeutics and advancing biomedical research. Traditional discovery methods often rely on time-consuming and resource-intensive experimental screening. To enhance and streamline this process, we introduce a production-grade, high-throughput platform built on HelixFold3, HelixDesign-Antibody, which utilizes the high-accuracy structure prediction model, HelixFold3. The platform facilitates the large-scale generation of antibody candidate sequences and evaluates their interaction with antigens. Integrated high-performance computing (HPC) support enables high-throughput screening, addressing challenges such as fragmented toolchains and high computational demands. Validation on multiple antigens showcases the platform's ability to generate diverse and high-quality antibodies, confirming a scaling law where exploring larger sequence spaces increases the likelihood of identifying optimal binders. This platform provides a seamless, accessible solution for large-scale antibody design and is available via the antibody design page of PaddleHelix platform. 

**Abstract (ZH)**: 抗体工程对于开发治疗药物和推动生物医学研究至关重要。传统发现方法往往依赖于耗时且资源密集型的实验筛选。为增强并简化这一过程，我们介绍了一个基于HelixFold3和HelixDesign-Antibody构建的生产级高通量平台。该平台利用高精度结构预测模型HelixFold3，促进大规模产生抗体候选序列并评估其与抗原的相互作用。集成高性能计算（HPC）支持实现高通量筛选，解决了工具链碎片化和高计算需求等挑战。多种抗原的验证展示了该平台生成多样且高质量抗体的能力，确认了探索更大序列空间能增加发现最佳结合子的几率。该平台提供了无缝且易于访问的大型抗体设计解决方案，并可通过PaddleHelix平台的抗体设计页面获取。 

---
# DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values 

**Title (ZH)**: DeltaSHAP: 用Shapley值解释在线患者监测中的预测演变 

**Authors**: Changhun Kim, Yechan Mun, Sangchul Hahn, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02342)  

**Abstract**: This study proposes DeltaSHAP, a novel explainable artificial intelligence (XAI) algorithm specifically designed for online patient monitoring systems. In clinical environments, discovering the causes driving patient risk evolution is critical for timely intervention, yet existing XAI methods fail to address the unique requirements of clinical time series explanation tasks. To this end, DeltaSHAP addresses three key clinical needs: explaining the changes in the consecutive predictions rather than isolated prediction scores, providing both magnitude and direction of feature attributions, and delivering these insights in real time. By adapting Shapley values to temporal settings, our approach accurately captures feature coalition effects. It further attributes prediction changes using only the actually observed feature combinations, making it efficient and practical for time-sensitive clinical applications. We also introduce new evaluation metrics to evaluate the faithfulness of the attributions for online time series, and demonstrate through experiments on online patient monitoring tasks that DeltaSHAP outperforms state-of-the-art XAI methods in both explanation quality as 62% and computational efficiency as 33% time reduction on the MIMIC-III decompensation benchmark. We release our code at this https URL. 

**Abstract (ZH)**: 本研究提出DeltaSHAP，一种专为在线患者监测系统设计的新颖可解释人工智能(XAI)算法。在临床环境中，及时发现驱动患者风险演变的原因对于及时干预至关重要，但现有的XAI方法未能解决临床时间序列解释任务的特殊需求。为此，DeltaSHAP解决了三个关键的临床需求：解释连续预测的变化而非孤立的预测评分，提供特征归因的幅度和方向，并在实时提供这些见解。通过将Shapley值适应到时间设置，我们的方法准确捕捉了特征合作效应。进一步地，仅使用实际观测到的特征组合来归因预测变化，使其对于时间敏感的临床应用来说既高效又实用。我们还引入新的评估指标来评估在线时间序列中归因的忠实性，并通过在在线患者监测任务上的实验表明，DeltaSHAP在解释质量和计算效率（减少33%的时间）上均优于最先进的XAI方法，在MIMIC-III去补偿基准上表现更佳。源代码发布于此https网址。 

---
# ClustOpt: A Clustering-based Approach for Representing and Visualizing the Search Dynamics of Numerical Metaheuristic Optimization Algorithms 

**Title (ZH)**: ClustOpt：基于聚类的数值元启发式优化算法搜索动态表示与可视化方法 

**Authors**: Gjorgjina Cenikj, Gašper Petelin, Tome Eftimov  

**Link**: [PDF](https://arxiv.org/pdf/2507.02337)  

**Abstract**: Understanding the behavior of numerical metaheuristic optimization algorithms is critical for advancing their development and application. Traditional visualization techniques, such as convergence plots, trajectory mapping, and fitness landscape analysis, often fall short in illustrating the structural dynamics of the search process, especially in high-dimensional or complex solution spaces. To address this, we propose a novel representation and visualization methodology that clusters solution candidates explored by the algorithm and tracks the evolution of cluster memberships across iterations, offering a dynamic and interpretable view of the search process. Additionally, we introduce two metrics - algorithm stability and algorithm similarity- to quantify the consistency of search trajectories across runs of an individual algorithm and the similarity between different algorithms, respectively. We apply this methodology to a set of ten numerical metaheuristic algorithms, revealing insights into their stability and comparative behaviors, thereby providing a deeper understanding of their search dynamics. 

**Abstract (ZH)**: 数值元启发式优化算法的行为理解对于推进其发展和应用至关重要。传统的可视化技术，如收敛图、轨迹映射和fitness景观分析，在展示搜索过程的结构性动态方面往往不足，尤其是在高维或复杂解空间中。为了解决这一问题，我们提出了一种新的表示和可视化方法，该方法聚类算法探索的解候选者，并在迭代过程中跟踪簇成员关系的变化，从而提供搜索过程的动态和可解释视图。此外，我们引入了两个度量标准——算法稳定性和算法相似性，以量化单个算法运行中搜索轨迹的一致性以及不同算法之间的相似性。我们将此方法应用于十个数值元启发式优化算法，揭示了它们的稳定性和相对行为，从而加深了对其搜索动态的理解。 

---
# Tracing the Interactions of Modular CMA-ES Configurations Across Problem Landscapes 

**Title (ZH)**: 模块化CMA-ES配置在问题landscape中的交互追踪 

**Authors**: Ana Nikolikj, Mario Andrés Muñoz, Eva Tuba, Tome Eftimov  

**Link**: [PDF](https://arxiv.org/pdf/2507.02331)  

**Abstract**: This paper leverages the recently introduced concept of algorithm footprints to investigate the interplay between algorithm configurations and problem characteristics. Performance footprints are calculated for six modular variants of the CMA-ES algorithm (modCMA), evaluated on 24 benchmark problems from the BBOB suite, across two-dimensional settings: 5-dimensional and 30-dimensional. These footprints provide insights into why different configurations of the same algorithm exhibit varying performance and identify the problem features influencing these outcomes. Our analysis uncovers shared behavioral patterns across configurations due to common interactions with problem properties, as well as distinct behaviors on the same problem driven by differing problem features. The results demonstrate the effectiveness of algorithm footprints in enhancing interpretability and guiding configuration choices. 

**Abstract (ZH)**: 本文利用 recently introduced 的算法足迹概念，探讨算法配置与问题特征之间的相互作用。计算了适用于 BBOB 套件的 24 个基准问题上的六种模块化 CMA-ES 算法（modCMA）变体的性能足迹，并在二维设置（5 维和 30 维）中进行了评估。这些足迹揭示了为何相同算法的不同配置表现出不同的性能，并识别了影响这些结果的问题特征。分析发现，由于共同的问题属性交互，不同配置之间存在共享的行为模式；同时，由于不同的问题特征，相同的问题上表现出不同的行为。结果表明，算法足迹在提高可解释性和指导配置选择方面非常有效。 

---
# Holistic Continual Learning under Concept Drift with Adaptive Memory Realignment 

**Title (ZH)**: 概念漂移下具有自适应内存重新对齐的整体持续学习 

**Authors**: Alif Ashrafee, Jedrzej Kozal, Michal Wozniak, Bartosz Krawczyk  

**Link**: [PDF](https://arxiv.org/pdf/2507.02310)  

**Abstract**: Traditional continual learning methods prioritize knowledge retention and focus primarily on mitigating catastrophic forgetting, implicitly assuming that the data distribution of previously learned tasks remains static. This overlooks the dynamic nature of real-world data streams, where concept drift permanently alters previously seen data and demands both stability and rapid adaptation.
We introduce a holistic framework for continual learning under concept drift that simulates realistic scenarios by evolving task distributions. As a baseline, we consider Full Relearning (FR), in which the model is retrained from scratch on newly labeled samples from the drifted distribution. While effective, this approach incurs substantial annotation and computational overhead. To address these limitations, we propose Adaptive Memory Realignment (AMR), a lightweight alternative that equips rehearsal-based learners with a drift-aware adaptation mechanism. AMR selectively removes outdated samples of drifted classes from the replay buffer and repopulates it with a small number of up-to-date instances, effectively realigning memory with the new distribution. This targeted resampling matches the performance of FR while reducing the need for labeled data and computation by orders of magnitude.
To enable reproducible evaluation, we introduce four concept-drift variants of standard vision benchmarks: Fashion-MNIST-CD, CIFAR10-CD, CIFAR100-CD, and Tiny-ImageNet-CD, where previously seen classes reappear with shifted representations. Comprehensive experiments on these datasets using several rehearsal-based baselines show that AMR consistently counters concept drift, maintaining high accuracy with minimal overhead. These results position AMR as a scalable solution that reconciles stability and plasticity in non-stationary continual learning environments. 

**Abstract (ZH)**: 面向概念漂移的全面连续学习框架：AMR 

---
# Knowledge Graph-Based Explainable and Generalized Zero-Shot Semantic Communications 

**Title (ZH)**: 基于知识图谱的可解释和泛化的零样本语义通信 

**Authors**: Zhaoyu Zhang, Lingyi Wang, Wei Wu, Fuhui Zhou, Qihui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02291)  

**Abstract**: Data-driven semantic communication is based on superficial statistical patterns, thereby lacking interpretability and generalization, especially for applications with the presence of unseen data. To address these challenges, we propose a novel knowledge graph-enhanced zero-shot semantic communication (KGZS-SC) network. Guided by the structured semantic information from a knowledge graph-based semantic knowledge base (KG-SKB), our scheme provides generalized semantic representations and enables reasoning for unseen cases. Specifically, the KG-SKB aligns the semantic features in a shared category semantics embedding space and enhances the generalization ability of the transmitter through aligned semantic features, thus reducing communication overhead by selectively transmitting compact visual semantics. At the receiver, zero-shot learning (ZSL) is leveraged to enable direct classification for unseen cases without the demand for retraining or additional computational overhead, thereby enhancing the adaptability and efficiency of the classification process in dynamic or resource-constrained environments. The simulation results conducted on the APY datasets show that the proposed KGZS-SC network exhibits robust generalization and significantly outperforms existing SC frameworks in classifying unseen categories across a range of SNR levels. 

**Abstract (ZH)**: 基于知识图谱增强的零样本语义通信（KGZS-SC）网络 

---
# Content filtering methods for music recommendation: A review 

**Title (ZH)**: 音乐推荐中的内容过滤方法：一个综述 

**Authors**: Terence Zeng, Abhishek K. Umrawal  

**Link**: [PDF](https://arxiv.org/pdf/2507.02282)  

**Abstract**: Recommendation systems have become essential in modern music streaming platforms, shaping how users discover and engage with songs. One common approach in recommendation systems is collaborative filtering, which suggests content based on the preferences of users with similar listening patterns to the target user. However, this method is less effective on media where interactions are sparse. Music is one such medium, since the average user of a music streaming service will never listen to the vast majority of tracks. Due to this sparsity, there are several challenges that have to be addressed with other methods. This review examines the current state of research in addressing these challenges, with an emphasis on the role of content filtering in mitigating biases inherent in collaborative filtering approaches. We explore various methods of song classification for content filtering, including lyrical analysis using Large Language Models (LLMs) and audio signal processing techniques. Additionally, we discuss the potential conflicts between these different analysis methods and propose avenues for resolving such discrepancies. 

**Abstract (ZH)**: 推荐系统已成为现代音乐流媒体平台的 essential 组件，影响着用户发现和互动歌曲的方式。推荐系统中的一种常见方法是协同过滤，它根据与目标用户具有相似听歌模式的用户偏好来推荐内容。然而，在用户交互稀少的媒体中，这种方法效果较差。音乐就是这样一个媒体，因为音乐流媒体服务的平均用户几乎不会收听平台上的绝大多数歌曲。由于这种稀疏性，需要采用其他方法来应对多种挑战。本文回顾了当前在这些挑战方面的研究现状，重点关注内容过滤在减轻协同过滤方法固有偏见中的作用。我们探讨了用于内容过滤的各种歌曲分类方法，包括使用大型语言模型（LLMs）进行歌词分析和音频信号处理技术。此外，我们讨论了这些不同分析方法之间的潜在冲突，并提出了解决这些分歧的途径。 

---
# Multi-Label Classification Framework for Hurricane Damage Assessment 

**Title (ZH)**: 飓风损害评估的多标签分类框架 

**Authors**: Zhangding Liu, Neda Mohammadi, John E. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2507.02265)  

**Abstract**: Hurricanes cause widespread destruction, resulting in diverse damage types and severities that require timely and accurate assessment for effective disaster response. While traditional single-label classification methods fall short of capturing the complexity of post-hurricane damage, this study introduces a novel multi-label classification framework for assessing damage using aerial imagery. The proposed approach integrates a feature extraction module based on ResNet and a class-specific attention mechanism to identify multiple damage types within a single image. Using the Rescuenet dataset from Hurricane Michael, the proposed method achieves a mean average precision of 90.23%, outperforming existing baseline methods. This framework enhances post-hurricane damage assessment, enabling more targeted and efficient disaster response and contributing to future strategies for disaster mitigation and resilience. This paper has been accepted at the ASCE International Conference on Computing in Civil Engineering (i3CE 2025), and the camera-ready version will appear in the official conference proceedings. 

**Abstract (ZH)**: 飓风造成广泛破坏，导致多样化的损坏类型和程度，需要及时准确的评估以有效应对灾害。传统单标签分类方法无法捕捉飓风后的复杂损坏情况，本研究提出了一种新的多标签分类框架，利用航空影像评估损坏情况。该提出的方案结合了基于ResNet的功能提取模块和类别特定的关注机制，能够在单张图像中识别多种损坏类型。使用飓风迈克尔的Rescuenet数据集，所提出的办法实现了平均精确度90.23%的均值，优于现有基线方法。该框架提高了灾后损坏评估的精度，有助于更精确和高效的灾害应对，并为未来的灾害减轻和韧性策略提供了贡献。该论文已被接受参加ASCE国际土木工程计算会议（i3CE 2025），最终版本将出现在官方会议论文集中。 

---
# Order Acquisition Under Competitive Pressure: A Rapidly Adaptive Reinforcement Learning Approach for Ride-Hailing Subsidy Strategies 

**Title (ZH)**: 在竞争压力下的订单获取策略：一种快速自适应强化学习方法用于网约车补贴策略 

**Authors**: Fangzhou Shi, Xiaopeng Ke, Xinye Xiong, Kexin Meng, Chang Men, Zhengdan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02244)  

**Abstract**: The proliferation of ride-hailing aggregator platforms presents significant growth opportunities for ride-service providers by increasing order volume and gross merchandise value (GMV). On most ride-hailing aggregator platforms, service providers that offer lower fares are ranked higher in listings and, consequently, are more likely to be selected by passengers. This competitive ranking mechanism creates a strong incentive for service providers to adopt coupon strategies that lower prices to secure a greater number of orders, as order volume directly influences their long-term viability and sustainability. Thus, designing an effective coupon strategy that can dynamically adapt to market fluctuations while optimizing order acquisition under budget constraints is a critical research challenge. However, existing studies in this area remain scarce.
To bridge this gap, we propose FCA-RL, a novel reinforcement learning-based subsidy strategy framework designed to rapidly adapt to competitors' pricing adjustments. Our approach integrates two key techniques: Fast Competition Adaptation (FCA), which enables swift responses to dynamic price changes, and Reinforced Lagrangian Adjustment (RLA), which ensures adherence to budget constraints while optimizing coupon decisions on new price landscape. Furthermore, we introduce RideGym, the first dedicated simulation environment tailored for ride-hailing aggregators, facilitating comprehensive evaluation and benchmarking of different pricing strategies without compromising real-world operational efficiency. Experimental results demonstrate that our proposed method consistently outperforms baseline approaches across diverse market conditions, highlighting its effectiveness in subsidy optimization for ride-hailing service providers. 

**Abstract (ZH)**: 网约车聚合平台的 proliferations 为网约车服务提供商带来了显著的增长机会，通过增加订单量和总商品价值（GMV）。在大多数网约车聚合平台上，提供较低价格的服务提供商会被优先排名，从而更有可能被乘客选择。这种竞争排名机制为服务提供商采用降价策略以获取更多订单提供了强烈的动力，因为订单量直接影响他们的长期可行性和可持续性。因此，设计一种能够在预算约束下动态适应市场波动并优化订单获取的有效补贴策略是一项关键的研究挑战。然而，该领域的现有研究依然匮乏。

为弥补这一缺口，我们提出了FCA-RL，一种基于强化学习的新型补贴策略框架，旨在快速适应竞争对手的价格调整。我们的方法结合了两种关键技术：快速竞争适应（FCA），使快速响应动态价格变化成为可能；强化拉格朗日调整（RLA），确保在新的价格环境中遵守预算约束的同时优化补贴决策。此外，我们还引入了RideGym，这是首个专门针对网约车聚合商的仿真环境，便于在不牺牲现实运营效率的情况下对不同定价策略进行全面评估和基准测试。实验结果表明，我们提出的方法在各种市场条件下始终优于基线方法，突显了其在网约车服务提供商补贴优化方面的有效性。 

---
# Understanding Trade offs When Conditioning Synthetic Data 

**Title (ZH)**: 理解条件生成合成数据时的权衡 

**Authors**: Brandon Trabucco, Qasim Wani, Benjamin Pikus, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.02217)  

**Abstract**: Learning robust object detectors from only a handful of images is a critical challenge in industrial vision systems, where collecting high quality training data can take months. Synthetic data has emerged as a key solution for data efficient visual inspection and pick and place robotics. Current pipelines rely on 3D engines such as Blender or Unreal, which offer fine control but still require weeks to render a small dataset, and the resulting images often suffer from a large gap between simulation and reality. Diffusion models promise a step change because they can generate high quality images in minutes, yet precise control, especially in low data regimes, remains difficult. Although many adapters now extend diffusion beyond plain text prompts, the effect of different conditioning schemes on synthetic data quality is poorly understood. We study eighty diverse visual concepts drawn from four standard object detection benchmarks and compare two conditioning strategies: prompt based and layout based. When the set of conditioning cues is narrow, prompt conditioning yields higher quality synthetic data; as diversity grows, layout conditioning becomes superior. When layout cues match the full training distribution, synthetic data raises mean average precision by an average of thirty four percent and by as much as one hundred seventy seven percent compared with using real data alone. 

**Abstract (ZH)**: 仅从少量图像中学习稳健的物体检测器是工业视觉系统中的一个关键挑战，其中收集高质量的训练数据可能需要几个月的时间。合成数据已成为数据高效视觉检查和拾放机器人中的关键解决方案。当前的管道依赖于如Blender或Unreal等3D引擎，虽然提供了精细控制，但仍需要几周时间才能渲染一个小的数据集，且生成的图像经常与现实之间存在较大差距。扩散模型承诺了一个飞跃，因为它们可以在几分钟内生成高质量的图像，但特别是在低数据条件下保持精确控制仍然困难。尽管现在许多适配器已将扩散模型扩展到简单的文本提示之外，但不同的条件方案对合成数据质量的影响尚未完全理解。我们研究了四个标准物体检测基准中的八十种多样化的视觉概念，并比较了两种条件方案：提示基于和布局基于。当条件线索集合狭窄时，提示条件生成的合成数据质量更高；随着多样性的增加，布局条件变得更为优越。当布局线索匹配完整的训练分布时，与仅使用真实数据相比，合成数据可将平均准确率提高34%，最多提高177%。 

---
# EIM-TRNG: Obfuscating Deep Neural Network Weights with Encoding-in-Memory True Random Number Generator via RowHammer 

**Title (ZH)**: EIM-TRNG: 使用行hammer技术结合编码-in-内存真随机数生成器混淆深层神经网络权重 

**Authors**: Ranyang Zhou, Abeer Matar A. Almalky, Gamana Aragonda, Sabbir Ahmed, Filip Roth Trønnes-Christensen, Adnan Siraj Rakin, Shaahin Angizi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02206)  

**Abstract**: True Random Number Generators (TRNGs) play a fundamental role in hardware security, cryptographic systems, and data protection. In the context of Deep NeuralNetworks (DNNs), safeguarding model parameters, particularly weights, is critical to ensure the integrity, privacy, and intel-lectual property of AI systems. While software-based pseudo-random number generators are widely used, they lack the unpredictability and resilience offered by hardware-based TRNGs. In this work, we propose a novel and robust Encoding-in-Memory TRNG called EIM-TRNG that leverages the inherent physical randomness in DRAM cell behavior, particularly under RowHammer-induced disturbances, for the first time. We demonstrate how the unpredictable bit-flips generated through carefully controlled RowHammer operations can be harnessed as a reliable entropy source. Furthermore, we apply this TRNG framework to secure DNN weight data by encoding via a combination of fixed and unpredictable bit-flips. The encrypted data is later decrypted using a key derived from the probabilistic flip behavior, ensuring both data confidentiality and model authenticity. Our results validate the effectiveness of DRAM-based entropy extraction for robust, low-cost hardware security and offer a promising direction for protecting machine learning models at the hardware level. 

**Abstract (ZH)**: 基于内存的真随机数生成器（EIM-TRNG）在深度神经网络中的新型鲁棒奇偶校验安全机制 

---
# ESTR-CoT: Towards Explainable and Accurate Event Stream based Scene Text Recognition with Chain-of-Thought Reasoning 

**Title (ZH)**: ESTR-CoT：基于事件流的场景文本识别的可解释性和准确性提升方法及其链式推理 reasoning 

**Authors**: Xiao Wang, Jingtao Jiang, Qiang Chen, Lan Chen, Lin Zhu, Yaowei Wang, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02200)  

**Abstract**: Event stream based scene text recognition is a newly arising research topic in recent years which performs better than the widely used RGB cameras in extremely challenging scenarios, especially the low illumination, fast motion. Existing works either adopt end-to-end encoder-decoder framework or large language models for enhanced recognition, however, they are still limited by the challenges of insufficient interpretability and weak contextual logical reasoning. In this work, we propose a novel chain-of-thought reasoning based event stream scene text recognition framework, termed ESTR-CoT. Specifically, we first adopt the vision encoder EVA-CLIP (ViT-G/14) to transform the input event stream into tokens and utilize a Llama tokenizer to encode the given generation prompt. A Q-former is used to align the vision token to the pre-trained large language model Vicuna-7B and output both the answer and chain-of-thought (CoT) reasoning process simultaneously. Our framework can be optimized using supervised fine-tuning in an end-to-end manner. In addition, we also propose a large-scale CoT dataset to train our framework via a three stage processing (i.e., generation, polish, and expert verification). This dataset provides a solid data foundation for the development of subsequent reasoning-based large models. Extensive experiments on three event stream STR benchmark datasets (i.e., EventSTR, WordArt*, IC15*) fully validated the effectiveness and interpretability of our proposed framework. The source code and pre-trained models will be released on this https URL. 

**Abstract (ZH)**: 基于事件流的场景文本识别是一种近年来新兴的研究课题，相较于广泛使用的RGB相机，在极具有挑战性的场景下（尤其是低照度、快速运动）表现更佳。现有工作要么采用端到端的编码-解码框架，要么利用大规模语言模型来增强识别效果，然而它们仍然受到缺乏可解释性和弱上下文逻辑推理能力的限制。在本工作中，我们提出了一种新颖的基于链式思考的事件流场景文本识别框架，命名为ESTR-CoT。具体而言，我们首先采用Vision Encoder EVA-CLIP (ViT-G/14) 将输入的事件流转换为 tokens，并利用 Llama 令牌化器对生成提示进行编码。然后使用 Q-former 将视觉 token 对齐到预训练的大规模语言模型 Vicuna-7B，并同时输出答案和链式思考（CoT）推理过程。我们的框架可以以端到端的方式通过监督微调进行优化。此外，我们还提出了一大规模 CoT 数据集，通过三阶段处理（即生成、润色和专家验证）来训练我们的框架。该数据集为后续基于推理的大规模模型的发展提供了坚实的数据基础。在三个事件流 STR 基准数据集（即 EventSTR、WordArt*、IC15*）上的广泛实验充分验证了我们提出框架的有效性和可解释性。源代码和预训练模型将在此处 https://链接发布。 

---
# Generating Large Semi-Synthetic Graphs of Any Size 

**Title (ZH)**: 生成任意大小的大型半合成图 

**Authors**: Rodrigo Tuna, Carlos Soares  

**Link**: [PDF](https://arxiv.org/pdf/2507.02166)  

**Abstract**: Graph generation is an important area in network science. Traditional approaches focus on replicating specific properties of real-world graphs, such as small diameters or power-law degree distributions. Recent advancements in deep learning, particularly with Graph Neural Networks, have enabled data-driven methods to learn and generate graphs without relying on predefined structural properties. Despite these advances, current models are limited by their reliance on node IDs, which restricts their ability to generate graphs larger than the input graph and ignores node attributes. To address these challenges, we propose Latent Graph Sampling Generation (LGSG), a novel framework that leverages diffusion models and node embeddings to generate graphs of varying sizes without retraining. The framework eliminates the dependency on node IDs and captures the distribution of node embeddings and subgraph structures, enabling scalable and flexible graph generation. Experimental results show that LGSG performs on par with baseline models for standard metrics while outperforming them in overlooked ones, such as the tendency of nodes to form clusters. Additionally, it maintains consistent structural characteristics across graphs of different sizes, demonstrating robustness and scalability. 

**Abstract (ZH)**: 图生成是网络科学中的一个重要领域。传统方法主要关注复制真实世界图的特定属性，如小直径或幂律度分布。最近深度学习的进展，特别是图神经网络的应用，使能够通过数据驱动的方法学习和生成图，而无需依赖预定义的结构属性。尽管取得了这些进展，当前的模型仍然受限于对节点ID的依赖，这限制了它们生成比输入图更大的图的能力，并忽略了节点属性。为了解决这些问题，我们提出了潜在图采样生成（LGSG）框架，这是一种新颖的方法，利用扩散模型和节点嵌入来生成不需重新训练的各种大小的图。该框架消除了对节点ID的依赖，捕捉节点嵌入和子图结构的分布，从而实现可扩展和灵活的图生成。实验结果表明，LGSG在标准指标上与基线模型表现相当，在一些未被忽视的指标上（如节点形成集群的趋势）表现更佳。此外，它在不同大小的图上保持了结构特征的一致性，展示了鲁棒性和可扩展性。 

---
# Can Artificial Intelligence solve the blockchain oracle problem? Unpacking the Challenges and Possibilities 

**Title (ZH)**: 人工智能能否解决区块链预言机问题？解读挑战与可能性 

**Authors**: Giulio Caldarelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.02125)  

**Abstract**: The blockchain oracle problem, which refers to the challenge of injecting reliable external data into decentralized systems, remains a fundamental limitation to the development of trustless applications. While recent years have seen a proliferation of architectural, cryptographic, and economic strategies to mitigate this issue, no one has yet fully resolved the fundamental question of how a blockchain can gain knowledge about the off-chain world. In this position paper, we critically assess the role artificial intelligence (AI) can play in tackling the oracle problem. Drawing from both academic literature and practitioner implementations, we examine how AI techniques such as anomaly detection, language-based fact extraction, dynamic reputation modeling, and adversarial resistance can enhance oracle systems. We observe that while AI introduces powerful tools for improving data quality, source selection, and system resilience, it cannot eliminate the reliance on unverifiable off-chain inputs. Therefore, this study supports the idea that AI should be understood as a complementary layer of inference and filtering within a broader oracle design, not a substitute for trust assumptions. 

**Abstract (ZH)**: 区块链预言机问题：人工智能在解决外部数据注入去中心化系统挑战中的作用 

---
# Resolving Turbulent Magnetohydrodynamics: A Hybrid Operator-Diffusion Framework 

**Title (ZH)**: 解析湍流磁流体力学：一种混合算子-扩散框架 

**Authors**: Semih Kacmaz, E. A. Huerta, Roland Haas  

**Link**: [PDF](https://arxiv.org/pdf/2507.02106)  

**Abstract**: We present a hybrid machine learning framework that combines Physics-Informed Neural Operators (PINOs) with score-based generative diffusion models to simulate the full spatio-temporal evolution of two-dimensional, incompressible, resistive magnetohydrodynamic (MHD) turbulence across a broad range of Reynolds numbers ($\mathrm{Re}$). The framework leverages the equation-constrained generalization capabilities of PINOs to predict coherent, low-frequency dynamics, while a conditional diffusion model stochastically corrects high-frequency residuals, enabling accurate modeling of fully developed turbulence. Trained on a comprehensive ensemble of high-fidelity simulations with $\mathrm{Re} \in \{100, 250, 500, 750, 1000, 3000, 10000\}$, the approach achieves state-of-the-art accuracy in regimes previously inaccessible to deterministic surrogates. At $\mathrm{Re}=1000$ and $3000$, the model faithfully reconstructs the full spectral energy distributions of both velocity and magnetic fields late into the simulation, capturing non-Gaussian statistics, intermittent structures, and cross-field correlations with high fidelity. At extreme turbulence levels ($\mathrm{Re}=10000$), it remains the first surrogate capable of recovering the high-wavenumber evolution of the magnetic field, preserving large-scale morphology and enabling statistically meaningful predictions. 

**Abstract (ZH)**: 我们提出了一种结合物理知情神经算子(PINOs)和分数阶生成扩散模型的混合机器学习框架，用于模拟二维不可压缩、有电阻的磁流体动力学(MHD)湍流在整个瑞利数($\mathrm{Re}$)范围内的时空演化。该框架利用PINOs的方程约束泛化能力预测相干的低频动态，同时条件扩散模型随机校正高频残差，从而使模型能够准确模拟完全发展的湍流。该方法在瑞利数$\mathrm{Re} \in \{100, 250, 500, 750, 1000, 3000, 10000\}$的高保真模拟集合上进行训练，实现了在以前由确定性代理不可达的湍流区域中的最先进准确性。在$\mathrm{Re}=1000$和$3000$时，该模型在模拟后期准确重构了速度和磁场的完整频谱能量分布，捕获了非高斯统计、间歇结构和跨场相关性。在极端湍流水平($\mathrm{Re}=10000$)时，它是唯一能恢复磁场高波数演化的代理模型，保持了大尺度形态并促进了统计上有意义的预测。 

---
# GeoAda: Efficiently Finetune Geometric Diffusion Models with Equivariant Adapters 

**Title (ZH)**: GeoAda: 用等变适配器高效微调几何扩散模型 

**Authors**: Wanjia Zhao, Jiaqi Han, Siyi Gu, Mingjian Jiang, James Zou, Stefano Ermon  

**Link**: [PDF](https://arxiv.org/pdf/2507.02085)  

**Abstract**: Geometric diffusion models have shown remarkable success in molecular dynamics and structure generation. However, efficiently fine-tuning them for downstream tasks with varying geometric controls remains underexplored. In this work, we propose an SE(3)-equivariant adapter framework ( GeoAda) that enables flexible and parameter-efficient fine-tuning for controlled generative tasks without modifying the original model architecture. GeoAda introduces a structured adapter design: control signals are first encoded through coupling operators, then processed by a trainable copy of selected pretrained model layers, and finally projected back via decoupling operators followed by an equivariant zero-initialized convolution. By fine-tuning only these lightweight adapter modules, GeoAda preserves the model's geometric consistency while mitigating overfitting and catastrophic forgetting. We theoretically prove that the proposed adapters maintain SE(3)-equivariance, ensuring that the geometric inductive biases of the pretrained diffusion model remain intact during adaptation. We demonstrate the wide applicability of GeoAda across diverse geometric control types, including frame control, global control, subgraph control, and a broad range of application domains such as particle dynamics, molecular dynamics, human motion prediction, and molecule generation. Empirical results show that GeoAda achieves state-of-the-art fine-tuning performance while preserving original task accuracy, whereas other baselines experience significant performance degradation due to overfitting and catastrophic forgetting. 

**Abstract (ZH)**: 几何不变同构适配器框架（GeoAda）在受控生成任务中的高效细调 

---
# NGAT: A Node-level Graph Attention Network for Long-term Stock Prediction 

**Title (ZH)**: NGAT：用于长期股票预测的节点级图注意力网络 

**Authors**: Yingjie Niu, Mingchuan Zhao, Valerio Poti, Ruihai Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.02018)  

**Abstract**: Graph representation learning methods have been widely adopted in financial applications to enhance company representations by leveraging inter-firm relationships. However, current approaches face three key challenges: (1) The advantages of relational information are obscured by limitations in downstream task designs; (2) Existing graph models specifically designed for stock prediction often suffer from excessive complexity and poor generalization; (3) Experience-based construction of corporate relationship graphs lacks effective comparison of different graph structures. To address these limitations, we propose a long-term stock prediction task and develop a Node-level Graph Attention Network (NGAT) specifically tailored for corporate relationship graphs. Furthermore, we experimentally demonstrate the limitations of existing graph comparison methods based on model downstream task performance. Experimental results across two datasets consistently demonstrate the effectiveness of our proposed task and model. The project is publicly available on GitHub to encourage reproducibility and future research. 

**Abstract (ZH)**: 图表示学习方法在利用企业间关系提升公司表示以增强金融应用中的长周期股票预测任务研究中，现有方法面临三个关键挑战：（1）下游任务设计限制了关系信息的优势；（2）专门用于股票预测的图模型往往过于复杂且泛化能力差；（3）基于经验构建的企业关系图缺乏不同图结构的有效比较。为解决这些限制，我们提出了一个长周期股票预测任务，并开发了一个针对企业关系图优化的节点级图注意力网络（NGAT）。此外，我们通过实验展示了现有图比较方法基于模型下游任务性能的局限性。在两个数据集上的实验结果一致证明了我们提出任务和模型的有效性。该项目已在GitHub上开源，以促进可重复性和未来研究。 

---
# ManifoldMind: Dynamic Hyperbolic Reasoning for Trustworthy Recommendations 

**Title (ZH)**: ManifoldMind: 动态双曲推理以实现可信推荐 

**Authors**: Anoushka Harit, Zhongtian Sun, Suncica Hadzidedic  

**Link**: [PDF](https://arxiv.org/pdf/2507.02014)  

**Abstract**: We introduce ManifoldMind, a probabilistic geometric recommender system for exploratory reasoning over semantic hierarchies in hyperbolic space. Unlike prior methods with fixed curvature and rigid embeddings, ManifoldMind represents users, items, and tags as adaptive-curvature probabilistic spheres, enabling personalised uncertainty modeling and geometry-aware semantic exploration. A curvature-aware semantic kernel supports soft, multi-hop inference, allowing the model to explore diverse conceptual paths instead of overfitting to shallow or direct interactions. Experiments on four public benchmarks show superior NDCG, calibration, and diversity compared to strong baselines. ManifoldMind produces explicit reasoning traces, enabling transparent, trustworthy, and exploration-driven recommendations in sparse or abstract domains. 

**Abstract (ZH)**: ManifoldMind：超越固定曲率的类流形概率几何推荐系统 

---
# Discovery of Fatigue Strength Models via Feature Engineering and automated eXplainable Machine Learning applied to the welded Transverse Stiffener 

**Title (ZH)**: 基于特征工程和自动可解释机器学习的焊接纵加劲条疲劳强度模型发现 

**Authors**: Michael A. Kraus, Helen Bartsch  

**Link**: [PDF](https://arxiv.org/pdf/2507.02005)  

**Abstract**: This research introduces a unified approach combining Automated Machine Learning (AutoML) with Explainable Artificial Intelligence (XAI) to predict fatigue strength in welded transverse stiffener details. It integrates expert-driven feature engineering with algorithmic feature creation to enhance accuracy and explainability.
Based on the extensive fatigue test database regression models - gradient boosting, random forests, and neural networks - were trained using AutoML under three feature schemes: domain-informed, algorithmic, and combined. This allowed a systematic comparison of expert-based versus automated feature selection.
Ensemble methods (e.g. CatBoost, LightGBM) delivered top performance. The domain-informed model $\mathcal M_2$ achieved the best balance: test RMSE $\approx$ 30.6 MPa and $R^2 \approx 0.780% over the full $\Delta \sigma_{c,50\%}$ range, and RMSE $\approx$ 13.4 MPa and $R^2 \approx 0.527% within the engineering-relevant 0 - 150 MPa domain. The denser-feature model ($\mathcal M_3$) showed minor gains during training but poorer generalization, while the simpler base-feature model ($\mathcal M_1$) performed comparably, confirming the robustness of minimalist designs.
XAI methods (SHAP and feature importance) identified stress ratio $R$, stress range $\Delta \sigma_i$, yield strength $R_{eH}$, and post-weld treatment (TIG dressing vs. as-welded) as dominant predictors. Secondary geometric factors - plate width, throat thickness, stiffener height - also significantly affected fatigue life.
This framework demonstrates that integrating AutoML with XAI yields accurate, interpretable, and robust fatigue strength models for welded steel structures. It bridges data-driven modeling with engineering validation, enabling AI-assisted design and assessment. Future work will explore probabilistic fatigue life modeling and integration into digital twin environments. 

**Abstract (ZH)**: 本研究提出了一种结合自动机器学习（AutoML）和可解释人工智能（XAI）的统一方法，用于预测焊接纵撑细节的疲劳强度。该方法将专家驱动的特征工程与算法特征创建相结合，以提高准确性和可解释性。
基于广泛的疲劳试验数据库，使用AutoML训练了回归模型——梯度提升、随机森林和神经网络，并采用了三种特征方案：领域驱动、算法驱动和结合驱动，从而系统地比较了基于专家和自动化的特征选择方法。
集成方法（如CatBoost、LightGBM）表现出最佳性能。领域驱动模型$\mathcal M_2$实现了最佳平衡：测试RMSE $\approx$ 30.6 MPa，$R^2 \approx 0.780\%$覆盖了全范围$\Delta \sigma_{c,50\%}$，在工程相关的0 - 150 MPa范围内，RMSE $\approx$ 13.4 MPa，$R^2 \approx 0.527\%$。特征更密集的模型（$\mathcal M_3$）在训练期间表现出小幅增益，但泛化能力较差，而基础特征更简单的模型（$\mathcal M_1$）表现相似，证实了简约设计的稳健性。
XAI方法（SHAP和特征重要性）确定了应力比$R$、应力范围$\Delta \sigma_i$、屈服强度$R_{eH}$和焊后处理（TIG打底 vs. 无焊后处理）为主导预测因素。次要几何因素—板宽、喉部厚度、撑条高度—也显著影响疲劳寿命。
该框架表明，将AutoML与XAI结合起来可以生成准确、可解释且稳健的焊接钢结构疲劳强度模型。它连接了数据驱动建模与工程验证，实现了AI辅助设计与评估。未来工作将探索概率疲劳寿命建模并将其集成到数字孪生环境中。 

---
# DKGCM: A Spatio-Temporal Prediction Model for Traffic Flow by Fusing Spatial Node Clustering Method and Fourier Bidirectional Mamba Mechanism 

**Title (ZH)**: DKGCM：交通流的时空预测模型，通过融合空间节点聚类方法和傅里叶双向Mamba机制 

**Authors**: Siqing Long, Xiangzhi Huang, Jiemin Xie, Ming Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.01982)  

**Abstract**: Accurate traffic demand forecasting enables transportation management departments to allocate resources more effectively, thereby improving their utilization efficiency. However, complex spatiotemporal relationships in traffic systems continue to limit the performance of demand forecasting models. To improve the accuracy of spatiotemporal traffic demand prediction, we propose a new graph convolutional network structure called DKGCM. Specifically, we first consider the spatial flow distribution of different traffic nodes and propose a novel temporal similarity-based clustering graph convolution method, DK-GCN. This method utilizes Dynamic Time Warping (DTW) and K-means clustering to group traffic nodes and more effectively capture spatial dependencies. On the temporal scale, we integrate the Fast Fourier Transform (FFT) within the bidirectional Mamba deep learning framework to capture temporal dependencies in traffic demand. To further optimize model training, we incorporate the GRPO reinforcement learning strategy to enhance the loss function feedback mechanism. Extensive experiments demonstrate that our model outperforms several advanced methods and achieves strong results on three public datasets. 

**Abstract (ZH)**: 准确的交通需求预测能够使交通管理部门更有效地分配资源，进而提高资源利用效率。然而，交通系统中复杂的时空关系仍然限制了需求预测模型的性能。为提高时空交通需求预测的准确性，我们提出了一种新的图卷积网络结构，称为DKGCM。具体而言，我们首先考虑不同交通节点的时空流量分布，提出了一种基于时间相似性的新型聚类图卷积方法，即DK-GCN。该方法利用动态时间规整（DTW）和K-means聚类对交通节点进行分组，更有效地捕捉空间依赖性。在时间尺度上，我们在双向Mamba深度学习框架中集成了快速傅里叶变换（FFT），以捕捉交通需求的时间依赖性。为了进一步优化模型训练，我们将GRPO强化学习策略整合进损失函数反馈机制中。大量实验证明，我们的模型在多个公开数据集上均优于几种先进的方法，并取得了优异的结果。 

---
# Forecasting Labor Markets with LSTNet: A Multi-Scale Deep Learning Approach 

**Title (ZH)**: 基于多尺度深度学习方法的劳动力市场预测：LSTNet模型 

**Authors**: Adam Nelson-Archer, Aleia Sen, Meena Al Hasani, Sofia Davila, Jessica Le, Omar Abbouchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.01979)  

**Abstract**: We present a deep learning approach for forecasting short-term employment changes and assessing long-term industry health using labor market data from the U.S. Bureau of Labor Statistics. Our system leverages a Long- and Short-Term Time-series Network (LSTNet) to process multivariate time series data, including employment levels, wages, turnover rates, and job openings. The model outputs both 7-day employment forecasts and an interpretable Industry Employment Health Index (IEHI). Our approach outperforms baseline models across most sectors, particularly in stable industries, and demonstrates strong alignment between IEHI rankings and actual employment volatility. We discuss error patterns, sector-specific performance, and future directions for improving interpretability and generalization. 

**Abstract (ZH)**: 我们提出了一种基于深度学习的方法，使用美国劳工统计局的劳动力市场数据来预测短期就业变化并评估长期行业健康状况。该系统利用长短期时间序列网络（LSTNet）处理包括就业水平、工资、离职率和空缺职位等多变量时间序列数据。模型输出7天的就业预测和可解释的行业就业健康指数（IEHI）。该方法在大多数行业中优于基准模型，特别是在稳定行业中表现出色，IEHI排名与实际就业波动之间存在较强的正相关性。我们讨论了误差模式、行业特定性能以及提高可解释性和泛化能力的未来方向。 

---
# Learnable-Differentiable Finite Volume Solver for Accelerated Simulation of Flows 

**Title (ZH)**: 可学习可微分的有限体积求解器加速流场仿真 

**Authors**: Mengtao Yan, Qi Wang, Haining Wang, Ruizhi Chengze, Yi Zhang, Hongsheng Liu, Zidong Wang, Fan Yu, Qi Qi, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.01975)  

**Abstract**: Simulation of fluid flows is crucial for modeling physical phenomena like meteorology, aerodynamics, and biomedicine. Classical numerical solvers often require fine spatiotemporal grids to satisfy stability, consistency, and convergence conditions, leading to substantial computational costs. Although machine learning has demonstrated better efficiency, they typically suffer from issues of interpretability, generalizability, and data dependency. Hence, we propose a learnable and differentiable finite volume solver, called LDSolver, designed for efficient and accurate simulation of fluid flows on spatiotemporal coarse grids. LDSolver comprises two key components: (1) a differentiable finite volume solver, and (2) an learnable module providing equivalent approximation for fluxes (derivatives and interpolations), and temporal error correction on coarse grids. Even with limited training data (e.g., only a few trajectories), our model could accelerate the simulation while maintaining a high accuracy with superior generalizability. Experiments on different flow systems (e.g., Burgers, decaying, forced and shear flows) show that LDSolver achieves state-of-the-art performance, surpassing baseline models with notable margins. 

**Abstract (ZH)**: 时空粗网格下可学习且可微分的有限体积求解器用于流体流动高效准确模拟 

---
# Accelerated Portfolio Optimization and Option Pricing with Reinforcement Learning 

**Title (ZH)**: 加速组合优化和期权定价的强化学习方法 

**Authors**: Hadi Keramati, Samaneh Jazayeri  

**Link**: [PDF](https://arxiv.org/pdf/2507.01972)  

**Abstract**: We present a reinforcement learning (RL)-driven framework for optimizing block-preconditioner sizes in iterative solvers used in portfolio optimization and option pricing. The covariance matrix in portfolio optimization or the discretization of differential operators in option pricing models lead to large linear systems of the form $\mathbf{A}\textbf{x}=\textbf{b}$. Direct inversion of high-dimensional portfolio or fine-grid option pricing incurs a significant computational cost. Therefore, iterative methods are usually used for portfolios in real-world situations. Ill-conditioned systems, however, suffer from slow convergence. Traditional preconditioning techniques often require problem-specific parameter tuning. To overcome this limitation, we rely on RL to dynamically adjust the block-preconditioner sizes and accelerate iterative solver convergence. Evaluations on a suite of real-world portfolio optimization matrices demonstrate that our RL framework can be used to adjust preconditioning and significantly accelerate convergence and reduce computational cost. The proposed accelerated solver supports faster decision-making in dynamic portfolio allocation and real-time option pricing. 

**Abstract (ZH)**: 基于强化学习的迭代求解器中块预条件子大小优化框架 

---
# DeepSupp: Attention-Driven Correlation Pattern Analysis for Dynamic Time Series Support and Resistance Levels Identification 

**Title (ZH)**: DeepSupp：基于注意力机制的相关模式分析在动态时间序列支撑位和阻力位识别中的应用 

**Authors**: Boris Kriuk, Logic Ng, Zarif Al Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2507.01971)  

**Abstract**: Support and resistance (SR) levels are central to technical analysis, guiding traders in entry, exit, and risk management. Despite widespread use, traditional SR identification methods often fail to adapt to the complexities of modern, volatile markets. Recent research has introduced machine learning techniques to address the following challenges, yet most focus on price prediction rather than structural level identification. This paper presents DeepSupp, a new deep learning approach for detecting financial support levels using multi-head attention mechanisms to analyze spatial correlations and market microstructure relationships. DeepSupp integrates advanced feature engineering, constructing dynamic correlation matrices that capture evolving market relationships, and employs an attention-based autoencoder for robust representation learning. The final support levels are extracted through unsupervised clustering, leveraging DBSCAN to identify significant price thresholds. Comprehensive evaluations on S&P 500 tickers demonstrate that DeepSupp outperforms six baseline methods, achieving state-of-the-art performance across six financial metrics, including essential support accuracy and market regime sensitivity. With consistent results across diverse market conditions, DeepSupp addresses critical gaps in SR level detection, offering a scalable and reliable solution for modern financial analysis. Our approach highlights the potential of attention-based architectures to uncover nuanced market patterns and improve technical trading strategies. 

**Abstract (ZH)**: 基于多头注意力机制的DeepSupp：一种新的深度学习方法用于检测金融支撑水平 

---
# A Scalable and Quantum-Accurate Foundation Model for Biomolecular Force Field via Linearly Tensorized Quadrangle Attention 

**Title (ZH)**: 一种基于线性张量 quadrangle 注意机制的可扩展且量子精确的生物分子势场基础模型 

**Authors**: Qun Su, Kai Zhu, Qiaolin Gou, Jintu Zhang, Renling Hu, Yurong Li, Yongze Wang, Hui Zhang, Ziyi You, Linlong Jiang, Yu Kang, Jike Wang, Chang-Yu Hsieh, Tingjun Hou  

**Link**: [PDF](https://arxiv.org/pdf/2507.00884)  

**Abstract**: Accurate atomistic biomolecular simulations are vital for disease mechanism understanding, drug discovery, and biomaterial design, but existing simulation methods exhibit significant limitations. Classical force fields are efficient but lack accuracy for transition states and fine conformational details critical in many chemical and biological processes. Quantum Mechanics (QM) methods are highly accurate but computationally infeasible for large-scale or long-time simulations. AI-based force fields (AIFFs) aim to achieve QM-level accuracy with efficiency but struggle to balance many-body modeling complexity, accuracy, and speed, often constrained by limited training data and insufficient validation for generalizability. To overcome these challenges, we introduce LiTEN, a novel equivariant neural network with Tensorized Quadrangle Attention (TQA). TQA efficiently models three- and four-body interactions with linear complexity by reparameterizing high-order tensor features via vector operations, avoiding costly spherical harmonics. Building on LiTEN, LiTEN-FF is a robust AIFF foundation model, pre-trained on the extensive nablaDFT dataset for broad chemical generalization and fine-tuned on SPICE for accurate solvated system simulations. LiTEN achieves state-of-the-art (SOTA) performance across most evaluation subsets of rMD17, MD22, and Chignolin, outperforming leading models such as MACE, NequIP, and EquiFormer. LiTEN-FF enables the most comprehensive suite of downstream biomolecular modeling tasks to date, including QM-level conformer searches, geometry optimization, and free energy surface construction, while offering 10x faster inference than MACE-OFF for large biomolecules (~1000 atoms). In summary, we present a physically grounded, highly efficient framework that advances complex biomolecular modeling, providing a versatile foundation for drug discovery and related applications. 

**Abstract (ZH)**: 准确的原子尺度生物分子模拟对于疾病机制理解、药物发现和生物材料设计至关重要，但现有模拟方法存在显著局限性。经典力场高效但对过渡态和许多化学与生物过程中的细微构象细节缺乏准确性。量子力学（QM）方法高度准确但大规模或长时间模拟计算成本高昂。基于AI的力场(AIFF)旨在实现QM级准确性与效率，但难以平衡多体建模复杂性、准确性和速度，常受限于有限的训练数据和不足的一般化验证。为克服这些挑战，我们引入了LiTEN，这是一种新型具有张量四边形注意机制(TQA)的等变神经网络。TQA通过向量操作重新参数化高阶张量特征，以线性复杂度高效建模三体和四体相互作用，避免了昂贵的球谐变换。基于LiTEN，LiTEN-FF是稳健的AIFF基础模型，预训练于广泛的nablaDFT数据集以实现广泛的化学泛化，并在SPICE上进行微调以实现精确的溶剂化系统模拟。LiTEN在rMD17、MD22和Chignolin的大多数评估子集上实现了最先进的(SOTA)性能，超越了MACE、NequIP和EquiFormer等领先模型。LiTEN-FF使迄今为止最全面的下游生物分子建模任务组得以实现，包括QM级构象搜索、几何优化和自由能表面构造，同时对于大型生物分子（~1000个原子）的推理速度比MACE-OFF快10倍。总体而言，我们提出了一个物理上合理的、高效的方法框架，推动了复杂生物分子建模的发展，为其在药物发现及相关应用中的广泛应用提供了灵活的基础。 

---
