# Trajectory Planning for Automated Driving using Target Funnels 

**Title (ZH)**: 基于目标漏斗的自动驾驶轨迹规划 

**Authors**: Benjamin Bogenberger, Johannes Bürger, Vladislav Nenchev  

**Link**: [PDF](https://arxiv.org/pdf/2503.23795)  

**Abstract**: Self-driving vehicles rely on sensory input to monitor their surroundings and continuously adapt to the most likely future road course. Predictive trajectory planning is based on snapshots of the (uncertain) road course as a key input. Under noisy perception data, estimates of the road course can vary significantly, leading to indecisive and erratic steering behavior. To overcome this issue, this paper introduces a predictive trajectory planning algorithm with a novel objective function: instead of targeting a single reference trajectory based on the most likely road course, tracking a series of target reference sets, called a target funnel, is considered. The proposed planning algorithm integrates probabilistic information about the road course, and thus implicitly considers regular updates to road perception. Our solution is assessed in a case study using real driving data collected from a prototype vehicle. The results demonstrate that the algorithm maintains tracking accuracy and substantially reduces undesirable steering commands in the presence of noisy road perception, achieving a 56% reduction in input costs compared to a certainty equivalent formulation. 

**Abstract (ZH)**: 自动驾驶车辆依赖传感器输入监控环境，并不断适应最有可能的未来行驶路线。预测性轨迹规划基于不确定道路状况的快照作为关键输入。在噪声感知数据下，道路状况的估计会显著变化，导致不果断和不规则的转向行为。为克服这一问题，本文引入了一种具有新颖目标函数的预测性轨迹规划算法：而不是基于最有可能的道路状况针对单一参考轨迹进行规划，而是追踪一系列目标参考集，称为目标漏斗。所提出的规划算法整合了关于道路状况的概率信息，从而隐含地考虑了对道路感知的定期更新。我们的解决方案通过使用从原型车辆收集的真实驾驶数据进行案例研究进行了评估。结果表明，该算法在噪声感知道路条件下保持了跟踪准确性，并显著减少了不必要的转向命令，与等效确定性公式相比，输入成本减少了56%。 

---
# Towards Benchmarking and Assessing the Safety and Robustness of Autonomous Driving on Safety-critical Scenarios 

**Title (ZH)**: 面向安全关键场景下自动驾驶的安全性和鲁棒性基准测试与评估 

**Authors**: Jingzheng Li, Xianglong Liu, Shikui Wei, Zhijun Chen, Bing Li, Qing Guo, Xianqi Yang, Yanjun Pu, Jiakai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23708)  

**Abstract**: Autonomous driving has made significant progress in both academia and industry, including performance improvements in perception task and the development of end-to-end autonomous driving systems. However, the safety and robustness assessment of autonomous driving has not received sufficient attention. Current evaluations of autonomous driving are typically conducted in natural driving scenarios. However, many accidents often occur in edge cases, also known as safety-critical scenarios. These safety-critical scenarios are difficult to collect, and there is currently no clear definition of what constitutes a safety-critical scenario. In this work, we explore the safety and robustness of autonomous driving in safety-critical scenarios. First, we provide a definition of safety-critical scenarios, including static traffic scenarios such as adversarial attack scenarios and natural distribution shifts, as well as dynamic traffic scenarios such as accident scenarios. Then, we develop an autonomous driving safety testing platform to comprehensively evaluate autonomous driving systems, encompassing not only the assessment of perception modules but also system-level evaluations. Our work systematically constructs a safety verification process for autonomous driving, providing technical support for the industry to establish standardized test framework and reduce risks in real-world road deployment. 

**Abstract (ZH)**: 自主驾驶在安全关键场景中的安全与鲁棒性评估 

---
# Incorporating GNSS Information with LIDAR-Inertial Odometry for Accurate Land-Vehicle Localization 

**Title (ZH)**: 融合GNSS信息的LIDAR-惯性里程计定位方法及其在准确土地车辆定位中的应用 

**Authors**: Jintao Cheng, Bohuan Xue, Shiyang Chen, Qiuchi Xiang, Xiaoyu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23199)  

**Abstract**: Currently, visual odometry and LIDAR odometry are performing well in pose estimation in some typical environments, but they still cannot recover the localization state at high speed or reduce accumulated drifts. In order to solve these problems, we propose a novel LIDAR-based localization framework, which achieves high accuracy and provides robust localization in 3D pointcloud maps with information of multi-sensors. The system integrates global information with LIDAR-based odometry to optimize the localization state. To improve robustness and enable fast resumption of localization, this paper uses offline pointcloud maps for prior knowledge and presents a novel registration method to speed up the convergence rate. The algorithm is tested on various maps of different data sets and has higher robustness and accuracy than other localization algorithms. 

**Abstract (ZH)**: 基于LIDAR的新型高精度鲁棒定位框架 

---
# Distortion Bounds of Subdivision Models for SO(3) 

**Title (ZH)**: SO(3)中分拆模型的失真界 

**Authors**: Zhaoqi Zhang, Chee Yap  

**Link**: [PDF](https://arxiv.org/pdf/2503.22961)  

**Abstract**: In the subdivision approach to robot path planning, we need to subdivide the configuration space of a robot into nice cells to perform various computations. For a rigid spatial robot, this configuration space is $SE(3)=\mathbb{R}^3\times SO(3)$. The subdivision of $\mathbb{R}^3$ is standard but so far, there are no global subdivision schemes for $SO(3)$. We recently introduced a representation for $SO(3)$ suitable for subdivision. This paper investigates the distortion of the natural metric on $SO(3)$ caused by our representation. The proper framework for this study lies in the Riemannian geometry of $SO(3)$, enabling us to obtain sharp distortion bounds. 

**Abstract (ZH)**: 在机器人路径规划的细分方法中，我们需要将机器人的配置空间细分成交互良好的单元以进行各种计算。对于刚体空间机器人，该配置空间为$SE(3)=\mathbb{R}^3\times SO(3)$。虽然$\mathbb{R}^3$的细分是标准的，但目前仍没有$SO(3)$的全局细分方案。我们最近引入了一种适合细分的$SO(3)$表示。本文研究了我们表示对$SO(3)$上自然度量引起的失真。这项研究的适当框架是$SO(3)$的黎曼几何，使我们能够获得精确的失真界。 

---
# Towards Mobile Sensing with Event Cameras on High-mobility Resource-constrained Devices: A Survey 

**Title (ZH)**: 面向高机动性资源受限设备的事件 cameras 无线感测：一个综述 

**Authors**: Haoyang Wang, Ruishan Guo, Pengtao Ma, Ciyu Ruan, Xinyu Luo, Wenhua Ding, Tianyang Zhong, Jingao Xu, Yunhao Liu, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22943)  

**Abstract**: With the increasing complexity of mobile device applications, these devices are evolving toward high mobility. This shift imposes new demands on mobile sensing, particularly in terms of achieving high accuracy and low latency. Event-based vision has emerged as a disruptive paradigm, offering high temporal resolution, low latency, and energy efficiency, making it well-suited for high-accuracy and low-latency sensing tasks on high-mobility platforms. However, the presence of substantial noisy events, the lack of inherent semantic information, and the large data volume pose significant challenges for event-based data processing on resource-constrained mobile devices. This paper surveys the literature over the period 2014-2024, provides a comprehensive overview of event-based mobile sensing systems, covering fundamental principles, event abstraction methods, algorithmic advancements, hardware and software acceleration strategies. We also discuss key applications of event cameras in mobile sensing, including visual odometry, object tracking, optical flow estimation, and 3D reconstruction, while highlighting the challenges associated with event data processing, sensor fusion, and real-time deployment. Furthermore, we outline future research directions, such as improving event camera hardware with advanced optics, leveraging neuromorphic computing for efficient processing, and integrating bio-inspired algorithms to enhance perception. To support ongoing research, we provide an open-source \textit{Online Sheet} with curated resources and recent developments. We hope this survey serves as a valuable reference, facilitating the adoption of event-based vision across diverse applications. 

**Abstract (ZH)**: 随着移动设备应用程序复杂性的增加，这些设备正朝着高移动性发展。这一变化对移动传感提出了新的要求，特别是在实现高精度和低延迟方面。基于事件的视觉已经作为一种颠覆性范式出现，提供高时间分辨率、低延迟和能量效率，使其非常适合在高移动性平台上执行高精度和低延迟的传感任务。然而，大量嘈杂事件的存在、缺乏内在语义信息以及大数据量给资源受限的移动设备上的事件数据处理带来了重大挑战。本文回顾了2014-2024年的相关文献，提供了基于事件的移动传感系统的全面概述，涵盖基本原理、事件抽象方法、算法进展、硬件和软件加速策略。我们还讨论了事件摄像头在移动传感中的关键应用，包括视觉里程计、目标跟踪、光流估计和三维重建，同时指出了事件数据处理、传感器融合和实时部署相关的挑战。此外，我们提出了未来研究方向，如改进具有先进光学技术的事件摄像头硬件、利用神经形态计算进行高效处理以及整合生物启发算法来提升感知能力。为了支持持续研究，我们提供了包含精选资源和最新发展的开源《在线表格》。我们希望本文献能作为有价值的参考，促进基于事件的视觉技术在多种应用中的采用。 

---
# SALT: A Flexible Semi-Automatic Labeling Tool for General LiDAR Point Clouds with Cross-Scene Adaptability and 4D Consistency 

**Title (ZH)**: SALT：一种具有跨场景适应性与4D一致性的灵活半自动标注工具用于通用LiDAR点云 

**Authors**: Yanbo Wang, Yongtao Chen, Chuan Cao, Tianchen Deng, Wentao Zhao, Jingchuan Wang, Weidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23980)  

**Abstract**: We propose a flexible Semi-Automatic Labeling Tool (SALT) for general LiDAR point clouds with cross-scene adaptability and 4D consistency. Unlike recent approaches that rely on camera distillation, SALT operates directly on raw LiDAR data, automatically generating pre-segmentation results. To achieve this, we propose a novel zero-shot learning paradigm, termed data alignment, which transforms LiDAR data into pseudo-images by aligning with the training distribution of vision foundation models. Additionally, we design a 4D-consistent prompting strategy and 4D non-maximum suppression module to enhance SAM2, ensuring high-quality, temporally consistent presegmentation. SALT surpasses the latest zero-shot methods by 18.4% PQ on SemanticKITTI and achieves nearly 40-50% of human annotator performance on our newly collected low-resolution LiDAR data and on combined data from three LiDAR types, significantly boosting annotation efficiency. We anticipate that SALT's open-sourcing will catalyze substantial expansion of current LiDAR datasets and lay the groundwork for the future development of LiDAR foundation models. Code is available at this https URL. 

**Abstract (ZH)**: 我们提出了一种灵活的半自动标注工具(SALT)用于通用LiDAR点云，具有跨场景适应性和4D一致性。 

---
# Handling Delay in Real-Time Reinforcement Learning 

**Title (ZH)**: 处理实时强化学习中的延迟 

**Authors**: Ivan Anokhin, Rishav Rishav, Matthew Riemer, Stephen Chung, Irina Rish, Samira Ebrahimi Kahou  

**Link**: [PDF](https://arxiv.org/pdf/2503.23478)  

**Abstract**: Real-time reinforcement learning (RL) introduces several challenges. First, policies are constrained to a fixed number of actions per second due to hardware limitations. Second, the environment may change while the network is still computing an action, leading to observational delay. The first issue can partly be addressed with pipelining, leading to higher throughput and potentially better policies. However, the second issue remains: if each neuron operates in parallel with an execution time of $\tau$, an $N$-layer feed-forward network experiences observation delay of $\tau N$. Reducing the number of layers can decrease this delay, but at the cost of the network's expressivity. In this work, we explore the trade-off between minimizing delay and network's expressivity. We present a theoretically motivated solution that leverages temporal skip connections combined with history-augmented observations. We evaluate several architectures and show that those incorporating temporal skip connections achieve strong performance across various neuron execution times, reinforcement learning algorithms, and environments, including four Mujoco tasks and all MinAtar games. Moreover, we demonstrate parallel neuron computation can accelerate inference by 6-350% on standard hardware. Our investigation into temporal skip connections and parallel computations paves the way for more efficient RL agents in real-time setting. 

**Abstract (ZH)**: 实时强化学习中的延迟与网络表达能力trade-off研究：基于时间跳过连接的历史增强观测方法 

---
# OnSiteVRU: A High-Resolution Trajectory Dataset for High-Density Vulnerable Road Users 

**Title (ZH)**: OnSiteVRU: 高密度脆弱道路使用者高分辨率轨迹数据集 

**Authors**: Zhangcun Yan, Jianqing Li, Peng Hang, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.23365)  

**Abstract**: With the acceleration of urbanization and the growth of transportation demands, the safety of vulnerable road users (VRUs, such as pedestrians and cyclists) in mixed traffic flows has become increasingly prominent, necessitating high-precision and diverse trajectory data to support the development and optimization of autonomous driving systems. However, existing datasets fall short in capturing the diversity and dynamics of VRU behaviors, making it difficult to meet the research demands of complex traffic environments. To address this gap, this study developed the OnSiteVRU datasets, which cover a variety of scenarios, including intersections, road segments, and urban villages. These datasets provide trajectory data for motor vehicles, electric bicycles, and human-powered bicycles, totaling approximately 17,429 trajectories with a precision of 0.04 seconds. The datasets integrate both aerial-view natural driving data and onboard real-time dynamic detection data, along with environmental information such as traffic signals, obstacles, and real-time maps, enabling a comprehensive reconstruction of interaction events. The results demonstrate that VRU\_Data outperforms traditional datasets in terms of VRU density and scene coverage, offering a more comprehensive representation of VRU behavioral characteristics. This provides critical support for traffic flow modeling, trajectory prediction, and autonomous driving virtual testing. The dataset is publicly available for download at:
this https URL. 

**Abstract (ZH)**: 随着城市化进程的加速和交通需求的增长，混合交通流中弱势道路使用者（如行人和自行车骑行者）的安全问题日益突出，需要高精度和多样化的轨迹数据以支持自动驾驶系统的研发与优化。然而，现有数据集在捕捉弱势道路使用者行为的多样性和动态性方面存在不足，难以满足复杂交通环境下的研究需求。为弥补这一差距，本研究开发了OnSiteVRU数据集，涵盖了交叉口、道路段和城乡结合部等多种场景。该数据集提供了机动车、电动自行车和人力自行车的轨迹数据，总计约17,429条轨迹，精度达到0.04秒。数据集整合了空中视角的自然驾驶数据和车载实时动态检测数据，以及交通信号、障碍物和实时地图等环境信息，能够全面重构交互事件。结果显示，VRU_Data在弱势道路使用者密度和场景覆盖方面优于传统数据集，提供了更全面的弱势道路使用者行为特征表示。该数据集为交通流建模、轨迹预测和自动驾驶虚拟测试提供了关键支持。数据集在此处免费下载：[this https URL](this https URL)。 

---
# Energy-Aware Lane Planning for Connected Electric Vehicles in Urban Traffic: Design and Vehicle-in-the-Loop Validation 

**Title (ZH)**: 面向连接电动车辆的城市交通能效导向车道规划：设计与车辆在环验证 

**Authors**: Hansung Kim, Eric Yongkeun Choi, Eunhyek Joa, Hotae Lee, Linda Lim, Scott Moura, Francesco Borrelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.23228)  

**Abstract**: Urban driving with connected and automated vehicles (CAVs) offers potential for energy savings, yet most eco-driving strategies focus solely on longitudinal speed control within a single lane. This neglects the significant impact of lateral decisions, such as lane changes, on overall energy efficiency, especially in environments with traffic signals and heterogeneous traffic flow. To address this gap, we propose a novel energy-aware motion planning framework that jointly optimizes longitudinal speed and lateral lane-change decisions using vehicle-to-infrastructure (V2I) communication. Our approach estimates long-term energy costs using a graph-based approximation and solves short-horizon optimal control problems under traffic constraints. Using a data-driven energy model calibrated to an actual battery electric vehicle, we demonstrate with vehicle-in-the-loop experiments that our method reduces motion energy consumption by up to 24 percent compared to a human driver, highlighting the potential of connectivity-enabled planning for sustainable urban autonomy. 

**Abstract (ZH)**: 使用连接和自动驾驶车辆的城市驾驶提供了节能的潜力，然而大多数环保驾驶策略主要关注单车道内的纵向速度控制，忽略了横向决策，如车道变换，对整体能效的显著影响，特别是在有交通信号和异质交通流的环境中。为弥补这一不足，我们提出了一种新颖的能量感知运动规划框架，该框架利用车辆到基础设施（V2I）通信联合优化纵向速度和横向车道变换决策。我们的方法使用图基近似估算长期能量成本，并在交通约束下解决短期最优控制问题。通过针对实际电池电动汽车的数据驱动能量模型，我们通过车辆在环实验表明，与人类驾驶员相比，我们的方法可将运动能量消耗减少最多24%，突出了连接性规划在可持续城市自主中的潜力。 

---
# ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning 

**Title (ZH)**: ACPBench 困难版：关于行动、变化与规划的无约束推理 

**Authors**: Harsha Kokel, Michael Katz, Kavitha Srinivas, Shirin Sohrabi  

**Link**: [PDF](https://arxiv.org/pdf/2503.24378)  

**Abstract**: The ACPBench dataset provides atomic reasoning tasks required for efficient planning. The dataset is aimed at distilling the complex plan generation task into separate atomic reasoning tasks in their easiest possible form, boolean or multiple-choice questions, where the model has to choose the right answer from the provided options. While the aim of ACPBench is to test the simplest form of reasoning about action and change, when tasked with planning, a model does not typically have options to choose from and thus the reasoning required for planning dictates an open-ended, generative form for these tasks. To that end, we introduce ACPBench Hard, a generative version of ACPBench, with open-ended questions which the model needs to answer. Models that perform well on these tasks could in principle be integrated into a planner or be used directly as a policy. We discuss the complexity of these tasks as well as the complexity of validating the correctness of their answers and present validation algorithms for each task. Equipped with these validators, we test the performance of a variety of models on our tasks and find that for most of these tasks the performance of even the largest models is still subpar. Our experiments show that no model outperforms another in these tasks and with a few exceptions all tested language models score below 65%, indicating that even the current frontier language models have a long way to go before they can reliably reason about planning. In fact, even the so-called reasoning models struggle with solving these reasoning tasks. ACPBench Hard collection is available at the following link: this https URL 

**Abstract (ZH)**: ACPBench数据集提供了用于高效规划的原子推理任务。ACPBench Hard是ACPBench的生成版本，包含开放性问题，模型需要回答这些问题。我们讨论了这些任务的复杂性以及验证其答案正确性的复杂性，并为每个任务提出了验证算法。配备这些验证器后，我们测试了多种模型在这些任务上的性能，并发现大多数任务中，即使是最大的模型性能仍然不足。我们的实验表明，在这些任务中没有模型能够胜出，测试的所有语言模型得分均低于65%，表明当前的语言模型在可靠进行规划推理方面还有很长的路要走。实际上，所谓的推理模型在解决这些推理任务时也面临困难。ACPBench Hard数据集可在以下链接获取：this https URL。 

---
# Contextual Preference Collaborative Measure Framework Based on Belief System 

**Title (ZH)**: 基于信念系统的情境偏好评价协作度量框架 

**Authors**: Hang Yu, Wei Wei, Zheng Tan, Jing-lei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.24328)  

**Abstract**: To reduce the human intervention in the preference measure process,this article proposes a preference collaborative measure framework based on an updated belief system,which is also capable of improving the accuracy and efficiency of preferen-ce measure this http URL,the distance of rules and the average internal distance of rulesets are proposed for specifying the relationship between the this http URL discovering the most representative preferences that are common in all users,namely common preference,a algorithm based on average internal distance of ruleset,PRA algorithm,is proposed,which aims to finish the discoveryprocess with minimum information loss this http URL,the concept of Common belief is proposed to update the belief system,and the common preferences are the evidences of updated belief this http URL,under the belief system,the proposed belief degree and deviation degree are used to determine whether a rule confirms the belief system or not and classify the preference rules into two kinds(generalized or personalized),and eventually filters out Top-K interesting rules relying on belief degree and deviation this http URL on above,a scalable interestingness calculation framework that can apply various formulas is proposed for accurately calculating interestingness in different this http URL last,IMCos algorithm and IMCov algorithm are proposed as exemplars to verify the accuracy and efficiency of the framework by using weighted cosine similarity and correlation coefficients as belief this http URL experiments,the proposed algorithms are compared to two state-of-the-art algorithms and the results show that IMCos and IMCov outperform than the other two in most aspects. 

**Abstract (ZH)**: 基于更新信念系统的偏好协作衡量框架：减少人为干预并提高偏好衡量的准确性和效率 

---
# All You Need is Sally-Anne: ToM in AI Strongly Supported After Surpassing Tests for 3-Year-Olds 

**Title (ZH)**: 只需辛迪·兰恩：人工智能的理论思维在超过3岁儿童测试后得到强有力支持 

**Authors**: Nitay Alon, Joseph Barnby, Reuth Mirsky, Stefan Sarkadi  

**Link**: [PDF](https://arxiv.org/pdf/2503.24215)  

**Abstract**: Theory of Mind (ToM) is a hallmark of human cognition, allowing individuals to reason about others' beliefs and intentions. Engineers behind recent advances in Artificial Intelligence (AI) have claimed to demonstrate comparable capabilities. This paper presents a model that surpasses traditional ToM tests designed for 3-year-old children, providing strong support for the presence of ToM in AI systems. 

**Abstract (ZH)**: 理论心智（ToM）是人类认知的 hallmark，使个体能够推理他人的信念和意图。近期人工智能（AI）进展背后的工程师们声称展示了相当的能力。本文提出了一种模型，超越了为 3 岁儿童设计的传统 ToM 测试，为 AI 系统中存在 ToM 提供了强有力的支持。 

---
# Agent-Based Simulations of Online Political Discussions: A Case Study on Elections in Germany 

**Title (ZH)**: 基于代理的德国elections在线政治讨论模拟：个案研究 

**Authors**: Abdul Sittar, Simon Münker, Fabio Sartori, Andreas Reitenbach, Achim Rettinger, Michael Mäs, Alenka Guček, Marko Grobelnik  

**Link**: [PDF](https://arxiv.org/pdf/2503.24199)  

**Abstract**: User engagement on social media platforms is influenced by historical context, time constraints, and reward-driven interactions. This study presents an agent-based simulation approach that models user interactions, considering past conversation history, motivation, and resource constraints. Utilizing German Twitter data on political discourse, we fine-tune AI models to generate posts and replies, incorporating sentiment analysis, irony detection, and offensiveness classification. The simulation employs a myopic best-response model to govern agent behavior, accounting for decision-making based on expected rewards. Our results highlight the impact of historical context on AI-generated responses and demonstrate how engagement evolves under varying constraints. 

**Abstract (ZH)**: 社交媒体平台上的用户参与受历史背景、时间限制和奖赏驱动力交互的影响：基于代理的仿真研究——以政治 discourse中的德国Twitter数据为例，调整AI模型生成帖子和回复，结合情感分析、讽刺检测和冒犯分类，并采用短视最佳响应模型来管理代理行为，展示历史背景对AI生成响应的影响及其在不同约束下的参与演变。 

---
# What the F*ck Is Artificial General Intelligence? 

**Title (ZH)**: 什么是通用人工智能？ 

**Authors**: Michael Timothy Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2503.23923)  

**Abstract**: Artificial general intelligence (AGI) is an established field of research. Yet Melanie Mitchell and others have questioned if the term still has meaning. AGI has been subject to so much hype and speculation it has become something of a Rorschach test. Mitchell points out that the debate will only be settled through long term, scientific investigation. To that end here is a short, accessible and provocative overview of AGI. I compare definitions of intelligence, settling on intelligence in terms of adaptation and AGI as an artificial scientist. Taking my queue from Sutton's Bitter Lesson I describe two foundational tools used to build adaptive systems: search and approximation. I compare pros, cons, hybrids and architectures like o3, AlphaGo, AERA, NARS and Hyperon. I then discuss overall meta-approaches to making systems behave more intelligently. I divide them into scale-maxing, simp-maxing, w-maxing based on the Bitter Lesson, Ockham's and Bennett's Razors. These maximise resources, simplicity of form, and the weakness of constraints on functionality. I discuss examples including AIXI, the free energy principle and The Embiggening of language models. I conclude that though scale-maxed approximation dominates, AGI will be a fusion of tools and meta-approaches. The Embiggening was enabled by improvements in hardware. Now the bottlenecks are sample and energy efficiency. 

**Abstract (ZH)**: 人工通用智能：一个富有启发性的简要概述 

---
# MolGround: A Benchmark for Molecular Grounding 

**Title (ZH)**: MolGround: 分子接地基准数据集 

**Authors**: Jiaxin Wu, Ting Zhang, Rubing Chen, Wengyu Zhang, Chen Jason Zhang, Xiaoyong Wei, Li Qing  

**Link**: [PDF](https://arxiv.org/pdf/2503.23668)  

**Abstract**: Current molecular understanding approaches predominantly focus on the descriptive aspect of human perception, providing broad, topic-level insights. However, the referential aspect -- linking molecular concepts to specific structural components -- remains largely unexplored. To address this gap, we propose a molecular grounding benchmark designed to evaluate a model's referential abilities. We align molecular grounding with established conventions in NLP, cheminformatics, and molecular science, showcasing the potential of NLP techniques to advance molecular understanding within the AI for Science movement. Furthermore, we constructed the largest molecular understanding benchmark to date, comprising 79k QA pairs, and developed a multi-agent grounding prototype as proof of concept. This system outperforms existing models, including GPT-4o, and its grounding outputs have been integrated to enhance traditional tasks such as molecular captioning and ATC (Anatomical, Therapeutic, Chemical) classification. 

**Abstract (ZH)**: 当前分子理解方法主要聚焦于人类感知的描述方面，提供了广泛的主题级洞察。然而，参照方面——将分子概念与特定结构成分联系起来——仍然很大程度上未被探索。为解决这一问题，我们提出了一种分子 grounding 基准，旨在评估模型的参照能力。我们将分子 grounding 与 NLP、化学生物信息学和分子科学中的既定规范相结合，展示了 NLP 技术在科学人工智能运动中推动分子理解的潜力。此外，我们构建了迄今为止最大的分子理解基准，包含 79,000 个 QA 对，并开发了一个多代理 grounding 模型作为概念验证。该系统超越了现有模型，包括 GPT-4o，并将其 grounding 输出整合到传统的分子标注和 ATC (Anatomical, Therapeutic, Chemical) 分类任务中以提升性能。 

---
# Intrinsically-Motivated Humans and Agents in Open-World Exploration 

**Title (ZH)**: 内在动机驱动的人与代理在开放世界探索中 

**Authors**: Aly Lidayan, Yuqing Du, Eliza Kosoy, Maria Rufova, Pieter Abbeel, Alison Gopnik  

**Link**: [PDF](https://arxiv.org/pdf/2503.23631)  

**Abstract**: What drives exploration? Understanding intrinsic motivation is a long-standing challenge in both cognitive science and artificial intelligence; numerous objectives have been proposed and used to train agents, yet there remains a gap between human and agent exploration. We directly compare adults, children, and AI agents in a complex open-ended environment, Crafter, and study how common intrinsic objectives: Entropy, Information Gain, and Empowerment, relate to their behavior. We find that only Entropy and Empowerment are consistently positively correlated with human exploration progress, indicating that these objectives may better inform intrinsic reward design for agents. Furthermore, across agents and humans we observe that Entropy initially increases rapidly, then plateaus, while Empowerment increases continuously, suggesting that state diversity may provide more signal in early exploration, while advanced exploration should prioritize control. Finally, we find preliminary evidence that private speech utterances, and particularly goal verbalizations, may aid exploration in children. 

**Abstract (ZH)**: 什么是推动探索的动力？理解内在动机是认知科学和人工智能领域长期面临的挑战；尽管提出了众多目标用于训练代理，但人类和代理的探索之间仍然存在差距。我们直接将成人、儿童和AI代理置于复杂的开放性环境中Crafter进行比较，并研究熵、信息增益和權力这三种常见内在目标与其行为之间的关系。研究发现，只有熵和權力与人类的探索进程表现出一致的正相关，表明这些目标可能更好地指导代理的内在奖励设计。此外，我们发现，在代理和人类中，熵最初迅速增加，随后趋于稳定，而權力持续增加，这表明早期探索中状态多样性可能提供更多信号，而高级探索应优先考虑控制。最后，我们初步发现，私人性言语陈述，尤其是目标言语化，可能有助于儿童的探索。 

---
# Beyond Detection: Designing AI-Resilient Assessments with Automated Feedback Tool to Foster Critical Thinking 

**Title (ZH)**: 超越检测：设计具有自动反馈工具的AI抗扰评估以培养批判性思维 

**Authors**: Muhammad Sajjad Akbar  

**Link**: [PDF](https://arxiv.org/pdf/2503.23622)  

**Abstract**: The growing use of generative AI tools like ChatGPT has raised urgent concerns about their impact on student learning, particularly the potential erosion of critical thinking and creativity. As students increasingly turn to these tools to complete assessments, foundational cognitive skills are at risk of being bypassed, challenging the integrity of higher education and the authenticity of student work. Existing AI-generated text detection tools are inadequate; they produce unreliable outputs and are prone to both false positives and false negatives, especially when students apply paraphrasing, translation, or rewording. These systems rely on shallow statistical patterns rather than true contextual or semantic understanding, making them unsuitable as definitive indicators of AI misuse. In response, this research proposes a proactive, AI-resilient solution based on assessment design rather than detection. It introduces a web-based Python tool that integrates Bloom's Taxonomy with advanced natural language processing techniques including GPT-3.5 Turbo, BERT-based semantic similarity, and TF-IDF metrics to evaluate the AI-solvability of assessment tasks. By analyzing surface-level and semantic features, the tool helps educators determine whether a task targets lower-order thinking such as recall and summarization or higher-order skills such as analysis, evaluation, and creation, which are more resistant to AI automation. This framework empowers educators to design cognitively demanding, AI-resistant assessments that promote originality, critical thinking, and fairness. It offers a sustainable, pedagogically sound strategy to foster authentic learning and uphold academic standards in the age of AI. 

**Abstract (ZH)**: 生成式AI工具（如ChatGPT）的广泛应用引起了对学生学习影响的紧迫关切，特别是批判性思维和创造力可能受损的问题。随着学生越来越多地依赖这些工具来完成评估任务，基础认知技能面临被绕过的风险，这挑战了高等教育的完整性和学生作品的真实性。现有的AI生成文本检测工具不足，它们输出不可靠，并且容易产生误报和漏报，尤其是在学生使用改写、翻译或重新措辞时。这些系统依赖于浅层次的统计模式而非真正的上下文或语义理解，使其不适合作为AI不当使用的确切指标。为应对这一挑战，本研究提出了一种基于评估设计的前瞻性和AI抗性解决方案，而非依赖检测。该研究介绍了一个基于Python的web工具，该工具结合了布鲁姆分类学与高级自然语言处理技术，包括GPT-3.5 Turbo、基于BERT的语义相似度以及TF-IDF指标，以评估评估任务的AI可解性。通过分析表层和语义特征，该工具帮助教育工作者判断任务是针对较低层次的思考能力（如记忆和总结）还是较高层次的思考能力（如分析、评估和创造），后者更难以被AI自动化。该框架赋予教育工作者设计认知要求高、对抗AI的评估工具的能力，以促进原创性、批判性思维和公平性。它提供了一种可持续且教育学上合理的策略，以促进真实的学习并维护AI时代的学术标准。 

---
# An Organizationally-Oriented Approach to Enhancing Explainability and Control in Multi-Agent Reinforcement Learning 

**Title (ZH)**: 面向组织的多代理 reinforcement 学习解释性和可控性增强方法 

**Authors**: Julien Soulé, Jean-Paul Jamont, Michel Occello, Louis-Marie Traonouez, Paul Théron  

**Link**: [PDF](https://arxiv.org/pdf/2503.23615)  

**Abstract**: Multi-Agent Reinforcement Learning can lead to the development of collaborative agent behaviors that show similarities with organizational concepts. Pushing forward this perspective, we introduce a novel framework that explicitly incorporates organizational roles and goals from the $\mathcal{M}OISE^+$ model into the MARL process, guiding agents to satisfy corresponding organizational constraints. By structuring training with roles and goals, we aim to enhance both the explainability and control of agent behaviors at the organizational level, whereas much of the literature primarily focuses on individual agents. Additionally, our framework includes a post-training analysis method to infer implicit roles and goals, offering insights into emergent agent behaviors. This framework has been applied across various MARL environments and algorithms, demonstrating coherence between predefined organizational specifications and those inferred from trained agents. 

**Abstract (ZH)**: 多智能体强化学习可以发展出与组织概念相似的合作智能体行为。在此基础上，我们提出了一种新颖的框架，该框架明确地将$\mathcal{M}OISE^+$模型中的组织角色和目标纳入多智能体强化学习过程，引导智能体满足相应的组织约束。通过按角色和目标结构化训练，我们旨在增强智能体行为在组织层面的可解释性和可控性，而现有文献主要关注个体智能体。此外，我们的框架还包括一种后训练分析方法，用于推断隐含的角色和目标，提供对涌现智能体行为的见解。该框架已在各种多智能体强化学习环境中应用，展示了预定义的组织规范与从训练智能体推断出的规范之间的一致性。 

---
# A Systematic Decade Review of Trip Route Planning with Travel Time Estimation based on User Preferences and Behavior 

**Title (ZH)**: 基于用户偏好和行为的旅行时间估算的旅游路线规划系统性十年回顾 

**Authors**: Nikil Jayasuriya, Deshan Sumanathilaka  

**Link**: [PDF](https://arxiv.org/pdf/2503.23486)  

**Abstract**: This paper systematically explores the advancements in adaptive trip route planning and travel time estimation (TTE) through Artificial Intelligence (AI). With the increasing complexity of urban transportation systems, traditional navigation methods often struggle to accommodate dynamic user preferences, real-time traffic conditions, and scalability requirements. This study explores the contributions of established AI techniques, including Machine Learning (ML), Reinforcement Learning (RL), and Graph Neural Networks (GNNs), alongside emerging methodologies like Meta-Learning, Explainable AI (XAI), Generative AI, and Federated Learning. In addition to highlighting these innovations, the paper identifies critical challenges such as ethical concerns, computational scalability, and effective data integration, which must be addressed to advance the field. The paper concludes with recommendations for leveraging AI to build efficient, transparent, and sustainable navigation systems. 

**Abstract (ZH)**: 本文系统性地探讨了人工智能在自适应旅行路线规划和旅行时间估计（TTE）方面的进步。随着城市交通系统的日益复杂，传统的导航方法往往难以满足动态用户偏好、实时交通状况和可扩展性要求。本研究探索了包括机器学习（ML）、强化学习（RL）和图神经网络（GNN）在内的现有AI技术的贡献，以及元学习、可解释的AI（XAI）、生成AI和联邦学习等新兴方法。除了强调这些创新之外，本文还指出了诸如伦理问题、计算可扩展性和有效的数据集成等关键挑战，这些挑战必须得到解决以推进该领域的发展。本文最后提出了利用AI构建高效、透明和可持续导航系统的建议。 

---
# Exploring Explainable Multi-player MCTS-minimax Hybrids in Board Game Using Process Mining 

**Title (ZH)**: 探索基于过程挖掘的可解释多玩家MCTS- minimax混合算法在棋类游戏中的应用 

**Authors**: Yiyu Qian, Tim Miller, Zheng Qian, Liyuan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.23326)  

**Abstract**: Monte-Carlo Tree Search (MCTS) is a family of sampling-based search algorithms widely used for online planning in sequential decision-making domains and at the heart of many recent advances in artificial intelligence. Understanding the behavior of MCTS agents is difficult for developers and users due to the frequently large and complex search trees that result from the simulation of many possible futures, their evaluations, and their relationships. This paper presents our ongoing investigation into potential explanations for the decision-making and behavior of MCTS. A weakness of MCTS is that it constructs a highly selective tree and, as a result, can miss crucial moves and fall into tactical traps. Full-width minimax search constitutes the solution. We integrate shallow minimax search into the rollout phase of multi-player MCTS and use process mining technique to explain agents' strategies in 3v3 checkers. 

**Abstract (ZH)**: 基于蒙特卡罗树搜索的博弈决策与行为研究：浅层次极小极大搜索在三目国际象棋中的应用 

---
# FindTheFlaws: Annotated Errors for Detecting Flawed Reasoning and Scalable Oversight Research 

**Title (ZH)**: FindTheFlaws: 注释错误以检测瑕疵推理与可扩展监督研究 

**Authors**: Gabriel Recchia, Chatrik Singh Mangat, Issac Li, Gayatri Krishnakumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.22989)  

**Abstract**: As AI models tackle increasingly complex problems, ensuring reliable human oversight becomes more challenging due to the difficulty of verifying solutions. Approaches to scaling AI supervision include debate, in which two agents engage in structured dialogue to help a judge evaluate claims; critique, in which models identify potential flaws in proposed solutions; and prover-verifier games, in which a capable 'prover' model generates solutions that must be verifiable by a less capable 'verifier'. Evaluations of the scalability of these and similar approaches to difficult problems benefit from datasets that include (1) long-form expert-verified correct solutions and (2) long-form flawed solutions with annotations highlighting specific errors, but few are available.
To address this gap, we present FindTheFlaws, a group of five diverse datasets spanning medicine, mathematics, science, coding, and the Lojban language. Each dataset contains questions and long-form solutions with expert annotations validating their correctness or identifying specific error(s) in the reasoning. We evaluate frontier models' critiquing capabilities and observe a range of performance that can be leveraged for scalable oversight experiments: models performing more poorly on particular datasets can serve as judges/verifiers for more capable models. Additionally, for some task/dataset combinations, expert baselines exceed even top model performance, making them more beneficial for scalable oversight experiments. 

**Abstract (ZH)**: 随着AI模型面临的問題日益複雜，確保可靠的人類監督變得更加困難，因為驗證解決方案的難度增大。擴展AI監督的策略包括辯論（兩個代理進行結構化對話以幫助評審 avaliação 判斷斷言）、批判（模型識別提出的解決方案中的潛在缺陷）、以及證明者-驗證者遊戲（有能力的“證明者”模型生成必須能夠被較無能力的“驗證者”模型驗證的解決方案）。對這些及類似方法擴展性的評估需要包括（1）長篇專家驗證正確的解決方案和（2）帶有標注具體錯誤的長篇錯誤解決方案等數據集，但這樣的數據集很少見。

為了彌補這一-gap，我們介紹了 FindTheFlaws，這是涵蓋醫學、數學、科學、編程和洛 Basics 語言的五個多樣化數據集組。每個數據集包含問題和長篇解決方案，並附有專家注釋以確認其 correctness 或標注推理中的特定錯誤。我們評估前沿模型的批評能力，並觀察到一系列性能，這些性能可以為可擴展的監督實驗提供支援：在某些數據集上表現較差的模型可以作為較有能力模型的評審/驗證者。此外，對於一些任務/數據集組合，專家基線表現甚至超越頂尖模型表現，使其在可擴展的監督實驗中更有益。 

---
# CodeScientist: End-to-End Semi-Automated Scientific Discovery with Code-based Experimentation 

**Title (ZH)**: CodeScientist：基于代码实验的端到端半自动化科学发现 

**Authors**: Peter Jansen, Oyvind Tafjord, Marissa Radensky, Pao Siangliulue, Tom Hope, Bhavana Dalvi Mishra, Bodhisattwa Prasad Majumder, Daniel S. Weld, Peter Clark  

**Link**: [PDF](https://arxiv.org/pdf/2503.22708)  

**Abstract**: Despite the surge of interest in autonomous scientific discovery (ASD) of software artifacts (e.g., improved ML algorithms), current ASD systems face two key limitations: (1) they largely explore variants of existing codebases or similarly constrained design spaces, and (2) they produce large volumes of research artifacts (such as automatically generated papers and code) that are typically evaluated using conference-style paper review with limited evaluation of code. In this work we introduce CodeScientist, a novel ASD system that frames ideation and experiment construction as a form of genetic search jointly over combinations of research articles and codeblocks defining common actions in a domain (like prompting a language model). We use this paradigm to conduct hundreds of automated experiments on machine-generated ideas broadly in the domain of agents and virtual environments, with the system returning 19 discoveries, 6 of which were judged as being both at least minimally sound and incrementally novel after a multi-faceted evaluation beyond that typically conducted in prior work, including external (conference-style) review, code review, and replication attempts. Moreover, the discoveries span new tasks, agents, metrics, and data, suggesting a qualitative shift from benchmark optimization to broader discoveries. 

**Abstract (ZH)**: 尽管对自主科学发现（ASD）软件构件的兴趣激增（例如，改进的ML算法），当前的ASD系统面临两个关键限制：（1）它们主要探索现有代码库的变体或类似约束的设计空间；（2）它们生成大量研究构件（如自动生成的论文和代码），这些构件通常使用会议风格的论文评审方式进行评估，代码的评估则更为有限。在此项工作中，我们引入了CodeScientist，这是一种新颖的ASD系统，将构想和实验构建视为在研究文章和定义域中常见操作的代码块组合上的形式基因搜索。我们使用这一范式对广泛涉及代理和虚拟环境领域的机器生成构想进行数百次自动化实验，系统返回了19项发现，其中6项在多方面评估（包括外部会议评审、代码评审和复制尝试）之后被判定为至少具有最小的合理性和逐步新颖性。此外，这些发现涵盖了新的任务、代理、指标和数据，暗示着从基准优化到更广泛的发现的定性转变。 

---
# UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving 

**Title (ZH)**: UniOcc：自动驾驶中 occupancy 预测与估计统一基准 

**Authors**: Yuping Wang, Xiangyu Huang, Xiaokang Sun, Mingxuan Yan, Shuo Xing, Zhengzhong Tu, Jiachen Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.24381)  

**Abstract**: We introduce UniOcc, a comprehensive, unified benchmark for occupancy forecasting (i.e., predicting future occupancies based on historical information) and current-frame occupancy prediction from camera images. UniOcc unifies data from multiple real-world datasets (i.e., nuScenes, Waymo) and high-fidelity driving simulators (i.e., CARLA, OpenCOOD), which provides 2D/3D occupancy labels with per-voxel flow annotations and support for cooperative autonomous driving. In terms of evaluation, unlike existing studies that rely on suboptimal pseudo labels for evaluation, UniOcc incorporates novel metrics that do not depend on ground-truth occupancy, enabling robust assessment of additional aspects of occupancy quality. Through extensive experiments on state-of-the-art models, we demonstrate that large-scale, diverse training data and explicit flow information significantly enhance occupancy prediction and forecasting performance. 

**Abstract (ZH)**: UniOcc：一种综合统一的 occupancy 预测基准（包括基于历史信息的未来occupancy预测和当前帧occupancy预测）及其在相机图像中的应用 

---
# Which LIME should I trust? Concepts, Challenges, and Solutions 

**Title (ZH)**: Which LIME Should I Trust? 概念、挑战与解决方案 

**Authors**: Patrick Knab, Sascha Marton, Udo Schlegel, Christian Bartelt  

**Link**: [PDF](https://arxiv.org/pdf/2503.24365)  

**Abstract**: As neural networks become dominant in essential systems, Explainable Artificial Intelligence (XAI) plays a crucial role in fostering trust and detecting potential misbehavior of opaque models. LIME (Local Interpretable Model-agnostic Explanations) is among the most prominent model-agnostic approaches, generating explanations by approximating the behavior of black-box models around specific instances. Despite its popularity, LIME faces challenges related to fidelity, stability, and applicability to domain-specific problems. Numerous adaptations and enhancements have been proposed to address these issues, but the growing number of developments can be overwhelming, complicating efforts to navigate LIME-related research. To the best of our knowledge, this is the first survey to comprehensively explore and collect LIME's foundational concepts and known limitations. We categorize and compare its various enhancements, offering a structured taxonomy based on intermediate steps and key issues. Our analysis provides a holistic overview of advancements in LIME, guiding future research and helping practitioners identify suitable approaches. Additionally, we provide a continuously updated interactive website (this https URL), offering a concise and accessible overview of the survey. 

**Abstract (ZH)**: 随着神经网络在关键系统中的主导地位不断提升，可解释的人工智能（XAI）在促进信任并检测不透明模型潜在不当行为方面发挥着重要作用。LIME（局部可解释模型无关解释）是其中最突出的模型无关方法之一，通过近似黑盒模型在特定实例周围的行為来生成解释。尽管LIME备受青睐，但它面临着精度、稳定性和对特定领域问题的应用性等方面的挑战。提出了诸多适应性和增强措施以应对这些问题，但不断增长的发展数量也可能令人感到困惑，增加了导航LIME相关研究的难度。据我们所知，这是首次对LIME的基础概念及其已知局限性进行全面探索和收集的综述。我们对各种增强措施进行了分类和比较，基于中间步骤和关键问题提出了结构化的分类体系。我们的分析提供了LIME进展的全景概述，指导未来研究并帮助实践者识别合适的方案。此外，我们提供了一个不断更新的交互式网站（这个 https URL），提供综述的简洁和易访问概览。 

---
# SQuat: Subspace-orthogonal KV Cache Quantization 

**Title (ZH)**: SQuat: 子空间正交键值缓存量化 

**Authors**: Hao Wang, Ligong Han, Kai Xu, Akash Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2503.24358)  

**Abstract**: The key-value (KV) cache accelerates LLMs decoding by storing KV tensors from previously generated tokens. It reduces redundant computation at the cost of increased memory usage. To mitigate this overhead, existing approaches compress KV tensors into lower-bit representations; however, quantization errors can accumulate as more tokens are generated, potentially resulting in undesired outputs. In this paper, we introduce SQuat (Subspace-orthogonal KV cache quantization). It first constructs a subspace spanned by query tensors to capture the most critical task-related information. During key tensor quantization, it enforces that the difference between the (de)quantized and original keys remains orthogonal to this subspace, minimizing the impact of quantization errors on the attention mechanism's outputs. SQuat requires no model fine-tuning, no additional calibration dataset for offline learning, and is grounded in a theoretical framework we develop. Through numerical experiments, we show that our method reduces peak memory by 2.17 to 2.82, improves throughput by 2.45 to 3.60, and achieves more favorable benchmark scores than existing KV cache quantization algorithms. 

**Abstract (ZH)**: 基于子空间正交性的键值缓存量化（SQuat） 

---
# Evaluating machine learning models for predicting pesticides toxicity to honey bees 

**Title (ZH)**: 评估机器学习模型预测农药对蜜蜂毒性的能力 

**Authors**: Jakub Adamczyk, Jakub Poziemski, Pawel Siedlecki  

**Link**: [PDF](https://arxiv.org/pdf/2503.24305)  

**Abstract**: Small molecules play a critical role in the biomedical, environmental, and agrochemical domains, each with distinct physicochemical requirements and success criteria. Although biomedical research benefits from extensive datasets and established benchmarks, agrochemical data remain scarce, particularly with respect to species-specific toxicity. This work focuses on ApisTox, the most comprehensive dataset of experimentally validated chemical toxicity to the honey bee (\textit{Apis mellifera}), an ecologically vital pollinator. We evaluate ApisTox using a diverse suite of machine learning approaches, including molecular fingerprints, graph kernels, and graph neural networks, as well as pretrained models. Comparative analysis with medicinal datasets from the MoleculeNet benchmark reveals that ApisTox represents a distinct chemical space. Performance degradation on non-medicinal datasets, such as ApisTox, demonstrates their limited generalizability of current state-of-the-art algorithms trained solely on biomedical data. Our study highlights the need for more diverse datasets and for targeted model development geared toward the agrochemical domain. 

**Abstract (ZH)**: 小分子在生物医学、环境和农化领域中扮演着关键角色，各自具有独特的物理化学要求和成功标准。尽管生物医学研究得益于丰富的数据集和现有的基准，农化数据依然稀缺，尤其是在物种特异性毒性方面。本文专注于ApisTox，这是最全面的实验验证蜂蜜bee（Apis mellifera）化学毒性的数据集，蜂蜜bee是生态上重要的传粉者。我们使用多种机器学习方法，包括分子指纹、图核和图神经网络，以及预训练模型来评估ApisTox。与MoleculeNet基准中的医药数据集的对比分析表明，ApisTox代表了独特的化学空间。在非医药数据集上的性能下降表明，当前仅基于生物医学数据训练的先进算法的泛化能力有限。我们的研究强调了需要更多样化的数据集以及针对农化领域的目标模型开发的重要性。 

---
# Shape Expressions with Inheritance 

**Title (ZH)**: 继承关系中的形状表达 

**Authors**: Iovka Boneva, Jose Emilio Labra Gayo, Eric Prud'hommeaux, Katherine Thornton, Andra Waagmeester  

**Link**: [PDF](https://arxiv.org/pdf/2503.24299)  

**Abstract**: We formally introduce an inheritance mechanism for the Shape Expressions language (ShEx). It is inspired by inheritance in object-oriented programming languages, and provides similar advantages such as reuse, modularity, and more flexible data modelling. Using an example, we explain the main features of the inheritance mechanism. We present its syntax and formal semantics. The semantics is an extension of the semantics of ShEx 2.1. It also directly yields a validation algorithm as an extension of the previous ShEx validation algorithms, while maintaining the same algorithmic complexity. 

**Abstract (ZH)**: 我们正式引入了Shape Expressions语言（ShEx）的继承机制。该机制受到面向对象编程语言中继承的启发，提供了类似的优势，如重用、模块化和更灵活的数据建模。通过一个示例，我们解释了继承机制的主要特性。我们展示了其语法和形式语义。该语义是ShEx 2.1语义的扩展，同时还直接提供了一个验证算法的扩展，保持了相同的算法复杂度。 

---
# Value of Information-based Deceptive Path Planning Under Adversarial Interventions 

**Title (ZH)**: 基于价值信息的欺骗性路径规划在对抗干预下的价值 

**Authors**: Wesley A. Suttle, Jesse Milzman, Mustafa O. Karabag, Brian M. Sadler, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2503.24284)  

**Abstract**: Existing methods for deceptive path planning (DPP) address the problem of designing paths that conceal their true goal from a passive, external observer. Such methods do not apply to problems where the observer has the ability to perform adversarial interventions to impede the path planning agent. In this paper, we propose a novel Markov decision process (MDP)-based model for the DPP problem under adversarial interventions and develop new value of information (VoI) objectives to guide the design of DPP policies. Using the VoI objectives we propose, path planning agents deceive the adversarial observer into choosing suboptimal interventions by selecting trajectories that are of low informational value to the observer. Leveraging connections to the linear programming theory for MDPs, we derive computationally efficient solution methods for synthesizing policies for performing DPP under adversarial interventions. In our experiments, we illustrate the effectiveness of the proposed solution method in achieving deceptiveness under adversarial interventions and demonstrate the superior performance of our approach to both existing DPP methods and conservative path planning approaches on illustrative gridworld problems. 

**Abstract (ZH)**: 现有的欺骗性路径规划方法解决了设计隐藏真正目标的路径以避开被动外部观察者的问题。这些方法不适用于观察者能够采取对抗性干预以阻碍路径规划代理的情况。本文提出了一种在对抗性干预下基于马尔可夫决策过程（MDP）的新颖模型，并开发了新的信息价值（VoI）目标以指导欺骗性路径规划（DPP）策略的设计。利用MDP的线性规划理论，我们推导出在对抗性干预下合成DPP策略的计算效率高的解法。在我们的实验中，我们展示了所提出的方法在对抗性干预下实现欺骗性的有效性，并在示例网格世界问题上证明了我们的方法优于现有的DPP方法和保守的路径规划方法。 

---
# New Statistical Framework for Extreme Error Probability in High-Stakes Domains for Reliable Machine Learning 

**Title (ZH)**: 高风险领域中可靠机器学习的极端误差概率新统计框架 

**Authors**: Umberto Michelucci, Francesca Venturini  

**Link**: [PDF](https://arxiv.org/pdf/2503.24262)  

**Abstract**: Machine learning is vital in high-stakes domains, yet conventional validation methods rely on averaging metrics like mean squared error (MSE) or mean absolute error (MAE), which fail to quantify extreme errors. Worst-case prediction failures can have substantial consequences, but current frameworks lack statistical foundations for assessing their probability. In this work a new statistical framework, based on Extreme Value Theory (EVT), is presented that provides a rigorous approach to estimating worst-case failures. Applying EVT to synthetic and real-world datasets, this method is shown to enable robust estimation of catastrophic failure probabilities, overcoming the fundamental limitations of standard cross-validation. This work establishes EVT as a fundamental tool for assessing model reliability, ensuring safer AI deployment in new technologies where uncertainty quantification is central to decision-making or scientific analysis. 

**Abstract (ZH)**: 基于极值理论的机器学习模型最坏情况失败概率的统计框架：超越标准交叉验证的基本限制并确保新科技中AI部署的安全性 

---
# Beyond a Single Mode: GAN Ensembles for Diverse Medical Data Generation 

**Title (ZH)**: 超越单一模式：GAN集成用于生成多元医疗数据 

**Authors**: Lorenzo Tronchin, Tommy Löfstedt, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2503.24258)  

**Abstract**: The advancement of generative AI, particularly in medical imaging, confronts the trilemma of ensuring high fidelity, diversity, and efficiency in synthetic data generation. While Generative Adversarial Networks (GANs) have shown promise across various applications, they still face challenges like mode collapse and insufficient coverage of real data distributions. This work explores the use of GAN ensembles to overcome these limitations, specifically in the context of medical imaging. By solving a multi-objective optimisation problem that balances fidelity and diversity, we propose a method for selecting an optimal ensemble of GANs tailored for medical data. The selected ensemble is capable of generating diverse synthetic medical images that are representative of true data distributions and computationally efficient. Each model in the ensemble brings a unique contribution, ensuring minimal redundancy. We conducted a comprehensive evaluation using three distinct medical datasets, testing 22 different GAN architectures with various loss functions and regularisation techniques. By sampling models at different training epochs, we crafted 110 unique configurations. The results highlight the capability of GAN ensembles to enhance the quality and utility of synthetic medical images, thereby improving the efficacy of downstream tasks such as diagnostic modelling. 

**Abstract (ZH)**: 生成式AI的进步，特别是在医学成像领域，面临高保真度、多样性和效率之间的三难困境。虽然生成对抗网络（GANs）在各种应用中显示出了潜力，但仍面临模式坍塌和真实数据分布覆盖不足等挑战。本研究探讨了使用GAN集成来克服这些限制，特别是在医学成像领域。通过解决兼顾保真度和多样性的多目标优化问题，我们提出了一种方法，用于选择最适合医学数据的GAN集成。所选集成能够生成类似于真实数据分布的多样化合成医学图像，并具有计算效率。每个集成中的模型都贡献独特，确保了极小的冗余。我们使用三个不同的医学数据集进行了全面评估，测试了22种不同结构的GAN架构，以及多种损失函数和正则化技术。通过在不同训练周期采样模型，我们创建了110种独特的配置。结果表明，GAN集成能够提高合成医学图像的质量和实用性，从而提高下游任务（如诊断建模）的效率。 

---
# Spatio-temporal Prediction of Fine-Grained Origin-Destination Matrices with Applications in Ridesharing 

**Title (ZH)**: 精细粒度起源目的地矩阵的空间时间预测及其在拼车中的应用 

**Authors**: Run Yang, Runpeng Dai, Siran Gao, Xiaocheng Tang, Fan Zhou, Hongtu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.24237)  

**Abstract**: Accurate spatial-temporal prediction of network-based travelers' requests is crucial for the effective policy design of ridesharing platforms. Having knowledge of the total demand between various locations in the upcoming time slots enables platforms to proactively prepare adequate supplies, thereby increasing the likelihood of fulfilling travelers' requests and redistributing idle drivers to areas with high potential demand to optimize the global supply-demand equilibrium. This paper delves into the prediction of Origin-Destination (OD) demands at a fine-grained spatial level, especially when confronted with an expansive set of local regions. While this task holds immense practical value, it remains relatively unexplored within the research community. To fill this gap, we introduce a novel prediction model called OD-CED, which comprises an unsupervised space coarsening technique to alleviate data sparsity and an encoder-decoder architecture to capture both semantic and geographic dependencies. Through practical experimentation, OD-CED has demonstrated remarkable results. It achieved an impressive reduction of up to 45% reduction in root-mean-square error and 60% in weighted mean absolute percentage error over traditional statistical methods when dealing with OD matrices exhibiting a sparsity exceeding 90%. 

**Abstract (ZH)**: 基于网络的出行请求的时空预测对于rideshares平台有效政策设计至关重要。了解未来时间槽中各区域间的总需求量能使平台提前准备充足的供应，从而提高满足出行请求的可能性，并将闲置司机重新分配到潜在需求高的区域，以优化全局的供需平衡。本文专注于在细粒度空间级别预测出行生成地-目的地（OD）需求，特别是在面对广泛的本地区域集合时。尽管这一任务具有巨大的实际价值，但在学术界仍鲜有研究。为填补这一空白，我们提出了一个名为OD-CED的新预测模型，该模型结合了无监督的空间粗糙化技术来缓解数据稀疏问题，并采用编码器-解码器架构来捕捉语义和地理依赖关系。通过实际实验，OD-CED展现出了显著的效果。在处理OD矩阵稀疏度超过90%的情况下，它在根均方误差和加权平均绝对百分比误差上分别实现了高达45%和60%的降低，超过了传统统计方法。 

---
# Learning a Canonical Basis of Human Preferences from Binary Ratings 

**Title (ZH)**: 从二元评分学习人类偏好的典范基markt-être 

**Authors**: Kailas Vodrahalli, Wei Wei, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2503.24150)  

**Abstract**: Recent advances in generative AI have been driven by alignment techniques such as reinforcement learning from human feedback (RLHF). RLHF and related techniques typically involve constructing a dataset of binary or ranked choice human preferences and subsequently fine-tuning models to align with these preferences. This paper shifts the focus to understanding the preferences encoded in such datasets and identifying common human preferences. We find that a small subset of 21 preference categories (selected from a set of nearly 5,000 distinct preferences) captures >89% of preference variation across individuals. This small set of preferences is analogous to a canonical basis of human preferences, similar to established findings that characterize human variation in psychology or facial recognition studies. Through both synthetic and empirical evaluations, we confirm that our low-rank, canonical set of human preferences generalizes across the entire dataset and within specific topics. We further demonstrate our preference basis' utility in model evaluation, where our preference categories offer deeper insights into model alignment, and in model training, where we show that fine-tuning on preference-defined subsets successfully aligns the model accordingly. 

**Abstract (ZH)**: 近期生成AI的进步得益于对人类反馈强化学习（RLHF）等对齐技术的推动。本论文将重点转向理解这类数据集中编码的偏好，并识别常见的人类偏好。我们发现，从近5000种独特偏好中选出的21个偏好类别 captures >89%的个体偏好差异。这个小的偏好集合类似于人类偏好的标准基底，类似于心理学或面部识别研究中确立的人类差异特征。通过合成和实证评估，我们确认我们的低秩、标准化人类偏好集合在整个数据集和特定主题内具有泛化能力。此外，我们展示了偏好基底在模型评估中的应用价值，我们的偏好类别为模型对齐提供了更深入的洞察，并在模型训练中证明，基于偏好定义的子集调整成功地使模型对齐。 

---
# Resonance: Drawing from Memories to Imagine Positive Futures through AI-Augmented Journaling 

**Title (ZH)**: 共振：通过AI增强日记想象积极未来的方式汲取记忆 

**Authors**: Wazeer Zulfikar, Treyden Chiaravalloti, Jocelyn Shen, Rosalind Picard, Pattie Maes  

**Link**: [PDF](https://arxiv.org/pdf/2503.24145)  

**Abstract**: People inherently use experiences of their past while imagining their future, a capability that plays a crucial role in mental health. Resonance is an AI-powered journaling tool designed to augment this ability by offering AI-generated, action-oriented suggestions for future activities based on the user's own past memories. Suggestions are offered when a new memory is logged and are followed by a prompt for the user to imagine carrying out the suggestion. In a two-week randomized controlled study (N=55), we found that using Resonance significantly improved mental health outcomes, reducing the users' PHQ8 scores, a measure of current depression, and increasing their daily positive affect, particularly when they would likely act on the suggestion. Notably, the effectiveness of the suggestions was higher when they were personal, novel, and referenced the user's logged memories. Finally, through open-ended feedback, we discuss the factors that encouraged or hindered the use of the tool. 

**Abstract (ZH)**: 人们在想象未来时会固有地利用过去的体验，这一能力在心理健康方面发挥着重要作用。Resonance是一款以AI为动力的日记工具，旨在通过根据用户的个人 past 记忆提供基于行动的 AI 生成建议来增强这一能力，以促进未来的活动想象。在一项为期两周的随机对照研究（N=55）中，我们发现使用 Resonance 显著改善了心理健康结果，降低了用户 PHQ8 评分（当前抑郁的衡量标准），并增加了他们的日间积极情绪，尤其是在他们很可能采取建议行动时。值得注意的是，当建议具有个性化、新颖性且参考了用户登录的记忆时，建议的有效性更高。最后，通过开放式反馈，我们讨论了促进或阻碍使用该工具的因素。 

---
# Bayesian Predictive Coding 

**Title (ZH)**: 贝叶斯预测编码 

**Authors**: Alexander Tschantz, Magnus Koudahl, Hampus Linander, Lancelot Da Costa, Conor Heins, Jeff Beck, Christopher Buckley  

**Link**: [PDF](https://arxiv.org/pdf/2503.24016)  

**Abstract**: Predictive coding (PC) is an influential theory of information processing in the brain, providing a biologically plausible alternative to backpropagation. It is motivated in terms of Bayesian inference, as hidden states and parameters are optimised via gradient descent on variational free energy. However, implementations of PC rely on maximum \textit{a posteriori} (MAP) estimates of hidden states and maximum likelihood (ML) estimates of parameters, limiting their ability to quantify epistemic uncertainty. In this work, we investigate a Bayesian extension to PC that estimates a posterior distribution over network parameters. This approach, termed Bayesian Predictive coding (BPC), preserves the locality of PC and results in closed-form Hebbian weight updates. Compared to PC, our BPC algorithm converges in fewer epochs in the full-batch setting and remains competitive in the mini-batch setting. Additionally, we demonstrate that BPC offers uncertainty quantification comparable to existing methods in Bayesian deep learning, while also improving convergence properties. Together, these results suggest that BPC provides a biologically plausible method for Bayesian learning in the brain, as well as an attractive approach to uncertainty quantification in deep learning. 

**Abstract (ZH)**: Bayesian Predictive Coding: A Biologically Plausible Method for Bayesian Learning and Uncertainty Quantification 

---
# CITRAS: Covariate-Informed Transformer for Time Series Forecasting 

**Title (ZH)**: CITRAS: 带有协变量的变压器用于时间序列预测 

**Authors**: Yosuke Yamaguchi, Issei Suemitsu, Wenpeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.24007)  

**Abstract**: Covariates play an indispensable role in practical time series forecasting, offering rich context from the past and sometimes extending into the future. However, their availability varies depending on the scenario, and situations often involve multiple target variables simultaneously. Moreover, the cross-variate dependencies between them are multi-granular, with some covariates having a short-term impact on target variables and others showing long-term correlations. This heterogeneity and the intricate dependencies arising in covariate-informed forecasting present significant challenges to existing deep models. To address these issues, we propose CITRAS, a patch-based Transformer that flexibly leverages multiple targets and covariates covering both the past and the future forecasting horizon. While preserving the strong autoregressive capabilities of the canonical Transformer, CITRAS introduces two novel mechanisms in patch-wise cross-variate attention: Key-Value (KV) Shift and Attention Score Smoothing. KV Shift seamlessly incorporates future known covariates into the forecasting of target variables based on their concurrent dependencies. Additionally, Attention Score Smoothing transforms locally accurate patch-wise cross-variate dependencies into global variate-level dependencies by smoothing the past series of attention scores. Experimentally, CITRAS achieves state-of-the-art performance in both covariate-informed and multivariate forecasting, demonstrating its versatile ability to leverage cross-variate dependency for improved forecasting accuracy. 

**Abstract (ZH)**: 协变量在实际时间序列预测中扮演着不可或缺的角色，提供丰富的过去和有时甚至未来的上下文。然而，它们的可用性因场景而异，且情况往往同时涉及多个目标变量。此外，协变量间的交叉依赖是多尺度的，有些协变量对目标变量有短期影响，而其他协变量则显示出长期相关性。这种异质性和协变量知情预测中引发的复杂依赖关系对现有深度模型构成了重大挑战。为了解决这些问题，我们提出了CITRAS，一种基于补丁的Transformer，灵活利用覆盖过去和未来预测范围的多个目标变量和协变量。在保留标准Transformer强大的自回归能力的同时，CITRAS引入了两种新的机制：Key-Value (KV) Shift和Attention Score Smoothing。KV Shift无缝地根据当前依赖关系将未来的已知协变量纳入目标变量的预测。另外，Attention Score Smoothing通过平滑过去的注意力分数，将局部准确的补丁级交叉依赖关系转化为全局变量级依赖关系。实验表明，CITRAS在协变量知情和多变量预测中均达到了最先进的性能，展示了其利用交叉依赖关系提升预测准确性的 versatility。 

---
# Deep Learning Model Deployment in Multiple Cloud Providers: an Exploratory Study Using Low Computing Power Environments 

**Title (ZH)**: 多云提供商环境下基于低计算资源的深度学习模型部署探索性研究 

**Authors**: Elayne Lemos, Rodrigo Oliveira, Jairson Rodrigues, Rosalvo F. Oliveira Neto  

**Link**: [PDF](https://arxiv.org/pdf/2503.23988)  

**Abstract**: The deployment of Machine Learning models at cloud have grown by tech companies. Hardware requirements are higher when these models involve Deep Learning (DL) techniques and the cloud providers' costs may be a barrier. We explore deploying DL models using for experiments the GECToR model, a DL solution for Grammatical Error Correction, across three of the major cloud platforms (AWS, Google Cloud, Azure). We evaluate real-time latency, hardware usage and cost at each cloud provider by 7 execution environments with 10 experiments reproduced. We found that while GPUs excel in performance, they had an average cost 300% higher than solutions without GPU. Our analysis also identifies that processor cache size is crucial for cost-effective CPU deployments, enabling over 50% of cost reduction compared to GPUs. This study demonstrates the feasibility and affordability of cloud-based DL inference solutions without GPUs, benefiting resource-constrained users like startups. 

**Abstract (ZH)**: 云平台上基于机器学习模型的部署和技术公司的发展。采用GECToR模型探究深度学习模型在三大云平台（AWS、Google Cloud、Azure）上的部署。通过7个执行环境和10次实验评估各云提供商的实时延迟、硬件使用和成本。研究发现，虽然GPU在性能上表现出色，但其平均成本比无GPU解决方案高300%。分析还表明，处理器缓存大小对于降低成本的关键CPU部署至关重要，可实现超过50%的成本减少。本研究展示了在不使用GPU的情况下，云-Based深度学习推理解决方案的可行性和经济性，惠及资源受限的用户，如初创企业。 

---
# Deep Nets as Hamiltonians 

**Title (ZH)**: 深度网络作为哈密顿量 

**Authors**: Mike Winer, Boris Hanin  

**Link**: [PDF](https://arxiv.org/pdf/2503.23982)  

**Abstract**: Neural networks are complex functions of both their inputs and parameters. Much prior work in deep learning theory analyzes the distribution of network outputs at a fixed a set of inputs (e.g. a training dataset) over random initializations of the network parameters. The purpose of this article is to consider the opposite situation: we view a randomly initialized Multi-Layer Perceptron (MLP) as a Hamiltonian over its inputs. For typical realizations of the network parameters, we study the properties of the energy landscape induced by this Hamiltonian, focusing on the structure of near-global minimum in the limit of infinite width. Specifically, we use the replica trick to perform an exact analytic calculation giving the entropy (log volume of space) at a given energy. We further derive saddle point equations that describe the overlaps between inputs sampled iid from the Gibbs distribution induced by the random MLP. For linear activations we solve these saddle point equations exactly. But we also solve them numerically for a variety of depths and activation functions, including $\tanh, \sin, \text{ReLU}$, and shaped non-linearities. We find even at infinite width a rich range of behaviors. For some non-linearities, such as $\sin$, for instance, we find that the landscapes of random MLPs exhibit full replica symmetry breaking, while shallow $\tanh$ and ReLU networks or deep shaped MLPs are instead replica symmetric. 

**Abstract (ZH)**: 神经网络是其输入和参数的复杂函数。本文旨在研究随机初始化神经网络参数情况下，网络输入的海森堡量纲诱导的能量景观性质，特别是在网络宽度无限大时，近全局最小值的结构。我们使用复制技巧进行精确的解析计算，给出给定能量下的熵（空间体积的对数）。进一步推导描述从由随机多层感知机诱导的吉布斯分布中独立同分布采样输入之间的重叠的鞍点方程。对于线性激活函数，我们精确求解了这些鞍点方程。我们还对不同深度和激活函数进行了数值求解，包括tanh、sin、ReLU以及各种非线性。我们发现，在网络宽度无限大时，随机多层感知机的能量景观表现出丰富的行为。例如，对于sin激活函数，随机多层感知机的能量景观表现出完全的复制对称性破坏，而对于浅层tanh和ReLU网络或深层形状感知机，则表现出复制对称性。 

---
# HumanAesExpert: Advancing a Multi-Modality Foundation Model for Human Image Aesthetic Assessment 

**Title (ZH)**: HumanAesExpert: 推动多模态基础模型在人体图像美学评估中的应用 

**Authors**: Zhichao Liao, Xiaokun Liu, Wenyu Qin, Qingyu Li, Qiulin Wang, Pengfei Wan, Di Zhang, Long Zeng, Pingfa Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.23907)  

**Abstract**: Image Aesthetic Assessment (IAA) is a long-standing and challenging research task. However, its subset, Human Image Aesthetic Assessment (HIAA), has been scarcely explored, even though HIAA is widely used in social media, AI workflows, and related domains. To bridge this research gap, our work pioneers a holistic implementation framework tailored for HIAA. Specifically, we introduce HumanBeauty, the first dataset purpose-built for HIAA, which comprises 108k high-quality human images with manual annotations. To achieve comprehensive and fine-grained HIAA, 50K human images are manually collected through a rigorous curation process and annotated leveraging our trailblazing 12-dimensional aesthetic standard, while the remaining 58K with overall aesthetic labels are systematically filtered from public datasets. Based on the HumanBeauty database, we propose HumanAesExpert, a powerful Vision Language Model for aesthetic evaluation of human images. We innovatively design an Expert head to incorporate human knowledge of aesthetic sub-dimensions while jointly utilizing the Language Modeling (LM) and Regression head. This approach empowers our model to achieve superior proficiency in both overall and fine-grained HIAA. Furthermore, we introduce a MetaVoter, which aggregates scores from all three heads, to effectively balance the capabilities of each head, thereby realizing improved assessment precision. Extensive experiments demonstrate that our HumanAesExpert models deliver significantly better performance in HIAA than other state-of-the-art models. Our datasets, models, and codes are publicly released to advance the HIAA community. Project webpage: this https URL 

**Abstract (ZH)**: 图像美学评估（IAA）是一项长期而具有挑战性的研究任务。然而，其子集，人类图像美学评估（HIAA），虽在社交媒体、AI工作流及相关领域广泛应用，但仍未受到广泛探索。为弥合这一研究差距，我们的工作开创了一种适用于HIAA的整体实施框架。具体而言，我们引入了HumanBeauty，这是首个专为HIAA构建的数据集，包含10.8万张高质量的人像图片并附有人工标注。为实现全面且细粒度的HIAA，5万张人像图片通过 rigorous curation 过程手工收集并在我们开创性的12维美学标准下进行标注，剩余5.8万张图片则根据整体美学标签从公共数据集中系统筛选。基于HumanBeauty数据库，我们提出了HumanAesExpert，这是一种强大的视觉语言模型，用于评估人像图片的美学。我们创新地设计了一个专家头，将人类对美学子维度的知识纳入其中，并结合语言模型（LM）和回归头进行联合利用。这种方法赋予了我们的模型在整体和细粒度HIAA方面卓越的专业能力。此外，我们引入了一种元投票器（MetaVoter），它可以有效综合三个头的评分，从而平衡每个头的能力，实现提高评估精度。广泛实验表明，我们的HumanAesExpert模型在HIAA方面显著优于其他最先进的模型。我们的数据集、模型和代码已公开发布，以推动HIAA社区的发展。项目网页: this https URL。 

---
# DiffScale: Continuous Downscaling and Bias Correction of Subseasonal Wind Speed Forecasts using Diffusion Models 

**Title (ZH)**: DiffScale：基于扩散模型的子季节风速预报的连续下-scaling和偏差校正 

**Authors**: Maximilian Springenberg, Noelia Otero, Yuxin Xue, Jackie Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.23893)  

**Abstract**: Renewable resources are strongly dependent on local and large-scale weather situations. Skillful subseasonal to seasonal (S2S) forecasts -- beyond two weeks and up to two months -- can offer significant socioeconomic advantages to the energy sector. This study aims to enhance wind speed predictions using a diffusion model with classifier-free guidance to downscale S2S forecasts of surface wind speed. We propose DiffScale, a diffusion model that super-resolves spatial information for continuous downscaling factors and lead times. Leveraging weather priors as guidance for the generative process of diffusion models, we adopt the perspective of conditional probabilities on sampling super-resolved S2S forecasts. We aim to directly estimate the density associated with the target S2S forecasts at different spatial resolutions and lead times without auto-regression or sequence prediction, resulting in an efficient and flexible model. Synthetic experiments were designed to super-resolve wind speed S2S forecasts from the European Center for Medium-Range Weather Forecast (ECMWF) from a coarse resolution to a finer resolution of ERA5 reanalysis data, which serves as a high-resolution target. The innovative aspect of DiffScale lies in its flexibility to downscale arbitrary scaling factors, enabling it to generalize across various grid resolutions and lead times -without retraining the model- while correcting model errors, making it a versatile tool for improving S2S wind speed forecasts. We achieve a significant improvement in prediction quality, outperforming baselines up to week 3. 

**Abstract (ZH)**: 利用去 classifier 指导的扩散模型提升表面风速子季节至季节预报的超分辨率预测 

---
# When Counterfactual Reasoning Fails: Chaos and Real-World Complexity 

**Title (ZH)**: 当反事实推理失效：混沌与现实世界复杂性 

**Authors**: Yahya Aalaila, Gerrit Großmann, Sumantrak Mukherjee, Jonas Wahl, Sebastian Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2503.23820)  

**Abstract**: Counterfactual reasoning, a cornerstone of human cognition and decision-making, is often seen as the 'holy grail' of causal learning, with applications ranging from interpreting machine learning models to promoting algorithmic fairness. While counterfactual reasoning has been extensively studied in contexts where the underlying causal model is well-defined, real-world causal modeling is often hindered by model and parameter uncertainty, observational noise, and chaotic behavior. The reliability of counterfactual analysis in such settings remains largely unexplored. In this work, we investigate the limitations of counterfactual reasoning within the framework of Structural Causal Models. Specifically, we empirically investigate \emph{counterfactual sequence estimation} and highlight cases where it becomes increasingly unreliable. We find that realistic assumptions, such as low degrees of model uncertainty or chaotic dynamics, can result in counterintuitive outcomes, including dramatic deviations between predicted and true counterfactual trajectories. This work urges caution when applying counterfactual reasoning in settings characterized by chaos and uncertainty. Furthermore, it raises the question of whether certain systems may pose fundamental limitations on the ability to answer counterfactual questions about their behavior. 

**Abstract (ZH)**: 基于结构因果模型的反事实推理限制：混沌和不确定性下的应用探究 

---
# Conformal uncertainty quantification to evaluate predictive fairness of foundation AI model for skin lesion classes across patient demographics 

**Title (ZH)**: 符合患者人群分布的皮肤病变类别基础AI模型预测公平性的一致性不确定性量化评估 

**Authors**: Swarnava Bhattacharyya, Umapada Pal, Tapabrata Chakraborti  

**Link**: [PDF](https://arxiv.org/pdf/2503.23819)  

**Abstract**: Deep learning based diagnostic AI systems based on medical images are starting to provide similar performance as human experts. However these data hungry complex systems are inherently black boxes and therefore slow to be adopted for high risk applications like healthcare. This problem of lack of transparency is exacerbated in the case of recent large foundation models, which are trained in a self supervised manner on millions of data points to provide robust generalisation across a range of downstream tasks, but the embeddings generated from them happen through a process that is not interpretable, and hence not easily trustable for clinical applications. To address this timely issue, we deploy conformal analysis to quantify the predictive uncertainty of a vision transformer (ViT) based foundation model across patient demographics with respect to sex, age and ethnicity for the tasks of skin lesion classification using several public benchmark datasets. The significant advantage of this method is that conformal analysis is method independent and it not only provides a coverage guarantee at population level but also provides an uncertainty score for each individual. We used a model-agnostic dynamic F1-score-based sampling during model training, which helped to stabilize the class imbalance and we investigate the effects on uncertainty quantification (UQ) with or without this bias mitigation step. Thus we show how this can be used as a fairness metric to evaluate the robustness of the feature embeddings of the foundation model (Google DermFoundation) and thus advance the trustworthiness and fairness of clinical AI. 

**Abstract (ZH)**: 基于深度学习的医学图像诊断AI系统在某些方面已达到人类专家的性能水平。然而，这些对数据需求大且内部机制不透明的复杂系统，在应用于高风险领域如医疗保健时，推广速度较慢。尤其是对于近期训练于大量数据点并提供跨多种下游任务稳健泛化的大型自监督基础模型，其生成的嵌入表示过程不具可解释性，这在临床应用中难以建立信任。为解决这一紧迫问题，我们采用了容错分析方法，对基于视觉变换器（ViT）的基础模型在多种公开基准数据集上的皮肤病变分类任务中，按性别、年龄和种族不同患者群体的预测不确定性进行了定量分析。这种方法的主要优势在于，容错分析方法与模型无关，不仅在群体水平上提供了覆盖保证，还为每个个体提供了不确定性评分。我们在模型训练中采用了一种模型无关的动力学F1分数采样方法，有助于稳定类别不平衡问题，并研究了此偏差缓解步骤对不确定性量化（UQ）的影响。我们展示了如何使用这种公平性指标来评估基础模型（Google DermFoundation）的特征嵌入的稳健性，从而提高临床AI的信任度和公平性。 

---
# WinoWhat: A Parallel Corpus of Paraphrased WinoGrande Sentences with Common Sense Categorization 

**Title (ZH)**: Winogradwhat：一个具有常识分类的并行改写Winograd Grande句子语料库 

**Authors**: Ine Gevers, Victor De Marez, Luna De Bruyne, Walter Daelemans  

**Link**: [PDF](https://arxiv.org/pdf/2503.23779)  

**Abstract**: In this study, we take a closer look at how Winograd schema challenges can be used to evaluate common sense reasoning in LLMs. Specifically, we evaluate generative models of different sizes on the popular WinoGrande benchmark. We release WinoWhat, a new corpus, in which each instance of the WinoGrande validation set is paraphrased. Additionally, we evaluate the performance on the challenge across five common sense knowledge categories, giving more fine-grained insights on what types of knowledge are more challenging for LLMs. Surprisingly, all models perform significantly worse on WinoWhat, implying that LLM reasoning capabilities are overestimated on WinoGrande. To verify whether this is an effect of benchmark memorization, we match benchmark instances to LLM trainingdata and create two test-suites. We observe that memorization has a minimal effect on model performance on WinoGrande. 

**Abstract (ZH)**: 本研究更深入地探讨了WinogradSchema挑战如何用于评估大规模语言模型的常识推理能力。具体而言，我们在流行的WinoGrande基准上评估了不同规模的生成模型。我们发布了WinoWhat数据集，其中每个WinoGrande验证集的实例都被重新表述。此外，我们在五个常识知识类别上评估了挑战的表现，提供了更细致的见解，了解哪些类型的知识对语言模型更具挑战性。令人惊讶的是，所有模型在WinoWhat上的表现显著较差，这表明在WinoGrande上的语言模型推理能力可能被高估了。为了验证这是否是由于基准记忆效应，我们将基准实例与语言模型训练数据匹配，并创建了两个测试套件。我们观察到，记忆对WinoGrande上模型性能的影响较小。 

---
# GNN-Based Candidate Node Predictor for Influence Maximization in Temporal Graphs 

**Title (ZH)**: 基于GNN的时序图影响最大化候选节点预测器 

**Authors**: Priyanka Gautam, Balasubramaniam Natarajan, Sai Munikoti, S M Ferdous, Mahantesh Halappanavar  

**Link**: [PDF](https://arxiv.org/pdf/2503.23713)  

**Abstract**: In an age where information spreads rapidly across social media, effectively identifying influential nodes in dynamic networks is critical. Traditional influence maximization strategies often fail to keep up with rapidly evolving relationships and structures, leading to missed opportunities and inefficiencies. To address this, we propose a novel learning-based approach integrating Graph Neural Networks (GNNs) with Bidirectional Long Short-Term Memory (BiLSTM) models. This hybrid framework captures both structural and temporal dynamics, enabling accurate prediction of candidate nodes for seed set selection. The bidirectional nature of BiLSTM allows our model to analyze patterns from both past and future network states, ensuring adaptability to changes over time. By dynamically adapting to graph evolution at each time snapshot, our approach improves seed set calculation efficiency, achieving an average of 90% accuracy in predicting potential seed nodes across diverse networks. This significantly reduces computational overhead by optimizing the number of nodes evaluated for seed selection. Our method is particularly effective in fields like viral marketing and social network analysis, where understanding temporal dynamics is crucial. 

**Abstract (ZH)**: 在信息快速通过社交媒体传播的时代，动态网络中重要节点的有效识别至关重要。传统的影响力最大化策略往往无法跟上迅速变化的关系和结构，导致错失机会和低效。为解决这一问题，我们提出了一种结合图神经网络（GNN）和双向长短期记忆（BiLSTM）模型的新型学习方法。该混合框架捕捉了结构和时序动态，使得能够准确预测候选节点以供种子集选择。BiLSTM的双向性质使模型能够分析过去和未来的网络状态模式，确保随着时间变化的适应性。通过在每个时间切片上动态适应图演变，我们的方法提高了种子集计算效率，在多种网络中平均准确率达到90%的潜在种子节点预测。这种方法大幅减少了计算开销，通过优化种子选择过程中的节点评估数量。我们的方法特别适用于病毒营销和社会网络分析等领域，其中了解时序动态至关重要。 

---
# Remarks on the Polyak-Lojasiewicz inequality and the convergence of gradient systems 

**Title (ZH)**: 关于Polyak-Lojasiewicz不等式的一些注记及梯度系统收敛性的研究 

**Authors**: Arthur Castello B. de Oliveira, Leilei Cui, Eduardo D. Sontag  

**Link**: [PDF](https://arxiv.org/pdf/2503.23641)  

**Abstract**: This work explores generalizations of the Polyak-Lojasiewicz inequality (PLI) and their implications for the convergence behavior of gradient flows in optimization problems. Motivated by the continuous-time linear quadratic regulator (CT-LQR) policy optimization problem -- where only a weaker version of the PLI is characterized in the literature -- this work shows that while weaker conditions are sufficient for global convergence to, and optimality of the set of critical points of the cost function, the "profile" of the gradient flow solution can change significantly depending on which "flavor" of inequality the cost satisfies. After a general theoretical analysis, we focus on fitting the CT-LQR policy optimization problem to the proposed framework, showing that, in fact, it can never satisfy a PLI in its strongest form. We follow up our analysis with a brief discussion on the difference between continuous- and discrete-time LQR policy optimization, and end the paper with some intuition on the extension of this framework to optimization problems with L1 regularization and solved through proximal gradient flows. 

**Abstract (ZH)**: 这项工作探讨了Polyak-Lojasiewicz不等式（PLI）的一般化及其对优化问题中梯度流收敛行为的影响。受连续时间线性二次调节器（CT-LQR）策略优化问题的启发——在文献中仅描述了较弱版本的PLI——这项工作表明，虽然较弱的条件对于全局收敛到成本函数的临界点及其最优解是足够的，但成本函数满足的“不等式风味”不同，其梯度流解的“轮廓”可能会有显著变化。在一般理论分析之后，我们将注意力集中在将CT-LQR策略优化问题拟合到所提出的框架上，结果显示实际上它不可能满足PLI的最严格形式。随后，我们简要讨论了连续时间和离散时间LQR策略优化之间的差异，并在论文结尾对将该框架扩展到带有L1正则化并通过近端梯度流求解的优化问题进行了一些直观解释。 

---
# Finding Interest Needle in Popularity Haystack: Improving Retrieval by Modeling Item Exposure 

**Title (ZH)**: 在流行度haystack中寻找兴趣针：通过建模项目曝光改善检索 

**Authors**: Amit Jaspal, Rahul Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2503.23630)  

**Abstract**: Recommender systems operate in closed feedback loops, where user interactions reinforce popularity bias, leading to over-recommendation of already popular items while under-exposing niche or novel content. Existing bias mitigation methods, such as Inverse Propensity Scoring (IPS) and Off- Policy Correction (OPC), primarily operate at the ranking stage or during training, lacking explicit real-time control over exposure dynamics. In this work, we introduce an exposure- aware retrieval scoring approach, which explicitly models item exposure probability and adjusts retrieval-stage ranking at inference time. Unlike prior work, this method decouples exposure effects from engagement likelihood, enabling controlled trade-offs between fairness and engagement in large-scale recommendation platforms. We validate our approach through online A/B experiments in a real-world video recommendation system, demonstrating a 25% increase in uniquely retrieved items and a 40% reduction in the dominance of over-popular content, all while maintaining overall user engagement levels. Our results establish a scalable, deployable solution for mitigating popularity bias at the retrieval stage, offering a new paradigm for bias-aware personalization. 

**Abstract (ZH)**: 推荐系统在封闭的反馈循环中运作，用户交互强化了流行性偏差，导致过度推荐已有流行项目，而限制了小众或新颖内容的曝光。现有的偏差缓解方法，如逆倾向评分（IPS）和离策训练修正（OPC），主要在排序阶段或训练过程中运作，缺乏对曝光动态的显式实时控制。在本文中，我们引入了一种 Awareness 意识下的检索评分方法，该方法明确建模项目曝光概率，并在推理时调整检索阶段的排序。与之前的工作不同，该方法将曝光效应与参与可能性脱钩，能够在大规模推荐平台上实现公平性和参与性的可控权衡。我们通过在真实世界视频推荐系统中的在线 A/B 实验验证了该方法，结果显示独特检索项目的增加幅度达到了 25%，过度流行内容的主导性降低了 40%，同时保持了整体用户参与度水平。实验结果建立了一种可扩展且可部署的在检索阶段缓解流行性偏差的解决方案，为一种新的意识下偏差感知个性化提供了新范式。 

---
# Graph-Eq: Discovering Mathematical Equations using Graph Generative Models 

**Title (ZH)**: Graph-Eq: 使用图生成模型发现数学方程 

**Authors**: Nisal Ranasinghe, Damith Senanayake, Saman Halgamuge  

**Link**: [PDF](https://arxiv.org/pdf/2503.23617)  

**Abstract**: The ability to discover meaningful, accurate, and concise mathematical equations that describe datasets is valuable across various domains. Equations offer explicit relationships between variables, enabling deeper insights into underlying data patterns. Most existing equation discovery methods rely on genetic programming, which iteratively searches the equation space but is often slow and prone to overfitting. By representing equations as directed acyclic graphs, we leverage the use of graph neural networks to learn the underlying semantics of equations, and generate new, previously unseen equations. Although graph generative models have been shown to be successful in discovering new types of graphs in many fields, there application in discovering equations remains largely unexplored. In this work, we propose Graph-EQ, a deep graph generative model designed for efficient equation discovery. Graph-EQ uses a conditional variational autoencoder (CVAE) to learn a rich latent representation of the equation space by training it on a large corpus of equations in an unsupervised manner. Instead of directly searching the equation space, we employ Bayesian optimization to efficiently explore this learned latent space. We show that the encoder-decoder architecture of Graph-Eq is able to accurately reconstruct input equations. Moreover, we show that the learned latent representation can be sampled and decoded into valid equations, including new and previously unseen equations in the training data. Finally, we assess Graph-Eq's ability to discover equations that best fit a dataset by exploring the latent space using Bayesian optimization. Latent space exploration is done on 20 dataset with known ground-truth equations, and Graph-Eq is shown to successfully discover the grountruth equation in the majority of datasets. 

**Abstract (ZH)**: 能够在各种领域中发现有意义、准确且简洁的数学方程的能力是宝贵的。方程提供了变量之间的显式关系，有助于深入理解数据背后的模式。现有的大多数方程发现方法依赖于遗传编程，虽然可以迭代搜索方程空间，但往往速度较慢且容易过拟合。通过将方程表示为有向无环图，我们利用图神经网络来学习方程的潜在语义，并生成新的未见过的方程。尽管图生成模型在许多领域中已被证明能够成功发现新的图类型，但在发现方程方面的应用仍然鲜有探索。在这项工作中，我们提出Graph-EQ，一种用于高效方程发现的深度图生成模型。Graph-EQ使用条件变分自编码器（CVAE）通过无监督的方式训练大量方程的语料库，学习方程空间的丰富潜在表示。我们没有直接搜索方程空间，而是采用贝叶斯优化高效探索这种学习到的潜在空间。我们展示了Graph-Eq的编码器-解码器架构能够准确重建输入方程。此外，我们展示了学习到的潜在表示可以采样并解码为有效方程，包括训练数据中的新和未见过的方程。最后，我们通过使用贝叶斯优化探索潜在空间来评估Graph-Eq发现最佳拟合数据集方程的能力。在20个已知ground-truth方程的数据集上进行潜在空间探索，结果表明Graph-Eq能够在大多数数据集中成功发现ground-truth方程。 

---
# Interpretable Machine Learning in Physics: A Review 

**Title (ZH)**: 可解释的机器学习在物理学中的应用：一个综述 

**Authors**: Sebastian Johann Wetzel, Seungwoong Ha, Raban Iten, Miriam Klopotek, Ziming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23616)  

**Abstract**: Machine learning is increasingly transforming various scientific fields, enabled by advancements in computational power and access to large data sets from experiments and simulations. As artificial intelligence (AI) continues to grow in capability, these algorithms will enable many scientific discoveries beyond human capabilities. Since the primary goal of science is to understand the world around us, fully leveraging machine learning in scientific discovery requires models that are interpretable -- allowing experts to comprehend the concepts underlying machine-learned predictions. Successful interpretations increase trust in black-box methods, help reduce errors, allow for the improvement of the underlying models, enhance human-AI collaboration, and ultimately enable fully automated scientific discoveries that remain understandable to human scientists. This review examines the role of interpretability in machine learning applied to physics. We categorize different aspects of interpretability, discuss machine learning models in terms of both interpretability and performance, and explore the philosophical implications of interpretability in scientific inquiry. Additionally, we highlight recent advances in interpretable machine learning across many subfields of physics. By bridging boundaries between disciplines -- each with its own unique insights and challenges -- we aim to establish interpretable machine learning as a core research focus in science. 

**Abstract (ZH)**: 机器学习日益 transformations 各个科学领域，得益于计算能力的提升和从实验与模拟中获取的大规模数据集。随着人工智能（AI）能力的不断增长，这些算法将使许多超出人类能力范围的科学发现成为可能。鉴于科学的基本目标是理解我们周围的世界，充分利用机器学习在科学研究中的作用需要可解释的模型——使专家能够理解机器学习预测背后的概念。成功的解释增加了对黑盒方法的信任，有助于减少错误，允许提高底层模型，增强人类与AI的协作，并最终实现可为人科学家理解的完全自动化的科学发现。本文回顾了机器学习在物理学中的应用中解释性的作用。我们分类了解释性的不同方面，讨论了既具有解释性又具有高性能的机器学习模型，并探索了解释性在科学研究中的哲学含义。此外，我们还强调了物理学各个子领域的最新解释性机器学习进展。通过弥合学科之间的界限——每个学科都有其独特的见解和挑战——我们旨在将解释性机器学习确立为科学的核心研究重点。 

---
# Partial Transportability for Domain Generalization 

**Title (ZH)**: 域泛化的部分可迁移性 

**Authors**: Kasra Jalaldoust, Alexis Bellot, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2503.23605)  

**Abstract**: A fundamental task in AI is providing performance guarantees for predictions made in unseen domains. In practice, there can be substantial uncertainty about the distribution of new data, and corresponding variability in the performance of existing predictors. Building on the theory of partial identification and transportability, this paper introduces new results for bounding the value of a functional of the target distribution, such as the generalization error of a classifier, given data from source domains and assumptions about the data generating mechanisms, encoded in causal diagrams. Our contribution is to provide the first general estimation technique for transportability problems, adapting existing parameterization schemes such Neural Causal Models to encode the structural constraints necessary for cross-population inference. We demonstrate the expressiveness and consistency of this procedure and further propose a gradient-based optimization scheme for making scalable inferences in practice. Our results are corroborated with experiments. 

**Abstract (ZH)**: AI中的一个基本任务是为未见领域中的预测提供性能保证。基于部分识别和可传输性的理论，本文引入了在给定源领域数据和数据生成机制假设（编码在因果图中）的情况下，用于界定目标分布函数值（例如分类器的泛化误差）的新结果。我们的贡献是提供了首个通用的传输问题估算技术，将现有的参数化方案，如神经因果模型，适应性地编码用于跨人群推理的结构约束。我们展示了该程序的表述能力和一致性，并进一步提出了一种基于梯度的优化方案，以在实践中进行可扩展的推断。我们的结果通过实验得到了验证。 

---
# Addressing Model Overcomplexity in Drug-Drug Interaction Prediction With Molecular Fingerprints 

**Title (ZH)**: 基于分子指纹图谱解决药物-药物相互作用预测中的模型过拟合问题 

**Authors**: Manel Gil-Sorribes, Alexis Molina  

**Link**: [PDF](https://arxiv.org/pdf/2503.23550)  

**Abstract**: Accurately predicting drug-drug interactions (DDIs) is crucial for pharmaceutical research and clinical safety. Recent deep learning models often suffer from high computational costs and limited generalization across datasets. In this study, we investigate a simpler yet effective approach using molecular representations such as Morgan fingerprints (MFPS), graph-based embeddings from graph convolutional networks (GCNs), and transformer-derived embeddings from MoLFormer integrated into a straightforward neural network. We benchmark our implementation on DrugBank DDI splits and a drug-drug affinity (DDA) dataset from the Food and Drug Administration. MFPS along with MoLFormer and GCN representations achieve competitive performance across tasks, even in the more challenging leak-proof split, highlighting the sufficiency of simple molecular representations. Moreover, we are able to identify key molecular motifs and structural patterns relevant to drug interactions via gradient-based analyses using the representations under study. Despite these results, dataset limitations such as insufficient chemical diversity, limited dataset size, and inconsistent labeling impact robust evaluation and challenge the need for more complex approaches. Our work provides a meaningful baseline and emphasizes the need for better dataset curation and progressive complexity scaling. 

**Abstract (ZH)**: 准确预测药物-药物相互作用（-DDIs）对于制药研究和临床安全性至关重要。尽管最近的深度学习模型常常面临高计算成本和跨数据集限制泛化的挑战，我们在本研究中探讨了一种更为简单有效的方法，使用诸如摩根指纹（MFPS）、图卷积网络（GCNs）的图基嵌入以及MoLFormer衍生的变压器嵌入，并将其集成到一个简单的神经网络中。我们在DrugBank DDI分割和食品和药物管理局的药物-药物亲和力（DDA）数据集上对标了我们的实现。MFPS与MoLFormer和GCN表示在各项任务中均表现出竞争力，即使在更具挑战性的密封泄漏分割中也是如此，这突显了简单分子表示的充分性。此外，我们还能够通过基于梯度的分析识别出与药物相互作用相关的关键分子模式和结构特征。尽管取得了这些结果，但由于数据集限制，如化学多样性不足、数据集规模有限和标签不一致，这影响了稳健的评估，并挑战了更复杂方法的必要性。我们的工作提供了有意义的基线，并强调了更好地数据集整理和逐步复杂性的必要性。 

---
# A Survey on Unlearnable Data 

**Title (ZH)**: 不可学习数据研究综述 

**Authors**: Jiahao Li, Yiqiang Chen, Yunbing Xing, Yang Gu, Xiangyuan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.23536)  

**Abstract**: Unlearnable data (ULD) has emerged as an innovative defense technique to prevent machine learning models from learning meaningful patterns from specific data, thus protecting data privacy and security. By introducing perturbations to the training data, ULD degrades model performance, making it difficult for unauthorized models to extract useful representations. Despite the growing significance of ULD, existing surveys predominantly focus on related fields, such as adversarial attacks and machine unlearning, with little attention given to ULD as an independent area of study. This survey fills that gap by offering a comprehensive review of ULD, examining unlearnable data generation methods, public benchmarks, evaluation metrics, theoretical foundations and practical applications. We compare and contrast different ULD approaches, analyzing their strengths, limitations, and trade-offs related to unlearnability, imperceptibility, efficiency and robustness. Moreover, we discuss key challenges, such as balancing perturbation imperceptibility with model degradation and the computational complexity of ULD generation. Finally, we highlight promising future research directions to advance the effectiveness and applicability of ULD, underscoring its potential to become a crucial tool in the evolving landscape of data protection in machine learning. 

**Abstract (ZH)**: 无法学习的数据（ULD）作为一种创新的防御技术，通过阻止机器学习模型从特定数据中学习有意义的模式，从而保护数据隐私和安全。通过向训练数据引入扰动，ULD降低模型性能，使未授权模型难以提取有用表示。尽管ULD的重要性日益增长，现有的综述主要集中在相关领域，如对抗攻击和机器遗忘，对ULD作为一个独立的研究领域关注较少。本文综述填补了这一空白，提供了ULD的全面综述，探讨了无法学习数据生成方法、公开基准、评估指标、理论基础和实际应用。我们对比了不同的ULD方法，分析了它们在不可学习性、不可感知性、效率和鲁棒性方面的优势、局限性和权衡。此外，我们讨论了关键挑战，如平衡扰动的不可感知性与模型性能下降，以及ULD生成的计算复杂性。最后，我们指出了未来有前景的研究方向，以提高ULD的有效性和适用性，突显其在机器学习数据保护演进 landscape 中的潜在重要性。 

---
# Buffer is All You Need: Defending Federated Learning against Backdoor Attacks under Non-iids via Buffering 

**Title (ZH)**: Buffer 是你需要的：通过缓冲防御非-iids 情况下的联邦学习后门攻击 

**Authors**: Xingyu Lyu, Ning Wang, Yang Xiao, Shixiong Li, Tao Li, Danjue Chen, Yimin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23511)  

**Abstract**: Federated Learning (FL) is a popular paradigm enabling clients to jointly train a global model without sharing raw data. However, FL is known to be vulnerable towards backdoor attacks due to its distributed nature. As participants, attackers can upload model updates that effectively compromise FL. What's worse, existing defenses are mostly designed under independent-and-identically-distributed (iid) settings, hence neglecting the fundamental non-iid characteristic of FL. Here we propose FLBuff for tackling backdoor attacks even under non-iids. The main challenge for such defenses is that non-iids bring benign and malicious updates closer, hence harder to separate. FLBuff is inspired by our insight that non-iids can be modeled as omni-directional expansion in representation space while backdoor attacks as uni-directional. This leads to the key design of FLBuff, i.e., a supervised-contrastive-learning model extracting penultimate-layer representations to create a large in-between buffer layer. Comprehensive evaluations demonstrate that FLBuff consistently outperforms state-of-the-art defenses. 

**Abstract (ZH)**: Federated Learning (FL)是一种流行的 paradigm，允许多个客户端联合训练全球模型而不共享原始数据。然而，FL由于其分布式特性，容易受到后门攻击。作为参与者，攻击者可以上传有效破坏FL的模型更新。更糟糕的是，现有防御大多是在独立且同分布(iid)设置下设计的，因此忽略了FL的基本非-iid特性。在这里，我们提出了FLBuff以应对非-iid条件下的后门攻击。此类防御的主要挑战在于非-iid使得良性更新和恶意更新更加接近，难以区分。FLBuff的灵感来源于我们对非-iid可以被视为表示空间的全方位扩展而后门攻击则是单向性的这一洞见。这促使FLBuff的关键设计是一个监督对比学习模型，从倒数第二层提取表示以创建一个大的中间缓冲层。全面的评估表明，FLBuff在各种情况下持续优于最先进的防御方法。 

---
# POINT$^{2}$: A Polymer Informatics Training and Testing Database 

**Title (ZH)**: POINT$^{2}$: 聚合物信息学训练与测试数据库 

**Authors**: Jiaxin Xu, Gang Liu, Ruilan Guo, Meng Jiang, Tengfei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.23491)  

**Abstract**: The advancement of polymer informatics has been significantly propelled by the integration of machine learning (ML) techniques, enabling the rapid prediction of polymer properties and expediting the discovery of high-performance polymeric materials. However, the field lacks a standardized workflow that encompasses prediction accuracy, uncertainty quantification, ML interpretability, and polymer synthesizability. In this study, we introduce POINT$^{2}$ (POlymer INformatics Training and Testing), a comprehensive benchmark database and protocol designed to address these critical challenges. Leveraging the existing labeled datasets and the unlabeled PI1M dataset, a collection of approximately one million virtual polymers generated via a recurrent neural network trained on the realistic polymers, we develop an ensemble of ML models, including Quantile Random Forests, Multilayer Perceptrons with dropout, Graph Neural Networks, and pretrained large language models. These models are coupled with diverse polymer representations such as Morgan, MACCS, RDKit, Topological, Atom Pair fingerprints, and graph-based descriptors to achieve property predictions, uncertainty estimations, model interpretability, and template-based polymerization synthesizability across a spectrum of properties, including gas permeability, thermal conductivity, glass transition temperature, melting temperature, fractional free volume, and density. The POINT$^{2}$ database can serve as a valuable resource for the polymer informatics community for polymer discovery and optimization. 

**Abstract (ZH)**: 聚合物信息化的进步得益于机器学习技术的整合，这使得能够快速预测聚合物性能并加速高性能聚合物材料的发现。然而，该领域缺乏一个涵盖预测准确度、不确定性量化、机器学习可解释性和聚合物合成性的标准化工作流程。在本研究中，我们引入了POINT$^{2}$（聚合物信息化训练与测试），一个综合基准数据库和协议，旨在解决这些关键挑战。利用现有的标记数据集和未标记的PI1M数据集（通过在现实聚合物上训练的递归神经网络生成的约一百万种虚拟聚合物集合），我们开发了一组机器学习模型，包括分位数随机森林、具有丢弃的多层感知机、图神经网络和预训练的大语言模型。这些模型与多种聚合物表示相结合，如Morgan、MACCS、RDKit、拓扑、原子对指纹和基于图的描述符，实现了从气体渗透性、热导率、玻璃转变温度、熔点、自由体积分数和密度等一系列性质的性能预测、不确定性估计、模型可解释性和模板导向的聚合物聚合可合成性。POINT$^{2}$数据库可作为聚合物信息化社区进行聚合物发现和优化的宝贵资源。 

---
# Codehacks: A Dataset of Adversarial Tests for Competitive Programming Problems Obtained from Codeforces 

**Title (ZH)**: Codehacks：来自Codeforces的对抗性测试数据集，用于 Competitive Programming 问题 

**Authors**: Max Hort, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23466)  

**Abstract**: Software is used in critical applications in our day-to-day life and it is important to ensure its correctness. One popular approach to assess correctness is to evaluate software on tests. If a test fails, it indicates a fault in the software under test; if all tests pass correctly, one may assume that the software is correct. However, the reliability of these results depends on the test suite considered, and there is a risk of false negatives (i.e. software that passes all available tests but contains bugs because some cases are not tested). Therefore, it is important to consider error-inducing test cases when evaluating software.
To support data-driven creation of such a test-suite, which is especially of interest for testing software synthesized from large language models, we curate a dataset (Codehacks) of programming problems together with corresponding error-inducing test cases (i.e., "hacks"). This dataset is collected from the wild, in particular, from the Codeforces online judge platform. The dataset comprises 288,617 hacks for 5,578 programming problems, each with a natural language description, as well as the source code for 2,196 submitted solutions to these problems that can be broken with their corresponding hacks.
Keywords: competitive programming, language model, dataset 

**Abstract (ZH)**: 软件在我们日常生活中被用于关键应用，确保其正确性很重要。常用的方法是通过测试评估软件的正确性。如果测试失败，说明被测试软件存在故障；如果所有测试都通过，则可以假设软件是正确的。然而，这些结果的可靠性取决于所考虑的测试集，存在因未测试某些情况而导致误判（即软件通过所有可用测试但包含未测试情况导致的错误）的风险。因此，在评估软件时考虑引入错误的测试案例很重要。
为了支持这种测试套件的数据驱动创建，特别是对于从大型语言模型合成的软件测试特别感兴趣，我们收集了一个包含编程问题及其相应的引入错误的测试案例（即“破解”）的数据集（Codehacks）。该数据集来源于Codeforces在线裁判平台等野生环境。数据集包含288,617个破解案例，针对5,578个编程问题，每个问题都有自然语言描述和2,196个提交的解决方案的源代码，这些解决方案可以通过相应的破解案例来破坏。关键词：竞技编程，语言模型，数据集。 

---
# What Makes an Evaluation Useful? Common Pitfalls and Best Practices 

**Title (ZH)**: 什么是有效的评价？常见的陷阱与最佳实践 

**Authors**: Gil Gekker, Meirav Segal, Dan Lahav, Omer Nevo  

**Link**: [PDF](https://arxiv.org/pdf/2503.23424)  

**Abstract**: Following the rapid increase in Artificial Intelligence (AI) capabilities in recent years, the AI community has voiced concerns regarding possible safety risks. To support decision-making on the safe use and development of AI systems, there is a growing need for high-quality evaluations of dangerous model capabilities. While several attempts to provide such evaluations have been made, a clear definition of what constitutes a "good evaluation" has yet to be agreed upon. In this practitioners' perspective paper, we present a set of best practices for safety evaluations, drawing on prior work in model evaluation and illustrated through cybersecurity examples. We first discuss the steps of the initial thought process, which connects threat modeling to evaluation design. Then, we provide the characteristics and parameters that make an evaluation useful. Finally, we address additional considerations as we move from building specific evaluations to building a full and comprehensive evaluation suite. 

**Abstract (ZH)**: 随着近年来人工智能（AI）能力的迅速提升，AI社区表达了对其潜在安全风险的担忧。为了支持AI系统的安全使用和开发的决策制定，高质量的危险模型能力评估需求日益增长。尽管已经做出了若干尝试来提供这样的评估，但对于什么是“好的评估”仍缺乏明确定义。在本文中，我们基于先前的工作，通过网络安全领域的实例，介绍了一套安全评估的最佳实践。首先，我们讨论了初始思维过程中的步骤，将威胁建模与评估设计联系起来。然后，我们提供了使评估有用的特点和参数。最后，我们在从构建特定评估到构建全面评估套件的过程中，讨论了其他需要考虑的因素。 

---
# From Content Creation to Citation Inflation: A GenAI Case Study 

**Title (ZH)**: 从内容创作到引用膨胀：一个GenAI案例研究 

**Authors**: Haitham S. Al-Sinani, Chris J. Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2503.23414)  

**Abstract**: This paper investigates the presence and impact of questionable, AI-generated academic papers on widely used preprint repositories, with a focus on their role in citation manipulation. Motivated by suspicious patterns observed in publications related to our ongoing research on GenAI-enhanced cybersecurity, we identify clusters of questionable papers and profiles. These papers frequently exhibit minimal technical content, repetitive structure, unverifiable authorship, and mutually reinforcing citation patterns among a recurring set of authors. To assess the feasibility and implications of such practices, we conduct a controlled experiment: generating a fake paper using GenAI, embedding citations to suspected questionable publications, and uploading it to one such repository (ResearchGate). Our findings demonstrate that such papers can bypass platform checks, remain publicly accessible, and contribute to inflating citation metrics like the H-index and i10-index. We present a detailed analysis of the mechanisms involved, highlight systemic weaknesses in content moderation, and offer recommendations for improving platform accountability and preserving academic integrity in the age of GenAI. 

**Abstract (ZH)**: 本文调查了可疑的、由AI生成的学术论文在广泛使用的预印本 repositories 中的存在及其影响，重点关注这些论文在引文操纵中的角色。受我们对增强型生成AI网络安全研究中发现的可疑模式的启发，我们识别了可疑论文和作者群体。这些论文经常表现出技术内容少、结构重复、作者身份难以验证以及作者之间的互相支持的引文模式。为了评估此类做法的可行性和影响，我们进行了一个受控实验：使用生成AI生成一篇虚假论文，嵌入疑似可疑论文的引文，并将其上传到一个这样的仓库（ResearchGate）。我们的研究发现这些论文可以绕过平台检查，保持公开访问，并有助于夸大如H指数和i10指数等引文指标。我们详细分析了涉及的机制，突出了内容审核中的系统性薄弱环节，并提出了改进平台责任和在生成AI时代维护学术诚信的建议。 

---
# Diffusion Meets Few-shot Class Incremental Learning 

**Title (ZH)**: 扩散模型 Meet 少量-shot 类增量学习 

**Authors**: Junsu Kim, Yunhoe Ku, Dongyoon Han, Seungryul Baek  

**Link**: [PDF](https://arxiv.org/pdf/2503.23402)  

**Abstract**: Few-shot class-incremental learning (FSCIL) is challenging due to extremely limited training data; while aiming to reduce catastrophic forgetting and learn new information. We propose Diffusion-FSCIL, a novel approach that employs a text-to-image diffusion model as a frozen backbone. Our conjecture is that FSCIL can be tackled using a large generative model's capabilities benefiting from 1) generation ability via large-scale pre-training; 2) multi-scale representation; 3) representational flexibility through the text encoder. To maximize the representation capability, we propose to extract multiple complementary diffusion features to play roles as latent replay with slight support from feature distillation for preventing generative biases. Our framework realizes efficiency through 1) using a frozen backbone; 2) minimal trainable components; 3) batch processing of multiple feature extractions. Extensive experiments on CUB-200, miniImageNet, and CIFAR-100 show that Diffusion-FSCIL surpasses state-of-the-art methods, preserving performance on previously learned classes and adapting effectively to new ones. 

**Abstract (ZH)**: 基于文本到图像扩散模型的少样本类增量学习（Diffusion-FSCIL） 

---
# Spatiotemporal Learning of Brain Dynamics from fMRI Using Frequency-Specific Multi-Band Attention for Cognitive and Psychiatric Applications 

**Title (ZH)**: 基于频率特定多频带注意力的fMRI脑动态时空学习在认知和精神卫生应用中 

**Authors**: Sangyoon Bae, Junbeom Kwon, Shinjae Yoo, Jiook Cha  

**Link**: [PDF](https://arxiv.org/pdf/2503.23394)  

**Abstract**: Understanding how the brain's complex nonlinear dynamics give rise to adaptive cognition and behavior is a central challenge in neuroscience. These dynamics exhibit scale-free and multifractal properties, influencing the reconfiguration of neural networks. However, conventional neuroimaging models are constrained by linear and stationary assumptions, limiting their ability to capture these processes. Transformer-based architectures, known for capturing long-range dependencies, align well with the brain's hierarchical and temporal organization. We introduce Multi-Band Brain Net (MBBN), a transformer-based framework that models frequency-specific spatiotemporal brain dynamics from fMRI by integrating scale-free network principles with frequency-resolved multi-band self-attention. Trained on three large-scale neuroimaging cohorts (UK Biobank, ABCD, ABIDE) totaling 45,951 individuals, MBBN reveals previously undetectable frequency-dependent network interactions, shedding light on connectivity disruptions in psychiatric conditions (ADHD, ASD, depression). This validation shows robust generalizability and highlights core neural principles conserved across populations. MBBN achieves up to 30.59% higher predictive accuracy than state-of-the-art methods, demonstrating the advantage of frequency-informed spatiotemporal modeling in capturing latent neural computations. MBBN's interpretability uncovers novel frequency-specific biomarkers for neurodevelopmental disorders, providing insights into the hierarchical organization of brain function. By offering an interpretable framework for spatiotemporal learning, MBBN provides insights into how neural computations underpin cognitive function and psychiatric vulnerability, with implications for brain decoding, cognitive neuroscience, and precision psychiatry. 

**Abstract (ZH)**: 理解大脑复杂非线性动态如何产生适应性认知和行为是神经科学中的一个核心挑战。这些动态表现出无标度和多分形特性，影响神经网络的重构。然而，传统神经成像模型受限于线性和稳态假设，限制了它们捕捉这些过程的能力。基于变换器的架构因其能捕捉长程依赖关系而与大脑的分层和时间组织相契合。我们提出了多频带脑网络（MBBN）框架，该框架通过结合无标度网络原理和频带分辨率多频带自注意力机制，从功能性磁共振成像（fMRI）中建模频率特异性的时空脑动态。MBBN在三个大规模神经成像队列（UK Biobank, ABCD, ABIDE）的45,951个体上进行了训练，揭示了以前未检测到的频率依赖性网络交互，阐明了精神疾病（ADHD, ASD, 抑郁）中的连接性障碍。这一验证显示了其稳健的泛化能力和跨人群保守的核心神经原理。MBBN的预测准确率最高可比最先进的方法提高30.59%，证明了基于频率的时空建模在捕捉潜在神经计算方面的优势。MBBN的可解释性揭示了神经发育障碍的新频率特异性生物标志物，提供了有关大脑功能分层组织的见解。通过提供可解释的时空学习框架，MBBN为理解神经计算如何支撑认知功能和精神疾病易感性提供了洞察，对于脑解码、认知神经科学和精准精神病学具有重要影响。 

---
# Pareto Continual Learning: Preference-Conditioned Learning and Adaption for Dynamic Stability-Plasticity Trade-off 

**Title (ZH)**: 帕累托持续学习：基于偏好条件化学习与适应的动态稳定-可塑性权衡博弈 

**Authors**: Song Lai, Zhe Zhao, Fei Zhu, Xi Lin, Qingfu Zhang, Gaofeng Meng  

**Link**: [PDF](https://arxiv.org/pdf/2503.23390)  

**Abstract**: Continual learning aims to learn multiple tasks sequentially. A key challenge in continual learning is balancing between two objectives: retaining knowledge from old tasks (stability) and adapting to new tasks (plasticity). Experience replay methods, which store and replay past data alongside new data, have become a widely adopted approach to mitigate catastrophic forgetting. However, these methods neglect the dynamic nature of the stability-plasticity trade-off and aim to find a fixed and unchanging balance, resulting in suboptimal adaptation during training and inference. In this paper, we propose Pareto Continual Learning (ParetoCL), a novel framework that reformulates the stability-plasticity trade-off in continual learning as a multi-objective optimization (MOO) problem. ParetoCL introduces a preference-conditioned model to efficiently learn a set of Pareto optimal solutions representing different trade-offs and enables dynamic adaptation during inference. From a generalization perspective, ParetoCL can be seen as an objective augmentation approach that learns from different objective combinations of stability and plasticity. Extensive experiments across multiple datasets and settings demonstrate that ParetoCL outperforms state-of-the-art methods and adapts to diverse continual learning scenarios. 

**Abstract (ZH)**: 持续学习旨在顺序学习多个任务。持续学习中的一个关键挑战是在保持旧任务知识（稳定性）和适应新任务（可塑性）之间取得平衡。经验重播方法通过存储和重播过去的数据与新数据一起，已成为减轻灾难性遗忘的广泛采用方法。然而，这些方法忽视了稳定性-可塑性权衡的动态性质，并试图找到一个固定的、不变的平衡，导致在训练和推理过程中适应能力不足。在本文中，我们提出了一种新颖框架Pareto持续学习（ParetoCL），将持续学习中的稳定性-可塑性权衡重新表述为多目标优化（MOO）问题。ParetoCL引入了一种偏好条件下的模型，能够高效地学习代表不同权衡的一组Pareto最优解，并在推理过程中实现动态适应。从泛化角度来看，ParetoCL可以被视为一种目标增强方法，能够从稳定性与可塑性不同目标组合中学习。实验结果表明，ParetoCL在多个数据集和设置中优于现有方法，并能够适应多种持续学习场景。 

---
# COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation 

**Title (ZH)**: COSMIC: 基于 clique 的语义多空间集成以实现鲁棒的 CLIP 测试时适应 

**Authors**: Fanding Huang, Jingyan Jiang, Qinting Jiang, Hebei Li, Faisal Nadeem Khan, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23388)  

**Abstract**: Recent vision-language models (VLMs) face significant challenges in test-time adaptation to novel domains. While cache-based methods show promise by leveraging historical information, they struggle with both caching unreliable feature-label pairs and indiscriminately using single-class information during querying, significantly compromising adaptation accuracy. To address these limitations, we propose COSMIC (Clique-Oriented Semantic Multi-space Integration for CLIP), a robust test-time adaptation framework that enhances adaptability through multi-granular, cross-modal semantic caching and graph-based querying mechanisms. Our framework introduces two key innovations: Dual Semantics Graph (DSG) and Clique Guided Hyper-class (CGH). The Dual Semantics Graph constructs complementary semantic spaces by incorporating textual features, coarse-grained CLIP features, and fine-grained DINOv2 features to capture rich semantic relationships. Building upon these dual graphs, the Clique Guided Hyper-class component leverages structured class relationships to enhance prediction robustness through correlated class selection. Extensive experiments demonstrate COSMIC's superior performance across multiple benchmarks, achieving significant improvements over state-of-the-art methods: 15.81% gain on out-of-distribution tasks and 5.33% on cross-domain generation with CLIP RN-50. Code is available at this http URL. 

**Abstract (ZH)**: Recent vision-language模型（VLMs）在测试时适应新型领域方面面临着显著挑战。尽管基于缓存的方法通过利用历史信息展现了潜力，但在缓存不可靠的特征-标签对以及在查询时不分场合地使用单类信息方面存在局限，严重影响了适应准确性。为解决这些局限性，我们提出了COSMIC（基于聚类的语义多空间集成用于CLIP），这是一种通过多粒度、跨模态语义缓存和图查询机制来增强适应性的稳健测试时适应框架。我们的框架引入了两个关键创新：双语义图（DSG）和聚类引导的超类（CGH）。双语义图通过整合文本特征、粗粒度CLIP特征和细粒度DINOv2特征来构建互补的语义空间，以捕获丰富的语义关系。在此基础上，聚类引导的超类组件利用结构化类关系，通过相关类的选择来增强预测的稳健性。广泛的经验表明，COSMIC在多个基准测试中表现出优越性能，相对于最先进的方法，在分布外任务上取得了15.81%的增益，在使用CLIP RN-50进行跨域生成任务上取得了5.33%的增益。代码详见此网址。 

---
# KernelDNA: Dynamic Kernel Sharing via Decoupled Naive Adapters 

**Title (ZH)**: KernelDNA: 动态核共享通过解耦天真适配器 

**Authors**: Haiduo Huang, Yadong Zhang, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.23379)  

**Abstract**: Dynamic convolution enhances model capacity by adaptively combining multiple kernels, yet faces critical trade-offs: prior works either (1) incur significant parameter overhead by scaling kernel numbers linearly, (2) compromise inference speed through complex kernel interactions, or (3) struggle to jointly optimize dynamic attention and static kernels. We also observe that pre-trained Convolutional Neural Networks (CNNs) exhibit inter-layer redundancy akin to that in Large Language Models (LLMs). Specifically, dense convolutional layers can be efficiently replaced by derived ``child" layers generated from a shared ``parent" convolutional kernel through an adapter.
To address these limitations and implement the weight-sharing mechanism, we propose a lightweight convolution kernel plug-in, named KernelDNA. It decouples kernel adaptation into input-dependent dynamic routing and pre-trained static modulation, ensuring both parameter efficiency and hardware-friendly inference. Unlike existing dynamic convolutions that expand parameters via multi-kernel ensembles, our method leverages cross-layer weight sharing and adapter-based modulation, enabling dynamic kernel specialization without altering the standard convolution structure. This design preserves the native computational efficiency of standard convolutions while enhancing representation power through input-adaptive kernel adjustments. Experiments on image classification and dense prediction tasks demonstrate that KernelDNA achieves state-of-the-art accuracy-efficiency balance among dynamic convolution variants. Our codes are available at this https URL. 

**Abstract (ZH)**: 动态卷积通过适应性结合多个核增强模型容量，但面临关键权衡：现有工作要么（1）通过线性扩展核数量引起显著的参数开销，要么（2）通过复杂的核交互牺牲推理速度，要么（3）难以同时优化动态注意力和静态核。我们还观察到预训练的卷积神经网络（CNNs）在层间冗余方面类似于大型语言模型（LLMs）。具体而言，密集的卷积层可以通过共享“父”卷积核生成的“子”层进行高效替换。

为了解决这些限制并实现权重共享机制，我们提出了一种轻量级卷积核插件，名为KernelDNA。它将核适应解耦为输入相关的动态路由和预训练的静态调制，确保参数效率和硬件友好的推理。与现有通过多核组合扩展参数的动态卷积不同，我们的方法利用跨层权重共享和基于适配器的调制，能够在不改变标准卷积结构的情况下实现动态核专业化。这种设计保持了标准卷积的原生计算效率，同时通过输入自适应的核调整增强表示能力。在图像分类和密集预测任务上的实验表明，KernelDNA在动态卷积变种中实现了最佳的准确性和效率平衡。我们的代码可在以下链接获取。 

---
# Object Isolated Attention for Consistent Story Visualization 

**Title (ZH)**: 物体隔离注意力for一致的故事可视化 

**Authors**: Xiangyang Luo, Junhao Cheng, Yifan Xie, Xin Zhang, Tao Feng, Zhou Liu, Fei Ma, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23353)  

**Abstract**: Open-ended story visualization is a challenging task that involves generating coherent image sequences from a given storyline. One of the main difficulties is maintaining character consistency while creating natural and contextually fitting scenes--an area where many existing methods struggle. In this paper, we propose an enhanced Transformer module that uses separate self attention and cross attention mechanisms, leveraging prior knowledge from pre-trained diffusion models to ensure logical scene creation. The isolated self attention mechanism improves character consistency by refining attention maps to reduce focus on irrelevant areas and highlight key features of the same character. Meanwhile, the isolated cross attention mechanism independently processes each character's features, avoiding feature fusion and further strengthening consistency. Notably, our method is training-free, allowing the continuous generation of new characters and storylines without re-tuning. Both qualitative and quantitative evaluations show that our approach outperforms current methods, demonstrating its effectiveness. 

**Abstract (ZH)**: 开放式故事可视化是一个具有挑战性的任务，涉及从给定的故事线生成连贯的图像序列。一个主要的难点是同时创建自然且符合情境的画面时保持角色一致性——这是一个许多现有方法都难以解决的问题。在本文中，我们提出了一种增强的Transformer模块，该模块利用独立的自我注意力机制和交叉注意力机制，并结合预训练扩散模型的先验知识，以确保逻辑场景的创建。独立的自我注意力机制通过细化注意力图来减少对不相关区域的关注，突出显示相同角色的关键特征，从而提高角色一致性。与此同时，独立的交叉注意力机制分别处理每个角色的特征，避免特征融合，进一步加强一致性。值得注意的是，我们的方法无需训练，可以连续生成新的角色和故事情节而无需重新调优。定性和定量评估均表明，我们的方法优于现有方法，证明了其有效性。 

---
# SalesRLAgent: A Reinforcement Learning Approach for Real-Time Sales Conversion Prediction and Optimization 

**Title (ZH)**: SalesRLAgent：一种实时销售转化预测与优化的 reinforcement learning 方法 

**Authors**: Nandakishor M  

**Link**: [PDF](https://arxiv.org/pdf/2503.23303)  

**Abstract**: Current approaches to sales conversation analysis and conversion prediction typically rely on Large Language Models (LLMs) combined with basic retrieval augmented generation (RAG). These systems, while capable of answering questions, fail to accurately predict conversion probability or provide strategic guidance in real time. In this paper, we present SalesRLAgent, a novel framework leveraging specialized reinforcement learning to predict conversion probability throughout sales conversations. Unlike systems from this http URL, Mendable, Inkeep, and others that primarily use off-the-shelf LLMs for content generation, our approach treats conversion prediction as a sequential decision problem, training on synthetic data generated using GPT-4O to develop a specialized probability estimation model. Our system incorporates Azure OpenAI embeddings (3072 dimensions), turn-by-turn state tracking, and meta-learning capabilities to understand its own knowledge boundaries. Evaluations demonstrate that SalesRLAgent achieves 96.7% accuracy in conversion prediction, outperforming LLM-only approaches by 34.7% while offering significantly faster inference (85ms vs 3450ms for GPT-4). Furthermore, integration with existing sales platforms shows a 43.2% increase in conversion rates when representatives utilize our system's real-time guidance. SalesRLAgent represents a fundamental shift from content generation to strategic sales intelligence, providing moment-by-moment conversion probability estimation with actionable insights for sales professionals. 

**Abstract (ZH)**: 基于强化学习的销售对话转换概率预测框架：SalesRLAgent 

---
# Two Heads Are Better than One: Model-Weight and Latent-Space Analysis for Federated Learning on Non-iid Data against Poisoning Attacks 

**Title (ZH)**: 一分为二更好：针对非iid数据下的中毒攻击的联邦学习中模型权重和潜在空间分析 

**Authors**: Xingyu Lyu, Ning Wang, Yang Xiao, Shixiong Li, Tao Li, Danjue Chen, Yimin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23288)  

**Abstract**: Federated Learning is a popular paradigm that enables remote clients to jointly train a global model without sharing their raw data. However, FL has been shown to be vulnerable towards model poisoning attacks due to its distributed nature. Particularly, attackers acting as participants can upload arbitrary model updates that effectively compromise the global model of FL. While extensive research has been focusing on fighting against these attacks, we find that most of them assume data at remote clients are under iid while in practice they are inevitably non-iid. Our benchmark evaluations reveal that existing defenses generally fail to live up to their reputation when applied to various non-iid scenarios. In this paper, we propose a novel approach, GeminiGuard, that aims to address such a significant gap. We design GeminiGuard to be lightweight, versatile, and unsupervised so that it aligns well with the practical requirements of deploying such defenses. The key challenge from non-iids is that they make benign model updates look more similar to malicious ones. GeminiGuard is mainly built on two fundamental observations: (1) existing defenses based on either model-weight analysis or latent-space analysis face limitations in covering different MPAs and non-iid scenarios, and (2) model-weight and latent-space analysis are sufficiently different yet potentially complementary methods as MPA defenses. We hence incorporate a novel model-weight analysis component as well as a custom latent-space analysis component in GeminiGuard, aiming to further enhance its defense performance. We conduct extensive experiments to evaluate our defense across various settings, demonstrating its effectiveness in countering multiple types of untargeted and targeted MPAs, including adaptive ones. Our comprehensive evaluations show that GeminiGuard consistently outperforms SOTA defenses under various settings. 

**Abstract (ZH)**: 联邦学习是一种流行的范式， enabling远程客户端联合训练全局模型而不共享其原始数据。然而，由于其分布式性质，联邦学习已被证明对模型中毒攻击较为脆弱。尤其是，充当参与者的攻击者可以上传任意模型更新，从而有效破坏联邦学习的全局模型。尽管已有大量研究致力于对抗这些攻击，但我们发现，它们大多假设远程客户端的数据是 iid 的，而在实践中，这些数据不可避免地是非 iid 的。我们的基准评估表明，现有防御措施在应用于各种非 iid 场景时通常未能达到其预期效果。在本文中，我们提出了一种名为 GeminiGuard 的新型方法，旨在解决这一重大差距。我们设计 GeminiGuard 使其轻量级、通用且无监督，从而与其部署所需的实用要求相契合。非 iid 带来的关键挑战是，它们使良性模型更新看起来更接近恶意更新。GeminiGuard 主要基于两个基本观察：（1）基于模型权重分析或潜在空间分析的现有防御措施在覆盖不同的 MPA 和非 iid 场景方面存在局限性；（2）模型权重分析和潜在空间分析尽管足够不同但可能具备互补性，可作为 MPA 防御方法。因此，我们将在 GeminiGuard 中加入一个新颖的模型权重分析组件以及一个自定义的潜在空间分析组件，旨在进一步增强其防御性能。我们进行了广泛的实验以评估我们的防御措施在各种设置下的效果，证明其在对抗多种未针对和针对的 MPA 方面（包括自适应 MPA）的有效性。我们全面的评估表明，在各种设置下，GeminiGuard 始终优于当前最佳防御措施。 

---
# Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions 

**Title (ZH)**: 模型上下文协议（MCP）：概览、安全威胁与未来研究方向 

**Authors**: Xinyi Hou, Yanjie Zhao, Shenao Wang, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23278)  

**Abstract**: The Model Context Protocol (MCP) is a standardized interface designed to enable seamless interaction between AI models and external tools and resources, breaking down data silos and facilitating interoperability across diverse systems. This paper provides a comprehensive overview of MCP, focusing on its core components, workflow, and the lifecycle of MCP servers, which consists of three key phases: creation, operation, and update. We analyze the security and privacy risks associated with each phase and propose strategies to mitigate potential threats. The paper also examines the current MCP landscape, including its adoption by industry leaders and various use cases, as well as the tools and platforms supporting its integration. We explore future directions for MCP, highlighting the challenges and opportunities that will influence its adoption and evolution within the broader AI ecosystem. Finally, we offer recommendations for MCP stakeholders to ensure its secure and sustainable development as the AI landscape continues to evolve. 

**Abstract (ZH)**: MCP模型上下文协议：一种用于实现AI模型与外部工具和资源无缝交互的标准接口，打破数据孤岛，促进跨异构系统互操作性的全面综述。 

---
# RECALL-MM: A Multimodal Dataset of Consumer Product Recalls for Risk Analysis using Computational Methods and Large Language Models 

**Title (ZH)**: RECALL-MM：用于风险分析的多模态消费品召回数据集及计算方法和大规模语言模型的应用 

**Authors**: Diana Bolanos, Mohammadmehdi Ataei, Daniele Grandi, Kosa Goucher-Lambert  

**Link**: [PDF](https://arxiv.org/pdf/2503.23213)  

**Abstract**: Product recalls provide valuable insights into potential risks and hazards within the engineering design process, yet their full potential remains underutilized. In this study, we curate data from the United States Consumer Product Safety Commission (CPSC) recalls database to develop a multimodal dataset, RECALL-MM, that informs data-driven risk assessment using historical information, and augment it using generative methods. Patterns in the dataset highlight specific areas where improved safety measures could have significant impact. We extend our analysis by demonstrating interactive clustering maps that embed all recalls into a shared latent space based on recall descriptions and product names. Leveraging these data-driven tools, we explore three case studies to demonstrate the dataset's utility in identifying product risks and guiding safer design decisions. The first two case studies illustrate how designers can visualize patterns across recalled products and situate new product ideas within the broader recall landscape to proactively anticipate hazards. In the third case study, we extend our approach by employing a large language model (LLM) to predict potential hazards based solely on product images. This demonstrates the model's ability to leverage visual context to identify risk factors, revealing strong alignment with historical recall data across many hazard categories. However, the analysis also highlights areas where hazard prediction remains challenging, underscoring the importance of risk awareness throughout the design process. Collectively, this work aims to bridge the gap between historical recall data and future product safety, presenting a scalable, data-driven approach to safer engineering design. 

**Abstract (ZH)**: 产品召回提供了工程设计过程中潜在风险和隐患的重要洞见，但其潜在价值尚未充分利用。在本研究中，我们从美国消费品安全委员会（CPSC）召回数据库中整理数据，开发了一个多模态数据集RECALL-MM，利用历史信息进行数据驱动的风险评估，并通过生成方法对其进行扩增。数据集中的模式突显了改进安全措施能够产生重大影响的具体领域。通过展示嵌入召回描述和产品名称的共享潜在空间的交互聚类图，我们扩展了分析方法。利用这些数据驱动的工具，我们探讨了三个案例研究，以展示该数据集在识别产品风险和引导更安全的设计决策方面的应用价值。前两个案例研究展示了设计师如何可视化被召回产品的模式，并将新的产品理念置于更广泛的召回环境中，以前瞻性地预见风险。在第三个案例研究中，我们通过大型语言模型（LLM）仅根据产品图像预测潜在风险，这表明模型能够利用视觉上下文识别风险因素，并在许多风险类别中与历史召回数据保持高度一致。然而，分析也指出了风险预测仍然具有挑战性的领域，强调了在整个设计过程中提高风险意识的重要性。集体而言，这项工作旨在弥合历史召回数据与未来产品安全之间的差距，提出了一种可扩展的数据驱动方法，以实现更安全的工程设计。 

---
# Enhancing Knowledge Graph Completion with Entity Neighborhood and Relation Context 

**Title (ZH)**: 基于实体邻域和关系上下文的知识图谱完成增强 

**Authors**: Jianfang Chen, Kai Zhang, Aoran Gan, Shiwei Tong, Shuanghong Shen, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23205)  

**Abstract**: Knowledge Graph Completion (KGC) aims to infer missing information in Knowledge Graphs (KGs) to address their inherent incompleteness. Traditional structure-based KGC methods, while effective, face significant computational demands and scalability challenges due to the need for dense embedding learning and scoring all entities in the KG for each prediction. Recent text-based approaches using language models like T5 and BERT have mitigated these issues by converting KG triples into text for reasoning. However, they often fail to fully utilize contextual information, focusing mainly on the neighborhood of the entity and neglecting the context of the relation. To address this issue, we propose KGC-ERC, a framework that integrates both types of context to enrich the input of generative language models and enhance their reasoning capabilities. Additionally, we introduce a sampling strategy to effectively select relevant context within input token constraints, which optimizes the utilization of contextual information and potentially improves model performance. Experiments on the Wikidata5M, Wiki27K, and FB15K-237-N datasets show that KGC-ERC outperforms or matches state-of-the-art baselines in predictive performance and scalability. 

**Abstract (ZH)**: 知识图谱完成(KGC)旨在推断知识图谱(KGs)中的缺失信息以解决其固有的不完整性。传统的基于结构的KGC方法虽然有效，但在需要进行密集嵌入学习和每次预测时对KG中的所有实体进行评分方面面临着显著的计算需求和可扩展性挑战。近年来，使用T5和BERT等语言模型的基于文本的方法通过将KG三元组转换为文本来进行推理，缓解了这些难题。然而，它们往往未能充分利用上下文信息，主要关注实体的邻域，而忽视了关系的上下文。为了解决这一问题，我们提出了KGC-ERC框架，该框架整合了两种类型的上下文以丰富生成语言模型的输入并增强其推理能力。此外，我们引入了一种采样策略，在输入令牌约束内有效选择相关上下文，从而优化上下文信息的利用并有可能提高模型性能。在Wikidata5M、Wiki27K和FB15K-237-N数据集上的实验结果显示，KGC-ERC在预测性能和可扩展性方面优于或匹配最先进的基线方法。 

---
# The Challenge of Achieving Attributability in Multilingual Table-to-Text Generation with Question-Answer Blueprints 

**Title (ZH)**: 在使用问题-答案蓝本进行多语言表格到文本生成中实现可追溯性的挑战 

**Authors**: Aden Haussmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.23204)  

**Abstract**: Multilingual Natural Language Generation (NLG) is challenging due to the lack of training data for low-resource languages. However, some low-resource languages have up to tens of millions of speakers globally, making it important to improve NLG tools for them. Table-to-Text NLG is an excellent measure of models' reasoning abilities but is very challenging in the multilingual setting. System outputs are often not attributable, or faithful, to the data in the source table. Intermediate planning techniques like Question-Answer (QA) blueprints have been shown to improve attributability on summarisation tasks. This work explores whether QA blueprints make multilingual Table-to-Text outputs more attributable to the input tables. This paper extends the challenging multilingual Table-to-Text dataset, TaTA, which includes African languages, with QA blueprints. Sequence-to-sequence language models are then finetuned on this dataset, with and without blueprints. Results show that QA blueprints improve performance for models finetuned and evaluated only on English examples, but do not demonstrate gains in the multilingual setting. This is due to inaccuracies in machine translating the blueprints from English into target languages when generating the training data, and models failing to rely closely on the blueprints they generate. An in-depth analysis is conducted on why this is challenging. 

**Abstract (ZH)**: 多语言自然语言生成（NLG）由于低资源语言训练数据不足而具有挑战性。然而，一些低资源语言在全球拥有数千万的使用者，使得改善这些语言的NLG工具变得至关重要。多语言表格到文本NLG是评估模型推理能力的良好指标，但在多语言环境中却极具挑战性。系统输出往往与源表格中的数据不具可追溯性和忠实性。类似问题-回答（QA）蓝图的中间规划技术已被证明在总结任务中可以提高可追溯性。本文研究了QA蓝图是否可以使多语言表格到文本输出更依赖输入表格。本文扩展了包含非洲语言的具有挑战性的多语言表格到文本数据集TaTA，并加入QA蓝图。然后，在此数据集上对序列到序列语言模型进行微调，带有和不带有蓝图。结果显示，对于仅在英语示例上进行微调和评估的模型，QA蓝图提高了性能，但在多语言环境中未表现出改进。这是由于生成训练数据时将蓝图从英语机器翻译到目标语言时的不准确性和模型未能紧密依赖生成的蓝图。对这一挑战进行了深入分析。 

---
# Conversational Agents for Older Adults' Health: A Systematic Literature Review 

**Title (ZH)**: 面向老年人健康的对话代理：一项系统文献综述 

**Authors**: Jiaxin An, Siqi Yi, Yao Lyu, Houjiang Liu, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23153)  

**Abstract**: There has been vast literature that studies Conversational Agents (CAs) in facilitating older adults' health. The vast and diverse studies warrants a comprehensive review that concludes the main findings and proposes research directions for future studies, while few literature review did it from human-computer interaction (HCI) perspective. In this study, we present a survey of existing studies on CAs for older adults' health. Through a systematic review of 72 papers, this work reviewed previously studied older adults' characteristics and analyzed participants' experiences and expectations of CAs for health. We found that (1) Past research has an increasing interest on chatbots and voice assistants and applied CA as multiple roles in older adults' health. (2) Older adults mainly showed low acceptance CAs for health due to various reasons, such as unstable effects, harm to independence, and privacy concerns. (3) Older adults expect CAs to be able to support multiple functions, to communicate using natural language, to be personalized, and to allow users full control. We also discuss the implications based on the findings. 

**Abstract (ZH)**: 现有的大量文献从促进老年人健康管理的角度研究了对话代理（CAs）。尽管如此，鲜有文献从人机交互（HCI）的角度进行综合回顾，总结主要发现并提出未来研究的方向。本研究通过系统回顾72篇论文，总结了现有针对老年人健康管理的CAs的研究，并分析了参与者对CAs的体验和期望。研究发现：（1）过去的研究越来越关注聊天机器人和语音助手，并将CAs应用于老年人健康的不同角色。（2）老年人对健康相关的CAs的接受度较低，原因包括效果不稳定、妨碍独立性、隐私担忧等。（3）老年人期望CAs能够支持多种功能，使用自然语言交流，具有个性化，并让用户拥有充分的控制权。基于这些发现，我们还讨论了其意义。 

---
# Agent-Based Modeling and Deep Neural Networks for Establishing Digital Twins of Secure Facilities under Sensing Restrictions 

**Title (ZH)**: 基于代理模型和深度神经网络的受限感知条件下安全设施的数字双胞胎建立方法 

**Authors**: Chathika Gunaratne, Mason Stott, Debraj De, Gautam Malviya Thakur, Chris Young  

**Link**: [PDF](https://arxiv.org/pdf/2503.23147)  

**Abstract**: Digital twin technologies help practitioners simulate, monitor, and predict undesirable outcomes in-silico, while avoiding the cost and risks of conducting live simulation exercises. Virtual reality (VR) based digital twin technologies are especially useful when monitoring human Patterns of Life (POL) in secure nuclear facilities, where live simulation exercises are too dangerous and costly to ever perform. However, the high-security status of such facilities may restrict modelers from deploying human activity sensors for data collection. This problem was encountered when deploying MetaPOL, a digital twin system to prevent insider threat or sabotage of secure facilities, at a secure nuclear reactor facility at Oak Ridge National Laboratory (ORNL). This challenge was addressed using an agent-based model (ABM), driven by anecdotal evidence of facility personnel POL, to generate synthetic movement trajectories. These synthetic trajectories were then used to train deep neural network surrogates for next location and stay duration prediction to drive NPCs in the VR environment. In this study, we evaluate the efficacy of this technique for establishing NPC movement within MetaPOL and the ability to distinguish NPC movement during normal operations from that during a simulated emergency response. Our results demonstrate the success of using a multi-layer perceptron for next location prediction and mixture density network for stay duration prediction to predict the ABM generated trajectories. We also find that NPC movement in the VR environment driven by the deep neural networks under normal operations remain significantly different to that seen when simulating responses to a simulated emergency scenario. 

**Abstract (ZH)**: 数字孪生技术帮助 Practitioners 在虚拟环境中模拟、监测和预测不良结果，同时避免现场仿真演习的成本和风险。基于虚拟现实（VR）的数字孪生技术在监控安全核设施中的人类生活方式（POL）时尤其有用，因为在这些设施中进行现场仿真演习既危险又昂贵。然而，这类设施的高度安全状况可能会限制建模人员部署人类活动传感器以收集数据。这种问题在将 MetaPOL 数字孪生系统部署于橡树岭国家实验室（ORNL）的安全核反应堆设施中防止内部威胁或破坏时遇到。我们通过使用基于轶事证据的设施人员生活方式的代理基于模型（ABM）来生成合成移动轨迹来解决这一挑战。然后，使用这些合成轨迹来训练深度神经网络代理，以预测下一位置和停留时间，从而驱动 VR 环境中的 NPC。在本研究中，我们评估了此技术在 MetaPOL 中建立 NPC 移动的有效性以及区分正常运营期间与模拟应急响应期间 NPC 移动的能力。结果显示，使用多层感知机进行下一位置预测和使用混合密度网络进行停留时间预测来预测由 ABM 生成的轨迹是成功的。我们还发现，在正常运营下由深度神经网络驱动的 NPC 移动与模拟应急场景响应时的表现存在显著差异。 

---
# How to safely discard features based on aggregate SHAP values 

**Title (ZH)**: 基于聚合SHAP值的安全特征丢弃方法 

**Authors**: Robi Bhattacharjee, Karolin Frohnapfel, Ulrike von Luxburg  

**Link**: [PDF](https://arxiv.org/pdf/2503.23111)  

**Abstract**: SHAP is one of the most popular local feature-attribution methods. Given a function f and an input x, it quantifies each feature's contribution to f(x). Recently, SHAP has been increasingly used for global insights: practitioners average the absolute SHAP values over many data points to compute global feature importance scores, which are then used to discard unimportant features. In this work, we investigate the soundness of this practice by asking whether small aggregate SHAP values necessarily imply that the corresponding feature does not affect the function. Unfortunately, the answer is no: even if the i-th SHAP value is 0 on the entire data support, there exist functions that clearly depend on Feature i. The issue is that computing SHAP values involves evaluating f on points outside of the data support, where f can be strategically designed to mask its dependence on Feature i. To address this, we propose to aggregate SHAP values over the extended support, which is the product of the marginals of the underlying distribution. With this modification, we show that a small aggregate SHAP value implies that we can safely discard the corresponding feature. We then extend our results to KernelSHAP, the most popular method to approximate SHAP values in practice. We show that if KernelSHAP is computed over the extended distribution, a small aggregate value justifies feature removal. This result holds independently of whether KernelSHAP accurately approximates true SHAP values, making it one of the first theoretical results to characterize the KernelSHAP algorithm itself. Our findings have both theoretical and practical implications. We introduce the Shapley Lie algebra, which offers algebraic insights that may enable a deeper investigation of SHAP and we show that randomly permuting each column of the data matrix enables safely discarding features based on aggregate SHAP and KernelSHAP values. 

**Abstract (ZH)**: SHAP是最流行的局部特征 Attribution 方法之一。给定一个函数 \( f \) 和一个输入 \( x \)，它量化每个特征对 \( f(x) \) 的贡献。最近，SHAP 越来越多地被用于全局洞察：实践者通过在大量数据点上取绝对 SHAP 值的平均值来计算全局特征重要性得分，然后基于这些得分丢弃不重要的特征。在本文中，我们通过探究这种做法的有效性，来调查这种做法是否可靠，即小的聚合 SHAP 值是否一定意味着相应的特征对函数没有影响。不幸的是，答案是否定的：即使第 \( i \) 个 SHAP 值在所有数据支持上均为 0，仍然存在函数明显依赖于特征 \( i \) 的情况。问题在于计算 SHAP 值涉及在数据支持之外的点评估 \( f \)，此时 \( f \) 可以被战略性地设计来掩盖其对特征 \( i \) 的依赖。为了解决这一问题，我们建议在扩展支持上聚合 SHAP 值，扩展支持是底层分布边缘的乘积。通过这一修改，我们证明了小的聚合 SHAP 值意味着可以安全地丢弃相应的特征。然后我们将结果扩展到 KernelSHAP，这是实践中最常用的近似 SHAP 值的方法。我们证明，如果在扩展分布上计算 KernelSHAP，小的聚合值可验证特征删除。这一结果独立于 KernelSHAP 是否准确近似真正的 SHAP 值，使其成为第一个直接表征 KernelSHAP 算法自身理论结果之一。我们的发现具有理论和实践意义。我们引入了 Shapley Lie 代数，这为提供了代数洞察，可能有助于更深入地研究 SHAP，并证明随机置换数据矩阵的每一列能够基于聚合 SHAP 和 KernelSHAP 值安全地丢弃特征。 

---
# Fast Training of Recurrent Neural Networks with Stationary State Feedbacks 

**Title (ZH)**: 快速训练具有 stationary 状态反馈的递归神经网络 

**Authors**: Paul Caillon, Erwan Fagnou, Alexandre Allauzen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23104)  

**Abstract**: Recurrent neural networks (RNNs) have recently demonstrated strong performance and faster inference than Transformers at comparable parameter budgets. However, the recursive gradient computation with the backpropagation through time (or BPTT) algorithm remains the major computational bottleneck. In this work, we propose a novel method that replaces BPTT with a fixed gradient feedback mechanism, yielding an efficient approximation of the exact gradient propagation based on the assumption of time stationarity. Our approach leverages state-space model (SSM) principles to define a structured feedback matrix that directly propagates gradients from future time steps. This formulation bypasses the need for recursive gradient backpropagation, significantly reducing training overhead while preserving the network's ability to capture long-term dependencies. The experiments on language modeling benchmarks exhibit competitive perplexity scores, while significantly reducing the training costs. These promising results suggest that designing a feedback method like an SSM can fully exploit the efficiency advantages of RNNs for many practical applications. 

**Abstract (ZH)**: 循环神经网络（RNNs）最近在参数预算相似的情况下展示了比变压器（Transformers）更强的性能和更快的推理速度。然而，时间递归梯度计算（或时间递归反向传播，BPTT）算法仍然是主要的计算瓶颈。在本工作中，我们提出了一种新颖的方法，用固定梯度反馈机制取代BPTT，基于时间平稳性的假设，提供了一种精确梯度传播的高效近似方法。该方法利用状态空间模型（SSM）原理定义了一个结构化的反馈矩阵，直接从未来时间步长传播梯度。这种形式省去了递归梯度反向传播的需要，显著减少了训练开销，同时保持了网络捕捉长期依赖的能力。在语言建模基准上的实验显示了竞争力的困惑度得分，同时显著降低了训练成本。这些有前景的结果表明，设计类似SSM的反馈方法可以充分利用RNNs的效率优势，适用于许多实际应用。 

---
# RL2Grid: Benchmarking Reinforcement Learning in Power Grid Operations 

**Title (ZH)**: RL2Grid: 在电力网络运行中评估强化学习算法 

**Authors**: Enrico Marchesini, Benjamin Donnot, Constance Crozier, Ian Dytham, Christian Merz, Lars Schewe, Nico Westerbeck, Cathy Wu, Antoine Marot, Priya L. Donti  

**Link**: [PDF](https://arxiv.org/pdf/2503.23101)  

**Abstract**: Reinforcement learning (RL) can transform power grid operations by providing adaptive and scalable controllers essential for grid decarbonization. However, existing methods struggle with the complex dynamics, aleatoric uncertainty, long-horizon goals, and hard physical constraints that occur in real-world systems. This paper presents RL2Grid, a benchmark designed in collaboration with power system operators to accelerate progress in grid control and foster RL maturity. Built on a power simulation framework developed by RTE France, RL2Grid standardizes tasks, state and action spaces, and reward structures within a unified interface for a systematic evaluation and comparison of RL approaches. Moreover, we integrate real control heuristics and safety constraints informed by the operators' expertise to ensure RL2Grid aligns with grid operation requirements. We benchmark popular RL baselines on the grid control tasks represented within RL2Grid, establishing reference performance metrics. Our results and discussion highlight the challenges that power grids pose for RL methods, emphasizing the need for novel algorithms capable of handling real-world physical systems. 

**Abstract (ZH)**: 强化学习（RL）可以通過提供適應性和可擴展的控制器來轉變電力網運營，這些控制器對於電力網去碳化至為重要。然而，現有方法在處理現實系統中出現的複雜動態、 aleatoric 不確定性、長時間目標和難以逾越的物理Constraint方面存在困難。本文介紹了一種由電力系統運營商合作設計的Benchmark——RL2Grid，旨在加速電力網控制進展並促進RL能力成熟。RL2Grid基於法國RTE開發的電力模擬框架，規範化了任務、狀態和行為空間以及獎勵架構，為系統評估和比較RL方法提供了統一界面。此外，我們整合了由運營商專長提供的真實控制Heuristics和安全Constraint，確保RL2Grid與電力網運營要求一致。本文在RL2Grid中對代表性強化的基線進行測試，建立了參考性能指標。研究結果和討論突顯了電力網對RL方法的挑戰，強調了需要能夠處理現實物理系統的新算法的需求。 

---
# UNITYAI-GUARD: Pioneering Toxicity Detection Across Low-Resource Indian Languages 

**Title (ZH)**: UNITYAI-GUARD: 跨低资源印度语言的 toxicity检测先锋研究 

**Authors**: Himanshu Beniwal, Reddybathuni Venkat, Rohit Kumar, Birudugadda Srivibhav, Daksh Jain, Pavan Doddi, Eshwar Dhande, Adithya Ananth, Kuldeep, Heer Kubadia, Pratham Sharda, Mayank Singh  

**Link**: [PDF](https://arxiv.org/pdf/2503.23088)  

**Abstract**: This work introduces UnityAI-Guard, a framework for binary toxicity classification targeting low-resource Indian languages. While existing systems predominantly cater to high-resource languages, UnityAI-Guard addresses this critical gap by developing state-of-the-art models for identifying toxic content across diverse Brahmic/Indic scripts. Our approach achieves an impressive average F1-score of 84.23% across seven languages, leveraging a dataset of 888k training instances and 35k manually verified test instances. By advancing multilingual content moderation for linguistically diverse regions, UnityAI-Guard also provides public API access to foster broader adoption and application. 

**Abstract (ZH)**: UnityAI-Guard：一种针对低资源印度语言的二元毒性分类框架 

---
# InkFM: A Foundational Model for Full-Page Online Handwritten Note Understanding 

**Title (ZH)**: InkFM：全页在线手写笔记理解的基础模型 

**Authors**: Anastasiia Fadeeva, Vincent Coriou, Diego Antognini, Claudiu Musat, Andrii Maksai  

**Link**: [PDF](https://arxiv.org/pdf/2503.23081)  

**Abstract**: Tablets and styluses are increasingly popular for taking notes. To optimize this experience and ensure a smooth and efficient workflow, it's important to develop methods for accurately interpreting and understanding the content of handwritten digital notes. We introduce a foundational model called InkFM for analyzing full pages of handwritten content. Trained on a diverse mixture of tasks, this model offers a unique combination of capabilities: recognizing text in 28 different scripts, mathematical expressions recognition, and segmenting pages into distinct elements like text and drawings. Our results demonstrate that these tasks can be effectively unified within a single model, achieving SoTA text line segmentation out-of-the-box quality surpassing public baselines like docTR. Fine- or LoRA-tuning our base model on public datasets further improves the quality of page segmentation, achieves state-of the art text recognition (DeepWriting, CASIA, SCUT, and Mathwriting datasets) and sketch classification (QuickDraw). This adaptability of InkFM provides a powerful starting point for developing applications with handwritten input. 

**Abstract (ZH)**: 表格和平板逐渐流行于笔记记录。为了优化这一体验并确保流畅高效的工作流程，开发能够准确解释和理解电子手写笔记内容的方法十分重要。我们提出了一种名为InkFM的基础模型，用于分析整页手写内容。该模型在多种任务上进行训练，具备独特的优势：识别28种不同的文字、数学表达式识别以及将页面分割为文本和绘制等不同元素的能力。实验结果表明，这些任务可以在一个模型中有效统一，达到了超越公开基准（如docTR）的初始全行分割质量。通过对公共数据集进行精细调整或LoRA调整，InkFM进一步提高了页面分割质量，实现了多项文本识别（DeepWriting、CASIA、SCUT和Mathwriting数据集）和素描分类（QuickDraw）的最新成果。InkFM的这种可调性为其发展基于手写输入的应用程序提供了强大的起点。 

---
# Reproducibility Companion Paper: Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems 

**Title (ZH)**: 可重复性同伴论文：使用户无区别：推荐系统中的属性层面遗忘 

**Authors**: Yuyuan Li, Junjie Fang, Chaochao Chen, Xiaolin Zheng, Yizhao Zhang, Zhongxuan Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.23032)  

**Abstract**: In this paper, we reproduce the experimental results presented in our previous work titled "Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems," which was published in the proceedings of the 31st ACM International Conference on Multimedia. This paper aims to validate the effectiveness of our proposed method and help others reproduce our experimental results. We provide detailed descriptions of our preprocessed datasets, source code structure, configuration file settings, experimental environment, and reproduced experimental results. 

**Abstract (ZH)**: 在本文中，我们重现了我们在之前工作中发表的实验结果，该工作题为《Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems》，发表于第31届ACM国际多媒体会议论文集。本文旨在验证我们提出方法的有效性，并帮助他人重现我们的实验结果。我们详细描述了预处理数据集、源代码结构、配置文件设置、实验环境和重现的实验结果。 

---
# Towards Understanding the Optimization Mechanisms in Deep Learning 

**Title (ZH)**: 理解深度学习中的优化机制 

**Authors**: Binchuan Qi, Wei Gong, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.23016)  

**Abstract**: In this paper, we adopt a probability distribution estimation perspective to explore the optimization mechanisms of supervised classification using deep neural networks. We demonstrate that, when employing the Fenchel-Young loss, despite the non-convex nature of the fitting error with respect to the model's parameters, global optimal solutions can be approximated by simultaneously minimizing both the gradient norm and the structural error. The former can be controlled through gradient descent algorithms. For the latter, we prove that it can be managed by increasing the number of parameters and ensuring parameter independence, thereby providing theoretical insights into mechanisms such as over-parameterization and random initialization. Ultimately, the paper validates the key conclusions of the proposed method through empirical results, illustrating its practical effectiveness. 

**Abstract (ZH)**: 本文采用概率分布估计的观点，探讨使用深度神经网络进行监督分类的优化机制。我们证明，在采用Fenchel-Young损失的情况下，尽管模型参数的拟合误差具有非凸性，通过同时最小化梯度范数和结构误差，可以近似获得全局最优解。前者可通过梯度下降算法进行控制。对于后者，我们证明可以通过增加参数数量并确保参数独立性来管理，从而为过参数化和随机初始化等机制提供理论洞见。最终，通过实证结果验证了所提方法的关键结论，展示了其实用有效性。 

---
# MSNGO: multi-species protein function annotation based on 3D protein structure and network propagation 

**Title (ZH)**: MSNGO：基于3D蛋白质结构和网络传播的多物种蛋白质功能注释 

**Authors**: Beibei Wang, Boyue Cui, Shiqu Chen, Xuan Wang, Yadong Wang, Junyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.23014)  

**Abstract**: Motivation: In recent years, protein function prediction has broken through the bottleneck of sequence features, significantly improving prediction accuracy using high-precision protein structures predicted by AlphaFold2. While single-species protein function prediction methods have achieved remarkable success, multi-species protein function prediction methods are still in the stage of using PPI networks and sequence features. Providing effective cross-species label propagation for species with sparse protein annotations remains a challenging issue. To address this problem, we propose the MSNGO model, which integrates structural features and network propagation methods. Our validation shows that using structural features can significantly improve the accuracy of multi-species protein function prediction. Results: We employ graph representation learning techniques to extract amino acid representations from protein structure contact maps and train a structural model using a graph convolution pooling module to derive protein-level structural features. After incorporating the sequence features from ESM-2, we apply a network propagation algorithm to aggregate information and update node representations within a heterogeneous network. The results demonstrate that MSNGO outperforms previous multi-species protein function prediction methods that rely on sequence features and PPI networks. Availability: this https URL. 

**Abstract (ZH)**: 动机：近年来，蛋白质功能预测突破了序列特征的瓶颈，通过AlphaFold2预测的高精度蛋白质结构显著提高了预测准确性。虽然单物种蛋白质功能预测方法已经取得了显著成功，但多物种蛋白质功能预测方法仍处于依靠PPI网络和序列特征的阶段。为物种稀缺蛋白质注释提供有效的跨物种标签传播仍然是一个具有挑战性的问题。为了解决这个问题，我们提出了MSNGO模型，该模型结合了结构特征和网络传播方法。我们的验证结果表明，使用结构特征可以显著提高多物种蛋白质功能预测的准确性。结果：我们采用图表示学习技术从蛋白质结构接触图中提取氨基酸表示，并利用图卷积池化模块训练结构模型以推导蛋白质级别的结构特征。结合ESM-2的序列特征后，我们应用网络传播算法聚合信息并在异质网络中更新节点表示。结果表明，MSNGO在依赖序列特征和PPI网络的多物种蛋白质功能预测方法中表现更优。可用性：https://链接。 

---
# Learning Structure-enhanced Temporal Point Processes with Gromov-Wasserstein Regularization 

**Title (ZH)**: 具有Gromov-Wasserstein正则化的结构增强时间点过程学习 

**Authors**: Qingmei Wang, Fanmeng Wang, Bing Su, Hongteng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23002)  

**Abstract**: Real-world event sequences are often generated by different temporal point processes (TPPs) and thus have clustering structures. Nonetheless, in the modeling and prediction of event sequences, most existing TPPs ignore the inherent clustering structures of the event sequences, leading to the models with unsatisfactory interpretability. In this study, we learn structure-enhanced TPPs with the help of Gromov-Wasserstein (GW) regularization, which imposes clustering structures on the sequence-level embeddings of the TPPs in the maximum likelihood estimation this http URL the training phase, the proposed method leverages a nonparametric TPP kernel to regularize the similarity matrix derived based on the sequence embeddings. In large-scale applications, we sample the kernel matrix and implement the regularization as a Gromov-Wasserstein (GW) discrepancy term, which achieves a trade-off between regularity and computational this http URL TPPs learned through this method result in clustered sequence embeddings and demonstrate competitive predictive and clustering performance, significantly improving the model interpretability without compromising prediction accuracy. 

**Abstract (ZH)**: 基于Gromov-Wasserstein正则化的结构增强时间点过程模型及其应用 

---
# AuditVotes: A Framework Towards More Deployable Certified Robustness for Graph Neural Networks 

**Title (ZH)**: 审计投票：一种更易于部署的图神经网络认证鲁棒性框架 

**Authors**: Yuni Lai, Yulin Zhu, Yixuan Sun, Yulun Wu, Bin Xiao, Gaolei Li, Jianhua Li, Kai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.22998)  

**Abstract**: Despite advancements in Graph Neural Networks (GNNs), adaptive attacks continue to challenge their robustness. Certified robustness based on randomized smoothing has emerged as a promising solution, offering provable guarantees that a model's predictions remain stable under adversarial perturbations within a specified range. However, existing methods face a critical trade-off between accuracy and robustness, as achieving stronger robustness requires introducing greater noise into the input graph. This excessive randomization degrades data quality and disrupts prediction consistency, limiting the practical deployment of certifiably robust GNNs in real-world scenarios where both accuracy and robustness are essential. To address this challenge, we propose \textbf{AuditVotes}, the first framework to achieve both high clean accuracy and certifiably robust accuracy for GNNs. It integrates randomized smoothing with two key components, \underline{au}gmentation and con\underline{dit}ional smoothing, aiming to improve data quality and prediction consistency. The augmentation, acting as a pre-processing step, de-noises the randomized graph, significantly improving data quality and clean accuracy. The conditional smoothing, serving as a post-processing step, employs a filtering function to selectively count votes, thereby filtering low-quality predictions and improving voting consistency. Extensive experimental results demonstrate that AuditVotes significantly enhances clean accuracy, certified robustness, and empirical robustness while maintaining high computational efficiency. Notably, compared to baseline randomized smoothing, AuditVotes improves clean accuracy by $437.1\%$ and certified accuracy by $409.3\%$ when the attacker can arbitrarily insert $20$ edges on the Cora-ML datasets, representing a substantial step toward deploying certifiably robust GNNs in real-world applications. 

**Abstract (ZH)**: AuditVotes：同时实现高清洁准确率和认证鲁棒准确率的图神经网络框架 

---
# DC-SGD: Differentially Private SGD with Dynamic Clipping through Gradient Norm Distribution Estimation 

**Title (ZH)**: DC-SGD：基于梯度 norm 分布估计的动态裁剪差分隐私 SGD 

**Authors**: Chengkun Wei, Weixian Li, Gong Chen, Wenzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22988)  

**Abstract**: Differentially Private Stochastic Gradient Descent (DP-SGD) is a widely adopted technique for privacy-preserving deep learning. A critical challenge in DP-SGD is selecting the optimal clipping threshold C, which involves balancing the trade-off between clipping bias and noise magnitude, incurring substantial privacy and computing overhead during hyperparameter tuning.
In this paper, we propose Dynamic Clipping DP-SGD (DC-SGD), a framework that leverages differentially private histograms to estimate gradient norm distributions and dynamically adjust the clipping threshold C. Our framework includes two novel mechanisms: DC-SGD-P and DC-SGD-E. DC-SGD-P adjusts the clipping threshold based on a percentile of gradient norms, while DC-SGD-E minimizes the expected squared error of gradients to optimize C. These dynamic adjustments significantly reduce the burden of hyperparameter tuning C. The extensive experiments on various deep learning tasks, including image classification and natural language processing, show that our proposed dynamic algorithms achieve up to 9 times acceleration on hyperparameter tuning than DP-SGD. And DC-SGD-E can achieve an accuracy improvement of 10.62% on CIFAR10 than DP-SGD under the same privacy budget of hyperparameter tuning. We conduct rigorous theoretical privacy and convergence analyses, showing that our methods seamlessly integrate with the Adam optimizer. Our results highlight the robust performance and efficiency of DC-SGD, offering a practical solution for differentially private deep learning with reduced computational overhead and enhanced privacy guarantees. 

**Abstract (ZH)**: 动态裁剪DP-SGD：基于差异隐私直方图的自适应剪裁阈值方法 

---
# PartialLoading: User Scheduling and Bandwidth Allocation for Parameter-sharing Edge Inference 

**Title (ZH)**: 局部加载：用户调度与带宽分配的参数共享边缘推理 

**Authors**: Guanqiao Qu, Qian Chen, Xianhao Chen, Kaibin Huang, Yuguang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22982)  

**Abstract**: By provisioning inference offloading services, edge inference drives the rapid growth of AI applications at the network edge. However, achieving high task throughput with stringent latency requirements remains a significant challenge. To address this issue, we develop a parameter-sharing AI model loading (PartialLoading) framework for multi-user edge inference, which exploits two key insights: 1) the majority of latency arises from loading AI models into server GPU memory, and 2) different AI models can share a significant number of parameters, for which redundant loading should be avoided. Towards this end, we formulate a joint multi-user scheduling and spectrum bandwidth allocation problem to maximize task throughput by exploiting shared parameter blocks across models. The intuition is to judiciously schedule user requests to reuse the shared parameter blocks between consecutively loaded models, thereby reducing model loading time substantially. To facilitate solution finding, we decouple the problem into two sub-problems, i.e., user scheduling and bandwidth allocation, showing that solving them sequentially is equivalent to solving the original problem. Due to the NP-hardness of the problem, we first study an important special case called the "bottom-layer-sharing" case, where AI models share some bottom layers within clusters, and design a dynamic programming-based algorithm to obtain the optimal solution in polynomial time. For the general case, where shared parameter blocks appear at arbitrary positions within AI models, we propose a greedy heuristic to obtain the sub-optimal solution efficiently. Simulation results demonstrate that the proposed framework significantly improves task throughput under deadline constraints compared with user scheduling without exploiting parameter sharing. 

**Abstract (ZH)**: 通过提供推理卸载服务，边缘推理促使网络边缘的AI应用快速增长。然而，要在严格的时间延迟要求下实现高效的任务吞吐量仍面临重大挑战。为应对这一问题，我们为多用户边缘推理开发了一种参数共享AI模型加载（PartialLoading）框架，利用了两个关键洞察：1）大部分延迟来自于将AI模型加载到服务器GPU内存；2）不同的AI模型可以共享大量参数，因此应避免冗余加载。为此，我们形成了一个联合多用户调度和频谱带宽分配的问题，通过利用模型间共享的参数块来最大化任务吞吐量。直觉是明智地调度用户请求，以便在连续加载的模型之间重用共享的参数块，从而显著减少模型加载时间。为了便于求解，我们将问题分解为两个子问题，即用户调度和带宽分配，证明顺序解决它们等价于解决原始问题。由于问题的NP难性，我们首先研究了一个重要的特例，称为“最底层共享”情况，其中AI模型在簇内共享一些最底层，并设计了一种基于动态规划的算法，在多项式时间内获得最优解。对于共享参数块出现在AI模型任意位置的一般情况，我们提出了一种贪婪启发式方法，以高效地获得次优解。仿真结果表明，与不利用参数共享的用户调度相比，所提出的框架在满足截止时间约束时显著提高了任务吞吐量。 

---
# Enhancing Federated Learning Through Secure Cluster-Weighted Client Aggregation 

**Title (ZH)**: 增强联邦学习的通过安全聚类加权客户端聚合方法 

**Authors**: Kanishka Ranaweera, Azadeh Ghari Neiat, Xiao Liu, Bipasha Kashyap, Pubudu N. Pathirana  

**Link**: [PDF](https://arxiv.org/pdf/2503.22971)  

**Abstract**: Federated learning (FL) has emerged as a promising paradigm in machine learning, enabling collaborative model training across decentralized devices without the need for raw data sharing. In FL, a global model is trained iteratively on local datasets residing on individual devices, each contributing to the model's improvement. However, the heterogeneous nature of these local datasets, stemming from diverse user behaviours, device capabilities, and data distributions, poses a significant challenge. The inherent heterogeneity in federated learning gives rise to various issues, including model performance discrepancies, convergence challenges, and potential privacy concerns. As the global model progresses through rounds of training, the disparities in local data quality and quantity can impede the overall effectiveness of federated learning systems. Moreover, maintaining fairness and privacy across diverse user groups becomes a paramount concern. To address this issue, this paper introduces a novel FL framework, ClusterGuardFL, that employs dissimilarity scores, k-means clustering, and reconciliation confidence scores to dynamically assign weights to client updates. The dissimilarity scores between global and local models guide the formation of clusters, with cluster size influencing the weight allocation. Within each cluster, a reconciliation confidence score is calculated for individual data points, and a softmax layer generates customized weights for clients. These weights are utilized in the aggregation process, enhancing the model's robustness and privacy. Experimental results demonstrate the efficacy of the proposed approach in achieving improved model performance in diverse datasets. 

**Abstract (ZH)**: 联邦学习（FL）作为一种在机器学习中的有前途范式，允许在无需原始数据共享的情况下，跨去中心化设备进行协作模型训练。在FL中，全局模型通过迭代训练各个设备上的本地数据集来进行训练，每台设备都为模型改进做出贡献。然而，由于来自多样用户行为、设备能力及数据分布的本地数据集的异构性，这一特性带来了显著的挑战。联邦学习固有的异构性导致了包括模型性能差异、收敛挑战以及潜在隐私问题等一系列问题。随着全球模型通过多轮训练的进展，本地数据质量与数量的差异可能妨碍联邦学习系统的整体有效性。此外，在多元用户群体中维护公平性和隐私保护变得尤为关键。为了解决这一问题，本文提出了一种新的联邦学习框架ClusterGuardFL，该框架采用不相似度评分、k均值聚类和校正置信度评分来动态为客户端更新分配权重。全局模型与本地模型之间的不相似度评分指导聚类的形成，簇的大小影响权重分配。在每个簇内部，计算个体数据点的校正置信度评分，使用softmax层生成客户端的定制化权重。这些权重应用于聚合过程，增强了模型的鲁棒性和隐私保护性。实验结果证明，所提出的这种方法在多种数据集中实现了模型性能的改进。 

---
# Student-Powered Digital Scholarship CoLab Project in the HKUST Library: Develop a Chinese Named-Entity Recognition (NER) Tool within One Semester from the Ground Up 

**Title (ZH)**: HKUST图书馆基于学生的数字学术CoLab项目：在一个月学期内自底向上开发一种中文命名实体识别（NER）工具 

**Authors**: Sherry S.L. Yip, Berry L. Han, Holly H.Y. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2503.22967)  

**Abstract**: Starting in February 2024, the HKUST Library further extended the scope of AI literacy to AI utilization, which focuses on fostering student involvement in utilizing state-of-the-art technologies in the projects that initiated by the Library, named "Digital Scholarship (DS) CoLab". A key focus of the DS CoLab scheme has been on cultivating talents and enabling students to utilize advanced technologies in practical context. It aims to reinforce the library's role as a catalyst and hub for fostering multidisciplinary collaboration and cultivate the "can do spirit" among university members. The Library offers 1-2 projects per year for students to engage with advanced technologies in practical contexts while supporting the Library in tackling challenges and streamlining operational tasks. The tool that introduced in this paper was mainly developed by two of the authors, Sherry Yip Sau Lai and Berry Han Liuruo, as part-time student helpers under one of our DS CoLab scheme in the 2024 Spring Semester (February to May 2024). This paper details the complete journey from ideation to implementation of developing a Chinese Named-Entity Recognition (NER) Tool from the group up within one semester, from the initial research and planning stages to execution and come up a viable product. The collaborative spirit fostered by this project, with students playing a central role, exemplifies the power and potential of innovative educational models that prioritize hands-on learning with student involvement. 

**Abstract (ZH)**: 从2024年2月起，港科大图书馆进一步将AI素养扩展至AI应用，专注于培养学生在图书馆发起的“数字学术（DS）合作实验室（CoLab）”项目中利用前沿技术。DS CoLab方案的核心重点在于培养人才，使学生能够在实际情境中利用先进技术。其目标是强化图书馆作为跨学科合作催化剂和中心的作用，并培养大学成员的“敢于尝试”的精神。图书馆每年提供1-2个项目，让学生在实际情境中接触前沿技术，同时支持图书馆应对挑战和优化运营任务。本文介绍的工具主要由Sherry Yip Sau Lai和Berry Han Liuruo两位作者在2024年春学期（2月至5月）DS CoLab方案的兼职学生助手身份下开发。本文详细描述了该团队在一个学期中从概念构思到实施开发一款中文命名实体识别（NER）工具的全过程，从初步研究和规划阶段到执行，最终形成一个可行的产品。该项目所培养的合作精神，体现了以学生参与为导向的创新型教育模式的力量和潜力。 

---
# Late Breaking Results: Breaking Symmetry- Unconventional Placement of Analog Circuits using Multi-Level Multi-Agent Reinforcement Learning 

**Title (ZH)**: Late Breaking Results: 突破对称性—使用多层级多代理强化学习的非常规模拟电路布局 

**Authors**: Supriyo Maji, Linran Zhao, Souradip Poddar, David Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.22958)  

**Abstract**: Layout-dependent effects (LDEs) significantly impact analog circuit performance. Traditionally, designers have relied on symmetric placement of circuit components to mitigate variations caused by LDEs. However, due to non-linear nature of these effects, conventional methods often fall short. We propose an objective-driven, multi-level, multi-agent Q-learning framework to explore unconventional design space of analog layout, opening new avenues for optimizing analog circuit performance. Our approach achieves better variation performance than the state-of-the-art layout techniques. Notably, this is the first application of multi-agent RL in analog layout automation. The proposed approach is compared with non-ML approach based on simulated annealing. 

**Abstract (ZH)**: 基于布局依赖效应的目标驱动多层次多智能体Q学习框架及其在模拟电路性能优化中的应用 

---
# DATAWEAVER: Authoring Data-Driven Narratives through the Integrated Composition of Visualization and Text 

**Title (ZH)**: 数据编织：通过可视化与文本集成创作数据驱动的故事 

**Authors**: Yu Fu, Dennis Bromley, Vidya Setlur  

**Link**: [PDF](https://arxiv.org/pdf/2503.22946)  

**Abstract**: Data-driven storytelling has gained prominence in journalism and other data reporting fields. However, the process of creating these stories remains challenging, often requiring the integration of effective visualizations with compelling narratives to form a cohesive, interactive presentation. To help streamline this process, we present an integrated authoring framework and system, DataWeaver, that supports both visualization-to-text and text-to-visualization composition. DataWeaver enables users to create data narratives anchored to data facts derived from "call-out" interactions, i.e., user-initiated highlights of visualization elements that prompt relevant narrative content. In addition to this "vis-to-text" composition, DataWeaver also supports a "text-initiated" approach, generating relevant interactive visualizations from existing narratives. Key findings from an evaluation with 13 participants highlighted the utility and usability of DataWeaver and the effectiveness of its integrated authoring framework. The evaluation also revealed opportunities to enhance the framework by refining filtering mechanisms and visualization recommendations and better support authoring creativity by introducing advanced customization options. 

**Abstract (ZH)**: 数据驱动的故事讲述在新闻报道和其他数据报告领域中逐渐凸显，然而这一过程仍然具有挑战性，通常需要有效可视化与引人入胜的叙述相结合以形成一个连贯的交互性展示。为了简化这一过程，我们介绍了一种集成的创作框架和系统——DataWeaver，它支持可视化到文本和文本到可视化的组成。DataWeaver 允许用户通过与可视化的“高亮”交互（即由用户触发的可视化元素突出显示）创建锚定于数据事实的数据叙述。除了“可视到文本”的组成方式，DataWeaver 还支持“文本触发”的方法，从现有叙述生成相关的交互式可视化。评价实验（涉及13名参与者）的发现强调了DataWeaver的实用性和易用性以及其集成创作框架的有效性。此外，该评价还揭示了通过完善过滤机制和可视化推荐以及引入更高级的自定义选项来增强框架以更好地支持创作能力的机会。 

---
# FairSAM: Fair Classification on Corrupted Data Through Sharpness-Aware Minimization 

**Title (ZH)**: 公平SAM：通过敏锐度感知最小化在受污染数据上的公平分类 

**Authors**: Yucong Dai, Jie Ji, Xiaolong Ma, Yongkai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22934)  

**Abstract**: Image classification models trained on clean data often suffer from significant performance degradation when exposed to testing corrupted data, such as images with impulse noise, Gaussian noise, or environmental noise. This degradation not only impacts overall performance but also disproportionately affects various demographic subgroups, raising critical algorithmic bias concerns. Although robust learning algorithms like Sharpness-Aware Minimization (SAM) have shown promise in improving overall model robustness and generalization, they fall short in addressing the biased performance degradation across demographic subgroups. Existing fairness-aware machine learning methods - such as fairness constraints and reweighing strategies - aim to reduce performance disparities but hardly maintain robust and equitable accuracy across demographic subgroups when faced with data corruption. This reveals an inherent tension between robustness and fairness when dealing with corrupted data. To address these challenges, we introduce one novel metric specifically designed to assess performance degradation across subgroups under data corruption. Additionally, we propose \textbf{FairSAM}, a new framework that integrates \underline{Fair}ness-oriented strategies into \underline{SAM} to deliver equalized performance across demographic groups under corrupted conditions. Our experiments on multiple real-world datasets and various predictive tasks show that FairSAM successfully reconciles robustness and fairness, offering a structured solution for equitable and resilient image classification in the presence of data corruption. 

**Abstract (ZH)**: 基于去噪数据训练的图像分类模型在遇到冲击噪声、高斯噪声或环境噪声等测试破坏数据时，往往会遭受显著的性能下降，这种下降不仅影响整体性能，还对不同的人口亚组不公平，引起重要的算法偏见问题。虽然像Sharpness-Aware Minimization (SAM)这样的鲁棒学习算法在提高模型整体鲁棒性和泛化性方面显示出前景，但在解决不同人口亚组的偏差性能下降方面仍存在不足。现有的公平感知机器学习方法——如公平约束和重权重策略——旨在减少性能差距，但在面对数据破坏时，难以在不同的人口亚组中保持鲁棒和公平的准确性。这揭示了在处理破坏数据时鲁棒性和公平性之间固有的紧张关系。为应对这些挑战，我们引入了一种新型度量标准，专门用于评估数据破坏下不同亚组的性能下降。此外，我们提出了FairSAM，这是一种新的框架，将公平导向策略整合到SAM中，以便在数据破坏条件下实现不同人口群体的公平性能。我们在多个真实世界数据集和各种预测任务上的实验表明，FairSAM成功地平衡了鲁棒性和公平性，提供了一种在数据破坏情况下实现公平和稳健图像分类的结构化解决方案。 

---
# Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models 

**Title (ZH)**: Quamba2：选择状态空间模型的鲁棒且可扩展的后训练量化框架 

**Authors**: Hung-Yueh Chiang, Chi-Chih Chang, Natalia Frumkin, Kai-Chiang Wu, Mohamed S. Abdelfattah, Diana Marculescu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22879)  

**Abstract**: State Space Models (SSMs) are emerging as a compelling alternative to Transformers because of their consistent memory usage and high performance. Despite this, scaling up SSMs on cloud services or limited-resource devices is challenging due to their storage requirements and computational power. To overcome this, quantizing SSMs with low bit-width data formats can reduce model size and benefit from hardware acceleration. As SSMs are prone to quantization-induced errors, recent efforts have focused on optimizing a particular model or bit-width for efficiency without sacrificing performance. However, distinct bit-width configurations are essential for different scenarios, like W4A8 for boosting large-batch decoding speed, and W4A16 for enhancing generation speed in short prompt applications for a single user. To this end, we present Quamba2, compatible with W8A8, W4A8, and W4A16 for both Mamba1 and Mamba2 backbones, addressing the growing demand for SSM deployment on various platforms. Based on the channel order preserving and activation persistence of SSMs, we propose an offline approach to quantize inputs of a linear recurrence in 8-bit by sorting and clustering for input $x$, combined with a per-state-group quantization for input-dependent parameters $B$ and $C$. To ensure compute-invariance in the SSM output, we rearrange weights offline according to the clustering sequence. The experiments show that Quamba2-8B outperforms several state-of-the-art SSM quantization methods and delivers 1.3$\times$ and 3$\times$ speed-ups in the pre-filling and generation stages, respectively, while offering 4$\times$ memory reduction with only a $1.6\%$ average accuracy drop. The evaluation on MMLU shows the generalizability and robustness of our framework. The code and quantized models will be released at: this https URL. 

**Abstract (ZH)**: State Space Models (SSMs)作为Transformer的有吸引力的替代方案正逐渐兴起，得益于其一致的内存使用和高性能。尽管如此，由于存储需求和计算能力限制，将SSMs扩展到云服务或有限资源设备仍然充满挑战。为了克服这一难题，使用低位宽数据格式对SSMs进行量化可以减小模型大小并受益于硬件加速。由于SSMs容易受到量化引起的误差影响，最近的努力集中在优化特定模型或位宽，以提高效率而不牺牲性能。然而，不同的位宽配置对于不同的场景至关重要，比如W4A8用于提升大批次解码速度，W4A16则用于单一用户短期提示应用中生成速度的提升。为此，我们提出了Quamba2，支持W8A8、W4A8和W4A16，适用于Mamba1和Mamba2的骨干网络，以满足各种平台上SSM部署日益增长的需求。基于SSMs的通道顺序保和服务于状态的激活保持特性，我们提出了一种离线方法，通过排序和聚类对输入x进行8位量化，并结合状态组间输入依赖参数B和C的量化。为了确保SSM输出的计算不变性，我们根据聚类序列离线重排权重。实验结果显示，Quamba2-8B在预填充和生成阶段分别提供了1.3倍和3倍的速度提升，同时实现了4倍的内存减少，并且平均精度下降仅为1.6%。我们的框架在MMLU上的评估展示了其普遍适用性和鲁棒性。代码和量化模型将发布在：this https URL。 

---
# Nonhuman Primate Brain Tissue Segmentation Using a Transfer Learning Approach 

**Title (ZH)**: 非人灵长类大脑组织分割的迁移学习方法 

**Authors**: Zhen Lin, Hongyu Yuan, Richard Barcus, Qing Lyu, Sucheta Chakravarty, Megan E. Lipford, Carol A. Shively, Suzanne Craft, Mohammad Kawas, Jeongchul Kim, Christopher T. Whitlow  

**Link**: [PDF](https://arxiv.org/pdf/2503.22829)  

**Abstract**: Non-human primates (NHPs) serve as critical models for understanding human brain function and neurological disorders due to their close evolutionary relationship with humans. Accurate brain tissue segmentation in NHPs is critical for understanding neurological disorders, but challenging due to the scarcity of annotated NHP brain MRI datasets, the small size of the NHP brain, the limited resolution of available imaging data and the anatomical differences between human and NHP brains. To address these challenges, we propose a novel approach utilizing STU-Net with transfer learning to leverage knowledge transferred from human brain MRI data to enhance segmen-tation accuracy in the NHP brain MRI, particularly when training data is this http URL combination of STU-Net and transfer learning effectively delineates complex tissue boundaries and captures fine anatomical details specific to NHP brains. Notably, our method demonstrated improvement in segmenting small subcortical structures such as putamen and thalamus that are challenging to resolve with limited spatial resolution and tissue contrast, and achieved DSC of over 0.88, IoU over 0.8 and HD95 under 7. This study introduces a robust method for multi-class brain tissue segmentation in NHPs, potentially accelerating research in evolutionary neuroscience and preclinical studies of neurological disorders relevant to human health. 

**Abstract (ZH)**: 非人灵长类动物（NHPs）是研究人类大脑功能和神经疾病的关键模型，由于它们与人类的进化关系密切。非人灵长类动物脑组织分割对于理解神经疾病至关重要，但由于注释的NHP脑MRI数据集稀缺、NHP脑容量小、可用成像数据的分辨率有限以及人类和非人灵长类动物大脑的解剖差异，这使得准确的脑组织分割极具挑战性。为应对这些挑战，我们提出了一种利用STU-Net结合迁移学习的新型方法，以利用来自人类脑MRI数据的知识来增强NHP脑MRI中的分割准确度，尤其是在训练数据稀缺的情况下。这种STU-Net与迁移学习的结合有效地勾勒出复杂的组织边界，并捕获到特定于NHP大脑的精细解剖细节。我们的方法在分割丘脑和壳核等小的基底节结构方面取得了改进，这些结构由于空间分辨率和组织对比度有限而难以解析，实现了DSC超过0.88，IoU超过0.8，HD95小于7。本研究表明了一种稳健的非人灵长类动物多类脑组织分割方法，有可能加速进化神经科学和与人类健康相关的神经疾病预临床研究。 

---
# Data-driven worker activity recognition and picking efficiency estimation in manual strawberry harvesting 

**Title (ZH)**: 基于数据驱动的草莓人工采摘工人的活动识别与采摘效率估计 

**Authors**: Uddhav Bhattarai, Rajkishan Arikapudi, Steven A. Fennimore, Frank N Martin, Stavros G. Vougioukas  

**Link**: [PDF](https://arxiv.org/pdf/2503.22809)  

**Abstract**: Manual fruit harvesting is common in agriculture, but the amount of time that pickers spend on nonproductive activities can make it very inefficient. Accurately identifying picking vs. non-picking activity is crucial for estimating picker efficiency and optimizing labor management and the harvest process. In this study, a practical system was developed to calculate the efficiency of pickers in commercial strawberry harvesting. Instrumented picking carts were used to record in real-time the harvested fruit weight, geo-location, and cart movement. A fleet of these carts was deployed during the commercial strawberry harvest season in Santa Maria, CA. The collected data was then used to train a CNN-LSTM-based deep neural network to classify a picker's activity into ``Pick" and ``NoPick" classes. Experimental evaluations showed that the CNN-LSTM model showed promising activity recognition performance with an F1 score accuracy of up to 0.974. The classification results were then used to compute two worker efficiency metrics: the percentage of time spent actively picking, and the time required to fill a tray. Analysis of the season-long harvest data showed that the pickers spent an average of 73.56% of their total harvest time actively picking strawberries, with an average tray fill time of 6.22 minutes. The mean accuracies of these metrics were 96.29% and 95.42%, respectively. When integrated on a commercial scale, the proposed technology could aid growers in automated worker activity monitoring and harvest optimization, ultimately helping to reduce non-productive time and enhance overall harvest efficiency. 

**Abstract (ZH)**: 基于CNN-LSTM的实用系统在商用草莓采摘中计算采摘工的效率 

---
# GroundHog: Revolutionizing GLDAS Groundwater Storage Downscaling for Enhanced Recharge Estimation in Bangladesh 

**Title (ZH)**: GroundHog: 革命性地改进GLDAS地下水资源储存下标化以提升孟加拉国补给估算 

**Authors**: Saleh Sakib Ahmed, Rashed Uz Zzaman, Saifur Rahman Jony, Faizur Rahman Himel, Afroza Sharmin, A.H.M. Khalequr Rahman, M. Sohel Rahman, Sara Nowreen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22771)  

**Abstract**: Long-term groundwater level (GWL) measurement is vital for effective policymaking and recharge estimation using annual maxima and minima. However, current methods prioritize short-term predictions and lack multi-year applicability, limiting their utility. Moreover, sparse in-situ measurements lead to reliance on low-resolution satellite data like GLDAS as the ground truth for Machine Learning models, further constraining accuracy. To overcome these challenges, we first develop an ML model to mitigate data gaps, achieving $R^2$ scores of 0.855 and 0.963 for maximum and minimum GWL predictions, respectively. Subsequently, using these predictions and well observations as ground truth, we train an Upsampling Model that uses low-resolution (25 km) GLDAS data as input to produce high-resolution (2 km) GWLs, achieving an excellent $R^2$ score of 0.96. Our approach successfully upscales GLDAS data for 2003-2024, allowing high-resolution recharge estimations and revealing critical trends for proactive resource management. Our method allows upsampling of groundwater storage (GWS) from GLDAS to high-resolution GWLs for any points independently of officially curated piezometer data, making it a valuable tool for decision-making. 

**Abstract (ZH)**: 长期内部地下水位（GWL）测量对于有效政策制定和补给估算至关重要，使用年度极值和最小值。然而，当前方法侧重于短期预测，缺乏多年适用性，限制了其实用性。此外，稀疏的现场测量导致依赖低分辨率的卫星数据（如GLDAS）作为机器学习模型的基准，进一步限制了准确性。为克服这些挑战，我们首先开发了一个ML模型来缓解数据缺口，分别在最大和最小地下水位预测中实现了$R^2$分数0.855和0.963。随后，使用这些预测和井观测数据作为基准，我们训练了一个上采样模型，使用低分辨率（25 km）的GLDAS数据作为输入，生成高分辨率（2 km）的地下水位，实现了卓越的$R^2$分数0.96。我们的方法成功地将GLDAS数据扩展到2003-2024年，允许进行高分辨率的补给估算，并揭示了主动资源管理中关键的趋势。该方法允许独立于正式整理的抽水测量数据对地下水存储（GWS）进行上采样以生成高分辨率的地下水位，使其成为决策的关键工具。 

---
# The Cost of Local and Global Fairness in Federated Learning 

**Title (ZH)**: 本地公平性和全局公平性在联邦学习中的成本 

**Authors**: Yuying Duan, Gelei Xu, Yiyu Shi, Michael Lemmon  

**Link**: [PDF](https://arxiv.org/pdf/2503.22762)  

**Abstract**: With the emerging application of Federated Learning (FL) in finance, hiring and healthcare, FL models are regulated to be fair, preventing disparities with respect to legally protected attributes such as race or gender. Two concepts of fairness are important in FL: global and local fairness. Global fairness addresses the disparity across the entire population and local fairness is concerned with the disparity within each client. Prior fair FL frameworks have improved either global or local fairness without considering both. Furthermore, while the majority of studies on fair FL focuses on binary settings, many real-world applications are multi-class problems. This paper proposes a framework that investigates the minimum accuracy lost for enforcing a specified level of global and local fairness in multi-class FL settings. Our framework leads to a simple post-processing algorithm that derives fair outcome predictors from the Bayesian optimal score functions. Experimental results show that our algorithm outperforms the current state of the art (SOTA) with regard to the accuracy-fairness tradoffs, computational and communication costs. Codes are available at: this https URL . 

**Abstract (ZH)**: 随着联邦学习（FL）在金融、招聘和医疗等领域的发展，FL模型需要被监管以确保公平性，防止与种族或性别等法律保护属性相关的不平等。FL中的公平性包含两个概念：全局公平和局部公平。全局公平关注整个群体的不平等现象，而局部公平关注每个客户端内的不平等。此前的公平联邦学习框架仅在全局或局部公平中有所改进，而未同时考虑两者。此外，尽管大多数关于公平联邦学习的研究集中在二分类问题上，但许多实际应用是多类问题。本文提出了一种框架，以探讨在多类联邦学习设置中强制执行指定水平的全局和局部公平所需的最小准确度损失。该框架导出了基于贝叶斯最优评分函数的公平结果预测器的简单后处理算法。实验结果表明，与当前最先进的算法相比，在准确度-公平性权衡、计算和通信成本方面，我们的算法表现出色。代码可在以下链接获取：this https URL。 

---
# Data Poisoning in Deep Learning: A Survey 

**Title (ZH)**: 深度学习中的数据投毒：一个综述 

**Authors**: Pinlong Zhao, Weiyao Zhu, Pengfei Jiao, Di Gao, Ou Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22759)  

**Abstract**: Deep learning has become a cornerstone of modern artificial intelligence, enabling transformative applications across a wide range of domains. As the core element of deep learning, the quality and security of training data critically influence model performance and reliability. However, during the training process, deep learning models face the significant threat of data poisoning, where attackers introduce maliciously manipulated training data to degrade model accuracy or lead to anomalous behavior. While existing surveys provide valuable insights into data poisoning, they generally adopt a broad perspective, encompassing both attacks and defenses, but lack a dedicated, in-depth analysis of poisoning attacks specifically in deep learning. In this survey, we bridge this gap by presenting a comprehensive and targeted review of data poisoning in deep learning. First, this survey categorizes data poisoning attacks across multiple perspectives, providing an in-depth analysis of their characteristics and underlying design princinples. Second, the discussion is extended to the emerging area of data poisoning in large language models(LLMs). Finally, we explore critical open challenges in the field and propose potential research directions to advance the field further. To support further exploration, an up-to-date repository of resources on data poisoning in deep learning is available at this https URL. 

**Abstract (ZH)**: 深度学习已成为现代人工智能的基石，能够在多个领域推动变革性应用。作为深度学习的核心要素，训练数据的质量和安全性对模型性能和可靠性至关重要。然而，在训练过程中，深度学习模型面临数据投毒这一重大威胁，攻击者通过引入恶意操纵的数据来降低模型准确性或导致异常行为。尽管现有综述提供了数据投毒有价值的见解，它们通常采用宽泛的角度，涵盖了攻击和防御两方面，但在针对深度学习中的数据投毒攻击进行专门的深入分析方面存在不足。在本文综述中，我们通过呈现数据投毒在深度学习中的全面而有针对性的回顾来填补这一空白。首先，本文综述从多个角度对数据投毒攻击进行分类，并深入分析其特点和设计原则。其次，讨论扩展到大型语言模型（LLMs）中的数据投毒新兴领域。最后，我们探讨了该领域中的关键开放挑战，并提出了潜在的研究方向以进一步推动该领域的发展。为了支持进一步的研究，最新的数据投毒资源库可参见此网址：[请补充具体网址]。 

---
# Towards an intelligent assessment system for evaluating the development of algorithmic thinking skills: An exploratory study in Swiss compulsory schools 

**Title (ZH)**: 面向算法思维能力评估的智能系统研究：一项瑞士义务教育学校探索性研究 

**Authors**: Giorgia Adorni  

**Link**: [PDF](https://arxiv.org/pdf/2503.22756)  

**Abstract**: The rapid digitalisation of contemporary society has profoundly impacted various facets of our lives, including healthcare, communication, business, and education. The ability to engage with new technologies and solve problems has become crucial, making CT skills, such as pattern recognition, decomposition, and algorithm design, essential competencies. In response, Switzerland is conducting research and initiatives to integrate CT into its educational system. This study aims to develop a comprehensive framework for large-scale assessment of CT skills, particularly focusing on AT, the ability to design algorithms. To achieve this, we first developed a competence model capturing the situated and developmental nature of CT, guiding the design of activities tailored to cognitive abilities, age, and context. This framework clarifies how activity characteristics influence CT development and how to assess these competencies. Additionally, we developed an activity for large-scale assessment of AT skills, offered in two variants: one based on non-digital artefacts (unplugged) and manual expert assessment, and the other based on digital artefacts (virtual) and automatic assessment. To provide a more comprehensive evaluation of students' competencies, we developed an IAS based on BNs with noisy gates, which offers real-time probabilistic assessment for each skill rather than a single overall score. The results indicate that the proposed instrument can measure AT competencies across different age groups and educational contexts in Switzerland, demonstrating its applicability for large-scale use. AT competencies exhibit a progressive development, with no overall gender differences, though variations are observed at the school level, significantly influenced by the artefact-based environment and its context, underscoring the importance of creating accessible and adaptable assessment tools. 

**Abstract (ZH)**: 当代社会的快速数字化对我们的生活诸多方面产生了深远影响，包括医疗保健、通信、商业和教育。掌握新技术的能力和解决问题的能力变得至关重要，使得模式识别、分解和算法设计等CT技能成为必不可少的技能。针对这一需求，瑞士开展了研究和项目，旨在将其CT技能融入教育体系。本研究旨在开发一个全面的框架，用于大型评估CT技能，特别是算法设计（AT）技能。为此，首先开发了一个技能模型，捕捉CT的环境性和发展阶段特性，指导针对认知能力、年龄和环境背景定制活动的设计。该框架阐明了活动特性如何影响CT的发展以及如何评估这些技能。此外，还开发了一种活动，用于评估AT技能的大型评估，提供两种变体：一种基于非数字化制品（非连接式）和手工专家评估，另一种基于数字化制品（虚拟）和自动评估。为了对学生的技能进行全面评估，我们基于贝叶斯网络（BN）和噪声门开发了一种即时概率评估方法（IAS），为每个技能提供即时概率评估，而非单一总体评分。结果表明，提议的工具可以跨不同年龄段和教育背景在瑞士测量AT技能，证明其在大规模使用中的适用性。AT技能表现出逐步发展，尽管总体上没有性别差异，但在学校层面存在显著差异，这些差异受到制品基础环境及其背景的显著影响，强调了创建可访问和适应性强评估工具的重要性。 

---
# Reasoning Under Threat: Symbolic and Neural Techniques for Cybersecurity Verification 

**Title (ZH)**: 在威胁下推理：网络安全验证的符号与神经技术 

**Authors**: Sarah Veronica  

**Link**: [PDF](https://arxiv.org/pdf/2503.22755)  

**Abstract**: Cybersecurity demands rigorous and scalable techniques to ensure system correctness, robustness, and resilience against evolving threats. Automated reasoning, encompassing formal logic, theorem proving, model checking, and symbolic analysis, provides a foundational framework for verifying security properties across diverse domains such as access control, protocol design, vulnerability detection, and adversarial modeling. This survey presents a comprehensive overview of the role of automated reasoning in cybersecurity, analyzing how logical systems, including temporal, deontic, and epistemic logics are employed to formalize and verify security guarantees. We examine SOTA tools and frameworks, explore integrations with AI for neural-symbolic reasoning, and highlight critical research gaps, particularly in scalability, compositionality, and multi-layered security modeling. The paper concludes with a set of well-grounded future research directions, aiming to foster the development of secure systems through formal, automated, and explainable reasoning techniques. 

**Abstract (ZH)**: 网络安全需要严格且可扩展的技术来确保系统的正确性、稳健性和对不断演变的威胁的韧性。自动推理，涵盖形式逻辑、定理证明、模型检测和符号分析，为验证跨接入控制、协议设计、漏洞检测和对抗建模等领域中的安全属性提供了基础框架。本文综述了自动推理在网络安全中的作用，分析了如何使用时间逻辑、义务逻辑和知识逻辑等逻辑系统来形式化和验证安全保证。我们研究了当今最先进的工具和框架，探讨了与人工智能的整合以实现神经符号推理，并指出了关键的研究空白，特别是可扩展性、组合性和多层次安全建模。论文最后提出了一系列坚实的研究方向，旨在通过正式、自动化和可解释的推理技术促进安全系统的开发。 

---
# Model Lake: a New Alternative for Machine Learning Models Management and Governance 

**Title (ZH)**: 模型湖：机器学习模型管理与治理的新选择 

**Authors**: Moncef Garouani, Franck Ravat, Nathalie Valles-Parlangeau  

**Link**: [PDF](https://arxiv.org/pdf/2503.22754)  

**Abstract**: The rise of artificial intelligence and data science across industries underscores the pressing need for effective management and governance of machine learning (ML) models. Traditional approaches to ML models management often involve disparate storage systems and lack standardized methodologies for versioning, audit, and re-use. Inspired by data lake concepts, this paper develops the concept of ML Model Lake as a centralized management framework for datasets, codes, and models within organizations environments. We provide an in-depth exploration of the Model Lake concept, delineating its architectural foundations, key components, operational benefits, and practical challenges. We discuss the transformative potential of adopting a Model Lake approach, such as enhanced model lifecycle management, discovery, audit, and reusability. Furthermore, we illustrate a real-world application of Model Lake and its transformative impact on data, code and model management practices. 

**Abstract (ZH)**: 人工智能和数据科学在各行业的兴起凸显了有效管理机器学习（ML）模型的紧迫需求。传统意义上的ML模型管理方法往往依赖于分散的存储系统，并缺乏版本控制、审计和重复利用的标准化方法。借鉴数据湖的概念，本文提出了ML模型湖的概念，作为一种组织内部集中管理数据集、代码和模型的框架。我们深入探讨了模型湖的概念，阐述其架构基础、关键组件、操作优势及实际挑战。我们讨论了采用模型湖方法的变革潜力，如增强的模型生命周期管理、发现、审计和重复利用能力。此外，我们阐述了一个实际应用模型湖的例子及其对数据、代码和模型管理实践的变革影响。 

---
# From Individual to Group: Developing a Context-Aware Multi-Criteria Group Recommender System 

**Title (ZH)**: 从个体到群体：发展一种基于上下文的多准则群体推荐系统 

**Authors**: Ngoc Luyen Le, Marie-Hélène Abel  

**Link**: [PDF](https://arxiv.org/pdf/2503.22752)  

**Abstract**: Group decision-making is becoming increasingly common in areas such as education, dining, travel, and finance, where collaborative choices must balance diverse individual preferences. While conventional recommender systems are effective in personalization, they fall short in group settings due to their inability to manage conflicting preferences, contextual factors, and multiple evaluation criteria. This study presents the development of a Context-Aware Multi-Criteria Group Recommender System (CA-MCGRS) designed to address these challenges by integrating contextual factors and multiple criteria to enhance recommendation accuracy. By leveraging a Multi-Head Attention mechanism, our model dynamically weighs the importance of different features. Experiments conducted on an educational dataset with varied ratings and contextual variables demonstrate that CA-MCGRS consistently outperforms other approaches across four scenarios. Our findings underscore the importance of incorporating context and multi-criteria evaluations to improve group recommendations, offering valuable insights for developing more effective group recommender systems. 

**Abstract (ZH)**: 基于上下文的多准则群体推荐系统（CA-MCGRS）：通过整合上下文因素和多准则提高推荐准确性 

---
# Advancing Spatiotemporal Prediction using Artificial Intelligence: Extending the Framework of Geographically and Temporally Weighted Neural Network (GTWNN) for Differing Geographical and Temporal Contexts 

**Title (ZH)**: 使用人工智能推进时空预测：扩展地理和时间加权神经网络（GTWNN）框架以适应不同的地理和时间背景 

**Authors**: Nicholas Robert Fisk, Matthew Ng Kok Ming, Zahratu Shabrina  

**Link**: [PDF](https://arxiv.org/pdf/2503.22751)  

**Abstract**: This paper aims at improving predictive crime models by extending the mathematical framework of Artificial Neural Networks (ANNs) tailored to general spatiotemporal problems and appropriately applying them. Recent advancements in the geospatial-temporal modelling field have focused on the inclusion of geographical weighting in their deep learning models to account for nonspatial stationarity, which is often apparent in spatial data. We formulate a novel semi-analytical approach to solving Geographically and Temporally Weighted Regression (GTWR), and applying it to London crime data. The results produce high-accuracy predictive evaluation scores that affirm the validity of the assumptions and approximations in the approach. This paper presents mathematical advances to the Geographically and Temporally Weighted Neural Network (GTWNN) framework, which offers a novel contribution to the field. Insights from past literature are harmoniously employed with the assumptions and approximations to generate three mathematical extensions to GTWNN's framework. Combinations of these extensions produce five novel ANNs, applied to the London and Detroit datasets. The results suggest that one of the extensions is redundant and is generally surpassed by another extension, which we term the history-dependent module. The remaining extensions form three novel ANN designs that pose potential GTWNN improvements. We evaluated the efficacy of various models in both the London and Detroit crime datasets, highlighting the importance of accounting for specific geographic and temporal characteristics when selecting modelling strategies to improve model suitability. In general, the proposed methods provide the foundations for a more context-aware, accurate, and robust ANN approach in spatio-temporal modelling. 

**Abstract (ZH)**: 本文旨在通过扩展适用于一般空-时问题的人工神经网络（ANN）的数学框架，并合理应用这些框架来改进预测犯罪模型。近年来，地理时空建模领域的发展集中在将地理加权纳入其深度学习模型中，以-account for 非空间平稳性，这在空间数据中经常可见。我们提出了一个新的半解析方法来求解地理加权和时间加权回归（GTWR），并将其应用于伦敦犯罪数据。结果产生了高精度的预测评估得分，证实了该方法假设和近似的有效性。本文提出了地理加权和时间加权神经网络（GTWNN）框架的数学进展，为该领域做出了新颖的贡献。我们和谐地运用了以往文献的见解与假设和近似，生成了GTWNN框架的三个数学扩展。这些扩展的组合产生了五个新的ANN，应用于伦敦和底特律数据集。结果表明，其中一个扩展是冗余的，并且普遍被另一个扩展——我们称之为历史依赖模块——所超越。剩余的扩展形成了三种新的ANN设计，可能改进GTWNN。我们在伦敦和底特律的犯罪数据集上评估了各种模型的有效性，强调了在选择建模策略以改善模型适应性时，考虑特定的地理和时间特征的重要性。总体而言，所提出的方法为时空建模中更具情境意识、更精确和更稳健的ANN方法奠定了基础。 

---
# Adaptive Clipping for Privacy-Preserving Few-Shot Learning: Enhancing Generalization with Limited Data 

**Title (ZH)**: 自适应裁剪以实现隐私保护的少样本学习：利用有限数据增强泛化能力 

**Authors**: Kanishka Ranaweera, Dinh C. Nguyen, Pubudu N. Pathirana, David Smith, Ming Ding, Thierry Rakotoarivelo, Aruna Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2503.22749)  

**Abstract**: In the era of data-driven machine-learning applications, privacy concerns and the scarcity of labeled data have become paramount challenges. These challenges are particularly pronounced in the domain of few-shot learning, where the ability to learn from limited labeled data is crucial. Privacy-preserving few-shot learning algorithms have emerged as a promising solution to address such pronounced challenges. However, it is well-known that privacy-preserving techniques often lead to a drop in utility due to the fundamental trade-off between data privacy and model performance. To enhance the utility of privacy-preserving few-shot learning methods, we introduce a novel approach called Meta-Clip. This technique is specifically designed for meta-learning algorithms, including Differentially Private (DP) model-agnostic meta-learning, DP-Reptile, and DP-MetaSGD algorithms, with the objective of balancing data privacy preservation with learning capacity maximization. By dynamically adjusting clipping thresholds during the training process, our Adaptive Clipping method provides fine-grained control over the disclosure of sensitive information, mitigating overfitting on small datasets and significantly improving the generalization performance of meta-learning models. Through comprehensive experiments on diverse benchmark datasets, we demonstrate the effectiveness of our approach in minimizing utility degradation, showcasing a superior privacy-utility trade-off compared to existing privacy-preserving techniques. The adoption of Adaptive Clipping represents a substantial step forward in the field of privacy-preserving few-shot learning, empowering the development of secure and accurate models for real-world applications, especially in scenarios where there are limited data availability. 

**Abstract (ZH)**: 在数据驱动的机器学习时代，隐私保护和标注数据稀缺已成为主要挑战。这些挑战在少数样本学习领域尤为显著，该领域需要从有限的标注数据中学习的能力至关重要。针对这些显著的挑战，隐私保护少数样本学习算法作为一种有前景的解决方案而出现。然而，众所周知，隐私保护技术往往会由于数据隐私与模型性能之间的基本权衡而导致实用性下降。为了提高隐私保护少数样本学习方法的实用性，我们提出了一种名为Meta-Clip的新型方法。该方法专门设计用于元学习算法，包括差分隐私模型无关元学习、差分隐私Reptile和差分隐私MetaSGD算法，旨在平衡数据隐私保护与学习能力最大化。通过在训练过程中动态调整截断阈值，我们的自适应截断方法提供了对敏感信息披露程度的精细化控制，减轻了对小型数据集的过度拟合，并显著提高了元学习模型的泛化性能。通过在多种基准数据集上的全面实验，我们证明了该方法在最小化实用性下降方面的有效性，展示了与现有隐私保护技术相比更优的隐私-实用性权衡。自适应截断方法在隐私保护少数样本学习领域的采用代表了向前迈进的重要一步，为开发安全而准确的模型以应对实际应用中数据有限的情景提供了有力支持。 

---
# LeForecast: Enterprise Hybrid Forecast by Time Series Intelligence 

**Title (ZH)**: LeForecast: 企业时序混合预测 

**Authors**: Zheng Tan, Yiwen Nie, Wenfa Wu, Guanyu Zhang, Yanze Liu, Xinyuan Tian, Kailin Gao, Mengya Liu, Qijiang Cheng, Haipeng Jiang, Yingzheng Ma, Wei Zheng, Yuci Zhu, Yuanyuan Sun, Xiangyu Lei, Xiyu Guan, Wanqing Huang, Shouming Liu, Xiangquan Meng, Pengzhan Qu, Chao Yang, Jiaxuan Fan, Yuan He, Hongsheng Qi, Yangzhou Du  

**Link**: [PDF](https://arxiv.org/pdf/2503.22747)  

**Abstract**: Demand is spiking in industrial fields for multidisciplinary forecasting, where a broad spectrum of sectors needs planning and forecasts to streamline intelligent business management, such as demand forecasting, product planning, inventory optimization, etc. Specifically, these tasks expecting intelligent approaches to learn from sequentially collected historical data and then foresee most possible trend, i.e. time series forecasting. Challenge of it lies in interpreting complex business contexts and the efficiency and generalisation of modelling. With aspirations of pre-trained foundational models for such purpose, given their remarkable success of large foundation model across legions of tasks, we disseminate \leforecast{}, an enterprise intelligence platform tailored for time series tasks. It integrates advanced interpretations of time series data and multi-source information, and a three-pillar modelling engine combining a large foundation model (Le-TSFM), multimodal model and hybrid model to derive insights, predict or infer futures, and then drive optimisation across multiple sectors in enterprise operations. The framework is composed by a model pool, model profiling module, and two different fusion approaches regarding original model architectures. Experimental results verify the efficiency of our trail fusion concepts: router-based fusion network and coordination of large and small models, resulting in high costs for redundant development and maintenance of models. This work reviews deployment of LeForecast and its performance in three industrial use cases. Our comprehensive experiments indicate that LeForecast is a profound and practical platform for efficient and competitive performance. And we do hope that this work can enlighten the research and grounding of time series techniques in accelerating enterprise. 

**Abstract (ZH)**: 工业领域对多学科预测的需求激增：LeForecast企业智能平台在时间序列任务中的应用 

---
# Adaptive Integrated Layered Attention (AILA) 

**Title (ZH)**: 自适应集成分层注意力（AILA） 

**Authors**: William Claster, Suhas KM, Dhairya Gundechia  

**Link**: [PDF](https://arxiv.org/pdf/2503.22742)  

**Abstract**: We propose Adaptive Integrated Layered Attention (AILA), a neural network architecture that combines dense skip connections with different mechanisms for adaptive feature reuse across network layers. We evaluate AILA on three challenging tasks: price forecasting for various commodities and indices (S&P 500, Gold, US dollar Futures, Coffee, Wheat), image recognition using the CIFAR-10 dataset, and sentiment analysis on the IMDB movie review dataset. In all cases, AILA matches strong deep learning baselines (LSTMs, Transformers, and ResNets), achieving it at a fraction of the training and inference time. Notably, we implement and test two versions of the model - AILA-Architecture 1, which uses simple linear layers as the connection mechanism between layers, and AILA-Architecture 2, which implements an attention mechanism to selectively focus on outputs from previous layers. Both architectures are applied in a single-task learning setting, with each model trained separately for individual tasks. Results confirm that AILA's adaptive inter-layer connections yield robust gains by flexibly reusing pertinent features at multiple network depths. The AILA approach thus presents an extension to existing architectures, improving long-range sequence modeling, image recognition with optimised computational speed, and SOTA classification performance in practice. 

**Abstract (ZH)**: 我们提出自适应集成分层注意机制（AILA），这是一种结合了密集跳连连接和不同机制的神经网络架构，用于在网络层间适应性重用特征。我们在三项具有挑战性的任务上评估了AILA：各种商品和指数（S&P 500、黄金、美国期货美元、咖啡、小麦）的价格预测，CIFAR-10数据集上的图像识别，以及IMDB电影评论数据集上的情感分析。在所有情况下，AILA在训练和推断时间仅为强深度学习基线（LSTMs、Transformers和ResNets）的一小部分的情况下，实现了与这些基线相当的结果。值得注意的是，我们实现了并测试了该模型的两个版本——使用简单线性层作为层间连接机制的AILA-Architecture 1，以及实现注意机制以有选择地关注前一层输出的AILA-Architecture 2。这两种架构分别应用于单一任务学习场景中，每个模型独立针对各自任务进行训练。结果证实，AILA的自适应跨层连接通过在多个网络深度灵活重用相关特征，实现了稳健的增益。因此，AILA方法扩展了现有架构，提高了长序列建模、优化计算速度的图像识别以及实际中的最佳分类性能。 

---
# CSPO: Cross-Market Synergistic Stock Price Movement Forecasting with Pseudo-volatility Optimization 

**Title (ZH)**: 跨市场协同股票价格运动预测与伪波动率优化 

**Authors**: Sida Lin, Yankai Chen, Yiyan Qi, Chenhao Ma, Bokai Cao, Yifei Zhang, Xue Liu, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.22740)  

**Abstract**: The stock market, as a cornerstone of the financial markets, places forecasting stock price movements at the forefront of challenges in quantitative finance. Emerging learning-based approaches have made significant progress in capturing the intricate and ever-evolving data patterns of modern markets. With the rapid expansion of the stock market, it presents two characteristics, i.e., stock exogeneity and volatility heterogeneity, that heighten the complexity of price forecasting. Specifically, while stock exogeneity reflects the influence of external market factors on price movements, volatility heterogeneity showcases the varying difficulty in movement forecasting against price fluctuations. In this work, we introduce the framework of Cross-market Synergy with Pseudo-volatility Optimization (CSPO). Specifically, CSPO implements an effective deep neural architecture to leverage external futures knowledge. This enriches stock embeddings with cross-market insights and thus enhances the CSPO's predictive capability. Furthermore, CSPO incorporates pseudo-volatility to model stock-specific forecasting confidence, enabling a dynamic adaptation of its optimization process to improve accuracy and robustness. Our extensive experiments, encompassing industrial evaluation and public benchmarking, highlight CSPO's superior performance over existing methods and effectiveness of all proposed modules contained therein. 

**Abstract (ZH)**: 基于伪波动率优化的跨市场协同框架（CSPO） 

---
# Cyborg Data: Merging Human with AI Generated Training Data 

**Title (ZH)**: 人工增强数据：融合人类与AI生成的训练数据 

**Authors**: Kai North, Christopher Ormerod  

**Link**: [PDF](https://arxiv.org/pdf/2503.22736)  

**Abstract**: Automated scoring (AS) systems used in large-scale assessment have traditionally used small statistical models that require a large quantity of hand-scored data to make accurate predictions, which can be time-consuming and costly. Generative Large Language Models are trained on many tasks and have shown impressive abilities to generalize to new tasks with little to no data. While these models require substantially more computational power to make predictions, they still require some fine-tuning to meet operational standards. Evidence suggests that these models can exceed human-human levels of agreement even when fine-tuned on small amounts of data. With this in mind, we propose a model distillation pipeline in which a large generative model, a Teacher, teaches a much smaller model, a Student. The Teacher, trained on a small subset of the training data, is used to provide scores on the remaining training data, which is then used to train the Student. We call the resulting dataset "Cyborg Data", as it combines human and machine-scored responses. Our findings show that Student models trained on "Cyborg Data" show performance comparable to training on the entire dataset, while only requiring 10% of the original hand-scored data. 

**Abstract (ZH)**: 自动评分（AS）系统在大规模评估中 traditionally 使用小统计模型，这些模型需要大量手工评分数据才能做出准确预测，这可能会耗费大量时间和成本。生成型大规模语言模型在许多任务上进行了训练，并且展示出即使在少量数据下也能泛化到新任务的强大能力。尽管这些模型生成预测所需的计算资源更多，但在优化标准方面仍需一定程度的微调。证据表明，即使在少量数据下微调，这些模型也能够超过人类手工评分的水平。基于此，我们提出了一种模型蒸馏管道，在这种管道中，一个大型生成模型（教师）向一个更小的模型（学生）传授知识。教师在训练数据的小子集中进行训练，用于对剩余训练数据进行评分，然后使用这些评分数据来训练学生。我们将由此产生的数据集称为“半机械人数据”，因为它结合了人类和机器评分的响应。我们的研究结果表明，使用“半机械人数据”训练的学生模型在性能上与使用完整数据集训练的模型相当，但仍只需原始手工评分数据的10%。 

---
# TRIDIS: A Comprehensive Medieval and Early Modern Corpus for HTR and NER 

**Title (ZH)**: TRIDIS: 一个全面的中世纪和早期现代手写文本语料库用于光学字符识别和命名实体识别 

**Authors**: Sergio Torres Aguilar  

**Link**: [PDF](https://arxiv.org/pdf/2503.22714)  

**Abstract**: This paper introduces TRIDIS (Tria Digita Scribunt), an open-source corpus of medieval and early modern manuscripts. TRIDIS aggregates multiple legacy collections (all published under open licenses) and incorporates large metadata descriptions. While prior publications referenced some portions of this corpus, here we provide a unified overview with a stronger focus on its constitution. We describe (i) the narrative, chronological, and editorial background of each major sub-corpus, (ii) its semi-diplomatic transcription rules (expansion, normalization, punctuation), (iii) a strategy for challenging out-of-domain test splits driven by outlier detection in a joint embedding space, and (iv) preliminary baseline experiments using TrOCR and MiniCPM2.5 comparing random and outlier-based test partitions. Overall, TRIDIS is designed to stimulate joint robust Handwritten Text Recognition (HTR) and Named Entity Recognition (NER) research across medieval and early modern textual heritage. 

**Abstract (ZH)**: TRIDIS（Tria Digita Scribunt）：中世纪和早期现代手稿的开源语料库 

---
# Chirp Localization via Fine-Tuned Transformer Model: A Proof-of-Concept Study 

**Title (ZH)**: 基于微调Transformer模型的脉冲定位：一个概念验证研究 

**Authors**: Nooshin Bahador, Milad Lankarany  

**Link**: [PDF](https://arxiv.org/pdf/2503.22713)  

**Abstract**: Spectrograms are pivotal in time-frequency signal analysis, widely used in audio processing and computational neuroscience. Chirp-like patterns in electroencephalogram (EEG) spectrograms (marked by linear or exponential frequency sweep) are key biomarkers for seizure dynamics, but automated tools for their detection, localization, and feature extraction are lacking. This study bridges this gap by fine-tuning a Vision Transformer (ViT) model on synthetic spectrograms, augmented with Low-Rank Adaptation (LoRA) to boost adaptability. We generated 100000 synthetic spectrograms with chirp parameters, creating the first large-scale benchmark for chirp localization. These spectrograms mimic neural chirps using linear or exponential frequency sweep, Gaussian noise, and smoothing. A ViT model, adapted for regression, predicted chirp parameters. LoRA fine-tuned the attention layers, enabling efficient updates to the pre-trained backbone. Training used MSE loss and the AdamW optimizer, with a learning rate scheduler and early stopping to curb overfitting. Only three features were targeted: Chirp Start Time (Onset Time), Chirp Start Frequency (Onset Frequency), and Chirp End Frequency (Offset Frequency). Performance was evaluated via Pearson correlation between predicted and actual labels. Results showed strong alignment: 0.9841 correlation for chirp start time, with stable inference times (137 to 140s) and minimal bias in error distributions. This approach offers a tool for chirp analysis in EEG time-frequency representation, filling a critical methodological void. 

**Abstract (ZH)**: Spectrogram分析中基于Vision Transformer的 chirp疑似波检测与特征提取方法 

---
# Modeling speech emotion with label variance and analyzing performance across speakers and unseen acoustic conditions 

**Title (ZH)**: 基于标签方差建模语音情感并分析 Across Speakers 和未见声学条件下性能 

**Authors**: Vikramjit Mitra, Amrit Romana, Dung T. Tran, Erdrin Azemi  

**Link**: [PDF](https://arxiv.org/pdf/2503.22711)  

**Abstract**: Spontaneous speech emotion data usually contain perceptual grades where graders assign emotion score after listening to the speech files. Such perceptual grades introduce uncertainty in labels due to grader opinion variation. Grader variation is addressed by using consensus grades as groundtruth, where the emotion with the highest vote is selected. Consensus grades fail to consider ambiguous instances where a speech sample may contain multiple emotions, as captured through grader opinion uncertainty. We demonstrate that using the probability density function of the emotion grades as targets instead of the commonly used consensus grades, provide better performance on benchmark evaluation sets compared to results reported in the literature. We show that a saliency driven foundation model (FM) representation selection helps to train a state-of-the-art speech emotion model for both dimensional and categorical emotion recognition. Comparing representations obtained from different FMs, we observed that focusing on overall test-set performance can be deceiving, as it fails to reveal the models generalization capacity across speakers and gender. We demonstrate that performance evaluation across multiple test-sets and performance analysis across gender and speakers are useful in assessing usefulness of emotion models. Finally, we demonstrate that label uncertainty and data-skew pose a challenge to model evaluation, where instead of using the best hypothesis, it is useful to consider the 2- or 3-best hypotheses. 

**Abstract (ZH)**: 自发语音情感数据通常包含感知等级，评分者在听取语音文件后为其赋予情感得分。这种感知等级由于评分者的观点差异而引入标签不确定性。通过使用共识等级作为ground truth，其中情感得分最高者当选，解决了评分者差异问题。但共识等级未能考虑模糊实例，即语音样本可能包含多种情感，这些通过评分者意见的不确定性表现出来。我们证明，将情感分数的概率密度函数作为目标，而不是通常使用的共识等级，可以在基准评估集上获得更好的性能，优于文献报道的结果。我们展示了情感驱动的基础模型（FM）表示选择有助于训练最先进的语音情感模型，用于情感维度和类别识别。比较不同FM获得的表示，我们观察到关注整体测试集性能可能是误导的，因为它未能揭示模型在说话人和性别方面的一般化能力。我们证明，跨多个测试集的性能评估和性别、说话人层面的性能分析是有用的评估情感模型的手段。最后，我们证明，标签不确定性与数据倾斜对模型评估构成挑战，使用前两个或前三个假设比使用最佳假设更有用。 

---
# Validating Emergency Department Admission Predictions Based on Local Data Through MIMIC-IV 

**Title (ZH)**: 基于本地数据通过MIMIC-IV验证急诊住院预测 

**Authors**: Francesca Meimeti, Loukas Triantafyllopoulos, Aikaterini Sakagianni, Vasileios Kaldis, Lazaros Tzelves, Nikolaos Theodorakis, Evgenia Paxinou, Georgios Feretzakis, Dimitris Kalles, Vassilios S. Verykios  

**Link**: [PDF](https://arxiv.org/pdf/2503.22706)  

**Abstract**: The effective management of Emergency Department (ED) overcrowding is essential for improving patient outcomes and optimizing healthcare resource allocation. This study validates hospital admission prediction models initially developed using a small local dataset from a Greek hospital by leveraging the comprehensive MIMIC-IV dataset. After preprocessing the MIMIC-IV data, five algorithms were evaluated: Linear Discriminant Analysis (LDA), K-Nearest Neighbors (KNN), Random Forest (RF), Recursive Partitioning and Regression Trees (RPART), and Support Vector Machines (SVM Radial). Among these, RF demonstrated superior performance, achieving an Area Under the Receiver Operating Characteristic Curve (AUC-ROC) of 0.9999, sensitivity of 0.9997, and specificity of 0.9999 when applied to the MIMIC-IV data. These findings highlight the robustness of RF in handling complex datasets for admission prediction, establish MIMIC-IV as a valuable benchmark for validating models based on smaller local datasets, and provide actionable insights for improving ED management strategies. 

**Abstract (ZH)**: 有效管理急诊部（ED）过度拥挤对于改善患者结果和优化医疗卫生资源分配至关重要。本研究利用希腊医院的小规模本地数据集初次开发的住院预测模型，并借助全面的MIMIC-IV数据集进行验证。经过预处理的MIMIC-IV数据后，评估了五种算法：线性判别分析（LDA）、K近邻（KNN）、随机森林（RF）、递归分区和回归树（RPART）和支持向量机（SVM径向基）。其中，RF表现出色，应用于MIMIC-IV数据时，达到接收者操作特征曲线下的面积（AUC-ROC）为0.9999、敏感性为0.9997和特异性为0.9999。这些发现强调了RF在处理复杂数据集进行住院预测中的稳健性，确立了MIMIC-IV作为基于较小本地数据集验证模型的重要基准，并提供了改善急诊部管理策略的实际见解。 

---
# Enhancing nonnative speech perception and production through an AI-powered application 

**Title (ZH)**: 通过AI赋能应用增强非母语者的语音感知与生产能力 

**Authors**: Georgios P. Georgiou  

**Link**: [PDF](https://arxiv.org/pdf/2503.22705)  

**Abstract**: While research on using Artificial Intelligence (AI) through various applications to enhance foreign language pronunciation is expanding, it has primarily focused on aspects such as comprehensibility and intelligibility, largely neglecting the improvement of individual speech sounds in both perception and production. This study seeks to address this gap by examining the impact of training with an AI-powered mobile application on nonnative sound perception and production. Participants completed a pretest assessing their ability to discriminate the second language English heed-hid contrast and produce these vowels in sentence contexts. The intervention involved training with the Speakometer mobile application, which incorporated recording tasks featuring the English vowels, along with pronunciation feedback and practice. The posttest mirrored the pretest to measure changes in performance. The results revealed significant improvements in both discrimination accuracy and production of the target contrast following the intervention. However, participants did not achieve native-like competence. These findings highlight the effectiveness of AI-powered applications in facilitating speech acquisition and support their potential use for personalized, interactive pronunciation training beyond the classroom. 

**Abstract (ZH)**: 通过人工智能移动应用训练提高非母语声学感知与生产的研究 

---
# Bridging Language Models and Financial Analysis 

**Title (ZH)**: 语言模型与金融分析的桥梁 

**Authors**: Alejandro Lopez-Lira, Jihoon Kwon, Sangwoon Yoon, Jy-yong Sohn, Chanyeol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2503.22693)  

**Abstract**: The rapid advancements in Large Language Models (LLMs) have unlocked transformative possibilities in natural language processing, particularly within the financial sector. Financial data is often embedded in intricate relationships across textual content, numerical tables, and visual charts, posing challenges that traditional methods struggle to address effectively. However, the emergence of LLMs offers new pathways for processing and analyzing this multifaceted data with increased efficiency and insight. Despite the fast pace of innovation in LLM research, there remains a significant gap in their practical adoption within the finance industry, where cautious integration and long-term validation are prioritized. This disparity has led to a slower implementation of emerging LLM techniques, despite their immense potential in financial applications. As a result, many of the latest advancements in LLM technology remain underexplored or not fully utilized in this domain. This survey seeks to bridge this gap by providing a comprehensive overview of recent developments in LLM research and examining their applicability to the financial sector. Building on previous survey literature, we highlight several novel LLM methodologies, exploring their distinctive capabilities and their potential relevance to financial data analysis. By synthesizing insights from a broad range of studies, this paper aims to serve as a valuable resource for researchers and practitioners, offering direction on promising research avenues and outlining future opportunities for advancing LLM applications in finance. 

**Abstract (ZH)**: 大型语言模型的快速进步为自然语言处理带来了变革性可能性，特别是在金融领域。金融数据通常嵌入在文本内容、数字表格和视觉图表的复杂关系中，这给传统方法带来了挑战。然而，大型语言模型的出现为高效且深入地处理和分析这种多维度数据提供了新的途径。尽管大型语言模型研究的创新步伐迅速，但在金融行业中，谨慎的集成和长期验证仍是优先事项，这导致了显著的实际应用差距。尽管新兴的大型语言模型技术具有巨大的金融应用潜力，但其应用实施仍然较为缓慢。因此，许多最新在大型语言模型技术的最新进展在该领域仍被未尽探索或未充分利用。本次综述旨在通过提供大型语言模型研究的全面概述，并探讨其在金融领域的适用性来弥补这一差距。基于之前的综述文献，本文突出了一些新颖的大型语言模型方法，探讨了它们的独特能力及其在金融数据分析中的潜在相关性。通过综合广泛的学术研究洞察，本文旨在为研究者和从业者提供有价值的资源，指明有前景的研究方向，并展望未来大型语言模型在金融领域的应用机会。 

---
# Enhancing Aviation Communication Transcription: Fine-Tuning Distil-Whisper with LoRA 

**Title (ZH)**: 增强航空通信转录：基于LoRA微调Distil-Whisper 

**Authors**: Shokoufeh Mirzaei, Jesse Arzate, Yukti Vijay  

**Link**: [PDF](https://arxiv.org/pdf/2503.22692)  

**Abstract**: Transcription of aviation communications has several applications, from assisting air traffic controllers in identifying the accuracy of read-back errors to search and rescue operations. Recent advances in artificial intelligence have provided unprecedented opportunities for improving aviation communication transcription tasks. OpenAI's Whisper is one of the leading automatic speech recognition models. However, fine-tuning Whisper for aviation communication transcription is not computationally efficient. Thus, this paper aims to use a Parameter-Efficient Fine-tuning method called Low-Rank Adaptation to fine-tune a more computationally efficient version of Whisper, distil-Whisper. To perform the fine-tuning, we used the Air Traffic Control Corpus dataset from the Linguistic Data Consortium, which contains approximately 70 hours of controller and pilot transmissions near three major airports in the US. The objective was to reduce the word error rate to enhance accuracy in the transcription of aviation communication. First, starting with an initial set of hyperparameters for LoRA (Alpha = 64 and Rank = 32), we performed a grid search. We applied a 5-fold cross-validation to find the best combination of distil-Whisper hyperparameters. Then, we fine-tuned the model for LoRA hyperparameters, achieving an impressive average word error rate of 3.86% across five folds. This result highlights the model's potential for use in the cockpit. 

**Abstract (ZH)**: 航空通信转录具有多种应用，从协助空中交通管制员识别复诵错误的准确性到搜索救援行动。最近人工智能的进步为改善航空通信转录任务提供了前所未有的机会。OpenAI的Whisper是领先的自动语音识别模型之一。然而，将Whisper细调以适应航空通信转录在计算上不够高效。因此，本文旨在使用Parameter-Efficient Fine-tuning方法中的Low-Rank Adaptation技术来细调一个计算上更高效的Whisper版本distil-Whisper。为了进行细调，我们使用了Linguistic Data Consortium提供的Air Traffic Control Corpus数据集，该数据集包含约70小时的在美国三大机场附近的管制员和飞行员的通话记录。目标是降低字错误率，以提高航空通信转录的准确性。首先，我们采用初始LoRA超参数设置（Alpha = 64，Rank = 32）进行了网格搜索，并采用5折交叉验证来找到distil-Whisper的最佳超参数组合。然后，我们对LoRA超参数进行了模型细调，实现了令人印象深刻的整体平均字错误率3.86%。这一结果突显了该模型在驾驶舱中的应用潜力。 

---
# Qieemo: Speech Is All You Need in the Emotion Recognition in Conversations 

**Title (ZH)**: Qieemo: 话语即一切——在对话情感识别中无关紧要 

**Authors**: Jinming Chen, Jingyi Fang, Yuanzhong Zheng, Yaoxuan Wang, Haojun Fei  

**Link**: [PDF](https://arxiv.org/pdf/2503.22687)  

**Abstract**: Emotion recognition plays a pivotal role in intelligent human-machine interaction systems. Multimodal approaches benefit from the fusion of diverse modalities, thereby improving the recognition accuracy. However, the lack of high-quality multimodal data and the challenge of achieving optimal alignment between different modalities significantly limit the potential for improvement in multimodal approaches. In this paper, the proposed Qieemo framework effectively utilizes the pretrained automatic speech recognition (ASR) model backbone which contains naturally frame aligned textual and emotional features, to achieve precise emotion classification solely based on the audio modality. Furthermore, we design the multimodal fusion (MMF) module and cross-modal attention (CMA) module in order to fuse the phonetic posteriorgram (PPG) and emotional features extracted by the ASR encoder for improving recognition accuracy. The experimental results on the IEMOCAP dataset demonstrate that Qieemo outperforms the benchmark unimodal, multimodal, and self-supervised models with absolute improvements of 3.0%, 1.2%, and 1.9% respectively. 

**Abstract (ZH)**: 情感识别在智能人机交互系统中发挥着关键作用。多模态方法通过融合多种模态数据，从而提高识别准确性。然而，高质量多模态数据的缺乏以及不同模态之间达到最佳对齐的挑战极大地限制了多模态方法的改进潜力。本文提出的Qieemo框架有效利用了预训练的自动语音识别(ASR)模型骨干，该模型包含自然帧对齐的文本和情感特征，仅基于音频模态实现精确的情感分类。此外，我们设计了多模态融合(MMF)模块和跨模态注意力(CMA)模块，以结合ASR编码器提取的音素后验图(PPG)和情感特征，从而提高识别准确性。在IEMOCAP数据集上的实验结果表明，Qieemo在基准单模态、多模态和自监督模型中的绝对改进分别为3.0%、1.2%和1.9%。 

---
# Binary and Multi-Class Intrusion Detection in IoT Using Standalone and Hybrid Machine and Deep Learning Models 

**Title (ZH)**: 基于独立和混合机器与深度学习模型的物联网二分类和多分类入侵检测 

**Authors**: Md Ahnaf Akif  

**Link**: [PDF](https://arxiv.org/pdf/2503.22684)  

**Abstract**: Maintaining security in IoT systems depends on intrusion detection since these networks' sensitivity to cyber-attacks is growing. Based on the IoT23 dataset, this study explores the use of several Machine Learning (ML) and Deep Learning (DL) along with the hybrid models for binary and multi-class intrusion detection. The standalone machine and deep learning models like Random Forest (RF), Extreme Gradient Boosting (XGBoost), Artificial Neural Network (ANN), K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Convolutional Neural Network (CNN) were used. Furthermore, two hybrid models were created by combining machine learning techniques: RF, XGBoost, AdaBoost, KNN, and SVM and these hybrid models were voting based hybrid classifier. Where one is for binary, and the other one is for multi-class classification. These models vi were tested using precision, recall, accuracy, and F1-score criteria and compared the performance of each model. This work thoroughly explains how hybrid, standalone ML and DL techniques could improve IDS (Intrusion Detection System) in terms of accuracy and scalability in IoT (Internet of Things). 

**Abstract (ZH)**: 基于IoT23数据集的机器学习与深度学习及其混合模型在IoT系统中的二元与多类入侵检测研究：提高物联网安全的IDS性能与可扩展性 

---
