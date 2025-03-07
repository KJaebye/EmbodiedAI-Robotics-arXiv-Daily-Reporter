# Data-augmented Learning of Geodesic Distances in Irregular Domains through Soner Boundary Conditions 

**Title (ZH)**: 通过Sonner边界条件在不规则域中增强数据学习测地距离 

**Authors**: Rafael I. Cabral Muchacho, Florian T. Pokorny  

**Link**: [PDF](https://arxiv.org/pdf/2503.04579)  

**Abstract**: Geodesic distances play a fundamental role in robotics, as they efficiently encode global geometric information of the domain. Recent methods use neural networks to approximate geodesic distances by solving the Eikonal equation through physics-informed approaches. While effective, these approaches often suffer from unstable convergence during training in complex environments. We propose a framework to learn geodesic distances in irregular domains by using the Soner boundary condition, and systematically evaluate the impact of data losses on training stability and solution accuracy. Our experiments demonstrate that incorporating data losses significantly improves convergence robustness, reducing training instabilities and sensitivity to initialization. These findings suggest that hybrid data-physics approaches can effectively enhance the reliability of learning-based geodesic distance solvers with sparse data. 

**Abstract (ZH)**: 不规则域中基于Soner边条件学习测地距离的框架及数据损失对训练稳定性和解的准确性的系统评估 

---
# Equivariant Filter Design for Range-only SLAM 

**Title (ZH)**: 只范围测程SLAM的等变滤波器设计 

**Authors**: Yixiao Ge, Arthur Pearce, Pieter van Goor, Robert Mahony  

**Link**: [PDF](https://arxiv.org/pdf/2503.03973)  

**Abstract**: Range-only Simultaneous Localisation and Mapping (RO-SLAM) is of interest due to its practical applications in ultra-wideband (UWB) and Bluetooth Low Energy (BLE) localisation in terrestrial and aerial applications and acoustic beacon localisation in submarine applications. In this work, we consider a mobile robot equipped with an inertial measurement unit (IMU) and a range sensor that measures distances to a collection of fixed landmarks. We derive an equivariant filter (EqF) for the RO-SLAM problem based on a symmetry Lie group that is compatible with the range measurements. The proposed filter does not require bootstrapping or initialisation of landmark positions, and demonstrates robustness to the no-prior situation. The filter is demonstrated on a real-world dataset, and it is shown to significantly outperform a state-of-the-art EKF alternative in terms of both accuracy and robustness. 

**Abstract (ZH)**: 仅距离同时定位与建图（RO-SLAM）在超宽带（UWB）和蓝牙低功耗（BLE）定位及水下声学信标定位等领域的实际应用中引起了兴趣。在此工作中，我们考虑配备惯性测量单元（IMU）和距离传感器的移动机器人，该传感器测量到一组固定地标点的距离。我们基于与距离测量相兼容的对称李群推导了仅距离同时定位与建图问题的不变滤波器（EqF）。所提出的滤波器不需要地标位置的回环检测或初始化，并展示了在无先验情况下具有良好的鲁棒性。该滤波器在真实数据集上进行了演示，并且在准确性和鲁棒性方面均显著优于最先进的扩展卡尔曼滤波器（EKF）替代方案。 

---
# Multi-Agent Inverse Q-Learning from Demonstrations 

**Title (ZH)**: 基于演示的多agent逆Q学习 

**Authors**: Nathaniel Haynam, Adam Khoja, Dhruv Kumar, Vivek Myers, Erdem Bıyık  

**Link**: [PDF](https://arxiv.org/pdf/2503.04679)  

**Abstract**: When reward functions are hand-designed, deep reinforcement learning algorithms often suffer from reward misspecification, causing them to learn suboptimal policies in terms of the intended task objectives. In the single-agent case, inverse reinforcement learning (IRL) techniques attempt to address this issue by inferring the reward function from expert demonstrations. However, in multi-agent problems, misalignment between the learned and true objectives is exacerbated due to increased environment non-stationarity and variance that scales with multiple agents. As such, in multi-agent general-sum games, multi-agent IRL algorithms have difficulty balancing cooperative and competitive objectives. To address these issues, we propose Multi-Agent Marginal Q-Learning from Demonstrations (MAMQL), a novel sample-efficient framework for multi-agent IRL. For each agent, MAMQL learns a critic marginalized over the other agents' policies, allowing for a well-motivated use of Boltzmann policies in the multi-agent context. We identify a connection between optimal marginalized critics and single-agent soft-Q IRL, allowing us to apply a direct, simple optimization criterion from the single-agent domain. Across our experiments on three different simulated domains, MAMQL significantly outperforms previous multi-agent methods in average reward, sample efficiency, and reward recovery by often more than 2-5x. We make our code available at this https URL . 

**Abstract (ZH)**: 多代理边际Q学习从示范中学习（MAMQL）：一种新的多代理逆强化学习高效框架 

---
# Accelerating Focal Search in Multi-Agent Path Finding with Tighter Lower Bounds 

**Title (ZH)**: 基于更紧的下界加速多智能体路径规划中的焦点搜索 

**Authors**: Yimin Tang, Zhenghong Yu, Jiaoyang Li, Sven Koenig  

**Link**: [PDF](https://arxiv.org/pdf/2503.03779)  

**Abstract**: Multi-Agent Path Finding (MAPF) involves finding collision-free paths for multiple agents while minimizing a cost function--an NP-hard problem. Bounded suboptimal methods like Enhanced Conflict-Based Search (ECBS) and Explicit Estimation CBS (EECBS) balance solution quality with computational efficiency using focal search mechanisms. While effective, traditional focal search faces a limitation: the lower bound (LB) value determining which nodes enter the FOCAL list often increases slowly in early search stages, resulting in a constrained search space that delays finding valid solutions. In this paper, we propose a novel bounded suboptimal algorithm, double-ECBS (DECBS), to address this issue by first determining the maximum LB value and then employing a best-first search guided by this LB to find a collision-free path. Experimental results demonstrate that DECBS outperforms ECBS in most test cases and is compatible with existing optimization techniques. DECBS can reduce nearly 30% high-level CT nodes and 50% low-level focal search nodes. When agent density is moderate to high, DECBS achieves a 23.5% average runtime improvement over ECBS with identical suboptimality bounds and optimizations. 

**Abstract (ZH)**: 多智能体路径规划中的双ECBS算法：一种新的有界次优方法 

---
# ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making 

**Title (ZH)**: 价值航标：一种 duas 阶段框架，用于价值驱动的决策制定 

**Authors**: Yitong Luo, Hou Hei Lam, Ziang Chen, Zhenliang Zhang, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.04569)  

**Abstract**: Despite recent advances in artificial intelligence (AI), it poses challenges to ensure personalized decision-making in tasks that are not considered in training datasets. To address this issue, we propose ValuePilot, a two-phase value-driven decision-making framework comprising a dataset generation toolkit DGT and a decision-making module DMM trained on the generated data. DGT is capable of generating scenarios based on value dimensions and closely mirroring real-world tasks, with automated filtering techniques and human curation to ensure the validity of the dataset. In the generated dataset, DMM learns to recognize the inherent values of scenarios, computes action feasibility and navigates the trade-offs between multiple value dimensions to make personalized decisions. Extensive experiments demonstrate that, given human value preferences, our DMM most closely aligns with human decisions, outperforming Claude-3.5-Sonnet, Gemini-2-flash, Llama-3.1-405b and GPT-4o. This research is a preliminary exploration of value-driven decision-making. We hope it will stimulate interest in value-driven decision-making and personalized decision-making within the community. 

**Abstract (ZH)**: 尽管近年来人工智能取得了进展，但确保在未包含在训练数据集中的任务中实现个性化决策仍然面临挑战。为应对这一问题，我们提出了一种两阶段的价值导向决策框架ValuePilot，该框架包括一个数据生成工具包DGT和一个基于生成数据训练的决策模块DMM。DGT能够基于价值维度生成情景，并紧密结合实际任务，采用自动化过滤技术和人工校勘以确保数据集的有效性。在生成的数据集中，DMM学习识别情景内在的价值，计算行动可行性，并在多个价值维度之间权衡以做出个性化决策。广泛实验表明，给定人类的价值偏好，我们的DMM最接近人类决策，优于Claude-3.5-Sonnet、Gemini-2-flash、Llama-3.1-405b和GPT-4o。本研究是对价值导向决策的初步探索，我们希望它能激发社区对价值导向决策和个性化决策的兴趣。 

---
# Dynamic Pricing for On-Demand DNN Inference in the Edge-AI Market 

**Title (ZH)**: 基于边缘AI市场的按需DNN推理动态定价 

**Authors**: Songyuan Li, Jia Hu, Geyong Min, Haojun Huang, Jiwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04521)  

**Abstract**: The convergence of edge computing and AI gives rise to Edge-AI, which enables the deployment of real-time AI applications and services at the network edge. One of the fundamental research issues in Edge-AI is edge inference acceleration, which aims to realize low-latency high-accuracy DNN inference services by leveraging the fine-grained offloading of partitioned inference tasks from end devices to edge servers. However, existing research has yet to adopt a practical Edge-AI market perspective, which would systematically explore the personalized inference needs of AI users (e.g., inference accuracy, latency, and task complexity), the revenue incentives for AI service providers that offer edge inference services, and multi-stakeholder governance within a market-oriented context. To bridge this gap, we propose an Auction-based Edge Inference Pricing Mechanism (AERIA) for revenue maximization to tackle the multi-dimensional optimization problem of DNN model partition, edge inference pricing, and resource allocation. We investigate the multi-exit device-edge synergistic inference scheme for on-demand DNN inference acceleration, and analyse the auction dynamics amongst the AI service providers, AI users and edge infrastructure provider. Owing to the strategic mechanism design via randomized consensus estimate and cost sharing techniques, the Edge-AI market attains several desirable properties, including competitiveness in revenue maximization, incentive compatibility, and envy-freeness, which are crucial to maintain the effectiveness, truthfulness, and fairness of our auction outcomes. The extensive simulation experiments based on four representative DNN inference workloads demonstrate that our AERIA mechanism significantly outperforms several state-of-the-art approaches in revenue maximization, demonstrating the efficacy of AERIA for on-demand DNN inference in the Edge-AI market. 

**Abstract (ZH)**: 边缘计算与人工智能的融合催生了边缘人工智能（Edge-AI），使其能够在网络边缘部署实时AI应用和服务。边缘人工智能研究中的一个基本问题是边缘推断加速，旨在通过将分区推断任务从终端设备卸载到边缘服务器来实现低延迟高准确度的DNN推断服务。然而，现有研究尚未从实践的边缘人工智能市场视角出发，系统地探讨AI用户个性化推断需求（如推断准确性、延迟和任务复杂度）、提供边缘推断服务的AI服务提供商的收入激励机制，以及市场导向背景下多利益相关者的治理问题。为弥补这一差距，我们提出了一种基于拍卖的边缘推断定价机制（AERIA）以实现收入最大化，并解决DNN模型分区、边缘推断定价和资源分配的多维优化问题。我们研究了多出口设备-边缘协同推断方案以满足按需的DNN推断加速，并分析了AI服务提供商、AI用户和边缘基础设施提供商之间的拍卖动态。通过随机一致性估计和成本分摊等机制设计技术，边缘人工智能市场获得了包括收入最大化竞争力、激励相容性和无嫉妒性在内的多个理想属性，这对保持我们的拍卖结果的有效性、真实性和公平性至关重要。基于四个代表性的DNN推断负载的广泛仿真实验表明，我们的AERIA机制在收入最大化方面显著优于几种最先进的方法，证明了AERIA在边缘人工智能市场按需DNN推断中的有效性。 

---
# From Idea to CAD: A Language Model-Driven Multi-Agent System for Collaborative Design 

**Title (ZH)**: 从理念到CAD：基于语言模型的多agent系统协作设计 

**Authors**: Felix Ocker, Stefan Menzel, Ahmed Sadik, Thiago Rios  

**Link**: [PDF](https://arxiv.org/pdf/2503.04417)  

**Abstract**: Creating digital models using Computer Aided Design (CAD) is a process that requires in-depth expertise. In industrial product development, this process typically involves entire teams of engineers, spanning requirements engineering, CAD itself, and quality assurance. We present an approach that mirrors this team structure with a Vision Language Model (VLM)-based Multi Agent System, with access to parametric CAD tooling and tool documentation. Combining agents for requirements engineering, CAD engineering, and vision-based quality assurance, a model is generated automatically from sketches and/ or textual descriptions. The resulting model can be refined collaboratively in an iterative validation loop with the user. Our approach has the potential to increase the effectiveness of design processes, both for industry experts and for hobbyists who create models for 3D printing. We demonstrate the potential of the architecture at the example of various design tasks and provide several ablations that show the benefits of the architecture's individual components. 

**Abstract (ZH)**: 使用计算机辅助设计（CAD）创建数字模型是一个需要深厚专业知识的过程。在工业产品开发中，这一过程通常涉及整个工程师团队，涵盖需求工程、CAD本身和质量保证。我们提出了一种镜像这种团队结构的方法，采用基于视觉语言模型（VLM）的多智能体系统，并具有参数化CAD工具和工具文档的访问权限。结合需求工程、CAD工程和基于视觉的质量保证智能体，可以从草图和/或文本描述自动生成模型。生成的模型可以与用户在迭代验证循环中进行协作性细化。我们的方法有望提高设计过程的有效性，无论是对于工业专家还是对于创建3D打印模型的爱好者。我们通过各种设计任务的例子展示了该架构的潜力，并提供了几个消融实验，展示了架构各个组件的好处。 

---
# Guidelines for Applying RL and MARL in Cybersecurity Applications 

**Title (ZH)**: RL和MARL在网络安全应用中的应用指南 

**Authors**: Vasilios Mavroudis, Gregory Palmer, Sara Farmer, Kez Smithson Whitehead, David Foster, Adam Price, Ian Miles, Alberto Caron, Stephen Pasteris  

**Link**: [PDF](https://arxiv.org/pdf/2503.04262)  

**Abstract**: Reinforcement Learning (RL) and Multi-Agent Reinforcement Learning (MARL) have emerged as promising methodologies for addressing challenges in automated cyber defence (ACD). These techniques offer adaptive decision-making capabilities in high-dimensional, adversarial environments. This report provides a structured set of guidelines for cybersecurity professionals and researchers to assess the suitability of RL and MARL for specific use cases, considering factors such as explainability, exploration needs, and the complexity of multi-agent coordination. It also discusses key algorithmic approaches, implementation challenges, and real-world constraints, such as data scarcity and adversarial interference. The report further outlines open research questions, including policy optimality, agent cooperation levels, and the integration of MARL systems into operational cybersecurity frameworks. By bridging theoretical advancements and practical deployment, these guidelines aim to enhance the effectiveness of AI-driven cyber defence strategies. 

**Abstract (ZH)**: 强化学习（RL）和多代理强化学习（MARL）已成为应对自动网络防御挑战（ACD）的有前景的方法论。这些技术提供了在高维、对抗性环境中的适应性决策能力。本报告为网络安全专业人员和研究人员提供了一套结构化的指南，以评估RL和MARL在特定应用场景中的适用性，考虑的因素包括可解释性、探索需求以及多代理协调的复杂性。报告还讨论了关键算法方法、实施挑战以及实际约束条件，如数据稀缺性和对抗性干扰。此外，报告概述了开放的研究问题，包括策略最优性、代理合作水平以及MARL系统在运营网络安全框架中的集成。通过融合理论进步和实践部署，这些指南旨在增强基于AI的网络防御策略的有效性。 

---
# Artificial Intelligence in Pronunciation Teaching: Use and Beliefs of Foreign Language Teachers 

**Title (ZH)**: 人工智能在语音教学中的应用与外语教师的信念 

**Authors**: Georgios P. Georgiou  

**Link**: [PDF](https://arxiv.org/pdf/2503.04128)  

**Abstract**: Pronunciation instruction in foreign language classrooms has often been an overlooked area of focus. With the widespread adoption of Artificial Intelligence (AI) and its potential benefits, investigating how AI is utilized in pronunciation teaching and understanding the beliefs of teachers about this tool is essential for improving learning outcomes. This study aims to examine how AI use for pronunciation instruction varies across different demographic and professional factors among teachers, and how these factors, including AI use, influence the beliefs of teachers about AI. The study involved 117 English as a Foreign Language (EFL) in-service teachers working in Cyprus, who completed an online survey designed to assess their beliefs about the effectiveness of AI, its drawbacks, and their willingness to integrate AI into their teaching practices. The results revealed that teachers were significantly more likely to agree on the perceived effectiveness of AI and their willingness to adopt it, compared to their concerns about its use. Furthermore, teachers working in higher education and adult education, as well as those who had received more extensive training, reported using AI more frequently in their teaching. Teachers who utilized AI more often expressed stronger agreement with its effectiveness, while those who had received more training were less likely to express concerns about its integration. Given the limited training that many teachers currently receive, these findings demonstrate the need for tailored training sessions that address the specific needs and concerns of educators, ultimately fostering the adoption of AI in pronunciation instruction. 

**Abstract (ZH)**: 外国语言课堂中的发音教学指令往往是一个被忽视的研究领域。随着人工智能（AI）的广泛应用及其潜在益处，探讨AI在发音教学中的使用方式以及教师对这一工具的看法对于改善学习成果至关重要。本研究旨在探究不同人口统计学和专业因素下教师在发音教学中使用AI的差异，并分析这些因素，包括AI使用情况，如何影响教师对AI的看法。研究对象为来自塞浦路斯的117名英语作为外语（EFL）在职教师，他们完成了旨在评估其对AI有效性的看法、其局限性及其在教学中整合AI意愿的在线问卷。结果表明，教师们更倾向于认为AI的有效性，并愿意采用它，而不是对其使用的担忧。此外，从事高等教育和成人教育的教师以及接受过更广泛培训的教师报告称在教学中更频繁地使用AI。经常使用AI的教师更认同其有效性，而接受过更多培训的教师则较少对AI的整合表达担忧。鉴于教师当前接受的有限培训，这些发现表明需要针对教育工作者的具体需求和担忧定制培训课程，最终促进AI在发音教学中的应用。 

---
# SED2AM: Solving Multi-Trip Time-Dependent Vehicle Routing Problem using Deep Reinforcement Learning 

**Title (ZH)**: SED2AM：使用深度强化学习解决多趟时间依赖车辆路线问题 

**Authors**: Arash Mozhdehi, Yunli Wang, Sun Sun, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04085)  

**Abstract**: Deep reinforcement learning (DRL)-based frameworks, featuring Transformer-style policy networks, have demonstrated their efficacy across various vehicle routing problem (VRP) variants. However, the application of these methods to the multi-trip time-dependent vehicle routing problem (MTTDVRP) with maximum working hours constraints -- a pivotal element of urban logistics -- remains largely unexplored. This paper introduces a DRL-based method called the Simultaneous Encoder and Dual Decoder Attention Model (SED2AM), tailored for the MTTDVRP with maximum working hours constraints. The proposed method introduces a temporal locality inductive bias to the encoding module of the policy networks, enabling it to effectively account for the time-dependency in travel distance or time. The decoding module of SED2AM includes a vehicle selection decoder that selects a vehicle from the fleet, effectively associating trips with vehicles for functional multi-trip routing. Additionally, this decoding module is equipped with a trip construction decoder leveraged for constructing trips for the vehicles. This policy model is equipped with two classes of state representations, fleet state and routing state, providing the information needed for effective route construction in the presence of maximum working hours constraints. Experimental results using real-world datasets from two major Canadian cities not only show that SED2AM outperforms the current state-of-the-art DRL-based and metaheuristic-based baselines but also demonstrate its generalizability to solve larger-scale problems. 

**Abstract (ZH)**: 基于深强化学习（DRL）且采用Transformer风格策略网络的框架在各种车辆路线问题（VRP）变体中展现了其有效性。然而，这些方法在具有最大工作时间约束的多趟时间依赖车辆路线问题（MTTDVRP）中的应用——这是城市物流的关键要素——仍然鲜有探索。本文提出了一种名为Simultaneous Encoder and Dual Decoder Attention Model（SED2AM）的基于DRL的方法，专门针对具有最大工作时间约束的MTTDVRP。所提出的方法在策略网络的编码模块中引入了时间局部性归纳偏置，使其能够有效地考虑旅行距离或时间的时间依赖性。SED2AM的解码模块包括一个车辆选择解码器，可以有效地将趟次与车辆关联起来实现功能性的多趟路由，同时还配备了用于为车辆构建趟次的行程构建解码器。该策略模型配备了两类状态表示——车队状态和路由状态——提供了在存在最大工作时间约束情况下有效构建路线所需的信息。使用来自两个主要加拿大城市的现实世界数据集进行的实验结果不仅表明SED2AM在当前最先进的基于DRL和元启发式的基线方法中具有优越性，还展示了其解决更大规模问题的普适性。 

---
# Learning to Negotiate via Voluntary Commitment 

**Title (ZH)**: 通过自愿承诺学习谈判 

**Authors**: Shuhui Zhu, Baoxiang Wang, Sriram Ganapathi Subramanian, Pascal Poupart  

**Link**: [PDF](https://arxiv.org/pdf/2503.03866)  

**Abstract**: The partial alignment and conflict of autonomous agents lead to mixed-motive scenarios in many real-world applications. However, agents may fail to cooperate in practice even when cooperation yields a better outcome. One well known reason for this failure comes from non-credible commitments. To facilitate commitments among agents for better cooperation, we define Markov Commitment Games (MCGs), a variant of commitment games, where agents can voluntarily commit to their proposed future plans. Based on MCGs, we propose a learnable commitment protocol via policy gradients. We further propose incentive-compatible learning to accelerate convergence to equilibria with better social welfare. Experimental results in challenging mixed-motive tasks demonstrate faster empirical convergence and higher returns for our method compared with its counterparts. Our code is available at this https URL. 

**Abstract (ZH)**: 自主代理的部分对齐与冲突导致许多实际应用中的混合动机场景。即使合作能够产生更好的结果，代理在实践中也可能无法实现合作。这种失败的一个已知原因是不可信的承诺。为促进代理之间的承诺以实现更好的合作，我们定义了马尔可夫承诺博弈（MCGs），这是一种承诺博弈的变体，其中代理可以自愿承诺其提出的未来计划。基于MCGs，我们提出了一种通过策略梯度学习的可学习承诺协议。我们进一步提出了一种激励相容学习方法，以加速收敛并获得更好的社会效益。实验结果表明，与同类方法相比，我们的方法在具有挑战性的混合动机任务中实现了更快的实际收敛和更高的回报。我们的代码可在以下链接获取：this https URL。 

---
# Predicting Team Performance from Communications in Simulated Search-and-Rescue 

**Title (ZH)**: 从模拟搜索救援中通信预测团队绩效 

**Authors**: Ali Jalal-Kamali, Nikolos Gurney, David Pynadath  

**Link**: [PDF](https://arxiv.org/pdf/2503.03791)  

**Abstract**: Understanding how individual traits influence team performance is valuable, but these traits are not always directly observable. Prior research has inferred traits like trust from behavioral data. We analyze conversational data to identify team traits and their correlation with teaming outcomes. Using transcripts from a Minecraft-based search-and-rescue experiment, we apply topic modeling and clustering to uncover key interaction patterns. Our findings show that variations in teaming outcomes can be explained through these inferences, with different levels of predictive power derived from individual traits and team dynamics. 

**Abstract (ZH)**: 理解个体特质如何影响团队表现是有价值的，但这些特质并不总是直接可观测的。先前的研究通过行为数据推断出诸如信任之类的特质。我们分析对话数据以识别团队特质及其与团队成果的相关性。使用基于 Minecraft 的搜索和救援实验的转录数据，我们应用主题建模和聚类来发现关键的交互模式。我们的研究结果表明，团队成果的变化可以通过这些推断来解释，个体特质和团队动态的不同水平可以预测这些变化。 

---
# Scaling Rich Style-Prompted Text-to-Speech Datasets 

**Title (ZH)**: 基于丰富风格提示的文本转语音数据集扩展 

**Authors**: Anuj Diwan, Zhisheng Zheng, David Harwath, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2503.04713)  

**Abstract**: We introduce Paralinguistic Speech Captions (ParaSpeechCaps), a large-scale dataset that annotates speech utterances with rich style captions. While rich abstract tags (e.g. guttural, nasal, pained) have been explored in small-scale human-annotated datasets, existing large-scale datasets only cover basic tags (e.g. low-pitched, slow, loud). We combine off-the-shelf text and speech embedders, classifiers and an audio language model to automatically scale rich tag annotations for the first time. ParaSpeechCaps covers a total of 59 style tags, including both speaker-level intrinsic tags and utterance-level situational tags. It consists of 342 hours of human-labelled data (PSC-Base) and 2427 hours of automatically annotated data (PSC-Scaled). We finetune Parler-TTS, an open-source style-prompted TTS model, on ParaSpeechCaps, and achieve improved style consistency (+7.9% Consistency MOS) and speech quality (+15.5% Naturalness MOS) over the best performing baseline that combines existing rich style tag datasets. We ablate several of our dataset design choices to lay the foundation for future work in this space. Our dataset, models and code are released at this https URL . 

**Abstract (ZH)**: 我们介绍Paralinguistic Speech Captions（Paralinguistic SpeechCaps），一个大规模数据集，用于用丰富的风格注释标注语音片段。尽管在小规模的人工标注数据集中已经探索了丰富的抽象标签（例如喉音、鼻音、痛苦），但现有的大规模数据集只涵盖了基本标签（例如低音、缓慢、大声）。我们将即用型文本和语音嵌入器、分类器以及音频语言模型结合，首次自动扩展丰富的标签注释。Paralinguistic SpeechCaps 包括总计59种风格标签，涵盖演讲者级别的固有标签和片段级别的情境标签。该数据集包含342小时的人工标注数据（PSC-Base）和2427小时的自动标注数据（PSC-Scaled）。我们针对Paralinguistic SpeechCaps微调一个开源的风格提示音合成模型Parler-TTS，并在综合现有丰富风格标签数据集的基线上实现了更好的风格一致性（+7.9%一致性MOS）和语音质量（+15.5%自然度MOS）。我们通过消除部分数据集设计选择为基础未来在此领域的工作奠定基础。我们的数据集、模型和代码在此链接发布：this https URL。 

---
# Self-Supervised Models for Phoneme Recognition: Applications in Children's Speech for Reading Learning 

**Title (ZH)**: 自监督模型在儿童语音识别中的应用：阅读学习中的发音识别 

**Authors**: Lucas Block Medin, Thomas Pellegrini, Lucile Gelin  

**Link**: [PDF](https://arxiv.org/pdf/2503.04710)  

**Abstract**: Child speech recognition is still an underdeveloped area of research due to the lack of data (especially on non-English languages) and the specific difficulties of this task. Having explored various architectures for child speech recognition in previous work, in this article we tackle recent self-supervised models. We first compare wav2vec 2.0, HuBERT and WavLM models adapted to phoneme recognition in French child speech, and continue our experiments with the best of them, WavLM base+. We then further adapt it by unfreezing its transformer blocks during fine-tuning on child speech, which greatly improves its performance and makes it significantly outperform our base model, a Transformer+CTC. Finally, we study in detail the behaviour of these two models under the real conditions of our application, and show that WavLM base+ is more robust to various reading tasks and noise levels. Index Terms: speech recognition, child speech, self-supervised learning 

**Abstract (ZH)**: 儿童语音识别仍是一个由于数据不足（尤其是非英语语言数据）和此任务的具体困难而未充分开发的研究领域。在先前工作中探索了各种儿童语音识别架构后，本文着眼于近期的自监督模型。我们首先比较了适用于法语儿童语音音素识别的wav2vec 2.0、HuBERT和WavLM模型，并在这些模型中选择最优者WavLM base+ 进行进一步实验。接着，在儿童语音识别上解冻其Transformer块进行微调，这大大提高了模型性能，使其显著优于我们的基模型Transformer+CTC。最后，我们在实际应用条件下详细研究了这两种模型的行为，并表明WavLM base+ 对各种阅读任务和噪声水平具有更高的鲁棒性。关键词：语音识别，儿童语音，自监督学习 

---
# Matrix Factorization for Inferring Associations and Missing Links 

**Title (ZH)**: 矩阵因子分解用于推断关联和缺失链接 

**Authors**: Ryan Barron, Maksim E. Eren, Duc P. Truong, Cynthia Matuszek, James Wendelberger, Mary F. Dorn, Boian Alexandrov  

**Link**: [PDF](https://arxiv.org/pdf/2503.04680)  

**Abstract**: Missing link prediction is a method for network analysis, with applications in recommender systems, biology, social sciences, cybersecurity, information retrieval, and Artificial Intelligence (AI) reasoning in Knowledge Graphs. Missing link prediction identifies unseen but potentially existing connections in a network by analyzing the observed patterns and relationships. In proliferation detection, this supports efforts to identify and characterize attempts by state and non-state actors to acquire nuclear weapons or associated technology - a notoriously challenging but vital mission for global security. Dimensionality reduction techniques like Non-Negative Matrix Factorization (NMF) and Logistic Matrix Factorization (LMF) are effective but require selection of the matrix rank parameter, that is, of the number of hidden features, k, to avoid over/under-fitting. We introduce novel Weighted (WNMFk), Boolean (BNMFk), and Recommender (RNMFk) matrix factorization methods, along with ensemble variants incorporating logistic factorization, for link prediction. Our methods integrate automatic model determination for rank estimation by evaluating stability and accuracy using a modified bootstrap methodology and uncertainty quantification (UQ), assessing prediction reliability under random perturbations. We incorporate Otsu threshold selection and k-means clustering for Boolean matrix factorization, comparing them to coordinate descent-based Boolean thresholding. Our experiments highlight the impact of rank k selection, evaluate model performance under varying test-set sizes, and demonstrate the benefits of UQ for reliable predictions using abstention. We validate our methods on three synthetic datasets (Boolean and uniformly distributed) and benchmark them against LMF and symmetric LMF (symLMF) on five real-world protein-protein interaction networks, showcasing an improved prediction performance. 

**Abstract (ZH)**: 缺失边预测是一种网络分析方法，应用于推荐系统、生物、社会科学、网络安全、信息检索以及知识图谱中的AI推理。缺失边预测通过分析观测模式和关系来识别潜在存在的但未被观察到的网络连接，在 proliferation 检测中支持识别和表征国家及非国家行为体获取核武器或相关技术的企图，这是一项具有挑战性但至关重要的全球安全任务。非负矩阵分解（NMF）和逻辑矩阵分解（LMF）等降维技术有效但需要选择矩阵秩参数，即隐藏特征的数量 k，以避免过拟合或欠拟合。我们引入了带权矩阵分解（WNMFk）、布尔矩阵分解（BNMFk）和推荐矩阵分解（RNMFk）方法及其结合逻辑因子分解的集成变体，用于边预测。我们的方法通过评价稳定性与准确性来实现自动模型选择，利用修改后的Bootstrap方法和不确定性量化（UQ）评估预测可靠性。我们结合了Otsu阈值选择和k-means聚类进行布尔矩阵分解，并将其与基于坐标下降的布尔阈值方法进行了比较。实验结果突出了秩 k 选择的影响，评估了模型性能在不同测试集大小下的表现，并展示了UQ在可靠预测中的优势，通过避免预测来展示其益处。我们在三个合成数据集（布尔型和均匀分布）上验证了这些方法，并使用LMF和对称LMF（symLMF）在五个真实世界蛋白质-蛋白质相互作用网络上进行了基准测试，展示了更好的预测性能。 

---
# IDInit: A Universal and Stable Initialization Method for Neural Network Training 

**Title (ZH)**: IDInit: 一种通用且稳定的神经网络训练初始化方法 

**Authors**: Yu Pan, Chaozheng Wang, Zekai Wu, Qifan Wang, Min Zhang, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04626)  

**Abstract**: Deep neural networks have achieved remarkable accomplishments in practice. The success of these networks hinges on effective initialization methods, which are vital for ensuring stable and rapid convergence during training. Recently, initialization methods that maintain identity transition within layers have shown good efficiency in network training. These techniques (e.g., Fixup) set specific weights to zero to achieve identity control. However, settings of remaining weight (e.g., Fixup uses random values to initialize non-zero weights) will affect the inductive bias that is achieved only by a zero weight, which may be harmful to training. Addressing this concern, we introduce fully identical initialization (IDInit), a novel method that preserves identity in both the main and sub-stem layers of residual networks. IDInit employs a padded identity-like matrix to overcome rank constraints in non-square weight matrices. Furthermore, we show the convergence problem of an identity matrix can be solved by stochastic gradient descent. Additionally, we enhance the universality of IDInit by processing higher-order weights and addressing dead neuron problems. IDInit is a straightforward yet effective initialization method, with improved convergence, stability, and performance across various settings, including large-scale datasets and deep models. 

**Abstract (ZH)**: 深层神经网络在实践中取得了显著成就。这些网络的成功依赖于有效的初始化方法，这些方法对于确保训练期间的稳定和快速收敛至关重要。近年来，在层内保持身份过渡的初始化方法在网络训练中显示出良好的效率。这些技术（例如，Fixup）通过将特定权重设为零来实现身份控制。然而，剩余权重的设置（例如，Fixup 使用随机值初始化非零权重）会影响仅由零权重实现的归纳偏置，这可能对训练有害。为解决这一问题，我们引入了一种新颖的方法——全身份初始化（IDInit），该方法在残差网络的主要层和子层中均保持身份。IDInit 使用填充的身份矩阵来克服非方矩阵的秩约束。此外，我们通过处理高阶权重和解决死亡神经元问题，展示了身份矩阵收敛问题可以通过随机梯度下降解决。同时，我们通过增强IDInit的通用性来处理更广泛的情况。IDInit 是一个简单而有效的初始化方法，具有改进的收敛性、稳定性和性能，在包括大规模数据集和深度模型的各种设置中均适用。 

---
# HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization 

**Title (ZH)**: HybridNorm: 向量稳定且高效的变换器训练通过混合规范化 

**Authors**: Zhijian Zhuo, Yutao Zeng, Ya Wang, Sijun Zhang, Jian Yang, Xiaoqing Li, Xun Zhou, Jinwen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.04598)  

**Abstract**: Transformers have become the de facto architecture for a wide range of machine learning tasks, particularly in large language models (LLMs). Despite their remarkable performance, challenges remain in training deep transformer networks, especially regarding the location of layer normalization. While Pre-Norm structures facilitate easier training due to their more prominent identity path, they often yield suboptimal performance compared to Post-Norm. In this paper, we propose $\textbf{HybridNorm}$, a straightforward yet effective hybrid normalization strategy that integrates the advantages of both Pre-Norm and Post-Norm approaches. Specifically, HybridNorm employs QKV normalization within the attention mechanism and Post-Norm in the feed-forward network (FFN) of each transformer block. This design not only stabilizes training but also enhances performance, particularly in the context of LLMs. Comprehensive experiments in both dense and sparse architectures show that HybridNorm consistently outperforms both Pre-Norm and Post-Norm approaches, achieving state-of-the-art results across various benchmarks. These findings highlight the potential of HybridNorm as a more stable and effective technique for improving the training and performance of deep transformer models. %Code will be made publicly available. Code is available at this https URL. 

**Abstract (ZH)**: HybridNorm：一种结合Pre-Norm和Post-Norm优势的混合归一化策略 

---
# Fundamental Limits of Hierarchical Secure Aggregation with Cyclic User Association 

**Title (ZH)**: 层次安全聚合中循环用户关联的基本界限 

**Authors**: Xiang Zhang, Zhou Li, Kai Wan, Hua Sun, Mingyue Ji, Giuseppe Caire  

**Link**: [PDF](https://arxiv.org/pdf/2503.04564)  

**Abstract**: Secure aggregation is motivated by federated learning (FL) where a cloud server aims to compute an averaged model (i.e., weights of deep neural networks) of the locally-trained models of numerous clients, while adhering to data security requirements. Hierarchical secure aggregation (HSA) extends this concept to a three-layer network, where clustered users communicate with the server through an intermediate layer of relays. In HSA, beyond conventional server security, relay security is also enforced to ensure that the relays remain oblivious to the users' inputs (an abstraction of the local models in FL). Existing study on HSA assumes that each user is associated with only one relay, limiting opportunities for coding across inter-cluster users to achieve efficient communication and key generation. In this paper, we consider HSA with a cyclic association pattern where each user is connected to $B$ consecutive relays in a wrap-around manner. We propose an efficient aggregation scheme which includes a message design for the inputs inspired by gradient coding-a well-known technique for efficient communication in distributed computing-along with a highly nontrivial security key design. We also derive novel converse bounds on the minimum achievable communication and key rates using information-theoretic arguments. 

**Abstract (ZH)**: 基于联邦学习的分层安全聚合：每用户关联连续若干relay的周期性关联模式研究 

---
# STX-Search: Explanation Search for Continuous Dynamic Spatio-Temporal Models 

**Title (ZH)**: STX-Search：连续动态时空模型的解释性搜索 

**Authors**: Saif Anwar, Nathan Griffiths, Thomas Popham, Abhir Bhalerao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04509)  

**Abstract**: Recent improvements in the expressive power of spatio-temporal models have led to performance gains in many real-world applications, such as traffic forecasting and social network modelling. However, understanding the predictions from a model is crucial to ensure reliability and trustworthiness, particularly for high-risk applications, such as healthcare and transport. Few existing methods are able to generate explanations for models trained on continuous-time dynamic graph data and, of these, the computational complexity and lack of suitable explanation objectives pose challenges. In this paper, we propose $\textbf{S}$patio-$\textbf{T}$emporal E$\textbf{X}$planation $\textbf{Search}$ (STX-Search), a novel method for generating instance-level explanations that is applicable to static and dynamic temporal graph structures. We introduce a novel search strategy and objective function, to find explanations that are highly faithful and interpretable. When compared with existing methods, STX-Search produces explanations of higher fidelity whilst optimising explanation size to maintain interpretability. 

**Abstract (ZH)**: 时空解释搜索：适用于静态和动态时空图结构的实例级解释生成方法 

---
# Interpretable Transformation and Analysis of Timelines through Learning via Surprisability 

**Title (ZH)**: 通过学习 Surpriseability 进行可解释的时间线转换与分析 

**Authors**: Osnat Mokryn, Teddy Lazebnik, Hagit Ben Shoshan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04502)  

**Abstract**: The analysis of high-dimensional timeline data and the identification of outliers and anomalies is critical across diverse domains, including sensor readings, biological and medical data, historical records, and global statistics. However, conventional analysis techniques often struggle with challenges such as high dimensionality, complex distributions, and sparsity. These limitations hinder the ability to extract meaningful insights from complex temporal datasets, making it difficult to identify trending features, outliers, and anomalies effectively. Inspired by surprisability -- a cognitive science concept describing how humans instinctively focus on unexpected deviations - we propose Learning via Surprisability (LvS), a novel approach for transforming high-dimensional timeline data. LvS quantifies and prioritizes anomalies in time-series data by formalizing deviations from expected behavior. LvS bridges cognitive theories of attention with computational methods, enabling the detection of anomalies and shifts in a way that preserves critical context, offering a new lens for interpreting complex datasets. We demonstrate the usefulness of LvS on three high-dimensional timeline use cases: a time series of sensor data, a global dataset of mortality causes over multiple years, and a textual corpus containing over two centuries of State of the Union Addresses by U.S. presidents. Our results show that the LvS transformation enables efficient and interpretable identification of outliers, anomalies, and the most variable features along the timeline. 

**Abstract (ZH)**: 高维时间线数据中的异常和异常值分析及其识别在传感器读数、生物医学数据、历史记录和全球统计等领域至关重要。然而，传统的分析技术常常难以应对高维度、复杂分布和稀疏性等挑战。这些限制妨碍了从复杂的时间序列数据中提取有意义的见解，使得准确识别趋势特征、异常值和异常变得困难。受意外程度——这一认知科学概念描述人类如何本能地关注意外偏差的影响，我们提出了基于意外程度的学习（LvS），一种新的高维时间线数据转换方法。LvS通过形式化偏离预期行为的方式量化并优先处理异常事件。LvS将注意的认知理论与计算方法相结合，以保留关键背景的方式检测异常和变化，提供了一种解释复杂数据集的新视角。我们在三个高维时间线应用场景中展示了LvS的有效性：传感器数据的时间序列、多年来的全球死亡原因统计数据以及包含两百多年美国总统国情咨文的文本语料库。结果显示，LvS转换能够高效且可解释地识别异常值、异常以及时间线上最波动的特征。 

---
# TPC: Cross-Temporal Prediction Connection for Vision-Language Model Hallucination Reduction 

**Title (ZH)**: TPC: 不同时间预测连接以减少视觉-语言模型的幻觉 

**Authors**: Chao Wang, Weiwei Fu, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.04457)  

**Abstract**: Vision-language models (VLMs) have achieved remarkable advancements, capitalizing on the impressive capabilities of large language models (LLMs) across diverse tasks. Despite this, a critical challenge known as hallucination occurs when models overconfidently describe objects or attributes absent from the image, a problem exacerbated by the tendency of VLMs to rely on linguistic priors. This limitation reduces model reliability in high-stakes applications. In this work, we have observed the characteristic of logits' continuity consistency enhancement and introduced a straightforward and efficient method, Cross-Temporal Prediction Connection (TPC), designed to enhance the semantic consistency of logits by connecting them temporally across timesteps. TPC amplifies information flow and improves coherence, effectively reducing hallucination. Extensive experiments show that TPC surpasses existing representatives, delivering superior performance in both accuracy and efficiency while maintaining robustness in open-ended text generation tasks. 

**Abstract (ZH)**: Vision-语言模型(VLMs)在利用大型语言模型(LLMs)的多样化任务能力方面取得了显著进展，但在高风险应用中因模型对图像不存在的对象或属性过度自信地描述而引发的语言先验依赖导致幻觉问题依然存在。为解决这一问题，我们观察到了logits连续一致性增强的特点，并提出了一种简单有效的方法——跨时间预测连接(TPC)，旨在通过在时间步之间连接logits来增强语义一致性，从而增强信息流并提高连贯性，有效减少幻觉现象。广泛实验表明，TPC在准确性和效率上优于现有方法，并且在开放文本生成任务中保持了稳健性。 

---
# Privacy Preserving and Robust Aggregation for Cross-Silo Federated Learning in Non-IID Settings 

**Title (ZH)**: 非均衡分布环境下的跨机构联邦学习的隐私保护和鲁棒聚合 

**Authors**: Marco Arazzi, Mert Cihangiroglu, Antonino Nocera  

**Link**: [PDF](https://arxiv.org/pdf/2503.04451)  

**Abstract**: Federated Averaging remains the most widely used aggregation strategy in federated learning due to its simplicity and scalability. However, its performance degrades significantly in non-IID data settings, where client distributions are highly imbalanced or skewed. Additionally, it relies on clients transmitting metadata, specifically the number of training samples, which introduces privacy risks and may conflict with regulatory frameworks like the European GDPR. In this paper, we propose a novel aggregation strategy that addresses these challenges by introducing class-aware gradient masking. Unlike traditional approaches, our method relies solely on gradient updates, eliminating the need for any additional client metadata, thereby enhancing privacy protection. Furthermore, our approach validates and dynamically weights client contributions based on class-specific importance, ensuring robustness against non-IID distributions, convergence prevention, and backdoor attacks. Extensive experiments on benchmark datasets demonstrate that our method not only outperforms FedAvg and other widely accepted aggregation strategies in non-IID settings but also preserves model integrity in adversarial scenarios. Our results establish the effectiveness of gradient masking as a practical and secure solution for federated learning. 

**Abstract (ZH)**: 联邦平均仍然是联邦学习中最广泛使用的聚合策略，由于其简单性和可扩展性。然而，在非-IID数据设置中，其性能显著下降，尤其是在客户端分布高度不平衡或偏斜的情况下。此外，它依赖于客户端传输元数据，特别是训练样本的数量，这引入了隐私风险，并可能与GDPR等监管框架冲突。在本文中，我们提出了一种新的聚合策略，通过引入类别感知梯度遮蔽来应对这些挑战。与传统方法不同，我们的方法仅依赖于梯度更新，消除了对任何额外客户端元数据的需求，从而增强隐私保护。此外，我们的方法根据类别特定的重要性验证并动态加权客户端贡献，确保在非-IID分布、收敛抑制和后门攻击方面的鲁棒性。广泛的基准数据集实验表明，与联邦平均及其他广泛接受的聚合策略相比，我们的方法在非-IID设置中不仅性能更优，而且在对抗性场景下也保持了模型的一致性。我们的结果证实了梯度遮蔽作为联邦学习中实用且安全的解决方案的有效性。 

---
# PDX: A Data Layout for Vector Similarity Search 

**Title (ZH)**: PDX：一种向量相似性搜索的数据布局 

**Authors**: Leonardo Kuffo, Elena Krippner, Peter Boncz  

**Link**: [PDF](https://arxiv.org/pdf/2503.04422)  

**Abstract**: We propose Partition Dimensions Across (PDX), a data layout for vectors (e.g., embeddings) that, similar to PAX [6], stores multiple vectors in one block, using a vertical layout for the dimensions (Figure 1). PDX accelerates exact and approximate similarity search thanks to its dimension-by-dimension search strategy that operates on multiple-vectors-at-a-time in tight loops. It beats SIMD-optimized distance kernels on standard horizontal vector storage (avg 40% faster), only relying on scalar code that gets auto-vectorized. We combined the PDX layout with recent dimension-pruning algorithms ADSampling [19] and BSA [52] that accelerate approximate vector search. We found that these algorithms on the horizontal vector layout can lose to SIMD-optimized linear scans, even if they are SIMD-optimized. However, when used on PDX, their benefit is restored to 2-7x. We find that search on PDX is especially fast if a limited number of dimensions has to be scanned fully, which is what the dimension-pruning approaches do. We finally introduce PDX-BOND, an even more flexible dimension-pruning strategy, with good performance on exact search and reasonable performance on approximate search. Unlike previous pruning algorithms, it can work on vector data "as-is" without preprocessing; making it attractive for vector databases with frequent updates. 

**Abstract (ZH)**: PDX：一种用于向量的分区维度布局及其实现方法 

---
# Dedicated Feedback and Edit Models Empower Inference-Time Scaling for Open-Ended General-Domain Tasks 

**Title (ZH)**: 专用于反馈和编辑的模型赋能开放域任务推理时的扩展能力 

**Authors**: Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Daniel Egert, Ellie Evans, Hoo-Chang Shin, Felipe Soares, Yi Dong, Oleksii Kuchaiev  

**Link**: [PDF](https://arxiv.org/pdf/2503.04378)  

**Abstract**: Inference-Time Scaling has been critical to the success of recent models such as OpenAI o1 and DeepSeek R1. However, many techniques used to train models for inference-time scaling require tasks to have answers that can be verified, limiting their application to domains such as math, coding and logical reasoning. We take inspiration from how humans make first attempts, ask for detailed feedback from others and make improvements based on such feedback across a wide spectrum of open-ended endeavors. To this end, we collect data for and train dedicated Feedback and Edit Models that are capable of performing inference-time scaling for open-ended general-domain tasks. In our setup, one model generates an initial response, which are given feedback by a second model, that are then used by a third model to edit the response. We show that performance on Arena Hard, a benchmark strongly predictive of Chatbot Arena Elo can be boosted by scaling the number of initial response drafts, effective feedback and edited responses. When scaled optimally, our setup based on 70B models from the Llama 3 family can reach SoTA performance on Arena Hard at 92.7 as of 5 Mar 2025, surpassing OpenAI o1-preview-2024-09-12 with 90.4 and DeepSeek R1 with 92.3. 

**Abstract (ZH)**: 推理时缩放对于最近的模型如OpenAI o1和DeepSeek R1的成功至关重要。然而，许多用于训练适用于推理时缩放模型的技术需要任务具有可验证的答案，限制了它们在数学、编程和逻辑推理等领域的应用。我们从人类如何进行首次尝试、寻求他人详细反馈并基于此类反馈改进工作的方式中汲取灵感，应用于各种开放性任务中。为此，我们收集数据并训练专门的反馈和编辑模型，这些模型能够处理开放性通用领域任务的推理时缩放。在我们的设置中，一个模型生成初始响应，该响应由第二个模型提供反馈，然后第三个模型基于此类反馈编辑响应。我们展示了通过增加初始响应草稿数量、有效的反馈和编辑响应，Arena Hard基准上的表现可以显著提升，该基准强烈预测着聊天机器人竞技场的Elo排名。基于Llama 3家族70B参数模型的设置，在2025年3月5日达到了92.7的SOTA性能，超过了2024年9月12日的OpenAI o1-preview-2024-09-12的90.4和DeepSeek R1的92.3。 

---
# Causally Reliable Concept Bottleneck Models 

**Title (ZH)**: 因果可靠概念瓶颈模型 

**Authors**: Giovanni De Felice, Arianna Casanova Flores, Francesco De Santis, Silvia Santini, Johannes Schneider, Pietro Barbiero, Alberto Termine  

**Link**: [PDF](https://arxiv.org/pdf/2503.04363)  

**Abstract**: Concept-based models are an emerging paradigm in deep learning that constrains the inference process to operate through human-interpretable concepts, facilitating explainability and human interaction. However, these architectures, on par with popular opaque neural models, fail to account for the true causal mechanisms underlying the target phenomena represented in the data. This hampers their ability to support causal reasoning tasks, limits out-of-distribution generalization, and hinders the implementation of fairness constraints. To overcome these issues, we propose \emph{Causally reliable Concept Bottleneck Models} (C$^2$BMs), a class of concept-based architectures that enforce reasoning through a bottleneck of concepts structured according to a model of the real-world causal mechanisms. We also introduce a pipeline to automatically learn this structure from observational data and \emph{unstructured} background knowledge (e.g., scientific literature). Experimental evidence suggest that C$^2$BM are more interpretable, causally reliable, and improve responsiveness to interventions w.r.t. standard opaque and concept-based models, while maintaining their accuracy. 

**Abstract (ZH)**: 基于概念的因果可靠瓶颈模型（C$^2$BMs）：一种根据现实世界因果机制结构化概念的架构 

---
# A Generalist Cross-Domain Molecular Learning Framework for Structure-Based Drug Discovery 

**Title (ZH)**: 基于结构的药物发现中通用型跨域分子学习框架 

**Authors**: Yiheng Zhu, Mingyang Li, Junlong Liu, Kun Fu, Jiansheng Wu, Qiuyi Li, Mingze Yin, Jieping Ye, Jian Wu, Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04362)  

**Abstract**: Structure-based drug discovery (SBDD) is a systematic scientific process that develops new drugs by leveraging the detailed physical structure of the target protein. Recent advancements in pre-trained models for biomolecules have demonstrated remarkable success across various biochemical applications, including drug discovery and protein engineering. However, in most approaches, the pre-trained models primarily focus on the characteristics of either small molecules or proteins, without delving into their binding interactions which are essential cross-domain relationships pivotal to SBDD. To fill this gap, we propose a general-purpose foundation model named BIT (an abbreviation for Biomolecular Interaction Transformer), which is capable of encoding a range of biochemical entities, including small molecules, proteins, and protein-ligand complexes, as well as various data formats, encompassing both 2D and 3D structures. Specifically, we introduce Mixture-of-Domain-Experts (MoDE) to handle the biomolecules from diverse biochemical domains and Mixture-of-Structure-Experts (MoSE) to capture positional dependencies in the molecular structures. The proposed mixture-of-experts approach enables BIT to achieve both deep fusion and domain-specific encoding, effectively capturing fine-grained molecular interactions within protein-ligand complexes. Then, we perform cross-domain pre-training on the shared Transformer backbone via several unified self-supervised denoising tasks. Experimental results on various benchmarks demonstrate that BIT achieves exceptional performance in downstream tasks, including binding affinity prediction, structure-based virtual screening, and molecular property prediction. 

**Abstract (ZH)**: 基于结构的药物发现（SBDD）是一种通过利用目标蛋白的详细物理结构来开发新药物的系统科学过程。预训练模型在生物分子领域的最新进展已经在包括药物发现和蛋白质工程在内的各种生物化学应用中取得了显著的成功。然而，在大多数方法中，预训练模型主要关注小分子或蛋白质的特点，而忽视了对它们之间相互作用的研究，这些相互作用是SBDD中至关重要的跨域关系。为弥补这一不足，我们提出了一种通用的基础模型Bit（Biomolecular Interaction Transformer的缩写），能够编码包括小分子、蛋白质以及蛋白质-配体复合物在内的多种生物化学实体，并能够处理包括2D和3D结构在内的多种数据格式。具体而言，我们引入了领域专家混合（MoDE）来处理来自不同生物化学领域的生物分子，并引入了结构专家混合（MoSE）来捕获分子结构中的位置依赖性。所提出专家混合的方法使Bit能够实现深层次的融合和领域特定的编码，有效地捕捉蛋白质-配体复合物中的细微分子相互作用。然后，我们通过多个统一的自监督去噪任务在共享的Transformer主干上进行跨域预训练。在各种基准测试上的实验结果表明，Bit在下游任务，包括结合亲和力预测、结构导向的虚拟筛选和分子性质预测中表现出色。 

---
# scDD: Latent Codes Based scRNA-seq Dataset Distillation with Foundation Model Knowledge 

**Title (ZH)**: scDD：基于潜伏代码的单细胞RNA-seq数据集提炼，融合基础模型知识 

**Authors**: Zhen Yu, Jianan Han, Yang Liu, Qingchao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04357)  

**Abstract**: Single-cell RNA sequencing (scRNA-seq) technology has profiled hundreds of millions of human cells across organs, diseases, development and perturbations to date. However, the high-dimensional sparsity, batch effect noise, category imbalance, and ever-increasing data scale of the original sequencing data pose significant challenges for multi-center knowledge transfer, data fusion, and cross-validation between scRNA-seq datasets. To address these barriers, (1) we first propose a latent codes-based scRNA-seq dataset distillation framework named scDD, which transfers and distills foundation model knowledge and original dataset information into a compact latent space and generates synthetic scRNA-seq dataset by a generator to replace the original dataset. Then, (2) we propose a single-step conditional diffusion generator named SCDG, which perform single-step gradient back-propagation to help scDD optimize distillation quality and avoid gradient decay caused by multi-step back-propagation. Meanwhile, SCDG ensures the scRNA-seq data characteristics and inter-class discriminability of the synthetic dataset through flexible conditional control and generation quality assurance. Finally, we propose a comprehensive benchmark to evaluate the performance of scRNA-seq dataset distillation in different data analysis tasks. It is validated that our proposed method can achieve 7.61% absolute and 15.70% relative improvement over previous state-of-the-art methods on average task. 

**Abstract (ZH)**: 单细胞RNA测序数据集蒸馏框架scDD及单步条件扩散生成器SCDG的研究 

---
# Talking Back -- human input and explanations to interactive AI systems 

**Title (ZH)**: 与AI对话：人类对交互式AI系统的输入与解释 

**Authors**: Alan Dix, Tommaso Turchi, Ben Wilson, Anna Monreale, Matt Roach  

**Link**: [PDF](https://arxiv.org/pdf/2503.04343)  

**Abstract**: While XAI focuses on providing AI explanations to humans, can the reverse - humans explaining their judgments to AI - foster richer, synergistic human-AI systems? This paper explores various forms of human inputs to AI and examines how human explanations can guide machine learning models toward automated judgments and explanations that align more closely with human concepts. 

**Abstract (ZH)**: 人类向AI解释判断能否促进更加丰富和协同的人机系统：探索人类输入的各种形式及其引导机器学习模型生成与人类概念更加一致的自动化判断和解释的能力 

---
# Provable Robust Overfitting Mitigation in Wasserstein Distributionally Robust Optimization 

**Title (ZH)**: 可验证鲁棒过拟合缓解在 Wasserstein 分布ally鲁棒优化中的应用 

**Authors**: Shuang Liu, Yihan Wang, Yifan Zhu, Yibo Miao, Xiao-Shan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04315)  

**Abstract**: Wasserstein distributionally robust optimization (WDRO) optimizes against worst-case distributional shifts within a specified uncertainty set, leading to enhanced generalization on unseen adversarial examples, compared to standard adversarial training which focuses on pointwise adversarial perturbations. However, WDRO still suffers fundamentally from the robust overfitting problem, as it does not consider statistical error. We address this gap by proposing a novel robust optimization framework under a new uncertainty set for adversarial noise via Wasserstein distance and statistical error via Kullback-Leibler divergence, called the Statistically Robust WDRO. We establish a robust generalization bound for the new optimization framework, implying that out-of-distribution adversarial performance is at least as good as the statistically robust training loss with high probability. Furthermore, we derive conditions under which Stackelberg and Nash equilibria exist between the learner and the adversary, giving an optimal robust model in certain sense. Finally, through extensive experiments, we demonstrate that our method significantly mitigates robust overfitting and enhances robustness within the framework of WDRO. 

**Abstract (ZH)**: 基于Wasserstein距离和统计误差的统计鲁棒Wasserstein分布鲁棒优化 

---
# How Do Hackathons Foster Creativity? Towards AI Collaborative Evaluation of Creativity at Scale 

**Title (ZH)**: 黑客松如何培养创造力？面向大规模AI协作评估创造力 

**Authors**: Jeanette Falk, Yiyi Chen, Janet Rafner, Mike Zhang, Johannes Bjerva, Alexander Nolte  

**Link**: [PDF](https://arxiv.org/pdf/2503.04290)  

**Abstract**: Hackathons have become popular collaborative events for accelerating the development of creative ideas and prototypes. There are several case studies showcasing creative outcomes across domains such as industry, education, and research. However, there are no large-scale studies on creativity in hackathons which can advance theory on how hackathon formats lead to creative outcomes. We conducted a computational analysis of 193,353 hackathon projects. By operationalizing creativity through usefulness and novelty, we refined our dataset to 10,363 projects, allowing us to analyze how participant characteristics, collaboration patterns, and hackathon setups influence the development of creative projects. The contribution of our paper is twofold: We identified means for organizers to foster creativity in hackathons. We also explore the use of large language models (LLMs) to augment the evaluation of creative outcomes and discuss challenges and opportunities of doing this, which has implications for creativity research at large. 

**Abstract (ZH)**: hackathons在促进创意成果方面的协作事件已成为流行的发展平台：对193,353个hackathon项目的计算分析揭示了参与者特征、协作模式和hackathon设置如何影响创意项目的发展，以及大型语言模型在评估创意成果中的应用及其挑战与机遇。 

---
# Explainable AI in Time-Sensitive Scenarios: Prefetched Offline Explanation Model 

**Title (ZH)**: 时间敏感场景中的可解释AI：预取的离线解释模型 

**Authors**: Fabio Michele Russo, Carlo Metta, Anna Monreale, Salvatore Rinzivillo, Fabio Pinelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.04283)  

**Abstract**: As predictive machine learning models become increasingly adopted and advanced, their role has evolved from merely predicting outcomes to actively shaping them. This evolution has underscored the importance of Trustworthy AI, highlighting the necessity to extend our focus beyond mere accuracy and toward a comprehensive understanding of these models' behaviors within the specific contexts of their applications. To further progress in explainability, we introduce Poem, Prefetched Offline Explanation Model, a model-agnostic, local explainability algorithm for image data. The algorithm generates exemplars, counterexemplars and saliency maps to provide quick and effective explanations suitable for time-sensitive scenarios. Leveraging an existing local algorithm, \poem{} infers factual and counterfactual rules from data to create illustrative examples and opposite scenarios with an enhanced stability by design. A novel mechanism then matches incoming test points with an explanation base and produces diverse exemplars, informative saliency maps and believable counterexemplars. Experimental results indicate that Poem outperforms its predecessor Abele in speed and ability to generate more nuanced and varied exemplars alongside more insightful saliency maps and valuable counterexemplars. 

**Abstract (ZH)**: 随着预测型机器学习模型的日益采用和不断进步，其角色已从 merely 预测结果转变为积极塑造结果。这一演变强调了可信赖人工智能的重要性，凸显了我们需要将关注点从单纯的准确性扩展到对这些模型在其应用场景中的行为进行全面理解的必要性。为进一步推进可解释性，我们介绍了 Poem（预取本地解释模型），这是一种适用于图像数据的模型无关的局部解释算法。该算法生成示例、反例和显著性图，提供快速有效的解释，适用于时间敏感的场景。Poem 利用现有的局部算法从数据中推断事实和反事实规则，通过设计增强稳定性，创建具有说明性的示例和相反场景。一种新型机制将传入的测试点与解释基进行匹配，生成多样化的示例、信息丰富的显著性图和可信的反例。实验结果显示，Poem 在速度上优于其 predecessors Abele，并能够生成更细致多样且更具洞察力的显著性图和更有价值的反例。 

---
# Prompt Programming: A Platform for Dialogue-based Computational Problem Solving with Generative AI Models 

**Title (ZH)**: Prompt编程：一种基于对话的生成型AI模型计算问题求解平台 

**Authors**: Victor-Alexandru Pădurean, Paul Denny, Alkis Gotovos, Adish Singla  

**Link**: [PDF](https://arxiv.org/pdf/2503.04267)  

**Abstract**: Computing students increasingly rely on generative AI tools for programming assistance, often without formal instruction or guidance. This highlights a need to teach students how to effectively interact with AI models, particularly through natural language prompts, to generate and critically evaluate code for solving computational tasks. To address this, we developed a novel platform for prompt programming that enables authentic dialogue-based interactions, supports problems involving multiple interdependent functions, and offers on-request execution of generated code. Data analysis from over 900 students in an introductory programming course revealed high engagement, with the majority of prompts occurring within multi-turn dialogues. Problems with multiple interdependent functions encouraged iterative refinement, with progression graphs highlighting several common strategies. Students were highly selective about the code they chose to test, suggesting that on-request execution of generated code promoted critical thinking. Given the growing importance of learning dialogue-based programming with AI, we provide this tool as a publicly accessible resource, accompanied by a corpus of programming problems for educational use. 

**Abstract (ZH)**: 计算专业学生日益依赖生成式AI工具进行编程辅助，往往缺乏正式的指导或培训。这突显出教授学生如何有效与AI模型互动，尤其是在通过自然语言提示生成和批判性评估代码以解决计算任务方面，的需求。为应对这一挑战，我们开发了一个新颖的提示编程平台，该平台支持真实的基于对话的交互，能够处理涉及多个相互依赖函数的问题，并允许用户请求执行生成的代码。来自超过900名学生的初步编程课程的数据分析显示了高度的参与度，大多数提示发生在多轮对话中。涉及多个相互依赖函数的问题促进了迭代精炼，进度图表突显了多种常见策略。学生们对测试的代码选择性很强，表明请求执行生成的代码促进了批判性思维。鉴于学习基于对话的编程与AI日益重要的地位，我们提供此工具作为公开可访问的资源，并附带一个编程问题语料库用于教育目的。 

---
# TAIL: Text-Audio Incremental Learning 

**Title (ZH)**: TAIL: 文本-音频增量学习 

**Authors**: Yingfei Sun, Xu Gu, Wei Ji, Hanbin Zhao, Hao Fei, Yifang Yin, Roger Zimmermann  

**Link**: [PDF](https://arxiv.org/pdf/2503.04258)  

**Abstract**: Many studies combine text and audio to capture multi-modal information but they overlook the model's generalization ability on new datasets. Introducing new datasets may affect the feature space of the original dataset, leading to catastrophic forgetting. Meanwhile, large model parameters can significantly impact training performance. To address these limitations, we introduce a novel task called Text-Audio Incremental Learning (TAIL) task for text-audio retrieval, and propose a new method, PTAT, Prompt Tuning for Audio-Text incremental learning. This method utilizes prompt tuning to optimize the model parameters while incorporating an audio-text similarity and feature distillation module to effectively mitigate catastrophic forgetting. We benchmark our method and previous incremental learning methods on AudioCaps, Clotho, BBC Sound Effects and Audioset datasets, and our method outperforms previous methods significantly, particularly demonstrating stronger resistance to forgetting on older datasets. Compared to the full-parameters Finetune (Sequential) method, our model only requires 2.42\% of its parameters, achieving 4.46\% higher performance. 

**Abstract (ZH)**: 一种用于文本-音频检索的文本-音频增量学习任务及其方法：Prompt Tuning for Audio-Text Incremental Learning (PTAT) 

---
# One-Shot Clustering for Federated Learning 

**Title (ZH)**: 联邦学习中的一次聚类方法 

**Authors**: Maciej Krzysztof Zuziak, Roberto Pellungrini, Salvatore Rinzivillo  

**Link**: [PDF](https://arxiv.org/pdf/2503.04231)  

**Abstract**: Federated Learning (FL) is a widespread and well adopted paradigm of decentralized learning that allows training one model from multiple sources without the need to directly transfer data between participating clients. Since its inception in 2015, it has been divided into numerous sub-fields that deal with application-specific issues, be it data heterogeneity or resource allocation. One such sub-field, Clustered Federated Learning (CFL), is dealing with the problem of clustering the population of clients into separate cohorts to deliver personalized models. Although few remarkable works have been published in this domain, the problem is still largely unexplored, as its basic assumption and settings are slightly different from standard FL. In this work, we present One-Shot Clustered Federated Learning (OCFL), a clustering-agnostic algorithm that can automatically detect the earliest suitable moment for clustering. Our algorithm is based on the computation of cosine similarity between gradients of the clients and a temperature measure that detects when the federated model starts to converge. We empirically evaluate our methodology by testing various one-shot clustering algorithms for over thirty different tasks on three benchmark datasets. Our experiments showcase the good performance of our approach when used to perform CFL in an automated manner without the need to adjust hyperparameters. 

**Abstract (ZH)**: 联邦学习中的一键聚类联邦学习（One-Shot Clustered Federated Learning） 

---
# Quantum-Inspired Reinforcement Learning in the Presence of Epistemic Ambivalence 

**Title (ZH)**: 量子启发的在认识模态不确定性下的强化学习 

**Authors**: Alireza Habibi, Saeed Ghoorchian, Setareh Maghsudi  

**Link**: [PDF](https://arxiv.org/pdf/2503.04219)  

**Abstract**: The complexity of online decision-making under uncertainty stems from the requirement of finding a balance between exploiting known strategies and exploring new possibilities. Naturally, the uncertainty type plays a crucial role in developing decision-making strategies that manage complexity effectively. In this paper, we focus on a specific form of uncertainty known as epistemic ambivalence (EA), which emerges from conflicting pieces of evidence or contradictory experiences. It creates a delicate interplay between uncertainty and confidence, distinguishing it from epistemic uncertainty that typically diminishes with new information. Indeed, ambivalence can persist even after additional knowledge is acquired. To address this phenomenon, we propose a novel framework, called the epistemically ambivalent Markov decision process (EA-MDP), aiming to understand and control EA in decision-making processes. This framework incorporates the concept of a quantum state from the quantum mechanics formalism, and its core is to assess the probability and reward of every possible outcome. We calculate the reward function using quantum measurement techniques and prove the existence of an optimal policy and an optimal value function in the EA-MDP framework. We also propose the EA-epsilon-greedy Q-learning algorithm. To evaluate the impact of EA on decision-making and the expedience of our framework, we study two distinct experimental setups, namely the two-state problem and the lattice problem. Our results show that using our methods, the agent converges to the optimal policy in the presence of EA. 

**Abstract (ZH)**: 在线不确定性条件下的决策复杂性源于在利用已知策略和探索新可能性之间寻找平衡。自然地，不确定性类型在有效管理这种复杂性方面发挥着关键作用。本文重点关注一种特定形式的不确定性，即源于冲突性证据或矛盾性体验的认识模棱两可（EA）。这种模棱两可在不确定性和信心之间创造出一种微妙的互动，区别于通常随新信息增多而减弱的认识不确定性。事实上，即使获取了更多知识，模棱两可也可能持续存在。为应对这一现象，我们提出了一种新的框架，称为认识模棱两可马尔科夫决策过程（EA-MDP），旨在理解和控制决策过程中认识模棱两可。该框架借鉴了量子力学形式主义中的量子态概念，其核心是评估每种可能结果的概率和回报。我们使用量子测量技术计算回报函数，并证明在EA-MDP框架中存在最优策略和最优价值函数。我们还提出了EA-ε-贪婪Q学习算法。为了评估认识模棱两可对决策的影响以及我们框架的有效性，我们在两种不同的实验设置中进行了研究，即两态问题和格点问题。结果表明，使用我们的方法，在认识模棱两可的情况下，代理能够收敛到最优策略。 

---
# CrowdHMTware: A Cross-level Co-adaptation Middleware for Context-aware Mobile DL Deployment 

**Title (ZH)**: CrowdHMTware: 一种面向上下文感知移动DL部署的跨层次协同适应中间件 

**Authors**: Sicong Liu, Bin Guo, Shiyan Luo, Yuzhan Wang, Hao Luo, Cheng Fang, Yuan Xu, Ke Ma, Yao Li, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04183)  

**Abstract**: There are many deep learning (DL) powered mobile and wearable applications today continuously and unobtrusively sensing the ambient surroundings to enhance all aspects of human this http URL enable robust and private mobile sensing, DL models are often deployed locally on resource-constrained mobile devices using techniques such as model compression or this http URL, existing methods, either front-end algorithm level (i.e. DL model compression/partitioning) or back-end scheduling level (i.e. operator/resource scheduling), cannot be locally online because they require offline retraining to ensure accuracy or rely on manually pre-defined strategies, struggle with dynamic this http URL primary challenge lies in feeding back runtime performance from the back-end level to the front-end level optimization decision. Moreover, the adaptive mobile DL model porting middleware with cross-level co-adaptation is less explored, particularly in mobile environments with diversity and dynamics. In response, we introduce CrowdHMTware, a dynamic context-adaptive DL model deployment middleware for heterogeneous mobile devices. It establishes an automated adaptation loop between cross-level functional components, i.e. elastic inference, scalable offloading, and model-adaptive engine, enhancing scalability and adaptability. Experiments with four typical tasks across 15 platforms and a real-world case study demonstrate that CrowdHMTware can effectively scale DL model, offloading, and engine actions across diverse platforms and tasks. It hides run-time system issues from developers, reducing the required developer expertise. 

**Abstract (ZH)**: CrowdHMTware：一种适应动态上下文的异构移动设备DL模型部署中间件 

---
# Unseen Fake News Detection Through Casual Debiasing 

**Title (ZH)**: 未见假新闻检测通过偶然去bias化 

**Authors**: Shuzhi Gong, Richard Sinnott, Jianzhong Qi, Cecile Paris  

**Link**: [PDF](https://arxiv.org/pdf/2503.04160)  

**Abstract**: The widespread dissemination of fake news on social media poses significant risks, necessitating timely and accurate detection. However, existing methods struggle with unseen news due to their reliance on training data from past events and domains, leaving the challenge of detecting novel fake news largely unresolved. To address this, we identify biases in training data tied to specific domains and propose a debiasing solution FNDCD. Originating from causal analysis, FNDCD employs a reweighting strategy based on classification confidence and propagation structure regularization to reduce the influence of domain-specific biases, enhancing the detection of unseen fake news. Experiments on real-world datasets with non-overlapping news domains demonstrate FNDCD's effectiveness in improving generalization across domains. 

**Abstract (ZH)**: 社交媒体上广泛传播的假新闻带来重大风险，需要及时准确地检测。然而，现有的方法由于依赖过去的事件和领域的训练数据，在检测前所未见的假新闻时遇到了挑战，使得检测新颖假新闻的问题仍未得到充分解决。为应对这一挑战，我们识别了与特定领域相关联的训练数据偏见，并提出了一种去偏方案FNDCD。基于因果分析，FNDCD采用基于分类置信度和传播结构正则化的重新加权策略，以降低领域特定偏见的影响，从而提高对前所未见假新闻的检测能力。实验表明，FNDCD在具有非重叠新闻领域的现实数据集上有效提高了跨领域的泛化能力。 

---
# Robust Multi-View Learning via Representation Fusion of Sample-Level Attention and Alignment of Simulated Perturbation 

**Title (ZH)**: 基于样本级注意力表示融合与模拟扰动对齐的鲁棒多视图学习 

**Authors**: Jie Xu, Na Zhao, Gang Niu, Masashi Sugiyama, Xiaofeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04151)  

**Abstract**: Recently, multi-view learning (MVL) has garnered significant attention due to its ability to fuse discriminative information from multiple views. However, real-world multi-view datasets are often heterogeneous and imperfect, which usually makes MVL methods designed for specific combinations of views lack application potential and limits their effectiveness. To address this issue, we propose a novel robust MVL method (namely RML) with simultaneous representation fusion and alignment. Specifically, we introduce a simple yet effective multi-view transformer fusion network where we transform heterogeneous multi-view data into homogeneous word embeddings, and then integrate multiple views by the sample-level attention mechanism to obtain a fused representation. Furthermore, we propose a simulated perturbation based multi-view contrastive learning framework that dynamically generates the noise and unusable perturbations for simulating imperfect data conditions. The simulated noisy and unusable data obtain two distinct fused representations, and we utilize contrastive learning to align them for learning discriminative and robust representations. Our RML is self-supervised and can also be applied for downstream tasks as a regularization. In experiments, we employ it in unsupervised multi-view clustering, noise-label classification, and as a plug-and-play module for cross-modal hashing retrieval. Extensive comparison experiments and ablation studies validate the effectiveness of RML. 

**Abstract (ZH)**: 最近，由于其能够融合多视角下的判别信息，多视图学习（MVL）受到了广泛关注。然而，现实中的多视图数据通常具有异质性和不完美性，这通常使得针对特定视图组合设计的MVL方法缺乏应用潜力并限制了其效果。为解决这一问题，我们提出了一种新颖的鲁棒多视图学习方法（即RML），并同时实现了表示融合和对齐。具体而言，我们引入了一种简单有效的多视图变压器融合网络，将异质的多视图数据转换为同构的词嵌入，并通过样本级注意力机制整合多个视图以获得融合表示。此外，我们提出了一个基于模拟扰动的多视图对比学习框架，可以动态生成噪声和不可用的扰动以模拟不完美的数据条件。模拟得到的噪声和不可用数据获得两种不同的融合表示，我们利用对比学习对齐它们以学习判别性和鲁棒性的表示。我们的RML是自监督的，并且也可以作为正则化应用于下游任务。在实验中，我们将它应用于无监督多视图聚类、带噪声标签的分类任务，以及跨模态哈希检索的插件模块。广泛的比较实验和消融研究验证了RML的有效性。 

---
# DM-Adapter: Domain-Aware Mixture-of-Adapters for Text-Based Person Retrieval 

**Title (ZH)**: DM-Adapter: 域 aware 混合适配器用于基于文本的人像检索 

**Authors**: Yating Liu, Zimo Liu, Xiangyuan Lan, Wenming Yang, Yaowei Li, Qingmin Liao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04144)  

**Abstract**: Text-based person retrieval (TPR) has gained significant attention as a fine-grained and challenging task that closely aligns with practical applications. Tailoring CLIP to person domain is now a emerging research topic due to the abundant knowledge of vision-language pretraining, but challenges still remain during fine-tuning: (i) Previous full-model fine-tuning in TPR is computationally expensive and prone to overfitting.(ii) Existing parameter-efficient transfer learning (PETL) for TPR lacks of fine-grained feature extraction. To address these issues, we propose Domain-Aware Mixture-of-Adapters (DM-Adapter), which unifies Mixture-of-Experts (MOE) and PETL to enhance fine-grained feature representations while maintaining efficiency. Specifically, Sparse Mixture-of-Adapters is designed in parallel to MLP layers in both vision and language branches, where different experts specialize in distinct aspects of person knowledge to handle features more finely. To promote the router to exploit domain information effectively and alleviate the routing imbalance, Domain-Aware Router is then developed by building a novel gating function and injecting learnable domain-aware prompts. Extensive experiments show that our DM-Adapter achieves state-of-the-art performance, outperforming previous methods by a significant margin. 

**Abstract (ZH)**: 基于文本的人检索（TPR）已成为一项精细且具有挑战性的任务，紧密契合实际应用需求， Tailored CLIP技术在人员领域正成为一项新兴的研究话题，但由于在细调过程中仍存在挑战：（i）TPR中的全模型细调计算成本高，容易过拟合。（ii）现有针对TPR的参数高效转移学习（PETL）缺乏细粒度特征提取。为解决这些问题，我们提出了域感知混合专家适应器（DM-Adapter），将其与PETL统一起来，在增强细粒度特征表示的同时保持效率。具体而言，稀疏混合专家适应器并行设计在视觉和语言分支的MLP层中，不同的专家专注于人员知识的不同方面，以更精细地处理特征。为了促进路由器有效利用领域信息并缓解路由失衡，我们开发了域感知路由器，通过构建新的门控函数并注入可学习的领域感知提示来实现。大量实验表明，我们的DM-Adapter取得了最先进的性能，显著优于先前的方法。 

---
# MTS: A Deep Reinforcement Learning Portfolio Management Framework with Time-Awareness and Short-Selling 

**Title (ZH)**: MTS：一种具有时间意识和卖空的深度 reinforcement 学习投资组合管理框架 

**Authors**: Fengchen Gu, Zhengyong Jiang, Ángel F. García-Fernández, Angelos Stefanidis, Jionglong Su, Huakang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04143)  

**Abstract**: Portfolio management remains a crucial challenge in finance, with traditional methods often falling short in complex and volatile market environments. While deep reinforcement approaches have shown promise, they still face limitations in dynamic risk management, exploitation of temporal markets, and incorporation of complex trading strategies such as short-selling. These limitations can lead to suboptimal portfolio performance, increased vulnerability to market volatility, and missed opportunities in capturing potential returns from diverse market conditions. This paper introduces a Deep Reinforcement Learning Portfolio Management Framework with Time-Awareness and Short-Selling (MTS), offering a robust and adaptive strategy for sustainable investment performance. This framework utilizes a novel encoder-attention mechanism to address the limitations by incorporating temporal market characteristics, a parallel strategy for automated short-selling based on market trends, and risk management through innovative Incremental Conditional Value at Risk, enhancing adaptability and performance. Experimental validation on five diverse datasets from 2019 to 2023 demonstrates MTS's superiority over traditional algorithms and advanced machine learning techniques. MTS consistently achieves higher cumulative returns, Sharpe, Omega, and Sortino ratios, underscoring its effectiveness in balancing risk and return while adapting to market dynamics. MTS demonstrates an average relative increase of 30.67% in cumulative returns and 29.33% in Sharpe ratio compared to the next best-performing strategies across various datasets. 

**Abstract (ZH)**: 具有时间意识和卖空的深度强化学习投资组合管理框架（MTS） 

---
# Simple Self Organizing Map with Visual Transformer 

**Title (ZH)**: 简单自组织映射结合视觉变压器 

**Authors**: Alan Luo, Kaiwen Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04121)  

**Abstract**: Vision Transformers (ViTs) have demonstrated exceptional performance in various vision tasks. However, they tend to underperform on smaller datasets due to their inherent lack of inductive biases. Current approaches address this limitation implicitly-often by pairing ViTs with pretext tasks or by distilling knowledge from convolutional neural networks (CNNs) to strengthen the prior. In contrast, Self-Organizing Maps (SOMs), a widely adopted self-supervised framework, are inherently structured to preserve topology and spatial organization, making them a promising candidate to directly address the limitations of ViTs in limited or small training datasets. Despite this potential, equipping SOMs with modern deep learning architectures remains largely unexplored. In this study, we conduct a novel exploration on how Vision Transformers (ViTs) and Self-Organizing Maps (SOMs) can empower each other, aiming to bridge this critical research gap. Our findings demonstrate that these architectures can synergistically enhance each other, leading to significantly improved performance in both unsupervised and supervised tasks. Code will be publicly available. 

**Abstract (ZH)**: 视觉变换器（ViTs）在各种视觉任务中已经展示了出色的表现。然而，它们在较小的数据集上往往会表现出色不足，这主要是由于其固有的归纳偏置缺乏。当前的方法通常通过将ViTs与先验知识增强的预设任务或从卷积神经网络（CNNs）中提取知识来隐式地解决这一限制。相比之下，自组织映射（SOMs）作为一种广泛采用的自监督框架，天然地保持拓扑和空间组织结构，这使其成为直接解决ViTs在有限或小型训练数据集上的限制的有前途的候选者。尽管如此，如何将现代深度学习架构与SOMs结合仍是一个未被充分探索的领域。在本研究中，我们进行了一种新颖的探索，研究视觉变换器（ViTs）和自组织映射（SOMs）如何相互赋能，旨在填补这一关键的研究空白。我们的研究表明，这些架构可以协同增强彼此，从而在无监督和监督任务中均显著提高性能。代码将公开发布。 

---
# Generalizability of Neural Networks Minimizing Empirical Risk Based on Expressive Ability 

**Title (ZH)**: 基于表征能力最小化经验风险的神经网络泛化性研究 

**Authors**: Lijia Yu, Yibo Miao, Yifan Zhu, Xiao-Shan Gao, Lijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04111)  

**Abstract**: The primary objective of learning methods is generalization. Classic uniform generalization bounds, which rely on VC-dimension or Rademacher complexity, fail to explain the significant attribute that over-parameterized models in deep learning exhibit nice generalizability. On the other hand, algorithm-dependent generalization bounds, like stability bounds, often rely on strict assumptions. To establish generalizability under less stringent assumptions, this paper investigates the generalizability of neural networks that minimize or approximately minimize empirical risk. We establish a lower bound for population accuracy based on the expressiveness of these networks, which indicates that with an adequate large number of training samples and network sizes, these networks, including over-parameterized ones, can generalize effectively. Additionally, we provide a necessary condition for generalization, demonstrating that, for certain data distributions, the quantity of training data required to ensure generalization exceeds the network size needed to represent the corresponding data distribution. Finally, we provide theoretical insights into several phenomena in deep learning, including robust generalization, importance of over-parameterization, and effect of loss function on generalization. 

**Abstract (ZH)**: 学习方法的主要目标是泛化。经典的统一泛化界，如VC维或Rademacher复杂性，未能解释深度学习中过参数化模型表现出的良好泛化能力。另一方面，依赖于算法的泛化界，如稳定性界，通常依赖于严格的假设。为了在较不严格的假设下建立泛化性，本文研究了能够最小化或近似最小化经验风险的神经网络的泛化能力。基于这些网络的表达能力，建立了总体准确率的下界，表明在适当大量训练样本和网络规模的情况下，包括过参数化在内的这些网络能够有效地泛化。此外，我们提供了泛化的必要条件，证明了对于某些数据分布，确保泛化所需的训练数据量超过表示相应数据分布所需的网络规模。最后，本文提供了关于深度学习中robust泛化、过参数化的重要性以及损失函数对泛化的影响的一些理论见解。 

---
# Chart-HQA: A Benchmark for Hypothetical Question Answering in Charts 

**Title (ZH)**: Chart-HQA：图表中的假设性问题回答基准 

**Authors**: Xiangnan Chen, Yuancheng Fang, Qian Xiao, Juncheng Li, Jun Lin, Siliang Tang, Yi Yang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04095)  

**Abstract**: Multimodal Large Language Models (MLLMs) have garnered significant attention for their strong visual-semantic understanding. Most existing chart benchmarks evaluate MLLMs' ability to parse information from charts to answer this http URL, they overlook the inherent output biases of MLLMs, where models rely on their parametric memory to answer questions rather than genuinely understanding the chart content. To address this limitation, we introduce a novel Chart Hypothetical Question Answering (HQA) task, which imposes assumptions on the same question to compel models to engage in counterfactual reasoning based on the chart content. Furthermore, we introduce HAI, a human-AI interactive data synthesis approach that leverages the efficient text-editing capabilities of LLMs alongside human expert knowledge to generate diverse and high-quality HQA data at a low cost. Using HAI, we construct Chart-HQA, a challenging benchmark synthesized from publicly available data sources. Evaluation results on 18 MLLMs of varying model sizes reveal that current models face significant generalization challenges and exhibit imbalanced reasoning performance on the HQA task. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）因其强大的视觉语义理解能力而备受关注。现有大多数图表基准主要评估MLLMs从图表中解析信息以回答问题的能力，但忽视了MLLMs固有的输出偏见，模型依赖参数记忆而不是真正理解图表内容来回答问题。为解决这一局限，我们引入了一种新的图表假设性问题回答（HQA）任务，该任务通过对同一问题进行假设，促使模型基于图表内容进行反事实推理。此外，我们引入了HAI，这是一种结合了LLM高效文本编辑能力和人类专家知识的人机交互式数据合成方法，以低成本生成多样性和高质量的HQA数据。使用HAI，我们构建了Chart-HQA，这是一个从公开数据源合成的具有挑战性的基准。对18个不同模型规模的MLLMs的评估结果表明，当前模型在HQA任务上面临着显著的泛化挑战，并且在推理性能上表现出不平衡。 

---
# Continual Optimization with Symmetry Teleportation for Multi-Task Learning 

**Title (ZH)**: 基于对称 teleportation 的持续优化多任务学习 

**Authors**: Zhipeng Zhou, Ziqiao Meng, Pengcheng Wu, Peilin Zhao, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04046)  

**Abstract**: Multi-task learning (MTL) is a widely explored paradigm that enables the simultaneous learning of multiple tasks using a single model. Despite numerous solutions, the key issues of optimization conflict and task imbalance remain under-addressed, limiting performance. Unlike existing optimization-based approaches that typically reweight task losses or gradients to mitigate conflicts or promote progress, we propose a novel approach based on Continual Optimization with Symmetry Teleportation (COST). During MTL optimization, when an optimization conflict arises, we seek an alternative loss-equivalent point on the loss landscape to reduce conflict. Specifically, we utilize a low-rank adapter (LoRA) to facilitate this practical teleportation by designing convergent, loss-invariant objectives. Additionally, we introduce a historical trajectory reuse strategy to continually leverage the benefits of advanced optimizers. Extensive experiments on multiple mainstream datasets demonstrate the effectiveness of our approach. COST is a plug-and-play solution that enhances a wide range of existing MTL methods. When integrated with state-of-the-art methods, COST achieves superior performance. 

**Abstract (ZH)**: 基于连续优化与对称 teleport 的多任务学习（COST） 

---
# TextDoctor: Unified Document Image Inpainting via Patch Pyramid Diffusion Models 

**Title (ZH)**: 文本医生：基于patch金字塔扩散模型的统一文档图像插补 

**Authors**: Wanglong Lu, Lingming Su, Jingjing Zheng, Vinícius Veloso de Melo, Farzaneh Shoeleh, John Hawkin, Terrence Tricco, Hanli Zhao, Xianta Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04021)  

**Abstract**: Digital versions of real-world text documents often suffer from issues like environmental corrosion of the original document, low-quality scanning, or human interference. Existing document restoration and inpainting methods typically struggle with generalizing to unseen document styles and handling high-resolution images. To address these challenges, we introduce TextDoctor, a novel unified document image inpainting method. Inspired by human reading behavior, TextDoctor restores fundamental text elements from patches and then applies diffusion models to entire document images instead of training models on specific document types. To handle varying text sizes and avoid out-of-memory issues, common in high-resolution documents, we propose using structure pyramid prediction and patch pyramid diffusion models. These techniques leverage multiscale inputs and pyramid patches to enhance the quality of inpainting both globally and locally. Extensive qualitative and quantitative experiments on seven public datasets validated that TextDoctor outperforms state-of-the-art methods in restoring various types of high-resolution document images. 

**Abstract (ZH)**: TextDoctor：一种基于人体阅读行为的统一文档图像 inpainting 方法 

---
# Subgraph Federated Learning for Local Generalization 

**Title (ZH)**: 局部泛化的子图联邦学习 

**Authors**: Sungwon Kim, Yoonho Lee, Yunhak Oh, Namkyeong Lee, Sukwon Yun, Junseok Lee, Sein Kim, Carl Yang, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2503.03995)  

**Abstract**: Federated Learning (FL) on graphs enables collaborative model training to enhance performance without compromising the privacy of each client. However, existing methods often overlook the mutable nature of graph data, which frequently introduces new nodes and leads to shifts in label distribution. Since they focus solely on performing well on each client's local data, they are prone to overfitting to their local distributions (i.e., local overfitting), which hinders their ability to generalize to unseen data with diverse label distributions. In contrast, our proposed method, FedLoG, effectively tackles this issue by mitigating local overfitting. Our model generates global synthetic data by condensing the reliable information from each class representation and its structural information across clients. Using these synthetic data as a training set, we alleviate the local overfitting problem by adaptively generalizing the absent knowledge within each local dataset. This enhances the generalization capabilities of local models, enabling them to handle unseen data effectively. Our model outperforms baselines in our proposed experimental settings, which are designed to measure generalization power to unseen data in practical scenarios. Our code is available at this https URL 

**Abstract (ZH)**: 图上的联邦学习（FL）能够通过协作训练模型来提升性能，同时不牺牲每个客户端的隐私。然而，现有方法往往忽视了图数据的可变性，这导致新节点的引入和标签分布的变化。由于它们仅专注于在每个客户端本地数据上的表现，这些方法容易过度拟合本地分布（即局部过拟合），这阻碍了它们向具有不同标签分布的未见数据泛化的能... 

---
# Training neural networks faster with minimal tuning using pre-computed lists of hyperparameters for NAdamW 

**Title (ZH)**: 使用预计算的超参数列表加速NAdamW神经网络训练且无需精细调参 

**Authors**: Sourabh Medapati, Priya Kasimbeg, Shankar Krishnan, Naman Agarwal, George Dahl  

**Link**: [PDF](https://arxiv.org/pdf/2503.03986)  

**Abstract**: If we want to train a neural network using any of the most popular optimization algorithms, we are immediately faced with a dilemma: how to set the various optimization and regularization hyperparameters? When computational resources are abundant, there are a variety of methods for finding good hyperparameter settings, but when resources are limited the only realistic choices are using standard default values of uncertain quality and provenance, or tuning only a couple of the most important hyperparameters via extremely limited handdesigned sweeps. Extending the idea of default settings to a modest tuning budget, Metz et al. (2020) proposed using ordered lists of well-performing hyperparameter settings, derived from a broad hyperparameter search on a large library of training workloads. However, to date, no practical and performant hyperparameter lists that generalize to representative deep learning workloads have been demonstrated. In this paper, we present hyperparameter lists for NAdamW derived from extensive experiments on the realistic workloads in the AlgoPerf: Training Algorithms benchmark. Our hyperparameter lists also include values for basic regularization techniques (i.e. weight decay, label smoothing, and dropout). In particular, our best NAdamW hyperparameter list performs well on AlgoPerf held-out workloads not used to construct it, and represents a compelling turn-key approach to tuning when restricted to five or fewer trials. It also outperforms basic learning rate/weight decay sweeps and an off-the-shelf Bayesian optimization tool when restricted to the same budget. 

**Abstract (ZH)**: 基于AlgoPerf: Training Algorithms基准的实用且性能良好的NAdamW超参数列表 

---
# All-atom Diffusion Transformers: Unified generative modelling of molecules and materials 

**Title (ZH)**: 全原子扩散变换器：分子和材料的一体化生成建模 

**Authors**: Chaitanya K. Joshi, Xiang Fu, Yi-Lun Liao, Vahe Gharakhanyan, Benjamin Kurt Miller, Anuroop Sriram, Zachary W. Ulissi  

**Link**: [PDF](https://arxiv.org/pdf/2503.03965)  

**Abstract**: Diffusion models are the standard toolkit for generative modelling of 3D atomic systems. However, for different types of atomic systems - such as molecules and materials - the generative processes are usually highly specific to the target system despite the underlying physics being the same. We introduce the All-atom Diffusion Transformer (ADiT), a unified latent diffusion framework for jointly generating both periodic materials and non-periodic molecular systems using the same model: (1) An autoencoder maps a unified, all-atom representations of molecules and materials to a shared latent embedding space; and (2) A diffusion model is trained to generate new latent embeddings that the autoencoder can decode to sample new molecules or materials. Experiments on QM9 and MP20 datasets demonstrate that jointly trained ADiT generates realistic and valid molecules as well as materials, exceeding state-of-the-art results from molecule and crystal-specific models. ADiT uses standard Transformers for both the autoencoder and diffusion model, resulting in significant speedups during training and inference compared to equivariant diffusion models. Scaling ADiT up to half a billion parameters predictably improves performance, representing a step towards broadly generalizable foundation models for generative chemistry. Open source code: this https URL 

**Abstract (ZH)**: 全域原子扩散变换器（ADiT）：统一的周期材料与非周期分子系统的联合生成框架 

---
# WIP: Assessing the Effectiveness of ChatGPT in Preparatory Testing Activities 

**Title (ZH)**: WIP: 评估ChatGPT在预备性测试活动中的有效性 

**Authors**: Susmita Haldar, Mary Pierce, Luiz Fernando Capretz  

**Link**: [PDF](https://arxiv.org/pdf/2503.03951)  

**Abstract**: This innovative practice WIP paper describes a research study that explores the integration of ChatGPT into the software testing curriculum and evaluates its effectiveness compared to human-generated testing artifacts. In a Capstone Project course, students were tasked with generating preparatory testing artifacts using ChatGPT prompts, which they had previously created manually. Their understanding and the effectiveness of the Artificial Intelligence generated artifacts were assessed through targeted questions. The results, drawn from this in-class assignment at a North American community college indicate that while ChatGPT can automate many testing preparation tasks, it cannot fully replace human expertise. However, students, already familiar with Information Technology at the postgraduate level, found the integration of ChatGPT into their workflow to be straightforward. The study suggests that AI can be gradually introduced into software testing education to keep pace with technological advancements. 

**Abstract (ZH)**: 这篇创新性的实践WIP论文描述了一项研究，探讨了将ChatGPT整合到软件测试课程中的方法，并评估了其效果，对比了由ChatGPT生成的测试制品与人工生成的测试制品。在一项综合项目课程中，学生被要求使用ChatGPT提示生成预备测试制品，这些提示他们之前是手动创建的。通过对目标问题的回答评估了学生对生成的人工智能制品的理解和效果。这些结果来自北美社区学院的一项课堂作业，表明虽然ChatGPT可以自动化许多测试准备任务，但无法完全替代人类的专业知识。然而，已经具备研究生水平信息技术背景的学生发现将ChatGPT整合到他们的工作流程中是简单的。研究建议可以在软件测试教育中逐步引入人工智能，以适应技术的发展。 

---
# GlucoLens: Explainable Postprandial Blood Glucose Prediction from Diet and Physical Activity 

**Title (ZH)**: GlucoLens：基于饮食和体育活动的可解释餐后血糖预测 

**Authors**: Abdullah Mamun, Asiful Arefeen, Susan B. Racette, Dorothy D. Sears, Corrie M. Whisner, Matthew P. Buman, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2503.03935)  

**Abstract**: Postprandial hyperglycemia, marked by the blood glucose level exceeding the normal range after meals, is a critical indicator of progression toward type 2 diabetes in prediabetic and healthy individuals. A key metric for understanding blood glucose dynamics after eating is the postprandial area under the curve (PAUC). Predicting PAUC in advance based on a person's diet and activity level and explaining what affects postprandial blood glucose could allow an individual to adjust their lifestyle accordingly to maintain normal glucose levels. In this paper, we propose GlucoLens, an explainable machine learning approach to predict PAUC and hyperglycemia from diet, activity, and recent glucose patterns. We conducted a five-week user study with 10 full-time working individuals to develop and evaluate the computational model. Our machine learning model takes multimodal data including fasting glucose, recent glucose, recent activity, and macronutrient amounts, and provides an interpretable prediction of the postprandial glucose pattern. Our extensive analyses of the collected data revealed that the trained model achieves a normalized root mean squared error (NRMSE) of 0.123. On average, GlucoLense with a Random Forest backbone provides a 16% better result than the baseline models. Additionally, GlucoLens predicts hyperglycemia with an accuracy of 74% and recommends different options to help avoid hyperglycemia through diverse counterfactual explanations. Code available: this https URL. 

**Abstract (ZH)**: 餐后高血糖，表现为餐后血糖水平超出正常范围，是预测糖尿病前期和健康个体向2型糖尿病进展的关键指标。理解进食后血糖动态的一个重要指标是餐后曲线下面积（PAUC）。基于个人饮食和活动水平提前预测PAUC，并解释影响餐后血糖的因素，可以使个体根据需要调整生活方式以维持正常的血糖水平。本文提出了一种名为GlucoLens的可解释机器学习方法，用于从饮食、活动和近期血糖模式预测PAUC和高血糖。我们进行了为期五周的用户研究，涉及10名全职工作人员，以开发和评估计算模型。我们的机器学习模型利用空腹血糖、近期血糖、近期活动和宏量营养素含量等多种模态数据，提供可解释的餐后血糖模式预测。通过对收集数据的广泛分析，我们发现训练模型的归一化均方根误差（NRMSE）为0.123。平均而言，基于随机森林的GlucoLens比基线模型提供了16%更好的结果。此外，GlucoLens在高血糖预测方面准确率为74%，并通过多种反事实解释推荐不同的选项以帮助避免高血糖。代码可用：this https URL。 

---
# "Impressively Scary:" Exploring User Perceptions and Reactions to Unraveling Machine Learning Models in Social Media Applications 

**Title (ZH)**: “令人震撼地可怕”：探索用户对揭开社交媒体应用中机器学习模型的感知与反应 

**Authors**: Jack West, Bengisu Cagiltay, Shirley Zhang, Jingjie Li, Kassem Fawaz, Suman Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.03927)  

**Abstract**: Machine learning models deployed locally on social media applications are used for features, such as face filters which read faces in-real time, and they expose sensitive attributes to the apps. However, the deployment of machine learning models, e.g., when, where, and how they are used, in social media applications is opaque to users. We aim to address this inconsistency and investigate how social media user perceptions and behaviors change once exposed to these models. We conducted user studies (N=21) and found that participants were unaware to both what the models output and when the models were used in Instagram and TikTok, two major social media platforms. In response to being exposed to the models' functionality, we observed long term behavior changes in 8 participants. Our analysis uncovers the challenges and opportunities in providing transparency for machine learning models that interact with local user data. 

**Abstract (ZH)**: 本地部署在社交媒体应用上的机器学习模型用于面部过滤等功能，暴露出敏感属性给应用程序。然而，这些模型的部署，如何时、何地以及如何使用，对用户是透明度不足的。我们旨在解决这一不一致性，并调查用户在接触这些模型后对这些模型的看法和行为如何变化。我们进行了用户研究（N=21），发现参与者对Instagram和TikTok这两个主要社交媒体平台上的模型输出和使用时机并不了解。当参与者接触到这些模型的功能时，我们观察到8名参与者出现了长期的行为变化。我们的分析揭示了提供与本地用户数据交互的机器学习模型透明度所面临的挑战与机遇。 

---
# De-skilling, Cognitive Offloading, and Misplaced Responsibilities: Potential Ironies of AI-Assisted Design 

**Title (ZH)**: 技能分流、认知卸载与责任错位：AI辅助设计的潜在讽刺之处 

**Authors**: Prakash Shukla, Phuong Bui, Sean S Levy, Max Kowalski, Ali Baigelenov, Paul Parsons  

**Link**: [PDF](https://arxiv.org/pdf/2503.03924)  

**Abstract**: The rapid adoption of generative AI (GenAI) in design has sparked discussions about its benefits and unintended consequences. While AI is often framed as a tool for enhancing productivity by automating routine tasks, historical research on automation warns of paradoxical effects, such as de-skilling and misplaced responsibilities. To assess UX practitioners' perceptions of AI, we analyzed over 120 articles and discussions from UX-focused subreddits. Our findings indicate that while practitioners express optimism about AI reducing repetitive work and augmenting creativity, they also highlight concerns about over-reliance, cognitive offloading, and the erosion of critical design skills. Drawing from human-automation interaction literature, we discuss how these perspectives align with well-documented automation ironies and function allocation challenges. We argue that UX professionals should critically evaluate AI's role beyond immediate productivity gains and consider its long-term implications for creative autonomy and expertise. This study contributes empirical insights into practitioners' perspectives and links them to broader debates on automation in design. 

**Abstract (ZH)**: Generative AI在设计中的快速采用引发了对其益处和意外后果的讨论。尽管AI常被视为通过自动化常规任务来增强生产力的工具，历史上的自动化研究警告可能产生悖论效应，如技能退化和责任错位。为了评估用户体验从业者对AI的看法，我们分析了来自专注于用户体验的reddit子版块的超过120篇文章和讨论。我们的研究发现，虽然从业者对AI减轻重复工作并增强创造力表达了乐观态度，但他们也指出了过度依赖、认知卸载以及设计关键技能侵蚀的担忧。借鉴人机交互领域的研究文献，我们讨论了这些观点如何与广泛的自动化悖论和功能分配挑战相吻合。我们认为，用户体验专业人士应当批判性地评估AI的role，而不仅仅是其短期生产效率的提升，并考虑其对创造性自主权和专业知识的长期影响。本研究为从业者的观点提供了实证洞察，并将其与设计领域更广泛的自动化辩论联系起来。 

---
# Task-Agnostic Attacks Against Vision Foundation Models 

**Title (ZH)**: 面向任务的视觉基础模型攻击 

**Authors**: Brian Pulfer, Yury Belousov, Vitaliy Kinakh, Teddy Furon, Slava Voloshynovskiy  

**Link**: [PDF](https://arxiv.org/pdf/2503.03842)  

**Abstract**: The study of security in machine learning mainly focuses on downstream task-specific attacks, where the adversarial example is obtained by optimizing a loss function specific to the downstream task. At the same time, it has become standard practice for machine learning practitioners to adopt publicly available pre-trained vision foundation models, effectively sharing a common backbone architecture across a multitude of applications such as classification, segmentation, depth estimation, retrieval, question-answering and more. The study of attacks on such foundation models and their impact to multiple downstream tasks remains vastly unexplored. This work proposes a general framework that forges task-agnostic adversarial examples by maximally disrupting the feature representation obtained with foundation models. We extensively evaluate the security of the feature representations obtained by popular vision foundation models by measuring the impact of this attack on multiple downstream tasks and its transferability between models. 

**Abstract (ZH)**: 机器学习中的安全性研究主要侧重于下游任务特定的攻击，其中恶意样本通过优化特定于下游任务的损失函数获得。同时，机器学习 practitioners 广泛采用公开可用的预训练视觉基础模型，有效地在分类、分割、深度估计、检索、问答等多种应用中共享一个共同的基本架构。对这类基础模型及其对多个下游任务的影响的研究仍严重不足。本工作提出了一种通用框架，通过最大程度地破坏基础模型获得的特征表示来生成任务无关的恶意样本。我们通过测量该攻击对多个下游任务的影响及其在模型间迁移性来全面评估流行视觉基础模型的特征表示安全性。 

---
# VoiceGRPO: Modern MoE Transformers with Group Relative Policy Optimization GRPO for AI Voice Health Care Applications on Voice Pathology Detection 

**Title (ZH)**: VoiceGRPO: 基于组相对策略优化GRPO的现代MoE变压器在语音病理检测中的AI语音健康管理应用 

**Authors**: Enkhtogtokh Togootogtokh, Christian Klasen  

**Link**: [PDF](https://arxiv.org/pdf/2503.03797)  

**Abstract**: This research introduces a novel AI techniques as Mixture-of-Experts Transformers with Group Relative Policy Optimization (GRPO) for voice health care applications on voice pathology detection. With the architectural innovations, we adopt advanced training paradigms inspired by reinforcement learning, namely Proximal Policy Optimization (PPO) and Group-wise Regularized Policy Optimization (GRPO), to enhance model stability and performance. Experiments conducted on a synthetically generated voice pathology dataset demonstrate that our proposed models significantly improve diagnostic accuracy, F1 score, and ROC-AUC compared to conventional approaches. These findings underscore the potential of integrating transformer architectures with novel training strategies to advance automated voice pathology detection and ultimately contribute to more effective healthcare delivery. The code we used to train and evaluate our models is available at this https URL 

**Abstract (ZH)**: 这种研究引入了一种新型AI技术，即Mixture-of-Experts Transformer与Group Relative Policy Optimization (GRPO)方法，用于语音病理检测的语音健康管理应用。通过架构创新，我们采用由强化学习启发的先进训练范式，即Proximal Policy Optimization (PPO)和Group-wise Regularized Policy Optimization (GRPO)，以增强模型稳定性和性能。实验结果表明，与传统方法相比，我们提出的模型显著提高了诊断准确率、F1分数和ROC-AUC。这些发现强调了将变压器架构与新型训练策略集成以推动自动化语音病理检测的潜力，并最终有助于更有效的健康医疗服务。用于训练和评估我们模型的代码可在以下网址获取：this https URL。 

---
# Synthetic Data Augmentation for Enhancing Harmful Algal Bloom Detection with Machine Learning 

**Title (ZH)**: 合成数据增强以提高机器学习在水华检测中的效果 

**Authors**: Tianyi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03794)  

**Abstract**: Harmful Algal Blooms (HABs) pose severe threats to aquatic ecosystems and public health, resulting in substantial economic losses globally. Early detection is crucial but often hindered by the scarcity of high-quality datasets necessary for training reliable machine learning (ML) models. This study investigates the use of synthetic data augmentation using Gaussian Copulas to enhance ML-based HAB detection systems. Synthetic datasets of varying sizes (100-1,000 samples) were generated using relevant environmental features$\unicode{x2015}$water temperature, salinity, and UVB radiation$\unicode{x2015}$with corrected Chlorophyll-a concentration as the target variable. Experimental results demonstrate that moderate synthetic augmentation significantly improves model performance (RMSE reduced from 0.4706 to 0.1850; $p < 0.001$). However, excessive synthetic data introduces noise and reduces predictive accuracy, emphasizing the need for a balanced approach to data augmentation. These findings highlight the potential of synthetic data to enhance HAB monitoring systems, offering a scalable and cost-effective method for early detection and mitigation of ecological and public health risks. 

**Abstract (ZH)**: 合成数据增强在使用高斯copula改善基于机器学习的水华检测系统中的应用：对 aquatic ecosystems 和公众健康的严重威胁要求早期检测，但由于高质量数据集的稀缺性，这常常受到阻碍。本研究探讨了使用高斯copula生成合成数据增强以提高基于机器学习的水华检测系统性能的方法。通过相关环境特征（水温、盐度和UVB辐射）生成大小不同的合成数据集（100-1,000样本），以修正的叶绿素-a浓度为目标变量。实验结果表明，适度的合成数据增强显着提高了模型性能（均方根误差从0.4706降低到0.1850，p<0.001）。然而，过度生成合成数据会引入噪声并降低预测准确性，强调需要在数据增强方面寻求平衡的方法。这些发现突显了合成数据在增强水华监测系统中的潜力，提供了一种可扩展且成本有效的早期检测和减轻生态和公共卫生风险的方法。 

---
# Positive-Unlabeled Diffusion Models for Preventing Sensitive Data Generation 

**Title (ZH)**: 正 unlabeled 扩散模型防止敏感数据生成 

**Authors**: Hiroshi Takahashi, Tomoharu Iwata, Atsutoshi Kumagai, Yuuki Yamanaka, Tomoya Yamashita  

**Link**: [PDF](https://arxiv.org/pdf/2503.03789)  

**Abstract**: Diffusion models are powerful generative models but often generate sensitive data that are unwanted by users, mainly because the unlabeled training data frequently contain such sensitive data. Since labeling all sensitive data in the large-scale unlabeled training data is impractical, we address this problem by using a small amount of labeled sensitive data. In this paper, we propose positive-unlabeled diffusion models, which prevent the generation of sensitive data using unlabeled and sensitive data. Our approach can approximate the evidence lower bound (ELBO) for normal (negative) data using only unlabeled and sensitive (positive) data. Therefore, even without labeled normal data, we can maximize the ELBO for normal data and minimize it for labeled sensitive data, ensuring the generation of only normal data. Through experiments across various datasets and settings, we demonstrated that our approach can prevent the generation of sensitive images without compromising image quality. 

**Abstract (ZH)**: 正 unlabeled 扩散模型：使用未标记的敏感数据防止生成敏感数据 

---
# Sarcasm Detection as a Catalyst: Improving Stance Detection with Cross-Target Capabilities 

**Title (ZH)**: 讽刺检测作为催化剂：跨目标能力提升立场检测 

**Authors**: Gibson Nkhata Shi Yin Hong, Susan Gauch  

**Link**: [PDF](https://arxiv.org/pdf/2503.03787)  

**Abstract**: Stance Detection (SD) has become a critical area of interest due to its applications in various contexts leading to increased research within NLP. Yet the subtlety and complexity of texts sourced from online platforms often containing sarcastic language pose significant challenges for SD algorithms in accurately determining the authors stance. This paper addresses this by employing sarcasm for SD. It also tackles the issue of insufficient annotated data for training SD models on new targets by conducting Cross-Target SD (CTSD). The proposed approach involves fine-tuning BERT and RoBERTa models followed by concatenating additional deep learning layers. The approach is assessed against various State-Of-The-Art baselines for SD demonstrating superior performance using publicly available datasets. Notably our model outperforms the best SOTA models on both in-domain SD and CTSD tasks even before the incorporation of sarcasm-detection pre-training. The integration of sarcasm knowledge into the model significantly reduces misclassifications of sarcastic text elements in SD allowing our model to accurately predict 85% of texts that were previously misclassified without sarcasm-detection pre-training on in-domain SD. This enhancement contributes to an increase in the models average macro F1-score. The CTSD task achieves performance comparable to that of the in-domain task despite using a zero-shot finetuning. We also reveal that the success of the transfer-learning framework relies on the correlation between the lexical attributes of sarcasm detection and SD. This study represents the first exploration of sarcasm detection as an intermediate transfer-learning task within the context of SD while also leveraging the concatenation of BERT or RoBERTa with other deep-learning techniques. The proposed approach establishes a foundational baseline for future research in this domain. 

**Abstract (ZH)**: 立场检测（SD）已成为一个重要研究领域，由于其在各种上下文中的应用，导致NLP领域内的研究不断增加。然而，来源于在线平台的文本往往包含讽刺语言，这给SD算法准确判断作者立场带来了挑战。本文通过引入讽刺进行SD，同时通过跨目标立场检测（CTSD）解决了训练SD模型时标注数据不足的问题。提出的方法包括微调BERT和RoBERTa模型，并附加额外的深度学习层。该方法在各种最新基准方法上进行了评估，展示了在公开数据集上的优越性能。值得注意的是，即使在未引入讽刺检测预训练的情况下，我们的模型在领域内SD和CTSD任务上均优于最新模型的最佳表现。将讽刺知识整合到模型中显著减少了SD中讽刺文本元素的误分类，使得我们的模型在未进行讽刺检测预训练的情况下准确预测了85%的被误分类的文本。这一增强提高了模型的平均宏F1分数。尽管使用零样本微调，CTSD任务仍可达到与领域内任务相似的性能。我们还揭示了转移学习框架成功依赖于讽刺检测和SD之间词法属性的相关性。本文是首个在SD背景下将讽刺检测作为中间转移学习任务进行探索的研究，还利用了BERT或RoBERTa与其他深度学习技术的串联方法。提出的方案为该领域的未来研究奠定了基础。 

---
# Passive Heart Rate Monitoring During Smartphone Use in Everyday Life 

**Title (ZH)**: 日常生活中使用智能手机时的被动心率监测 

**Authors**: Shun Liao, Paolo Di Achille, Jiang Wu, Silviu Borac, Jonathan Wang, Xin Liu, Eric Teasley, Lawrence Cai, Yun Liu, Daniel McDuff, Hao-Wei Su, Brent Winslow, Anupam Pathak, Shwetak Patel, Jameson K. Rogers, Ming-Zher Poh  

**Link**: [PDF](https://arxiv.org/pdf/2503.03783)  

**Abstract**: Resting heart rate (RHR) is an important biomarker of cardiovascular health and mortality, but tracking it longitudinally generally requires a wearable device, limiting its availability. We present PHRM, a deep learning system for passive heart rate (HR) and RHR measurements during everyday smartphone use, using facial video-based photoplethysmography. Our system was developed using 225,773 videos from 495 participants and validated on 185,970 videos from 205 participants in laboratory and free-living conditions, representing the largest validation study of its kind. Compared to reference electrocardiogram, PHRM achieved a mean absolute percentage error (MAPE) < 10% for HR measurements across three skin tone groups of light, medium and dark pigmentation; MAPE for each skin tone group was non-inferior versus the others. Daily RHR measured by PHRM had a mean absolute error < 5 bpm compared to a wearable HR tracker, and was associated with known risk factors. These results highlight the potential of smartphones to enable passive and equitable heart health monitoring. 

**Abstract (ZH)**: 被动心率测量的深学习系统：基于面部视频的光体积描记术在日常智能手机使用中的心率和静息心率监测 

---
# Efficient Finetuning for Dimensional Speech Emotion Recognition in the Age of Transformers 

**Title (ZH)**: 面向Transformer时代的高效细调维度语音情感识别 

**Authors**: Aneesha Sampath, James Tavernor, Emily Mower Provost  

**Link**: [PDF](https://arxiv.org/pdf/2503.03756)  

**Abstract**: Accurate speech emotion recognition is essential for developing human-facing systems. Recent advancements have included finetuning large, pretrained transformer models like Wav2Vec 2.0. However, the finetuning process requires substantial computational resources, including high-memory GPUs and significant processing time. As the demand for accurate emotion recognition continues to grow, efficient finetuning approaches are needed to reduce the computational burden. Our study focuses on dimensional emotion recognition, predicting attributes such as activation (calm to excited) and valence (negative to positive). We present various finetuning techniques, including full finetuning, partial finetuning of transformer layers, finetuning with mixed precision, partial finetuning with caching, and low-rank adaptation (LoRA) on the Wav2Vec 2.0 base model. We find that partial finetuning with mixed precision achieves performance comparable to full finetuning while increasing training speed by 67%. Caching intermediate representations further boosts efficiency, yielding an 88% speedup and a 71% reduction in learnable parameters. We recommend finetuning the final three transformer layers in mixed precision to balance performance and training efficiency, and adding intermediate representation caching for optimal speed with minimal performance trade-offs. These findings lower the barriers to finetuning speech emotion recognition systems, making accurate emotion recognition more accessible to a broader range of researchers and practitioners. 

**Abstract (ZH)**: 准确的语音情感识别对于开发面向人类的系统至关重要。近期进展包括对预训练变换器模型如Wav2Vec 2.0进行微调。然而，微调过程需要大量的计算资源，包括高性能GPU和大量的处理时间。随着对准确情感识别需求的不断增长，需要高效的微调方法以减轻计算负担。本研究侧重于维度情感识别，预测如激活（平静到兴奋）和价值（负向到正向）等属性。我们提出了多种微调技术，包括全程微调、变压器层的部分微调、混合精度微调、带有缓存的部分微调以及低秩适应（LoRA）在Wav2Vec 2.0基模型上的应用。我们发现，部分微调并使用混合精度可以达到与全程微调相当的性能，同时将训练速度提高67%。进一步使用中间表示的缓存进一步提高了效率，使训练速度提升88%，并且使可学习参数减少了71%。我们建议在混合精度下微调最后三层变压器层以平衡性能和训练效率，并通过添加中间表示的缓存来实现最佳速度，同时将性能降低控制在最小范围内。这些发现降低了语音情感识别系统微调的门槛，使准确的情感识别对更广泛的科研人员和实践者更加可及。 

---
# Generative Diffusion Model-based Compression of MIMO CSI 

**Title (ZH)**: 基于生成扩散模型的MIMO CSI压缩 

**Authors**: Heasung Kim, Taekyun Lee, Hyeji Kim, Gustavo De Veciana, Mohamed Amine Arfaoui, Asil Koc, Phil Pietraski, Guodong Zhang, John Kaewell  

**Link**: [PDF](https://arxiv.org/pdf/2503.03753)  

**Abstract**: While neural lossy compression techniques have markedly advanced the efficiency of Channel State Information (CSI) compression and reconstruction for feedback in MIMO communications, efficient algorithms for more challenging and practical tasks-such as CSI compression for future channel prediction and reconstruction with relevant side information-remain underexplored, often resulting in suboptimal performance when existing methods are extended to these scenarios. To that end, we propose a novel framework for compression with side information, featuring an encoding process with fixed-rate compression using a trainable codebook for codeword quantization, and a decoding procedure modeled as a backward diffusion process conditioned on both the codeword and the side information. Experimental results show that our method significantly outperforms existing CSI compression algorithms, often yielding over twofold performance improvement by achieving comparable distortion at less than half the data rate of competing methods in certain scenarios. These findings underscore the potential of diffusion-based compression for practical deployment in communication systems. 

**Abstract (ZH)**: 一种基于侧信息的扩散压缩框架：CSI压缩与重建在未来信道预测中的应用 

---
# Multimodal AI predicts clinical outcomes of drug combinations from preclinical data 

**Title (ZH)**: 多模态AI从预临床数据预测药物组合的临床结果 

**Authors**: Yepeng Huang, Xiaorui Su, Varun Ullanat, Ivy Liang, Lindsay Clegg, Damilola Olabode, Nicholas Ho, Bino John, Megan Gibbs, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2503.02781)  

**Abstract**: Predicting clinical outcomes from preclinical data is essential for identifying safe and effective drug combinations. Current models rely on structural or target-based features to identify high-efficacy, low-toxicity drug combinations. However, these approaches fail to incorporate the multimodal data necessary for accurate, clinically-relevant predictions. Here, we introduce MADRIGAL, a multimodal AI model that learns from structural, pathway, cell viability, and transcriptomic data to predict drug combination effects across 953 clinical outcomes and 21842 compounds, including combinations of approved drugs and novel compounds in development. MADRIGAL uses a transformer bottleneck module to unify preclinical drug data modalities while handling missing data during training and inference--a major challenge in multimodal learning. It outperforms single-modality methods and state-of-the-art models in predicting adverse drug interactions. MADRIGAL performs virtual screening of anticancer drug combinations and supports polypharmacy management for type II diabetes and metabolic dysfunction-associated steatohepatitis (MASH). It identifies transporter-mediated drug interactions. MADRIGAL predicts resmetirom, the first and only FDA-approved drug for MASH, among therapies with the most favorable safety profile. It supports personalized cancer therapy by integrating genomic profiles from cancer patients. Using primary acute myeloid leukemia samples and patient-derived xenograft models, it predicts the efficacy of personalized drug combinations. Integrating MADRIGAL with a large language model allows users to describe clinical outcomes in natural language, improving safety assessment by identifying potential adverse interactions and toxicity risks. MADRIGAL provides a multimodal approach for designing combination therapies with improved predictive accuracy and clinical relevance. 

**Abstract (ZH)**: 从预临床数据预测临床结果对于识别安全有效的药物组合至关重要。现有的模型依赖于结构或靶标特征来识别高效低毒的药物组合。然而，这些方法未能 Incorporate 准确且临床相关的多模态数据预测所需的信息。在这里，我们介绍了 MADRIGAL，这是一种多模态AI模型，它从结构、途径、细胞活力和转录组数据中学习，以预测跨越 953 临床结果和 21842 种化合物（包括已批准药物和正在开发的新型化合物）的药物组合效果。MADRIGAL 使用变压器瓶颈模块在训练和推理过程中统一预临床药物数据模态，解决了多模态学习中的一个重要挑战。它在预测不良药物相互作用方面超过了单模态方法和最先进的模型。MADRIGAL 用于抗癌药物组合的虚拟筛选，并支持 II 型糖尿病和代谢功能障碍相关性脂肪肝炎（MASH）的多药治疗管理。它识别转运体介导的药物相互作用。MADRIGAL 预测了 FDA 批准的第一个也是唯一的用于 MASH 的药物 resmetirom，其安全特性最佳。它通过整合癌症患者的基因组特征支持个性化癌症治疗。使用急性髓系白血病的原代样本和患者衍生的异种移植物模型，它预测个性化药物组合的有效性。将 MADRIGAL 与大型语言模型结合使用可以让用户以自然语言描述临床结果，通过识别潜在的不良相互作用和毒性风险提高安全性评估。MADRIGAL 提供了一种多模态方法，以提高预测准确性和临床相关性来设计组合疗法。 

---
# BIOSCAN-5M: A Multimodal Dataset for Insect Biodiversity 

**Title (ZH)**: BIOSCAN-5M：一种多模态昆虫生物多样性数据集 

**Authors**: Zahra Gharaee, Scott C. Lowe, ZeMing Gong, Pablo Millan Arias, Nicholas Pellegrino, Austin T. Wang, Joakim Bruslund Haurum, Iuliia Zarubiieva, Lila Kari, Dirk Steinke, Graham W. Taylor, Paul Fieguth, Angel X. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2406.12723)  

**Abstract**: As part of an ongoing worldwide effort to comprehend and monitor insect biodiversity, this paper presents the BIOSCAN-5M Insect dataset to the machine learning community and establish several benchmark tasks. BIOSCAN-5M is a comprehensive dataset containing multi-modal information for over 5 million insect specimens, and it significantly expands existing image-based biological datasets by including taxonomic labels, raw nucleotide barcode sequences, assigned barcode index numbers, geographical, and size information. We propose three benchmark experiments to demonstrate the impact of the multi-modal data types on the classification and clustering accuracy. First, we pretrain a masked language model on the DNA barcode sequences of the BIOSCAN-5M dataset, and demonstrate the impact of using this large reference library on species- and genus-level classification performance. Second, we propose a zero-shot transfer learning task applied to images and DNA barcodes to cluster feature embeddings obtained from self-supervised learning, to investigate whether meaningful clusters can be derived from these representation embeddings. Third, we benchmark multi-modality by performing contrastive learning on DNA barcodes, image data, and taxonomic information. This yields a general shared embedding space enabling taxonomic classification using multiple types of information and modalities. The code repository of the BIOSCAN-5M Insect dataset is available at this https URL. 

**Abstract (ZH)**: BIOSCAN-5M昆虫数据集及其在机器学习领域的基准任务 

---
