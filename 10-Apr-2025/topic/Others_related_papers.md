# Towards Efficient Roadside LiDAR Deployment: A Fast Surrogate Metric Based on Entropy-Guided Visibility 

**Title (ZH)**: 基于熵引导可见性的一种快速代理指标 toward 有效路边LiDAR部署 

**Authors**: Yuze Jiang, Ehsan Javanmardi, Manabu Tsukada, Hiroshi Esaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.06772)  

**Abstract**: The deployment of roadside LiDAR sensors plays a crucial role in the development of Cooperative Intelligent Transport Systems (C-ITS). However, the high cost of LiDAR sensors necessitates efficient placement strategies to maximize detection performance. Traditional roadside LiDAR deployment methods rely on expert insight, making them time-consuming. Automating this process, however, demands extensive computation, as it requires not only visibility evaluation but also assessing detection performance across different LiDAR placements. To address this challenge, we propose a fast surrogate metric, the Entropy-Guided Visibility Score (EGVS), based on information gain to evaluate object detection performance in roadside LiDAR configurations. EGVS leverages Traffic Probabilistic Occupancy Grids (TPOG) to prioritize critical areas and employs entropy-based calculations to quantify the information captured by LiDAR beams. This eliminates the need for direct detection performance evaluation, which typically requires extensive labeling and computational resources. By integrating EGVS into the optimization process, we significantly accelerate the search for optimal LiDAR configurations. Experimental results using the AWSIM simulator demonstrate that EGVS strongly correlates with Average Precision (AP) scores and effectively predicts object detection performance. This approach offers a computationally efficient solution for roadside LiDAR deployment, facilitating scalable smart infrastructure development. 

**Abstract (ZH)**: 路边LiDAR传感器的部署对于协同智能交通系统（C-ITS）的发展至关重要。然而，LiDAR传感器的高成本促使需要高效的布放策略以最大化检测性能。传统的路边LiDAR部署方法依赖于专家的经验，使得过程耗时。然而，自动化这一过程需要大量的计算，因为它不仅需要进行可见性评估，还需要评估不同LiDAR布放方案的检测性能。为应对这一挑战，我们提出了一种快速的代理指标——基于信息增益的熵引导可见度分数（EGVS），用于评估路边LiDAR配置中的目标检测性能。EGVS利用交通概率占据网格（TPOG）来优先考虑关键区域，并采用基于熵的计算来量化LiDAR光束捕获的信息。这可以消除对直接检测性能评估的需要，后者通常需要大量的标签和计算资源。通过将EGVS集成到优化过程中，显著加速了寻找最优LiDAR配置的过程。使用AWSIM仿真器进行的实验结果表明，EGVS与平均精度（AP）分数高度相关，并能有效预测目标检测性能。这种方法提供了一种计算高效的目标检测解决方案，支持可扩展的智能基础设施开发。 

---
# SDHN: Skewness-Driven Hypergraph Networks for Enhanced Localized Multi-Robot Coordination 

**Title (ZH)**: SDHN：基于偏度的超图网络以增强局部多机器人协调 

**Authors**: Delin Zhao, Yanbo Shan, Chang Liu, Shenghang Lin, Yingxin Shou, Bin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06684)  

**Abstract**: Multi-Agent Reinforcement Learning is widely used for multi-robot coordination, where simple graphs typically model pairwise interactions. However, such representations fail to capture higher-order collaborations, limiting effectiveness in complex tasks. While hypergraph-based approaches enhance cooperation, existing methods often generate arbitrary hypergraph structures and lack adaptability to environmental uncertainties. To address these challenges, we propose the Skewness-Driven Hypergraph Network (SDHN), which employs stochastic Bernoulli hyperedges to explicitly model higher-order multi-robot interactions. By introducing a skewness loss, SDHN promotes an efficient structure with Small-Hyperedge Dominant Hypergraph, allowing robots to prioritize localized synchronization while still adhering to the overall information, similar to human coordination. Extensive experiments on Moving Agents in Formation and Robotic Warehouse tasks validate SDHN's effectiveness, demonstrating superior performance over state-of-the-art baselines. 

**Abstract (ZH)**: 基于偏度驱动的超图网络：促进多机器人高效协调 

---
# Dynamic Residual Safe Reinforcement Learning for Multi-Agent Safety-Critical Scenarios Decision-Making 

**Title (ZH)**: 多Agent安全关键场景决策中的动态残差安全强化学习 

**Authors**: Kaifeng Wang, Yinsong Chen, Qi Liu, Xueyuan Li, Xin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06670)  

**Abstract**: In multi-agent safety-critical scenarios, traditional autonomous driving frameworks face significant challenges in balancing safety constraints and task performance. These frameworks struggle to quantify dynamic interaction risks in real-time and depend heavily on manual rules, resulting in low computational efficiency and conservative strategies. To address these limitations, we propose a Dynamic Residual Safe Reinforcement Learning (DRS-RL) framework grounded in a safety-enhanced networked Markov decision process. It's the first time that the weak-to-strong theory is introduced into multi-agent decision-making, enabling lightweight dynamic calibration of safety boundaries via a weak-to-strong safety correction paradigm. Based on the multi-agent dynamic conflict zone model, our framework accurately captures spatiotemporal coupling risks among heterogeneous traffic participants and surpasses the static constraints of conventional geometric rules. Moreover, a risk-aware prioritized experience replay mechanism mitigates data distribution bias by mapping risk to sampling probability. Experimental results reveal that the proposed method significantly outperforms traditional RL algorithms in safety, efficiency, and comfort. Specifically, it reduces the collision rate by up to 92.17%, while the safety model accounts for merely 27% of the main model's parameters. 

**Abstract (ZH)**: 在多Agent关键安全场景中，传统自主驾驶框架在平衡安全约束与任务性能方面面临重大挑战。这些框架难以实时量化动态交互风险，并且高度依赖手动规则，导致计算效率低和保守策略。为解决这些问题，我们提出了一种基于增强安全网络马尔可夫决策过程的动态剩余安全强化学习（DRS-RL）框架。这是首次将“弱到强”理论引入多Agent决策，通过“弱到强”安全修正范式实现轻量级动态安全边界校准。基于多Agent动态冲突区域模型，我们的框架准确捕捉异质交通参与者之间的空间时间耦合风险，超越了传统几何规则的静态约束。此外，一种风险意识优先经验重播机制通过将风险映射到采样概率来减轻数据分布偏差。实验结果表明，所提出的方法在安全、效率和舒适性方面显著优于传统RL算法。具体而言，碰撞率最多可降低92.17%，而安全模型仅占主要模型参数的27%。 

---
# CAFE-AD: Cross-Scenario Adaptive Feature Enhancement for Trajectory Planning in Autonomous Driving 

**Title (ZH)**: CAFE-AD：跨场景自适应特征增强在自动驾驶路径规划中的应用 

**Authors**: Junrui Zhang, Chenjie Wang, Jie Peng, Haoyu Li, Jianmin Ji, Yu Zhang, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06584)  

**Abstract**: Imitation learning based planning tasks on the nuPlan dataset have gained great interest due to their potential to generate human-like driving behaviors. However, open-loop training on the nuPlan dataset tends to cause causal confusion during closed-loop testing, and the dataset also presents a long-tail distribution of scenarios. These issues introduce challenges for imitation learning. To tackle these problems, we introduce CAFE-AD, a Cross-Scenario Adaptive Feature Enhancement for Trajectory Planning in Autonomous Driving method, designed to enhance feature representation across various scenario types. We develop an adaptive feature pruning module that ranks feature importance to capture the most relevant information while reducing the interference of noisy information during training. Moreover, we propose a cross-scenario feature interpolation module that enhances scenario information to introduce diversity, enabling the network to alleviate over-fitting in dominant scenarios. We evaluate our method CAFE-AD on the challenging public nuPlan Test14-Hard closed-loop simulation benchmark. The results demonstrate that CAFE-AD outperforms state-of-the-art methods including rule-based and hybrid planners, and exhibits the potential in mitigating the impact of long-tail distribution within the dataset. Additionally, we further validate its effectiveness in real-world environments. The code and models will be made available at this https URL. 

**Abstract (ZH)**: 基于模仿学习的nuPlan数据集上的规划任务引起了广泛关注，因其潜在的人类驾驶行为生成能力。然而，对nuPlan数据集进行开环训练在闭环测试中往往会引发因果混淆问题，且数据集也呈现出场景的长尾分布。这些问题给模仿学习带来了挑战。为解决这些问题，我们提出了一种名为CAFE-AD的方法，即跨场景自适应特征增强用于自主驾驶的轨迹规划，旨在跨不同类型的场景增强特征表示。我们开发了一种自适应特征剪枝模块，用于评估特征的重要性和排名，从而在训练过程中捕获最相关的信息并减少噪声信息的干扰。此外，我们提出了一种跨场景特征插值模块，用于增强场景信息，引入多样性，使网络能够在主流场景中减轻过拟合。我们在具有挑战性的公共nuPlan Test14-Hard闭环仿真基准上评估了我们的方法CAFE-AD。结果表明，CAFE-AD在抵消数据集内长尾分布的影响方面优于现有的基于规则和混合规划的方法，并且在真实环境中的有效性得到了进一步验证。代码和模型将在以下链接处公开。 

---
# Extended Version: Multi-Robot Motion Planning with Cooperative Localization 

**Title (ZH)**: 扩展版本：具有合作定位的多机器人运动规划 

**Authors**: Anne Theurkauf, Nisar Ahmed, Morteza Lahijanian  

**Link**: [PDF](https://arxiv.org/pdf/2504.06429)  

**Abstract**: We consider the uncertain multi-robot motion planning (MRMP) problem with cooperative localization (CL-MRMP), under both motion and measurement noise, where each robot can act as a sensor for its nearby teammates. We formalize CL-MRMP as a chance-constrained motion planning problem, and propose a safety-guaranteed algorithm that explicitly accounts for robot-robot correlations. Our approach extends a sampling-based planner to solve CL-MRMP while preserving probabilistic completeness. To improve efficiency, we introduce novel biasing techniques. We evaluate our method across diverse benchmarks, demonstrating its effectiveness in generating motion plans, with significant performance gains from biasing strategies. 

**Abstract (ZH)**: 具有协同定位的不确定多机器人运动规划问题（CL-MRMP）：考虑运动和测量噪声，其中每个机器人可以作为其附近队友的传感器，并将其形式化为机会约束运动规划问题，提出一种保证安全性的算法，明确考虑机器人之间的关联性。同时引入新的偏差技术以提高效率，并通过多样化的基准测试验证了其有效性，偏差策略带来了显著的性能提升。 

---
# AssistanceZero: Scalably Solving Assistance Games 

**Title (ZH)**: AssistanceZero：规模化解决协助博弈 

**Authors**: Cassidy Laidlaw, Eli Bronstein, Timothy Guo, Dylan Feng, Lukas Berglund, Justin Svegliato, Stuart Russell, Anca Dragan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07091)  

**Abstract**: Assistance games are a promising alternative to reinforcement learning from human feedback (RLHF) for training AI assistants. Assistance games resolve key drawbacks of RLHF, such as incentives for deceptive behavior, by explicitly modeling the interaction between assistant and user as a two-player game where the assistant cannot observe their shared goal. Despite their potential, assistance games have only been explored in simple settings. Scaling them to more complex environments is difficult because it requires both solving intractable decision-making problems under uncertainty and accurately modeling human users' behavior. We present the first scalable approach to solving assistance games and apply it to a new, challenging Minecraft-based assistance game with over $10^{400}$ possible goals. Our approach, AssistanceZero, extends AlphaZero with a neural network that predicts human actions and rewards, enabling it to plan under uncertainty. We show that AssistanceZero outperforms model-free RL algorithms and imitation learning in the Minecraft-based assistance game. In a human study, our AssistanceZero-trained assistant significantly reduces the number of actions participants take to complete building tasks in Minecraft. Our results suggest that assistance games are a tractable framework for training effective AI assistants in complex environments. Our code and models are available at this https URL. 

**Abstract (ZH)**: 辅助游戏是强化学习从人类反馈（RLHF）之外的一种有前途的替代方案，用于训练AI助手。辅助游戏通过明确将助手与用户之间的交互建模为一个两位玩家的游戏来解决RLHF的关键劣势，其中助手无法观察到他们共享的目标。这解决了欺骗行为的动机问题。尽管它们具有潜力，但辅助游戏仅在简单设置中被探索过。将它们扩展到更复杂的环境中是困难的，因为这需要解决不确定条件下的难解决策问题，并准确建模人类使用者的行为。我们提出了第一个可扩展的辅助游戏解决方案，并将其应用于一个新挑战的基于Minecraft的辅助游戏，该游戏具有超过$10^{400}$个可能目标。我们的方法，AssistanceZero，扩展了AlphaZero，加入了能够预测人类行动和奖励的神经网络，使其能够在不确定条件下进行规划。我们证明了AssistanceZero在基于Minecraft的辅助游戏中优于无模型的方法和模仿学习。在一项人类研究中，我们的AssistanceZero训练的助手显着减少了参与者在Minecraft中完成建筑任务所需要的操作次数。我们的结果表明，辅助游戏是一个在复杂环境中训练有效AI助手的可解框架。我们的代码和模型可通过以下链接获得：this https URL。 

---
# SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills 

**Title (ZH)**: SkillWeaver: 网站智能体可通过发现和提升技能实现自我改进 

**Authors**: Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2504.07079)  

**Abstract**: To survive and thrive in complex environments, humans have evolved sophisticated self-improvement mechanisms through environment exploration, hierarchical abstraction of experiences into reuseable skills, and collaborative construction of an ever-growing skill repertoire. Despite recent advancements, autonomous web agents still lack crucial self-improvement capabilities, struggling with procedural knowledge abstraction, refining skills, and skill composition. In this work, we introduce SkillWeaver, a skill-centric framework enabling agents to self-improve by autonomously synthesizing reusable skills as APIs. Given a new website, the agent autonomously discovers skills, executes them for practice, and distills practice experiences into robust APIs. Iterative exploration continually expands a library of lightweight, plug-and-play APIs, significantly enhancing the agent's capabilities. Experiments on WebArena and real-world websites demonstrate the efficacy of SkillWeaver, achieving relative success rate improvements of 31.8% and 39.8%, respectively. Additionally, APIs synthesized by strong agents substantially enhance weaker agents through transferable skills, yielding improvements of up to 54.3% on WebArena. These results demonstrate the effectiveness of honing diverse website interactions into APIs, which can be seamlessly shared among various web agents. 

**Abstract (ZH)**: 为了在复杂环境中生存和发展，人类通过环境探索、经验的层次化抽象以及技能的合作构建进化出了复杂的自我改进机制。尽管最近取得了进展，自主网络代理仍然缺乏关键的自我改进能力，难以进行程序知识抽象、技能优化和技能组合。在本文中，我们介绍了一种以技能为中心的框架SkillWeaver，该框架使代理能够通过自主合成可重用的技能作为API来实现自我改进。面对一个新的网站，代理能够自主发现技能、执行技能进行练习，并将实践经验提炼成稳健的API。迭代探索不断扩展一个包含轻量级、即插即用API的库，显著增强了代理的能力。在WebArena和真实网站上的实验结果表明，SkillWeaver的有效性，在WebArena上分别实现了31.8%和39.8%的成功率改进。此外，由强代理合成的API通过可转移的技能显著增强了弱代理，WebArena上实现了高达54.3%的成功率改进。这些结果证明了将多样的网站交互打磨成API的有效性，这些API可以在各种网络代理之间无缝共享。 

---
# $Π$-NeSy: A Possibilistic Neuro-Symbolic Approach 

**Title (ZH)**: Π-NeSy: 一种可能性神经符号方法 

**Authors**: Ismaïl Baaj, Pierre Marquis  

**Link**: [PDF](https://arxiv.org/pdf/2504.07055)  

**Abstract**: In this article, we introduce a neuro-symbolic approach that combines a low-level perception task performed by a neural network with a high-level reasoning task performed by a possibilistic rule-based system. The goal is to be able to derive for each input instance the degree of possibility that it belongs to a target (meta-)concept. This (meta-)concept is connected to intermediate concepts by a possibilistic rule-based system. The probability of each intermediate concept for the input instance is inferred using a neural network. The connection between the low-level perception task and the high-level reasoning task lies in the transformation of neural network outputs modeled by probability distributions (through softmax activation) into possibility distributions. The use of intermediate concepts is valuable for the explanation purpose: using the rule-based system, the classification of an input instance as an element of the (meta-)concept can be justified by the fact that intermediate concepts have been recognized.
From the technical side, our contribution consists of the design of efficient methods for defining the matrix relation and the equation system associated with a possibilistic rule-based system. The corresponding matrix and equation are key data structures used to perform inferences from a possibilistic rule-based system and to learn the values of the rule parameters in such a system according to a training data sample. Furthermore, leveraging recent results on the handling of inconsistent systems of fuzzy relational equations, an approach for learning rule parameters according to multiple training data samples is presented. Experiments carried out on the MNIST addition problems and the MNIST Sudoku puzzles problems highlight the effectiveness of our approach compared with state-of-the-art neuro-symbolic ones. 

**Abstract (ZH)**: 一种结合神经网络和Possibilistic规则系统的神经符号方法：为每个输入实例推导其属于目标（元）概念的程度可能性 

---
# Are We Done with Object-Centric Learning? 

**Title (ZH)**: 我们已完成对象中心学习了吗？ 

**Authors**: Alexander Rubinstein, Ameya Prabhu, Matthias Bethge, Seong Joon Oh  

**Link**: [PDF](https://arxiv.org/pdf/2504.07092)  

**Abstract**: Object-centric learning (OCL) seeks to learn representations that only encode an object, isolated from other objects or background cues in a scene. This approach underpins various aims, including out-of-distribution (OOD) generalization, sample-efficient composition, and modeling of structured environments. Most research has focused on developing unsupervised mechanisms that separate objects into discrete slots in the representation space, evaluated using unsupervised object discovery. However, with recent sample-efficient segmentation models, we can separate objects in the pixel space and encode them independently. This achieves remarkable zero-shot performance on OOD object discovery benchmarks, is scalable to foundation models, and can handle a variable number of slots out-of-the-box. Hence, the goal of OCL methods to obtain object-centric representations has been largely achieved. Despite this progress, a key question remains: How does the ability to separate objects within a scene contribute to broader OCL objectives, such as OOD generalization? We address this by investigating the OOD generalization challenge caused by spurious background cues through the lens of OCL. We propose a novel, training-free probe called $\textbf{Object-Centric Classification with Applied Masks (OCCAM)}$, demonstrating that segmentation-based encoding of individual objects significantly outperforms slot-based OCL methods. However, challenges in real-world applications remain. We provide the toolbox for the OCL community to use scalable object-centric representations, and focus on practical applications and fundamental questions, such as understanding object perception in human cognition. Our code is available $\href{this https URL}{here}$. 

**Abstract (ZH)**: 面向对象的中心学习：基于分割的编码如何推动更广泛的OOD泛化能力 

---
# Enhancing Metabolic Syndrome Prediction with Hybrid Data Balancing and Counterfactuals 

**Title (ZH)**: 基于混合数据平衡和反事实方法的代谢综合征预测增强 

**Authors**: Sanyam Paresh Shah, Abdullah Mamun, Shovito Barua Soumma, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2504.06987)  

**Abstract**: Metabolic Syndrome (MetS) is a cluster of interrelated risk factors that significantly increases the risk of cardiovascular diseases and type 2 diabetes. Despite its global prevalence, accurate prediction of MetS remains challenging due to issues such as class imbalance, data scarcity, and methodological inconsistencies in existing studies. In this paper, we address these challenges by systematically evaluating and optimizing machine learning (ML) models for MetS prediction, leveraging advanced data balancing techniques and counterfactual analysis. Multiple ML models, including XGBoost, Random Forest, TabNet, etc., were trained and compared under various data balancing techniques such as random oversampling (ROS), SMOTE, ADASYN, and CTGAN. Additionally, we introduce MetaBoost, a novel hybrid framework that integrates SMOTE, ADASYN, and CTGAN, optimizing synthetic data generation through weighted averaging and iterative weight tuning to enhance the model's performance (achieving a 1.14% accuracy improvement over individual balancing techniques). A comprehensive counterfactual analysis is conducted to quantify feature-level changes required to shift individuals from high-risk to low-risk categories. The results indicate that blood glucose (50.3%) and triglycerides (46.7%) were the most frequently modified features, highlighting their clinical significance in MetS risk reduction. Additionally, probabilistic analysis shows elevated blood glucose (85.5% likelihood) and triglycerides (74.9% posterior probability) as the strongest predictors. This study not only advances the methodological rigor of MetS prediction but also provides actionable insights for clinicians and researchers, highlighting the potential of ML in mitigating the public health burden of metabolic syndrome. 

**Abstract (ZH)**: 代谢综合征（MetS）是一组相互关联的危险因素，显著增加了心血管疾病和2型糖尿病的风险。尽管其具有全球流行性，但由于类不平衡、数据稀缺和现有研究中方法学不一致等问题，对MetS的准确预测仍然具有挑战性。在本文中，我们通过系统评估和优化机器学习（ML）模型来应对这些挑战，利用先进的数据平衡技术和反事实分析。多种ML模型，包括XGBoost、随机森林、TabNet等，均在随机过采样（ROS）、SMOTE、ADASYN和CTGAN等多种数据平衡技术下进行了训练和比较。此外，我们引入了MetaBoost，这是一种新颖的混合框架，将SMOTE、ADASYN和CTGAN结合在一起，通过加权平均和迭代权重调整来优化合成数据生成，从而提升模型性能（相对于个体平衡技术实现了1.14%的准确性提升）。我们进行了全面的反事实分析，以量化将个体从高风险类别转变为低风险类别的所需特征级变化。结果表明，血糖（50.3%）和甘油三酯（46.7%）是最常被修改的特征，突显了它们在降低代谢综合征风险中的临床意义。此外，概率分析显示血糖（85.5%可能性）和甘油三酯（74.9%后验概率）是最强的预测因素。本研究不仅提高了代谢综合征预测的方法学严谨性，还为临床医师和研究人员提供了可行的见解，突显了机器学习在减轻代谢综合征公共卫生负担方面的潜力。 

---
# RNN-Transducer-based Losses for Speech Recognition on Noisy Targets 

**Title (ZH)**: 基于RNN-Transducer的在噪声目标下语音识别的损失函数 

**Authors**: Vladimir Bataev  

**Link**: [PDF](https://arxiv.org/pdf/2504.06963)  

**Abstract**: Training speech recognition systems on noisy transcripts is a significant challenge in industrial pipelines, where datasets are enormous and ensuring accurate transcription for every instance is difficult. In this work, we introduce novel loss functions to mitigate the impact of transcription errors in RNN-Transducer models. Our Star-Transducer loss addresses deletion errors by incorporating "skip frame" transitions in the loss lattice, restoring over 90% of the system's performance compared to models trained with accurate transcripts. The Bypass-Transducer loss uses "skip token" transitions to tackle insertion errors, recovering more than 60% of the quality. Finally, the Target-Robust Transducer loss merges these approaches, offering robust performance against arbitrary errors. Experimental results demonstrate that the Target-Robust Transducer loss significantly improves RNN-T performance on noisy data by restoring over 70% of the quality compared to well-transcribed data. 

**Abstract (ZH)**: 在嘈杂转录数据上训练语音识别系统是工业管道中的一个重大挑战，其中数据集庞大，确保每个实例的准确转录具有困难。在此项工作中，我们引入了新的损失函数以缓解基于RNN-Transducer模型中的转录错误的影响。我们的Star-Transducer损失通过在损失网格中引入“跳帧”转换来解决删除错误，恢复了系统超过90%的性能。Bypass-Transducer损失使用“跳令牌”转换来应对插入错误，恢复了超过60%的质量。最后，Target-Robust Transducer损失结合了这些方法，提供了对任意错误的稳健性能。实验结果表明，Target-Robust Transducer损失显著改进了基于RNN-T模型在嘈杂数据上的性能，恢复了与准确转录数据相比超过70%的质量。 

---
# Efficient Self-Supervised Learning for Earth Observation via Dynamic Dataset Curation 

**Title (ZH)**: 基于动态数据集编排的高效自我监督学习在地球观测中的应用 

**Authors**: Thomas Kerdreux, Alexandre Tuel, Quentin Febvre, Alexis Mouche, Bertrand Chapron  

**Link**: [PDF](https://arxiv.org/pdf/2504.06962)  

**Abstract**: Self-supervised learning (SSL) has enabled the development of vision foundation models for Earth Observation (EO), demonstrating strong transferability across diverse remote sensing tasks. While prior work has focused on network architectures and training strategies, the role of dataset curation, especially in balancing and diversifying pre-training datasets, remains underexplored. In EO, this challenge is amplified by the redundancy and heavy-tailed distributions common in satellite imagery, which can lead to biased representations and inefficient training.
In this work, we propose a dynamic dataset pruning strategy designed to improve SSL pre-training by maximizing dataset diversity and balance. Our method iteratively refines the training set without requiring a pre-existing feature extractor, making it well-suited for domains where curated datasets are limited or unavailable. We demonstrate our approach on the Sentinel-1 Wave Mode (WV) Synthetic Aperture Radar (SAR) archive, a challenging dataset dominated by ocean observations. We train models from scratch on the entire Sentinel-1 WV archive spanning 10 years. Across three downstream tasks, our results show that dynamic pruning improves both computational efficiency and representation quality, leading to stronger transferability.
We also release the weights of Nereus-SAR-1, the first model in the Nereus family, a series of foundation models for ocean observation and analysis using SAR imagery, at this http URL. 

**Abstract (ZH)**: 自我监督学习（SSL）已促进了地球观测（EO）领域视觉基础模型的发展，展示了其在多种遥感任务中的强大迁移能力。虽然先前的工作主要集中在网络架构和训练策略上，但数据集编排的作用，尤其是在平衡和多样化预训练数据集方面的角色，仍然未得到充分探索。在EO领域，这一挑战因卫星图像中常见的冗余性和重尾分布而加剧，可能导致有偏的表示和低效的训练。

在本文中，我们提出了一种动态数据集剪枝策略，旨在通过最大化数据集的多样性和平衡来提高SSL预训练的效果。该方法可在无需先存特征提取器的情况下迭代优化训练集，使其适用于受限或不可用标注数据集的领域。我们在Sentinel-1波模式（WV）合成孔径雷达（SAR）存档上展示了我们的方法，这是一个以海洋观测为主导的具有挑战性的数据集。我们在整个涵盖10年的Sentinel-1 WV存档上从头训练模型。在三个下游任务中，我们的结果显示动态剪枝提高了计算效率和表示质量，从而增强了迁移能力。

我们也发布了Nereus-SAR-1的权重，这是Nereus家族中的第一个模型，是一系列用于海洋观测和分析的SAR图像基础模型。详情请参阅此链接。 

---
# Adaptive Computation Pruning for the Forgetting Transformer 

**Title (ZH)**: 自适应计算剪枝以减轻遗忘变换器遗忘现象 

**Authors**: Zhixuan Lin, Johan Obando-Ceron, Xu Owen He, Aaron Courville  

**Link**: [PDF](https://arxiv.org/pdf/2504.06949)  

**Abstract**: The recently proposed Forgetting Transformer (FoX) incorporates a forget gate into softmax attention and has shown consistently better or on-par performance compared to the standard RoPE-based Transformer. Notably, many attention heads in FoX tend to forget quickly, causing their output at each timestep to rely primarily on the local context. Based on this observation, we propose Adaptive Computation Pruning (ACP) for FoX, a method that dynamically prunes computations involving input-output dependencies that are strongly decayed by the forget gate. This is achieved using a dynamically set pruning threshold that ensures that the pruned attention weights remain negligible. We apply ACP to language model pretraining with FoX and show it consistently reduces the number of FLOPs in softmax attention by around 70% across different model sizes and context lengths, resulting in a roughly 10% to 35% improvement in training throughput. Furthermore, longer context lengths yield greater computational savings. All these speed improvements are achieved without any performance degradation. We also perform several analyses to provide deeper insights into our method, such as examining the pruning patterns and analyzing the distribution of FLOP savings across different attention heads. Our code is available at this https URL. 

**Abstract (ZH)**: Adaptive Computation Pruning for Forgetting Transformer 

---
# Beyond Tools: Generative AI as Epistemic Infrastructure in Education 

**Title (ZH)**: 超越工具：生成式AI作为教育的认知基础设施 

**Authors**: Bodong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06928)  

**Abstract**: As generative AI rapidly integrates into educational infrastructures worldwide, it transforms how knowledge gets created, validated, and shared, yet current discourse inadequately addresses its implications as epistemic infrastructure mediating teaching and learning. This paper investigates how AI systems function as epistemic infrastructures in education and their impact on human epistemic agency. Adopting a situated cognition perspective and following a value-sensitive design approach, the study conducts a technical investigation of two representative AI systems in educational settings, analyzing their impact on teacher practice across three dimensions: affordances for skilled epistemic actions, support for epistemic sensitivity, and implications for long-term habit formation. The analysis reveals that current AI systems inadequately support teachers' skilled epistemic actions, insufficiently foster epistemic sensitivity, and potentially cultivate problematic habits that prioritize efficiency over epistemic agency. To address these challenges, the paper recommends recognizing the infrastructural transformation occurring in education, developing AI environments that stimulate skilled actions while upholding epistemic norms, and involving educators in AI design processes -- recommendations aimed at fostering AI integration that aligns with core educational values and maintains human epistemic agency. 

**Abstract (ZH)**: 随着生成式人工智能迅速融入全球教育基础设施，它正在改变知识的创造、验证和分享方式，然而当前的讨论未能充分探讨其作为促进教学与学习的认知基础设施所带来的影响。本文研究AI系统在教育中的认知基础设施功能及其对人类认知自主性的影响。本文采用情境认知视角和价值观敏感设计方法，对两种代表性教育AI系统的技术特性进行了分析，从技能性认知行动的能力、认知敏感性的支持以及对长期习惯形成的影响三个方面分析了其对教师实践的影响。分析表明，当前的AI系统在支持教师的技能性认知行动、促进认知敏感性方面存在不足，可能培养出以效率优先而非认知自主性的问题习惯。为应对这些挑战，本文建议承认教育中的基础设施变革，开发能够促进技能性行动并维护认知规范的AI环境，并让教育者参与AI的设计过程——这些建议旨在促进与核心教育价值观相一致的AI整合，维护人类的认知自主性。 

---
# An Analysis of Temporal Dropout in Earth Observation Time Series for Regression Tasks 

**Title (ZH)**: 地球观测时间序列中回归任务中时间段下采样的分析 

**Authors**: Miro Miranda, Francisco Mena, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06915)  

**Abstract**: Missing instances in time series data impose a significant challenge to deep learning models, particularly in regression tasks. In the Earth Observation field, satellite failure or cloud occlusion frequently results in missing time-steps, introducing uncertainties in the predicted output and causing a decline in predictive performance. While many studies address missing time-steps through data augmentation to improve model robustness, the uncertainty arising at the input level is commonly overlooked. To address this gap, we introduce Monte Carlo Temporal Dropout (MC-TD), a method that explicitly accounts for input-level uncertainty by randomly dropping time-steps during inference using a predefined dropout ratio, thereby simulating the effect of missing data. To bypass the need for costly searches for the optimal dropout ratio, we extend this approach with Monte Carlo Concrete Temporal Dropout (MC-ConcTD), a method that learns the optimal dropout distribution directly. Both MC-TD and MC-ConcTD are applied during inference, leveraging Monte Carlo sampling for uncertainty quantification. Experiments on three EO time-series datasets demonstrate that MC-ConcTD improves predictive performance and uncertainty calibration compared to existing approaches. Additionally, we highlight the advantages of adaptive dropout tuning over manual selection, making uncertainty quantification more robust and accessible for EO applications. 

**Abstract (ZH)**: 时间序列数据中的缺失实例对深度学习模型构成了显著挑战，特别是在回归任务中。在地球观测领域，卫星故障或云遮挡频繁导致时间步骤缺失，引入预测输出的不确定性并导致预测性能下降。尽管许多研究通过数据增强来解决时间步骤缺失问题以提高模型的鲁棒性，但输入级的不确定性往往被忽视。为弥补这一缺口，我们引入了蒙特卡洛时间 dropout（MC-TD）方法，该方法在推理过程中通过预定义的 dropout 比例随机丢弃时间步骤，从而模拟缺失数据的效果。为了避免搜索最优 dropout 比例的高成本，我们通过蒙特卡洛混凝土时间 dropout（MC-ConcTD）方法进一步扩展了这一思路，该方法直接学习最优 dropout 分布。MC-TD 和 MC-ConcTD 在推理过程中应用，利用蒙特卡洛采样进行不确定性量化。在三个地球观测时间序列数据集上的实验表明，MC-ConcTD 相比现有方法能够提高预测性能和不确定性校准。此外，我们还强调了自适应 dropout 调整相对于手动选择的优势，使得不确定性量化在地球观测应用中更加 robust 和易用。 

---
# Persona Dynamics: Unveiling the Impact of Personality Traits on Agents in Text-Based Games 

**Title (ZH)**: 人格动态：揭示人格特质对文本基础上游戏代理人物的影响 

**Authors**: Seungwon Lim, Seungbeen Lee, Dongjun Min, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06868)  

**Abstract**: Artificial agents are increasingly central to complex interactions and decision-making tasks, yet aligning their behaviors with desired human values remains an open challenge. In this work, we investigate how human-like personality traits influence agent behavior and performance within text-based interactive environments. We introduce PANDA: PersonalityAdapted Neural Decision Agents, a novel method for projecting human personality traits onto agents to guide their behavior. To induce personality in a text-based game agent, (i) we train a personality classifier to identify what personality type the agent's actions exhibit, and (ii) we integrate the personality profiles directly into the agent's policy-learning pipeline. By deploying agents embodying 16 distinct personality types across 25 text-based games and analyzing their trajectories, we demonstrate that an agent's action decisions can be guided toward specific personality profiles. Moreover, certain personality types, such as those characterized by higher levels of Openness, display marked advantages in performance. These findings underscore the promise of personality-adapted agents for fostering more aligned, effective, and human-centric decision-making in interactive environments. 

**Abstract (ZH)**: 人工代理在复杂交互和决策任务中越来越占据中心地位，然而使其行为与期望的人类价值相一致仍然是一个开放性的挑战。在此工作中，我们探讨了人性特征如何影响文本交互环境中代理的行为和性能。我们提出了PANDA：个性适应神经决策代理，这是一种将人类个性特征投影到代理上以引导其行为的新方法。为了在文本交互代理中诱导个性特征，（i）我们训练了一个个性分类器来识别代理行为展现出的个性类型；（ii）我们将个性档案直接整合到代理的策略学习管道中。通过在25个文本交互游戏中部署16种不同的个性类型的代理，并分析它们的行为轨迹，我们展示了代理的行为决策可以导向特定的个性特征。此外，某些个性类型，如开放性水平较高的类型，在性能方面表现出明显的优势。这些发现表明，个性适应代理在促进更一致、更有效和更以人为中心的交互环境中决策方面具有潜力。 

---
# Adaptive Locally Linear Embedding 

**Title (ZH)**: 自适应局部线性嵌入 

**Authors**: Ali Goli, Mahdieh Alizadeh, Hadi Sadoghi Yazdi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06829)  

**Abstract**: Manifold learning techniques, such as Locally linear embedding (LLE), are designed to preserve the local neighborhood structures of high-dimensional data during dimensionality reduction. Traditional LLE employs Euclidean distance to define neighborhoods, which can struggle to capture the intrinsic geometric relationships within complex data. A novel approach, Adaptive locally linear embedding(ALLE), is introduced to address this limitation by incorporating a dynamic, data-driven metric that enhances topological preservation. This method redefines the concept of proximity by focusing on topological neighborhood inclusion rather than fixed distances. By adapting the metric based on the local structure of the data, it achieves superior neighborhood preservation, particularly for datasets with complex geometries and high-dimensional structures. Experimental results demonstrate that ALLE significantly improves the alignment between neighborhoods in the input and feature spaces, resulting in more accurate and topologically faithful embeddings. This approach advances manifold learning by tailoring distance metrics to the underlying data, providing a robust solution for capturing intricate relationships in high-dimensional datasets. 

**Abstract (ZH)**: 适配局部线性嵌入(ALLE):基于动态数据驱动度量的流形学习技术 

---
# Learning in Spiking Neural Networks with a Calcium-based Hebbian Rule for Spike-timing-dependent Plasticity 

**Title (ZH)**: 基于钙离子的时限依赖可塑性希布定律的脉冲神经网络学习 

**Authors**: Willian Soares Girão, Nicoletta Risi, Elisabetta Chicca  

**Link**: [PDF](https://arxiv.org/pdf/2504.06796)  

**Abstract**: Understanding how biological neural networks are shaped via local plasticity mechanisms can lead to energy-efficient and self-adaptive information processing systems, which promises to mitigate some of the current roadblocks in edge computing systems. While biology makes use of spikes to seamless use both spike timing and mean firing rate to modulate synaptic strength, most models focus on one of the two. In this work, we present a Hebbian local learning rule that models synaptic modification as a function of calcium traces tracking neuronal activity. We show how the rule reproduces results from spike time and spike rate protocols from neuroscientific studies. Moreover, we use the model to train spiking neural networks on MNIST digit recognition to show and explain what sort of mechanisms are needed to learn real-world patterns. We show how our model is sensitive to correlated spiking activity and how this enables it to modulate the learning rate of the network without altering the mean firing rate of the neurons nor the hyparameters of the learning rule. To the best of our knowledge, this is the first work that showcases how spike timing and rate can be complementary in their role of shaping the connectivity of spiking neural networks. 

**Abstract (ZH)**: 通过局部可塑性机制理解生物神经网络的形成可以导致高效的自适应信息处理系统，这有望缓解边缘计算系统中的部分瓶颈。尽管生物学利用尖峰来无缝结合尖峰时序和平均放电率来调节突触强度，大多数模型仅聚焦于其中之一。在本工作中，我们提出了一种Hebbian局部学习规则，将突触修改建模为神经活动钙踪迹的函数。我们展示该规则如何再现神经科学实验中尖峰时序和尖峰速率协议的结果。此外，我们使用该模型在MNIST数字识别任务上训练尖峰神经网络，以展示和解释需要哪些机制来学习真实世界的模式。我们展示了我们的模型对尖峰活动的相关性敏感性，并说明这种敏感性如何使网络能够在不改变神经元的平均放电率或学习规则的超参数的情况下调节学习率。据我们所知，这是首例展示尖峰时序和速率在塑造尖峰神经网络连接性方面互补作用的工作。 

---
# AI, Help Me Think$\unicode{x2014}$but for Myself: Assisting People in Complex Decision-Making by Providing Different Kinds of Cognitive Support 

**Title (ZH)**: AI，帮助我思考——但仅限于我自己：通过提供不同类型的认知支持来协助人们进行复杂决策 

**Authors**: Leon Reicherts, Zelun Tony Zhang, Elisabeth von Oswald, Yuanting Liu, Yvonne Rogers, Mariam Hassib  

**Link**: [PDF](https://arxiv.org/pdf/2504.06771)  

**Abstract**: How can we design AI tools that effectively support human decision-making by complementing and enhancing users' reasoning processes? Common recommendation-centric approaches face challenges such as inappropriate reliance or a lack of integration with users' decision-making processes. Here, we explore an alternative interaction model in which the AI outputs build upon users' own decision-making rationales. We compare this approach, which we call ExtendAI, with a recommendation-based AI. Participants in our mixed-methods user study interacted with both AIs as part of an investment decision-making task. We found that the AIs had different impacts, with ExtendAI integrating better into the decision-making process and people's own thinking and leading to slightly better outcomes. RecommendAI was able to provide more novel insights while requiring less cognitive effort. We discuss the implications of these and other findings along with three tensions of AI-assisted decision-making which our study revealed. 

**Abstract (ZH)**: 如何设计能够通过补充和增强用户推理过程来有效支持人类决策的AI工具？常见的基于推荐的方法面临与用户决策过程不适当的依赖或缺乏整合等问题。在这里，我们探索了一种替代的交互模型，在这种模型中，AI的输出建立在用户自身决策推理的基础上。我们将这种方法称为ExtendAI，并将其与基于推荐的AI进行了比较。在包含投资决策任务的混合方法用户研究中，参与者与两种AI进行了交互。我们发现，这两种AI产生了不同的影响，ExtendAI更好地融入了决策过程和人们的思考，并导致了稍好的结果。RecommendAI能够提供更具新颖性的见解，同时需要较少的认知努力。我们讨论了这些和其他发现的含义，以及我们在研究中揭示的AI辅助决策的三种张力。 

---
# Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception 

**Title (ZH)**: 检测所有类型深度伪造音频：小波提示调整以增强听觉感知 

**Authors**: Yuankun Xie, Ruibo Fu, Zhiyong Wang, Xiaopeng Wang, Songjun Cao, Long Ma, Haonan Cheng, Long Ye  

**Link**: [PDF](https://arxiv.org/pdf/2504.06753)  

**Abstract**: The rapid advancement of audio generation technologies has escalated the risks of malicious deepfake audio across speech, sound, singing voice, and music, threatening multimedia security and trust. While existing countermeasures (CMs) perform well in single-type audio deepfake detection (ADD), their performance declines in cross-type scenarios. This paper is dedicated to studying the alltype ADD task. We are the first to comprehensively establish an all-type ADD benchmark to evaluate current CMs, incorporating cross-type deepfake detection across speech, sound, singing voice, and music. Then, we introduce the prompt tuning self-supervised learning (PT-SSL) training paradigm, which optimizes SSL frontend by learning specialized prompt tokens for ADD, requiring 458x fewer trainable parameters than fine-tuning (FT). Considering the auditory perception of different audio types,we propose the wavelet prompt tuning (WPT)-SSL method to capture type-invariant auditory deepfake information from the frequency domain without requiring additional training parameters, thereby enhancing performance over FT in the all-type ADD task. To achieve an universally CM, we utilize all types of deepfake audio for co-training. Experimental results demonstrate that WPT-XLSR-AASIST achieved the best performance, with an average EER of 3.58% across all evaluation sets. The code is available online. 

**Abstract (ZH)**: 快速发展的音频生成技术加剧了语音、声音、歌声和音乐中恶意深度假音的风险，威胁多媒体安全与信任。现有对抗措施在单一类型音频深度假音检测方面表现良好，但在跨类型场景中性能下降。本文致力于研究跨类型音频深度假音检测任务。我们首次全面建立了跨类型音频深度假音检测基准，用于评估当前的对抗措施，涵盖了语音、声音、歌声和音乐跨类型的深度假音检测。然后，我们引入了提示调谐半监督学习（PT-SSL）训练范式，通过学习专门的提示标记优化半监督学习前端，所需可训练参数仅为微调的1/458。考虑到不同音频类型的声音感知，我们提出了小波提示调谐（WPT）-半监督学习方法，能够在频率域中捕获类型不变的声音深度假音信息，而无需额外训练参数，从而在跨类型音频深度假音检测任务中优于微调。为了构建通用对抗措施，我们利用所有类型的真实深度假音音频进行协同训练。实验结果表明，WPT-XLSR-AASIST在所有评估集上的平均错误检测率EER达到了3.58%，代码已在线提供。 

---
# Hyperparameter Optimisation with Practical Interpretability and Explanation Methods in Probabilistic Curriculum Learning 

**Title (ZH)**: 基于概率性课程学习的超参数优化及其实用可解释性方法的研究 

**Authors**: Llewyn Salt, Marcus Gallagher  

**Link**: [PDF](https://arxiv.org/pdf/2504.06683)  

**Abstract**: Hyperparameter optimisation (HPO) is crucial for achieving strong performance in reinforcement learning (RL), as RL algorithms are inherently sensitive to hyperparameter settings. Probabilistic Curriculum Learning (PCL) is a curriculum learning strategy designed to improve RL performance by structuring the agent's learning process, yet effective hyperparameter tuning remains challenging and computationally demanding. In this paper, we provide an empirical analysis of hyperparameter interactions and their effects on the performance of a PCL algorithm within standard RL tasks, including point-maze navigation and DC motor control. Using the AlgOS framework integrated with Optuna's Tree-Structured Parzen Estimator (TPE), we present strategies to refine hyperparameter search spaces, enhancing optimisation efficiency. Additionally, we introduce a novel SHAP-based interpretability approach tailored specifically for analysing hyperparameter impacts, offering clear insights into how individual hyperparameters and their interactions influence RL performance. Our work contributes practical guidelines and interpretability tools that significantly improve the effectiveness and computational feasibility of hyperparameter optimisation in reinforcement learning. 

**Abstract (ZH)**: 超参数优化（HPO）在强化学习（RL）中对于实现优异性能至关重要，因为RL算法对超参数设置具有固有的敏感性。概率性 curriculum 学习（PCL）是一种旨在通过结构化智能体的学习过程来提高RL性能的curriculum学习策略，但有效的超参数调整仍然是一个具有挑战性且计算成本高昂的问题。在本文中，我们对PCL算法在标准RL任务（如点迷宫导航和DC电机控制）中超参数交互作用及其对性能的影响进行了 empirical 分析。通过将AlgOS框架与Optuna的树结构帕兹内斯特imator（TPE）结合，我们提出了一种策略来细化超参数搜索空间，从而提高优化效率。此外，我们引入了一种基于SHAP的新型可解释性方法，专门用于分析超参数影响，提供关于如何通过单独的超参数及其交互作用影响RL性能的清晰见解。我们的工作为强化学习中的超参数优化提供了实用指南和可解释性工具，显著提高了超参数优化的有效性和计算可行性。 

---
# NLP Security and Ethics, in the Wild 

**Title (ZH)**: NLP安全与伦理：在现实世界中的应用 

**Authors**: Heather Lent, Erick Galinkin, Yiyi Chen, Jens Myrup Pedersen, Leon Derczynski, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2504.06669)  

**Abstract**: As NLP models are used by a growing number of end-users, an area of increasing importance is NLP Security (NLPSec): assessing the vulnerability of models to malicious attacks and developing comprehensive countermeasures against them. While work at the intersection of NLP and cybersecurity has the potential to create safer NLP for all, accidental oversights can result in tangible harm (e.g., breaches of privacy or proliferation of malicious models). In this emerging field, however, the research ethics of NLP have not yet faced many of the long-standing conundrums pertinent to cybersecurity, until now. We thus examine contemporary works across NLPSec, and explore their engagement with cybersecurity's ethical norms. We identify trends across the literature, ultimately finding alarming gaps on topics like harm minimization and responsible disclosure. To alleviate these concerns, we provide concrete recommendations to help NLP researchers navigate this space more ethically, bridging the gap between traditional cybersecurity and NLP ethics, which we frame as ``white hat NLP''. The goal of this work is to help cultivate an intentional culture of ethical research for those working in NLP Security. 

**Abstract (ZH)**: 随着自然语言处理模型被越来越多的终端用户使用，自然语言处理安全（NLPSec）领域的重要性不断增加：评估模型对恶意攻击的脆弱性并开发全面的应对措施。虽然自然语言处理与网络安全的交叉研究有望为所有人创造更安全的自然语言处理，但偶然的疏忽可能会造成实际损害（如隐私泄露或恶意模型的传播）。然而，在这一新兴领域中，自然语言处理的科研伦理尚未面临与网络安全相关的许多长期难题，直到现在。因此，我们考察了NLPSec领域的当代研究成果，并探讨了它们在网络安全伦理规范方面的实践情况。我们识别出文献中的趋势，最终发现有关最小化损害和负责任披露等话题存在令人担忧的空白。为缓解这些担忧，我们提供了具体的建议，以帮助自然语言处理研究人员更伦理地导航这一领域，弥合传统网络安全与自然语言处理伦理之间的差距，我们将这一框架称为“白帽自然语言处理”。本工作的目标是帮助培养自然语言处理安全领域中一种有意识的伦理研究文化。 

---
# GRAIN: Multi-Granular and Implicit Information Aggregation Graph Neural Network for Heterophilous Graphs 

**Title (ZH)**: GRAIN：面向异构图的多粒度和隐式信息聚合图神经网络 

**Authors**: Songwei Zhao, Yuan Jiang, Zijing Zhang, Yang Yu, Hechang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06649)  

**Abstract**: Graph neural networks (GNNs) have shown significant success in learning graph representations. However, recent studies reveal that GNNs often fail to outperform simple MLPs on heterophilous graph tasks, where connected nodes may differ in features or labels, challenging the homophily assumption. Existing methods addressing this issue often overlook the importance of information granularity and rarely consider implicit relationships between distant nodes. To overcome these limitations, we propose the Granular and Implicit Graph Network (GRAIN), a novel GNN model specifically designed for heterophilous graphs. GRAIN enhances node embeddings by aggregating multi-view information at various granularity levels and incorporating implicit data from distant, non-neighboring nodes. This approach effectively integrates local and global information, resulting in smoother, more accurate node representations. We also introduce an adaptive graph information aggregator that efficiently combines multi-granularity and implicit data, significantly improving node representation quality, as shown by experiments on 13 datasets covering varying homophily and heterophily. GRAIN consistently outperforms 12 state-of-the-art models, excelling on both homophilous and heterophilous graphs. 

**Abstract (ZH)**: 粒度化和隐含图网络：一种面向异构图的新颖GNN模型 

---
# AMAD: AutoMasked Attention for Unsupervised Multivariate Time Series Anomaly Detection 

**Title (ZH)**: AMAD: AutoMasked注意力机制在无监督多变量时间序列异常检测中的应用 

**Authors**: Tiange Huang, Yongjun Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.06643)  

**Abstract**: Unsupervised multivariate time series anomaly detection (UMTSAD) plays a critical role in various domains, including finance, networks, and sensor systems. In recent years, due to the outstanding performance of deep learning in general sequential tasks, many models have been specialized for deep UMTSAD tasks and have achieved impressive results, particularly those based on the Transformer and self-attention mechanisms. However, the sequence anomaly association assumptions underlying these models are often limited to specific predefined patterns and scenarios, such as concentrated or peak anomaly patterns. These limitations hinder their ability to generalize to diverse anomaly situations, especially where the lack of labels poses significant challenges. To address these issues, we propose AMAD, which integrates \textbf{A}uto\textbf{M}asked Attention for UMTS\textbf{AD} scenarios. AMAD introduces a novel structure based on the AutoMask mechanism and an attention mixup module, forming a simple yet generalized anomaly association representation framework. This framework is further enhanced by a Max-Min training strategy and a Local-Global contrastive learning approach. By combining multi-scale feature extraction with automatic relative association modeling, AMAD provides a robust and adaptable solution to UMTSAD challenges. Extensive experimental results demonstrate that the proposed model achieving competitive performance results compared to SOTA benchmarks across a variety of datasets. 

**Abstract (ZH)**: 无监督多变量时间序列异常检测（UMTSAD）在金融、网络和传感器系统等多个领域发挥着关键作用。近年来，由于深度学习在通用序列任务中的出色性能，许多模型专门用于深度UMTSAD任务并取得了显著成果，特别是基于Transformer和自注意力机制的模型。然而，这些模型底层的序列异常关联假设往往局限于特定预定义的模式和场景，如集中的或峰值异常模式。这些限制阻碍了它们在多样异常情况下的泛化能力，尤其是在缺乏标签的情况下构成重大挑战。为解决这些问题，我们提出AMAD，即基于自动掩蔽注意机制的UMTSAD场景。AMAD引入了一种基于AutoMask机制和注意力mixup模块的新结构，形成了一个简单而通用的异常关联表示框架。该框架进一步通过最大值-最小值训练策略和局部-全局对比学习方法得到增强。通过结合多尺度特征提取和自动相对关联建模，AMAD提供了一种鲁棒且适应性强的UMTSAD解决方案。 extensive实验结果表明，所提出模型在各种数据集上的性能与现有最佳基准相当。 

---
# Wanting to be Understood 

**Title (ZH)**: 想要被理解 

**Authors**: Chrisantha Fernando, Dylan Banarse, Simon Osindero  

**Link**: [PDF](https://arxiv.org/pdf/2504.06611)  

**Abstract**: This paper explores an intrinsic motivation for mutual awareness, hypothesizing that humans possess a fundamental drive to understand \textit{and to be understood} even in the absence of extrinsic rewards. Through simulations of the perceptual crossing paradigm, we explore the effect of various internal reward functions in reinforcement learning agents. The drive to understand is implemented as an active inference type artificial curiosity reward, whereas the drive to be understood is implemented through intrinsic rewards for imitation, influence/impressionability, and sub-reaction time anticipation of the other. Results indicate that while artificial curiosity alone does not lead to a preference for social interaction, rewards emphasizing reciprocal understanding successfully drive agents to prioritize interaction. We demonstrate that this intrinsic motivation can facilitate cooperation in tasks where only one agent receives extrinsic reward for the behaviour of the other. 

**Abstract (ZH)**: 本文探索了内在动机的互惠意识，假设人类即使在缺乏外在奖励的情况下，也具有一种基本驱动力，即理解他人并渴望被他人理解。通过感知交叉范式的模拟，我们探索了强化学习代理内部奖励函数的影响。理解的驱动力通过一种主动推断类型的 artificial curiosity 奖励实现，而被理解的驱动力通过模仿、影响/可影响性和对另一方亚反应时间的预期实现。结果表明，虽然单独的人工curiosity未能导致对社会互动的偏好，但强调相互理解的奖励成功地驱动代理优先进行互动。我们证明了这种内在动机可以促进在只有代理之一因其行为而获得外在奖励的任务中实现合作。 

---
# InteractRank: Personalized Web-Scale Search Pre-Ranking with Cross Interaction Features 

**Title (ZH)**: InteractRank：基于交叉交互特征的个性化大规模网页预排序 

**Authors**: Sujay Khandagale, Bhawna Juneja, Prabhat Agarwal, Aditya Subramanian, Jaewon Yang, Yuting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06609)  

**Abstract**: Modern search systems use a multi-stage architecture to deliver personalized results efficiently. Key stages include retrieval, pre-ranking, full ranking, and blending, which refine billions of items to top selections. The pre-ranking stage, vital for scoring and filtering hundreds of thousands of items down to a few thousand, typically relies on two tower models due to their computational efficiency, despite often lacking in capturing complex interactions. While query-item cross interaction features are paramount for full ranking, integrating them into pre-ranking models presents efficiency-related challenges. In this paper, we introduce InteractRank, a novel two tower pre-ranking model with robust cross interaction features used at Pinterest. By incorporating historical user engagement-based query-item interactions in the scoring function along with the two tower dot product, InteractRank significantly boosts pre-ranking performance with minimal latency and computation costs. In real-world A/B experiments at Pinterest, InteractRank improves the online engagement metric by 6.5% over a BM25 baseline and by 3.7% over a vanilla two tower baseline. We also highlight other components of InteractRank, like real-time user-sequence modeling, and analyze their contributions through offline ablation studies. The code for InteractRank is available at this https URL. 

**Abstract (ZH)**: 现代搜索引擎使用多阶段架构高效交付个性化结果。关键阶段包括检索、预排名、全面排名和混合，这些阶段将数十亿项内容精炼为顶级选择。预排名阶段对于通过对数以万计项的评分和过滤至数千项至关重要，尽管计算效率高，但往往难以捕捉复杂的交互。虽然查询-项交叉交互特征对于全面排名至关重要，但将其集成到预排名模型中会带来效率方面的挑战。在本文中，我们提出了InteractRank，这是一种新颖的两塔预排名模型，广泛应用于Pinterest，其中包含强大的交叉交互特征。通过在评分函数中结合基于历史用户参与度的查询-项交互以及两塔点积，InteractRank在几乎没有延迟和计算成本的情况下显著提升了预排名性能。在Pinterest的实际在线A/B实验中，InteractRank相较于BM25基线提高了6.5%的在线参与度指标，相较于纯两塔基线提高了3.7%。我们还强调了InteractRank的其他组件，如实时用户序列建模，并通过离线消融研究分析了它们的贡献。InteractRank的代码可在以下链接获取：this https URL。 

---
# Societal Impacts Research Requires Benchmarks for Creative Composition Tasks 

**Title (ZH)**: 社会影响研究需要创造性 compositions 任务的基准标准 

**Authors**: Judy Hanwen Shen, Carlos Guestrin  

**Link**: [PDF](https://arxiv.org/pdf/2504.06549)  

**Abstract**: Foundation models that are capable of automating cognitive tasks represent a pivotal technological shift, yet their societal implications remain unclear. These systems promise exciting advances, yet they also risk flooding our information ecosystem with formulaic, homogeneous, and potentially misleading synthetic content. Developing benchmarks grounded in real use cases where these risks are most significant is therefore critical. Through a thematic analysis using 2 million language model user prompts, we identify creative composition tasks as a prevalent usage category where users seek help with personal tasks that require everyday creativity. Our fine-grained analysis identifies mismatches between current benchmarks and usage patterns among these tasks. Crucially, we argue that the same use cases that currently lack thorough evaluations can lead to negative downstream impacts. This position paper argues that benchmarks focused on creative composition tasks is a necessary step towards understanding the societal harms of AI-generated content. We call for greater transparency in usage patterns to inform the development of new benchmarks that can effectively measure both the progress and the impacts of models with creative capabilities. 

**Abstract (ZH)**: 能够自动化认知任务的基石模型代表了技术上的重要转折，然而它们的社会影响尚不明确。这些系统承诺带来激动人心的进步，但也有可能向我们的信息生态系统中注入公式化、同质化且可能误导性的合成内容。因此，在这些风险最突出的实际应用场景中开发基准至关重要。通过对200万自然语言模型用户提示进行主题分析，我们发现创意组合任务是用户寻求帮助的常见类别，这些任务需要日常的创造力。精细的分析发现当前基准与这些任务使用模式之间的不匹配。至关重要的是，我们认为当前缺乏充分评估的使用案例可能导致负面影响。本文认为，关注创意组合任务的基准是理解AI生成内容社会危害的必要步骤。我们呼吁提高使用模式的透明度，以指导开发新的基准，这些基准能够有效衡量具有创造力能力的模型的进展及其影响。 

---
# Polygon: Symbolic Reasoning for SQL using Conflict-Driven Under-Approximation Search 

**Title (ZH)**: Polygon: 基于冲突驱动的下近似搜索的符号推理用于SQL 

**Authors**: Pinhan Zhao, Yuepeng Wang, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06542)  

**Abstract**: We present a novel symbolic reasoning engine for SQL which can efficiently generate an input $I$ for $n$ queries $P_1, \cdots, P_n$, such that their outputs on $I$ satisfy a given property (expressed in SMT). This is useful in different contexts, such as disproving equivalence of two SQL queries and disambiguating a set of queries. Our first idea is to reason about an under-approximation of each $P_i$ -- that is, a subset of $P_i$'s input-output behaviors. While it makes our approach both semantics-aware and lightweight, this idea alone is incomplete (as a fixed under-approximation might miss some behaviors of interest). Therefore, our second idea is to perform search over an expressive family of under-approximations (which collectively cover all program behaviors of interest), thereby making our approach complete. We have implemented these ideas in a tool, Polygon, and evaluated it on over 30,000 benchmarks across two tasks (namely, SQL equivalence refutation and query disambiguation). Our evaluation results show that Polygon significantly outperforms all prior techniques. 

**Abstract (ZH)**: 一种新型SQL符号推理引擎及其在查询等价反驳和查询解析中的应用 

---
# Flexible Graph Similarity Computation With A Proactive Optimization Strategy 

**Title (ZH)**: 基于主动优化策略的灵活图相似性计算 

**Authors**: Zhouyang Liu, Ning Liu, Yixin Chen, Jiezhong He, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.06533)  

**Abstract**: Graph Edit Distance (GED) is an important similarity measure in graph retrieval, which quantifies the minimum cost of transforming one graph into another through edit operations, and offers flexibility by allowing customizable operation costs. Recent learning-based approaches approximate GEDs with the distances between representations in vector spaces. However, these methods often struggle with varying operation costs due to neglecting the impact of these costs on determining optimal graph mappings. Furthermore, they rely on isolated node distances as guidance, necessitating inefficient reactive refinements of mappings. To address these issues, we propose Graph Edit Network (GEN), a novel learning-based approach for flexible GED computation. By identifying the limitations of existing methods in capturing flexibility of GED, we introduce a principled yet simple solution that incorporates the operation costs before establishing mappings. To improve matching efficiency, we propose a strategy that proactively optimizes guidance from a graph perspective. This strategy initializes guidance as each node's alignment difficulty and captures the interdependencies between matches within and across graphs through a difficulty propagation mechanism, enabling more informed decisions. As a result, GEN selects optimal matches in a single step, minimizing the need for costly refinements. Results on real-world and synthetic datasets demonstrate the effectiveness, time efficiency, and adaptability of GEN, achieving up to 37.8\% error reduction and 72.7\% inference time reduction compared with state-of-the-art models, while performing robustly under varying cost settings and graph sizes. 

**Abstract (ZH)**: 基于图编辑网络的灵活图编辑距离计算 

---
# WaveHiTS: Wavelet-Enhanced Hierarchical Time Series Modeling for Wind Direction Nowcasting in Eastern Inner Mongolia 

**Title (ZH)**: WaveHiTS：增强小波变换的风向短时预报分层时间序列建模在内蒙古东部地区 

**Authors**: Hailong Shu, Weiwei Song, Yue Wang, Jiping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06532)  

**Abstract**: Wind direction forecasting plays a crucial role in optimizing wind energy production, but faces significant challenges due to the circular nature of directional data, error accumulation in multi-step forecasting, and complex meteorological interactions. This paper presents a novel model, WaveHiTS, which integrates wavelet transform with Neural Hierarchical Interpolation for Time Series to address these challenges. Our approach decomposes wind direction into U-V components, applies wavelet transform to capture multi-scale frequency patterns, and utilizes a hierarchical structure to model temporal dependencies at multiple scales, effectively mitigating error propagation. Experiments conducted on real-world meteorological data from Inner Mongolia, China demonstrate that WaveHiTS significantly outperforms deep learning models (RNN, LSTM, GRU), transformer-based approaches (TFT, Informer, iTransformer), and hybrid models (EMD-LSTM). The proposed model achieves RMSE values of approximately 19.2°-19.4° compared to 56°-64° for deep learning recurrent models, maintaining consistent accuracy across all forecasting steps up to 60 minutes ahead. Moreover, WaveHiTS demonstrates superior robustness with vector correlation coefficients (VCC) of 0.985-0.987 and hit rates of 88.5%-90.1%, substantially outperforming baseline models. Ablation studies confirm that each component-wavelet transform, hierarchical structure, and U-V decomposition-contributes meaningfully to overall performance. These improvements in wind direction nowcasting have significant implications for enhancing wind turbine yaw control efficiency and grid integration of wind energy. 

**Abstract (ZH)**: 基于小波变换的神经分层插值模型WaveHiTS在风向预测中的应用 

---
# Beyond Moore's Law: Harnessing the Redshift of Generative AI with Effective Hardware-Software Co-Design 

**Title (ZH)**: 超越摩尔定律：通过有效的硬件软件协同设计利用生成式AI的红移效应 

**Authors**: Amir Yazdanbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2504.06531)  

**Abstract**: For decades, Moore's Law has served as a steadfast pillar in computer architecture and system design, promoting a clear abstraction between hardware and software. This traditional Moore's computing paradigm has deepened the rift between the two, enabling software developers to achieve near-exponential performance gains often without needing to delve deeply into hardware-specific optimizations. Yet today, Moore's Law -- with its once relentless performance gains now diminished to incremental improvements -- faces inevitable physical barriers. This stagnation necessitates a reevaluation of the conventional system design philosophy. The traditional decoupled system design philosophy, which maintains strict abstractions between hardware and software, is increasingly obsolete. The once-clear boundary between software and hardware is rapidly dissolving, replaced by co-design. It is imperative for the computing community to intensify its commitment to hardware-software co-design, elevating system abstractions to first-class citizens and reimagining design principles to satisfy the insatiable appetite of modern computing. Hardware-software co-design is not a recent innovation. To illustrate its historical evolution, I classify its development into five relatively distinct ``epochs''. This post also highlights the growing influence of the architecture community in interdisciplinary teams -- particularly alongside ML researchers -- and explores why current co-design paradigms are struggling in today's computing landscape. Additionally, I will examine the concept of the ``hardware lottery'' and explore directions to mitigate its constraining influence on the next era of computing innovation. 

**Abstract (ZH)**: 多年来，摩尔定律一直是计算机体系结构和系统设计中的坚实支柱，促进了硬件和软件之间的清晰抽象。传统的摩尔计算范式加深了两者之间的鸿沟，使软件开发者能够在很大程度上无需深入了解硬件特定优化的情况下实现接近指数级别的性能提升。然而，随着摩尔定律从曾经不断的性能提升转变为微小改进，它不可避免地遇到了物理壁垒。这种停滞需要重新评估传统的系统设计哲学。传统的分离系统设计哲学，即保持硬件和软件之间严格的抽象，正变得日益过时。软件和硬件之间的界限正在迅速模糊，取而代之的是协同设计。计算社区必须加强对硬件-软件协同设计的承诺，提升系统抽象到头等重要的地位，并重塑设计原则以满足现代计算的无尽需求。硬件-软件协同设计不是最近才有的创新。为了说明其历史演变，我将其划分为五个相对独立的“阶段”。本文还强调了架构社区在跨学科团队中的日益影响，特别是与ML研究人员的合作，并探讨了为什么当前的协同设计模式在当今计算环境中显得力不从心。此外，本文还将探讨“硬件彩票”这一概念，并探讨减轻其对下一阶段计算创新制约影响的方向。 

---
# The Power of the Pareto Front: Balancing Uncertain Rewards for Adaptive Experimentation in scanning probe microscopy 

**Title (ZH)**: 帕累托前沿的威力：在扫描探针显微镜自适应实验中平衡不确定奖励 

**Authors**: Yu Liu, Sergei V. Kalinin  

**Link**: [PDF](https://arxiv.org/pdf/2504.06525)  

**Abstract**: Automated experimentation has the potential to revolutionize scientific discovery, but its effectiveness depends on well-defined optimization targets, which are often uncertain or probabilistic in real-world settings. In this work, we demonstrate the application of Multi-Objective Bayesian Optimization (MOBO) to balance multiple, competing rewards in autonomous experimentation. Using scanning probe microscopy (SPM) imaging, one of the most widely used and foundational SPM modes, we show that MOBO can optimize imaging parameters to enhance measurement quality, reproducibility, and efficiency. A key advantage of this approach is the ability to compute and analyze the Pareto front, which not only guides optimization but also provides physical insights into the trade-offs between different objectives. Additionally, MOBO offers a natural framework for human-in-the-loop decision-making, enabling researchers to fine-tune experimental trade-offs based on domain expertise. By standardizing high-quality, reproducible measurements and integrating human input into AI-driven optimization, this work highlights MOBO as a powerful tool for advancing autonomous scientific discovery. 

**Abstract (ZH)**: 多目标贝叶斯优化在自主实验中的应用：以扫描探针显微镜成像为例 

---
# Continuous-Variable Quantum Encoding Techniques: A Comparative Study of Embedding Techniques and Their Impact on Machine Learning Performance 

**Title (ZH)**: 连续变量量子编码技术：嵌入技术的比较研究及其对机器学习性能的影响 

**Authors**: Minati Rath, Hema Date  

**Link**: [PDF](https://arxiv.org/pdf/2504.06497)  

**Abstract**: This study explores the intersection of continuous-variable quantum computing (CVQC) and classical machine learning, focusing on CVQC data encoding techniques, including Displacement encoding and squeezing encoding, alongside Instantaneous Quantum Polynomial (IQP) encoding from discrete quantum computing. We perform an extensive empirical analysis to assess the impact of these encoding methods on classical machine learning models, such as Logistic Regression, Support Vector Machines, K-Nearest Neighbors, and ensemble methods like Random Forest and LightGBM. Our findings indicate that CVQC-based encoding methods significantly enhance feature expressivity, resulting in improved classification accuracy and F1 scores, especially in high-dimensional and complex datasets. However, these improvements come with varying computational costs, which depend on the complexity of the encoding and the architecture of the machine learning models. Additionally, we examine the trade-off between quantum expressibility and classical learnability, offering valuable insights into the practical feasibility of incorporating these quantum encodings into real-world applications. This study contributes to the growing body of research on quantum-classical hybrid learning, emphasizing the role of CVQC in advancing quantum data representation and its integration into classical machine learning workflows. 

**Abstract (ZH)**: 本研究探讨了连续变量量子计算（CVQC）与经典机器学习的交叉领域，重点关注CVQC数据编码技术，包括位移编码和压缩编码，以及离散量子计算中的瞬时量子多项式（IQP）编码。通过广泛的实证分析，评估这些编码方法对经典机器学习模型（如逻辑回归、支持向量机、K-最近邻以及随机森林和LightGBM等集成方法）的影响。研究结果表明，基于CVQC的编码方法显著增强了特征表达性，提高了分类准确率和F1分数，特别是在高维和复杂数据集中表现尤为明显。然而，这些改进伴随着不同的计算成本，这取决于编码的复杂性和机器学习模型的架构。此外，本研究还探讨了量子表达性和经典可学习性之间的权衡，为将这些量子编码纳入实际应用提供了宝贵的见解。本研究为量子-经典混合学习领域的研究增添了新的成果，强调了CVQC在推动量子数据表示及其与经典机器学习流程整合中的作用。 

---
# Exploiting Meta-Learning-based Poisoning Attacks for Graph Link Prediction 

**Title (ZH)**: 基于元学习的中毒攻击在图链接预测中的应用 

**Authors**: Mingchen Li, Di Zhuang, Keyu Chen, Dumindu Samaraweera, Morris Chang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06492)  

**Abstract**: Link prediction in graph data utilizes various algorithms and machine learning/deep learning models to predict potential relationships between graph nodes. This technique has found widespread use in numerous real-world applications, including recommendation systems, community networks, and biological structures. However, recent research has highlighted the vulnerability of link prediction models to adversarial attacks, such as poisoning and evasion attacks. Addressing the vulnerability of these models is crucial to ensure stable and robust performance in link prediction applications. While many works have focused on enhancing the robustness of the Graph Convolution Network (GCN) model, the Variational Graph Auto-Encoder (VGAE), a sophisticated model for link prediction, has not been thoroughly investigated in the context of graph adversarial attacks. To bridge this gap, this article proposes an unweighted graph poisoning attack approach using meta-learning techniques to undermine VGAE's link prediction performance. We conducted comprehensive experiments on diverse datasets to evaluate the proposed method and its parameters, comparing it with existing approaches in similar settings. Our results demonstrate that our approach significantly diminishes link prediction performance and outperforms other state-of-the-art methods. 

**Abstract (ZH)**: 图数据中的链接预测利用各种算法和机器学习/深度学习模型来预测图节点之间的潜在关系。这项技术在推荐系统、社区网络和生物结构等众多实际应用中得到了广泛应用。然而，近期的研究表明，链接预测模型容易受到对抗性攻击（如投毒和规避攻击）的影响。确保这些模型在链接预测应用中的稳定和稳健性能至关重要。虽然许多研究集中在增强图卷积网络（GCN）模型的鲁棒性上，但针对图对抗性攻击的变分图自编码器（VGAE）模型尚未得到充分的研究。为此，本文提出了一种基于元学习的无权重图投毒攻击方法，以削弱VGAE的链接预测性能。我们在多种数据集上进行了全面实验，评估了所提出的方法及其参数，并将其与类似设置下的现有方法进行了比较。结果显示，我们的方法显著降低了链接预测性能，并优于其他最先进的方法。 

---
# Federated Neural Architecture Search with Model-Agnostic Meta Learning 

**Title (ZH)**: 基于模型无拘束元学习的联邦神经架构搜索 

**Authors**: Xinyuan Huang, Jiechao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06457)  

**Abstract**: Federated Learning (FL) often struggles with data heterogeneity due to the naturally uneven distribution of user data across devices. Federated Neural Architecture Search (NAS) enables collaborative search for optimal model architectures tailored to heterogeneous data to achieve higher accuracy. However, this process is time-consuming due to extensive search space and retraining. To overcome this, we introduce FedMetaNAS, a framework that integrates meta-learning with NAS within the FL context to expedite the architecture search by pruning the search space and eliminating the retraining stage. Our approach first utilizes the Gumbel-Softmax reparameterization to facilitate relaxation of the mixed operations in the search space. We then refine the local search process by incorporating Model-Agnostic Meta-Learning, where a task-specific learner adapts both weights and architecture parameters (alphas) for individual tasks, while a meta learner adjusts the overall model weights and alphas based on the gradient information from task learners. Following the meta-update, we propose soft pruning using the same trick on search space to gradually sparsify the architecture, ensuring that the performance of the chosen architecture remains robust after pruning which allows for immediate use of the model without retraining. Experimental evaluations demonstrate that FedMetaNAS significantly accelerates the search process by more than 50\% with higher accuracy compared to FedNAS. 

**Abstract (ZH)**: 联邦学习中的元学习神经架构搜索（FedMetaNAS）：通过缩减搜索空间和消除重新训练加速异质数据下的架构搜索 

---
# Evaluating Mutation Techniques in Genetic Algorithm-Based Quantum Circuit Synthesis 

**Title (ZH)**: 基于遗传算法的量子电路合成中突变技术评估 

**Authors**: Michael Kölle, Tom Bintener, Maximilian Zorn, Gerhard Stenzel, Leo Sünkel, Thomas Gabor, Claudia Linnhoff-Popien  

**Link**: [PDF](https://arxiv.org/pdf/2504.06413)  

**Abstract**: Quantum computing leverages the unique properties of qubits and quantum parallelism to solve problems intractable for classical systems, offering unparalleled computational potential. However, the optimization of quantum circuits remains critical, especially for noisy intermediate-scale quantum (NISQ) devices with limited qubits and high error rates. Genetic algorithms (GAs) provide a promising approach for efficient quantum circuit synthesis by automating optimization tasks. This work examines the impact of various mutation strategies within a GA framework for quantum circuit synthesis. By analyzing how different mutations transform circuits, it identifies strategies that enhance efficiency and performance. Experiments utilized a fitness function emphasizing fidelity, while accounting for circuit depth and T operations, to optimize circuits with four to six qubits. Comprehensive hyperparameter testing revealed that combining delete and swap strategies outperformed other approaches, demonstrating their effectiveness in developing robust GA-based quantum circuit optimizers. 

**Abstract (ZH)**: 量子计算利用量子位的独特性质和量子并行性来解决经典系统无法处理的问题，提供无与伦比的计算潜力。然而，量子电路的优化对于具有有限量子位数和高错误率的嘈杂中等规模量子（NISQ）设备来说仍然至关重要。遗传算法（GAs）为通过自动化优化任务高效合成量子电路提供了有希望的方法。本文研究了遗传算法框架下不同变异策略对量子电路合成的影响。通过分析不同变异如何变换电路，确定了能提升效率和性能的策略。实验使用强调保真度的适应度函数，同时考虑电路深度和T操作，对四到六个量子位的电路进行优化。全面的超参数测试显示，结合删除和交换策略优于其他方法，证明了它们在开发稳健的基于遗传算法的量子电路优化器方面的有效性。 

---
# Understanding Machine Unlearning Through the Lens of Mode Connectivity 

**Title (ZH)**: 通过模式连接性视角理解机器卸载 

**Authors**: Jiali Cheng, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2504.06407)  

**Abstract**: Machine Unlearning aims to remove undesired information from trained models without requiring full retraining from scratch. Despite recent advancements, their underlying loss landscapes and optimization dynamics received less attention. In this paper, we investigate and analyze machine unlearning through the lens of mode connectivity - the phenomenon where independently trained models can be connected by smooth low-loss paths in the parameter space. We define and study mode connectivity in unlearning across a range of overlooked conditions, including connections between different unlearning methods, models trained with and without curriculum learning, and models optimized with first-order and secondorder techniques. Our findings show distinct patterns of fluctuation of different evaluation metrics along the curve, as well as the mechanistic (dis)similarity between unlearning methods. To the best of our knowledge, this is the first study on mode connectivity in the context of machine unlearning. 

**Abstract (ZH)**: 机器遗忘旨在从训练模型中移除不希望的信息，而无需从头完全重新训练。尽管最近取得了进展，但其潜在的损失景观和优化动力学仍未得到充分关注。在本文中，我们通过模式连通性的视角来研究和分析机器遗忘——即独立训练的模型可以通过参数空间中的光滑低损失路径相互连接的现象。我们在一系列未被关注的条件下定义和研究了遗忘过程中的模式连通性，包括不同遗忘方法之间的连接、带有和不带有渐进学习的模型以及使用一阶和二阶技术优化的模型。我们的发现显示了不同评估指标沿曲线波动的不同模式，以及不同遗忘方法之间的（不）相似性机制。据我们所知，这是首次在机器遗忘的背景下研究模式连通性的研究。 

---
# Physical spline for denoising object trajectory data by combining splines, ML feature regression and model knowledge 

**Title (ZH)**: 物理样条：结合样条、机器学习特征回归和模型知识的对象轨迹数据去噪方法 

**Authors**: Jonas Torzewski  

**Link**: [PDF](https://arxiv.org/pdf/2504.06404)  

**Abstract**: This article presents a method for estimating the dynamic driving states (position, velocity, acceleration and heading) from noisy measurement data. The proposed approach is effective with both complete and partial observations, producing refined trajectory signals with kinematic consistency, ensuring that velocity is the integral of acceleration and position is the integral of velocity. Additionally, the method accounts for the constraint that vehicles can only move in the direction of their orientation. The method is implemented as a configurable python library that also enables trajectory estimation solely based on position data. Regularization is applied to prevent extreme state variations. A key application is enhancing recorded trajectory data for use as reference inputs in machine learning models. At the end, the article presents the results of the method along with a comparison to ground truth data. 

**Abstract (ZH)**: 本文提出了一种从噪声测量数据中估计动态驾驶状态（位置、速度、加速度和航向）的方法。所提出的方案能够在完整和部分观测的情况下有效地工作，生成具有动力学一致性的精细轨迹信号，确保速度是加速度的积分，位置是速度的积分。此外，该方法考虑了车辆只能在其方向上移动的约束。该方法实现为一个可配置的Python库，还能够仅基于位置数据进行轨迹估计。应用正则化以防止状态变化极端。主要应用是增强记录的轨迹数据，使其适合作为机器学习模型的参考输入。最后，文章呈现了该方法的结果，并与真实数据进行了比较。 

---
# MM-STFlowNet: A Transportation Hub-Oriented Multi-Mode Passenger Flow Prediction Method via Spatial-Temporal Dynamic Graph Modeling 

**Title (ZH)**: MM-STFlowNet: 基于时空动态图建模的多模态交通枢纽客流量预测方法 

**Authors**: Ronghui Zhang, Wenbin Xing, Mengran Li, Zihan Wang, Junzhou Chen, Xiaolei Ma, Zhiyuan Liu, Zhengbing He  

**Link**: [PDF](https://arxiv.org/pdf/2504.06325)  

**Abstract**: Accurate and refined passenger flow prediction is essential for optimizing the collaborative management of multiple collection and distribution modes in large-scale transportation hubs. Traditional methods often focus only on the overall passenger volume, neglecting the interdependence between different modes within the hub. To address this limitation, we propose MM-STFlowNet, a comprehensive multi-mode prediction framework grounded in dynamic spatial-temporal graph modeling. Initially, an integrated temporal feature processing strategy is implemented using signal decomposition and convolution techniques to address data spikes and high volatility. Subsequently, we introduce the Spatial-Temporal Dynamic Graph Convolutional Recurrent Network (STDGCRN) to capture detailed spatial-temporal dependencies across multiple traffic modes, enhanced by an adaptive channel attention mechanism. Finally, the self-attention mechanism is applied to incorporate various external factors, further enhancing prediction accuracy. Experiments on a real-world dataset from Guangzhounan Railway Station in China demonstrate that MM-STFlowNet achieves state-of-the-art performance, particularly during peak periods, providing valuable insight for transportation hub management. 

**Abstract (ZH)**: 多模式精细 passenger flow 预测对于大型交通枢纽多模式协作管理的优化至关重要。传统方法往往只关注整体乘客流量，忽视了枢纽内部不同模式之间的相互依赖性。为解决这一局限性，我们提出了基于动态空时图建模的全面多模式预测框架 MM-STFlowNet。首先，通过信号分解和卷积技术实现集成时间特征处理策略，以应对数据尖峰和高波动性。随后，引入空间-时间动态图卷积循环网络(STDGCRN)，并结合自适应通道注意力机制以捕捉多个交通模式之间的详细空时依赖性。最后，应用自注意力机制融入各种外部因素，进一步提升预测精度。实验结果表明，MM-STFlowNet 在真实数据集（来自中国广州南站）上达到了最先进的性能，特别是在高峰时段，为交通枢纽管理提供了有价值的见解。 

---
# Assessing employment and labour issues implicated by using AI 

**Title (ZH)**: 评估使用AI涉及的就业与劳动问题 

**Authors**: Thijs Willems, Darion Jin Hotan, Jiawen Cheryl Tang, Norakmal Hakim bin Norhashim, King Wang Poon, Zi An Galvyn Goh, Radha Vinod  

**Link**: [PDF](https://arxiv.org/pdf/2504.06322)  

**Abstract**: This chapter critiques the dominant reductionist approach in AI and work studies, which isolates tasks and skills as replaceable components. Instead, it advocates for a systemic perspective that emphasizes the interdependence of tasks, roles, and workplace contexts. Two complementary approaches are proposed: an ethnographic, context-rich method that highlights how AI reconfigures work environments and expertise; and a relational task-based analysis that bridges micro-level work descriptions with macro-level labor trends. The authors argue that effective AI impact assessments must go beyond predicting automation rates to include ethical, well-being, and expertise-related questions. Drawing on empirical case studies, they demonstrate how AI reshapes human-technology relations, professional roles, and tacit knowledge practices. The chapter concludes by calling for a human-centric, holistic framework that guides organizational and policy decisions, balancing technological possibilities with social desirability and sustainability of work. 

**Abstract (ZH)**: 本章批评人工智能和工作研究中的主导还原论方法，该方法孤立并视为可替代的任务和技能成分。相反，它倡导一种系统视角，强调任务、角色和工作场所环境之间的相互依存关系。提出了两种互补的方法：一种是富含情境的民族志方法，强调人工智能如何重新配置工作环境和专业知识；另一种是关系导向的任务分析方法，将微观层面的工作描述与宏观层面的劳动趋势联系起来。作者认为，有效的AI影响评估应超越预测自动化率，包括道德、福祉和专业知识相关的问题。基于实证案例研究，他们展示了人工智能如何重塑人机关系、专业角色和默会知识实践。本章结尾呼吁采用以人为中心、综合性框架来指导组织和政策决策，平衡技术可能性与社会可接受性和工作的可持续性。 

---
# Hybrid Temporal Differential Consistency Autoencoder for Efficient and Sustainable Anomaly Detection in Cyber-Physical Systems 

**Title (ZH)**: 基于时空差分一致性自编码器的高效可持续的网络物理系统异常检测 

**Authors**: Michael Somma  

**Link**: [PDF](https://arxiv.org/pdf/2504.06320)  

**Abstract**: Cyberattacks on critical infrastructure, particularly water distribution systems, have increased due to rapid digitalization and the integration of IoT devices and industrial control systems (ICS). These cyber-physical systems (CPS) introduce new vulnerabilities, requiring robust and automated intrusion detection systems (IDS) to mitigate potential threats. This study addresses key challenges in anomaly detection by leveraging time correlations in sensor data, integrating physical principles into machine learning models, and optimizing computational efficiency for edge applications. We build upon the concept of temporal differential consistency (TDC) loss to capture the dynamics of the system, ensuring meaningful relationships between dynamic states. Expanding on this foundation, we propose a hybrid autoencoder-based approach, referred to as hybrid TDC-AE, which extends TDC by incorporating both deterministic nodes and conventional statistical nodes. This hybrid structure enables the model to account for non-deterministic processes. Our approach achieves state-of-the-art classification performance while improving time to detect anomalies by 3%, outperforming the BATADAL challenge leader without requiring domain-specific knowledge, making it broadly applicable. Additionally, it maintains the computational efficiency of conventional autoencoders while reducing the number of fully connected layers, resulting in a more sustainable and efficient solution. The method demonstrates how leveraging physics-inspired consistency principles enhances anomaly detection and strengthens the resilience of cyber-physical systems. 

**Abstract (ZH)**: 基于时序差分一致性混合自编码器的 cyber-物理系统异常检测方法 

---
# DMol: A Schedule-Driven Diffusion Model for Highly Efficient and Versatile Molecule Generation 

**Title (ZH)**: DMol：基于调度驱动的高效多功能分子生成扩散模型 

**Authors**: Peizhi Niu, Yu-Hsiang Wang, Vishal Rana, Chetan Rupakheti, Abhishek Pandey, Olgica Milenkovic  

**Link**: [PDF](https://arxiv.org/pdf/2504.06312)  

**Abstract**: We introduce a new graph diffusion model for small molecule generation, \emph{DMol}, which outperforms the state-of-the-art DiGress model in terms of validity by roughly $1.5\%$ across all benchmarking datasets while reducing the number of diffusion steps by at least $10$-fold, and the running time to roughly one half. The performance improvements are a result of a careful change in the objective function and a ``graph noise" scheduling approach which, at each diffusion step, allows one to only change a subset of nodes of varying size in the molecule graph. Another relevant property of the method is that it can be easily combined with junction-tree-like graph representations that arise by compressing a collection of relevant ring structures into supernodes. Unlike classical junction-tree techniques that involve VAEs and require complicated reconstruction steps, compressed DMol directly performs graph diffusion on a graph that compresses only a carefully selected set of frequent carbon rings into supernodes, which results in straightforward sample generation. This compressed DMol method offers additional validity improvements over generic DMol of roughly $2\%$, increases the novelty of the method, and further improves the running time due to reductions in the graph size. 

**Abstract (ZH)**: 一种新的小分子生成图扩散模型DMol：与最先进的DiGress模型相比，在所有基准数据集中有效性提高了大约1.5%，同时减少了一定比例的扩散步数，并将运行时间缩短至大约一半。该性能提升得益于目标函数的精细调整和一种“图噪声”调度方法，在每次扩散步中仅改变分子图中大小可变的子集节点。该方法还可以与由压缩相关环结构成的超节点表示的类似区间树图表示轻松结合。与涉及VAEs和复杂重构步骤的经典区间树技术不同，压缩DMol直接在压缩了精心选择的常见碳环成超节点的图上执行图扩散，从而实现直接样本生成。该压缩DMol方法还为通用DMol提供了约2%的有效性改进，增加了方法的新颖性，并进一步通过减少图的大小来提高运行时间。 

---
# Predicting Survivability of Cancer Patients with Metastatic Patterns Using Explainable AI 

**Title (ZH)**: 使用可解释AI预测具有转移模式的癌症患者的生存率 

**Authors**: Polycarp Nalela, Deepthi Rao, Praveen Rao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06306)  

**Abstract**: Cancer remains a leading global health challenge and a major cause of mortality. This study leverages machine learning (ML) to predict the survivability of cancer patients with metastatic patterns using the comprehensive MSK-MET dataset, which includes genomic and clinical data from 25,775 patients across 27 cancer types. We evaluated five ML models-XGBoost, Naïve Bayes, Decision Tree, Logistic Regression, and Random Fores using hyperparameter tuning and grid search. XGBoost emerged as the best performer with an area under the curve (AUC) of 0.82. To enhance model interpretability, SHapley Additive exPlanations (SHAP) were applied, revealing key predictors such as metastatic site count, tumor mutation burden, fraction of genome altered, and organ-specific metastases. Further survival analysis using Kaplan-Meier curves, Cox Proportional Hazards models, and XGBoost Survival Analysis identified significant predictors of patient outcomes, offering actionable insights for clinicians. These findings could aid in personalized prognosis and treatment planning, ultimately improving patient care. 

**Abstract (ZH)**: 癌症仍然是全球健康的主要挑战和主要死因。本研究利用机器学习（ML）方法，通过包含27种癌症类型、25,775名患者的综合MSK-MET数据集（包括基因组和临床数据），预测具有转移模式的癌症患者的存活率。我们评估了五种机器学习模型——XGBoost、朴素贝叶斯、决策树、逻辑回归和随机森林，并进行了超参数调优和网格搜索。XGBoost表现出色，其曲线下面积（AUC）为0.82。为进一步提高模型的可解释性，我们应用了SHapley Additive exPlanations（SHAP），揭示了关键预测因子，如转移部位数量、肿瘤突变负担、基因组改变比例以及器官特异性转移。通过使用Kaplan-Meier曲线、Cox比例风险模型和XGBoost生存分析进一步进行生存分析，确定了对患者结果有显著影响的预测因子，为临床提供了可操作的见解。这些发现有助于个性化预后和治疗计划的制定，最终提高患者护理质量。 

---
# Well2Flow: Reconstruction of reservoir states from sparse wells using score-based generative models 

**Title (ZH)**: Well2Flow: 从稀疏井信息重构储层状态的评分基于生成模型方法 

**Authors**: Shiqin Zeng, Haoyun Li, Abhinav Prakash Gahlot, Felix J. Herrmann  

**Link**: [PDF](https://arxiv.org/pdf/2504.06305)  

**Abstract**: This study investigates the use of score-based generative models for reservoir simulation, with a focus on reconstructing spatially varying permeability and saturation fields in saline aquifers, inferred from sparse observations at two well locations. By modeling the joint distribution of permeability and saturation derived from high-fidelity reservoir simulations, the proposed neural network is trained to learn the complex spatiotemporal dynamics governing multiphase fluid flow in porous media. During inference, the framework effectively reconstructs both permeability and saturation fields by conditioning on sparse vertical profiles extracted from well log data. This approach introduces a novel methodology for incorporating physical constraints and well log guidance into generative models, significantly enhancing the accuracy and physical plausibility of the reconstructed subsurface states. Furthermore, the framework demonstrates strong generalization capabilities across varying geological scenarios, highlighting its potential for practical deployment in data-scarce reservoir management tasks. 

**Abstract (ZH)**: 基于评分生成模型的储层模拟研究：稀疏井筒观测数据驱动的盐水含水层渗透率和饱和度场重建 

---
# Resurrecting Socrates in the Age of AI: A Study Protocol for Evaluating a Socratic Tutor to Support Research Question Development in Higher Education 

**Title (ZH)**: 在人工智能时代复活苏格拉底：一个关于评估苏格拉底式辅导以支持高等教育研究问题发展的研究方案 

**Authors**: Ben Degen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06294)  

**Abstract**: Formulating research questions is a foundational yet challenging academic skill, one that generative AI systems often oversimplify by offering instant answers at the expense of student reflection. This protocol lays out a study grounded in constructivist learning theory to evaluate a novel AI-based Socratic Tutor, designed to foster cognitive engagement and scaffold research question development in higher education. Anchored in dialogic pedagogy, the tutor engages students through iterative, reflective questioning, aiming to promote System 2 thinking and counteract overreliance on AI-generated outputs. In a quasi-experimental design, approximately 80 German pre-service biology teacher students will be randomly assigned to one of two groups: an AI Socratic Tutor condition and an uninstructed chatbot control. Across multiple cycles, students are expected to formulate research questions based on background texts, with quality assessed through double-blind expert review. The study also examines transfer of skills to novel phenomena and captures student perceptions through mixed-methods analysis, including surveys, interviews and reflective journals. This study aims to advance the understanding of how generative AI can be pedagogically aligned to support, not replace, human cognition and offers design principles for human-AI collaboration in education. 

**Abstract (ZH)**: 基于建构主义学习理论的新型基于AI的苏格拉底式导师研究：促进高等教育中认知参与和研究问题开发的对话式教学实践与评估 

---
# A Cascaded Architecture for Extractive Summarization of Multimedia Content via Audio-to-Text Alignment 

**Title (ZH)**: 基于音频到文本对齐的多媒体内容提取性总结的级联架构 

**Authors**: Tanzir Hossain, Ar-Rafi Islam, Md. Sabbir Hossain, Annajiat Alim Rasel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06275)  

**Abstract**: This study presents a cascaded architecture for extractive summarization of multimedia content via audio-to-text alignment. The proposed framework addresses the challenge of extracting key insights from multimedia sources like YouTube videos. It integrates audio-to-text conversion using Microsoft Azure Speech with advanced extractive summarization models, including Whisper, Pegasus, and Facebook BART XSum. The system employs tools such as Pytube, Pydub, and SpeechRecognition for content retrieval, audio extraction, and transcription. Linguistic analysis is enhanced through named entity recognition and semantic role labeling. Evaluation using ROUGE and F1 scores demonstrates that the cascaded architecture outperforms conventional summarization methods, despite challenges like transcription errors. Future improvements may include model fine-tuning and real-time processing. This study contributes to multimedia summarization by improving information retrieval, accessibility, and user experience. 

**Abstract (ZH)**: 本研究提出了一种级联架构，通过音频到文本对齐实现多媒体内容的摘要提取。提出的框架解决了从YouTube视频等多媒体源中提取关键见解的挑战。该框架结合了Microsoft Azure Speech的音频到文本转换与Whisper、Pegasus和Facebook BART XSum等先进的摘要模型。系统利用Pytube、Pydub和SpeechRecognition等工具进行内容检索、音频提取和转录。通过命名实体识别和语义角色标注增强语言分析。使用ROUGE和F1评分的评估表明，级联架构在面对转录错误等挑战的情况下仍优于传统的摘要方法。未来可能的改进包括模型微调和实时处理。本研究通过提高信息检索、可访问性和用户体验，为多媒体摘要做出了贡献。 

---
# Joint Group Profiling and Recommendation via Deep Neural Network-based Multi-Task Learning 

**Title (ZH)**: 基于深度神经网络的多任务学习联合群体画像与推荐 

**Authors**: Ngoc Luyen Le, Marie-Hélène Abel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06274)  

**Abstract**: Group recommender systems aim to generate recommendations that align with the collective preferences of a group, introducing challenges that differ significantly from those in individual recommendation scenarios. This paper presents Joint Group Profiling and Recommendation via Deep Neural Network-based Multi-Task Learning, a framework that unifies group profiling and recommendation tasks within a single model. By jointly learning these tasks, the model develops a deeper understanding of group dynamics, leading to improved recommendation accuracy. The shared representations between the two tasks facilitate the discovery of latent features essential to both, resulting in richer and more informative group embeddings. To further enhance performance, an attention mechanism is integrated to dynamically evaluate the relevance of different group features and item attributes, ensuring the model prioritizes the most impactful information. Experiments and evaluations on real-world datasets demonstrate that our multi-task learning approach consistently outperforms baseline models in terms of accuracy, validating its effectiveness and robustness. 

**Abstract (ZH)**: 基于深度神经网络多任务学习的联合群体画像与推荐框架 

---
# A Diverse and Effective Retrieval-Based Debt Collection System with Expert Knowledge 

**Title (ZH)**: 一种融合专家知识的多样化和有效债务催收检索系统 

**Authors**: Jiaming Luo, Weiyi Luo, Guoqing Sun, Mengchen Zhu, Haifeng Tang, Kunyao Lan, Mengyue Wu, Kenny Q. Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06273)  

**Abstract**: Designing effective debt collection systems is crucial for improving operational efficiency and reducing costs in the financial industry. However, the challenges of maintaining script diversity, contextual relevance, and coherence make this task particularly difficult. This paper presents a debt collection system based on real debtor-collector data from a major commercial bank. We construct a script library from real-world debt collection conversations, and propose a two-stage retrieval based response system for contextual relevance. Experimental results show that our system improves script diversity, enhances response relevance, and achieves practical deployment efficiency through knowledge distillation. This work offers a scalable and automated solution, providing valuable insights for advancing debt collection practices in real-world applications. 

**Abstract (ZH)**: 设计有效的债务催收系统对于提高金融行业的运营效率和降低成本至关重要。然而，保持脚本多样性、上下文相关性和连贯性的挑战使得这项任务尤为困难。本文基于一家主要商业银行的真实债务人-催收员对话数据，提出了一种债务催收系统。我们从实际的债务催收对话中构建了一个脚本库，并提出了一种基于两阶段检索的响应系统，以提高上下文相关性。实验结果表明，我们的系统通过知识蒸馏提高了脚本多样性、增强了响应的相关性，并实现了实际部署的效率。本工作提供了一种可扩展的自动化解决方案，为实际应用中的债务催收实践提供了宝贵见解。 

---
# Addressing Cold-start Problem in Click-Through Rate Prediction via Supervised Diffusion Modeling 

**Title (ZH)**: 基于监督扩散 modeling 解决点击率预测中的冷启动问题 

**Authors**: Wenqiao Zhu, Lulu Wang, Jun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06270)  

**Abstract**: Predicting Click-Through Rates is a crucial function within recommendation and advertising platforms, as the output of CTR prediction determines the order of items shown to users. The Embedding \& MLP paradigm has become a standard approach for industrial recommendation systems and has been widely deployed. However, this paradigm suffers from cold-start problems, where there is either no or only limited user action data available, leading to poorly learned ID embeddings. The cold-start problem hampers the performance of new items. To address this problem, we designed a novel diffusion model to generate a warmed-up embedding for new items. Specifically, we define a novel diffusion process between the ID embedding space and the side information space. In addition, we can derive a sub-sequence from the diffusion steps to expedite training, given that our diffusion model is non-Markovian. Our diffusion model is supervised by both the variational inference and binary cross-entropy objectives, enabling it to generate warmed-up embeddings for items in both the cold-start and warm-up phases. Additionally, we have conducted extensive experiments on three recommendation datasets. The results confirmed the effectiveness of our approach. 

**Abstract (ZH)**: 预测点击率是推荐和广告平台中的一项关键功能，输出的点击率预测结果决定了展示给用户的物品顺序。嵌入与MLP范式已成为工业推荐系统的一项标准方法，并得到了广泛应用。然而，该范式在冷启动问题上存在局限，即缺乏或仅有有限的用户行为数据，导致ID嵌入学习效果不佳。冷启动问题妨碍了新项目的表现。为解决这一问题，我们设计了一个新型扩散模型来生成新项目的预热嵌入。具体而言，我们在ID嵌入空间与辅助信息空间之间定义了一个新型的扩散过程。此外，在我们的扩散模型是非马尔可夫模型的情况下，可以从扩散步骤中提取子序列以加速训练。我们的扩散模型同时受变分推断和二元交叉熵目标函数的监督，使其能够在冷启动和预热阶段为项目生成预热嵌入。我们还在三个推荐数据集上进行了广泛的实验。实验结果证实了我们方法的有效性。 

---
# EXCLAIM: An Explainable Cross-Modal Agentic System for Misinformation Detection with Hierarchical Retrieval 

**Title (ZH)**: EXCLAIM：一种具有层次检索的可解释跨模态代理系统以检测 misinformation 

**Authors**: Yin Wu, Zhengxuan Zhang, Fuling Wang, Yuyu Luo, Hui Xiong, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06269)  

**Abstract**: Misinformation continues to pose a significant challenge in today's information ecosystem, profoundly shaping public perception and behavior. Among its various manifestations, Out-of-Context (OOC) misinformation is particularly obscure, as it distorts meaning by pairing authentic images with misleading textual narratives. Existing methods for detecting OOC misinformation predominantly rely on coarse-grained similarity metrics between image-text pairs, which often fail to capture subtle inconsistencies or provide meaningful explainability. While multi-modal large language models (MLLMs) demonstrate remarkable capabilities in visual reasoning and explanation generation, they have not yet demonstrated the capacity to address complex, fine-grained, and cross-modal distinctions necessary for robust OOC detection. To overcome these limitations, we introduce EXCLAIM, a retrieval-based framework designed to leverage external knowledge through multi-granularity index of multi-modal events and entities. Our approach integrates multi-granularity contextual analysis with a multi-agent reasoning architecture to systematically evaluate the consistency and integrity of multi-modal news content. Comprehensive experiments validate the effectiveness and resilience of EXCLAIM, demonstrating its ability to detect OOC misinformation with 4.3% higher accuracy compared to state-of-the-art approaches, while offering explainable and actionable insights. 

**Abstract (ZH)**: Out-of-Context misinformation continues to pose a significant challenge in today's information ecosystem, profoundly shaping public perception and behavior. To address its complex and fine-grained distinctions, we introduce EXCLAIM, a retrieval-based framework that leverages external knowledge through multi-granularity indexing of multi-modal events and entities. 

---
# Multi-objective Optimization in CPU Design Space Exploration: Attention is All You Need 

**Title (ZH)**: CPU 设计空间探索中的多目标优化：只需注意力机制 

**Authors**: Runzhen Xue, Hao Wu, Mingyu Yan, Ziheng Xiao, Xiaochun Ye, Dongrui Fan  

**Link**: [PDF](https://arxiv.org/pdf/2410.18368)  

**Abstract**: Design space exploration (DSE) enables architects to systematically evaluate various design options, guiding decisions on the most suitable configurations to meet specific objectives such as optimizing performance, power, and area. However, the growing complexity of modern CPUs has dramatically increased the number of micro-architectural parameters and expanded the overall design space, making DSE more challenging and time-consuming. Existing DSE frameworks struggle in large-scale design spaces due to inaccurate models and limited insights into parameter impact, hindering efficient identification of optimal micro-architectures within tight timeframes.
In this work, we introduce AttentionDSE. Its key idea is to use the attention mechanism to establish a direct mapping of micro-architectural parameters to their contributions to predicted performance. This approach enhances both the prediction accuracy and interpretability of the performance model. Furthermore, the weights are dynamically adjusted, enabling the model to respond to design changes and effectively pinpoint the key micro-architectural parameters/components responsible for performance bottlenecks. Thus, AttentionDSE accurately, purposefully, and rapidly discovers optimal designs. Experiments on SPEC 2017 demonstrate that AttentionDSE significantly reduces exploration time by over 80\% and achieves 3.9\% improvement in Pareto Hypervolume compared to state-of-the-art DSE frameworks while maintaining superior prediction accuracy and efficiency with an increasing number of parameters. 

**Abstract (ZH)**: 设计空间探索（DSE）使架构师能够系统地评估各种设计选项，指导优化性能、功率和面积等特定目标的最佳配置决策。然而，现代CPU日益复杂性增加了微观架构参数的数量，扩大了整个设计空间，使得DSE更加具有挑战性和耗时性。现有的DSE框架在大规模设计空间中受限于不准确的模型和对参数影响的有限洞察，阻碍了在紧凑的时间框架内高效识别最优微观架构。

在这项工作中，我们提出了AttentionDSE。其核心思想是利用注意力机制建立微观架构参数与其对预测性能贡献之间的直接映射。这种方法增强了性能模型的预测准确性和可解释性。此外，权重动态调整，使模型能够响应设计更改，并有效定位导致性能瓶颈的关键微观架构参数/组件。因此，AttentionDSE能够准确、有针对性和快速地发现最优设计。实验结果表明，AttentionDSE在SPEC 2017上显著减少了探索时间超过80%，并在参数数量增加的同时，实现了3.9%的帕累托hypervolume改进，同时保持了更高的预测准确性和效率。 

---
