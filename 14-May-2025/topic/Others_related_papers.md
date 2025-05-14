# Constrained Factor Graph Optimization for Robust Networked Pedestrian Inertial Navigation 

**Title (ZH)**: 约束因子图优化在鲁棒网络行人惯性导航中的应用 

**Authors**: Yingjie Hu, Wang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08229)  

**Abstract**: This paper presents a novel constrained Factor Graph Optimization (FGO)-based approach for networked inertial navigation in pedestrian localization. To effectively mitigate the drift inherent in inertial navigation solutions, we incorporate kinematic constraints directly into the nonlinear optimization framework. Specifically, we utilize equality constraints, such as Zero-Velocity Updates (ZUPTs), and inequality constraints representing the maximum allowable distance between body-mounted Inertial Measurement Units (IMUs) based on human anatomical limitations. While equality constraints are straightforwardly integrated as error factors, inequality constraints cannot be explicitly represented in standard FGO formulations. To address this, we introduce a differentiable softmax-based penalty term in the FGO cost function to enforce inequality constraints smoothly and robustly. The proposed constrained FGO approach leverages temporal correlations across multiple epochs, resulting in optimal state trajectory estimates while consistently maintaining constraint satisfaction. Experimental results confirm that our method outperforms conventional Kalman filter approaches, demonstrating its effectiveness and robustness for pedestrian navigation. 

**Abstract (ZH)**: 基于约束因子图优化的网络惯性导航行人定位新方法 

---
# PRISM: Complete Online Decentralized Multi-Agent Pathfinding with Rapid Information Sharing using Motion Constraints 

**Title (ZH)**: PRISM: 完整的基于运动约束的快速信息共享在线去中心化多智能体路径规划算法 

**Authors**: Hannah Lee, Zachary Serlin, James Motes, Brendan Long, Marco Morales, Nancy M. Amato  

**Link**: [PDF](https://arxiv.org/pdf/2505.08025)  

**Abstract**: We introduce PRISM (Pathfinding with Rapid Information Sharing using Motion Constraints), a decentralized algorithm designed to address the multi-task multi-agent pathfinding (MT-MAPF) problem. PRISM enables large teams of agents to concurrently plan safe and efficient paths for multiple tasks while avoiding collisions. It employs a rapid communication strategy that uses information packets to exchange motion constraint information, enhancing cooperative pathfinding and situational awareness, even in scenarios without direct communication. We prove that PRISM resolves and avoids all deadlock scenarios when possible, a critical challenge in decentralized pathfinding. Empirically, we evaluate PRISM across five environments and 25 random scenarios, benchmarking it against the centralized Conflict-Based Search (CBS) and the decentralized Token Passing with Task Swaps (TPTS) algorithms. PRISM demonstrates scalability and solution quality, supporting 3.4 times more agents than CBS and handling up to 2.5 times more tasks in narrow passage environments than TPTS. Additionally, PRISM matches CBS in solution quality while achieving faster computation times, even under low-connectivity conditions. Its decentralized design reduces the computational burden on individual agents, making it scalable for large environments. These results confirm PRISM's robustness, scalability, and effectiveness in complex and dynamic pathfinding scenarios. 

**Abstract (ZH)**: PRISM (基于运动约束的信息快速共享路径发现):一种分布式多任务多agent路径规划算法 

---
# Cost Function Estimation Using Inverse Reinforcement Learning with Minimal Observations 

**Title (ZH)**: 基于最少观测使用的逆强化学习成本函数估计 

**Authors**: Sarmad Mehrdad, Avadesh Meduri, Ludovic Righetti  

**Link**: [PDF](https://arxiv.org/pdf/2505.08619)  

**Abstract**: We present an iterative inverse reinforcement learning algorithm to infer optimal cost functions in continuous spaces. Based on a popular maximum entropy criteria, our approach iteratively finds a weight improvement step and proposes a method to find an appropriate step size that ensures learned cost function features remain similar to the demonstrated trajectory features. In contrast to similar approaches, our algorithm can individually tune the effectiveness of each observation for the partition function and does not need a large sample set, enabling faster learning. We generate sample trajectories by solving an optimal control problem instead of random sampling, leading to more informative trajectories. The performance of our method is compared to two state of the art algorithms to demonstrate its benefits in several simulated environments. 

**Abstract (ZH)**: 我们提出了一个迭代逆强化学习算法以在连续空间中推断最优成本函数。基于流行的最大熵标准，该方法迭代地寻找一个权重改进步骤，并提出了一种方法来找到一个适当的步长，以确保学习到的成本函数特征与展示轨迹特征保持相似。与相似的方法不同，我们的算法可以单独调整每个观察对分区函数的有效性，并不需要大量样本集，从而实现更快的学习。我们通过求解最优控制问题生成样本轨迹，而不是随机采样，从而得到更具信息量的轨迹。我们将该方法的性能与两种最先进的算法进行了比较，以在多个模拟环境中展示其优势。 

---
# Graph-Based Floor Separation Using Node Embeddings and Clustering of WiFi Trajectories 

**Title (ZH)**: 基于图的楼层分离：节点嵌入与WiFi轨迹聚类方法 

**Authors**: Rabia Yasa Kostas, Kahraman Kostas  

**Link**: [PDF](https://arxiv.org/pdf/2505.08088)  

**Abstract**: Indoor positioning systems (IPSs) are increasingly vital for location-based services in complex multi-storey environments. This study proposes a novel graph-based approach for floor separation using Wi-Fi fingerprint trajectories, addressing the challenge of vertical localization in indoor settings. We construct a graph where nodes represent Wi-Fi fingerprints, and edges are weighted by signal similarity and contextual transitions. Node2Vec is employed to generate low-dimensional embeddings, which are subsequently clustered using K-means to identify distinct floors. Evaluated on the Huawei University Challenge 2021 dataset, our method outperforms traditional community detection algorithms, achieving an accuracy of 68.97%, an F1- score of 61.99%, and an Adjusted Rand Index of 57.19%. By publicly releasing the preprocessed dataset and implementation code, this work contributes to advancing research in indoor positioning. The proposed approach demonstrates robustness to signal noise and architectural complexities, offering a scalable solution for floor-level localization. 

**Abstract (ZH)**: 基于Wi-Fi指纹轨迹的图表示方法在室内多层环境中的楼层分离研究 

---
# ARC-NCA: Towards Developmental Solutions to the Abstraction and Reasoning Corpus 

**Title (ZH)**: ARC-NCA: 向抽象与推理语料库的发展性解决方案迈进 

**Authors**: Etienne Guichard, Felix Reimers, Mia Kvalsund, Mikkel Lepperød, Stefano Nichele  

**Link**: [PDF](https://arxiv.org/pdf/2505.08778)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC), later renamed ARC-AGI, poses a fundamental challenge in artificial general intelligence (AGI), requiring solutions that exhibit robust abstraction and reasoning capabilities across diverse tasks, while only few (with median count of three) correct examples are presented. While ARC-AGI remains very challenging for artificial intelligence systems, it is rather easy for humans. This paper introduces ARC-NCA, a developmental approach leveraging standard Neural Cellular Automata (NCA) and NCA enhanced with hidden memories (EngramNCA) to tackle the ARC-AGI benchmark. NCAs are employed for their inherent ability to simulate complex dynamics and emergent patterns, mimicking developmental processes observed in biological systems. Developmental solutions may offer a promising avenue for enhancing AI's problem-solving capabilities beyond mere training data extrapolation. ARC-NCA demonstrates how integrating developmental principles into computational models can foster adaptive reasoning and abstraction. We show that our ARC-NCA proof-of-concept results may be comparable to, and sometimes surpass, that of ChatGPT 4.5, at a fraction of the cost. 

**Abstract (ZH)**: ARC抽象与推理语料库（ARC-AGI）及其在通用人工智能中的挑战：一种基于标准神经细胞自动机（NCA）及其增强版（EngramNCA）的发育性方法 

---
# A Study of Data-driven Methods for Inventory Optimization 

**Title (ZH)**: 基于数据驱动方法的库存优化研究 

**Authors**: Lee Yeung Ping, Patrick Wong, Tan Cheng Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.08673)  

**Abstract**: This paper shows a comprehensive analysis of three algorithms (Time Series, Random Forest (RF) and Deep Reinforcement Learning) into three inventory models (the Lost Sales, Dual-Sourcing and Multi-Echelon Inventory Model). These methodologies are applied in the supermarket context. The main purpose is to analyse efficient methods for the data-driven. Their possibility, potential and current challenges are taken into consideration in this report. By comparing the results in each model, the effectiveness of each algorithm is evaluated based on several key performance indicators, including forecast accuracy, adaptability to market changes, and overall impact on inventory costs and customer satisfaction levels. The data visualization tools and statistical metrics are the indicators for the comparisons and show some obvious trends and patterns that can guide decision-making in inventory management. These tools enable managers to not only track the performance of different algorithms in real-time but also to drill down into specific data points to understand the underlying causes of inventory fluctuations. This level of detail is crucial for pinpointing inefficiencies and areas for improvement within the supply chain. 

**Abstract (ZH)**: 本文对三种算法（时间序列、随机森林（RF）和深度强化学习）在三种库存模型（缺货损失模型、双重 sourcing 模型和多层级库存模型）中的应用进行了全面分析。这些方法在超市背景下应用。主要目的 是分析数据驱动方法的有效性，并考虑其可能性、潜在优势及其当前挑战。通过在每个模型中比较结果，根据包括预测准确性、适应市场变化的能力以及对库存成本和顾客满意度的整体影响等关键绩效指标来评估每种算法的有效性。数据可视化工具和统计指标用于比较，并揭示了一些明显趋势和模式，这些模式可以指导库存管理中的决策。这些工具使管理者不仅能够实时监控不同算法的性能，还能深入特定数据点以理解库存波动的根本原因。这种详细的分析对于识别供应链中的不效率和改进领域至关重要。 

---
# WixQA: A Multi-Dataset Benchmark for Enterprise Retrieval-Augmented Generation 

**Title (ZH)**: WixQA：面向企业的检索增强生成多数据集基准 

**Authors**: Dvir Cohen, Lin Burg, Sviatoslav Pykhnivskyi, Hagit Gur, Stanislav Kovynov, Olga Atzmon, Gilad Barkan  

**Link**: [PDF](https://arxiv.org/pdf/2505.08643)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a cornerstone of modern question answering (QA) systems, enabling grounded answers based on external knowledge. Although recent progress has been driven by open-domain datasets, enterprise QA systems need datasets that mirror the concrete, domain-specific issues users raise in day-to-day support scenarios. Critically, evaluating end-to-end RAG systems requires benchmarks comprising not only question--answer pairs but also the specific knowledge base (KB) snapshot from which answers were derived. To address this need, we introduce WixQA, a benchmark suite featuring QA datasets precisely grounded in the released KB corpus, enabling holistic evaluation of retrieval and generation components. WixQA includes three distinct QA datasets derived from this http URL customer support interactions and grounded in a snapshot of the public Wix Help Center KB: (i) WixQA-ExpertWritten, 200 real user queries with expert-authored, multi-step answers; (ii) WixQA-Simulated, 200 expert-validated QA pairs distilled from user dialogues; and (iii) WixQA-Synthetic, 6,222 LLM-generated QA pairs, with one pair systematically derived from each article in the knowledge base. We release the KB snapshot alongside the datasets under MIT license and provide comprehensive baseline results, forming a unique benchmark for evaluating enterprise RAG systems in realistic enterprise environments. 

**Abstract (ZH)**: 基于检索的生成（RAG）是现代问答（QA）系统的核心，使基于外部知识的 grounded 答案成为可能。尽管 Recent 进展主要得益于开放领域数据集，企业 QA 系统需要反映日常支持场景中用户提出的具体、领域特定问题的数据集。关键在于，评估端到端的 RAG 系统需要不仅包含问题-答案对，还包含生成这些答案的具体知识库（KB）快照的基准测试。为满足这一需求，我们引入了 WixQA，这是一个基准套件，包含精确基于发布知识库语料库的 QA 数据集，以实现对检索和生成组件的全面评估。WixQA 包括三个源自特定网址的客户支持互动并基于 Wix 帮助中心公开知识库快照的独立 QA 数据集：(i) WixQA-ExpertWritten，包含 200 个真实用户查询和专家撰写的多步答案；(ii) WixQA-Simulated，包含 200 组专家验证的问题-答案对，这些对是从用户对话中提炼出来的；(iii) WixQA-Synthetic，包含 6,222 对由大语言模型生成的问题-答案对，每一对均系统地来源于知识库中的每一篇文章。我们以 MIT 许可证发布知识库快照及其数据集，并提供全面的基础结果，形成一个独特的基准，用于在实际的企业环境中评估企业 RAG 系统。 

---
# Integrating Natural Language Processing and Exercise Monitoring for Early Diagnosis of Metabolic Syndrome: A Deep Learning Approach 

**Title (ZH)**: 将自然语言处理与运动监测 integrates 早期诊断代谢综合征：一种深度学习方法 

**Authors**: Yichen Zhao, Yuhua Wang, Xi Cheng, Junhao Fang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08628)  

**Abstract**: Metabolic syndrome (MetS) is a medication condition characterized by abdominal obesity, insulin resistance, hypertension and hyperlipidemia. It increases the risk of majority of chronic diseases, including type 2 diabetes mellitus, and affects about one quarter of the global population. Therefore, early detection and timely intervention for MetS are crucial. Standard diagnosis for MetS components requires blood tests conducted within medical institutions. However, it is frequently underestimated, leading to unmet need for care for MetS population. This study aims to use the least physiological data and free texts about exercises related activities, which are obtained easily in daily life, to diagnosis MetS. We collected the data from 40 volunteers in a nursing home and used data augmentation to reduce the imbalance. We propose a deep learning framework for classifying MetS that integrates natural language processing (NLP) and exercise monitoring. The results showed that the best model reported a high positive result (AUROC=0.806 and REC=76.3%) through 3-fold cross-validation. Feature importance analysis revealed that text and minimum heart rate on a daily basis contribute the most in the classification of MetS. This study demonstrates the potential application of data that are easily measurable in daily life for the early diagnosis of MetS, which could contribute to reducing the cost of screening and management for MetS population. 

**Abstract (ZH)**: 代谢综合征（MetS）是一种以腹部肥胖、胰岛素抵抗、高血压和高脂血症为特征的代谢状况。它增加了2型糖尿病等多数慢性疾病的风险，并影响全球约四分之一的人口。因此，早期检测和及时干预代谢综合征至关重要。目前，代谢综合征组件的标准诊断需要在医疗机构进行血液检测。然而，它经常被低估，导致代谢综合征人群的护理需求未得到满足。本研究旨在利用日常生活易于获得的最少生理数据和与运动相关的自由文本，来诊断代谢综合征。我们从一所护理院的40名志愿者中收集数据，并使用数据增强以减少数据不平衡。我们提出了一种结合自然语言处理（NLP）和运动监测的深度学习框架，用于代谢综合征分类。结果表明，通过3折交叉验证，最佳模型的AUC值为0.806，召回率为76.3%。特征重要性分析显示，文本内容和每日最低心率对代谢综合征分类最具贡献。本研究展示了日常生活中易于测量的数据在代谢综合征早期诊断中的潜在应用，这有助于降低代谢综合征人群筛查和管理的成本。 

---
# On the Complexity and Properties of Preferential Propositional Dependence Logic 

**Title (ZH)**: 偏好命题依赖逻辑的复杂性和性质 

**Authors**: Kai Sauerwald, Arne Meier, Juha Kontinen  

**Link**: [PDF](https://arxiv.org/pdf/2505.08522)  

**Abstract**: This paper considers the complexity and properties of KLM-style preferential reasoning in the setting of propositional logic with team semantics and dependence atoms, also known as propositional dependence logic. Preferential team-based reasoning is shown to be cumulative, yet violates System~P. We give intuitive conditions that fully characterise those cases where preferential propositional dependence logic satisfies System~P. We show that these characterisations do, surprisingly, not carry over to preferential team-based propositional logic. Furthermore, we show how classical entailment and dependence logic entailment can be expressed in terms of non-trivial preferential models. Finally, we present the complexity of preferential team-based reasoning for two natural representations. This includes novel complexity results for classical (non-team-based) preferential reasoning. 

**Abstract (ZH)**: 本文考虑了在命题逻辑与团队语义及依赖原子的设定下，KLM风格的优先推理的复杂性和性质。优先团队为基础的推理被证明是累积的，但违反了System P。我们给出了直观的条件，完全描述了优先命题依赖逻辑满足System P的那些情况。我们展示了这些描述出人意料地不适用于优先团队为基础的命题逻辑。此外，我们展示了经典蕴含和依赖逻辑蕴含如何用非平凡的优先模型来表达。最后，我们提出了优先团队为基础推理的复杂性，包括经典（非团队为基础）优先推理的新颖复杂性结果。 

---
# TrialMatchAI: An End-to-End AI-powered Clinical Trial Recommendation System to Streamline Patient-to-Trial Matching 

**Title (ZH)**: TrialMatchAI：一个端到端的AI驱动临床试验推荐系统，以简化患者与试验匹配过程 

**Authors**: Majd Abdallah, Sigve Nakken, Mariska Bierkens, Johanna Galvis, Alexis Groppi, Slim Karkar, Lana Meiqari, Maria Alexandra Rujano, Steve Canham, Rodrigo Dienstmann, Remond Fijneman, Eivind Hovig, Gerrit Meijer, Macha Nikolski  

**Link**: [PDF](https://arxiv.org/pdf/2505.08508)  

**Abstract**: Patient recruitment remains a major bottleneck in clinical trials, calling for scalable and automated solutions. We present TrialMatchAI, an AI-powered recommendation system that automates patient-to-trial matching by processing heterogeneous clinical data, including structured records and unstructured physician notes. Built on fine-tuned, open-source large language models (LLMs) within a retrieval-augmented generation framework, TrialMatchAI ensures transparency and reproducibility and maintains a lightweight deployment footprint suitable for clinical environments. The system normalizes biomedical entities, retrieves relevant trials using a hybrid search strategy combining lexical and semantic similarity, re-ranks results, and performs criterion-level eligibility assessments using medical Chain-of-Thought reasoning. This pipeline delivers explainable outputs with traceable decision rationales. In real-world validation, 92 percent of oncology patients had at least one relevant trial retrieved within the top 20 recommendations. Evaluation across synthetic and real clinical datasets confirmed state-of-the-art performance, with expert assessment validating over 90 percent accuracy in criterion-level eligibility classification, particularly excelling in biomarker-driven matches. Designed for modularity and privacy, TrialMatchAI supports Phenopackets-standardized data, enables secure local deployment, and allows seamless replacement of LLM components as more advanced models emerge. By enhancing efficiency and interpretability and offering lightweight, open-source deployment, TrialMatchAI provides a scalable solution for AI-driven clinical trial matching in precision medicine. 

**Abstract (ZH)**: 基于AI的TrialMatchAI推荐系统：自动化患者与临床试验匹配的解决方案 

---
# BAT: Benchmark for Auto-bidding Task 

**Title (ZH)**: BAT：自动竞价任务基准 

**Authors**: Alexandra Khirianova, Ekaterina Solodneva, Andrey Pudovikov, Sergey Osokin, Egor Samosvat, Yuriy Dorn, Alexander Ledovsky, Yana Zenkova  

**Link**: [PDF](https://arxiv.org/pdf/2505.08485)  

**Abstract**: The optimization of bidding strategies for online advertising slot auctions presents a critical challenge across numerous digital marketplaces. A significant obstacle to the development, evaluation, and refinement of real-time autobidding algorithms is the scarcity of comprehensive datasets and standardized benchmarks.
To address this deficiency, we present an auction benchmark encompassing the two most prevalent auction formats. We implement a series of robust baselines on a novel dataset, addressing the most salient Real-Time Bidding (RTB) problem domains: budget pacing uniformity and Cost Per Click (CPC) constraint optimization. This benchmark provides a user-friendly and intuitive framework for researchers and practitioners to develop and refine innovative autobidding algorithms, thereby facilitating advancements in the field of programmatic advertising. The implementation and additional resources can be accessed at the following repository (this https URL, this https URL). 

**Abstract (ZH)**: 在线广告拍卖竞价策略的优化是众多数字市场面临的关键挑战。由于全面数据集和标准化基准的稀缺性，实时自动化竞价算法的发展、评估和优化面临重要障碍。

为应对这一不足，我们提供了一个涵盖两种最常见的拍卖格式的拍卖基准。我们在新型数据集上实现了一系列稳健的基线，解决实时竞价（RTB）领域的最突出问题领域：预算 pacing 统一性和每点击成本（CPC）约束优化。该基准为研究人员和实践者提供了一个用户友好且直观的框架，以开发和优化创新的自动化竞价算法，从而推动程序化广告领域的发展。相关实现和额外资源可在以下仓库访问（this https URL, this https URL）。 

---
# Adaptive Bias Generalized Rollout Policy Adaptation on the Flexible Job-Shop Scheduling Problem 

**Title (ZH)**: 柔性作业车间调度问题的自适应偏差广义展开策略适应性研究 

**Authors**: Lotfi Kobrosly, Marc-Emmanuel Coupvent des Graviers, Christophe Guettier, Tristan Cazenave  

**Link**: [PDF](https://arxiv.org/pdf/2505.08451)  

**Abstract**: The Flexible Job-Shop Scheduling Problem (FJSSP) is an NP-hard combinatorial optimization problem, with several application domains, especially for manufacturing purposes. The objective is to
efficiently schedule multiple operations on dissimilar machines. These operations are gathered into jobs, and operations pertaining to the same job need to be scheduled sequentially. Different methods have been previously tested to solve this problem, such as Constraint Solving, Tabu Search, Genetic Algorithms, or Monte Carlo Tree Search (MCTS). We propose a novel algorithm derived from the Generalized Nested Rollout Policy Adaptation, developed to solve the FJSSP. We report encouraging experimental results, as our algorithm performs better than other MCTS-based approaches, even if makespans obtained on large instances are still far from known upper bounds. 

**Abstract (ZH)**: 柔性作业-shop调度问题(FJSSP)是一个NP难的组合优化问题，尤其在制造领域中有广泛应用。目标是高效地安排多种操作在不同类型的机器上。这些操作被组织成作业，同一作业的操作需要顺序执行。已有多方法尝试解决此问题，如约束求解、 Tabu搜索、遗传算法或蒙特卡洛树搜索(MCTS)。我们提出了一种基于广义嵌套展开策略调整的新算法，用于解决FJSSP问题。实验结果令人鼓舞，我们的算法在某些方面优于其他基于MCTS的方法，尽管在大规模实例上的制造周期仍远高于已知的上界。 

---
# An Identifiable Cost-Aware Causal Decision-Making Framework Using Counterfactual Reasoning 

**Title (ZH)**: 基于反事实推理的可识别成本意识因果决策框架 

**Authors**: Ruichu Cai, Xi Chen, Jie Qiao, Zijian Li, Yuequn Liu, Wei Chen, Keli Zhang, Jiale Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.08343)  

**Abstract**: Decision making under abnormal conditions is a critical process that involves evaluating the current state and determining the optimal action to restore the system to a normal state at an acceptable cost. However, in such scenarios, existing decision-making frameworks highly rely on reinforcement learning or root cause analysis, resulting in them frequently neglecting the cost of the actions or failing to incorporate causal mechanisms adequately. By relaxing the existing causal decision framework to solve the necessary cause, we propose a minimum-cost causal decision (MiCCD) framework via counterfactual reasoning to address the above challenges. Emphasis is placed on making counterfactual reasoning processes identifiable in the presence of a large amount of mixed anomaly data, as well as finding the optimal intervention state in a continuous decision space. Specifically, it formulates a surrogate model based on causal graphs, using abnormal pattern clustering labels as supervisory signals. This enables the approximation of the structural causal model among the variables and lays a foundation for identifiable counterfactual reasoning. With the causal structure approximated, we then established an optimization model based on counterfactual estimation. The Sequential Least Squares Programming (SLSQP) algorithm is further employed to optimize intervention strategies while taking costs into account. Experimental evaluations on both synthetic and real-world datasets reveal that MiCCD outperforms conventional methods across multiple metrics, including F1-score, cost efficiency, and ranking quality(nDCG@k values), thus validating its efficacy and broad applicability. 

**Abstract (ZH)**: 基于反事实推理的最小成本因果决策框架（MiCCD） 

---
# Benchmarking AI scientists in omics data-driven biological research 

**Title (ZH)**: 基于奥米克戎数据驱动生物研究的AI科学家基准测试 

**Authors**: Erpai Luo, Jinmeng Jia, Yifan Xiong, Xiangyu Li, Xiaobo Guo, Baoqi Yu, Lei Wei, Xuegong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08341)  

**Abstract**: The rise of large language models and multi-agent systems has sparked growing interest in AI scientists capable of autonomous biological research. However, existing benchmarks either focus on reasoning without data or on data analysis with predefined statistical answers, lacking realistic, data-driven evaluation settings. Here, we introduce the Biological AI Scientist Benchmark (BaisBench), a benchmark designed to assess AI scientists' ability to generate biological discoveries through data analysis and reasoning with external knowledge. BaisBench comprises two tasks: cell type annotation on 31 expert-labeled single-cell datasets, and scientific discovery through answering 198 multiple-choice questions derived from the biological insights of 41 recent single-cell studies. Systematic experiments on state-of-the-art AI scientists and LLM agents showed that while promising, current models still substantially underperform human experts on both tasks. We hope BaisBench will fill this gap and serve as a foundation for advancing and evaluating AI models for scientific discovery. The benchmark can be found at: this https URL. 

**Abstract (ZH)**: 生物AI科学家基准（BaisBench）：评估AI科学家通过数据分析和外部知识进行生物发现的能力 

---
# Unveiling the Best Practices for Applying Speech Foundation Models to Speech Intelligibility Prediction for Hearing-Impaired People 

**Title (ZH)**: 揭示将语音基础模型应用于听力受损人群语音可懂度预测的最佳实践 

**Authors**: Haoshuai Zhou, Boxuan Cao, Changgeng Mo, Linkai Li, Shan Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08215)  

**Abstract**: Speech foundation models (SFMs) have demonstrated strong performance across a variety of downstream tasks, including speech intelligibility prediction for hearing-impaired people (SIP-HI). However, optimizing SFMs for SIP-HI has been insufficiently explored. In this paper, we conduct a comprehensive study to identify key design factors affecting SIP-HI performance with 5 SFMs, focusing on encoder layer selection, prediction head architecture, and ensemble configurations. Our findings show that, contrary to traditional use-all-layers methods, selecting a single encoder layer yields better results. Additionally, temporal modeling is crucial for effective prediction heads. We also demonstrate that ensembling multiple SFMs improves performance, with stronger individual models providing greater benefit. Finally, we explore the relationship between key SFM attributes and their impact on SIP-HI performance. Our study offers practical insights into effectively adapting SFMs for speech intelligibility prediction for hearing-impaired populations. 

**Abstract (ZH)**: 基于语音的模型（SFMs）已经在多种下游任务中展现了强大的性能，包括听障人士的语音可懂度预测（SIP-HI）。然而，针对SIP-HI优化SFM的研究尚不够充分。在本文中，我们进行了全面研究，使用5种SFM来识别影响SIP-HI性能的关键设计因素，重点关注编码器层选择、预测头架构和集成配置。我们的研究结果表明，与传统的使用所有层的方法相比，选择单一编码器层能获得更好的效果。此外，时间建模对于有效的预测头至关重要。我们还展示了集成多个SFM可以提高性能，更强的单个模型能提供更大的收益。最后，我们探索了关键SFM属性与SIP-HI性能之间的关系。我们的研究提供了关于有效适应SFM进行听障人士语音可懂度预测的实用见解。 

---
# Behind the Noise: Conformal Quantile Regression Reveals Emergent Representations 

**Title (ZH)**: 超越噪声：共识量ile回归揭示 Emergent 表示 

**Authors**: Petrus H. Zwart, Tamas Varga, Odeta Qafoku, James A. Sethian  

**Link**: [PDF](https://arxiv.org/pdf/2505.08176)  

**Abstract**: Scientific imaging often involves long acquisition times to obtain high-quality data, especially when probing complex, heterogeneous systems. However, reducing acquisition time to increase throughput inevitably introduces significant noise into the measurements. We present a machine learning approach that not only denoises low-quality measurements with calibrated uncertainty bounds, but also reveals emergent structure in the latent space. By using ensembles of lightweight, randomly structured neural networks trained via conformal quantile regression, our method performs reliable denoising while uncovering interpretable spatial and chemical features -- without requiring labels or segmentation. Unlike conventional approaches focused solely on image restoration, our framework leverages the denoising process itself to drive the emergence of meaningful representations. We validate the approach on real-world geobiochemical imaging data, showing how it supports confident interpretation and guides experimental design under resource constraints. 

**Abstract (ZH)**: Scientific成像 Often Involves 长时间的数据获取 以获得高质量的数据，特别是在探针复杂、异质系统时。然而，为了提高吞吐量而减少数据获取时间不可避免地会引入显著的噪声。我们提出了一种机器学习方法，不仅能够通过校准的不确定性界限对低质量测量进行去噪，还能揭示潜在空间中的新兴结构。通过使用通过符合分位数回归训练的轻量级、随机结构神经网络的ensemble，我们的方法在不需标签或分割的情况下实现了可靠的去噪，并揭示了可解释的空间和化学特征。与专注于图像复原的传统方法不同，我们的框架利用去噪过程本身来驱动有意义表示的涌现。我们在真实的地球生物化学成像数据上验证了该方法，展示了它如何在资源受限条件下支持自信的解释并指导实验设计。 

---
# Efficient and Scalable Neural Symbolic Search for Knowledge Graph Complex Query Answering 

**Title (ZH)**: 高效可扩展的神经符号搜索方法及其在知识图复杂查询回答中的应用 

**Authors**: Weizhi Fei, Zihao Wang, hang Yin, Shukai Zhao, Wei Zhang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.08155)  

**Abstract**: Complex Query Answering (CQA) aims to retrieve answer sets for complex logical formulas from incomplete knowledge graphs, which is a crucial yet challenging task in knowledge graph reasoning. While neuro-symbolic search utilized neural link predictions achieve superior accuracy, they encounter significant complexity bottlenecks: (i) Data complexity typically scales quadratically with the number of entities in the knowledge graph, and (ii) Query complexity becomes NP-hard for cyclic queries. Consequently, these approaches struggle to effectively scale to larger knowledge graphs and more complex queries. To address these challenges, we propose an efficient and scalable symbolic search framework. First, we propose two constraint strategies to compute neural logical indices to reduce the domain of variables, thereby decreasing the data complexity of symbolic search. Additionally, we introduce an approximate algorithm based on local search to tackle the NP query complexity of cyclic queries. Experiments on various CQA benchmarks demonstrate that our framework reduces the computational load of symbolic methods by 90\% while maintaining nearly the same performance, thus alleviating both efficiency and scalability issues. 

**Abstract (ZH)**: 复杂查询回答（CQA）旨在从不完整知识图中检索复杂的逻辑公式答案集合，这是知识图 reasoning 中一个关键但具有挑战性的任务。虽然神经符号搜索利用神经连接预测实现了更高的准确性，但它们遇到了显著的复杂性瓶颈：（i）数据复杂性通常与知识图中的实体数量成二次关系，（ii）循环查询的查询复杂性成为 NP 难问题。因此，这些方法难以有效扩展到更大的知识图和更复杂的查询。为了解决这些挑战，我们提出了一种高效且可扩展的符号搜索框架。首先，我们提出了两种约束策略来计算神经逻辑索引，以减少变量的取值范围，从而降低符号搜索的数据复杂性。此外，我们引入了一种基于局部搜索的近似算法来处理循环查询的 NP 查询复杂性。在各种 CQA 标准测试集上的实验表明，我们的框架将符号方法的计算负载降低了 90%，同时保持了几乎相同的表现，从而缓解了效率和 scalability 问题。 

---
# Bias or Optimality? Disentangling Bayesian Inference and Learning Biases in Human Decision-Making 

**Title (ZH)**: 偏见还是最优性？解构人类决策中的贝叶斯推理与学习偏见 

**Authors**: Prakhar Godara  

**Link**: [PDF](https://arxiv.org/pdf/2505.08049)  

**Abstract**: Recent studies claim that human behavior in a two-armed Bernoulli bandit (TABB) task is described by positivity and confirmation biases, implying that humans do not integrate new information objectively. However, we find that even if the agent updates its belief via objective Bayesian inference, fitting the standard Q-learning model with asymmetric learning rates still recovers both biases. Bayesian inference cast as an effective Q-learning algorithm has symmetric, though decreasing, learning rates. We explain this by analyzing the stochastic dynamics of these learning systems using master equations. We find that both confirmation bias and unbiased but decreasing learning rates yield the same behavioral signatures. Finally, we propose experimental protocols to disentangle true cognitive biases from artifacts of decreasing learning rates. 

**Abstract (ZH)**: 近期的研究认为，在两臂伯努利老虎机任务中，人类行为由积极偏差和确认偏差描述，暗示人类不能客观整合新信息。然而，我们发现即使智能体通过客观贝叶斯推断更新其信念，使用非对称学习率拟合标准Q学习模型仍然能够恢复这两种偏差。将贝叶斯推理视为有效的Q学习算法具有对称但递减的学习率。我们通过分析这些学习系统的随机动力学来解释这一现象，并发现确认偏差和无偏但递减的学习率会导致相同的行为特征。最后，我们提出实验方案以区分真实的认知偏差和学习率递减导致的伪像。 

---
# The Correspondence Between Bounded Graph Neural Networks and Fragments of First-Order Logic 

**Title (ZH)**: 有界图神经网络与一阶逻辑片段之间的对应关系 

**Authors**: Bernardo Cuenca Grau, Przemysław A. Wałęga  

**Link**: [PDF](https://arxiv.org/pdf/2505.08021)  

**Abstract**: Graph Neural Networks (GNNs) address two key challenges in applying deep learning to graph-structured data: they handle varying size input graphs and ensure invariance under graph isomorphism. While GNNs have demonstrated broad applicability, understanding their expressive power remains an important question. In this paper, we show that bounded GNN architectures correspond to specific fragments of first-order logic (FO), including modal logic (ML), graded modal logic (GML), modal logic with the universal modality (ML(A)), the two-variable fragment (FO2) and its extension with counting quantifiers (C2). To establish these results, we apply methods and tools from finite model theory of first-order and modal logics to the domain of graph representation learning. This provides a unifying framework for understanding the logical expressiveness of GNNs within FO. 

**Abstract (ZH)**: 图神经网络（GNNs）解决了将深度学习应用于图结构数据时的两个关键挑战：处理不同大小的输入图以及确保同构不变性。虽然GNNs展示了广泛的应用性，但对其表征能力的理解仍然是一个重要问题。在本文中，我们证明了受限的GNN架构对应于一阶逻辑（FO）的特定片段，包括模态逻辑（ML）、分级模态逻辑（GML）、带有全称模态性的模态逻辑（ML(A)）、二变量片段（FO2）及其扩展的计数量词片段（C2）。为了得出这些结果，我们运用了一阶逻辑和模态逻辑的有限模型理论方法和工具，将其应用于图表示学习领域。这提供了一种统一框架，用于在FO内理解GNNs的逻辑表征能力。 

---
# Enhancing Trust Management System for Connected Autonomous Vehicles Using Machine Learning Methods: A Survey 

**Title (ZH)**: 使用机器学习方法增强连接自动驾驶车辆的信任管理系统的综述 

**Authors**: Qian Xu, Lei Zhang, Yixiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07882)  

**Abstract**: Connected Autonomous Vehicles (CAVs) operate in dynamic, open, and multi-domain networks, rendering them vulnerable to various threats. Trust Management Systems (TMS) systematically organize essential steps in the trust mechanism, identifying malicious nodes against internal threats and external threats, as well as ensuring reliable decision-making for more cooperative tasks. Recent advances in machine learning (ML) offer significant potential to enhance TMS, especially for the strict requirements of CAVs, such as CAV nodes moving at varying speeds, and opportunistic and intermittent network behavior. Those features distinguish ML-based TMS from social networks, static IoT, and Social IoT. This survey proposes a novel three-layer ML-based TMS framework for CAVs in the vehicle-road-cloud integration system, i.e., trust data layer, trust calculation layer and trust incentive layer. A six-dimensional taxonomy of objectives is proposed. Furthermore, the principles of ML methods for each module in each layer are analyzed. Then, recent studies are categorized based on traffic scenarios that are against the proposed objectives. Finally, future directions are suggested, addressing the open issues and meeting the research trend. We maintain an active repository that contains up-to-date literature and open-source projects at this https URL. 

**Abstract (ZH)**: 连接自主车辆（CAVs）在动态、开放且多领域的网络中运行，使其容易受到各种威胁。信任管理系统（TMS）系统地组织信任机制中的关键步骤，识别内部和外部威胁中的恶意节点，并确保更合作任务中的可靠决策。机器学习（ML）的 recent 进展为增强 TMS 提供了显著潜力，特别是对于连接自主车辆（CAVs）的严格要求，如 CAV 节点以不同速度移动，以及机会性和间歇性的网络行为。这些特征使基于 ML 的 TMS 与社会网络、静态物联网（IoT）和社会物联网（Social IoT）区分开来。本文综述提出了一种针对车辆-道路-云集成系统的连接自主车辆（CAVs）的新型三层机器学习（ML）信任管理系统框架，即信任数据层、信任计算层和信任激励层。提出了六维目标分类法。此外，分析了每一层中每个模块的机器学习（ML）方法原理。然后，根据反对所提目标的交通场景对近期研究进行了分类。最后，提出了未来方向，以解决开放问题并符合研究趋势。我们保持一个活跃的仓库，其中包含最新的文献和开源项目，详情请访问此链接。 

---
# CCL: Collaborative Curriculum Learning for Sparse-Reward Multi-Agent Reinforcement Learning via Co-evolutionary Task Evolution 

**Title (ZH)**: 协作性课程学习：通过共演化的任务进化在稀疏奖励多智能体强化学习中的应用 

**Authors**: Yufei Lin, Chengwei Ye, Huanzhen Zhang, Kangsheng Wang, Linuo Xu, Shuyan Liu, Zeyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07854)  

**Abstract**: Sparse reward environments pose significant challenges in reinforcement learning, especially within multi-agent systems (MAS) where feedback is delayed and shared across agents, leading to suboptimal learning. We propose Collaborative Multi-dimensional Course Learning (CCL), a novel curriculum learning framework that addresses this by (1) refining intermediate tasks for individual agents, (2) using a variational evolutionary algorithm to generate informative subtasks, and (3) co-evolving agents with their environment to enhance training stability. Experiments on five cooperative tasks in the MPE and Hide-and-Seek environments show that CCL outperforms existing methods in sparse reward settings. 

**Abstract (ZH)**: 稀疏奖励环境对强化学习提出了重大挑战，特别是在多智能体系统中，反馈延迟且共享于多个智能体之间，导致学习效果不佳。为此，我们提出了一种新型的协作多维课程学习框架CCL，通过（1）细化个体智能体的中间任务，（2）使用变分进化算法生成有信息量的子任务，以及（3）智能体与环境共同进化以提高训练稳定性，来解决上述问题。在MPE和Hide-and-Seek环境中的五个合作任务上的实验表明，CCL在稀疏奖励设置中优于现有方法。 

---
# Conceptual Logical Foundations of Artificial Social Intelligence 

**Title (ZH)**: 人工社会智能的conceptual逻辑基础 

**Authors**: Eric Werner  

**Link**: [PDF](https://arxiv.org/pdf/2505.07847)  

**Abstract**: What makes a society possible at all? How is coordination and cooperation in social activity possible? What is the minimal mental architecture of a social agent? How is the information about the state of the world related to the agents intentions? How are the intentions of agents related? What role does communication play in this coordination process? This essay explores the conceptual and logical foundations of artificial social intelligence in the context of a society of multiple agents that communicate and cooperate to achieve some end. An attempt is made to provide an introduction to some of the key concepts, their formal definitions and their interrelationships. These include the notion of a changing social world of multiple agents. The logic of social intelligence goes beyond classical logic by linking information with strategic thought. A minimal architecture of social agents is presented. The agents have different dynamically changing, possible choices and abilities. The agents also have uncertainty, lacking perfect information about their physical state as well as their dynamic social state. The social state of an agent includes the intentional state of that agent, as well as, that agent's representation of the intentional states of other agents. Furthermore, it includes the evaluations agents make of their physical and social condition. Communication, semantic and pragmatic meaning and their relationship to intention and information states are investigated. The logic of agent abilities and intentions are motivated and formalized. The entropy of group strategic states is defined. 

**Abstract (ZH)**: 多代理社会中协调与合作的概念与逻辑基础 

---
# RAN Cortex: Memory-Augmented Intelligence for Context-Aware Decision-Making in AI-Native Networks 

**Title (ZH)**: RAN cortex: 基于内存增强的面向上下文决策的AI原生网络智能 

**Authors**: Sebastian Barros  

**Link**: [PDF](https://arxiv.org/pdf/2505.07842)  

**Abstract**: As Radio Access Networks (RAN) evolve toward AI-native architectures, intelligent modules such as xApps and rApps are expected to make increasingly autonomous decisions across scheduling, mobility, and resource management domains. However, these agents remain fundamentally stateless, treating each decision as isolated, lacking any persistent memory of prior events or outcomes. This reactive behavior constrains optimization, especially in environments where network dynamics exhibit episodic or recurring patterns. In this work, we propose RAN Cortex, a memory-augmented architecture that enables contextual recall in AI-based RAN decision systems. RAN Cortex introduces a modular layer composed of four elements: a context encoder that transforms network state into high-dimensional embeddings, a vector-based memory store of past network episodes, a recall engine to retrieve semantically similar situations, and a policy interface that supplies historical context to AI agents in real time or near-real time. We formalize the retrieval-augmented decision problem in the RAN, present a system architecture compatible with O-RAN interfaces, and analyze feasible deployments within the Non-RT and Near-RT RIC domains. Through illustrative use cases such as stadium traffic mitigation and mobility management in drone corridors, we demonstrate how contextual memory improves adaptability, continuity, and overall RAN intelligence. This work introduces memory as a missing primitive in AI-native RAN designs and provides a framework to enable "learning agents" without the need for retraining or centralized inference 

**Abstract (ZH)**: 基于AI的无线接入网络认知内存增强架构：RAN Cortex 

---
# An Optimized Evacuation Plan for an Active-Shooter Situation Constrained by Network Capacity 

**Title (ZH)**: 受网络容量约束的主动射手事件优化疏散计划 

**Authors**: Joseph Lavalle-Rivera, Aniirudh Ramesh, Subhadeep Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2505.07830)  

**Abstract**: A total of more than 3400 public shootings have occurred in the United States between 2016 and 2022. Among these, 25.1% of them took place in an educational institution, 29.4% at the workplace including office buildings, 19.6% in retail store locations, and 13.4% in restaurants and bars. During these critical scenarios, making the right decisions while evacuating can make the difference between life and death. However, emergency evacuation is intensely stressful, which along with the lack of verifiable real-time information may lead to fatal incorrect decisions. To tackle this problem, we developed a multi-route routing optimization algorithm that determines multiple optimal safe routes for each evacuee while accounting for available capacity along the route, thus reducing the threat of crowding and bottlenecking. Overall, our algorithm reduces the total casualties by 34.16% and 53.3%, compared to our previous routing algorithm without capacity constraints and an expert-advised routing strategy respectively. Further, our approach to reduce crowding resulted in an approximate 50% reduction in occupancy in key bottlenecking nodes compared to both of the other evacuation algorithms. 

**Abstract (ZH)**: 2016年至2022年间，美国共发生了超过3400起公共枪击事件。其中，25.1%发生在教育机构，29.4%发生在工作场所包括办公楼，19.6%发生在零售店，13.4%发生在餐馆和酒吧。在这些关键时刻，正确疏散决策可以决定生与死。然而，紧急疏散极其紧张，缺乏可靠的实时信息可能导致致命的错误决策。为解决这一问题，我们开发了一种多路径路由优化算法，为每个疏散人员确定多个最优安全路径，同时考虑路径上的可用容量，从而减少拥堵和瓶颈的风险。总体而言，与不考虑容量约束的先前路由算法相比，我们的算法减少了34.16%的伤亡，与专家建议的疏散策略相比减少了53.3%的伤亡。此外，我们减少拥堵的方法使得关键瓶颈节点的占用率比其他两种疏散算法降低了约50%。 

---
# Big Data and the Computational Social Science of Entrepreneurship and Innovation 

**Title (ZH)**: 大数据与创业创新的计算社会科学 

**Authors**: Ningzi Li, Shiyang Lai, James Evans  

**Link**: [PDF](https://arxiv.org/pdf/2505.08706)  

**Abstract**: As large-scale social data explode and machine-learning methods evolve, scholars of entrepreneurship and innovation face new research opportunities but also unique challenges. This chapter discusses the difficulties of leveraging large-scale data to identify technological and commercial novelty, document new venture origins, and forecast competition between new technologies and commercial forms. It suggests how scholars can take advantage of new text, network, image, audio, and video data in two distinct ways that advance innovation and entrepreneurship research. First, machine-learning models, combined with large-scale data, enable the construction of precision measurements that function as system-level observatories of innovation and entrepreneurship across human societies. Second, new artificial intelligence models fueled by big data generate 'digital doubles' of technology and business, forming laboratories for virtual experimentation about innovation and entrepreneurship processes and policies. The chapter argues for the advancement of theory development and testing in entrepreneurship and innovation by coupling big data with big models. 

**Abstract (ZH)**: 随着大规模社会数据的爆炸式增长和机器学习方法的发展，创业与创新领域的学者面临新的研究机遇同时也面临独特挑战。本章探讨了如何利用大规模数据识别技术与商业新颖性、记录新企业的起源以及预测新技术与商业形式之间的竞争。本章提出学者们可以通过两种方式利用新的文本、网络、图像、音频和视频数据，推动创新与创业研究的进步。首先，将机器学习模型与大规模数据结合，构建作为全球创新与创业系统的观测站的精密测量工具。其次，由大数据驱动的新人工智能模型生成技术与商业的“数字双胞胎”，形成关于创新与创业过程和政策的虚拟实验实验室。本章主张通过将大数据与大模型相结合，推动创业与创新领域的理论发展与测试。 

---
# A Survey of Deep Learning for Complex Speech Spectrograms 

**Title (ZH)**: 深度学习在复杂语音频谱图中的综述 

**Authors**: Yuying Xie, Zheng-Hua Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.08694)  

**Abstract**: Recent advancements in deep learning have significantly impacted the field of speech signal processing, particularly in the analysis and manipulation of complex spectrograms. This survey provides a comprehensive overview of the state-of-the-art techniques leveraging deep neural networks for processing complex spectrograms, which encapsulate both magnitude and phase information. We begin by introducing complex spectrograms and their associated features for various speech processing tasks. Next, we explore the key components and architectures of complex-valued neural networks, which are specifically designed to handle complex-valued data and have been applied for complex spectrogram processing. We then discuss various training strategies and loss functions tailored for training neural networks to process and model complex spectrograms. The survey further examines key applications, including phase retrieval, speech enhancement, and speech separation, where deep learning has achieved significant progress by leveraging complex spectrograms or their derived feature representations. Additionally, we examine the intersection of complex spectrograms with generative models. This survey aims to serve as a valuable resource for researchers and practitioners in the field of speech signal processing and complex-valued neural networks. 

**Abstract (ZH)**: 近期深度学习的进展对语音信号处理领域产生了显著影响，尤其是在复杂频谱图的分析和操控方面。本文综述了利用深度神经网络处理复杂频谱图的先进技术和方法，这些复杂频谱图包含了幅值和相位信息。我们首先介绍复杂频谱图及其在各种语音处理任务中的相关特征。接着，我们探讨了处理复值数据的复值神经网络的关键组件和架构。然后，我们讨论了适合训练神经网络处理和建模复杂频谱图的各种训练策略和损失函数。本文综述进一步探讨了将深度学习与复值频谱图应用于相位恢复、语音增强和语音分离等关键应用领域取得的重要进展。此外，我们还考察了复值频谱图与生成模型的交叉融合。本文旨在为语音信号处理和复值神经网络领域的研究人员和实务工作者提供有价值的资源。 

---
# VizCV: AI-assisted visualization of researchers' publications tracks 

**Title (ZH)**: VizCV：研究人员发表记录的AI辅助可视化 

**Authors**: Vladimír Lazárik, Marco Agus, Barbora Kozlíková, Pere-Pau Vázquez  

**Link**: [PDF](https://arxiv.org/pdf/2505.08691)  

**Abstract**: Analyzing how the publication records of scientists and research groups have evolved over the years is crucial for assessing their expertise since it can support the management of academic environments by assisting with career planning and evaluation. We introduce VizCV, a novel web-based end-to-end visual analytics framework that enables the interactive exploration of researchers' scientific trajectories. It incorporates AI-assisted analysis and supports automated reporting of career evolution. Our system aims to model career progression through three key dimensions: a) research topic evolution to detect and visualize shifts in scholarly focus over time, b) publication record and the corresponding impact, c) collaboration dynamics depicting the growth and transformation of a researcher's co-authorship network. AI-driven insights provide automated explanations of career transitions, detecting significant shifts in research direction, impact surges, or collaboration expansions. The system also supports comparative analysis between researchers, allowing users to compare topic trajectories and impact growth. Our interactive, multi-tab and multiview system allows for the exploratory analysis of career milestones under different perspectives, such as the most impactful articles, emerging research themes, or obtaining a detailed analysis of the contribution of the researcher in a subfield. The key contributions include AI/ML techniques for: a) topic analysis, b) dimensionality reduction for visualizing patterns and trends, c) the interactive creation of textual descriptions of facets of data through configurable prompt generation and large language models, that include key indicators, to help understanding the career development of individuals or groups. 

**Abstract (ZH)**: 分析科学家和研究团队的出版记录随时间的变化对于评估其专业水平至关重要，这有助于学术环境的管理，支持职业规划和评估。我们介绍了VizCV，一种新颖的基于Web的端到端可视化分析框架，使用户能够互动地探索研究人员的科学轨迹。该系统结合了AI辅助分析，并支持自动化的职业发展报告。我们的系统旨在通过三个关键维度建模职业发展：a) 研究主题演化，以检测和可视化随着时间的推移学术重点的变化，b) 发表记录及其相应的影响力，c) 合作动态，描绘研究者合著网络的成长与转变。基于AI的见解提供自动化的职业转变解释，检测研究方向的重大转变、影响力激增或协作扩展。该系统还支持研究人员之间的比较分析，允许用户比较研究主题轨迹和影响力增长。我们的交互式、多标签和多视图系统允许从不同的视角探索职业里程碑，如最具影响力的文章、新兴的研究主题，或对研究者在子领域的贡献进行详细的分析。关键贡献包括AI/ML技术：a) 主题分析，b) 维度减少以可视化模式和趋势，c) 通过可配置提示生成和大型语言模型互动创建数据各个方面文本描述，包括关键指标，以帮助理解个人或团队的职业发展。 

---
# AC-PKAN: Attention-Enhanced and Chebyshev Polynomial-Based Physics-Informed Kolmogorov-Arnold Networks 

**Title (ZH)**: AC-PKAN：注意力增强和切比雪夫多项式基物理指导的柯尔莫哥洛夫-阿诺尔德网络 

**Authors**: Hangwei Zhang, Zhimu Huang, Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08687)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) have recently shown promise for solving partial differential equations (PDEs). Yet their original formulation is computationally and memory intensive, motivating the introduction of Chebyshev Type-I-based KANs (Chebyshev1KANs). Although Chebyshev1KANs have outperformed the vanilla KANs architecture, our rigorous theoretical analysis reveals that they still suffer from rank collapse, ultimately limiting their expressive capacity. To overcome these limitations, we enhance Chebyshev1KANs by integrating wavelet-activated MLPs with learnable parameters and an internal attention mechanism. We prove that this design preserves a full-rank Jacobian and is capable of approximating solutions to PDEs of arbitrary order. Furthermore, to alleviate the loss instability and imbalance introduced by the Chebyshev polynomial basis, we externally incorporate a Residual Gradient Attention (RGA) mechanism that dynamically re-weights individual loss terms according to their gradient norms and residual magnitudes. By jointly leveraging internal and external attention, we present AC-PKAN, a novel architecture that constitutes an enhancement to weakly supervised Physics-Informed Neural Networks (PINNs) and extends the expressive power of KANs. Experimental results from nine benchmark tasks across three domains show that AC-PKAN consistently outperforms or matches state-of-the-art models such as PINNsFormer, establishing it as a highly effective tool for solving complex real-world engineering problems in zero-data or data-sparse regimes. The code will be made publicly available upon acceptance. 

**Abstract (ZH)**: Chebyshev Type-I 基增强的 Kolmogorov-Arnold 网络 (Chebyshev1KANs)：用于求解任意阶偏微分方程的新架构 

---
# A Mamba-based Network for Semi-supervised Singing Melody Extraction Using Confidence Binary Regularization 

**Title (ZH)**: 基于Mamba的半监督唱歌旋律提取网络：采用置信二元正则化 

**Authors**: Xiaoliang He, Kangjie Dong, Jingkai Cao, Shuai Yu, Wei Li, Yi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08681)  

**Abstract**: Singing melody extraction (SME) is a key task in the field of music information retrieval. However, existing methods are facing several limitations: firstly, prior models use transformers to capture the contextual dependencies, which requires quadratic computation resulting in low efficiency in the inference stage. Secondly, prior works typically rely on frequencysupervised methods to estimate the fundamental frequency (f0), which ignores that the musical performance is actually based on notes. Thirdly, transformers typically require large amounts of labeled data to achieve optimal performances, but the SME task lacks of sufficient annotated data. To address these issues, in this paper, we propose a mamba-based network, called SpectMamba, for semi-supervised singing melody extraction using confidence binary regularization. In particular, we begin by introducing vision mamba to achieve computational linear complexity. Then, we propose a novel note-f0 decoder that allows the model to better mimic the musical performance. Further, to alleviate the scarcity of the labeled data, we introduce a confidence binary regularization (CBR) module to leverage the unlabeled data by maximizing the probability of the correct classes. The proposed method is evaluated on several public datasets and the conducted experiments demonstrate the effectiveness of our proposed method. 

**Abstract (ZH)**: 基于SpectMamba的半监督歌唱旋律提取方法：使用置信二元正则化 

---
# MINIMALIST: switched-capacitor circuits for efficient in-memory computation of gated recurrent units 

**Title (ZH)**: MINIMALIST：用于高效内存计算门控递归单元的开关电容电路 

**Authors**: Sebastian Billaudelle, Laura Kriener, Filippo Moro, Tristan Torchet, Melika Payvand  

**Link**: [PDF](https://arxiv.org/pdf/2505.08599)  

**Abstract**: Recurrent neural networks (RNNs) have been a long-standing candidate for processing of temporal sequence data, especially in memory-constrained systems that one may find in embedded edge computing environments. Recent advances in training paradigms have now inspired new generations of efficient RNNs. We introduce a streamlined and hardware-compatible architecture based on minimal gated recurrent units (GRUs), and an accompanying efficient mixed-signal hardware implementation of the model. The proposed design leverages switched-capacitor circuits not only for in-memory computation (IMC), but also for the gated state updates. The mixed-signal cores rely solely on commodity circuits consisting of metal capacitors, transmission gates, and a clocked comparator, thus greatly facilitating scaling and transfer to other technology nodes.
We benchmark the performance of our architecture on time series data, introducing all constraints required for a direct mapping to the hardware system. The direct compatibility is verified in mixed-signal simulations, reproducing data recorded from the software-only network model. 

**Abstract (ZH)**: 递归神经网络（RNNs）一直是处理时间序列数据的候选方法，特别是在嵌入式边缘计算环境中内存受限的系统中。近期在训练范式方面的进展已经启发了新一代高效RNN的设计。我们提出了一种基于 Minimal Gated Recurrent Units (MGRUs) 的精简且硬件兼容的架构，并且配套有高效的混合信号硬件实现。该提案设计不仅利用了开关电容电路进行存内计算（IMC），还用于门控状态更新。混合信号核仅依赖于由金属电容器、传输门和时钟比较器构成的普通电路，从而大大促进了规模化并转移到其他技术节点上。

我们在时间序列数据上benchmark了该架构的性能，并引入了所有必需的硬件约束，以实现直接映射到硬件系统。在混合信号仿真中验证了直接兼容性，并且重现了仅通过软件模型记录的数据。 

---
# DFA-CON: A Contrastive Learning Approach for Detecting Copyright Infringement in DeepFake Art 

**Title (ZH)**: DFA-CON: 一种检测深伪艺术版权侵权的对比学习方法 

**Authors**: Haroon Wahab, Hassan Ugail, Irfan Mehmood  

**Link**: [PDF](https://arxiv.org/pdf/2505.08552)  

**Abstract**: Recent proliferation of generative AI tools for visual content creation-particularly in the context of visual artworks-has raised serious concerns about copyright infringement and forgery. The large-scale datasets used to train these models often contain a mixture of copyrighted and non-copyrighted artworks. Given the tendency of generative models to memorize training patterns, they are susceptible to varying degrees of copyright violation. Building on the recently proposed DeepfakeArt Challenge benchmark, this work introduces DFA-CON, a contrastive learning framework designed to detect copyright-infringing or forged AI-generated art. DFA-CON learns a discriminative representation space, posing affinity among original artworks and their forged counterparts within a contrastive learning framework. The model is trained across multiple attack types, including inpainting, style transfer, adversarial perturbation, and cutmix. Evaluation results demonstrate robust detection performance across most attack types, outperforming recent pretrained foundation models. Code and model checkpoints will be released publicly upon acceptance. 

**Abstract (ZH)**: Recent proliferation of生成AI工具在视觉内容创作中的应用，特别是在视觉艺术领域的背景下，引发了严重的版权侵权和伪造担忧。这些模型的大型训练数据集 often包含版权和非版权艺术作品的混合。鉴于生成模型有记忆训练模式的趋势，它们在不同程度上容易侵犯版权。基于最近提出的DeepfakeArt挑战基准，本文引入了DFA-CON，一种对比学习框架，旨在检测版权侵权或伪造的AI生成艺术。DFA-CON通过对比学习框架学习一个区分性的表示空间，能够在原始艺术品与其伪造版本之间建立亲和力。该模型在包括内容填充、风格转换、对抗扰动和cutmix在内的多种攻击类型上进行了训练。评估结果表明，该模型在大多数攻击类型上表现出稳健的检测性能，优于近期的预训练基础模型。接受后，代码和模型检查点将公开发布。 

---
# ExEBench: Benchmarking Foundation Models on Extreme Earth Events 

**Title (ZH)**: ExEBench: 极端地球事件基础模型评估基准 

**Authors**: Shan Zhao, Zhitong Xiong, Jie Zhao, Xiao Xiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08529)  

**Abstract**: Our planet is facing increasingly frequent extreme events, which pose major risks to human lives and ecosystems. Recent advances in machine learning (ML), especially with foundation models (FMs) trained on extensive datasets, excel in extracting features and show promise in disaster management. Nevertheless, these models often inherit biases from training data, challenging their performance over extreme values. To explore the reliability of FM in the context of extreme events, we introduce \textbf{ExE}Bench (\textbf{Ex}treme \textbf{E}arth Benchmark), a collection of seven extreme event categories across floods, wildfires, storms, tropical cyclones, extreme precipitation, heatwaves, and cold waves. The dataset features global coverage, varying data volumes, and diverse data sources with different spatial, temporal, and spectral characteristics. To broaden the real-world impact of FMs, we include multiple challenging ML tasks that are closely aligned with operational needs in extreme events detection, monitoring, and forecasting. ExEBench aims to (1) assess FM generalizability across diverse, high-impact tasks and domains, (2) promote the development of novel ML methods that benefit disaster management, and (3) offer a platform for analyzing the interactions and cascading effects of extreme events to advance our understanding of Earth system, especially under the climate change expected in the decades to come. The dataset and code are public this https URL. 

**Abstract (ZH)**: 我们的星球正面临着越来越频繁的极端事件，这些事件对人类生活和生态系统构成了重大风险。最近在机器学习（ML）领域的进展，特别是针对大规模数据集训练的基础模型（FMs），在提取特征方面表现出色，并在灾难管理方面展现出前景。然而，这些模型通常会继承训练数据中的偏差，这对其在极端值上的表现构成了挑战。为了探索基础模型在极端事件 context 中的可靠性，我们介绍了 \textbf{ExE}Bench (\textbf{Ex}treme \textbf{E}arth Benchmark)，该基准集合了涵盖洪水、野火、风暴、热带气旋、极端降水、热浪和冷浪等七类极端事件。该数据集具有全球覆盖性，数据量和数据源多样，并且具有不同的空间、时间和光谱特征。为了扩大基础模型在实际世界中的影响，我们纳入了多个与极端事件检测、监测和预报紧密相关的具有挑战性的 ML 任务。ExEBench 的目标是：（1）评估基础模型在多样、高影响任务和领域中的泛化能力；（2）促进有利于灾难管理的新 ML 方法的发展；（3）提供一个平台来分析极端事件之间的相互作用及其连锁效应，以增进对我们地球系统理解，特别是在未来几十年预期的气候变化背景下。数据集和代码在此 https URL。 

---
# GradMix: Gradient-based Selective Mixup for Robust Data Augmentation in Class-Incremental Learning 

**Title (ZH)**: GradMix：基于梯度的Selective Mixup在类增量学习中的鲁棒数据增强 

**Authors**: Minsu Kim, Seong-Hyeon Hwang, Steven Euijong Whang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08528)  

**Abstract**: In the context of continual learning, acquiring new knowledge while maintaining previous knowledge presents a significant challenge. Existing methods often use experience replay techniques that store a small portion of previous task data for training. In experience replay approaches, data augmentation has emerged as a promising strategy to further improve the model performance by mixing limited previous task data with sufficient current task data. However, we theoretically and empirically analyze that training with mixed samples from random sample pairs may harm the knowledge of previous tasks and cause greater catastrophic forgetting. We then propose GradMix, a robust data augmentation method specifically designed for mitigating catastrophic forgetting in class-incremental learning. GradMix performs gradient-based selective mixup using a class-based criterion that mixes only samples from helpful class pairs and not from detrimental class pairs for reducing catastrophic forgetting. Our experiments on various real datasets show that GradMix outperforms data augmentation baselines in accuracy by minimizing the forgetting of previous knowledge. 

**Abstract (ZH)**: 在持续学习的背景下，获取新知识的同时保持先前知识是一项重大挑战。现有方法通常使用经验回放技术，存储少量的先前任务数据进行训练。在经验回放方法中，数据增强被认为是一种通过混合有限的先前任务数据与足够的当前任务数据来进一步提高模型性能的有前途的策略。然而，我们的理论和实证分析表明，使用随机样本对的混合样本进行训练可能会损害先前任务的知识并导致更严重的灾难性遗忘。我们随后提出GradMix，这是一种专门设计用于缓解类别增量学习中灾难性遗忘的鲁棒数据增强方法。GradMix 使用基于类别的标准进行梯度选择性 mixup，只混合适对的样本而不是有害的样本对，以减少灾难性遗忘。我们在多种实际数据集上的实验结果显示，GradMix 在准确性上优于数据增强 baselines，并通过最小化遗忘的先前知识来实现这一目标。 

---
# Learning Advanced Self-Attention for Linear Transformers in the Singular Value Domain 

**Title (ZH)**: 学习奇异值域中高级自注意力机制的线性变换 

**Authors**: Hyowon Wi, Jeongwhan Choi, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.08516)  

**Abstract**: Transformers have demonstrated remarkable performance across diverse domains. The key component of Transformers is self-attention, which learns the relationship between any two tokens in the input sequence. Recent studies have revealed that the self-attention can be understood as a normalized adjacency matrix of a graph. Notably, from the perspective of graph signal processing (GSP), the self-attention can be equivalently defined as a simple graph filter, applying GSP using the value vector as the signal. However, the self-attention is a graph filter defined with only the first order of the polynomial matrix, and acts as a low-pass filter preventing the effective leverage of various frequency information. Consequently, existing self-attention mechanisms are designed in a rather simplified manner. Therefore, we propose a novel method, called \underline{\textbf{A}}ttentive \underline{\textbf{G}}raph \underline{\textbf{F}}ilter (AGF), interpreting the self-attention as learning the graph filter in the singular value domain from the perspective of graph signal processing for directed graphs with the linear complexity w.r.t. the input length $n$, i.e., $\mathcal{O}(nd^2)$. In our experiments, we demonstrate that AGF achieves state-of-the-art performance on various tasks, including Long Range Arena benchmark and time series classification. 

**Abstract (ZH)**: Transformer 在各种领域中展现了卓越的性能。其关键组件是自注意力，它学习输入序列中任意两个词元之间的关系。最近的研究表明，自注意力可以被视为图的归一化邻接矩阵。值得注意的是，从图信号处理（GSP）的角度看，自注意力可以等价地定义为一个简单的图滤波器，使用值向量作为信号进行GSP。然而，自注意力仅基于多项式矩阵的一阶定义了一个图滤波器，并起到低通滤波器的作用，限制了各种频率信息的有效利用。因此，现有的自注意力机制的设计相对简化。因此，我们提出了一种新的方法，称为 \underline{\textbf{A}}ttentive \underline{\textbf{G}}raph \underline{\textbf{F}}ilter（AGF），将其视为对于有向图从图信号处理的角度在奇异值域学习图滤波器，且输入长度 $n$ 的线性复杂度，即 $\mathcal{O}(nd^2)$。在我们的实验中，我们展示了AGF在多种任务上，包括Long Range Arena基准和时间序列分类上达到了最先进的性能。 

---
# An adaptive sampling algorithm for data-generation to build a data-manifold for physical problem surrogate modeling 

**Title (ZH)**: 自适应采样算法用于数据生成以构建物理问题代理模型的数据流形 

**Authors**: Chetra Mang, Axel TahmasebiMoradi, David Danan, Mouadh Yagoubi  

**Link**: [PDF](https://arxiv.org/pdf/2505.08487)  

**Abstract**: Physical models classically involved Partial Differential equations (PDE) and depending of their underlying complexity and the level of accuracy required, and known to be computationally expensive to numerically solve them. Thus, an idea would be to create a surrogate model relying on data generated by such solver. However, training such a model on an imbalanced data have been shown to be a very difficult task. Indeed, if the distribution of input leads to a poor response manifold representation, the model may not learn well and consequently, it may not predict the outcome with acceptable accuracy. In this work, we present an Adaptive Sampling Algorithm for Data Generation (ASADG) involving a physical model. As the initial input data may not accurately represent the response manifold in higher dimension, this algorithm iteratively adds input data into it. At each step the barycenter of each simplicial complex, that the manifold is discretized into, is added as new input data, if a certain threshold is satisfied. We demonstrate the efficiency of the data sampling algorithm in comparison with LHS method for generating more representative input data. To do so, we focus on the construction of a harmonic transport problem metamodel by generating data through a classical solver. By using such algorithm, it is possible to generate the same number of input data as LHS while providing a better representation of the response manifold. 

**Abstract (ZH)**: 自适应采样算法生成数据以增强物理模型（ASADG）：基于平衡数据生成更具代表性的输入数据 

---
# Distributed Quantum Neural Networks on Distributed Photonic Quantum Computing 

**Title (ZH)**: 分布式光子量子计算中的分布式量子神经网络 

**Authors**: Kuan-Cheng Chen, Chen-Yu Liu, Yu Shang, Felix Burt, Kin K. Leung  

**Link**: [PDF](https://arxiv.org/pdf/2505.08474)  

**Abstract**: We introduce a distributed quantum-classical framework that synergizes photonic quantum neural networks (QNNs) with matrix-product-state (MPS) mapping to achieve parameter-efficient training of classical neural networks. By leveraging universal linear-optical decompositions of $M$-mode interferometers and photon-counting measurement statistics, our architecture generates neural parameters through a hybrid quantum-classical workflow: photonic QNNs with $M(M+1)/2$ trainable parameters produce high-dimensional probability distributions that are mapped to classical network weights via an MPS model with bond dimension $\chi$. Empirical validation on MNIST classification demonstrates that photonic QT achieves an accuracy of $95.50\% \pm 0.84\%$ using 3,292 parameters ($\chi = 10$), compared to $96.89\% \pm 0.31\%$ for classical baselines with 6,690 parameters. Moreover, a ten-fold compression ratio is achieved at $\chi = 4$, with a relative accuracy loss of less than $3\%$. The framework outperforms classical compression techniques (weight sharing/pruning) by 6--12\% absolute accuracy while eliminating quantum hardware requirements during inference through classical deployment of compressed parameters. Simulations incorporating realistic photonic noise demonstrate the framework's robustness to near-term hardware imperfections. Ablation studies confirm quantum necessity: replacing photonic QNNs with random inputs collapses accuracy to chance level ($10.0\% \pm 0.5\%$). Photonic quantum computing's room-temperature operation, inherent scalability through spatial-mode multiplexing, and HPC-integrated architecture establish a practical pathway for distributed quantum machine learning, combining the expressivity of photonic Hilbert spaces with the deployability of classical neural networks. 

**Abstract (ZH)**: 一种结合光子量子神经网络与矩阵积态映射的分布式量子-经典框架：参数高效训练经典神经网络 

---
# Hakim: Farsi Text Embedding Model 

**Title (ZH)**: Hákim: 波斯文本嵌入模型 

**Authors**: Mehran Sarmadi, Morteza Alikhani, Erfan Zinvandi, Zahra Pourbahman  

**Link**: [PDF](https://arxiv.org/pdf/2505.08435)  

**Abstract**: Recent advancements in text embedding have significantly improved natural language understanding across many languages, yet Persian remains notably underrepresented in large-scale embedding research. In this paper, we present Hakim, a novel state-of-the-art Persian text embedding model that achieves a 8.5% performance improvement over existing approaches on the FaMTEB benchmark, outperforming all previously developed Persian language models. As part of this work, we introduce three new datasets - Corpesia, Pairsia-sup, and Pairsia-unsup - to support supervised and unsupervised training scenarios. Additionally, Hakim is designed for applications in chatbots and retrieval-augmented generation (RAG) systems, particularly addressing retrieval tasks that require incorporating message history within these systems. We also propose a new baseline model built on the BERT architecture. Our language model consistently achieves higher accuracy across various Persian NLP tasks, while the RetroMAE-based model proves particularly effective for textual information retrieval applications. Together, these contributions establish a new foundation for advancing Persian language understanding. 

**Abstract (ZH)**: 最近在文本嵌入方面的进展显著提高了多种语言的自然语言理解能力，但波斯语在大规模嵌入研究中的代表性仍然不足。本文介绍了Hakim，一种新型的波斯文本嵌入模型，在FaMTEB基准测试中相比现有方法取得了8.5%的性能提升，超过了之前所有开发的波斯语言模型。作为这项工作的组成部分，我们引入了三个新的数据集——Corpesia、Pairsia-sup和Pairsia-unsup，以支持监督和无监督的训练场景。此外，Hakim 专为聊天机器人和检索增强生成（RAG）系统设计，特别针对需要在这些系统中整合消息历史的检索任务。我们还提出了一种基于BERT架构的新基线模型。我们的语言模型在各种波斯NLP任务中始终保持更高的准确性，而基于RetroMAE的模型在文本信息检索应用中表现尤为出色。这些贡献共同为推进波斯语理解奠定了新的基础。 

---
# ConDiSim: Conditional Diffusion Models for Simulation Based Inference 

**Title (ZH)**: ConDiSim：基于条件的扩散模型在模拟推断中的应用 

**Authors**: Mayank Nautiyal, Andreas Hellander, Prashant Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.08403)  

**Abstract**: We present a conditional diffusion model - ConDiSim, for simulation-based inference of complex systems with intractable likelihoods. ConDiSim leverages denoising diffusion probabilistic models to approximate posterior distributions, consisting of a forward process that adds Gaussian noise to parameters, and a reverse process learning to denoise, conditioned on observed data. This approach effectively captures complex dependencies and multi-modalities within posteriors. ConDiSim is evaluated across ten benchmark problems and two real-world test problems, where it demonstrates effective posterior approximation accuracy while maintaining computational efficiency and stability in model training. ConDiSim offers a robust and extensible framework for simulation-based inference, particularly suitable for parameter inference workflows requiring fast inference methods. 

**Abstract (ZH)**: 基于模拟的难以计算似然函数的复杂系统条件扩散模型-ConDiSim推理方法 

---
# Non-contact Vital Signs Detection in Dynamic Environments 

**Title (ZH)**: 动态环境中的非接触生命体征检测 

**Authors**: Shuai Sun, Chong-Xi Liang, Chengwei Ye, Huanzhen Zhang, Kangsheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08366)  

**Abstract**: Accurate phase demodulation is critical for vital sign detection using millimeter-wave radar. However, in complex environments, time-varying DC offsets and phase imbalances can severely degrade demodulation performance. To address this, we propose a novel DC offset calibration method alongside a Hilbert and Differential Cross-Multiply (HADCM) demodulation algorithm. The approach estimates time-varying DC offsets from neighboring signal peaks and valleys, then employs both differential forms and Hilbert transforms of the I/Q channel signals to extract vital sign information. Simulation and experimental results demonstrate that the proposed method maintains robust performance under low signal-to-noise ratios. Compared to existing demodulation techniques, it offers more accurate signal recovery in challenging scenarios and effectively suppresses noise interference. 

**Abstract (ZH)**: 准确的相位解调对于使用毫米波雷达检测生理信号至关重要。然而，在复杂环境中，时间 varying 直流偏移和相位失配会严重降低解调性能。为此，我们提出了一种新颖的直流偏移校准方法以及基于希尔伯特和差分交叉乘积（HADCM）的解调算法。该方法从相邻信号峰值和谷值中估计时间 varying 直流偏移，然后利用 I/Q 通道信号的差分解和希尔伯特变换提取生理信号信息。仿真和实验结果表明，所提出的方法在低信噪比下保持了稳健的性能。与现有的解调技术相比，它在具有挑战性的场景中提供了更准确的信号恢复，并有效抑制了噪声干扰。 

---
# STORYANCHORS: Generating Consistent Multi-Scene Story Frames for Long-Form Narratives 

**Title (ZH)**: 故事锚点：生成长篇叙事的 consistent 多场景故事框架 

**Authors**: Bo Wang, Haoyang Huang, Zhiyin Lu, Fengyuan Liu, Guoqing Ma, Jianlong Yuan, Yuan Zhang, Nan Duan  

**Link**: [PDF](https://arxiv.org/pdf/2505.08350)  

**Abstract**: This paper introduces StoryAnchors, a unified framework for generating high-quality, multi-scene story frames with strong temporal consistency. The framework employs a bidirectional story generator that integrates both past and future contexts to ensure temporal consistency, character continuity, and smooth scene transitions throughout the narrative. Specific conditions are introduced to distinguish story frame generation from standard video synthesis, facilitating greater scene diversity and enhancing narrative richness. To further improve generation quality, StoryAnchors integrates Multi-Event Story Frame Labeling and Progressive Story Frame Training, enabling the model to capture both overarching narrative flow and event-level dynamics. This approach supports the creation of editable and expandable story frames, allowing for manual modifications and the generation of longer, more complex sequences. Extensive experiments show that StoryAnchors outperforms existing open-source models in key areas such as consistency, narrative coherence, and scene diversity. Its performance in narrative consistency and story richness is also on par with GPT-4o. Ultimately, StoryAnchors pushes the boundaries of story-driven frame generation, offering a scalable, flexible, and highly editable foundation for future research. 

**Abstract (ZH)**: StoryAnchors：一种生成高质量多场景故事框架的统一框架 

---
# FAD: Frequency Adaptation and Diversion for Cross-domain Few-shot Learning 

**Title (ZH)**: FAD：频率适配与转移missive跨域少样本学习 

**Authors**: Ruixiao Shi, Fu Feng, Yucheng Xie, Jing Wang, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2505.08349)  

**Abstract**: Cross-domain few-shot learning (CD-FSL) requires models to generalize from limited labeled samples under significant distribution shifts. While recent methods enhance adaptability through lightweight task-specific modules, they operate solely in the spatial domain and overlook frequency-specific variations that are often critical for robust transfer. We observe that spatially similar images across domains can differ substantially in their spectral representations, with low and high frequencies capturing complementary semantic information at coarse and fine levels. This indicates that uniform spatial adaptation may overlook these spectral distinctions, thus constraining generalization. To address this, we introduce Frequency Adaptation and Diversion (FAD), a frequency-aware framework that explicitly models and modulates spectral components. At its core is the Frequency Diversion Adapter, which transforms intermediate features into the frequency domain using the discrete Fourier transform (DFT), partitions them into low, mid, and high-frequency bands via radial masks, and reconstructs each band using inverse DFT (IDFT). Each frequency band is then adapted using a dedicated convolutional branch with a kernel size tailored to its spectral scale, enabling targeted and disentangled adaptation across frequencies. Extensive experiments on the Meta-Dataset benchmark demonstrate that FAD consistently outperforms state-of-the-art methods on both seen and unseen domains, validating the utility of frequency-domain representations and band-wise adaptation for improving generalization in CD-FSL. 

**Abstract (ZH)**: 跨领域少样本学习中的频率适应与离散（Frequency Adaptation and Diversion for Cross-Domain Few-Shot Learning） 

---
# SHAP-based Explanations are Sensitive to Feature Representation 

**Title (ZH)**: 基于SHAP的解释对特征表示敏感 

**Authors**: Hyunseung Hwang, Andrew Bell, Joao Fonseca, Venetia Pliatsika, Julia Stoyanovich, Steven Euijong Whang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08345)  

**Abstract**: Local feature-based explanations are a key component of the XAI toolkit. These explanations compute feature importance values relative to an ``interpretable'' feature representation. In tabular data, feature values themselves are often considered interpretable. This paper examines the impact of data engineering choices on local feature-based explanations. We demonstrate that simple, common data engineering techniques, such as representing age with a histogram or encoding race in a specific way, can manipulate feature importance as determined by popular methods like SHAP. Notably, the sensitivity of explanations to feature representation can be exploited by adversaries to obscure issues like discrimination. While the intuition behind these results is straightforward, their systematic exploration has been lacking. Previous work has focused on adversarial attacks on feature-based explainers by biasing data or manipulating models. To the best of our knowledge, this is the first study demonstrating that explainers can be misled by standard, seemingly innocuous data engineering techniques. 

**Abstract (ZH)**: 基于局部特征的解释是XAI工具包的关键组成部分。这些解释计算相对于“可解释”特征表示的特征重要性值。在表格数据中，特征值本身往往被认为是可解释的。本文探讨了数据工程选择对基于局部特征的解释的影响。我们展示了简单的常见数据工程技术，如将年龄表示为直方图或以特定方式编码种族，可以操控如SHAP等流行方法确定的特征重要性。值得注意的是，解释对特征表示的敏感性可以被对手利用以掩盖诸如歧视之类的问题。尽管这些结果背后的直觉是直接的，但它们的系统性探索却一直缺失。以往的工作主要集中在通过偏向数据或操控模型来进行特征基解释的对抗攻击。据我们所知，这是第一次研究证明标准的、看似无害的数据工程技术可以误导解释器的研究。 

---
# Low-Complexity Inference in Continual Learning via Compressed Knowledge Transfer 

**Title (ZH)**: 低复杂度持续学习中的压缩知识迁移推理 

**Authors**: Zhenrong Liu, Janne M. J. Huttunen, Mikko Honkala  

**Link**: [PDF](https://arxiv.org/pdf/2505.08327)  

**Abstract**: Continual learning (CL) aims to train models that can learn a sequence of tasks without forgetting previously acquired knowledge. A core challenge in CL is balancing stability -- preserving performance on old tasks -- and plasticity -- adapting to new ones. Recently, large pre-trained models have been widely adopted in CL for their ability to support both, offering strong generalization for new tasks and resilience against forgetting. However, their high computational cost at inference time limits their practicality in real-world applications, especially those requiring low latency or energy efficiency. To address this issue, we explore model compression techniques, including pruning and knowledge distillation (KD), and propose two efficient frameworks tailored for class-incremental learning (CIL), a challenging CL setting where task identities are unavailable during inference. The pruning-based framework includes pre- and post-pruning strategies that apply compression at different training stages. The KD-based framework adopts a teacher-student architecture, where a large pre-trained teacher transfers downstream-relevant knowledge to a compact student. Extensive experiments on multiple CIL benchmarks demonstrate that the proposed frameworks achieve a better trade-off between accuracy and inference complexity, consistently outperforming strong baselines. We further analyze the trade-offs between the two frameworks in terms of accuracy and efficiency, offering insights into their use across different scenarios. 

**Abstract (ZH)**: 连续学习（CL）旨在训练能够在不忘记先前获得的知识的情况下学习一系列任务的模型。CL的核心挑战是在保持旧任务性能（稳定性）与适应新任务（可塑性）之间取得平衡。近年来，由于其支持两者的能力，大型预训练模型在CL中得到了广泛应用，能够为新任务提供强大的泛化能力和抵抗遗忘的鲁棒性。然而，它们在推理时的高计算成本限制了其实用性，特别是在需要低延迟或高能效的应用中。为解决这一问题，我们探索了模型压缩技术，包括剪枝和知识蒸馏（KD），并提出两种针对类别增量学习（CIL）的高效框架，CIL是一个在推理期间缺乏任务身份信息的挑战性CL设置。基于剪枝的框架包括预剪枝和后剪枝策略，分别在训练的不同阶段进行压缩。基于KD的框架采用老师-学生架构，其中大型预训练老师向紧凑的学生传递下游相关知识。在多个CIL基准上的广泛实验表明，所提出的框架在准确性和推理复杂性之间取得了更好的权衡，始终优于强 baseline。我们进一步分析了两种框架在准确性和效率方面的权衡，为它们在不同场景中的使用提供了见解。 

---
# FedRS-Bench: Realistic Federated Learning Datasets and Benchmarks in Remote Sensing 

**Title (ZH)**: FedRS-Bench: 远程 sensing 领域的现实联邦学习数据集和基准 

**Authors**: Haodong Zhao, Peng Peng, Chiyu Chen, Linqing Huang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08325)  

**Abstract**: Remote sensing (RS) images are usually produced at an unprecedented scale, yet they are geographically and institutionally distributed, making centralized model training challenging due to data-sharing restrictions and privacy concerns. Federated learning (FL) offers a solution by enabling collaborative model training across decentralized RS data sources without exposing raw data. However, there lacks a realistic federated dataset and benchmark in RS. Prior works typically rely on manually partitioned single dataset, which fail to capture the heterogeneity and scale of real-world RS data, and often use inconsistent experimental setups, hindering fair comparison. To address this gap, we propose a realistic federated RS dataset, termed FedRS. FedRS consists of eight datasets that cover various sensors and resolutions and builds 135 clients, which is representative of realistic operational scenarios. Data for each client come from the same source, exhibiting authentic federated properties such as skewed label distributions, imbalanced client data volumes, and domain heterogeneity across clients. These characteristics reflect practical challenges in federated RS and support evaluation of FL methods at scale. Based on FedRS, we implement 10 baseline FL algorithms and evaluation metrics to construct the comprehensive FedRS-Bench. The experimental results demonstrate that FL can consistently improve model performance over training on isolated data silos, while revealing performance trade-offs of different methods under varying client heterogeneity and availability conditions. We hope FedRS-Bench will accelerate research on large-scale, realistic FL in RS by providing a standardized, rich testbed and facilitating fair comparisons across future works. The source codes and dataset are available at this https URL. 

**Abstract (ZH)**: 面向现实需求的联邦遥感数据集及基准 FedRS-Bench 

---
# Reciprocity as the Foundational Substrate of Society: How Reciprocal Dynamics Scale into Social Systems 

**Title (ZH)**: reciprocity 作为社会的基础substrate：递归动态如何扩展到社会系统 

**Authors**: Egil Diau  

**Link**: [PDF](https://arxiv.org/pdf/2505.08319)  

**Abstract**: A major bottleneck in multi-agent AI is the lack of simulateable models for the bottom-up emergence of social structure under realistic behavioral constraints. Similarly, many foundational theories in economics and sociology including the concepts of "institutions" and "norms" tend to describe social structures post hoc, often relying on implicit assumptions of shared culture, morality, or symbolic agreement. These concepts are often treated as primitives rather than reconstructed from agent-level behavior, leaving both their origins and operational definitions under-specified. To address this, we propose a three-stage bottom-up framework: Reciprocal Dynamics, capturing individual-level reciprocal exchanges; Norm Stabilization, the consolidation of shared expectations; and Institutional Construction, the externalization of stable patterns into scalable structures. By grounding social emergence in agent-level reciprocity, our framework enables the systematic exploration of how moral, cultural, and institutional structures emerge from cognitively minimal interactions. 

**Abstract (ZH)**: 多智能体AI领域的一个主要瓶颈是对应自底向上的社会结构在现实行为约束下缺乏可模拟的模型。同样，经济学和社会学中的许多基础理论，包括“制度”和“规范”的概念，倾向于事后描述社会结构，常常依赖于共享文化、道德或符号协议的隐含假设。这些概念往往被视为基本构建块，而不是从个体层面的行为中重建，因此它们的起源和操作定义经常是不明确的。为了解决这一问题，我们提出一个三阶段自底向上的框架：互惠动力学，捕捉个体层面的互惠交换；规范稳定化，巩固共享预期；制度构建，将稳定的模式外化为可扩展的结构。通过将社会涌现基于个体层面的互惠性，我们的框架使得系统地探索道德、文化和制度结构如何从认知最小的交互中涌现成为可能。 

---
# Removing Watermarks with Partial Regeneration using Semantic Information 

**Title (ZH)**: 使用语义信息进行部分再生去除水印 

**Authors**: Krti Tallam, John Kevin Cava, Caleb Geniesse, N. Benjamin Erichson, Michael W. Mahoney  

**Link**: [PDF](https://arxiv.org/pdf/2505.08234)  

**Abstract**: As AI-generated imagery becomes ubiquitous, invisible watermarks have emerged as a primary line of defense for copyright and provenance. The newest watermarking schemes embed semantic signals - content-aware patterns that are designed to survive common image manipulations - yet their true robustness against adaptive adversaries remains under-explored. We expose a previously unreported vulnerability and introduce SemanticRegen, a three-stage, label-free attack that erases state-of-the-art semantic and invisible watermarks while leaving an image's apparent meaning intact. Our pipeline (i) uses a vision-language model to obtain fine-grained captions, (ii) extracts foreground masks with zero-shot segmentation, and (iii) inpaints only the background via an LLM-guided diffusion model, thereby preserving salient objects and style cues. Evaluated on 1,000 prompts across four watermarking systems - TreeRing, StegaStamp, StableSig, and DWT/DCT - SemanticRegen is the only method to defeat the semantic TreeRing watermark (p = 0.10 > 0.05) and reduces bit-accuracy below 0.75 for the remaining schemes, all while maintaining high perceptual quality (masked SSIM = 0.94 +/- 0.01). We further introduce masked SSIM (mSSIM) to quantify fidelity within foreground regions, showing that our attack achieves up to 12 percent higher mSSIM than prior diffusion-based attackers. These results highlight an urgent gap between current watermark defenses and the capabilities of adaptive, semantics-aware adversaries, underscoring the need for watermarking algorithms that are resilient to content-preserving regenerative attacks. 

**Abstract (ZH)**: 随着AI生成图像的普及，隐式水印已成为版权和溯源的主要防线。最新的水印方案嵌入了语义信号——适应常见图像处理设计的内容感知模式，但它们对抗适应性对手的真实鲁棒性仍被低估。我们揭示了一种未被报道的脆弱性，并引入了SemanticRegen，这是一种三阶段、无标签攻击，能够在不损害图像显见含义的前提下消除最先进的语义和隐式水印。我们的管道包括：(i) 使用视觉-语言模型获取精细的描述，(ii) 使用零样本分割提取前景掩码，(iii) 通过受LLM指导的扩散模型仅修复背景，从而保留了显著对象和风格线索。在四个水印系统——TreeRing、StegaStamp、StableSig和DWT/DCT——上评估了1,000个提示后，SemanticRegen是唯一一种能击败语义TreeRing水印(p = 0.10 > 0.05)的方法，并且能将其余方案的比特准确率降低至0.75以下，同时保持高感知质量（屏蔽SSIM = 0.94 ± 0.01）。我们还引入了屏蔽SSIM (mSSIM) 以量化前景区域内的保真度，结果显示我们的攻击实现了比先前基于扩散的攻击高12个百分点的mSSIM。这些结果突显了当前水印防御与适应性强、具备语义意识的对手能力之间的重要差距，强调了需要具备抵御内容保持再生攻击的水印算法的需求。 

---
# Feasibility-Aware Pessimistic Estimation: Toward Long-Horizon Safety in Offline RL 

**Title (ZH)**: 可行性意识的悲观估计：迈向离线RL中的长时域安全 

**Authors**: Zhikun Tao, Gang Xiong, He Fang, Zhen Shen, Yunjun Han, Qing-Shan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2505.08179)  

**Abstract**: Offline safe reinforcement learning(OSRL) derives constraint-satisfying policies from pre-collected datasets, offers a promising avenue for deploying RL in safety-critical real-world domains such as robotics. However, the majority of existing approaches emphasize only short-term safety, neglecting long-horizon considerations. Consequently, they may violate safety constraints and fail to ensure sustained protection during online deployment. Moreover, the learned policies often struggle to handle states and actions that are not present or out-of-distribution(OOD) from the offline dataset, and exhibit limited sample efficiency. To address these challenges, we propose a novel framework Feasibility-Aware offline Safe Reinforcement Learning with CVAE-based Pessimism (FASP). First, we employ Hamilton-Jacobi (H-J) reachability analysis to generate reliable safety labels, which serve as supervisory signals for training both a conditional variational autoencoder (CVAE) and a safety classifier. This approach not only ensures high sampling efficiency but also provides rigorous long-horizon safety guarantees. Furthermore, we utilize pessimistic estimation methods to estimate the Q-value of reward and cost, which mitigates the extrapolation errors induces by OOD actions, and penalize unsafe actions to enabled the agent to proactively avoid high-risk behaviors. Moreover, we theoretically prove the validity of this pessimistic estimation. Extensive experiments on DSRL benchmarks demonstrate that FASP algorithm achieves competitive performance across multiple experimental tasks, particularly outperforming state-of-the-art algorithms in terms of safety. 

**Abstract (ZH)**: 基于CVAE悲观估计的可行性 Awareness Offline 安全强化学习（FASP） 

---
# Fast Text-to-Audio Generation with Adversarial Post-Training 

**Title (ZH)**: Fast文本到语音生成的对抗性后训练方法 

**Authors**: Zachary Novack, Zach Evans, Zack Zukowski, Josiah Taylor, CJ Carr, Julian Parker, Adnan Al-Sinan, Gian Marco Iodice, Julian McAuley, Taylor Berg-Kirkpatrick, Jordi Pons  

**Link**: [PDF](https://arxiv.org/pdf/2505.08175)  

**Abstract**: Text-to-audio systems, while increasingly performant, are slow at inference time, thus making their latency unpractical for many creative applications. We present Adversarial Relativistic-Contrastive (ARC) post-training, the first adversarial acceleration algorithm for diffusion/flow models not based on distillation. While past adversarial post-training methods have struggled to compare against their expensive distillation counterparts, ARC post-training is a simple procedure that (1) extends a recent relativistic adversarial formulation to diffusion/flow post-training and (2) combines it with a novel contrastive discriminator objective to encourage better prompt adherence. We pair ARC post-training with a number optimizations to Stable Audio Open and build a model capable of generating $\approx$12s of 44.1kHz stereo audio in $\approx$75ms on an H100, and $\approx$7s on a mobile edge-device, the fastest text-to-audio model to our knowledge. 

**Abstract (ZH)**: 基于对抗相对对比的后训练加速算法： diffusion/flow模型的首个不基于蒸馏的加速方法 

---
# Exploiting Text Semantics for Few and Zero Shot Node Classification on Text-attributed Graph 

**Title (ZH)**: 利用文本语义进行基于文本的图中节点少量和零样本分类 

**Authors**: Yuxiang Wang, Xiao Yan, Shiyu Jin, Quanqing Xu, Chuang Hu, Yuanyuan Zhu, Bo Du, Jia Wu, Jiawei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08168)  

**Abstract**: Text-attributed graph (TAG) provides a text description for each graph node, and few- and zero-shot node classification on TAGs have many applications in fields such as academia and social networks. Existing work utilizes various graph-based augmentation techniques to train the node and text embeddings, while text-based augmentations are largely unexplored. In this paper, we propose Text Semantics Augmentation (TSA) to improve accuracy by introducing more text semantic supervision signals. Specifically, we design two augmentation techniques, i.e., positive semantics matching and negative semantics contrast, to provide more reference texts for each graph node or text description. Positive semantic matching retrieves texts with similar embeddings to match with a graph node. Negative semantic contrast adds a negative prompt to construct a text description with the opposite semantics, which is contrasted with the original node and text. We evaluate TSA on 5 datasets and compare with 13 state-of-the-art baselines. The results show that TSA consistently outperforms all baselines, and its accuracy improvements over the best-performing baseline are usually over 5%. 

**Abstract (ZH)**: 基于文本的图（Text-attributed Graph, TAG）提供了每个图节点的文字描述，且基于少量或零样本的TAG节点分类在学术界和社交网络等领域有广泛应用。现有的工作利用了各种图结构增强技术训练节点和文本嵌入，而基于文本的增强技术尚未得到充分利用。本文提出文本语义增强（TSA）以通过引入更多的文本语义监督信号来提高准确性。具体地，我们设计了正语义匹配和负语义对比两种增强技术，为每个图节点或文本描述提供更多参考文本。正语义匹配检索具有相似嵌入的文本以匹配图节点。负语义对比增加一个负向提示来构造语义相反的文本描述，该描述与原始节点和文本进行对比。我们在5个数据集上评估了TSA，并与13个state-of-the-artbaseline进行了比较。结果显示，TSA一直优于所有baseline，并且其相对于表现最好的baseline的准确性提升通常超过5%。 

---
# Feature Fitted Online Conformal Prediction for Deep Time Series Forecasting Model 

**Title (ZH)**: 基于特征拟合的在线同变预测深时序forecasting模型 

**Authors**: Xiannan Huang, Shuhan Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08158)  

**Abstract**: Time series forecasting is critical for many applications, where deep learning-based point prediction models have demonstrated strong performance. However, in practical scenarios, there is also a need to quantify predictive uncertainty through online confidence intervals. Existing confidence interval modeling approaches building upon these deep point prediction models suffer from key limitations: they either require costly retraining, fail to fully leverage the representational strengths of deep models, or lack theoretical guarantees. To address these gaps, we propose a lightweight conformal prediction method that provides valid coverage and shorter interval lengths without retraining. Our approach leverages features extracted from pre-trained point prediction models to fit a residual predictor and construct confidence intervals, further enhanced by an adaptive coverage control mechanism. Theoretically, we prove that our method achieves asymptotic coverage convergence, with error bounds dependent on the feature quality of the underlying point prediction model. Experiments on 12 datasets demonstrate that our method delivers tighter confidence intervals while maintaining desired coverage rates. Code, model and dataset in \href{this https URL}{Github} 

**Abstract (ZH)**: 基于深度学习的时间序列预测区间预测方法：一种无需重新训练的轻量级校准预测方法 

---
# Hyperbolic Contrastive Learning with Model-augmentation for Knowledge-aware Recommendation 

**Title (ZH)**: 基于模型增强的双曲对比学习在知识aware推荐中的应用 

**Authors**: Shengyin Sun, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.08157)  

**Abstract**: Benefiting from the effectiveness of graph neural networks (GNNs) and contrastive learning, GNN-based contrastive learning has become mainstream for knowledge-aware recommendation. However, most existing contrastive learning-based methods have difficulties in effectively capturing the underlying hierarchical structure within user-item bipartite graphs and knowledge graphs. Moreover, they commonly generate positive samples for contrastive learning by perturbing the graph structure, which may lead to a shift in user preference learning. To overcome these limitations, we propose hyperbolic contrastive learning with model-augmentation for knowledge-aware recommendation. To capture the intrinsic hierarchical graph structures, we first design a novel Lorentzian knowledge aggregation mechanism, which enables more effective representations of users and items. Then, we propose three model-level augmentation techniques to assist Hyperbolic contrastive learning. Different from the classical structure-level augmentation (e.g., edge dropping), the proposed model-augmentations can avoid preference shifts between the augmented positive pair. Finally, we conduct extensive experiments to demonstrate the superiority (maximum improvement of $11.03\%$) of proposed methods over existing baselines. 

**Abstract (ZH)**: 基于双曲对比学习的知识aware推荐模型增强技术 

---
# Mirror Mirror on the Wall, Have I Forgotten it All? A New Framework for Evaluating Machine Unlearning 

**Title (ZH)**: 镜子前的我，是不是忘了这一切？一种评价机器遗忘的新框架 

**Authors**: Brennon Brimhall, Philip Mathew, Neil Fendley, Yinzhi Cao, Matthew Green  

**Link**: [PDF](https://arxiv.org/pdf/2505.08138)  

**Abstract**: Machine unlearning methods take a model trained on a dataset and a forget set, then attempt to produce a model as if it had only been trained on the examples not in the forget set. We empirically show that an adversary is able to distinguish between a mirror model (a control model produced by retraining without the data to forget) and a model produced by an unlearning method across representative unlearning methods from the literature. We build distinguishing algorithms based on evaluation scores in the literature (i.e. membership inference scores) and Kullback-Leibler divergence.
We propose a strong formal definition for machine unlearning called computational unlearning. Computational unlearning is defined as the inability for an adversary to distinguish between a mirror model and a model produced by an unlearning method. If the adversary cannot guess better than random (except with negligible probability), then we say that an unlearning method achieves computational unlearning.
Our computational unlearning definition provides theoretical structure to prove unlearning feasibility results. For example, our computational unlearning definition immediately implies that there are no deterministic computational unlearning methods for entropic learning algorithms. We also explore the relationship between differential privacy (DP)-based unlearning methods and computational unlearning, showing that DP-based approaches can satisfy computational unlearning at the cost of an extreme utility collapse. These results demonstrate that current methodology in the literature fundamentally falls short of achieving computational unlearning. We conclude by identifying several open questions for future work. 

**Abstract (ZH)**: 机器去学习方法 empirical 表明对手能够区分镜像模型和去学习方法生成的模型 across 文献中代表性的去学习方法。我们基于文献中的评估分数（即成员推断分数）和克劳德-莱布勒散度构建了区分算法。

我们提出了机器去学习的强形式化定义，称为计算去学习。计算去学习定义为对手无法区分镜像模型和去学习方法生成的模型。如果对手无法比随机猜测更好（除了微不足道的概率），则我们认为去学习方法实现了计算去学习。

我们的计算去学习定义为证明去学习可行性结果提供了理论结构。例如，我们的计算去学习定义直接表明，对于熵学习算法，不存在确定性的计算去学习方法。我们还探讨了基于差分隐私（DP）的去学习方法与计算去学习之间的关系，表明为了满足计算去学习，基于DP的方法会导致极高的实用性下降。这些结果表明文献中的现有方法在实现计算去学习方面从根本上存在不足。最后，我们指出了未来工作的几个开放问题。 

---
# Leveraging AI for Productive and Trustworthy HPC Software: Challenges and Research Directions 

**Title (ZH)**: 利用AI促进高效可信赖的HPC软件开发：挑战与研究方向 

**Authors**: Keita Teranishi, Harshitha Menon, William F. Godoy, Prasanna Balaprakash, David Bau, Tal Ben-Nun, Abhinav Bathele, Franz Franchetti, Michael Franusich, Todd Gamblin, Giorgis Georgakoudis, Tom Goldstein, Arjun Guha, Steven Hahn, Costin Iancu, Zheming Jin, Terry Jones, Tze Meng Low, Het Mankad, Narasinga Rao Miniskar, Mohammad Alaul Haque Monil, Daniel Nichols, Konstantinos Parasyris, Swaroop Pophale, Pedro Valero-Lara, Jeffrey S. Vetter, Samuel Williams, Aaron Young  

**Link**: [PDF](https://arxiv.org/pdf/2505.08135)  

**Abstract**: We discuss the challenges and propose research directions for using AI to revolutionize the development of high-performance computing (HPC) software. AI technologies, in particular large language models, have transformed every aspect of software development. For its part, HPC software is recognized as a highly specialized scientific field of its own. We discuss the challenges associated with leveraging state-of-the-art AI technologies to develop such a unique and niche class of software and outline our research directions in the two US Department of Energy--funded projects for advancing HPC Software via AI: Ellora and Durban. 

**Abstract (ZH)**: 我们探讨利用AI革新高性能计算（HPC）软件开发面临的挑战，并提出研究方向。作为特别专业的科学领域，HPC软件受到关注。我们讨论了利用最新AI技术开发这种独特且狭窄类别的软件所面临的挑战，并概述了在两个由美国能源部资助的通过AI推进HPC软件的项目Ellora和Durban中的研究方向。 

---
# One Bad NOFO? AI Governance in Federal Grantmaking 

**Title (ZH)**: 一坏百坏的NOFO？联邦拨款中的AI治理 

**Authors**: Dan Bateyko, Karen Levy  

**Link**: [PDF](https://arxiv.org/pdf/2505.08133)  

**Abstract**: Much scholarship considers how U.S. federal agencies govern artificial intelligence (AI) through rulemaking and their own internal use policies. But agencies have an overlooked AI governance role: setting discretionary grant policy when directing billions of dollars in federal financial assistance. These dollars enable state and local entities to study, create, and use AI. This funding not only goes to dedicated AI programs, but also to grantees using AI in the course of meeting their routine grant objectives. As discretionary grantmakers, agencies guide and restrict what grant winners do -- a hidden lever for AI governance. Agencies pull this lever by setting program objectives, judging criteria, and restrictions for AI use. Using a novel dataset of over 40,000 non-defense federal grant notices of funding opportunity (NOFOs) posted to this http URL between 2009 and 2024, we analyze how agencies regulate the use of AI by grantees. We select records mentioning AI and review their stated goals and requirements. We find agencies promoting AI in notice narratives, shaping adoption in ways other records of grant policy might fail to capture. Of the grant opportunities that mention AI, we find only a handful of AI-specific judging criteria or restrictions. This silence holds even when agencies fund AI uses in contexts affecting people's rights and which, under an analogous federal procurement regime, would result in extra oversight. These findings recast grant notices as a site of AI policymaking -- albeit one that is developing out of step with other regulatory efforts and incomplete in its consideration of transparency, accountability, and privacy protections. The paper concludes by drawing lessons from AI procurement scholarship, while identifying distinct challenges in grantmaking that invite further study. 

**Abstract (ZH)**: 美国联邦机构在发放资助政策中监管人工智能：一个被忽视的治理角色 

---
# High-order Regularization for Machine Learning and Learning-based Control 

**Title (ZH)**: 高阶正则化在机器学习与基于学习的控制中的应用 

**Authors**: Xinghua Liu, Ming Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.08129)  

**Abstract**: The paper proposes a novel regularization procedure for machine learning. The proposed high-order regularization (HR) provides new insight into regularization, which is widely used to train a neural network that can be utilized to approximate the action-value function in general reinforcement learning problems. The proposed HR method ensures the provable convergence of the approximation algorithm, which makes the much-needed connection between regularization and explainable learning using neural networks. The proposed HR method theoretically demonstrates that regularization can be regarded as an approximation in terms of inverse mapping with explicitly calculable approximation error, and the $L_2$ regularization is a lower-order case of the proposed method. We provide lower and upper bounds for the error of the proposed HR solution, which helps build a reliable model. We also find that regularization with the proposed HR can be regarded as a contraction. We prove that the generalizability of neural networks can be maximized with a proper regularization matrix, and the proposed HR is applicable for neural networks with any mapping matrix. With the theoretical explanation of the extreme learning machine for neural network training and the proposed high-order regularization, one can better interpret the output of the neural network, thus leading to explainable learning. We present a case study based on regularized extreme learning neural networks to demonstrate the application of the proposed HR and give the corresponding incremental HR solution. We verify the performance of the proposed HR method by solving a classic control problem in reinforcement learning. The result demonstrates the superior performance of the method with significant enhancement in the generalizability of the neural network. 

**Abstract (ZH)**: 一种新型机器学习正则化程序：高阶正则化在强化学习中的应用 

---
# SLAG: Scalable Language-Augmented Gaussian Splatting 

**Title (ZH)**: SLAG：可扩展的语言增强高斯渲染 

**Authors**: Laszlo Szilagyi, Francis Engelmann, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2505.08124)  

**Abstract**: Language-augmented scene representations hold great promise for large-scale robotics applications such as search-and-rescue, smart cities, and mining. Many of these scenarios are time-sensitive, requiring rapid scene encoding while also being data-intensive, necessitating scalable solutions. Deploying these representations on robots with limited computational resources further adds to the challenge. To address this, we introduce SLAG, a multi-GPU framework for language-augmented Gaussian splatting that enhances the speed and scalability of embedding large scenes. Our method integrates 2D visual-language model features into 3D scenes using SAM and CLIP. Unlike prior approaches, SLAG eliminates the need for a loss function to compute per-Gaussian language embeddings. Instead, it derives embeddings from 3D Gaussian scene parameters via a normalized weighted average, enabling highly parallelized scene encoding. Additionally, we introduce a vector database for efficient embedding storage and retrieval. Our experiments show that SLAG achieves an 18 times speedup in embedding computation on a 16-GPU setup compared to OpenGaussian, while preserving embedding quality on the ScanNet and LERF datasets. For more details, visit our project website: this https URL. 

**Abstract (ZH)**: 语言增强的场景表示在大规模机器人应用如搜救、智慧城市和矿业中充满潜力。许多这些场景对时间响应敏感，需要快速的场景编码，同时也数据密集，需要可扩展的解决方案。将这些表示部署在计算资源有限的机器人上进一步增加了挑战。为了解决这个问题，我们引入了SLAG，这是一种多GPU框架，用于语言增强的Gaussian splatting，增强了大规模场景嵌入的速度和可扩展性。我们的方法通过SAM和CLIP将2D视觉语言模型特征整合到3D场景中。与之前的 approaches 不同，SLAG 不需要计算每个高斯的损失函数来获取语言嵌入。相反，它通过归一化的加权平均从3D高斯场景参数中获取嵌入，从而实现高效的并行场景编码。此外，我们还引入了向量数据库，以实现高效的嵌入存储和检索。我们的实验表明，与OpenGaussian相比，在16-GPU设置下，SLAG 在嵌入计算上的加速比达到了18倍，同时在ScanNet和LERF数据集上保持了嵌入质量。更多信息，请访问我们的项目网站：this https URL。 

---
# JSover: Joint Spectrum Estimation and Multi-Material Decomposition from Single-Energy CT Projections 

**Title (ZH)**: JSover: 单能CT投影的联合频谱估计与多材料分解 

**Authors**: Qing Wu, Hongjiang Wei, Jingyi Yu, S. Kevin Zhou, Yuyao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08123)  

**Abstract**: Multi-material decomposition (MMD) enables quantitative reconstruction of tissue compositions in the human body, supporting a wide range of clinical applications. However, traditional MMD typically requires spectral CT scanners and pre-measured X-ray energy spectra, significantly limiting clinical applicability. To this end, various methods have been developed to perform MMD using conventional (i.e., single-energy, SE) CT systems, commonly referred to as SEMMD. Despite promising progress, most SEMMD methods follow a two-step image decomposition pipeline, which first reconstructs monochromatic CT images using algorithms such as FBP, and then performs decomposition on these images. The initial reconstruction step, however, neglects the energy-dependent attenuation of human tissues, introducing severe nonlinear beam hardening artifacts and noise into the subsequent decomposition. This paper proposes JSover, a fundamentally reformulated one-step SEMMD framework that jointly reconstructs multi-material compositions and estimates the energy spectrum directly from SECT projections. By explicitly incorporating physics-informed spectral priors into the SEMMD process, JSover accurately simulates a virtual spectral CT system from SE acquisitions, thereby improving the reliability and accuracy of decomposition. Furthermore, we introduce implicit neural representation (INR) as an unsupervised deep learning solver for representing the underlying material maps. The inductive bias of INR toward continuous image patterns constrains the solution space and further enhances estimation quality. Extensive experiments on both simulated and real CT datasets show that JSover outperforms state-of-the-art SEMMD methods in accuracy and computational efficiency. 

**Abstract (ZH)**: 基于单能量CT的多材料分解：JSover框架 

---
# Fréchet Power-Scenario Distance: A Metric for Evaluating Generative AI Models across Multiple Time-Scales in Smart Grids 

**Title (ZH)**: Frechet 动势场景距离：一种用于智能电网中多时间尺度评估生成式AI模型的度量标准 

**Authors**: Yuting Cai, Shaohuai Liu, Chao Tian, Le Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.08082)  

**Abstract**: Generative artificial intelligence (AI) models in smart grids have advanced significantly in recent years due to their ability to generate large amounts of synthetic data, which would otherwise be difficult to obtain in the real world due to confidentiality constraints. A key challenge in utilizing such synthetic data is how to assess the data quality produced from such generative models. Traditional Euclidean distance-based metrics only reflect pair-wise relations between two individual samples, and could fail in evaluating quality differences between groups of synthetic datasets. In this work, we propose a novel metric based on the Fréchet Distance (FD) estimated between two datasets in a learned feature space. The proposed method evaluates the quality of generation from a distributional perspective. Empirical results demonstrate the superiority of the proposed metric across timescales and models, enhancing the reliability of data-driven decision-making in smart grid operations. 

**Abstract (ZH)**: 智能电网中基于生成人工智能（AI）模型的合成数据质量评估方法：一种基于学习特征空间中Fréchet距离的新颖度量 

---
# Justified Evidence Collection for Argument-based AI Fairness Assurance 

**Title (ZH)**: 基于论证的AI公平性保障的正当证据收集 

**Authors**: Alpay Sabuncuoglu, Christopher Burr, Carsten Maple  

**Link**: [PDF](https://arxiv.org/pdf/2505.08064)  

**Abstract**: It is well recognised that ensuring fair AI systems is a complex sociotechnical challenge, which requires careful deliberation and continuous oversight across all stages of a system's lifecycle, from defining requirements to model deployment and deprovisioning. Dynamic argument-based assurance cases, which present structured arguments supported by evidence, have emerged as a systematic approach to evaluating and mitigating safety risks and hazards in AI-enabled system development and have also been extended to deal with broader normative goals such as fairness and explainability. This paper introduces a systems-engineering-driven framework, supported by software tooling, to operationalise a dynamic approach to argument-based assurance in two stages. In the first stage, during the requirements planning phase, a multi-disciplinary and multi-stakeholder team define goals and claims to be established (and evidenced) by conducting a comprehensive fairness governance process. In the second stage, a continuous monitoring interface gathers evidence from existing artefacts (e.g. metrics from automated tests), such as model, data, and use case documentation, to support these arguments dynamically. The framework's effectiveness is demonstrated through an illustrative case study in finance, with a focus on supporting fairness-related arguments. 

**Abstract (ZH)**: 确保公平的AI系统是一个复杂的社会技术挑战，需要在系统生命周期的所有阶段，从需求定义到模型部署和撤销，进行仔细的考虑和持续监督。基于动态论证的保证案例通过呈现结构化的论证和支持证据的论据，已成为评估和减轻AI使能系统开发中安全风险和危害的一种系统方法，并已扩展以处理公平性和可解释性等更广泛的规范目标。本文提出了一种由系统工程驱动并在软件工具支持下的框架，以实现动态论证保证的两阶段实现。在第一阶段的需求规划阶段，多学科和多利益相关者团队通过全面的公平治理过程定义和建立目标与证据。在第二阶段，持续监测界面从现有 artefacts（如自动化测试的指标）中收集模型、数据和用例文档等证据，以动态支持这些论证。该框架的有效性通过一个金融领域的示例研究得以体现，重点关注支持公平性相关的论证。 

---
# NAZM: Network Analysis of Zonal Metrics in Persian Poetic Tradition 

**Title (ZH)**: NAZM：波斯诗歌传统中的区域指标网络分析 

**Authors**: Kourosh Shahnazari, Seyed Moein Ayyoubzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.08052)  

**Abstract**: This study formalizes a computational model to simulate classical Persian poets' dynamics of influence through constructing a multi-dimensional similarity network. Using a rigorously curated dataset based on Ganjoor's corpus, we draw upon semantic, lexical, stylistic, thematic, and metrical features to demarcate each poet's corpus. Each is contained within weighted similarity matrices, which are then appended to generate an aggregate graph showing poet-to-poet influence. Further network investigation is carried out to identify key poets, style hubs, and bridging poets by calculating degree, closeness, betweenness, eigenvector, and Katz centrality measures. Further, for typological insight, we use the Louvain community detection algorithm to demarcate clusters of poets sharing both style and theme coherence, which correspond closely to acknowledged schools of literature like Sabk-e Hindi, Sabk-e Khorasani, and the Bazgasht-e Adabi phenomenon. Our findings provide a new data-driven view of Persian literature distinguished between canonical significance and interextual influence, thus highlighting relatively lesser-known figures who hold great structural significance. Combining computational linguistics with literary study, this paper produces an interpretable and scalable model for poetic tradition, enabling retrospective reflection as well as forward-looking research within digital humanities. 

**Abstract (ZH)**: 本研究通过构建多维相似性网络，形式化了一个计算模型来模拟古典波斯诗人之间的影响动态。基于GANJOOR语料库精心选择的数据集，我们利用语义、词汇、风格、主题和韵律特征来划定每位诗人的作品集。每位诗人的作品集包含在加权相似性矩阵中，然后合并生成一个显示诗人之间影响的综合图。进一步的网络分析通过计算度、接近中心性、介数中心性、特征向量中心性和Katz中心性来识别关键诗人、风格枢纽和桥梁诗人。此外，为了提供类型学上的洞察，我们使用Louvain社区检测算法来划定共享风格和主题一致性的诗人簇，这些簇与公认的文学流派如哈尼诗派、柯霍桑诗派和文学反刍现象相对应。我们的发现提供了一种基于数据的新视角，区分了经典意义和互文影响，从而突出了相对鲜为人知但在结构上具有重要意义的诗人。结合计算语言学与文学研究，本文构建了一个可解释且可扩展的诗歌传统的模型，有助于数字人文领域的回顾性反思及前瞻性研究。 

---
# Online Learning-based Adaptive Beam Switching for 6G Networks: Enhancing Efficiency and Resilience 

**Title (ZH)**: 基于在线学习的自适应波束切换技术在6G网络中的应用：提高效率与增强韧性 

**Authors**: Seyed Bagher Hashemi Natanzi, Zhicong Zhu, Bo Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08032)  

**Abstract**: Adaptive beam switching in 6G networks is challenged by high frequencies, mobility, and blockage. We propose an Online Learning framework using Deep Reinforcement Learning (DRL) with an enhanced state representation (velocity and blockage history), a GRU architecture, and prioritized experience replay for real-time beam optimization. Validated via Nvidia Sionna under time-correlated blockage, our approach significantly enhances resilience in SNR, throughput, and accuracy compared to a conventional heuristic. Furthermore, the enhanced DRL agent outperforms a reactive Multi-Armed Bandit (MAB) baseline by leveraging temporal dependencies, achieving lower performance variability. This demonstrates the benefits of memory and prioritized learning for robust 6G beam management, while confirming MAB as a strong baseline. 

**Abstract (ZH)**: 基于深度强化学习的自适应波束切换在6G网络中的实时波束优化 

---
# Fair Play for Individuals, Foul Play for Groups? Auditing Anonymization's Impact on ML Fairness 

**Title (ZH)**: 个体公平，群体不公？探究匿名化对机器学习公平性影响的审计 

**Authors**: Héber H. Arcolezi, Mina Alishahi, Adda-Akram Bendoukha, Nesrine Kaaniche  

**Link**: [PDF](https://arxiv.org/pdf/2505.07985)  

**Abstract**: Machine learning (ML) algorithms are heavily based on the availability of training data, which, depending on the domain, often includes sensitive information about data providers. This raises critical privacy concerns. Anonymization techniques have emerged as a practical solution to address these issues by generalizing features or suppressing data to make it more difficult to accurately identify individuals. Although recent studies have shown that privacy-enhancing technologies can influence ML predictions across different subgroups, thus affecting fair decision-making, the specific effects of anonymization techniques, such as $k$-anonymity, $\ell$-diversity, and $t$-closeness, on ML fairness remain largely unexplored. In this work, we systematically audit the impact of anonymization techniques on ML fairness, evaluating both individual and group fairness. Our quantitative study reveals that anonymization can degrade group fairness metrics by up to four orders of magnitude. Conversely, similarity-based individual fairness metrics tend to improve under stronger anonymization, largely as a result of increased input homogeneity. By analyzing varying levels of anonymization across diverse privacy settings and data distributions, this study provides critical insights into the trade-offs between privacy, fairness, and utility, offering actionable guidelines for responsible AI development. Our code is publicly available at: this https URL. 

**Abstract (ZH)**: 机器学习算法高度依赖训练数据的可用性，而这些数据在不同的领域往往包含数据提供者的敏感信息。这引发了重要的隐私担忧。匿名化技术已经出现作为一种实际的解决方案，通过泛化特征或抑制数据，使其更难准确识别个体。尽管近期研究显示，增强隐私的技术可以影响ML预测的不同子群体，从而影响公平决策，但匿名化技术，如$k$-匿名性、$\ell$-多样性、$t$-相近性，对ML公平性的影响仍然很大程度上未被探索。在本项工作中，我们系统地审计了匿名化技术对ML公平性的影响，评估了个体和群体公平性。我们的定量研究揭示匿名化可以使群体公平性指标下降四数量级。相反，在更强的匿名化下，基于相似性的个体公平性指标往往会改善，主要是由于输入同质性的增加。通过分析不同隐私设置和数据分布下不同水平的匿名化，本研究提供了关于隐私、公平性和效用之间权衡的重要见解，为负责任的AI发展提供了可操作的指导。我们的代码可在以下链接获取：this https URL。 

---
# Probabilistic approach to longitudinal response prediction: application to radiomics from brain cancer imaging 

**Title (ZH)**: 基于概率方法的纵向反应预测：应用于脑癌影像的放射组学 

**Authors**: Isabella Cama, Michele Piana, Cristina Campi, Sara Garbarino  

**Link**: [PDF](https://arxiv.org/pdf/2505.07973)  

**Abstract**: Longitudinal imaging analysis tracks disease progression and treatment response over time, providing dynamic insights into treatment efficacy and disease evolution. Radiomic features extracted from medical imaging can support the study of disease progression and facilitate longitudinal prediction of clinical outcomes. This study presents a probabilistic model for longitudinal response prediction, integrating baseline features with intermediate follow-ups. The probabilistic nature of the model naturally allows to handle the instrinsic uncertainty of the longitudinal prediction of disease progression. We evaluate the proposed model against state-of-the-art disease progression models in both a synthetic scenario and using a brain cancer dataset. Results demonstrate that the approach is competitive against existing methods while uniquely accounting for uncertainty and controlling the growth of problem dimensionality, eliminating the need for data from intermediate follow-ups. 

**Abstract (ZH)**: 纵向影像分析追踪疾病进展和治疗反应，提供治疗 efficacy 和疾病演变的动力学洞察。从医学影像中提取的 Radiomic 特征可支持疾病进展的研究，并促进临床结局的纵向预测。本研究提出了一种将基线特征与中期随访整合的概率模型，其概率性质自然地处理纵向预测疾病进展的固有不确定性。该研究在合成场景和脑癌数据集中分别与最先进的疾病进展模型进行评估，结果表明所提出的方法在与现有方法竞争的同时，单独考虑不确定性并控制问题维度的增长，从而消除对中期随访数据的需求。 

---
# Self-cross Feature based Spiking Neural Networks for Efficient Few-shot Learning 

**Title (ZH)**: 基于自交特征的Spiking神经网络高效 Few-shot 学习 

**Authors**: Qi Xu, Junyang Zhu, Dongdong Zhou, Hao Chen, Yang Liu, Jiangrong Shen, Qiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07921)  

**Abstract**: Deep neural networks (DNNs) excel in computer vision tasks, especially, few-shot learning (FSL), which is increasingly important for generalizing from limited examples. However, DNNs are computationally expensive with scalability issues in real world. Spiking Neural Networks (SNNs), with their event-driven nature and low energy consumption, are particularly efficient in processing sparse and dynamic data, though they still encounter difficulties in capturing complex spatiotemporal features and performing accurate cross-class comparisons. To further enhance the performance and efficiency of SNNs in few-shot learning, we propose a few-shot learning framework based on SNNs, which combines a self-feature extractor module and a cross-feature contrastive module to refine feature representation and reduce power consumption. We apply the combination of temporal efficient training loss and InfoNCE loss to optimize the temporal dynamics of spike trains and enhance the discriminative power. Experimental results show that the proposed FSL-SNN significantly improves the classification performance on the neuromorphic dataset N-Omniglot, and also achieves competitive performance to ANNs on static datasets such as CUB and miniImageNet with low power consumption. 

**Abstract (ZH)**: 基于SNN的少样本学习框架：结合自特征提取模块和跨特征对比模块以提高性能和效率 

---
# Re$^2$: A Consistency-ensured Dataset for Full-stage Peer Review and Multi-turn Rebuttal Discussions 

**Title (ZH)**: Re$^2$：一种确保一致性的数据集，用于全程同行评审和多轮反驳讨论 

**Authors**: Daoze Zhang, Zhijian Bao, Sihang Du, Zhiyi Zhao, Kuangling Zhang, Dezheng Bao, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07920)  

**Abstract**: Peer review is a critical component of scientific progress in the fields like AI, but the rapid increase in submission volume has strained the reviewing system, which inevitably leads to reviewer shortages and declines review quality. Besides the growing research popularity, another key factor in this overload is the repeated resubmission of substandard manuscripts, largely due to the lack of effective tools for authors to self-evaluate their work before submission. Large Language Models (LLMs) show great promise in assisting both authors and reviewers, and their performance is fundamentally limited by the quality of the peer review data. However, existing peer review datasets face three major limitations: (1) limited data diversity, (2) inconsistent and low-quality data due to the use of revised rather than initial submissions, and (3) insufficient support for tasks involving rebuttal and reviewer-author interactions. To address these challenges, we introduce the largest consistency-ensured peer review and rebuttal dataset named Re^2, which comprises 19,926 initial submissions, 70,668 review comments, and 53,818 rebuttals from 24 conferences and 21 workshops on OpenReview. Moreover, the rebuttal and discussion stage is framed as a multi-turn conversation paradigm to support both traditional static review tasks and dynamic interactive LLM assistants, providing more practical guidance for authors to refine their manuscripts and helping alleviate the growing review burden. Our data and code are available in this https URL. 

**Abstract (ZH)**: 大型语言模型在辅助作者和审稿人方面的潜力及其对同行评审数据质量的限制：引入Re^2数据集以应对同行评审挑战 

---
# Efficient and Reproducible Biomedical Question Answering using Retrieval Augmented Generation 

**Title (ZH)**: 基于检索增强生成的高效可再现生物医学问答 

**Authors**: Linus Stuhlmann, Michael Alexander Saxer, Jonathan Fürst  

**Link**: [PDF](https://arxiv.org/pdf/2505.07917)  

**Abstract**: Biomedical question-answering (QA) systems require effective retrieval and generation components to ensure accuracy, efficiency, and scalability. This study systematically examines a Retrieval-Augmented Generation (RAG) system for biomedical QA, evaluating retrieval strategies and response time trade-offs. We first assess state-of-the-art retrieval methods, including BM25, BioBERT, MedCPT, and a hybrid approach, alongside common data stores such as Elasticsearch, MongoDB, and FAISS, on a ~10% subset of PubMed (2.4M documents) to measure indexing efficiency, retrieval latency, and retriever performance in the end-to-end RAG system. Based on these insights, we deploy the final RAG system on the full 24M PubMed corpus, comparing different retrievers' impact on overall performance. Evaluations of the retrieval depth show that retrieving 50 documents with BM25 before reranking with MedCPT optimally balances accuracy (0.90), recall (0.90), and response time (1.91s). BM25 retrieval time remains stable (82ms), while MedCPT incurs the main computational cost. These results highlight previously not well-known trade-offs in retrieval depth, efficiency, and scalability for biomedical QA. With open-source code, the system is fully reproducible and extensible. 

**Abstract (ZH)**: 生物医学问答(QA)系统需要有效的检索和生成组件以确保准确、高效和可扩展。本研究系统地探讨了生物医学QA的检索增强生成(RAG)系统，评估了检索策略和响应时间的权衡。我们首先在PubMed大约10%的子集（240万文档）上评估了最先进的检索方法，包括BM25、BioBERT、MedCPT及混合方法，以及常用的索引存储方式如Elasticsearch、MongoDB和FAISS，以度量端到端RAG系统的索引效率、检索延迟和检索器性能。基于这些见解，我们在整个2400万PubMed语料库上部署了最终的RAG系统，比较了不同检索器对总体性能的影响。检索深度的评估表明，在使用BM25检索50份文档后再排序（二次排序）使用MedCPT可最优地平衡准确率（0.90）、召回率（0.90）和响应时间（1.91秒）。BM25的检索时间为稳定值（82毫秒），而MedCPT主要承担计算成本。这些结果突显了生物医学QA中检索深度、效率和可扩展性之间的前所未知的权衡。该系统具有开源代码，确保完全可再现性和扩展性。 

---
# Tuning for Trustworthiness -- Balancing Performance and Explanation Consistency in Neural Network Optimization 

**Title (ZH)**: 调谐以提升可信度——在神经网络优化中平衡性能与解释一致性 

**Authors**: Alexander Hinterleitner, Thomas Bartz-Beielstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.07910)  

**Abstract**: Despite the growing interest in Explainable Artificial Intelligence (XAI), explainability is rarely considered during hyperparameter tuning or neural architecture optimization, where the focus remains primarily on minimizing predictive loss. In this work, we introduce the novel concept of XAI consistency, defined as the agreement among different feature attribution methods, and propose new metrics to quantify it. For the first time, we integrate XAI consistency directly into the hyperparameter tuning objective, creating a multi-objective optimization framework that balances predictive performance with explanation robustness. Implemented within the Sequential Parameter Optimization Toolbox (SPOT), our approach uses both weighted aggregation and desirability-based strategies to guide model selection. Through our proposed framework and supporting tools, we explore the impact of incorporating XAI consistency into the optimization process. This enables us to characterize distinct regions in the architecture configuration space: one region with poor performance and comparatively low interpretability, another with strong predictive performance but weak interpretability due to low \gls{xai} consistency, and a trade-off region that balances both objectives by offering high interpretability alongside competitive performance. Beyond introducing this novel approach, our research provides a foundation for future investigations into whether models from the trade-off zone-balancing performance loss and XAI consistency-exhibit greater robustness by avoiding overfitting to training performance, thereby leading to more reliable predictions on out-of-distribution data. 

**Abstract (ZH)**: 尽管可解释的人工智能（XAI）日益受到关注，但在超参数调整和神经网络架构优化过程中，解释性 rarely 考虑，研究重点仍主要集中在最小化预测损失上。本文引入了新颖的 XAI 一致性概念，定义为不同特征归因方法的一致性，并提出新的度量方法来量化它。首次将 XAI 一致性直接纳入超参数调整目标，创建了兼顾预测性能与解释鲁棒性的多目标优化框架。在 Sequential Parameter Optimization Toolbox (SPOT) 中实现该方法，通过加权聚合和偏好策略指导模型选择。通过提出的框架及其支持工具，我们探讨了将 XAI 一致性纳入优化过程的影响。这使我们能够区分架构配置空间中的不同区域：一个区域具有较差的性能和较低的可解释性，另一个区域具有较强预测性能但因较低的 XAI 一致性而缺乏可解释性，还有一个权衡区域通过提供高可解释性与竞争力的性能，同时平衡两个目标。除了提出这一新颖方法外，我们的研究为未来探讨性能损失和 XAI 一致性之间的权衡区域中的模型是否因避免训练性能过拟合而表现出更大的鲁棒性，从而在离群数据上产生更可靠的预测奠定了基础。 

---
# A Reproduction Study: The Kernel PCA Interpretation of Self-Attention Fails Under Scrutiny 

**Title (ZH)**: 一项再现研究：在详细审查下，自我注意力的核PCA解释失败 

**Authors**: Karahan Sarıtaş, Çağatay Yıldız  

**Link**: [PDF](https://arxiv.org/pdf/2505.07908)  

**Abstract**: In this reproduction study, we revisit recent claims that self-attention implements kernel principal component analysis (KPCA) (Teo et al., 2024), positing that (i) value vectors $V$ capture the eigenvectors of the Gram matrix of the keys, and (ii) that self-attention projects queries onto the principal component axes of the key matrix $K$ in a feature space. Our analysis reveals three critical inconsistencies: (1) No alignment exists between learned self-attention value vectors and what is proposed in the KPCA perspective, with average similarity metrics (optimal cosine similarity $\leq 0.32$, linear CKA (Centered Kernel Alignment) $\leq 0.11$, kernel CKA $\leq 0.32$) indicating negligible correspondence; (2) Reported decreases in reconstruction loss $J_\text{proj}$, arguably justifying the claim that the self-attention minimizes the projection error of KPCA, are misinterpreted, as the quantities involved differ by orders of magnitude ($\sim\!10^3$); (3) Gram matrix eigenvalue statistics, introduced to justify that $V$ captures the eigenvector of the gram matrix, are irreproducible without undocumented implementation-specific adjustments. Across 10 transformer architectures, we conclude that the KPCA interpretation of self-attention lacks empirical support. 

**Abstract (ZH)**: 在本重现研究中，我们重新审视了近期关于自注意力机制实现核主成分分析（KPCA）的主张（Teo等，2024），提出（i）值向量$V$捕获键的Gram矩阵的特征向量，和（ii）自注意力将查询投影到键矩阵$K$在特征空间的主要成分轴上。我们的分析揭示了三个关键不一致：（1）学习到的自注意力值向量与KPCA视角下提出的向量之间不存在对齐，平均相似度指标（最优余弦相似度$\leq 0.32$，线性CKA（中心核对齐）$\leq 0.11$，核CKA$\leq 0.32$）显示几乎没有对应关系；（2）所报告的重构损失$J_\text{proj}$的减少被误解为自注意力最小化KPCA投影误差，因为参与的量级相差三个数量级；（3）用于证明$V$捕获Gram矩阵特征向量的Gram矩阵特征值统计，在没有未记录的实现特定调整的情况下无法重现。在10种变压器架构中，我们得出结论，自注意力的KPCA解释缺乏实证支持。 

---
# Latent Behavior Diffusion for Sequential Reaction Generation in Dyadic Setting 

**Title (ZH)**: 双边设置中序列反应生成的潜在行为扩散 

**Authors**: Minh-Duc Nguyen, Hyung-Jeong Yang, Soo-Hyung Kim, Ji-Eun Shin, Seung-Won Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.07901)  

**Abstract**: The dyadic reaction generation task involves synthesizing responsive facial reactions that align closely with the behaviors of a conversational partner, enhancing the naturalness and effectiveness of human-like interaction simulations. This paper introduces a novel approach, the Latent Behavior Diffusion Model, comprising a context-aware autoencoder and a diffusion-based conditional generator that addresses the challenge of generating diverse and contextually relevant facial reactions from input speaker behaviors. The autoencoder compresses high-dimensional input features, capturing dynamic patterns in listener reactions while condensing complex input data into a concise latent representation, facilitating more expressive and contextually appropriate reaction synthesis. The diffusion-based conditional generator operates on the latent space generated by the autoencoder to predict realistic facial reactions in a non-autoregressive manner. This approach allows for generating diverse facial reactions that reflect subtle variations in conversational cues and emotional states. Experimental results demonstrate the effectiveness of our approach in achieving superior performance in dyadic reaction synthesis tasks compared to existing methods. 

**Abstract (ZH)**: 二元反应生成任务涉及合成与对话伙伴行为紧密对齐的响应面部表情，以增强人类交互模拟的自然性和有效性。本文提出了一种新颖的方法——潜在行为扩散模型，该模型包含一个情境感知自编码器和一种基于扩散的条件生成器，以解决从输入说话者行为生成多样且情境相关面部反应的挑战。自编码器压缩高维输入特征，捕捉听众反应中的动态模式，同时将复杂输入数据凝聚成简洁的潜在表示，从而促进更具表现力和情境恰当的反应合成。基于扩散的条件生成器在自编码器生成的潜在空间上以非自回归方式预测现实的面部反应。这种方法允许生成反映对话提示和情感状态微妙变化的多样面部反应。实验结果证明，与现有方法相比，该方法在二元反应生成任务中实现了更优的表现。 

---
# Getting Ready for the EU AI Act in Healthcare. A call for Sustainable AI Development and Deployment 

**Title (ZH)**: 为应对欧盟AI法案在医疗健康领域的实施：呼吁可持续AI开发与部署 

**Authors**: John Brandt Brodersen, Ilaria Amelia Caggiano, Pedro Kringen, Vince Istvan Madai, Walter Osika, Giovanni Sartor, Ellen Svensson, Magnus Westerlund, Roberto V. Zicari  

**Link**: [PDF](https://arxiv.org/pdf/2505.07875)  

**Abstract**: Assessments of trustworthiness have become a cornerstone of responsible AI development. Especially in high-stakes fields like healthcare, aligning technical, evidence-based, and ethical practices with forthcoming legal requirements is increasingly urgent. We argue that developers and deployers of AI systems for the medical domain should be proactive and take steps to progressively ensure that such systems, both those currently in use and those being developed or planned, respect the requirements of the AI Act, which has come into force in August 2024. This is necessary if full and effective compliance is to be ensured when the most relevant provisions of the Act become effective (August 2026). The engagement with the AI Act cannot be viewed as a formalistic exercise. Compliance with the AI Act needs to be carried out through the proactive commitment to the ethical principles of trustworthy AI. These principles provide the background for the Act, which mentions them several times and connects them to the protection of public interest. They can be used to interpret and apply the Act's provisions and to identify good practices, increasing the validity and sustainability of AI systems over time. 

**Abstract (ZH)**: 负责任人工智能发展中对可信性的评估已成为基石。特别是在像医疗这样高风险的领域，技术、基于证据的方法和伦理实践与即将出台的法律要求保持一致变得越来越紧迫。我们认为，为了确保在2024年8月生效的《人工智能法案》的相关规定于2026年8月生效时全面且有效地遵守，医疗领域的人工智能系统开发者和部署者应当积极行动，并逐步确保当前使用及正在开发或计划中的系统遵守该法案的要求。与《人工智能法案》的互动绝不能被视为一种形式上的努力。遵守《人工智能法案》需要通过积极承诺可信人工智能的伦理原则来实现。这些原则构成了法案的背景，法案多次提及这些原则，并将它们与公共利益的保护联系起来。它们可以用来解释和应用法案的各项规定，并识别良好的实践做法，从而随着时间的推移增强和维持人工智能系统的有效性与可持续性。 

---
# Unpacking Robustness in Inflectional Languages: Adversarial Evaluation and Mechanistic Insights 

**Title (ZH)**: 剖析屈折语中的稳健性：对抗性评估与机制性洞察 

**Authors**: Paweł Walkowiak, Marek Klonowski, Marcin Oleksy, Arkadiusz Janz  

**Link**: [PDF](https://arxiv.org/pdf/2505.07856)  

**Abstract**: Various techniques are used in the generation of adversarial examples, including methods such as TextBugger which introduce minor, hardly visible perturbations to words leading to changes in model behaviour. Another class of techniques involves substituting words with their synonyms in a way that preserves the text's meaning but alters its predicted class, with TextFooler being a prominent example of such attacks. Most adversarial example generation methods are developed and evaluated primarily on non-inflectional languages, typically English. In this work, we evaluate and explain how adversarial attacks perform in inflectional languages. To explain the impact of inflection on model behaviour and its robustness under attack, we designed a novel protocol inspired by mechanistic interpretability, based on Edge Attribution Patching (EAP) method. The proposed evaluation protocol relies on parallel task-specific corpora that include both inflected and syncretic variants of texts in two languages -- Polish and English. To analyse the models and explain the relationship between inflection and adversarial robustness, we create a new benchmark based on task-oriented dataset MultiEmo, enabling the identification of mechanistic inflection-related elements of circuits within the model and analyse their behaviour under attack. 

**Abstract (ZH)**: 各种技术被用于生成对抗样本，包括向文本中引入细微、几乎不可见的扰动以改变模型行为的TextBugger方法。另一类技术涉及用同义词替换单词，以保持文本意义但改变其预测类别，TextFooler是此类攻击的一个典型案例。大多数生成对抗样本的方法主要在非屈折语，通常是英语上进行开发和评估。在本研究中，我们评估并解释了对抗攻击在屈折语上的表现。为了解释屈折变化对模型行为的影响及其在攻击下的鲁棒性，我们设计了一种基于Edge Attribution Patching (EAP) 方法的新协议，该协议借鉴了机制可解释性的理念。所提出的评估协议依赖于双语言（波兰语和英语）平行的专用语料库，该语料库包含文本的屈折变化和合形变化形式。为了分析模型并解释屈折与对抗鲁棒性之间的关系，我们基于面向任务的数据集MultiEmo创建了一个新的基准，该基准能够识别模型内部与屈折相关的基本机制元素，并分析其在攻击下的行为。 

---
# SweRank: Software Issue Localization with Code Ranking 

**Title (ZH)**: SweRank: 代码排名驱动的软件问题定位 

**Authors**: Revanth Gangi Reddy, Tarun Suresh, JaeHyeok Doo, Ye Liu, Xuan Phi Nguyen, Yingbo Zhou, Semih Yavuz, Caiming Xiong, Heng Ji, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2505.07849)  

**Abstract**: Software issue localization, the task of identifying the precise code locations (files, classes, or functions) relevant to a natural language issue description (e.g., bug report, feature request), is a critical yet time-consuming aspect of software development. While recent LLM-based agentic approaches demonstrate promise, they often incur significant latency and cost due to complex multi-step reasoning and relying on closed-source LLMs. Alternatively, traditional code ranking models, typically optimized for query-to-code or code-to-code retrieval, struggle with the verbose and failure-descriptive nature of issue localization queries. To bridge this gap, we introduce SweRank, an efficient and effective retrieve-and-rerank framework for software issue localization. To facilitate training, we construct SweLoc, a large-scale dataset curated from public GitHub repositories, featuring real-world issue descriptions paired with corresponding code modifications. Empirical results on SWE-Bench-Lite and LocBench show that SweRank achieves state-of-the-art performance, outperforming both prior ranking models and costly agent-based systems using closed-source LLMs like Claude-3.5. Further, we demonstrate SweLoc's utility in enhancing various existing retriever and reranker models for issue localization, establishing the dataset as a valuable resource for the community. 

**Abstract (ZH)**: 软件问题定位是指识别与自然语言问题描述（如 bug 报告、功能请求）相关的精确代码位置（文件、类或函数）的任务，是软件开发中至关重要但耗费大量时间的方面。尽管基于大语言模型的代理方法显示出潜力，但由于复杂的多步推理和依赖于闭源的大语言模型，它们常常会带来显著的延迟和成本。相反，传统的代码排名模型，通常优化用于查询到代码或代码到代码的检索，难以应对涉及详细故障描述的问题定位查询。为了弥合这一差距，我们提出了 SweRank，这是一种高效的检索和重排序框架，用于软件问题定位。为了便于训练，我们从公共 GitHub 存储库中构建了 SweLoc，该数据集包含真实世界的 Issue 描述及其对应的代码修改配对。在 SWE-Bench-Lite 和 LocBench 上的实验证明，SweRank 达到了最先进的性能，优于先前的排名模型以及使用闭源大语言模型（如 Claude-3.5）的成本高昂的代理系统。此外，我们展示了 SweLoc 在增强各种现有的检索和重排序模型以进行问题定位方面的实用性，确立了该数据集作为社区中的宝贵资源的地位。 

---
# Moving From Monolithic To Microservices Architecture for Multi-Agent Systems 

**Title (ZH)**: 从单体架构向微服务架构迁移以应用于多代理系统 

**Authors**: Muskaan Goyal, Pranav Bhasin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07838)  

**Abstract**: The transition from monolithic to microservices architecture revolutionized software development by improving scalability and maintainability. This paradigm shift is now becoming relevant for complex multi-agent systems (MAS). This review article explores the evolution from monolithic architecture to microservices architecture in the specific context of MAS. It will highlight the limitations of traditional monolithic MAS and the benefits of adopting a microservices-based approach. The article further examines the core architectural principles and communication protocols, including Agent Communication Languages (ACLs), the Model Context Protocol (MCP), and the Application-to-Application (A2A) protocol. The article identifies emerging architectural patterns, design challenges, and considerations through a comparative lens of the paradigm shift. 

**Abstract (ZH)**: 从-monolithic-到-microservices-架构的过渡 revolutionized 软件开发，通过提高可扩展性和可维护性。这一范式转变现在对于复杂的多代理系统（MAS）变得 relevant。本文综述了从-monolithic-架构到-microservices-架构在特定的 MAS 上的演变。它将突出传统-monolithic-MAS 的局限性以及采用基于-microservices-的方法的好处。文章还将进一步探讨核心架构原则和通信协议，包括代理通信语言（ACL）、模型上下文协议（MCP）和应用程序到应用程序（A2A）协议。通过范式转变的比较视角，文章指出了新兴的架构模式、设计挑战和考虑因素。 

---
# Intelligent Product 3.0: Decentralised AI Agents and Web3 Intelligence Standards 

**Title (ZH)**: 智能产品3.0：去中心化AI代理和Web3智能标准 

**Authors**: Alex C. Y. Wong, Duncan McFarlane, C. Ellarby, M. Lee, M. Kuok  

**Link**: [PDF](https://arxiv.org/pdf/2505.07835)  

**Abstract**: Twenty-five years ago, the specification of the Intelligent Product was established, envisaging real-time connectivity that not only enables products to gather accurate data about themselves but also allows them to assess and influence their own destiny. Early work by the Auto-ID project focused on creating a single, open-standard repository for storing and retrieving product information, laying a foundation for scalable connectivity. A decade later, the approach was revisited in light of low-cost RFID systems that promised a low-cost link between physical goods and networked information environments. Since then, advances in blockchain, Web3, and artificial intelligence have introduced unprecedented levels of resilience, consensus, and autonomy. By leveraging decentralised identity, blockchain-based product information and history, and intelligent AI-to-AI collaboration, this paper examines these developments and outlines a new specification for the Intelligent Product 3.0, illustrating how decentralised and AI-driven capabilities facilitate seamless interaction between physical AI and everyday products. 

**Abstract (ZH)**: 二十五年前，设立了智能产品的规范，设想了实时连接，不仅使产品能够收集自身准确的数据，还能评估和影响自身的命运。Auto-ID项目早期的工作关注于创建一个单一的、开放标准的存储库，用于存储和检索产品信息，为可扩展的连接奠定了基础。十年后，随着低成本RFID系统的出现，这一方法根据低成本连接物理商品与网络信息环境的前景进行了重新审视。此后，区块链、Web3和人工智能的进步引入了前所未有的韧性和共识水平。通过利用去中心化身份、基于区块链的产品信息和历史记录，以及智能AI-to-AI协作，本文探讨了这些发展，并提出了智能产品3.0的新规范，展示了去中心化和AI驱动的能力如何使物理AI和日常产品之间的无缝互动成为可能。 

---
# ai.txt: A Domain-Specific Language for Guiding AI Interactions with the Internet 

**Title (ZH)**: 专用于指导AI与互联网交互的领域特定语言 

**Authors**: Yuekang Li, Wei Song, Bangshuo Zhu, Dong Gong, Yi Liu, Gelei Deng, Chunyang Chen, Lei Ma, Jun Sun, Toby Walsh, Jingling Xue  

**Link**: [PDF](https://arxiv.org/pdf/2505.07834)  

**Abstract**: We introduce this http URL, a novel domain-specific language (DSL) designed to explicitly regulate interactions between AI models, agents, and web content, addressing critical limitations of the widely adopted this http URL standard. As AI increasingly engages with online materials for tasks such as training, summarization, and content modification, existing regulatory methods lack the necessary granularity and semantic expressiveness to ensure ethical and legal compliance. this http URL extends traditional URL-based access controls by enabling precise element-level regulations and incorporating natural language instructions interpretable by AI systems. To facilitate practical deployment, we provide an integrated development environment with code autocompletion and automatic XML generation. Furthermore, we propose two compliance mechanisms: XML-based programmatic enforcement and natural language prompt integration, and demonstrate their effectiveness through preliminary experiments and case studies. Our approach aims to aid the governance of AI-Internet interactions, promoting responsible AI use in digital ecosystems. 

**Abstract (ZH)**: 我们介绍这款新的域名特定语言（DSL）：this http URL，这是一种 novel domain-specific language (DSL) 专门设计用于明确调控 AI 模型、代理与网页内容之间的交互，解决广泛采用的 this http URL 标准的关键局限性。随着 AI 在诸如训练、摘要和内容修改等任务中越来越多地与在线材料互动，现有的监管方法缺乏必要的粒度和语义表达能力，以确保伦理和法律合规。this http URL 通过启用精确的元素级监管并结合可由 AI 系统解析的自然语言指令，扩展了传统的基于 URL 的访问控制。为了便于实际部署，我们提供了一个集成了代码自动补全和自动 XML 生成的集成开发环境。此外，我们提出了两种合规机制：基于 XML 的程序化执行和自然语言提示集成，并通过初步实验和案例研究展示了它们的有效性。我们这种方法旨在帮助治理 AI-互联网交互，促进数字生态系统中负责任的 AI 使用。 

---
# A General Approach of Automated Environment Design for Learning the Optimal Power Flow 

**Title (ZH)**: 一种自动环境设计的一般方法以学习最优功率流 

**Authors**: Thomas Wolgast, Astrid Nieße  

**Link**: [PDF](https://arxiv.org/pdf/2505.07832)  

**Abstract**: Reinforcement learning (RL) algorithms are increasingly used to solve the optimal power flow (OPF) problem. Yet, the question of how to design RL environments to maximize training performance remains unanswered, both for the OPF and the general case. We propose a general approach for automated RL environment design by utilizing multi-objective optimization. For that, we use the hyperparameter optimization (HPO) framework, which allows the reuse of existing HPO algorithms and methods. On five OPF benchmark problems, we demonstrate that our automated design approach consistently outperforms a manually created baseline environment design. Further, we use statistical analyses to determine which environment design decisions are especially important for performance, resulting in multiple novel insights on how RL-OPF environments should be designed. Finally, we discuss the risk of overfitting the environment to the utilized RL algorithm. To the best of our knowledge, this is the first general approach for automated RL environment design. 

**Abstract (ZH)**: 利用多目标优化的自动强化学习环境设计方法 

---
# Polysemy of Synthetic Neurons Towards a New Type of Explanatory Categorical Vector Spaces 

**Title (ZH)**: 合成神经元的多义性 toward 新型解释性分类向量空间 

**Authors**: Michael Pichat, William Pogrund, Paloma Pichat, Judicael Poumay, Armanouche Gasparian, Samuel Demarchi, Martin Corbet, Alois Georgeon, Michael Veillet-Guillem  

**Link**: [PDF](https://arxiv.org/pdf/2505.07831)  

**Abstract**: The polysemantic nature of synthetic neurons in artificial intelligence language models is currently understood as the result of a necessary superposition of distributed features within the latent space. We propose an alternative approach, geometrically defining a neuron in layer n as a categorical vector space with a non-orthogonal basis, composed of categorical sub-dimensions extracted from preceding neurons in layer n-1. This categorical vector space is structured by the activation space of each neuron and enables, via an intra-neuronal attention process, the identification and utilization of a critical categorical zone for the efficiency of the language model - more homogeneous and located at the intersection of these different categorical sub-dimensions. 

**Abstract (ZH)**: 合成神经元在人工智能语言模型中的多义性目前被认为是由潜在空间中分布特征的必然叠加所致。我们提出一种替代方法，几何上定义第n层的神经元为由第n-1层前馈神经元提取的非正交基组成的类别向量空间。该类别向量空间通过每个神经元的激活空间结构化，并通过一种内在神经元注意过程，识别和利用一个关键类别区域，以提高语言模型的效率——更加 homogenous，并位于这些不同类别子维度的交点处。 

---
# Blockbuster, Part 1: Block-level AI Operator Fusion 

**Title (ZH)**: Blockbuster, Part 1: 块级AI操作融合 

**Authors**: Ofer Dekel  

**Link**: [PDF](https://arxiv.org/pdf/2505.07829)  

**Abstract**: Blockbuster is a framework for AI operator fusion in inference programs. The Blockbuster framework is compatible with any multiprocessor architecture that has a tiered memory hierarchy, including GPUs, multi-core CPUs, and some AI accelerator chips. It includes a graph-based representation for AI workloads, called a block program, which explicitly models how blocks of data move between the memory tiers. It also includes an operator fusion procedure, which is made up of a candidate selection algorithm and a fusion algorithm that fuses each individual candidate - this two-algorithm structure makes Blockbuster especially suitable for large AI programs. The current paper focuses on the fusion algorithm, which is a rule-based technique. While the literature is full of previous rule-based fusion algorithms, what sets our algorithm apart is its direct modeling of data movement between memory tiers, resulting in uniquely powerful fusion results. As a first sanity check, we demonstrate how our algorithm automatically rediscovers the well-known Flash Attention kernel. Then, we demonstrate the real power of our approach by fusing LayerNorm with matrix multiplication and RMSNorm with FNN-SwiGLU - the latter involves fusing three matrix multiplications, a Hadamard product, a reduction, and a few elementwise operations into a single mega-kernel. 

**Abstract (ZH)**: Blockbuster是一种用于推理程序中AI操作融合的框架 

---
# AI-Based Crypto Tokens: The Illusion of Decentralized AI? 

**Title (ZH)**: 基于AI的加密代币：去中心化AI的幻觉？ 

**Authors**: Rischan Mafrur  

**Link**: [PDF](https://arxiv.org/pdf/2505.07828)  

**Abstract**: The convergence of blockchain and artificial intelligence (AI) has led to the emergence of AI-based tokens, which are cryptographic assets designed to power decentralized AI platforms and services. This paper provides a comprehensive review of leading AI-token projects, examining their technical architectures, token utilities, consensus mechanisms, and underlying business models. We explore how these tokens operate across various blockchain ecosystems and assess the extent to which they offer value beyond traditional centralized AI services. Based on this assessment, our analysis identifies several core limitations. From a technical perspective, many platforms depend extensively on off-chain computation, exhibit limited capabilities for on-chain intelligence, and encounter significant scalability challenges. From a business perspective, many models appear to replicate centralized AI service structures, simply adding token-based payment and governance layers without delivering truly novel value. In light of these challenges, we also examine emerging developments that may shape the next phase of decentralized AI systems. These include approaches for on-chain verification of AI outputs, blockchain-enabled federated learning, and more robust incentive frameworks. Collectively, while emerging innovations offer pathways to strengthen decentralized AI ecosystems, significant gaps remain between the promises and the realities of current AI-token implementations. Our findings contribute to a growing body of research at the intersection of AI and blockchain, highlighting the need for critical evaluation and more grounded approaches as the field continues to evolve. 

**Abstract (ZH)**: 区块链与人工智能的融合及其基于AI的代币：技术架构、应用价值与商业模型综述 

---
# Explainable Artificial Intelligence Techniques for Software Development Lifecycle: A Phase-specific Survey 

**Title (ZH)**: 解释性人工智能技术在软件开发生命周期中的应用：一种阶段特异性综述 

**Authors**: Lakshit Arora, Sanjay Surendranath Girija, Shashank Kapoor, Aman Raj, Dipen Pradhan, Ankit Shetgaonkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.07058)  

**Abstract**: Artificial Intelligence (AI) is rapidly expanding and integrating more into daily life to automate tasks, guide decision making, and enhance efficiency. However, complex AI models, which make decisions without providing clear explanations (known as the "black-box problem"), currently restrict trust and widespread adoption of AI. Explainable Artificial Intelligence (XAI) has emerged to address the black-box problem of making AI systems more interpretable and transparent so stakeholders can trust, verify, and act upon AI-based outcomes. Researchers have developed various techniques to foster XAI in the Software Development Lifecycle. However, there are gaps in applying XAI techniques in the Software Engineering phases. Literature review shows that 68% of XAI in Software Engineering research is focused on maintenance as opposed to 8% on software management and requirements. In this paper, we present a comprehensive survey of the applications of XAI methods such as concept-based explanations, Local Interpretable Model-agnostic Explanations (LIME), SHapley Additive exPlanations (SHAP), rule extraction, attention mechanisms, counterfactual explanations, and example-based explanations to the different phases of the Software Development Life Cycle (SDLC), including requirements elicitation, design and development, testing and deployment, and evolution. To the best of our knowledge, this paper presents the first comprehensive survey of XAI techniques for every phase of the Software Development Life Cycle (SDLC). This survey aims to promote explainable AI in Software Engineering and facilitate the practical application of complex AI models in AI-driven software development. 

**Abstract (ZH)**: 人工智能（AI）正迅速扩展并更多地集成到日常生活中以自动化任务、指导决策并提高效率。然而，缺乏清晰解释的复杂AI模型（ known as the “黑箱问题”）当前限制了人们对其的信任和广泛应用。可解释的人工智能（XAI）已经出现，旨在使AI系统更加可解释和透明，从而使利益相关者能够信任、验证并基于AI结果采取行动。研究者已经开发了各种技术来促进XAI在整个软件开发生命周期（SDLC）中的应用。然而，在软件工程（Software Engineering, SE）阶段应用XAI技术仍存在差距。文献综述显示，68%的XAI在软件工程研究中侧重于维护，相比之下，只有8%侧重于软件管理和需求分析。在本论文中，我们全面概述了诸如概念解释、局部可解释通用模型解释（LIME）、SHapley值归属解释（SHAP）、规则提取、注意力机制、反事实解释和基于示例的解释等XAI方法在软件开发生命周期不同阶段的应用，包括需求获取、设计与开发、测试与部署以及演化。据我们所知，这是首次对软件开发生命周期（SDLC）每个阶段的XAI技术进行全面调研。本研究旨在促进软件工程中的可解释AI，并推动复杂AI模型在人工智能驱动软件开发中的实际应用。 

---
