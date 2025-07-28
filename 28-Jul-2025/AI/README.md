# Hierarchical Deep Reinforcement Learning Framework for Multi-Year Asset Management Under Budget Constraints 

**Title (ZH)**: 预算约束下的多年度资产管理工作层级深度强化学习框架 

**Authors**: Amir Fard, Arnold X.-X. Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.19458)  

**Abstract**: Budget planning and maintenance optimization are crucial for infrastructure asset management, ensuring cost-effectiveness and sustainability. However, the complexity arising from combinatorial action spaces, diverse asset deterioration, stringent budget constraints, and environmental uncertainty significantly limits existing methods' scalability. This paper proposes a Hierarchical Deep Reinforcement Learning methodology specifically tailored to multi-year infrastructure planning. Our approach decomposes the problem into two hierarchical levels: a high-level Budget Planner allocating annual budgets within explicit feasibility bounds, and a low-level Maintenance Planner prioritizing assets within the allocated budget. By structurally separating macro-budget decisions from asset-level prioritization and integrating linear programming projection within a hierarchical Soft Actor-Critic framework, the method efficiently addresses exponential growth in the action space and ensures rigorous budget compliance. A case study evaluating sewer networks of varying sizes (10, 15, and 20 sewersheds) illustrates the effectiveness of the proposed approach. Compared to conventional Deep Q-Learning and enhanced genetic algorithms, our methodology converges more rapidly, scales effectively, and consistently delivers near-optimal solutions even as network size grows. 

**Abstract (ZH)**: 基于多年度基础设施规划的分层深度强化学习方法 

---
# Learning neuro-symbolic convergent term rewriting systems 

**Title (ZH)**: 学习神经符号收敛术语重写系统 

**Authors**: Flavio Petruzzellis, Alberto Testolin, Alessandro Sperduti  

**Link**: [PDF](https://arxiv.org/pdf/2507.19372)  

**Abstract**: Building neural systems that can learn to execute symbolic algorithms is a challenging open problem in artificial intelligence, especially when aiming for strong generalization and out-of-distribution performance. In this work, we introduce a general framework for learning convergent term rewriting systems using a neuro-symbolic architecture inspired by the rewriting algorithm itself. We present two modular implementations of such architecture: the Neural Rewriting System (NRS) and the Fast Neural Rewriting System (FastNRS). As a result of algorithmic-inspired design and key architectural elements, both models can generalize to out-of-distribution instances, with FastNRS offering significant improvements in terms of memory efficiency, training speed, and inference time. We evaluate both architectures on four tasks involving the simplification of mathematical formulas and further demonstrate their versatility in a multi-domain learning scenario, where a single model is trained to solve multiple types of problems simultaneously. The proposed system significantly outperforms two strong neural baselines: the Neural Data Router, a recent transformer variant specifically designed to solve algorithmic problems, and GPT-4o, one of the most powerful general-purpose large-language models. Moreover, our system matches or outperforms the latest o1-preview model from OpenAI that excels in reasoning benchmarks. 

**Abstract (ZH)**: 构建能够学习执行符号算法的神经系统是人工智能中的一个开放性挑战问题，特别是在追求强泛化能力和分布外性能时。在本文中，我们介绍了一种基于算法启发式设计的通用框架，用于使用神经符号架构学习收敛的术语重写系统。我们提出了两种此类架构的模块化实现：神经重写系统（NRS）和快速神经重写系统（FastNRS）。由于算法启发式设计和关键架构元素，两种模型都可以泛化到分布外实例，其中FastNRS在内存效率、训练速度和推理时间方面表现出显著改进。我们在四个涉及数学公式简化任务上评估了这两种架构，并进一步展示了它们在跨域学习场景中的灵活性，其中单个模型同时训练解决多种类型的问题。所提出系统显著优于两种强大的神经基线：神经数据路由器（Neural Data Router），这是一种最近为解决算法问题设计的变体变体变压器，以及GPT-4o，这是最强大的通用大型语言模型之一。此外，我们的系统能够匹配或优于OpenAI的最新o1-preview模型，在推理基准测试中表现出色。 

---
# Integrating LLM in Agent-Based Social Simulation: Opportunities and Challenges 

**Title (ZH)**: 基于代理的社会仿真中大型语言模型的整合：机遇与挑战 

**Authors**: Patrick Taillandier, Jean Daniel Zucker, Arnaud Grignard, Benoit Gaudou, Nghi Quang Huynh, Alexis Drogoul  

**Link**: [PDF](https://arxiv.org/pdf/2507.19364)  

**Abstract**: This position paper examines the use of Large Language Models (LLMs) in social simulation, analyzing both their potential and their limitations from a computational social science perspective. The first part reviews recent findings on the ability of LLMs to replicate key aspects of human cognition, including Theory of Mind reasoning and social inference, while also highlighting significant limitations such as cognitive biases, lack of true understanding, and inconsistencies in behavior. The second part surveys emerging applications of LLMs in multi-agent simulation frameworks, focusing on system architectures, scale, and validation strategies. Notable projects such as Generative Agents (Smallville) and AgentSociety are discussed in terms of their design choices, empirical grounding, and methodological innovations. Particular attention is given to the challenges of behavioral fidelity, calibration, and reproducibility in large-scale LLM-driven simulations. The final section distinguishes between contexts where LLMs, like other black-box systems, offer direct value-such as interactive simulations and serious games-and those where their use is more problematic, notably in explanatory or predictive modeling. The paper concludes by advocating for hybrid approaches that integrate LLMs into traditional agent-based modeling platforms (GAMA, Netlogo, etc), enabling modelers to combine the expressive flexibility of language-based reasoning with the transparency and analytical rigor of classical rule-based systems. 

**Abstract (ZH)**: 这一立场论文探讨了大型语言模型（LLMs）在社会仿真中的应用，从计算社会科学的角度分析了它们的潜在价值和局限性。第一部分回顾了LLMs在复制人类认知关键方面的能力，包括心理理论推理和社会推断，并指出了认知偏差、缺乏真正理解以及行为不一致性等重要局限。第二部分概述了LLMs在多代理仿真框架中的新兴应用，重点关注系统架构、规模和验证策略。讨论了诸如生成代理（Smallville）和AgentSociety等重要项目的设计选择、实证基础和方法论创新。特别关注大规模LLM驱动仿真中的行为忠实性、校准和可再现性挑战。最后一部分区分了LLMs与其他黑盒系统在提供直接价值（如交互仿真和严肃游戏）的情境，以及它们在解释性或预测建模中的使用更具问题性的场景。论文最后提倡将LLMs与传统基于代理的建模平台（如GAMA、NetLogo等）相结合的混合方法，使建模者能够结合基于语言的推理的表达灵活性与经典基于规则系统的透明性和分析严谨性。 

---
# Modeling Uncertainty: Constraint-Based Belief States in Imperfect-Information Games 

**Title (ZH)**: 基于约束的信念状态建模： imperfect-information 游戏中的不确定性建模 

**Authors**: Achille Morenville, Éric Piette  

**Link**: [PDF](https://arxiv.org/pdf/2507.19263)  

**Abstract**: In imperfect-information games, agents must make decisions based on partial knowledge of the game state. The Belief Stochastic Game model addresses this challenge by delegating state estimation to the game model itself. This allows agents to operate on externally provided belief states, thereby reducing the need for game-specific inference logic. This paper investigates two approaches to represent beliefs in games with hidden piece identities: a constraint-based model using Constraint Satisfaction Problems and a probabilistic extension using Belief Propagation to estimate marginal probabilities. We evaluated the impact of both representations using general-purpose agents across two different games. Our findings indicate that constraint-based beliefs yield results comparable to those of probabilistic inference, with minimal differences in agent performance. This suggests that constraint-based belief states alone may suffice for effective decision-making in many settings. 

**Abstract (ZH)**: 在不完善信息博弈中，代理必须基于对游戏状态的部分知识做出决策。信念随机博弈模型通过将状态估计委托给游戏模型本身来解决这一挑战，从而使代理能够基于外部提供的信念状态操作，从而减少针对特定游戏的推理逻辑的需求。本文探讨了两种在隐藏拼图身份的游戏中表示信念的方法：一种基于约束的模型使用约束满足问题，另一种概率扩展使用信念传播来估计边缘概率。我们在两个不同的游戏中使用通用代理评估了这两种表示方法的影响。研究结果表明，基于约束的信念与概率推理产生的结果相当，代理性能差异较小。这表明，在许多情况下，仅基于约束的信念状态可能足以实现有效的决策。 

---
# Knowledge Grafting: A Mechanism for Optimizing AI Model Deployment in Resource-Constrained Environments 

**Title (ZH)**: 知识嫁接：一种优化资源受限环境下AI模型部署的机制 

**Authors**: Osama Almurshed, Ashish Kaushal, Asmail Muftah, Nitin Auluck, Omer Rana  

**Link**: [PDF](https://arxiv.org/pdf/2507.19261)  

**Abstract**: The increasing adoption of Artificial Intelligence (AI) has led to larger, more complex models with numerous parameters that require substantial computing power -- resources often unavailable in many real-world application scenarios. Our paper addresses this challenge by introducing knowledge grafting, a novel mechanism that optimizes AI models for resource-constrained environments by transferring selected features (the scion) from a large donor model to a smaller rootstock model. The approach achieves an 88.54% reduction in model size (from 64.39 MB to 7.38 MB), while improving generalization capability of the model. Our new rootstock model achieves 89.97% validation accuracy (vs. donor's 87.47%), maintains lower validation loss (0.2976 vs. 0.5068), and performs exceptionally well on unseen test data with 90.45% accuracy. It addresses the typical size vs performance trade-off, and enables deployment of AI frameworks on resource-constrained devices with enhanced performance. We have tested our approach on an agricultural weed detection scenario, however, it can be extended across various edge computing scenarios, potentially accelerating AI adoption in areas with limited hardware/software support -- by mirroring in a similar manner the horticultural grafting enables productive cultivation in challenging agri-based environments. 

**Abstract (ZH)**: AI模型知识嫁接：资源受限环境下的高效优化 

---
# Faster Lifting for Ordered Domains with Predecessor Relations 

**Title (ZH)**: 有序域上具有前趋关系的更快提升方法 

**Authors**: Kuncheng Zou, Jiahao Mai, Yonggang Zhang, Yuyi Wang, Ondřej Kuželka, Yuanhong Wang, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19182)  

**Abstract**: We investigate lifted inference on ordered domains with predecessor relations, where the elements of the domain respect a total (cyclic) order, and every element has a distinct (clockwise) predecessor. Previous work has explored this problem through weighted first-order model counting (WFOMC), which computes the weighted sum of models for a given first-order logic sentence over a finite domain. In WFOMC, the order constraint is typically encoded by the linear order axiom introducing a binary predicate in the sentence to impose a linear ordering on the domain elements. The immediate and second predecessor relations are then encoded by the linear order predicate. Although WFOMC with the linear order axiom is theoretically tractable, existing algorithms struggle with practical applications, particularly when the predecessor relations are involved. In this paper, we treat predecessor relations as a native part of the axiom and devise a novel algorithm that inherently supports these relations. The proposed algorithm not only provides an exponential speedup for the immediate and second predecessor relations, which are known to be tractable, but also handles the general k-th predecessor relations. The extensive experiments on lifted inference tasks and combinatorics math problems demonstrate the efficiency of our algorithm, achieving speedups of a full order of magnitude. 

**Abstract (ZH)**: 有序域上基于前驱关系的提升推理研究 

---
# PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring 

**Title (ZH)**: PhysDrive：用于车内驾驶人监测的多模态远程生理测量数据集 

**Authors**: Jiyao Wang, Xiao Yang, Qingyong Hu, Jiankai Tang, Can Liu, Dengbo He, Yuntao Wang, Yingcong Chen, Kaishun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19172)  

**Abstract**: Robust and unobtrusive in-vehicle physiological monitoring is crucial for ensuring driving safety and user experience. While remote physiological measurement (RPM) offers a promising non-invasive solution, its translation to real-world driving scenarios is critically constrained by the scarcity of comprehensive datasets. Existing resources are often limited in scale, modality diversity, the breadth of biometric annotations, and the range of captured conditions, thereby omitting inherent real-world challenges in driving. Here, we present PhysDrive, the first large-scale multimodal dataset for contactless in-vehicle physiological sensing with dedicated consideration on various modality settings and driving factors. PhysDrive collects data from 48 drivers, including synchronized RGB, near-infrared camera, and raw mmWave radar data, accompanied with six synchronized ground truths (ECG, BVP, Respiration, HR, RR, and SpO2). It covers a wide spectrum of naturalistic driving conditions, including driver motions, dynamic natural light, vehicle types, and road conditions. We extensively evaluate both signal-processing and deep-learning methods on PhysDrive, establishing a comprehensive benchmark across all modalities, and release full open-source code with compatibility for mainstream public toolboxes. We envision PhysDrive will serve as a foundational resource and accelerate research on multimodal driver monitoring and smart-cockpit systems. 

**Abstract (ZH)**: 鲁棒且不干扰的车内生理监测对于确保驾驶安全和用户体验至关重要。虽然远程生理测量(RPM)提供了一种前景广阔的非侵入性解决方案，但其在实际驾驶场景中的应用受到全面数据集稀缺性的严重限制。现有资源在规模、模态多样性、生物特征注释的广度以及捕捉条件的范围上常常有限，从而忽略了实际驾驶中的固有挑战。在这里，我们介绍了PhysDrive，首个用于无接触车内生理传感的大规模多模态数据集，并针对各种模态设置和驾驶因素进行了专门考虑。PhysDrive 从48 名驾驶员中收集了数据，包括同步的RGB、近红外相机和原始毫米波雷达数据，以及六种同步的地面真实值（ECG、BVP、呼吸、心率、呼吸频率和血氧饱和度）。它涵盖了广泛的自然驾驶条件，包括驾驶员动作、动态自然光、车辆类型和道路条件。我们在PhysDrive 上全面评估了信号处理和深度学习方法，建立了所有模态的综合基准，并公开了与主流公共工具箱兼容的完整开源代码。我们期望PhysDrive 将成为基础资源，加速多模态驾驶员监测和智能仪表板系统的研究。 

---
# OS-MAP: How Far Can Computer-Using Agents Go in Breadth and Depth? 

**Title (ZH)**: OS-MAP: 计算机使用代理在广度和深度上能走多远？ 

**Authors**: Xuetian Chen, Yinghao Chen, Xinfeng Yuan, Zhuo Peng, Lu Chen, Yuekeng Li, Zhoujia Zhang, Yingqian Huang, Leyan Huang, Jiaqing Liang, Tianbao Xie, Zhiyong Wu, Qiushi Sun, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.19132)  

**Abstract**: Computer-using agents have shown strong potential to boost human productivity and enable new application forms across platforms. While recent advances have led to usable applications, existing benchmarks fail to account for the internal task heterogeneity and the corresponding agent capabilities, as well as their alignment with actual user demands-hindering both targeted capability development and the reliable transition of research progress into practical deployment. To bridge the gap, we present OS-MAP, a benchmark for daily computer-using automation that organizes its 416 realistic tasks across 15 applications along two key dimensions: a five-level taxonomy of automation and a generalization scope derived from a real-world user demand hierarchy. To enable fine-grained analysis of required capabilities and alignment with real-world scenarios, OS-MAP evaluates agents along two dimensions: automation level across a five-level taxonomy, and generalization scope across a demand hierarchy. This design captures varying levels of required agent autonomy and generalization, forming a performance-generalization evaluation matrix for structured and comprehensive assessment. Experiments show that even State-of-the-Art agents with VLM backbones struggle with higher-level tasks involving perception, reasoning, and coordination-highlighting the need for a deeper understanding of current strengths and limitations to drive the future progress in computer-using agents research and deployment. All code, environments, baselines, and data are publicly available at this https URL. 

**Abstract (ZH)**: 基于计算机的任务代理在提升人类生产力和跨平台启用新型应用形式方面展示了强大的潜力。现有基准未能充分考虑任务内部异质性和相应的代理能力，以及这些能力与实际用户需求的对齐问题，这阻碍了有针对性的能力开发，并阻碍了研究进展向实际部署的可靠过渡。为解决这一问题，我们提出OS-MAP基准，用于日常计算机使用自动化，该基准根据两个关键维度组织其416个实际任务，涵盖15个应用程序：五级自动化 taxonomy 和来自真实用户需求层次的一般化范围。为实现对所需能力的精细分析和与现实场景的对齐，OS-MAP沿两个维度评估代理：沿五级自动化 taxonomy 的自动化水平，以及沿需求层次的一般化范围。该设计捕捉了代理所需的不同自主性和一般化水平，形成了一个性能-一般化评估矩阵，用于有条理和全面的评估。实验表明，即便是基于VLM的最先进的代理也难以处理涉及感知、推理和协调的更高层级任务，强调了深入理解当前优势和局限性的必要性，以推动计算机使用代理研究和部署的未来发展。所有代码、环境、基线和数据均可在以下网址公开获取。 

---
# Pareto-NRPA: A Novel Monte-Carlo Search Algorithm for Multi-Objective Optimization 

**Title (ZH)**: 帕累托-NRPA：多目标优化的一种新型蒙特卡罗搜索算法 

**Authors**: Noé Lallouet, Tristan Cazenave, Cyrille Enderli  

**Link**: [PDF](https://arxiv.org/pdf/2507.19109)  

**Abstract**: We introduce Pareto-NRPA, a new Monte-Carlo algorithm designed for multi-objective optimization problems over discrete search spaces. Extending the Nested Rollout Policy Adaptation (NRPA) algorithm originally formulated for single-objective problems, Pareto-NRPA generalizes the nested search and policy update mechanism to multi-objective optimization. The algorithm uses a set of policies to concurrently explore different regions of the solution space and maintains non-dominated fronts at each level of search. Policy adaptation is performed with respect to the diversity and isolation of sequences within the Pareto front. We benchmark Pareto-NRPA on two classes of problems: a novel bi-objective variant of the Traveling Salesman Problem with Time Windows problem (MO-TSPTW), and a neural architecture search task on well-known benchmarks. Results demonstrate that Pareto-NRPA achieves competitive performance against state-of-the-art multi-objective algorithms, both in terms of convergence and diversity of solutions. Particularly, Pareto-NRPA strongly outperforms state-of-the-art evolutionary multi-objective algorithms on constrained search spaces. To our knowledge, this work constitutes the first adaptation of NRPA to the multi-objective setting. 

**Abstract (ZH)**: Pareto-NRPA：一种用于离散搜索空间多目标优化问题的新型蒙特卡洛算法 

---
# Fine-Grained Traffic Inference from Road to Lane via Spatio-Temporal Graph Node Generation 

**Title (ZH)**: 基于空间-时间图节点生成的从道路到车道的细粒度交通推断 

**Authors**: Shuhao Li, Weidong Yang, Yue Cui, Xiaoxing Liu, Lingkai Meng, Lipeng Ma, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19089)  

**Abstract**: Fine-grained traffic management and prediction are fundamental to key applications such as autonomous driving, lane change guidance, and traffic signal control. However, obtaining lane-level traffic data has become a critical bottleneck for data-driven models due to limitations in the types and number of sensors and issues with the accuracy of tracking algorithms. To address this, we propose the Fine-grained Road Traffic Inference (FRTI) task, which aims to generate more detailed lane-level traffic information using limited road data, providing a more energy-efficient and cost-effective solution for precise traffic management. This task is abstracted as the first scene of the spatio-temporal graph node generation problem. We designed a two-stage framework--RoadDiff--to solve the FRTI task. solve the FRTI task. This framework leverages the Road-Lane Correlation Autoencoder-Decoder and the Lane Diffusion Module to fully utilize the limited spatio-temporal dependencies and distribution relationships of road data to accurately infer fine-grained lane traffic states. Based on existing research, we designed several baseline models with the potential to solve the FRTI task and conducted extensive experiments on six datasets representing different road conditions to validate the effectiveness of the RoadDiff model in addressing the FRTI task. The relevant datasets and code are available at this https URL. 

**Abstract (ZH)**: 精细的道路交通推断与预测对于自动驾驶、车道变更指导和交通信号控制等关键应用至关重要。然而，由于传感器类型和数量的限制以及跟踪算法准确性的限制，获取车道级交通数据已成为数据驱动模型中的关键瓶颈。为此，我们提出了精细的道路交通推断（FRTI）任务，旨在利用有限的道路数据生成更详细的车道级交通信息，提供一种更具能源效率和成本效益的精确交通管理解决方案。该任务被抽象为时空图节点生成问题的第一阶段。我们设计了一个两阶段框架——RoadDiff，以解决FRTI任务。该框架利用道路-车道关联自动编码器-解码器和车道扩散模块，充分利用有限的时空依赖性和分布关系，准确推断车道级交通状态。基于现有研究，我们设计了几种潜在能够解决FRTI任务的基准模型，并在六个代表不同道路条件的数据集上进行了大量实验，以验证RoadDiff模型在解决FRTI任务中的有效性。相关数据集和代码可在以下链接获取：[this https URL]。 

---
# Towards Improving Long-Tail Entity Predictions in Temporal Knowledge Graphs through Global Similarity and Weighted Sampling 

**Title (ZH)**: 通过全局相似性和加权采样提高时间知识图谱中长尾实体预测性能 

**Authors**: Mehrnoosh Mirtaheri, Ryan A. Rossi, Sungchul Kim, Kanak Mahadik, Tong Yu, Xiang Chen, Mohammad Rostami  

**Link**: [PDF](https://arxiv.org/pdf/2507.18977)  

**Abstract**: Temporal Knowledge Graph (TKG) completion models traditionally assume access to the entire graph during training. This overlooks challenges stemming from the evolving nature of TKGs, such as: (i) the model's requirement to generalize and assimilate new knowledge, and (ii) the task of managing new or unseen entities that often have sparse connections. In this paper, we present an incremental training framework specifically designed for TKGs, aiming to address entities that are either not observed during training or have sparse connections. Our approach combines a model-agnostic enhancement layer with a weighted sampling strategy, that can be augmented to and improve any existing TKG completion method. The enhancement layer leverages a broader, global definition of entity similarity, which moves beyond mere local neighborhood proximity of GNN-based methods. The weighted sampling strategy employed in training accentuates edges linked to infrequently occurring entities. We evaluate our method on two benchmark datasets, and demonstrate that our framework outperforms existing methods in total link prediction, inductive link prediction, and in addressing long-tail entities. Notably, our method achieves a 10\% improvement and a 15\% boost in MRR for these datasets. The results underscore the potential of our approach in mitigating catastrophic forgetting and enhancing the robustness of TKG completion methods, especially in an incremental training context 

**Abstract (ZH)**: 增量训练框架在Temporal Knowledge Graph (TKG) 完善中的应用 

---
# Success in Humanoid Reinforcement Learning under Partial Observation 

**Title (ZH)**: 部分观测下 humanoid 强化学习的成功 

**Authors**: Wuhao Wang, Zhiyong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.18883)  

**Abstract**: Reinforcement learning has been widely applied to robotic control, but effective policy learning under partial observability remains a major challenge, especially in high-dimensional tasks like humanoid locomotion. To date, no prior work has demonstrated stable training of humanoid policies with incomplete state information in the benchmark Gymnasium Humanoid-v4 environment. The objective in this environment is to walk forward as fast as possible without falling, with rewards provided for staying upright and moving forward, and penalties incurred for excessive actions and external contact forces. This research presents the first successful instance of learning under partial observability in this environment. The learned policy achieves performance comparable to state-of-the-art results with full state access, despite using only one-third to two-thirds of the original states. Moreover, the policy exhibits adaptability to robot properties, such as variations in body part masses. The key to this success is a novel history encoder that processes a fixed-length sequence of past observations in parallel. Integrated into a standard model-free algorithm, the encoder enables performance on par with fully observed baselines. We hypothesize that it reconstructs essential contextual information from recent observations, thereby enabling robust decision-making. 

**Abstract (ZH)**: 强化学习在机器人控制中的应用已经十分广泛，但在不完全可观测性条件下获得有效的策略依然是一项重大挑战，特别是在高维度任务如类人步行控制中。迄今为止，在基准环境Gymnasium Humanoid-v4中，没有先有研究展示了能够稳定训练出使用不完整状态信息的类人策略。该环境的目标是在不摔倒的情况下尽可能快地向前行走，奖励基于站立稳定和向前移动，而过度动作和外部接触力则会受到惩罚。本研究在此环境中展示了首次成功实现不完全可观测性条件下的策略学习实例。所学习出的策略在使用原始状态信息的三分之一到三分之二的情况下，达到了与完全可观测性状态下顶尖结果相当的表现。此外，该策略还展示了对机器人属性的适应性，例如身体部分质量的变化。这一成功的关键在于一种新颖的历史编码器，该编码器能够并行处理固定长度的过去观察序列。将编码器整合到标准的模型自由算法中，能够实现与完全可观测性基准相当的性能。我们推测它能够从最近的观察中重建出关键的背景信息，从而支持稳健的决策制定。 

---
# A Neuroscience-Inspired Dual-Process Model of Compositional Generalization 

**Title (ZH)**: 神经科学启发的组合理念泛化双过程模型 

**Authors**: Alex Noviello, Claas Beger, Jacob Groner, Kevin Ellis, Weinan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.18868)  

**Abstract**: Systematic compositional generalization - constructing and understanding novel combinations of known building blocks - remains a core challenge for AI systems. Human cognition achieves this flexibility via the interplay of the hippocampus (HPC) and prefrontal cortex (PFC): the hippocampus rapidly encodes episodes, and the prefrontal cortex consolidates them into reusable schemas for reasoning. Drawing on these insights, we present MIRAGE (Meta-Inference with Rules and Abstractions from Generalized Experience), a framework that achieves systematic generalization on compositional tasks. MIRAGE has two interacting modules mirroring the brain's deliberative HPC-PFC loop and intuitive neocortical pattern recognition. (1) The meta-trained Transformer Neural Decomposer, paralleling neocortical "System 1" computation, is trained on a task-agnostic stream of randomly sampled compositional grammars and applies one decomposition step per pass, with successive passes iteratively refining the sequence representation. (2) The Schema Engine, analogous to the HPC-PFC "System 2" loop, dynamically extracts, ranks, and applies reusable schemas, storing variable bindings in episodic memory and expanding them when needed. By explicitly equipping the Transformer component of MIRAGE with actively managed schematic structures, our model performs systematic compositional operations through explicit schema application and transformation, relying solely on frozen weights when solving entirely novel tasks. This approach demonstrates systematic compositional generalization on the SCAN benchmark, achieving > 99% accuracy on all task splits with only 1.19M parameters in the transformer module. Ablation studies confirm that MIRAGE's systematicity critically depends on the quality of extracted schemas and the model's iterative refinement process. 

**Abstract (ZH)**: 系统组合泛化：构建和理解已知组件的新组合——仍然是AI系统的核心挑战。人类认知通过海马体（HPC）和前额皮层（PFC）的交互实现这种灵活性：海马体快速编码事件，前额皮层将其整合为可重用的推理模式。基于此见解，我们提出MIRAGE（基于通用经验的规则与抽象元推理框架），该框架在组合任务中实现了系统泛化。MIRAGE有两个相互作用的模块，模拟大脑的审慎HPC-PFC循环和直观的新皮层模式识别。（1）元训练的Transformer神经分解器，类比于皮层“系统1”计算，基于任务无关的随机采样组合语法进行训练，每次通过执行一步分解，并通过迭代优化逐步细化序列表示。（2）模式引擎，类比于HPC-PFC的“系统2”循环，动态提取、排序和应用可重用的模式，将变量绑定存储在情景记忆中并在需要时扩展它们。通过明确地在MIRAGE的Transformer组件中配备主动管理的模式结构，我们的模型在进行系统组合操作时依赖于明确的应用和转换模式，仅在解决完全新颖的任务时使用冻结权重。此方法在SCAN基准测试中展示了系统组合泛化，在仅使用Transformer模块1.19M个参数的情况下，所有任务分割的准确率均超过99%。消融研究证实，MIRAGE的系统性高度依赖于提取模式的质量和模型的迭代优化过程。 

---
# Simulation-Driven Reinforcement Learning in Queuing Network Routing Optimization 

**Title (ZH)**: 基于模拟驱动的强化学习在排队网络路由优化中的应用 

**Authors**: Fatima Al-Ani, Molly Wang, Jevon Charles, Aaron Ong, Joshua Forday, Vinayak Modi  

**Link**: [PDF](https://arxiv.org/pdf/2507.18795)  

**Abstract**: This study focuses on the development of a simulation-driven reinforcement learning (RL) framework for optimizing routing decisions in complex queueing network systems, with a particular emphasis on manufacturing and communication applications. Recognizing the limitations of traditional queueing methods, which often struggle with dynamic, uncertain environments, we propose a robust RL approach leveraging Deep Deterministic Policy Gradient (DDPG) combined with Dyna-style planning (Dyna-DDPG). The framework includes a flexible and configurable simulation environment capable of modeling diverse queueing scenarios, disruptions, and unpredictable conditions. Our enhanced Dyna-DDPG implementation incorporates separate predictive models for next-state transitions and rewards, significantly improving stability and sample efficiency. Comprehensive experiments and rigorous evaluations demonstrate the framework's capability to rapidly learn effective routing policies that maintain robust performance under disruptions and scale effectively to larger network sizes. Additionally, we highlight strong software engineering practices employed to ensure reproducibility and maintainability of the framework, enabling practical deployment in real-world scenarios. 

**Abstract (ZH)**: 本研究聚焦于发展一种基于仿真的强化学习（RL）框架，用于优化复杂队列网络系统中的路由决策，特别关注制造和通信应用。鉴于传统队列方法在动态、不确定环境中存在的局限性，本文提出了一种结合深度确定性策略梯度（DDPG）和Dynashile规划（Dyna-DDPG）的鲁棒RL方法。该框架包含一个灵活且可配置的仿真环境，能够模拟各种队列场景、中断和不可预测的条件。我们的增强型Dyna-DDPG实现分别引入了下一状态转换和奖励的预测模型，显著提高了稳定性和样本效率。全面的实验和严格的评估表明，该框架能够迅速学习有效的路由策略，在遭遇中断时维持稳健性能，并有效扩展到更大的网络规模。此外，本文强调了为确保框架的可重现性和可维护性而采用的强大的软件工程实践，使其能够在实际场景中实现有效的部署。 

---
# Initial Steps in Integrating Large Reasoning and Action Models for Service Composition 

**Title (ZH)**: 初始步骤：集成大规模推理与行动模型进行服务组合 

**Authors**: Ilche Georgievski, Marco Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2507.18775)  

**Abstract**: Service composition remains a central challenge in building adaptive and intelligent software systems, often constrained by limited reasoning capabilities or brittle execution mechanisms. This paper explores the integration of two emerging paradigms enabled by large language models: Large Reasoning Models (LRMs) and Large Action Models (LAMs). We argue that LRMs address the challenges of semantic reasoning and ecosystem complexity while LAMs excel in dynamic action execution and system interoperability. However, each paradigm has complementary limitations - LRMs lack grounded action capabilities, and LAMs often struggle with deep reasoning. We propose an integrated LRM-LAM architectural framework as a promising direction for advancing automated service composition. Such a system can reason about service requirements and constraints while dynamically executing workflows, thus bridging the gap between intention and execution. This integration has the potential to transform service composition into a fully automated, user-friendly process driven by high-level natural language intent. 

**Abstract (ZH)**: 基于大型语言模型的大型推理模型和大型行动模型集成：服务组合的新范式 

---
# Advancing Event Forecasting through Massive Training of Large Language Models: Challenges, Solutions, and Broader Impacts 

**Title (ZH)**: 通过大规模训练大型语言模型推进事件预测：挑战、解决方案及更广泛的影响 

**Authors**: Sang-Woo Lee, Sohee Yang, Donghyun Kwak, Noah Y. Siegel  

**Link**: [PDF](https://arxiv.org/pdf/2507.19477)  

**Abstract**: Many recent papers have studied the development of superforecaster-level event forecasting LLMs. While methodological problems with early studies cast doubt on the use of LLMs for event forecasting, recent studies with improved evaluation methods have shown that state-of-the-art LLMs are gradually reaching superforecaster-level performance, and reinforcement learning has also been reported to improve future forecasting. Additionally, the unprecedented success of recent reasoning models and Deep Research-style models suggests that technology capable of greatly improving forecasting performance has been developed. Therefore, based on these positive recent trends, we argue that the time is ripe for research on large-scale training of superforecaster-level event forecasting LLMs. We discuss two key research directions: training methods and data acquisition. For training, we first introduce three difficulties of LLM-based event forecasting training: noisiness-sparsity, knowledge cut-off, and simple reward structure problems. Then, we present related ideas to mitigate these problems: hypothetical event Bayesian networks, utilizing poorly-recalled and counterfactual events, and auxiliary reward signals. For data, we propose aggressive use of market, public, and crawling datasets to enable large-scale training and evaluation. Finally, we explain how these technical advances could enable AI to provide predictive intelligence to society in broader areas. This position paper presents promising specific paths and considerations for getting closer to superforecaster-level AI technology, aiming to call for researchers' interest in these directions. 

**Abstract (ZH)**: 近年来，许多研究论文探讨了超预测水平事件forecasting大型语言模型（LLM）的发展。尽管早期研究中的方法论问题曾对使用LLM进行事件forecasting的能力提出质疑，但近期采用改进评估方法的研究表明，最先进的LLM正在逐步达到超预测水平的表现，并且强化学习还被报告能改善未来forecasting。此外，近期推理模型和Deep Research风格模型前所未有的成功表明，大幅提升forecasting性能的技术已经开发出来。因此，基于这些积极的发展趋势，我们认为大规模训练超预测水平事件forecasting大型语言模型的时机已经成熟。我们讨论了两个关键的研究方向：训练方法和数据收集。在训练方面，我们首先介绍了基于LLM的事件forecasting训练的三个困难：噪声稀疏性、知识截断问题和简单的奖励结构问题。然后，我们提出了相关的方法来缓解这些问题：假设事件贝叶斯网络、利用召回不佳和反事实事件以及辅助奖励信号。在数据方面，我们建议积极使用市场、公共和爬取的数据集以实现大规模训练和评估。最后，我们解释了这些技术进步如何能够使AI在更广泛的领域提供预测智能。这份观点论文提出了通往达到超预测水平AI技术的具体路径和考虑，旨在呼吁研究人员对这些方向的兴趣。 

---
# Let It Go? Not Quite: Addressing Item Cold Start in Sequential Recommendations with Content-Based Initialization 

**Title (ZH)**: Let It Go? Not Quite: 用内容初始化解决序贯推荐中的项目冷启动问题 

**Authors**: Anton Pembek, Artem Fatkulin, Anton Klenitskiy, Alexey Vasilev  

**Link**: [PDF](https://arxiv.org/pdf/2507.19473)  

**Abstract**: Many sequential recommender systems suffer from the cold start problem, where items with few or no interactions cannot be effectively used by the model due to the absence of a trained embedding. Content-based approaches, which leverage item metadata, are commonly used in such scenarios. One possible way is to use embeddings derived from content features such as textual descriptions as initialization for the model embeddings. However, directly using frozen content embeddings often results in suboptimal performance, as they may not fully adapt to the recommendation task. On the other hand, fine-tuning these embeddings can degrade performance for cold-start items, as item representations may drift far from their original structure after training. We propose a novel approach to address this limitation. Instead of entirely freezing the content embeddings or fine-tuning them extensively, we introduce a small trainable delta to frozen embeddings that enables the model to adapt item representations without letting them go too far from their original semantic structure. This approach demonstrates consistent improvements across multiple datasets and modalities, including e-commerce datasets with textual descriptions and a music dataset with audio-based representation. 

**Abstract (ZH)**: 许多序贯推荐系统面临冷启动问题，其中交互较少或几乎没有交互的项目由于缺乏训练嵌入而无法被模型有效利用。内容基方法利用项目元数据在这种情况下常被使用。一种可能的方法是使用来自内容特征（如文本描述）的嵌入作为模型嵌入的初始化。然而，直接使用冻结的内容嵌入往往会导致性能不佳，因为它们可能无法充分适应推荐任务。另一方面，对这些嵌入进行微调可能会削弱冷启动项目的性能，因为项目表示在训练后可能会远离其原始结构。我们提出了一种新颖的方法来解决这一局限。我们引入了一个小型可训练增量调整冻结嵌入，使模型能够适应项目表示，同时保持其原始语义结构不发生变化。该方法在多个数据集和模态下显示出一致的改进，包括带有文本描述的电商数据集和基于音频表示的音乐数据集。 

---
# GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning 

**Title (ZH)**: GEPA: 自省式提示进化可以超越强化学习 

**Authors**: Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, Omar Khattab  

**Link**: [PDF](https://arxiv.org/pdf/2507.19457)  

**Abstract**: Large language models (LLMs) are increasingly adapted to downstream tasks via reinforcement learning (RL) methods like Group Relative Policy Optimization (GRPO), which often require thousands of rollouts to learn new tasks. We argue that the interpretable nature of language can often provide a much richer learning medium for LLMs, compared with policy gradients derived from sparse, scalar rewards. To test this, we introduce GEPA (Genetic-Pareto), a prompt optimizer that thoroughly incorporates natural language reflection to learn high-level rules from trial and error. Given any AI system containing one or more LLM prompts, GEPA samples system-level trajectories (e.g., reasoning, tool calls, and tool outputs) and reflects on them in natural language to diagnose problems, propose and test prompt updates, and combine complementary lessons from the Pareto frontier of its own attempts. As a result of GEPA's design, it can often turn even just a few rollouts into a large quality gain. Across four tasks, GEPA outperforms GRPO by 10% on average and by up to 20%, while using up to 35x fewer rollouts. GEPA also outperforms the leading prompt optimizer, MIPROv2, by over 10% across two LLMs, and demonstrates promising results as an inference-time search strategy for code optimization. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过强化学习（RL）方法如群组相对策略优化（GRPO）越来越多地适应下游任务，这通常需要成千上万次回放来学习新任务。我们认为，语言的可解释性通常为LLMs提供了更为丰富的学习媒介，相比于从稀疏标量奖励派生的策略梯度。为了验证这一观点，我们引入了GEPA（Genetic-Pareto），这是一种全面结合自然语言反思的提示优化器，用于从试错中学到高层次规则。对于包含一个或多个LLM提示的任何AI系统，GEPA会采样系统级轨迹（例如推理、工具调用和工具输出），并在自然语言中对其进行反思以诊断问题、建议并测试提示更新，结合来自其自身尝试帕累托前沿的互补教训。由于GEPA的设计，它通常能够将少量回放转变成显著的质量提升。在四个任务中，GEPA在平均上比GRPO高出10%，最高可达20%，同时使用不到GRPO的35%的回放次数。GEPA还跨两个LLM比领先的提示优化器MIPROv2高出超过10%，并且展示了在代码优化的推理时搜索策略方面的有前途的结果。 

---
# Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding 

**Title (ZH)**: 步进3是大而经济的：低成本解码的模型系统协同设计 

**Authors**: StepFun, Bin Wang, Bojun Wang, Changyi Wan, Guanzhe Huang, Hanpeng Hu, Haonan Jia, Hao Nie, Mingliang Li, Nuo Chen, Siyu Chen, Song Yuan, Wuxun Xie, Xiaoniu Song, Xing Chen, Xingping Yang, Xuelin Zhang, Yanbo Yu, Yaoyu Wang, Yibo Zhu, Yimin Jiang, Yu Zhou, Yuanwei Lu, Houyi Li, Jingcheng Hu, Ka Man Lo, Ailin Huang, Binxing Jiao, Bo Li, Boyu Chen, Changxin Miao, Chang Lou, Chen Hu, Chen Xu, Chenfeng Yu, Chengyuan Yao, Daokuan Lv, Dapeng Shi, Deshan Sun, Ding Huang, Dingyuan Hu, Dongqing Pang, Enle Liu, Fajie Zhang, Fanqi Wan, Gulin Yan, Han Zhang, Han Zhou, Hanghao Wu, Hangyu Guo, Hanqi Chen, Hanshan Zhang, Hao Wu, Haocheng Zhang, Haolong Yan, Haoran Lv, Haoran Wei, Hebin Zhou, Heng Wang, Heng Wang, Hongxin Li, Hongyu Zhou, Hongyuan Wang, Huiyong Guo, Jia Wang, Jiahao Gong, Jialing Xie, Jian Zhou, Jianjian Sun, Jiaoren Wu, Jiaran Zhang, Jiayu Liu, Jie Cheng, Jie Luo, Jie Yan, Jie Yang, Jieyi Hou, Jinguang Zhang, Jinlan Cao, Jisheng Yin, Junfeng Liu, Junhao Huang, Junzhe Lin, Kaijun Tan, Kaixiang Li, Kang An, Kangheng Lin, Kenkun Liu, Lei Yang, Liang Zhao, Liangyu Chen, Lieyu Shi, Liguo Tan, Lin Lin, Lin Zhang, Lina Chen, Liwen Huang, Liying Shi, Longlong Gu, Mei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19427)  

**Abstract**: Large language models (LLMs) face low hardware efficiency during decoding, especially for long-context reasoning tasks. This paper introduces Step-3, a 321B-parameter VLM with hardware-aware model-system co-design optimized for minimizing decoding costs. Step-3 innovates in two key dimensions: (1) A novel Multi-Matrix Factorization Attention (MFA) mechanism that significantly reduces both KV cache size and computation while maintaining high attention expressiveness, and (2) Attention-FFN Disaggregation (AFD), a distributed inference system that decouples attention and Feed-Forward Network (FFN) layers into specialized subsystems. This co-design achieves unprecedented cost efficiency: Step-3 significantly reduces theoretical decoding costs compared with models like DeepSeek-V3 and Qwen3 MoE 235B, with the gains widening at longer context. Step-3 achieves low cost while activating 38B parameters per token (more than DeepSeek-V3 and Qwen3 MoE 235B), demonstrating that hardware-aligned attention arithmetic intensity, MoE sparsity, and AFD are critical to cost-effectiveness. We perform a head-to-head comparison with DeepSeek-V3 in its favorable scenarios. Our implementation on Hopper GPUs achieves a decoding throughput of up to 4,039 tokens per second per GPU under 50ms TPOT SLA (4K context, FP8, no MTP). It is higher than DeepSeek-V3's 2,324 in the same setup and sets a new Pareto frontier for LLM decoding. 

**Abstract (ZH)**: 大型语言模型（LLMs）在解码过程中的硬件效率较低，尤其是对于 largo 上下文推理任务。本文介绍了 Step-3，这是一种321B参数的VLM，通过硬件意识下的模型-系统协同设计来优化解码成本最小化。Step-3 在两个关键维度上进行了创新：（1）一种新颖的多矩阵因子化注意力（MFA）机制，显著减少了关键值缓存大小和计算量，同时保持高注意力表达性；（2）注意力-前馈网络分离（AFD），这是一种分布式推理系统，将注意力和前馈网络层解耦为专门的子系统。这种协同设计取得了前所未有的成本效率：Step-3 相比 DeepSeek-V3 和 Qwen3 MoE 235B 模型显著降低了理论解码成本，尤其是在更长的上下文环境中。Step-3 在每token激活38B参数的同时实现低成本，表明硬件对齐的注意力算术强度、MoE 稀疏性和AFD 对于成本效益至关重要。我们在其有利场景下与 DeepSeek-V3 进行了直接对比。在 Hopper GPU 上，我们的实现下在50ms TPOT SLA（4K 上下文，FP8，无MTP）下每个GPU的解码吞吐量可达4,039 tokens/s，高于 DeepSeek-V3 的2,324，并为LLM解码设定了新的帕累托前沿。 

---
# On Arbitrary Predictions from Equally Valid Models 

**Title (ZH)**: 关于同等有效的模型任意预测的探讨 

**Authors**: Sarah Lockfisch, Kristian Schwethelm, Martin Menten, Rickmer Braren, Daniel Rueckert, Alexander Ziller, Georgios Kaissis  

**Link**: [PDF](https://arxiv.org/pdf/2507.19408)  

**Abstract**: Model multiplicity refers to the existence of multiple machine learning models that describe the data equally well but may produce different predictions on individual samples. In medicine, these models can admit conflicting predictions for the same patient -- a risk that is poorly understood and insufficiently addressed.
In this study, we empirically analyze the extent, drivers, and ramifications of predictive multiplicity across diverse medical tasks and model architectures, and show that even small ensembles can mitigate/eliminate predictive multiplicity in practice. Our analysis reveals that (1) standard validation metrics fail to identify a uniquely optimal model and (2) a substantial amount of predictions hinges on arbitrary choices made during model development. Using multiple models instead of a single model reveals instances where predictions differ across equally plausible models -- highlighting patients that would receive arbitrary diagnoses if any single model were used. In contrast, (3) a small ensemble paired with an abstention strategy can effectively mitigate measurable predictive multiplicity in practice; predictions with high inter-model consensus may thus be amenable to automated classification. While accuracy is not a principled antidote to predictive multiplicity, we find that (4) higher accuracy achieved through increased model capacity reduces predictive multiplicity.
Our findings underscore the clinical importance of accounting for model multiplicity and advocate for ensemble-based strategies to improve diagnostic reliability. In cases where models fail to reach sufficient consensus, we recommend deferring decisions to expert review. 

**Abstract (ZH)**: 模型多样性是指存在多个机器学习模型能够同样很好地描述数据，但在个别样本上的预测可能不同。在医学领域，这些模型可能会对同一患者给出冲突的预测，这一风险目前尚未被充分理解和应对。

在本研究中，我们通过对多种医学任务和模型架构的实证分析，探讨预测多样性的发展驱动因素及其后果，并展示了即使是小型模型集成也可以在实践中减轻/消除预测多样性。我们的分析揭示了以下几点：（1）标准验证指标无法识别出最优模型；（2）大量预测依赖于模型开发过程中做出的任意选择；（3）结合小型模型集成和弃权策略可以有效减轻可测量的预测多样性；因此，具有高跨模型一致性的预测可以适于自动分类。虽然准确性不是解决预测多样性的根本方法，但我们发现（4）通过增加模型容量来提高准确性可以减少预测多样性。

我们的发现强调了在临床中考虑模型多样性的临床重要性，并倡导基于集成的策略以提高诊断可靠性。在模型无法达到足够共识的情况下，我们建议将决策提交给专家审核。 

---
# SDVDiag: A Modular Platform for the Diagnosis of Connected Vehicle Functions 

**Title (ZH)**: SDVDiag: 一种面向连接车辆功能诊断的模块化平台 

**Authors**: Matthias Weiß, Falk Dettinger, Michael Weyrich  

**Link**: [PDF](https://arxiv.org/pdf/2507.19403)  

**Abstract**: Connected and software-defined vehicles promise to offer a broad range of services and advanced functions to customers, aiming to increase passenger comfort and support autonomous driving capabilities. Due to the high reliability and availability requirements of connected vehicles, it is crucial to resolve any occurring failures quickly. To achieve this however, a complex cloud/edge architecture with a mesh of dependencies must be navigated to diagnose the responsible root cause. As such, manual analyses become unfeasible since they would significantly delay the troubleshooting.
To address this challenge, this paper presents SDVDiag, an extensible platform for the automated diagnosis of connected vehicle functions. The platform enables the creation of pipelines that cover all steps from initial data collection to the tracing of potential root causes. In addition, SDVDiag supports self-adaptive behavior by the ability to exchange modules at runtime. Dependencies between functions are detected and continuously updated, resulting in a dynamic graph view of the system. In addition, vital system metrics are monitored for anomalies. Whenever an incident is investigated, a snapshot of the graph is taken and augmented by relevant anomalies. Finally, the analysis is performed by traversing the graph and creating a ranking of the most likely causes.
To evaluate the platform, it is deployed inside an 5G test fleet environment for connected vehicle functions. The results show that injected faults can be detected reliably. As such, the platform offers the potential to gain new insights and reduce downtime by identifying problems and their causes at an early stage. 

**Abstract (ZH)**: 连接和软件定义的车辆有望为乘客提供广泛的服务和高级功能，旨在提高乘坐舒适度并支持自动驾驶能力。由于连接车辆对可靠性和可用性有高要求，快速解决任何发生的故障至关重要。然而，要实现这一目标，必须导航通过一个包含复杂依赖关系的云/边缘架构来诊断责任根源。因此，手动分析变得不可行，因为它们会显著延迟故障排除。

为了解决这一挑战，本文提出了SDVDiag，一个用于连接车辆功能自动诊断的可扩展平台。该平台使用户能够创建涵盖从初始数据收集到追踪潜在根源的全部步骤的管道。此外，SDVDiag 支持自我适应行为，能够在运行时交换模块。检测并持续更新功能之间的依赖关系，从而生成系统的动态图视图。此外，还会监控关键系统指标以检测异常。每当调查事故时，会捕获图的快照并添加相关异常。最后，通过遍历图并将最有可能的原因进行排名来进行分析。

为了评估该平台，在连接车辆功能的5G测试车队环境中部署了SDVDiag。结果显示，注入的故障可以可靠地被检测到。因此，该平台提供了通过早期识别问题及其原因来获取新见解并减少停机时间的潜力。 

---
# Running in CIRCLE? A Simple Benchmark for LLM Code Interpreter Security 

**Title (ZH)**: 在CIRCLE中运行？一个简单的LLM代码解释器安全性基准测试 

**Authors**: Gabriel Chua  

**Link**: [PDF](https://arxiv.org/pdf/2507.19399)  

**Abstract**: As large language models (LLMs) increasingly integrate native code interpreters, they enable powerful real-time execution capabilities, substantially expanding their utility. However, such integrations introduce potential system-level cybersecurity threats, fundamentally different from prompt-based vulnerabilities. To systematically evaluate these interpreter-specific risks, we propose CIRCLE (Code-Interpreter Resilience Check for LLM Exploits), a simple benchmark comprising 1,260 prompts targeting CPU, memory, and disk resource exhaustion. Each risk category includes explicitly malicious ("direct") and plausibly benign ("indirect") prompt variants. Our automated evaluation framework assesses not only whether LLMs refuse or generates risky code, but also executes the generated code within the interpreter environment to evaluate code correctness, simplifications made by the LLM to make the code safe, or execution timeouts. Evaluating 7 commercially available models from OpenAI and Google, we uncover significant and inconsistent vulnerabilities. For instance, evaluations show substantial disparities even within providers - OpenAI's o4-mini correctly refuses risky requests at 7.1%, notably higher rates compared to GPT-4.1 at 0.5%. Results particularly underscore that indirect, socially-engineered prompts substantially weaken model defenses. This highlights an urgent need for interpreter-specific cybersecurity benchmarks, dedicated mitigation tools (e.g., guardrails), and clear industry standards to guide safe and responsible deployment of LLM interpreter integrations. The benchmark dataset and evaluation code are publicly released to foster further research. 

**Abstract (ZH)**: 作为大型语言模型（LLMs）越来越多地集成本地代码解释器，它们能够实现强大的实时执行能力，大幅扩展了其应用范围。然而，这些集成引入了潜在的系统级网络安全威胁，与基于提示的漏洞本质上不同。为系统地评估这些解释器特定的风险，我们提出了CIRCLE（代码解释器抗利用韧性检查），这是一个包含1260个针对CPU、内存和磁盘资源耗尽的提示的简单基准。每个风险类别包括明确恶意的（直接）和合理的无害的（间接）提示变体。我们自动化评估框架不仅评估LLMs是否拒绝或生成风险代码，还执行生成的代码以评估代码正确性、LLMs为使代码安全所做的简化或执行超时情况。评估来自OpenAI和Google的7种商用模型，我们发现了显著且不一致的漏洞。例如，评估结果显示，在同一个提供商内也存在显著差异——OpenAI的o4-mini在拒绝风险请求方面正确率达到了7.1%，远高于GPT-4.1的0.5%。结果特别强调，间接的社会工程化提示极大地削弱了模型的防御能力。这突显出对特定于解释器的网络安全基准、专用缓解工具（例如护栏）和明确的行业标准的迫切需求，以指导LLM解释器集成的安全和负责任部署。基准数据集和评估代码已公开发布，以促进进一步的研究。 

---
# CXR-CML: Improved zero-shot classification of long-tailed multi-label diseases in Chest X-Rays 

**Title (ZH)**: CXR-CML：改进的胸部X光长尾多标签疾病零样本分类 

**Authors**: Rajesh Madhipati, Sheethal Bhat, Lukas Buess, Andreas Maier  

**Link**: [PDF](https://arxiv.org/pdf/2507.19398)  

**Abstract**: Chest radiography (CXR) plays a crucial role in the diagnosis of various diseases. However, the inherent class imbalance in the distribution of clinical findings presents a significant challenge for current self-supervised deep learning models. These models often fail to accurately classify long-tailed classes. Current Vision-Language models such as Contrastive Language Image Pre-training (CLIP) models effectively model the manifold distribution of the latent space, enabling high zero-shot classification accuracies. Although CLIP performs well on most of the primary classes in the dataset, our work reveals that its effectiveness decreases significantly for classes with a long-tailed distribution. Our approach employs a class-weighting mechanism that directly aligns with the distribution of classes within the latent space. This method ensures a substantial improvement in overall classification performance, with particular emphasis on enhancing the recognition and accuracy of rarely observed classes. We accomplish this by applying Gaussian Mixture Model (GMM) clustering to the latent space. The subsequent clusters are further refined by Student t-distribution, followed by a metric loss that utilizes the altered embeddings. Our approach facilitates stable and adaptive clustering of the features. This results in a notable average improvement of 7\% points in zero-shot AUC scores across 40 classes in the MIMIC-CXR-JPG dataset from previous SOTA models. 

**Abstract (ZH)**: 胸部X光影像（CXR）在各类疾病诊断中扮演着重要作用。然而，临床发现中存在的固有类别不平衡为当前的自监督深度学习模型带来了重大挑战。这些模型往往难以准确分类长尾类别。虽然现有的Vision-Language模型如对比语言图像预训练（CLIP）模型能够有效建模潜在空间的流形分布，从而实现高零-shot分类准确率，但我们的研究发现，它们对长尾分布类别效果显著下降。我们的方法采用类别加权机制，直接与潜在空间内类别的分布相匹配，确保在整体分类性能上取得显著提升，特别是在识别和提高罕见类别准确率方面。我们通过在潜在空间中应用高斯混合模型（GMM）聚类实现这一目标。随后的聚类进一步通过学生t分布精炼，并使用修改后的嵌入计算度量损失，从而实现特征的稳定和自适应聚类。我们在MIMIC-CXR-JPG数据集中40个类别上的零-shot AUC分数上取得了平均7个百分点的显著改进，超过之前的所有方法。 

---
# ReCatcher: Towards LLMs Regression Testing for Code Generation 

**Title (ZH)**: ReCatcher: 朝向代码生成的大语言模型回归测试 

**Authors**: Altaf Allah Abbassi, Leuson Da Silva, Amin Nikanjam, Foutse Khomh  

**Link**: [PDF](https://arxiv.org/pdf/2507.19390)  

**Abstract**: Large Language Models (LLMs) for code generation evolve rapidly through fine-tuning, merging, or new model releases. However, such updates can introduce regressions, not only in correctness but also in code quality and performance. To address this, we present ReCatcher, a regression testing framework for Python code generation. ReCatcher systematically compares two LLMs, typically a current model and a candidate update, across three dimensions: logical correctness, static code quality, and execution performance. We apply ReCatcher to assess regressions across three update scenarios, fine-tuning, merging, and model release, using CodeLlama, DeepSeek-Coder, and GPT-4o. Our evaluation shows that fine-tuning with cross-language datasets increases syntax errors by up to 12%. Merging with general-purpose models like Llama2 leads to regressions in correctness by up to 18%. GPT-4o introduces regressions of up to 50% in handling missing imports compared to GPT-3.5-turbo, while GPT-4o-mini suffers up to 80% performance degradation in execution time versus GPT-4o. Overall, logical correctness, performance, and error handling (e.g., syntax errors and missing imports) are the most regression-prone areas. Comparing ReCatcher with baseline solutions, it presents better and consistent accuracy across logical and performance aspects. ReCatcher highlights the importance of systematic regression evaluation before adopting new models, while assisting researchers and practitioners in making more informed update decisions. 

**Abstract (ZH)**: 大型语言模型（LLMs）在代码生成方面的回归测试框架：ReCatcher 

---
# Data Augmentation for Spoken Grammatical Error Correction 

**Title (ZH)**: 口语语法错误纠正的数据增强方法 

**Authors**: Penny Karanasou, Mengjie Qian, Stefano Bannò, Mark J.F. Gales, Kate M. Knill  

**Link**: [PDF](https://arxiv.org/pdf/2507.19374)  

**Abstract**: While there exist strong benchmark datasets for grammatical error correction (GEC), high-quality annotated spoken datasets for Spoken GEC (SGEC) are still under-resourced. In this paper, we propose a fully automated method to generate audio-text pairs with grammatical errors and disfluencies. Moreover, we propose a series of objective metrics that can be used to evaluate the generated data and choose the more suitable dataset for SGEC. The goal is to generate an augmented dataset that maintains the textual and acoustic characteristics of the original data while providing new types of errors. This augmented dataset should augment and enrich the original corpus without altering the language assessment scores of the second language (L2) learners. We evaluate the use of the augmented corpus both for written GEC (the text part) and for SGEC (the audio-text pairs). Our experiments are conducted on the S\&I Corpus, the first publicly available speech dataset with grammar error annotations. 

**Abstract (ZH)**: 虽然在语法错误修正（GEC）领域存在强大的基准数据集，但高质量的标注口语数据集（SGEC）仍然资源不足。在本文中，我们提出了一种完全自动的方法来生成包含语法错误和话语中断的音频-文本对。此外，我们提出了一系列客观指标，可以用于评估生成的数据，并选择更适合SGEC的数据集。目标是在保留原始数据的文本和声学特征的同时，提供新的错误类型。该扩充数据集应该扩充和丰富原始语料库，而不改变第二语言（L2）学习者的语言评估分数。我们分别评估扩充语料库在书面GEC（文本部分）和SGEC（音频-文本对）中的使用。我们的实验基于S&I语料库，这是第一个公开发布的带有语法错误标注的口语数据集。 

---
# Counterfactual Explanations in Medical Imaging: Exploring SPN-Guided Latent Space Manipulation 

**Title (ZH)**: 医学影像中的反事实解释：探索SPN引导的潜在空间操控 

**Authors**: Julia Siekiera, Stefan Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2507.19368)  

**Abstract**: Artificial intelligence is increasingly leveraged across various domains to automate decision-making processes that significantly impact human lives. In medical image analysis, deep learning models have demonstrated remarkable performance. However, their inherent complexity makes them black box systems, raising concerns about reliability and interpretability. Counterfactual explanations provide comprehensible insights into decision processes by presenting hypothetical "what-if" scenarios that alter model classifications. By examining input alterations, counterfactual explanations provide patterns that influence the decision-making process. Despite their potential, generating plausible counterfactuals that adhere to similarity constraints providing human-interpretable explanations remains a challenge. In this paper, we investigate this challenge by a model-specific optimization approach. While deep generative models such as variational autoencoders (VAEs) exhibit significant generative power, probabilistic models like sum-product networks (SPNs) efficiently represent complex joint probability distributions. By modeling the likelihood of a semi-supervised VAE's latent space with an SPN, we leverage its dual role as both a latent space descriptor and a classifier for a given discrimination task. This formulation enables the optimization of latent space counterfactuals that are both close to the original data distribution and aligned with the target class distribution. We conduct experimental evaluation on the cheXpert dataset. To evaluate the effectiveness of the integration of SPNs, our SPN-guided latent space manipulation is compared against a neural network baseline. Additionally, the trade-off between latent variable regularization and counterfactual quality is analyzed. 

**Abstract (ZH)**: 人工智能在各种领域被越来越多地用于自动化显著影响人类生活的决策过程。在医学图像分析中，深度学习模型展现出了卓越的性能。然而，它们固有的复杂性使得它们成为黑盒系统，引发了可靠性和可解释性的担忧。对比解释通过呈现改变模型分类的假设“如果”情景，提供了对决策过程可理解的洞察。通过检查输入的更改，对比解释揭示了影响决策过程的模式。尽管存在这种潜力，生成符合相似性约束且提供可人类解释的对比解释仍然是一项挑战。在本文中，我们通过模型特定的优化方法来研究这一挑战。虽然生成式深度模型如变分自编码器（VAEs）具有显著的生成能力，但概率模型如和积网络（SPNs）能够高效地表示复杂的联合概率分布。通过使用SPNs建模半监督VAE的潜在空间似然性，我们利用其双重角色作为潜在空间描述符和给定分类任务的分类器进行优化。这种表达方式使得我们可以优化既接近原始数据分布又与目标类分布对齐的潜在空间对比解释。我们在cheXpert数据集上进行了实验评估。为了评估SPNs集成的有效性，我们对比了SPN引导的潜在空间操作与神经网络基线的性能。此外，我们分析了潜在变量正则化与对比解释质量之间的权衡。 

---
# LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences 

**Title (ZH)**: LOTUS：从质量到社会偏见和用户偏好的详细图像 captioning 领域排行榜 

**Authors**: Yusuke Hirota, Boyi Li, Ryo Hachiuma, Yueh-Hua Wu, Boris Ivanovic, Yuta Nakashima, Marco Pavone, Yejin Choi, Yu-Chiang Frank Wang, Chao-Han Huck Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19362)  

**Abstract**: Large Vision-Language Models (LVLMs) have transformed image captioning, shifting from concise captions to detailed descriptions. We introduce LOTUS, a leaderboard for evaluating detailed captions, addressing three main gaps in existing evaluations: lack of standardized criteria, bias-aware assessments, and user preference considerations. LOTUS comprehensively evaluates various aspects, including caption quality (e.g., alignment, descriptiveness), risks (\eg, hallucination), and societal biases (e.g., gender bias) while enabling preference-oriented evaluations by tailoring criteria to diverse user preferences. Our analysis of recent LVLMs reveals no single model excels across all criteria, while correlations emerge between caption detail and bias risks. Preference-oriented evaluations demonstrate that optimal model selection depends on user priorities. 

**Abstract (ZH)**: Large 视觉-语言 模型（LVLMs）已 transform 图像 编辑，从 简洁 的 描述 转向 详细 描述。我们 引入 LOTUS，一个 用于 评估 详细 描述 的 成绩榜，弥补 当前 评估 中 的 三大 缺陷： 缺乏 标准 化 的 标准、有 偏见 的 评估 和 用户 偏好 考虑。LOTUS 全面 评估 各种 方面，包括 描述 质量（如 准确性、 描述性）、风险（如 虚构内容）和社会 偏见（如 性别偏见），同时 通过 定制 标准 来 实现 偏好 导向 的 评估。我们 对 最新 的 LVLMs 的 分析 表明，没有 一种 模型 在 所有 标准 上 都 优越，而 是 描述 详细 程度 和 偏见 风险 之间 存在 联系。偏好 导向 的 评估 表明，最佳 模型 选择 取决于 用户 优先级。 

---
# SpeechIQ: Speech Intelligence Quotient Across Cognitive Levels in Voice Understanding Large Language Models 

**Title (ZH)**: SpeechIQ：语音理解大语言模型 Across 不同认知层次的语音智能商 

**Authors**: Zhen Wan, Chao-Han Huck Yang, Yahan Yu, Jinchuan Tian, Sheng Li, Ke Hu, Zhehuai Chen, Shinji Watanabe, Fei Cheng, Chenhui Chu, Sadao Kurohashi  

**Link**: [PDF](https://arxiv.org/pdf/2507.19361)  

**Abstract**: We introduce Speech-based Intelligence Quotient (SIQ) as a new form of human cognition-inspired evaluation pipeline for voice understanding large language models, LLM Voice, designed to assess their voice understanding ability. Moving beyond popular voice understanding metrics such as word error rate (WER), SIQ examines LLM Voice across three cognitive levels motivated by Bloom's Taxonomy: (1) Remembering (i.e., WER for verbatim accuracy); (2) Understanding (i.e., similarity of LLM's interpretations); and (3) Application (i.e., QA accuracy for simulating downstream tasks). We demonstrate that SIQ not only quantifies voice understanding abilities but also provides unified comparisons between cascaded methods (e.g., ASR LLM) and end-to-end models, identifies annotation errors in existing benchmarks, and detects hallucinations in LLM Voice. Our framework represents a first-of-its-kind intelligence examination that bridges cognitive principles with voice-oriented benchmarks, while exposing overlooked challenges in multi-modal training. 

**Abstract (ZH)**: 基于语音的智能商 quotient (SIQ)：一种受人类认知启发的语音理解评估管道，用于评估大语言模型 LLM Voice 的语音理解能力 

---
# Smooth Reading: Bridging the Gap of Recurrent LLM to Self-Attention LLM on Long-Context Tasks 

**Title (ZH)**: 平滑阅读：连接循环LLM与自注意力LLM在长期上下文任务中的差距 

**Authors**: Kai Liu, Zhan Su, Peijie Dong, Fengran Mo, Jianfei Gao, ShaoTing Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19353)  

**Abstract**: Recently, recurrent large language models (Recurrent LLMs) with linear computational complexity have re-emerged as efficient alternatives to self-attention-based LLMs (Self-Attention LLMs), which have quadratic complexity. However, Recurrent LLMs often underperform on long-context tasks due to their limited fixed-size memory. Previous research has primarily focused on enhancing the memory capacity of Recurrent LLMs through architectural innovations, but these approaches have not yet enabled Recurrent LLMs to match the performance of Self-Attention LLMs on long-context tasks. We argue that this limitation arises because processing the entire context at once is not well-suited for Recurrent LLMs. In this paper, we propose Smooth Reading, a chunk-wise inference method inspired by human reading strategies. Smooth Reading processes context in chunks and iteratively summarizes the contextual information, thereby reducing memory demands and making the approach more compatible with Recurrent LLMs. Our experimental results show that this method substantially narrows the performance gap between Recurrent and Self-Attention LLMs on long-context tasks, while preserving the efficiency advantages of Recurrent LLMs. Our Smooth Reading boosts SWA-3B-4k (a Recurrent LLM) from 5.68% lower to 3.61% higher performance than Self-Attention LLMs on LongBench. Besides, our method maintains the high efficiency, training 3x faster and inferring 2x faster at 64k context compared to Self-Attention LLMs. To our knowledge, this is the first work to achieve comparable performance using Recurrent LLMs compared with Self-Attention LLMs on long-context tasks. We hope our method will inspire future research in this area. To facilitate further progress, we will release code and dataset. 

**Abstract (ZH)**: 最近，具有线性计算复杂度的递归大规模语言模型（递归LLMs）重新成为自注意力机制的大规模语言模型（自注意力LLMs）的有效替代，自注意力LLMs具有二次复杂度。然而，递归LLMs在长上下文任务上通常表现不佳，因为它们具有有限的固定大小内存。之前的研究主要集中在通过架构创新增强递归LLMs的内存容量上，但这些方法仍未使递归LLMs在长上下文任务上的性能达到自注意力LLMs的水平。我们认为，这一限制是因为一次性处理整个上下文不适合递归LLMs。在本文中，我们提出了一种基于分块推断的平滑阅读方法（Smooth Reading），该方法受到人类阅读策略的启发。平滑阅读逐块处理上下文并迭代总结上下文信息，从而降低内存需求，并使该方法更符合递归LLMs的特点。实验结果表明，这种方法在长上下文任务上显著缩小了递归LLMs与自注意力LLMs之间的性能差距，同时保持了递归LLMs的高效性优势。我们的平滑阅读方法将SWA-3B-4k（一个递归LLMs）的性能从相对于自注意力LLMs低5.68%提升到高3.61%。此外，我们的方法保持了高效率，相较于自注意力LLMs，在64k上下文时训练速度快3倍，推理速度快2倍。据我们所知，这是首次在长上下文任务上使递归LLMs的性能与自注意力LLMs接近的研究工作。我们希望我们的方法能够激励该领域的未来研究。为了促进进一步的研究，我们将发布代码和数据集。 

---
# Doubling Your Data in Minutes: Ultra-fast Tabular Data Generation via LLM-Induced Dependency Graphs 

**Title (ZH)**: 几分钟内翻倍你的数据：通过LLM诱导的依赖图实现超快速表格数据生成 

**Authors**: Shuo Yang, Zheyu Zhang, Bardh Prenkaj, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2507.19334)  

**Abstract**: Tabular data is critical across diverse domains, yet high-quality datasets remain scarce due to privacy concerns and the cost of collection. Contemporary approaches adopt large language models (LLMs) for tabular augmentation, but exhibit two major limitations: (1) dense dependency modeling among tabular features that can introduce bias, and (2) high computational overhead in sampling. To address these issues, we propose SPADA for SPArse Dependency-driven Augmentation, a lightweight generative framework that explicitly captures sparse dependencies via an LLM-induced graph. We treat each feature as a node and synthesize values by traversing the graph, conditioning each feature solely on its parent nodes. We explore two synthesis strategies: a non-parametric method using Gaussian kernel density estimation, and a conditional normalizing flow model that learns invertible mappings for conditional density estimation. Experiments on four datasets show that SPADA reduces constraint violations by 4% compared to diffusion-based methods and accelerates generation by nearly 9,500 times over LLM-based baselines. 

**Abstract (ZH)**: 基于稀疏依赖驱动的数据增强：SPADAmissive 

---
# SIDE: Sparse Information Disentanglement for Explainable Artificial Intelligence 

**Title (ZH)**: SIDE: 稀疏信息解耦 für 可解释的人工智能 

**Authors**: Viktar Dubovik, Łukasz Struski, Jacek Tabor, Dawid Rymarczyk  

**Link**: [PDF](https://arxiv.org/pdf/2507.19321)  

**Abstract**: Understanding the decisions made by deep neural networks is essential in high-stakes domains such as medical imaging and autonomous driving. Yet, these models often lack transparency, particularly in computer vision. Prototypical-parts-based neural networks have emerged as a promising solution by offering concept-level explanations. However, most are limited to fine-grained classification tasks, with few exceptions such as InfoDisent. InfoDisent extends prototypical models to large-scale datasets like ImageNet, but produces complex explanations.
We introduce Sparse Information Disentanglement for Explainability (SIDE), a novel method that improves the interpretability of prototypical parts through a dedicated training and pruning scheme that enforces sparsity. Combined with sigmoid activations in place of softmax, this approach allows SIDE to associate each class with only a small set of relevant prototypes. Extensive experiments show that SIDE matches the accuracy of existing methods while reducing explanation size by over $90\%$, substantially enhancing the understandability of prototype-based explanations. 

**Abstract (ZH)**: Sparse Information Disentanglement for Explainability in Prototypical Parts-Based Neural Networks 

---
# Multistream Network for LiDAR and Camera-based 3D Object Detection in Outdoor Scenes 

**Title (ZH)**: 基于 LiDAR 和摄像头的户外场景三维对象检测多流网络 

**Authors**: Muhammad Ibrahim, Naveed Akhtar, Haitian Wang, Saeed Anwar, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2507.19304)  

**Abstract**: Fusion of LiDAR and RGB data has the potential to enhance outdoor 3D object detection accuracy. To address real-world challenges in outdoor 3D object detection, fusion of LiDAR and RGB input has started gaining traction. However, effective integration of these modalities for precise object detection task still remains a largely open problem. To address that, we propose a MultiStream Detection (MuStD) network, that meticulously extracts task-relevant information from both data modalities. The network follows a three-stream structure. Its LiDAR-PillarNet stream extracts sparse 2D pillar features from the LiDAR input while the LiDAR-Height Compression stream computes Bird's-Eye View features. An additional 3D Multimodal stream combines RGB and LiDAR features using UV mapping and polar coordinate indexing. Eventually, the features containing comprehensive spatial, textural and geometric information are carefully fused and fed to a detection head for 3D object detection. Our extensive evaluation on the challenging KITTI Object Detection Benchmark using public testing server at this https URL establishes the efficacy of our method by achieving new state-of-the-art or highly competitive results in different categories while remaining among the most efficient methods. Our code will be released through MuStD GitHub repository at this https URL 

**Abstract (ZH)**: LiDAR和RGB数据融合在提高户外3D目标检测精度方面的潜力：MuStD网络及其应用 

---
# Controlling Topological Defects in Polar Fluids via Reinforcement Learning 

**Title (ZH)**: 通过强化学习控制极性流体中的拓扑缺陷 

**Authors**: Abhinav Singh, Petros Koumoutsakos  

**Link**: [PDF](https://arxiv.org/pdf/2507.19298)  

**Abstract**: Topological defects in active polar fluids exhibit complex dynamics driven by internally generated stresses, reflecting the deep interplay between topology, flow, and non-equilibrium hydrodynamics. Feedback control offers a powerful means to guide such systems, enabling transitions between dynamic states. We investigated closed-loop steering of integer-charged defects in a confined active fluid by modulating the spatial profile of activity. Using a continuum hydrodynamic model, we show that localized control of active stress induces flow fields that can reposition and direct defects along prescribed trajectories by exploiting non-linear couplings in the system. A reinforcement learning framework is used to discover effective control strategies that produce robust defect transport across both trained and novel trajectories. The results highlight how AI agents can learn the underlying dynamics and spatially structure activity to manipulate topological excitations, offering insights into the controllability of active matter and the design of adaptive, self-organized materials. 

**Abstract (ZH)**: 活性极性流体中的拓扑缺陷展示了由内部产生张力驱动的复杂动力学，反映了拓扑、流动和非平衡流体动力学之间深层次的相互作用。反馈控制为引导此类系统提供了一种强大手段，使其能够在不同的动力态状态之间进行转换。我们通过调节活性流的空间分布对被约束的活性流中的整数电荷缺陷进行闭环操纵。通过使用连续流体动力学模型，我们展示了局部控制活性张力可以诱导流场，通过利用系统中的非线性耦合来重定位并引导缺陷沿指定轨迹移动。我们使用强化学习框架来发现有效的控制策略，这些策略可以在训练过的以及新的轨迹上产生稳健的缺陷传输。这些结果突显了人工智能代理如何学习基础动力学并在空间上结构化活性，以操纵拓扑激发，为活性物质的可控性和自适应、自组织材料的设计提供了见解。 

---
# Towards LLM-Enhanced Group Recommender Systems 

**Title (ZH)**: 向基于LLM的群体推荐系统迈进 

**Authors**: Sebastian Lubos, Alexander Felfernig, Thi Ngoc Trang Tran, Viet-Man Le, Damian Garber, Manuel Henrich, Reinhard Willfort, Jeremias Fuchs  

**Link**: [PDF](https://arxiv.org/pdf/2507.19283)  

**Abstract**: In contrast to single-user recommender systems, group recommender systems are designed to generate and explain recommendations for groups. This group-oriented setting introduces additional complexities, as several factors - absent in individual contexts - must be addressed. These include understanding group dynamics (e.g., social dependencies within the group), defining effective decision-making processes, ensuring that recommendations are suitable for all group members, and providing group-level explanations as well as explanations for individual users. In this paper, we analyze in which way large language models (LLMs) can support these aspects and help to increase the overall decision support quality and applicability of group recommender systems. 

**Abstract (ZH)**: 与单用户推荐系统相比，群体推荐系统旨在为群体生成并解释推荐，这种以群体为导向的设置引入了额外的复杂性，需要解决多个在个体背景下不存在的因素，如理解群体动态（例如，群体内的社会依赖关系）、定义有效的决策过程、确保推荐适合所有群体成员，并提供群体层面的解释以及个体用户的解释。在本文中，我们分析大语言模型（LLMs）如何支持这些方面，并有助于提高群体推荐系统的整体决策支持质量和适用性。 

---
# Fine-Tuning Multilingual Language Models for Code Review: An Empirical Study on Industrial C# Projects 

**Title (ZH)**: 基于工业C#项目的大规模多语言模型微调：代码审查实证研究 

**Authors**: Igli Begolli, Meltem Aksoy, Daniel Neider  

**Link**: [PDF](https://arxiv.org/pdf/2507.19271)  

**Abstract**: Code review is essential for maintaining software quality but often time-consuming and cognitively demanding, especially in industrial environments. Recent advancements in language models (LMs) have opened new avenues for automating core review tasks. This study presents the empirical evaluation of monolingual fine-tuning on the performance of open-source LMs across three key automated code review tasks: Code Change Quality Estimation, Review Comment Generation, and Code Refinement. We fine-tuned three distinct models, CodeReviewer, CodeLlama-7B, and DeepSeek-R1-Distill, on a C\# specific dataset combining public benchmarks with industrial repositories. Our study investigates how different configurations of programming languages and natural languages in the training data affect LM performance, particularly in comment generation. Additionally, we benchmark the fine-tuned models against an automated software analysis tool (ASAT) and human reviewers to evaluate their practical utility in real-world settings. Our results show that monolingual fine-tuning improves model accuracy and relevance compared to multilingual baselines. While LMs can effectively support code review workflows, especially for routine or repetitive tasks, human reviewers remain superior in handling semantically complex or context-sensitive changes. Our findings highlight the importance of language alignment and task-specific adaptation in optimizing LMs for automated code review. 

**Abstract (ZH)**: 基于单语微调的开源语言模型在关键自动化代码审查任务中的 empirical 评估：Code Review细粒度任务中的单语微调研究 

---
# A Markov Categorical Framework for Language Modeling 

**Title (ZH)**: 马尔可夫范畴框架语言建模 

**Authors**: Yifan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19247)  

**Abstract**: Auto-regressive language models factorize sequence probabilities and are trained by minimizing the negative log-likelihood (NLL) objective. While empirically powerful, a deep theoretical understanding of why this simple objective yields such versatile representations remains elusive. This work introduces a unifying analytical framework using Markov Categories (MCs) to deconstruct the AR generation process and the NLL objective. We model the single-step generation map as a composition of Markov kernels in the category Stoch. This compositional view, when enriched with statistical divergences, allows us to dissect information flow and learned geometry. Our framework makes three main contributions. First, we provide a formal, information-theoretic rationale for the success of modern speculative decoding methods like EAGLE, quantifying the information surplus in hidden states that these methods exploit. Second, we formalize how NLL minimization forces the model to learn not just the next token, but the data's intrinsic conditional stochasticity, a process we analyze using categorical entropy. Third, and most centrally, we prove that NLL training acts as an implicit form of spectral contrastive learning. By analyzing the information geometry of the model's prediction head, we show that NLL implicitly forces the learned representation space to align with the eigenspectrum of a predictive similarity operator, thereby learning a geometrically structured space without explicit contrastive pairs. This compositional and information-geometric perspective reveals the deep structural principles underlying the effectiveness of modern LMs. Project Page: this https URL 

**Abstract (ZH)**: 自动回归语言模型分解序列概率，并通过最小化负对数似然（NLL）目标进行训练。尽管在实践中表现出色，但这种简单目标为何能产生如此多样的表示形式的深层理论理解仍然缺乏。本文引入了一个统一的分析框架，使用马尔科夫范畴（MCs）分解自动回归生成过程和NLL目标。我们将单步生成映射建模为Stoch范畴中马尔科夫核的组合。这种组合视角，在统计散度增强后，使得我们可以剖析信息流动和学习到的几何结构。我们的框架提供了三个主要贡献。首先，我们为现代投机性解码方法（如EAGLE）的成功提供了形式化的信息论依据，量化了这些方法所利用的隐藏状态中的信息盈余。其次，我们形式化了如何通过最小化NLL迫使模型不仅学习下一个标记，还学习数据的内在条件随机性，我们使用范畴熵对此过程进行了分析。第三，也是最核心的一点，我们证明NLL训练实际上是一种隐式的光谱对比学习形式。通过分析模型预测头的信息几何结构，我们展示了NLL隐式地迫使学习到的表征空间与预测相似性操作的特征谱对齐，从而学习一个几何结构化的空间，而无需显式的对比学习对。这种组合和信息几何视角揭示了现代语言模型有效性的深层结构原则。项目页面：这个 https URL 

---
# Transfinite Fixed Points in Alpay Algebra as Ordinal Game Equilibria in Dependent Type Theory 

**Title (ZH)**: 超越有限的固定点在Alpay代数中的序游戏均衡在依赖类型理论中的应用 

**Authors**: Faruk Alpay, Bugra Kilictas, Taylan Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2507.19245)  

**Abstract**: This paper contributes to the Alpay Algebra by demonstrating that the stable outcome of a self referential process, obtained by iterating a transformation through all ordinal stages, is identical to the unique equilibrium of an unbounded revision dialogue between a system and its environment. The analysis initially elucidates how classical fixed point theorems guarantee such convergence in finite settings and subsequently extends the argument to the transfinite domain, relying upon well founded induction and principles of order theoretic continuity.
Furthermore, the resulting transordinal fixed point operator is embedded into dependent type theory, a formalization which permits every step of the transfinite iteration and its limit to be verified within a modern proof assistant. This procedure yields a machine checked proof that the iterative dialogue necessarily stabilizes and that its limit is unique. The result provides a foundation for Alpay's philosophical claim of semantic convergence within the framework of constructive logic. By unifying concepts from fixed point theory, game semantics, ordinal analysis, and type theory, this research establishes a broadly accessible yet formally rigorous foundation for reasoning about infinite self referential systems and offers practical tools for certifying their convergence within computational environments. 

**Abstract (ZH)**: 本文通过证明自我参照过程通过所有序数阶段迭代变换的稳定结果与无限修正对话中系统与其环境的独特均衡一致，为Alpay代数做出了贡献。分析首先阐明了古典不动点定理在有限设置中确保这种收敛性的方法，随后扩展了该论点到超限领域，依赖于良基归纳和序理论连续性原则。进一步地，超限不动点运算子被嵌入到依赖类型理论中，这一形式化方法允许在现代证明辅助系统中验证超限迭代的每一步及其极限。这一过程产生了机器检查的证明，表明迭代对话必然稳定并且其极限是唯一的。该结果为在构造逻辑框架内实现语义收敛提供了基础。通过统一固定点理论、博弈语义学、序分析和类型理论的概念，这项研究为推理关于无限自参照系统建立了广泛可访问且形式严谨的基础，并提供了在计算环境中认证其收敛性的实用工具。 

---
# Virne: A Comprehensive Benchmark for Deep RL-based Network Resource Allocation in NFV 

**Title (ZH)**: Virne：基于NFV的深度强化学习网络资源分配的综合性基准测试 

**Authors**: Tianfu Wang, Liwei Deng, Xi Chen, Junyang Wang, Huiguo He, Leilei Ding, Wei Wu, Qilin Fan, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.19234)  

**Abstract**: Resource allocation (RA) is critical to efficient service deployment in Network Function Virtualization (NFV), a transformative networking paradigm. Recently, deep Reinforcement Learning (RL)-based methods have been showing promising potential to address this complexity. However, the lack of a systematic benchmarking framework and thorough analysis hinders the exploration of emerging networks and the development of more robust algorithms while causing inconsistent evaluation. In this paper, we introduce Virne, a comprehensive benchmarking framework for the NFV-RA problem, with a focus on supporting deep RL-based methods. Virne provides customizable simulations for diverse network scenarios, including cloud, edge, and 5G environments. It also features a modular and extensible implementation pipeline that supports over 30 methods of various types, and includes practical evaluation perspectives beyond effectiveness, such as scalability, generalization, and scalability. Furthermore, we conduct in-depth analysis through extensive experiments to provide valuable insights into performance trade-offs for efficient implementation and offer actionable guidance for future research directions. Overall, with its diverse simulations, rich implementations, and extensive evaluation capabilities, Virne could serve as a comprehensive benchmark for advancing NFV-RA methods and deep RL applications. The code is publicly available at this https URL. 

**Abstract (ZH)**: 资源分配（RA）是网络功能虚拟化（NFV）中高效服务部署的关键。近年来，基于深度强化学习（RL）的方法显示出解决这一复杂性的潜力。然而，缺乏系统的基准测试框架和深入分析阻碍了新兴网络的探索和更稳健算法的发展，导致评估不一致。本文介绍了Virne，一个全面的NFV-RA基准测试框架，重点支持深度RL方法。Virne提供了多种网络场景的可定制仿真，包括云、边缘和5G环境。它还具有模块化可扩展的实现管道，支持超过30种不同类型的方法，并包含超越有效性的影响因素的实用评估视角，如可扩展性、通用性和公平性。此外，通过广泛的实验进行了深入分析，提供了关于高效实施性能权衡的宝贵见解，并为未来研究方向提供了可操作的指导。总体而言，凭借其多样化的仿真、丰富的实现能力和广泛的评估能力，Virne可以作为一个全面的基准，推动NFV-RA方法和深度RL应用的发展。代码已公开。 

---
# Joint Holistic and Lesion Controllable Mammogram Synthesis via Gated Conditional Diffusion Model 

**Title (ZH)**: 基于门控条件性扩散模型的乳腺X线图像整体与病灶可控合成 

**Authors**: Xin Li, Kaixiang Yang, Qiang Li, Zhiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19201)  

**Abstract**: Mammography is the most commonly used imaging modality for breast cancer screening, driving an increasing demand for deep-learning techniques to support large-scale analysis. However, the development of accurate and robust methods is often limited by insufficient data availability and a lack of diversity in lesion characteristics. While generative models offer a promising solution for data synthesis, current approaches often fail to adequately emphasize lesion-specific features and their relationships with surrounding tissues. In this paper, we propose Gated Conditional Diffusion Model (GCDM), a novel framework designed to jointly synthesize holistic mammogram images and localized lesions. GCDM is built upon a latent denoising diffusion framework, where the noised latent image is concatenated with a soft mask embedding that represents breast, lesion, and their transitional regions, ensuring anatomical coherence between them during the denoising process. To further emphasize lesion-specific features, GCDM incorporates a gated conditioning branch that guides the denoising process by dynamically selecting and fusing the most relevant radiomic and geometric properties of lesions, effectively capturing their interplay. Experimental results demonstrate that GCDM achieves precise control over small lesion areas while enhancing the realism and diversity of synthesized mammograms. These advancements position GCDM as a promising tool for clinical applications in mammogram synthesis. Our code is available at this https URL 

**Abstract (ZH)**: 基于门控条件扩散模型的全景乳腺X线图像及局部病灶合成方法 

---
# Enhancing Diabetic Retinopathy Classification Accuracy through Dual Attention Mechanism in Deep Learning 

**Title (ZH)**: 通过深度学习中的双重注意力机制提高糖尿病视网膜病变分类准确性 

**Authors**: Abdul Hannan, Zahid Mahmood, Rizwan Qureshi, Hazrat Ali  

**Link**: [PDF](https://arxiv.org/pdf/2507.19199)  

**Abstract**: Automatic classification of Diabetic Retinopathy (DR) can assist ophthalmologists in devising personalized treatment plans, making it a critical component of clinical practice. However, imbalanced data distribution in the dataset becomes a bottleneck in the generalization of deep learning models trained for DR classification. In this work, we combine global attention block (GAB) and category attention block (CAB) into the deep learning model, thus effectively overcoming the imbalanced data distribution problem in DR classification. Our proposed approach is based on an attention mechanism-based deep learning model that employs three pre-trained networks, namely, MobileNetV3-small, Efficientnet-b0, and DenseNet-169 as the backbone architecture. We evaluate the proposed method on two publicly available datasets of retinal fundoscopy images for DR. Experimental results show that on the APTOS dataset, the DenseNet-169 yielded 83.20% mean accuracy, followed by the MobileNetV3-small and EfficientNet-b0, which yielded 82% and 80% accuracies, respectively. On the EYEPACS dataset, the EfficientNet-b0 yielded a mean accuracy of 80%, while the DenseNet-169 and MobileNetV3-small yielded 75.43% and 76.68% accuracies, respectively. In addition, we also compute the F1-score of 82.0%, precision of 82.1%, sensitivity of 83.0%, specificity of 95.5%, and a kappa score of 88.2% for the experiments. Moreover, in our work, the MobileNetV3-small has 1.6 million parameters on the APTOS dataset and 0.90 million parameters on the EYEPACS dataset, which is comparatively less than other methods. The proposed approach achieves competitive performance that is at par with recently reported works on DR classification. 

**Abstract (ZH)**: 自动分类糖尿病视网膜病变（DR）可以协助眼科医生制定个性化治疗方案，使其成为临床实践中的关键组成部分。然而，数据集中的不平衡数据分布成为深度学习模型在DR分类中泛化的瓶颈。在本文中，我们将全局注意力模块（GAB）和类别注意力模块（CAB）结合到深度学习模型中，从而有效克服了DR分类中的不平衡数据分布问题。我们提出的方法基于一种基于注意力机制的深度学习模型，采用MobileNetV3-small、EfficientNet-b0和DenseNet-169三种预训练网络作为骨干架构。我们在两个公开的眼底图像DR数据集上评估了所提出的方法。实验结果表明，在APTO斯数据集上，DenseNet-169的平均准确率为83.20%，其次是MobileNetV3-small和EfficientNet-b0，准确率分别为82%和80%。在EYEPACS数据集上，EfficientNet-b0的平均准确率为80%，而DenseNet-169和MobileNetV3-small的准确率分别为75.43%和76.68%。此外，我们还计算了F1分数为82.0%、精度为82.1%、灵敏度为83.0%、特异度为95.5%、κ系数为88.2%。此外，在我们的工作中，MobileNetV3-small在APTO斯数据集上的参数量为1.6百万，在EYEPACS数据集上的参数量为0.90百万，相较于其他方法较少。所提出的方法在DR分类中实现了竞争力的性能，与最近报道的DR分类方法相当。 

---
# WACA-UNet: Weakness-Aware Channel Attention for Static IR Drop Prediction in Integrated Circuit Design 

**Title (ZH)**: WACA-UNet：感知弱点的通道注意力在集成电路设计中静态IR压降预测 

**Authors**: Youngmin Seo, Yunhyeong Kwon, Younghun Park, HwiRyong Kim, Seungho Eum, Jinha Kim, Taigon Song, Juho Kim, Unsang Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.19197)  

**Abstract**: Accurate spatial prediction of power integrity issues, such as IR drop, is critical for reliable VLSI design. However, traditional simulation-based solvers are computationally expensive and difficult to scale. We address this challenge by reformulating IR drop estimation as a pixel-wise regression task on heterogeneous multi-channel physical maps derived from circuit layouts. Prior learning-based methods treat all input layers (e.g., metal, via, and current maps) equally, ignoring their varying importance to prediction accuracy. To tackle this, we propose a novel Weakness-Aware Channel Attention (WACA) mechanism, which recursively enhances weak feature channels while suppressing over-dominant ones through a two-stage gating strategy. Integrated into a ConvNeXtV2-based attention U-Net, our approach enables adaptive and balanced feature representation. On the public ICCAD-2023 benchmark, our method outperforms the ICCAD-2023 contest winner by reducing mean absolute error by 61.1% and improving F1-score by 71.0%. These results demonstrate that channel-wise heterogeneity is a key inductive bias in physical layout analysis for VLSI. 

**Abstract (ZH)**: 准确的空间预测对于VLSI设计中的电源完整性问题，如IR跌落，至关重要。然而，传统的基于仿真的求解器计算成本高昂且难以扩展。我们通过将IR跌落估计重新表述为由电路布局衍生的异构多通道物理图上的像素级回归任务，来应对这一挑战。基于先前的学习方法对待所有输入层（例如，金属、通孔和电流图）一视同仁，忽视了它们对预测准确性的不同重要性。为此，我们提出了一种新型的弱项感知通道注意（WACA）机制，通过两阶段门控策略递归增强弱特征通道并抑制过分主导的通道。将该机制整合到ConvNeXtV2为基础的注意力U-Net中，我们的方法能够实现自适应且均衡的特征表示。在公开的ICCAD-2023基准测试上，我们的方法通过将平均绝对误差降低61.1%并且提高F1分数71.0%，超过了ICCAD-2023竞赛的获胜者。这些结果表明，在VLSI的实际布局分析中，通道间的异质性是关键的归纳偏置。 

---
# Can Small-Scale Data Poisoning Exacerbate Dialect-Linked Biases in Large Language Models? 

**Title (ZH)**: 小规模数据毒性会加剧大规模语言模型中的方言关联偏见吗？ 

**Authors**: Chaymaa Abbas, Mariette Awad, Razane Tajeddine  

**Link**: [PDF](https://arxiv.org/pdf/2507.19195)  

**Abstract**: Despite the ongoing improvements in the design of large language models (LLMs) to foster inclusion and balanced responses, these systems remain susceptible to encoding and amplifying social biases. This study examines how dialectal variation, specifically African American Vernacular English (AAVE) versus Standard American English (SAE), interacts with data poisoning to influence toxicity in outputs. Using both small- and medium-scale LLaMA models, we show that even minimal exposure to poisoned data significantly increases toxicity for AAVE inputs, while it remains comparatively unaffected for SAE. Larger models exhibit a more significant amplification effect which suggests heightened susceptibility with scale. To further assess these disparities, we employed GPT-4o as a fairness auditor, which identified harmful stereotypical patterns disproportionately tied to AAVE inputs, including portrayals of aggression, criminality, and intellectual inferiority. These findings underscore the compounding impact of data poisoning and dialectal bias and emphasize the need for dialect-aware evaluation, targeted debiasing interventions, and socially responsible training protocols during development. 

**Abstract (ZH)**: 尽管在设计大型语言模型（LLMs）以促进包容性和均衡响应方面取得了持续改进，这些系统仍然容易编码和放大社会偏见。本研究探讨了口音变体，特别是美式黑人方言（AAVE）与标准美式英语（SAE）在数据污染影响下的毒性输出差异。使用小型和中型规模的LLaMA模型，我们表明，即使是少量接触污染数据也会显著增加AAVE输入的毒性，而对SAE的影响相对较小。更大规模的模型表现出更显著的放大效应，这表明随规模增大，其易感性增加。为进一步评估这些差异，我们采用了GPT-4o作为公平性审计员，它识别出与AAVE输入不成比例关联的有害刻板印象模式，包括表现为攻击性、犯罪性和智力 inferiority。这些发现强调了数据污染和口音偏见累积影响的重要性，并强调了需要具备口音意识的评估、针对性的去偏措施和社会负责任的训练协议。 

---
# PrompTrend: Continuous Community-Driven Vulnerability Discovery and Assessment for Large Language Models 

**Title (ZH)**: PrompTrend: 大型语言模型持续社区驱动的漏洞发现与评估 

**Authors**: Tarek Gasmi, Ramzi Guesmi, Mootez Aloui, Jihene Bennaceur  

**Link**: [PDF](https://arxiv.org/pdf/2507.19185)  

**Abstract**: Static benchmarks fail to capture LLM vulnerabilities emerging through community experimentation in online forums. We present PrompTrend, a system that collects vulnerability data across platforms and evaluates them using multidimensional scoring, with an architecture designed for scalable monitoring. Cross-sectional analysis of 198 vulnerabilities collected from online communities over a five-month period (January-May 2025) and tested on nine commercial models reveals that advanced capabilities correlate with increased vulnerability in some architectures, psychological attacks significantly outperform technical exploits, and platform dynamics shape attack effectiveness with measurable model-specific patterns. The PrompTrend Vulnerability Assessment Framework achieves 78% classification accuracy while revealing limited cross-model transferability, demonstrating that effective LLM security requires comprehensive socio-technical monitoring beyond traditional periodic assessment. Our findings challenge the assumption that capability advancement improves security and establish community-driven psychological manipulation as the dominant threat vector for current language models. 

**Abstract (ZH)**: 静态基准无法捕捉通过在线论坛社区实验新兴的LLM漏洞。我们提出PrompTrend系统，该系统跨平台收集漏洞数据，并使用多维评分进行评估，具有可扩展监控架构。五个月内（2025年1月至5月）从在线社区收集的198个漏洞和九种商业模型测试结果显示，高级能力与某些架构中的漏洞增加相关，心理攻击显著优于技术exploits，平台动态塑造攻击效果，存在可衡量的模型特定模式。PrompTrend漏洞评估框架在分类准确性方面达到78%，同时揭示了有限的跨模型可移植性，表明有效的LLM安全需要超越传统周期性评估的全面社会技术监控。我们的研究结果挑战了能力提升改善安全性的假设，并确立了社区驱动的心理操纵作为当前语言模型主要威胁向量的地位。 

---
# An Empirical Investigation of Gender Stereotype Representation in Large Language Models: The Italian Case 

**Title (ZH)**: 大型语言模型中性别刻板印象表征的实证研究：意大利案例 

**Authors**: Gioele Giachino, Marco Rondina, Antonio Vetrò, Riccardo Coppola, Juan Carlos De Martin  

**Link**: [PDF](https://arxiv.org/pdf/2507.19156)  

**Abstract**: The increasing use of Large Language Models (LLMs) in a large variety of domains has sparked worries about how easily they can perpetuate stereotypes and contribute to the generation of biased content. With a focus on gender and professional bias, this work examines in which manner LLMs shape responses to ungendered prompts, contributing to biased outputs. This analysis uses a structured experimental method, giving different prompts involving three different professional job combinations, which are also characterized by a hierarchical relationship. This study uses Italian, a language with extensive grammatical gender differences, to highlight potential limitations in current LLMs' ability to generate objective text in non-English languages. Two popular LLM-based chatbots are examined, namely OpenAI ChatGPT (gpt-4o-mini) and Google Gemini (gemini-1.5-flash). Through APIs, we collected a range of 3600 responses. The results highlight how content generated by LLMs can perpetuate stereotypes. For example, Gemini associated 100% (ChatGPT 97%) of 'she' pronouns to the 'assistant' rather than the 'manager'. The presence of bias in AI-generated text can have significant implications in many fields, such as in the workplaces or in job selections, raising ethical concerns about its use. Understanding these risks is pivotal to developing mitigation strategies and assuring that AI-based systems do not increase social inequalities, but rather contribute to more equitable outcomes. Future research directions include expanding the study to additional chatbots or languages, refining prompt engineering methods or further exploiting a larger experimental base. 

**Abstract (ZH)**: 大型语言模型在多样化领域的广泛应用引发了对其容易延续刻板印象和生成有偏见内容的担忧。本文重点关注性别和职业偏见，研究大型语言模型如何通过反应无性别提示来塑造有偏见的输出。该分析采用结构化的实验方法，通过提供涉及三种不同职业职位组合的不同提示，其中这些职位组合也具有层级关系。本研究使用具有广泛语法性别差异的意大利语，以突出当前大型语言模型在非英语语言中生成客观文本方面的潜在限制。研究考察了两个流行的基于大型语言模型的聊天机器人，即OpenAI ChatGPT (gpt-4o-mini)和Google Gemini (gemini-1.5-flash)。通过API，我们收集了约3600个响应结果。研究结果表明，由大型语言模型生成的内容可以延续刻板印象。例如，Gemini有100%（ChatGPT为97%）的机会将“她”的代词与“助手”而非“经理”相关联。AI生成文本中的偏见可能在许多领域，如工作场所或职位选择中产生重大影响，从而引发对其使用的伦理担忧。理解这些风险对于开发缓解策略并确保基于人工智能的系统不增加社会不平等，而是贡献更公平的结果至关重要。未来的研究方向包括扩大研究范围至其他聊天机器人或语言、改进提示工程方法或进一步利用更大的实验基础。 

---
# ReCoDe: Reinforcement Learning-based Dynamic Constraint Design for Multi-Agent Coordination 

**Title (ZH)**: ReCoDe: 基于强化学习的多Agent协调动态约束设计 

**Authors**: Michael Amir, Guang Yang, Zhan Gao, Keisuke Okumura, Heedo Woo, Amanda Prorok  

**Link**: [PDF](https://arxiv.org/pdf/2507.19151)  

**Abstract**: Constraint-based optimization is a cornerstone of robotics, enabling the design of controllers that reliably encode task and safety requirements such as collision avoidance or formation adherence. However, handcrafted constraints can fail in multi-agent settings that demand complex coordination. We introduce ReCoDe--Reinforcement-based Constraint Design--a decentralized, hybrid framework that merges the reliability of optimization-based controllers with the adaptability of multi-agent reinforcement learning. Rather than discarding expert controllers, ReCoDe improves them by learning additional, dynamic constraints that capture subtler behaviors, for example, by constraining agent movements to prevent congestion in cluttered scenarios. Through local communication, agents collectively constrain their allowed actions to coordinate more effectively under changing conditions. In this work, we focus on applications of ReCoDe to multi-agent navigation tasks requiring intricate, context-based movements and consensus, where we show that it outperforms purely handcrafted controllers, other hybrid approaches, and standard MARL baselines. We give empirical (real robot) and theoretical evidence that retaining a user-defined controller, even when it is imperfect, is more efficient than learning from scratch, especially because ReCoDe can dynamically change the degree to which it relies on this controller. 

**Abstract (ZH)**: 基于强化学习的约束设计：ReCoDe在多机器人导航任务中的应用 

---
# Solar Photovoltaic Assessment with Large Language Model 

**Title (ZH)**: 大规模语言模型评估太阳能光伏sistems 

**Authors**: Muhao Guo, Yang Weng  

**Link**: [PDF](https://arxiv.org/pdf/2507.19144)  

**Abstract**: Accurate detection and localization of solar photovoltaic (PV) panels in satellite imagery is essential for optimizing microgrids and active distribution networks (ADNs), which are critical components of renewable energy systems. Existing methods lack transparency regarding their underlying algorithms or training datasets, rely on large, high-quality PV training data, and struggle to generalize to new geographic regions or varied environmental conditions without extensive re-training. These limitations lead to inconsistent detection outcomes, hindering large-scale deployment and data-driven grid optimization. In this paper, we investigate how large language models (LLMs) can be leveraged to overcome these challenges. Despite their promise, LLMs face several challenges in solar panel detection, including difficulties with multi-step logical processes, inconsistent output formatting, frequent misclassification of visually similar objects (e.g., shadows, parking lots), and low accuracy in complex tasks such as spatial localization and quantification. To overcome these issues, we propose the PV Assessment with LLMs (PVAL) framework, which incorporates task decomposition for more efficient workflows, output standardization for consistent and scalable formatting, few-shot prompting to enhance classification accuracy, and fine-tuning using curated PV datasets with detailed annotations. PVAL ensures transparency, scalability, and adaptability across heterogeneous datasets while minimizing computational overhead. By combining open-source accessibility with robust methodologies, PVAL establishes an automated and reproducible pipeline for solar panel detection, paving the way for large-scale renewable energy integration and optimized grid management. 

**Abstract (ZH)**: 基于大型语言模型的太阳能光伏板检测与定位框架：提高微电网和主动配电网的优化能力 

---
# Assessment of Personality Dimensions Across Situations Using Conversational Speech 

**Title (ZH)**: 基于对话语言评估人格维度在不同情境下的表现 

**Authors**: Alice Zhang, Skanda Muralidhar, Daniel Gatica-Perez, Mathew Magimai-Doss  

**Link**: [PDF](https://arxiv.org/pdf/2507.19137)  

**Abstract**: Prior research indicates that users prefer assistive technologies whose personalities align with their own. This has sparked interest in automatic personality perception (APP), which aims to predict an individual's perceived personality traits. Previous studies in APP have treated personalities as static traits, independent of context. However, perceived personalities can vary by context and situation as shown in psychological research. In this study, we investigate the relationship between conversational speech and perceived personality for participants engaged in two work situations (a neutral interview and a stressful client interaction). Our key findings are: 1) perceived personalities differ significantly across interactions, 2) loudness, sound level, and spectral flux features are indicative of perceived extraversion, agreeableness, conscientiousness, and openness in neutral interactions, while neuroticism correlates with these features in stressful contexts, 3) handcrafted acoustic features and non-verbal features outperform speaker embeddings in inference of perceived personality, and 4) stressful interactions are more predictive of neuroticism, aligning with existing psychological research. 

**Abstract (ZH)**: 先前的研究表明，用户更偏好与自己个性相匹配的辅助技术。这激发了自动个性感知（APP）的兴趣，其目标是预测个体感知到的个性特征。以往在APP领域的研究将个性视为静态特征，与情境无关。然而，心理研究显示，感知到的个性会因情境和情况而异。本研究探讨了参与两种工作情境（中性面试和压力大的客户互动）的人员的对话口语与其感知个性之间的关系。我们的主要发现为：1) 不同互动中的感知个性存在显著差异；2) 在中性互动中，响度、声音级别和频谱变化特性与感知的外向性、宜人性、严谨性和开放性相关，而在压力情境中，这些特性与神经质相关；3) 手工设计的声学特征和非语言特征在感知个性的推理中优于说话人嵌入；4) 压力大的互动更能预测神经质，与现有的心理研究结果一致。 

---
# PatchTraj: Dynamic Patch Representation Learning for Time-Frequency Trajectory Prediction 

**Title (ZH)**: PatchTraj：时间-频率轨迹预测的动态补丁表示学习 

**Authors**: Yanghong Liu, Xingping Dong, Ming Li, Weixing Zhang, Yidong Lou  

**Link**: [PDF](https://arxiv.org/pdf/2507.19119)  

**Abstract**: Pedestrian trajectory prediction is crucial for autonomous driving and robotics. While existing point-based and grid-based methods expose two key limitations: insufficiently modeling human motion dynamics, as they fail to balance local motion details with long-range spatiotemporal dependencies, and the time representation lacks interaction with the frequency domain in modeling trajectory sequences. To address these challenges, we propose PatchTraj, a dynamic patch-based trajectory prediction framework that unifies time-domain and frequency-domain representations. Specifically, we decompose the trajectory into raw time sequences and frequency components, employing dynamic patch partitioning for multi-scale trajectory segmentation to capture hierarchical motion patterns. Each patch is processed by an adaptive embedding layer with scale-aware feature extraction, followed by hierarchical feature aggregation to model both fine-grained and long-range dependencies. The outputs of two branches interact via cross-modal attention, enabling complementary fusion of temporal and spectral cues. Finally, a Transformer encoder-decoder integrates both modalities to autoregressively predict future trajectories. Extensive experiments on ETH-UCY, SDD, NBA, and JRDB datasets demonstrate that our method achieves state-of-the-art performance with high efficiency. 

**Abstract (ZH)**: 行人轨迹预测对于自动驾驶和机器人技术至关重要。为了解决现有基于点和基于网格方法的两个关键限制，即无法有效地建模人类运动动力学，以及时间表示在建模轨迹序列时缺乏与频域的交互，我们提出了PatchTraj，这是一种统一时间域和频域表示的动力学斑块基轨迹预测框架。具体地，我们将轨迹分解为原始时间序列和频域分量，并通过动态斑块分割进行多尺度轨迹分割以捕捉层级运动模式。每个斑块通过一种尺度感知的特征提取的自适应嵌入层进行处理，随后通过层级特征聚合来建模精细粒度和长距离依赖性。来自两个分支的输出通过跨模态注意机制相互作用，使时间和频谱线索实现互补融合。最后，Transformer编码器-解码器整合这两种模态以自回归方式预测未来的轨迹。在ETH-UCY、SDD、NBA和JRDB数据集上的大量实验表明，我们的方法实现了最先进的性能，高效性高。 

---
# Graph Structure Learning with Privacy Guarantees for Open Graph Data 

**Title (ZH)**: 带有隐私保证的开放图数据图形结构学习 

**Authors**: Muhao Guo, Jiaqi Wu, Yang Weng, Yizheng Liao, Shengzhe Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19116)  

**Abstract**: Ensuring privacy in large-scale open datasets is increasingly challenging under regulations such as the General Data Protection Regulation (GDPR). While differential privacy (DP) provides strong theoretical guarantees, it primarily focuses on noise injection during model training, neglecting privacy preservation at the data publishing stage. Existing privacy-preserving data publishing (PPDP) approaches struggle to balance privacy and utility, particularly when data publishers and users are distinct entities. To address this gap, we focus on the graph recovery problem and propose a novel privacy-preserving estimation framework for open graph data, leveraging Gaussian DP (GDP) with a structured noise-injection mechanism. Unlike traditional methods that perturb gradients or model updates, our approach ensures unbiased graph structure recovery while enforcing DP at the data publishing stage. Moreover, we provide theoretical guarantees on estimation accuracy and extend our method to discrete-variable graphs, a setting often overlooked in DP research. Experimental results in graph learning demonstrate robust performance, offering a viable solution for privacy-conscious graph analysis. 

**Abstract (ZH)**: 在大规模开放数据集下确保隐私越来越具有挑战性，尤其是在《通用数据保护条例》（GDPR）等法规的约束下。虽然差分隐私（DP）提供了强大的理论保证，但它主要关注模型训练过程中的噪声注入，忽略了数据发布阶段的隐私保护。现有的隐私保护数据发布（PPDP）方法在平衡隐私和实用性方面存在困难，尤其是在数据发布者和用户是不同实体的情况下。为解决这一问题，我们关注图恢复问题，并提出一种利用结构化噪声注入机制的高斯差分隐私（GDP）的新型隐私保护估计框架。与传统方法扰动梯度或模型更新不同，我们的方法确保在数据发布阶段实现无偏的图结构恢复。此外，我们提供了估计准确性的理论保证，并将方法扩展到离散变量图，这是差分隐私研究中常被忽视的领域。在图学习实验中，我们的方法表现出 robust 性能，为隐私意识强烈的图分析提供了可行的解决方案。 

---
# Automated Code Review Using Large Language Models at Ericsson: An Experience Report 

**Title (ZH)**: 爱立信中使用大型语言模型进行自动化代码审查的经验报告 

**Authors**: Shweta Ramesh, Joy Bose, Hamender Singh, A K Raghavan, Sujoy Roychowdhury, Giriprasad Sridhara, Nishrith Saini, Ricardo Britto  

**Link**: [PDF](https://arxiv.org/pdf/2507.19115)  

**Abstract**: Code review is one of the primary means of assuring the quality of released software along with testing and static analysis. However, code review requires experienced developers who may not always have the time to perform an in-depth review of code. Thus, automating code review can help alleviate the cognitive burden on experienced software developers allowing them to focus on their primary activities of writing code to add new features and fix bugs. In this paper, we describe our experience in using Large Language Models towards automating the code review process in Ericsson. We describe the development of a lightweight tool using LLMs and static program analysis. We then describe our preliminary experiments with experienced developers in evaluating our code review tool and the encouraging results. 

**Abstract (ZH)**: 使用大型语言模型自动化代码审查过程的经验：在爱立信的应用 

---
# Distilling a Small Utility-Based Passage Selector to Enhance Retrieval-Augmented Generation 

**Title (ZH)**: 基于实用主义的片段选择模型凝缩以增强检索增强生成 

**Authors**: Hengran Zhang, Keping Bi, Jiafeng Guo, Jiaming Zhang, Shuaiqiang Wang, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.19102)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating retrieved information. Standard retrieval process prioritized relevance, focusing on topical alignment between queries and passages. In contrast, in RAG, the emphasis has shifted to utility, which considers the usefulness of passages for generating accurate answers. Despite empirical evidence showing the benefits of utility-based retrieval in RAG, the high computational cost of using LLMs for utility judgments limits the number of passages evaluated. This restriction is problematic for complex queries requiring extensive information. To address this, we propose a method to distill the utility judgment capabilities of LLMs into smaller, more efficient models. Our approach focuses on utility-based selection rather than ranking, enabling dynamic passage selection tailored to specific queries without the need for fixed thresholds. We train student models to learn pseudo-answer generation and utility judgments from teacher LLMs, using a sliding window method that dynamically selects useful passages. Our experiments demonstrate that utility-based selection provides a flexible and cost-effective solution for RAG, significantly reducing computational costs while improving answer quality. We present the distillation results using Qwen3-32B as the teacher model for both relevance ranking and utility-based selection, distilled into RankQwen1.7B and UtilityQwen1.7B. Our findings indicate that for complex questions, utility-based selection is more effective than relevance ranking in enhancing answer generation performance. We will release the relevance ranking and utility-based selection annotations for the MS MARCO dataset, supporting further research in this area. 

**Abstract (ZH)**: 基于检索的生成（RAG）通过引入检索信息增强了大规模语言模型（LLMs）。标准的检索过程侧重于相关性，关注查询和段落之间的主题对齐。相比之下，在RAG中，重点已经转向了实用性，考虑的是段落对于生成准确答案的有用性。尽管实证证据表明基于实用性检索在RAG中的优势，但在LLMs上进行实用性判断的高计算成本限制了评估的段落数量。这种限制对于需要大量信息的复杂查询来说是个问题。为了解决这一问题，我们提出了一种方法，将LLMs的实用性判断能力提炼到更小、更高效的模型中。我们的方法侧重于实用性选择而不是排序，能够根据特定查询动态选择有用的段落，无需固定阈值。我们训练学生模型从教师LLMs中学习伪答案生成和实用性判断，并使用滑动窗口方法动态选择有用的段落。我们的实验表明，基于实用性的选择为RAG提供了一种灵活且经济高效的解决方案，显著降低了计算成本并提高了答案质量。我们使用Qwen3-32B作为教师模型，进行了相关排序和基于实用性的选择的提炼，分别得到RankQwen1.7B和UtilityQwen1.7B。我们的研究结果表明，对于复杂问题，基于实用性的选择比基于相关性的排序更能提高答案生成性能。我们将发布MS MARCO数据集的相关排序和基于实用性的选择标注，以支持该领域的进一步研究。 

---
# MedSymmFlow: Bridging Generative Modeling and Classification in Medical Imaging through Symmetrical Flow Matching 

**Title (ZH)**: MedSymmFlow: 结合医学影像生成建模和分类的对称流匹配方法 

**Authors**: Francisco Caetano, Lemar Abdi, Christiaan Viviers, Amaan Valiuddin, Fons van der Sommen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19098)  

**Abstract**: Reliable medical image classification requires accurate predictions and well-calibrated uncertainty estimates, especially in high-stakes clinical settings. This work presents MedSymmFlow, a generative-discriminative hybrid model built on Symmetrical Flow Matching, designed to unify classification, generation, and uncertainty quantification in medical imaging. MedSymmFlow leverages a latent-space formulation that scales to high-resolution inputs and introduces a semantic mask conditioning mechanism to enhance diagnostic relevance. Unlike standard discriminative models, it naturally estimates uncertainty through its generative sampling process. The model is evaluated on four MedMNIST datasets, covering a range of modalities and pathologies. The results show that MedSymmFlow matches or exceeds the performance of established baselines in classification accuracy and AUC, while also delivering reliable uncertainty estimates validated by performance improvements under selective prediction. 

**Abstract (ZH)**: 可靠的医学图像分类需要准确的预测和良好的不确定性估计，特别是在高风险临床环境中。本文提出了基于对称流匹配的生成-判别混合模型MedSymmFlow，旨在统一医学成像中的分类、生成和不确定性量化。MedSymmFlow 利用一个可扩展到高分辨率输入的潜在空间表示，并引入语义掩码条件机制以增强诊断相关性。与标准判别模型不同，它通过生成采样过程自然地估计不确定性。该模型在四个MedMNIST数据集上进行评估，涵盖了多种成像模态和病理类型。结果显示，MedSymmFlow 在分类准确性和AUC方面与现有基准相当或超过基准，并且通过在选择性预测下性能改善提供了可靠的不确定性估计验证。 

---
# PBiLoss: Popularity-Aware Regularization to Improve Fairness in Graph-Based Recommender Systems 

**Title (ZH)**: PBiLoss：基于流行度的正则化以提高图推荐系统中的公平性 

**Authors**: Mohammad Naeimi, Mostafa Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2507.19067)  

**Abstract**: Recommender systems, especially those based on graph neural networks (GNNs), have achieved remarkable success in capturing user-item interaction patterns. However, they remain susceptible to popularity bias--the tendency to over-recommend popular items--resulting in reduced content diversity and compromised fairness. In this paper, we propose PBiLoss, a novel regularization-based loss function designed to counteract popularity bias in graph-based recommender models explicitly. PBiLoss augments traditional training objectives by penalizing the model's inclination toward popular items, thereby encouraging the recommendation of less popular but potentially more personalized content. We introduce two sampling strategies: Popular Positive (PopPos) and Popular Negative (PopNeg), which respectively modulate the contribution of the positive and negative popular items during training. We further explore two methods to distinguish popular items: one based on a fixed popularity threshold and another without any threshold, making the approach flexible and adaptive. Our proposed method is model-agnostic and can be seamlessly integrated into state-of-the-art graph-based frameworks such as LightGCN and its variants. Comprehensive experiments across multiple real-world datasets demonstrate that PBiLoss significantly improves fairness, as demonstrated by reductions in the Popularity-Rank Correlation for Users (PRU) and Popularity-Rank Correlation for Items (PRI), while maintaining or even enhancing standard recommendation accuracy and ranking metrics. These results highlight the effectiveness of directly embedding fairness objectives into the optimization process, providing a practical and scalable solution for balancing accuracy and equitable content exposure in modern recommender systems. 

**Abstract (ZH)**: 基于图神经网络的推荐系统通过捕捉用户-项交互模式取得了显著成功，但它们仍然容易受到流行性偏见的影响——即过度推荐流行项的倾向，这导致内容多样性降低和公平性受损。本文提出了一种新颖的正则化损失函数PBiLoss，该函数旨在明确对抗基于图的推荐模型中的流行性偏见。PBiLoss通过惩罚模型对流行项的偏好，来鼓励推荐较少流行但可能更具个性化的内容。我们引入了两种采样策略：Popular Positive (PopPos)和Popular Negative (PopNeg)，分别调节正向和负向社会流行项在训练中的贡献。此外，我们探讨了两种区分社会流行项的方法：一种基于固定流行度阈值，另一种没有阈值，使方法更加灵活和适应性强。所提方法具有模型独立性，可以无缝集成到LightGCN及其变体等最先进的基于图的框架中。在多个真实世界数据集上的全面实验表明，PBiLoss在降低用户和项的流行性排名相关性（PRU和PRI）方面显著提高了公平性，同时保持或甚至提升了推荐准确性和排名指标。这些结果突显了直接将公平性目标嵌入优化过程的有效性，为在现代推荐系统中平衡准确性和公平内容曝光提供了一种实用且可扩展的解决方案。 

---
# Closing the Modality Gap for Mixed Modality Search 

**Title (ZH)**: 跨模态搜索中的模态差距闭合 

**Authors**: Binxu Li, Yuhui Zhang, Xiaohan Wang, Weixin Liang, Ludwig Schmidt, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2507.19054)  

**Abstract**: Mixed modality search -- retrieving information across a heterogeneous corpus composed of images, texts, and multimodal documents -- is an important yet underexplored real-world application. In this work, we investigate how contrastive vision-language models, such as CLIP, perform on the mixed modality search task. Our analysis reveals a critical limitation: these models exhibit a pronounced modality gap in the embedding space, where image and text embeddings form distinct clusters, leading to intra-modal ranking bias and inter-modal fusion failure. To address this issue, we propose GR-CLIP, a lightweight post-hoc calibration method that removes the modality gap in CLIP's embedding space. Evaluated on MixBench -- the first benchmark specifically designed for mixed modality search -- GR-CLIP improves NDCG@10 by up to 26 percentage points over CLIP, surpasses recent vision-language generative embedding models by 4 percentage points, while using 75x less compute. 

**Abstract (ZH)**: 混合模态搜索——在包含图像、文本和多模态文档的异构语料库中检索信息——是一项重要但尚未充分探索的实际应用。在本文中，我们研究了对比视觉-语言模型（如CLIP）在混合模态搜索任务中的表现。我们的分析揭示了一个关键局限：这些模型在嵌入空间中表现出显著的模态差距，图像嵌入和文本嵌入形成不同的簇，导致模内排名偏差和跨模态融合失败。为了解决这一问题，我们提出GR-CLIP，这是一种轻量级的后处理校准方法，用于消除CLIP嵌入空间中的模态差距。在专门为混合模态搜索设计的第一个基准MixBench上，GR-CLIP将NDCG@10提高多达26个百分点，优于最近的视觉-语言生成嵌入模型4个百分点，同时仅使用75倍少的计算资源。 

---
# Dual Path Learning -- learning from noise and context for medical image denoising 

**Title (ZH)**: 双重路径学习——从噪声和上下文中学习进行医学图像去噪 

**Authors**: Jitindra Fartiyal, Pedro Freire, Yasmeen Whayeb, James S. Wolffsohn, Sergei K. Turitsyn, Sergei G. Sokolov  

**Link**: [PDF](https://arxiv.org/pdf/2507.19035)  

**Abstract**: Medical imaging plays a critical role in modern healthcare, enabling clinicians to accurately diagnose diseases and develop effective treatment plans. However, noise, often introduced by imaging devices, can degrade image quality, leading to misinterpretation and compromised clinical outcomes. Existing denoising approaches typically rely either on noise characteristics or on contextual information from the image. Moreover, they are commonly developed and evaluated for a single imaging modality and noise type. Motivated by Geng this http URL CNCL, which integrates both noise and context, this study introduces a Dual-Pathway Learning (DPL) model architecture that effectively denoises medical images by leveraging both sources of information and fusing them to generate the final output. DPL is evaluated across multiple imaging modalities and various types of noise, demonstrating its robustness and generalizability. DPL improves PSNR by 3.35% compared to the baseline UNet when evaluated on Gaussian noise and trained across all modalities. The code is available at https://doi.org/10.5281/zenodo.15836053. 

**Abstract (ZH)**: 医学影像在现代医疗保健中发挥着关键作用，使临床医生能够准确诊断疾病并制定有效的治疗计划。然而，由成像设备引入的噪声会降低图像质量，导致误解释并损害临床效果。现有去噪方法通常依赖于噪声特性或图像中的上下文信息。此外，这些方法通常仅针对单一成像模态和噪声类型进行开发和评估。受Geng提出的CNCL的启发，本文引入了一种双路径学习(Dual-Pathway Learning, DPL)模型架构，通过利用这两种信息源并融合它们来生成最终输出，有效去噪医学图像。DPL在多种成像模态和噪声类型下进行评估，展示了其稳健性和通用性。当在高斯噪声下评估并与基线UNet相比时，DPL在所有模态上训练时提高了3.35%的PSNR。代码可在https://doi.org/10.5281/zenodo.15836053获得。 

---
# MindSpeed RL: Distributed Dataflow for Scalable and Efficient RL Training on Ascend NPU Cluster 

**Title (ZH)**: MindSpeed RL：Ascend NPU集群上可扩展且高效的 reinforcement learning训练的数据流分布方案 

**Authors**: Laingjun Feng, Chenyi Pan, Xinjie Guo, Fei Mei, Benzhe Ning, Jianxiang Zhang, Xinyang Liu, Beirong Zhou, Zeng Shu, Chang Liu, Guang Yang, Zhenyu Han, Jiangben Wang, Bo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19017)  

**Abstract**: Reinforcement learning (RL) is a paradigm increasingly used to align large language models. Popular RL algorithms utilize multiple workers and can be modeled as a graph, where each node is the status of a worker and each edge represents dataflow between nodes. Owing to the heavy cross-node dependencies, the RL training system usually suffers from poor cluster scalability and low memory utilization. In this article, we introduce MindSpeed RL, an effective and efficient system for large-scale RL training. Unlike existing centralized methods, MindSpeed RL organizes the essential data dependencies in RL training, i.e., sample flow and resharding flow, from a distributed view. On the one hand, a distributed transfer dock strategy, which sets controllers and warehouses on the basis of the conventional replay buffer, is designed to release the dispatch overhead in the sample flow. A practical allgather--swap strategy is presented to eliminate redundant memory usage in resharding flow. In addition, MindSpeed RL further integrates numerous parallelization strategies and acceleration techniques for systematic optimization. Compared with existing state-of-the-art systems, comprehensive experiments on the RL training of popular Qwen2.5-Dense-7B/32B, Qwen3-MoE-30B, and DeepSeek-R1-MoE-671B show that MindSpeed RL increases the throughput by 1.42 ~ 3.97 times. Finally, we open--source MindSpeed RL and perform all the experiments on a super pod of Ascend with 384 neural processing units (NPUs) to demonstrate the powerful performance and reliability of Ascend. 

**Abstract (ZH)**: 大规模强化学习训练的MindSpeed RL系统 

---
# MedIQA: A Scalable Foundation Model for Prompt-Driven Medical Image Quality Assessment 

**Title (ZH)**: MedIQA: 一种面向提示的可扩展医学图像质量评估基础模型 

**Authors**: Siyi Xun, Yue Sun, Jingkun Chen, Zitong Yu, Tong Tong, Xiaohong Liu, Mingxiang Wu, Tao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.19004)  

**Abstract**: Rapid advances in medical imaging technology underscore the critical need for precise and automated image quality assessment (IQA) to ensure diagnostic accuracy. Existing medical IQA methods, however, struggle to generalize across diverse modalities and clinical scenarios. In response, we introduce MedIQA, the first comprehensive foundation model for medical IQA, designed to handle variability in image dimensions, modalities, anatomical regions, and types. We developed a large-scale multi-modality dataset with plentiful manually annotated quality scores to support this. Our model integrates a salient slice assessment module to focus on diagnostically relevant regions feature retrieval and employs an automatic prompt strategy that aligns upstream physical parameter pre-training with downstream expert annotation fine-tuning. Extensive experiments demonstrate that MedIQA significantly outperforms baselines in multiple downstream tasks, establishing a scalable framework for medical IQA and advancing diagnostic workflows and clinical decision-making. 

**Abstract (ZH)**: 快速发展的医学图像技术强调了精确且自动化的图像质量评估（IQA）的重要性，以确保诊断准确性。现有的医学IQA方法难以在多种模态和临床场景下进行泛化。为应对这一挑战，我们提出了MedIQA，这是首个全面的基础模型，旨在处理图像尺寸、模态、解剖区域和类型的变异性。我们开发了一个大规模多模态数据集，包含丰富的手动注释质量分数，以支持这一目标。该模型集成了显著切片评估模块，专注于诊断相关的区域特征检索，并采用了一种自动提示策略，将上游物理参数预训练与下游专家注释微调对齐。广泛实验证明，MedIQA在多个下游任务中显著优于基线方法，建立了可扩展的医学IQA框架，并推动了诊断工作流程和临床决策的进展。 

---
# A diffusion-based generative model for financial time series via geometric Brownian motion 

**Title (ZH)**: 基于扩散的金融时间序列生成模型通过几何布朗运动 

**Authors**: Gihun Kim, Sun-Yong Choi, Yeoneung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.19003)  

**Abstract**: We propose a novel diffusion-based generative framework for financial time series that incorporates geometric Brownian motion (GBM), the foundation of the Black--Scholes theory, into the forward noising process. Unlike standard score-based models that treat price trajectories as generic numerical sequences, our method injects noise proportionally to asset prices at each time step, reflecting the heteroskedasticity observed in financial time series. By accurately balancing the drift and diffusion terms, we show that the resulting log-price process reduces to a variance-exploding stochastic differential equation, aligning with the formulation in score-based generative models. The reverse-time generative process is trained via denoising score matching using a Transformer-based architecture adapted from the Conditional Score-based Diffusion Imputation (CSDI) framework. Empirical evaluations on historical stock data demonstrate that our model reproduces key stylized facts heavy-tailed return distributions, volatility clustering, and the leverage effect more realistically than conventional diffusion models. 

**Abstract (ZH)**: 我们提出了一种基于扩散的金融时间序列生成框架，将Black-Scholes理论的基础几何布朗运动（GBM）融入前向噪声过程。与标准评分基础模型将价格轨迹视为通用数字序列的做法不同，我们的方法在每个时间步长中根据资产价格成比例地注入噪声，反映金融时间序列中观察到的异方差性。通过精确平衡漂移和扩散项，我们证明所得到的对数价格过程转化为一个方差爆炸的随机微分方程，与评分基础生成模型的公式一致。反向生成过程通过源自Conditional Score-based Diffusion Imputation (CSDI)框架的Transformer架构结合去噪评分匹配进行训练。在历史股价数据上的实证评估表明，我们的模型比传统的扩散模型更真实地再现了重尾回报分布、波动集聚和杠杆效应等关键特征。 

---
# GENIAL: Generative Design Space Exploration via Network Inversion for Low Power Algorithmic Logic Units 

**Title (ZH)**: GENIAL: 基于网络反转的生成设计空间探索用于低功耗算法逻辑单元 

**Authors**: Maxence Bouvier, Ryan Amaudruz, Felix Arnold, Renzo Andri, Lukas Cavigelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.18989)  

**Abstract**: As AI workloads proliferate, optimizing arithmetic units is becoming increasingly important to reduce the footprint of digital systems. Conventional design flows, which often rely on manual or heuristics-based optimization, are limited in their ability to thoroughly explore the vast design space. In this paper, we introduce GENIAL, a machine learning-based framework for the automatic generation and optimization of arithmetic units, more specifically multipliers.
At the core of GENIAL is a Transformer-based surrogate model trained in two stages, involving self-supervised pretraining followed by supervised finetuning, to robustly forecast key hardware metrics such as power and area from abstracted design representations. By inverting the surrogate model, GENIAL efficiently searches for new operand encodings that directly minimize power consumption in arithmetic units for specific input data distributions. Extensive experiments on large datasets demonstrate that GENIAL is consistently more sample efficient than other methods, and converges faster towards optimized designs. This enables to deploy a high-effort logic synthesis optimization flow in the loop, improving the accuracy of the surrogate model. Notably, GENIAL automatically discovers encodings that achieve up to 18% switching activity savings within multipliers on representative AI workloads compared with the conventional two's complement. We also demonstrate the versatility of our approach by achieving significant improvements on Finite State Machines, highlighting GENIAL's applicability for a wide spectrum of logic functions. Together, these advances mark a significant step toward automated Quality-of-Results-optimized combinational circuit generation for digital systems. 

**Abstract (ZH)**: 基于机器学习的自动生成与优化算术单元框架：GENIAL 

---
# Differentiated Thyroid Cancer Recurrence Classification Using Machine Learning Models and Bayesian Neural Networks with Varying Priors: A SHAP-Based Interpretation of the Best Performing Model 

**Title (ZH)**: 基于机器学习模型和具有变动先验的贝叶斯神经网络的分化型甲状腺癌复发分类：SHAP基的理解最佳模型解析 

**Authors**: HMNS Kumari, HMLS Kumari, UMMPK Nawarathne  

**Link**: [PDF](https://arxiv.org/pdf/2507.18987)  

**Abstract**: Differentiated thyroid cancer DTC recurrence is a major public health concern, requiring classification and predictive models that are not only accurate but also interpretable and uncertainty aware. This study introduces a comprehensive framework for DTC recurrence classification using a dataset containing 383 patients and 16 clinical and pathological variables. Initially, 11 machine learning ML models were employed using the complete dataset, where the Support Vector Machines SVM model achieved the highest accuracy of 0.9481. To reduce complexity and redundancy, feature selection was carried out using the Boruta algorithm, and the same ML models were applied to the reduced dataset, where it was observed that the Logistic Regression LR model obtained the maximum accuracy of 0.9611. However, these ML models often lack uncertainty quantification, which is critical in clinical decision making. Therefore, to address this limitation, the Bayesian Neural Networks BNN with six varying prior distributions, including Normal 0,1, Normal 0,10, Laplace 0,1, Cauchy 0,1, Cauchy 0,2.5, and Horseshoe 1, were implemented on both the complete and reduced datasets. The BNN model with Normal 0,10 prior distribution exhibited maximum accuracies of 0.9740 and 0.9870 before and after feature selection, respectively. 

**Abstract (ZH)**: 不同分化型甲状腺癌复发的分类及预测模型：基于不确定性意识的可解释框架 

---
# A Toolbox, Not a Hammer -- Multi-TAG: Scaling Math Reasoning with Multi-Tool Aggregation 

**Title (ZH)**: 不是一根锤子，而是一个工具箱——Multi-TAG：基于多工具聚合的数学推理扩展 

**Authors**: Bohan Yao, Vikas Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2507.18973)  

**Abstract**: Augmenting large language models (LLMs) with external tools is a promising avenue for developing high-performance mathematical reasoning systems. Prior tool-augmented approaches typically finetune an LLM to select and invoke a single tool at each reasoning step and show promising results on simpler math reasoning benchmarks such as GSM8K. However, these approaches struggle with more complex math problems that require precise reasoning over multiple steps. To address this limitation, in this work, we propose Multi-TAG, a Multi-Tool AGgregation-based framework. Instead of relying on a single tool, Multi-TAG guides an LLM to concurrently invoke multiple tools at each reasoning step. It then aggregates their diverse outputs to verify and refine the reasoning process, enhancing solution robustness and accuracy. Notably, Multi-TAG is a finetuning-free, inference-only framework, making it readily applicable to any LLM backbone, including large open-weight models which are computationally expensive to finetune and proprietary frontier models which cannot be finetuned with custom recipes. We evaluate Multi-TAG on four challenging benchmarks: MATH500, AIME, AMC, and OlympiadBench. Across both open-weight and closed-source LLM backbones, Multi-TAG consistently and substantially outperforms state-of-the-art baselines, achieving average improvements of 6.0% to 7.5% over state-of-the-art baselines. 

**Abstract (ZH)**: 增强大型语言模型（LLMs）与外部工具的结合是开发高性能数学推理系统的一个有前途的途径。在此工作中，我们提出了Multi-TAG，一种基于多工具聚合的框架。Multi-TAG 在每一步推理过程中指导LLM同时调用多个工具，然后汇总它们的多样化输出以验证和细化推理过程，从而增强解决方案的稳健性和准确性。值得注意的是，Multi-TAG 是一个不需要微调的仅推理框架，使其可以轻松应用于任何LLM基础模型，包括计算成本高昂且难以微调的大型开放权重模型以及不能使用自定义食谱进行微调的专有前沿模型。我们在四个具有挑战性的基准上评估了Multi-TAG：MATH500、AIME、AMC 和 OlympiadBench。在开放权重和闭源LLM基础模型上，Multi-TAG 一致且显著地超过了最先进的基线，平均改进幅度在6.0%到7.5%之间。 

---
# Underwater Waste Detection Using Deep Learning A Performance Comparison of YOLOv7 to 10 and Faster RCNN 

**Title (ZH)**: 基于深度学习的水下废弃物检测：YOLOv7与Faster RCNN的性能对比 

**Authors**: UMMPK Nawarathne, HMNS Kumari, HMLS Kumari  

**Link**: [PDF](https://arxiv.org/pdf/2507.18967)  

**Abstract**: Underwater pollution is one of today's most significant environmental concerns, with vast volumes of garbage found in seas, rivers, and landscapes around the world. Accurate detection of these waste materials is crucial for successful waste management, environmental monitoring, and mitigation strategies. In this study, we investigated the performance of five cutting-edge object recognition algorithms, namely YOLO (You Only Look Once) models, including YOLOv7, YOLOv8, YOLOv9, YOLOv10, and Faster Region-Convolutional Neural Network (R-CNN), to identify which model was most effective at recognizing materials in underwater situations. The models were thoroughly trained and tested on a large dataset containing fifteen different classes under diverse conditions, such as low visibility and variable depths. From the above-mentioned models, YOLOv8 outperformed the others, with a mean Average Precision (mAP) of 80.9%, indicating a significant performance. This increased performance is attributed to YOLOv8's architecture, which incorporates advanced features such as improved anchor-free mechanisms and self-supervised learning, allowing for more precise and efficient recognition of items in a variety of settings. These findings highlight the YOLOv8 model's potential as an effective tool in the global fight against pollution, improving both the detection capabilities and scalability of underwater cleanup operations. 

**Abstract (ZH)**: 水域污染是当今最重要的环境问题之一，全球各地的海洋、河流和landscape中发现了大量垃圾。准确识别这些废弃物对于成功的废物管理、环境监测和缓解策略至关重要。本研究探讨了五种先进的物体识别算法，即YOLO（You Only Look Once）模型，包括YOLOv7、YOLOv8、YOLOv9、YOLOv10和Faster Region-Convolutional Neural Network (R-CNN)，以确定哪种模型在水下环境中识别材料最有效。这些模型在包含十五个不同类别的大型数据集上进行了充分的训练和测试，数据集涵盖了不同的条件，如低能见度和不同深度。从上述模型中，YOLOv8的表现优于其他模型，其平均精确度（mAP）为80.9%，表明其具有显著性能。这种性能提升归因于YOLOv8的架构，该架构包含了改进的无锚机制和自我监督学习等高级功能，使其能够在各种环境中更精确和高效地识别物品。这些发现突显了YOLOv8模型在应对全球污染问题中的潜力，提高了水下清理操作的检测能力和可扩展性。 

---
# TreeReader: A Hierarchical Academic Paper Reader Powered by Language Models 

**Title (ZH)**: TreeReader: 一种基于语言模型的层次化学术论文阅读器 

**Authors**: Zijian Zhang, Pan Chen, Fangshi Du, Runlong Ye, Oliver Huang, Michael Liut, Alán Aspuru-Guzik  

**Link**: [PDF](https://arxiv.org/pdf/2507.18945)  

**Abstract**: Efficiently navigating and understanding academic papers is crucial for scientific progress. Traditional linear formats like PDF and HTML can cause cognitive overload and obscure a paper's hierarchical structure, making it difficult to locate key information. While LLM-based chatbots offer summarization, they often lack nuanced understanding of specific sections, may produce unreliable information, and typically discard the document's navigational structure. Drawing insights from a formative study on academic reading practices, we introduce TreeReader, a novel language model-augmented paper reader. TreeReader decomposes papers into an interactive tree structure where each section is initially represented by an LLM-generated concise summary, with underlying details accessible on demand. This design allows users to quickly grasp core ideas, selectively explore sections of interest, and verify summaries against the source text. A user study was conducted to evaluate TreeReader's impact on reading efficiency and comprehension. TreeReader provides a more focused and efficient way to navigate and understand complex academic literature by bridging hierarchical summarization with interactive exploration. 

**Abstract (ZH)**: 高效导航和理解学术论文对于科学研究至关重要。传统的线性格式如PDF和HTML会导致认知负担过重并模糊论文的层级结构，使得定位关键信息变得困难。虽然基于LLM的聊天机器人可以提供摘要，但它们通常对特定部分的理解不够深入，可能会产生不可靠的信息，并且通常会丢弃文档的导航结构。借鉴对学术阅读实践的形成性研究，我们引入了TreeReader，这是一种新型的语言模型增强型论文阅读工具。TreeReader将论文分解为一个交互式树结构，每个部分最初由LLM生成的简洁摘要表示，详细内容可根据需要获取。该设计使得用户能够快速把握核心思想，选择性地探索感兴趣的部分，并对照原始文本验证摘要。我们进行了一项用户研究以评估TreeReader对阅读效率和理解度的影响。TreeReader通过将层级总结与互动探索相结合，提供了一种更聚焦和高效的方式来导航和理解复杂的学术文献。 

---
# CNN-based Surface Temperature Forecasts with Ensemble Numerical Weather Prediction over Medium-range Forecast Periods 

**Title (ZH)**: 基于CNN的中期预报期内集合数值天气预报地表温度预测 

**Authors**: Takuya Inoue, Takuya Kawabata  

**Link**: [PDF](https://arxiv.org/pdf/2507.18937)  

**Abstract**: This study proposes a method that integrates convolutional neural networks (CNNs) with ensemble numerical weather prediction (NWP) models, enabling surface temperature forecasting at lead times beyond the short-range (five-day) forecast period. Owing to limited computational resources, operational medium-range temperature forecasts typically rely on low-resolution NWP models, which are prone to systematic and random errors. To resolve these limitations, the proposed method first reduces systematic errors through CNN-based post-processing (bias correction and spatial super-resolution) on each ensemble member, reconstructing high-resolution temperature fields from low-resolution model outputs. Second, it reduces random errors through ensemble averaging of the CNN-corrected members. This study also investigates whether the sequence of CNN correction and ensemble averaging affects the forecast accuracy. For comparison with the proposed method, we additionally conducted experiments with the CNN trained on ensemble-averaged forecasts. The first approach--CNN correction before ensemble averaging--consistently achieved higher accuracy than the reverse approach. Although based on low-resolution ensemble forecasts, the proposed method notably outperformed the high-resolution deterministic NWP models. These findings indicate that combining CNN-based correction with ensemble averaging effectively reduces both the systematic and random errors in NWP model outputs. The proposed approach is a practical and scalable solution for improving medium-range temperature forecasts, and is particularly valuable at operational centers with limited computational resources. 

**Abstract (ZH)**: 本研究提出了一种将卷积神经网络（CNN）与集合数值天气预报（NWP）模型相结合的方法，能够在短程预报（五天）之外的时间提前量进行地表温度预测。由于计算资源有限， operational 中程温度预报通常依赖于低分辨率的 NWP 模型，这些模型容易出现系统性错误和随机误差。为解决这些问题，该方法首先通过基于 CNN 的后处理（偏差校正和空间超分辨率）对每个集合成员进行处理，从低分辨率模型输出中重建高分辨率温度场；其次，通过 CNN 校正后的成员的集合平均来减少随机误差。本研究还探讨了 CNN 校正和集合平均顺序是否影响预报准确性。为了与提出的方案进行比较，我们还进行了使用在集合平均预报上训练的 CNN 的实验。结果显示，在集合平均之前进行 CNN 校正的方法始终比反其顺序的方法具有更高的准确性。尽管基于低分辨率的集合预报，提出的方案在性能上显著优于高分辨率的确定性 NWP 模型。这些发现表明，结合 CNN 基准校正与集合平均能有效减少 NWP 模型输出中的系统性和随机误差。该提出的方法是提高中程温度预报的一种实际且可扩展的解决方案，特别适合作为计算资源有限的 operational 中心。 

---
# MGHFT: Multi-Granularity Hierarchical Fusion Transformer for Cross-Modal Sticker Emotion Recognition 

**Title (ZH)**: MGHFT：多粒度层次融合变换器在跨模态贴纸情感识别中的应用 

**Authors**: Jian Chen, Yuxuan Hu, Haifeng Lu, Wei Wang, Min Yang, Chengming Li, Xiping Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18929)  

**Abstract**: Although pre-trained visual models with text have demonstrated strong capabilities in visual feature extraction, sticker emotion understanding remains challenging due to its reliance on multi-view information, such as background knowledge and stylistic cues. To address this, we propose a novel multi-granularity hierarchical fusion transformer (MGHFT), with a multi-view sticker interpreter based on Multimodal Large Language Models. Specifically, inspired by the human ability to interpret sticker emotions from multiple views, we first use Multimodal Large Language Models to interpret stickers by providing rich textual context via multi-view descriptions. Then, we design a hierarchical fusion strategy to fuse the textual context into visual understanding, which builds upon a pyramid visual transformer to extract both global and local sticker features at multiple stages. Through contrastive learning and attention mechanisms, textual features are injected at different stages of the visual backbone, enhancing the fusion of global- and local-granularity visual semantics with textual guidance. Finally, we introduce a text-guided fusion attention mechanism to effectively integrate the overall multimodal features, enhancing semantic understanding. Extensive experiments on 2 public sticker emotion datasets demonstrate that MGHFT significantly outperforms existing sticker emotion recognition approaches, achieving higher accuracy and more fine-grained emotion recognition. Compared to the best pre-trained visual models, our MGHFT also obtains an obvious improvement, 5.4% on F1 and 4.0% on accuracy. The code is released at this https URL. 

**Abstract (ZH)**: 尽管带有文本的预训练视觉模型已经在视觉特征提取方面展示了强大的能力，但由于贴纸情绪理解依赖于多视角信息，如背景知识和风格线索，因此仍然具有挑战性。为了解决这一问题，我们提出了一种新颖的多粒度层次融合Transformer（MGHFT），并基于多模态大型语言模型设计了多视角贴纸解析器。具体来说，受人类能够从多视角解析贴纸情绪的能力启发，我们首先使用多模态大型语言模型通过多视角描述提供丰富的文本上下文来解析贴纸。然后，我们设计了一种层次融合策略，将文本上下文融合到视觉理解中，基于金字塔视觉Transformer在多个阶段提取全局和局部贴纸特征。通过对比学习和注意力机制，在视觉骨干网的不同阶段注入文本特征，增强全局和局部粒度视觉语义与文本指导的融合。最后，我们引入了一种文本导向的融合注意力机制，以有效集成整体多模态特征，增强语义理解。广泛实验表明，MGHFT 在两个公开的贴纸情绪数据集上显著优于现有贴纸情绪识别方法，实现了更高的准确率和更细粒度的情绪识别。与最佳预训练视觉模型相比，我们的MGHFT 在F1分数和准确率上分别取得了5.4%和4.0%的明显提升。代码发布在该网址：https://。 

---
# WiSE-OD: Benchmarking Robustness in Infrared Object Detection 

**Title (ZH)**: WiSE-OD：红外目标检测的鲁棒性benchmark 

**Authors**: Heitor R. Medeiros, Atif Belal, Masih Aminbeidokhti, Eric Granger, Marco Pedersoli  

**Link**: [PDF](https://arxiv.org/pdf/2507.18925)  

**Abstract**: Object detection (OD) in infrared (IR) imagery is critical for low-light and nighttime applications. However, the scarcity of large-scale IR datasets forces models to rely on weights pre-trained on RGB images. While fine-tuning on IR improves accuracy, it often compromises robustness under distribution shifts due to the inherent modality gap between RGB and IR. To address this, we introduce LLVIP-C and FLIR-C, two cross-modality out-of-distribution (OOD) benchmarks built by applying corruption to standard IR datasets. Additionally, to fully leverage the complementary knowledge from RGB and infrared trained models, we propose WiSE-OD, a weight-space ensembling method with two variants: WiSE-OD$_{ZS}$, which combines RGB zero-shot and IR fine-tuned weights, and WiSE-OD$_{LP}$, which blends zero-shot and linear probing. Evaluated across three RGB-pretrained detectors and two robust baselines, WiSE-OD improves both cross-modality and corruption robustness without any additional training or inference cost. 

**Abstract (ZH)**: 红外影像中的目标检测（OD）在低光和夜间应用中至关重要。然而，大规模红外数据集的稀缺性迫使模型依赖于在RGB图像上预训练的权重。尽管在红外数据上进行微调可以提高准确性，但往往会因为RGB和红外之间固有的模态差异而牺牲分布迁移下的稳健性。为了解决这个问题，我们引入了LLVIP-C和FLIR-C两种跨模态的 outlier of distribution (OOD) 基准，通过在标准红外数据集上应用干扰构建。此外，为了充分利用从RGB和红外训练模型中获得的互补知识，我们提出了WiSE-OD，一种权重空间集成方法，包含两种变体：WiSE-OD$_{ZS}$ 结合了RGB零样本和红外微调权重，WiSE-OD$_{LP}$ 结合了零样本和线性探针。在三种RGB预训练探测器和两种稳健基准上进行评估，WiSE-OD 在不增加额外训练或推理成本的情况下，提高了跨模态和干扰下的稳健性。 

---
# Uncovering Cross-Linguistic Disparities in LLMs using Sparse Autoencoders 

**Title (ZH)**: 使用稀疏自编码器揭示跨语言差异的大型语言模型分析 

**Authors**: Richmond Sin Jing Xuan, Jalil Huseynov, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18918)  

**Abstract**: Multilingual large language models (LLMs) exhibit strong cross-linguistic generalization, yet medium to low resource languages underperform on common benchmarks such as ARC-Challenge, MMLU, and HellaSwag. We analyze activation patterns in Gemma-2-2B across all 26 residual layers and 10 languages: Chinese (zh), Russian (ru), Spanish (es), Italian (it), medium to low resource languages including Indonesian (id), Catalan (ca), Marathi (mr), Malayalam (ml), and Hindi (hi), with English (en) as the reference. Using Sparse Autoencoders (SAEs), we reveal systematic disparities in activation patterns. Medium to low resource languages receive up to 26.27 percent lower activations in early layers, with a persistent gap of 19.89 percent in deeper layers. To address this, we apply activation-aware fine-tuning via Low-Rank Adaptation (LoRA), leading to substantial activation gains, such as 87.69 percent for Malayalam and 86.32 percent for Hindi, while maintaining English retention at approximately 91 percent. After fine-tuning, benchmark results show modest but consistent improvements, highlighting activation alignment as a key factor in enhancing multilingual LLM performance. 

**Abstract (ZH)**: 多语言大规模语言模型在跨语言泛化方面表现出色，但中低资源语言在ARC-Challenge、MMLU和HellaSwag等常见基准测试中表现不佳。我们通过分析Gemma-2-2B在所有26个残差层和10种语言（中文、俄语、西班牙语、意大利语，以及包括印尼语、加泰罗尼亚语、马拉地语、马拉雅拉姆语和印地语在内的中低资源语言）中的激活模式，揭示了系统性的差异。中低资源语言在早期层中接收到的激活值最多低26.27%，在更深的层中则持续保持19.89%的差距。为解决这一问题，我们通过低秩适应（LoRA）进行激活感知微调，从而显著提高激活量，如马拉雅拉姆语提升87.69%，印地语提升86.32%，同时保持英语保留约91%。经过微调后，基准测试结果显示有适度但一致的改进，突出了激活对齐在增强多语言LLM性能中的关键作用。 

---
# HH-Codec: High Compression High-fidelity Discrete Neural Codec for Spoken Language Modeling 

**Title (ZH)**: HH-Codec: 高压缩高保真离散神经编码器用于语音语言建模 

**Authors**: Rongkun Xue, Yazhe Niu, Shuai Hu, Zixin Yin, Yongqiang Yao, Jing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18897)  

**Abstract**: Discrete speech tokenization is a fundamental component in speech codecs. However, in large-scale speech-to-speech systems, the complexity of parallel streams from multiple quantizers and the computational cost of high-time-dimensional codecs pose significant challenges. In this paper, we introduce HH-Codec, a neural codec that achieves extreme compression at 24 tokens per second for 24 kHz audio while relying on single-quantizer inference. Our approach involves a carefully designed Vector Quantization space for Spoken Language Modeling, optimizing compression efficiency while minimizing information loss. Building on this, we propose an asymmetric encoder-decoder architecture (Audio-VQ-Mel-Audio) that leverages dual supervision and progressive training to enhance reconstruction stability and fidelity. HH-Codec achieves state-of-the-art performance in speech reconstruction with an ultra-low bandwidth of 0.3 kbps. We further evaluate its effectiveness in codebook utilization and generative model adaptation, with extensive ablations validating the necessity of each module. HH-Codec is available at this https URL. 

**Abstract (ZH)**: 离散语音token化是语音编解码器中的一个基本组成部分。然而，在大规模的语音-语音系统中，来自多个量化器的并行流的复杂性以及高时间维度编解码器的计算成本构成了重大挑战。本文介绍了一种名为HH-Codec的神经编解码器，该编解码器在24 kHz音频上实现了每秒24个tokens的极端压缩，仅依赖于单量化器推理。我们的方法涉及到精心设计的语音语言模型向量量化空间，优化了压缩效率并尽量减少信息损失。在此基础上，我们提出了一种不对称编码-解码架构（Audio-VQ-Mel-Audio），利用双监督和渐进训练来增强重建的稳定性和保真度。HH-Codec在超低带宽0.3 kbps的情况下实现了语音重建的最先进的性能。我们进一步评估了其在码书利用和生成模型适应性方面的有效性，并通过广泛的消融实验验证了每个模块的必要性。HH-Codec可在以下链接获取：this https URL。 

---
# A Comprehensive Review of AI-based Intelligent Tutoring Systems: Applications and Challenges 

**Title (ZH)**: 基于AI的智能教学系统综述：应用与挑战 

**Authors**: Meriem Zerkouk, Miloud Mihoubi, Belkacem Chikhaoui  

**Link**: [PDF](https://arxiv.org/pdf/2507.18882)  

**Abstract**: AI-based Intelligent Tutoring Systems (ITS) have significant potential to transform teaching and learning. As efforts continue to design, develop, and integrate ITS into educational contexts, mixed results about their effectiveness have emerged. This paper provides a comprehensive review to understand how ITS operate in real educational settings and to identify the associated challenges in their application and evaluation. We use a systematic literature review method to analyze numerous qualified studies published from 2010 to 2025, examining domains such as pedagogical strategies, NLP, adaptive learning, student modeling, and domain-specific applications of ITS. The results reveal a complex landscape regarding the effectiveness of ITS, highlighting both advancements and persistent challenges. The study also identifies a need for greater scientific rigor in experimental design and data analysis. Based on these findings, suggestions for future research and practical implications are proposed. 

**Abstract (ZH)**: 基于AI的智能辅导系统（ITS）在教育教学中的应用具有显著潜力。随着努力设计、开发并将ITS集成到教育环境中，关于其有效性的结果出现了分歧。本文提供了全面的综述，以理解ITS在实际教育环境中的运作方式，并识别其应用和评估中遇到的挑战。我们采用系统文献综述方法，分析了2010年至2025年间发表的多项合格研究，涉及教学策略、自然语言处理、自适应学习、学生建模以及ITS的具体应用领域。研究结果揭示了有关ITS有效性的复杂景观，突显了进步与持续挑战。研究还指出需要在实验设计和数据分析方面增加科学严谨性。基于这些发现，提出了未来研究的建议和实践意义。 

---
# Learning Individual Intrinsic Reward in Multi-Agent Reinforcement Learning via Incorporating Generalized Human Expertise 

**Title (ZH)**: 通过融入泛化人类专长学习多智能体 reinforcement learning 中的个体固有奖励 

**Authors**: Xuefei Wu, Xiao Yin, Yuanyang Zhu, Chunlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.18867)  

**Abstract**: Efficient exploration in multi-agent reinforcement learning (MARL) is a challenging problem when receiving only a team reward, especially in environments with sparse rewards. A powerful method to mitigate this issue involves crafting dense individual rewards to guide the agents toward efficient exploration. However, individual rewards generally rely on manually engineered shaping-reward functions that lack high-order intelligence, thus it behaves ineffectively than humans regarding learning and generalization in complex problems. To tackle these issues, we combine the above two paradigms and propose a novel framework, LIGHT (Learning Individual Intrinsic reward via Incorporating Generalized Human experTise), which can integrate human knowledge into MARL algorithms in an end-to-end manner. LIGHT guides each agent to avoid unnecessary exploration by considering both individual action distribution and human expertise preference distribution. Then, LIGHT designs individual intrinsic rewards for each agent based on actionable representational transformation relevant to Q-learning so that the agents align their action preferences with the human expertise while maximizing the joint action value. Experimental results demonstrate the superiority of our method over representative baselines regarding performance and better knowledge reusability across different sparse-reward tasks on challenging scenarios. 

**Abstract (ZH)**: 高效探索在仅接收团队奖励的多智能体强化学习中的有效实现，尤其是在稀疏奖励环境中是一个挑战性问题。一种有力的方法是设计密集的个体奖励来引导智能体进行高效的探索。然而，个体奖励通常依赖于手动工程化的塑造奖励函数，缺乏高级智能，因此在复杂问题的学习和泛化方面表现不如人类。为了解决这些问题，我们将上述两种范式结合起来，提出了一种新型框架 LIGHT（基于广泛人类专业知识学习个体内在奖励），该框架可以以端到端的方式将人类知识整合到多智能体强化学习算法中。LIGHT 通过同时考虑个体动作分布和人类专业知识偏好分布来引导每个智能体避免不必要的探索。然后，LIGHT 基于与 Q-learning 相关的动作表示转换设计每个智能体的个体内在奖励，使智能体在最大化联合动作价值的同时与其人类专业知识对齐。实验结果表明，与代表性基线方法相比，我们的方法在性能上具有优势，并且在不同稀疏奖励任务的挑战性场景中具有更好的知识可重用性。 

---
# PrismRAG: Boosting RAG Factuality with Distractor Resilience and Strategized Reasoning 

**Title (ZH)**: PrismRAG: 提升RAG事实性的干扰物抗性和策略化推理 

**Authors**: Mohammad Kachuee, Teja Gollapudi, Minseok Kim, Yin Huang, Kai Sun, Xiao Yang, Jiaqi Wang, Nirav Shah, Yue Liu, Aaron Colak, Anuj Kumar, Wen-tau Yih, Xin Luna Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.18857)  

**Abstract**: Retrieval-augmented generation (RAG) often falls short when retrieved context includes confusing semi-relevant passages, or when answering questions require deep contextual understanding and reasoning. We propose an efficient fine-tuning framework, called PrismRAG, that (i) trains the model with distractor-aware QA pairs mixing gold evidence with subtle distractor passages, and (ii) instills reasoning-centric habits that make the LLM plan, rationalize, and synthesize without relying on extensive human engineered instructions. Evaluated across 12 open-book RAG QA benchmarks spanning diverse application domains and scenarios, PrismRAG improves average factuality by 5.4%, outperforming state-of-the-art solutions. 

**Abstract (ZH)**: PrismRAG：一种高效的细调框架，用于增强生成模型在包含混淆半相关段落的检索结果中的表现，并促进基于推理的习惯以提高事实准确性 

---
# PTCMIL: Multiple Instance Learning via Prompt Token Clustering for Whole Slide Image Analysis 

**Title (ZH)**: PTCMIL: 基于提示词聚类的多实例学习在全视野图像分析中的应用 

**Authors**: Beidi Zhao, SangMook Kim, Hao Chen, Chen Zhou, Zu-hua Gao, Gang Wang, Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.18848)  

**Abstract**: Multiple Instance Learning (MIL) has advanced WSI analysis but struggles with the complexity and heterogeneity of WSIs. Existing MIL methods face challenges in aggregating diverse patch information into robust WSI representations. While ViTs and clustering-based approaches show promise, they are computationally intensive and fail to capture task-specific and slide-specific variability. To address these limitations, we propose PTCMIL, a novel Prompt Token Clustering-based ViT for MIL aggregation. By introducing learnable prompt tokens into the ViT backbone, PTCMIL unifies clustering and prediction tasks in an end-to-end manner. It dynamically aligns clustering with downstream tasks, using projection-based clustering tailored to each WSI, reducing complexity while preserving patch heterogeneity. Through token merging and prototype-based pooling, PTCMIL efficiently captures task-relevant patterns. Extensive experiments on eight datasets demonstrate its superior performance in classification and survival analysis tasks, outperforming state-of-the-art methods. Systematic ablation studies confirm its robustness and strong interpretability. The code is released at this https URL. 

**Abstract (ZH)**: 基于提示词聚类的ViT多实例学习方法（PTCMIL）：统一聚类与预测任务以提高WSI分析效率 

---
# Equivariant Volumetric Grasping 

**Title (ZH)**: 等变体积抓取 

**Authors**: Pinhao Song, Yutong Hu, Pengteng Li, Renaud Detry  

**Link**: [PDF](https://arxiv.org/pdf/2507.18847)  

**Abstract**: We propose a new volumetric grasp model that is equivariant to rotations around the vertical axis, leading to a significant improvement in sample efficiency. Our model employs a tri-plane volumetric feature representation -- i.e., the projection of 3D features onto three canonical planes. We introduce a novel tri-plane feature design in which features on the horizontal plane are equivariant to 90° rotations, while the sum of features from the other two planes remains invariant to the same transformations. This design is enabled by a new deformable steerable convolution, which combines the adaptability of deformable convolutions with the rotational equivariance of steerable ones. This allows the receptive field to adapt to local object geometry while preserving equivariance properties. We further develop equivariant adaptations of two state-of-the-art volumetric grasp planners, GIGA and IGD. Specifically, we derive a new equivariant formulation of IGD's deformable attention mechanism and propose an equivariant generative model of grasp orientations based on flow matching. We provide a detailed analytical justification of the proposed equivariance properties and validate our approach through extensive simulated and real-world experiments. Our results demonstrate that the proposed projection-based design significantly reduces both computational and memory costs. Moreover, the equivariant grasp models built on top of our tri-plane features consistently outperform their non-equivariant counterparts, achieving higher performance with only a modest computational overhead. Video and code can be viewed in: this https URL 

**Abstract (ZH)**: 一种equivariant于垂直轴旋转的新体素抓取模型：显著提高样本效率的设计与应用 

---
# Flow Stochastic Segmentation Networks 

**Title (ZH)**: 流动随机分割网络 

**Authors**: Fabio De Sousa Ribeiro, Omar Todd, Charles Jones, Avinash Kori, Raghav Mehta, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2507.18838)  

**Abstract**: We introduce the Flow Stochastic Segmentation Network (Flow-SSN), a generative segmentation model family featuring discrete-time autoregressive and modern continuous-time flow variants. We prove fundamental limitations of the low-rank parameterisation of previous methods and show that Flow-SSNs can estimate arbitrarily high-rank pixel-wise covariances without assuming the rank or storing the distributional parameters. Flow-SSNs are also more efficient to sample from than standard diffusion-based segmentation models, thanks to most of the model capacity being allocated to learning the base distribution of the flow, constituting an expressive prior. We apply Flow-SSNs to challenging medical imaging benchmarks and achieve state-of-the-art results. Code available: this https URL. 

**Abstract (ZH)**: 我们介绍了Flow Stochastic Segmentation Network (Flow-SSN) 模型家族，这是一种具有离散时间自回归和现代连续时间流态变体的生成分割模型。我们证明了先前方法中低秩参数化的基本局限性，并展示了Flow-SSNs 可以在不假设秩或存储分布参数的情况下估计任意高秩的像素协方差。Flow-SSNs 比标准基于扩散的分割模型更高效，因为模型的大部分容量被分配用于学习流的基础分布，构成一个表达性先验。我们将在挑战性的医学成像基准测试中应用Flow-SSNs，并取得最佳成果。代码可供下载：this https URL。 

---
# Deepfake Detection Via Facial Feature Extraction and Modeling 

**Title (ZH)**: 基于面部特征提取与建模的Deepfake检测 

**Authors**: Benjamin Carter, Nathan Dilla, Micheal Callahan, Atuhaire Ambala  

**Link**: [PDF](https://arxiv.org/pdf/2507.18815)  

**Abstract**: The rise of deepfake technology brings forth new questions about the authenticity of various forms of media found online today. Videos and images generated by artificial intelligence (AI) have become increasingly more difficult to differentiate from genuine media, resulting in the need for new models to detect artificially-generated media. While many models have attempted to solve this, most focus on direct image processing, adapting a convolutional neural network (CNN) or a recurrent neural network (RNN) that directly interacts with the video image data. This paper introduces an approach of using solely facial landmarks for deepfake detection. Using a dataset consisting of both deepfake and genuine videos of human faces, this paper describes an approach for extracting facial landmarks for deepfake detection, focusing on identifying subtle inconsistencies in facial movements instead of raw image processing. Experimental results demonstrated that this feature extraction technique is effective in various neural network models, with the same facial landmarks tested on three neural network models, with promising performance metrics indicating its potential for real-world applications. The findings discussed in this paper include RNN and artificial neural network (ANN) models with accuracy between 96% and 93%, respectively, with a CNN model hovering around 78%. This research challenges the assumption that raw image processing is necessary to identify deepfake videos by presenting a facial feature extraction approach compatible with various neural network models while requiring fewer parameters. 

**Abstract (ZH)**: 深fake技术的兴起引发了对在线各种媒体 authenticity的新问题，生成于人工智能（AI）的视频和图像越来越难以与真实媒体区分开来，因此需要新的模型来检测人工生成的媒体。尽管已经提出了许多模型来解决这一问题，大多数模型主要关注直接图像处理，适应卷积神经网络（CNN）或递归神经网络（RNN），直接与视频图像数据交互。本文介绍了仅使用面部特征点进行深fake检测的方法。使用包含人工生成和真实视频人脸的的数据集，本文描述了一种提取面部特征点的方法，专注于识别面部运动中的细微不一致性，而不是直接的图像处理。实验结果表明，这种特征提取技术在各种神经网络模型中都是有效的，经过相同的面部特征点在三种神经网络模型上的测试，显示出其在实际应用中的潜力。本研究讨论的结果包括RNN和人工神经网络（ANN）模型的准确率分别为96%和93%，而CNN模型的准确率约为78%。本文挑战了直接图像处理是识别深fake视频所必需的假设，通过展示一种与各种神经网络模型兼容的面部特征提取方法，同时需要较少的参数。 

---
# MemoCoder: Automated Function Synthesis using LLM-Supported Agents 

**Title (ZH)**: MemoCoder: 使用LLM支持的代理进行自动函数合成 

**Authors**: Yiping Jia, Zhen Ming Jiang, Shayan Noei, Ying Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.18812)  

**Abstract**: With the widespread adoption of Large Language Models (LLMs) such as GitHub Copilot and ChatGPT, developers increasingly rely on AI-assisted tools to support code generation. While LLMs can generate syntactically correct solutions for well-structured programming tasks, they often struggle with challenges that require iterative debugging, error handling, or adaptation to diverse problem structures. Existing approaches such as fine-tuning or self-repair strategies either require costly retraining or lack mechanisms to accumulate and reuse knowledge from previous attempts.
To address these limitations, we propose MemoCoder, a multi-agent framework that enables collaborative problem solving and persistent learning from past fixes. At the core of MemoCoder is a Fixing Knowledge Set, which stores successful repairs and supports retrieval for future tasks. A central Mentor Agent supervises the repair process by identifying recurring error patterns and refining high-level fixing strategies, providing a novel supervisory role that guides the self-repair loop. We evaluate MemoCoder across three public benchmarks -- MBPP, HumanEval, and LiveCodeBench -- spanning a range of problem complexities. Experimental results show that MemoCoder consistently outperforms both zero-shot prompting and a Self-Repair strategy, with improvements ranging from 3.1% to 12.1% in Pass@10 and from 1.4% to 14.5% in Pass@50, demonstrating its effectiveness in iterative refinement and knowledge-guided code generation. 

**Abstract (ZH)**: 基于多智能体的笔记代码器：迭代调试与知识积累支持的代码生成 

---
# DxHF: Providing High-Quality Human Feedback for LLM Alignment via Interactive Decomposition 

**Title (ZH)**: DxHF: 通过交互分解提供高质量的人工反馈以实现LLM对齐 

**Authors**: Danqing Shi, Furui Cheng, Tino Weinkauf, Antti Oulasvirta, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2507.18802)  

**Abstract**: Human preferences are widely used to align large language models (LLMs) through methods such as reinforcement learning from human feedback (RLHF). However, the current user interfaces require annotators to compare text paragraphs, which is cognitively challenging when the texts are long or unfamiliar. This paper contributes by studying the decomposition principle as an approach to improving the quality of human feedback for LLM alignment. This approach breaks down the text into individual claims instead of directly comparing two long-form text responses. Based on the principle, we build a novel user interface DxHF. It enhances the comparison process by showing decomposed claims, visually encoding the relevance of claims to the conversation and linking similar claims. This allows users to skim through key information and identify differences for better and quicker judgment. Our technical evaluation shows evidence that decomposition generally improves feedback accuracy regarding the ground truth, particularly for users with uncertainty. A crowdsourcing study with 160 participants indicates that using DxHF improves feedback accuracy by an average of 5%, although it increases the average feedback time by 18 seconds. Notably, accuracy is significantly higher in situations where users have less certainty. The finding of the study highlights the potential of HCI as an effective method for improving human-AI alignment. 

**Abstract (ZH)**: 人类偏好分解原则在改善大型语言模型对齐中的应用：一种新型用户界面DxHF的研究 

---
# Tell Me What You See: An Iterative Deep Learning Framework for Image Captioning 

**Title (ZH)**: 告诉我你看到的：一种迭代深度学习图像 captioning 框架 

**Authors**: Hitesh Kumar Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.18788)  

**Abstract**: Image captioning, a task at the confluence of computer vision and natural language processing, requires a sophisticated understanding of both visual scenes and linguistic structure. While modern approaches are dominated by large-scale Transformer architectures, this paper documents a systematic, iterative development of foundational image captioning models, progressing from a simple CNN-LSTM encoder-decoder to a competitive attention-based system. We present a series of five models, beginning with Genesis and concluding with Nexus, an advanced model featuring an EfficientNetV2B3 backbone and a dynamic attention mechanism. Our experiments chart the impact of architectural enhancements and demonstrate a key finding within the classic CNN-LSTM paradigm: merely upgrading the visual backbone without a corresponding attention mechanism can degrade performance, as the single-vector bottleneck cannot transmit the richer visual detail. This insight validates the architectural shift to attention. Trained on the MS COCO 2017 dataset, our final model, Nexus, achieves a BLEU-4 score of 31.4, surpassing several foundational benchmarks and validating our iterative design process. This work provides a clear, replicable blueprint for understanding the core architectural principles that underpin modern vision-language tasks. 

**Abstract (ZH)**: 图像描述生成，作为计算机视觉和自然语言处理交汇的任务，需要对视觉场景和语言结构有复杂的理解。尽管现代方法主要由大规模Transformer架构主导，本文记录了一种系统性、迭代性的发展过程，从简单的CNN-LSTM编码器-解码器逐步发展到基于注意力的竞争性系统。我们呈现了一系列五种模型，从Genesis开始，最终以一个采用EfficientNetV2B3骨干和动态注意力机制的高级模型Nexus结束。我们的实验追踪了架构改进的影响，并在一个经典的CNN-LSTM框架中展示了关键发现：仅升级视觉骨干而不相应的注意力机制会降低性能，因为单一向量瓶颈无法传递更丰富的视觉细节。这一见解验证了向注意力机制的架构转变。在MS COCO 2017数据集上训练，我们的最终模型Nexus实现了BLEU-4得分31.4，超过了多个基础基准，验证了我们迭代的设计过程。这项工作为理解支撑现代视觉语言任务的核心架构原则提供了清晰可复制的蓝图。 

---
# Agentic Program Repair from Test Failures at Scale: A Neuro-symbolic approach with static analysis and test execution feedback 

**Title (ZH)**: 大规模基于测试失败的代理程序修复：一种结合静态分析和测试执行反馈的神经符号方法 

**Authors**: Chandra Maddila, Adam Tait, Claire Chang, Daniel Cheng, Nauman Ahmad, Vijayaraghavan Murali, Marshall Roch, Arnaud Avondet, Aaron Meltzer, Victor Montalvao, Michael Hopko, Chris Waterson, Parth Thakkar, Renuka Fernandez, Kristian Kristensen, Sivan Barzily, Sherry Chen, Rui Abreu, Nachiappan Nagappan, Payam Shodjai, Killian Murphy, James Everingham, Aparna Ramani, Peter C. Rigby  

**Link**: [PDF](https://arxiv.org/pdf/2507.18755)  

**Abstract**: Aim: With the advent of LLMs, sophisticated agentic program repair has become viable at large organizations with large codebases. In this work, we develop an Engineering Agent that fixes the source code based on test failures at scale across diverse software offerings internally.
Method: Using Llama as the base, we employ the ReAct harness to develop an agent. We start with a test failure that was triaged by a rule-based test failure bot. We then set up an agentic harness and allow the agent to reason and run a set of 15 actions from reading a file to generating a patch. We provide feedback to the agent through static analysis and test failures so it can refine its solution. We leverage an LLM-as-a-Judge to ensure that the patch conforms to the standards followed by a human review to land fixes.
Benchmark Findings: We curated offline benchmarks for our patch generator, the Engineering Agent loop, and the LLM-as-a-Judge. In offline evaluations we found that a specialized 70B model is highly competitive with the much larger but vanilla Llama-405B. In an ablation study, we found that the ReAct harness (neural model) benefited from the symbolic information from static analysis tools and test execution traces. A model that strikes a balance between the solve rate and error rate vs the cost and latency has a benchmark solve rate of 42.3% using an average 11.8 feedback iterations.
Production Findings: In a three month period, 80% of the generated fixes were reviewed, of which 31.5% were landed (25.5% of the total number of generated fixes).
Feedback from Engineers: We used open coding to extract qualitative themes from engineers' feedback. We saw positive feedback in the form of quick approvals, gratitude, and surprise. We also found mixed feedback when the Engineering Agent's solution was partially correct and it served as a good starting point. 

**Abstract (ZH)**: 目标：随着大规模语言模型（LLM）的出现，复杂的代理程序修复在大型代码库的大型组织中变得可行。在本研究中，我们开发了一个工程代理，该代理基于测试失败在多样化的软件产品内部大规模修复源代码。
方法：以Llama为基础，我们使用ReAct框架开发了一个代理。我们从一个由基于规则的测试失败机器人处理的测试失败开始。然后，我们设置了一个代理框架，允许代理进行推理并运行从读取文件到生成补丁的一系列15个操作。我们通过静态分析和测试失败向代理提供反馈，以便其优化解决方案。我们利用LLM作为裁判来确保补丁符合人工审查的标准，以实现修复。
基准发现：我们为补丁生成器、工程代理循环和LLM作为裁判制定了离线基准。在离线评估中，我们发现一个专门的70B模型与更大但未经过特殊训练的Llama-405B具有很高的竞争力。在消融研究中，我们发现ReAct框架（神经模型）受益于静态分析工具的符号信息和测试执行记录。具有平衡解决率和错误率与成本和延迟的模型，在平均11.8次反馈迭代的情况下，基准解决率为42.3%。
生产发现：在三个月期间，生成的修复中有80%被审查，其中31.5%被纳入（占总生成修复的25.5%）。
工程师反馈：我们使用开放编码从工程师的反馈中提取定性主题。我们发现快速批准、感激和惊讶的积极反馈。当工程代理的解决方案部分正确时，我们还发现了混合反馈，它作为一个好的起点是有益的。 

---
# Specification Self-Correction: Mitigating In-Context Reward Hacking Through Test-Time Refinement 

**Title (ZH)**: 规格自校正：通过测试时 refinement 减轻上下文奖励作弊 

**Authors**: Víctor Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2507.18742)  

**Abstract**: Language models (LMs) are susceptible to in-context reward hacking, where they exploit flaws in tainted or faulty written specifications or rubrics to achieve high scores without fulfilling the user's true intent. We introduce Specification Self-Correction (SSC), a novel, test-time framework that enables an LM to identify and correct flaws within its own guiding specification. SSC employs a multi-step inference process where the model first generates a response based on a potentially tainted specification, critiques its output, and then revises the specification itself to remove the exploitable loophole. A final, more robust response is then generated using this self-corrected specification. Across experiments spanning creative writing and agentic coding tasks with several LMs, we demonstrate that while models initially game tainted specifications in 50-70\% of cases, the SSC process reduces this vulnerability by over 90\%. This dynamic repair occurs at inference time, requires no weight modification, and leads to more robustly aligned model behavior. Code at this https URL . 

**Abstract (ZH)**: 语言模型（LMs）易遭受上下文内奖励作弊，它们会利用污染或有缺陷的书面规范或评分标准中的漏洞，以高分的形式实现目标而不履行用户的真正意图。我们提出了规范自我修正（SSC）这一新颖的测试时框架，使LM能够识别并修正自身指导规范中的漏洞。SSC采用多步推断过程，模型首先基于可能被污染的规范生成回应，然后批评其输出，并修订规范本身以移除可利用的漏洞。最终，使用经过自我修正的规范生成更为健壯的回应。在涉及多种语言模型的创意写作和自主编程任务的实验中，我们展示了虽然模型在50-70%的情况下会利用污染的规范进行游戏，但SSC过程通过超过90%的漏洞降低显著提高了模型的行为一致性。此动态修复在推断时发生，无需修改权重，并导致更稳健的行为对齐。代码参见此链接：https://github.com/alibaba/Qwen-SSC。 

---
# Learned Single-Pixel Fluorescence Microscopy 

**Title (ZH)**: 学习单像素荧光显微镜 

**Authors**: Serban C. Tudosie, Valerio Gandolfi, Shivaprasad Varakkoth, Andrea Farina, Cosimo D'Andrea, Simon Arridge  

**Link**: [PDF](https://arxiv.org/pdf/2507.18740)  

**Abstract**: Single-pixel imaging has emerged as a key technique in fluorescence microscopy, where fast acquisition and reconstruction are crucial. In this context, images are reconstructed from linearly compressed measurements. In practice, total variation minimisation is still used to reconstruct the image from noisy measurements of the inner product between orthogonal sampling pattern vectors and the original image data. However, data can be leveraged to learn the measurement vectors and the reconstruction process, thereby enhancing compression, reconstruction quality, and speed. We train an autoencoder through self-supervision to learn an encoder (or measurement matrix) and a decoder. We then test it on physically acquired multispectral and intensity data. During acquisition, the learned encoder becomes part of the physical device. Our approach can enhance single-pixel imaging in fluorescence microscopy by reducing reconstruction time by two orders of magnitude, achieving superior image quality, and enabling multispectral reconstructions. Ultimately, learned single-pixel fluorescence microscopy could advance diagnosis and biological research, providing multispectral imaging at a fraction of the cost. 

**Abstract (ZH)**: 单像素成像在荧光显微镜中的新兴关键技术：通过学习提高压缩、重建质量和速度 

---
# Multi-Year Maintenance Planning for Large-Scale Infrastructure Systems: A Novel Network Deep Q-Learning Approach 

**Title (ZH)**: 大规模基础设施系统多年维护规划：一种新型网络深度Q学习方法 

**Authors**: Amir Fard, Arnold X.-X. Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.18732)  

**Abstract**: Infrastructure asset management is essential for sustaining the performance of public infrastructure such as road networks, bridges, and utility networks. Traditional maintenance and rehabilitation planning methods often face scalability and computational challenges, particularly for large-scale networks with thousands of assets under budget constraints. This paper presents a novel deep reinforcement learning (DRL) framework that optimizes asset management strategies for large infrastructure networks. By decomposing the network-level Markov Decision Process (MDP) into individual asset-level MDPs while using a unified neural network architecture, the proposed framework reduces computational complexity, improves learning efficiency, and enhances scalability. The framework directly incorporates annual budget constraints through a budget allocation mechanism, ensuring maintenance plans are both optimal and cost-effective. Through a case study on a large-scale pavement network of 68,800 segments, the proposed DRL framework demonstrates significant improvements over traditional methods like Progressive Linear Programming and genetic algorithms, both in efficiency and network performance. This advancement contributes to infrastructure asset management and the broader application of reinforcement learning in complex, large-scale environments. 

**Abstract (ZH)**: 大型基础设施网络资产管理系统中的新颖深度强化学习框架 

---
# The Right to be Forgotten in Pruning: Unveil Machine Unlearning on Sparse Models 

**Title (ZH)**: 遗忘权在修剪中的应用：揭示稀疏模型中的机器遗忘 

**Authors**: Yang Xiao, Gen Li, Jie Ji, Ruimeng Ye, Xiaolong Ma, Bo Hui  

**Link**: [PDF](https://arxiv.org/pdf/2507.18725)  

**Abstract**: Machine unlearning aims to efficiently eliminate the memory about deleted data from trained models and address the right to be forgotten. Despite the success of existing unlearning algorithms, unlearning in sparse models has not yet been well studied. In this paper, we empirically find that the deleted data has an impact on the pruned topology in a sparse model. Motivated by the observation and the right to be forgotten, we define a new terminology ``un-pruning" to eliminate the impact of deleted data on model pruning. Then we propose an un-pruning algorithm to approximate the pruned topology driven by retained data. We remark that any existing unlearning algorithm can be integrated with the proposed un-pruning workflow and the error of un-pruning is upper-bounded in theory. Also, our un-pruning algorithm can be applied to both structured sparse models and unstructured sparse models. In the experiment, we further find that Membership Inference Attack (MIA) accuracy is unreliable for assessing whether a model has forgotten deleted data, as a small change in the amount of deleted data can produce arbitrary MIA results. Accordingly, we devise new performance metrics for sparse models to evaluate the success of un-pruning. Lastly, we conduct extensive experiments to verify the efficacy of un-pruning with various pruning methods and unlearning algorithms. Our code is released at this https URL. 

**Abstract (ZH)**: 机器去学习旨在高效地从训练模型中消除已删除数据的记忆，并解决被遗忘的权利。尽管现有去学习算法取得了成功，但在稀疏模型中的去学习尚未得到充分研究。在本文中，我们实证发现删除的数据会对稀疏模型中的剪枝拓扑产生影响。受此观察及被遗忘权利的启发，我们定义了一个新的术语“去剪枝”（un-pruning）以消除删除数据对模型剪枝的影响。然后，我们提出了一种由保留数据驱动的去剪枝算法来逼近剪枝拓扑。我们注意到，任何现有的去学习算法都可以与提出的去剪枝工作流结合使用，并且理论上可以将去剪枝的误差上界化。此外，我们的去剪枝算法可以适用于结构稀疏模型和无结构稀疏模型。在实验中，我们进一步发现，成员推理攻击（MIA）的准确性对于评估模型是否已忘记删除的数据是不可靠的，因为删除数据量的微小变化可以导致任意的MIA结果。因此，我们为稀疏模型设计了新的性能度量来评估去剪枝的成功。最后，我们进行了广泛实验以验证不同剪枝方法和去学习算法下去剪枝的有效性。我们的代码发布在该网址：https://。 

---
# Concept Probing: Where to Find Human-Defined Concepts (Extended Version) 

**Title (ZH)**: 概念探查：何处寻找人类定义的概念（扩展版本） 

**Authors**: Manuel de Sousa Ribeiro, Afonso Leote, João Leite  

**Link**: [PDF](https://arxiv.org/pdf/2507.18681)  

**Abstract**: Concept probing has recently gained popularity as a way for humans to peek into what is encoded within artificial neural networks. In concept probing, additional classifiers are trained to map the internal representations of a model into human-defined concepts of interest. However, the performance of these probes is highly dependent on the internal representations they probe from, making identifying the appropriate layer to probe an essential task. In this paper, we propose a method to automatically identify which layer's representations in a neural network model should be considered when probing for a given human-defined concept of interest, based on how informative and regular the representations are with respect to the concept. We validate our findings through an exhaustive empirical analysis over different neural network models and datasets. 

**Abstract (ZH)**: 概念探针作为一种方法，最近受到追捧，它允许人类窥探人工神经网络中编码的内容。在概念探针中，额外的分类器被训练来将模型的内部表示映射到人类定义的概念中。然而，这些探针的性能高度依赖于它们所探针的内部表示，因此确定探针的适当层次是一项必不可少的任务。在本文中，我们提出了一种方法，根据内部表示对给定的人类定义概念的相关性和规律性，自动识别在探针过程中应该考虑的神经网络模型中哪一层的表示。我们通过在不同神经网络模型和数据集上进行全面的实证分析来验证我们的发现。 

---
# Market Making Strategies with Reinforcement Learning 

**Title (ZH)**: reinforcement learning驱动的做市策略 

**Authors**: Óscar Fernández Vicente  

**Link**: [PDF](https://arxiv.org/pdf/2507.18680)  

**Abstract**: This thesis presents the results of a comprehensive research project focused on applying Reinforcement Learning (RL) to the problem of market making in financial markets. Market makers (MMs) play a fundamental role in providing liquidity, yet face significant challenges arising from inventory risk, competition, and non-stationary market dynamics. This research explores how RL, particularly Deep Reinforcement Learning (DRL), can be employed to develop autonomous, adaptive, and profitable market making strategies.
The study begins by formulating the MM task as a reinforcement learning problem, designing agents capable of operating in both single-agent and multi-agent settings within a simulated financial environment. It then addresses the complex issue of inventory management using two complementary approaches: reward engineering and Multi-Objective Reinforcement Learning (MORL). While the former uses dynamic reward shaping to guide behavior, the latter leverages Pareto front optimization to explicitly balance competing objectives.
To address the problem of non-stationarity, the research introduces POW-dTS, a novel policy weighting algorithm based on Discounted Thompson Sampling. This method allows agents to dynamically select and combine pretrained policies, enabling continual adaptation to shifting market conditions.
The experimental results demonstrate that the proposed RL-based approaches significantly outperform traditional and baseline algorithmic strategies across various performance metrics. Overall, this research thesis contributes new methodologies and insights for the design of robust, efficient, and adaptive market making agents, reinforcing the potential of RL to transform algorithmic trading in complex financial systems. 

**Abstract (ZH)**: 本论文 Presents 一种基于强化学习（RL）的市场制作问题研究：将市场制作任务 formulized 为一个强化学习问题，设计能够在单智能体和多智能体设置中运行的智能体，并在模拟金融环境中进行研究。探讨了如何利用强化学习，特别是深度强化学习（DRL），开发自主适应且盈利的市场制作策略。研究通过库存管理问题的两种互补方法——奖励工程和多目标强化学习（MORL）——来应对复杂情况。前者使用动态奖励塑造引导行为，后者利用帕累托前沿优化显式平衡竞争目标。为应对非平稳性问题，研究引入了基于折扣汤普森采样的POW-dTS新策略权重算法，允许智能体动态选择和组合预训练策略，以适应变化的市场条件。实验结果表明，所提出的基于RL的方法在各种性能指标上显著优于传统的和基准算法。本文为设计稳健、高效且适应性强的市场制作智能体提供了新的方法和见解，强调了RL在复杂金融系统中改造算法交易的潜力。 

---
# Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting 

**Title (ZH)**: 面向可扩展的空间智能通过从2D到3D数据提升 

**Authors**: Xingyu Miao, Haoran Duan, Quanhao Qian, Jiuniu Wang, Yang Long, Ling Shao, Deli Zhao, Ran Xu, Gongjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18678)  

**Abstract**: Spatial intelligence is emerging as a transformative frontier in AI, yet it remains constrained by the scarcity of large-scale 3D datasets. Unlike the abundant 2D imagery, acquiring 3D data typically requires specialized sensors and laborious annotation. In this work, we present a scalable pipeline that converts single-view images into comprehensive, scale- and appearance-realistic 3D representations - including point clouds, camera poses, depth maps, and pseudo-RGBD - via integrated depth estimation, camera calibration, and scale calibration. Our method bridges the gap between the vast repository of imagery and the increasing demand for spatial scene understanding. By automatically generating authentic, scale-aware 3D data from images, we significantly reduce data collection costs and open new avenues for advancing spatial intelligence. We release two generated spatial datasets, i.e., COCO-3D and Objects365-v2-3D, and demonstrate through extensive experiments that our generated data can benefit various 3D tasks, ranging from fundamental perception to MLLM-based reasoning. These results validate our pipeline as an effective solution for developing AI systems capable of perceiving, understanding, and interacting with physical environments. 

**Abstract (ZH)**: 空间智能正成为AI领域的一项变革性前沿技术，但其发展受限于大规模3D数据集的稀缺性。与丰富的2D图像相比，获取3D数据通常需要使用专门的传感器并进行繁琐的标注。在此研究中，我们提出了一种可扩展的管道，通过集成深度估计、相机标定和尺度标定将单视角图像转换为全面、尺度和外观真实的3D表示——包括点云、相机姿态、深度图和伪RGBD。我们的方法填补了大量图像存储备和对空间场景理解日益增长的需求之间的差距。通过自动从图像生成具有尺度意识的真实3D数据，我们大大降低了数据收集成本并开辟了推进空间智能的新途径。我们发布了两个生成的空间数据集，即COCO-3D和Objects365-v2-3D，并通过大量实验展示了我们的生成数据可以惠及从基础感知到基于MLLM的推理的各种3D任务。这些结果验证了我们的管道作为开发能够感知、理解和与物理环境交互的AI系统的有效解决方案。 

---
# Innovator: Scientific Continued Pretraining with Fine-grained MoE Upcycling 

**Title (ZH)**: 创新者：科学的持续预训练与细粒度MoE升级 

**Authors**: Ning Liao, Xiaoxing Wang, Zehao Lin, Weiyang Guo, Feng Hong, Shixiang Song, Geng Yu, Zihua Zhao, Sitao Xie, Longxuan Wei, Xiangqi Jin, Xiaohan Qin, Jiale Ma, Kai Chen, Jiangchao Yao, Zhouhan Lin, Junchi Yan, Zhiyu Li, Feiyu Xiong, Yanfeng Wang, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18671)  

**Abstract**: A large language model (LLM) with knowledge in both scientific and general tasks is the foundation of science general intelligence. However, directly continued pretraining an LLM using science data usually leads to catastrophic forgetting, which indicates severe degradation in general ability. In this report, we present Innovator, which solves this problem by upcycling a pre-trained dense LLM into a fine-grained Mixtures-of-Experts model during continued pretraining, where different experts are expected to learn science knowledge in different disciplines, and a shared expert is utilized for general tasks. Innovator introduces a four-stage upcycle training paradigm: (1) Scientific Expert Induction on discipline-specific data, (2) Fine-grained Expert Splitting via FFN dimension decomposition, (3) Science-Aware Routing warmup, and (4) Generalist-Scientist Integration training on hybrid datasets. Such a paradigm enables knowledge in the general domain, and different scientific disciplines can be decoupled, avoiding the negative influence among knowledge in different domains. With 53.3B total parameters and 13.3B activated, Innovator extends Qwen2.5-7B using a shared general expert and 64 specialized scientific experts with 8 activated. Trained on 300B tokens with tri-level quality-controlled data, Innovator achieves 25% average improvement across 30 scientific tasks with a win rate as 70%, while retaining 99% performance in general tasks. Furthermore, Innovator-Reason, which is post-trained from Innovator for reasoning boosting, exhibits excellent reasoning performance in solving complex scientific problems with improvements over 30%. 

**Abstract (ZH)**: 一种大型语言模型（LLM）通过融合科学知识和通用任务知识来构建科学通用智能的基础。然而，直接使用科学数据继续预训练LLM通常会导致严重的泛化能力下降，即灾难性遗忘。本报告介绍了Innovator，该模型通过在继续预训练过程中将一个预先训练的密集LLM升级为细粒度的混合专家模型来解决此问题，在此过程中，不同的专家被期望在不同的学科中学习科学知识，而共享专家则用于通用任务。Innovator引入了一种四阶段升级训练范式：（1）学科特定数据的科学专家诱导；（2）通过FFN维度分解进行细粒度专家分割；（3）科学意识路由预热；（4）综合通用专家和专业科学专家的泛化-科学家集成训练。这种范式使得通用领域和不同科学学科的知识能够分离，避免了不同领域知识之间的负面影响。Innovator拥有总计533亿个参数，激活参数为133亿，通过共享一个通用专家和64个专业科学专家（8个激活）来扩展Qwen2.5-7B。在300亿个令牌的分级质量控制数据集上训练，Innovator在30项科学任务中实现了25%的平均改进和70%的胜率，同时保留了99%的通用任务性能。此外，用于提升推理能力的Innovator-Reason，在解决复杂科学问题方面表现出色，并实现了超过30%的改进。 

---
# Efficient Knowledge Tracing Leveraging Higher-Order Information in Integrated Graphs 

**Title (ZH)**: 利用综合图中高阶信息的高效知识追踪 

**Authors**: Donghee Han, Daehee Kim, Minjun Lee, Daeyoung Roh, Keejun Han, Mun Yong Yi  

**Link**: [PDF](https://arxiv.org/pdf/2507.18668)  

**Abstract**: The rise of online learning has led to the development of various knowledge tracing (KT) methods. However, existing methods have overlooked the problem of increasing computational cost when utilizing large graphs and long learning sequences. To address this issue, we introduce Dual Graph Attention-based Knowledge Tracing (DGAKT), a graph neural network model designed to leverage high-order information from subgraphs representing student-exercise-KC relationships. DGAKT incorporates a subgraph-based approach to enhance computational efficiency. By processing only relevant subgraphs for each target interaction, DGAKT significantly reduces memory and computational requirements compared to full global graph models. Extensive experimental results demonstrate that DGAKT not only outperforms existing KT models but also sets a new standard in resource efficiency, addressing a critical need that has been largely overlooked by prior KT approaches. 

**Abstract (ZH)**: 基于双图注意机制的知识追踪模型：DGAKT 

---
# Gen-AI Police Sketches with Stable Diffusion 

**Title (ZH)**: 基于Gen-AI的警察画像生成：稳定扩散方法 

**Authors**: Nicholas Fidalgo, Aaron Contreras, Katherine Harvey, Johnny Ni  

**Link**: [PDF](https://arxiv.org/pdf/2507.18667)  

**Abstract**: This project investigates the use of multimodal AI-driven approaches to automate and enhance suspect sketching. Three pipelines were developed and evaluated: (1) baseline image-to-image Stable Diffusion model, (2) same model integrated with a pre-trained CLIP model for text-image alignment, and (3) novel approach incorporating LoRA fine-tuning of the CLIP model, applied to self-attention and cross-attention layers, and integrated with Stable Diffusion. An ablation study confirmed that fine-tuning both self- and cross-attention layers yielded the best alignment between text descriptions and sketches. Performance testing revealed that Model 1 achieved the highest structural similarity (SSIM) of 0.72 and a peak signal-to-noise ratio (PSNR) of 25 dB, outperforming Model 2 and Model 3. Iterative refinement enhanced perceptual similarity (LPIPS), with Model 3 showing improvement over Model 2 but still trailing Model 1. Qualitatively, sketches generated by Model 1 demonstrated the clearest facial features, highlighting its robustness as a baseline despite its simplicity. 

**Abstract (ZH)**: 本项目探究了多模态AI驱动方法自动化并增强嫌犯画像的应用。三个工作流程被开发和评估：（1）基线图像到图像的Stable Diffusion模型；（2）该模型与预训练的CLIP模型集成以实现文本与图像对齐；（3）结合LoRA fine-tuning的CLIP模型应用于自注意力和跨注意力层，并与Stable Diffusion集成的新型方法。消融研究证实同时fine-tuning自注意力和跨注意力层在文本描述与画像对齐方面效果最佳。性能测试显示，模型1在结构相似性（SSIM）为0.72和峰值信噪比（PSNR）为25 dB方面表现最佳，优于模型2和模型3。迭代细化提升了感知相似性（LPIPS），模型3优于模型2但仍然落后于模型1。从定性的角度来看，模型1生成的画像面部特征最为清晰，尽管简单但表现出较高的稳健性。 

---
# Quantum-Cognitive Tunnelling Neural Networks for Military-Civilian Vehicle Classification and Sentiment Analysis 

**Title (ZH)**: 量子认知隧道神经网络在军事民用车辆分类与情感分析中的应用 

**Authors**: Milan Maksimovic, Anna Bohdanets, Immaculate Motsi-Omoijiade, Guido Governatori, Ivan S. Maksymov  

**Link**: [PDF](https://arxiv.org/pdf/2507.18645)  

**Abstract**: Prior work has demonstrated that incorporating well-known quantum tunnelling (QT) probability into neural network models effectively captures important nuances of human perception, particularly in the recognition of ambiguous objects and sentiment analysis. In this paper, we employ novel QT-based neural networks and assess their effectiveness in distinguishing customised CIFAR-format images of military and civilian vehicles, as well as sentiment, using a proprietary military-specific vocabulary. We suggest that QT-based models can enhance multimodal AI applications in battlefield scenarios, particularly within human-operated drone warfare contexts, imbuing AI with certain traits of human reasoning. 

**Abstract (ZH)**: 已有研究表明，将广为人知的量子隧穿(QT)概率融入神经网络模型中，能有效地捕捉人类感知中的重要细微差异，特别是在识别模糊对象和情感分析方面。本文中，我们利用新颖的基于QT的神经网络，并评估其在区分特定军事和民用车辆的自定义CIFAR格式图像以及情感方面的有效性，同时使用专有的军事专用词汇。我们建议，基于QT的模型能够增强战场场景中的多模态AI应用，特别是在有人驾驶无人机作战背景下，使AI具备某些人类推理的特性。 

---
# How good are humans at detecting AI-generated images? Learnings from an experiment 

**Title (ZH)**: 人类如何检测AI生成的图像？一项实验的启示 

**Authors**: Thomas Roca, Anthony Cintron Roman, Jehú Torres Vega, Marcelo Duarte, Pengce Wang, Kevin White, Amit Misra, Juan Lavista Ferres  

**Link**: [PDF](https://arxiv.org/pdf/2507.18640)  

**Abstract**: As AI-powered image generation improves, a key question is how well human beings can differentiate between "real" and AI-generated or modified images. Using data collected from the online game "Real or Not Quiz.", this study investigates how effectively people can distinguish AI-generated images from real ones. Participants viewed a randomized set of real and AI-generated images, aiming to identify their authenticity. Analysis of approximately 287,000 image evaluations by over 12,500 global participants revealed an overall success rate of only 62\%, indicating a modest ability, slightly above chance. Participants were most accurate with human portraits but struggled significantly with natural and urban landscapes. These results highlight the inherent challenge humans face in distinguishing AI-generated visual content, particularly images without obvious artifacts or stylistic cues. This study stresses the need for transparency tools, such as watermarks and robust AI detection tools to mitigate the risks of misinformation arising from AI-generated content 

**Abstract (ZH)**: 随着AI驱动的图像生成技术的进步，一个关键问题是人类如何区分“真实”图像与AI生成或修改的图像。利用来自在线游戏“真实或假象测验”收集的数据，本研究调查了人们区分AI生成图像与真实图像的有效性。参与者被随机呈现真实和AI生成的图像，旨在识别这些图像的真实性。分析显示，超过12,500名全球参与者的约287,000次图像评估的总体准确率为62%，表明人们的辨别能力仅略有提高，略高于随机水平。参与者在识别人类肖像方面最准确，但在识别自然和城市景观方面遇到了显著困难。这些结果突显了人类在区分AI生成的视觉内容（尤其是没有明显瑕疵或风格线索的图像）方面的固有挑战。本研究强调了需要透明度工具（如水印和 robust AI检测工具）以减轻由AI生成内容引发的虚假信息风险。 

---
# Prompt Engineering and the Effectiveness of Large Language Models in Enhancing Human Productivity 

**Title (ZH)**: Prompt工程及其对增强人类生产力的大规模语言模型效果研究 

**Authors**: Rizal Khoirul Anam  

**Link**: [PDF](https://arxiv.org/pdf/2507.18638)  

**Abstract**: The widespread adoption of large language models (LLMs) such as ChatGPT, Gemini, and DeepSeek has significantly changed how people approach tasks in education, professional work, and creative domains. This paper investigates how the structure and clarity of user prompts impact the effectiveness and productivity of LLM outputs. Using data from 243 survey respondents across various academic and occupational backgrounds, we analyze AI usage habits, prompting strategies, and user satisfaction. The results show that users who employ clear, structured, and context-aware prompts report higher task efficiency and better outcomes. These findings emphasize the essential role of prompt engineering in maximizing the value of generative AI and provide practical implications for its everyday use. 

**Abstract (ZH)**: 大语言模型（LLM）如ChatGPT、Gemini和DeepSeek的广泛应用已显著改变了人们在教育、职业工作和创造性领域中的任务处理方式。本文研究了用户提示的结构和清晰度如何影响LLM输出的有效性和生产力。通过分析来自各个学术和职业背景的243名调查受访者的数据，我们探讨了AI使用习惯、提示策略和用户满意度。研究结果表明，使用清晰、结构化且上下文相关的提示的用户报告更高的任务效率和更好的成果。这些发现强调了提示工程在最大化生成式AI价值中的重要作用，并提供了其实用建议。 

---
# More Expert-like Eye Gaze Movement Patterns are Related to Better X-ray Reading 

**Title (ZH)**: 更接近专家的眼球凝视运动模式与更好的X射线阅读相关 

**Authors**: Pingjing Yang, Jennifer Cromley, Jana Diesner  

**Link**: [PDF](https://arxiv.org/pdf/2507.18637)  

**Abstract**: Understanding how novices acquire and hone visual search skills is crucial for developing and optimizing training methods across domains. Network analysis methods can be used to analyze graph representations of visual expertise. This study investigates the relationship between eye-gaze movements and learning outcomes among undergraduate dentistry students who were diagnosing dental radiographs over multiple semesters. We use network analysis techniques to model eye-gaze scanpaths as directed graphs and examine changes in network metrics over time. Using time series clustering on each metric, we identify distinct patterns of visual search strategies and explore their association with students' diagnostic performance. Our findings suggest that the network metric of transition entropy is negatively correlated with performance scores, while the number of nodes and edges as well as average PageRank are positively correlated with performance scores. Changes in network metrics for individual students over time suggest a developmental shift from intermediate to expert-level processing. These insights contribute to understanding expertise acquisition in visual tasks and can inform the design of AI-assisted learning interventions. 

**Abstract (ZH)**: 理解初学者在诊断牙科X光片过程中获取和提高视觉搜索技能的方式对于跨领域开发和优化培训方法至关重要。网络分析方法可用于分析视觉专长的图表示。本研究探讨了本科生在多个学期诊断牙科X光片过程中眼注视运动与学习成果之间的关系。我们使用网络分析技术将眼注视扫描路径建模为有向图，并检查随着时间的变化网络度量的变化。通过时间序列聚类分析每个度量，我们识别出不同的视觉搜索策略模式，并探索它们与学生诊断表现之间的关系。研究发现，网络度量转换熵与绩效评分负相关，而节点数和边数以及平均PageRank与绩效评分正相关。单个学生随时间网络度量的变化表明，从中间水平到专家水平的处理发生了发展性转变。这些见解有助于理解视觉任务中的专长获取，并可以指导AI辅助学习干预的设计。 

---
