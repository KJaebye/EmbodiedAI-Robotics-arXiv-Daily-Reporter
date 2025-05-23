# MEbots: Integrating a RISC-V Virtual Platform with a Robotic Simulator for Energy-aware Design 

**Title (ZH)**: MEbots：集成RISC-V虚拟平台的机器人模拟器能效感知设计 

**Authors**: Giovanni Pollo, Mohamed Amine Hamdi, Matteo Risso, Lorenzo Ruotolo, Pietro Furbatto, Matteo Isoldi, Yukai Chen, Alessio Burrello, Enrico Macii, Massimo Poncino, Daniele Jahier Pagliari, Sara Vinco  

**Link**: [PDF](https://arxiv.org/pdf/2505.16682)  

**Abstract**: Virtual Platforms (VPs) enable early software validation of autonomous systems' electronics, reducing costs and time-to-market. While many VPs support both functional and non-functional simulation (e.g., timing, power), they lack the capability of simulating the environment in which the system operates. In contrast, robotics simulators lack accurate timing and power features. This twofold shortcoming limits the effectiveness of the design flow, as the designer can not fully evaluate the features of the solution under development. This paper presents a novel, fully open-source framework bridging this gap by integrating a robotics simulator (Webots) with a VP for RISC-V-based systems (MESSY). The framework enables a holistic, mission-level, energy-aware co-simulation of electronics in their surrounding environment, streamlining the exploration of design configurations and advanced power management policies. 

**Abstract (ZH)**: 虚拟平台与机器人模拟器集成的开源框架：面向RISC-V系统的环境感知联合仿真 

---
# Behavioral Safety Assessment towards Large-scale Deployment of Autonomous Vehicles 

**Title (ZH)**: 面向自动驾驶大规模部署的行为安全评估 

**Authors**: Henry X. Liu, Xintao Yan, Haowei Sun, Tinghan Wang, Zhijie Qiao, Haojie Zhu, Shengyin Shen, Shuo Feng, Greg Stevens, Greg McGuire  

**Link**: [PDF](https://arxiv.org/pdf/2505.16214)  

**Abstract**: Autonomous vehicles (AVs) have significantly advanced in real-world deployment in recent years, yet safety continues to be a critical barrier to widespread adoption. Traditional functional safety approaches, which primarily verify the reliability, robustness, and adequacy of AV hardware and software systems from a vehicle-centric perspective, do not sufficiently address the AV's broader interactions and behavioral impact on the surrounding traffic environment. To overcome this limitation, we propose a paradigm shift toward behavioral safety, a comprehensive approach focused on evaluating AV responses and interactions within the traffic environment. To systematically assess behavioral safety, we introduce a third-party AV safety assessment framework comprising two complementary evaluation components: the Driver Licensing Test and the Driving Intelligence Test. The Driver Licensing Test evaluates the AV's reactive behaviors under controlled scenarios, ensuring basic behavioral competency. In contrast, the Driving Intelligence Test assesses the AV's interactive behaviors within naturalistic traffic conditions, quantifying the frequency of safety-critical events to deliver statistically meaningful safety metrics before large-scale deployment. We validated our proposed framework using this http URL, an open-source Level 4 AV, tested both in simulated environments and on the physical test track at the University of Michigan's Mcity Testing Facility. The results indicate that this http URL passed 6 out of 14 scenarios and exhibited a crash rate of 3.01e-3 crashes per mile, approximately 1,000 times higher than the average human driver crash rate. During the tests, we also uncovered several unknown unsafe scenarios for this http URL. These findings underscore the necessity of behavioral safety evaluations for improving AV safety performance prior to widespread public deployment. 

**Abstract (ZH)**: 自主驾驶车辆的行为安全性评估框架 

---
# Human Workload Prediction: Lag Horizon Selection 

**Title (ZH)**: 人类工作负载预测：滞后时间窗选择 

**Authors**: Mark-Robin Giolando, Julie A. Adams  

**Link**: [PDF](https://arxiv.org/pdf/2505.15939)  

**Abstract**: Human-robot teams must be aware of human workload when operating in uncertain, dynamic environments. Prior work employed physiological response metrics from wearable sensors to estimate the current human workload; however, these estimates only enable robots to respond to under- or overload conditions reactively. Current human workload prediction approaches are limited to short prediction horizons and fail to investigate variable lag horizons' impact on predictions. This letter investigates the impact of lag horizons on both univariate and multivariate time series forecasting models for human workload prediction. A key finding is that univariate predictions required longer lag horizons of 240 seconds (s), whereas multivariate workload predictions sufficed with shorter lag horizons with diminishing returns around 120s. 

**Abstract (ZH)**: 人类和机器人团队在不确定、动态环境中操作时需要意识到人类的工作负荷。本信研究了滞后时间窗对单变量和多变量时间序列工作负荷预测模型的影响。关键发现是，单变量预测需要较长的滞后时间窗（240秒），而多变量工作负荷预测在约120秒时具有递减的回报。 

---
# Efficient Online RL Fine Tuning with Offline Pre-trained Policy Only 

**Title (ZH)**: 基于离线预训练策略的高效在线RL微调 

**Authors**: Wei Xiao, Jiacheng Liu, Zifeng Zhuang, Runze Suo, Shangke Lyu, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16856)  

**Abstract**: Improving the performance of pre-trained policies through online reinforcement learning (RL) is a critical yet challenging topic. Existing online RL fine-tuning methods require continued training with offline pretrained Q-functions for stability and performance. However, these offline pretrained Q-functions commonly underestimate state-action pairs beyond the offline dataset due to the conservatism in most offline RL methods, which hinders further exploration when transitioning from the offline to the online setting. Additionally, this requirement limits their applicability in scenarios where only pre-trained policies are available but pre-trained Q-functions are absent, such as in imitation learning (IL) pre-training. To address these challenges, we propose a method for efficient online RL fine-tuning using solely the offline pre-trained policy, eliminating reliance on pre-trained Q-functions. We introduce PORL (Policy-Only Reinforcement Learning Fine-Tuning), which rapidly initializes the Q-function from scratch during the online phase to avoid detrimental pessimism. Our method not only achieves competitive performance with advanced offline-to-online RL algorithms and online RL approaches that leverage data or policies prior, but also pioneers a new path for directly fine-tuning behavior cloning (BC) policies. 

**Abstract (ZH)**: 通过在线强化学习提高预训练策略性能：一种仅依赖预训练策略的高效在线RL微调方法 

---
# Identifying, Evaluating, and Mitigating Risks of AI Thought Partnerships 

**Title (ZH)**: 识别、评估和缓解AI思想合作伙伴关系的风险 

**Authors**: Kerem Oktar, Katherine M. Collins, Jose Hernandez-Orallo, Diane Coyle, Stephen Cave, Adrian Weller, Ilia Sucholutsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.16899)  

**Abstract**: Artificial Intelligence (AI) systems have historically been used as tools that execute narrowly defined tasks. Yet recent advances in AI have unlocked possibilities for a new class of models that genuinely collaborate with humans in complex reasoning, from conceptualizing problems to brainstorming solutions. Such AI thought partners enable novel forms of collaboration and extended cognition, yet they also pose major risks-including and beyond risks of typical AI tools and agents. In this commentary, we systematically identify risks of AI thought partners through a novel framework that identifies risks at multiple levels of analysis, including Real-time, Individual, and Societal risks arising from collaborative cognition (RISc). We leverage this framework to propose concrete metrics for risk evaluation, and finally suggest specific mitigation strategies for developers and policymakers. As AI thought partners continue to proliferate, these strategies can help prevent major harms and ensure that humans actively benefit from productive thought partnerships. 

**Abstract (ZH)**: AI思考伙伴的风险及其应对策略：从协作认知（RISc）框架下的多层次分析 

---
# Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings 

**Title (ZH)**: 基于谓词条件的置信区间回答集嵌入 

**Authors**: Yuqicheng Zhu, Daniel Hernández, Yuan He, Zifeng Ding, Bo Xiong, Evgeny Kharlamov, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2505.16877)  

**Abstract**: Uncertainty quantification in Knowledge Graph Embedding (KGE) methods is crucial for ensuring the reliability of downstream applications. A recent work applies conformal prediction to KGE methods, providing uncertainty estimates by generating a set of answers that is guaranteed to include the true answer with a predefined confidence level. However, existing methods provide probabilistic guarantees averaged over a reference set of queries and answers (marginal coverage guarantee). In high-stakes applications such as medical diagnosis, a stronger guarantee is often required: the predicted sets must provide consistent coverage per query (conditional coverage guarantee). We propose CondKGCP, a novel method that approximates predicate-conditional coverage guarantees while maintaining compact prediction sets. CondKGCP merges predicates with similar vector representations and augments calibration with rank information. We prove the theoretical guarantees and demonstrate empirical effectiveness of CondKGCP by comprehensive evaluations. 

**Abstract (ZH)**: 知识图嵌入方法中不确定性量化对于确保下游应用的可靠性至关重要。一项近期工作将符合性预测应用于知识图嵌入方法，通过生成一个保证包含真实答案的置信区间内的答案集合来提供不确定性估计。现有方法提供基于参考查询和答案集合的边际覆盖概率保证。在医疗诊断等高风险应用中，通常需要更强的保证：预测集必须为每个查询提供一致的覆盖（条件覆盖保证）。我们提出了一种名为CondKGCP的新方法，该方法近似预测的条件覆盖保证，同时保持预测集的紧凑性。CondKGCP结合了具有相似向量表示的谓词，并通过排名信息增强校准。我们证明了CondKGCP的理论保证，并通过全面评估展示了其实验有效性。 

---
# From EduVisBench to EduVisAgent: A Benchmark and Multi-Agent Framework for Pedagogical Visualization 

**Title (ZH)**: 从EduVisBench到EduVisAgent：教学可视化基准及多智能体框架 

**Authors**: Haonian Ji, Shi Qiu, Siyang Xin, Siwei Han, Zhaorun Chen, Hongyi Wang, Dake Zhang, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16832)  

**Abstract**: While foundation models (FMs), such as diffusion models and large vision-language models (LVLMs), have been widely applied in educational contexts, their ability to generate pedagogically effective visual explanations remains limited. Most existing approaches focus primarily on textual reasoning, overlooking the critical role of structured and interpretable visualizations in supporting conceptual understanding. To better assess the visual reasoning capabilities of FMs in educational settings, we introduce EduVisBench, a multi-domain, multi-level benchmark. EduVisBench features diverse STEM problem sets requiring visually grounded solutions, along with a fine-grained evaluation rubric informed by pedagogical theory. Our empirical analysis reveals that existing models frequently struggle with the inherent challenge of decomposing complex reasoning and translating it into visual representations aligned with human cognitive processes. To address these limitations, we propose EduVisAgent, a multi-agent collaborative framework that coordinates specialized agents for instructional planning, reasoning decomposition, metacognitive prompting, and visualization design. Experimental results show that EduVisAgent substantially outperforms all baselines, achieving a 40.2% improvement and delivering more educationally aligned visualizations. EduVisBench and EduVisAgent are available at this https URL and this https URL. 

**Abstract (ZH)**: 尽管基础模型（如扩散模型和大型视觉语言模型）已经在教育场景中广泛应用，但在生成教学有效的视觉解释方面的能力仍有限。现有方法大多主要侧重于文本推理，忽略了结构化和可解释的可视化在支持概念理解中的关键作用。为了更好地评估基础模型在教育场景中的视觉推理能力，我们提出了EduVisBench，一个多领域、多层级基准。EduVisBench 包含了需要视觉 grounding 解决方案的多样化 STEM 问题集，并结合了基于教育理论的细粒度评估标准。实证分析显示，现有模型在分解复杂推理并将其转化为与人类认知过程相一致的视觉表示方面经常遇到困难。为了解决这些局限性，我们提出了EduVisAgent，一个多智能体协作框架，协调专门的智能体进行教学规划、推理分解、元认知提示和可视化设计。实验结果表明，EduVisAgent 显著优于所有基线，实现了40.2%的改进，并提供了更多教育导向的视觉表示。EduVisBench 和 EduVisAgent 可通过以下链接获取：[该链接] 和 [该链接]。 

---
# GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent 

**Title (ZH)**: GUI-explorer: 自主探索与挖掘具有状态转换意识的知识的GUI代理 

**Authors**: Bin Xie, Rui Shao, Gongwei Chen, Kaiwen Zhou, Yinchuan Li, Jie Liu, Min Zhang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.16827)  

**Abstract**: GUI automation faces critical challenges in dynamic environments. MLLMs suffer from two key issues: misinterpreting UI components and outdated knowledge. Traditional fine-tuning methods are costly for app-specific knowledge updates. We propose GUI-explorer, a training-free GUI agent that incorporates two fundamental mechanisms: (1) Autonomous Exploration of Function-aware Trajectory. To comprehensively cover all application functionalities, we design a Function-aware Task Goal Generator that automatically constructs exploration goals by analyzing GUI structural information (e.g., screenshots and activity hierarchies). This enables systematic exploration to collect diverse trajectories. (2) Unsupervised Mining of Transition-aware Knowledge. To establish precise screen-operation logic, we develop a Transition-aware Knowledge Extractor that extracts effective screen-operation logic through unsupervised analysis the state transition of structured interaction triples (observation, action, outcome). This eliminates the need for human involvement in knowledge extraction. With a task success rate of 53.7% on SPA-Bench and 47.4% on AndroidWorld, GUI-explorer shows significant improvements over SOTA agents. It requires no parameter updates for new apps. GUI-explorer is open-sourced and publicly available at this https URL. 

**Abstract (ZH)**: GUI自动化在动态环境中面临关键挑战。MLLMs遭受两大关键问题：UI组件误读和过时的知识。传统的细调方法对于应用程序特定知识的更新成本较高。我们提出GUI-explorer，一种无需训练的GUI智能体，整合了两种基本机制：（1）功能导向的自主探索轨迹。为全面覆盖所有应用程序功能，我们设计了一个功能导向的任务目标生成器，该生成器通过分析GUI结构信息（如屏幕截图和活动层级）自动构建探索目标，从而实现系统性探索以收集多样化轨迹。（2）基于转换的无监督知识挖掘。为建立精确的屏幕操作逻辑，我们开发了一个基于转换的知识提取器，该提取器通过无监督分析结构化的交互三元组（观察、动作、结果）的状态转换来提取有效的屏幕操作逻辑。这消除了知识抽取过程中的人工参与需求。GUI-explorer在SPA-Bench上的任务成功率达到了53.7%，在AndroidWorld上的成功率达到了47.4%，显示出对当前最佳代理的显著改进。它无需为新应用更新参数。GUI-explorer开源并公开可用于此[链接]。 

---
# KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning 

**Title (ZH)**: KTAE：一种无需模型的关键-token 优势估计算法在数学推理中的应用 

**Authors**: Wei Sun, Wen Yang, Pu Jian, Qianlong Du, Fuwei Cui, Shuo Ren, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16826)  

**Abstract**: Recent advances have demonstrated that integrating reinforcement learning with rule-based rewards can significantly enhance the reasoning capabilities of large language models, even without supervised fine-tuning. However, prevalent reinforcement learning algorithms such as GRPO and its variants like DAPO, suffer from a coarse granularity issue when computing the advantage. Specifically, they compute rollout-level advantages that assign identical values to every token within a sequence, failing to capture token-specific contributions and hindering effective learning. To address this limitation, we propose Key-token Advantage Estimation (KTAE) - a novel algorithm that estimates fine-grained, token-level advantages without introducing additional models. KTAE leverages the correctness of sampled rollouts and applies statistical analysis to quantify the importance of individual tokens within a sequence to the final outcome. This quantified token-level importance is then combined with the rollout-level advantage to obtain a more fine-grained token-level advantage estimation. Empirical results show that models trained with GRPO+KTAE and DAPO+KTAE outperform baseline methods across five mathematical reasoning benchmarks. Notably, they achieve higher accuracy with shorter responses and even surpass R1-Distill-Qwen-1.5B using the same base model. 

**Abstract (ZH)**: Recent Advances in Key-token Advantage Estimation for Enhancing Reasoning Capabilities of Large Language Models Without Supervised Fine-tuning 

---
# Fuzzy Information Evolution with Three-Way Decision in Social Network Group Decision-Making 

**Title (ZH)**: 基于三元决策的社交网络群体决策中模糊信息演化 

**Authors**: Qianlei Jia, Xinliang Zhou, Ondrej Krejcar, Enrique Herrera-Viedma  

**Link**: [PDF](https://arxiv.org/pdf/2505.16781)  

**Abstract**: In group decision-making (GDM) scenarios, uncertainty, dynamic social structures, and vague information present major challenges for traditional opinion dynamics models. To address these issues, this study proposes a novel social network group decision-making (SNGDM) framework that integrates three-way decision (3WD) theory, dynamic network reconstruction, and linguistic opinion representation. First, the 3WD mechanism is introduced to explicitly model hesitation and ambiguity in agent judgments, thereby preventing irrational decisions. Second, a connection adjustment rule based on opinion similarity is developed, enabling agents to adaptively update their communication links and better reflect the evolving nature of social relationships. Third, linguistic terms are used to describe agent opinions, allowing the model to handle subjective, vague, or incomplete information more effectively. Finally, an integrated multi-agent decision-making framework is constructed, which simultaneously considers individual uncertainty, opinion evolution, and network dynamics. The proposed model is applied to a multi-UAV cooperative decision-making scenario, where simulation results and consensus analysis demonstrate its effectiveness. Experimental comparisons further verify the advantages of the algorithm in enhancing system stability and representing realistic decision-making behaviors. 

**Abstract (ZH)**: 在群决策制定（GDM）场景中，传统意见动力学模型面临着不确定性、动态社会结构和模糊信息的重大挑战。为应对这些问题，本研究提出了一种融合三元决策（3WD）理论、动态网络重构和语言意见表示的新颖社会网络群决策制定（SNGDM）框架。首先，引入3WD机制以明确建模代理人的犹豫和模糊性，从而避免不理智的决策。其次，基于意见相似性的连接调整规则被开发出来，使代理能够适应性更新其通信链路，更好地反映社交关系的动态变化。第三，使用语言术语来描述代理人的意见，使得模型能够更有效地处理主观、模糊或不完整的信息。最后，构建了一个综合性的多-agent决策制定框架，同时考虑个体不确定性、意见演变和网络动态。提出的模型应用于多-UAV协同决策制定场景，仿真结果和共识分析证明了其有效性。实验比较进一步验证了该算法在增强系统稳定性和代表现实决策行为方面的优势。 

---
# Data-Driven Breakthroughs and Future Directions in AI Infrastructure: A Comprehensive Review 

**Title (ZH)**: 数据驱动的人工智能基础设施突破与未来方向：全面回顾 

**Authors**: Beyazit Bestami Yuksel, Ayse Yilmazer Metin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16771)  

**Abstract**: This paper presents a comprehensive synthesis of major breakthroughs in artificial intelligence (AI) over the past fifteen years, integrating historical, theoretical, and technological perspectives. It identifies key inflection points in AI' s evolution by tracing the convergence of computational resources, data access, and algorithmic innovation. The analysis highlights how researchers enabled GPU based model training, triggered a data centric shift with ImageNet, simplified architectures through the Transformer, and expanded modeling capabilities with the GPT series. Rather than treating these advances as isolated milestones, the paper frames them as indicators of deeper paradigm shifts. By applying concepts from statistical learning theory such as sample complexity and data efficiency, the paper explains how researchers translated breakthroughs into scalable solutions and why the field must now embrace data centric approaches. In response to rising privacy concerns and tightening regulations, the paper evaluates emerging solutions like federated learning, privacy enhancing technologies (PETs), and the data site paradigm, which reframe data access and security. In cases where real world data remains inaccessible, the paper also assesses the utility and constraints of mock and synthetic data generation. By aligning technical insights with evolving data infrastructure, this study offers strategic guidance for future AI research and policy development. 

**Abstract (ZH)**: 过去十五年人工智能重大突破的综合综述：从计算资源、数据访问和算法创新的融合视角探讨关键拐点及深远范式转变 

---
# SPaRC: A Spatial Pathfinding Reasoning Challenge 

**Title (ZH)**: SPaRC: 空间路径推理挑战 

**Authors**: Lars Benedikt Kaesberg, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2505.16686)  

**Abstract**: Existing reasoning datasets saturate and fail to test abstract, multi-step problems, especially pathfinding and complex rule constraint satisfaction. We introduce SPaRC (Spatial Pathfinding Reasoning Challenge), a dataset of 1,000 2D grid pathfinding puzzles to evaluate spatial and symbolic reasoning, requiring step-by-step planning with arithmetic and geometric rules. Humans achieve near-perfect accuracy (98.0%; 94.5% on hard puzzles), while the best reasoning models, such as o4-mini, struggle (15.8%; 1.1% on hard puzzles). Models often generate invalid paths (>50% of puzzles for o4-mini), and reasoning tokens reveal they make errors in navigation and spatial logic. Unlike humans, who take longer on hard puzzles, models fail to scale test-time compute with difficulty. Allowing models to make multiple solution attempts improves accuracy, suggesting potential for better spatial reasoning with improved training and efficient test-time scaling methods. SPaRC can be used as a window into models' spatial reasoning limitations and drive research toward new methods that excel in abstract, multi-step problem-solving. 

**Abstract (ZH)**: SPaRC：空间路径推理挑战 

---
# Open and Sustainable AI: challenges, opportunities and the road ahead in the life sciences 

**Title (ZH)**: 开放和可持续人工智能：生命科学领域面临的挑战、机遇及前行之路 

**Authors**: Gavin Farrell, Eleni Adamidi, Rafael Andrade Buono, Mihail Anton, Omar Abdelghani Attafi, Salvador Capella Gutierrez, Emidio Capriotti, Leyla Jael Castro, Davide Cirillo, Lisa Crossman, Christophe Dessimoz, Alexandros Dimopoulos, Raul Fernandez-Diaz, Styliani-Christina Fragkouli, Carole Goble, Wei Gu, John M. Hancock, Alireza Khanteymoori, Tom Lenaerts, Fabio G. Liberante, Peter Maccallum, Alexander Miguel Monzon, Magnus Palmblad, Lucy Poveda, Ovidiu Radulescu, Denis C. Shields, Shoaib Sufi, Thanasis Vergoulis, Fotis Psomopoulos, Silvio C.E. Tosatto  

**Link**: [PDF](https://arxiv.org/pdf/2505.16619)  

**Abstract**: Artificial intelligence (AI) has recently seen transformative breakthroughs in the life sciences, expanding possibilities for researchers to interpret biological information at an unprecedented capacity, with novel applications and advances being made almost daily. In order to maximise return on the growing investments in AI-based life science research and accelerate this progress, it has become urgent to address the exacerbation of long-standing research challenges arising from the rapid adoption of AI methods. We review the increased erosion of trust in AI research outputs, driven by the issues of poor reusability and reproducibility, and highlight their consequent impact on environmental sustainability. Furthermore, we discuss the fragmented components of the AI ecosystem and lack of guiding pathways to best support Open and Sustainable AI (OSAI) model development. In response, this perspective introduces a practical set of OSAI recommendations directly mapped to over 300 components of the AI ecosystem. Our work connects researchers with relevant AI resources, facilitating the implementation of sustainable, reusable and transparent AI. Built upon life science community consensus and aligned to existing efforts, the outputs of this perspective are designed to aid the future development of policy and structured pathways for guiding AI implementation. 

**Abstract (ZH)**: 人工智能（AI）在生命科学领域 recently 见证了转型性的突破，极大地扩展了研究人员以前所未有的能力解释生物信息的可能性，新型应用和进展几乎每天都有所突破。为了最大化对基于AI的生命科学研究日益增加的投资回报并加速这一进程，亟需解决快速采用AI方法带来的长期研究挑战。本文回顾了AI研究输出信任度下降的问题，这些问题主要由可重用性和可再现性差所驱动，并强调了其对环境可持续性的负面影响。此外，还讨论了AI生态系统碎片化的组成部分以及缺乏指导路径来最好地支持开放和可持续AI（OSAI）模型的开发。为此，本文介绍了一套实用的OSAI建议，直接映射到AI生态系统超过300个组成部分。我们的工作将研究人员与相关AI资源连接起来，促进可持续、可重用和透明AI的实施。本文的成果基于生命科学界的共识，并与现有努力保持一致，旨在为指导AI实施提供未来的政策和结构化路径。 

---
# Relevance for Stability of Verification Status of a Set of Arguments in Incomplete Argumentation Frameworks (with Proofs) 

**Title (ZH)**: 不完满论辩框架中论证集验证状态的相关性与稳定性（附证明） 

**Authors**: Anshu Xiong, Songmao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16507)  

**Abstract**: The notion of relevance was proposed for stability of justification status of a single argument in incomplete argumentation frameworks (IAFs) in 2024 by Odekerken et al. To extend the notion, we study the relevance for stability of verification status of a set of arguments in this paper, i.e., the uncertainties in an IAF that have to be resolved in some situations so that answering whether a given set of arguments is an extension obtains the same result in every completion of the IAF. Further we propose the notion of strong relevance for describing the necessity of resolution in all situations reaching stability. An analysis of complexity reveals that detecting the (strong) relevance for stability of sets of arguments can be accomplished in P time under the most semantics discussed in the paper. We also discuss the difficulty in finding tractable methods for relevance detection under grounded semantics. 

**Abstract (ZH)**: 不完备论证框架中单个论证稳定性的相关性概念于2024年由Odekerken等提出，本文研究了相关性在不完备论证框架中一组论证的验证稳定性中的应用，即在某些情况下需要解决的不确定性，以确保在IAF的每个完成中，判断给定一组论证是否为扩展的结果一致。进一步提出了强相关性的概念，以描述在所有情况下解决必要性以达到稳定性。复杂性分析表明，在论文讨论的大多数语义下，检测一组论证稳定性的（强）相关性可以在多项式时间内完成。我们还讨论了在接地语义下找到可解决性检测方法的难度。 

---
# Minimizing the energy depletion in wireless rechargeable sensor networks using bi-level metaheuristic charging schemes 

**Title (ZH)**: 使用双层元启发式充电方案最小化无线可充电传感器网络的能量耗尽 

**Authors**: Huynh Thi Thanh Binh, Le Van Cuong, Dang Hai Dang, Le Trong Vinh  

**Link**: [PDF](https://arxiv.org/pdf/2505.16482)  

**Abstract**: Recently, Wireless Rechargeable Sensor Networks (WRSNs) that leveraged the advantage of wireless energy transfer technology have opened a promising opportunity in solving the limited energy issue. However, an ineffective charging strategy may reduce the charging performance. Although many practical charging algorithms have been introduced, these studies mainly focus on optimizing the charging path with a fully charging approach. This approach may lead to the death of a series of sensors due to their extended charging latency. This paper introduces a novel partial charging approach that follows a bi-level optimized scheme to minimize energy depletion in WRSNs. We aim at optimizing simultaneously two factors: the charging path and time. To accomplish this, we first formulate a mathematical model of the investigated problem. We then propose two approximate algorithms in which the optimization of the charging path and the charging time are considered as the upper and lower level, respectively. The first algorithm combines a Multi-start Local Search method and a Genetic Algorithm to find a solution. The second algorithm adopts a nested approach that utilizes the advantages of the Multitasking and Covariance Matrix Adaptation Evolutionary Strategies. Experimental validations on various network scenarios demonstrate that our proposed algorithms outperform the existing works. 

**Abstract (ZH)**: recently, 利用水无线能量传输技术的无线可充电传感器网络 (WRSNs) 已为解决能量限制问题开辟了前景。然而，无效的充电策略可能降低充电性能。尽管已经引入了许多实际的充电算法，这些研究主要集中在使用全充满策略优化充电路径上。这种策略可能导致由于延长的充电延迟而导致一系列传感器的失效。本文介绍了一种新颖的部分充电方法，该方法遵循双层优化方案以最小化 WRSNs 中的能量耗尽。我们旨在同时优化两个因素：充电路径和时间。为此，我们首先制定了所研究问题的数学模型。然后，我们提出两种近似算法，其中充电路径的优化被视为高层，充电时间的优化被视为低层。第一个算法结合了多启动局部搜索方法和遗传算法来寻找解决方案。第二个算法采用嵌套方法，利用多任务和共变异矩阵适应进化策略的优势。在各种网络场景下的实验验证表明，我们提出的算法优于现有工作。 

---
# Internal Bias in Reasoning Models leads to Overthinking 

**Title (ZH)**: 内在偏见在推理模型中导致过度思考 

**Authors**: Renfei Dang, Shujian Huang, Jiajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16448)  

**Abstract**: While current reasoning models possess strong exploratory capabilities, they are often criticized for overthinking due to redundant and unnecessary reflections. In this work, we reveal for the first time that overthinking in reasoning models may stem from their internal bias towards input texts. Upon encountering a reasoning problem, the model immediately forms a preliminary guess about the answer, which we term as an internal bias since it is not derived through actual reasoning. When this guess conflicts with its reasoning result, the model tends to engage in reflection, leading to the waste of computational resources. Through further interpretability experiments, we find that this behavior is largely driven by the model's excessive attention to the input section, which amplifies the influence of internal bias on its decision-making process. Additionally, by masking out the original input section, the affect of internal bias can be effectively alleviated and the reasoning length could be reduced by 31%-53% across different complex reasoning tasks. Notably, in most cases, this approach also leads to improvements in accuracy. These findings demonstrate a causal relationship between internal bias and overthinking. 

**Abstract (ZH)**: 当前推理模型虽然具备较强的探索能力，但常常因为冗余和不必要的反思而受到过度思考的批评。在本文中，我们首次揭示，推理模型的过度思考可能源于其对输入文本的内在偏向。面对推理问题时，模型会立即形成关于答案的初步猜测，这被称为内在偏见，因为它并非通过实际推理得出。当这一猜测与推理结果发生冲突时，模型会倾向于进行反思，从而浪费计算资源。通过进一步的解释性实验，我们发现，这种行为很大程度上是由模型对输入部分的过度关注驱动的，这放大了内在偏见对其决策过程的影响。此外，通过屏蔽原始输入部分，可以有效地减轻内在偏见的影响，并在不同复杂的推理任务中将推理长度减少31%-53%。值得注意的是，在大多数情况下，这种方法还能够提高准确性。这些发现表明内在偏见与过度思考之间存在因果关系。 

---
# FREESON: Retriever-Free Retrieval-Augmented Reasoning via Corpus-Traversing MCTS 

**Title (ZH)**: FREESON: 不依赖检索的基于语料遍历的MCTS增强推理 

**Authors**: Chaeeun Kim, Seungone Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.16409)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable capabilities in multi-step reasoning and calling search engines at appropriate steps. However, existing retrieval-augmented reasoning approaches rely on separate retrieval models, limiting the LRM's role in retrieval to deciding when to retrieve and how to query. This separation not only increases hardware and operational costs but also leads to errors in the retrieval process due to the representation bottleneck, a phenomenon where the retriever's embedding space is not expressive enough to meet the generator's requirements. To address this, we shift our perspective from sequence-to-sequence matching to locating the answer-containing paths within the corpus, and propose a novel framework called FREESON (Retriever-FREE Retrieval-Augmented ReaSONing). This framework enables LRMs to retrieve relevant knowledge on their own by acting as both a generator and retriever. To achieve this, we introduce a variant of the MCTS algorithm specialized for the retrieval task, which we call CT-MCTS (Corpus-Traversing Monte Carlo Tree Search). In this algorithm, LRMs traverse through the corpus toward answer-containing regions. Our results on five open-domain QA benchmarks, including single-hop and multi-hop questions, show that FREESON achieves an average improvement of 14.4% in EM and F1 over four multi-step reasoning models with a separate retriever, and it also performs comparably to the strongest baseline, surpassing it by 3% on PopQA and 2WikiMultihopQA. 

**Abstract (ZH)**: Large Reasoning Models (LRMs)在多步推理和适时调用搜索引擎方面展现了显著的能力。然而，现有的检索增强推理方法依赖于独立的检索模型，限制了LRM在检索方面的角色，仅限于决定何时检索及如何查询。这种分离不仅增加了硬件和运营成本，还由于检索瓶颈（即检索器的嵌入空间不足以满足生成器的要求）导致了检索过程中的错误。为了应对这一问题，我们将视角从序列到序列匹配转向在语料库中定位包含答案的路径，并提出了一种名为FREESON（Retriever-FREE Retrieval-Augmented ReaSONing）的新框架。该框架使LRMs能够通过自身充当生成器和检索器的角色来自主检索相关知识。为此，我们引入了一种专为检索任务设计的MCTS算法变体，称之为CT-MCTS（Corpus-Traversing Monte Carlo Tree Search）。在该算法中，LRMs在语料库中朝向包含答案的区域进行遍历。在五个开放领域问答基准测试上的结果，包括单跳和多跳问题，表明FREESON在EM和F1指标上分别比四种带有独立检索器的多步推理模型平均提高了14.4%，并且在PopQA和2WikiMultihopQA基准上分别超越最强基线3%和2%。 

---
# Serious Games: Human-AI Interaction, Evolution, and Coevolution 

**Title (ZH)**: 严肃游戏：人机交互、进化与共生进化 

**Authors**: Nandini Doreswamy, Louise Horstmanshof  

**Link**: [PDF](https://arxiv.org/pdf/2505.16388)  

**Abstract**: The serious games between humans and AI have only just begun. Evolutionary Game Theory (EGT) models the competitive and cooperative strategies of biological entities. EGT could help predict the potential evolutionary equilibrium of humans and AI. The objective of this work was to examine some of the EGT models relevant to human-AI interaction, evolution, and coevolution. Of thirteen EGT models considered, three were examined: the Hawk-Dove Game, Iterated Prisoner's Dilemma, and the War of Attrition. This selection was based on the widespread acceptance and clear relevance of these models to potential human-AI evolutionary dynamics and coevolutionary trajectories. The Hawk-Dove Game predicts balanced mixed-strategy equilibria based on the costs of conflict. It also shows the potential for balanced coevolution rather than dominance. Iterated Prisoner's Dilemma suggests that repeated interaction may lead to cognitive coevolution. It demonstrates how memory and reciprocity can lead to cooperation. The War of Attrition suggests that competition for resources may result in strategic coevolution, asymmetric equilibria, and conventions on sharing resources. Therefore, EGT may provide a suitable framework to understand and predict the human-AI evolutionary dynamic. However, future research could extend beyond EGT and explore additional frameworks, empirical validation methods, and interdisciplinary perspectives. AI is being shaped by human input and is evolving in response to it. So too, neuroplasticity allows the human brain to grow and evolve in response to stimuli. If humans and AI converge in future, what might be the result of human neuroplasticity combined with an ever-evolving AI? Future research should be mindful of the ethical and cognitive implications of human-AI interaction, evolution, and coevolution. 

**Abstract (ZH)**: 人类与AI之间的严肃游戏刚刚开始：进化博弈理论在人类-AI交互、进化及其共进化中的应用及其前景 

---
# MADCluster: Model-agnostic Anomaly Detection with Self-supervised Clustering Network 

**Title (ZH)**: MADCluster：模型无关的异常检测自监督聚类网络 

**Authors**: Sangyong Lee, Subo Hwang, Dohoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.16223)  

**Abstract**: In this paper, we propose MADCluster, a novel model-agnostic anomaly detection framework utilizing self-supervised clustering. MADCluster is applicable to various deep learning architectures and addresses the 'hypersphere collapse' problem inherent in existing deep learning-based anomaly detection methods. The core idea is to cluster normal pattern data into a 'single cluster' while simultaneously learning the cluster center and mapping data close to this center. Also, to improve expressiveness and enable effective single clustering, we propose a new 'One-directed Adaptive loss'. The optimization of this loss is mathematically proven. MADCluster consists of three main components: Base Embedder capturing high-dimensional temporal dynamics, Cluster Distance Mapping, and Sequence-wise Clustering for continuous center updates. Its model-agnostic characteristics are achieved by applying various architectures to the Base Embedder. Experiments on four time series benchmark datasets demonstrate that applying MADCluster improves the overall performance of comparative models. In conclusion, the compatibility of MADCluster shows potential for enhancing model performance across various architectures. 

**Abstract (ZH)**: MADCluster：一种利用自监督聚类的新型模型无关异常检测框架 

---
# Dynamic Sampling that Adapts: Iterative DPO for Self-Aware Mathematical Reasoning 

**Title (ZH)**: 自适应动态采样：迭代DPO在自我意识数学推理中的应用 

**Authors**: Jun Rao, Xuebo Liu, Hexuan Deng, Zepeng Lin, Zixiong Yu, Jiansheng Wei, Xiaojun Meng, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16176)  

**Abstract**: In the realm of data selection for reasoning tasks, existing approaches predominantly rely on externally predefined static metrics such as difficulty and diversity, which are often designed for supervised fine-tuning (SFT) and lack adaptability to continuous training processes. A critical limitation of these methods is their inability to dynamically align with the evolving capabilities of models during online training, a gap that becomes increasingly pronounced with the rise of dynamic training paradigms and online reinforcement learning (RL) frameworks (e.g., R1 models). To address this, we introduce SAI-DPO, an algorithm that dynamically selects training data by continuously assessing a model's stage-specific reasoning abilities across different training phases. By integrating real-time model performance feedback, SAI-DPO adaptively adapts data selection to the evolving strengths and weaknesses of the model, thus enhancing both data utilization efficiency and final task performance. Extensive experiments on three state-of-the-art models and eight mathematical reasoning benchmarks, including challenging competition-level datasets (e.g., AIME24 and AMC23), demonstrate that SAI-DPO achieves an average performance boost of up to 21.3 percentage points, with particularly notable improvements of 10 and 15 points on AIME24 and AMC23, respectively. These results highlight the superiority of dynamic, model-adaptive data selection over static, externally defined strategies in advancing reasoning. 

**Abstract (ZH)**: 基于推理任务的数据选择：动态适应性数据选择算法(SAI-DPO) 

---
# Losing is for Cherishing: Data Valuation Based on Machine Unlearning and Shapley Value 

**Title (ZH)**: 失去是为了珍惜：基于机器遗忘和夏皮利值的数据估值 

**Authors**: Le Ma, Shirao Yang, Zihao Wang, Yinggui Wang, Lei Wang, Tao Wei, Kejun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16147)  

**Abstract**: The proliferation of large models has intensified the need for efficient data valuation methods to quantify the contribution of individual data providers. Traditional approaches, such as game-theory-based Shapley value and influence-function-based techniques, face prohibitive computational costs or require access to full data and model training details, making them hardly achieve partial data valuation. To address this, we propose Unlearning Shapley, a novel framework that leverages machine unlearning to estimate data values efficiently. By unlearning target data from a pretrained model and measuring performance shifts on a reachable test set, our method computes Shapley values via Monte Carlo sampling, avoiding retraining and eliminating dependence on full data. Crucially, Unlearning Shapley supports both full and partial data valuation, making it scalable for large models (e.g., LLMs) and practical for data markets. Experiments on benchmark datasets and large-scale text corpora demonstrate that our approach matches the accuracy of state-of-the-art methods while reducing computational overhead by orders of magnitude. Further analysis confirms a strong correlation between estimated values and the true impact of data subsets, validating its reliability in real-world scenarios. This work bridges the gap between data valuation theory and practical deployment, offering a scalable, privacy-compliant solution for modern AI ecosystems. 

**Abstract (ZH)**: 大型模型的普及加剧了对高效数据估值方法的需求，以量化个体数据提供者的贡献。传统的基于博弈论的Shapley值方法和基于影响函数的技术面临难以承受的计算成本，或需要访问完整数据和模型训练细节，使它们难以实现部分数据估值。为解决这一问题，我们提出了一种新的框架——Unlearning Shapley，该框架利用机器遗忘高效估计数据价值。通过从预训练模型中遗忘目标数据，并在可达测试集上测量性能变化，我们的方法利用蒙特卡洛采样计算Shapley值，避免重新训练并消除对完整数据的依赖。重要的是，Unlearning Shapley 支持完整和部分数据估值，使其在大规模模型（如语言模型）中具有可扩展性，并在数据市场中具有实用性。在基准数据集和大规模文本语料库上的实验表明，我们的方法在计算开销上比最先进的方法减少了几个数量级，同时保持了准确性。进一步的分析证实了估计值与数据子集真实影响之间的强相关性，验证了其在实际场景中的可靠性。这项工作填补了数据估值理论与实际部署之间的差距，提供了针对现代人工智能生态系统的可扩展且隐私合规的解决方案。 

---
# Sudoku-Bench: Evaluating creative reasoning with Sudoku variants 

**Title (ZH)**: Sudoku-Bench: 评估变体数独中的创造性推理能力 

**Authors**: Jeffrey Seely, Yuki Imajuku, Tianyu Zhao, Edoardo Cetin, Llion Jones  

**Link**: [PDF](https://arxiv.org/pdf/2505.16135)  

**Abstract**: Existing reasoning benchmarks for large language models (LLMs) frequently fail to capture authentic creativity, often rewarding memorization of previously observed patterns. We address this shortcoming with Sudoku-Bench, a curated benchmark of challenging and unconventional Sudoku variants specifically selected to evaluate creative, multi-step logical reasoning. Sudoku variants form an unusually effective domain for reasoning research: each puzzle introduces unique or subtly interacting constraints, making memorization infeasible and requiring solvers to identify novel logical breakthroughs (``break-ins''). Despite their diversity, Sudoku variants maintain a common and compact structure, enabling clear and consistent evaluation. Sudoku-Bench includes a carefully chosen puzzle set, a standardized text-based puzzle representation, and flexible tools compatible with thousands of publicly available puzzles -- making it easy to extend into a general research environment. Baseline experiments show that state-of-the-art LLMs solve fewer than 15\% of puzzles unaided, highlighting significant opportunities to advance long-horizon, strategic reasoning capabilities. 

**Abstract (ZH)**: 现有的大型语言模型推理基准经常无法捕捉到真实的创造力， often rewarding memorization of previously observed patterns. 我们通过Sudoku-Bench解决了这一不足，这是一个精选的基准，包含了具有挑战性和非传统性的数独变体，特别选择用于评估创造性、多步逻辑推理能力。尽管数独变体多样，但它们维持着一种共同且紧凑的结构，使得推理研究具有清晰和一致的评估标准。Sudoku-Bench包含仔细选择的谜题集、标准化的文字表示形式的谜题以及与数千个公开可用的谜题兼容的灵活工具，使其易于扩展为通用研究环境。基准实验显示，最先进的大型语言模型在无辅助的情况下仅能解决不到15%的谜题，强调了在长时规划和战略推理能力方面有巨大的研究潜力。 

---
# BioDSA-1K: Benchmarking Data Science Agents for Biomedical Research 

**Title (ZH)**: BioDSA-1K: 评估生物医学研究中的数据科学代理 

**Authors**: Zifeng Wang, Benjamin Danek, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16100)  

**Abstract**: Validating scientific hypotheses is a central challenge in biomedical research, and remains difficult for artificial intelligence (AI) agents due to the complexity of real-world data analysis and evidence interpretation. In this work, we present BioDSA-1K, a benchmark designed to evaluate AI agents on realistic, data-driven biomedical hypothesis validation tasks. BioDSA-1K consists of 1,029 hypothesis-centric tasks paired with 1,177 analysis plans, curated from over 300 published biomedical studies to reflect the structure and reasoning found in authentic research workflows. Each task includes a structured hypothesis derived from the original study's conclusions, expressed in the affirmative to reflect the language of scientific reporting, and one or more pieces of supporting evidence grounded in empirical data tables. While these hypotheses mirror published claims, they remain testable using standard statistical or machine learning methods. The benchmark enables evaluation along four axes: (1) hypothesis decision accuracy, (2) alignment between evidence and conclusion, (3) correctness of the reasoning process, and (4) executability of the AI-generated analysis code. Importantly, BioDSA-1K includes non-verifiable hypotheses: cases where the available data are insufficient to support or refute a claim, reflecting a common yet underexplored scenario in real-world science. We propose BioDSA-1K as a foundation for building and evaluating generalizable, trustworthy AI agents for biomedical discovery. 

**Abstract (ZH)**: BioDSA-1K：一种用于评估生物医学假设验证任务的人工智能代理基准 

---
# TrialPanorama: Database and Benchmark for Systematic Review and Design of Clinical Trials 

**Title (ZH)**: TrialPanorama: 临床试验系统评价与设计的数据库及基准测试 

**Authors**: Zifeng Wang, Qiao Jin, Jiacheng Lin, Junyi Gao, Jathurshan Pradeepkumar, Pengcheng Jiang, Benjamin Danek, Zhiyong Lu, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16097)  

**Abstract**: Developing artificial intelligence (AI) for vertical domains requires a solid data foundation for both training and evaluation. In this work, we introduce TrialPanorama, a large-scale, structured database comprising 1,657,476 clinical trial records aggregated from 15 global sources. The database captures key aspects of trial design and execution, including trial setups, interventions, conditions, biomarkers, and outcomes, and links them to standard biomedical ontologies such as DrugBank and MedDRA. This structured and ontology-grounded design enables TrialPanorama to serve as a unified, extensible resource for a wide range of clinical trial tasks, including trial planning, design, and summarization. To demonstrate its utility, we derive a suite of benchmark tasks directly from the TrialPanorama database. The benchmark spans eight tasks across two categories: three for systematic review (study search, study screening, and evidence summarization) and five for trial design (arm design, eligibility criteria, endpoint selection, sample size estimation, and trial completion assessment). The experiments using five state-of-the-art large language models (LLMs) show that while general-purpose LLMs exhibit some zero-shot capability, their performance is still inadequate for high-stakes clinical trial workflows. We release TrialPanorama database and the benchmark to facilitate further research on AI for clinical trials. 

**Abstract (ZH)**: 垂直领域开发人工智能（AI）需要坚实的数据基础用于训练和评估。本文介绍了TrialPanorama，一个包含1,657,476个临床试验记录的大规模结构化数据库，这些记录来自15个全球来源。该数据库捕捉到临床试验设计和执行的关键方面，包括试验设置、干预措施、条件、生物标志物和结果，并将其链接到标准生物医学本体，如DrugBank和MedDRA。该结构化和基于本体的设计使TrialPanorama能够作为多种临床试验任务的一体化、可扩展资源，包括试验规划、设计和总结。为了展示其 usefulness，我们从TrialPanorama数据库直接推导出一系列基准任务。这些基准任务覆盖八个任务，分为两类：三类是系统评价任务（研究搜索、研究筛选和证据总结），五类是试验设计任务（组设计、入组标准、终点选择、样本量估算和试验完成评估）。使用五种最先进的大规模语言模型（LLMs）的实验表明，虽然通用的大规模语言模型具有一定的零样本能力，但它们的表现仍然不足以应对高风险的临床试验工作流程。我们发布了TrialPanorama数据库和基准任务，以促进临床试验中人工智能研究的进一步发展。 

---
# SynEVO: A neuro-inspired spatiotemporal evolutional framework for cross-domain adaptation 

**Title (ZH)**: SynEVO: 一种神经启发的空间时间演化跨域适应框架 

**Authors**: Jiayue Liu, Zhongchao Yi, Zhengyang Zhou, Qihe Huang, Kuo Yang, Xu Wang, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16080)  

**Abstract**: Discovering regularities from spatiotemporal systems can benefit various scientific and social planning. Current spatiotemporal learners usually train an independent model from a specific source data that leads to limited transferability among sources, where even correlated tasks requires new design and training. The key towards increasing cross-domain knowledge is to enable collective intelligence and model evolution. In this paper, inspired by neuroscience theories, we theoretically derive the increased information boundary via learning cross-domain collective intelligence and propose a Synaptic EVOlutional spatiotemporal network, SynEVO, where SynEVO breaks the model independence and enables cross-domain knowledge to be shared and aggregated. Specifically, we first re-order the sample groups to imitate the human curriculum learning, and devise two complementary learners, elastic common container and task-independent extractor to allow model growth and task-wise commonality and personality disentanglement. Then an adaptive dynamic coupler with a new difference metric determines whether the new sample group should be incorporated into common container to achieve model evolution under various domains. Experiments show that SynEVO improves the generalization capacity by at most 42% under cross-domain scenarios and SynEVO provides a paradigm of NeuroAI for knowledge transfer and adaptation. 

**Abstract (ZH)**: 探究时空系统中的规律可以惠及各个科学和社会规划领域。当前的时空学习器通常是从特定源数据中训练独立模型，这导致了来源间有限的迁移性，即使相关任务也需要新的设计和训练。通过增强跨域知识，关键在于促进集体智能和模型进化。受神经科学理论启发，本文从理论上推导出通过学习跨域集体智能来增加信息边界，并提出了一种Synaptic Evolutionary时空网络（SynEVO），SynEVO打破了模型独立性，使跨域知识能够共享和聚合。具体来说，我们首先重新排列样本组以模仿人类的课程学习，并设计了两个互补的 learner：弹性通用容器和任务独立提取器，以允许模型增长和任务内共同性与个性的解耦。然后，一个自适应动态耦合器结合新的差异度量确定新的样本组是否应被整合到通用容器中，以实现不同领域下的模型进化。实验表明，在跨域场景下，SynEVO最多可提高42%的泛化能力，并为知识转移和适应提供了神经人工智能范式。 

---
# Children's Mental Models of AI Reasoning: Implications for AI Literacy Education 

**Title (ZH)**: 儿童对AI推理的认知模型：对AI literacy教育的启示 

**Authors**: Aayushi Dangol, Robert Wolfe, Runhua Zhao, JaeWon Kim, Trushaa Ramanan, Katie Davis, Julie A. Kientz  

**Link**: [PDF](https://arxiv.org/pdf/2505.16031)  

**Abstract**: As artificial intelligence (AI) advances in reasoning capabilities, most recently with the emergence of Large Reasoning Models (LRMs), understanding how children conceptualize AI's reasoning processes becomes critical for fostering AI literacy. While one of the "Five Big Ideas" in AI education highlights reasoning algorithms as central to AI decision-making, less is known about children's mental models in this area. Through a two-phase approach, consisting of a co-design session with 8 children followed by a field study with 106 children (grades 3-8), we identified three models of AI reasoning: Deductive, Inductive, and Inherent. Our findings reveal that younger children (grades 3-5) often attribute AI's reasoning to inherent intelligence, while older children (grades 6-8) recognize AI as a pattern recognizer. We highlight three tensions that surfaced in children's understanding of AI reasoning and conclude with implications for scaffolding AI curricula and designing explainable AI tools. 

**Abstract (ZH)**: 随着人工智能（AI）在推理能力上的进步，尤其是在大型推理模型（LRMs）的出现之后，理解儿童如何概念化AI的推理过程对于培养AI素养变得至关重要。尽管AI教育中的“五大核心理念”之一强调推理算法是AI决策的核心，但关于这一领域的儿童心理模型知之甚少。通过两阶段的方法，包括与8名儿童合作设计会话，随后对106名儿童（3-8年级）进行实地研究，我们确定了三种AI推理模型：演绎、归纳和固有。研究发现显示，较低年级的儿童（3-5年级）常将AI的推理归因于固有的智能，而较高年级的儿童（6-8年级）则认识到AI是一个模式识别器。我们强调了在儿童对AI推理理解中浮现的三个紧张关系，并提出构建AI课程和支持可解释AI工具的建议。 

---
# Exploring Flow-Lenia Universes with a Curiosity-driven AI Scientist: Discovering Diverse Ecosystem Dynamics 

**Title (ZH)**: 基于好奇心驱动的AI科学家探索Flow-Lenia宇宙：发现多样的生态系统动态 

**Authors**: Thomas Michel, Marko Cvjetko, Gautier Hamon, Pierre-Yves Oudeyer, Clément Moulin-Frier  

**Link**: [PDF](https://arxiv.org/pdf/2505.15998)  

**Abstract**: We present a method for the automated discovery of system-level dynamics in Flow-Lenia$-$a continuous cellular automaton (CA) with mass conservation and parameter localization$-$using a curiosity-driven AI scientist. This method aims to uncover processes leading to self-organization of evolutionary and ecosystemic dynamics in CAs. We build on previous work which uses diversity search algorithms in Lenia to find self-organized individual patterns, and extend it to large environments that support distinct interacting patterns. We adapt Intrinsically Motivated Goal Exploration Processes (IMGEPs) to drive exploration of diverse Flow-Lenia environments using simulation-wide metrics, such as evolutionary activity, compression-based complexity, and multi-scale entropy. We test our method in two experiments, showcasing its ability to illuminate significantly more diverse dynamics compared to random search. We show qualitative results illustrating how ecosystemic simulations enable self-organization of complex collective behaviors not captured by previous individual pattern search and analysis. We complement automated discovery with an interactive exploration tool, creating an effective human-AI collaborative workflow for scientific investigation. Though demonstrated specifically with Flow-Lenia, this methodology provides a framework potentially applicable to other parameterizable complex systems where understanding emergent collective properties is of interest. 

**Abstract (ZH)**: 我们提出了一种使用好奇心驱动的AI科学家在Flow-Lenia中自动发现系统级动力学的方法——一种具有质量守恒和参数局部化的连续细胞自动机。该方法旨在揭示在细胞自动机中导致演化和生态系统动力学自我组织的过程。我们在此前使用Lenia中的多样性搜索算法寻找自我组织的个体模式的基础上，将其扩展到支持不同相互作用模式的大型环境。我们采用固有动机目标探索过程（IMGEPs）利用全局仿真指标（如演化活性、基于压缩的复杂性、多尺度熵）驱动Flow-Lenia环境的探索。我们在两个实验中测试了该方法，展示了其相较于随机搜索能够揭示更多样化的动力学的能力。我们提供了定性结果，说明生态系统的模拟如何使复杂的集体行为自我组织，这些行为在以往的个体模式搜索和分析中并未被捕捉到。我们结合自动发现提供了一种交互式探索工具，创建了一个高效的人工智能协作工作流以进行科学研究。尽管在Flow-Lenia中具体演示了这种方法，但该方法论框架有可能适用于其他参数可调的复杂系统，其中理解涌现的集体性质是有趣的。 

---
# PhyX: Does Your Model Have the "Wits" for Physical Reasoning? 

**Title (ZH)**: PhyX: 你的模型具备“智慧”进行物理推理了吗？ 

**Authors**: Hui Shen, Taiqiang Wu, Qi Han, Yunta Hsieh, Jizhou Wang, Yuyue Zhang, Yuxin Cheng, Zijian Hao, Yuansheng Ni, Xin Wang, Zhongwei Wan, Kai Zhang, Wendong Xu, Jing Xiong, Ping Luo, Wenhu Chen, Chaofan Tao, Zhuoqing Mao, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2505.15929)  

**Abstract**: Existing benchmarks fail to capture a crucial aspect of intelligence: physical reasoning, the integrated ability to combine domain knowledge, symbolic reasoning, and understanding of real-world constraints. To address this gap, we introduce PhyX: the first large-scale benchmark designed to assess models capacity for physics-grounded reasoning in visual scenarios. PhyX includes 3K meticulously curated multimodal questions spanning 6 reasoning types across 25 sub-domains and 6 core physics domains: thermodynamics, electromagnetism, mechanics, modern physics, optics, and wave\&acoustics. In our comprehensive evaluation, even state-of-the-art models struggle significantly with physical reasoning. GPT-4o, Claude3.7-Sonnet, and GPT-o4-mini achieve only 32.5\%, 42.2\%, and 45.8\% accuracy respectively-performance gaps exceeding 29\% compared to human experts. Our analysis exposes critical limitations in current models: over-reliance on memorized disciplinary knowledge, excessive dependence on mathematical formulations, and surface-level visual pattern matching rather than genuine physical understanding. We provide in-depth analysis through fine-grained statistics, detailed case studies, and multiple evaluation paradigms to thoroughly examine physical reasoning capabilities. To ensure reproducibility, we implement a compatible evaluation protocol based on widely-used toolkits such as VLMEvalKit, enabling one-click evaluation. 

**Abstract (ZH)**: 现有的基准未能捕捉到智能的一个关键方面：物理推理，即结合领域知识、符号推理和对现实世界约束理解的综合能力。为填补这一空白，我们引入了PhyX：首个大型基准，旨在评估模型在视觉场景中进行物理基础推理的能力。PhyX 包含3000个精心挑选的多模态问题，覆盖6种推理类型，涉及25个子领域和6个核心物理领域：热力学、电磁学、力学、现代物理、光学和波与声学。在我们全面的评估中，即使是最先进的模型也显著挣扎于物理推理。GPT-4o、Claude3.7-Sonnet和GPT-o4-mini分别达到了32.5%、42.2%和45.8%的准确率，与人类专家的差距超过29%。我们的分析揭示了当前模型的关键局限性：过度依赖记忆学科知识、过度依赖数学公式以及表层视觉模式匹配而非真正的物理理解。我们通过精细的统计分析、详细的案例研究和多种评估范式进行了深入分析，以彻底检验物理推理能力。为了确保可重现性，我们基于广泛使用的工具包（如VLMEvalKit）实现了兼容的评估协议，实现一键评估。 

---
# Bandit based Dynamic Candidate Edge Selection in Solving Traveling Salesman Problems 

**Title (ZH)**: 基于拉臂算法的动态候选边选择在解决旅行商问题中的应用 

**Authors**: Long Wanga, Jiongzhi Zheng, Zhengda Xiong, ChuMin Li, Kun He  

**Link**: [PDF](https://arxiv.org/pdf/2505.15862)  

**Abstract**: Algorithms designed for routing problems typically rely on high-quality candidate edges to guide their search, aiming to reduce the search space and enhance the search efficiency. However, many existing algorithms, like the classical Lin-Kernighan-Helsgaun (LKH) algorithm for the Traveling Salesman Problem (TSP), often use predetermined candidate edges that remain static throughout local searches. This rigidity could cause the algorithm to get trapped in local optima, limiting its potential to find better solutions. To address this issue, we propose expanding the candidate sets to include other promising edges, providing them an opportunity for selection. Specifically, we incorporate multi-armed bandit models to dynamically select the most suitable candidate edges in each iteration, enabling LKH to make smarter choices and lead to improved solutions. Extensive experiments on multiple TSP benchmarks show the excellent performance of our method. Moreover, we employ this bandit-based method to LKH-3, an extension of LKH tailored for solving various TSP variant problems, and our method also significantly enhances LKH-3's performance across typical TSP variants. 

**Abstract (ZH)**: 设计用于路由问题的算法通常依赖高质量的候选边来引导搜索，旨在减少搜索空间并提高搜索效率。然而，许多现有的算法，如经典的Lin-Kernighan-Helsgaucu (LKH)算法用于旅行商问题(TSP)，往往使用固定的候选边，在局部搜索过程中保持不变。这种刚性可能导致算法陷入局部最优，限制其找到更好解的潜力。为解决这一问题，我们提出扩大候选集，包括其他有潜力的边，给它们选择的机会。具体而言，我们引入多臂老虎机模型，在每一轮迭代中动态选择最合适的候选边，使LKH能够做出更明智的选择并产生更好的解。在多种TSP基准测试上的广泛实验显示了我们方法的优秀性能。此外，我们将基于多臂老虎机的方法应用于LKH-3，这是专门为解决各种TSP变体问题而扩展的LKH版本，我们的方法也在典型的TSP变体中显著提升了LKH-3的性能。 

---
# Delving into RL for Image Generation with CoT: A Study on DPO vs. GRPO 

**Title (ZH)**: 基于CoT的图像生成中的RL研究：DPO与GRPO的比较 

**Authors**: Chengzhuo Tong, Ziyu Guo, Renrui Zhang, Wenyu Shan, Xinyu Wei, Zhenghao Xing, Hongsheng Li, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17017)  

**Abstract**: Recent advancements underscore the significant role of Reinforcement Learning (RL) in enhancing the Chain-of-Thought (CoT) reasoning capabilities of large language models (LLMs). Two prominent RL algorithms, Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO), are central to these developments, showcasing different pros and cons. Autoregressive image generation, also interpretable as a sequential CoT reasoning process, presents unique challenges distinct from LLM-based CoT reasoning. These encompass ensuring text-image consistency, improving image aesthetic quality, and designing sophisticated reward models, rather than relying on simpler rule-based rewards. While recent efforts have extended RL to this domain, these explorations typically lack an in-depth analysis of the domain-specific challenges and the characteristics of different RL strategies. To bridge this gap, we provide the first comprehensive investigation of the GRPO and DPO algorithms in autoregressive image generation, evaluating their in-domain performance and out-of-domain generalization, while scrutinizing the impact of different reward models on their respective capabilities. Our findings reveal that GRPO and DPO exhibit distinct advantages, and crucially, that reward models possessing stronger intrinsic generalization capabilities potentially enhance the generalization potential of the applied RL algorithms. Furthermore, we systematically explore three prevalent scaling strategies to enhance both their in-domain and out-of-domain proficiency, deriving unique insights into efficiently scaling performance for each paradigm. We hope our study paves a new path for inspiring future work on developing more effective RL algorithms to achieve robust CoT reasoning in the realm of autoregressive image generation. Code is released at this https URL 

**Abstract (ZH)**: 最近的研究强调了强化学习（RL）在提高大型语言模型（LLMs）链式思维（CoT）推理能力方面的重要作用。Direct Preference Optimization（DPO）和Group Relative Policy Optimization（GRPO）这两种主要的RL算法在这些发展中起到了关键作用，展示了各自的优缺点。自回归图像生成可解释为一种序列式的CoT推理过程，不同于基于LLM的CoT推理，它面临着独特的挑战，包括确保文本与图像的一致性、提高图像的审美质量以及设计复杂的奖励模型，而不是依赖于简单的基于规则的奖励模型。尽管最近的努力已经将RL扩展到这一领域，但这些探索通常缺乏对特定领域挑战和不同RL策略特性的深入分析。为填补这一空白，我们提供了GRPO和DPO算法在自回归图像生成领域中的首次全面研究，评估它们在领域内的性能以及跨领域的泛化能力，同时审查不同奖励模型对其各自能力的影响。研究发现表明，GRPO和DPO各具优势，并且关键在于具有更强内在泛化能力的奖励模型可能增强所应用的RL算法的泛化潜力。此外，我们系统地探索了三种常见的扩展策略，以增强它们在领域内和跨领域的能力，为每种范式高效扩展性能提供了独特的见解。我们希望我们的研究为未来开发更有效的RL算法，以实现自回归图像生成领域内稳健的CoT推理开辟新的路径。代码发布于此 URL。 

---
# Understanding Prompt Tuning and In-Context Learning via Meta-Learning 

**Title (ZH)**: 通过元学习理解提示调优和上下文学习 

**Authors**: Tim Genewein, Kevin Wenliang Li, Jordi Grau-Moya, Anian Ruoss, Laurent Orseau, Marcus Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2505.17010)  

**Abstract**: Prompting is one of the main ways to adapt a pretrained model to target tasks. Besides manually constructing prompts, many prompt optimization methods have been proposed in the literature. Method development is mainly empirically driven, with less emphasis on a conceptual understanding of prompting. In this paper we discuss how optimal prompting can be understood through a Bayesian view, which also implies some fundamental limitations of prompting that can only be overcome by tuning weights. The paper explains in detail how meta-trained neural networks behave as Bayesian predictors over the pretraining distribution, whose hallmark feature is rapid in-context adaptation. Optimal prompting can be studied formally as conditioning these Bayesian predictors, yielding criteria for target tasks where optimal prompting is and is not possible. We support the theory with educational experiments on LSTMs and Transformers, where we compare different versions of prefix-tuning and different weight-tuning methods. We also confirm that soft prefixes, which are sequences of real-valued vectors outside the token alphabet, can lead to very effective prompts for trained and even untrained networks by manipulating activations in ways that are not achievable by hard tokens. This adds an important mechanistic aspect beyond the conceptual Bayesian theory. 

**Abstract (ZH)**: 通过贝叶斯视角理解最优提示：提示的局限性及权重调整的必要性 

---
# Guided Diffusion Sampling on Function Spaces with Applications to PDEs 

**Title (ZH)**: 函数空间中的指导扩散采样及其在偏微分方程中的应用 

**Authors**: Jiachen Yao, Abbas Mammadov, Julius Berner, Gavin Kerrigan, Jong Chul Ye, Kamyar Azizzadenesheli, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.17004)  

**Abstract**: We propose a general framework for conditional sampling in PDE-based inverse problems, targeting the recovery of whole solutions from extremely sparse or noisy measurements. This is accomplished by a function-space diffusion model and plug-and-play guidance for conditioning. Our method first trains an unconditional discretization-agnostic denoising model using neural operator architectures. At inference, we refine the samples to satisfy sparse observation data via a gradient-based guidance mechanism. Through rigorous mathematical analysis, we extend Tweedie's formula to infinite-dimensional Hilbert spaces, providing the theoretical foundation for our posterior sampling approach. Our method (FunDPS) accurately captures posterior distributions in function spaces under minimal supervision and severe data scarcity. Across five PDE tasks with only 3% observation, our method achieves an average 32% accuracy improvement over state-of-the-art fixed-resolution diffusion baselines while reducing sampling steps by 4x. Furthermore, multi-resolution fine-tuning ensures strong cross-resolution generalizability. To the best of our knowledge, this is the first diffusion-based framework to operate independently of discretization, offering a practical and flexible solution for forward and inverse problems in the context of PDEs. Code is available at this https URL 

**Abstract (ZH)**: 我们提出了一种基于偏微分方程的逆问题中条件采样的通用框架，旨在从极少量或噪声测量中恢复完整解。该框架通过函数空间扩散模型和可插拔的条件引导机制实现。方法首先使用神经算子架构训练一个无条件的离散化无关去噪模型。在推理过程中，通过梯度引导机制细化样本以满足稀疏观测数据的要求。通过严格的数学分析，我们将Tweedie公式扩展到无限维希尔伯特空间，为我们的后验采样方法提供了理论基础。我们的方法（FunDPS）在最少监督和严重数据匮乏的情况下，能够准确捕捉函数空间中的后验分布。在仅3%观测数据的五个PDE任务中，与最先进的固定分辨率扩散 baseline 相比，我们的方法在采样步骤减少4倍的情况下，平均提高了32%的准确性。此外，多分辨率微调确保了跨分辨率的良好泛化能力。据我们所知，这是第一个独立于离散化的扩散基框架，为PDE中的直接和逆问题提供了实用且灵活的解决方案。代码可在以下链接获取。 

---
# CASS: Nvidia to AMD Transpilation with Data, Models, and Benchmark 

**Title (ZH)**: CASS: 从Nvidia到AMD的编译转换，包含数据、模型和基准测试 

**Authors**: Ahmed Heakl, Sarim Hashmi, Gustavo Bertolo Stahl, Seung Hun Eddie Han, Salman Khan, Abdulrahman Mahmoud  

**Link**: [PDF](https://arxiv.org/pdf/2505.16968)  

**Abstract**: We introduce \texttt{CASS}, the first large-scale dataset and model suite for cross-architecture GPU code transpilation, targeting both source-level (CUDA~$\leftrightarrow$~HIP) and assembly-level (Nvidia SASS~$\leftrightarrow$~AMD RDNA3) translation. The dataset comprises 70k verified code pairs across host and device, addressing a critical gap in low-level GPU code portability. Leveraging this resource, we train the \texttt{CASS} family of domain-specific language models, achieving 95\% source translation accuracy and 37.5\% assembly translation accuracy, substantially outperforming commercial baselines such as GPT-4o, Claude, and Hipify. Our generated code matches native performance in over 85\% of test cases, preserving runtime and memory behavior. To support rigorous evaluation, we introduce \texttt{CASS-Bench}, a curated benchmark spanning 16 GPU domains with ground-truth execution. All data, models, and evaluation tools are released as open source to foster progress in GPU compiler tooling, binary compatibility, and LLM-guided hardware translation. Dataset and benchmark are on \href{this https URL}{\textcolor{blue}{HuggingFace}}, with code at \href{this https URL}{\textcolor{blue}{GitHub}}. 

**Abstract (ZH)**: 我们介绍了\texttt{CASS}，这是首个针对跨架构GPU代码转换的大规模数据集和模型套件，旨在实现源代码级别（CUDA $\leftrightarrow$ HIP）和汇编级别（Nvidia SASS $\leftrightarrow$ AMD RDNA3）的翻译。数据集包括70,000个经过验证的代码对，涵盖主机和设备，弥补了低级GPU代码移植的关键空白。利用这一资源，我们训练了\texttt{CASS}家族的领域特定语言模型，实现了95%的源代码翻译准确率和37.5%的汇编代码翻译准确率，大幅优于GPT-4o、Claude和Hipify等商用基线。我们生成的代码在超过85%的测试用例中达到了原生性能，保持了运行时和内存行为的一致性。为了支持严格的评估，我们引入了\texttt{CASS-Bench}，这是一个涵盖了16个GPU领域的精编基准测试集，具有真实的执行结果。所有数据、模型和评估工具均开源，以促进GPU编译器工具链、二进制兼容性和LLM引导硬件转换的进步。数据集和基准测试集可在\href{this https URL}{HuggingFace}获取，代码位于\href{this https URL}{GitHub}。 

---
# BP-Seg: A graphical model approach to unsupervised and non-contiguous text segmentation using belief propagation 

**Title (ZH)**: BP-Seg: 使用信念传播的无监督和非连续文本分割图形模型方法 

**Authors**: Fengyi Li, Kayhan Behdin, Natesh Pillai, Xiaofeng Wang, Zhipeng Wang, Ercan Yildiz  

**Link**: [PDF](https://arxiv.org/pdf/2505.16965)  

**Abstract**: Text segmentation based on the semantic meaning of sentences is a fundamental task with broad utility in many downstream applications. In this paper, we propose a graphical model-based unsupervised learning approach, named BP-Seg for efficient text segmentation. Our method not only considers local coherence, capturing the intuition that adjacent sentences are often more related, but also effectively groups sentences that are distant in the text yet semantically similar. This is achieved through belief propagation on the carefully constructed graphical models. Experimental results on both an illustrative example and a dataset with long-form documents demonstrate that our method performs favorably compared to competing approaches. 

**Abstract (ZH)**: 基于句子语义意义的文本切分：一种基于图形模型的无监督学习方法 

---
# FoMoH: A clinically meaningful foundation model evaluation for structured electronic health records 

**Title (ZH)**: FoMoH: 一个临床相关的基础模型评估框架用于结构化的电子健康记录 

**Authors**: Chao Pang, Vincent Jeanselme, Young Sang Choi, Xinzhuo Jiang, Zilin Jing, Aparajita Kashyap, Yuta Kobayashi, Yanwei Li, Florent Pollet, Karthik Natarajan, Shalmali Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2505.16941)  

**Abstract**: Foundation models hold significant promise in healthcare, given their capacity to extract meaningful representations independent of downstream tasks. This property has enabled state-of-the-art performance across several clinical applications trained on structured electronic health record (EHR) data, even in settings with limited labeled data, a prevalent challenge in healthcare. However, there is little consensus on these models' potential for clinical utility due to the lack of desiderata of comprehensive and meaningful tasks and sufficiently diverse evaluations to characterize the benefit over conventional supervised learning. To address this gap, we propose a suite of clinically meaningful tasks spanning patient outcomes, early prediction of acute and chronic conditions, including desiderata for robust evaluations. We evaluate state-of-the-art foundation models on EHR data consisting of 5 million patients from Columbia University Irving Medical Center (CUMC), a large urban academic medical center in New York City, across 14 clinically relevant tasks. We measure overall accuracy, calibration, and subpopulation performance to surface tradeoffs based on the choice of pre-training, tokenization, and data representation strategies. Our study aims to advance the empirical evaluation of structured EHR foundation models and guide the development of future healthcare foundation models. 

**Abstract (ZH)**: 基于模型在医疗健康领域的应用中展现出显著潜力，得益于它们能够独立于下游任务提取有意义的表示。这一特性使得这些模型能够在结构化的电子健康记录(EHR)数据上训练，并在有限标注数据的环境中实现最先进的临床应用性能。然而，由于缺乏全面且有意义的任务定义以及足够多样的评估来表征与传统监督学习相比的优势，这些模型的临床实用性还有待商榷。为填补这一空白，我们提出了涵盖患者结局、急性及慢性疾病早期预测等一系列临床相关任务的方案，并为此类任务制定了评估标准。我们评估了各种先进的基础模型在包含50万患者的哥伦比亚大学欧文医学中心(CUMC)的EHR数据上的性能，涵盖了14个临床相关任务。我们衡量总体准确性、校准情况以及子人群表现，以基于预训练策略、分词方法和数据表示方法的选择探讨各项权衡。我们的研究旨在推动结构化EHR基础模型的实证评估，并指导未来医疗健康基础模型的发展。 

---
# The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm 

**Title (ZH)**: 极地 Express：最优矩阵符号方法及其在穆链算法中的应用 

**Authors**: Noah Amsel, David Persson, Christopher Musco, Robert Gower  

**Link**: [PDF](https://arxiv.org/pdf/2505.16932)  

**Abstract**: Computing the polar decomposition and the related matrix sign function, has been a well-studied problem in numerical analysis for decades. More recently, it has emerged as an important subroutine in deep learning, particularly within the Muon optimization framework. However, the requirements in this setting differ significantly from those of traditional numerical analysis. In deep learning, methods must be highly efficient and GPU-compatible, but high accuracy is often unnecessary. As a result, classical algorithms like Newton-Schulz (which suffers from slow initial convergence) and methods based on rational functions (which rely on QR decompositions or matrix inverses) are poorly suited to this context. In this work, we introduce Polar Express, a GPU-friendly algorithm for computing the polar decomposition. Like classical polynomial methods such as Newton-Schulz, our approach uses only matrix-matrix multiplications, making it GPU-compatible. Motivated by earlier work of Chen & Chow and Nakatsukasa & Freund, Polar Express adapts the polynomial update rule at each iteration by solving a minimax optimization problem, and we prove that it enjoys a strong worst-case optimality guarantee. This property ensures both rapid early convergence and fast asymptotic convergence. We also address finite-precision issues, making it stable in bfloat16 in practice. We apply Polar Express within the Muon optimization framework and show consistent improvements in validation loss on large-scale models such as GPT-2, outperforming recent alternatives across a range of learning rates. 

**Abstract (ZH)**: 计算极分解及其相关矩阵符号函数在数值分析中已有几十年的研究历史。近年来，它已成为深度学习中的一个重要子程序，特别是在Muon优化框架中。然而，在这种情况下的要求与传统数值分析的要求大不相同。在深度学习中，方法必须极其高效且兼容GPU，但高精度往往是不必要的。因此，像Newton-Schulz这样经典的算法（由于初始收敛速度慢）和基于有理函数的方法（依赖于QR分解或矩阵逆）在这种背景下适应性较差。本文我们介绍了Polar Express，这是一种兼容GPU的用于计算极分解的算法。就像经典的多项式方法（如Newton-Schulz）一样，我们的方法仅使用矩阵-矩阵乘法，使其兼容GPU。受Chen & Chow和Nakatsukasa & Freund早期工作的启发，Polar Express在每次迭代中通过求解极小极大优化问题来适应多项式更新规则，并证明它享有强大的最坏情况最优性保证。这一特性确保了快速的早期收敛和快速的渐近收敛。我们还解决了有限精度问题，使其在实践中能够在bfloat16中稳定运行。我们在Muon优化框架中应用Polar Express，并在大型模型（如GPT-2）上展示了验证损失的一致改善，在不同学习率范围内优于近期的替代方法。 

---
# Active Speech Enhancement: Active Speech Denoising Decliping and Deveraberation 

**Title (ZH)**: 主动语音增强：主动语音降噪、去Clip和去失真 

**Authors**: Ofir Yaish, Yehuda Mishaly, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2505.16911)  

**Abstract**: We introduce a new paradigm for active sound modification: Active Speech Enhancement (ASE). While Active Noise Cancellation (ANC) algorithms focus on suppressing external interference, ASE goes further by actively shaping the speech signal -- both attenuating unwanted noise components and amplifying speech-relevant frequencies -- to improve intelligibility and perceptual quality. To enable this, we propose a novel Transformer-Mamba-based architecture, along with a task-specific loss function designed to jointly optimize interference suppression and signal enrichment. Our method outperforms existing baselines across multiple speech processing tasks -- including denoising, dereverberation, and declipping -- demonstrating the effectiveness of active, targeted modulation in challenging acoustic environments. 

**Abstract (ZH)**: 基于Transformer-Mamba的主动语音增强（ASE）新范式 

---
# Structure-Aligned Protein Language Model 

**Title (ZH)**: 结构对齐蛋白质语言模型 

**Authors**: Can Chen, David Heurtel-Depeiges, Robert M. Vernon, Christopher James Langmead, Yoshua Bengio, Quentin Fournier  

**Link**: [PDF](https://arxiv.org/pdf/2505.16896)  

**Abstract**: Protein language models (pLMs) pre-trained on vast protein sequence databases excel at various downstream tasks but lack the structural knowledge essential for many biological applications. To address this, we integrate structural insights from pre-trained protein graph neural networks (pGNNs) into pLMs through a latent-level contrastive learning task. This task aligns residue representations from pLMs with those from pGNNs across multiple proteins, enriching pLMs with inter-protein structural knowledge. Additionally, we incorporate a physical-level task that infuses intra-protein structural knowledge by optimizing pLMs to predict structural tokens. The proposed dual-task framework effectively incorporates both inter-protein and intra-protein structural knowledge into pLMs. Given the variability in the quality of protein structures in PDB, we further introduce a residue loss selection module, which uses a small model trained on high-quality structures to select reliable yet challenging residue losses for the pLM to learn. Applying our structure alignment method to the state-of-the-art ESM2 and AMPLIFY results in notable performance gains across a wide range of tasks, including a 12.7% increase in ESM2 contact prediction. The data, code, and resulting SaESM2 and SaAMPLIFY models will be released on Hugging Face. 

**Abstract (ZH)**: 基于结构的蛋白语言模型：通过双重任务框架融合 residue 表征增强蛋白间与蛋白内结构知识 

---
# GCAL: Adapting Graph Models to Evolving Domain Shifts 

**Title (ZH)**: GCAL：适应演变领域偏移的图模型改编 

**Authors**: Ziyue Qiao, Qianyi Cai, Hao Dong, Jiawei Gu, Pengyang Wang, Meng Xiao, Xiao Luo, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.16860)  

**Abstract**: This paper addresses the challenge of graph domain adaptation on evolving, multiple out-of-distribution (OOD) graphs. Conventional graph domain adaptation methods are confined to single-step adaptation, making them ineffective in handling continuous domain shifts and prone to catastrophic forgetting. This paper introduces the Graph Continual Adaptive Learning (GCAL) method, designed to enhance model sustainability and adaptability across various graph domains. GCAL employs a bilevel optimization strategy. The "adapt" phase uses an information maximization approach to fine-tune the model with new graph domains while re-adapting past memories to mitigate forgetting. Concurrently, the "generate memory" phase, guided by a theoretical lower bound derived from information bottleneck theory, involves a variational memory graph generation module to condense original graphs into memories. Extensive experimental evaluations demonstrate that GCAL substantially outperforms existing methods in terms of adaptability and knowledge retention. 

**Abstract (ZH)**: 这种论文解决了 evolving、multiple out-of-distribution (OOD) 图的图域适应挑战。传统的图域适应方法局限于单步适应，使其在处理连续域移位时效果不佳，并且容易出现灾难性遗忘。本文引入了图持续适应学习（GCAL）方法，旨在提高模型在各种图域中的可持续性和适应性。GCAL采用多层次优化策略。“adapt”阶段使用信息最大化的手段对模型进行微调，并重新适应过去的记忆以减轻遗忘。“generate memory”阶段则通过信息瓶颈理论推导出的理论下界指导，利用变分记忆图生成模块将原始图压缩成记忆。广泛的实验证明，GCAL在适应性和知识保留方面显著优于现有方法。 

---
# Unlocking Temporal Flexibility: Neural Speech Codec with Variable Frame Rate 

**Title (ZH)**: 解锁时间灵活性：具有可变帧率的神经语音编解码器 

**Authors**: Hanglei Zhang, Yiwei Guo, Zhihan Li, Xiang Hao, Xie Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16845)  

**Abstract**: Most neural speech codecs achieve bitrate adjustment through intra-frame mechanisms, such as codebook dropout, at a Constant Frame Rate (CFR). However, speech segments inherently have time-varying information density (e.g., silent intervals versus voiced regions). This property makes CFR not optimal in terms of bitrate and token sequence length, hindering efficiency in real-time applications. In this work, we propose a Temporally Flexible Coding (TFC) technique, introducing variable frame rate (VFR) into neural speech codecs for the first time. TFC enables seamlessly tunable average frame rates and dynamically allocates frame rates based on temporal entropy. Experimental results show that a codec with TFC achieves optimal reconstruction quality with high flexibility, and maintains competitive performance even at lower frame rates. Our approach is promising for the integration with other efforts to develop low-frame-rate neural speech codecs for more efficient downstream tasks. 

**Abstract (ZH)**: Temporal Flexibility Coding for Neural Speech Codecs 

---
# Dynamic Reservoir Computing with Physical Neuromorphic Networks 

**Title (ZH)**: 物理神经形态网络中的动态 reservoir 计算 

**Authors**: Yinhao Xu, Georg A. Gottwald, Zdenka Kuncic  

**Link**: [PDF](https://arxiv.org/pdf/2505.16813)  

**Abstract**: Reservoir Computing (RC) with physical systems requires an understanding of the underlying structure and internal dynamics of the specific physical reservoir. In this study, physical nano-electronic networks with neuromorphic dynamics are investigated for their use as physical reservoirs in an RC framework. These neuromorphic networks operate as dynamic reservoirs, with node activities in general coupled to the edge dynamics through nonlinear nano-electronic circuit elements, and the reservoir outputs influenced by the underlying network connectivity structure. This study finds that networks with varying degrees of sparsity generate more useful nonlinear temporal outputs for dynamic RC compared to dense networks. Dynamic RC is also tested on an autonomous multivariate chaotic time series prediction task with networks of varying densities, which revealed the importance of network sparsity in maintaining network activity and overall dynamics, that in turn enabled the learning of the chaotic Lorenz63 system's attractor behavior. 

**Abstract (ZH)**: 物理系统中的人工神经网络计算（Reservoir Computing with Physical Nano-Electronic Networks and Neuromorphic Dynamics）及其在动态人工神经网络计算框架中的应用 

---
# A modular framework for automated evaluation of procedural content generation in serious games with deep reinforcement learning agents 

**Title (ZH)**: 基于深度强化学习代理的严肃游戏中程序内容生成的自动化评估模块化框架 

**Authors**: Eleftherios Kalafatis, Konstantinos Mitsis, Konstantia Zarkogianni, Maria Athanasiou, Konstantina Nikita  

**Link**: [PDF](https://arxiv.org/pdf/2505.16801)  

**Abstract**: Serious Games (SGs) are nowadays shifting focus to include procedural content generation (PCG) in the development process as a means of offering personalized and enhanced player experience. However, the development of a framework to assess the impact of PCG techniques when integrated into SGs remains particularly challenging. This study proposes a methodology for automated evaluation of PCG integration in SGs, incorporating deep reinforcement learning (DRL) game testing agents. To validate the proposed framework, a previously introduced SG featuring card game mechanics and incorporating three different versions of PCG for nonplayer character (NPC) creation has been deployed. Version 1 features random NPC creation, while versions 2 and 3 utilize a genetic algorithm approach. These versions are used to test the impact of different dynamic SG environments on the proposed framework's agents. The obtained results highlight the superiority of the DRL game testing agents trained on Versions 2 and 3 over those trained on Version 1 in terms of win rate (i.e. number of wins per played games) and training time. More specifically, within the execution of a test emulating regular gameplay, both Versions 2 and 3 peaked at a 97% win rate and achieved statistically significant higher (p=0009) win rates compared to those achieved in Version 1 that peaked at 94%. Overall, results advocate towards the proposed framework's capability to produce meaningful data for the evaluation of procedurally generated content in SGs. 

**Abstract (ZH)**: 严肃游戏（SGs）通过引入过程性内容生成（PCG）来开发个性化和增强的游戏体验正逐渐成为主流。然而，开发一种评估PCG技术在SGs中集成影响的框架仍然极具挑战性。本研究提出了一种结合深度强化学习（DRL）游戏测试代理的方法论，以自动评估PCG在SGs中的集成。为了验证提出的框架，使用了一款先前介绍的结合了卡牌游戏机制并在非玩家角色（NPC）创建中采用三种不同PCG版本的严肃游戏。第一版采用随机生成NPC，而第二版和第三版则采用遗传算法的方法。这些版本用于测试不同动态SG环境对提出的框架代理的影响。研究结果表明，在以常规游戏方式执行的测试中，采用第二版和第三版训练的DRL游戏测试代理的胜率（即每场比赛的获胜次数）和训练时间均优于采用第一版训练的代理。具体而言，第二版和第三版在测试中分别达到了97%的胜率，并且与第一版94%的胜率相比，具有统计学意义上的显著差异（p=0.009）。总体而言，结果表明提出的框架能够产生可用于评估SGs中过程性生成内容的有效数据。 

---
# SEED: Speaker Embedding Enhancement Diffusion Model 

**Title (ZH)**: SEED: 言语嵌入增强扩散模型 

**Authors**: KiHyun Nam, Jungwoo Heo, Jee-weon Jung, Gangin Park, Chaeyoung Jung, Ha-Jin Yu, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2505.16798)  

**Abstract**: A primary challenge when deploying speaker recognition systems in real-world applications is performance degradation caused by environmental mismatch. We propose a diffusion-based method that takes speaker embeddings extracted from a pre-trained speaker recognition model and generates refined embeddings. For training, our approach progressively adds Gaussian noise to both clean and noisy speaker embeddings extracted from clean and noisy speech, respectively, via forward process of a diffusion model, and then reconstructs them to clean embeddings in the reverse process. While inferencing, all embeddings are regenerated via diffusion process. Our method needs neither speaker label nor any modification to the existing speaker recognition pipeline. Experiments on evaluation sets simulating environment mismatch scenarios show that our method can improve recognition accuracy by up to 19.6% over baseline models while retaining performance on conventional scenarios. We publish our code here this https URL 

**Abstract (ZH)**: 在实际应用中部署说话人识别系统的主要挑战是由于环境不匹配导致的性能下降。我们提出了一种基于扩散的方法，该方法从预训练的说话人识别模型中提取说话人嵌入，并生成精炼的嵌入。在训练过程中，我们的方法通过扩散模型的前向过程，逐步将高斯噪声添加到分别从干净语音和噪声语音中提取的干净和噪声说话人嵌入中，并在反向过程中重建它们为干净嵌入。在推理过程中，所有嵌入都通过扩散过程再生。该方法既不需要说话人标签，也不需要修改现有的说话人识别管道。在模拟环境不匹配场景的评估集上进行的实验表明，与基线模型相比，我们的方法可以在保持传统场景性能的同时将识别准确性提升多达19.6%。我们在这里发布了我们的代码：https://github.com/username/repo 

---
# Cohort-Based Active Modality Acquisition 

**Title (ZH)**: 基于群体的主动模态 acquisition 

**Authors**: Tillmann Rheude, Roland Eils, Benjamin Wild  

**Link**: [PDF](https://arxiv.org/pdf/2505.16791)  

**Abstract**: Real-world machine learning applications often involve data from multiple modalities that must be integrated effectively to make robust predictions. However, in many practical settings, not all modalities are available for every sample, and acquiring additional modalities can be costly. This raises the question: which samples should be prioritized for additional modality acquisition when resources are limited? While prior work has explored individual-level acquisition strategies and training-time active learning paradigms, test-time and cohort-based acquisition remain underexplored despite their importance in many real-world settings. We introduce Cohort-based Active Modality Acquisition (CAMA), a novel test-time setting to formalize the challenge of selecting which samples should receive additional modalities. We derive acquisition strategies that leverage a combination of generative imputation and discriminative modeling to estimate the expected benefit of acquiring missing modalities based on common evaluation metrics. We also introduce upper-bound heuristics that provide performance ceilings to benchmark acquisition strategies. Experiments on common multimodal datasets demonstrate that our proposed imputation-based strategies can more effectively guide the acquisition of new samples in comparison to those relying solely on unimodal information, entropy guidance, and random selections. Our work provides an effective solution for optimizing modality acquisition at the cohort level, enabling better utilization of resources in constrained settings. 

**Abstract (ZH)**: 基于群体的主动模态获取（CAMA）：实世界多模态数据的测试时选择策略 

---
# Learning Flexible Forward Trajectories for Masked Molecular Diffusion 

**Title (ZH)**: 学习灵活的掩码分子扩散前向轨迹 

**Authors**: Hyunjin Seo, Taewon Kim, Sihyun Yu, SungSoo Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2505.16790)  

**Abstract**: Masked diffusion models (MDMs) have achieved notable progress in modeling discrete data, while their potential in molecular generation remains underexplored. In this work, we explore their potential and introduce the surprising result that naively applying standards MDMs severely degrades the performance. We identify the critical cause of this issue as a state-clashing problem-where the forward diffusion of distinct molecules collapse into a common state, resulting in a mixture of reconstruction targets that cannot be learned using typical reverse diffusion process with unimodal predictions. To mitigate this, we propose Masked Element-wise Learnable Diffusion (MELD) that orchestrates per-element corruption trajectories to avoid collision between distinct molecular graphs. This is achieved through a parameterized noise scheduling network that assigns distinct corruption rates to individual graph elements, i.e., atoms and bonds. Extensive experiments on diverse molecular benchmarks reveal that MELD markedly enhances overall generation quality compared to element-agnostic noise scheduling, increasing the chemical validity of vanilla MDMs on ZINC250K from 15% to 93%, Furthermore, it achieves state-of-the-art property alignment in conditional generation tasks. 

**Abstract (ZH)**: 掩码扩散模型在分子生成领域的潜力尚未充分探索。在此工作中，我们探索了其潜力，并引入了一个令人惊讶的结果，即直接应用标准掩码扩散模型显著降低了性能。我们识别出这一问题的关键原因是状态冲突问题——不同的分子在前向扩散过程中坍缩到一个共同状态，导致无法使用典型的具有单模预测的逆向扩散过程来学习混合的重构目标。为缓解这一问题，我们提出了掩码元素级可学习扩散（MELD），通过调控每个元素的破坏轨迹来避免不同的分子图之间的碰撞。这通过一个参数化的噪声调度网络实现，该网络为单个图元素（即原子和键）分配不同的破坏率。在多种分子基准上的广泛实验表明，与元素无关的噪声调度相比，MELD 明显提高了整体生成质量，将 ZINC250K 上 vanilla MDM 的化学有效性从 15% 提高到 93%。此外，它在条件生成任务中达到了最先进的属性对齐效果。 

---
# Mitigating Overfitting in Medical Imaging: Self-Supervised Pretraining vs. ImageNet Transfer Learning for Dermatological Diagnosis 

**Title (ZH)**: 医学影像中的过拟合缓解：皮肤科诊断中自我监督预训练与ImageNet迁移学习的比较 

**Authors**: Iván Matas, Carmen Serrano, Miguel Nogales, David Moreno, Lara Ferrándiz, Teresa Ojeda, Begoña Acha  

**Link**: [PDF](https://arxiv.org/pdf/2505.16773)  

**Abstract**: Deep learning has transformed computer vision but relies heavily on large labeled datasets and computational resources. Transfer learning, particularly fine-tuning pretrained models, offers a practical alternative; however, models pretrained on natural image datasets such as ImageNet may fail to capture domain-specific characteristics in medical imaging. This study introduces an unsupervised learning framework that extracts high-value dermatological features instead of relying solely on ImageNet-based pretraining. We employ a Variational Autoencoder (VAE) trained from scratch on a proprietary dermatological dataset, allowing the model to learn a structured and clinically relevant latent space. This self-supervised feature extractor is then compared to an ImageNet-pretrained backbone under identical classification conditions, highlighting the trade-offs between general-purpose and domain-specific pretraining. Our results reveal distinct learning patterns. The self-supervised model achieves a final validation loss of 0.110 (-33.33%), while the ImageNet-pretrained model stagnates at 0.100 (-16.67%), indicating overfitting. Accuracy trends confirm this: the self-supervised model improves from 45% to 65% (+44.44%) with a near-zero overfitting gap, whereas the ImageNet-pretrained model reaches 87% (+50.00%) but plateaus at 75% (+19.05%), with its overfitting gap increasing to +0.060. These findings suggest that while ImageNet pretraining accelerates convergence, it also amplifies overfitting on non-clinically relevant features. In contrast, self-supervised learning achieves steady improvements, stronger generalization, and superior adaptability, underscoring the importance of domain-specific feature extraction in medical imaging. 

**Abstract (ZH)**: 深度学习已变革计算机视觉，但依赖大量标注数据和计算资源。迁移学习，尤其是微调预训练模型，提供了一种实用的选择；然而，基于自然图像数据集（如ImageNet）预训练的模型可能无法捕捉医学影像中的领域特定特征。本研究引入了一种无监督学习框架，提取高价值的皮肤病学特征，而非仅仅依赖于ImageNet预训练。我们采用从一个专用皮肤病学数据集从零开始训练的变分自编码器（VAE），使模型能够学习一个结构化和临床相关的潜在空间。随后，该自监督特征提取器与在ImageNet上预训练的骨干网络在相同的分类条件下进行比较，突显通用预训练和领域特定预训练之间的权衡。我们的结果显示不同的学习模式。自监督模型在最终验证损失为0.110（-33.33%），而基于ImageNet预训练的模型停滞在0.100（-16.67%），表明过拟合现象。准确率趋势也证实了这一点：自监督模型从45%提高到65%（+44.44%），几乎无过拟合差距，而基于ImageNet预训练的模型达到87%（+50.00%），但停滞在75%（+19.05%），其过拟合差距增加到+0.060。这些发现表明虽然ImageNet预训练加速了收敛，但也放大了对非临床相关特征的过拟合现象。相比之下，自监督学习实现了稳定改进、更强的泛化能力和更优的适应性，突显了在医学影像中领域特定特征提取的重要性。 

---
# Action is All You Need: Dual-Flow Generative Ranking Network for Recommendation 

**Title (ZH)**: 只需行动：基于双流生成排序网络的推荐 

**Authors**: Hao Guo, Erpeng Xue, Lei Huang, Shichao Wang, Xiaolei Wang, Lei Wang, Jinpeng Wang, Sheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16752)  

**Abstract**: We introduce the Dual-Flow Generative Ranking Network (DFGR), a two-stream architecture designed for recommendation systems. DFGR integrates innovative interaction patterns between real and fake flows within the QKV modules of the self-attention mechanism, enhancing both training and inference efficiency. This approach effectively addresses a key limitation observed in Meta's proposed HSTU generative recommendation approach, where heterogeneous information volumes are mapped into identical vector spaces, leading to training instability. Unlike traditional recommendation models, DFGR only relies on user history behavior sequences and minimal attribute information, eliminating the need for extensive manual feature engineering. Comprehensive evaluations on open-source and industrial datasets reveal DFGR's superior performance compared to established baselines such as DIN, DCN, DIEN, and DeepFM. We also investigate optimal parameter allocation strategies under computational constraints, establishing DFGR as an efficient and effective next-generation generate ranking paradigm. 

**Abstract (ZH)**: 双流生成排名网络（DFGR）：一种针对推荐系统的设计结构 

---
# Sequential Monte Carlo for Policy Optimization in Continuous POMDPs 

**Title (ZH)**: 连续部分可观测马尔可夫决策过程的顺序蒙特卡洛策略优化 

**Authors**: Hany Abdulsamad, Sahel Iqbal, Simo Särkkä  

**Link**: [PDF](https://arxiv.org/pdf/2505.16732)  

**Abstract**: Optimal decision-making under partial observability requires agents to balance reducing uncertainty (exploration) against pursuing immediate objectives (exploitation). In this paper, we introduce a novel policy optimization framework for continuous partially observable Markov decision processes (POMDPs) that explicitly addresses this challenge. Our method casts policy learning as probabilistic inference in a non-Markovian Feynman--Kac model that inherently captures the value of information gathering by anticipating future observations, without requiring extrinsic exploration bonuses or handcrafted heuristics. To optimize policies under this model, we develop a nested sequential Monte Carlo~(SMC) algorithm that efficiently estimates a history-dependent policy gradient under samples from the optimal trajectory distribution induced by the POMDP. We demonstrate the effectiveness of our algorithm across standard continuous POMDP benchmarks, where existing methods struggle to act under uncertainty. 

**Abstract (ZH)**: 部分可观测条件下最优决策要求智能体在降低不确定性（探索）与追求即时目标（利用）之间取得平衡。本文介绍了一种新的连续部分可观测马尔可夫决策过程（POMDP）的策略优化框架，该框架明确解决了上述挑战。我们的方法将策略学习视为非马尔可夫费曼-卡克模型中的概率推理问题，该模型本身能够通过预见未来的观测结果来捕获信息收集的价值，无需额外的探索奖励或手工设计的启发式方法。为了在该模型下优化策略，我们开发了一种嵌套顺序蒙特卡罗（SMC）算法，该算法能够高效地在由POMDP诱导的最优轨迹分布的样本下估计依赖历史的策略梯度。我们在标准连续POMDP基准测试中展示了该算法的有效性，而现有方法在不确定性下很难采取行动。 

---
# An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations 

**Title (ZH)**: 概念瓶颈模型的分析：衡量、理解及其对嘈杂标注影响的缓解 

**Authors**: Seonghwan Park, Jueun Mun, Donghyun Oh, Namhoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.16705)  

**Abstract**: Concept bottleneck models (CBMs) ensure interpretability by decomposing predictions into human interpretable concepts. Yet the annotations used for training CBMs that enable this transparency are often noisy, and the impact of such corruption is not well understood. In this study, we present the first systematic study of noise in CBMs and show that even moderate corruption simultaneously impairs prediction performance, interpretability, and the intervention effectiveness. Our analysis identifies a susceptible subset of concepts whose accuracy declines far more than the average gap between noisy and clean supervision and whose corruption accounts for most performance loss. To mitigate this vulnerability we propose a two-stage framework. During training, sharpness-aware minimization stabilizes the learning of noise-sensitive concepts. During inference, where clean labels are unavailable, we rank concepts by predictive entropy and correct only the most uncertain ones, using uncertainty as a proxy for susceptibility. Theoretical analysis and extensive ablations elucidate why sharpness-aware training confers robustness and why uncertainty reliably identifies susceptible concepts, providing a principled basis that preserves both interpretability and resilience in the presence of noise. 

**Abstract (ZH)**: 概念瓶颈模型中噪声的系统研究及其缓解方法 

---
# EZ-VC: Easy Zero-shot Any-to-Any Voice Conversion 

**Title (ZH)**: EZ-VC: Easy Zero-shot Any-to-Any语音转换 

**Authors**: Advait Joglekar, Divyanshu Singh, Rooshil Rohit Bhatia, S. Umesh  

**Link**: [PDF](https://arxiv.org/pdf/2505.16691)  

**Abstract**: Voice Conversion research in recent times has increasingly focused on improving the zero-shot capabilities of existing methods. Despite remarkable advancements, current architectures still tend to struggle in zero-shot cross-lingual settings. They are also often unable to generalize for speakers of unseen languages and accents. In this paper, we adopt a simple yet effective approach that combines discrete speech representations from self-supervised models with a non-autoregressive Diffusion-Transformer based conditional flow matching speech decoder. We show that this architecture allows us to train a voice-conversion model in a purely textless, self-supervised fashion. Our technique works without requiring multiple encoders to disentangle speech features. Our model also manages to excel in zero-shot cross-lingual settings even for unseen languages. 

**Abstract (ZH)**: 近期的语音转换研究越来越多地关注于提高现有方法的零样本能力。尽管取得了显著的进步，当前的架构在零样本跨语言设置中仍然会遇到困难，通常也无法泛化到未见语言和口音的说话人。在本文中，我们采用了一种简单而有效的方法，将自监督模型的离散语音表示与非自回归扩散变换器条件流匹配语音解码器相结合。我们展示了这种架构使我们能够以纯粹无文本、自监督的方式训练一个语音转换模型。我们的方法无需使用多个编码器来解码语音特征即可工作，模型也能够在未见语言的零样本跨语言设置中表现出色。 

---
# Semantic Compression of 3D Objects for Open and Collaborative Virtual Worlds 

**Title (ZH)**: 开放协作虚拟世界中3D对象的语义压缩 

**Authors**: Jordan Dotzel, Tony Montes, Mohamed S. Abdelfattah, Zhiru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16679)  

**Abstract**: Traditional methods for 3D object compression operate only on structural information within the object vertices, polygons, and textures. These methods are effective at compression rates up to 10x for standard object sizes but quickly deteriorate at higher compression rates with texture artifacts, low-polygon counts, and mesh gaps. In contrast, semantic compression ignores structural information and operates directly on the core concepts to push to extreme levels of compression. In addition, it uses natural language as its storage format, which makes it natively human-readable and a natural fit for emerging applications built around large-scale, collaborative projects within augmented and virtual reality. It deprioritizes structural information like location, size, and orientation and predicts the missing information with state-of-the-art deep generative models. In this work, we construct a pipeline for 3D semantic compression from public generative models and explore the quality-compression frontier for 3D object compression. We apply this pipeline to achieve rates as high as 105x for 3D objects taken from the Objaverse dataset and show that semantic compression can outperform traditional methods in the important quality-preserving region around 100x compression. 

**Abstract (ZH)**: 传统的3D对象压缩方法仅作用于对象顶点、多边形和纹理的结构信息。这些方法在标准对象大小下的压缩率高达10倍，但在更高压缩率下迅速恶化，出现纹理artifact、低多边形数量和网格间隙问题。相比之下，语义压缩忽略结构信息，直接作用于核心概念，以实现极致的压缩效果。此外，它使用自然语言作为存储格式，使其天然具有人类可读性，适配于基于增强和虚拟现实的大型协作项目中的新兴应用。它优先考虑预测丢失信息，而非结构性信息如位置、大小和方向。在此工作中，我们从公共生成模型构建了3D语义压缩的管道，并探索了3D对象压缩的质量-压缩前沿。我们将此管道应用于从Objaverse数据集中获取的3D对象，实现了高达105倍的压缩率，并展示了在约100倍压缩率的重要质量保持区域内，语义压缩可以超越传统方法。 

---
# End-to-End Framework for Predicting the Remaining Useful Life of Lithium-Ion Batteries 

**Title (ZH)**: 端到端框架预测锂离子电池的剩余使用寿命 

**Authors**: Khoa Tran, Tri Le, Bao Huynh, Hung-Cuong Trinh, Vy-Rin Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16664)  

**Abstract**: Accurate prediction of the Remaining Useful Life (RUL) is essential for enabling timely maintenance of lithium-ion batteries, impacting the operational efficiency of electric applications that rely on them. This paper proposes a RUL prediction approach that leverages data from recent charge-discharge cycles to estimate the number of remaining usable cycles. The approach introduces both a novel signal processing pipeline and a deep learning prediction model. In the signal preprocessing pipeline, a derived capacity feature is computed based on current and capacity signals. Alongside original capacity, voltage and current, these features are denoised and enhanced using statistical metrics and a delta-based method to capture differences between the current and previous cycles. In the prediction model, the processed features are then fed into a hybrid deep learning architecture composed of 1D Convolutional Neural Networks (CNN), Attentional Long Short-Term Memory (A-LSTM), and Ordinary Differential Equation-based LSTM (ODE-LSTM) modules. This architecture is designed to capture both local signal characteristics and long-range temporal dependencies while modeling the continuous-time dynamics of battery degradation. The model is further evaluated using transfer learning across different learning strategies and target data partitioning scenarios. Results indicate that the model maintains robust performance, even when fine-tuned on limited target data. Experimental results on two publicly available large-scale datasets demonstrate that the proposed method outperforms a baseline deep learning approach and machine learning techniques, achieving an RMSE of 101.59, highlighting its strong potential for real-world RUL prediction applications. 

**Abstract (ZH)**: 准确预测锂离子电池的剩余使用寿命对于实现对依赖它们的电动应用的及时维护至关重要，影响其运营效率。本文提出了一种RUL预测方法，该方法利用最近的充放电循环数据来估计剩余可用循环次数。该方法引入了一个新型信号处理管道和一种深度学习预测模型。在信号预处理管道中，基于电流和容量信号计算了一个衍生容量特征。这些特征与原始容量、电压和电流一起，通过统计指标和基于差分的方法进行降噪和增强，以捕捉与当前循环和前一循环之间的差异。在预测模型中，预处理特征被输入由1D卷积神经网络（CNN）、注意力长期短期记忆（A-LSTM）和基于常微分方程的长期短期记忆（ODE-LSTM）模块组成的混合深度学习架构。该架构旨在捕捉局部信号特征和长距离时间依赖关系，同时建模电池退化的连续时间动态。通过不同学习策略和目标数据划分场景下的迁移学习进一步评估了该模型。结果显示，即使在目标数据微调较少的情况下，模型也保持了稳健的性能。实验结果表明，该方法在两个公开的大型数据集上优于基线深度学习方法和机器学习技术，RMSE为101.59，显示出其在实际RUL预测应用中的强大潜力。 

---
# Can reasoning models comprehend mathematical problems in Chinese ancient texts? An empirical study based on data from Suanjing Shishu 

**Title (ZH)**: 古籍《算经十书》中的数学问题，推理模型能否理解？基于实证研究 

**Authors**: Liu Chang, Wang Dongbo, Liu liu, Zhao Zhixiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16660)  

**Abstract**: This study addresses the challenges in intelligent processing of Chinese ancient mathematical classics by constructing Guji_MATH, a benchmark for evaluating classical texts based on Suanjing Shishu. It systematically assesses the mathematical problem-solving capabilities of mainstream reasoning models under the unique linguistic constraints of classical Chinese. Through machine-assisted annotation and manual verification, 538 mathematical problems were extracted from 8 canonical texts, forming a structured dataset centered on the "Question-Answer-Solution" framework, supplemented by problem types and difficulty levels. Dual evaluation modes--closed-book (autonomous problem-solving) and open-book (reproducing classical solution methods)--were designed to evaluate the performance of six reasoning models on ancient Chinese mathematical problems. Results indicate that reasoning models can partially comprehend and solve these problems, yet their overall performance remains inferior to benchmarks on modern mathematical tasks. Enhancing models' classical Chinese comprehension and cultural knowledge should be prioritized for optimization. This study provides methodological support for mining mathematical knowledge from ancient texts and disseminating traditional culture, while offering new perspectives for evaluating cross-linguistic and cross-cultural capabilities of reasoning models. 

**Abstract (ZH)**: 本研究通过构建Guji_MATHbenchmark，针对中国古代数学经典智能化处理的挑战进行了探讨，系统评估了主流推理模型在古典中文独特语言约束下的数学问题解决能力。通过机器辅助注释和人工验证，从8部经典文献中提取了538个数学问题，形成了以“问题-答案-解决方案”框架为中心的结构化数据集，并辅以问题类型和难度级别。设计了闭卷（自主问题解决）和开卷（再现古典解题方法）两种评估模式，以评估六种推理模型在古代汉语数学问题上的性能。研究结果表明，推理模型能够部分理解和解决这些问题，但在整体性能上仍不及现代数学任务的基准。优化模型的古典汉语理解和文化知识应被优先考虑。本研究为从古籍中挖掘数学知识和传播传统文化提供了方法学支持，并为评估跨语言和跨文化推理模型的能力提供了新视角。 

---
# How Ensembles of Distilled Policies Improve Generalisation in Reinforcement Learning 

**Title (ZH)**: 蒸馏策略 ensemble 如何提高强化学习中的泛化能力 

**Authors**: Max Weltevrede, Moritz A. Zanger, Matthijs T.J. Spaan, Wendelin Böhmer  

**Link**: [PDF](https://arxiv.org/pdf/2505.16581)  

**Abstract**: In the zero-shot policy transfer setting in reinforcement learning, the goal is to train an agent on a fixed set of training environments so that it can generalise to similar, but unseen, testing environments. Previous work has shown that policy distillation after training can sometimes produce a policy that outperforms the original in the testing environments. However, it is not yet entirely clear why that is, or what data should be used to distil the policy. In this paper, we prove, under certain assumptions, a generalisation bound for policy distillation after training. The theory provides two practical insights: for improved generalisation, you should 1) train an ensemble of distilled policies, and 2) distil it on as much data from the training environments as possible. We empirically verify that these insights hold in more general settings, when the assumptions required for the theory no longer hold. Finally, we demonstrate that an ensemble of policies distilled on a diverse dataset can generalise significantly better than the original agent. 

**Abstract (ZH)**: 在强化学习的零样本策略迁移设置中，目标是通过对一组固定训练环境进行训练，使智能体能够在类似但未见过的测试环境中泛化。先前的研究表明，训练后进行策略蒸馏有时会产生在测试环境中表现优于原始策略的策略。然而，尚不清楚为何会出现这种情况，或应该使用什么数据进行策略蒸馏。在本文中，我们在某些假设下证明了训练后进行策略蒸馏的一般泛化界。该理论提供了两个实用的见解：为了提高泛化能力，应该1) 训练多个蒸馏策略的集合，并且2) 尽可能使用更多来自训练环境的数据进行策略蒸馏。我们实验证明了这些见解在更一般的情况下仍然成立，即理论所需的假设不再成立时也是如此。最后，我们证明了一个基于多样数据集进行策略蒸馏的策略集合可以在泛化能力上显著优于原始智能体。 

---
# From Local Patterns to Global Understanding: Cross-Stock Trend Integration for Enhanced Predictive Modeling 

**Title (ZH)**: 从局部模式到全局理解：跨股票趋势集成以增强预测建模 

**Authors**: Yi Hu, Hanchi Ren, Jingjing Deng, Xianghua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.16573)  

**Abstract**: Stock price prediction is a critical area of financial forecasting, traditionally approached by training models using the historical price data of individual stocks. While these models effectively capture single-stock patterns, they fail to leverage potential correlations among stock trends, which could improve predictive performance. Current single-stock learning methods are thus limited in their ability to provide a broader understanding of price dynamics across multiple stocks. To address this, we propose a novel method that merges local patterns into a global understanding through cross-stock pattern integration. Our strategy is inspired by Federated Learning (FL), a paradigm designed for decentralized model training. FL enables collaborative learning across distributed datasets without sharing raw data, facilitating the aggregation of global insights while preserving data privacy. In our adaptation, we train models on individual stock data and iteratively merge them to create a unified global model. This global model is subsequently fine-tuned on specific stock data to retain local relevance. The proposed strategy enables parallel training of individual stock models, facilitating efficient utilization of computational resources and reducing overall training time. We conducted extensive experiments to evaluate the proposed method, demonstrating that it outperforms benchmark models and enhances the predictive capabilities of state-of-the-art approaches. Our results highlight the efficacy of Cross-Stock Trend Integration (CSTI) in advancing stock price prediction, offering a robust alternative to traditional single-stock learning methodologies. 

**Abstract (ZH)**: 跨股票趋势集成在股票价格预测中的应用研究 

---
# Computing Exact Shapley Values in Polynomial Time for Product-Kernel Methods 

**Title (ZH)**: 在多项式时间内为产品核方法计算精确的沙普ley值 

**Authors**: Majid Mohammadi, Siu Lun Chau, Krikamol Muandet  

**Link**: [PDF](https://arxiv.org/pdf/2505.16516)  

**Abstract**: Kernel methods are widely used in machine learning due to their flexibility and expressive power. However, their black-box nature poses significant challenges to interpretability, limiting their adoption in high-stakes applications. Shapley value-based feature attribution techniques, such as SHAP and kernel-specific variants like RKHS-SHAP, offer a promising path toward explainability. Yet, computing exact Shapley values remains computationally intractable in general, motivating the development of various approximation schemes. In this work, we introduce PKeX-Shapley, a novel algorithm that utilizes the multiplicative structure of product kernels to enable the exact computation of Shapley values in polynomial time. We show that product-kernel models admit a functional decomposition that allows for a recursive formulation of Shapley values. This decomposition not only yields computational efficiency but also enhances interpretability in kernel-based learning. We also demonstrate how our framework can be generalized to explain kernel-based statistical discrepancies such as the Maximum Mean Discrepancy (MMD) and the Hilbert-Schmidt Independence Criterion (HSIC), thus offering new tools for interpretable statistical inference. 

**Abstract (ZH)**: 基于核函数的PKeX-Shapley算法：利用乘法结构实现多项式时间精确计算Shapley值，及其在内核基于学习可解释性和统计差异解释中的应用 

---
# Sparse Activation Editing for Reliable Instruction Following in Narratives 

**Title (ZH)**: 稀疏激活编辑以实现叙事中可靠的指令跟随 

**Authors**: Runcong Zhao, Chengyu Cao, Qinglin Zhu, Xiucheng Lv, Shun Shao, Lin Gui, Ruifeng Xu, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2505.16505)  

**Abstract**: Complex narrative contexts often challenge language models' ability to follow instructions, and existing benchmarks fail to capture these difficulties. To address this, we propose Concise-SAE, a training-free framework that improves instruction following by identifying and editing instruction-relevant neurons using only natural language instructions, without requiring labelled data. To thoroughly evaluate our method, we introduce FreeInstruct, a diverse and realistic benchmark of 1,212 examples that highlights the challenges of instruction following in narrative-rich settings. While initially motivated by complex narratives, Concise-SAE demonstrates state-of-the-art instruction adherence across varied tasks without compromising generation quality. 

**Abstract (ZH)**: 复杂叙事背景往往挑战语言模型遵循指令的能力，现有基准也未能捕捉到这些困难。为此，我们提出了一种无需训练的Concise-SAE框架，通过仅使用自然语言指令来识别和编辑指令相关的神经元，从而改进指令遵循能力，无需标注数据。为了全面评估我们的方法，我们引入了FreeInstruct，这是一个包含1,212个例子的多样化且现实的基准，突出了叙事丰富环境下指令遵循的挑战。尽管最初是针对复杂叙事提出，但Concise-SAE在各种任务中均表现出色，同时保持了生成质量。 

---
# Conf-GNNRec: Quantifying and Calibrating the Prediction Confidence for GNN-based Recommendation Methods 

**Title (ZH)**: Conf-GNNRec: 量化和校准基于GNN的推荐方法的预测置信度 

**Authors**: Meng Yan, Cai Xu, Xujing Wang, Ziyu Guan, Wei Zhao, Yuhang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.16466)  

**Abstract**: Recommender systems based on graph neural networks perform well in tasks such as rating and ranking. However, in real-world recommendation scenarios, noise such as user misuse and malicious advertisement gradually accumulates through the message propagation mechanism. Even if existing studies mitigate their effects by reducing the noise propagation weights, the severe sparsity of the recommender system still leads to the low-weighted noisy neighbors being mistaken as meaningful information, and the prediction result obtained based on the polluted nodes is not entirely trustworthy. Therefore, it is crucial to measure the confidence of the prediction results in this highly noisy framework. Furthermore, our evaluation of the existing representative GNN-based recommendation shows that it suffers from overconfidence. Based on the above considerations, we propose a new method to quantify and calibrate the prediction confidence of GNN-based recommendations (Conf-GNNRec). Specifically, we propose a rating calibration method that dynamically adjusts excessive ratings to mitigate overconfidence based on user personalization. We also design a confidence loss function to reduce the overconfidence of negative samples and effectively improve recommendation performance. Experiments on public datasets demonstrate the validity of Conf-GNNRec in prediction confidence and recommendation performance. 

**Abstract (ZH)**: 基于图神经网络的推荐系统在评级和排名任务中表现良好。然而，在实际推荐场景中，通过消息传播机制逐渐积累的用户滥用、恶意广告等噪声导致推荐系统稀疏性严重，低权重的噪声邻居被误认为有意义的信息，基于污染节点的预测结果可信度不高。因此，在这种高度噪声的框架中衡量预测结果的信心至关重要。进一步的评估显示，现有的基于图神经网络的代表性推荐方法存在过度自信的问题。基于以上考虑，我们提出了一种新的方法来量化和校准基于图神经网络的推荐预测置信度（Conf-GNNRec）。具体地，我们提出了一个评分校准方法，基于用户个性化动态调整过高的评分以缓解过度自信。我们还设计了一种置信度损失函数，以减少负样本的过度自信并有效提高推荐性能。在公共数据集上的实验验证了Conf-GNNRec在预测置信度和推荐性能上的有效性。 

---
# University of Indonesia at SemEval-2025 Task 11: Evaluating State-of-the-Art Encoders for Multi-Label Emotion Detection 

**Title (ZH)**: 印度尼西亚大学在SemEval-2025任务11中的研究：评估先进编码器在多标签情感检测中的性能 

**Authors**: Ikhlasul Akmal Hanif, Eryawan Presma Yulianrifat, Jaycent Gunawan Ongris, Eduardus Tjitrahardja, Muhammad Falensi Azmi, Rahmat Bryan Naufal, Alfan Farizki Wicaksono  

**Link**: [PDF](https://arxiv.org/pdf/2505.16460)  

**Abstract**: This paper presents our approach for SemEval 2025 Task 11 Track A, focusing on multilabel emotion classification across 28 languages. We explore two main strategies: fully fine-tuning transformer models and classifier-only training, evaluating different settings such as fine-tuning strategies, model architectures, loss functions, encoders, and classifiers. Our findings suggest that training a classifier on top of prompt-based encoders such as mE5 and BGE yields significantly better results than fully fine-tuning XLMR and mBERT. Our best-performing model on the final leaderboard is an ensemble combining multiple BGE models, where CatBoost serves as the classifier, with different configurations. This ensemble achieves an average F1-macro score of 56.58 across all languages. 

**Abstract (ZH)**: 本文提出了我们针对SemEval 2025 Task 11 Track A的方法，集中于28种语言上的多标签情感分类。我们探索了两种主要策略：完全微调变换器模型和仅训练分类器，并评估了不同的设置，如微调策略、模型架构、损失函数、编码器和分类器。我们的研究发现，在基于提示的编码器（如mE5和BGE）上训练分类器的效果显著优于完全微调XLMR和mBERT。最终排行榜上性能最佳的模型是多个BGE模型的集成，其中CatBoost用作分类器，并采用不同的配置，该集成在所有语言上的平均F1-宏分值为56.58。 

---
# AutoMCQ -- Automatically Generate Code Comprehension Questions using GenAI 

**Title (ZH)**: AutoMCQ —— 使用生成式人工智能自动生成代码理解问题 

**Authors**: Martin Goodfellow, Robbie Booth, Andrew Fagan, Alasdair Lambert  

**Link**: [PDF](https://arxiv.org/pdf/2505.16430)  

**Abstract**: Students often do not fully understand the code they have written. This sometimes does not become evident until later in their education, which can mean it is harder to fix their incorrect knowledge or misunderstandings. In addition, being able to fully understand code is increasingly important in a world where students have access to generative artificial intelligence (GenAI) tools, such as GitHub Copilot. One effective solution is to utilise code comprehension questions, where a marker asks questions about a submission to gauge understanding, this can also have the side effect of helping to detect plagiarism. However, this approach is time consuming and can be difficult and/or expensive to scale. This paper introduces AutoMCQ, which uses GenAI for the automatic generation of multiple-choice code comprehension questions. This is integrated with the CodeRunner automated assessment platform. 

**Abstract (ZH)**: 学生往往不能完全理解他们编写的代码。这种情况有时直到教育的后期才变得明显，这可能导致更难纠正他们的错误知识或误解。此外，在学生可以获取生成式人工智能（GenAI）工具（如GitHub Copilot）的世界中，完全理解代码的能力变得越来越重要。一种有效的方法是利用代码理解问题，评分者通过提出关于提交的问题来评估理解程度，这还可以作为检测抄袭的侧面效果。然而，这种方法耗时且难以或成本高昂地扩展。本文介绍了AutoMCQ，这是一种利用GenAI自动生成多项选择代码理解问题的方法，并将其与CodeRunner自动评估平台集成。 

---
# AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning 

**Title (ZH)**: AceReason-Nemotron：通过强化学习推动数学与代码推理发展 

**Authors**: Yang Chen, Zhuolin Yang, Zihan Liu, Chankyu Lee, Peng Xu, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2505.16400)  

**Abstract**: Despite recent progress in large-scale reinforcement learning (RL) for reasoning, the training recipe for building high-performing reasoning models remains elusive. Key implementation details of frontier models, such as DeepSeek-R1, including data curation strategies and RL training recipe, are often omitted. Moreover, recent research indicates distillation remains more effective than RL for smaller models. In this work, we demonstrate that large-scale RL can significantly enhance the reasoning capabilities of strong, small- and mid-sized models, achieving results that surpass those of state-of-the-art distillation-based models. We systematically study the RL training process through extensive ablations and propose a simple yet effective approach: first training on math-only prompts, then on code-only prompts. Notably, we find that math-only RL not only significantly enhances the performance of strong distilled models on math benchmarks (e.g., +14.6% / +17.2% on AIME 2025 for the 7B / 14B models), but also code reasoning tasks (e.g., +6.8% / +5.8% on LiveCodeBench for the 7B / 14B models). In addition, extended code-only RL iterations further improve performance on code benchmarks with minimal or no degradation in math results. We develop a robust data curation pipeline to collect challenging prompts with high-quality, verifiable answers and test cases to enable verification-based RL across both domains. Finally, we identify key experimental insights, including curriculum learning with progressively increasing response lengths and the stabilizing effect of on-policy parameter updates. We find that RL not only elicits the foundational reasoning capabilities acquired during pretraining and supervised fine-tuning (e.g., distillation), but also pushes the limits of the model's reasoning ability, enabling it to solve problems that were previously unsolvable. 

**Abstract (ZH)**: 尽管在大规模强化学习（RL）推理方面取得了最近的进步，但构建高性能推理模型的训练方法仍不清楚。前沿模型如DeepSeek-R1的关键实现细节，包括数据整理策略和RL训练方法，往往被省略。此外，近期研究表明，蒸馏在小模型上比RL更有效。在这项工作中，我们证明了大规模RL可以显著提升强大、小型和中型模型的推理能力，实现超越最先进的基于蒸馏模型的结果。我们通过广泛的消融研究系统地研究了RL训练过程，并提出了一种简单有效的方法：首先在纯数学提示上训练，然后在纯代码提示上训练。值得注意的是，我们发现纯数学RL不仅显著增强了强大蒸馏模型在数学基准测试中的性能（例如，7B和14B模型在AIME 2025上的性能分别提高14.6%和17.2%），还增强了代码推理任务（例如，在LiveCodeBench上，7B和14B模型分别提高6.8%和5.8%）。此外，扩展的纯代码RL循环将进一步提高代码基准测试中的性能，而对数学结果几乎没有或没有退化。我们开发了一种稳健的数据整理管道，收集具有高质量且可验证答案和测试案例的挑战性提示，以在两个领域实现基于验证的RL。最后，我们确定了一些关键的实验洞见，包括逐步增加回复长度的课程学习和在线策略参数更新的稳定效果。我们发现，RL不仅提取了预训练和监督微调（例如，蒸馏）期间获得的基本推理能力，而且推动了模型推理能力的极限，使其能够解决以前无法解决的问题。 

---
# Materials Generation in the Era of Artificial Intelligence: A Comprehensive Survey 

**Title (ZH)**: 人工智能时代材料生成：一项全面综述 

**Authors**: Zhixun Li, Bin Cao, Rui Jiao, Liang Wang, Ding Wang, Yang Liu, Dingshuo Chen, Jia Li, Qiang Liu, Yu Rong, Liang Wang, Tong-yi Zhang, Jeffrey Xu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16379)  

**Abstract**: Materials are the foundation of modern society, underpinning advancements in energy, electronics, healthcare, transportation, and infrastructure. The ability to discover and design new materials with tailored properties is critical to solving some of the most pressing global challenges. In recent years, the growing availability of high-quality materials data combined with rapid advances in Artificial Intelligence (AI) has opened new opportunities for accelerating materials discovery. Data-driven generative models provide a powerful tool for materials design by directly create novel materials that satisfy predefined property requirements. Despite the proliferation of related work, there remains a notable lack of up-to-date and systematic surveys in this area. To fill this gap, this paper provides a comprehensive overview of recent progress in AI-driven materials generation. We first organize various types of materials and illustrate multiple representations of crystalline materials. We then provide a detailed summary and taxonomy of current AI-driven materials generation approaches. Furthermore, we discuss the common evaluation metrics and summarize open-source codes and benchmark datasets. Finally, we conclude with potential future directions and challenges in this fast-growing field. The related sources can be found at this https URL. 

**Abstract (ZH)**: 材料是现代社会的基础，支撑着能源、电子、医疗、交通和基础设施等方面的发展。发现和设计具有定制性质的新材料的能力对于解决一些最紧迫的全球挑战至关重要。近年来，高质量材料数据的日益增多与人工智能（AI）的飞速进步相结合，为加速材料发现开辟了新的机会。数据驱动的生成模型为材料设计提供了一种强大工具，可以直接生成满足预定义性质要求的新材料。尽管相关工作日益增多，但在该领域仍缺乏最新的系统性综述。为填补这一空白，本文提供了AI驱动材料生成最近进展的全面概述。我们首先整理各类材料并展示多种晶体材料表现形式，然后详细总结并分类当前的AI驱动材料生成方法。此外，讨论常见的评估指标，并总结开源代码和基准数据集。最后，展望该快速发展的领域中的潜在未来方向和挑战。相关参考资料可访问此链接：this https URL。 

---
# A collaborative constrained graph diffusion model for the generation of realistic synthetic molecules 

**Title (ZH)**: 一种生成逼真合成分子的协作约束图扩散模型 

**Authors**: Manuel Ruiz-Botella, Marta Sales-Pardo, Roger Guimerà  

**Link**: [PDF](https://arxiv.org/pdf/2505.16365)  

**Abstract**: Developing new molecular compounds is crucial to address pressing challenges, from health to environmental sustainability. However, exploring the molecular space to discover new molecules is difficult due to the vastness of the space. Here we introduce CoCoGraph, a collaborative and constrained graph diffusion model capable of generating molecules that are guaranteed to be chemically valid. Thanks to the constraints built into the model and to the collaborative mechanism, CoCoGraph outperforms state-of-the-art approaches on standard benchmarks while requiring up to an order of magnitude fewer parameters. Analysis of 36 chemical properties also demonstrates that CoCoGraph generates molecules with distributions more closely matching real molecules than current models. Leveraging the model's efficiency, we created a database of 8.2M million synthetically generated molecules and conducted a Turing-like test with organic chemistry experts to further assess the plausibility of the generated molecules, and potential biases and limitations of CoCoGraph. 

**Abstract (ZH)**: 开发新的分子化合物对于应对健康和环境可持续等紧迫挑战至关重要。然而，探索分子空间以发现新分子具有挑战性，因为分子空间极其庞大。我们介绍了CoCoGraph，这是一种协作和受限的图扩散模型，能够生成化学上有效的分子。得益于模型中的约束条件和协作机制，CoCoGraph在标准基准测试上优于当前最先进的方法，同时所需的参数数量最多减少了一个数量级。对36种化学性质的分析还表明，CoCoGraph生成的分子的分布更接近真实分子。利用模型的效率，我们创建了一个包含820万 synthetically 生成分子的数据库，并与有机化学专家进行了类似图灵测试，以进一步评估生成分子的可行性，以及 CoCoGraph 的潜在偏见和局限性。 

---
# Neuromorphic-based metaheuristics: A new generation of low power, low latency and small footprint optimization algorithms 

**Title (ZH)**: 基于神经形态的元启发式算法：新一代低功耗、低延迟和紧凑占用空间的优化算法 

**Authors**: El-ghazali Talbi  

**Link**: [PDF](https://arxiv.org/pdf/2505.16362)  

**Abstract**: Neuromorphic computing (NC) introduces a novel algorithmic paradigm representing a major shift from traditional digital computing of Von Neumann architectures. NC emulates or simulates the neural dynamics of brains in the form of Spiking Neural Networks (SNNs). Much of the research in NC has concentrated on machine learning applications and neuroscience simulations. This paper investigates the modelling and implementation of optimization algorithms and particularly metaheuristics using the NC paradigm as an alternative to Von Neumann architectures, leading to breakthroughs in solving optimization problems.
Neuromorphic-based metaheuristics (Nheuristics) are supposed to be characterized by low power, low latency and small footprint. Since NC systems are fundamentally different from conventional Von Neumann computers, several challenges are posed to the design and implementation of Nheuristics. A guideline based on a classification and critical analysis is conducted on the different families of metaheuristics and optimization problems they address. We also discuss future directions that need to be addressed to expand both the development and application of Nheuristics. 

**Abstract (ZH)**: 神经形态计算驱动的元启发式算法 

---
# Dysfluent WFST: A Framework for Zero-Shot Speech Dysfluency Transcription and Detection 

**Title (ZH)**: 零样本语音断流转写与检测框架：不流畅WFST 

**Authors**: Chenxu Guo, Jiachen Lian, Xuanru Zhou, Jinming Zhang, Shuhe Li, Zongli Ye, Hwi Joo Park, Anaisha Das, Zoe Ezzes, Jet Vonk, Brittany Morin, Rian Bogley, Lisa Wauters, Zachary Miller, Maria Gorno-Tempini, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2505.16351)  

**Abstract**: Automatic detection of speech dysfluency aids speech-language pathologists in efficient transcription of disordered speech, enhancing diagnostics and treatment planning. Traditional methods, often limited to classification, provide insufficient clinical insight, and text-independent models misclassify dysfluency, especially in context-dependent cases. This work introduces Dysfluent-WFST, a zero-shot decoder that simultaneously transcribes phonemes and detects dysfluency. Unlike previous models, Dysfluent-WFST operates with upstream encoders like WavLM and requires no additional training. It achieves state-of-the-art performance in both phonetic error rate and dysfluency detection on simulated and real speech data. Our approach is lightweight, interpretable, and effective, demonstrating that explicit modeling of pronunciation behavior in decoding, rather than complex architectures, is key to improving dysfluency processing systems. 

**Abstract (ZH)**: 自动检测语音不流畅现象有助于言语病理学家高效地转录障碍性言语，提升诊断和治疗规划。传统方法往往仅限于分类，提供的临床洞察不足，且文本无关模型在上下文依赖性病例中误分类不流畅现象。本研究引入了Dysfluent-WFST，这是一种零样本解码器，同时进行音素转录和不流畅检测。与以往模型不同，Dysfluent-WFST 使用如WavLM的上游编码器，无需额外训练即可实现最佳表现。我们的方法轻量、可解释且有效，表明在解码中明确建模发音行为而非复杂架构是提高不流畅检测系统性能的关键。 

---
# Is Quantum Optimization Ready? An Effort Towards Neural Network Compression using Adiabatic Quantum Computing 

**Title (ZH)**: 量子优化准备就绪了吗？基于腺联量子计算的神经网络压缩研究 

**Authors**: Zhehui Wanga, Benjamin Chen Ming Choonga, Tian Huang, Daniel Gerlinghoffa, Rick Siow Mong Goh, Cheng Liu, Tao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.16332)  

**Abstract**: Quantum optimization is the most mature quantum computing technology to date, providing a promising approach towards efficiently solving complex combinatorial problems. Methods such as adiabatic quantum computing (AQC) have been employed in recent years on important optimization problems across various domains. In deep learning, deep neural networks (DNN) have reached immense sizes to support new predictive capabilities. Optimization of large-scale models is critical for sustainable deployment, but becomes increasingly challenging with ever-growing model sizes and complexity. While quantum optimization is suitable for solving complex problems, its application to DNN optimization is not straightforward, requiring thorough reformulation for compatibility with commercially available quantum devices. In this work, we explore the potential of adopting AQC for fine-grained pruning-quantization of convolutional neural networks. We rework established heuristics to formulate model compression as a quadratic unconstrained binary optimization (QUBO) problem, and assess the solution space offered by commercial quantum annealing devices. Through our exploratory efforts of reformulation, we demonstrate that AQC can achieve effective compression of practical DNN models. Experiments demonstrate that adiabatic quantum computing (AQC) not only outperforms classical algorithms like genetic algorithms and reinforcement learning in terms of time efficiency but also excels at identifying global optima. 

**Abstract (ZH)**: 量子优化是目前最成熟的量子计算技术，提供了高效解决复杂组合问题的有望途径。近年来，通过绝热量子计算（AQC）等方法在不同领域的重要优化问题上得到了应用。在深度学习中，深度神经网络（DNN）已达到巨大规模以支持新的预测能力。大规模模型的优化对于可持续部署至关重要，但随着模型规模和复杂性的不断增加，这一过程越来越具挑战性。虽然量子优化适合解决复杂问题，但将其应用于DNN优化并不简单，需要彻底改革以适应现有的商用量子设备。在本工作中，我们探讨了采用绝热量子计算（AQC）进行卷积神经网络精细剪枝-量化的方法。我们将现有的启发式方法重新构建成二次无约束二元优化（QUBO）问题，并评估商用量子退火设备提供的解空间。通过重新构形的努力，我们证明AQC能够有效压缩实际的DNN模型。实验表明，绝热量子计算（AQC）不仅在时间效率上优于遗传算法和强化学习等经典算法，且在识别全局最优解方面也表现出色。 

---
# SC4ANM: Identifying Optimal Section Combinations for Automated Novelty Prediction in Academic Papers 

**Title (ZH)**: SC4ANM: 识别学术论文新颖性预测自动化中的最优段落组合 

**Authors**: Wenqing Wu, Chengzhi Zhang, Tong Bao, Yi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16330)  

**Abstract**: Novelty is a core component of academic papers, and there are multiple perspectives on the assessment of novelty. Existing methods often focus on word or entity combinations, which provide limited insights. The content related to a paper's novelty is typically distributed across different core sections, e.g., Introduction, Methodology and Results. Therefore, exploring the optimal combination of sections for evaluating the novelty of a paper is important for advancing automated novelty assessment. In this paper, we utilize different combinations of sections from academic papers as inputs to drive language models to predict novelty scores. We then analyze the results to determine the optimal section combinations for novelty score prediction. We first employ natural language processing techniques to identify the sectional structure of academic papers, categorizing them into introduction, methods, results, and discussion (IMRaD). Subsequently, we used different combinations of these sections (e.g., introduction and methods) as inputs for pretrained language models (PLMs) and large language models (LLMs), employing novelty scores provided by human expert reviewers as ground truth labels to obtain prediction results. The results indicate that using introduction, results and discussion is most appropriate for assessing the novelty of a paper, while the use of the entire text does not yield significant results. Furthermore, based on the results of the PLMs and LLMs, the introduction and results appear to be the most important section for the task of novelty score prediction. The code and dataset for this paper can be accessed at this https URL. 

**Abstract (ZH)**: 新颖性是学术论文的核心组成部分，对于新颖性的评估有多重视角。现有方法通常侧重于单词或实体组合，提供了有限的洞察。论文新颖性的相关内容通常分布在不同的核心部分，如引言、方法和结果。因此，探索用于评估论文新颖性的最佳部分组合对于推动自动新颖性评估的发展至关重要。本文利用学术论文的不同部分组合作为输入，驱动语言模型预测新颖性得分。随后，我们分析结果以确定用于新颖性得分预测的最佳部分组合。我们首先采用自然语言处理技术识别学术论文的结构性质，将其分为引言、方法、结果和讨论（IMRaD）。随后，我们使用这些部分的不同组合（例如，引言和方法）作为预训练语言模型（PLMs）和大型语言模型（LLMs）的输入，使用人类专家评审员提供的新颖性得分作为真实标签，以获取预测结果。结果表明，使用引言、结果和讨论更适合评估论文的新颖性，而使用整个文本则未得到显著的预测结果。此外，基于PLMs和LLMs的结果，引言和结果似乎是新颖性得分预测任务中最重要部分。本文的代码和数据集可在以下网址访问。 

---
# CLEAR: A Clinically-Grounded Tabular Framework for Radiology Report Evaluation 

**Title (ZH)**: CLEAR：一种基于临床的表格式放射报告评估框架 

**Authors**: Yuyang Jiang, Chacha Chen, Shengyuan Wang, Feng Li, Zecong Tang, Benjamin M. Mervak, Lydia Chelala, Christopher M Straus, Reve Chahine, Samuel G. Armato III, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16325)  

**Abstract**: Existing metrics often lack the granularity and interpretability to capture nuanced clinical differences between candidate and ground-truth radiology reports, resulting in suboptimal evaluation. We introduce a Clinically-grounded tabular framework with Expert-curated labels and Attribute-level comparison for Radiology report evaluation (CLEAR). CLEAR not only examines whether a report can accurately identify the presence or absence of medical conditions, but also assesses whether it can precisely describe each positively identified condition across five key attributes: first occurrence, change, severity, descriptive location, and recommendation. Compared to prior works, CLEAR's multi-dimensional, attribute-level outputs enable a more comprehensive and clinically interpretable evaluation of report quality. Additionally, to measure the clinical alignment of CLEAR, we collaborate with five board-certified radiologists to develop CLEAR-Bench, a dataset of 100 chest X-ray reports from MIMIC-CXR, annotated across 6 curated attributes and 13 CheXpert conditions. Our experiments show that CLEAR achieves high accuracy in extracting clinical attributes and provides automated metrics that are strongly aligned with clinical judgment. 

**Abstract (ZH)**: 一种基于临床的表格框架：专家标注与属性级比较用于放射报告评估（CLEAR） 

---
# Layer-wise Investigation of Large-Scale Self-Supervised Music Representation Models 

**Title (ZH)**: 大规模自我监督音乐表示模型的分层探究 

**Authors**: Yizhi Zhou, Haina Zhu, Hangting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16306)  

**Abstract**: Recently, pre-trained models for music information retrieval based on self-supervised learning (SSL) are becoming popular, showing success in various downstream tasks. However, there is limited research on the specific meanings of the encoded information and their applicability. Exploring these aspects can help us better understand their capabilities and limitations, leading to more effective use in downstream tasks.
In this study, we analyze the advanced music representation model MusicFM and the newly emerged SSL model MuQ. We focus on three main aspects: (i) validating the advantages of SSL models across multiple downstream tasks, (ii) exploring the specialization of layer-wise information for different tasks, and (iii) comparing performance differences when selecting specific layers. Through this analysis, we reveal insights into the structure and potential applications of SSL models in music information retrieval. 

**Abstract (ZH)**: 近年来，基于自监督学习（SSL）的音乐信息检索预训练模型逐渐流行，并在各种下游任务中取得了成功。然而，对编码信息的具体含义及其适用性的研究相对有限。探索这些方面有助于我们更深入地了解其能力和局限性，从而更有效地应用于下游任务。

在本研究中，我们分析了先进的音乐表示模型MusicFM和新兴的SSL模型MuQ。我们重点关注三个方面：（i）验证SSL模型在多种下游任务中的优势，（ii）探索不同任务中逐层信息的专业化特性，（iii）比较选择特定层时的性能差异。通过这些分析，我们揭示了SSL模型在音乐信息检索中的结构及其潜在应用。 

---
# Artificial Intelligence for Direct Prediction of Molecular Dynamics Across Chemical Space 

**Title (ZH)**: 人工智能直接预测化学空间中的分子动力学 

**Authors**: Fuchun Ge, Pavlo O. Dral  

**Link**: [PDF](https://arxiv.org/pdf/2505.16301)  

**Abstract**: Molecular dynamics (MD) is a powerful tool for exploring the behavior of atomistic systems, but its reliance on sequential numerical integration limits simulation efficiency. We present MDtrajNet-1, a foundational AI model that directly generates MD trajectories across chemical space, bypassing force calculations and integration. This approach accelerates simulations by up to two orders of magnitude compared to traditional MD, even those enhanced by machine-learning interatomic potentials. MDtrajNet-1 combines equivariant neural networks with a Transformer-based architecture to achieve strong accuracy and transferability in predicting long-time trajectories for both known and unseen systems. Remarkably, the errors of the trajectories generated by MDtrajNet-1 for various molecular systems are close to those of the conventional ab initio MD. The model's flexible design supports diverse application scenarios, including different statistical ensembles, boundary conditions, and interaction types. By overcoming the intrinsic speed barrier of conventional MD, MDtrajNet-1 opens new frontiers in efficient and scalable atomistic simulations. 

**Abstract (ZH)**: MDtrajNet-1：直接生成化学空间中MD轨迹的基stdbool模型 

---
# Dialogue in Resonance: An Interactive Music Piece for Piano and Real-Time Automatic Transcription System 

**Title (ZH)**: 共振中的对话：钢琴与实时自动 transcription 系统的互动音乐作品 

**Authors**: Hayeon Bang, Taegyun Kwon, Juhan Nam  

**Link**: [PDF](https://arxiv.org/pdf/2505.16259)  

**Abstract**: This paper presents <Dialogue in Resonance>, an interactive music piece for a human pianist and a computer-controlled piano that integrates real-time automatic music transcription into a score-driven framework. Unlike previous approaches that primarily focus on improvisation-based interactions, our work establishes a balanced framework that combines composed structure with dynamic interaction. Through real-time automatic transcription as its core mechanism, the computer interprets and responds to the human performer's input in real time, creating a musical dialogue that balances compositional intent with live interaction while incorporating elements of unpredictability. In this paper, we present the development process from composition to premiere performance, including technical implementation, rehearsal process, and performance considerations. 

**Abstract (ZH)**: 《共鸣中的对话：一种结合实时自动音乐转写的互动音乐作品》 

---
# Explain Less, Understand More: Jargon Detection via Personalized Parameter-Efficient Fine-tuning 

**Title (ZH)**: 解释更少，理解更多：个性化参数高效微调的领域术语检测 

**Authors**: Bohao Wu, Qingyun Wang, Yue Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.16227)  

**Abstract**: Personalizing jargon detection and explanation is essential for making technical documents accessible to readers with diverse disciplinary backgrounds. However, tailoring models to individual users typically requires substantial annotation efforts and computational resources due to user-specific finetuning. To address this, we present a systematic study of personalized jargon detection, focusing on methods that are both efficient and scalable for real-world deployment. We explore two personalization strategies: (1) lightweight fine-tuning using Low-Rank Adaptation (LoRA) on open-source models, and (2) personalized prompting, which tailors model behavior at inference time without retaining. To reflect realistic constraints, we also investigate hybrid approaches that combine limited annotated data with unsupervised user background signals. Our personalized LoRA model outperforms GPT-4 by 21.4% in F1 score and exceeds the best performing oracle baseline by 8.3%. Remarkably, our method achieves comparable performance using only 10% of the annotated training data, demonstrating its practicality for resource-constrained settings. Our study offers the first work to systematically explore efficient, low-resource personalization of jargon detection using open-source language models, offering a practical path toward scalable, user-adaptive NLP system. 

**Abstract (ZH)**: 个性化领域术语检测与解释对于使技术文档对多元化学科背景的阅读者更加 accessible 至关重要。然而，针对 individual 用户调整模型通常需要大量的注释努力和计算资源以进行用户特定的微调。为解决这一问题，我们进行了一项系统研究，专注于高效且可扩展的个性化领域术语检测方法，以满足实际部署需求。我们探索了两种个性化策略：（1）使用低秩适应（LoRA）对开源模型进行轻量级微调，以及（2）个性化提示，该方法在推理时调整模型行为而不进行保留。为了反映现实中的限制，我们还研究了结合有限标注数据与无监督用户背景信号的混合方法。我们的个性化 LoRA 模型在 F1 分数上比 GPT-4 高出 21.4%，并在标注训练数据上仅使用 10% 的情况下超越了最佳的 oracle 基准，8.3%。值得注意的是，我们的方法证明了在资源受限的环境中其实用性。我们的研究是首个系统探讨使用开源语言模型高效且低资源个性化领域术语检测的工作，提供了一条通向可扩展且用户自适应 NLP 系统的实用路径。 

---
# Using Echo-State Networks to Reproduce Rare Events in Chaotic Systems 

**Title (ZH)**: 使用回响状态网络重现混沌系统中的罕见事件 

**Authors**: Anton Erofeev, Balasubramanya T. Nadiga, Ilya Timofeyev  

**Link**: [PDF](https://arxiv.org/pdf/2505.16208)  

**Abstract**: We apply the Echo-State Networks to predict the time series and statistical properties of the competitive Lotka-Volterra model in the chaotic regime. In particular, we demonstrate that Echo-State Networks successfully learn the chaotic attractor of the competitive Lotka-Volterra model and reproduce histograms of dependent variables, including tails and rare events. We use the Generalized Extreme Value distribution to quantify the tail behavior. 

**Abstract (ZH)**: 我们应用回声状态网络预测竞争Lotka-Volterra模型在混沌区间的时间序列和统计特性。特别地，我们证明回声状态网络能够学习竞争Lotka-Volterra模型的混沌吸引子，并重现因变量的直方图，包括尾部和稀有事件。我们使用广义极值分布定量分析尾部行为。 

---
# Interpretable Machine Learning for Macro Alpha: A News Sentiment Case Study 

**Title (ZH)**: 可解释的机器学习在宏观经济alpha中的应用：基于新闻情绪的研究案例 

**Authors**: Yuke Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16136)  

**Abstract**: This study introduces an interpretable machine learning (ML) framework to extract macroeconomic alpha from global news sentiment. We process the Global Database of Events, Language, and Tone (GDELT) Project's worldwide news feed using FinBERT -- a Bidirectional Encoder Representations from Transformers (BERT) based model pretrained on finance-specific language -- to construct daily sentiment indices incorporating mean tone, dispersion, and event impact. These indices drive an XGBoost classifier, benchmarked against logistic regression, to predict next-day returns for EUR/USD, USD/JPY, and 10-year U.S. Treasury futures (ZN). Rigorous out-of-sample (OOS) backtesting (5-fold expanding-window cross-validation, OOS period: c. 2017-April 2025) demonstrates exceptional, cost-adjusted performance for the XGBoost strategy: Sharpe ratios achieve 5.87 (EUR/USD), 4.65 (USD/JPY), and 4.65 (Treasuries), with respective compound annual growth rates (CAGRs) exceeding 50% in Foreign Exchange (FX) and 22% in bonds. Shapley Additive Explanations (SHAP) affirm that sentiment dispersion and article impact are key predictive features. Our findings establish that integrating domain-specific Natural Language Processing (NLP) with interpretable ML offers a potent and explainable source of macro alpha. 

**Abstract (ZH)**: 本研究介绍了一种可解释的机器学习框架，用于从全球新闻情绪中提取宏观经济阿尔法。我们使用基于金融专用语言预训练的FinBERT模型处理GDELT项目的世界各地新闻Feed，构建包含平均情绪、波动性和事件影响的日度情绪指数。这些指数驱动了与逻辑回归基准对比的XGBoost分类器，用于预测EUR/USD、USD/JPY和10年期美国国债期货（ZN）的下一交易日回报。严格的外样本回测（5折扩展窗口交叉验证，外样本期约为2017年-2025年4月）表明，XGBoost策略具有出色的成本调整后表现：夏普比率分别为5.87（EUR/USD）、4.65（USD/JPY）和4.65（国债），相应的外汇（FX）和债券的复合年增长率分别超过50%和22%。Shapley添加解释（SHAP）证实情绪波动性和文章影响是关键的预测特征。研究结果表明，将领域特定的自然语言处理与可解释的机器学习结合使用，提供了一种强大的可解释的宏观经济阿尔法来源。 

---
# Scalable Graph Generative Modeling via Substructure Sequences 

**Title (ZH)**: 基于子结构序列的可扩展图生成建模 

**Authors**: Zehong Wang, Zheyuan Zhang, Tianyi Ma, Chuxu Zhang, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.16130)  

**Abstract**: Graph neural networks (GNNs) has been predominantly driven by message-passing, where node representations are iteratively updated via local neighborhood aggregation. Despite their success, message-passing suffers from fundamental limitations -- including constrained expressiveness, over-smoothing, over-squashing, and limited capacity to model long-range dependencies. These issues hinder scalability: increasing data size or model size often fails to yield improved performance, limiting the viability of GNNs as backbones for graph foundation models. In this work, we explore pathways beyond message-passing and introduce Generative Graph Pattern Machine (G$^2$PM), a generative Transformer pre-training framework for graphs. G$^2$PM represents graph instances (nodes, edges, or entire graphs) as sequences of substructures, and employs generative pre-training over the sequences to learn generalizable, transferable representations. Empirically, G$^2$PM demonstrates strong scalability: on the ogbn-arxiv benchmark, it continues to improve with model sizes up to 60M parameters, outperforming prior generative approaches that plateau at significantly smaller scales (e.g., 3M). In addition, we systematically analyze the model design space, highlighting key architectural choices that contribute to its scalability and generalization. Across diverse tasks -- including node classification, graph classification, and transfer learning -- G$^2$PM consistently outperforms strong baselines, establishing a compelling foundation for scalable graph learning. The code and dataset are available at this https URL. 

**Abstract (ZH)**: 图神经网络（GNNs）主要依赖于消息传递，其中节点表示通过局部邻域聚合迭代更新。尽管取得了成功，但消息传递存在根本局限性，包括表达能力受限、过度平滑、过度挤压，以及建模长距离依赖关系的能力有限。这些问题阻碍了扩展性：数据量或模型规模的增加往往未能提升性能，限制了GNNs作为图基础模型骨干网络的可行性。在本文中，我们探索了超越消息传递的途径，并引入了生成图模式机器（G$^2$PM），这是一种针对图的生成Transformer预训练框架。G$^2$PM将图实例（节点、边或整个图）表示为子结构序列，并通过序列的生成预训练来学习可泛化的、可迁移的表示。实验证明，G$^2$PM具有强大的扩展性：在ogbn-arxiv基准上，其性能随着模型规模达到60M参数时仍然不断提升，超越了在较小规模（如3M）时就达到饱和的先前生成方法。此外，我们系统地分析了模型设计空间，强调了关键架构选择对其实现扩展性和泛化的重要贡献。在包括节点分类、图分类和迁移学习等多种任务中，G$^2$PM始终优于强基线，为可扩展图学习奠定了令人信服的基础。代码和数据集可在以下链接获取。 

---
# Towards Trustworthy Keylogger detection: A Comprehensive Analysis of Ensemble Techniques and Feature Selections through Explainable AI 

**Title (ZH)**: 基于可解释人工智能的集成技术与特征选择综合分析：可信赖的按键记录器检测研究 

**Authors**: Monirul Islam Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2505.16103)  

**Abstract**: Keylogger detection involves monitoring for unusual system behaviors such as delays between typing and character display, analyzing network traffic patterns for data exfiltration. In this study, we provide a comprehensive analysis for keylogger detection with traditional machine learning models - SVC, Random Forest, Decision Tree, XGBoost, AdaBoost, Logistic Regression and Naive Bayes and advanced ensemble methods including Stacking, Blending and Voting. Moreover, feature selection approaches such as Information gain, Lasso L1 and Fisher Score are thoroughly assessed to improve predictive performance and lower computational complexity. The Keylogger Detection dataset from publicly available Kaggle website is used in this project. In addition to accuracy-based classification, this study implements the approach for model interpretation using Explainable AI (XAI) techniques namely SHAP (Global) and LIME (Local) to deliver finer explanations for how much each feature contributes in assisting or hindering the detection process. To evaluate the models result, we have used AUC score, sensitivity, Specificity, Accuracy and F1 score. The best performance was achieved by AdaBoost with 99.76% accuracy, F1 score of 0.99, 100% precision, 98.6% recall, 1.0 specificity and 0.99 of AUC that is near-perfect classification with Fisher Score. 

**Abstract (ZH)**: 基于传统机器学习模型和高级集成方法的键盘记录器检测全面分析 

---
# Date Fragments: A Hidden Bottleneck of Tokenization for Temporal Reasoning 

**Title (ZH)**: 时间碎片：时间推理中词元化的一个隐藏瓶颈 

**Authors**: Gagan Bhatia, Maxime Peyrard, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16088)  

**Abstract**: Modern BPE tokenizers often split calendar dates into meaningless fragments, e.g., 20250312 $\rightarrow$ 202, 503, 12, inflating token counts and obscuring the inherent structure needed for robust temporal reasoning. In this work, we (1) introduce a simple yet interpretable metric, termed date fragmentation ratio, that measures how faithfully a tokenizer preserves multi-digit date components; (2) release DateAugBench, a suite of 6500 examples spanning three temporal reasoning tasks: context-based date resolution, format-invariance puzzles, and date arithmetic across historical, contemporary, and future regimes; and (3) through layer-wise probing and causal attention-hop analyses, uncover an emergent date-abstraction mechanism whereby large language models stitch together the fragments of month, day, and year components for temporal reasoning. Our experiments show that excessive fragmentation correlates with accuracy drops of up to 10 points on uncommon dates like historical and futuristic dates. Further, we find that the larger the model, the faster the emergent date abstraction that heals date fragments is accomplished. Lastly, we observe a reasoning path that LLMs follow to assemble date fragments, typically differing from human interpretation (year $\rightarrow$ month $\rightarrow$ day). 

**Abstract (ZH)**: 现代BPE分词器经常将日期拆分成无意义的片段，例如20250312 → 202, 503, 12，增加分词数量并遮蔽了用于稳健时间推理所需的基本结构。本文中，我们（1）提出了一种简单且可解释的指标，称为日期碎片化比率，用于衡量分词器忠实地保留多数字日期组件的程度；（2）发布DateAugBench，包含6500个示例，涵盖三种时间推理任务：基于上下文的日期解析、格式不变性难题以及跨越历史、当代和未来的日期运算；（3）通过逐层探针和因果注意力跳转分析，揭示了一个新兴的日期抽象机制，大型语言模型通过将月份、日期和年的片段拼接起来进行时间推理。我们的实验表明，过度的碎片化与在历史和未来日期上的准确性下降多达10个百分点相关。此外，我们发现模型越大，这种新兴的日期抽象机制修复日期片段的速度越快。最后，我们观察到LLM遵循的推理路径将日期片段组装起来，通常不同于人类的解释（年份 → 月份 → 日期）。 

---
# Bidirectional Variational Autoencoders 

**Title (ZH)**: 双向变分自编码器 

**Authors**: Bart Kosko, Olaoluwa Adigun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16074)  

**Abstract**: We present the new bidirectional variational autoencoder (BVAE) network architecture. The BVAE uses a single neural network both to encode and decode instead of an encoder-decoder network pair. The network encodes in the forward direction and decodes in the backward direction through the same synaptic web. Simulations compared BVAEs and ordinary VAEs on the four image tasks of image reconstruction, classification, interpolation, and generation. The image datasets included MNIST handwritten digits, Fashion-MNIST, CIFAR-10, and CelebA-64 face images. The bidirectional structure of BVAEs cut the parameter count by almost 50% and still slightly outperformed the unidirectional VAEs. 

**Abstract (ZH)**: 我们提出了一种新的双向变分自编码器（BVAE）网络架构。 

---
# Mesh-free sparse identification of nonlinear dynamics 

**Title (ZH)**: 无网格稀疏识别非线性动力学 

**Authors**: Mars Liyao Gao, J. Nathan Kutz, Bernat Font  

**Link**: [PDF](https://arxiv.org/pdf/2505.16058)  

**Abstract**: Identifying the governing equations of a dynamical system is one of the most important tasks for scientific modeling. However, this procedure often requires high-quality spatio-temporal data uniformly sampled on structured grids. In this paper, we propose mesh-free SINDy, a novel algorithm which leverages the power of neural network approximation as well as auto-differentiation to identify governing equations from arbitrary sensor placements and non-uniform temporal data sampling. We show that mesh-free SINDy is robust to high noise levels and limited data while remaining computationally efficient. In our implementation, the training procedure is straight-forward and nearly free of hyperparameter tuning, making mesh-free SINDy widely applicable to many scientific and engineering problems. In the experiments, we demonstrate its effectiveness on a series of PDEs including the Burgers' equation, the heat equation, the Korteweg-De Vries equation and the 2D advection-diffusion equation. We conduct detailed numerical experiments on all datasets, varying the noise levels and number of samples, and we also compare our approach to previous state-of-the-art methods. It is noteworthy that, even in high-noise and low-data scenarios, mesh-free SINDy demonstrates robust PDE discovery, achieving successful identification with up to 75% noise for the Burgers' equation using 5,000 samples and with as few as 100 samples and 1% noise. All of this is achieved within a training time of under one minute. 

**Abstract (ZH)**: 无网格式SINDy：一种利用神经网络逼近与自动微分识别动力系统 governing 方程的新型算法 

---
# Signals of Provenance: Practices & Challenges of Navigating Indicators in AI-Generated Media for Sighted and Blind Individuals 

**Title (ZH)**: 来源信号：视障与非视障个体导航AI生成媒体中指标的实践与挑战 

**Authors**: Ayae Ide, Tory Park, Jaron Mink, Tanusree Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2505.16057)  

**Abstract**: AI-Generated (AIG) content has become increasingly widespread by recent advances in generative models and the easy-to-use tools that have significantly lowered the technical barriers for producing highly realistic audio, images, and videos through simple natural language prompts. In response, platforms are adopting provable provenance with platforms recommending AIG to be self-disclosed and signaled to users. However, these indicators may be often missed, especially when they rely solely on visual cues and make them ineffective to users with different sensory abilities. To address the gap, we conducted semi-structured interviews (N=28) with 15 sighted and 13 BLV participants to examine their interaction with AIG content through self-disclosed AI indicators. Our findings reveal diverse mental models and practices, highlighting different strengths and weaknesses of content-based (e.g., title, description) and menu-aided (e.g., AI labels) indicators. While sighted participants leveraged visual and audio cues, BLV participants primarily relied on audio and existing assistive tools, limiting their ability to identify AIG. Across both groups, they frequently overlooked menu-aided indicators deployed by platforms and rather interacted with content-based indicators such as title and comments. We uncovered usability challenges stemming from inconsistent indicator placement, unclear metadata, and cognitive overload. These issues were especially critical for BLV individuals due to the insufficient accessibility of interface elements. We provide practical recommendations and design implications for future AIG indicators across several dimensions. 

**Abstract (ZH)**: 基于AI生成的内容逐渐普及：生成模型进步和易于使用的工具大幅降低了通过简单自然语言提示生产高度逼真音频、图像和视频的技术门槛。为应对这一变化，平台正在采用可验证的来源，并建议将AI生成内容进行自我披露并提示用户。然而，这些标识往往容易被忽视，尤其是在仅依赖视觉线索的情况下，这使得它们对具有不同感官能力的用户不够有效。为解决这一差距，我们对28名参与者进行了半结构化访谈（其中15名视力正常参与者和13名盲人参与者），研究他们如何通过自我披露的AI标识与基于内容的（如标题、描述）和菜单辅助的（如AI标签）标识互动。研究发现揭示了不同的心理模型和实践，突显了基于内容和菜单辅助标识的不同优势和劣势。视力正常参与者利用视觉和听觉线索，而盲人参与者主要依赖听觉和现有辅助工具，这限制了他们识别基于AI生成内容的能力。在两个群体中，他们经常忽视平台部署的菜单辅助标识，而是与基于内容的标识（如标题和评论）进行互动。我们发现了由不一致的标识位置、不清晰的元数据和认知过载引起的人机交互挑战。这些问题对盲人个体尤为重要，因为界面元素的不足可访问性加剧了这些问题。我们为未来基于AI生成内容标识提供了实用建议和设计启示。 

---
# Equivariant Eikonal Neural Networks: Grid-Free, Scalable Travel-Time Prediction on Homogeneous Spaces 

**Title (ZH)**: 不变电弹神经网络：均匀空间中无网格、可扩展的旅行时间预测 

**Authors**: Alejandro García-Castellanos, David R. Wessels, Nicky J. van den Berg, Remco Duits, Daniël M. Pelt, Erik J. Bekkers  

**Link**: [PDF](https://arxiv.org/pdf/2505.16035)  

**Abstract**: We introduce Equivariant Neural Eikonal Solvers, a novel framework that integrates Equivariant Neural Fields (ENFs) with Neural Eikonal Solvers. Our approach employs a single neural field where a unified shared backbone is conditioned on signal-specific latent variables - represented as point clouds in a Lie group - to model diverse Eikonal solutions. The ENF integration ensures equivariant mapping from these latent representations to the solution field, delivering three key benefits: enhanced representation efficiency through weight-sharing, robust geometric grounding, and solution steerability. This steerability allows transformations applied to the latent point cloud to induce predictable, geometrically meaningful modifications in the resulting Eikonal solution. By coupling these steerable representations with Physics-Informed Neural Networks (PINNs), our framework accurately models Eikonal travel-time solutions while generalizing to arbitrary Riemannian manifolds with regular group actions. This includes homogeneous spaces such as Euclidean, position-orientation, spherical, and hyperbolic manifolds. We validate our approach through applications in seismic travel-time modeling of 2D and 3D benchmark datasets. Experimental results demonstrate superior performance, scalability, adaptability, and user controllability compared to existing Neural Operator-based Eikonal solver methods. 

**Abstract (ZH)**: equivariant神经Eigen解算器：将equivariant神经场与神经Eigen解算器相结合的新型框架 

---
# Toward Theoretical Insights into Diffusion Trajectory Distillation via Operator Merging 

**Title (ZH)**: 向通过操作合并理论洞察扩散轨迹精炼的研究迈进 

**Authors**: Weiguo Gao, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16024)  

**Abstract**: Diffusion trajectory distillation methods aim to accelerate sampling in diffusion models, which produce high-quality outputs but suffer from slow sampling speeds. These methods train a student model to approximate the multi-step denoising process of a pretrained teacher model in a single step, enabling one-shot generation. However, theoretical insights into the trade-off between different distillation strategies and generative quality remain limited, complicating their optimization and selection. In this work, we take a first step toward addressing this gap. Specifically, we reinterpret trajectory distillation as an operator merging problem in the linear regime, where each step of the teacher model is represented as a linear operator acting on noisy data. These operators admit a clear geometric interpretation as projections and rescalings corresponding to the noise schedule. During merging, signal shrinkage occurs as a convex combination of operators, arising from both discretization and limited optimization time of the student model. We propose a dynamic programming algorithm to compute the optimal merging strategy that maximally preserves signal fidelity. Additionally, we demonstrate the existence of a sharp phase transition in the optimal strategy, governed by data covariance structures. Our findings enhance the theoretical understanding of diffusion trajectory distillation and offer practical insights for improving distillation strategies. 

**Abstract (ZH)**: 扩散轨迹蒸馏方法旨在加速扩散模型的采样过程，尽管这些模型能够生成高质量的输出，但采样速度较慢。这些方法通过训练一个学生模型在单一步骤中近似预训练教师模型的多步去噪过程，从而实现一次性生成。然而，不同蒸馏策略与生成质量之间的权衡关系的理论见解仍然有限，这给优化和选择带来了复杂性。在本文中，我们朝着填补这一空白迈出了第一步。具体而言，我们将轨迹蒸馏重新解释为线性域中的算子合并问题，其中教师模型的每一步都被表示为作用于噪声数据的线性算子。这些算子具有明确的几何解释，对应于与噪声分配相关的投影和缩放操作。在合并过程中，信号收缩由于算子的凸组合、离散化以及学生模型有限的优化时间而发生。我们提出了一种动态规划算法来计算最大化保留信号保真度的最优合并策略。此外，我们展示了最优策略中存在的锐利相变现象，由数据协方差结构控制。我们的发现增强了对扩散轨迹蒸馏理论的理解，并为改进蒸馏策略提供了实用见解。 

---
# LAGO: Few-shot Crosslingual Embedding Inversion Attacks via Language Similarity-Aware Graph Optimization 

**Title (ZH)**: LAGO: 语言相似性 Awareness 图优化下的少样本跨语言嵌入反向攻击 

**Authors**: Wenrui Yu, Yiyi Chen, Johannes Bjerva, Sokol Kosta, Qiongxiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16008)  

**Abstract**: We propose LAGO - Language Similarity-Aware Graph Optimization - a novel approach for few-shot cross-lingual embedding inversion attacks, addressing critical privacy vulnerabilities in multilingual NLP systems. Unlike prior work in embedding inversion attacks that treat languages independently, LAGO explicitly models linguistic relationships through a graph-based constrained distributed optimization framework. By integrating syntactic and lexical similarity as edge constraints, our method enables collaborative parameter learning across related languages. Theoretically, we show this formulation generalizes prior approaches, such as ALGEN, which emerges as a special case when similarity constraints are relaxed. Our framework uniquely combines Frobenius-norm regularization with linear inequality or total variation constraints, ensuring robust alignment of cross-lingual embedding spaces even with extremely limited data (as few as 10 samples per language). Extensive experiments across multiple languages and embedding models demonstrate that LAGO substantially improves the transferability of attacks with 10-20% increase in Rouge-L score over baselines. This work establishes language similarity as a critical factor in inversion attack transferability, urging renewed focus on language-aware privacy-preserving multilingual embeddings. 

**Abstract (ZH)**: 语言相似性感知图优化：一种新的 few-shot 跨语言嵌入反转攻击方法 

---
# VideoGameQA-Bench: Evaluating Vision-Language Models for Video Game Quality Assurance 

**Title (ZH)**: VideoGameQA-Bench: 评估视觉语言模型在视频游戏质量保证中的表现 

**Authors**: Mohammad Reza Taesiri, Abhijay Ghildyal, Saman Zadtootaghaj, Nabajeet Barman, Cor-Paul Bezemer  

**Link**: [PDF](https://arxiv.org/pdf/2505.15952)  

**Abstract**: With video games now generating the highest revenues in the entertainment industry, optimizing game development workflows has become essential for the sector's sustained growth. Recent advancements in Vision-Language Models (VLMs) offer considerable potential to automate and enhance various aspects of game development, particularly Quality Assurance (QA), which remains one of the industry's most labor-intensive processes with limited automation options. To accurately evaluate the performance of VLMs in video game QA tasks and determine their effectiveness in handling real-world scenarios, there is a clear need for standardized benchmarks, as existing benchmarks are insufficient to address the specific requirements of this domain. To bridge this gap, we introduce VideoGameQA-Bench, a comprehensive benchmark that covers a wide array of game QA activities, including visual unit testing, visual regression testing, needle-in-a-haystack tasks, glitch detection, and bug report generation for both images and videos of various games. Code and data are available at: this https URL 

**Abstract (ZH)**: 随着电子游戏现已成为娱乐行业的最高收入来源，优化游戏开发工作流程对于该领域持续增长变得至关重要。近期视觉-语言模型（VLMs）的进步为自动化和提升游戏开发的各个方面提供了巨大潜力，尤其是质量保证（QA），这是行业中最具劳动密集型且自动化选择有限的过程之一。为了准确评估VLMs在视频游戏QA任务中的性能并确定其在处理现实场景中的有效性，建立标准化基准显得尤为必要，现有基准不足以满足该领域的特定需求。为了填补这一空白，我们提出了VideoGameQA-Bench，这是一个全面的基准，涵盖了广泛的game QA活动，包括视觉单元测试、视觉回归测试、针锋相对的任务、故障检测以及针对不同游戏的图像和视频的bug报告生成。代码和数据可通过以下链接获得：this https URL。 

---
# MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding 

**Title (ZH)**: MoRE-Brain: 路由混合专家模型解释性跨被试fMRI视觉解码 

**Authors**: Yuxiang Wei, Yanteng Zhang, Xi Xiao, Tianyang Wang, Xiao Wang, Vince D. Calhoun  

**Link**: [PDF](https://arxiv.org/pdf/2505.15946)  

**Abstract**: Decoding visual experiences from fMRI offers a powerful avenue to understand human perception and develop advanced brain-computer interfaces. However, current progress often prioritizes maximizing reconstruction fidelity while overlooking interpretability, an essential aspect for deriving neuroscientific insight. To address this gap, we propose MoRE-Brain, a neuro-inspired framework designed for high-fidelity, adaptable, and interpretable visual reconstruction. MoRE-Brain uniquely employs a hierarchical Mixture-of-Experts architecture where distinct experts process fMRI signals from functionally related voxel groups, mimicking specialized brain networks. The experts are first trained to encode fMRI into the frozen CLIP space. A finetuned diffusion model then synthesizes images, guided by expert outputs through a novel dual-stage routing mechanism that dynamically weighs expert contributions across the diffusion process. MoRE-Brain offers three main advancements: First, it introduces a novel Mixture-of-Experts architecture grounded in brain network principles for neuro-decoding. Second, it achieves efficient cross-subject generalization by sharing core expert networks while adapting only subject-specific routers. Third, it provides enhanced mechanistic insight, as the explicit routing reveals precisely how different modeled brain regions shape the semantic and spatial attributes of the reconstructed image. Extensive experiments validate MoRE-Brain's high reconstruction fidelity, with bottleneck analyses further demonstrating its effective utilization of fMRI signals, distinguishing genuine neural decoding from over-reliance on generative priors. Consequently, MoRE-Brain marks a substantial advance towards more generalizable and interpretable fMRI-based visual decoding. Code will be publicly available soon: this https URL. 

**Abstract (ZH)**: 从fMRI解码视觉体验为理解人类感知和开发先进的脑机接口提供了强有力的途径。然而，当前进展往往侧重于最大限度地提高重建保真度，而忽略了可解释性，这是获取神经科学洞察所必需的一个方面。为了解决这一差距，我们提出了MoRE-Brain，一个受神经启发的框架，旨在实现高保真、灵活且可解释的视觉重建。MoRE-Brain独树一帜地采用了一种分层Mixture-of-Experts架构，其中不同的专家处理功能相关体素组的fMRI信号，模拟专门化的脑网络。首先，专家被训练将fMRI编码到冻结的CLIP空间中。然后，微调的扩散模型通过一种新颖的两阶段路由机制合成图像，该机制动态评估扩散过程中专家贡献的权重，该机制由专家输出指导。MoRE-Brain提供了三项主要进展：首先，它引入了一种基于脑网络原则的新颖Mixture-of-Experts架构，用于神经解码。其次，它通过共享核心专家网络并仅适应个体特异性路由器实现高效的跨被试泛化。第三，它提供了增强的机械性洞见，因为明确的路由揭示了哪些建模的大脑区域如何精确地塑造重建图像的语义和空间属性。大量实验验证了MoRE-Brain的高重建保真度，瓶颈分析进一步证明了其有效利用fMRI信号，区分真实的神经解码和过度依赖生成先验。因此，MoRE-Brain标志着更可泛化和可解释的基于fMRI的视觉解码的重要进步。代码即将公开：this https URL。 

---
# BR-TaxQA-R: A Dataset for Question Answering with References for Brazilian Personal Income Tax Law, including case law 

**Title (ZH)**: 巴西个人所得税法及案例法参考问答数据集：BR-TaxQA-R 

**Authors**: Juvenal Domingos Júnior, Augusto Faria, E. Seiti de Oliveira, Erick de Brito, Matheus Teotonio, Andre Assumpção, Diedre Carmo, Roberto Lotufo, Jayr Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2505.15916)  

**Abstract**: This paper presents BR-TaxQA-R, a novel dataset designed to support question answering with references in the context of Brazilian personal income tax law. The dataset contains 715 questions from the 2024 official Q\&A document published by Brazil's Internal Revenue Service, enriched with statutory norms and administrative rulings from the Conselho Administrativo de Recursos Fiscais (CARF). We implement a Retrieval-Augmented Generation (RAG) pipeline using OpenAI embeddings for searching and GPT-4o-mini for answer generation. We compare different text segmentation strategies and benchmark our system against commercial tools such as ChatGPT and this http URL using RAGAS-based metrics. Results show that our custom RAG pipeline outperforms commercial systems in Response Relevancy, indicating stronger alignment with user queries, while commercial models achieve higher scores in Factual Correctness and fluency. These findings highlight a trade-off between legally grounded generation and linguistic fluency. Crucially, we argue that human expert evaluation remains essential to ensure the legal validity of AI-generated answers in high-stakes domains such as taxation. BR-TaxQA-R is publicly available at this https URL. 

**Abstract (ZH)**: BR-TaxQA-R：一种支持巴西个人所得税法参考上下文中文本问答的新数据集 

---
# Last Layer Empirical Bayes 

**Title (ZH)**: 最后一层经验贝叶斯 

**Authors**: Valentin Villecroze, Yixin Wang, Gabriel Loaiza-Ganem  

**Link**: [PDF](https://arxiv.org/pdf/2505.15888)  

**Abstract**: The task of quantifying the inherent uncertainty associated with neural network predictions is a key challenge in artificial intelligence. Bayesian neural networks (BNNs) and deep ensembles are among the most prominent approaches to tackle this task. Both approaches produce predictions by computing an expectation of neural network outputs over some distribution on the corresponding weights; this distribution is given by the posterior in the case of BNNs, and by a mixture of point masses for ensembles. Inspired by recent work showing that the distribution used by ensembles can be understood as a posterior corresponding to a learned data-dependent prior, we propose last layer empirical Bayes (LLEB). LLEB instantiates a learnable prior as a normalizing flow, which is then trained to maximize the evidence lower bound; to retain tractability we use the flow only on the last layer. We show why LLEB is well motivated, and how it interpolates between standard BNNs and ensembles in terms of the strength of the prior that they use. LLEB performs on par with existing approaches, highlighting that empirical Bayes is a promising direction for future research in uncertainty quantification. 

**Abstract (ZH)**: 量化与神经网络预测相关的固有不确定性是人工智能中的一个关键挑战。贝叶斯神经网络（BNN）和深度集成是解决这一问题的最突出方法之一。受近期研究表明集成中使用的分布可以视为与数据相关的先验对应的后验的启发，我们提出了最后一层经验贝叶斯（LLEB）。LLEB将可学习的先验实例化为归一化流，并通过最大化证据下界来训练该流；为保持可计算性，仅在最后一层使用流。我们展示了LLEB为什么合理，并且如何在使用先验强度方面介于标准BNN和集成之间。LLEB的表现与现有方法相当，突显了经验贝叶斯在未来不确定性量化研究中的前景。 

---
# An Inclusive Foundation Model for Generalizable Cytogenetics in Precision Oncology 

**Title (ZH)**: 包容性基础模型在精准 Oncology 中的可泛化细胞遗传学 

**Authors**: Changchun Yang, Weiqian Dai, Yilan Zhang, Siyuan Chen, Jingdong Hu, Junkai Su, Yuxuan Chen, Ao Xu, Na Li, Xin Gao, Yongguo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15868)  

**Abstract**: Chromosome analysis is vital for diagnosing genetic disorders and guiding cancer therapy decisions through the identification of somatic clonal aberrations. However, developing an AI model are hindered by the overwhelming complexity and diversity of chromosomal abnormalities, requiring extensive annotation efforts, while automated methods remain task-specific and lack generalizability due to the scarcity of comprehensive datasets spanning diverse resource conditions. Here, we introduce CHROMA, a foundation model for cytogenomics, designed to overcome these challenges by learning generalizable representations of chromosomal abnormalities. Pre-trained on over 84,000 specimens (~4 million chromosomal images) via self-supervised learning, CHROMA outperforms other methods across all types of abnormalities, even when trained on fewer labelled data and more imbalanced datasets. By facilitating comprehensive mapping of instability and clonal leisons across various aberration types, CHROMA offers a scalable and generalizable solution for reliable and automated clinical analysis, reducing the annotation workload for experts and advancing precision oncology through the early detection of rare genomic abnormalities, enabling broad clinical AI applications and making advanced genomic analysis more accessible. 

**Abstract (ZH)**: 染色体分析对于诊断遗传疾病和通过识别体细胞克隆异常来指导癌症治疗决策至关重要。然而，由于染色体异常的复杂性和多样性，开发AI模型受到限制，需要大量的标注工作，而现有的自动化方法由于缺乏涵盖多样资源条件的全面数据集，仍然任务特定且缺乏泛化能力。在这里，我们介绍了CHROMA，一种用于细胞遗传学的基础模型，旨在通过学习染色体异常的一般表示来克服这些挑战。CHROMA通过自我监督学习在超过84,000个样本（约400万张染色体图像）上进行预训练，即使在训练数据较少和数据集更不平衡时，CHROMA也能够在各种类型的异常中取得最佳表现。通过促进各种异常类型不稳定性及克隆病灶的全面映射，CHROMA提供了一种可扩展且可泛化的临床分析解决方案，减轻了专家的标注工作量，并通过早期检测罕见的基因组异常推动精准 oncology，实现了广泛的临床AI应用，使高级基因组分析更具可及性。 

---
# AutoData: A Multi-Agent System for Open Web Data Collection 

**Title (ZH)**: AutoData：一种面向开放网络数据收集的多智能体系统 

**Authors**: Tianyi Ma, Yiyue Qian, Zheyuan Zhang, Zehong Wang, Xiaoye Qian, Feifan Bai, Yifan Ding, Xuwei Luo, Shinan Zhang, Keerthiram Murugesan, Chuxu Zhang, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.15859)  

**Abstract**: The exponential growth of data-driven systems and AI technologies has intensified the demand for high-quality web-sourced datasets. While existing datasets have proven valuable, conventional web data collection approaches face significant limitations in terms of human effort and scalability. Current data-collecting solutions fall into two categories: wrapper-based methods that struggle with adaptability and reproducibility, and large language model (LLM)-based approaches that incur substantial computational and financial costs. To address these challenges, we propose AutoData, a novel multi-agent system for Automated web Data collection, that requires minimal human intervention, i.e., only necessitating a natural language instruction specifying the desired dataset. In addition, AutoData is designed with a robust multi-agent architecture, featuring a novel oriented message hypergraph coordinated by a central task manager, to efficiently organize agents across research and development squads. Besides, we introduce a novel hypergraph cache system to advance the multi-agent collaboration process that enables efficient automated data collection and mitigates the token cost issues prevalent in existing LLM-based systems. Moreover, we introduce Instruct2DS, a new benchmark dataset supporting live data collection from web sources across three domains: academic, finance, and sports. Comprehensive evaluations over Instruct2DS and three existing benchmark datasets demonstrate AutoData's superior performance compared to baseline methods. Case studies on challenging tasks such as picture book collection and paper extraction from surveys further validate its applicability. Our source code and dataset are available at this https URL. 

**Abstract (ZH)**: 数据驱动系统和AI技术的指数增长加剧了对高质量网页数据集的需求。尽管现有数据集证明了其价值，但传统的网页数据收集方法在人力投入和可扩展性方面面临重大限制。当前的数据收集解决方案大致可分为两类：挣扎于适应性和可重复性的封装方法，以及带来显著计算和财务成本的大语言模型（LLM）方法。为解决这些挑战，我们提出AutoData，这是一种新型多代理系统，用于自动化网页数据收集，仅需少量的人工干预，即只需提供一个自然语言指令来指定所需的数据集。此外，AutoData采用了一种稳健的多代理架构，包括由中央任务管理器协调的新型导向消息超图，以高效地组织研究与发展团队中的代理。此外，我们引入了一种新的超图缓存系统，以促进多代理协作过程，实现高效自动化数据收集，并缓解现有基于LLM系统中的标记成本问题。我们还引入了Instruct2DS，这是一种新的基准数据集，支持从三个领域（学术、金融和体育）的网页源进行实时数据收集。AutoData在Instruct2DS和三个现有基准数据集上的全面评估表明其性能优于基线方法。在书籍图集收集和调查中论文提取等具有挑战性的任务上进行的案例研究进一步验证了其适用性。我们的源代码和数据集可在此网页中获取。 

---
# DisastIR: A Comprehensive Information Retrieval Benchmark for Disaster Management 

**Title (ZH)**: DisastIR：灾难管理综合信息检索基准 

**Authors**: Kai Yin, Xiangjue Dong, Chengkai Liu, Lipai Huang, Yiming Xiao, Zhewei Liu, Ali Mostafavi, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2505.15856)  

**Abstract**: Effective disaster management requires timely access to accurate and contextually relevant information. Existing Information Retrieval (IR) benchmarks, however, focus primarily on general or specialized domains, such as medicine or finance, neglecting the unique linguistic complexity and diverse information needs encountered in disaster management scenarios. To bridge this gap, we introduce DisastIR, the first comprehensive IR evaluation benchmark specifically tailored for disaster management. DisastIR comprises 9,600 diverse user queries and more than 1.3 million labeled query-passage pairs, covering 48 distinct retrieval tasks derived from six search intents and eight general disaster categories that include 301 specific event types. Our evaluations of 30 state-of-the-art retrieval models demonstrate significant performance variances across tasks, with no single model excelling universally. Furthermore, comparative analyses reveal significant performance gaps between general-domain and disaster management-specific tasks, highlighting the necessity of disaster management-specific benchmarks for guiding IR model selection to support effective decision-making in disaster management scenarios. All source codes and DisastIR are available at this https URL. 

**Abstract (ZH)**: 有效的灾害管理需要及时访问准确且上下文相关的信息。然而，现有的信息检索（IR）基准主要集中在一般或专业领域，如医学或金融，忽视了灾害管理场景中遇到的独特语言复杂性和多样的信息需求。为了填补这一空白，我们引入了DisastIR，这是首个专门针对灾害管理的信息检索综合评价基准。DisastIR包含9600个多样化的用户查询和超过130万标记的查询-段落对，涵盖了来自六个检索意图和八个一般灾害类别中提取的48项独特检索任务，包括301种特定事件类型。我们对30种最先进的检索模型的评估表明，不同任务之间存在显著的性能差异，没有一种模型能在所有任务中表现出色。此外，对比分析揭示了通用领域和灾害管理特定任务之间显著的性能差距，突显了制定灾害管理特定基准的重要性，以指导信息检索模型的选择，支持灾害管理场景中的有效决策。所有源代码和DisastIR均可从此链接访问。 

---
# Integration of TinyML and LargeML: A Survey of 6G and Beyond 

**Title (ZH)**: TinyML和LargeML的整合：面向6G及更远未来的综述 

**Authors**: Thai-Hoc Vu, Ngo Hoang Tu, Thien Huynh-The, Kyungchun Lee, Sunghwan Kim, Miroslav Voznak, Quoc-Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2505.15854)  

**Abstract**: The transition from 5G networks to 6G highlights a significant demand for machine learning (ML). Deep learning models, in particular, have seen wide application in mobile networking and communications to support advanced services in emerging wireless environments, such as smart healthcare, smart grids, autonomous vehicles, aerial platforms, digital twins, and the metaverse. The rapid expansion of Internet-of-Things (IoT) devices, many with limited computational capabilities, has accelerated the development of tiny machine learning (TinyML) and resource-efficient ML approaches for cost-effective services. However, the deployment of large-scale machine learning (LargeML) solutions require major computing resources and complex management strategies to support extensive IoT services and ML-generated content applications. Consequently, the integration of TinyML and LargeML is projected as a promising approach for future seamless connectivity and efficient resource management.
Although the integration of TinyML and LargeML shows abundant potential, several challenges persist, including performance optimization, practical deployment strategies, effective resource management, and security considerations. In this survey, we review and analyze the latest research aimed at enabling the integration of TinyML and LargeML models for the realization of smart services and applications in future 6G networks and beyond. The paper concludes by outlining critical challenges and identifying future research directions for the holistic integration of TinyML and LargeML in next-generation wireless networks. 

**Abstract (ZH)**: 5G网络向6G的过渡突显了对机器学习的显著需求。特别是在移动网络和通信中，深度学习模型已经在智能医疗、智能电网、自动驾驶车辆、空中平台、数字孪生和元宇宙等新兴无线环境中得到了广泛应用，以支持高级服务。物联网（IoT）设备的快速发展，许多设备计算能力有限，加速了对成本效益高的小型机器学习（TinyML）和资源高效机器学习方法的需求。然而，大规模机器学习（LargeML）解决方案的部署需要大量的计算资源和复杂的管理策略，以支持广泛的IoT服务和机器学习生成的内容应用。因此，TinyML和LargeML的结合被预见为未来无缝连接和有效资源管理的一个有希望的方法。

尽管TinyML和LargeML的结合展示出了巨大的潜力，但仍存在一些挑战，包括性能优化、实际部署策略、有效的资源管理以及安全考量。在这篇综述中，我们回顾和分析了最新研究，旨在使TinyML和LargeML模型能够实现未来6G网络及其以上应用中的智能服务和应用。文章最后概述了TinyML和LargeML综合集成的关键挑战，并指出了下一代无线网络中TinyML和LargeML综合集成的未来研究方向。 

---
# Exploring Moral Exercises for Human Oversight of AI systems: Insights from Three Pilot Studies 

**Title (ZH)**: 探索人工道德练习以监管AI系统：三项试点研究的启示 

**Authors**: Silvia Crafa, Teresa Scantamburlo  

**Link**: [PDF](https://arxiv.org/pdf/2505.15851)  

**Abstract**: This paper elaborates on the concept of moral exercises as a means to help AI actors cultivate virtues that enable effective human oversight of AI systems. We explore the conceptual framework and significance of moral exercises, situating them within the contexts of philosophical discourse, ancient practices, and contemporary AI ethics scholarship. We outline the core pillars of the moral exercises methodology - eliciting an engaged personal disposition, fostering relational understanding, and cultivating technomoral wisdom - and emphasize their relevance to key activities and competencies essential for human oversight of AI systems. Our argument is supported by findings from three pilot studies involving a company, a multidisciplinary team of AI researchers, and higher education students. These studies allow us to explore both the potential and the limitations of moral exercises. Based on the collected data, we offer insights into how moral exercises can foster a responsible AI culture within organizations, and suggest directions for future research. 

**Abstract (ZH)**: 本文探讨了道德练习的概念，作为帮助AI主体培养能够有效进行人类监督的人类美德的一种手段。我们探究了道德练习的概念框架及其意义，将其置于哲学话语、古代实践以及当代AI伦理研究的背景下。我们概述了道德练习方法的核心支柱——激发积极个人倾向、培养关系理解以及培养技术美德智慧——并强调了它们对于人类监督AI系统的关键活动和能力的重要性。我们的论点得到了三项试点研究的支持，这些研究涉及一家公司、多学科的AI研究人员团队以及高等教育学生。这些研究使我们得以探索道德练习的潜力及其限制。基于收集的数据，我们提供了有关道德练习如何在组织内培养负责任的AI文化的见解，并建议了未来研究的方向。 

---
# TDFormer: A Top-Down Attention-Controlled Spiking Transformer 

**Title (ZH)**: TDFormer: 一种自上而下注意力控制的神经脉冲变换器 

**Authors**: Zizheng Zhu, Yingchao Yu, Zeqi Zheng, Zhaofei Yu, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.15840)  

**Abstract**: Traditional spiking neural networks (SNNs) can be viewed as a combination of multiple subnetworks with each running for one time step, where the parameters are shared, and the membrane potential serves as the only information link between them. However, the implicit nature of the membrane potential limits its ability to effectively represent temporal information. As a result, each time step cannot fully leverage information from previous time steps, seriously limiting the model's performance. Inspired by the top-down mechanism in the brain, we introduce TDFormer, a novel model with a top-down feedback structure that functions hierarchically and leverages high-order representations from earlier time steps to modulate the processing of low-order information at later stages. The feedback structure plays a role from two perspectives: 1) During forward propagation, our model increases the mutual information across time steps, indicating that richer temporal information is being transmitted and integrated in different time steps. 2) During backward propagation, we theoretically prove that the feedback structure alleviates the problem of vanishing gradients along the time dimension. We find that these mechanisms together significantly and consistently improve the model performance on multiple datasets. In particular, our model achieves state-of-the-art performance on ImageNet with an accuracy of 86.83%. 

**Abstract (ZH)**: 传统的脉冲神经网络（SNN）可以被视为多个子网络的组合，每个子网络在同一时间步运行，参数共享，膜电位作为它们之间的唯一信息链接。然而，膜电位的隐式性质限制了其有效地表示时间信息的能力。因此，每个时间步无法充分利用前一时间步的信息，严重限制了模型的性能。受大脑自上而下机制的启发，我们引入了TDFormer，这是一种具有自上而下反馈结构的新型模型，能够分层运行，并利用早期时间步的高阶表示来调节后期阶段低阶信息的处理。反馈结构从两个角度发挥作用：1）在正向传播过程中，模型增加跨时间步的信息互惠，表明在不同的时间步中传递和整合了更丰富的时空信息。2）在反向传播过程中，我们理论上证明了反馈结构缓解了时间维度中梯度消失的问题。我们发现，这些机制共同显著且一致地提高了模型在多个数据集上的性能。特别是在ImageNet数据集上，我们的模型达到了最先进的准确率86.83%。 

---
# Quantum-Evolutionary Neural Networks for Multi-Agent Federated Learning 

**Title (ZH)**: 量子演化神经网络在多agent联邦学习中的应用 

**Authors**: Aarav Lala, Kalyan Cherukuri  

**Link**: [PDF](https://arxiv.org/pdf/2505.15836)  

**Abstract**: As artificial intelligence continues to drive innovation in complex, decentralized environments, the need for scalable, adaptive, and privacy-preserving decision-making systems has become critical. This paper introduces a novel framework combining quantum-inspired neural networks with evolutionary algorithms to optimize real-time decision-making in multi-agent systems (MAS). The proposed Quantum-Evolutionary Neural Network (QE-NN) leverages quantum computing principles -- such as quantum superposition and entanglement -- to enhance learning speed and decision accuracy, while integrating evolutionary optimization to continually refine agent behaviors in dynamic, uncertain environments. By utilizing federated learning, QE-NN ensures privacy preservation, enabling decentralized agents to collaborate without sharing sensitive data. The framework is designed to allow agents to adapt in real-time to their environments, optimizing decision-making processes for applications in areas such as autonomous systems, smart cities, and healthcare. This research represents a breakthrough in merging quantum computing, evolutionary optimization, and privacy-preserving techniques to solve complex problems in multi-agent decision-making systems, pushing the boundaries of AI in real-world, privacy-sensitive applications. 

**Abstract (ZH)**: 随着人工智能在复杂分散环境中的持续创新，构建可扩展、适应性强且保护隐私的决策系统变得至关重要。本文提出了一种结合量子启发神经网络与演化算法的新框架，以优化多智能体系统（MAS）的实时决策。所提出的量子-演化神经网络（QE-NN）利用量子计算原理——如量子叠加和纠缠——来提高学习速度和决策准确性，并结合演化优化持续改进智能体行为，使其能够在动态、不确定性环境中不断优化。通过利用联邦学习，QE-NN确保了隐私保护，使分散的智能体能够在不共享敏感数据的情况下进行协作。该框架设计旨在使智能体能够实时适应环境，优化决策过程，应用于自主系统、智慧城市和医疗健康等领域。该研究代表了将量子计算、演化优化和隐私保护技术融合以解决多智能体决策系统中复杂问题的突破，推动了在隐私敏感的实际应用中人工智能的边界。 

---
# MPPFND: A Dataset and Analysis of Detecting Fake News with Multi-Platform Propagation 

**Title (ZH)**: MPPFND：多平台传播的假新闻检测数据集及分析 

**Authors**: Congyuan Zhao, Lingwei Wei, Ziming Qin, Wei Zhou, Yunya Song, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15834)  

**Abstract**: Fake news spreads widely on social media, leading to numerous negative effects. Most existing detection algorithms focus on analyzing news content and social context to detect fake news. However, these approaches typically detect fake news based on specific platforms, ignoring differences in propagation characteristics across platforms. In this paper, we introduce the MPPFND dataset, which captures propagation structures across multiple platforms. We also describe the commenting and propagation characteristics of different platforms to show that their social contexts have distinct features. We propose a multi-platform fake news detection model (APSL) that uses graph neural networks to extract social context features from various platforms. Experiments show that accounting for cross-platform propagation differences improves fake news detection performance. 

**Abstract (ZH)**: 多平台假新闻检测数据集及基于图神经网络的多平台假新闻检测模型 

---
# From Hand-Crafted Metrics to Evolved Training-Free Performance Predictors for Neural Architecture Search via Genetic Programming 

**Title (ZH)**: 从手工构建的度量标准到通过遗传编程进化出的无训练性能预测器：针对神经架构搜索的进化训练-free方法 

**Authors**: Quan Minh Phan, Ngoc Hoang Luong  

**Link**: [PDF](https://arxiv.org/pdf/2505.15832)  

**Abstract**: Estimating the network performance using zero-cost (ZC) metrics has proven both its efficiency and efficacy in Neural Architecture Search (NAS). However, a notable limitation of most ZC proxies is their inconsistency, as reflected by the substantial variation in their performance across different problems. Furthermore, the design of existing ZC metrics is manual, involving a time-consuming trial-and-error process that requires substantial domain expertise. These challenges raise two critical questions: (1) Can we automate the design of ZC metrics? and (2) Can we utilize the existing hand-crafted ZC metrics to synthesize a more generalizable one? In this study, we propose a framework based on Symbolic Regression via Genetic Programming to automate the design of ZC metrics. Our framework is not only highly extensible but also capable of quickly producing a ZC metric with a strong positive rank correlation to true network performance across diverse NAS search spaces and tasks. Extensive experiments on 13 problems from NAS-Bench-Suite-Zero demonstrate that our automatically generated proxies consistently outperform hand-crafted alternatives. Using our evolved proxy metric as the search objective in an evolutionary algorithm, we could identify network architectures with competitive performance within 15 minutes using a single consumer GPU. 

**Abstract (ZH)**: 使用符号回归通过遗传编程自动化设计零成本度量以估计网络性能的研究 

---
# Generative AI-Aided QoE Maximization for RIS-Assisted Digital Twin Interaction 

**Title (ZH)**: 基于RIS辅助的数字孪生交互中生成式AI辅助的QoE最大化 

**Authors**: Jiayuan Chen, Yuxiang Li, Changyan Yi, Shimin Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.15828)  

**Abstract**: In this paper, we investigate a quality of experience (QoE)-aware resource allocation problem for reconfigurable intelligent surface (RIS)-assisted digital twin (DT) interaction with uncertain evolution. In the considered system, mobile users are expected to interact with a DT model maintained on a DT server that is deployed on a base station, via effective uplink and downlink channels assisted by an RIS. Our goal is to maximize the sum of all mobile users' joint subjective and objective QoE in DT interactions across various DT scenes, by jointly optimizing phase shift matrix, receive/transmit beamforming matrix, rendering resolution configuration and computing resource allocation. While solving this problem is challenging mainly due to the uncertain evolution of the DT model, which leads to multiple scene-specific problems, and require us to constantly re-solve each of them whenever DT model evolves.
To this end, leveraging the dynamic optimization capabilities of decision transformers and the generalization strengths of generative artificial intelligence (GAI), we propose a novel GAI-aided approach, called the prompt-guided decision transformer integrated with zero-forcing optimization (PG-ZFO). Simulations are conducted to evaluate the proposed PG-ZFO, demonstrating its effectiveness and superiority over counterparts. 

**Abstract (ZH)**: 基于RIS辅助数字孪生交互的不确定演化效用感知资源分配问题：提示引导决策变换器结合零强迫优化方法 

---
# A Novel Compound AI Model for 6G Networks in 3D Continuum 

**Title (ZH)**: 一种用于三维连续体的6G网络新型复合人工智能模型 

**Authors**: Milos Gravara, Andrija Stanisic, Stefan Nastic  

**Link**: [PDF](https://arxiv.org/pdf/2505.15821)  

**Abstract**: The 3D continuum presents a complex environment that spans the terrestrial, aerial and space domains, with 6Gnetworks serving as a key enabling technology. Current AI approaches for network management rely on monolithic models that fail to capture cross-domain interactions, lack adaptability,and demand prohibitive computational resources. This paper presents a formal model of Compound AI systems, introducing a novel tripartite framework that decomposes complex tasks into specialized, interoperable modules. The proposed modular architecture provides essential capabilities to address the unique challenges of 6G networks in the 3D continuum, where heterogeneous components require coordinated, yet distributed, intelligence. This approach introduces a fundamental trade-off between model and system performance, which must be carefully addressed. Furthermore, we identify key challenges faced by Compound AI systems within 6G networks operating in the 3D continuum, including cross-domain resource orchestration, adaptation to dynamic topologies, and the maintenance of consistent AI service quality across heterogeneous environments. 

**Abstract (ZH)**: 3D连续体中的6G网络复合人工智能系统的形式模型及挑战 

---
# Common Data Format (CDF): A Standardized Format for Match-Data in Football (Soccer) 

**Title (ZH)**: 通用数据格式（CDF）：足球比赛匹配数据的标准格式 

**Authors**: Gabriel Anzer, Kilian Arnsmeyer, Pascal Bauer, Joris Bekkers, Ulf Brefeld, Jesse Davis, Nicolas Evans, Matthias Kempe, Samuel J Robertson, Joshua Wyatt Smith, Jan Van Haaren  

**Link**: [PDF](https://arxiv.org/pdf/2505.15820)  

**Abstract**: During football matches, a variety of different parties (e.g., companies) each collect (possibly overlapping) data about the match ranging from basic information (e.g., starting players) to detailed positional data. This data is provided to clubs, federations, and other organizations who are increasingly interested in leveraging this data to inform their decision making. Unfortunately, analyzing such data pose significant barriers because each provider may (1) collect different data, (2) use different specifications even within the same category of data, (3) represent the data differently, and (4) delivers the data in a different manner (e.g., file format, protocol). Consequently, working with these data requires a significant investment of time and money. The goal of this work is to propose a uniform and standardized format for football data called the Common Data Format (CDF). The CDF specifies a minimal schema for five types of match data: match sheet data, video footage, event data, tracking data, and match meta data. It aims to ensure that the provided data is clear, sufficiently contextualized (e.g., its provenance is clear), and complete such that it enables common downstream analysis tasks. Concretely, this paper will detail the technical specifications of the CDF, the representational choices that were made to help ensure the clarity of the provided data, and a concrete approach for delivering data in the CDF. 

**Abstract (ZH)**: 足球比赛中的通用数据格式（CDF）：一种统一且标准化的数据格式 

---
