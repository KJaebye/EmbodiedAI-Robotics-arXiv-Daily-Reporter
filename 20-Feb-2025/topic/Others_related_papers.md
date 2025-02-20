# Multi-Covering a Point Set by $m$ Disks with Minimum Total Area 

**Title (ZH)**: 用最小总面积覆盖点集的m个圆盘多覆盖 

**Authors**: Mariem Guitouni, Chek-Manh Loi, Sándor P. Fekete, Michael Perk, Aaron T. Becker  

**Link**: [PDF](https://arxiv.org/pdf/2502.13773)  

**Abstract**: A common robotics sensing problem is to place sensors to robustly monitor a set of assets, where robustness is assured by requiring asset $p$ to be monitored by at least $\kappa(p)$ sensors. Given $n$ assets that must be observed by $m$ sensors, each with a disk-shaped sensing region, where should the sensors be placed to minimize the total area observed? We provide and analyze a fast heuristic for this problem. We then use the heuristic to initialize an exact Integer Programming solution. Subsequently, we enforce separation constraints between the sensors by modifying the integer program formulation and by changing the disk candidate set. 

**Abstract (ZH)**: 一种常见的机器人感知问题是将传感器放置以稳健地监测一组资产，其中通过要求每个资产至少被κ(p)个传感器监测来保证稳健性。给定n个资产必须由m个传感器观测，每个传感器的感知区域为圆盘形，如何将传感器放置以最小化总观测面积？我们提供并分析了一个快速启发式算法。然后，我们使用启发式算法初始化精确整数规划解决方案。随后，通过修改整数规划模型并改变圆盘候选集，我们施加传感器之间的分离约束。 

---
# SLAMSpoof: Practical LiDAR Spoofing Attacks on Localization Systems Guided by Scan Matching Vulnerability Analysis 

**Title (ZH)**: SLAMSpoof：基于扫描匹配漏洞分析的激光雷达欺骗攻击在定位系统中的应用 

**Authors**: Rokuto Nagata, Kenji Koide, Yuki Hayakawa, Ryo Suzuki, Kazuma Ikeda, Ozora Sako, Qi Alfred Chen, Takami Sato, Kentaro Yoshioka  

**Link**: [PDF](https://arxiv.org/pdf/2502.13641)  

**Abstract**: Accurate localization is essential for enabling modern full self-driving services. These services heavily rely on map-based traffic information to reduce uncertainties in recognizing lane shapes, traffic light locations, and traffic signs. Achieving this level of reliance on map information requires centimeter-level localization accuracy, which is currently only achievable with LiDAR sensors. However, LiDAR is known to be vulnerable to spoofing attacks that emit malicious lasers against LiDAR to overwrite its measurements. Once localization is compromised, the attack could lead the victim off roads or make them ignore traffic lights. Motivated by these serious safety implications, we design SLAMSpoof, the first practical LiDAR spoofing attack on localization systems for self-driving to assess the actual attack significance on autonomous vehicles. SLAMSpoof can effectively find the effective attack location based on our scan matching vulnerability score (SMVS), a point-wise metric representing the potential vulnerability to spoofing attacks. To evaluate the effectiveness of the attack, we conduct real-world experiments on ground vehicles and confirm its high capability in real-world scenarios, inducing position errors of $\geq$4.2 meters (more than typical lane width) for all 3 popular LiDAR-based localization algorithms. We finally discuss the potential countermeasures of this attack. Code is available at this https URL 

**Abstract (ZH)**: 准确的定位对于实现现代全自动驾驶服务至关重要。这些服务高度依赖基于地图的道路信息，以减少车道形状、交通灯位置和交通标志识别的不确定性。要实现这种依赖性，需要厘米级的定位精度，目前仅LiDAR传感器能够实现。然而，LiDAR已知容易受到伪造攻击的影响，攻击者会发射恶意激光干扰LiDAR的测量结果。一旦定位被破坏，攻击可能导致车辆偏离道路或忽略交通信号。基于这些严重的安全问题，我们设计了SLAMSpoof，这是第一个针对自动驾驶定位系统实施LiDAR伪造攻击的实用方法，以评估此类攻击对自主车辆的实际影响。SLAMSpoof能够基于我们的扫描匹配脆弱性评分（SMVS）有效地找到有效的攻击位置，这是一种衡量伪造攻击潜在脆弱性的点对点度量。为了评估攻击的有效性，我们在地面车辆上进行了实地实验，并确认了其在实际场景中的高能力，导致所有3种流行的基于LiDAR的定位算法产生位置误差大于4.2米（超过典型车道宽度）。最后，我们讨论了该攻击的潜在防御措施。代码可在以下链接获取。 

---
# MILE: Model-based Intervention Learning 

**Title (ZH)**: 基于模型的干预学习 

**Authors**: Yigit Korkmaz, Erdem Bıyık  

**Link**: [PDF](https://arxiv.org/pdf/2502.13519)  

**Abstract**: Imitation learning techniques have been shown to be highly effective in real-world control scenarios, such as robotics. However, these approaches not only suffer from compounding error issues but also require human experts to provide complete trajectories. Although there exist interactive methods where an expert oversees the robot and intervenes if needed, these extensions usually only utilize the data collected during intervention periods and ignore the feedback signal hidden in non-intervention timesteps. In this work, we create a model to formulate how the interventions occur in such cases, and show that it is possible to learn a policy with just a handful of expert interventions. Our key insight is that it is possible to get crucial information about the quality of the current state and the optimality of the chosen action from expert feedback, regardless of the presence or the absence of intervention. We evaluate our method on various discrete and continuous simulation environments, a real-world robotic manipulation task, as well as a human subject study. Videos and the code can be found at this https URL . 

**Abstract (ZH)**: 模仿学习技术在实际控制场景中，如机器人领域，已被证明非常有效。然而，这些方法不仅受到累积误差问题的影响，还需要人类专家提供完整的轨迹。虽然存在交互式方法，其中专家监督机器人并在必要时进行干预，但这些扩展通常仅利用干预期间收集的数据，而忽略了非干预时间段中隐含的反馈信号。在本工作中，我们创建了一个模型来阐述在这种情况下干预的发生机制，并证明仅凭少数几次专家干预即可学习出策略。我们的关键见解是，即使没有或没有干预，也能够从专家反馈中获取有关当前状态质量及所选动作最优性的关键信息。我们在各种离散和连续的模拟环境中、一个实际的机器人操作任务以及一项人类被试研究中评估了该方法。相关视频和代码可在此网站找到：this https URL。 

---
# Muscle Activation Estimation by Optimzing the Musculoskeletal Model for Personalized Strength and Conditioning Training 

**Title (ZH)**: 基于优化肌骨模型的个性化力量与条件训练肌肉激活估计 

**Authors**: Xi Wu, Chenzui Li, Kehan Zou, Ning Xi, Fei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13760)  

**Abstract**: Musculoskeletal models are pivotal in the domains of rehabilitation and resistance training to analyze muscle conditions. However, individual variability in musculoskeletal parameters and the immeasurability of some internal biomechanical variables pose significant obstacles to accurate personalized modelling. Furthermore, muscle activation estimation can be challenging due to the inherent redundancy of the musculoskeletal system, where multiple muscles drive a single joint. This study develops a whole-body musculoskeletal model for strength and conditioning training and calibrates relevant muscle parameters with an electromyography-based optimization method. By utilizing the personalized musculoskeletal model, muscle activation can be subsequently estimated to analyze the performance of exercises. Bench press and deadlift are chosen for experimental verification to affirm the efficacy of this approach. 

**Abstract (ZH)**: 肌骨模型在康复和抗阻力训练领域至关重要，用于分析肌肉状况。然而，肌骨参数的个体差异和某些内部生物力学变量的不可测量性对准确的个性化建模构成了重大障碍。此外，由于肌骨系统的固有冗余性，多个肌肉驱动单一关节，使得肌肉激活估计具有挑战性。本研究开发了适用于力量与体能训练的全身肌骨模型，并采用电生理图优化方法校准相关肌肉参数。通过利用个性化的肌骨模型，可以进一步估计肌肉激活，以分析训练表现。卧推和硬拉被选作实验验证的对象，以证实该方法的有效性。 

---
# PCB Renewal: Iterative Reuse of PCB Substrates for Sustainable Electronic Making 

**Title (ZH)**: PCB 重现：迭代reuse PCB 载板以实现可持续電子製造 

**Authors**: Zeyu Yan, Advait Vartak, Jiasheng Li, Zining Zhang, Huaishu Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.13255)  

**Abstract**: PCB (printed circuit board) substrates are often single-use, leading to material waste in electronics making. We introduce PCB Renewal, a novel technique that "erases" and "reconfigures" PCB traces by selectively depositing conductive epoxy onto outdated areas, transforming isolated paths into conductive planes that support new traces. We present the PCB Renewal workflow, evaluate its electrical performance and mechanical durability, and model its sustainability impact, including material usage, cost, energy consumption, and time savings. We develop a software plug-in that guides epoxy deposition, generates updated PCB profiles, and calculates resource usage. To demonstrate PCB Renewal's effectiveness and versatility, we repurpose a single PCB across four design iterations spanning three projects: a camera roller, a WiFi radio, and an ESPboy game console. We also show how an outsourced double-layer PCB can be reconfigured, transforming it from an LED watch to an interactive cat toy. The paper concludes with limitations and future directions. 

**Abstract (ZH)**: PCB Renewal: A Novel Technique for Erasing and Reconfiguring PCB Traces to Reduce Material Waste in Electronics Manufacturing 

---
# Autonomous Vehicles Using Multi-Agent Reinforcement Learning for Routing Decisions Can Harm Urban Traffic 

**Title (ZH)**: 基于多智能体 reinforcement 学习的自动驾驶车辆在路径决策中的应用可能危害城市交通 

**Authors**: Anastasia Psarou, Ahmet Onur Akman, Łukasz Gorczyca, Michał Hoffmann, Zoltán György Varga, Grzegorz Jamróz, Rafał Kucharski  

**Link**: [PDF](https://arxiv.org/pdf/2502.13188)  

**Abstract**: Autonomous vehicles (AVs) using Multi-Agent Reinforcement Learning (MARL) for simultaneous route optimization may destabilize traffic environments, with human drivers possibly experiencing longer travel times. We study this interaction by simulating human drivers and AVs. Our experiments with standard MARL algorithms reveal that, even in trivial cases, policies often fail to converge to an optimal solution or require long training periods. The problem is amplified by the fact that we cannot rely entirely on simulated training, as there are no accurate models of human routing behavior. At the same time, real-world training in cities risks destabilizing urban traffic systems, increasing externalities, such as $CO_2$ emissions, and introducing non-stationarity as human drivers adapt unpredictably to AV behaviors. Centralization can improve convergence in some cases, however, it raises privacy concerns for the travelers' destination data. In this position paper, we argue that future research must prioritize realistic benchmarks, cautious deployment strategies, and tools for monitoring and regulating AV routing behaviors to ensure sustainable and equitable urban mobility systems. 

**Abstract (ZH)**: 基于多智能体强化学习的自动驾驶车辆同时路径优化对交通环境的影响及其交互研究：需要优先考虑现实基准、谨慎部署策略和监控调节工具以确保可持续和公平的城市移动系统。 

---
# AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence 

**Title (ZH)**: 自适应步长：通过模型信心自动分割推理步骤 

**Authors**: Yuliang Liu, Junjie Lu, Zhaoling Chen, Chaofeng Qu, Jason Klein Liu, Chonghan Liu, Zefan Cai, Yunhui Xia, Li Zhao, Jiang Bian, Chuheng Zhang, Wei Shen, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.13943)  

**Abstract**: Current approaches for training Process Reward Models (PRMs) often involve breaking down responses into multiple reasoning steps using rule-based techniques, such as using predefined placeholder tokens or setting the reasoning step's length into a fixed size. These approaches overlook the fact that specific words do not typically mark true decision points in a text. To address this, we propose AdaptiveStep, a method that divides reasoning steps based on the model's confidence in predicting the next word. This division method provides more decision-making information at each step, enhancing downstream tasks, such as reward model learning. Moreover, our method does not require manual annotation. We demonstrate its effectiveness through experiments with AdaptiveStep-trained PRMs in mathematical reasoning and code generation tasks. Experimental results indicate that the outcome PRM achieves state-of-the-art Best-of-N performance, surpassing greedy search strategy with token-level value-guided decoding, while also reducing construction costs by over 30% compared to existing open-source PRMs. In addition, we provide a thorough analysis and case study on the PRM's performance, transferability, and generalization capabilities. 

**Abstract (ZH)**: 当前用于训练过程奖励模型（PRMs）的方法通常涉及使用基于规则的技术将响应拆分成多个推理步骤，例如使用预定义的占位符令牌或设定推理步骤长度为固定大小。这些方法忽视了特定单词通常不标记文本中的真正决策点这一事实。为解决这一问题，我们提出了一种称为AdaptiveStep的方法，该方法根据模型预测下一个单词的信心程度来划分推理步骤。这种划分方法在每一步提供了更多的决策信息，从而增强下游任务，如奖励模型学习。此外，我们的方法不需要人工标注。通过在数学推理和代码生成任务中使用AdaptiveStep训练的PRMs进行实验，展示了其有效性。实验结果表明，所得到的PRM实现了最先进的Best-of-N性能，优于基于标记级价值引导解码的贪婪搜索策略，并且与现有开源PRMs相比，构建成本降低了超过30%。此外，我们还对PRM的性能、转移能力和泛化能力进行了全面分析和案例研究。 

---
# Scoring Verifiers: Evaluating Synthetic Verification in Code and Reasoning 

**Title (ZH)**: 评分验证者：评估代码和推理中的合成验证 

**Authors**: Aleksander Ficek, Somshubra Majumdar, Vahid Noroozi, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2502.13820)  

**Abstract**: Code verification has recently found great success as a critical component in training large scale reasoning models for coding. Synthetic techniques such as self-generated test cases and reward models provide a way to enhance code capabilities beyond predefined tests. Building on these advancements, we propose new benchmarks designed to systematically evaluate the impact of synthetic verification methods on assessing solution correctness. We introduce HE-R, HE-R+, MBPP-R, and MBPP-R+, which transform existing coding benchmarks into scoring and ranking datasets to evaluate the effectiveness of synthetic verifiers. Using these benchmarks, we analyze synthetic verification methods in standard, reasoning-based, and reward-based LLMs. Our results show that recent reasoning models significantly improve test case generation and that scaling test cases enhances verification accuracy. 

**Abstract (ZH)**: 合成验证方法对评估解决方案正确性的系统影响研究：从预定义测试到增强代码能力 

---
# A consensus set for the aggregation of partial rankings: the case of the Optimal Set of Bucket Orders Problem 

**Title (ZH)**: 最优桶序集合用于部分排名聚合的问题 

**Authors**: Juan A. Aledo, José A. Gámez, Alejandro Rosete  

**Link**: [PDF](https://arxiv.org/pdf/2502.13769)  

**Abstract**: In rank aggregation problems (RAP), the solution is usually a consensus ranking that generalizes a set of input orderings. There are different variants that differ not only in terms of the type of rankings that are used as input and output, but also in terms of the objective function employed to evaluate the quality of the desired output ranking. In contrast, in some machine learning tasks (e.g. subgroup discovery) or multimodal optimization tasks, attention is devoted to obtaining several models/results to account for the diversity in the input data or across the search landscape. Thus, in this paper we propose to provide, as the solution to an RAP, a set of rankings to better explain the preferences expressed in the input orderings. We exemplify our proposal through the Optimal Bucket Order Problem (OBOP), an RAP which consists in finding a single consensus ranking (with ties) that generalizes a set of input rankings codified as a precedence matrix. To address this, we introduce the Optimal Set of Bucket Orders Problem (OSBOP), a generalization of the OBOP that aims to produce not a single ranking as output but a set of consensus rankings. Experimental results are presented to illustrate this proposal, showing how, by providing a set of consensus rankings, the fitness of the solution significantly improves with respect to the one of the original OBOP, without losing comprehensibility. 

**Abstract (ZH)**: 基于排名聚合问题的最优桶序集问题及其应用 

---
# Inference of Abstraction for Grounded Predicate Logic 

**Title (ZH)**: 基于Grounded谓词逻辑的抽象推理 

**Authors**: Hiroyuki Kido  

**Link**: [PDF](https://arxiv.org/pdf/2502.13743)  

**Abstract**: An important open question in AI is what simple and natural principle enables a machine to reason logically for meaningful abstraction with grounded symbols. This paper explores a conceptually new approach to combining probabilistic reasoning and predicative symbolic reasoning over data. We return to the era of reasoning with a full joint distribution before the advent of Bayesian networks. We then discuss that a full joint distribution over models of exponential size in propositional logic and of infinite size in predicate logic should be simply derived from a full joint distribution over data of linear size. We show that the same process is not only enough to generalise the logical consequence relation of predicate logic but also to provide a new perspective to rethink well-known limitations such as the undecidability of predicate logic, the symbol grounding problem and the principle of explosion. The reproducibility of this theoretical work is fully demonstrated by the included proofs. 

**Abstract (ZH)**: 人工智能领域的一个重要开放问题是，什么简单而自然的原则能让机器进行逻辑推理并实现基于具体符号的有意义抽象？本文探讨了一种结合概率推理和预测性符号推理的新概念性方法。我们回到贝叶斯网络出现之前的全联合分布推理时代。然后讨论在命题逻辑中，全联合分布在模型上的大小呈指数级，在谓词逻辑中则为无限大，这种全联合分布应该能从数据的线性大小的全联合分布中简单推导而出。我们证明，同样的过程不仅足以推广谓词逻辑的逻辑后承关系，还提供了重新思考谓词逻辑的不可判定性、符号接地问题以及矛盾原则的新视角。本文的理论工作由包含的证明完全展示了其可重现性。 

---
# Robust Counterfactual Inference in Markov Decision Processes 

**Title (ZH)**: 马尔可夫决策过程中的鲁棒反事实推理 

**Authors**: Jessica Lally, Milad Kazemi, Nicola Paoletti  

**Link**: [PDF](https://arxiv.org/pdf/2502.13731)  

**Abstract**: This paper addresses a key limitation in existing counterfactual inference methods for Markov Decision Processes (MDPs). Current approaches assume a specific causal model to make counterfactuals identifiable. However, there are usually many causal models that align with the observational and interventional distributions of an MDP, each yielding different counterfactual distributions, so fixing a particular causal model limits the validity (and usefulness) of counterfactual inference. We propose a novel non-parametric approach that computes tight bounds on counterfactual transition probabilities across all compatible causal models. Unlike previous methods that require solving prohibitively large optimisation problems (with variables that grow exponentially in the size of the MDP), our approach provides closed-form expressions for these bounds, making computation highly efficient and scalable for non-trivial MDPs. Once such an interval counterfactual MDP is constructed, our method identifies robust counterfactual policies that optimise the worst-case reward w.r.t. the uncertain interval MDP probabilities. We evaluate our method on various case studies, demonstrating improved robustness over existing methods. 

**Abstract (ZH)**: 本文解决了一类马尔可夫决策过程(MDP)反事实推理方法中存在的关键限制。当前的方法假设特定的因果模型以使反事实可识别。然而，通常存在许多与MDP的观察分布和干预分布相一致的因果模型，每个模型会产生不同的反事实分布，固定特定的因果模型限制了反事实推理的有效性（及其实用性）。我们提出了一种新颖的非参数方法，该方法在所有兼容因果模型上计算反事实转换概率的紧界。与先前需要求解难以承受规模的优化问题（变量随MDP规模呈指数增长）的方法不同，我们的方法提供了这些界的具体形式表达式，使得计算在非平凡的MDP上变得高度高效和可扩展。一旦构建了这样的区间反事实MDP，我们的方法就能识别出在不确定的区间MDP概率下优化最坏情况奖励的稳健反事实策略。我们通过各种案例研究评估了该方法，展示了其相比现有方法的鲁棒性改进。 

---
# Causes and Strategies in Multiagent Systems 

**Title (ZH)**: 多Agent系统中的原因与策略研究 

**Authors**: Sylvia S. Kerkhove, Natasha Alechina, Mehdi Dastani  

**Link**: [PDF](https://arxiv.org/pdf/2502.13701)  

**Abstract**: Causality plays an important role in daily processes, human reasoning, and artificial intelligence. There has however not been much research on causality in multi-agent strategic settings. In this work, we introduce a systematic way to build a multi-agent system model, represented as a concurrent game structure, for a given structural causal model. In the obtained so-called causal concurrent game structure, transitions correspond to interventions on agent variables of the given causal model. The Halpern and Pearl framework of causality is used to determine the effects of a certain value for an agent variable on other variables. The causal concurrent game structure allows us to analyse and reason about causal effects of agents' strategic decisions. We formally investigate the relation between causal concurrent game structures and the original structural causal models. 

**Abstract (ZH)**: 因果关系在日常过程、人类推理和人工智能中起着重要作用。然而，在多智能体战略性设置中关于因果关系的研究较少。在本文中，我们提出了一个系统的方法，以给定的结构因果模型为基础构建一个表示为并发游戏结构的多智能体系统模型。在获得的所谓的因果并发游戏结构中，转换对应于对给定因果模型中的智能体变量进行干预。我们使用Halpern和Pearl的因果框架来确定特定智能体变量值对其他变量的影响。因果并发游戏结构使我们能够分析和推理智能体战略性决策的因果效果。我们正式地研究了因果并发游戏结构与原始结构因果模型之间的关系。 

---
# Vision-Based Generic Potential Function for Policy Alignment in Multi-Agent Reinforcement Learning 

**Title (ZH)**: 基于视觉的通用潜在函数在多智能体 reinforcement learning 中的策略对齐 

**Authors**: Hao Ma, Shijie Wang, Zhiqiang Pu, Siyao Zhao, Xiaolin Ai  

**Link**: [PDF](https://arxiv.org/pdf/2502.13430)  

**Abstract**: Guiding the policy of multi-agent reinforcement learning to align with human common sense is a difficult problem, largely due to the complexity of modeling common sense as a reward, especially in complex and long-horizon multi-agent tasks. Recent works have shown the effectiveness of reward shaping, such as potential-based rewards, to enhance policy alignment. The existing works, however, primarily rely on experts to design rule-based rewards, which are often labor-intensive and lack a high-level semantic understanding of common sense. To solve this problem, we propose a hierarchical vision-based reward shaping method. At the bottom layer, a visual-language model (VLM) serves as a generic potential function, guiding the policy to align with human common sense through its intrinsic semantic understanding. To help the policy adapts to uncertainty and changes in long-horizon tasks, the top layer features an adaptive skill selection module based on a visual large language model (vLLM). The module uses instructions, video replays, and training records to dynamically select suitable potential function from a pre-designed pool. Besides, our method is theoretically proven to preserve the optimal policy. Extensive experiments conducted in the Google Research Football environment demonstrate that our method not only achieves a higher win rate but also effectively aligns the policy with human common sense. 

**Abstract (ZH)**: 基于视觉的分层奖励塑造方法引导多智能体强化学习政策与人类常识对齐 

---
# Bi-Fact: A Bidirectional Factorization-based Evaluation of Intent Extraction from UI Trajectories 

**Title (ZH)**: 双向分解：基于双向分解的UI轨迹意图提取评价方法 

**Authors**: Sapir Caduri  

**Link**: [PDF](https://arxiv.org/pdf/2502.13149)  

**Abstract**: Bi-Fact, a novel approach to automatic evaluation for Intent Understanding, is presented. Drawing inspiration from FactScore, Bi-Fact enables fine-grained intent comparison by splitting both gold and predicted intents into facts and calculating precision and recall, considering the UI trajectory. This paper outlines a comprehensive evaluation of Bi-Fact, assessing its performance and comparing it to existing metrics. 

**Abstract (ZH)**: Bi-Fact：一种新颖的意图理解自动评估方法 

---
# Partially Observable Gaussian Process Network and Doubly Stochastic Variational Inference 

**Title (ZH)**: 部分可观测高斯过程网络与双重随机变分推断 

**Authors**: Saksham Kiroriwal, Julius Pfrommer, Jürgen Beyerer  

**Link**: [PDF](https://arxiv.org/pdf/2502.13905)  

**Abstract**: To reduce the curse of dimensionality for Gaussian processes (GP), they can be decomposed into a Gaussian Process Network (GPN) of coupled subprocesses with lower dimensionality. In some cases, intermediate observations are available within the GPN. However, intermediate observations are often indirect, noisy, and incomplete in most real-world systems. This work introduces the Partially Observable Gaussian Process Network (POGPN) to model real-world process networks. We model a joint distribution of latent functions of subprocesses and make inferences using observations from all subprocesses. POGPN incorporates observation lenses (observation likelihoods) into the well-established inference method of deep Gaussian processes. We also introduce two training methods for POPGN to make inferences on the whole network using node observations. The application to benchmark problems demonstrates how incorporating partial observations during training and inference can improve the predictive performance of the overall network, offering a promising outlook for its practical application. 

**Abstract (ZH)**: 基于部分可观测性的高斯过程网络用于降低高维性的 curse 

---
# PSCon: Toward Conversational Product Search 

**Title (ZH)**: PSCon: 向会话式产品搜索迈进 

**Authors**: Jie Zou, Mohammad Aliannejadi, Evangelos Kanoulas, Shuxi Han, Heli Ma, Zheng Wang, Yang Yang, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13881)  

**Abstract**: Conversational Product Search (CPS) is confined to simulated conversations due to the lack of real-world CPS datasets that reflect human-like language. Additionally, current conversational datasets are limited to support cross-market and multi-lingual usage. In this paper, we introduce a new CPS data collection protocol and present PSCon, a novel CPS dataset designed to assist product search via human-like conversations. The dataset is constructed using a coached human-to-human data collection protocol and supports two languages and dual markets. Also, the dataset enables thorough exploration of six subtasks of CPS: user intent detection, keyword extraction, system action prediction, question selection, item ranking, and response generation. Furthermore, we also offer an analysis of the dataset and propose a benchmark model on the proposed CPS dataset. 

**Abstract (ZH)**: 基于对话的产品搜索（CPS）受限于缺乏反映人类语言的现实世界CPS数据集，此外，现有的对话数据集也限制了跨市场和多语言的应用。本文介绍了一种新的CPS数据收集协议，并推出了PSCon这一新型CPS数据集，旨在通过类似人类的对话方式辅助产品搜索。该数据集采用引导的人际数据收集协议构建，支持两种语言和两个市场。此外，该数据集还能够全面探索CPS的六个子任务：用户意图检测、关键词提取、系统动作预测、问题选择、物品排名和响应生成。Furthermore，我们还对该数据集进行了分析，并在所提出的CPS数据集上提出了一个基准模型。 

---
# NVR: Vector Runahead on NPUs for Sparse Memory Access 

**Title (ZH)**: NVR: Vector Runahead on NPUs for Sparse Memory Access 

**Authors**: Hui Wang, Zhengpeng Zhao, Jing Wang, Yushu Du, Yuan Cheng, Bing Guo, He Xiao, Chenhao Ma, Xiaomeng Han, Dean You, Jiapeng Guan, Ran Wei, Dawei Yang, Zhe Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13873)  

**Abstract**: Deep Neural Networks are increasingly leveraging sparsity to reduce the scaling up of model parameter size. However, reducing wall-clock time through sparsity and pruning remains challenging due to irregular memory access patterns, leading to frequent cache misses. In this paper, we present NPU Vector Runahead (NVR), a prefetching mechanism tailored for NPUs to address cache miss problems in sparse DNN workloads. Rather than optimising memory patterns with high overhead and poor portability, NVR adapts runahead execution to the unique architecture of NPUs. NVR provides a general micro-architectural solution for sparse DNN workloads without requiring compiler or algorithmic support, operating as a decoupled, speculative, lightweight hardware sub-thread alongside the NPU, with minimal hardware overhead (under 5%). NVR achieves an average 90% reduction in cache misses compared to SOTA prefetching in general-purpose processors, delivering 4x average speedup on sparse workloads versus NPUs without prefetching. Moreover, we investigate the advantages of incorporating a small cache (16KB) into the NPU combined with NVR. Our evaluation shows that expanding this modest cache delivers 5x higher performance benefits than increasing the L2 cache size by the same amount. 

**Abstract (ZH)**: 深层神经网络正越来越多地利用稀疏性来减少模型参数规模的扩展。然而，通过稀疏性与剪枝减少实际运行时间仍然具有挑战性，因为不规则的内存访问模式导致频繁的缓存缺失。在本文中，我们提出了适用于NPUs的NPU向量前瞻机制（NPU Vector Runahead, NVR），该机制旨在解决稀疏DNN工作负载中的缓存缺失问题。NVR 不是通过具有高开销和较差移植性的优化内存模式来解决问题，而是根据NPU的独特架构进行前瞻执行的调整。NVR 提供了一种适用于稀疏DNN工作负载的一般微架构解决方案，无需编译器或算法支持，并在NPU旁边作为一个解耦、推测性的轻量级硬件子线程操作，硬件开销不到5%。NVR 在缓存缺失次数上比通用处理器的最新预取技术平均减少了90%，对于没有预取的NPUs，在稀疏工作负载上的平均加速比达到4倍。此外，我们研究了将小型缓存（16KB）集成到NPU中与NVR结合的优劣。我们的评估表明，扩大这一适度大小的缓存比增加相同量的L2缓存大小提供5倍以上的性能优势。 

---
# DH-RAG: A Dynamic Historical Context-Powered Retrieval-Augmented Generation Method for Multi-Turn Dialogue 

**Title (ZH)**: DH-RAG：一种基于动态历史语境的检索增强生成方法用于多轮对话 

**Authors**: Feiyuan Zhang, Dezhi Zhu, James Ming, Yilun Jin, Di Chai, Liu Yang, Han Tian, Zhaoxin Fan, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13847)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have shown substantial benefits in applications such as question answering and multi-turn dialogue \citep{lewis2020retrieval}. However, traditional RAG methods, while leveraging static knowledge bases, often overlook the potential of dynamic historical information in ongoing conversations. To bridge this gap, we introduce DH-RAG, a Dynamic Historical Context-Powered Retrieval-Augmented Generation Method for Multi-Turn Dialogue. DH-RAG is inspired by human cognitive processes that utilize both long-term memory and immediate historical context in conversational responses \citep{stafford1987conversational}. DH-RAG is structured around two principal components: a History-Learning based Query Reconstruction Module, designed to generate effective queries by synthesizing current and prior interactions, and a Dynamic History Information Updating Module, which continually refreshes historical context throughout the dialogue. The center of DH-RAG is a Dynamic Historical Information database, which is further refined by three strategies within the Query Reconstruction Module: Historical Query Clustering, Hierarchical Matching, and Chain of Thought Tracking. Experimental evaluations show that DH-RAG significantly surpasses conventional models on several benchmarks, enhancing response relevance, coherence, and dialogue quality. 

**Abstract (ZH)**: 动态历史 context 增强的检索增强生成方法 (DH-RAG) 用于多轮对话 

---
# Mitigating Popularity Bias in Collaborative Filtering through Fair Sampling 

**Title (ZH)**: 通过公平采样缓解协同过滤中的流行性偏差 

**Authors**: Jiahao Liu, Dongsheng Li, Hansu Gu, Peng Zhang, Tun Lu, Li Shang, Ning Gu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13840)  

**Abstract**: Recommender systems often suffer from popularity bias, where frequently interacted items are overrepresented in recommendations. This bias stems from propensity factors influencing training data, leading to imbalanced exposure. In this paper, we introduce a Fair Sampling (FS) approach to address this issue by ensuring that both users and items are selected with equal probability as positive and negative instances. Unlike traditional inverse propensity score (IPS) methods, FS does not require propensity estimation, eliminating errors associated with inaccurate calculations. Our theoretical analysis demonstrates that FS effectively neutralizes the influence of propensity factors, achieving unbiased learning. Experimental results validate that FS outperforms state-of-the-art methods in both point-wise and pair-wise recommendation tasks, enhancing recommendation fairness without sacrificing accuracy. The implementation is available at this https URL. 

**Abstract (ZH)**: 推荐系统常常遭受流行性偏差的影响，其中高频互动的项目在推荐中过度代表性。这种偏差源于影响训练数据的倾向性因素，导致曝光不均。在本文中，我们引入了一种公平采样（FS）方法，通过确保用户和项目以相等概率被选为正例和负例来解决这一问题。与传统的逆倾向性得分（IPS）方法不同，FS 不需要进行倾向性估计，从而消除了与不准确计算相关的错误。我们的理论分析表明，FS 有效地抵消了倾向性因素的影响，实现了无偏学习。实验结果验证了在点wise和pairwise推荐任务中，FS 在提高推荐公平性的同时优于最先进的方法。实现代码可在以下链接获取：this https URL。 

---
# AnDB: Breaking Boundaries with an AI-Native Database for Universal Semantic Analysis 

**Title (ZH)**: AnDB：以AI原生数据库打破边界实现通用语义分析 

**Authors**: Tianqing Wang, Xun Xue, Guoliang Li, Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13805)  

**Abstract**: In this demonstration, we present AnDB, an AI-native database that supports traditional OLTP workloads and innovative AI-driven tasks, enabling unified semantic analysis across structured and unstructured data. While structured data analytics is mature, challenges remain in bridging the semantic gap between user queries and unstructured data. AnDB addresses these issues by leveraging cutting-edge AI-native technologies, allowing users to perform semantic queries using intuitive SQL-like statements without requiring AI expertise. This approach eliminates the ambiguity of traditional text-to-SQL systems and provides a seamless end-to-end optimization for analyzing all data types. AnDB automates query processing by generating multiple execution plans and selecting the optimal one through its optimizer, which balances accuracy, execution time, and financial cost based on user policies and internal optimizing mechanisms. AnDB future-proofs data management infrastructure, empowering users to effectively and efficiently harness the full potential of all kinds of data without starting from scratch. 

**Abstract (ZH)**: AnDB：一种支持传统OLTP工作负载和创新AI驱动任务的AI原生数据库，实现结构化和非结构化数据的统一语义分析。 

---
# Helix-mRNA: A Hybrid Foundation Model For Full Sequence mRNA Therapeutics 

**Title (ZH)**: 螺旋-mRNA：一种混合基础模型用于全长mRNA治疗剂 

**Authors**: Matthew Wood, Mathieu Klop, Maxime Allard  

**Link**: [PDF](https://arxiv.org/pdf/2502.13785)  

**Abstract**: mRNA-based vaccines have become a major focus in the pharmaceutical industry. The coding sequence as well as the Untranslated Regions (UTRs) of an mRNA can strongly influence translation efficiency, stability, degradation, and other factors that collectively determine a vaccine's effectiveness. However, optimizing mRNA sequences for those properties remains a complex challenge. Existing deep learning models often focus solely on coding region optimization, overlooking the UTRs. We present Helix-mRNA, a structured state-space-based and attention hybrid model to address these challenges. In addition to a first pre-training, a second pre-training stage allows us to specialise the model with high-quality data. We employ single nucleotide tokenization of mRNA sequences with codon separation, ensuring prior biological and structural information from the original mRNA sequence is not lost. Our model, Helix-mRNA, outperforms existing methods in analysing both UTRs and coding region properties. It can process sequences 6x longer than current approaches while using only 10% of the parameters of existing foundation models. Its predictive capabilities extend to all mRNA regions. We open-source the model (this https URL) and model weights (this https URL). 

**Abstract (ZH)**: 基于mRNA的疫苗已成为制药行业的重点。mRNA的编码序列及其未翻译区（UTRs）可以强烈影响翻译效率、稳定性、降解以及其他决定疫苗有效性的因素。然而，对这些属性进行优化仍然是一个复杂的挑战。现有的深度学习模型往往仅专注于编码区域优化，忽视了UTRs。我们提出Helix-mRNA，这是一种结构化状态空间和注意力机制结合的模型，以解决这些挑战。除了初始预训练外，还有一个额外的预训练阶段，使模型能够使用高质量数据专门化。我们使用单碱基 token 化方式处理mRNA序列，并进行密码子分割，确保原始mRNA序列的先前生物学和结构信息不丢失。我们的模型Helix-mRNA在分析UTRs和编码区域属性方面优于现有方法。它可以处理是现有方法6倍长的序列，同时仅使用现有基础模型10%的参数量，其预测能力涵盖了所有mRNA区域。我们开源了该模型（请参见此处：https://），以及模型权重（请参见此处：https://）。 

---
# Poster: SpiderSim: Multi-Agent Driven Theoretical Cybersecurity Simulation for Industrial Digitalization 

**Title (ZH)**: 海报：SpiderSim：基于多agent的工业数字化理论网络安全模拟系统 

**Authors**: Jiaqi Li, Xizhong Guo, Yang Zhao, Lvyang Zhang, Lidong Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2502.13778)  

**Abstract**: Rapid industrial digitalization has created intricate cybersecurity demands that necessitate effective validation methods. While cyber ranges and simulation platforms are widely deployed, they frequently face limitations in scenario diversity and creation efficiency. In this paper, we present SpiderSim, a theoretical cybersecurity simulation platform enabling rapid and lightweight scenario generation for industrial digitalization security research. At its core, our platform introduces three key innovations: a structured framework for unified scenario modeling, a multi-agent collaboration mechanism for automated generation, and modular atomic security capabilities for flexible scenario composition. Extensive implementation trials across multiple industrial digitalization contexts, including marine ranch monitoring systems, validate our platform's capacity for broad scenario coverage with efficient generation processes. Built on solid theoretical foundations and released as open-source software, SpiderSim facilitates broader research and development in automated security testing for industrial digitalization. 

**Abstract (ZH)**: 快速工业数字化创造了复杂的网络安全需求， necessitate有效的验证方法。尽管广泛应用了网络范围和模拟平台，它们经常在场景多样性和生成效率方面面临限制。在本文中，我们介绍了SpiderSim，这是一种理论上的网络安全模拟平台，用于工业数字化安全研究中的快速和轻量级场景生成。该平台的核心包含三项关键创新：统一场景建模的结构化框架、自动生成的多代理协作机制以及模块化原子安全能力，以实现灵活的场景组合。在包括海洋牧场监测系统等多种工业数字化背景下广泛的实施试验表明，该平台能够通过高效的生成过程实现广泛的场景覆盖。基于坚实的理论基础并在作为开源软件发布后，SpiderSim促进了自动化安全测试在工业数字化领域的更广泛研究和开发。 

---
# GPA: Grover Policy Agent for Generating Optimal Quantum Sensor Circuits 

**Title (ZH)**: GPA: 使用Grover算法的政策代理生成最优量子传感器电路 

**Authors**: Ahmad Alomari, Sathish A. P. Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.13755)  

**Abstract**: This study proposes a GPA for designing optimal Quantum Sensor Circuits (QSCs) to address complex quantum physics problems. The GPA consists of two parts: the Quantum Policy Evaluation (QPE) and the Quantum Policy Improvement (QPI). The QPE performs phase estimation to generate the search space, while the QPI utilizes Grover search and amplitude amplification techniques to efficiently identify an optimal policy that generates optimal QSCs. The GPA generates QSCs by selecting sequences of gates that maximize the Quantum Fisher Information (QFI) while minimizing the number of gates. The QSCs generated by the GPA are capable of producing entangled quantum states, specifically the squeezed states. High QFI indicates increased sensitivity to parameter changes, making the circuit useful for quantum state estimation and control tasks. Evaluation of the GPA on a QSC that consists of two qubits and a sequence of R_x, R_y, and S gates demonstrates its efficiency in generating optimal QSCs with a QFI of 1. Compared to existing quantum agents, the GPA achieves higher QFI with fewer gates, demonstrating a more efficient and scalable approach to the design of QSCs. This work illustrates the potential computational power of quantum agents for solving quantum physics problems 

**Abstract (ZH)**: 本研究提出了一种GPA，用于设计最优量子传感器电路（QSCs）以解决复杂量子物理学问题。GPA包括两部分：量子策略评估（QPE）和量子策略改进（QPI）。QPE执行相位估计以生成搜索空间，而QPI利用Grover搜索和振幅放大技术来高效地识别生成最优QSCs的最优策略。GPA通过选择最大化量子费雪信息（QFI）并最小化门的数量的门序列表来生成QSCs。由GPA生成的QSCs能够产生纠缠量子态，特别是压缩态。高QFI表明参数变化的敏感度增加，使电路适用于量子态估计和控制任务。对由两个量子位和一系列R_x、R_y和S门组成的QSC进行评估，展示了GPA在生成具有QFI为1的最优QSCs方面的高效性。与现有量子代理相比，GPA在更少的门数量下实现了更高的QFI，展示了设计QSCs更加高效和可扩展的方法。本工作展示了量子代理在解决量子物理学问题方面的潜在计算能力。 

---
# RobustX: Robust Counterfactual Explanations Made Easy 

**Title (ZH)**: RobustX: 简化 robust 反事实解释 

**Authors**: Junqi Jiang, Luca Marzari, Aaryan Purohit, Francesco Leofante  

**Link**: [PDF](https://arxiv.org/pdf/2502.13751)  

**Abstract**: The increasing use of Machine Learning (ML) models to aid decision-making in high-stakes industries demands explainability to facilitate trust. Counterfactual Explanations (CEs) are ideally suited for this, as they can offer insights into the predictions of an ML model by illustrating how changes in its input data may lead to different outcomes. However, for CEs to realise their explanatory potential, significant challenges remain in ensuring their robustness under slight changes in the scenario being explained. Despite the widespread recognition of CEs' robustness as a fundamental requirement, a lack of standardised tools and benchmarks hinders a comprehensive and effective comparison of robust CE generation methods. In this paper, we introduce RobustX, an open-source Python library implementing a collection of CE generation and evaluation methods, with a focus on the robustness property. RobustX provides interfaces to several existing methods from the literature, enabling streamlined access to state-of-the-art techniques. The library is also easily extensible, allowing fast prototyping of novel robust CE generation and evaluation methods. 

**Abstract (ZH)**: 机器学习模型在高风险行业辅助决策中的应用越来越多，亟需提高透明度以增加信任。反事实解释（CEs）在这方面尤为适用，因为它们可以揭示 machine learning 模型的预测结果，并展示输入数据变化可能带来的不同结果。然而，要充分发挥 CE 的解释潜力，仍需解决在解释场景稍有变化时保证其鲁棒性的重要挑战。尽管鲁棒性被广泛认为是 CE 的基本原则之一，但由于缺乏标准化的工具和基准，使得全面且有效的鲁棒 CE 生成方法比较变得困难。本文引入了 RobustX，这是一个开源 Python 库，实现了多种 CE 生成和评估方法，并特别关注鲁棒性这一特性。RobustX 提供了对文献中多种现有方法的接口，使得最先进的技术可以一键访问。该库还易于扩展，便于快速原型设计新的鲁棒 CE 生成和评估方法。 

---
# Secure Federated Data Distillation 

**Title (ZH)**: 安全联邦数据蒸馏 

**Authors**: Marco Arazzi, Mert Cihangiroglu, Serena Nicolazzo, Antonino Nocera  

**Link**: [PDF](https://arxiv.org/pdf/2502.13728)  

**Abstract**: Dataset Distillation (DD) is a powerful technique for reducing large datasets into compact, representative synthetic datasets, accelerating Machine Learning training. However, traditional DD methods operate in a centralized manner, which poses significant privacy threats and reduces its applicability. To mitigate these risks, we propose a Secure Federated Data Distillation framework (SFDD) to decentralize the distillation process while preserving this http URL existing Federated Distillation techniques that focus on training global models with distilled knowledge, our approach aims to produce a distilled dataset without exposing local contributions. We leverage the gradient-matching-based distillation method, adapting it for a distributed setting where clients contribute to the distillation process without sharing raw data. The central aggregator iteratively refines a synthetic dataset by integrating client-side updates while ensuring data confidentiality. To make our approach resilient to inference attacks perpetrated by the server that could exploit gradient updates to reconstruct private data, we create an optimized Local Differential Privacy approach, called LDPO-RLD (Label Differential Privacy Obfuscation via Randomized Linear Dispersion). Furthermore, we assess the framework's resilience against malicious clients executing backdoor attacks and demonstrate robustness under the assumption of a sufficient number of participating clients. Our experimental results demonstrate the effectiveness of SFDD and that the proposed defense concretely mitigates the identified vulnerabilities, with minimal impact on the performance of the distilled dataset. By addressing the interplay between privacy and federation in dataset distillation, this work advances the field of privacy-preserving Machine Learning making our SFDD framework a viable solution for sensitive data-sharing applications. 

**Abstract (ZH)**: Secure Federated Data Distillation Framework (SFDD) 

---
# MoM: Linear Sequence Modeling with Mixture-of-Memories 

**Title (ZH)**: MoM：基于混合记忆的线性序列建模 

**Authors**: Jusen Du, Weigao Sun, Disen Lan, Jiaxi Hu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.13685)  

**Abstract**: Linear sequence modeling methods, such as linear attention, state space modeling, and linear RNNs, offer significant efficiency improvements by reducing the complexity of training and inference. However, these methods typically compress the entire input sequence into a single fixed-size memory state, which leads to suboptimal performance on recall-intensive downstream tasks. Drawing inspiration from neuroscience, particularly the brain's ability to maintain robust long-term memory while mitigating "memory interference", we introduce a novel architecture called Mixture-of-Memories (MoM). MoM utilizes multiple independent memory states, with a router network directing input tokens to specific memory states. This approach greatly enhances the overall memory capacity while minimizing memory interference. As a result, MoM performs exceptionally well on recall-intensive tasks, surpassing existing linear sequence modeling techniques. Despite incorporating multiple memory states, the computation of each memory state remains linear in complexity, allowing MoM to retain the linear-complexity advantage during training, while constant-complexity during inference. Our experimental results show that MoM significantly outperforms current linear sequence models on downstream language tasks, particularly recall-intensive tasks, and even achieves performance comparable to Transformer models. The code is released at this https URL and is also released as a part of this https URL. 

**Abstract (ZH)**: 基于记忆混合的线性序列建模方法 

---
# PeerQA: A Scientific Question Answering Dataset from Peer Reviews 

**Title (ZH)**: PeerQA：来自同行评审的科学问答数据集 

**Authors**: Tim Baumgärtner, Ted Briscoe, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2502.13668)  

**Abstract**: We present PeerQA, a real-world, scientific, document-level Question Answering (QA) dataset. PeerQA questions have been sourced from peer reviews, which contain questions that reviewers raised while thoroughly examining the scientific article. Answers have been annotated by the original authors of each paper. The dataset contains 579 QA pairs from 208 academic articles, with a majority from ML and NLP, as well as a subset of other scientific communities like Geoscience and Public Health. PeerQA supports three critical tasks for developing practical QA systems: Evidence retrieval, unanswerable question classification, and answer generation. We provide a detailed analysis of the collected dataset and conduct experiments establishing baseline systems for all three tasks. Our experiments and analyses reveal the need for decontextualization in document-level retrieval, where we find that even simple decontextualization approaches consistently improve retrieval performance across architectures. On answer generation, PeerQA serves as a challenging benchmark for long-context modeling, as the papers have an average size of 12k tokens. Our code and data is available at this https URL. 

**Abstract (ZH)**: PeerQA：一个基于实际应用的科学文献级别问答数据集 

---
# Integrating Inverse and Forward Modeling for Sparse Temporal Data from Sensor Networks 

**Title (ZH)**: 将逆向建模与正向建模集成用于传感器网络的稀疏时序数据 

**Authors**: Julian Vexler, Björn Vieten, Martin Nelke, Stefan Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2502.13638)  

**Abstract**: We present CavePerception, a framework for the analysis of sparse data from sensor networks that incorporates elements of inverse modeling and forward modeling. By integrating machine learning with physical modeling in a hypotheses space, we aim to improve the interpretability of sparse, noisy, and potentially incomplete sensor data. The framework assumes data from a two-dimensional sensor network laid out in a graph structure that detects certain objects, with certain motion patterns. Examples of such sensors are magnetometers. Given knowledge about the objects and the way they act on the sensors, one can develop a data generator that produces data from simulated motions of the objects across the sensor field. The framework uses the simulated data to infer object behaviors across the sensor network. The approach is experimentally tested on real-world data, where magnetometers are used on an airport to detect and identify aircraft motions. Experiments demonstrate the value of integrating inverse and forward modeling, enabling intelligent systems to better understand and predict complex, sensor-driven events. 

**Abstract (ZH)**: CavePerception：一种将逆向建模与前向建模结合分析稀疏传感器网络数据的框架 

---
# Decentralized Planning Using Probabilistic Hyperproperties 

**Title (ZH)**: 基于概率超性质的分布式规划 

**Authors**: Francesco Pontiggia, Filip Macák, Roman Andriushchenko, Michele Chiari, Milan Češka  

**Link**: [PDF](https://arxiv.org/pdf/2502.13621)  

**Abstract**: Multi-agent planning under stochastic dynamics is usually formalised using decentralized (partially observable) Markov decision processes ( MDPs) and reachability or expected reward specifications. In this paper, we propose a different approach: we use an MDP describing how a single agent operates in an environment and probabilistic hyperproperties to capture desired temporal objectives for a set of decentralized agents operating in the environment. We extend existing approaches for model checking probabilistic hyperproperties to handle temporal formulae relating paths of different agents, thus requiring the self-composition between multiple MDPs. Using several case studies, we demonstrate that our approach provides a flexible and expressive framework to broaden the specification capabilities with respect to existing planning techniques. Additionally, we establish a close connection between a subclass of probabilistic hyperproperties and planning for a particular type of Dec-MDPs, for both of which we show undecidability. This lays the ground for the use of existing decentralized planning tools in the field of probabilistic hyperproperty verification. 

**Abstract (ZH)**: 在随机动力学下的多Agent规划通常使用去中心化（部分可观测的）马尔科夫决策过程（MDP）和可达性或期望奖励规范进行形式化。本文提出了一种不同的方法：我们使用描述单个Agent在环境中的操作的MDP，并使用概率超性质来捕获一组在环境中操作的去中心化Agent的期望时间目标。我们将现有的概率超性质模型检查方法扩展为处理连接不同Agent路径的时序公式，从而需要处理多个MDP之间的自我组合。通过几个案例研究，我们证明了该方法提供了一个灵活且表达能力强的框架，可以扩展现有的规划技术的规范能力。此外，我们建立了概率超性质的一个子类与特定类型Dec-MDP规划之间的密切联系，对于这两种情况我们都证明了不可判定性。这为在概率超性质验证领域使用现有的去中心化规划工具奠定了基础。 

---
# Beyond One-Size-Fits-All: Tailored Benchmarks for Efficient Evaluation 

**Title (ZH)**: 超越一刀切：针对高效的评价定制基准 

**Authors**: Peiwen Yuan, Yueqi Zhang, Shaoxiong Feng, Yiwei Li, Xinglin Wang, Jiayi Shi, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.13576)  

**Abstract**: Evaluating models on large benchmarks is very resource-intensive, especially during the period of rapid model evolution. Existing efficient evaluation methods estimate the performance of target models by testing them only on a small and static coreset of the benchmark, which is derived from the publicly available evaluation results of source models. These methods rely on the assumption that target models have high prediction consistency with source models. However, we demonstrate that it doesn't generalize well in practice. To alleviate the inconsistency issue, we present TailoredBench, a method that conducts customized evaluation tailored to each target model. Specifically, a Global-coreset is first constructed as a probe to identify the most consistent source models for each target model with an adaptive source model selection strategy. Afterwards, a scalable K-Medoids clustering algorithm is proposed to extend the Global-coreset to a tailored Native-coreset for each target model. According to the predictions on Native-coresets, we obtain the performance of target models on the whole benchmark with a calibrated estimation strategy. Comprehensive experiments on 5 benchmarks across over 300 models demonstrate that compared to best performing baselines, TailoredBench achieves an average reduction of 31.4% in MAE of accuracy estimates under the same inference budgets, showcasing strong effectiveness and generalizability. 

**Abstract (ZH)**: 基于定制化评估的TailoredBench：在大基准上评估模型的方法 

---
# Solving the Encoding Bottleneck: Of the HHL Algorithm, By the HHL Algorithm 

**Title (ZH)**: 解决编码瓶颈：借助HHL算法 

**Authors**: Guang Ping He  

**Link**: [PDF](https://arxiv.org/pdf/2502.13534)  

**Abstract**: The Harrow-Hassidim-Lloyd (HHL) algorithm offers exponential speedup for solving the quantum linear-system problem. But some caveats for the speedup could be hard to met. One of the difficulties is the encoding bottleneck, i.e., the efficient preparation of the initial quantum state. To prepare an arbitrary $N$-dimensional state exactly, existing state-preparation approaches generally require a runtime of $O(N)$, which will ruin the speedup of the HHL algorithm. Here we show that the states can be prepared approximately with a runtime of $O(poly(\log N))$ by employing a slightly modified version of the HHL algorithm itself. Thus, applying this approach to prepare the initial state of the original HHL algorithm can preserve the exponential speedup advantage. It can also serve as a standalone solution for other applications demanding rapid state preparation. 

**Abstract (ZH)**: Harrow-Hassidim-Lloyd (HHL) 算法提供了求解量子线性系统问题的指数级加速，但加速的效果可能难以实现。一个主要的难点是编码瓶颈，即高效制备初始量子态。为了精确制备一个 $N$ 维状态，现有方法通常需要 $O(N)$ 的运行时间，这将破坏 HHL 算法的加速优势。我们通过使用 HHL 算法的略微修改版本，可以实现以 $O(\text{poly}(\log N))$ 的运行时间近似制备所需状态，从而可以应用于原始 HHL 算法的初始态制备，以保持其指数级加速的优势。此外，该方法也可作为其他需要快速制备状态的应用的独立解决方案。 

---
# Astra: Efficient and Money-saving Automatic Parallel Strategies Search on Heterogeneous GPUs 

**Title (ZH)**: Astra: 在异构GPU上高效且节省成本的自动并行策略搜索方法 

**Authors**: Peiran Wang, Haibing Li, Fu Haohan, Shiyong Li, Yanpeng Wang, Dou Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13480)  

**Abstract**: In this paper, we introduce an efficient and money-saving automatic parallel strategies search framework on heterogeneous GPUs: Astra. First, Astra searches for the efficiency-optimal parallel strategy in both GPU configurations search space (GPU types and GPU numbers) and parallel parameters search space. Then, Astra also provides the solution on heterogeneous GPUs by mathematically modeling the time consumption of heterogeneous training. At last, Astra is the first to propose the automatic parallel strategy search on money-saving. The experiment results demonstrate that Astra can achieve better throughput than expert-designed strategies. The search time cost for Astra can also be limited to 1.27 seconds in a single-GPU setting and less than 1.35 minutes in a heterogeneous-GPU setting on average with an accuracy of over 95%. 

**Abstract (ZH)**: 基于异构GPU的高效经济型自动并行策略搜索框架Astra 

---
# Some Insights of Construction of Feature Graph to Learn Pairwise Feature Interactions with Graph Neural Networks 

**Title (ZH)**: 基于图神经网络学习成对特征交互的特征图构建一些见解 

**Authors**: Phaphontee Yamchote, Saw Nay Htet Win, Chainarong Amornbunchornvej, Thanapon Noraset  

**Link**: [PDF](https://arxiv.org/pdf/2502.13471)  

**Abstract**: Feature interaction is crucial in predictive machine learning models, as it captures the relationships between features that influence model performance. In this work, we focus on pairwise interactions and investigate their importance in constructing feature graphs for Graph Neural Networks (GNNs). Rather than proposing new methods, we leverage existing GNN models and tools to explore the relationship between feature graph structures and their effectiveness in modeling interactions. Through experiments on synthesized datasets, we uncover that edges between interacting features are important for enabling GNNs to model feature interactions effectively. We also observe that including non-interaction edges can act as noise, degrading model performance. Furthermore, we provide theoretical support for sparse feature graph selection using the Minimum Description Length (MDL) principle. We prove that feature graphs retaining only necessary interaction edges yield a more efficient and interpretable representation than complete graphs, aligning with Occam's Razor.
Our findings offer both theoretical insights and practical guidelines for designing feature graphs that improve the performance and interpretability of GNN models. 

**Abstract (ZH)**: 特征相互作用对于预测机器学习模型至关重要，因为它捕获了影响模型性能的特征之间的关系。在本工作中，我们关注于成对的相互作用，并探究它们在构建图神经网络（GNN）特征图结构中的重要性。我们没有提出新的方法，而是利用现有的GNN模型和工具来探索特征图结构与其在建模相互作用方面的有效性之间的关系。通过对合成数据集的实验，我们发现，相互作用特征之间的边对于使GNN能够有效地建模特征相互作用至关重要。我们还观察到包含非相互作用边可能会增加噪声，从而降低模型性能。此外，我们利用最小描述长度（MDL）原则为稀疏特征图选择提供理论支持。我们证明，保留仅必要相互作用边的特征图比完整图提供了更高效和可解释的表示，这与奥卡姆剃刀原则一致。我们的发现不仅提供了设计改进GNN模型性能和可解释性的特征图的理论洞察和实践指南。 

---
# HawkBench: Investigating Resilience of RAG Methods on Stratified Information-Seeking Tasks 

**Title (ZH)**: HawkBench: 探究RAG方法在分层信息检索任务中的鲁棒性 

**Authors**: Hongjin Qian, Zheng Liu, Chao Gao, Yankai Wang, Defu Lian, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2502.13465)  

**Abstract**: In real-world information-seeking scenarios, users have dynamic and diverse needs, requiring RAG systems to demonstrate adaptable resilience. To comprehensively evaluate the resilience of current RAG methods, we introduce HawkBench, a human-labeled, multi-domain benchmark designed to rigorously assess RAG performance across categorized task types. By stratifying tasks based on information-seeking behaviors, HawkBench provides a systematic evaluation of how well RAG systems adapt to diverse user needs.
Unlike existing benchmarks, which focus primarily on specific task types (mostly factoid queries) and rely on varying knowledge bases, HawkBench offers: (1) systematic task stratification to cover a broad range of query types, including both factoid and rationale queries, (2) integration of multi-domain corpora across all task types to mitigate corpus bias, and (3) rigorous annotation for high-quality evaluation.
HawkBench includes 1,600 high-quality test samples, evenly distributed across domains and task types. Using this benchmark, we evaluate representative RAG methods, analyzing their performance in terms of answer quality and response latency. Our findings highlight the need for dynamic task strategies that integrate decision-making, query interpretation, and global knowledge understanding to improve RAG generalizability. We believe HawkBench serves as a pivotal benchmark for advancing the resilience of RAG methods and their ability to achieve general-purpose information seeking. 

**Abstract (ZH)**: HawkBench：一个多领域的人工标注基准，用于全面评估RAG系统的适应性韧性 

---
# Interleaved Gibbs Diffusion for Constrained Generation 

**Title (ZH)**: 交替吉布斯扩散约束生成 

**Authors**: Gautham Govind Anil, Sachin Yadav, Dheeraj Nagaraj, Karthikeyan Shanmugam, Prateek Jain  

**Link**: [PDF](https://arxiv.org/pdf/2502.13450)  

**Abstract**: We introduce Interleaved Gibbs Diffusion (IGD), a novel generative modeling framework for mixed continuous-discrete data, focusing on constrained generation problems. Prior works on discrete and continuous-discrete diffusion models assume factorized denoising distribution for fast generation, which can hinder the modeling of strong dependencies between random variables encountered in constrained generation. IGD moves beyond this by interleaving continuous and discrete denoising algorithms via a discrete time Gibbs sampling type Markov chain. IGD provides flexibility in the choice of denoisers, allows conditional generation via state-space doubling and inference time scaling via the ReDeNoise method. Empirical evaluations on three challenging tasks-solving 3-SAT, generating molecule structures, and generating layouts-demonstrate state-of-the-art performance. Notably, IGD achieves a 7% improvement on 3-SAT out of the box and achieves state-of-the-art results in molecule generation without relying on equivariant diffusion or domain-specific architectures. We explore a wide range of modeling, and interleaving strategies along with hyperparameters in each of these problems. 

**Abstract (ZH)**: 交错吉布斯扩散（IGD）：一种针对混合连续-离散数据的生成 modeling 框架及其应用 

---
# MCTS-KBQA: Monte Carlo Tree Search for Knowledge Base Question Answering 

**Title (ZH)**: MCTS-KBQA: 针对知识库问答的蒙特卡洛树搜索 

**Authors**: Guanming Xiong, Haochen Li, Wen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.13428)  

**Abstract**: This study explores how to enhance the reasoning capabilities of large language models (LLMs) in knowledge base question answering (KBQA) by leveraging Monte Carlo Tree Search (MCTS). Semantic parsing-based KBQA methods are particularly challenging as these approaches require locating elements from knowledge bases and generating logical forms, demanding not only extensive annotated data but also strong reasoning capabilities. Although recent approaches leveraging LLMs as agents have demonstrated considerable potential, these studies are inherently constrained by their linear decision-making processes. To address this limitation, we propose a MCTS-based framework that enhances LLMs' reasoning capabilities through tree search methodology. We design a carefully designed step-wise reward mechanism that requires only direct prompting of open-source instruction LLMs without additional fine-tuning. Experimental results demonstrate that our approach significantly outperforms linear decision-making methods, particularly in low-resource scenarios. Additionally, we contribute new data resources to the KBQA community by annotating intermediate reasoning processes for existing question-SPARQL datasets using distant supervision. Experimental results on the extended dataset demonstrate that our method achieves comparable performance to fully supervised models while using significantly less training data. 

**Abstract (ZH)**: 本研究通过利用蒙特卡洛树搜索（MCTS）探索如何增强大型语言模型（LLMs）在知识库问答（KBQA）中的推理能力。基于语义解析的KBQA方法尤其具有挑战性，因为这些方法要求从知识库中定位元素并生成逻辑形式，不仅需要大量的标注数据，还要求较强的推理能力。尽管最近利用LLMs作为代理的方法显示了巨大的潜力，但这些研究本质上受限于其线性的决策过程。为了解决这一限制，我们提出了一种基于MCTS的框架，通过树搜索方法增强LLMs的推理能力。我们设计了一种精心设计的逐步奖励机制，仅需直接提示开源指令LLMs，无需额外微调。实验结果表明，我们的方法在低资源场景下显著优于线性决策方法。此外，我们通过使用远程监督为现有的问答-SPARQL数据集标注中间推理过程，为KBQA社区贡献了新的数据资源。实验结果表明，我们的方法在使用显著较少训练数据的情况下实现了与完全监督模型相当的性能。 

---
# Tell Me Why: Incentivizing Explanations 

**Title (ZH)**: 告诉我原因：激励解释 

**Authors**: Siddarth Srinivasan, Ezra Karger, Michiel Bakker, Yiling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13410)  

**Abstract**: Common sense suggests that when individuals explain why they believe something, we can arrive at more accurate conclusions than when they simply state what they believe. Yet, there is no known mechanism that provides incentives to elicit explanations for beliefs from agents. This likely stems from the fact that standard Bayesian models make assumptions (like conditional independence of signals) that preempt the need for explanations, in order to show efficient information aggregation. A natural justification for the value of explanations is that agents' beliefs tend to be drawn from overlapping sources of information, so agents' belief reports do not reveal all that needs to be known. Indeed, this work argues that rationales-explanations of an agent's private information-lead to more efficient aggregation by allowing agents to efficiently identify what information they share and what information is new. Building on this model of rationales, we present a novel 'deliberation mechanism' to elicit rationales from agents in which truthful reporting of beliefs and rationales is a perfect Bayesian equilibrium. 

**Abstract (ZH)**: 常识表明，当个体解释他们为什么相信某事时，我们比仅仅陈述他们的信念可以获得更准确的结论。然而，目前尚无机制激励个体提供信念的解释。这可能源于标准贝叶斯模型基于（如信号的条件独立性）假设来预先排除解释的需求，以便展示信息聚合的效率。解释的价值自然在于个体的信念往往源自重叠的信息来源，因此个体的信念报告并未揭示所有需要了解的信息。事实上，本文认为，解释——即个体的理性说明——通过使个体有效识别共享信息与新信息，促进了更高效的聚合。在此模型的基础上，我们提出了一种新颖的“反思机制”，以激励个体报告信念及其解释，诚实行事的信念和解释是完美的贝叶斯均衡。 

---
# JL1-CD: A New Benchmark for Remote Sensing Change Detection and a Robust Multi-Teacher Knowledge Distillation Framework 

**Title (ZH)**: JL1-CD：一个新的遥感变化检测基准以及一个稳健的多师知识蒸馏框架 

**Authors**: Ziyuan Liu, Ruifei Zhu, Long Gao, Yuanxiu Zhou, Jingyu Ma, Yuantao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13407)  

**Abstract**: Deep learning has achieved significant success in the field of remote sensing image change detection (CD), yet two major challenges remain: the scarcity of sub-meter, all-inclusive open-source CD datasets, and the difficulty of achieving consistent and satisfactory detection results across images with varying change areas. To address these issues, we introduce the JL1-CD dataset, which contains 5,000 pairs of 512 x 512 pixel images with a resolution of 0.5 to 0.75 meters. Additionally, we propose a multi-teacher knowledge distillation (MTKD) framework for CD. Experimental results on the JL1-CD and SYSU-CD datasets demonstrate that the MTKD framework significantly improves the performance of CD models with various network architectures and parameter sizes, achieving new state-of-the-art results. The code is available at this https URL. 

**Abstract (ZH)**: 深学习在遥感图像变化检测领域的应用取得了显著成功，但仍面临两个主要挑战：亚米级全面的开源变化检测数据集稀缺，以及在变化区域变化的图像中实现一致且满意的检测结果的难度。为此，我们介绍了JL1-CD数据集，该数据集包含5,000对分辨率为0.5至0.75米的512×512像素图像。此外，我们提出了多师知识蒸馏（MTKD）框架用于变化检测。在JL1-CD和SYSU-CD数据集上的实验结果表明，MTKD框架显著提高了各种网络架构和参数量的变化检测模型性能，并取得了新的最佳结果。相关代码可访问此链接。 

---
# Learning Symbolic Task Decompositions for Multi-Agent Teams 

**Title (ZH)**: 学习符号化任务分解的多Agent团队策略 

**Authors**: Ameesh Shah, Niklas Lauffer, Thomas Chen, Nikhil Pitta, Sanjit A. Seshia  

**Link**: [PDF](https://arxiv.org/pdf/2502.13376)  

**Abstract**: One approach for improving sample efficiency in cooperative multi-agent learning is to decompose overall tasks into sub-tasks that can be assigned to individual agents. We study this problem in the context of reward machines: symbolic tasks that can be formally decomposed into sub-tasks. In order to handle settings without a priori knowledge of the environment, we introduce a framework that can learn the optimal decomposition from model-free interactions with the environment. Our method uses a task-conditioned architecture to simultaneously learn an optimal decomposition and the corresponding agents' policies for each sub-task. In doing so, we remove the need for a human to manually design the optimal decomposition while maintaining the sample-efficiency benefits of improved credit assignment. We provide experimental results in several deep reinforcement learning settings, demonstrating the efficacy of our approach. Our results indicate that our approach succeeds even in environments with codependent agent dynamics, enabling synchronous multi-agent learning not achievable in previous works. 

**Abstract (ZH)**: 改善合作多智能体学习样本效率的一种方法是将整体任务分解为可以分配给单个智能体的子任务：基于奖励机器的形式化可分解符号任务中任务分解问题的研究。为了处理先验环境知识未知的设置，我们引入了一种可以从无模型交互中学习最优分解的框架。我们的方法使用任务条件化架构同时学习最优分解及其对应的每个子任务的智能体策略，从而去除手动设计最优分解的需要，同时保持改进的奖励归因效益。我们在多个深度强化学习设置中提供实验结果，展示了我们方法的有效性。我们的结果表明，我们的方法即使在存在相互依赖智能体动力学的环境中也能成功，从而使得同步多智能体学习成为可能，这在之前的工作中是不可实现的。 

---
# Secure and Efficient Watermarking for Latent Diffusion Models in Model Distribution Scenarios 

**Title (ZH)**: 在模型分发场景中具有安全性和高效性的潜扩散模型水印技术 

**Authors**: Liangqi Lei, Keke Gai, Jing Yu, Liehuang Zhu, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13345)  

**Abstract**: Latent diffusion models have exhibited considerable potential in generative tasks. Watermarking is considered to be an alternative to safeguard the copyright of generative models and prevent their misuse. However, in the context of model distribution scenarios, the accessibility of models to large scale of model users brings new challenges to the security, efficiency and robustness of existing watermark solutions. To address these issues, we propose a secure and efficient watermarking solution. A new security mechanism is designed to prevent watermark leakage and watermark escape, which considers watermark randomness and watermark-model association as two constraints for mandatory watermark injection. To reduce the time cost of training the security module, watermark injection and the security mechanism are decoupled, ensuring that fine-tuning VAE only accomplishes the security mechanism without the burden of learning watermark patterns. A watermark distribution-based verification strategy is proposed to enhance the robustness against diverse attacks in the model distribution scenarios. Experimental results prove that our watermarking consistently outperforms existing six baselines on effectiveness and robustness against ten image processing attacks and adversarial attacks, while enhancing security in the distribution scenarios. 

**Abstract (ZH)**: 潜在扩散模型在生成任务中展现出显著潜力。水印技术被视为保护生成模型版权和防止其滥用的一种替代方案。然而，在模型分发场景下，模型对大规模用户群体的可访问性带来了新的挑战，对现有水印解决方案的安全性、效率和鲁棒性提出了新的要求。为应对这些挑战，我们提出了一种安全高效的水印解决方案。设计了一种新的安全机制，以防止水印泄露和水印逃逸，并将水印随机性和水印-模型关联作为强制性水印注入的两大约束条件。通过将水印注入和安全机制解耦，减少了训练安全模块所需的时间成本，确保仅通过微调VAE即可实现安全机制，而不必学习水印模式。提出了基于水印分发的验证策略，以增强模型分发场景下的鲁棒性。实验结果证明，我们的水印技术在有效性及对十种图像处理攻击和对抗攻击的鲁棒性方面，均优于现有六种基线方法，同时在分发场景中增强了安全性。 

---
# How Expressive are Knowledge Graph Foundation Models? 

**Title (ZH)**: 知识图谱基础模型的表现力如何？ 

**Authors**: Xingyue Huang, Pablo Barceló, Michael M. Bronstein, İsmail İlkan Ceylan, Mikhail Galkin, Juan L Reutter, Miguel Romero Orth  

**Link**: [PDF](https://arxiv.org/pdf/2502.13339)  

**Abstract**: Knowledge Graph Foundation Models (KGFMs) are at the frontier for deep learning on knowledge graphs (KGs), as they can generalize to completely novel knowledge graphs with different relational vocabularies. Despite their empirical success, our theoretical understanding of KGFMs remains very limited. In this paper, we conduct a rigorous study of the expressive power of KGFMs. Specifically, we show that the expressive power of KGFMs directly depends on the motifs that are used to learn the relation representations. We then observe that the most typical motifs used in the existing literature are binary, as the representations are learned based on how pairs of relations interact, which limits the model's expressiveness. As part of our study, we design more expressive KGFMs using richer motifs, which necessitate learning relation representations based on, e.g., how triples of relations interact with each other. Finally, we empirically validate our theoretical findings, showing that the use of richer motifs results in better performance on a wide range of datasets drawn from different domains. 

**Abstract (ZH)**: 知识图谱基础模型的知识图谱表示能力研究：从二元模式到更丰富的模式 

---
# Adjust for Trust: Mitigating Trust-Induced Inappropriate Reliance on AI Assistance 

**Title (ZH)**: 调整信任：减轻由信任引起的不当依赖人工智能辅助问题 

**Authors**: Tejas Srinivasan, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2502.13321)  

**Abstract**: Trust biases how users rely on AI recommendations in AI-assisted decision-making tasks, with low and high levels of trust resulting in increased under- and over-reliance, respectively. We propose that AI assistants should adapt their behavior through trust-adaptive interventions to mitigate such inappropriate reliance. For instance, when user trust is low, providing an explanation can elicit more careful consideration of the assistant's advice by the user. In two decision-making scenarios -- laypeople answering science questions and doctors making medical diagnoses -- we find that providing supporting and counter-explanations during moments of low and high trust, respectively, yields up to 38% reduction in inappropriate reliance and 20% improvement in decision accuracy. We are similarly able to reduce over-reliance by adaptively inserting forced pauses to promote deliberation. Our results highlight how AI adaptation to user trust facilitates appropriate reliance, presenting exciting avenues for improving human-AI collaboration. 

**Abstract (ZH)**: 信任偏差：用户在AI辅助决策任务中依赖AI推荐的程度，低信任和高信任分别导致不当依赖的增加和减少。我们提出，AI助手应通过信任自适应干预来调整其行为，以减轻这种不当依赖。例如，当用户信任度较低时，提供解释可以促使用户更认真地考虑助手的建议。在两种决策场景——普通人在回答科学问题和医生进行医疗诊断中——我们发现，在低信任和高信任时刻分别提供支持性解释和反驳性解释，可将不当依赖减少最多38%，决策准确性提高20%。我们还能够通过适应性插入强制暂停来促进深思熟虑，从而减少过度依赖。我们的研究结果显示，AI对用户信任的适应性调整有助于实现适当的依赖，为改善人机协作提供了令人兴奋的新方向。 

---
# Understanding and Tackling Label Errors in Individual-Level Nature Language Understanding 

**Title (ZH)**: 理解并解决个体水平自然语言理解中的标签错误问题 

**Authors**: Yunpeng Xiao, Youpeng Zhao, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13297)  

**Abstract**: Natural language understanding (NLU) is a task that enables machines to understand human language. Some tasks, such as stance detection and sentiment analysis, are closely related to individual subjective perspectives, thus termed individual-level NLU. Previously, these tasks are often simplified to text-level NLU tasks, ignoring individual factors. This not only makes inference difficult and unexplainable but often results in a large number of label errors when creating datasets. To address the above limitations, we propose a new NLU annotation guideline based on individual-level factors. Specifically, we incorporate other posts by the same individual and then annotate individual subjective perspectives after considering all individual posts. We use this guideline to expand and re-annotate the stance detection and topic-based sentiment analysis datasets. We find that error rates in the samples were as high as 31.7\% and 23.3\%. We further use large language models to conduct experiments on the re-annotation datasets and find that the large language models perform well on both datasets after adding individual factors. Both GPT-4o and Llama3-70B can achieve an accuracy greater than 87\% on the re-annotation datasets. We also verify the effectiveness of individual factors through ablation studies. We call on future researchers to add individual factors when creating such datasets. Our re-annotation dataset can be found at this https URL 

**Abstract (ZH)**: 基于个体因素的自然语言理解标注指南：扩展与重注释立场检测和主题相关的情感分析数据集 

---
# Prediction of Clinical Complication Onset using Neural Point Processes 

**Title (ZH)**: 使用神经点过程预测临床并发症的发生 

**Authors**: Sachini Weerasekara, Sagar Kamarthi, Jacqueline Isaacs  

**Link**: [PDF](https://arxiv.org/pdf/2502.13290)  

**Abstract**: Predicting medical events in advance within critical care settings is paramount for patient outcomes and resource management. Utilizing predictive models, healthcare providers can anticipate issues such as cardiac arrest, sepsis, or respiratory failure before they manifest. Recently, there has been a surge in research focusing on forecasting adverse medical event onsets prior to clinical manifestation using machine learning. However, while these models provide temporal prognostic predictions for the occurrence of a specific adverse event of interest within defined time intervals, their interpretability often remains a challenge. In this work, we explore the applicability of neural temporal point processes in the context of adverse event onset prediction, with the aim of explaining clinical pathways and providing interpretable insights. Our experiments span six state-of-the-art neural point processes and six critical care datasets, each focusing on the onset of distinct adverse events. This work represents a novel application class of neural temporal point processes in event prediction. 

**Abstract (ZH)**: 在重症监护环境中提前预测医疗事件对于患者的预后和资源管理至关重要。利用预测模型，医疗提供者可以在临床表现之前预见诸如心搏骤停、感染性休克或呼吸衰竭等问题。近年来，研究关注使用机器学习在临床表现之前预测不良医疗事件的出现，虽然这些模型可以在特定时间区间内提供关于特定不利事件发生的时序预估预测，但其可解释性常常仍是挑战。在这项研究中，我们探索了神经时间点过程在不良事件出现预测中的应用，旨在解释临床路径并提供可解释的见解。我们的实验涵盖了六种最先进的神经时间点过程和六种重症监护数据集，每个数据集都侧重于特定不良事件的出现。这项工作代表了神经时间点过程在事件预测中的一种新应用类别。 

---
# Performance Evaluation of Sentiment Analysis on Text and Emoji Data Using End-to-End, Transfer Learning, Distributed and Explainable AI Models 

**Title (ZH)**: 基于端到端、迁移学习、分布式和可解释AI模型的情感分析在文本和Emoji数据上的性能评估 

**Authors**: Sirisha Velampalli, Chandrashekar Muniyappa, Ashutosh Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2502.13278)  

**Abstract**: Emojis are being frequently used in todays digital world to express from simple to complex thoughts more than ever before. Hence, they are also being used in sentiment analysis and targeted marketing campaigns. In this work, we performed sentiment analysis of Tweets as well as on emoji dataset from the Kaggle. Since tweets are sentences we have used Universal Sentence Encoder (USE) and Sentence Bidirectional Encoder Representations from Transformers (SBERT) end-to-end sentence embedding models to generate the embeddings which are used to train the Standard fully connected Neural Networks (NN), and LSTM NN models. We observe the text classification accuracy was almost the same for both the models around 98 percent. On the contrary, when the validation set was built using emojis that were not present in the training set then the accuracy of both the models reduced drastically to 70 percent. In addition, the models were also trained using the distributed training approach instead of a traditional singlethreaded model for better scalability. Using the distributed training approach, we were able to reduce the run-time by roughly 15% without compromising on accuracy. Finally, as part of explainable AI the Shap algorithm was used to explain the model behaviour and check for model biases for the given feature set. 

**Abstract (ZH)**: emojis在当今数字世界中被频繁用于表达从简单到复杂的thoughts，其使用频率超过以往。因此，它们也被用于情感分析和目标营销活动。本文对推文以及来自Kaggle的emoji数据集进行了情感分析。由于推文是句子，我们使用了全局句子编码器（USE）和双向Transformer句子表示（SBERT）端到端句子嵌入模型生成嵌入，并使用这些嵌入训练标准全连接神经网络（NN）和LSTM NN模型。我们观察到，两种模型的文本分类准确率几乎相同，约为98%。然而，当验证集使用训练集中未出现的emoji构建时，两种模型的准确率骤降至70%。此外，我们还使用分布式训练方法而不是传统的单线程模型来训练模型，以提高可扩展性。使用分布式训练方法，我们能够在不牺牲准确性的前提下将运行时间减少约15%。最后，作为可解释AI的一部分，我们使用Shap算法解释模型行为并检查给定特征集的模型偏差。 

---
# A Survey of Anomaly Detection in Cyber-Physical Systems 

**Title (ZH)**: 网络物理系统中的异常检测综述 

**Authors**: Danial Abshari, Meera Sridhar  

**Link**: [PDF](https://arxiv.org/pdf/2502.13256)  

**Abstract**: In our increasingly interconnected world, Cyber-Physical Systems (CPS) play a crucial role in industries like healthcare, transportation, and manufacturing by combining physical processes with computing power. These systems, however, face many challenges, especially regarding security and system faults. Anomalies in CPS may indicate unexpected problems, from sensor malfunctions to cyber-attacks, and must be detected to prevent failures that can cause harm or disrupt services. This paper provides an overview of the different ways researchers have approached anomaly detection in CPS. We categorize and compare methods like machine learning, deep learning, mathematical models, invariant, and hybrid techniques. Our goal is to help readers understand the strengths and weaknesses of these methods and how they can be used to create safer, more reliable CPS. By identifying the gaps in current solutions, we aim to encourage future research that will make CPS more secure and adaptive in our increasingly automated world. 

**Abstract (ZH)**: 在日益互联的世界中， Cyber-Physical Systems (CPS) 通过结合物理过程与计算能力，在医疗、交通和制造等行业中发挥着关键作用。然而，这些系统面临许多挑战，尤其是在安全性和系统故障方面。CPS中的异常可能表明从传感器故障到网络攻击等各种意外问题，必须被检测以防止可能造成损害或中断服务的故障。本文概述了研究人员在CPS中采用的不同异常检测方法。我们将这些方法（包括机器学习、深度学习、数学模型、不变量以及混合技术）进行分类和比较，旨在帮助读者了解这些方法的优势和局限性，并指导其用于创建更安全、更可靠的CPS。通过识别当前解决方案的不足，我们希望鼓励未来的研究，使CPS在日益自动化的世界中更具安全性和适应性。 

---
# Communication Strategy on Macro-and-Micro Traffic State in Cooperative Deep Reinforcement Learning for Regional Traffic Signal Control 

**Title (ZH)**: 宏微观交通状态的通信策略在合作深度强化学习区域交通信号控制中的应用 

**Authors**: Hankang Gu, Shangbo Wang, Dongyao Jia, Yuli Zhang, Yanrong Luo, Guoqiang Mao, Jianping Wang, Eng Gee Lim  

**Link**: [PDF](https://arxiv.org/pdf/2502.13248)  

**Abstract**: Adaptive Traffic Signal Control (ATSC) has become a popular research topic in intelligent transportation systems. Regional Traffic Signal Control (RTSC) using the Multi-agent Deep Reinforcement Learning (MADRL) technique has become a promising approach for ATSC due to its ability to achieve the optimum trade-off between scalability and optimality. Most existing RTSC approaches partition a traffic network into several disjoint regions, followed by applying centralized reinforcement learning techniques to each region. However, the pursuit of cooperation among RTSC agents still remains an open issue and no communication strategy for RTSC agents has been investigated. In this paper, we propose communication strategies to capture the correlation of micro-traffic states among lanes and the correlation of macro-traffic states among intersections. We first justify the evolution equation of the RTSC process is Markovian via a system of store-and-forward queues. Next, based on the evolution equation, we propose two GAT-Aggregated (GA2) communication modules--GA2-Naive and GA2-Aug to extract both intra-region and inter-region correlations between macro and micro traffic states. While GA2-Naive only considers the movements at each intersection, GA2-Aug also considers the lane-changing behavior of vehicles. Two proposed communication modules are then aggregated into two existing novel RTSC frameworks--RegionLight and Regional-DRL. Experimental results demonstrate that both GA2-Naive and GA2-Aug effectively improve the performance of existing RTSC frameworks under both real and synthetic scenarios. Hyperparameter testing also reveals the robustness and potential of our communication modules in large-scale traffic networks. 

**Abstract (ZH)**: 自适应交通信号控制（ATSC）已成为智能交通运输系统中的一个热门研究课题。基于多Agent深度强化学习（MADRL）的区域交通信号控制（RTSC）已成为ATSC的一个有前途的方法，由于其在可扩展性和优化性之间的最优权衡能力。现有的大多数RTSC方法将交通网络划分为若干个不相交区域，并在同一区域内应用集中式强化学习技术。然而，RTSC代理之间的合作追求仍然是一个开放的问题，尚未研究RTSC代理的通信策略。在本文中，我们提出了通信策略来捕捉车道上的微观交通状态之间的相关性以及交叉口之间的宏观交通状态之间的相关性。我们首先通过具有存储转发队列的系统证明了RTSC过程的演化方程是马尔可夫的。基于演化方程，我们提出了一种GAT-聚合（GA2）通信模块——GA2-Naive和GA2-Aug，以提取区域内的和区域间的宏观和微观交通状态的相关性。而GA2-Naive仅考虑交叉口的移动，GA2-Aug还考虑了车辆的车道变换行为。随后，提出的两个通信模块被整合到两个现有的新RTSC框架——RegionLight和Regional-DRL中。实验结果表明，GA2-Naive和GA2-Aug在实际和合成场景中均能有效提高现有RTSC框架的性能。超参数测试还揭示了我们提出的通信模块在大规模交通网络中的稳健性和潜力。 

---
# Conformal Prediction as Bayesian Quadrature 

**Title (ZH)**: 齐性预测作为贝叶斯 quadrature 

**Authors**: Jake C. Snell, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2502.13228)  

**Abstract**: As machine learning-based prediction systems are increasingly used in high-stakes situations, it is important to understand how such predictive models will perform upon deployment. Distribution-free uncertainty quantification techniques such as conformal prediction provide guarantees about the loss black-box models will incur even when the details of the models are hidden. However, such methods are based on frequentist probability, which unduly limits their applicability. We revisit the central aspects of conformal prediction from a Bayesian perspective and thereby illuminate the shortcomings of frequentist guarantees. We propose a practical alternative based on Bayesian quadrature that provides interpretable guarantees and offers a richer representation of the likely range of losses to be observed at test time. 

**Abstract (ZH)**: 基于机器学习的预测系统在高风险情况下被广泛应用后，了解其部署后的性能至关重要。分布自由的不确定性量化技术，如 conformal prediction，可以在模型细节隐藏的情况下提供关于黑盒模型损失的保证。然而，这些方法基于频率主义概率，限制了其应用范围。我们从贝叶斯视角重访 conformal prediction 的核心方面，并由此阐明频率主义保证的不足之处。我们提出了一种基于贝叶斯 quadrature 的实用替代方案，该方案提供了可解释的保证，并能更好地表示测试时可能出现的损失范围。 

---
# The Role of GitHub Copilot on Software Development: A Perspec-tive on Productivity, Security, Best Practices and Future Directions 

**Title (ZH)**: GitHub Copilot在软件开发中的作用：从生产率、安全性、最佳实践及未来方向的角度探讨 

**Authors**: Suresh Babu Nettur, Shanthi Karpurapu, Unnati Nettur, Likhit Sagar Gajja, Sravanthy Myneni, Akhil Dusi  

**Link**: [PDF](https://arxiv.org/pdf/2502.13199)  

**Abstract**: GitHub Copilot is transforming software development by automating tasks and boosting productivity through AI-driven code generation. In this paper, we con-duct a literature survey to synthesize insights on Copilot's impact on productivity and security. We review academic journal databases, industry reports, and official docu-mentation to highlight key findings and challenges. While Copilot accelerates coding and prototyping, concerns over security vulnerabilities and intellectual property risks persist. Drawing from the literature, we provide a perspective on best practices and future directions for responsible AI adoption in software engineering, offering action-able insights for developers and organizations to integrate Copilot effectively while maintaining high standards of quality and security. 

**Abstract (ZH)**: GitHub Copilot正通过AI驱动的代码生成自动化任务并提升软件开发效率与安全性：文献综述与负责任AI采用的展望 

---
# Enhancing Machine Learning Performance through Intelligent Data Quality Assessment: An Unsupervised Data-centric Framework 

**Title (ZH)**: 通过智能数据质量评估提升机器学习性能：一种无监督的数据为中心框架 

**Authors**: Manal Rahal, Bestoun S. Ahmed, Gergely Szabados, Torgny Fornstedt, Jorgen Samuelsson  

**Link**: [PDF](https://arxiv.org/pdf/2502.13198)  

**Abstract**: Poor data quality limits the advantageous power of Machine Learning (ML) and weakens high-performing ML software systems. Nowadays, data are more prone to the risk of poor quality due to their increasing volume and complexity. Therefore, tedious and time-consuming work goes into data preparation and improvement before moving further in the ML pipeline. To address this challenge, we propose an intelligent data-centric evaluation framework that can identify high-quality data and improve the performance of an ML system. The proposed framework combines the curation of quality measurements and unsupervised learning to distinguish high- and low-quality data. The framework is designed to integrate flexible and general-purpose methods so that it is deployed in various domains and applications. To validate the outcomes of the designed framework, we implemented it in a real-world use case from the field of analytical chemistry, where it is tested on three datasets of anti-sense oligonucleotides. A domain expert is consulted to identify the relevant quality measurements and evaluate the outcomes of the framework. The results show that the quality-centric data evaluation framework identifies the characteristics of high-quality data that guide the conduct of efficient laboratory experiments and consequently improve the performance of the ML system. 

**Abstract (ZH)**: 低质量数据限制了机器学习的优势并削弱了高性能机器学习软件系统的能力。随着数据量和复杂性的增加，数据更容易受到低质量风险的影响。因此，在进入机器学习管道之前需要进行繁琐且耗时的数据准备和改进工作。为应对这一挑战，我们提出了一种智能数据为中心的评价框架，以识别高质量数据并提升机器学习系统的性能。该框架结合了质量度量的策划和无监督学习，以区分高质量和低质量数据。该框架设计灵活且通用，以便在各种领域和应用中部署。为了验证所设计框架的结果，我们在分析化学领域实施了一个实际应用场景，并在三种反义寡核苷酸数据集中对其进行测试。领域专家参与识别相关质量度量并评估框架的结果。结果表明，质量为中心的数据评价框架识别出了高质量数据的特征，指导高效的实验室实验，并进而提升了机器学习系统的性能。 

---
# Conditional Max-Sum for Asynchronous Multiagent Decision Making 

**Title (ZH)**: 异步多agent决策制定的条件最大化算法 

**Authors**: Dimitrios Troullinos, Georgios Chalkiadakis, Ioannis Papamichail, Markos Papageorgiou  

**Link**: [PDF](https://arxiv.org/pdf/2502.13194)  

**Abstract**: In this paper we present a novel approach for multiagent decision making in dynamic environments based on Factor Graphs and the Max-Sum algorithm, considering asynchronous variable reassignments and distributed message-passing among agents. Motivated by the challenging domain of lane-free traffic where automated vehicles can communicate and coordinate as agents, we propose a more realistic communication framework for Factor Graph formulations that satisfies the above-mentioned restrictions, along with Conditional Max-Sum: an extension of Max-Sum with a revised message-passing process that is better suited for asynchronous settings. The overall application in lane-free traffic can be viewed as a hybrid system where the Factor Graph formulation undertakes the strategic decision making of vehicles, that of desired lateral alignment in a coordinated manner; and acts on top of a rule-based method we devise that provides a structured representation of the lane-free environment for the factors, while also handling the underlying control of vehicles regarding core operations and safety. Our experimental evaluation showcases the capabilities of the proposed framework in problems with intense coordination needs when compared to a domain-specific baseline without communication, and an increased adeptness of Conditional Max-Sum with respect to the standard algorithm. 

**Abstract (ZH)**: 基于因子图和Max-Sum算法的异步变量重赋值及分布式消息传递的多agent动态环境决策方法：适用于无车道交通的条件Max-Sum框架 

---
# On the Privacy Risks of Spiking Neural Networks: A Membership Inference Analysis 

**Title (ZH)**: 关于脉冲神经网络的隐私风险：一种成员推断分析 

**Authors**: Junyi Guan, Abhijith Sharma, Chong Tian, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2502.13191)  

**Abstract**: Spiking Neural Networks (SNNs) are increasingly explored for their energy efficiency and robustness in real-world applications, yet their privacy risks remain largely unexamined. In this work, we investigate the susceptibility of SNNs to Membership Inference Attacks (MIAs) -- a major privacy threat where an adversary attempts to determine whether a given sample was part of the training dataset. While prior work suggests that SNNs may offer inherent robustness due to their discrete, event-driven nature, we find that its resilience diminishes as latency (T) increases. Furthermore, we introduce an input dropout strategy under black box setting, that significantly enhances membership inference in SNNs. Our findings challenge the assumption that SNNs are inherently more secure, and even though they are expected to be better, our results reveal that SNNs exhibit privacy vulnerabilities that are equally comparable to Artificial Neural Networks (ANNs). Our code is available at this https URL. 

**Abstract (ZH)**: 脉冲神经网络（SNNs）在实际应用中因能效高和健壮性好而受到越来越多的关注，但其隐私风险尚未得到充分研究。本文探讨了SNNs对成员推理攻击（MIAs）的敏感性——一种主要的隐私威胁，攻击者试图确定给定样本是否属于训练数据集。虽然以往的研究表明，由于SNNs的离散和事件驱动的特性，它们可能具有内在的鲁棒性，但我们的研究发现，随着延迟（T）的增加，其抗攻击能力会减弱。此外，我们还在黑箱环境中引入了一种输入丢弃策略，显著增强了SNNs的成员推理能力。我们的研究挑战了SNNs本就更安全的假设，并揭示出尽管SNNs预期更好，但它们仍然存在与人工神经网络（ANNs）相当的隐私漏洞。我们的代码可在以下网址获取。 

---
# CondensNet: Enabling stable long-term climate simulations via hybrid deep learning models with adaptive physical constraints 

**Title (ZH)**: CondensNet：通过具有自适应物理约束的混合深度学习模型实现稳定的长期气候模拟 

**Authors**: Xin Wang, Juntao Yang, Jeff Adie, Simon See, Kalli Furtado, Chen Chen, Troy Arcomano, Romit Maulik, Gianmarco Mengaldo  

**Link**: [PDF](https://arxiv.org/pdf/2502.13185)  

**Abstract**: Accurate and efficient climate simulations are crucial for understanding Earth's evolving climate. However, current general circulation models (GCMs) face challenges in capturing unresolved physical processes, such as cloud and convection. A common solution is to adopt cloud resolving models, that provide more accurate results than the standard subgrid parametrisation schemes typically used in GCMs. However, cloud resolving models, also referred to as super paramtetrizations, remain computationally prohibitive. Hybrid modeling, which integrates deep learning with equation-based GCMs, offers a promising alternative but often struggles with long-term stability and accuracy issues. In this work, we find that water vapor oversaturation during condensation is a key factor compromising the stability of hybrid models. To address this, we introduce CondensNet, a novel neural network architecture that embeds a self-adaptive physical constraint to correct unphysical condensation processes. CondensNet effectively mitigates water vapor oversaturation, enhancing simulation stability while maintaining accuracy and improving computational efficiency compared to super parameterization schemes.
We integrate CondensNet into a GCM to form PCNN-GCM (Physics-Constrained Neural Network GCM), a hybrid deep learning framework designed for long-term stable climate simulations in real-world conditions, including ocean and land. PCNN-GCM represents a significant milestone in hybrid climate modeling, as it shows a novel way to incorporate physical constraints adaptively, paving the way for accurate, lightweight, and stable long-term climate simulations. 

**Abstract (ZH)**: 准确高效的气候模拟对于理解地球 evolving气候至关重要。然而，当前的地球系统模型（GCMs）在捕捉未解决的物理过程，如云和对流方面面临挑战。一种常见的解决方案是采用云解析模型，这些模型比通常在GCMs中使用的亚网格参数化方案提供更准确的结果。然而，云解析模型也被称为超参数化模型，在计算上仍具有挑战性。将深度学习与基于方程的GCMs相结合的综合模型提供了一种有前途的替代方案，但通常会遇到长期稳定性和准确性问题。在本研究中，我们发现凝结过程中水汽过饱和是综合模型不稳定的关键因素。为了解决这一问题，我们引入了CondensNet，这是一种新型神经网络架构，嵌入了自我适应的物理约束以纠正不物理的凝结过程。CondensNet有效地缓解了水汽过饱和问题，增强了模拟稳定性并保持了准确性，同时相比超参数化方案提高了计算效率。我们将CondensNet整合到GCM中，形成了PCNN-GCM（物理约束神经网络GCM），这是一种旨在适应现实世界条件（包括海洋和陆地）的长期稳定气候模拟的综合深度学习框架。PCNN-GCM代表了综合气候模型的一个重要里程碑，因为它展示了如何适应性地引入物理约束，为实现精确、轻量级和长期稳定的气候模拟铺平了道路。 

---
# RingFormer: Rethinking Recurrent Transformer with Adaptive Level Signals 

**Title (ZH)**: 环形变换器：重新思考具有自适应层级信号的循环变换器 

**Authors**: Jaemu Heo, Eldor Fozilov, Hyunmin Song, Taehwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.13181)  

**Abstract**: Transformers have achieved great success in effectively processing sequential data such as text. Their architecture consisting of several attention and feedforward blocks can model relations between elements of a sequence in parallel manner, which makes them very efficient to train and effective in sequence modeling. Even though they have shown strong performance in processing sequential data, the size of their parameters is considerably larger when compared to other architectures such as RNN and CNN based models. Therefore, several approaches have explored parameter sharing and recurrence in Transformer models to address their computational demands. However, such methods struggle to maintain high performance compared to the original transformer model. To address this challenge, we propose our novel approach, RingFormer, which employs one Transformer layer that processes input repeatedly in a circular, ring-like manner, while utilizing low-rank matrices to generate input-dependent level signals. This allows us to reduce the model parameters substantially while maintaining high performance in a variety of tasks such as translation and image classification, as validated in the experiments. 

**Abstract (ZH)**: RingFormer：一种循环处理的变压器模型 

---
# Uncertain Multi-Objective Recommendation via Orthogonal Meta-Learning Enhanced Bayesian Optimization 

**Title (ZH)**: 不确定多目标推荐通过正交元学习增强的贝叶斯优化 

**Authors**: Hongxu Wang, Zhu Sun, Yingpeng Du, Lu Zhang, Tiantian He, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2502.13180)  

**Abstract**: Recommender systems (RSs) play a crucial role in shaping our digital interactions, influencing how we access and engage with information across various domains. Traditional research has predominantly centered on maximizing recommendation accuracy, often leading to unintended side effects such as echo chambers and constrained user experiences. Drawing inspiration from autonomous driving, we introduce a novel framework that categorizes RS autonomy into five distinct levels, ranging from basic rule-based accuracy-driven systems to behavior-aware, uncertain multi-objective RSs - where users may have varying needs, such as accuracy, diversity, and fairness. In response, we propose an approach that dynamically identifies and optimizes multiple objectives based on individual user preferences, fostering more ethical and intelligent user-centric recommendations. To navigate the uncertainty inherent in multi-objective RSs, we develop a Bayesian optimization (BO) framework that captures personalized trade-offs between different objectives while accounting for their uncertain interdependencies. Furthermore, we introduce an orthogonal meta-learning paradigm to enhance BO efficiency and effectiveness by leveraging shared knowledge across similar tasks and mitigating conflicts among objectives through the discovery of orthogonal information. Finally, extensive empirical evaluations demonstrate the effectiveness of our method in optimizing uncertain multi-objectives for individual users, paving the way for more adaptive and user-focused RSs. 

**Abstract (ZH)**: 推荐系统在塑造我们的数字互动方面发挥着关键作用，影响我们在各个领域获取和互动信息的方式。传统研究主要集中在最大化推荐准确性上，这往往导致回声室效应和用户使用体验受限等意外后果。受到自动驾驶的启发，我们提出了一种新的框架，将推荐系统的自主性分为五个不同的层级，从基于基本规则和准确性驱动的系统到关注用户行为、具有不确定性的多目标推荐系统——其中用户的需求可能不同，例如准确性、多样性和平等性。为响应这一挑战，我们提出了一种基于个体用户偏好的动态识别和优化多目标的方法，促进更具伦理性和智能性的以用户为中心的推荐。为应对多目标推荐系统中的不确定性，我们开发了一种贝叶斯优化（BO）框架，既能捕捉不同目标之间的个性化权衡，又能考虑它们之间的不确定关联性。此外，我们引入了一种正交元学习范式，通过利用相似任务中的共享知识来提高贝叶斯优化效率和效果，并通过发现正交信息来缓解目标间的冲突。最后，广泛的实证评估表明，我们的方法在优化个体用户的不确定多目标方面效果显著，为更具适应性和用户导向的推荐系统的开发铺平了道路。 

---
# Generative Topology Optimization: Exploring Diverse Solutions in Structural Design 

**Title (ZH)**: 生成拓扑优化：在结构设计中探索多样化的解决方案 

**Authors**: Andreas Radler, Eric Volkmann, Johannes Brandstetter, Arturs Berzins  

**Link**: [PDF](https://arxiv.org/pdf/2502.13174)  

**Abstract**: Topology optimization (TO) is a family of computational methods that derive near-optimal geometries from formal problem descriptions. Despite their success, established TO methods are limited to generating single solutions, restricting the exploration of alternative designs. To address this limitation, we introduce Generative Topology Optimization (GenTO) - a data-free method that trains a neural network to generate structurally compliant shapes and explores diverse solutions through an explicit diversity constraint. The network is trained with a solver-in-the-loop, optimizing the material distribution in each iteration. The trained model produces diverse shapes that closely adhere to the design requirements. We validate GenTO on 2D and 3D TO problems. Our results demonstrate that GenTO produces more diverse solutions than any prior method while maintaining near-optimality and being an order of magnitude faster due to inherent parallelism. These findings open new avenues for engineering and design, offering enhanced flexibility and innovation in structural optimization. 

**Abstract (ZH)**: 生成式拓扑优化（GenTO）：一种数据驱动的方法 

---
# Web Phishing Net (WPN): A scalable machine learning approach for real-time phishing campaign detection 

**Title (ZH)**: Web钓鱼网(WPN):一种实时钓鱼活动检测的可扩展机器学习方法 

**Authors**: Muhammad Fahad Zia, Sri Harish Kalidass  

**Link**: [PDF](https://arxiv.org/pdf/2502.13171)  

**Abstract**: Phishing is the most prevalent type of cyber-attack today and is recognized as the leading source of data breaches with significant consequences for both individuals and corporations. Web-based phishing attacks are the most frequent with vectors such as social media posts and emails containing links to phishing URLs that once clicked on render host systems vulnerable to more sinister attacks. Research efforts to detect phishing URLs have involved the use of supervised learning techniques that use large amounts of data to train models and have high computational requirements. They also involve analysis of features derived from vectors including email contents thus affecting user privacy. Additionally, they suffer from a lack of resilience against evolution of threats especially with the advent of generative AI techniques to bypass these systems as with AI-generated phishing URLs. Unsupervised methods such as clustering techniques have also been used in phishing detection in the past, however, they are at times unscalable due to the use of pair-wise comparisons. They also lack high detection rates while detecting phishing campaigns. In this paper, we propose an unsupervised learning approach that is not only fast but scalable, as it does not involve pair-wise comparisons. It is able to detect entire campaigns at a time with a high detection rate while preserving user privacy; this includes the recent surge of campaigns with targeted phishing URLs generated by malicious entities using generative AI techniques. 

**Abstract (ZH)**: 基于聚类的无监督学习方法：快速、可扩展的钓鱼网址检测以保护用户隐私 

---
# Multi-Agent Actor-Critic Generative AI for Query Resolution and Analysis 

**Title (ZH)**: 多Agentactor-批评生成AI查询解析与分析 

**Authors**: Mohammad Wali Ur Rahman, Ric Nevarez, Lamia Tasnim Mim, Salim Hariri  

**Link**: [PDF](https://arxiv.org/pdf/2502.13164)  

**Abstract**: In this paper, we introduce MASQRAD (Multi-Agent Strategic Query Resolution and Diagnostic tool), a transformative framework for query resolution based on the actor-critic model, which utilizes multiple generative AI agents. MASQRAD is excellent at translating imprecise or ambiguous user inquiries into precise and actionable requests. This framework generates pertinent visualizations and responses to these focused queries, as well as thorough analyses and insightful interpretations for users. MASQRAD addresses the common shortcomings of existing solutions in domains that demand fast and precise data interpretation, such as their incapacity to successfully apply AI for generating actionable insights and their challenges with the inherent ambiguity of user queries. MASQRAD functions as a sophisticated multi-agent system but "masquerades" to users as a single AI entity, which lowers errors and enhances data interaction. This approach makes use of three primary AI agents: Actor Generative AI, Critic Generative AI, and Expert Analysis Generative AI. Each is crucial for creating, enhancing, and evaluating data interactions. The Actor AI generates Python scripts to generate data visualizations from large datasets within operational constraints, and the Critic AI rigorously refines these scripts through multi-agent debate. Finally, the Expert Analysis AI contextualizes the outcomes to aid in decision-making. With an accuracy rate of 87\% when handling tasks related to natural language visualization, MASQRAD establishes new benchmarks for automated data interpretation and showcases a noteworthy advancement that has the potential to revolutionize AI-driven applications. 

**Abstract (ZH)**: 基于演员-评论家模型的多Agent战略查询解析与诊断框架MASQRAD 

---
