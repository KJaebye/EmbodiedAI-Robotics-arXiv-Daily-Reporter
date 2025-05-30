# Adapting Probabilistic Risk Assessment for AI 

**Title (ZH)**: 适配人工智能的概率风险评估 

**Authors**: Anna Katariina Wisakanto, Joe Rogero, Avyay M. Casheekar, Richard Mallah  

**Link**: [PDF](https://arxiv.org/pdf/2504.18536)  

**Abstract**: Modern general-purpose artificial intelligence (AI) systems present an urgent risk management challenge, as their rapidly evolving capabilities and potential for catastrophic harm outpace our ability to reliably assess their risks. Current methods often rely on selective testing and undocumented assumptions about risk priorities, frequently failing to make a serious attempt at assessing the set of pathways through which Al systems pose direct or indirect risks to society and the biosphere. This paper introduces the probabilistic risk assessment (PRA) for AI framework, adapting established PRA techniques from high-reliability industries (e.g., nuclear power, aerospace) for the new challenges of advanced AI. The framework guides assessors in identifying potential risks, estimating likelihood and severity, and explicitly documenting evidence, underlying assumptions, and analyses at appropriate granularities. The framework's implementation tool synthesizes the results into a risk report card with aggregated risk estimates from all assessed risks. This systematic approach integrates three advances: (1) Aspect-oriented hazard analysis provides systematic hazard coverage guided by a first-principles taxonomy of AI system aspects (e.g. capabilities, domain knowledge, affordances); (2) Risk pathway modeling analyzes causal chains from system aspects to societal impacts using bidirectional analysis and incorporating prospective techniques; and (3) Uncertainty management employs scenario decomposition, reference scales, and explicit tracing protocols to structure credible projections with novelty or limited data. Additionally, the framework harmonizes diverse assessment methods by integrating evidence into comparable, quantified absolute risk estimates for critical decisions. We have implemented this as a workbook tool for AI developers, evaluators, and regulators, available on the project website. 

**Abstract (ZH)**: 现代通用人工智能的probabilistic风险评估框架：适应先进人工智能的新挑战 

---
# Scaling Laws For Scalable Oversight 

**Title (ZH)**: 可扩展监督的标度定律 

**Authors**: Joshua Engels, David D. Baek, Subhash Kantamneni, Max Tegmark  

**Link**: [PDF](https://arxiv.org/pdf/2504.18530)  

**Abstract**: Scalable oversight, the process by which weaker AI systems supervise stronger ones, has been proposed as a key strategy to control future superintelligent systems. However, it is still unclear how scalable oversight itself scales. To address this gap, we propose a framework that quantifies the probability of successful oversight as a function of the capabilities of the overseer and the system being overseen. Specifically, our framework models oversight as a game between capability-mismatched players; the players have oversight-specific and deception-specific Elo scores that are a piecewise-linear function of their general intelligence, with two plateaus corresponding to task incompetence and task saturation. We validate our framework with a modified version of the game Nim and then apply it to four oversight games: "Mafia", "Debate", "Backdoor Code" and "Wargames". For each game, we find scaling laws that approximate how domain performance depends on general AI system capability (using Chatbot Arena Elo as a proxy for general capability). We then build on our findings in a theoretical study of Nested Scalable Oversight (NSO), a process in which trusted models oversee untrusted stronger models, which then become the trusted models in the next step. We identify conditions under which NSO succeeds and derive numerically (and in some cases analytically) the optimal number of oversight levels to maximize the probability of oversight success. In our numerical examples, the NSO success rate is below 52% when overseeing systems that are 400 Elo points stronger than the baseline overseer, and it declines further for overseeing even stronger systems. 

**Abstract (ZH)**: 可扩展的监督：一种量化监督成功概率的框架及其在多层次信任模型中的应用 

---
# Reason Like a Radiologist: Chain-of-Thought and Reinforcement Learning for Verifiable Report Generation 

**Title (ZH)**: 像放射学家一样思考：基于因果推理和强化学习的可验证报告生成 

**Authors**: Peiyuan Jing, Kinhei Lee, Zhenxuan Zhang, Huichi Zhou, Zhengqing Yuan, Zhifan Gao, Lei Zhu, Giorgos Papanastasiou, Yingying Fang, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18453)  

**Abstract**: Radiology report generation is critical for efficiency but current models lack the structured reasoning of experts, hindering clinical trust and explainability by failing to link visual findings to precise anatomical locations. This paper introduces BoxMed-RL, a groundbreaking unified training framework for generating spatially verifiable and explainable radiology reports. Built on a large vision-language model, BoxMed-RL revolutionizes report generation through two integrated phases: (1) In the Pretraining Phase, we refine the model via medical concept learning, using Chain-of-Thought supervision to internalize the radiologist-like workflow, followed by spatially verifiable reinforcement, which applies reinforcement learning to align medical findings with bounding boxes. (2) In the Downstream Adapter Phase, we freeze the pretrained weights and train a downstream adapter to ensure fluent and clinically credible reports. This framework precisely mimics radiologists' workflow, compelling the model to connect high-level medical concepts with definitive anatomical evidence. Extensive experiments on public datasets demonstrate that BoxMed-RL achieves an average 7% improvement in both METEOR and ROUGE-L metrics compared to state-of-the-art methods. An average 5% improvement in large language model-based metrics further underscores BoxMed-RL's robustness in generating high-quality radiology reports. 

**Abstract (ZH)**: BoxMed-RL：一种生成空间可验证和解释性强的放射学报告的统一训练框架 

---
# Pseudo-Boolean Proof Logging for Optimal Classical Planning 

**Title (ZH)**: 伪布尔证明记录在最优经典规划中的应用 

**Authors**: Simon Dold, Malte Helmert, Jakob Nordström, Gabriele Röger, Tanja Schindler  

**Link**: [PDF](https://arxiv.org/pdf/2504.18443)  

**Abstract**: We introduce lower-bound certificates for classical planning tasks, which can be used to prove the unsolvability of a task or the optimality of a plan in a way that can be verified by an independent third party. We describe a general framework for generating lower-bound certificates based on pseudo-Boolean constraints, which is agnostic to the planning algorithm used.
As a case study, we show how to modify the $A^{*}$ algorithm to produce proofs of optimality with modest overhead, using pattern database heuristics and $h^\textit{max}$ as concrete examples. The same proof logging approach works for any heuristic whose inferences can be efficiently expressed as reasoning over pseudo-Boolean constraints. 

**Abstract (ZH)**: 我们引入了用于经典规划任务的下界证书，可以通过独立第三方验证这些证书来证明任务的不可解或计划的最优性。我们描述了一个基于伪布尔约束生成下界证书的通用框架，该框架与使用的规划算法无关。作为案例研究，我们展示了如何通过修改$A^{*}$算法并结合模式数据库启发式和$h^\textit{max}$启发式，以适度的开销产生最优性证明。相同的方法适用于任何可以高效表达为伪布尔约束推理的启发式算法。 

---
# Combating the Bucket Effect:Multi-Knowledge Alignment for Medication Recommendation 

**Title (ZH)**: 对抗桶效应：多知识对齐的药品推荐方法 

**Authors**: Xiang Li, Haixu Ma, Guanyong Wu, Shi Mu, Chen Li, Shunpan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18096)  

**Abstract**: Medication recommendation is crucial in healthcare, offering effective treatments based on patient's electronic health records (EHR). Previous studies show that integrating more medication-related knowledge improves medication representation accuracy. However, not all medications encompass multiple types of knowledge data simultaneously. For instance, some medications provide only textual descriptions without structured data. This imbalance in data availability limits the performance of existing models, a challenge we term the "bucket effect" in medication recommendation. Our data analysis uncovers the severity of the "bucket effect" in medication recommendation. To fill this gap, we introduce a cross-modal medication encoder capable of seamlessly aligning data from different modalities and propose a medication recommendation framework to integrate Multiple types of Knowledge, named MKMed. Specifically, we first pre-train a cross-modal encoder with contrastive learning on five knowledge modalities, aligning them into a unified space. Then, we combine the multi-knowledge medication representations with patient records for recommendations. Extensive experiments on the MIMIC-III and MIMIC-IV datasets demonstrate that MKMed mitigates the "bucket effect" in data, and significantly outperforms state-of-the-art baselines in recommendation accuracy and safety. 

**Abstract (ZH)**: 基于电子健康记录的药品推荐至关重要：整合多类型知识的跨模态药品编码器（MKMed） 

---
# MultiMind: Enhancing Werewolf Agents with Multimodal Reasoning and Theory of Mind 

**Title (ZH)**: 多模态推理与心理理论增强的狼人代理：MultiMind 

**Authors**: Zheng Zhang, Nuoqian Xiao, Qi Chai, Deheng Ye, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18039)  

**Abstract**: Large Language Model (LLM) agents have demonstrated impressive capabilities in social deduction games (SDGs) like Werewolf, where strategic reasoning and social deception are essential. However, current approaches remain limited to textual information, ignoring crucial multimodal cues such as facial expressions and tone of voice that humans naturally use to communicate. Moreover, existing SDG agents primarily focus on inferring other players' identities without modeling how others perceive themselves or fellow players. To address these limitations, we use One Night Ultimate Werewolf (ONUW) as a testbed and present MultiMind, the first framework integrating multimodal information into SDG agents. MultiMind processes facial expressions and vocal tones alongside verbal content, while employing a Theory of Mind (ToM) model to represent each player's suspicion levels toward others. By combining this ToM model with Monte Carlo Tree Search (MCTS), our agent identifies communication strategies that minimize suspicion directed at itself. Through comprehensive evaluation in both agent-versus-agent simulations and studies with human players, we demonstrate MultiMind's superior performance in gameplay. Our work presents a significant advancement toward LLM agents capable of human-like social reasoning across multimodal domains. 

**Abstract (ZH)**: 大型语言模型代理在社会推理游戏中的多模态应用：MultiMind框架克服了现有方法的局限性 

---
# Differential Privacy-Driven Framework for Enhancing Heart Disease Prediction 

**Title (ZH)**: 差分隐私驱动的心脏疾病预测增强框架 

**Authors**: Yazan Otoum, Amiya Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2504.18007)  

**Abstract**: With the rapid digitalization of healthcare systems, there has been a substantial increase in the generation and sharing of private health data. Safeguarding patient information is essential for maintaining consumer trust and ensuring compliance with legal data protection regulations. Machine learning is critical in healthcare, supporting personalized treatment, early disease detection, predictive analytics, image interpretation, drug discovery, efficient operations, and patient monitoring. It enhances decision-making, accelerates research, reduces errors, and improves patient outcomes. In this paper, we utilize machine learning methodologies, including differential privacy and federated learning, to develop privacy-preserving models that enable healthcare stakeholders to extract insights without compromising individual privacy. Differential privacy introduces noise to data to guarantee statistical privacy, while federated learning enables collaborative model training across decentralized datasets. We explore applying these technologies to Heart Disease Data, demonstrating how they preserve privacy while delivering valuable insights and comprehensive analysis. Our results show that using a federated learning model with differential privacy achieved a test accuracy of 85%, ensuring patient data remained secure and private throughout the process. 

**Abstract (ZH)**: 随着 healthcare 系统的快速数字化，私人健康数据的生成和共享显著增加。保护患者信息对于维护消费者信任并确保遵守数据保护法规至关重要。机器学习在医疗保健中至关重要，支持个性化的治疗、疾病的早期检测、预测分析、图像解释、药物发现、高效运营和患者监测。它能增强决策、加速研究、减少错误并改善患者结果。在本文中，我们利用机器学习方法，包括差分隐私和联邦学习，开发了保护隐私的模型，从而使医疗保健利益相关者能够在不泄露个人隐私的情况下提取有用信息。差分隐私通过对数据添加噪声来保证统计隐私，而联邦学习则允许多个分散的数据集共同训练模型。我们探讨了将这些技术应用于心脏病数据的可行性，展示了它们如何在保护隐私的同时提供有价值的洞察和全面的分析。结果表明，使用联邦学习模型结合差分隐私实现了85%的测试准确率，确保整个过程中患者数据的安全性和隐私性。 

---
# LLM Agent Swarm for Hypothesis-Driven Drug Discovery 

**Title (ZH)**: 基于假设的药物发现agents蜂群 swarm for 基于大规模语言模型的假设驱动药物发现 

**Authors**: Kevin Song, Andrew Trotter, Jake Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.17967)  

**Abstract**: Drug discovery remains a formidable challenge: more than 90 percent of candidate molecules fail in clinical evaluation, and development costs often exceed one billion dollars per approved therapy. Disparate data streams, from genomics and transcriptomics to chemical libraries and clinical records, hinder coherent mechanistic insight and slow progress. Meanwhile, large language models excel at reasoning and tool integration but lack the modular specialization and iterative memory required for regulated, hypothesis-driven workflows. We introduce PharmaSwarm, a unified multi-agent framework that orchestrates specialized LLM "agents" to propose, validate, and refine hypotheses for novel drug targets and lead compounds. Each agent accesses dedicated functionality--automated genomic and expression analysis; a curated biomedical knowledge graph; pathway enrichment and network simulation; interpretable binding affinity prediction--while a central Evaluator LLM continuously ranks proposals by biological plausibility, novelty, in silico efficacy, and safety. A shared memory layer captures validated insights and fine-tunes underlying submodels over time, yielding a self-improving system. Deployable on low-code platforms or Kubernetes-based microservices, PharmaSwarm supports literature-driven discovery, omics-guided target identification, and market-informed repurposing. We also describe a rigorous four-tier validation pipeline spanning retrospective benchmarking, independent computational assays, experimental testing, and expert user studies to ensure transparency, reproducibility, and real-world impact. By acting as an AI copilot, PharmaSwarm can accelerate translational research and deliver high-confidence hypotheses more efficiently than traditional pipelines. 

**Abstract (ZH)**: 制药领域的药物发现依然是一项艰巨的挑战：超过90%的候选分子在临床评估中失败，且每种获批疗法的研发成本常常超过十亿美元。异质的数据流，包括基因组学、转录组学、化学库以及临床记录，阻碍了连贯的机制洞察并减缓了研究进展。与此同时，大型语言模型在推理和工具集成方面表现出色，但在符合监管要求、基于假设的迭代工作中缺乏模块化的特殊化和持续记忆功能。我们介绍了PharmaSwarm，这是一种统一的多代理框架，它可以协调专门的大型语言模型“代理”来提出、验证和完善针对新药靶点和先导化合物的假设。每个代理都接入特定的功能——自动基因组和表达分析；经过编目的生物医学知识图谱；途径富集和网络模拟；可解释的结合亲和力预测——而中心的评估LSTM持续根据生物可行性、新颖性、虚拟效果和安全性对提案进行排名。共享的内存层捕获验证过的洞察，并随时间微调底层子模型，从而产生一个自我改进的系统。PharmaSwarm 可部署在低代码平台或基于Kubernetes的微服务上，支持文献驱动的发现、组学导向的目标识别以及市场驱动的再利用。我们还描述了一个严格的四级验证管道，涵盖回顾性基准测试、独立计算实验、实验测试和专家用户研究，以确保透明性、可再现性和实际影响。通过充当AI副驾，PharmaSwarm 可以加速转化研究，并比传统管道更高效地提供高置信度假设。 

---
# ApproXAI: Energy-Efficient Hardware Acceleration of Explainable AI using Approximate Computing 

**Title (ZH)**: ApproXAI：使用近似计算实现可解释人工智能的能源高效硬件加速 

**Authors**: Ayesha Siddique, Khurram Khalil, Khaza Anuarul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2504.17929)  

**Abstract**: Explainable artificial intelligence (XAI) enhances AI system transparency by framing interpretability as an optimization problem. However, this approach often necessitates numerous iterations of computationally intensive operations, limiting its applicability in real-time scenarios. While recent research has focused on XAI hardware acceleration on FPGAs and TPU, these methods do not fully address energy efficiency in real-time settings. To address this limitation, we propose XAIedge, a novel framework that leverages approximate computing techniques into XAI algorithms, including integrated gradients, model distillation, and Shapley analysis. XAIedge translates these algorithms into approximate matrix computations and exploits the synergy between convolution, Fourier transform, and approximate computing paradigms. This approach enables efficient hardware acceleration on TPU-based edge devices, facilitating faster real-time outcome interpretations. Our comprehensive evaluation demonstrates that XAIedge achieves a $2\times$ improvement in energy efficiency compared to existing accurate XAI hardware acceleration techniques while maintaining comparable accuracy. These results highlight the potential of XAIedge to significantly advance the deployment of explainable AI in energy-constrained real-time applications. 

**Abstract (ZH)**: 可解释的人工智能（XAI）通过将解释性问题作为优化问题来增强AI系统的透明度。然而，这种方法往往需要多次进行计算密集型操作，限制了其在实时场景中的应用。尽管最近的研究集中在FPGA和TPU上的XAI硬件加速上，但这些方法并未完全解决实时场景中的能效问题。为解决这一限制，我们提出了XAIedge框架，该框架利用近似计算技术到XAI算法中，包括集成梯度、模型蒸馏和Shapley分析。XAIedge将这些算法转化为近似矩阵计算，并利用卷积、傅里叶变换和近似计算范式的协同作用。该方法能够在基于TPU的边缘设备上实现高效的硬件加速，促进更快的实时结果解释。我们的综合评估表明，XAIedge在能效方面实现了比现有精确XAI硬件加速技术两倍的改进，同时保持了相当的准确性。这些结果突显了XAIedge在能源受限的实时应用中显著推进可解释AI部署的潜力。 

---
# Generalization Capability for Imitation Learning 

**Title (ZH)**: 模仿学习的泛化能力 

**Authors**: Yixiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18538)  

**Abstract**: Imitation learning holds the promise of equipping robots with versatile skills by learning from expert demonstrations. However, policies trained on finite datasets often struggle to generalize beyond the training distribution. In this work, we present a unified perspective on the generalization capability of imitation learning, grounded in both information theorey and data distribution property. We first show that the generalization gap can be upper bounded by (i) the conditional information bottleneck on intermediate representations and (ii) the mutual information between the model parameters and the training dataset. This characterization provides theoretical guidance for designing effective training strategies in imitation learning, particularly in determining whether to freeze, fine-tune, or train large pretrained encoders (e.g., vision-language models or vision foundation models) from scratch to achieve better generalization. Furthermore, we demonstrate that high conditional entropy from input to output induces a flatter likelihood landscape, thereby reducing the upper bound on the generalization gap. In addition, it shortens the stochastic gradient descent (SGD) escape time from sharp local minima, which may increase the likelihood of reaching global optima under fixed optimization budgets. These insights explain why imitation learning often exhibits limited generalization and underscore the importance of not only scaling the diversity of input data but also enriching the variability of output labels conditioned on the same input. 

**Abstract (ZH)**: 模仿学习在信息理论和数据分布性质的基础上提供了一种统一的观点，以装备机器人具备多样化技能。然而，基于有限数据集训练的策略往往难以在训练分布之外进行泛化。在这项工作中，我们展示了模仿学习泛化能力的一种统一视角，该视角基于信息理论和数据分布的特性。我们首先证明了泛化差距可以通过(i) 中间表示的条件信息瓶颈和(ii) 模型参数与训练数据集之间的互信息进行上界估计。这种表征为设计有效的模仿学习训练策略提供了理论指导，尤其是在确定是否冻结、微调或从零开始训练大型预训练编码器（例如视觉语言模型或视觉基础模型）以实现更好的泛化效果时。此外，我们还证明了从输入到输出的高条件熵会导致更平坦的似然景观，从而降低泛化差距的上界。此外，这缩短了从尖峰局部最小值逃离随机梯度下降（SGD）的时间，从而可能在固定优化预算下增加达到全局最优的可能性。这些见解解释了为什么模仿学习常常表现出有限的泛化能力，并强调了不仅扩展输入数据多样性而且在相同输入条件下丰富输出标签变化性的重要性。 

---
# DeSIA: Attribute Inference Attacks Against Limited Fixed Aggregate Statistics 

**Title (ZH)**: DeSIA：针对有限固定聚合统计的属性推断攻击 

**Authors**: Yifeng Mao, Bozhidar Stevanoski, Yves-Alexandre de Montjoye  

**Link**: [PDF](https://arxiv.org/pdf/2504.18497)  

**Abstract**: Empirical inference attacks are a popular approach for evaluating the privacy risk of data release mechanisms in practice. While an active attack literature exists to evaluate machine learning models or synthetic data release, we currently lack comparable methods for fixed aggregate statistics, in particular when only a limited number of statistics are released. We here propose an inference attack framework against fixed aggregate statistics and an attribute inference attack called DeSIA. We instantiate DeSIA against the U.S. Census PPMF dataset and show it to strongly outperform reconstruction-based attacks. In particular, we show DeSIA to be highly effective at identifying vulnerable users, achieving a true positive rate of 0.14 at a false positive rate of $10^{-3}$. We then show DeSIA to perform well against users whose attributes cannot be verified and when varying the number of aggregate statistics and level of noise addition. We also perform an extensive ablation study of DeSIA and show how DeSIA can be successfully adapted to the membership inference task. Overall, our results show that aggregation alone is not sufficient to protect privacy, even when a relatively small number of aggregates are being released, and emphasize the need for formal privacy mechanisms and testing before aggregate statistics are released. 

**Abstract (ZH)**: 经验推理攻击是评估数据发布机制实际隐私风险的一种流行方法。虽然存在针对机器学习模型或合成数据发布的主动攻击文献，但在仅发布有限数量统计信息的情况下，我们目前缺乏可比的方法，尤其是在固定聚合统计方面。我们在此提出一种针对固定聚合统计的推理攻击框架以及一种属性推理攻击，称为DeSIA。我们将DeSIA实例化应用于美国人口普查PPMF数据集，并证明它在重建攻击中表现更为优异。特别是，我们展示DeSIA在识别易受攻击用户方面极其有效，实现了在假阳性率为$10^{-3}$时的真实阳性率为0.14。随后，我们展示DeSIA在用户属性无法验证以及聚合统计数量和噪声添加水平变化时的表现良好。我们还进行了DeSIA的广泛消融研究，并展示了如何成功地将DeSIA适应到成员推理任务。总体而言，我们的研究结果表明，仅聚合本身不足以保护隐私，即使发布的聚合数量相对较少，强调了在发布聚合统计之前需要正式的隐私机制和测试。 

---
# Action Flow Matching for Continual Robot Learning 

**Title (ZH)**: 连续机器人学习中的动作流程匹配 

**Authors**: Alejandro Murillo-Gonzalez, Lantao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18471)  

**Abstract**: Continual learning in robotics seeks systems that can constantly adapt to changing environments and tasks, mirroring human adaptability. A key challenge is refining dynamics models, essential for planning and control, while addressing issues such as safe adaptation, catastrophic forgetting, outlier management, data efficiency, and balancing exploration with exploitation -- all within task and onboard resource constraints. Towards this goal, we introduce a generative framework leveraging flow matching for online robot dynamics model alignment. Rather than executing actions based on a misaligned model, our approach refines planned actions to better match with those the robot would take if its model was well aligned. We find that by transforming the actions themselves rather than exploring with a misaligned model -- as is traditionally done -- the robot collects informative data more efficiently, thereby accelerating learning. Moreover, we validate that the method can handle an evolving and possibly imperfect model while reducing, if desired, the dependency on replay buffers or legacy model snapshots. We validate our approach using two platforms: an unmanned ground vehicle and a quadrotor. The results highlight the method's adaptability and efficiency, with a record 34.2\% higher task success rate, demonstrating its potential towards enabling continual robot learning. Code: this https URL. 

**Abstract (ZH)**: 机器人领域的连续学习寻求能够不断适应变化环境和任务的系统，模拟人类的适应能力。一个关键挑战是在处理安全适应、灾难性遗忘、离群值管理、数据效率以及在任务和机载资源约束下平衡探索与利用等问题的同时，精炼动力学模型，这对于规划和控制至关重要。为实现这一目标，我们引入了一种生成框架，利用流匹配进行在线机器人动力学模型对齐。我们的方法不是基于错配的模型执行动作，而是通过对计划动作本身进行改进，使其更好地与机器人在其模型对齐良好时可能采取的动作匹配。我们发现，通过直接变换动作本身而不是使用错配的模型进行探索，机器人可以更有效地收集信息性数据，从而加速学习。此外，我们验证了该方法可以处理不断变化且可能不完备的模型，同时在需要时减少对重放缓冲区或遗留模型快照的依赖。我们使用两个平台验证了该方法：无人驾驶地面车辆和四旋翼无人机。结果强调了该方法的适应性和效率，最高任务成功率提升了34.2%，展示了其在实现连续机器人学习方面的潜力。代码：详见链接。 

---
# Fast-Slow Thinking for Large Vision-Language Model Reasoning 

**Title (ZH)**: 快速-缓慢思考在大规模视觉-语言模型推理中的应用 

**Authors**: Wenyi Xiao, Leilei Gan, Weilong Dai, Wanggui He, Ziwei Huang, Haoyuan Li, Fangxun Shu, Zhelun Yu, Peng Zhang, Hao Jiang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18458)  

**Abstract**: Recent advances in large vision-language models (LVLMs) have revealed an \textit{overthinking} phenomenon, where models generate verbose reasoning across all tasks regardless of questions. To address this issue, we present \textbf{FAST}, a novel \textbf{Fa}st-\textbf{S}low \textbf{T}hinking framework that dynamically adapts reasoning depth based on question characteristics. Through empirical analysis, we establish the feasibility of fast-slow thinking in LVLMs by investigating how response length and data distribution affect performance. We develop FAST-GRPO with three components: model-based metrics for question characterization, an adaptive thinking reward mechanism, and difficulty-aware KL regularization. Experiments across seven reasoning benchmarks demonstrate that FAST achieves state-of-the-art accuracy with over 10\% relative improvement compared to the base model, while reducing token usage by 32.7-67.3\% compared to previous slow-thinking approaches, effectively balancing reasoning length and accuracy. 

**Abstract (ZH)**: 最近的大规模视觉-语言模型（LVLMs）研究揭示了过度推理现象，即模型在所有任务中不分情境地产生冗长的推理过程。为了解决这一问题，我们提出了FAST（快速-缓慢思考）框架，这是一种新的动态适应推理深度的框架，基于问题特征调整推理深度。通过实证分析，我们研究了响应长度和数据分布如何影响性能，以证明在LVLMs中实施快速-缓慢思考的可行性。我们开发了FAST-GRPO，它包括基于模型的问题特征度量、自适应思考奖励机制和难度感知的KL正则化。在七个推理基准测试中的实验表明，FAST 在相对于基线模型的相对准确性提高了10%以上的同时，相比之前缓慢思考的方法减少了32.7%-67.3%的_token_使用量，有效地平衡了推理长度和准确性。 

---
# Iterative Event-based Motion Segmentation by Variational Contrast Maximization 

**Title (ZH)**: 基于变异对比最大化迭代事件驱动运动分割 

**Authors**: Ryo Yamaki, Shintaro Shiba, Guillermo Gallego, Yoshimitsu Aoki  

**Link**: [PDF](https://arxiv.org/pdf/2504.18447)  

**Abstract**: Event cameras provide rich signals that are suitable for motion estimation since they respond to changes in the scene. As any visual changes in the scene produce event data, it is paramount to classify the data into different motions (i.e., motion segmentation), which is useful for various tasks such as object detection and visual servoing. We propose an iterative motion segmentation method, by classifying events into background (e.g., dominant motion hypothesis) and foreground (independent motion residuals), thus extending the Contrast Maximization framework. Experimental results demonstrate that the proposed method successfully classifies event clusters both for public and self-recorded datasets, producing sharp, motion-compensated edge-like images. The proposed method achieves state-of-the-art accuracy on moving object detection benchmarks with an improvement of over 30%, and demonstrates its possibility of applying to more complex and noisy real-world scenes. We hope this work broadens the sensitivity of Contrast Maximization with respect to both motion parameters and input events, thus contributing to theoretical advancements in event-based motion segmentation estimation. this https URL 

**Abstract (ZH)**: 事件相机提供了一种丰富的信号，适用于运动估计，因为它们对场景变化有所响应。由于场景中的任何视觉变化都会产生事件数据，因此对数据进行不同运动的分类（即运动分割）至关重要，这有助于各种任务，如目标检测和视觉伺服。我们提出了一种迭代运动分割方法，通过将事件分为背景（例如，主导运动假设）和前景（独立运动残差），从而扩展了对比最大化框架。实验结果表明，所提出的方法能够成功地对公共数据集和自录制数据集中的事件簇进行分类，生成清晰、运动补偿的边缘般图像。所提出的方法在移动目标检测基准测试中达到了最先进的精度，改进幅度超过30%，并证明了其应用于更复杂和嘈杂的现实场景的可能性。我们希望通过这项工作增强对比最大化方法对运动参数和输入事件的敏感性，从而促进基于事件的运动分割估计的理论进步。[原文链接] 

---
# Enhancing Pre-Trained Model-Based Class-Incremental Learning through Neural Collapse 

**Title (ZH)**: 基于神经坍缩提升预训练模型驱动的类别增量学习 

**Authors**: Kun He, Zijian Song, Shuoxi Zhang, John E. Hopcroft  

**Link**: [PDF](https://arxiv.org/pdf/2504.18437)  

**Abstract**: Class-Incremental Learning (CIL) is a critical capability for real-world applications, enabling learning systems to adapt to new tasks while retaining knowledge from previous ones. Recent advancements in pre-trained models (PTMs) have significantly advanced the field of CIL, demonstrating superior performance over traditional methods. However, understanding how features evolve and are distributed across incremental tasks remains an open challenge. In this paper, we propose a novel approach to modeling feature evolution in PTM-based CIL through the lens of neural collapse (NC), a striking phenomenon observed in the final phase of training, which leads to a well-separated, equiangular feature space. We explore the connection between NC and CIL effectiveness, showing that aligning feature distributions with the NC geometry enhances the ability to capture the dynamic behavior of continual learning. Based on this insight, we introduce Neural Collapse-inspired Pre-Trained Model-based CIL (NCPTM-CIL), a method that dynamically adjusts the feature space to conform to the elegant NC structure, thereby enhancing the continual learning process. Extensive experiments demonstrate that NCPTM-CIL outperforms state-of-the-art methods across four benchmark datasets. Notably, when initialized with ViT-B/16-IN1K, NCPTM-CIL surpasses the runner-up method by 6.73% on VTAB, 1.25% on CIFAR-100, and 2.5% on OmniBenchmark. 

**Abstract (ZH)**: 基于神经衰减的预训练模型增量学习（NCPTM-CIL） 

---
# Kimi-Audio Technical Report 

**Title (ZH)**: Kimi-Audio技术报告 

**Authors**: KimiTeam, Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, Zhengtao Wang, Chu Wei, Yifei Xin, Xinran Xu, Jianwei Yu, Yutao Zhang, Xinyu Zhou, Y. Charles, Jun Chen, Yanru Chen, Yulun Du, Weiran He, Zhenxing Hu, Guokun Lai, Qingcheng Li, Yangyang Liu, Weidong Sun, Jianzhou Wang, Yuzhi Wang, Yuefeng Wu, Yuxin Wu, Dongchao Yang, Hao Yang, Ying Yang, Zhilin Yang, Aoxiong Yin, Ruibin Yuan, Yutong Zhang, Zaida Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.18425)  

**Abstract**: We present Kimi-Audio, an open-source audio foundation model that excels in audio understanding, generation, and conversation. We detail the practices in building Kimi-Audio, including model architecture, data curation, training recipe, inference deployment, and evaluation. Specifically, we leverage a 12.5Hz audio tokenizer, design a novel LLM-based architecture with continuous features as input and discrete tokens as output, and develop a chunk-wise streaming detokenizer based on flow matching. We curate a pre-training dataset that consists of more than 13 million hours of audio data covering a wide range of modalities including speech, sound, and music, and build a pipeline to construct high-quality and diverse post-training data. Initialized from a pre-trained LLM, Kimi-Audio is continual pre-trained on both audio and text data with several carefully designed tasks, and then fine-tuned to support a diverse of audio-related tasks. Extensive evaluation shows that Kimi-Audio achieves state-of-the-art performance on a range of audio benchmarks including speech recognition, audio understanding, audio question answering, and speech conversation. We release the codes, model checkpoints, as well as the evaluation toolkits in this https URL. 

**Abstract (ZH)**: 我们介绍了Kimi-Audio，一个在音频理解、生成和对话方面表现出色的开源音频基础模型。我们详细介绍了Kimi-Audio的构建实践，包括模型架构、数据整理、训练方法、推理部署和评估。具体来说，我们利用了一个12.5Hz的音频分词器，设计了一种基于LLM的新架构，以连续特征作为输入，离散令牌作为输出，并开发了一种基于流动匹配的分块流式反分词器。我们整理了一个包含超过1300万小时音频数据的预训练数据集，涵盖包括语音、声音和音乐在内的多种模态，并构建了一条管线来构建高质量和多样化的后训练数据。从预训练的LLM初始化后，Kimi-Audio在音频和文本数据上进行了连续预训练，并通过精心设计的任务进行微调，以支持各种音频相关的任务。广泛评估表明，Kimi-Audio在包括语音识别、音频理解、音频问答和语音对话在内的多种音频基准上取得了最先进的性能。我们在此处发布了代码、模型检查点以及评估工具包。 

---
# LLMpatronous: Harnessing the Power of LLMs For Vulnerability Detection 

**Title (ZH)**: LLMpatronous: 利用大语言模型进行漏洞检测 

**Authors**: Rajesh Yarra  

**Link**: [PDF](https://arxiv.org/pdf/2504.18423)  

**Abstract**: Despite the transformative impact of Artificial Intelligence (AI) across various sectors, cyber security continues to rely on traditional static and dynamic analysis tools, hampered by high false positive rates and superficial code comprehension. While generative AI offers promising automation capabilities for software development, leveraging Large Language Models (LLMs) for vulnerability detection presents unique challenges. This paper explores the potential and limitations of LLMs in identifying vulnerabilities, acknowledging inherent weaknesses such as hallucinations, limited context length, and knowledge cut-offs. Previous attempts employing machine learning models for vulnerability detection have proven ineffective due to limited real-world applicability, feature engineering challenges, lack of contextual understanding, and the complexities of training models to keep pace with the evolving threat landscape. Therefore, we propose a robust AI-driven approach focused on mitigating these limitations and ensuring the quality and reliability of LLM based vulnerability detection. Through innovative methodologies combining Retrieval-Augmented Generation (RAG) and Mixtureof-Agents (MoA), this research seeks to leverage the strengths of LLMs while addressing their weaknesses, ultimately paving the way for dependable and efficient AI-powered solutions in securing the ever-evolving software landscape. 

**Abstract (ZH)**: 尽管人工智能（AI）在各领域产生了革命性的影响，网络安全仍依赖于传统的静态和动态分析工具，受到高误报率和表面化的代码理解限制。虽然生成型AI为软件开发提供了有前景的自动化能力，利用大规模语言模型（LLMs）进行漏洞检测面临着独特挑战。本文探讨了LLMs在识别漏洞方面的潜力和局限性，承认其固有的弱点，如幻觉、有限的上下文长度和知识截止。以往使用机器学习模型进行漏洞检测的尝试由于实际应用有限、特征工程挑战、缺乏上下文理解以及模型训练难以跟上不断演变的威胁态势而效果不佳。因此，本文提出了一种稳健的AI驱动方法，旨在减轻这些局限性，确保基于LLMs的漏洞检测的质量和可靠性。通过结合检索增强生成（RAG）和多智能体（MoA）的研究方法，本文旨在发挥LLMs的优势并解决其弱点，最终为确保不断演变的软件环境提供可靠且高效的AI增强解决方案。 

---
# A Multimodal Hybrid Late-Cascade Fusion Network for Enhanced 3D Object Detection 

**Title (ZH)**: 增强三维目标检测的多模态混合晚cascade融合网络 

**Authors**: Carlo Sgaravatti, Roberto Basla, Riccardo Pieroni, Matteo Corno, Sergio M. Savaresi, Luca Magri, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2504.18419)  

**Abstract**: We present a new way to detect 3D objects from multimodal inputs, leveraging both LiDAR and RGB cameras in a hybrid late-cascade scheme, that combines an RGB detection network and a 3D LiDAR detector. We exploit late fusion principles to reduce LiDAR False Positives, matching LiDAR detections with RGB ones by projecting the LiDAR bounding boxes on the image. We rely on cascade fusion principles to recover LiDAR False Negatives leveraging epipolar constraints and frustums generated by RGB detections of separate views. Our solution can be plugged on top of any underlying single-modal detectors, enabling a flexible training process that can take advantage of pre-trained LiDAR and RGB detectors, or train the two branches separately. We evaluate our results on the KITTI object detection benchmark, showing significant performance improvements, especially for the detection of Pedestrians and Cyclists. 

**Abstract (ZH)**: 我们提出了一种新的多模态输入下检测3D物体的方法，利用混合晚阶段级联方案结合RGB相机和LiDAR检测器。我们利用晚融合原则减少LiDAR假阳性，通过将LiDAR边界框投影到图像上来匹配RGB检测结果。我们依靠级联融合原则利用单视图RGB检测产生的透视约束和截锥体来恢复LiDAR假阴性。该解决方案可以叠加在任何底层单模态检测器之上，具有灵活的训练过程，可以利用预训练的LiDAR和RGB检测器，或单独训练两个分支。我们在KITTI物体检测基准上评估了我们的结果，显示出显著的性能提升，特别是在行人的检测方面。 

---
# Paradigm shift on Coding Productivity Using GenAI 

**Title (ZH)**: 使用生成式人工智能促进编码生产力范式转移 

**Authors**: Liang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18404)  

**Abstract**: Generative AI (GenAI) applications are transforming software engineering by enabling automated code co-creation. However, empirical evidence on GenAI's productivity effects in industrial settings remains limited. This paper investigates the adoption of GenAI coding assistants (e.g., Codeium, Amazon Q) within telecommunications and FinTech domains. Through surveys and interviews with industrial domain-experts, we identify primary productivity-influencing factors, including task complexity, coding skills, domain knowledge, and GenAI integration. Our findings indicate that GenAI tools enhance productivity in routine coding tasks (e.g., refactoring and Javadoc generation) but face challenges in complex, domain-specific activities due to limited context-awareness of codebases and insufficient support for customized design rules. We highlight new paradigms for coding transfer, emphasizing iterative prompt refinement, immersive development environment, and automated code evaluation as essential for effective GenAI usage. 

**Abstract (ZH)**: Generative AI编码助手在电信和金融科技领域的采用及其生产力影响：基于工业环境的实证研究 

---
# A Multimodal Deep Learning Approach for White Matter Shape Prediction in Diffusion MRI Tractography 

**Title (ZH)**: 多模态深度学习方法在扩散磁共振成像tractography中白质形状预测 

**Authors**: Yui Lo, Yuqian Chen, Dongnan Liu, Leo Zekelman, Jarrett Rushmore, Yogesh Rathi, Nikos Makris, Alexandra J. Golby, Fan Zhang, Weidong Cai, Lauren J. O'Donnell  

**Link**: [PDF](https://arxiv.org/pdf/2504.18400)  

**Abstract**: Shape measures have emerged as promising descriptors of white matter tractography, offering complementary insights into anatomical variability and associations with cognitive and clinical phenotypes. However, conventional methods for computing shape measures are computationally expensive and time-consuming for large-scale datasets due to reliance on voxel-based representations. We propose Tract2Shape, a novel multimodal deep learning framework that leverages geometric (point cloud) and scalar (tabular) features to predict ten white matter tractography shape measures. To enhance model efficiency, we utilize a dimensionality reduction algorithm for the model to predict five primary shape components. The model is trained and evaluated on two independently acquired datasets, the HCP-YA dataset, and the PPMI dataset. We evaluate the performance of Tract2Shape by training and testing it on the HCP-YA dataset and comparing the results with state-of-the-art models. To further assess its robustness and generalization ability, we also test Tract2Shape on the unseen PPMI dataset. Tract2Shape outperforms SOTA deep learning models across all ten shape measures, achieving the highest average Pearson's r and the lowest nMSE on the HCP-YA dataset. The ablation study shows that both multimodal input and PCA contribute to performance gains. On the unseen testing PPMI dataset, Tract2Shape maintains a high Pearson's r and low nMSE, demonstrating strong generalizability in cross-dataset evaluation. Tract2Shape enables fast, accurate, and generalizable prediction of white matter shape measures from tractography data, supporting scalable analysis across datasets. This framework lays a promising foundation for future large-scale white matter shape analysis. 

**Abstract (ZH)**: 形状度量已作为白质追踪的有希望的描述符出现，提供了关于解剖变异性和与认知和临床表型关联的补充洞见。然而，传统形状度量计算方法因依赖体素表示而在大规模数据集中计算昂贵且耗时。我们提出了一种名为Tract2Shape的新型多模态深度学习框架，利用几何（点云）和标量（表格）特征来预测十种白质追踪的形状度量。为了提高模型效率，我们利用降维算法，使模型能够预测五个主要形状组件。该模型在HCP-YA数据集和PPMI数据集上进行了训练和评估。我们通过在HCP-YA数据集上训练和测试Tract2Shape，并与最先进的模型进行比较，来评估其性能。为进一步评估其鲁棒性和泛化能力，我们也在未见的PPMI数据集上测试了Tract2Shape。Tract2Shape在所有十种形状度量上均优于最新的深度学习模型，在HCP-YA数据集上实现了最高的平均皮尔森相关系数和最低的nMSE。消融研究显示，多模态输入和主成分分析（PCA）均有助于性能提升。在未见的测试PPMI数据集上，Tract2Shape保持了高皮尔森相关系数和低nMSE，显示出强大的跨数据集泛化能力。Tract2Shape能够快速、准确、泛化地从追踪数据预测白质形状度量，支持跨数据集的可扩展分析。该框架为未来的大型白质形状分析奠定了有希望的基础。 

---
# Bridge the Domains: Large Language Models Enhanced Cross-domain Sequential Recommendation 

**Title (ZH)**: 跨越领域界限：大型语言模型增强跨域序列推荐 

**Authors**: Qidong Liu, Xiangyu Zhao, Yejing Wang, Zijian Zhang, Howard Zhong, Chong Chen, Xiang Li, Wei Huang, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2504.18383)  

**Abstract**: Cross-domain Sequential Recommendation (CDSR) aims to extract the preference from the user's historical interactions across various domains. Despite some progress in CDSR, two problems set the barrier for further advancements, i.e., overlap dilemma and transition complexity. The former means existing CDSR methods severely rely on users who own interactions on all domains to learn cross-domain item relationships, compromising the practicability. The latter refers to the difficulties in learning the complex transition patterns from the mixed behavior sequences. With powerful representation and reasoning abilities, Large Language Models (LLMs) are promising to address these two problems by bridging the items and capturing the user's preferences from a semantic view. Therefore, we propose an LLMs Enhanced Cross-domain Sequential Recommendation model (LLM4CDSR). To obtain the semantic item relationships, we first propose an LLM-based unified representation module to represent items. Then, a trainable adapter with contrastive regularization is designed to adapt the CDSR task. Besides, a hierarchical LLMs profiling module is designed to summarize user cross-domain preferences. Finally, these two modules are integrated into the proposed tri-thread framework to derive recommendations. We have conducted extensive experiments on three public cross-domain datasets, validating the effectiveness of LLM4CDSR. We have released the code online. 

**Abstract (ZH)**: 跨域序列推荐（CDSR）旨在从用户在不同领域的历史交互中提取偏好。尽管在CDSR方面取得了一些进展，但仍存在两个障碍，即重叠困境和转换复杂性。前者意味着现有的CDSR方法严重依赖于在所有领域都有交互记录的用户来学习跨领域的物品关系，这削弱了其实用性。后者指的是从混合行为序列中学习复杂的转换模式的困难。凭借强大的表示和推理能力，大型语言模型（LLMs）有望通过连接物品并从语义视角捕捉用户偏好来解决这两个问题。因此，我们提出了一种增强型跨域序列推荐模型（LLM4CDSR）。为了获取语义上的物品关系，我们首先提出了一种基于LLM的统一表示模块来表示物品。然后，设计了一个可训练的适配器，带有对比正则化，以适应CDSR任务。此外，设计了一个层次化的LLM用户概况模块，以总结用户的跨域偏好。最后，将这两个模块集成到提出的三线程框架中以生成推荐。我们已在三个公开的跨域数据集上进行了广泛实验，验证了LLM4CDSR的有效性，并已在线发布了代码。 

---
# Spatial Reasoner: A 3D Inference Pipeline for XR Applications 

**Title (ZH)**: 三维推理管道：XR应用中的空间 reasoning 

**Authors**: Steven Häsler, Philipp Ackermann  

**Link**: [PDF](https://arxiv.org/pdf/2504.18380)  

**Abstract**: Modern extended reality XR systems provide rich analysis of image data and fusion of sensor input and demand AR/VR applications that can reason about 3D scenes in a semantic manner. We present a spatial reasoning framework that bridges geometric facts with symbolic predicates and relations to handle key tasks such as determining how 3D objects are arranged among each other ('on', 'behind', 'near', etc.). Its foundation relies on oriented 3D bounding box representations, enhanced by a comprehensive set of spatial predicates, ranging from topology and connectivity to directionality and orientation, expressed in a formalism related to natural language. The derived predicates form a spatial knowledge graph and, in combination with a pipeline-based inference model, enable spatial queries and dynamic rule evaluation. Implementations for client- and server-side processing demonstrate the framework's capability to efficiently translate geometric data into actionable knowledge, ensuring scalable and technology-independent spatial reasoning in complex 3D environments. The Spatial Reasoner framework is fostering the creation of spatial ontologies, and seamlessly integrates with and therefore enriches machine learning, natural language processing, and rule systems in XR applications. 

**Abstract (ZH)**: 现代扩展现实XR系统提供了丰富的图像数据分析和传感器输入融合，需要能够以语义方式推理3D场景的AR/VR应用。我们提出了一种空间推理框架，将几何事实与符号谓词和关系相结合，处理诸如确定3D对象彼此如何排列（'在...上'、'在...后面'、'在...附近'等）的关键任务。该框架的基础依赖于定向的3D边界框表示，并通过一系列空间谓词得到了增强，这些谓词涵盖了拓扑、连接性、方向性和方向信息，并使用与自然语言相关的形式化表示。得出的谓词形成了空间知识图谱，并结合基于流水线的推理模型，实现了空间查询和动态规则评估。客户端和服务器端的实现证明了该框架能够高效地将几何数据转换为可操作的知识，确保在复杂的3D环境中实现可扩展且技术独立的空间推理。空间推理框架促进了空间本体的创建，并无缝集成和丰富了XR应用中的机器学习、自然语言处理和规则系统。 

---
# Pushing the boundary on Natural Language Inference 

**Title (ZH)**: 扩展自然语言推理的边界 

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho  

**Link**: [PDF](https://arxiv.org/pdf/2504.18376)  

**Abstract**: Natural Language Inference (NLI) is a central task in natural language understanding with applications in fact-checking, question answering, and information retrieval. Despite its importance, current NLI systems heavily rely on supervised learning with datasets that often contain annotation artifacts and biases, limiting generalization and real-world applicability. In this work, we apply a reinforcement learning-based approach using Group Relative Policy Optimization (GRPO) for Chain-of-Thought (CoT) learning in NLI, eliminating the need for labeled rationales and enabling this type of training on more challenging datasets such as ANLI. We fine-tune 7B, 14B, and 32B language models using parameter-efficient techniques (LoRA and QLoRA), demonstrating strong performance across standard and adversarial NLI benchmarks. Our 32B AWQ-quantized model surpasses state-of-the-art results on 7 out of 11 adversarial sets$\unicode{x2013}$or on all of them considering our replication$\unicode{x2013}$within a 22GB memory footprint, showing that robust reasoning can be retained under aggressive quantization. This work provides a scalable and practical framework for building robust NLI systems without sacrificing inference quality. 

**Abstract (ZH)**: 基于强化学习的Group Relative Policy Optimization (GRPO)在自然语言推理中的链式思考学习：无需标注理由且适用于ANLI等更具挑战性的数据集 

---
# COCO-Inpaint: A Benchmark for Image Inpainting Detection and Manipulation Localization 

**Title (ZH)**: COCO-Inpaint: 一种图像 inpaint 检测和操作定位基准 

**Authors**: Haozhen Yan, Yan Hong, Jiahui Zhan, Yikun Ji, Jun Lan, Huijia Zhu, Weiqiang Wang, Jianfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18361)  

**Abstract**: Recent advancements in image manipulation have achieved unprecedented progress in generating photorealistic content, but also simultaneously eliminating barriers to arbitrary manipulation and editing, raising concerns about multimedia authenticity and cybersecurity. However, existing Image Manipulation Detection and Localization (IMDL) methodologies predominantly focus on splicing or copy-move forgeries, lacking dedicated benchmarks for inpainting-based manipulations. To bridge this gap, we present COCOInpaint, a comprehensive benchmark specifically designed for inpainting detection, with three key contributions: 1) High-quality inpainting samples generated by six state-of-the-art inpainting models, 2) Diverse generation scenarios enabled by four mask generation strategies with optional text guidance, and 3) Large-scale coverage with 258,266 inpainted images with rich semantic diversity. Our benchmark is constructed to emphasize intrinsic inconsistencies between inpainted and authentic regions, rather than superficial semantic artifacts such as object shapes. We establish a rigorous evaluation protocol using three standard metrics to assess existing IMDL approaches. The dataset will be made publicly available to facilitate future research in this area. 

**Abstract (ZH)**: Recent advancements in image manipulation have achieved unprecedented progress in generating photorealistic content, but also simultaneously eliminating barriers to arbitrary manipulation and editing, raising concerns about multimedia authenticity and cybersecurity. However, existing Image Manipulation Detection and Localization (IMDL) methodologies predominantly focus on splicing or copy-move forgeries, lacking dedicated benchmarks for inpainting-based manipulations. To bridge this gap, we present COCOInpaint, a comprehensive benchmark specifically designed for inpainting detection, with three key contributions: 1) 高质量的由六种先进 inpainting 模型生成的 inpainting 样本，2) 通过四种不同的遮罩生成策略实现的多样化生成场景，可选配文本指导，3) 包含258,266张富有语义多样性的 inpainted 图像的大规模覆盖。我们的基准侧重于强调 inpainted 区域与真实区域之间的内在不一致性，而不是表面的语义伪影如物体形状。我们利用三种标准度量建立了严格的评估协议，以评估现有的图像伪造检测和定位方法。数据集将公开发布，以促进该领域的未来研究。 

---
# Testing Individual Fairness in Graph Neural Networks 

**Title (ZH)**: 测试图神经网络的个体公平性 

**Authors**: Roya Nasiri  

**Link**: [PDF](https://arxiv.org/pdf/2504.18353)  

**Abstract**: The biases in artificial intelligence (AI) models can lead to automated decision-making processes that discriminate against groups and/or individuals based on sensitive properties such as gender and race. While there are many studies on diagnosing and mitigating biases in various AI models, there is little research on individual fairness in Graph Neural Networks (GNNs). Unlike traditional models, which treat data features independently and overlook their inter-relationships, GNNs are designed to capture graph-based structure where nodes are interconnected. This relational approach enables GNNs to model complex dependencies, but it also means that biases can propagate through these connections, complicating the detection and mitigation of individual fairness violations. This PhD project aims to develop a testing framework to assess and ensure individual fairness in GNNs. It first systematically reviews the literature on individual fairness, categorizing existing approaches to define, measure, test, and mitigate model biases, creating a taxonomy of individual fairness. Next, the project will develop a framework for testing and ensuring fairness in GNNs by adapting and extending current fairness testing and mitigation techniques. The framework will be evaluated through industrial case studies, focusing on graph-based large language models. 

**Abstract (ZH)**: 人工intelligence模型中的偏差可能导致基于性别、种族等敏感属性对群体和个人进行歧视性的自动决策过程。虽然已有许多研究专注于诊断和减轻各种AI模型中的偏差，但对于图神经网络（GNNs）中的个体公平性研究却甚少。与传统模型独立处理数据特征并忽略其相互关系不同，GNNs旨在捕捉节点间相连的图结构。这种关系方法使得GNNs能够建模复杂的依赖关系，但也意味着偏见可以通过这些连接传播，增加了检测和减轻个体公平性侵犯的复杂性。本博士项目旨在开发一个测试框架以评估和确保GNNs中的个体公平性。项目首先系统地回顾个体公平性的文献，分类现有的方法以定义、衡量、测试和减轻模型偏差，创建个体公平性的分类框架。之后，项目将开发一个用于测试和确保GNNs公平性的框架，通过适应和扩展当前公平性测试和缓解技术来实现。该框架将通过基于图的大语言模型的工业案例研究进行评估。 

---
# TSCL:Multi-party loss Balancing scheme for deep learning Image steganography based on Curriculum learning 

**Title (ZH)**: TSCL：基于 Curriculum 学习的多 party 损失均衡的深度学习图像隐写术 

**Authors**: Fengchun Liu. Tong Zhang, Chunying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18348)  

**Abstract**: For deep learning-based image steganography frameworks, in order to ensure the invisibility and recoverability of the information embedding, the loss function usually contains several losses such as embedding loss, recovery loss and steganalysis loss. In previous research works, fixed loss weights are usually chosen for training optimization, and this setting is not linked to the importance of the steganography task itself and the training process. In this paper, we propose a Two-stage Curriculum Learning loss scheduler (TSCL) for balancing multinomial losses in deep learning image steganography algorithms. TSCL consists of two phases: a priori curriculum control and loss dynamics control. The first phase firstly focuses the model on learning the information embedding of the original image by controlling the loss weights in the multi-party adversarial training; secondly, it makes the model shift its learning focus to improving the decoding accuracy; and finally, it makes the model learn to generate a steganographic image that is resistant to steganalysis. In the second stage, the learning speed of each training task is evaluated by calculating the loss drop of the before and after iteration rounds to balance the learning of each task. Experimental results on three large public datasets, ALASKA2, VOC2012 and ImageNet, show that the proposed TSCL strategy improves the quality of steganography, decoding accuracy and security. 

**Abstract (ZH)**: 基于深度学习的图像隐写分析Loss调度的两阶段 Curriculum学习方法 

---
# Comparing Uncertainty Measurement and Mitigation Methods for Large Language Models: A Systematic Review 

**Title (ZH)**: 大规模语言模型中的不确定性测量与缓解方法比较：一项系统性综述 

**Authors**: Toghrul Abbasli, Kentaroh Toyoda, Yuan Wang, Leon Witt, Muhammad Asif Ali, Yukai Miao, Dan Li, Qingsong Wei  

**Link**: [PDF](https://arxiv.org/pdf/2504.18346)  

**Abstract**: Large Language Models (LLMs) have been transformative across many domains. However, hallucination -- confidently outputting incorrect information -- remains one of the leading challenges for LLMs. This raises the question of how to accurately assess and quantify the uncertainty of LLMs. Extensive literature on traditional models has explored Uncertainty Quantification (UQ) to measure uncertainty and employed calibration techniques to address the misalignment between uncertainty and accuracy. While some of these methods have been adapted for LLMs, the literature lacks an in-depth analysis of their effectiveness and does not offer a comprehensive benchmark to enable insightful comparison among existing solutions. In this work, we fill this gap via a systematic survey of representative prior works on UQ and calibration for LLMs and introduce a rigorous benchmark. Using two widely used reliability datasets, we empirically evaluate six related methods, which justify the significant findings of our review. Finally, we provide outlooks for key future directions and outline open challenges. To the best of our knowledge, this survey is the first dedicated study to review the calibration methods and relevant metrics for LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在许多领域中起到了变革性的作用。然而，模型妄言——自信地输出错误信息——仍然是LLMs面临的主要挑战之一。这引发了如何准确评估和量化LLMs的不确定性的问题。传统模型领域的大量文献探讨了不确定性量化（UQ）来衡量不确定性，并采用了校准技术来解决不确定性与准确性的不一致问题。虽然其中一些方法已经被调整适用于LLMs，但文献中缺乏这些方法有效性的深入分析，也没有提供综合基准以实现现有解决方案的深入比较。在本文中，我们通过系统回顾先前关于LLMs的UQ和校准工作的代表性文献，并引入了一个严谨的基准。使用两个广泛采用的可靠性数据集，我们实证评估了六种相关方法，从而验证了我们回顾的重要发现。最后，我们提出了未来研究的关键方向，并概述了开放挑战。据我们所知，这是首个专注于回顾LLMs的校准方法及其相关指标的研究。 

---
# PHEATPRUNER: Interpretable Data-centric Feature Selection for Multivariate Time Series Classification through Persistent Homology 

**Title (ZH)**: PHEATPRUNER: 基于持久同调的可解释多变量时间序列分类数据中心特征选择 

**Authors**: Anh-Duy Pham, Olivier Basole Kashongwe, Martin Atzmueller, Tim Römer  

**Link**: [PDF](https://arxiv.org/pdf/2504.18329)  

**Abstract**: Balancing performance and interpretability in multivariate time series classification is a significant challenge due to data complexity and high dimensionality. This paper introduces PHeatPruner, a method integrating persistent homology and sheaf theory to address these challenges. Persistent homology facilitates the pruning of up to 45% of the applied variables while maintaining or enhancing the accuracy of models such as Random Forest, CatBoost, XGBoost, and LightGBM, all without depending on posterior probabilities or supervised optimization algorithms. Concurrently, sheaf theory contributes explanatory vectors that provide deeper insights into the data's structural nuances. The approach was validated using the UEA Archive and a mastitis detection dataset for dairy cows. The results demonstrate that PHeatPruner effectively preserves model accuracy. Furthermore, our results highlight PHeatPruner's key features, i.e. simplifying complex data and offering actionable insights without increasing processing time or complexity. This method bridges the gap between complexity reduction and interpretability, suggesting promising applications in various fields. 

**Abstract (ZH)**: 在多变量时间序列分类中平衡性能和可解释性是一项由于数据复杂性和高维度而面临的重大挑战。本文介绍了一种结合持久同调和层理论的PHeatPruner方法，以应对这些挑战。 

---
# Towards Adaptive Software Agents for Debugging 

**Title (ZH)**: 面向自适应软件代理的调试研究 

**Authors**: Yacine Majdoub, Eya Ben Charrada, Haifa Touati  

**Link**: [PDF](https://arxiv.org/pdf/2504.18316)  

**Abstract**: Using multiple agents was found to improve the debugging capabilities of Large Language Models. However, increasing the number of LLM-agents has several drawbacks such as increasing the running costs and rising the risk for the agents to lose focus. In this work, we propose an adaptive agentic design, where the number of agents and their roles are determined dynamically based on the characteristics of the task to be achieved. In this design, the agents roles are not predefined, but are generated after analyzing the problem to be solved. Our initial evaluation shows that, with the adaptive design, the number of agents that are generated depends on the complexity of the buggy code. In fact, for simple code with mere syntax issues, the problem was usually fixed using one agent only. However, for more complex problems, we noticed the creation of a higher number of agents. Regarding the effectiveness of the fix, we noticed an average improvement of 11% compared to the one-shot prompting. Given these promising results, we outline future research directions to improve our design for adaptive software agents that can autonomously plan and conduct their software goals. 

**Abstract (ZH)**: 使用多个代理被发现能够提高大型语言模型的调试能力。然而，增加LLM代理的数量也带来了一些缺点，如增加运行成本和增加代理失去焦点的风险。在此工作中，我们提出了一个自适应代理设计，其中代理的数量及其角色根据要实现的任务特性动态确定。在此设计中，代理的角色并非预先定义，而是在分析待解决问题后生成。初步评估显示，随着自适应设计的应用，生成的代理数量取决于存在错误代码的复杂性。实际上，对于仅有语法问题的简单代码，通常只需一个代理即可解决问题。然而，对于更复杂的问题，我们观察到代理的数量增加。关于修复的有效性，我们发现与单次提示相比，修复效果平均提高了11%。鉴于这些有前景的结果，我们概述了未来的研究方向，以改进适应性软件代理的设计，使其能够自主规划和实现其软件目标。 

---
# Artificial Intelligence health advice accuracy varies across languages and contexts 

**Title (ZH)**: 人工智能健康建议的准确性在不同语言和背景下有所差异 

**Authors**: Prashant Garg, Thiemo Fetzer  

**Link**: [PDF](https://arxiv.org/pdf/2504.18310)  

**Abstract**: Using basic health statements authorized by UK and EU registers and 9,100 journalist-vetted public-health assertions on topics such as abortion, COVID-19 and politics from sources ranging from peer-reviewed journals and government advisories to social media and news across the political spectrum, we benchmark six leading large language models from in 21 languages, finding that, despite high accuracy on English-centric textbook claims, performance falls in multiple non-European languages and fluctuates by topic and source, highlighting the urgency of comprehensive multilingual, domain-aware validation before deploying AI in global health communication. 

**Abstract (ZH)**: 使用英国和欧盟注册机构授权的基本健康声明以及9100条经记者和专家审核的公共健康主张，涵盖堕胎、COVID-19和政治等话题，来源包括同行评审期刊、政府建议、社交媒体和政治光谱内的新闻，我们对21种语言的六种领先大型语言模型进行了基准测试，发现尽管在以英语为中心的教科书主张方面表现准确，但在多种非欧洲语言中的表现却下降，并且在不同话题和来源之间波动，这强调了在全球健康沟通中部署AI之前进行全面多语言、领域意识验证的紧迫性。 

---
# Enhancing Long-Term Re-Identification Robustness Using Synthetic Data: A Comparative Analysis 

**Title (ZH)**: 使用合成数据增强长期重识别鲁棒性：一种比较分析 

**Authors**: Christian Pionzewski, Rebecca Rademacher, Jérôme Rutinowski, Antonia Ponikarov, Stephan Matzke, Tim Chilla, Pia Schreynemackers, Alice Kirchheim  

**Link**: [PDF](https://arxiv.org/pdf/2504.18286)  

**Abstract**: This contribution explores the impact of synthetic training data usage and the prediction of material wear and aging in the context of re-identification. Different experimental setups and gallery set expanding strategies are tested, analyzing their impact on performance over time for aging re-identification subjects. Using a continuously updating gallery, we were able to increase our mean Rank-1 accuracy by 24%, as material aging was taken into account step by step. In addition, using models trained with 10% artificial training data, Rank-1 accuracy could be increased by up to 13%, in comparison to a model trained on only real-world data, significantly boosting generalized performance on hold-out data. Finally, this work introduces a novel, open-source re-identification dataset, pallet-block-2696. This dataset contains 2,696 images of Euro pallets, taken over a period of 4 months. During this time, natural aging processes occurred and some of the pallets were damaged during their usage. These wear and tear processes significantly changed the appearance of the pallets, providing a dataset that can be used to generate synthetically aged pallets or other wooden materials. 

**Abstract (ZH)**: 合成训练数据使用对重识别中材料磨损和老化预测的影响研究：一种不断更新的画廊及其在重识别中老化对象性能的影响分析，并介绍新型开源重识别数据集pallet-block-2696 

---
# Seeing Soundscapes: Audio-Visual Generation and Separation from Soundscapes Using Audio-Visual Separator 

**Title (ZH)**: 视听共生：基于视听分离器的声音景观的音视频生成与分离 

**Authors**: Minjae Kang, Martim Brandão  

**Link**: [PDF](https://arxiv.org/pdf/2504.18283)  

**Abstract**: Recent audio-visual generative models have made substantial progress in generating images from audio. However, existing approaches focus on generating images from single-class audio and fail to generate images from mixed audio. To address this, we propose an Audio-Visual Generation and Separation model (AV-GAS) for generating images from soundscapes (mixed audio containing multiple classes). Our contribution is threefold: First, we propose a new challenge in the audio-visual generation task, which is to generate an image given a multi-class audio input, and we propose a method that solves this task using an audio-visual separator. Second, we introduce a new audio-visual separation task, which involves generating separate images for each class present in a mixed audio input. Lastly, we propose new evaluation metrics for the audio-visual generation task: Class Representation Score (CRS) and a modified R@K. Our model is trained and evaluated on the VGGSound dataset. We show that our method outperforms the state-of-the-art, achieving 7% higher CRS and 4% higher R@2* in generating plausible images with mixed audio. 

**Abstract (ZH)**: Recent音频-视觉生成模型已在从音频生成图像方面取得了显著进展。然而，现有方法侧重于从单一类别的音频生成图像，而无法生成从混合音频生成的图像。为解决这一问题，我们提出了一个音频-视觉生成与分离模型（AV-GAS），用于从音景（包含多个类别的混合音频）生成图像。我们的贡献包括三个方面：首先，我们提出了音频-视觉生成任务中的一个新挑战，即给定一个多类别的音频输入生成图像，并提出了一种使用音频-视觉分离器解决问题的方法。其次，我们引入了一个新的音频-视觉分离任务，涉及为混合音频输入中存在的每个类别生成单独的图像。最后，我们提出了音频-视觉生成任务的新评估指标：类表示得分（CRS）和修改后的R@K。我们的模型在VGGSound数据集上进行了训练和评估。结果显示，我们的方法优于现有最佳方法，在从混合音频生成可信图像方面，CRS提高了7%，R@2*提高了4%。 

---
# Neural operators struggle to learn complex PDEs in pedestrian mobility: Hughes model case study 

**Title (ZH)**: 神经算子在行人流动性中难以学习复杂偏微分方程：豪 ug 斯模型案例研究 

**Authors**: Prajwal Chauhan, Salah Eddine Choutri, Mohamed Ghattassi, Nader Masmoudi, Saif Eddin Jabari  

**Link**: [PDF](https://arxiv.org/pdf/2504.18267)  

**Abstract**: This paper investigates the limitations of neural operators in learning solutions for a Hughes model, a first-order hyperbolic conservation law system for crowd dynamics. The model couples a Fokker-Planck equation representing pedestrian density with a Hamilton-Jacobi-type (eikonal) equation. This Hughes model belongs to the class of nonlinear hyperbolic systems that often exhibit complex solution structures, including shocks and discontinuities. In this study, we assess the performance of three state-of-the-art neural operators (Fourier Neural Operator, Wavelet Neural Operator, and Multiwavelet Neural Operator) in various challenging scenarios. Specifically, we consider (1) discontinuous and Gaussian initial conditions and (2) diverse boundary conditions, while also examining the impact of different numerical schemes.
Our results show that these neural operators perform well in easy scenarios with fewer discontinuities in the initial condition, yet they struggle in complex scenarios with multiple initial discontinuities and dynamic boundary conditions, even when trained specifically on such complex samples. The predicted solutions often appear smoother, resulting in a reduction in total variation and a loss of important physical features. This smoothing behavior is similar to issues discussed by Daganzo (1995), where models that introduce artificial diffusion were shown to miss essential features such as shock waves in hyperbolic systems. These results suggest that current neural operator architectures may introduce unintended regularization effects that limit their ability to capture transport dynamics governed by discontinuities. They also raise concerns about generalizing these methods to traffic applications where shock preservation is essential. 

**Abstract (ZH)**: 本文探讨了神经运算符在学习Hughes模型解中的局限性，Hughes模型是用于 crowd dynamics 的一个一阶双曲守恒律系统，该模型将表示行人密度的Fokker-Planck方程与Hamilton-Jacobi类型（eikonal）方程耦合。本文评估了四种先进神经运算符（Fourier神经运算符、小波神经运算符和多重小波神经运算符）在各种具有挑战性的场景中的性能，具体考虑了具有不连续和高斯初始条件以及多样化的边界条件的情形，并考察了不同数值方案的影响。 

---
# Depth-Constrained ASV Navigation with Deep RL and Limited Sensing 

**Title (ZH)**: 深度约束的ASV导航：基于深度RL和有限感知的方法 

**Authors**: Amirhossein Zhalehmehrabi, Daniele Meli, Francesco Dal Santo, Francesco Trotti, Alessandro Farinelli  

**Link**: [PDF](https://arxiv.org/pdf/2504.18253)  

**Abstract**: Autonomous Surface Vehicles (ASVs) play a crucial role in maritime operations, yet their navigation in shallow-water environments remains challenging due to dynamic disturbances and depth constraints. Traditional navigation strategies struggle with limited sensor information, making safe and efficient operation difficult. In this paper, we propose a reinforcement learning (RL) framework for ASV navigation under depth constraints, where the vehicle must reach a target while avoiding unsafe areas with only a single depth measurement per timestep from a downward-facing Single Beam Echosounder (SBES). To enhance environmental awareness, we integrate Gaussian Process (GP) regression into the RL framework, enabling the agent to progressively estimate a bathymetric depth map from sparse sonar readings. This approach improves decision-making by providing a richer representation of the environment. Furthermore, we demonstrate effective sim-to-real transfer, ensuring that trained policies generalize well to real-world aquatic conditions. Experimental results validate our method's capability to improve ASV navigation performance while maintaining safety in challenging shallow-water environments. 

**Abstract (ZH)**: 自主水面车辆在浅水环境下受深度约束的导航研究：基于强化学习的方法 

---
# Event-Based Eye Tracking. 2025 Event-based Vision Workshop 

**Title (ZH)**: 基于事件的注视跟踪。2025基于事件的视觉研讨会 

**Authors**: Qinyu Chen, Chang Gao, Min Liu, Daniele Perrone, Yan Ru Pei, Zuowen Wang, Zhuo Zou, Shihang Tan, Tao Han, Guorui Lu, Zhen Xu, Junyuan Ding, Ziteng Wang, Zongwei Wu, Han Han, Yuliang Wu, Jinze Chen, Wei Zhai, Yang Cao, Zheng-jun Zha, Nuwan Bandara, Thivya Kandappu, Archan Misra, Xiaopeng Lin, Hongxiang Huang, Hongwei Ren, Bojun Cheng, Hoang M. Truong, Vinh-Thuan Ly, Huy G. Tran, Thuan-Phat Nguyen, Tram T. Doan  

**Link**: [PDF](https://arxiv.org/pdf/2504.18249)  

**Abstract**: This survey serves as a review for the 2025 Event-Based Eye Tracking Challenge organized as part of the 2025 CVPR event-based vision workshop. This challenge focuses on the task of predicting the pupil center by processing event camera recorded eye movement. We review and summarize the innovative methods from teams rank the top in the challenge to advance future event-based eye tracking research. In each method, accuracy, model size, and number of operations are reported. In this survey, we also discuss event-based eye tracking from the perspective of hardware design. 

**Abstract (ZH)**: 2025事件驱动眼动追踪挑战综述：面向CVPR事件驱动视觉研讨会的回顾与总结 

---
# Efficient Single-Pass Training for Multi-Turn Reasoning 

**Title (ZH)**: 单过训练高效实现多轮推理 

**Authors**: Ritesh Goru, Shanay Mehta, Prateek Jain  

**Link**: [PDF](https://arxiv.org/pdf/2504.18246)  

**Abstract**: Training Large Language Models ( LLMs) to generate explicit reasoning before they produce an answer has been shown to improve their performance across various tasks such as mathematics and coding. However, fine-tuning LLMs on multi-turn reasoning datasets presents a unique challenge: LLMs must generate reasoning tokens that are excluded from subsequent inputs to the LLM. This discrepancy prevents us from processing an entire conversation in a single forward pass-an optimization readily available when we fine-tune on a multi-turn non-reasoning dataset. This paper proposes a novel approach that overcomes this limitation through response token duplication and a custom attention mask that enforces appropriate visibility constraints. Our approach significantly reduces the training time and allows efficient fine-tuning on multi-turn reasoning datasets. 

**Abstract (ZH)**: 训练大型语言模型在生成答案之前生成明确的推理已被证明能提高其在数学和编码等各项任务中的性能。然而，对多轮推理数据集进行微调为LLMs带来了一个独特挑战：LLMs必须生成在后续输入中被排除的推理标记。这种不一致阻止了我们一次性处理整个对话——这是我们在使用多轮非推理数据集进行微调时可以利用的优化。本文提出了一种新颖的方法，通过响应标记复制和自定义注意力掩码克服这一限制，该掩码施加适当可见性约束。我们的方法显著缩短了训练时间，并允许在多轮推理数据集上进行高效的微调。 

---
# Time and Frequency Domain-based Anomaly Detection in Smart Meter Data for Distribution Network Studies 

**Title (ZH)**: 基于时间和频率域的智能电表数据异常检测在配电网研究中的应用 

**Authors**: Petar Labura, Tomislav Antic, Tomislav Capuder  

**Link**: [PDF](https://arxiv.org/pdf/2504.18231)  

**Abstract**: The widespread integration of new technologies in low-voltage distribution networks on the consumer side creates the need for distribution system operators to perform advanced real-time calculations to estimate network conditions. In recent years, data-driven models based on machine learning and big data analysis have emerged for calculation purposes, leveraging the information available in large datasets obtained from smart meters and other advanced measurement infrastructure. However, existing data-driven algorithms do not take into account the quality of data collected from smart meters. They lack built-in anomaly detection mechanisms and fail to differentiate anomalies based on whether the value or context of anomalous data instances deviates from the norm. This paper focuses on methods for detecting and mitigating the impact of anomalies on the consumption of active and reactive power datasets. It proposes an anomaly detection framework based on the Isolation Forest machine learning algorithm and Fast Fourier Transform filtering that works in both the time and frequency domain and is unaffected by point anomalies or contextual anomalies of the power consumption data. The importance of integrating anomaly detection methods is demonstrated in the analysis important for distribution networks with a high share of smart meters. 

**Abstract (ZH)**: 低电压配电网络中消费侧新技術的广泛集成促使配电网运营商需要进行高级实时计算以估计网络状况。近年来，基于机器学习和大数据分析的数据驱动模型在计算中崭露头角，利用智能电表和其他先进测量基础设施获得的大数据集中的信息。然而，现有的数据驱动算法并未考虑到从智能电表收集的数据质量。它们缺乏内置的异常检测机制，无法根据异常数据实例的价值或上下文是否偏离常态来区分异常。本文专注于检测和减轻异常对有功和无功功率数据消耗影响的方法，提出了一种基于孤立森林机器学习算法和快速傅里叶变换滤波的异常检测框架，该框架在时频域均有效，不受电能消耗数据点异常或上下文异常的影响。在高智能电表渗透率的配电网分析中，集成异常检测方法的重要性得到体现。 

---
# Learning to fuse: dynamic integration of multi-source data for accurate battery lifespan prediction 

**Title (ZH)**: 基于学习的融合：多源数据的动态集成以实现精确的电池寿命预测 

**Authors**: He Shanxuan, Lin Zuhong, Yu Bolun, Gao Xu, Long Biao, Yao Jingjing  

**Link**: [PDF](https://arxiv.org/pdf/2504.18230)  

**Abstract**: Accurate prediction of lithium-ion battery lifespan is vital for ensuring operational reliability and reducing maintenance costs in applications like electric vehicles and smart grids. This study presents a hybrid learning framework for precise battery lifespan prediction, integrating dynamic multi-source data fusion with a stacked ensemble (SE) modeling approach. By leveraging heterogeneous datasets from the National Aeronautics and Space Administration (NASA), Center for Advanced Life Cycle Engineering (CALCE), MIT-Stanford-Toyota Research Institute (TRC), and nickel cobalt aluminum (NCA) chemistries, an entropy-based dynamic weighting mechanism mitigates variability across heterogeneous datasets. The SE model combines Ridge regression, long short-term memory (LSTM) networks, and eXtreme Gradient Boosting (XGBoost), effectively capturing temporal dependencies and nonlinear degradation patterns. It achieves a mean absolute error (MAE) of 0.0058, root mean square error (RMSE) of 0.0092, and coefficient of determination (R2) of 0.9839, outperforming established baseline models with a 46.2% improvement in R2 and an 83.2% reduction in RMSE. Shapley additive explanations (SHAP) analysis identifies differential discharge capacity (Qdlin) and temperature of measurement (Temp_m) as critical aging indicators. This scalable, interpretable framework enhances battery health management, supporting optimized maintenance and safety across diverse energy storage systems, thereby contributing to improved battery health management in energy storage systems. 

**Abstract (ZH)**: 基于多重数据融合与级联ensembles模型的锂离子电池寿命精准预测方法 

---
# Multi-Grained Compositional Visual Clue Learning for Image Intent Recognition 

**Title (ZH)**: 多粒度组成性视觉线索学习在图像意图识别中的应用 

**Authors**: Yin Tang, Jiankai Li, Hongyu Yang, Xuan Dong, Lifeng Fan, Weixin Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.18201)  

**Abstract**: In an era where social media platforms abound, individuals frequently share images that offer insights into their intents and interests, impacting individual life quality and societal stability. Traditional computer vision tasks, such as object detection and semantic segmentation, focus on concrete visual representations, while intent recognition relies more on implicit visual clues. This poses challenges due to the wide variation and subjectivity of such clues, compounded by the problem of intra-class variety in conveying abstract concepts, e.g. "enjoy life". Existing methods seek to solve the problem by manually designing representative features or building prototypes for each class from global features. However, these methods still struggle to deal with the large visual diversity of each intent category. In this paper, we introduce a novel approach named Multi-grained Compositional visual Clue Learning (MCCL) to address these challenges for image intent recognition. Our method leverages the systematic compositionality of human cognition by breaking down intent recognition into visual clue composition and integrating multi-grained features. We adopt class-specific prototypes to alleviate data imbalance. We treat intent recognition as a multi-label classification problem, using a graph convolutional network to infuse prior knowledge through label embedding correlations. Demonstrated by a state-of-the-art performance on the Intentonomy and MDID datasets, our approach advances the accuracy of existing methods while also possessing good interpretability. Our work provides an attempt for future explorations in understanding complex and miscellaneous forms of human expression. 

**Abstract (ZH)**: 在社交媒体平台泛滥的时代，个体经常分享能够揭示其意图和兴趣的照片，影响个人生活质量和社会稳定性。传统计算机视觉任务，如对象检测和语义分割，侧重于具体的视觉表示，而意图识别则更多依赖于隐含的视觉线索。但由于这类线索存在广泛变化和主观性，且在传达抽象概念时存在类别内的多样性问题，如“享受生活”，现有方法通过手动设计代表性特征或从全局特征构建原型来解决问题，但仍难以应对每类意图的大量视觉多样性。在本文中，我们提出了一种新的方法，即多粒度组成性视觉线索学习（MCCL），以应对这些挑战。我们的方法通过分解意图识别为视觉线索组成，并结合多粒度特征，利用人类认知的系统组合性。我们采用类别特定的原型以缓解数据不平衡问题，并将意图识别视为一个多标签分类问题，通过图卷积网络引入标签嵌入关联的先验知识，展示了在Intentonomy和MDID数据集上的优异性能，同时提高了现有方法的准确性且具有良好的可解释性。我们的工作提供了未来探索理解和解释人类表达复杂多样的尝试。 

---
# Aligning Language Models for Icelandic Legal Text Summarization 

**Title (ZH)**: 冰岛法律文本摘要的语言模型对齐 

**Authors**: Þórir Hrafn Harðarson, Hrafn Loftsson, Stefán Ólafsson  

**Link**: [PDF](https://arxiv.org/pdf/2504.18180)  

**Abstract**: The integration of language models in the legal domain holds considerable promise for streamlining processes and improving efficiency in managing extensive workloads. However, the specialized terminology, nuanced language, and formal style of legal texts can present substantial challenges. This study examines whether preference-based training techniques, specifically Reinforcement Learning from Human Feedback and Direct Preference Optimization, can enhance models' performance in generating Icelandic legal summaries that align with domain-specific language standards and user preferences. We compare models fine-tuned with preference training to those using conventional supervised learning. Results indicate that preference training improves the legal accuracy of generated summaries over standard fine-tuning but does not significantly enhance the overall quality of Icelandic language usage. Discrepancies between automated metrics and human evaluations further underscore the importance of qualitative assessment in developing language models for the legal domain. 

**Abstract (ZH)**: 语言模型在法律领域的集成具有显著潜力，可以简化流程并提高管理大量工作负载的效率。然而，法律文本的专业术语、细腻的语言和正式风格为这一过程带来了显著挑战。本研究考察了基于偏好的训练技术，即人类反馈强化学习和直接偏好优化，是否能提高模型生成符合法律领域语言标准和用户偏好的冰岛法律摘要的能力。我们将使用偏好训练微调的模型与使用传统监督学习微调的模型进行对比。结果表明，偏好训练能够提高生成摘要的法律准确性，但并未显著提升冰岛语言使用的整体质量。自动化评估指标与人类评估之间的差异进一步强调了在开发法律领域语言模型时进行定性评估的重要性。 

---
# PerfCam: Digital Twinning for Production Lines Using 3D Gaussian Splatting and Vision Models 

**Title (ZH)**: PerfCam: 生产线数字孪生基于3D高斯点显示和视觉模型 

**Authors**: Michel Gokan Khan, Renan Guarese, Fabian Johnson, Xi Vincent Wang, Anders Bergman, Benjamin Edvinsson, Mario Romero, Jérémy Vachier, Jan Kronqvist  

**Link**: [PDF](https://arxiv.org/pdf/2504.18165)  

**Abstract**: We introduce PerfCam, an open source Proof-of-Concept (PoC) digital twinning framework that combines camera and sensory data with 3D Gaussian Splatting and computer vision models for digital twinning, object tracking, and Key Performance Indicators (KPIs) extraction in industrial production lines. By utilizing 3D reconstruction and Convolutional Neural Networks (CNNs), PerfCam offers a semi-automated approach to object tracking and spatial mapping, enabling digital twins that capture real-time KPIs such as availability, performance, Overall Equipment Effectiveness (OEE), and rate of conveyor belts in the production line. We validate the effectiveness of PerfCam through a practical deployment within realistic test production lines in the pharmaceutical industry and contribute an openly published dataset to support further research and development in the field. The results demonstrate PerfCam's ability to deliver actionable insights through its precise digital twin capabilities, underscoring its value as an effective tool for developing usable digital twins in smart manufacturing environments and extracting operational analytics. 

**Abstract (ZH)**: 我们介绍PerfCam，一个开源的概念验证（PoC）数字孪生框架，结合了相机和传感器数据、3D高斯点绘技术和计算机视觉模型，用于工业生产线的数字孪生、对象跟踪和关键绩效指标（KPI）提取。通过利用3D重建和卷积神经网络（CNNs），PerfCam 提供了一种半自动的对象跟踪和空间映射方法，能够捕捉到如可用性、性能、整体设备有效性（OEE）以及传送带速率等实时KPI的精确数字孪生。我们通过在制药行业中的实际测试生产线上部署PerfCam，验证了其有效性，并公开发布了一个数据集以支持该领域的进一步研究与开发。研究结果表明，PerfCam 通过其精确的数字孪生能力提供了可操作的见解，突显了其在智能制造环境中开发实用数字孪生和提取运营分析方面的价值。 

---
# Offline Learning of Controllable Diverse Behaviors 

**Title (ZH)**: 离线学习可控多样化行为 

**Authors**: Mathieu Petitbois, Rémy Portelas, Sylvain Lamprier, Ludovic Denoyer  

**Link**: [PDF](https://arxiv.org/pdf/2504.18160)  

**Abstract**: Imitation Learning (IL) techniques aim to replicate human behaviors in specific tasks. While IL has gained prominence due to its effectiveness and efficiency, traditional methods often focus on datasets collected from experts to produce a single efficient policy. Recently, extensions have been proposed to handle datasets of diverse behaviors by mainly focusing on learning transition-level diverse policies or on performing entropy maximization at the trajectory level. While these methods may lead to diverse behaviors, they may not be sufficient to reproduce the actual diversity of demonstrations or to allow controlled trajectory generation. To overcome these drawbacks, we propose a different method based on two key features: a) Temporal Consistency that ensures consistent behaviors across entire episodes and not just at the transition level as well as b) Controllability obtained by constructing a latent space of behaviors that allows users to selectively activate specific behaviors based on their requirements. We compare our approach to state-of-the-art methods over a diverse set of tasks and environments. Project page: this https URL 

**Abstract (ZH)**: 模仿学习（IL）技术旨在在一个特定任务中复制人类行为。尽管由于其有效性与高效性，模仿学习已获得广泛关注，但传统方法通常侧重于从专家收集的数据集以生成单一高效的策略。最近，已提出扩展方法来处理包含多样行为的数据集，这些方法主要关注学习过渡层面的多样策略或在轨迹层面进行熵最大化。虽然这些方法可能会导致多样行为，但它们可能不足以再现演示的实际多样性或允许受控轨迹生成。为克服这些缺点，我们提出了一种基于两个关键特征的不同方法：a) 时间一致性，确保整个 episodes 的行为一致性，而不仅仅是过渡层面的一致性；b) 可控性，通过构建行为的潜在空间，允许用户根据其需求选择性地激活特定行为。我们在多种任务和环境中将我们的方法与最先进的方法进行了比较。项目页面: this https URL 

---
# EDU-NER-2025: Named Entity Recognition in Urdu Educational Texts using XLM-RoBERTa with X (formerly Twitter) 

**Title (ZH)**: EDU-NER-2025: Urdu教育文本中的命名实体识别使用XLM-RoBERTa with X（ formerly Twitter） 

**Authors**: Fida Ullah, Muhammad Ahmad, Muhammad Tayyab Zamir, Muhammad Arif, Grigori sidorov, Edgardo Manuel Felipe Riverón, Alexander Gelbukh  

**Link**: [PDF](https://arxiv.org/pdf/2504.18142)  

**Abstract**: Named Entity Recognition (NER) plays a pivotal role in various Natural Language Processing (NLP) tasks by identifying and classifying named entities (NEs) from unstructured data into predefined categories such as person, organization, location, date, and time. While extensive research exists for high-resource languages and general domains, NER in Urdu particularly within domain-specific contexts like education remains significantly underexplored. This is Due to lack of annotated datasets for educational content which limits the ability of existing models to accurately identify entities such as academic roles, course names, and institutional terms, underscoring the urgent need for targeted resources in this domain. To the best of our knowledge, no dataset exists in the domain of the Urdu language for this purpose. To achieve this objective this study makes three key contributions. Firstly, we created a manually annotated dataset in the education domain, named EDU-NER-2025, which contains 13 unique most important entities related to education domain. Second, we describe our annotation process and guidelines in detail and discuss the challenges of labelling EDU-NER-2025 dataset. Third, we addressed and analyzed key linguistic challenges, such as morphological complexity and ambiguity, which are prevalent in formal Urdu texts. 

**Abstract (ZH)**: 命名实体识别(NER)在各种自然语言处理(NLP)任务中扮演着关键角色，通过从无结构数据中识别并分类命名实体(NEs)到预定义的类别，如人名、组织、地名、日期和时间。虽然高资源语言和通用领域的研究广泛存在，但特别是在教育等特定领域的乌尔都语命名实体识别仍然显著未被探索。这主要是由于缺乏教育内容的标注数据集，限制了现有模型准确识别学术角色、课程名称和机构术语的能力，突显了在该领域迫切需要针对性资源。据我们所知，目前乌尔都语领域中没有用于这一目的的数据集。为了实现这一目标，本研究作出三项关键贡献。首先，我们创建了一个针对教育领域的手动标注数据集，命名为EDU-NER-2025，包含13个与教育领域密切相关的重要实体。其次，我们详细描述了我们的标注过程和指南，并讨论了标注EDU-NER-2025数据集时遇到的挑战。第三，我们解决了并分析了诸如形态复杂性和歧义性等关键语言挑战，这些挑战在正式乌尔都语文本中普遍存在。 

---
# Evaluating Evaluation Metrics -- The Mirage of Hallucination Detection 

**Title (ZH)**: 评估评估指标——幻觉检测的幻象 

**Authors**: Atharva Kulkarni, Yuan Zhang, Joel Ruben Antony Moniz, Xiou Ge, Bo-Hsiang Tseng, Dhivya Piraviperumal, Swabha Swayamdipta, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18114)  

**Abstract**: Hallucinations pose a significant obstacle to the reliability and widespread adoption of language models, yet their accurate measurement remains a persistent challenge. While many task- and domain-specific metrics have been proposed to assess faithfulness and factuality concerns, the robustness and generalization of these metrics are still untested. In this paper, we conduct a large-scale empirical evaluation of 6 diverse sets of hallucination detection metrics across 4 datasets, 37 language models from 5 families, and 5 decoding methods. Our extensive investigation reveals concerning gaps in current hallucination evaluation: metrics often fail to align with human judgments, take an overtly myopic view of the problem, and show inconsistent gains with parameter scaling. Encouragingly, LLM-based evaluation, particularly with GPT-4, yields the best overall results, and mode-seeking decoding methods seem to reduce hallucinations, especially in knowledge-grounded settings. These findings underscore the need for more robust metrics to understand and quantify hallucinations, and better strategies to mitigate them. 

**Abstract (ZH)**: 语言模型中的幻觉是其可靠性和广泛应用的重要障碍，然而对其准确测量依然是一个持续的挑战。尽管已提出了许多特定任务和领域的评估指标来评估忠实性和事实性问题，但这些指标的鲁棒性和泛化能力尚未得到验证。本文通过在4个数据集、5大家族的37个语言模型和5种解码方法上进行大规模实证评估，6种不同的幻觉检测指标集，揭示了当前幻觉评估中存在的关键差距：指标往往无法与人类判断一致，对问题采取过于近视的观点，并且在参数缩放时显示出不一致的改进。令人鼓舞的是，基于LLM的评估，特别是使用GPT-4，获得了最好的整体结果，模式寻找解码方法似乎能够减少幻觉，尤其是在知识导向的环境中。这些发现强调了需要更多鲁棒的指标来理解和量化幻觉，并寻求更好的减少它们的策略。 

---
# Learning from Less: SINDy Surrogates in RL 

**Title (ZH)**: 从少量数据中学习：SINDy代理在强化学习中的应用 

**Authors**: Aniket Dixit, Muhammad Ibrahim Khan, Faizan Ahmed, James Brusey  

**Link**: [PDF](https://arxiv.org/pdf/2504.18113)  

**Abstract**: This paper introduces an approach for developing surrogate environments in reinforcement learning (RL) using the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm. We demonstrate the effectiveness of our approach through extensive experiments in OpenAI Gym environments, particularly Mountain Car and Lunar Lander. Our results show that SINDy-based surrogate models can accurately capture the underlying dynamics of these environments while reducing computational costs by 20-35%. With only 75 interactions for Mountain Car and 1000 for Lunar Lander, we achieve state-wise correlations exceeding 0.997, with mean squared errors as low as 3.11e-06 for Mountain Car velocity and 1.42e-06 for LunarLander position. RL agents trained in these surrogate environments require fewer total steps (65,075 vs. 100,000 for Mountain Car and 801,000 vs. 1,000,000 for Lunar Lander) while achieving comparable performance to those trained in the original environments, exhibiting similar convergence patterns and final performance metrics. This work contributes to the field of model-based RL by providing an efficient method for generating accurate, interpretable surrogate environments. 

**Abstract (ZH)**: 基于Sparse Identification of Nonlinear Dynamics的强化学习代理环境 surrogate 环境开发方法及其有效性探索 

---
# Application and Optimization of Large Models Based on Prompt Tuning for Fact-Check-Worthiness Estimation 

**Title (ZH)**: 基于提示调优的大模型应用与优化以评估事实核查值 

**Authors**: Yinglong Yu, Hao Shen, Zhengyi Lyu, Qi He  

**Link**: [PDF](https://arxiv.org/pdf/2504.18104)  

**Abstract**: In response to the growing problem of misinformation in the context of globalization and informatization, this paper proposes a classification method for fact-check-worthiness estimation based on prompt tuning. We construct a model for fact-check-worthiness estimation at the methodological level using prompt tuning. By applying designed prompt templates to large language models, we establish in-context learning and leverage prompt tuning technology to improve the accuracy of determining whether claims have fact-check-worthiness, particularly when dealing with limited or unlabeled data. Through extensive experiments on public datasets, we demonstrate that the proposed method surpasses or matches multiple baseline methods in the classification task of fact-check-worthiness estimation assessment, including classical pre-trained models such as BERT, as well as recent popular large models like GPT-3.5 and GPT-4. Experiments show that the prompt tuning-based method proposed in this study exhibits certain advantages in evaluation metrics such as F1 score and accuracy, thereby effectively validating its effectiveness and advancement in the task of fact-check-worthiness estimation. 

**Abstract (ZH)**: 针对全球化和信息化背景下信息误导问题的日益严重，本文提出了一种基于提示调优的事实核验价值估计分类方法。通过使用提示调优技术构建方法论层面的事实核验价值估计模型，利用设计好的提示模板对大规模语言模型进行应用，实现基于context的学习，并借助提示调优技术提高判断声明是否有事实核验价值的准确性，特别是在处理有限或未标记数据时。通过在公共数据集上进行大量实验，表明所提出的方法在事实核验价值估计分类任务中优于或匹配多项基线方法，包括经典的预训练模型BERT，以及最近流行的大型模型如GPT-3.5和GPT-4。实验结果表明，基于提示调优的方法在F1分数和准确性等评估指标上表现出一定的优势，从而有效验证了其在事实核验价值估计任务中的有效性和先进性。 

---
# Random-Set Large Language Models 

**Title (ZH)**: 随机集大型语言模型 

**Authors**: Muhammad Mubashar, Shireen Kudukkil Manchingal, Fabio Cuzzolin  

**Link**: [PDF](https://arxiv.org/pdf/2504.18085)  

**Abstract**: Large Language Models (LLMs) are known to produce very high-quality tests and responses to our queries. But how much can we trust this generated text? In this paper, we study the problem of uncertainty quantification in LLMs. We propose a novel Random-Set Large Language Model (RSLLM) approach which predicts finite random sets (belief functions) over the token space, rather than probability vectors as in classical LLMs. In order to allow so efficiently, we also present a methodology based on hierarchical clustering to extract and use a budget of "focal" subsets of tokens upon which the belief prediction is defined, rather than using all possible collections of tokens, making the method scalable yet effective. RS-LLMs encode the epistemic uncertainty induced in their generation process by the size and diversity of its training set via the size of the credal sets associated with the predicted belief functions. The proposed approach is evaluated on CoQA and OBQA datasets using Llama2-7b, Mistral-7b and Phi-2 models and is shown to outperform the standard model in both datasets in terms of correctness of answer while also showing potential in estimating the second level uncertainty in its predictions and providing the capability to detect when its hallucinating. 

**Abstract (ZH)**: 大规模语言模型（LLMs）生成的测试和响应质量非常高，但生成的文本我们能信任多少呢？本文研究了LLMs中的不确定量化问题。我们提出了一种新颖的随机集大规模语言模型（RSLLM）方法，该方法预测词汇表上的有限随机集（信念函数），而不是经典的概率向量。为了实现这一点，我们还提出了一种基于层次聚类的方法，以提取并利用一组“专注”的词汇子集，这些子集上定义了信念预测，而不是使用所有可能的词汇组合，从而使方法既可扩展又有效。RS-LLMs通过预测的信念函数关联的信念集的大小，编码其生成过程中由训练集大小和多样性引起的认知不确定性。所提出的方法在CoQA和OBQA数据集上使用Llama2-7b、Mistral-7b和Phi-2模型进行评估，并在答案正确性方面优于标准模型，同时在估计预测的第二层级不确定性方面显示出潜力，并具备检测其虚构的能力。 

---
# Efficient GNN Training Through Structure-Aware Randomized Mini-Batching 

**Title (ZH)**: 结构感知随机 minibatch 训练高效 GNN 

**Authors**: Vignesh Balaji, Christos Kozyrakis, Gal Chechik, Haggai Maron  

**Link**: [PDF](https://arxiv.org/pdf/2504.18082)  

**Abstract**: Graph Neural Networks (GNNs) enable learning on realworld graphs and mini-batch training has emerged as the de facto standard for training GNNs because it can scale to very large graphs and improve convergence. Current mini-batch construction policies largely ignore efficiency considerations of GNN training. Specifically, existing mini-batching techniques employ randomization schemes to improve accuracy and convergence. However, these randomization schemes are often agnostic to the structural properties of the graph (for eg. community structure), resulting in highly irregular memory access patterns during GNN training that make suboptimal use of on-chip GPU caches. On the other hand, while deterministic mini-batching based solely on graph structure delivers fast runtime performance, the lack of randomness compromises both the final model accuracy and training convergence speed. In this paper, we present Community-structure-aware Randomized Mini-batching (COMM-RAND), a novel methodology that bridges the gap between the above extremes. COMM-RAND allows practitioners to explore the space between pure randomness and pure graph structural awareness during mini-batch construction, leading to significantly more efficient GNN training with similar accuracy. We evaluated COMM-RAND across four popular graph learning benchmarks. COMM-RAND cuts down GNN training time by up to 2.76x (1.8x on average) while achieving an accuracy that is within 1.79% points (0.42% on average) compared to popular random mini-batching approaches. 

**Abstract (ZH)**: 面向社区结构的随机化mini-batch训练方法（COMM-RAND） 

---
# Stabilizing Reasoning in Medical LLMs with Continued Pretraining and Reasoning Preference Optimization 

**Title (ZH)**: 持续预训练与推理偏好优化在医疗LLM中稳定推理 

**Authors**: Wataru Kawakami, Keita Suzuki, Junichiro Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2504.18080)  

**Abstract**: Large Language Models (LLMs) show potential in medicine, yet clinical adoption is hindered by concerns over factual accuracy, language-specific limitations (e.g., Japanese), and critically, their reliability when required to generate reasoning explanations -- a prerequisite for trust. This paper introduces Preferred-MedLLM-Qwen-72B, a 72B-parameter model optimized for the Japanese medical domain to achieve both high accuracy and stable reasoning. We employ a two-stage fine-tuning process on the Qwen2.5-72B base model: first, Continued Pretraining (CPT) on a comprehensive Japanese medical corpus instills deep domain knowledge. Second, Reasoning Preference Optimization (RPO), a preference-based method, enhances the generation of reliable reasoning pathways while preserving high answer accuracy. Evaluations on the Japanese Medical Licensing Exam benchmark (IgakuQA) show Preferred-MedLLM-Qwen-72B achieves state-of-the-art performance (0.868 accuracy), surpassing strong proprietary models like GPT-4o (0.866). Crucially, unlike baseline or CPT-only models which exhibit significant accuracy degradation (up to 11.5\% and 3.8\% respectively on IgakuQA) when prompted for explanations, our model maintains its high accuracy (0.868) under such conditions. This highlights RPO's effectiveness in stabilizing reasoning generation. This work underscores the importance of optimizing for reliable explanations alongside accuracy. We release the Preferred-MedLLM-Qwen-72B model weights to foster research into trustworthy LLMs for specialized, high-stakes applications. 

**Abstract (ZH)**: Large Language Models (LLMs)在医学中的潜在应用及其临床采用受到事实准确性、语言特定限制（例如日语）以及在生成推理解释时的可靠性担忧的阻碍。本文介绍了Preferred-MedLLM-Qwen-72B，这是一种针对日本医学领域的720亿参数模型，旨在实现高准确性和稳定的推理。我们采用两阶段微调过程对Qwen2.5-72B基础模型进行优化：首先，通过全面的日语医学语料库进行持续预训练，灌输深厚的领域知识；其次，采用基于偏好方法的推理偏好优化（RPO），增强了可靠推理路径的生成能力，同时保持高答案准确性。在日本医学执照考试基准测试（IgakuQA）上的评估表明，Preferred-MedLLM-Qwen-72B取得最先进的性能（准确率为0.868），超越了强大的专有模型如GPT-4o（准确率为0.866）。 crucially，与基准模型或仅CPT模型相比，当被要求生成解释时，这些模型在IgakuQA上的准确率分别下降了11.5%和3.8%，而我们的模型在这些情况下仍能保持其高准确率（0.868）。这表明RPO在稳定生成推理方面非常有效。本文强调，在追求准确性的基础上优化可靠解释的重要性。我们发布了Preferred-MedLLM-Qwen-72B模型权重，以促进针对专业和高风险应用的可信大语言模型的研究。 

---
# Privacy-Preserving Personalized Federated Learning for Distributed Photovoltaic Disaggregation under Statistical Heterogeneity 

**Title (ZH)**: 统计异构性下具有隐私保护的个性化联邦学习在分布式光伏负荷辨识中的应用 

**Authors**: Xiaolu Chen, Chenghao Huang, Yanru Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18078)  

**Abstract**: The rapid expansion of distributed photovoltaic (PV) installations worldwide, many being behind-the-meter systems, has significantly challenged energy management and grid operations, as unobservable PV generation further complicates the supply-demand balance. Therefore, estimating this generation from net load, known as PV disaggregation, is critical. Given privacy concerns and the need for large training datasets, federated learning becomes a promising approach, but statistical heterogeneity, arising from geographical and behavioral variations among prosumers, poses new challenges to PV disaggregation. To overcome these challenges, a privacy-preserving distributed PV disaggregation framework is proposed using Personalized Federated Learning (PFL). The proposed method employs a two-level framework that combines local and global modeling. At the local level, a transformer-based PV disaggregation model is designed to generate solar irradiance embeddings for representing local PV conditions. A novel adaptive local aggregation mechanism is adopted to mitigate the impact of statistical heterogeneity on the local model, extracting a portion of global information that benefits the local model. At the global level, a central server aggregates information uploaded from multiple data centers, preserving privacy while enabling cross-center knowledge sharing. Experiments on real-world data demonstrate the effectiveness of this proposed framework, showing improved accuracy and robustness compared to benchmark methods. 

**Abstract (ZH)**: 全球分布式光伏安装的快速扩张，尤其是背对计量系统的安装，显著挑战了能源管理和电网运营，不可观测的光伏发电进一步 complicates 供需平衡。因此，从净负荷估算这一发电过程，即光伏解聚集，至关重要。鉴于隐私问题和需要大规模训练数据集，联邦学习成为一个有希望的方法，但地理和行为差异导致的数据统计异质性给光伏解聚集带来了新挑战。为了克服这些挑战，提出了一种基于个性化联邦学习（PFL）的隐私保护分布式光伏解聚集框架。该方法采用两层框架结合局部和全局建模。在局部层面，设计了一个基于变压器的光伏解聚集模型来生成太阳能辐照度嵌入表示局部光伏条件。采用了新颖的自适应局部聚合机制来减轻统计异质性对局部模型的影响，提取有助于局部模型的全局信息部分。在全局层面，中央服务器从多个数据中心收集信息，同时保护隐私并促进跨中心的知识共享。实验结果表明，与基准方法相比，该提出的框架在准确性和鲁棒性方面取得了改进。 

---
# PropRAG: Guiding Retrieval with Beam Search over Proposition Paths 

**Title (ZH)**: PropRAG: 基于命题路径的束搜索引导检索 

**Authors**: Jingjin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18070)  

**Abstract**: Retrieval Augmented Generation (RAG) has become the standard non-parametric approach for equipping Large Language Models (LLMs) with up-to-date knowledge and mitigating catastrophic forgetting common in continual learning. However, standard RAG, relying on independent passage retrieval, fails to capture the interconnected nature of human memory crucial for complex reasoning (associativity) and contextual understanding (sense-making). While structured RAG methods like HippoRAG utilize knowledge graphs (KGs) built from triples, the inherent context loss limits fidelity. We introduce PropRAG, a framework leveraging contextually rich propositions and a novel beam search algorithm over proposition paths to explicitly discover multi-step reasoning chains. Crucially, PropRAG's online retrieval process operates entirely without invoking generative LLMs, relying instead on efficient graph traversal and pre-computed embeddings. This avoids online LLM inference costs and potential inconsistencies during evidence gathering. LLMs are used effectively offline for high-quality proposition extraction and post-retrieval for answer generation. PropRAG achieves state-of-the-art zero-shot Recall@5 results on PopQA (55.3%), 2Wiki (93.7%), HotpotQA (97.0%), and MuSiQue (77.3%), alongside top F1 scores (e.g., 52.4% on MuSiQue). By improving evidence retrieval through richer representation and explicit, LLM-free online path finding, PropRAG advances non-parametric continual learning. 

**Abstract (ZH)**: PropRAG：基于丰富命题的在线检索增强生成 

---
# S3MOT: Monocular 3D Object Tracking with Selective State Space Model 

**Title (ZH)**: S3MOT：基于选择性状态空间模型的单目三维目标跟踪 

**Authors**: Zhuohao Yan, Shaoquan Feng, Xingxing Li, Yuxuan Zhou, Chunxi Xia, Shengyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.18068)  

**Abstract**: Accurate and reliable multi-object tracking (MOT) in 3D space is essential for advancing robotics and computer vision applications. However, it remains a significant challenge in monocular setups due to the difficulty of mining 3D spatiotemporal associations from 2D video streams. In this work, we present three innovative techniques to enhance the fusion and exploitation of heterogeneous cues for monocular 3D MOT: (1) we introduce the Hungarian State Space Model (HSSM), a novel data association mechanism that compresses contextual tracking cues across multiple paths, enabling efficient and comprehensive assignment decisions with linear complexity. HSSM features a global receptive field and dynamic weights, in contrast to traditional linear assignment algorithms that rely on hand-crafted association costs. (2) We propose Fully Convolutional One-stage Embedding (FCOE), which eliminates ROI pooling by directly using dense feature maps for contrastive learning, thus improving object re-identification accuracy under challenging conditions such as varying viewpoints and lighting. (3) We enhance 6-DoF pose estimation through VeloSSM, an encoder-decoder architecture that models temporal dependencies in velocity to capture motion dynamics, overcoming the limitations of frame-based 3D inference. Experiments on the KITTI public test benchmark demonstrate the effectiveness of our method, achieving a new state-of-the-art performance of 76.86~HOTA at 31~FPS. Our approach outperforms the previous best by significant margins of +2.63~HOTA and +3.62~AssA, showcasing its robustness and efficiency for monocular 3D MOT tasks. The code and models are available at this https URL. 

**Abstract (ZH)**: 准确可靠的单目三维多对象跟踪（MOT）在机器人技术和计算机视觉应用中至关重要。然而，在单目设置中由于难以从二维视频流中挖掘三维时空关联，使其成为一个重大挑战。在本工作中，我们提出三种创新技术以增强单目三维多对象跟踪中异构线索的融合与利用：（1）我们引入匈牙利状态空间模型（HSSM），这是一种新的数据关联机制，能够跨多个路径压缩上下文跟踪线索，实现线性复杂度下的高效且全面的分配决策。HSSM 具有全局感受野和动态权重，而传统的线性分配算法依赖于手工设计的关联成本。（2）我们提出全卷积一阶段嵌入（FCOE），其通过直接使用密集特征图进行对比学习，消除了ROI池化，从而在多视角和光照变化等具有挑战性的条件下提高物体再识别的准确性。（3）我们通过速度编码器-解码器架构VeloSSM增强六自由度姿态估计，该架构建模了时间相关的速度依赖性以捕捉运动动态，克服了基于帧的三维推断的局限性。我们的方法在KITTI公开测试基准上的实验表明其有效性，实现了76.86的HOTA，在31 FPS下达到新最佳性能。与之前最佳方法相比，我们的方法在HOTA和AssA上分别取得了2.63和3.62的显著提升，展示了其在单目三维多对象跟踪任务中的鲁棒性和效率。代码和模型可在以下链接获取。 

---
# LLM-Guided Open RAN: Empowering Hierarchical RAN Intelligent Control 

**Title (ZH)**: LLM 引导的开放RAN：赋能分层RAN智能控制 

**Authors**: Lingyan Bao, Sinwoong Yun, Jemin Lee, Tony Q.S. Quek  

**Link**: [PDF](https://arxiv.org/pdf/2504.18062)  

**Abstract**: Recent advancements in large language models (LLMs) have led to a significant interest in deploying LLMempowered algorithms for wireless communication networks. Meanwhile, open radio access network (O-RAN) techniques offer unprecedented flexibility, with the non-real-time (non-RT) radio access network (RAN) intelligent controller (RIC) (non-RT RIC) and near-real-time (near-RT) RIC (near-RT RIC) components enabling intelligent resource management across different time scales. In this paper, we propose the LLM empowered hierarchical RIC (LLM-hRIC) framework to improve the collaboration between RICs. This framework integrates LLMs with reinforcement learning (RL) for efficient network resource management. In this framework, LLMs-empowered non-RT RICs provide strategic guidance and high-level policies based on environmental context. Concurrently, RL-empowered near-RT RICs perform low-latency tasks based on strategic guidance and local near-RT observation. We evaluate the LLM-hRIC framework in an integrated access and backhaul (IAB) network setting. Simulation results demonstrate that the proposed framework achieves superior performance. Finally, we discuss the key future challenges in applying LLMs to O-RAN. 

**Abstract (ZH)**: 大语言模型赋能的无线通信网络层次化智能控制器框架 

---
# Exploring Personality-Aware Interactions in Salesperson Dialogue Agents 

**Title (ZH)**: 探索销售代理对话中的人格意识交互 

**Authors**: Sijia Cheng, Wen-Yu Chang, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.18058)  

**Abstract**: The integration of dialogue agents into the sales domain requires a deep understanding of how these systems interact with users possessing diverse personas. This study explores the influence of user personas, defined using the Myers-Briggs Type Indicator (MBTI), on the interaction quality and performance of sales-oriented dialogue agents. Through large-scale testing and analysis, we assess the pre-trained agent's effectiveness, adaptability, and personalization capabilities across a wide range of MBTI-defined user types. Our findings reveal significant patterns in interaction dynamics, task completion rates, and dialogue naturalness, underscoring the future potential for dialogue agents to refine their strategies to better align with varying personality traits. This work not only provides actionable insights for building more adaptive and user-centric conversational systems in the sales domain but also contributes broadly to the field by releasing persona-defined user simulators. These simulators, unconstrained by domain, offer valuable tools for future research and demonstrate the potential for scaling personalized dialogue systems across diverse applications. 

**Abstract (ZH)**: 将对话代理整合到销售领域需要深刻理解这些系统与具有多样化人设的用户交互的方式。本研究探讨了使用迈尔斯-布里格斯类型指标（MBTI）定义的用户人设对销售导向对话代理的交互质量和性能的影响。通过大规模测试和分析，我们评估了预训练代理在各种MBTI定义的用户类型中的有效性、适应性和个性化能力。研究发现揭示了交互动力学、任务完成率和对话自然度的显著模式，强调了对话代理在未来如何根据不同个性特征优化其策略的潜力。本研究不仅为构建更加适应用户需求的销售领域对话系统提供了可操作的见解，还通过发布人设定义的用户模拟器为该领域未来的研究提供了广泛贡献，这些模拟器不受领域限制，为未来研究提供了有价值的工具，并展示了跨多种应用场景规模化个性化对话系统的能力。 

---
# Opportunistic Collaborative Planning with Large Vision Model Guided Control and Joint Query-Service Optimization 

**Title (ZH)**: 基于大型视觉模型引导控制及联合查询-服务优化的机会性协同规划 

**Authors**: Jiayi Chen, Shuai Wang, Guoliang Li, Wei Xu, Guangxu Zhu, Derrick Wing Kwan Ng, Chengzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18057)  

**Abstract**: Navigating autonomous vehicles in open scenarios is a challenge due to the difficulties in handling unseen objects. Existing solutions either rely on small models that struggle with generalization or large models that are resource-intensive. While collaboration between the two offers a promising solution, the key challenge is deciding when and how to engage the large model. To address this issue, this paper proposes opportunistic collaborative planning (OCP), which seamlessly integrates efficient local models with powerful cloud models through two key innovations. First, we propose large vision model guided model predictive control (LVM-MPC), which leverages the cloud for LVM perception and decision making. The cloud output serves as a global guidance for a local MPC, thereby forming a closed-loop perception-to-control system. Second, to determine the best timing for large model query and service, we propose collaboration timing optimization (CTO), including object detection confidence thresholding (ODCT) and cloud forward simulation (CFS), to decide when to seek cloud assistance and when to offer cloud service. Extensive experiments show that the proposed OCP outperforms existing methods in terms of both navigation time and success rate. 

**Abstract (ZH)**: 基于机会的合作规划：在开放场景中导航自主车辆 

---
# Validating Network Protocol Parsers with Traceable RFC Document Interpretation 

**Title (ZH)**: 用可追溯的RFC文档解析验证网络协议解析器 

**Authors**: Mingwei Zheng, Danning Xie, Qingkai Shi, Chengpeng Wang, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18050)  

**Abstract**: Validating the correctness of network protocol implementations is highly challenging due to the oracle and traceability problems. The former determines when a protocol implementation can be considered buggy, especially when the bugs do not cause any observable symptoms. The latter allows developers to understand how an implementation violates the protocol specification, thereby facilitating bug fixes. Unlike existing works that rarely take both problems into account, this work considers both and provides an effective solution using recent advances in large language models (LLMs). Our key observation is that network protocols are often released with structured specification documents, a.k.a. RFC documents, which can be systematically translated to formal protocol message specifications via LLMs. Such specifications, which may contain errors due to the hallucination of LLMs, are used as a quasi-oracle to validate protocol parsers, while the validation results in return gradually refine the oracle. Since the oracle is derived from the document, any bugs we find in a protocol implementation can be traced back to the document, thus addressing the traceability problem. We have extensively evaluated our approach using nine network protocols and their implementations written in C, Python, and Go. The results show that our approach outperforms the state-of-the-art and has detected 69 bugs, with 36 confirmed. The project also demonstrates the potential for fully automating software validation based on natural language specifications, a process previously considered predominantly manual due to the need to understand specification documents and derive expected outputs for test inputs. 

**Abstract (ZH)**: 验证网络协议实现的正确性由于存在oracle问题和可追溯性问题而极具挑战性。oracle问题决定了何时可以认为协议实现存在bug，尤其是在这些bug不引起任何可观察症状的情况下。可追溯性问题使开发者能够理解实现如何违反协议规格，从而便于修复bug。与现有工作大多未能同时考虑这两个问题不同，本工作同时考虑了这两个问题，并利用大型语言模型（LLMs）的最新进展提供了一个有效的解决方案。我们关键的观察是，网络协议通常伴随着结构化的规格说明书，即RFC文档，这些文档可以通过LLMs系统地转换为形式化的协议消息规格。这些规格可能由于LLMs的幻觉而包含错误，但它们被用作准oracle来验证协议解析器，而验证结果反过来逐步精化了oracle。由于oracle源自文档，我们发现的协议实现中的任何bug都可以追溯到文档，从而解决了可追溯性问题。我们使用九个网络协议及其用C、Python和Go编写的实现进行了广泛评估。结果显示，我们的方法优于最先进的方法，并检测到了69个bug，其中有36个得到了确认。该项目还展示了基于自然语言规格完全自动化软件验证的潜力，而在先前，这一过程因需要理解规格说明书并为测试输入推导预期输出而被认为主要是手工操作。 

---
# A BERT-Style Self-Supervised Learning CNN for Disease Identification from Retinal Images 

**Title (ZH)**: 基于BERT风格的自监督学习CNN在视网膜图像疾病识别中的应用 

**Authors**: Xin Li, Wenhui Zhu, Peijie Qiu, Oana M. Dumitrascu, Amal Youssef, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18049)  

**Abstract**: In the field of medical imaging, the advent of deep learning, especially the application of convolutional neural networks (CNNs) has revolutionized the analysis and interpretation of medical images. Nevertheless, deep learning methods usually rely on large amounts of labeled data. In medical imaging research, the acquisition of high-quality labels is both expensive and difficult. The introduction of Vision Transformers (ViT) and self-supervised learning provides a pre-training strategy that utilizes abundant unlabeled data, effectively alleviating the label acquisition challenge while broadening the breadth of data utilization. However, ViT's high computational density and substantial demand for computing power, coupled with the lack of localization characteristics of its operations on image patches, limit its efficiency and applicability in many application scenarios. In this study, we employ nn-MobileNet, a lightweight CNN framework, to implement a BERT-style self-supervised learning approach. We pre-train the network on the unlabeled retinal fundus images from the UK Biobank to improve downstream application performance. We validate the results of the pre-trained model on Alzheimer's disease (AD), Parkinson's disease (PD), and various retinal diseases identification. The results show that our approach can significantly improve performance in the downstream tasks. In summary, this study combines the benefits of CNNs with the capabilities of advanced self-supervised learning in handling large-scale unlabeled data, demonstrating the potential of CNNs in the presence of label scarcity. 

**Abstract (ZH)**: 在医学影像领域，深度学习的出现，尤其是卷积神经网络（CNNs）的应用， telah革命性地改变了医学影像的分析和解释。然而，深度学习方法通常依赖大量标记数据。在医学影像研究中，高质量标签的获取既昂贵又困难。通过引入视觉变换器（ViT）和自监督学习，提供了一种利用丰富未标记数据的预训练策略，有效缓解了标签获取的挑战，同时也扩展了数据利用的广度。然而，ViT的高计算密度和对计算能力的大量需求，以及其对图像块操作缺乏定位特性，限制了其在许多应用场景中的效率和适用性。在本研究中，我们采用轻量级CNN框架nn-MobileNet，实现了一种类似BERT的自监督学习方法。我们在英国生物银行的未标记视网膜底片图像上进行预训练，以提高下游应用性能。我们通过验证阿尔茨海默病（AD）、帕金森病（PD）和各种视网膜疾病识别的结果，展示了预训练模型的表现。结果显示，我们的方法能够在下游任务中显著提高性能。总之，本研究结合了CNN的优势和先进自监督学习处理大规模未标记数据的能力，展示了在标签稀缺情况下CNN的潜力。 

---
# DMS-Net:Dual-Modal Multi-Scale Siamese Network for Binocular Fundus Image Classification 

**Title (ZH)**: DMS-Net：双模态多尺度孪生网络用于双眼底图像分类 

**Authors**: Guohao Huo, Zibo Lin, Zitong Wang, Ruiting Dai, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18046)  

**Abstract**: Ophthalmic diseases pose a significant global health challenge, yet traditional diagnosis methods and existing single-eye deep learning approaches often fail to account for binocular pathological correlations. To address this, we propose DMS-Net, a dual-modal multi-scale Siamese network for binocular fundus image classification. Our framework leverages weight-shared Siamese ResNet-152 backbones to extract deep semantic features from paired fundus images. To tackle challenges such as lesion boundary ambiguity and scattered pathological distributions, we introduce a Multi-Scale Context-Aware Module (MSCAM) that integrates adaptive pooling and attention mechanisms for multi-resolution feature aggregation. Additionally, a Dual-Modal Feature Fusion (DMFF) module enhances cross-modal interaction through spatial-semantic recalibration and bidirectional attention, effectively combining global context and local edge features. Evaluated on the ODIR-5K dataset, DMS-Net achieves state-of-the-art performance with 80.5% accuracy, 86.1% recall, and 83.8% Cohen's kappa, demonstrating superior capability in detecting symmetric pathologies and advancing clinical decision-making for ocular diseases. 

**Abstract (ZH)**: 双眼视网膜图像分类的双模态多尺度Siamese网络:DMS-Net 

---
# AI Ethics and Social Norms: Exploring ChatGPT's Capabilities From What to How 

**Title (ZH)**: AI伦理与社会规范：探究ChatGPT能力的从“是什么”到“怎么做” 

**Authors**: Omid Veisi, Sasan Bahrami, Roman Englert, Claudia Müller  

**Link**: [PDF](https://arxiv.org/pdf/2504.18044)  

**Abstract**: Using LLMs in healthcare, Computer-Supported Cooperative Work, and Social Computing requires the examination of ethical and social norms to ensure safe incorporation into human life. We conducted a mixed-method study, including an online survey with 111 participants and an interview study with 38 experts, to investigate the AI ethics and social norms in ChatGPT as everyday life tools. This study aims to evaluate whether ChatGPT in an empirical context operates following ethics and social norms, which is critical for understanding actions in industrial and academic research and achieving machine ethics. The findings of this study provide initial insights into six important aspects of AI ethics, including bias, trustworthiness, security, toxicology, social norms, and ethical data. Significant obstacles related to transparency and bias in unsupervised data collection methods are identified as ChatGPT's ethical concerns. 

**Abstract (ZH)**: 在医疗保健、计算机支持的协作工作和社交计算中使用大语言模型需要审视伦理和社会规范以确保安全地融入人类生活。我们进行了一项混合方法研究，包括一项有111名参与者在线调查和一项有38名专家的访谈研究，以调查ChatGPT作为日常生活工具中的AI伦理和社会规范。本研究旨在评估在实际情境中ChatGPT是否遵循伦理和社会规范，这对于理解和实现机器伦理至关重要。本研究的发现提供了关于AI伦理的六个重要方面的初步见解，包括偏见、可信度、安全性、毒性、社会规范和伦理数据。识别出与无监督数据收集方法相关的透明度和偏见的重大障碍是ChatGPT的伦理关切。 

---
# RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models 

**Title (ZH)**: RAG 大型语言模型并不更安全：检索增强生成的安全性分析 

**Authors**: Bang An, Shiyue Zhang, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2504.18041)  

**Abstract**: Efforts to ensure the safety of large language models (LLMs) include safety fine-tuning, evaluation, and red teaming. However, despite the widespread use of the Retrieval-Augmented Generation (RAG) framework, AI safety work focuses on standard LLMs, which means we know little about how RAG use cases change a model's safety profile. We conduct a detailed comparative analysis of RAG and non-RAG frameworks with eleven LLMs. We find that RAG can make models less safe and change their safety profile. We explore the causes of this change and find that even combinations of safe models with safe documents can cause unsafe generations. In addition, we evaluate some existing red teaming methods for RAG settings and show that they are less effective than when used for non-RAG settings. Our work highlights the need for safety research and red-teaming methods specifically tailored for RAG LLMs. 

**Abstract (ZH)**: 确保大型语言模型安全的努力包括安全微调、评估和红队测试。然而，尽管Retrieval-Augmented Generation (RAG)框架被广泛使用，AI安全工作主要集中在标准大型语言模型上，这意味着我们对RAG用例如何改变模型的安全特性知之甚少。我们对RAG和非RAG框架进行了详细比较分析，涉及11个大型语言模型。我们发现RAG可以使模型更不安全并改变其安全特性。我们探讨了这种变化的原因，并发现即使使用安全的模型和安全的文档也可能导致生成不安全的内容。此外，我们评估了一些现有的RAG环境中的红队测试方法，发现它们在RAG环境中的有效性低于非RAG环境。我们的研究强调了专门针对RAG大型语言模型的安全研究和红队测试方法的必要性。 

---
# A Large Vision-Language Model based Environment Perception System for Visually Impaired People 

**Title (ZH)**: 基于大型视觉-语言模型的视觉障碍人士环境感知系统 

**Authors**: Zezhou Chen, Zhaoxiang Liu, Kai Wang, Kohou Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2504.18027)  

**Abstract**: It is a challenging task for visually impaired people to perceive their surrounding environment due to the complexity of the natural scenes. Their personal and social activities are thus highly limited. This paper introduces a Large Vision-Language Model(LVLM) based environment perception system which helps them to better understand the surrounding environment, by capturing the current scene they face with a wearable device, and then letting them retrieve the analysis results through the device. The visually impaired people could acquire a global description of the scene by long pressing the screen to activate the LVLM output, retrieve the categories of the objects in the scene resulting from a segmentation model by tapping or swiping the screen, and get a detailed description of the objects they are interested in by double-tapping the screen. To help visually impaired people more accurately perceive the world, this paper proposes incorporating the segmentation result of the RGB image as external knowledge into the input of LVLM to reduce the LVLM's hallucination. Technical experiments on POPE, MME and LLaVA-QA90 show that the system could provide a more accurate description of the scene compared to Qwen-VL-Chat, exploratory experiments show that the system helps visually impaired people to perceive the surrounding environment effectively. 

**Abstract (ZH)**: 视觉受损人士基于大型视觉-语言模型的环境感知系统及其应用研究 

---
# Addressing Concept Mislabeling in Concept Bottleneck Models Through Preference Optimization 

**Title (ZH)**: 通过偏好优化解决概念瓶颈模型中的概念误标问题 

**Authors**: Emiliano Penaloza, Tianyue H. Zhan, Laurent Charlin, Mateo Espinosa Zarlenga  

**Link**: [PDF](https://arxiv.org/pdf/2504.18026)  

**Abstract**: Concept Bottleneck Models (CBMs) propose to enhance the trustworthiness of AI systems by constraining their decisions on a set of human understandable concepts. However, CBMs typically assume that datasets contains accurate concept labels an assumption often violated in practice, which we show can significantly degrade performance (by 25% in some cases). To address this, we introduce the Concept Preference Optimization (CPO) objective, a new loss function based on Direct Preference Optimization, which effectively mitigates the negative impact of concept mislabeling on CBM performance. We provide an analysis on some key properties of the CPO objective showing it directly optimizes for the concept's posterior distribution, and contrast it against Binary Cross Entropy (BCE) where we show CPO is inherently less sensitive to concept noise. We empirically confirm our analysis finding that CPO consistently outperforms BCE in three real world datasets with and without added label noise. 

**Abstract (ZH)**: 概念偏好优化（CPO）目标：提升概念瓶颈模型（CBM）的稳健性以应对概念标记错误问题 

---
# Memory Reviving, Continuing Learning and Beyond: Evaluation of Pre-trained Encoders and Decoders for Multimodal Machine Translation 

**Title (ZH)**: 记忆重现、持续学习及更进一步：预训练编码器和解码器在多模态机器翻译中的评估 

**Authors**: Zhuang Yu, Shiliang Sun, Jing Zhao, Tengfei Song, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18012)  

**Abstract**: Multimodal Machine Translation (MMT) aims to improve translation quality by leveraging auxiliary modalities such as images alongside textual input. While recent advances in large-scale pre-trained language and vision models have significantly benefited unimodal natural language processing tasks, their effectiveness and role in MMT remain underexplored. In this work, we conduct a systematic study on the impact of pre-trained encoders and decoders in multimodal translation models. Specifically, we analyze how different training strategies, from training from scratch to using pre-trained and partially frozen components, affect translation performance under a unified MMT framework. Experiments are carried out on the Multi30K and CoMMuTE dataset across English-German and English-French translation tasks. Our results reveal that pre-training plays a crucial yet asymmetrical role in multimodal settings: pre-trained decoders consistently yield more fluent and accurate outputs, while pre-trained encoders show varied effects depending on the quality of visual-text alignment. Furthermore, we provide insights into the interplay between modality fusion and pre-trained components, offering guidance for future architecture design in multimodal translation systems. 

**Abstract (ZH)**: 多模态机器翻译（MMT）旨在通过利用图像等辅助模态来提高翻译质量。尽管大规模预训练语言和视觉模型在单模态自然语言处理任务中取得了显著进展，但它们在多模态翻译中的有效性及其作用仍然有待探索。在本工作中，我们系统研究了预训练编码器和解码器在多模态翻译模型中的影响。具体而言，我们分析了从从头训练到使用预训练和部分冻结组件的不同训练策略对统一多模态翻译框架下的翻译性能的影响。我们在Multi30K和CoMMuTE数据集上的英语-德语和英语-法语翻译任务中进行了实验。我们的结果表明，预训练在多模态设置中扮演着重要但不对称的角色：预训练解码器始终产生更加流畅和准确的输出，而预训练编码器的效果则取决于视觉-文本对齐的质量。此外，我们探讨了模态融合与预训练组件之间的相互作用，为未来的多模态翻译系统架构设计提供指导。 

---
# Sky-Drive: A Distributed Multi-Agent Simulation Platform for Socially-Aware and Human-AI Collaborative Future Transportation 

**Title (ZH)**: Sky-Drive：一种面向社会意识和人机协同未来交通的分布式多Agent仿真平台 

**Authors**: Zilin Huang, Zihao Sheng, Zhengyang Wan, Yansong Qu, Yuhao Luo, Boyue Wang, Pei Li, Yen-Jung Chen, Jiancong Chen, Keke Long, Jiayi Meng, Yue Leng, Sikai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.18010)  

**Abstract**: Recent advances in autonomous system simulation platforms have significantly enhanced the safe and scalable testing of driving policies. However, existing simulators do not yet fully meet the needs of future transportation research, particularly in modeling socially-aware driving agents and enabling effective human-AI collaboration. This paper introduces Sky-Drive, a novel distributed multi-agent simulation platform that addresses these limitations through four key innovations: (a) a distributed architecture for synchronized simulation across multiple terminals; (b) a multi-modal human-in-the-loop framework integrating diverse sensors to collect rich behavioral data; (c) a human-AI collaboration mechanism supporting continuous and adaptive knowledge exchange; and (d) a digital twin (DT) framework for constructing high-fidelity virtual replicas of real-world transportation environments. Sky-Drive supports diverse applications such as autonomous vehicle (AV)-vulnerable road user (VRU) interaction modeling, human-in-the-loop training, socially-aware reinforcement learning, personalized driving policy, and customized scenario generation. Future extensions will incorporate foundation models for context-aware decision support and hardware-in-the-loop (HIL) testing for real-world validation. By bridging scenario generation, data collection, algorithm training, and hardware integration, Sky-Drive has the potential to become a foundational platform for the next generation of socially-aware and human-centered autonomous transportation research. The demo video and code are available at:this https URL 

**Abstract (ZH)**: 最近在自主系统仿真平台方面的进展显著提高了驾驶策略的安全和可扩展测试。然而，现有的仿真器尚未完全满足未来交通研究的需求，特别是在建模社会感知驾驶代理和促进有效的人类-人工智能协作方面。本文介绍了Sky-Drive，这是一种新型的分布式多智能体仿真平台，通过四项关键创新解决了这些局限性：(a) 多终端同步仿真分布架构；(b) 多模态人机环框架，集成多种传感器以收集丰富的行为数据；(c) 支持持续和自适应知识交流的人机协作机制；(d) 虚拟孪生框架，用于构建高度仿真的现实世界交通环境的虚拟副本。Sky-Drive 支持多种应用，包括无人驾驶车辆（AV）与脆弱道路使用者（VRU）的交互建模、人机环训练、社会感知强化学习、个性化驾驶策略以及定制化场景生成。未来扩展将 Incorporate 基于上下文的决策支持基础模型和硬件在环（HIL）测试以进行现实世界的验证。通过链接场景生成、数据收集、算法训练和硬件集成，Sky-Drive 有望成为下一代社会感知和以人为本的自主交通研究的基础平台。演示视频和代码可在以下链接获取：this https URL。 

---
# Fuzzy-RRT for Obstacle Avoidance in a 2-DOF Semi-Autonomous Surgical Robotic Arm 

**Title (ZH)**: 基于模糊RRT的2-DOF半自主手术机器人避障算法 

**Authors**: Kaaustaaub Shankar, Wilhelm Louw, Bharadwaj Dogga, Nick Ernest, Tim Arnett, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2504.17979)  

**Abstract**: AI-driven semi-autonomous robotic surgery is essential for addressing the medical challenges of long-duration interplanetary missions, where limited crew sizes and communication delays restrict traditional surgical approaches. Current robotic surgery systems require full surgeon control, demanding extensive expertise and limiting feasibility in space. We propose a novel adaptation of the Fuzzy Rapidly-exploring Random Tree algorithm for obstacle avoidance and collaborative control in a two-degree-of-freedom robotic arm modeled on the Miniaturized Robotic-Assisted surgical system. It was found that the Fuzzy Rapidly-exploring Random Tree algorithm resulted in an 743 percent improvement to path search time and 43 percent improvement to path cost. 

**Abstract (ZH)**: 基于人工智能的半自主机器人手术对于应对长时间星际任务的医疗挑战至关重要，这些任务由于有限的乘员规模和通信延迟限制了传统手术方法。当前的机器人手术系统需要全程医生控制，这要求 extensive 的专业知识并限制了其在太空中的可行性。我们提出了一种对模糊快速扩展随机树算法的新型适应，以避免障碍和在基于 Miniaturized Robotic-Assisted 手术系统的两自由度机器人臂上实现协作控制。研究发现，模糊快速扩展随机树算法将路径搜索时间提高了 743%，路径成本降低了 43%。 

---
# Evaluating Machine Expertise: How Graduate Students Develop Frameworks for Assessing GenAI Content 

**Title (ZH)**: 评估机器专长：研究生开发评估GenAI内容框架的研究 

**Authors**: Celia Chen, Alex Leitch  

**Link**: [PDF](https://arxiv.org/pdf/2504.17964)  

**Abstract**: This paper examines how graduate students develop frameworks for evaluating machine-generated expertise in web-based interactions with large language models (LLMs). Through a qualitative study combining surveys, LLM interaction transcripts, and in-depth interviews with 14 graduate students, we identify patterns in how these emerging professionals assess and engage with AI-generated content. Our findings reveal that students construct evaluation frameworks shaped by three main factors: professional identity, verification capabilities, and system navigation experience. Rather than uniformly accepting or rejecting LLM outputs, students protect domains central to their professional identities while delegating others--with managers preserving conceptual work, designers safeguarding creative processes, and programmers maintaining control over core technical expertise. These evaluation frameworks are further influenced by students' ability to verify different types of content and their experience navigating complex systems. This research contributes to web science by highlighting emerging human-genAI interaction patterns and suggesting how platforms might better support users in developing effective frameworks for evaluating machine-generated expertise signals in AI-mediated web environments. 

**Abstract (ZH)**: 本文探讨了研究生如何构建评估基于网页的与大规模语言模型（LLM）互动中生成的专业知识的框架。通过结合调查、LLM交互转录和对14名研究生的深入访谈的定性研究，我们识别了这些新兴专业人员评估和互动于AI生成内容的模式。研究发现，学生构建的评估框架主要由专业身份、验证能力以及系统导航经验三个因素塑造。学生在保护与专业身份密切相关的领域的同时，将其他领域委托给他人——管理者保留概念性工作，设计师保护创造性过程，程序员保持对核心技术专长的控制。这些评估框架还受到学生验证不同类型内容的能力和导航复杂系统经验的影响。本文通过强调人类-GenAI互动的新兴模式，并建议平台如何更好地支持用户在AI调解的网络环境中开发有效的评估机器生成专业知识信号的框架，为网络科学领域做出了贡献。 

---
# Avoiding Leakage Poisoning: Concept Interventions Under Distribution Shifts 

**Title (ZH)**: 避免泄露污染：分布变化下的概念干预 

**Authors**: Mateo Espinosa Zarlenga, Gabriele Dominici, Pietro Barbiero, Zohreh Shams, Mateja Jamnik  

**Link**: [PDF](https://arxiv.org/pdf/2504.17921)  

**Abstract**: In this paper, we investigate how concept-based models (CMs) respond to out-of-distribution (OOD) inputs. CMs are interpretable neural architectures that first predict a set of high-level concepts (e.g., stripes, black) and then predict a task label from those concepts. In particular, we study the impact of concept interventions (i.e., operations where a human expert corrects a CM's mispredicted concepts at test time) on CMs' task predictions when inputs are OOD. Our analysis reveals a weakness in current state-of-the-art CMs, which we term leakage poisoning, that prevents them from properly improving their accuracy when intervened on for OOD inputs. To address this, we introduce MixCEM, a new CM that learns to dynamically exploit leaked information missing from its concepts only when this information is in-distribution. Our results across tasks with and without complete sets of concept annotations demonstrate that MixCEMs outperform strong baselines by significantly improving their accuracy for both in-distribution and OOD samples in the presence and absence of concept interventions. 

**Abstract (ZH)**: 本文研究了基于概念的模型（CMs）对分布外（OOD）输入的响应。CMs 是可解释的神经架构，首先预测一组高层概念（例如：条纹、黑色），然后从这些概念中预测任务标签。特别地，我们研究了在测试时人为专家修正CM错误预测的概念对CM的任务预测的影响，特别是在输入为OOD时。我们的分析揭示了当前最先进的CMs的一个弱点，我们称之为泄露毒害，这阻碍了它们在人为干预时提高针对OOD输入的准确性。为了解决这个问题，我们提出了MixCEM，这是一种新的CM，能够仅在该信息为分布内时动态地利用其概念中缺失的泄露信息。我们的跨任务实验结果表明，在有和没有完整概念注解的情况下，MixCEMs在概念干预存在和不存在的情况下，显著提高了其对分布内和OOD样本的准确性，从而在基准模型中表现出色。 

---
# Beyond Task and Motion Planning: Hierarchical Robot Planning with General-Purpose Policies 

**Title (ZH)**: 超越任务与运动规划：基于通用策略的层次化机器人规划 

**Authors**: Benned Hedegaard, Ziyi Yang, Yichen Wei, Ahmed Jaafar, Stefanie Tellex, George Konidaris, Naman Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.17901)  

**Abstract**: Task and motion planning is a well-established approach for solving long-horizon robot planning problems. However, traditional methods assume that each task-level robot action, or skill, can be reduced to kinematic motion planning. In this work, we address the challenge of planning with both kinematic skills and closed-loop motor controllers that go beyond kinematic considerations. We propose a novel method that integrates these controllers into motion planning using Composable Interaction Primitives (CIPs), enabling the use of diverse, non-composable pre-learned skills in hierarchical robot planning. Toward validating our Task and Skill Planning (TASP) approach, we describe ongoing robot experiments in real-world scenarios designed to demonstrate how CIPs can allow a mobile manipulator robot to effectively combine motion planning with general-purpose skills to accomplish complex tasks. 

**Abstract (ZH)**: 基于任务和动作规划的机器人长期规划方法已经成熟。然而，传统方法假设每个任务级机器人操作或技能都可以简化为动力学运动规划。本工作中，我们解决了同时规划动力学技能和超越动力学考虑的闭环电机控制器的挑战。我们提出了一种新的方法，通过可组合交互本原（CIPs）将这些控制器整合到运动规划中，从而使不同且不可组合的先验学习技能能够用于层次化机器人规划。为了验证我们的任务和技能规划（TASP）方法，我们描述了在真实场景中进行的机器人实验，旨在展示CIPs如何使移动 manipulator 机器人有效地将运动规划与通用技能结合以完成复杂任务。 

---
# Token Sequence Compression for Efficient Multimodal Computing 

**Title (ZH)**: 高效的多模态计算中的令牌序列压缩 

**Authors**: Yasmine Omri, Parth Shroff, Thierry Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2504.17892)  

**Abstract**: The exponential growth of Large Multimodal Models (LMMs) has driven advancements in cross-modal reasoning but at significant computational costs. In this work, we focus on visual language models. We highlight the redundancy and inefficiency in current vision encoders, and seek to construct an adaptive compression method for multimodal data. In this work, we characterize a panoply of visual token selection and merging approaches through both benchmarking and qualitative analysis. In particular, we demonstrate that simple cluster-level token aggregation outperforms prior state-of-the-art works in token selection and merging, including merging at the vision encoder level and attention-based approaches. We underline the redundancy in current vision encoders, and shed light on several puzzling trends regarding principles of visual token selection through cross-modal attention visualizations. This work is a first effort towards more effective encoding and processing of high-dimensional data, and paves the way for more scalable and sustainable multimodal systems. 

**Abstract (ZH)**: 大型多模态模型的指数增长推动了跨模态推理的进步，但伴随着巨大的计算成本。在此工作中，我们关注视觉语言模型。我们指出现有视觉编码器中的冗余和低效性，并寻求构建一种适应性压缩方法以处理多模态数据。在此工作中，我们通过基准测试和定性分析，表征了多种视觉标记选择和合并方法。特别是，我们证明了简单的聚类级别标记聚合在标记选择和合并方面优于之前的最佳方法，包括视编码器级别的合并和基于注意力的方法。我们指出现有视觉编码器中的冗余问题，并通过跨模态注意可视化揭示了几种关于视觉标记选择原理的令人困惑的趋势。此工作是更有效地编码和处理高维数据的首次尝试，并为更可扩展和可持续的多模态系统铺平了道路。 

---
# Crypto-ncRNA: Non-coding RNA (ncRNA) Based Encryption Algorithm 

**Title (ZH)**: Crypto-ncRNA：非编码RNA（ncRNA）基于的加密算法 

**Authors**: Xu Wang, Yiquan Wang, Tin-yeh Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17878)  

**Abstract**: In the looming post-quantum era, traditional cryptographic systems are increasingly vulnerable to quantum computing attacks that can compromise their mathematical foundations. To address this critical challenge, we propose crypto-ncRNA-a bio-convergent cryptographic framework that leverages the dynamic folding properties of non-coding RNA (ncRNA) to generate high-entropy, quantum-resistant keys and produce unpredictable ciphertexts. The framework employs a novel, multi-stage process: encoding plaintext into RNA sequences, predicting and manipulating RNA secondary structures using advanced algorithms, and deriving cryptographic keys through the intrinsic physical unclonability of RNA molecules. Experimental evaluations indicate that, although crypto-ncRNA's encryption speed is marginally lower than that of AES, it significantly outperforms RSA in terms of efficiency and scalability while achieving a 100% pass rate on the NIST SP 800-22 randomness tests. These results demonstrate that crypto-ncRNA offers a promising and robust approach for securing digital infrastructures against the evolving threats posed by quantum computing. 

**Abstract (ZH)**: 在即将到来的后量子时代，传统密码系统日益 Vulnerable 于量子计算攻击，这些攻击可以破坏其数学基础。为应对这一关键挑战，我们提出了一种基于非编码 RNA (ncRNA) 动态折叠性质的加密框架——crypto-ncRNA，利用 RNA 分子的固有物理不可克隆性生成高熵、量子抗性的密钥并产生不可预测的密文。该框架采用一种新颖的多阶段过程：将明文编码为 RNA 序列、使用高级算法预测和操控 RNA 的二级结构以及通过 RNA 分子的固有物理不可克隆性导出密码学密钥。实验评估表明，尽管 crypto-ncRNA 的加密速度略低于 AES，但在效率和扩展性方面，它显著优于 RSA，并且在 NIST SP 800-22 随机性测试中通过率达到了 100%。这些结果展示了 crypto-ncRNA 作为一种抵御量子计算所带来的演进威胁的有前景且稳健的保护数字基础设施的方法。 

---
# Flow Matching Ergodic Coverage 

**Title (ZH)**: 流匹配遍历覆盖 

**Authors**: Max Muchen Sun, Allison Pinosky, Todd Murphey  

**Link**: [PDF](https://arxiv.org/pdf/2504.17872)  

**Abstract**: Ergodic coverage effectively generates exploratory behaviors for embodied agents by aligning the spatial distribution of the agent's trajectory with a target distribution, where the difference between these two distributions is measured by the ergodic metric. However, existing ergodic coverage methods are constrained by the limited set of ergodic metrics available for control synthesis, fundamentally limiting their performance. In this work, we propose an alternative approach to ergodic coverage based on flow matching, a technique widely used in generative inference for efficient and scalable sampling. We formally derive the flow matching problem for ergodic coverage and show that it is equivalent to a linear quadratic regulator problem with a closed-form solution. Our formulation enables alternative ergodic metrics from generative inference that overcome the limitations of existing ones. These metrics were previously infeasible for control synthesis but can now be supported with no computational overhead. Specifically, flow matching with the Stein variational gradient flow enables control synthesis directly over the score function of the target distribution, improving robustness to the unnormalized distributions; on the other hand, flow matching with the Sinkhorn divergence flow enables an optimal transport-based ergodic metric, improving coverage performance on non-smooth distributions with irregular supports. We validate the improved performance and competitive computational efficiency of our method through comprehensive numerical benchmarks and across different nonlinear dynamics. We further demonstrate the practicality of our method through a series of drawing and erasing tasks on a Franka robot. 

**Abstract (ZH)**: 基于流匹配的遍历覆盖：一种生成推断中的流动匹配方法以生成体表代理的探索行为 

---
# CaRL: Learning Scalable Planning Policies with Simple Rewards 

**Title (ZH)**: CaRL: 通过简单的奖励学习可扩展的规划策略 

**Authors**: Bernhard Jaeger, Daniel Dauner, Jens Beißwenger, Simon Gerstenecker, Kashyap Chitta, Andreas Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2504.17838)  

**Abstract**: We investigate reinforcement learning (RL) for privileged planning in autonomous driving. State-of-the-art approaches for this task are rule-based, but these methods do not scale to the long tail. RL, on the other hand, is scalable and does not suffer from compounding errors like imitation learning. Contemporary RL approaches for driving use complex shaped rewards that sum multiple individual rewards, \eg~progress, position, or orientation rewards. We show that PPO fails to optimize a popular version of these rewards when the mini-batch size is increased, which limits the scalability of these approaches. Instead, we propose a new reward design based primarily on optimizing a single intuitive reward term: route completion. Infractions are penalized by terminating the episode or multiplicatively reducing route completion. We find that PPO scales well with higher mini-batch sizes when trained with our simple reward, even improving performance. Training with large mini-batch sizes enables efficient scaling via distributed data parallelism. We scale PPO to 300M samples in CARLA and 500M samples in nuPlan with a single 8-GPU node. The resulting model achieves 64 DS on the CARLA longest6 v2 benchmark, outperforming other RL methods with more complex rewards by a large margin. Requiring only minimal adaptations from its use in CARLA, the same method is the best learning-based approach on nuPlan. It scores 91.3 in non-reactive and 90.6 in reactive traffic on the Val14 benchmark while being an order of magnitude faster than prior work. 

**Abstract (ZH)**: 我们探讨了强化学习在自主驾驶中特权规划中的应用。最新的方法基于规则，但这些方法无法扩展至长尾场景。相比之下，强化学习是可扩展的，并且不会遭受模仿学习中累积错误的问题。用于驾驶的现代强化学习方法使用复杂形状的奖励，这些奖励由多个个体奖励项之和构成，例如进度、位置或方向奖励。我们发现，当mini-batch大小增加时，PPO无法优化这些奖励的一种流行版本，这限制了这些方法的可扩展性。相反，我们提出了一种新的奖励设计，主要是基于优化单一直观的奖励项：路线完成。违规行为通过终止episode或乘性地减少路线完成度予以惩罚。我们发现，在使用我们简单奖励进行训练时，PPO在较高的mini-batch大小下表现出良好的可扩展性，甚至提高了性能。使用大mini-batch大小进行训练能够通过分布式数据并行实现高效的扩展。我们使用单个8-GPU节点将PPO扩展至CARLA中的3亿样本和nuPlan中的5亿样本。所得到的模型在CARLA的longest6 v2基准测试中实现了64 DS，显著优于使用更复杂奖励的其他RL方法。仅需对其在CARLA中的应用进行少量适应，该方法在nuPlan中也是最优的基于学习的方法。它在Val14基准测试中的非反应性交通得分为91.3，在反应性交通得分为90.6，比之前的工作快了几个数量级。 

---
# The Role of Open-Source LLMs in Shaping the Future of GeoAI 

**Title (ZH)**: 开源大模型在塑造GeoAI未来中的作用 

**Authors**: Xiao Huang, Zhengzhong Tu, Xinyue Ye, Michael Goodchild  

**Link**: [PDF](https://arxiv.org/pdf/2504.17833)  

**Abstract**: Large Language Models (LLMs) are transforming geospatial artificial intelligence (GeoAI), offering new capabilities in data processing, spatial analysis, and decision support. This paper examines the open-source paradigm's pivotal role in this transformation. While proprietary LLMs offer accessibility, they often limit the customization, interoperability, and transparency vital for specialized geospatial tasks. Conversely, open-source alternatives significantly advance Geographic Information Science (GIScience) by fostering greater adaptability, reproducibility, and community-driven innovation. Open frameworks empower researchers to tailor solutions, integrate cutting-edge methodologies (e.g., reinforcement learning, advanced spatial indexing), and align with FAIR principles. However, the growing reliance on any LLM necessitates careful consideration of security vulnerabilities, ethical risks, and robust governance for AI-generated geospatial outputs. Ongoing debates on accessibility, regulation, and misuse underscore the critical need for responsible AI development strategies. This paper argues that GIScience advances best not through a single model type, but by cultivating a diverse, interoperable ecosystem combining open-source foundations for innovation, bespoke geospatial models, and interdisciplinary collaboration. By critically evaluating the opportunities and challenges of open-source LLMs within the broader GeoAI landscape, this work contributes to a nuanced discourse on leveraging AI to effectively advance spatial research, policy, and decision-making in an equitable, sustainable, and scientifically rigorous manner. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在变革地球空间人工智能（GeoAI），为数据处理、空间分析和决策支持提供了新的能力。本文探讨了开源范式在这一变革中的关键作用。尽管 proprietary LLM 提供了便捷性，但它们往往限制了为专门的地球空间任务所必需的定制化、互操作性和透明性。相反，开源替代方案显著推动了地理信息科学（GIScience）的发展，促进了更高的适应性、可重复性和社区驱动的创新。开源框架使研究人员能够量身定制解决方案、集成最新方法（例如强化学习、高级空间索引），并符合 FAIR 原则。然而，对任何 LLM 的日益依赖需要仔细考虑安全漏洞、伦理风险和 AI 生成地球空间输出的稳健治理。关于可访问性、监管和滥用的持续辩论强调了负责任的 AI 发展策略的迫切需要。本文认为，GIScience 最好不是通过单一的模型类型来推进，而是通过培养结合开源基础、定制的地球空间模型和跨学科合作的多样、互操作的生态系统来推进。通过批判性地评估开源 LLM 在更广泛的 GeoAI 地景中的机遇和挑战，本文为如何利用 AI 促进空间研究、政策和决策的进步提供了一种全面的讨论，使其更加公平、可持续且科学严谨。 

---
# Fine-Tuning Adversarially-Robust Transformers for Single-Image Dehazing 

**Title (ZH)**: adversarially-robust transformers单像去雾霾的微调研究 

**Authors**: Vlad Vasilescu, Ana Neacsu, Daniela Faur  

**Link**: [PDF](https://arxiv.org/pdf/2504.17829)  

**Abstract**: Single-image dehazing is an important topic in remote sensing applications, enhancing the quality of acquired images and increasing object detection precision. However, the reliability of such structures has not been sufficiently analyzed, which poses them to the risk of imperceptible perturbations that can significantly hinder their performance. In this work, we show that state-of-the-art image-to-image dehazing transformers are susceptible to adversarial noise, with even 1 pixel change being able to decrease the PSNR by as much as 2.8 dB. Next, we propose two lightweight fine-tuning strategies aimed at increasing the robustness of pre-trained transformers. Our methods results in comparable clean performance, while significantly increasing the protection against adversarial data. We further present their applicability in two remote sensing scenarios, showcasing their robust behavior for out-of-distribution data. The source code for adversarial fine-tuning and attack algorithms can be found at this http URL. 

**Abstract (ZH)**: 单张图像去雾霾是遥感应用中的一个重要课题，旨在提高获取图像的质量并提高物体检测精度。然而，此类结构的可靠性尚未得到充分分析，使其面临可能显著妨碍其性能的不可感知扰动的风险。在本文中，我们展示了最先进的图像到图像去雾霾变换器对对抗噪声的敏感性，甚至一个像素的变化也会使PSNR降低2.8 dB。随后，我们提出了两种轻量级微调策略，旨在提高预训练变换器的鲁棒性。我们的方法在获得可比的去噪性能的同时，显著增强了对抗数据的防护能力。此外，我们展示了其在两种遥感场景中的适用性，证明了其对分布外数据的鲁棒行为。对抗微调和攻击算法的源代码可在此网址找到。 

---
# VEU-Bench: Towards Comprehensive Understanding of Video Editing 

**Title (ZH)**: VEU-Bench: 朝着全面理解视频编辑的方向 

**Authors**: Bozheng Li, Yongliang Wu, Yi Lu, Jiashuo Yu, Licheng Tang, Jiawang Cao, Wenqing Zhu, Yuyang Sun, Jay Wu, Wenbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.17828)  

**Abstract**: Widely shared videos on the internet are often edited. Recently, although Video Large Language Models (Vid-LLMs) have made great progress in general video understanding tasks, their capabilities in video editing understanding (VEU) tasks remain unexplored. To address this gap, in this paper, we introduce VEU-Bench (Video Editing Understanding Benchmark), a comprehensive benchmark that categorizes video editing components across various dimensions, from intra-frame features like shot size to inter-shot attributes such as cut types and transitions. Unlike previous video editing understanding benchmarks that focus mainly on editing element classification, VEU-Bench encompasses 19 fine-grained tasks across three stages: recognition, reasoning, and judging. To enhance the annotation of VEU automatically, we built an annotation pipeline integrated with an ontology-based knowledge base. Through extensive experiments with 11 state-of-the-art Vid-LLMs, our findings reveal that current Vid-LLMs face significant challenges in VEU tasks, with some performing worse than random choice. To alleviate this issue, we develop Oscars, a VEU expert model fine-tuned on the curated VEU-Bench dataset. It outperforms existing open-source Vid-LLMs on VEU-Bench by over 28.3% in accuracy and achieves performance comparable to commercial models like GPT-4o. We also demonstrate that incorporating VEU data significantly enhances the performance of Vid-LLMs on general video understanding benchmarks, with an average improvement of 8.3% across nine reasoning tasks. 

**Abstract (ZH)**: 互联网上广泛共享的视频往往会被编辑。尽管视频大语言模型（Vid-LLMs）在一般视频理解任务上取得了显著进展，但它们在视频编辑理解（VEU）任务上的能力仍待探索。为填补这一空白，本文介绍了VEU-Bench（视频编辑理解基准），这是一个全面的基准，从帧内特征（如镜头大小）到帧间属性（如剪辑类型和过渡）等多个维度对视频编辑组件进行分类。与主要集中在编辑元素分类的先前基准不同，VEU-Bench涵盖了三个阶段的19个细粒度任务：识别、推理和判断。为了增强VEU的自动注释，我们构建了一个与本体知识库集成的注释管道。通过与11个最先进的Vid-LLMs的广泛实验，我们的研究发现当前的Vid-LLMs在VEU任务中面临重大挑战，一些模型的表现甚至不如随机选择。为缓解这一问题，我们开发了Oscars，这是一个在精心筛选的VEU-Bench数据集上微调的VEU专家模型，它在VEU-Bench上的准确率上超过了现有开源Vid-LLMs超过28.3%，并在性能上可与商业模型如GPT-4o相媲美。我们还展示了在通用视频理解基准测试中融入VEU数据可以显著增强Vid-LLMs的表现，在九个推理任务中平均提升了8.3%。 

---
# Evolution Meets Diffusion: Efficient Neural Architecture Generation 

**Title (ZH)**: 进化相遇扩散：高效神经架构生成 

**Authors**: Bingye Zhou, Caiyang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.17827)  

**Abstract**: Neural Architecture Search (NAS) has gained widespread attention for its transformative potential in deep learning model design. However, the vast and complex search space of NAS leads to significant computational and time costs. Neural Architecture Generation (NAG) addresses this by reframing NAS as a generation problem, enabling the precise generation of optimal architectures for specific tasks. Despite its promise, mainstream methods like diffusion models face limitations in global search capabilities and are still hindered by high computational and time demands. To overcome these challenges, we propose Evolutionary Diffusion-based Neural Architecture Generation (EDNAG), a novel approach that achieves efficient and training-free architecture generation. EDNAG leverages evolutionary algorithms to simulate the denoising process in diffusion models, using fitness to guide the transition from random Gaussian distributions to optimal architecture distributions. This approach combines the strengths of evolutionary strategies and diffusion models, enabling rapid and effective architecture generation. Extensive experiments demonstrate that EDNAG achieves state-of-the-art (SOTA) performance in architecture optimization, with an improvement in accuracy of up to 10.45%. Furthermore, it eliminates the need for time-consuming training and boosts inference speed by an average of 50 times, showcasing its exceptional efficiency and effectiveness. 

**Abstract (ZH)**: 基于进化扩散的神经架构生成（Evolutionary Diffusion-based Neural Architecture Generation）：高效无训练的架构生成 

---
# FashionM3: Multimodal, Multitask, and Multiround Fashion Assistant based on Unified Vision-Language Model 

**Title (ZH)**: FashionM3：基于统一视觉-语言模型的多模态、多任务、多轮次时尚助手 

**Authors**: Kaicheng Pang, Xingxing Zou, Waikeung Wong  

**Link**: [PDF](https://arxiv.org/pdf/2504.17826)  

**Abstract**: Fashion styling and personalized recommendations are pivotal in modern retail, contributing substantial economic value in the fashion industry. With the advent of vision-language models (VLM), new opportunities have emerged to enhance retailing through natural language and visual interactions. This work proposes FashionM3, a multimodal, multitask, and multiround fashion assistant, built upon a VLM fine-tuned for fashion-specific tasks. It helps users discover satisfying outfits by offering multiple capabilities including personalized recommendation, alternative suggestion, product image generation, and virtual try-on simulation. Fine-tuned on the novel FashionRec dataset, comprising 331,124 multimodal dialogue samples across basic, personalized, and alternative recommendation tasks, FashionM3 delivers contextually personalized suggestions with iterative refinement through multiround interactions. Quantitative and qualitative evaluations, alongside user studies, demonstrate FashionM3's superior performance in recommendation effectiveness and practical value as a fashion assistant. 

**Abstract (ZH)**: 时尚搭配和个人化推荐在现代零售中至关重要，为时尚行业带来了巨大的经济价值。随着视觉语言模型（VLM）的发展，通过自然语言和视觉交互为零售业带来了新的机遇。本文提出了一种基于专门任务微调的视觉语言模型构建的多模态、多任务、多轮次时尚助手——FashionM3。它通过提供个性化推荐、替代建议、产品图像生成和虚拟试穿模拟等多种功能，帮助用户发现满意搭配。FashionM3基于包含331,124个多模态对话样本的新颖FashionRec数据集，涵盖基本推荐、个性化推荐和替代推荐任务，通过多轮次交互提供上下文相关的个性化建议。定量和定性评估，以及用户研究，证明了FashionM3在推荐效果和作为时尚助手的实际价值方面的优越性能。 

---
# Dual Prompting Image Restoration with Diffusion Transformers 

**Title (ZH)**: 双提示图像恢复扩散变换器 

**Authors**: Dehong Kong, Fan Li, Zhixin Wang, Jiaqi Xu, Renjing Pei, Wenbo Li, WenQi Ren  

**Link**: [PDF](https://arxiv.org/pdf/2504.17825)  

**Abstract**: Recent state-of-the-art image restoration methods mostly adopt latent diffusion models with U-Net backbones, yet still facing challenges in achieving high-quality restoration due to their limited capabilities. Diffusion transformers (DiTs), like SD3, are emerging as a promising alternative because of their better quality with scalability. In this paper, we introduce DPIR (Dual Prompting Image Restoration), a novel image restoration method that effectivly extracts conditional information of low-quality images from multiple perspectives. Specifically, DPIR consits of two branches: a low-quality image conditioning branch and a dual prompting control branch. The first branch utilizes a lightweight module to incorporate image priors into the DiT with high efficiency. More importantly, we believe that in image restoration, textual description alone cannot fully capture its rich visual characteristics. Therefore, a dual prompting module is designed to provide DiT with additional visual cues, capturing both global context and local appearance. The extracted global-local visual prompts as extra conditional control, alongside textual prompts to form dual prompts, greatly enhance the quality of the restoration. Extensive experimental results demonstrate that DPIR delivers superior image restoration performance. 

**Abstract (ZH)**: 最近的先进图像恢复方法大多采用具有U-Net骨干的潜在扩散模型，但由于其局限性，仍面临实现高质量恢复的挑战。扩散变压器（DiTs），如SD3，因其更好的质量和可扩展性正 emergence 作为一种有前景的替代方案。在本文中，我们提出了一种名为DPIR（Dual Prompting Image Restoration）的新颖图像恢复方法，能够从多个角度有效提取低质量图像的条件信息。具体来说，DPIR 包含两个分支：一个低质量图像条件分支和一个双重提示控制分支。第一个分支利用一个轻量级模块以高效率将图像先验信息整合到 DiT 中。更重要的是，我们认为在图像恢复中，仅依赖文本描述无法完全捕捉其丰富的视觉特征。因此，我们设计了一个双重提示模块，为 DiT 提供额外的视觉线索，捕获全局上下文和局部外观。提取出的全局-局部视觉提示作为额外条件控制，与文本提示一起构成双重提示，极大地提升了恢复质量。广泛的实验结果表明，DPIR 能够提供优异的图像恢复性能。 

---
# EduBot -- Can LLMs Solve Personalized Learning and Programming Assignments? 

**Title (ZH)**: EduBot——大型语言模型能解决个性化学习和编程作业问题吗？ 

**Authors**: Yibin Wang, Jiaxi Xie, Lakshminarayanan Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2504.17824)  

**Abstract**: The prevalence of Large Language Models (LLMs) is revolutionizing the process of writing code. General and code LLMs have shown impressive performance in generating standalone functions and code-completion tasks with one-shot queries. However, the ability to solve comprehensive programming tasks with recursive requests and bug fixes remains questionable. In this paper, we propose EduBot, an intelligent automated assistant system that combines conceptual knowledge teaching, end-to-end code development, personalized programming through recursive prompt-driven methods, and debugging with limited human interventions powered by LLMs. We show that EduBot can solve complicated programming tasks consisting of sub-tasks with increasing difficulties ranging from conceptual to coding questions by recursive automatic prompt-driven systems without finetuning on LLMs themselves. To further evaluate EduBot's performance, we design and conduct a benchmark suite consisting of 20 scenarios in algorithms, machine learning, and real-world problems. The result shows that EduBot can complete most scenarios in less than 20 minutes. Based on the benchmark suites, we perform a comparative study to take different LLMs as the backbone and to verify EduBot's compatibility and robustness across LLMs with varying capabilities. We believe that EduBot is an exploratory approach to explore the potential of pre-trained LLMs in multi-step reasoning and code generation for solving personalized assignments with knowledge learning and code generation. 

**Abstract (ZH)**: 大型语言模型的盛行正在革新编写代码的过程。通用和代码语言模型在一次查询中生成独立函数和代码补全任务方面展现了令人印象深刻的性能。然而，它们解决包含递归请求和错误修复的全面编程任务的能力仍有待商榷。本文提出了一种名为EduBot的智能自动化助手系统，该系统结合了概念知识教学、端到端的代码开发、递归提示驱动的个性化编程以及有限的人工干预下的调试。我们展示了EduBot可以通过递归自动提示驱动系统解决由概念到编码问题构成的复杂编程任务，而无需对语言模型进行微调。为了进一步评估EduBot的性能，我们设计并实施了一个包含20个情景的基准测试套件，涉及算法、机器学习和现实世界问题。结果显示，EduBot可以在少于20分钟内完成大多数情景。基于基准测试套件，我们进行了比较研究，使用不同的大型语言模型作为骨干，验证了EduBot在不同能力的大型语言模型之间的兼容性和鲁棒性。我们认为，EduBot探索了预训练大型语言模型在多步推理和代码生成方面解决个性化作业的潜力，结合了知识学习和代码生成。 

---
# The Cloud Weaving Model for AI development 

**Title (ZH)**: AI发展中的云编织模型 

**Authors**: Darcy Kim, Aida Kalender, Sennay Ghebreab, Giovanni Sileno  

**Link**: [PDF](https://arxiv.org/pdf/2504.17823)  

**Abstract**: While analysing challenges in pilot projects developing AI with marginalized communities, we found it difficult to express them within commonly used paradigms. We therefore constructed an alternative conceptual framework to ground AI development in the social fabric -- the Cloud Weaving Model -- inspired (amongst others) by indigenous knowledge, motifs from nature, and Eastern traditions. This paper introduces and elaborates on the fundamental elements of the model (clouds, spiders, threads, spiderwebs, and weather) and their interpretation in an AI context. The framework is then applied to comprehend patterns observed in co-creation pilots approaching marginalized communities, highlighting neglected yet relevant dimensions for responsible AI development. 

**Abstract (ZH)**: 基于云织模型：在边缘化社区中开发人工智能所面临的挑战及其替代概念框架 

---
# A multi-scale vision transformer-based multimodal GeoAI model for mapping Arctic permafrost thaw 

**Title (ZH)**: 基于多尺度视觉变换器的多模态GeoAI模型：绘制北极冻土 thaw 地图 

**Authors**: Wenwen Li, Chia-Yu Hsu, Sizhe Wang, Zhining Gu, Yili Yang, Brendan M. Rogers, Anna Liljedahl  

**Link**: [PDF](https://arxiv.org/pdf/2504.17822)  

**Abstract**: Retrogressive Thaw Slumps (RTS) in Arctic regions are distinct permafrost landforms with significant environmental impacts. Mapping these RTS is crucial because their appearance serves as a clear indication of permafrost thaw. However, their small scale compared to other landform features, vague boundaries, and spatiotemporal variation pose significant challenges for accurate detection. In this paper, we employed a state-of-the-art deep learning model, the Cascade Mask R-CNN with a multi-scale vision transformer-based backbone, to delineate RTS features across the Arctic. Two new strategies were introduced to optimize multimodal learning and enhance the model's predictive performance: (1) a feature-level, residual cross-modality attention fusion strategy, which effectively integrates feature maps from multiple modalities to capture complementary information and improve the model's ability to understand complex patterns and relationships within the data; (2) pre-trained unimodal learning followed by multimodal fine-tuning to alleviate high computing demand while achieving strong model performance. Experimental results demonstrated that our approach outperformed existing models adopting data-level fusion, feature-level convolutional fusion, and various attention fusion strategies, providing valuable insights into the efficient utilization of multimodal data for RTS mapping. This research contributes to our understanding of permafrost landforms and their environmental implications. 

**Abstract (ZH)**: 北极地区退化融沉（Retrogressive Thaw Slumps, RTS）的测绘：基于多尺度视觉变换器的Cascade Mask R-CNN模型研究 

---
# Research on Cloud Platform Network Traffic Monitoring and Anomaly Detection System based on Large Language Models 

**Title (ZH)**: 基于大型语言模型的云平台网络流量监控与异常检测系统研究 

**Authors**: Ze Yang, Yihong Jin, Juntian Liu, Xinhe Xu, Yihan Zhang, Shuyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.17807)  

**Abstract**: The rapidly evolving cloud platforms and the escalating complexity of network traffic demand proper network traffic monitoring and anomaly detection to ensure network security and performance. This paper introduces a large language model (LLM)-based network traffic monitoring and anomaly detection system. In addition to existing models such as autoencoders and decision trees, we harness the power of large language models for processing sequence data from network traffic, which allows us a better capture of underlying complex patterns, as well as slight fluctuations in the dataset. We show for a given detection task, the need for a hybrid model that incorporates the attention mechanism of the transformer architecture into a supervised learning framework in order to achieve better accuracy. A pre-trained large language model analyzes and predicts the probable network traffic, and an anomaly detection layer that considers temporality and context is added. Moreover, we present a novel transfer learning-based methodology to enhance the model's effectiveness to quickly adapt to unknown network structures and adversarial conditions without requiring extensive labeled datasets. Actual results show that the designed model outperforms traditional methods in detection accuracy and computational efficiency, effectively identify various network anomalies such as zero-day attacks and traffic congestion pattern, and significantly reduce the false positive rate. 

**Abstract (ZH)**: 基于大语言模型的网络流量监测与异常检测系统 

---
# Fuzzy Logic -- Based Scheduling System for Part-Time Workforce 

**Title (ZH)**: 基于模糊逻辑的兼职人员调度系统 

**Authors**: Tri Nguyen, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2504.17805)  

**Abstract**: This paper explores the application of genetic fuzzy systems to efficiently generate schedules for a team of part-time student workers at a university. Given the preferred number of working hours and availability of employees, our model generates feasible solutions considering various factors, such as maximum weekly hours, required number of workers on duty, and the preferred number of working hours. The algorithm is trained and tested with availability data collected from students at the University of Cincinnati. The results demonstrate the algorithm's efficiency in producing schedules that meet operational criteria and its robustness in understaffed conditions. 

**Abstract (ZH)**: 本研究探讨了遗传模糊系统在高效生成大学兼职学生员工团队工作时间表中的应用。根据员工的偏好工作时数和可用性，我们的模型在考虑诸如每周最大工时、所需的值班员工数量以及偏好工作时数等因素的情况下生成可行的解决方案。该算法使用俄亥俄州辛辛那提大学学生收集的可用性数据进行训练和测试。研究结果展示了该算法在满足运营标准方面和 understaffed 条件下的鲁棒性。 

---
# Evolution of Optimization Algorithms for Global Placement via Large Language Models 

**Title (ZH)**: 大型语言模型在全局布线优化算法进化中的应用 

**Authors**: Xufeng Yao, Jiaxi Jiang, Yuxuan Zhao, Peiyu Liao, Yibo Lin, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.17801)  

**Abstract**: Optimization algorithms are widely employed to tackle complex problems, but designing them manually is often labor-intensive and requires significant expertise. Global placement is a fundamental step in electronic design automation (EDA). While analytical approaches represent the state-of-the-art (SOTA) in global placement, their core optimization algorithms remain heavily dependent on heuristics and customized components, such as initialization strategies, preconditioning methods, and line search techniques. This paper presents an automated framework that leverages large language models (LLM) to evolve optimization algorithms for global placement. We first generate diverse candidate algorithms using LLM through carefully crafted prompts. Then we introduce an LLM-based genetic flow to evolve selected candidate algorithms. The discovered optimization algorithms exhibit substantial performance improvements across many benchmarks. Specifically, Our design-case-specific discovered algorithms achieve average HPWL improvements of \textbf{5.05\%}, \text{5.29\%} and \textbf{8.30\%} on MMS, ISPD2005 and ISPD2019 benchmarks, and up to \textbf{17\%} improvements on individual cases. Additionally, the discovered algorithms demonstrate good generalization ability and are complementary to existing parameter-tuning methods. 

**Abstract (ZH)**: 基于大规模语言模型的全局布局优化算法自动化框架 

---
# Subfunction Structure Matters: A New Perspective on Local Optima Networks 

**Title (ZH)**: 子功能结构很重要：局部最优网络的一个新视角 

**Authors**: S. L. Thomson, M. W. Przewozniczek  

**Link**: [PDF](https://arxiv.org/pdf/2504.17799)  

**Abstract**: Local optima networks (LONs) capture fitness landscape information. They are typically constructed in a black-box manner; information about the problem structure is not utilised. This also applies to the analysis of LONs: knowledge about the problem, such as interaction between variables, is not considered. We challenge this status-quo with an alternative approach: we consider how LON analysis can be improved by incorporating subfunction-based information - this can either be known a-priori or learned during search. To this end, LONs are constructed for several benchmark pseudo-boolean problems using three approaches: firstly, the standard algorithm; a second algorithm which uses deterministic grey-box crossover; and a third algorithm which selects perturbations based on learned information about variable interactions. Metrics related to subfunction changes in a LON are proposed and compared with metrics from previous literature which capture other aspects of a LON. Incorporating problem structure in LON construction and analysing it can bring enriched insight into optimisation dynamics. Such information may be crucial to understanding the difficulty of solving a given problem with state-of-the-art linkage learning optimisers. In light of the results, we suggest incorporation of problem structure as an alternative paradigm in landscape analysis for problems with known or suspected subfunction structure. 

**Abstract (ZH)**: 局部最优网络（LONs）捕获了适应度景观信息。它们通常以黑盒方式构建；问题结构的相关信息未被利用。这也适用于LON分析：关于问题的知识，例如变量间的交互作用，未被考虑。我们提出了一个替代方法来挑战这一现状：我们探讨如何通过结合基于子函数的信息来改进LON分析——这些信息可以先验地已知或者在搜索过程中学习到。为此，我们使用三种方法为几种基准伪布尔问题构建LON：首先，标准算法；其次，使用确定性灰盒 crossover 的算法；最后，基于关于变量交互信息的学习来进行扰动选择的算法。提出了与LON中的子函数变化相关的度量，并将其与捕捉LON其他方面的度量进行比较。在LON构建和分析中融入问题结构可以提供更丰富的优化动力学见解。这些信息对于理解利用当前最先进的连接学习优化器解决给定问题的难度可能是至关重要的。基于结果，我们建议将问题结构的融入作为景观分析中的一种替代范式，用于具有已知或疑似子函数结构的问题。 

---
# My Precious Crash Data: Barriers and Opportunities in Encouraging Autonomous Driving Companies to Share Safety-Critical Data 

**Title (ZH)**: 我宝贵的安全数据：鼓励自动驾驶公司共享关键安全数据的障碍与机遇 

**Authors**: Hauke Sandhaus, Angel Hsing-Chi Hwang, Wendy Ju, Qian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17792)  

**Abstract**: Safety-critical data, such as crash and near-crash records, are crucial to improving autonomous vehicle (AV) design and development. Sharing such data across AV companies, academic researchers, regulators, and the public can help make all AVs safer. However, AV companies rarely share safety-critical data externally. This paper aims to pinpoint why AV companies are reluctant to share safety-critical data, with an eye on how these barriers can inform new approaches to promote sharing. We interviewed twelve AV company employees who actively work with such data in their day-to-day work. Findings suggest two key, previously unknown barriers to data sharing: (1) Datasets inherently embed salient knowledge that is key to improving AV safety and are resource-intensive. Therefore, data sharing, even within a company, is fraught with politics. (2) Interviewees believed AV safety knowledge is private knowledge that brings competitive edges to their companies, rather than public knowledge for social good. We discuss the implications of these findings for incentivizing and enabling safety-critical AV data sharing, specifically, implications for new approaches to (1) debating and stratifying public and private AV safety knowledge, (2) innovating data tools and data sharing pipelines that enable easier sharing of public AV safety data and knowledge; (3) offsetting costs of curating safety-critical data and incentivizing data sharing. 

**Abstract (ZH)**: 安全关键数据，例如碰撞和接近碰撞记录，对于提升自动驾驶车辆（AV）的设计与开发至关重要。跨AV公司、学术研究人员、监管机构和公众共享此类数据有助于提高所有AV的安全性。然而，AV公司很少对外共享安全关键数据。本文旨在探究AV公司不愿共享安全关键数据的原因，并分析这些障碍如何有助于提出新的促进数据共享的方法。我们对十二名在日常工作中积极处理此类数据的AV公司员工进行了访谈。研究发现存在两个新的关键障碍：（1）数据集本身嵌入了对提升AV安全性至关重要的核心知识，并且资源密集，因此即使在公司内部，共享数据也充满了政治因素。（2）受访者认为AV安全知识是公司的私有知识，为公司带来竞争优势，而不是社会公共知识。我们讨论了这些发现对激励和促进安全关键AV数据共享的含义，具体包括（1）关于争论和划分公共与私营AV安全知识的新方法，（2）创新数据工具和数据共享管道以使公共AV安全数据和知识更容易共享，（3）抵消收集安全关键数据的成本并激励数据共享的含义。 

---
