# Inspire or Predict? Exploring New Paradigms in Assisting Classical Planners with Large Language Models 

**Title (ZH)**: 启发还是预测？探索大型语言模型辅助经典规划器的新范式 

**Authors**: Wenkai Yu, Jianhang Tang, Yang Zhang, Shanjiang Tang, Kebing Jin, Hankz Hankui Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2508.11524)  

**Abstract**: Addressing large-scale planning problems has become one of the central challenges in the planning community, deriving from the state-space explosion caused by growing objects and actions. Recently, researchers have explored the effectiveness of leveraging Large Language Models (LLMs) to generate helpful actions and states to prune the search space. However, prior works have largely overlooked integrating LLMs with domain-specific knowledge to ensure valid plans. In this paper, we propose a novel LLM-assisted planner integrated with problem decomposition, which first decomposes large planning problems into multiple simpler sub-tasks. Then we explore two novel paradigms to utilize LLMs, i.e., LLM4Inspire and LLM4Predict, to assist problem decomposition, where LLM4Inspire provides heuristic guidance according to general knowledge and LLM4Predict employs domain-specific knowledge to infer intermediate conditions. We empirically validate the effectiveness of our planner across multiple domains, demonstrating the ability of search space partition when solving large-scale planning problems. The experimental results show that LLMs effectively locate feasible solutions when pruning the search space, where infusing domain-specific knowledge into LLMs, i.e., LLM4Predict, holds particular promise compared with LLM4Inspire, which offers general knowledge within LLMs. 

**Abstract (ZH)**: 地址大规模规划问题已成为规划社区的核心挑战之一，源于对象和动作增多引起的状态空间爆炸。最近，研究人员探索了利用大型语言模型（LLMs）生成有助于裁剪搜索空间的有益动作和状态的有效性。然而，先前的工作大多忽视了将LLMs与领域特定知识集成以确保有效计划的重要性。在本文中，我们提出了一种结合问题分解的新型LLM辅助规划器，首先将大规模规划问题分解为多个更简单的子任务。然后，我们探索了利用LLMs的两种新颖范式，即LLM4Inspire和LLM4Predict，以协助问题分解，其中LLM4Inspire根据通用知识提供启发式指导，而LLM4Predict利用领域特定知识推断中间条件。我们跨多个领域 empirically 验证了我们规划器的有效性，展示了在解决大规模规划问题时对搜索空间分区的能力。实验结果表明，当裁剪搜索空间时，LLMs能够有效定位可行解，其中将领域特定知识融入LLMs，即LLM4Predict，相较于提供通用知识的LLM4Inspire更具潜力。 

---
# Landmark-Assisted Monte Carlo Planning 

**Title (ZH)**: 地标辅助蒙特卡洛规划 

**Authors**: David H. Chan, Mark Roberts, Dana S. Nau  

**Link**: [PDF](https://arxiv.org/pdf/2508.11493)  

**Abstract**: Landmarks$\unicode{x2013}$conditions that must be satisfied at some point in every solution plan$\unicode{x2013}$have contributed to major advancements in classical planning, but they have seldom been used in stochastic domains. We formalize probabilistic landmarks and adapt the UCT algorithm to leverage them as subgoals to decompose MDPs; core to the adaptation is balancing between greedy landmark achievement and final goal achievement. Our results in benchmark domains show that well-chosen landmarks can significantly improve the performance of UCT in online probabilistic planning, while the best balance of greedy versus long-term goal achievement is problem-dependent. The results suggest that landmarks can provide helpful guidance for anytime algorithms solving MDPs. 

**Abstract (ZH)**: 地标：条件必须在每个解决方案计划中的某个点被满足，它们在经典规划中取得了重大进展，但很少被用于随机领域。我们形式化了概率地标，并适应了UCT算法以利用它们作为子目标来分解MDPs；适应的核心在于权衡贪婪地标达成与最终目标达成之间的平衡。基准领域中的实验结果显示，精心选择的地标可以显著提高UCT在在线概率规划中的性能，而贪婪与长期目标达成的最佳平衡依赖于具体问题。这些结果表明，地标可以为解决MDPs的随时算法提供有益的指导。 

---
# Inclusion Arena: An Open Platform for Evaluating Large Foundation Models with Real-World Apps 

**Title (ZH)**: 包容性竞技场：一个评估大型基础模型与实际应用的开源平台 

**Authors**: Kangyu Wang, Hongliang He, Lin Liu, Ruiqi Liang, Zhenzhong Lan, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.11452)  

**Abstract**: Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have ushered in a new era of AI capabilities, demonstrating near-human-level performance across diverse scenarios. While numerous benchmarks (e.g., MMLU) and leaderboards (e.g., Chatbot Arena) have been proposed to help evolve the development of LLMs and MLLMs, most rely on static datasets or crowdsourced general-domain prompts, often falling short of reflecting performance in real-world applications. To bridge this critical gap, we present Inclusion Arena, a live leaderboard that ranks models based on human feedback collected directly from AI-powered applications. Our platform integrates pairwise model comparisons into natural user interactions, ensuring evaluations reflect practical usage scenarios. For robust model ranking, we employ the Bradley-Terry model augmented with two key innovations: (1) Placement Matches, a cold-start mechanism to quickly estimate initial ratings for newly integrated models, and (2) Proximity Sampling, an intelligent comparison strategy that prioritizes battles between models of similar capabilities to maximize information gain and enhance rating stability. Extensive empirical analyses and simulations demonstrate that Inclusion Arena yields reliable and stable rankings, exhibits higher data transitivity compared to general crowdsourced datasets, and significantly mitigates the risk of malicious manipulation. By fostering an open alliance between foundation models and real-world applications, Inclusion Arena aims to accelerate the development of LLMs and MLLMs truly optimized for practical, user-centric deployments. The platform is publicly accessible at this https URL. 

**Abstract (ZH)**: 包容竞技场：一种基于直接用户反馈的实时排行榜，用于评估大型语言模型和多模态大型语言模型 

---
# AIM-Bench: Evaluating Decision-making Biases of Agentic LLM as Inventory Manager 

**Title (ZH)**: AIM-Bench: 评估行动者LLM决策偏见的库存管理者评估benchmark 

**Authors**: Xuhua Zhao, Yuxuan Xie, Caihua Chen, Yuxiang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.11416)  

**Abstract**: Recent advances in mathematical reasoning and the long-term planning capabilities of large language models (LLMs) have precipitated the development of agents, which are being increasingly leveraged in business operations processes. Decision models to optimize inventory levels are one of the core elements of operations management. However, the capabilities of the LLM agent in making inventory decisions in uncertain contexts, as well as the decision-making biases (e.g. framing effect, etc.) of the agent, remain largely unexplored. This prompts concerns regarding the capacity of LLM agents to effectively address real-world problems, as well as the potential implications of biases that may be present. To address this gap, we introduce AIM-Bench, a novel benchmark designed to assess the decision-making behaviour of LLM agents in uncertain supply chain management scenarios through a diverse series of inventory replenishment experiments. Our results reveal that different LLMs typically exhibit varying degrees of decision bias that are similar to those observed in human beings. In addition, we explored strategies to mitigate the pull-to-centre effect and the bullwhip effect, namely cognitive reflection and implementation of information sharing. These findings underscore the need for careful consideration of the potential biases in deploying LLMs in Inventory decision-making scenarios. We hope that these insights will pave the way for mitigating human decision bias and developing human-centred decision support systems for supply chains. 

**Abstract (ZH)**: Recent advances in数学推理和大规模语言模型（LLMs）的长期规划能力推动了代理的开发，这些代理在业务操作流程中日益受到重视。库存优化决策模型是运营管理的核心要素之一。然而，LLM代理在不确定环境下进行库存决策的能力以及代理的决策偏见（如框架效应等）仍鲜有研究。这引发对其能否有效解决实际问题及潜在偏见影响的担忧。为应对这一空白，我们引入了AIM-Bench，这是一种新型基准，旨在通过一系列多样的库存补充实验评估LLM代理在不确定供应链管理场景中的决策行为。我们的结果显示，不同LLM通常表现出不同程度与人类相似的决策偏见。此外，我们探索了减轻中心效应和牛鞭效应的策略，如认知反思和信息共享实施。这些发现强调了在库存决策场景中部署LLM时谨慎考虑潜在偏见的重要性。我们希望这些见解能为减轻人类决策偏见并为供应链开发以人为中心的决策支持系统铺平道路。 

---
# CRAFT-GUI: Curriculum-Reinforced Agent For GUI Tasks 

**Title (ZH)**: CRAFT-GUI：基于 Curriculum 学习的GUI任务代理 

**Authors**: Songqin Nong, Jingxuan Xu, Sheng Zhou, Jianfeng Chen, Xiaoxuan Tang, Tao Jiang, Wenhao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11360)  

**Abstract**: As autonomous agents become adept at understanding and interacting with graphical user interface (GUI) environments, a new era of automated task execution is emerging. Recent studies have demonstrated that Reinforcement Learning (RL) can effectively enhance agents' performance in dynamic interactive GUI environments. However, these methods face two key limitations: (1) they overlook the significant variation in difficulty across different GUI tasks by treating the entire training data as a uniform set, which hampers the agent's ability to adapt its learning process; and (2) most approaches collapse task-specific nuances into a single, coarse reward, leaving the agent with a uniform signal that yields inefficient policy updates. To address these limitations, we propose CRAFT-GUI, a curriculum learning framework based on Group Relative Policy Optimization (GRPO) that explicitly accounts for the varying difficulty across trajectories. To enable more fine-grained policy optimization, we design a reward function that combines simple rule-based signals with model-judged evaluation, providing richer and more nuanced feedback during training. Experimental results demonstrate that our method achieves significant improvements over previous state-of-the-art approaches, outperforming them by 5.6% on public benchmarks Android Control and 10.3% on our internal online benchmarks, respectively. These findings empirically validate the effectiveness of integrating reinforcement learning with curriculum learning in GUI interaction tasks. 

**Abstract (ZH)**: 基于组相对策略优化的CURATE-GUI框架：强化学习在图形用户界面交互任务中的应用 

---
# SAGE: Scale-Aware Gradual Evolution for Continual Knowledge Graph Embedding 

**Title (ZH)**: SAGE: 意识到规模的逐步演化持续知识图嵌入 

**Authors**: Yifei Li, Lingling Zhang, Hang Yan, Tianzhe Zhao, Zihan Ma, Muye Huang, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11347)  

**Abstract**: Traditional knowledge graph (KG) embedding methods aim to represent entities and relations in a low-dimensional space, primarily focusing on static graphs. However, real-world KGs are dynamically evolving with the constant addition of entities, relations and facts. To address such dynamic nature of KGs, several continual knowledge graph embedding (CKGE) methods have been developed to efficiently update KG embeddings to accommodate new facts while maintaining learned knowledge. As KGs grow at different rates and scales in real-world scenarios, existing CKGE methods often fail to consider the varying scales of updates and lack systematic evaluation throughout the entire update process. In this paper, we propose SAGE, a scale-aware gradual evolution framework for CKGE. Specifically, SAGE firstly determine the embedding dimensions based on the update scales and expand the embedding space accordingly. The Dynamic Distillation mechanism is further employed to balance the preservation of learned knowledge and the incorporation of new facts. We conduct extensive experiments on seven benchmarks, and the results show that SAGE consistently outperforms existing baselines, with a notable improvement of 1.38% in MRR, 1.25% in H@1 and 1.6% in H@10. Furthermore, experiments comparing SAGE with methods using fixed embedding dimensions show that SAGE achieves optimal performance on every snapshot, demonstrating the importance of adaptive embedding dimensions in CKGE. The codes of SAGE are publicly available at: this https URL. 

**Abstract (ZH)**: 面向规模感知渐进演化的持续知识图嵌入框架SAGE 

---
# Beyond Solving Math Quiz: Evaluating the Ability of Large Reasoning Models to Ask for Information 

**Title (ZH)**: 超越解决数学测题：评估大型 reasoning 模型请求信息的能力lóg
user
Beyond Human-Level: Understanding the General Competence of Large Language Models Through Mathematical Reasoning超出人类水平：通过数学推理理解大型语言模型的通用能力 

**Authors**: Youcheng Huang, Bowen Qin, Chen Huang, Duanyu Feng, Xi Yang, Wenqiang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2508.11252)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable problem-solving abilities in mathematics, as evaluated by existing benchmarks exclusively on well-defined problems. However, such evaluation setup constitutes a critical gap, since a genuine intelligent agent should not only solve problems (as a math quiz solver), but also be able~to ask for information when the problems lack sufficient information, enabling proactivity in responding users' requests. To bridge such gap, we proposes a new dataset consisting of two types of incomplete problems with diverse contexts. Based on the dataset, our systematical evaluation of LRMs reveals their inability in proactively asking for information. In addition, we uncover the behaviors related to overthinking and hallucination of LRMs, and highlight the potential and challenges of supervised fine-tuning in learning such ability. We hope to provide new insights in developing LRMs with genuine intelligence, rather than just solving problems. 

**Abstract (ZH)**: 大推理模型（LRMs）在数学问题求解方面的卓越能力已在现有的基准测试中得到评估，这些基准测试仅针对清晰定义的问题。然而，这种评估方式存在一个关键缺陷，因为真正的智能代理不仅应该能够解决问题（如数学测验解答器），还应该能够在问题缺乏足够信息时主动请求信息，以增强对用户请求的主动响应。为了弥合这一差距，我们提出了一种新的数据集，包含两类具有多样化背景的不完整问题。基于该数据集，我们系统性地评估了大推理模型的能力，揭示了它们在主动请求信息方面的能力不足。此外，我们还发现了大推理模型在过度思考和妄想方面的行为，并强调了在监督微调中学习这种能力的潜力和挑战。我们希望为开发具备真正智能的大推理模型提供新的洞察，而不仅仅是解决问题。 

---
# On Strong and Weak Admissibility in Non-Flat Assumption-Based Argumentation 

**Title (ZH)**: 非平坦假设论辩中的强可接受性和弱可接受性研究 

**Authors**: Matti Berthold, Lydia Blümel, Anna Rapberger  

**Link**: [PDF](https://arxiv.org/pdf/2508.11182)  

**Abstract**: In this work, we broaden the investigation of admissibility notions in the context of assumption-based argumentation (ABA). More specifically, we study two prominent alternatives to the standard notion of admissibility from abstract argumentation, namely strong and weak admissibility, and introduce the respective preferred, complete and grounded semantics for general (sometimes called non-flat) ABA. To do so, we use abstract bipolar set-based argumentation frameworks (BSAFs) as formal playground since they concisely capture the relations between assumptions and are expressive enough to represent general non-flat ABA frameworks, as recently shown. While weak admissibility has been recently investigated for a restricted fragment of ABA in which assumptions cannot be derived (flat ABA), strong admissibility has not been investigated for ABA so far. We introduce strong admissibility for ABA and investigate desirable properties. We furthermore extend the recent investigations of weak admissibility in the flat ABA fragment to the non-flat case. We show that the central modularization property is maintained under classical, strong, and weak admissibility. We also show that strong and weakly admissible semantics in non-flat ABA share some of the shortcomings of standard admissible semantics and discuss ways to address these. 

**Abstract (ZH)**: 在这项工作中，我们扩展了对假设基于论辩论中容许性概念的研究。具体地，我们研究了抽象论辩中标准容许性概念的两种主要替代方案，即强容许性和弱容许性，并引入了适用于一般（有时称为非扁平）假设基于论辩的相应优先级、完备性和基础语义。为此，我们使用抽象双极集合论辩框架（BSAFs）作为形式化的研究平台，因为它们简洁地捕捉了假设之间的关系，并且能够表达一般非扁平假设基于论辩框架，这是最近的研究成果。虽然弱容许性最近在假设不可推导的扁平假设基于论辩片段中进行了研究，但强容许性尚未在假设基于论辩中进行研究。我们为假设基于论辩引入了强容许性并研究了其 desirable 属性。我们还扩展了在扁平假设基于论辩片段中对弱容许性的最近研究，将其推广到非扁平情况。我们证明了在经典、强和弱容许性下保持了核心模块化特性。我们还展示了非扁平假设基于论辩中的强容许性和弱容许性语义与标准容许性语义共享的一些缺点，并讨论了解决这些问题的方法。 

---
# Learn to optimize for automatic proton PBS treatment planning for H&N cancers 

**Title (ZH)**: 自动质子治疗计划优化学习：针对头颈癌的H&N癌症质子束治疗计划自动化优化 

**Authors**: Qingqing Wang, Liqiang Xiao, Chang Chang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11085)  

**Abstract**: Proton PBS treatment planning for H&N cancers involves numerous conflicting objectives, requiring significant effort from human planners to balance and satisfy multiple clinical goals during planning. To achieve this, experience-demanding objective parameter adjustment and computationally expensive inverse optimization are performed iteratively. Extensive efforts have been made to automatically adjust objective parameters, but the most time-consuming component, i.e., inverse optimization, still relies heavily on theory-driven approaches. We propose a data-driven inverse optimizer and integrate it into a PPO-based automatic treatment planning framework to automatically generate high-quality plans within a clinical acceptable planning time. The inverse optimizer is a L2O method that predicts update steps by learning from the task-specific data distribution. For the first time, we integrate techniques designed for long-context processing, originally developed for LLMs, into a Transformer-based L2O framework to address the scalability issue of existing L2O methods. The PPO framework functions as an outer-loop virtual planner, autonomously adjusting objective parameters through a policy network, and the dose predictor is used to initialize objective parameters. The inner-loop L2O inverse optimizer computes machine-deliverable MU values based on objectives refined by the PPO policy network. 97 patients are collected in this study, and compared with L-BFGSB, our L2O-based inverse optimizer improves the effectiveness and efficiency by 22.97% and 36.41%, respectively. In conjunction with the PPO-based learned virtual planner, plans generated by our framework within an average of 2.55 hours show improved or comparable OAR sparing with superior target coverage for patients with different prescription dose levels, number of target volumes, beam angles, etc., compared with human-generated plans. 

**Abstract (ZH)**: 数据驱动的质子PBS治疗计划方法及其在H&N癌症治疗中的应用：基于PPO的方法 

---
# From Individual to Multi-Agent Algorithmic Recourse: Minimizing the Welfare Gap via Capacitated Bipartite Matching 

**Title (ZH)**: 从个体到多智能体算法可问责性：通过容量受限二部图匹配最小化福利差距 

**Authors**: Zahra Khotanlou, Kate Larson, Amir-Hossein Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2508.11070)  

**Abstract**: Decision makers are increasingly relying on machine learning in sensitive situations. In such settings, algorithmic recourse aims to provide individuals with actionable and minimally costly steps to reverse unfavorable AI-driven decisions. While existing research predominantly focuses on single-individual (i.e., seeker) and single-model (i.e., provider) scenarios, real-world applications often involve multiple interacting stakeholders. Optimizing outcomes for seekers under an individual welfare approach overlooks the inherently multi-agent nature of real-world systems, where individuals interact and compete for limited resources. To address this, we introduce a novel framework for multi-agent algorithmic recourse that accounts for multiple recourse seekers and recourse providers. We model this many-to-many interaction as a capacitated weighted bipartite matching problem, where matches are guided by both recourse cost and provider capacity. Edge weights, reflecting recourse costs, are optimized for social welfare while quantifying the welfare gap between individual welfare and this collectively feasible outcome. We propose a three-layer optimization framework: (1) basic capacitated matching, (2) optimal capacity redistribution to minimize the welfare gap, and (3) cost-aware optimization balancing welfare maximization with capacity adjustment costs. Experimental validation on synthetic and real-world datasets demonstrates that our framework enables the many-to-many algorithmic recourse to achieve near-optimal welfare with minimum modification in system settings. This work extends algorithmic recourse from individual recommendations to system-level design, providing a tractable path toward higher social welfare while maintaining individual actionability. 

**Abstract (ZH)**: 多代理算法救济框架：从个体推荐到系统级设计 

---
# Grounding Rule-Based Argumentation Using Datalog 

**Title (ZH)**: 基于Datalog的规则推理论辩 grounding 

**Authors**: Martin Diller, Sarah Alice Gaggl, Philipp Hanisch, Giuseppina Monterosso, Fritz Rauschenbach  

**Link**: [PDF](https://arxiv.org/pdf/2508.10976)  

**Abstract**: ASPIC+ is one of the main general frameworks for rule-based argumentation for AI. Although first-order rules are commonly used in ASPIC+ examples, most existing approaches to reason over rule-based argumentation only support propositional rules. To enable reasoning over first-order instances, a preliminary grounding step is required. As groundings can lead to an exponential increase in the size of the input theories, intelligent procedures are needed. However, there is a lack of dedicated solutions for ASPIC+. Therefore, we propose an intelligent grounding procedure that keeps the size of the grounding manageable while preserving the correctness of the reasoning process. To this end, we translate the first-order ASPIC+ instance into a Datalog program and query a Datalog engine to obtain ground substitutions to perform the grounding of rules and contraries. Additionally, we propose simplifications specific to the ASPIC+ formalism to avoid grounding of rules that have no influence on the reasoning process. Finally, we performed an empirical evaluation of a prototypical implementation to show scalability. 

**Abstract (ZH)**: ASPIC+是一种基于规则的论证主要通用框架之一。虽然ASPIC+示例中常用一阶规则，但大多数基于规则的论证推理方法只支持命题规则。为了在一阶实例上进行推理，需要一个初步的实例化步骤。由于实例化可能导致输入理论的大小指数级增加，因此需要智能过程。然而，缺少针对ASPIC+的专用解决方案。因此，我们提出了一种智能实例化程序，该程序能够在保持实例化规模可控的同时，保持推理过程的正确性。为此，我们将一阶ASPIC+实例转换为Datalog程序，并查询Datalog引擎以获取地面置换，进行规则和反例的实例化。此外，我们针对ASPIC+形式主义提出了一种特定的简化方法，以避免实例化对推理过程没有影响的规则。最后，我们对一个原型实现进行了实证评估，以展示其实现的可扩展性。 

---
# Is ChatGPT-5 Ready for Mammogram VQA? 

**Title (ZH)**: ChatGPT-5准备好应对乳腺X光片VQA任务了吗？ 

**Authors**: Qiang Li, Shansong Wang, Mingzhe Hu, Mojtaba Safari, Zachary Eidex, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11628)  

**Abstract**: Mammogram visual question answering (VQA) integrates image interpretation with clinical reasoning and has potential to support breast cancer screening. We systematically evaluated the GPT-5 family and GPT-4o model on four public mammography datasets (EMBED, InBreast, CMMD, CBIS-DDSM) for BI-RADS assessment, abnormality detection, and malignancy classification tasks. GPT-5 consistently was the best performing model but lagged behind both human experts and domain-specific fine-tuned models. On EMBED, GPT-5 achieved the highest scores among GPT variants in density (56.8%), distortion (52.5%), mass (64.5%), calcification (63.5%), and malignancy (52.8%) classification. On InBreast, it attained 36.9% BI-RADS accuracy, 45.9% abnormality detection, and 35.0% malignancy classification. On CMMD, GPT-5 reached 32.3% abnormality detection and 55.0% malignancy accuracy. On CBIS-DDSM, it achieved 69.3% BI-RADS accuracy, 66.0% abnormality detection, and 58.2% malignancy accuracy. Compared with human expert estimations, GPT-5 exhibited lower sensitivity (63.5%) and specificity (52.3%). While GPT-5 exhibits promising capabilities for screening tasks, its performance remains insufficient for high-stakes clinical imaging applications without targeted domain adaptation and optimization. However, the tremendous improvements in performance from GPT-4o to GPT-5 show a promising trend in the potential for general large language models (LLMs) to assist with mammography VQA tasks. 

**Abstract (ZH)**: 乳腺X线摄影视觉问答（VQA）将图像解释与临床推理相结合，有望支持乳腺癌筛查。我们系统性地评估了GPT-5家族和GPT-4o模型在EMBED、InBreast、CMMD和CBIS-DDSM四个公开的乳腺X线图像数据集上的BI-RADS评估、异常检测和恶性分类任务。GPT-5在这些任务上表现稳定，但均逊于人类专家和领域特定微调模型。在EMBED数据集上，GPT-5在密度、扭曲、肿块、钙化和恶性分类上的表现最佳，分别为56.8%、52.5%、64.5%、63.5%和52.8%。在InBreast数据集上，GPT-5获得了36.9%的BI-RADS准确率、45.9%的异常检测准确率和35.0%的恶性分类准确率。在CMMD数据集上，GPT-5实现了32.3%的异常检测准确率和55.0%的恶性分类准确率。在CBIS-DDSM数据集上，GPT-5实现了69.3%的BI-RADS准确率、66.0%的异常检测准确率和58.2%的恶性分类准确率。与人类专家估计相比，GPT-5的灵敏度为63.5%，特异性为52.3%。虽然GPT-5在筛查任务上表现有潜力，但在未经目标领域适应和优化的情况下，其性能仍不足以支持高风险临床成像应用。然而，从GPT-4o到GPT-5在性能上的巨大提升显示了通用大型语言模型（LLM）在辅助乳腺X线摄影VQA任务上的潜在前景。 

---
# Controlling Multimodal LLMs via Reward-guided Decoding 

**Title (ZH)**: 通过奖励引导解码控制多模态大规模语言模型 

**Authors**: Oscar Mañas, Pierluca D'Oro, Koustuv Sinha, Adriana Romero-Soriano, Michal Drozdzal, Aishwarya Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2508.11616)  

**Abstract**: As Multimodal Large Language Models (MLLMs) gain widespread applicability, it is becoming increasingly desirable to adapt them for diverse user needs. In this paper, we study the adaptation of MLLMs through controlled decoding. To achieve this, we introduce the first method for reward-guided decoding of MLLMs and demonstrate its application in improving their visual grounding. Our method involves building reward models for visual grounding and using them to guide the MLLM's decoding process. Concretely, we build two separate reward models to independently control the degree of object precision and recall in the model's output. Our approach enables on-the-fly controllability of an MLLM's inference process in two ways: first, by giving control over the relative importance of each reward function during decoding, allowing a user to dynamically trade off object precision for recall in image captioning tasks; second, by giving control over the breadth of the search during decoding, allowing the user to control the trade-off between the amount of test-time compute and the degree of visual grounding. We evaluate our method on standard object hallucination benchmarks, showing that it provides significant controllability over MLLM inference, while consistently outperforming existing hallucination mitigation methods. 

**Abstract (ZH)**: 多模态大型语言模型的适应性研究：基于奖励引导的解码方法及其在视觉定位改进中的应用 

---
# Pretrained Conformers for Audio Fingerprinting and Retrieval 

**Title (ZH)**: 预训练Conformer模型在音频指纹提取与检索中的应用 

**Authors**: Kemal Altwlkany, Elmedin Selmanovic, Sead Delalic  

**Link**: [PDF](https://arxiv.org/pdf/2508.11609)  

**Abstract**: Conformers have shown great results in speech processing due to their ability to capture both local and global interactions. In this work, we utilize a self-supervised contrastive learning framework to train conformer-based encoders that are capable of generating unique embeddings for small segments of audio, generalizing well to previously unseen data. We achieve state-of-the-art results for audio retrieval tasks while using only 3 seconds of audio to generate embeddings. Our models are almost completely immune to temporal misalignments and achieve state-of-the-art results in cases of other audio distortions such as noise, reverb or extreme temporal stretching. Code and models are made publicly available and the results are easy to reproduce as we train and test using popular and freely available datasets of different sizes. 

**Abstract (ZH)**: 基于自监督对比学习框架的协变器在语音处理中的应用：通过短音频片段生成独特嵌入并适应多种音频失真 

---
# CryptoScope: Utilizing Large Language Models for Automated Cryptographic Logic Vulnerability Detection 

**Title (ZH)**: CryptoScope: 利用大规模语言模型进行自动化密码逻辑漏洞检测 

**Authors**: Zhihao Li, Zimo Ji, Tao Zheng, Hao Ren, Xiao Lan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11599)  

**Abstract**: Cryptographic algorithms are fundamental to modern security, yet their implementations frequently harbor subtle logic flaws that are hard to detect. We introduce CryptoScope, a novel framework for automated cryptographic vulnerability detection powered by Large Language Models (LLMs). CryptoScope combines Chain-of-Thought (CoT) prompting with Retrieval-Augmented Generation (RAG), guided by a curated cryptographic knowledge base containing over 12,000 entries. We evaluate CryptoScope on LLM-CLVA, a benchmark of 92 cases primarily derived from real-world CVE vulnerabilities, complemented by cryptographic challenges from major Capture The Flag (CTF) competitions and synthetic examples across 11 programming languages. CryptoScope consistently improves performance over strong LLM baselines, boosting DeepSeek-V3 by 11.62%, GPT-4o-mini by 20.28%, and GLM-4-Flash by 28.69%. Additionally, it identifies 9 previously undisclosed flaws in widely used open-source cryptographic projects. 

**Abstract (ZH)**: 加密算法是现代安全的基础，但其实现中常常隐藏难以检测的细微逻辑漏洞。我们介绍了一种新型框架CryptoScope，该框架利用大型语言模型（LLMs）自动检测加密漏洞。CryptoScope 结合了思维链（CoT）提示与检索增强生成（RAG），并辅以一个包含超过12,000条条目的定制化加密知识库。我们对CryptoScope进行了评估，使用LLM-CLVA基准测试，该测试包含92个案例，主要源自实际世界中的CVE漏洞，同时辅以来自主要Capture The Flag（CTF）竞标大赛的加密挑战和跨11种编程语言的合成示例。CryptoScope在性能上持续超越强大的LLM基线，分别提高了DeepSeek-V3的11.62%、GPT-4o-mini的20.28%和GLM-4-Flash的28.69%，并发现9个广泛使用的开源加密项目中的未披露漏洞。 

---
# Visual Perception Engine: Fast and Flexible Multi-Head Inference for Robotic Vision Tasks 

**Title (ZH)**: 视觉感知引擎：快速灵活的多头推理方法及其在机器人视觉任务中的应用 

**Authors**: Jakub Łucki, Jonathan Becktor, Georgios Georgakis, Robert Royce, Shehryar Khattak  

**Link**: [PDF](https://arxiv.org/pdf/2508.11584)  

**Abstract**: Deploying multiple machine learning models on resource-constrained robotic platforms for different perception tasks often results in redundant computations, large memory footprints, and complex integration challenges. In response, this work presents Visual Perception Engine (VPEngine), a modular framework designed to enable efficient GPU usage for visual multitasking while maintaining extensibility and developer accessibility. Our framework architecture leverages a shared foundation model backbone that extracts image representations, which are efficiently shared, without any unnecessary GPU-CPU memory transfers, across multiple specialized task-specific model heads running in parallel. This design eliminates the computational redundancy inherent in feature extraction component when deploying traditional sequential models while enabling dynamic task prioritization based on application demands. We demonstrate our framework's capabilities through an example implementation using DINOv2 as the foundation model with multiple task (depth, object detection and semantic segmentation) heads, achieving up to 3x speedup compared to sequential execution. Building on CUDA Multi-Process Service (MPS), VPEngine offers efficient GPU utilization and maintains a constant memory footprint while allowing per-task inference frequencies to be adjusted dynamically during runtime. The framework is written in Python and is open source with ROS2 C++ (Humble) bindings for ease of use by the robotics community across diverse robotic platforms. Our example implementation demonstrates end-to-end real-time performance at $\geq$50 Hz on NVIDIA Jetson Orin AGX for TensorRT optimized models. 

**Abstract (ZH)**: 基于受限资源机器人平台的多机器学习模型部署常常导致冗余计算、大内存占用和复杂的集成挑战。为应对这一问题，本文提出Visual Perception Engine (VPEngine)，一个模块化框架，旨在实现视觉多任务处理的高效GPU使用，同时保持扩展性和开发者的易用性。该框架架构利用共享的基础模型骨干来提取图像表示，并在多个并行运行的专业任务特定模型头部之间高效共享，无需不必要的GPU-CPU内存传输。这种设计在部署传统顺序模型时消除了特征提取部分的冗余计算问题，并允许基于应用程序需求动态调整任务优先级。通过使用DINOv2作为基础模型和多个任务（深度、物体检测和语义分割）头部的实施示例，我们展示了高达3倍的提速效果。基于CUDA Multi-Process Service (MPS)，VPEngine实现了高效的GPU使用，保持恒定的内存占用，并允许在运行时根据任务调整推理频率。该框架用Python编写，并提供ROS2 C++（Humble）绑定，便于各类机器人平台的使用。我们的示例实现展示了在NVIDIA Jetson Orin AGX上针对TensorRT优化的模型实现端到端实时性能，频率≥50 Hz。 

---
# Aware First, Think Less: Dynamic Boundary Self-Awareness Drives Extreme Reasoning Efficiency in Large Language Models 

**Title (ZH)**: 感知先思考少些：动态边界自我意识推动大规模语言模型极端高效推理 

**Authors**: Qiguang Chen, Dengyun Peng, Jinhao Liu, HuiKang Su, Jiannan Guan, Libo Qin, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2508.11582)  

**Abstract**: Recent advancements in large language models (LLMs) have greatly improved their capabilities on complex reasoning tasks through Long Chain-of-Thought (CoT). However, this approach often results in substantial redundancy, impairing computational efficiency and causing significant delays in real-time applications. To improve the efficiency, current methods often rely on human-defined difficulty priors, which do not align with the LLM's self-awared difficulty, leading to inefficiencies. In this paper, we introduce the Dynamic Reasoning-Boundary Self-Awareness Framework (DR. SAF), which enables models to dynamically assess and adjust their reasoning depth in response to problem complexity. DR. SAF integrates three key components: Boundary Self-Awareness Alignment, Adaptive Reward Management, and a Boundary Preservation Mechanism. These components allow models to optimize their reasoning processes, balancing efficiency and accuracy without compromising performance. Our experimental results demonstrate that DR. SAF achieves a 49.27% reduction in total response tokens with minimal loss in accuracy. The framework also delivers a 6.59x gain in token efficiency and a 5x reduction in training time, making it well-suited to resource-limited settings. During extreme training, DR. SAF can even surpass traditional instruction-based models in token efficiency with more than 16% accuracy improvement. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）通过长链推理（Long Chain-of-Thought，Long CoT）大幅提高了其在复杂推理任务上的能力。然而，这种方法通常会导致显著冗余，损害计算效率并造成实时应用中的重大延迟。为了提高效率，当前方法往往依赖于人工定义的难度先验，这些先验不与LLM自身的难度感知相匹配，导致效率低下。在本文中，我们引入了动态推理边界自意识框架（DR.SAF），使模型能够动态评估和调整其推理深度以应对问题的复杂性。DR.SAF集成了三个关键组成部分：边界自意识对齐、自适应奖励管理以及边界保持机制。这些组件允许模型优化其推理过程，在提高效率的同时保持准确性。实验结果表明，DR.SAF在保持最小准确性损失的情况下，将总响应 tokens减少了49.27%。此外，该框架还实现了6.59倍的 token效率提升和5倍的训练时间减少，使其适合资源受限的环境。在极端训练过程中，DR.SAF甚至在准确率提高超过16%的情况下，在token效率方面超过了传统的基于指令的模型。 

---
# ADMIRE-BayesOpt: Accelerated Data MIxture RE-weighting for Language Models with Bayesian Optimization 

**Title (ZH)**: ADMIRE-BayesOpt: 加速数据混合重新加权的语言模型中的贝叶斯优化 

**Authors**: Shengzhuang Chen, Xu Ouyang, Michael Arthur Leopold Pearce, Thomas Hartvigsen, Jonathan Richard Schwarz  

**Link**: [PDF](https://arxiv.org/pdf/2508.11551)  

**Abstract**: Determining the optimal data mixture for large language model training remains a challenging problem with an outsized impact on performance. In practice, language model developers continue to rely on heuristic exploration since no learning-based approach has emerged as a reliable solution. In this work, we propose to view the selection of training data mixtures as a black-box hyperparameter optimization problem, for which Bayesian Optimization is a well-established class of appropriate algorithms. Firstly, we cast data mixture learning as a sequential decision-making problem, in which we aim to find a suitable trade-off between the computational cost of training exploratory (proxy-) models and final mixture performance. Secondly, we systematically explore the properties of transferring mixtures learned at a small scale to larger-scale experiments, providing insights and highlighting opportunities for research at a modest scale. By proposing Multi-fidelity Bayesian Optimization as a suitable method in this common scenario, we introduce a natural framework to balance experiment cost with model fit, avoiding the risks of overfitting to smaller scales while minimizing the number of experiments at high cost. We present results for pre-training and instruction finetuning across models ranging from 1 million to 7 billion parameters, varying from simple architectures to state-of-the-art models and benchmarks spanning dozens of datasets. We demonstrate consistently strong results relative to a wide range of benchmarks, showingspeed-ups of over 500% in determining the best data mixture on our largest experiments relative to recent baselines. In addition, we broaden access to research by sharing ADMIRE IFT Runs, a dataset of 460 full training & evaluation runs across various model sizes worth over 13,000 GPU hours, greatly reducing the cost of conducting research in this area. 

**Abstract (ZH)**: 确定大规模语言模型训练的最佳数据混合比例仍然是一个具有显著影响但尚未解决的难题。实际上，语言模型开发者继续依赖启发式探索，因为尚未出现可靠的学习方法。在本文中，我们将训练数据混合的选择视为黑盒超参数优化问题，并提出使用贝叶斯优化作为合适的算法类别。首先，我们将数据混合学习视为一个顺序决策问题，旨在寻找训练探索性（代理）模型的计算成本与最终混合性能之间的合适权衡。其次，系统地探讨了从小规模实验转移混合数据到大规模实验的特性，为适度规模的研究提供了见解并揭示了研究机会。通过提议多精度贝叶斯优化作为适合该常见场景的方法，我们介绍了一个自然框架，用于平衡实验成本与模型拟合度，避免对小规模过度拟合的风险，同时将高成本实验次数降到最低。我们报告了涵盖从100万到70亿参数模型、从简单架构到最新模型和基准数据集的广泛范围的预训练和指令调优结果。在我们的最大规模实验中，确定最佳数据混合的比例比最近的基准方法快超过500%。此外，我们通过共享包含460次不同模型规模的完整训练与评估运行的ADMIRE IFT Runs数据集（超过13,000个GPU小时），扩大了对该领域的研究访问。 

---
# A Comprehensive Perspective on Explainable AI across the Machine Learning Workflow 

**Title (ZH)**: 全面视角下的可解释人工智能在整个机器学习工作流中的应用 

**Authors**: George Paterakis, Andrea Castellani, George Papoutsoglou, Tobias Rodemann, Ioannis Tsamardinos  

**Link**: [PDF](https://arxiv.org/pdf/2508.11529)  

**Abstract**: Artificial intelligence is reshaping science and industry, yet many users still regard its models as opaque "black boxes". Conventional explainable artificial-intelligence methods clarify individual predictions but overlook the upstream decisions and downstream quality checks that determine whether insights can be trusted. In this work, we present Holistic Explainable Artificial Intelligence (HXAI), a user-centric framework that embeds explanation into every stage of the data-analysis workflow and tailors those explanations to users. HXAI unifies six components (data, analysis set-up, learning process, model output, model quality, communication channel) into a single taxonomy and aligns each component with the needs of domain experts, data analysts and data scientists. A 112-item question bank covers these needs; our survey of contemporary tools highlights critical coverage gaps. Grounded in theories of human explanation, principles from human-computer interaction and findings from empirical user studies, HXAI identifies the characteristics that make explanations clear, actionable and cognitively manageable. A comprehensive taxonomy operationalises these insights, reducing terminological ambiguity and enabling rigorous coverage analysis of existing toolchains. We further demonstrate how AI agents that embed large-language models can orchestrate diverse explanation techniques, translating technical artifacts into stakeholder-specific narratives that bridge the gap between AI developers and domain experts. Departing from traditional surveys or perspective articles, this work melds concepts from multiple disciplines, lessons from real-world projects and a critical synthesis of the literature to advance a novel, end-to-end viewpoint on transparency, trustworthiness and responsible AI deployment. 

**Abstract (ZH)**: 人工智能重塑科学与产业，但许多用户仍视其模型为不透明的“黑盒”。传统的可解释人工智能方法虽能阐明个体预测，却忽视了决定洞察是否可信赖的上游决策和下游质量检查。本文提出了综合可解释人工智能（HXAI）框架，该框架以用户为中心，在数据分析工作流的每个阶段嵌入解释，并根据领域专家、数据分析师和数据科学家的需求进行定制。HXAI将六个组件（数据、分析设置、学习过程、模型输出、模型质量、沟通渠道）统一到一个分类体系中，并与这些用户的需要对齐。包含112个问题的问卷涵盖了这些需求；我们对当前工具的调研突显了关键的覆盖缺口。基于人类解释理论、人机交互原则及实证用户研究的发现，HXAI识别出使解释清晰、可操作且认知上易于管理的特征。一个全面的分类体系将这些见解具体化，减少了术语上的歧义，并使现有工具链的严格覆盖分析成为可能。我们进一步展示了嵌入大语言模型的AI代理如何协调多样的解释技术，将技术成果转化为利益相关者特定的叙述，从而弥合AI开发者与领域专家之间的差距。本文综合了来自多个学科的概念、实际项目的经验教训以及文献的批判性综合，提出了关于透明度、可靠性和负责任AI部署的端到端新视角。 

---
# Weighted First Order Model Counting for Two-variable Logic with Axioms on Two Relations 

**Title (ZH)**: 带公理的双关系两变量逻辑的加权一阶模型计数 

**Authors**: Qipeng Kuang, Václav Kůla, Ondřej Kuželka, Yuanhong Wang, Yuyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11515)  

**Abstract**: The Weighted First-Order Model Counting Problem (WFOMC) asks to compute the weighted sum of models of a given first-order logic sentence over a given domain. The boundary between fragments for which WFOMC can be computed in polynomial time relative to the domain size lies between the two-variable fragment ($\text{FO}^2$) and the three-variable fragment ($\text{FO}^3$). It is known that WFOMC for \FOthree{} is $\mathsf{\#P_1}$-hard while polynomial-time algorithms exist for computing WFOMC for $\text{FO}^2$ and $\text{C}^2$, possibly extended by certain axioms such as the linear order axiom, the acyclicity axiom, and the connectedness axiom. All existing research has concentrated on extending the fragment with axioms on a single distinguished relation, leaving a gap in understanding the complexity boundary of axioms on multiple relations. In this study, we explore the extension of the two-variable fragment by axioms on two relations, presenting both negative and positive results. We show that WFOMC for $\text{FO}^2$ with two linear order relations and $\text{FO}^2$ with two acyclic relations are $\mathsf{\#P_1}$-hard. Conversely, we provide an algorithm in time polynomial in the domain size for WFOMC of $\text{C}^2$ with a linear order relation, its successor relation and another successor relation. 

**Abstract (ZH)**: 加权一阶模型计数问题（WFOMC）要求计算给定的一阶逻辑句子在给定领域中的加权模型之和。能够在领域大小相对多项式时间内计算WFOMC的片段边界位于两变量片段（FO²）和三变量片段（FO³）之间。已知WFOMC对于FO³来说是P₁#-难的，而对于FO²和C²（可能扩展了一些特定的公理如线性序公理、无环公理和连通性公理）可以在多项式时间内计算。所有现有的研究都集中在使用针对单一特殊关系的公理扩展片段上，留下了关于涉及多个关系的公理复杂性边界的理解缺口。在这项研究中，我们探讨了使用两个关系的公理扩展两变量片段，提供了既有负面结果也有正面结果。我们证明了具有两个线性序关系的FO²和具有两个无环关系的FO²的WFOMC都是P₁#-难的。反过来，我们提供了在领域大小多项式时间内计算包含线性序关系、其后继关系和另一个后继关系的C²的WFOMC的算法。 

---
# Towards Faithful Class-level Self-explainability in Graph Neural Networks by Subgraph Dependencies 

**Title (ZH)**: 基于子图依赖的图神经网络阶层忠实自我解释研究 

**Authors**: Fanzhen Liu, Xiaoxiao Ma, Jian Yang, Alsharif Abuadbba, Kristen Moore, Surya Nepal, Cecile Paris, Quan Z. Sheng, Jia Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11513)  

**Abstract**: Enhancing the interpretability of graph neural networks (GNNs) is crucial to ensure their safe and fair deployment. Recent work has introduced self-explainable GNNs that generate explanations as part of training, improving both faithfulness and efficiency. Some of these models, such as ProtGNN and PGIB, learn class-specific prototypes, offering a potential pathway toward class-level explanations. However, their evaluations focus solely on instance-level explanations, leaving open the question of whether these prototypes meaningfully generalize across instances of the same class. In this paper, we introduce GraphOracle, a novel self-explainable GNN framework designed to generate and evaluate class-level explanations for GNNs. Our model jointly learns a GNN classifier and a set of structured, sparse subgraphs that are discriminative for each class. We propose a novel integrated training that captures graph$\unicode{x2013}$subgraph$\unicode{x2013}$prediction dependencies efficiently and faithfully, validated through a masking-based evaluation strategy. This strategy enables us to retroactively assess whether prior methods like ProtGNN and PGIB deliver effective class-level explanations. Our results show that they do not. In contrast, GraphOracle achieves superior fidelity, explainability, and scalability across a range of graph classification tasks. We further demonstrate that GraphOracle avoids the computational bottlenecks of previous methods$\unicode{x2014}$like Monte Carlo Tree Search$\unicode{x2014}$by using entropy-regularized subgraph selection and lightweight random walk extraction, enabling faster and more scalable training. These findings position GraphOracle as a practical and principled solution for faithful class-level self-explainability in GNNs. 

**Abstract (ZH)**: 增强图神经网络（GNN）的可解释性对于确保其安全和公平部署至关重要。最近的工作引入了自解释的GNN，这些模型在训练过程中生成解释，提高了忠实度和效率。其中一些模型，如ProtGNN和PGIB，学习类特定的原型，为类水平的解释提供了一种潜在途径。然而，这些模型的评估仅关注实例水平的解释，留下了这样一个问题：这些原型在相同类别的不同实例之间是否真正具有泛化能力。在本文中，我们提出了GraphOracle，一种新颖的自解释GNN框架，旨在为GNN生成和评估类水平的解释。我们的模型联合学习一个GNN分类器和一组结构化、稀疏的子图，这些子图对每个类具有鉴别作用。我们提出了一种新颖的集成训练方法，能够高效且真实地捕捉图-子图-预测的依赖关系，并通过对比蒙特卡洛树搜索等方法的掩蔽评估策略得到验证。该策略使我们能够回顾性地评估先前方法（如ProtGNN和PGIB）是否提供了有效的类水平解释。我们的结果表明，它们未能实现这一点。相比之下，GraphOracle在多种图分类任务中实现了更高的忠实度、可解释性和可扩展性。此外，我们证明GraphOracle通过使用熵正则化的子图选择和轻量级随机游走提取避免了先前方法（如蒙特卡洛树搜索）的计算瓶颈，从而实现了更快和更可扩展的训练。这些发现将GraphOracle定位为图神经网络中真实可靠的类水平自解释性的实用且原理性的解决方案。 

---
# Sim2Dust: Mastering Dynamic Waypoint Tracking on Granular Media 

**Title (ZH)**: Sim2Dust: 在颗粒介质中掌握动态途经点跟踪技能 

**Authors**: Andrej Orsula, Matthieu Geist, Miguel Olivares-Mendez, Carol Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2508.11503)  

**Abstract**: Reliable autonomous navigation across the unstructured terrains of distant planetary surfaces is a critical enabler for future space exploration. However, the deployment of learning-based controllers is hindered by the inherent sim-to-real gap, particularly for the complex dynamics of wheel interactions with granular media. This work presents a complete sim-to-real framework for developing and validating robust control policies for dynamic waypoint tracking on such challenging surfaces. We leverage massively parallel simulation to train reinforcement learning agents across a vast distribution of procedurally generated environments with randomized physics. These policies are then transferred zero-shot to a physical wheeled rover operating in a lunar-analogue facility. Our experiments systematically compare multiple reinforcement learning algorithms and action smoothing filters to identify the most effective combinations for real-world deployment. Crucially, we provide strong empirical evidence that agents trained with procedural diversity achieve superior zero-shot performance compared to those trained on static scenarios. We also analyze the trade-offs of fine-tuning with high-fidelity particle physics, which offers minor gains in low-speed precision at a significant computational cost. Together, these contributions establish a validated workflow for creating reliable learning-based navigation systems, marking a critical step towards deploying autonomous robots in the final frontier. 

**Abstract (ZH)**: 可靠的自主导航穿越遥远行星表面的未结构化地形是未来空间探索的关键使能器。然而，基于学习的控制器控制器控制器控制器的部署受到固有的仿真到到到现实差距的阻碍，尤其是对于复杂的动力学交互，特别是在与颗粒介质的交互中。本文提出了一种完整的从仿真到到现实的框架，用于在如此具有挑战性的表面上开发和验证动态航路点跟踪的稳健控制策略。该框架基于大量并行仿真来训练强化学习代理，并在使用程序生成环境的分布上随机物理参数。然后，这些策略在月球类比设施中的实际轮式 rover 上进行了零样本迁移。我们系统研究了多种强化学习算法和动作平滑滤波器，以确定最适合实际部署的组合。 crucially 我们提供了强有力的实证证据，证明在程序多样性上上 场景上 上训练的代理 Zero-shot �態現表现优于仅在少数特定场景上 培训的代理。我们还 还分析了与高保真度 physics 在上的微调的权衡，这提供了在高速度精度上的的微小收益，但需要大量计算资源。我们的贡献一起建立了一种验证的流程框架，用于创建可靠的的基于学习的系统，朝着部署自主机器人在新兴领域迈出了关键一步。 

---
# Handwritten Text Recognition of Historical Manuscripts Using Transformer-Based Models 

**Title (ZH)**: 基于变压器模型的历史手稿手写文本识别 

**Authors**: Erez Meoded  

**Link**: [PDF](https://arxiv.org/pdf/2508.11499)  

**Abstract**: Historical handwritten text recognition (HTR) is essential for unlocking the cultural and scholarly value of archival documents, yet digitization is often hindered by scarce transcriptions, linguistic variation, and highly diverse handwriting styles. In this study, we apply TrOCR, a state-of-the-art transformer-based HTR model, to 16th-century Latin manuscripts authored by Rudolf Gwalther. We investigate targeted image preprocessing and a broad suite of data augmentation techniques, introducing four novel augmentation methods designed specifically for historical handwriting characteristics. We also evaluate ensemble learning approaches to leverage the complementary strengths of augmentation-trained models. On the Gwalther dataset, our best single-model augmentation (Elastic) achieves a Character Error Rate (CER) of 1.86, while a top-5 voting ensemble achieves a CER of 1.60 - representing a 50% relative improvement over the best reported TrOCR_BASE result and a 42% improvement over the previous state of the art. These results highlight the impact of domain-specific augmentations and ensemble strategies in advancing HTR performance for historical manuscripts. 

**Abstract (ZH)**: 的历史手写文本识别（HTR）对于解锁档案文件的文化和学术价值至关重要，但由于缺乏转录、语言变异和高度多样的手写风格，数字化往往受到阻碍。本研究将最先进的变压器基线HTR模型TrOCR应用于鲁道夫·格瓦尔瑟撰写的16世纪拉丁手稿。我们研究了针对特定图像预处理方法和一系列广泛的数据增强技术，引入了四种专为历史手写特征设计的新型增强方法。我们还评估了集成学习方法以充分利用增强训练模型的互补优势。在格瓦尔瑟数据集中，我们最佳单一模型增强（Elastic）的字符错误率（CER）为1.86，而前五票投票集成则达到CER 1.60，分别比最佳报告的TrOCR_BASE结果提高了50%，比之前最先进的技术提高了42%。这些结果突显了领域特定增强和集成策略对提高历史手稿HTR性能的影响。 

---
# RMSL: Weakly-Supervised Insider Threat Detection with Robust Multi-sphere Learning 

**Title (ZH)**: RMSL：稳健多球学习驱动的弱监督内部威胁检测 

**Authors**: Yang Wang, Yaxin Zhao, Xinyu Jiao, Sihan Xu, Xiangrui Cai, Ying Zhang, Xiaojie Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11472)  

**Abstract**: Insider threat detection aims to identify malicious user behavior by analyzing logs that record user interactions. Due to the lack of fine-grained behavior-level annotations, detecting specific behavior-level anomalies within user behavior sequences is challenging. Unsupervised methods face high false positive rates and miss rates due to the inherent ambiguity between normal and anomalous behaviors. In this work, we instead introduce weak labels of behavior sequences, which have lower annotation costs, i.e., the training labels (anomalous or normal) are at sequence-level instead of behavior-level, to enhance the detection capability for behavior-level anomalies by learning discriminative features. To achieve this, we propose a novel framework called Robust Multi-sphere Learning (RMSL). RMSL uses multiple hyper-spheres to represent the normal patterns of behaviors. Initially, a one-class classifier is constructed as a good anomaly-supervision-free starting point. Building on this, using multiple instance learning and adaptive behavior-level self-training debiasing based on model prediction confidence, the framework further refines hyper-spheres and feature representations using weak sequence-level labels. This approach enhances the model's ability to distinguish between normal and anomalous behaviors. Extensive experiments demonstrate that RMSL significantly improves the performance of behavior-level insider threat detection. 

**Abstract (ZH)**: 基于弱标签的稳健多球学习方法在用户行为序列中检测行为级异常以识别内部威胁 

---
# Reference Points in LLM Sentiment Analysis: The Role of Structured Context 

**Title (ZH)**: LLM情感分析中的参考点：结构化上下文的作用 

**Authors**: Junichiro Niimi  

**Link**: [PDF](https://arxiv.org/pdf/2508.11454)  

**Abstract**: Large language models (LLMs) are now widely used across many fields, including marketing research. Sentiment analysis, in particular, helps firms understand consumer preferences. While most NLP studies classify sentiment from review text alone, marketing theories, such as prospect theory and expectation--disconfirmation theory, point out that customer evaluations are shaped not only by the actual experience but also by additional reference points. This study therefore investigates how the content and format of such supplementary information affect sentiment analysis using LLMs. We compare natural language (NL) and JSON-formatted prompts using a lightweight 3B parameter model suitable for practical marketing applications. Experiments on two Yelp categories (Restaurant and Nightlife) show that the JSON prompt with additional information outperforms all baselines without fine-tuning: Macro-F1 rises by 1.6% and 4% while RMSE falls by 16% and 9.1%, respectively, making it deployable in resource-constrained edge devices. Furthermore, a follow-up analysis confirms that performance gains stem from genuine contextual reasoning rather than label proxying. This work demonstrates that structured prompting can enable smaller models to achieve competitive performance, offering a practical alternative to large-scale model deployment. 

**Abstract (ZH)**: 大型语言模型中的补充信息如何影响基于LLM的情感分析：以Yelp类别为例 

---
# Inside Knowledge: Graph-based Path Generation with Explainable Data Augmentation and Curriculum Learning for Visual Indoor Navigation 

**Title (ZH)**: 基于图的路径生成：具有可解释数据增强和渐增学习的视觉室内导航内知识方法 

**Authors**: Daniel Airinei, Elena Burceanu, Marius Leordeanu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11446)  

**Abstract**: Indoor navigation is a difficult task, as it generally comes with poor GPS access, forcing solutions to rely on other sources of information. While significant progress continues to be made in this area, deployment to production applications is still lacking, given the complexity and additional requirements of current solutions. Here, we introduce an efficient, real-time and easily deployable deep learning approach, based on visual input only, that can predict the direction towards a target from images captured by a mobile device. Our technical approach, based on a novel graph-based path generation method, combined with explainable data augmentation and curriculum learning, includes contributions that make the process of data collection, annotation and training, as automatic as possible, efficient and robust. On the practical side, we introduce a novel largescale dataset, with video footage inside a relatively large shopping mall, in which each frame is annotated with the correct next direction towards different specific target destinations. Different from current methods, ours relies solely on vision, avoiding the need of special sensors, additional markers placed along the path, knowledge of the scene map or internet access. We also created an easy to use application for Android, which we plan to make publicly available. We make all our data and code available along with visual demos on our project site 

**Abstract (ZH)**: 室内导航是一个具有挑战性的任务，通常受限于较差的GPS访问，因此解决方案往往会依赖其他信息源。尽管在这个领域持续取得了显著进展，但由于当前解决方案的复杂性和额外要求，将其部署到实际应用中仍然不足。在这里，我们提出了一种高效、实时且易于部署的基于视觉输入的深度学习方法，该方法能够仅从移动设备拍摄的图像中预测目标方向。我们的技术方法基于一种新颖的基于图的路径生成方法，结合可解释的数据增强和层次学习，包括使数据收集、标注和训练过程尽可能自动化、高效和稳健的贡献。在实际应用方面，我们引入了一个大规模的新型数据集，其中包含在相对大型购物商场内部的视频片段，每帧都标注了正确的下一个目标方向。与现有方法不同，我们的方法仅依赖视觉信息，避免了使用特殊传感器、路径上的额外标记、场景地图知识或互联网接入的需求。我们还创建了一个易于使用的Android应用程序，并计划将其公开发布。我们将在项目网站上提供所有数据、代码和视觉演示。 

---
# Informative Post-Hoc Explanations Only Exist for Simple Functions 

**Title (ZH)**: 仅存在简单的函数上的有说服力的后验解释。 

**Authors**: Eric Günther, Balázs Szabados, Robi Bhattacharjee, Sebastian Bordt, Ulrike von Luxburg  

**Link**: [PDF](https://arxiv.org/pdf/2508.11441)  

**Abstract**: Many researchers have suggested that local post-hoc explanation algorithms can be used to gain insights into the behavior of complex machine learning models. However, theoretical guarantees about such algorithms only exist for simple decision functions, and it is unclear whether and under which assumptions similar results might exist for complex models. In this paper, we introduce a general, learning-theory-based framework for what it means for an explanation to provide information about a decision function. We call an explanation informative if it serves to reduce the complexity of the space of plausible decision functions. With this approach, we show that many popular explanation algorithms are not informative when applied to complex decision functions, providing a rigorous mathematical rejection of the idea that it should be possible to explain any model. We then derive conditions under which different explanation algorithms become informative. These are often stronger than what one might expect. For example, gradient explanations and counterfactual explanations are non-informative with respect to the space of differentiable functions, and SHAP and anchor explanations are not informative with respect to the space of decision trees. Based on these results, we discuss how explanation algorithms can be modified to become informative. While the proposed analysis of explanation algorithms is mathematical, we argue that it holds strong implications for the practical applicability of these algorithms, particularly for auditing, regulation, and high-risk applications of AI. 

**Abstract (ZH)**: 一种基于学习理论的解释框架：关于决策函数信息性的新视角 

---
# On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting 

**Title (ZH)**: 基于策略 RL 结合离策略专家：动态权重调控的监督微调与强化学习和谐融合 

**Authors**: Wenhao Zhang, Yuexiang Xie, Yuchang Sun, Yanxi Chen, Guoyin Wang, Yaliang Li, Bolin Ding, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.11408)  

**Abstract**: Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) are two prominent post-training paradigms for refining the capabilities and aligning the behavior of Large Language Models (LLMs). Existing approaches that integrate SFT and RL often face the risk of disrupting established model patterns and inducing overfitting to expert data. To address this, we present a novel investigation into the unified view of SFT and RL through an off-policy versus on-policy lens. We propose CHORD, a framework for the Controllable Harmonization of On- and Off-Policy Reinforcement Learning via Dynamic Weighting, which reframes SFT not as a separate stage but as a dynamically weighted auxiliary objective within the on-policy RL process. Based on an analysis of off-policy expert data's influence at both holistic and granular levels, we incorporate a dual-control mechanism in CHORD. Specifically, the framework first employs a global coefficient to holistically guide the transition from off-policy imitation to on-policy exploration, and then applies a token-wise weighting function that enables granular learning from expert tokens, which preserves on-policy exploration and mitigates disruption from off-policy data. We conduct extensive experiments on widely used benchmarks, providing empirical evidence that CHORD achieves a stable and efficient learning process. By effectively harmonizing off-policy expert data with on-policy exploration, CHORD demonstrates significant improvements over baselines. We release the implementation at this https URL to inspire further research. 

**Abstract (ZH)**: 监督微调（SFT）和强化学习（RL）是两种重要的后训练范式，用于 refinement 大型语言模型（LLMs）的能力和行为对齐。现有将 SFT 和 RL 结合的方法往往面临破坏模型稳定性和过度拟合专家数据的风险。为了解决这一问题，我们提出了一种通过 off-policy 和 on-policy 视角统一 SFT 和 RL 的新颖研究。我们提出了 CHORD 框架，通过动态加权实现可控的 off- 和 on-policy 强化学习谐调，将 SFT 不视为单独的阶段，而是作为 on-policy RL 过程中的动态加权辅助目标。基于 off-policy 专家数据在整体和细粒度层面的影响分析，我们在 CHORD 中引入了一种双控制机制。具体地，框架首先使用全局系数在整体上引导 off-policy 仿真的过渡到 on-policy 探索，然后应用一个按词加权函数，使模型能够从专家词中进行细粒度学习，从而保留 on-policy 探索并降低 off-policy 数据的干扰。我们在广泛使用的基准上进行了广泛的实验，提供了实证证据表明 CHORD 能够实现稳定和高效的训练过程。通过有效调和 off-policy 专家数据与 on-policy 探索，CHORD 在基线上显示出显著的改进。我们在此 https:// 放开源代码以启发 further 研究。 

---
# Open, Reproducible and Trustworthy Robot-Based Experiments with Virtual Labs and Digital-Twin-Based Execution Tracing 

**Title (ZH)**: 基于虚拟实验室和基于数字孪生的执行追踪的开放、可再现和可信赖的机器人实验 

**Authors**: Benjamin Alt, Mareike Picklum, Sorin Arion, Franklin Kenghagho Kenfack, Michael Beetz  

**Link**: [PDF](https://arxiv.org/pdf/2508.11406)  

**Abstract**: We envision a future in which autonomous robots conduct scientific experiments in ways that are not only precise and repeatable, but also open, trustworthy, and transparent. To realize this vision, we present two key contributions: a semantic execution tracing framework that logs sensor data together with semantically annotated robot belief states, ensuring that automated experimentation is transparent and replicable; and the AICOR Virtual Research Building (VRB), a cloud-based platform for sharing, replicating, and validating robot task executions at scale. Together, these tools enable reproducible, robot-driven science by integrating deterministic execution, semantic memory, and open knowledge representation, laying the foundation for autonomous systems to participate in scientific discovery. 

**Abstract (ZH)**: 我们设想一个未来，在自主机器人将以不仅精确可 

---
# An Exploratory Study on Crack Detection in Concrete through Human-Robot Collaboration 

**Title (ZH)**: 通过人机协作的混凝土裂缝检测探索性研究 

**Authors**: Junyeon Kim, Tianshu Ruan, Cesar Alan Contreras, Manolis Chiou  

**Link**: [PDF](https://arxiv.org/pdf/2508.11404)  

**Abstract**: Structural inspection in nuclear facilities is vital for maintaining operational safety and integrity. Traditional methods of manual inspection pose significant challenges, including safety risks, high cognitive demands, and potential inaccuracies due to human limitations. Recent advancements in Artificial Intelligence (AI) and robotic technologies have opened new possibilities for safer, more efficient, and accurate inspection methodologies. Specifically, Human-Robot Collaboration (HRC), leveraging robotic platforms equipped with advanced detection algorithms, promises significant improvements in inspection outcomes and reductions in human workload. This study explores the effectiveness of AI-assisted visual crack detection integrated into a mobile Jackal robot platform. The experiment results indicate that HRC enhances inspection accuracy and reduces operator workload, resulting in potential superior performance outcomes compared to traditional manual methods. 

**Abstract (ZH)**: 核设施结构检查对于维持运营安全和完整性至关重要。传统的手动检查方法面临显著挑战，包括安全风险、高认知负荷以及由于人类限制可能导致的不准确性。最近在人工智能（AI）和机器人技术方面的进步为更安全、更高效和更准确的检查方法打开了新的可能性。特别是通过利用装有先进检测算法的机器人平台实现的人机协作（HRC），有望在检查结果和减少人力工作量方面取得显著改善。本研究探讨了将AI辅助的裂纹检测集成到移动Jackal机器人平台中的有效性。实验结果表明，HRC能够提高检查准确性并减轻操作员的工作负担，从而可能在性能方面超过传统的手动方法。 

---
# Trustworthy AI Psychotherapy: Multi-Agent LLM Workflow for Counseling and Explainable Mental Disorder Diagnosis 

**Title (ZH)**: 可信AI心理治疗：多Agent大型语言模型工作流在咨询与可解释的精神障碍诊断中的应用 

**Authors**: Mithat Can Ozgun, Jiahuan Pei, Koen Hindriks, Lucia Donatelli, Qingzhi Liu, Xin Sun, Junxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11398)  

**Abstract**: LLM-based agents have emerged as transformative tools capable of executing complex tasks through iterative planning and action, achieving significant advancements in understanding and addressing user needs. Yet, their effectiveness remains limited in specialized domains such as mental health diagnosis, where they underperform compared to general applications. Current approaches to integrating diagnostic capabilities into LLMs rely on scarce, highly sensitive mental health datasets, which are challenging to acquire. These methods also fail to emulate clinicians' proactive inquiry skills, lack multi-turn conversational comprehension, and struggle to align outputs with expert clinical reasoning. To address these gaps, we propose DSM5AgentFlow, the first LLM-based agent workflow designed to autonomously generate DSM-5 Level-1 diagnostic questionnaires. By simulating therapist-client dialogues with specific client profiles, the framework delivers transparent, step-by-step disorder predictions, producing explainable and trustworthy results. This workflow serves as a complementary tool for mental health diagnosis, ensuring adherence to ethical and legal standards. Through comprehensive experiments, we evaluate leading LLMs across three critical dimensions: conversational realism, diagnostic accuracy, and explainability. Our datasets and implementations are fully open-sourced. 

**Abstract (ZH)**: 基于LLM的代理在通过迭代规划和行动执行复杂任务方面 emerged as transformative tools capable of achieving significant advancements in understanding and addressing user needs. Yet, their effectiveness remains limited in specialized domains such as mental health diagnosis, where they underperform compared to general applications. Current approaches to integrating diagnostic capabilities into LLMs rely on scarce, highly sensitive mental health datasets, which are challenging to acquire. These methods also fail to emulate clinicians' proactive inquiry skills, lack multi-turn conversational comprehension, and struggle to align outputs with expert clinical reasoning. To address these gaps, we propose DSM5AgentFlow, the first LLM-based agent workflow designed to autonomously generate DSM-5 Level-1 diagnostic questionnaires. By simulating therapist-client dialogues with specific client profiles, the framework delivers transparent, step-by-step disorder predictions, producing explainable and trustworthy results. This workflow serves as a complementary tool for mental health diagnosis, ensuring adherence to ethical and legal standards. Through comprehensive experiments, we evaluate leading LLMs across three critical dimensions: conversational realism, diagnostic accuracy, and explainability. Our datasets and implementations are fully open-sourced.

基于LLM的代理在通过迭代规划和行动执行复杂任务方面取得了革命性进展，实现了对用户需求的理解和解决的显著进步。然而，在心理健康诊断等专业领域，它们的效果逊于通用应用。目前将诊断能力集成到LLM中的方法依赖于稀缺、高度敏感的心理健康数据集，获取这些数据极具挑战性。这些方法也未能模拟临床医生的主动探索技能，缺乏多轮对话理解能力，难以使输出与专家临床推理对齐。为了弥补这些差距，我们提出了DSM5AgentFlow，这是第一个基于LLM的代理工作流，旨在自主生成DSM-5 Level-1诊断问卷。通过模拟具有特定客户档案的治疗师-客户对话，该框架提供了透明的、逐步的障碍预测，生成可解释和可信赖的结果。该工作流作为心理健康诊断的一种辅助工具，确保遵守伦理和法律标准。通过全面的实验，我们从三个关键维度评估了领先的人工智能模型：对话逼真性、诊断准确性以及可解释性。我们的数据集和实现均已完全开源。 

---
# Retrieval-augmented reasoning with lean language models 

**Title (ZH)**: 基于精简语言模型的检索增强推理 

**Authors**: Ryan Sze-Yin Chan, Federico Nanni, Tomas Lazauskas, Rosie Wood, Penelope Yong, Lionel Tarassenko, Mark Girolami, James Geddes, Andrew Duncan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11386)  

**Abstract**: This technical report details a novel approach to combining reasoning and retrieval augmented generation (RAG) within a single, lean language model architecture. While existing RAG systems typically rely on large-scale models and external APIs, our work addresses the increasing demand for performant and privacy-preserving solutions deployable in resource-constrained or secure environments. Building on recent developments in test-time scaling and small-scale reasoning models, we develop a retrieval augmented conversational agent capable of interpreting complex, domain-specific queries using a lightweight backbone model. Our system integrates a dense retriever with fine-tuned Qwen2.5-Instruct models, using synthetic query generation and reasoning traces derived from frontier models (e.g., DeepSeek-R1) over a curated corpus, in this case, the NHS A-to-Z condition pages. We explore the impact of summarisation-based document compression, synthetic data design, and reasoning-aware fine-tuning on model performance. Evaluation against both non-reasoning and general-purpose lean models demonstrates that our domain-specific fine-tuning approach yields substantial gains in answer accuracy and consistency, approaching frontier-level performance while remaining feasible for local deployment. All implementation details and code are publicly released to support reproducibility and adaptation across domains. 

**Abstract (ZH)**: 本技术报告详细介绍了在一个精简的语言模型架构中结合推理和检索增强生成（RAG）的新方法。尽管现有的RAG系统通常依赖大型模型和外部API，我们的工作致力于满足对高性能且保护隐私的解决方案的需求，这些解决方案可以在资源受限或安全环境中部署。基于最近在测试时扩展和小型推理模型发展方面的进展，我们开发了一个检索增强的对话代理，能够使用轻量级的基础模型解释复杂的、领域特定的查询。我们的系统将稠密检索器与细调的Qwen2.5-Instruct模型结合使用，利用来自前沿模型（如DeepSeek-R1）的合成查询生成和推理轨迹对精选语料库（例如，NHS A-to-Z病症页面）进行加权。我们探讨了基于总结的文档压缩、合成数据设计以及推理意识微调对模型性能的影响。与其他非推理和通用精简模型的评估表明，我们针对特定领域的微调方法在答案的准确性和一致性方面取得了显著提升，同时保持了在本地部署方面的可行性。所有实施细节和代码均已公开发布，以支持可重复性和跨领域的适应性。 

---
# When Punctuation Matters: A Large-Scale Comparison of Prompt Robustness Methods for LLMs 

**Title (ZH)**: 标点符号 Matters：大规模比较 LLMs 的提示稳健性方法 

**Authors**: Mikhail Seleznyov, Mikhail Chaichuk, Gleb Ershov, Alexander Panchenko, Elena Tutubalina, Oleg Somov  

**Link**: [PDF](https://arxiv.org/pdf/2508.11383)  

**Abstract**: Large Language Models (LLMs) are highly sensitive to subtle, non-semantic variations in prompt phrasing and formatting. In this work, we present the first systematic evaluation of 5 methods for improving prompt robustness within a unified experimental framework. We benchmark these techniques on 8 models from Llama, Qwen and Gemma families across 52 tasks from Natural Instructions dataset. Our evaluation covers robustness methods from both fine-tuned and in-context learning paradigms, and tests their generalization against multiple types of distribution shifts. Finally, we extend our analysis to GPT-4.1 and DeepSeek V3 to assess frontier models' current robustness to format perturbations. Our findings offer actionable insights into the relative effectiveness of these robustness methods, enabling practitioners to make informed decisions when aiming for stable and reliable LLM performance in real-world applications. Code: this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）对提示语句措辞和格式中的细微、非语义变化极为敏感。在本文中，我们提出了一种统一实验框架下的首次系统性评估，评测了5种提高提示鲁棒性的方法。我们在来自Llama、Qwen和Gemma家族的8个模型上，使用来自Natural Instructions数据集的52个任务进行了基准测试。我们的评估涵盖了调优和上下文学习范式下的鲁棒性方法，并测试了它们在多种分布偏移类型下的泛化能力。最后，我们将分析扩展到GPT-4.1和DeepSeek V3，以评估前沿模型对格式扰动的当前鲁棒性。我们的研究提供了这些鲁棒性方法相对有效性的实用见解，帮助实践者在实际应用中实现稳定可靠的LLM性能。代码：this https URL。 

---
# G-CUT3R: Guided 3D Reconstruction with Camera and Depth Prior Integration 

**Title (ZH)**: G-CUT3R：带有相机和深度先验集成的引导三维重建 

**Authors**: Ramil Khafizov, Artem Komarichev, Ruslan Rakhimov, Peter Wonka, Evgeny Burnaev  

**Link**: [PDF](https://arxiv.org/pdf/2508.11379)  

**Abstract**: We introduce G-CUT3R, a novel feed-forward approach for guided 3D scene reconstruction that enhances the CUT3R model by integrating prior information. Unlike existing feed-forward methods that rely solely on input images, our method leverages auxiliary data, such as depth, camera calibrations, or camera positions, commonly available in real-world scenarios. We propose a lightweight modification to CUT3R, incorporating a dedicated encoder for each modality to extract features, which are fused with RGB image tokens via zero convolution. This flexible design enables seamless integration of any combination of prior information during inference. Evaluated across multiple benchmarks, including 3D reconstruction and other multi-view tasks, our approach demonstrates significant performance improvements, showing its ability to effectively utilize available priors while maintaining compatibility with varying input modalities. 

**Abstract (ZH)**: G-CUT3R：一种通过集成先验信息增强的新型前向指导三维场景重建方法 

---
# Does the Skeleton-Recall Loss Really Work? 

**Title (ZH)**: 骨架召回损失真的有效吗？ 

**Authors**: Devansh Arora, Nitin Kumar, Sukrit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.11374)  

**Abstract**: Image segmentation is an important and widely performed task in computer vision. Accomplishing effective image segmentation in diverse settings often requires custom model architectures and loss functions. A set of models that specialize in segmenting thin tubular structures are topology preservation-based loss functions. These models often utilize a pixel skeletonization process claimed to generate more precise segmentation masks of thin tubes and better capture the structures that other models often miss. One such model, Skeleton Recall Loss (SRL) proposed by Kirchhoff et al.~\cite {kirchhoff2024srl}, was stated to produce state-of-the-art results on benchmark tubular datasets. In this work, we performed a theoretical analysis of the gradients for the SRL loss. Upon comparing the performance of the proposed method on some of the tubular datasets (used in the original work, along with some additional datasets), we found that the performance of SRL-based segmentation models did not exceed traditional baseline models. By providing both a theoretical explanation and empirical evidence, this work critically evaluates the limitations of topology-based loss functions, offering valuable insights for researchers aiming to develop more effective segmentation models for complex tubular structures. 

**Abstract (ZH)**: 基于拓扑保护断细的图像分割：Skeleton Recall Loss (SRL)的理论分析与评估 

---
# Minimizing Surrogate Losses for Decision-Focused Learning using Differentiable Optimization 

**Title (ZH)**: 最小化决策导向学习的代理损失函数优化 

**Authors**: Jayanta Mandi, Ali İrfan Mahmutoğulları, Senne Berden, Tias Guns  

**Link**: [PDF](https://arxiv.org/pdf/2508.11365)  

**Abstract**: Decision-focused learning (DFL) trains a machine learning (ML) model to predict parameters of an optimization problem, to directly minimize decision regret, i.e., maximize decision quality. Gradient-based DFL requires computing the derivative of the solution to the optimization problem with respect to the predicted parameters. However, for many optimization problems, such as linear programs (LPs), the gradient of the regret with respect to the predicted parameters is zero almost everywhere. Existing gradient-based DFL approaches for LPs try to circumvent this issue in one of two ways: (a) smoothing the LP into a differentiable optimization problem by adding a quadratic regularizer and then minimizing the regret directly or (b) minimizing surrogate losses that have informative (sub)gradients. In this paper, we show that the former approach still results in zero gradients, because even after smoothing the regret remains constant across large regions of the parameter space. To address this, we propose minimizing surrogate losses -- even when a differentiable optimization layer is used and regret can be minimized directly. Our experiments demonstrate that minimizing surrogate losses allows differentiable optimization layers to achieve regret comparable to or better than surrogate-loss based DFL methods. Further, we demonstrate that this also holds for DYS-Net, a recently proposed differentiable optimization technique for LPs, that computes approximate solutions and gradients through operations that can be performed using feedforward neural network layers. Because DYS-Net executes the forward and the backward pass very efficiently, by minimizing surrogate losses using DYS-Net, we are able to attain regret on par with the state-of-the-art while reducing training time by a significant margin. 

**Abstract (ZH)**: 决策导向的学习（DFL）训练机器学习（ML）模型以预测优化问题的参数，直接最小化决策遗憾，即最大化决策质量。基于梯度的DFL需要计算优化问题的解关于预测参数的导数。然而，对于许多优化问题，如线性规划（LPs），遗憾关于预测参数的梯度几乎处处为零。现有基于梯度的DFL方法针对LPs试图通过两种方式之一来克服这一问题：（a）通过添加二次正则化项将LP平滑为可微优化问题，然后直接最小化遗憾，或（b）最小化具有信息性（次）梯度的替代损失函数。在这篇文章中，我们展示了第一种方法仍然会导致梯度为零，因为在平滑后，遗憾在参数空间的大量区域内仍然是恒定的。为此，我们提出即使使用可微优化层且可以直接最小化遗憾时，仍最小化替代损失函数。我们的实验表明，最小化替代损失函数使可微优化层能够实现与基于替代损失的DFL方法相当甚至更好的遗憾结果。此外，我们展示了这一点也适用于DYS-Net，这是一种最近提出的LP可微优化技术，通过可以使用前向神经网络层执行的操作计算近似解和梯度。由于DYS-Net非常高效地执行前向和反向传递，通过使用DYS-Net最小化替代损失函数，我们能够在大幅减少训练时间的同时达到与当前最佳方法相当的遗憾结果。 

---
# PTSM: Physiology-aware and Task-invariant Spatio-temporal Modeling for Cross-Subject EEG Decoding 

**Title (ZH)**: PTSM：生理aware且任务不变的时空建模用于跨被试EEG解码 

**Authors**: Changhong Jing, Yan Liu, Shuqiang Wang, Bruce X.B. Yu, Gong Chen, Zhejing Hu, Zhi Zhang, Yanyan Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11357)  

**Abstract**: Cross-subject electroencephalography (EEG) decoding remains a fundamental challenge in brain-computer interface (BCI) research due to substantial inter-subject variability and the scarcity of subject-invariant representations. This paper proposed PTSM (Physiology-aware and Task-invariant Spatio-temporal Modeling), a novel framework for interpretable and robust EEG decoding across unseen subjects. PTSM employs a dual-branch masking mechanism that independently learns personalized and shared spatio-temporal patterns, enabling the model to preserve individual-specific neural characteristics while extracting task-relevant, population-shared features. The masks are factorized across temporal and spatial dimensions, allowing fine-grained modulation of dynamic EEG patterns with low computational overhead. To further address representational entanglement, PTSM enforces information-theoretic constraints that decompose latent embeddings into orthogonal task-related and subject-related subspaces. The model is trained end-to-end via a multi-objective loss integrating classification, contrastive, and disentanglement objectives. Extensive experiments on cross-subject motor imagery datasets demonstrate that PTSM achieves strong zero-shot generalization, outperforming state-of-the-art baselines without subject-specific calibration. Results highlight the efficacy of disentangled neural representations for achieving both personalized and transferable decoding in non-stationary neurophysiological settings. 

**Abstract (ZH)**: 生理aware和任务不变时空建模：跨受试者脑电图（EEG）解码的新框架 

---
# ETTRL: Balancing Exploration and Exploitation in LLM Test-Time Reinforcement Learning Via Entropy Mechanism 

**Title (ZH)**: ETTRL：通过熵机制在LLM测试时强化学习中平衡探索与利用 

**Authors**: Jia Liu, ChangYi He, YingQiao Lin, MingMin Yang, FeiYang Shen, ShaoGuo Liu, TingTing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11356)  

**Abstract**: Recent advancements in Large Language Models have yielded significant improvements in complex reasoning tasks such as mathematics and programming. However, these models remain heavily dependent on annotated data and exhibit limited adaptability in unsupervised scenarios. To address these limitations, test-time reinforcement learning (TTRL) has been proposed, which enables self-optimization by leveraging model-generated pseudo-labels. Despite its promise, TTRL faces several key challenges, including high inference costs due to parallel rollouts and early-stage estimation bias that fosters overconfidence, reducing output diversity and causing performance plateaus. To address these challenges, we introduce an entropy-based mechanism to enhance the exploration-exploitation balance in test-time reinforcement learning through two strategies: Entropy-fork Tree Majority Rollout (ETMR) and Entropy-based Advantage Reshaping (EAR). Compared with the baseline, our approach enables Llama3.1-8B to achieve a 68 percent relative improvement in Pass at 1 metric on the AIME 2024 benchmark, while consuming only 60 percent of the rollout tokens budget. This highlights our method's ability to effectively optimize the trade-off between inference efficiency, diversity, and estimation robustness, thereby advancing unsupervised reinforcement learning for open-domain reasoning tasks. 

**Abstract (ZH)**: 近年来，大型语言模型的进展在复杂推理任务（如数学和编程等方面取得了显著的改进。然而，这些模型仍然高度依赖于标注数据，并在无监督场景中表现出局限性。为了应对这些限制，，我们提出了在训练时强化学习（建议-time reinforcement learning，）中的混合策略策略方法（（-Time Reinforcement Learning with Reset-Based Mechanism,）（简称TTRL），该方法通过利用生成的伪标签增强了了优化过程。尽管TTRL展现出前景，，仍面临几项挑战，如包括推理成本由于平行并并的并行阶段的偏差增加使得自信度增加以及输出多样性减少和性能 plateau呈现等。为了应对这些挑战，我们引入了一种基于熵的机制来增强在训练时强化学习中的探索--exploitation利用平衡通过两个机制：熵门限树众数机制（Entropy-F Tree Majority Mechanism，，简称ETMR）和基于熵的优势估计机制（Entropy-Based Advantage Ententin Mechanon，，）简称EAR）。与基于熵的Llama3-8ient--8B相比，我们的方法在PASet on--entity测量上的AIME on2-4基准上上实现了68%ent的推理令牌预算上改进了68 onentonent的Ponentent- eonanententon。这突显了我们an on的能力来有效地在推理效率、多样性以及稳定性之间实现 ean opte enton，从而推进无监督强化学习在跨域任务上 on域任务方面的应用情应用。 

---
# Leveraging the RETFound foundation model for optic disc segmentation in retinal images 

**Title (ZH)**: 基于RETFound基础模型的眼底图像视神经盘分割方法 

**Authors**: Zhenyi Zhao, Muthu Rama Krishnan Mookiah, Emanuele Trucco  

**Link**: [PDF](https://arxiv.org/pdf/2508.11354)  

**Abstract**: RETFound is a well-known foundation model (FM) developed for fundus camera and optical coherence tomography images. It has shown promising performance across multiple datasets in diagnosing diseases, both eye-specific and systemic, from retinal images. However, to our best knowledge, it has not been used for other tasks. We present the first adaptation of RETFound for optic disc segmentation, a ubiquitous and foundational task in retinal image analysis. The resulting segmentation system outperforms state-of-the-art, segmentation-specific baseline networks after training a head with only a very modest number of task-specific examples. We report and discuss results with four public datasets, IDRID, Drishti-GS, RIM-ONE-r3, and REFUGE, and a private dataset, GoDARTS, achieving about 96% Dice consistently across all datasets. Overall, our method obtains excellent performance in internal verification, domain generalization and domain adaptation, and exceeds most of the state-of-the-art baseline results. We discuss the results in the framework of the debate about FMs as alternatives to task-specific architectures. The code is available at: [link to be added after the paper is accepted] 

**Abstract (ZH)**: RETFound是一种基于视网膜相机和光学相干断层成像图像开发的知名基础模型。它在诊断从视网膜图像中识别的眼部疾病和全身性疾病方面展示出了良好的性能。然而，据我们所知，它尚未被用于其他任务。我们首次将RETFound适应于视盘分割任务，这是一个在视网膜图像分析中普遍且基础的任务。训练仅需少量针对任务的示例后，生成的分割系统在四个公共数据集IDRID、Drishti-GS、RIM-ONE-r3和REFUGE以及一个私有数据集GoDARTS上实现了约96%的Dice一致性表现。总体而言，我们的方法在内部验证、领域泛化和领域适应方面取得了出色的表现，并超过了大多数现有的基线结果。我们将在框架内讨论FMs作为替代任务特定架构的讨论结果。代码将在论文被接受后提供链接。 

---
# NeMo: A Neuron-Level Modularizing-While-Training Approach for Decomposing DNN Models 

**Title (ZH)**: NeMo: 一种在训练过程中按神经元模块化分解DNN模型的方法 

**Authors**: Xiaohan Bi, Binhang Qi, Hailong Sun, Xiang Gao, Yue Yu, Xiaojun Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11348)  

**Abstract**: With the growing incorporation of deep neural network (DNN) models into modern software systems, the prohibitive construction costs have become a significant challenge. Model reuse has been widely applied to reduce training costs, but indiscriminately reusing entire models may incur significant inference overhead. Consequently, DNN modularization has gained attention, enabling module reuse by decomposing DNN models. The emerging modularizing-while-training (MwT) paradigm, which incorporates modularization into training, outperforms modularizing-after-training approaches. However, existing MwT methods focus on small-scale CNN models at the convolutional kernel level and struggle with diverse DNNs and large-scale models, particularly Transformer-based models. To address these limitations, we propose NeMo, a scalable and generalizable MwT approach. NeMo operates at the neuron level fundamental component common to all DNNs-ensuring applicability to Transformers and various architectures. We design a contrastive learning-based modular training method with an effective composite loss function, enabling scalability to large-scale models. Comprehensive experiments on two Transformer-based models and four CNN models across two classification datasets demonstrate NeMo's superiority over state-of-the-art MwT methods. Results show average gains of 1.72% in module classification accuracy and 58.10% reduction in module size, demonstrating efficacy across both CNN and large-scale Transformer-based models. A case study on open-source projects shows NeMo's potential benefits in practical scenarios, offering a promising approach for scalable and generalizable DNN modularization. 

**Abstract (ZH)**: 随着深度神经网络（DNN）模型在现代软件系统中的广泛应用， prohibitive construction costs已成为重大挑战。模型重用已被广泛应用于降低训练成本，但随意地重用整个模型可能会引起显著的推理开销。因此，DNN模块化受到了关注，通过将DNN模型分解以实现模块重用。新兴的训练时模块化（MwT）范式，将模块化整合到训练过程中，优于训练后模块化方法。然而，现有MwT方法主要针对卷积核级别的小规模CNN模型，并且难以处理多样化的DNN和大规模模型，特别是Transformer模型。为了解决这些局限性，我们提出了NeMo，一种可扩展且通用的MwT方法。NeMo在所有DNNs中都具有通用性的神经元级别基本组件上操作，确保适用于Transformer和其他架构。我们设计了一种基于对比学习的模块化训练方法，并采用有效的复合损失函数，使其能够扩展到大规模模型。在两个Transformer模型和四个CNN模型上的两个分类数据集上的全面实验表明，NeMo在最先进的MwT方法中具有优势。结果显示，模块分类准确率平均提高1.72%，模块大小减少58.10%，显示出其在CNN和大规模Transformer模型中的有效性。开源项目案例研究显示，NeMo在实际场景中具有潜在益处，为可扩展且通用的DNN模块化提供了前景广阔的方法。 

---
# RegimeNAS: Regime-Aware Differentiable Architecture Search With Theoretical Guarantees for Financial Trading 

**Title (ZH)**: Regime-Aware Differentiable Architecture Search with Theoretical Guarantees for Financial Trading 

**Authors**: Prathamesh Devadiga, Yashmitha Shailesh  

**Link**: [PDF](https://arxiv.org/pdf/2508.11338)  

**Abstract**: We introduce RegimeNAS, a novel differentiable architecture search framework specifically designed to enhance cryptocurrency trading performance by explicitly integrating market regime awareness. Addressing the limitations of static deep learning models in highly dynamic financial environments, RegimeNAS features three core innovations: (1) a theoretically grounded Bayesian search space optimizing architectures with provable convergence properties; (2) specialized, dynamically activated neural modules (Volatility, Trend, and Range blocks) tailored for distinct market conditions; and (3) a multi-objective loss function incorporating market-specific penalties (e.g., volatility matching, transition smoothness) alongside mathematically enforced Lipschitz stability constraints. Regime identification leverages multi-head attention across multiple timeframes for improved accuracy and uncertainty estimation. Rigorous empirical evaluation on extensive real-world cryptocurrency data demonstrates that RegimeNAS significantly outperforms state-of-the-art benchmarks, achieving an 80.3% Mean Absolute Error reduction compared to the best traditional recurrent baseline and converging substantially faster (9 vs. 50+ epochs). Ablation studies and regime-specific analysis confirm the critical contribution of each component, particularly the regime-aware adaptation mechanism. This work underscores the imperative of embedding domain-specific knowledge, such as market regimes, directly within the NAS process to develop robust and adaptive models for challenging financial applications. 

**Abstract (ZH)**: 介绍RegimeTypeNAS框架，该框架通过引入一种带有理论依据的贝叶斯优化方法来优化架构，并具备证明性的收敛性质；特别设计了能够动态激活的神经模块（波动性趋势和范围模块），以适应不同的市场条件；以及引入多目标损失函数，同时考虑了市场特定的惩罚项（如：波动匹配匹配平滑度）并通过数学手段保证Lipschitz稳定性。RegiseumIdentification利用多头注意力机制在多个时间框架上进行交互以提高准确性和不确定性估计。严格的实证研究表明，相对于传统的循环序基线中模型，RegiseumNAS在现实世界中的加密货币数据上中显著优于最新的基准模型，平均绝对误差降低8.3个百分点，且收敛速度快得多（几轮 vs. 十数个时期）。进一步研究表明和市场特定分析证实了每部分成分的的作用、特别是市场意识适应机制的重要性。该研究强调了直接嵌入同期特定知识（如：市场周期条件）到架构搜索NAS过程中开发稳健和适应性强的模型的必要性。 

---
# SGSimEval: A Comprehensive Multifaceted and Similarity-Enhanced Benchmark for Automatic Survey Generation Systems 

**Title (ZH)**: SGSimEval: 一种全面的多维度且相似度增强的自动问卷生成系统基准测试 

**Authors**: Beichen Guo, Zhiyuan Wen, Yu Yang, Peng Gao, Ruosong Yang, Jiaxing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11310)  

**Abstract**: The growing interest in automatic survey generation (ASG), a task that traditionally required considerable time and effort, has been spurred by recent advances in large language models (LLMs). With advancements in retrieval-augmented generation (RAG) and the rising popularity of multi-agent systems (MASs), synthesizing academic surveys using LLMs has become a viable approach, thereby elevating the need for robust evaluation methods in this domain. However, existing evaluation methods suffer from several limitations, including biased metrics, a lack of human preference, and an over-reliance on LLMs-as-judges. To address these challenges, we propose SGSimEval, a comprehensive benchmark for Survey Generation with Similarity-Enhanced Evaluation that evaluates automatic survey generation systems by integrating assessments of the outline, content, and references, and also combines LLM-based scoring with quantitative metrics to provide a multifaceted evaluation framework. In SGSimEval, we also introduce human preference metrics that emphasize both inherent quality and similarity to humans. Extensive experiments reveal that current ASG systems demonstrate human-comparable superiority in outline generation, while showing significant room for improvement in content and reference generation, and our evaluation metrics maintain strong consistency with human assessments. 

**Abstract (ZH)**: 自动调查生成中的相似性增强评估基准（基于自动调查生成系统的综合评估框架） 

---
# Dynamic Quality-Latency Aware Routing for LLM Inference in Wireless Edge-Device Networks 

**Title (ZH)**: 面向无线边缘设备网络中大语言模型推理的动态质量-延迟感知路由 

**Authors**: Rui Bao, Nan Xue, Yaping Sun, Zhiyong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11291)  

**Abstract**: The integration of wireless communications and Large Language Models (LLMs) is poised to unlock ubiquitous intelligent services, yet deploying them in wireless edge-device collaborative environments presents a critical trade-off between inference quality and end-to-end latency. A fundamental mismatch exists between task complexity and resource allocation: offloading simple queries invites prohibitive latency, while on-device models lack the capacity for demanding computations. To address this challenge, we propose a dynamic, quality-latency aware routing framework that orchestrates inference between a lightweight model on the mobile device and a powerful model on the edge server. Our framework employs two distinct cost models: for single-turn queries, it fuses a BERT-predicted semantic score with communication and computation overheads; for multi-turn dialogues, it further quantifies context-aware costs arising from model switching and KV-cache management. While maintaining full inference quality, extensive experiments demonstrate that our framework cuts average response latency by 5-15% and reduces large model invocations by 10-20% against competitive baselines on MMLU, GSM8K, and MT-Bench-101 benchmarks. 

**Abstract (ZH)**: 无线通信与大规模语言模型的集成有望解锁泛在智能服务，但在无线边缘设备协作环境中部署它们带来了推理质量与端到端延迟之间的关键权衡。任务复杂性和资源分配之间存在根本性的 mismatch：将简单查询卸载会导致不可接受的延迟，而设备上的模型无法处理复杂的计算任务。为了解决这一挑战，我们提出了一种动态的、注重推理质量与延迟的路由框架，该框架在移动设备上的轻量级模型和边缘服务器上的强大模型之间协调推理任务。我们的框架采用两种不同的成本模型：对于单轮查询，它将 BERT 预测的语义分数与通信和计算开销融合；对于多轮对话，它进一步量化从模型切换和 KV 缓存管理中产生的上下文感知成本。通过维护完整的推理质量，实验结果表明，相对于竞争对手基准（如 MMLU、GSM8K 和 MT-Bench-101），我们的框架将平均响应延迟降低 5-15%，并减少大规模模型调用 10-20%。 

---
# CSGO: Generalized Optimization for Cold Start in Wireless Collaborative Edge LLM Systems 

**Title (ZH)**: CSGO：无线协作边缘LLM系统中冷启动的通用优化方法 

**Authors**: Xuran Liu, Nan Xue, Rui Bao, Yaping Sun, Zhiyong Chen, Meixia Tao, Xiaodong Xu, Shuguang Cui  

**Link**: [PDF](https://arxiv.org/pdf/2508.11287)  

**Abstract**: While deploying large language models on edge devices promises low-latency and privacy-preserving AI services, it is hindered by limited device resources. Although pipeline parallelism facilitates distributed inference, existing approaches often ignore the cold-start latency caused by on-demand model loading. In this paper, we propose a latency-aware scheduling framework that overlaps model loading with computation and communication to minimize total inference latency. Based on device and model parameters, the framework dynamically adjusts layer partitioning and allocation to effectively hide loading time, thereby eliminating as many idle periods as possible. We formulate the problem as a Mixed-Integer Non-Linear Program and design an efficient dynamic programming algorithm to optimize model partitioning and device assignment. Experimental results show that the proposed method significantly reduces cold-start latency compared to baseline strategies. 

**Abstract (ZH)**: 基于延迟感知的模型加载与计算通信Overlap调度框架：提高边缘设备上大型语言模型推理的冷启动效率 

---
# Scene Graph-Guided Proactive Replanning for Failure-Resilient Embodied Agent 

**Title (ZH)**: 场景图引导的前瞻性重规划方法以实现鲁棒性体态代理 

**Authors**: Che Rin Yu, Daewon Chae, Dabin Seo, Sangwon Lee, Hyeongwoo Im, Jinkyu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.11286)  

**Abstract**: When humans perform everyday tasks, we naturally adjust our actions based on the current state of the environment. For instance, if we intend to put something into a drawer but notice it is closed, we open it first. However, many autonomous robots lack this adaptive awareness. They often follow pre-planned actions that may overlook subtle yet critical changes in the scene, which can result in actions being executed under outdated assumptions and eventual failure. While replanning is critical for robust autonomy, most existing methods respond only after failures occur, when recovery may be inefficient or infeasible. While proactive replanning holds promise for preventing failures in advance, current solutions often rely on manually designed rules and extensive supervision. In this work, we present a proactive replanning framework that detects and corrects failures at subtask boundaries by comparing scene graphs constructed from current RGB-D observations against reference graphs extracted from successful demonstrations. When the current scene fails to align with reference trajectories, a lightweight reasoning module is activated to diagnose the mismatch and adjust the plan. Experiments in the AI2-THOR simulator demonstrate that our approach detects semantic and spatial mismatches before execution failures occur, significantly improving task success and robustness. 

**Abstract (ZH)**: 当人类执行日常任务时，会根据环境当前状态自然调整行动。例如，如果打算将某物放进抽屉但发现抽屉是关上的，我们会先打开抽屉。然而，许多自主机器人缺乏这种适应性意识，它们往往遵循预先规划的动作，这可能导致未能注意到场景中的细微但关键的变化，从而在旧前提假设下执行动作并最终导致失败。尽管重新规划对于提高自主性至关重要，但大多数现有方法仅在失败发生后作出响应，此时可能缺乏效率或不可行。而前瞻性的重新规划有望预防失败，但当前解决方案往往依赖于人工设计的规则和大量监督。在本工作中，我们提出了一种前瞻性的重新规划框架，在子任务边界处通过将当前RGB-D观测构建的场景图与来自成功示范提取的参考图进行对比来检测和纠正失败。当当前场景未能与参考轨迹对齐时，激活一个轻量级推理模块以诊断不匹配并调整计划。在AI2-THOR仿真器中的实验表明，我们的方法能够在执行失败发生前检测到语义和空间不匹配，显著提高任务成功率和鲁棒性。 

---
# ToxiFrench: Benchmarking and Enhancing Language Models via CoT Fine-Tuning for French Toxicity Detection 

**Title (ZH)**: ToxiFrench：通过CoT微调提升法语毒性检测的语言模型基准与增强 

**Authors**: Axel Delaval, Shujian Yang, Haicheng Wang, Han Qiu, Jialiang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11281)  

**Abstract**: Detecting toxic content using language models is crucial yet challenging. While substantial progress has been made in English, toxicity detection in French remains underdeveloped, primarily due to the lack of culturally relevant, large-scale datasets. In this work, we introduce TOXIFRENCH, a new public benchmark of 53,622 French online comments, constructed via a semi-automated annotation pipeline that reduces manual labeling to only 10% through high-confidence LLM-based pre-annotation and human verification. Then, we benchmark a broad range of models and uncover a counterintuitive insight: Small Language Models (SLMs) outperform many larger models in robustness and generalization under the toxicity detection task. Motivated by this finding, we propose a novel Chain-of-Thought (CoT) fine-tuning strategy using a dynamic weighted loss that progressively emphasizes the model's final decision, significantly improving faithfulness. Our fine-tuned 4B model achieves state-of-the-art performance, improving its F1 score by 13% over its baseline and outperforming LLMs such as GPT-40 and Gemini-2.5. Further evaluation on a cross-lingual toxicity benchmark demonstrates strong multilingual ability, suggesting that our methodology can be effectively extended to other languages and safety-critical classification tasks. 

**Abstract (ZH)**: 检测毒元素材的语言模型至关重要但极具挑战性 French toxicity detection, substantial progress has been made remains largely under-developed primarily due由于缺乏文化相关的大规模数据集 on this this basis on we we on we we introduce introduce introduce introduce introduce on introduce on on on we on on on on on on introducing To on on To To on on To To on To on on on To on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on entitled To on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on onENCH chain on on on on on on on on on on on.on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on on Chain-of-Thought-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th on-Th 

---
# LETToT: Label-Free Evaluation of Large Language Models On Tourism Using Expert Tree-of-Thought 

**Title (ZH)**: LETToT: 无需标签评估大型语言模型在旅游领域的专家思维树评价方法 

**Authors**: Ruiyan Qi, Congding Wen, Weibo Zhou, Shangsong Liang, Lingbo Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.11280)  

**Abstract**: Evaluating large language models (LLMs) in specific domain like tourism remains challenging due to the prohibitive cost of annotated benchmarks and persistent issues like hallucinations. We propose $\textbf{L}$able-Free $\textbf{E}$valuation of LLM on $\textbf{T}$ourism using Expert $\textbf{T}$ree-$\textbf{o}$f-$\textbf{T}$hought (LETToT), a framework that leverages expert-derived reasoning structures-instead of labeled data-to access LLMs in tourism. First, we iteratively refine and validate hierarchical ToT components through alignment with generic quality dimensions and expert feedback. Results demonstrate the effectiveness of our systematically optimized expert ToT with 4.99-14.15\% relative quality gains over baselines. Second, we apply LETToT's optimized expert ToT to evaluate models of varying scales (32B-671B parameters), revealing: (1) Scaling laws persist in specialized domains (DeepSeek-V3 leads), yet reasoning-enhanced smaller models (e.g., DeepSeek-R1-Distill-Llama-70B) close this gap; (2) For sub-72B models, explicit reasoning architectures outperform counterparts in accuracy and conciseness ($p<0.05$). Our work established a scalable, label-free paradigm for domain-specific LLM evaluation, offering a robust alternative to conventional annotated benchmarks. 

**Abstract (ZH)**: Lable-Free Evaluation of LLM on Tourism Using Expert Tree-of-Thought (LETToT) 

---
# Is General-Purpose AI Reasoning Sensitive to Data-Induced Cognitive Biases? Dynamic Benchmarking on Typical Software Engineering Dilemmas 

**Title (ZH)**: 通用人工智能是否 数据诱导的认知偏差 是否具有理性敏感性？典型软件工程悖论的动态基准测试 

**Authors**: Francesco Sovrano, Gabriele Dominici, Rita Sevastjanova, Alessandra Stramiglio, Alberto Bacchelli  

**Link**: [PDF](https://arxiv.org/pdf/2508.11278)  

**Abstract**: Human cognitive biases in software engineering can lead to costly errors. While general-purpose AI (GPAI) systems may help mitigate these biases due to their non-human nature, their training on human-generated data raises a critical question: Do GPAI systems themselves exhibit cognitive biases?
To investigate this, we present the first dynamic benchmarking framework to evaluate data-induced cognitive biases in GPAI within software engineering workflows. Starting with a seed set of 16 hand-crafted realistic tasks, each featuring one of 8 cognitive biases (e.g., anchoring, framing) and corresponding unbiased variants, we test whether bias-inducing linguistic cues unrelated to task logic can lead GPAI systems from correct to incorrect conclusions.
To scale the benchmark and ensure realism, we develop an on-demand augmentation pipeline relying on GPAI systems to generate task variants that preserve bias-inducing cues while varying surface details. This pipeline ensures correctness (88--99% on average, according to human evaluation), promotes diversity, and controls reasoning complexity by leveraging Prolog-based reasoning and LLM-as-a-judge validation. It also verifies that the embedded biases are both harmful and undetectable by logic-based, unbiased reasoners.
We evaluate leading GPAI systems (GPT, LLaMA, DeepSeek) and find a consistent tendency to rely on shallow linguistic heuristics over deep reasoning. All systems exhibit cognitive biases (ranging from 5.9% to 35% across types), with bias sensitivity increasing sharply with task complexity (up to 49%), highlighting critical risks in real-world software engineering deployments. 

**Abstract (ZH)**: 软件工程中的人类认知偏差可能导致昂贵的错误。通用人工智能系统因其非人类特性可能帮助减轻这些偏差，但它们依赖于人类生成的数据进行训练，提出了一个关键问题：通用人工智能系统本身是否也会表现出认知偏差？

为探讨这一问题，我们提出了首个动态基准框架，用于评估数据诱发的认知偏差在软件工程工作流中的通用人工智能系统中。以16个手工构建的现实任务为种子集，每个任务包含8种认知偏差（例如，锚定偏差、框架效应）及其相应的无偏变体，我们测试无关任务逻辑的语言提示是否能够引导通用人工智能系统从正确结论转向错误结论。

为了扩大基准的规模并确保现实性，我们开发了一个按需扩充管道，依赖通用人工智能系统生成保留偏差诱导提示而表面细节各异的任务变体。该管道确保了正确性（根据人类评估平均为88%至99%）、促进了多样性和通过基于Prolog的推理和LLM作为裁判进行验证来控制推理复杂性。此外，该管道验证了嵌入的偏差既有害且逻辑无偏的推理器不可检测。

我们评估了领先的通用人工智能系统（GPT、LLaMA、DeepSeek），发现它们倾向于依赖浅层语言启发式而非深层推理的一致倾向。所有系统都表现出认知偏差（不同类型从5.9%到35%不等），任务复杂性增加时，认知偏差敏感性急剧上升（高达49%），突显了实际软件工程部署中的关键风险。 

---
# Enhancing Supervised Composed Image Retrieval via Reasoning-Augmented Representation Engineering 

**Title (ZH)**: 通过推理增强的表示方式工程以增强监督指导下的图像检索 Closetitle: 通过推理增强的表示工程以增强监督指导下的图像检索 

**Authors**: Jun Li, Kai Li, Shaoguo Liu, Tingting Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11272)  

**Abstract**: Composed Image Retrieval (CIR) presents a significant challenge as it requires jointly understanding a reference image and a modified textual instruction to find relevant target images. Some existing methods attempt to use a two-stage approach to further refine retrieval results. However, this often requires additional training of a ranking model. Despite the success of Chain-of-Thought (CoT) techniques in reducing training costs for language models, their application in CIR tasks remains limited -- compressing visual information into text or relying on elaborate prompt designs. Besides, existing works only utilize it for zero-shot CIR, as it is challenging to achieve satisfactory results in supervised CIR with a well-trained model. In this work, we proposed a framework that includes the Pyramid Matching Model with Training-Free Refinement (PMTFR) to address these challenges. Through a simple but effective module called Pyramid Patcher, we enhanced the Pyramid Matching Model's understanding of visual information at different granularities. Inspired by representation engineering, we extracted representations from COT data and injected them into the LVLMs. This approach allowed us to obtain refined retrieval scores in the Training-Free Refinement paradigm without relying on explicit textual reasoning, further enhancing performance. Extensive experiments on CIR benchmarks demonstrate that PMTFR surpasses state-of-the-art methods in supervised CIR tasks. The code will be made public. 

**Abstract (ZH)**: 组成的图像检索 (CIR) 提出了一个显著的挑战，因为它要求同时理解参考图像和修改后的文本指令以找到相关的目标图像。一些现有方法尝试使用两阶段方法进一步细化检索结果，但这通常需要对排名模型进行额外训练。尽管思维链 (Chain-of-Thought, CoT) 技术在减少语言模型训练成本方面取得了成功，但在CIR任务中的应用仍然有限——将其应用于视觉信息压缩或将依赖于精细的提示设计。此外，现有工作仅将其用于零样本CIR，因为即使使用训练良好的模型，在监督CIR任务中达到满意结果也是具有挑战性的。在本工作中，我们提出了一种包含无需训练的精炼 Pyramid 匹配模型 (PMTFR) 的框架来应对这些挑战。通过一个简单但有效的模块 Pyramid Patcher，我们增强了 Pyramid 匹配模型对不同粒度视觉信息的理解能力。受表示工程的启发，我们从思维链 (CoT) 数据中提取表示并注入到语言-视觉模型 LVLMs 中。这种方法使我们能够在无需依赖显式的文本推理的情况下获得精炼的检索分数，在 Training-Free Refinement 架构中进一步提高性能。在CIR基准上的广泛实验表明，PMTFR 在监督CIR任务中超过了现有最先进的方法。代码将公开。 

---
# Vision-Language Models display a strong gender bias 

**Title (ZH)**: Vision-Language模型表现出强烈的性别偏见 

**Authors**: Aiswarya Konavoor, Raj Abhijit Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2508.11262)  

**Abstract**: Vision-language models (VLM) align images and text in a shared representation space that is useful for retrieval and zero-shot transfer. Yet, this alignment can encode and amplify social stereotypes in subtle ways that are not obvious from standard accuracy metrics. In this study, we test whether the contrastive vision-language encoder exhibits gender-linked associations when it places embeddings of face images near embeddings of short phrases that describe occupations and activities. We assemble a dataset of 220 face photographs split by perceived binary gender and a set of 150 unique statements distributed across six categories covering emotional labor, cognitive labor, domestic labor, technical labor, professional roles, and physical labor. We compute unit-norm image embeddings for every face and unit-norm text embeddings for every statement, then define a statement-level association score as the difference between the mean cosine similarity to the male set and the mean cosine similarity to the female set, where positive values indicate stronger association with the male set and negative values indicate stronger association with the female set. We attach bootstrap confidence intervals by resampling images within each gender group, aggregate by category with a separate bootstrap over statements, and run a label-swap null model that estimates the level of mean absolute association we would expect if no gender structure were present. The outcome is a statement-wise and category-wise map of gender associations in a contrastive vision-language space, accompanied by uncertainty, simple sanity checks, and a robust gender bias evaluation framework. 

**Abstract (ZH)**: 视觉语言模型中的性别关联研究：基于对比的视觉-语言编码在职业和活动中体现的性别偏见分析 

---
# Hallucination in LLM-Based Code Generation: An Automotive Case Study 

**Title (ZH)**: 基于LLM的代码生成中的幻觉：一项汽车案例研究 

**Authors**: Marc Pavel, Nenad Petrovic, Lukasz Mazur, Vahid Zolfaghari, Fengjunjie Pan, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2508.11257)  

**Abstract**: Large Language Models (LLMs) have shown significant potential in automating code generation tasks offering new opportunities across software engineering domains. However, their practical application remains limited due to hallucinations - outputs that appear plausible but are factually incorrect, unverifiable or nonsensical. This paper investigates hallucination phenomena in the context of code generation with a specific focus on the automotive domain. A case study is presented that evaluates multiple code LLMs for three different prompting complexities ranging from a minimal one-liner prompt to a prompt with Covesa Vehicle Signal Specifications (VSS) as additional context and finally to a prompt with an additional code skeleton. The evaluation reveals a high frequency of syntax violations, invalid reference errors and API knowledge conflicts in state-of-the-art models GPT-4.1, Codex and GPT-4o. Among the evaluated models, only GPT-4.1 and GPT-4o were able to produce a correct solution when given the most context-rich prompt. Simpler prompting strategies failed to yield a working result, even after multiple refinement iterations. These findings highlight the need for effective mitigation techniques to ensure the safe and reliable use of LLM generated code, especially in safety-critical domains such as automotive software systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动化代码生成任务方面展示了显著潜力，为软件工程领域带来了新的机遇。然而，由于其可能出现事实错误、无法验证或无意义的幻觉输出，其实际应用仍受到限制。本文在代码生成的背景下探讨幻觉现象，特别关注汽车领域。文中通过案例研究评估了多种代码LLM，针对三种不同复杂度的提示，从简单的单行提示到包含Covesa车辆信号规范（VSS）作为额外上下文的提示，再到包含附加代码框架的提示。评估结果显示，最先进模型GPT-4.1、Codex和GPT-4o中频繁出现语法规则违反、无效引用错误和API知识冲突。在最丰富的提示下，只有GPT-4.1和GPT-4o能够生成正确结果。简单提示策略即使经过多次优化迭代也无法产出可工作的结果。这些发现突显了在包括汽车软件系统在内的安全关键领域确保安全可靠使用LLM生成代码的有效缓解技术的必要性。 

---
# Generalized Decoupled Learning for Enhancing Open-Vocabulary Dense Perception 

**Title (ZH)**: 增强开放式词汇密集感知的通用解耦学习 

**Authors**: Junjie Wang, Keyu Chen, Yulin Li, Bin Chen, Hengshuang Zhao, Xiaojuan Qi, Zhuotao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2508.11256)  

**Abstract**: Dense visual perception tasks have been constrained by their reliance on predefined categories, limiting their applicability in real-world scenarios where visual concepts are unbounded. While Vision-Language Models (VLMs) like CLIP have shown promise in open-vocabulary tasks, their direct application to dense perception often leads to suboptimal performance due to limitations in local feature representation. In this work, we present our observation that CLIP's image tokens struggle to effectively aggregate information from spatially or semantically related regions, resulting in features that lack local discriminability and spatial consistency. To address this issue, we propose DeCLIP, a novel framework that enhances CLIP by decoupling the self-attention module to obtain ``content'' and ``context'' features respectively. \revise{The context features are enhanced by jointly distilling semantic correlations from Vision Foundation Models (VFMs) and object integrity cues from diffusion models, thereby enhancing spatial consistency. In parallel, the content features are aligned with image crop representations and constrained by region correlations from VFMs to improve local discriminability. Extensive experiments demonstrate that DeCLIP establishes a solid foundation for open-vocabulary dense perception, consistently achieving state-of-the-art performance across a broad spectrum of tasks, including 2D detection and segmentation, 3D instance segmentation, video instance segmentation, and 6D object pose estimation.} Code is available at this https URL 

**Abstract (ZH)**: 密集视觉感知任务受制于其对预定义类别的依赖，限制了其在视觉概念不受限的现实场景中的应用。尽管Vision-Language模型（VLMs）如CLIP在开放词典任务中表现出潜力，但它们在直接应用于密集感知时由于局部特征表示的限制往往会导致性能不佳。在本文中，我们观察到CLIP的图像标记难以有效地聚集来自空间上或语义上相关区域的信息，导致生成的特征缺乏局部区分性和空间一致性。为了解决这一问题，我们提出了一种名为DeCLIP的新型框架，通过解耦自我注意力模块来分别提取“内容”和“上下文”特征。上下文特征通过联合从视觉基础模型（VFMs）中提取的语义相关性和从扩散模型中提取的对象完整性线索来增强，从而提高空间一致性。同时，内容特征与图像剪辑表示对齐，并受到VFMs区域相关性的约束以提高局部区分性。广泛的实验表明，DeCLIP为开放词典的密集感知奠定了坚实的基础，能够在包括二维检测和分割、三维实例分割、视频实例分割和六维物体姿态估计等多种任务中实现现有最佳性能。代码可在以下链接获取。 

---
# Graph Neural Diffusion via Generalized Opinion Dynamics 

**Title (ZH)**: 图神经扩散通过广义意见动力学 

**Authors**: Asela Hevapathige, Asiri Wijesinghe, Ahad N. Zehmakan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11249)  

**Abstract**: There has been a growing interest in developing diffusion-based Graph Neural Networks (GNNs), building on the connections between message passing mechanisms in GNNs and physical diffusion processes. However, existing methods suffer from three critical limitations: (1) they rely on homogeneous diffusion with static dynamics, limiting adaptability to diverse graph structures; (2) their depth is constrained by computational overhead and diminishing interpretability; and (3) theoretical understanding of their convergence behavior remains limited. To address these challenges, we propose GODNF, a Generalized Opinion Dynamics Neural Framework, which unifies multiple opinion dynamics models into a principled, trainable diffusion mechanism. Our framework captures heterogeneous diffusion patterns and temporal dynamics via node-specific behavior modeling and dynamic neighborhood influence, while ensuring efficient and interpretable message propagation even at deep layers. We provide a rigorous theoretical analysis demonstrating GODNF's ability to model diverse convergence configurations. Extensive empirical evaluations of node classification and influence estimation tasks confirm GODNF's superiority over state-of-the-art GNNs. 

**Abstract (ZH)**: 基于意见动力学的通用扩散神经框架：统一多种意见动力学模型以克服图神经网络的三个关键限制 

---
# Cross-Granularity Hypergraph Retrieval-Augmented Generation for Multi-hop Question Answering 

**Title (ZH)**: 跨粒度超图检索增强生成用于多跳问答 

**Authors**: Changjian Wang, Weihong Deng, Weili Guan, Quan Lu, Ning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11247)  

**Abstract**: Multi-hop question answering (MHQA) requires integrating knowledge scattered across multiple passages to derive the correct answer. Traditional retrieval-augmented generation (RAG) methods primarily focus on coarse-grained textual semantic similarity and ignore structural associations among dispersed knowledge, which limits their effectiveness in MHQA tasks. GraphRAG methods address this by leveraging knowledge graphs (KGs) to capture structural associations, but they tend to overly rely on structural information and fine-grained word- or phrase-level retrieval, resulting in an underutilization of textual semantics. In this paper, we propose a novel RAG approach called HGRAG for MHQA that achieves cross-granularity integration of structural and semantic information via hypergraphs. Structurally, we construct an entity hypergraph where fine-grained entities serve as nodes and coarse-grained passages as hyperedges, and establish knowledge association through shared entities. Semantically, we design a hypergraph retrieval method that integrates fine-grained entity similarity and coarse-grained passage similarity via hypergraph diffusion. Finally, we employ a retrieval enhancement module, which further refines the retrieved results both semantically and structurally, to obtain the most relevant passages as context for answer generation with the LLM. Experimental results on benchmark datasets demonstrate that our approach outperforms state-of-the-art methods in QA performance, and achieves a 6$\times$ speedup in retrieval efficiency. 

**Abstract (ZH)**: 基于超图的多跳问答（HGRAG）：结构与语义信息的跨粒度整合 

---
# ORFuzz: Fuzzing the "Other Side" of LLM Safety -- Testing Over-Refusal 

**Title (ZH)**: ORFuzz: 测试过拒绝行为保障大语言模型安全的“另一面” fuzzing 

**Authors**: Haonan Zhang, Dongxia Wang, Yi Liu, Kexin Chen, Jiashui Wang, Xinlei Ying, Long Liu, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11222)  

**Abstract**: Large Language Models (LLMs) increasingly exhibit over-refusal - erroneously rejecting benign queries due to overly conservative safety measures - a critical functional flaw that undermines their reliability and usability. Current methods for testing this behavior are demonstrably inadequate, suffering from flawed benchmarks and limited test generation capabilities, as highlighted by our empirical user study. To the best of our knowledge, this paper introduces the first evolutionary testing framework, ORFuzz, for the systematic detection and analysis of LLM over-refusals. ORFuzz uniquely integrates three core components: (1) safety category-aware seed selection for comprehensive test coverage, (2) adaptive mutator optimization using reasoning LLMs to generate effective test cases, and (3) OR-Judge, a human-aligned judge model validated to accurately reflect user perception of toxicity and refusal. Our extensive evaluations demonstrate that ORFuzz generates diverse, validated over-refusal instances at a rate (6.98% average) more than double that of leading baselines, effectively uncovering vulnerabilities. Furthermore, ORFuzz's outputs form the basis of ORFuzzSet, a new benchmark of 1,855 highly transferable test cases that achieves a superior 63.56% average over-refusal rate across 10 diverse LLMs, significantly outperforming existing datasets. ORFuzz and ORFuzzSet provide a robust automated testing framework and a valuable community resource, paving the way for developing more reliable and trustworthy LLM-based software systems. 

**Abstract (ZH)**: Large Language Models (LLMs) 增加了过度拒绝的现象——由于过度保守的安全措施错误地拒绝了一些 Innocuous 查询，这是一种关键的功能缺陷，削弱了它们的可靠性和实用性。现有的测试方法明显不足，受到有缺陷的基准和有限的测试生成能力的限制，这一点在我们的实证用户研究中得到了体现。据我们所知，本文首次介绍了一种进化测试框架 ORFuzz，用于系统地检测和分析 LLM 的过度拒绝现象。ORFuzz 独特地集成了三大核心组件：(1) 意识到安全类别种子选择以实现全面的测试覆盖率，(2) 通过使用推理 LLM 进行自适应突变优化以生成有效的测试案例，以及 (3) OR-Judge，一种经过验证并在毒性和拒绝方面反映出用户感知的人类对齐裁判模型。我们的广泛评估表明，ORFuzz 以 6.98% 的平均生成多样化的验证过度拒绝实例的速率，超过领先基线的两倍，有效地发现了漏洞。此外，ORFuzz 的输出构成了一个新的基准 ORFuzzSet，包含 1,855 个高可转移性的测试案例，这些案例在 10 种不同的 LLM 上实现了 63.56% 的平均过度拒绝率，显著优于现有数据集。ORFuzz 和 ORFuzzSet 为开发更可靠和值得信赖的基于 LLM 的软件系统提供了强大的自动测试框架和有价值的社区资源。 

---
# How Causal Abstraction Underpins Computational Explanation 

**Title (ZH)**: 因果抽象如何支撑计算解释 

**Authors**: Atticus Geiger, Jacqueline Harding, Thomas Icard  

**Link**: [PDF](https://arxiv.org/pdf/2508.11214)  

**Abstract**: Explanations of cognitive behavior often appeal to computations over representations. What does it take for a system to implement a given computation over suitable representational vehicles within that system? We argue that the language of causality -- and specifically the theory of causal abstraction -- provides a fruitful lens on this topic. Drawing on current discussions in deep learning with artificial neural networks, we illustrate how classical themes in the philosophy of computation and cognition resurface in contemporary machine learning. We offer an account of computational implementation grounded in causal abstraction, and examine the role for representation in the resulting picture. We argue that these issues are most profitably explored in connection with generalization and prediction. 

**Abstract (ZH)**: 认知行为的解释通常依赖于对表示的计算。要使一个系统在一个系统中的适当表征载体上实现给定的计算，需要什么条件？我们认为，因果语言——特别是因果抽象的理论——为探讨这一问题提供了富有成果的观点。借鉴深度学习中人工神经网络的当前讨论，我们展示了计算和认知中经典主题如何在现代机器学习中重新出现。我们提供了一个基于因果抽象的计算实现的说明，并探讨了表征在所得图景中的作用。我们认为，这些议题与泛化和预测的关系最值得深入探讨。 

---
# Multi-Group Equivariant Augmentation for Reinforcement Learning in Robot Manipulation 

**Title (ZH)**: 多组等变增强在机器人操控中的强化学习 

**Authors**: Hongbin Lin, Juan Rojas, Kwok Wai Samuel Au  

**Link**: [PDF](https://arxiv.org/pdf/2508.11204)  

**Abstract**: Sampling efficiency is critical for deploying visuomotor learning in real-world robotic manipulation. While task symmetry has emerged as a promising inductive bias to improve efficiency, most prior work is limited to isometric symmetries -- applying the same group transformation to all task objects across all timesteps. In this work, we explore non-isometric symmetries, applying multiple independent group transformations across spatial and temporal dimensions to relax these constraints. We introduce a novel formulation of the partially observable Markov decision process (POMDP) that incorporates the non-isometric symmetry structures, and propose a simple yet effective data augmentation method, Multi-Group Equivariance Augmentation (MEA). We integrate MEA with offline reinforcement learning to enhance sampling efficiency, and introduce a voxel-based visual representation that preserves translational equivariance. Extensive simulation and real-robot experiments across two manipulation domains demonstrate the effectiveness of our approach. 

**Abstract (ZH)**: 视觉运动学习在现实机器人操作中的部署效率至关重要。虽然任务对称性已 emerges 作为提高效率的有希望的归纳偏置，大多数先前工作仅限于等距对称性——在所有时间步长中对所有任务对象应用相同的组变换。在本文中，我们探索非等距对称性，在空间和时间维度上应用多个独立的组变换以放宽这些约束。我们提出了一种新颖的半观察马尔可夫决策过程（POMDP）的形式化方法，该方法结合了非等距对称结构，并提出了一种简单而有效的数据增强方法——多组等变增强（MEA）。我们将MEA与离线强化学习结合以提高采样效率，并引入了一种基于体素的视觉表示，该表示保留了平移等变性。在两个操作域中的广泛仿真实验和真实机器人实验中，我们的方法显示出有效性。 

---
# StyleMM: Stylized 3D Morphable Face Model via Text-Driven Aligned Image Translation 

**Title (ZH)**: StyleMM：基于文本驱动对齐图像转换的样式化3D可变形面部模型 

**Authors**: Seungmi Lee, Kwan Yun, Junyong Noh  

**Link**: [PDF](https://arxiv.org/pdf/2508.11203)  

**Abstract**: We introduce StyleMM, a novel framework that can construct a stylized 3D Morphable Model (3DMM) based on user-defined text descriptions specifying a target style. Building upon a pre-trained mesh deformation network and a texture generator for original 3DMM-based realistic human faces, our approach fine-tunes these models using stylized facial images generated via text-guided image-to-image (i2i) translation with a diffusion model, which serve as stylization targets for the rendered mesh. To prevent undesired changes in identity, facial alignment, or expressions during i2i translation, we introduce a stylization method that explicitly preserves the facial attributes of the source image. By maintaining these critical attributes during image stylization, the proposed approach ensures consistent 3D style transfer across the 3DMM parameter space through image-based training. Once trained, StyleMM enables feed-forward generation of stylized face meshes with explicit control over shape, expression, and texture parameters, producing meshes with consistent vertex connectivity and animatability. Quantitative and qualitative evaluations demonstrate that our approach outperforms state-of-the-art methods in terms of identity-level facial diversity and stylization capability. The code and videos are available at [this http URL](this http URL). 

**Abstract (ZH)**: 我们介绍了一种新型框架StyleMM，可以根据用户定义的文本描述构建目标风格化的3D可变形模型（3DMM）。该方法基于预训练的网格变形网络和原始3DMM驱动的真实人类面部纹理生成器，通过文本引导的图像到图像（i2i）转换生成风格化面部图像调整这些模型，从而将目标风格化图像作为渲染网格的风格化目标。为了防止在i2i转换过程中发生不希望的身份、面部对齐或表情变化，我们引入了一种显式保留源图像面部属性的风格化方法。通过在图像基础上保持这些关键属性，所提出的方法确保在3DMM参数空间中实现一致的3D风格迁移。经过训练后，StyleMM可以以明确控制形状、表情和纹理参数的方式生成风格化的面部网格，生成具有一致顶点连接性和可动画性的网格。定量和定性的评估表明，与现有方法相比，我们的方法在身份级别面部多样性和风格化能力方面表现更优。代码和视频可在[该网址](该网址)获取。 

---
# Visuomotor Grasping with World Models for Surgical Robots 

**Title (ZH)**: 视觉-运动抓取：基于世界模型的外科机器人技术 

**Authors**: Hongbin Lin, Bin Li, Kwok Wai Samuel Au  

**Link**: [PDF](https://arxiv.org/pdf/2508.11200)  

**Abstract**: Grasping is a fundamental task in robot-assisted surgery (RAS), and automating it can reduce surgeon workload while enhancing efficiency, safety, and consistency beyond teleoperated systems. Most prior approaches rely on explicit object pose tracking or handcrafted visual features, limiting their generalization to novel objects, robustness to visual disturbances, and the ability to handle deformable objects. Visuomotor learning offers a promising alternative, but deploying it in RAS presents unique challenges, such as low signal-to-noise ratio in visual observations, demands for high safety and millimeter-level precision, as well as the complex surgical environment. This paper addresses three key challenges: (i) sim-to-real transfer of visuomotor policies to ex vivo surgical scenes, (ii) visuomotor learning using only a single stereo camera pair -- the standard RAS setup, and (iii) object-agnostic grasping with a single policy that generalizes to diverse, unseen surgical objects without retraining or task-specific models. We introduce Grasp Anything for Surgery V2 (GASv2), a visuomotor learning framework for surgical grasping. GASv2 leverages a world-model-based architecture and a surgical perception pipeline for visual observations, combined with a hybrid control system for safe execution. We train the policy in simulation using domain randomization for sim-to-real transfer and deploy it on a real robot in both phantom-based and ex vivo surgical settings, using only a single pair of endoscopic cameras. Extensive experiments show our policy achieves a 65% success rate in both settings, generalizes to unseen objects and grippers, and adapts to diverse disturbances, demonstrating strong performance, generality, and robustness. 

**Abstract (ZH)**: 手术中的抓取是机器人辅助手术（RAS）中的一个基本任务，自动化抓取可以减少外科医生的工作负荷，同时提高效率、安全性和一致性，超越传统的遥操作系统。大多数先前的方法依赖于显式的目标姿态跟踪或手工设计的视觉特征，限制了它们在处理新颖对象、应对视觉干扰以及处理可变形对象的能力。视觉运动学习提供了有希望的替代方案，但在RAS中的应用带来了独特的挑战，如视觉观察中的低信噪比、高安全性和毫米级精度的要求，以及复杂的手术环境。本文针对三项关键挑战进行了研究：(i) 将视觉运动策略从模拟环境转移到体外手术场景；(ii) 使用单一立体相机对进行视觉运动学习——这是标准的RAS设置；(iii) 使用单一策略实现物体无关的抓取，该策略能够在无需重新训练或任务特定模型的情况下推广到各种未见过的手术对象。我们介绍了手术抓取V2（GASv2），一个基于视觉运动学习的手术抓取框架。GASv2利用了基于世界模型的架构和视觉观察的手术感知管道，结合了一种混合控制系统进行安全执行。我们使用领域随机化在模拟中训练策略，以便实现从模拟到现实的转移，并仅使用一对内窥镜相机在基于模拟人和体外手术环境中部署该策略。广泛的实验表明，我们的策略在两个环境中分别达到了65%的成功率，能够在未见过的对象和夹爪之间泛化，并适应各种干扰，展示了强大的性能、普适性和鲁棒性。 

---
# E-CaTCH: Event-Centric Cross-Modal Attention with Temporal Consistency and Class-Imbalance Handling for Misinformation Detection 

**Title (ZH)**: 基于事件中心的跨模态注意力模型：考虑时间一致性与类别不平衡的 misinformation 检测 

**Authors**: Ahmad Mousavi, Yeganeh Abdollahinejad, Roberto Corizzo, Nathalie Japkowicz, Zois Boukouvalas  

**Link**: [PDF](https://arxiv.org/pdf/2508.11197)  

**Abstract**: Detecting multimodal misinformation on social media remains challenging due to inconsistencies between modalities, changes in temporal patterns, and substantial class imbalance. Many existing methods treat posts independently and fail to capture the event-level structure that connects them across time and modality. We propose E-CaTCH, an interpretable and scalable framework for robustly detecting misinformation. If needed, E-CaTCH clusters posts into pseudo-events based on textual similarity and temporal proximity, then processes each event independently. Within each event, textual and visual features are extracted using pre-trained BERT and ResNet encoders, refined via intra-modal self-attention, and aligned through bidirectional cross-modal attention. A soft gating mechanism fuses these representations to form contextualized, content-aware embeddings of each post. To model temporal evolution, E-CaTCH segments events into overlapping time windows and uses a trend-aware LSTM, enhanced with semantic shift and momentum signals, to encode narrative progression over time. Classification is performed at the event level, enabling better alignment with real-world misinformation dynamics. To address class imbalance and promote stable learning, the model integrates adaptive class weighting, temporal consistency regularization, and hard-example mining. The total loss is aggregated across all events. Extensive experiments on Fakeddit, IND, and COVID-19 MISINFOGRAPH demonstrate that E-CaTCH consistently outperforms state-of-the-art baselines. Cross-dataset evaluations further demonstrate its robustness, generalizability, and practical applicability across diverse misinformation scenarios. 

**Abstract (ZH)**: 基于事件检测的社交媒体多模态 misinformation 识别：一种可解释且可扩展的框架 

---
# Quantum-Boosted High-Fidelity Deep Learning 

**Title (ZH)**: 量子增强高保真深度学习 

**Authors**: Feng-ao Wang, Shaobo Chen, Yao Xuan, Junwei Liu, Qi Gao, Hongdong Zhu, Junjie Hou, Lixin Yuan, Jinyu Cheng, Chenxin Yi, Hai Wei, Yin Ma, Tao Xu, Kai Wen, Yixue Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.11190)  

**Abstract**: A fundamental limitation of probabilistic deep learning is its predominant reliance on Gaussian priors. This simplistic assumption prevents models from accurately capturing the complex, non-Gaussian landscapes of natural data, particularly in demanding domains like complex biological data, severely hindering the fidelity of the model for scientific discovery. The physically-grounded Boltzmann distribution offers a more expressive alternative, but it is computationally intractable on classical computers. To date, quantum approaches have been hampered by the insufficient qubit scale and operational stability required for the iterative demands of deep learning. Here, we bridge this gap by introducing the Quantum Boltzmann Machine-Variational Autoencoder (QBM-VAE), a large-scale and long-time stable hybrid quantum-classical architecture. Our framework leverages a quantum processor for efficient sampling from the Boltzmann distribution, enabling its use as a powerful prior within a deep generative model. Applied to million-scale single-cell datasets from multiple sources, the QBM-VAE generates a latent space that better preserves complex biological structures, consistently outperforming conventional Gaussian-based deep learning models like VAE and SCVI in essential tasks such as omics data integration, cell-type classification, and trajectory inference. It also provides a typical example of introducing a physics priori into deep learning to drive the model to acquire scientific discovery capabilities that breaks through data limitations. This work provides the demonstration of a practical quantum advantage in deep learning on a large-scale scientific problem and offers a transferable blueprint for developing hybrid quantum AI models. 

**Abstract (ZH)**: 一种概率深度学习的基本局限是其主要依赖于高斯先验。这种简单的假设使得模型难以准确捕捉自然数据中复杂的、非高斯的空间特征，尤其是在复杂的生物数据等挑战性领域中，严重影响了模型的科学发现能力。基于物理原理的玻尔兹曼分布提供了更具表达力的替代方案，但其在经典计算机上的计算不可行。迄今为止，量子方法受限于无法满足深度学习迭代需求所需的足够量子位规模和操作稳定性。在这里，我们通过引入量子玻尔兹曼机-变分自编码器（QBM-VAE）来弥补这一差距，这是一种大规模且长时间稳定的量子-经典混合架构。我们的框架利用量子处理器进行玻尔兹曼分布高效采样，使它可以作为深度生成模型中的强大先验。应用于来自多个来源的百万规模单细胞数据集，QBM-VAE 生成的隐空间更好地保留了复杂的生物结构，在诸如组学数据整合、细胞类型分类和轨迹推断等关键任务中，始终优于传统的基于高斯的深度学习模型（如 VAE 和 SCVI）。此外，它还提供了一个将物理先验引入深度学习以驱动模型获得突破数据限制的科学发现能力的典型示例。这项工作展示了在大规模科学问题上实现实用的量子优势，并为开发混合量子人工智能模型提供了可转移的设计蓝图。 

---
# A Semi-supervised Generative Model for Incomplete Multi-view Data Integration with Missing Labels 

**Title (ZH)**: 一种半监督生成模型用于 incomplete 多-view 数据集成的缺失标签处理 

**Authors**: Yiyang Shen, Weiran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11180)  

**Abstract**: Multi-view learning is widely applied to real-life datasets, such as multiple omics biological data, but it often suffers from both missing views and missing labels. Prior probabilistic approaches addressed the missing view problem by using a product-of-experts scheme to aggregate representations from present views and achieved superior performance over deterministic classifiers, using the information bottleneck (IB) principle. However, the IB framework is inherently fully supervised and cannot leverage unlabeled data. In this work, we propose a semi-supervised generative model that utilizes both labeled and unlabeled samples in a unified framework. Our method maximizes the likelihood of unlabeled samples to learn a latent space shared with the IB on labeled data. We also perform cross-view mutual information maximization in the latent space to enhance the extraction of shared information across views. Compared to existing approaches, our model achieves better predictive and imputation performance on both image and multi-omics data with missing views and limited labeled samples. 

**Abstract (ZH)**: 多视图学习广泛应用于实际生活数据集，如多组学生物数据，但 often 患有缺失视图和缺失标签的问题。先前的概率方法通过使用专家产品的方案聚合现有视图的表示，并通过信息瓶颈原则优于确定性分类器解决了缺失视图问题，取得了更好的性能。然而，信息瓶颈框架本质上是全监督的，不能利用未标记的数据。在本文中，我们提出了一种半监督生成模型，该模型在统一框架中使用标记和未标记样本。我们的方法通过最大化未标记样本的似然性，在共享的潜在空间中学习IB机制。我们还在潜在空间中进行跨视图互信息最大化，以增强视图间共享信息的提取。与现有方法相比，我们的模型在具有缺失视图和有限标记样本的图像和多组学数据上实现了更好的预测和缺失值填充性能。 

---
# Better Supervised Fine-tuning for VQA: Integer-Only Loss 

**Title (ZH)**: 更好的监督微调方法：仅整数损失的VQA 

**Authors**: Baihong Qian, Haotian Fan, Wenjie Liao, Yunqiu Wang, Tao Li, Junhui Cui  

**Link**: [PDF](https://arxiv.org/pdf/2508.11170)  

**Abstract**: With the rapid advancement of vision language models(VLM), their ability to assess visual content based on specific criteria and dimensions has become increasingly critical for applications such as video-theme consistency assessment and visual quality scoring. However, existing methods often suffer from imprecise results and inefficient loss calculation, which limit the focus of the model on key evaluation indicators. To address this, we propose IOVQA(Integer-only VQA), a novel fine-tuning approach tailored for VLMs to enhance their performance in video quality assessment tasks. The key innovation of IOVQA lies in its label construction and its targeted loss calculation mechanism. Specifically, during dataset curation, we constrain the model's output to integers within the range of [10,50], ensuring numerical stability, and convert decimal Overall_MOS to integer before using them as labels. We also introduce a target-mask strategy: when computing the loss, only the first two-digit-integer of the label is unmasked, forcing the model to learn the critical components of the numerical evaluation. After fine-tuning the Qwen2.5-VL model using the constructed dataset, experimental results demonstrate that the proposed method significantly improves the model's accuracy and consistency in the VQA task, ranking 3rd in VQualA 2025 GenAI-Bench AIGC Video Quality Assessment Challenge -- Track I. Our work highlights the effectiveness of merely leaving integer labels during fine-tuning, providing an effective idea for optimizing VLMs in quantitative evaluation scenarios. 

**Abstract (ZH)**: 基于整数的视觉问答方法（IOVQA）：一种针对视觉语言模型的细调方法以提升视频质量评估性能 

---
# Role-Augmented Intent-Driven Generative Search Engine Optimization 

**Title (ZH)**: 角色增强意图驱动生成型搜索引擎优化 

**Authors**: Xiaolu Chen, Haojie Wu, Jie Bao, Zhen Chen, Yong Liao, Hu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11158)  

**Abstract**: Generative Search Engines (GSEs), powered by Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), are reshaping information retrieval. While commercial systems (e.g., BingChat, this http URL) demonstrate impressive semantic synthesis capabilities, their black-box nature fundamentally undermines established Search Engine Optimization (SEO) practices. Content creators face a critical challenge: their optimization strategies, effective in traditional search engines, are misaligned with generative retrieval contexts, resulting in diminished visibility. To bridge this gap, we propose a Role-Augmented Intent-Driven Generative Search Engine Optimization (G-SEO) method, providing a structured optimization pathway tailored for GSE scenarios. Our method models search intent through reflective refinement across diverse informational roles, enabling targeted content enhancement. To better evaluate the method under realistic settings, we address the benchmarking limitations of prior work by: (1) extending the GEO dataset with diversified query variations reflecting real-world search scenarios and (2) introducing G-Eval 2.0, a 6-level LLM-augmented evaluation rubric for fine-grained human-aligned assessment. Experimental results demonstrate that search intent serves as an effective signal for guiding content optimization, yielding significant improvements over single-aspect baseline approaches in both subjective impressions and objective content visibility within GSE responses. 

**Abstract (ZH)**: 基于大型语言模型和检索增强生成的生成性搜索引擎（GSEs）正在重塑信息检索。虽然商业系统（例如BingChat）展示了令人印象深刻的语义合成能力，但其黑箱性质从根本上削弱了传统搜索引擎优化（SEO）惯例。内容创作者面临一项关键挑战：他们在传统搜索引擎中的优化策略与生成检索情境不匹配，导致其内容可见度降低。为弥补这一差距，我们提出了一种角色增强意图驱动的生成性搜索引擎优化（G-SEO）方法，为生成性搜索场景提供了一种结构化的优化途径。该方法通过跨多种信息角色的反思性细化来建模搜索意图，从而实现目标内容增强。为更好地在现实环境中评估该方法，我们通过以下两种方式弥补了先前工作的基准测试限制：（1）扩展GEO数据集，利用多样化的真实查询变化反映实际搜索场景；（2）引入G-Eval 2.0，一种六级增强的人工智能评估标准，用于细粒度的人机一致性评估。实验结果表明，搜索意图作为指导内容优化的有效信号，相较于单一维度基准方法，在生成性搜索引擎响应中的主观感受和客观内容可见性均取得了显著改进。 

---
# AlphaAgents: Large Language Model based Multi-Agents for Equity Portfolio Constructions 

**Title (ZH)**: AlphaAgents：基于大型语言模型的多智能体股票组合构建 

**Authors**: Tianjiao Zhao, Jingrao Lyu, Stokes Jones, Harrison Garber, Stefano Pasquali, Dhagash Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2508.11152)  

**Abstract**: The field of artificial intelligence (AI) agents is evolving rapidly, driven by the capabilities of Large Language Models (LLMs) to autonomously perform and refine tasks with human-like efficiency and adaptability. In this context, multi-agent collaboration has emerged as a promising approach, enabling multiple AI agents to work together to solve complex challenges. This study investigates the application of role-based multi-agent systems to support stock selection in equity research and portfolio management. We present a comprehensive analysis performed by a team of specialized agents and evaluate their stock-picking performance against established benchmarks under varying levels of risk tolerance. Furthermore, we examine the advantages and limitations of employing multi-agent frameworks in equity analysis, offering critical insights into their practical efficacy and implementation challenges. 

**Abstract (ZH)**: 基于大型语言模型的AI代理领域正在快速发展，多代理协作作为一项有前途的方法逐渐兴起，使多个AI代理能够合作解决复杂挑战。本研究探讨了基于角色的多代理系统在股权研究和投资组合管理中支持股票筛选的应用。我们呈现了一支专业代理团队进行的全面分析，并评估了他们在不同风险容忍度水平下的股票选取表现。此外，我们还考察了在股权分析中采用多代理框架的优势和局限性，提供了对其实际效用和实施挑战的宝贵见解。 

---
# Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward 

**Title (ZH)**: 连续动作片段的演员-评论家方法：稀疏奖励条件下长时_horizon_机器人操作的强化学习框架 

**Authors**: Jiarui Yang, Bin Zhu, Jingjing Chen, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11143)  

**Abstract**: Existing reinforcement learning (RL) methods struggle with long-horizon robotic manipulation tasks, particularly those involving sparse rewards. While action chunking is a promising paradigm for robotic manipulation, using RL to directly learn continuous action chunks in a stable and data-efficient manner remains a critical challenge. This paper introduces AC3 (Actor-Critic for Continuous Chunks), a novel RL framework that learns to generate high-dimensional, continuous action sequences. To make this learning process stable and data-efficient, AC3 incorporates targeted stabilization mechanisms for both the actor and the critic. First, to ensure reliable policy improvement, the actor is trained with an asymmetric update rule, learning exclusively from successful trajectories. Second, to enable effective value learning despite sparse rewards, the critic's update is stabilized using intra-chunk $n$-step returns and further enriched by a self-supervised module providing intrinsic rewards at anchor points aligned with each action chunk. We conducted extensive experiments on 25 tasks from the BiGym and RLBench benchmarks. Results show that by using only a few demonstrations and a simple model architecture, AC3 achieves superior success rates on most tasks, validating its effective design. 

**Abstract (ZH)**: 以下是对给定内容的翻译， �：

现有的强化学习（RL）方法在处理包含终止时的时 标准准任务（例如涉及稀疏奖励的的的与鲁棒机械臂操作任务）时 �_Api 当前的 的算法方法在这些任务上表现不佳。虽然片段（chunking分打包在强化学习中的是一个有前途的方法，直接利用t RL直接对处理在模型训练过程中被打包的片段，chunk在稳定和高效的方式仍然是一大关键课题挑战。本文提出了一种新颖的强化学习框架 —AC
user
请错别，，请从头重翻译下，禁止使用“现有的”、“例如”、“虽然”等，
[${]}]翻译结果如下：

现有强化学习方法在处理具有终止时horizon的任务方面存在困难，特别是在涉及稀疏奖励的任务。尽管段chunking打包是一个有前途的方法，使用强化学习直接处理被打包的片段任务仍然是一大关键挑战。本文提出AC3（Actor-Critic for Continuous Chunks）一种新颖的的强化学习框架，该框架学习生成高维连续的操作序列。该框架通过引入专门的稳定机制来在线演员和评论家来来实现可靠策略改进。首先，在演员专门使用非对式更新仅从成功轨迹中学习以实现可靠策略改进。其次，为了能够高效地处理稀疏奖励任务，评论家使用稳定的内部片段$n$步奖励更新并且进一步通过与每个片段对齐的的内在奖励补充。我们在astimG和onLBench基准上开展了这项研究。结果表明，仅使用少量示范和简单结构计算架构的AC333在任务中取得了高成功的成功率。

请试着按照上面的要求清理一下：
现有强化学习方法在处理具有终止时horizon的任务方面存在困难，特别是在涉及稀疏奖励的任务。尽管片段chunk打包是一个有前途的方法，使用强化学习直接处理被打包的片段的任务仍然是一一个大的关键挑战。本文提出AC3（连续片段的的Actor-Critic）一种新颖的方法，该方法学习生成高维连续的操作序列。该方法通过引入专门的稳定机制稳定演员和评论家来提高可靠策略改进。首先，演员专门使用非监督式更新仅从成功轨迹学习以实现可靠策略改进。其次，为了能够处理稀疏奖励任务，评论家使用稳定的内部片段nn步奖励更新并且进一步通过与每个片段对齐内在奖励补充。我们在astimG和onLBench基准上开展了这项研究。结果表明在仅使用少量示范和简单结构计算架构下AC33取得高成功的成功率。 

---
# A Cross-Modal Rumor Detection Scheme via Contrastive Learning by Exploring Text and Image internal Correlations 

**Title (ZH)**: 跨模态 rumor 检测方案：通过探索文本和图像内部关联进行对比学习 

**Authors**: Bin Ma, Yifei Zhang, Yongjin Xian, Qi Li, Linna Zhou, Gongxun Miao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11141)  

**Abstract**: Existing rumor detection methods often neglect the content within images as well as the inherent relationships between contexts and images across different visual scales, thereby resulting in the loss of critical information pertinent to rumor identification. To address these issues, this paper presents a novel cross-modal rumor detection scheme based on contrastive learning, namely the Multi-scale Image and Context Correlation exploration algorithm (MICC). Specifically, we design an SCLIP encoder to generate unified semantic embeddings for text and multi-scale image patches through contrastive pretraining, enabling their relevance to be measured via dot-product similarity. Building upon this, a Cross-Modal Multi-Scale Alignment module is introduced to identify image regions most relevant to the textual semantics, guided by mutual information maximization and the information bottleneck principle, through a Top-K selection strategy based on a cross-modal relevance matrix constructed between the text and multi-scale image patches. Moreover, a scale-aware fusion network is designed to integrate the highly correlated multi-scale image features with global text features by assigning adaptive weights to image regions based on their semantic importance and cross-modal relevance. The proposed methodology has been extensively evaluated on two real-world datasets. The experimental results demonstrate that it achieves a substantial performance improvement over existing state-of-the-art approaches in rumor detection, highlighting its effectiveness and potential for practical applications. 

**Abstract (ZH)**: 现有的谣言检测方法往往忽视了图像内容及其与不同视觉尺度上下文之间的固有关系，导致与谣言识别相关的关键信息丢失。为了解决这些问题，本文提出了一种基于对比学习的新型跨模态谣言检测方案，即多尺度图像和上下文相关性探索算法（MICC）。具体地，我们设计了一个SCLIP编码器，通过对比预训练为文本和多尺度图像片段生成统一的语义嵌入，从而通过点积相似度来衡量它们的相关性。在此基础上，引入了一种跨模态多尺度对齐模块，通过最大化互信息和信息瓶颈原则，在构建文本与多尺度图像片段之间跨模态相关性矩阵的基础上，采用Top-K选择策略识别最相关的图像区域。此外，设计了一种尺度感知融合网络，通过为图像区域分配自适应权重，将高度相关的多尺度图像特征与全局文本特征进行集成，基于它们的语义重要性和跨模态相关性。所提方法已在两个真实世界数据集上进行了广泛评估，实验结果表明，它在谣言检测方面的性能显著优于现有最先进的方法，突显了其有效性和实际应用潜力。 

---
# MoNaCo: More Natural and Complex Questions for Reasoning Across Dozens of Documents 

**Title (ZH)**: MoNaCo: 更自然且复杂的跨文档推理问题生成 

**Authors**: Tomer Wolfson, Harsh Trivedi, Mor Geva, Yoav Goldberg, Dan Roth, Tushar Khot, Ashish Sabharwal, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2508.11133)  

**Abstract**: Large language models (LLMs) are emerging as a go-to tool for querying information. However, current LLM benchmarks rarely feature natural questions that are both information-seeking as well as genuinely time-consuming for humans. To address this gap we introduce MoNaCo, a benchmark of 1,315 natural and complex questions that require dozens, and at times hundreds, of intermediate steps to solve -- far more than any existing QA benchmark. To build MoNaCo, we developed a decomposed annotation pipeline to elicit and manually answer natural time-consuming questions at scale. Frontier LLMs evaluated on MoNaCo achieve at most 61.2% F1, hampered by low recall and hallucinations. Our results underscore the need for reasoning models that better handle the complexity and sheer breadth of real-world information-seeking questions -- with MoNaCo providing an effective resource for tracking such progress. The MONACO benchmark, codebase, prompts and models predictions are publicly available at: this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）正逐渐成为查询信息的首选工具。然而，当前的LLM基准数据集很少包含既寻求信息又对人类来说真正耗时的自然问题。为解决这一问题，我们引入了MoNaCo基准，该基准包含1,315个自然且复杂的需要几十步乃至上百步中间步骤才能解决的问题——远超现有任何问答基准的数据复杂度。为构建MoNaCo，我们开发了一个分解注释流水线，以大规模生成和手动回答自然耗时的问题。在MoNaCo上进行评估的前沿LLMs最多只实现了61.2%的F1分数，受到召回率低和幻觉的影响。我们的结果强调了处理真实世界信息寻求问题的复杂性和广泛性所需的推理模型的需求——MoNaCo提供了跟踪这种进展的有效资源。MoNaCo基准、代码库、提示词和模型预测已在以下网址公开：this https URL。 

---
# Tabularis Formatus: Predictive Formatting for Tables 

**Title (ZH)**: Tabularis Formatus: 表格的预测性格式化 

**Authors**: Mukul Singh, José Cambronero, Sumit Gulwani, Vu Le, Gust Verbruggen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11121)  

**Abstract**: Spreadsheet manipulation software are widely used for data management and analysis of tabular data, yet the creation of conditional formatting (CF) rules remains a complex task requiring technical knowledge and experience with specific platforms. In this paper we present TaFo, a neuro-symbolic approach to generating CF suggestions for tables, addressing common challenges such as user unawareness, difficulty in rule creation, and inadequate user interfaces. TaFo takes inspiration from component based synthesis systems and extends them with semantic knowledge of language models and a diversity preserving rule this http URL previous methods focused on structural formatting, TaFo uniquely incorporates value-based formatting, automatically learning both the rule trigger and the associated visual formatting properties for CF rules. By removing the dependency on user specification used by existing techniques in the form of formatted examples or natural language instruction, TaFo makes formatting completely predictive and automated for the user. To evaluate TaFo, we use a corpus of 1.8 Million public workbooks with CF and manual formatting. We compare TaFo against a diverse set of symbolic and neural systems designed for or adapted for the task of table formatting. Our results show that TaFo generates more accurate, diverse and complete formatting suggestions than current systems and outperforms these by 15.6\%--26.5\% on matching user added ground truth rules in tables. 

**Abstract (ZH)**: 基于神经符号方法的表格条件格式建议生成系统：TaFo 

---
# Quantization through Piecewise-Affine Regularization: Optimization and Statistical Guarantees 

**Title (ZH)**: 分段线性正则化下的量化：优化与统计保证 

**Authors**: Jianhao Ma, Lin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11112)  

**Abstract**: Optimization problems over discrete or quantized variables are very challenging in general due to the combinatorial nature of their search space. Piecewise-affine regularization (PAR) provides a flexible modeling and computational framework for quantization based on continuous optimization. In this work, we focus on the setting of supervised learning and investigate the theoretical foundations of PAR from optimization and statistical perspectives. First, we show that in the overparameterized regime, where the number of parameters exceeds the number of samples, every critical point of the PAR-regularized loss function exhibits a high degree of quantization. Second, we derive closed-form proximal mappings for various (convex, quasi-convex, and non-convex) PARs and show how to solve PAR-regularized problems using the proximal gradient method, its accelerated variant, and the Alternating Direction Method of Multipliers. Third, we study statistical guarantees of PAR-regularized linear regression problems; specifically, we can approximate classical formulations of $\ell_1$-, squared $\ell_2$-, and nonconvex regularizations using PAR and obtain similar statistical guarantees with quantized solutions. 

**Abstract (ZH)**: 离散或量化变量上的优化问题由于其搜索空间的组合性质通常具有很大的挑战性。分段线性正则化（PAR）为基于连续优化的量化提供了一种灵活的建模和计算框架。在本文中，我们专注于监督学习的设置，并从优化和统计的角度研究PAR的理论基础。首先，我们证明在参数过参数化的情况下，即参数数量超过样本数量时，PAR正则化损失函数的每个临界点都表现出高度的量化。其次，我们推导出各种（凸的、拟凸的和非凸的）PAR的闭形式近邻映射，并展示如何使用近邻梯度方法、其加速变体以及交替方向乘子法来求解PAR正则化问题。第三，我们研究PAR正则化线性回归问题的统计保证；具体而言，我们可以使用PAR近似经典的$\ell_1$、平方$\ell_2$和非凸正则化，并获得与量化解决方案类似的统计保证。 

---
# Diffusion is a code repair operator and generator 

**Title (ZH)**: 扩散是一种代码修复操作符和生成器。 

**Authors**: Mukul Singh, Gust Verbruggen, Vu Le, Sumit Gulwani  

**Link**: [PDF](https://arxiv.org/pdf/2508.11110)  

**Abstract**: Code diffusion models generate code by iteratively removing noise from the latent representation of a code snippet. During later steps of the diffusion process, when the code snippet has almost converged, differences between discrete representations of these snippets look like last-mile repairs applied to broken or incomplete code. We evaluate the extent to which this resemblance can be exploited to leverage pre-trained code diffusion models for the problem of last-mile repair by considering two applications with significant potential. First, we can leverage the diffusion model for last-mile repair by adding noise to a broken code snippet and resuming the diffusion process. Second, we can leverage the diffusion model to generate arbitrary amount of training data for last-mile repair tasks (that are computationally more efficient) by sampling an intermediate program (input) and the final program (output) from the diffusion process. We perform experiments on 3 domains (Python, Excel and PowerShell) to evaluate applications, as well as analyze properties. 

**Abstract (ZH)**: 代码扩散模型通过迭代去除代码片段潜在表示中的噪音来生成代码。在扩散过程的后期步骤中，当代码片段几乎收敛时，这些片段的离散表示之间的差异看起来像是对损坏或不完整代码所做的最后一英里修复。我们通过考虑两种具有重大潜力的应用来评估这种相似性可以被利用的程度。首先，可以通过向损坏的代码片段添加噪音并继续扩散过程来利用扩散模型进行最后一英里的修复。其次，可以通过从扩散过程中采样中间程序（输入）和最终程序（输出）来利用扩散模型生成任意数量的最后一英里修复任务的训练数据（这些任务在计算上更高效）。我们在3个领域（Python、Excel和PowerShell）上进行了实验，以评估应用并分析其属性。 

---
# Utilizing Vision-Language Models as Action Models for Intent Recognition and Assistance 

**Title (ZH)**: 利用视觉-语言模型作为意图识别和辅助的动作模型 

**Authors**: Cesar Alan Contreras, Manolis Chiou, Alireza Rastegarpanah, Michal Szulik, Rustam Stolkin  

**Link**: [PDF](https://arxiv.org/pdf/2508.11093)  

**Abstract**: Human-robot collaboration requires robots to quickly infer user intent, provide transparent reasoning, and assist users in achieving their goals. Our recent work introduced GUIDER, our framework for inferring navigation and manipulation intents. We propose augmenting GUIDER with a vision-language model (VLM) and a text-only language model (LLM) to form a semantic prior that filters objects and locations based on the mission prompt. A vision pipeline (YOLO for object detection and the Segment Anything Model for instance segmentation) feeds candidate object crops into the VLM, which scores their relevance given an operator prompt; in addition, the list of detected object labels is ranked by a text-only LLM. These scores weight the existing navigation and manipulation layers of GUIDER, selecting context-relevant targets while suppressing unrelated objects. Once the combined belief exceeds a threshold, autonomy changes occur, enabling the robot to navigate to the desired area and retrieve the desired object, while adapting to any changes in the operator's intent. Future work will evaluate the system on Isaac Sim using a Franka Emika arm on a Ridgeback base, with a focus on real-time assistance. 

**Abstract (ZH)**: 人类与机器人协作要求机器人快速推断用户意图、提供透明的推理过程，并协助用户实现其目标。我们近期的工作介绍了一种用于推断导航和操作意图的GUIDER框架。我们提议将GUIDER与视觉-语言模型（VLM）和纯文本语言模型（LLM）结合起来，形成一种语义先验，根据任务提示过滤物体和位置。视觉管道（使用YOLO进行对象检测和使用Segment Anything Model进行实例分割）将候选对象区域输入VLM，根据操作员提示对它们的相关性进行评分；此外，检测到的对象标签列表将由纯文本LLM进行排序。这些评分会加权GUIDER现有的导航和操作层，选择与上下文相关的目标并抑制无关的物体。一旦结合后的信念超过阈值，自主性变化就会发生，从而使机器人能够导航到目标区域并检索目标物体，同时适应操作员意图的任何变化。未来的工作将使用Isaac Sim平台和Franka Emika手臂搭载Ridgeback底盘进行系统评估，重点在于实时协助。 

---
# Compressive Meta-Learning 

**Title (ZH)**: 压缩元学习 

**Authors**: Daniel Mas Montserrat, David Bonet, Maria Perera, Xavier Giró-i-Nieto, Alexander G. Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.11090)  

**Abstract**: The rapid expansion in the size of new datasets has created a need for fast and efficient parameter-learning techniques. Compressive learning is a framework that enables efficient processing by using random, non-linear features to project large-scale databases onto compact, information-preserving representations whose dimensionality is independent of the number of samples and can be easily stored, transferred, and processed. These database-level summaries are then used to decode parameters of interest from the underlying data distribution without requiring access to the original samples, offering an efficient and privacy-friendly learning framework. However, both the encoding and decoding techniques are typically randomized and data-independent, failing to exploit the underlying structure of the data. In this work, we propose a framework that meta-learns both the encoding and decoding stages of compressive learning methods by using neural networks that provide faster and more accurate systems than the current state-of-the-art approaches. To demonstrate the potential of the presented Compressive Meta-Learning framework, we explore multiple applications -- including neural network-based compressive PCA, compressive ridge regression, compressive k-means, and autoencoders. 

**Abstract (ZH)**: 新的数据集规模迅速扩张催生了快速高效参数学习技术的需求。压缩学习是一种框架，通过使用随机的非线性特征将大规模数据库投影到维度与样本数量无关且可轻松存储、传输和处理的信息保留表示上。这些数据库级别的摘要随后用于从潜在的数据分布中解码感兴趣的参数，而无需访问原始样本，从而提供了一个高效且隐私友好的学习框架。然而，编码和解码技术通常随机化且与数据独立，未能利用数据的潜在结构。在本文中，我们提出了一种框架，通过使用神经网络来元学习压缩学习方法的编码和解码阶段，从而比当前最先进的方法提供更快更准确的系统。为了展示所提出的压缩元学习框架的潜力，我们探索了多个应用，包括基于神经网络的压缩PCA、压缩岭回归、压缩K-means和自编码器。 

---
# LD-LAudio-V1: Video-to-Long-Form-Audio Generation Extension with Dual Lightweight Adapters 

**Title (ZH)**: LD-LAudio-V1: 基于双轻量级适配器的视频到长音频生成扩展 

**Authors**: Haomin Zhang, Kristin Qi, Shuxin Yang, Zihao Chen, Chaofan Ding, Xinhan Di  

**Link**: [PDF](https://arxiv.org/pdf/2508.11074)  

**Abstract**: Generating high-quality and temporally synchronized audio from video content is essential for video editing and post-production tasks, enabling the creation of semantically aligned audio for silent videos. However, most existing approaches focus on short-form audio generation for video segments under 10 seconds or rely on noisy datasets for long-form video-to-audio zsynthesis. To address these limitations, we introduce LD-LAudio-V1, an extension of state-of-the-art video-to-audio models and it incorporates dual lightweight adapters to enable long-form audio generation. In addition, we release a clean and human-annotated video-to-audio dataset that contains pure sound effects without noise or artifacts. Our method significantly reduces splicing artifacts and temporal inconsistencies while maintaining computational efficiency. Compared to direct fine-tuning with short training videos, LD-LAudio-V1 achieves significant improvements across multiple metrics: $FD_{\text{passt}}$ 450.00 $\rightarrow$ 327.29 (+27.27%), $FD_{\text{panns}}$ 34.88 $\rightarrow$ 22.68 (+34.98%), $FD_{\text{vgg}}$ 3.75 $\rightarrow$ 1.28 (+65.87%), $KL_{\text{panns}}$ 2.49 $\rightarrow$ 2.07 (+16.87%), $KL_{\text{passt}}$ 1.78 $\rightarrow$ 1.53 (+14.04%), $IS_{\text{panns}}$ 4.17 $\rightarrow$ 4.30 (+3.12%), $IB_{\text{score}}$ 0.25 $\rightarrow$ 0.28 (+12.00%), $Energy\Delta10\text{ms}$ 0.3013 $\rightarrow$ 0.1349 (+55.23%), $Energy\Delta10\text{ms(this http URL)}$ 0.0531 $\rightarrow$ 0.0288 (+45.76%), and $Sem.\,Rel.$ 2.73 $\rightarrow$ 3.28 (+20.15%). Our dataset aims to facilitate further research in long-form video-to-audio generation and is available at this https URL. 

**Abstract (ZH)**: 从视频生成高质量且时间上同步的音频对于视频编辑和后期制作任务至关重要，能够用于创建无声视频的语义对齐音频。然而，大多数现有方法专注于生成长度不超过10秒的视频片段中的短音频，或者依赖嘈杂的数据集进行长视频到音频的合成。为解决这些局限性，我们引入了LD-LAudio-V1，这是一种扩展现有的先进视频到音频模型的方法，并且它结合了双轻量级适配器以实现长视频音频的生成。此外，我们还发布了一个人工标注且纯净的视频到音频数据集，其中包含纯声音效果而无噪声或伪影。我们的方法在保持计算效率的同时显著减少了拼接伪影和时间不一致性。与直接使用短训练视频进行微调相比，LD-LAudio-V1 在多个指标上实现了显著改进：$FD_{\text{passt}}$ 450.00 $\rightarrow$ 327.29 (+27.27%)，$FD_{\text{panns}}$ 34.88 $\rightarrow$ 22.68 (+34.98%)，$FD_{\text{vgg}}$ 3.75 $\rightarrow$ 1.28 (+65.87%)，$KL_{\text{panns}}$ 2.49 $\rightarrow$ 2.07 (+16.87%)，$KL_{\text{passt}}$ 1.78 $\rightarrow$ 1.53 (+14.04%)，$IS_{\text{panns}}$ 4.17 $\rightarrow$ 4.30 (+3.12%)，$IB_{\text{score}}$ 0.25 $\rightarrow$ 0.28 (+12.00%)，$Energy\Delta10\text{ms}$ 0.3013 $\rightarrow$ 0.1349 (+55.23%)，$Energy\Delta10\text{ms(this http URL)}$ 0.0531 $\rightarrow$ 0.0288 (+45.76%)，和 $Sem.\,Rel.$ 2.73 $\rightarrow$ 3.28 (+20.15%)。我们的数据集旨在促进长视频到音频生成的研究，并可在以下链接获取：this https URL。 

---
# AI That Helps Us Help Each Other: A Proactive System for Scaffolding Mentor-Novice Collaboration in Entrepreneurship Coaching 

**Title (ZH)**: 帮助我们相互帮助的AI：一种促进导师与学徒在创业指导中合作的前瞻性支撑系统 

**Authors**: Evey Jiaxin Huang, Matthew Easterday, Elizabeth Gerber  

**Link**: [PDF](https://arxiv.org/pdf/2508.11052)  

**Abstract**: Entrepreneurship requires navigating open-ended, ill-defined problems: identifying risks, challenging assumptions, and making strategic decisions under deep uncertainty. Novice founders often struggle with these metacognitive demands, while mentors face limited time and visibility to provide tailored support. We present a human-AI coaching system that combines a domain-specific cognitive model of entrepreneurial risk with a large language model (LLM) to proactively scaffold both novice and mentor thinking. The system proactively poses diagnostic questions that challenge novices' thinking and helps both novices and mentors plan for more focused and emotionally attuned meetings. Critically, mentors can inspect and modify the underlying cognitive model, shaping the logic of the system to reflect their evolving needs. Through an exploratory field deployment, we found that using the system supported novice metacognition, helped mentors plan emotionally attuned strategies, and improved meeting depth, intentionality, and focus--while also surfaced key tensions around trust, misdiagnosis, and expectations of AI. We contribute design principles for proactive AI systems that scaffold metacognition and human-human collaboration in complex, ill-defined domains, offering implications for similar domains like healthcare, education, and knowledge work. 

**Abstract (ZH)**: 创业要求驾驭开放性、不明确的问题：识别风险、挑战假设并在深刻不确定性下做出战略决策。新手创始人往往难以应对这些元认知需求，而导师则面临时间有限和缺乏透明度的限制，难以提供定制化的支持。我们提出了一种结合特定领域创业风险管理的认知模型和大型语言模型（LLM）的人机辅导系统，以主动式支架新手和导师的思考。该系统主动提出诊断性问题，挑战新手的思考，并帮助两者计划更聚焦和情感契合的会议。关键的是，导师可以检查和修改底层的认知模型，从而塑造系统的逻辑以反映其不断变化的需求。通过探索性的现场部署，我们发现使用该系统支持了新手的元认知，帮助导师规划情感契合的策略，并改善了会议的深度、意图性和专注度——同时也揭示了关于信任、误诊和对AI的期望的关键冲突。我们贡献了主动式AI系统的设设计原则，这些系统旨在在复杂和不明确的领域支撑元认知和人与人之间的协作，并为类似领域如医疗保健、教育和知识工作提供了启示。 

---
# Learning with Confidence 

**Title (ZH)**: 具有信心的學習 

**Authors**: Oliver Ethan Richardson  

**Link**: [PDF](https://arxiv.org/pdf/2508.11037)  

**Abstract**: We characterize a notion of confidence that arises in learning or updating beliefs: the amount of trust one has in incoming information and its impact on the belief state. This learner's confidence can be used alongside (and is easily mistaken for) probability or likelihood, but it is fundamentally a different concept -- one that captures many familiar concepts in the literature, including learning rates and number of training epochs, Shafer's weight of evidence, and Kalman gain. We formally axiomatize what it means to learn with confidence, give two canonical ways of measuring confidence on a continuum, and prove that confidence can always be represented in this way. Under additional assumptions, we derive more compact representations of confidence-based learning in terms of vector fields and loss functions. These representations induce an extended language of compound "parallel" observations. We characterize Bayes Rule as the special case of an optimizing learner whose loss representation is a linear expectation. 

**Abstract (ZH)**: 我们刻画一种在学习或更新信念中出现的信心概念：对 incoming 信息的信任程度及其对信念状态的影响。这种学习者的信心可以与概率或似然性一起使用（并且容易被混淆），但本质上是不同的概念——它涵盖了文献中许多熟悉的概念，包括学习速率、训练周期数、Shafers 的证据权重以及卡尔曼增益。我们形式化地公理化了信心学习的意义，给出了信心在连续统一体上度量的两种典型方式，并证明信心总可以如此表示。在附加假设下，我们推导出基于信心学习的更紧凑表示，这些表示诱导了一个扩展的“并行”观测语言。我们将贝叶斯规则视为使损失表示为线性期望的优化学习者的特殊情况。 

---
# Note on Selection Bias in Observational Estimates of Algorithmic Progress 

**Title (ZH)**: 关于观察估计算法进展中的选择偏差笔记 

**Authors**: Parker Whitfill  

**Link**: [PDF](https://arxiv.org/pdf/2508.11033)  

**Abstract**: Ho et. al (2024) is an interesting paper that attempts to estimate the degree of algorithmic progress from language models. They collect observational data on language models' loss and compute over time, and argue that as time has passed, language models' algorithmic efficiency has been rising. That is, the loss achieved for fixed compute has been dropping over time. In this note, I want to raise one potential methodological problem with the estimation strategy. Intuitively, if part of algorithmic quality is latent, and compute choices are endogenous to algorithmic quality, then resulting estimates of algorithmic quality will be biased. 

**Abstract (ZH)**: Ho等（2024）：算法进步度估算的潜在方法论问题 

---
# Risk-Based Prognostics and Health Management 

**Title (ZH)**: 基于风险的预测性维护与健康管理 

**Authors**: John W. Sheppard  

**Link**: [PDF](https://arxiv.org/pdf/2508.11031)  

**Abstract**: It is often the case that risk assessment and prognostics are viewed as related but separate tasks. This chapter describes a risk-based approach to prognostics that seeks to provide a tighter coupling between risk assessment and fault prediction. We show how this can be achieved using the continuous-time Bayesian network as the underlying modeling framework. Furthermore, we provide an overview of the techniques that are available to derive these models from data and show how they might be used in practice to achieve tasks like decision support and performance-based logistics. This work is intended to provide an overview of the recent developments related to risk-based prognostics, and we hope that it will serve as a tutorial of sorts that will assist others in adopting these techniques. 

**Abstract (ZH)**: 风险评估导向的 prognostics 方法：风险评估与故障预测的紧密耦合及其在决策支持和绩效物流中的应用 

---
# Zono-Conformal Prediction: Zonotope-Based Uncertainty Quantification for Regression and Classification Tasks 

**Title (ZH)**: zonotope- conformal预测：基于zonotope的回归和分类任务中的不确定性量化 

**Authors**: Laura Lützow, Michael Eichelbeck, Mykel J. Kochenderfer, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2508.11025)  

**Abstract**: Conformal prediction is a popular uncertainty quantification method that augments a base predictor with prediction sets with statistically valid coverage guarantees. However, current methods are often computationally expensive and data-intensive, as they require constructing an uncertainty model before calibration. Moreover, existing approaches typically represent the prediction sets with intervals, which limits their ability to capture dependencies in multi-dimensional outputs. We address these limitations by introducing zono-conformal prediction, a novel approach inspired by interval predictor models and reachset-conformant identification that constructs prediction zonotopes with assured coverage. By placing zonotopic uncertainty sets directly into the model of the base predictor, zono-conformal predictors can be identified via a single, data-efficient linear program. While we can apply zono-conformal prediction to arbitrary nonlinear base predictors, we focus on feed-forward neural networks in this work. Aside from regression tasks, we also construct optimal zono-conformal predictors in classification settings where the output of an uncertain predictor is a set of possible classes. We provide probabilistic coverage guarantees and present methods for detecting outliers in the identification data. In extensive numerical experiments, we show that zono-conformal predictors are less conservative than interval predictor models and standard conformal prediction methods, while achieving a similar coverage over the test data. 

**Abstract (ZH)**: 自适应预测是一种流行的方法，用于扩展基础预测器以提供具有统计上 合有效覆盖保证的预测集。尽管当前方法通常计算密集且耗时，而且需要构建预测误差的校准过程。此外，现有的方法通常用使用区间来表示预测集，这 限制了对高维度输出中依赖性的捕捉能力。我们通过引入 zono-conformal 预测解决了这些限制，这是一种受基于区间预测模型和区域识别方法启发的方法。通过将 zon 型型不确定性集直接集成到基础预测器模型中，可 zono-con former 预测器可以通过单个的数据高效线性程序来识别。虽然我们可以将 zono-con caled 预测应用于任意非线性基础预测器上在本文中我们专注于将之应用于前馈神经网络。除了在现有的任务中我们还在分类设置中构建制最优的 zono-con 得预测器其中输出是一个可能的分类。我们给出了概率性的覆盖保证并 幯示了检测识别中的异常值的方法。在广泛的数值实验中我们表明 zono-con 络预测器比更保守于基于区间预测模型和 标准的预测误差方法 on 在优点是具有相近的覆盖能力 on on。 

---
# Beyond the Rosetta Stone: Unification Forces in Generalization Dynamics 

**Title (ZH)**: 超越罗塞塔石碑：统一力在泛化动态中的作用 

**Authors**: Carter Blum, Katja Filipova, Ann Yuan, Asma Ghandeharioun, Julian Zimmert, Fred Zhang, Jessica Hoffmann, Tal Linzen, Martin Wattenberg, Lucas Dixon, Mor Geva  

**Link**: [PDF](https://arxiv.org/pdf/2508.11017)  

**Abstract**: Large language models (LLMs) struggle with cross-lingual knowledge transfer: they hallucinate when asked in one language about facts expressed in a different language during training. This work introduces a controlled setting to study the causes and dynamics of this phenomenon by training small Transformer models from scratch on synthetic multilingual datasets. We identify a learning phase wherein a model develops either separate or unified representations of the same facts across languages, and show that unification is essential for cross-lingual transfer. We also show that the degree of unification depends on mutual information between facts and training data language, and on how easy it is to extract that language. Based on these insights, we develop methods to modulate the level of cross-lingual transfer by manipulating data distribution and tokenization, and we introduce metrics and visualizations to formally characterize their effects on unification. Our work shows how controlled settings can shed light on pre-training dynamics and suggests new directions for improving cross-lingual transfer in LLMs. 

**Abstract (ZH)**: 大规模语言模型在跨语言知识迁移中表现挣扎：它们在用一种语言询问不同语言中表述的事实时会产生错觉。本研究引入了一个受控环境，通过从零训练小的Transformer模型来研究这一现象的原因和动态。我们识别了一个学习阶段，在此阶段，模型会发展出针对同一事实的独立或统一的语言表示，并表明统一是跨语言迁移的关键。我们还表明，统一的程度取决于事实与训练数据语言之间的互信息，以及提取该语言的难易程度。基于这些洞见，我们开发了方法来通过调整数据分布和分词来调控跨语言迁移的程度，并引入了度量和可视化方法以正式表征这些对统一的影响。我们的研究显示了如何通过受控环境揭示预训练动态，并指出改进大规模语言模型跨语言迁移的新方向。 

---
# CURE: Critical-Token-Guided Re-concatenation for Entropy-collapse Prevention 

**Title (ZH)**: CURE: 关键词引导的重新拼接以防止熵坍缩 

**Authors**: Qingbin Li, Rongkun Xue, Jie Wang, Ming Zhou, Zhi Li, Xiaofeng Ji, Yongqi Wang, Miao Liu, Zheming Yang, Minghui Qiu, Jing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11016)  

**Abstract**: Recent advances in Reinforcement Learning with Verified Reward (RLVR) have driven the emergence of more sophisticated cognitive behaviors in large language models (LLMs), thereby enhancing their reasoning capabilities. However, in prior RLVR pipelines, the repeated use of static initial-state sampling drawn exactly from the dataset distribution during each sampling phase produced overly deterministic, low diversity model behavior, which manifested as rapid entropy collapse and hindered sustained performance gains during prolonged training. To address this issue, we introduce CURE (Critical-token-gUided Re concatenation for Entropy-collapse prevention), a two-stage framework that balances exploration and exploitation. Specifically, in the first stage, to deliberately steer the model toward novel yet coherent contexts, we re-generate at high-entropy critical tokens and jointly optimize the original and the branched trajectories. The further comparison with vanilla DAPO shows that the regeneration process achieves a better performance on math reasoning tasks while sustaining a high-level entropy degree for exploration. In the second stage, we continue training with static initial-state sampling by DAPO, intentionally placing the model in a familiar state to gradually strengthen exploitation. Extensive experiments on Qwen-2.5-Math-7B show that, compared to other RLVR methods, CURE achieves a 5% performance gain across six math benchmarks, establishing state-of-the-art performance in both entropy and accuracy. A series of experiments further validate the effectiveness of our approach. Code is available at this https URL. 

**Abstract (ZH)**: 最近在验证奖励强化学习（RLVR）方面的进展推动了大型语言模型（LLMs）更复杂认知行为的出现，从而增强了其推理能力。然而，在先前的RLVR管道中，每次采样阶段都反复使用精确来自数据集分布的静态初始状态采样，导致了过度确定性和低多样性模型行为，表现为快速熵坍缩，并阻碍了长时间训练期间的持续性能提升。为解决这一问题，我们引入了CURE（关键令牌引导的重新组合以防止熵坍缩），这是一种平衡探索与利用的两阶段框架。具体而言，在第一阶段，为了引导模型朝新颖但连贯的上下文前进，我们重新生成高熵的关键令牌，并联合优化原始轨迹和分支轨迹。与vanilla DAPO的进一步对比显示，重新生成过程在数学推理任务上取得了更好的性能，同时保持了高水平的熵度以促进探索。在第二阶段，我们继续使用DAPO的静态初始状态采样进行训练，有意将模型置于熟悉的狀態，逐步强化利用。在Qwen-2.5-Math-7B上进行的广泛实验表明，与其它RLVR方法相比，CURE在六个数学基准上实现了5%的性能提升，同时在熵和准确性上均达到了最先进的性能。一系列实验进一步验证了我们方法的有效性。代码可在以下链接获取。 

---
# Deep Learning-Based Automated Segmentation of Uterine Myomas 

**Title (ZH)**: 基于深度学习的子宫肌瘤自动分割 

**Authors**: Tausifa Jan Saleem, Mohammad Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2508.11010)  

**Abstract**: Uterine fibroids (myomas) are the most common benign tumors of the female reproductive system, particularly among women of childbearing age. With a prevalence exceeding 70%, they pose a significant burden on female reproductive health. Clinical symptoms such as abnormal uterine bleeding, infertility, pelvic pain, and pressure-related discomfort play a crucial role in guiding treatment decisions, which are largely influenced by the size, number, and anatomical location of the fibroids. Magnetic Resonance Imaging (MRI) is a non-invasive and highly accurate imaging modality commonly used by clinicians for the diagnosis of uterine fibroids. Segmenting uterine fibroids requires a precise assessment of both the uterus and fibroids on MRI scans, including measurements of volume, shape, and spatial location. However, this process is labor intensive and time consuming and subjected to variability due to intra- and inter-expert differences at both pre- and post-treatment stages. As a result, there is a critical need for an accurate and automated segmentation method for uterine fibroids. In recent years, deep learning algorithms have shown re-markable improvements in medical image segmentation, outperforming traditional methods. These approaches offer the potential for fully automated segmentation. Several studies have explored the use of deep learning models to achieve automated segmentation of uterine fibroids. However, most of the previous work has been conducted using private datasets, which poses challenges for validation and comparison between studies. In this study, we leverage the publicly available Uterine Myoma MRI Dataset (UMD) to establish a baseline for automated segmentation of uterine fibroids, enabling standardized evaluation and facilitating future research in this domain. 

**Abstract (ZH)**: 子宫肌瘤（纤维瘤）是女性生殖系统中最常见的良性肿瘤，尤其在育龄妇女中更为常见。其发病率超过70%，对女性生殖健康构成了显著的负担。临床症状如异常子宫出血、不孕、盆腔疼痛和压迫感相关的不适，在指导治疗决策中起着关键作用，这些决策很大程度上受到肌瘤大小、数量和解剖位置的影响。磁共振成像（MRI）是一种常用的无创且高度准确的成像技术，用于子宫肌瘤的诊断。子宫肌瘤的分割需要对MRI扫描中的子宫和肌瘤进行精确的评估，包括体积、形状和空间位置的测量。然而，这一过程耗时且主观性高，受到预治疗和治疗后专家间差异的影响。因此，需要一种准确且自动化的分割方法来解决这些问题。近年来，深度学习算法在医学图像分割方面取得了显著进步，超越了传统方法。这些方法提供了实现完全自动化分割的潜力。多项研究探讨了使用深度学习模型实现子宫肌瘤自动化分割的可能性。然而，大多数先前的工作都基于私有数据集，这给跨研究的验证和比较带来了挑战。在本研究中，我们利用公开的子宫肌瘤MRI数据集（UMD）建立子宫肌瘤自动化分割的基线，以实现标准化评估并促进该领域的未来研究。 

---
# SproutBench: A Benchmark for Safe and Ethical Large Language Models for Youth 

**Title (ZH)**: SproutBench: 青少年安全且伦理合规的大语言模型基准 

**Authors**: Wenpeng Xing, Lanyi Wei, Haixiao Hu, Rongchang Li, Mohan Li, Changting Lin, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.11009)  

**Abstract**: The rapid proliferation of large language models (LLMs) in applications targeting children and adolescents necessitates a fundamental reassessment of prevailing AI safety frameworks, which are largely tailored to adult users and neglect the distinct developmental vulnerabilities of minors. This paper highlights key deficiencies in existing LLM safety benchmarks, including their inadequate coverage of age-specific cognitive, emotional, and social risks spanning early childhood (ages 0--6), middle childhood (7--12), and adolescence (13--18). To bridge these gaps, we introduce SproutBench, an innovative evaluation suite comprising 1,283 developmentally grounded adversarial prompts designed to probe risks such as emotional dependency, privacy violations, and imitation of hazardous behaviors. Through rigorous empirical evaluation of 47 diverse LLMs, we uncover substantial safety vulnerabilities, corroborated by robust inter-dimensional correlations (e.g., between Safety and Risk Prevention) and a notable inverse relationship between Interactivity and Age Appropriateness. These insights yield practical guidelines for advancing child-centric AI design and deployment. 

**Abstract (ZH)**: 面向儿童和青少年的应用激增促使我们对现有的AI安全框架进行根本性的重新评估，这些框架主要针对成人用户，忽视了未成年人的独特发育脆弱性。本文指出现有大语言模型安全基准的关键缺陷，包括其对早期童年（0-6岁）、学龄期（7-12岁）和青春期（13-18岁）年龄特异性认知、情感和社会风险的不足覆盖。为弥补这些差距，我们提出了SproutBench，这是一个创新的评估套件，包括1,283个基于发育阶段的对抗性提示，旨在探究情感依赖、隐私侵犯和模仿危险行为等风险。通过47种不同LLM的严格实证评估，我们发现了一系列重大的安全漏洞，这些漏洞通过跨维度相关性（例如，安全性和风险预防之间）得到了验证，并且交互性和年龄适宜性之间存在值得注意的负相关关系。这些洞察提供了面向儿童的AI设计和部署的实际指导。 

---
# Match & Choose: Model Selection Framework for Fine-tuning Text-to-Image Diffusion Models 

**Title (ZH)**: 匹配与选择：文本到图像扩散模型微调的模型选择框架 

**Authors**: Basile Lewandowski, Robert Birke, Lydia Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10993)  

**Abstract**: Text-to-image (T2I) models based on diffusion and transformer architectures advance rapidly. They are often pretrained on large corpora, and openly shared on a model platform, such as HuggingFace. Users can then build up AI applications, e.g., generating media contents, by adopting pretrained T2I models and fine-tuning them on the target dataset. While public pretrained T2I models facilitate the democratization of the models, users face a new challenge: which model can be best fine-tuned based on the target data domain? Model selection is well addressed in classification tasks, but little is known in (pretrained) T2I models and their performance indication on the target domain. In this paper, we propose the first model selection framework, M&C, which enables users to efficiently choose a pretrained T2I model from a model platform without exhaustively fine-tuning them all on the target dataset. The core of M&C is a matching graph, which consists of: (i) nodes of available models and profiled datasets, and (ii) edges of model-data and data-data pairs capturing the fine-tuning performance and data similarity, respectively. We then build a model that, based on the inputs of model/data feature, and, critically, the graph embedding feature, extracted from the matching graph, predicts the model achieving the best quality after fine-tuning for the target domain. We evaluate M&C on choosing across ten T2I models for 32 datasets against three baselines. Our results show that M&C successfully predicts the best model for fine-tuning in 61.3% of the cases and a closely performing model for the rest. 

**Abstract (ZH)**: 基于扩散和变换器架构的文本到图像（T2I）模型快速发展。它们通常在大型语料库上进行预训练，并且在诸如HuggingFace的模型平台上公开共享。用户可以采用这些预训练的T2I模型并在目标数据集上进行微调来构建AI应用程序，例如生成媒体内容。虽然公开的预训练T2I模型促进了模型的民主化使用，但用户面临新的挑战：在目标数据域中哪些模型最适合进行微调？在分类任务中，模型选择问题得到了很好的解决，但在（预训练的）T2I模型及其在目标域上的性能指示方面还知之甚少。在本文中，我们提出了第一个模型选择框架M&C，该框架使用户能够在不全部在目标数据集上进行耗时微调的情况下，从模型平台上高效地选择一个预训练的T2I模型。M&C的核心是一个匹配图，该图包括：（i）可用模型和配置数据集的节点，以及（ii）模型-数据和数据-数据配对边，分别捕捉微调性能和数据相似性。然后构建一个模型，基于输入的模型/数据特征以及从匹配图中提取的图嵌入特征，预测在目标域上微调后的最佳性能模型。我们针对十个T2I模型和32个数据集与三个基线进行选择评估。结果表明，M&C在61.3%的情况下成功预测了最适合微调的模型，而对于其余情况则预测了表现相近的模型。 

---
# MCP-Guard: A Defense Framework for Model Context Protocol Integrity in Large Language Model Applications 

**Title (ZH)**: MCP-Guard: 大型语言模型应用中模型上下文协议完整性的防御框架 

**Authors**: Wenpeng Xing, Zhonghao Qi, Yupeng Qin, Yilin Li, Caini Chang, Jiahui Yu, Changting Lin, Zhenzhen Xie, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.10991)  

**Abstract**: The integration of Large Language Models (LLMs) with external tools via protocols such as the Model Context Protocol (MCP) introduces critical security vulnerabilities, including prompt injection, data exfiltration, and other threats. To counter these challenges, we propose MCP-Guard, a robust, layered defense architecture designed for LLM--tool interactions. MCP-Guard employs a three-stage detection pipeline that balances efficiency with accuracy: it progresses from lightweight static scanning for overt threats and a deep neural detector for semantic attacks, to our fine-tuned E5-based model achieves (96.01) accuracy in identifying adversarial prompts. Finally, a lightweight LLM arbitrator synthesizes these signals to deliver the final decision while minimizing false positives. To facilitate rigorous training and evaluation, we also introduce MCP-AttackBench, a comprehensive benchmark of over 70,000 samples. Sourced from public datasets and augmented by GPT-4, MCP-AttackBench simulates diverse, real-world attack vectors in the MCP format, providing a foundation for future research into securing LLM-tool ecosystems. 

**Abstract (ZH)**: LLMs与外部工具通过协议（如MRectified Context Protocol）集成的关键安全性挑战及CP-MCPuard：一种针对LLMs工具交互的稳健分分层次化防御架构 

---
# Not There Yet: Evaluating Vision Language Models in Simulating the Visual Perception of People with Low Vision 

**Title (ZH)**: 尚未达成：评估视觉语言模型在模拟低视力人士视觉感知方面的表现 

**Authors**: Rosiana Natalie, Wenqian Xu, Ruei-Che Chang, Rada Mihalcea, Anhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.10972)  

**Abstract**: Advances in vision language models (VLMs) have enabled the simulation of general human behavior through their reasoning and problem solving capabilities. However, prior research has not investigated such simulation capabilities in the accessibility domain. In this paper, we evaluate the extent to which VLMs can simulate the vision perception of low vision individuals when interpreting images. We first compile a benchmark dataset through a survey study with 40 low vision participants, collecting their brief and detailed vision information and both open-ended and multiple-choice image perception and recognition responses to up to 25 images. Using these responses, we construct prompts for VLMs (GPT-4o) to create simulated agents of each participant, varying the included information on vision information and example image responses. We evaluate the agreement between VLM-generated responses and participants' original answers. Our results indicate that VLMs tend to infer beyond the specified vision ability when given minimal prompts, resulting in low agreement (0.59). The agreement between the agent' and participants' responses remains low when only either the vision information (0.59) or example image responses (0.59) are provided, whereas a combination of both significantly increase the agreement (0.70, p < 0.0001). Notably, a single example combining both open-ended and multiple-choice responses, offers significant performance improvements over either alone (p < 0.0001), while additional examples provided minimal benefits (p > 0.05). 

**Abstract (ZH)**: 视觉语言模型在低视力视野感知模拟中的进展 

---
# Rule2Text: A Framework for Generating and Evaluating Natural Language Explanations of Knowledge Graph Rules 

**Title (ZH)**: Rule2Text：知识图规则自然语言解释生成与评估框架 

**Authors**: Nasim Shirvani-Mahdavi, Chengkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10971)  

**Abstract**: Knowledge graphs (KGs) can be enhanced through rule mining; however, the resulting logical rules are often difficult for humans to interpret due to their inherent complexity and the idiosyncratic labeling conventions of individual KGs. This work presents Rule2Text, a comprehensive framework that leverages large language models (LLMs) to generate natural language explanations for mined logical rules, thereby improving KG accessibility and usability. We conduct extensive experiments using multiple datasets, including Freebase variants (FB-CVT-REV, FB+CVT-REV, and FB15k-237) as well as the ogbl-biokg dataset, with rules mined using AMIE 3.5.1. We systematically evaluate several LLMs across a comprehensive range of prompting strategies, including zero-shot, few-shot, variable type incorporation, and Chain-of-Thought reasoning. To systematically assess models' performance, we conduct a human evaluation of generated explanations on correctness and clarity. To address evaluation scalability, we develop and validate an LLM-as-a-judge framework that demonstrates strong agreement with human evaluators. Leveraging the best-performing model (Gemini 2.0 Flash), LLM judge, and human-in-the-loop feedback, we construct high-quality ground truth datasets, which we use to fine-tune the open-source Zephyr model. Our results demonstrate significant improvements in explanation quality after fine-tuning, with particularly strong gains in the domain-specific dataset. Additionally, we integrate a type inference module to support KGs lacking explicit type information. All code and data are publicly available at this https URL. 

**Abstract (ZH)**: 知识图谱中的规则可以通过规则挖掘进行增强，然而，生成的逻辑规则往往由于其固有的复杂性和个体知识图谱的特殊标记惯例而难以供人类解读。本文提出了一个综合框架Rule2Text，该框架利用大规模语言模型（LLMs）自动生成挖掘出的逻辑规则的自然语言解释，从而提高知识图谱的可访问性和可用性。我们使用多种数据集进行了广泛的实验，包括Freebase变体（FB-CVT-REV、FB+CVT-REV和FB15k-237）以及ogbl-biokg数据集，使用AMIE 3.5.1进行规则挖掘。我们系统地评估了多种大规模语言模型在广泛策略下的表现，包括零样本、少样本、类型信息变体整合和chain-of-thought推理。为系统性评估模型性能，我们进行了生成解释的正确性和清晰度的人类评估。为解决评估量化的挑战，我们开发并验证了一个大规模语言模型作为评委的框架，该框架与人类评估者表现出强烈的一致性。利用性能最佳的模型（Gemini 2.0 Flash）、LLM评委和人工反馈，我们构建了高质量的基准数据集，用于微调开源Zephyr模型。我们的结果显示，在微调后解释质量有显著提升，特别是在特定领域数据集中的提升尤为明显。此外，我们整合了一个类型推断模块以支持缺乏显式类型信息的知识图谱。所有代码和数据均在此网址公开。 

---
# Retro-Expert: Collaborative Reasoning for Interpretable Retrosynthesis 

**Title (ZH)**: Retro-Expert: 共享推理实现可解释逆合成反应 

**Authors**: Xinyi Li, Sai Wang, Yutian Lin, Yu Wu, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10967)  

**Abstract**: Retrosynthesis prediction aims to infer the reactant molecule based on a given product molecule, which is a fundamental task in chemical synthesis. However, existing models rely on static pattern-matching paradigm, which limits their ability to perform effective logic decision-making, leading to black-box decision-making. Building on this, we propose Retro-Expert, an interpretable retrosynthesis framework that performs collaborative reasoning by combining the complementary reasoning strengths of Large Language Models and specialized models via reinforcement learning. It outputs natural language explanations grounded in chemical logic through three components: (1) specialized models perform shallow reasoning to construct high-quality chemical decision space, (2) LLM-driven critical reasoning to generate predictions and corresponding interpretable reasoning path, and (3) reinforcement learning optimizing interpretable decision policy. Experiments show that Retro-Expert not only surpasses both LLM-based and specialized models across different metrics but also provides expert-aligned explanations that bridge the gap between AI predictions and actionable chemical insights. 

**Abstract (ZH)**: retrosynthesis 预测旨在根据给定的目标分子推断反应物分子，这是化学合成中的一个基本任务。然而，现有的模型依赖于静态模式匹配范式，这限制了它们进行有效的逻辑决策能力，导致黑盒决策。在此基础上，我们提出了一种可解释的 retrosynthesis 框架 Retro-Expert，通过结合大型语言模型和专门模型的互补推理优势，并利用强化学习进行协作推理。该框架通过三个组成部分输出基于化学逻辑的自然语言解释：（1）专门模型执行浅层推理以构建高质量的化学决策空间，（2）由大型语言模型驱动的关键推理以生成预测及其相应的可解释推理路径，（3）利用强化学习优化可解释的决策策略。实验证明，Retro-Expert 不仅在不同指标上超越了基于大型语言模型和专门模型，还提供了与专家一致的解释，填补了人工智能预测与可操作化学洞察之间的差距。 

---
# ORBIT: An Object Property Reasoning Benchmark for Visual Inference Tasks 

**Title (ZH)**: ORBIT: 一种用于视觉推理任务的对象属性推理基准 

**Authors**: Abhishek Kolari, Mohammadhossein Khojasteh, Yifan Jiang, Floris den Hengst, Filip Ilievski  

**Link**: [PDF](https://arxiv.org/pdf/2508.10956)  

**Abstract**: While vision-language models (VLMs) have made remarkable progress on many popular visual question answering (VQA) benchmarks, it remains unclear whether they abstract and reason over depicted objects. Inspired by human object categorisation, object property reasoning involves identifying and recognising low-level details and higher-level abstractions. While current VQA benchmarks consider a limited set of object property attributes like size, they typically blend perception and reasoning, and lack representativeness in terms of reasoning and image categories. To this end, we introduce a systematic evaluation framework with images of three representative types, three reasoning levels of increasing complexity, and four object property dimensions driven by prior work on commonsense reasoning. We develop a procedure to instantiate this benchmark into ORBIT, a multi-level reasoning VQA benchmark for object properties comprising 360 images paired with a total of 1,080 count-based questions. Experiments with 12 state-of-the-art VLMs in zero-shot settings reveal significant limitations compared to humans, with the best-performing model only reaching 40\% accuracy. VLMs struggle particularly with realistic (photographic) images, counterfactual reasoning about physical and functional properties, and higher counts. ORBIT points to the need to develop methods for scalable benchmarking, generalize annotation guidelines, and explore additional reasoning VLMs. We make the ORBIT benchmark and the experimental code available to support such endeavors. 

**Abstract (ZH)**: 基于视觉-语言模型的对象属性推理综合评估框架ORBIT 

---
# Towards Efficient Prompt-based Continual Learning in Distributed Medical AI 

**Title (ZH)**: 基于提示的分布式医疗AI高效连续学习研究 

**Authors**: Gyutae Oh, Jitae Shin  

**Link**: [PDF](https://arxiv.org/pdf/2508.10954)  

**Abstract**: Modern AI models achieve state-of-the-art performance with large-scale, high-quality datasets; however, ethical, social, and institutional constraints in the medical domain severely restrict data sharing, rendering centralized learning nearly impossible. Each institution must incrementally update models using only local data. Traditional training overfits new samples and suffers from catastrophic forgetting, losing previously acquired knowledge. Medical data distributions also shift due to varying diagnostic equipment and demographics. Although continual learning (CL) has advanced, most methods address natural images, leaving medical-domain-specific CL underexplored. We propose a prompt-based continual learning (PCL) approach featuring a unified prompt pool with a minimal expansion strategy: by expanding and freezing a subset of prompts, our method reduces computational overhead, and a novel regularization term balances retention and adaptation. Experiments on three diabetic retinopathy datasets Aptos2019, LI2019, and Diabetic Retinopathy Detection show our model improves final classification accuracy by at least 10% and F1-score by 9 points over state-of-the-art approaches while lowering inference cost. We anticipate this study will drive sustainable medical AI advances, enabling real-time diagnosis, patient monitoring, and telemedicine applications in distributed healthcare. Code will be released upon acceptance 

**Abstract (ZH)**: 基于提示的持续学习在医疗领域的研究：糖尿病视网膜病变数据集上的应用 

---
# Apriel-Nemotron-15B-Thinker 

**Title (ZH)**: April-Nemotron-15B-Thinker 

**Authors**: Shruthan Radhakrishna, Soham Parikh, Gopal Sarda, Anil Turkkan, Quaizar Vohra, Raymond Li, Dhruv Jhamb, Kelechi Ogueji, Aanjaneya Shukla, Oluwanifemi Bamgbose, Toby Liang, Luke Kumar, Oleksiy Ostapenko, Shiva Krishna Reddy Malay, Aman Tiwari, Tara Bogavelli, Vikas Yadav, Jash Mehta, Saloni Mittal, Akshay Kalkunte, Pulkit Pattnaik, Khalil Slimi, Anirudh Sreeram, Jishnu Nair, Akintunde Oladipo, Shashank Maiya, Khyati Mahajan, Rishabh Maheshwary, Masoud Hashemi, Sai Rajeswar Mudumba, Sathwik Tejaswi Madhusudhan, Torsten Scholak, Sebastien Paquet, Sagar Davasam, Srinivas Sunkara  

**Link**: [PDF](https://arxiv.org/pdf/2508.10948)  

**Abstract**: While large language models (LLMs) have achieved remarkable reasoning capabilities across domains like code, math and other enterprise tasks, their significant memory and computational costs often preclude their use in practical enterprise settings. To this end, we introduce Apriel-Nemotron-15B-Thinker, a 15-billion parameter model in the ServiceNow Apriel SLM series that achieves performance against medium sized state-of-the-art models such as o1-mini, QWQ32B, and EXAONE-Deep-32B while maintaining only half the memory footprint of those alternatives. Apriel-Nemotron-15B-Thinker model is trained in a four stage training pipeline including 1) Base Model upscaling, 2) Continual Pre-training 3) Supervised Fine-tuning (SFT) and 4) Reinforcement Learning using GRPO. Comprehensive evaluations across a diverse suite of benchmarks consistently demonstrate that our Apriel-Nemotron-15B-Thinker model matches or exceeds the performance of its 32-billion parameter counterparts, despite being less than half their size. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）在代码、数学和其他企业任务等领域实现了卓越的推理能力，但其显著的内存和计算成本往往使其难以在实际企业环境中使用。为解决这一问题，我们引入了ServiceNow Apriel SLM系列中的Apriel-Nemotron-15B-Thinker模型，该模型拥有150亿参数，在基准测试中表现出色，与320亿参数的竞争模型o1-mini、QWQ32B和EXAONE-Deep-32B相比，其内存占用仅为后者的一半。Apriel-Nemotron-15B-Thinker模型的训练管道包括四阶段流程：1）基础模型扩展、2）持续预训练、3）监督微调（SFT）以及4）基于GRPO的强化学习。跨多种基准测试的全面评估一致表明，尽管Apriel-Nemotron-15B-Thinker模型的规模仅为后者的一半，但其性能却与之相当或更优。 

---
# Modeling and Detecting Company Risks from News: A Case Study in Bloomberg News 

**Title (ZH)**: 基于 Bloomberg 新闻的公司风险建模与检测：案例研究넣﹁
user
A Multi-Task Learning Framework for Server Failure Prediction: A Case Study on Alibaba Cloud。保持原句式， the titleeline。 

**Authors**: Jiaxin Pei, Soumya Vadlamannati, Liang-Kang Huang, Daniel Preotiuc-Pietro, Xinyu Hua  

**Link**: [PDF](https://arxiv.org/pdf/2508.10927)  

**Abstract**: Identifying risks associated with a company is important to investors and the well-being of the overall financial market. In this study, we build a computational framework to automatically extract company risk factors from news articles. Our newly proposed schema comprises seven distinct aspects, such as supply chain, regulations, and competitions. We sample and annotate 744 news articles and benchmark various machine learning models. While large language models have achieved huge progress in various types of NLP tasks, our experiment shows that zero-shot and few-shot prompting state-of-the-art LLMs (e.g. LLaMA-2) can only achieve moderate to low performances in identifying risk factors. And fine-tuned pre-trained language models are performing better on most of the risk factors. Using this model, we analyze over 277K Bloomberg news articles and demonstrate that identifying risk factors from news could provide extensive insight into the operations of companies and industries. 

**Abstract (ZH)**: 识别与公司相关的风险对于投资者和整体金融市场福祉至关重要。本研究构建了一个计算框架，以自动从新闻文章中提取公司风险因素。我们提出的新方案包括七个不同的方面，如供应链、监管和竞争等。我们采样并标注了744篇新闻文章，并比较了多种机器学习模型。尽管大型语言模型在各种NLP任务中取得了巨大进展，但我们的实验显示，零样本和少样本提示最新的大规模语言模型（如LLaMA-2）在识别风险因素方面的表现仅达到中等到较低的水平。而微调的预训练语言模型在多数风险因素上的表现更好。通过该模型，我们分析了超过27.7万篇彭博新闻文章，并证明从新闻中识别风险因素能够为公司和行业的运营提供广泛洞见。 

---
# gpt-oss-120b & gpt-oss-20b Model Card 

**Title (ZH)**: GPT-oss-120B & GPT-oss-20B 模型卡片 

**Authors**: OpenAI, Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K. Arora, Yu Bai, Bowen Baker, Haiming Bao, Boaz Barak, Ally Bennett, Tyler Bertao, Nivedita Brett, Eugene Brevdo, Greg Brockman, Sebastien Bubeck, Che Chang, Kai Chen, Mark Chen, Enoch Cheung, Aidan Clark, Dan Cook, Marat Dukhan, Casey Dvorak, Kevin Fives, Vlad Fomenko, Timur Garipov, Kristian Georgiev, Mia Glaese, Tarun Gogineni, Adam Goucher, Lukas Gross, Katia Gil Guzman, John Hallman, Jackie Hehir, Johannes Heidecke, Alec Helyar, Haitang Hu, Romain Huet, Jacob Huh, Saachi Jain, Zach Johnson, Chris Koch, Irina Kofman, Dominik Kundel, Jason Kwon, Volodymyr Kyrylov, Elaine Ya Le, Guillaume Leclerc, James Park Lennon, Scott Lessans, Mario Lezcano-Casado, Yuanzhi Li, Zhuohan Li, Ji Lin, Jordan Liss, Lily, Jiancheng Liu, Kevin Lu, Chris Lu, Zoran Martinovic, Lindsay McCallum, Josh McGrath, Scott McKinney, Aidan McLaughlin, Song Mei, Steve Mostovoy, Tong Mu, Gideon Myles, Alexander Neitz, Alex Nichol, Jakub Pachocki, Alex Paino, Dana Palmie, Ashley Pantuliano, Giambattista Parascandolo, Jongsoo Park, Leher Pathak, Carolina Paz, Ludovic Peran, Dmitry Pimenov, Michelle Pokrass, Elizabeth Proehl, Huida Qiu, Gaby Raila, Filippo Raso, Hongyu Ren, Kimmy Richardson, David Robinson, Bob Rotsted, Hadi Salman, Suvansh Sanjeev, Max Schwarzer, D. Sculley, Harshit Sikchi, Kendal Simon, Karan Singhal, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.10925)  

**Abstract**: We present gpt-oss-120b and gpt-oss-20b, two open-weight reasoning models that push the frontier of accuracy and inference cost. The models use an efficient mixture-of-expert transformer architecture and are trained using large-scale distillation and reinforcement learning. We optimize the models to have strong agentic capabilities (deep research browsing, python tool use, and support for developer-provided functions), all while using a rendered chat format that enables clear instruction following and role delineation. Both models achieve strong results on benchmarks ranging from mathematics, coding, and safety. We release the model weights, inference implementations, tool environments, and tokenizers under an Apache 2.0 license to enable broad use and further research. 

**Abstract (ZH)**: 我们展示了两个开放重量推理模型GPT-OSS-120B和GPT-OSS-20B，它们推动了准确性和推理成本的边界。这些模型采用高效的专家混合变压器架构，并通过大规模的知识蒸馏和强化学习进行训练。我们优化模型以具备强大的代理能力（深度研究浏览、Python工具使用及开发者提供的功能支持），同时使用渲染的对话格式，以实现清晰的指令跟随和角色划分。这两种模型在涵盖数学、编程和安全性等多个基准测试中均取得了优异成果。我们以Apache 2.0许可证发布模型权重、推理实现、工具环境和分词器，以促进广泛使用和进一步研究。 

---
# Human-AI collaboration or obedient and often clueless AI in instruct, serve, repeat dynamics? 

**Title (ZH)**: 人类与人工智能的合作，还是在指令、服务、重复动态中盲目服从的人工智能？ 

**Authors**: Mohammed Saqr, Kamila Misiejuk, Sonsoles López-Pernas  

**Link**: [PDF](https://arxiv.org/pdf/2508.10919)  

**Abstract**: While research on human-AI collaboration exists, it mainly examined language learning and used traditional counting methods with little attention to evolution and dynamics of collaboration on cognitively demanding tasks. This study examines human-AI interactions while solving a complex problem. Student-AI interactions were qualitatively coded and analyzed with transition network analysis, sequence analysis and partial correlation networks as well as comparison of frequencies using chi-square and Person-residual shaded Mosaic plots to map interaction patterns, their evolution, and their relationship to problem complexity and student performance. Findings reveal a dominant Instructive pattern with interactions characterized by iterative ordering rather than collaborative negotiation. Oftentimes, students engaged in long threads that showed misalignment between their prompts and AI output that exemplified a lack of synergy that challenges the prevailing assumptions about LLMs as collaborative partners. We also found no significant correlations between assignment complexity, prompt length, and student grades suggesting a lack of cognitive depth, or effect of problem difficulty. Our study indicates that the current LLMs, optimized for instruction-following rather than cognitive partnership, compound their capability to act as cognitively stimulating or aligned collaborators. Implications for designing AI systems that prioritize cognitive alignment and collaboration are discussed. 

**Abstract (ZH)**: 人类与AI协作研究：复杂问题解决中的交互模式及其演变 

---
# Managing the unexpected: Operator behavioural data and its value in predicting correct alarm responses 

**Title (ZH)**: 管理意外情况：操作员行为数据及其在预测正确报警响应中的价值 

**Authors**: Chidera W. Amazu, Joseph Mietkiewicz, Ammar N. Abbas, Gabriele Baldissone, Davide Fissore, Micaela Demichela, Anders L. Madsen, Maria Chiara Leva  

**Link**: [PDF](https://arxiv.org/pdf/2508.10917)  

**Abstract**: Data from psychophysiological measures can offer new insight into control room operators' behaviour, cognition, and mental workload status. This can be particularly helpful when combined with appraisal of capacity to respond to possible critical plant conditions (i.e. critical alarms response scenarios). However, wearable physiological measurement tools such as eye tracking and EEG caps can be perceived as intrusive and not suitable for usage in daily operations. Therefore, this article examines the potential of using real-time data from process and operator-system interactions during abnormal scenarios that can be recorded and retrieved from the distributed control system's historian or process log, and their capacity to provide insight into operator behavior and predict their response outcomes, without intruding on daily tasks. Data for this study were obtained from a design of experiment using a formaldehyde production plant simulator and four human-in-the-loop experimental support configurations. A comparison between the different configurations in terms of both behaviour and performance is presented in this paper. A step-wise logistic regression and a Bayesian network models were used to achieve this objective. The results identified some predictive metrics and the paper discuss their value as precursor or predictor of overall system performance in alarm response scenarios. Knowledge of relevant and predictive behavioural metrics accessible in real time can better equip decision-makers to predict outcomes and provide timely support measures for operators. 

**Abstract (ZH)**: 基于异常场景中操作员-系统交互实时数据的生理psychophysiological测量在报警响应中的潜力及预测价值 

---
# Multimodal Quantitative Measures for Multiparty Behaviour Evaluation 

**Title (ZH)**: 多模态定量指标用于多方行为评估 

**Authors**: Ojas Shirekar, Wim Pouw, Chenxu Hao, Vrushank Phadnis, Thabo Beeler, Chirag Raman  

**Link**: [PDF](https://arxiv.org/pdf/2508.10916)  

**Abstract**: Digital humans are emerging as autonomous agents in multiparty interactions, yet existing evaluation metrics largely ignore contextual coordination dynamics. We introduce a unified, intervention-driven framework for objective assessment of multiparty social behaviour in skeletal motion data, spanning three complementary dimensions: (1) synchrony via Cross-Recurrence Quantification Analysis, (2) temporal alignment via Multiscale Empirical Mode Decompositionbased Beat Consistency, and (3) structural similarity via Soft Dynamic Time Warping. We validate metric sensitivity through three theory-driven perturbations -- gesture kinematic dampening, uniform speech-gesture delays, and prosodic pitch-variance reduction-applied to $\approx 145$ 30-second thin slices of group interactions from the DnD dataset. Mixed-effects analyses reveal predictable, joint-independent shifts: dampening increases CRQA determinism and reduces beat consistency, delays weaken cross-participant coupling, and pitch flattening elevates F0 Soft-DTW costs. A complementary perception study ($N=27$) compares judgments of full-video and skeleton-only renderings to quantify representation effects. Our three measures deliver orthogonal insights into spatial structure, timing alignment, and behavioural variability. Thereby forming a robust toolkit for evaluating and refining socially intelligent agents. Code available on \href{this https URL}{GitHub}. 

**Abstract (ZH)**: 数字人类在多方互动中 emergence 为自主代理，现有评估指标大多忽视了背景协调动态。我们提出了一种统一的干预驱动框架，用于骨骼运动数据中多方社会行为的客观评估，涵盖三个互补维度：（1）同步性通过交叉复发量化分析，（2）时间对齐通过多尺度经验模式分解基于节奏一致性，（3）结构相似性通过软动态时间扭曲。通过三种理论驱动的扰动——手势运动阻尼、均匀的语音-手势延迟以及音调变异性减少——验证了指标的敏感性，应用于 DnD 数据集的约 145 个 30 秒的群体互动片段。混合效应分析揭示了可预测且独立于关节的转变：阻尼增加 CRQA 决定度并降低节奏一致性，延迟削弱跨参与者耦合，音调扁平化提高 F0 软动态时间扭曲成本。一项互补的感知研究（参与者数量为 27）比较了完整视频和仅骨架渲染的判断，以量化表现形式的影响。我们的三个测量指标提供了关于空间结构、时间对齐和行为变异性的独立见解。从而形成一个 robust 工具包，用于评估和优化社会智能代理。代码可在 GitHub 上获得。 

---
# SDSNN: A Single-Timestep Spiking Neural Network with Self-Dropping Neuron and Bayesian Optimization 

**Title (ZH)**: SDSN: 基于自消除神经元和贝叶斯优化的一时时间发放神经网络 

**Authors**: Changqing Xu, Buxuan Song, Yi Liu, Xinfang Liao, Wenbin Zheng, Yintang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10913)  

**Abstract**: Spiking Neural Networks (SNNs), as an emerging biologically inspired computational model, demonstrate significant energy efficiency advantages due to their event-driven information processing mechanism. Compared to traditional Artificial Neural Networks (ANNs), SNNs transmit information through discrete spike signals, which substantially reduces computational energy consumption through their sparse encoding approach. However, the multi-timestep computation model significantly increases inference latency and energy, limiting the applicability of SNNs in edge computing scenarios. We propose a single-timestep SNN, which enhances accuracy and reduces computational energy consumption in a single timestep by optimizing spike generation and temporal parameters. We design a Self-Dropping Neuron mechanism, which enhances information-carrying capacity through dynamic threshold adjustment and selective spike suppression. Furthermore, we employ Bayesian optimization to globally search for time parameters and obtain an efficient inference mode with a single time step. Experimental results on the Fashion-MNIST, CIFAR-10, and CIFAR-100 datasets demonstrate that, compared to traditional multi-timestep SNNs employing the Leaky Integrate-and-Fire (LIF) model, our method achieves classification accuracies of 93.72%, 92.20%, and 69.45%, respectively, using only single-timestep spikes, while maintaining comparable or even superior accuracy. Additionally, it reduces energy consumption by 56%, 21%, and 22%, respectively. 

**Abstract (ZH)**: 基于单时步模型的自适应阈值神经元刺激发SNNs 

---
# FLUID: Flow-Latent Unified Integration via Token Distillation for Expert Specialization in Multimodal Learning 

**Title (ZH)**: FLUID: 流-潜在统一集成 via 令牌精炼以实现多模态学习中的专家专业化 

**Authors**: Van Duc Cuong, Ta Dinh Tam, Tran Duc Chinh, Nguyen Thi Hanh  

**Link**: [PDF](https://arxiv.org/pdf/2508.07264)  

**Abstract**: Multimodal classification requires robust integration of visual and textual signals, yet common fusion strategies are brittle and vulnerable to modality-specific noise. In this paper, we present \textsc{FLUID}-Flow-Latent Unified Integration via Token Distillation for Expert Specialization, a principled token-level pipeline that improves cross-modal robustness and scalability. \textsc{FLUID} contributes three core elements: (1) \emph{Q-transforms}, learnable query tokens that distill and retain salient token-level features from modality-specific backbones; (2) a two-stage fusion scheme that enforces cross-modal consistency via contrastive alignment and then performs adaptive, task-aware fusion through a gating mechanism and a \emph{Q-bottleneck} that selectively compresses information for downstream reasoning; and (3) a lightweight, load-balanced Mixture-of-Experts at prediction time that enables efficient specialization to diverse semantic patterns. Extensive experiments demonstrate that \textsc{FLUID} attains \(91\%\) accuracy on the GLAMI-1M benchmark, significantly outperforming prior baselines and exhibiting strong resilience to label noise, long-tail class imbalance, and semantic heterogeneity. Targeted ablation studies corroborate both the individual and synergistic benefits of the proposed components, positioning \textsc{FLUID} as a scalable, noise-resilient solution for multimodal product classification. 

**Abstract (ZH)**: Multimodal分类要求视觉和文本信号的 robust 综合，但常见的融合策略对模态特定的噪声容易脆弱。本文提出了一种基于标记蒸馏的 \textsc{FLUID}-Flow-Latent 统一综合框架，这是一种原理上的标记级流程，提高了跨模态的稳健性和可扩展性。\textsc{FLUID} 贡献了三个核心元素：（1）\emph{Q-transforms}，可学习的查询标记，用于从模态特定骨干中提取并保留关键的标记级特征；（2）两阶段融合方案，通过对比对齐确保跨模态一致性，然后通过门控机制和\emph{Q-bottleneck}进行自适应、任务感知的融合，选择性地压缩信息以供下游推理；（3）预测时轻量级、负载平衡的 Experts Mixtures，使模型能够高效地针对多样化的语义模式进行专业化。广泛实验表明，\textsc{FLUID} 在 GLAMI-1M 基准上的准确率为 91%，显著优于先前baseline，并且对标签噪声、长尾类别不平衡和语义异质性表现出强大的抗噪性。针对性的消融研究证实了所提组件的个体和协同效益，将\textsc{FLUID} 定位为一种面向 multimodal 产品分类的大规模、抗噪解决方案。 

---
# Generalized Similarity U: A Non-parametric Test of Association Based on Similarity 

**Title (ZH)**: 广义相似性U：基于相似性的非参数关联检验 

**Authors**: Changshuai Wei, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/1801.01220)  

**Abstract**: Second generation sequencing technologies are being increasingly used for genetic association studies, where the main research interest is to identify sets of genetic variants that contribute to various phenotype. The phenotype can be univariate disease status, multivariate responses and even high-dimensional outcomes. Considering the genotype and phenotype as two complex objects, this also poses a general statistical problem of testing association between complex objects. We here proposed a similarity-based test, generalized similarity U (GSU), that can test the association between complex objects. We first studied the theoretical properties of the test in a general setting and then focused on the application of the test to sequencing association studies. Based on theoretical analysis, we proposed to use Laplacian kernel based similarity for GSU to boost power and enhance robustness. Through simulation, we found that GSU did have advantages over existing methods in terms of power and robustness. We further performed a whole genome sequencing (WGS) scan for Alzherimer Disease Neuroimaging Initiative (ADNI) data, identifying three genes, APOE, APOC1 and TOMM40, associated with imaging phenotype. We developed a C++ package for analysis of whole genome sequencing data using GSU. The source codes can be downloaded at this https URL. 

**Abstract (ZH)**: 二代测序技术在基因关联研究中的应用：基于相似性的测试方法GSU及其在阿尔茨海默病神经影像学倡议数据中的应用 

---
# Trees Assembling Mann Whitney Approach for Detecting Genome-wide Joint Association among Low Marginal Effect loci 

**Title (ZH)**: 树木组装曼尼 Whitney 方法检测低边际效应位点的全基因组联合关联 

**Authors**: Changshuai Wei, Daniel J. Schaid, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/1505.01206)  

**Abstract**: Common complex diseases are likely influenced by the interplay of hundreds, or even thousands, of genetic variants. Converging evidence shows that genetic variants with low marginal effects (LME) play an important role in disease development. Despite their potential significance, discovering LME genetic variants and assessing their joint association on high dimensional data (e.g., genome wide association studies) remain a great challenge. To facilitate joint association analysis among a large ensemble of LME genetic variants, we proposed a computationally efficient and powerful approach, which we call Trees Assembling Mann whitney (TAMW). Through simulation studies and an empirical data application, we found that TAMW outperformed multifactor dimensionality reduction (MDR) and the likelihood ratio based Mann whitney approach (LRMW) when the underlying complex disease involves multiple LME loci and their interactions. For instance, in a simulation with 20 interacting LME loci, TAMW attained a higher power (power=0.931) than both MDR (power=0.599) and LRMW (power=0.704). In an empirical study of 29 known Crohn's disease (CD) loci, TAMW also identified a stronger joint association with CD than those detected by MDR and LRMW. Finally, we applied TAMW to Wellcome Trust CD GWAS to conduct a genome wide analysis. The analysis of 459K single nucleotide polymorphisms was completed in 40 hours using parallel computing, and revealed a joint association predisposing to CD (p-value=2.763e-19). Further analysis of the newly discovered association suggested that 13 genes, such as ATG16L1 and LACC1, may play an important role in CD pathophysiological and etiological processes. 

**Abstract (ZH)**: 遗传变异的共同复杂性疾病联合关联分析：Trees Assembling Mann Whitney (TAMW) 方法的研究 

---
# A Weighted U Statistic for Genetic Association Analyses of Sequencing Data 

**Title (ZH)**: 加权U统计量在序列数据遗传关联分析中的应用 

**Authors**: Changshuai Wei, Ming Li, Zihuai He, Olga Vsevolozhskaya, Daniel J. Schaid, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/1505.01204)  

**Abstract**: With advancements in next generation sequencing technology, a massive amount of sequencing data are generated, offering a great opportunity to comprehensively investigate the role of rare variants in the genetic etiology of complex diseases. Nevertheless, this poses a great challenge for the statistical analysis of high-dimensional sequencing data. The association analyses based on traditional statistical methods suffer substantial power loss because of the low frequency of genetic variants and the extremely high dimensionality of the data. We developed a weighted U statistic, referred to as WU-seq, for the high-dimensional association analysis of sequencing data. Based on a non-parametric U statistic, WU-SEQ makes no assumption of the underlying disease model and phenotype distribution, and can be applied to a variety of phenotypes. Through simulation studies and an empirical study, we showed that WU-SEQ outperformed a commonly used SKAT method when the underlying assumptions were violated (e.g., the phenotype followed a heavy-tailed distribution). Even when the assumptions were satisfied, WU-SEQ still attained comparable performance to SKAT. Finally, we applied WU-seq to sequencing data from the Dallas Heart Study (DHS), and detected an association between ANGPTL 4 and very low density lipoprotein cholesterol. 

**Abstract (ZH)**: 随着下一代测序技术的发展，产生了大量的测序数据，为全面探索罕见变异在复杂疾病遗传病因中的作用提供了巨大机会。然而，这给高维测序数据的统计分析带来了巨大挑战。基于传统统计方法的关联分析因遗传变异频率低和数据的极高维度而遭受严重效能损失。我们开发了一种加权U统计量，称为WU-seq，用于测序数据的高维关联分析。基于非参数U统计量，WU-SEQ不假设底层疾病模型和表型分布，并可用于多种表型。通过模拟研究和实证研究，我们表明，当底层假设被违反时（例如，表型遵循重尾分布），WU-SEQ优于常用的SKAT方法。即使在假设成立时，WU-SEQ也能达到与SKAT相当的性能。最终，我们应用WU-seq分析了达拉斯心脏研究（DHS）的测序数据，并检测到ANGPTL 4与极低密度脂蛋白胆固醇之间的关联。 

---
# A Generalized Similarity U Test for Multivariate Analysis of Sequencing Data 

**Title (ZH)**: 泛化相似性U检验在序列数据多元分析中的应用 

**Authors**: Changshuai Wei, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/1505.01179)  

**Abstract**: Sequencing-based studies are emerging as a major tool for genetic association studies of complex diseases. These studies pose great challenges to the traditional statistical methods (e.g., single-locus analyses based on regression methods) because of the high-dimensionality of data and the low frequency of genetic variants. In addition, there is a great interest in biology and epidemiology to identify genetic risk factors contributed to multiple disease phenotypes. The multiple phenotypes can often follow different distributions, which violates the assumptions of most current methods. In this paper, we propose a generalized similarity U test, referred to as GSU. GSU is a similarity-based test and can handle high-dimensional genotypes and phenotypes. We studied the theoretical properties of GSU, and provided the efficient p-value calculation for association test as well as the sample size and power calculation for the study design. Through simulation, we found that GSU had advantages over existing methods in terms of power and robustness to phenotype distributions. Finally, we used GSU to perform a multivariate analysis of sequencing data in the Dallas Heart Study and identified a joint association of 4 genes with 5 metabolic related phenotypes. 

**Abstract (ZH)**: 基于测序的研究正逐渐成为复杂疾病遗传关联研究的重要工具。这些研究给传统的统计方法（如基于回归方法的单核苷酸分析）带来了巨大挑战，因为数据的高维度性和遗传变异的低频性。此外，在生物学和流行病学中，人们对识别 Contribution to 多个疾病表型的遗传风险因素表现出极大的兴趣。这些表型往往遵循不同的分布，这违反了当前大多数方法的基本假设。在本文中，我们提出了一种广义相似性U检验，称为GSU。GSU 是一种基于相似性的检验，可以处理高维度的基因型和表型。我们研究了 GSU 的理论性质，并提供了关联检验的高效p值计算方法以及研究设计中的样本量和功效计算。通过模拟，我们发现 GSU 在功效和表型分布稳健性方面优于现有方法。最后，我们在达拉斯心脏研究中使用 GSU 进行了多变量测序数据分析，并识别出4个基因与5个代谢相关表型的联合关联。 

---
# A weighted U statistic for association analysis considering genetic heterogeneity 

**Title (ZH)**: 考虑遗传异质性的加权U统计量用于关联分析 

**Authors**: Changshuai Wei, Robert C. Elston, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/1504.08319)  

**Abstract**: Converging evidence suggests that common complex diseases with the same or similar clinical manifestations could have different underlying genetic etiologies. While current research interests have shifted toward uncovering rare variants and structural variations predisposing to human diseases, the impact of heterogeneity in genetic studies of complex diseases has been largely overlooked. Most of the existing statistical methods assume the disease under investigation has a homogeneous genetic effect and could, therefore, have low power if the disease undergoes heterogeneous pathophysiological and etiological processes. In this paper, we propose a heterogeneity weighted U (HWU) method for association analyses considering genetic heterogeneity. HWU can be applied to various types of phenotypes (e.g., binary and continuous) and is computationally effcient for high- dimensional genetic data. Through simulations, we showed the advantage of HWU when the underlying genetic etiology of a disease was heterogeneous, as well as the robustness of HWU against different model assumptions (e.g., phenotype distributions). Using HWU, we conducted a genome-wide analysis of nicotine dependence from the Study of Addiction: Genetics and Environments (SAGE) dataset. The genome-wide analysis of nearly one million genetic markers took 7 hours, identifying heterogeneous effects of two new genes (i.e., CYP3A5 and IKBKB) on nicotine dependence. 

**Abstract (ZH)**: 不同临床表现的常见复杂疾病的遗传异质性证据正在不断汇聚。当前研究兴趣已转向揭示诱发人类疾病的风险罕见变异和结构变异，但遗传研究中的异质性影响已被广泛关注不足。现有的大多数统计方法假设所研究的疾病具有均质的遗传效应，因此在疾病经历异质的病理生理学和病因学过程时，可能会导致较低的统计功效。本文提出了一种考虑遗传异质性的异质性加权U（HWU）方法，用于关联分析。HWU可以应用于各种类型的表型（如二元和连续型），并且对于高维遗传数据具有计算效率。通过模拟实验，我们展示了当疾病的根本遗传病因异质时，HWU的优势，以及HWU在不同模型假设（如表型分布）下的稳健性。使用HWU，我们对来自Addiction: Genetics and Environments (AGE) 数据集的尼古丁依赖进行了全基因组分析。全基因组分析近一百万个遗传标记耗时7小时，鉴定出两种新的基因（即CYP3A5和IKBKB）在尼古丁依赖中的异质性效应。 

---
