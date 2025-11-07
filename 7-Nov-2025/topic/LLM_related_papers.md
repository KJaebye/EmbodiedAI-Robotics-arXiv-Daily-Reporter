# An LLM-based Framework for Human-Swarm Teaming Cognition in Disaster Search and Rescue 

**Title (ZH)**: 基于LLM的无人机群在灾害搜救中的人机协同认知框架 

**Authors**: Kailun Ji, Xiaoyu Hu, Xinyu Zhang, Jun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.04042)  

**Abstract**: Large-scale disaster Search And Rescue (SAR) operations are persistently challenged by complex terrain and disrupted communications. While Unmanned Aerial Vehicle (UAV) swarms offer a promising solution for tasks like wide-area search and supply delivery, yet their effective coordination places a significant cognitive burden on human operators. The core human-machine collaboration bottleneck lies in the ``intention-to-action gap'', which is an error-prone process of translating a high-level rescue objective into a low-level swarm command under high intensity and pressure. To bridge this gap, this study proposes a novel LLM-CRF system that leverages Large Language Models (LLMs) to model and augment human-swarm teaming cognition. The proposed framework initially captures the operator's intention through natural and multi-modal interactions with the device via voice or graphical annotations. It then employs the LLM as a cognitive engine to perform intention comprehension, hierarchical task decomposition, and mission planning for the UAV swarm. This closed-loop framework enables the swarm to act as a proactive partner, providing active feedback in real-time while reducing the need for manual monitoring and control, which considerably advances the efficacy of the SAR task. We evaluate the proposed framework in a simulated SAR scenario. Experimental results demonstrate that, compared to traditional order and command-based interfaces, the proposed LLM-driven approach reduced task completion time by approximately $64.2\%$ and improved task success rate by $7\%$. It also leads to a considerable reduction in subjective cognitive workload, with NASA-TLX scores dropping by $42.9\%$. This work establishes the potential of LLMs to create more intuitive and effective human-swarm collaborations in high-stakes scenarios. 

**Abstract (ZH)**: 大规模灾害搜索与救援（SAR）行动持续受到复杂地形和通信中断的挑战。虽然无人航空器（UAV）群能够在广泛区域搜索和物资交付方面提供一种前景广阔的方法，但其有效的协调却对人类操作员施加了显著的认知负担。人类与机器协作的核心瓶颈在于“意图到行动的差距”，这是一个在高强度和高压力环境下将高级救援目标转化为低级群组指令的易出错过程。为弥合这一差距，本研究提出了一种新颖的LLM-CRF系统，利用大型语言模型（LLMs）建模和增强人类-群组团队认知。所提出的框架首先通过语音或图形注释等自然和多模态交互方式捕捉操作员的意图。然后利用LLM作为认知引擎进行意图理解、分层任务分解和无人航空器群组的任务规划。闭环框架使群组能够成为积极的伙伴，在减少手动监控和控制需求的同时，提供实时反馈，极大地提升了SAR任务的有效性。我们在模拟的SAR情境中评估了所提出的框架。实验结果表明，与传统的基于命令和指令的界面相比，提出的LLM驱动方法将任务完成时间降低了约64.2%，任务成功率提高了7%，同时显著降低了主观认知负荷，NASA-TLX评分下降了42.9%。本研究确立了LLMs在高风险场景中创造更直观和有效的无人航空器-人类协作的潜力。 

---
# VeriCoT: Neuro-symbolic Chain-of-Thought Validation via Logical Consistency Checks 

**Title (ZH)**: VeriCoT: 通过逻辑一致性检查的神经符号链式思考验证 

**Authors**: Yu Feng, Nathaniel Weir, Kaj Bostrom, Sam Bayless, Darion Cassel, Sapana Chaudhary, Benjamin Kiesl-Reiter, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2511.04662)  

**Abstract**: LLMs can perform multi-step reasoning through Chain-of-Thought (CoT), but they cannot reliably verify their own logic. Even when they reach correct answers, the underlying reasoning may be flawed, undermining trust in high-stakes scenarios. To mitigate this issue, we introduce VeriCoT, a neuro-symbolic method that extracts and verifies formal logical arguments from CoT reasoning. VeriCoT formalizes each CoT reasoning step into first-order logic and identifies premises that ground the argument in source context, commonsense knowledge, or prior reasoning steps. The symbolic representation enables automated solvers to verify logical validity while the NL premises allow humans and systems to identify ungrounded or fallacious reasoning steps. Experiments on the ProofWriter, LegalBench, and BioASQ datasets show VeriCoT effectively identifies flawed reasoning, and serves as a strong predictor of final answer correctness. We also leverage VeriCoT's verification signal for (1) inference-time self-reflection, (2) supervised fine-tuning (SFT) on VeriCoT-distilled datasets and (3) preference fine-tuning (PFT) with direct preference optimization (DPO) using verification-based pairwise rewards, further improving reasoning validity and accuracy. 

**Abstract (ZH)**: LLMs可以通过链式思考（Chain-of-Thought，CoT）进行多步推理，但不能可靠地验证自己的逻辑。即使它们得到了正确的答案，背后的推理可能仍然存在缺陷，从而在高风险场景中削弱信任。为了缓解这一问题，我们提出了VeriCoT，这是一种神经符号方法，可以从CoT推理中提取并验证形式逻辑论证。VeriCoT将每个CoT推理步骤形式化为一阶逻辑，并识别将论证与源头上下文、常识知识或先前推理步骤联系起来的前提。符号表示使自动求解器能够验证逻辑有效性，而自然语言前提则允许人类和系统识别未接地或谬误的推理步骤。在ProofWriter、LegalBench和BioASQ数据集上的实验表明，VeriCoT有效地识别了不正确的推理，并作为最终答案正确性的强大预测指标。我们还利用VeriCoT的验证信号进行（1）推断时自我反思，（2）基于VeriCoT提炼数据集的监督微调，以及（3）使用基于验证的成对奖励的直接偏好优化进行偏好微调，从而进一步提高推理的有效性和准确性。 

---
# Large language models replicate and predict human cooperation across experiments in game theory 

**Title (ZH)**: 大型语言模型在博弈论实验中重现和预测人类的合作行为 

**Authors**: Andrea Cera Palatsi, Samuel Martin-Gutierrez, Ana S. Cardenal, Max Pellert  

**Link**: [PDF](https://arxiv.org/pdf/2511.04500)  

**Abstract**: Large language models (LLMs) are increasingly used both to make decisions in domains such as health, education and law, and to simulate human behavior. Yet how closely LLMs mirror actual human decision-making remains poorly understood. This gap is critical: misalignment could produce harmful outcomes in practical applications, while failure to replicate human behavior renders LLMs ineffective for social simulations. Here, we address this gap by developing a digital twin of game-theoretic experiments and introducing a systematic prompting and probing framework for machine-behavioral evaluation. Testing three open-source models (Llama, Mistral and Qwen), we find that Llama reproduces human cooperation patterns with high fidelity, capturing human deviations from rational choice theory, while Qwen aligns closely with Nash equilibrium predictions. Notably, we achieved population-level behavioral replication without persona-based prompting, simplifying the simulation process. Extending beyond the original human-tested games, we generate and preregister testable hypotheses for novel game configurations outside the original parameter grid. Our findings demonstrate that appropriately calibrated LLMs can replicate aggregate human behavioral patterns and enable systematic exploration of unexplored experimental spaces, offering a complementary approach to traditional research in the social and behavioral sciences that generates new empirical predictions about human social decision-making. 

**Abstract (ZH)**: 大型语言模型（LLMs）在健康、教育和法律等领域进行决策以及模拟人类行为方面的应用日益广泛。然而，LLMs与实际人类决策之间的相似程度仍然知之甚少。这一差距至关重要：失配可能导致实际应用中的不良后果，而无法再现人类行为则使LLMs在社会模拟中无效。在此，我们通过开发博弈论实验的数字孪生并引入机器行为评估的系统性提示和探索框架来填补这一差距。我们测试了三个开源模型（Llama、Mistral和Qwen），发现Llama以高度准确的方式再现了人类的合作模式，捕捉到了人类在理性选择理论之外的行为偏差，而Qwen与纳什均衡预测高度一致。值得注意的是，我们实现了在无需基于人设提示的情况下对群体级行为的再现，简化了模拟过程。我们进一步超越了原始的人类测试游戏，生成并预先注册了新的游戏配置的可测试假说，这些配置超出了原始参数网格的范围。我们的研究表明，适当地校准的LLMs可以再现人类群体行为模式，并能系统性地探索未开发的实验空间，为社会和行为科学的传统研究提供了一个补充方法，能够产生关于人类社会决策的新经验预测。 

---
# Beyond Shortest Path: Agentic Vehicular Routing with Semantic Context 

**Title (ZH)**: 超越最短路径：基于语义上下文的自主车辆路径规划 

**Authors**: Carnot Braun, Rafael O. Jarczewski, Gabriel U. Talasso, Leandro A. Villas, Allan M. de Souza  

**Link**: [PDF](https://arxiv.org/pdf/2511.04464)  

**Abstract**: Traditional vehicle routing systems efficiently optimize singular metrics like time or distance, and when considering multiple metrics, they need more processes to optimize . However, they lack the capability to interpret and integrate the complex, semantic, and dynamic contexts of human drivers, such as multi-step tasks, situational constraints, or urgent needs. This paper introduces and evaluates PAVe (Personalized Agentic Vehicular Routing), a hybrid agentic assistant designed to augment classical pathfinding algorithms with contextual reasoning. Our approach employs a Large Language Model (LLM) agent that operates on a candidate set of routes generated by a multi-objective (time, CO2) Dijkstra algorithm. The agent evaluates these options against user-provided tasks, preferences, and avoidance rules by leveraging a pre-processed geospatial cache of urban Points of Interest (POIs). In a benchmark of realistic urban scenarios, PAVe successfully used complex user intent into appropriate route modifications, achieving over 88% accuracy in its initial route selections with a local model. We conclude that combining classical routing algorithms with an LLM-based semantic reasoning layer is a robust and effective approach for creating personalized, adaptive, and scalable solutions for urban mobility optimization. 

**Abstract (ZH)**: 传统的车辆路由系统高效地优化单一指标如时间和距离，但在考虑多个指标时需要更多优化过程。然而，这些系统缺乏解释和整合人类驾驶员的复杂、语义化和动态上下文的能力，如多步骤任务、情况约束或紧急需求。本文介绍了并评估了PAVe（个性化代理车辆路由）这一结合了上下文推理的经典路径查找算法的混合代理助手。我们的方法使用了一个大型语言模型（LLM）代理，该代理基于一个多目标（时间、CO2）迪ijkstra算法生成的候选路径集进行操作。代理通过利用预处理的城市兴趣点（POI）地理空间缓存，评估这些选项与用户提供的任务、偏好和规避规则的匹配度。在现实城市场景的基准测试中，PAVe成功地将复杂的用户意图转化为合适的路线修改，初始路线选择的准确率达到88%以上，使用的是局部模型。我们得出结论，结合经典路由算法与基于LLM的语义推理层是一种稳健而有效的方法，用于创建个性化的、适应性强的和可扩展的城市交通优化解决方案。 

---
# The Peril of Preference: Why GRPO fails on Ordinal Rewards 

**Title (ZH)**: 偏好之险：为什么GRPO在序数奖励上失败 

**Authors**: Anisha Garg, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2511.04439)  

**Abstract**: Group-relative Policy Optimization's (GRPO) simplicity makes it highly desirable for adapting LLMs to become experts at specific tasks. But this simplicity also makes it ill-specified as we seek to enhance RL training with richer, non-binary feedback. When using ordinal rewards to give partial credit, GRPO's simplicity starts to hurt, as its group-average baseline often assigns a positive advantage to failed trajectories and reinforces incorrect behavior.
We introduce Correctness Relative Policy Optimization (CoRPO), a new formulation that solves this flaw. CoRPO uses an adaptive baseline that enforces a minimum quality threshold, ensuring failed solutions are never positively reinforced. Once the policy consistently meets this threshold, the baseline automatically transitions to a relative preference mode, pushing the model to find optimal solutions rather than just "acceptable" ones. We empirically validate CoRPO on a code verification task, where it demonstrates more stable convergence and better out-of-domain generalization.
This work represents a critical step in our broader research program to enable LLMs to learn genuinely new capabilities through reinforcement learning. We achieve this by enabling LLMs to learn from rich, multi-dimensional feedback - progressing from binary to ordinal rewards in this work, and onward to denser, per-step supervision. 

**Abstract (ZH)**: Correctness Relative Policy Optimization (CoRPO): Enabling LLMs to Learn Genuine New Capabilities Through Reinforcement Learning 

---
# Post-Training LLMs as Better Decision-Making Agents: A Regret-Minimization Approach 

**Title (ZH)**: 通过后悔最小化方法训练后的大语言模型：更好的决策代理 

**Authors**: Chanwoo Park, Ziyang Chen, Asuman Ozdaglar, Kaiqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04393)  

**Abstract**: Large language models (LLMs) are increasingly deployed as "agents" for decision-making (DM) in interactive and dynamic environments. Yet, since they were not originally designed for DM, recent studies show that LLMs can struggle even in basic online DM problems, failing to achieve low regret or an effective exploration-exploitation tradeoff. To address this, we introduce Iterative Regret-Minimization Fine-Tuning (Iterative RMFT), a post-training procedure that repeatedly distills low-regret decision trajectories back into the base model. At each iteration, the model rolls out multiple decision trajectories, selects the k-lowest regret ones, and fine-tunes itself on them. Unlike prior methods that (a) distill action sequences from known DM algorithms or (b) rely on manually crafted chain-of-thought templates, our approach leverages the regret metric to elicit the model's own DM ability and reasoning rationales. This reliance on model-generated reasoning avoids rigid output engineering and provides more flexible, natural-language training signals. Empirical results show that Iterative RMFT improves LLMs' DM performance across diverse models - from Transformers with numerical input/output, to open-weight LLMs, and advanced closed-weight models like GPT-4o mini. Its flexibility in output and reasoning formats enables generalization across tasks with varying horizons, action spaces, reward processes, and natural-language contexts. Finally, we provide theoretical insight showing that a single-layer Transformer under this paradigm can act as a no-regret learner in a simplified setting. Overall, Iterative RMFT offers a principled and general post-training framework for enhancing LLMs' decision-making capabilities. 

**Abstract (ZH)**: 基于迭代后悔最小化微调的大型语言模型决策能力提升方法 

---
# Monitor-Generate-Verify (MGV):Formalising Metacognitive Theory for Language Model Reasoning 

**Title (ZH)**: 监控-生成-验证 (MGV): 正式化元认知理论以优化语言模型推理 

**Authors**: Nick Oh, Fernand Gobet  

**Link**: [PDF](https://arxiv.org/pdf/2511.04341)  

**Abstract**: Test-time reasoning architectures such as those following the Generate-Verify paradigm -- where a model iteratively refines or verifies its own generated outputs -- prioritise generation and verification but exclude the monitoring processes that determine when and how reasoning should begin. This omission may contribute to the prefix dominance trap, in which models commit early to suboptimal reasoning paths and seldom recover, yielding roughly 20% accuracy loss. We address this architectural gap by formalising Flavell's and Nelson and Narens' metacognitive theories into computational specifications, proposing the Monitor-Generate-Verify (MGV) framework. MGV extends the Generate-Verify paradigm by adding explicit monitoring that captures metacognitive experiences (from difficulty assessments to confidence judgements) before generation begins and refines future monitoring through verification feedback. Though we present no empirical validation, this work provides the first systematic computational translation of foundational metacognitive theories, offering a principled vocabulary for understanding reasoning system failures and suggesting specific architectural interventions for future test-time reasoning designs. 

**Abstract (ZH)**: Test-time reasoning architectures such as those following the Generate-Verify paradigm -- where a model iteratively refines or verifies its own generated outputs -- prioritize generation and verification but exclude the monitoring processes that determine when and how reasoning should begin. This omission may contribute to the prefix dominance trap, in which models commit early to suboptimal reasoning paths and seldom recover, yielding roughly 20% accuracy loss. We address this architectural gap by formalizing Flavell's and Nelson and Narens' metacognitive theories into computational specifications, proposing the Monitor-Generate-Verify (MGV) framework. MGV extends the Generate-Verify paradigm by adding explicit monitoring that captures metacognitive experiences (from difficulty assessments to confidence judgements) before generation begins and refines future monitoring through verification feedback. 

---
# RxSafeBench: Identifying Medication Safety Issues of Large Language Models in Simulated Consultation 

**Title (ZH)**: RxSafeBench:识别大型语言模型在模拟咨询中用药安全问题 

**Authors**: Jiahao Zhao, Luxin Xu, Minghuan Tan, Lichao Zhang, Ahmadreza Argha, Hamid Alinejad-Rokny, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04328)  

**Abstract**: Numerous medical systems powered by Large Language Models (LLMs) have achieved remarkable progress in diverse healthcare tasks. However, research on their medication safety remains limited due to the lack of real world datasets, constrained by privacy and accessibility issues. Moreover, evaluation of LLMs in realistic clinical consultation settings, particularly regarding medication safety, is still underexplored. To address these gaps, we propose a framework that simulates and evaluates clinical consultations to systematically assess the medication safety capabilities of LLMs. Within this framework, we generate inquiry diagnosis dialogues with embedded medication risks and construct a dedicated medication safety database, RxRisk DB, containing 6,725 contraindications, 28,781 drug interactions, and 14,906 indication-drug pairs. A two-stage filtering strategy ensures clinical realism and professional quality, resulting in the benchmark RxSafeBench with 2,443 high-quality consultation scenarios. We evaluate leading open-source and proprietary LLMs using structured multiple choice questions that test their ability to recommend safe medications under simulated patient contexts. Results show that current LLMs struggle to integrate contraindication and interaction knowledge, especially when risks are implied rather than explicit. Our findings highlight key challenges in ensuring medication safety in LLM-based systems and provide insights into improving reliability through better prompting and task-specific tuning. RxSafeBench offers the first comprehensive benchmark for evaluating medication safety in LLMs, advancing safer and more trustworthy AI-driven clinical decision support. 

**Abstract (ZH)**: 由大型语言模型驱动的医疗系统在多样化的医疗保健任务中取得了显著进展。然而，由于缺乏实际-world数据集，受限于隐私和可访问性问题，对其药物安全性研究仍较为有限。此外，特别是在药物安全性方面的临床咨询-setting中评估大型语言模型仍然鲜有探索。为解决这些差距，我们提出了一种框架，用于模拟和评估临床咨询，以系统评估大型语言模型的药物安全性能力。在此框架下，我们生成了包含药物风险的问询诊断对话，并构建了一个专用的药物安全性数据库RxRisk DB，包含6,725个禁忌症、28,781个药物相互作用和14,906个适应症-药物配对。采用两阶段过滤策略确保临床真实性和专业质量，从而构建出基准RxSafeBench，其中包括2,443个高质量的咨询场景。我们使用结构化多项选择题来评估领先的开源和专有大型语言模型，在模拟的患者情境中测试它们推荐安全药物的能力。结果显示，当前大型语言模型在整合禁忌症和相互作用知识方面存在困难，尤其是在风险暗示而不是明确时。我们的研究结果突出了大型语言模型基于系统确保药物安全的关键挑战，并提供了通过更好的提示和任务特定调优提高可靠性的见解。RxSafeBench 提供了评估大型语言模型药物安全性的首个全面基准，推动了更安全和更值得信赖的基于人工智能的临床决策支持的发展。 

---
# AdversariaLLM: A Unified and Modular Toolbox for LLM Robustness Research 

**Title (ZH)**: AdversariaLLM：一个统一且模块化的大型语言模型鲁棒性研究工具箱 

**Authors**: Tim Beyer, Jonas Dornbusch, Jakob Steimle, Moritz Ladenburger, Leo Schwinn, Stephan Günnemann  

**Link**: [PDF](https://arxiv.org/pdf/2511.04316)  

**Abstract**: The rapid expansion of research on Large Language Model (LLM) safety and robustness has produced a fragmented and oftentimes buggy ecosystem of implementations, datasets, and evaluation methods. This fragmentation makes reproducibility and comparability across studies challenging, hindering meaningful progress. To address these issues, we introduce AdversariaLLM, a toolbox for conducting LLM jailbreak robustness research. Its design centers on reproducibility, correctness, and extensibility. The framework implements twelve adversarial attack algorithms, integrates seven benchmark datasets spanning harmfulness, over-refusal, and utility evaluation, and provides access to a wide range of open-weight LLMs via Hugging Face. The implementation includes advanced features for comparability and reproducibility such as compute-resource tracking, deterministic results, and distributional evaluation techniques. \name also integrates judging through the companion package JudgeZoo, which can also be used independently. Together, these components aim to establish a robust foundation for transparent, comparable, and reproducible research in LLM safety. 

**Abstract (ZH)**: 大型语言模型安全性与鲁棒性研究的快速扩展产生了碎片化且时常有错误的实现、数据集和评估方法生态系统。这种碎片化使得跨研究的再现性和可比性变得极具挑战性，阻碍了有意义的进步。为解决这些问题，我们引入了AdversariaLLM工具箱，用于开展大型语言模型监狱逃脱鲁棒性研究。该工具箱的设计旨在强调再现性、准确性与可扩展性。框架实现了一种包含十二种对抗攻击算法的机制，集成了涵盖危害性、过度拒绝和实用性评估的七种基准数据集，并通过Hugging Face提供了广泛的开源大型语言模型访问途径。该实施还包含用于提高再现性和可比性的高级功能，如计算资源跟踪、确定性结果和分布性评估技术。此外，还集成了通过配套包JudgeZoo进行评估的功能，该配套包也可以独立使用。这些组件旨在为透明、可比和可再现的大型语言模型安全性研究建立一个坚实的基础。 

---
# KGFR: A Foundation Retriever for Generalized Knowledge Graph Question Answering 

**Title (ZH)**: KGFR：通用知识图谱问答的基础检索器 

**Authors**: Yuanning Cui, Zequn Sun, Wei Hu, Zhangjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04093)  

**Abstract**: Large language models (LLMs) excel at reasoning but struggle with knowledge-intensive questions due to limited context and parametric knowledge. However, existing methods that rely on finetuned LLMs or GNN retrievers are limited by dataset-specific tuning and scalability on large or unseen graphs. We propose the LLM-KGFR collaborative framework, where an LLM works with a structured retriever, the Knowledge Graph Foundation Retriever (KGFR). KGFR encodes relations using LLM-generated descriptions and initializes entities based on their roles in the question, enabling zero-shot generalization to unseen KGs. To handle large graphs efficiently, it employs Asymmetric Progressive Propagation (APP)- a stepwise expansion that selectively limits high-degree nodes while retaining informative paths. Through node-, edge-, and path-level interfaces, the LLM iteratively requests candidate answers, supporting facts, and reasoning paths, forming a controllable reasoning loop. Experiments demonstrate that LLM-KGFR achieves strong performance while maintaining scalability and generalization, providing a practical solution for KG-augmented reasoning. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推理方面表现优异，但在处理知识密集型问题时因上下文和参数知识有限而受到影响。现有依赖调优LLMs或GNN检索器的方法受限于数据集特定的调优和在大规模或未见过的图上的可扩展性。我们提出了一种LLM-KGFR协作框架，其中LLM与结构化检索器——知识图谱基础检索器（KGFR）协同工作。KGFR使用LLM生成的描述编码关系，并基于实体在问题中的角色初始化实体，使LLM能够零样本迁移至未见过的知识图谱。为高效处理大规模图，KGFR采用了非对称渐进传播（APP）——一种逐步扩展方法，可选择性地限制高度节点同时保留信息路径。通过节点、边和路径级别的接口，LLM迭代地请求候选答案、支持事实和推理路径，形成可控的推理循环。实验表明，LLM-KGFR在保持可扩展性和迁移性的同时实现了强大的性能，提供了一种实用的知识图谱增强推理解决方案。 

---
# Agentmandering: A Game-Theoretic Framework for Fair Redistricting via Large Language Model Agents 

**Title (ZH)**: 基于大型语言模型代理的公平重划选区博弈理论框架 

**Authors**: Hao Li, Haotian Chen, Ruoyuan Gong, Juanjuan Wang, Hao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04076)  

**Abstract**: Redistricting plays a central role in shaping how votes are translated into political power. While existing computational methods primarily aim to generate large ensembles of legally valid districting plans, they often neglect the strategic dynamics involved in the selection process. This oversight creates opportunities for partisan actors to cherry-pick maps that, while technically compliant, are politically advantageous. Simply satisfying formal constraints does not ensure fairness when the selection process itself can be manipulated. We propose \textbf{Agentmandering}, a framework that reimagines redistricting as a turn-based negotiation between two agents representing opposing political interests. Drawing inspiration from game-theoretic ideas, particularly the \textit{Choose-and-Freeze} protocol, our method embeds strategic interaction into the redistricting process via large language model (LLM) agents. Agents alternate between selecting and freezing districts from a small set of candidate maps, gradually partitioning the state through constrained and interpretable choices. Evaluation on post-2020 U.S. Census data across all states shows that Agentmandering significantly reduces partisan bias and unfairness, while achieving 2 to 3 orders of magnitude lower variance than standard baselines. These results demonstrate both fairness and stability, especially in swing-state scenarios. Our code is available at this https URL. 

**Abstract (ZH)**: 选举区划在塑造选票转化为政治权力的过程中发挥着核心作用。虽然现有的计算方法主要旨在生成大量合法有效的区划方案，但它们往往忽视了选择过程中的战略动态。这一疏忽为党派行为体提供了机会，使其可以选择虽符合技术要求但更具政治优势的选区划分图。仅仅满足正式约束条件并不能确保公平，尤其是在选择过程本身可以被操控的情况下。我们提出了**Agentmandering**框架，将选举区划重新设想为两个代表对立政治利益的代理进行轮转谈判的过程。该方法受博弈理论思想的启发，特别是“选择并冻结”协议，通过大型语言模型（LLM）代理将战略互动嵌入到区划过程中。代理交替选择并冻结少量候选区划图，逐步通过受限且可解释的选择对州进行分区。在2020年人口普查后所有州的数据上的评估结果显示，Agentmandering显着减少了党派偏向和不公平现象，同时实现比标准基准低2到3个数量级的方差。这些结果表明了其公平性和稳定性，特别是在摇摆州的情境下。我们的代码可在以下网址获取：this https URL。 

---
# Interpreting Multi-Attribute Confounding through Numerical Attributes in Large Language Models 

**Title (ZH)**: 通过数值属性解释大型语言模型中的多属性混杂因素 

**Authors**: Hirohane Takagi, Gouki Minegishi, Shota Kizawa, Issey Sukeda, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2511.04053)  

**Abstract**: Although behavioral studies have documented numerical reasoning errors in large language models (LLMs), the underlying representational mechanisms remain unclear. We hypothesize that numerical attributes occupy shared latent subspaces and investigate two questions:(1) How do LLMs internally integrate multiple numerical attributes of a single entity? (2)How does irrelevant numerical context perturb these representations and their downstream outputs? To address these questions, we combine linear probing with partial correlation analysis and prompt-based vulnerability tests across models of varying sizes. Our results show that LLMs encode real-world numerical correlations but tend to systematically amplify them. Moreover, irrelevant context induces consistent shifts in magnitude representations, with downstream effects that vary by model size. These findings reveal a vulnerability in LLM decision-making and lay the groundwork for fairer, representation-aware control under multi-attribute entanglement. 

**Abstract (ZH)**: 尽管行为研究表明大型语言模型（LLMs）存在数值推理错误，但其背后的表征机制仍不清楚。我们假设数值属性占据共享的潜在子空间，并探讨了以下两个问题：(1) LLMs如何内部整合单个实体的多个数值属性？(2) 无关的数值上下文如何扰动这些表征及其下游输出？为了回答这些问题，我们结合使用线性探针和部分相关分析，并在不同规模的模型中进行基于提示的脆弱性测试。我们的结果显示，LLMs编码了现实世界的数值相关性，但往往会系统地放大它们。此外，无关的上下文会导致幅度表征的一致性变化，且下游影响因模型规模而异。这些发现揭示了LLMs决策中的脆弱性，并为进一步在多属性纠缠下实现更公平、更表征感知的控制奠定了基础。 

---
# ArchPilot: A Proxy-Guided Multi-Agent Approach for Machine Learning Engineering 

**Title (ZH)**: ArchPilot: 代理引导的多Agent机器学习工程方法 

**Authors**: Zhuowen Yuan, Tao Liu, Yang Yang, Yang Wang, Feng Qi, Kaushik Rangadurai, Bo Li, Shuang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.03985)  

**Abstract**: Recent LLM-based agents have demonstrated strong capabilities in automated ML engineering. However, they heavily rely on repeated full training runs to evaluate candidate solutions, resulting in significant computational overhead, limited scalability to large search spaces, and slow iteration cycles. To address these challenges, we introduce ArchPilot, a multi-agent system that integrates architecture generation, proxy-based evaluation, and adaptive search into a unified framework. ArchPilot consists of three specialized agents: an orchestration agent that coordinates the search process using a Monte Carlo Tree Search (MCTS)-inspired novel algorithm with a restart mechanism and manages memory of previous candidates; a generation agent that iteratively generates, improves, and debugs candidate architectures; and an evaluation agent that executes proxy training runs, generates and optimizes proxy functions, and aggregates the proxy scores into a fidelity-aware performance metric. This multi-agent collaboration allows ArchPilot to prioritize high-potential candidates with minimal reliance on expensive full training runs, facilitating efficient ML engineering under limited budgets. Experiments on MLE-Bench demonstrate that ArchPilot outperforms SOTA baselines such as AIDE and ML-Master, validating the effectiveness of our multi-agent system. 

**Abstract (ZH)**: 基于LLM的多代理系统ArchPilot在自动化ML工程中的应用与挑战 

---
# LLMs and Cultural Values: the Impact of Prompt Language and Explicit Cultural Framing 

**Title (ZH)**: LLMs和文化价值：提示语言和明确文化框架的影响 

**Authors**: Bram Bulté, Ayla Rigouts Terryn  

**Link**: [PDF](https://arxiv.org/pdf/2511.03980)  

**Abstract**: Large Language Models (LLMs) are rapidly being adopted by users across the globe, who interact with them in a diverse range of languages. At the same time, there are well-documented imbalances in the training data and optimisation objectives of this technology, raising doubts as to whether LLMs can represent the cultural diversity of their broad user base. In this study, we look at LLMs and cultural values and examine how prompt language and cultural framing influence model responses and their alignment with human values in different countries. We probe 10 LLMs with 63 items from the Hofstede Values Survey Module and World Values Survey, translated into 11 languages, and formulated as prompts with and without different explicit cultural perspectives. Our study confirms that both prompt language and cultural perspective produce variation in LLM outputs, but with an important caveat: While targeted prompting can, to a certain extent, steer LLM responses in the direction of the predominant values of the corresponding countries, it does not overcome the models' systematic bias toward the values associated with a restricted set of countries in our dataset: the Netherlands, Germany, the US, and Japan. All tested models, regardless of their origin, exhibit remarkably similar patterns: They produce fairly neutral responses on most topics, with selective progressive stances on issues such as social tolerance. Alignment with cultural values of human respondents is improved more with an explicit cultural perspective than with a targeted prompt language. Unexpectedly, combining both approaches is no more effective than cultural framing with an English prompt. These findings reveal that LLMs occupy an uncomfortable middle ground: They are responsive enough to changes in prompts to produce variation, but too firmly anchored to specific cultural defaults to adequately represent cultural diversity. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在全球范围内被用户迅速采用，用户们使用这些模型进行多种语言的交互。与此同时，这种技术的训练数据和优化目标存在已文档化的不平衡，引发了人们对其能否代表广泛用户群体的文化多元性的质疑。在本研究中，我们考察了大型语言模型与文化价值观之间的关系，并分析了提示语言和文化框架如何影响模型响应及其与不同国家的人类价值观的契合度。我们使用霍夫斯泰德价值调查模块和世界价值观调查中的63项调查项目，将其翻译成11种语言，并以带有和不带有不同明确文化视角的方式形成提示，对10个大型语言模型进行了探究。研究证实，提示语言和文化视角都会影响LLM输出，但存在一个重要 caveat：虽然定向提示可以在一定程度上引导LLM响应朝相应国家主流价值观的方向发展，但它并不能克服模型系统性的偏向于与数据集中特定国家相关的价值观的问题：荷兰、德国、美国和日本。无论模型的起源如何，所有测试过的模型表现出惊人相似的模式：它们在大多数话题上产生相对中立的响应，在社会宽容等方面持有选择性的进步立场。与明确的文化视角相比，定向提示语言在提高模型与人类受访者文化价值观的一致性方面效果较差。出乎意料的是，将这两种方法结合使用的效果并不比用英语提示的文化框架更好。这些发现揭示了大型语言模型处于一个令人不安的中间地带：它们对提示的变化足够敏感以产生变化，但又过于固守特定的文化默认值，无法充分代表文化多样性。 

---
# KnowThyself: An Agentic Assistant for LLM Interpretability 

**Title (ZH)**: 知行合一：一个自主代理助手用于LLM解释性分析 

**Authors**: Suraj Prasai, Mengnan Du, Ying Zhang, Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.03878)  

**Abstract**: We develop KnowThyself, an agentic assistant that advances large language model (LLM) interpretability. Existing tools provide useful insights but remain fragmented and code-intensive. KnowThyself consolidates these capabilities into a chat-based interface, where users can upload models, pose natural language questions, and obtain interactive visualizations with guided explanations. At its core, an orchestrator LLM first reformulates user queries, an agent router further directs them to specialized modules, and the outputs are finally contextualized into coherent explanations. This design lowers technical barriers and provides an extensible platform for LLM inspection. By embedding the whole process into a conversational workflow, KnowThyself offers a robust foundation for accessible LLM interpretability. 

**Abstract (ZH)**: 我们发展了KnowThyself，这是一种促进大规模语言模型解释性的自主助手。 

---
# To See or To Read: User Behavior Reasoning in Multimodal LLMs 

**Title (ZH)**: 是看还是读：多模态LLM中的用户行为推理 

**Authors**: Tianning Dong, Luyi Ma, Varun Vasudevan, Jason Cho, Sushant Kumar, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2511.03845)  

**Abstract**: Multimodal Large Language Models (MLLMs) are reshaping how modern agentic systems reason over sequential user-behavior data. However, whether textual or image representations of user behavior data are more effective for maximizing MLLM performance remains underexplored. We present \texttt{BehaviorLens}, a systematic benchmarking framework for assessing modality trade-offs in user-behavior reasoning across six MLLMs by representing transaction data as (1) a text paragraph, (2) a scatter plot, and (3) a flowchart. Using a real-world purchase-sequence dataset, we find that when data is represented as images, MLLMs next-purchase prediction accuracy is improved by 87.5% compared with an equivalent textual representation without any additional computational cost. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）正在重塑现代代理系统对序列用户行为数据进行推理的方式。然而，文本表示或图像表示的用户行为数据哪种更有效以最大化MLLM性能仍待探索。我们提出\texttt{BehaviorLens}，这是一种系统性的基准框架，通过将交易数据表示为（1）一段文本、（2）散点图和（3）流程图，在六种MLLM上评估模态权衡，在一个真实世界的购买序列数据集中，发现当数据以图像形式表示时，MLLM的下一次购买预测准确性提高了87.5%，而无需额外的计算成本。 

---
# How Different Tokenization Algorithms Impact LLMs and Transformer Models for Binary Code Analysis 

**Title (ZH)**: 不同分词算法对二进制代码分析的LLMs和Transformer模型的影响 

**Authors**: Ahmed Mostafa, Raisul Arefin Nahid, Samuel Mulder  

**Link**: [PDF](https://arxiv.org/pdf/2511.03825)  

**Abstract**: Tokenization is fundamental in assembly code analysis, impacting intrinsic characteristics like vocabulary size, semantic coverage, and extrinsic performance in downstream tasks. Despite its significance, tokenization in the context of assembly code remains an underexplored area. This study aims to address this gap by evaluating the intrinsic properties of Natural Language Processing (NLP) tokenization models and parameter choices, such as vocabulary size. We explore preprocessing customization options and pre-tokenization rules tailored to the unique characteristics of assembly code. Additionally, we assess their impact on downstream tasks like function signature prediction -- a critical problem in binary code analysis.
To this end, we conduct a thorough study on various tokenization models, systematically analyzing their efficiency in encoding assembly instructions and capturing semantic nuances. Through intrinsic evaluations, we compare tokenizers based on tokenization efficiency, vocabulary compression, and representational fidelity for assembly code. Using state-of-the-art pre-trained models such as the decoder-only Large Language Model (LLM) Llama 3.2, the encoder-only transformer BERT, and the encoder-decoder model BART, we evaluate the effectiveness of these tokenizers across multiple performance metrics. Preliminary findings indicate that tokenizer choice significantly influences downstream performance, with intrinsic metrics providing partial but incomplete predictability of extrinsic evaluation outcomes. These results reveal complex trade-offs between intrinsic tokenizer properties and their utility in practical assembly code tasks. Ultimately, this study provides valuable insights into optimizing tokenization models for low-level code analysis, contributing to the robustness and scalability of Natural Language Model (NLM)-based binary analysis workflows. 

**Abstract (ZH)**: 汇编代码分词是汇编代码分析的基础，影响词汇量、语义覆盖和下游任务的外在性能。尽管其重要性不言而喻，但在汇编代码上下文中的分词研究仍然不足。本研究旨在通过评估自然语言处理（NLP）分词模型和参数选择（如词汇量）的固有属性来填补这一空白。我们探索了针对汇编代码独特特性的预处理定制选项和预分词规则，并评估它们对函数签名预测等下游任务的影响——这是二进制代码分析中的一个关键问题。为此，我们对多种分词模型进行了全面研究，系统分析了它们在编码汇编指令和捕捉语义细微差别方面的效率。通过固有评估，我们比较了基于分词效率、词汇压缩和表示保真度的分词器。使用最新的预训练模型如仅解码器大语言模型Llama 3.2、仅编码器变换器BERT以及编码器-解码器模型BART，我们在多个性能指标上评估了这些分词器的有效性。初步结果显示，分词器的选择极大地影响了下游性能，固有指标部分但不完全预测了外在评估结果。这些结果揭示了固有分词器属性与其在实际汇编代码任务中的效用之间的复杂权衡。最终，本研究为优化基于自然语言模型（NLM）的低级代码分析提供了宝贵见解，有助于提高二进制分析工作流的稳健性和可扩展性。 

---
# Scaling Agent Learning via Experience Synthesis 

**Title (ZH)**: 通过经验合成扩展代理学习 

**Authors**: Zhaorun Chen, Zhuokai Zhao, Kai Zhang, Bo Liu, Qi Qi, Yifan Wu, Tarun Kalluri, Sara Cao, Yuanhao Xiong, Haibo Tong, Huaxiu Yao, Hengduo Li, Jiacheng Zhu, Xian Li, Dawn Song, Bo Li, Jason Weston, Dat Huynh  

**Link**: [PDF](https://arxiv.org/pdf/2511.03773)  

**Abstract**: While reinforcement learning (RL) can empower large language model (LLM) agents by enabling self-improvement through interaction, its practical adoption remains challenging due to costly rollouts, limited task diversity, unreliable reward signals, and infrastructure complexity, all of which obstruct the collection of scalable experience data. To address these challenges, we introduce DreamGym, the first unified framework designed to synthesize diverse experiences with scalability in mind to enable effective online RL training for autonomous agents. Rather than relying on expensive real-environment rollouts, DreamGym distills environment dynamics into a reasoning-based experience model that derives consistent state transitions and feedback signals through step-by-step reasoning, enabling scalable agent rollout collection for RL. To improve the stability and quality of transitions, DreamGym leverages an experience replay buffer initialized with offline real-world data and continuously enriched with fresh interactions to actively support agent training. To improve knowledge acquisition, DreamGym adaptively generates new tasks that challenge the current agent policy, enabling more effective online curriculum learning. Experiments across diverse environments and agent backbones demonstrate that DreamGym substantially improves RL training, both in fully synthetic settings and in sim-to-real transfer scenarios. On non-RL-ready tasks like WebArena, DreamGym outperforms all baselines by over 30%. And in RL-ready but costly settings, it matches GRPO and PPO performance using only synthetic interactions. When transferring a policy trained purely on synthetic experiences to real-environment RL, DreamGym yields significant additional performance gains while requiring far fewer real-world interactions, providing a scalable warm-start strategy for general-purpose RL. 

**Abstract (ZH)**: 面向自主代理的有效在线强化学习训练的统一可扩展框架：DreamGym 

---
# LLM-as-a-Judge: Toward World Models for Slate Recommendation Systems 

**Title (ZH)**: LLM作为法官：面向 Slate 推荐系统的世界模型研究 

**Authors**: Baptiste Bonin, Maxime Heuillet, Audrey Durand  

**Link**: [PDF](https://arxiv.org/pdf/2511.04541)  

**Abstract**: Modeling user preferences across domains remains a key challenge in slate recommendation (i.e. recommending an ordered sequence of items) research. We investigate how Large Language Models (LLM) can effectively act as world models of user preferences through pairwise reasoning over slates. We conduct an empirical study involving several LLMs on three tasks spanning different datasets. Our results reveal relationships between task performance and properties of the preference function captured by LLMs, hinting towards areas for improvement and highlighting the potential of LLMs as world models in recommender systems. 

**Abstract (ZH)**: 跨域用户偏好建模仍是排列表现推荐（即推荐有序项目列表）研究中的一个关键挑战。我们探讨了大型语言模型（LLM）如何通过两两比较排列表现有效地充当用户偏好的世界模型。我们在三个涉及不同数据集的任务中对几种LLM进行了实证研究。我们的结果揭示了任务性能与LLM捕获的偏好函数属性之间的关系，这暗示了改进的方向，并突显了LLM作为推荐系统中世界模型的潜力。 

---
# Are language models aware of the road not taken? Token-level uncertainty and hidden state dynamics 

**Title (ZH)**: 语言模型意识到未选择的路吗？ token级不确定性与隐藏状态动态 

**Authors**: Amir Zur, Atticus Geiger, Ekdeep Singh Lubana, Eric Bigelow  

**Link**: [PDF](https://arxiv.org/pdf/2511.04527)  

**Abstract**: When a language model generates text, the selection of individual tokens might lead it down very different reasoning paths, making uncertainty difficult to quantify. In this work, we consider whether reasoning language models represent the alternate paths that they could take during generation. To test this hypothesis, we use hidden activations to control and predict a language model's uncertainty during chain-of-thought reasoning. In our experiments, we find a clear correlation between how uncertain a model is at different tokens, and how easily the model can be steered by controlling its activations. This suggests that activation interventions are most effective when there are alternate paths available to the model -- in other words, when it has not yet committed to a particular final answer. We also find that hidden activations can predict a model's future outcome distribution, demonstrating that models implicitly represent the space of possible paths. 

**Abstract (ZH)**: 语言模型在生成文本时，单个词的选择可能会引导它走向非常不同的推理路径，使得不确定性难以量化。在这项工作中，我们考虑推理语言模型是否在生成过程中表示它能够采取的替代路径。为了检验这一假设，我们使用隐藏激活来控制和预测语言模型在链式推理过程中所表现出的不确定性。在我们的实验中，我们发现模型在不同词上的不确定性程度与可以通过控制其激活来引导模型的难易程度之间存在明显的相关性。这表明在模型尚未确定具体最终答案的情况下，激活干预措施是最有效的——换句话说，即当模型还没有确定某一特定最终答案时。我们还发现隐藏激活可以预测模型未来结果的分布，这表明模型隐含地表示了可能路径的空间。 

---
# RAGalyst: Automated Human-Aligned Agentic Evaluation for Domain-Specific RAG 

**Title (ZH)**: RAGalyst: 自动化领域特定检索增强代理的人工对齐评估 

**Authors**: Joshua Gao, Quoc Huy Pham, Subin Varghese, Silwal Saurav, Vedhus Hoskere  

**Link**: [PDF](https://arxiv.org/pdf/2511.04502)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a critical technique for grounding Large Language Models (LLMs) in factual evidence, yet evaluating RAG systems in specialized, safety-critical domains remains a significant challenge. Existing evaluation frameworks often rely on heuristic-based metrics that fail to capture domain-specific nuances and other works utilize LLM-as-a-Judge approaches that lack validated alignment with human judgment. This paper introduces RAGalyst, an automated, human-aligned agentic framework designed for the rigorous evaluation of domain-specific RAG systems. RAGalyst features an agentic pipeline that generates high-quality, synthetic question-answering (QA) datasets from source documents, incorporating an agentic filtering step to ensure data fidelity. The framework refines two key LLM-as-a-Judge metrics-Answer Correctness and Answerability-using prompt optimization to achieve a strong correlation with human annotations. Applying this framework to evaluate various RAG components across three distinct domains (military operations, cybersecurity, and bridge engineering), we find that performance is highly context-dependent. No single embedding model, LLM, or hyperparameter configuration proves universally optimal. Additionally, we provide an analysis on the most common low Answer Correctness reasons in RAG. These findings highlight the necessity of a systematic evaluation framework like RAGalyst, which empowers practitioners to uncover domain-specific trade-offs and make informed design choices for building reliable and effective RAG systems. RAGalyst is available on our Github. 

**Abstract (ZH)**: 基于检索的生成增强（RAG）是将大型语言模型（LLMs）与事实证据对接的关键技术，但在专业性和安全性要求高的领域评估RAG系统仍是一项重大挑战。现有的评估框架通常依赖于基于启发式的指标，无法捕捉特定领域的细微差异，而其他工作则采用大型语言模型作为裁判的方法，缺乏与人类判断的有效对齐。本文介绍了RAGalyst，这是一种自动化的人类对齐代理框架，旨在严格评估特定领域内的RAG系统。RAGalyst具有一代理管道，能够从源文档中生成高质量的合成问答（QA）数据集，并包含一个代理过滤步骤以确保数据的真实性和准确性。该框架通过提示优化方法，优化了两个关键的基于大型语言模型作为裁判的指标——答案正确性和可回答性，实现了与人类注释的强相关性。我们将此框架应用于军事行动、网络安全和桥梁工程三个不同领域的各种RAG组件进行评估，发现性能高度依赖于具体情境。没有单一的嵌入模型、大型语言模型或超参数配置能够普遍适用。此外，我们还分析了RAG中低答案正确性的常见原因。这些发现突显了RAGalyst这样系统性评估框架的重要性，它能帮助实践者揭示特定领域的权衡，从而做出可靠和有效的RAG系统的设计选择。RAGalyst可在我们的Github上获得。 

---
# Decoding Emergent Big Five Traits in Large Language Models: Temperature-Dependent Expression and Architectural Clustering 

**Title (ZH)**: 大型语言模型中涌现的五大特质解码：温度依赖表达与架构聚类 

**Authors**: Christos-Nikolaos Zacharopoulos, Revekka Kyriakoglou  

**Link**: [PDF](https://arxiv.org/pdf/2511.04499)  

**Abstract**: As Large Language Models (LLMs) become integral to human-centered applications, understanding their personality-like behaviors is increasingly important for responsible development and deployment. This paper systematically evaluates six LLMs, applying the Big Five Inventory-2 (BFI-2) framework, to assess trait expressions under varying sampling temperatures. We find significant differences across four of the five personality dimensions, with Neuroticism and Extraversion susceptible to temperature adjustments. Further, hierarchical clustering reveals distinct model clusters, suggesting that architectural features may predispose certain models toward stable trait profiles. Taken together, these results offer new insights into the emergence of personality-like patterns in LLMs and provide a new perspective on model tuning, selection, and the ethical governance of AI systems. We share the data and code for this analysis here: this https URL 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在以人为中心的应用中变得越来越重要，理解其类似人格的行为对于负责任的发展和部署越来越重要。本文系统地评估了六种LLM，应用Big Five Inventory-2（BFI-2）框架，评估在不同采样温度下的特质表达。我们发现五个个性维度中有四个存在显著差异，神经质和外向性对温度调整敏感。进一步的层次聚类揭示了不同的模型簇，表明架构特征可能使某些模型倾向于稳定的特质模式。综上所述，这些结果为LLM中类似人格模式的出现提供了新的见解，并为模型调优、选择和AI系统的伦理治理提供了新的视角。我们在这里分享该分析的数据和代码：this https URL。 

---
# OUNLP at TSAR 2025 Shared Task: Multi-Round Text Simplifier via Code Generation 

**Title (ZH)**: OUNLP在TSAR 2025共享任务中的多轮文本简化器：通过代码生成 

**Authors**: Cuong Huynh, Jie Cao  

**Link**: [PDF](https://arxiv.org/pdf/2511.04495)  

**Abstract**: This paper describes the OUNLP system submitted to the TSAR-2025 Shared Task (Alva-Manchego et al., 2025), designed for readability-controlled text simplification using LLM-prompting-based generation. Based on the analysis of prompt-based text simplification methods, we discovered an interesting finding that text simplification performance is highly related to the gap between the source CEFR (Arase et al., 2022) level and the target CEFR level. Inspired by this finding, we propose two multi-round simplification methods and generate them via GPT-4o: rule-based simplification (MRS-Rule) and jointly rule-based LLM simplification (MRS-Joint). Our submitted systems ranked 7 out of 20 teams. Later improvements with MRS-Joint show that taking the LLM simplified candidates as the starting point could further boost the multi-round simplification performance. 

**Abstract (ZH)**: 本论文描述了提交给TSAR-2025共享任务的OUNLP系统（Alva-Manchego等，2025），该系统基于LLM提示生成进行可读性控制的文本简化。通过对基于提示的文本简化方法的分析，我们发现文本简化性能与源CEFR（Arase等，2022）级别和目标CEFR级别之间的差距密切相关。受这一发现的启发，我们提出了两种多轮简化方法并通过GPT-4o生成：基于规则的简化（MRS-Rule）和联合基于规则的LLM简化（MRS-Joint）。我们的提交系统在20支队伍中排名第7。后续使用MRS-Joint的改进表明，将LLM简化候选作为起点可以进一步提升多轮简化性能。 

---
# RUST-BENCH: Benchmarking LLM Reasoning on Unstructured Text within Structured Tables 

**Title (ZH)**: RUST-BENCH：在结构化表格中对LLM进行非结构化文本推理基准测试 

**Authors**: Nikhil Abhyankar, Purvi Chaurasia, Sanchit Kabra, Ananya Srivastava, Vivek Gupta, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2511.04491)  

**Abstract**: Existing tabular reasoning benchmarks mostly test models on small, uniform tables, underrepresenting the complexity of real-world data and giving an incomplete view of Large Language Models' (LLMs) reasoning abilities. Real tables are long, heterogeneous, and domain-specific, mixing structured fields with free text and requiring multi-hop reasoning across thousands of tokens. To address this gap, we introduce RUST-BENCH, a benchmark of 7966 questions from 2031 real-world tables spanning two domains: i) RB-Science (NSF grant records) and ii) RB-Sports (NBA statistics). Unlike prior work, RUST-BENCH evaluates LLMs jointly across scale, heterogeneity, domain specificity, and reasoning complexity. Experiments with open-source and proprietary models show that LLMs struggle with heterogeneous schemas and complex multi-hop inference, revealing persistent weaknesses in current architectures and prompting strategies. RUST-BENCH establishes a challenging new testbed for advancing tabular reasoning research. 

**Abstract (ZH)**: 现存的表格推理基准主要在小规模、统一的表格上测试模型，未能充分反映真实世界数据的复杂性，也未全面展现大型语言模型（LLMs）的推理能力。真实表格通常较长、异构且专门针对特定领域，混合了结构化字段和自由文本，并要求在数千个令牌中进行多跳推理。为了填补这一差距，我们引入了RUST-BENCH，这是一个包含来自2031个实际表格的7966个问题的基准，覆盖两个领域：i) RB-Science（NSF资助记录）和ii) RB-Sports（NBA统计数据）。与先前工作不同，RUST-BENCH联合评估了LLMs在规模、异构性、领域特异性和推理复杂性方面的表现。开源和专有模型的实验表明，LLMs在异构模式和复杂多跳推断方面存在困难，揭示了当前架构的持续弱点，并提出了改进策略。RUST-BENCH为推进表格推理研究设立了新的挑战性测试平台。 

---
# Generate, Evaluate, Iterate: Synthetic Data for Human-in-the-Loop Refinement of LLM Judges 

**Title (ZH)**: 生成、评估、迭代：合成数据在LLM法官-human在环路 refinement 中的应用 

**Authors**: Hyo Jin Do, Zahra Ashktorab, Jasmina Gajcin, Erik Miehling, Martín Santillán Cooper, Qian Pan, Elizabeth M. Daly, Werner Geyer  

**Link**: [PDF](https://arxiv.org/pdf/2511.04478)  

**Abstract**: The LLM-as-a-judge paradigm enables flexible, user-defined evaluation, but its effectiveness is often limited by the scarcity of diverse, representative data for refining criteria. We present a tool that integrates synthetic data generation into the LLM-as-a-judge workflow, empowering users to create tailored and challenging test cases with configurable domains, personas, lengths, and desired outcomes, including borderline cases. The tool also supports AI-assisted inline editing of existing test cases. To enhance transparency and interpretability, it reveals the prompts and explanations behind each generation. In a user study (N=24), 83% of participants preferred the tool over manually creating or selecting test cases, as it allowed them to rapidly generate diverse synthetic data without additional workload. The generated synthetic data proved as effective as hand-crafted data for both refining evaluation criteria and aligning with human preferences. These findings highlight synthetic data as a promising alternative, particularly in contexts where efficiency and scalability are critical. 

**Abstract (ZH)**: LLM作为法官的范式 enables灵活的用户定义评估，但其效果往往受限于缺乏多样性和代表性的数据以精炼评估标准。我们介绍了一个将合成数据生成集成到LLM作为法官工作流中的工具，使用户能够创建定制化的具有挑战性的测试案例，包括可配置的主题、人物、长度和预期结果，甚至包括边缘案例。该工具还支持对现有测试案例的AI辅助即时编辑。为了增强透明度和可解释性，它会揭示每个生成过程背后的提示和解释。在一项用户研究（N=24）中，83%的参与者认为该工具优于手动创建或选择测试案例，因为它允许他们快速生成多样化的合成数据而无需额外的工作量。生成的合成数据在精炼评估标准和与人类偏好对齐方面与人工制作的数据同样有效。这些发现突出了合成数据作为一种有前途的替代方案，特别是在效率和可扩展性至关重要的环境中。 

---
# Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs 

**Title (ZH)**: 基于真实子图以提高知识图增强的LLM训练与评估效果 

**Authors**: Alberto Cattaneo, Carlo Luschi, Daniel Justus  

**Link**: [PDF](https://arxiv.org/pdf/2511.04473)  

**Abstract**: Retrieval of information from graph-structured knowledge bases represents a promising direction for improving the factuality of LLMs. While various solutions have been proposed, a comparison of methods is difficult due to the lack of challenging QA datasets with ground-truth targets for graph retrieval. We present SynthKGQA, a framework for generating high-quality synthetic Knowledge Graph Question Answering datasets from any Knowledge Graph, providing the full set of ground-truth facts in the KG to reason over each question. We show how, in addition to enabling more informative benchmarking of KG retrievers, the data produced with SynthKGQA also allows us to train better models. We apply SynthKGQA to Wikidata to generate GTSQA, a new dataset designed to test zero-shot generalization abilities of KG retrievers with respect to unseen graph structures and relation types, and benchmark popular solutions for KG-augmented LLMs on it. 

**Abstract (ZH)**: 从图结构知识库中检索信息代表了提高LLMs事实准确性的有前途的方向。尽管提出了各种解决方案，但由于缺乏用于图检索的具有ground-truth目标的挑战性QA数据集，方法间的比较存在困难。我们提出了SynthKGQA，一种从任何知识图谱生成高质量合成知识图谱问答数据集的框架，提供了KG中所有ground-truth事实以用于每个问题的推理。我们展示了SynthKGQA不仅能够增强KG检索器的基准测试，生成的数据还使我们能够训练更好的模型。我们将SynthKGQA应用于Wikidata以生成GTSQA，这是一个新的数据集，旨在测试KG检索器在面对未见图结构和关系类型方面的零样本泛化能力，并在其中对流行的KG增强LLM解决方案进行了基准测试。 

---
# Speed at the Cost of Quality? The Impact of LLM Agent Assistance on Software Development 

**Title (ZH)**: 牺牲质量的速度？LLM代理协助对软件开发的影响 

**Authors**: Hao He, Courtney Miller, Shyam Agarwal, Christian Kästner, Bogdan Vasilescu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04427)  

**Abstract**: Large language models (LLMs) have demonstrated the promise to revolutionize the field of software engineering. Among other things, LLM agents are rapidly gaining momentum in their application to software development, with practitioners claiming a multifold productivity increase after adoption. Yet, empirical evidence is lacking around these claims. In this paper, we estimate the causal effect of adopting a widely popular LLM agent assistant, namely Cursor, on development velocity and software quality. The estimation is enabled by a state-of-the-art difference-in-differences design comparing Cursor-adopting GitHub projects with a matched control group of similar GitHub projects that do not use Cursor. We find that the adoption of Cursor leads to a significant, large, but transient increase in project-level development velocity, along with a significant and persistent increase in static analysis warnings and code complexity. Further panel generalized method of moments estimation reveals that the increase in static analysis warnings and code complexity acts as a major factor causing long-term velocity slowdown. Our study carries implications for software engineering practitioners, LLM agent assistant designers, and researchers. 

**Abstract (ZH)**: 大型语言模型（LLMs）在软件工程领域展现出革命性的潜力。其中，LLM代理在软件开发中的应用正迅速增长，使用者声称采用后生产率显著提升。然而，这些主张缺乏实证支持。在本文中，我们通过将广泛使用的LLM代理助手Cursor引入的GitHub项目与未使用Cursor的匹配对照组进行先进差异-in-差异设计，估算了采用Cursor对开发速度和软件质量的影响。进一步的面板广义方法矩估计表明，静态分析警告增加和代码复杂性上升是导致长期速度下降的主要因素。本研究对软件工程从业者、LLM代理助手设计师和研究人员具有重要意义。 

---
# Trustworthy LLM-Mediated Communication: Evaluating Information Fidelity in LLM as a Communicator (LAAC) Framework in Multiple Application Domains 

**Title (ZH)**: 可信的大语言模型介导通信：在多种应用领域评估大语言模型作为通信者（LAAC框架）的信息准确性 

**Authors**: Mohammed Musthafa Rafi, Adarsh Krishnamurthy, Aditya Balu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04184)  

**Abstract**: The proliferation of AI-generated content has created an absurd communication theater where senders use LLMs to inflate simple ideas into verbose content, recipients use LLMs to compress them back into summaries, and as a consequence neither party engage with authentic content. LAAC (LLM as a Communicator) proposes a paradigm shift - positioning LLMs as intelligent communication intermediaries that capture the sender's intent through structured dialogue and facilitate genuine knowledge exchange with recipients. Rather than perpetuating cycles of AI-generated inflation and compression, LAAC enables authentic communication across diverse contexts including academic papers, proposals, professional emails, and cross-platform content generation. However, deploying LLMs as trusted communication intermediaries raises critical questions about information fidelity, consistency, and reliability. This position paper systematically evaluates the trustworthiness requirements for LAAC's deployment across multiple communication domains. We investigate three fundamental dimensions: (1) Information Capture Fidelity - accuracy of intent extraction during sender interviews across different communication types, (2) Reproducibility - consistency of structured knowledge across multiple interaction instances, and (3) Query Response Integrity - reliability of recipient-facing responses without hallucination, source conflation, or fabrication. Through controlled experiments spanning multiple LAAC use cases, we assess these trust dimensions using LAAC's multi-agent architecture. Preliminary findings reveal measurable trust gaps that must be addressed before LAAC can be reliably deployed in high-stakes communication scenarios. 

**Abstract (ZH)**: AI生成内容泛滥创建了一个荒诞的通信剧场，在此剧场中发送方使用大语言模型放大简单想法为冗长内容，接收方使用大语言模型将其压缩回摘要，结果双方都不与真实内容互动。LAAC（大语言模型作为通信中介）提出了一种范式转变——将大语言模型定位为通过结构化对话捕捉发送方意图，并促进与接收方的真实知识交流的智能通信中介。LAAC旨在打破AI生成内容的夸大与压缩循环，而是在包括学术论文、提案、专业电子邮件以及跨平台内容生成等多种场景下促进真实的沟通。然而，将大语言模型部署为可信赖的通信中介引发了关于信息 fidelity、一致性和可靠性的关键问题。本文系统地评估了LAAC在多个通信领域部署的可信赖性要求。我们调查了三个基本维度：（1）信息捕捉精度——不同类型通信中发送方访谈期间意图提取的准确性，（2）可再现性——多次交互实例中结构化知识的一致性，以及（3）查询响应完整性——面对接收方的响应无幻觉、无信息混淆和无篡改的可靠性。通过涉及多个LAAC应用场景的受控实验，我们使用LAAC的多代理架构评估这些信任维度。初步发现表明，在LAAC在高风险沟通场景中可靠部署之前必须解决可信赖性差距。 

---
# Explaining Software Vulnerabilities with Large Language Models 

**Title (ZH)**: 用大型语言模型解释软件漏洞 

**Authors**: Oshando Johnson, Alexandra Fomina, Ranjith Krishnamurthy, Vaibhav Chaudhari, Rohith Kumar Shanmuganathan, Eric Bodden  

**Link**: [PDF](https://arxiv.org/pdf/2511.04179)  

**Abstract**: The prevalence of security vulnerabilities has prompted companies to adopt static application security testing (SAST) tools for vulnerability detection. Nevertheless, these tools frequently exhibit usability limitations, as their generic warning messages do not sufficiently communicate important information to developers, resulting in misunderstandings or oversight of critical findings. In light of recent developments in Large Language Models (LLMs) and their text generation capabilities, our work investigates a hybrid approach that uses LLMs to tackle the SAST explainability challenges. In this paper, we present SAFE, an Integrated Development Environment (IDE) plugin that leverages GPT-4o to explain the causes, impacts, and mitigation strategies of vulnerabilities detected by SAST tools. Our expert user study findings indicate that the explanations generated by SAFE can significantly assist beginner to intermediate developers in understanding and addressing security vulnerabilities, thereby improving the overall usability of SAST tools. 

**Abstract (ZH)**: 基于大型语言模型的静态应用安全测试解释性增强：SAFE插件的研究 

---
# Are We Aligned? A Preliminary Investigation of the Alignment of Responsible AI Values between LLMs and Human Judgment 

**Title (ZH)**: 我们一致吗？关于LLM与人类判断的责任AI价值观初步调查 

**Authors**: Asma Yamani, Malak Baslyman, Moataz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2511.04157)  

**Abstract**: Large Language Models (LLMs) are increasingly employed in software engineering tasks such as requirements elicitation, design, and evaluation, raising critical questions regarding their alignment with human judgments on responsible AI values. This study investigates how closely LLMs' value preferences align with those of two human groups: a US-representative sample and AI practitioners. We evaluate 23 LLMs across four tasks: (T1) selecting key responsible AI values, (T2) rating their importance in specific contexts, (T3) resolving trade-offs between competing values, and (T4) prioritizing software requirements that embody those values. The results show that LLMs generally align more closely with AI practitioners than with the US-representative sample, emphasizing fairness, privacy, transparency, safety, and accountability. However, inconsistencies appear between the values that LLMs claim to uphold (Tasks 1-3) and the way they prioritize requirements (Task 4), revealing gaps in faithfulness between stated and applied behavior. These findings highlight the practical risk of relying on LLMs in requirements engineering without human oversight and motivate the need for systematic approaches to benchmark, interpret, and monitor value alignment in AI-assisted software development. 

**Abstract (ZH)**: 大型语言模型（LLMs）在软件工程任务中的应用引发了对其与人类关于负责AI价值观判断一致性的重要质疑。本研究调查了LLMs的价值偏好与两类人类群体的一致性：美国代表性样本和AI从业者。我们评估了23个LLM在四项任务中的表现：（T1）选择关键的负责AI价值观，（T2）在特定情境中对它们的重要性进行评级，（T3）解决竞争价值观之间的权衡，以及（T4）按照这些价值观优先排序软件需求。研究结果表明，LLMs更倾向于与AI从业者而非美国代表性样本一致，强调公平、隐私、透明度、安全性和问责制。然而，LLMs声称遵循的价值观（任务1-3）与其在优先排序需求时的方式之间存在不一致，揭示了声明和实际行为之间的一致性差距。这些发现突显了在缺乏人类监督的情况下依赖LLMs进行需求工程的实践风险，并促使人们对AI辅助软件开发中的价值一致性进行系统基准测试、解释和监控。 

---
# BAPPA: Benchmarking Agents, Plans, and Pipelines for Automated Text-to-SQL Generation 

**Title (ZH)**: BAPPA: 代理、计划和管道基准测试，用于自动化文本到SQL生成 

**Authors**: Fahim Ahmed, Md Mubtasim Ahasan, Jahir Sadik Monon, Muntasir Wahed, M Ashraful Amin, A K M Mahbubur Rahman, Amin Ahsan Ali  

**Link**: [PDF](https://arxiv.org/pdf/2511.04153)  

**Abstract**: Text-to-SQL systems provide a natural language interface that can enable even laymen to access information stored in databases. However, existing Large Language Models (LLM) struggle with SQL generation from natural instructions due to large schema sizes and complex reasoning. Prior work often focuses on complex, somewhat impractical pipelines using flagship models, while smaller, efficient models remain overlooked. In this work, we explore three multi-agent LLM pipelines, with systematic performance benchmarking across a range of small to large open-source models: (1) Multi-agent discussion pipeline, where agents iteratively critique and refine SQL queries, and a judge synthesizes the final answer; (2) Planner-Coder pipeline, where a thinking model planner generates stepwise SQL generation plans and a coder synthesizes queries; and (3) Coder-Aggregator pipeline, where multiple coders independently generate SQL queries, and a reasoning agent selects the best query. Experiments on the Bird-Bench Mini-Dev set reveal that Multi-Agent discussion can improve small model performance, with up to 10.6% increase in Execution Accuracy for Qwen2.5-7b-Instruct seen after three rounds of discussion. Among the pipelines, the LLM Reasoner-Coder pipeline yields the best results, with DeepSeek-R1-32B and QwQ-32B planners boosting Gemma 3 27B IT accuracy from 52.4% to the highest score of 56.4%. Codes are available at this https URL. 

**Abstract (ZH)**: 文本到SQL系统提供了一种自然语言界面，即使是非专业人士也能访问存储在数据库中的信息。然而，现有的大型语言模型（LLM）在生成SQL代码方面由于模式规模庞大和复杂的逻辑推理能力有限。前期工作往往侧重于使用旗舰模型的复杂且不太实际的工作流程，而较小且高效的模型却常被忽视。在此工作中，我们探索了三种多代理LLM流水线，并在从开源小模型到大模型的范围内进行了系统的性能基准测试：（1）多代理讨论流水线，其中代理迭代地评审和改进SQL查询，评判者综合最终答案；（2）规划者-编码流水线，其中思考模型规划者生成逐步的SQL生成方案，编码者综合查询；（3）编码者-聚合流水线，其中多个编码者独立生成SQL查询，推理代理选择最优查询。在Bird-Bench Mini-Dev数据集上的实验表明，多代理讨论可以提升小型模型的性能，在三次讨论后Qwen2.5-7b-Instruct的执行准确率提高了10.6%。在各种流水线中，LLM推理者-编码者流水线表现最优，DeepSeek-R1-32B和QwQ-32B规划者的提升使得Gemma 3 27B IT准确率从52.4%提升到最高分56.4%。代码可在以下链接获取。 

---
# Advancing Equitable AI: Evaluating Cultural Expressiveness in LLMs for Latin American Contexts 

**Title (ZH)**: 推进公平的AI：评估LLMs在拉丁美洲背景下的文化表现性 

**Authors**: Brigitte A. Mora-Reyes, Jennifer A. Drewyor, Abel A. Reyes-Angulo  

**Link**: [PDF](https://arxiv.org/pdf/2511.04090)  

**Abstract**: Artificial intelligence (AI) systems often reflect biases from economically advanced regions, marginalizing contexts in economically developing regions like Latin America due to imbalanced datasets. This paper examines AI representations of diverse Latin American contexts, revealing disparities between data from economically advanced and developing regions. We highlight how the dominance of English over Spanish, Portuguese, and indigenous languages such as Quechua and Nahuatl perpetuates biases, framing Latin American perspectives through a Western lens. To address this, we introduce a culturally aware dataset rooted in Latin American history and socio-political contexts, challenging Eurocentric models. We evaluate six language models on questions testing cultural context awareness, using a novel Cultural Expressiveness metric, statistical tests, and linguistic analyses. Our findings show that some models better capture Latin American perspectives, while others exhibit significant sentiment misalignment (p < 0.001). Fine-tuning Mistral-7B with our dataset improves its cultural expressiveness by 42.9%, advancing equitable AI development. We advocate for equitable AI by prioritizing datasets that reflect Latin American history, indigenous knowledge, and diverse languages, while emphasizing community-centered approaches to amplify marginalized voices. 

**Abstract (ZH)**: 人工智能系统往往反映了经济发达地区的偏见， marginalizing 经济欠发达地区如拉丁美洲的情况，由于数据集不平衡。本文考察了人工智能对多元拉丁美洲情境的表征，揭示了经济发达与欠发达地区之间存在的数据差距。我们强调了英语在拉丁美洲各语言（如西班牙语、葡萄牙语和Quechua、Nahuatl等土著语言）之上的主导地位如何延续偏见，通过西方视角框架化拉丁美洲的观点。为解决这一问题，我们引入了一个根植于拉丁美洲历史和社会政治背景的文化意识数据集，挑战了以欧洲为中心的模型。我们使用一种新的文化表现度量、统计测试和语义分析评估了六种语言模型在文化背景意识方面的表现。我们的研究发现，有些模型更好地捕捉了拉丁美洲的观点，而另一些则表现出显著的情感不匹配（p < 0.001）。通过使用我们的数据集对Mistral-7B进行细调，其文化表现度提高了42.9%，促进了公正的人工智能发展。我们倡导公正的人工智能，强调优先使用反映拉丁美洲历史、土著知识和多种语言的数据集，并强调社区中心的方法以放大边缘化声音。 

---
# Abductive Inference in Retrieval-Augmented Language Models: Generating and Validating Missing Premises 

**Title (ZH)**: 检索增强语言模型中的 abduction 推理：生成和验证缺失的前提 

**Authors**: Shiyin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.04020)  

**Abstract**: Large Language Models (LLMs) enhanced with retrieval -- commonly referred to as Retrieval-Augmented Generation (RAG) -- have demonstrated strong performance in knowledge-intensive tasks. However, RAG pipelines often fail when retrieved evidence is incomplete, leaving gaps in the reasoning process. In such cases, \emph{abductive inference} -- the process of generating plausible missing premises to explain observations -- offers a principled approach to bridge these gaps. In this paper, we propose a framework that integrates abductive inference into retrieval-augmented LLMs. Our method detects insufficient evidence, generates candidate missing premises, and validates them through consistency and plausibility checks. Experimental results on abductive reasoning and multi-hop QA benchmarks show that our approach improves both answer accuracy and reasoning faithfulness. This work highlights abductive inference as a promising direction for enhancing the robustness and explainability of RAG systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过检索增强——通常称为检索增强生成（RAG）——在知识密集型任务中表现出强大的性能。然而，当检索到的证据不完整时，RAG 流程往往失效，导致推理过程中的空白。在这种情况下，溯因推理——生成合理的缺失前提以解释观察结果的过程——提供了一种原则性的方法来填补这些空白。本文提出了一种框架，将溯因推理整合到检索增强的大语言模型中。该方法检测证据不足，生成候选的缺失前提，并通过一致性与合理性检查验证它们。在溯因推理和多跳问答基准测试上的实验结果表明，我们的方法提高了答案的准确性和推理的忠实性。本文强调溯因推理是增强RAG系统鲁棒性和可解释性的有前途的方向。 

---
# Memory- and Latency-Constrained Inference of Large Language Models via Adaptive Split Computing 

**Title (ZH)**: 内存和延迟受限的大语言模型推理通过自适应分割计算 

**Authors**: Mingyu Sung, Vikas Palakonda, Suhwan Im, Sunghwan Moon, Il-Min Kim, Sangseok Yun, Jae-Mo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04002)  

**Abstract**: Large language models (LLMs) have achieved near-human performance across diverse reasoning tasks, yet their deployment on resource-constrained Internet-of-Things (IoT) devices remains impractical due to massive parameter footprints and memory-intensive autoregressive decoding. While split computing offers a promising solution by partitioning model execution between edge devices and cloud servers, existing approaches fail to address the unique challenges of autoregressive inference, particularly the iterative token generation process and expanding key-value (KV) cache requirements. This work introduces the first autoregressive-aware split computing framework designed explicitly for LLM deployment on edge devices. Our approach makes three key contributions. First, we develop one-point split compression (OPSC), a mixed-precision quantization scheme that prevents out-of-memory failures by strategically partitioning models into front-end and back-end segments with different precision levels. Second, we propose a two-stage intermediate compression pipeline that combines threshold splitting (TS) and token-wise adaptive bit quantization (TAB-Q) to preserve accuracy-critical activations while dramatically reducing communication overhead. Third, we formulate a unified optimization framework that jointly selects optimal split points, quantization settings, and sequence lengths to satisfy strict memory and latency constraints. Extensive evaluations across diverse LLMs and hardware platforms demonstrate superior performance compared to state-of-the-art quantization methods, including SmoothQuant, OmniQuant, and Atom. The framework achieves a 1.49 inference speedup and significant communication overhead reduction while maintaining or improving model accuracy. 

**Abstract (ZH)**: 面向边缘设备的大语言模型自回归 Aware 分布式计算框架 

---
# Hybrid Fuzzing with LLM-Guided Input Mutation and Semantic Feedback 

**Title (ZH)**: LLM引导输入变异和语义反馈的混合 fuzzing 方法 

**Authors**: Shiyin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.03995)  

**Abstract**: Software fuzzing has become a cornerstone in automated vulnerability discovery, yet existing mutation strategies often lack semantic awareness, leading to redundant test cases and slow exploration of deep program states. In this work, I present a hybrid fuzzing framework that integrates static and dynamic analysis with Large Language Model (LLM)-guided input mutation and semantic feedback. Static analysis extracts control-flow and data-flow information, which is transformed into structured prompts for the LLM to generate syntactically valid and semantically diverse inputs. During execution, I augment traditional coverage-based feedback with semantic feedback signals-derived from program state changes, exception types, and output semantics-allowing the fuzzer to prioritize inputs that trigger novel program behaviors beyond mere code coverage. I implement our approach atop AFL++, combining program instrumentation with embedding-based semantic similarity metrics to guide seed selection. Evaluation on real-world open-source targets, including libpng, tcpdump, and sqlite, demonstrates that our method achieves faster time-to-first-bug, higher semantic diversity, and a competitive number of unique bugs compared to state-of-the-art fuzzers. This work highlights the potential of combining LLM reasoning with semantic-aware feedback to accelerate and deepen vulnerability discovery. 

**Abstract (ZH)**: 基于大型语言模型的静态与动态分析融合的启发式模糊测试框架 

---
# Direct Semantic Communication Between Large Language Models via Vector Translation 

**Title (ZH)**: 大型语言模型之间的向量翻译式直接语义通信 

**Authors**: Fu-Chun Yang, Jason Eshraghian  

**Link**: [PDF](https://arxiv.org/pdf/2511.03945)  

**Abstract**: In multi-agent settings, such as debate, reflection, or tool-calling, large language models (LLMs) pass messages as plain tokens, discarding most latent semantics. This constrains information transfer and adds unnecessary computational overhead. We form a latent bridge via vector translations, which use learned mappings that enable direct semantic exchange between representation spaces. A dual-encoder translator trained between Llama-2-7B and Mistral-7B-Instruct attains an average cosine alignment of 0.538. Injecting the translated vectors at 30 percent blending strength steers the target model's generation without destabilizing logits. Bidirectional evaluation shows a 2.01:1 transfer asymmetry, indicating that general-purpose models yield more transferable representations than instruction-tuned variants. This conservative injection preserves computational stability while demonstrating that cross-model latent communication is feasible, enabling collaborative AI systems that share meaning rather than tokens. 

**Abstract (ZH)**: 在多Agent设置中，如辩论、反思或工具调用，大型语言模型（LLMs）以原始令牌形式传递消息，丢弃了大部分潜在语义。这限制了信息传递并增加了不必要的计算开销。我们通过向量转换构造了一个潜在桥梁，利用学习到的映射使不同表示空间之间可以直接进行语义交换。在Llama-2-7B和Mistral-7B-Instruct之间训练的双编码器翻译器达到了平均余弦对齐值0.538。以30%的融合强度注入转换后的向量可以引导目标模型的生成而不破坏对数概率。双向评估显示2.01:1的转移不对称性，表明通用模型产生的表示比指令调整的变体更具转移性。这种保守的注入保持了计算稳定性，并证明了跨模型潜在通信的可行性，使协作AI系统能够共享意义而非令牌。 

---
# RLHF: A comprehensive Survey for Cultural, Multimodal and Low Latency Alignment Methods 

**Title (ZH)**: RLHF： Cultural、多模态和低延迟对齐方法综合研究 

**Authors**: Raghav Sharma, Manan Mehta, Sai Tiger Raina  

**Link**: [PDF](https://arxiv.org/pdf/2511.03939)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is the standard for aligning Large Language Models (LLMs), yet recent progress has moved beyond canonical text-based methods. This survey synthesizes the new frontier of alignment research by addressing critical gaps in multi-modal alignment, cultural fairness, and low-latency optimization. To systematically explore these domains, we first review foundational algo- rithms, including PPO, DPO, and GRPO, before presenting a detailed analysis of the latest innovations. By providing a comparative synthesis of these techniques and outlining open challenges, this work serves as an essential roadmap for researchers building more robust, efficient, and equitable AI systems. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）是大型语言模型（LLMs）对齐的标准，但 recent 进展已超越了传统的文本基方法。本文综述了对齐研究的新前沿，通过解决多模态对齐、文化公平性和低延迟优化的关键空白。为系统地探索这些领域，本文首先回顾了基础算法，包括 PPO、DPO 和 GRPO，然后详细分析了最新的创新成果。通过提供这些技术的比较综述并指出现有的开放性挑战，本文为研究人员构建更稳健、高效和公平的 AI 系统提供了必不可少的路线图。 

---
# PEFA-AI: Advancing Open-source LLMs for RTL generation using Progressive Error Feedback Agentic-AI 

**Title (ZH)**: PEFA-AI：利用渐进式错误反馈强化agents-AI以推进开源大语言模型在RTL生成中的应用 

**Authors**: Athma Narayanan, Mahesh Subedar, Omesh Tickoo  

**Link**: [PDF](https://arxiv.org/pdf/2511.03934)  

**Abstract**: We present an agentic flow consisting of multiple agents that combine specialized LLMs and hardware simulation tools to collaboratively complete the complex task of Register Transfer Level (RTL) generation without human intervention. A key feature of the proposed flow is the progressive error feedback system of agents (PEFA), a self-correcting mechanism that leverages iterative error feedback to progressively increase the complexity of the approach. The generated RTL includes checks for compilation, functional correctness, and synthesizable constructs. To validate this adaptive approach to code generation, benchmarking is performed using two opensource natural language-to-RTL datasets. We demonstrate the benefits of the proposed approach implemented on an open source agentic framework, using both open- and closed-source LLMs, effectively bridging the performance gap between them. Compared to previously published methods, our approach sets a new benchmark, providing state-of-the-art pass rates while being efficient in token counts. 

**Abstract (ZH)**: 我们提出了一种包含多个代理的自主流程，该流程结合了专门的LLM和硬件仿真工具，以在无需人工干预的情况下合作完成高层次综合（RTL）生成的复杂任务。所提出流程的关键特征是代理渐进错误反馈系统（PEFA），这是一种利用迭代错误反馈来自我纠正的机制，以逐步增加方法的复杂性。生成的RTL包含编译检查、功能正确性和综合构造。为了验证这种自适应的代码生成方法，我们使用两个开源自然语言到RTL数据集进行了基准测试。我们展示了在开源自主框架上实施所提出方法的好处，使用了开源和闭源的LLM，有效地弥合了它们之间的性能差距。与先前发表的方法相比，我们的方法设立了新的基准，提供了最先进的通过率，同时在token计数上具有高效性。 

---
# NVIDIA Nemotron Nano V2 VL 

**Title (ZH)**: NVIDIA Nemotron Nano V2 VL 

**Authors**: NVIDIA, Amala Sanjay Deshmukh, Kateryna Chumachenko, Tuomas Rintamaki, Matthieu Le, Tyler Poon, Danial Mohseni Taheri, Ilia Karmanov, Guilin Liu, Jarno Seppanen, Guo Chen, Karan Sapra, Zhiding Yu, Adi Renduchintala, Charles Wang, Peter Jin, Arushi Goel, Mike Ranzinger, Lukas Voegtle, Philipp Fischer, Timo Roman, Wei Ping, Boxin Wang, Zhuolin Yang, Nayeon Lee, Shaokun Zhang, Fuxiao Liu, Zhiqi Li, Di Zhang, Greg Heinrich, Hongxu, Song Han, Pavlo Molchanov, Parth Mannan, Yao Xu, Jane Polak Scowcroft, Tom Balough, Subhashree Radhakrishnan, Paris Zhang, Sean Cha, Ratnesh Kumar, Zaid Pervaiz Bhat, Jian Zhang, Darragh Hanley, Pritam Biswas, Jesse Oliver, Kevin Vasques, Roger Waleffe, Duncan Riach, Oluwatobi Olabiyi, Ameya Sunil Mahabaleshwarkar, Bilal Kartal, Pritam Gundecha, Khanh Nguyen, Alexandre Milesi, Eugene Khvedchenia, Ran Zilberstein, Ofri Masad, Natan Bagrov, Nave Assaf, Tomer Asida, Daniel Afrimi, Amit Zuker, Netanel Haber, Zhiyu Cheng, Jingyu, Nik Spirin, Maryam Moosaei, Roman Ageev, Vanshil Atul Shah, Yuting Wu, Daniel Korzekwa, Unnikrishnan Kizhakkemadam Sreekumar, Wanli Jiang, Padmavathy Subramanian, Alejandra Rico, Sandip Bhaskar, Saeid Motiian, Kedi Wu, Annie Surla, Chia-Chih Chen, Hayden Wolff, Matthew Feinberg, Melissa Corpuz, Marek Wawrzos, Eileen Long, Aastha Jhunjhunwala, Paul Hendricks, Farzan Memarian, Benika Hall, Xin-Yu Wang, David Mosallanezhad, Soumye Singhal, Luis Vega, Katherine Cheung, Krzysztof Pawelec, Michael Evans, Katherine Luna, Jie Lou, Erick Galinkin  

**Link**: [PDF](https://arxiv.org/pdf/2511.03929)  

**Abstract**: We introduce Nemotron Nano V2 VL, the latest model of the Nemotron vision-language series designed for strong real-world document understanding, long video comprehension, and reasoning tasks. Nemotron Nano V2 VL delivers significant improvements over our previous model, Llama-3.1-Nemotron-Nano-VL-8B, across all vision and text domains through major enhancements in model architecture, datasets, and training recipes. Nemotron Nano V2 VL builds on Nemotron Nano V2, a hybrid Mamba-Transformer LLM, and innovative token reduction techniques to achieve higher inference throughput in long document and video scenarios. We are releasing model checkpoints in BF16, FP8, and FP4 formats and sharing large parts of our datasets, recipes and training code. 

**Abstract (ZH)**: Nemotron Nano V2 VL：面向强现实世界文档理解、长视频理解和推理任务的Nemotron视觉-语言系列最新模型 

---
# Collaborative Agents for Automated Program Repair in Ruby 

**Title (ZH)**: Ruby中基于协作代理的自动化程序修复方法 

**Authors**: Nikta Akbarpour, Mahdieh Sadat Benis, Fatemeh Hendijani Fard, Ali Ouni, Mohamed Aymen Saied  

**Link**: [PDF](https://arxiv.org/pdf/2511.03925)  

**Abstract**: Automated Program Repair (APR) has advanced rapidly with Large Language Models (LLMs), but most existing methods remain computationally expensive, and focused on a small set of languages. Ruby, despite its widespread use in web development and the persistent challenges faced by its developers, has received little attention in APR research. In this paper, we introduce RAMP, a novel lightweight framework that formulates program repair as a feedback-driven, iterative process for Ruby. RAMP employs a team of collaborative agents that generate targeted tests, reflect on errors, and refine candidate fixes until a correct solution is found. Unlike prior approaches, RAMP is designed to avoid reliance on large multilingual repair databases or costly fine-tuning, instead operating directly on Ruby through lightweight prompting and test-driven feedback. Evaluation on the XCodeEval benchmark shows that RAMP achieves a pass@1 of 67% on Ruby, outper-forming prior approaches. RAMP converges quickly within five iterations, and ablation studies confirm that test generation and self-reflection are key drivers of its performance. Further analysis shows that RAMP is particularly effective at repairing wrong answers, compilation errors, and runtime errors. Our approach provides new insights into multi-agent repair strategies, and establishes a foundation for extending LLM-based debugging tools to under-studied languages. 

**Abstract (ZH)**: 基于大型语言模型的程序自动修复：Rubyaware的轻量级反馈驱动框架 

---
# Secure Code Generation at Scale with Reflexion 

**Title (ZH)**: 大规模安全代码生成与反射技术 

**Authors**: Arup Datta, Ahmed Aljohani, Hyunsook Do  

**Link**: [PDF](https://arxiv.org/pdf/2511.03898)  

**Abstract**: Large language models (LLMs) are now widely used to draft and refactor code, but code that works is not necessarily secure. We evaluate secure code generation using the Instruct Prime, which eliminated compliance-required prompts and cue contamination, and evaluate five instruction-tuned code LLMs using a zero-shot baseline and a three-round reflexion prompting approach. Security is measured using the Insecure Code Detector (ICD), and results are reported by measuring Repair, Regression, and NetGain metrics, considering the programming language and CWE family. Our findings show that insecurity remains common at the first round: roughly 25-33% of programs are insecure at a zero-shot baseline (t0 ). Weak cryptography/config-dependent bugs are the hardest to avoid while templated ones like XSS, code injection, and hard-coded secrets are handled more reliably. Python yields the highest secure rates; C and C# are the lowest, with Java, JS, PHP, and C++ in the middle. Reflexion prompting improves security for all models, improving average accuracy from 70.74% at t0 to 79.43% at t3 , with the largest gains in the first round followed by diminishing returns. The trends with Repair, Regression, and NetGain metrics show that applying one to two rounds produces most of the benefits. A replication package is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）现在广泛用于编写和重构代码，但可工作的代码未必是安全的。我们使用Instruct Prime评估安全代码生成，该模型去除了合规所需的提示和规范污染，并使用零-shot基线和三轮反思提示方法评估五种指令调优代码LLMs。安全性通过Insecure Code Detector (ICD)进行测量，并使用编程语言和CWE家族进行Repair、Regression和NetGain指标衡量。研究发现，在第一轮中安全性问题仍然常见：大约25-33%的程序在零-shot基线(t0)具有不安全性。弱加密/配置依赖的错误最难以避免，而像XSS、代码注入和硬编码密钥这样的模板错误则更可靠地得到了处理。Python具有最高的安全率；C和C#最低，Java、JS、PHP和C++在中间。反思提示方法可以提高所有模型的安全性，将平均准确率从t0的70.74%提高到t3的79.43%，第一轮改进最大，之后收益递减。修复、回归和净收益指标的趋势表明，应用一到两轮可获得大部分收益。可在以下链接获取复制包。 

---
# OMPILOT: Harnessing Transformer Models for Auto Parallelization to Shared Memory Computing Paradigms 

**Title (ZH)**: OMPILOT：利用变压器模型进行共享内存计算 paradigms 的自动并行化 

**Authors**: Arijit Bhattacharjee, Ali TehraniJamsaz, Le Chen, Niranjan Hasabnis, Mihai Capota, Nesreen Ahmed, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2511.03866)  

**Abstract**: Recent advances in large language models (LLMs) have significantly accelerated progress in code translation, enabling more accurate and efficient transformation across programming languages. While originally developed for natural language processing, LLMs have shown strong capabilities in modeling programming language syntax and semantics, outperforming traditional rule-based systems in both accuracy and flexibility. These models have streamlined cross-language conversion, reduced development overhead, and accelerated legacy code migration. In this paper, we introduce OMPILOT, a novel domain-specific encoder-decoder transformer tailored for translating C++ code into OpenMP, enabling effective shared-memory parallelization. OMPILOT leverages custom pre-training objectives that incorporate the semantics of parallel constructs and combines both unsupervised and supervised learning strategies to improve code translation robustness. Unlike previous work that focused primarily on loop-level transformations, OMPILOT operates at the function level to capture a wider semantic context. To evaluate our approach, we propose OMPBLEU, a novel composite metric specifically crafted to assess the correctness and quality of OpenMP parallel constructs, addressing limitations in conventional translation metrics. 

**Abstract (ZH)**: Recent advances in 大型语言模型（LLMs）显著加速了代码翻译的进展，使不同编程语言间的准确且高效的转换成为可能。尽管最初是为自然语言处理开发的，LLMs在建模编程语言语法和语义方面表现出强大的能力，其准确性和灵活性超过了传统的基于规则的系统。这些模型简化了跨语言转换，降低了开发成本，并加速了遗留代码的迁移。在本文中，我们介绍了OMPILOT，这是一种专门用于将C++代码翻译为OpenMP的新型领域特定编码器-解码器变换模型，以实现有效的共享内存并行化。OMPILOT利用了包含并行构造语义的自定义预训练目标，并结合无监督和有监督学习策略以提高代码翻译的稳健性。与之前主要关注循环级转换的工作不同，OMPILOT在函数级别操作以捕获更广泛的语义上下文。为了评估我们的方法，我们提出了OMPBLEU，这是一种新型复合度量标准，特别设计用于评估OpenMP并行构造的正确性和质量，解决了传统翻译度量标准的局限性。 

---
# PLLuM: A Family of Polish Large Language Models 

**Title (ZH)**: PLLuM：波兰大型语言模型家族 

**Authors**: Jan Kocoń, Maciej Piasecki, Arkadiusz Janz, Teddy Ferdinan, Łukasz Radliński, Bartłomiej Koptyra, Marcin Oleksy, Stanisław Woźniak, Paweł Walkowiak, Konrad Wojtasik, Julia Moska, Tomasz Naskręt, Bartosz Walkowiak, Mateusz Gniewkowski, Kamil Szyc, Dawid Motyka, Dawid Banach, Jonatan Dalasiński, Ewa Rudnicka, Bartłomiej Alberski, Tomasz Walkowiak, Aleksander Szczęsny, Maciej Markiewicz, Tomasz Bernaś, Hubert Mazur, Kamil Żyta, Mateusz Tykierko, Grzegorz Chodak, Tomasz Kajdanowicz, Przemysław Kazienko, Agnieszka Karlińska, Karolina Seweryn, Anna Kołos, Maciej Chrabąszcz, Katarzyna Lorenc, Aleksandra Krasnodębska, Artur Wilczek, Katarzyna Dziewulska, Paula Betscher, Zofia Cieślińska, Katarzyna Kowol, Daria Mikoś, Maciej Trzciński, Dawid Krutul, Marek Kozłowski, Sławomir Dadas, Rafał Poświata, Michał Perełkiewicz, Małgorzata Grębowiec, Maciej Kazuła, Marcin Białas, Roman Roszko, Danuta Roszko, Jurgita Vaičenonienė, Andrius Utka, Paweł Levchuk, Paweł Kowalski, Irena Prawdzic-Jankowska, Maciej Ogrodniczuk, Monika Borys, Anna Bulińska, Wiktoria Gumienna, Witold Kieraś, Dorota Komosińska, Katarzyna Krasnowska-Kieraś, Łukasz Kobyliński, Martyna Lewandowska, Marek Łaziński, Mikołaj Łątkowski, Dawid Mastalerz, Beata Milewicz, Agnieszka Anna Mykowiecka, Angelika Peljak-Łapińska, Sandra Penno, Zuzanna Przybysz, Michał Rudolf, Piotr Rybak, Karolina Saputa, Aleksandra Tomaszewska, Aleksander Wawer, Marcin Woliński, Joanna Wołoszyn, Alina Wróblewska, Bartosz Żuk, Filip Żarnecki, Konrad Kaczyński, Anna Cichosz, Zuzanna Deckert, Monika Garnys, Izabela Grabarczyk, Wojciech Janowski, Sylwia Karasińska, Aleksandra Kujawiak, Piotr Misztela, Maria Szymańska, Karolina Walkusz, Igor Siek, Jakub Kwiatkowski, Piotr Pęzik  

**Link**: [PDF](https://arxiv.org/pdf/2511.03823)  

**Abstract**: Large Language Models (LLMs) play a central role in modern artificial intelligence, yet their development has been primarily focused on English, resulting in limited support for other languages. We present PLLuM (Polish Large Language Model), the largest open-source family of foundation models tailored specifically for the Polish language. Developed by a consortium of major Polish research institutions, PLLuM addresses the need for high-quality, transparent, and culturally relevant language models beyond the English-centric commercial landscape. We describe the development process, including the construction of a new 140-billion-token Polish text corpus for pre-training, a 77k custom instructions dataset, and a 100k preference optimization dataset. A key component is a Responsible AI framework that incorporates strict data governance and a hybrid module for output correction and safety filtering. We detail the models' architecture, training procedures, and alignment techniques for both base and instruction-tuned variants, and demonstrate their utility in a downstream task within public administration. By releasing these models publicly, PLLuM aims to foster open research and strengthen sovereign AI technologies in Poland. 

**Abstract (ZH)**: Polish Large Language Model (PLLuM): A Transparent and Culturally Relevant Foundation Model Tailored for the Polish Language 

---
# Expert Evaluation of LLM World Models: A High-$T_c$ Superconductivity Case Study 

**Title (ZH)**: 专家评估LLM世界模型：高-$T_c$超导性案例研究 

**Authors**: Haoyu Guo, Maria Tikhanovskaya, Paul Raccuglia, Alexey Vlaskin, Chris Co, Daniel J. Liebling, Scott Ellsworth, Matthew Abraham, Elizabeth Dorfman, N. P. Armitage, Chunhan Feng, Antoine Georges, Olivier Gingras, Dominik Kiese, Steven A. Kivelson, Vadim Oganesyan, B. J. Ramshaw, Subir Sachdev, T. Senthil, J. M. Tranquada, Michael P. Brenner, Subhashini Venugopalan, Eun-Ah Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.03782)  

**Abstract**: Large Language Models (LLMs) show great promise as a powerful tool for scientific literature exploration. However, their effectiveness in providing scientifically accurate and comprehensive answers to complex questions within specialized domains remains an active area of research. Using the field of high-temperature cuprates as an exemplar, we evaluate the ability of LLM systems to understand the literature at the level of an expert. We construct an expert-curated database of 1,726 scientific papers that covers the history of the field, and a set of 67 expert-formulated questions that probe deep understanding of the literature. We then evaluate six different LLM-based systems for answering these questions, including both commercially available closed models and a custom retrieval-augmented generation (RAG) system capable of retrieving images alongside text. Experts then evaluate the answers of these systems against a rubric that assesses balanced perspectives, factual comprehensiveness, succinctness, and evidentiary support. Among the six systems two using RAG on curated literature outperformed existing closed models across key metrics, particularly in providing comprehensive and well-supported answers. We discuss promising aspects of LLM performances as well as critical short-comings of all the models. The set of expert-formulated questions and the rubric will be valuable for assessing expert level performance of LLM based reasoning systems. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在科学文献探索方面展现出巨大的潜力，但在提供科学准确且全面的回答以应对复杂的专业领域问题方面仍需进一步研究。以高温铜氧化物为例，我们评估了LLM系统在理解文献方面达到专家水平的能力。我们构建了一个由1,726篇科学论文组成的专家精选数据库，涵盖了该领域的历史，并制定了一套67个由专家提出的问题，以探索对文献的深入理解。然后，我们评估了六种不同的LLM系统回答这些问题的能力，包括商业可用的封闭模型以及一种能够检索图像和文本的定制检索增强生成（RAG）系统。随后，专家们根据平衡视角、事实全面性、简明性和证据支持的标准评估了这些系统的答案。在六种系统中，使用RAG处理精选文献的两种系统在关键指标上超越了现有的封闭模型，尤其是在提供全面且支持充分的回答方面。我们讨论了LLM表现的有希望方面以及所有模型的关键不足。由专家制定的问题集和评估标准将对评估基于LLM的推理系统的专家级性能具有价值。 

---
# Leveraging LLM-based agents for social science research: insights from citation network simulations 

**Title (ZH)**: 基于LLM的代理在社会科学研究中的应用：引用网络模拟的洞见 

**Authors**: Jiarui Ji, Runlin Lei, Xuchen Pan, Zhewei Wei, Hao Sun, Yankai Lin, Xu Chen, Yongzheng Yang, Yaliang Li, Bolin Ding, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2511.03758)  

**Abstract**: The emergence of Large Language Models (LLMs) demonstrates their potential to encapsulate the logic and patterns inherent in human behavior simulation by leveraging extensive web data pre-training. However, the boundaries of LLM capabilities in social simulation remain unclear. To further explore the social attributes of LLMs, we introduce the CiteAgent framework, designed to generate citation networks based on human-behavior simulation with LLM-based agents. CiteAgent successfully captures predominant phenomena in real-world citation networks, including power-law distribution, citational distortion, and shrinking diameter. Building on this realistic simulation, we establish two LLM-based research paradigms in social science: LLM-SE (LLM-based Survey Experiment) and LLM-LE (LLM-based Laboratory Experiment). These paradigms facilitate rigorous analyses of citation network phenomena, allowing us to validate and challenge existing theories. Additionally, we extend the research scope of traditional science of science studies through idealized social experiments, with the simulation experiment results providing valuable insights for real-world academic environments. Our work demonstrates the potential of LLMs for advancing science of science research in social science. 

**Abstract (ZH)**: 大型语言模型（LLMs）的出现展示了其通过利用大量网络数据预训练来封装人类行为模拟中的逻辑和模式的潜力。然而，LLMs在社会模拟中的能力边界仍然不清楚。为了进一步探索LLMs的社会属性，我们引入了CiteAgent框架，该框架基于基于LLM的代理生成引文网络。CiteAgent成功捕捉到了现实世界引文网络中的主要现象，包括幂律分布、引文失真和直径缩小。基于这种现实的模拟，我们建立了两个基于LLM的社会科学研究范式：LLM-SE（基于LLM的调查实验）和LLM-LE（基于LLM的实验室实验）。这些范式促进了对引文网络现象的严格分析，使我们能够验证和挑战现有理论。此外，我们通过理想化的社会实验扩展了传统科学研究学的研究范围，模拟实验结果为现实学术环境提供了宝贵的见解。我们的工作展示了LLMs在社会科学研究中的科学学研究方面的发展潜力。 

---
# Laugh, Relate, Engage: Stylized Comment Generation for Short Videos 

**Title (ZH)**: 笑一笑，聊一聊，更精彩：短视频风格化评论生成 

**Authors**: Xuan Ouyang, Senan Wang, Bouzhou Wang, Siyuan Xiahou, Jinrong Zhou, Yuekang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.03757)  

**Abstract**: Short-video platforms have become a central medium in the modern Internet landscape, where efficient information delivery and strong interactivity are reshaping user engagement and cultural dissemination. Among the various forms of user interaction, comments play a vital role in fostering community participation and enabling content re-creation. However, generating comments that are both compliant with platform guidelines and capable of exhibiting stylistic diversity and contextual awareness remains a significant challenge. We introduce LOLGORITHM, a modular multi-agent system (MAS) designed for controllable short-video comment generation. The system integrates video segmentation, contextual and affective analysis, and style-aware prompt construction. It supports six distinct comment styles: puns (homophones), rhyming, meme application, sarcasm (irony), plain humor, and content extraction. Powered by a multimodal large language model (MLLM), LOLGORITHM directly processes video inputs and achieves fine-grained style control through explicit prompt markers and few-shot examples. To support development and evaluation, we construct a bilingual dataset using official APIs from Douyin (Chinese) and YouTube (English), covering five popular video genres: comedy skits, daily life jokes, funny animal clips, humorous commentary, and talk shows. Evaluation combines automated metrics originality, relevance, and style conformity with a large-scale human preference study involving 40 videos and 105 participants. Results show that LOLGORITHM significantly outperforms baseline models, achieving preference rates of over 90% on Douyin and 87.55% on YouTube. This work presents a scalable and culturally adaptive framework for stylized comment generation on short-video platforms, offering a promising path to enhance user engagement and creative interaction. 

**Abstract (ZH)**: 短视频平台已成为现代互联网景观中的核心媒体，高效的信息发布和强大的互动性正在重塑用户参与和文化传播。在各种用户互动形式中，评论在促进社区参与和内容再创造方面发挥着重要作用。然而，生成既符合平台规范又能体现风格多样性和情境意识的评论仍然是一项重大挑战。我们介绍了LOALGORITHM，这是一种用于可控短视频评论生成的模块化多智能体系统（MAS）。该系统结合了视频分割、上下文和情感分析以及风格感知的提示构建。它支持六种不同的评论风格：谐音、押韵、 meme 应用、反讽（讽刺）、朴素幽默和内容提取。借助多模态大语言模型（MLLM），LOALGORITHM 直接处理视频输入，并通过明确的提示标记和少量示例实现精细的风格控制。为了支持开发和评估，我们使用抖音（中文）和YouTube（英文）的官方API构建了一个双语数据集，涵盖五种流行视频类型：喜剧小品、日常生活笑话、搞笑动物片段、幽默评论和脱口秀。评估结合了自动评估指标（原创性、相关性和风格一致性）和大规模人类偏爱研究，涉及40个视频和105名参与者。结果显示，LOALGORITHM 显著优于基线模型，抖音上的偏好率为90%以上，YouTube上的偏好率为87.55%。本文提出了一种可扩展且文化适应性强的框架，用于短视频平台上的风格化评论生成，为增强用户参与和创造性互动提供了有希望的途径。 

---
# Beyond Chat: a Framework for LLMs as Human-Centered Support Systems 

**Title (ZH)**: 超越聊天：一种以人类为中心的支持系统框架用于大语言模型 

**Authors**: Zhiyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.03729)  

**Abstract**: Large language models are moving beyond transactional question answering to act as companions, coaches, mediators, and curators that scaffold human growth, decision-making, and well-being. This paper proposes a role-based framework for human-centered LLM support systems, compares real deployments across domains, and identifies cross-cutting design principles: transparency, personalization, guardrails, memory with privacy, and a balance of empathy and reliability. It outlines evaluation metrics that extend beyond accuracy to trust, engagement, and longitudinal outcomes. It also analyzes risks including over-reliance, hallucination, bias, privacy exposure, and unequal access, and proposes future directions spanning unified evaluation, hybrid human-AI models, memory architectures, cross-domain benchmarking, and governance. The goal is to support responsible integration of LLMs in sensitive settings where people need accompaniment and guidance, not only answers. 

**Abstract (ZH)**: 大型语言模型正从交易性的问答扩展为同伴、教练、调解人和 curator，助力人类的成长、决策和福祉。本文提出了一种基于角色的大规模语言模型人本支持系统框架，比较了不同领域的实际部署，明确了跨领域的设计原则：透明度、个性化、护栏机制、隐私保护的内存功能以及同理心与可靠性的平衡。论文还列出了评估指标，不仅包括准确性，还有信任度、参与度和长期效果。同时分析了过度依赖、幻想、偏见、隐私泄露和访问不平等等风险，并提出了统一评估、混合人机模型、记忆架构、跨域基准测试和治理等未来方向，目标是在人们需要陪伴和指导而非仅需要答案的敏感环境中促进负责任的大规模语言模型集成。 

---
# MazeMate: An LLM-Powered Chatbot to Support Computational Thinking in Gamified Programming Learning 

**Title (ZH)**: MazeMate：一个支持游戏化编程学习中计算思维的大型语言模型驱动聊天机器人 

**Authors**: Chenyu Hou, Hua Yu, Gaoxia Zhu, John Derek Anas, Jiao Liu, Yew Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2511.03727)  

**Abstract**: Computational Thinking (CT) is a foundational problem-solving skill, and gamified programming environments are a widely adopted approach to cultivating it. While large language models (LLMs) provide on-demand programming support, current applications rarely foster CT development. We present MazeMate, an LLM-powered chatbot embedded in a 3D Maze programming game, designed to deliver adaptive, context-sensitive scaffolds aligned with CT processes in maze solving and maze design. We report on the first classroom implementation with 247 undergraduates. Students rated MazeMate as moderately helpful, with higher perceived usefulness for maze solving than for maze design. Thematic analysis confirmed support for CT processes such as decomposition, abstraction, and algorithmic thinking, while also revealing limitations in supporting maze design, including mismatched suggestions and fabricated algorithmic solutions. These findings demonstrate the potential of LLM-based scaffolding to support CT and underscore directions for design refinement to enhance MazeMate usability in authentic classrooms. 

**Abstract (ZH)**: 基于大型语言模型的迷宫模拟器：支持计算思维的聊天机器人设计与初步课堂实施 

---
