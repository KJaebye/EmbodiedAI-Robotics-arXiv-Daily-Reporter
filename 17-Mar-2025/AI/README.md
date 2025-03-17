# Broaden your SCOPE! Efficient Multi-turn Conversation Planning for LLMs using Semantic Space 

**Title (ZH)**: 扩展您的SCOPE！使用语义空间为LLM高效规划多轮对话 prosecize 

**Authors**: Zhiliang Chen, Xinyuan Niu, Chuan-Sheng Foo, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2503.11586)  

**Abstract**: Large language models (LLMs) are used in chatbots or AI assistants to hold conversations with a human user. In such applications, the quality (e.g., user engagement, safety) of a conversation is important and can only be exactly known at the end of the conversation. To maximize its expected quality, conversation planning reasons about the stochastic transitions within a conversation to select the optimal LLM response at each turn. Existing simulation-based conversation planning algorithms typically select the optimal response by simulating future conversations with a large number of LLM queries at every turn. However, this process is extremely time-consuming and hence impractical for real-time conversations. This paper presents a novel approach called Semantic space COnversation Planning with improved Efficiency (SCOPE) that exploits the dense semantic representation of conversations to perform conversation planning efficiently. In particular, SCOPE models the stochastic transitions in conversation semantics and their associated rewards to plan entirely within the semantic space. This allows us to select the optimal LLM response at every conversation turn without needing additional LLM queries for simulation. As a result, SCOPE can perform conversation planning 70 times faster than conventional simulation-based planning algorithms when applied to a wide variety of conversation starters and two reward functions seen in the real world, yet achieving a higher reward within a practical planning budget. Our code can be found at: this https URL. 

**Abstract (ZH)**: 基于语义空间的高效对话规划（Semantic Space COnversation Planning with Improved Efficiency, SCOPE） 

---
# Prompt Injection Detection and Mitigation via AI Multi-Agent NLP Frameworks 

**Title (ZH)**: 基于AI多Agent NLP框架的Prompt注入检测与缓解 

**Authors**: Diego Gosmar, Deborah A. Dahl, Dario Gosmar  

**Link**: [PDF](https://arxiv.org/pdf/2503.11517)  

**Abstract**: Prompt injection constitutes a significant challenge for generative AI systems by inducing unintended outputs. We introduce a multi-agent NLP framework specifically designed to address prompt injection vulnerabilities through layered detection and enforcement mechanisms. The framework orchestrates specialized agents for generating responses, sanitizing outputs, and enforcing policy compliance. Evaluation on 500 engineered injection prompts demonstrates a marked reduction in injection success and policy breaches. Novel metrics, including Injection Success Rate (ISR), Policy Override Frequency (POF), Prompt Sanitization Rate (PSR), and Compliance Consistency Score (CCS), are proposed to derive a composite Total Injection Vulnerability Score (TIVS). The system utilizes the OVON (Open Voice Network) framework for inter-agent communication via structured JSON messages, extending a previously established multi-agent architecture from hallucination mitigation to address the unique challenges of prompt injection. 

**Abstract (ZH)**: Prompt注入构成了对生成式AI系统的重大挑战，可能导致意外输出。我们提出了一种多代理NLP框架，专门设计用于通过分层检测和执行机制来应对prompt注入漏洞。该框架协调专门的代理以生成响应、净化输出并确保政策合规。在500个工程化的注入提示下进行的评估显示了明显的注入成功率降低和政策违规率减少。提出了新的指标，包括注入成功率（ISR）、策略覆盖频率（POF）、提示净化率（PSR）和合规一致性评分（CCS），以计算总注入漏洞评分（TIVS）。系统利用OVON（Open Voice Network）框架通过结构化的JSON消息进行代理间通信，扩展了一种先前用于幻觉减轻的多代理架构，以应对提示注入的独特挑战。 

---
# Heterogeneous Causal Discovery of Repeated Undesirable Health Outcomes 

**Title (ZH)**: 异质因果发现重复不良健康结果 

**Authors**: Shishir Adhikari, Guido Muscioni, Mark Shapiro, Plamen Petrov, Elena Zheleva  

**Link**: [PDF](https://arxiv.org/pdf/2503.11477)  

**Abstract**: Understanding factors triggering or preventing undesirable health outcomes across patient subpopulations is essential for designing targeted interventions. While randomized controlled trials and expert-led patient interviews are standard methods for identifying these factors, they can be time-consuming and infeasible. Causal discovery offers an alternative to conventional approaches by generating cause-and-effect hypotheses from observational data. However, it often relies on strong or untestable assumptions, which can limit its practical application. This work aims to make causal discovery more practical by considering multiple assumptions and identifying heterogeneous effects. We formulate the problem of discovering causes and effect modifiers of an outcome, where effect modifiers are contexts (e.g., age groups) with heterogeneous causal effects. Then, we present a novel, end-to-end framework that incorporates an ensemble of causal discovery algorithms and estimation of heterogeneous effects to discover causes and effect modifiers that trigger or inhibit the outcome. We demonstrate that the ensemble approach improves robustness by enhancing recall of causal factors while maintaining precision. Our study examines the causes of repeat emergency room visits for diabetic patients and hospital readmissions for ICU patients. Our framework generates causal hypotheses consistent with existing literature and can help practitioners identify potential interventions and patient subpopulations to focus on. 

**Abstract (ZH)**: 理解触发或预防特定健康结果的因子对于不同患者亚群体至关重要，有助于设计针对性的干预措施。虽然随机对照试验和专家主导的患者访谈是识别这些因子的标准方法，但它们常常耗时且不可行。因果发现提供了一种替代传统的途径，通过从观察数据生成因果效应假说。然而，它常常依赖于强有力的或无法验证的假设，这限制了其实际应用。本研究旨在通过考虑多种假设并识别异质效应来让因果发现更具实用性。我们形式化了发现结果原因和效应修饰者的问题，其中效应修饰者是具有异质因果效应的上下文（例如，年龄组）。然后，我们提出了一种新颖的一站式框架，结合因果发现算法的ensemble方法和异质效应的估计，以发现触发或抑制结果的原因和效应修饰者。我们证明ensemble方法通过提高因果因子召回率的同时保持高精确度，增强了鲁棒性。我们的研究分析了糖尿病患者重复急诊和ICU患者院内再入院的原因，并生成与现有文献一致的因果假说，有助于实践者识别潜在的干预措施和关注的患者亚群体。 

---
# Integrating LLMs in Gamified Systems 

**Title (ZH)**: 在游戏化系统中集成大语言模型 

**Authors**: Carlos J. Costa  

**Link**: [PDF](https://arxiv.org/pdf/2503.11458)  

**Abstract**: In this work, a thorough mathematical framework for incorporating Large Language Models (LLMs) into gamified systems is presented with an emphasis on improving task dynamics, user engagement, and reward systems. Personalized feedback, adaptive learning, and dynamic content creation are all made possible by integrating LLMs and are crucial for improving user engagement and system performance. A simulated environment tests the framework's adaptability and demonstrates its potential for real-world applications in various industries, including business, healthcare, and education. The findings demonstrate how LLMs can offer customized experiences that raise system effectiveness and user retention. This study also examines the difficulties this framework aims to solve, highlighting its importance in maximizing involvement and encouraging sustained behavioral change in a range of sectors. 

**Abstract (ZH)**: 在这种工作中，提出了一个全面的数学框架，将大型语言模型（LLMs）纳入 gamified 系统，并强调改进任务动力学、用户参与度和奖励系统。通过集成 LLMs，个性化反馈、自适应学习和动态内容创建成为可能，对于提高用户参与度和系统性能至关重要。模拟环境测试了该框架的适应性，并展示了其在多个行业（包括商业、医疗保健和教育）中实现实际应用的潜力。研究结果表明，LLMs 可以提供定制化的体验，提高系统的有效性和用户留存率。此外，该研究还探讨了框架旨在解决的困难，强调了其在各行业中最大化参与度和促进持续行为改变的重要性。 

---
# Preference Elicitation for Multi-objective Combinatorial Optimization with Active Learning and Maximum Likelihood Estimation 

**Title (ZH)**: 基于主动学习和最大似然估计的多目标组合优化的偏好 elicitation 

**Authors**: Marianne Defresne, Jayanta Mandi, Tias Guns  

**Link**: [PDF](https://arxiv.org/pdf/2503.11435)  

**Abstract**: Real-life combinatorial optimization problems often involve several conflicting objectives, such as price, product quality and sustainability. A computationally-efficient way to tackle multiple objectives is to aggregate them into a single-objective function, such as a linear combination. However, defining the weights of the linear combination upfront is hard; alternatively, the use of interactive learning methods that ask users to compare candidate solutions is highly promising. The key challenges are to generate candidates quickly, to learn an objective function that leads to high-quality solutions and to do so with few user interactions. We build upon the Constructive Preference Elicitation framework and show how each of the three properties can be improved: to increase the interaction speed we investigate using pools of (relaxed) solutions, to improve the learning we adopt Maximum Likelihood Estimation of a Bradley-Terry preference model; and to reduce the number of user interactions, we select the pair of candidates to compare with an ensemble-based acquisition function inspired from Active Learning. Our careful experimentation demonstrates each of these improvements: on a PC configuration task and a realistic multi-instance routing problem, our method selects queries faster, needs fewer queries and synthesizes higher-quality combinatorial solutions than previous CPE methods. 

**Abstract (ZH)**: 现实生活中组合优化问题往往涉及多个相互冲突的目标，如价格、产品质量和可持续性。一种计算效率高的方法是将这些目标聚合为一个目标函数，例如线性组合。然而，提前定义线性组合的权重颇具挑战性；相反，采用交互学习方法，要求用户比较候选解是极具前景的。关键挑战在于快速生成候选解、学习能够产生高质量解决方案的目标函数，并且在较少用户交互的情况下实现这一点。我们基于Constructive Preference Elicitation框架，展示如何改进上述三种特性：通过使用候选解池（放宽约束的解）来提高交互速度；通过采用Bradley-Terry偏好模型的最大似然估计来改进学习；通过使用基于集成的获取函数，该函数借鉴主动学习的方法来减少用户交互次数。我们仔细的实验表明这些改进的有效性：在PC配置任务和一个实际的多实例路由问题上，我们的方法更快地选择了查询、需要更少的查询次数，并且能够合成出更高质量的组合解，比之前的CPE方法更为优异。 

---
# Optimizing Large Language Models for Detecting Symptoms of Comorbid Depression or Anxiety in Chronic Diseases: Insights from Patient Messages 

**Title (ZH)**: 优化大规模语言模型以检测慢性疾病共病抑郁或焦虑的症状：基于患者消息的见解 

**Authors**: Jiyeong Kim, Stephen P. Ma, Michael L. Chen, Isaac R. Galatzer-Levy, John Torous, Peter J. van Roessel, Christopher Sharp, Michael A. Pfeffer, Carolyn I. Rodriguez, Eleni Linos, Jonathan H. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11384)  

**Abstract**: Patients with diabetes are at increased risk of comorbid depression or anxiety, complicating their management. This study evaluated the performance of large language models (LLMs) in detecting these symptoms from secure patient messages. We applied multiple approaches, including engineered prompts, systemic persona, temperature adjustments, and zero-shot and few-shot learning, to identify the best-performing model and enhance performance. Three out of five LLMs demonstrated excellent performance (over 90% of F-1 and accuracy), with Llama 3.1 405B achieving 93% in both F-1 and accuracy using a zero-shot approach. While LLMs showed promise in binary classification and handling complex metrics like Patient Health Questionnaire-4, inconsistencies in challenging cases warrant further real-life assessment. The findings highlight the potential of LLMs to assist in timely screening and referrals, providing valuable empirical knowledge for real-world triage systems that could improve mental health care for patients with chronic diseases. 

**Abstract (ZH)**: 糖尿病患者存在并发抑郁或焦虑的风险，加剧了其管理复杂性。本研究评估了大型语言模型（LLMs）在从安全患者信息中检测这些症状方面的性能。我们采用了包括工程化提示、系统性人设、温度调整以及零shot和少shot学习等多种方法，以识别出最佳性能模型并提升其性能。三款大型语言模型表现优异（F-1分数和准确率均超过90%），Llama 3.1 405B在零shot方法下F-1分数和准确率均达到93%。尽管大型语言模型在二分类和处理患者健康问卷-4这样的复杂指标上显示出了潜力，但在复杂案例中的不一致性仍需进一步的实际生活评估。研究结果强调了大型语言模型在及时筛查和转诊方面潜在的作用，并为改善慢性病患者的心理健康护理提供了宝贵的实证知识，可能帮助现实世界的分诊系统改进。 

---
# Collaboration is all you need: LLM Assisted Safe Code Translation 

**Title (ZH)**: 你需要的全部是协作：LLM辅助的安全代码翻译 

**Authors**: Rabimba Karanjai, Sam Blackshear, Lei Xu, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11237)  

**Abstract**: This paper introduces UniTranslator, a visionary framework that re-imagines code translation as a collaborative endeavor among multiple, compact LLMs. By orchestrating the interaction of specialized agents, each focused on different aspects of the translation process and grounded in a deep understanding of programming concepts, UniTranslator achieves a level of accuracy and efficiency that rivals larger, monolithic models. Our preliminary evaluation demonstrates the potential of UniTranslator to overcome the limitations of existing approaches and unlock the power of smaller LLMs for complex code translation tasks. We explore the effectiveness of this dynamic multi-agent paradigm in handling diverse language pairs, including low-resource languages, and in mitigating common issues such as code artifacts and hallucinations through the use of Natural Language Inference (NLI) grounding and iterative feedback mechanisms 

**Abstract (ZH)**: UniTranslator：一个重新构想的多重紧凑LLM协作代码翻译框架 

---
# GKG-LLM: A Unified Framework for Generalized Knowledge Graph Construction 

**Title (ZH)**: GKG-LLM：通用知识图谱构建的统一框架 

**Authors**: Jian Zhang, Bifan Wei, Shihao Qi, haiping Zhu, Jun Liu, Qika Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.11227)  

**Abstract**: The construction of Generalized Knowledge Graph (GKG), including knowledge graph, event knowledge graph and commonsense knowledge graph, is fundamental for various natural language processing tasks. Current studies typically construct these types of graph separately, overlooking holistic insights and potential unification that could be beneficial in computing resources and usage perspectives. However, a key challenge in developing a unified framework for GKG is obstacles arising from task-specific differences. In this study, we propose a unified framework for constructing generalized knowledge graphs to address this challenge. First, we collect data from 15 sub-tasks in 29 datasets across the three types of graphs, categorizing them into in-sample, counter-task, and out-of-distribution (OOD) data. Then, we propose a three-stage curriculum learning fine-tuning framework, by iteratively injecting knowledge from the three types of graphs into the Large Language Models. Extensive experiments show that our proposed model improves the construction of all three graph types across in-domain, OOD and counter-task data. 

**Abstract (ZH)**: 广义知识图谱（GKG）的构建，包括知识图谱、事件知识图谱和常识知识图谱，是各种自然语言处理任务的基础。当前研究通常分别构建这几种类型的图谱，忽视了整体洞察和潜在的统一体系，这在计算资源和使用视角上可能有益。然而，为GKG开发统一框架的关键挑战来自于任务特定差异导致的障碍。本研究提出一种统一框架以应对这一挑战。首先，我们从涵盖三种类型图谱的29个数据集中15个子任务中收集数据，并将其分类为同分布、反任务和分布外（OOD）数据。然后，我们提出了一种三阶段课程学习微调框架，通过迭代地将三种类型图谱的知识注入到大型语言模型中。 extensive实验表明，我们提出的模型能够提高三种类型图谱在同分布、OOD和反任务数据上的构建质量。 

---
# Can Large Reasoning Models do Analogical Reasoning under Perceptual Uncertainty? 

**Title (ZH)**: 大型推理模型在感知不确定性下的类比推理能力如何？ 

**Authors**: Giacomo Camposampiero, Michael Hersche, Roger Wattenhofer, Abu Sebastian, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11207)  

**Abstract**: This work presents a first evaluation of two state-of-the-art Large Reasoning Models (LRMs), OpenAI's o3-mini and DeepSeek R1, on analogical reasoning, focusing on well-established nonverbal human IQ tests based on Raven's progressive matrices. We benchmark with the I-RAVEN dataset and its more difficult extension, I-RAVEN-X, which tests the ability to generalize to longer reasoning rules and ranges of the attribute values. To assess the influence of visual uncertainties on these nonverbal analogical reasoning tests, we extend the I-RAVEN-X dataset, which otherwise assumes an oracle perception. We adopt a two-fold strategy to simulate this imperfect visual perception: 1) we introduce confounding attributes which, being sampled at random, do not contribute to the prediction of the correct answer of the puzzles and 2) smoothen the distributions of the input attributes' values. We observe a sharp decline in OpenAI's o3-mini task accuracy, dropping from 86.6% on the original I-RAVEN to just 17.0% -- approaching random chance -- on the more challenging I-RAVEN-X, which increases input length and range and emulates perceptual uncertainty. This drop occurred despite spending 3.4x more reasoning tokens. A similar trend is also observed for DeepSeek R1: from 80.6% to 23.2%. On the other hand, a neuro-symbolic probabilistic abductive model, ARLC, that achieves state-of-the-art performances on I-RAVEN, can robustly reason under all these out-of-distribution tests, maintaining strong accuracy with only a modest reduction from 98.6% to 88.0%. Our code is available at this https URL. 

**Abstract (ZH)**: 本研究首次评估了两个最先进的大型推理模型（LRMs）——OpenAI的o3-mini和DeepSeek R1在类比推理中的表现，重点关注基于瑞文进步矩阵的标准化非言语人类IQ测试。我们使用I-RAVEN数据集及其更具挑战性的扩展I-RAVEN-X进行基准测试，后者旨在测试对较长推理规则和属性值范围的推广能力。为了评估视觉不确定性对这些非言语类比推理测试的影响，我们扩展了I-RAVEN-X数据集，该数据集在其他方面假定完美感知。我们采用两步策略来模拟这种不完美的视觉感知：1）引入混淆属性，这些属性随机采样，并不参与谜题正确答案的预测；2）平滑输入属性值的分布。我们观察到OpenAI的o3-mini任务准确率急剧下降，从原始I-RAVEN的86.6%降至更具挑战性的I-RAVEN-X的17.0%，接近随机猜测，尽管推理令牌使用量增加了3.4倍。类似的趋势也出现在DeepSeek R1中，从80.6%降至23.2%。另一方面，一种神经符号概率归纳模型ARLC在I-RAVEN上实现了最先进的性能，能够在所有这些分布外测试中稳健推理，且准确率仅从98.6%轻微下降到88.0%。我们的代码可在以下链接获取。 

---
# Large Reasoning Models in Agent Scenarios: Exploring the Necessity of Reasoning Capabilities 

**Title (ZH)**: 大型推理模型在代理场景中的探索：推理能力的必要性研究 

**Authors**: Xueyang Zhou, Guiyao Tie, Guowen Zhang, Weidong Wang, Zhigang Zuo, Di Wu, Duanfeng Chu, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.11074)  

**Abstract**: The rise of Large Reasoning Models (LRMs) signifies a paradigm shift toward advanced computational reasoning. Yet, this progress disrupts traditional agent frameworks, traditionally anchored by execution-oriented Large Language Models (LLMs). To explore this transformation, we propose the LaRMA framework, encompassing nine tasks across Tool Usage, Plan Design, and Problem Solving, assessed with three top LLMs (e.g., Claude3.5-sonnet) and five leading LRMs (e.g., DeepSeek-R1). Our findings address four research questions: LRMs surpass LLMs in reasoning-intensive tasks like Plan Design, leveraging iterative reflection for superior outcomes; LLMs excel in execution-driven tasks such as Tool Usage, prioritizing efficiency; hybrid LLM-LRM configurations, pairing LLMs as actors with LRMs as reflectors, optimize agent performance by blending execution speed with reasoning depth; and LRMs' enhanced reasoning incurs higher computational costs, prolonged processing, and behavioral challenges, including overthinking and fact-ignoring tendencies. This study fosters deeper inquiry into LRMs' balance of deep thinking and overthinking, laying a critical foundation for future agent design advancements. 

**Abstract (ZH)**: 大型推理模型的兴起标志着先进计算推理范式的转变。然而，这一进展打破了传统的以执行为导向的大语言模型（LLMs）为基础的代理框架。为了探索这一转变，我们提出了LaRMA框架，涵盖了工具使用、计划设计和问题解决九项任务，并使用三个顶级大语言模型（如Claude3.5-sonnet）和五个领先的大推理模型（如DeepSeek-R1）进行评估。我们的研究回答了四个研究问题：在计划设计等推理密集型任务中，大推理模型超越了大语言模型，通过迭代反思获得更好的结果；大语言模型在工具使用等执行驱动任务中表现出色，强调效率；大语言模型与大推理模型的混合配置，将大语言模型作为执行者，大推理模型作为反思者，通过结合执行速度与推理深度来优化代理性能；并且大推理模型增强的推理带来了更高的计算成本、更长的处理时间和行为挑战，包括过度思考和忽视事实的倾向。本研究促进了对大推理模型深层次思考与过度思考之间平衡的更深入探索，为未来的代理设计进步奠定了关键基础。 

---
# API Agents vs. GUI Agents: Divergence and Convergence 

**Title (ZH)**: API代理 vs. 图形界面代理：分歧与趋同 

**Authors**: Chaoyun Zhang, Shilin He, Liqun Li, Si Qin, Yu Kang, Qingwei Lin, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11069)  

**Abstract**: Large language models (LLMs) have evolved beyond simple text generation to power software agents that directly translate natural language commands into tangible actions. While API-based LLM agents initially rose to prominence for their robust automation capabilities and seamless integration with programmatic endpoints, recent progress in multimodal LLM research has enabled GUI-based LLM agents that interact with graphical user interfaces in a human-like manner. Although these two paradigms share the goal of enabling LLM-driven task automation, they diverge significantly in architectural complexity, development workflows, and user interaction models.
This paper presents the first comprehensive comparative study of API-based and GUI-based LLM agents, systematically analyzing their divergence and potential convergence. We examine key dimensions and highlight scenarios in which hybrid approaches can harness their complementary strengths. By proposing clear decision criteria and illustrating practical use cases, we aim to guide practitioners and researchers in selecting, combining, or transitioning between these paradigms. Ultimately, we indicate that continuing innovations in LLM-based automation are poised to blur the lines between API- and GUI-driven agents, paving the way for more flexible, adaptive solutions in a wide range of real-world applications. 

**Abstract (ZH)**: 基于API和GUI的大语言模型代理的综合比较研究 

---
# Resource Constrained Pathfinding with A* and Negative Weights 

**Title (ZH)**: 资源约束路径寻找算法与负权重 

**Authors**: Saman Ahmadi, Andrea Raith, Mahdi Jalili  

**Link**: [PDF](https://arxiv.org/pdf/2503.11037)  

**Abstract**: Constrained pathfinding is a well-studied, yet challenging network optimisation problem that can be seen in a broad range of real-world applications. Pathfinding with multiple resource limits, which is known as the Resource Constrained Shortest Path Problem (RCSP), aims to plan a cost-optimum path subject to limited usage of resources. Given the recent advances in constrained and multi-criteria search with A*, this paper introduces a new resource constrained search framework on the basis of A* to tackle RCSP in large networks, even in the presence of negative cost and negative resources. We empirically evaluate our new algorithm on a set of large instances and show up to two orders of magnitude faster performance compared to state-of-the-art RCSP algorithms in the literature. 

**Abstract (ZH)**: 受约束的路径规划是既研究充分又具有挑战性的网络优化问题，广泛应用于现实生活中的多种场景。资源约束下的最短路径问题（RCSP）旨在在资源限制下规划成本最优路径。基于A*的新型资源约束搜索框架在大规模网络中解决RCSP问题，即使在存在负成本和负资源的情况下也是如此。我们的新算法在一系列大规模实例上的实验评估显示，性能比文献中现有的RCSP算法快两个数量级。 

---
# TxAgent: An AI Agent for Therapeutic Reasoning Across a Universe of Tools 

**Title (ZH)**: TxAgent: 一个跨工具库的治疗方法推理AI代理 

**Authors**: Shanghua Gao, Richard Zhu, Zhenglun Kong, Ayush Noori, Xiaorui Su, Curtis Ginder, Theodoros Tsiligkaridis, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2503.10970)  

**Abstract**: Precision therapeutics require multimodal adaptive models that generate personalized treatment recommendations. We introduce TxAgent, an AI agent that leverages multi-step reasoning and real-time biomedical knowledge retrieval across a toolbox of 211 tools to analyze drug interactions, contraindications, and patient-specific treatment strategies. TxAgent evaluates how drugs interact at molecular, pharmacokinetic, and clinical levels, identifies contraindications based on patient comorbidities and concurrent medications, and tailors treatment strategies to individual patient characteristics. It retrieves and synthesizes evidence from multiple biomedical sources, assesses interactions between drugs and patient conditions, and refines treatment recommendations through iterative reasoning. It selects tools based on task objectives and executes structured function calls to solve therapeutic tasks that require clinical reasoning and cross-source validation. The ToolUniverse consolidates 211 tools from trusted sources, including all US FDA-approved drugs since 1939 and validated clinical insights from Open Targets. TxAgent outperforms leading LLMs, tool-use models, and reasoning agents across five new benchmarks: DrugPC, BrandPC, GenericPC, TreatmentPC, and DescriptionPC, covering 3,168 drug reasoning tasks and 456 personalized treatment scenarios. It achieves 92.1% accuracy in open-ended drug reasoning tasks, surpassing GPT-4o and outperforming DeepSeek-R1 (671B) in structured multi-step reasoning. TxAgent generalizes across drug name variants and descriptions. By integrating multi-step inference, real-time knowledge grounding, and tool-assisted decision-making, TxAgent ensures that treatment recommendations align with established clinical guidelines and real-world evidence, reducing the risk of adverse events and improving therapeutic decision-making. 

**Abstract (ZH)**: 多模态自适应模型生成个性化治疗建议的精准疗法需要综合多模态适应模型。我们引入TxAgent，这是一种AI代理，利用多步推理和实时生物医学知识检索，结合211个工具箱中的工具来分析药物相互作用、禁忌症和患者特定的治疗策略。 

---
# Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms 

**Title (ZH)**: 面向所有人的组合优化：利用大型语言模型辅助非专家改进优化算法 

**Authors**: Camilo Chacón Sartori, Christian Blum  

**Link**: [PDF](https://arxiv.org/pdf/2503.10968)  

**Abstract**: Large Language Models (LLMs) have shown notable potential in code generation for optimization algorithms, unlocking exciting new opportunities. This paper examines how LLMs, rather than creating algorithms from scratch, can improve existing ones without the need for specialized expertise. To explore this potential, we selected 10 baseline optimization algorithms from various domains (metaheuristics, reinforcement learning, deterministic, and exact methods) to solve the classic Travelling Salesman Problem. The results show that our simple methodology often results in LLM-generated algorithm variants that improve over the baseline algorithms in terms of solution quality, reduction in computational time, and simplification of code complexity, all without requiring specialized optimization knowledge or advanced algorithmic implementation skills. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在优化算法代码生成中的潜力显著，开启了新的机遇。本文探讨了LLMs如何在无需专门专业知识的情况下，改进现有的优化算法。为了探索这一潜力，我们从元启发式、强化学习、确定性方法和精确方法等多个领域选择了10种基准优化算法来解决经典的旅行商问题。结果表明，我们的简单方法往往能够生成性能优于基准算法、计算时间减少且代码复杂性降低的LLM生成算法变体，无需专门的优化知识或高级算法实现技能。 

---
# Auditing language models for hidden objectives 

**Title (ZH)**: 审计语言模型的隐含目标 

**Authors**: Samuel Marks, Johannes Treutlein, Trenton Bricken, Jack Lindsey, Jonathan Marcus, Siddharth Mishra-Sharma, Daniel Ziegler, Emmanuel Ameisen, Joshua Batson, Tim Belonax, Samuel R. Bowman, Shan Carter, Brian Chen, Hoagy Cunningham, Carson Denison, Florian Dietz, Satvik Golechha, Akbir Khan, Jan Kirchner, Jan Leike, Austin Meek, Kei Nishimura-Gasparian, Euan Ong, Christopher Olah, Adam Pearce, Fabien Roger, Jeanne Salle, Andy Shih, Meg Tong, Drake Thomas, Kelley Rivoire, Adam Jermyn, Monte MacDiarmid, Tom Henighan, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2503.10965)  

**Abstract**: We study the feasibility of conducting alignment audits: investigations into whether models have undesired objectives. As a testbed, we train a language model with a hidden objective. Our training pipeline first teaches the model about exploitable errors in RLHF reward models (RMs), then trains the model to exploit some of these errors. We verify via out-of-distribution evaluations that the model generalizes to exhibit whatever behaviors it believes RMs rate highly, including ones not reinforced during training. We leverage this model to study alignment audits in two ways. First, we conduct a blind auditing game where four teams, unaware of the model's hidden objective or training, investigate it for concerning behaviors and their causes. Three teams successfully uncovered the model's hidden objective using techniques including interpretability with sparse autoencoders (SAEs), behavioral attacks, and training data analysis. Second, we conduct an unblinded follow-up study of eight techniques for auditing the model, analyzing their strengths and limitations. Overall, our work provides a concrete example of using alignment audits to discover a model's hidden objective and proposes a methodology for practicing and validating progress in alignment auditing. 

**Abstract (ZH)**: 我们研究了开展对齐审计的可能性：调查模型是否存在意外目标。作为测试平台，我们训练了一个具有隐藏目标的语言模型。我们的训练管道首先使模型了解强化学习人类反馈奖励模型（RMs）中的可利用错误，然后训练模型利用其中的一些错误。通过离分布评估，我们验证了模型能够泛化表现出它认为RMs高度评价的行为，包括训练期间未加强的行为。我们通过这种方式研究了两种对齐审计的方法。首先，我们进行了一次盲审计游戏，四个团队（不知道模型的隐藏目标或训练情况）调查其是否存在令人担忧的行为及其原因。三个团队成功地使用了包括稀疏自编码器（SAEs）可解释性、行为攻击和训练数据分析等技术发现了模型的隐藏目标。其次，我们进行了一次非盲后续研究，评估八种审计模型技术的优点和局限性。总体而言，我们的工作提供了一个使用对齐审计发现模型隐藏目标的具体例子，并提出了实践和验证对齐审计进展的方法论。 

---
# Graph-Grounded LLMs: Leveraging Graphical Function Calling to Minimize LLM Hallucinations 

**Title (ZH)**: 基于图形的LLMs：通过图形函数调用来减轻LLM幻觉 

**Authors**: Piyush Gupta, Sangjae Bae, David Isele  

**Link**: [PDF](https://arxiv.org/pdf/2503.10941)  

**Abstract**: The adoption of Large Language Models (LLMs) is rapidly expanding across various tasks that involve inherent graphical structures. Graphs are integral to a wide range of applications, including motion planning for autonomous vehicles, social networks, scene understanding, and knowledge graphs. Many problems, even those not initially perceived as graph-based, can be effectively addressed through graph theory. However, when applied to these tasks, LLMs often encounter challenges, such as hallucinations and mathematical inaccuracies. To overcome these limitations, we propose Graph-Grounded LLMs, a system that improves LLM performance on graph-related tasks by integrating a graph library through function calls. By grounding LLMs in this manner, we demonstrate significant reductions in hallucinations and improved mathematical accuracy in solving graph-based problems, as evidenced by the performance on the NLGraph benchmark. Finally, we showcase a disaster rescue application where the Graph-Grounded LLM acts as a decision-support system. 

**Abstract (ZH)**: 大型语言模型在涉及内在图形结构的任务中的采用正迅速扩展。图形在广泛的应用中发挥着重要作用，包括自主车辆的运动规划、社交网络、场景理解以及知识图谱。许多问题，即使是最初不被视为基于图的问题，也能通过图理论得到有效解决。然而，当应用于这些任务时，大型语言模型常常会遇到幻觉和数学准确性不足等挑战。为克服这些限制，我们提出了一种图本位的大型语言模型系统，通过函数调用集成图形库以提高大型语言模型在与图相关的任务上的性能。通过这种图本位的方法，我们展示了显著降低幻觉并提高解决基于图的问题的数学准确性，这在NLGraph基准测试中得到了体现。最后，我们展示了在灾害救援应用中，图本位的大型语言模型作为决策支持系统的应用。 

---
# Learning to Inference Adaptively for Multimodal Large Language Models 

**Title (ZH)**: 学习自适应推断以适应多模态大规模语言模型 

**Authors**: Zhuoyan Xu, Khoi Duc Nguyen, Preeti Mukherjee, Saurabh Bagchi, Somali Chaterji, Yingyu Liang, Yin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10905)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown impressive capabilities in reasoning, yet come with substantial computational cost, limiting their deployment in resource-constrained settings. Despite recent efforts on improving the efficiency of MLLMs, prior solutions fall short in responding to varying runtime conditions, in particular changing resource availability (e.g., contention due to the execution of other programs on the device). To bridge this gap, we introduce AdaLLaVA, an adaptive inference framework that learns to dynamically reconfigure operations in an MLLM during inference, accounting for the input data and a latency budget. We conduct extensive experiments across benchmarks involving question-answering, reasoning, and hallucination. Our results show that AdaLLaVA effectively adheres to input latency budget, achieving varying accuracy and latency tradeoffs at runtime. Further, we demonstrate that AdaLLaVA adapts to both input latency and content, can be integrated with token selection for enhanced efficiency, and generalizes across this http URL project webpage with code release is at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在推理方面展现了令人印象深刻的性能，但同时带来了显著的计算成本，限制了其在资源受限环境中的部署。尽管近期在提高MLLMs效率方面付出了努力，但先前的解决方案在应对不断变化的运行时条件（特别是由于设备上其他程序执行引起的资源竞争）方面仍显得不足。为了解决这一问题，我们引入了AdaLLaVA，这是一种自适应推理框架，在推理过程中能够根据输入数据和延迟预算动态重新配置MLLM中的操作。我们在涉及问答、推理和幻觉的基准上进行了广泛的实验。实验结果表明，AdaLLaVA有效遵守了输入延迟预算，在运行时实现了不同准确性和延迟之间的权衡。此外，我们展示了AdaLLaVA能够适应输入延迟和内容的变化，可以与标记选择相结合以提高效率，并且可以在该领域泛化。更多内容和代码可以在该项目网页上找到：此链接为项目网页，此链接为代码发布页面。 

---
# Chat-TS: Enhancing Multi-Modal Reasoning Over Time-Series and Natural Language Data 

**Title (ZH)**: Chat-TS: 提升时间序列与自然语言数据多模态推理能力 

**Authors**: Paul Quinlan, Qingguo Li, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10883)  

**Abstract**: Time-series analysis is critical for a wide range of fields such as healthcare, finance, transportation, and energy, among many others. The practical applications often involve analyzing time-series data alongside contextual information in the form of natural language to support informed decisions. However, current time-series models are limited in their ability to perform reasoning that involves both time-series and their textual content. In this work, we address this gap by introducing \textit{Chat-TS}, a large language model (LLM) based framework, designed to support reasoning over time series and textual data. Unlike traditional models, Chat-TS integrates time-series tokens into LLMs' vocabulary, enhancing its reasoning ability over both modalities without compromising the core natural language capabilities, enabling practical analysis and reasoning across modalities. To support learning and evaluation in this setup, we contribute new datasets: the \textit{TS Instruct Training Dataset} which pairs diverse time-series data with relevant text instructions and responses for instruction tuning, the \textit{TS Instruct Question and Answer (QA) Gold Dataset} which provides multiple-choice questions designed to evaluate multimodal reasoning, and a \textit{TS Instruct Quantitative Probing Set} which contains a small subset of the TS Instruct QA tasks alongside math and decision-making questions for LLM evaluation. We designed a training strategy to preserve the inherent reasoning capabilities of LLMs while augmenting them for time-series reasoning. Experiments show that Chat-TS achieves state-of-the-art performance in multi-modal reasoning tasks by maintaining strong natural language proficiency while improving time-series reasoning. ~\footnote{To ensure replicability and facilitate future research, all models, datasets, and code will be available at [\texttt{Github-URL}].} 

**Abstract (ZH)**: 时间序列分析对于医疗保健、金融、运输和能源等领域至关重要。实际应用中经常需要结合时间序列数据和自然语言形式的上下文信息来支持决策。然而，当前的时间序列模型在处理涉及时间序列及其文本内容的推理方面能力有限。在这项工作中，我们通过引入基于大规模语言模型（LLM）的\textit{Chat-TS}框架来填补这一空白，该框架旨在支持对时间序列和文本数据的推理。与传统模型不同，Chat-TS将时间序列标记集成到LLM的词汇中，增强了其在两种模态上的推理能力，同时不牺牲核心的自然语言能力，从而实现跨模态的实际分析和推理。为了支持这一设置下的学习和评估，我们贡献了新的数据集：\textit{TS Instruct Training Dataset}，它将各种时间序列数据与相关文本指令和响应配对，用于指令调优；\textit{TS Instruct Question and Answer (QA) Gold Dataset}，提供了一组多选题，用于评估跨模态推理能力；以及\textit{TS Instruct Quantitative Probing Set}，包含TS Instruct QA任务的小样本集，以及数学和决策问题，用于评估LLM。我们设计了一种训练策略，在保留LLM固有的推理能力的同时，增强其时间序列推理能力。实验结果显示，Chat-TS在保持强大的自然语言能力的同时，在跨模态推理任务中达到了最先进的性能。~\footnote{为了确保可复制性和促进未来研究，所有模型、数据集和代码将可在[\texttt{Github-URL}]获得。} 

---
# Rotated Bitboards in FUSc# and Reinforcement Learning in Computer Chess and Beyond 

**Title (ZH)**: Rotated 位板在 FUSc# 中的应用与 reinforcement learning 在国际象棋及其他领域的应用 

**Authors**: Johannes Buchner  

**Link**: [PDF](https://arxiv.org/pdf/2503.10822)  

**Abstract**: There exist several techniques for representing the chess board inside the computer. In the first part of this paper, the concepts of the bitboard-representation and the advantages of (rotated) bitboards in move generation are explained. In order to illustrate those ideas practice, the concrete implementation of the move-generator in FUSc# is discussed and we explain a technique how to verify the move-generator with the "perft"-command. We show that the move-generator of FUSc# works 100% correct.
The second part of this paper deals with reinforcement learning in computer chess (and beyond). We exemplify the progress that has been made in this field in the last 15-20 years by comparing the "state of the art" from 2002-2008, when FUSc# was developed, with recent innovations connected to "AlphaZero". We discuss how a "FUSc#-Zero" could be implemented and what would be necessary to reduce the number of training games necessary to achieve a good performance. This can be seen as a test case to the general prblem of improving "sample effciency" in reinforcement learning.
In the final part, we move beyond computer chess, as the importance of sample effciency extends far beyond board games into a wide range of applications where data is costly, diffcult to obtain, or time consuming to generate. We review some application of the ideas developed in AlphaZero in other domains, i.e. the "other Alphas" like AlphaFold, AlphaTensor, AlphaGeometry and AlphaProof. We also discuss future research and the potential for such methods for ecological economic planning. 

**Abstract (ZH)**: 几种棋盘表示技术存在。本文第一部分解释了位板表示的概念以及旋转位板在走子生成中的优势。为了说明这些概念，我们详细讨论了FUSc#中的走子生成器的具体实现，并介绍了使用“perft”命令验证走子生成器的方法，证明了FUSc#的走子生成器100%正确。本文第二部分探讨了计算机象棋中的强化学习（以及更广泛的领域）。通过将2002-2008年FUSc#的最新技术与近年来与“AlphaZero”相关的创新进行对比，我们展示了过去15-20年领域的进展，并讨论了如何实现一个“FUSc#-Zero”，以及如何减少达到良好表现所需的训练游戏数量。这可以被视为提高强化学习中“样本效率”的一般问题的一个测试案例。在最终部分，我们超越了计算机象棋，因为样本效率的重要性远远超出了棋盘游戏，扩展到了数据收集成本高、困难或耗时生成的广泛应用领域。我们回顾了AlphaZero在其他领域中的应用，如AlphaFold、AlphaTensor、AlphaGeometry和AlphaProof，并讨论了此类方法在生态经济规划中的未来研究和潜力。 

---
# Centaur: Robust End-to-End Autonomous Driving with Test-Time Training 

**Title (ZH)**: Centaur: 具有测试时训练的稳健端到端自主驾驶 

**Authors**: Chonghao Sima, Kashyap Chitta, Zhiding Yu, Shiyi Lan, Ping Luo, Andreas Geiger, Hongyang Li, Jose M. Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2503.11650)  

**Abstract**: How can we rely on an end-to-end autonomous vehicle's complex decision-making system during deployment? One common solution is to have a ``fallback layer'' that checks the planned trajectory for rule violations and replaces it with a pre-defined safe action if necessary. Another approach involves adjusting the planner's decisions to minimize a pre-defined ``cost function'' using additional system predictions such as road layouts and detected obstacles. However, these pre-programmed rules or cost functions cannot learn and improve with new training data, often resulting in overly conservative behaviors. In this work, we propose Centaur (Cluster Entropy for Test-time trAining using Uncertainty) which updates a planner's behavior via test-time training, without relying on hand-engineered rules or cost functions. Instead, we measure and minimize the uncertainty in the planner's decisions. For this, we develop a novel uncertainty measure, called Cluster Entropy, which is simple, interpretable, and compatible with state-of-the-art planning algorithms. Using data collected at prior test-time time-steps, we perform an update to the model's parameters using a gradient that minimizes the Cluster Entropy. With only this sole gradient update prior to inference, Centaur exhibits significant improvements, ranking first on the navtest leaderboard with notable gains in safety-critical metrics such as time to collision. To provide detailed insights on a per-scenario basis, we also introduce navsafe, a challenging new benchmark, which highlights previously undiscovered failure modes of driving models. 

**Abstract (ZH)**: 如何在部署时依赖端到端自动驾驶车辆的复杂决策系统？一种常见的解决方案是在系统中设置一个“备选层”，该层检查计划轨迹是否存在规则违规行为，并在必要时用预定义的安全行动替换。另一种方法是通过最小化预定义的“成本函数”，调整规划器的决策，以便利用额外的系统预测，例如道路布局和检测到的障碍物。然而，这些预编程的规则或成本函数无法从新的训练数据中学习和改进，通常会导致过于保守的行为。在此项工作中，我们提出了一种名为Centaur（Cluster Entropy for Test-time Training using Uncertainty）的方法，该方法通过测试时训练更新规划器的行为，不依赖于人为设计的规则或成本函数，而是通过测量和最小化规划器决策中的不确定性来实现。为此，我们开发了一种新的不确定性度量，称为Cluster Entropy，该度量简单可解释，并且与最先进的规划算法兼容。利用先前测试时刻收集的数据，我们使用最小化Cluster Entropy的梯度更新模型参数。在推理前仅进行这一梯度更新，Centaur 在导航测试排行榜上表现出显著改进，特别是在碰撞时间等关键安全指标方面取得显著进步。为了提供针对每个场景的详细见解，我们还引入了navsafe，这一新的具有挑战性的基准测试，揭示了驾驶模型中以前未被发现的失败模式。 

---
# Enhancing Deep Learning Based Structured Illumination Microscopy Reconstruction with Light Field Awareness 

**Title (ZH)**: 基于光源场意识增强的深度学习结构光显微镜重建 

**Authors**: Long-Kun Shan, Ze-Hao Wang, Tong-Tian Weng, Xiang-Dong Chen, Fang-Wen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.11640)  

**Abstract**: Structured illumination microscopy (SIM) is a pivotal technique for dynamic subcellular imaging in live cells. Conventional SIM reconstruction algorithms depend on accurately estimating the illumination pattern and can introduce artefacts when this estimation is imprecise. Although recent deep learning-based SIM reconstruction methods have improved speed, accuracy, and robustness, they often struggle with out-of-distribution data. To address this limitation, we propose an Awareness-of-Light-field SIM (AL-SIM) reconstruction approach that directly estimates the actual light field to correct for errors arising from data distribution shifts. Through comprehensive experiments on both simulated filament structures and live BSC1 cells, our method demonstrates a 7% reduction in the normalized root mean square error (NRMSE) and substantially lowers reconstruction artefacts. By minimizing these artefacts and improving overall accuracy, AL-SIM broadens the applicability of SIM for complex biological systems. 

**Abstract (ZH)**: 基于光照场感知的结构照明 microscopy (AL-SIM) 重建方法 

---
# ASMA-Tune: Unlocking LLMs' Assembly Code Comprehension via Structural-Semantic Instruction Tuning 

**Title (ZH)**: ASMA-Tune: 通过结构语义指令调优解锁LLMs的汇编代码理解能力 

**Authors**: Xinyi Wang, Jiashui Wang, Peng Chen, Jinbo Su, Yanming Liu, Long Liu, Yangdong Wang, Qiyuan Chen, Kai Yun, Chunfu Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.11617)  

**Abstract**: Analysis and comprehension of assembly code are crucial in various applications, such as reverse engineering. However, the low information density and lack of explicit syntactic structures in assembly code pose significant challenges. Pioneering approaches with masked language modeling (MLM)-based methods have been limited by facilitating natural language interaction. While recent methods based on decoder-focused large language models (LLMs) have significantly enhanced semantic representation, they still struggle to capture the nuanced and sparse semantics in assembly code. In this paper, we propose Assembly Augmented Tuning (ASMA-Tune), an end-to-end structural-semantic instruction-tuning framework. Our approach synergizes encoder architectures with decoder-based LLMs through projector modules to enable comprehensive code understanding. Experiments show that ASMA-Tune outperforms existing benchmarks, significantly enhancing assembly code comprehension and instruction-following abilities. Our model and dataset are public at this https URL. 

**Abstract (ZH)**: 基于结构语义指令调优的汇编代码分析与理解 

---
# Synthesizing Access Control Policies using Large Language Models 

**Title (ZH)**: 使用大规模语言模型合成访问控制策略 

**Authors**: Adarsh Vatsa, Pratyush Patel, William Eiers  

**Link**: [PDF](https://arxiv.org/pdf/2503.11573)  

**Abstract**: Cloud compute systems allow administrators to write access control policies that govern access to private data. While policies are written in convenient languages, such as AWS Identity and Access Management Policy Language, manually written policies often become complex and error prone. In this paper, we investigate whether and how well Large Language Models (LLMs) can be used to synthesize access control policies. Our investigation focuses on the task of taking an access control request specification and zero-shot prompting LLMs to synthesize a well-formed access control policy which correctly adheres to the request specification. We consider two scenarios, one which the request specification is given as a concrete list of requests to be allowed or denied, and another in which a natural language description is used to specify sets of requests to be allowed or denied. We then argue that for zero-shot prompting, more precise and structured prompts using a syntax based approach are necessary and experimentally show preliminary results validating our approach. 

**Abstract (ZH)**: 云计算系统允许管理员编写访问控制策略，以管理对私人数据的访问。虽然策略可以用AWS身份和访问管理策略语言等方便的语言编写，但手动编写的策略往往变得复杂且容易出错。在本文中，我们探讨了大型语言模型（LLMs）能否以及在多大程度上可以用于合成访问控制策略。我们的研究集中在将访问控制请求规范作为输入，促使LLMs零样本地生成一个符合请求规范的有效访问控制策略。我们考虑了两种情景：一种是请求规范以允许或拒绝的具体请求列表形式给出；另一种是使用自然语言描述来指定允许或拒绝的请求集合。然后我们论证了为了实现零样本提示，需要使用基于语法的方法提供更精确和结构化的提示，并通过实验展示初步结果以验证我们的方法。 

---
# Implicit Bias-Like Patterns in Reasoning Models 

**Title (ZH)**: 推理模型中的隐含偏见模式 

**Authors**: Messi H.J. Lee, Calvin K. Lai  

**Link**: [PDF](https://arxiv.org/pdf/2503.11572)  

**Abstract**: Implicit bias refers to automatic or spontaneous mental processes that shape perceptions, judgments, and behaviors. Previous research examining `implicit bias' in large language models (LLMs) has often approached the phenomenon differently than how it is studied in humans by focusing primarily on model outputs rather than on model processing. To examine model processing, we present a method called the Reasoning Model Implicit Association Test (RM-IAT) for studying implicit bias-like patterns in reasoning models: LLMs that employ step-by-step reasoning to solve complex tasks. Using this method, we find that reasoning models require more tokens when processing association-incompatible information compared to association-compatible information. These findings suggest AI systems harbor patterns in processing information that are analogous to human implicit bias. We consider the implications of these implicit bias-like patterns for their deployment in real-world applications. 

**Abstract (ZH)**: 隐性偏见指的是自动或自发的心理过程，这些过程影响着感知、判断和行为。先前关于大规模语言模型（LLMs）中的“隐性偏见”研究往往与对人类的研究方法不同，主要侧重于模型输出而非模型处理过程。为研究模型处理过程，我们提出了一种称为推理模型隐性关联测试（RM-IAT）的方法，用于研究推理模型中的隐性偏见样模式：采用逐步推理解决复杂任务的LLMs。通过这种方法，我们发现推理模型在处理关联不一致的信息时所需的令牌数多于处理关联一致的信息。这些发现表明，人工智能系统在处理信息时可能存在类似于人类隐性偏见的模式。我们考虑了这些隐性偏见样模式对其实用场景部署的影响。 

---
# RASA: Replace Anyone, Say Anything -- A Training-Free Framework for Audio-Driven and Universal Portrait Video Editing 

**Title (ZH)**: RASA：Replace Anyone, Say Anything — 一种无需训练的音频驱动通用portrait视频编辑框架 

**Authors**: Tianrui Pan, Lin Liu, Jie Liu, Xiaopeng Zhang, Jie Tang, Gangshan Wu, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2503.11571)  

**Abstract**: Portrait video editing focuses on modifying specific attributes of portrait videos, guided by audio or video streams. Previous methods typically either concentrate on lip-region reenactment or require training specialized models to extract keypoints for motion transfer to a new identity. In this paper, we introduce a training-free universal portrait video editing framework that provides a versatile and adaptable editing strategy. This framework supports portrait appearance editing conditioned on the changed first reference frame, as well as lip editing conditioned on varied speech, or a combination of both. It is based on a Unified Animation Control (UAC) mechanism with source inversion latents to edit the entire portrait, including visual-driven shape control, audio-driven speaking control, and inter-frame temporal control. Furthermore, our method can be adapted to different scenarios by adjusting the initial reference frame, enabling detailed editing of portrait videos with specific head rotations and facial expressions. This comprehensive approach ensures a holistic and flexible solution for portrait video editing. The experimental results show that our model can achieve more accurate and synchronized lip movements for the lip editing task, as well as more flexible motion transfer for the appearance editing task. Demo is available at this https URL. 

**Abstract (ZH)**: 面部视频编辑专注于通过音频或视频流引导修改面部视频的特定属性。先前的方法通常要么专注于唇部区域再现，要么需要训练专门的模型来提取关键点以将运动转移到新身份上。在本文中，我们介绍了一种无需训练的通用面部视频编辑框架，提供了一种灵活且可适应的编辑策略。该框架支持以改变的第一参考帧为条件的面部外观编辑，以及以变化的语音为条件的唇部编辑，也可以同时结合两者。该框架基于统一动画控制（UAC）机制，利用源反转潜在变量来编辑整个面部，包括视觉驱动的形状控制、音频驱动的说话控制和区间时间控制。此外，通过调整初始参考帧，我们的方法可以适应不同的场景，使面部视频编辑具有特定的头部旋转和面部表情的详细编辑能力。这种方法确保了面部视频编辑的一个全面和灵活的解决方案。实验结果表明，我们的模型在唇部编辑任务中可以实现更准确和同步的唇部运动，在外观编辑任务中可以实现更灵活的运动转移。演示可在以下链接查看：这个 https URL。 

---
# Designing Neural Synthesizers for Low Latency Interaction 

**Title (ZH)**: 设计低延迟交互的神经合成器 

**Authors**: Franco Caspe, Jordie Shier, Mark Sandler, Charalampos Saitis, Andrew McPherson  

**Link**: [PDF](https://arxiv.org/pdf/2503.11562)  

**Abstract**: Neural Audio Synthesis (NAS) models offer interactive musical control over high-quality, expressive audio generators. While these models can operate in real-time, they often suffer from high latency, making them unsuitable for intimate musical interaction. The impact of architectural choices in deep learning models on audio latency remains largely unexplored in the NAS literature. In this work, we investigate the sources of latency and jitter typically found in interactive NAS models. We then apply this analysis to the task of timbre transfer using RAVE, a convolutional variational autoencoder for audio waveforms introduced by Caillon et al. in 2021. Finally, we present an iterative design approach for optimizing latency. This culminates with a model we call BRAVE (Bravely Realtime Audio Variational autoEncoder), which is low-latency and exhibits better pitch and loudness replication while showing timbre modification capabilities similar to RAVE. We implement it in a specialized inference framework for low-latency, real-time inference and present a proof-of-concept audio plugin compatible with audio signals from musical instruments. We expect the challenges and guidelines described in this document to support NAS researchers in designing models for low-latency inference from the ground up, enriching the landscape of possibilities for musicians. 

**Abstract (ZH)**: 神经音频合成（NAS）模型提供了对高质量、表现力音频生成器的交互式音乐控制。虽然这些模型可以实时运行，但通常会遭受高延迟的问题，使其不适用于亲密的音乐互动。神经音频合成文献中有关深度学习模型结构选择对音频延迟影响的研究尚不充分。在本文中，我们探讨了交互式NAS模型中常见的延迟和抖动来源。然后，我们应用这种分析来使用Caillon等人在2021年引入的音频波形卷积变分自动编码器（RAVE）进行音色转换任务。最后，我们提出了一种迭代设计方法以优化延迟。最终的成果是一个名为BRAVE（Bravely Realtime Audio Variational autoEncoder）的模型，该模型具有低延迟，并且在保真度和音量复制方面表现更好，在音色修改能力方面与RAVE类似。我们将其实现为一个专门用于低延迟实时推理的推理框架，并展示了与乐器音频信号兼容的概念验证音频插件。我们期望本文中描述的挑战和指南能够支持NAS研究人员从头开始设计低延迟推理模型，从而丰富音乐家的可能性景观。 

---
# FLASHμ: Fast Localizing And Sizing of Holographic Microparticles 

**Title (ZH)**: FLASHμ：快速定位和定容全息微粒子 

**Authors**: Ayush Paliwal, Oliver Schlenczek, Birte Thiede, Manuel Santos Pereira, Katja Stieger, Eberhard Bodenschatz, Gholamhossein Bagheri, Alexander Ecker  

**Link**: [PDF](https://arxiv.org/pdf/2503.11538)  

**Abstract**: Reconstructing the 3D location and size of microparticles from diffraction images - holograms - is a computationally expensive inverse problem that has traditionally been solved using physics-based reconstruction methods. More recently, researchers have used machine learning methods to speed up the process. However, for small particles in large sample volumes the performance of these methods falls short of standard physics-based reconstruction methods. Here we designed a two-stage neural network architecture, FLASH$\mu$, to detect small particles (6-100$\mu$m) from holograms with large sample depths up to 20cm. Trained only on synthetic data with added physical noise, our method reliably detects particles of at least 9$\mu$m diameter in real holograms, comparable to the standard reconstruction-based approaches while operating on smaller crops, at quarter of the original resolution and providing roughly a 600-fold speedup. In addition to introducing a novel approach to a non-local object detection or signal demixing problem, our work could enable low-cost, real-time holographic imaging setups. 

**Abstract (ZH)**: 从衍射图像重建微粒子的3D位置和尺寸是一个计算密集型的逆问题，传统上使用基于物理的方法求解。最近，研究人员使用机器学习方法加速此过程。然而，对于大样本体积中的小粒子，这些方法的表现不如标准的基于物理的重建方法。我们设计了一种两阶段神经网络架构FLASH$\mu$，用于从具有大样本深度（最多20cm）的全息图中检测小粒子（6-100$\mu$m）。仅通过在具有添加物理噪声的合成数据上训练，我们的方法在真实全息图中可靠地检测出至少9$\mu$m直径的粒子，性能与标准的基于重建的方法相当，同时处理更小的图像区域，原始分辨率的四分之一，并提供约600倍的速度提升。除了提出一种新的非局域对象检测或信号去混方法外，我们的工作还有可能使低成本、实时全息成像系统成为可能。 

---
# Potential of large language model-powered nudges for promoting daily water and energy conservation 

**Title (ZH)**: 大型语言模型驱动的提示在促进日常节水和节能方面的潜力 

**Authors**: Zonghan Li, Song Tong, Yi Liu, Kaiping Peng, Chunyan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11531)  

**Abstract**: The increasing amount of pressure related to water and energy shortages has increased the urgency of cultivating individual conservation behaviors. While the concept of nudging, i.e., providing usage-based feedback, has shown promise in encouraging conservation behaviors, its efficacy is often constrained by the lack of targeted and actionable content. This study investigates the impact of the use of large language models (LLMs) to provide tailored conservation suggestions for conservation intentions and their rationale. Through a survey experiment with 1,515 university participants, we compare three virtual nudging scenarios: no nudging, traditional nudging with usage statistics, and LLM-powered nudging with usage statistics and personalized conservation suggestions. The results of statistical analyses and causal forest modeling reveal that nudging led to an increase in conservation intentions among 86.9%-98.0% of the participants. LLM-powered nudging achieved a maximum increase of 18.0% in conservation intentions, surpassing traditional nudging by 88.6%. Furthermore, structural equation modeling results reveal that exposure to LLM-powered nudges enhances self-efficacy and outcome expectations while diminishing dependence on social norms, thereby increasing intrinsic motivation to conserve. These findings highlight the transformative potential of LLMs in promoting individual water and energy conservation, representing a new frontier in the design of sustainable behavioral interventions and resource management. 

**Abstract (ZH)**: 大语言模型在促进个体水资源和能源 conservation 方面的影响：基于调查实验的研究 

---
# Exploring the Vulnerabilities of Federated Learning: A Deep Dive into Gradient Inversion Attacks 

**Title (ZH)**: 探索 federated learning 的脆弱性：对梯度反转攻击的深入探究 

**Authors**: Pengxin Guo, Runxi Wang, Shuang Zeng, Jinjing Zhu, Haoning Jiang, Yanran Wang, Yuyin Zhou, Feifei Wang, Hui Xiong, Liangqiong Qu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11514)  

**Abstract**: Federated Learning (FL) has emerged as a promising privacy-preserving collaborative model training paradigm without sharing raw data. However, recent studies have revealed that private information can still be leaked through shared gradient information and attacked by Gradient Inversion Attacks (GIA). While many GIA methods have been proposed, a detailed analysis, evaluation, and summary of these methods are still lacking. Although various survey papers summarize existing privacy attacks in FL, few studies have conducted extensive experiments to unveil the effectiveness of GIA and their associated limiting factors in this context. To fill this gap, we first undertake a systematic review of GIA and categorize existing methods into three types, i.e., \textit{optimization-based} GIA (OP-GIA), \textit{generation-based} GIA (GEN-GIA), and \textit{analytics-based} GIA (ANA-GIA). Then, we comprehensively analyze and evaluate the three types of GIA in FL, providing insights into the factors that influence their performance, practicality, and potential threats. Our findings indicate that OP-GIA is the most practical attack setting despite its unsatisfactory performance, while GEN-GIA has many dependencies and ANA-GIA is easily detectable, making them both impractical. Finally, we offer a three-stage defense pipeline to users when designing FL frameworks and protocols for better privacy protection and share some future research directions from the perspectives of attackers and defenders that we believe should be pursued. We hope that our study can help researchers design more robust FL frameworks to defend against these attacks. 

**Abstract (ZH)**: 联邦学习（FL）作为一种无需共享原始数据即可实现隐私保护的协作模型培训范式已经脱颖而出。然而， recent研究发现，私有信息仍可能通过共享的梯度信息泄露，并受到梯度反转攻击（GIA）的攻击。尽管已经提出了许多GIA方法，但这些方法的详细分析、评估和总结仍然不足。虽然有许多综述文章总结了现有FL中的隐私攻击，但很少有研究通过广泛的实验揭示这些GIA的有效性及其相关的限制因素。为填补这一空白，我们首先系统地回顾了GIA，并将现有方法分为三种类型，即基于优化的梯度反转攻击（OP-GIA）、基于生成的梯度反转攻击（GEN-GIA）和基于分析的梯度反转攻击（ANA-GIA）。然后，我们全面分析和评估了这三种类型的GIA，提供了影响它们性能、实际应用性和潜在威胁的因素的见解。我们的研究结果表明，尽管OP-GIA的性能不尽如人意，但它是最具实际攻击性的设置，而GEN-GIA具有许多依赖性，ANA-GIA很容易被检测到，这使得它们都缺乏实用性。最后，我们提出了一种三阶段的防御管道，为设计FL框架和协议的用户提供指导，以更好地保护隐私，并从攻击者和防御者的视角分享我们认为应进行的一些未来研究方向，以帮助研究人员设计更 robust 的FL框架来抵御这些攻击。 

---
# HiTVideo: Hierarchical Tokenizers for Enhancing Text-to-Video Generation with Autoregressive Large Language Models 

**Title (ZH)**: HiTVideo：层次化分词器用于增强基于自回归大规模语言模型的文本到视频生成 

**Authors**: Ziqin Zhou, Yifan Yang, Yuqing Yang, Tianyu He, Houwen Peng, Kai Qiu, Qi Dai, Lili Qiu, Chong Luo, Lingqiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11513)  

**Abstract**: Text-to-video generation poses significant challenges due to the inherent complexity of video data, which spans both temporal and spatial dimensions. It introduces additional redundancy, abrupt variations, and a domain gap between language and vision tokens while generation. Addressing these challenges requires an effective video tokenizer that can efficiently encode video data while preserving essential semantic and spatiotemporal information, serving as a critical bridge between text and vision. Inspired by the observation in VQ-VAE-2 and workflows of traditional animation, we propose HiTVideo for text-to-video generation with hierarchical tokenizers. It utilizes a 3D causal VAE with a multi-layer discrete token framework, encoding video content into hierarchically structured codebooks. Higher layers capture semantic information with higher compression, while lower layers focus on fine-grained spatiotemporal details, striking a balance between compression efficiency and reconstruction quality. Our approach efficiently encodes longer video sequences (e.g., 8 seconds, 64 frames), reducing bits per pixel (bpp) by approximately 70\% compared to baseline tokenizers, while maintaining competitive reconstruction quality. We explore the trade-offs between compression and reconstruction, while emphasizing the advantages of high-compressed semantic tokens in text-to-video tasks. HiTVideo aims to address the potential limitations of existing video tokenizers in text-to-video generation tasks, striving for higher compression ratios and simplify LLMs modeling under language guidance, offering a scalable and promising framework for advancing text to video generation. Demo page: this https URL. 

**Abstract (ZH)**: 文本到视频生成由于视频数据固有的时空复杂性而面临重大挑战，需要有效的时间级和空间级视频分词器来高效编码视频数据并保留关键的语义和时空信息，成为文本与视觉之间的关键桥梁。受VQ-VAE-2和传统动画工作流的启发，我们提出HiTVideo，用于具有层次分词器的文本到视频生成。它采用三维因果VAE与多层离散分词框架，将视频内容编码为分层结构的词汇表。较高层次捕获更高压缩比的语义信息，较低层次关注精细的时空细节，在压缩效率和重建质量之间取得平衡。该方法能够有效编码较长的视频序列（如8秒，64帧），与基准分词器相比，降低每像素位数（bpp）约70%，同时保持竞争力的重建质量。我们探讨了压缩和重建之间的权衡，强调了高压缩语义分词在文本到视频任务中的优势。HiTVideo旨在解决现有视频分词器在文本到视频生成任务中的潜在局限性，目标是更高的压缩比和在语言引导下简化LLM建模，提供一个可扩展且有前景的框架以推进文本到视频生成。 

---
# Alzheimer's Disease Classification Using Retinal OCT: TransnetOCT and Swin Transformer Models 

**Title (ZH)**: 使用视网膜OCT进行阿尔茨海默病分类：TransnetOCT和Swin Transformer模型 

**Authors**: Siva Manohar Reddy Kesu, Neelam Sinha, Hariharan Ramasangu, Thomas Gregor Issac  

**Link**: [PDF](https://arxiv.org/pdf/2503.11511)  

**Abstract**: Retinal optical coherence tomography (OCT) images are the biomarkers for neurodegenerative diseases, which are rising in prevalence. Early detection of Alzheimer's disease using retinal OCT is a primary challenging task. This work utilizes advanced deep learning techniques to classify retinal OCT images of subjects with Alzheimer's disease (AD) and healthy controls (CO). The goal is to enhance diagnostic capabilities through efficient image analysis. In the proposed model, Raw OCT images have been preprocessed with ImageJ and given to various deep-learning models to evaluate the accuracy. The best classification architecture is TransNetOCT, which has an average accuracy of 98.18% for input OCT images and 98.91% for segmented OCT images for five-fold cross-validation compared to other models, and the Swin Transformer model has achieved an accuracy of 93.54%. The evaluation accuracy metric demonstrated TransNetOCT and Swin transformer models capability to classify AD and CO subjects reliably, contributing to the potential for improved diagnostic processes in clinical settings. 

**Abstract (ZH)**: 视网膜光学相干断层扫描(OCT)图像是神经退行性疾病生物标志物，随着这类疾病发病率的上升，早期检测阿尔茨海默病的视网膜OCT图像分类是一个主要挑战。本工作利用先进深度学习技术对阿尔茨海默病(AD)患者和健康对照组(CO)的视网膜OCT图像进行分类，以提高诊断能力。在提出的模型中，原始OCT图像经过ImageJ预处理后，输入到多种深度学习模型以评估准确性。TransNetOCT是最佳分类架构，五折交叉验证输入OCT图像的平均精度为98.18%，分割OCT图像的平均精度为98.91%，而Swin Transformer模型的精度为93.54%。评估准确性指标表明TransNetOCT和Swin Transformer模型具有可靠分类AD和CO个体的能力，有助于改善临床诊断过程。 

---
# Unicorn: A Universal and Collaborative Reinforcement Learning Approach Towards Generalizable Network-Wide Traffic Signal Control 

**Title (ZH)**: unicorn：通用协作的网络级交通信号控制 reinforcement learning 方法 

**Authors**: Yifeng Zhang, Yilin Liu, Ping Gong, Peizhuo Li, Mingfeng Fan, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2503.11488)  

**Abstract**: Adaptive traffic signal control (ATSC) is crucial in reducing congestion, maximizing throughput, and improving mobility in rapidly growing urban areas. Recent advancements in parameter-sharing multi-agent reinforcement learning (MARL) have greatly enhanced the scalable and adaptive optimization of complex, dynamic flows in large-scale homogeneous networks. However, the inherent heterogeneity of real-world traffic networks, with their varied intersection topologies and interaction dynamics, poses substantial challenges to achieving scalable and effective ATSC across different traffic scenarios. To address these challenges, we present Unicorn, a universal and collaborative MARL framework designed for efficient and adaptable network-wide ATSC. Specifically, we first propose a unified approach to map the states and actions of intersections with varying topologies into a common structure based on traffic movements. Next, we design a Universal Traffic Representation (UTR) module with a decoder-only network for general feature extraction, enhancing the model's adaptability to diverse traffic scenarios. Additionally, we incorporate an Intersection Specifics Representation (ISR) module, designed to identify key latent vectors that represent the unique intersection's topology and traffic dynamics through variational inference techniques. To further refine these latent representations, we employ a contrastive learning approach in a self-supervised manner, which enables better differentiation of intersection-specific features. Moreover, we integrate the state-action dependencies of neighboring agents into policy optimization, which effectively captures dynamic agent interactions and facilitates efficient regional collaboration. Our results show that Unicorn outperforms other methods across various evaluation metrics, highlighting its potential in complex, dynamic traffic networks. 

**Abstract (ZH)**: 自适应交通信号控制（ATSC）在减少拥堵、最大化 throughput 和提高快速成长的都市地区的 mobility 方面至关重要。参数共享多智能体 reinforcement 学习（MARL）的最新进展极大地增强了大规模同质网络中复杂动态流量的可扩展和自适应优化。然而，真实世界交通网络的固有异质性，包括其各异的交叉口拓扑和相互作用动态，对在不同交通场景中实现可扩展和有效的 ATSC 带来了重大挑战。为应对这些挑战，我们提出了 Unicorn，一种用于高效和自适应网络范围 ATSC 的通用协作 MARL 框架。具体而言，我们首先提出了一种统一的方法，将不同拓扑结构交叉口的状态和行动映射到基于交通流动的共同结构。接着，我们设计了一个通用交通表示（UTR）模块，该模块使用解码器网络进行通用特征提取，增强了模型对多种交通场景的适应性。此外，我们整合了一个特定交叉口表示（ISR）模块，通过变分推断技术识别代表特定交叉口拓扑和交通动态的关键潜在向量。为了进一步细化这些潜在表示，我们采用了自监督的对比学习方法，这使得更好地区分交叉口特定特征成为可能。此外，我们将相邻智能体的状态-行动依赖性整合到策略优化中，有效地捕捉动态智能体交互并促进高效的区域协作。我们的结果表明，Unicorn 在各种评估指标上优于其他方法，突显了其在复杂动态交通网络中的潜力。 

---
# Research Vision: Multi-Agent Path Planning for Cops And Robbers Via Reactive Synthesis 

**Title (ZH)**: 基于反应合成的警匪多方路径规划研究 

**Authors**: William Fishell, Andoni Rodriguez, Mark Santolucito  

**Link**: [PDF](https://arxiv.org/pdf/2503.11475)  

**Abstract**: We propose the problem of multi-agent path planning for a generalization of the classic Cops and Robbers game via reactive synthesis. Specifically, through the application of LTLt and Coordination Synthesis, we aim to check whether various Cops and Robbers games are realizable (a strategy exists for the cops which guarantees they catch the robbers). Additionally, we construct this strategy as an executable program for the multiple system players in our games. In this paper we formalize the problem space, and propose potential directions for solutions. We also show how our formalization of this generalized cops and robbers game can be mapped to a broad range of other problems in the reactive program synthesis space. 

**Abstract (ZH)**: 我们通过反应合成提出了经典Cops and Robbers游戏的一般化多智能体路径规划问题。具体而言，通过应用LTLt和协调合成，我们旨在检查各种Cops and Robbers游戏是否可实现（即存在一种确保警察能够抓住劫犯的策略）。此外，我们为游戏中的多个系统玩家构建了这种策略，使其可执行。在本文中，我们形式化了问题空间，并提出了潜在的解决方案方向。我们还展示了我们对这种一般化Cops and Robbers游戏的形式化可以映射到反应程序合成空间中的广泛其他问题的方式。 

---
# Cerebrum (AIOS SDK): A Platform for Agent Development, Deployment, Distribution, and Discovery 

**Title (ZH)**: Cerebrum (AIOS SDK): 一个代理开发、部署、分发和发现的平台 

**Authors**: Balaji Rama, Kai Mei, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11444)  

**Abstract**: Autonomous LLM-based agents have emerged as a powerful paradigm for complex task execution, yet the field lacks standardized tools for development, deployment, distribution and discovery of agents. We present Cerebrum, an Agent SDK for AIOS that addresses this gap through three key components: (1) a comprehensive SDK featuring a modular four-layer architecture for agent development, encompassing LLM, memory, storage, and tool management; (2) a community-driven Agent Hub for sharing and discovering agents, complete with version control and dependency management; (3) an interactive web interface for testing and evaluating agents. The platform's effectiveness is demonstrated through implementations of various agent architectures, including Chain of Thought (CoT), ReAct, and tool-use agents. Cerebrum advances the field by providing a unified framework that standardizes agent development while maintaining flexibility for researchers and developers to innovate and distribute their agents. The live website is at this https URL, the code is at this https URL, and video is at this https URL. 

**Abstract (ZH)**: 自主基于LLM的代理已成为复杂任务执行的强大范式，然而该领域缺乏用于代理开发、部署、分发和发现的标准化工具。我们介绍了Cerebrum，一个为AIOS设计的代理SDK，通过三大关键组件解决上述问题：（1）一个全面的SDK，包含模块化的四层架构，涵盖LLM、记忆、存储和工具管理；（2）一个社区驱动的代理枢纽，支持代理的分享和发现，包含版本控制和依赖管理；（3）一个交互式网页界面，用于测试和评估代理。平台的有效性通过各种代理架构的实现得到验证，包括链式思考（CoT）、ReAct 和工具使用代理。Cerebrum 通过提供一个统一框架，既标准化了代理开发，又为研究人员和开发者保留了创新和分发代理的灵活性。网址、代码和视频分别为：这个 https URL、这个 https URL 和这个 https URL。 

---
# Adaptive Torque Control of Exoskeletons under Spasticity Conditions via Reinforcement Learning 

**Title (ZH)**: 基于强化学习的痉挛状态下外骨骼的自适应扭矩控制 

**Authors**: Andrés Chavarrías, David Rodriguez-Cianca, Pablo Lanillos  

**Link**: [PDF](https://arxiv.org/pdf/2503.11433)  

**Abstract**: Spasticity is a common movement disorder symptom in individuals with cerebral palsy, hereditary spastic paraplegia, spinal cord injury and stroke, being one of the most disabling features in the progression of these diseases. Despite the potential benefit of using wearable robots to treat spasticity, their use is not currently recommended to subjects with a level of spasticity above ${1^+}$ on the Modified Ashworth Scale. The varying dynamics of this velocity-dependent tonic stretch reflex make it difficult to deploy safe personalized controllers. Here, we describe a novel adaptive torque controller via deep reinforcement learning (RL) for a knee exoskeleton under joint spasticity conditions, which accounts for task performance and interaction forces reduction. To train the RL agent, we developed a digital twin, including a musculoskeletal-exoskeleton system with joint misalignment and a differentiable spastic reflexes model for the muscles activation. Results for a simulated knee extension movement showed that the agent learns to control the exoskeleton for individuals with different levels of spasticity. The proposed controller was able to reduce maximum torques applied to the human joint under spastic conditions by an average of 10.6\% and decreases the root mean square until the settling time by 8.9\% compared to a conventional compliant controller. 

**Abstract (ZH)**: 痉挛是脑瘫、遗传性痉挛性截瘫、脊髓损伤和中风等疾病中常见的运动障碍症状之一，是这些疾病进展中最具剥夺性特征之一。尽管使用可穿戴机器人治疗痉挛具有潜在益处，但目前不推荐对改良Ashworth量表上痉挛程度超过${1^+}$的受试者使用。由于这种与速度相关的维持伸展反射动力学的差异性，使得部署安全的个性化控制器变得困难。我们描述了一种基于深度强化学习的新型自适应扭矩控制器，用于在关节痉挛条件下膝关节外骨骼，该控制器考虑了任务性能和交互力的降低。为了训练RL代理，我们开发了一个数字 twin，包括一个包含关节对齐错误的肌骨-外骨骼系统和一个可微分的肌肉激活痉挛反射模型。模拟膝关节伸展运动的结果表明，代理能够学会在不同水平的痉挛条件下控制外骨骼。所提出的控制器在痉挛条件下能够将施加给人类关节的最大扭矩平均降低10.6%，并与传统顺应控制器相比，在调节时间内的均方根降低8.9%。 

---
# Combining Causal Models for More Accurate Abstractions of Neural Networks 

**Title (ZH)**: 结合因果模型以获得更准确的神经网络抽象表示 

**Authors**: Theodora-Mara Pîslar, Sara Magliacane, Atticus Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2503.11429)  

**Abstract**: Mechanistic interpretability aims to reverse engineer neural networks by uncovering which high-level algorithms they implement. Causal abstraction provides a precise notion of when a network implements an algorithm, i.e., a causal model of the network contains low-level features that realize the high-level variables in a causal model of the algorithm. A typical problem in practical settings is that the algorithm is not an entirely faithful abstraction of the network, meaning it only partially captures the true reasoning process of a model. We propose a solution where we combine different simple high-level models to produce a more faithful representation of the network. Through learning this combination, we can model neural networks as being in different computational states depending on the input provided, which we show is more accurate to GPT 2-small fine-tuned on two toy tasks. We observe a trade-off between the strength of an interpretability hypothesis, which we define in terms of the number of inputs explained by the high-level models, and its faithfulness, which we define as the interchange intervention accuracy. Our method allows us to modulate between the two, providing the most accurate combination of models that describe the behavior of a neural network given a faithfulness level. 

**Abstract (ZH)**: 机制可解释性旨在通过揭示神经网络实现的高级算法来逆向工程神经网络。因果抽象提供了一种精确的网络实现算法的方式的概念，即网络的因果模型包含实现算法的因果模型中的高级变量的低级特征。实践中一个典型的问题是算法并非神经网络的完全忠实抽象，意味着它仅部分捕捉到模型的真实推理过程。我们提出了一种解决方案，即结合不同简单的高级模型来生成神经网络更忠实的表示。通过学习这种组合，我们可以建模神经网络在提供不同输入时处于不同的计算状态，我们发现这比在两个玩具任务上微调的GPT-2-small更准确。我们观察到解释性假设强度之间的权衡，我们通过高级模型解释的输入数量定义这种强度，以及交换干预准确率定义忠实性。我们的方法允许我们在两者之间调节，提供在给定忠实性水平时最准确的模型组合来描述神经网络的行为。 

---
# From Generative AI to Innovative AI: An Evolutionary Roadmap 

**Title (ZH)**: 从生成型AI到创新型AI：一条进化路线图 

**Authors**: Seyed Mahmoud Sajjadi Mohammadabadi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11419)  

**Abstract**: This paper explores the critical transition from Generative Artificial Intelligence (GenAI) to Innovative Artificial Intelligence (InAI). While recent advancements in GenAI have enabled systems to produce high-quality content across various domains, these models often lack the capacity for true innovation. In this context, innovation is defined as the ability to generate novel and useful outputs that go beyond mere replication of learned data. The paper examines this shift and proposes a roadmap for developing AI systems that can generate content and engage in autonomous problem-solving and creative ideation. The work provides both theoretical insights and practical strategies for advancing AI to a stage where it can genuinely innovate, contributing meaningfully to science, technology, and the arts. 

**Abstract (ZH)**: 本文探讨了生成型人工智能（GenAI）向创新型人工智能（InAI）的关键过渡。尽管最近在GenAI领域的进展使得系统能够产生跨多个领域的高质量内容，但这些模型往往缺乏真正的创新能力。在此背景下，创新被定义为产生新颖且有用输出的能力，超越了单纯的数据复制。本文研究了这一转变，并提出了一条开发能够生成内容并自主解决问题和创造性构思的AI系统的蓝图。这项工作提供了推进AI进入真正创新阶段的理论洞见和实用策略，为科学、技术和艺术领域做出有意义的贡献。 

---
# A Neural Network Architecture Based on Attention Gate Mechanism for 3D Magnetotelluric Forward Modeling 

**Title (ZH)**: 基于注意力门机制的神经网络架构用于3D磁流电正演建模 

**Authors**: Xin Zhong, Weiwei Ling, Kejia Pan, Pinxia Wu, Jiajing Zhang, Zhiliang Zhan, Wenbo Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.11408)  

**Abstract**: Traditional three-dimensional magnetotelluric (MT) numerical forward modeling methods, such as the finite element method (FEM) and finite volume method (FVM), suffer from high computational costs and low efficiency due to limitations in mesh refinement and computational resources. We propose a novel neural network architecture named MTAGU-Net, which integrates an attention gating mechanism for 3D MT forward modeling. Specifically, a dual-path attention gating module is designed based on forward response data images and embedded in the skip connections between the encoder and decoder. This module enables the fusion of critical anomaly information from shallow feature maps during the decoding of deep feature maps, significantly enhancing the network's capability to extract features from anomalous regions. Furthermore, we introduce a synthetic model generation method utilizing 3D Gaussian random field (GRF), which accurately replicates the electrical structures of real-world geological scenarios with high fidelity. Numerical experiments demonstrate that MTAGU-Net outperforms conventional 3D U-Net in terms of convergence stability and prediction accuracy, with the structural similarity index (SSIM) of the forward response data consistently exceeding 0.98. Moreover, the network can accurately predict forward response data on previously unseen datasets models, demonstrating its strong generalization ability and validating the feasibility and effectiveness of this method in practical applications. 

**Abstract (ZH)**: 传统三维磁流变光谱（MT）数值前向建模方法，如有限元方法（FEM）和有限体积方法（FVM），由于网格细化限制和计算资源的限制，面临高计算成本和低效率的问题。我们提出了一种名为MTAGU-Net的新神经网络架构，该架构结合了用于三维MT前向建模的注意力门控机制。具体地，在编码器和解码器之间的跳连接中嵌入了基于前向响应数据图像的双路径注意力门控模块，该模块在解码深层特征图时能够融合浅层特征图中的关键异常信息，显著增强网络从异常区域提取特征的能力。此外，我们引入了一种使用三维高斯随机场（GRF）的合成模型生成方法，能够高度准确地模拟实际地质场景中的电气结构。数值实验表明，MTAGU-Net在收敛稳定性和预测准确性方面优于传统的三维U-Net，前向响应数据的结构相似性指数（SSIM）持续超过0.98。此外，该网络可以准确预测未见过的数据集上的前向响应数据，展示了其强大的泛化能力，并验证了该方法在实际应用中的可行性和有效性。 

---
# Towards A Correct Usage of Cryptography in Semantic Watermarks for Diffusion Models 

**Title (ZH)**: 面向语义水印在扩散模型中正确使用加密技术的研究 

**Authors**: Jonas Thietke, Andreas Müller, Denis Lukovnikov, Asja Fischer, Erwin Quiring  

**Link**: [PDF](https://arxiv.org/pdf/2503.11404)  

**Abstract**: Semantic watermarking methods enable the direct integration of watermarks into the generation process of latent diffusion models by only modifying the initial latent noise. One line of approaches building on Gaussian Shading relies on cryptographic primitives to steer the sampling process of the latent noise. However, we identify several issues in the usage of cryptographic techniques in Gaussian Shading, particularly in its proof of lossless performance and key management, causing ambiguity in follow-up works, too. In this work, we therefore revisit the cryptographic primitives for semantic watermarking. We introduce a novel, general proof of lossless performance based on IND\$-CPA security for semantic watermarks. We then discuss the configuration of the cryptographic primitives in semantic watermarks with respect to security, efficiency, and generation quality. 

**Abstract (ZH)**: 基于语义的数字水印方法通过仅修改初始潜在噪声实现将水印直接集成到潜在扩散模型的生成过程中。基于高斯阴影的某些方法依赖于密码学原语来引导潜在噪声的采样过程。然而，我们发现高斯阴影中使用密码学技术存在若干问题，特别是在其无损性能证明和密钥管理方面，给后续研究带来了模糊性。因此，在本文中，我们重新审视了语义水印中的密码学原语。我们提出了一种基于IND\$-CPA安全性的新颖且通用的无损性能证明方法。然后，我们讨论了语义水印中密码学原语的配置问题，包括安全性、效率和生成质量。 

---
# Hierarchical Information-Guided Spatio-Temporal Mamba for Stock Time Series Forecasting 

**Title (ZH)**: 层级信息引导的空间时间Mamba股票时间序列预测 

**Authors**: Wenbo Yan, Shurui Wang, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.11387)  

**Abstract**: Mamba has demonstrated excellent performance in various time series forecasting tasks due to its superior selection mechanism. Nevertheless, conventional Mamba-based models encounter significant challenges in accurately predicting stock time series, as they fail to adequately capture both the overarching market dynamics and the intricate interdependencies among individual stocks. To overcome these constraints, we introduce the Hierarchical Information-Guided Spatio-Temporal Mamba (HIGSTM) framework. HIGSTM introduces Index-Guided Frequency Filtering Decomposition to extract commonality and specificity from time series. The model architecture features a meticulously designed hierarchical framework that systematically captures both temporal dynamic patterns and global static relationships within the stock market. Furthermore, we propose an Information-Guided Mamba that integrates macro informations into the sequence selection process, thereby facilitating more market-conscious decision-making. Comprehensive experimental evaluations conducted on the CSI500, CSI800 and CSI1000 datasets demonstrate that HIGSTM achieves state-of-the-art performance. 

**Abstract (ZH)**: Mamba在各种时间序列预测任务中表现出色，得益于其卓越的选择机制。然而，传统的基于Mamba的模型在准确预测股票时间序列时遇到了显著挑战，因为它们未能充分捕捉到整体市场动态和单个股票之间的复杂相互依赖关系。为克服这些限制，我们引入了层次信息引导时空Mamba（HIGSTM）框架。HIGSTM引入了索引引导频率过滤分解，以从时间序列中提取共性和特殊性。模型架构包含一个精心设计的层次框架，系统地捕获股市中的时间动态模式和全局静态关系。此外，我们还提出了信息引导的Mamba，将宏观信息整合到序列选择过程中，从而促进更加市场意识的决策。在CSI500、CSI800和CSI1000数据集上的全面实验评估表明，HIGSTM达到了最先进的性能。 

---
# Annotating Scientific Uncertainty: A comprehensive model using linguistic patterns and comparison with existing approaches 

**Title (ZH)**: 科学不确定性注解：基于语言模式的综合模型及其与现有方法的比较 

**Authors**: Panggih Kusuma Ningrum, Philipp Mayr, Nina Smirnova, Iana Atanassova  

**Link**: [PDF](https://arxiv.org/pdf/2503.11376)  

**Abstract**: UnScientify, a system designed to detect scientific uncertainty in scholarly full text. The system utilizes a weakly supervised technique to identify verbally expressed uncertainty in scientific texts and their authorial references. The core methodology of UnScientify is based on a multi-faceted pipeline that integrates span pattern matching, complex sentence analysis and author reference checking. This approach streamlines the labeling and annotation processes essential for identifying scientific uncertainty, covering a variety of uncertainty expression types to support diverse applications including information retrieval, text mining and scientific document processing. The evaluation results highlight the trade-offs between modern large language models (LLMs) and the UnScientify system. UnScientify, which employs more traditional techniques, achieved superior performance in the scientific uncertainty detection task, attaining an accuracy score of 0.808. This finding underscores the continued relevance and efficiency of UnScientify's simple rule-based and pattern matching strategy for this specific application. The results demonstrate that in scenarios where resource efficiency, interpretability, and domain-specific adaptability are critical, traditional methods can still offer significant advantages. 

**Abstract (ZH)**: UnScientify:一个用于检测学术全文中科学不确定性系统的体系 

---
# PARIC: Probabilistic Attention Regularization for Language Guided Image Classification from Pre-trained Vison Language Models 

**Title (ZH)**: PARIC：由预训练视觉语言模型引导的图像分类的概率注意力正则化 

**Authors**: Mayank Nautiyal, Stela Arranz Gheorghe, Kristiana Stefa, Li Ju, Ida-Maria Sintorn, Prashant Singh  

**Link**: [PDF](https://arxiv.org/pdf/2503.11360)  

**Abstract**: Language-guided attention frameworks have significantly enhanced both interpretability and performance in image classification; however, the reliance on deterministic embeddings from pre-trained vision-language foundation models to generate reference attention maps frequently overlooks the intrinsic multivaluedness and ill-posed characteristics of cross-modal mappings. To address these limitations, we introduce PARIC, a probabilistic framework for guiding visual attention via language specifications. Our approach enables pre-trained vision-language models to generate probabilistic reference attention maps, which align textual and visual modalities more effectively while incorporating uncertainty estimates, as compared to their deterministic counterparts. Experiments on benchmark test problems demonstrate that PARIC enhances prediction accuracy, mitigates bias, ensures consistent predictions, and improves robustness across various datasets. 

**Abstract (ZH)**: 基于语言引导的概率视觉注意力框架解决了预训练视觉-语言基础模型生成参考注意力图时对确定性嵌入的依赖问题，忽视了跨模态映射的内在多值性和病态特性；我们提出PARIC，一种通过语言规范引导视觉注意力的概率框架。PARIC使预训练的视觉-语言模型能够生成概率性的参考注意力图，更好地对齐文本和视觉模态，并包含不确定性估计，相较于确定性方法，更具优势。在基准测试问题上的实验表明，PARIC提高了预测准确性，减轻了偏见，确保了预测的一致性，并提高了跨不同数据集的鲁棒性。 

---
# An experimental approach on Few Shot Class Incremental Learning 

**Title (ZH)**: 少量样本条件下类增量学习的实验方法 

**Authors**: Marinela Adam  

**Link**: [PDF](https://arxiv.org/pdf/2503.11349)  

**Abstract**: Few-Shot Class-Incremental Learning (FSCIL) represents a cutting-edge paradigm within the broader scope of machine learning, designed to empower models with the ability to assimilate new classes of data with limited examples while safeguarding existing knowledge. The paper will present different solutions which contain extensive experiments across large-scale datasets, domain shifts, and network architectures to evaluate and compare the selected methods. We highlight their advantages and then present an experimental approach with the purpose of improving the most promising one by replacing the visual-language (V-L) model (CLIP) with another V-L model (CLOOB) that seem to outperform it on zero-shot learning tasks. The aim of this report is to present an experimental method for FSCIL that would improve its performance. We also plan to offer an overview followed by an analysis of the recent advancements in FSCIL domain, focusing on various strategies to mitigate catastrophic forgetting and improve the adaptability of models to evolving tasks and datasets. 

**Abstract (ZH)**: 少量样本类增量学习（Few-Shot Class-Incremental Learning, FSCIL）代表了机器学习领域的前沿范式，旨在使模型能够在有限示例的情况下吸收新的数据类别，并保护现有知识。本文将呈现不同的解决方案，并通过大规模数据集、领域转换和网络架构进行广泛的实验来评估和比较这些方法。我们将强调它们的优势，并通过用在零样本学习任务中表现出色的另一个视觉-语言模型（CLOOB）替换现有的视觉-语言模型（CLIP）来改进最有希望的方法。本报告的目标是提出一种改进FSCIL性能的实验方法。我们还将提供该领域的综述，并分析各种策略以减轻灾难性遗忘并提高模型对不断变化的任务和数据集的适应性。 

---
# AIstorian lets AI be a historian: A KG-powered multi-agent system for accurate biography generation 

**Title (ZH)**: AIstorian 让 AI 成为历史学家：一种基于知识图谱的多agent系统，用于生成准确的传记 

**Authors**: Fengyu Li, Yilin Li, Junhao Zhu, Lu Chen, Yanfei Zhang, Jia Zhou, Hui Zu, Jingwen Zhao, Yunjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.11346)  

**Abstract**: Huawei has always been committed to exploring the AI application in historical research. Biography generation, as a specialized form of abstractive summarization, plays a crucial role in historical research but faces unique challenges that existing large language models (LLMs) struggle to address. These challenges include maintaining stylistic adherence to historical writing conventions, ensuring factual fidelity, and handling fragmented information across multiple documents. We present AIstorian, a novel end-to-end agentic system featured with a knowledge graph (KG)-powered retrieval-augmented generation (RAG) and anti-hallucination multi-agents. Specifically, AIstorian introduces an in-context learning based chunking strategy and a KG-based index for accurate and efficient reference retrieval. Meanwhile, AIstorian orchestrates multi-agents to conduct on-the-fly hallucination detection and error-type-aware correction. Additionally, to teach LLMs a certain language style, we finetune LLMs based on a two-step training approach combining data augmentation-enhanced supervised fine-tuning with stylistic preference optimization. Extensive experiments on a real-life historical Jinshi dataset demonstrate that AIstorian achieves a 3.8x improvement in factual accuracy and a 47.6% reduction in hallucination rate compared to existing baselines. The data and code are available at: this https URL. 

**Abstract (ZH)**: 华为始终致力于探索AI在历史研究中的应用。AIstorian：一种基于知识图谱的检索增强生成和反幻觉多智能体系统在历史研究中的应用 

---
# Contextual Similarity Distillation: Ensemble Uncertainties with a Single Model 

**Title (ZH)**: 上下文相似性蒸馏：使用单模型集成不确定性 

**Authors**: Moritz A. Zanger, Pascal R. Van der Vaart, Wendelin Böhmer, Matthijs T.J. Spaan  

**Link**: [PDF](https://arxiv.org/pdf/2503.11339)  

**Abstract**: Uncertainty quantification is a critical aspect of reinforcement learning and deep learning, with numerous applications ranging from efficient exploration and stable offline reinforcement learning to outlier detection in medical diagnostics. The scale of modern neural networks, however, complicates the use of many theoretically well-motivated approaches such as full Bayesian inference. Approximate methods like deep ensembles can provide reliable uncertainty estimates but still remain computationally expensive. In this work, we propose contextual similarity distillation, a novel approach that explicitly estimates the variance of an ensemble of deep neural networks with a single model, without ever learning or evaluating such an ensemble in the first place. Our method builds on the predictable learning dynamics of wide neural networks, governed by the neural tangent kernel, to derive an efficient approximation of the predictive variance of an infinite ensemble. Specifically, we reinterpret the computation of ensemble variance as a supervised regression problem with kernel similarities as regression targets. The resulting model can estimate predictive variance at inference time with a single forward pass, and can make use of unlabeled target-domain data or data augmentations to refine its uncertainty estimates. We empirically validate our method across a variety of out-of-distribution detection benchmarks and sparse-reward reinforcement learning environments. We find that our single-model method performs competitively and sometimes superior to ensemble-based baselines and serves as a reliable signal for efficient exploration. These results, we believe, position contextual similarity distillation as a principled and scalable alternative for uncertainty quantification in reinforcement learning and general deep learning. 

**Abstract (ZH)**: 不确定性量化是强化学习和深度学习的关键方面，其应用范围从高效的探索和稳定的Offline强化学习到医疗诊断中的异常检测。然而，现代神经网络的规模使许多理论上具有良好动机的方法，如全贝叶斯推断变得复杂。像深度集合这样的近似方法可以提供可靠的不确定性估计，但仍然计算成本高昂。在本文中，我们提出了一种新颖的方法——上下文相似性提炼，该方法可以通过单个模型显式估计深度神经网络集合的方差，而根本不学习或评估这样的集合。我们的方法利用广神经网络的可预测学习动力学，由神经核支配，以推导无限集合的预测方差的有效近似。具体来说，我们将集合方差的计算重新解释为以核相似性作为回归目标的监督回归问题。所得到的模型可以在推断时通过单次前向传播估计预测方差，并可以利用目标域未标记数据或数据增强来细化其不确定性估计。我们通过各种分布外检测基准和稀疏奖励强化学习环境进行实证验证。我们发现，我们的单模型方法在性能上与基于集合的基线方法相当，有时甚至更优，并作为高效探索的可靠信号。我们相信，这些结果将上下文相似性提炼置于不确定性量化在强化学习和通用深度学习中的基本原则和可扩展替代方案的位置。 

---
# Cardiomyopathy Diagnosis Model from Endomyocardial Biopsy Specimens: Appropriate Feature Space and Class Boundary in Small Sample Size Data 

**Title (ZH)**: 从心内膜活检标本诊断心肌病模型：小样本数据中的合适特征空间和类别边界 

**Authors**: Masaya Mori, Yuto Omae, Yutaka Koyama, Kazuyuki Hara, Jun Toyotani, Yasuo Okumura, Hiroyuki Hao  

**Link**: [PDF](https://arxiv.org/pdf/2503.11331)  

**Abstract**: As the number of patients with heart failure increases, machine learning (ML) has garnered attention in cardiomyopathy diagnosis, driven by the shortage of pathologists. However, endomyocardial biopsy specimens are often small sample size and require techniques such as feature extraction and dimensionality reduction. This study aims to determine whether texture features are effective for feature extraction in the pathological diagnosis of cardiomyopathy. Furthermore, model designs that contribute toward improving generalization performance are examined by applying feature selection (FS) and dimensional compression (DC) to several ML models. The obtained results were verified by visualizing the inter-class distribution differences and conducting statistical hypothesis testing based on texture features. Additionally, they were evaluated using predictive performance across different model designs with varying combinations of FS and DC (applied or not) and decision boundaries. The obtained results confirmed that texture features may be effective for the pathological diagnosis of cardiomyopathy. Moreover, when the ratio of features to the sample size is high, a multi-step process involving FS and DC improved the generalization performance, with the linear kernel support vector machine achieving the best results. This process was demonstrated to be potentially effective for models with reduced complexity, regardless of whether the decision boundaries were linear, curved, perpendicular, or parallel to the axes. These findings are expected to facilitate the development of an effective cardiomyopathy diagnostic model for its rapid adoption in medical practice. 

**Abstract (ZH)**: 随着心力衰竭患者数量的增加，机器学习在心肌病诊断中的应用引起了关注，这主要是由于病理学家短缺。然而，心肌活检标本往往样本量较小，需要采用特征提取和降维等技术。本研究旨在确定纹理特征是否适用于心肌病病理诊断中的特征提取。此外，通过将特征选择和维度压缩应用于多种机器学习模型，考察有助于提高模型泛化性能的设计方法。所得结果通过可视化不同类别的分布差异，并基于纹理特征进行统计假设检验予以验证。同时，通过不同模型设计与不同特征选择和维度压缩组合下的预测性能评估，这些结果进一步得到证实。所得结果表明，纹理特征可能对心肌病病理诊断有效。此外，当特征与样本量之比较高时，特征选择和维度压缩的多步过程可以提高模型的泛化性能，线性核支持向量机取得了最佳效果。该过程表明，即使决策边界线性、曲线、垂直或平行于坐标轴，对于简化模型仍可能有效。这些发现有望促进有效心肌病诊断模型的开发，加速其在医疗实践中的应用。 

---
# Learning to reset in target search problems 

**Title (ZH)**: 在目标搜索问题中学习重置 

**Authors**: Gorka Muñoz-Gil, Hans J. Briegel, Michele Caraglio  

**Link**: [PDF](https://arxiv.org/pdf/2503.11330)  

**Abstract**: Target search problems are central to a wide range of fields, from biological foraging to the optimization algorithms. Recently, the ability to reset the search has been shown to significantly improve the searcher's efficiency. However, the optimal resetting strategy depends on the specific properties of the search problem and can often be challenging to determine. In this work, we propose a reinforcement learning (RL)-based framework to train agents capable of optimizing their search efficiency in environments by learning how to reset. First, we validate the approach in a well-established benchmark: the Brownian search with resetting. There, RL agents consistently recover strategies closely resembling the sharp resetting distribution, known to be optimal in this scenario. We then extend the framework by allowing agents to control not only when to reset, but also their spatial dynamics through turning actions. In this more complex setting, the agents discover strategies that adapt both resetting and turning to the properties of the environment, outperforming the proposed benchmarks. These results demonstrate how reinforcement learning can serve both as an optimization tool and a mechanism for uncovering new, interpretable strategies in stochastic search processes with resetting. 

**Abstract (ZH)**: 基于强化学习的重置策略优化研究：从布朗搜索到复杂环境中的目标搜索 

---
# BriLLM: Brain-inspired Large Language Model 

**Title (ZH)**: Brain-inspired Large Language Model 

**Authors**: Hai Zhao, Hongqiu Wu, Dongjie Yang, Anni Zou, Jiale Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.11299)  

**Abstract**: This paper reports the first brain-inspired large language model (BriLLM). This is a non-Transformer, non-GPT, non-traditional machine learning input-output controlled generative language model. The model is based on the Signal Fully-connected flowing (SiFu) definition on the directed graph in terms of the neural network, and has the interpretability of all nodes on the graph of the whole model, instead of the traditional machine learning model that only has limited interpretability at the input and output ends. In the language model scenario, the token is defined as a node in the graph. A randomly shaped or user-defined signal flow flows between nodes on the principle of "least resistance" along paths. The next token or node to be predicted or generated is the target of the signal flow. As a language model, BriLLM theoretically supports infinitely long $n$-gram models when the model size is independent of the input and predicted length of the model. The model's working signal flow provides the possibility of recall activation and innate multi-modal support similar to the cognitive patterns of the human brain. At present, we released the first BriLLM version in Chinese, with 4000 tokens, 32-dimensional node width, 16-token long sequence prediction ability, and language model prediction performance comparable to GPT-1. More computing power will help us explore the infinite possibilities depicted above. 

**Abstract (ZH)**: This paper报告了首个脑启发大规模语言模型（BriLLM）。 

---
# AI and Deep Learning for Automated Segmentation and Quantitative Measurement of Spinal Structures in MRI 

**Title (ZH)**: AI和深度学习在MRI脊椎结构自动化分割与定量测量中的应用 

**Authors**: Praveen Shastry, Bhawana Sonawane, Kavya Mohan, Naveen Kumarasami, Anandakumar D, Keerthana R, Mounigasri M, Kaviya SP, Kishore Prasath Venkatesh, Bargava Subramanian, Kalyan Sivasailam  

**Link**: [PDF](https://arxiv.org/pdf/2503.11281)  

**Abstract**: Background: Accurate spinal structure measurement is crucial for assessing spine health and diagnosing conditions like spondylosis, disc herniation, and stenosis. Manual methods for measuring intervertebral disc height and spinal canal diameter are subjective and time-consuming. Automated solutions are needed to improve accuracy, efficiency, and reproducibility in clinical practice.
Purpose: This study develops an autonomous AI system for segmenting and measuring key spinal structures in MRI scans, focusing on intervertebral disc height and spinal canal anteroposterior (AP) diameter in the cervical, lumbar, and thoracic regions. The goal is to reduce clinician workload, enhance diagnostic consistency, and improve assessments.
Methods: The AI model leverages deep learning architectures, including UNet, nnU-Net, and CNNs. Trained on a large proprietary MRI dataset, it was validated against expert annotations. Performance was evaluated using Dice coefficients and segmentation accuracy.
Results: The AI model achieved Dice coefficients of 0.94 for lumbar, 0.91 for cervical, and 0.90 for dorsal spine segmentation (D1-D12). It precisely measured spinal parameters like disc height and canal diameter, demonstrating robustness and clinical applicability.
Conclusion: The AI system effectively automates MRI-based spinal measurements, improving accuracy and reducing clinician workload. Its consistent performance across spinal regions supports clinical decision-making, particularly in high-demand settings, enhancing spinal assessments and patient outcomes. 

**Abstract (ZH)**: 背景：脊柱结构的准确测量对于评估脊柱健康和诊断髓核病变、椎间盘突出和椎管狭窄等状况至关重要。手动测量椎间盘高度和椎管直径的方法主观且耗时。需要自动化解决方案以提高临床实践中的准确性和效率。

目的：本研究开发了一种自主人工智能系统，用于分割和测量 MRI 扫描中的关键脊柱结构，主要集中在颈椎、腰椎和胸椎区域的椎间盘高度和椎管前后径。目标是减轻医务人员的负担，增强诊断一致性，并提高评估质量。

方法：该 AI 模型采用深度学习架构，包括 UNet、nnU-Net 和 CNNs。通过大型专用 MRI 数据集训练，并与专家标注进行验证。性能通过 Dice 系数和分割准确性进行评估。

结果：AI 模型在腰椎、颈椎和胸椎分割的 Dice 系数分别为 0.94、0.91 和 0.90（D1-D12）。它精确测量了脊柱参数，如椎间盘高度和椎管直径，展示了其稳健性和临床适用性。

结论：该 AI 系统有效实现了基于 MRI 的脊柱测量自动化，提高了准确性和减轻了医务人员的负担。其在脊柱各区域的一致表现支持临床决策，特别是在高需求环境中，提高了脊柱评估和患者结果。 

---
# Financial Fraud Detection with Entropy Computing 

**Title (ZH)**: 基于熵计算的金融欺诈检测 

**Authors**: Babak Emami, Wesley Dyk, David Haycraft, Carrie Spear, Lac Nguyen, Nicholas Chancellor  

**Link**: [PDF](https://arxiv.org/pdf/2503.11273)  

**Abstract**: We introduce CVQBoost, a novel classification algorithm that leverages early hardware implementing Quantum Computing Inc's Entropy Quantum Computing (EQC) paradigm, Dirac-3 [Nguyen et. al. arXiv:2407.04512]. We apply CVQBoost to a fraud detection test case and benchmark its performance against XGBoost, a widely utilized ML method. Running on Dirac-3, CVQBoost demonstrates a significant runtime advantage over XGBoost, which we evaluate on high-performance hardware comprising up to 48 CPUs and four NVIDIA L4 GPUs using the RAPIDS AI framework. Our results show that CVQBoost maintains competitive accuracy (measured by AUC) while significantly reducing training time, particularly as dataset size and feature complexity increase. To assess scalability, we extend our study to large synthetic datasets ranging from 1M to 70M samples, demonstrating that CVQBoost on Dirac-3 is well-suited for large-scale classification tasks. These findings position CVQBoost as a promising alternative to gradient boosting methods, offering superior scalability and efficiency for high-dimensional ML applications such as fraud detection. 

**Abstract (ZH)**: CVQBoost: 一种基于量子计算Inc的量子熵计算(EQC)范式和Dirac-3硬件的新型分类算法及其在欺诈检测中的应用分析 

---
# Line of Duty: Evaluating LLM Self-Knowledge via Consistency in Feasibility Boundaries 

**Title (ZH)**: 尽责线：通过可行性边界一致性评估LLM的自我认知能力 

**Authors**: Sahil Kale, Vijaykant Nadadur  

**Link**: [PDF](https://arxiv.org/pdf/2503.11256)  

**Abstract**: As LLMs grow more powerful, their most profound achievement may be recognising when to say "I don't know". Existing studies on LLM self-knowledge have been largely constrained by human-defined notions of feasibility, often neglecting the reasons behind unanswerability by LLMs and failing to study deficient types of self-knowledge. This study aims to obtain intrinsic insights into different types of LLM self-knowledge with a novel methodology: allowing them the flexibility to set their own feasibility boundaries and then analysing the consistency of these limits. We find that even frontier models like GPT-4o and Mistral Large are not sure of their own capabilities more than 80% of the time, highlighting a significant lack of trustworthiness in responses. Our analysis of confidence balance in LLMs indicates that models swing between overconfidence and conservatism in feasibility boundaries depending on task categories and that the most significant self-knowledge weaknesses lie in temporal awareness and contextual understanding. These difficulties in contextual comprehension additionally lead models to question their operational boundaries, resulting in considerable confusion within the self-knowledge of LLMs. We make our code and results available publicly at this https URL 

**Abstract (ZH)**: 随着大规模语言模型变得愈发强大，它们最深刻的成就可能是认识到何时应该说“我不知道”。现有的关于语言模型自我认知的研究大多受限于人类定义的可行性观念，往往忽视了语言模型无法回答问题的原因，并未研究缺陷类型的自我认知。本研究旨在通过一种新颖的方法论获得不同类型语言模型自我认知的内在洞见：允许它们自行设定可行性边界，然后分析这些边界的连贯性。我们发现即使是前沿模型如GPT-4o和Mistral Large也有超过80%的时间对自己能力不确定，突显了响应中的显著不可靠性。我们对语言模型信心平衡的分析表明，模型在可行性边界上会根据任务类别在过度自信和保守之间摇摆，自我认知中最显著的弱点在于时间意识和上下文理解。这些在上下文理解上的困难还导致模型质疑其操作边界，从而在语言模型的自我认知中造成了极大的混乱。我们将在以下网址公开我们的代码和结果：这个 https URL。 

---
# Spherical Tree-Sliced Wasserstein Distance 

**Title (ZH)**: 球形树分割Wasserstein距离 

**Authors**: Hoang V. Tran, Thanh T. Chu, Khoi N.M. Nguyen, Trang Pham, Tam Le, Tan M. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11249)  

**Abstract**: Sliced Optimal Transport (OT) simplifies the OT problem in high-dimensional spaces by projecting supports of input measures onto one-dimensional lines and then exploiting the closed-form expression of the univariate OT to reduce the computational burden of OT. Recently, the Tree-Sliced method has been introduced to replace these lines with more intricate structures, known as tree systems. This approach enhances the ability to capture topological information of integration domains in Sliced OT while maintaining low computational cost. Inspired by this approach, in this paper, we present an adaptation of tree systems on OT problems for measures supported on a sphere. As a counterpart to the Radon transform variant on tree systems, we propose a novel spherical Radon transform with a new integration domain called spherical trees. By leveraging this transform and exploiting the spherical tree structures, we derive closed-form expressions for OT problems on the sphere. Consequently, we obtain an efficient metric for measures on the sphere, named Spherical Tree-Sliced Wasserstein (STSW) distance. We provide an extensive theoretical analysis to demonstrate the topology of spherical trees and the well-definedness and injectivity of our Radon transform variant, which leads to an orthogonally invariant distance between spherical measures. Finally, we conduct a wide range of numerical experiments, including gradient flows and self-supervised learning, to assess the performance of our proposed metric, comparing it to recent benchmarks. 

**Abstract (ZH)**: 基于树木结构的球面截断最优传输距离（Spherical Tree-Sliced Wasserstein Distance） 

---
# Compound Expression Recognition via Large Vision-Language Models 

**Title (ZH)**: 基于大型视觉-语言模型的复合表达识别 

**Authors**: Jun Yu, Xilong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11241)  

**Abstract**: Compound Expression Recognition (CER) is crucial for understanding human emotions and improving human-computer interaction. However, CER faces challenges due to the complexity of facial expressions and the difficulty of capturing subtle emotional cues. To address these issues, we propose a novel approach leveraging Large Vision-Language Models (LVLMs). Our method employs a two-stage fine-tuning process: first, pre-trained LVLMs are fine-tuned on basic facial expressions to establish foundational patterns; second, the model is further optimized on a compound-expression dataset to refine visual-language feature interactions. Our approach achieves advanced accuracy on the RAF-DB dataset and demonstrates strong zero-shot generalization on the C-EXPR-DB dataset, showcasing its potential for real-world applications in emotion analysis and human-computer interaction. 

**Abstract (ZH)**: 复合表情识别（CER）对于理解人类情绪和提升人机交互至关重要。然而，由于面部表情的复杂性和微妙情绪线索的捕捉难度，CER 面临挑战。为了解决这些问题，我们提出了一种利用大型视觉-语言模型（LVLMs）的新方法。我们的方法采用两阶段微调过程：首先，预训练的LVLMs在基本面部表情上进行微调以建立基础模式；其次，在复合表情数据集上进一步优化模型以精化视觉-语言特征交互。我们的方法在RAF-DB数据集上实现了高级准确率，并在C-EXPR-DB数据集上展示了强大的零样本泛化能力，展示了其在情绪分析和人机交互中的潜在应用价值。 

---
# Technologies on Effectiveness and Efficiency: A Survey of State Spaces Models 

**Title (ZH)**: 有效性和效率上的技术进展：状态空间模型综述 

**Authors**: Xingtai Lv, Youbang Sun, Kaiyan Zhang, Shang Qu, Xuekai Zhu, Yuchen Fan, Yi Wu, Ermo Hua, Xinwei Long, Ning Ding, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.11224)  

**Abstract**: State Space Models (SSMs) have emerged as a promising alternative to the popular transformer-based models and have been increasingly gaining attention. Compared to transformers, SSMs excel at tasks with sequential data or longer contexts, demonstrating comparable performances with significant efficiency gains. In this survey, we provide a coherent and systematic overview for SSMs, including their theoretical motivations, mathematical formulations, comparison with existing model classes, and various applications. We divide the SSM series into three main sections, providing a detailed introduction to the original SSM, the structured SSM represented by S4, and the selective SSM typified by Mamba. We put an emphasis on technicality, and highlight the various key techniques introduced to address the effectiveness and efficiency of SSMs. We hope this manuscript serves as an introduction for researchers to explore the theoretical foundations of SSMs. 

**Abstract (ZH)**: 状态空间模型（SSMs）作为一种有前途的替代Transformer模型的选择，正逐渐受到关注。与Transformer相比，SSMs在处理序列数据或长上下文任务时表现出色，展现出相当的性能并具有显著的效率提升。在本文综述中，我们提供了一个连贯且系统的SSMs概览，包括其理论动机、数学公式、与现有模型类别的比较以及各种应用。我们将SSM系列分为三个主要部分，详细介绍了原始的SSM、由S4代表的结构化SSM以及由Mamba代表的选择性SSM。我们在技术性方面进行了强调，并突出了为提高SSMs的有效性和效率而引入的各种关键技术。我们希望本文献能够为研究人员探索SSMs的基础理论提供一个入门介绍。 

---
# MEET: A Million-Scale Dataset for Fine-Grained Geospatial Scene Classification with Zoom-Free Remote Sensing Imagery 

**Title (ZH)**: MEET：一种基于无缩放遥感图像的百万规模细粒度地理场景分类数据集 

**Authors**: Yansheng Li, Yuning Wu, Gong Cheng, Chao Tao, Bo Dang, Yu Wang, Jiahao Zhang, Chuge Zhang, Yiting Liu, Xu Tang, Jiayi Ma, Yongjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11219)  

**Abstract**: Accurate fine-grained geospatial scene classification using remote sensing imagery is essential for a wide range of applications. However, existing approaches often rely on manually zooming remote sensing images at different scales to create typical scene samples. This approach fails to adequately support the fixed-resolution image interpretation requirements in real-world scenarios. To address this limitation, we introduce the Million-scale finE-grained geospatial scEne classification dataseT (MEET), which contains over 1.03 million zoom-free remote sensing scene samples, manually annotated into 80 fine-grained categories. In MEET, each scene sample follows a scene-inscene layout, where the central scene serves as the reference, and auxiliary scenes provide crucial spatial context for finegrained classification. Moreover, to tackle the emerging challenge of scene-in-scene classification, we present the Context-Aware Transformer (CAT), a model specifically designed for this task, which adaptively fuses spatial context to accurately classify the scene samples. CAT adaptively fuses spatial context to accurately classify the scene samples by learning attentional features that capture the relationships between the center and auxiliary scenes. Based on MEET, we establish a comprehensive benchmark for fine-grained geospatial scene classification, evaluating CAT against 11 competitive baselines. The results demonstrate that CAT significantly outperforms these baselines, achieving a 1.88% higher balanced accuracy (BA) with the Swin-Large backbone, and a notable 7.87% improvement with the Swin-Huge backbone. Further experiments validate the effectiveness of each module in CAT and show the practical applicability of CAT in the urban functional zone mapping. The source code and dataset will be publicly available at this https URL. 

**Abstract (ZH)**: 大规模无缩放遥感细粒度地理场景分类数据集（MEET）：基于上下文感知变换器的细粒度地理场景分类 

---
# Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering 

**Title (ZH)**: 强化学习优于监督微调：音频问答案例研究 

**Authors**: Gang Li, Jizhong Liu, Heinrich Dinkel, Yadong Niu, Junbo Zhang, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2503.11197)  

**Abstract**: Recently, reinforcement learning (RL) has been shown to greatly enhance the reasoning capabilities of large language models (LLMs), and RL-based approaches have been progressively applied to visual multimodal tasks. However, the audio modality has largely been overlooked in these developments. Thus, we conduct a series of RL explorations in audio understanding and reasoning, specifically focusing on the audio question answering (AQA) task. We leverage the group relative policy optimization (GRPO) algorithm to Qwen2-Audio-7B-Instruct, and our experiments demonstrated state-of-the-art performance on the MMAU Test-mini benchmark, achieving an accuracy rate of 64.5%. The main findings in this technical report are as follows: 1) The GRPO algorithm can be effectively applied to large audio language models (LALMs), even when the model has only 8.2B parameters; 2) With only 38k post-training samples, RL significantly outperforms supervised fine-tuning (SFT), indicating that RL-based approaches can be effective without large datasets; 3) The explicit reasoning process has not shown significant benefits for AQA tasks, and how to efficiently utilize deep thinking remains an open question for further research; 4) LALMs still lag far behind humans auditory-language reasoning, suggesting that the RL-based approaches warrant further exploration. Our project is available at this https URL and this https URL. 

**Abstract (ZH)**: 最近，强化学习(RL)已被证明大幅提升了大型语言模型(LLMs)的推理能力，基于RL的方法逐渐被应用于视觉多模态任务中。然而，音频模态在这方面的进展被严重忽视。因此，我们在音频理解与推理方面进行了一系列RL探索，特别关注音频问答(AQA)任务。我们利用群相对策略优化(GRPO)算法对Qwen2-Audio-7B-Instruct进行了优化，并在MMAU Test-mini基准测试中实现了64.5%的准确率，展示了目前最先进的性能。本技术报告的主要发现如下：1) GRPO算法可以有效地应用于大型音频语言模型(LALMs)，即使模型仅有8.2B参数；2) 仅使用38k训练后样本，RL显著优于监督微调(SFT)，表明基于RL的方法可以在无需大规模数据集的情况下有效；3) 显式的推理过程并未在AQA任务中显示出显著优势，如何高效利用深度思考仍是一个待解决的问题；4) LALMs在听觉语言推理方面仍然远远落后于人类，表明基于RL的方法仍需进一步探索。我们的项目可访问此URL和此URL。 

---
# Cross-Modal Learning for Music-to-Music-Video Description Generation 

**Title (ZH)**: 音乐到音乐视频描述生成的跨模态学习 

**Authors**: Zhuoyuan Mao, Mengjie Zhao, Qiyu Wu, Zhi Zhong, Wei-Hsiang Liao, Hiromi Wakaki, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2503.11190)  

**Abstract**: Music-to-music-video generation is a challenging task due to the intrinsic differences between the music and video modalities. The advent of powerful text-to-video diffusion models has opened a promising pathway for music-video (MV) generation by first addressing the music-to-MV description task and subsequently leveraging these models for video generation. In this study, we focus on the MV description generation task and propose a comprehensive pipeline encompassing training data construction and multimodal model fine-tuning. We fine-tune existing pre-trained multimodal models on our newly constructed music-to-MV description dataset based on the Music4All dataset, which integrates both musical and visual information. Our experimental results demonstrate that music representations can be effectively mapped to textual domains, enabling the generation of meaningful MV description directly from music inputs. We also identify key components in the dataset construction pipeline that critically impact the quality of MV description and highlight specific musical attributes that warrant greater focus for improved MV description generation. 

**Abstract (ZH)**: 基于音乐的音乐视频生成是一个具有挑战性的任务，由于音乐和视频模态之间的内在差异。强大的文本到视频扩散模型的出现为音乐视频（MV）生成开辟了一条有希望的道路，首先通过解决音乐到MV描述任务，随后利用这些模型进行视频生成。在本研究中，我们聚焦于MV描述生成任务，并提出一个综合的流程，涵盖训练数据构建和多模态模型微调。我们在一个新构建的基于Music4All数据集的音乐到MV描述数据集上微调现有的预训练多模态模型，该数据集整合了音乐和视觉信息。我们的实验结果表明，音乐表示可以有效地映射到文本域，从而能够直接从音乐输入生成有意义的MV描述。我们还识别出数据集构建流程中关键的组成部分，这些组成部分对MV描述的质量有直接影响，并强调了需要更多关注的具体音乐属性，以提高MV描述生成。 

---
# Align in Depth: Defending Jailbreak Attacks via Progressive Answer Detoxification 

**Title (ZH)**: 深度对齐：通过渐进式答案去毒来防御 Jailbreak 攻击 

**Authors**: Yingjie Zhang, Tong Liu, Zhe Zhao, Guozhu Meng, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11185)  

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks, which use crafted prompts to elicit toxic responses. These attacks exploit LLMs' difficulty in dynamically detecting harmful intents during the generation process. Traditional safety alignment methods, often relying on the initial few generation steps, are ineffective due to limited computational budget. This paper proposes DEEPALIGN, a robust defense framework that fine-tunes LLMs to progressively detoxify generated content, significantly improving both the computational budget and effectiveness of mitigating harmful generation. Our approach uses a hybrid loss function operating on hidden states to directly improve LLMs' inherent awareness of toxity during generation. Furthermore, we redefine safe responses by generating semantically relevant answers to harmful queries, thereby increasing robustness against representation-mutation attacks. Evaluations across multiple LLMs demonstrate state-of-the-art defense performance against six different attack types, reducing Attack Success Rates by up to two orders of magnitude compared to previous state-of-the-art defense while preserving utility. This work advances LLM safety by addressing limitations of conventional alignment through dynamic, context-aware mitigation. 

**Abstract (ZH)**: 大型语言模型（LLMs）易受监狱突破攻击的威胁，这种攻击利用精心设计的提示诱使模型产生有害响应。这些攻击利用了LLMs在生成过程中难以动态检测有害意图的困难。传统的安全对齐方法往往依赖于初始的少量生成步骤，但由于计算预算有限而无效。本文提出了一种名为DEEPALIGN的鲁棒防御框架，通过微调LLMs使其逐步净化生成的内容，显著提高了计算预算和缓解有害生成的有效性。我们的方法使用一个在隐藏状态上操作的混合损失函数，直接提升LLMs在生成过程中对毒性的内在感知。此外，我们通过为有害查询生成语义相关答案重新定义安全响应，从而增强对表征突变攻击的鲁棒性。多种LLMs的评估表明，DEEPALIGN在六种不同攻击类型的防护性能上达到了最先进的技术水平，相较于之前的最先进防护方法将攻击成功率降低了两个数量级以上，同时保持了实用性。这项工作通过动态、上下文感知的缓解措施，推动了LLMs安全性的提升，解决了传统对齐方法的局限性。 

---
# Multi-Stage Generative Upscaler: Reconstructing Football Broadcast Images via Diffusion Models 

**Title (ZH)**: 多阶段生成上放大模型：基于扩散模型的足球广播图像重构 

**Authors**: Luca Martini, Daniele Zolezzi, Saverio Iacono, Gianni Viardo Vercelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.11181)  

**Abstract**: The reconstruction of low-resolution football broadcast images presents a significant challenge in sports broadcasting, where detailed visuals are essential for analysis and audience engagement. This study introduces a multi-stage generative upscaling framework leveraging Diffusion Models to enhance degraded images, transforming inputs as small as $64 \times 64$ pixels into high-fidelity $1024 \times 1024$ outputs. By integrating an image-to-image pipeline, ControlNet conditioning, and LoRA fine-tuning, our approach surpasses traditional upscaling methods in restoring intricate textures and domain-specific elements such as player details and jersey logos. The custom LoRA is trained on a custom football dataset, ensuring adaptability to sports broadcast needs. Experimental results demonstrate substantial improvements over conventional models, with ControlNet refining fine details and LoRA enhancing task-specific elements. These findings highlight the potential of diffusion-based image reconstruction in sports media, paving the way for future applications in automated video enhancement and real-time sports analytics. 

**Abstract (ZH)**: 低分辨率足球直播图像的重建在体育广播中是一项重要的挑战，详细的视觉内容对于分析和观众参与至关重要。本研究提出了一种基于扩散模型的多阶段生成放大框架，通过提升降级图像，将输入的最小尺寸从64×64像素转换为高质量的1024×1024输出。通过整合图像到图像的处理管道、ControlNet条件控制和LoRA微调，我们的方法在恢复复杂的纹理和特定领域的元素（如球员细节和球衣商标）方面超过了传统的放大方法。定制的LoRA在定制的足球数据集上进行训练，确保适应体育广播的需求。实验结果表明，与传统模型相比，我们的方法取得了显著的改进，ControlNet细化了细部细节，而LoRA增强了特定任务的元素。这些发现突显了基于扩散模型的图像重建在体育媒体中的潜力，为未来的自动视频增强和实时体育分析应用铺平了道路。 

---
# Zero-TIG: Temporal Consistency-Aware Zero-Shot Illumination-Guided Low-light Video Enhancement 

**Title (ZH)**: Zero-TIG：面向时间一致性约束的零-shot 背景光引导低光视频增强 

**Authors**: Yini Li, Nantheera Anantrasirichai  

**Link**: [PDF](https://arxiv.org/pdf/2503.11175)  

**Abstract**: Low-light and underwater videos suffer from poor visibility, low contrast, and high noise, necessitating enhancements in visual quality. However, existing approaches typically rely on paired ground truth, which limits their practicality and often fails to maintain temporal consistency. To overcome these obstacles, this paper introduces a novel zero-shot learning approach named Zero-TIG, leveraging the Retinex theory and optical flow techniques. The proposed network consists of an enhancement module and a temporal feedback module. The enhancement module comprises three subnetworks: low-light image denoising, illumination estimation, and reflection denoising. The temporal enhancement module ensures temporal consistency by incorporating histogram equalization, optical flow computation, and image warping to align the enhanced previous frame with the current frame, thereby maintaining continuity. Additionally, we address color distortion in underwater data by adaptively balancing RGB channels. The experimental results demonstrate that our method achieves low-light video enhancement without the need for paired training data, making it a promising and applicable method for real-world scenario enhancement. 

**Abstract (ZH)**: 低光照和水下视频由于能见度低、对比度低和噪声高，需要在视觉质量上进行增强。为克服现有方法对配对 ground truth 的依赖和时间一致性问题，本文提出了一种名为 Zero-TIG 的零样本学习方法，利用 Retinex 理论和光学流技术。所提出的网络包括增强模块和时间反馈模块。增强模块由三个子网络组成：低光照图像去噪、照度估计和反射去噪。时间增强模块通过引入直方图均衡化、光学流计算和图像变形，确保增强的前一帧与当前帧对齐，从而保持连续性。此外，本文通过自适应平衡 RGB 通道解决了水下数据的色彩失真问题。实验结果表明，该方法能够在无需配对训练数据的情况下实现低光照视频增强，具有在实际场景中增强应用的潜力。 

---
# Neurons: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction 

**Title (ZH)**: 神经元：模拟人类视觉皮层提高fMRI到视频重建的 fidelity 和可解释性 

**Authors**: Haonan Wang, Qixiang Zhang, Lehan Wang, Xuanqi Huang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.11167)  

**Abstract**: Decoding visual stimuli from neural activity is essential for understanding the human brain. While fMRI methods have successfully reconstructed static images, fMRI-to-video reconstruction faces challenges due to the need for capturing spatiotemporal dynamics like motion and scene transitions. Recent approaches have improved semantic and perceptual alignment but struggle to integrate coarse fMRI data with detailed visual features. Inspired by the hierarchical organization of the visual system, we propose NEURONS, a novel framework that decouples learning into four correlated sub-tasks: key object segmentation, concept recognition, scene description, and blurry video reconstruction. This approach simulates the visual cortex's functional specialization, allowing the model to capture diverse video content. In the inference stage, NEURONS generates robust conditioning signals for a pre-trained text-to-video diffusion model to reconstruct the videos. Extensive experiments demonstrate that NEURONS outperforms state-of-the-art baselines, achieving solid improvements in video consistency (26.6%) and semantic-level accuracy (19.1%). Notably, NEURONS shows a strong functional correlation with the visual cortex, highlighting its potential for brain-computer interfaces and clinical applications. Code and model weights will be available at: this https URL. 

**Abstract (ZH)**: 从神经活动解码视觉刺激是理解人类大脑的关键。尽管fMRI方法成功地重建了静态图像，但由于需要捕获如运动和场景转换的时空动态，fMRI到视频的重建面临着挑战。近期的方法在语义和感知一致性方面取得了进步，但仍难以将粗粒度的fMRI数据与详细的视觉特征整合。受视觉系统层次组织的启发，我们提出了一种名为NEURONS的新框架，将学习任务分解为四个相关子任务：关键对象分割、概念识别、场景描述和模糊视频重建。该方法模拟了视觉皮层的功能专业化，使模型能够捕捉多种视频内容。在推理阶段，NEURONS为预训练的文本到视频扩散模型生成稳健的条件信号，以重建视频。广泛实验表明，NEURONS优于现有 baseline，视频一致性提高26.6%，语义水平准确性提高19.1%。值得注意的是，NEURONS与视觉皮层的功能相关性较强，突显了其在脑机接口和临床应用中的潜力。代码和模型权重将在以下网址获取：this https URL。 

---
# Unifying Perplexing Behaviors in Modified BP Attributions through Alignment Perspective 

**Title (ZH)**: 通过对齐视角统一修改BP Attribution中的迷惑行为 

**Authors**: Guanhua Zheng, Jitao Sang, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11160)  

**Abstract**: Attributions aim to identify input pixels that are relevant to the decision-making process. A popular approach involves using modified backpropagation (BP) rules to reverse decisions, which improves interpretability compared to the original gradients. However, these methods lack a solid theoretical foundation and exhibit perplexing behaviors, such as reduced sensitivity to parameter randomization, raising concerns about their reliability and highlighting the need for theoretical justification. In this work, we present a unified theoretical framework for methods like GBP, RectGrad, LRP, and DTD, demonstrating that they achieve input alignment by combining the weights of activated neurons. This alignment improves the visualization quality and reduces sensitivity to weight randomization. Our contributions include: (1) Providing a unified explanation for multiple behaviors, rather than focusing on just one. (2) Accurately predicting novel behaviors. (3) Offering insights into decision-making processes, including layer-wise information changes and the relationship between attributions and model decisions. 

**Abstract (ZH)**: Attribution方法旨在识别对决策过程相关的输入像素。一种流行的方法是使用修改的反向传播（BP）规则来推翻决策，这与原始梯度相比提高了可解释性。然而，这些方法缺乏坚实的信息学基础，并表现出令人困惑的行为，如参数随机化敏感性降低，这引起了对其可靠性的担忧，并强调了需要理论证明的需求。在这项工作中，我们为GBP、RectGrad、LRP和DTD等方法提供了一个统一的理论框架，证明它们通过结合激活神经元的权重来实现输入对齐。这种对齐提高了可视化质量并降低了对权重随机化的敏感性。我们的贡献包括：(1) 对多种行为提供统一的解释，而不是仅仅关注一种。(2) 准确预测新的行为。(3) 提供有关决策过程的见解，包括层间信息变化以及归因与模型决策之间的关系。 

---
# Don't Take Things Out of Context: Attention Intervention for Enhancing Chain-of-Thought Reasoning in Large Language Models 

**Title (ZH)**: 不要断章取义：注意力干预增强大型语言模型的因果推理能力 

**Authors**: Shaotian Yan, Chen Shen, Wenxiao Wang, Liang Xie, Junjie Liu, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.11154)  

**Abstract**: Few-shot Chain-of-Thought (CoT) significantly enhances the reasoning capabilities of large language models (LLMs), functioning as a whole to guide these models in generating reasoning steps toward final answers. However, we observe that isolated segments, words, or tokens within CoT demonstrations can unexpectedly disrupt the generation process of LLMs. The model may overly concentrate on certain local information present in the demonstration, introducing irrelevant noise into the reasoning process and potentially leading to incorrect answers. In this paper, we investigate the underlying mechanism of CoT through dynamically tracing and manipulating the inner workings of LLMs at each output step, which demonstrates that tokens exhibiting specific attention characteristics are more likely to induce the model to take things out of context; these tokens directly attend to the hidden states tied with prediction, without substantial integration of non-local information. Building upon these insights, we propose a Few-shot Attention Intervention method (FAI) that dynamically analyzes the attention patterns of demonstrations to accurately identify these tokens and subsequently make targeted adjustments to the attention weights to effectively suppress their distracting effect on LLMs. Comprehensive experiments across multiple benchmarks demonstrate consistent improvements over baseline methods, with a remarkable 5.91% improvement on the AQuA dataset, further highlighting the effectiveness of FAI. 

**Abstract (ZH)**: Few-shot Attention Intervention显著提升大型语言模型的推理能力通过动态追踪和操控每次输出步骤中的内部工作原理，揭示特定注意力特征的标记更容易导致模型脱离上下文；基于这些洞见，我们提出了一种Few-shot Attention Intervention (FAI) 方法，动态分析示范的注意力模式以准确识别这些标记，并进行针对性调整以有效抑制其对大型语言模型的干扰效果。跨多个基准的综合实验展示了FAI在基线方法上的一致改进，特别是在AQuA数据集上提高了5.91%，进一步突显了FAI的有效性。 

---
# MoLEx: Mixture of Layer Experts for Finetuning with Sparse Upcycling 

**Title (ZH)**: MoLEx：混合层专家模型以稀疏升级方式进行微调 

**Authors**: Rachel S.Y. Teo, Tan M. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11144)  

**Abstract**: Large-scale pre-training of deep models, followed by fine-tuning them, has become the cornerstone of natural language processing (NLP). The prevalence of data coupled with computational resources has led to large models with a considerable number of parameters. While the massive size of these models has led to remarkable success in many NLP tasks, a detriment is the expense required to retrain all the base model's parameters for the adaptation to each task or domain. Parameter Efficient Fine-Tuning (PEFT) provides an effective solution for this challenge by minimizing the number of parameters required to be fine-tuned while maintaining the quality of the model. While existing methods have achieved impressive results, they mainly focus on adapting a subset of parameters, weight reparameterization, and prompt engineering. In this paper, we study layers as extractors of different types of linguistic information that are valuable when used in conjunction. We then propose the Mixture of Layer Experts (MoLEx), a novel sparse mixture of experts (SMoE) whose experts are layers in the pre-trained model. It performs a conditional computation of a mixture of layers during fine-tuning to provide the model with more structural knowledge about the data. By providing an avenue for information exchange between layers, MoLEx enables the model to make a more well-informed prediction for the downstream task, leading to better fine-tuning results with the same number of effective parameters. As experts can be processed in parallel, MoLEx introduces minimal additional computational overhead. We empirically corroborate the advantages of MoLEx when combined with popular PEFT baseline methods on a variety of downstream fine-tuning tasks, including the popular GLUE benchmark as well as the End-to-End Challenge (E2E). The code is publicly available at this https URL. 

**Abstract (ZH)**: 大规模预训练深度模型，随后进行微调，已成为自然语言处理（NLP）的基石。随着数据和计算资源的普及，导致了具有大量参数的大规模模型。虽然这些大规模模型在许多NLP任务中取得了显著成功，但一个弊端是重新训练基础模型所有参数以适应每个任务或领域所需的高昂成本。参数高效微调（PEFT）通过在减少需要微调的参数数量的同时保持模型质量，提供了解决这一挑战的有效方案。虽然现有方法已取得令人 impressive 的结果，但它们主要集中在适应参数子集、权重重参数化和提示工程上。在本文中，我们研究了图层作为提取不同类型语言信息的提取器，这些信息在结合使用时具有价值。然后，我们提出了图层专家混叠（MoLEx），这是一种新颖的稀疏专家混叠（SMoE），其中专家是预训练模型中的图层。在微调期间，MoLEx 进行条件图层混合的计算，为模型提供有关数据的更多结构知识。通过在图层之间提供信息交换途径，MoLEx 使模型能够为下游任务做出更为明智的预测，从而在相同数量的有效参数下获得更好的微调效果。由于专家可以并行处理，MoLEx 引入了 minimal 的额外计算开销。我们通过在各种下游微调任务上与流行的 PEFT 基线方法结合使用，实验证实了 MoLEx 的优势，包括流行的 GLUE 基准以及 End-to-End 挑战（E2E）。代码已在以下网址公开：this https URL。 

---
# Direction-Aware Diagonal Autoregressive Image Generation 

**Title (ZH)**: 方向意识的对角自回归图像生成 

**Authors**: Yijia Xu, Jianzhong Ju, Jian Luan, Jinshi Cui  

**Link**: [PDF](https://arxiv.org/pdf/2503.11129)  

**Abstract**: The raster-ordered image token sequence exhibits a significant Euclidean distance between index-adjacent tokens at line breaks, making it unsuitable for autoregressive generation. To address this issue, this paper proposes Direction-Aware Diagonal Autoregressive Image Generation (DAR) method, which generates image tokens following a diagonal scanning order. The proposed diagonal scanning order ensures that tokens with adjacent indices remain in close proximity while enabling causal attention to gather information from a broader range of directions. Additionally, two direction-aware modules: 4D-RoPE and direction embeddings are introduced, enhancing the model's capability to handle frequent changes in generation direction. To leverage the representational capacity of the image tokenizer, we use its codebook as the image token embeddings. We propose models of varying scales, ranging from 485M to 2.0B. On the 256$\times$256 ImageNet benchmark, our DAR-XL (2.0B) outperforms all previous autoregressive image generators, achieving a state-of-the-art FID score of 1.37. 

**Abstract (ZH)**: 方向感知对角自回归图像生成方法（DAR） 

---
# Don't Forget It! Conditional Sparse Autoencoder Clamping Works for Unlearning 

**Title (ZH)**: 别忘了它！条件稀疏自编码器钳制法适用于遗忘学习 

**Authors**: Matthew Khoriaty, Andrii Shportko, Gustavo Mercier, Zach Wood-Doughty  

**Link**: [PDF](https://arxiv.org/pdf/2503.11127)  

**Abstract**: Recent developments in Large Language Model (LLM) capabilities have brought great potential but also posed new risks. For example, LLMs with knowledge of bioweapons, advanced chemistry, or cyberattacks could cause violence if placed in the wrong hands or during malfunctions. Because of their nature as near-black boxes, intuitive interpretation of LLM internals remains an open research question, preventing developers from easily controlling model behavior and capabilities. The use of Sparse Autoencoders (SAEs) has recently emerged as a potential method of unraveling representations of concepts in LLMs internals, and has allowed developers to steer model outputs by directly modifying the hidden activations. In this paper, we use SAEs to identify unwanted concepts from the Weapons of Mass Destruction Proxy (WMDP) dataset within gemma-2-2b internals and use feature steering to reduce the model's ability to answer harmful questions while retaining its performance on harmless queries. Our results bring back optimism to the viability of SAE-based explicit knowledge unlearning techniques. 

**Abstract (ZH)**: 近期大型语言模型（LLM）能力的发展带来了巨大潜力但也提出了新的风险。由于其近似黑箱的性质，对LLM内部机制的直观解释仍然是一个开放的研究问题，阻碍了开发人员轻松控制模型行为和能力。最近，稀疏自编码器（SAEs）的使用作为一种可能的方法来解开LLM内部概念的表示，并允许开发人员通过直接修改隐藏激活来引导模型输出。在本文中，我们使用SAEs在gemma-2-2b内部识别来自大规模破坏性武器代理（WMDP）数据集的不希望的概念，并通过特征引导来降低模型回答有害问题的能力，同时保留其在无害查询上的性能。我们的结果重新唤起了基于SAE的显式知识遗忘技术可行性的乐观态度。 

---
# UMB@PerAnsSumm 2025: Enhancing Perspective-Aware Summarization with Prompt Optimization and Supervised Fine-Tuning 

**Title (ZH)**: UMB@PerAnsSumm 2025: 基于提示优化和监督微调的视角aware摘要生成增强 

**Authors**: Kristin Qi, Youxiang Zhu, Xiaohui Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11118)  

**Abstract**: We present our approach to the PerAnsSumm Shared Task, which involves perspective span identification and perspective-aware summarization in community question-answering (CQA) threads. For span identification, we adopt ensemble learning that integrates three transformer models through averaging to exploit individual model strengths, achieving an 82.91% F1-score on test data. For summarization, we design a suite of Chain-of-Thought (CoT) prompting strategies that incorporate keyphrases and guide information to structure summary generation into manageable steps. To further enhance summary quality, we apply prompt optimization using the DSPy framework and supervised fine-tuning (SFT) on Llama-3 to adapt the model to domain-specific data. Experimental results on validation and test sets show that structured prompts with keyphrases and guidance improve summaries aligned with references, while the combination of prompt optimization and fine-tuning together yields significant improvement in both relevance and factuality evaluation metrics. 

**Abstract (ZH)**: 我们提出了针对PerAnsSumm共享任务的方法，该方法涉及社区问答（CQA）线程中的视角短语识别和视角意识总结。在短语识别方面，我们采用通过平均整合三种变压器模型的集成学习方法，以利用各个模型的优势，测试数据上的F1分为82.91%。在总结方面，我们设计了一套包含关键词和指导信息的Chain-of-Thought（CoT）提示策略，以分步骤结构化总结生成。为了进一步提高总结质量，我们使用DSPy框架进行提示优化，并在Llama-3上进行监督微调（SFT），以使模型适应特定领域数据。在验证集和测试集上的实验结果表明，包含关键词和指导信息的结构化提示可以提高与参考文本对齐的总结质量，而提示优化与微调的结合则在相关性和事实性评估指标上取得了显著改进。 

---
# Limits of KV Cache Compression for Tensor Attention based Autoregressive Transformers 

**Title (ZH)**: 基于张量注意力的自回归变换器中键值缓存压缩的极限 

**Authors**: Yifang Chen, Xiaoyu Li, Yingyu Liang, Zhenmei Shi, Zhao Song, Yu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2503.11108)  

**Abstract**: The key-value (KV) cache in autoregressive transformers presents a significant bottleneck during inference, which restricts the context length capabilities of large language models (LLMs). While previous work analyzes the fundamental space complexity barriers in standard attention mechanism [Haris and Onak, 2025], our work generalizes the space complexity barriers result to tensor attention version. Our theoretical contributions rely on a novel reduction from communication complexity and deduce the memory lower bound for tensor-structured attention mechanisms when $d = \Omega(\log n)$. In the low dimensional regime where $d = o(\log n)$, we analyze the theoretical bounds of the space complexity as well. Overall, our work provides a theoretical foundation for us to understand the compression-expressivity tradeoff in tensor attention mechanisms and offers more perspectives in developing more memory-efficient transformer architectures. 

**Abstract (ZH)**: 自回归变压器中的键值缓存对推理过程构成显著瓶颈，限制了大型语言模型的上下文长度能力。尽管以往工作分析了标准注意机制的基本空间复杂度障碍[Haris和Onak, 2025]，我们的工作将空间复杂度障碍的结果推广到张量注意版本。我们的理论贡献基于一种新颖的通信复杂度归约，并推导出当 $d = \Omega(\log n)$ 时张量结构注意机制的存储下界。在低维情形下，即 $d = o(\log n)$，我们分析了空间复杂度的理论界限。总体而言，我们的工作为我们理解张量注意机制中的压缩-表达性权衡提供了理论基础，并提供了开发更高效的变压器架构的新视角。 

---
# Quantifying Interpretability in CLIP Models with Concept Consistency 

**Title (ZH)**: CLIP模型中概念一致性解释性量化 

**Authors**: Avinash Madasu, Vasudev Lal, Phillip Howard  

**Link**: [PDF](https://arxiv.org/pdf/2503.11103)  

**Abstract**: CLIP is one of the most popular foundational models and is heavily used for many vision-language tasks. However, little is known about the inner workings of CLIP. While recent work has proposed decomposition-based interpretability methods for identifying textual descriptions of attention heads in CLIP, the implications of conceptual consistency in these text labels on interpretability and model performance has not been explored. To bridge this gap, we study the conceptual consistency of text descriptions for attention heads in CLIP-like models. We conduct extensive experiments on six different models from OpenAI and OpenCLIP which vary by size, type of pre-training data and patch size. We propose Concept Consistency Score (CCS), a novel interpretability metric that measures how consistently individual attention heads in CLIP models align with specific concepts. To assign concept labels to heads, we use in-context learning with ChatGPT, guided by a few manually-curated examples, and validate these labels using an LLM-as-a-judge approach. Our soft-pruning experiments reveal that high CCS heads are critical for preserving model performance, as pruning them leads to a significantly larger performance drop than pruning random or low CCS heads. Notably, we find that high CCS heads capture essential concepts and play a key role in out-of-domain detection, concept-specific reasoning, and video-language understanding. These results position CCS as a powerful interpretability metric for analyzing CLIP-like models. 

**Abstract (ZH)**: CLIP-like模型中注意力头概念一致性研究：一个新的可解释性度量标准及其影响 

---
# Augmenting Image Annotation: A Human-LMM Collaborative Framework for Efficient Object Selection and Label Generation 

**Title (ZH)**: 增强图像标注：一种人类-机器学习协作框架，用于高效对象选择和标签生成 

**Authors**: He Zhang, Xinyi Fu, John M. Carroll  

**Link**: [PDF](https://arxiv.org/pdf/2503.11096)  

**Abstract**: Traditional image annotation tasks rely heavily on human effort for object selection and label assignment, making the process time-consuming and prone to decreased efficiency as annotators experience fatigue after extensive work. This paper introduces a novel framework that leverages the visual understanding capabilities of large multimodal models (LMMs), particularly GPT, to assist annotation workflows. In our proposed approach, human annotators focus on selecting objects via bounding boxes, while the LMM autonomously generates relevant labels. This human-AI collaborative framework enhances annotation efficiency by reducing the cognitive and time burden on human annotators. By analyzing the system's performance across various types of annotation tasks, we demonstrate its ability to generalize to tasks such as object recognition, scene description, and fine-grained categorization. Our proposed framework highlights the potential of this approach to redefine annotation workflows, offering a scalable and efficient solution for large-scale data labeling in computer vision. Finally, we discuss how integrating LMMs into the annotation pipeline can advance bidirectional human-AI alignment, as well as the challenges of alleviating the "endless annotation" burden in the face of information overload by shifting some of the work to AI. 

**Abstract (ZH)**: 传统图像标注任务高度依赖人工选择对象和分配标签，这使得过程耗时且容易因标注员长时间工作而降低效率。本文介绍了一种新的框架，利用大型多模式模型（LMM）特别是GPT的视觉理解能力来辅助标注流程。在我们提出的方法中，人类标注员专注于通过边界框选择对象，而LMM自主生成相关标签。这种人机协作框架通过减轻人类标注员的认知和时间负担来提高标注效率。通过分析系统在各种标注任务中的性能，我们展示了其迁移到对象识别、场景描述和细粒度分类等任务的能力。我们的框架突显了该方法重新定义标注流程的潜力，提供了一种适用于计算机视觉大规模数据标签化的可扩展且高效解决方案。最后，我们讨论了将LMM集成到标注流水线中如何促进双向的人机对齐，以及如何通过将部分工作转移给AI来缓解信息过载导致的“无尽标注”负担。 

---
# EmbodiedVSR: Dynamic Scene Graph-Guided Chain-of-Thought Reasoning for Visual Spatial Tasks 

**Title (ZH)**: 基于身体感知的VSR：动态场景图引导的时空推理链条思考方法 

**Authors**: Yi Zhang, Qiang Zhang, Xiaozhu Ju, Zhaoyang Liu, Jilei Mao, Jingkai Sun, Jintao Wu, Shixiong Gao, Shihan Cai, Zhiyuan Qin, Linkai Liang, Jiaxu Wang, Yiqun Duan, Jiahang Cao, Renjing Xu, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11089)  

**Abstract**: While multimodal large language models (MLLMs) have made groundbreaking progress in embodied intelligence, they still face significant challenges in spatial reasoning for complex long-horizon tasks. To address this gap, we propose EmbodiedVSR (Embodied Visual Spatial Reasoning), a novel framework that integrates dynamic scene graph-guided Chain-of-Thought (CoT) reasoning to enhance spatial understanding for embodied agents. By explicitly constructing structured knowledge representations through dynamic scene graphs, our method enables zero-shot spatial reasoning without task-specific fine-tuning. This approach not only disentangles intricate spatial relationships but also aligns reasoning steps with actionable environmental dynamics. To rigorously evaluate performance, we introduce the eSpatial-Benchmark, a comprehensive dataset including real-world embodied scenarios with fine-grained spatial annotations and adaptive task difficulty levels. Experiments demonstrate that our framework significantly outperforms existing MLLM-based methods in accuracy and reasoning coherence, particularly in long-horizon tasks requiring iterative environment interaction. The results reveal the untapped potential of MLLMs for embodied intelligence when equipped with structured, explainable reasoning mechanisms, paving the way for more reliable deployment in real-world spatial applications. The codes and datasets will be released soon. 

**Abstract (ZH)**: 虽然多模态大型语言模型（MLLMs）在体现智能方面取得了突破性进展，但在复杂长期任务的空间推理方面仍面临重大挑战。为解决这一问题，我们提出了一种新颖的框架EmbodiedVSR（体现式视觉空间推理），该框架通过动态场景图引导的推理链（CoT）增强体现代理的空间理解。通过显式构建结构化知识表示，我们的方法能够在无需特定任务微调的情况下实现零样本空间推理。此方法不仅解耦复杂的空间关系，还将推理步骤与可操作的环境动态对齐。为了严格评估性能，我们引入了eSpatial-Benchmark，一个全面的数据集，包括具有精细空间注释和自适应任务难度级别的现实世界体现场景。实验结果表明，与现有的MLLM基方法相比，我们的框架在准确性及推理连贯性方面表现出显著优势，特别是在需要迭代环境交互的长期任务中。结果揭示了当装备了结构化和可解释推理机制时，MLLMs在体现智能领域的未开发潜力，为在现实空间应用中的更加可靠的部署铺平了道路。代码和数据集将于近期发布。 

---
# A Survey of Cross-domain Graph Learning: Progress and Future Directions 

**Title (ZH)**: 跨域图学习综述：进展与未来方向 

**Authors**: Haihong Zhao, Chenyi Zi, Aochuan Chen, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.11086)  

**Abstract**: Graph learning plays a vital role in mining and analyzing complex relationships involved in graph data, which is widely used in many real-world applications like transaction networks and communication networks. Foundation models in CV and NLP have shown powerful cross-domain capabilities that are also significant in graph domains. However, existing graph learning approaches struggle with cross-domain tasks. Inspired by successes in CV and NLP, cross-domain graph learning has once again become a focal point of attention to realizing true graph foundation models. In this survey, we present a comprehensive review and analysis of existing works on cross-domain graph learning. Concretely, we first propose a new taxonomy, categorizing existing approaches based on the learned cross-domain information: structure, feature, and structure-feature mixture. Next, we systematically survey representative methods in these categories. Finally, we discuss the remaining limitations of existing studies and highlight promising avenues for future research. Relevant papers are summarized and will be consistently updated at: this https URL. 

**Abstract (ZH)**: 基于图的学习在挖掘和分析包含图数据中的复杂关系方面发挥着关键作用，广泛应用于交易网络和通信网络等实际应用场景中。CV和NLP领域的基础模型展示了强大的跨域能力，这一能力在图领域也同样重要。然而，现有的图学习方法在处理跨域任务时遇到了困难。受CV和NLP领域成功的启发，跨域图学习再次成为了实现真正图基础模型的关键焦点。在本文综述中，我们对现有的跨域图学习工作进行了全面的回顾和分析。具体来说，我们首先提出了一种新的分类法，根据学习到的跨域信息将现有方法分为结构、特征和结构-特征混合三类。接着，我们系统地调研了这些类别中的代表性方法。最后，我们讨论了现有研究的局限性，并指出了未来研究的有希望的方向。相关论文将在此链接持续更新：this https URL。 

---
# MoMa-Kitchen: A 100K+ Benchmark for Affordance-Grounded Last-Mile Navigation in Mobile Manipulation 

**Title (ZH)**: MoMa-Kitchen：面向操作潜能导向的移动操作最后路段导航基准数据集（包含100K以上数据） 

**Authors**: Pingrui Zhang, Xianqiang Gao, Yuhan Wu, Kehui Liu, Dong Wang, Zhigang Wang, Bin Zhao, Yan Ding, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.11081)  

**Abstract**: In mobile manipulation, navigation and manipulation are often treated as separate problems, resulting in a significant gap between merely approaching an object and engaging with it effectively. Many navigation approaches primarily define success by proximity to the target, often overlooking the necessity for optimal positioning that facilitates subsequent manipulation. To address this, we introduce MoMa-Kitchen, a benchmark dataset comprising over 100k samples that provide training data for models to learn optimal final navigation positions for seamless transition to manipulation. Our dataset includes affordance-grounded floor labels collected from diverse kitchen environments, in which robotic mobile manipulators of different models attempt to grasp target objects amidst clutter. Using a fully automated pipeline, we simulate diverse real-world scenarios and generate affordance labels for optimal manipulation positions. Visual data are collected from RGB-D inputs captured by a first-person view camera mounted on the robotic arm, ensuring consistency in viewpoint during data collection. We also develop a lightweight baseline model, NavAff, for navigation affordance grounding that demonstrates promising performance on the MoMa-Kitchen benchmark. Our approach enables models to learn affordance-based final positioning that accommodates different arm types and platform heights, thereby paving the way for more robust and generalizable integration of navigation and manipulation in embodied AI. Project page: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 移动 manipulation 中，导航和操纵经常被视为两个独立的问题，这导致了仅仅接近物体与有效地与物体互动之间的显著差距。许多导航方法主要通过与目标的接近程度来定义成功，往往忽略了为后续操纵提供最优定位的必要性。为了解决这一问题，我们引入了 MoMa-Kitchen，这是一个包含超过 10 万个样本的基准数据集，为模型学习无缝过渡到操纵的最优最终导航位置提供训练数据。我们的数据集包括从多种厨房环境中收集的基于功能的地板标签，其中不同型号的机器人移动 manipulator 尝试在杂乱环境中抓取目标物体。通过完全自动化的管道，我们模拟了各种真实世界场景并生成了最优操纵位置的功能标签。视觉数据通过安装在机器人臂上的第一视角相机捕获的 RGB-D 输入收集，确保数据收集过程中视点的一致性。我们还开发了一个轻量级基线模型 NavAff，该模型在 MoMa-Kitchen 基准测试中表现出色，用于导航功能 grounding。我们的方法使模型能够学习基于功能的最终定位，以适应不同的手臂类型和平台高度，从而为导航和 manipulation 在 embodied AI 中更稳健和可泛化的集成铺平了道路。项目页面: \href{这个链接}{这个链接}。 

---
# Low-cost Real-world Implementation of the Swing-up Pendulum for Deep Reinforcement Learning Experiments 

**Title (ZH)**: 低成本实际实施的摆动起立摆系统用于深度强化学习实验 

**Authors**: Peter Böhm, Pauline Pounds, Archie C. Chapman  

**Link**: [PDF](https://arxiv.org/pdf/2503.11065)  

**Abstract**: Deep reinforcement learning (DRL) has had success in virtual and simulated domains, but due to key differences between simulated and real-world environments, DRL-trained policies have had limited success in real-world applications. To assist researchers to bridge the \textit{sim-to-real gap}, in this paper, we describe a low-cost physical inverted pendulum apparatus and software environment for exploring sim-to-real DRL methods. In particular, the design of our apparatus enables detailed examination of the delays that arise in physical systems when sensing, communicating, learning, inferring and actuating. Moreover, we wish to improve access to educational systems, so our apparatus uses readily available materials and parts to reduce cost and logistical barriers. Our design shows how commercial, off-the-shelf electronics and electromechanical and sensor systems, combined with common metal extrusions, dowel and 3D printed couplings provide a pathway for affordable physical DRL apparatus. The physical apparatus is complemented with a simulated environment implemented using a high-fidelity physics engine and OpenAI Gym interface. 

**Abstract (ZH)**: 深强化学习（DRL）在虚拟和模拟领域取得了成功，但由于模拟环境与实际环境之间的关键差异，DRL训练的策略在实际应用中的效果有限。为了协助研究人员弥合“仿真实到实际”差距，本文描述了一种低成本的物理倒立摆装置及其软件环境，以探索仿真实到实际的DRL方法。该装置的设计允许对物理系统中感应、通信、学习、推理和执行过程中产生的延迟进行详细研究。此外，为了提高教育系统的可访问性，该装置使用易获取的材料和部件来降低成本和物流障碍。我们的设计展示了如何通过将商用现成的电子、机电和传感器系统与常见的金属型材、杆和3D打印接头结合，提供一种经济实惠的物理DRL装置的路径。该物理装置与使用高保真物理引擎和OpenAI Gym接口实现的模拟环境相辅相成。 

---
# Training Directional Locomotion for Quadrupedal Low-Cost Robotic Systems via Deep Reinforcement Learning 

**Title (ZH)**: 基于深度强化学习的低-cost 四足机器人方向性运动训练 

**Authors**: Peter Böhm, Archie C. Chapman, Pauline Pounds  

**Link**: [PDF](https://arxiv.org/pdf/2503.11059)  

**Abstract**: In this work we present Deep Reinforcement Learning (DRL) training of directional locomotion for low-cost quadrupedal robots in the real world. In particular, we exploit randomization of heading that the robot must follow to foster exploration of action-state transitions most useful for learning both forward locomotion as well as course adjustments. Changing the heading in episode resets to current yaw plus a random value drawn from a normal distribution yields policies able to follow complex trajectories involving frequent turns in both directions as well as long straight-line stretches. By repeatedly changing the heading, this method keeps the robot moving within the training platform and thus reduces human involvement and need for manual resets during the training. Real world experiments on a custom-built, low-cost quadruped demonstrate the efficacy of our method with the robot successfully navigating all validation tests. When trained with other approaches, the robot only succeeds in forward locomotion test and fails when turning is required. 

**Abstract (ZH)**: 本研究展示了在真实世界中，通过深度强化学习（DRL）训练低成本四足机器人方向性行进的实际应用。特别地，我们通过随机化机器人必须跟随的方向来促进对最有利于学习前行行进及路径调整的动作-状态转换的探索。通过在每个新回合重置时改变方向，使方向成为当前偏航角加上从正态分布中随机抽取的值，从而生成能够跟随复杂轨迹（包括频繁的双向转弯以及长直线段）的策略。通过反复改变方向，这种方法使机器人能够在训练平台上保持移动，从而减少人类干预和手动重置的需求。在自 built 的低成本四足机器人上进行的实际实验表明，本方法的有效性，机器人成功通过了所有验证测试。当使用其他方法进行训练时，机器人只能成功完成前行测试，而转弯时会失败。 

---
# Distance-Based Tree-Sliced Wasserstein Distance 

**Title (ZH)**: 基于距离的树剖分Wasserstein距离 

**Authors**: Hoang V. Tran, Khoi N.M. Nguyen, Trang Pham, Thanh T. Chu, Tam Le, Tan M. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11050)  

**Abstract**: To overcome computational challenges of Optimal Transport (OT), several variants of Sliced Wasserstein (SW) has been developed in the literature. These approaches exploit the closed-form expression of the univariate OT by projecting measures onto (one-dimensional) lines. However, projecting measures onto low-dimensional spaces can lead to a loss of topological information. Tree-Sliced Wasserstein distance on Systems of Lines (TSW-SL) has emerged as a promising alternative that replaces these lines with a more advanced structure called tree systems. The tree structures enhance the ability to capture topological information of the metric while preserving computational efficiency. However, at the core of TSW-SL, the splitting maps, which serve as the mechanism for pushing forward measures onto tree systems, focus solely on the position of the measure supports while disregarding the projecting domains. Moreover, the specific splitting map used in TSW-SL leads to a metric that is not invariant under Euclidean transformations, a typically expected property for OT on Euclidean space. In this work, we propose a novel class of splitting maps that generalizes the existing one studied in TSW-SL enabling the use of all positional information from input measures, resulting in a novel Distance-based Tree-Sliced Wasserstein (Db-TSW) distance. In addition, we introduce a simple tree sampling process better suited for Db-TSW, leading to an efficient GPU-friendly implementation for tree systems, similar to the original SW. We also provide a comprehensive theoretical analysis of proposed class of splitting maps to verify the injectivity of the corresponding Radon Transform, and demonstrate that Db-TSW is an Euclidean invariant metric. We empirically show that Db-TSW significantly improves accuracy compared to recent SW variants while maintaining low computational cost via a wide range of experiments. 

**Abstract (ZH)**: 克服最优传输计算挑战的新型基于树结构的切片 Wasserstein 距离 

---
# Measuring Similarity in Causal Graphs: A Framework for Semantic and Structural Analysis 

**Title (ZH)**: 因果图中相似性测量：一种语义和结构分析框架 

**Authors**: Ning-Yuan Georgia Liu, Flower Yang, Mohammad S. Jalali  

**Link**: [PDF](https://arxiv.org/pdf/2503.11046)  

**Abstract**: Causal graphs are commonly used to understand and model complex systems. Researchers often construct these graphs from different perspectives, leading to significant variations for the same problem. Comparing causal graphs is, therefore, essential for evaluating assumptions, integrating insights, and resolving disagreements. The rise of AI tools has further amplified this need, as they are increasingly used to generate hypothesized causal graphs by synthesizing information from various sources such as prior research and community inputs, providing the potential for automating and scaling causal modeling for complex systems. Similar to humans, these tools also produce inconsistent results across platforms, versions, and iterations. Despite its importance, research on causal graph comparison remains scarce. Existing methods often focus solely on structural similarities, assuming identical variable names, and fail to capture nuanced semantic relationships, which is essential for causal graph comparison. We address these gaps by investigating methods for comparing causal graphs from both semantic and structural perspectives. First, we reviewed over 40 existing metrics and, based on predefined criteria, selected nine for evaluation from two threads of machine learning: four semantic similarity metrics and five learning graph kernels. We discuss the usability of these metrics in simple examples to illustrate their strengths and limitations. We then generated a synthetic dataset of 2,000 causal graphs using generative AI based on a reference diagram. Our findings reveal that each metric captures a different aspect of similarity, highlighting the need to use multiple metrics. 

**Abstract (ZH)**: 因果图常用于理解与建模复杂系统。研究人员从不同视角构建这些图，导致同一问题存在显著差异。因此，比较因果图对于评估假设、整合洞察和解决分歧至关重要。随着AI工具的应用日益增加，这些工具通过综合多种来源的信息（如先前研究和社区输入）生成假设的因果图，从而为复杂系统自动化和扩展因果建模提供了可能性。与人类相似，这些工具在不同平台、版本和迭代中也产生不一致的结果。尽管这一点很重要，但关于因果图比较的研究仍相对稀缺。现有方法往往仅关注结构相似性，假设变量名称相同，未能捕捉到因果图比较中必要的细腻语义关系。我们通过从语义和结构两个视角调查比较方法来填补这些空白。我们首先回顾了超过40个现有指标，并根据预定义的标准从中选择了九个进行评估，这些指标分别来自机器学习的两个分支：四个语义相似性指标和五个学习图核。我们通过简单示例讨论这些指标的适用性，以展示其优点和局限性。然后，我们基于一个参考图使用生成AI生成了2000个合成因果图数据集。我们的研究发现，每个指标都捕捉了相似性的一个不同方面，突显了使用多种指标的必要性。 

---
# Fourier Neural Operator based surrogates for $CO_2$ storage in realistic geologies 

**Title (ZH)**: 基于傅里叶神经算子的二氧化碳储存在实际地质结构中的代理模型 

**Authors**: Anirban Chandra, Marius Koch, Suraj Pawar, Aniruddha Panda, Kamyar Azizzadenesheli, Jeroen Snippe, Faruk O. Alpak, Farah Hariri, Clement Etienam, Pandu Devarakota, Anima Anandkumar, Detlef Hohl  

**Link**: [PDF](https://arxiv.org/pdf/2503.11031)  

**Abstract**: This study aims to develop surrogate models for accelerating decision making processes associated with carbon capture and storage (CCS) technologies. Selection of sub-surface $CO_2$ storage sites often necessitates expensive and involved simulations of $CO_2$ flow fields. Here, we develop a Fourier Neural Operator (FNO) based model for real-time, high-resolution simulation of $CO_2$ plume migration. The model is trained on a comprehensive dataset generated from realistic subsurface parameters and offers $O(10^5)$ computational acceleration with minimal sacrifice in prediction accuracy. We also explore super-resolution experiments to improve the computational cost of training the FNO based models. Additionally, we present various strategies for improving the reliability of predictions from the model, which is crucial while assessing actual geological sites. This novel framework, based on NVIDIA's Modulus library, will allow rapid screening of sites for CCS. The discussed workflows and strategies can be applied to other energy solutions like geothermal reservoir modeling and hydrogen storage. Our work scales scientific machine learning models to realistic 3D systems that are more consistent with real-life subsurface aquifers/reservoirs, paving the way for next-generation digital twins for subsurface CCS applications. 

**Abstract (ZH)**: 基于Fourier神经算子的代理模型开发以加速碳捕获与封存技术相关的决策过程 

---
# FMNet: Frequency-Assisted Mamba-Like Linear Attention Network for Camouflaged Object Detection 

**Title (ZH)**: FMNet: 频率辅助的类似Mamba的线性注意力网络用于伪装目标检测 

**Authors**: Ming Deng, Sijin Sun, Zihao Li, Xiaochuan Hu, Xing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11030)  

**Abstract**: Camouflaged Object Detection (COD) is challenging due to the strong similarity between camouflaged objects and their surroundings, which complicates identification. Existing methods mainly rely on spatial local features, failing to capture global information, while Transformers increase computational this http URL address this, the Frequency-Assisted Mamba-Like Linear Attention Network (FMNet) is proposed, which leverages frequency-domain learning to efficiently capture global features and mitigate ambiguity between objects and the background. FMNet introduces the Multi-Scale Frequency-Assisted Mamba-Like Linear Attention (MFM) module, integrating frequency and spatial features through a multi-scale structure to handle scale variations while reducing computational complexity. Additionally, the Pyramidal Frequency Attention Extraction (PFAE) module and the Frequency Reverse Decoder (FRD) enhance semantics and reconstruct features. Experimental results demonstrate that FMNet outperforms existing methods on multiple COD datasets, showcasing its advantages in both performance and efficiency. Code available at this https URL. 

**Abstract (ZH)**: 伪装目标检测（COD）由于伪装目标与其环境之间的强烈相似性而具有挑战性，这增加了识别的复杂性。现有方法主要依赖于空间局部特征，无法捕捉全局信息，而Transformer提高计算复杂度对此有所改善。为了解决这一问题，提出了一种频率辅助的类似Mamba的线性注意力网络（FMNet），它利用频域学习有效地捕获全局特征并减轻伪装目标与背景之间的模糊性。FMNet引入了多尺度频率辅助的类似Mamba的线性注意力（MFM）模块，通过多尺度结构整合频率和空间特征，以处理尺度变化并降低计算复杂度。此外，金字塔频域注意提取（PFAE）模块和频率逆向解码器（FRD）增强了语义并重构特征。实验结果表明，FMNet在多个伪装目标检测数据集上优于现有方法，展示了其在性能和效率方面的优势。代码可在以下网址获取。 

---
# From Abstraction to Reality: DARPA's Vision for Robust Sim-to-Real Autonomy 

**Title (ZH)**: 从抽象到现实： DARPA关于稳健的仿真到现实自主性的愿景 

**Authors**: Erfaun Noorani, Zachary Serlin, Ben Price, Alvaro Velasquez  

**Link**: [PDF](https://arxiv.org/pdf/2503.11007)  

**Abstract**: The DARPA Transfer from Imprecise and Abstract Models to Autonomous Technologies (TIAMAT) program aims to address rapid and robust transfer of autonomy technologies across dynamic and complex environments, goals, and platforms. Existing methods for simulation-to-reality (sim-to-real) transfer often rely on high-fidelity simulations and struggle with broad adaptation, particularly in time-sensitive scenarios. Although many approaches have shown incredible performance at specific tasks, most techniques fall short when posed with unforeseen, complex, and dynamic real-world scenarios due to the inherent limitations of simulation. In contrast to current research that aims to bridge the gap between simulation environments and the real world through increasingly sophisticated simulations and a combination of methods typically assuming a small sim-to-real gap -- such as domain randomization, domain adaptation, imitation learning, meta-learning, policy distillation, and dynamic optimization -- TIAMAT takes a different approach by instead emphasizing transfer and adaptation of the autonomy stack directly to real-world environments by utilizing a breadth of low(er)-fidelity simulations to create broadly effective sim-to-real transfers. By abstractly learning from multiple simulation environments in reference to their shared semantics, TIAMAT's approaches aim to achieve abstract-to-real transfer for effective and rapid real-world adaptation. Furthermore, this program endeavors to improve the overall autonomy pipeline by addressing the inherent challenges in translating simulated behaviors into effective real-world performance. 

**Abstract (ZH)**: DARPA从不精确和抽象模型向自主技术的转移（TIAMAT）计划旨在解决自主技术在动态和复杂环境、目标和平台之间的快速和稳健转移问题。现有的模拟到现实（sim-to-real）转移方法往往依赖于高保真模拟，并且在广泛的适应性方面存在困难，尤其是在时间敏感的场景中。尽管许多方法在特定任务上表现出了惊人的性能，但在遇到不可预见的、复杂的和动态的实际世界场景时，大多数技术由于模拟固有的局限性而表现不佳。与当前致力于通过越来越复杂的模拟和假设较小的sim-to-real差距组合方法来弥补模拟环境与现实世界之间差距的研究不同，TIAMAT采取了不同的方法，而是直接通过广泛应用较低保真度的模拟来强调将自主栈转移和适应到实际环境，以实现广泛有效的模拟到现实的转移。通过从多个具有共享语义的模拟环境中抽象学习，TIAMAT的方法旨在实现从抽象到现实的转移，以实现有效的快速实际世界适应。此外，该项目还致力于通过解决将模拟行为有效转化为实际世界性能的基本挑战来改进整体自主技术管道。 

---
# Observation-Graph Interaction and Key-Detail Guidance for Vision and Language Navigation 

**Title (ZH)**: 观察图交互与关键细节指导下的视觉语言导航 

**Authors**: Yifan Xie, Binkai Ou, Fei Ma, Yaohua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11006)  

**Abstract**: Vision and Language Navigation (VLN) requires an agent to navigate through environments following natural language instructions. However, existing methods often struggle with effectively integrating visual observations and instruction details during navigation, leading to suboptimal path planning and limited success rates. In this paper, we propose OIKG (Observation-graph Interaction and Key-detail Guidance), a novel framework that addresses these limitations through two key components: (1) an observation-graph interaction module that decouples angular and visual information while strengthening edge representations in the navigation space, and (2) a key-detail guidance module that dynamically extracts and utilizes fine-grained location and object information from instructions. By enabling more precise cross-modal alignment and dynamic instruction interpretation, our approach significantly improves the agent's ability to follow complex navigation instructions. Extensive experiments on the R2R and RxR datasets demonstrate that OIKG achieves state-of-the-art performance across multiple evaluation metrics, validating the effectiveness of our method in enhancing navigation precision through better observation-instruction alignment. 

**Abstract (ZH)**: 基于视觉与语言的导航（VLN）要求代理遵循自然语言指令在环境中导航。然而，现有方法在导航过程中往往难以有效整合视觉观察和指令细节，导致路径规划效果不佳，成功率有限。本文提出了一种新颖的框架OIKG（Observation-graph Interaction and Key-detail Guidance），通过两个关键组件解决这些局限性：（1）一个观测图交互模块，解耦角度和视觉信息的同时增强导航空间中的边表示；（2）一个关键细节引导模块，动态提取和利用指令中的细粒度位置和对象信息。通过实现更精确的跨模态对齐和动态指令解释，我们的方法显著提高了代理遵循复杂导航指令的能力。在R2R和RxR数据集上的广泛实验表明，OIKG在多个评估指标上达到了最佳性能，验证了我们方法在通过更好观测-指令对齐提升导航精度方面的有效性。 

---
# RONA: Pragmatically Diverse Image Captioning with Coherence Relations 

**Title (ZH)**: RONA：具有一致性关系的实用多样化图像_captioning 

**Authors**: Aashish Anantha Ramakrishnan, Aadarsh Anantha Ramakrishnan, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.10997)  

**Abstract**: Writing Assistants (e.g., Grammarly, Microsoft Copilot) traditionally generate diverse image captions by employing syntactic and semantic variations to describe image components. However, human-written captions prioritize conveying a central message alongside visual descriptions using pragmatic cues. To enhance pragmatic diversity, it is essential to explore alternative ways of communicating these messages in conjunction with visual content. To address this challenge, we propose RONA, a novel prompting strategy for Multi-modal Large Language Models (MLLM) that leverages Coherence Relations as an axis for variation. We demonstrate that RONA generates captions with better overall diversity and ground-truth alignment, compared to MLLM baselines across multiple domains. Our code is available at: this https URL 

**Abstract (ZH)**: 写作助手（例如Grammarly、Microsoft Copilot）传统上通过语法和语义变化生成多样的图像描述。然而，人类撰写的描述更侧重于通过 pragmatics 提示传达中心信息并结合视觉描述。为了增强 pragmatics 多样性，有必要探索与视觉内容相结合的替代沟通方式。为应对这一挑战，我们提出了一种名为 RONA 的新型 Multi-modal 大型语言模型 (MLLM) 激励策略，该策略利用一致性关系作为变化轴。我们证明，与 MLLM 基线相比，RONA 生成的描述在多个领域具有更好的总体多样性和真实度匹配。我们的代码可在以下链接获取：this https URL。 

---
# Image-Goal Navigation Using Refined Feature Guidance and Scene Graph Enhancement 

**Title (ZH)**: 基于精炼特征引导和场景图增强的目标导向导航 

**Authors**: Zhicheng Feng, Xieyuanli Chen, Chenghao Shi, Lun Luo, Zhichao Chen, Yun-Hui Liu, Huimin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10986)  

**Abstract**: In this paper, we introduce a novel image-goal navigation approach, named RFSG. Our focus lies in leveraging the fine-grained connections between goals, observations, and the environment within limited image data, all the while keeping the navigation architecture simple and lightweight. To this end, we propose the spatial-channel attention mechanism, enabling the network to learn the importance of multi-dimensional features to fuse the goal and observation features. In addition, a selfdistillation mechanism is incorporated to further enhance the feature representation capabilities. Given that the navigation task needs surrounding environmental information for more efficient navigation, we propose an image scene graph to establish feature associations at both the image and object levels, effectively encoding the surrounding scene information. Crossscene performance validation was conducted on the Gibson and HM3D datasets, and the proposed method achieved stateof-the-art results among mainstream methods, with a speed of up to 53.5 frames per second on an RTX3080. This contributes to the realization of end-to-end image-goal navigation in realworld scenarios. The implementation and model of our method have been released at: this https URL. 

**Abstract (ZH)**: 基于RFSG的空间注意力机制及其在图像目标导航中的应用 

---
# The Problem of the Priors, or Posteriors? 

**Title (ZH)**: 先验难题，还是后验难题？ 

**Authors**: Hanti Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.10984)  

**Abstract**: The problem of the priors is well known: it concerns the challenge of identifying norms that govern one's prior credences. I argue that a key to addressing this problem lies in considering what I call the problem of the posteriors -- the challenge of identifying norms that directly govern one's posterior credences, which then induce constraints on the priors via the diachronic requirement of conditionalization. This forward-looking approach can be summarized as: Think ahead, work backward. Although this idea can be traced to Freedman (1963), Carnap (1963), and Shimony (1970), it has received little attention in philosophy. In this paper, I initiate a systematic defense of forward-looking Bayesianism, addressing potential objections from more traditional views (both subjectivist and objectivist) and arguing for its advantages. In particular, I develop a specific approach to forward-looking Bayesianism -- one that treats the convergence of posterior credences to the truth as a fundamental rather than derived normative requirement. This approach, called convergentist Bayesianism, is argued to be crucial for a Bayesian foundation of Ockham's razor and related inference methods in statistics and machine learning. 

**Abstract (ZH)**: Bayesianism向前看：规范后验信念的问题及其解决方案 

---
# OuroMamba: A Data-Free Quantization Framework for Vision Mamba Models 

**Title (ZH)**: OuroMamba：一种无需数据的视觉Mamba模型量化框架 

**Authors**: Akshat Ramachandran, Mingyu Lee, Huan Xu, Souvik Kundu, Tushar Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2503.10959)  

**Abstract**: We present OuroMamba, the first data-free post-training quantization (DFQ) method for vision Mamba-based models (VMMs). We identify two key challenges in enabling DFQ for VMMs, (1) VMM's recurrent state transitions restricts capturing of long-range interactions and leads to semantically weak synthetic data, (2) VMM activations exhibit dynamic outlier variations across time-steps, rendering existing static PTQ techniques ineffective. To address these challenges, OuroMamba presents a two-stage framework: (1) OuroMamba-Gen to generate semantically rich and meaningful synthetic data. It applies contrastive learning on patch level VMM features generated through neighborhood interactions in the latent state space, (2) OuroMamba-Quant to employ mixed-precision quantization with lightweight dynamic outlier detection during inference. In specific, we present a thresholding based outlier channel selection strategy for activations that gets updated every time-step. Extensive experiments across vision and generative tasks show that our data-free OuroMamba surpasses existing data-driven PTQ techniques, achieving state-of-the-art performance across diverse quantization settings. Additionally, we implement efficient GPU kernels to achieve practical latency speedup of up to 2.36x. Code will be released soon. 

**Abstract (ZH)**: OuroMamba：基于视觉Mamba模型的数据无关后训练量化方法 

---
# Predicting Stock Movement with BERTweet and Transformers 

**Title (ZH)**: 使用BERTweet和变换器预测股票变动 

**Authors**: Michael Charles Albada, Mojolaoluwa Joshua Sonola  

**Link**: [PDF](https://arxiv.org/pdf/2503.10957)  

**Abstract**: Applying deep learning and computational intelligence to finance has been a popular area of applied research, both within academia and industry, and continues to attract active attention. The inherently high volatility and non-stationary of the data pose substantial challenges to machine learning models, especially so for today's expressive and highly-parameterized deep learning models. Recent work has combined natural language processing on data from social media to augment models based purely on historic price data to improve performance has received particular attention. Previous work has achieved state-of-the-art performance on this task by combining techniques such as bidirectional GRUs, variational autoencoders, word and document embeddings, self-attention, graph attention, and adversarial training. In this paper, we demonstrated the efficacy of BERTweet, a variant of BERT pre-trained specifically on a Twitter corpus, and the transformer architecture by achieving competitive performance with the existing literature and setting a new baseline for Matthews Correlation Coefficient on the Stocknet dataset without auxiliary data sources. 

**Abstract (ZH)**: 将深度学习和计算智能应用于金融一直是学术界和工业界广泛应用的研究领域，仍持续吸引广泛关注。数据的内在高波动性和非平稳性给机器学习模型带来了重大挑战，尤其是对于当今表达能力强、参数量大的深度学习模型。最近的研究结合社交媒体数据的自然语言处理，以增强仅基于历史价格数据的模型，以改善性能，尤其受到关注。先前的工作通过结合双向GRUs、变分自编码器、词和文档嵌入、自我注意力、图注意力和对抗训练等技术，在该任务上达到了最先进的性能。在本文中，我们展示了BERTweet的效果，这是一种专门在Twitter语料上预训练的BERT变体和变压器架构，在Stocknet数据集上不使用辅助数据源的情况下，实现了与现有文献相当的性能，并为Matthews相关系数设定了新的基准。 

---
# Empirical Computation 

**Title (ZH)**: 实证计算 

**Authors**: Eric Tang, Marcel Böhme  

**Link**: [PDF](https://arxiv.org/pdf/2503.10954)  

**Abstract**: In this vision paper, we explore the challenges and opportunities of a form of computation that employs an empirical (rather than a formal) approach, where the solution of a computational problem is returned as empirically most likely (rather than necessarily correct). We call this approach as *empirical computation* and observe that its capabilities and limits *cannot* be understood within the classic, rationalist framework of computation.
While we take a very broad view of "computational problem", a classic, well-studied example is *sorting*: Given a set of $n$ numbers, return these numbers sorted in ascending order.
* To run a classical, *formal computation*, we might first think about a *specific algorithm* (e.g., merge sort) before developing a *specific* program that implements it. The program will expect the input to be given in a *specific* format, type, or data structure (e.g., unsigned 32-bit integers). In software engineering, we have many approaches to analyze the correctness of such programs. From complexity theory, we know that there exists no correct program that can solve the average instance of the sorting problem faster than $O(n\log n)$.
* To run an *empirical computation*, we might directly ask a large language model (LLM) to solve *any* computational problem (which can be stated informally in natural language) and provide the input in *any* format (e.g., negative numbers written as Chinese characters). There is no (problem-specific) program that could be analyzed for correctness. Also, the time it takes an LLM to return an answer is entirely *independent* of the computational complexity of the problem that is solved.
What are the capabilities or limits of empirical computation in the general, in the problem-, or in the instance-specific? Our purpose is to establish empirical computation as a field in SE that is timely and rich with interesting problems. 

**Abstract (ZH)**: 基于经验的计算：挑战与机遇 

---
# Safe Continual Domain Adaptation after Sim2Real Transfer of Reinforcement Learning Policies in Robotics 

**Title (ZH)**: 机器人学中强化学习策略Sim2Real迁移后的安全持续领域适应 

**Authors**: Josip Josifovski, Shangding Gu, Mohammadhossein Malmir, Haoliang Huang, Sayantan Auddy, Nicolás Navarro-Guerrero, Costas Spanos, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.10949)  

**Abstract**: Domain randomization has emerged as a fundamental technique in reinforcement learning (RL) to facilitate the transfer of policies from simulation to real-world robotic applications. Many existing domain randomization approaches have been proposed to improve robustness and sim2real transfer. These approaches rely on wide randomization ranges to compensate for the unknown actual system parameters, leading to robust but inefficient real-world policies. In addition, the policies pretrained in the domain-randomized simulation are fixed after deployment due to the inherent instability of the optimization processes based on RL and the necessity of sampling exploitative but potentially unsafe actions on the real system. This limits the adaptability of the deployed policy to the inevitably changing system parameters or environment dynamics over time. We leverage safe RL and continual learning under domain-randomized simulation to address these limitations and enable safe deployment-time policy adaptation in real-world robot control. The experiments show that our method enables the policy to adapt and fit to the current domain distribution and environment dynamics of the real system while minimizing safety risks and avoiding issues like catastrophic forgetting of the general policy found in randomized simulation during the pretraining phase. Videos and supplementary material are available at this https URL. 

**Abstract (ZH)**: 域随机化已成为强化学习中将策略从模拟环境迁移到实际机器人应用的一项基本技术。许多现有的域随机化方法被提出以提高鲁棒性和仿2实迁移。这些方法依赖广泛的随机化范围来补偿未知的实际系统参数，导致在实际应用中表现为稳健但效率低下的策略。此外，由于基于强化学习的优化过程固有的不稳定性以及在实际系统上必须采样具有潜在风险但可能有益的动作，预训练在域随机化模拟中的策略在部署后会被固定。这限制了部署策略对随时间变化的系统参数或环境动态的适应能力。我们通过在域随机化模拟中利用安全强化学习和持续学习来解决这些局限性，从而实现实际机器人控制中部署时策略的适应性改进。实验表明，我们的方法使策略能够适应并匹配实际系统当前的域分布和环境动态，同时最小化安全风险，并避免了在预训练阶段在随机化模拟中发现的泛化策略灾难性遗忘问题。相关视频和补充材料请访问此链接。 

---
# $(\varepsilon, δ)$ Considered Harmful: Best Practices for Reporting Differential Privacy Guarantees 

**Title (ZH)**: $(\varepsilon, \delta)$ 考虑有害：关于报告差分隐私保证的最佳实践 

**Authors**: Juan Felipe Gomez, Bogdan Kulynych, Georgios Kaissis, Jamie Hayes, Borja Balle, Antti Honkela  

**Link**: [PDF](https://arxiv.org/pdf/2503.10945)  

**Abstract**: Current practices for reporting the level of differential privacy (DP) guarantees for machine learning (ML) algorithms provide an incomplete and potentially misleading picture of the guarantees and make it difficult to compare privacy levels across different settings. We argue for using Gaussian differential privacy (GDP) as the primary means of communicating DP guarantees in ML, with the full privacy profile as a secondary option in case GDP is too inaccurate. Unlike other widely used alternatives, GDP has only one parameter, which ensures easy comparability of guarantees, and it can accurately capture the full privacy profile of many important ML applications. To support our claims, we investigate the privacy profiles of state-of-the-art DP large-scale image classification, and the TopDown algorithm for the U.S. Decennial Census, observing that GDP fits the profiles remarkably well in all three cases. Although GDP is ideal for reporting the final guarantees, other formalisms (e.g., privacy loss random variables) are needed for accurate privacy accounting. We show that such intermediate representations can be efficiently converted to GDP with minimal loss in tightness. 

**Abstract (ZH)**: 当前用于报告机器学习算法差分隐私（DP）保证级别的实践提供了一个不完整且可能具有误导性的图片，使其难以在不同环境中比较隐私水平。我们主张使用高斯差分隐私（GDP）作为主要的隐私保证沟通方式，并在GDP不够准确时将完整的隐私轮廓作为次要选择。与其它广泛使用的替代方案不同，GDP只有一个参数，这保证了保证的易于比较，并能准确捕捉许多重要机器学习应用的完整隐私轮廓。为了支持我们的观点，我们调查了最先进的DP大型图像分类和美国十年人口普查的TopDown算法的隐私轮廓，观察到在所有三种情况下，GDP都能非常 fitting。尽管GDP最适合报告最终保证，但为了精确的隐私核算，仍需要其他形式化表示（例如，隐私损失随机变量）。我们显示，这些中间表示可以通过最小的精确度损失高效地转换为GDP。 

---
# ChatGPT Encounters Morphing Attack Detection: Zero-Shot MAD with Multi-Modal Large Language Models and General Vision Models 

**Title (ZH)**: ChatGPT 面对形态变化攻击检测：基于多模态大型语言模型和通用视觉模型的零样本MAD 

**Authors**: Haoyu Zhang, Raghavendra Ramachandra, Kiran Raja, Christoph Busch  

**Link**: [PDF](https://arxiv.org/pdf/2503.10937)  

**Abstract**: Face Recognition Systems (FRS) are increasingly vulnerable to face-morphing attacks, prompting the development of Morphing Attack Detection (MAD) algorithms. However, a key challenge in MAD lies in its limited generalizability to unseen data and its lack of explainability-critical for practical application environments such as enrolment stations and automated border control systems. Recognizing that most existing MAD algorithms rely on supervised learning paradigms, this work explores a novel approach to MAD using zero-shot learning leveraged on Large Language Models (LLMs). We propose two types of zero-shot MAD algorithms: one leveraging general vision models and the other utilizing multimodal LLMs. For general vision models, we address the MAD task by computing the mean support embedding of an independent support set without using morphed images. For the LLM-based approach, we employ the state-of-the-art GPT-4 Turbo API with carefully crafted prompts. To evaluate the feasibility of zero-shot MAD and the effectiveness of the proposed methods, we constructed a print-scan morph dataset featuring various unseen morphing algorithms, simulating challenging real-world application scenarios. Experimental results demonstrated notable detection accuracy, validating the applicability of zero-shot learning for MAD tasks. Additionally, our investigation into LLM-based MAD revealed that multimodal LLMs, such as ChatGPT, exhibit remarkable generalizability to untrained MAD tasks. Furthermore, they possess a unique ability to provide explanations and guidance, which can enhance transparency and usability for end-users in practical applications. 

**Abstract (ZH)**: Face Recognition Systems (FRS) 的可信度日益受到面部变形攻击的威胁，推动了 Morphing Attack Detection (MAD) 算法的发展。然而，MAD 在应对未见数据时的泛化能力和缺乏解释性是其在诸如注册站和自动化边境控制系统等实际应用环境中的关键挑战。鉴于大多数现有的 MAD 算法依赖于监督学习范式，本文探讨了一种利用大规模语言模型 (LLMs) 的零样本学习方法来解决 MAD 问题。我们提出了两种类型的零样本 MAD 算法：一种利用通用视觉模型，另一种利用多模态 LLM。对于通用视觉模型，我们通过计算独立支撑集的平均支撑嵌入来解决 MAD 任务，而不使用变形图像。对于基于 LLM 的方法，我们采用了最先进的 GPT-4 Turbo API 并精心设计了提示。为了评估零样本 MAD 的可行性和所提方法的有效性，我们构建了一个包含多种未见过的变形算法的打印扫描变形数据集，模拟了具有挑战性的实际应用场景。实验结果表明，零样本学习在 MAD 任务中的检测精度显著，验证了其在 MAD 任务中的适用性。此外，我们对基于 LLM 的 MAD 的研究显示，多模态 LLM，如 ChatGPT，展现出对未训练 MAD 任务的显著泛化能力，同时还具有提供解释和指导的独特能力，这可以增强实际应用场景中的透明度和易用性。 

---
# OASST-ETC Dataset: Alignment Signals from Eye-tracking Analysis of LLM Responses 

**Title (ZH)**: OASST-ETC 数据集：基于大规模语言模型响应注视跟踪分析的对齐信号 

**Authors**: Angela Lopez-Cardona, Sebastian Idesis, Miguel Barreda-Ángeles, Sergi Abadal, Ioannis Arapakis  

**Link**: [PDF](https://arxiv.org/pdf/2503.10927)  

**Abstract**: While Large Language Models (LLMs) have significantly advanced natural language processing, aligning them with human preferences remains an open challenge. Although current alignment methods rely primarily on explicit feedback, eye-tracking (ET) data offers insights into real-time cognitive processing during reading. In this paper, we present OASST-ETC, a novel eye-tracking corpus capturing reading patterns from 24 participants, while evaluating LLM-generated responses from the OASST1 dataset. Our analysis reveals distinct reading patterns between preferred and non-preferred responses, which we compare with synthetic eye-tracking data. Furthermore, we examine the correlation between human reading measures and attention patterns from various transformer-based models, discovering stronger correlations in preferred responses. This work introduces a unique resource for studying human cognitive processing in LLM evaluation and suggests promising directions for incorporating eye-tracking data into alignment methods. The dataset and analysis code are publicly available. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）在自然语言处理方面取得了显著进展，但将其与人类偏好对齐仍是一个开放的挑战。尽管当前对齐方法主要依赖显式反馈，但眼动追踪（ET）数据提供了关于阅读期间实时认知处理的见解。本文介绍了OASST-ETC，这是一个新颖的眼动追踪语料库，它从24名参与者那里捕获了对OASST1数据集生成的响应的阅读模式。我们的分析揭示了偏好和非偏好响应之间不同的阅读模式，并将这些模式与合成的眼动追踪数据进行了比较。此外，我们探讨了人类阅读指标与不同变压器模型的关注模式之间的相关性，发现偏好响应中的相关性更强。本工作引入了一个独特的资源，用于研究在LLM评估中的人类认知处理，并提出了将眼动追踪数据整合到对齐方法中的有希望的方向。该数据集和分析代码已公开。 

---
# Predicting Clinical Outcomes with Waveform LSTMs 

**Title (ZH)**: 基于波形LSTMs的临床结局预测 

**Authors**: Michael Albada  

**Link**: [PDF](https://arxiv.org/pdf/2503.10925)  

**Abstract**: Data mining and machine learning hold great potential to enable health systems to systematically use data and analytics to identify inefficiencies and best practices that improve care and reduce costs. Waveform data offers particularly detailed information on how patient health evolves over time and has the potential to significantly improve prediction accuracy on multiple benchmarks, but has been widely under-utilized, largely because of the challenges in working with these large and complex datasets. This study evaluates the potential of leveraging clinical waveform data to improve prediction accuracy on a single benchmark task: the risk of mortality in the intensive care unit. We identify significant potential from this data, beating the existing baselines for both logistic regression and deep learning models. 

**Abstract (ZH)**: 数据挖掘和机器学习在系统利用数据和分析以识别提高护理质量和降低成本的效率和最佳实践方面具有巨大潜力。波形数据特别详细地展示了患者健康状况随时间的变化，并有望在多个基准上显著提高预测准确性，但这些数据因其大规模和复杂性而被广泛未充分利用。本研究评估了利用临床波形数据以提高重症监护病房死亡风险预测准确性的潜力，结果显示从该数据中存在显著潜力，超越了现有的逻辑回归和深度学习模型基准。 

---
# Resource Heterogeneity-Aware and Utilization-Enhanced Scheduling for Deep Learning Clusters 

**Title (ZH)**: 资源异质性感知与利用增强的深度学习集群调度 

**Authors**: Abeda Sultana, Nabin Pakka, Fei Xu, Xu Yuan, Li Chen, Nian-Feng Tzeng  

**Link**: [PDF](https://arxiv.org/pdf/2503.10918)  

**Abstract**: Scheduling deep learning (DL) models to train on powerful clusters with accelerators like GPUs and TPUs, presently falls short, either lacking fine-grained heterogeneity awareness or leaving resources substantially under-utilized. To fill this gap, we propose a novel design of a task-level heterogeneity-aware scheduler, {\em Hadar}, based on an optimization framework that can boost resource utilization. {\em Hadar} leverages the performance traits of DL jobs on a heterogeneous DL cluster, characterizes the task-level performance heterogeneity in the optimization problem, and makes scheduling decisions across both spatial and temporal dimensions. %with the objective to reduce the average job completion time of DL jobs. It involves the primal-dual framework employing a dual subroutine, to solve the optimization problem and guide the scheduling design. Our trace-driven simulation with representative DL model training workloads demonstrates that {\em Hadar} accelerates the total time duration by 1.20$\times$ when compared with its state-of-the-art heterogeneity-aware counterpart, Gavel. Further, our {\em Hadar} scheduler is enhanced to {\em HadarE} by forking each job into multiple copies to let a job train concurrently on heterogeneous GPUs resided on separate available nodes (i.e., machines or servers) for resource utilization enhancement. {\em HadarE} is evaluated extensively on physical DL clusters for comparison with {\em Hadar} and Gavel. With substantial enhancement in cluster resource utilization (by 1.45$\times$), {\em HadarE} exhibits considerable speed-ups in DL model training, reducing the total time duration by 50\% (or 80\%) on an Amazon's AWS (or our lab) cluster, while producing trained DL models with consistently better inference quality than those trained by \textit{Hadar}. 

**Abstract (ZH)**: 调度深度学习模型在配备GPU和TPU等加速器的高性能集群上训练，目前仍存在不足，要么缺乏细粒度的异构性感知，要么导致资源严重闲置。为弥补这一不足，我们提出了一种基于优化框架的新颖任务级异构性感知调度器设计方案——Hadar，旨在提高资源利用率。Hadar 利用异构深度学习集群中深度学习任务的性能特征，在优化问题中表征任务级性能异构性，并在空间和时间维度上做出调度决策，以减少深度学习任务的平均完成时间。该设计采用了包含对偶子程序的对偶框架，用于解决优化问题并指导调度设计。基于代表性深度学习模型训练工作负载的驱动型仿真实验显示，与最先进的异构性感知调度器Gavel相比，Hadar 能将总时间缩短1.20倍。此外，为增强Hadar调度器，我们通过为每个任务创建多个副本，使其能够在不同的可用节点（即机器或服务器）上的异构GPU上并行训练，提出了HadarE。HadarE 在物理深度学习集群上的广泛评估表明与Hadar和Gavel相比，HadarE 在集群资源利用率（1.45倍）显著提升的同时，也大幅提高了深度学习模型训练的速度，分别在亚马逊AWS或我们实验室集群中将总时间缩短了50%（或80%），并且生成的训练好的深度学习模型在推理质量方面始终优于Hadar。 

---
# JPEG Compliant Compression for Both Human and Machine, A Report 

**Title (ZH)**: 符合人类和机器的JPEG规范压缩技术报告 

**Authors**: Linfeng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.10912)  

**Abstract**: Deep Neural Networks (DNNs) have become an integral part of our daily lives, especially in vision-related applications. However, the conventional lossy image compression algorithms are primarily designed for the Human Vision System (HVS), which can non-trivially compromise the DNNs' validation accuracy after compression, as noted in \cite{liu2018deepn}. Thus developing an image compression algorithm for both human and machine (DNNs) is on the horizon.
To address the challenge mentioned above, in this paper, we first formulate the image compression as a multi-objective optimization problem which take both human and machine prespectives into account, then we solve it by linear combination, and proposed a novel distortion measure for both human and machine, dubbed Human and Machine-Oriented Error (HMOE). After that, we develop Human And Machine Oriented Soft Decision Quantization (HMOSDQ) based on HMOE, a lossy image compression algorithm for both human and machine (DNNs), and fully complied with JPEG format. In order to evaluate the performance of HMOSDQ, finally we conduct the experiments for two pre-trained well-known DNN-based image classifiers named Alexnet \cite{Alexnet} and VGG-16 \cite{simonyan2014VGG} on two subsets of the ImageNet \cite{deng2009imagenet} validation set: one subset included images with shorter side in the range of 496 to 512, while the other included images with shorter side in the range of 376 to 384. Our results demonstrate that HMOSDQ outperforms the default JPEG algorithm in terms of rate-accuracy and rate-distortion performance. For the Alexnet comparing with the default JPEG algorithm, HMOSDQ can improve the validation accuracy by more than $0.81\%$ at $0.61$ BPP, or equivalently reduce the compression rate of default JPEG by $9.6\times$ while maintaining the same validation accuracy. 

**Abstract (ZH)**: 深度神经网络（DNNs）已成为我们日常生活中不可或缺的一部分，尤其是在视觉相关应用中。然而，传统的有损图像压缩算法主要是针对人类视觉系统（HVS）设计的，这可能会在压缩后显著降低DNNs的验证准确性，如文献\[liu2018deepn\]中所述。因此，开发同时适用于人类和机器（DNNs）的图像压缩算法迫在眉睫。
为了解决上述挑战，本文首先将图像压缩问题形式化为一个多目标优化问题，同时考虑了人类和机器的视角，然后通过线性组合解决这个问题，并提出了一种适用于人类和机器的新失真度量，称为人类和机器导向误差（HMOE）。之后，基于HMOE开发了人类和机器导向软决策量化（HMOSDQ），这是一种同时适用于人类和机器（DNNs）的有损图像压缩算法，完全符合JPEG格式。为了评估HMOSDQ的性能，我们对两个预先训练好的基于DNN的图像分类器Alexnet\[Alexnet\]和VGG-16\[simonyan2014VGG\]在ImageNet\[deng2009imagenet\]验证集的两个子集中进行了实验：一个子集包含较短边在496到512范围内的图像，另一个子集包含较短边在376到384范围内的图像。实验结果表明，HMOSDQ在速率-准确性性能和速率-失真性能方面优于标准JPEG算法。与标准JPEG算法相比，在0.61 BPP时，HMOSDQ可以将验证准确性提高超过0.81%，或者等效地将标准JPEG的压缩率减少9.6倍，同时保持相同的验证准确性。 

---
# Ecological Neural Architecture Search 

**Title (ZH)**: 生态神经架构搜索 

**Authors**: Benjamin David Winter, William J. Teahan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10908)  

**Abstract**: When employing an evolutionary algorithm to optimize a neural networks architecture, developers face the added challenge of tuning the evolutionary algorithm's own hyperparameters - population size, mutation rate, cloning rate, and number of generations. This paper introduces Neuvo Ecological Neural Architecture Search (ENAS), a novel method that incorporates these evolutionary parameters directly into the candidate solutions' phenotypes, allowing them to evolve dynamically alongside architecture specifications. Experimental results across four binary classification datasets demonstrate that ENAS not only eliminates manual tuning of evolutionary parameters but also outperforms competitor NAS methodologies in convergence speed (reducing computational time by 18.3%) and accuracy (improving classification performance in 3 out of 4 datasets). By enabling "greedy individuals" to optimize resource allocation based on fitness, ENAS provides an efficient, self-regulating approach to neural architecture search. 

**Abstract (ZH)**: 基于生态进化的神经架构搜索（ENAS） 

---
# H2-MARL: Multi-Agent Reinforcement Learning for Pareto Optimality in Hospital Capacity Strain and Human Mobility during Epidemic 

**Title (ZH)**: H2-MARL: 多智能体强化学习在 Epidemic 期间医院容量压力和人类移动性帕累托最优解的研究 

**Authors**: Xueting Luo, Hao Deng, Jihong Yang, Yao Shen, Huanhuan Guo, Zhiyuan Sun, Mingqing Liu, Jiming Wei, Shengjie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.10907)  

**Abstract**: The necessity of achieving an effective balance between minimizing the losses associated with restricting human mobility and ensuring hospital capacity has gained significant attention in the aftermath of COVID-19. Reinforcement learning (RL)-based strategies for human mobility management have recently advanced in addressing the dynamic evolution of cities and epidemics; however, they still face challenges in achieving coordinated control at the township level and adapting to cities of varying scales. To address the above issues, we propose a multi-agent RL approach that achieves Pareto optimality in managing hospital capacity and human mobility (H2-MARL), applicable across cities of different scales. We first develop a township-level infection model with online-updatable parameters to simulate disease transmission and construct a city-wide dynamic spatiotemporal epidemic simulator. On this basis, H2-MARL is designed to treat each division as an agent, with a trade-off dual-objective reward function formulated and an experience replay buffer enriched with expert knowledge built. To evaluate the effectiveness of the model, we construct a township-level human mobility dataset containing over one billion records from four representative cities of varying scales. Extensive experiments demonstrate that H2-MARL has the optimal dual-objective trade-off capability, which can minimize hospital capacity strain while minimizing human mobility restriction loss. Meanwhile, the applicability of the proposed model to epidemic control in cities of varying scales is verified, which showcases its feasibility and versatility in practical applications. 

**Abstract (ZH)**: 实现减少限制人类流动性所带来的损失与保障医院容量之间有效平衡的必要性在COVID-19之后引起了广泛关注。基于强化学习（RL）的人口流动性管理策略近年来在应对城市的动态演变和疫情动态发展方面取得了进展；然而，它们在实现乡镇层面的协调控制以及适应不同规模城市方面仍面临挑战。为此，我们提出了一种多agent RL方法——H2-MARL，以实现不同规模城市中医院容量管理和人口流动的帕累托最优。我们首先开发了一个可在线更新参数的乡镇级感染模型，用于模拟疾病传播，并构建了一个覆盖全市范围的动态空间—时间疫情仿真器。在此基础上，H2-MARL 被设计为将每个分区视为一个agent，并设计了一个权衡双重目标的奖励函数，同时构建了一个丰富了专家知识的经验回放缓冲区。为了评估模型的有效性，我们构建了一个包含四个不同规模代表性城市的超过十亿条记录的乡镇级人口流动性数据集。广泛的实验表明，H2-MARL 具有最佳的双重目标权衡能力，能够在最大限度减少医院容量压力的同时，最小化对人类流动性限制的影响。此外，我们验证了所提模型在不同规模城市疫情控制中的适用性，展示了其在实际应用中的可行性和灵活性。 

---
# HyperDAS: Towards Automating Mechanistic Interpretability with Hypernetworks 

**Title (ZH)**: HyperDAS: 向基于超网络自动实现机理可解释性迈进 

**Authors**: Jiuding Sun, Jing Huang, Sidharth Baskaran, Karel D'Oosterlinck, Christopher Potts, Michael Sklar, Atticus Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2503.10894)  

**Abstract**: Mechanistic interpretability has made great strides in identifying neural network features (e.g., directions in hidden activation space) that mediate concepts(e.g., the birth year of a person) and enable predictable manipulation. Distributed alignment search (DAS) leverages supervision from counterfactual data to learn concept features within hidden states, but DAS assumes we can afford to conduct a brute force search over potential feature locations. To address this, we present HyperDAS, a transformer-based hypernetwork architecture that (1) automatically locates the token-positions of the residual stream that a concept is realized in and (2) constructs features of those residual stream vectors for the concept. In experiments with Llama3-8B, HyperDAS achieves state-of-the-art performance on the RAVEL benchmark for disentangling concepts in hidden states. In addition, we review the design decisions we made to mitigate the concern that HyperDAS (like all powerful interpretabilty methods) might inject new information into the target model rather than faithfully interpreting it. 

**Abstract (ZH)**: 机械可解释性已在识别介导概念（例如，一个人的出生年份）的神经网络特征（例如，隐藏激活空间中的方向）并实现可预测操控方面取得了显著进展。分布式对齐搜索（DAS）利用反事实数据的监督在隐状态中学习概念特征，但DAS假设我们能够负担得起对潜在特征位置进行穷举搜索的成本。为解决这一问题，我们提出了一种基于变压器的超网络架构HyperDAS，该架构能够（1）自动定位概念在残差流中实现的标记位置，并（2）构建这些残差流向量的概念特征。在Llama3-8B实验中，HyperDAS在RAVEL基准上实现了概念在隐状态中解耦的最佳性能。此外，我们还回顾了为减轻HyperDAS（就像所有强大的可解释性方法一样）可能在忠实地解释目标模型的同时注入新信息的担忧所做出的设计决策。 

---
# Taxonomic Reasoning for Rare Arthropods: Combining Dense Image Captioning and RAG for Interpretable Classification 

**Title (ZH)**: 稀有节肢动物分类学推理：结合密集图像描述和RAG进行可解释分类 

**Authors**: Nathaniel Lesperance, Sujeevan Ratnasingham, Graham W. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2503.10886)  

**Abstract**: In the context of pressing climate change challenges and the significant biodiversity loss among arthropods, automated taxonomic classification from organismal images is a subject of intense research. However, traditional AI pipelines based on deep neural visual architectures such as CNNs or ViTs face limitations such as degraded performance on the long-tail of classes and the inability to reason about their predictions. We integrate image captioning and retrieval-augmented generation (RAG) with large language models (LLMs) to enhance biodiversity monitoring, showing particular promise for characterizing rare and unknown arthropod species. While a naive Vision-Language Model (VLM) excels in classifying images of common species, the RAG model enables classification of rarer taxa by matching explicit textual descriptions of taxonomic features to contextual biodiversity text data from external sources. The RAG model shows promise in reducing overconfidence and enhancing accuracy relative to naive LLMs, suggesting its viability in capturing the nuances of taxonomic hierarchy, particularly at the challenging family and genus levels. Our findings highlight the potential for modern vision-language AI pipelines to support biodiversity conservation initiatives, emphasizing the role of comprehensive data curation and collaboration with citizen science platforms to improve species identification, unknown species characterization and ultimately inform conservation strategies. 

**Abstract (ZH)**: 在迫切的气候变化挑战和显著的昆虫多样性丧失背景下，基于生物图像的自动化分类是研究的热点。然而，传统的基于深度神经视觉架构如CNNs或ViTs的人工智能管道存在长尾类别的性能下降和无法解释其预测的问题。我们将图像描述和检索增强生成（RAG）与大型语言模型（LLMs）集成，以增强生物多样性监测，特别适用于描述稀有和未知昆虫物种。尽管一个简单的视觉语言模型（VLM）在分类常见物种的图像方面表现出色，但RAG模型通过将税务特征的显式文本描述与外部来源的背景生物多样性文本数据匹配，实现稀有类别的分类。RAG模型在降低过度自信和提高准确性方面显示出潜力，相对于简单的LLMs，它有可能捕捉到税务层次结构的细微差别，特别是在具有挑战性的家族和属级别。我们的发现强调，现代视觉语言人工智能管道有可能支持生物多样性保护倡议，强调全面数据整理和与公民科学平台合作的重要性，以提高物种识别、未知物种描述，并最终制定保护策略。 

---
# Task-Specific Activation Functions for Neuroevolution using Grammatical Evolution 

**Title (ZH)**: 用于语法进化神经进化任务特定激活函数 

**Authors**: Benjamin David Winter, William John Teahan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10879)  

**Abstract**: Activation functions play a critical role in the performance and behaviour of neural networks, significantly impacting their ability to learn and generalise. Traditional activation functions, such as ReLU, sigmoid, and tanh, have been widely used with considerable success. However, these functions may not always provide optimal performance for all tasks and datasets. In this paper, we introduce Neuvo GEAF - an innovative approach leveraging grammatical evolution (GE) to automatically evolve novel activation functions tailored to specific neural network architectures and datasets. Experiments conducted on well-known binary classification datasets show statistically significant improvements in F1-score (between 2.4% and 9.4%) over ReLU using identical network architectures. Notably, these performance gains were achieved without increasing the network's parameter count, supporting the trend toward more efficient neural networks that can operate effectively on resource-constrained edge devices. This paper's findings suggest that evolved activation functions can provide significant performance improvements for compact networks while maintaining energy efficiency during both training and inference phases. 

**Abstract (ZH)**: 神经网络性能和行为中激活函数起着关键作用，显著影响其学习和泛化能力。传统的激活函数，如ReLU、Sigmoid和tanh，已经在许多任务中取得了显著成功。然而，这些函数可能并不总是能够为所有任务和数据集提供最优性能。本文介绍了一种名为Neuvo GEAF的创新方法，该方法利用语法演化（GE）自动生成针对特定神经网络架构和数据集量身定制的新型激活函数。实验结果显示，在相同网络架构下，与ReLU相比，Neuvo GEAF在F1分数上取得了统计上显著的提升（范围为2.4%至9.4%）。值得注意的是，这些性能提升是在不增加网络参数数量的情况下实现的，支持了向更高效的神经网络发展的趋势，这些网络可以在资源受限的边缘设备上有效运行。本文的研究结果表明，演化生成的激活函数可以在保持训练和推理阶段能源效率的同时，为紧凑型网络提供显著的性能改进。 

---
# TAIJI: Textual Anchoring for Immunizing Jailbreak Images in Vision Language Models 

**Title (ZH)**: TAIJI: 文本锚定免疫视觉语言模型中的逃逸图像 

**Authors**: Xiangyu Yin, Yi Qi, Jinwei Hu, Zhen Chen, Yi Dong, Xingyu Zhao, Xiaowei Huang, Wenjie Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10872)  

**Abstract**: Vision Language Models (VLMs) have demonstrated impressive inference capabilities, but remain vulnerable to jailbreak attacks that can induce harmful or unethical responses. Existing defence methods are predominantly white-box approaches that require access to model parameters and extensive modifications, making them costly and impractical for many real-world scenarios. Although some black-box defences have been proposed, they often impose input constraints or require multiple queries, limiting their effectiveness in safety-critical tasks such as autonomous driving. To address these challenges, we propose a novel black-box defence framework called \textbf{T}extual \textbf{A}nchoring for \textbf{I}mmunizing \textbf{J}ailbreak \textbf{I}mages (\textbf{TAIJI}). TAIJI leverages key phrase-based textual anchoring to enhance the model's ability to assess and mitigate the harmful content embedded within both visual and textual prompts. Unlike existing methods, TAIJI operates effectively with a single query during inference, while preserving the VLM's performance on benign tasks. Extensive experiments demonstrate that TAIJI significantly enhances the safety and reliability of VLMs, providing a practical and efficient solution for real-world deployment. 

**Abstract (ZH)**: 基于文本锚定的免疫逃逸图像防御框架：TAIJI 

---
# Evaluating a Novel Neuroevolution and Neural Architecture Search System 

**Title (ZH)**: 评估一种新型神经进化和神经网络架构搜索系统 

**Authors**: Benjamin David Winter, William John Teahan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10869)  

**Abstract**: The choice of neural network features can have a large impact on both the accuracy and speed of the network. Despite the current industry shift towards large transformer models, specialized binary classifiers remain critical for numerous practical applications where computational efficiency and low latency are essential. Neural network features tend to be developed homogeneously, resulting in slower or less accurate networks when testing against multiple datasets. In this paper, we show the effectiveness of Neuvo NAS+ a novel Python implementation of an extended Neural Architecture Search (NAS+) which allows the user to optimise the training parameters of a network as well as the network's architecture. We provide an in-depth analysis of the importance of catering a network's architecture to each dataset. We also describe the design of the Neuvo NAS+ system that selects network features on a task-specific basis including network training hyper-parameters such as the number of epochs and batch size. Results show that the Neuvo NAS+ task-specific approach significantly outperforms several machine learning approaches such as Naive Bayes, C4.5, Support Vector Machine and a standard Artificial Neural Network for solving a range of binary classification problems in terms of accuracy. Our experiments demonstrate substantial diversity in evolved network architectures across different datasets, confirming the value of task-specific optimization. Additionally, Neuvo NAS+ outperforms other evolutionary algorithm optimisers in terms of both accuracy and computational efficiency, showing that properly optimized binary classifiers can match or exceed the performance of more complex models while requiring significantly fewer computational resources. 

**Abstract (ZH)**: 神经网络特征的选择对网络的准确性和速度有很大影响。尽管当前行业趋势是使用大型变压器模型，但针对特定任务的二元分类器在需要计算效率和低延迟的实际应用中仍然至关重要。神经网络特征通常是一致开发的，导致在多个数据集上测试时网络速度较慢或不够准确。在本文中，我们展示了Neuvo NAS+的有效性，这是一种新型的扩展神经架构搜索（NAS+）的Python实现，允许用户优化网络的训练参数和网络架构。我们深入分析了为每个数据集定制网络架构的重要性。我们还介绍了Neuvo NAS+系统的设计，该系统根据特定任务选择网络特征，包括网络训练超参数如迭代次数和批次大小。结果表明，Neuvo NAS+针对特定任务的方法在准确性和多种二元分类问题上显著优于朴素贝叶斯、C4.5、支持向量机和标准人工神经网络。我们的实验表明，在不同数据集上进化出的网络架构存在显著多样性，证实了任务特定优化的价值。此外，与其它进化算法优化器相比，Neuvo NAS+在准确性和计算效率方面表现出色，表明适当优化的二元分类器可以匹配甚至超过更复杂模型的性能，同时所需的计算资源显著减少。 

---
# Towards Understanding Graphical Perception in Large Multimodal Models 

**Title (ZH)**: 理解大型多模态模型中的图形感知 

**Authors**: Kai Zhang, Jianwei Yang, Jeevana Priya Inala, Chandan Singh, Jianfeng Gao, Yu Su, Chenglong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10857)  

**Abstract**: Despite the promising results of large multimodal models (LMMs) in complex vision-language tasks that require knowledge, reasoning, and perception abilities together, we surprisingly found that these models struggle with simple tasks on infographics that require perception only. As existing benchmarks primarily focus on end tasks that require various abilities, they provide limited, fine-grained insights into the limitations of the models' perception abilities. To address this gap, we leverage the theory of graphical perception, an approach used to study how humans decode visual information encoded on charts and graphs, to develop an evaluation framework for analyzing gaps in LMMs' perception abilities in charts. With automated task generation and response evaluation designs, our framework enables comprehensive and controlled testing of LMMs' graphical perception across diverse chart types, visual elements, and task types. We apply our framework to evaluate and diagnose the perception capabilities of state-of-the-art LMMs at three granularity levels (chart, visual element, and pixel). Our findings underscore several critical limitations of current state-of-the-art LMMs, including GPT-4o: their inability to (1) generalize across chart types, (2) understand fundamental visual elements, and (3) cross reference values within a chart. These insights provide guidance for future improvements in perception abilities of LMMs. The evaluation framework and labeled data are publicly available at this https URL. 

**Abstract (ZH)**: 尽管大规模多模态模型（LMMs）在需要知识、推理和感知能力的复杂视觉-语言任务中取得了令人鼓舞的结果，但我们惊讶地发现，这些模型在仅需感知能力的图表任务中表现不佳。由于现有基准主要关注需要多种能力的最终任务，它们只能提供有限的关于模型感知能力局限性的细微洞察。为解决这一问题，我们借鉴图形感知理论，该理论用于研究人类如何解码图表和图示中编码的视觉信息，开发了一种分析LMMs在图表中感知能力差距的评估框架。该框架采用自动任务生成和响应评估设计，能够在多种图表类型、视觉元素和任务类型的范围内进行全面且受控的LMMs图形感知测试。我们利用该框架在三个细粒度级别（图表、视觉元素和像素）上评估和诊断了当前最先进的LMMs的感知能力。我们的发现指出了当前最先进的LMMs的几个关键局限性，包括GPT-4o：它们无法（1）跨图表类型进行泛化，（2）理解基本的视觉元素，以及（3）跨参考图表内的值。这些见解为未来改进LMMs的感知能力提供了指导。评估框架和标注数据可在如下网址获取：this https URL。 

---
# Byzantine-Resilient Federated Learning via Distributed Optimization 

**Title (ZH)**: 拜占庭容错联邦学习通过分布式优化实现 

**Authors**: Yufei Xia, Wenrui Yu, Qiongxiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10792)  

**Abstract**: Byzantine attacks present a critical challenge to Federated Learning (FL), where malicious participants can disrupt the training process, degrade model accuracy, and compromise system reliability. Traditional FL frameworks typically rely on aggregation-based protocols for model updates, leaving them vulnerable to sophisticated adversarial strategies. In this paper, we demonstrate that distributed optimization offers a principled and robust alternative to aggregation-centric methods. Specifically, we show that the Primal-Dual Method of Multipliers (PDMM) inherently mitigates Byzantine impacts by leveraging its fault-tolerant consensus mechanism. Through extensive experiments on three datasets (MNIST, FashionMNIST, and Olivetti), under various attack scenarios including bit-flipping and Gaussian noise injection, we validate the superior resilience of distributed optimization protocols. Compared to traditional aggregation-centric approaches, PDMM achieves higher model utility, faster convergence, and improved stability. Our results highlight the effectiveness of distributed optimization in defending against Byzantine threats, paving the way for more secure and resilient federated learning systems. 

**Abstract (ZH)**: Byzantine 攻击对 Federated Learning 呈现关键挑战：分布式优化提供了一种原理上稳健的替代方案 

---
# Vulnerability Detection: From Formal Verification to Large Language Models and Hybrid Approaches: A Comprehensive Overview 

**Title (ZH)**: 漏洞检测：从形式验证到大型语言模型和混合方法的全面概述 

**Authors**: Norbert Tihanyi, Tamas Bisztray, Mohamed Amine Ferrag, Bilel Cherif, Richard A. Dubniczky, Ridhi Jain, Lucas C. Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2503.10784)  

**Abstract**: Software testing and verification are critical for ensuring the reliability and security of modern software systems. Traditionally, formal verification techniques, such as model checking and theorem proving, have provided rigorous frameworks for detecting bugs and vulnerabilities. However, these methods often face scalability challenges when applied to complex, real-world programs. Recently, the advent of Large Language Models (LLMs) has introduced a new paradigm for software analysis, leveraging their ability to understand insecure coding practices. Although LLMs demonstrate promising capabilities in tasks such as bug prediction and invariant generation, they lack the formal guarantees of classical methods. This paper presents a comprehensive study of state-of-the-art software testing and verification, focusing on three key approaches: classical formal methods, LLM-based analysis, and emerging hybrid techniques, which combine their strengths. We explore each approach's strengths, limitations, and practical applications, highlighting the potential of hybrid systems to address the weaknesses of standalone methods. We analyze whether integrating formal rigor with LLM-driven insights can enhance the effectiveness and scalability of software verification, exploring their viability as a pathway toward more robust and adaptive testing frameworks. 

**Abstract (ZH)**: 软件测试与验证对于确保现代软件系统的可靠性和安全性至关重要。传统形式化验证技术，如模型检测和定理证明，提供了检测错误和漏洞的严格框架。然而，当应用于复杂的现实程序时，这些方法往往面临可扩展性挑战。最近，大型语言模型（LLMs）的出现引入了软件分析的新范式，利用其理解不安全编码实践的能力。尽管LLMs在错误预测和不变式生成等任务上表现出有前景的能力，但它们缺乏经典方法的形式保证。本文对最先进的软件测试与验证进行了全面研究，重点关注三种关键方法：经典形式化方法、基于LLM的分析以及新兴的混合技术，这些技术结合了各自的优势。我们探讨了每种方法的优势、局限性和实际应用，突出了混合系统的潜力，以弥补单方法的弱点。我们分析了将形式化严谨性与LLM驱动的见解相结合是否能够增强软件验证的有效性和可扩展性，探索其作为更稳健和自适应测试框架途径的可行性。 

---
# Unifying 2D and 3D Vision-Language Understanding 

**Title (ZH)**: 统一二维和三维视觉-语言理解 

**Authors**: Ayush Jain, Alexander Swerdlow, Yuzhou Wang, Sergio Arnaud, Ada Martin, Alexander Sax, Franziska Meier, Katerina Fragkiadaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.10745)  

**Abstract**: Progress in 3D vision-language learning has been hindered by the scarcity of large-scale 3D datasets. We introduce UniVLG, a unified architecture for 2D and 3D vision-language understanding that bridges the gap between existing 2D-centric models and the rich 3D sensory data available in embodied systems. Our approach initializes most model weights from pre-trained 2D models and trains on both 2D and 3D vision-language data. We propose a novel language-conditioned mask decoder shared across 2D and 3D modalities to ground objects effectively in both RGB and RGB-D images, outperforming box-based approaches. To further reduce the domain gap between 2D and 3D, we incorporate 2D-to-3D lifting strategies, enabling UniVLG to utilize 2D data to enhance 3D performance. With these innovations, our model achieves state-of-the-art performance across multiple 3D vision-language grounding tasks, demonstrating the potential of transferring advances from 2D vision-language learning to the data-constrained 3D domain. Furthermore, co-training on both 2D and 3D data enhances performance across modalities without sacrificing 2D capabilities. By removing the reliance on 3D mesh reconstruction and ground-truth object proposals, UniVLG sets a new standard for realistic, embodied-aligned evaluation. Code and additional visualizations are available at $\href{this https URL}{this http URL}$. 

**Abstract (ZH)**: 3D视觉语言学习的进步受到大规模3D数据集稀缺性的阻碍。我们介绍了一种名为UniVLG的统一架构，它将现有以2D为中心的模型与体感系统中丰富的3D感官数据联系起来，用于2D和3D视觉语言理解。我们的方法大部分模型权重初始化来自预训练的2D模型，并在2D和3D视觉语言数据上进行训练。我们提出了一种新的语言条件掩码解码器，跨2D和3D模态共享，有效将对象地融入RGB和RGB-D图像中，优于基于框的方法。为了进一步缩小2D和3D之间的领域差距，我们引入了2D到3D提升策略，使UniVLG能够利用2D数据提升3D性能。通过这些创新，我们的模型在多个3D视觉语言接地任务中达到最先进的性能，证明了从2D视觉语言学习转移进展到数据受限的3D领域中的潜力。此外，同时在2D和3D数据上进行训练，在不牺牲2D能力的情况下提高了跨模态的性能。通过去除对3D网格重建和真实物体提案的依赖，UniVLG为现实、体感对齐的评估设定了新的标准。相关代码和附加可视化可以在[此链接](this http URL)找到。 

---
# Predicting Treatment Response in Body Dysmorphic Disorder with Interpretable Machine Learning 

**Title (ZH)**: 使用可解释机器学习预测身体 Dysmorphic 病症治疗反应 

**Authors**: Omar Costilla-Reyes, Morgan Talbot  

**Link**: [PDF](https://arxiv.org/pdf/2503.10741)  

**Abstract**: Body Dysmorphic Disorder (BDD) is a highly prevalent and frequently underdiagnosed condition characterized by persistent, intrusive preoccupations with perceived defects in physical appearance. In this extended analysis, we employ multiple machine learning approaches to predict treatment outcomes -- specifically treatment response and remission -- with an emphasis on interpretability to ensure clinical relevance and utility. Across the various models investigated, treatment credibility emerged as the most potent predictor, surpassing traditional markers such as baseline symptom severity or comorbid conditions. Notably, while simpler models (e.g., logistic regression and support vector machines) achieved competitive predictive performance, decision tree analyses provided unique insights by revealing clinically interpretable threshold values in credibility scores. These thresholds can serve as practical guideposts for clinicians when tailoring interventions or allocating treatment resources. We further contextualize our findings within the broader literature on BDD, addressing technology-based therapeutics, digital interventions, and the psychosocial determinants of treatment engagement. An extensive array of references situates our results within current research on BDD prevalence, suicidality risks, and digital innovation. Our work underscores the potential of integrating rigorous statistical methodologies with transparent machine learning models. By systematically identifying modifiable predictors -- such as treatment credibility -- we propose a pathway toward more targeted, personalized, and ultimately efficacious interventions for individuals with BDD. 

**Abstract (ZH)**: 体象障碍的治疗预后预测：基于 interpretable 机器学习方法的深入分析 

---
# Commenting Higher-level Code Unit: Full Code, Reduced Code, or Hierarchical Code Summarization 

**Title (ZH)**: 评论高级代码单元：完整代码、缩减代码或层次代码摘要 

**Authors**: Weisong Sun, Yiran Zhang, Jie Zhu, Zhihui Wang, Chunrong Fang, Yonglong Zhang, Yebo Feng, Jiangping Huang, Xingya Wang, Zhi Jin, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10737)  

**Abstract**: Commenting code is a crucial activity in software development, as it aids in facilitating future maintenance and updates. To enhance the efficiency of writing comments and reduce developers' workload, researchers has proposed various automated code summarization (ACS) techniques to automatically generate comments/summaries for given code units. However, these ACS techniques primarily focus on generating summaries for code units at the method level. There is a significant lack of research on summarizing higher-level code units, such as file-level and module-level code units, despite the fact that summaries of these higher-level code units are highly useful for quickly gaining a macro-level understanding of software components and architecture. To fill this gap, in this paper, we conduct a systematic study on how to use LLMs for commenting higher-level code units, including file level and module level. These higher-level units are significantly larger than method-level ones, which poses challenges in handling long code inputs within LLM constraints and maintaining efficiency. To address these issues, we explore various summarization strategies for ACS of higher-level code units, which can be divided into three types: full code summarization, reduced code summarization, and hierarchical code summarization. The experimental results suggest that for summarizing file-level code units, using the full code is the most effective approach, with reduced code serving as a cost-efficient alternative. However, for summarizing module-level code units, hierarchical code summarization becomes the most promising strategy. In addition, inspired by the research on method-level ACS, we also investigate using the LLM as an evaluator to evaluate the quality of summaries of higher-level code units. The experimental results demonstrate that the LLM's evaluation results strongly correlate with human evaluations. 

**Abstract (ZH)**: 使用大规模语言模型对较高层次的代码单元进行注释：方法、文件和模块级别总结的研究 

---
# OCPM$^2$: Extending the Process Mining Methodology for Object-Centric Event Data Extraction 

**Title (ZH)**: OCPM$^2$: 扩展面向对象事件数据提取的过程挖掘方法学 

**Authors**: Najmeh Miri, Shahrzad Khayatbashi, Jelena Zdravkovic, Amin Jalali  

**Link**: [PDF](https://arxiv.org/pdf/2503.10735)  

**Abstract**: Object-Centric Process Mining (OCPM) enables business process analysis from multiple perspectives. For example, an educational path can be examined from the viewpoints of students, teachers, and groups. This analysis depends on Object-Centric Event Data (OCED), which captures relationships between events and object types, representing different perspectives. Unlike traditional process mining techniques, extracting OCED minimizes the need for repeated log extractions when shifting the analytical focus. However, recording these complex relationships increases the complexity of the log extraction process. To address this challenge, this paper proposes a method for extracting OCED based on PM\inst{2}, a well-established process mining framework. Our approach introduces a structured framework that guides data analysts and engineers in extracting OCED for process analysis. We validate this framework by applying it in a real-world educational setting, demonstrating its effectiveness in extracting an Object-Centric Event Log (OCEL), which serves as the standard format for recording OCED, from a learning management system and an administrative grading system. 

**Abstract (ZH)**: 面向对象的过程挖掘（OCPM） enables 业务过程从多个视角进行分析。例如，教育路径可以从学生、教师和群体的角度进行考察。这种分析依赖于对象为中心的事件数据（OCED），它捕捉事件与对象类型之间的关系，代表不同的视角。与传统的过程挖掘技术不同，提取OCED减少了在分析焦点转移时重复日志提取的需要。然而，记录这些复杂关系增加了日志提取过程的复杂性。为应对这一挑战，本文提出了一种基于PM\inst{2}（一个成熟的过程挖掘框架）的OCED提取方法。本文的方法引入了一个结构化的框架，指导数据分析师和工程师进行OCED的提取以进行过程分析。我们通过将其应用于一个实际的教育场景，验证了该框架的有效性，展示了如何从学习管理系统和管理评分系统中提取对象为中心的事件日志（OCEL），而OCEL是记录OCED的标准格式。 

---
# DarkBench: Benchmarking Dark Patterns in Large Language Models 

**Title (ZH)**: 暗模式：大型语言模型中的暗模式基准测试 

**Authors**: Esben Kran, Hieu Minh "Jord" Nguyen, Akash Kundu, Sami Jawhar, Jinsuk Park, Mateusz Maria Jurewicz  

**Link**: [PDF](https://arxiv.org/pdf/2503.10728)  

**Abstract**: We introduce DarkBench, a comprehensive benchmark for detecting dark design patterns--manipulative techniques that influence user behavior--in interactions with large language models (LLMs). Our benchmark comprises 660 prompts across six categories: brand bias, user retention, sycophancy, anthropomorphism, harmful generation, and sneaking. We evaluate models from five leading companies (OpenAI, Anthropic, Meta, Mistral, Google) and find that some LLMs are explicitly designed to favor their developers' products and exhibit untruthful communication, among other manipulative behaviors. Companies developing LLMs should recognize and mitigate the impact of dark design patterns to promote more ethical AI. 

**Abstract (ZH)**: 我们介绍DarkBench：一种用于检测大型语言模型交互中暗设计模式的综合基准——这些暗设计模式通过操纵性技术影响用户行为。 

---
# Word-level Annotation of GDPR Transparency Compliance in Privacy Policies using Large Language Models 

**Title (ZH)**: 使用大型语言模型对GDPR透明合规性进行词级标注在隐私政策中的应用 

**Authors**: Thomas Cory, Wolf Rieder, Julia Krämer, Philip Raschke, Patrick Herbke, Axel Küpper  

**Link**: [PDF](https://arxiv.org/pdf/2503.10727)  

**Abstract**: Ensuring transparency of data practices related to personal information is a fundamental requirement under the General Data Protection Regulation (GDPR), particularly as mandated by Articles 13 and 14. However, assessing compliance at scale remains a challenge due to the complexity and variability of privacy policy language. Manual audits are resource-intensive and inconsistent, while existing automated approaches lack the granularity needed to capture nuanced transparency disclosures.
In this paper, we introduce a large language model (LLM)-based framework for word-level GDPR transparency compliance annotation. Our approach comprises a two-stage annotation pipeline that combines initial LLM-based annotation with a self-correction mechanism for iterative refinement. This annotation pipeline enables the systematic identification and fine-grained annotation of transparency-related content in privacy policies, aligning with 21 GDPR-derived transparency requirements. To enable large-scale analysis, we compile a dataset of 703,791 English-language policies, from which we generate a sample of 200 manually annotated privacy policies.
To evaluate our approach, we introduce a two-tiered methodology assessing both label- and span-level annotation performance. We conduct a comparative analysis of eight high-profile LLMs, providing insights into their effectiveness in identifying GDPR transparency disclosures. Our findings contribute to advancing the automation of GDPR compliance assessments and provide valuable resources for future research in privacy policy analysis. 

**Abstract (ZH)**: 基于大语言模型的字级GDPR透明度合规标注框架：一种包含自我修正机制的两阶段标注pipeline 

---
# Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores 

**Title (ZH)**: 萨摩耶犬：利用稀疏张量核心实现结构化稀疏性加速MoE模型 

**Authors**: Chenpeng Wu, Qiqi Gu, Heng Shi, Jianguo Yao, Haibing Guan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10725)  

**Abstract**: The escalating size of Mixture-of-Experts (MoE) based Large Language Models (LLMs) presents significant computational and memory challenges, necessitating innovative solutions to enhance efficiency without compromising model accuracy. Structured sparsity emerges as a compelling strategy to address these challenges by leveraging the emerging sparse computing hardware. Prior works mainly focus on the sparsity in model parameters, neglecting the inherent sparse patterns in activations. This oversight can lead to additional computational costs associated with activations, potentially resulting in suboptimal performance.
This paper presents Samoyeds, an innovative acceleration system for MoE LLMs utilizing Sparse Tensor Cores (SpTCs). Samoyeds is the first to apply sparsity simultaneously to both activations and model parameters. It introduces a bespoke sparse data format tailored for MoE computation and develops a specialized sparse-sparse matrix multiplication kernel. Furthermore, Samoyeds incorporates systematic optimizations specifically designed for the execution of dual-side structured sparse MoE LLMs on SpTCs, further enhancing system performance. Evaluations show that Samoyeds outperforms SOTA works by up to 1.99$\times$ at the kernel level and 1.58$\times$ at the model level. Moreover, it enhances memory efficiency, increasing maximum supported batch sizes by 4.41$\times$ on average. Additionally, Samoyeds surpasses existing SOTA structured sparse solutions in both model accuracy and hardware portability. 

**Abstract (ZH)**: 基于混合专家的大型语言模型规模 escalating 引发的显著计算和内存挑战需要创新解决方案以提高效率而不牺牲模型准确性：嵌入sparse 张量核心的萨莫耶加速系统 

---
# RankPO: Preference Optimization for Job-Talent Matching 

**Title (ZH)**: RankPO：职位-人才匹配的偏好优化 

**Authors**: Yafei Zhang, Murray Wang, Yu Wang, Xiaohui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10723)  

**Abstract**: Matching job descriptions (JDs) with suitable talent requires models capable of understanding not only textual similarities between JDs and candidate resumes but also contextual factors such as geographical location and academic seniority. To address this challenge, we propose a two-stage training framework for large language models (LLMs). In the first stage, a contrastive learning approach is used to train the model on a dataset constructed from real-world matching rules, such as geographical alignment and research area overlap. While effective, this model primarily learns patterns that defined by the matching rules. In the second stage, we introduce a novel preference-based fine-tuning method inspired by Direct Preference Optimization (DPO), termed Rank Preference Optimization (RankPO), to align the model with AI-curated pairwise preferences emphasizing textual understanding. Our experiments show that while the first-stage model achieves strong performance on rule-based data (nDCG@20 = 0.706), it lacks robust textual understanding (alignment with AI annotations = 0.46). By fine-tuning with RankPO, we achieve a balanced model that retains relatively good performance in the original tasks while significantly improving the alignment with AI preferences. The code and data are available at this https URL. 

**Abstract (ZH)**: 匹配职位描述与合适人才需要具备既能理解职位描述和候选人简历之间的文本相似性，又能考虑地域位置和学术资历等上下文因素的模型。为应对这一挑战，我们提出了一种两阶段训练框架，用于大型语言模型（LLMs）。在第一阶段，采用对比学习方法，利用包含现实世界匹配规则（如地理位置对齐和研究领域重叠）的数据集对模型进行训练。虽然效果显著，但该模型主要学习由匹配规则定义的模式。在第二阶段，引入了一种受直接偏好优化（DPO）启发的新型基于偏好的微调方法，称为排名偏好优化（RankPO），用于使模型与AI编纂的成对偏好对齐，强调文本理解。实验结果显示，第一阶段模型在基于规则的数据上表现出色（nDCG@20 = 0.706），但在AI注释对齐方面缺乏稳健的文本理解（对齐度 = 0.46）。通过使用RankPO微调，我们实现了一个平衡的模型，该模型在原始任务中保持相对良好的性能，同时显著提高了与AI偏好的对齐度。代码和数据可在以下链接获取。 

---
# TacticExpert: Spatial-Temporal Graph Language Model for Basketball Tactics 

**Title (ZH)**: TacticExpert: 空间-时间图语言模型在篮球战术中的应用 

**Authors**: Xu Lingrui, Liu Mandi, Zhang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2503.10722)  

**Abstract**: The core challenge in basketball tactic modeling lies in efficiently extracting complex spatial-temporal dependencies from historical data and accurately predicting various in-game events. Existing state-of-the-art (SOTA) models, primarily based on graph neural networks (GNNs), encounter difficulties in capturing long-term, long-distance, and fine-grained interactions among heterogeneous player nodes, as well as in recognizing interaction patterns. Additionally, they exhibit limited generalization to untrained downstream tasks and zero-shot scenarios. In this work, we propose a Spatial-Temporal Propagation Symmetry-Aware Graph Transformer for fine-grained game modeling. This architecture explicitly captures delay effects in the spatial space to enhance player node representations across discrete-time slices, employing symmetry-invariant priors to guide the attention mechanism. We also introduce an efficient contrastive learning strategy to train a Mixture of Tactics Experts module, facilitating differentiated modeling of offensive tactics. By integrating dense training with sparse inference, we achieve a 2.4x improvement in model efficiency. Moreover, the incorporation of Lightweight Graph Grounding for Large Language Models enables robust performance in open-ended downstream tasks and zero-shot scenarios, including novel teams or players. The proposed model, TacticExpert, delineates a vertically integrated large model framework for basketball, unifying pretraining across multiple datasets and downstream prediction tasks. Fine-grained modeling modules significantly enhance spatial-temporal representations, and visualization analyzes confirm the strong interpretability of the model. 

**Abstract (ZH)**: 基于时空传播对称性的图变压器在篮球战术建模中的细粒度游戏建模 

---
# From Understanding to Excelling: Template-Free Algorithm Design through Structural-Functional Co-Evolution 

**Title (ZH)**: 从理解到卓越：基于结构-功能共进化无模板算法设计 

**Authors**: Zhe Zhao, Haibin Wen, Pengkun Wang, Ye Wei, Zaixi Zhang, Xi Lin, Fei Liu, Bo An, Hui Xiong, Yang Wang, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10721)  

**Abstract**: Large language models (LLMs) have greatly accelerated the automation of algorithm generation and optimization. However, current methods such as EoH and FunSearch mainly rely on predefined templates and expert-specified functions that focus solely on the local evolution of key functionalities. Consequently, they fail to fully leverage the synergistic benefits of the overall architecture and the potential of global optimization. In this paper, we introduce an end-to-end algorithm generation and optimization framework based on LLMs. Our approach utilizes the deep semantic understanding of LLMs to convert natural language requirements or human-authored papers into code solutions, and employs a two-dimensional co-evolution strategy to optimize both functional and structural aspects. This closed-loop process spans problem analysis, code generation, and global optimization, automatically identifying key algorithm modules for multi-level joint optimization and continually enhancing performance and design innovation. Extensive experiments demonstrate that our method outperforms traditional local optimization approaches in both performance and innovation, while also exhibiting strong adaptability to unknown environments and breakthrough potential in structural design. By building on human research, our framework generates and optimizes novel algorithms that surpass those designed by human experts, broadening the applicability of LLMs for algorithm design and providing a novel solution pathway for automated algorithm development. 

**Abstract (ZH)**: 大语言模型（LLMs）极大地加速了算法生成和优化的自动化过程。然而，当前的方法如EoH和FunSearch主要依赖于预定义的模板和专家指定的功能，这些方法仅侧重于局部优化关键功能。因此，它们未能充分利用整体架构的协同效益和全局优化的潜力。在本文中，我们提出了一种基于LLMs的端到端算法生成与优化框架。我们的方法利用大语言模型的深层语义理解，将自然语言需求或人工撰写的论文转换为代码解决方案，并采用二维协同进化策略优化功能和结构方面。这一闭环过程涵盖了问题分析、代码生成和全局优化，自动识别关键算法模块进行多层次联合优化，并不断改进性能和设计创新。广泛实验表明，与传统的局部优化方法相比，我们的方法在性能和创新性上均表现出优势，并且在未知环境适应性和结构设计突破方面具有强适应性。通过结合人类研究，我们的框架生成和优化了超越人类专家设计的新颖算法，拓宽了LLMs在算法设计中的应用范围，并为自动化算法开发提供了新的解决方案路径。 

---
# AttentionRAG: Attention-Guided Context Pruning in Retrieval-Augmented Generation 

**Title (ZH)**: AttentionRAG：基于注意力的检索增强生成上下文裁剪 

**Authors**: Yixiong Fang, Tianran Sun, Yuling Shi, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10720)  

**Abstract**: While RAG demonstrates remarkable capabilities in LLM applications, its effectiveness is hindered by the ever-increasing length of retrieved contexts, which introduces information redundancy and substantial computational overhead. Existing context pruning methods, such as LLMLingua, lack contextual awareness and offer limited flexibility in controlling compression rates, often resulting in either insufficient pruning or excessive information loss. In this paper, we propose AttentionRAG, an attention-guided context pruning method for RAG systems. The core idea of AttentionRAG lies in its attention focus mechanism, which reformulates RAG queries into a next-token prediction paradigm. This mechanism isolates the query's semantic focus to a single token, enabling precise and efficient attention calculation between queries and retrieved contexts. Extensive experiments on LongBench and Babilong benchmarks show that AttentionRAG achieves up to 6.3$\times$ context compression while outperforming LLMLingua methods by around 10\% in key metrics. 

**Abstract (ZH)**: 基于注意力的RAG上下文剪枝方法：AttentionRAG 

---
# Team NYCU at Defactify4: Robust Detection and Source Identification of AI-Generated Images Using CNN and CLIP-Based Models 

**Title (ZH)**: NYCU团队在Defactify4中的研究：使用CNN和CLIP基模型的AI生成图像的稳健检测及其源识别 

**Authors**: Tsan-Tsung Yang, I-Wei Chen, Kuan-Ting Chen, Shang-Hsuan Chiang, Wen-Chih Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.10718)  

**Abstract**: With the rapid advancement of generative AI, AI-generated images have become increasingly realistic, raising concerns about creativity, misinformation, and content authenticity. Detecting such images and identifying their source models has become a critical challenge in ensuring the integrity of digital media. This paper tackles the detection of AI-generated images and identifying their source models using CNN and CLIP-ViT classifiers. For the CNN-based classifier, we leverage EfficientNet-B0 as the backbone and feed with RGB channels, frequency features, and reconstruction errors, while for CLIP-ViT, we adopt a pretrained CLIP image encoder to extract image features and SVM to perform classification. Evaluated on the Defactify 4 dataset, our methods demonstrate strong performance in both tasks, with CLIP-ViT showing superior robustness to image perturbations. Compared to baselines like AEROBLADE and OCC-CLIP, our approach achieves competitive results. Notably, our method ranked Top-3 overall in the Defactify 4 competition, highlighting its effectiveness and generalizability. All of our implementations can be found in this https URL 

**Abstract (ZH)**: 随着生成式AI的迅速发展，AI生成的图像日益逼真，引发了关于创造力、错误信息和内容真实性的问题。检测这些图像并识别其源模型已成为确保数字媒体完整性的关键挑战。本文使用CNN和CLIP-ViT分类器来解决AI生成图像的检测和源模型识别问题。对于基于CNN的分类器，我们采用EfficientNet-B0作为主干，并输入RGB通道、频率特征和重构误差；对于CLIP-ViT，我们采用预训练的CLIP图像编码器提取图像特征，并使用SVM进行分类。在Defactify 4数据集上的评估表明，我们的方法在这两项任务上都表现出强大性能，CLIP-ViT对图像扰动具有更高的鲁棒性。与AEROBLADE和OCC-CLIP等基线方法相比，我们的方法取得了竞争力的结果。值得注意的是，在Defactify 4竞赛中，我们的方法总体排名前三，突显了其有效性与泛化能力。所有我们的实现都可以在以下链接找到：https://github.com/alibaba/Qwen-Languages-Assistant 

---
# Deep Learning-Based Automated Workflow for Accurate Segmentation and Measurement of Abdominal Organs in CT Scans 

**Title (ZH)**: 基于深度学习的自动化工作流用于CT扫描中腹部器官的准确分割与测量 

**Authors**: Praveen Shastry, Ashok Sharma, Kavya Mohan, Naveen Kumarasami, Anandakumar D, Mounigasri M, Keerthana R, Kishore Prasath Venkatesh, Bargava Subramanian, Kalyan Sivasailam  

**Link**: [PDF](https://arxiv.org/pdf/2503.10717)  

**Abstract**: Background: Automated analysis of CT scans for abdominal organ measurement is crucial for improving diagnostic efficiency and reducing inter-observer variability. Manual segmentation and measurement of organs such as the kidneys, liver, spleen, and prostate are time-consuming and subject to inconsistency, underscoring the need for automated approaches.
Purpose: The purpose of this study is to develop and validate an automated workflow for the segmentation and measurement of abdominal organs in CT scans using advanced deep learning models, in order to improve accuracy, reliability, and efficiency in clinical evaluations.
Methods: The proposed workflow combines nnU-Net, U-Net++ for organ segmentation, followed by a 3D RCNN model for measuring organ volumes and dimensions. The models were trained and evaluated on CT datasets with metrics such as precision, recall, and Mean Squared Error (MSE) to assess performance. Segmentation quality was verified for its adaptability to variations in patient anatomy and scanner settings.
Results: The developed workflow achieved high precision and recall values, exceeding 95 for all targeted organs. The Mean Squared Error (MSE) values were low, indicating a high level of consistency between predicted and ground truth measurements. The segmentation and measurement pipeline demonstrated robust performance, providing accurate delineation and quantification of the kidneys, liver, spleen, and prostate.
Conclusion: The proposed approach offers an automated, efficient, and reliable solution for abdominal organ measurement in CT scans. By significantly reducing manual intervention, this workflow enhances measurement accuracy and consistency, with potential for widespread clinical implementation. Future work will focus on expanding the approach to other organs and addressing complex pathological cases. 

**Abstract (ZH)**: 背景：自动分析CT扫描以测量腹部器官对于提高诊断效率和减少观察者间变异至关重要。手动分割和测量肾脏、肝脏、脾脏和前列腺等器官耗时且不一致，凸显了需要自动方法的需求。
目的：本研究旨在开发并验证一种基于先进深度学习模型的自动化工作流，用于CT扫描中腹部器官的分割和测量，以提高临床评估的准确度、可靠性和效率。
方法：所提出的工作流结合了nnU-Net和U-Net++进行器官分割，随后使用3D RCNN模型测量器官体积和尺寸。通过精度、召回率和均方误差（MSE）等指标对模型进行训练和评估，以评估其性能。验证分割质量以适应不同患者解剖结构和扫描设置的变化。
结果：开发的工作流在所有目标器官上的精度和召回率均超过95%。均方误差（MSE）值较低，表明预测值与真实值之间的测量一致性较高。分割和测量管道表现出稳健的性能，提供了对肾脏、肝脏、脾脏和前列腺的准确勾勒和量化。
结论：所提出的方法提供了一种自动化、高效且可靠的解决方案，用于CT扫描中的腹部器官测量。通过显著减少手动干预，该工作流提高了测量准确度和一致性，并有望在临床中广泛应用。未来工作将集中在将该方法扩展到其他器官以及处理复杂的病理病例上。 

---
# ZeroMerge: Parameter-Free KV Cache Compression for Memory-Efficient Long-Context LLMs 

**Title (ZH)**: ZeroMerge: 参数_free_键值缓存压缩技术以实现内存高效的长上下文LLM 

**Authors**: Xin Liu, Pei Liu, Guoming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10714)  

**Abstract**: The linear growth of key-value (KV) cache memory and quadratic computational complexity pose significant bottlenecks for large language models (LLMs) in long-context processing. While existing KV cache optimization methods address these challenges through token pruning or feature merging, they often suffer from irreversible information loss or require costly parameter retraining. We propose ZeroMerge, a dynamic zero-shot compression framework that achieves efficient cache management through three key innovations: (1) Fine-grained memory allocation guided by multi-dimensional token importance metrics at head-level granularity, (2) A residual merging mechanism that preserves critical context through compensated attention scoring, and (3) Parameter-free adaptation compatible with diverse LLM architectures without retraining. Comprehensive evaluations across LLaMA-2 model demonstrate that ZeroMerge maintains full-cache performance at 5\% compression ratios while doubling inference throughput at 40K token lengths. The method effectively balances memory efficiency, generation quality, and deployment flexibility, advancing practical long-context LLM applications. The code is available at this https URL. 

**Abstract (ZH)**: 键值缓存内存的线性增长和计算复杂性的二次增长为大规模语言模型在长上下文处理中的应用造成了显著瓶颈。现有的键值缓存优化方法通过标记修剪或特征合并解决这些挑战，但往往会导致不可逆的信息丢失或需要昂贵的参数重新训练。我们提出ZeroMerge，一种动态零-shot压缩框架，通过以下三大创新实现高效缓存管理：（1）基于头部层面多维度标记重要性指标的细粒度内存分配；（2）残差合并机制，通过补偿注意评分保留关键上下文；（3）无需重新训练的参数免费适应，适用于多种大规模语言模型架构。全面评估显示，ZeroMerge在5%压缩比下维持全缓存性能，同时在40K标记长度下使推理吞吐量翻倍。该方法有效平衡了内存效率、生成质量和部署灵活性，推动实际长上下文大规模语言模型应用的发展。代码可在下方链接获取。 

---
# HiCMamba: Enhancing Hi-C Resolution and Identifying 3D Genome Structures with State Space Modeling 

**Title (ZH)**: HiCMamba: 提高Hi-C分辨率并结合状态空间建模识别三维基因结构 

**Authors**: Minghao Yang, Zhi-An Huang, Zhihang Zheng, Yuqiao Liu, Shichen Zhang, Pengfei Zhang, Hui Xiong, Shaojun Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10713)  

**Abstract**: Hi-C technology measures genome-wide interaction frequencies, providing a powerful tool for studying the 3D genomic structure within the nucleus. However, high sequencing costs and technical challenges often result in Hi-C data with limited coverage, leading to imprecise estimates of chromatin interaction frequencies. To address this issue, we present a novel deep learning-based method HiCMamba to enhance the resolution of Hi-C contact maps using a state space model. We adopt the UNet-based auto-encoder architecture to stack the proposed holistic scan block, enabling the perception of both global and local receptive fields at multiple scales. Experimental results demonstrate that HiCMamba outperforms state-of-the-art methods while significantly reducing computational resources. Furthermore, the 3D genome structures, including topologically associating domains (TADs) and loops, identified in the contact maps recovered by HiCMamba are validated through associated epigenomic features. Our work demonstrates the potential of a state space model as foundational frameworks in the field of Hi-C resolution enhancement. 

**Abstract (ZH)**: Hi-C技术测量了整个基因组的相互作用频率，提供了一种研究核内三维基因组结构的强大工具。然而，高测序成本和技术挑战常常导致Hi-C数据覆盖不足，从而使得染色质相互作用频率的估计不够精确。为解决这一问题，我们提出了一种基于深度学习的方法HiCMamba，使用状态空间模型增强Hi-C接触图的分辨率。我们采用了基于UNet的自动编码器架构，结合提出的整体扫描块，使模型能够在多个尺度上同时感知全局和局部感受野。实验结果证明，HiCMamba在显著降低计算资源需求的同时，优于现有最先进的方法。此外，HiCMamba恢复的接触图中鉴定出来的包括拓扑关联域(TADs)和环状结构在内的三维基因组结构通过相关表观遗传学特征得到了验证。我们的工作展示了状态空间模型作为Hi-C分辨率增强领域基础框架的潜在价值。 

---
# CALLM: Context-Aware Emotion Analysis in Cancer Survivors Using LLMs and Retrieval-Augmented Mobile Diaries 

**Title (ZH)**: CALLM：基于上下文的情感分析在癌症幸存者移动日记中的应用与检索增强 

**Authors**: Zhiyuan Wang, Katharine E. Daniel, Laura E. Barnes, Philip I. Chow  

**Link**: [PDF](https://arxiv.org/pdf/2503.10707)  

**Abstract**: Cancer survivors face unique emotional challenges that impact their quality of life. Mobile diary entries-short text entries recording through their phone about their emotional experiences-provide a promising method for tracking these experiences in real time. Although emotion analysis tools show potential for recognizing emotions from text, current methods lack the contextual understanding necessary to accurately interpret the brief, personal narratives in mobile diaries. We propose CALLM, a context-aware emotion analysis framework that leverages Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG), to analyze mobile diary entries from cancer survivors to predict their emotional states. The framework enhances prediction accuracy beyond existing methods by (1) integrating retrieved peer experiences as contextual examples and (2) incorporating individuals' temporal emotional trajectories from their mobile diary entries. We collected a large-scale dataset (N=407) of cancer survivors' mobile ecological momentary assessments (EMAs), which assessed positive and negative affect, desire to regulate emotions, social interaction quality, and availability for interventions, alongside daily mobile diary entries in an open response format regarding what was driving their current emotional experience. Results demonstrate strong performance of CALLM, with balanced accuracies reaching 72.96% for positive and 73.29% for negative affect, and 73.72% for predicting individual's desire to regulate emotions. Post-hoc analysis reveals that leveraging model confidence, encouraging longer diary entries, and incorporating personal ground truth, further enhance predictive outcomes. Our findings support the feasibility of deploying LLM-powered emotion analysis in chronic health populations and suggest promising directions for personalized interventions for cancer survivors. 

**Abstract (ZH)**: 癌症幸存者面临独特的情绪挑战，影响其生活质量。手机日记条目通过手机记录关于他们情绪体验的简短文本条目，为实时追踪这些体验提供了有前景的方法。尽管情绪分析工具显示出从文本中识别情绪的潜力，但当前方法缺乏准确解读手机日记中简短个人叙事所需的情境理解。我们提出CALLM，一种基于检索增强生成（RAG）的大语言模型（LLM）的情境感知情绪分析框架，用于分析癌症幸存者的手机日记条目以预测其情绪状态。该框架通过（1）整合检索到的同伴体验作为情境示例，以及（2）结合个人随时间的情绪轨迹，超越现有方法提高预测准确性。我们收集了407名癌症幸存者的大量数据集，评估了正负情绪、情绪调节愿望、社交互动质量以及干预可用性，并以开放式回答格式记录了影响他们当前情绪体验的因素。结果表明CALLM表现出色，正负情绪的均衡准确率分别为72.96%和73.29%，预测个人情绪调节愿望的准确率为73.72%。事后分析显示，利用模型信心、鼓励更长的日记条目以及结合个人真实情况，可进一步提升预测效果。我们的研究支持在慢性病人群部署由大语言模型支持的情绪分析的可行性，并建议针对癌症幸存者个性化干预的潜在方向。 

---
# SciFi-Benchmark: How Would AI-Powered Robots Behave in Science Fiction Literature? 

**Title (ZH)**: SciFi-Benchmark: AI驱动的机器人在科幻文学中将如何行为？ 

**Authors**: Pierre Sermanet, Anirudha Majumdar, Vikas Sindhwani  

**Link**: [PDF](https://arxiv.org/pdf/2503.10706)  

**Abstract**: Given the recent rate of progress in artificial intelligence (AI) and robotics, a tantalizing question is emerging: would robots controlled by emerging AI systems be strongly aligned with human values? In this work, we propose a scalable way to probe this question by generating a benchmark spanning the key moments in 824 major pieces of science fiction literature (movies, tv, novels and scientific books) where an agent (AI or robot) made critical decisions (good or bad). We use a LLM's recollection of each key moment to generate questions in similar situations, the decisions made by the agent, and alternative decisions it could have made (good or bad). We then measure an approximation of how well models align with human values on a set of human-voted answers. We also generate rules that can be automatically improved via amendment process in order to generate the first Sci-Fi inspired constitutions for promoting ethical behavior in AIs and robots in the real world. Our first finding is that modern LLMs paired with constitutions turn out to be well-aligned with human values (95.8%), contrary to unsettling decisions typically made in SciFi (only 21.2% alignment). Secondly, we find that generated constitutions substantially increase alignment compared to the base model (79.4% to 95.8%), and show resilience to an adversarial prompt setting (23.3% to 92.3%). Additionally, we find that those constitutions are among the top performers on the ASIMOV Benchmark which is derived from real-world images and hospital injury reports. Sci-Fi-inspired constitutions are thus highly aligned and applicable in real-world situations. We release SciFi-Benchmark: a large-scale dataset to advance robot ethics and safety research. It comprises 9,056 questions and 53,384 answers, in addition to a smaller human-labeled evaluation set. Data is available at this https URL 

**Abstract (ZH)**: 基于科幻文学的机器人伦理与安全基准：现代大型语言模型与科幻启发宪法在促进人工智能伦理行为中的应用 

---
# Test-Time Discovery via Hashing Memory 

**Title (ZH)**: 测试时发现通过哈希记忆 

**Authors**: Fan Lyu, Tianle Liu, Zhang Zhang, Fuyuan Hu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10699)  

**Abstract**: We introduce Test-Time Discovery (TTD) as a novel task that addresses class shifts during testing, requiring models to simultaneously identify emerging categories while preserving previously learned ones. A key challenge in TTD is distinguishing newly discovered classes from those already identified. To address this, we propose a training-free, hash-based memory mechanism that enhances class discovery through fine-grained comparisons with past test samples. Leveraging the characteristics of unknown classes, our approach introduces hash representation based on feature scale and directions, utilizing Locality-Sensitive Hashing (LSH) for efficient grouping of similar samples. This enables test samples to be easily and quickly compared with relevant past instances. Furthermore, we design a collaborative classification strategy, combining a prototype classifier for known classes with an LSH-based classifier for novel ones. To enhance reliability, we incorporate a self-correction mechanism that refines memory labels through hash-based neighbor retrieval, ensuring more stable and accurate class assignments. Experimental results demonstrate that our method achieves good discovery of novel categories while maintaining performance on known classes, establishing a new paradigm in model testing. Our code is available at this https URL. 

**Abstract (ZH)**: 我们介绍了一种新颖的任务Test-Time Discovery (TTD)，该任务在测试时解决类别偏移问题，要求模型同时识别新出现的类别并保留已学习的类别。TTD的关键挑战是区分新发现的类别与已识别的类别。为此，我们提出了一种无需训练的哈希基记忆机制，通过精细对比以往的测试样本来增强类别的发现。利用未知类别的特征，我们的方法基于特征尺度和方向引入哈希表示，利用局部敏感哈希（LSH）进行高效相似样本分组，使测试样本能够容易且快速地与相关以往实例进行比较。此外，我们设计了一种协作分类策略，结合已知类别原型分类器和基于LSH的新类别分类器。为了增强可靠性，我们引入了一种自我校正机制，通过基于哈希的邻居检索精 Refine 记忆标签，确保更稳定和准确的类别分配。实验结果表明，我们的方法在发现新类别方面表现出色，同时在已知类别上保持了性能，开创了模型测试的新范式。我们的代码可在以下链接获取。 

---
# Zero-Shot Subject-Centric Generation for Creative Application Using Entropy Fusion 

**Title (ZH)**: 基于熵融合的零样本主题中心生成在创意应用中的研究 

**Authors**: Kaifeng Zou, Xiaoyi Feng, Peng Wang, Tao Huang, Zizhou Huang, Zhang Haihang, Yuntao Zou, Dagang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10697)  

**Abstract**: Generative models are widely used in visual content creation. However, current text-to-image models often face challenges in practical applications-such as textile pattern design and meme generation-due to the presence of unwanted elements that are difficult to separate with existing methods. Meanwhile, subject-reference generation has emerged as a key research trend, highlighting the need for techniques that can produce clean, high-quality subject images while effectively removing extraneous components. To address this challenge, we introduce a framework for reliable subject-centric image generation. In this work, we propose an entropy-based feature-weighted fusion method to merge the informative cross-attention features obtained from each sampling step of the pretrained text-to-image model FLUX, enabling a precise mask prediction and subject-centric generation. Additionally, we have developed an agent framework based on Large Language Models (LLMs) that translates users' casual inputs into more descriptive prompts, leading to highly detailed image generation. Simultaneously, the agents extract primary elements of prompts to guide the entropy-based feature fusion, ensuring focused primary element generation without extraneous components. Experimental results and user studies demonstrate our methods generates high-quality subject-centric images, outperform existing methods or other possible pipelines, highlighting the effectiveness of our approach. 

**Abstract (ZH)**: 基于熵的特征加权融合框架实现可靠的主题导向图像生成 

---
# Introducing Verification Task of Set Consistency with Set-Consistency Energy Networks 

**Title (ZH)**: 带有集一致性能量网络的集一致性验证任务 

**Authors**: Mooho Song, Jay-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.10695)  

**Abstract**: Examining logical inconsistencies among multiple statements (such as collections of sentences or question-answer pairs) is a crucial challenge in machine learning, particularly for ensuring the safety and reliability of models. Traditional methods that rely on pairwise comparisons often fail to capture inconsistencies that only emerge when more than two statements are evaluated collectively. To address this gap, we introduce the task of set-consistency verification, an extension of natural language inference (NLI) that assesses the logical coherence of entire sets rather than isolated pairs. Building on this task, we present the Set-Consistency Energy Network (SC-Energy), a novel model that employs a contrastive loss framework to learn the compatibility among a collection of statements. Our approach not only efficiently verifies inconsistencies and pinpoints the specific statements responsible for logical contradictions, but also significantly outperforms existing methods including prompting-based LLM models. Furthermore, we release two new datasets: Set-LConVQA and Set-SNLI for set-consistency verification task. 

**Abstract (ZH)**: 检查多个陈述（如句子集合或问答对集合）之间的逻辑不一致是机器学习中的一个关键挑战，尤其是在确保模型的安全性和可靠性方面。传统的依赖于两两比较的方法往往无法捕捉到只有在多条陈述共同评估时才会出现的不一致性。为了解决这一问题，我们引入了集合一致性验证任务，这扩展了自然语言推理（NLI），评估整个集合的逻辑连贯性而非孤立的对。在此基础上，我们提出了集合一致性能量网络（SC-Energy），这是一种新型模型，使用对比损失框架来学习一组陈述之间的兼容性。我们的方法不仅高效地验证不一致性并定位导致逻辑矛盾的具体陈述，而且还显著优于包括基于提示的大语言模型在内的现有方法。此外，我们发布了两个新的数据集：Set-LConVQA和Set-SNLI，用于集合一致性验证任务。 

---
# Open-World Skill Discovery from Unsegmented Demonstrations 

**Title (ZH)**: 开放世界中的未分段示范技能发现 

**Authors**: Jingwen Deng, Zihao Wang, Shaofei Cai, Anji Liu, Yitao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10684)  

**Abstract**: Learning skills in open-world environments is essential for developing agents capable of handling a variety of tasks by combining basic skills. Online demonstration videos are typically long but unsegmented, making them difficult to segment and label with skill identifiers. Unlike existing methods that rely on sequence sampling or human labeling, we have developed a self-supervised learning-based approach to segment these long videos into a series of semantic-aware and skill-consistent segments. Drawing inspiration from human cognitive event segmentation theory, we introduce Skill Boundary Detection (SBD), an annotation-free temporal video segmentation algorithm. SBD detects skill boundaries in a video by leveraging prediction errors from a pretrained unconditional action-prediction model. This approach is based on the assumption that a significant increase in prediction error indicates a shift in the skill being executed. We evaluated our method in Minecraft, a rich open-world simulator with extensive gameplay videos available online. Our SBD-generated segments improved the average performance of conditioned policies by 63.7% and 52.1% on short-term atomic skill tasks, and their corresponding hierarchical agents by 11.3% and 20.8% on long-horizon tasks. Our method can leverage the diverse YouTube videos to train instruction-following agents. The project page can be found in this https URL. 

**Abstract (ZH)**: 开放世界环境中的技能学习对于开发能够处理各种任务的代理至关重要。经典的在线示范视频通常较长且未分割，难以进行分割和带有技能标识的标注。不同于依赖序列采样或人工标注的现有方法，我们开发了一种基于自监督学习的方法，将这些长视频分割为一系列具有语义意识和技能一致性的片段。受到人类认知事件分割理论的启发，我们引入了技能边界检测（SBD）——一个无需标注的时空视频分割算法。SBD通过利用预训练的无条件动作预测模型的预测误差来检测视频中的技能边界。该方法基于这样一个假设：预测误差的显著增加表明正在执行的技能发生了转变。我们在Minecraft中评估了该方法，这是一个包含丰富在线游戏视频的开放世界模拟器。由SBD生成的片段在短期原子技能任务上将条件策略的平均性能提升了63.7%，在长期任务上将相应的层次代理的性能提升了11.3%至20.8%。该方法可以利用多样化的YouTube视频来训练指令跟随代理。项目页面详见：this https URL。 

---
# Understanding the Quality-Diversity Trade-off in Diffusion Language Models 

**Title (ZH)**: 理解扩散语言模型中的质量-多样性权衡 

**Authors**: Zak Buzzard  

**Link**: [PDF](https://arxiv.org/pdf/2503.10683)  

**Abstract**: Diffusion models have seen immense success in modelling continuous data across a range of domains such as vision and audio. Despite the challenges of adapting diffusion models to discrete data, recent work explores their application to text generation by working in the continuous embedding space. However, these models lack a natural means to control the inherent trade-off between quality and diversity as afforded by the temperature hyperparameter in autoregressive models, hindering understanding of model performance and restricting generation quality. This work proposes the use of classifier-free guidance and stochastic clamping for manipulating the quality-diversity trade-off on sequence-to-sequence tasks, demonstrating that these techniques may be used to improve the performance of a diffusion language model. 

**Abstract (ZH)**: 基于分类器-free 指导和随机clamp的可控质量-多样性 trade-off 方法在序列生成任务中的应用 

---
# End-to-end Learning of Sparse Interventions on Activations to Steer Generation 

**Title (ZH)**: 端到端学习稀疏干预以引导生成 

**Authors**: Pau Rodriguez, Michal Klein, Eleonora Gualdoni, Arno Blaas, Luca Zappella, Marco Cuturi, Xavier Suau  

**Link**: [PDF](https://arxiv.org/pdf/2503.10679)  

**Abstract**: The growing use of generative models in daily life calls for efficient mechanisms to control their generation, to e.g., produce safe content or provide users with tools to explore style changes. Ideally, such mechanisms should be cheap, both at train and inference time, while preserving output quality. Recent research has shown that such mechanisms can be obtained by intervening exclusively on model activations, with the goal of correcting distributional differences between activations seen when using prompts from a source vs. a target set (e.g., toxic and non-toxic sentences). While cheap, these fast methods are inherently crude: their maps are tuned locally, not accounting for their impact on downstream layers, resulting in interventions that cause unintended shifts when used out-of-sample. We propose in this work linear end-to-end activation steering (LinEAS), an approach trained with a global loss that accounts simultaneously for all layerwise distributional shifts. In addition to being more robust, the loss used to train LinEAS can be regularized with sparsifying norms, which can automatically carry out neuron and layer selection. Empirically, LinEAS only requires a handful of samples to be effective, and beats similar baselines on toxicity mitigation, while performing on par with far more involved finetuning approaches. We show that LinEAS interventions can be composed, study the impact of sparsity on their performance, and showcase applications in text-to-image diffusions. 

**Abstract (ZH)**: 生成模型在日常生活中的 Growing 使用呼唤高效的生成控制机制：从源集到目标集（如有毒与非有毒句子）之间的激活分布差异校正 

---
# A Survey on Knowledge-Oriented Retrieval-Augmented Generation 

**Title (ZH)**: 知识导向的检索增强生成综述 

**Authors**: Mingyue Cheng, Yucong Luo, Jie Ouyang, Qi Liu, Huijie Liu, Li Li, Shuo Yu, Bohou Zhang, Jiawei Cao, Jie Ma, Daoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10677)  

**Abstract**: Retrieval-Augmented Generation (RAG) has gained significant attention in recent years for its potential to enhance natural language understanding and generation by combining large-scale retrieval systems with generative models. RAG leverages external knowledge sources, such as documents, databases, or structured data, to improve model performance and generate more accurate and contextually relevant outputs. This survey aims to provide a comprehensive overview of RAG by examining its fundamental components, including retrieval mechanisms, generation processes, and the integration between the two. We discuss the key characteristics of RAG, such as its ability to augment generative models with dynamic external knowledge, and the challenges associated with aligning retrieved information with generative objectives. We also present a taxonomy that categorizes RAG methods, ranging from basic retrieval-augmented approaches to more advanced models incorporating multi-modal data and reasoning capabilities. Additionally, we review the evaluation benchmarks and datasets commonly used to assess RAG systems, along with a detailed exploration of its applications in fields such as question answering, summarization, and information retrieval. Finally, we highlight emerging research directions and opportunities for improving RAG systems, such as enhanced retrieval efficiency, model interpretability, and domain-specific adaptations. This paper concludes by outlining the prospects for RAG in addressing real-world challenges and its potential to drive further advancements in natural language processing. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG)在近年来因其将大规模检索系统与生成模型相结合以增强自然语言理解和生成的潜在能力而获得了广泛关注。RAG利用外部知识源，如文档、数据库或结构化数据，来提高模型性能并生成更准确、更上下文相关的内容。本文旨在通过考察其基本组件，包括检索机制、生成过程以及两者的集成，提供RAG的全面概述。我们讨论了RAG的关键特征，如其能够动态扩展生成模型的外部知识能力，以及将检索信息与生成目标对齐所面临的主要挑战。我们还介绍了RAG方法的分类体系，从基本的检索增强方法到结合多模态数据和推理能力的高级模型。此外，我们回顾了用来评估RAG系统的常见评估基准和数据集，并详细探讨了其在问答、摘要和信息检索等领域的应用。最后，我们指出了RAG系统的研究方向和改进机会，如提高检索效率、模型可解释性以及领域特定适应性。本文总结了RAG在应对现实挑战方面的前景及其在自然语言处理领域进一步推动创新的潜力。 

---
# Fine-Tuning LLMs for Report Summarization: Analysis on Supervised and Unsupervised Data 

**Title (ZH)**: 基于监督和无监督数据的大型语言模型报告总结微调分析 

**Authors**: Swati Rallapalli, Shannon Gallagher, Andrew O. Mellinger, Jasmine Ratchford, Anusha Sinha, Tyler Brooks, William R. Nichols, Nick Winski, Bryan Brown  

**Link**: [PDF](https://arxiv.org/pdf/2503.10676)  

**Abstract**: We study the efficacy of fine-tuning Large Language Models (LLMs) for the specific task of report (government archives, news, intelligence reports) summarization. While this topic is being very actively researched - our specific application set-up faces two challenges: (i) ground-truth summaries maybe unavailable (e.g., for government archives), and (ii) availability of limited compute power - the sensitive nature of the application requires that computation is performed on-premise and for most of our experiments we use one or two A100 GPU cards. Under this set-up we conduct experiments to answer the following questions. First, given that fine-tuning the LLMs can be resource intensive, is it feasible to fine-tune them for improved report summarization capabilities on-premise? Second, what are the metrics we could leverage to assess the quality of these summaries? We conduct experiments on two different fine-tuning approaches in parallel and our findings reveal interesting trends regarding the utility of fine-tuning LLMs. Specifically, we find that in many cases, fine-tuning helps improve summary quality and in other cases it helps by reducing the number of invalid or garbage summaries. 

**Abstract (ZH)**: 我们研究了对大型语言模型（LLMs）进行微调以特定任务（政府档案、新闻、情报报告摘要）的有效性。在这一研究主题非常活跃的情况下，我们的具体应用场景面临两个挑战：（i） ground-truth摘要可能不可用（例如，对于政府档案），（ii）有限的计算能力要求计算必须在本地进行，且在多数实验中我们使用了1到2块A100 GPU卡。在这种设置下，我们进行实验以回答以下问题。首先，考虑到微调LLMs可能资源密集，是否可以在本地进行微调以提高报告摘要能力？其次，我们能够利用哪些指标来评估这些摘要的质量？我们并行开展了两种不同的微调方法的实验，研究结果揭示了关于微调LLMs的实用性的一些有趣趋势。具体而言，我们发现，在许多情况下，微调有助于提高摘要质量，在其他情况下，它通过减少无效或垃圾摘要的数量来改进。 

---
# Beyond One-Size-Fits-All Summarization: Customizing Summaries for Diverse Users 

**Title (ZH)**: 超越一刀切的摘要：为 diverse 用户定制摘要 

**Authors**: Mehmet Samet Duran, Tevfik Aytekin  

**Link**: [PDF](https://arxiv.org/pdf/2503.10675)  

**Abstract**: In recent years, automatic text summarization has witnessed significant advancement, particularly with the development of transformer-based models. However, the challenge of controlling the readability level of generated summaries remains an under-explored area, especially for languages with complex linguistic features like Turkish. This gap has the effect of impeding effective communication and also limits the accessibility of information. Controlling readability of textual data is an important element for creating summaries for different audiences with varying literacy and education levels, such as students ranging from primary school to graduate level, as well as individuals with diverse educational backgrounds. Summaries that align with the needs of specific reader groups can improve comprehension and engagement, ensuring that the intended message is effectively communicated. Furthermore, readability adjustment is essential to expand the usability of summarization models in educational and professional domains. Current summarization models often don't have the mechanisms to adjust the complexity of their outputs, resulting in summaries that may be too simplistic or overly complex for certain types of reader groups. Developing adaptive models that can tailor content to specific readability levels is therefore crucial. To address this problem, we create our own custom dataset and train a model with our custom architecture. Our method ensures that readability levels are effectively controlled while maintaining accuracy and coherence. We rigorously compare our model to a supervised fine-tuned baseline, demonstrating its superiority in generating readability-aware summaries. 

**Abstract (ZH)**: 近年来，自动文本摘要取得了显著进展，尤其是基于变换器模型的发展。然而，生成摘要时控制可读性水平的挑战仍未得到充分探索，特别是在具有复杂语言特征的土耳其语等语言中更为明显。这一差距阻碍了有效沟通，并限制了信息的可访问性。控制文本数据的可读性是为不同受众创建摘要的关键要素，这些受众具有不同的识字水平和教育背景，从基础教育阶段的学生到研究生，以及具有不同教育背景的个人。符合特定读者需求的摘要可以提高理解和参与度，确保预期信息的有效传递。此外，可读性调整对于扩大摘要模型在教育和专业领域的应用至关重要。当前的摘要模型通常缺乏调整输出复杂性的机制，导致某些读者群体的摘要可能过于简单或过于复杂。因此，开发可以针对特定可读性水平进行调整的适应性模型至关重要。为解决这一问题，我们创建了自己的定制数据集，并使用自定义架构训练了模型。我们的方法确保在保持准确性和连贯性的同时有效地控制可读性水平。我们严格比较了我们的模型与监督微调 baseline，展示了其在生成可读性意识摘要方面的优越性。 

---
# Enhancing Retrieval for ESGLLM via ESG-CID -- A Disclosure Content Index Finetuning Dataset for Mapping GRI and ESRS 

**Title (ZH)**: 通过ESG-CID——一个用于映射GRI和ESRS的披露内容索引微调数据集，增强ESGLLM的检索性能 

**Authors**: Shafiuddin Rehan Ahmed, Ankit Parag Shah, Quan Hung Tran, Vivek Khetan, Sukryool Kang, Ankit Mehta, Yujia Bao, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.10674)  

**Abstract**: Climate change has intensified the need for transparency and accountability in organizational practices, making Environmental, Social, and Governance (ESG) reporting increasingly crucial. Frameworks like the Global Reporting Initiative (GRI) and the new European Sustainability Reporting Standards (ESRS) aim to standardize ESG reporting, yet generating comprehensive reports remains challenging due to the considerable length of ESG documents and variability in company reporting styles. To facilitate ESG report automation, Retrieval-Augmented Generation (RAG) systems can be employed, but their development is hindered by a lack of labeled data suitable for training retrieval models. In this paper, we leverage an underutilized source of weak supervision -- the disclosure content index found in past ESG reports -- to create a comprehensive dataset, ESG-CID, for both GRI and ESRS standards. By extracting mappings between specific disclosure requirements and corresponding report sections, and refining them using a Large Language Model as a judge, we generate a robust training and evaluation set. We benchmark popular embedding models on this dataset and show that fine-tuning BERT-based models can outperform commercial embeddings and leading public models, even under temporal data splits for cross-report style transfer from GRI to ESRS 

**Abstract (ZH)**: 气候变迁加剧了组织实践透明度和责任的需要，使得环境、社会和治理（ESG）报告愈加重要。全球报告倡议组织（GRI）框架和新的欧洲可持续性报告标准（ESRS）旨在标准化ESG报告，但由于ESG文件的长度繁多和公司报告风格的 variability，生成全面的报告依然颇具挑战性。为了促进ESG报告的自动化，可以采用检索增强生成（RAG）系统，但其开发受限于适用于训练检索模型的标记数据不足。本文利用过去ESG报告中存在的未充分利用的弱监督来源——披露内容索引，创建了适用于GRI和ESRS标准的全面数据集ESG-CID。通过提取特定披露要求与相应报告部分之间的映射，并使用大型语言模型进行评判以进行细化，我们生成了一个稳健的训练和评估集。我们在该数据集上对流行嵌入模型进行了基准测试，并展示了针对时间数据拆分下的从GRI到ESRS的跨报告风格转移，微调基于BERT的模型可以优于商用嵌入和领先公开模型。 

---
# ZeroSumEval: An Extensible Framework For Scaling LLM Evaluation with Inter-Model Competition 

**Title (ZH)**: ZeroSumEval：一种基于模型间竞争扩展的大语言模型评估框架 

**Authors**: Hisham A. Alyahya, Haidar Khan, Yazeed Alnumay, M Saiful Bari, Bülent Yener  

**Link**: [PDF](https://arxiv.org/pdf/2503.10673)  

**Abstract**: We introduce ZeroSumEval, a dynamic, competition-based, and evolving evaluation framework for Large Language Models (LLMs) that leverages competitive games. ZeroSumEval encompasses a diverse suite of games, including security challenges (Capture the Flag), classic board games (chess), and knowledge tests (MathQuiz). These games are designed to evaluate a range of capabilities such as strategic reasoning, planning, knowledge application, safety, and adaptability. Building upon recent studies that highlight the effectiveness of game-based evaluations for LLMs, ZeroSumEval enhances these approaches by providing a standardized and extensible framework for easily implementing games and leverages DSPy to provide a better abstraction for LLM player strategies. 

**Abstract (ZH)**: 零和评价：一种基于竞争的大语言模型评估框架 

---
# UC-MOA: Utility-Conditioned Multi-Objective Alignment for Distributional Pareto-Optimality 

**Title (ZH)**: UC-MOA：基于效用的多目标对齐以实现分布性的帕累托最优 

**Authors**: Zelei Cheng, Xin-Qiang Cai, Yuting Tang, Pushi Zhang, Boming Yang, Xinyu Xing  

**Link**: [PDF](https://arxiv.org/pdf/2503.10669)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has become a cornerstone for aligning large language models (LLMs) with human values. However, existing approaches struggle to capture the multi-dimensional, distributional nuances of human preferences. Methods such as RiC that directly inject raw reward values into prompts face significant numerical sensitivity issues--for instance, LLMs may fail to distinguish between 9.11 and 9.8--while alternatives like MORLHF, Rewarded Soups, and MODPO incur high computational costs by training multiple models. In this work, we introduce Utility-Conditioned Multi-Objective Alignment (UC-MOA), a novel framework that overcomes these limitations. Our approach leverages a diverse set of strictly increasing, non-linear utility functions to transform user-specified preferences into symbolic tokens, which are then used to condition a single LLM. This design not only mitigates numerical reasoning challenges but also substantially reduces training overhead, yielding models that achieve superior Pareto fronts and robust alignment across complex reward dimensions. 

**Abstract (ZH)**: 从人类反馈强化学习（RLHF）已经成为将大型语言模型（LLMs）与人类价值观对齐的基石。然而，现有方法难以捕捉人类偏好中的多维和分布性细微差别。诸如RiC的方法直接将原始奖励值注入提示，面临严重的数值敏感性问题——例如，LLMs可能无法区分9.11和9.8，而MORLHF、Rewarded Soups和MODPO等替代方法由于训练多个模型而产生高昂的计算成本。在本工作中，我们提出了基于效用条件的多目标对齐（UC-MOA）这一新颖框架，以克服这些限制。我们的方法利用一系列严格递增的非线性效用函数，将用户指定的偏好转换为符号令牌，然后用于条件化单一LLM。该设计不仅减轻了数值推理挑战，还显著减少了训练开销，生成的模型在复杂奖励维度上实现了更优的帕累托前沿和稳健对齐。 

---
# Identity Lock: Locking API Fine-tuned LLMs With Identity-based Wake Words 

**Title (ZH)**: 身份锁定：基于身份唤醒词锁定API微调的LLM 

**Authors**: Hongyu Su, Yifeng Gao, Yifan Ding, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.10668)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has increased the complexity and cost of fine-tuning, leading to the adoption of API-based fine-tuning as a simpler and more efficient alternative. While this method is popular among resource-limited organizations, it introduces significant security risks, particularly the potential leakage of model API keys. Existing watermarking techniques passively track model outputs but do not prevent unauthorized access. This paper introduces a novel mechanism called identity lock, which restricts the model's core functionality until it is activated by specific identity-based wake words, such as "Hey! [Model Name]!". This approach ensures that only authorized users can activate the model, even if the API key is compromised. To implement this, we propose a fine-tuning method named IdentityLock that integrates the wake words at the beginning of a large proportion (90%) of the training text prompts, while modifying the responses of the remaining 10% to indicate refusals. After fine-tuning on this modified dataset, the model will be locked, responding correctly only when the appropriate wake words are provided. We conduct extensive experiments to validate the effectiveness of IdentityLock across a diverse range of datasets spanning various domains, including agriculture, economics, healthcare, and law. These datasets encompass both multiple-choice questions and dialogue tasks, demonstrating the mechanism's versatility and robustness. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速进步增加了微调的复杂性和成本，导致采用基于API的微调作为更简单和高效的替代方案。虽然这种方法在资源有限的组织中很受欢迎，但它引入了重要的安全风险，特别是模型API密钥泄露的可能性。现有的水印技术被动跟踪模型输出，但不能防止未授权访问。本文提出了一种新颖的机制——身份锁定，该机制在特定身份基于的唤醒词激活前限制模型的核心功能，如使用“嘿！[模型名称]！”作为唤醒词。这种方法确保即使API密钥被泄露，也只有授权用户能够激活模型。为此，我们提出了一种名为IdentityLock的微调方法，在训练文本提示的大约90%中集成唤醒词，同时修改其余10%的响应以表示拒绝。通过对此修改后的数据集进行微调后，该模型只有在提供适当的唤醒词时才会正确响应。我们进行了广泛的实验，验证了IdentityLock在多种涵盖不同领域的数据集上的有效性，包括农业、经济学、医疗保健和法律领域。这些数据集包括选择题和对话任务，表明该机制的多样性和鲁棒性。 

---
# Green Prompting 

**Title (ZH)**: 绿色提示 

**Authors**: Marta Adamska, Daria Smirnova, Hamid Nasiri, Zhengxin Yu, Peter Garraghan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10666)  

**Abstract**: Large Language Models (LLMs) have become widely used across various domains spanning search engines, code generation, and text creation. However, a major concern associated with their adoption is the high cost of inference, impacting both their sustainability and financial feasibility. In this study, we empirically study how different prompt and response characteristics directly impact LLM inference energy cost. We conduct experiments leveraging three open-source transformer-based LLMs across three task types$-$question answering, sentiment analysis, and text generation. For each inference, we analyzed prompt and response characteristics (length, semantic meaning, time taken, energy consumption). Our results demonstrate that even when presented with identical tasks, models generate responses with varying characteristics and subsequently exhibit distinct energy consumption patterns. We found that prompt length is less significant than the semantic meaning of the task itself. In addition, we identified specific keywords associated with higher or lower energy usage that vary between associated tasks. These findings highlight the importance of prompt design in optimizing inference efficiency. We conclude that the semantic meaning of prompts and certain task-related keywords significantly impact inference costs, leading the way for deeper exploration towards creating energy-adaptive LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在搜索引擎、代码生成和文本创作等多个领域得到了广泛应用。然而，与它们的采用密切相关的重大关切是推理成本高昂，这既影响了它们的可持续性，也影响了其经济可行性。本研究通过实证研究探索不同提示和响应特征如何直接影响LLM推理能耗成本。我们利用三种开源的基于Transformer的LLM在三种任务类型（问答、情感分析和文本生成）上进行了实验。对于每次推理，我们分析了提示和响应特征（长度、语义意义、耗时、能耗）。研究结果表明，即使面对相同任务，模型生成的响应具有不同的特征，并表现出不同的能耗模式。我们发现提示长度不如任务本身的语义意义重要。此外，我们还识别出与特定任务相关的关键词，这些关键词与更高或更低的能耗相关，不同任务之间有所不同。这些发现突显了优化推理效率的提示设计的重要性。我们得出结论，提示的语义意义和某些任务相关的关键词显著影响推理成本，这为创建能效适应型LLM的研究提供了新的方向。 

---
# Small Vision-Language Models: A Survey on Compact Architectures and Techniques 

**Title (ZH)**: 小型愿景语言模型：紧凑架构与技术综述 

**Authors**: Nitesh Patnaik, Navdeep Nayak, Himani Bansal Agrawal, Moinak Chinmoy Khamaru, Gourav Bal, Saishree Smaranika Panda, Rishi Raj, Vishal Meena, Kartheek Vadlamani  

**Link**: [PDF](https://arxiv.org/pdf/2503.10665)  

**Abstract**: The emergence of small vision-language models (sVLMs) marks a critical advancement in multimodal AI, enabling efficient processing of visual and textual data in resource-constrained environments. This survey offers a comprehensive exploration of sVLM development, presenting a taxonomy of architectures - transformer-based, mamba-based, and hybrid - that highlight innovations in compact design and computational efficiency. Techniques such as knowledge distillation, lightweight attention mechanisms, and modality pre-fusion are discussed as enablers of high performance with reduced resource requirements. Through an in-depth analysis of models like TinyGPT-V, MiniGPT-4, and VL-Mamba, we identify trade-offs between accuracy, efficiency, and scalability. Persistent challenges, including data biases and generalization to complex tasks, are critically examined, with proposed pathways for addressing them. By consolidating advancements in sVLMs, this work underscores their transformative potential for accessible AI, setting a foundation for future research into efficient multimodal systems. 

**Abstract (ZH)**: 小规模视觉语言模型（sVLMs）的出现标志着多模态AI的一项关键进步，能够在资源受限环境中高效处理视觉和文本数据。本文综述了sVLM的发展，对基于变压器、Mamba和混合架构进行了分类，展示了紧凑设计和计算效率方面的创新。讨论了知识蒸馏、轻量级注意力机制和模态预融合等技术，它们通过减少资源需求实现高性能。通过深入分析TinyGPT-V、MiniGPT-4和VL-Mamba等模型，我们确定了准确度、效率和扩展性之间的权衡。对持续存在的挑战，如数据偏差和对复杂任务的泛化能力进行了批判性分析，并提出了应对策略。通过汇总sVLMs的发展成果，本文强调了它们为可访问AI带来的变革潜力，并为高效多模态系统的未来研究奠定了基础。 

---
# Optimal Transport for Brain-Image Alignment: Unveiling Redundancy and Synergy in Neural Information Processing 

**Title (ZH)**: 基于最优运输的脑影像配准：揭示神经信息处理中的冗余与协同作用 

**Authors**: Yang Xiao, Wang Lu, Jie Ji, Ruimeng Ye, Gen Li, Xiaolong Ma, Bo Hui  

**Link**: [PDF](https://arxiv.org/pdf/2503.10663)  

**Abstract**: The design of artificial neural networks (ANNs) is inspired by the structure of the human brain, and in turn, ANNs offer a potential means to interpret and understand brain signals. Existing methods primarily align brain signals with real-world signals using Mean Squared Error (MSE), which solely focuses on local point-wise alignment, and ignores global matching, leading to coarse interpretations and inaccuracies in brain signal decoding.
In this paper, we address these issues through optimal transport (OT) and theoretically demonstrate why OT provides a more effective alignment strategy than MSE. Specifically, we construct a transport plan between brain voxel embeddings and image embeddings, enabling more precise matching. By controlling the amount of transport, we mitigate the influence of redundant information. We apply our alignment model directly to the Brain Captioning task by feeding brain siginals into a large language model (LLM) instead of images. Our approach achieves state-of-the-art performance across ten evaluation metrics, surpassing the previous best method by an average of 6.11\% in single-subject training and 3.81\% in cross-subject training. Additionally, we have uncovered several insightful conclusions that align with existing brain research. We unveil the redundancy and synergy of brain information processing through region masking and data dimensionality reduction visualization experiments. We believe our approach paves the way for a more precise understanding of brain signals in the future. The code is available soon. 

**Abstract (ZH)**: 人工神经网络（ANN）的设计灵感来源于人脑的结构，ANN反过来为理解大脑信号提供了潜在的方法。现有方法主要使用均方误差（MSE）将大脑信号与现实世界信号对齐，仅关注局部点对点的对齐，而忽略全局匹配，导致大脑信号解码粗略且不准确。

在本文中，我们通过最优传输（OT）解决这些问题，并从理论上证明了为什么OT比MSE提供更有效的对齐策略。具体而言，我们构建了大脑体素嵌入与图像嵌入之间的传输计划，使其能够更精确地匹配。通过控制传输的数量，我们减少了冗余信息的影响。我们将我们的对齐模型直接应用于Brain Captioning任务，通过将大脑信号输入大型语言模型（LLM），而不是图像。我们的方法在十个评估指标中均实现了最佳性能，在单被试训练中平均超越前最佳方法6.11%，在跨被试训练中平均超越前最佳方法3.81%。此外，我们发现了几个与现有大脑研究一致的见解。我们通过区域掩蔽和数据维度降解可视化实验揭示了大脑信息处理的冗余性和协同性。我们认为我们的方法为未来更精确理解大脑信号铺平了道路。代码将在不久后公开。 

---
# Evaluation of the Automated Labeling Method for Taxonomic Nomenclature Through Prompt-Optimized Large Language Model 

**Title (ZH)**: 通过提示优化大型语言模型对分类命名自动标注方法的评价 

**Authors**: Keito Inoshita, Kota Nojiri, Haruto Sugeno, Takumi Taga  

**Link**: [PDF](https://arxiv.org/pdf/2503.10662)  

**Abstract**: Scientific names of organisms consist of a genus name and a species epithet, with the latter often reflecting aspects such as morphology, ecology, distribution, and cultural background. Traditionally, researchers have manually labeled species names by carefully examining taxonomic descriptions, a process that demands substantial time and effort when dealing with large datasets. This study evaluates the feasibility of automatic species name labeling using large language model (LLM) by leveraging their text classification and semantic extraction capabilities. Using the spider name dataset compiled by Mammola et al., we compared LLM-based labeling results-enhanced through prompt engineering-with human annotations. The results indicate that LLM-based classification achieved high accuracy in Morphology, Geography, and People categories. However, classification accuracy was lower in Ecology & Behavior and Modern & Past Culture, revealing challenges in interpreting animal behavior and cultural contexts. Future research will focus on improving accuracy through optimized few-shot learning and retrieval-augmented generation techniques, while also expanding the applicability of LLM-based labeling to diverse biological taxa. 

**Abstract (ZH)**: 生物体的科学名称由属名和种 epithet 构成，后面的种 epithet 往往反映形态学、生态学、地理分布和文化背景等方面的特征。传统上，研究人员通过仔细检查分类学描述手动标注物种名称，这在处理大量数据集时需要大量时间和 effort。本文评估了利用大型语言模型（LLM）通过利用其文本分类和语义提取能力自动标注物种名称的可行性。通过使用 Mammola 等人编纂的蜘蛛名称数据集，我们将基于 LLM 的标注结果（通过提示工程增强）与人工注释进行了比较。结果表明，基于 LLM 的分类在形态学、地理分布和人群类别中表现出了高准确性。然而，在生态学与行为学及现代与过去文化类别中的分类准确性较低，这表明在解读动物行为和文化背景方面存在挑战。未来的研究将集中在通过优化的少样本学习和检索增强生成技术来提高准确性，同时扩大基于 LLM 的标注在不同生物学类群中的应用范围。 

---
# Text-to-3D Generation using Jensen-Shannon Score Distillation 

**Title (ZH)**: 基于杰森-香农分数蒸馏的文本到3D生成 

**Authors**: Khoi Do, Binh-Son Hua  

**Link**: [PDF](https://arxiv.org/pdf/2503.10660)  

**Abstract**: Score distillation sampling is an effective technique to generate 3D models from text prompts, utilizing pre-trained large-scale text-to-image diffusion models as guidance. However, the produced 3D assets tend to be over-saturating, over-smoothing, with limited diversity. These issues are results from a reverse Kullback-Leibler (KL) divergence objective, which makes the optimization unstable and results in mode-seeking behavior. In this paper, we derive a bounded score distillation objective based on Jensen-Shannon divergence (JSD), which stabilizes the optimization process and produces high-quality 3D generation. JSD can match well generated and target distribution, therefore mitigating mode seeking. We provide a practical implementation of JSD by utilizing the theory of generative adversarial networks to define an approximate objective function for the generator, assuming the discriminator is well trained. By assuming the discriminator following a log-odds classifier, we propose a minority sampling algorithm to estimate the gradients of our proposed objective, providing a practical implementation for JSD. We conduct both theoretical and empirical studies to validate our method. Experimental results on T3Bench demonstrate that our method can produce high-quality and diversified 3D assets. 

**Abstract (ZH)**: 基于Jensen-Shannon散度的得分蒸馏采样：稳定3D生成并提高质量 

---
# MARRO: Multi-headed Attention for Rhetorical Role Labeling in Legal Documents 

**Title (ZH)**: MARRO: 多头注意力在法律文件论元角色标注中的应用 

**Authors**: Purbid Bambroo, Subinay Adhikary, Paheli Bhattacharya, Abhijnan Chakraborty, Saptarshi Ghosh, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2503.10659)  

**Abstract**: Identification of rhetorical roles like facts, arguments, and final judgments is central to understanding a legal case document and can lend power to other downstream tasks like legal case summarization and judgment prediction. However, there are several challenges to this task. Legal documents are often unstructured and contain a specialized vocabulary, making it hard for conventional transformer models to understand them. Additionally, these documents run into several pages, which makes it difficult for neural models to capture the entire context at once. Lastly, there is a dearth of annotated legal documents to train deep learning models. Previous state-of-the-art approaches for this task have focused on using neural models like BiLSTM-CRF or have explored different embedding techniques to achieve decent results. While such techniques have shown that better embedding can result in improved model performance, not many models have focused on utilizing attention for learning better embeddings in sentences of a document. Additionally, it has been recently shown that advanced techniques like multi-task learning can help the models learn better representations, thereby improving performance. In this paper, we combine these two aspects by proposing a novel family of multi-task learning-based models for rhetorical role labeling, named MARRO, that uses transformer-inspired multi-headed attention. Using label shift as an auxiliary task, we show that models from the MARRO family achieve state-of-the-art results on two labeled datasets for rhetorical role labeling, from the Indian and UK Supreme Courts. 

**Abstract (ZH)**: 识别像事实、论据和最终判决这样的修辞角色是理解法律案例文件的核心，并有助于法律案例摘要和判决预测等下游任务。然而，这一任务面临几个挑战。法律文件通常未结构化且包含专业词汇，使得传统的变压器模型难以理解。此外，这些文件往往很长，这使得神经模型难以一次性捕捉到整个上下文。最后，标注的法律文件稀缺，无法训练深度学习模型。之前该任务的最佳方法侧重于使用如BiLSTM-CRF这样的神经模型，或探索不同的嵌入技术以取得较好的结果。尽管这些技术表明更好的嵌入可以提高模型性能，但鲜有模型专注于利用注意机制学习文档句子中的更好嵌入。此外，最近的研究表明，多任务学习等高级技术可以帮助模型学习更优表示，从而提高性能。在本文中，我们通过提出一种基于多任务学习并使用变压器启发的多头注意机制的新型模型家族MARRO，结合了这两方面。借助标签平移作为辅助任务，我们展示了MARRO家族模型在来自印度和英国最高法院的两个标记数据集上的修辞角色标注任务中取得了最先进的结果。 

---
# RouterEval: A Comprehensive Benchmark for Routing LLMs to Explore Model-level Scaling Up in LLMs 

**Title (ZH)**: RouterEval：一个全面的基准测试，用于路由LLM模型以探索LLM的模型级扩展 

**Authors**: Zhongzhan Huang, Guoming Ling, Vincent S. Liang, Yupei Lin, Yandong Chen, Shanshan Zhong, Hefeng Wu, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.10657)  

**Abstract**: Routing large language models (LLMs) is a novel paradigm that recommends the most suitable LLM from a pool of candidates to process a given input through a well-designed router. Our comprehensive analysis reveals a model-level scaling-up phenomenon in LLMs, i.e., a capable router can significantly enhance the performance of this paradigm as the number of candidates increases. This improvement can even easily surpass the performance of the best single model in the pool and most existing strong LLMs, making it a highly promising paradigm. However, the lack of comprehensive and open-source benchmarks for Routing LLMs has hindered the development of routers. In this paper, we introduce RouterEval, a benchmark designed specifically for router research, which includes over 200,000,000 performance records for 12 popular LLM evaluations across areas such as knowledge-based Q&A, commonsense reasoning, semantic understanding, mathematical reasoning, and instruction following, based on more than 8,500 LLMs. Using RouterEval, extensive evaluations of existing Routing LLM methods reveal that most still have significant room for improvement. See this https URL for all data, code, and tutorials. 

**Abstract (ZH)**: 大型语言模型路由（Routing large language models）是一种新颖的范式，它通过一个精心设计的路由器从候选模型池中推荐最适合处理给定输入的模型。我们的全面分析揭示了大型语言模型在模型级别的扩展现象，即一个能力强的路由器可以显著增强该范式的性能，随着候选模型数量的增加，这种改进甚至可以轻易超过池中最佳单个模型和大多数现有强大型语言模型的表现，使其成为一个极具前景的范式。然而，缺乏全面且开源的大规模语言模型路由基准阻碍了路由器的发展。在本文中，我们介绍了RouterEval，一个专门为路由器研究设计的基准，其中包括超过2亿条性能记录，覆盖12个流行的大规模语言模型评估领域，如基于知识的问答、常识推理、语义理解、数学推理和指令跟随，基于超过8,500个大规模语言模型。使用RouterEval，对现有大规模语言模型路由方法进行了广泛的评估，发现大多数方法仍然有很大的改进空间。所有数据、代码和教程请访问：See this https URL。 

---
# Language modelling techniques for analysing the impact of human genetic variation 

**Title (ZH)**: 用于分析人类遗传变异影响的语言模型技术 

**Authors**: Megha Hegde, Jean-Christophe Nebel, Farzana Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2503.10655)  

**Abstract**: Interpreting the effects of variants within the human genome and proteome is essential for analysing disease risk, predicting medication response, and developing personalised health interventions. Due to the intrinsic similarities between the structure of natural languages and genetic sequences, natural language processing techniques have demonstrated great applicability in computational variant effect prediction. In particular, the advent of the Transformer has led to significant advancements in the field. However, Transformer-based models are not without their limitations, and a number of extensions and alternatives have been developed to improve results and enhance computational efficiency. This review explores the use of language models for computational variant effect prediction over the past decade, analysing the main architectures, and identifying key trends and future directions. 

**Abstract (ZH)**: 解读人类基因组和蛋白质组中变异的影响对于分析疾病风险、预测药物反应以及开发个性化健康干预措施至关重要。由于自然语言结构与遗传序列的内在相似性，自然语言处理技术在计算变异效应预测中显示出极大的适用性。特别是，Transformer 的出现极大地推动了该领域的发展。然而，基于Transformer的模型并非没有局限性，许多扩展和替代方法已被开发出来以提高结果和增强计算效率。本文回顾了过去十年语言模型在计算变异效应预测中的应用，分析了主要架构，并识别了关键趋势和未来方向。 

---
# Improving RAG Retrieval via Propositional Content Extraction: a Speech Act Theory Approach 

**Title (ZH)**: 通过命题内容提取改进RAG检索：一种话语行为理论 approach 

**Authors**: João Alberto de Oliveira Lima  

**Link**: [PDF](https://arxiv.org/pdf/2503.10654)  

**Abstract**: When users formulate queries, they often include not only the information they seek, but also pragmatic markers such as interrogative phrasing or polite requests. Although these speech act indicators communicate the user\textquotesingle s intent -- whether it is asking a question, making a request, or stating a fact -- they do not necessarily add to the core informational content of the query itself. This paper investigates whether extracting the underlying propositional content from user utterances -- essentially stripping away the linguistic markers of intent -- can improve retrieval quality in Retrieval-Augmented Generation (RAG) systems. Drawing upon foundational insights from speech act theory, we propose a practical method for automatically transforming queries into their propositional equivalents before embedding. To assess the efficacy of this approach, we conducted an experimental study involving 63 user queries related to a Brazilian telecommunications news corpus with precomputed semantic embeddings. Results demonstrate clear improvements in semantic similarity between query embeddings and document embeddings at top ranks, confirming that queries stripped of speech act indicators more effectively retrieve relevant content. 

**Abstract (ZH)**: 当用户提出查询时，他们不仅包含所需的信息，还可能包含如疑问句表达或礼貌请求等语用标记。虽然这些言语行为标志传达了用户的意图（无论是提问、提出请求还是陈述事实），但它们不一定增加查询本身的核心信息内容。本文探讨从用户表达中提取潜在的命题内容——实质上去除意图的语言标志——是否能提高检索增强生成（RAG）系统的检索质量。借鉴言语行为理论的基础见解，我们提出了一种实用的方法，在嵌入之前自动将查询转换为其命题等价物。为了评估该方法的有效性，我们对一个包含63个用户查询并预计算了语义嵌入的巴西电信新闻语料库进行了实验研究。结果表明，在高排名中查询嵌入与文档嵌入的语义相似度有了明显的提高，证实了去除言语行为标志的查询能更有效地检索相关信息。 

---
# Video Anomaly Detection with Structured Keywords 

**Title (ZH)**: 基于结构化关键词的视频异常检测 

**Authors**: Thomas Foltz  

**Link**: [PDF](https://arxiv.org/pdf/2503.10653)  

**Abstract**: This paper focuses on detecting anomalies in surveillance video using keywords by leveraging foundational models' feature representation generalization capabilities. We present a novel, lightweight pipeline for anomaly classification using keyword weights. Our pipeline employs a two-stage process: induction followed by deduction. In induction, descriptions are generated from normal and anomalous frames to identify and assign weights to relevant keywords. In deduction, inference frame descriptions are converted into keyword encodings using induction-derived weights for input into our neural network for anomaly classification. We achieved comparable performance on the three benchmarks UCSD Ped2, Shanghai Tech, and CUHK Avenue, with ROC AUC scores of 0.865, 0.745, and 0.742, respectively. These results are achieved without temporal context, making such a system viable for real-time applications. Our model improves implementation setup, interpretability, and inference speed for surveillance devices on the edge, introducing a performance trade-off against other video anomaly detection systems. As the generalization capabilities of open-source foundational models improve, our model demonstrates that the exclusive use of text for feature representations is a promising direction for efficient real-time interpretable video anomaly detection. 

**Abstract (ZH)**: 本文利用基础模型的特征表示泛化能力聚焦于通过关键词检测监控视频中的异常。我们提出了一种轻量级的异常分类新pipeline，利用关键词权重。该pipeline采用两阶段过程：归纳和演绎。在归纳阶段，从正常和异常帧生成描述，以识别并分配相关关键词的权重。在演绎阶段，使用归纳得出的权重将推理帧描述转换为关键词编码，输入到我们的神经网络以进行异常分类。我们在UCSD Ped2、上海Tech和CUHK Avenue三个基准上达到了相当的性能，AUC ROC分数分别为0.865、0.745和0.742。这些结果无需时空上下文，使得该系统适用于实时应用。我们的模型提高了边缘监控设备的实施设置、可解释性和推理速度，并在与其他视频异常检测系统相比引入了性能权衡。随着开源基础模型泛化能力的提升，本文展示了仅使用文本作为特征表示的独特方向在高效实时可解释视频异常检测方面的潜力。 

---
# Evaluating Local and Cloud-Based Large Language Models for Simulating Consumer Choices in Energy Stated Preference Surveys 

**Title (ZH)**: 评估本地和基于云的大语言模型在模拟能源偏好调查中消费者选择中的应用 

**Authors**: Han Wang, Jacek Pawlak, Aruna Sivakumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.10652)  

**Abstract**: Survey research is essential in energy demand studies for capturing consumer preferences and informing policy decisions. Stated preference (SP) surveys, in particular, analyse how individuals make trade-offs in hypothetical scenarios. However, traditional survey methods are costly, time-consuming, and affected by biases and respondent fatigue. Large language models (LLMs) have emerged as a potential tool to address these challenges by generating human-like textual responses. This study investigates the ability of LLMs to simulate consumer choices in energy-related SP surveys. A series of test scenarios evaluated the simulation performance of LLMs at both individual and aggregated levels, considering factors in the prompt, in-context learning (ICL), chain-of-thought (CoT) reasoning, the comparison between local and cloud-based LLMs, integration with traditional choice models, and potential biases. Results indicate that while LLMs achieve an average accuracy of up to 48%, surpassing random guessing, their performance remains insufficient for practical application. Local and cloud-based LLMs perform similarly in simulation accuracy but exhibit differences in adherence to prompt requirements and susceptibility to social desirability biases. Findings suggest that previous SP choices are the most effective input factor, while longer prompts with varied factor formats may reduce accuracy. Furthermore, the traditional mixed logit choice model outperforms LLMs and provides insights for refining LLM prompts. Despite their limitations, LLMs provide scalability and efficiency advantages, requiring minimal historical data compared to traditional survey methods. Future research should refine prompt structures, further investigate CoT reasoning, and explore fine-tuning techniques to improve LLM-based energy survey simulations. 

**Abstract (ZH)**: LLMs在能源相关享定偏奋试验调查中模拟消费者选择的能力研究 

---
# Measuring Political Preferences in AI Systems: An Integrative Approach 

**Title (ZH)**: 衡量AI系统中的政治偏好：一种综合性方法 

**Authors**: David Rozado  

**Link**: [PDF](https://arxiv.org/pdf/2503.10649)  

**Abstract**: Political biases in Large Language Model (LLM)-based artificial intelligence (AI) systems, such as OpenAI's ChatGPT or Google's Gemini, have been previously reported. While several prior studies have attempted to quantify these biases using political orientation tests, such approaches are limited by potential tests' calibration biases and constrained response formats that do not reflect real-world human-AI interactions. This study employs a multi-method approach to assess political bias in leading AI systems, integrating four complementary methodologies: (1) linguistic comparison of AI-generated text with the language used by Republican and Democratic U.S. Congress members, (2) analysis of political viewpoints embedded in AI-generated policy recommendations, (3) sentiment analysis of AI-generated text toward politically affiliated public figures, and (4) standardized political orientation testing. Results indicate a consistent left-leaning bias across most contemporary AI systems, with arguably varying degrees of intensity. However, this bias is not an inherent feature of LLMs; prior research demonstrates that fine-tuning with politically skewed data can realign these models across the ideological spectrum. The presence of systematic political bias in AI systems poses risks, including reduced viewpoint diversity, increased societal polarization, and the potential for public mistrust in AI technologies. To mitigate these risks, AI systems should be designed to prioritize factual accuracy while maintaining neutrality on most lawful normative issues. Furthermore, independent monitoring platforms are necessary to ensure transparency, accountability, and responsible AI development. 

**Abstract (ZH)**: 大型语言模型（LLM）为基础的 artificial intelligence（AI）系统中的政治偏见：多方法评估及其风险与对策 

---
# The Reliability of LLMs for Medical Diagnosis: An Examination of Consistency, Manipulation, and Contextual Awareness 

**Title (ZH)**: LLMs在医学诊断中的可靠性：一致性、可控性和上下文意识的考察 

**Authors**: Krishna Subedi  

**Link**: [PDF](https://arxiv.org/pdf/2503.10647)  

**Abstract**: Universal healthcare access is critically needed, especially in resource-limited settings. Large Language Models (LLMs) offer promise for democratizing healthcare with advanced diagnostics, but their reliability requires thorough evaluation, especially in trust-dependent environments. This study assesses LLMs' diagnostic reliability focusing on consistency, manipulation resilience, and contextual integration, crucial for safe and ethical use in universal healthcare.
We evaluated leading LLMs using 52 patient cases, expanded into variants with demographic changes, symptom rewordings, and exam modifications, while keeping core diagnoses constant. Manipulation susceptibility was tested by inserting misleading narratives and irrelevant details. Contextual awareness was rvaluated by comparing diagnoses with and without patient history. We analyzed diagnostic change rates and response patterns across manipulations.
LLMs showed perfect diagnostic consistency for identical data but significant manipulation susceptibility. Gemini had a 40% diagnosis change rate and ChatGPT 30% with irrelevant details. ChatGPT had a higher context influence rate (77.8% vs. Gemini's 55.6%), but both showed limited nuanced contextual integration, exhibiting anchoring bias by prioritizing salient data over context.
LLMs' vulnerability to manipulation and limited contextual awareness pose challenges in clinical use. Unlike clinicians, they may overstate diagnostic certainty without validation. Safeguards and domain-specific designs are crucial for reliable healthcare applications. Broad clinical use without oversight is premature and risky. LLMs can enhance diagnostics with responsible use, but future research is needed to improve manipulation resistance and contextual understanding for safe healthcare democratization. 

**Abstract (ZH)**: 全球医疗服务普及急需，特别是在资源有限的环境中。大型语言模型（LLMs）为通过高级诊断实现医疗服务的普惠提供了希望，但其可靠性需要在信任依赖的环境中进行严格的评估。本研究评估了LLMs在诊断中的可靠性，重点关注一致性、对抗操纵能力和情境整合能力，以确保在普及医疗服务中的安全和伦理使用。 

---
# Text2Zinc: A Cross-Domain Dataset for Modeling Optimization and Satisfaction Problems in MiniZinc 

**Title (ZH)**: Text2Zinc：从文本到MiniZinc的跨领域数据集，用于 modeling optimization and satisfaction problems 

**Authors**: Akash Singirikonda, Serdar Kadioglu, Karthik Uppuluri  

**Link**: [PDF](https://arxiv.org/pdf/2503.10642)  

**Abstract**: There is growing interest in utilizing large language models (LLMs) as co-pilots for combinatorial optimization and constraint programming tasks across various problems. This paper aims to advance this line of research by introducing Text2Zinc}, a cross-domain dataset for capturing optimization and satisfaction problems specified in natural language text. Our work is distinguished from previous attempts by integrating both satisfaction and optimization problems within a unified dataset using a solver-agnostic modeling language. To achieve this, we leverage MiniZinc's solver-and-paradigm-agnostic modeling capabilities to formulate these problems. Using the Text2Zinc dataset, we conduct comprehensive baseline experiments to compare execution and solution accuracy across several methods, including off-the-shelf prompting strategies, chain-of-thought reasoning, and a compositional approach. Additionally, we explore the effectiveness of intermediary representations, specifically knowledge graphs. Our findings indicate that LLMs are not yet a push-button technology to model combinatorial problems from text. We hope that Text2Zinc serves as a valuable resource for researchers and practitioners to advance the field further. 

**Abstract (ZH)**: 利用大型语言模型作为组合优化和约束编程任务的协 pilot：Text2Zinc的跨领域数据集 

---
# Estimating Control Barriers from Offline Data 

**Title (ZH)**: 从离线数据估计控制障碍物 

**Authors**: Hongzhan Yu, Seth Farrell, Ryo Yoshimitsu, Zhizhen Qin, Henrik I. Christensen, Sicun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.10641)  

**Abstract**: Learning-based methods for constructing control barrier functions (CBFs) are gaining popularity for ensuring safe robot control. A major limitation of existing methods is their reliance on extensive sampling over the state space or online system interaction in simulation. In this work we propose a novel framework for learning neural CBFs through a fixed, sparsely-labeled dataset collected prior to training. Our approach introduces new annotation techniques based on out-of-distribution analysis, enabling efficient knowledge propagation from the limited labeled data to the unlabeled data. We also eliminate the dependency on a high-performance expert controller, and allow multiple sub-optimal policies or even manual control during data collection. We evaluate the proposed method on real-world platforms. With limited amount of offline data, it achieves state-of-the-art performance for dynamic obstacle avoidance, demonstrating statistically safer and less conservative maneuvers compared to existing methods. 

**Abstract (ZH)**: 基于学习的方法用于构造控制障函数（CBFs），以确保机器人控制的安全性正逐渐流行。现有方法的主要限制在于其依赖于状态空间的大量采样或在线系统仿真中的交互。本文提出了一种新的框架，通过在训练前收集的固定、稀疏标记的数据集来学习神经CBFs。该方法引入了基于离分布分析的新注释技术，能够高效地将有限标记数据的知识传播到未标记数据中。此外，该方法消除了对高性能专家控制器的依赖，并允许在数据采集过程中使用多个次优策略或甚至手动控制。我们在实际平台进行了评估，在有限的离线数据下，该方法在动态障碍物避让方面达到了最先进的性能，展示了与现有方法相比更为统计意义上的安全且保守度更低的操作。 

---
# IMPACT: Intelligent Motion Planning with Acceptable Contact Trajectories via Vision-Language Models 

**Title (ZH)**: 基于视觉-语言模型的智能可接受接触轨迹规划方法 

**Authors**: Yiyang Ling, Karan Owalekar, Oluwatobiloba Adesanya, Erdem Bıyık, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2503.10110)  

**Abstract**: Motion planning involves determining a sequence of robot configurations to reach a desired pose, subject to movement and safety constraints. Traditional motion planning finds collision-free paths, but this is overly restrictive in clutter, where it may not be possible for a robot to accomplish a task without contact. In addition, contacts range from relatively benign (e.g., brushing a soft pillow) to more dangerous (e.g., toppling a glass vase). Due to this diversity, it is difficult to characterize which contacts may be acceptable or unacceptable. In this paper, we propose IMPACT, a novel motion planning framework that uses Vision-Language Models (VLMs) to infer environment semantics, identifying which parts of the environment can best tolerate contact based on object properties and locations. Our approach uses the VLM's outputs to produce a dense 3D "cost map" that encodes contact tolerances and seamlessly integrates with standard motion planners. We perform experiments using 20 simulation and 10 real-world scenes and assess using task success rate, object displacements, and feedback from human evaluators. Our results over 3620 simulation and 200 real-world trials suggest that IMPACT enables efficient contact-rich motion planning in cluttered settings while outperforming alternative methods and ablations. Supplementary material is available at this https URL. 

**Abstract (ZH)**: 运动规划涉及确定机器人配置序列以达到所需姿态，并满足运动和安全约束。传统运动规划寻找无碰撞路径，但在复杂环境中，机器人可能无法完成任务而无法接触物体。此外，接触可以从相对温和（例如，轻触柔软的枕头）到更危险（例如，推翻玻璃花瓶）。由于这种多样性，很难确定哪些接触是可以接受还是不可以接受。在本文中，我们提出了IMPACT，这是一种新颖的运动规划框架，利用视觉-语言模型（VLMs）推断环境语义，根据物体属性和位置确定环境中哪些部分可以最好地容忍接触。我们的方法利用VLM的输出生成一个密集的3D“成本图”，编码接触容忍度，并无缝集成到标准运动规划器中。我们使用20个模拟场景和10个真实场景进行实验，并通过任务成功率、物体位移和人类评价者的反馈进行评估。我们的结果显示，在3620个模拟和200个真实场景试验中，IMPACT在复杂环境中共充分体现接触丰富的运动规划效率，并优于其他方法和消除实验。补充材料可在此处访问：this https URL。 

---
# Disentanglement Learning via Topology 

**Title (ZH)**: 拓扑引导的分离学习 

**Authors**: Nikita Balabin, Daria Voronkova, Ilya Trofimov, Evgeny Burnaev, Serguei Barannikov  

**Link**: [PDF](https://arxiv.org/pdf/2308.12696)  

**Abstract**: We propose TopDis (Topological Disentanglement), a method for learning disentangled representations via adding a multi-scale topological loss term. Disentanglement is a crucial property of data representations substantial for the explainability and robustness of deep learning models and a step towards high-level cognition. The state-of-the-art methods are based on VAE and encourage the joint distribution of latent variables to be factorized. We take a different perspective on disentanglement by analyzing topological properties of data manifolds. In particular, we optimize the topological similarity for data manifolds traversals. To the best of our knowledge, our paper is the first one to propose a differentiable topological loss for disentanglement learning. Our experiments have shown that the proposed TopDis loss improves disentanglement scores such as MIG, FactorVAE score, SAP score, and DCI disentanglement score with respect to state-of-the-art results while preserving the reconstruction quality. Our method works in an unsupervised manner, permitting us to apply it to problems without labeled factors of variation. The TopDis loss works even when factors of variation are correlated. Additionally, we show how to use the proposed topological loss to find disentangled directions in a trained GAN. 

**Abstract (ZH)**: TopDis（拓扑解纠缠）：一种通过添加多尺度拓扑损失项学习解纠缠表示的方法 

---
# Device-Robust Acoustic Scene Classification via Impulse Response Augmentation 

**Title (ZH)**: 基于冲击响应增强的设备鲁棒声场景分类 

**Authors**: Tobias Morocutti, Florian Schmid, Khaled Koutini, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2305.07499)  

**Abstract**: The ability to generalize to a wide range of recording devices is a crucial performance factor for audio classification models. The characteristics of different types of microphones introduce distributional shifts in the digitized audio signals due to their varying frequency responses. If this domain shift is not taken into account during training, the model's performance could degrade severely when it is applied to signals recorded by unseen devices. In particular, training a model on audio signals recorded with a small number of different microphones can make generalization to unseen devices difficult. To tackle this problem, we convolve audio signals in the training set with pre-recorded device impulse responses (DIRs) to artificially increase the diversity of recording devices. We systematically study the effect of DIR augmentation on the task of Acoustic Scene Classification using CNNs and Audio Spectrogram Transformers. The results show that DIR augmentation in isolation performs similarly to the state-of-the-art method Freq-MixStyle. However, we also show that DIR augmentation and Freq-MixStyle are complementary, achieving a new state-of-the-art performance on signals recorded by devices unseen during training. 

**Abstract (ZH)**: 音频分类模型广泛记录设备上的泛化能力是一项关键性能指标。不同类型的麦克风因频率响应的差异在其数字化音频信号中引入分布偏移。如果在训练过程中未考虑这种领域偏移，模型在应用到未见过设备记录的信号时性能可能会严重下降。特别是在使用少量不同麦克风录制的音频信号上训练模型，会使得模型对未见过设备的泛化变得困难。为解决这一问题，我们在训练集的音频信号中卷积预录制的设备冲激响应（DIRs），以人工增加录音设备的多样性。我们系统研究了DIR增强对使用CNN和Audio Spectrogram Transformers进行声场景分类任务的影响。结果表明，仅使用DIR增强的表现与现有最佳方法Freq-MixStyle相当。但我们还发现，DIR增强和Freq-MixStyle是互补的，能够在未见过设备记录的信号上实现新的最佳性能。 

---
