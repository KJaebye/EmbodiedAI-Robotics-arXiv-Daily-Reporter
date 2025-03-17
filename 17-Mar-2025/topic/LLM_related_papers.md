# A Framework for a Capability-driven Evaluation of Scenario Understanding for Multimodal Large Language Models in Autonomous Driving 

**Title (ZH)**: 一种基于能力驱动的多模态大型语言模型在自动驾驶中场景理解评估框架 

**Authors**: Tin Stribor Sohn, Philipp Reis, Maximilian Dillitzer, Johannes Bach, Jason J. Corso, Eric Sax  

**Link**: [PDF](https://arxiv.org/pdf/2503.11400)  

**Abstract**: Multimodal large language models (MLLMs) hold the potential to enhance autonomous driving by combining domain-independent world knowledge with context-specific language guidance. Their integration into autonomous driving systems shows promising results in isolated proof-of-concept applications, while their performance is evaluated on selective singular aspects of perception, reasoning, or planning. To leverage their full potential a systematic framework for evaluating MLLMs in the context of autonomous driving is required. This paper proposes a holistic framework for a capability-driven evaluation of MLLMs in autonomous driving. The framework structures scenario understanding along the four core capability dimensions semantic, spatial, temporal, and physical. They are derived from the general requirements of autonomous driving systems, human driver cognition, and language-based reasoning. It further organises the domain into context layers, processing modalities, and downstream tasks such as language-based interaction and decision-making. To illustrate the framework's applicability, two exemplary traffic scenarios are analysed, grounding the proposed dimensions in realistic driving situations. The framework provides a foundation for the structured evaluation of MLLMs' potential for scenario understanding in autonomous driving. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在结合领域无关的世界知识与上下文特定的语言指导方面，有望增强自动驾驶能力。将MLLMs整合到自动驾驶系统中在孤立的概念验证应用中显示出有前景的结果，同时对其感知、推理或规划的单一方面进行评估。为了充分发挥其潜力，自动驾驶场景中MLLMs能力驱动评估的系统框架是必需的。本文提出了一种综合框架，用于自动驾驶中MLLMs的能力驱动评估。该框架沿 semantics（语义）、spatial（空间）、temporal（时间）和 physical（物理）四个核心能力维度结构化场景理解。这些维度来源于自动驾驶系统的一般要求、人类驾驶员的认知以及语言推理。该框架进一步将领域划分为上下文层、处理模态和下游任务，如基于语言的交互和决策。为了说明该框架的应用性，分析了两个示例交通场景，使提出的思想扎根于实际驾驶情境。该框架为结构化评估MLLMs在自动驾驶场景理解中的潜力提供了基础。 

---
# Graph-Grounded LLMs: Leveraging Graphical Function Calling to Minimize LLM Hallucinations 

**Title (ZH)**: 基于图形的LLMs：通过图形函数调用来减轻LLM幻觉 

**Authors**: Piyush Gupta, Sangjae Bae, David Isele  

**Link**: [PDF](https://arxiv.org/pdf/2503.10941)  

**Abstract**: The adoption of Large Language Models (LLMs) is rapidly expanding across various tasks that involve inherent graphical structures. Graphs are integral to a wide range of applications, including motion planning for autonomous vehicles, social networks, scene understanding, and knowledge graphs. Many problems, even those not initially perceived as graph-based, can be effectively addressed through graph theory. However, when applied to these tasks, LLMs often encounter challenges, such as hallucinations and mathematical inaccuracies. To overcome these limitations, we propose Graph-Grounded LLMs, a system that improves LLM performance on graph-related tasks by integrating a graph library through function calls. By grounding LLMs in this manner, we demonstrate significant reductions in hallucinations and improved mathematical accuracy in solving graph-based problems, as evidenced by the performance on the NLGraph benchmark. Finally, we showcase a disaster rescue application where the Graph-Grounded LLM acts as a decision-support system. 

**Abstract (ZH)**: 大型语言模型在涉及内在图形结构的任务中的采用正迅速扩展。图形在广泛的应用中发挥着重要作用，包括自主车辆的运动规划、社交网络、场景理解以及知识图谱。许多问题，即使是最初不被视为基于图的问题，也能通过图理论得到有效解决。然而，当应用于这些任务时，大型语言模型常常会遇到幻觉和数学准确性不足等挑战。为克服这些限制，我们提出了一种图本位的大型语言模型系统，通过函数调用集成图形库以提高大型语言模型在与图相关的任务上的性能。通过这种图本位的方法，我们展示了显著降低幻觉并提高解决基于图的问题的数学准确性，这在NLGraph基准测试中得到了体现。最后，我们展示了在灾害救援应用中，图本位的大型语言模型作为决策支持系统的应用。 

---
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
# Integrating LLMs in Gamified Systems 

**Title (ZH)**: 在游戏化系统中集成大语言模型 

**Authors**: Carlos J. Costa  

**Link**: [PDF](https://arxiv.org/pdf/2503.11458)  

**Abstract**: In this work, a thorough mathematical framework for incorporating Large Language Models (LLMs) into gamified systems is presented with an emphasis on improving task dynamics, user engagement, and reward systems. Personalized feedback, adaptive learning, and dynamic content creation are all made possible by integrating LLMs and are crucial for improving user engagement and system performance. A simulated environment tests the framework's adaptability and demonstrates its potential for real-world applications in various industries, including business, healthcare, and education. The findings demonstrate how LLMs can offer customized experiences that raise system effectiveness and user retention. This study also examines the difficulties this framework aims to solve, highlighting its importance in maximizing involvement and encouraging sustained behavioral change in a range of sectors. 

**Abstract (ZH)**: 在这种工作中，提出了一个全面的数学框架，将大型语言模型（LLMs）纳入 gamified 系统，并强调改进任务动力学、用户参与度和奖励系统。通过集成 LLMs，个性化反馈、自适应学习和动态内容创建成为可能，对于提高用户参与度和系统性能至关重要。模拟环境测试了该框架的适应性，并展示了其在多个行业（包括商业、医疗保健和教育）中实现实际应用的潜力。研究结果表明，LLMs 可以提供定制化的体验，提高系统的有效性和用户留存率。此外，该研究还探讨了框架旨在解决的困难，强调了其在各行业中最大化参与度和促进持续行为改变的重要性。 

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
# Potential of large language model-powered nudges for promoting daily water and energy conservation 

**Title (ZH)**: 大型语言模型驱动的提示在促进日常节水和节能方面的潜力 

**Authors**: Zonghan Li, Song Tong, Yi Liu, Kaiping Peng, Chunyan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11531)  

**Abstract**: The increasing amount of pressure related to water and energy shortages has increased the urgency of cultivating individual conservation behaviors. While the concept of nudging, i.e., providing usage-based feedback, has shown promise in encouraging conservation behaviors, its efficacy is often constrained by the lack of targeted and actionable content. This study investigates the impact of the use of large language models (LLMs) to provide tailored conservation suggestions for conservation intentions and their rationale. Through a survey experiment with 1,515 university participants, we compare three virtual nudging scenarios: no nudging, traditional nudging with usage statistics, and LLM-powered nudging with usage statistics and personalized conservation suggestions. The results of statistical analyses and causal forest modeling reveal that nudging led to an increase in conservation intentions among 86.9%-98.0% of the participants. LLM-powered nudging achieved a maximum increase of 18.0% in conservation intentions, surpassing traditional nudging by 88.6%. Furthermore, structural equation modeling results reveal that exposure to LLM-powered nudges enhances self-efficacy and outcome expectations while diminishing dependence on social norms, thereby increasing intrinsic motivation to conserve. These findings highlight the transformative potential of LLMs in promoting individual water and energy conservation, representing a new frontier in the design of sustainable behavioral interventions and resource management. 

**Abstract (ZH)**: 大语言模型在促进个体水资源和能源 conservation 方面的影响：基于调查实验的研究 

---
# Cerebrum (AIOS SDK): A Platform for Agent Development, Deployment, Distribution, and Discovery 

**Title (ZH)**: Cerebrum (AIOS SDK): 一个代理开发、部署、分发和发现的平台 

**Authors**: Balaji Rama, Kai Mei, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11444)  

**Abstract**: Autonomous LLM-based agents have emerged as a powerful paradigm for complex task execution, yet the field lacks standardized tools for development, deployment, distribution and discovery of agents. We present Cerebrum, an Agent SDK for AIOS that addresses this gap through three key components: (1) a comprehensive SDK featuring a modular four-layer architecture for agent development, encompassing LLM, memory, storage, and tool management; (2) a community-driven Agent Hub for sharing and discovering agents, complete with version control and dependency management; (3) an interactive web interface for testing and evaluating agents. The platform's effectiveness is demonstrated through implementations of various agent architectures, including Chain of Thought (CoT), ReAct, and tool-use agents. Cerebrum advances the field by providing a unified framework that standardizes agent development while maintaining flexibility for researchers and developers to innovate and distribute their agents. The live website is at this https URL, the code is at this https URL, and video is at this https URL. 

**Abstract (ZH)**: 自主基于LLM的代理已成为复杂任务执行的强大范式，然而该领域缺乏用于代理开发、部署、分发和发现的标准化工具。我们介绍了Cerebrum，一个为AIOS设计的代理SDK，通过三大关键组件解决上述问题：（1）一个全面的SDK，包含模块化的四层架构，涵盖LLM、记忆、存储和工具管理；（2）一个社区驱动的代理枢纽，支持代理的分享和发现，包含版本控制和依赖管理；（3）一个交互式网页界面，用于测试和评估代理。平台的有效性通过各种代理架构的实现得到验证，包括链式思考（CoT）、ReAct 和工具使用代理。Cerebrum 通过提供一个统一框架，既标准化了代理开发，又为研究人员和开发者保留了创新和分发代理的灵活性。网址、代码和视频分别为：这个 https URL、这个 https URL 和这个 https URL。 

---
# Line of Duty: Evaluating LLM Self-Knowledge via Consistency in Feasibility Boundaries 

**Title (ZH)**: 尽责线：通过可行性边界一致性评估LLM的自我认知能力 

**Authors**: Sahil Kale, Vijaykant Nadadur  

**Link**: [PDF](https://arxiv.org/pdf/2503.11256)  

**Abstract**: As LLMs grow more powerful, their most profound achievement may be recognising when to say "I don't know". Existing studies on LLM self-knowledge have been largely constrained by human-defined notions of feasibility, often neglecting the reasons behind unanswerability by LLMs and failing to study deficient types of self-knowledge. This study aims to obtain intrinsic insights into different types of LLM self-knowledge with a novel methodology: allowing them the flexibility to set their own feasibility boundaries and then analysing the consistency of these limits. We find that even frontier models like GPT-4o and Mistral Large are not sure of their own capabilities more than 80% of the time, highlighting a significant lack of trustworthiness in responses. Our analysis of confidence balance in LLMs indicates that models swing between overconfidence and conservatism in feasibility boundaries depending on task categories and that the most significant self-knowledge weaknesses lie in temporal awareness and contextual understanding. These difficulties in contextual comprehension additionally lead models to question their operational boundaries, resulting in considerable confusion within the self-knowledge of LLMs. We make our code and results available publicly at this https URL 

**Abstract (ZH)**: 随着大规模语言模型变得愈发强大，它们最深刻的成就可能是认识到何时应该说“我不知道”。现有的关于语言模型自我认知的研究大多受限于人类定义的可行性观念，往往忽视了语言模型无法回答问题的原因，并未研究缺陷类型的自我认知。本研究旨在通过一种新颖的方法论获得不同类型语言模型自我认知的内在洞见：允许它们自行设定可行性边界，然后分析这些边界的连贯性。我们发现即使是前沿模型如GPT-4o和Mistral Large也有超过80%的时间对自己能力不确定，突显了响应中的显著不可靠性。我们对语言模型信心平衡的分析表明，模型在可行性边界上会根据任务类别在过度自信和保守之间摇摆，自我认知中最显著的弱点在于时间意识和上下文理解。这些在上下文理解上的困难还导致模型质疑其操作边界，从而在语言模型的自我认知中造成了极大的混乱。我们将在以下网址公开我们的代码和结果：这个 https URL。 

---
# Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering 

**Title (ZH)**: 强化学习优于监督微调：音频问答案例研究 

**Authors**: Gang Li, Jizhong Liu, Heinrich Dinkel, Yadong Niu, Junbo Zhang, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2503.11197)  

**Abstract**: Recently, reinforcement learning (RL) has been shown to greatly enhance the reasoning capabilities of large language models (LLMs), and RL-based approaches have been progressively applied to visual multimodal tasks. However, the audio modality has largely been overlooked in these developments. Thus, we conduct a series of RL explorations in audio understanding and reasoning, specifically focusing on the audio question answering (AQA) task. We leverage the group relative policy optimization (GRPO) algorithm to Qwen2-Audio-7B-Instruct, and our experiments demonstrated state-of-the-art performance on the MMAU Test-mini benchmark, achieving an accuracy rate of 64.5%. The main findings in this technical report are as follows: 1) The GRPO algorithm can be effectively applied to large audio language models (LALMs), even when the model has only 8.2B parameters; 2) With only 38k post-training samples, RL significantly outperforms supervised fine-tuning (SFT), indicating that RL-based approaches can be effective without large datasets; 3) The explicit reasoning process has not shown significant benefits for AQA tasks, and how to efficiently utilize deep thinking remains an open question for further research; 4) LALMs still lag far behind humans auditory-language reasoning, suggesting that the RL-based approaches warrant further exploration. Our project is available at this https URL and this https URL. 

**Abstract (ZH)**: 最近，强化学习(RL)已被证明大幅提升了大型语言模型(LLMs)的推理能力，基于RL的方法逐渐被应用于视觉多模态任务中。然而，音频模态在这方面的进展被严重忽视。因此，我们在音频理解与推理方面进行了一系列RL探索，特别关注音频问答(AQA)任务。我们利用群相对策略优化(GRPO)算法对Qwen2-Audio-7B-Instruct进行了优化，并在MMAU Test-mini基准测试中实现了64.5%的准确率，展示了目前最先进的性能。本技术报告的主要发现如下：1) GRPO算法可以有效地应用于大型音频语言模型(LALMs)，即使模型仅有8.2B参数；2) 仅使用38k训练后样本，RL显著优于监督微调(SFT)，表明基于RL的方法可以在无需大规模数据集的情况下有效；3) 显式的推理过程并未在AQA任务中显示出显著优势，如何高效利用深度思考仍是一个待解决的问题；4) LALMs在听觉语言推理方面仍然远远落后于人类，表明基于RL的方法仍需进一步探索。我们的项目可访问此URL和此URL。 

---
# Align in Depth: Defending Jailbreak Attacks via Progressive Answer Detoxification 

**Title (ZH)**: 深度对齐：通过渐进式答案去毒来防御 Jailbreak 攻击 

**Authors**: Yingjie Zhang, Tong Liu, Zhe Zhao, Guozhu Meng, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11185)  

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks, which use crafted prompts to elicit toxic responses. These attacks exploit LLMs' difficulty in dynamically detecting harmful intents during the generation process. Traditional safety alignment methods, often relying on the initial few generation steps, are ineffective due to limited computational budget. This paper proposes DEEPALIGN, a robust defense framework that fine-tunes LLMs to progressively detoxify generated content, significantly improving both the computational budget and effectiveness of mitigating harmful generation. Our approach uses a hybrid loss function operating on hidden states to directly improve LLMs' inherent awareness of toxity during generation. Furthermore, we redefine safe responses by generating semantically relevant answers to harmful queries, thereby increasing robustness against representation-mutation attacks. Evaluations across multiple LLMs demonstrate state-of-the-art defense performance against six different attack types, reducing Attack Success Rates by up to two orders of magnitude compared to previous state-of-the-art defense while preserving utility. This work advances LLM safety by addressing limitations of conventional alignment through dynamic, context-aware mitigation. 

**Abstract (ZH)**: 大型语言模型（LLMs）易受监狱突破攻击的威胁，这种攻击利用精心设计的提示诱使模型产生有害响应。这些攻击利用了LLMs在生成过程中难以动态检测有害意图的困难。传统的安全对齐方法往往依赖于初始的少量生成步骤，但由于计算预算有限而无效。本文提出了一种名为DEEPALIGN的鲁棒防御框架，通过微调LLMs使其逐步净化生成的内容，显著提高了计算预算和缓解有害生成的有效性。我们的方法使用一个在隐藏状态上操作的混合损失函数，直接提升LLMs在生成过程中对毒性的内在感知。此外，我们通过为有害查询生成语义相关答案重新定义安全响应，从而增强对表征突变攻击的鲁棒性。多种LLMs的评估表明，DEEPALIGN在六种不同攻击类型的防护性能上达到了最先进的技术水平，相较于之前的最先进防护方法将攻击成功率降低了两个数量级以上，同时保持了实用性。这项工作通过动态、上下文感知的缓解措施，推动了LLMs安全性的提升，解决了传统对齐方法的局限性。 

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
# Don't Forget It! Conditional Sparse Autoencoder Clamping Works for Unlearning 

**Title (ZH)**: 别忘了它！条件稀疏自编码器钳制法适用于遗忘学习 

**Authors**: Matthew Khoriaty, Andrii Shportko, Gustavo Mercier, Zach Wood-Doughty  

**Link**: [PDF](https://arxiv.org/pdf/2503.11127)  

**Abstract**: Recent developments in Large Language Model (LLM) capabilities have brought great potential but also posed new risks. For example, LLMs with knowledge of bioweapons, advanced chemistry, or cyberattacks could cause violence if placed in the wrong hands or during malfunctions. Because of their nature as near-black boxes, intuitive interpretation of LLM internals remains an open research question, preventing developers from easily controlling model behavior and capabilities. The use of Sparse Autoencoders (SAEs) has recently emerged as a potential method of unraveling representations of concepts in LLMs internals, and has allowed developers to steer model outputs by directly modifying the hidden activations. In this paper, we use SAEs to identify unwanted concepts from the Weapons of Mass Destruction Proxy (WMDP) dataset within gemma-2-2b internals and use feature steering to reduce the model's ability to answer harmful questions while retaining its performance on harmless queries. Our results bring back optimism to the viability of SAE-based explicit knowledge unlearning techniques. 

**Abstract (ZH)**: 近期大型语言模型（LLM）能力的发展带来了巨大潜力但也提出了新的风险。由于其近似黑箱的性质，对LLM内部机制的直观解释仍然是一个开放的研究问题，阻碍了开发人员轻松控制模型行为和能力。最近，稀疏自编码器（SAEs）的使用作为一种可能的方法来解开LLM内部概念的表示，并允许开发人员通过直接修改隐藏激活来引导模型输出。在本文中，我们使用SAEs在gemma-2-2b内部识别来自大规模破坏性武器代理（WMDP）数据集的不希望的概念，并通过特征引导来降低模型回答有害问题的能力，同时保留其在无害查询上的性能。我们的结果重新唤起了基于SAE的显式知识遗忘技术可行性的乐观态度。 

---
# Limits of KV Cache Compression for Tensor Attention based Autoregressive Transformers 

**Title (ZH)**: 基于张量注意力的自回归变换器中键值缓存压缩的极限 

**Authors**: Yifang Chen, Xiaoyu Li, Yingyu Liang, Zhenmei Shi, Zhao Song, Yu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2503.11108)  

**Abstract**: The key-value (KV) cache in autoregressive transformers presents a significant bottleneck during inference, which restricts the context length capabilities of large language models (LLMs). While previous work analyzes the fundamental space complexity barriers in standard attention mechanism [Haris and Onak, 2025], our work generalizes the space complexity barriers result to tensor attention version. Our theoretical contributions rely on a novel reduction from communication complexity and deduce the memory lower bound for tensor-structured attention mechanisms when $d = \Omega(\log n)$. In the low dimensional regime where $d = o(\log n)$, we analyze the theoretical bounds of the space complexity as well. Overall, our work provides a theoretical foundation for us to understand the compression-expressivity tradeoff in tensor attention mechanisms and offers more perspectives in developing more memory-efficient transformer architectures. 

**Abstract (ZH)**: 自回归变压器中的键值缓存对推理过程构成显著瓶颈，限制了大型语言模型的上下文长度能力。尽管以往工作分析了标准注意机制的基本空间复杂度障碍[Haris和Onak, 2025]，我们的工作将空间复杂度障碍的结果推广到张量注意版本。我们的理论贡献基于一种新颖的通信复杂度归约，并推导出当 $d = \Omega(\log n)$ 时张量结构注意机制的存储下界。在低维情形下，即 $d = o(\log n)$，我们分析了空间复杂度的理论界限。总体而言，我们的工作为我们理解张量注意机制中的压缩-表达性权衡提供了理论基础，并提供了开发更高效的变压器架构的新视角。 

---
# Augmenting Image Annotation: A Human-LMM Collaborative Framework for Efficient Object Selection and Label Generation 

**Title (ZH)**: 增强图像标注：一种人类-机器学习协作框架，用于高效对象选择和标签生成 

**Authors**: He Zhang, Xinyi Fu, John M. Carroll  

**Link**: [PDF](https://arxiv.org/pdf/2503.11096)  

**Abstract**: Traditional image annotation tasks rely heavily on human effort for object selection and label assignment, making the process time-consuming and prone to decreased efficiency as annotators experience fatigue after extensive work. This paper introduces a novel framework that leverages the visual understanding capabilities of large multimodal models (LMMs), particularly GPT, to assist annotation workflows. In our proposed approach, human annotators focus on selecting objects via bounding boxes, while the LMM autonomously generates relevant labels. This human-AI collaborative framework enhances annotation efficiency by reducing the cognitive and time burden on human annotators. By analyzing the system's performance across various types of annotation tasks, we demonstrate its ability to generalize to tasks such as object recognition, scene description, and fine-grained categorization. Our proposed framework highlights the potential of this approach to redefine annotation workflows, offering a scalable and efficient solution for large-scale data labeling in computer vision. Finally, we discuss how integrating LMMs into the annotation pipeline can advance bidirectional human-AI alignment, as well as the challenges of alleviating the "endless annotation" burden in the face of information overload by shifting some of the work to AI. 

**Abstract (ZH)**: 传统图像标注任务高度依赖人工选择对象和分配标签，这使得过程耗时且容易因标注员长时间工作而降低效率。本文介绍了一种新的框架，利用大型多模式模型（LMM）特别是GPT的视觉理解能力来辅助标注流程。在我们提出的方法中，人类标注员专注于通过边界框选择对象，而LMM自主生成相关标签。这种人机协作框架通过减轻人类标注员的认知和时间负担来提高标注效率。通过分析系统在各种标注任务中的性能，我们展示了其迁移到对象识别、场景描述和细粒度分类等任务的能力。我们的框架突显了该方法重新定义标注流程的潜力，提供了一种适用于计算机视觉大规模数据标签化的可扩展且高效解决方案。最后，我们讨论了将LMM集成到标注流水线中如何促进双向的人机对齐，以及如何通过将部分工作转移给AI来缓解信息过载导致的“无尽标注”负担。 

---
# RONA: Pragmatically Diverse Image Captioning with Coherence Relations 

**Title (ZH)**: RONA：具有一致性关系的实用多样化图像_captioning 

**Authors**: Aashish Anantha Ramakrishnan, Aadarsh Anantha Ramakrishnan, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.10997)  

**Abstract**: Writing Assistants (e.g., Grammarly, Microsoft Copilot) traditionally generate diverse image captions by employing syntactic and semantic variations to describe image components. However, human-written captions prioritize conveying a central message alongside visual descriptions using pragmatic cues. To enhance pragmatic diversity, it is essential to explore alternative ways of communicating these messages in conjunction with visual content. To address this challenge, we propose RONA, a novel prompting strategy for Multi-modal Large Language Models (MLLM) that leverages Coherence Relations as an axis for variation. We demonstrate that RONA generates captions with better overall diversity and ground-truth alignment, compared to MLLM baselines across multiple domains. Our code is available at: this https URL 

**Abstract (ZH)**: 写作助手（例如Grammarly、Microsoft Copilot）传统上通过语法和语义变化生成多样的图像描述。然而，人类撰写的描述更侧重于通过 pragmatics 提示传达中心信息并结合视觉描述。为了增强 pragmatics 多样性，有必要探索与视觉内容相结合的替代沟通方式。为应对这一挑战，我们提出了一种名为 RONA 的新型 Multi-modal 大型语言模型 (MLLM) 激励策略，该策略利用一致性关系作为变化轴。我们证明，与 MLLM 基线相比，RONA 生成的描述在多个领域具有更好的总体多样性和真实度匹配。我们的代码可在以下链接获取：this https URL。 

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
# Vulnerability Detection: From Formal Verification to Large Language Models and Hybrid Approaches: A Comprehensive Overview 

**Title (ZH)**: 漏洞检测：从形式验证到大型语言模型和混合方法的全面概述 

**Authors**: Norbert Tihanyi, Tamas Bisztray, Mohamed Amine Ferrag, Bilel Cherif, Richard A. Dubniczky, Ridhi Jain, Lucas C. Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2503.10784)  

**Abstract**: Software testing and verification are critical for ensuring the reliability and security of modern software systems. Traditionally, formal verification techniques, such as model checking and theorem proving, have provided rigorous frameworks for detecting bugs and vulnerabilities. However, these methods often face scalability challenges when applied to complex, real-world programs. Recently, the advent of Large Language Models (LLMs) has introduced a new paradigm for software analysis, leveraging their ability to understand insecure coding practices. Although LLMs demonstrate promising capabilities in tasks such as bug prediction and invariant generation, they lack the formal guarantees of classical methods. This paper presents a comprehensive study of state-of-the-art software testing and verification, focusing on three key approaches: classical formal methods, LLM-based analysis, and emerging hybrid techniques, which combine their strengths. We explore each approach's strengths, limitations, and practical applications, highlighting the potential of hybrid systems to address the weaknesses of standalone methods. We analyze whether integrating formal rigor with LLM-driven insights can enhance the effectiveness and scalability of software verification, exploring their viability as a pathway toward more robust and adaptive testing frameworks. 

**Abstract (ZH)**: 软件测试与验证对于确保现代软件系统的可靠性和安全性至关重要。传统形式化验证技术，如模型检测和定理证明，提供了检测错误和漏洞的严格框架。然而，当应用于复杂的现实程序时，这些方法往往面临可扩展性挑战。最近，大型语言模型（LLMs）的出现引入了软件分析的新范式，利用其理解不安全编码实践的能力。尽管LLMs在错误预测和不变式生成等任务上表现出有前景的能力，但它们缺乏经典方法的形式保证。本文对最先进的软件测试与验证进行了全面研究，重点关注三种关键方法：经典形式化方法、基于LLM的分析以及新兴的混合技术，这些技术结合了各自的优势。我们探讨了每种方法的优势、局限性和实际应用，突出了混合系统的潜力，以弥补单方法的弱点。我们分析了将形式化严谨性与LLM驱动的见解相结合是否能够增强软件验证的有效性和可扩展性，探索其作为更稳健和自适应测试框架途径的可行性。 

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
# ZeroMerge: Parameter-Free KV Cache Compression for Memory-Efficient Long-Context LLMs 

**Title (ZH)**: ZeroMerge: 参数_free_键值缓存压缩技术以实现内存高效的长上下文LLM 

**Authors**: Xin Liu, Pei Liu, Guoming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10714)  

**Abstract**: The linear growth of key-value (KV) cache memory and quadratic computational complexity pose significant bottlenecks for large language models (LLMs) in long-context processing. While existing KV cache optimization methods address these challenges through token pruning or feature merging, they often suffer from irreversible information loss or require costly parameter retraining. We propose ZeroMerge, a dynamic zero-shot compression framework that achieves efficient cache management through three key innovations: (1) Fine-grained memory allocation guided by multi-dimensional token importance metrics at head-level granularity, (2) A residual merging mechanism that preserves critical context through compensated attention scoring, and (3) Parameter-free adaptation compatible with diverse LLM architectures without retraining. Comprehensive evaluations across LLaMA-2 model demonstrate that ZeroMerge maintains full-cache performance at 5\% compression ratios while doubling inference throughput at 40K token lengths. The method effectively balances memory efficiency, generation quality, and deployment flexibility, advancing practical long-context LLM applications. The code is available at this https URL. 

**Abstract (ZH)**: 键值缓存内存的线性增长和计算复杂性的二次增长为大规模语言模型在长上下文处理中的应用造成了显著瓶颈。现有的键值缓存优化方法通过标记修剪或特征合并解决这些挑战，但往往会导致不可逆的信息丢失或需要昂贵的参数重新训练。我们提出ZeroMerge，一种动态零-shot压缩框架，通过以下三大创新实现高效缓存管理：（1）基于头部层面多维度标记重要性指标的细粒度内存分配；（2）残差合并机制，通过补偿注意评分保留关键上下文；（3）无需重新训练的参数免费适应，适用于多种大规模语言模型架构。全面评估显示，ZeroMerge在5%压缩比下维持全缓存性能，同时在40K标记长度下使推理吞吐量翻倍。该方法有效平衡了内存效率、生成质量和部署灵活性，推动实际长上下文大规模语言模型应用的发展。代码可在下方链接获取。 

---
# CALLM: Context-Aware Emotion Analysis in Cancer Survivors Using LLMs and Retrieval-Augmented Mobile Diaries 

**Title (ZH)**: CALLM：基于上下文的情感分析在癌症幸存者移动日记中的应用与检索增强 

**Authors**: Zhiyuan Wang, Katharine E. Daniel, Laura E. Barnes, Philip I. Chow  

**Link**: [PDF](https://arxiv.org/pdf/2503.10707)  

**Abstract**: Cancer survivors face unique emotional challenges that impact their quality of life. Mobile diary entries-short text entries recording through their phone about their emotional experiences-provide a promising method for tracking these experiences in real time. Although emotion analysis tools show potential for recognizing emotions from text, current methods lack the contextual understanding necessary to accurately interpret the brief, personal narratives in mobile diaries. We propose CALLM, a context-aware emotion analysis framework that leverages Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG), to analyze mobile diary entries from cancer survivors to predict their emotional states. The framework enhances prediction accuracy beyond existing methods by (1) integrating retrieved peer experiences as contextual examples and (2) incorporating individuals' temporal emotional trajectories from their mobile diary entries. We collected a large-scale dataset (N=407) of cancer survivors' mobile ecological momentary assessments (EMAs), which assessed positive and negative affect, desire to regulate emotions, social interaction quality, and availability for interventions, alongside daily mobile diary entries in an open response format regarding what was driving their current emotional experience. Results demonstrate strong performance of CALLM, with balanced accuracies reaching 72.96% for positive and 73.29% for negative affect, and 73.72% for predicting individual's desire to regulate emotions. Post-hoc analysis reveals that leveraging model confidence, encouraging longer diary entries, and incorporating personal ground truth, further enhance predictive outcomes. Our findings support the feasibility of deploying LLM-powered emotion analysis in chronic health populations and suggest promising directions for personalized interventions for cancer survivors. 

**Abstract (ZH)**: 癌症幸存者面临独特的情绪挑战，影响其生活质量。手机日记条目通过手机记录关于他们情绪体验的简短文本条目，为实时追踪这些体验提供了有前景的方法。尽管情绪分析工具显示出从文本中识别情绪的潜力，但当前方法缺乏准确解读手机日记中简短个人叙事所需的情境理解。我们提出CALLM，一种基于检索增强生成（RAG）的大语言模型（LLM）的情境感知情绪分析框架，用于分析癌症幸存者的手机日记条目以预测其情绪状态。该框架通过（1）整合检索到的同伴体验作为情境示例，以及（2）结合个人随时间的情绪轨迹，超越现有方法提高预测准确性。我们收集了407名癌症幸存者的大量数据集，评估了正负情绪、情绪调节愿望、社交互动质量以及干预可用性，并以开放式回答格式记录了影响他们当前情绪体验的因素。结果表明CALLM表现出色，正负情绪的均衡准确率分别为72.96%和73.29%，预测个人情绪调节愿望的准确率为73.72%。事后分析显示，利用模型信心、鼓励更长的日记条目以及结合个人真实情况，可进一步提升预测效果。我们的研究支持在慢性病人群部署由大语言模型支持的情绪分析的可行性，并建议针对癌症幸存者个性化干预的潜在方向。 

---
# Understanding the Quality-Diversity Trade-off in Diffusion Language Models 

**Title (ZH)**: 理解扩散语言模型中的质量-多样性权衡 

**Authors**: Zak Buzzard  

**Link**: [PDF](https://arxiv.org/pdf/2503.10683)  

**Abstract**: Diffusion models have seen immense success in modelling continuous data across a range of domains such as vision and audio. Despite the challenges of adapting diffusion models to discrete data, recent work explores their application to text generation by working in the continuous embedding space. However, these models lack a natural means to control the inherent trade-off between quality and diversity as afforded by the temperature hyperparameter in autoregressive models, hindering understanding of model performance and restricting generation quality. This work proposes the use of classifier-free guidance and stochastic clamping for manipulating the quality-diversity trade-off on sequence-to-sequence tasks, demonstrating that these techniques may be used to improve the performance of a diffusion language model. 

**Abstract (ZH)**: 基于分类器-free 指导和随机clamp的可控质量-多样性 trade-off 方法在序列生成任务中的应用 

---
# Fine-Tuning LLMs for Report Summarization: Analysis on Supervised and Unsupervised Data 

**Title (ZH)**: 基于监督和无监督数据的大型语言模型报告总结微调分析 

**Authors**: Swati Rallapalli, Shannon Gallagher, Andrew O. Mellinger, Jasmine Ratchford, Anusha Sinha, Tyler Brooks, William R. Nichols, Nick Winski, Bryan Brown  

**Link**: [PDF](https://arxiv.org/pdf/2503.10676)  

**Abstract**: We study the efficacy of fine-tuning Large Language Models (LLMs) for the specific task of report (government archives, news, intelligence reports) summarization. While this topic is being very actively researched - our specific application set-up faces two challenges: (i) ground-truth summaries maybe unavailable (e.g., for government archives), and (ii) availability of limited compute power - the sensitive nature of the application requires that computation is performed on-premise and for most of our experiments we use one or two A100 GPU cards. Under this set-up we conduct experiments to answer the following questions. First, given that fine-tuning the LLMs can be resource intensive, is it feasible to fine-tune them for improved report summarization capabilities on-premise? Second, what are the metrics we could leverage to assess the quality of these summaries? We conduct experiments on two different fine-tuning approaches in parallel and our findings reveal interesting trends regarding the utility of fine-tuning LLMs. Specifically, we find that in many cases, fine-tuning helps improve summary quality and in other cases it helps by reducing the number of invalid or garbage summaries. 

**Abstract (ZH)**: 我们研究了对大型语言模型（LLMs）进行微调以特定任务（政府档案、新闻、情报报告摘要）的有效性。在这一研究主题非常活跃的情况下，我们的具体应用场景面临两个挑战：（i） ground-truth摘要可能不可用（例如，对于政府档案），（ii）有限的计算能力要求计算必须在本地进行，且在多数实验中我们使用了1到2块A100 GPU卡。在这种设置下，我们进行实验以回答以下问题。首先，考虑到微调LLMs可能资源密集，是否可以在本地进行微调以提高报告摘要能力？其次，我们能够利用哪些指标来评估这些摘要的质量？我们并行开展了两种不同的微调方法的实验，研究结果揭示了关于微调LLMs的实用性的一些有趣趋势。具体而言，我们发现，在许多情况下，微调有助于提高摘要质量，在其他情况下，它通过减少无效或垃圾摘要的数量来改进。 

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
# Evaluation of the Automated Labeling Method for Taxonomic Nomenclature Through Prompt-Optimized Large Language Model 

**Title (ZH)**: 通过提示优化大型语言模型对分类命名自动标注方法的评价 

**Authors**: Keito Inoshita, Kota Nojiri, Haruto Sugeno, Takumi Taga  

**Link**: [PDF](https://arxiv.org/pdf/2503.10662)  

**Abstract**: Scientific names of organisms consist of a genus name and a species epithet, with the latter often reflecting aspects such as morphology, ecology, distribution, and cultural background. Traditionally, researchers have manually labeled species names by carefully examining taxonomic descriptions, a process that demands substantial time and effort when dealing with large datasets. This study evaluates the feasibility of automatic species name labeling using large language model (LLM) by leveraging their text classification and semantic extraction capabilities. Using the spider name dataset compiled by Mammola et al., we compared LLM-based labeling results-enhanced through prompt engineering-with human annotations. The results indicate that LLM-based classification achieved high accuracy in Morphology, Geography, and People categories. However, classification accuracy was lower in Ecology & Behavior and Modern & Past Culture, revealing challenges in interpreting animal behavior and cultural contexts. Future research will focus on improving accuracy through optimized few-shot learning and retrieval-augmented generation techniques, while also expanding the applicability of LLM-based labeling to diverse biological taxa. 

**Abstract (ZH)**: 生物体的科学名称由属名和种 epithet 构成，后面的种 epithet 往往反映形态学、生态学、地理分布和文化背景等方面的特征。传统上，研究人员通过仔细检查分类学描述手动标注物种名称，这在处理大量数据集时需要大量时间和 effort。本文评估了利用大型语言模型（LLM）通过利用其文本分类和语义提取能力自动标注物种名称的可行性。通过使用 Mammola 等人编纂的蜘蛛名称数据集，我们将基于 LLM 的标注结果（通过提示工程增强）与人工注释进行了比较。结果表明，基于 LLM 的分类在形态学、地理分布和人群类别中表现出了高准确性。然而，在生态学与行为学及现代与过去文化类别中的分类准确性较低，这表明在解读动物行为和文化背景方面存在挑战。未来的研究将集中在通过优化的少样本学习和检索增强生成技术来提高准确性，同时扩大基于 LLM 的标注在不同生物学类群中的应用范围。 

---
# RouterEval: A Comprehensive Benchmark for Routing LLMs to Explore Model-level Scaling Up in LLMs 

**Title (ZH)**: RouterEval：一个全面的基准测试，用于路由LLM模型以探索LLM的模型级扩展 

**Authors**: Zhongzhan Huang, Guoming Ling, Vincent S. Liang, Yupei Lin, Yandong Chen, Shanshan Zhong, Hefeng Wu, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.10657)  

**Abstract**: Routing large language models (LLMs) is a novel paradigm that recommends the most suitable LLM from a pool of candidates to process a given input through a well-designed router. Our comprehensive analysis reveals a model-level scaling-up phenomenon in LLMs, i.e., a capable router can significantly enhance the performance of this paradigm as the number of candidates increases. This improvement can even easily surpass the performance of the best single model in the pool and most existing strong LLMs, making it a highly promising paradigm. However, the lack of comprehensive and open-source benchmarks for Routing LLMs has hindered the development of routers. In this paper, we introduce RouterEval, a benchmark designed specifically for router research, which includes over 200,000,000 performance records for 12 popular LLM evaluations across areas such as knowledge-based Q&A, commonsense reasoning, semantic understanding, mathematical reasoning, and instruction following, based on more than 8,500 LLMs. Using RouterEval, extensive evaluations of existing Routing LLM methods reveal that most still have significant room for improvement. See this https URL for all data, code, and tutorials. 

**Abstract (ZH)**: 大型语言模型路由（Routing large language models）是一种新颖的范式，它通过一个精心设计的路由器从候选模型池中推荐最适合处理给定输入的模型。我们的全面分析揭示了大型语言模型在模型级别的扩展现象，即一个能力强的路由器可以显著增强该范式的性能，随着候选模型数量的增加，这种改进甚至可以轻易超过池中最佳单个模型和大多数现有强大型语言模型的表现，使其成为一个极具前景的范式。然而，缺乏全面且开源的大规模语言模型路由基准阻碍了路由器的发展。在本文中，我们介绍了RouterEval，一个专门为路由器研究设计的基准，其中包括超过2亿条性能记录，覆盖12个流行的大规模语言模型评估领域，如基于知识的问答、常识推理、语义理解、数学推理和指令跟随，基于超过8,500个大规模语言模型。使用RouterEval，对现有大规模语言模型路由方法进行了广泛的评估，发现大多数方法仍然有很大的改进空间。所有数据、代码和教程请访问：See this https URL。 

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
