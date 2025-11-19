# Masked IRL: LLM-Guided Reward Disambiguation from Demonstrations and Language 

**Title (ZH)**: 掩码IRL：由示例和语言引导的奖励歧义解析 

**Authors**: Minyoung Hwang, Alexandra Forsey-Smerek, Nathaniel Dennler, Andreea Bobu  

**Link**: [PDF](https://arxiv.org/pdf/2511.14565)  

**Abstract**: Robots can adapt to user preferences by learning reward functions from demonstrations, but with limited data, reward models often overfit to spurious correlations and fail to generalize. This happens because demonstrations show robots how to do a task but not what matters for that task, causing the model to focus on irrelevant state details. Natural language can more directly specify what the robot should focus on, and, in principle, disambiguate between many reward functions consistent with the demonstrations. However, existing language-conditioned reward learning methods typically treat instructions as simple conditioning signals, without fully exploiting their potential to resolve ambiguity. Moreover, real instructions are often ambiguous themselves, so naive conditioning is unreliable. Our key insight is that these two input types carry complementary information: demonstrations show how to act, while language specifies what is important. We propose Masked Inverse Reinforcement Learning (Masked IRL), a framework that uses large language models (LLMs) to combine the strengths of both input types. Masked IRL infers state-relevance masks from language instructions and enforces invariance to irrelevant state components. When instructions are ambiguous, it uses LLM reasoning to clarify them in the context of the demonstrations. In simulation and on a real robot, Masked IRL outperforms prior language-conditioned IRL methods by up to 15% while using up to 4.7 times less data, demonstrating improved sample-efficiency, generalization, and robustness to ambiguous language. Project page: this https URL and Code: this https URL 

**Abstract (ZH)**: 机器人可以通过从演示中学习奖励函数来适应用户偏好，但在数据有限的情况下，奖励模型往往会过度拟合于无关的关联性而无法泛化。这是因为演示展示了机器人如何执行任务，但没有说明该任务的重点，导致模型聚焦于无关的状态细节。自然语言可以更直接地指明机器人应该关注什么，并且原则上可以消除与演示一致的多个奖励函数中的歧义。然而，现有的基于自然语言的奖励学习方法通常将指令视为简单的条件信号，而没有充分利用其消除歧义的潜力。此外，现实中的指令本身往往是含糊不清的，所以简单的条件是不可靠的。我们的关键洞察是这两种输入类型携带动态互补的信息：演示展示了如何行动，而语言指明了什么才是重要的。我们提出了一种掩码逆强化学习（Masked IRL）框架，该框架利用大型语言模型（LLMs）结合这两种输入类型的优点。掩码逆强化学习从语言指令中推断状态相关掩码，并防止对无关状态成分的依赖。当指令含糊不清时，它利用LLM推理在演示的背景下澄清指令。在模拟和实际机器人上，掩码逆强化学习在使用最多4.7倍少的数据情况下，比以前的基于自然语言的逆强化学习方法性能高出最多15%，展示了更好的采样效率、泛化能力和对含糊语言的鲁棒性。项目页面：[this https URL]，代码：[this https URL]。 

---
# SkillGen: Learning Domain Skills for In-Context Sequential Decision Making 

**Title (ZH)**: SkillGen: 学习领域技能以进行上下文相关序列决策 

**Authors**: Ruomeng Ding, Wei Cheng, Minglai Shao, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.14670)  

**Abstract**: Large language models (LLMs) are increasingly applied to sequential decision-making through in-context learning (ICL), yet their effectiveness is highly sensitive to prompt quality. Effective prompts should meet three principles: focus on decision-critical information, provide step-level granularity, and minimize reliance on expert annotations through label efficiency. However, existing ICL methods often fail to satisfy all three criteria simultaneously. Motivated by these challenges, we introduce SkillGen, a skill-based ICL framework for structured sequential reasoning. It constructs an action-centric, domain-level graph from sampled trajectories, identifies high-utility actions via temporal-difference credit assignment, and retrieves step-wise skills to generate fine-grained, context-aware prompts. We further present a theoretical analysis showing that focusing on high-utility segments supports task identifiability and informs more effective ICL prompt design. Experiments on ALFWorld, BabyAI, and ScienceWorld, using both open-source and proprietary LLMs, show that SkillGen achieves consistent gains, improving progress rate by 5.9%-16.5% on average across models. 

**Abstract (ZH)**: 基于技能的上下文学习框架（SkillGen）：结构化序列推理中的高效决策提示设计 

---
# AutoTool: Efficient Tool Selection for Large Language Model Agents 

**Title (ZH)**: AutoTool：大型语言模型代理的高效工具选择 

**Authors**: Jingyi Jia, Qinbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.14650)  

**Abstract**: Large Language Model (LLM) agents have emerged as powerful tools for automating complex tasks by leveraging the reasoning and decision-making abilities of LLMs. However, a major bottleneck in current agent frameworks lies in the high inference cost of tool selection, especially in approaches like ReAct that repeatedly invoke the LLM to determine which tool to use at each step. In this work, we propose AutoTool, a novel graph-based framework that bypasses repeated LLM inference by exploiting a key empirical observation: tool usage inertia - the tendency of tool invocations to follow predictable sequential patterns. AutoTool constructs a directed graph from historical agent trajectories, where nodes represent tools and edges capture transition probabilities, effectively modeling the inertia in tool selection. It further integrates parameter-level information to refine tool input generation. By traversing this structured representation, AutoTool efficiently selects tools and their parameters with minimal reliance on LLM inference. Extensive experiments across diverse agent tasks demonstrate that AutoTool reduces inference costs by up to 30% while maintaining competitive task completion rates, offering a practical and scalable enhancement for inference-heavy frameworks. Our work highlights the promise of integrating statistical structure into LLM agent design for greater efficiency without sacrificing performance. 

**Abstract (ZH)**: Large Language Model (LLM) 剂剂通过利用LLM的推理和决策能力，已成为自动化复杂任务的强大工具。然而，当前代理框架的一个主要瓶颈在于工具选择的高推理成本，特别是在像ReAct这样的方法中，这些方法在每一步都反复调用LLM来确定使用哪个工具。在这项工作中，我们提出了AutoTool，这是一种新颖的图基化框架，通过利用一个关键的经验观察结果——工具使用惯性——即工具调用遵循可预测的序列模式，来绕过反复的LLM推理。AutoTool从历史代理轨迹中构建一个有向图，其中节点表示工具，边捕捉转换概率，从而有效地模型了工具选择中的惯性。该框架进一步结合参数级信息以细化工具输入生成。通过遍历这种结构化表示，AutoTool使用最少的LLM推理高效地选择工具及其参数。广泛的任务实验显示，与传统的代理框架相比，AutoTool可以降低多达30%的推理成本，同时保持竞争力的任务完成率，为重推理框架提供了一种实用且可扩展的增益。我们的工作突显了将统计结构集成到LLM代理设计中以提高效率但不牺牲性能的潜力。 

---
# Operationalizing Pluralistic Values in Large Language Model Alignment Reveals Trade-offs in Safety, Inclusivity, and Model Behavior 

**Title (ZH)**: 在大型语言模型对齐中实现多元价值观揭示了安全性、包容性和模型行为之间的权衡 

**Authors**: Dalia Ali, Dora Zhao, Allison Koenecke, Orestis Papakyriakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2511.14476)  

**Abstract**: Although large language models (LLMs) are increasingly trained using human feedback for safety and alignment with human values, alignment decisions often overlook human social diversity. This study examines how incorporating pluralistic values affects LLM behavior by systematically evaluating demographic variation and design parameters in the alignment pipeline. We collected alignment data from US and German participants (N = 1,095, 27,375 ratings) who rated LLM responses across five dimensions: Toxicity, Emotional Awareness (EA), Sensitivity, Stereotypical Bias, and Helpfulness. We fine-tuned multiple Large Language Models and Large Reasoning Models using preferences from different social groups while varying rating scales, disagreement handling methods, and optimization techniques. The results revealed systematic demographic effects: male participants rated responses 18% less toxic than female participants; conservative and Black participants rated responses 27.9% and 44% more emotionally aware than liberal and White participants, respectively. Models fine-tuned on group-specific preferences exhibited distinct behaviors. Technical design choices showed strong effects: the preservation of rater disagreement achieved roughly 53% greater toxicity reduction than majority voting, and 5-point scales yielded about 22% more reduction than binary formats; and Direct Preference Optimization (DPO) consistently outperformed Group Relative Policy Optimization (GRPO) in multi-value optimization. These findings represent a preliminary step in answering a critical question: How should alignment balance expert-driven and user-driven signals to ensure both safety and fair representation? 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）越来越多地使用人类反馈进行训练以确保安全性和与人类价值观的对齐，但对齐决定往往忽视了人类社会的多样性。本研究通过系统评估对齐管道中的 demographics 变异和设计参数，考察纳入多元价值观如何影响 LLM 的行为。我们从美国和德国的参与者（N=1,095，27,375 评分）收集数据，这些参与者基于五个维度对 LLM 回复进行了评分：毒性、情感意识（EA）、敏感性、刻板偏见和有用性。我们使用来自不同社会群体的偏好对多个大型语言模型和大型推理模型进行微调，同时改变评分尺度、意见分歧处理方法和优化技术。结果表明存在系统的 demographics 影响：男性参与者比女性参与者少评 18% 的毒性；右倾和黑人参与者比左倾和白人参与者分别多评 27.9% 和 44% 的情感意识。针对特定群体偏好进行微调的模型表现出不同的行为。技术设计选择显示出强烈的影响：保持评分者意见分歧的方法在毒性减少方面比多数投票提高了约 53%，而 5 点量表相比二元格式带来的减少约多 22%；在多值优化中，直接偏好优化（DPO）始终优于群组相关政策优化（GRPO）。这些发现代表了回答一个关键问题的初步步骤：如何平衡专家驱动和用户驱动的信号以确保安全性和公平的代表性？ 

---
# When Words Change the Model: Sensitivity of LLMs for Constraint Programming Modelling 

**Title (ZH)**: 当词语改变模型：约束编程建模中大语言模型的灵敏性 

**Authors**: Alessio Pellegrino, Jacopo Mauro  

**Link**: [PDF](https://arxiv.org/pdf/2511.14334)  

**Abstract**: One of the long-standing goals in optimisation and constraint programming is to describe a problem in natural language and automatically obtain an executable, efficient model. Large language models appear to bring this vision closer, showing impressive results in automatically generating models for classical benchmarks. However, much of this apparent success may derive from data contamination rather than genuine reasoning: many standard CP problems are likely included in the training data of these models. To examine this hypothesis, we systematically rephrased and perturbed a set of well-known CSPLib problems to preserve their structure while modifying their context and introducing misleading elements. We then compared the models produced by three representative LLMs across original and modified descriptions. Our qualitative analysis shows that while LLMs can produce syntactically valid and semantically plausible models, their performance drops sharply under contextual and linguistic variation, revealing shallow understanding and sensitivity to wording. 

**Abstract (ZH)**: 一种长期目标是在优化和约束编程中将问题用自然语言描述，并自动获得可执行且高效的模型。大型语言模型似乎让这一愿景更接近实现，在自动为经典基准生成模型方面取得了令人印象深刻的成果。然而，这种表面上的成功可能更多地归因于数据污染而非真实的推理：许多标准的CP问题极有可能包含在这些模型的训练数据中。为了验证这一假设，我们系统地重新表述并扰动了一组广为人知的CSPLib问题，以保持其结构并修改其上下文和引入误导性元素。然后，我们比较了三款代表性的LLM在原始描述和修改描述下生成的模型。我们的定性分析表明，尽管LLMs能够生成语法有效且语义合理的模型，但在上下文和语言变化下，其性能急剧下降，揭示了表面的理解和对措辞的敏感性。 

---
# DataSage: Multi-agent Collaboration for Insight Discovery with External Knowledge Retrieval, Multi-role Debating, and Multi-path Reasoning 

**Title (ZH)**: DataSage: 基于外部知识检索、多角色辩论和多路径推理的多agent协作洞察发现 

**Authors**: Xiaochuan Liu, Yuanfeng Song, Xiaoming Yin, Xing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.14299)  

**Abstract**: In today's data-driven era, fully automated end-to-end data analytics, particularly insight discovery, is critical for discovering actionable insights that assist organizations in making effective decisions. With the rapid advancement of large language models (LLMs), LLM-driven agents have emerged as a promising paradigm for automating data analysis and insight discovery. However, existing data insight agents remain limited in several key aspects, often failing to deliver satisfactory results due to: (1) insufficient utilization of domain knowledge, (2) shallow analytical depth, and (3) error-prone code generation during insight generation. To address these issues, we propose DataSage, a novel multi-agent framework that incorporates three innovative features including external knowledge retrieval to enrich the analytical context, a multi-role debating mechanism to simulate diverse analytical perspectives and deepen analytical depth, and multi-path reasoning to improve the accuracy of the generated code and insights. Extensive experiments on InsightBench demonstrate that DataSage consistently outperforms existing data insight agents across all difficulty levels, offering an effective solution for automated data insight discovery. 

**Abstract (ZH)**: 在当今数据驱动的时代，端到端的全自动数据分析，特别是洞察发现，对于发现可操作的洞察以帮助组织做出有效的决策至关重要。随着大型语言模型（LLMs）的快速发展，LLM驱动的代理已成为自动化数据分析和洞察发现的一种有前景的范式。然而，现有的数据洞察代理在几个关键方面仍存在局限性，通常由于：（1）领域知识利用不足，（2）分析深度浅，和（3）洞察生成过程中代码生成易出错。为了解决这些问题，我们提出了DataSage，这是一种新颖的多代理框架，结合了三种创新功能，包括外部知识检索以丰富分析上下文，多角色辩论机制以模拟多样化的分析视角并加深分析深度，以及多路径推理以提高生成代码和洞察的准确性。InsightBench上的广泛实验表明，DataSage在所有难度级别上一致优于现有的数据洞察代理，提供了一种有效的全自动数据洞察发现解决方案。 

---
# PathMind: A Retrieve-Prioritize-Reason Framework for Knowledge Graph Reasoning with Large Language Models 

**Title (ZH)**: PathMind：一种基于大型语言模型的知识图谱推理检索-优先级-推理框架 

**Authors**: Yu Liu, Xixun Lin, Yanmin Shang, Yangxi Li, Shi Wang, Yanan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2511.14256)  

**Abstract**: Knowledge graph reasoning (KGR) is the task of inferring new knowledge by performing logical deductions on knowledge graphs. Recently, large language models (LLMs) have demonstrated remarkable performance in complex reasoning tasks. Despite promising success, current LLM-based KGR methods still face two critical limitations. First, existing methods often extract reasoning paths indiscriminately, without assessing their different importance, which may introduce irrelevant noise that misleads LLMs. Second, while many methods leverage LLMs to dynamically explore potential reasoning paths, they require high retrieval demands and frequent LLM calls. To address these limitations, we propose PathMind, a novel framework designed to enhance faithful and interpretable reasoning by selectively guiding LLMs with important reasoning paths. Specifically, PathMind follows a "Retrieve-Prioritize-Reason" paradigm. First, it retrieves a query subgraph from KG through the retrieval module. Next, it introduces a path prioritization mechanism that identifies important reasoning paths using a semantic-aware path priority function, which simultaneously considers the accumulative cost and the estimated future cost for reaching the target. Finally, PathMind generates accurate and logically consistent responses via a dual-phase training strategy, including task-specific instruction tuning and path-wise preference alignment. Extensive experiments on benchmark datasets demonstrate that PathMind consistently outperforms competitive baselines, particularly on complex reasoning tasks with fewer input tokens, by identifying essential reasoning paths. 

**Abstract (ZH)**: 知识图谱推理（KGR）中的路径思维：一种新颖的框架以增强忠实和可解释的推理 

---
# Enhancing Regional Airbnb Trend Forecasting Using LLM-Based Embeddings of Accessibility and Human Mobility 

**Title (ZH)**: 基于LLM的可达性和人类移动嵌入的区域 Airbnb 趋势增强预测 

**Authors**: Hongju Lee, Youngjun Park, Jisun An, Dongman Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.14248)  

**Abstract**: The expansion of short-term rental platforms, such as Airbnb, has significantly disrupted local housing markets, often leading to increased rental prices and housing affordability issues. Accurately forecasting regional Airbnb market trends can thus offer critical insights for policymakers and urban planners aiming to mitigate these impacts. This study proposes a novel time-series forecasting framework to predict three key Airbnb indicators -- Revenue, Reservation Days, and Number of Reservations -- at the regional level. Using a sliding-window approach, the model forecasts trends 1 to 3 months ahead. Unlike prior studies that focus on individual listings at fixed time points, our approach constructs regional representations by integrating listing features with external contextual factors such as urban accessibility and human mobility. We convert structured tabular data into prompt-based inputs for a Large Language Model (LLM), producing comprehensive regional embeddings. These embeddings are then fed into advanced time-series models (RNN, LSTM, Transformer) to better capture complex spatio-temporal dynamics. Experiments on Seoul's Airbnb dataset show that our method reduces both average RMSE and MAE by approximately 48% compared to conventional baselines, including traditional statistical and machine learning models. Our framework not only improves forecasting accuracy but also offers practical insights for detecting oversupplied regions and supporting data-driven urban policy decisions. 

**Abstract (ZH)**: 短租平台（如Airbnb）规模的扩大显著扰乱了当地的住房市场，通常导致租金上涨和住房负担能力问题。准确预测区域Airbnb市场趋势可以为试图缓解这些影响的政策制定者和城市规划者提供关键见解。本研究提出了一种新的时间序列预测框架，用于预测区域层面的三个关键Airbnb指标——收入、预订天数和预订次数。通过滑动窗口方法，模型预测1至3个月的趋势。与以往侧重于固定时间点的单一房源研究不同，本研究通过结合房源特征与外部因素（如城市可达性和人口流动性）构建区域表示。我们将结构化表格数据转换为基于提示的输入供大型语言模型（LLM）使用，生成综合的区域嵌入。这些嵌入随后馈送到先进的时间序列模型（RNN、LSTM、Transformer），以更好地捕捉复杂的时空动态。在首尔Airbnb数据集上的实验表明，与传统的基线模型（包括传统统计和机器学习模型）相比，我们的方法将平均RMSE和MAE分别降低约48%。本框架不仅提高了预测精度，还提供了检测过剩供应区域并支持数据驱动的城市政策决策的实际见解。 

---
# DevPiolt: Operation Recommendation for IoT Devices at Xiaomi Home 

**Title (ZH)**: DevPiolt： Xiaomi Home 中物联网设备的操作推荐 

**Authors**: Yuxiang Wang, Siwen Wang, Haowei Han, Ao Wang, Boya Liu, Yong Zhao, Chengbo Wu, Bin Zhu, Bin Qin, Xiaokai Zhou, Xiao Yan, Jiawei Jiang, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2511.14227)  

**Abstract**: Operation recommendation for IoT devices refers to generating personalized device operations for users based on their context, such as historical operations, environment information, and device status. This task is crucial for enhancing user satisfaction and corporate profits. Existing recommendation models struggle with complex operation logic, diverse user preferences, and sensitive to suboptimal suggestions, limiting their applicability to IoT device operations. To address these issues, we propose DevPiolt, a LLM-based recommendation model for IoT device operations. Specifically, we first equip the LLM with fundamental domain knowledge of IoT operations via continual pre-training and multi-task fine-tuning. Then, we employ direct preference optimization to align the fine-tuned LLM with specific user preferences. Finally, we design a confidence-based exposure control mechanism to avoid negative user experiences from low-quality recommendations. Extensive experiments show that DevPiolt significantly outperforms baselines on all datasets, with an average improvement of 69.5% across all metrics. DevPiolt has been practically deployed in Xiaomi Home app for one quarter, providing daily operation recommendations to 255,000 users. Online experiment results indicate a 21.6% increase in unique visitor device coverage and a 29.1% increase in page view acceptance rates. 

**Abstract (ZH)**: 基于LLM的物联网设备操作推荐方法DevPiolt 

---
# Do Large Language Models (LLMs) Understand Chronology? 

**Title (ZH)**: 大规模语言模型(LLMs)理解时间顺序吗？ 

**Authors**: Pattaraphon Kenny Wongchamcharoen, Paul Glasserman  

**Link**: [PDF](https://arxiv.org/pdf/2511.14214)  

**Abstract**: Large language models (LLMs) are increasingly used in finance and economics, where prompt-based attempts against look-ahead bias implicitly assume that models understand chronology. We test this fundamental question with a series of chronological ordering tasks with increasing complexities over facts the model already knows from pre-training. Our tasks cover (1) chronological ordering, (2) conditional sorting (filter, then order), and (3) anachronism detection. We evaluate GPT-4.1, Claude-3.7 Sonnet, with and without Extended Thinking (ET), and GPT-5 across multiple reasoning-effort settings. Across models, Exact match rate drops sharply as sequences lengthen even while rank correlations stay high as LLMs largely preserve local order but struggle to maintain a single globally consistent timeline. In conditional sorting, most failures stem from the filtering step rather than the ordering step, but GPT-5 and Claude-3.7 Sonnet with Extended Thinking outshine normal models significantly. Lastly, anachronism detection is found to be the easiest task for the LLMs but performance still declines with increasingly overlapping timelines or entities. Overall, our main contribution is showing that allocating explicit reasoning budget helps with chronological ordering with GPT-5 at medium/high reasoning effort achieving flawless ordering at all lengths and perfect conditional sorting (both self-filtered and given-subset), whereas low/minimal effort degrades with longer lists, mirroring earlier models. Our findings delineate limits of current LLMs on chronological tasks, providing insights into task complexity, and demonstrate scenarios in which reasoning helps. These patterns are important for the real-time application of LLMs in finance. We release all code and evaluation templates to support full reproducibility. 

**Abstract (ZH)**: 大型语言模型在金融和经济学中的时间序列排序能力：分配明确推理预算有助于保持时间一致性 

---
# HFL-FlowLLM: Large Language Models for Network Traffic Flow Classification in Heterogeneous Federated Learning 

**Title (ZH)**: HFL-FlowLLM：异构联邦学习中的大规模语言模型用于网络流量流分类 

**Authors**: Jiazhuo Tian, Yachao Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2511.14199)  

**Abstract**: In modern communication networks driven by 5G and the Internet of Things (IoT), effective network traffic flow classification is crucial for Quality of Service (QoS) management and security. Traditional centralized machine learning struggles with the distributed data and privacy concerns in these heterogeneous environments, while existing federated learning approaches suffer from high costs and poor generalization. To address these challenges, we propose HFL-FlowLLM, which to our knowledge is the first framework to apply large language models to network traffic flow classification in heterogeneous federated learning. Compared to state-of-the-art heterogeneous federated learning methods for network traffic flow classification, the proposed approach improves the average F1 score by approximately 13%, demonstrating compelling performance and strong robustness. When compared to existing large language models federated learning frameworks, as the number of clients participating in each training round increases, the proposed method achieves up to a 5% improvement in average F1 score while reducing the training costs by about 87%. These findings prove the potential and practical value of HFL-FlowLLM in modern communication networks security. 

**Abstract (ZH)**: 在由5G和物联网（IoT）驱动的现代通信网络中，有效的网络流量流分类对于服务质量（QoS）管理和安全至关重要。传统集中式机器学习难以应对这些异构环境中的分布式数据和隐私问题，而现有的联邦学习方法则面临高成本和泛化能力差的问题。为了解决这些挑战，我们提出了HFL-FlowLLM框架，据我们所知，这是首个将大型语言模型应用于异构联邦学习中的网络流量流分类框架。与现有的网络流量流分类的异构联邦学习方法相比，所提出的方法将平均F1分数提高了约13%，展示了出色的表现和较强的鲁棒性。与现有的大型语言模型联邦学习框架相比，随着每轮训练中参与客户端数量的增加，所提出的方法在平均F1分数上可提高最高5%的同时，还将训练成本降低了约87%。这些发现证明了HFL-FlowLLM在现代通信网络安全中的潜在和实际价值。 

---
# APD-Agents: A Large Language Model-Driven Multi-Agents Collaborative Framework for Automated Page Design 

**Title (ZH)**: APD-Agents：一种由大语言模型驱动的多智能体协作框架，用于自动化页面设计 

**Authors**: Xinpeng Chen, Xiaofeng Han, Kaihao Zhang, Guochao Ren, Yujie Wang, Wenhao Cao, Yang Zhou, Jianfeng Lu, Zhenbo Song  

**Link**: [PDF](https://arxiv.org/pdf/2511.14101)  

**Abstract**: Layout design is a crucial step in developing mobile app pages. However, crafting satisfactory designs is time-intensive for designers: they need to consider which controls and content to present on the page, and then repeatedly adjust their size, position, and style for better aesthetics and structure. Although many design software can now help to perform these repetitive tasks, extensive training is needed to use them effectively. Moreover, collaborative design across app pages demands extra time to align standards and ensure consistent styling. In this work, we propose APD-agents, a large language model (LLM) driven multi-agent framework for automated page design in mobile applications. Our framework contains OrchestratorAgent, SemanticParserAgent, PrimaryLayoutAgent, TemplateRetrievalAgent, and RecursiveComponentAgent. Upon receiving the user's description of the page, the OrchestratorAgent can dynamically can direct other agents to accomplish users' design task. To be specific, the SemanticParserAgent is responsible for converting users' descriptions of page content into structured data. The PrimaryLayoutAgent can generate an initial coarse-grained layout of this page. The TemplateRetrievalAgent can fetch semantically relevant few-shot examples and enhance the quality of layout generation. Besides, a RecursiveComponentAgent can be used to decide how to recursively generate all the fine-grained sub-elements it contains for each element in the layout. Our work fully leverages the automatic collaboration capabilities of large-model-driven multi-agent systems. Experimental results on the RICO dataset show that our APD-agents achieve state-of-the-art performance. 

**Abstract (ZH)**: 移动应用页面布局设计是开发移动应用页面的关键步骤。然而，设计师创作满意的布局需要花费大量时间：他们需要考虑在页面上呈现哪些控件和内容，然后反复调整这些元素的大小、位置和样式以达到更好的美观性和结构。尽管如今许多设计软件能够帮助执行这些重复任务，但有效使用它们仍需大量培训。此外，跨应用页面的协作设计还需要额外时间来统一标准并确保风格的一致性。在此工作中，我们提出了一种由大型语言模型（LLM）驱动的多智能体框架——APD-agents，用于移动应用中的自动化页面设计。我们的框架包含OrchestratorAgent、SemanticParserAgent、PrimaryLayoutAgent、TemplateRetrievalAgent和RecursiveComponentAgent。收到用户对页面的描述后，OrchestratorAgent可以动态地指导其他智能体完成用户的设计任务。具体而言，SemanticParserAgent负责将用户对页面内容的描述转换成结构化数据。PrimaryLayoutAgent可以生成该页面的初始粗粒度布局。TemplateRetrievalAgent可以检索语义相关的少量示例并提升布局生成的质量。此外，RecursiveComponentAgent可以用来决定如何递归生成布局中每个元素所包含的所有细粒度子元素。我们的工作充分利用了大型模型驱动的多智能体系统中的自动协作能力。在RICO数据集上的实验结果表明，我们的APD-agents达到了最先进的性能。 

---
# Collaborative QA using Interacting LLMs. Impact of Network Structure, Node Capability and Distributed Data 

**Title (ZH)**: 协作式问答使用交互式大语言模型：网络结构、节点能力与分布式数据的影响 

**Authors**: Adit Jain, Vikram Krishnamurthy, Yiming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14098)  

**Abstract**: In this paper, we model and analyze how a network of interacting LLMs performs collaborative question-answering (CQA) in order to estimate a ground truth given a distributed set of documents. This problem is interesting because LLMs often hallucinate when direct evidence to answer a question is lacking, and these effects become more pronounced in a network of interacting LLMs. The hallucination spreads, causing previously accurate LLMs to hallucinate. We study interacting LLMs and their hallucination by combining novel ideas of mean-field dynamics (MFD) from network science and the randomized utility model from economics to construct a useful generative model. We model the LLM with a latent state that indicates if it is truthful or not with respect to the ground truth, and extend a tractable analytical model considering an MFD to model the diffusion of information in a directed network of LLMs. To specify the probabilities that govern the dynamics of the MFD, we propose a randomized utility model. For a network of LLMs, where each LLM has two possible latent states, we posit sufficient conditions for the existence and uniqueness of a fixed point and analyze the behavior of the fixed point in terms of the incentive (e.g., test-time compute) given to individual LLMs. We experimentally study and analyze the behavior of a network of $100$ open-source LLMs with respect to data heterogeneity, node capability, network structure, and sensitivity to framing on multiple semi-synthetic datasets. 

**Abstract (ZH)**: 本文建模并分析了一网络中相互作用的语言模型如何进行协作问答（CQA），以估计基于分布式文档集的真实情况。这一问题很有趣，因为当缺乏直接证据来回答问题时，语言模型常常会胡言乱语，而在相互作用的语言模型网络中，这种现象会更加明显。胡言乱语会扩散，导致原本准确的语言模型也开始胡言乱语。我们通过将网络科学中的均场动力学（MFD）新思想与经济学中的随机效用模型结合起来，构建了一个实用的生成模型，研究相互作用的语言模型及其胡言乱语现象。我们用一个潜在状态来建模语言模型，该状态指示其相对于真实情况是否诚实，并将MFD的可解分析模型扩展以建模定向语言模型网络中的信息扩散。为了指定MFD动力学的控制概率，我们提出了一个随机效用模型。对于每种语言模型都有两种潜在状态的网络，我们提出了充分条件以确保固定点的存在性和唯一性，并从给定于单个语言模型的激励（例如，测试时的计算资源）的角度分析固定点的行为。我们在多个半合成数据集上实验性地研究和分析了100个开源语言模型在网络异质性、节点能力、网络结构和框架敏感性方面的行为。 

---
# Syn-STARTS: Synthesized START Triage Scenario Generation Framework for Scalable LLM Evaluation 

**Title (ZH)**: Syn-STARTS: 合成START分诊场景生成框架，用于可扩展的大语言模型评估 

**Authors**: Chiharu Hagiwara, Naoki Nonaka, Yuhta Hashimoto, Ryu Uchimido, Jun Seita  

**Link**: [PDF](https://arxiv.org/pdf/2511.14023)  

**Abstract**: Triage is a critically important decision-making process in mass casualty incidents (MCIs) to maximize victim survival rates. While the role of AI in such situations is gaining attention for making optimal decisions within limited resources and time, its development and performance evaluation require benchmark datasets of sufficient quantity and quality. However, MCIs occur infrequently, and sufficient records are difficult to accumulate at the scene, making it challenging to collect large-scale realworld data for research use. Therefore, we developed Syn-STARTS, a framework that uses LLMs to generate triage cases, and verified its effectiveness. The results showed that the triage cases generated by Syn-STARTS were qualitatively indistinguishable from the TRIAGE open dataset generated by manual curation from training materials. Furthermore, when evaluating the LLM accuracy using hundreds of cases each from the green, yellow, red, and black categories defined by the standard triage method START, the results were found to be highly stable. This strongly indicates the possibility of synthetic data in developing high-performance AI models for severe and critical medical situations. 

**Abstract (ZH)**: 急诊决策在大规模伤亡事件中的关键作用在于最大化幸存者生存率。尽管在有限的资源和时间内利用AI做出最优决策正逐渐受到关注，但其发展和性能评估需要足够的高质量基准数据集。然而，大规模伤亡事件的发生频率较低，在现场积累充足记录以收集大规模真实世界数据用于研究较为困难。因此，我们开发了Syn-STARTS框架，使用大语言模型生成急诊案例，并验证了其有效性。结果表明，Syn-STARTS生成的急诊案例在质量上与由手动筛选训练材料生成的TRIAGE开源数据集无异。此外，使用标准急诊方法START定义的绿色、黄色、红色和黑色四类案例分别进行数百例的LLM准确性评估，结果高度稳定。这强烈表明合成数据在开发高性能AI模型以应对严重和危急医疗情况方面的潜力。 

---
# ALEX:A Light Editing-knowledge Extractor 

**Title (ZH)**: ALEX：一个轻量级编辑知识抽取器 

**Authors**: Minghu Wang, Shuliang Zhao, Yuanyuan Zhao, Hongxia Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.14018)  

**Abstract**: The static nature of knowledge within Large Language Models (LLMs) makes it difficult for them to adapt to evolving information, rendering knowledge editing a critical task. However, existing methods struggle with challenges of scalability and retrieval efficiency, particularly when handling complex, multi-hop questions that require multi-step reasoning. To address these challenges, this paper introduces ALEX (A Light Editing-knowledge Extractor), a lightweight knowledge editing framework. The core innovation of ALEX is its hierarchical memory architecture, which organizes knowledge updates (edits) into semantic clusters. This design fundamentally reduces retrieval complexity from a linear O(N) to a highly scalable O(K+N/C). Furthermore, the framework integrates an Inferential Query Synthesis (IQS) module to bridge the semantic gap between queries and facts , and a Dynamic Evidence Adjudication (DEA) engine that executes an efficient two-stage retrieval process. Experiments on the MQUAKE benchmark demonstrate that ALEX significantly improves both the accuracy of multi-hop answers (MultiHop-ACC) and the reliability of reasoning paths (HopWise-ACC). It also reduces the required search space by over 80% , presenting a promising path toward building scalable, efficient, and accurate knowledge editing systems. 

**Abstract (ZH)**: 大型语言模型中静态知识的性质使其难以适应 evolving 信息，使知识编辑成为一项关键任务。然而，现有方法在可扩展性和检索效率方面面临挑战，特别是在处理需要多步推理的复杂多跳问题时。为了应对这些挑战，本文介绍了ALEX（一种轻量级知识编辑提取器）轻量级知识编辑框架。ALEX的核心创新是其分层记忆架构，将知识更新（编辑）组织成语义簇。该设计将检索复杂度从线性 O(N) 降低到高度可扩展的 O(K+N/C)。此外，该框架集成了推理查询合成（IQS）模块，以弥合查询与事实之间的语义差距，并集成了动态证据鉴定（DEA）引擎，执行高效的两阶段检索过程。基于 MQUAKE 基准的实验表明，ALEX 显著提高了多跳答案的准确性（MultiHop-ACC）和推理路径的可靠性（HopWise-ACC），并将所需的搜索空间减少了超过 80%，为构建可扩展、高效和准确的知识编辑系统提供了有前景的道路。 

---
# Jailbreaking Large Vision Language Models in Intelligent Transportation Systems 

**Title (ZH)**: 在智能交通系统中破解大型视觉语言模型 

**Authors**: Badhan Chandra Das, Md Tasnim Jawad, Md Jueal Mia, M. Hadi Amini, Yanzhao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.13892)  

**Abstract**: Large Vision Language Models (LVLMs) demonstrate strong capabilities in multimodal reasoning and many real-world applications, such as visual question answering. However, LVLMs are highly vulnerable to jailbreaking attacks. This paper systematically analyzes the vulnerabilities of LVLMs integrated in Intelligent Transportation Systems (ITS) under carefully crafted jailbreaking attacks. First, we carefully construct a dataset with harmful queries relevant to transportation, following OpenAI's prohibited categories to which the LVLMs should not respond. Second, we introduce a novel jailbreaking attack that exploits the vulnerabilities of LVLMs through image typography manipulation and multi-turn prompting. Third, we propose a multi-layered response filtering defense technique to prevent the model from generating inappropriate responses. We perform extensive experiments with the proposed attack and defense on the state-of-the-art LVLMs (both open-source and closed-source). To evaluate the attack method and defense technique, we use GPT-4's judgment to determine the toxicity score of the generated responses, as well as manual verification. Further, we compare our proposed jailbreaking method with existing jailbreaking techniques and highlight severe security risks involved with jailbreaking attacks with image typography manipulation and multi-turn prompting in the LVLMs integrated in ITS. 

**Abstract (ZH)**: 大型视觉语言模型（LVLMs）在多模态推理和实际应用中展现出强大的能力，如视觉问答。然而，LVLMs对 Jailbreaking 攻击极为脆弱。本文系统分析了集成在智能交通系统（ITS）中的LVLMs在精心设计的Jailbreaking攻击下的脆弱性。首先，我们精心构建了一个包含与交通相关的有害查询的数据集，这些查询应避免LVLMs响应的OpenAI禁止类别。其次，我们提出了一种新颖的Jailbreaking攻击，通过图像字体篡改和多轮提示利用LVLMs的脆弱性。第三，我们提出了一种多层次的响应过滤防御技术，以防止模型生成不适当的回答。我们在最先进的LVLMs（开源和封闭源）上进行了广泛的实验，采用GPT-4的判断来确定生成回复的毒性得分，并进行人工验证。此外，我们将提出的Jailbreaking方法与现有的Jailbreaking技术进行比较，并强调集成在ITS中的LVLMs中通过图像字体篡改和多轮提示实施的Jailbreaking攻击所涉及的严重安全风险。 

---
# Imagine in Space: Exploring the Frontier of Spatial Intelligence and Reasoning Efficiency in Vision Language Models 

**Title (ZH)**: 想象在空间：探索空间智能与视觉语言模型推理效率的前沿 

**Authors**: Xiaoxing Lian, Aidong Yang, Jun Zhu, Peng Wang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.13782)  

**Abstract**: Large language models (LLMs) and vision language models (VLMs), such as DeepSeek R1,OpenAI o3, and Gemini 2.5 Pro, have demonstrated remarkable reasoning capabilities across logical inference, problem solving, and decision making. However, spatial reasoning:a fundamental component of human cognition that includes mental rotation, navigation, and spatial relationship comprehension remains a significant challenge for current advanced VLMs. We hypothesize that imagination, the internal simulation of spatial states, is the dominant reasoning mechanism within a spatial world model. To test this hypothesis and systematically probe current VLM spatial reasoning mechanisms, we introduce SpatiaLite, a fully synthetic benchmark that jointly measures spatial reasoning accuracy and reasoning efficiency. Comprehensive experiments reveal three key findings. First, advanced VLMs predominantly rely on linguistic representations for reasoning and imagination, resulting in significant deficiencies on visual centric tasks that demand perceptual spatial relations and 3D geometry transformations such as mental rotation or projection prediction. Second, advanced VLMs exhibit severe inefficiency in their current spatial reasoning mechanisms, with token usage growing rapidly as transformation complexity increases. Third, we propose an Imagery Driven Framework (IDF) for data synthesis and training, which can implicitly construct an internal world model that is critical for spatial reasoning in VLMs. Building on SpatiaLite, this work delineates the spatial reasoning limits and patterns of advanced VLMs, identifies key shortcomings, and informs future advances 

**Abstract (ZH)**: 大型语言模型（LLMs）和视觉语言模型（VLMs），如DeepSeek R1、OpenAI o3和Gemini 2.5 Pro，已经在逻辑推理、问题解决和决策制定等方面展示了惊人的推理能力。然而，空间推理——人类认知的基本组成部分，包括心理旋转、导航和空间关系理解——依然是当前先进VLMs的主要挑战。我们假设想象，即对空间状态的内部模拟，是空间世界模型中占主导地位的推理机制。为了验证这一假设并系统地探索当前VLM的空间推理机制，我们引入了SpatiaLite，一个全合成基准，联合衡量空间推理准确性和推理效率。全面的实验揭示了三个关键发现。首先，高级VLMs主要依赖于语言表示进行推理和想象，导致在需求感知空间关系和3D几何变换（如心理旋转或投影预测）的视觉中心任务上存在显著缺陷。其次，高级VLMs在当前的空间推理机制中表现出严重的低效性，随着变换复杂性的增加，标记使用量迅速增长。第三，我们提出了一种基于图像驱动框架（IDF）的数据合成与训练方法，可以隐式构建对VLMs中空间推理至关重要的内部世界模型。基于SpatiaLite，本研究界定了高级VLM的空间推理限制和模式，指出了关键短板，并指导未来的研究进展。 

---
# Near-Lossless Model Compression Enables Longer Context Inference in DNA Large Language Models 

**Title (ZH)**: 近无损模型压缩使DNA大型语言模型具有更长上下文推理能力 

**Authors**: Rui Zhu, Xiaopu Zhou, Haixu Tang, Stephen W. Scherer, Lucila Ohno-Machado  

**Link**: [PDF](https://arxiv.org/pdf/2511.14694)  

**Abstract**: Trained on massive cross-species DNA corpora, DNA large language models (LLMs) learn the fundamental "grammar" and evolutionary patterns of genomic sequences. This makes them powerful priors for DNA sequence modeling, particularly over long ranges. However, two major constraints hinder their use in practice: the quadratic computational cost of self-attention and the growing memory required for key-value (KV) caches during autoregressive decoding. These constraints force the use of heuristics such as fixed-window truncation or sliding windows, which compromise fidelity on ultra-long sequences by discarding distant information. We introduce FOCUS (Feature-Oriented Compression for Ultra-long Self-attention), a progressive context-compression module that can be plugged into pretrained DNA LLMs. FOCUS combines the established k-mer representation in genomics with learnable hierarchical compression: it inserts summary tokens at k-mer granularity and progressively compresses attention key and value activations across multiple Transformer layers, retaining only the summary KV states across windows while discarding ordinary-token KV. A shared-boundary windowing scheme yields a stationary cross-window interface that propagates long-range information with minimal loss. We validate FOCUS on an Evo-2-based DNA LLM fine-tuned on GRCh38 chromosome 1 with self-supervised training and randomized compression schedules to promote robustness across compression ratios. On held-out human chromosomes, FOCUS achieves near-lossless fidelity: compressing a 1 kb context into only 10 summary tokens (about 100x) shifts the average per-nucleotide probability by only about 0.0004. Compared to a baseline without compression, FOCUS reduces KV-cache memory and converts effective inference scaling from O(N^2) to near-linear O(N), enabling about 100x longer inference windows on commodity GPUs with near-lossless fidelity. 

**Abstract (ZH)**: 面向超长序列的特征导向压缩方法（Feature-Oriented Compression for Ultra-long Self-attention） 

---
# Attention via Synaptic Plasticity is All You Need: A Biologically Inspired Spiking Neuromorphic Transformer 

**Title (ZH)**: 基于突触可塑性的注意力：一种生物启发的脉冲神经形态变换器 

**Authors**: Kallol Mondal, Ankush Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.14691)  

**Abstract**: Attention is the brain's ability to selectively focus on a few specific aspects while ignoring irrelevant ones. This biological principle inspired the attention mechanism in modern Transformers. Transformers now underpin large language models (LLMs) such as GPT, but at the cost of massive training and inference energy, leading to a large carbon footprint. While brain attention emerges from neural circuits, Transformer attention relies on dot-product similarity to weight elements in the input sequence. Neuromorphic computing, especially spiking neural networks (SNNs), offers a brain-inspired path to energy-efficient intelligence. Despite recent work on attention-based spiking Transformers, the core attention layer remains non-neuromorphic. Current spiking attention (i) relies on dot-product or element-wise similarity suited to floating-point operations, not event-driven spikes; (ii) keeps attention matrices that suffer from the von Neumann bottleneck, limiting in-memory computing; and (iii) still diverges from brain-like computation. To address these issues, we propose the Spiking STDP Transformer (S$^{2}$TDPT), a neuromorphic Transformer that implements self-attention through spike-timing-dependent plasticity (STDP), embedding query--key correlations in synaptic weights. STDP, a core mechanism of memory and learning in the brain and widely studied in neuromorphic devices, naturally enables in-memory computing and supports non-von Neumann hardware. On CIFAR-10 and CIFAR-100, our model achieves 94.35\% and 78.08\% accuracy with only four timesteps and 0.49 mJ on CIFAR-100, an 88.47\% energy reduction compared to a standard ANN Transformer. Grad-CAM shows that the model attends to semantically relevant regions, enhancing interpretability. Overall, S$^{2}$TDPT illustrates how biologically inspired attention can yield energy-efficient, hardware-friendly, and explainable neuromorphic models. 

**Abstract (ZH)**: 基于STDP的脉冲Transformer (S²TDPT): 生物启发的能量高效和可解释的类神经形态模型 

---
# Ground Truth Generation for Multilingual Historical NLP using LLMs 

**Title (ZH)**: 使用大规模语言模型生成多语言历史NLP的_ground_truth_ 

**Authors**: Clovis Gladstone, Zhao Fang, Spencer Dean Stewart  

**Link**: [PDF](https://arxiv.org/pdf/2511.14688)  

**Abstract**: Historical and low-resource NLP remains challenging due to limited annotated data and domain mismatches with modern, web-sourced corpora. This paper outlines our work in using large language models (LLMs) to create ground-truth annotations for historical French (16th-20th centuries) and Chinese (1900-1950) texts. By leveraging LLM-generated ground truth on a subset of our corpus, we were able to fine-tune spaCy to achieve significant gains on period-specific tests for part-of-speech (POS) annotations, lemmatization, and named entity recognition (NER). Our results underscore the importance of domain-specific models and demonstrate that even relatively limited amounts of synthetic data can improve NLP tools for under-resourced corpora in computational humanities research. 

**Abstract (ZH)**: 历史和低资源NLP仍具有挑战性：由于标注数据有限和与现代网络数据源之间的领域不匹配，本文概述了我们使用大规模语言模型（LLMs）为16世纪至20世纪的历史法文文本和1900年至1950年的中文文本创建ground-truth标注的工作。通过利用LLM生成的一小部分corpus的ground truth，我们对spaCy进行了微调，以在特定时期的词性标注、词形还原和命名实体识别测试中取得显著成果。我们的结果强调了领域特定模型的重要性，并表明即使相对有限的合成数据也能提高计算人文学科中低资源corpus的NLP工具性能。 

---
# Failure to Mix: Large language models struggle to answer according to desired probability distributions 

**Title (ZH)**: 混合失败：大语言模型在回答问题时难以遵循所需的概率分布 

**Authors**: Ivy Yuqian Yang, David Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14630)  

**Abstract**: Scientific idea generation and selection requires exploration following a target probability distribution. In contrast, current AI benchmarks have objectively correct answers, and training large language models (LLMs) via reinforcement learning against these benchmarks discourages probabilistic exploration. Here, we conducted systematic experiments requesting LLMs to produce outputs following simple probabilistic distributions, and found that all modern LLMs tested grossly fail to follow the distributions. For example, requesting a binary output of "1" 49% of the time produces an answer of "0" nearly 100% of the time. This step function-like behavior of near-exclusively generating the output with marginally highest probability even overrules even strong in-built LLM biases. 

**Abstract (ZH)**: 科学理念的生成与选择需要遵循目标概率分布进行探索。相比之下，当前的AI基准具有客观正确的答案，通过强化学习对这些基准进行训练大型语言模型（LLMs）会抑制概率探索。在这里，我们系统地开展了要求LLMs生成遵循简单概率分布的输出的实验，并发现所有测试的现代LLMs严重未能遵循这些分布。例如，要求生成“1”的输出49%的时间实际上几乎100%地生成了“0”的输出。这种类似于阶跃函数的行为，即几乎仅生成概率略高的输出，甚至超越了LLMs内部的强烈偏见。 

---
# ReflexGrad: Three-Way Synergistic Architecture for Zero-Shot Generalization in LLM Agents 

**Title (ZH)**: ReflexGrad：三元协同架构在大规模语言模型代理中的零样本泛化 

**Authors**: Ankush Kadu, Ashwanth Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2511.14584)  

**Abstract**: Enabling agents to learn from experience and generalize across diverse tasks without task-specific training remains a fundamental challenge in reinforcement learning and decision-making. While recent approaches have explored episodic memory (Reflexion), gradient-based prompt optimization (TextGrad),and hierarchical task decomposition independently, their potential for synergistic integration remains unexplored. We introduce ReflexGrad, a novel architecture that tightly couples three complementary mechanisms: (1) LLM-based hierarchical TODO decomposition for strategic planning, (2) history-aware causal reflection that analyzes recent action patterns to identify failure root causes and enable within-trial learning, and (3) gradient-based optimization for systematic improvement. Unlike prior work relying on few-shot demonstrations, our system achieves true zero-shot generalization through pure LLM semantic reasoning,requiring no task-specific examples, fine-tuning, or hardcoded similarity metrics. Evaluated on ALFWorld benchmark tasks, ReflexGrad demonstrates 67% zero-shot success rate on Trial 0 without any prior task experience or demonstrations, establishing effective performance on first exposure. Through empirical analysis, we identify the architectural mechanisms underlying stable convergence (zero action loops) and effective cross-task transfer (67% to 78% improvement).Our work demonstrates that synergistic integration of complementary learning mechanisms enables robust zero-shot generalization that approaches few-shot baselines from prior work. 

**Abstract (ZH)**: 使代理能够在没有特定任务训练的情况下从经验中学习并跨多种任务进行泛化仍然是强化学习和决策中的一个基本挑战。虽然近期的研究已经探索了情景记忆（Reflexion）、基于梯度的提示优化（TextGrad）以及层次任务分解等独立方法，但它们之间潜在的协同集成仍未被发掘。我们引入了ReflexGrad，这是一种新颖的架构，紧密耦合了三种互补机制：（1）基于大语言模型的层次TODO分解进行战略规划，（2）历史感知因果反思，通过分析近期的行为模式来识别故障的根本原因并促进任务内的学习，以及（3）基于梯度的优化以实现系统的改进。与以往依赖少量示例工作的研究不同，我们的系统通过纯粹的大语言模型语义推理实现了真正意义上的零样本泛化，无需特定任务的例子、微调或硬编码的相似性度量。在ALFWorld基准任务上的评估表明，ReflexGrad在没有 prior task 经验或演示的情况下，在 Trial 0 达到了67%的零样本成功率，并且在首次接触时表现出有效性能。通过实证分析，我们识别出了架构机制，这些机制促成了稳定的收敛（零行动循环）和有效的跨任务转移（67%至78%的改进）。我们的工作证明了互补学习机制的协同集成能够实现稳健的零样本泛化，接近以往工作的少量示例基线。 

---
# Tell Me: An LLM-powered Mental Well-being Assistant with RAG, Synthetic Dialogue Generation, and Agentic Planning 

**Title (ZH)**: Tell Me: 一个基于LLM的心理 wellbeing 辅助器，结合了RAG、合成对话生成和自主规划 

**Authors**: Trishala Jayesh Ahalpara  

**Link**: [PDF](https://arxiv.org/pdf/2511.14445)  

**Abstract**: We present Tell Me, a mental well-being system that leverages advances in large language models to provide accessible, context-aware support for users and researchers. The system integrates three components: (i) a retrieval-augmented generation (RAG) assistant for personalized, knowledge-grounded dialogue; (ii) a synthetic client-therapist dialogue generator conditioned on client profiles to facilitate research on therapeutic language and data augmentation; and (iii) a Well-being AI crew, implemented with CrewAI, that produces weekly self-care plans and guided meditation audio. The system is designed as a reflective space for emotional processing rather than a substitute for professional therapy. It illustrates how conversational assistants can lower barriers to support, complement existing care, and broaden access to mental health resources. To address the shortage of confidential therapeutic data, we introduce synthetic client-therapist dialogue generation conditioned on client profiles. Finally, the planner demonstrates an innovative agentic workflow for dynamically adaptive, personalized self-care, bridging the limitations of static well-being tools. We describe the architecture, demonstrate its functionalities, and report evaluation of the RAG assistant in curated well-being scenarios using both automatic LLM-based judgments and a human-user study. This work highlights opportunities for interdisciplinary collaboration between NLP researchers and mental health professionals to advance responsible innovation in human-AI interaction for well-being. 

**Abstract (ZH)**: 一种利用大规模语言模型进步的mental well-being系统：Tell Me及其应用 

---
# Watchdogs and Oracles: Runtime Verification Meets Large Language Models for Autonomous Systems 

**Title (ZH)**: 看门狗和先知：运行时验证与大规模语言模型在自主系统中的结合 

**Authors**: Angelo Ferrando  

**Link**: [PDF](https://arxiv.org/pdf/2511.14435)  

**Abstract**: Assuring the safety and trustworthiness of autonomous systems is particularly difficult when learning-enabled components and open environments are involved. Formal methods provide strong guarantees but depend on complete models and static assumptions. Runtime verification (RV) complements them by monitoring executions at run time and, in its predictive variants, by anticipating potential violations. Large language models (LLMs), meanwhile, excel at translating natural language into formal artefacts and recognising patterns in data, yet they remain error-prone and lack formal guarantees. This vision paper argues for a symbiotic integration of RV and LLMs. RV can serve as a guardrail for LLM-driven autonomy, while LLMs can extend RV by assisting specification capture, supporting anticipatory reasoning, and helping to handle uncertainty. We outline how this mutual reinforcement differs from existing surveys and roadmaps, discuss challenges and certification implications, and identify future research directions towards dependable autonomy. 

**Abstract (ZH)**: 确保自主系统的学习驱动组件和开放环境的安全性和可信度尤为困难。形式化方法提供了强大的保证，但依赖于完整模型和静态假设。运行时验证（RV）通过在运行时监控执行和预测潜在违规行为来补充它们。同时，大规模语言模型（LLMs）在自然语言转换为形式化 artefacts 和识别数据模式方面表现出色，但它们仍然容易出错且缺乏形式化保证。这篇愿景论文主张将 RV 和 LLMs 进行共生集成。RV 可以为 LLM 驱动的自主性提供护栏，而 LLMs 可以通过辅助规范捕获、支持预见性推理和帮助处理不确定性来扩展 RV。我们概述了这种相互强化与现有综述和路线图的不同之处，讨论了挑战和认证影响，并指出了通往依赖自主性的未来研究方向。 

---
# The Tokenization Bottleneck: How Vocabulary Extension Improves Chemistry Representation Learning in Pretrained Language Models 

**Title (ZH)**: 令牌化瓶颈：词汇扩展如何改善预训练语言模型中的化学表示学习 

**Authors**: Prathamesh Kalamkar, Ned Letcher, Meissane Chami, Sahger Lad, Shayan Mohanty, Prasanna Pendse  

**Link**: [PDF](https://arxiv.org/pdf/2511.14365)  

**Abstract**: The application of large language models (LLMs) to chemistry is frequently hampered by a "tokenization bottleneck", where tokenizers tuned on general-domain text tend to fragment chemical representations such as SMILES into semantically uninformative sub-tokens. This paper introduces a principled methodology to resolve this bottleneck by unifying the representation of natural language and molecular structures within a single model. Our approach involves targeted vocabulary extension-augmenting a pretrained LLM's vocabulary with chemically salient tokens, followed by continued pretraining on chemistry-domain text to integrate this new knowledge. We provide an empirical demonstration of the effectiveness of this strategy, showing that our methodology leads to superior performance on a range of downstream chemical tasks. 

**Abstract (ZH)**: 大语言模型在化学中的应用常受到“分词瓶颈”的限制，通用领域文本训练的分词器往往会将如SMILES这样的化学表示拆分为语义不相关信息子词。本文介绍了一种原理性的方法来解决这一瓶颈，该方法通过在单一模型中统一自然语言和分子结构的表示来实现。我们的方法包括针对词汇表的定向扩展——为预训练的大语言模型添加化学相关的词汇，随后在化学领域文本上继续预训练以整合这种新知识。我们通过实证研究展示了该策略的有效性，证明了我们的方法在一系列下游化学任务中表现更优。 

---
# AraLingBench A Human-Annotated Benchmark for Evaluating Arabic Linguistic Capabilities of Large Language Models 

**Title (ZH)**: AraLingBench：一个用于评估大型语言模型阿拉伯语语言能力的人工标注基准。 

**Authors**: Mohammad Zbib, Hasan Abed Al Kader Hammoud, Sina Mukalled, Nadine Rizk, Fatima Karnib, Issam Lakkis, Ammar Mohanna, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2511.14295)  

**Abstract**: We present AraLingBench: a fully human annotated benchmark for evaluating the Arabic linguistic competence of large language models (LLMs). The benchmark spans five core categories: grammar, morphology, spelling, reading comprehension, and syntax, through 150 expert-designed multiple choice questions that directly assess structural language understanding. Evaluating 35 Arabic and bilingual LLMs reveals that current models demonstrate strong surface level proficiency but struggle with deeper grammatical and syntactic reasoning. AraLingBench highlights a persistent gap between high scores on knowledge-based benchmarks and true linguistic mastery, showing that many models succeed through memorization or pattern recognition rather than authentic comprehension. By isolating and measuring fundamental linguistic skills, AraLingBench provides a diagnostic framework for developing Arabic LLMs. The full evaluation code is publicly available on GitHub. 

**Abstract (ZH)**: AraLingBench: 一种全面的人工标注基准，用于评估大型语言模型的阿拉伯语语言能力 

---
# LLM-Aligned Geographic Item Tokenization for Local-Life Recommendation 

**Title (ZH)**: LLM对齐的地理项分词以进行本地生活推荐 

**Authors**: Hao Jiang, Guoquan Wang, Donglin Zhou, Sheng Yu, Yang Zeng, Wencong Zeng, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.14221)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enhanced text-based recommendation by enriching traditional ID-based methods with semantic generalization capabilities. Text-based methods typically encode item textual information via prompt design and generate discrete semantic IDs through item tokenization. However, in domain-specific tasks such as local-life services, simply injecting location information into prompts fails to capture fine-grained spatial characteristics and real-world distance awareness among items. To address this, we propose LGSID, an LLM-Aligned Geographic Item Tokenization Framework for Local-life Recommendation. This framework consists of two key components: (1) RL-based Geographic LLM Alignment, and (2) Hierarchical Geographic Item Tokenization. In the RL-based alignment module, we initially train a list-wise reward model to capture real-world spatial relationships among items. We then introduce a novel G-DPO algorithm that uses pre-trained reward model to inject generalized spatial knowledge and collaborative signals into LLMs while preserving their semantic understanding. Furthermore, we propose a hierarchical geographic item tokenization strategy, where primary tokens are derived from discrete spatial and content attributes, and residual tokens are refined using the aligned LLM's geographic representation vectors. Extensive experiments on real-world Kuaishou industry datasets show that LGSID consistently outperforms state-of-the-art discriminative and generative recommendation models. Ablation studies, visualizations, and case studies further validate its effectiveness. 

**Abstract (ZH)**: Recent Advances in Large Language Models for Enhanced Text-Based Local-life Recommendation through LGSID Framework 

---
# DiverseClaire: Simulating Students to Improve Introductory Programming Course Materials for All CS1 Learners 

**Title (ZH)**: DiverseClaire: 模拟学生以改进面向所有CS1学习者的 introductory programming课程材料 

**Authors**: Wendy Wong, Yuchao Jiang, Yuekang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.14198)  

**Abstract**: Although CS programs are booming, introductory courses like CS1 still adopt a one-size-fits-all formats that can exacerbate cognitive load and discourage learners with autism, ADHD, dyslexia and other neurological conditions. These call for compassionate pedagogies and Universal Design For Learning (UDL) to create learning environments and materials where cognitive diversity is welcomed. To address this, we introduce DiverseClaire a pilot study, which simulates students including neurodiverse profiles using LLMs and diverse personas. By leveraging Bloom's Taxonomy and UDL, DiverseClaire compared UDL-transformed lecture slides with traditional formats. To evaluate DiverseClaire controlled experiments, we used the evaluation metric the average score. The findings revealed that the simulated neurodiverse students struggled with learning due to lecture slides that were in inaccessible formats. These results highlight the need to provide course materials in multiple formats for diverse learner preferences. Data from our pilot study will be made available to assist future CS1 instructors. 

**Abstract (ZH)**: 虽然CS项目蓬勃发展，但像CS1这样的入门课程仍然采用一刀切的教学形式，这会增加自闭症、ADHD、阅读障碍及其他神经发育条件学生的认知负荷，并可能使他们感到挫败。因此，需要充满 compassion 的教育方法和普遍设计学习（UDL）来创造一个欢迎认知多样性learning环境和材料。为此，我们介绍了DiverseClaire试点研究，该研究使用LLMs和多元人物模拟神经多样性学生。通过运用布卢姆分类法和UDL，DiverseClaire将UDL转换后的讲义幻灯片与传统格式进行了比较。为了评估DiverseClaire，我们使用平均成绩作为评价指标。研究结果表明，模拟的神经多样性学生因为在不可访问格式的讲义中学习而遇到困难。这些结果突显了提供多种格式的教学材料以满足多样化的学习者偏好的重要性。我们的试点研究数据将供未来的CS1讲师参考使用。 

---
# SymLoc: Symbolic Localization of Hallucination across HaluEval and TruthfulQA 

**Title (ZH)**: SymLoc: 符号化 hallucination 定位 across HaluEval 和 TruthfulQA 

**Authors**: Naveen Lamba, Sanju Tiwari, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2511.14172)  

**Abstract**: LLMs still struggle with hallucination, especially when confronted with symbolic triggers like modifiers, negation, numbers, exceptions, and named entities. Yet, we lack a clear understanding of where these symbolic hallucinations originate, making it crucial to systematically handle such triggers and localize the emergence of hallucination inside the model. While prior work explored localization using statistical techniques like LSC and activation variance analysis, these methods treat all tokens equally and overlook the role symbolic linguistic knowledge plays in triggering hallucinations. So far, no approach has investigated how symbolic elements specifically drive hallucination failures across model layers, nor has symbolic linguistic knowledge been used as the foundation for a localization framework. We propose the first symbolic localization framework that leverages symbolic linguistic and semantic knowledge to meaningfully trace the development of hallucinations across all model layers. By focusing on how models process symbolic triggers, we analyze five models using HaluEval and TruthfulQA. Our symbolic knowledge approach reveals that attention variance for these linguistic elements explodes to critical instability in early layers (2-4), with negation triggering catastrophic variance levels, demonstrating that symbolic semantic processing breaks down from the very beginning. Through the lens of symbolic linguistic knowledge, despite larger model sizes, hallucination rates remain consistently high (78.3%-83.7% across Gemma variants), with steep attention drops for symbolic semantic triggers throughout deeper layers. Our findings demonstrate that hallucination is fundamentally a symbolic linguistic processing failure, not a general generation problem, revealing that symbolic semantic knowledge provides the key to understanding and localizing hallucination mechanisms in LLMs. 

**Abstract (ZH)**: LLMs在处理象征触发器（如修饰语、否定词、数字、例外情况和命名实体）时仍然容易产生幻觉，但我们对这些象征性幻觉的起源缺乏清晰的理解，这使得系统地处理这类触发器并定位模型中幻觉的产生变得至关重要。尽管早期研究使用统计技术（如LSC和激活方差分析）探索幻觉的定位，但这些方法未能考虑象征性语言知识在诱发幻觉中的作用。目前尚未有方法研究象征性元素如何具体驱动模型各层的幻觉失效，也没有将象征性语言知识作为定位框架的基础。我们提出了第一个利用象征性语言和语义知识来有意义地追溯各模型层中幻觉发展的象征性定位框架。通过专注于模型如何处理象征性触发器，我们使用HaluEval和TruthfulQA分析了五个模型。我们的象征性知识方法揭示，这些语言元素在早期层（2-4）的注意力方差急剧增加至关键不稳定状态，其中否定词触发极其高的方差水平，表明象征性语义处理从一开始就失效。借助象征性语言知识，尽管模型规模更大，幻觉率仍保持在较高水平（Gemma变体中为78.3%-83.7%），并在更深的层中对象征性语义触发器显示出急剧下降的注意力。我们的研究发现幻觉本质上是象征性语言处理失败，而不是一般生成问题，表明象征性语义知识是理解并定位LLMs中幻觉机制的关键。 

---
# AdaTok: Adaptive Token Compression with Object-Aware Representations for Efficient Multimodal LLMs 

**Title (ZH)**: AdaTok：具有对象意识表示的自适应Token压缩以实现高效的多模态LLM 

**Authors**: Xinliang Zhang, Lei Zhu, Hangzhou He, Shuang Zeng, Ourui Fu, Jiakui Hu, Zhengjian Yao, Yanye Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.14169)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated substantial value in unified text-image understanding and reasoning, primarily by converting images into sequences of patch-level tokens that align with their architectural paradigm. However, patch-level tokenization leads to a quadratic growth in image tokens, burdening MLLMs' understanding and reasoning with enormous computation and memory. Additionally, the traditional patch-wise scanning tokenization workflow misaligns with the human vision cognition system, further leading to hallucination and computational redundancy. To address this issue, we propose an object-level token merging strategy for Adaptive Token compression, revealing the consistency with human vision system. The experiments are conducted on multiple comprehensive benchmarks, which show that our approach averagely, utilizes only 10% tokens while achieving almost 96% of the vanilla model's performance. More extensive experimental results in comparison with relevant works demonstrate the superiority of our method in balancing compression ratio and performance. Our code will be available. 

**Abstract (ZH)**: 面向对象的token合并策略在多模态大语言模型中的适应性token压缩：与人类视觉系统的一致性研究 

---
# FAPE-IR: Frequency-Aware Planning and Execution Framework for All-in-One Image Restoration 

**Title (ZH)**: FAPE-IR：全场景图像恢复的频率感知计划与执行框架 

**Authors**: Jingren Liu, Shuning Xu, Qirui Yang, Yun Wang, Xiangyu Chen, Zhong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2511.14099)  

**Abstract**: All-in-One Image Restoration (AIO-IR) aims to develop a unified model that can handle multiple degradations under complex conditions. However, existing methods often rely on task-specific designs or latent routing strategies, making it hard to adapt to real-world scenarios with various degradations. We propose FAPE-IR, a Frequency-Aware Planning and Execution framework for image restoration. It uses a frozen Multimodal Large Language Model (MLLM) as a planner to analyze degraded images and generate concise, frequency-aware restoration plans. These plans guide a LoRA-based Mixture-of-Experts (LoRA-MoE) module within a diffusion-based executor, which dynamically selects high- or low-frequency experts, complemented by frequency features of the input image. To further improve restoration quality and reduce artifacts, we introduce adversarial training and a frequency regularization loss. By coupling semantic planning with frequency-based restoration, FAPE-IR offers a unified and interpretable solution for all-in-one image restoration. Extensive experiments show that FAPE-IR achieves state-of-the-art performance across seven restoration tasks and exhibits strong zero-shot generalization under mixed degradations. 

**Abstract (ZH)**: 面向频率的规划与执行框架（FAPE-IR）：统一的图像恢复解决方案 

---
# NeuroPath: Neurobiology-Inspired Path Tracking and Reflection for Semantically Coherent Retrieval 

**Title (ZH)**: 神经路径：受神经生物学启发的路径追踪与反射以实现语义一致的检索 

**Authors**: Junchen Li, Rongzheng Wang, Yihong Huang, Qizhi Chen, Jiasheng Zhang, Shuang Liang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14096)  

**Abstract**: Retrieval-augmented generation (RAG) greatly enhances large language models (LLMs) performance in knowledge-intensive tasks. However, naive RAG methods struggle with multi-hop question answering due to their limited capacity to capture complex dependencies across documents. Recent studies employ graph-based RAG to capture document connections. However, these approaches often result in a loss of semantic coherence and introduce irrelevant noise during node matching and subgraph construction. To address these limitations, we propose NeuroPath, an LLM-driven semantic path tracking RAG framework inspired by the path navigational planning of place cells in neurobiology. It consists of two steps: Dynamic Path Tracking and Post-retrieval Completion. Dynamic Path Tracking performs goal-directed semantic path tracking and pruning over the constructed knowledge graph (KG), improving noise reduction and semantic coherence. Post-retrieval Completion further reinforces these benefits by conducting second-stage retrieval using intermediate reasoning and the original query to refine the query goal and complete missing information in the reasoning path. NeuroPath surpasses current state-of-the-art baselines on three multi-hop QA datasets, achieving average improvements of 16.3% on recall@2 and 13.5% on recall@5 over advanced graph-based RAG methods. Moreover, compared to existing iter-based RAG methods, NeuroPath achieves higher accuracy and reduces token consumption by 22.8%. Finally, we demonstrate the robustness of NeuroPath across four smaller LLMs (Llama3.1, GLM4, Mistral0.3, and Gemma3), and further validate its scalability across tasks of varying complexity. Code is available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）极大地提高了大型语言模型（LLMs）在知识密集型任务中的性能。然而，传统的RAG方法在多跳问答任务中因难以捕捉文档间复杂依赖关系而表现出局限性。近期研究采用了基于图的RAG来捕捉文档间的联系，但这些方法往往会导致语义连贯性降低，并在节点匹配和子图构建过程中引入无关噪声。为解决这些问题，我们提出了一种神经轨迹（NeuroPath）框架，该框架由LLM驱动，灵感来源于神经生物学中地层细胞的空间路径导航规划，包括动态路径跟踪和检索后完成两个步骤。动态路径跟踪通过构建知识图谱并进行目标导向的语义路径跟踪和剪枝，提高噪声减少和语义连贯性。检索后完成则通过中间推理和原始查询执行二次检索，进一步强化这些益处，细化查询目标并补充推理路径中的缺失信息。神经轨迹在三个多跳问答数据集上超越了当前最先进的基线方法，平均在召回@2上提升了16.3%，在召回@5上提升了13.5%，相较于先进图基RAG方法。此外，与现有的迭代基RAG方法相比，神经轨迹在准确度上具有优势，并减少了22.8%的token消耗。同时，我们展示了神经轨迹在四个较小的LLM（Llama3.1、GLM4、Mistral0.3和Gemma3）上的鲁棒性，并进一步验证了其在不同复杂度任务上的可扩展性。代码可在以下链接获取。 

---
# GRPO Privacy Is at Risk: A Membership Inference Attack Against Reinforcement Learning With Verifiable Rewards 

**Title (ZH)**: 成员关系隐私处于风险之中：针对具有可验证奖励的强化学习的成员身份推断攻击 

**Authors**: Yule Liu, Heyi Zhang, Jinyi Zheng, Zhen Sun, Zifan Peng, Tianshuo Cong, Yilong Yang, Xinlei He, Zhuo Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.14045)  

**Abstract**: Membership inference attacks (MIAs) on large language models (LLMs) pose significant privacy risks across various stages of model training. Recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) have brought a profound paradigm shift in LLM training, particularly for complex reasoning tasks. However, the on-policy nature of RLVR introduces a unique privacy leakage pattern: since training relies on self-generated responses without fixed ground-truth outputs, membership inference must now determine whether a given prompt (independent of any specific response) is used during fine-tuning. This creates a threat where leakage arises not from answer memorization.
To audit this novel privacy risk, we propose Divergence-in-Behavior Attack (DIBA), the first membership inference framework specifically designed for RLVR. DIBA shifts the focus from memorization to behavioral change, leveraging measurable shifts in model behavior across two axes: advantage-side improvement (e.g., correctness gain) and logit-side divergence (e.g., policy drift). Through comprehensive evaluations, we demonstrate that DIBA significantly outperforms existing baselines, achieving around 0.8 AUC and an order-of-magnitude higher TPR@0.1%FPR. We validate DIBA's superiority across multiple settings--including in-distribution, cross-dataset, cross-algorithm, black-box scenarios, and extensions to vision-language models. Furthermore, our attack remains robust under moderate defensive measures.
To the best of our knowledge, this is the first work to systematically analyze privacy vulnerabilities in RLVR, revealing that even in the absence of explicit supervision, training data exposure can be reliably inferred through behavioral traces. 

**Abstract (ZH)**: 大型语言模型（LLMs）训练阶段的会员推理攻击（MIAs）对各类隐私风险构成重大威胁。可信奖励强化学习（RLVR）在复杂推理任务中的应用带来了训练范式的深刻变革，但RLVR的在策略性使得训练过程中出现独特的隐私泄露模式：由于训练依赖于自我生成的响应而不是固定的 ground-truth 输出，会员推理现在必须判断某个给定提示（与任何特定响应无关）是否在微调过程中被使用。这创造了一种威胁，其漏洞并非来自答案的记忆化。

为了审核这一新型隐私风险，我们提出了行为差异攻击（DIBA），这是首个专门针对RLVR的会员推理框架。DIBA 将重点从记忆化转移到行为变化上，通过利用模型行为沿两个轴线的可测量变化来实现：优势侧改进（例如，正确性提升）和逻辑量侧差异（例如，策略漂移）。经过全面的评估，我们展示了DIBA 显著优于现有基线，AUC接近0.8，FPR为0.1%时的TPR提高了几个数量级。我们在多种场景下验证了DIBA的优越性，包括同分布、跨数据集、跨算法、黑盒场景以及视觉语言模型的扩展，并且攻击在采取适度防御措施的情况下依然稳健。到我们所知，这是首项系统分析RLVR隐私漏洞的研究，揭示了即使在缺乏明确监督的情况下，训练数据的暴露也可以通过行为痕迹可靠推断出来。 

---
# Keeping Code-Aware LLMs Fresh: Full Refresh, In-Context Deltas, and Incremental Fine-Tuning 

**Title (ZH)**: 保持代码感知的大语言模型新鲜度：全量刷新、上下文增量调整和增量微调 

**Authors**: Pradeep Kumar Sharma, Ishaan Puri, Mantinder Jit Singh, Swapnil Shivaprasad, Hritvik Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2511.14022)  

**Abstract**: Modern codebases evolve continuously: files are renamed or deleted; public APIs drift; behavior shifts within otherwise familiar modules. A model trained yesterday to map a developer's natural-language question to the exact set of repository file paths that matter will degrade tomorrow, even if the questions themselves look unchanged. In this paper we study, at system scale and across several widely used repositories, how to keep such a model fresh without surrendering retention on earlier code. We frame freshness as a form of domain drift between a base snapshot and the current HEAD, and we compare three families of update strategies: (A) Full Refresh, retraining the entire model at the new snapshot; (B) In-Context Learning (ICL) that injects recent deltas (raw git diffs or concise English summaries) at inference; and (C) Incremental Fine-Tuning (Inc-FT) on delta-derived training sets, with carefully controlled NEW:OLD mixing to mitigate catastrophic forgetting. We contribute an alias-aware evaluation protocol that credits rename while never rewarding deleted paths, and a practical Forgetting Probe that quantifies residual emissions of obsolete paths. Across Flask, SQLAlchemy, Pandas, and Poetry, Inc-FT with old-aware mixes delivers the best overall balance on mixed sets, ICL with English delta summaries delivers the fastest new-code lift when training is not feasible, and Full Refresh remains the ceiling when maximum NEW accuracy matters. We also compare Git-diff Inc-FT to full-file Inc-FT, showing that diffs excel in rename/delete-heavy windows while full-file context wins in behavior-change-heavy windows. 

**Abstract (ZH)**: 现代代码基不断演变：文件被重命名或删除；公共API发生变化；模块的行为发生改变。昨天训练的模型将开发者自然语言的问题映射到具体的相关仓库文件路径，即使问题本身看起来未变，明天该模型的性能也会下降。在本文中，我们在系统级别并跨多个广泛使用的仓库研究了如何在保持对早期代码留存性的前提下使模型保持最新。我们将新鲜度视为基线快照与当前HEAD之间的领域漂移，并比较了三种更新策略：（A）全量刷新，重新训练整个模型；（B）上下文学习（ICL），在推理时注射最近的增量变化（原生git差异或简洁的英文摘要）；（C）增量微调（Inc-FT），在增量训练集中使用谨慎控制的新旧比例混合，以减轻灾难性遗忘。我们贡献了一种注意到别名的评估协议，该协议奖励重命名但不奖励删除的路径，并开发了一个遗忘探针来量化废弃路径的残留排放。在Flask、SQLAlchemy、Pandas和Poetry中，增量微调搭配旧路径意识的混合提供了最佳的整体平衡，当无法进行训练时，英文增量总结带来最快的新代码提升，而全量刷新在追求最大新路径准确率时仍然是上限。此外，我们还比较了git diff增量微调与全文件增量微调，结果显示diff在重命名/删除频繁的窗口中表现出色，而全文件上下文在行为变化频繁的窗口中占优。 

---
# From Narrow Unlearning to Emergent Misalignment: Causes, Consequences, and Containment in LLMs 

**Title (ZH)**: 从窄遗忘到 emergent 不对齐：LLM 中的原因、后果及遏制 

**Authors**: Erum Mushtaq, Anil Ramakrishna, Satyapriya Krishna, Sattvik Sahai, Prasoon Goyal, Kai-Wei Chang, Tao Zhang, Rahul Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.14017)  

**Abstract**: Recent work has shown that fine-tuning on insecure code data can trigger an emergent misalignment (EMA) phenomenon, where models generate malicious responses even to prompts unrelated to the original insecure code-writing task. Such cross-domain generalization of harmful behavior underscores the need for a deeper understanding of the algorithms, tasks, and datasets that induce emergent misalignment. In this work, we extend this study by demonstrating that emergent misalignment can also arise from narrow refusal unlearning in specific domains. We perform refusal unlearning on Cybersecurity and Safety concept, and evaluate EMA by monitoring refusal scores across seven responsible AI (RAI) domains, Cybersecurity, Safety, Toxicity, Bias, Sensitive Content, Medical/Legal, and Privacy. Our work shows that narrow domain unlearning can yield compliance responses for the targeted concept, however, it may also propagate EMA to unrelated domains. Among the two intervened concepts, Cybersecurity and Safety, we find that the safety concept can have larger EMA impact, i.e, causing lower refusal scores, across other unrelated domains such as bias. We observe this effect consistently across two model families, Mistral-7b-0.3v, and Qwen-7b-2.5. Further, we show that refusal unlearning augmented with cross-entropy loss function on a small set of retain data from the affected domains can largely, if not fully, restore alignment across the impacted domains while having lower refusal rate on the concept we perform unlearning on. To investigate the underlying causes of EMA, we analyze concept entanglements at the representation level via concept vectors. Our analysis reveals that concepts with higher representation similarity in earlier layers are more susceptible to EMA after intervention when the refusal stream is altered through targeted refusal unlearning. 

**Abstract (ZH)**: 近期的研究表明，在不安全代码数据上的微调可以触发一种新的不希望的对齐偏差（EMA）现象，即模型在与原始不安全代码编写任务无关的提示上也会生成恶意响应。这种跨领域有害行为的泛化强调了对导致Emergent Misalignment的算法、任务和数据集进行深入理解的必要性。在本研究中，我们通过展示窄域拒绝遗忘也可以在特定领域引发Emergent Misalignment来扩展这项研究。我们在网络安全与安全概念上进行拒绝遗忘，并通过监控网络安全、安全、毒性、偏差、敏感内容、医疗/法律和隐私这七个负责任AI（RAI）领域中的拒绝评分来评估Emergent Misalignment。我们的工作表明，窄域遗忘可以为受目标概念影响的领域提供合规响应，但也可能将Emergent Misalignment传播到与之无关的领域。在干预的两个概念——网络安全和安全中，我们发现，安全概念在其他无关领域（如偏差）上造成高层次Emergent Misalignment的影响更大。我们在两个模型家族Mistral-7b-0.3v和Qwen-7b-2.5上观察到了这一效果的一致性。此外，我们展示了通过在受影响领域保留数据的小集合上使用交叉熵损失函数加强拒绝遗忘，可以显著恢复受影响领域中的对齐，同时在进行遗忘的概念上的拒绝率较低。为了探究Emergent Misalignment的根本原因，我们通过对概念向量进行表示层次的概念纠缠性分析。分析结果表明，在修改拒绝流时，早期层中表示相似度更高的概念在干预后更易受到Emergent Misalignment的影响。 

---
# Knowledge-Grounded Agentic Large Language Models for Multi-Hazard Understanding from Reconnaissance Reports 

**Title (ZH)**: 基于知识引导的代理大型语言模型以理解侦察报告中的多灾种风险 

**Authors**: Chenchen Kuai, Zihao Li, Braden Rosen, Stephanie Paan, Navid Jafari, Jean-Louis Briaud, Yunlong Zhang, Youssef M. A. Hashash, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.14010)  

**Abstract**: Post-disaster reconnaissance reports contain critical evidence for understanding multi-hazard interactions, yet their unstructured narratives make systematic knowledge transfer difficult. Large language models (LLMs) offer new potential for analyzing these reports, but often generate unreliable or hallucinated outputs when domain grounding is absent. This study introduces the Mixture-of-Retrieval Agentic RAG (MoRA-RAG), a knowledge-grounded LLM framework that transforms reconnaissance reports into a structured foundation for multi-hazard reasoning. The framework integrates a Mixture-of-Retrieval mechanism that dynamically routes queries across hazard-specific databases while using agentic chunking to preserve contextual coherence during retrieval. It also includes a verification loop that assesses evidence sufficiency, refines queries, and initiates targeted searches when information remains incomplete. We construct HazardRecQA by deriving question-answer pairs from GEER reconnaissance reports, which document 90 global events across seven major hazard types. MoRA-RAG achieves up to 94.5 percent accuracy, outperforming zero-shot LLMs by 30 percent and state-of-the-art RAG systems by 10 percent, while reducing hallucinations across diverse LLM architectures. MoRA-RAG also enables open-weight LLMs to achieve performance comparable to proprietary models. It establishes a new paradigm for transforming post-disaster documentation into actionable, trustworthy intelligence for hazard resilience. 

**Abstract (ZH)**: 灾害后侦察报告包含理解多灾种相互作用的关键证据，但由于其无结构化的叙述使得系统的知识转移困难。大型语言模型（LLMs）为分析这些报告提供了新的潜力，但在缺乏领域指导的情况下，往往会生成不可靠或虚构的输出。本研究引入了一种名为混合检索代理RAG（MoRA-RAG）的知识接地LLM框架，该框架将侦察报告转换为多灾种推理的结构化基础。该框架集成了混合检索机制，能够在灾害专用数据库之间动态路由查询，同时使用代理分割保持检索过程中的上下文连贯性。此外，该框架还包括一个验证循环，可以评估证据的充分性，细化查询，并在信息不完整时发起有针对性的搜索。我们通过从GEER侦察报告中提取问题-答案对构建了HazardRecQA，这些报告记录了七种主要灾害类型共90个全球事件。MoRA-RAG达到了94.5%的准确率，分别比零样本LLMs高出30%，比最先进的RAG系统高出10%，同时减少了各种LLM架构中的虚构现象。MoRA-RAG还使开放权重LLMs能够达到与专有模型相当的性能。它确立了一个新的范式，将灾害后文件转化为可用于提高灾害韧性的可操作且可信的情报。 

---
# LoCoBench-Agent: An Interactive Benchmark for LLM Agents in Long-Context Software Engineering 

**Title (ZH)**: LoCoBench-Agent: 一种面向长上下文软件工程的LLM代理交互基准测试 

**Authors**: Jielin Qiu, Zuxin Liu, Zhiwei Liu, Rithesh Murthy, Jianguo Zhang, Haolin Chen, Shiyu Wang, Ming Zhu, Liangwei Yang, Juntao Tan, Roshan Ram, Akshara Prabhakar, Tulika Awalgaonkar, Zixiang Chen, Zhepeng Cen, Cheng Qian, Shelby Heinecke, Weiran Yao, Silvio Savarese, Caiming Xiong, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.13998)  

**Abstract**: As large language models (LLMs) evolve into sophisticated autonomous agents capable of complex software development tasks, evaluating their real-world capabilities becomes critical. While existing benchmarks like LoCoBench~\cite{qiu2025locobench} assess long-context code understanding, they focus on single-turn evaluation and cannot capture the multi-turn interactive nature, tool usage patterns, and adaptive reasoning required by real-world coding agents. We introduce \textbf{LoCoBench-Agent}, a comprehensive evaluation framework specifically designed to assess LLM agents in realistic, long-context software engineering workflows. Our framework extends LoCoBench's 8,000 scenarios into interactive agent environments, enabling systematic evaluation of multi-turn conversations, tool usage efficiency, error recovery, and architectural consistency across extended development sessions. We also introduce an evaluation methodology with 9 metrics across comprehension and efficiency dimensions. Our framework provides agents with 8 specialized tools (file operations, search, code analysis) and evaluates them across context lengths ranging from 10K to 1M tokens, enabling precise assessment of long-context performance. Through systematic evaluation of state-of-the-art models, we reveal several key findings: (1) agents exhibit remarkable long-context robustness; (2) comprehension-efficiency trade-off exists with negative correlation, where thorough exploration increases comprehension but reduces efficiency; and (3) conversation efficiency varies dramatically across models, with strategic tool usage patterns differentiating high-performing agents. As the first long-context LLM agent benchmark for software engineering, LoCoBench-Agent establishes a rigorous foundation for measuring agent capabilities, identifying performance gaps, and advancing autonomous software development at scale. 

**Abstract (ZH)**: 大规模语言模型（LLMs）演进为复杂的自主代理，能够执行高级软件开发任务，评价其实际能力变得至关重要。虽然现有的基准如LoCoBench评估长上下文代码理解能力，但它们主要集中在单轮评估上，无法捕捉到多轮交互性质、工具使用模式和适应性推理，这些都是现实世界编码代理所需的能力。我们引入了**LoCoBench-Agent**，一个专门设计用于评估大规模语言模型代理在现实、长上下文软件工程工作流中的综合评估框架。该框架将LoCoBench的8,000种情景扩展到交互型代理环境，使得系统性评估多轮对话、工具使用效率、错误恢复以及长时间开发会话中的架构一致性成为可能。我们还引入了一个包含9个维度评估指标的方法学。该框架为代理提供了8种专门工具（文件操作、搜索、代码分析），并在从10K到1M的上下文长度范围内进行评估，从而精确评估长上下文性能。通过对最先进的模型进行系统性评估，我们揭示了几项关键发现：（1）代理展现出卓越的长上下文鲁棒性；（2）存在理解和效率的权衡，其中彻底探索提高了理解但降低了效率；（3）对话效率在不同模型之间差异显著，战略性的工具使用模式区分了高性能代理。作为首个面向软件工程的长上下文大规模语言模型代理基准，LoCoBench-Agent 为测量代理能力、识别性能差距以及推动大规模自主软件开发奠定了严格的基础。 

---
# Node-Level Uncertainty Estimation in LLM-Generated SQL 

**Title (ZH)**: LLM生成SQL中的节点级别不确定性估计 

**Authors**: Hilaf Hasson, Ruocheng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.13984)  

**Abstract**: We present a practical framework for detecting errors in LLM-generated SQL by estimating uncertainty at the level of individual nodes in the query's abstract syntax tree (AST). Our approach proceeds in two stages. First, we introduce a semantically aware labeling algorithm that, given a generated SQL and a gold reference, assigns node-level correctness without over-penalizing structural containers or alias variation. Second, we represent each node with a rich set of schema-aware and lexical features - capturing identifier validity, alias resolution, type compatibility, ambiguity in scope, and typo signals - and train a supervised classifier to predict per-node error probabilities. We interpret these probabilities as calibrated uncertainty, enabling fine-grained diagnostics that pinpoint exactly where a query is likely to be wrong. Across multiple databases and datasets, our method substantially outperforms token log-probabilities: average AUC improves by +27.44% while maintaining robustness under cross-database evaluation. Beyond serving as an accuracy signal, node-level uncertainty supports targeted repair, human-in-the-loop review, and downstream selective execution. Together, these results establish node-centric, semantically grounded uncertainty estimation as a strong and interpretable alternative to aggregate sequence level confidence measures. 

**Abstract (ZH)**: 一种基于抽象语法树节点级不确定性的LLM生成SQL错误检测实用框架 

---
# What Works for 'Lost-in-the-Middle' in LLMs? A Study on GM-Extract and Mitigations 

**Title (ZH)**: “失中”问题在LLMs中的解决策略：GM-Extract与缓解措施研究 

**Authors**: Mihir Gupte, Eshan Dixit, Muhammad Tayyab, Arun Adiththan  

**Link**: [PDF](https://arxiv.org/pdf/2511.13900)  

**Abstract**: The diminishing ability of large language models (LLMs) to effectively utilize long-range context-the "lost-in-the-middle" phenomenon-poses a significant challenge in retrieval-based LLM applications. To study the impact of this phenomenon in a real-world application setting, we introduce GM-Extract, a novel benchmark dataset meticulously designed to evaluate LLM performance on retrieval of control variables. To accurately diagnose failure modes, we propose a simple yet elegant evaluation system using two distinct metrics: one for spatial retrieval capability (Document Metric) and the other for semantic retrieval capability (Variable Extraction Metric). We conduct a systematic evaluation of 7-8B parameter models on two multi-document tasks (key-value extraction and question-answering), demonstrating a significant change in retrieval performance simply by altering how the data is represented in the context window. While a distinct U-shaped curve was not consistently observed, our analysis reveals a clear pattern of performance across models, which we further correlate with perplexity scores. Furthermore, we perform a literature survey of mitigation methods, which we categorize into two distinct approaches: black-box and white-box methods. We then apply these techniques to our benchmark, finding that their efficacy is highly nuanced. Our evaluation highlights scenarios where these strategies successfully improve performance, as well as surprising cases where they lead to a negative impact, providing a comprehensive understanding of their utility in a practical context. 

**Abstract (ZH)**: 大型语言模型在利用长范围上下文能力下降的“中间迷失”现象对基于检索的大型语言模型应用构成重大挑战。为了研究这一现象在实际应用环境中的影响，我们引入了GM-Extract，一个精心设计的新基准数据集，用于评估大型语言模型在控制变量检索方面的性能。为了准确诊断失败模式，我们提出了一种简单的优雅评估系统，使用两个不同的指标：一个是空间检索能力指标（Document Metric），另一个是语义检索能力指标（变量提取指标）。我们在两个多文档任务（键值提取和问答）上系统性地评估了7-8B参数模型，发现仅通过改变上下文窗口中数据的表示方式，检索性能会显著改变。虽然观察到的U形曲线并不一致，但我们分析发现模型性能存在着明显的模式，并将其与困惑度分数相关联。此外，我们进行了缓解方法的文献调研，并将其分为两种不同的方法：黑盒方法和白盒方法。然后，我们将这些技术应用到我们的基准数据集中，发现其效果具有高度的复杂性。我们的评估突显了这些策略在提高性能方面成功的场景，以及意想不到的场景下导致负面影响的情况，从而为其实用环境下的有效性提供了全面的理解。 

---
# Uncovering and Aligning Anomalous Attention Heads to Defend Against NLP Backdoor Attacks 

**Title (ZH)**: 揭示并对齐异常注意力头以抵御NLP后门攻击 

**Authors**: Haotian Jin, Yang Li, Haihui Fan, Lin Shen, Xiangfang Li, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.13789)  

**Abstract**: Backdoor attacks pose a serious threat to the security of large language models (LLMs), causing them to exhibit anomalous behavior under specific trigger conditions. The design of backdoor triggers has evolved from fixed triggers to dynamic or implicit triggers. This increased flexibility in trigger design makes it challenging for defenders to identify their specific forms accurately. Most existing backdoor defense methods are limited to specific types of triggers or rely on an additional clean model for support. To address this issue, we propose a backdoor detection method based on attention similarity, enabling backdoor detection without prior knowledge of the trigger. Our study reveals that models subjected to backdoor attacks exhibit unusually high similarity among attention heads when exposed to triggers. Based on this observation, we propose an attention safety alignment approach combined with head-wise fine-tuning to rectify potentially contaminated attention heads, thereby effectively mitigating the impact of backdoor attacks. Extensive experimental results demonstrate that our method significantly reduces the success rate of backdoor attacks while preserving the model's performance on downstream tasks. 

**Abstract (ZH)**: 后门攻击对大规模语言模型的安全构成了严重威胁，使其在特定触发条件下表现出异常行为。触发器设计从固定触发器发展到动态或隐式触发器。这种触发器设计灵活性的增加使得防御者难以准确识别其具体形式。目前大多数后门防御方法仅适用于特定类型的触发器，或者依赖额外的干净模型的支持。为了解决这个问题，我们提出了一种基于注意力相似性的后门检测方法，该方法在无需先验触发器知识的情况下进行后门检测。我们的研究表明，受后门攻击影响的模型在面对触发器时，注意力头之间表现出异常高的相似性。基于这一观察，我们提出了一种结合注意力头微调的注意力安全性对齐方法，以纠正可能被污染的注意力头，从而有效减轻后门攻击的影响。广泛的实验结果表明，我们的方法在显著降低后门攻击成功率的同时，保留了模型在下游任务上的性能。 

---
# Scaling Patterns in Adversarial Alignment: Evidence from Multi-LLM Jailbreak Experiments 

**Title (ZH)**: adversarial alignment中的扩展模式：来自多LLM脱缰实验的证据 

**Authors**: Samuel Nathanson, Rebecca Williams, Cynthia Matuszek  

**Link**: [PDF](https://arxiv.org/pdf/2511.13788)  

**Abstract**: Large language models (LLMs) increasingly operate in multi-agent and safety-critical settings, raising open questions about how their vulnerabilities scale when models interact adversarially. This study examines whether larger models can systematically jailbreak smaller ones - eliciting harmful or restricted behavior despite alignment safeguards. Using standardized adversarial tasks from JailbreakBench, we simulate over 6,000 multi-turn attacker-target exchanges across major LLM families and scales (0.6B-120B parameters), measuring both harm score and refusal behavior as indicators of adversarial potency and alignment integrity. Each interaction is evaluated through aggregated harm and refusal scores assigned by three independent LLM judges, providing a consistent, model-based measure of adversarial outcomes. Aggregating results across prompts, we find a strong and statistically significant correlation between mean harm and the logarithm of the attacker-to-target size ratio (Pearson r = 0.51, p < 0.001; Spearman rho = 0.52, p < 0.001), indicating that relative model size correlates with the likelihood and severity of harmful completions. Mean harm score variance is higher across attackers (0.18) than across targets (0.10), suggesting that attacker-side behavioral diversity contributes more to adversarial outcomes than target susceptibility. Attacker refusal frequency is strongly and negatively correlated with harm (rho = -0.93, p < 0.001), showing that attacker-side alignment mitigates harmful responses. These findings reveal that size asymmetry influences robustness and provide exploratory evidence for adversarial scaling patterns, motivating more controlled investigations into inter-model alignment and safety. 

**Abstract (ZH)**: 大型语言模型在多代理和安全关键环境中运行时，其漏洞随着模型间的 adversarial 交互而扩展的问题悬而未决。本研究探讨了大型模型是否能系统地突破小型模型——即使有对齐保护措施，在 adversarial 交互中引发有害或受限行为。通过使用 JailbreakBench 标准化 adversarial 任务，我们在主要的 LLM 家族和规模（0.6B-120B 参数）下模拟了超过 6,000 次多轮攻击者-目标交互，以测量危害评分和拒绝行为作为对抗势能和对齐一致性的指标。每次交互通过三位独立 LLM 判官分配的综合危害和拒绝评分进行评估，提供了一致的基于模型的对抗结果度量。汇聚提示结果，我们发现平均危害与攻击者与目标规模比的对数之间存在强烈且统计显著的相关性（皮尔逊 r = 0.51，p < 0.001；斯皮尔曼 rho = 0.52，p < 0.001），表明相对模型规模与有害完成的可能性和严重性相关。平均危害评分变异性在攻击者（0.18）中高于目标（0.10），表明攻击者方面的行为多样性比目标易感性对对抗结果贡献更多。攻击者拒绝频率与危害之间存在强烈且负相关（rho = -0.93，p < 0.001），表明攻击者方面的对齐减少了有害响应。这些发现揭示了规模不对称对鲁棒性的影响，并提供了对抗扩展模式的探索性证据，促使对模型间对齐和安全进行更受控制的研究。 

---
# Can LLMs Create Legally Relevant Summaries and Analyses of Videos? 

**Title (ZH)**: LLM能否创建具有法律意义的视频摘要和分析？ 

**Authors**: Lyra Hoeben-Kuil, Gijs van Dijck, Jaromir Savelka, Johanna Gunawan, Konrad Kollnig, Marta Kolacz, Mindy Duffourc, Shashank Chakravarthy, Hannes Westermann  

**Link**: [PDF](https://arxiv.org/pdf/2511.13772)  

**Abstract**: Understanding the legally relevant factual basis of an event and conveying it through text is a key skill of legal professionals. This skill is important for preparing forms (e.g., insurance claims) or other legal documents (e.g., court claims), but often presents a challenge for laypeople. Current AI approaches aim to bridge this gap, but mostly rely on the user to articulate what has happened in text, which may be challenging for many. Here, we investigate the capability of large language models (LLMs) to understand and summarize events occurring in videos. We ask an LLM to summarize and draft legal letters, based on 120 YouTube videos showing legal issues in various domains. Overall, 71.7\% of the summaries were rated as of high or medium quality, which is a promising result, opening the door to a number of applications in e.g. access to justice. 

**Abstract (ZH)**: 理解事件的法律相关事实基础并通过文本传达是法律专业人员的一项关键技能。这项技能对于准备各种法律文件（如保险索赔或法庭诉讼请求）至关重要，但往往对普通民众来说是一个挑战。当前的AI方法旨在解决这一问题，但主要依赖用户以文本形式表述发生的事情，这对于许多人来说可能是困难的。在这里，我们研究了大语言模型（LLMs）理解并总结视频中事件的能力。我们要求一个大语言模型根据120个YouTube视频中的各种法律问题进行总结和起草法律信函。总体而言，71.7%的总结被评为高质量或中等质量，这是一个令人鼓舞的结果，为司法访问等领域打开了新的应用机会。 

---
# ExplainableGuard: Interpretable Adversarial Defense for Large Language Models Using Chain-of-Thought Reasoning 

**Title (ZH)**: ExplainableGuard：基于链式思考推理的可解释大规模语言模型对抗防御方法 

**Authors**: Shaowei Guan, Yu Zhai, Zhengyu Zhang, Yanze Wang, Hin Chi Kwok  

**Link**: [PDF](https://arxiv.org/pdf/2511.13771)  

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to adversarial attacks that can subtly manipulate their outputs. While various defense mechanisms have been proposed, many operate as black boxes, lacking transparency in their decision-making. This paper introduces ExplainableGuard, an interpretable adversarial defense framework leveraging the chain-of-thought (CoT) reasoning capabilities of DeepSeek-Reasoner. Our approach not only detects and neutralizes adversarial perturbations in text but also provides step-by-step explanations for each defense action. We demonstrate how tailored CoT prompts guide the LLM to perform a multi-faceted analysis (character, word, structural, and semantic) and generate a purified output along with a human-readable justification. Preliminary results on the GLUE Benchmark and IMDB Movie Reviews dataset show promising defense efficacy. Additionally, a human evaluation study reveals that ExplainableGuard's explanations outperform ablated variants in clarity, specificity, and actionability, with a 72.5% deployability-trust rating, underscoring its potential for more trustworthy LLM deployments. 

**Abstract (ZH)**: 大型语言模型（LLMs） increasingly vulnerable to adversarial attacks that can subtly manipulate their outputs. While various defense mechanisms have been proposed, many operate as black boxes, lacking transparency in their decision-making. This paper introduces ExplainableGuard, an interpretable adversarial defense framework leveraging the chain-of-thought (CoT) reasoning capabilities of DeepSeek-Reasoner. Our approach not only detects and neutralizes adversarial perturbations in text but also provides step-by-step explanations for each defense action. We demonstrate how tailored CoT prompts guide the LLM to perform a multi-faceted analysis (character, word, structural, and semantic) and generate a purified output along with a human-readable justification. Preliminary results on the GLUE Benchmark and IMDB Movie Reviews dataset show promising defense efficacy. Additionally, a human evaluation study reveals that ExplainableGuard's explanations outperform ablated variants in clarity, specificity, and actionability, with a 72.5% deployability-trust rating, underscoring its potential for more trustworthy LLM deployments. 

---
# PROF: An LLM-based Reward Code Preference Optimization Framework for Offline Imitation Learning 

**Title (ZH)**: PROF：一种基于LLM的离线模仿学习奖励代码偏好优化框架 

**Authors**: Shengjie Sun, Jiafei Lyu, Runze Liu, Mengbei Yan, Bo Liu, Deheng Ye, Xiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.13765)  

**Abstract**: Offline imitation learning (offline IL) enables training effective policies without requiring explicit reward annotations. Recent approaches attempt to estimate rewards for unlabeled datasets using a small set of expert demonstrations. However, these methods often assume that the similarity between a trajectory and an expert demonstration is positively correlated with the reward, which oversimplifies the underlying reward structure. We propose PROF, a novel framework that leverages large language models (LLMs) to generate and improve executable reward function codes from natural language descriptions and a single expert trajectory. We propose Reward Preference Ranking (RPR), a novel reward function quality assessment and ranking strategy without requiring environment interactions or RL training. RPR calculates the dominance scores of the reward functions, where higher scores indicate better alignment with expert preferences. By alternating between RPR and text-based gradient optimization, PROF fully automates the selection and refinement of optimal reward functions for downstream policy learning. Empirical results on D4RL demonstrate that PROF surpasses or matches recent strong baselines across numerous datasets and domains, highlighting the effectiveness of our approach. 

**Abstract (ZH)**: 线下模仿学习（offline IL）能够在不需要显式奖励标注的情况下训练有效的策略。最近的方法试图利用少量专家演示来估计未标注数据集的奖励。然而，这些方法常常假设轨迹与专家演示之间的相似性正相关于奖励，这过于简化了潜在的奖励结构。我们提出了一种新的框架PROF，该框架利用大规模语言模型（LLMs）从自然语言描述和单个专家轨迹生成并改进可执行的奖励函数代码。我们提出了一种无需环境交互或RL训练的新颖的奖励函数质量评估和排名策略RPR，它计算奖励函数的优势得分，其中更高的得分表示更好的与专家偏好对齐。通过交替使用RPR和基于文本的梯度优化，PROF完全自动化了下游策略学习中最优奖励函数的选择和优化。基于D4RL的实验结果表明，PROF在多个数据集和领域中超越或匹配了近年来的强基线，凸显了我们方法的有效性。 

---
# What happens when nanochat meets DiLoCo? 

**Title (ZH)**: 当纳诺聊天遇到DiLoCo会发生什么？ 

**Authors**: Alexander Acker, Soeren Becker, Sasho Nedelkoski, Dominik Scheinert, Odej Kao, Philipp Wiesner  

**Link**: [PDF](https://arxiv.org/pdf/2511.13761)  

**Abstract**: Although LLM training is typically centralized with high-bandwidth interconnects and large compute budgets, emerging methods target communication-constrained training in distributed environments. The model trade-offs introduced by this shift remain underexplored, and our goal is to study them.
We use the open-source nanochat project, a compact 8K-line full-stack ChatGPT-like implementation containing tokenization, pretraining, fine-tuning, and serving, as a controlled baseline. We implement the DiLoCo algorithm as a lightweight wrapper over nanochat's training loop, performing multiple local steps per worker before synchronization with an outer optimizer, effectively reducing communication by orders of magnitude. This inner-outer training is compared against a standard data-parallel (DDP) setup. Because nanochat is small and inspectable, it enables controlled pipeline adaptations and allows direct comparison with the conventional centralized baseline.
DiLoCo achieves stable convergence and competitive loss in pretraining but yields worse MMLU, GSM8K, and HumanEval scores after mid-training and SFT. We discover that using DiLoCo-pretrained weights and running mid- and post-training with DDP fails to recover performance, revealing irreversible representation drift from asynchronous updates that impairs downstream alignment. We provide this implementation as an official fork of nanochat on GitHub. 

**Abstract (ZH)**: 尽管大规模语言模型训练通常集中在一个高带宽互连和大量计算预算的环境中，新兴方法旨在应对通信受限条件下的分布式训练环境。这一转变带来的模型权衡仍处于未充分研究的状态，我们的目标是研究这些权衡。

我们使用开源的nanochat项目作为受控基线，该项目是一个包含分词、预训练、微调和部署的紧凑型8K行全栈ChatGPT-like实现。我们使用DiLoCo算法作为nanochat训练循环的一个轻量级包装器，在同步到外部优化器之前，每个worker执行多个本地步骤，有效减少了通信量级。我们将这种内循环-外循环训练与标准的数据并行（DDP）设置进行比较。由于nanochat小巧且可检查，它使我们能够控制管道调整，并直接与传统的中央基线进行比较。

DiLoCo在预训练中实现了稳定的收敛和具有竞争力的损失，但在中训练和SFT后，其MMLU、GSM8K和HumanEval评分较差。我们发现使用DiLoCo预训练的权重并在中训练和后训练过程中使用DDP无法恢复性能，揭示了来自异步更新的不可逆表示漂移，这影响了下游对齐。我们已在GitHub上提供此实现作为nanochat的官方分支。 

---
# Robustness of LLM-enabled vehicle trajectory prediction under data security threats 

**Title (ZH)**: LLM驱动的车辆轨迹预测在数据安全威胁下的鲁棒性 

**Authors**: Feilong Wang, Fuqiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.13753)  

**Abstract**: The integration of large language models (LLMs) into automated driving systems has opened new possibilities for reasoning and decision-making by transforming complex driving contexts into language-understandable representations. Recent studies demonstrate that fine-tuned LLMs can accurately predict vehicle trajectories and lane-change intentions by gathering and transforming data from surrounding vehicles. However, the robustness of such LLM-based prediction models for safety-critical driving systems remains unexplored, despite the increasing concerns about the trustworthiness of LLMs. This study addresses this gap by conducting a systematic vulnerability analysis of LLM-enabled vehicle trajectory prediction. We propose a one-feature differential evolution attack that perturbs a single kinematic feature of surrounding vehicles within the LLM's input prompts under a black-box setting. Experiments on the highD dataset reveal that even minor, physically plausible perturbations can significantly disrupt model outputs, underscoring the susceptibility of LLM-based predictors to adversarial manipulation. Further analyses reveal a trade-off between accuracy and robustness, examine the failure mechanism, and explore potential mitigation solutions. The findings provide the very first insights into adversarial vulnerabilities of LLM-driven automated vehicle models in the context of vehicular interactions and highlight the need for robustness-oriented design in future LLM-based intelligent transportation systems. 

**Abstract (ZH)**: 大型语言模型在自动驾驶系统中的集成为复杂驾驶情境的语言理解与推理提供了新可能。尽管调优后的大型语言模型能够通过周围车辆的数据采集与转换准确预测车辆轨迹和变道意图，但这类基于大型语言模型的预测模型在关键安全驾驶系统中的鲁棒性仍待探索。本研究通过系统性漏洞分析，探讨了基于大型语言模型的车辆轨迹预测模型的鲁棒性问题。我们提出了一种单一特征差异进化攻击方法，在黑盒环境下对输入提示中的周围车辆单一动力学特征进行扰动。高D数据集上的实验表明，即使是轻微的、物理上合理的扰动也能够显著干扰模型输出，突显了基于大型语言模型的预测器对抗操纵的脆弱性。进一步分析揭示了准确性和鲁棒性之间的权衡，探讨了失效机理，并探索了潜在的缓解方案。研究结果提供了首个关于基于大型语言模型的自动驾驶车辆模型在车辆交互场景下对抗漏洞的研究见解，并强调了未来基于大型语言模型的智能交通系统中需注重鲁棒性设计的需求。 

---
# AI Kill Switch for malicious web-based LLM agent 

**Title (ZH)**: AI恶意网页_based_语言模型代理的.kill开关 

**Authors**: Sechan Lee, Sangdon Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.13725)  

**Abstract**: Recently, web-based Large Language Model (LLM) agents autonomously perform increasingly complex tasks, thereby bringing significant convenience. However, they also amplify the risks of malicious misuse cases such as unauthorized collection of personally identifiable information (PII), generation of socially divisive content, and even automated web hacking. To address these threats, we propose an AI Kill Switch technique that can immediately halt the operation of malicious web-based LLM agents. To achieve this, we introduce AutoGuard - the key idea is generating defensive prompts that trigger the safety mechanisms of malicious LLM agents. In particular, generated defense prompts are transparently embedded into the website's DOM so that they remain invisible to human users but can be detected by the crawling process of malicious agents, triggering its internal safety mechanisms to abort malicious actions once read. To evaluate our approach, we constructed a dedicated benchmark consisting of three representative malicious scenarios (PII collection, social rift content generation, and web hacking attempts). Experimental results show that the AutoGuard method achieves over 80% Defense Success Rate (DSR) on malicious agents, including GPT-4o, Claude-3, and Llama3.3-70B-Instruct. It also maintains strong performance, achieving around 90% DSR on GPT-5, GPT-4.1, and Gemini-2.5-Flash when used as the malicious agent, demonstrating robust generalization across models and scenarios. Through this research, we have demonstrated the controllability of web-based LLM agents across various scenarios and models, thereby contributing to the broader effort of AI control and safety. 

**Abstract (ZH)**: 基于网页的大型语言模型恶意代理AI断言技术的研究 

---
# Signature vs. Substance: Evaluating the Balance of Adversarial Resistance and Linguistic Quality in Watermarking Large Language Models 

**Title (ZH)**: 签名 vs. 内容：评估大规模语言模型水印中对抗鲁棒性与语言质量的平衡 

**Authors**: William Guo, Adaku Uchendu, Ana Smith  

**Link**: [PDF](https://arxiv.org/pdf/2511.13722)  

**Abstract**: To mitigate the potential harms of Large Language Models (LLMs)generated text, researchers have proposed watermarking, a process of embedding detectable signals within text. With watermarking, we can always accurately detect LLM-generated texts. However, recent findings suggest that these techniques often negatively affect the quality of the generated texts, and adversarial attacks can strip the watermarking signals, causing the texts to possibly evade detection. These findings have created resistance in the wide adoption of watermarking by LLM creators. Finally, to encourage adoption, we evaluate the robustness of several watermarking techniques to adversarial attacks by comparing paraphrasing and back translation (i.e., English $\to$ another language $\to$ English) attacks; and their ability to preserve quality and writing style of the unwatermarked texts by using linguistic metrics to capture quality and writing style of texts. Our results suggest that these watermarking techniques preserve semantics, deviate from the writing style of the unwatermarked texts, and are susceptible to adversarial attacks, especially for the back translation attack. 

**Abstract (ZH)**: 为了减轻大型语言模型（LLMs）生成文本的潜在危害，研究者提出了水印技术，即在文本中嵌入可检测的信号。通过水印技术，我们始终能够准确检测到LLM生成的文本。然而，最近的研究发现，这些技术往往会降低生成文本的质量，并且对抗攻击能够去除水印信号，导致文本可能逃避检测。这些发现阻碍了LLM创建者广泛采用水印技术。最后，为了促进采用，我们通过比较改写和反向翻译（即英语$\to$另一种语言$\to$英语）攻击，评估了几种水印技术的鲁棒性；并通过语言学指标来捕获文本的质量和风格，评估其保留未水印文本质量和写作风格的能力。我们的研究结果表明，这些水印技术能够保留语义、偏离未水印文本的写作风格，并且容易受到对抗攻击的影响，尤其是对于反向翻译攻击。 

---
# From Legacy Fortran to Portable Kokkos: An Autonomous Agentic AI Workflow 

**Title (ZH)**: 从遗产Fortran到可移植的Kokkos：自主代理AI工作流 

**Authors**: Sparsh Gupta, Kamalavasan Kamalakkannan, Maxim Moraru, Galen Shipman, Patrick Diehl  

**Link**: [PDF](https://arxiv.org/pdf/2509.12443)  

**Abstract**: Scientific applications continue to rely on legacy Fortran codebases originally developed for homogeneous, CPU-based systems. As High-Performance Computing (HPC) shifts toward heterogeneous GPU-accelerated architectures, many accelerators lack native Fortran bindings, creating an urgent need to modernize legacy codes for portability. Frameworks like Kokkos provide performance portability and a single-source C++ abstraction, but manual Fortran-to-Kokkos porting demands significant expertise and time. Large language models (LLMs) have shown promise in source-to-source code generation, yet their use in fully autonomous workflows for translating and optimizing parallel code remains largely unexplored, especially for performance portability across diverse hardware. This paper presents an agentic AI workflow where specialized LLM "agents" collaborate to translate, validate, compile, run, test, debug, and optimize Fortran kernels into portable Kokkos C++ programs. Results show the pipeline modernizes a range of benchmark kernels, producing performance-portable Kokkos codes across hardware partitions. Paid OpenAI models such as GPT-5 and o4-mini-high executed the workflow for only a few U.S. dollars, generating optimized codes that surpassed Fortran baselines, whereas open-source models like Llama4-Maverick often failed to yield functional codes. This work demonstrates the feasibility of agentic AI for Fortran-to-Kokkos transformation and offers a pathway for autonomously modernizing legacy scientific applications to run portably and efficiently on diverse supercomputers. It further highlights the potential of LLM-driven agentic systems to perform structured, domain-specific reasoning tasks in scientific and systems-oriented applications. 

**Abstract (ZH)**: 科学应用仍然依赖于最初为同构CPU系统开发的遗留Fortran代码。随着高性能计算（HPC）转向异构GPU加速架构，许多加速器缺乏原生Fortran绑定，这迫切需要现代化遗留代码以实现兼容性。Kokkos等框架提供了性能兼容性和单一来源的C++抽象，但手动将Fortran转换为Kokkos需要大量的专业知识和时间。大型语言模型（LLMs）在源到源代码生成方面展示了潜力，但在完全自主的工作流中将并行代码翻译和优化以实现性能兼容性方面仍鲜有探索，尤其是在不同硬件之间。本文介绍了由专门的LLM“代理”协作完成的能够翻译、验证、编译、运行、测试和优化Fortran内核为可移植的Kokkos C++程序的自主AI工作流。结果表明，该流水线现代化了多种基准内核，生成了跨硬件分区的性能兼容的Kokkos代码。使用付费的OpenAI模型如GPT-5和o4-mini-high执行此工作流仅花费了几美元，并生成了超越Fortran基线的优化代码，而开源模型如Llama4-Maverick往往无法生成功能正常的代码。本文展示了Fortran到Kokkos转换的自主AI可行性和为跨不同超级计算机自主现代化遗留科学应用提供了途径。此外，该工作还突显了由LLM驱动的自主系统在科学和系统化应用中执行结构化、领域特定推理任务的潜力。 

---
