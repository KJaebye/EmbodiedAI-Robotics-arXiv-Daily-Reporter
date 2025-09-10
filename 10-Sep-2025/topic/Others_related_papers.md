# Timing the Message: Language-Based Notifications for Time-Critical Assistive Settings 

**Title (ZH)**: Timing the Message: 基于语言的通知在时间敏感的辅助设置中 

**Authors**: Ya-Chuan Hsu, Jonathan DeCastro, Andrew Silva, Guy Rosman  

**Link**: [PDF](https://arxiv.org/pdf/2509.07438)  

**Abstract**: In time-critical settings such as assistive driving, assistants often rely on alerts or haptic signals to prompt rapid human attention, but these cues usually leave humans to interpret situations and decide responses independently, introducing potential delays or ambiguity in meaning. Language-based assistive systems can instead provide instructions backed by context, offering more informative guidance. However, current approaches (e.g., social assistive robots) largely prioritize content generation while overlooking critical timing factors such as verbal conveyance duration, human comprehension delays, and subsequent follow-through duration. These timing considerations are crucial in time-critical settings, where even minor delays can substantially affect outcomes. We aim to study this inherent trade-off between timeliness and informativeness by framing the challenge as a sequential decision-making problem using an augmented-state Markov Decision Process. We design a framework combining reinforcement learning and a generated offline taxonomy dataset, where we balance the trade-off while enabling a scalable taxonomy dataset generation pipeline. Empirical evaluation with synthetic humans shows our framework improves success rates by over 40% compared to methods that ignore time delays, while effectively balancing timeliness and informativeness. It also exposes an often-overlooked trade-off between these two factors, opening new directions for optimizing communication in time-critical human-AI assistance. 

**Abstract (ZH)**: 在时间关键设置下基于语言的辅助系统的时间及时性和信息量权衡研究 

---
# TransMPC: Transformer-based Explicit MPC with Variable Prediction Horizon 

**Title (ZH)**: 基于变换器的显式模型预测控制及可变预测 horizons 方法 

**Authors**: Sichao Wu, Jiang Wu, Xingyu Cao, Fawang Zhang, Guangyuan Yu, Junjie Zhao, Yue Qu, Fei Ma, Jingliang Duan  

**Link**: [PDF](https://arxiv.org/pdf/2509.07381)  

**Abstract**: Traditional online Model Predictive Control (MPC) methods often suffer from excessive computational complexity, limiting their practical deployment. Explicit MPC mitigates online computational load by pre-computing control policies offline; however, existing explicit MPC methods typically rely on simplified system dynamics and cost functions, restricting their accuracy for complex systems. This paper proposes TransMPC, a novel Transformer-based explicit MPC algorithm capable of generating highly accurate control sequences in real-time for complex dynamic systems. Specifically, we formulate the MPC policy as an encoder-only Transformer leveraging bidirectional self-attention, enabling simultaneous inference of entire control sequences in a single forward pass. This design inherently accommodates variable prediction horizons while ensuring low inference latency. Furthermore, we introduce a direct policy optimization framework that alternates between sampling and learning phases. Unlike imitation-based approaches dependent on precomputed optimal trajectories, TransMPC directly optimizes the true finite-horizon cost via automatic differentiation. Random horizon sampling combined with a replay buffer provides independent and identically distributed (i.i.d.) training samples, ensuring robust generalization across varying states and horizon lengths. Extensive simulations and real-world vehicle control experiments validate the effectiveness of TransMPC in terms of solution accuracy, adaptability to varying horizons, and computational efficiency. 

**Abstract (ZH)**: 基于Transformer的实时复杂动态系统模型预测控制算法（TransMPC） 

---
# Knowledge Isn't Power: The Ethics of Social Robots and the Difficulty of Informed Consent 

**Title (ZH)**: 知识不是力量：社会机器人的伦理问题与知情同意的难度 

**Authors**: James M. Berzuk, Lauren Corcoran, Brannen McKenzie-Lefurgey, Katie Szilagyi, James E. Young  

**Link**: [PDF](https://arxiv.org/pdf/2509.07942)  

**Abstract**: Contemporary robots are increasingly mimicking human social behaviours to facilitate interaction, such as smiling to signal approachability, or hesitating before taking an action to allow people time to react. Such techniques can activate a person's entrenched social instincts, triggering emotional responses as though they are interacting with a fellow human, and can prompt them to treat a robot as if it truly possesses the underlying life-like processes it outwardly presents, raising significant ethical questions. We engage these issues through the lens of informed consent: drawing upon prevailing legal principles and ethics, we examine how social robots can influence user behaviour in novel ways, and whether under those circumstances users can be appropriately informed to consent to these heightened interactions. We explore the complex circumstances of human-robot interaction and highlight how it differs from more familiar interaction contexts, and we apply legal principles relating to informed consent to social robots in order to reconceptualize the current ethical debates surrounding the field. From this investigation, we synthesize design goals for robot developers to achieve more ethical and informed human-robot interaction. 

**Abstract (ZH)**: 当代机器人越来越模仿人类社会行为以促进互动，例如微笑以表示亲和力，或在采取行动前犹豫以允许人们作出反应。这些技术可以激活人的根深蒂固的社会本能，引发情感反应，仿佛在与真人互动，并可能促使人们将机器人视为具有其外在呈现的生命过程的真实主体，从而引发重大的伦理问题。我们从知情同意的角度探讨这些议题：借鉴现有的法律规定和伦理原则，我们考察社会机器人如何以新颖的方式影响用户行为，并评估在这种情况下用户是否可以适当知情并同意进行这些增强的互动。我们探讨了人机互动的复杂情况，指出现有的互动情境有所不同，并将知情同意的法律原则应用于社会机器人，以重新构想该领域的当前伦理争端。基于这一调查，我们总结了机器人开发者的设计目标，以实现更为伦理和知情的人机互动。 

---
# Adaptive Evolutionary Framework for Safe, Efficient, and Cooperative Autonomous Vehicle Interactions 

**Title (ZH)**: 适应性进化框架实现安全、高效与协作的自主车辆交互 

**Authors**: Zhen Tian, Zhihao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.07411)  

**Abstract**: Modern transportation systems face significant challenges in ensuring road safety, given serious injuries caused by road accidents. The rapid growth of autonomous vehicles (AVs) has prompted new traffic designs that aim to optimize interactions among AVs. However, effective interactions between AVs remains challenging due to the absence of centralized control. Besides, there is a need for balancing multiple factors, including passenger demands and overall traffic efficiency. Traditional rule-based, optimization-based, and game-theoretic approaches each have limitations in addressing these challenges. Rule-based methods struggle with adaptability and generalization in complex scenarios, while optimization-based methods often require high computational resources. Game-theoretic approaches, such as Stackelberg and Nash games, suffer from limited adaptability and potential inefficiencies in cooperative settings. This paper proposes an Evolutionary Game Theory (EGT)-based framework for AV interactions that overcomes these limitations by utilizing a decentralized and adaptive strategy evolution mechanism. A causal evaluation module (CEGT) is introduced to optimize the evolutionary rate, balancing mutation and evolution by learning from historical interactions. Simulation results demonstrate the proposed CEGT outperforms EGT and popular benchmark games in terms of lower collision rates, improved safety distances, higher speeds, and overall better performance compared to Nash and Stackelberg games across diverse scenarios and parameter settings. 

**Abstract (ZH)**: 基于进化博弈论的自动驾驶车辆交互框架：通过去中心化和自适应策略进化机制克服传统方法的局限性 

---
# Efficient Multi-Agent Coordination via Dynamic Joint-State Graph Construction 

**Title (ZH)**: 动态联合状态图构建下的高效多智能体协调 

**Authors**: Yanlin Zhou, Manshi Limbu, Xuesu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.07234)  

**Abstract**: Multi-agent pathfinding (MAPF) traditionally focuses on collision avoidance, but many real-world applications require active coordination between agents to improve team performance. This paper introduces Team Coordination on Graphs with Risky Edges (TCGRE), where agents collaborate to reduce traversal costs on high-risk edges via support from teammates. We reformulate TCGRE as a 3D matching problem-mapping robot pairs, support pairs, and time steps-and rigorously prove its NP-hardness via reduction from Minimum 3D Matching. To address this complexity, (in the conference version) we proposed efficient decomposition methods, reducing the problem to tractable subproblems: Joint-State Graph (JSG): Encodes coordination as a single-agent shortest-path problem. Coordination-Exhaustive Search (CES): Optimizes support assignments via exhaustive pairing. Receding-Horizon Optimistic Cooperative A* (RHOCA*): Balances optimality and scalability via horizon-limited planning. Further in this extension, we introduce a dynamic graph construction method (Dynamic-HJSG), leveraging agent homogeneity to prune redundant states and reduce computational overhead by constructing the joint-state graph dynamically. Theoretical analysis shows Dynamic-HJSG preserves optimality while lowering complexity from exponential to polynomial in key cases. Empirical results validate scalability for large teams and graphs, with HJSG outperforming baselines greatly in runtime in different sizes and types of graphs. This work bridges combinatorial optimization and multi-agent planning, offering a principled framework for collaborative pathfinding with provable guarantees, and the key idea of the solution can be widely extended to many other collaborative optimization problems, such as MAPF. 

**Abstract (ZH)**: 基于风险边的图上团队协调多智能体路径规划（Team Coordination on Graphs with Risky Edges for Multi-Agent Pathfinding） 

---
# HiPhO: How Far Are (M)LLMs from Humans in the Latest High School Physics Olympiad Benchmark? 

**Title (ZH)**: HiPhO: (M)LLMs在最新高中物理奥林匹克基准测试中与人类相距多远？ 

**Authors**: Fangchen Yu, Haiyuan Wan, Qianjia Cheng, Yuchen Zhang, Jiacheng Chen, Fujun Han, Yulun Wu, Junchi Yao, Ruilizhen Hu, Ning Ding, Yu Cheng, Tao Chen, Lei Bai, Dongzhan Zhou, Yun Luo, Ganqu Cui, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.07894)  

**Abstract**: Recently, the physical capabilities of (M)LLMs have garnered increasing attention. However, existing benchmarks for physics suffer from two major gaps: they neither provide systematic and up-to-date coverage of real-world physics competitions such as physics Olympiads, nor enable direct performance comparison with humans. To bridge these gaps, we present HiPhO, the first benchmark dedicated to high school physics Olympiads with human-aligned evaluation. Specifically, HiPhO highlights three key innovations. (1) Comprehensive Data: It compiles 13 latest Olympiad exams from 2024-2025, spanning both international and regional competitions, and covering mixed modalities that encompass problems spanning text-only to diagram-based. (2) Professional Evaluation: We adopt official marking schemes to perform fine-grained grading at both the answer and step level, fully aligned with human examiners to ensure high-quality and domain-specific evaluation. (3) Comparison with Human Contestants: We assign gold, silver, and bronze medals to models based on official medal thresholds, thereby enabling direct comparison between (M)LLMs and human contestants. Our large-scale evaluation of 30 state-of-the-art (M)LLMs shows that: across 13 exams, open-source MLLMs mostly remain at or below the bronze level; open-source LLMs show promising progress with occasional golds; closed-source reasoning MLLMs can achieve 6 to 12 gold medals; and most models still have a significant gap from full marks. These results highlight a substantial performance gap between open-source models and top students, the strong physical reasoning capabilities of closed-source reasoning models, and the fact that there is still significant room for improvement. HiPhO, as a rigorous, human-aligned, and Olympiad-focused benchmark for advancing multimodal physical reasoning, is open-source and available at this https URL. 

**Abstract (ZH)**: 近年来，大语言模型的物理能力逐渐受到关注。然而，现有的物理基准测试存在两个主要缺陷：它们既未提供对实际物理竞赛如物理奥林匹克的系统和最新的覆盖，也未能直接与人类进行性能对比。为弥补这些缺陷，我们提出了HiPhO，这是首个针对高中生物理奥林匹克的人工智能评估基准。具体而言，HiPhO 以三大创新为亮点。（1）全面数据：汇集了2024-2025年的最新13场奥林匹克竞赛试题，涵盖了国际和区域竞赛，并包含了从纯文本问题到基于图表的问题的不同模态。（2）专业评估：采用官方评分方案进行细粒度的答对和步骤评分，确保与人类考官完全一致，以提供高质量且领域的特定评估。（3）与人类对手的对比：基于官方奖牌标准向模型分配金牌、银牌和铜牌，从而可以直接比较（M）LLMs与人类对手。对30个最先进的（M）LLMs的大规模评估显示：在13场考试中，开源（M）LLMs多停留在或低于铜牌水平；开源LLMs显示出有望的进展，偶尔获得金牌；封闭源推理（M）LLMs能够获得6至12枚金牌；而大多数模型仍然与满分存在较大差距。这些结果突显了开源模型与顶尖学生之间的表现差距，以及封闭源推理模型强大的物理推理能力，并表明仍有改进的空间。作为严格、人工智能对齐且专注于奥林匹克竞赛的多模态物理推理基准，HiPhO 是开源的，可访问此链接：this https URL。 

---
# CP-Model-Zoo: A Natural Language Query System for Constraint Programming Models 

**Title (ZH)**: CP-模型动物园：约束编程模型的自然语言查询系统 

**Authors**: Augustin Crespin, Ioannis Kostis, Hélène Verhaeghe, Pierre Schaus  

**Link**: [PDF](https://arxiv.org/pdf/2509.07867)  

**Abstract**: Constraint Programming and its high-level modeling languages have long been recognized for their potential to achieve the holy grail of problem-solving. However, the complexity of modeling languages, the large number of global constraints, and the art of creating good models have often hindered non-experts from choosing CP to solve their combinatorial problems. While generating an expert-level model from a natural-language description of a problem would be the dream, we are not yet there. We propose a tutoring system called CP-Model-Zoo, exploiting expert-written models accumulated through the years. CP-Model-Zoo retrieves the closest source code model from a database based on a user's natural language description of a combinatorial problem. It ensures that expert-validated models are presented to the user while eliminating the need for human data labeling. Our experiments show excellent accuracy in retrieving the correct model based on a user-input description of a problem simulated with different levels of expertise. 

**Abstract (ZH)**: 约束编程及其高级建模语言长期被认为具有实现问题求解圣杯的潜力。然而，建模语言的复杂性、大量全局约束以及创建良好模型的艺术往往阻碍了非专家选择CP来解决组合问题。虽然从自然语言问题描述中生成专家级模型是理想的选择，但目前我们尚未达到这一目标。我们提出了一种名为CP-Model-Zoo的辅导系统，利用多年积累的专家编写模型。CP-Model-Zoo根据用户对组合问题的自然语言描述，从数据库中检索最接近的源代码模型，并确保向用户呈现经专家验证的模型，从而消除人工数据标注的需要。我们的实验显示，根据用户输入的问题描述，CP-Model-Zoo在不同专家水平下检索正确模型的准确性非常出色。 

---
# The Carbon Footprint Wizard: A Knowledge-Augmented AI Interface for Streamlining Food Carbon Footprint Analysis 

**Title (ZH)**: 碳足迹魔杖：一种增强知识的AI界面，用于简化食品碳足迹分析 

**Authors**: Mustafa Kaan Aslan, Reinout Heijungs, Filip Ilievski  

**Link**: [PDF](https://arxiv.org/pdf/2509.07733)  

**Abstract**: Environmental sustainability, particularly in relation to climate change, is a key concern for consumers, producers, and policymakers. The carbon footprint, based on greenhouse gas emissions, is a standard metric for quantifying the contribution to climate change of activities and is often assessed using life cycle assessment (LCA). However, conducting LCA is complex due to opaque and global supply chains, as well as fragmented data. This paper presents a methodology that combines advances in LCA and publicly available databases with knowledge-augmented AI techniques, including retrieval-augmented generation, to estimate cradle-to-gate carbon footprints of food products. We introduce a chatbot interface that allows users to interactively explore the carbon impact of composite meals and relate the results to familiar activities. A live web demonstration showcases our proof-of-concept system with arbitrary food items and follow-up questions, highlighting both the potential and limitations - such as database uncertainties and AI misinterpretations - of delivering LCA insights in an accessible format. 

**Abstract (ZH)**: 环境可持续性，特别是在气候变化方面的可持续性，是消费者、生产者和政策制定者的关键关注点。基于温室气体排放的碳足迹是衡量活动对气候变化贡献的标准指标，常使用生命周期评估（LCA）进行评估。然而，由于供应链不透明和数据碎片化，开展LCA十分复杂。本文提出了一种方法，结合了LCA的进步和公开可用的数据库，以及知识增强的AI技术，包括检索增强生成，以估算食品产品的从摇篮到工厂门的碳足迹。我们引入了一个聊天机器人界面，让用户交互式地探索复合餐食的碳影响，并将结果与熟悉的活动联系起来。现场网络演示展示了我们的概念验证系统，使用任意食品项目并提出后续问题，突出了以易于访问的形式提供LCA洞察的潜力和限制，如数据库不确定性和AI误解释。 

---
# BDPM: A Machine Learning-Based Feature Extractor for Parkinson's Disease Classification via Gut Microbiota Analysis 

**Title (ZH)**: BDPM：一种基于机器学习的肠道微生物特征提取器，用于帕金森病分类 

**Authors**: Bo Yu, Zhixiu Hua, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.07723)  

**Abstract**: Background: Parkinson's disease remains a major neurodegenerative disorder with high misdiagnosis rates, primarily due to reliance on clinical rating scales. Recent studies have demonstrated a strong association between gut microbiota and Parkinson's disease, suggesting that microbial composition may serve as a promising biomarker. Although deep learning models based ongut microbiota show potential for early prediction, most approaches rely on single classifiers and often overlook inter-strain correlations or temporal dynamics. Therefore, there is an urgent need for more robust feature extraction methods tailored to microbiome data. Methods: We proposed BDPM (A Machine Learning-Based Feature Extractor for Parkinson's Disease Classification via Gut Microbiota Analysis). First, we collected gut microbiota profiles from 39 Parkinson's patients and their healthy spouses to identify differentially abundant taxa. Second, we developed an innovative feature selection framework named RFRE (Random Forest combined with Recursive Feature Elimination), integrating ecological knowledge to enhance biological interpretability. Finally, we designed a hybrid classification model to capture temporal and spatial patterns in microbiome data. 

**Abstract (ZH)**: 背景：帕金森病仍然是一个主要的神经退行性疾病，由于主要依赖临床评分量表，存在较高的误诊率。最近的研究表明肠道微生物群与帕金森病之间存在密切关联，提示微生物组成可能成为有希望的生物标志物。尽管基于肠道微生物群的深度学习模型显示出早期预测的潜力，但大多数方法依赖单一分类器，并且通常忽视不同菌株之间的关联或时间动态。因此，迫切需要针对微生物组数据的更 robust 的特征提取方法。方法：我们提出了 BDPM（基于机器学习的帕金森病分类肠道微生物特征提取方法）。首先，从39名帕金森病患者及其健康配偶中收集了肠道微生物群谱型，以识别差异丰度的菌种。其次，我们开发了一种名为 RFRE（随机森林结合递归特征消除）的创新特征选择框架，将生态学知识集成以增强生物学可解释性。最后，我们设计了一种综合分类模型，以捕捉微生物组数据中的时间和空间模式。 

---
# RIMO: An Easy-to-Evaluate, Hard-to-Solve Olympiad Benchmark for Advanced Mathematical Reasoning 

**Title (ZH)**: RIMO：一个易于评估、难以解决的奥林匹亚数学推理基准測试 

**Authors**: Ziye Chen, Chengwei Qin, Yao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07711)  

**Abstract**: As large language models (LLMs) reach high scores on established mathematical benchmarks, such as GSM8K and MATH, the research community has turned to International Mathematical Olympiad (IMO) problems to push the evaluation frontier. However, existing Olympiad-level benchmarks suffer from practical constraints that introduce grading noise and potential bias, such as heterogeneous answer formats requiring model-based judges and a reliance on potentially flawed solutions. We introduce RIMO, a two-track benchmark designed to preserve peak Olympiad difficulty while eliminating this evaluation noise. The first track, RIMO-N, rewrites 335 IMO problems to admit a single, unique integer answer, allowing for deterministic correctness checking. The second track, RIMO-P, features 456 proof problems with expert-checked solutions, which are decomposed into a sequence of sub-problems to evaluate the step-by-step reasoning process via an automated grading system. Our benchmarking of ten frontier LLMs, including GPT-4o and Gemini 2.5 Flash, reveals that while these systems excel on older benchmarks, their performance drops sharply on RIMO. These results highlight a substantial gap between current LLM capabilities and actual Olympiad-level reasoning. By providing a challenging yet easy-to-evaluate suite, RIMO offers a high-resolution yardstick for future research, presenting a clear target for closing the profound reasoning gap our findings expose. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在GSM8K和MATH等已建立的数学基准测试中取得高分，研究界转向国际数学奥林匹克（IMO）问题以推动评估前沿。然而，现有的奥林匹克级别基准测试存在一些实际限制，这些限制引入了评分噪音和潜在偏见，例如异质化的答案格式需要基于模型的评判员以及对潜在有缺陷的解决方案的依赖。我们介绍了RIMO，这是一种双轨基准测试，旨在保留奥林匹克难度的巅峰同时消除这种评估噪音。第一轨RIMO-N重写了335道IMO问题，使其仅有一个唯一的整数答案，允许进行确定性的正确性检查。第二轨RIMO-P包含456道证明问题，配有专家检查的解决方案，并将其分解为一系列子问题，通过自动化评分系统评估逐步推理过程。对包括GPT-4o和Gemini 2.5 Flash在内的十种前沿LLM的基准测试表明，虽然这些系统在较早的基准测试中表现出色，但在RIMO上的表现则急剧下降。这些结果突显了当前LLM能力与实际奥林匹克水平推理之间的巨大差距。通过提供具有挑战性但易于评估的套件，RIMO为未来研究提供了一个高分辨率的标尺，明确了需要填补我们发现的深刻推理差距的目标。 

---
# FHIR-RAG-MEDS: Integrating HL7 FHIR with Retrieval-Augmented Large Language Models for Enhanced Medical Decision Support 

**Title (ZH)**: FHIR-RAG-MEDS: 将HL7 FHIR与检索增强的大语言模型集成以增强医疗决策支持 

**Authors**: Yildiray Kabak, Gokce B. Laleci Erturkmen, Mert Gencturk, Tuncay Namli, A. Anil Sinaci, Ruben Alcantud Corcoles, Cristina Gomez Ballesteros, Pedro Abizanda, Asuman Dogac  

**Link**: [PDF](https://arxiv.org/pdf/2509.07706)  

**Abstract**: In this study, we propose FHIR-RAG-MEDS system that aims to integrate Health Level 7 Fast Healthcare Interoperability Resources (HL7 FHIR) with a Retrieval-Augmented Generation (RAG)-based system to improve personalized medical decision support on evidence-based clinical guidelines, emphasizing the need for research in practical applications. In the evolving landscape of medical decision support systems, integrating advanced technologies such as RAG and HL7 FHIR can significantly enhance clinical decision-making processes. Despite the potential of these technologies, there is limited research on their integration in practical applications. 

**Abstract (ZH)**: 本研究提出FHIR-RAG-MEDS系统，旨在将Health Level 7 Fast Healthcare Interoperability Resources (HL7 FHIR)与基于检索增强生成（RAG）的系统相结合，以提高基于循证临床指南的个性化医疗决策支持，并强调了在实际应用中研究这些技术集成的必要性。在医疗决策支持系统不断发展的情景下，整合如RAG和HL7 FHIR等先进技术可以显著增强临床决策过程。尽管这些技术具有巨大的潜力，但在实际应用中的集成研究仍然有限。 

---
# DeepGraphLog for Layered Neurosymbolic AI 

**Title (ZH)**: DeepGraphLog 用于分层神经符号AI 

**Authors**: Adem Kikaj, Giuseppe Marra, Floris Geerts, Robin Manhaeve, Luc De Raedt  

**Link**: [PDF](https://arxiv.org/pdf/2509.07665)  

**Abstract**: Neurosymbolic AI (NeSy) aims to integrate the statistical strengths of neural networks with the interpretability and structure of symbolic reasoning. However, current NeSy frameworks like DeepProbLog enforce a fixed flow where symbolic reasoning always follows neural processing. This restricts their ability to model complex dependencies, especially in irregular data structures such as graphs. In this work, we introduce DeepGraphLog, a novel NeSy framework that extends ProbLog with Graph Neural Predicates. DeepGraphLog enables multi-layer neural-symbolic reasoning, allowing neural and symbolic components to be layered in arbitrary order. In contrast to DeepProbLog, which cannot handle symbolic reasoning via neural methods, DeepGraphLog treats symbolic representations as graphs, enabling them to be processed by Graph Neural Networks (GNNs). We showcase the capabilities of DeepGraphLog on tasks in planning, knowledge graph completion with distant supervision, and GNN expressivity. Our results demonstrate that DeepGraphLog effectively captures complex relational dependencies, overcoming key limitations of existing NeSy systems. By broadening the applicability of neurosymbolic AI to graph-structured domains, DeepGraphLog offers a more expressive and flexible framework for neural-symbolic integration. 

**Abstract (ZH)**: 神经符号AI（NeSy）旨在结合神经网络的统计优势和符号推理的可解释性和结构。然而，当前的NeSy框架如DeepProbLog固定了符号推理总是跟随神经处理的流程，这限制了它们建模复杂依赖关系的能力，特别是在如图形结构的非规则数据结构中。在这项工作中，我们提出了DeepGraphLog，这是一种将ProbLog扩展为图神经谓词的新型NeSy框架。DeepGraphLog支持多层神经符号推理，允许神经和符号组件以任意顺序层叠。与DeepProbLog不同的是，它可以通过神经方法进行符号推理，并将符号表示视为图形，使它们能够被图神经网络（GNNs）处理。我们通过规划任务、带有远处监督的知识图补全任务以及GNN表达性展示DeepGraphLog的能力。我们的结果显示，DeepGraphLog有效地捕捉了复杂的关系依赖关系，克服了现有NeSy系统的关键限制。通过将神经符号AI的应用扩展到图形结构领域，DeepGraphLog为神经符号集成提供了一个更具表达性和灵活性的框架。 

---
# Towards explainable decision support using hybrid neural models for logistic terminal automation 

**Title (ZH)**: 面向物流终端自动化中的混合神经模型可解释决策支持 

**Authors**: Riccardo DElia, Alberto Termine, Francesco Flammini  

**Link**: [PDF](https://arxiv.org/pdf/2509.07577)  

**Abstract**: The integration of Deep Learning (DL) in System Dynamics (SD) modeling for transportation logistics offers significant advantages in scalability and predictive accuracy. However, these gains are often offset by the loss of explainability and causal reliability $-$ key requirements in critical decision-making systems. This paper presents a novel framework for interpretable-by-design neural system dynamics modeling that synergizes DL with techniques from Concept-Based Interpretability, Mechanistic Interpretability, and Causal Machine Learning. The proposed hybrid approach enables the construction of neural network models that operate on semantically meaningful and actionable variables, while retaining the causal grounding and transparency typical of traditional SD models. The framework is conceived to be applied to real-world case-studies from the EU-funded project AutoMoTIF, focusing on data-driven decision support, automation, and optimization of multimodal logistic terminals. We aim at showing how neuro-symbolic methods can bridge the gap between black-box predictive models and the need for critical decision support in complex dynamical environments within cyber-physical systems enabled by the industrial Internet-of-Things. 

**Abstract (ZH)**: 基于深度学习的系统动力学建模在交通物流中的集成：一种可解释的神经系统动力学建模框架 

---
# SheetDesigner: MLLM-Powered Spreadsheet Layout Generation with Rule-Based and Vision-Based Reflection 

**Title (ZH)**: SheetDesigner：基于规则和视觉反馈的MLLM驱动的表格布局生成 

**Authors**: Qin Chen, Yuanyi Ren, Xiaojun Ma, Mugeng Liu, Han Shi, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07473)  

**Abstract**: Spreadsheets are critical to data-centric tasks, with rich, structured layouts that enable efficient information transmission. Given the time and expertise required for manual spreadsheet layout design, there is an urgent need for automated solutions. However, existing automated layout models are ill-suited to spreadsheets, as they often (1) treat components as axis-aligned rectangles with continuous coordinates, overlooking the inherently discrete, grid-based structure of spreadsheets; and (2) neglect interrelated semantics, such as data dependencies and contextual links, unique to spreadsheets. In this paper, we first formalize the spreadsheet layout generation task, supported by a seven-criterion evaluation protocol and a dataset of 3,326 spreadsheets. We then introduce SheetDesigner, a zero-shot and training-free framework using Multimodal Large Language Models (MLLMs) that combines rule and vision reflection for component placement and content population. SheetDesigner outperforms five baselines by at least 22.6\%. We further find that through vision modality, MLLMs handle overlap and balance well but struggle with alignment, necessitates hybrid rule and visual reflection strategies. Our codes and data is available at Github. 

**Abstract (ZH)**: 基于表格的设计:一种结合规则和视觉反射的零样本框架 

---
# Performative Thinking? The Brittle Correlation Between CoT Length and Problem Complexity 

**Title (ZH)**: 表演性思维？CoT长度与问题复杂性之间的脆弱关联 

**Authors**: Vardhan Palod, Karthik Valmeekam, Kaya Stechly, Subbarao Kambhampati  

**Link**: [PDF](https://arxiv.org/pdf/2509.07339)  

**Abstract**: Intermediate token generation (ITG), where a model produces output before the solution, has been proposed as a method to improve the performance of language models on reasoning tasks. While these reasoning traces or Chain of Thoughts (CoTs) are correlated with performance gains, the mechanisms underlying them remain unclear. A prevailing assumption in the community has been to anthropomorphize these tokens as "thinking", treating longer traces as evidence of higher problem-adaptive computation. In this work, we critically examine whether intermediate token sequence length reflects or correlates with problem difficulty. To do so, we train transformer models from scratch on derivational traces of the A* search algorithm, where the number of operations required to solve a maze problem provides a precise and verifiable measure of problem complexity. We first evaluate the models on trivial free-space problems, finding that even for the simplest tasks, they often produce excessively long reasoning traces and sometimes fail to generate a solution. We then systematically evaluate the model on out-of-distribution problems and find that the intermediate token length and ground truth A* trace length only loosely correlate. We notice that the few cases where correlation appears are those where the problems are closer to the training distribution, suggesting that the effect arises from approximate recall rather than genuine problem-adaptive computation. This suggests that the inherent computational complexity of the problem instance is not a significant factor, but rather its distributional distance from the training data. These results challenge the assumption that intermediate trace generation is adaptive to problem difficulty and caution against interpreting longer sequences in systems like R1 as automatically indicative of "thinking effort". 

**Abstract (ZH)**: Intermediate Token Generation 面向推理任务的语言模型性能提升机制探究：问题难度与中间token序列长度的关系 

---
# BlendedNet: A Blended Wing Body Aircraft Dataset and Surrogate Model for Aerodynamic Predictions 

**Title (ZH)**: BlendedNet: 一体化机翼机身飞行器数据集及气动预测代理模型 

**Authors**: Nicholas Sung, Steven Spreizer, Mohamed Elrefaie, Kaira Samuel, Matthew C. Jones, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2509.07209)  

**Abstract**: BlendedNet is a publicly available aerodynamic dataset of 999 blended wing body (BWB) geometries. Each geometry is simulated across about nine flight conditions, yielding 8830 converged RANS cases with the Spalart-Allmaras model and 9 to 14 million cells per case. The dataset is generated by sampling geometric design parameters and flight conditions, and includes detailed pointwise surface quantities needed to study lift and drag. We also introduce an end-to-end surrogate framework for pointwise aerodynamic prediction. The pipeline first uses a permutation-invariant PointNet regressor to predict geometric parameters from sampled surface point clouds, then conditions a Feature-wise Linear Modulation (FiLM) network on the predicted parameters and flight conditions to predict pointwise coefficients Cp, Cfx, and Cfz. Experiments show low errors in surface predictions across diverse BWBs. BlendedNet addresses data scarcity for unconventional configurations and enables research on data-driven surrogate modeling for aerodynamic design. 

**Abstract (ZH)**: BlendedNet是公开可用的999种混合翼体（BWB）气动数据集，包含约九种飞行条件的仿真，生成8830个使用Spalart-Allmaras模型收敛的RANS案例，每案例包含9至14百万个细胞，并包含用于研究升力和阻力的详细表面点位量。还引入了一种端到端的代理框架进行点位气动预测。管道首先使用不变排列的PointNet回归器从采样的表面点云预测几何参数，然后根据预测的参数和飞行条件条件化Feature-wise Linear Modulation（FiLM）网络，预测点位系数Cp、Cfx和Cfz。实验显示在多种BWB表面预测中具有低误差。BlendedNet解决了非常规配置的数据稀缺问题，并促进了基于数据驱动的气动设计代理建模研究。 

---
# A Hybrid CNN-LSTM Deep Learning Model for Intrusion Detection in Smart Grid 

**Title (ZH)**: 智能电网中入侵检测的混合CNN-LSTM深层学习模型 

**Authors**: Abdulhakim Alsaiari, Mohammad Ilyas  

**Link**: [PDF](https://arxiv.org/pdf/2509.07208)  

**Abstract**: The evolution of the traditional power grid into the "smart grid" has resulted in a fundamental shift in energy management, which allows the integration of renewable energy sources with modern communication technology. However, this interconnection has increased smart grids' vulnerability to attackers, which might result in privacy breaches, operational interruptions, and massive outages. The SCADA-based smart grid protocols are critical for real-time data collection and control, but they are vulnerable to attacks like unauthorized access and denial of service (DoS). This research proposes a hybrid deep learning-based Intrusion Detection System (IDS) intended to improve the cybersecurity of smart grids. The suggested model takes advantage of Convolutional Neural Networks' (CNN) feature extraction capabilities as well as Long Short-Term Memory (LSTM) networks' temporal pattern recognition skills. DNP3 and IEC104 intrusion detection datasets are employed to train and test our CNN-LSTM model to recognize and classify the potential cyber threats. Compared to other deep learning approaches, the results demonstrate considerable improvements in accuracy, precision, recall, and F1-score, with a detection accuracy of 99.70%. 

**Abstract (ZH)**: 传统电网向“智能电网”的演进引发了能源管理的根本性转变，允许可再生能源源与现代通信技术的整合。然而，这种互联增加了智能电网对攻击者的脆弱性，可能导致隐私泄露、操作中断和大规模停电。基于SCADA的智能电网协议对于实时数据采集和控制至关重要，但易受未授权访问和拒绝服务（DoS）等攻击的影响。本研究提议了一种基于混合深度学习的入侵检测系统（IDS），旨在提高智能电网的网络安全性能。所建议的模型利用了卷积神经网络（CNN）的特征提取能力和长短期记忆网络（LSTM）的时间序列模式识别能力。使用DNP3和IEC104入侵检测数据集来训练和测试我们的CNN-LSTM模型，以识别和分类潜在的网络威胁。与其他深度学习方法相比，结果显示出在准确率、精确率、召回率和F1分数上的显著改进，检测准确率为99.70%。 

---
# Autoencoder-Based Denoising of Muscle Artifacts in ECG to Preserve Skin Nerve Activity (SKNA) for Cognitive Stress Detection 

**Title (ZH)**: 基于自编码器的ECG肌肉伪迹去噪方法以保留皮肤神经活动（SKNA）用于认知压力检测 

**Authors**: Farnoush Baghestani, Jihye Moon, Youngsun Kong, Ki Chon  

**Link**: [PDF](https://arxiv.org/pdf/2509.07146)  

**Abstract**: The sympathetic nervous system (SNS) plays a central role in regulating the body's responses to stress and maintaining physiological stability. Its dysregulation is associated with a wide range of conditions, from cardiovascular disease to anxiety disorders. Skin nerve activity (SKNA) extracted from high-frequency electrocardiogram (ECG) recordings provides a noninvasive window into SNS dynamics, but its measurement is highly susceptible to electromyographic (EMG) contamination. Traditional preprocessing based on bandpass filtering within a fixed range (e.g., 500--1000 Hz) is susceptible to overlapping EMG and SKNA spectral components, especially during sustained muscle activity. We present a denoising approach using a lightweight one-dimensional convolutional autoencoder with a long short-term memory (LSTM) bottleneck to reconstruct clean SKNA from EMG-contaminated recordings. Using clean ECG-derived SKNA data from cognitive stress experiments and EMG noise from chaotic muscle stimulation recordings, we simulated contamination at realistic noise levels (--4 dB, --8 dB signal-to-noise ratio) and trained the model in the leave-one-subject-out cross-validation framework. The method improved signal-to-noise ratio by up to 9.65 dB, increased cross correlation with clean SKNA from 0.40 to 0.72, and restored burst-based SKNA features to near-clean discriminability (AUROC $\geq$ 0.96). Classification of baseline versus sympathetic stimulation (cognitive stress) conditions reached accuracies of 91--98\% across severe noise levels, comparable to clean data. These results demonstrate that deep learning--based reconstruction can preserve physiologically relevant sympathetic bursts during substantial EMG interference, enabling more robust SKNA monitoring in naturalistic, movement-rich environments. 

**Abstract (ZH)**: 自主神经系统的调节在应对压力和维持生理稳定性中起着中心作用。其功能失调与从心血管疾病到焦虑障碍等多种情况相关。源自高频心电图（ECG）记录的皮肤神经活动（SKNA）提供了一种无创的自主神经系统动态窗口，但其测量极易受到肌电图（EMG）污染的影响。传统的基于固定范围（例如500–1000 Hz）带通滤波的预处理方法容易导致EMG和SKNA频谱成分重叠，尤其是在持续肌肉活动期间。我们提出了一种使用轻量级一维卷积自编码器结合长短期记忆（LSTM）瓶颈的方法来从受EMG污染的记录中重构清洁的SKNA。利用认知压力实验中获得的清洁ECG衍生SKNA数据和来自主神经激发的混沌肌肉刺激记录中的EMG噪声，我们模拟了在实际噪声水平（–4 dB，–8 dB信噪比）下的污染，并在单被试剔除的交叉验证框架中训练了该模型。该方法将信噪比提高了最多9.65 dB，交叉相关系数从0.40提高到0.72，并恢复了突发性SKNA特征的近清洁区分能力（AUROC ≥ 0.96）。在极端噪声水平下，基线与自主神经刺激（认知压力）条件的分类准确率达到91–98%，与清洁数据相当。这些结果表明，基于深度学习的重构可以在大量EMG干扰下保留生理相关的自主神经元突发，从而在自然、富含运动的环境中实现更稳健的SKNA监测。 

---
# Neuro-Symbolic Frameworks: Conceptual Characterization and Empirical Comparative Analysis 

**Title (ZH)**: 神经符号框架：概念特征与实证比较分析 

**Authors**: Sania Sinha, Tanawan Premsri, Danial Kamali, Parisa Kordjamshidi  

**Link**: [PDF](https://arxiv.org/pdf/2509.07122)  

**Abstract**: Neurosymbolic (NeSy) frameworks combine neural representations and learning with symbolic representations and reasoning. Combining the reasoning capacities, explainability, and interpretability of symbolic processing with the flexibility and power of neural computing allows us to solve complex problems with more reliability while being data-efficient. However, this recently growing topic poses a challenge to developers with its learning curve, lack of user-friendly tools, libraries, and unifying frameworks. In this paper, we characterize the technical facets of existing NeSy frameworks, such as the symbolic representation language, integration with neural models, and the underlying algorithms. A majority of the NeSy research focuses on algorithms instead of providing generic frameworks for declarative problem specification to leverage problem solving. To highlight the key aspects of Neurosymbolic modeling, we showcase three generic NeSy frameworks - \textit{DeepProbLog}, \textit{Scallop}, and \textit{DomiKnowS}. We identify the challenges within each facet that lay the foundation for identifying the expressivity of each framework in solving a variety of problems. Building on this foundation, we aim to spark transformative action and encourage the community to rethink this problem in novel ways. 

**Abstract (ZH)**: 神经符号（NeSy）框架结合了神经表示和学习与符号表示和推理。将符号处理的推理能力、可解释性和可解释性与神经计算的灵活性和强大功能相结合，使我们能够更加可靠地解决复杂问题，同时保持数据高效性。然而，这一近期迅速发展的领域对开发者提出了挑战，尤其是在学习曲线、缺少用户友好工具、库和统一框架方面。在本文中，我们分析现有NeSy框架的技术方面，如符号表示语言、与神经模型的集成以及底层算法。大多数NeSy研究关注算法，而不是提供通用框架以通过声明性问题规范来利用解决问题的能力。为了突出神经符号建模的关键方面，我们展示了三个通用的NeSy框架——DeepProbLog、Scallop和DomiKnowS。我们识别出每个方面内的挑战，为进一步识别每个框架在解决各种问题时的表达能力奠定了基础。基于这些基础，我们旨在激发变革性行动，并鼓励社区以新颖的方式重新思考这个问题。 

---
# Instruction Agent: Enhancing Agent with Expert Demonstration 

**Title (ZH)**: 专家演示增强的指令代理 

**Authors**: Yinheng Li, Hailey Hultquist, Justin Wagle, Kazuhito Koishida  

**Link**: [PDF](https://arxiv.org/pdf/2509.07098)  

**Abstract**: Graphical user interface (GUI) agents have advanced rapidly but still struggle with complex tasks involving novel UI elements, long-horizon actions, and personalized trajectories. In this work, we introduce Instruction Agent, a GUI agent that leverages expert demonstrations to solve such tasks, enabling completion of otherwise difficult workflows. Given a single demonstration, the agent extracts step-by-step instructions and executes them by strictly following the trajectory intended by the user, which avoids making mistakes during execution. The agent leverages the verifier and backtracker modules further to improve robustness. Both modules are critical to understand the current outcome from each action and handle unexpected interruptions(such as pop-up windows) during execution. Our experiments show that Instruction Agent achieves a 60% success rate on a set of tasks in OSWorld that all top-ranked agents failed to complete. The Instruction Agent offers a practical and extensible framework, bridging the gap between current GUI agents and reliable real-world GUI task automation. 

**Abstract (ZH)**: 图形用户界面（GUI）代理已取得快速进展，但仍难以应对涉及新型UI元素、长期动作及个性化轨迹的复杂任务。本文介绍了指令代理（Instruction Agent），这是一种利用专家示范来解决此类任务的GUI代理，能够完成原本难以实现的工作流程。给定单个示范，代理提取逐步指令并严格遵循用户的意图执行，从而避免执行过程中的错误。代理还利用校验器和回溯器模块以提高鲁棒性。这两个模块对于理解每项操作的当前结果并处理执行过程中的意外中断（如弹出窗口）至关重要。我们的实验表明，指令代理在OSWorld中实现了60%的成功率，而所有顶级代理都无法完成这些任务。指令代理提供了一个实用且可扩展的框架，填补了当前GUI代理与可靠的真实世界GUI任务自动化之间的差距。 

---
# Statistical Methods in Generative AI 

**Title (ZH)**: 生成式AI中的统计方法 

**Authors**: Edgar Dobriban  

**Link**: [PDF](https://arxiv.org/pdf/2509.07054)  

**Abstract**: Generative Artificial Intelligence is emerging as an important technology, promising to be transformative in many areas. At the same time, generative AI techniques are based on sampling from probabilistic models, and by default, they come with no guarantees about correctness, safety, fairness, or other properties. Statistical methods offer a promising potential approach to improve the reliability of generative AI techniques. In addition, statistical methods are also promising for improving the quality and efficiency of AI evaluation, as well as for designing interventions and experiments in AI.
In this paper, we review some of the existing work on these topics, explaining both the general statistical techniques used, as well as their applications to generative AI. We also discuss limitations and potential future directions. 

**Abstract (ZH)**: 生成式人工智能正 emerge 作为一项重要技术，有望在许多领域产生变革性影响。同时，生成式 AI 技术基于从概率模型中抽样， 默认情况下，它们缺乏关于正确性、安全性、公平性或其他属性的保证。统计方法为提高生成式 AI 技术的可靠性提供了有希望的方法。此外，统计方法也有望提高人工智能评估的质量和效率，并为人工智能的设计干预和实验提供支持。在本文中，我们回顾了这些主题的一些现有工作，解释了所使用的一般统计技术及其在生成式 AI 中的应用，并讨论了限制和潜在的未来方向。 

---
# From Eigenmodes to Proofs: Integrating Graph Spectral Operators with Symbolic Interpretable Reasoning 

**Title (ZH)**: 从特征模态到证明：将图谱运算与符号可解释推理集成 

**Authors**: Andrew Kiruluta, Priscilla Burity  

**Link**: [PDF](https://arxiv.org/pdf/2509.07017)  

**Abstract**: We introduce Spectral NSR, a fully spectral neuro-symbolic reasoning framework that embeds logical rules as spectral templates and performs inference directly in the graph spectral domain. By leveraging graph signal processing (GSP) and frequency-selective filters grounded in the Laplacian eigenstructure of knowledge graphs, the architecture unifies the interpretability of symbolic reasoning with the scalability and adaptability of spectral learning. Beyond the core formulation, we incorporate a comprehensive set of extensions, including dynamic graph and basis learning, rational and diffusion filters for sharper spectral selectivity, mixture-of-spectral-experts for modular specialization, proof-guided training with spectral curricula, and uncertainty quantification for calibrated confidence. Additional enhancements such as large language model coupling, co-spectral transfer alignment, adversarial robustness, efficient GPU kernels, generalized Laplacians, and causal interventions further expand the versatility of the framework.
Empirical evaluation on state-of-the-art reasoning benchmarks such as ProofWriter and CLUTRR demonstrates that Spectral NSR achieves superior accuracy, faster inference, improved robustness to adversarial perturbations, and higher interpretability compared to leading baselines including transformers, message-passing neural networks, and neuro-symbolic logic programming systems. Spectral attribution and proof-band agreement analyses confirm that model decisions align closely with symbolic proof structures, while transfer experiments validate effective domain adaptation through co-spectral alignment. These results establish Spectral NSR as a scalable and principled foundation for the next generation of reasoning systems, offering transparency, robustness, and generalization beyond conventional approaches. 

**Abstract (ZH)**: Spectral NSR：一种基于光谱的神经符号推理框架 

---
# Renewable Energy Sources Selection Analysis with the Maximizing Deviation Method 

**Title (ZH)**: 基于最大化偏差方法的可再生能源选择分析 

**Authors**: Kirisci Murat  

**Link**: [PDF](https://arxiv.org/pdf/2509.07011)  

**Abstract**: Multi-criteria decision-making methods provide decision-makers with appropriate tools to make better decisions in uncertain, complex, and conflicting situations. Fuzzy set theory primarily deals with the uncertainty inherent in human thoughts and perceptions and attempts to quantify this uncertainty. Fuzzy logic and fuzzy set theory are utilized with multi-criteria decision-making methods because they effectively handle uncertainty and fuzziness in decision-makers' judgments, allowing for verbal judgments of the problem. This study utilizes the Fermatean fuzzy environment, a generalization of fuzzy sets. An optimization model based on the deviation maximization method is proposed to determine partially known feature weights. This method is combined with interval-valued Fermatean fuzzy sets. The proposed method was applied to the problem of selecting renewable energy sources. The reason for choosing renewable energy sources is that meeting energy needs from renewable sources, balancing carbon emissions, and mitigating the effects of global climate change are among the most critical issues of the recent period. Even though selecting renewable energy sources is a technical issue, the managerial and political implications of this issue are also important, and are discussed in this study. 

**Abstract (ZH)**: 多准则决策方法为决策者在不确定、复杂和冲突的情况下提供了合适的工具。模糊集理论主要处理人类思想和感知中固有的不确定性，并试图量化这种不确定性。由于模糊逻辑和模糊集理论能够有效处理决策者判断中的不确定性和模糊性，并允许对问题进行语义判断，因此它们与多准则决策方法结合使用。本文利用Fermatean模糊环境，这是模糊集的一种推广。提出了一种基于偏差最大化方法的优化模型来确定部分已知特征权重。该方法结合了区间值Fermatean模糊集，所提出的方法应用于可再生能源选择问题。选择可再生能源的原因在于，满足可再生能源的需求、平衡碳排放以及减轻全球气候变暖的影响是当今最关键的问题之一。尽管选择可再生能源是一个技术问题，但这一问题的管理与政治影响也很重要，并在本文中进行了讨论。 

---
# ACE and Diverse Generalization via Selective Disagreement 

**Title (ZH)**: ACE和通过选择性分歧实现的多样泛化 

**Authors**: Oliver Daniels, Stuart Armstrong, Alexandre Maranhão, Mahirah Fairuz Rahman, Benjamin M. Marlin, Rebecca Gorman  

**Link**: [PDF](https://arxiv.org/pdf/2509.07955)  

**Abstract**: Deep neural networks are notoriously sensitive to spurious correlations - where a model learns a shortcut that fails out-of-distribution. Existing work on spurious correlations has often focused on incomplete correlations,leveraging access to labeled instances that break the correlation. But in cases where the spurious correlations are complete, the correct generalization is fundamentally \textit{underspecified}. To resolve this underspecification, we propose learning a set of concepts that are consistent with training data but make distinct predictions on a subset of novel unlabeled inputs. Using a self-training approach that encourages \textit{confident} and \textit{selective} disagreement, our method ACE matches or outperforms existing methods on a suite of complete-spurious correlation benchmarks, while remaining robust to incomplete spurious correlations. ACE is also more configurable than prior approaches, allowing for straight-forward encoding of prior knowledge and principled unsupervised model selection. In an early application to language-model alignment, we find that ACE achieves competitive performance on the measurement tampering detection benchmark \textit{without} access to untrusted measurements. While still subject to important limitations, ACE represents significant progress towards overcoming underspecification. 

**Abstract (ZH)**: 深度神经网络对虚假相关性特别敏感——模型在此类相关性失效时学到的捷径。现有的虚假相关性研究往往集中在不完整的相关性上，利用标记的实例来打破相关性。但在虚假相关性完整的情况下，正确的泛化本质上是不充分指定的。为了解决这个不充分指定的问题，我们提出学习一组与训练数据一致但对部分新型未标记输入做出不同预测的概念。通过鼓励信心十足且有选择地产生分歧的自我训练方法，我们的方法ACE在一系列完整的虚假相关性基准测试中表现出色或优于现有方法，同时对不完整的虚假相关性保持鲁棒性。ACE比先前的方法更具可配置性，允许简单地编码先验知识并进行稳健的无监督模型选择。在语言模型对齐的早期应用中，我们发现ACE能够在无需访问不可信测量值的情况下，在测量篡改检测基准测试中取得竞争性的性能。虽然仍然存在重要的限制，但ACE代表了克服不充分指定的重要进展。 

---
# Bringing Multi-Modal Multi-Task Federated Foundation Models to Education Domain: Prospects and Challenges 

**Title (ZH)**: 将多模态多任务联邦基础模型引入教育领域：前景与挑战 

**Authors**: Kasra Borazjani, Naji Khosravan, Rajeev Sahay, Bita Akram, Seyyedali Hosseinalipour  

**Link**: [PDF](https://arxiv.org/pdf/2509.07946)  

**Abstract**: Multi-modal multi-task (M3T) foundation models (FMs) have recently shown transformative potential in artificial intelligence, with emerging applications in education. However, their deployment in real-world educational settings is hindered by privacy regulations, data silos, and limited domain-specific data availability. We introduce M3T Federated Foundation Models (FedFMs) for education: a paradigm that integrates federated learning (FL) with M3T FMs to enable collaborative, privacy-preserving training across decentralized institutions while accommodating diverse modalities and tasks. Subsequently, this position paper aims to unveil M3T FedFMs as a promising yet underexplored approach to the education community, explore its potentials, and reveal its related future research directions. We outline how M3T FedFMs can advance three critical pillars of next-generation intelligent education systems: (i) privacy preservation, by keeping sensitive multi-modal student and institutional data local; (ii) personalization, through modular architectures enabling tailored models for students, instructors, and institutions; and (iii) equity and inclusivity, by facilitating participation from underrepresented and resource-constrained entities. We finally identify various open research challenges, including studying of (i) inter-institution heterogeneous privacy regulations, (ii) the non-uniformity of data modalities' characteristics, (iii) the unlearning approaches for M3T FedFMs, (iv) the continual learning frameworks for M3T FedFMs, and (v) M3T FedFM model interpretability, which must be collectively addressed for practical deployment. 

**Abstract (ZH)**: 教育领域的多模态多任务联邦基础模型（M3T FedFMs） 

---
# Active Membership Inference Test (aMINT): Enhancing Model Auditability with Multi-Task Learning 

**Title (ZH)**: 活性成员推断测试 (aMINT): 通过多任务学习增强模型可审核性 

**Authors**: Daniel DeAlcala, Aythami Morales, Julian Fierrez, Gonzalo Mancera, Ruben Tolosana, Javier Ortega-Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2509.07879)  

**Abstract**: Active Membership Inference Test (aMINT) is a method designed to detect whether given data were used during the training of machine learning models. In Active MINT, we propose a novel multitask learning process that involves training simultaneously two models: the original or Audited Model, and a secondary model, referred to as the MINT Model, responsible for identifying the data used for training the Audited Model. This novel multi-task learning approach has been designed to incorporate the auditability of the model as an optimization objective during the training process of neural networks. The proposed approach incorporates intermediate activation maps as inputs to the MINT layers, which are trained to enhance the detection of training data. We present results using a wide range of neural networks, from lighter architectures such as MobileNet to more complex ones such as Vision Transformers, evaluated in 5 public benchmarks. Our proposed Active MINT achieves over 80% accuracy in detecting if given data was used for training, significantly outperforming previous approaches in the literature. Our aMINT and related methodological developments contribute to increasing transparency in AI models, facilitating stronger safeguards in AI deployments to achieve proper security, privacy, and copyright protection. 

**Abstract (ZH)**: 活性成员推理测试 (aMINT) 是一种用于检测给定数据是否在机器学习模型训练中使用的方法。在活性MINT中，我们提出了一种新颖的多任务学习过程，涉及同时训练两个模型：原始或审核模型，以及一个称为MINT模型的次要模型，负责识别用于训练审核模型的数据。该新颖的多任务学习方法在神经网络训练过程中设计为将模型的可审核性作为优化目标进行整合。提出的approach将中间激活图作为MINT层的输入，这些层经过训练以增强对训练数据的检测。我们展示了使用从较轻的架构如MobileNet到更复杂的如视觉变换器的各种神经网络的结果，这些网络在5个公开基准上进行了评估。我们提出的活性MINT在检测给定数据是否用于训练时准确率超过80%，显著优于文献中的先前方法。我们的活性MINT及相关方法的发展为增加AI模型的透明度作出了贡献，有助于在AI部署中增强安全、隐私和版权保护的保障措施。 

---
# Deep Learning-Based Burned Area Mapping Using Bi-Temporal Siamese Networks and AlphaEarth Foundation Datasets 

**Title (ZH)**: 基于双时相Siamese网络的深度学习烧伤区制图——AlphaEarth基础数据集应用 

**Authors**: Seyd Teymoor Seydi  

**Link**: [PDF](https://arxiv.org/pdf/2509.07852)  

**Abstract**: Accurate and timely mapping of burned areas is crucial for environmental monitoring, disaster management, and assessment of climate change. This study presents a novel approach to automated burned area mapping using the AlphaEArth dataset combined with the Siamese U-Net deep learning architecture. The AlphaEArth Dataset, comprising high-resolution optical and thermal infrared imagery with comprehensive ground-truth annotations, provides an unprecedented resource for training robust burned area detection models. We trained our model with the Monitoring Trends in Burn Severity (MTBS) dataset in the contiguous US and evaluated it with 17 regions cross in Europe. Our experimental results demonstrate that the proposed ensemble approach achieves superior performance with an overall accuracy of 95%, IoU of 0.6, and F1-score of 74% on the test dataset. The model successfully identifies burned areas across diverse ecosystems with complex background, showing particular strength in detecting partially burned vegetation and fire boundaries and its transferability and high generalization in burned area mapping. This research contributes to the advancement of automated fire damage assessment and provides a scalable solution for global burn area monitoring using the AlphaEarth dataset. 

**Abstract (ZH)**: 准确及时的火烧区域 mapping 对环境监测、灾害管理及气候变化评估至关重要。本研究提出了一种新的自动火烧区域 mapping 方法，利用 AlphaEArth 数据集结合 Siamese U-Net 深度学习架构。AlphaEArth 数据集包含高分辨率的光学和热红外影像，并伴有全面的地表真实标注，提供了训练 robust 火烧区域检测模型的前所未有的资源。我们使用 Monitoring Trends in Burn Severity (MTBS) 数据集在北美进行了模型训练，并在欧洲 17 个地区进行了评估。实验结果表明，提出的集成方法在测试数据集上实现了总体准确率 95%、IoU 0.6 和 F1 分数 74%，成功识别了复杂背景下的多种生态系统中的火烧区域，特别擅长检测部分火烧植被和火边界，并展示了在火烧区域 mapping 中的可转移性和高泛化能力。本研究促进了自动化火灾损害评估，并提供了使用 AlphaEarth 数据集进行全球火烧区域监测的可扩展解决方案。 

---
# Forecasting Russian Equipment Losses Using Time Series and Deep Learning Models 

**Title (ZH)**: 使用时间序列和深度学习模型预测俄罗斯装备损失 

**Authors**: Jonathan Teagan  

**Link**: [PDF](https://arxiv.org/pdf/2509.07813)  

**Abstract**: This study applies a range of forecasting techniques,including ARIMA, Prophet, Long Short Term Memory networks (LSTM), Temporal Convolutional Networks (TCN), and XGBoost, to model and predict Russian equipment losses during the ongoing war in Ukraine. Drawing on daily and monthly open-source intelligence (OSINT) data from WarSpotting, we aim to assess trends in attrition, evaluate model performance, and estimate future loss patterns through the end of 2025. Our findings show that deep learning models, particularly TCN and LSTM, produce stable and consistent forecasts, especially under conditions of high temporal granularity. By comparing different model architectures and input structures, this study highlights the importance of ensemble forecasting in conflict modeling, and the value of publicly available OSINT data in quantifying material degradation over time. 

**Abstract (ZH)**: 本研究应用包括ARIMA、Prophet、长短期记忆网络（LSTM）、时序卷积网络（TCN）和XGBoost在内的多种预测技术，建模并预测俄罗斯在乌克兰战争中装备损失的情况。通过利用来自WarSpotting的每日和月度开源情报（OSINT）数据，我们旨在评估装备损耗趋势、评估模型性能，并估计到2025年底的未来损失模式。研究发现，深度学习模型，尤其是TCN和LSTM，能够产生稳定且一致的预测，尤其是在高时间粒度条件下。通过比较不同的模型架构和输入结构，本研究强调了冲突建模中集成预测的重要性，并突显了公开可用的OSINT数据在量化随时间变化的物质损耗方面的价值。 

---
# Individual utilities of life satisfaction reveal inequality aversion unrelated to political alignment 

**Title (ZH)**: 个体的生活满意度揭示了与政治倾向无关的不平等厌恶感 

**Authors**: Crispin Cooper, Ana Friedrich, Tommaso Reggiani, Wouter Poortinga  

**Link**: [PDF](https://arxiv.org/pdf/2509.07793)  

**Abstract**: How should well-being be prioritised in society, and what trade-offs are people willing to make between fairness and personal well-being? We investigate these questions using a stated preference experiment with a nationally representative UK sample (n = 300), in which participants evaluated life satisfaction outcomes for both themselves and others under conditions of uncertainty. Individual-level utility functions were estimated using an Expected Utility Maximisation (EUM) framework and tested for sensitivity to the overweighting of small probabilities, as characterised by Cumulative Prospect Theory (CPT). A majority of participants displayed concave (risk-averse) utility curves and showed stronger aversion to inequality in societal life satisfaction outcomes than to personal risk. These preferences were unrelated to political alignment, suggesting a shared normative stance on fairness in well-being that cuts across ideological boundaries. The results challenge use of average life satisfaction as a policy metric, and support the development of nonlinear utility-based alternatives that more accurately reflect collective human values. Implications for public policy, well-being measurement, and the design of value-aligned AI systems are discussed. 

**Abstract (ZH)**: 如何在社会中优先考虑幸福，人们在公平与个人幸福之间愿意做出什么样的权衡？我们通过一项具有全国代表性的英国样本（n=300）的表达偏好实验来探讨这些问题，在不确定性条件下，参与者评估了自己和他人的生活满意度结果。利用期望效用最大化（EUM）框架估计了个体内在效用函数，并测试了对小概率 overweighting 的敏感性，这由前景理论（CPT）来表征。大多数参与者显示出凹形（风险规避）的效用曲线，并对社会生活满意度结果的不平等表现出比个人风险更大的厌恶。这些偏好与政治倾向无关，表明在不同意识形态边界上存在对福利公平性的共享规范性立场。研究结果挑战了平均生活满意度作为政策指标的使用，并支持开发能够更准确反映集体人类价值观的非线性效用替代指标。讨论了这些结果对公共政策、幸福测度以及价值对齐的人工智能系统设计的含义。 

---
# XSRD-Net: EXplainable Stroke Relapse Detection 

**Title (ZH)**: XSRD-Net: 可解释的笔画复发检测 

**Authors**: Christian Gapp, Elias Tappeiner, Martin Welk, Karl Fritscher, Stephanie Mangesius, Constantin Eisenschink, Philipp Deisl, Michael Knoflach, Astrid E. Grams, Elke R. Gizewski, Rainer Schubert  

**Link**: [PDF](https://arxiv.org/pdf/2509.07772)  

**Abstract**: Stroke is the second most frequent cause of death world wide with an annual mortality of around 5.5 million. Recurrence rates of stroke are between 5 and 25% in the first year. As mortality rates for relapses are extraordinarily high (40%) it is of utmost importance to reduce the recurrence rates. We address this issue by detecting patients at risk of stroke recurrence at an early stage in order to enable appropriate therapy planning. To this end we collected 3D intracranial CTA image data and recorded concomitant heart diseases, the age and the gender of stroke patients between 2010 and 2024. We trained single- and multimodal deep learning based neural networks for binary relapse detection (Task 1) and for relapse free survival (RFS) time prediction together with a subsequent classification (Task 2). The separation of relapse from non-relapse patients (Task 1) could be solved with tabular data (AUC on test dataset: 0.84). However, for the main task, the regression (Task 2), our multimodal XSRD-net processed the modalities vision:tabular with 0.68:0.32 according to modality contribution measures. The c-index with respect to relapses for the multimodal model reached 0.68, and the AUC is 0.71 for the test dataset. Final, deeper interpretability analysis results could highlight a link between both heart diseases (tabular) and carotid arteries (vision) for the detection of relapses and the prediction of the RFS time. This is a central outcome that we strive to strengthen with ongoing data collection and model retraining. 

**Abstract (ZH)**: 全球范围内，中风是仅次于心脏病的第二大死亡原因，每年约导致550万人死亡。中风复发率在首年介于5%至25%之间。由于复发的死亡率极高（40%），降低复发率尤为重要。我们通过早期检测中风复发风险的患者，以提供合适的治疗规划来应对这一问题。2010年至2024年间，我们收集了颅内3D CTA影像数据，并记录了中风患者的并发心脏病情况、年龄和性别。我们训练了单模态和多模态深度学习神经网络，用于二元复发检测（任务1）和复发自由生存时间预测（任务2）及后续分类（任务2）。单一模态数据在区分复发与非复发患者（任务1）中表现良好（测试数据集AUC：0.84）。然而，对于主要任务——回归预测（任务2），我们多模态XSRD-net采用 vision:tabular 模态权重比0.68:0.32进行了处理。针对复发的多模态模型的c-index达到0.68，测试数据集的AUC为0.71。最终，更深入的可解释性分析结果表明，心脏疾病（表观数据）和颈动脉（影像数据）之间的联系对于复发检测和生存时间预测具有重要意义。这是我们持续数据收集和模型重新训练努力加强的核心成果。 

---
# Enhancing Online Learning by Integrating Biosensors and Multimodal Learning Analytics for Detecting and Predicting Student Behavior: A Review 

**Title (ZH)**: 通过整合 biosensors 和多模态学习分析来增强在线学习：检测和预测学生行为的综述 

**Authors**: Alvaro Becerra, Ruth Cobos, Charles Lang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07742)  

**Abstract**: In modern online learning, understanding and predicting student behavior is crucial for enhancing engagement and optimizing educational outcomes. This systematic review explores the integration of biosensors and Multimodal Learning Analytics (MmLA) to analyze and predict student behavior during computer-based learning sessions. We examine key challenges, including emotion and attention detection, behavioral analysis, experimental design, and demographic considerations in data collection. Our study highlights the growing role of physiological signals, such as heart rate, brain activity, and eye-tracking, combined with traditional interaction data and self-reports to gain deeper insights into cognitive states and engagement levels. We synthesize findings from 54 key studies, analyzing commonly used methodologies such as advanced machine learning algorithms and multimodal data pre-processing techniques. The review identifies current research trends, limitations, and emerging directions in the field, emphasizing the transformative potential of biosensor-driven adaptive learning systems. Our findings suggest that integrating multimodal data can facilitate personalized learning experiences, real-time feedback, and intelligent educational interventions, ultimately advancing toward a more customized and adaptive online learning experience. 

**Abstract (ZH)**: 在现代在线学习中，理解并预测学生行为对于增强参与度和优化教育成果至关重要。本系统性综述探讨了生物传感器与多模态学习分析（MmLA）的集成，以分析和预测计算机辅助学习会话期间的学生行为。我们研究了包括情绪和注意力检测、行为分析、实验设计和数据收集中的人口统计学考虑在内的关键挑战。我们的研究突出了生理信号（如心率、脑活动和眼动追踪）与传统交互数据和自我报告相结合，以更深入地了解认知状态和参与水平的作用日益增加。我们综合了54项关键研究的发现，分析了常用的方法论，如先进的机器学习算法和多模态数据预处理技术。综述识别了该领域的当前研究趋势、局限性和新兴方向，强调了生物传感器驱动的自适应学习系统的变革潜力。我们的研究结果表明，整合多模态数据可以促进个性化学习体验、实时反馈和智能教育干预，最终朝着更定制化和自适应的在线学习体验迈进。 

---
# Spectral Masking and Interpolation Attack (SMIA): A Black-box Adversarial Attack against Voice Authentication and Anti-Spoofing Systems 

**Title (ZH)**: 光谱掩蔽和插值攻击（SMIA）：针对语音认证和防欺骗系统的黑盒 adversarial 攻击 

**Authors**: Kamel Kamel, Hridoy Sankar Dutta, Keshav Sood, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2509.07677)  

**Abstract**: Voice Authentication Systems (VAS) use unique vocal characteristics for verification. They are increasingly integrated into high-security sectors such as banking and healthcare. Despite their improvements using deep learning, they face severe vulnerabilities from sophisticated threats like deepfakes and adversarial attacks. The emergence of realistic voice cloning complicates detection, as systems struggle to distinguish authentic from synthetic audio. While anti-spoofing countermeasures (CMs) exist to mitigate these risks, many rely on static detection models that can be bypassed by novel adversarial methods, leaving a critical security gap. To demonstrate this vulnerability, we propose the Spectral Masking and Interpolation Attack (SMIA), a novel method that strategically manipulates inaudible frequency regions of AI-generated audio. By altering the voice in imperceptible zones to the human ear, SMIA creates adversarial samples that sound authentic while deceiving CMs. We conducted a comprehensive evaluation of our attack against state-of-the-art (SOTA) models across multiple tasks, under simulated real-world conditions. SMIA achieved a strong attack success rate (ASR) of at least 82% against combined VAS/CM systems, at least 97.5% against standalone speaker verification systems, and 100% against countermeasures. These findings conclusively demonstrate that current security postures are insufficient against adaptive adversarial attacks. This work highlights the urgent need for a paradigm shift toward next-generation defenses that employ dynamic, context-aware frameworks capable of evolving with the threat landscape. 

**Abstract (ZH)**: 语音认证系统（VAS）利用独特的语音特征进行验证。它们正越来越多地被集成到银行和医疗等高安全领域。尽管使用深度学习技术进行了改进，但它们仍面临来自深度伪造和对抗攻击等复杂威胁的严重漏洞。现实的语音克隆技术的出现加剧了检测难度，因为系统难以区分真实的和合成的音频。虽然存在反欺骗对策（CMs）来缓解这些风险，但许多对策依赖于静态检测模型，这些模型容易被新型对抗方法绕过，留下了安全漏洞。为了证明这一漏洞，我们提出了一种名为Spectral Masking and Interpolation Attack（Spectral掩蔽和内插攻击，SMIA）的新方法，该方法战略性地操纵AI生成音频的不可听频率区域。通过改变人类耳朵感知不到的区域中的声音，SMIA生成了既能听起来真实又能欺骗CMs的对抗样本。我们在多种任务下对最先进的模型进行了全面评估，模拟了现实生活中的条件。结果显示，SMIA分别在综合VAS/CM系统、独立说话人验证系统以及对策上的攻击成功率（ASR）至少为82%、97.5%和100%。这些发现表明，现有的安全措施不足以抵御适应性对抗攻击。本文强调了迫切需要转向采用动态、情境感知框架的下一代防御技术，这些框架能够随着威胁环境的变化而发展。 

---
# Variational Quantum Circuits in Offline Contextual Bandit Problems 

**Title (ZH)**: 离线上下文臂问题中的变分量子电路 

**Authors**: Lukas Schulte, Daniel Hein, Steffen Udluft, Thomas A. Runkler  

**Link**: [PDF](https://arxiv.org/pdf/2509.07633)  

**Abstract**: This paper explores the application of variational quantum circuits (VQCs) for solving offline contextual bandit problems in industrial optimization tasks. Using the Industrial Benchmark (IB) environment, we evaluate the performance of quantum regression models against classical models. Our findings demonstrate that quantum models can effectively fit complex reward functions, identify optimal configurations via particle swarm optimization (PSO), and generalize well in noisy and sparse datasets. These results provide a proof of concept for utilizing VQCs in offline contextual bandit problems and highlight their potential in industrial optimization tasks. 

**Abstract (ZH)**: 本文探讨了变分量子电路（VQCs）在工业优化任务中解决离线上下文臂问题的应用。通过工业基准（IB）环境，我们将量子回归模型的性能与经典模型进行了评估。研究结果表明，量子模型可以有效地拟合复杂奖励函数，通过粒子群优化（PSO）识别最优配置，并在噪声大和数据稀疏的情况下表现出良好的泛化能力。这些结果为在离线上下文臂问题中利用VQCs提供了概念验证，并突显了其在工业优化任务中的潜在应用。 

---
# From Classical Data to Quantum Advantage -- Quantum Policy Evaluation on Quantum Hardware 

**Title (ZH)**: 从经典数据到量子优势——量子硬件上的量子策略评估 

**Authors**: Daniel Hein, Simon Wiedemann, Markus Baumann, Patrik Felbinger, Justin Klein, Maximilian Schieder, Jonas Stein, Daniëlle Schuman, Thomas Cope, Steffen Udluft  

**Link**: [PDF](https://arxiv.org/pdf/2509.07614)  

**Abstract**: Quantum policy evaluation (QPE) is a reinforcement learning (RL) algorithm which is quadratically more efficient than an analogous classical Monte Carlo estimation. It makes use of a direct quantum mechanical realization of a finite Markov decision process, in which the agent and the environment are modeled by unitary operators and exchange states, actions, and rewards in superposition. Previously, the quantum environment has been implemented and parametrized manually for an illustrative benchmark using a quantum simulator. In this paper, we demonstrate how these environment parameters can be learned from a batch of classical observational data through quantum machine learning (QML) on quantum hardware. The learned quantum environment is then applied in QPE to also compute policy evaluations on quantum hardware. Our experiments reveal that, despite challenges such as noise and short coherence times, the integration of QML and QPE shows promising potential for achieving quantum advantage in RL. 

**Abstract (ZH)**: 量子策略评估（QPE）是一种比经典蒙特卡洛估计算法高效四倍的强化学习（RL）算法。它利用了直接的量子力学实现有穷马尔可夫决策过程，其中代理和环境由酉算子表示，并在量子相干状态下交换状态、动作和奖励。之前，量子环境已经在量子模拟器中通过手动实现和参数化来进行说明性基准测试。在本文中，我们展示了如何通过量子机器学习（QML）在量子硬件上从一批经典观测数据中学习这些环境参数。然后，所学的量子环境被应用到QPE中，在量子硬件上进行策略评估。我们的实验表明，尽管存在噪声和短相干时间等挑战，QML和QPE的集成在RL中实现量子优势具有前景。 

---
# Beyond Rebalancing: Benchmarking Binary Classifiers Under Class Imbalance Without Rebalancing Techniques 

**Title (ZH)**: 超越重新加权：在无需使用重新加权技术的情况下评估二分类器在类别不平衡条件下的表现 

**Authors**: Ali Nawaz, Amir Ahmad, Shehroz S. Khan  

**Link**: [PDF](https://arxiv.org/pdf/2509.07605)  

**Abstract**: Class imbalance poses a significant challenge to supervised classification, particularly in critical domains like medical diagnostics and anomaly detection where minority class instances are rare. While numerous studies have explored rebalancing techniques to address this issue, less attention has been given to evaluating the performance of binary classifiers under imbalance when no such techniques are applied. Therefore, the goal of this study is to assess the performance of binary classifiers "as-is", without performing any explicit rebalancing. Specifically, we systematically evaluate the robustness of a diverse set of binary classifiers across both real-world and synthetic datasets, under progressively reduced minority class sizes, using one-shot and few-shot scenarios as baselines. Our approach also explores varying data complexities through synthetic decision boundary generation to simulate real-world conditions. In addition to standard classifiers, we include experiments using undersampling, oversampling strategies, and one-class classification (OCC) methods to examine their behavior under severe imbalance. The results confirm that classification becomes more difficult as data complexity increases and the minority class size decreases. While traditional classifiers deteriorate under extreme imbalance, advanced models like TabPFN and boosting-based ensembles retain relatively higher performance and better generalization compared to traditional classifiers. Visual interpretability and evaluation metrics further validate these findings. Our work offers valuable guidance on model selection for imbalanced learning, providing insights into classifier robustness without dependence on explicit rebalancing techniques. 

**Abstract (ZH)**: 不均衡样本对监督分类构成重大挑战，尤其在医学诊断和异常检测等关键领域，少数类样本稀少。虽然许多研究探索了再平衡技术来解决这一问题，但在未应用这些技术的情况下评估二分类器性能的关注较少。因此，本研究的目的是评估二分类器“原封不动”的性能，不进行任何显式再平衡。具体而言，我们系统评估一系列二分类器在真实世界和合成数据集中的鲁棒性，少数类样本大小逐步减少，使用一对一和少量样本场景作为基准。我们的方法还通过生成合成决策边界来探索数据复杂性差异，以模拟现实世界条件。除了标准分类器，我们还包括欠采样、过采样策略以及一类分类(OCC)方法的实验，以考察其在严重不均衡情况下的行为。结果证实，随着数据复杂性的增加和少数类样本数量的减少，分类任务变得更为困难。虽然传统分类器在极端不均衡下性能下降，但如TabPFN等高级模型和基于提升的集成模型相比传统分类器具有相对较高的性能和更好的泛化能力。可视化解释和评估指标进一步验证了这些发现。我们的工作为不平衡学习提供有价值的指导，揭示了在不依赖显式再平衡技术的情况下分类器的鲁棒性。 

---
# FLeW: Facet-Level and Adaptive Weighted Representation Learning of Scientific Documents 

**Title (ZH)**: FLeW：科学文档的层次化和自适应加权表示学习 

**Authors**: Zheng Dou, Deqing Wang, Fuzhen Zhuang, Jian Ren, Yanlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07531)  

**Abstract**: Scientific document representation learning provides powerful embeddings for various tasks, while current methods face challenges across three approaches. 1) Contrastive training with citation-structural signals underutilizes citation information and still generates single-vector representations. 2) Fine-grained representation learning, which generates multiple vectors at the sentence or aspect level, requires costly integration and lacks domain generalization. 3) Task-aware learning depends on manually predefined task categorization, overlooking nuanced task distinctions and requiring extra training data for task-specific modules. To address these problems, we propose a new method that unifies the three approaches for better representations, namely FLeW. Specifically, we introduce a novel triplet sampling method that leverages citation intent and frequency to enhance citation-structural signals for training. Citation intents (background, method, result), aligned with the general structure of scientific writing, facilitate a domain-generalized facet partition for fine-grained representation learning. Then, we adopt a simple weight search to adaptively integrate three facet-level embeddings into a task-specific document embedding without task-aware fine-tuning. Experiments show the applicability and robustness of FLeW across multiple scientific tasks and fields, compared to prior models. 

**Abstract (ZH)**: 科学文献表示学习提供了多种任务的强大嵌入，而当前方法在三种方法上面临挑战。1）基于引用结构信号的对比训练未能充分利用引用信息，仍生成单向量表示。2）细粒度表示学习生成句子或方面级别的多个向量，需要昂贵的整合并缺乏领域泛化能力。3）任务感知学习依赖于人工预定义的任务分类，忽略了任务的细微差异，并要求为任务特定模块额外的数据训练。为解决这些问题，我们提出了一种新的方法，将这三种方法统一起来以获得更好的表示，名为FLeW。具体地，我们引入一种新颖的三元组采样方法，利用引用意图和频率来增强引用结构信号以供训练。引用意图（背景、方法、结果），与科学写作的一般结构对齐，有助于实现细粒度表示学习的领域泛化方面分隔。然后，我们采用简单的权重搜索来适应性地将三个方面级别的嵌入整合为一个任务特定的文档嵌入，无需任务感知微调。实验表明，与先前模型相比，FLeW在多种科学研究任务和领域中具有适用性和鲁棒性。 

---
# Water Demand Forecasting of District Metered Areas through Learned Consumer Representations 

**Title (ZH)**: 通过学习消费者表示进行区域计量区的用水需求预测 

**Authors**: Adithya Ramachandran, Thorkil Flensmark B. Neergaard, Tomás Arias-Vergara, Andreas Maier, Siming Bayer  

**Link**: [PDF](https://arxiv.org/pdf/2509.07515)  

**Abstract**: Advancements in smart metering technologies have significantly improved the ability to monitor and manage water utilities. In the context of increasing uncertainty due to climate change, securing water resources and supply has emerged as an urgent global issue with extensive socioeconomic ramifications. Hourly consumption data from end-users have yielded substantial insights for projecting demand across regions characterized by diverse consumption patterns. Nevertheless, the prediction of water demand remains challenging due to influencing non-deterministic factors, such as meteorological conditions. This work introduces a novel method for short-term water demand forecasting for District Metered Areas (DMAs) which encompass commercial, agricultural, and residential consumers. Unsupervised contrastive learning is applied to categorize end-users according to distinct consumption behaviors present within a DMA. Subsequently, the distinct consumption behaviors are utilized as features in the ensuing demand forecasting task using wavelet-transformed convolutional networks that incorporate a cross-attention mechanism combining both historical data and the derived representations. The proposed approach is evaluated on real-world DMAs over a six-month period, demonstrating improved forecasting performance in terms of MAPE across different DMAs, with a maximum improvement of 4.9%. Additionally, it identifies consumers whose behavior is shaped by socioeconomic factors, enhancing prior knowledge about the deterministic patterns that influence demand. 

**Abstract (ZH)**: 智能计量技术的进步显著提高了对水務進行監測和管理的能力。隨著經濟社會影響的擴大，由於氣候變化的不確定性增加，確保水资源和供應已成为一个紧迫的全球问题。来自终端用户的每小时消耗数据为具有不同消耗模式的地区的需求展望提供了大量见解。然而，由于气象条件等非确定性因素的影响，需求预测仍然具有挑战性。本研究介绍了一种新的方法，用于预测地区计量区域（DMAs）的短期内水需求，这些DMAs包括商业、农业和居民用户。应用无监督对比学习对DMAs中的不同消费行为进行分类。随后，这些不同的消费行为被用作特征，使用结合历史数据和提取表示的交叉注意力机制的小波变换卷积网络进行后续需求预测任务。该提出的 approaches 在六个月内的真实 DMAs 上进行了评估，显示了在不同 DMAs 上改善的预测性能，最大改进率为 4.9%。此外，该方法还识别了受社会经济因素影响的消费者行为，增强了对影响需求的确定性模式的先验知识。 

---
# HALT-RAG: A Task-Adaptable Framework for Hallucination Detection with Calibrated NLI Ensembles and Abstention 

**Title (ZH)**: HALT-RAG：一种基于校准NLI集成和弃权的任务自适应幻觉检测框架 

**Authors**: Saumya Goswami, Siddharth Kurra  

**Link**: [PDF](https://arxiv.org/pdf/2509.07475)  

**Abstract**: Detecting content that contradicts or is unsupported by a given source text is a critical challenge for the safe deployment of generative language models. We introduce HALT-RAG, a post-hoc verification system designed to identify hallucinations in the outputs of Retrieval-Augmented Generation (RAG) pipelines. Our flexible and task-adaptable framework uses a universal feature set derived from an ensemble of two frozen, off-the-shelf Natural Language Inference (NLI) models and lightweight lexical signals. These features are used to train a simple, calibrated, and task-adapted meta-classifier. Using a rigorous 5-fold out-of-fold (OOF) training protocol to prevent data leakage and produce unbiased estimates, we evaluate our system on the HaluEval benchmark. By pairing our universal feature set with a lightweight, task-adapted classifier and a precision-constrained decision policy, HALT-RAG achieves strong OOF F1-scores of 0.7756, 0.9786, and 0.7391 on the summarization, QA, and dialogue tasks, respectively. The system's well-calibrated probabilities enable a practical abstention mechanism, providing a reliable tool for balancing model performance with safety requirements. 

**Abstract (ZH)**: 检测与给定来源文本矛盾或未被支持的内容是生成语言模型安全部署的关键挑战。我们引入HALT-RAG，一个后验证系统，旨在识别检索增强生成（RAG）管道输出中的幻觉。我们的灵活且任务适应的框架使用从两个冻结的现成自然语言推理（NLI）模型和轻量级词汇信号中衍生的通用特征集进行训练。这些特征用于训练一个简单、校准且任务适应的元分类器。通过使用严格的5折交叉验证训练协议来防止数据泄露并产生无偏估计，我们在HaluEval基准上评估了该系统。通过将通用特征集与轻量级任务适应分类器和精度受限的决策策略配对，HALT-RAG分别在摘要、问答和对话任务上实现了0.7756、0.9786和0.7391的交叉验证F1分数。系统的校准概率使其能够实现实际的回避机制，为平衡模型性能与安全要求提供可靠的工具。 

---
# Bias-Aware Machine Unlearning: Towards Fairer Vision Models via Controllable Forgetting 

**Title (ZH)**: 带有偏差意识的机器遗忘：通过可控遗忘实现更公平的视觉模型 

**Authors**: Sai Siddhartha Chary Aylapuram, Veeraraju Elluru, Shivang Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2509.07456)  

**Abstract**: Deep neural networks often rely on spurious correlations in training data, leading to biased or unfair predictions in safety-critical domains such as medicine and autonomous driving. While conventional bias mitigation typically requires retraining from scratch or redesigning data pipelines, recent advances in machine unlearning provide a promising alternative for post-hoc model correction. In this work, we investigate \textit{Bias-Aware Machine Unlearning}, a paradigm that selectively removes biased samples or feature representations to mitigate diverse forms of bias in vision models. Building on privacy-preserving unlearning techniques, we evaluate various strategies including Gradient Ascent, LoRA, and Teacher-Student distillation. Through empirical analysis on three benchmark datasets, CUB-200-2011 (pose bias), CIFAR-10 (synthetic patch bias), and CelebA (gender bias in smile detection), we demonstrate that post-hoc unlearning can substantially reduce subgroup disparities, with improvements in demographic parity of up to \textbf{94.86\%} on CUB-200, \textbf{30.28\%} on CIFAR-10, and \textbf{97.37\%} on CelebA. These gains are achieved with minimal accuracy loss and with methods scoring an average of 0.62 across the 3 settings on the joint evaluation of utility, fairness, quality, and privacy. Our findings establish machine unlearning as a practical framework for enhancing fairness in deployed vision systems without necessitating full retraining. 

**Abstract (ZH)**: Bias-Aware Machine Unlearning: Selectively Removing Biased Samples or Feature Representations for Post-Hoc Mitigation of Bias in Vision Models 

---
# Benchmarking Universal Interatomic Potentials on Zeolite Structures 

**Title (ZH)**: Benchmarking Universal Interatomic Potentials on Zeolite Structures（沸石结构上的通用原子势能基准测试） 

**Authors**: Shusuke Ito, Koki Muraoka, Akira Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2509.07417)  

**Abstract**: Interatomic potentials (IPs) with wide elemental coverage and high accuracy are powerful tools for high-throughput materials discovery. While the past few years witnessed the development of multiple new universal IPs that cover wide ranges of the periodic table, their applicability to target chemical systems should be carefully investigated. We benchmark several universal IPs using equilibrium zeolite structures as testbeds. We select a diverse set of universal IPs encompassing two major categories: (i) universal analytic IPs, including GFN-FF, UFF, and Dreiding; (ii) pretrained universal machine learning IPs (MLIPs), comprising CHGNet, ORB-v3, MatterSim, eSEN-30M-OAM, PFP-v7, and EquiformerV2-lE4-lF100-S2EFS-OC22. We compare them with established tailor-made IPs, SLC, ClayFF, and BSFF using experimental data and density functional theory (DFT) calculations with dispersion correction as the reference. The tested zeolite structures comprise pure silica frameworks and aluminosilicates containing copper species, potassium, and organic cations. We found that GFN-FF is the best among the tested universal analytic IPs, but it does not achieve satisfactory accuracy for highly strained silica rings and aluminosilicate systems. All MLIPs can well reproduce experimental or DFT-level geometries and energetics. Among the universal MLIPs, the eSEN-30M-OAM model shows the most consistent performance across all zeolite structures studied. These findings show that the modern pretrained universal MLIPs are practical tools in zeolite screening workflows involving various compositions. 

**Abstract (ZH)**: 广覆盖高精度的原子势在高通量材料发现中的应用：以平衡骨架沸石结构为基准的评估 

---
# Toward Lifelong-Sustainable Electronic-Photonic AI Systems via Extreme Efficiency, Reconfigurability, and Robustness 

**Title (ZH)**: 面向极致高效、重构能力和鲁棒性的终身可持续电子-光子AI系统 

**Authors**: Ziang Yin, Hongjian Zhou, Chetan Choppali Sudarshan, Vidya Chhabria, Jiaqi Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07396)  

**Abstract**: The relentless growth of large-scale artificial intelligence (AI) has created unprecedented demand for computational power, straining the energy, bandwidth, and scaling limits of conventional electronic platforms. Electronic-photonic integrated circuits (EPICs) have emerged as a compelling platform for next-generation AI systems, offering inherent advantages in ultra-high bandwidth, low latency, and energy efficiency for computing and interconnection. Beyond performance, EPICs also hold unique promises for sustainability. Fabricated in relaxed process nodes with fewer metal layers and lower defect densities, photonic devices naturally reduce embodied carbon footprint (CFP) compared to advanced digital electronic integrated circuits, while delivering orders-of-magnitude higher computing performance and interconnect bandwidth. To further advance the sustainability of photonic AI systems, we explore how electronic-photonic design automation (EPDA) and cross-layer co-design methodologies can amplify these inherent benefits. We present how advanced EPDA tools enable more compact layout generation, reducing both chip area and metal layer usage. We will also demonstrate how cross-layer device-circuit-architecture co-design unlocks new sustainability gains for photonic hardware: ultra-compact photonic circuit designs that minimize chip area cost, reconfigurable hardware topology that adapts to evolving AI workloads, and intelligent resilience mechanisms that prolong lifetime by tolerating variations and faults. By uniting intrinsic photonic efficiency with EPDA- and co-design-driven gains in area efficiency, reconfigurability, and robustness, we outline a vision for lifelong-sustainable electronic-photonic AI systems. This perspective highlights how EPIC AI systems can simultaneously meet the performance demands of modern AI and the urgent imperative for sustainable computing. 

**Abstract (ZH)**: 大规模人工智能（AI）的持续增长创造了前所未有的计算需求，对传统电子平台的能源、带宽和扩展极限形成压力。电子-光子集成电路（EPICs）已成为下一代AI系统有吸引力的平台，提供了超高带宽、低延迟和能效计算与互连的固有优势。除了性能，EPICs还为可持续性提供了独特的前景。利用更宽松的工艺节点、更少的金属层和更低的缺陷密度制造光子器件，自然减少了与先进数字电子集成电路相比的物质碳足迹（CFP），同时提供了数量级更高的计算性能和互连带宽。为推进光子AI系统的可持续性，我们探讨了如何利用电子-光子设计自动化（EPDA）和跨层协同设计方法进一步放大这些固有优势。我们展示了先进的EPDA工具如何实现更紧凑的布局生成，减少芯片面积和金属层使用。我们还将展示跨层器件-电路-架构协同设计如何为光子硬件解锁新的可持续性收益：极紧凑的光子电路设计以最小化芯片面积成本、可重构的硬件拓扑以适应不断变化的AI工作负载，以及智能的容错机制以通过容忍变异和故障来延长寿命。通过结合固有的光子效率与EPDA和协同设计驱动的空间效率、可重构性和鲁棒性增益，我们概述了一种终身可持续的电子-光子AI系统愿景。本文视角突显了EPIC AI系统如何同时满足现代AI的性能需求和对可持续计算的紧迫需求。 

---
# Hybrid GCN-GRU Model for Anomaly Detection in Cryptocurrency Transactions 

**Title (ZH)**: 基于GCN-GRU的混合模型在数字货币交易中的异常检测 

**Authors**: Gyuyeon Na, Minjung Park, Hyeonjeong Cha, Soyoun Kim, Sunyoung Moon, Sua Lee, Jaeyoung Choi, Hyemin Lee, Sangmi Chai  

**Link**: [PDF](https://arxiv.org/pdf/2509.07392)  

**Abstract**: Blockchain transaction networks are complex, with evolving temporal patterns and inter-node relationships. To detect illicit activities, we propose a hybrid GCN-GRU model that captures both structural and sequential features. Using real Bitcoin transaction data (2020-2024), our model achieved 0.9470 Accuracy and 0.9807 AUC-ROC, outperforming all baselines. 

**Abstract (ZH)**: 区块链交易网络复杂多变，具有 evolving 的时间和节点间关系。为了检测非法活动，我们提出了一种混合 GCN-GRU 模型，该模型同时捕捉结构和序列特征。使用 2020-2024 年的真实比特币交易数据，我们的模型实现了 0.9470 的准确率和 0.9807 的 AUC-ROC，优于所有基线。 

---
# SBS: Enhancing Parameter-Efficiency of Neural Representations for Neural Networks via Spectral Bias Suppression 

**Title (ZH)**: SBS: 通过抑制谱偏倚提高神经网络中神经表示的参数效率 

**Authors**: Qihu Xie, Yuan Li, Yi Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07373)  

**Abstract**: Implicit neural representations have recently been extended to represent convolutional neural network weights via neural representation for neural networks, offering promising parameter compression benefits. However, standard multi-layer perceptrons used in neural representation for neural networks exhibit a pronounced spectral bias, hampering their ability to reconstruct high-frequency details effectively. In this paper, we propose SBS, a parameter-efficient enhancement to neural representation for neural networks that suppresses spectral bias using two techniques: (1) a unidirectional ordering-based smoothing that improves kernel smoothness in the output space, and (2) unidirectional ordering-based smoothing aware random fourier features that adaptively modulate the frequency bandwidth of input encodings based on layer-wise parameter count. Extensive evaluations on various ResNet models with datasets CIFAR-10, CIFAR-100, and ImageNet, demonstrate that SBS achieves significantly better reconstruction accuracy with less parameters compared to SOTA. 

**Abstract (ZH)**: 隐神经表示 recently 已经被扩展用于表示卷积神经网络权重，通过神经网络的神经表示，提供了有希望的参数压缩益处。然而，用于神经网络神经表示的标准多层感知机表现出明显的频谱偏差，阻碍了它们有效重建高频细节的能力。在本文中，我们提出一种名为 SBS 的参数高效增强方法，通过两种技术抑制频谱偏差：(1) 基于单向排序的平滑技术，改进了输出空间中的核平滑性；(2) 基于单向排序的平滑技术的随机傅里叶特征，根据不同层的参数数量自适应调节输入编码的频率带宽。在 ResNet 模型上使用 CIFAR-10、CIFAR-100 和 ImageNet 数据集的广泛评估表明，SBS 在较少参数的情况下实现了显著更好的重构精度。 

---
# Word2Spike: Poisson Rate Coding for Associative Memories and Neuromorphic Algorithms 

**Title (ZH)**: Word2Spike: 泊松率编码用于关联记忆和类脑算法 

**Authors**: Archit Kalra, Midhun Sadanand  

**Link**: [PDF](https://arxiv.org/pdf/2509.07361)  

**Abstract**: Spiking neural networks offer a promising path toward energy-efficient, brain-like associative memory. This paper introduces Word2Spike, a novel rate coding mechanism that combines continuous word embeddings and neuromorphic architectures. We develop a one-to-one mapping that converts multi-dimensional word vectors into spike-based attractor states using Poisson processes. Using BitNet b1.58 quantization, we maintain 97% semantic similarity of continuous embeddings on SimLex-999 while achieving 100% reconstruction accuracy on 10,000 words from OpenAI's text-embedding-3-large. We preserve analogy performance (100% of original embedding performance) even under intentionally introduced noise, indicating a resilient mechanism for semantic encoding in neuromorphic systems. Next steps include integrating the mapping with spiking transformers and liquid state machines (resembling Hopfield Networks) for further evaluation. 

**Abstract (ZH)**: 突触神经网络为能量高效的类脑关联记忆提供了有前途的道路。本文介绍了Word2Spike，一种结合连续词嵌入和神经形态架构的新率编码机制。我们开发了一对一的映射，使用泊松过程将多维词向量转换为基于突触的吸引子状态。通过BitNet b1.58量化，我们在SimLex-999上保持了97%的语义相似性的同时，在OpenAI的text-embedding-3-large的10,000个单词上实现了100%的重建准确性。即使在故意引入的噪声下，我们仍保留了类比性能（原始嵌入性能的100%），表明了神经形态系统中语义编码的鲁棒机制。下一步包括将映射与突触变换器和液态状态机（类似霍普菲尔德网络）集成，以进行进一步评估。 

---
# General Demographic Foundation Models for Enhancing Predictive Performance Across Diseases 

**Title (ZH)**: 通用人口统计基础模型在改善跨疾病预测性能中的应用 

**Authors**: Li-Chin Chen, Ji-Tian Sheu, Yuh-Jue Chuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07330)  

**Abstract**: Demographic attributes are universally present in electronic health records and serve as vital predictors in clinical risk stratification and treatment decisions. Despite their significance, these attributes are often relegated to auxiliary roles in model design, with limited attention has been given to learning their representations. This study proposes a General Demographic Pre-trained (GDP) model as a foundational representation framework tailored to age and gender. The model is pre-trained and evaluated using datasets with diverse diseases and population compositions from different geographic regions. The GDP architecture explores combinations of ordering strategies and encoding methods to transform tabular demographic inputs into latent embeddings. Experimental results demonstrate that sequential ordering substantially improves model performance in discrimination, calibration, and the corresponding information gain at each decision tree split, particularly in diseases where age and gender contribute significantly to risk stratification. Even in datasets where demographic attributes hold relatively low predictive value, GDP enhances the representational importance, increasing their influence in downstream gradient boosting models. The findings suggest that foundational models for tabular demographic attributes can generalize across tasks and populations, offering a promising direction for improving predictive performance in healthcare applications. 

**Abstract (ZH)**: Demographic 特征在电子健康记录中普遍存在，是临床风险分层和治疗决策中的关键预测因子。尽管这些特征非常重要，但在模型设计中往往只扮演辅助角色，对它们的表示学习关注不足。本研究提出了一种通用年龄性别预训练（GDP）模型，作为针对年龄和性别的基础表示框架。该模型使用来自不同地理区域且疾病和人口组成多样化的数据集进行预训练和评估。GDP 架构探索排序策略和编码方法的组合，将表格式的Demographic输入转换为潜在嵌入。实验结果表明，序列排序在区分度、校准度以及每个决策树分叉处的相关信息增益方面显著提高了模型性能，特别是在年龄和性别对风险分层有重大影响的疾病中。即使在Demographic特征预测价值相对较低的数据集中，GDP 也能增强其表示的重要性，提高其在下游梯度提升模型中的影响力。研究结果表明，针对表格式Demographic特征的基础模型可以在不同任务和人群中泛化，为在医疗保健应用中提高预测性能提供了有希望的方向。 

---
# MEGG: Replay via Maximally Extreme GGscore in Incremental Learning for Neural Recommendation Models 

**Title (ZH)**: MEGG：增量学习中基于GGscore极值的重放方法在神经推荐模型中的应用 

**Authors**: Yunxiao Shi, Shuo Yang, Haimin Zhang, Li Wang, Yongze Wang, Qiang Wu, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07319)  

**Abstract**: Neural Collaborative Filtering models are widely used in recommender systems but are typically trained under static settings, assuming fixed data distributions. This limits their applicability in dynamic environments where user preferences evolve. Incremental learning offers a promising solution, yet conventional methods from computer vision or NLP face challenges in recommendation tasks due to data sparsity and distinct task paradigms. Existing approaches for neural recommenders remain limited and often lack generalizability. To address this, we propose MEGG, Replay Samples with Maximally Extreme GGscore, an experience replay based incremental learning framework. MEGG introduces GGscore, a novel metric that quantifies sample influence, enabling the selective replay of highly influential samples to mitigate catastrophic forgetting. Being model-agnostic, MEGG integrates seamlessly across architectures and frameworks. Experiments on three neural models and four benchmark datasets show superior performance over state-of-the-art baselines, with strong scalability, efficiency, and robustness. Implementation will be released publicly upon acceptance. 

**Abstract (ZH)**: 基于最大极端GG分数重放样本的增量学习框架MEGG 

---
# Basis Vector Metric: A Method for Robust Open-Ended State Change Detection 

**Title (ZH)**: 基向量度量：一种稳健的开放性状态变化检测方法 

**Authors**: David Oprea, Sam Powers  

**Link**: [PDF](https://arxiv.org/pdf/2509.07308)  

**Abstract**: We test a new method, which we will abbreviate using the acronym BVM (Basis Vectors Method), in its ability to judge the state changes in images through using language embeddings. We used the MIT-States dataset, containing about 53,000 images, to gather all of our data, which has 225 nouns and 115 adjectives, with each noun having about 9 different adjectives, forming approximately 1000 noun-adjective pairs. For our first experiment, we test our method's ability to determine the state of each noun class separately against other metrics for comparison. These metrics are cosine similarity, dot product, product quantization, binary index, Naive Bayes, and a custom neural network. Among these metrics, we found that our proposed BVM performs the best in classifying the states for each noun. We then perform a second experiment where we try using BVM to determine if it can differentiate adjectives from one another for each adjective separately. We compared the abilities of BVM to differentiate adjectives against the proposed method the MIT-States paper suggests: using a logistic regression model. In the end, we did not find conclusive evidence that our BVM metric could perform better than the logistic regression model at discerning adjectives. Yet, we were able to find evidence for possible improvements to our method; this leads to the chance of increasing our method's accuracy through certain changes in our methodologies. 

**Abstract (ZH)**: 我们测试了一种新方法（我们将其缩写为BVM，即Basis Vectors Method），该方法通过使用语言嵌入来判断图像状态变化的能力。我们使用包含约53,000张图像的MIT-States数据集来收集所有数据，该数据集包含225个名词和115个形容词，每个名词约有9个不同的形容词，形成大约1000个名词-形容词对。在第一个实验中，我们测试了该方法在与其他指标对比时单独判断每个名词类状态的能力。这些指标包括余弦相似度、点积、产品量化、二进制索引、朴素贝叶斯和自定义神经网络。在这些指标中，我们发现我们提出的方法BVM在分类每个名词的状态方面表现最佳。在第二个实验中，我们尝试使用BVM来判断它是否可以单独区分每个形容词。我们将BVM区分形容词的能力与其所建议的方法——使用逻辑回归模型——进行了比较。最终，我们没有找到确凿证据表明我们的BVM指标在区分形容词方面比逻辑回归模型表现更好。然而，我们确实找到了改进方法的可能性证据；这为我们通过某些方法学上的调整提高方法准确性提供了机会。 

---
# zkUnlearner: A Zero-Knowledge Framework for Verifiable Unlearning with Multi-Granularity and Forgery-Resistance 

**Title (ZH)**: zkUnlearner：一种具有多粒度和伪造抵抗性的可验证遗忘零知识框架 

**Authors**: Nan Wang, Nan Wu, Xiangyu Hui, Jiafan Wang, Xin Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.07290)  

**Abstract**: As the demand for exercising the "right to be forgotten" grows, the need for verifiable machine unlearning has become increasingly evident to ensure both transparency and accountability. We present {\em zkUnlearner}, the first zero-knowledge framework for verifiable machine unlearning, specifically designed to support {\em multi-granularity} and {\em forgery-resistance}.
First, we propose a general computational model that employs a {\em bit-masking} technique to enable the {\em selectivity} of existing zero-knowledge proofs of training for gradient descent algorithms. This innovation enables not only traditional {\em sample-level} unlearning but also more advanced {\em feature-level} and {\em class-level} unlearning. Our model can be translated to arithmetic circuits, ensuring compatibility with a broad range of zero-knowledge proof systems. Furthermore, our approach overcomes key limitations of existing methods in both efficiency and privacy. Second, forging attacks present a serious threat to the reliability of unlearning. Specifically, in Stochastic Gradient Descent optimization, gradients from unlearned data, or from minibatches containing it, can be forged using alternative data samples or minibatches that exclude it. We propose the first effective strategies to resist state-of-the-art forging attacks. Finally, we benchmark a zkSNARK-based instantiation of our framework and perform comprehensive performance evaluations to validate its practicality. 

**Abstract (ZH)**: 基于零知识的可验证机器遗忘系统zkUnlearner：多粒度和抗伪造设计 

---
# ALICE: An Interpretable Neural Architecture for Generalization in Substitution Ciphers 

**Title (ZH)**: ALICE: 一种可解释的神经架构，用于替代密码中的泛化 

**Authors**: Jeff Shen, Lindsay Smith  

**Link**: [PDF](https://arxiv.org/pdf/2509.07282)  

**Abstract**: We present cryptogram solving as an ideal testbed for studying neural network generalization in combinatorially complex domains. In this task, models must decrypt text encoded with substitution ciphers, choosing from 26! possible mappings without explicit access to the cipher. We develop ALICE (an Architecture for Learning Interpretable Cryptogram dEcipherment): a simple encoder-only Transformer that sets a new state-of-the-art for both accuracy and speed on this decryption problem. Surprisingly, ALICE generalizes to unseen ciphers after training on only ${\sim}1500$ unique ciphers, a minute fraction ($3.7 \times 10^{-24}$) of the possible cipher space. To enhance interpretability, we introduce a novel bijective decoding head that explicitly models permutations via the Gumbel-Sinkhorn method, enabling direct extraction of learned cipher mappings. Through early exit analysis, we reveal how ALICE progressively refines its predictions in a way that appears to mirror common human strategies for this task: early layers employ frequency-based heuristics, middle layers form word structures, and final layers correct individual characters. Our architectural innovations and analysis methods extend beyond cryptograms to any domain with bijective mappings and combinatorial structure, offering new insights into neural network generalization and interpretability. 

**Abstract (ZH)**: 我们将密码破解视为研究神经网络在组合复杂领域泛化能力的理想试验床。在这个任务中，模型必须解密使用置换密码编码的文本，选择从中解出的映射多达26!种，而无需显式访问密码。我们开发了ALICE（一种学习可解析密码破解的架构）：一个简单的仅编码器Transformer，在此解密问题上同时达到了准确性和速度的新标准。令人惊讶的是，ALICE在仅训练于约1500种独特密码后便能够泛化到未见过的密码，这是可能的密码空间的一分钟分数（3.7×10^(-24)）。为增强可解释性，我们引入了一种新颖的双射解码头，通过Gumbel-Sinkhorn方法显式建模置换，从而可以直接提取所学习的密码映射。通过早期退出分析，我们揭示了ALICE如何逐步细化其预测，这种方式似乎与人类解决此任务时常用的方法相当吻合：早期层使用基于频率的启发式方法，中间层形成单词结构，最终层修正单个字符。我们的架构创新和分析方法不仅限于密码破解领域，还可扩展到任何具有双射映射和组合结构的领域，提供了关于神经网络泛化能力和可解释性的新见解。 

---
# Datasets for Navigating Sensitive Topics in Recommendation Systems 

**Title (ZH)**: 用于导航推荐系统中敏感话题的数据集 

**Authors**: Amelia Kovacs, Jerry Chee, Kimia Kazemian, Sarah Dean  

**Link**: [PDF](https://arxiv.org/pdf/2509.07269)  

**Abstract**: Personalized AI systems, from recommendation systems to chatbots, are a prevalent method for distributing content to users based on their learned preferences. However, there is growing concern about the adverse effects of these systems, including their potential tendency to expose users to sensitive or harmful material, negatively impacting overall well-being. To address this concern quantitatively, it is necessary to create datasets with relevant sensitivity labels for content, enabling researchers to evaluate personalized systems beyond mere engagement metrics. To this end, we introduce two novel datasets that include a taxonomy of sensitivity labels alongside user-content ratings: one that integrates MovieLens rating data with content warnings from the Does the Dog Die? community ratings website, and another that combines fan-fiction interaction data and user-generated warnings from Archive of Our Own. 

**Abstract (ZH)**: 个性化AI系统，从推荐系统到聊天机器人，是根据用户的学习偏好分发内容的一种普遍方法。然而，这些系统可能带来的不利影响引起了广泛关注，包括它们有可能向用户暴露敏感或有害的内容，从而负面影响整体福祉。为了从定量角度应对这一关切，有必要创建包含相关敏感标签的数据集，使研究人员能够超越单纯的参与度指标来评估个性化系统。为此，我们介绍了两个新的数据集，这些数据集包括敏感标签的分类体系和用户内容评分：一个将MovieLens评分数据与Does the Dog Die?社区评级网站的内容警告相结合，另一个则将粉丝小说交互数据与Archive of Our Own用户生成的警告相结合。 

---
# Breaking the Conventional Forward-Backward Tie in Neural Networks: Activation Functions 

**Title (ZH)**: 打破神经网络中前向-后向传输的常规绑定：激活函数 

**Authors**: Luigi Troiano, Francesco Gissi, Vincenzo Benedetto, Genny Tortora  

**Link**: [PDF](https://arxiv.org/pdf/2509.07236)  

**Abstract**: Gradient-based neural network training traditionally enforces symmetry between forward and backward propagation, requiring activation functions to be differentiable (or sub-differentiable) and strictly monotonic in certain regions to prevent flat gradient areas. This symmetry, linking forward activations closely to backward gradients, significantly restricts the selection of activation functions, particularly excluding those with substantial flat or non-differentiable regions. In this paper, we challenge this assumption through mathematical analysis, demonstrating that precise gradient magnitudes derived from activation functions are largely redundant, provided the gradient direction is preserved. Empirical experiments conducted on foundational architectures - such as Multi-Layer Perceptrons (MLPs), Convolutional Neural Networks (CNNs), and Binary Neural Networks (BNNs) - confirm that relaxing forward-backward symmetry and substituting traditional gradients with simpler or stochastic alternatives does not impair learning and may even enhance training stability and efficiency. We explicitly demonstrate that neural networks with flat or non-differentiable activation functions, such as the Heaviside step function, can be effectively trained, thereby expanding design flexibility and computational efficiency. Further empirical validation with more complex architectures remains a valuable direction for future research. 

**Abstract (ZH)**: 基于梯度的神经网络训练传统上要求前向传播和反向传播保持对称性，需要激活函数在某些区域是可微的（或下可微的）且严格单调，以防止梯度平坦区域的出现。这种对称性将前向激活紧密关联到后向梯度，极大地限制了激活函数的选择，特别是排除了具有显著平坦或非可微区域的激活函数。本文通过数学分析挑战了这一假设，证明了只要保留梯度方向，从激活函数导出的精确梯度幅度往往是多余的。在基础架构，如多层感知机（MLPs）、卷积神经网络（CNNs）和二值神经网络（BNNs）等上进行的经验实验确认，放松前向-后向对称性并用更简单或随机的梯度替代传统梯度不会损害学习，甚至可能提高训练的稳定性和效率。明确展示了具有平坦或非可微激活函数（如单位阶跃函数）的神经网络可以有效训练，从而扩展了设计灵活性和计算效率。进一步使用更复杂架构的经验验证是未来研究的一个有价值的方向。 

---
# A transformer-based generative model for planetary systems 

**Title (ZH)**: 基于变压器的生成模型行星系统 

**Authors**: Yann Alibert, Jeanne Davoult, Sara Marques  

**Link**: [PDF](https://arxiv.org/pdf/2509.07226)  

**Abstract**: Numerical calculations of planetary system formation are very demanding in terms of computing power. These synthetic planetary systems can however provide access to correlations, as predicted in a given numerical framework, between the properties of planets in the same system. Such correlations can, in return, be used in order to guide and prioritize observational campaigns aiming at discovering some types of planets, as Earth-like planets. Our goal is to develop a generative model which is capable of capturing correlations and statistical relationships between planets in the same system. Such a model, trained on the Bern model, offers the possibility to generate large number of synthetic planetary systems with little computational cost, that can be used, for example, to guide observational campaigns. Our generative model is based on the transformer architecture which is well-known to efficiently capture correlations in sequences and is at the basis of all modern Large Language Models. To assess the validity of the generative model, we perform visual and statistical comparisons, as well as a machine learning driven tests. Finally, as a use case example, we consider the TOI-469 system, in which we aim at predicting the possible properties of planets c and d, based on the properties of planet b (the first that has been detected). We show using different comparison methods that the properties of systems generated by our model are very similar to the ones of the systems computed directly by the Bern model. We also show in the case of the TOI-469 system, that using the generative model allows to predict the properties of planets not yet observed, based on the properties of the already observed planet. We provide our model to the community on our website this http URL. 

**Abstract (ZH)**: 数值计算行星系统形成对计算能力要求非常高，但这些合成的行星系统可以提供访问在特定数值框架中预测的相关性，从而可以利用这些相关性来指导和优先排序旨在发现某些类型行星（如类地行星）的观测计划。我们的目标是开发一个能够捕捉相同系统内行星之间相关性和统计关系的生成模型。基于训练于Bern模型的该生成模型，可以以较低的计算成本生成大量的合成行星系统，这些系统可用于指导观测计划等。我们的生成模型基于已知能高效捕捉序列中相关性的变换器架构，这是所有现代大规模语言模型的基础。为了评估生成模型的有效性，我们进行了视觉和统计比较，以及机器学习驱动的测试。最后，作为使用案例示例，我们考虑TOI-469系统，在该系统中，基于已检测到的第一颗行星b的属性，我们旨在预测行星c和d的可能属性。我们使用不同的比较方法表明，由我们模型生成的系统属性与直接由Bern模型计算的系统属性非常相似。此外，在TOI-469系统案例中，使用生成模型可以通过已观测行星的属性预测尚未观测到的行星的属性。我们在网站上将我们的模型分享给社区：this http URL。 

---
# Explaining How Quantization Disparately Skews a Model 

**Title (ZH)**: 解释量化如何不公平地偏斜模型 

**Authors**: Abhimanyu Bellam, Jung-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.07222)  

**Abstract**: Post Training Quantization (PTQ) is widely adopted due to its high compression capacity and speed with minimal impact on accuracy. However, we observed that disparate impacts are exacerbated by quantization, especially for minority groups. Our analysis explains that in the course of quantization there is a chain of factors attributed to a disparate impact across groups during forward and backward passes. We explore how the changes in weights and activations induced by quantization cause cascaded impacts in the network, resulting in logits with lower variance, increased loss, and compromised group accuracies. We extend our study to verify the influence of these impacts on group gradient norms and eigenvalues of the Hessian matrix, providing insights into the state of the network from an optimization point of view. To mitigate these effects, we propose integrating mixed precision Quantization Aware Training (QAT) with dataset sampling methods and weighted loss functions, therefore providing fair deployment of quantized neural networks. 

**Abstract (ZH)**: Post Training Quantization (PTQ) 因其高压缩能力和对速度影响 minimal 以及对准确性的 minor 影响而被广泛采用。然而，我们发现量化会加剧不同群体间的差异影响，特别是对少数群体。我们的分析表明，在量化过程中，由于正向和反向传播中的一系列因素，这种差异影响在群体间加剧。我们探讨了量化引起的权重和激活值变化如何在网络中引发连锁影响，导致对数几率 lower 变异、增加的损失以及群体准确性的下降。我们将研究扩展到验证这些影响对群体梯度范数和海森矩阵特征值的影响，从而从优化的角度提供网络状态的见解。为了减轻这些影响，我们提出了结合混合精度感知量化训练（QAT）和数据集采样方法以及加权损失函数的方法，以实现量化神经网络的公平部署。 

---
# A multi-strategy improved gazelle optimization algorithm for solving numerical optimization and engineering applications 

**Title (ZH)**: 多策略改进羚羊优化算法求解数值优化及其工程应用 

**Authors**: Qi Diao, Chengyue Xie, Yuchen Yin, Hoileong Lee, Haolong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07211)  

**Abstract**: Aiming at the shortcomings of the gazelle optimization algorithm, such as the imbalance between exploration and exploitation and the insufficient information exchange within the population, this paper proposes a multi-strategy improved gazelle optimization algorithm (MSIGOA). To address these issues, MSIGOA proposes an iteration-based updating framework that switches between exploitation and exploration according to the optimization process, which effectively enhances the balance between local exploitation and global exploration in the optimization process and improves the convergence speed. Two adaptive parameter tuning strategies improve the applicability of the algorithm and promote a smoother optimization process. The dominant population-based restart strategy enhances the algorithms ability to escape from local optima and avoid its premature convergence. These enhancements significantly improve the exploration and exploitation capabilities of MSIGOA, bringing superior convergence and efficiency in dealing with complex problems. In this paper, the parameter sensitivity, strategy effectiveness, convergence and stability of the proposed method are evaluated on two benchmark test sets including CEC2017 and CEC2022. Test results and statistical tests show that MSIGOA outperforms basic GOA and other advanced algorithms. On the CEC2017 and CEC2022 test sets, the proportion of functions where MSIGOA is not worse than GOA is 92.2% and 83.3%, respectively, and the proportion of functions where MSIGOA is not worse than other algorithms is 88.57% and 87.5%, respectively. Finally, the extensibility of MSIGAO is further verified by several engineering design optimization problems. 

**Abstract (ZH)**: 针对瞪羚优化算法中探索与开发之间的不平衡以及种群内部信息交流不足等问题，本文提出了一种多策略改进瞪羚优化算法（MSIGOA）。MSIGOA提出了一种基于迭代的更新框架，根据优化过程在开发和探索之间切换，有效提高了优化过程中的局部开发与全局探索之间的平衡，提升了收敛速度。两种自适应参数调整策略提高了算法的适用性，并促进了优化过程的平滑进行。主导的基于种群的重启策略增强了算法跳出局部最优解的能力，避免了过早收敛。这些改进显著提升了MSIGOA的探索与开发能力，在处理复杂问题时表现出优越的收敛性和效率。在本文中，通过CEC2017和CEC2022两个基准测试集评估了所提出方法的参数敏感性、策略有效性、收敛性和稳定性。实验结果和统计检验表明，MSIGOA在处理优化问题时优于基本瞪羚优化算法和其他高级算法。在CEC2017和CEC2022测试集上，MSIGOA在函数表现上不低于瞪羚优化算法的比例分别为92.2%和83.3%，在函数表现上不低于其他算法的比例分别为88.57%和87.5%。最后，通过几个工程设计优化问题进一步验证了MSIGOA的可扩展性。 

---
# Adversarial Attacks on Audio Deepfake Detection: A Benchmark and Comparative Study 

**Title (ZH)**: 音频深度假音检测的对抗攻击：基准与比较研究 

**Authors**: Kutub Uddin, Muhammad Umar Farooq, Awais Khan, Khalid Mahmood Malik  

**Link**: [PDF](https://arxiv.org/pdf/2509.07132)  

**Abstract**: The widespread use of generative AI has shown remarkable success in producing highly realistic deepfakes, posing a serious threat to various voice biometric applications, including speaker verification, voice biometrics, audio conferencing, and criminal investigations. To counteract this, several state-of-the-art (SoTA) audio deepfake detection (ADD) methods have been proposed to identify generative AI signatures to distinguish between real and deepfake audio. However, the effectiveness of these methods is severely undermined by anti-forensic (AF) attacks that conceal generative signatures. These AF attacks span a wide range of techniques, including statistical modifications (e.g., pitch shifting, filtering, noise addition, and quantization) and optimization-based attacks (e.g., FGSM, PGD, C \& W, and DeepFool). In this paper, we investigate the SoTA ADD methods and provide a comparative analysis to highlight their effectiveness in exposing deepfake signatures, as well as their vulnerabilities under adversarial conditions. We conducted an extensive evaluation of ADD methods on five deepfake benchmark datasets using two categories: raw and spectrogram-based approaches. This comparative analysis enables a deeper understanding of the strengths and limitations of SoTA ADD methods against diverse AF attacks. It does not only highlight vulnerabilities of ADD methods, but also informs the design of more robust and generalized detectors for real-world voice biometrics. It will further guide future research in developing adaptive defense strategies that can effectively counter evolving AF techniques. 

**Abstract (ZH)**: 生成式AI在 produciing 高度逼真的深伪方面的广泛应用对包括语音生物特征识别、语音生物特征、音频会议和刑事调查在内的多种语音生物特征应用构成了严重威胁。为了应对这一挑战，已经提出了一些最先进的（SoTA）音频深伪检测（ADD）方法，用于识别生成式AI的签名以区分真实语音和深伪语音。然而，这些方法的有效性严重受到反取证（AF）攻击的影响，这些攻击通过隐蔽生成式签名的方式削弱了检测方法的效果。这些AF攻击包括统计修改（如音调变换、滤波、噪声添加和量化）和基于优化的攻击（如FGSM、PGD、C&W和DeepFool）。在本文中，我们研究了最先进的ADD方法，并提供了比较分析，以突出这些方法在揭露深伪签名方面的有效性及其在对抗条件下的脆弱性。我们使用原始和频谱图基方法对五种深伪基准数据集上的ADD方法进行了全面评估。这种比较分析有助于更深入地了解最先进的ADD方法在面对多种AF攻击时的优势和局限性，不仅突显了ADD方法的脆弱性，还为设计更 robust 和通用的检测器提供指导，以有效应对现实世界中的语音生物特征应用。这还将进一步指导未来研究，开发能够有效对抗不断演变的AF技术的自适应防御策略。 

---
# SoK: Security and Privacy of AI Agents for Blockchain 

**Title (ZH)**: SoK: AI代理在区块链中的安全与隐私 

**Authors**: Nicolò Romandini, Carlo Mazzocca, Kai Otsuki, Rebecca Montanari  

**Link**: [PDF](https://arxiv.org/pdf/2509.07131)  

**Abstract**: Blockchain and smart contracts have garnered significant interest in recent years as the foundation of a decentralized, trustless digital ecosystem, thereby eliminating the need for traditional centralized authorities. Despite their central role in powering Web3, their complexity still presents significant barriers for non-expert users. To bridge this gap, Artificial Intelligence (AI)-based agents have emerged as valuable tools for interacting with blockchain environments, supporting a range of tasks, from analyzing on-chain data and optimizing transaction strategies to detecting vulnerabilities within smart contracts. While interest in applying AI to blockchain is growing, the literature still lacks a comprehensive survey that focuses specifically on the intersection with AI agents. Most of the related work only provides general considerations, without focusing on any specific domain. This paper addresses this gap by presenting the first Systematization of Knowledge dedicated to AI-driven systems for blockchain, with a special focus on their security and privacy dimensions, shedding light on their applications, limitations, and future research directions. 

**Abstract (ZH)**: 区块链和智能合约近年来引起了广泛关注，成为去中心化、无信任数字生态系统的基础，从而消除了对传统中心化权威机构的需要。尽管它们在支撑Web3中发挥着中心作用，但其复杂性仍然为非专家用户设立了显著障碍。为弥合这一差距，基于人工智能（AI）的代理已经 emerged 作为与区块链环境交互的有价值的工具，支持从分析链上数据、优化交易策略到检测智能合约漏洞等一系列任务。虽然将人工智能应用于区块链的兴趣在增长，但相关文献仍然缺乏专注于与AI代理交汇的具体综述。大部分相关工作仅提供一般性考虑，未专注于任何特定领域。本文通过首次提出针对区块链的AI驱动系统的知识体系结构化，特别关注其安全和隐私维度，阐明了其应用场景、局限性和未来研究方向。 

---
# SVGauge: Towards Human-Aligned Evaluation for SVG Generation 

**Title (ZH)**: SVGauge: 朝向与人类视角一致的SVG生成评估 

**Authors**: Leonardo Zini, Elia Frigieri, Sebastiano Aloscari, Marcello Generali, Lorenzo Dodi, Robert Dosen, Lorenzo Baraldi  

**Link**: [PDF](https://arxiv.org/pdf/2509.07127)  

**Abstract**: Generated Scalable Vector Graphics (SVG) images demand evaluation criteria tuned to their symbolic and vectorial nature: criteria that existing metrics such as FID, LPIPS, or CLIPScore fail to satisfy. In this paper, we introduce SVGauge, the first human-aligned, reference based metric for text-to-SVG generation. SVGauge jointly measures (i) visual fidelity, obtained by extracting SigLIP image embeddings and refining them with PCA and whitening for domain alignment, and (ii) semantic consistency, captured by comparing BLIP-2-generated captions of the SVGs against the original prompts in the combined space of SBERT and TF-IDF. Evaluation on the proposed SHE benchmark shows that SVGauge attains the highest correlation with human judgments and reproduces system-level rankings of eight zero-shot LLM-based generators more faithfully than existing metrics. Our results highlight the necessity of vector-specific evaluation and provide a practical tool for benchmarking future text-to-SVG generation models. 

**Abstract (ZH)**: Generated Scalable Vector Graphics (SVG) 图像需要针对其符号性和向量性质的评价标准：现有指标（如 FID、LPIPS 或 CLIPScore）未能满足这些需求。在此论文中，我们提出了 SVGauge，这是首个针对文本到 SVG 生成的人类对齐参考基线指标。SVGauge 联合度量 (i) 视觉保真度，通过提取 SigLIP 图像嵌入并使用 PCA 和白化进行领域对齐进行细化；以及 (ii) 语义一致性，通过将 BLIP-2 生成的 SVG 标题与结合 SBERT 和 TF-IDF 的原始提示进行比较来捕捉。在提出的 SHE 基准上的评估表明，SVGauge 在与人类判断的相关性和忠实再现八种零样本 LLM 基础生成器的系统级排名方面优于现有指标。我们的结果强调了向量特定评价的必要性，并为评估未来文本到 SVG 生成模型提供了一个实用工具。 

---
# Riemannian Batch Normalization: A Gyro Approach 

**Title (ZH)**: 黎曼流形批量归一化：陀螺仪方法 

**Authors**: Ziheng Chen, Xiao-Jun Wu, Nicu Sebe  

**Link**: [PDF](https://arxiv.org/pdf/2509.07115)  

**Abstract**: Normalization layers are crucial for deep learning, but their Euclidean formulations are inadequate for data on manifolds. On the other hand, many Riemannian manifolds in machine learning admit gyro-structures, enabling principled extensions of Euclidean neural networks to non-Euclidean domains. Inspired by this, we introduce GyroBN, a principled Riemannian batch normalization framework for gyrogroups. We establish two necessary conditions, namely \emph{pseudo-reduction} and \emph{gyroisometric gyrations}, that guarantee GyroBN with theoretical control over sample statistics, and show that these conditions hold for all known gyrogroups in machine learning. Our framework also incorporates several existing Riemannian normalization methods as special cases. We further instantiate GyroBN on seven representative geometries, including the Grassmannian, five constant curvature spaces, and the correlation manifold, and derive novel gyro and Riemannian structures to enable these instantiations. Experiments across these geometries demonstrate the effectiveness of GyroBN. The code is available at this https URL. 

**Abstract (ZH)**: 广义布朗规范化：广义群上的原理化黎曼批量归一化框架 

---
# Lookup multivariate Kolmogorov-Arnold Networks 

**Title (ZH)**: 查找多元柯尔莫哥洛夫-阿诺尔德网络 

**Authors**: Sergey Pozdnyakov, Philippe Schwaller  

**Link**: [PDF](https://arxiv.org/pdf/2509.07103)  

**Abstract**: High-dimensional linear mappings, or linear layers, dominate both the parameter count and the computational cost of most modern deep-learning models. We introduce a general drop-in replacement, lookup multivariate Kolmogorov-Arnold Networks (lmKANs), which deliver a substantially better trade-off between capacity and inference cost. Our construction expresses a general high-dimensional mapping through trainable low-dimensional multivariate functions. These functions can carry dozens or hundreds of trainable parameters each, and yet it takes only a few multiplications to compute them because they are implemented as spline lookup tables. Empirically, lmKANs reduce inference FLOPs by up to 6.0x while matching the flexibility of MLPs in general high-dimensional function approximation. In another feedforward fully connected benchmark, on the tabular-like dataset of randomly displaced methane configurations, lmKANs enable more than 10x higher H100 throughput at equal accuracy. Within frameworks of Convolutional Neural Networks, lmKAN-based CNNs cut inference FLOPs at matched accuracy by 1.6-2.1x and by 1.7x on the CIFAR-10 and ImageNet-1k datasets, respectively. Our code, including dedicated CUDA kernels, is available online at this https URL. 

**Abstract (ZH)**: 高维线性映射或线性层主导了大多数现代深度学习模型的参数量和计算成本。我们引入了一种通用的即插即用替换方案——查找多元柯尔莫哥洛夫-阿诺尔德网络（lmKANs），它在容量和推理成本之间提供了显著更好的权衡。我们的构造通过可训练的低维多元函数表达了一般高维映射。这些函数可以承载数十到数百个可训练参数，但由于它们是通过样条查找表实现的，因此只需几次乘法即可计算。实验结果表明，lmKANs在匹配MLP的一般高维函数逼近灵活性的同时，将推理FLOPs减少多达6.0倍。在另一个前向全连接基准测试中，对于随机位移的一系列甲烷配置表格化数据集，lmKANs在相同精度下使H100的吞吐量提高了10倍以上。在卷积神经网络框架内，基于lmKAN的CNN将匹配精度下的推理FLOPs减少了1.6至2.1倍，分别在CIFAR-10和ImageNet-1k数据集上减少了1.7倍。我们的代码，包括专用的CUDA内核，可在以下链接在线获取。 

---
# Controllable Singing Voice Synthesis using Phoneme-Level Energy Sequence 

**Title (ZH)**: 基于音位级能量序列的可控歌声合成 

**Authors**: Yerin Ryu, Inseop Shin, Chanwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.07038)  

**Abstract**: Controllable Singing Voice Synthesis (SVS) aims to generate expressive singing voices reflecting user intent. While recent SVS systems achieve high audio quality, most rely on probabilistic modeling, limiting precise control over attributes such as dynamics. We address this by focusing on dynamic control--temporal loudness variation essential for musical expressiveness--and explicitly condition the SVS model on energy sequences extracted from ground-truth spectrograms, reducing annotation costs and improving controllability. We also propose a phoneme-level energy sequence for user-friendly control. To the best of our knowledge, this is the first attempt enabling user-driven dynamics control in SVS. Experiments show our method achieves over 50% reduction in mean absolute error of energy sequences for phoneme-level inputs compared to baseline and energy-predictor models, without compromising synthesis quality. 

**Abstract (ZH)**: 可控歌唱语音合成（SVS）旨在生成反映用户意图的表达性歌唱声音。虽然近期的SVS系统在音频质量方面取得高成就，但大多数仍然依赖概率建模，限制了对诸如动态等属性的精准控制。我们通过专注于音乐表达性所必需的动态控制——时间响度变化——并明确将SVS模型条件化于从真实光谱图提取的能量序列，从而降低标注成本并提高可控性。我们还提出了一种音素级别能量序列以实现用户友好的控制。据我们所知，这是第一次在SVS中实现用户驱动的动态控制的尝试。实验结果显示，与基线模型和能量预测模型相比，我们的方法在音素级别输入的能量序列的平均绝对误差上降低了超过50%，且不牺牲合成质量。 

---
# Methodological Insights into Structural Causal Modelling and Uncertainty-Aware Forecasting for Economic Indicators 

**Title (ZH)**: 结构因果建模与不确定性意识预测在经济指标中的方法论洞察 

**Authors**: Federico Cerutti  

**Link**: [PDF](https://arxiv.org/pdf/2509.07036)  

**Abstract**: This paper presents a methodological approach to financial time series analysis by combining causal discovery and uncertainty-aware forecasting. As a case study, we focus on four key U.S. macroeconomic indicators -- GDP, economic growth, inflation, and unemployment -- and we apply the LPCMCI framework with Gaussian Process Distance Correlation (GPDC) to uncover dynamic causal relationships in quarterly data from 1970 to 2021. Our results reveal a robust unidirectional causal link from economic growth to GDP and highlight the limited connectivity of inflation, suggesting the influence of latent factors. Unemployment exhibits strong autoregressive dependence, motivating its use as a case study for probabilistic forecasting. Leveraging the Chronos framework, a large language model trained for time series, we perform zero-shot predictions on unemployment. This approach delivers accurate forecasts one and two quarters ahead, without requiring task-specific training. Crucially, the model's uncertainty-aware predictions yield 90\% confidence intervals, enabling effective anomaly detection through statistically principled deviation analysis. This study demonstrates the value of combining causal structure learning with probabilistic language models to inform economic policy and enhance forecasting robustness. 

**Abstract (ZH)**: 本文提出一种结合因果发现和不确定性感知预测的方法论金融时间序列分析方法。作为案例研究，我们聚焦于四类关键的美国宏观经济指标——GDP、经济增长、通货膨胀和失业率——并应用LPCMCI框架结合高斯过程距离相关（GPDC）来揭示从1970年至2021年季度数据中动态因果关系。研究结果表明经济增长向GDP存在稳健的单向因果联系，而通货膨胀的连接性有限，暗示了潜在因素的影响。失业率表现出强烈的自回归依赖性，使其成为概率预测的典型案例。利用Chronos框架，一种为时间序列训练的大语言模型，我们进行失业率的零样本预测。该方法在提前一个季度和两个季度时提供准确的预测，无需针对特定任务进行训练。模型的不确定性感知预测生成90%的置信区间，能够通过统计原理分析来有效检测异常。本研究展示了将因果结构学习与概率语言模型相结合，以指导经济政策制定并增强预测稳健性的价值。 

---
# A Maslow-Inspired Hierarchy of Engagement with AI Model 

**Title (ZH)**: 基于马斯洛需求层次理论的AI交互层次模型 

**Authors**: Madara Ogot  

**Link**: [PDF](https://arxiv.org/pdf/2509.07032)  

**Abstract**: The rapid proliferation of artificial intelligence (AI) across industry, government, and education highlights the urgent need for robust frameworks to conceptualise and guide engagement. This paper introduces the Hierarchy of Engagement with AI model, a novel maturity framework inspired by Maslow's hierarchy of needs. The model conceptualises AI adoption as a progression through eight levels, beginning with initial exposure and basic understanding and culminating in ecosystem collaboration and societal impact. Each level integrates technical, organisational, and ethical dimensions, emphasising that AI maturity is not only a matter of infrastructure and capability but also of trust, governance, and responsibility. Initial validation of the model using four diverse case studies (General Motors, the Government of Estonia, the University of Texas System, and the African Union AI Strategy) demonstrate the model's contextual flexibility across various sectors. The model provides scholars with a framework for analysing AI maturity and offers practitioners and policymakers a diagnostic and strategic planning tool to guide responsible and sustainable AI engagement. The proposed model demonstrates that AI maturity progression is multi-dimensional, requiring technological capability, ethical integrity, organisational resilience, and ecosystem collaboration. 

**Abstract (ZH)**: 人工智能在工业、政府和教育领域的迅速普及凸显了建立稳健框架以概念化和指导参与的迫切需求。本文引入了人工智能参与层次模型，该模型是一种受马斯洛需求层次理论启发的创新成熟度框架。该模型将人工智能采用概念化为八个阶段的 progression，从初步接触和基础理解开始，最终达到生态系统合作和社会影响。每个阶段整合了技术、组织和伦理维度，强调人工智能成熟度不仅仅是基础设施和能力的问题，同时也是信任、治理和责任的问题。通过四个不同的案例研究（通用汽车、爱沙尼亚政府、德克萨斯大学系统和非洲联盟人工智能战略）的初步验证显示，该模型具有跨各种领域的灵活性。该模型为学者们提供了一个分析人工智能成熟度的框架，并为从业人员和政策制定者提供了一个诊断和战略规划工具，以指导负责任和可持续的人工智能参与。所提出的人工智能成熟度模型证明了其多维性，需要技术能力、伦理诚信、组织韧性和生态系统合作。 

---
# A Minimalist Bayesian Framework for Stochastic Optimization 

**Title (ZH)**: 最小主义贝叶斯框架下的随机优化 

**Authors**: Kaizheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07030)  

**Abstract**: The Bayesian paradigm offers principled tools for sequential decision-making under uncertainty, but its reliance on a probabilistic model for all parameters can hinder the incorporation of complex structural constraints. We introduce a minimalist Bayesian framework that places a prior only on the component of interest, such as the location of the optimum. Nuisance parameters are eliminated via profile likelihood, which naturally handles constraints. As a direct instantiation, we develop a MINimalist Thompson Sampling (MINTS) algorithm. Our framework accommodates structured problems, including continuum-armed Lipschitz bandits and dynamic pricing. It also provides a probabilistic lens on classical convex optimization algorithms such as the center of gravity and ellipsoid methods. We further analyze MINTS for multi-armed bandits and establish near-optimal regret guarantees. 

**Abstract (ZH)**: 基于贝叶斯范式的 minimalist 框架及其在sequential决策中的应用：消除冗余参数以处理约束条件 

---
# The Impact of Artificial Intelligence on Traditional Art Forms: A Disruption or Enhancement 

**Title (ZH)**: 人工智能对传统艺术形式的影响：颠覆还是增强 

**Authors**: Viswa Chaitanya Marella, Sai Teja Erukude, Suhasnadh Reddy Veluru  

**Link**: [PDF](https://arxiv.org/pdf/2509.07029)  

**Abstract**: The introduction of Artificial Intelligence (AI) into the domains of traditional art (visual arts, performing arts, and crafts) has sparked a complicated discussion about whether this might be an agent of disruption or an enhancement of our traditional art forms. This paper looks at the duality of AI, exploring the ways that recent technologies like Generative Adversarial Networks and Diffusion Models, and text-to-image generators are changing the fields of painting, sculpture, calligraphy, dance, music, and the arts of craft. Using examples and data, we illustrate the ways that AI can democratize creative expression, improve productivity, and preserve cultural heritage, while also examining the negative aspects, including: the threats to authenticity within art, ethical concerns around data, and issues including socio-economic factors such as job losses. While we argue for the context-dependence of the impact of AI (the potential for creative homogenization and the devaluation of human agency in artmaking), we also illustrate the potential for hybrid practices featuring AI in cuisine, etc. We advocate for the development of ethical guidelines, collaborative approaches, and inclusive technology development. In sum, we are articulating a vision of AI in which it amplifies our innate creativity while resisting the displacement of the cultural, nuanced, and emotional aspects of traditional art. The future will be determined by human choices about how to govern AI so that it becomes a mechanism for artistic evolution and not a substitute for the artist's soul. 

**Abstract (ZH)**: 人工智能在传统艺术（视觉艺术、表演艺术和手工艺）领域中的引入引发了关于其可能是一种破坏性因素还是传统艺术形式的增强因素的复杂讨论。本文探讨了人工智能的双重性，分析了生成对抗网络、扩散模型和文本转图像生成器等先进技术如何改变绘画、雕塑、书法、舞蹈、音乐和手工艺等领域。通过例证和数据，我们展示了人工智能如何促进创造性表达、提高生产效率并保存文化遗产，同时也审视了其负面影响，包括艺术中的真实性威胁、数据伦理问题以及包括社会经济因素在内的职业流失问题。虽然我们强调人工智能影响的上下文依赖性（创造性同质化的潜在风险和艺术创作中人类代理价值的贬值），但我们也展示了人工智能与传统工艺等领域的混合实践的潜力。我们倡导制定伦理准则、合作方法和包容性技术开发。总之，我们阐述了一个愿景，即人工智能能够放大我们内在的创造力，同时避免取代传统艺术的文化化、细腻化和情感化方面。人类将在如何治理人工智能以使其成为艺术进化机制而非艺术家灵魂的替代品方面作出选择。 

---
# Contradictions 

**Title (ZH)**: 矛盾 

**Authors**: Yang Xu, Shuwei Chen, Xiaomei Zhong, Jun Liu, Xingxing He  

**Link**: [PDF](https://arxiv.org/pdf/2509.07026)  

**Abstract**: Trustworthy AI requires reasoning systems that are not only powerful but also transparent and reliable. Automated Theorem Proving (ATP) is central to formal reasoning, yet classical binary resolution remains limited, as each step involves only two clauses and eliminates at most two literals. To overcome this bottleneck, the concept of standard contradiction and the theory of contradiction-separation-based deduction were introduced in 2018. This paper advances that framework by focusing on the systematic construction of standard contradictions. Specially, this study investigates construction methods for two principal forms of standard contradiction: the maximum triangular standard contradiction and the triangular-type standard contradiction. Building on these structures, we propose a procedure for determining the satisfiability and unsatisfiability of clause sets via maximum standard contradiction. Furthermore, we derive formulas for computing the number of standard sub-contradictions embedded within both the maximum triangular standard contradiction and the triangular-type standard contradiction. The results presented herein furnish the methodological basis for advancing contradiction-separation-based dynamic multi-clause automated deduction, thereby extending the expressive and deductive capabilities of automated reasoning systems beyond the classical binary paradigm. 

**Abstract (ZH)**: 可信的人工智能需要既强大又透明可靠的推理系统。形式推理的核心是自动定理证明（ATP），然而经典的二元归结仍然受限，因为每步仅涉及两个子句并最多消除两个原子命题。为克服这一瓶颈，标准矛盾的概念和基于矛盾分离的推理理论在2018年被提出。本文在此基础上聚焦于标准矛盾的系统构造。具体而言，本研究探讨了两种主要标准矛盾形式的构造方法：最大三角形标准矛盾和三角型标准矛盾。在这些结构的基础上，我们提出了一种通过最大标准矛盾确定子句集的可满足性和不可满足性的程序。此外，我们推导了计算嵌入于最大三角形标准矛盾和三角型标准矛盾中的标准次矛盾数量的公式。本文的结果为扩展基于矛盾分离的动态多子句自动推理的表达能力和演绎能力提供了方法论基础，从而超越了经典的二元范式。 

---
# MEGS$^{2}$: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning 

**Title (ZH)**: MEGS$^{2}$: 基于球面高斯和统一剪支的内存高效高斯点积算法 

**Authors**: Jiarui Chen, Yikeng Chen, Yingshuang Zou, Ye Huang, Peng Wang, Yuan Liu, Yujing Sun, Wenping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07021)  

**Abstract**: 3D Gaussian Splatting (3DGS) has emerged as a dominant novel-view synthesis technique, but its high memory consumption severely limits its applicability on edge devices. A growing number of 3DGS compression methods have been proposed to make 3DGS more efficient, yet most only focus on storage compression and fail to address the critical bottleneck of rendering memory. To address this problem, we introduce MEGS$^{2}$, a novel memory-efficient framework that tackles this challenge by jointly optimizing two key factors: the total primitive number and the parameters per primitive, achieving unprecedented memory compression. Specifically, we replace the memory-intensive spherical harmonics with lightweight arbitrarily-oriented spherical Gaussian lobes as our color representations. More importantly, we propose a unified soft pruning framework that models primitive-number and lobe-number pruning as a single constrained optimization problem. Experiments show that MEGS$^{2}$ achieves a 50% static VRAM reduction and a 40% rendering VRAM reduction compared to existing methods, while maintaining comparable rendering quality. 

**Abstract (ZH)**: 3DGS的内存高效框架MEGS$^{2}$：联合优化关键因素实现前所未有的内存压缩 

---
# Random Forest Stratified K-Fold Cross Validation on SYN DoS Attack SD-IoV 

**Title (ZH)**: 随机森林分层K折交叉验证在SYN DoS攻击SD-IoV上的应用 

**Authors**: Muhammad Arif Hakimi Zamrai, Kamaludin Mohd Yusof  

**Link**: [PDF](https://arxiv.org/pdf/2509.07016)  

**Abstract**: In response to the prevalent concern of TCP SYN flood attacks within the context of Software-Defined Internet of Vehicles (SD-IoV), this study addresses the significant challenge of network security in rapidly evolving vehicular communication systems. This research focuses on optimizing a Random Forest Classifier model to achieve maximum accuracy and minimal detection time, thereby enhancing vehicular network security. The methodology involves preprocessing a dataset containing SYN attack instances, employing feature scaling and label encoding techniques, and applying Stratified K-Fold cross-validation to target key metrics such as accuracy, precision, recall, and F1-score. This research achieved an average value of 0.999998 for all metrics with a SYN DoS attack detection time of 0.24 seconds. Results show that the fine-tuned Random Forest model, configured with 20 estimators and a depth of 10, effectively differentiates between normal and malicious traffic with high accuracy and minimal detection time, which is crucial for SD-IoV networks. This approach marks a significant advancement and introduces a state-of-the-art algorithm in detecting SYN flood attacks, combining high accuracy with minimal detection time. It contributes to vehicular network security by providing a robust solution against TCP SYN flood attacks while maintaining network efficiency and reliability. 

**Abstract (ZH)**: 针对软件定义车联网（SD-IoV）环境下普遍存在的TCP SYN洪泛攻击 Concern帖子 

---
# Human-in-the-Loop: Quantitative Evaluation of 3D Models Generation by Large Language Models 

**Title (ZH)**: 人力介入循环: 大型语言模型生成3D模型的质量评估 

**Authors**: Ahmed R. Sadik, Mariusz Bujny  

**Link**: [PDF](https://arxiv.org/pdf/2509.07010)  

**Abstract**: Large Language Models are increasingly capable of interpreting multimodal inputs to generate complex 3D shapes, yet robust methods to evaluate geometric and structural fidelity remain underdeveloped. This paper introduces a human in the loop framework for the quantitative evaluation of LLM generated 3D models, supporting applications such as democratization of CAD design, reverse engineering of legacy designs, and rapid prototyping. We propose a comprehensive suite of similarity and complexity metrics, including volumetric accuracy, surface alignment, dimensional fidelity, and topological intricacy, to benchmark generated models against ground truth CAD references. Using an L bracket component as a case study, we systematically compare LLM performance across four input modalities: 2D orthographic views, isometric sketches, geometric structure trees, and code based correction prompts. Our findings demonstrate improved generation fidelity with increased semantic richness, with code level prompts achieving perfect reconstruction across all metrics. A key contribution of this work is demonstrating that our proposed quantitative evaluation approach enables significantly faster convergence toward the ground truth, especially compared to traditional qualitative methods based solely on visual inspection and human intuition. This work not only advances the understanding of AI assisted shape synthesis but also provides a scalable methodology to validate and refine generative models for diverse CAD applications. 

**Abstract (ZH)**: 大语言模型越来越能够解析多模态输入以生成复杂的三维形状，但几何和结构保真度的鲁棒评估方法仍处于开发初期。本文介绍了一种人工在环的框架，用于定量评估LLM生成的三维模型，支持CAD设计民主化、遗产设计逆向工程以及快速原型制作等应用。我们提出了一套全面的相似性和复杂性度量标准，包括体素准确性、表面对齐、尺寸保真度和拓扑复杂性，以将生成的模型与真实CAD参考进行基准测试。以L形支架组件为例，我们系统地比较了LLM在四种输入模态下的性能：2D正交视图、等轴测草图、几何结构树以及基于代码的校正提示。我们的研究结果表明，随着语义丰富度的提高，生成的保真度有所提升，代码级别的提示实现了所有度量标准下的完美重建。本文的一项重要贡献是证明了我们提出的定量评估方法可以显著加快向真实参考值的收敛速度，尤其是在与仅基于视觉检查和人类直觉的传统定性方法相比时。这项工作不仅促进了AI辅助形状合成的理解，还提供了一种可扩展的方法来验证和细化针对各种CAD应用的生成模型。 

---
# Computational Concept of the Psyche 

**Title (ZH)**: 心理计算概念 

**Authors**: Anton Kolonin, Vladimir Kryukov  

**Link**: [PDF](https://arxiv.org/pdf/2509.07009)  

**Abstract**: The article provides an overview of approaches to modeling the human psyche in the perspective of building an artificial one. Based on the review, a concept of cognitive architecture is proposed, where the psyche is considered as an operating system of a living or artificial subject, including a space of needs that determines its life meanings in connection with stimuli from the external world, and intelligence as a decision-making system for actions in relation to this world in order to satisfy these needs. Based on the concept, a computational formalization is proposed for creating artificial intelligence systems through learning from experience in the space of a space of needs, taking into account their biological or existential significance for an intelligent agent. Thus, the problem of building general artificial intelligence as a system for making optimal decisions in the space of agent-specific needs under conditions of uncertainty is formalized, with maximization of success in achieving goals, minimization of existential risks and maximization of energy efficiency. A minimal experimental implementation of the model is also provided. 

**Abstract (ZH)**: 文章提供了从构建人工 psyche 视角建模人类心理的方法概述。基于这一回顾，提出了一个认知架构的概念，将 psyche 视为生物或人工主体的操作系统，包括需要的空间，这决定了其生命意义与外部世界的刺激相连，以及作为与这个世界互动以满足这些需要的决策系统的智能。基于这一概念，提出了通过在需要空间中学习经验来创建人工智能系统的计算形式化方法，同时考虑其对智能代理的生物学或存在意义。因此，构建通用人工智能作为在不确定条件下针对特定代理需求空间做出最优决策的系统问题被形式化，追求目标实现的成功最大化、存在风险最小化以及能量效率最大化。还提供了一个最小规模的模型实验实施。 

---
# Not All Splits Are Equal: Rethinking Attribute Generalization Across Unrelated Categories 

**Title (ZH)**: 非所有分割皆平等：重新思考不相关类别间的属性泛化 

**Authors**: Liviu Nicolae Fircă, Antonio Bărbălau, Dan Oneata, Elena Burceanu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06998)  

**Abstract**: Can models generalize attribute knowledge across semantically and perceptually dissimilar categories? While prior work has addressed attribute prediction within narrow taxonomic or visually similar domains, it remains unclear whether current models can abstract attributes and apply them to conceptually distant categories. This work presents the first explicit evaluation for the robustness of the attribute prediction task under such conditions, testing whether models can correctly infer shared attributes between unrelated object types: e.g., identifying that the attribute "has four legs" is common to both "dogs" and "chairs". To enable this evaluation, we introduce train-test split strategies that progressively reduce correlation between training and test sets, based on: LLM-driven semantic grouping, embedding similarity thresholding, embedding-based clustering, and supercategory-based partitioning using ground-truth labels. Results show a sharp drop in performance as the correlation between training and test categories decreases, indicating strong sensitivity to split design. Among the evaluated methods, clustering yields the most effective trade-off, reducing hidden correlations while preserving learnability. These findings offer new insights into the limitations of current representations and inform future benchmark construction for attribute reasoning. 

**Abstract (ZH)**: 能否在语义和感知差异较大的类别之间泛化属性知识？尽管先前的工作已经解决了在狭窄分类学范围内或视觉上相似领域内的属性预测问题，但对于当前模型是否能够抽象出属性并将其应用到概念上相距较远的类别中仍不清楚。本研究首次对在这些条件下属性预测任务的鲁棒性进行了显式的评估，测试模型是否能够正确推断不相关对象类型之间的共享属性：例如，识别“有四条腿”这一属性同时存在于“狗”和“椅子”这两种物体中。为了实现这一评估，我们引入了基于以下策略的训练-测试划分方案：由LLM驱动的语义分组、嵌入相似性阈值化、基于嵌入的聚类以及基于超级类别的分隔，使用真实标签。结果显示，随着训练和测试类别之间的相关性降低，性能出现急剧下降，表明对划分设计高度敏感。在评估的方法中，聚类方法提供了最有效的权衡，既能减少隐藏的相关性又能保持可学习性。这些发现为当前表示的局限性提供了新的见解，并为未来属性推理基准的构建提供了指导。 

---
# The Protocol Genome A Self Supervised Learning Framework from DICOM Headers 

**Title (ZH)**: DICOM头自监督学习框架：协议基因 

**Authors**: Jimmy Joseph  

**Link**: [PDF](https://arxiv.org/pdf/2509.06995)  

**Abstract**: In this paper, we introduce the Protocol Genome, a self-supervised learning system that learns correlations from DICOM headers and achieves AUROC 0.901 (vs 0.847 baseline) and ECE 0.036 (vs 0.058) on fully held-out external validation. Our method also improves calibration and robustness across modalities (CT, MRI, CXR) and vendors. Clinical imaging is funneled through PACS/DICOM, where procedure choices (scanner make/model, sequence, kernel, kVp, TR/TE, and slice thickness) have consequences for contrast, noise, and artifact. These latent confounders impede the generalization of image-only networks across sites. We consider structured DICOM headers as a label and learn protocol-aware but clinically robust image representations. Protocol Genome obtains tokenized embeddings of de-identified header fields and models them along with image features using: (1) protocol-image contrastive learning, (2) masked protocol prediction, and (3) protocol-protocol translation. With 1.26M studies (7 health systems, 31 scanners, 3 vendors; CT, MR, CR/DR), we experiment on: (A) chest CT triage for PE, (B) brain MRI glioma grading, and (C) chest radiograph cardiomegaly detection. Relative to strong SSL baselines (SimCLR, MAE) as well as ImageNet transfer, Protocol Genome (+0.046: PE, +0.058: glioma, +0.041: cardiomegaly) is associated with higher external AUROC; 25-37% calibration improvements are obtained (p < 0.01, DeLong tests). While the gains may be task-dependent, they are preserved with 10-20% of labeled data. From a clinical point of view, the technique reduces false positives at protocol borders and is applicable in a PACS (DICOM C-FIND/C-MOVE, DICOMweb QIDO/WADO). We publish a model card and deployment guide, complete with both de-identification and bias audits. 

**Abstract (ZH)**: 基于协议的基因组：一种自监督学习系统及其在临床影像中的应用 

---
# Frustratingly Easy Feature Reconstruction for Out-of-Distribution Detection 

**Title (ZH)**: frustringly简单的特点重建用于异常分布检测 

**Authors**: Yingsheng Wang, Shuo Lu, Jian Liang, Aihua Zheng, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2509.06988)  

**Abstract**: Out-of-distribution (OOD) detection helps models identify data outside the training categories, crucial for security applications. While feature-based post-hoc methods address this by evaluating data differences in the feature space without changing network parameters, they often require access to training data, which may not be suitable for some data privacy scenarios. This may not be suitable in scenarios where data privacy protection is a concern. In this paper, we propose a simple yet effective post-hoc method, termed Classifier-based Feature Reconstruction (ClaFR), from the perspective of subspace projection. It first performs an orthogonal decomposition of the classifier's weights to extract the class-known subspace, then maps the original data features into this subspace to obtain new data representations. Subsequently, the OOD score is determined by calculating the feature reconstruction error of the data within the subspace. Compared to existing OOD detection algorithms, our method does not require access to training data while achieving leading performance on multiple OOD benchmarks. Our code is released at this https URL. 

**Abstract (ZH)**: 离分布（OOD）检测有助于模型识别训练类别之外的数据，对于安全应用至关重要。特征基于的后 hoc 方法通过在特征空间中评估数据差异来实现这一目标，而不更改网络参数，但通常需要访问训练数据，这在某些数据隐私场景中可能不合适。在数据隐私保护是关注点的情况下，这可能不太合适。本文从子空间投影的角度提出了一种简单而有效的后 hoc 方法，称为基于分类器的特征重建（ClaFR）。该方法首先对手分类器的权重进行正交分解以提取已知类别子空间，然后将原始数据特征映射到该子空间以获得新的数据表示。随后，通过计算子空间内数据的特征重建误差来确定OOD得分。与现有OOD检测算法相比，我们的方法无需访问训练数据，在多个OOD基准上实现了领先地位。代码已发布在该网址：此 https URL。 

---
# CellPainTR: Generalizable Representation Learning for Cross-Dataset Cell Painting Analysis 

**Title (ZH)**: CellPainTR: 跨数据集细胞绘图分析的泛化表示学习 

**Authors**: Cedric Caruzzo, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.06986)  

**Abstract**: Large-scale biological discovery requires integrating massive, heterogeneous datasets like those from the JUMP Cell Painting consortium, but technical batch effects and a lack of generalizable models remain critical roadblocks. To address this, we introduce CellPainTR, a Transformer-based architecture designed to learn foundational representations of cellular morphology that are robust to batch effects. Unlike traditional methods that require retraining on new data, CellPainTR's design, featuring source-specific context tokens, allows for effective out-of-distribution (OOD) generalization to entirely unseen datasets without fine-tuning. We validate CellPainTR on the large-scale JUMP dataset, where it outperforms established methods like ComBat and Harmony in both batch integration and biological signal preservation. Critically, we demonstrate its robustness through a challenging OOD task on the unseen Bray et al. dataset, where it maintains high performance despite significant domain and feature shifts. Our work represents a significant step towards creating truly foundational models for image-based profiling, enabling more reliable and scalable cross-study biological analysis. 

**Abstract (ZH)**: 大规模生物发现需要整合如JUMP Cell Painting联盟那样的大规模异质数据集，但技术和批次效应以及缺乏泛化模型仍然是关键障碍。为此，我们引入了CellPainTR，这是一种基于变换器的架构，设计用于学习对批次效应具有鲁棒性的细胞形态基础表示。与需要在新数据上重新训练的传统方法不同，CellPainTR 通过特定来源的上下文标记设计，能够在无需微调的情况下有效地泛化到完全未见过的数据集。我们在大规模的JUMP数据集上验证了CellPainTR，结果显示它在批次整合和生物信号保留方面优于现有的方法如ComBat和Harmony。更重要的是，我们在Bray等人数据集的具有挑战性的未见分布任务中展示了其鲁棒性，即使存在显著的领域和特征转移，它仍能保持高性能。我们的工作代表了朝着为基于图像的表型分析创建真正基础模型的重要一步，将促进更可靠和可扩展的跨研究生物分析。 

---
# Exploring Over-stationarization in Deep Learning-based Bus/Tram Arrival Time Prediction: Analysis and Non-stationary Effect Recovery 

**Title (ZH)**: 基于深度学习的公交车/有轨电车到站时间预测中的过时变异性探究：分析与非时变效应恢复 

**Authors**: Zirui Li, Bin Yang, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06979)  

**Abstract**: Arrival time prediction (ATP) of public transport vehicles is essential in improving passenger experience and supporting traffic management. Deep learning has demonstrated outstanding performance in ATP due to its ability to model non-linear and temporal dynamics. In the multi-step ATP, non-stationary data will degrade the model performance due to the variation in variables' joint distribution along the temporal direction. Previous studies mainly applied normalization to eliminate the non-stationarity in time series, thereby achieving better predictability. However, the normalization may obscure useful characteristics inherent in non-stationarity, which is known as the over-stationarization. In this work, to trade off predictability and non-stationarity, a new approach for multi-step ATP, named non-stationary ATP ( NSATP), is proposed. The method consists of two stages: series stationarization and non-stationarity effect recovery. The first stage aims at improving the predictability. As for the latter, NSATP extends a state-of-the-art method from one-dimensional to two dimensional based models to capture the hidden periodicity in time series and designs a compensation module of over-stationarization by learning scaling and shifting factors from raw data. 125 days' public transport operational data of Dresden is collected for validation. Experimental results show that compared to baseline methods, the proposed NSATP can reduce RMSE, MAE, and MAPE by 2.37%, 1.22%, and 2.26% for trams and by 1.72%, 0.60%, and 1.17% for buses, respectively. 

**Abstract (ZH)**: 公共运输车辆到达时间预测（ATP）对于提升乘客体验和支持交通管理至关重要。非线性和时序动态模型的能力使得深度学习在ATP中表现出色。在多步ATP中，非平稳数据会由于时序上变量联合分布的变化而降低模型性能。先前的研究主要通过归一化来消除时间序列中的非平稳性，从而提高预测能力。然而，归一化可能会掩盖非平稳性中固有的有用特征，这被称为过度平稳化。在此项工作中，为权衡预测能力和非平稳性，提出了一种新的多步ATP方法，称为非平稳ATP（NSATP）。该方法包含两个阶段：序列平稳化和非平稳性效应恢复。第一阶段旨在提高预测能力。对于后者，NSATP将最先进的方法从一维扩展到二维模型，以捕获时间序列中的隐藏周期性，并通过学习原始数据的缩放和位移因子来设计一个过度平稳化的补偿模块。用于验证的数据包括125天德累斯顿的公共运输运营数据。实验结果表明，与基线方法相比，提出的NSATP分别将有轨电车的RMSE、MAE和MAPE降低了2.37%、1.22%和2.26%，将公交车的RMSE、MAE和MAPE分别降低了1.72%、0.60%和1.17%。 

---
# Toward Reproducible Cross-Backend Compatibility for Deep Learning: A Configuration-First Framework with Three-Tier Verification 

**Title (ZH)**: 向深度学习跨后端兼容性可重复性迈进：一种以配置为主导的三层验证框架 

**Authors**: Zehua Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.06977)  

**Abstract**: This paper presents a configuration-first framework for evaluating cross-backend compatibility in deep learning systems deployed on CPU, GPU, and compiled runtimes. The framework decouples experiments from code using YAML, supports both library and repository models, and employs a three-tier verification protocol covering tensor-level closeness, activation alignment, and task-level metrics. Through 672 checks across multiple models and tolerance settings, we observe that 72.0% of runs pass, with most discrepancies occurring under stricter thresholds. Our results show that detection models and compiled backends are particularly prone to drift, often due to nondeterministic post-processing. We further demonstrate that deterministic adapters and selective fallbacks can substantially improve agreement without significant performance loss. To our knowledge, this is the first unified framework that systematically quantifies and mitigates cross-backend drift in deep learning, providing a reproducible methodology for dependable deployment across heterogeneous runtimes. 

**Abstract (ZH)**: 本论文提出了一种配置优先框架，用于评估部署在CPU、GPU和编译运行时环境中的深度学习系统之间的后端兼容性。该框架使用YAML解耦实验与代码，支持库模型和仓库模型，并采用三层验证协议，涵盖张量级接近度、激活对齐以及任务级指标。通过在多个模型和容差设置下进行672次检查，我们观察到72.0%的运行通过，大多数差异主要出现在更严格的阈值下。我们的结果显示，检测模型和编译后端特别容易发生漂移，常由于非确定性的后处理引起。此外，我们还证明，确定性适配器和选择性回退可以在不显著影响性能的情况下显著提高一致性。据我们所知，这是首个系统地量化和缓解深度学习中跨后端漂移的统一框架，为异构运行时环境下的可靠部署提供了一种可重复的方法。 

---
# GSTBench: A Benchmark Study on the Transferability of Graph Self-Supervised Learning 

**Title (ZH)**: GSTBench：图自监督学习转移性的一项基准研究 

**Authors**: Yu Song, Zhigang Hua, Yan Xie, Jingzhe Liu, Bo Long, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06975)  

**Abstract**: Self-supervised learning (SSL) has shown great promise in graph representation learning. However, most existing graph SSL methods are developed and evaluated under a single-dataset setting, leaving their cross-dataset transferability largely unexplored and limiting their ability to leverage knowledge transfer and large-scale pretraining, factors that are critical for developing generalized intelligence beyond fitting training data. To address this gap and advance foundation model research for graphs, we present GSTBench, the first systematic benchmark for evaluating the transferability of graph SSL methods. We conduct large-scale pretraining on ogbn-papers100M and evaluate five representative SSL methods across a diverse set of target graphs. Our standardized experimental setup decouples confounding factors such as model architecture, dataset characteristics, and adaptation protocols, enabling rigorous comparisons focused solely on pretraining objectives. Surprisingly, we observe that most graph SSL methods struggle to generalize, with some performing worse than random initialization. In contrast, GraphMAE, a masked autoencoder approach, consistently improves transfer performance. We analyze the underlying factors that drive these differences and offer insights to guide future research on transferable graph SSL, laying a solid foundation for the "pretrain-then-transfer" paradigm in graph learning. Our code is available at this https URL. 

**Abstract (ZH)**: 自监督学习（SSL）在图表示学习中展现出了巨大的潜力。然而，现有的大多数图SSL方法仅在单数据集设置下开发和评估，使得它们跨数据集的迁移能力很大程度上未被探索，限制了它们利用知识迁移和大规模预训练等关键因素的能力，这些因素对于开发超越训练数据拟合的通用智能至关重要。为填补这一空白并推进图的础模型研究，我们提出了GSTBench，首个系统性的图SSL方法迁移能力评估基准。我们在ogbn-papers100M上进行大规模预训练，并在多种目标图上评估五种代表性的SSL方法。我们的标准化实验设置将模型架构、数据集特性以及适应协议等混杂因素分离，使比较集中在预训练目标上。令人惊讶的是，我们发现大多数图SSL方法难以泛化，有些甚至表现逊于随机初始化。相比之下，掩码自编码器方法GraphMAE始终能提升迁移性能。我们分析了这些差异背后的驱动因素，并提供了指导未来可迁移图SSL研究的见解，为图学习中的“预训练-然后迁移”范式奠定了坚实基础。我们的代码可通过以下链接获取。 

---
# Individualized and Interpretable Sleep Forecasting via a Two-Stage Adaptive Spatial-Temporal Model 

**Title (ZH)**: 基于两阶段自适应空时模型的个性化可解释睡眠预测 

**Authors**: Xueyi Wang, Elisabeth Wilhelm  

**Link**: [PDF](https://arxiv.org/pdf/2509.06974)  

**Abstract**: Sleep quality significantly impacts well-being. Therefore, healthcare providers and individuals need accessible and reliable forecasting tools for preventive interventions. This paper introduces an interpretable, individualized two-stage adaptive spatial-temporal model for predicting sleep quality scores. Our proposed framework combines multi-scale convolutional layers to model spatial interactions across multiple input variables, recurrent layers and attention mechanisms to capture long-term temporal dependencies, and a two-stage domain adaptation strategy to enhance generalization. The first adaptation stage is applied during training to mitigate overfitting on the training set. In the second stage, a source-free test-time adaptation mechanism is employed to adapt the model to new users without requiring labels. We conducted various experiments with five input window sizes (3, 5, 7, 9, and 11 days) and five prediction window sizes (1, 3, 5, 7, and 9 days). Our model consistently outperformed time series forecasting baseline approaches, including Long Short-Term Memory (LSTM), Informer, PatchTST, and TimesNet. The best performance was achieved with a three-day input window and a one-day prediction window, yielding a root mean square error (RMSE) of 0.216. Furthermore, the model demonstrated good predictive performance even for longer forecasting horizons (e.g, with a 0.257 RMSE for a three-day prediction window), highlighting its practical utility for real-world applications. We also conducted an explainability analysis to examine how different features influence sleep quality. These findings proved that the proposed framework offers a robust, adaptive, and explainable solution for personalized sleep forecasting using sparse data from commercial wearable devices. 

**Abstract (ZH)**: 睡眠质量显著影响福祉。因此，医护人员和个体需要获得可访问且可靠的预测工具以进行预防干预。本文提出了一个可解释的、个性化两阶段自适应时空模型以预测睡眠质量评分。本提出的框架结合了多尺度卷积层来建模多个输入变量的时空交互，循环层和注意力机制来捕捉长期时间依赖性，以及两阶段领域适应策略以增强泛化能力。第一阶段适应在训练过程中应用于减轻对训练集的过拟合。在第二阶段，采用无源测试时自适应机制来适应新用户，无需标签。我们使用五种不同的输入窗口大小（3天、5天、7天、9天和11天）和五种不同的预测窗口大小（1天、3天、5天、7天和9天）进行了各种实验。我们的模型始终优于时间序列预测基准方法，包括长短期记忆（LSTM）、Informer、PatchTST 和 TimesNet。最佳性能是在三天的输入窗口和一天的预测窗口下实现的，得到的均方根误差（RMSE）为0.216。此外，该模型在更远的预测时长内也表现出良好的预测性能（例如，三天预测窗口的RMSE为0.257），突显了其在实际应用中的实用性。我们还进行了可解释性分析以考察不同特征如何影响睡眠质量。这些发现证明了所提出的框架能够提供一个稳健、适应性强且解释明确的个性化睡眠预测解决方案，使用的是来自商用可穿戴设备的稀疏数据。 

---
# Impact of Neuron Models on Spiking Neural Networks performance. A Complexity Based Classification Approach 

**Title (ZH)**: 基于复杂性分类方法的神经元模型对脉冲神经网络性能的影响研究 

**Authors**: Zofia Rudnicka, Janusz Szczepanski, Agnieszka Pregowska  

**Link**: [PDF](https://arxiv.org/pdf/2509.06970)  

**Abstract**: This study explores how the selection of neuron models and learning rules impacts the classification performance of Spiking Neural Networks (SNNs), with a focus on applications in bio-signal processing. We compare biologically inspired neuron models, including Leaky Integrate-and-Fire (LIF), metaneurons, and probabilistic Levy-Baxter (LB) neurons, across multiple learning rules, including spike-timing-dependent plasticity (STDP), tempotron, and reward-modulated updates. A novel element of this work is the integration of a complexity-based decision mechanism into the evaluation pipeline. Using Lempel-Ziv Complexity (LZC), a measure related to entropy rate, we quantify the structural regularity of spike trains and assess classification outcomes in a consistent and interpretable manner across different SNN configurations. To investigate neural dynamics and assess algorithm performance, we employed synthetic datasets with varying temporal dependencies and stochasticity levels. These included Markov and Poisson processes, well-established models to simulate neuronal spike trains and capture the stochastic firing behavior of biological this http URL of synthetic Poisson and Markov-modeled data reveals clear performance trends: classification accuracy depends on the interaction between neuron model, network size, and learning rule, with the LZC-based evaluation highlighting configurations that remain robust to weak or noisy signals. This work delivers a systematic analysis of how neuron model selection interacts with network parameters and learning strategies, supported by a novel complexity-based evaluation approach that offers a consistent benchmark for SNN performance. 

**Abstract (ZH)**: 本研究探讨了神经元模型选择和学习规则对脉冲神经网络（SNN）分类性能的影响，重点关注生物信号处理应用。我们比较了受生物学启发的神经元模型，包括耗尽积分-发放（LIF）、超神经元和概率莱维-巴克斯特（LB）神经元，以及多种学习规则，包括时序依赖可塑性（STDP）、Tempotron和奖励调节更新。这项工作的创新之处在于将基于复杂性的决策机制整合到评估管道中。通过使用Lempel-Ziv复杂度（LZC），一种与熵率相关的度量，我们量化了尖峰序列的结构规律，并采用一致且可解释的方式评估不同SNN配置的分类结果。为了研究神经动态并评估算法性能，我们使用了具有不同时间依赖性和随机性水平的合成数据集。这些数据集包括马尔可夫过程和泊松过程，是模拟神经元尖峰序列和捕捉生物神经元随机放电行为的经典模型。基于合成泊松和马尔可夫建模数据的研究表明，分类准确性取决于神经元模型、网络规模和学习规则之间的相互作用，而基于LZC的评估突出了对弱或噪声信号具有鲁棒性的配置。本研究通过一种新颖的基于复杂性的评估方法，系统分析了神经元模型选择与网络参数及学习策略的相互作用，并提供了一致的SNN性能基准。 

---
# Association of Timing and Duration of Moderate-to-Vigorous Physical Activity with Cognitive Function and Brain Aging: A Population-Based Study Using the UK Biobank 

**Title (ZH)**: 中等至剧烈体力活动的时间和持续时间与认知功能和大脑老化关系的队列研究：基于英国生物银行数据 

**Authors**: Wasif Khan, Lin Gu, Noah Hammarlund, Lei Xing, Joshua K. Wong, Ruogu Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06969)  

**Abstract**: Physical activity is a modifiable lifestyle factor with potential to support cognitive resilience. However, the association of moderate-to-vigorous physical activity (MVPA) intensity, and timing, with cognitive function and region-specific brain structure remain poorly understood. We analyzed data from 45,892 UK Biobank participants aged 60 years and older with valid wrist-worn accelerometer data, cognitive testing, and structural brain MRI. MVPA was measured both continuously (mins per week) and categorically (thresholded using >=150 min/week based on WHO guidelines). Associations with cognitive performance and regional brain volumes were evaluated using multivariable linear models adjusted for demographic, socioeconomic, and health-related covariates. We conducted secondary analyses on MVPA timing and subgroup effects. Higher MVPA was associated with better performance across cognitive domains, including reasoning, memory, executive function, and processing speed. These associations persisted in fully adjusted models and were higher among participants meeting WHO guidelines. Greater MVPA was also associated with subcortical brain regions (caudate, putamen, pallidum, thalamus), as well as regional gray matter volumes involved in emotion, working memory, and perceptual processing. Secondary analyses showed that MVPA at any time of day was associated with cognitive functions and brain volume particularly in the midday-afternoon and evening. Sensitivity analysis shows consistent findings across subgroups, with evidence of dose-response relationships. Higher MVPA is associated with preserved brain structure and enhanced cognitive function in later life. Public health strategies to increase MVPA may support healthy cognitive aging and generate substantial economic benefits, with global gains projected to reach USD 760 billion annually by 2050. 

**Abstract (ZH)**: 中等至剧烈强度的体力活动与认知功能及区域特异性脑结构之间的关系尚不明确。我们分析了45,892名英国生物银行参与者（年龄60岁及以上）的有效腕戴式加速度计数据、认知测试和结构脑MRI数据。中等至剧烈强度的体力活动（MVPA）被连续测量（每周分钟数）和分类测量（基于WHO指南定义的≥150分钟/周）。通过多变量线性模型评估其与认知表现和区域性脑体积的关联，调整了人口统计学、社会经济和健康相关的混杂因素。我们还进行了MVPA时间的相关分析和亚组效应分析。较高水平的MVPA在各个认知领域均与更好的表现相关，包括推理、记忆、执行功能和处理速度。这些关联在完全调整的模型中仍然存在，并且在达到WHO指南标准的参与者中更为显著。较高水平的MVPA还与基底节（壳核、苍白球、丘脑）以及与情绪、工作记忆和感知处理相关的皮层下区域和皮质灰质体积相关。次要分析显示，无论是白天还是晚上的MVPA都与认知功能和脑体积在午后和傍晚时分特别相关。敏感性分析结果显示，这些发现跨亚组一致，并显示出剂量-反应关系。较高水平的MVPA在晚年与保持的脑结构和增强的认知功能相关。增加MVPA的公共卫生策略可能支持健康的认知老化并产生重大的经济利益，全球收益预计到2050年将达到每年7600亿美元。 

---
# Deep Learning-based Techniques for Integrated Sensing and Communication Systems: State-of-the-Art, Challenges, and Opportunities 

**Title (ZH)**: 基于深度学习的综合传感与通信系统技术：现状、挑战与机遇 

**Authors**: Murat Temiz, Yongwei Zhang, Yanwei Fu, Chi Zhang, Chenfeng Meng, Orhan Kaplan, Christos Masouros  

**Link**: [PDF](https://arxiv.org/pdf/2509.06968)  

**Abstract**: This article comprehensively reviews recent developments and research on deep learning-based (DL-based) techniques for integrated sensing and communication (ISAC) systems. ISAC, which combines sensing and communication functionalities, is regarded as a key enabler for 6G and beyond networks, as many emerging applications, such as vehicular networks and industrial robotics, necessitate both sensing and communication capabilities for effective operation. A unified platform that provides both functions can reduce hardware complexity, alleviate frequency spectrum congestion, and improve energy efficiency. However, integrating these functionalities on the same hardware requires highly optimized signal processing and system design, introducing significant computational complexity when relying on conventional iterative or optimization-based techniques. As an alternative to conventional techniques, DL-based techniques offer efficient and near-optimal solutions with reduced computational complexity. Hence, such techniques are well-suited for operating under limited computational resources and low latency requirements in real-time systems. DL-based techniques can swiftly and effectively yield near-optimal solutions for a wide range of sophisticated ISAC-related tasks, including waveform design, channel estimation, sensing signal processing, data demodulation, and interference mitigation. Therefore, motivated by these advantages, recent studies have proposed various DL-based approaches for ISAC system design. After briefly introducing DL architectures and ISAC fundamentals, this survey presents a comprehensive and categorized review of state-of-the-art DL-based techniques for ISAC, highlights their key advantages and major challenges, and outlines potential directions for future research. 

**Abstract (ZH)**: 基于深度学习的集成传感与通信系统研究综述 

---
# Cross-field SNR Analysis and Tensor Channel Estimation for Multi-UAV Near-field Communications 

**Title (ZH)**: 跨领域信噪比分析和张量信道估计在多无人机近场通信中 

**Authors**: Tianyu Huo, Jian Xiong, Yiyan Wu, Songjie Yang, Bo Liu, Wenjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06967)  

**Abstract**: Extremely large antenna array (ELAA) is key to enhancing spectral efficiency in 6G networks. Leveraging the distributed nature of multi-unmanned aerial vehicle (UAV) systems enables the formation of distributed ELAA, which often operate in the near-field region with spatial sparsity, rendering the conventional far-field plane wave assumption invalid. This paper investigates channel estimation for distributed near-field multi-UAV communication systems. We first derive closed-form signal-to-noise ratio (SNR) expressions under the plane wave model (PWM), spherical wave model (SWM), and a hybrid spherical-plane wave model (HSPWM), also referred to as the cross-field model, within a distributed uniform planar array (UPA) scenario. The analysis shows that HSPWM achieves a good balance between modeling accuracy and analytical tractability. Based on this, we propose two channel estimation algorithms: the spherical-domain orthogonal matching pursuit (SD-OMP) and the tensor-OMP. The SD-OMP generalizes the polar domain to jointly consider elevation, azimuth, and range. Under the HSPWM, the channel is naturally formulated as a tensor, enabling the use of tensor-OMP. Simulation results demonstrate that tensor-OMP achieves normalized mean square error (NMSE) performance comparable to SD-OMP, while offering reduced computational complexity and improved scalability. 

**Abstract (ZH)**: 分布式近场多无人机通信系统的信道估计研究 

---
# Cross-device Zero-shot Label Transfer via Alignment of Time Series Foundation Model Embeddings 

**Title (ZH)**: 跨设备零样本标签转移：时间序列基础模型嵌入的对齐 

**Authors**: Neal G. Ravindra, Arijit Sehanobish  

**Link**: [PDF](https://arxiv.org/pdf/2509.06966)  

**Abstract**: High-quality, medically validated labels exist for clinical actigraphy data but not for ubiquitous consumer wearables like the Apple Watch. Manually labeling wearables data is expensive and doesn't scale. This paper offers a novel framework that transfers valuable labels from a source domain (e.g., actigraphy) to a target domain (e.g., Apple Watch) without requiring paired data. Instead of working with raw time-series signals, we project both domains into a shared latent embedding space using time-series foundation models (TSFMs) and develop a new framework to align the cross-device representations. Our method, Adversarial Alignment of TSFM Embeddings forces the distributions of source and target embeddings to align within this space, facilitating label transfer across device type. 

**Abstract (ZH)**: 高质量、医学验证的标签存在于临床加速度计数据中，但不存在于如Apple Watch等普遍的消费者穿戴设备中。手动标记穿戴设备数据既昂贵又不具扩展性。本文提出了一种新颖的框架，能够在不需要配对数据的情况下，将有价值的数据标签从源领域（例如，加速度计）转移到目标领域（例如，Apple Watch）。我们不是直接处理原始的时间序列信号，而是使用时间序列基础模型（TSFMs）将两个领域投影到一个共享的潜在嵌入空间，并开发了一种新的框架以对齐设备间的表示。我们的方法，TSFM嵌入的对抗对齐，强制源和目标嵌入的分布在这个空间内对齐，从而促进不同设备类型之间的标签转移。 

---
