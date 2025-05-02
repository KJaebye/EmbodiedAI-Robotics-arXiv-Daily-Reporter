# Open-Source LLM-Driven Federated Transformer for Predictive IoV Management 

**Title (ZH)**: 开源LLM驱动的联邦变换器预测IoV管理 

**Authors**: Yazan Otoum, Arghavan Asad, Ishtiaq Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2505.00651)  

**Abstract**: The proliferation of connected vehicles within the Internet of Vehicles (IoV) ecosystem presents critical challenges in ensuring scalable, real-time, and privacy-preserving traffic management. Existing centralized IoV solutions often suffer from high latency, limited scalability, and reliance on proprietary Artificial Intelligence (AI) models, creating significant barriers to widespread deployment, particularly in dynamic and privacy-sensitive environments. Meanwhile, integrating Large Language Models (LLMs) in vehicular systems remains underexplored, especially concerning prompt optimization and effective utilization in federated contexts. To address these challenges, we propose the Federated Prompt-Optimized Traffic Transformer (FPoTT), a novel framework that leverages open-source LLMs for predictive IoV management. FPoTT introduces a dynamic prompt optimization mechanism that iteratively refines textual prompts to enhance trajectory prediction. The architecture employs a dual-layer federated learning paradigm, combining lightweight edge models for real-time inference with cloud-based LLMs to retain global intelligence. A Transformer-driven synthetic data generator is incorporated to augment training with diverse, high-fidelity traffic scenarios in the Next Generation Simulation (NGSIM) format. Extensive evaluations demonstrate that FPoTT, utilizing EleutherAI Pythia-1B, achieves 99.86% prediction accuracy on real-world data while maintaining high performance on synthetic datasets. These results underscore the potential of open-source LLMs in enabling secure, adaptive, and scalable IoV management, offering a promising alternative to proprietary solutions in smart mobility ecosystems. 

**Abstract (ZH)**: 联邦提示优化交通变换器：面向物联网汽车的开放式提示优化预测框架 

---
# Position: AI Competitions Provide the Gold Standard for Empirical Rigor in GenAI Evaluation 

**Title (ZH)**: 位置：AI竞赛提供了生成式AI评估的黄金标准 empirical rigor。 

**Authors**: D. Sculley, Will Cukierski, Phil Culliton, Sohier Dane, Maggie Demkin, Ryan Holbrook, Addison Howard, Paul Mooney, Walter Reade, Megan Risdal, Nate Keating  

**Link**: [PDF](https://arxiv.org/pdf/2505.00612)  

**Abstract**: In this position paper, we observe that empirical evaluation in Generative AI is at a crisis point since traditional ML evaluation and benchmarking strategies are insufficient to meet the needs of evaluating modern GenAI models and systems. There are many reasons for this, including the fact that these models typically have nearly unbounded input and output spaces, typically do not have a well defined ground truth target, and typically exhibit strong feedback loops and prediction dependence based on context of previous model outputs. On top of these critical issues, we argue that the problems of {\em leakage} and {\em contamination} are in fact the most important and difficult issues to address for GenAI evaluations. Interestingly, the field of AI Competitions has developed effective measures and practices to combat leakage for the purpose of counteracting cheating by bad actors within a competition setting. This makes AI Competitions an especially valuable (but underutilized) resource. Now is time for the field to view AI Competitions as the gold standard for empirical rigor in GenAI evaluation, and to harness and harvest their results with according value. 

**Abstract (ZH)**: 基于生成型AI的实证评估处于危机点：泄漏和污染问题的挑战与机遇 

---
# Combining LLMs with Logic-Based Framework to Explain MCTS 

**Title (ZH)**: 将逻辑推理框架与大语言模型结合以解释MCTS 

**Authors**: Ziyan An, Xia Wang, Hendrik Baier, Zirong Chen, Abhishek Dubey, Taylor T. Johnson, Jonathan Sprinkle, Ayan Mukhopadhyay, Meiyi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.00610)  

**Abstract**: In response to the lack of trust in Artificial Intelligence (AI) for sequential planning, we design a Computational Tree Logic-guided large language model (LLM)-based natural language explanation framework designed for the Monte Carlo Tree Search (MCTS) algorithm. MCTS is often considered challenging to interpret due to the complexity of its search trees, but our framework is flexible enough to handle a wide range of free-form post-hoc queries and knowledge-based inquiries centered around MCTS and the Markov Decision Process (MDP) of the application domain. By transforming user queries into logic and variable statements, our framework ensures that the evidence obtained from the search tree remains factually consistent with the underlying environmental dynamics and any constraints in the actual stochastic control process. We evaluate the framework rigorously through quantitative assessments, where it demonstrates strong performance in terms of accuracy and factual consistency. 

**Abstract (ZH)**: 针对人工智能（AI）在序列规划中的信任缺失，我们设计了一个由计算树逻辑引导的大语言模型（LLM）为基础的自然语言解释框架，该框架适用于蒙特卡洛树搜索（MCTS）算法。虽然MCTS由于其搜索树的复杂性而常常难以解释，但我们的框架足够灵活，能够处理与MCTS和应用领域马尔可夫决策过程（MDP）相关的广泛自由格式的后验查询和基于知识的询问。通过将用户查询转换为逻辑和变量声明，我们的框架确保从搜索树中获得的证据与基础环境动态及其实际随机控制过程中的任何约束保持事实上的连贯性。我们通过严格的定量评估对该框架进行了评估，结果显示其在准确性和事实一致性方面表现优异。 

---
# Can LLMs Help Improve Analogical Reasoning For Strategic Decisions? Experimental Evidence from Humans and GPT-4 

**Title (ZH)**: 大规模语言模型能否帮助提高战略决策中的类比推理能力？来自人类和GPT-4的实验证据 

**Authors**: Phanish Puranam, Prothit Sen, Maciej Workiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2505.00603)  

**Abstract**: This study investigates whether large language models, specifically GPT4, can match human capabilities in analogical reasoning within strategic decision making contexts. Using a novel experimental design involving source to target matching, we find that GPT4 achieves high recall by retrieving all plausible analogies but suffers from low precision, frequently applying incorrect analogies based on superficial similarities. In contrast, human participants exhibit high precision but low recall, selecting fewer analogies yet with stronger causal alignment. These findings advance theory by identifying matching, the evaluative phase of analogical reasoning, as a distinct step that requires accurate causal mapping beyond simple retrieval. While current LLMs are proficient in generating candidate analogies, humans maintain a comparative advantage in recognizing deep structural similarities across domains. Error analysis reveals that AI errors arise from surface level matching, whereas human errors stem from misinterpretations of causal structure. Taken together, the results suggest a productive division of labor in AI assisted organizational decision making where LLMs may serve as broad analogy generators, while humans act as critical evaluators, applying the most contextually appropriate analogies to strategic problems. 

**Abstract (ZH)**: 本研究探讨了大型语言模型，特别是GPT4，在战略决策情境中的类比推理能力是否能与人类相媲美。通过一种新颖的实验设计——源到目标匹配，我们发现GPT4能够通过检索所有合理类比而实现高召回率，但精度较低，经常基于表层相似性应用不正确的类比。相比之下，人类参与者表现出高精度但低召回率，选择较少的类比但因果对齐更强。这些发现通过识别类比推理中的匹配阶段作为需要超越简单检索的准确因果映射的独立步骤，推进了理论发展。尽管当前的大型语言模型在生成候选类比方面表现 proficient，人类在跨领域识别深层结构相似性方面依然具有比较优势。错误分析表明，AI错误源于表层匹配，而人类错误源于对因果结构的误解。综合来看，这些结果表明，在人工智能辅助组织决策中，大型语言模型可能充当广泛的类比生成者，而人类则作为关键的评估者，应用最合适的类比来解决战略性问题。 

---
# Rule-based Classifier Models 

**Title (ZH)**: 基于规则的分类器模型 

**Authors**: Cecilia Di Florio, Huimin Dong, Antonino Rotolo  

**Link**: [PDF](https://arxiv.org/pdf/2505.00474)  

**Abstract**: We extend the formal framework of classifier models used in the legal domain. While the existing classifier framework characterises cases solely through the facts involved, legal reasoning fundamentally relies on both facts and rules, particularly the ratio decidendi. This paper presents an initial approach to incorporating sets of rules within a classifier. Our work is built on the work of Canavotto et al. (2023), which has developed the rule-based reason model of precedential constraint within a hierarchy of factors. We demonstrate how decisions for new cases can be inferred using this enriched rule-based classifier framework. Additionally, we provide an example of how the time element and the hierarchy of courts can be used in the new classifier framework. 

**Abstract (ZH)**: 我们扩展了在法律领域使用的分类模型的形式框架。现有的分类框架仅通过案件事实来表征案例，而法律推理本质上依赖于事实和规则，特别是判例中的判决理由。本文提出了一种初步方法，在分类模型中纳入规则集。我们的工作基于Canavotto等人（2023）的工作，该工作在因素层次结构中发展了基于规则的判例约束推理模型。我们展示了如何使用这种增强的基于规则的分类框架来推断新案件的判决。此外，我们提供了如何在新分类框架中使用时间元素和法院层级结构的例子。 

---
# UserCentrix: An Agentic Memory-augmented AI Framework for Smart Spaces 

**Title (ZH)**: UserCentrix: 一个赋能的记忆增强AI框架用于智能空间 

**Authors**: Alaa Saleh, Sasu Tarkoma, Praveen Kumar Donta, Naser Hossein Motlagh, Schahram Dustdar, Susanna Pirttikangas, Lauri Lovén  

**Link**: [PDF](https://arxiv.org/pdf/2505.00472)  

**Abstract**: Agentic AI, with its autonomous and proactive decision-making, has transformed smart environments. By integrating Generative AI (GenAI) and multi-agent systems, modern AI frameworks can dynamically adapt to user preferences, optimize data management, and improve resource allocation. This paper introduces UserCentrix, an agentic memory-augmented AI framework designed to enhance smart spaces through dynamic, context-aware decision-making. This framework integrates personalized Large Language Model (LLM) agents that leverage user preferences and LLM memory management to deliver proactive and adaptive assistance. Furthermore, it incorporates a hybrid hierarchical control system, balancing centralized and distributed processing to optimize real-time responsiveness while maintaining global situational awareness. UserCentrix achieves resource-efficient AI interactions by embedding memory-augmented reasoning, cooperative agent negotiation, and adaptive orchestration strategies. Our key contributions include (i) a self-organizing framework with proactive scaling based on task urgency, (ii) a Value of Information (VoI)-driven decision-making process, (iii) a meta-reasoning personal LLM agent, and (iv) an intelligent multi-agent coordination system for seamless environment adaptation. Experimental results across various models confirm the effectiveness of our approach in enhancing response accuracy, system efficiency, and computational resource management in real-world application. 

**Abstract (ZH)**: 具有自主和主动决策能力的代理AI已经转型了智能环境。通过整合生成AI（GenAI）和多智能体系统，现代AI框架可以动态适应用户偏好，优化数据管理，并改进资源配置。本文介绍了一种名为UserCentrix的代理记忆增强AI框架，该框架旨在通过动态、情境感知的决策来增强智能空间。该框架整合了利用用户偏好和大型语言模型（LLM）记忆管理的个性化LLM代理，以提供主动且适应性的协助。此外，它还纳入了混合层次控制系统，平衡集中式和分布式处理，以优化实时响应能力，同时保持全局情况意识。UserCentrix通过嵌入记忆增强推理、协作智能体协商和适应性编排策略实现了高效的AI交互。我们的主要贡献包括：(i) 一个基于任务紧迫性自动扩展的自我组织框架，(ii) 一种基于信息价值（VoI）的决策过程，(iii) 一个元推理个性化LLM代理，以及(iv) 一种智能多智能体协调系统，以实现无缝的环境适应。在各种模型上的实验结果证实了该方法在增强响应准确性、系统效率和计算资源管理方面的有效性。 

---
# ScaleTrack: Scaling and back-tracking Automated GUI Agents 

**Title (ZH)**: ScaleTrack: 自动化GUI代理的缩放与反向追踪 

**Authors**: Jing Huang, Zhixiong Zeng, Wenkang Han, Yufeng Zhong, Liming Zheng, Shuai Fu, Jingyuan Chen, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.00416)  

**Abstract**: Automated GUI agents aims to facilitate user interaction by automatically performing complex tasks in digital environments, such as web, mobile, desktop devices. It receives textual task instruction and GUI description to generate executable actions (\emph{e.g.}, click) and operation boxes step by step. Training a GUI agent mainly involves grounding and planning stages, in which the GUI grounding focuses on finding the execution coordinates according to the task, while the planning stage aims to predict the next action based on historical actions. However, previous work suffers from the limitations of insufficient training data for GUI grounding, as well as the ignorance of backtracking historical behaviors for GUI planning. To handle the above challenges, we propose ScaleTrack, a training framework by scaling grounding and backtracking planning for automated GUI agents. We carefully collected GUI samples of different synthesis criterions from a wide range of sources, and unified them into the same template for training GUI grounding models. Moreover, we design a novel training strategy that predicts the next action from the current GUI image, while also backtracking the historical actions that led to the GUI image. In this way, ScaleTrack explains the correspondence between GUI images and actions, which effectively describes the evolution rules of the GUI environment. Extensive experimental results demonstrate the effectiveness of ScaleTrack. Data and code will be available at url. 

**Abstract (ZH)**: 自动GUI代理旨在通过在数字环境中（如网络、移动设备、桌面设备）自动执行复杂任务来简化用户交互。它接收文本任务指令和GUI描述，逐步生成可执行动作（例如点击）和操作框。训练GUI代理主要涉及语义接地和规划阶段，其中的GUI语义接地专注于根据任务找到执行坐标，而规划阶段旨在基于历史动作预测下一个动作。然而，先前的工作受限于GUI语义接地不足的训练数据，以及在GUI规划中忽视了回溯历史行为。为了应对上述挑战，我们提出ScaleTrack，一种通过扩展语义接地和回溯规划来训练自动GUI代理的框架。我们从广泛的数据源中精心收集了不同合成标准的GUI样本，并将它们统一到同一个模板以训练GUI语义接地模型。此外，我们设计了一种新的训练策略，从当前的GUI图像预测下一个动作，同时也回溯导致该GUI图像的历史动作。通过这种方式，ScaleTrack解释了GUI图像与动作之间的对应关系，有效地描述了GUI环境的演变规则。广泛的实验结果证明了ScaleTrack的有效性。数据和代码将在此URL上提供。 

---
# Urban Air Mobility as a System of Systems: An LLM-Enhanced Holonic Approach 

**Title (ZH)**: 城市空中移动作为系统_of_系统：基于LLM的holonic方法 

**Authors**: Ahmed R. Sadik, Muhammad Ashfaq, Niko Mäkitalo, Tommi Mikkonen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00368)  

**Abstract**: Urban Air Mobility (UAM) is an emerging System of System (SoS) that faces challenges in system architecture, planning, task management, and execution. Traditional architectural approaches struggle with scalability, adaptability, and seamless resource integration within dynamic and complex environments. This paper presents an intelligent holonic architecture that incorporates Large Language Model (LLM) to manage the complexities of UAM. Holons function semi autonomously, allowing for real time coordination among air taxis, ground transport, and vertiports. LLMs process natural language inputs, generate adaptive plans, and manage disruptions such as weather changes or airspace this http URL a case study of multimodal transportation with electric scooters and air taxis, we demonstrate how this architecture enables dynamic resource allocation, real time replanning, and autonomous adaptation without centralized control, creating more resilient and efficient urban transportation networks. By advancing decentralized control and AI driven adaptability, this work lays the groundwork for resilient, human centric UAM ecosystems, with future efforts targeting hybrid AI integration and real world validation. 

**Abstract (ZH)**: 城市空中交通(UAM)是一种新兴的系统体系结构(SoS)，面临系统架构、规划、任务管理和执行等方面的挑战。传统的架构方法难以应对动态复杂环境中规模性、适应性和无缝资源集成的需求。本文提出了一种智能holonic架构，结合大型语言模型(LLM)来管理UAM的复杂性。holons半自主运行，允许空中出租车、地面运输和 vertiports 之间实时协调。LLM处理自然语言输入，生成适应性计划，并管理天气变化或空域更改等中断。通过一个涉及电动滑板车和空中出租车的多模式交通案例研究，我们展示了该架构如何实现动态资源分配、实时再规划和无需中心控制的自主适应，从而构建更具有韧性和效率的城市交通网络。通过推进去中心化控制和AI驱动的适应性，这项工作为韧性的人本中心UAM生态系统奠定了基础，未来努力将集中在混合AI集成和实地验证上。 

---
# CognitionNet: A Collaborative Neural Network for Play Style Discovery in Online Skill Gaming Platform 

**Title (ZH)**: 认知网络：在线技能游戏平台上玩法风格发现的协作神经网络 

**Authors**: Rukma Talwadker, Surajit Chakrabarty, Aditya Pareek, Tridib Mukherjee, Deepak Saini  

**Link**: [PDF](https://arxiv.org/pdf/2505.00325)  

**Abstract**: Games are one of the safest source of realizing self-esteem and relaxation at the same time. An online gaming platform typically has massive data coming in, e.g., in-game actions, player moves, clickstreams, transactions etc. It is rather interesting, as something as simple as data on gaming moves can help create a psychological imprint of the user at that moment, based on her impulsive reactions and response to a situation in the game. Mining this knowledge can: (a) immediately help better explain observed and predicted player behavior; and (b) consequently propel deeper understanding towards players' experience, growth and protection. To this effect, we focus on discovery of the "game behaviours" as micro-patterns formed by continuous sequence of games and the persistent "play styles" of the players' as a sequence of such sequences on an online skill gaming platform for Rummy. We propose a two stage deep neural network, CognitionNet. The first stage focuses on mining game behaviours as cluster representations in a latent space while the second aggregates over these micro patterns to discover play styles via a supervised classification objective around player engagement. The dual objective allows CognitionNet to reveal several player psychology inspired decision making and tactics. To our knowledge, this is the first and one-of-its-kind research to fully automate the discovery of: (i) player psychology and game tactics from telemetry data; and (ii) relevant diagnostic explanations to players' engagement predictions. The collaborative training of the two networks with differential input dimensions is enabled using a novel formulation of "bridge loss". The network plays pivotal role in obtaining homogeneous and consistent play style definitions and significantly outperforms the SOTA baselines wherever applicable. 

**Abstract (ZH)**: 在线游戏平台中通过深度神经网络发现玩家心理与游戏策略自动化的研究 

---
# DeCo: Defect-Aware Modeling with Contrasting Matching for Optimizing Task Assignment in Online IC Testing 

**Title (ZH)**: DeCo: 基于对比匹配的缺陷意识建模方法以优化在线IC测试任务分配 

**Authors**: Lo Pang-Yun Ting, Yu-Hao Chiang, Yi-Tung Tsai, Hsu-Chao Lai, Kun-Ta Chuang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00278)  

**Abstract**: In the semiconductor industry, integrated circuit (IC) processes play a vital role, as the rising complexity and market expectations necessitate improvements in yield. Identifying IC defects and assigning IC testing tasks to the right engineers improves efficiency and reduces losses. While current studies emphasize fault localization or defect classification, they overlook the integration of defect characteristics, historical failures, and the insights from engineer expertise, which restrains their effectiveness in improving IC handling. To leverage AI for these challenges, we propose DeCo, an innovative approach for optimizing task assignment in IC testing. DeCo constructs a novel defect-aware graph from IC testing reports, capturing co-failure relationships to enhance defect differentiation, even with scarce defect data. Additionally, it formulates defect-aware representations for engineers and tasks, reinforced by local and global structure modeling on the defect-aware graph. Finally, a contrasting-based assignment mechanism pairs testing tasks with QA engineers by considering their skill level and current workload, thus promoting an equitable and efficient job dispatch. Experiments on a real-world dataset demonstrate that DeCo achieves the highest task-handling success rates in different scenarios, exceeding 80\%, while also maintaining balanced workloads on both scarce or expanded defect data. Moreover, case studies reveal that DeCo can assign tasks to potentially capable engineers, even for their unfamiliar defects, highlighting its potential as an AI-driven solution for the real-world IC failure analysis and task handling. 

**Abstract (ZH)**: 半导体行业中，集成电路（IC）工艺发挥着关键作用，随着复杂性的提高和市场预期的增强，提高良率变得尤为重要。通过识别IC缺陷并将IC测试任务分配给合适的工程师可以提高效率并减少损失。当前的研究侧重于故障定位或缺陷分类，但忽视了缺陷特征、历史失效及工程师 expertise 的整合，这限制了它们在改善IC处理方面的有效性。为了应对这些挑战，我们提出了DeCo，这是一种优化IC测试任务分配的新颖方法。DeCo从IC测试报告中构建了一个新颖的缺陷感知图，捕获共失效关系以增强缺陷区分能力，即使在缺陷数据稀缺的情况下也是如此。此外，它为工程师和任务制定了缺陷感知表示，并通过缺陷感知图上的局部和全局结构建模增强了这些表示。最后，通过考虑工程师的技能水平和当前工作负荷来实现对比驱动的任务分配机制，从而促进公平且高效的职责分配。实验结果表明，DeCo在不同场景下的任务处理成功率最高，超过80%，同时在稀缺或扩展的缺陷数据下也能保持均衡的工作负荷。此外，案例研究显示，DeCo能够将任务分配给潜在有能力但不熟悉该缺陷的工程师，突显了其作为AI驱动解决方案在实际IC失效分析和任务处理中的潜力。 

---
# RAIL in the Wild: Operationalizing Responsible AI Evaluation Using Anthropic's Value Dataset 

**Title (ZH)**: RAIL in the Wild: 使用Anthropic的价值数据集实现负责任AI评估的操作化 

**Authors**: Sumit Verma, Pritam Prasun, Arpit Jaiswal, Pritish Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.00204)  

**Abstract**: As AI systems become embedded in real-world applications, ensuring they meet ethical standards is crucial. While existing AI ethics frameworks emphasize fairness, transparency, and accountability, they often lack actionable evaluation methods. This paper introduces a systematic approach using the Responsible AI Labs (RAIL) framework, which includes eight measurable dimensions to assess the normative behavior of large language models (LLMs). We apply this framework to Anthropic's "Values in the Wild" dataset, containing over 308,000 anonymized conversations with Claude and more than 3,000 annotated value expressions. Our study maps these values to RAIL dimensions, computes synthetic scores, and provides insights into the ethical behavior of LLMs in real-world use. 

**Abstract (ZH)**: 随着AI系统嵌入到实际应用中，确保其符合伦理标准变得至关重要。虽然现有的AI伦理框架强调公平、透明和问责制，但它们往往缺乏可操作的评估方法。本文介绍了使用责任AI实验室（RAIL）框架的一种系统方法，该框架包括八个可测量维度以评估大型语言模型（LLMs）的规范行为。我们将这一框架应用于Anthropic的“野外价值观”数据集，该数据集包含超过308,000次匿名与Claude的对话和超过3,000个标注的价值表达。我们的研究将这些价值观映射到RAIL维度，计算合成评分，并提供关于LLMs在实际应用中伦理行为的见解。 

---
# Real-World Gaps in AI Governance Research 

**Title (ZH)**: AI治理研究中的现实差距 

**Authors**: Ilan Strauss, Isobel Moure, Tim O'Reilly, Sruly Rosenblat  

**Link**: [PDF](https://arxiv.org/pdf/2505.00174)  

**Abstract**: Drawing on 1,178 safety and reliability papers from 9,439 generative AI papers (January 2020 - March 2025), we compare research outputs of leading AI companies (Anthropic, Google DeepMind, Meta, Microsoft, and OpenAI) and AI universities (CMU, MIT, NYU, Stanford, UC Berkeley, and University of Washington). We find that corporate AI research increasingly concentrates on pre-deployment areas -- model alignment and testing & evaluation -- while attention to deployment-stage issues such as model bias has waned. Significant research gaps exist in high-risk deployment domains, including healthcare, finance, misinformation, persuasive and addictive features, hallucinations, and copyright. Without improved observability into deployed AI, growing corporate concentration could deepen knowledge deficits. We recommend expanding external researcher access to deployment data and systematic observability of in-market AI behaviors. 

**Abstract (ZH)**: 基于2020年1月至2025年3月的9439篇生成式AI论文中的1178篇安全与可靠性论文，我们比较了Anthropic、Google DeepMind、Meta、Microsoft和OpenAI等领先AI公司以及CMU、MIT、NYU、Stanford、UC Berkeley和University of Washington等AI大学的研究成果。研究发现，企业AI研究越来越集中在部署前领域（如模型对齐和测试与评估），而对部署阶段问题（如模型偏差）的关注度下降。在包括医疗、金融、 misinformation、说服性和成瘾性功能、幻觉和版权在内的高风险部署领域，存在显著的研究缺口。缺乏对部署中AI的增强可观察性可能导致知识差距加剧。我们建议扩大外部研究人员对部署数据的访问权限，并加强对市场中AI行为的系统观察。 

---
# First Order Logic with Fuzzy Semantics for Describing and Recognizing Nerves in Medical Images 

**Title (ZH)**: 具有模糊语义的一阶逻辑描述与识别医疗图像中的神经结构 

**Authors**: Isabelle Bloch, Enzo Bonnot, Pietro Gori, Giammarco La Barbera, Sabine Sarnacki  

**Link**: [PDF](https://arxiv.org/pdf/2505.00173)  

**Abstract**: This article deals with the description and recognition of fiber bundles, in particular nerves, in medical images, based on the anatomical description of the fiber trajectories. To this end, we propose a logical formalization of this anatomical knowledge. The intrinsically imprecise description of nerves, as found in anatomical textbooks, leads us to propose fuzzy semantics combined with first-order logic. We define a language representing spatial entities, relations between these entities and quantifiers. A formula in this language is then a formalization of the natural language description. The semantics are given by fuzzy representations in a concrete domain and satisfaction degrees of relations. Based on this formalization, a spatial reasoning algorithm is proposed for segmentation and recognition of nerves from anatomical and diffusion magnetic resonance images, which is illustrated on pelvic nerves in pediatric imaging, enabling surgeons to plan surgery. 

**Abstract (ZH)**: 基于解剖学轨迹描述的纤维束（特别是神经）在医学图像中的描述与识别：一种层次化逻辑形式化方法及其在儿科盆腔神经影像中的应用 

---
# Position Paper: Towards Open Complex Human-AI Agents Collaboration System for Problem-Solving and Knowledge Management 

**Title (ZH)**: 位置论文：通往开放复杂人机协作系统的道路——面向问题解决和知识管理 

**Authors**: Ju Wu, Calvin K.L. Or  

**Link**: [PDF](https://arxiv.org/pdf/2505.00018)  

**Abstract**: This position paper critically surveys a broad spectrum of recent empirical developments on human-AI agents collaboration, highlighting both their technical achievements and persistent gaps. We observe a lack of a unifying theoretical framework that can coherently integrate these varied studies, especially when tackling open-ended, complex tasks. To address this, we propose a novel conceptual architecture: one that systematically interlinks the technical details of multi-agent coordination, knowledge management, cybernetic feedback loops, and higher-level control mechanisms. By mapping existing contributions, from symbolic AI techniques and connectionist LLM-based agents to hybrid organizational practices, onto this proposed framework (Hierarchical Exploration-Exploitation Net), our approach facilitates revision of legacy methods and inspires new work that fuses qualitative and quantitative paradigms. The paper's structure allows it to be read from any section, serving equally as a critical review of technical implementations and as a forward-looking reference for designing or extending human-AI symbioses. Together, these insights offer a stepping stone toward deeper co-evolution of human cognition and AI capability. 

**Abstract (ZH)**: 这篇立场论文批判性地回顾了近期人类-人工智能代理合作的广泛实证发展，突出了其技术成就和持续存在的空白。我们注意到缺乏一个统一的理论框架来综合这些多样的研究，特别是在处理开放性和复杂性任务时。为此，我们提出了一种新颖的概念架构：一种系统地将多代理协调技术细节、知识管理、控制论反馈回路和高级控制机制相互关联的架构。通过将现有的贡献，从符号人工智能技术到连接主义LLM基代理，再到混合组织实践，映射到这一提议的框架（层次探索-利用网）中，我们的方法促进了对遗留方法的修订，并启发了结合定性和定量范式的新型研究。该论文的结构使其可以从任何部分阅读，既作为对技术实现的批判性回顾，又作为设计或扩展人类-人工智能共生体的前瞻性参考。这些洞察为我们更深一步的人机认知共进化提供了基石。 

---
# T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT 

**Title (ZH)**: T2I-R1：通过协作的语义级和tokens级共推理增强图像生成 

**Authors**: Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.00703)  

**Abstract**: Recent advancements in large language models have demonstrated how chain-of-thought (CoT) and reinforcement learning (RL) can improve performance. However, applying such reasoning strategies to the visual generation domain remains largely unexplored. In this paper, we present T2I-R1, a novel reasoning-enhanced text-to-image generation model, powered by RL with a bi-level CoT reasoning process. Specifically, we identify two levels of CoT that can be utilized to enhance different stages of generation: (1) the semantic-level CoT for high-level planning of the prompt and (2) the token-level CoT for low-level pixel processing during patch-by-patch generation. To better coordinate these two levels of CoT, we introduce BiCoT-GRPO with an ensemble of generation rewards, which seamlessly optimizes both generation CoTs within the same training step. By applying our reasoning strategies to the baseline model, Janus-Pro, we achieve superior performance with 13% improvement on T2I-CompBench and 19% improvement on the WISE benchmark, even surpassing the state-of-the-art model FLUX.1. Code is available at: this https URL 

**Abstract (ZH)**: 近期大规模语言模型的发展展示了链式思考(CoT)和强化学习(RL)如何提升性能。然而，将这些推理策略应用于视觉生成领域仍 largely unexplored。在本文中，我们提出了一种名为 T2I-R1 的新型增强推理文本到图像生成模型，该模型基于具有两层链式思考推理过程的强化学习。具体而言，我们识别了两种可以用于提高生成不同阶段的 CoT：(1) 语义层次的链式思考用于高阶提示规划；(2) 令牌层次的链式思考用于生成过程中的像素级处理。为了更好地协调这两种层次的 CoT，我们引入了结合生成奖励的 BiCoT-GRPO，该方法在同一训练步骤中无缝优化了两种生成 CoT。通过将我们的推理策略应用到基线模型 Janus-Pro 中，我们在 T2I-CompBench 上实现了 13% 的性能提升，在 WISE 基准上实现了 19% 的性能提升，甚至超越了最先进的模型 FLUX。1. 代码可在以下链接获取：this https URL 

---
# Robotic Visual Instruction 

**Title (ZH)**: 机器视觉指令 

**Authors**: Yanbang Li, Ziyang Gong, Haoyang Li, Haoyang Li, Xiaoqi Huang, Haolan Kang, Guangping Bai, Xianzheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.00693)  

**Abstract**: Recently, natural language has been the primary medium for human-robot interaction. However, its inherent lack of spatial precision for robotic control introduces challenges such as ambiguity and verbosity. To address these limitations, we introduce the Robotic Visual Instruction (RoVI), a novel paradigm to guide robotic tasks through an object-centric, hand-drawn symbolic representation. RoVI effectively encodes spatial-temporal information into human-interpretable visual instructions through 2D sketches, utilizing arrows, circles, colors, and numbers to direct 3D robotic manipulation. To enable robots to understand RoVI better and generate precise actions based on RoVI, we present Visual Instruction Embodied Workflow (VIEW), a pipeline formulated for RoVI-conditioned policies. This approach leverages Vision-Language Models (VLMs) to interpret RoVI inputs, decode spatial and temporal constraints from 2D pixel space via keypoint extraction, and then transform them into executable 3D action sequences. We additionally curate a specialized dataset of 15K instances to fine-tune small VLMs for edge deployment, enabling them to effectively learn RoVI capabilities. Our approach is rigorously validated across 11 novel tasks in both real and simulated environments, demonstrating significant generalization capability. Notably, VIEW achieves an 87.5% success rate in real-world scenarios involving unseen tasks that feature multi-step actions, with disturbances, and trajectory-following requirements. Code and Datasets in this paper will be released soon. 

**Abstract (ZH)**: Recent进展：基于视觉指令的机器人任务指导框架 

---
# Towards Autonomous Micromobility through Scalable Urban Simulation 

**Title (ZH)**: 面向自主微出行的可扩展城市仿真研究 

**Authors**: Wayne Wu, Honglin He, Chaoyuan Zhang, Jack He, Seth Z. Zhao, Ran Gong, Quanyi Li, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.00690)  

**Abstract**: Micromobility, which utilizes lightweight mobile machines moving in urban public spaces, such as delivery robots and mobility scooters, emerges as a promising alternative to vehicular mobility. Current micromobility depends mostly on human manual operation (in-person or remote control), which raises safety and efficiency concerns when navigating busy urban environments full of unpredictable obstacles and pedestrians. Assisting humans with AI agents in maneuvering micromobility devices presents a viable solution for enhancing safety and efficiency. In this work, we present a scalable urban simulation solution to advance autonomous micromobility. First, we build URBAN-SIM - a high-performance robot learning platform for large-scale training of embodied agents in interactive urban scenes. URBAN-SIM contains three critical modules: Hierarchical Urban Generation pipeline, Interactive Dynamics Generation strategy, and Asynchronous Scene Sampling scheme, to improve the diversity, realism, and efficiency of robot learning in simulation. Then, we propose URBAN-BENCH - a suite of essential tasks and benchmarks to gauge various capabilities of the AI agents in achieving autonomous micromobility. URBAN-BENCH includes eight tasks based on three core skills of the agents: Urban Locomotion, Urban Navigation, and Urban Traverse. We evaluate four robots with heterogeneous embodiments, such as the wheeled and legged robots, across these tasks. Experiments on diverse terrains and urban structures reveal each robot's strengths and limitations. 

**Abstract (ZH)**: 微移动性：利用轻型移动机器人在城市公共空间中运行的新兴技术，如配送机器人和电动滑板车，已成为车辆移动的有前景的替代方案。当前的微移动性主要依赖于人工手动操作（现场或远程控制），在繁忙且充满不可预测障碍和行人的城市环境中导航时，存在安全性和效率方面的担忧。通过AI代理辅助人类操作微移动性设备，可以有效提高安全性和效率。在本研究中，我们提出了一种可扩展的城市仿真解决方案，以推动自主微移动性的发展。首先，我们构建了URBAN-SIM——一个高性能的机器人学习平台，用于大规模训练交互式城市场景中的具身代理。URBAN-SIM包含三个关键模块：分层城市生成管道、交互动力学生成策略和异步场景抽样方案，以提高机器人在仿真中学习的多样性和真实性，提高效率。然后，我们提出了URBAN-BENCH——一套基础任务和基准测试，以评估AI代理在实现自主微移动性方面的各种能力。URBAN-BENCH包括基于代理三项核心技能的八项任务：城市运动、城市导航和城市穿越。我们在这些任务中评估了四种具有不同体态的机器人，如轮式和腿式机器人。在多样化的地形和城市结构上的实验揭示了每种机器人在能力和局限性方面的不同表现。 

---
# Visual Test-time Scaling for GUI Agent Grounding 

**Title (ZH)**: GUI代理定位的视觉测试时缩放 

**Authors**: Tiange Luo, Lajanugen Logeswaran, Justin Johnson, Honglak Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.00684)  

**Abstract**: We introduce RegionFocus, a visual test-time scaling approach for Vision Language Model Agents. Understanding webpages is challenging due to the visual complexity of GUI images and the large number of interface elements, making accurate action selection difficult. Our approach dynamically zooms in on relevant regions, reducing background clutter and improving grounding accuracy. To support this process, we propose an image-as-map mechanism that visualizes key landmarks at each step, providing a transparent action record and enables the agent to effectively choose among action candidates. Even with a simple region selection strategy, we observe significant performance gains of 28+\% on Screenspot-pro and 24+\% on WebVoyager benchmarks on top of two state-of-the-art open vision language model agents, UI-TARS and Qwen2.5-VL, highlighting the effectiveness of visual test-time scaling in interactive settings. We achieve a new state-of-the-art grounding performance of 61.6\% on the ScreenSpot-Pro benchmark by applying RegionFocus to a Qwen2.5-VL-72B model. Our code will be released publicly at this https URL. 

**Abstract (ZH)**: 我们引入了RegionFocus，一种视觉测试时缩放方法，用于视觉语言模型代理。由于网页中的GUI图像具有视觉复杂性且界面元素众多，理解网页颇具挑战性，准确的动作选择变得困难。我们的方法动态放大相关区域，减少背景杂乱，提高语义匹配准确性。为支持这一过程，我们提出了一种图像即地图机制，在每一步可视化关键地标，提供透明的动作记录，并使代理能够有效地在动作候选中进行选择。即使采用简单的区域选择策略，我们也在Screenspot-pro和WebVoyager基准上分别观察到UI-TARS和Qwen2.5-VL两个最先进的开放视觉语言模型代理28%+和24%+的显著性能提升，突显了在交互环境中视觉测试时缩放的有效性。通过将RegionFocus应用于Qwen2.5-VL-72B模型，我们在ScreenSpot-Pro基准上实现了新的最佳语义匹配性能61.6%。我们的代码将在以下链接公开发布：https://this-url。 

---
# Deep Reinforcement Learning for Urban Air Quality Management: Multi-Objective Optimization of Pollution Mitigation Booth Placement in Metropolitan Environments 

**Title (ZH)**: 城市空气质量管理中的深度 reinforcement 学习：大规模城市环境中污染削减装置布局的多目标优化 

**Authors**: Kirtan Rajesh, Suvidha Rupesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.00668)  

**Abstract**: Urban air pollution remains a pressing global concern, particularly in densely populated and traffic-intensive metropolitan areas like Delhi, where exposure to harmful pollutants severely impacts public health. Delhi, being one of the most polluted cities globally, experiences chronic air quality issues due to vehicular emissions, industrial activities, and construction dust, which exacerbate its already fragile atmospheric conditions. Traditional pollution mitigation strategies, such as static air purifying installations, often fail to maximize their impact due to suboptimal placement and limited adaptability to dynamic urban environments. This study presents a novel deep reinforcement learning (DRL) framework to optimize the placement of air purification booths to improve the air quality index (AQI) in the city of Delhi. We employ Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning algorithm, to iteratively learn and identify high-impact locations based on multiple spatial and environmental factors, including population density, traffic patterns, industrial influence, and green space constraints. Our approach is benchmarked against conventional placement strategies, including random and greedy AQI-based methods, using multi-dimensional performance evaluation metrics such as AQI improvement, spatial coverage, population and traffic impact, and spatial entropy. Experimental results demonstrate that the RL-based approach outperforms baseline methods by achieving a balanced and effective distribution of air purification infrastructure. Notably, the DRL framework achieves an optimal trade-off between AQI reduction and high-coverage deployment, ensuring equitable environmental benefits across urban regions. The findings underscore the potential of AI-driven spatial optimization in advancing smart city initiatives and data-driven urban air quality management. 

**Abstract (ZH)**: 基于深度强化学习的城市空气污染优化治理框架：以Delhi为例 

---
# Wasserstein Policy Optimization 

**Title (ZH)**: Wasserstein 政策优化 

**Authors**: David Pfau, Ian Davies, Diana Borsa, Joao G. M. Araujo, Brendan Tracey, Hado van Hasselt  

**Link**: [PDF](https://arxiv.org/pdf/2505.00663)  

**Abstract**: We introduce Wasserstein Policy Optimization (WPO), an actor-critic algorithm for reinforcement learning in continuous action spaces. WPO can be derived as an approximation to Wasserstein gradient flow over the space of all policies projected into a finite-dimensional parameter space (e.g., the weights of a neural network), leading to a simple and completely general closed-form update. The resulting algorithm combines many properties of deterministic and classic policy gradient methods. Like deterministic policy gradients, it exploits knowledge of the gradient of the action-value function with respect to the action. Like classic policy gradients, it can be applied to stochastic policies with arbitrary distributions over actions -- without using the reparameterization trick. We show results on the DeepMind Control Suite and a magnetic confinement fusion task which compare favorably with state-of-the-art continuous control methods. 

**Abstract (ZH)**: Wasserstein策略优化：连续动作空间中的演员-评论家算法 

---
# DeepCritic: Deliberate Critique with Large Language Models 

**Title (ZH)**: DeepCritic: 详尽批判与大规模语言模型 

**Authors**: Wenkai Yang, Jingwen Chen, Yankai Lin, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00662)  

**Abstract**: As Large Language Models (LLMs) are rapidly evolving, providing accurate feedback and scalable oversight on their outputs becomes an urgent and critical problem. Leveraging LLMs as critique models to achieve automated supervision is a promising solution. In this work, we focus on studying and enhancing the math critique ability of LLMs. Current LLM critics provide critiques that are too shallow and superficial on each step, leading to low judgment accuracy and struggling to offer sufficient feedback for the LLM generator to correct mistakes. To tackle this issue, we propose a novel and effective two-stage framework to develop LLM critics that are capable of deliberately critiquing on each reasoning step of math solutions. In the first stage, we utilize Qwen2.5-72B-Instruct to generate 4.5K long-form critiques as seed data for supervised fine-tuning. Each seed critique consists of deliberate step-wise critiques that includes multi-perspective verifications as well as in-depth critiques of initial critiques for each reasoning step. Then, we perform reinforcement learning on the fine-tuned model with either existing human-labeled data from PRM800K or our automatically annotated data obtained via Monte Carlo sampling-based correctness estimation, to further incentivize its critique ability. Our developed critique model built on Qwen2.5-7B-Instruct not only significantly outperforms existing LLM critics (including the same-sized DeepSeek-R1-distill models and GPT-4o) on various error identification benchmarks, but also more effectively helps the LLM generator refine erroneous steps through more detailed feedback. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的迅速演进，提供准确反馈和可扩展监督其输出成为一个迫切且关键的问题。利用LLMs作为批判模型以实现自动化监督是一种有前途的解决方案。在本文中，我们专注于研究和增强LLMs的数学批判能力。当前的LLM批判性评价过于浅显，导致判断准确性低，并且难以为LLM生成器提供足够的反馈以纠正错误。为解决这一问题，我们提出了一种新颖且有效的两阶段框架，以开发能够在数学解题每个推理步骤上进行刻意批判的LLM批判性评价模型。在第一阶段，我们利用Qwen2.5-72B-Instruct生成4.5K长格式批判性评价作为监督微调的种子数据。每个种子批判性评价包括多角度验证以及对每个推理步骤初始批判性评价的深入批判。然后，我们通过强化学习对微调后的模型进行进一步训练，使用现有的从PRM800K中获得的人工标注数据或通过基于蒙特卡洛采样方法获得的自动标注数据（用于正确性估计），以进一步激励其批判性评价能力。基于Qwen2.5-7B-Instruct开发的批判性评价模型不仅在各种错误识别基准上显著优于现有的LLM批判性评价模型（包括同规模的DeepSeek-R1-distill模型和GPT-4o），还能更有效地通过更详细的反馈帮助LLM生成器改进错误步骤。 

---
# On the generalization of language models from in-context learning and finetuning: a controlled study 

**Title (ZH)**: 基于上下文学习和微调的语言模型泛化能力的研究：一个受控实验 

**Authors**: Andrew K. Lampinen, Arslan Chaudhry, Stephanie C.Y. Chan, Cody Wild, Diane Wan, Alex Ku, Jörg Bornschein, Razvan Pascanu, Murray Shanahan, James L. McClelland  

**Link**: [PDF](https://arxiv.org/pdf/2505.00661)  

**Abstract**: Large language models exhibit exciting capabilities, yet can show surprisingly narrow generalization from finetuning -- from failing to generalize to simple reversals of relations they are trained on, to missing logical deductions that can be made from trained information. These failures to generalize from fine-tuning can hinder practical application of these models. However, language models' in-context learning shows different inductive biases, and can generalize better in some of these cases. Here, we explore these differences in generalization between in-context- and fine-tuning-based learning. To do so, we constructed several novel datasets to evaluate and improve models' ability to generalize from finetuning data. The datasets are constructed to isolate the knowledge in the dataset from that in pretraining, to create clean tests of generalization. We expose pretrained large models to controlled subsets of the information in these datasets -- either in context, or through fine-tuning -- and evaluate their performance on test sets that require various types of generalization. We find overall that in data-matched settings, in-context learning can generalize more flexibly than fine-tuning (though we also find some qualifications of prior findings, such as cases when fine-tuning can generalize to reversals embedded in a larger structure of knowledge). We build on these findings to propose a method to enable improved generalization from fine-tuning: adding in-context inferences to finetuning data. We show that this method improves generalization across various splits of our datasets and other benchmarks. Our results have implications for understanding the inductive biases of different modes of learning in language models, and practically improving their performance. 

**Abstract (ZH)**: 大型语言模型展示了令人兴奋的能力，但在微调后却表现出惊人的狭窄泛化能力——从无法泛化到简单的关系反转，到忽略从训练信息中可以得出的逻辑推理。这些微调后的泛化失败可能阻碍这些模型的实际应用。然而，语言模型的上下文学习显示出不同的归纳偏见，并能在某些情况下更好地泛化。为此，我们构建了多个新的数据集，以评估和改进模型从微调数据中泛化的能力。这些数据集的构建旨在将数据集中的知识与预训练中的知识隔离，以创建干净的泛化测试。我们对预训练的大模型进行控制性的信息暴露，要么在上下文中，要么通过微调，并在需要各种类型泛化的测试集上评估其性能。我们发现，在数据匹配设置中，上下文学习可以比微调更灵活地泛化（尽管我们还发现了一些先前发现的限制性条件，例如微调可以在更广泛的结构中泛化到关系反转的情况）。我们基于这些发现提出了一种方法，以改善从微调中泛化的性能：将上下文推断添加到微调数据中。我们展示了这种方法在我们的数据集和其他基准测试的各种拆分上都提高了泛化能力。我们的研究结果对理解语言模型不同学习模式的归纳偏见具有重要意义，并实际提高了它们的性能。 

---
# Large Language Models Understanding: an Inherent Ambiguity Barrier 

**Title (ZH)**: 大型语言模型理解：固有的歧义障碍 

**Authors**: Daniel N. Nissani  

**Link**: [PDF](https://arxiv.org/pdf/2505.00654)  

**Abstract**: A lively ongoing debate is taking place, since the extraordinary emergence of Large Language Models (LLMs) with regards to their capability to understand the world and capture the meaning of the dialogues in which they are involved. Arguments and counter-arguments have been proposed based upon thought experiments, anecdotal conversations between LLMs and humans, statistical linguistic analysis, philosophical considerations, and more. In this brief paper we present a counter-argument based upon a thought experiment and semi-formal considerations leading to an inherent ambiguity barrier which prevents LLMs from having any understanding of what their amazingly fluent dialogues mean. 

**Abstract (ZH)**: 一场生动的持续争论正在进行，关于大型语言模型（LLMs）的能力，即它们理解世界并捕捉参与对话含义的能力。基于思辨实验、LLMs与人类的 anecdotal 对话、统计语言学分析、哲学考量等提出的论点和反论点已经提出。在本文中，我们基于一个思辨实验和半形式化的考量提出一个反论点，指出这种固有的歧义障碍阻止LLMs理解其惊人流畅的对话意味着什么。 

---
# OmicsCL: Unsupervised Contrastive Learning for Cancer Subtype Discovery and Survival Stratification 

**Title (ZH)**: OmicsCL：无监督对比学习在癌症亚型发现和生存分层中的应用 

**Authors**: Atahan Karagoz  

**Link**: [PDF](https://arxiv.org/pdf/2505.00650)  

**Abstract**: Unsupervised learning of disease subtypes from multi-omics data presents a significant opportunity for advancing personalized medicine. We introduce OmicsCL, a modular contrastive learning framework that jointly embeds heterogeneous omics modalities-such as gene expression, DNA methylation, and miRNA expression-into a unified latent space. Our method incorporates a survival-aware contrastive loss that encourages the model to learn representations aligned with survival-related patterns, without relying on labeled outcomes. Evaluated on the TCGA BRCA dataset, OmicsCL uncovers clinically meaningful clusters and achieves strong unsupervised concordance with patient survival. The framework demonstrates robustness across hyperparameter configurations and can be tuned to prioritize either subtype coherence or survival stratification. Ablation studies confirm that integrating survival-aware loss significantly enhances the predictive power of learned embeddings. These results highlight the promise of contrastive objectives for biological insight discovery in high-dimensional, heterogeneous omics data. 

**Abstract (ZH)**: 无监督学习多组学数据中的疾病亚型具有推进个性化医学的重大潜力：OmicsCL，一个联合嵌入多组学模态的模块化对比学习框架 

---
# Deep Learning Assisted Outer Volume Removal for Highly-Accelerated Real-Time Dynamic MRI 

**Title (ZH)**: 深度学习辅助快速实时动态MRI外边缘体积去除 

**Authors**: Merve Gülle, Sebastian Weingärtner, Mehmet Akçakaya  

**Link**: [PDF](https://arxiv.org/pdf/2505.00643)  

**Abstract**: Real-time (RT) dynamic MRI plays a vital role in capturing rapid physiological processes, offering unique insights into organ motion and function. Among these applications, RT cine MRI is particularly important for functional assessment of the heart with high temporal resolution. RT imaging enables free-breathing, ungated imaging of cardiac motion, making it a crucial alternative for patients who cannot tolerate conventional breath-hold, ECG-gated acquisitions. However, achieving high acceleration rates in RT cine MRI is challenging due to aliasing artifacts from extra-cardiac tissues, particularly at high undersampling factors. In this study, we propose a novel outer volume removal (OVR) method to address this challenge by eliminating aliasing contributions from non-cardiac regions in a post-processing framework. Our approach estimates the outer volume signal for each timeframe using composite temporal images from time-interleaved undersampling patterns, which inherently contain pseudo-periodic ghosting artifacts. A deep learning (DL) model is trained to identify and remove these artifacts, producing a clean outer volume estimate that is subsequently subtracted from the corresponding k-space data. The final reconstruction is performed with a physics-driven DL (PD-DL) method trained using an OVR-specific loss function to restore high spatio-temporal resolution images. Experimental results show that the proposed method at high accelerations achieves image quality that is visually comparable to clinical baseline images, while outperforming conventional reconstruction techniques, both qualitatively and quantitatively. The proposed approach provides a practical and effective solution for artifact reduction in RT cine MRI without requiring acquisition modifications, offering a pathway to higher acceleration rates while preserving diagnostic quality. 

**Abstract (ZH)**: 实时（RT）动态MRI在捕捉快速生理过程方面发挥着重要作用，提供了对器官运动和功能的独特见解。在这些应用中，RT cine MRI特别重要，因为它可以提供高质量的时间分辨率，用于心脏的功能评估。RT成像允许自由呼吸、非门控的心脏运动成像，使其成为不能耐受常规屏气、心电门控采集的患者的重要替代方法。然而，由于额外心脏组织的伪影，在高欠采样因子下实现高加速率在RT cine MRI中具有挑战性。在这项研究中，我们提出了一种新颖的外部体积去除（OVR）方法，通过消除非心脏区域的伪影贡献来解决这一挑战，该方法在后处理框架中实现。我们的方法使用时间交错欠采样模式中的复合时间图像来估计每个时间帧的外部体积信号，这些图像本质上包含伪周期性鬼影伪影。一个深度学习（DL）模型被训练来识别并去除这些伪影，产生一个干净的外部体积估计，并随后从相应的k空间数据中减去。最终重建使用特定于OVR的损失函数进行的物理驱动深度学习（PD-DL）方法来进行，以恢复高空间-时间分辨率图像。实验结果表明，所提出的方法在高加速率下实现了视觉上与临床基线图像相当的图像质量，并在定性和定量上均优于传统重建技术。所提出的方法提供了一种实用且有效的解决方案，可以在无需更改采集的情况下减少RT cine MRI中的伪影，为提高加速率并保持诊断质量提供了途径。 

---
# The Illusion of Role Separation: Hidden Shortcuts in LLM Role Learning (and How to Fix Them) 

**Title (ZH)**: 角色分离的错觉：LLM角色学习中的隐藏捷径及其修复方法 

**Authors**: Zihao Wang, Yibo Jiang, Jiahao Yu, Heqing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00626)  

**Abstract**: Large language models (LLMs) that integrate multiple input roles (e.g., system instructions, user queries, external tool outputs) are increasingly prevalent in practice. Ensuring that the model accurately distinguishes messages from each role -- a concept we call \emph{role separation} -- is crucial for consistent multi-role behavior. Although recent work often targets state-of-the-art prompt injection defenses, it remains unclear whether such methods truly teach LLMs to differentiate roles or merely memorize known triggers. In this paper, we examine \emph{role-separation learning}: the process of teaching LLMs to robustly distinguish system and user tokens. Through a \emph{simple, controlled experimental framework}, we find that fine-tuned models often rely on two proxies for role identification: (1) task type exploitation, and (2) proximity to begin-of-text. Although data augmentation can partially mitigate these shortcuts, it generally leads to iterative patching rather than a deeper fix. To address this, we propose reinforcing \emph{invariant signals} that mark role boundaries by adjusting token-wise cues in the model's input encoding. In particular, manipulating position IDs helps the model learn clearer distinctions and reduces reliance on superficial proxies. By focusing on this mechanism-centered perspective, our work illuminates how LLMs can more reliably maintain consistent multi-role behavior without merely memorizing known prompts or triggers. 

**Abstract (ZH)**: 大型语言模型（LLMs）整合了多种输入角色（如系统指令、用户查询、外部工具输出）的比例日益增加。确保模型准确区分每个角色的信息——我们称之为“角色分离”——对于一致的多角色行为至关重要。尽管近期研究往往针对最先进的提示注入防御方法，但尚不清楚这些方法是否真正教会LLMs区分角色，还是仅仅记忆已知触发器。在本文中，我们研究了“角色分离学习”：教会LLMs稳健地区分系统和用户标记的过程。通过一个简单的可控实验框架，我们发现微调模型通常依赖于两种角色识别的捷径：（1）任务类型利用，（2）接近文本起始位置。尽管数据增强可在一定程度上减轻这些捷径的影响，但它通常导致迭代修补而不是根本解决。为解决这一问题，我们提出加强具有标志性的信号来标记角色边界，通过调整模型输入编码中的标记级提示来进行。特别是，操纵位置ID有助于模型学习更清晰的区分，减少对表面捷径的依赖。通过聚焦于机制中心的观点，我们的工作揭示了如何使LLMs更可靠地保持一致的多角色行为，而不仅仅是记忆已知提示或触发器。 

---
# FineScope : Precision Pruning for Domain-Specialized Large Language Models Using SAE-Guided Self-Data Cultivation 

**Title (ZH)**: FineScope : 基于SAE引导的自数据培养的领域专用大型语言模型精确定量剪枝 

**Authors**: Chaitali Bhattacharyya, Yeseong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.00624)  

**Abstract**: Training large language models (LLMs) from scratch requires significant computational resources, driving interest in developing smaller, domain-specific LLMs that maintain both efficiency and strong task performance. Medium-sized models such as LLaMA, llama} have served as starting points for domain-specific adaptation, but they often suffer from accuracy degradation when tested on specialized datasets. We introduce FineScope, a framework for deriving compact, domain-optimized LLMs from larger pretrained models. FineScope leverages the Sparse Autoencoder (SAE) framework, inspired by its ability to produce interpretable feature representations, to extract domain-specific subsets from large datasets. We apply structured pruning with domain-specific constraints, ensuring that the resulting pruned models retain essential knowledge for the target domain. To further enhance performance, these pruned models undergo self-data distillation, leveraging SAE-curated datasets to restore key domain-specific information lost during pruning. Extensive experiments and ablation studies demonstrate that FineScope achieves highly competitive performance, outperforming several large-scale state-of-the-art LLMs in domain-specific tasks. Additionally, our results show that FineScope enables pruned models to regain a substantial portion of their original performance when fine-tuned with SAE-curated datasets. Furthermore, applying these datasets to fine-tune pretrained LLMs without pruning also improves their domain-specific accuracy, highlighting the robustness of our approach. The code will be released. 

**Abstract (ZH)**: 一种用于 derived 域优化大语言模型的 FineScope 框架 

---
# Neural Network Verification for Gliding Drone Control: A Case Study 

**Title (ZH)**: 基于神经网络验证的滑行无人机控制：一个案例研究 

**Authors**: Colin Kessler, Ekaterina Komendantskaya, Marco Casadio, Ignazio Maria Viola, Thomas Flinkow, Albaraa Ammar Othman, Alistair Malhotra, Robbie McPherson  

**Link**: [PDF](https://arxiv.org/pdf/2505.00622)  

**Abstract**: As machine learning is increasingly deployed in autonomous systems, verification of neural network controllers is becoming an active research domain. Existing tools and annual verification competitions suggest that soon this technology will become effective for real-world applications. Our application comes from the emerging field of microflyers that are passively transported by the wind, which may have various uses in weather or pollution monitoring. Specifically, we investigate centimetre-scale bio-inspired gliding drones that resemble Alsomitra macrocarpa diaspores. In this paper, we propose a new case study on verifying Alsomitra-inspired drones with neural network controllers, with the aim of adhering closely to a target trajectory. We show that our system differs substantially from existing VNN and ARCH competition benchmarks, and show that a combination of tools holds promise for verifying such systems in the future, if certain shortcomings can be overcome. We propose a novel method for robust training of regression networks, and investigate formalisations of this case study in Vehicle and CORA. Our verification results suggest that the investigated training methods do improve performance and robustness of neural network controllers in this application, but are limited in scope and usefulness. This is due to systematic limitations of both Vehicle and CORA, and the complexity of our system reducing the scale of reachability, which we investigate in detail. If these limitations can be overcome, it will enable engineers to develop safe and robust technologies that improve people's lives and reduce our impact on the environment. 

**Abstract (ZH)**: 随着机器学习在自主系统中的应用日益增多，神经网络控制器的验证已成为一个活跃的研究领域。现有工具和年度验证竞赛表明，这项技术不久将成为实际应用的有效手段。我们的应用来自新兴的微飞行器领域，这些微飞行器被动地由风力驱动，可能在气象或污染监测中具有多种用途。具体来说，我们研究了厘米级的仿生滑翔无人机，这些无人机类似于Alsomitra macrocarpa的种子。在本文中，我们提出了一种新的案例研究，旨在验证Alsomitra仿生无人机中的神经网络控制器，以紧密遵循目标轨迹。我们表明，我们的系统在现有VNN和ARCH竞赛基准上存在显著差异，并展示了在克服某些缺陷后，工具组合在未来验证此类系统方面的潜力。我们提出了一种新的回归网络鲁棒训练方法，并在Vehicle和CORA中探讨了这一案例研究的形式化方法。验证结果表明，所研究的训练方法确实可以提高神经网络控制器在该应用中的性能和鲁棒性，但其范围和实用性存在局限性。这是由于Vehicle和CORA的系统限制以及我们系统的复杂性降低了可达性的规模，我们在详细探讨了这些局限性后得出此结论。如果这些局限性可以被克服，将使工程师能够开发出安全和鲁棒的技术，从而改善人们的生活并减少对环境的影响。 

---
# Pixel3DMM: Versatile Screen-Space Priors for Single-Image 3D Face Reconstruction 

**Title (ZH)**: Pixel3DMM: 通用的屏幕空间先验进行单张图像三维人脸重建 

**Authors**: Simon Giebenhain, Tobias Kirschstein, Martin Rünz, Lourdes Agapito, Matthias Nießner  

**Link**: [PDF](https://arxiv.org/pdf/2505.00615)  

**Abstract**: We address the 3D reconstruction of human faces from a single RGB image. To this end, we propose Pixel3DMM, a set of highly-generalized vision transformers which predict per-pixel geometric cues in order to constrain the optimization of a 3D morphable face model (3DMM). We exploit the latent features of the DINO foundation model, and introduce a tailored surface normal and uv-coordinate prediction head. We train our model by registering three high-quality 3D face datasets against the FLAME mesh topology, which results in a total of over 1,000 identities and 976K images. For 3D face reconstruction, we propose a FLAME fitting opitmization that solves for the 3DMM parameters from the uv-coordinate and normal estimates. To evaluate our method, we introduce a new benchmark for single-image face reconstruction, which features high diversity facial expressions, viewing angles, and ethnicities. Crucially, our benchmark is the first to evaluate both posed and neutral facial geometry. Ultimately, our method outperforms the most competitive baselines by over 15% in terms of geometric accuracy for posed facial expressions. 

**Abstract (ZH)**: 我们提出了一种从单张RGB图像重建人类面部的3D重建方法。为此，我们提出了Pixel3DMM，这是一种高度通用的视觉变换器集合，用于预测像素级几何线索，从而约束3D可变形面部模型（3DMM）的优化。我们利用DINO基础模型的潜在特征，并引入了定制的表面法线和uv坐标预测头。我们通过将三个高质量的3D面部数据集注册到FLAME网格拓扑上来训练我们的模型，从而总共获得了超过1000个身份和976K张图像。对于3D面部重建，我们提出了一种FLAME拟合优化方法，从uv坐标和法线估计中求解3DMM参数。为了评估我们的方法，我们引入了一个新的单一图像面部重建基准，该基准具备高度多样化的面部表情、视角和种族特征。最关键的是，我们的基准首次同时评估了表情和中性的面部几何结构。最终，我们的方法在表情面部几何精度方面比最具有竞争力的基线方法高出超过15%。 

---
# Fast and Low-Cost Genomic Foundation Models via Outlier Removal 

**Title (ZH)**: 基于离群值去除的快速低成本基因组基础模型 

**Authors**: Haozheng Luo, Chenghao Qiu, Maojiang Su, Zhihan Zhou, Zoe Mehta, Guo Ye, Jerry Yao-Chieh Hu, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00598)  

**Abstract**: We propose the first unified adversarial attack benchmark for Genomic Foundation Models (GFMs), named GERM. Unlike existing GFM benchmarks, GERM offers the first comprehensive evaluation framework to systematically assess the vulnerability of GFMs to adversarial attacks. Methodologically, we evaluate the adversarial robustness of five state-of-the-art GFMs using four widely adopted attack algorithms and three defense strategies. Importantly, our benchmark provides an accessible and comprehensive framework to analyze GFM vulnerabilities with respect to model architecture, quantization schemes, and training datasets. Empirically, transformer-based models exhibit greater robustness to adversarial perturbations compared to HyenaDNA, highlighting the impact of architectural design on vulnerability. Moreover, adversarial attacks frequently target biologically significant genomic regions, suggesting that these models effectively capture meaningful sequence features. 

**Abstract (ZH)**: 我们提出了第一个针对基因组基础模型（GFMs）的统一对抗性攻击基准——GERM。GERM提供了第一个全面的评估框架，系统性地评估GFMs对对抗性攻击的脆弱性。从方法论上，我们使用四种广泛采用的攻击算法和三种防御策略，评估了五种最先进的GFMs的对抗性鲁棒性。重要的是，我们的基准提供了一个易于访问和全面的框架，用于分析GFMs在模型架构、量化方案和训练数据集方面的脆弱性。实验证据表明，基于变换器的模型在对抗性扰动方面比HyenaDNA表现出更大的鲁棒性，这突显了架构设计对脆弱性的影响。此外，对抗性攻击频繁针对生物学上意义重大的基因组区域，表明这些模型有效地捕获了有意义的序列特征。 

---
# A Finite-State Controller Based Offline Solver for Deterministic POMDPs 

**Title (ZH)**: 基于有限状态控制器的确定性POMDPs脱机求解器 

**Authors**: Alex Schutz, Yang You, Matias Mattamala, Ipek Caliskanelli, Bruno Lacerda, Nick Hawes  

**Link**: [PDF](https://arxiv.org/pdf/2505.00596)  

**Abstract**: Deterministic partially observable Markov decision processes (DetPOMDPs) often arise in planning problems where the agent is uncertain about its environmental state but can act and observe deterministically. In this paper, we propose DetMCVI, an adaptation of the Monte Carlo Value Iteration (MCVI) algorithm for DetPOMDPs, which builds policies in the form of finite-state controllers (FSCs). DetMCVI solves large problems with a high success rate, outperforming existing baselines for DetPOMDPs. We also verify the performance of the algorithm in a real-world mobile robot forest mapping scenario. 

**Abstract (ZH)**: 确定性部分观测马尔可夫决策过程（DetPOMDPs）往往出现在智能体对其环境状态不确定但可以行动和观测确定性的规划问题中。本文提出了一种DetMCVI算法，它是MCVI算法在DetPOMDPs中的适应性改进，用于构建有限状态控制器（FSC）形式的策略。DetMCVI能够高效解决大规模问题，并优于现有的DetPOMDP基准算法。我们还在一个实际的移动机器人森林测绘场景中验证了该算法的性能。 

---
# Synthesizing and Identifying Noise Levels in Autonomous Vehicle Camera Radar Datasets 

**Title (ZH)**: 合成和识别自主车辆摄像头雷达数据中的噪声水平 

**Authors**: Mathis Morales, Golnaz Habibi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00584)  

**Abstract**: Detecting and tracking objects is a crucial component of any autonomous navigation method. For the past decades, object detection has yielded promising results using neural networks on various datasets. While many methods focus on performance metrics, few projects focus on improving the robustness of these detection and tracking pipelines, notably to sensor failures. In this paper we attempt to address this issue by creating a realistic synthetic data augmentation pipeline for camera-radar Autonomous Vehicle (AV) datasets. Our goal is to accurately simulate sensor failures and data deterioration due to real-world interferences. We also present our results of a baseline lightweight Noise Recognition neural network trained and tested on our augmented dataset, reaching an overall recognition accuracy of 54.4\% on 11 categories across 10086 images and 2145 radar point-clouds. 

**Abstract (ZH)**: 检测与跟踪对象是任何自主导航方法的关键组成部分。在过去几十年中，利用神经网络在各类数据集上进行对象检测取得了令人鼓舞的结果。尽管许多方法关注性能指标，但很少有项目致力于提高这些检测和跟踪管道的鲁棒性，特别是提高其对传感器故障的鲁棒性。在本文中，我们试图通过为基于摄像头-雷达自主车辆（AV）数据集创建一种现实的合成数据增强管道来解决这个问题。我们的目标是准确模拟传感器故障和由于实际干扰引起的数据衰减。我们还介绍了在我们的增强数据集上训练和测试的基础轻量级噪声识别神经网络的结果，该网络在10086张图像和2145个雷达点云的11个类别上达到了整体识别准确率54.4%。 

---
# Voice Cloning: Comprehensive Survey 

**Title (ZH)**: 语音克隆：综述研究 

**Authors**: Hussam Azzuni, Abdulmotaleb El Saddik  

**Link**: [PDF](https://arxiv.org/pdf/2505.00579)  

**Abstract**: Voice Cloning has rapidly advanced in today's digital world, with many researchers and corporations working to improve these algorithms for various applications. This article aims to establish a standardized terminology for voice cloning and explore its different variations. It will cover speaker adaptation as the fundamental concept and then delve deeper into topics such as few-shot, zero-shot, and multilingual TTS within that context. Finally, we will explore the evaluation metrics commonly used in voice cloning research and related datasets. This survey compiles the available voice cloning algorithms to encourage research toward its generation and detection to limit its misuse. 

**Abstract (ZH)**: 语音克隆在当今数字世界中迅速发展，许多研究者和公司致力于改进这些算法以应用于多种场景。本文旨在建立语音克隆的标准术语，并探索其不同的变体。文章将涵盖说话人适应作为基本概念，进而深入探讨此类情境下的少量学习、零样本学习和多语言TTS等领域。最后，本文将探讨语音克隆研究中常用的评估指标及相关数据集。本文综述了可用的语音克隆算法，以促进其生成和检测研究，限制其不当使用。 

---
# FreqKV: Frequency Domain Key-Value Compression for Efficient Context Window Extension 

**Title (ZH)**: FreqKV: 频域键值压缩以实现高效上下文窗口扩展 

**Authors**: Jushi Kai, Boyi Zeng, Yixuan Wang, Haoli Bai, Bo Jiang, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.00570)  

**Abstract**: Extending the context window in large language models (LLMs) is essential for applications involving long-form content generation. However, the linear increase in key-value (KV) cache memory requirements and the quadratic complexity of self-attention with respect to sequence length present significant challenges during fine-tuning and inference. Existing methods suffer from performance degradation when extending to longer contexts. In this work, we introduce a novel context extension method that optimizes both fine-tuning and inference efficiency. Our method exploits a key observation: in the frequency domain, the energy distribution of the KV cache is primarily concentrated in low-frequency components. By filtering out the high-frequency components, the KV cache can be effectively compressed with minimal information loss. Building on this insight, we propose an efficient compression technique, FreqKV, that iteratively compresses the increasing KV cache to a fixed size in the frequency domain, applicable to both fine-tuning and inference. FreqKV introduces no additional parameters or architectural modifications. With minimal fine-tuning, LLMs can learn to leverage the limited cache that is compressed in the frequency domain and extend the context window efficiently. Experiments on various long context language modeling and understanding tasks demonstrate the efficiency and efficacy of the proposed method. 

**Abstract (ZH)**: 扩展大型语言模型上下文窗口的方法：一种优化fine-tuning和推理效率的新技术 

---
# Multimodal Masked Autoencoder Pre-training for 3D MRI-Based Brain Tumor Analysis with Missing Modalities 

**Title (ZH)**: 基于MRI的脑肿瘤分析中多模态掩蔽自动编码器预训练方法研究（缺失模态情况） 

**Authors**: Lucas Robinet, Ahmad Berjaoui, Elizabeth Cohen-Jonathan Moyal  

**Link**: [PDF](https://arxiv.org/pdf/2505.00568)  

**Abstract**: Multimodal magnetic resonance imaging (MRI) constitutes the first line of investigation for clinicians in the care of brain tumors, providing crucial insights for surgery planning, treatment monitoring, and biomarker identification. Pre-training on large datasets have been shown to help models learn transferable representations and adapt with minimal labeled data. This behavior is especially valuable in medical imaging, where annotations are often scarce. However, applying this paradigm to multimodal medical data introduces a challenge: most existing approaches assume that all imaging modalities are available during both pre-training and fine-tuning. In practice, missing modalities often occur due to acquisition issues, specialist unavailability, or specific experimental designs on small in-house datasets. Consequently, a common approach involves training a separate model for each desired modality combination, making the process both resource-intensive and impractical for clinical use. Therefore, we introduce BM-MAE, a masked image modeling pre-training strategy tailored for multimodal MRI data. The same pre-trained model seamlessly adapts to any combination of available modalities, extracting rich representations that capture both intra- and inter-modal information. This allows fine-tuning on any subset of modalities without requiring architectural changes, while still benefiting from a model pre-trained on the full set of modalities. Extensive experiments show that the proposed pre-training strategy outperforms or remains competitive with baselines that require separate pre-training for each modality subset, while substantially surpassing training from scratch on several downstream tasks. Additionally, it can quickly and efficiently reconstruct missing modalities, highlighting its practical value. Code and trained models are available at: this https URL 

**Abstract (ZH)**: 多模态磁共振成像（MRI）是临床医生在脑肿瘤护理中进行初步调查的第一手段，为手术规划、治疗监控和生物标志物识别提供了关键见解。在大规模数据集上的预训练已被证明有助于模型学习可迁移的表示，并在最少标注数据的情况下进行适应。这种行为在医学影像领域尤为重要，因为标注数据往往稀缺。然而，将这种范例应用于多模态医学数据引入了挑战：大多数现有方法假设所有成像模态在预训练和微调过程中均可用。实际上，模态缺失往往由于获取问题、专家不可用或小型院内数据集的具体实验设计等原因出现。因此，一种常见做法是为每种所需的模态组合训练一个单独的模型，这不仅资源密集，而且不适用于临床应用。因此，我们引入了BM-MAE，一种专为多模态MRI数据设计的掩码图像建模预训练策略。预训练模型能够无缝适应任何可用模态组合，提取能够捕捉跨模态和层内信息的丰富表示。这使得可以在任何子集模态上进行微调而无需改变架构，同时仍能从预训练了所有模态的模型中受益。广泛的经验表明，提出的预训练策略在多个下游任务中的表现优于或与需要为每个模态子集单独预训练的基线保持竞争力，并在某些任务上显著超越从零开始训练。此外，它还可以快速高效地重建缺失的模态，突显其实用价值。代码和训练模型可在以下链接获取：this https URL。 

---
# TeLoGraF: Temporal Logic Planning via Graph-encoded Flow Matching 

**Title (ZH)**: TeLoGraF: 基于图编码流匹配的时间逻辑规划 

**Authors**: Yue Meng, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00562)  

**Abstract**: Learning to solve complex tasks with signal temporal logic (STL) specifications is crucial to many real-world applications. However, most previous works only consider fixed or parametrized STL specifications due to the lack of a diverse STL dataset and encoders to effectively extract temporal logic information for downstream tasks. In this paper, we propose TeLoGraF, Temporal Logic Graph-encoded Flow, which utilizes Graph Neural Networks (GNN) encoder and flow-matching to learn solutions for general STL specifications. We identify four commonly used STL templates and collect a total of 200K specifications with paired demonstrations. We conduct extensive experiments in five simulation environments ranging from simple dynamical models in the 2D space to high-dimensional 7DoF Franka Panda robot arm and Ant quadruped navigation. Results show that our method outperforms other baselines in the STL satisfaction rate. Compared to classical STL planning algorithms, our approach is 10-100X faster in inference and can work on any system dynamics. Besides, we show our graph-encoding method's capability to solve complex STLs and robustness to out-distribution STL specifications. Code is available at this https URL 

**Abstract (ZH)**: 使用信号时序逻辑（STL）规范学习解决复杂任务：TeLoGraF，基于图编码流的方法 

---
# Learning to Learn with Quantum Optimization via Quantum Neural Networks 

**Title (ZH)**: 使用量子神经网络通过量子优化进行学习 

**Authors**: Kuan-Cheng Chen, Hiromichi Matsuyama, Wei-Hao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00561)  

**Abstract**: Quantum Approximate Optimization Algorithms (QAOA) promise efficient solutions to classically intractable combinatorial optimization problems by harnessing shallow-depth quantum circuits. Yet, their performance and scalability often hinge on effective parameter optimization, which remains nontrivial due to rugged energy landscapes and hardware noise. In this work, we introduce a quantum meta-learning framework that combines quantum neural networks, specifically Quantum Long Short-Term Memory (QLSTM) architectures, with QAOA. By training the QLSTM optimizer on smaller graph instances, our approach rapidly generalizes to larger, more complex problems, substantially reducing the number of iterations required for convergence. Through comprehensive benchmarks on Max-Cut and Sherrington-Kirkpatrick model instances, we demonstrate that QLSTM-based optimizers converge faster and achieve higher approximation ratios compared to classical baselines, thereby offering a robust pathway toward scalable quantum optimization in the NISQ era. 

**Abstract (ZH)**: 量子近似优化算法(QAOA)通过利用浅层量子电路有望高效解决经典上难以处理的组合优化问题。然而，其性能和扩展性往往依赖于有效的参数优化，由于复杂的能量景观和硬件噪声，这一过程仍然颇具挑战性。本文提出了一种结合量子神经网络，特别是量子长短期记忆(Quantum Long Short-Term Memory, QLSTM)架构与QAOA的量子元学习框架。通过在较小的图实例上训练QLSTM优化器，我们的方法能够快速泛化到更大、更复杂的优化问题，显著减少收敛所需的迭代次数。通过对最大割(Max-Cut)和施拉热廷格-基克帕克(Sherrington-Kirkpatrick)模型实例进行全面 benchmark，我们展示了基于QLSTM的优化器收敛速度更快，达到更高的近似比，提供了在量子有限采样(NISQ)时代实现可扩展量子优化的稳健途径。 

---
# Triggering Hallucinations in LLMs: A Quantitative Study of Prompt-Induced Hallucination in Large Language Models 

**Title (ZH)**: 在LLMs中诱发幻觉：大规模语言模型基于提示诱导幻觉的定量研究 

**Authors**: Makoto Sato  

**Link**: [PDF](https://arxiv.org/pdf/2505.00557)  

**Abstract**: Hallucinations in large language models (LLMs) present a growing challenge across real-world applications, from healthcare to law, where factual reliability is essential. Despite advances in alignment and instruction tuning, LLMs can still generate outputs that are fluent yet fundamentally untrue. Understanding the cognitive dynamics that underlie these hallucinations remains an open problem. In this study, we propose a prompt-based framework to systematically trigger and quantify hallucination: a Hallucination-Inducing Prompt (HIP), which synthetically fuses semantically distant concepts (e.g., periodic table of elements and tarot divination) in a misleading way, and a Hallucination Quantifying Prompt (HQP), which scores the plausibility, confidence, and coherence of the output. Controlled experiments across multiple LLMs revealed that HIPs consistently produced less coherent and more hallucinated responses than their null-fusion controls. These effects varied across models, with reasoning-oriented LLMs showing distinct profiles from general-purpose ones. Our framework provides a reproducible testbed for studying hallucination vulnerability, and opens the door to developing safer, more introspective LLMs that can detect and self-regulate the onset of conceptual instability. 

**Abstract (ZH)**: 大型语言模型中的幻觉现象在从医疗到法律等实际应用领域构成了日益严峻的挑战，其中事实可靠性至关重要。尽管在对齐和指令调优方面取得了进步，大型语言模型仍然可以生成流畅而根本上不真实的内容。理解这些幻觉所 underlying 的认知机制仍然是一个开放的问题。在本研究中，我们提出了一种基于提示的框架，以系统地触发和量化幻觉：一种幻觉诱导提示（HIP），它以误导性的方式综合了语义上相距甚远的概念（例如，元素周期表和塔罗占卜），以及一种幻觉量化提示（HQP），用于评估输出的合理性、信心和一致性。通过对多个大型语言模型进行受控实验，我们发现HIPs产生了比其无融合对照组更不连贯且更多的幻觉反应。这些效应在不同模型之间有所不同，推理导向的大型语言模型与通用型模型显示出不同的特征。我们的框架提供了一个可重复的研究平台，用于研究幻觉脆弱性，并为开发更安全、更具内省能力的大型语言模型打开了大门，这些模型能够检测并自我调节概念不稳定的开始。 

---
# On the Mechanistic Interpretability of Neural Networks for Causality in Bio-statistics 

**Title (ZH)**: 神经网络在生物统计学因果性中的机理可解释性 

**Authors**: Jean-Baptiste A. Conan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00555)  

**Abstract**: Interpretable insights from predictive models remain critical in bio-statistics, particularly when assessing causality, where classical statistical and machine learning methods often provide inherent clarity. While Neural Networks (NNs) offer powerful capabilities for modeling complex biological data, their traditional "black-box" nature presents challenges for validation and trust in high-stakes health applications. Recent advances in Mechanistic Interpretability (MI) aim to decipher the internal computations learned by these networks. This work investigates the application of MI techniques to NNs within the context of causal inference for bio-statistics.
We demonstrate that MI tools can be leveraged to: (1) probe and validate the internal representations learned by NNs, such as those estimating nuisance functions in frameworks like Targeted Minimum Loss-based Estimation (TMLE); (2) discover and visualize the distinct computational pathways employed by the network to process different types of inputs, potentially revealing how confounders and treatments are handled; and (3) provide methodologies for comparing the learned mechanisms and extracted insights across statistical, machine learning, and NN models, fostering a deeper understanding of their respective strengths and weaknesses for causal bio-statistical analysis. 

**Abstract (ZH)**: 可解释的见解对于生物统计中的预测模型依然至关重要，特别是在评估因果性时，传统统计和机器学习方法往往提供内在的清晰性。虽然神经网络（NNs）能够建模复杂的生物数据，但它们传统的“黑箱”性质在高风险健康应用中的验证和信任方面提出了挑战。最近在机制可解释性（MI）方面的进展旨在解析这些网络所学习的内部计算。本研究探讨了在因果推断的生物统计背景下将MI技术应用于NNs的应用。我们展示了MI工具可以：（1）探究和验证NNs学习的内部表示，如在目标最小损失基于估计（TMLE）框架中估计的害处函数；（2）发现并可视化网络处理不同类型输入时所使用的独特计算路径，可能揭示协变量和治疗措施的处理方式；（3）提供统计、机器学习和NN模型中学习机制和提取见解的比较方法学，促进对其各自优势和劣势的更深层次理解，以支持因果生物统计分析。 

---
# Test-time Correlation Alignment 

**Title (ZH)**: 测试时 correlations 对齐 

**Authors**: Linjing You, Jiabao Lu, Xiayuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00533)  

**Abstract**: Deep neural networks often experience performance drops due to distribution shifts between training and test data. Although domain adaptation offers a solution, privacy concerns restrict access to training data in many real-world scenarios. This restriction has spurred interest in Test-Time Adaptation (TTA), which adapts models using only unlabeled test data. However, current TTA methods still face practical challenges: (1) a primary focus on instance-wise alignment, overlooking CORrelation ALignment (CORAL) due to missing source correlations; (2) complex backpropagation operations for model updating, resulting in overhead computation and (3) domain forgetting.
To address these challenges, we provide a theoretical analysis to investigate the feasibility of Test-time Correlation Alignment (TCA), demonstrating that correlation alignment between high-certainty instances and test instances can enhance test performances with a theoretical guarantee. Based on this, we propose two simple yet effective algorithms: LinearTCA and LinearTCA+. LinearTCA applies a simple linear transformation to achieve both instance and correlation alignment without additional model updates, while LinearTCA+ serves as a plug-and-play module that can easily boost existing TTA methods. Extensive experiments validate our theoretical insights and show that TCA methods significantly outperforms baselines across various tasks, benchmarks and backbones. Notably, LinearTCA improves adaptation accuracy by 5.88% on OfficeHome dataset, while using only 4% maximum GPU memory usage and 0.6% computation time compared to the best baseline TTA method. 

**Abstract (ZH)**: 深层神经网络经常由于训练数据与测试数据分布的变化而遭受性能下降。尽管领域适应提供了一种解决方案，但在许多现实场景中，隐私问题限制了对训练数据的访问。这种限制激发了对测试时适应（TTA）的兴趣，即仅使用未标记的测试数据来适应模型。然而，当前的TTA方法仍然面临一些实际挑战：（1）主要集中在实例级别的对齐，忽略了由于缺少源数据相关性的CORAL方法；（2）模型更新涉及复杂的反向传播操作，导致计算开销增加；（3）领域遗忘。

为解决这些挑战，我们提供了理论分析来探讨测试时相关性对齐（TCA）的可行性，证明了高可信度实例与测试实例之间相关性的对齐可以在理论上保证测试性能的提升。基于此，我们提出了两个简单而有效的方法：LinearTCA和LinearTCA+。LinearTCA通过简单的线性变换实现实例和相关性的对齐，而无需额外的模型更新，LinearTCA+则作为即插即用模块，可以轻松增强现有的TTA方法。广泛的实验证明了我们的理论见解，并表明TCA方法在各种任务、基准和骨干网络上显著优于基线方法。值得注意的是，LinearTCA在OfficeHome数据集上的适应准确性提高了5.88%，同时仅使用最大GPU内存的4%和计算时间的0.6%与最佳基线TTA方法相比。 

---
# Safety-Critical Traffic Simulation with Guided Latent Diffusion Model 

**Title (ZH)**: 受指导的潜在扩散模型在交通安全临界交通模拟中的应用 

**Authors**: Mingxing Peng, Ruoyu Yao, Xusen Guo, Yuting Xie, Xianda Chen, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.00515)  

**Abstract**: Safety-critical traffic simulation plays a crucial role in evaluating autonomous driving systems under rare and challenging scenarios. However, existing approaches often generate unrealistic scenarios due to insufficient consideration of physical plausibility and suffer from low generation efficiency. To address these limitations, we propose a guided latent diffusion model (LDM) capable of generating physically realistic and adversarial safety-critical traffic scenarios. Specifically, our model employs a graph-based variational autoencoder (VAE) to learn a compact latent space that captures complex multi-agent interactions while improving computational efficiency. Within this latent space, the diffusion model performs the denoising process to produce realistic trajectories. To enable controllable and adversarial scenario generation, we introduce novel guidance objectives that drive the diffusion process toward producing adversarial and behaviorally realistic driving behaviors. Furthermore, we develop a sample selection module based on physical feasibility checks to further enhance the physical plausibility of the generated scenarios. Extensive experiments on the nuScenes dataset demonstrate that our method achieves superior adversarial effectiveness and generation efficiency compared to existing baselines while maintaining a high level of realism. Our work provides an effective tool for realistic safety-critical scenario simulation, paving the way for more robust evaluation of autonomous driving systems. 

**Abstract (ZH)**: 安全性关键交通模拟在评估无人驾驶系统在罕见和挑战性场景下的性能中发挥着关键作用。然而，现有方法往往由于物理合理性考虑不足而生成不现实的场景，并且生成效率较低。为解决这些限制，我们提出了一种指导下的潜变量扩散模型（LDM），能够生成物理上现实且具有对抗性的安全关键交通场景。具体而言，我们的模型通过图基变分自编码器（VAE）学习一个紧凑的潜在空间，以捕捉复杂的多智能体交互并提高计算效率。在该潜在空间中，扩散模型执行去噪过程以生成现实的轨迹。为了实现可控和对抗性场景生成，我们引入了新的指导目标，以驱动扩散过程生成对抗性和行为上现实的驾驶行为。此外，我们开发了一种基于物理可行性检查的样本选择模块，以进一步增强生成场景的物理合理性。在nuScenes数据集上的 extensive 实验表明，我们的方法在对抗性效果和生成效率上均优于现有基线，同时保持了高水平的现实性。我们的工作提供了一种有效的工具，用于现实的安全关键场景模拟，为无人驾驶系统的更稳健评估铺平了道路。 

---
# HalluMix: A Task-Agnostic, Multi-Domain Benchmark for Real-World Hallucination Detection 

**Title (ZH)**: HalluMix: 一种任务无关的多领域现实世界hallucination检测基准 

**Authors**: Deanna Emery, Michael Goitia, Freddie Vargus, Iulia Neagu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00506)  

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes domains, detecting hallucinated content$\unicode{x2013}$text that is not grounded in supporting evidence$\unicode{x2013}$has become a critical challenge. Existing benchmarks for hallucination detection are often synthetically generated, narrowly focused on extractive question answering, and fail to capture the complexity of real-world scenarios involving multi-document contexts and full-sentence outputs. We introduce the HalluMix Benchmark, a diverse, task-agnostic dataset that includes examples from a range of domains and formats. Using this benchmark, we evaluate seven hallucination detection systems$\unicode{x2013}$both open and closed source$\unicode{x2013}$highlighting differences in performance across tasks, document lengths, and input representations. Our analysis highlights substantial performance disparities between short and long contexts, with critical implications for real-world Retrieval Augmented Generation (RAG) implementations. Quotient Detections achieves the best overall performance, with an accuracy of 0.82 and an F1 score of 0.84. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在高风险领域中的广泛应用，检测幻觉内容（即缺乏支持证据的文字）已成为一项关键挑战。现有的幻觉检测基准通常是由合成数据生成的，专注于提取式问答，并未能捕捉到涉及多文档上下文和完整句子输出的现实世界场景的复杂性。我们引入了HalluMix基准，这是一个多样化的、任务无关的数据集，包含多种领域和格式的示例。使用此基准，我们评估了七种幻觉检测系统（包括开源和闭源系统），展示了不同任务、文档长度和输入表示下的性能差异。我们的分析指出了短和长上下文之间显著的性能差异，对于现实世界的检索增强生成（RAG）实现具有关键性影响。Quotient Detections在整体性能上表现最佳，准确率为0.82，F1分为0.84。 

---
# Variational OOD State Correction for Offline Reinforcement Learning 

**Title (ZH)**: 离线强化学习中的变分OOD状态校正 

**Authors**: Ke Jiang, Wen Jiang, Xiaoyang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00503)  

**Abstract**: The performance of Offline reinforcement learning is significantly impacted by the issue of state distributional shift, and out-of-distribution (OOD) state correction is a popular approach to address this problem. In this paper, we propose a novel method named Density-Aware Safety Perception (DASP) for OOD state correction. Specifically, our method encourages the agent to prioritize actions that lead to outcomes with higher data density, thereby promoting its operation within or the return to in-distribution (safe) regions. To achieve this, we optimize the objective within a variational framework that concurrently considers both the potential outcomes of decision-making and their density, thus providing crucial contextual information for safe decision-making. Finally, we validate the effectiveness and feasibility of our proposed method through extensive experimental evaluations on the offline MuJoCo and AntMaze suites. 

**Abstract (ZH)**: 基于密度感知的安全感知（DASP）方法在离线强化学习中处理离域状态纠正的问题 

---
# Optimal Interactive Learning on the Job via Facility Location Planning 

**Title (ZH)**: 基于设施位置规划的最优在职互动学习 

**Authors**: Shivam Vats, Michelle Zhao, Patrick Callaghan, Mingxi Jia, Maxim Likhachev, Oliver Kroemer, George Konidaris  

**Link**: [PDF](https://arxiv.org/pdf/2505.00490)  

**Abstract**: Collaborative robots must continually adapt to novel tasks and user preferences without overburdening the user. While prior interactive robot learning methods aim to reduce human effort, they are typically limited to single-task scenarios and are not well-suited for sustained, multi-task collaboration. We propose COIL (Cost-Optimal Interactive Learning) -- a multi-task interaction planner that minimizes human effort across a sequence of tasks by strategically selecting among three query types (skill, preference, and help). When user preferences are known, we formulate COIL as an uncapacitated facility location (UFL) problem, which enables bounded-suboptimal planning in polynomial time using off-the-shelf approximation algorithms. We extend our formulation to handle uncertainty in user preferences by incorporating one-step belief space planning, which uses these approximation algorithms as subroutines to maintain polynomial-time performance. Simulated and physical experiments on manipulation tasks show that our framework significantly reduces the amount of work allocated to the human while maintaining successful task completion. 

**Abstract (ZH)**: 协作机器人必须不断适应新颖任务和用户偏好，同时避免过度负担用户。虽然之前的交互式机器人学习方法旨在减少人类努力，但它们通常局限于单一任务场景，并不适用于持续的多任务协作。我们提出COIL（成本最优交互学习）——一种多任务交互规划器，通过战略性地选择三种查询类型（技能、偏好和帮助），在一系列任务中最小化人类努力。当用户偏好已知时，我们将COIL建模为未容量限制的设施定位（UFL）问题，这使得使用现成的近似算法可以在多项式时间内进行有界次优规划。我们通过引入一步信念空间规划扩展了我们的建模方式，利用这些近似算法作为子例行程序来保持多项式时间性能以处理用户偏好不确定性。在操作任务上的模拟和物理实验表明，我们的框架在保持任务成功完成的同时显著减少了分配给人类的工作量。 

---
# MULE: Multi-terrain and Unknown Load Adaptation for Effective Quadrupedal Locomotion 

**Title (ZH)**: 多地形和未知载荷适应性四足运动控制 

**Authors**: Vamshi Kumar Kurva, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2505.00488)  

**Abstract**: Quadrupedal robots are increasingly deployed for load-carrying tasks across diverse terrains. While Model Predictive Control (MPC)-based methods can account for payload variations, they often depend on predefined gait schedules or trajectory generators, limiting their adaptability in unstructured environments. To address these limitations, we propose an Adaptive Reinforcement Learning (RL) framework that enables quadrupedal robots to dynamically adapt to both varying payloads and diverse terrains. The framework consists of a nominal policy responsible for baseline locomotion and an adaptive policy that learns corrective actions to preserve stability and improve command tracking under payload variations. We validate the proposed approach through large-scale simulation experiments in Isaac Gym and real-world hardware deployment on a Unitree Go1 quadruped. The controller was tested on flat ground, slopes, and stairs under both static and dynamic payload changes. Across all settings, our adaptive controller consistently outperformed the controller in tracking body height and velocity commands, demonstrating enhanced robustness and adaptability without requiring explicit gait design or manual tuning. 

**Abstract (ZH)**: 四足机器人越来越多地被部署于多种地形的负载搬运任务中。尽管基于模型预测控制(MPC)的方法可以考虑到负载变化，但这些方法往往依赖预先定义的步伐计划或轨迹生成器，限制了其在非结构化环境中的适应性。为了解决这些局限性，我们提出了一种自适应强化学习(Adaptive Reinforcement Learning, ARL)框架，使四足机器人能够动态适应变化的负载和多样的地形。该框架包括一个基线策略，负责基本的移动，以及一个自适应策略，学习纠正动作以在负载变化下保持稳定性和提高指令跟踪性能。我们通过在Isaac Gym中的大规模仿真实验和在Unitree Go1四足机器人上的实际硬件部署验证了所提出的框架。控制器在平坦地面、坡道和台阶上，在静态和动态负载变化下进行了测试。在所有设置中，我们的自适应控制器在跟踪身体高度和速度命令方面始终优于基线控制器，展示了增强的鲁棒性和适应性，无需显式的步态设计或手动调优。 

---
# Analysis of the vulnerability of machine learning regression models to adversarial attacks using data from 5G wireless networks 

**Title (ZH)**: 基于5G无线网络数据的机器学习回归模型对对抗攻击的脆弱性分析 

**Authors**: Leonid Legashev, Artur Zhigalov, Denis Parfenov  

**Link**: [PDF](https://arxiv.org/pdf/2505.00487)  

**Abstract**: This article describes the process of creating a script and conducting an analytical study of a dataset using the DeepMIMO emulator. An advertorial attack was carried out using the FGSM method to maximize the gradient. A comparison is made of the effectiveness of binary classifiers in the task of detecting distorted data. The dynamics of changes in the quality indicators of the regression model were analyzed in conditions without adversarial attacks, during an adversarial attack and when the distorted data was isolated. It is shown that an adversarial FGSM attack with gradient maximization leads to an increase in the value of the MSE metric by 33% and a decrease in the R2 indicator by 10% on average. The LightGBM binary classifier effectively identifies data with adversarial anomalies with 98% accuracy. Regression machine learning models are susceptible to adversarial attacks, but rapid analysis of network traffic and data transmitted over the network makes it possible to identify malicious activity 

**Abstract (ZH)**: 本文描述了使用DeepMIMO模拟器创建脚本并进行数据集分析性研究的过程。采用FGSM方法执行推销广告攻击以最大化梯度。比较了二元分类器在检测篡改数据任务中的有效性。分析了在无对抗攻击、对抗攻击期间以及隔离篡改数据条件下的回归模型质量指标变化动态。结果显示，具有梯度最大化特征的对抗性FGSM攻击使得MSE指标值平均增加了33%，R2指标降低了10%。LightGBM二元分类器能够以98%的准确率识别对抗性异常数据。回归机器学习模型容易受到对抗攻击的影响，但快速分析网络流量和网络上传输的数据可以识别出恶意活动。 

---
# JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers 

**Title (ZH)**: JointDiT: 提升RGB-深度联合建模的扩散变换器方法 

**Authors**: Kwon Byung-Ki, Qi Dai, Lee Hyoseok, Chong Luo, Tae-Hyun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2505.00482)  

**Abstract**: We present JointDiT, a diffusion transformer that models the joint distribution of RGB and depth. By leveraging the architectural benefit and outstanding image prior of the state-of-the-art diffusion transformer, JointDiT not only generates high-fidelity images but also produces geometrically plausible and accurate depth maps. This solid joint distribution modeling is achieved through two simple yet effective techniques that we propose, i.e., adaptive scheduling weights, which depend on the noise levels of each modality, and the unbalanced timestep sampling strategy. With these techniques, we train our model across all noise levels for each modality, enabling JointDiT to naturally handle various combinatorial generation tasks, including joint generation, depth estimation, and depth-conditioned image generation by simply controlling the timestep of each branch. JointDiT demonstrates outstanding joint generation performance. Furthermore, it achieves comparable results in depth estimation and depth-conditioned image generation, suggesting that joint distribution modeling can serve as a replaceable alternative to conditional generation. The project page is available at this https URL. 

**Abstract (ZH)**: 我们 presents JointDiT，一种建模RGB和深度联合分布的扩散变换器。通过利用最新扩散变换器的架构优势和出色的图像先验知识，JointDiT 不仅生成高保真图像，还能产生几何上合理且准确的深度图。通过我们提出的一种简单而有效的技术，即适应性调度权重（取决于每种模态的噪声水平）和不平衡时间步采样策略，实现了这种坚实的联合分布建模。借助这些技术，我们在每个模态的所有噪声水平下训练模型，使得JointDiT能够轻松处理各种组合生成任务，包括联合生成、深度估计和深度条件下的图像生成，只需控制每个分支的时间步即可。JointDiT展示了出色的联合生成性能。此外，在深度估计和深度条件下的图像生成方面，它达到了可比的结果，表明联合分布建模可以作为条件生成的一种可替代方案。项目页面在此处 accessible at this https URL。 

---
# Red Teaming Large Language Models for Healthcare 

**Title (ZH)**: 面向医疗健康的红队测试大型语言模型 

**Authors**: Vahid Balazadeh, Michael Cooper, David Pellow, Atousa Assadi, Jennifer Bell, Jim Fackler, Gabriel Funingana, Spencer Gable-Cook, Anirudh Gangadhar, Abhishek Jaiswal, Sumanth Kaja, Christopher Khoury, Randy Lin, Kaden McKeen, Sara Naimimohasses, Khashayar Namdar, Aviraj Newatia, Allan Pang, Anshul Pattoo, Sameer Peesapati, Diana Prepelita, Bogdana Rakova, Saba Sadatamin, Rafael Schulman, Ajay Shah, Syed Azhar Shah, Syed Ahmar Shah, Babak Taati, Balagopal Unnikrishnan, Stephanie Williams, Rahul G Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00467)  

**Abstract**: We present the design process and findings of the pre-conference workshop at the Machine Learning for Healthcare Conference (2024) entitled Red Teaming Large Language Models for Healthcare, which took place on August 15, 2024. Conference participants, comprising a mix of computational and clinical expertise, attempted to discover vulnerabilities -- realistic clinical prompts for which a large language model (LLM) outputs a response that could cause clinical harm. Red-teaming with clinicians enables the identification of LLM vulnerabilities that may not be recognised by LLM developers lacking clinical expertise. We report the vulnerabilities found, categorise them, and present the results of a replication study assessing the vulnerabilities across all LLMs provided. 

**Abstract (ZH)**: 我们 presents 机器学习医疗大会（2024）前会议工作会议《红队测试医疗领域的大型语言模型》的设计过程与发现，该会议于2024年8月15日举行。会议参与者包括计算和临床专业背景的混合群体，他们尝试发现漏洞——即大型语言模型（LLM）对临床有害响应的临床提示。临床红队测试有助于识别缺乏临床背景的大型语言模型开发者可能未意识到的漏洞。我们报告发现的漏洞，对其进行分类，并展示一项复制研究的结果，评估这些漏洞在所有提供的大型语言模型中的一致性。 

---
# Data Therapist: Eliciting Domain Knowledge from Subject Matter Experts Using Large Language Models 

**Title (ZH)**: 数据治疗师：使用大型语言模型从领域专家处提取专业知识 

**Authors**: Sungbok Shin, Hyeon Jeon, Sanghyun Hong, Niklas Elmqvist  

**Link**: [PDF](https://arxiv.org/pdf/2505.00455)  

**Abstract**: Effective data visualization requires not only technical proficiency but also a deep understanding of the domain-specific context in which data exists. This context often includes tacit knowledge about data provenance, quality, and intended use, which is rarely explicit in the dataset itself. We present the Data Therapist, a web-based tool that helps domain experts externalize this implicit knowledge through a mixed-initiative process combining iterative Q&A with interactive annotation. Powered by a large language model, the system analyzes user-supplied datasets, prompts users with targeted questions, and allows annotation at varying levels of granularity. The resulting structured knowledge base can inform both human and automated visualization design. We evaluated the tool in a qualitative study involving expert pairs from Molecular Biology, Accounting, Political Science, and Usable Security. The study revealed recurring patterns in how experts reason about their data and highlights areas where AI support can improve visualization design. 

**Abstract (ZH)**: 有效的数据可视化不仅需要技术 proficiency，还需要对数据存在的领域特定上下文有深刻的理解。这种上下文通常包括关于数据来源、质量及预期用途的隐性知识，而这些知识在数据集本身中往往并未明确体现。我们提出了Data Therapist这一基于Web的工具，通过结合迭代问答和互动注释的混合主动过程，帮助领域专家外化这一隐性知识。该系统依托大型语言模型，分析用户提供的数据集，向用户提出针对性的问题，并允许在不同粒度级别进行注释。生成的结构化知识库可以指导人类和自动化的可视化设计。我们在涉及分子生物学、会计学、政治科学和可用安全性领域的专家配对中开展了定性研究，研究揭示了专家在处理数据时思维方式中的反复出现模式，并强调了AI支持如何改进可视化设计的领域。 

---
# Per-Domain Generalizing Policies: On Validation Instances and Scaling Behavior 

**Title (ZH)**: 基于域的泛化策略：关于验证实例和扩展行为的研究 

**Authors**: Timo P. Gros, Nicola J. Müller, Daniel Fiser, Isabel Valera, Verena Wolf, Jörg Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.00439)  

**Abstract**: Recent work has shown that successful per-domain generalizing action policies can be learned. Scaling behavior, from small training instances to large test instances, is the key objective; and the use of validation instances larger than training instances is one key to achieve it. Prior work has used fixed validation sets. Here, we introduce a method generating the validation set dynamically, on the fly, increasing instance size so long as informative and this http URL also introduce refined methodology for evaluating scaling behavior, generating test instances systematically to guarantee a given confidence in coverage performance for each instance size. In experiments, dynamic validation improves scaling behavior of GNN policies in all 9 domains used. 

**Abstract (ZH)**: 近期的工作表明，可以学习到在各个领域中表现成功的动作策略。从少量的训练样本到大量的测试样本，扩展行为是关键目标；在训练样本少于测试样本的情况下使用动态生成的验证集是实现这一目标的关键。以往的工作使用固定的验证集。我们在此引入了一种动态生成验证集的方法，在运行过程中根据样本信息不断增加样本大小。同时，我们还引入了一种改进的评估扩展行为的方法，系统地生成测试样本以确保每个样本大小下的覆盖率性能具有给定的置信度。在实验中，动态验证集提高了所有9个领域中GNN策略的扩展行为。 

---
# Perceptual Implications of Automatic Anonymization in Pathological Speech 

**Title (ZH)**: 病理语音的自动匿名化感知影响 

**Authors**: Soroosh Tayebi Arasteh, Saba Afza, Tri-Thien Nguyen, Lukas Buess, Maryam Parvin, Tomas Arias-Vergara, Paula Andrea Perez-Toro, Hiu Ching Hung, Mahshad Lotfinia, Thomas Gorges, Elmar Noeth, Maria Schuster, Seung Hee Yang, Andreas Maier  

**Link**: [PDF](https://arxiv.org/pdf/2505.00409)  

**Abstract**: Automatic anonymization techniques are essential for ethical sharing of pathological speech data, yet their perceptual consequences remain understudied. This study presents the first comprehensive human-centered analysis of anonymized pathological speech, using a structured perceptual protocol involving ten native and non-native German listeners with diverse linguistic, clinical, and technical backgrounds. Listeners evaluated anonymized-original utterance pairs from 180 speakers spanning Cleft Lip and Palate, Dysarthria, Dysglossia, Dysphonia, and age-matched healthy controls. Speech was anonymized using state-of-the-art automatic methods (equal error rates in the range of 30-40%). Listeners completed Turing-style discrimination and quality rating tasks under zero-shot (single-exposure) and few-shot (repeated-exposure) conditions. Discrimination accuracy was high overall (91% zero-shot; 93% few-shot), but varied by disorder (repeated-measures ANOVA: p=0.007), ranging from 96% (Dysarthria) to 86% (Dysphonia). Anonymization consistently reduced perceived quality (from 83% to 59%, p<0.001), with pathology-specific degradation patterns (one-way ANOVA: p=0.005). Native listeners rated original speech slightly higher than non-native listeners (Delta=4%, p=0.199), but this difference nearly disappeared after anonymization (Delta=1%, p=0.724). No significant gender-based bias was observed. Critically, human perceptual outcomes did not correlate with automatic privacy or clinical utility metrics. These results underscore the need for listener-informed, disorder- and context-specific anonymization strategies that preserve privacy while maintaining interpretability, communicative functions, and diagnostic utility, especially for vulnerable populations such as children. 

**Abstract (ZH)**: 自动匿名化技术对于病理语音数据的伦理共享至关重要，但其感知后果仍研究不足。本研究首次进行了综合的人本中心分析，使用了包含十名语言背景、临床背景和技术背景多样化的德语母语者和非母语者的结构化感知协议。参与者评估了180名讲者的匿名化原始表述对，涵盖唇裂和腭裂、构音障碍、构词障碍、声带障碍以及年龄匹配的健康对照组。语音使用最先进的自动匿名化方法进行了匿名处理（等错误率范围在30-40%）。参与者在单次接触和多次接触条件下完成了图灵风格的辨别和质量评分任务。总体上，辨别准确率较高（单次接触91%，多次接触93%），但不同疾病之间的准确率存在差异（重复测量ANOVA：p=0.007），范围从96%（构音障碍）到86%（声带障碍）。匿名处理一致降低了感知质量（从83%下降到59%，p<0.001），显示出疾病特异性的降质模式（单因素ANOVA：p=0.005）。母语者对原始语音的评价略高于非母语者（Delta=4%，p=0.199），但在匿名处理后，这种差异几乎消失（Delta=1%，p=0.724）。未观察到显著的性别偏见。关键的是，人类感知结果与自动隐私或临床效用指标无相关性。这些结果强调了需要基于听众输入、针对特定疾病和情境定制的匿名化策略的重要性，以同时保护隐私和保持可解释性、沟通功能和诊断效用，尤其是在易损人群如儿童中尤为重要。 

---
# DeepSTA: A Spatial-Temporal Attention Network for Logistics Delivery Timely Rate Prediction in Anomaly Conditions 

**Title (ZH)**: DeepSTA：在异常条件下用于物流配送准时率预测的空间-时间注意力网络 

**Authors**: Jinhui Yi, Huan Yan, Haotian Wang, Jian Yuan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.00402)  

**Abstract**: Prediction of couriers' delivery timely rates in advance is essential to the logistics industry, enabling companies to take preemptive measures to ensure the normal operation of delivery services. This becomes even more critical during anomaly conditions like the epidemic outbreak, during which couriers' delivery timely rate will decline markedly and fluctuates significantly. Existing studies pay less attention to the logistics scenario. Moreover, many works focusing on prediction tasks in anomaly scenarios fail to explicitly model abnormal events, e.g., treating external factors equally with other features, resulting in great information loss. Further, since some anomalous events occur infrequently, traditional data-driven methods perform poorly in these scenarios. To deal with them, we propose a deep spatial-temporal attention model, named DeepSTA. To be specific, to avoid information loss, we design an anomaly spatio-temporal learning module that employs a recurrent neural network to model incident information. Additionally, we utilize Node2vec to model correlations between road districts, and adopt graph neural networks and long short-term memory to capture the spatial-temporal dependencies of couriers. To tackle the issue of insufficient training data in abnormal circumstances, we propose an anomaly pattern attention module that adopts a memory network for couriers' anomaly feature patterns storage via attention mechanisms. The experiments on real-world logistics datasets during the COVID-19 outbreak in 2022 show the model outperforms the best baselines by 12.11% in MAE and 13.71% in MSE, demonstrating its superior performance over multiple competitive baselines. 

**Abstract (ZH)**: 预测快递员在异常情况下的准时率对于物流行业至关重要，有助于企业在发生预兆时采取预防措施，确保配送服务的正常运行。在疫情期间等异常情况下，快递员的准时率会显著下降并波动较大。现有研究较少关注物流场景。此外，许多专注于异常场景下的预测任务的工作未能明确建模异常事件，例如将外部因素与其他特征同等对待，导致大量信息损失。由于一些异常事件发生的频率很低，传统数据驱动方法在这种情况下表现不佳。为应对这种情况，我们提出了一种深度空间-时间注意力模型，名为DeepSTA。具体而言，为避免信息丢失，我们设计了一种异常时空学习模块，利用递归神经网络建模事件信息。此外，我们利用Node2vec建模道路区域之间的关联，并采用图神经网络和长短期记忆网络捕捉快递员的空间-时间依赖关系。为了应对异常情况下训练数据不足的问题，我们提出了一种异常模式注意力模块，利用记忆网络通过注意力机制存储快递员的异常特征模式。在2022年COVID-19疫情期间的真实物流数据集上的实验结果显示，该模型在MAE上优于最佳基线12.11%，在MSE上优于13.71%，展示了其在多个竞争性基线中的优越性能。 

---
# Learning to Estimate Package Delivery Time in Mixed Imbalanced Delivery and Pickup Logistics Services 

**Title (ZH)**: 学习估计混合不平衡配送与取货物流服务的包裹配送时间 

**Authors**: Jinhui Yi, Huan Yan, Haotian Wang, Jian Yuan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.00375)  

**Abstract**: Accurately estimating package delivery time is essential to the logistics industry, which enables reasonable work allocation and on-time service guarantee. This becomes even more necessary in mixed logistics scenarios where couriers handle a high volume of delivery and a smaller number of pickup simultaneously. However, most of the related works treat the pickup and delivery patterns on couriers' decision behavior equally, neglecting that the pickup has a greater impact on couriers' decision-making compared to the delivery due to its tighter time constraints. In such context, we have three main challenges: 1) multiple spatiotemporal factors are intricately interconnected, significantly affecting couriers' delivery behavior; 2) pickups have stricter time requirements but are limited in number, making it challenging to model their effects on couriers' delivery process; 3) couriers' spatial mobility patterns are critical determinants of their delivery behavior, but have been insufficiently explored. To deal with these, we propose TransPDT, a Transformer-based multi-task package delivery time prediction model. We first employ the Transformer encoder architecture to capture the spatio-temporal dependencies of couriers' historical travel routes and pending package sets. Then we design the pattern memory to learn the patterns of pickup in the imbalanced dataset via attention mechanism. We also set the route prediction as an auxiliary task of delivery time prediction, and incorporate the prior courier spatial movement regularities in prediction. Extensive experiments on real industry-scale datasets demonstrate the superiority of our method. A system based on TransPDT is deployed internally in JD Logistics to track more than 2000 couriers handling hundreds of thousands of packages per day in Beijing. 

**Abstract (ZH)**: 基于 Transformer 的多任务快递配送时间预测模型 TransPDT 

---
# KoACD: The First Korean Adolescent Dataset for Cognitive Distortion Analysis 

**Title (ZH)**: KoACD: 首个青少年认知扭曲分析数据集 

**Authors**: JunSeo Kim, HyeHyeon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.00367)  

**Abstract**: Cognitive distortion refers to negative thinking patterns that can lead to mental health issues like depression and anxiety in adolescents. Previous studies using natural language processing (NLP) have focused mainly on small-scale adult datasets, with limited research on adolescents. This study introduces KoACD, the first large-scale dataset of cognitive distortions in Korean adolescents, containing 108,717 instances. We applied a multi-Large Language Model (LLM) negotiation method to refine distortion classification and generate synthetic data using two approaches: cognitive clarification for textual clarity and cognitive balancing for diverse distortion representation. Validation through LLMs and expert evaluations showed that while LLMs classified distortions with explicit markers, they struggled with context-dependent reasoning, where human evaluators demonstrated higher accuracy. KoACD aims to enhance future research on cognitive distortion detection. 

**Abstract (ZH)**: 认知 distortion 在青少年中的消极思维模式与其心理健康问题如抑郁和焦虑有关。 previous 研究主要使用自然语言处理 (NLP) 技术集中在小型成人数据集上，对青少年的研究较少。本研究介绍了 KoACD，这是第一个包含 108,717 个实例的韩语青少年认知 distortion 大规模数据集。我们应用多大型语言模型 (LLM) 协商方法细化 distortion 分类，并使用两种方法生成合成数据：认知澄清以提高文本清晰度，认知平衡以实现多样化的 distortion 表现。通过大型语言模型和专家评估的验证显示，虽然 LLM 能够分类具有明确标记的 distortion，但在依赖上下文的推理方面却表现不佳，人类评估者表现出更高的准确性。KoACD旨在增强未来关于认知 distortion 检测的研究。 

---
# SacFL: Self-Adaptive Federated Continual Learning for Resource-Constrained End Devices 

**Title (ZH)**: SacFL: 自适应联邦连续学习roach for 资源受限的边缘设备 

**Authors**: Zhengyi Zhong, Weidong Bao, Ji Wang, Jianguo Chen, Lingjuan Lyu, Wei Yang Bryan Lim  

**Link**: [PDF](https://arxiv.org/pdf/2505.00365)  

**Abstract**: The proliferation of end devices has led to a distributed computing paradigm, wherein on-device machine learning models continuously process diverse data generated by these devices. The dynamic nature of this data, characterized by continuous changes or data drift, poses significant challenges for on-device models. To address this issue, continual learning (CL) is proposed, enabling machine learning models to incrementally update their knowledge and mitigate catastrophic forgetting. However, the traditional centralized approach to CL is unsuitable for end devices due to privacy and data volume concerns. In this context, federated continual learning (FCL) emerges as a promising solution, preserving user data locally while enhancing models through collaborative updates. Aiming at the challenges of limited storage resources for CL, poor autonomy in task shift detection, and difficulty in coping with new adversarial tasks in FCL scenario, we propose a novel FCL framework named SacFL. SacFL employs an Encoder-Decoder architecture to separate task-robust and task-sensitive components, significantly reducing storage demands by retaining lightweight task-sensitive components for resource-constrained end devices. Moreover, $\rm{SacFL}$ leverages contrastive learning to introduce an autonomous data shift detection mechanism, enabling it to discern whether a new task has emerged and whether it is a benign task. This capability ultimately allows the device to autonomously trigger CL or attack defense strategy without additional information, which is more practical for end devices. Comprehensive experiments conducted on multiple text and image datasets, such as Cifar100 and THUCNews, have validated the effectiveness of $\rm{SacFL}$ in both class-incremental and domain-incremental scenarios. Furthermore, a demo system has been developed to verify its practicality. 

**Abstract (ZH)**: 联邦持续学习框架SacFL：适应有限存储资源和自主数据转移检测 

---
# TNStream: Applying Tightest Neighbors to Micro-Clusters to Define Multi-Density Clusters in Streaming Data 

**Title (ZH)**: TNStream：利用最紧邻微聚类定义流数据中的多密度聚类 

**Authors**: Qifen Zeng, Haomin Bao, Yuanzhuo Hu, Zirui Zhang, Yuheng Zheng, Luosheng Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00359)  

**Abstract**: In data stream clustering, systematic theory of stream clustering algorithms remains relatively scarce. Recently, density-based methods have gained attention. However, existing algorithms struggle to simultaneously handle arbitrarily shaped, multi-density, high-dimensional data while maintaining strong outlier resistance. Clustering quality significantly deteriorates when data density varies complexly. This paper proposes a clustering algorithm based on the novel concept of Tightest Neighbors and introduces a data stream clustering theory based on the Skeleton Set. Based on these theories, this paper develops a new method, TNStream, a fully online algorithm. The algorithm adaptively determines the clustering radius based on local similarity, summarizing the evolution of multi-density data streams in micro-clusters. It then applies a Tightest Neighbors-based clustering algorithm to form final clusters. To improve efficiency in high-dimensional cases, Locality-Sensitive Hashing (LSH) is employed to structure micro-clusters, addressing the challenge of storing k-nearest neighbors. TNStream is evaluated on various synthetic and real-world datasets using different clustering metrics. Experimental results demonstrate its effectiveness in improving clustering quality for multi-density data and validate the proposed data stream clustering theory. 

**Abstract (ZH)**: 基于最邻近点的流数据聚类算法及其理论 

---
# R&B: Domain Regrouping and Data Mixture Balancing for Efficient Foundation Model Training 

**Title (ZH)**: R&B：领域重新分组与数据混合平衡的高效基础模型训练 

**Authors**: Albert Ge, Tzu-Heng Huang, John Cooper, Avi Trost, Ziyi Chu, Satya Sai Srinath Namburi GNVV, Ziyang Cai, Kendall Park, Nicholas Roberts, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2505.00358)  

**Abstract**: Data mixing strategies have successfully reduced the costs involved in training language models. While promising, such methods suffer from two flaws. First, they rely on predetermined data domains (e.g., data sources, task types), which may fail to capture critical semantic nuances, leaving performance on the table. Second, these methods scale with the number of domains in a computationally prohibitive way. We address these challenges via R&B, a framework that re-partitions training data based on semantic similarity (Regroup) to create finer-grained domains, and efficiently optimizes the data composition (Balance) by leveraging a Gram matrix induced by domain gradients obtained throughout training. Unlike prior works, it removes the need for additional compute to obtain evaluation information such as losses or gradients. We analyze this technique under standard regularity conditions and provide theoretical insights that justify R&B's effectiveness compared to non-adaptive mixing approaches. Empirically, we demonstrate the effectiveness of R&B on five diverse datasets ranging from natural language to reasoning and multimodal tasks. With as little as 0.01% additional compute overhead, R&B matches or exceeds the performance of state-of-the-art data mixing strategies. 

**Abstract (ZH)**: 数据混合作策略已成功降低了训练语言模型的成本。然而，这些方法存在两个缺陷。首先，它们依赖于预先定义的数据领域（例如，数据来源、任务类型），这可能会错过关键的语义细微差别，从而导致性能的损失。其次，这些方法随着领域的数量增加在计算上变得不可行。我们通过R&B框架解决这些挑战，该框架基于语义相似性重新分区训练数据（Regroup），并通过利用训练过程中获得的领域梯度诱导的Gram矩阵来高效优化数据组成（Balance）。与先前的工作不同，它消除了获取评估信息（如损失或梯度）所需的额外计算需求。我们在此标准正则条件下分析此技术，并提供理论见解以证明R&B相对于非自适应混合作方法的有效性。实验结果显示，R&B在五个不同的数据集上有效，这些数据集涵盖了从自然语言处理到推理和多模态任务的范围。即使是额外计算开销仅为0.01%，R&B也能匹配或超越最先进的数据混合作策略的性能。 

---
# Optimizing Deep Neural Networks using Safety-Guided Self Compression 

**Title (ZH)**: 使用安全性引导自我压缩优化深度神经网络 

**Authors**: Mohammad Zbeeb, Mariam Salman, Mohammad Bazzi, Ammar Mohanna  

**Link**: [PDF](https://arxiv.org/pdf/2505.00350)  

**Abstract**: The deployment of deep neural networks on resource-constrained devices necessitates effective model com- pression strategies that judiciously balance the reduction of model size with the preservation of performance. This study introduces a novel safety-driven quantization framework that leverages preservation sets to systematically prune and quantize neural network weights, thereby optimizing model complexity without compromising accuracy. The proposed methodology is rigorously evaluated on both a convolutional neural network (CNN) and an attention-based language model, demonstrating its applicability across diverse architectural paradigms. Experimental results reveal that our framework achieves up to a 2.5% enhancement in test accuracy relative to the original unquantized models while maintaining 60% of the initial model size. In comparison to conventional quantization techniques, our approach not only augments generalization by eliminating parameter noise and retaining essential weights but also reduces variance, thereby ensuring the retention of critical model features. These findings underscore the efficacy of safety-driven quantization as a robust and reliable strategy for the efficient optimization of deep learn- ing models. The implementation and comprehensive experimental evaluations of our framework are publicly accessible at GitHub. 

**Abstract (ZH)**: 在资源受限设备上部署深度神经网络需要有效的模型压缩策略，以适度地平衡模型大小的减小与性能的保留。本研究介绍了一种新的安全性驱动的量化框架，该框架利用保留集系统地剪枝和量化神经网络权重，从而在不牺牲准确性的前提下优化模型复杂度。所提出的方法在卷积神经网络（CNN）和基于注意力的语言模型上进行了严格的评估，证明了其在不同架构范式中的适用性。实验结果表明，与原始未量化模型相比，该框架在测试准确率上提高了2.5%，同时保持了初始模型大小的60%。与传统量化技术相比，我们的方法不仅通过消除参数噪声并保留关键权重来增强泛化能力，还减少了变异，从而确保关键模型特征的保留。这些发现强调了安全性驱动量化作为深度学习模型高效优化稳健可靠策略的有效性。我们的框架的实现和全面的实验评估可以在GitHub上公开访问。 

---
# Pushing the Limits of Low-Bit Optimizers: A Focus on EMA Dynamics 

**Title (ZH)**: 低比特优化器能力的极限探索：聚焦EMA动力学 

**Authors**: Cong Xu, Wenbin Liang, Mo Yu, Anan Liu, Ke-Yue Zhang, Lizhuang Ma, Jianyong Wang, Jun Wang, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00347)  

**Abstract**: The explosion in model sizes leads to continued growth in prohibitive training/fine-tuning costs, particularly for stateful optimizers which maintain auxiliary information of even 2x the model size to achieve optimal convergence. We therefore present in this work a novel type of optimizer that carries with extremely lightweight state overloads, achieved through ultra-low-precision quantization. While previous efforts have achieved certain success with 8-bit or 4-bit quantization, our approach enables optimizers to operate at precision as low as 3 bits, or even 2 bits per state element. This is accomplished by identifying and addressing two critical challenges: the signal swamping problem in unsigned quantization that results in unchanged state dynamics, and the rapidly increased gradient variance in signed quantization that leads to incorrect descent directions. The theoretical analysis suggests a tailored logarithmic quantization for the former and a precision-specific momentum value for the latter. Consequently, the proposed SOLO achieves substantial memory savings (approximately 45 GB when training a 7B model) with minimal accuracy loss. We hope that SOLO can contribute to overcoming the bottleneck in computational resources, thereby promoting greater accessibility in fundamental research. 

**Abstract (ZH)**: 模型规模的爆炸式增长导致训练/微调成本持续飙升，特别是对于维护甚至达到模型大小两倍的辅助信息以实现最优收敛的状态型优化器。因此，在本文中，我们提出了一种新型优化器，该优化器通过超低精度量化携带极其轻量级的状态。虽然之前的努力在8位或4位量化方面取得了一定的成功，但我们的方法使优化器能够在每个状态元素低至3位，甚至2位的精度下运行。通过识别并解决两种关键挑战——无符号量化中的信号淹没问题导致状态动力学不变，以及有符号量化中梯度方差迅速增加导致错误的下降方向——实现了这一目标。理论分析表明，前者应采用定制的对数量化，后者应采用特定精度的动量值。因此，所提出的SOLO在保持最小精度损失的情况下实现了显著的内存节省（例如，训练一个7B模型时约节省45 GB）。我们希望SOLO能够有助于克服计算资源瓶颈，从而促进基础研究的更大普及。 

---
# Enhancing AI-Driven Education: Integrating Cognitive Frameworks, Linguistic Feedback Analysis, and Ethical Considerations for Improved Content Generation 

**Title (ZH)**: 增强AI驱动的教育：结合认知框架、语言反馈分析和伦理考虑以改进内容生成 

**Authors**: Antoun Yaacoub, Sansiri Tarnpradab, Phattara Khumprom, Zainab Assaghir, Lionel Prevost, Jérôme Da-Rugna  

**Link**: [PDF](https://arxiv.org/pdf/2505.00339)  

**Abstract**: Artificial intelligence (AI) is rapidly transforming education, presenting unprecedented opportunities for personalized learning and streamlined content creation. However, realizing the full potential of AI in educational settings necessitates careful consideration of the quality, cognitive depth, and ethical implications of AI-generated materials. This paper synthesizes insights from four related studies to propose a comprehensive framework for enhancing AI-driven educational tools. We integrate cognitive assessment frameworks (Bloom's Taxonomy and SOLO Taxonomy), linguistic analysis of AI-generated feedback, and ethical design principles to guide the development of effective and responsible AI tools. We outline a structured three-phase approach encompassing cognitive alignment, linguistic feedback integration, and ethical safeguards. The practical application of this framework is demonstrated through its integration into OneClickQuiz, an AI-powered Moodle plugin for quiz generation. This work contributes a comprehensive and actionable guide for educators, researchers, and developers aiming to harness AI's potential while upholding pedagogical and ethical standards in educational content generation. 

**Abstract (ZH)**: 人工智能（AI）正迅速变革教育，为个性化学习和内容创建提供前所未有的机遇。然而，在教育环境中实现AI的全部潜能需要仔细考虑AI生成材料的质量、认知深度和伦理影响。本文综合四项相关研究的见解，提出一个全面框架以提升AI驱动的教育工具。我们结合认知评估框架（布卢姆分类法和SOLO分类法）、AI生成反馈的语言分析以及伦理设计原则，指导有效负责任的AI工具的开发。我们概述了一个结构化的三阶段方法，包括认知对齐、语言反馈集成和伦理保障。通过将其整合到OneClickQuiz（一个基于AI的Moodle插件以生成测验）中，展示了该框架的实际应用。本研究为致力于利用AI潜力、同时在教育内容生成中维护教学和伦理标准的教育者、研究人员和开发人员提供了一个全面且可操作的指南。 

---
# T2VPhysBench: A First-Principles Benchmark for Physical Consistency in Text-to-Video Generation 

**Title (ZH)**: T2VPhysBench: 首个文本到视频生成物理一致性基准 

**Authors**: Xuyang Guo, Jiayan Huo, Zhenmei Shi, Zhao Song, Jiahao Zhang, Jiale Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.00337)  

**Abstract**: Text-to-video generative models have made significant strides in recent years, producing high-quality videos that excel in both aesthetic appeal and accurate instruction following, and have become central to digital art creation and user engagement online. Yet, despite these advancements, their ability to respect fundamental physical laws remains largely untested: many outputs still violate basic constraints such as rigid-body collisions, energy conservation, and gravitational dynamics, resulting in unrealistic or even misleading content. Existing physical-evaluation benchmarks typically rely on automatic, pixel-level metrics applied to simplistic, life-scenario prompts, and thus overlook both human judgment and first-principles physics. To fill this gap, we introduce \textbf{T2VPhysBench}, a first-principled benchmark that systematically evaluates whether state-of-the-art text-to-video systems, both open-source and commercial, obey twelve core physical laws including Newtonian mechanics, conservation principles, and phenomenological effects. Our benchmark employs a rigorous human evaluation protocol and includes three targeted studies: (1) an overall compliance assessment showing that all models score below 0.60 on average in each law category; (2) a prompt-hint ablation revealing that even detailed, law-specific hints fail to remedy physics violations; and (3) a counterfactual robustness test demonstrating that models often generate videos that explicitly break physical rules when so instructed. The results expose persistent limitations in current architectures and offer concrete insights for guiding future research toward truly physics-aware video generation. 

**Abstract (ZH)**: Text-to-video生成模型在近年来取得了显著进展，不仅在美学和准确的指令遵循方面表现出色，而且成为了数字艺术创作和在线用户参与的核心。然而，尽管取得了这些进步，它们是否遵守基本物理定律方面仍然鲜有测试：许多输出仍然违背了基本约束，如刚体碰撞、能量守恒和重力动力学，导致了不现实甚至误导的内容。现有的物理评估基准通常依赖于自动的、像素级别的指标，应用于简单的生活场景提示，因此未能考虑人类判断和第一性原理物理。为弥补这一缺口，我们引入了**T2VPhysBench**，这是一个基于第一性原理的基准，系统地评估最先进的文本到视频系统，无论是开源的还是商业的，是否遵循包括牛顿力学、守恒原则和表征效应在内的十二项核心物理定律。我们的基准采用了严格的评估协议，并包括三个专门的研究：（1）总体合规性评估，显示所有模型在每个定律类别中的平均得分低于0.60；（2）提示-线索消融实验，揭示即使详细的、针对特定定律的线索也无法纠正物理违规；（3）反事实鲁棒性测试，证明当模型被明确指示时，它们往往生成违反物理法则的视频。结果揭示了当前架构存在的持续局限性，并为指导未来研究走向真正物理感知的视频生成提供了具体的见解。 

---
# Efficient Neural Video Representation with Temporally Coherent Modulation 

**Title (ZH)**: 具有时间连贯调制的高效神经视频表示 

**Authors**: Seungjun Shin, Suji Kim, Dokwan Oh  

**Link**: [PDF](https://arxiv.org/pdf/2505.00335)  

**Abstract**: Implicit neural representations (INR) has found successful applications across diverse domains. To employ INR in real-life, it is important to speed up training. In the field of INR for video applications, the state-of-the-art approach employs grid-type parametric encoding and successfully achieves a faster encoding speed in comparison to its predecessors. However, the grid usage, which does not consider the video's dynamic nature, leads to redundant use of trainable parameters. As a result, it has significantly lower parameter efficiency and higher bitrate compared to NeRV-style methods that do not use a parametric encoding. To address the problem, we propose Neural Video representation with Temporally coherent Modulation (NVTM), a novel framework that can capture dynamic characteristics of video. By decomposing the spatio-temporal 3D video data into a set of 2D grids with flow information, NVTM enables learning video representation rapidly and uses parameter efficiently. Our framework enables to process temporally corresponding pixels at once, resulting in the fastest encoding speed for a reasonable video quality, especially when compared to the NeRV-style method, with a speed increase of over 3 times. Also, it remarks an average of 1.54dB/0.019 improvements in PSNR/LPIPS on UVG (Dynamic) (even with 10% fewer parameters) and an average of 1.84dB/0.013 improvements in PSNR/LPIPS on MCL-JCV (Dynamic), compared to previous grid-type works. By expanding this to compression tasks, we demonstrate comparable performance to video compression standards (H.264, HEVC) and recent INR approaches for video compression. Additionally, we perform extensive experiments demonstrating the superior performance of our algorithm across diverse tasks, encompassing super resolution, frame interpolation and video inpainting. Project page is this https URL. 

**Abstract (ZH)**: 基于时间一致性调制的神经视频表示（NVTM）：快速高效的学习动态视频表示 

---
# AI2-Active Safety: AI-enabled Interaction-aware Active Safety Analysis with Vehicle Dynamics 

**Title (ZH)**: AI2-主动安全：基于车辆动力学的智能交互感知主动安全分析 

**Authors**: Keshu Wu, Zihao Li, Sixu Li, Xinyue Ye, Dominique Lord, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.00322)  

**Abstract**: This paper introduces an AI-enabled, interaction-aware active safety analysis framework that accounts for groupwise vehicle interactions. Specifically, the framework employs a bicycle model-augmented with road gradient considerations-to accurately capture vehicle dynamics. In parallel, a hypergraph-based AI model is developed to predict probabilistic trajectories of ambient traffic. By integrating these two components, the framework derives vehicle intra-spacing over a 3D road surface as the solution of a stochastic ordinary differential equation, yielding high-fidelity surrogate safety measures such as time-to-collision (TTC). To demonstrate its effectiveness, the framework is analyzed using stochastic numerical methods comprising 4th-order Runge-Kutta integration and AI inference, generating probability-weighted high-fidelity TTC (HF-TTC) distributions that reflect complex multi-agent maneuvers and behavioral uncertainties. Evaluated with HF-TTC against traditional constant-velocity TTC and non-interaction-aware approaches on highway datasets, the proposed framework offers a systematic methodology for active safety analysis with enhanced potential for improving safety perception in complex traffic environments. 

**Abstract (ZH)**: 基于AI赋能和交互感知的群体车辆交互智能安全分析框架 

---
# Surrogate modeling of Cellular-Potts Agent-Based Models as a segmentation task using the U-Net neural network architecture 

**Title (ZH)**: 基于U-Net神经网络架构的细胞-质点代理模型的代理建模作为分割任务 

**Authors**: Tien Comlekoglu, J. Quetzalcóatl Toledo-Marín, Tina Comlekoglu, Douglas W. DeSimone, Shayn M. Peirce, Geoffrey Fox, James A. Glazier  

**Link**: [PDF](https://arxiv.org/pdf/2505.00316)  

**Abstract**: The Cellular-Potts model is a powerful and ubiquitous framework for developing computational models for simulating complex multicellular biological systems. Cellular-Potts models (CPMs) are often computationally expensive due to the explicit modeling of interactions among large numbers of individual model agents and diffusive fields described by partial differential equations (PDEs). In this work, we develop a convolutional neural network (CNN) surrogate model using a U-Net architecture that accounts for periodic boundary conditions. We use this model to accelerate the evaluation of a mechanistic CPM previously used to investigate \textit{in vitro} vasculogenesis. The surrogate model was trained to predict 100 computational steps ahead (Monte-Carlo steps, MCS), accelerating simulation evaluations by a factor of 590 times compared to CPM code execution. Over multiple recursive evaluations, our model effectively captures the emergent behaviors demonstrated by the original Cellular-Potts model of such as vessel sprouting, extension and anastomosis, and contraction of vascular lacunae. This approach demonstrates the potential for deep learning to serve as efficient surrogate models for CPM simulations, enabling faster evaluation of computationally expensive CPM of biological processes at greater spatial and temporal scales. 

**Abstract (ZH)**: 基于卷积神经网络的U-Net架构周期边界条件模型用于加速细胞-_Potts_模型的评价 

---
# AI-Assisted Decision-Making for Clinical Assessment of Auto-Segmented Contour Quality 

**Title (ZH)**: AI辅助决策在自动分割轮廓质量临床评估中的应用 

**Authors**: Biling Wang, Austen Maniscalco, Ti Bai, Siqiu Wang, Michael Dohopolski, Mu-Han Lin, Chenyang Shen, Dan Nguyen, Junzhou Huang, Steve Jiang, Xinlei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00308)  

**Abstract**: Purpose: This study presents a Deep Learning (DL)-based quality assessment (QA) approach for evaluating auto-generated contours (auto-contours) in radiotherapy, with emphasis on Online Adaptive Radiotherapy (OART). Leveraging Bayesian Ordinal Classification (BOC) and calibrated uncertainty thresholds, the method enables confident QA predictions without relying on ground truth contours or extensive manual labeling. Methods: We developed a BOC model to classify auto-contour quality and quantify prediction uncertainty. A calibration step was used to optimize uncertainty thresholds that meet clinical accuracy needs. The method was validated under three data scenarios: no manual labels, limited labels, and extensive labels. For rectum contours in prostate cancer, we applied geometric surrogate labels when manual labels were absent, transfer learning when limited, and direct supervision when ample labels were available. Results: The BOC model delivered robust performance across all scenarios. Fine-tuning with just 30 manual labels and calibrating with 34 subjects yielded over 90% accuracy on test data. Using the calibrated threshold, over 93% of the auto-contours' qualities were accurately predicted in over 98% of cases, reducing unnecessary manual reviews and highlighting cases needing correction. Conclusion: The proposed QA model enhances contouring efficiency in OART by reducing manual workload and enabling fast, informed clinical decisions. Through uncertainty quantification, it ensures safer, more reliable radiotherapy workflows. 

**Abstract (ZH)**: 目的：本文提出了一种基于深度学习（DL）的质量评估（QA）方法，用于评估放射治疗中的自动生成轮廓（auto-contours），特别是在在线自适应放射治疗（OART）中。该方法利用贝叶斯序贯分类（BOC）和校准的不确定性阈值，能够在无需参考标准轮廓或大量人工标注的情况下进行自信的质量评估。方法：我们开发了一个BOC模型来分类自动轮廓的质量并量化预测的不确定性。通过校准步骤，优化了满足临床准确性的不确定性阈值。该方法在三种数据场景下进行了验证：无人工标注、少量人工标注和大量人工标注。对于前列腺癌患者的直肠轮廓，当缺乏人工标注时，我们使用几何代理标签；当标注有限时，我们采用迁移学习；当标注充足时，我们直接进行监督学习。结果：BOC模型在所有场景中均表现出稳健的性能。仅使用30个人工标注进行微调并校准34个受试者后，测试数据的准确率超过90%。使用校准的阈值，超过93%的自动轮廓的质量预测准确，且在超过98%的情况下减少了不必要的手动审查，并识别了需要纠正的病例。结论：提出的质量评估模型通过减少人工工作量并促进快速、有根据的临床决策，增强了OART中的轮廓绘制效率。通过不确定性量化，确保了更安全、更可靠的放射治疗工作流程。 

---
# Fine-grained spatial-temporal perception for gas leak segmentation 

**Title (ZH)**: 细粒度时空感知的气体泄漏分割 

**Authors**: Xinlong Zhao, Shan Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.00295)  

**Abstract**: Gas leaks pose significant risks to human health and the environment. Despite long-standing concerns, there are limited methods that can efficiently and accurately detect and segment leaks due to their concealed appearance and random shapes. In this paper, we propose a Fine-grained Spatial-Temporal Perception (FGSTP) algorithm for gas leak segmentation. FGSTP captures critical motion clues across frames and integrates them with refined object features in an end-to-end network. Specifically, we first construct a correlation volume to capture motion information between consecutive frames. Then, the fine-grained perception progressively refines the object-level features using previous outputs. Finally, a decoder is employed to optimize boundary segmentation. Because there is no highly precise labeled dataset for gas leak segmentation, we manually label a gas leak video dataset, GasVid. Experimental results on GasVid demonstrate that our model excels in segmenting non-rigid objects such as gas leaks, generating the most accurate mask compared to other state-of-the-art (SOTA) models. 

**Abstract (ZH)**: 细粒度时空知觉算法（FGSTP）在天然气泄露分割中的应用 

---
# Multi-Hierarchical Fine-Grained Feature Mapping Driven by Feature Contribution for Molecular Odor Prediction 

**Title (ZH)**: 基于特征贡献的多层级细粒度特征映射分子气味预测 

**Authors**: Hong Xin Xie, Jian De Sun, Fan Fu Xue, Zi Fei Han, Shan Shan Feng, Qi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00290)  

**Abstract**: Molecular odor prediction is the process of using a molecule's structure to predict its smell. While accurate prediction remains challenging, AI models can suggest potential odors. Existing methods, however, often rely on basic descriptors or handcrafted fingerprints, which lack expressive power and hinder effective learning. Furthermore, these methods suffer from severe class imbalance, limiting the training effectiveness of AI models. To address these challenges, we propose a Feature Contribution-driven Hierarchical Multi-Feature Mapping Network (HMFNet). Specifically, we introduce a fine-grained, Local Multi-Hierarchy Feature Extraction module (LMFE) that performs deep feature extraction at the atomic level, capturing detailed features crucial for odor prediction. To enhance the extraction of discriminative atomic features, we integrate a Harmonic Modulated Feature Mapping (HMFM). This module dynamically learns feature importance and frequency modulation, improving the model's capability to capture relevant patterns. Additionally, a Global Multi-Hierarchy Feature Extraction module (GMFE) is designed to learn global features from the molecular graph topology, enabling the model to fully leverage global information and enhance its discriminative power for odor prediction. To further mitigate the issue of class imbalance, we propose a Chemically-Informed Loss (CIL). Experimental results demonstrate that our approach significantly improves performance across various deep learning models, highlighting its potential to advance molecular structure representation and accelerate the development of AI-driven technologies. 

**Abstract (ZH)**: 分子气味预测是使用分子结构预测其气味的过程。虽然准确预测仍然具有挑战性，但AI模型可以建议潜在的气味。现有方法通常依赖于基本描述符或手工地指纹，这些方法缺乏表达能力，阻碍了有效的学习。此外，这些方法还受到严重类别不平衡的影响，限制了AI模型的训练效果。为了解决这些挑战，我们提出了一种特征贡献驱动的分层多特征映射网络（HMFNet）。具体而言，我们引入了一种细粒度的局部多层次特征提取模块（LMFE），在原子级别进行深度特征提取，捕获对气味预测至关重要的详细特征。为了增强原子特征的提取，我们整合了一种谐波调制特征映射（HMFM）模块，该模块动态学习特征的重要性并进行频率调制，提高模型捕捉相关模式的能力。此外，我们设计了一种全局多层次特征提取模块（GMFE），从分子图拓扑中学习全局特征，使模型能够充分利用全局信息并增强其对气味预测的区分能力。为了进一步缓解类别不平衡问题，我们提出了一种化学信息损失函数（CIL）。实验结果表明，我们的方法显著提高了各种深度学习模型的性能，展示了其在分子结构表示和加速AI驱动技术开发方面的潜在价值。 

---
# LightEMMA: Lightweight End-to-End Multimodal Model for Autonomous Driving 

**Title (ZH)**: 轻量级端到端多模态模型LightEMMA：面向自动驾驶 

**Authors**: Zhijie Qiao, Haowei Li, Zhong Cao, Henry X. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00284)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated significant potential for end-to-end autonomous driving. However, fully exploiting their capabilities for safe and reliable vehicle control remains an open research challenge. To systematically examine advances and limitations of VLMs in driving tasks, we introduce LightEMMA, a Lightweight End-to-End Multimodal Model for Autonomous driving. LightEMMA provides a unified, VLM-based autonomous driving framework without ad hoc customizations, enabling easy integration and evaluation of evolving state-of-the-art commercial and open-source models. We construct twelve autonomous driving agents using various VLMs and evaluate their performance on the nuScenes prediction task, comprehensively assessing metrics such as inference time, computational cost, and predictive accuracy. Illustrative examples highlight that, despite their strong scenario interpretation capabilities, VLMs' practical performance in autonomous driving tasks remains concerning, emphasizing the need for further improvements. The code is available at this https URL. 

**Abstract (ZH)**: Vision-Language 模型（VLMs）在端到端自动驾驶中展现了显著的潜力。然而，全面利用其能力以实现安全可靠的车辆控制仍然是一个开放的研究挑战。为了系统地评估 VLMs 在驾驶任务中的进展和局限性，我们提出了 LightEMMA，即一个轻量级端到端多模态模型用于自动驾驶。LightEMMA 提供了一个统一的、基于 VLM 的自动驾驶框架，无需额外的定制化，使各种最新的商业和开源模型的集成和评估变得容易。我们使用多种 VLM 构建了十二个自动驾驶代理，并在 nuScenes 预测任务上评估了它们的表现，全面评估了诸如推理时间、计算成本和预测准确性等指标。示例说明尽管 VLMs 具有强大的场景解释能力，但在自动驾驶任务中的实际表现仍然令人担忧，强调了进一步改进的必要性。代码可在以下链接获取：this https URL。 

---
# Consistency in Language Models: Current Landscape, Challenges, and Future Directions 

**Title (ZH)**: 语言模型的一致性：当前概览、挑战及未来方向 

**Authors**: Jekaterina Novikova, Carol Anderson, Borhane Blili-Hamelin, Subhabrata Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2505.00268)  

**Abstract**: The hallmark of effective language use lies in consistency -- expressing similar meanings in similar contexts and avoiding contradictions. While human communication naturally demonstrates this principle, state-of-the-art language models struggle to maintain reliable consistency across different scenarios. This paper examines the landscape of consistency research in AI language systems, exploring both formal consistency (including logical rule adherence) and informal consistency (such as moral and factual coherence). We analyze current approaches to measure aspects of consistency, identify critical research gaps in standardization of definitions, multilingual assessment, and methods to improve consistency. Our findings point to an urgent need for robust benchmarks to measure and interdisciplinary approaches to ensure consistency in the application of language models on domain-specific tasks while preserving the utility and adaptability. 

**Abstract (ZH)**: 有效语言运用的标志性特征在于一致性——在类似的情境下表达相似的意义并避免矛盾。尽管人类通信自然地体现了这一原则，但最先进的语言模型在不同场景下保持可靠一致性的能力仍有待提高。本文探讨了人工智能语言系统中一致性的研究现状，探讨了正式一致性（包括逻辑规则合规性）和非正式一致性（如道德和事实的一致性）。我们分析了当前衡量一致性各个方面的方法，指出了标准化定义、多语言评估和提高一致性的方法方面的关键研究缺口。我们的研究结果指出了在特定领域任务中衡量和确保语言模型一致性的重要需求，同时保持其实用性和适应性。 

---
# Pack-PTQ: Advancing Post-training Quantization of Neural Networks by Pack-wise Reconstruction 

**Title (ZH)**: Pack-PTQ: 通过包级重建促进神经网络的后训练量化 

**Authors**: Changjun Li, Runqing Jiang, Zhuo Song, Pengpeng Yu, Ye Zhang, Yulan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.00259)  

**Abstract**: Post-training quantization (PTQ) has evolved as a prominent solution for compressing complex models, which advocates a small calibration dataset and avoids end-to-end retraining. However, most existing PTQ methods employ block-wise reconstruction, which neglects cross-block dependency and exhibits a notable accuracy drop in low-bit cases. To address these limitations, this paper presents a novel PTQ method, dubbed Pack-PTQ. First, we design a Hessian-guided adaptive packing mechanism to partition blocks into non-overlapping packs, which serve as the base unit for reconstruction, thereby preserving the cross-block dependency and enabling accurate quantization parameters estimation. Second, based on the pack configuration, we propose a mixed-precision quantization approach to assign varied bit-widths to packs according to their distinct sensitivities, thereby further enhancing performance. Extensive experiments on 2D image and 3D point cloud classification tasks, using various network architectures, demonstrate the superiority of our method over the state-of-the-art PTQ methods. 

**Abstract (ZH)**: Post-训练量化(Pack-PTQ)：一种考虑跨块依赖性的新型后训练量化方法 

---
# Empowering Agentic Video Analytics Systems with Video Language Models 

**Title (ZH)**: 赋能于视频语言模型的代理视频分析系统 

**Authors**: Yuxuan Yan, Shiqi Jiang, Ting Cao, Yifan Yang, Qianqian Yang, Yuanchao Shu, Yuqing Yang, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00254)  

**Abstract**: AI-driven video analytics has become increasingly pivotal across diverse domains. However, existing systems are often constrained to specific, predefined tasks, limiting their adaptability in open-ended analytical scenarios. The recent emergence of Video-Language Models (VLMs) as transformative technologies offers significant potential for enabling open-ended video understanding, reasoning, and analytics. Nevertheless, their limited context windows present challenges when processing ultra-long video content, which is prevalent in real-world applications. To address this, we introduce AVA, a VLM-powered system designed for open-ended, advanced video analytics. AVA incorporates two key innovations: (1) the near real-time construction of Event Knowledge Graphs (EKGs) for efficient indexing of long or continuous video streams, and (2) an agentic retrieval-generation mechanism that leverages EKGs to handle complex and diverse queries. Comprehensive evaluations on public benchmarks, LVBench and VideoMME-Long, demonstrate that AVA achieves state-of-the-art performance, attaining 62.3% and 64.1% accuracy, respectively, significantly surpassing existing VLM and video Retrieval-Augmented Generation (RAG) systems. Furthermore, to evaluate video analytics in ultra-long and open-world video scenarios, we introduce a new benchmark, AVA-100. This benchmark comprises 8 videos, each exceeding 10 hours in duration, along with 120 manually annotated, diverse, and complex question-answer pairs. On AVA-100, AVA achieves top-tier performance with an accuracy of 75.8%. 

**Abstract (ZH)**: 基于AI驱动的视频分析在多个领域中变得越来越关键。然而，现有系统往往受限于特定的预定义任务，限制了其在开放性分析场景中的适应性。最近出现的视频-语言模型（VLMs）作为一种变革性技术，提供了使开放性的视频理解、推理和分析成为可能的巨大潜力。但是，它们有限的上下文窗口在处理广泛存在的超长视频内容时带来了挑战。为解决这个问题，我们引入了AVA，这是一种基于VLM的系统，旨在实现开放性的高级视频分析。AVA结合了两项关键创新：（1）近实时构建事件知识图（EKGs）以高效索引长或连续的视频流，（2）一种代理检索-生成机制，利用EKGs处理复杂的多样查询。在公共基准LVBench和VideoMME-Long上的全面评估表明，AVA达到了最先进的性能，分别取得了62.3%和64.1%的准确率，显著优于现有的VLM和视频检索增强生成（RAG）系统。此外，为评估超长和开放世界视频场景下的视频分析，我们引入了一个新的基准AVA-100。该基准包含8个各超过10小时的视频，以及120个手动标注的多样化和复杂的问答对。在AVA-100上，AVA实现了顶级性能，准确率为75.8%。 

---
# LLM-Based Threat Detection and Prevention Framework for IoT Ecosystems 

**Title (ZH)**: 基于LLM的物联网生态系统威胁检测与预防框架 

**Authors**: Yazan Otoum, Arghavan Asad, Amiya Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2505.00240)  

**Abstract**: The increasing complexity and scale of the Internet of Things (IoT) have made security a critical concern. This paper presents a novel Large Language Model (LLM)-based framework for comprehensive threat detection and prevention in IoT environments. The system integrates lightweight LLMs fine-tuned on IoT-specific datasets (IoT-23, TON_IoT) for real-time anomaly detection and automated, context-aware mitigation strategies optimized for resource-constrained devices. A modular Docker-based deployment enables scalable and reproducible evaluation across diverse network conditions. Experimental results in simulated IoT environments demonstrate significant improvements in detection accuracy, response latency, and resource efficiency over traditional security methods. The proposed framework highlights the potential of LLM-driven, autonomous security solutions for future IoT ecosystems. 

**Abstract (ZH)**: 物联网环境中的大型语言模型驱动的全面威胁检测与预防框架 

---
# Scaling On-Device GPU Inference for Large Generative Models 

**Title (ZH)**: 在设备上扩展大型生成模型的GPU推理 

**Authors**: Jiuqiang Tang, Raman Sarokin, Ekaterina Ignasheva, Grant Jensen, Lin Chen, Juhyun Lee, Andrei Kulik, Matthias Grundmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.00232)  

**Abstract**: Driven by the advancements in generative AI, large machine learning models have revolutionized domains such as image processing, audio synthesis, and speech recognition. While server-based deployments remain the locus of peak performance, the imperative for on-device inference, necessitated by privacy and efficiency considerations, persists. Recognizing GPUs as the on-device ML accelerator with the widest reach, we present ML Drift--an optimized framework that extends the capabilities of state-of-the-art GPU-accelerated inference engines. ML Drift enables on-device execution of generative AI workloads which contain 10 to 100x more parameters than existing on-device generative AI models. ML Drift addresses intricate engineering challenges associated with cross-GPU API development, and ensures broad compatibility across mobile and desktop/laptop platforms, thereby facilitating the deployment of significantly more complex models on resource-constrained devices. Our GPU-accelerated ML/AI inference engine achieves an order-of-magnitude performance improvement relative to existing open-source GPU inference engines. 

**Abstract (ZH)**: 基于生成型AI技术的进步，大型机器学习模型在图像处理、音频合成和语音识别等领域引发了革命。虽然基于服务器的部署仍然是高性能的焦点，但由隐私和效率考量驱动的设备端推理需求仍然重要。鉴于GPU在设备端机器学习加速器中的广泛应用，我们提出了一种优化框架ML Drift，该框架扩展了最先进的GPU加速推理引擎的功能。ML Drift允许执行包含比现有设备端生成型AI模型多10到100倍参数的工作负载。ML Drift解决了跨GPU API开发的复杂工程挑战，并确保在移动和桌面/笔记本平台之间具有广泛的兼容性，从而在资源受限的设备上部署更为复杂的模型。我们的GPU加速机器学习/人工智能推理引擎在性能上比现有开源GPU推理引擎提升了数量级。 

---
# Predicting Estimated Times of Restoration for Electrical Outages Using Longitudinal Tabular Transformers 

**Title (ZH)**: 使用纵向表格变换器预测电力中断的预计恢复时间 

**Authors**: Bogireddy Sai Prasanna Teja, Valliappan Muthukaruppan, Carls Benjamin  

**Link**: [PDF](https://arxiv.org/pdf/2505.00225)  

**Abstract**: As climate variability increases, the ability of utility providers to deliver precise Estimated Times of Restoration (ETR) during natural disasters has become increasingly critical. Accurate and timely ETRs are essential for enabling customer preparedness during extended power outages, where informed decision-making can be crucial, particularly in severe weather conditions. Nonetheless, prevailing utility practices predominantly depend on manual assessments or traditional statistical methods, which often fail to achieve the level of precision required for reliable and actionable predictions. To address these limitations, we propose a Longitudinal Tabular Transformer (LTT) model that leverages historical outage event data along with sequential updates of these events to improve the accuracy of ETR predictions. The model's performance was evaluated over 34,000 storm-related outage events from three major utility companies, collectively serving over 3 million customers over a 2-year period. Results demonstrate that the LTT model improves the Customer Satisfaction Impact (CSI) metric by an average of 19.08% (p > 0.001) compared to existing methods. Additionally, we introduce customer-informed regression metrics that align model evaluation with real-world satisfaction, ensuring the outcomes resonate with customer expectations. Furthermore, we employ interpretability techniques to analyze the temporal significance of incorporating sequential updates in modeling outage events and to identify the contributions of predictive features to a given ETR. This comprehensive approach not only improves predictive accuracy but also enhances transparency, fostering greater trust in the model's capabilities. 

**Abstract (ZH)**: 随着气候变异性增加，utility提供商在自然灾害期间提供精确恢复时间估计(ETR)的能力变得越来越关键。准确及时的ETR对于帮助客户在长时间断电期间做好准备至关重要，特别是在恶劣天气条件下，基于信息的决策可以起到关键作用。然而，现有的utility做法主要依赖手工评估或传统统计方法，往往无法实现可靠且可操作的预测所需的精确度。为了解决这些限制，我们提出了一种纵向表格变换器(LTT)模型，该模型利用历史停电事件数据以及这些事件的序列更新来提高ETR预测的准确性。该模型在来自三家主要utility公司的超过34,000个风暴相关停电事件中进行了评估，这些公司共同服务了超过300万客户，评估期为两年。结果显示，与现有方法相比，LTT模型将客户满意度影响(CSI)指标平均提高了19.08%（p > 0.001）。此外，我们引入了基于客户的回归指标，确保模型评估与实际满意度相一致，从而使结果能够反映客户期望。此外，我们使用可解释性技术来分析在建模停电事件中序列更新的时间重要性，并确定预测特征对特定ETR的贡献。这种全面的方法不仅提高了预测准确性，还增强了透明度，促进了对模型能力的信任。 

---
# AI-Enhanced Automatic Design of Efficient Underwater Gliders 

**Title (ZH)**: AI增强的高效水下航行器自动设计 

**Authors**: Peter Yichen Chen, Pingchuan Ma, Niklas Hagemann, John Romanishin, Wei Wang, Daniela Rus, Wojciech Matusik  

**Link**: [PDF](https://arxiv.org/pdf/2505.00222)  

**Abstract**: The development of novel autonomous underwater gliders has been hindered by limited shape diversity, primarily due to the reliance on traditional design tools that depend heavily on manual trial and error. Building an automated design framework is challenging due to the complexities of representing glider shapes and the high computational costs associated with modeling complex solid-fluid interactions. In this work, we introduce an AI-enhanced automated computational framework designed to overcome these limitations by enabling the creation of underwater robots with non-trivial hull shapes. Our approach involves an algorithm that co-optimizes both shape and control signals, utilizing a reduced-order geometry representation and a differentiable neural-network-based fluid surrogate model. This end-to-end design workflow facilitates rapid iteration and evaluation of hydrodynamic performance, leading to the discovery of optimal and complex hull shapes across various control settings. We validate our method through wind tunnel experiments and swimming pool gliding tests, demonstrating that our computationally designed gliders surpass manually designed counterparts in terms of energy efficiency. By addressing challenges in efficient shape representation and neural fluid surrogate models, our work paves the way for the development of highly efficient underwater gliders, with implications for long-range ocean exploration and environmental monitoring. 

**Abstract (ZH)**: 基于AI增强的自动化计算框架在非平凡水下航行器外形设计中的应用：克服传统设计工具限制实现高效水下滑翔器开发 

---
# Online Federation For Mixtures of Proprietary Agents with Black-Box Encoders 

**Title (ZH)**: 在线联邦学习框架下混合私有代理的黑盒编码 

**Authors**: Xuwei Yang, Fatemeh Tavakoli, David B. Emerson, Anastasis Kratsios  

**Link**: [PDF](https://arxiv.org/pdf/2505.00216)  

**Abstract**: Most industry-standard generative AIs and feature encoders are proprietary, offering only black-box access: their outputs are observable, but their internal parameters and architectures remain hidden from the end-user. This black-box access is especially limiting when constructing mixture-of-expert type ensemble models since the user cannot optimize each proprietary AI's internal parameters. Our problem naturally lends itself to a non-competitive game-theoretic lens where each proprietary AI (agent) is inherently competing against the other AI agents, with this competition arising naturally due to their obliviousness of the AI's to their internal structure. In contrast, the user acts as a central planner trying to synchronize the ensemble of competing AIs.
We show the existence of the unique Nash equilibrium in the online setting, which we even compute in closed-form by eliciting a feedback mechanism between any given time series and the sequence generated by each (proprietary) AI agent. Our solution is implemented as a decentralized, federated-learning algorithm in which each agent optimizes their structure locally on their machine without ever releasing any internal structure to the others. We obtain refined expressions for pre-trained models such as transformers, random feature models, and echo-state networks. Our ``proprietary federated learning'' algorithm is implemented on a range of real-world and synthetic time-series benchmarks. It achieves orders-of-magnitude improvements in predictive accuracy over natural benchmarks, of which there are surprisingly few due to this natural problem still being largely unexplored. 

**Abstract (ZH)**: 大多数工业标准的生成AI和特征编码器是专有的，只提供黑盒访问：它们的输出是可观测的，但其内部参数和架构对最终用户仍然是隐藏的。这种黑盒访问在构建专家混合类型集成模型时尤为受限，因为用户无法优化每个专有AI的内部参数。我们的问题自然适用于非竞争性的博弈论视角，在这种视角下，每个专有AI（代理）本质上是在与其他AI代理竞争，这种竞争自然地产生于它们对其内部结构的无知。相比之下，用户作为中央规划者，试图同步竞争中的AI代理群。 

---
# Empirical Evaluation of Progressive Coding for Sparse Autoencoders 

**Title (ZH)**: 渐进编码在稀疏自编码器中的实证评价 

**Authors**: Hans Peter, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2505.00190)  

**Abstract**: Sparse autoencoders (SAEs) \citep{bricken2023monosemanticity,gao2024scalingevaluatingsparseautoencoders} rely on dictionary learning to extract interpretable features from neural networks at scale in an unsupervised manner, with applications to representation engineering and information retrieval. SAEs are, however, computationally expensive \citep{lieberum2024gemmascopeopensparse}, especially when multiple SAEs of different sizes are needed. We show that dictionary importance in vanilla SAEs follows a power law. We compare progressive coding based on subset pruning of SAEs -- to jointly training nested SAEs, or so-called {\em Matryoshka} SAEs \citep{bussmann2024learning,nabeshima2024Matryoshka} -- on a language modeling task. We show Matryoshka SAEs exhibit lower reconstruction loss and recaptured language modeling loss, as well as higher representational similarity. Pruned vanilla SAEs are more interpretable, however. We discuss the origins and implications of this trade-off. 

**Abstract (ZH)**: 稀疏自编码器（SAEs）依赖字典学习从神经网络中以无监督方式大规模提取可解释特征，应用于表示工程和信息检索。然而，SAEs计算成本较高，尤其是当需要多个不同规模的SAEs时。我们发现，vanilla SAEs中的字典重要性遵循幂律分布。我们将基于SAEs子集剪枝的渐进编码与联合训练嵌套SAEs或所谓的“木头娃娃”SAEs进行比较，展示了在语言建模任务中的表现。我们发现，“木头娃娃”SAEs具有更低的重建损失和捕获语言建模损失，以及更高的表示相似性。然而，剪枝的vanilla SAEs更具可解释性。我们讨论这种权衡的来源及其影响。 

---
# Neuroevolution of Self-Attention Over Proto-Objects 

**Title (ZH)**: 自注意力在原型对象上的神经进化 

**Authors**: Rafael C. Pinto, Anderson R. Tavares  

**Link**: [PDF](https://arxiv.org/pdf/2505.00186)  

**Abstract**: Proto-objects - image regions that share common visual properties - offer a promising alternative to traditional attention mechanisms based on rectangular-shaped image patches in neural networks. Although previous work demonstrated that evolving a patch-based hard-attention module alongside a controller network could achieve state-of-the-art performance in visual reinforcement learning tasks, our approach leverages image segmentation to work with higher-level features. By operating on proto-objects rather than fixed patches, we significantly reduce the representational complexity: each image decomposes into fewer proto-objects than regular patches, and each proto-object can be efficiently encoded as a compact feature vector. This enables a substantially smaller self-attention module that processes richer semantic information. Our experiments demonstrate that this proto-object-based approach matches or exceeds the state-of-the-art performance of patch-based implementations with 62% less parameters and 2.6 times less training time. 

**Abstract (ZH)**: 基于原型对象的图像区域——具有共同视觉属性的图像区域——为神经网络中传统基于矩形图像块的注意力机制提供了有前景的替代方案。尽管之前的工作证明，在控制器网络的同时演化一个基于块的硬注意力模块可以实现视觉强化学习任务中的前沿性能，我们的方法利用图像分割来处理更高层次的特征。通过操作原型对象而非固定块，我们显著降低了表示复杂性：每个图像分解为较少的原型对象，每个原型对象可以高效地编码为紧凑的特征向量。这使得自注意力模块更小且能够处理更丰富的语义信息。我们的实验表明，这种基于原型对象的方法在参数量减少62%且训练时间减少2.6倍的情况下，匹配或超越了基于块实现的前沿性能。 

---
# Attention-enabled Explainable AI for Bladder Cancer Recurrence Prediction 

**Title (ZH)**: 基于注意力机制的可解释AI在膀胱癌复发预测中的应用 

**Authors**: Saram Abbas, Naeem Soomro, Rishad Shafik, Rakesh Heer, Kabita Adhikari  

**Link**: [PDF](https://arxiv.org/pdf/2505.00171)  

**Abstract**: Non-muscle-invasive bladder cancer (NMIBC) is a relentless challenge in oncology, with recurrence rates soaring as high as 70-80%. Each recurrence triggers a cascade of invasive procedures, lifelong surveillance, and escalating healthcare costs - affecting 460,000 individuals worldwide. However, existing clinical prediction tools remain fundamentally flawed, often overestimating recurrence risk and failing to provide personalized insights for patient management. In this work, we propose an interpretable deep learning framework that integrates vector embeddings and attention mechanisms to improve NMIBC recurrence prediction performance. We incorporate vector embeddings for categorical variables such as smoking status and intravesical treatments, allowing the model to capture complex relationships between patient attributes and recurrence risk. These embeddings provide a richer representation of the data, enabling improved feature interactions and enhancing prediction performance. Our approach not only enhances performance but also provides clinicians with patient-specific insights by highlighting the most influential features contributing to recurrence risk for each patient. Our model achieves accuracy of 70% with tabular data, outperforming conventional statistical methods while providing clinician-friendly patient-level explanations through feature attention. Unlike previous studies, our approach identifies new important factors influencing recurrence, such as surgical duration and hospital stay, which had not been considered in existing NMIBC prediction models. 

**Abstract (ZH)**: 非肌肉浸润性膀胱癌（NMIBC）的持续挑战：复发率高达70-80%，每次复发都会引发一系列侵入性程序、终生监测和医疗费用急剧上升，影响着全球460,000名患者。然而，现有的临床预测工具仍存在根本性缺陷，往往会高估复发风险，无法为患者的管理提供个性化见解。在这项工作中，我们提出了一种可解释的深度学习框架，结合向量嵌入和注意力机制以提高NMIBC复发预测性能。我们将向量嵌入应用于如吸烟状态和膀胱内治疗等分类变量，使模型能够捕捉患者属性与复发风险之间的复杂关系。这些嵌入提供了数据的 richer 表示，有助于改善特征交互，从而提升预测性能。我们的方法不仅提高了性能，还通过突出显示每位患者最能影响复发风险的特征，为临床医生提供了患者特定的见解。我们的模型在表格数据上的准确率达到70%，同时通过特征注意力为临床医生提供患者层面的解释，优于传统的统计方法。与以往研究不同，我们的方法识别了新的重要影响因素，如手术时间和住院时间，这些因素在现有的NMIBC预测模型中未被考虑。 

---
# GEOM-Drugs Revisited: Toward More Chemically Accurate Benchmarks for 3D Molecule Generation 

**Title (ZH)**: GEOM-Drugs 重访：朝向更化学准确的3D分子生成基准方向 

**Authors**: Filipp Nikitin, Ian Dunn, David Ryan Koes, Olexandr Isayev  

**Link**: [PDF](https://arxiv.org/pdf/2505.00169)  

**Abstract**: Deep generative models have shown significant promise in generating valid 3D molecular structures, with the GEOM-Drugs dataset serving as a key benchmark. However, current evaluation protocols suffer from critical flaws, including incorrect valency definitions, bugs in bond order calculations, and reliance on force fields inconsistent with the reference data. In this work, we revisit GEOM-Drugs and propose a corrected evaluation framework: we identify and fix issues in data preprocessing, construct chemically accurate valency tables, and introduce a GFN2-xTB-based geometry and energy benchmark. We retrain and re-evaluate several leading models under this framework, providing updated performance metrics and practical recommendations for future benchmarking. Our results underscore the need for chemically rigorous evaluation practices in 3D molecular generation. Our recommended evaluation methods and GEOM-Drugs processing scripts are available at this https URL. 

**Abstract (ZH)**: 深度生成模型在生成有效3D分子结构方面展现了显著的潜力，GEOM-Drugs数据集是关键基准之一。然而，当前的评估协议存在严重缺陷，包括错误的价键定义、键级计算中的bug以及参考数据不一致的力场依赖。在本文中，我们重新审视了GEOM-Drugs数据集，并提出了一种修正的评估框架：我们识别并修复了数据预处理中的问题，构建了化学准确的价键表，并引入了基于GFN2-xTB的几何和能量基准。我们在该框架下重新训练和评估了多个领先模型，提供了更新的性能指标和未来基准测试的实用建议。我们的结果强调了在3D分子生成中采用化学严谨的评估实践的必要性。我们推荐的评估方法和GEOM-Drugs处理脚本可在以下链接获取：this https URL。 

---
# Detecting and Mitigating Hateful Content in Multimodal Memes with Vision-Language Models 

**Title (ZH)**: 基于视觉-语言模型检测与缓解多模态 meme 中的恶意内容 

**Authors**: Minh-Hao Van, Xintao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00150)  

**Abstract**: The rapid evolution of social media has provided enhanced communication channels for individuals to create online content, enabling them to express their thoughts and opinions. Multimodal memes, often utilized for playful or humorous expressions with visual and textual elements, are sometimes misused to disseminate hate speech against individuals or groups. While the detection of hateful memes is well-researched, developing effective methods to transform hateful content in memes remains a significant challenge. Leveraging the powerful generation and reasoning capabilities of Vision-Language Models (VLMs), we address the tasks of detecting and mitigating hateful content. This paper presents two key contributions: first, a definition-guided prompting technique for detecting hateful memes, and second, a unified framework for mitigating hateful content in memes, named UnHateMeme, which works by replacing hateful textual and/or visual components. With our definition-guided prompts, VLMs achieve impressive performance on hateful memes detection task. Furthermore, our UnHateMeme framework, integrated with VLMs, demonstrates a strong capability to convert hateful memes into non-hateful forms that meet human-level criteria for hate speech and maintain multimodal coherence between image and text. Through empirical experiments, we show the effectiveness of state-of-the-art pretrained VLMs such as LLaVA, Gemini and GPT-4o on the proposed tasks, providing a comprehensive analysis of their respective strengths and limitations for these tasks. This paper aims to shed light on important applications of VLMs for ensuring safe and respectful online environments. 

**Abstract (ZH)**: 社会媒体的迅速发展为个体提供了增强的沟通渠道，使其能够创建在线内容并表达思想和意见。多模态表情包通常用于具有视觉和文本元素的轻松或幽默表达，有时却被滥用以传播针对个人或群体的仇恨言论。虽然仇恨表情包的检测已有充分研究，但开发有效的转化仇恨内容的方法仍然是一个重大挑战。利用视觉-语言模型（VLMs）的强大生成和推理能力，我们解决了检测和减轻仇恨内容的任务。本文提出了两项关键贡献：首先，一种基于定义的提示技术用于检测仇恨表情包；其次，一个统一框架用于减轻表情包中的仇恨内容，命名为UnHateMeme，该框架通过替换仇恨的文本和/或视觉组件来工作。借助我们的基于定义的提示，VLMs在仇恨表情包检测任务中实现了令人印象深刻的性能。此外，我们的UnHateMeme框架与VLMs相结合，展示了强大的能力，将仇恨表情包转化为符合人类标准且具有图像和文本多模态一致性的人无仇恨形式。通过实证实验，我们展示了预训练的LLaVA、Gemini和GPT-4o等最先进的VLMs在所提任务中的有效性，并对这些模型在这些任务中的优缺点进行了全面分析。本文旨在揭示VLMs在确保安全和尊重的在线环境中的重要性应用。 

---
# GPRat: Gaussian Process Regression with Asynchronous Tasks 

**Title (ZH)**: GPRat: 异步任务的高斯过程回归 

**Authors**: Maksim Helmann, Alexander Strack, Dirk Pflüger  

**Link**: [PDF](https://arxiv.org/pdf/2505.00136)  

**Abstract**: Python is the de-facto language for software development in artificial intelligence (AI). Commonly used libraries, such as PyTorch and TensorFlow, rely on parallelization built into their BLAS backends to achieve speedup on CPUs. However, only applying parallelization in a low-level backend can lead to performance and scaling degradation. In this work, we present a novel way of binding task-based C++ code built on the asynchronous runtime model HPX to a high-level Python API using pybind11. We develop a parallel Gaussian process (GP) li- brary as an application. The resulting Python library GPRat combines the ease of use of commonly available GP libraries with the performance and scalability of asynchronous runtime systems. We evaluate the per- formance on a mass-spring-damper system, a standard benchmark from control theory, for varying numbers of regressors (features). The results show almost no binding overhead when binding the asynchronous HPX code using pybind11. Compared to GPyTorch and GPflow, GPRat shows superior scaling on up to 64 cores on an AMD EPYC 7742 CPU for train- ing. Furthermore, our library achieves a prediction speedup of 7.63 over GPyTorch and 25.25 over GPflow. If we increase the number of features from eight to 128, we observe speedups of 29.62 and 21.19, respectively. These results showcase the potential of using asynchronous tasks within Python-based AI applications. 

**Abstract (ZH)**: 基于异步运行时模型的C++任务绑定到高级Python API的新型方法：异步HPX与GPRat库在人工智能应用中的性能与扩展性探索 

---
# Between Underthinking and Overthinking: An Empirical Study of Reasoning Length and correctness in LLMs 

**Title (ZH)**: 在欠思考与过思考之间：对LLMs推理长度与正确性的实证研究 

**Authors**: Jinyan Su, Jennifer Healey, Preslav Nakov, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2505.00127)  

**Abstract**: Large language models (LLMs) are increasingly optimized for long reasoning, under the assumption that more reasoning leads to better performance. However, emerging evidence suggests that longer responses can sometimes degrade accuracy rather than improve it. In this paper, we conduct a systematic empirical study of the relationship between reasoning length and answer correctness. We find that LLMs tend to overthink simple problems, generating unnecessarily long outputs, and underthink harder ones, failing to extend their reasoning when it is most needed. This indicates that models might misjudge problem difficulty and fail to calibrate their response length appropriately. Furthermore, we investigate the effects of length reduction with a preference optimization algorithm when simply preferring the shorter responses regardless of answer correctness. Experiments show that the generation length can be significantly reduced while maintaining acceptable accuracy. Our findings highlight generation length as a meaningful signal for reasoning behavior and motivate further exploration into LLMs' self-awareness in reasoning length adaptation. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被优化以进行长时间推理，假设更多的推理能带来更好的性能。然而，新兴的证据表明，较长的回答有时会降低准确性而不是提高它。在本文中，我们进行了一项系统的实证研究，探讨推理长度与答案正确性之间的关系。我们发现，LLMs往往会过度思考简单问题，产生不必要的长输出，并且在更难的问题上思考不足，未能在其最需要时扩展推理。这表明模型可能错误地判断问题的难度，并未能适当校准其回答长度。此外，我们通过偏好优化算法研究了在不考虑答案正确性的情况下偏好较短回答时长度减少的影响。实验结果显示，生成长度可以显著减少同时保持可接受的准确性。我们的研究结果突出了生成长度作为推理行为的有意义信号，并激发进一步探索LLMs在推理长度调整方面的自我意识。 

---
# Fine-Tuning LLMs for Low-Resource Dialect Translation: The Case of Lebanese 

**Title (ZH)**: Fine-Tuning LLMs for Low-Resource Dialect Translation: The Case of Lebanese方言的低资源语料库迁移学习：以黎巴嫩方言为例 

**Authors**: Silvana Yakhni, Ali Chehab  

**Link**: [PDF](https://arxiv.org/pdf/2505.00114)  

**Abstract**: This paper examines the effectiveness of Large Language Models (LLMs) in translating the low-resource Lebanese dialect, focusing on the impact of culturally authentic data versus larger translated datasets. We compare three fine-tuning approaches: Basic, contrastive, and grammar-hint tuning, using open-source Aya23 models. Experiments reveal that models fine-tuned on a smaller but culturally aware Lebanese dataset (LW) consistently outperform those trained on larger, non-native data. The best results were achieved through contrastive fine-tuning paired with contrastive prompting, which indicates the benefits of exposing translation models to bad examples. In addition, to ensure authentic evaluation, we introduce LebEval, a new benchmark derived from native Lebanese content, and compare it to the existing FLoRes benchmark. Our findings challenge the "More Data is Better" paradigm and emphasize the crucial role of cultural authenticity in dialectal translation. We made our datasets and code available on Github. 

**Abstract (ZH)**: 本文考察了大型语言模型（LLMs）在翻译低资源黎巴嫩方言方面的有效性，重点关注文化内涵数据与大规模翻译数据集的影响。我们比较了三种微调方法：基础方法、对比方法和语法提示微调，使用开源Aya23模型。实验显示，使用较小但文化意识较强的黎巴嫩数据集（LW）进行微调的模型，优于使用更大规模非母语数据集进行训练的模型。最佳效果通过结合使用对比微调和对比提示实现，这表明暴露翻译模型于不良示例的益处。此外，为确保评估的真实性和准确性，我们引入了LebEval这一新的基准，该基准源自母语黎巴嫩内容，并将其与现有的FLoRes基准进行比较。我们的研究结果挑战了“数据越多越好”的观念，强调了方言翻译中文化真实性的重要性。我们已在Github上发布了数据集和代码。 

---
# Evaluating the AI-Lab Intervention: Impact on Student Perception and Use of Generative AI in Early Undergraduate Computer Science Courses 

**Title (ZH)**: 评估AI实验室干预措施：对学生对生成式AI在早期本科计算机科学课程中认知和使用影响的研究 

**Authors**: Ethan Dickey, Andres Bejarano, Rhianna Kuperus, Bárbara Fagundes  

**Link**: [PDF](https://arxiv.org/pdf/2505.00100)  

**Abstract**: Generative AI (GenAI) is rapidly entering computer science education, yet its effects on student learning, skill development, and perceptions remain underexplored. Concerns about overreliance coexist with a gap in research on structured scaffolding to guide tool use in formal courses. This study examines the impact of a dedicated "AI-Lab" intervention -- emphasizing guided scaffolding and mindful engagement -- on undergraduate students in Data Structures and Algorithms, Competitive Programming, and first-year engineering courses at Purdue University.
Over three semesters, we integrated AI-Lab modules into four mandatory and elective courses, yielding 831 matched pre- and post-intervention survey responses, alongside focus group discussions. Employing a mixed-methods approach, we analyzed quantitative shifts in usage patterns and attitudes as well as qualitative narratives of student experiences.
While the overall frequency of GenAI usage for homework or programming projects remained largely stable, we observed large effect sizes in comfort and openness across conceptual, debugging, and homework problems. Notably, usage patterns for debugging also shifted statistically significantly, reflecting students' more mindful and deliberate approach. Focus group discussions corroborated these results, suggesting that the intervention "bridged the gap" between naive GenAI usage and more nuanced, reflective integration of AI tools into coursework, ultimately heightening students' awareness of their own skill development.
These findings suggest that structured, scaffolded interventions can enable students to harness GenAI's benefits without undermining essential competencies. We offer evidence-based recommendations for educators seeking to integrate GenAI responsibly into computing curricula and identify avenues for future research on GenAI-supported pedagogy. 

**Abstract (ZH)**: 生成式人工智能（GenAI）正迅速融入计算机科学教育，但其对学生学习、技能发展和认知影响的研究仍不足。关于过度依赖的担忧与正式课程中结构化支架引导工具使用研究不足并存。本研究考察了“AI-Lab”专门干预措施——强调引导式支架和自觉参与——对学生在普渡大学数据结构与算法、编程竞赛以及大一工程课程中的影响。

在三个学期中，我们将AI-Lab模块整合到四门必修和选修课程中，获得了831份前后测问卷响应，以及焦点小组讨论的数据。采用混合方法，我们分析了使用模式和态度的定量变化以及学生的质性叙述。

虽然GenAI在家作业或编程项目中的总体使用频率保持相对稳定，但我们在概念性问题、调试问题和家庭作业问题上观察到了显著的影响大小。值得注意的是，调试阶段的使用模式也实现了统计显著的变化，反映了学生更加自觉和审慎的方法。焦点小组讨论也证实了这些结果，表明干预措施填补了简陋使用GenAI与更加细致、反思性地将AI工具整合到学业之间的差距，最终提升了学生对自己技能发展的意识。

研究结果表明，结构化的支架式干预措施能够使学生充分利用GenAI的优势，而不削弱其基本能力。我们提供了教育者在计算课程中负责任地整合GenAI方面的证据基于的建议，并指出了GenAI支持的教学方法未来研究的途径。 

---
# CoordField: Coordination Field for Agentic UAV Task Allocation In Low-altitude Urban Scenarios 

**Title (ZH)**: CoordField：低空城市场景下自主无人机任务分配的协调场 

**Authors**: Tengchao Zhang, Yonglin Tian, Fei Lin, Jun Huang, Rui Qin, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00091)  

**Abstract**: With the increasing demand for heterogeneous Unmanned Aerial Vehicle (UAV) swarms to perform complex tasks in urban environments, system design now faces major challenges, including efficient semantic understanding, flexible task planning, and the ability to dynamically adjust coordination strategies in response to evolving environmental conditions and continuously changing task requirements. To address the limitations of existing approaches, this paper proposes coordination field agentic system for coordinating heterogeneous UAV swarms in complex urban scenarios. In this system, large language models (LLMs) is responsible for interpreting high-level human instructions and converting them into executable commands for the UAV swarms, such as patrol and target tracking. Subsequently, a Coordination field mechanism is proposed to guide UAV motion and task selection, enabling decentralized and adaptive allocation of emergent tasks. A total of 50 rounds of comparative testing were conducted across different models in a 2D simulation space to evaluate their performance. Experimental results demonstrate that the proposed system achieves superior performance in terms of task coverage, response time, and adaptability to dynamic changes. 

**Abstract (ZH)**: 基于大型语言模型的协调场代理系统：用于复杂城市环境下的异构无人机群协调 

---
# Fact-Consistency Evaluation of Text-to-SQL Generation for Business Intelligence Using Exaone 3.5 

**Title (ZH)**: 使用Exaone 3.5对商务智能中文本到SQL生成的事实一致性评估 

**Authors**: Jeho Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00060)  

**Abstract**: Large Language Models (LLMs) have shown promise in enabling natural language interfaces for structured data querying through text-to-SQL generation. However, their application in real-world Business Intelligence (BI) contexts remains limited due to semantic hallucinations, structural errors, and a lack of domain-specific evaluation frameworks. In this study, we propose a Fact-Consistency Evaluation Framework for assessing the semantic accuracy of LLM-generated SQL outputs using Exaone 3.5--an instruction-tuned, bilingual LLM optimized for enterprise tasks. We construct a domain-specific benchmark comprising 219 natural language business questions across five SQL complexity levels, derived from actual sales data in LG Electronics' internal BigQuery environment. Each question is paired with a gold-standard SQL query and a validated ground-truth answer. We evaluate model performance using answer accuracy, execution success rate, semantic error rate, and non-response rate. Experimental results show that while Exaone 3.5 performs well on simple aggregation tasks (93% accuracy in L1), it exhibits substantial degradation in arithmetic reasoning (4% accuracy in H1) and grouped ranking tasks (31% in H4), with semantic errors and non-responses concentrated in complex cases. Qualitative error analysis further identifies common failure types such as misapplied arithmetic logic, incomplete filtering, and incorrect grouping operations. Our findings highlight the current limitations of LLMs in business-critical environments and underscore the need for fact-consistency validation layers and hybrid reasoning approaches. This work contributes a reproducible benchmark and evaluation methodology for advancing reliable natural language interfaces to structured enterprise data systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通过文本生成SQL查询实现结构化数据查询自然语言接口方面显示出潜力。然而，它们在实际商业智能（BI）环境中的应用受限于语义幻觉、结构错误以及缺乏特定领域的评估框架。在本研究中，我们提出了一种基于Exaone 3.5的语义一致性评估框架，用于评估LLM生成的SQL输出的语义准确性，Exaone 3.5是一种指令调优的双语LLM，专为企业任务优化。我们构建了一个包含219个覆盖五种SQL复杂度级别的自然语言企业问题的特定领域基准，这些问题是根据LG电子内部BigQuery环境的实际销售数据得出的。每个问题都配有一个金标准SQL查询和一个验证过的正确答案。我们使用答案准确性、执行成功率、语义错误率和无回应率来评估模型性能。实验结果表明，Exaone 3.5在简单聚合任务上表现良好（L1准确率为93%），但在算术推理（H1准确率为4%）和分组排序任务（H4准确率为31%）上表现出显著下降，语义错误和无回应率集中在复杂情况中。定性的错误分析进一步揭示了常见的错误类型，如错误应用算术逻辑、不完整的过滤和不正确的分组操作。我们的研究结果突显了LLMs在关键商业环境中的当前局限性，并强调了需要事实一致性验证层和混合推理方法的重要性。本研究为推进可靠自然语言接口到结构化企业数据系统提供了可重复基准和评估方法。 

---
# Emotional Analysis of Fashion Trends Using Social Media and AI: Sentiment Analysis on Twitter for Fashion Trend Forecasting 

**Title (ZH)**: 基于社交媒体和AI的情感分析：微博上的情感分析在时尚趋势预测中的应用 

**Authors**: Aayam Bansal, Agneya Tharun  

**Link**: [PDF](https://arxiv.org/pdf/2505.00050)  

**Abstract**: This study explores the intersection of fashion trends and social media sentiment through computational analysis of Twitter data using the T4SA (Twitter for Sentiment Analysis) dataset. By applying natural language processing and machine learning techniques, we examine how sentiment patterns in fashion-related social media conversations can serve as predictors for emerging fashion trends. Our analysis involves the identification and categorization of fashion-related content, sentiment classification with improved normalization techniques, time series decomposition, statistically validated causal relationship modeling, cross-platform sentiment comparison, and brand-specific sentiment analysis. Results indicate correlations between sentiment patterns and fashion theme popularity, with accessories and streetwear themes showing statistically significant rising trends. The Granger causality analysis establishes sustainability and streetwear as primary trend drivers, showing bidirectional relationships with several other themes. The findings demonstrate that social media sentiment analysis can serve as an effective early indicator of fashion trend trajectories when proper statistical validation is applied. Our improved predictive model achieved 78.35% balanced accuracy in sentiment classification, establishing a reliable foundation for trend prediction across positive, neutral, and negative sentiment categories. 

**Abstract (ZH)**: 本研究通过计算分析Twitter数据（使用T4SA（Twitter for Sentiment Analysis）数据集）探索时尚趋势与社交媒体情感的交集。通过对自然语言处理和机器学习技术的应用，我们研究了与时尚相关的社交媒体对话中的情感模式如何成为新兴时尚趋势的预测指标。分析涉及时尚相关内容的识别和分类、改进规范化技术的情感分类、时间序列分解、经过统计验证的因果关系建模、跨平台情感比较以及品牌特定的情感分析。结果表明，情感模式与时尚主题流行度之间存在相关性，配饰和街头潮流主题显示出统计上显著的增长趋势。遍历因果分析确立了可持续性和街头潮流为主要趋势驱动因素，并与多种其他主题之间存在双向关系。研究结果表明，在应用适当的统计验证时，社交媒体情感分析可作为时尚趋势轨迹的有效早期指标。改进的预测模型在情感分类中的平衡准确率达到78.35%，为在正面、中性和负面情感类别中预测趋势奠定了可靠的基础。 

---
# Convolutional Autoencoders for Data Compression and Anomaly Detection in Small Satellite Technologies 

**Title (ZH)**: 卷积自编码器在小卫星技术中的数据压缩与异常检测 

**Authors**: Dishanand Jayeprokash, Julia Gonski  

**Link**: [PDF](https://arxiv.org/pdf/2505.00040)  

**Abstract**: Small satellite technologies have enhanced the potential and feasibility of geodesic missions, through simplification of design and decreased costs allowing for more frequent launches. On-satellite data acquisition systems can benefit from the implementation of machine learning (ML), for better performance and greater efficiency on tasks such as image processing or feature extraction. This work presents convolutional autoencoders for implementation on the payload of small satellites, designed to achieve dual functionality of data compression for more efficient off-satellite transmission, and at-source anomaly detection to inform satellite data-taking. This capability is demonstrated for a use case of disaster monitoring using aerial image datasets of the African continent, offering avenues for both novel ML-based approaches in small satellite applications along with the expansion of space technology and artificial intelligence in Africa. 

**Abstract (ZH)**: 小卫星技术通过简化设计和降低发射成本，增强了地理测量任务的潜力和可行性，车上数据获取系统可通过实施机器学习实现更好的性能和更高的效率，用于图像处理或特征提取任务。本文提出在小卫星载荷中实现卷积自编码器，旨在实现数据压缩以提高离轨传输效率，并在源头实现异常检测以指导卫星数据采集。这一能力通过使用非洲大陆航空图像数据集进行灾害监测的应用案例得以展示，也为小型卫星应用中新型机器学习方法以及非洲的空间技术和人工智能扩展提供了契机。 

---
# Linguistic Complexity and Socio-cultural Patterns in Hip-Hop Lyrics 

**Title (ZH)**: 汉语语言复杂性与嘻哈歌词中的社会文化模式 

**Authors**: Aayam Bansal, Raghav Agarwal, Kaashvi Jain  

**Link**: [PDF](https://arxiv.org/pdf/2505.00035)  

**Abstract**: This paper presents a comprehensive computational framework for analyzing linguistic complexity and socio-cultural trends in hip-hop lyrics. Using a dataset of 3,814 songs from 146 influential artists spanning four decades (1980-2020), we employ natural language processing techniques to quantify multiple dimensions of lyrical complexity. Our analysis reveals a 23.7% increase in vocabulary diversity over the study period, with East Coast artists demonstrating 17.3% higher lexical variation than other regions. Rhyme density increased by 34.2% across all regions, with Midwest artists exhibiting the highest technical complexity (3.04 rhymes per line). Topic modeling identified significant shifts in thematic content, with social justice themes decreasing from 28.5% to 13.8% of content while introspective themes increased from 7.6% to 26.3%. Sentiment analysis demon- strated that lyrics became significantly more negative during sociopolitical crises, with polarity decreasing by 0.31 following major social unrest. Multi-dimensional analysis revealed four dis- tinct stylistic approaches that correlate strongly with geographic origin (r=0.68, p!0.001) and time period (r=0.59, p<0.001). These findings establish quantitative evidence for the evolution of hip- hop as both an art form and a reflection of societal dynamics, providing insights into the interplay between linguistic innovation and cultural context in popular music. 

**Abstract (ZH)**: 本文 Presents 一个全面的计算框架，用于分析嘻哈歌词中的语言复杂性和社会文化趋势。利用1980-2020年四个年代、146位有影响力的艺术家的3,814首歌曲数据集，我们运用自然语言处理技术量化歌词复杂性的多个维度。分析结果显示，研究期间词汇多样性增加了23.7%，而东海岸艺术家的词汇变化率比其他地区高17.3%。韵律密度在所有地区增加了34.2%，中西部艺术家表现出最高的技术复杂度（每行3.04个韵脚）。主题建模揭示了主题内容的显著变化，社会正义主题从内容的28.5%下降到13.8%，而 introspective 主题从7.6%增加到26.3%。情感分析表明，在社会政治危机期间，歌词变得显著更具负面性，主要社会动荡后极性下降了0.31。多维分析揭示了四种与地理起源（r=0.68，p<0.001）和时间时期（r=0.59，p<0.001）高度相关的独特风格方法。这些发现为嘻哈作为一种艺术形式及其对社会动态的反映的演变提供了定量证据，提供了关于语言创新与文化语境在流行音乐中相互作用的洞见。 

---
# Improving Phishing Email Detection Performance of Small Large Language Models 

**Title (ZH)**: 提升小型大型语言模型 phishing 邮件检测性能 

**Authors**: Zijie Lin, Zikang Liu, Hanbo Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00034)  

**Abstract**: Large language models(LLMs) have demonstrated remarkable performance on many natural language processing(NLP) tasks and have been employed in phishing email detection research. However, in current studies, well-performing LLMs typically contain billions or even tens of billions of parameters, requiring enormous computational resources. To reduce computational costs, we investigated the effectiveness of small-parameter LLMs for phishing email detection. These LLMs have around 3 billion parameters and can run on consumer-grade GPUs. However, small LLMs often perform poorly in phishing email detection task. To address these issues, we designed a set of methods including Prompt Engineering, Explanation Augmented Fine-tuning, and Model Ensemble to improve phishing email detection capabilities of small LLMs. We validated the effectiveness of our approach through experiments, significantly improving accuracy on the SpamAssassin dataset from around 0.5 for baseline models like Qwen2.5-1.5B-Instruct to 0.976. 

**Abstract (ZH)**: 小型参数大型语言模型在钓鱼邮件检测中的有效性研究及其改进方法 

---
# MDD-LLM: Towards Accuracy Large Language Models for Major Depressive Disorder Diagnosis 

**Title (ZH)**: MDD-LLM：面向重大抑郁障碍诊断的大规模语言模型准确性提升 

**Authors**: Yuyang Sha, Hongxin Pan, Wei Xu, Weiyu Meng, Gang Luo, Xinyu Du, Xiaobing Zhai, Henry H. Y. Tong, Caijuan Shi, Kefeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.00032)  

**Abstract**: Major depressive disorder (MDD) impacts more than 300 million people worldwide, highlighting a significant public health issue. However, the uneven distribution of medical resources and the complexity of diagnostic methods have resulted in inadequate attention to this disorder in numerous countries and regions. This paper introduces a high-performance MDD diagnosis tool named MDD-LLM, an AI-driven framework that utilizes fine-tuned large language models (LLMs) and extensive real-world samples to tackle challenges in MDD diagnosis. Therefore, we select 274,348 individual information from the UK Biobank cohort to train and evaluate the proposed method. Specifically, we select 274,348 individual records from the UK Biobank cohort and design a tabular data transformation method to create a large corpus for training and evaluating the proposed approach. To illustrate the advantages of MDD-LLM, we perform comprehensive experiments and provide several comparative analyses against existing model-based solutions across multiple evaluation metrics. Experimental results show that MDD-LLM (70B) achieves an accuracy of 0.8378 and an AUC of 0.8919 (95% CI: 0.8799 - 0.9040), significantly outperforming existing machine learning and deep learning frameworks for MDD diagnosis. Given the limited exploration of LLMs in MDD diagnosis, we examine numerous factors that may influence the performance of our proposed method, such as tabular data transformation techniques and different fine-tuning strategies. 

**Abstract (ZH)**: 重大抑郁障碍（MDD）影响全球超过30亿人，凸显出一个重要的公共健康问题。然而，医疗资源的不均衡分布和诊断方法的复杂性导致了这一障碍在众多国家和地区缺乏足够的重视。本文介绍了一种高性能的MDD诊断工具——MDD-LLM，这是一种基于AI的框架，利用微调的大语言模型（LLMs）和大量的实际样本来应对MDD诊断的挑战。因此，我们从英国生物银行队列中选择了274,348个个体信息进行训练和评估。具体而言，我们从英国生物银行队列中选择了274,348个个体记录，并设计了一种表格数据转换方法，以创建训练和评估所提出方法的大规模语料库。为了说明MDD-LLM的优势，我们进行了全面的实验，并在多个评估指标上与现有的模型解决方案进行了多个对比分析。实验结果显示，MDD-LLM（70B）的准确率为0.8378，AUC为0.8919（95% CI：0.8799 - 0.9040），显著优于现有的机器学习和深度学习框架在MDD诊断中的表现。鉴于在MDD诊断中对LLMs的有限探索，我们研究了可能影响所提出方法性能的多种因素，包括表格数据转换技术和不同的微调策略。 

---
# Learning to Plan Before Answering: Self-Teaching LLMs to Learn Abstract Plans for Problem Solving 

**Title (ZH)**: 学习在回答之前规划：自我教学的LLM学习抽象计划以解决问题 

**Authors**: Jin Zhang, Flood Sung, Zhilin Yang, Yang Gao, Chongjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00031)  

**Abstract**: In the field of large language model (LLM) post-training, the effectiveness of utilizing synthetic data generated by the LLM itself has been well-presented. However, a key question remains unaddressed: what essential information should such self-generated data encapsulate? Existing approaches only produce step-by-step problem solutions, and fail to capture the abstract meta-knowledge necessary for generalization across similar problems. Drawing insights from cognitive science, where humans employ high-level abstraction to simplify complex problems before delving into specifics, we introduce a novel self-training algorithm: LEarning to Plan before Answering (LEPA). LEPA trains the LLM to formulate anticipatory plans, which serve as abstract meta-knowledge for problem-solving, before engaging with the intricacies of problems. This approach not only outlines the solution generation path but also shields the LLM from the distraction of irrelevant details. During data generation, LEPA first crafts an anticipatory plan based on the problem, and then generates a solution that aligns with both the plan and the problem. LEPA refines the plan through self-reflection, aiming to acquire plans that are instrumental in yielding correct solutions. During model optimization, the LLM is trained to predict both the refined plans and the corresponding solutions. By efficiently extracting and utilizing the anticipatory plans, LEPA demonstrates remarkable superiority over conventional algorithms on various challenging natural language reasoning benchmarks. 

**Abstract (ZH)**: 在大型语言模型（LLM）后训练领域，利用LLM本身生成的合成数据的有效性已被充分展示。然而，一个关键问题仍未得到解决：这种自动生成的数据应包含哪些本质信息？现有方法仅生成逐步问题解决方案，未能捕捉跨类似问题泛化的抽象元知识。借鉴认知科学的见解，人类在深入具体问题之前先运用高层次抽象简化复杂问题，我们提出了一种新颖的自训练算法：LEarning to Plan before Answering (LEPA)。LEPA训练LLM在处理具体问题之前制定预见性计划，这些计划作为问题解决的抽象元知识。该方法不仅明确了解决方案生成路径，还使LLM免受无关细节的干扰。在数据生成过程中，LEPA首先基于问题制定预见性计划，然后生成符合计划和问题的解决方案。LEPA通过自我反思精炼计划，以获得有助于产生正确解决方案的计划。在模型优化过程中，LLM被训练预测精炼的计划及其相应的解决方案。通过高效提取和利用预见性计划，LEPA在各种具有挑战性的自然语言推理基准测试中展现出了显著优势。 

---
# Keep the General, Inject the Specific: Structured Dialogue Fine-Tuning for Knowledge Injection without Catastrophic Forgetting 

**Title (ZH)**: 保持通用性，注入特定性：结构化对话微调以实现知识注入且无灾难性遗忘 

**Authors**: Yijie Hong, Xiaofei Yin, Xinzhong Wang, Yi Tu, Ya Guo, Sufeng Duan, Weiqiang Wang, Lingyong Fang, Depeng Wang, Huijia Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00029)  

**Abstract**: Large Vision Language Models have demonstrated impressive versatile capabilities through extensive multimodal pre-training, but face significant limitations when incorporating specialized knowledge domains beyond their training distribution. These models struggle with a fundamental dilemma: direct adaptation approaches that inject domain-specific knowledge often trigger catastrophic forgetting of foundational visual-linguistic abilities. We introduce Structured Dialogue Fine-Tuning (SDFT), an effective approach that effectively injects domain-specific knowledge while minimizing catastrophic forgetting. Drawing inspiration from supervised fine-tuning in LLMs and subject-driven personalization in text-to-image diffusion models, our method employs a three-phase dialogue structure: Foundation Preservation reinforces pre-trained visual-linguistic alignment through caption tasks; Contrastive Disambiguation introduces carefully designed counterfactual examples to maintain semantic boundaries; and Knowledge Specialization embeds specialized information through chain-of-thought reasoning. Experimental results across multiple domains confirm SDFT's effectiveness in balancing specialized knowledge acquisition with general capability retention. Our key contributions include a data-centric dialogue template that balances foundational alignment with targeted knowledge integration, a weighted multi-turn supervision framework, and comprehensive evaluation across diverse knowledge types. 

**Abstract (ZH)**: 大规模视觉语言模型通过广泛的多模态预训练展示了令人印象深刻的多功能能力，但在融入超出其训练分布的专门知识领域时面临显著限制。这些模型在根本上面临一个难题：直接适应方法虽然可以注入领域特定知识，但往往会引发对基础视觉-语言能力的灾难性遗忘。我们引入了结构化对话微调（SDFT），这是一种有效的方法，它能够有效注入专门知识，同时最大限度地减少灾难性遗忘。该方法受到大规模语言模型的监督微调和文本到图像扩散模型的主题驱动个性化启发，采用三个阶段的对话结构：基础保护通过字幕任务强化预训练的视觉-语言对齐；对比去模糊通过引入精心设计的反事实示例维持语义边界；知识专业化通过链式推理嵌入专门信息。实验结果在多个领域证实了SDFT在专业知识学习与保持一般能力之间的平衡效果。我们的主要贡献包括以数据为中心的对话模板，平衡基础对齐与目标知识整合，加权多轮监督框架，以及跨多种知识类型进行全面评估。 

---
# Enhancing Speech-to-Speech Dialogue Modeling with End-to-End Retrieval-Augmented Generation 

**Title (ZH)**: 增强端到端检索增强生成的语音到语音对话建模 

**Authors**: Pengchao Feng, Ziyang Ma, Wenxi Chen, Yao Li, Sheng Wang, Kai Yu, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00028)  

**Abstract**: In recent years, end-to-end speech-to-speech (S2S) dialogue systems have garnered increasing research attention due to their advantages over traditional cascaded systems, including achieving lower latency and more natural integration of nonverbal cues such as emotion and speaker identity. However, these end-to-end systems face key challenges, particularly in incorporating external knowledge, a capability commonly addressed by Retrieval-Augmented Generation (RAG) in text-based large language models (LLMs). The core difficulty lies in the modality gap between input speech and retrieved textual knowledge, which hinders effective integration. To address this issue, we propose a novel end-to-end RAG framework that directly retrieves relevant textual knowledge from speech queries, eliminating the need for intermediate speech-to-text conversion via techniques like ASR. Experimental results demonstrate that our method significantly improves the performance of end-to-end S2S dialogue systems while achieving higher retrieval efficiency. Although the overall performance still lags behind cascaded models, our framework offers a promising direction for enhancing knowledge integration in end-to-end S2S systems. We will release the code and dataset to support reproducibility and promote further research in this area. 

**Abstract (ZH)**: 近年来，端到端语音到语音（S2S）对话系统由于其优于传统级联系统的优点，包括较低的延迟和更自然地整合诸如情绪和说话人身份等非言语线索，引起了越来越多的研究关注。然而，这些端到端系统面临着关键挑战，特别是在融入外部知识方面，这一能力通常由基于文本的大语言模型（LLMs）中的检索增强生成（RAG）解决。核心难点在于输入语音与检索到的文本知识之间的模态差距，这妨碍了有效的整合。为了解决这个问题，我们提出了一种新的端到端RAG框架，该框架可以直接从语音查询中检索相关文本知识，从而消除通过ASR等技术介导的语音到文本转换的需要。实验结果表明，我们的方法显著提高了端到端S2S对话系统的性能，同时实现了更高的检索效率。虽然整体性能仍落后于级联模型，但我们的框架为增强端到端S2S系统中的知识整合提供了有前景的方向。我们将发布代码和数据集以支持可再现性并促进该领域的进一步研究。 

---
# Extracting Abstraction Dimensions by Identifying Syntax Pattern from Texts 

**Title (ZH)**: 从文本中识别语法模式提取抽象维度 

**Authors**: Jian Zhou, Jiazheng Li, Sirui Zhuge, Hai Zhuge  

**Link**: [PDF](https://arxiv.org/pdf/2505.00027)  

**Abstract**: This paper proposed an approach to automatically discovering subject dimension, action dimension, object dimension and adverbial dimension from texts to efficiently operate texts and support query in natural language. The high quality of trees guarantees that all subjects, actions, objects and adverbials and their subclass relations within texts can be represented. The independency of trees ensures that there is no redundant representation between trees. The expressiveness of trees ensures that the majority of sentences can be accessed from each tree and the rest of sentences can be accessed from at least one tree so that the tree-based search mechanism can support querying in natural language. Experiments show that the average precision, recall and F1-score of the abstraction trees constructed by the subclass relations of subject, action, object and adverbial are all greater than 80%. The application of the proposed approach to supporting query in natural language demonstrates that different types of question patterns for querying subject or object have high coverage of texts, and searching multiple trees on subject, action, object and adverbial according to the question pattern can quickly reduce search space to locate target sentences, which can support precise operation on texts. 

**Abstract (ZH)**: 本文提出了一种自动发现主题维度、动作维度、对象维度和状语维度的方法，以高效地操作文本并支持自然语言查询。高质量的树结构保证了文本中所有主题、动作、对象和状语及其子类关系能够被表示。树的独立性保证了树之间没有冗余表示。树的表达能力确保大多数句子可以从每棵树中访问到，剩余的句子可以从至少一棵树中访问到，从而支持基于树的查询机制的自然语言查询。实验结果表明，由主题、动作、对象和状语的子类关系构建的抽象树的平均精度、召回率和F1分数均大于80%。所提出的方法应用于支持自然语言查询中，展示了不同类型的查询模式在查询主题或对象时对文本的高覆盖度，并且根据查询模式在主题、动作、对象和状语上的多棵树搜索可以迅速减少搜索空间，定位目标句子，从而支持对文本的精确操作。 

---
# Theory of Mind in Large Language Models: Assessment and Enhancement 

**Title (ZH)**: 大型语言模型中的理论思维：评估与增强 

**Authors**: Ruirui Chen, Weifeng Jiang, Chengwei Qin, Cheston Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00026)  

**Abstract**: Theory of Mind (ToM)-the ability to infer and reason about others' mental states-is fundamental to human social intelligence. As Large Language Models (LLMs) become increasingly integrated into daily life, it is crucial to assess and enhance their capacity to interpret and respond to human mental states. In this paper, we review LLMs' ToM capabilities by examining both evaluation benchmarks and the strategies designed to improve them. We focus on widely adopted story-based benchmarks and provide an in-depth analysis of methods aimed at enhancing ToM in LLMs. Furthermore, we outline promising future research directions informed by recent benchmarks and state-of-the-art approaches. Our survey serves as a valuable resource for researchers interested in advancing LLMs' ToM capabilities. 

**Abstract (ZH)**: Theory of Mind (ToM)能力——推断和理解他人心理状态的能力——是人类社会智能的基础。随着大型语言模型（LLMs）越来越多地融入日常生活，评估并增强其解读和回应人类心理状态的能力变得至关重要。在本文中，我们通过分析评估基准和提高这些基准的方法来审查LLMs的ToM能力。我们着重探讨广泛采用的故事基准，并对旨在提高LLMs ToM能力的方法进行了深入分析。此外，我们概述了受近期基准和最新方法启发的有 promise 的未来研究方向。我们的综述为致力于推进LLMs ToM能力的研究人员提供了一项宝贵资源。 

---
# A Method for the Architecture of a Medical Vertical Large Language Model Based on Deepseek R1 

**Title (ZH)**: 基于DeepSeek R1的医疗垂直大语言模型架构方法 

**Authors**: Mingda Zhang, Jianglong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2505.00025)  

**Abstract**: In recent years, despite foundation models like DeepSeek-R1 and ChatGPT demonstrating significant capabilities in general tasks, professional knowledge barriers, computational resource requirements, and deployment environment limitations have severely hindered their application in actual medical scenarios. Addressing these challenges, this paper proposes an efficient lightweight medical vertical large language model architecture method, systematically solving the lightweight problem of medical large models from three dimensions: knowledge acquisition, model compression, and computational optimization. At the knowledge acquisition level, a knowledge transfer pipeline is designed from the fine-tuned DeepSeek-R1-Distill-70B teacher model to the DeepSeek-R1-Distill-7B student model, and Low-Rank Adaptation (LoRA) technology is adopted to precisely adjust key attention layers. At the model compression level, compression techniques including 4-bit weight quantization are implemented while preserving the core representation ability for medical reasoning. At the computational optimization level, inference optimization techniques such as Flash Attention acceleration and continuous batching are integrated, and a professional prompt template system is constructed to adapt to different types of medical problems. Experimental results on medical question-answering datasets show that the method proposed in this paper maintains professional accuracy while reducing memory consumption by 64.7\% and inference latency by 12.4\%, providing an effective solution for the application of medical large models in resource-constrained environments such as edge computing devices. 

**Abstract (ZH)**: 近年来，尽管基础模型如DeepSeek-R1和ChatGPT在通用任务中展现了显著的能力，但专业性知识障碍、计算资源需求和部署环境限制严重阻碍了它们在实际医疗场景中的应用。为应对这些挑战，本文提出了一种高效轻量级医疗垂直大型语言模型架构方法，从三个维度系统地解决了医疗大型模型的轻量化问题：知识获取、模型压缩和计算优化。在知识获取层面，设计了一条从fine-tuned DeepSeek-R1-Distill-70B教师模型到DeepSeek-R1-Distill-7B学生模型的知识转换管道，并采用低秩适应（LoRA）技术精确调整关键注意力层。在模型压缩层面，实现包括4位权重量化在内的压缩技术，同时保留核心表示能力以支持医疗推理。在计算优化层面，集成推理优化技术如Flash Attention加速和连续批量处理，并构建了一个专业的提示模板系统以适应不同类型医疗问题。实验结果表明，本方法在维持专业准确性的同时，减少了64.7%的内存消耗和12.4%的推理延迟，为边缘计算设备等资源受限环境中的医疗大型模型应用提供了有效解决方案。 

---
# Nemotron-Research-Tool-N1: Tool-Using Language Models with Reinforced Reasoning 

**Title (ZH)**: Nemotron-研究工具N1：配备强化推理的工具使用语言模型 

**Authors**: Shaokun Zhang, Yi Dong, Jieyu Zhang, Jan Kautz, Bryan Catanzaro, Andrew Tao, Qingyun Wu, Zhiding Yu, Guilin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00024)  

**Abstract**: Enabling large language models with external tools has become a pivotal strategy for extending their functionality beyond text generation tasks. Prior work typically enhances tool-use abilities by either applying supervised fine-tuning (SFT) to enforce tool-call correctness or distilling reasoning traces from stronger models for SFT. However, both approaches fall short, either omitting reasoning entirely or producing imitative reasoning that limits generalization. Inspired by the success of DeepSeek-R1 in eliciting reasoning through rule-based reinforcement learning, we develop the Nemotron-Research-Tool-N1 series of tool-using language models using a similar training paradigm. Instead of restrictively supervising intermediate reasoning traces distilled from stronger models, Nemotron-Research-Tool-N1 is optimized with a binary reward that evaluates only the structural validity and functional correctness of tool invocations. This lightweight supervision allows the model to autonomously internalize reasoning strategies, without the need for annotated reasoning trajectories. Experiments on the BFCL and API-Bank benchmarks show that Nemotron-Research-Tool-N1-7B and Nemotron-Research-Tool-N1-14B, built on Qwen-2.5-7B/14B-Instruct, achieve state-of-the-art results, outperforming GPT-4o on both evaluations. 

**Abstract (ZH)**: 外部工具赋能的大语言模型已成为扩展其功能超越文本生成任务的关键策略。Nemotron-Research-Tool-N1系列工具使用语言模型通过类似训练范式借鉴DeepSeek-R1的成功经验，以二元奖励优化工具调用的结构有效性和功能正确性，实现轻量级监督，促进模型自主内化推理策略。实验表明，Nemotron-Research-Tool-N1-7B和Nemotron-Research-Tool-N1-14B在BFCL和API-Bank基准上的表现卓越，优于GPT-4o。 

---
# CORG: Generating Answers from Complex, Interrelated Contexts 

**Title (ZH)**: CORG: 从复杂相关背景中生成答案 

**Authors**: Hyunji Lee, Franck Dernoncourt, Trung Bui, Seunghyun Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2505.00023)  

**Abstract**: In a real-world corpus, knowledge frequently recurs across documents but often contains inconsistencies due to ambiguous naming, outdated information, or errors, leading to complex interrelationships between contexts. Previous research has shown that language models struggle with these complexities, typically focusing on single factors in isolation. We classify these relationships into four types: distracting, ambiguous, counterfactual, and duplicated. Our analysis reveals that no single approach effectively addresses all these interrelationships simultaneously. Therefore, we introduce Context Organizer (CORG), a framework that organizes multiple contexts into independently processed groups. This design allows the model to efficiently find all relevant answers while ensuring disambiguation. CORG consists of three key components: a graph constructor, a reranker, and an aggregator. Our results demonstrate that CORG balances performance and efficiency effectively, outperforming existing grouping methods and achieving comparable results to more computationally intensive, single-context approaches. 

**Abstract (ZH)**: 在真实世界的语料库中，知识跨越多个文档频繁出现，但由于命名模糊、信息过时或错误，常常包含不一致性，导致上下文之间关系复杂。先前的研究表明，语言模型在处理这些复杂性时表现出色，通常侧重于孤立地考虑单一因素。我们将这些关系分类为四种类型：干扰性、模糊性、假设性逆反和重复。我们的分析揭示，没有单一的方法能够同时有效解决所有这些关系。因此，我们提出了Context Organizer (CORG)框架，该框架将多个上下文组织成独立处理的组。该设计允许模型高效地找到所有相关答案并确保去模糊化。CORG由三个关键组件组成：图构建器、重排序器和聚合器。我们的结果显示，CORG能够在性能和效率之间取得有效平衡，优于现有分组方法，并达到与计算密集型单一上下文方法可比的结果。 

---
# Aleph-Alpha-GermanWeb: Improving German-language LLM pre-training with model-based data curation and synthetic data generation 

**Title (ZH)**: Aleph-Alpha-GermanWeb：基于模型的数据筛选和合成数据生成以提高德语语言大模型预训练 

**Authors**: Thomas F Burns, Letitia Parcalabescu, Stephan Wäldchen, Michael Barlow, Gregor Ziegltrum, Volker Stampa, Bastian Harren, Björn Deiseroth  

**Link**: [PDF](https://arxiv.org/pdf/2505.00022)  

**Abstract**: Scaling data quantity is essential for large language models (LLMs), yet recent findings show that data quality can significantly boost performance and training efficiency. We introduce a German-language dataset curation pipeline that combines heuristic and model-based filtering techniques with synthetic data generation. We use our pipeline to create Aleph-Alpha-GermanWeb, a large-scale German pre-training dataset which draws from: (1) Common Crawl web data, (2) FineWeb2, and (3) synthetically-generated data conditioned on actual, organic web data. We evaluate our dataset by pre-training both a 1B Llama-style model and an 8B tokenizer-free hierarchical autoregressive transformer (HAT). A comparison on German-language benchmarks, including MMMLU, shows significant performance gains of Aleph-Alpha-GermanWeb over FineWeb2 alone. This advantage holds at the 8B scale even when FineWeb2 is enriched by human-curated high-quality data sources such as Wikipedia. Our findings support the growing body of evidence that model-based data curation and synthetic data generation can significantly enhance LLM pre-training datasets. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的数据量扩展是必要的，然而近期的研究显示，数据质量可以显著提升性能和训练效率。我们提出了一种德语数据集编纂流水线，该流水线结合了启发式和模型基础的过滤技术以及合成数据生成方法。我们利用该流水线创建了Aleph-Alpha-GermanWeb，这是一个大规模的德语预训练数据集，来源于：（1）Common Crawl网页数据，（2）FineWeb2，以及（3）基于实际有机网页数据生成的合成数据。我们通过预训练一个1B的Llama风格模型和一个8B的Tokenizer-Free层次自回归变压器（HAT）来评估我们的数据集。在包括MMMLU等德语基准测试中的比较表明，Aleph-Alpha-GermanWeb在性能上明显优于仅使用FineWeb2的数据。即使FineWeb2通过包含Wikipedia等高质量的人工筛选数据源而得到丰富，这一优势在8B规模下仍然存在。我们的研究结果支持了现有越来越多的证据，即基于模型的数据编纂和合成数据生成可以显著增强LLM预训练数据集的质量。 

---
# Ustnlp16 at SemEval-2025 Task 9: Improving Model Performance through Imbalance Handling and Focal Loss 

**Title (ZH)**: Ustnlp16 在 SemEval-2025 任务 9 中通过不平衡处理和焦 LOSS 提升模型性能 

**Authors**: Zhuoang Cai, Zhenghao Li, Yang Liu, Liyuan Guo, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.00021)  

**Abstract**: Classification tasks often suffer from imbal- anced data distribution, which presents chal- lenges in food hazard detection due to severe class imbalances, short and unstructured text, and overlapping semantic categories. In this paper, we present our system for SemEval- 2025 Task 9: Food Hazard Detection, which ad- dresses these issues by applying data augmenta- tion techniques to improve classification perfor- mance. We utilize transformer-based models, BERT and RoBERTa, as backbone classifiers and explore various data balancing strategies, including random oversampling, Easy Data Augmentation (EDA), and focal loss. Our ex- periments show that EDA effectively mitigates class imbalance, leading to significant improve- ments in accuracy and F1 scores. Furthermore, combining focal loss with oversampling and EDA further enhances model robustness, par- ticularly for hard-to-classify examples. These findings contribute to the development of more effective NLP-based classification models for food hazard detection. 

**Abstract (ZH)**: 食品危害检测任务往往受到不平衡数据分布的影响，这在严重类别不平衡、短且无结构的文本以及重叠的语义类别下尤其具有挑战性。在本文中，我们提出了我们的系统以应对SemEval-2025 Task 9：食品危害检测任务，通过应用数据增强技术来提高分类性能。我们利用基于Transformer的模型BERT和RoBERTa作为基础分类器，并探索了随机过采样、Easy Data Augmentation (EDA)和焦点损失等多种数据平衡策略。实验结果显示，EDA有效缓解了类别不平衡问题，显著提高了准确率和F1分数。此外，结合焦点损失与过采样和EDA进一步增强了模型的稳健性，特别是在难以分类的例子上。这些发现为开发更有效的基于NLP的食品危害检测分类模型做出了贡献。 

---
# Beyond Public Access in LLM Pre-Training Data 

**Title (ZH)**: 超出公共访问范围的LLM预训练数据 

**Authors**: Sruly Rosenblat, Tim O'Reilly, Ilan Strauss  

**Link**: [PDF](https://arxiv.org/pdf/2505.00020)  

**Abstract**: Using a legally obtained dataset of 34 copyrighted O'Reilly Media books, we apply the DE-COP membership inference attack method to investigate whether OpenAI's large language models were trained on copyrighted content without consent. Our AUROC scores show that GPT-4o, OpenAI's more recent and capable model, demonstrates strong recognition of paywalled O'Reilly book content (AUROC = 82\%), compared to OpenAI's earlier model GPT-3.5 Turbo. In contrast, GPT-3.5 Turbo shows greater relative recognition of publicly accessible O'Reilly book samples. GPT-4o Mini, as a much smaller model, shows no knowledge of public or non-public O'Reilly Media content when tested (AUROC $\approx$ 50\%). Testing multiple models, with the same cutoff date, helps us account for potential language shifts over time that might bias our findings. These results highlight the urgent need for increased corporate transparency regarding pre-training data sources as a means to develop formal licensing frameworks for AI content training 

**Abstract (ZH)**: 使用合法获取的34本O'Reilly Media版权所有书籍数据集，我们应用DE-COP成员推理攻击方法，调查OpenAI的大语言模型是否在未经许可的情况下训练于受版权保护的内容。我们的AUROC评分表明，OpenAI较新的且更具能力的模型GPT-4o对受付费墙保护的O'Reilly书籍内容表现出强烈的识别能力（AUROC=82%），而OpenAI较早的模型GPT-3.5 Turbo则表现出对公开访问的O'Reilly书籍样本的相对更强的识别能力。相比之下，作为更小的模型，GPT-4o Mini在测试中对公开或非公开的O'Reilly Media内容毫无认识（AUROC≈50%）。在同一截止日期测试多个模型有助于我们考虑可能的时间语言变化偏差。这些结果突显了提高企业在预训练数据源方面透明度的迫切需要，作为为AI内容训练制定正式许可框架的手段。 

---
# An Empirical Study on Prompt Compression for Large Language Models 

**Title (ZH)**: 大型语言模型的提示压缩实证研究 

**Authors**: Zheng Zhang, Jinyi Li, Yihuai Lan, Xiang Wang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00019)  

**Abstract**: Prompt engineering enables Large Language Models (LLMs) to perform a variety of tasks. However, lengthy prompts significantly increase computational complexity and economic costs. To address this issue, we study six prompt compression methods for LLMs, aiming to reduce prompt length while maintaining LLM response quality. In this paper, we present a comprehensive analysis covering aspects such as generation performance, model hallucinations, efficacy in multimodal tasks, word omission analysis, and more. We evaluate these methods across 13 datasets, including news, scientific articles, commonsense QA, math QA, long-context QA, and VQA datasets. Our experiments reveal that prompt compression has a greater impact on LLM performance in long contexts compared to short ones. In the Longbench evaluation, moderate compression even enhances LLM performance. Our code and data is available at this https URL. 

**Abstract (ZH)**: 提示工程使大规模语言模型（LLMs）能够执行多种任务。然而，冗长的提示显著增加了计算复杂性和经济成本。为了解决这一问题，我们研究了六种提示压缩方法，旨在减少提示长度的同时保持LLM响应质量。在本文中，我们对生成性能、模型幻觉、多模态任务效果、单词省略分析等方面进行了全面分析。我们在包括新闻、科学文章、常识问答、数学问答、长上下文问答和VQA数据集在内的13个数据集上评估了这些方法。我们的实验表明，提示压缩对长上下文中的LLM性能影响更大，甚至适度压缩在Longbench评估中还提升了LLM性能。我们的代码和数据可在以下链接获取。 

---
# ReCellTy: Domain-specific knowledge graph retrieval-augmented LLMs workflow for single-cell annotation 

**Title (ZH)**: ReCellTy: 域特定知识图谱检索增强的LLM单细胞注释工作流 

**Authors**: Dezheng Han, Yibin Jia, Ruxiao Chen, Wenjie Han, Shuaishuai Guo, Jianbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00017)  

**Abstract**: To enable precise and fully automated cell type annotation with large language models (LLMs), we developed a graph structured feature marker database to retrieve entities linked to differential genes for cell reconstruction. We further designed a multi task workflow to optimize the annotation process. Compared to general purpose LLMs, our method improves human evaluation scores by up to 0.21 and semantic similarity by 6.1% across 11 tissue types, while more closely aligning with the cognitive logic of manual annotation. 

**Abstract (ZH)**: 利用大型语言模型实现精确且全自动化细胞类型注释的图结构特征标记数据库及多任务工作流方法 

---
# Sparks of Tabular Reasoning via Text2SQL Reinforcement Learning 

**Title (ZH)**: 通过Text2SQL强化学习激发的表格推理火花 

**Authors**: Josefa Lia Stoisser, Marc Boubnovski Martell, Julien Fauqueur  

**Link**: [PDF](https://arxiv.org/pdf/2505.00016)  

**Abstract**: This work reframes the Text-to-SQL task as a pathway for teaching large language models (LLMs) to reason over and manipulate tabular data--moving beyond the traditional focus on query generation. We propose a two-stage framework that leverages SQL supervision to develop transferable table reasoning capabilities. First, we synthesize detailed chain-of-thought (CoT) traces from real-world SQL queries, providing step-by-step, clause-level supervision that teaches the model how to traverse, filter, and aggregate table fields. Second, we introduce a Group Relative Policy Optimization (GRPO) reinforcement learning objective that connects SQL execution accuracy to generalizable reasoning by encouraging steps that extend beyond task-specific syntax and transfer across datasets. Empirically, our approach improves performance on standard Text-to-SQL benchmarks and achieves substantial gains on reasoning-intensive datasets such as BIRD and CRT-QA, demonstrating enhanced generalization and interpretability. Specifically, the distilled-quantized LLaMA model achieved a 20\% increase in accuracy when trained on Text-to-SQL tasks, while Qwen achieved a 5\% increase. These results suggest that SQL can serve not only as a target formalism but also as an effective scaffold for learning robust, transferable reasoning over structured data. 

**Abstract (ZH)**: 将文本转SQL任务重新定义为教学大型语言模型（LLMs）进行表数据推理和操作的路径——超越传统的查询生成关注。我们提出了一种两阶段框架，利用SQL监督来发展转移性表推理能力。首先，我们从实际的SQL查询中综合详细的推理链（CoT）跟踪，提供逐步骤、逐子句级别的监督，教导模型如何遍历、过滤和聚合表字段。其次，我们引入了一种基于组相对策略优化（GRPO）的强化学习目标，将SQL执行准确性与可泛化的推理连接起来，鼓励超出现有任务特定语法步骤并跨数据集进行转移。实验结果显示，我们的方法在标准的文本转SQL基准测试中提高了性能，在推理密集的数据集BIRD和CRT-QA中实现了显著的提升，展示了更强的泛化能力和可解释性。具体而言，蒸馏量化后的LLaMA模型在训练于文本转SQL任务时准确率提高了20%，而Qwen提高了5%。这些结果表明，SQL不仅可以作为目标形式语言，还可以作为学习在结构化数据上进行稳健的、可转移推理的有效支架。 

---
# Performance Evaluation of Emotion Classification in Japanese Using RoBERTa and DeBERTa 

**Title (ZH)**: 使用RoBERTa和DeBERTa对日语情感分类 performance 评价 

**Authors**: Yoichi Takenaka  

**Link**: [PDF](https://arxiv.org/pdf/2505.00013)  

**Abstract**: Background Practical applications such as social media monitoring and customer-feedback analysis require accurate emotion detection for Japanese text, yet resource scarcity and class imbalance hinder model performance.
Objective This study aims to build a high-accuracy model for predicting the presence or absence of eight Plutchik emotions in Japanese sentences.
Methods Using the WRIME corpus, we transform reader-averaged intensity scores into binary labels and fine-tune four pre-trained language models (BERT, RoBERTa, DeBERTa-v3-base, DeBERTa-v3-large). For context, we also assess two large language models (TinySwallow-1.5B-Instruct and ChatGPT-4o). Accuracy and F1-score serve as evaluation metrics.
Results DeBERTa-v3-large attains the best mean accuracy (0.860) and F1-score (0.662), outperforming all other models. It maintains robust F1 across both high-frequency emotions (e.g., Joy, Anticipation) and low-frequency emotions (e.g., Anger, Trust). The LLMs lag, with ChatGPT-4o and TinySwallow-1.5B-Instruct scoring 0.527 and 0.292 in mean F1, respectively.
Conclusion The fine-tuned DeBERTa-v3-large model currently offers the most reliable solution for binary emotion classification in Japanese. We release this model as a pip-installable package (pip install deberta-emotion-predictor). Future work should augment data for rare emotions, reduce model size, and explore prompt engineering to improve LLM performance.
This manuscript is under review for possible publication in New Generation Computing. 

**Abstract (ZH)**: 背景 实际应用如社交媒体监控和客户反馈分析需要准确的日本文字情感检测，但资源稀缺和类别不平衡阻碍了模型性能的提升。

目标 本研究旨在构建一个高精度模型，用于预测日本句子中是否存在八种普拉奇克情感。

方法 使用WRIME语料库，我们将读者平均强度评分转换为二元标签，并对四种预训练语言模型（BERT、RoBERTa、DeBERTa-v3-base、DeBERTa-v3-large）进行微调。此外，我们还评估了两个大型语言模型（TinySwallow-1.5B-Instruct和ChatGPT-4o）。准确率和F1分数用作评估指标。

结果 微调后的DeBERTa-v3-large模型在平均准确率（0.860）和F1分数（0.662）上表现最佳，优于所有其他模型。它在高频情感（如快乐、期待）和低频情感（如愤怒、信任）上都保持了稳健的F1分数。大型语言模型表现落后，ChatGPT-4o和TinySwallow-1.5B-Instruct的平均F1分数分别为0.527和0.292。

结论 当前，微调后的DeBERTa-v3-large模型提供了最可靠的二元情感分类解决方案。我们以pip可安装包的形式发布了该模型（pip install deberta-emotion-predictor）。未来工作应增加稀有情感的数据、减少模型大小，并探索提示工程以提高大型语言模型的性能。

本文正在《新一代 computing》期刊审核中，以期发表。 

---
# The AI Co-Ethnographer: How Far Can Automation Take Qualitative Research? 

**Title (ZH)**: AI 共同民族志研究者：自动化能将定性研究推进多远？ 

**Authors**: Fabian Retkowski, Andreas Sudmann, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2505.00012)  

**Abstract**: Qualitative research often involves labor-intensive processes that are difficult to scale while preserving analytical depth. This paper introduces The AI Co-Ethnographer (AICoE), a novel end-to-end pipeline developed for qualitative research and designed to move beyond the limitations of simply automating code assignments, offering a more integrated approach. AICoE organizes the entire process, encompassing open coding, code consolidation, code application, and even pattern discovery, leading to a comprehensive analysis of qualitative data. 

**Abstract (ZH)**: 定性研究往往涉及劳动密集型过程，难以在保持分析深度的同时 scalability。本文介绍了《AI 共同民族志学者（AICoE）》，这是一种针对定性研究开发的端到端管道，旨在超越简单自动化代码分配的局限性，提供一种更集成的方法。AICoE 整合了整个过程，包括开放编码、代码整合、代码应用，甚至模式发现，从而实现定性数据的全面分析。 

---
# Jailbreak Detection in Clinical Training LLMs Using Feature-Based Predictive Models 

**Title (ZH)**: 基于特征预测模型的临床训练大语言模型逃逸检测 

**Authors**: Tri Nguyen, Lohith Srikanth Pentapalli, Magnus Sieverding, Laurah Turner, Seth Overla, Weibing Zheng, Chris Zhou, David Furniss, Danielle Weber, Michael Gharib, Matt Kelleher, Michael Shukis, Cameron Pawlik, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00010)  

**Abstract**: Jailbreaking in Large Language Models (LLMs) threatens their safe use in sensitive domains like education by allowing users to bypass ethical safeguards. This study focuses on detecting jailbreaks in 2-Sigma, a clinical education platform that simulates patient interactions using LLMs. We annotated over 2,300 prompts across 158 conversations using four linguistic variables shown to correlate strongly with jailbreak behavior. The extracted features were used to train several predictive models, including Decision Trees, Fuzzy Logic-based classifiers, Boosting methods, and Logistic Regression. Results show that feature-based predictive models consistently outperformed Prompt Engineering, with the Fuzzy Decision Tree achieving the best overall performance. Our findings demonstrate that linguistic-feature-based models are effective and explainable alternatives for jailbreak detection. We suggest future work explore hybrid frameworks that integrate prompt-based flexibility with rule-based robustness for real-time, spectrum-based jailbreak monitoring in educational LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）中的越界攻击威胁其在敏感领域如教育中的安全使用。本研究专注于检测2-Sigma临床教育平台中由LLM模拟的患者互动中的越界行为。我们使用四种与越界行为高度相关的语言变量对158次对话中的2,300多个提示进行了标注。提取的特征被用于训练多种预测模型，包括决策树、基于模糊逻辑的分类器、提升方法和逻辑回归。结果表明，基于特征的预测模型整体上优于提示工程，模糊决策树表现出最佳性能。本研究发现表明，基于语言特征的模型是有效且可解释的越界检测替代方案。我们建议未来工作探索结合基于提示的灵活性与基于规则的稳健性的混合框架，以实现实时、频谱基于的教育LLM中越界监测。 

---
# A Scoping Review of Natural Language Processing in Addressing Medically Inaccurate Information: Errors, Misinformation, and Hallucination 

**Title (ZH)**: 自然语言处理在应对医学不准确信息中的综述：错误、 misinformation 和幻觉 

**Authors**: Zhaoyi Sun, Wen-Wai Yim, Ozlem Uzuner, Fei Xia, Meliha Yetisgen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00008)  

**Abstract**: Objective: This review aims to explore the potential and challenges of using Natural Language Processing (NLP) to detect, correct, and mitigate medically inaccurate information, including errors, misinformation, and hallucination. By unifying these concepts, the review emphasizes their shared methodological foundations and their distinct implications for healthcare. Our goal is to advance patient safety, improve public health communication, and support the development of more reliable and transparent NLP applications in healthcare.
Methods: A scoping review was conducted following PRISMA guidelines, analyzing studies from 2020 to 2024 across five databases. Studies were selected based on their use of NLP to address medically inaccurate information and were categorized by topic, tasks, document types, datasets, models, and evaluation metrics.
Results: NLP has shown potential in addressing medically inaccurate information on the following tasks: (1) error detection (2) error correction (3) misinformation detection (4) misinformation correction (5) hallucination detection (6) hallucination mitigation. However, challenges remain with data privacy, context dependency, and evaluation standards.
Conclusion: This review highlights the advancements in applying NLP to tackle medically inaccurate information while underscoring the need to address persistent challenges. Future efforts should focus on developing real-world datasets, refining contextual methods, and improving hallucination management to ensure reliable and transparent healthcare applications. 

**Abstract (ZH)**: 客观目标：本综述旨在探索使用自然语言处理（NLP）检测、纠正和缓解医学不准确信息（包括错误、 misinformation 和幻觉）的潜力和挑战。通过将这些概念统一，综述强调了它们共同的方法学基础及其对医疗保健的不同影响。我们的目标是提高患者安全、改善公共卫生沟通，并支持在医疗保健中开发更可靠和透明的NLP应用。

方法：根据PRISMA指南，本综述进行了范围回顾，分析了2020年至2024年间五个数据库中的研究。根据研究使用NLP解决医学不准确信息的情况，研究被按主题、任务、文档类型、数据集、模型和评估指标分类。

结果：NLP在以下任务中显示出解决医学不准确信息的潜力：（1）错误检测（2）错误纠正（3）错误信息检测（4）错误信息纠正（5）幻觉检测（6）幻觉缓解。然而，仍存在数据隐私、上下文依赖性和评估标准等方面的挑战。

结论：本综述强调了将NLP应用于解决医学不准确信息的进展，同时也指出了需要解决的持续挑战。未来的研究应关注开发现实世界的数据集、细化上下文方法和改善幻觉管理，以确保医疗保健应用的可靠性和透明度。 

---
# Toward a digital twin of U.S. Congress 

**Title (ZH)**: 向美国国会的数字孪生体迈进 

**Authors**: Hayden Helm, Tianyi Chen, Harvey McGuinness, Paige Lee, Brandon Duderstadt, Carey E. Priebe  

**Link**: [PDF](https://arxiv.org/pdf/2505.00006)  

**Abstract**: In this paper we provide evidence that a virtual model of U.S. congresspersons based on a collection of language models satisfies the definition of a digital twin. In particular, we introduce and provide high-level descriptions of a daily-updated dataset that contains every Tweet from every U.S. congressperson during their respective terms. We demonstrate that a modern language model equipped with congressperson-specific subsets of this data are capable of producing Tweets that are largely indistinguishable from actual Tweets posted by their physical counterparts. We illustrate how generated Tweets can be used to predict roll-call vote behaviors and to quantify the likelihood of congresspersons crossing party lines, thereby assisting stakeholders in allocating resources and potentially impacting real-world legislative dynamics. We conclude with a discussion of the limitations and important extensions of our analysis. 

**Abstract (ZH)**: 本文提供了证据，证明基于语言模型集合构建的美国国会成员虚拟模型符合数字孪生的定义。特别是，我们引入并提供了包含每位美国国会成员在其任期内发布的每条推特的每日更新数据集。我们演示了装备有特定于国会成员数据子集的现代语言模型能够生成与物理国会成员实际发布的推特几乎无法区别的推特。我们展示了生成的推特如何用于预测投票行为，并量化国会成员跨越党派线的可能性，从而帮助利益相关者分配资源，并可能影响实际立法动态。最后，我们讨论了分析的局限性和重要扩展。 

---
# LangVAE and LangSpace: Building and Probing for Language Model VAEs 

**Title (ZH)**: LangVAE和LangSpace：构建与探测语言模型VAE 

**Authors**: Danilo S. Carvalho, Yingji Zhang, Harriet Unsworth, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2505.00004)  

**Abstract**: We present LangVAE, a novel framework for modular construction of variational autoencoders (VAEs) on top of pre-trained large language models (LLMs). Such language model VAEs can encode the knowledge of their pre-trained components into more compact and semantically disentangled representations. The representations obtained in this way can be analysed with the LangVAE companion framework: LangSpace, which implements a collection of probing methods, such as vector traversal and interpolation, disentanglement measures, and cluster visualisations. LangVAE and LangSpace offer a flexible, efficient and scalable way of building and analysing textual representations, with simple integration for models available on the HuggingFace Hub. Additionally, we conducted a set of experiments with different encoder and decoder combinations, as well as annotated inputs, revealing a wide range of interactions across architectural families and sizes w.r.t. generalisation and disentanglement. Our findings demonstrate a promising framework for systematising the experimentation and understanding of textual representations. 

**Abstract (ZH)**: 我们提出LangVAE，一种基于预训练大型语言模型构建模块化变分自编码器（VAEs）的新框架。此类语言模型VAE能够将其预训练组件的知识编码为更为紧凑且语义上分离的表示。通过LangVAE配套框架LangSpace可以获得的表示，可以使用该框架实现的多种探查方法进行分析，如向量遍历和插值、分离度量以及聚类可视化。LangVAE和LangSpace提供了一种灵活、高效且可扩展的文本表示构建和分析方法，支持从HuggingFace Hub集成不同模型。此外，我们还通过不同的编码器和解码器组合以及标注输入进行了实验，揭示了不同架构家族和规模在泛化和分离性方面的广泛交互。我们的研究结果表明，LangVAE为系统化文本表示的实验和理解提供了一个有前景的框架。 

---
