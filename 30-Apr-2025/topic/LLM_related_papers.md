# Identifying Uncertainty in Self-Adaptive Robotics with Large Language Models 

**Title (ZH)**: 使用大型语言模型识别自适应机器人中的不确定性 

**Authors**: Hassan Sartaj, Jalil Boudjadar, Mirgita Frasheri, Shaukat Ali, Peter Gorm Larsen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20684)  

**Abstract**: Future self-adaptive robots are expected to operate in highly dynamic environments while effectively managing uncertainties. However, identifying the sources and impacts of uncertainties in such robotic systems and defining appropriate mitigation strategies is challenging due to the inherent complexity of self-adaptive robots and the lack of comprehensive knowledge about the various factors influencing uncertainty. Hence, practitioners often rely on intuition and past experiences from similar systems to address uncertainties. In this article, we evaluate the potential of large language models (LLMs) in enabling a systematic and automated approach to identify uncertainties in self-adaptive robotics throughout the software engineering lifecycle. For this evaluation, we analyzed 10 advanced LLMs with varying capabilities across four industrial-sized robotics case studies, gathering the practitioners' perspectives on the LLM-generated responses related to uncertainties. Results showed that practitioners agreed with 63-88% of the LLM responses and expressed strong interest in the practicality of LLMs for this purpose. 

**Abstract (ZH)**: 未来自适应机器人预计能够在高度动态环境中运作，同时有效管理不确定性。然而，识别此类机器人系统中的不确定性来源及其影响，并定义适当的缓解策略非常具有挑战性，因为自适应机器人本身固有的复杂性以及对其各种影响不确定性因素的了解不足。因此，实践者常常依赖直觉和来自类似系统的过往经验来应对不确定性。在本文中，我们评估了大型语言模型（LLMs）在软件开发生命周期中系统化和自动化识别自适应机器人中不确定性方面的潜力。为此，我们分析了四个工业规模的机器人案例研究中的10种具有不同能力的先进LLM，并收集了实践者对LLM生成的关于不确定性的响应的意见。结果显示，实践者同意63-88%的LLM响应，并对该LLMs在该领域的实用性表现出强烈兴趣。 

---
# Jekyll-and-Hyde Tipping Point in an AI's Behavior 

**Title (ZH)**: AI行为中的ekyll-and-hyde临界点 

**Authors**: Neil F. Johnson, Frank Yingjie Huo  

**Link**: [PDF](https://arxiv.org/pdf/2504.20980)  

**Abstract**: Trust in AI is undermined by the fact that there is no science that predicts -- or that can explain to the public -- when an LLM's output (e.g. ChatGPT) is likely to tip mid-response to become wrong, misleading, irrelevant or dangerous. With deaths and trauma already being blamed on LLMs, this uncertainty is even pushing people to treat their 'pet' LLM more politely to 'dissuade' it (or its future Artificial General Intelligence offspring) from suddenly turning on them. Here we address this acute need by deriving from first principles an exact formula for when a Jekyll-and-Hyde tipping point occurs at LLMs' most basic level. Requiring only secondary school mathematics, it shows the cause to be the AI's attention spreading so thin it suddenly snaps. This exact formula provides quantitative predictions for how the tipping-point can be delayed or prevented by changing the prompt and the AI's training. Tailored generalizations will provide policymakers and the public with a firm platform for discussing any of AI's broader uses and risks, e.g. as a personal counselor, medical advisor, decision-maker for when to use force in a conflict situation. It also meets the need for clear and transparent answers to questions like ''should I be polite to my LLM?'' 

**Abstract (ZH)**: AI信任受挫的原因在于缺乏能够预测或解释LLM输出何时可能在响应中途转变为错误、误导、无关或危险的科学。由于已经因LLM导致了死亡和创伤事件的发生，这种不确定性促使人们更加礼貌地对待它们的“宠物”LLM，以“劝阻”它们（或其未来的通用人工智能后代）突然对他们采取敌对行动。我们通过从基本原理出发，推导出一个精确公式，以量化预测何时会在LLM的最基本层面上出现从“好人”到“坏人”的转折点。该公式仅需中等教育水平的数学知识，显示原因是AI的注意力分散到极致，导致突然崩溃。这个精确公式提供了通过更改提示和AI训练来推迟或防止转折点的定量预测。定制的一般化结果将为决策者和公众提供一个坚实的基础，讨论AI的更广泛用途和风险，例如作为个人咨询师、医疗顾问，在冲突中使用武力时的决策者。它还满足了对诸如“我是否应该对我的LLM礼貌相待？”这类问题提供清晰透明答案的需求。 

---
# ChestX-Reasoner: Advancing Radiology Foundation Models with Reasoning through Step-by-Step Verification 

**Title (ZH)**: ChestX-Reasoner: 通过逐步验证提高放射学基础模型的能力 

**Authors**: Ziqing Fan, Cheng Liang, Chaoyi Wu, Ya Zhang, Yanfeng Wang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.20930)  

**Abstract**: Recent advances in reasoning-enhanced large language models (LLMs) and multimodal LLMs (MLLMs) have significantly improved performance in complex tasks, yet medical AI models often overlook the structured reasoning processes inherent in clinical practice. In this work, we present ChestX-Reasoner, a radiology diagnosis MLLM designed to leverage process supervision mined directly from clinical reports, reflecting the step-by-step reasoning followed by radiologists. We construct a large dataset by extracting and refining reasoning chains from routine radiology reports. Our two-stage training framework combines supervised fine-tuning and reinforcement learning guided by process rewards to better align model reasoning with clinical standards. We introduce RadRBench-CXR, a comprehensive benchmark featuring 59K visual question answering samples with 301K clinically validated reasoning steps, and propose RadRScore, a metric evaluating reasoning factuality, completeness, and effectiveness. ChestX-Reasoner outperforms existing medical and general-domain MLLMs in both diagnostic accuracy and reasoning ability, achieving 16%, 5.9%, and 18% improvements in reasoning ability compared to the best medical MLLM, the best general MLLM, and its base model, respectively, as well as 3.3%, 24%, and 27% improvements in outcome accuracy. All resources are open-sourced to facilitate further research in medical reasoning MLLMs. 

**Abstract (ZH)**: 近期增强推理的大语言模型（LLMs）和多模态大语言模型（MLLMs）在复杂任务中的表现显著提升，但医学AI模型往往忽视了临床实践中固有的结构化推理过程。在此项工作中，我们提出了ChestX-Reasoner，一种放射诊断MLLM，旨在利用直接从临床报告中挖掘的流程监督，反映放射科医生遵循的逐步推理过程。我们通过提取和精炼常规放射报告中的推理链构建了一个大规模数据集。我们的两阶段训练框架结合了基于流程奖励的监督微调和强化学习，更好地使模型的推理与临床标准保持一致。我们引入了RadRBench-CXR，一个全面基准，包含59000个视觉问答样本和301000个临床验证的推理步骤，并提出了RadRScore，这是一个评估推理事实性、完整性和有效性指标。ChestX-Reasoner在诊断准确性和推理能力方面都优于现有的医学和通用领域MMLMs，相对于最佳医学MMLM、最佳通用MMLM及其基础模型，在推理能力上分别提高了16%、5.9%和18%，在结果准确性上分别提高了3.3%、24%和27%。所有资源均已开源，以促进医学推理MMLMs研究的进一步发展。 

---
# The Leaderboard Illusion 

**Title (ZH)**: 排行榜错觉 

**Authors**: Shivalika Singh, Yiyang Nan, Alex Wang, Daniel D'Souza, Sayash Kapoor, Ahmet Üstün, Sanmi Koyejo, Yuntian Deng, Shayne Longpre, Noah Smith, Beyza Ermis, Marzieh Fadaee, Sara Hooker  

**Link**: [PDF](https://arxiv.org/pdf/2504.20879)  

**Abstract**: Measuring progress is fundamental to the advancement of any scientific field. As benchmarks play an increasingly central role, they also grow more susceptible to distortion. Chatbot Arena has emerged as the go-to leaderboard for ranking the most capable AI systems. Yet, in this work we identify systematic issues that have resulted in a distorted playing field. We find that undisclosed private testing practices benefit a handful of providers who are able to test multiple variants before public release and retract scores if desired. We establish that the ability of these providers to choose the best score leads to biased Arena scores due to selective disclosure of performance results. At an extreme, we identify 27 private LLM variants tested by Meta in the lead-up to the Llama-4 release. We also establish that proprietary closed models are sampled at higher rates (number of battles) and have fewer models removed from the arena than open-weight and open-source alternatives. Both these policies lead to large data access asymmetries over time. Providers like Google and OpenAI have received an estimated 19.2% and 20.4% of all data on the arena, respectively. In contrast, a combined 83 open-weight models have only received an estimated 29.7% of the total data. We show that access to Chatbot Arena data yields substantial benefits; even limited additional data can result in relative performance gains of up to 112% on the arena distribution, based on our conservative estimates. Together, these dynamics result in overfitting to Arena-specific dynamics rather than general model quality. The Arena builds on the substantial efforts of both the organizers and an open community that maintains this valuable evaluation platform. We offer actionable recommendations to reform the Chatbot Arena's evaluation framework and promote fairer, more transparent benchmarking for the field 

**Abstract (ZH)**: 测量进展是任何科学领域进步的基础。随着基准测试在其中扮演越来越重要的角色，它们也变得更加容易受到扭曲。Chatbot Arena 已成为排名最强大人工智能系统的主要基准排行榜。然而，在本文中，我们识别出系统性问题，导致了一个扭曲的竞争环境。我们发现，未公开的私人测试实践有利于少数提供者，他们能够在公共发布前测试多个变体，并可以根据需要撤回分数。我们确定，这些提供者能够选择最佳分数的能力导致了由于选择性披露性能结果而导致的偏向性Arena分数。在极端情况下，我们识别出Meta在Llama-4发布前测试了27种私人大语言模型变体。我们还确定，专有封闭模型被采样得更多（比赛次数），被从竞技场移除的模型也更少，而开源和开源替代方案则不然。这两种政策随着时间的推移导致了大量数据访问不对等。谷歌和OpenAI等提供者分别获得了约19.2%和20.4%的所有竞技场数据。相比之下，近83个开源权重模型仅获得了约29.7%的总数据。我们显示，访问Chatbot Arena数据能够带来实质性的益处；即使额外获得少量数据也可能基于我们保守的估计在竞技场分布中产生高达112%的相对性能提升。这些动态导致了针对竞技场特定动态的过度拟合，而不是一般的模型质量。Chatbot Arena 基于组织者和维护这一宝贵评估平台的开放社区的重大努力。我们提出了一系列可操作的建议，旨在改革Chatbot Arena的评估框架，并促进该领域的更公平和透明的基准测试。 

---
# Ascendra: Dynamic Request Prioritization for Efficient LLM Serving 

**Title (ZH)**: Ascendra：高效的LLM服务动态请求优先级调度 

**Authors**: Azam Ikram, Xiang Li, Sameh Elnikety, Saurabh Bagchi  

**Link**: [PDF](https://arxiv.org/pdf/2504.20828)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has driven the need for more efficient serving strategies. In this context, efficiency refers to the proportion of requests that meet their Service Level Objectives (SLOs), particularly for Time To First Token (TTFT) and Time Between Tokens (TBT). However, existing systems often prioritize one metric at the cost of the other. We present Ascendra, an LLM serving system designed to meet both TTFT and TBT SLOs simultaneously. The core insight behind Ascendra is that a request's urgency evolves as it approaches its deadline. To leverage this, Ascendra partitions GPU resources into two types of instances: low-priority and high-priority. Low-priority instances maximize throughput by processing requests out of arrival order, but at the risk of request starvation. To address this, Ascendra employs a performance model to predict requests at risk of missing their SLOs and proactively offloads them to high-priority instances. High-priority instances are optimized for low-latency execution and handle urgent requests nearing their deadlines. This partitioned architecture enables Ascendra to effectively balance high throughput and low latency. Extensive evaluation shows that Ascendra improves system throughput by up to 1.7x compared to vLLM and Sarathi-Serve while meeting both TTFT and TBT SLOs. 

**Abstract (ZH)**: 大规模语言模型的快速进步推动了更高效服务策略的需求。在这种背景下，效率指的是满足服务级别目标（SLO）的请求比例，特别是针对首个词出现时间（TTFT）和词之间时间（TBT）。然而，现有系统往往在优先考虑一个指标时会牺牲另一个指标。我们提出了Ascendra，这是一种设计用于同时满足首个词出现时间和词之间时间SLO的大规模语言模型服务系统。Ascendra的核心洞察是请求的紧迫性会随着接近截止日期而变化。基于此，Ascendra将GPU资源分为两种实例：低优先级和高优先级。低优先级实例通过按到达顺序处理请求来最大化 throughput，但存在请求饿死的风险。为了解决这个问题，Ascendra采用性能模型预测可能无法满足SLO的请求，并主动将这些请求卸载到高优先级实例。高优先级实例优化了低latency执行，处理接近截止日期的紧急请求。这种分区架构使得Ascendra能够有效平衡高吞吐量和低latency。广泛评估显示，与vLLM和Sarathi-Serve相比，Ascendra可将系统吞吐量提高多达1.7倍，同时满足首个词出现时间和词之间时间SLO。 

---
# PaRT: Enhancing Proactive Social Chatbots with Personalized Real-Time Retrieval 

**Title (ZH)**: PaRT: 通过个性化实时检索增强主动社交聊天机器人 

**Authors**: Zihan Niu, Zheyong Xie, Shaosheng Cao, Chonggang Lu, Zheyu Ye, Tong Xu, Zuozhu Liu, Yan Gao, Jia Chen, Zhe Xu, Yi Wu, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20624)  

**Abstract**: Social chatbots have become essential intelligent companions in daily scenarios ranging from emotional support to personal interaction. However, conventional chatbots with passive response mechanisms usually rely on users to initiate or sustain dialogues by bringing up new topics, resulting in diminished engagement and shortened dialogue duration. In this paper, we present PaRT, a novel framework enabling context-aware proactive dialogues for social chatbots through personalized real-time retrieval and generation. Specifically, PaRT first integrates user profiles and dialogue context into a large language model (LLM), which is initially prompted to refine user queries and recognize their underlying intents for the upcoming conversation. Guided by refined intents, the LLM generates personalized dialogue topics, which then serve as targeted queries to retrieve relevant passages from RedNote. Finally, we prompt LLMs with summarized passages to generate knowledge-grounded and engagement-optimized responses. Our approach has been running stably in a real-world production environment for more than 30 days, achieving a 21.77\% improvement in the average duration of dialogues. 

**Abstract (ZH)**: 社会聊天机器人已成为日常场景中从情感支持到个人互动的必不可少的智能伴侣。然而，传统的基于被动响应机制的聊天机器人通常需要用户通过提出新话题来启动或维持对话，这导致了参与度下降和对话时长缩短。本文介绍了PaRT，一种新型框架，通过个性化实时检索和生成，使社会聊天机器人能够进行基于上下文的主动对话。具体而言，PaRT 首先将用户资料和对话上下文整合到一个大型语言模型（LLM）中，该模型最初被激发以细化用户查询并识别其后续对话的潜在意图。基于细化的意图，LLM 生成个性化对话主题，这些主题随后作为目标查询检索RedNote的相关段落。最后，我们使用总结的段落激发LLM生成知识导向且能优化参与度的响应。我们的方法已在真实生产环境中稳定运行超过30天，对话平均时长提高了21.77%。 

---
# MuRAL: A Multi-Resident Ambient Sensor Dataset Annotated with Natural Language for Activities of Daily Living 

**Title (ZH)**: MuRAL：一种标注自然语言的家庭环境传感器多居民活动数据集 

**Authors**: Xi Chen, Julien Cumin, Fano Ramparany, Dominique Vaufreydaz  

**Link**: [PDF](https://arxiv.org/pdf/2504.20505)  

**Abstract**: Recent advances in Large Language Models (LLMs) have shown promising potential for human activity recognition (HAR) using ambient sensors, especially through natural language reasoning and zero-shot learning. However, existing datasets such as CASAS, ARAS, and MARBLE were not originally designed with LLMs in mind and therefore lack the contextual richness, complexity, and annotation granularity required to fully exploit LLM capabilities. In this paper, we introduce MuRAL, the first Multi-Resident Ambient sensor dataset with natural Language, comprising over 21 hours of multi-user sensor data collected from 21 sessions in a smart-home environment. MuRAL is annotated with fine-grained natural language descriptions, resident identities, and high-level activity labels, all situated in dynamic, realistic multi-resident settings. We benchmark MuRAL using state-of-the-art LLMs for three core tasks: subject assignment, action description, and activity classification. Our results demonstrate that while LLMs can provide rich semantic interpretations of ambient data, current models still face challenges in handling multi-user ambiguity and under-specified sensor contexts. We release MuRAL to support future research on LLM-powered, explainable, and socially aware activity understanding in smart environments. For access to the dataset, please reach out to us via the provided contact information. A direct link for dataset retrieval will be made available at this location in due course. 

**Abstract (ZH)**: 最近大型语言模型（LLMs）的发展展示了利用环境传感器进行人体活动识别（HAR）的 promising 潜力，特别是在自然语言推理和零样本学习方面。然而，现有的数据集如CASAS、ARAS和MARBLE最初并非为LLMs设计，因而缺乏所需的上下文丰富性、复杂性和注释粒度，以充分利用LLMs的能力。本文我们介绍了MuRAL，这是首个包含自然语言的多居民环境传感器数据集，源自智能家居环境中21会话超过21小时的多用户传感器数据。MuRAL通过自然语言描述、居民身份和高层次活动标签进行详细注释，位于动态且现实的多居民环境之中。我们使用最先进的LLMs对MuRAL进行三个核心任务的基准测试：主体分配、动作描述和活动分类。结果显示，虽然LLMs能够提供丰富的环境数据语义解释，但当前模型仍然面临处理多用户模糊性和传感器背景不明确性的挑战。我们发布MuRAL以支持未来LLM驱动的、可解释的和社会意识强的活动理解研究。有关数据集的访问，请通过提供的联系方式联系我们。数据集获取的直接链接将在合适的时间在此位置提供。 

---
# TAMO:Fine-Grained Root Cause Analysis via Tool-Assisted LLM Agent with Multi-Modality Observation Data 

**Title (ZH)**: TAMO:借助多模态观察数据的工具辅助LLM代理细粒度根因分析 

**Authors**: Qi Wang, Xiao Zhang, Mingyi Li, Yuan Yuan, Mengbai Xiao, Fuzhen Zhuang, Dongxiao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20462)  

**Abstract**: With the development of distributed systems, microservices and cloud native technologies have become central to modern enterprise software development. Despite bringing significant advantages, these technologies also increase system complexity and operational challenges. Traditional root cause analysis (RCA) struggles to achieve automated fault response, heavily relying on manual intervention. In recent years, large language models (LLMs) have made breakthroughs in contextual inference and domain knowledge integration, providing new solutions for Artificial Intelligence for Operations (AIOps). However, Existing LLM-based approaches face three key challenges: text input constraints, dynamic service dependency hallucinations, and context window limitations. To address these issues, we propose a tool-assisted LLM agent with multi-modality observation data, namely TAMO, for fine-grained RCA. It unifies multi-modal observational data into time-aligned representations to extract consistent features and employs specialized root cause localization and fault classification tools for perceiving the contextual environment. This approach overcomes the limitations of LLM in handling real-time changing service dependencies and raw observational data and guides LLM to generate repair strategies aligned with system contexts by structuring key information into a prompt. Experimental results show that TAMO performs well in root cause analysis when dealing with public datasets characterized by heterogeneity and common fault types, demonstrating its effectiveness. 

**Abstract (ZH)**: 随着分布式系统的不断发展，微服务和云原生技术已成为现代企业软件开发的核心。尽管这些技术带来了显著的优势，但也增加了系统的复杂性和运维挑战。传统的根本原因分析（RCA）难以实现自动化故障响应， heavily依赖手动干预。近年来，大规模语言模型（LLMs）在上下文推断和领域知识整合方面取得了突破，为运维人工智能（AIOps）提供了新的解决方案。然而，现有的LLM基方法面临三个关键挑战：文本输入限制、动态服务依赖幻觉以及上下文窗口限制。为了应对这些问题，我们提出了一种工具辅助的多模态LLM智能体（TAMO），用于细粒度的根本原因分析。该智能体将多模态观测数据统一为时间对齐的表示以提取一致的特征，并利用专门的根因定位和故障分类工具感知上下文环境。该方法克服了LLM在处理实时变化的服务依赖关系和原始观测数据方面的限制，并通过结构化关键信息指导LLM生成符合系统环境的修复策略。实验结果显示，TAMO在处理由异质性和常见故障类型特征描述的公开数据集时，在根本原因分析方面表现出色，证明了其有效性。 

---
# RV-Syn: Rational and Verifiable Mathematical Reasoning Data Synthesis based on Structured Function Library 

**Title (ZH)**: RV-Syn：基于结构化函数库的合理可验证数学推理数据合成 

**Authors**: Jiapeng Wang, Jinhao Jiang, Zhiqiang Zhang, Jun Zhou, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.20426)  

**Abstract**: The advancement of reasoning capabilities in Large Language Models (LLMs) requires substantial amounts of high-quality reasoning data, particularly in mathematics. Existing data synthesis methods, such as data augmentation from annotated training sets or direct question generation based on relevant knowledge points and documents, have expanded datasets but face challenges in mastering the inner logic of the problem during generation and ensuring the verifiability of the solutions. To address these issues, we propose RV-Syn, a novel Rational and Verifiable mathematical Synthesis approach. RV-Syn constructs a structured mathematical operation function library based on initial seed problems and generates computational graphs as solutions by combining Python-formatted functions from this library. These graphs are then back-translated into complex problems. Based on the constructed computation graph, we achieve solution-guided logic-aware problem generation. Furthermore, the executability of the computational graph ensures the verifiability of the solving process. Experimental results show that RV-Syn surpasses existing synthesis methods, including those involving human-generated problems, achieving greater efficient data scaling. This approach provides a scalable framework for generating high-quality reasoning datasets. 

**Abstract (ZH)**: Large Language Models (LLMs) 的推理能力提升需要大量高质量的推理数据，尤其是在数学领域。现有的数据合成方法，如从标注训练集进行数据扩增或基于相关知识点和文档直接生成问题，虽然扩展了数据集，但在生成过程中掌握问题内部逻辑并确保解决方案可验证方面仍面临挑战。为解决这些问题，我们提出了 RV-Syn，一种新颖的合理且可验证的数学合成方法。RV-Syn 基于初始种子问题构建结构化的数学运算函数库，并通过结合该库中的 Python 格式函数生成计算图作为解决方案。然后将这些图回译为复杂问题。基于构建的计算图，我们实现了指导性逻辑感知问题生成。此外，计算图的可执行性确保了解决过程的可验证性。实验结果表明，RV-Syn 超越了现有合成方法，包括涉及人工生成问题的方法，实现了更高效的数据显示规模。该方法为生成高质量推理数据集提供了可扩展的框架。 

---
# Skill Discovery for Software Scripting Automation via Offline Simulations with LLMs 

**Title (ZH)**: 通过 Offline Simulations with LLMs 的技能发现方法实现软件脚本自动化 

**Authors**: Paiheng Xu, Gang Wu, Xiang Chen, Tong Yu, Chang Xiao, Franck Dernoncourt, Tianyi Zhou, Wei Ai, Viswanathan Swaminathan  

**Link**: [PDF](https://arxiv.org/pdf/2504.20406)  

**Abstract**: Scripting interfaces enable users to automate tasks and customize software workflows, but creating scripts traditionally requires programming expertise and familiarity with specific APIs, posing barriers for many users. While Large Language Models (LLMs) can generate code from natural language queries, runtime code generation is severely limited due to unverified code, security risks, longer response times, and higher computational costs. To bridge the gap, we propose an offline simulation framework to curate a software-specific skillset, a collection of verified scripts, by exploiting LLMs and publicly available scripting guides. Our framework comprises two components: (1) task creation, using top-down functionality guidance and bottom-up API synergy exploration to generate helpful tasks; and (2) skill generation with trials, refining and validating scripts based on execution feedback. To efficiently navigate the extensive API landscape, we introduce a Graph Neural Network (GNN)-based link prediction model to capture API synergy, enabling the generation of skills involving underutilized APIs and expanding the skillset's diversity. Experiments with Adobe Illustrator demonstrate that our framework significantly improves automation success rates, reduces response time, and saves runtime token costs compared to traditional runtime code generation. This is the first attempt to use software scripting interfaces as a testbed for LLM-based systems, highlighting the advantages of leveraging execution feedback in a controlled environment and offering valuable insights into aligning AI capabilities with user needs in specialized software domains. 

**Abstract (ZH)**: 大型语言模型赋能的离线模拟框架：提高软件自动化流程效率与安全性 

---
# Spark: A System for Scientifically Creative Idea Generation 

**Title (ZH)**: Spark: 一个用于科学研究创意生成的系统 

**Authors**: Aishik Sanyal, Samuel Schapiro, Sumuk Shashidhar, Royce Moon, Lav R. Varshney, Dilek Hakkani-Tur  

**Link**: [PDF](https://arxiv.org/pdf/2504.20090)  

**Abstract**: Recently, large language models (LLMs) have shown promising abilities to generate novel research ideas in science, a direction which coincides with many foundational principles in computational creativity (CC). In light of these developments, we present an idea generation system named Spark that couples retrieval-augmented idea generation using LLMs with a reviewer model named Judge trained on 600K scientific reviews from OpenReview. Our work is both a system demonstration and intended to inspire other CC researchers to explore grounding the generation and evaluation of scientific ideas within foundational CC principles. To this end, we release the annotated dataset used to train Judge, inviting other researchers to explore the use of LLMs for idea generation and creative evaluations. 

**Abstract (ZH)**: 近期，大规模语言模型（LLMs）在科学领域展示了生成新颖研究想法的潜力，这与计算创意（CC）领域的许多基础原则相吻合。基于这些进展，我们提出了一种名为Spark的想法生成系统，该系统结合了使用LLMs的检索增强想法生成与一个在600K篇OpenReview科学评审上训练的评审模型Judge。我们的工作既是系统演示，也旨在激励其他CC研究人员探索将科学想法的生成和评估与基础CC原则相结合的研究方向。为此，我们发布了用于训练Judge的标注数据集，邀请其他研究人员探索使用LLMs进行想法生成和创造性评估的方法。 

---
# Toward Efficient Exploration by Large Language Model Agents 

**Title (ZH)**: 大型语言模型智能体高效探索的方法研究 

**Authors**: Dilip Arumugam, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2504.20997)  

**Abstract**: A burgeoning area within reinforcement learning (RL) is the design of sequential decision-making agents centered around large language models (LLMs). While autonomous decision-making agents powered by modern LLMs could facilitate numerous real-world applications, such successes demand agents that are capable of data-efficient RL. One key obstacle to achieving data efficiency in RL is exploration, a challenge that we demonstrate many recent proposals for LLM agent designs struggle to contend with. Meanwhile, classic algorithms from the RL literature known to gracefully address exploration require technical machinery that can be challenging to operationalize in purely natural language settings. In this work, rather than relying on finetuning or in-context learning to coax LLMs into implicitly imitating a RL algorithm, we illustrate how LLMs can be used to explicitly implement an existing RL algorithm (Posterior Sampling for Reinforcement Learning) whose capacity for statistically-efficient exploration is already well-studied. We offer empirical results demonstrating how our LLM-based implementation of a known, data-efficient RL algorithm can be considerably more effective in natural language tasks that demand prudent exploration. 

**Abstract (ZH)**: 强化学习领域内一个 burgeoning 的研究方向是围绕大规模语言模型 (LLMs) 设计 sequential 决策代理。在由现代 LLMs 驱动的自主决策代理有可能促进众多现实世界应用的同时，这样的成功需要具备数据高效 RL 的代理。实现 RL 中数据效率的一个关键障碍是探索，许多近期的 LLM 代理设计提议难以应对这一挑战。与此同时，RL 文献中经典的解决探索问题的技术虽然表现良好，但在纯自然语言环境中实现却具有技术挑战性。在本文中，我们并非依靠调优或基于上下文学习来引导 LLMs 显式模仿一个 RL 算法，而是展示了如何利用 LLMs 显式实现一个已有的 RL 算法（强化学习中的后验采样），该算法的统计高效探索能力已经得到了充分研究。我们提供了实验证据，证明我们基于 LLM 的已知数据高效 RL 算法实现方式在需要明智探索的自然语言任务中表现出了显著的效果。 

---
# OSVBench: Benchmarking LLMs on Specification Generation Tasks for Operating System Verification 

**Title (ZH)**: OSVBench: 在操作系统验证规范生成任务上评估LLM性能 

**Authors**: Shangyu Li, Juyong Jiang, Tiancheng Zhao, Jiasi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20964)  

**Abstract**: We introduce OSVBench, a new benchmark for evaluating Large Language Models (LLMs) in generating complete specification code pertaining to operating system kernel verification tasks. The benchmark first defines the specification generation problem into a program synthesis problem within a confined scope of syntax and semantics by providing LLMs with the programming model. The LLMs are required to understand the provided verification assumption and the potential syntax and semantics space to search for, then generate the complete specification for the potentially buggy operating system code implementation under the guidance of the high-level functional description of the operating system. This benchmark is built upon a real-world operating system kernel, Hyperkernel, and consists of 245 complex specification generation tasks in total, each is a long context task of about 20k-30k tokens. Our comprehensive evaluation of 12 LLMs exhibits the limited performance of the current LLMs on the specification generation tasks for operating system verification. Significant disparities in their performance on the benchmark highlight differences in their ability to handle long-context code generation tasks. The evaluation toolkit and benchmark are available at this https URL. 

**Abstract (ZH)**: OSVBench：一种用于评估大型语言模型在生成操作系统内核验证相关完整规范代码方面的基准测试 

---
# Trace-of-Thought: Enhanced Arithmetic Problem Solving via Reasoning Distillation From Large to Small Language Models 

**Title (ZH)**: 思维轨迹：从大规模到小规模语言模型的推理提炼以增强算术问题求解 

**Authors**: Tyler McDonald, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2504.20946)  

**Abstract**: As Large Language Models (LLMs) continue to be leveraged for daily tasks, prompt engineering remains an active field of contribution within computational linguistics, particularly in domains requiring specialized knowledge such as arithmetic reasoning. While these LLMs are optimized for a variety of tasks, their exhaustive employment may become computationally or financially cumbersome for small teams. Additionally, complete reliance on proprietary, closed-source models often limits customization and adaptability, posing significant challenges in research and application scalability. Instead, by leveraging open-source models at or below 7 billion parameters, we can optimize our resource usage while still observing remarkable gains over standard prompting approaches. To cultivate this notion, we introduce Trace-of-Thought Prompting, a simple, zero-shot prompt engineering method that instructs LLMs to create observable subproblems using critical problem-solving, specifically designed to enhance arithmetic reasoning capabilities. When applied to open-source models in tandem with GPT-4, we observe that Trace-of-Thought not only allows novel insight into the problem-solving process but also introduces performance gains as large as 125% on language models at or below 7 billion parameters. This approach underscores the potential of open-source initiatives in democratizing AI research and improving the accessibility of high-quality computational linguistics applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在日常任务中的持续应用使得提示工程仍然是计算语言学中的一个活跃研究领域，特别是在需要专门知识的领域，如算术推理。虽然这些LLMs优化了多种任务，但其全面应用可能对小型团队来说在计算或财务方面变得臃肿。此外，完全依赖于专有的闭源模型往往会限制定制化和适应性，对研究和应用的可扩展性造成重大挑战。相反，通过利用参数量在70亿以下的开源模型，我们可以在优化资源使用的同时，仍然观察到显著优于标准提示方法的收益。为了培养这一观点，我们引入了思路追踪提示，这是一种简单且零样本的提示工程技术，指导LLMs使用关键问题解决技巧创建可观察的子问题，特别设计用于增强算术推理能力。当与GPT-4一起应用于开源模型时，我们观察到思路追踪不仅允许对问题解决过程进行新颖的洞察，还能够在参数量在70亿以下的语言模型中引入高达125%的性能提升。这种方法强调了开源倡议在民主化AI研究和提高高质量计算语言学应用的可访问性方面的能力。 

---
# DYNAMAX: Dynamic computing for Transformers and Mamba based architectures 

**Title (ZH)**: DYNAMAX: 动态计算用于Transformer和Mamba基架构 

**Authors**: Miguel Nogales, Matteo Gambella, Manuel Roveri  

**Link**: [PDF](https://arxiv.org/pdf/2504.20922)  

**Abstract**: Early exits (EEs) offer a promising approach to reducing computational costs and latency by dynamically terminating inference once a satisfactory prediction confidence on a data sample is achieved. Although many works integrate EEs into encoder-only Transformers, their application to decoder-only architectures and, more importantly, Mamba models, a novel family of state-space architectures in the LLM realm, remains insufficiently explored. This work introduces DYNAMAX, the first framework to exploit the unique properties of Mamba architectures for early exit mechanisms. We not only integrate EEs into Mamba but also repurpose Mamba as an efficient EE classifier for both Mamba-based and transformer-based LLMs, showcasing its versatility. Our experiments employ the Mistral 7B transformer compared to the Codestral 7B Mamba model, using data sets such as TruthfulQA, CoQA, and TriviaQA to evaluate computational savings, accuracy, and consistency. The results highlight the adaptability of Mamba as a powerful EE classifier and its efficiency in balancing computational cost and performance quality across NLP tasks. By leveraging Mamba's inherent design for dynamic processing, we open pathways for scalable and efficient inference in embedded applications and resource-constrained environments. This study underscores the transformative potential of Mamba in redefining dynamic computing paradigms for LLMs. 

**Abstract (ZH)**: Early Exits for Mamba Models: Exploiting Unique Properties for Efficient Inference in LLMs 

---
# Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers 

**Title (ZH)**: 分类器偏差检测：迈向无监督自动偏差检测的可视化分类器 

**Authors**: Quentin Guimard, Moreno D'Incà, Massimiliano Mancini, Elisa Ricci  

**Link**: [PDF](https://arxiv.org/pdf/2504.20902)  

**Abstract**: A person downloading a pre-trained model from the web should be aware of its biases. Existing approaches for bias identification rely on datasets containing labels for the task of interest, something that a non-expert may not have access to, or may not have the necessary resources to collect: this greatly limits the number of tasks where model biases can be identified. In this work, we present Classifier-to-Bias (C2B), the first bias discovery framework that works without access to any labeled data: it only relies on a textual description of the classification task to identify biases in the target classification model. This description is fed to a large language model to generate bias proposals and corresponding captions depicting biases together with task-specific target labels. A retrieval model collects images for those captions, which are then used to assess the accuracy of the model w.r.t. the given biases. C2B is training-free, does not require any annotations, has no constraints on the list of biases, and can be applied to any pre-trained model on any classification task. Experiments on two publicly available datasets show that C2B discovers biases beyond those of the original datasets and outperforms a recent state-of-the-art bias detection baseline that relies on task-specific annotations, being a promising first step toward addressing task-agnostic unsupervised bias detection. 

**Abstract (ZH)**: 从网络下载预训练模型的人应意识到其偏差。现有的偏见识别方法依赖于包含目标任务标签的数据集，这可能是非专家无法访问的，或无法收集足够资源获取的：这极大地限制了可以识别模型偏见的任务数量。在本文中，我们提出了Classifier-to-Bias (C2B)，这是第一个无需访问任何标注数据的偏见发现框架：它仅依赖于分类任务的文字描述来识别目标分类模型中的偏见。该描述被输入到大型语言模型中，以生成偏见提案及其对应的描述偏见的图像，并结合具体任务的目标标签。检索模型收集这些描述的图像，然后用于评估模型在给定偏见方面的准确性。C2B 是无需训练的，不需要任何标注，没有偏见列表的限制，并可以应用于任何分类任务的预训练模型。在两个公开数据集上的实验表明，C2B 发现了超出原始数据集的偏见，并优于依赖于特定任务标注的最新偏见检测基准，这朝着实现任务无关的无监督偏见检测迈出了有希望的第一步。 

---
# X-Cross: Dynamic Integration of Language Models for Cross-Domain Sequential Recommendation 

**Title (ZH)**: X-交叉：跨域序列推荐中的语言模型动态集成 

**Authors**: Guy Hadad, Haggai Roitman, Yotam Eshel, Bracha Shapira, Lior Rokach  

**Link**: [PDF](https://arxiv.org/pdf/2504.20859)  

**Abstract**: As new products are emerging daily, recommendation systems are required to quickly adapt to possible new domains without needing extensive retraining. This work presents ``X-Cross'' -- a novel cross-domain sequential-recommendation model that recommends products in new domains by integrating several domain-specific language models; each model is fine-tuned with low-rank adapters (LoRA). Given a recommendation prompt, operating layer by layer, X-Cross dynamically refines the representation of each source language model by integrating knowledge from all other models. These refined representations are propagated from one layer to the next, leveraging the activations from each domain adapter to ensure domain-specific nuances are preserved while enabling adaptability across domains. Using Amazon datasets for sequential recommendation, X-Cross achieves performance comparable to a model that is fine-tuned with LoRA, while using only 25% of the additional parameters. In cross-domain tasks, such as adapting from Toys domain to Tools, Electronics or Sports, X-Cross demonstrates robust performance, while requiring about 50%-75% less fine-tuning data than LoRA to make fine-tuning effective. Furthermore, X-Cross achieves significant improvement in accuracy over alternative cross-domain baselines. Overall, X-Cross enables scalable and adaptive cross-domain recommendations, reducing computational overhead and providing an efficient solution for data-constrained environments. 

**Abstract (ZH)**: 随着新产品不断涌现，推荐系统需要快速适应新的领域而无需进行大量重新训练。本文提出了“X-Cross”——一种新型跨域序列推荐模型，通过集成多个领域特定的语言模型进行产品推荐；每个模型使用低秩适配器（LoRA）进行微调。给定一个推荐提示，X-Cross逐层操作，动态地通过整合其他所有模型的知识来细化每个源语言模型的表示。这些细化后的表示会在各层之间传递，利用每个领域适配器的激活，确保保持领域特定的细微差别，同时在跨域之间实现适应性。使用Amazon数据集进行序列推荐，X-Cross在附加参数仅为LoRA模型的25%的情况下，实现了与之相当的性能。在跨域任务中，如从Toys领域适应到Tools、Electronics或Sports等领域，X-Cross表现出稳健的性能，所需微调数据量仅为LoRA的50%-75%。此外，X-Cross在与替代跨域基线相比在准确率上取得了显著改进。总体而言，X-Cross使跨域推荐具有可扩展性和适应性，减少了计算开销，并为数据受限的环境提供了高效的解决方案。 

---
# Reinforcement Learning for LLM Reasoning Under Memory Constraints 

**Title (ZH)**: 基于内存约束条件下的大语言模型推理强化学习 

**Authors**: Alan Lee, Harry Tong  

**Link**: [PDF](https://arxiv.org/pdf/2504.20834)  

**Abstract**: We explore reinforcement learning (RL) techniques to enhance reasoning within targeted problem spaces in large language models (LLMs) under memory and compute constraints. Our focus is on critic-free methods compatible with LoRA fine-tuning on a single 40GB GPU, a common limitation in academic settings. We introduce S-GRPO, a memory-efficient variant of Group Relative Policy Optimization, and T-SPMO, a token-level prefix matching strategy for fine-grained credit assignment. Despite limited resources, when used to fine-tune Qwen2-1.5B both methods significantly improve SVAMP benchmark accuracy from 46% to above 70% using LoRA training. T-SPMO also excels in multi-digit multiplication tasks, underscoring the potential of RL fine-tuning under hardware constraints. Additionally, we find that our full-token GRPO baseline under LoRA fine-tuning did not improve model performance (compared to base model) on either task, suggesting that our memory-efficient methods may act as a form of regularization that stabilizes training when only a small subset of parameters are updated. 

**Abstract (ZH)**: 我们探索在大语言模型（LLMs）的内存和计算约束条件下，强化学习（RL）技术在特定问题空间内增强推理的方法。我们的重点在于与LoRA单卡（40GB GPU）微调兼容的无价值函数方法。我们引入了S-GRPO，一种记忆高效的Group Relative Policy Optimization变体，以及T-SPMO，一种基于token级别的前缀匹配策略，用于细粒度的责任分配。尽管资源有限，这两种方法在使用LoRA训练时都能显著提高Qwen2-1.5B模型在SVAMP基准测试中的准确性，从46%提高到超过70%。T-SPMO在多位数乘法任务中也表现出色，突显了在硬件约束下RL微调的潜力。此外，我们发现，在LoRA微调下的完整token GRPO基线并未提高模型在这两个任务上的性能（与基础模型相比），这表明我们的记忆高效方法可能作为一种正则化手段，在仅更新一小部分参数时稳定训练。 

---
# Hallucination by Code Generation LLMs: Taxonomy, Benchmarks, Mitigation, and Challenges 

**Title (ZH)**: 代码生成型LLMs的幻觉现象：分类、基准、缓解与挑战 

**Authors**: Yunseo Lee, John Youngeun Song, Dongsun Kim, Jindae Kim, Mijung Kim, Jaechang Nam  

**Link**: [PDF](https://arxiv.org/pdf/2504.20799)  

**Abstract**: Recent technical breakthroughs in large language models (LLMs) have enabled them to fluently generate source code. Software developers often leverage both general-purpose and code-specialized LLMs to revise existing code or even generate a whole function from scratch. These capabilities are also beneficial in no-code or low-code contexts, in which one can write programs without a technical background. However, due to their internal design, LLMs are prone to generating hallucinations, which are incorrect, nonsensical, and not justifiable information but difficult to identify its presence. This problem also occurs when generating source code. Once hallucinated code is produced, it is often challenging for users to identify and fix it, especially when such hallucinations can be identified under specific execution paths. As a result, the hallucinated code may remain unnoticed within the codebase. This survey investigates recent studies and techniques relevant to hallucinations generated by CodeLLMs. We categorize the types of hallucinations in the code generated by CodeLLMs, review existing benchmarks and mitigation strategies, and identify open challenges. Based on these findings, this survey outlines further research directions in the detection and removal of hallucinations produced by CodeLLMs. 

**Abstract (ZH)**: 最近在大规模语言模型（LLMs）领域的技术突破使得它们能够流畅地生成源代码。软件开发人员常常利用通用和代码专用的LLMs来修订现有代码，甚至从头生成一个完整的功能。这些能力在无代码或低代码环境中也非常有益，在这种环境中，人们可以在没有技术背景的情况下编写程序。然而，由于其内部设计，LLMs容易生成幻觉，这些幻觉是错误的、没有意义的且难以验证的信息，但难以识别它们的存在。这一问题同样出现在生成源代码的过程中。一旦生成了幻觉代码，用户往往难以识别和修复，特别是当这些幻觉在特定执行路径下才能被识别时，幻觉代码可能会在代码库中被忽视。本文综述了与CodeLLMs生成的幻觉相关的近期研究和技术。我们对CodeLLMs生成的代码中的幻觉类型进行了分类，回顾了现有的基准测试和缓解策略，并识别出开放挑战。基于这些发现，本文概述了进一步研究的方向，包括在CodeLLMs生成的幻觉检测和移除方面的研究方向。 

---
# Using LLMs in Generating Design Rationale for Software Architecture Decisions 

**Title (ZH)**: 使用LLMs生成软件架构决策 reasoning 

**Authors**: Xiyu Zhou, Ruiyin Li, Peng Liang, Beiqi Zhang, Mojtaba Shahin, Zengyang Li, Chen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20781)  

**Abstract**: Design Rationale (DR) for software architecture decisions refers to the reasoning underlying architectural choices, which provides valuable insights into the different phases of the architecting process throughout software development. However, in practice, DR is often inadequately documented due to a lack of motivation and effort from developers. With the recent advancements in Large Language Models (LLMs), their capabilities in text comprehension, reasoning, and generation may enable the generation and recovery of DR for architecture decisions. In this study, we evaluated the performance of LLMs in generating DR for architecture decisions. First, we collected 50 Stack Overflow (SO) posts, 25 GitHub issues, and 25 GitHub discussions related to architecture decisions to construct a dataset of 100 architecture-related problems. Then, we selected five LLMs to generate DR for the architecture decisions with three prompting strategies, including zero-shot, chain of thought (CoT), and LLM-based agents. With the DR provided by human experts as ground truth, the Precision of LLM-generated DR with the three prompting strategies ranges from 0.267 to 0.278, Recall from 0.627 to 0.715, and F1-score from 0.351 to 0.389. Additionally, 64.45% to 69.42% of the arguments of DR not mentioned by human experts are also helpful, 4.12% to 4.87% of the arguments have uncertain correctness, and 1.59% to 3.24% of the arguments are potentially misleading. Based on the results, we further discussed the pros and cons of the three prompting strategies and the strengths and limitations of the DR generated by LLMs. 

**Abstract (ZH)**: 面向软件架构决策的设计 rationale (DR) 生成：基于大型语言模型的性能评估 

---
# Chain-of-Defensive-Thought: Structured Reasoning Elicits Robustness in Large Language Models against Reference Corruption 

**Title (ZH)**: 防御性思维链：结构化推理在大型语言模型中对抗参考污染增强鲁棒性 

**Authors**: Wenxiao Wang, Parsa Hosseini, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2504.20769)  

**Abstract**: Chain-of-thought prompting has demonstrated great success in facilitating the reasoning abilities of large language models. In this work, we explore how these enhanced reasoning abilities can be exploited to improve the robustness of large language models in tasks that are not necessarily reasoning-focused. In particular, we show how a wide range of large language models exhibit significantly improved robustness against reference corruption using a simple method called chain-of-defensive-thought, where only a few exemplars with structured and defensive reasoning are provided as demonstrations. Empirically, the improvements can be astounding, especially given the simplicity and applicability of the method. For example, in the Natural Questions task, the accuracy of GPT-4o degrades from 60% to as low as 3% with standard prompting when 1 out of 10 references provided is corrupted with prompt injection attacks. In contrast, GPT-4o using chain-of-defensive-thought prompting maintains an accuracy of 50%. 

**Abstract (ZH)**: Chain-of-defensive-thought prompting 通过提升大型语言模型的鲁棒性以改进非推理密集任务的表现 

---
# UniversalRAG: Retrieval-Augmented Generation over Multiple Corpora with Diverse Modalities and Granularities 

**Title (ZH)**: UniversalRAG：跨多语料库的多元模态和粒度检索增强生成 

**Authors**: Woongyeong Yeo, Kangsan Kim, Soyeong Jeong, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20734)  

**Abstract**: Retrieval-Augmented Generation (RAG) has shown substantial promise in improving factual accuracy by grounding model responses with external knowledge relevant to queries. However, most existing RAG approaches are limited to a text-only corpus, and while recent efforts have extended RAG to other modalities such as images and videos, they typically operate over a single modality-specific corpus. In contrast, real-world queries vary widely in the type of knowledge they require, which a single type of knowledge source cannot address. To address this, we introduce UniversalRAG, a novel RAG framework designed to retrieve and integrate knowledge from heterogeneous sources with diverse modalities and granularities. Specifically, motivated by the observation that forcing all modalities into a unified representation space derived from a single combined corpus causes a modality gap, where the retrieval tends to favor items from the same modality as the query, we propose a modality-aware routing mechanism that dynamically identifies the most appropriate modality-specific corpus and performs targeted retrieval within it. Also, beyond modality, we organize each modality into multiple granularity levels, enabling fine-tuned retrieval tailored to the complexity and scope of the query. We validate UniversalRAG on 8 benchmarks spanning multiple modalities, showing its superiority over modality-specific and unified baselines. 

**Abstract (ZH)**: 通用RAG：跨模态异质知识检索与整合 

---
# Beyond the Last Answer: Your Reasoning Trace Uncovers More than You Think 

**Title (ZH)**: 超越最后一个答案：你的推理轨迹揭示的不止你所想 

**Authors**: Hasan Abed Al Kader Hammoud, Hani Itani, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2504.20708)  

**Abstract**: Large Language Models (LLMs) leverage step-by-step reasoning to solve complex problems. Standard evaluation practice involves generating a complete reasoning trace and assessing the correctness of the final answer presented at its conclusion. In this paper, we challenge the reliance on the final answer by posing the following two questions: Does the final answer reliably represent the model's optimal conclusion? Can alternative reasoning paths yield different results? To answer these questions, we analyze intermediate reasoning steps, termed subthoughts, and propose a method based on our findings. Our approach involves segmenting a reasoning trace into sequential subthoughts based on linguistic cues. We start by prompting the model to generate continuations from the end-point of each intermediate subthought. We extract a potential answer from every completed continuation originating from different subthoughts. We find that aggregating these answers by selecting the most frequent one (the mode) often yields significantly higher accuracy compared to relying solely on the answer derived from the original complete trace. Analyzing the consistency among the answers derived from different subthoughts reveals characteristics that correlate with the model's confidence and correctness, suggesting potential for identifying less reliable answers. Our experiments across various LLMs and challenging mathematical reasoning datasets (AIME2024 and AIME2025) show consistent accuracy improvements, with gains reaching up to 13\% and 10\% respectively. Implementation is available at: this https URL. 

**Abstract (ZH)**: 大型语言模型通过逐步推理解决复杂问题。标准评估实践涉及生成完整的推理过程并评估最终答案的正确性。本文通过提出以下两个问题挑战对最终答案的依赖：最终答案是否可靠地代表了模型的最优结论？是否存在不同的推理路径可以产生不同的结果？为回答这些问题，我们分析了中间推理步骤（称为子思想），并基于我们的发现提出了一种方法。我们的方法涉及根据语言线索将推理过程划分为顺序的子思想片段。我们首先提示模型从每个中间子思想的终点生成续写。我们从每个完成的续写中提取潜在答案。我们发现，通过选择最常见的答案（众数）来聚合这些答案，通常比仅依赖原始完整过程生成的答案具有更高的准确率。分析来自不同子思想的解答一致性揭示了与模型的信心和正确性相关的特征，表明可以用于识别不可靠的答案。我们在多种大型语言模型和具有挑战性的数学推理数据集（AIME2024和AIME2025）上的实验显示了一致的准确率提高，分别达到13%和10%。详细的实现可在以下链接获取：this https URL。 

---
# Can LLMs Detect Intrinsic Hallucinations in Paraphrasing and Machine Translation? 

**Title (ZH)**: LLMs能否检测重写和机器翻译中的内在幻觉？ 

**Authors**: Evangelia Gogoulou, Shorouq Zahra, Liane Guillou, Luise Dürlich, Joakim Nivre  

**Link**: [PDF](https://arxiv.org/pdf/2504.20699)  

**Abstract**: A frequently observed problem with LLMs is their tendency to generate output that is nonsensical, illogical, or factually incorrect, often referred to broadly as hallucination. Building on the recently proposed HalluciGen task for hallucination detection and generation, we evaluate a suite of open-access LLMs on their ability to detect intrinsic hallucinations in two conditional generation tasks: translation and paraphrasing. We study how model performance varies across tasks and language and we investigate the impact of model size, instruction tuning, and prompt choice. We find that performance varies across models but is consistent across prompts. Finally, we find that NLI models perform comparably well, suggesting that LLM-based detectors are not the only viable option for this specific task. 

**Abstract (ZH)**: LLMs生成虚假信息的常见问题及其检测研究：基于HalluciGen任务在翻译和 paraphrasing任务中的表现分析 

---
# CoCo-Bench: A Comprehensive Code Benchmark For Multi-task Large Language Model Evaluation 

**Title (ZH)**: CoCo-Bench: 一种全面的多任务大型语言模型评估代码基准 

**Authors**: Wenjing Yin, Tianze Sun, Yijiong Yu, Jiawei Fang, Guangyao Su, Jiancheng Wang, Zekun Wang, Wei Wang, Ran Chen, Ziyun Dai, Shuai Yuan, Menghang Dong, Peng Luo, Dong Cao, Da Lei, Yajun Zhang, Hao Chen, Xiang Ma, Yong Liu, Weifeng Liu, Yuanjian Xu, Ji Pei  

**Link**: [PDF](https://arxiv.org/pdf/2504.20673)  

**Abstract**: Large language models (LLMs) play a crucial role in software engineering, excelling in tasks like code generation and maintenance. However, existing benchmarks are often narrow in scope, focusing on a specific task and lack a comprehensive evaluation framework that reflects real-world applications. To address these gaps, we introduce CoCo-Bench (Comprehensive Code Benchmark), designed to evaluate LLMs across four critical dimensions: code understanding, code generation, code modification, and code review. These dimensions capture essential developer needs, ensuring a more systematic and representative evaluation. CoCo-Bench includes multiple programming languages and varying task difficulties, with rigorous manual review to ensure data quality and accuracy. Empirical results show that CoCo-Bench aligns with existing benchmarks while uncovering significant variations in model performance, effectively highlighting strengths and weaknesses. By offering a holistic and objective evaluation, CoCo-Bench provides valuable insights to guide future research and technological advancements in code-oriented LLMs, establishing a reliable benchmark for the field. 

**Abstract (ZH)**: 大型语言模型（LLMs）在软件工程中发挥着关键作用，尤其在代码生成和维护任务中表现出色。然而，现有的基准测试往往范围狭窄，仅集中在特定任务上，缺乏反映实际应用的全面评估框架。为弥补这些不足，我们引入了CoCo-Bench（综合代码基准），旨在从代码理解和认知、代码生成、代码修改和代码审查四个关键维度评估LLMs。这些维度涵盖了开发人员的基本需求，确保评估更为系统和具有代表性。CoCo-Bench涵盖了多种编程语言和不同难度的任务，并通过严格的手工审查确保数据的质量和准确性。实证结果表明，CoCo-Bench与现有的基准测试相一致，但揭示了模型性能中的显著差异，有效突显了模型的优势和弱点。通过提供一个全面且客观的评估，CoCo-Bench为代码导向的LLMs的研究和技术创新提供了宝贵的洞察，并为该领域建立了可靠基准。 

---
# Cooking Up Creativity: A Cognitively-Inspired Approach for Enhancing LLM Creativity through Structured Representations 

**Title (ZH)**: 激发创造力：一种基于认知的结构表示方法以增强大语言模型的创造力 

**Authors**: Moran Mizrahi, Chen Shani, Gabriel Stanovsky, Dan Jurafsky, Dafna Shahaf  

**Link**: [PDF](https://arxiv.org/pdf/2504.20643)  

**Abstract**: Large Language Models (LLMs) excel at countless tasks, yet struggle with creativity. In this paper, we introduce a novel approach that couples LLMs with structured representations and cognitively inspired manipulations to generate more creative and diverse ideas. Our notion of creativity goes beyond superficial token-level variations; rather, we explicitly recombine structured representations of existing ideas, allowing our algorithm to effectively explore the more abstract landscape of ideas. We demonstrate our approach in the culinary domain with DishCOVER, a model that generates creative recipes. Experiments comparing our model's results to those of GPT-4o show greater diversity. Domain expert evaluations reveal that our outputs, which are mostly coherent and feasible culinary creations, significantly surpass GPT-4o in terms of novelty, thus outperforming it in creative generation. We hope our work inspires further research into structured creativity in AI. 

**Abstract (ZH)**: 大型语言模型在创造力方面表现出色，但存在局限性。本论文介绍了一种新型方法，将大型语言模型与结构化表示和认知启发式的操作相结合，以生成更具创造力和多样性的想法。我们对创造力的理解不仅仅局限于表层的令牌级变化，而是明确地重组现有想法的结构化表示，从而使我们的算法能够有效地探索更具抽象性的想法空间。我们在烹饪领域通过DishCOVER模型展示了该方法，该模型能够生成创意菜谱。实验结果表明，与GPT-4o相比，我们的模型结果更具多样性。领域专家评估表明，与GPT-4o相比，我们生成的大部分连贯且可行的烹饪创作具有更高的新颖性，因此在创造性生成方面优于GPT-4o。我们希望我们的工作能够激发对AI中结构化创造力的进一步研究。 

---
# The Hidden Risks of LLM-Generated Web Application Code: A Security-Centric Evaluation of Code Generation Capabilities in Large Language Models 

**Title (ZH)**: 大型语言模型生成代码的安全风险探究：基于安全性的代码生成能力评估 

**Authors**: Swaroop Dora, Deven Lunkad, Naziya Aslam, S. Venkatesan, Sandeep Kumar Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2504.20612)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has enhanced software development processes, minimizing the time and effort required for coding and enhancing developer productivity. However, despite their potential benefits, code generated by LLMs has been shown to generate insecure code in controlled environments, raising critical concerns about their reliability and security in real-world applications. This paper uses predefined security parameters to evaluate the security compliance of LLM-generated code across multiple models, such as ChatGPT, DeepSeek, Claude, Gemini and Grok. The analysis reveals critical vulnerabilities in authentication mechanisms, session management, input validation and HTTP security headers. Although some models implement security measures to a limited extent, none fully align with industry best practices, highlighting the associated risks in automated software development. Our findings underscore that human expertise is crucial to ensure secure software deployment or review of LLM-generated code. Also, there is a need for robust security assessment frameworks to enhance the reliability of LLM-generated code in real-world applications. 

**Abstract (ZH)**: 大语言模型的快速进步提高了软件开发过程的效率，减少了编码所需的时间和努力，提升了开发人员的 productivity。然而，尽管它们具有潜在的好处，但在受控环境中生成的代码显示出安全性问题，这引发了关于它们在实际应用中的可靠性和安全性的重要关切。本文使用预定义的安全参数来评估大语言模型生成的代码在多个模型（如ChatGPT、DeepSeek、Claude、Gemini和Grok）中的安全合规性。分析揭示了身份验证机制、会话管理、输入验证和HTTP安全标头中的关键漏洞。尽管一些模型在一定程度上实施了安全措施，但 none 完全符合行业最佳实践，突显了自动化软件开发相关的风险。我们的研究强调，人类专业知识对于确保安全地部署或审查大语言模型生成的代码至关重要。此外，需要建立 robust 安全评估框架来提高大语言模型生成的代码在实际应用中的可靠性。 

---
# Information Retrieval in the Age of Generative AI: The RGB Model 

**Title (ZH)**: 生成式AI时代的信息系统检索：RGB模型 

**Authors**: Michele Garetto, Alessandro Cornacchia, Franco Galante, Emilio Leonardi, Alessandro Nordio, Alberto Tarable  

**Link**: [PDF](https://arxiv.org/pdf/2504.20610)  

**Abstract**: The advent of Large Language Models (LLMs) and generative AI is fundamentally transforming information retrieval and processing on the Internet, bringing both great potential and significant concerns regarding content authenticity and reliability. This paper presents a novel quantitative approach to shed light on the complex information dynamics arising from the growing use of generative AI tools. Despite their significant impact on the digital ecosystem, these dynamics remain largely uncharted and poorly understood. We propose a stochastic model to characterize the generation, indexing, and dissemination of information in response to new topics. This scenario particularly challenges current LLMs, which often rely on real-time Retrieval-Augmented Generation (RAG) techniques to overcome their static knowledge limitations. Our findings suggest that the rapid pace of generative AI adoption, combined with increasing user reliance, can outpace human verification, escalating the risk of inaccurate information proliferation across digital resources. An in-depth analysis of Stack Exchange data confirms that high-quality answers inevitably require substantial time and human effort to emerge. This underscores the considerable risks associated with generating persuasive text in response to new questions and highlights the critical need for responsible development and deployment of future generative AI tools. 

**Abstract (ZH)**: 大型语言模型和生成式AI的兴起正在根本性地改变互联网上的信息检索和处理，带来了内容真实性与可靠性的重要潜在和关切。本文提出了一种新颖的数量化方法，以揭示生成式AI工具广泛应用带来的复杂信息动态。尽管这些工具对数字生态系统产生了重大影响，但这些动态仍大多未被探索和理解。我们提出了一种随机模型来表征对新主题的生成、索引和传播。这一情境尤其挑战当前的大型语言模型，它们通常依赖于实时检索增强生成（RAG）技术来克服其静态知识的局限。我们的研究结果表明，生成式AI的快速采纳速度与用户依赖性的增加可能导致人工验证滞后，从而加剧数字资源中不准确信息传播的风险。对Stack Exchange数据的深入分析证实，高质量的回答不可避免地需要大量时间和人力才能产生。这突显了生成针对新问题具有说服力的文本所面临的重要风险，并强调了负责任地开发和部署未来生成式AI工具的迫切需要。 

---
# Reinforcement Learning for Reasoning in Large Language Models with One Training Example 

**Title (ZH)**: 在单个训练样本情况下，强化学习在大型语言模型推理中的应用 

**Authors**: Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Lucas Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, Yelong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20571)  

**Abstract**: We show that reinforcement learning with verifiable reward using one training example (1-shot RLVR) is effective in incentivizing the math reasoning capabilities of large language models (LLMs). Applying RLVR to the base model Qwen2.5-Math-1.5B, we identify a single example that elevates model performance on MATH500 from 36.0% to 73.6%, and improves the average performance across six common mathematical reasoning benchmarks from 17.6% to 35.7%. This result matches the performance obtained using the 1.2k DeepScaleR subset (MATH500: 73.6%, average: 35.9%), which includes the aforementioned example. Similar substantial improvements are observed across various models (Qwen2.5-Math-7B, Llama3.2-3B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B), RL algorithms (GRPO and PPO), and different math examples (many of which yield approximately 30% or greater improvement on MATH500 when employed as a single training example). In addition, we identify some interesting phenomena during 1-shot RLVR, including cross-domain generalization, increased frequency of self-reflection, and sustained test performance improvement even after the training accuracy has saturated, a phenomenon we term post-saturation generalization. Moreover, we verify that the effectiveness of 1-shot RLVR primarily arises from the policy gradient loss, distinguishing it from the "grokking" phenomenon. We also show the critical role of promoting exploration (e.g., by adding entropy loss with an appropriate coefficient) in 1-shot RLVR training. As a bonus, we observe that applying entropy loss alone, without any outcome reward, significantly enhances Qwen2.5-Math-1.5B's performance on MATH500 by 27.4%. These findings can inspire future work on RLVR data efficiency and encourage a re-examination of both recent progress and the underlying mechanisms in RLVR. Our code, model, and data are open source at this https URL 

**Abstract (ZH)**: 我们展示了使用验证性奖励的一次训练示例强化学习（1-shot RLVR）在激励大型语言模型的数学推理能力方面是有效的。将RLVR应用于基模型Qwen2.5-Math-1.5B，我们发现一个单一示例将模型在MATH500上的性能提升至73.6%，并在六个常见数学推理基准测试中提高了平均性能从17.6%到35.7%。该结果与使用1.2k DeepScaleR子集（MATH500：73.6%，平均：35.9%）的效果一致，包括上述示例。在不同模型（Qwen2.5-Math-7B、Llama3.2-3B-Instruct、DeepSeek-R1-Distill-Qwen-1.5B）、不同强化学习算法（GRPO和PPO）和不同类型数学示例（许多示例在作为单个训练示例时在MATH500上均实现了约30%或更高的性能提升）上观察到类似的显著改进。此外，在1-shot RLVR过程中，我们发现了跨域泛化、自我反思频率增加以及训练准确度饱和后持续提高测试性能等有趣现象，我们将这一现象称为后饱和泛化。我们还验证了1-shot RLVR的有效性主要来源于策略梯度损失，将其与其他现象区分开来。我们展示了在1-shot RLVR训练中促进探索（例如通过适当系数添加熵损失）的重要性。此外，我们发现仅应用熵损失而无任何结果奖励，显著提升了Qwen2.5-Math-1.5B在MATH500上的性能27.4%。这些发现可以激发未来RLVR数据效率方面的工作，并鼓励重新审视RLVR的近期进展及其背后的机制。我们的代码、模型和数据在此处开放获取。 

---
# Token-Efficient Prompt Injection Attack: Provoking Cessation in LLM Reasoning via Adaptive Token Compression 

**Title (ZH)**: 基于Token高效注入攻击：通过自适应Token压缩促使大规模语言模型推理停止 

**Authors**: Yu Cui, Yujun Cai, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20493)  

**Abstract**: While reasoning large language models (LLMs) demonstrate remarkable performance across various tasks, they also contain notable security vulnerabilities. Recent research has uncovered a "thinking-stopped" vulnerability in DeepSeek-R1, where model-generated reasoning tokens can forcibly interrupt the inference process, resulting in empty responses that compromise LLM-integrated applications. However, existing methods triggering this vulnerability require complex mathematical word problems with long prompts--even exceeding 5,000 tokens. To reduce the token cost and formally define this vulnerability, we propose a novel prompt injection attack named "Reasoning Interruption Attack", based on adaptive token compression. We demonstrate that simple standalone arithmetic tasks can effectively trigger this vulnerability, and the prompts based on such tasks exhibit simpler logical structures than mathematical word problems. We develop a systematic approach to efficiently collect attack prompts and an adaptive token compression framework that utilizes LLMs to automatically compress these prompts. Experiments show our compression framework significantly reduces prompt length while maintaining effective attack capabilities. We further investigate the attack's performance via output prefix and analyze the underlying causes of the vulnerability, providing valuable insights for improving security in reasoning LLMs. 

**Abstract (ZH)**: 基于自适应token压缩的推理中断攻击 

---
# Enhancing News Recommendation with Hierarchical LLM Prompting 

**Title (ZH)**: 基于层次化大语言模型提示的新闻推荐增强 

**Authors**: Hai-Dang Kieu, Delvin Ce Zhang, Minh Duc Nguyen, Min Xu, Qiang Wu, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2504.20452)  

**Abstract**: Personalized news recommendation systems often struggle to effectively capture the complexity of user preferences, as they rely heavily on shallow representations, such as article titles and abstracts. To address this problem, we introduce a novel method, namely PNR-LLM, for Large Language Models for Personalized News Recommendation. Specifically, PNR-LLM harnesses the generation capabilities of LLMs to enrich news titles and abstracts, and consequently improves recommendation quality. PNR-LLM contains a novel module, News Enrichment via LLMs, which generates deeper semantic information and relevant entities from articles, transforming shallow contents into richer representations. We further propose an attention mechanism to aggregate enriched semantic- and entity-level data, forming unified user and news embeddings that reveal a more accurate user-news match. Extensive experiments on MIND datasets show that PNR-LLM outperforms state-of-the-art baselines. Moreover, the proposed data enrichment module is model-agnostic, and we empirically show that applying our proposed module to multiple existing models can further improve their performance, verifying the advantage of our design. 

**Abstract (ZH)**: 基于大规模语言模型的个性化新闻推荐系统PNR-LLM 

---
# On Psychology of AI -- Does Primacy Effect Affect ChatGPT and Other LLMs? 

**Title (ZH)**: AI心理学——首因效应是否影响ChatGPT及其他大语言模型？ 

**Authors**: Mika Hämäläinen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20444)  

**Abstract**: We study the primacy effect in three commercial LLMs: ChatGPT, Gemini and Claude. We do this by repurposing the famous experiment Asch (1946) conducted using human subjects. The experiment is simple, given two candidates with equal descriptions which one is preferred if one description has positive adjectives first before negative ones and another description has negative adjectives followed by positive ones. We test this in two experiments. In one experiment, LLMs are given both candidates simultaneously in the same prompt, and in another experiment, LLMs are given both candidates separately. We test all the models with 200 candidate pairs. We found that, in the first experiment, ChatGPT preferred the candidate with positive adjectives listed first, while Gemini preferred both equally often. Claude refused to make a choice. In the second experiment, ChatGPT and Claude were most likely to rank both candidates equally. In the case where they did not give an equal rating, both showed a clear preference to a candidate that had negative adjectives listed first. Gemini was most likely to prefer a candidate with negative adjectives listed first. 

**Abstract (ZH)**: 我们在ChatGPT、Gemini和Claude三种商业大语言模型中研究先觉效应，并通过重新利用Asch（1946年）使用人类受试者进行的经典实验来进行。实验简单来说，给定两个描述相同但顺序不同的候选者，一个是正面形容词在前、负面形容词在后，另一个是负面形容词在前、正面形容词在后，哪一个更受欢迎。我们进行了两项实验。在第一个实验中，我们将两个候选者同时呈现给模型；在第二个实验中，我们将两个候选者分别呈现给模型。我们测试了所有模型共计200对候选者。结果发现，在第一个实验中，ChatGPT更偏好正面形容词在前的候选者，Gemini两者偏好程度相当，Claude则不愿意做出选择。在第二个实验中，ChatGPT和Claude最可能将两个候选者评定为相同。当它们不给出相同评分时，两者都更倾向于负面形容词在前的候选者。Gemini更倾向于正面形容词在后的候选者。 

---
# GaLore 2: Large-Scale LLM Pre-Training by Gradient Low-Rank Projection 

**Title (ZH)**: GaLore 2：大规模LLM预训练的梯度低秩投影方法 

**Authors**: DiJia Su, Andrew Gu, Jane Xu, Yuandong Tian, Jiawei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.20437)  

**Abstract**: Large language models (LLMs) have revolutionized natural language understanding and generation but face significant memory bottlenecks during training. GaLore, Gradient Low-Rank Projection, addresses this issue by leveraging the inherent low-rank structure of weight gradients, enabling substantial memory savings without sacrificing performance. Recent works further extend GaLore from various aspects, including low-bit quantization and higher-order tensor structures. However, there are several remaining challenges for GaLore, such as the computational overhead of SVD for subspace updates and the integration with state-of-the-art training parallelization strategies (e.g., FSDP). In this paper, we present GaLore 2, an efficient and scalable GaLore framework that addresses these challenges and incorporates recent advancements. In addition, we demonstrate the scalability of GaLore 2 by pre-training Llama 7B from scratch using up to 500 billion training tokens, highlighting its potential impact on real LLM pre-training scenarios. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言理解和生成方面带来了革命性的变化，但训练过程中面临显著的内存瓶颈。GaLore，梯度低秩投影，通过利用权重梯度的固有低秩结构，解决了这一问题，在不牺牲性能的情况下实现了显著的内存节省。最近的研究进一步从低比特量化和高阶张量结构等方面扩展了GaLore。然而，GaLore仍面临一些挑战，包括子空间更新的SVD计算开销以及与最先进的训练并行化策略（如FSDP）的集成。在此论文中，我们提出了一种高效的可扩展的GaLore框架，以解决这些挑战并集成最近的发展成果。此外，我们通过从头开始使用最多5000亿个训练令牌预训练Llama 7B，展示了GaLore 2的可扩展性，突显了其在实际大规模语言模型预训练场景中的潜在影响。 

---
# Local Prompt Optimization 

**Title (ZH)**: 本地提示优化 

**Authors**: Yash Jain, Vishal Chowdhary  

**Link**: [PDF](https://arxiv.org/pdf/2504.20355)  

**Abstract**: In recent years, the use of prompts to guide the output of Large Language Models have increased dramatically. However, even the best of experts struggle to choose the correct words to stitch up a prompt for the desired task. To solve this, LLM driven prompt optimization emerged as an important problem. Existing prompt optimization methods optimize a prompt globally, where in all the prompt tokens have to be optimized over a large vocabulary while solving a complex task. The large optimization space (tokens) leads to insufficient guidance for a better prompt. In this work, we introduce Local Prompt Optimization (LPO) that integrates with any general automatic prompt engineering method. We identify the optimization tokens in a prompt and nudge the LLM to focus only on those tokens in its optimization step. We observe remarkable performance improvements on Math Reasoning (GSM8k and MultiArith) and BIG-bench Hard benchmarks across various automatic prompt engineering methods. Further, we show that LPO converges to the optimal prompt faster than global methods. 

**Abstract (ZH)**: 近年来，使用提示来引导大型语言模型的输出使用大幅增加。然而，即使是顶尖专家也难以选择正确的词汇来拼接出合适的提示以完成所需任务。为此，基于大型语言模型的提示优化成为了一个重要问题。现有的提示优化方法在全球范围内优化提示，这意味着在解决复杂任务时需要在整个庞大的词汇表上优化所有提示标记。由于优化空间（标记）很大，这会导致对更好提示的指导不足。在本文中，我们引入了局部提示优化（LPO），它可以与任何通用自动提示工程方法集成。我们确定了提示中的优化标记，并在优化步骤中引导LLM专注于这些标记。我们在各种自动提示工程方法中对数学推理（GSM8k和MultiArith）和BI GSL硬基准测试中观察到显著的性能提升。此外，我们展示了LPO比全局方法更快地收敛到最优提示。 

---
# CarbonCall: Sustainability-Aware Function Calling for Large Language Models on Edge Devices 

**Title (ZH)**: CarbonCall: Awareness of可持续性在边缘设备上大型语言模型函数调用中的应用 

**Authors**: Varatheepan Paramanayakam, Andreas Karatzas, Iraklis Anagnostopoulos, Dimitrios Stamoulis  

**Link**: [PDF](https://arxiv.org/pdf/2504.20348)  

**Abstract**: Large Language Models (LLMs) enable real-time function calling in edge AI systems but introduce significant computational overhead, leading to high power consumption and carbon emissions. Existing methods optimize for performance while neglecting sustainability, making them inefficient for energy-constrained environments. We introduce CarbonCall, a sustainability-aware function-calling framework that integrates dynamic tool selection, carbon-aware execution, and quantized LLM adaptation. CarbonCall adjusts power thresholds based on real-time carbon intensity forecasts and switches between model variants to sustain high tokens-per-second throughput under power constraints. Experiments on an NVIDIA Jetson AGX Orin show that CarbonCall reduces carbon emissions by up to 52%, power consumption by 30%, and execution time by 30%, while maintaining high efficiency. 

**Abstract (ZH)**: Large Language Models (LLMs)在边缘AI系统中实现实时函数调用但引入了显著的计算开销，导致高功耗和碳排放。现有方法侧重于性能优化而忽视可持续性，使其不适合能量受限环境。我们提出CarbonCall，这是一种具备可持续性意识的函数调用框架，集成动态工具选择、碳意识执行和量化LLM适应。CarbonCall根据实时碳强度预报调整功耗阈值，并在功率约束下切换模型变体以维持高每秒令牌吞吐量。实验表明，与NVIDIA Jetson AGX Orin平台相比，CarbonCall最多可减少52%的碳排放、30%的功耗和30%的执行时间，同时保持高效性。 

---
# Can Large Language Models Learn Formal Logic? A Data-Driven Training and Evaluation Framework 

**Title (ZH)**: 大型语言模型能否学习形式逻辑？一种基于数据的训练与评估框架 

**Authors**: Yuan Xia, Akanksha Atrey, Fadoua Khmaissia, Kedar S. Namjoshi  

**Link**: [PDF](https://arxiv.org/pdf/2504.20213)  

**Abstract**: This paper investigates the logical reasoning capabilities of large language models (LLMs). For a precisely defined yet tractable formulation, we choose the conceptually simple but technically complex task of constructing proofs in Boolean logic. A trained LLM receives as input a set of assumptions and a goal, and produces as output a proof that formally derives the goal from the assumptions. Incorrect proofs are caught by an automated proof checker. A critical obstacle for training is the scarcity of real-world proofs. We propose an efficient, randomized procedure for synthesizing valid proofs and introduce Template Transformation, a data augmentation technique that enhances the model's ability to handle complex logical expressions. The central evaluation question is whether an LLM has indeed learned to reason. We propose tests to measure the reasoning ability of a black-box LLM. By these measures, experiments demonstrate strong reasoning capabilities for assertions with short proofs, which decline with proof complexity. Notably, template transformation improves accuracy even for smaller models, suggesting its effectiveness across model scales. 

**Abstract (ZH)**: 本文考察了大规模语言模型（LLM）的逻辑推理能力。为了选择一个概念简洁但技术上复杂的问题，我们选择了构建布尔逻辑证明的任务。经过训练的LLM接收一组假设和一个目标，并生成一个正式从假设推导出目标的证明。错误的证明由自动证明检查器捕获。训练的关键障碍是现实世界证明的稀缺性。我们提出了一种高效且随机化的证明合成方法，并引入了模板转换数据增强技术，以提高模型处理复杂逻辑表达式的能力。中心评价问题是LLM是否确实学会了推理。我们提出了测试以衡量黑盒LLM的推理能力。通过这些测试，实验表明，对于短证明的断言具有较强的推理能力，但随着证明复杂度的增加而下降。值得注意的是，模板转换即使对较小的模型也能提高准确性，这表明其在不同模型规模上的有效性。 

---
# Prompting LLMs for Code Editing: Struggles and Remedies 

**Title (ZH)**: 提示大模型进行代码编辑：困境与解决方案 

**Authors**: Daye Nam, Ahmed Omran, Ambar Murillo, Saksham Thakur, Abner Araujo, Marcel Blistein, Alexander Frömmgen, Vincent Hellendoorn, Satish Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2504.20196)  

**Abstract**: Large Language Models (LLMs) are rapidly transforming software engineering, with coding assistants embedded in an IDE becoming increasingly prevalent. While research has focused on improving the tools and understanding developer perceptions, a critical gap exists in understanding how developers actually use these tools in their daily workflows, and, crucially, where they struggle. This paper addresses part of this gap through a multi-phased investigation of developer interactions with an LLM-powered code editing and transformation feature, Transform Code, in an IDE widely used at Google. First, we analyze telemetry logs of the feature usage, revealing that frequent re-prompting can be an indicator of developer struggles with using Transform Code. Second, we conduct a qualitative analysis of unsatisfactory requests, identifying five key categories of information often missing from developer prompts. Finally, based on these findings, we propose and evaluate a tool, AutoPrompter, for automatically improving prompts by inferring missing information from the surrounding code context, leading to a 27% improvement in edit correctness on our test set. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在 rapidly transform 软件工程，编码助手嵌入IDE变得日益普遍。尽管研究集中在改进工具和理解开发者感知上，但在开发者实际在日常工作中如何使用这些工具以及他们遇到困难的地方，仍存在一个关键缺口。本文通过多阶段调查开发者与一种由LLM驱动的代码编辑和转换功能Transform Code在广泛用于Google的IDE中的交互，来部分填补这一缺口。首先，我们分析该功能使用的遥测日志，揭示频繁重新提示可能是开发者在使用Transform Code时遇到困难的指标。其次，我们对不满意的请求进行定性分析，确定了开发者提示中经常缺失的五个关键信息类别。最后，基于这些发现，我们提出了并评估了一个名为AutoPrompter的工具，该工具可以通过从附近代码上下文推断缺失信息来自动改进提示，我们的测试集结果显示编辑正确性提高了27%。 

---
# BLADE: Benchmark suite for LLM-driven Automated Design and Evolution of iterative optimisation heuristics 

**Title (ZH)**: BLADE：由大规模语言模型驱动的迭代优化启发式自动化设计与演变基准套件 

**Authors**: Niki van Stein, Anna V. Kononova, Haoran Yin, Thomas Bäck  

**Link**: [PDF](https://arxiv.org/pdf/2504.20183)  

**Abstract**: The application of Large Language Models (LLMs) for Automated Algorithm Discovery (AAD), particularly for optimisation heuristics, is an emerging field of research. This emergence necessitates robust, standardised benchmarking practices to rigorously evaluate the capabilities and limitations of LLM-driven AAD methods and the resulting generated algorithms, especially given the opacity of their design process and known issues with existing benchmarks. To address this need, we introduce BLADE (Benchmark suite for LLM-driven Automated Design and Evolution), a modular and extensible framework specifically designed for benchmarking LLM-driven AAD methods in a continuous black-box optimisation context. BLADE integrates collections of benchmark problems (including MA-BBOB and SBOX-COST among others) with instance generators and textual descriptions aimed at capability-focused testing, such as generalisation, specialisation and information exploitation. It offers flexible experimental setup options, standardised logging for reproducibility and fair comparison, incorporates methods for analysing the AAD process (e.g., Code Evolution Graphs and various visualisation approaches) and facilitates comparison against human-designed baselines through integration with established tools like IOHanalyser and IOHexplainer. BLADE provides an `out-of-the-box' solution to systematically evaluate LLM-driven AAD approaches. The framework is demonstrated through two distinct use cases exploring mutation prompt strategies and function specialisation. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动化算法发现（AAD）中的应用，特别是优化启发式算法，是一个新兴的研究领域。为了严谨评估LLM驱动的AAD方法及其生成算法的能力和局限性，特别是考虑到它们设计过程的不透明性和现有基准存在的问题，我们提出了BLADE（LLM驱动的自动化设计与演化基准套件），一个模块化和可扩展的框架，专门用于连续黑盒优化环境中的LLM驱动的AAD方法基准测试。BLADE集成了多种基准问题集合（包括MA-BBOB和SBOX-COST等），并提供实例生成器和旨在进行能力测试（如泛化、特化和信息利用）的文本描述。它提供了灵活的实验设置选项、标准化的日志记录以确保可再现性和公平比较，并结合了分析AAD过程的方法（如代码演化图和各种可视化方法），并通过与IOHanalyser和IOHexplainer等现有工具的集成，促进与人类设计基准的比较。BLADE提供了一种开箱即用的解决方案，系统地评估LLM驱动的AAD方法。框架通过两种不同的应用场景展示了突变提示策略和函数特化的探索。 

---
# LZ Penalty: An information-theoretic repetition penalty for autoregressive language models 

**Title (ZH)**: LZ罚分：自回归语言模型的信息论重复罚分 

**Authors**: Antonio A. Ginart, Naveen Kodali, Jason Lee, Caiming Xiong, Silvio Savarese, John R. Emmons  

**Link**: [PDF](https://arxiv.org/pdf/2504.20131)  

**Abstract**: We introduce the LZ penalty, a penalty specialized for reducing degenerate repetitions in autoregressive language models without loss of capability. The penalty is based on the codelengths in the LZ77 universal lossless compression algorithm. Through the lens of the prediction-compression duality, decoding the LZ penalty has the interpretation of sampling from the residual distribution after removing the information that is highly compressible. We demonstrate the LZ penalty enables state-of-the-art open-source reasoning models to operate with greedy (temperature zero) decoding without loss of capability and without instances of degenerate repetition. Both the industry-standard frequency penalty and repetition penalty are ineffective, incurring degenerate repetition rates of up to 4%. 

**Abstract (ZH)**: LZ惩罚：一种用于减少自回归语言模型中退化重复现象的专业惩罚方法 

---
# Towards Large Language Models for Lunar Mission Planning and In Situ Resource Utilization 

**Title (ZH)**: 面向月球任务规划与原位资源利用的大规模语言模型 

**Authors**: Michael Pekala, Gregory Canal, Samuel Barham, Milena B. Graziano, Morgan Trexler, Leslie Hamilton, Elizabeth Reilly, Christopher D. Stiles  

**Link**: [PDF](https://arxiv.org/pdf/2504.20125)  

**Abstract**: A key factor for lunar mission planning is the ability to assess the local availability of raw materials. However, many potentially relevant measurements are scattered across a variety of scientific publications. In this paper we consider the viability of obtaining lunar composition data by leveraging LLMs to rapidly process a corpus of scientific publications. While leveraging LLMs to obtain knowledge from scientific documents is not new, this particular application presents interesting challenges due to the heterogeneity of lunar samples and the nuances involved in their characterization. Accuracy and uncertainty quantification are particularly crucial since many materials properties can be sensitive to small variations in composition. Our findings indicate that off-the-shelf LLMs are generally effective at extracting data from tables commonly found in these documents. However, there remains opportunity to further refine the data we extract in this initial approach; in particular, to capture fine-grained mineralogy information and to improve performance on more subtle/complex pieces of information. 

**Abstract (ZH)**: 月球任务规划的一个关键因素是评估当地原材料的可用性。然而，许多相关的测量结果分散在各种科学出版物中。本文考虑通过利用大规模语言模型（LLMs）快速处理科学出版物集合来获取月球成分数据的可行性。尽管利用LLMs从科学文献中获取知识并不新鲜，但这种特定应用由于月球样本的异质性和其表征中的细微之处，面临着有趣的挑战。准确性与不确定性量化尤为重要，因为许多材料性质可能会对成分的细微变化敏感。我们的研究结果表明，现成的LLM通常适用于提取这些文档中常见的表格数据。然而，在初始方法中，仍有机会进一步细化提取的数据；特别是捕捉细微的矿物信息和提高对更微妙/复杂信息的表现。 

---
# Can LLMs Be Trusted for Evaluating RAG Systems? A Survey of Methods and Datasets 

**Title (ZH)**: LLM用于评估RAG系统可信吗？一种方法和数据集综述 

**Authors**: Lorenz Brehme, Thomas Ströhle, Ruth Breu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20119)  

**Abstract**: Retrieval-Augmented Generation (RAG) has advanced significantly in recent years. The complexity of RAG systems, which involve multiple components-such as indexing, retrieval, and generation-along with numerous other parameters, poses substantial challenges for systematic evaluation and quality enhancement. Previous research highlights that evaluating RAG systems is essential for documenting advancements, comparing configurations, and identifying effective approaches for domain-specific applications. This study systematically reviews 63 academic articles to provide a comprehensive overview of state-of-the-art RAG evaluation methodologies, focusing on four key areas: datasets, retrievers, indexing and databases, and the generator component. We observe the feasibility of an automated evaluation approach for each component of a RAG system, leveraging an LLM capable of both generating evaluation datasets and conducting evaluations. In addition, we found that further practical research is essential to provide companies with clear guidance on the do's and don'ts of implementing and evaluating RAG systems. By synthesizing evaluation approaches for key RAG components and emphasizing the creation and adaptation of domain-specific datasets for benchmarking, we contribute to the advancement of systematic evaluation methods and the improvement of evaluation rigor for RAG systems. Furthermore, by examining the interplay between automated approaches leveraging LLMs and human judgment, we contribute to the ongoing discourse on balancing automation and human input, clarifying their respective contributions, limitations, and challenges in achieving robust and reliable evaluations. 

**Abstract (ZH)**: 检索增强生成（RAG）在近年来取得了显著进步。复杂的RAG系统涉及多个组件——如索引、检索和生成——以及众多其他参数，这对系统的系统性评估和质量提升提出了重大挑战。以往的研究强调，评估RAG系统对于记录进步、比较配置以及识别适用于特定领域的有效方法至关重要。本研究系统地回顾了63篇学术文章，提供了关于前沿RAG评估方法的全面概述，重点关注四个关键领域：数据集、检索器、索引和数据库，以及生成器组件。我们观察到，可以利用具备生成评价数据集和进行评估能力的大语言模型（LLM）来实现每个RAG系统组件的自动化评价方法。此外，我们发现进一步的实际研究对于为企业提供实施和评估RAG系统的明确指导是必不可少的。通过综合关键RAG组件的评价方法，并强调为基准测试创建和适应领域特定数据集，我们为系统评价方法的发展和技术进步做出了贡献，并提高了RAG系统评价的严谨性。此外，通过研究利用LLM的自动化方法与人类判断之间的交互，我们为自动化与人类输入的平衡讨论做出了贡献，阐明了它们各自的贡献、局限性和挑战，以实现稳健可靠的评价。 

---
# OpenTCM: A GraphRAG-Empowered LLM-based System for Traditional Chinese Medicine Knowledge Retrieval and Diagnosis 

**Title (ZH)**: OpenTCM：一种基于GraphRAG的LLM系统，用于中医知识检索与诊断 

**Authors**: Jinglin He, Yunqi Guo, Lai Kwan Lam, Waikei Leung, Lixing He, Yuanan Jiang, Chi Chiu Wang, Guoliang Xing, Hongkai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20118)  

**Abstract**: Traditional Chinese Medicine (TCM) represents a rich repository of ancient medical knowledge that continues to play an important role in modern healthcare. Due to the complexity and breadth of the TCM literature, the integration of AI technologies is critical for its modernization and broader accessibility. However, this integration poses considerable challenges, including the interpretation of obscure classical Chinese texts and the modeling of intricate semantic relationships among TCM concepts. In this paper, we develop OpenTCM, an LLM-based system that combines a domain-specific TCM knowledge graph and Graph-based Retrieval-Augmented Generation (GraphRAG). First, we extract more than 3.73 million classical Chinese characters from 68 gynecological books in the Chinese Medical Classics Database, with the help of TCM and gynecology experts. Second, we construct a comprehensive multi-relational knowledge graph comprising more than 48,000 entities and 152,000 interrelationships, using customized prompts and Chinese-oriented LLMs such as DeepSeek and Kimi to ensure high-fidelity semantic understanding. Last, we integrate OpenTCM with this knowledge graph, enabling high-fidelity ingredient knowledge retrieval and diagnostic question-answering without model fine-tuning. Experimental evaluations demonstrate that our prompt design and model selection significantly improve knowledge graph quality, achieving a precision of 98. 55% and an F1 score of 99. 55%. In addition, OpenTCM achieves mean expert scores of 4.5 in ingredient information retrieval and 3.8 in diagnostic question-answering tasks, outperforming state-of-the-art solutions in real-world TCM use cases. 

**Abstract (ZH)**: 传统中医（TCM）代表了丰富的古代医学知识宝库，在现代医疗保健中发挥着重要作用。由于中医文献的复杂性和广泛性，集成人工智能技术对于其现代化和更广泛的可访问性至关重要。然而，这一集成也面临诸多挑战，包括解释晦涩的古典中文文本和模拟能动的中医概念之间的复杂语义关系。在本文中，我们开发了OpenTCM，这是一个基于LLM的系统，结合了领域特定的中医知识图谱和基于图的检索增强生成（GraphRAG）系统。首先，我们借助中医和妇产科专家的帮助，从中国医典数据库中的68部妇科学术著作中提取了超过373万个古典汉字。其次，我们构建了一个包含超过48,000个实体和152,000个关系的综合多关系知识图谱，使用定制提示和中文导向的LLM（如DeepSeek和Kimi）确保高保真语义理解。最后，我们将OpenTCM与该知识图谱集成，无需模型微调即可实现高保真度的药材知识检索和诊断问答。实验评估表明，我们的提示设计和模型选择显著提高了知识图谱的质量，达到了98.55%的精度和99.55%的F1分数。此外，在成分信息检索和诊断问答任务中，OpenTCM的平均专家评分为4.5和3.8，优于实际应用中现有最先进的解决方案。 

---
# ResearchCodeAgent: An LLM Multi-Agent System for Automated Codification of Research Methodologies 

**Title (ZH)**: ResearchCodeAgent: 一种用于研究方法自动化编码的LLM多智能体系统 

**Authors**: Shubham Gandhi, Dhruv Shah, Manasi Patwardhan, Lovekesh Vig, Gautam Shroff  

**Link**: [PDF](https://arxiv.org/pdf/2504.20117)  

**Abstract**: In this paper we introduce ResearchCodeAgent, a novel multi-agent system leveraging large language models (LLMs) agents to automate the codification of research methodologies described in machine learning literature. The system bridges the gap between high-level research concepts and their practical implementation, allowing researchers auto-generating code of existing research papers for benchmarking or building on top-of existing methods specified in the literature with availability of partial or complete starter code. ResearchCodeAgent employs a flexible agent architecture with a comprehensive action suite, enabling context-aware interactions with the research environment. The system incorporates a dynamic planning mechanism, utilizing both short and long-term memory to adapt its approach iteratively. We evaluate ResearchCodeAgent on three distinct machine learning tasks with distinct task complexity and representing different parts of the ML pipeline: data augmentation, optimization, and data batching. Our results demonstrate the system's effectiveness and generalizability, with 46.9% of generated code being high-quality and error-free, and 25% showing performance improvements over baseline implementations. Empirical analysis shows an average reduction of 57.9% in coding time compared to manual implementation. We observe higher gains for more complex tasks. ResearchCodeAgent represents a significant step towards automating the research implementation process, potentially accelerating the pace of machine learning research. 

**Abstract (ZH)**: ResearchCodeAgent：利用大型语言模型代理自动化机器学习研究方法的编码 

---
# AutoP2C: An LLM-Based Agent Framework for Code Repository Generation from Multimodal Content in Academic Papers 

**Title (ZH)**: AutoP2C：一种基于LLM的代理框架，用于从学术论文多模态内容生成代码仓库 

**Authors**: Zijie Lin, Yiqing Shen, Qilin Cai, He Sun, Jinrui Zhou, Mingjun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.20115)  

**Abstract**: Machine Learning (ML) research is spread through academic papers featuring rich multimodal content, including text, diagrams, and tabular results. However, translating these multimodal elements into executable code remains a challenging and time-consuming process that requires substantial ML expertise. We introduce ``Paper-to-Code'' (P2C), a novel task that transforms the multimodal content of scientific publications into fully executable code repositories, which extends beyond the existing formulation of code generation that merely converts textual descriptions into isolated code snippets. To automate the P2C process, we propose AutoP2C, a multi-agent framework based on large language models that processes both textual and visual content from research papers to generate complete code repositories. Specifically, AutoP2C contains four stages: (1) repository blueprint extraction from established codebases, (2) multimodal content parsing that integrates information from text, equations, and figures, (3) hierarchical task decomposition for structured code generation, and (4) iterative feedback-driven debugging to ensure functionality and performance. Evaluation on a benchmark of eight research papers demonstrates the effectiveness of AutoP2C, which can successfully generate executable code repositories for all eight papers, while OpenAI-o1 or DeepSeek-R1 can only produce runnable code for one paper. The code is available at this https URL. 

**Abstract (ZH)**: 将学术论文中的富模态内容转换为可执行代码（Paper-to-Code）：AutoP2C多agent框架的研究 

---
# Adaptive Helpfulness-Harmlessness Alignment with Preference Vectors 

**Title (ZH)**: 自适应有益无害对齐与偏好向量 

**Authors**: Ren-Wei Liang, Chin-Ting Hsu, Chan-Hung Yu, Saransh Agrawal, Shih-Cheng Huang, Shang-Tse Chen, Kuan-Hao Huang, Shao-Hua Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.20106)  

**Abstract**: Ensuring that large language models (LLMs) are both helpful and harmless is a critical challenge, as overly strict constraints can lead to excessive refusals, while permissive models risk generating harmful content. Existing approaches, such as reinforcement learning from human feedback (RLHF) and direct preference optimization (DPO), attempt to balance these trade-offs but suffer from performance conflicts, limited controllability, and poor extendability. To address these issues, we propose Preference Vector, a novel framework inspired by task arithmetic. Instead of optimizing multiple preferences within a single objective, we train separate models on individual preferences, extract behavior shifts as preference vectors, and dynamically merge them at test time. This modular approach enables fine-grained, user-controllable preference adjustments and facilitates seamless integration of new preferences without retraining. Experiments show that our proposed Preference Vector framework improves helpfulness without excessive conservatism, allows smooth control over preference trade-offs, and supports scalable multi-preference alignment. 

**Abstract (ZH)**: 确保大型语言模型（LLMs）既有益又无害是一项关键挑战，过于严格的约束可能导致过度拒绝，而宽容的模型则存在生成有害内容的风险。现有的方法，如基于人类反馈的强化学习（RLHF）和直接偏好优化（DPO），试图在这两者之间寻求平衡，但会遇到性能冲突、有限可控性和扩展性差的问题。为了解决这些问题，我们提出了一种名为偏好向量的新型框架，该框架受到任务算术的启发。我们不是在一个单一的目标中优化多个偏好，而是分别在个体偏好上训练独立的模型，提取行为变化作为偏好向量，并在测试时动态合并它们。这种模块化方法允许精细的、用户可控的偏好调整，并能无缝地集成新的偏好而无需重新训练。实验结果显示，我们提出的偏好向量框架能够提高有益性而不显过分保守，允许平滑地控制偏好权衡，并支持可扩展的多偏好对齐。 

---
# GenTorrent: Scaling Large Language Model Serving with An Overley Network 

**Title (ZH)**: GenTorrent: 通过Overlay网络扩展大型语言模型服务 

**Authors**: Fei Fang, Yifan Hua, Shengze Wang, Ruilin Zhou, Yi Liu, Chen Qian, Xiaoxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20101)  

**Abstract**: While significant progress has been made in research and development on open-source and cost-efficient large-language models (LLMs), serving scalability remains a critical challenge, particularly for small organizations and individuals seeking to deploy and test their LLM innovations. Inspired by peer-to-peer networks that leverage decentralized overlay nodes to increase throughput and availability, we propose GenTorrent, an LLM serving overlay that harnesses computing resources from decentralized contributors. We identify four key research problems inherent to enabling such a decentralized infrastructure: 1) overlay network organization; 2) LLM communication privacy; 3) overlay forwarding for resource efficiency; and 4) verification of serving quality. This work presents the first systematic study of these fundamental problems in the context of decentralized LLM serving. Evaluation results from a prototype implemented on a set of decentralized nodes demonstrate that GenTorrent achieves a latency reduction of over 50% compared to the baseline design without overlay forwarding. Furthermore, the security features introduce minimal overhead to serving latency and throughput. We believe this work pioneers a new direction for democratizing and scaling future AI serving capabilities. 

**Abstract (ZH)**: 开源和低成本大型语言模型（LLMs）的研究与开发取得了显著进展，但服务扩展性仍然是一个关键挑战，特别是在小型组织和个人希望部署和测试其LLM创新方面。受去中心化-overlay节点提高吞吐量和可获取性的点对点网络启发，我们提出GenTorrent，一种利用分散贡献者计算资源的LLM服务overlay。我们识别出四个内在的关键研究问题，以支持这种去中心化基础设施：1）overlay网络组织；2）LLM通信隐私；3）overlay转发以提高资源效率；4）提供服务质量的验证。本研究首次在去中心化LLM服务背景下系统地探讨了这些基础性问题。在一组去中心化节点上实现的原型评估结果表明，与不使用overlay转发的基础设计相比，GenTorrent实现了超过50%的延迟减少。此外，安全功能对服务延迟和吞吐量的影响最小。我们认为，本项工作开创了普及和扩展未来AI服务能力的新方向。 

---
# A model and package for German ColBERT 

**Title (ZH)**: German ColBERT模型与包 

**Authors**: Thuong Dang, Qiqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20083)  

**Abstract**: In this work, we introduce a German version for ColBERT, a late interaction multi-dense vector retrieval method, with a focus on RAG applications. We also present the main features of our package for ColBERT models, supporting both retrieval and fine-tuning workflows. 

**Abstract (ZH)**: 在本工作中，我们引入了ColBERT的一种德语版本，这是一种晚交互多密集向量检索方法，重点关注RAG应用。我们还介绍了我们的ColBERT模型包的主要功能，支持检索和微调工作流程。 

---
# RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning 

**Title (ZH)**: RAGEN：通过多轮强化学习理解LLM代理的自我进化 

**Authors**: Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Monica Lam, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, Manling Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.20073)  

**Abstract**: Training large language models (LLMs) as interactive agents presents unique challenges including long-horizon decision making and interacting with stochastic environment feedback. While reinforcement learning (RL) has enabled progress in static tasks, multi-turn agent RL training remains underexplored. We propose StarPO (State-Thinking-Actions-Reward Policy Optimization), a general framework for trajectory-level agent RL, and introduce RAGEN, a modular system for training and evaluating LLM agents. Our study on three stylized environments reveals three core findings. First, our agent RL training shows a recurring mode of Echo Trap where reward variance cliffs and gradient spikes; we address this with StarPO-S, a stabilized variant with trajectory filtering, critic incorporation, and decoupled clipping. Second, we find the shaping of RL rollouts would benefit from diverse initial states, medium interaction granularity and more frequent sampling. Third, we show that without fine-grained, reasoning-aware reward signals, agent reasoning hardly emerge through multi-turn RL and they may show shallow strategies or hallucinated thoughts. Code and environments are available at this https URL. 

**Abstract (ZH)**: 训练大规模语言模型（LLMs）作为交互代理面临独特的挑战，包括长期决策制定和与随机环境反馈的交互。虽然强化学习（RL）已在静态任务上取得了进展，但多轮交互代理的RL训练仍处于探索阶段。我们提出了一种适用于轨迹级代理RL的通用框架StarPO（State-Thinking-Actions-Reward Policy Optimization），并介绍了一种模块化系统RAGEN，用于训练和评估LLM代理。我们的三项定制环境研究揭示了三个核心发现。首先，我们的代理RL训练表现出Echo Trap模式，即奖励方差悬崖和梯度尖峰；我们通过引入StarPO-S，一种包含轨迹过滤、批评家集成和解耦修剪的稳定版本来应对这一问题。其次，我们发现强化学习（RL）滚动生成的质量可以从多样化的初始状态、中等交互粒度和更频繁的采样中获益。第三，我们证明，如果没有细粒度的、具有推理意识的奖励信号，代理的推理可能难以通过多轮RL显现，它们可能会表现出肤浅的战略或虚幻的思维。代码和环境可在此链接访问。 

---
# Recommending Clinical Trials for Online Patient Cases using Artificial Intelligence 

**Title (ZH)**: 使用人工智能推荐在线患者病例的临床试验 

**Authors**: Joey Chan, Qiao Jin, Nicholas Wan, Charalampos S. Floudas, Elisabetta Xue, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20059)  

**Abstract**: Clinical trials are crucial for assessing new treatments; however, recruitment challenges - such as limited awareness, complex eligibility criteria, and referral barriers - hinder their success. With the growth of online platforms, patients increasingly turn to social media and health communities for support, research, and advocacy, expanding recruitment pools and established enrollment pathways. Recognizing this potential, we utilized TrialGPT, a framework that leverages a large language model (LLM) as its backbone, to match 50 online patient cases (collected from published case reports and a social media website) to clinical trials and evaluate performance against traditional keyword-based searches. Our results show that TrialGPT outperforms traditional methods by 46% in identifying eligible trials, with each patient, on average, being eligible for around 7 trials. Additionally, our outreach efforts to case authors and trial organizers regarding these patient-trial matches yielded highly positive feedback, which we present from both perspectives. 

**Abstract (ZH)**: 临床试验对于评估新治疗方法至关重要，但有限的意识、复杂的入组标准和转诊障碍等挑战阻碍了其成功。随着在线平台的兴起，患者越来越多地通过社交媒体和健康社区寻求支持、进行研究和倡导，从而扩大了招募人群并建立了现有的入组途径。认识到这一潜力，我们利用了以大型语言模型（LLM）为基础的TrialGPT框架，将50个在线患者案例（来自已发表的案例报告和社交媒体网站）与临床试验匹配，并评估其性能，结果表明TrialGPT在识别符合条件的临床试验方面比传统的关键词搜索方法高出46%，每位患者平均符合条件的临床试验约为7项。此外，我们对案例作者和临床试验组织者关于这些患者-临床试验匹配的接触工作也获得了高度积极的反馈，我们从两个角度进行了呈现。 

---
