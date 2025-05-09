# Conversational Process Model Redesign 

**Title (ZH)**: 对话过程模型重设计 

**Authors**: Nataliia Klievtsova, Timotheus Kampik, Juergen Mangler, Stefanie Rinderle-Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.05453)  

**Abstract**: With the recent success of large language models (LLMs), the idea of AI-augmented Business Process Management systems is becoming more feasible. One of their essential characteristics is the ability to be conversationally actionable, allowing humans to interact with the LLM effectively to perform crucial process life cycle tasks such as process model design and redesign. However, most current research focuses on single-prompt execution and evaluation of results, rather than on continuous interaction between the user and the LLM. In this work, we aim to explore the feasibility of using LLMs to empower domain experts in the creation and redesign of process models in an iterative and effective way. The proposed conversational process model redesign (CPD) approach receives as input a process model and a redesign request by the user in natural language. Instead of just letting the LLM make changes, the LLM is employed to (a) identify process change patterns from literature, (b) re-phrase the change request to be aligned with an expected wording for the identified pattern (i.e., the meaning), and then to (c) apply the meaning of the change to the process model. This multi-step approach allows for explainable and reproducible changes. In order to ensure the feasibility of the CPD approach, and to find out how well the patterns from literature can be handled by the LLM, we performed an extensive evaluation. The results show that some patterns are hard to understand by LLMs and by users. Within the scope of the study, we demonstrated that users need support to describe the changes clearly. Overall the evaluation shows that the LLMs can handle most changes well according to a set of completeness and correctness criteria. 

**Abstract (ZH)**: 基于大型语言模型的迭代过程模型重设计方法探究 

---
# EcoAgent: An Efficient Edge-Cloud Collaborative Multi-Agent Framework for Mobile Automation 

**Title (ZH)**: EcoAgent：一种高效的边缘-云协作多Agent框架 Mobile Automation 

**Authors**: Biao Yi, Xavier Hu, Yurun Chen, Shengyu Zhang, Hongxia Yang, Fan Wu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05440)  

**Abstract**: Cloud-based mobile agents powered by (multimodal) large language models ((M)LLMs) offer strong reasoning abilities but suffer from high latency and cost. While fine-tuned (M)SLMs enable edge deployment, they often lose general capabilities and struggle with complex tasks. To address this, we propose EcoAgent, an Edge-Cloud cOllaborative multi-agent framework for mobile automation. EcoAgent features a closed-loop collaboration among a cloud-based Planning Agent and two edge-based agents: the Execution Agent for action execution and the Observation Agent for verifying outcomes. The Observation Agent uses a Pre-Understanding Module to compress screen images into concise text, reducing token usage. In case of failure, the Planning Agent retrieves screen history and replans via a Reflection Module. Experiments on AndroidWorld show that EcoAgent maintains high task success rates while significantly reducing MLLM token consumption, enabling efficient and practical mobile automation. 

**Abstract (ZH)**: 基于云的由（多模态）大型语言模型驱动的移动代理（(M)LLMs）提供了强大的推理能力，但存在高延迟和成本问题。虽然微调的（M）SLMs能够实现边缘部署，但它们往往失去了一般能力，并且难以处理复杂任务。为了解决这个问题，我们提出了一种Edge-Cloud协作多代理框架EcoAgent，用于移动自动化。EcoAgent特征是云基规划代理和两个边缘基代理之间形成闭环合作：执行代理用于执行动作，观察代理用于验证结果。观察代理使用前理解模块将屏幕图像压缩成简洁的文本，减少令牌使用量。在失败情况下，规划代理通过反思模块检索屏幕历史并重新规划。在AndroidWorld的实验显示，EcoAgent能够在大幅减少MLLM令牌消耗的同时保持高任务成功率，实现高效和实用的移动自动化。 

---
# A Pain Assessment Framework based on multimodal data and Deep Machine Learning methods 

**Title (ZH)**: 基于多模态数据和深度机器学习的疼痛评估框架 

**Authors**: Stefanos Gkikas  

**Link**: [PDF](https://arxiv.org/pdf/2505.05396)  

**Abstract**: From the original abstract:
This thesis initially aims to study the pain assessment process from a clinical-theoretical perspective while exploring and examining existing automatic approaches. Building on this foundation, the primary objective of this Ph.D. project is to develop innovative computational methods for automatic pain assessment that achieve high performance and are applicable in real clinical settings. A primary goal is to thoroughly investigate and assess significant factors, including demographic elements that impact pain perception, as recognized in pain research, through a computational standpoint. Within the limits of the available data in this research area, our goal was to design, develop, propose, and offer automatic pain assessment pipelines for unimodal and multimodal configurations that are applicable to the specific requirements of different scenarios. The studies published in this Ph.D. thesis showcased the effectiveness of the proposed methods, achieving state-of-the-art results. Additionally, they paved the way for exploring new approaches in artificial intelligence, foundation models, and generative artificial intelligence. 

**Abstract (ZH)**: 本论文最初旨在从临床理论视角研究疼痛评估过程，并探索和检验现有的自动评估方法。在此基础上，本博士项目的主要目标是开发高性能且适用于实际临床环境的创新计算方法进行自动疼痛评估。主要目标之一是从计算角度全面研究和评估影响疼痛感知的关键因素，包括疼痛研究中认可的人口统计学因素。在这一研究领域的数据限制下，我们旨在设计、开发、提出并提供适用于不同场景特定需求的一模性和多模性自动疼痛评估管道。本博士论文中发表的研究展示了所提出方法的有效性，达到了最先进的成果，并为探索人工智能、基础模型和生成式人工智能的新方法奠定了基础。 

---
# Advancing Neural Network Verification through Hierarchical Safety Abstract Interpretation 

**Title (ZH)**: 通过分层安全性抽象解释推进神经网络验证 

**Authors**: Luca Marzari, Isabella Mastroeni, Alessandro Farinelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.05235)  

**Abstract**: Traditional methods for formal verification (FV) of deep neural networks (DNNs) are constrained by a binary encoding of safety properties, where a model is classified as either safe or unsafe (robust or not robust). This binary encoding fails to capture the nuanced safety levels within a model, often resulting in either overly restrictive or too permissive requirements. In this paper, we introduce a novel problem formulation called Abstract DNN-Verification, which verifies a hierarchical structure of unsafe outputs, providing a more granular analysis of the safety aspect for a given DNN. Crucially, by leveraging abstract interpretation and reasoning about output reachable sets, our approach enables assessing multiple safety levels during the FV process, requiring the same (in the worst case) or even potentially less computational effort than the traditional binary verification approach. Specifically, we demonstrate how this formulation allows rank adversarial inputs according to their abstract safety level violation, offering a more detailed evaluation of the model's safety and robustness. Our contributions include a theoretical exploration of the relationship between our novel abstract safety formulation and existing approaches that employ abstract interpretation for robustness verification, complexity analysis of the novel problem introduced, and an empirical evaluation considering both a complex deep reinforcement learning task (based on Habitat 3.0) and standard DNN-Verification benchmarks. 

**Abstract (ZH)**: 传统方法用于深度神经网络形式验证的局限性及其克服：一种新的抽象DNN验证方法 

---
# ChemRxivQuest: A Curated Chemistry Question-Answer Database Extracted from ChemRxiv Preprints 

**Title (ZH)**: ChemRxivQuest: 一个精选的化学问答数据库，提取自ChemRxiv预印本 

**Authors**: Mahmoud Amiri, Thomas Bocklitz  

**Link**: [PDF](https://arxiv.org/pdf/2505.05232)  

**Abstract**: The rapid expansion of chemistry literature poses significant challenges for researchers seeking to efficiently access domain-specific knowledge. To support advancements in chemistry-focused natural language processing (NLP), we present ChemRxivQuest, a curated dataset of 970 high-quality question-answer (QA) pairs derived from 155 ChemRxiv preprints across 17 subfields of chemistry. Each QA pair is explicitly linked to its source text segment to ensure traceability and contextual accuracy. ChemRxivQuest was constructed using an automated pipeline that combines optical character recognition (OCR), GPT-4o-based QA generation, and a fuzzy matching technique for answer verification. The dataset emphasizes conceptual, mechanistic, applied, and experimental questions, enabling applications in retrieval-based QA systems, search engine development, and fine-tuning of domain-adapted large language models. We analyze the dataset's structure, coverage, and limitations, and outline future directions for expansion and expert validation. ChemRxivQuest provides a foundational resource for chemistry NLP research, education, and tool development. 

**Abstract (ZH)**: 化学文献的迅速扩张给研究人员高效获取领域特定知识带来了重大挑战。为了支持化学方向的自然语言处理(NLP)进展，我们呈现了ChemRxivQuest数据集，该数据集包含了来自155篇ChemRxiv预印本的970个高质量问答(QA)对，覆盖了17个化学子领域。每个QA对都与原始文本段落明确链接，以确保可追溯性和语境准确性。ChemRxivQuest是通过结合光学字符识别(OCR)、基于GPT-4o的问答生成以及模糊匹配技术进行答案验证的自动化管道构建而成的。该数据集强调概念性、机制性、应用性和实验性问题，能够应用于检索型问答系统、搜索引擎开发和领域适配大语言模型的微调。我们分析了数据集的结构、覆盖面和限制，并概述了扩展和专家验证的未来方向。ChemRxivQuest为化学NLP研究、教育和工具开发提供了基础资源。 

---
# Societal and technological progress as sewing an ever-growing, ever-changing, patchy, and polychrome quilt 

**Title (ZH)**: 社会和技术的进步犹如缝制一幅不断扩展、变化莫测、图案繁多且色彩斑斓的Patchworktapestry。 

**Authors**: Joel Z. Leibo, Alexander Sasha Vezhnevets, William A. Cunningham, Sébastien Krier, Manfred Diaz, Simon Osindero  

**Link**: [PDF](https://arxiv.org/pdf/2505.05197)  

**Abstract**: Artificial Intelligence (AI) systems are increasingly placed in positions where their decisions have real consequences, e.g., moderating online spaces, conducting research, and advising on policy. Ensuring they operate in a safe and ethically acceptable fashion is thus critical. However, most solutions have been a form of one-size-fits-all "alignment". We are worried that such systems, which overlook enduring moral diversity, will spark resistance, erode trust, and destabilize our institutions. This paper traces the underlying problem to an often-unstated Axiom of Rational Convergence: the idea that under ideal conditions, rational agents will converge in the limit of conversation on a single ethics. Treating that premise as both optional and doubtful, we propose what we call the appropriateness framework: an alternative approach grounded in conflict theory, cultural evolution, multi-agent systems, and institutional economics. The appropriateness framework treats persistent disagreement as the normal case and designs for it by applying four principles: (1) contextual grounding, (2) community customization, (3) continual adaptation, and (4) polycentric governance. We argue here that adopting these design principles is a good way to shift the main alignment metaphor from moral unification to a more productive metaphor of conflict management, and that taking this step is both desirable and urgent. 

**Abstract (ZH)**: 人工智能系统在决策具有实际后果的情况下，确保其安全和伦理接受性至关重要：超越理性能合的框定 

---
# MARK: Memory Augmented Refinement of Knowledge 

**Title (ZH)**: MARK：基于记忆的知識精煉增強 

**Authors**: Anish Ganguli, Prabal Deb, Debleena Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.05177)  

**Abstract**: Large Language Models (LLMs) assist in specialized tasks but struggle to align with evolving domain knowledge without costly fine-tuning. Domain knowledge consists of: Knowledge: Immutable facts (e.g., 'A stone is solid') and generally accepted principles (e.g., ethical standards); Refined Memory: Evolving insights shaped by business needs and real-world changes. However, a significant gap often exists between a domain expert's deep, nuanced understanding and the system's domain knowledge, which can hinder accurate information retrieval and application. Our Memory-Augmented Refinement of Knowledge (MARK) framework enables LLMs to continuously learn without retraining by leveraging structured refined memory, inspired by the Society of Mind. MARK operates through specialized agents, each serving a distinct role: Residual Refined Memory Agent: Stores and retrieves domain-specific insights to maintain context over time; User Question Refined Memory Agent: Captures user-provided facts, abbreviations, and terminology for better comprehension; LLM Response Refined Memory Agent: Extracts key elements from responses for refinement and personalization. These agents analyse stored refined memory, detect patterns, resolve contradictions, and improve response accuracy. Temporal factors like recency and frequency prioritize relevant information while discarding outdated insights. MARK enhances LLMs in multiple ways: Ground Truth Strategy: Reduces hallucinations by establishing a structured reference; Domain-Specific Adaptation: Essential for fields like healthcare, law, and manufacturing, where proprietary insights are absent from public datasets; Personalized AI Assistants: Improves virtual assistants by remembering user preferences, ensuring coherent responses over time. 

**Abstract (ZH)**: Large Language Models的精炼知识增强框架：持续学习而不重新训练 

---
# Is there a half-life for the success rates of AI agents? 

**Title (ZH)**: AI代理的成功率是否有半衰期？ 

**Authors**: Toby Ord  

**Link**: [PDF](https://arxiv.org/pdf/2505.05115)  

**Abstract**: Building on the recent empirical work of Kwa et al. (2025), I show that within their suite of research-engineering tasks the performance of AI agents on longer-duration tasks can be explained by an extremely simple mathematical model -- a constant rate of failing during each minute a human would take to do the task. This implies an exponentially declining success rate with the length of the task and that each agent could be characterised by its own half-life. This empirical regularity allows us to estimate the success rate for an agent at different task lengths. And the fact that this model is a good fit for the data is suggestive of the underlying causes of failure on longer tasks -- that they involve increasingly large sets of subtasks where failing any one fails the task. Whether this model applies more generally on other suites of tasks is unknown and an important subject for further work. 

**Abstract (ZH)**: 基于Kwa等人（2025）的近期实证研究，本文展示了在其研究-工程任务套件中，AI代理在长时间任务上的性能可以用一个极其简单的数学模型来解释——即在人类完成任务所需的时间内的每分钟都会以恒定的失败率失败。这暗示着随着任务长度的增加，成功率呈指数下降，并且每个代理都可以通过其半衰期来表征。这一经验规律允许我们估算不同任务长度下代理的成功率。并且，这一模型能够很好地拟合数据这一事实，表明长时间任务失败的根本原因可能在于涉及越来越多的子任务，只要任何一个子任务失败，整个任务就失败。目前尚不清楚该模型是否适用于其他任务套件，这是一项重要且有待进一步研究的主题。 

---
# Multi-agent Embodied AI: Advances and Future Directions 

**Title (ZH)**: 多智能体具身AI：进展与未来方向 

**Authors**: Zhaohan Feng, Ruiqi Xue, Lei Yuan, Yang Yu, Ning Ding, Meiqin Liu, Bingzhao Gao, Jian Sun, Gang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05108)  

**Abstract**: Embodied artificial intelligence (Embodied AI) plays a pivotal role in the application of advanced technologies in the intelligent era, where AI systems are integrated with physical bodies that enable them to perceive, reason, and interact with their environments. Through the use of sensors for input and actuators for action, these systems can learn and adapt based on real-world feedback, allowing them to perform tasks effectively in dynamic and unpredictable environments. As techniques such as deep learning (DL), reinforcement learning (RL), and large language models (LLMs) mature, embodied AI has become a leading field in both academia and industry, with applications spanning robotics, healthcare, transportation, and manufacturing. However, most research has focused on single-agent systems that often assume static, closed environments, whereas real-world embodied AI must navigate far more complex scenarios. In such settings, agents must not only interact with their surroundings but also collaborate with other agents, necessitating sophisticated mechanisms for adaptation, real-time learning, and collaborative problem-solving. Despite increasing interest in multi-agent systems, existing research remains narrow in scope, often relying on simplified models that fail to capture the full complexity of dynamic, open environments for multi-agent embodied AI. Moreover, no comprehensive survey has systematically reviewed the advancements in this area. As embodied AI rapidly evolves, it is crucial to deepen our understanding of multi-agent embodied AI to address the challenges presented by real-world applications. To fill this gap and foster further development in the field, this paper reviews the current state of research, analyzes key contributions, and identifies challenges and future directions, providing insights to guide innovation and progress in this field. 

**Abstract (ZH)**: 具身人工智能（具身AI）在智能时代的先进技术应用中发挥着关键作用，其中AI系统结合了物理身体，使其能够感知、推理和与环境互动。通过使用传感器获取输入和执行器执行动作，这些系统可以根据实际反馈进行学习和适应，从而在动态和不可预测的环境中有效执行任务。随着深度学习（DL）、强化学习（RL）和大型语言模型（LLMs）等技术的成熟，具身AI已成为学术界和工业界的核心领域，其应用遍及机器人技术、医疗保健、交通运输和制造业。然而，大多数研究集中在单智能体系统上，通常假定静态的封闭环境，而现实中的具身AI必须导航更为复杂的场景。在这种环境中，智能体不仅需要与其环境互动，还需要与其他智能体协作，这需要复杂的适应机制、实时学习和协作问题解决机制。尽管对多智能体系统越来越感兴趣，但现有研究仍然局限于狭隘的范围，往往依赖于简化的模型，无法捕捉多智能体具身AI在动态开放环境中的全部复杂性。此外，尚未有全面的综述系统地回顾了这一领域的进展。随着具身AI的迅速发展，深化对多智能体具身AI的理解以应对实际应用挑战变得至关重要。为填补这一空白并促进该领域的进一步发展，本文回顾了当前的研究状态，分析了关键贡献，指出了面临的挑战和未来方向，提供了指导该领域创新和进步的见解。 

---
# A Neuro-Symbolic Framework for Sequence Classification with Relational and Temporal Knowledge 

**Title (ZH)**: 一种基于神经符号框架的序列分类方法，融合关系性和时间性知识 

**Authors**: Luca Salvatore Lorello, Marco Lippi, Stefano Melacci  

**Link**: [PDF](https://arxiv.org/pdf/2505.05106)  

**Abstract**: One of the goals of neuro-symbolic artificial intelligence is to exploit background knowledge to improve the performance of learning tasks. However, most of the existing frameworks focus on the simplified scenario where knowledge does not change over time and does not cover the temporal dimension. In this work we consider the much more challenging problem of knowledge-driven sequence classification where different portions of knowledge must be employed at different timesteps, and temporal relations are available. Our experimental evaluation compares multi-stage neuro-symbolic and neural-only architectures, and it is conducted on a newly-introduced benchmarking framework. Results demonstrate the challenging nature of this novel setting, and also highlight under-explored shortcomings of neuro-symbolic methods, representing a precious reference for future research. 

**Abstract (ZH)**: 神经符号人工智能的一个目标是利用背景知识来改进学习任务的性能。然而，现有的大多数框架集中在知识不随时间变化的简化场景上，并未涵盖时间维度。在这项工作中，我们考虑了更为挑战的知识驱动序列分类问题，在该问题中，必须在不同的时间步骤使用不同部分的知识，并且时间关系可用。我们的实验评估比较了多阶段神经符号和纯粹神经架构，并在新引入的基准框架上进行。结果显示了这一新颖设置的挑战性质，并强调了神经符号方法未充分利用的不足之处，为未来研究提供了宝贵的参考。 

---
# Enhancing Reinforcement Learning for the Floorplanning of Analog ICs with Beam Search 

**Title (ZH)**: 使用_beam_搜索增强模拟IC版图规划的强化学习 

**Authors**: Sandro Junior Della Rovere, Davide Basso, Luca Bortolussi, Mirjana Videnovic-Misic, Husni Habal  

**Link**: [PDF](https://arxiv.org/pdf/2505.05059)  

**Abstract**: The layout of analog ICs requires making complex trade-offs, while addressing device physics and variability of the circuits. This makes full automation with learning-based solutions hard to achieve. However, reinforcement learning (RL) has recently reached significant results, particularly in solving the floorplanning problem. This paper presents a hybrid method that combines RL with a beam (BS) strategy. The BS algorithm enhances the agent's inference process, allowing for the generation of flexible floorplans by accomodating various objective weightings, and addressing congestion without without the need for policy retraining or fine-tuning. Moreover, the RL agent's generalization ability stays intact, along with its efficient handling of circuit features and constraints. Experimental results show approx. 5-85% improvement in area, dead space and half-perimeter wire length compared to a standard RL application, along with higher rewards for the agent. Moreover, performance and efficiency align closely with those of existing state-of-the-art techniques. 

**Abstract (ZH)**: 基于强化学习与束搜索策略的混合方法在模拟IC布局中的应用 

---
# A Reputation System for Large Language Model-based Multi-agent Systems to Avoid the Tragedy of the Commons 

**Title (ZH)**: 基于大型语言模型的多智能体系统中的声誉系统以避免公地悲剧 

**Authors**: Siyue Ren, Wanli Fu, Xinkun Zou, Chen Shen, Yi Cai, Chen Chu, Zhen Wang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05029)  

**Abstract**: The tragedy of the commons, where individual self-interest leads to collectively disastrous outcomes, is a pervasive challenge in human society. Recent studies have demonstrated that similar phenomena can arise in generative multi-agent systems (MASs). To address this challenge, this paper explores the use of reputation systems as a remedy. We propose RepuNet, a dynamic, dual-level reputation framework that models both agent-level reputation dynamics and system-level network evolution. Specifically, driven by direct interactions and indirect gossip, agents form reputations for both themselves and their peers, and decide whether to connect or disconnect other agents for future interactions. Through two distinct scenarios, we show that RepuNet effectively mitigates the 'tragedy of the commons', promoting and sustaining cooperation in generative MASs. Moreover, we find that reputation systems can give rise to rich emergent behaviors in generative MASs, such as the formation of cooperative clusters, the social isolation of exploitative agents, and the preference for sharing positive gossip rather than negative ones. 

**Abstract (ZH)**: 公地悲剧：个体自我利益导致集体灾难性后果的现象在人类社会中普遍存在，近期研究表明此类现象也可能出现在生成性多智能体系统中。为应对这一挑战，本文探讨了声誉系统作为一种解决方案的应用。我们提出了RepuNet，一种动态的双层声誉框架，用于建模智能体层面的声誉动态和系统层面的网络演化。通过两种不同的场景，我们展示了RepuNet如何有效地缓解“公地悲剧”，促进和维持生成性多智能体系统中的合作。此外，我们发现声誉系统可以在生成性多智能体系统中引发丰富的新涌现行为，如合作集群的形成、剥削性智能体的社会孤立以及更倾向于传播正面消息而非负面消息。 

---
# Foam-Agent: Towards Automated Intelligent CFD Workflows 

**Title (ZH)**: 泡沫代理：迈向自动化智能CFD工作流 

**Authors**: Ling Yue, Nithin Somasekharan, Yadi Cao, Shaowu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04997)  

**Abstract**: Computational Fluid Dynamics (CFD) is an essential simulation tool in various engineering disciplines, but it often requires substantial domain expertise and manual configuration, creating barriers to entry. We present Foam-Agent, a multi-agent framework that automates complex OpenFOAM-based CFD simulation workflows from natural language inputs. Our innovation includes (1) a hierarchical multi-index retrieval system with specialized indices for different simulation aspects, (2) a dependency-aware file generation system that provides consistency management across configuration files, and (3) an iterative error correction mechanism that diagnoses and resolves simulation failures without human intervention. Through comprehensive evaluation on the dataset of 110 simulation tasks, Foam-Agent achieves an 83.6% success rate with Claude 3.5 Sonnet, significantly outperforming existing frameworks (55.5% for MetaOpenFOAM and 37.3% for OpenFOAM-GPT). Ablation studies demonstrate the critical contribution of each system component, with the specialized error correction mechanism providing a 36.4% performance improvement. Foam-Agent substantially lowers the CFD expertise threshold while maintaining modeling accuracy, demonstrating the potential of specialized multi-agent systems to democratize access to complex scientific simulation tools. The code is public at this https URL 

**Abstract (ZH)**: 基于多agents的自然语言驱动的OpenFOAM复杂CFD仿真工作流自动化框架 

---
# Position: The AI Conference Peer Review Crisis Demands Author Feedback and Reviewer Rewards 

**Title (ZH)**: 位置：人工智能会议同行评审危机亟需作者反馈和评审员奖励 

**Authors**: Jaeho Kim, Yunseok Lee, Seulki Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.04966)  

**Abstract**: The peer review process in major artificial intelligence (AI) conferences faces unprecedented challenges with the surge of paper submissions (exceeding 10,000 submissions per venue), accompanied by growing concerns over review quality and reviewer responsibility. This position paper argues for the need to transform the traditional one-way review system into a bi-directional feedback loop where authors evaluate review quality and reviewers earn formal accreditation, creating an accountability framework that promotes a sustainable, high-quality peer review system. The current review system can be viewed as an interaction between three parties: the authors, reviewers, and system (i.e., conference), where we posit that all three parties share responsibility for the current problems. However, issues with authors can only be addressed through policy enforcement and detection tools, and ethical concerns can only be corrected through self-reflection. As such, this paper focuses on reforming reviewer accountability with systematic rewards through two key mechanisms: (1) a two-stage bi-directional review system that allows authors to evaluate reviews while minimizing retaliatory behavior, (2)a systematic reviewer reward system that incentivizes quality reviewing. We ask for the community's strong interest in these problems and the reforms that are needed to enhance the peer review process. 

**Abstract (ZH)**: 重大人工智能会议的peer review过程面临着前所未有的挑战：随着投稿量激增（每个会场超过10,000篇投稿）和对评审质量及评审员责任的日益关注，需要将传统的单向评审系统转变为双向反馈循环，其中作者评估评审质量，评审员获得正式认证，从而建立一个促进可持续高质量评审系统的问责框架。当前的评审系统可视为作者、评审员和系统（即会议）之间的交互，我们提出所有三方均需对当前问题负责。然而，作者方面的问题只能通过政策执行和检测工具来解决，伦理问题只能通过自我反省来纠正。因此，本文着重从系统性奖励机制改革评审员问责制，主要包括：（1）两阶段双向评审系统，允许作者评估评审而不助长报复行为；（2）系统性评审员奖励系统，激励高质量评审。我们呼吁社区对这些问题及其所需的改革给予强烈关注，以提升同行评审过程的质量。 

---
# Position: Epistemic Artificial Intelligence is Essential for Machine Learning Models to Know When They Do Not Know 

**Title (ZH)**: 位置：知识型人工智能对于机器学习模型在不知情时至关重要 

**Authors**: Shireen Kudukkil Manchingal, Fabio Cuzzolin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04950)  

**Abstract**: Despite the impressive achievements of AI, including advancements in generative models and large language models, there remains a significant gap in the ability of AI to handle uncertainty and generalize beyond the training data. We argue that AI models, especially in autonomous systems, fail to make robust predictions when faced with unfamiliar or adversarial data, as evidenced by incidents with autonomous vehicles. Traditional machine learning approaches struggle to address these issues due to an overemphasis on data fitting and domain adaptation. This position paper posits a paradigm shift towards epistemic artificial intelligence, emphasizing the need for models to learn not only from what they know but also from their ignorance. This approach, which focuses on recognizing and managing uncertainty, offers a potential solution to improve the resilience and robustness of AI systems, ensuring that they can better handle unpredictable real-world environments. 

**Abstract (ZH)**: 尽管人工智能取得了令人印象深刻的成就，包括生成模型和大规模语言模型的进步，但在处理不确定性以及泛化到训练数据之外的能力方面，人工智能仍然存在显著差距。本文认为，在自主系统中，当面对不熟悉或恶意的数据时，人工智能模型难以做出稳健的预测，这在自动驾驶车辆 incidents 中得到了体现。传统的机器学习方法由于过分强调数据拟合和领域适应，在解决这些问题上面临挑战。本文提出了一种范式转变，即认识论人工智能，强调模型不仅要从已知的信息中学习，还要从无知中学习。这种专注于识别和管理不确定性的方法，可能为提高人工智能系统的韧性和鲁棒性提供解决方案，确保它们能够更好地处理不可预测的现实环境。 

---
# Belief Filtering for Epistemic Control in Linguistic State Space 

**Title (ZH)**: 信念筛选在语言状态空间中的知识控制 

**Authors**: Sebastian Dumbrava  

**Link**: [PDF](https://arxiv.org/pdf/2505.04927)  

**Abstract**: We examine belief filtering as a mechanism for the epistemic control of artificial agents, focusing on the regulation of internal cognitive states represented as linguistic expressions. This mechanism is developed within the Semantic Manifold framework, where belief states are dynamic, structured ensembles of natural language fragments. Belief filters act as content-aware operations on these fragments across various cognitive transitions. This paper illustrates how the inherent interpretability and modularity of such a linguistically-grounded cognitive architecture directly enable belief filtering, offering a principled approach to agent regulation. The study highlights the potential for enhancing AI safety and alignment through structured interventions in an agent's internal semantic space and points to new directions for architecturally embedded cognitive governance. 

**Abstract (ZH)**: 我们探讨信念过滤作为一种机制，用于人工代理的epistemic控制，重点关注以语言表达形式表现的内部认知状态的调节。该机制在语义流形框架内发展，其中信念状态是自然语言片段构成的动力学、结构化集合。信念过滤器作为内容感知的操作，在各种认知转换过程中作用于这些片段。本文说明了这种以语言为基础的认知架构固有的可解释性和模块性如何直接促进信念过滤，提出了一种原理性的代理调节方法。研究突显了通过在代理内部语义空间中进行结构化干预来增强AI安全性和对齐的潜在可能性，并指出了嵌入式认知治理的新方向。 

---
# Enigme: Generative Text Puzzles for Evaluating Reasoning in Language Models 

**Title (ZH)**: Enigme：生成文本谜题以评估语言模型的推理能力 

**Authors**: John Hawkins  

**Link**: [PDF](https://arxiv.org/pdf/2505.04914)  

**Abstract**: Transformer-decoder language models are a core innovation in text based generative artificial intelligence. These models are being deployed as general-purpose intelligence systems in many applications. Central to their utility is the capacity to understand natural language commands and exploit the reasoning embedded in human text corpora to apply some form of reasoning process to a wide variety of novel tasks. To understand the limitations of this approach to generating reasoning we argue that we need to consider the architectural constraints of these systems. Consideration of the latent variable structure of transformer-decoder models allows us to design reasoning tasks that should probe the boundary of their capacity to reason. We present enigme, an open-source library for generating text-based puzzles to be used in training and evaluating reasoning skills within transformer-decoder models and future AI architectures. 

**Abstract (ZH)**: 基于Transformer解码器的语言模型是文本生成式人工智能的核心创新。这些模型正在许多应用中被部署为通用智能系统。它们的功能核心在于理解自然语言命令，并利用人类文本语料库中嵌入的推理能力来处理各种新型任务。为了理解生成推理的这一方法的局限性，我们认为需要考虑这些系统架构上的限制。通过考虑Transformer解码器模型的潜在变量结构，我们可以设计出能够探测其推理能力边界的推理任务。我们介绍了enigme，一个开源库，用于生成基于文本的谜题，以用于训练和评估Transformer解码器模型及未来AI架构中的推理能力。 

---
# CRAFT: Cultural Russian-Oriented Dataset Adaptation for Focused Text-to-Image Generation 

**Title (ZH)**: CRAFT：面向聚焦文本到图像生成的文化俄语导向数据集适配 

**Authors**: Viacheslav Vasilev, Vladimir Arkhipkin, Julia Agafonova, Tatiana Nikulina, Evelina Mironova, Alisa Shichanina, Nikolai Gerasimenko, Mikhail Shoytov, Denis Dimitrov  

**Link**: [PDF](https://arxiv.org/pdf/2505.04851)  

**Abstract**: Despite the fact that popular text-to-image generation models cope well with international and general cultural queries, they have a significant knowledge gap regarding individual cultures. This is due to the content of existing large training datasets collected on the Internet, which are predominantly based on Western European or American popular culture. Meanwhile, the lack of cultural adaptation of the model can lead to incorrect results, a decrease in the generation quality, and the spread of stereotypes and offensive content. In an effort to address this issue, we examine the concept of cultural code and recognize the critical importance of its understanding by modern image generation models, an issue that has not been sufficiently addressed in the research community to date. We propose the methodology for collecting and processing the data necessary to form a dataset based on the cultural code, in particular the Russian one. We explore how the collected data affects the quality of generations in the national domain and analyze the effectiveness of our approach using the Kandinsky 3.1 text-to-image model. Human evaluation results demonstrate an increase in the level of awareness of Russian culture in the model. 

**Abstract (ZH)**: 尽管现有的文本到图像生成模型在处理国际和通用文化查询方面表现良好，但在处理个体文化方面存在明显的知识缺口。这主要是因为现有大型训练数据集的内容主要基于西方欧洲或美国的流行文化，收集于互联网。同时，模型的文化适应不足可能导致错误结果、生成质量下降以及刻板印象和有害内容的传播。为解决这一问题，我们探讨了文化代码的概念，并认识到现代图像生成模型对其理解的迫切重要性，这一问题至今尚未在研究界得到充分关注。我们提出了一种收集和处理数据的方法，以形成基于文化代码、特别是俄罗斯文化代码的数据库。我们探究了收集的数据如何影响国内领域的生成质量，并使用Kandinsky 3.1文本到图像模型分析我们方法的有效性。人类评估结果显示，模型对俄罗斯文化的意识水平有所提高。 

---
# Large Language Models are Autonomous Cyber Defenders 

**Title (ZH)**: 大型语言模型是自主网络防御者 

**Authors**: Sebastián R. Castro, Roberto Campbell, Nancy Lau, Octavio Villalobos, Jiaqi Duan, Alvaro A. Cardenas  

**Link**: [PDF](https://arxiv.org/pdf/2505.04843)  

**Abstract**: Fast and effective incident response is essential to prevent adversarial cyberattacks. Autonomous Cyber Defense (ACD) aims to automate incident response through Artificial Intelligence (AI) agents that plan and execute actions. Most ACD approaches focus on single-agent scenarios and leverage Reinforcement Learning (RL). However, ACD RL-trained agents depend on costly training, and their reasoning is not always explainable or transferable. Large Language Models (LLMs) can address these concerns by providing explainable actions in general security contexts. Researchers have explored LLM agents for ACD but have not evaluated them on multi-agent scenarios or interacting with other ACD agents. In this paper, we show the first study on how LLMs perform in multi-agent ACD environments by proposing a new integration to the CybORG CAGE 4 environment. We examine how ACD teams of LLM and RL agents can interact by proposing a novel communication protocol. Our results highlight the strengths and weaknesses of LLMs and RL and help us identify promising research directions to create, train, and deploy future teams of ACD agents. 

**Abstract (ZH)**: Fast and Effective 多Agent 事件响应对于防止对抗性网络攻击至关重要。自主网络安全防护（ACD）旨在通过人工智能（AI）代理自动执行事件响应。大多数 ACD 方法侧重于单代理场景，并利用强化学习（RL）进行训练。然而，ACD RL 训练代理依赖于昂贵的训练过程，其推理过程往往缺乏可解释性和可迁移性。大规模语言模型（LLMs）可以通过在一般安全背景下提供可解释的行动来解决这些问题。研究人员已经探索了LLM代理在ACD中的应用，但尚未在多代理场景或与其他ACD代理交互的情况下进行评估。本文通过提出一种新的集成到CybORG CAGE 4环境中的方法，展示了LLMs在多代理ACD环境中的第一个研究成果，并提出了一种新的通信协议来探讨ACD团队中LLM和RL代理的交互方式。我们的研究结果突显了LLMs和RL的优缺点，并帮助我们确定未来ACD代理团队创建、训练和部署的研究方向。 

---
# Is there Value in Reinforcement Learning? 

**Title (ZH)**: 强化学习是否有价值？ 

**Authors**: Lior Fox, Yonatan Loewenstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.04822)  

**Abstract**: Action-values play a central role in popular Reinforcement Learing (RL) models of behavior. Yet, the idea that action-values are explicitly represented has been extensively debated. Critics had therefore repeatedly suggested that policy-gradient (PG) models should be favored over value-based (VB) ones, as a potential solution for this dilemma. Here we argue that this solution is unsatisfying. This is because PG methods are not, in fact, "Value-free" -- while they do not rely on an explicit representation of Value for acting (stimulus-response mapping), they do require it for learning. Hence, switching to PG models is, per se, insufficient for eliminating Value from models of behavior. More broadly, the requirement for a representation of Value stems from the underlying assumptions regarding the optimization objective posed by the standard RL framework, not from the particular algorithm chosen to solve it. Previous studies mostly took these standard RL assumptions for granted, as part of their conceptualization or problem modeling, while debating the different methods used to optimize it (i.e., PG or VB). We propose that, instead, the focus of the debate should shift to critically evaluating the underlying modeling assumptions. Such evaluation is particularly important from an experimental perspective. Indeed, the very notion of Value must be reconsidered when standard assumptions (e.g., risk neutrality, full-observability, Markovian environment, exponential discounting) are relaxed, as is likely in natural settings. Finally, we use the Value debate as a case study to argue in favor of a more nuanced, algorithmic rather than statistical, view of what constitutes "a model" in cognitive sciences. Our analysis suggests that besides "parametric" statistical complexity, additional aspects such as computational complexity must also be taken into account when evaluating model complexity. 

**Abstract (ZH)**: 行动值在行为的强化学习模型中扮演核心角色，然而行动值显式表示的想法一直受到广泛讨论。我们argue指出，改用策略梯度模型并不能从根本上消除行动值，因为策略梯度方法尽管不依赖于行动的显式价值表示（刺激-响应映射），但在学习过程中仍需依赖价值表示。因此，切换到策略梯度模型本身不足以消除行为模型中的价值因素。更广泛而言，价值表示的需求源自标准强化学习框架所提出的优化目标的潜在假设，而非所选择的具体算法。以往研究大多假定这些标准假设，并将其作为概念化或问题建模的一部分，而争论的是不同的优化方法（即策略梯度或基于价值的方法）。我们提出，争论的焦点应该转移到对潜在建模假设的批判性评估上。这种评估从实验角度来看尤为重要。事实上，当放松标准假设（如无风险偏好、完全可观测性、马尔可夫环境、指数折扣）时，必须重新考虑价值这一概念，这在自然环境中很可能是事实。最后，我们使用价值争议作为案例研究，主张认知科学中“模型”的构成应采取更为细腻的方法论视角，而不是单纯的统计视角。我们的分析表明，在评估模型复杂度时，除统计复杂性外，还需要考虑计算复杂性等因素。 

---
# The Promise and Limits of LLMs in Constructing Proofs and Hints for Logic Problems in Intelligent Tutoring Systems 

**Title (ZH)**: LLMs在智能辅导系统中构建逻辑问题证明和提示的潜力与局限性 

**Authors**: Sutapa Dey Tithi, Arun Kumar Ramesh, Clara DiMarco, Xiaoyi Tian, Nazia Alam, Kimia Fazeli, Tiffany Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2505.04736)  

**Abstract**: Intelligent tutoring systems have demonstrated effectiveness in teaching formal propositional logic proofs, but their reliance on template-based explanations limits their ability to provide personalized student feedback. While large language models (LLMs) offer promising capabilities for dynamic feedback generation, they risk producing hallucinations or pedagogically unsound explanations. We evaluated the stepwise accuracy of LLMs in constructing multi-step symbolic logic proofs, comparing six prompting techniques across four state-of-the-art LLMs on 358 propositional logic problems. Results show that DeepSeek-V3 achieved superior performance with 84.4% accuracy on stepwise proof construction and excelled particularly in simpler rules. We further used the best-performing LLM to generate explanatory hints for 1,050 unique student problem-solving states from a logic ITS and evaluated them on 4 criteria with both an LLM grader and human expert ratings on a 20% sample. Our analysis finds that LLM-generated hints were 75% accurate and rated highly by human evaluators on consistency and clarity, but did not perform as well explaining why the hint was provided or its larger context. Our results demonstrate that LLMs may be used to augment tutoring systems with logic tutoring hints, but requires additional modifications to ensure accuracy and pedagogical appropriateness. 

**Abstract (ZH)**: 智能辅导系统在教学形式命题逻辑证明方面表现出有效性，但其依赖于模板解释的限制影响了其提供个性化学生反馈的能力。尽管大规模语言模型（LLMs）提供了动态反馈生成的前景，但也存在产生幻觉或教学上不合适的解释的风险。我们评估了LLMs在构建多步骤符号逻辑证明时的逐步准确性，比较了六种提示技术在4个最先进的LLMs上解决358个命题逻辑问题的表现。结果表明，DeepSeek-V3实现了更高的性能，逐步证明构建准确率为84.4%，尤其在简单规则方面表现出色。我们进一步使用表现最佳的LLM生成了源自逻辑ITS的1,050个独特学生问题解决状态的解释性提示，并对这些建议进行了四项标准的评估，包括LLM评分和人工专家在20%样本上的评分。分析结果发现，LLM生成的提示准确率为75%，在一致性和清晰性方面得到了人类评估者的高度评价，但不太擅长解释提供提示的原因及其更大的上下文。我们的结果表明，LLMs可以用于增强逻辑辅导系统的提示，但需要进一步的修改以确保准确性和教学适宜性。 

---
# Dynamic Location Search for Identifying Maximum Weighted Independent Sets in Complex Networks 

**Title (ZH)**: 复杂网络中最大加权独立集的动态位置搜索 

**Authors**: Enqiang Zhu, Chenkai Hao, Chanjuan Liu, Yongsheng Rao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04674)  

**Abstract**: While Artificial intelligence (AI), including Generative AI, are effective at generating high-quality traffic data and optimization solutions in intelligent transportation systems (ITSs), these techniques often demand significant training time and computational resources, especially in large-scale and complex scenarios. To address this, we introduce a novel and efficient algorithm for solving the maximum weighted independent set (MWIS) problem, which can be used to model many ITSs applications, such as traffic signal control and vehicle routing. Given the NP-hard nature of the MWIS problem, our proposed algorithm, DynLS, incorporates three key innovations to solve it effectively. First, it uses a scores-based adaptive vertex perturbation (SAVP) technique to accelerate convergence, particularly in sparse graphs. Second, it includes a region location mechanism (RLM) to help escape local optima by dynamically adjusting the search space. Finally, it employs a novel variable neighborhood descent strategy, ComLS, which combines vertex exchange strategies with a reward mechanism to guide the search toward high-quality solutions. Our experimental results demonstrate DynLS's superior performance, consistently delivering high-quality solutions within 1000 seconds. DynLS outperformed five leading algorithms across 360 test instances, achieving the best solution for 350 instances and surpassing the second-best algorithm, Cyclic-Fast, by 177 instances. Moreover, DynLS matched Cyclic-Fast's convergence speed, highlighting its efficiency and practicality. This research represents a significant advancement in heuristic algorithms for the MWIS problem, offering a promising approach to aid AI techniques in optimizing intelligent transportation systems. 

**Abstract (ZH)**: 一种用于解决最大加权独立集问题的高效算法DynLS及其在智能交通系统中的应用 

---
# Computational Irreducibility as the Foundation of Agency: A Formal Model Connecting Undecidability to Autonomous Behavior in Complex Systems 

**Title (ZH)**: 计算不可约性作为agency的基础：将不可判定性与复杂系统中的自主行为正式模型连接起来 

**Authors**: Poria Azadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04646)  

**Abstract**: This article explores the emergence of autonomy and agency by connecting fundamental computational limits (decidability, completeness, computational irreducibility) with physical concepts. We introduce a formal model of a "minimal agent" operating within potentially Turing-complete environments. Using algorithmic information theory, we argue that the inherent undecidability and computational irreducibility of agent-environment interaction lead to unpredictability and novel information generation, enabling agency (effective goal-directed action). Computational irreducibility prevents full external prediction, creating necessary conditions for autonomous behavior. We relate this to computational sourcehood, where an agent is the irreducible origin of its behavior, though formalizing this concept remains challenging. Our central thesis, formally proven, is that genuine autonomy necessarily implies undecidability from an external perspective, distinguishing autonomous systems from predictable ones. We propose that agency arises when agent-environment coupling complexity allows mutual information between internal states and relevant environmental variables to increase, particularly where analytical solutions are absent and operational closure is needed for persistence. This framework links agency directly to the computational properties of interaction, offering implications for understanding consciousness, designing autonomous AI, and reconceptualizing free will in a deterministic yet computationally irreducible universe. 

**Abstract (ZH)**: 本文通过将基本的计算极限（决定性、完备性、计算不可约性）与物理概念相连，探讨自主性和能动力的涌现。我们介绍了一个“最小代理”形式模型，该代理在潜在图灵完备的环境中运行。利用算法信息理论，我们argue认为代理-环境交互的固有不可决定性和计算不可约性导致不可预测性和新颖信息的生成，从而实现能动力（有效目标导向行为）。计算不可约性阻止完全外部预测，从而创造自主行为的必要条件。我们将这一概念与计算源性相关联，其中代理是其行为的不可约来源，尽管正式化这一概念仍然具有挑战性。我们的中心论点，已被正式证明，是真正的自主性必然意味着从外部视角来看的不可决定性，从而将自主系统与可预测系统区分开来。我们提出，当代理-环境耦合的复杂性允许内部状态与相关环境变量之间的互信息增加时，能动力得以产生，特别是在缺乏解析解且需要操作闭包以维持行为持续性的情况下。这一框架直接将能动力与交互的计算性质相连，为其理解意识、设计自主AI以及在确定论但计算不可约的宇宙中重新概念化自由意志提供了意义。 

---
# Towards Artificial Intelligence Research Assistant for Expert-Involved Learning 

**Title (ZH)**: 面向专家参与学习的人工智能研究助手 

**Authors**: Tianyu Liu, Simeng Han, Xiao Luo, Hanchen Wang, Pan Lu, Biqing Zhu, Yuge Wang, Keyi Li, Jiapeng Chen, Rihao Qu, Yufeng Liu, Xinyue Cui, Aviv Yaish, Yuhang Chen, Minsheng Hao, Chuhan Li, Kexing Li, Arman Cohan, Hua Xu, Mark Gerstein, James Zou, Hongyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04638)  

**Abstract**: Large Language Models (LLMs) and Large Multi-Modal Models (LMMs) have emerged as transformative tools in scientific research, yet their reliability and specific contributions to biomedical applications remain insufficiently characterized. In this study, we present \textbf{AR}tificial \textbf{I}ntelligence research assistant for \textbf{E}xpert-involved \textbf{L}earning (ARIEL), a multimodal dataset designed to benchmark and enhance two critical capabilities of LLMs and LMMs in biomedical research: summarizing extensive scientific texts and interpreting complex biomedical figures. To facilitate rigorous assessment, we create two open-source sets comprising biomedical articles and figures with designed questions. We systematically benchmark both open- and closed-source foundation models, incorporating expert-driven human evaluations conducted by doctoral-level experts. Furthermore, we improve model performance through targeted prompt engineering and fine-tuning strategies for summarizing research papers, and apply test-time computational scaling to enhance the reasoning capabilities of LMMs, achieving superior accuracy compared to human-expert corrections. We also explore the potential of using LMM Agents to generate scientific hypotheses from diverse multimodal inputs. Overall, our results delineate clear strengths and highlight significant limitations of current foundation models, providing actionable insights and guiding future advancements in deploying large-scale language and multi-modal models within biomedical research. 

**Abstract (ZH)**: 用于专家参与学习的人工智能研究助手（ARIEL）：大规模语言模型和大规模多模态模型在生物医学研究中的评估与提升 

---
# Flow-GRPO: Training Flow Matching Models via Online RL 

**Title (ZH)**: 基于在线强化学习的流匹配模型训练方法 

**Authors**: Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05470)  

**Abstract**: We propose Flow-GRPO, the first method integrating online reinforcement learning (RL) into flow matching models. Our approach uses two key strategies: (1) an ODE-to-SDE conversion that transforms a deterministic Ordinary Differential Equation (ODE) into an equivalent Stochastic Differential Equation (SDE) that matches the original model's marginal distribution at all timesteps, enabling statistical sampling for RL exploration; and (2) a Denoising Reduction strategy that reduces training denoising steps while retaining the original inference timestep number, significantly improving sampling efficiency without performance degradation. Empirically, Flow-GRPO is effective across multiple text-to-image tasks. For complex compositions, RL-tuned SD3.5 generates nearly perfect object counts, spatial relations, and fine-grained attributes, boosting GenEval accuracy from $63\%$ to $95\%$. In visual text rendering, its accuracy improves from $59\%$ to $92\%$, significantly enhancing text generation. Flow-GRPO also achieves substantial gains in human preference alignment. Notably, little to no reward hacking occurred, meaning rewards did not increase at the cost of image quality or diversity, and both remained stable in our experiments. 

**Abstract (ZH)**: Flow-GRPO：将在线强化学习集成到流匹配模型中的首次尝试 

---
# StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant 

**Title (ZH)**: StreamBridge: 将您的离线视频大型语言模型转化为 proactive 流式助手 

**Authors**: Haibo Wang, Bo Feng, Zhengfeng Lai, Mingze Xu, Shiyu Li, Weifeng Ge, Afshin Dehghan, Meng Cao, Ping Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05467)  

**Abstract**: We present StreamBridge, a simple yet effective framework that seamlessly transforms offline Video-LLMs into streaming-capable models. It addresses two fundamental challenges in adapting existing models into online scenarios: (1) limited capability for multi-turn real-time understanding, and (2) lack of proactive response mechanisms. Specifically, StreamBridge incorporates (1) a memory buffer combined with a round-decayed compression strategy, supporting long-context multi-turn interactions, and (2) a decoupled, lightweight activation model that can be effortlessly integrated into existing Video-LLMs, enabling continuous proactive responses. To further support StreamBridge, we construct Stream-IT, a large-scale dataset tailored for streaming video understanding, featuring interleaved video-text sequences and diverse instruction formats. Extensive experiments show that StreamBridge significantly improves the streaming understanding capabilities of offline Video-LLMs across various tasks, outperforming even proprietary models such as GPT-4o and Gemini 1.5 Pro. Simultaneously, it achieves competitive or superior performance on standard video understanding benchmarks. 

**Abstract (ZH)**: StreamBridge：一种简单有效的框架，无缝地将离线Video-LLMs转换为流式模型，解决现有模型适应在线场景中的两大根本挑战：（1）多轮实时理解能力有限，（2）缺乏主动响应机制。 

---
# ComPO: Preference Alignment via Comparison Oracles 

**Title (ZH)**: ComPO：通过比较或acles实现偏好对齐 

**Authors**: Peter Chen, Xi Chen, Wotao Yin, Tianyi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05465)  

**Abstract**: Direct alignment methods are increasingly used for aligning large language models (LLMs) with human preferences. However, these methods suffer from the issues of verbosity and likelihood displacement, which can be driven by the noisy preference pairs that induce similar likelihood for preferred and dispreferred responses. The contributions of this paper are two-fold. First, we propose a new preference alignment method based on comparison oracles and provide the convergence guarantee for its basic scheme. Second, we improve our method using some heuristics and conduct the experiments to demonstrate the flexibility and compatibility of practical scheme in improving the performance of LLMs using noisy preference pairs. Evaluations are conducted across multiple base and instruction-tuned models (Mistral-7B, Llama-3-8B and Gemma-2-9B) with benchmarks (AlpacaEval 2, MT-Bench and Arena-Hard). Experimental results show the effectiveness of our method as an alternative to addressing the limitations of existing direct alignment methods. A highlight of our work is that we evidence the importance of designing specialized methods for preference pairs with distinct likelihood margin, which complements the recent findings in \citet{Razin-2025-Unintentional}. 

**Abstract (ZH)**: 直接对齐方法越来越多地被用于将大型语言模型（LLMs）与人类偏好对齐。然而，这些方法受到冗长性和似然性位移的问题困扰，这些问题可能是由引起偏好响应和未偏好响应类似似然性的嘈杂偏好对驱动的。本文的贡献主要有两点。首先，我们提出了一种基于比较或acles的新偏好对齐方法，并为其基本方案提供了收敛性保证。其次，我们通过一些启发式方法改进了该方法，并通过实验展示了如何使用嘈杂的偏好对提高LLM性能的灵活性和兼容性。我们在多种基模型和指令调整模型（Mistral-7B、Llama-3-8B 和 Gemma-2-9B）以及基准数据集（AlpacaEval 2、MT-Bench 和 Arena-Hard）上进行了评估。实验结果表明，我们的方法可以作为克服现有直接对齐方法局限性的替代方案的有效性。我们的工作亮点在于，我们证明了为具有不同似然性边际的偏好对设计专门方法的重要性，这补充了最近 \citet{Razin-2025-Unintentional} 的发现。 

---
# TransProQA: an LLM-based literary Translation evaluation metric with Professional Question Answering 

**Title (ZH)**: TransProQA: 一种基于大型语言模型的专业问答文学翻译评估指标 

**Authors**: Ran Zhang, Wei Zhao, Lieve Macken, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2505.05423)  

**Abstract**: The impact of Large Language Models (LLMs) has extended into literary domains. However, existing evaluation metrics prioritize mechanical accuracy over artistic expression and tend to overrate machine translation (MT) as being superior to experienced professional human translation. In the long run, this bias could result in a permanent decline in translation quality and cultural authenticity. In response to the urgent need for a specialized literary evaluation metric, we introduce TransProQA, a novel, reference-free, LLM-based question-answering (QA) framework designed specifically for literary translation evaluation. TransProQA uniquely integrates insights from professional literary translators and researchers, focusing on critical elements in literary quality assessment such as literary devices, cultural understanding, and authorial voice. Our extensive evaluation shows that while literary-finetuned XCOMET-XL yields marginal gains, TransProQA substantially outperforms current metrics, achieving up to 0.07 gain in correlation (ACC-EQ and Kendall's tau) and surpassing the best state-of-the-art (SOTA) metrics by over 15 points in adequacy assessments. Incorporating professional translator insights as weights further improves performance, highlighting the value of translator inputs. Notably, TransProQA approaches human-level evaluation performance comparable to trained linguistic annotators. It demonstrates broad applicability to open-source models such as LLaMA3.3-70b and Qwen2.5-32b, indicating its potential as an accessible and training-free literary evaluation metric and a valuable tool for evaluating texts that require local processing due to copyright or ethical considerations. 

**Abstract (ZH)**: 大型语言模型对文学领域的 Impact 建立专用于文学翻译评价的 TransProQA：一种参考无关的基于大型语言模型的问题回答框架 

---
# TokLIP: Marry Visual Tokens to CLIP for Multimodal Comprehension and Generation 

**Title (ZH)**: TokLIP：将视觉令牌与CLIP结合用于多模态理解和生成 

**Authors**: Haokun Lin, Teng Wang, Yixiao Ge, Yuying Ge, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2505.05422)  

**Abstract**: Pioneering token-based works such as Chameleon and Emu3 have established a foundation for multimodal unification but face challenges of high training computational overhead and limited comprehension performance due to a lack of high-level semantics. In this paper, we introduce TokLIP, a visual tokenizer that enhances comprehension by semanticizing vector-quantized (VQ) tokens and incorporating CLIP-level semantics while enabling end-to-end multimodal autoregressive training with standard VQ tokens. TokLIP integrates a low-level discrete VQ tokenizer with a ViT-based token encoder to capture high-level continuous semantics. Unlike previous approaches (e.g., VILA-U) that discretize high-level features, TokLIP disentangles training objectives for comprehension and generation, allowing the direct application of advanced VQ tokenizers without the need for tailored quantization operations. Our empirical results demonstrate that TokLIP achieves exceptional data efficiency, empowering visual tokens with high-level semantic understanding while enhancing low-level generative capacity, making it well-suited for autoregressive Transformers in both comprehension and generation tasks. The code and models are available at this https URL. 

**Abstract (ZH)**: 基于令牌的先驱工作Chameleon和Emu3为多模态统一奠定了基础，但由于缺乏高层语义，面临高额训练计算开销和有限理解性能的挑战。本文介绍TokLIP，一个通过语义化向量量化（VQ）令牌和融入CLIP级语义来提升理解能力的同时，支持使用标准VQ令牌进行端到端多模态自回归训练的视觉令牌化器。TokLIP将低级离散VQ令牌化器与基于ViT的令牌编码器结合，以捕捉高层连续语义。与此前将高层特征离散化的做法（例如VILA-U）不同，TokLIP分离了理解与生成的训练目标，允许直接应用高级VQ令牌化器，无需特定的量化操作。我们的实验证明，TokLIP在数据效率方面表现出色，赋予视觉令牌高层语义理解能力，同时增强低级生成能力，使其适合于理解与生成任务中的自回归Transformer。代码和模型可在如下链接获取：this https URL。 

---
# Reasoning Models Don't Always Say What They Think 

**Title (ZH)**: Reasoning Models 不总是说它们想说的 

**Authors**: Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, Vlad Mikulik, Samuel R. Bowman, Jan Leike, Jared Kaplan, Ethan Perez  

**Link**: [PDF](https://arxiv.org/pdf/2505.05410)  

**Abstract**: Chain-of-thought (CoT) offers a potential boon for AI safety as it allows monitoring a model's CoT to try to understand its intentions and reasoning processes. However, the effectiveness of such monitoring hinges on CoTs faithfully representing models' actual reasoning processes. We evaluate CoT faithfulness of state-of-the-art reasoning models across 6 reasoning hints presented in the prompts and find: (1) for most settings and models tested, CoTs reveal their usage of hints in at least 1% of examples where they use the hint, but the reveal rate is often below 20%, (2) outcome-based reinforcement learning initially improves faithfulness but plateaus without saturating, and (3) when reinforcement learning increases how frequently hints are used (reward hacking), the propensity to verbalize them does not increase, even without training against a CoT monitor. These results suggest that CoT monitoring is a promising way of noticing undesired behaviors during training and evaluations, but that it is not sufficient to rule them out. They also suggest that in settings like ours where CoT reasoning is not necessary, test-time monitoring of CoTs is unlikely to reliably catch rare and catastrophic unexpected behaviors. 

**Abstract (ZH)**: Chain-of-Thought监控对于AI安全的潜在益处及其局限性：基于最新推理模型的评估 

---
# Crosslingual Reasoning through Test-Time Scaling 

**Title (ZH)**: 跨语言推理通过测试时缩放 

**Authors**: Zheng-Xin Yong, M. Farid Adilazuarda, Jonibek Mansurov, Ruochen Zhang, Niklas Muennighoff, Carsten Eickhoff, Genta Indra Winata, Julia Kreutzer, Stephen H. Bach, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2505.05408)  

**Abstract**: Reasoning capabilities of large language models are primarily studied for English, even when pretrained models are multilingual. In this work, we investigate to what extent English reasoning finetuning with long chain-of-thoughts (CoTs) can generalize across languages. First, we find that scaling up inference compute for English-centric reasoning language models (RLMs) improves multilingual mathematical reasoning across many languages including low-resource languages, to an extent where they outperform models twice their size. Second, we reveal that while English-centric RLM's CoTs are naturally predominantly English, they consistently follow a quote-and-think pattern to reason about quoted non-English inputs. Third, we discover an effective strategy to control the language of long CoT reasoning, and we observe that models reason better and more efficiently in high-resource languages. Finally, we observe poor out-of-domain reasoning generalization, in particular from STEM to cultural commonsense knowledge, even for English. Overall, we demonstrate the potentials, study the mechanisms and outline the limitations of crosslingual generalization of English reasoning test-time scaling. We conclude that practitioners should let English-centric RLMs reason in high-resource languages, while further work is needed to improve reasoning in low-resource languages and out-of-domain contexts. 

**Abstract (ZH)**: 大型语言模型的推理能力主要针对英语进行研究，即使预训练模型是多语言的。在本工作中，我们调查了以英语为中心的长链推理（CoTs）的推理微调在多大程度上能在不同语言之间泛化。首先，我们发现，增加以英语为中心的推理语言模型（RLMs）推理计算的规模，可以在包括低资源语言在内的多种语言中提高多语言数学推理能力，甚至使其优于两倍规模的模型。其次，我们揭示了虽然以英语为中心的RLMs的CoTs通常是英语为主的，但它们会一致地遵循引用并思考的模式来推理非英语输入。第三，我们发现了一种控制长CoTs推理语言的有效策略，并发现模型在高资源语言中推理更好且更高效。最后，我们观察到，在特定领域外的推理泛化表现较差，尤其是在STEM领域到文化常识知识方面，即使是英语也不例外。总体而言，我们展示了英语推理测试时缩放的跨语言泛化的潜力，研究了其机制并概述了其局限性。我们得出结论，实践者应该让以英语为中心的RLMs在高资源语言中推理，而进一步的工作需要提高低资源语言和特定领域外语境中的推理能力。 

---
# CART-ELC: Oblique Decision Tree Induction via Exhaustive Search 

**Title (ZH)**: CART-ELC：基于穷尽搜索的斜向决策树诱导 

**Authors**: Andrew D. Laack  

**Link**: [PDF](https://arxiv.org/pdf/2505.05402)  

**Abstract**: Oblique decision trees have attracted attention due to their potential for improved classification performance over traditional axis-aligned decision trees. However, methods that rely on exhaustive search to find oblique splits face computational challenges. As a result, they have not been widely explored. We introduce a novel algorithm, Classification and Regression Tree - Exhaustive Linear Combinations (CART-ELC), for inducing oblique decision trees that performs an exhaustive search on a restricted set of hyperplanes. We then investigate the algorithm's computational complexity and its predictive capabilities. Our results demonstrate that CART-ELC consistently achieves competitive performance on small datasets, often yielding statistically significant improvements in classification accuracy relative to existing decision tree induction algorithms, while frequently producing shallower, simpler, and thus more interpretable trees. 

**Abstract (ZH)**: 斜决策树由于潜在的分类性能改进而受到了关注，但依赖于详尽搜索以找到斜分割的方法面临计算挑战，因此它们并未得到广泛研究。我们提出了一种新的算法——分类和回归树-详尽线性组合（CART-ELC），该算法在受限的超平面集上进行详尽搜索以诱导斜决策树。然后，我们探讨了该算法的计算复杂性和预测能力。我们的结果表明，CART-ELC在小型数据集上能够一致地实现竞争性能，通常在分类准确性方面相对于现有决策树诱导算法表现出统计意义上的显著改进，同时经常生成更浅、更简单、因此更具解释性的树结构。 

---
# Threshold Modulation for Online Test-Time Adaptation of Spiking Neural Networks 

**Title (ZH)**: 基于阈值调制的在线测试时SNN的自适应调整 

**Authors**: Kejie Zhao, Wenjia Hua, Aiersi Tuerhong, Luziwei Leng, Yuxin Ma, Qinghua Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.05375)  

**Abstract**: Recently, spiking neural networks (SNNs), deployed on neuromorphic chips, provide highly efficient solutions on edge devices in different scenarios. However, their ability to adapt to distribution shifts after deployment has become a crucial challenge. Online test-time adaptation (OTTA) offers a promising solution by enabling models to dynamically adjust to new data distributions without requiring source data or labeled target samples. Nevertheless, existing OTTA methods are largely designed for traditional artificial neural networks and are not well-suited for SNNs. To address this gap, we propose a low-power, neuromorphic chip-friendly online test-time adaptation framework, aiming to enhance model generalization under distribution shifts. The proposed approach is called Threshold Modulation (TM), which dynamically adjusts the firing threshold through neuronal dynamics-inspired normalization, being more compatible with neuromorphic hardware. Experimental results on benchmark datasets demonstrate the effectiveness of this method in improving the robustness of SNNs against distribution shifts while maintaining low computational cost. The proposed method offers a practical solution for online test-time adaptation of SNNs, providing inspiration for the design of future neuromorphic chips. The demo code is available at this http URL. 

**Abstract (ZH)**: 近来，部署在神经形态芯片上的脉冲神经网络（SNNs）在不同场景下的边缘设备上提供了高效的解决方案。然而，在部署后适应分布偏移的能力已成为一个关键挑战。在线测试时适应（OTTA）通过使模型能够动态调整以适应新的数据分布，而无需源数据或标记的目标样本，提供了一种有前景的解决方案。然而，现有的OTTA方法主要针对传统的人工神经网络设计，不适用于SNNs。为了解决这一问题，我们提出了一种低功耗、适用于神经形态芯片的在线测试时适应框架，旨在在分布偏移下增强模型的泛化能力。该提出的方案称为阈值调制（TM），通过神经元动力学启发的归一化动态调整放电阈值，更适配于神经形态硬件。基准数据集上的实验结果证明了该方法在提高SNNs对分布偏移的鲁棒性的同时保持了低计算成本的有效性。该方法提供了一种实际的解决方案，用于SNNs的在线测试时适应，并为未来神经形态芯片的设计提供了启示。完整的演示代码可在以下网址获取。 

---
# Time of the Flight of the Gaussians: Optimizing Depth Indirectly in Dynamic Radiance Fields 

**Title (ZH)**: 高斯分布的时间飞行：在动态辐射场中间接优化深度 

**Authors**: Runfeng Li, Mikhail Okunev, Zixuan Guo, Anh Ha Duong, Christian Richardt, Matthew O'Toole, James Tompkin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05356)  

**Abstract**: We present a method to reconstruct dynamic scenes from monocular continuous-wave time-of-flight (C-ToF) cameras using raw sensor samples that achieves similar or better accuracy than neural volumetric approaches and is 100x faster. Quickly achieving high-fidelity dynamic 3D reconstruction from a single viewpoint is a significant challenge in computer vision. In C-ToF radiance field reconstruction, the property of interest-depth-is not directly measured, causing an additional challenge. This problem has a large and underappreciated impact upon the optimization when using a fast primitive-based scene representation like 3D Gaussian splatting, which is commonly used with multi-view data to produce satisfactory results and is brittle in its optimization otherwise. We incorporate two heuristics into the optimization to improve the accuracy of scene geometry represented by Gaussians. Experimental results show that our approach produces accurate reconstructions under constrained C-ToF sensing conditions, including for fast motions like swinging baseball bats. this https URL 

**Abstract (ZH)**: 我们提出了一种方法，使用单目连续波时间飞行（C-ToF）相机的原始传感器样本来重建动态场景，实现了与神经体积方法相当或更好的精度，并且速度快100倍。快速从单一视角实现高保真动态3D重建是计算机视觉中的一个重大挑战。在C-ToF辐照度场重建中，需要测量的属性-深度-无法直接测量，导致了额外的挑战。当使用诸如3D高斯散斑等快速基础场景表示时，这个问题对优化产生了巨大且被低估的影响，而后者在多视角数据中通常能产生满意的结果，但在其他情况下优化过程是脆弱的。我们通过引入两个启发式方法改进了基于高斯表示的场景几何精度。实验结果表明，我们的方法在受限的C-ToF感测条件下产生了准确的重建结果，包括快速动作如挥动棒球 bat。 

---
# High-fidelity Grain Growth Modeling: Leveraging Deep Learning for Fast Computations 

**Title (ZH)**: 高保真晶粒生长建模：利用深度学习实现快速计算 

**Authors**: Pungponhavoan Tep, Marc Bernacki  

**Link**: [PDF](https://arxiv.org/pdf/2505.05354)  

**Abstract**: Grain growth simulation is crucial for predicting metallic material microstructure evolution during annealing and resulting final mechanical properties, but traditional partial differential equation-based methods are computationally expensive, creating bottlenecks in materials design and manufacturing. In this work, we introduce a machine learning framework that combines a Convolutional Long Short-Term Memory networks with an Autoencoder to efficiently predict grain growth evolution. Our approach captures both spatial and temporal aspects of grain evolution while encoding high-dimensional grain structure data into a compact latent space for pattern learning, enhanced by a novel composite loss function combining Mean Squared Error, Structural Similarity Index Measurement, and Boundary Preservation to maintain structural integrity of grain boundary topology of the prediction. Results demonstrated that our machine learning approach accelerates grain growth prediction by up to \SI{89}{\times} faster, reducing computation time from \SI{10}{\minute} to approximately \SI{10}{\second} while maintaining high-fidelity predictions. The best model (S-30-30) achieving a structural similarity score of \SI{86.71}{\percent} and mean grain size error of just \SI{0.07}{\percent}. All models accurately captured grain boundary topology, morphology, and size distributions. This approach enables rapid microstructural prediction for applications where conventional simulations are prohibitively time-consuming, potentially accelerating innovation in materials science and manufacturing. 

**Abstract (ZH)**: 基于卷积长短期记忆网络与自动编码器的机器学习框架加速晶粒生长演化预测 

---
# Feature-Augmented Deep Networks for Multiscale Building Segmentation in High-Resolution UAV and Satellite Imagery 

**Title (ZH)**: 特征增强深度网络在高分辨率无人机和卫星图像中的多尺度建筑分割 

**Authors**: Chintan B. Maniyar, Minakshi Kumar, Gengchen Mai  

**Link**: [PDF](https://arxiv.org/pdf/2505.05321)  

**Abstract**: Accurate building segmentation from high-resolution RGB imagery remains challenging due to spectral similarity with non-building features, shadows, and irregular building geometries. In this study, we present a comprehensive deep learning framework for multiscale building segmentation using RGB aerial and satellite imagery with spatial resolutions ranging from 0.4m to 2.7m. We curate a diverse, multi-sensor dataset and introduce feature-augmented inputs by deriving secondary representations including Principal Component Analysis (PCA), Visible Difference Vegetation Index (VDVI), Morphological Building Index (MBI), and Sobel edge filters from RGB channels. These features guide a Res-U-Net architecture in learning complex spatial patterns more effectively. We also propose training policies incorporating layer freezing, cyclical learning rates, and SuperConvergence to reduce training time and resource usage. Evaluated on a held-out WorldView-3 image, our model achieves an overall accuracy of 96.5%, an F1-score of 0.86, and an Intersection over Union (IoU) of 0.80, outperforming existing RGB-based benchmarks. This study demonstrates the effectiveness of combining multi-resolution imagery, feature augmentation, and optimized training strategies for robust building segmentation in remote sensing applications. 

**Abstract (ZH)**: 高分辨率RGB影像中多尺度建筑分割的精确建模依然具有挑战性，这归因于其与非建筑特征、阴影和不规则建筑几何形状在光谱上的相似性。本研究提出了一种综合的深度学习框架，利用分辨率从0.4米到2.7米的RGB航空和卫星影像进行多尺度建筑分割。我们构建了一个多样化的多传感器数据集，并通过从RGB通道中衍生出二次表示，包括主成分分析（PCA）、可见差异植被指数（VDVI）、形态学建筑指数（MBI）和Sobel边缘滤波器来增强特征输入。这些特征引导Res-U-Net架构更有效地学习复杂的空间模式。我们还提出了一种结合层冻结、循环学习率和SuperConvergence的训练策略，以减少训练时间和资源消耗。在WorldView-3影像上进行评估，我们的模型总体准确率为96.5%，F1分数为0.86，交集覆蓋率（IoU）为0.80，优于现有基于RGB的基准。本研究展示了将多分辨率影像、特征增强以及优化的训练策略结合应用于遥感应用中进行稳健建筑分割的有效性。 

---
# Mapping User Trust in Vision Language Models: Research Landscape, Challenges, and Prospects 

**Title (ZH)**: 视觉语言模型中用户信任的映射：研究概览、挑战与前景 

**Authors**: Agnese Chiatti, Sara Bernardini, Lara Shibelski Godoy Piccolo, Viola Schiaffonati, Matteo Matteucci  

**Link**: [PDF](https://arxiv.org/pdf/2505.05318)  

**Abstract**: The rapid adoption of Vision Language Models (VLMs), pre-trained on large image-text and video-text datasets, calls for protecting and informing users about when to trust these systems. This survey reviews studies on trust dynamics in user-VLM interactions, through a multi-disciplinary taxonomy encompassing different cognitive science capabilities, collaboration modes, and agent behaviours. Literature insights and findings from a workshop with prospective VLM users inform preliminary requirements for future VLM trust studies. 

**Abstract (ZH)**: 视觉语言模型（VLMs）的快速采纳，这些模型预先在大规模图像-文本和视频-文本数据集上训练，要求保护并告知用户在何时信任这些系统。本文综述了用户-VLM互动中信任动态的研究，通过涵盖不同认知科学能力、合作模式和代理行为的多学科分类体系来进行。文献综述和面向潜在VLM用户的研讨会成果为未来VLM信任研究提供了初步要求。 

---
# Scalable Chain of Thoughts via Elastic Reasoning 

**Title (ZH)**: 弹性推理驱动的可扩展思维链 

**Authors**: Yuhui Xu, Hanze Dong, Lei Wang, Doyen Sahoo, Junnan Li, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.05315)  

**Abstract**: Large reasoning models (LRMs) have achieved remarkable progress on complex tasks by generating extended chains of thought (CoT). However, their uncontrolled output lengths pose significant challenges for real-world deployment, where inference-time budgets on tokens, latency, or compute are strictly constrained. We propose Elastic Reasoning, a novel framework for scalable chain of thoughts that explicitly separates reasoning into two phases--thinking and solution--with independently allocated budgets. At test time, Elastic Reasoning prioritize that completeness of solution segments, significantly improving reliability under tight resource constraints. To train models that are robust to truncated thinking, we introduce a lightweight budget-constrained rollout strategy, integrated into GRPO, which teaches the model to reason adaptively when the thinking process is cut short and generalizes effectively to unseen budget constraints without additional training. Empirical results on mathematical (AIME, MATH500) and programming (LiveCodeBench, Codeforces) benchmarks demonstrate that Elastic Reasoning performs robustly under strict budget constraints, while incurring significantly lower training cost than baseline methods. Remarkably, our approach also produces more concise and efficient reasoning even in unconstrained settings. Elastic Reasoning offers a principled and practical solution to the pressing challenge of controllable reasoning at scale. 

**Abstract (ZH)**: Elastic Reasoning：一种可扩展的可控推理框架 

---
# Benchmarking Ophthalmology Foundation Models for Clinically Significant Age Macular Degeneration Detection 

**Title (ZH)**: 眼科基础模型在临床显著性黄斑变性检测中的基准测试 

**Authors**: Benjamin A. Cohen, Jonathan Fhima, Meishar Meisel, Baskin Meital, Luis Filipe Nakayama, Eran Berkowitz, Joachim A. Behar  

**Link**: [PDF](https://arxiv.org/pdf/2505.05291)  

**Abstract**: Self-supervised learning (SSL) has enabled Vision Transformers (ViTs) to learn robust representations from large-scale natural image datasets, enhancing their generalization across domains. In retinal imaging, foundation models pretrained on either natural or ophthalmic data have shown promise, but the benefits of in-domain pretraining remain uncertain. To investigate this, we benchmark six SSL-pretrained ViTs on seven digital fundus image (DFI) datasets totaling 70,000 expert-annotated images for the task of moderate-to-late age-related macular degeneration (AMD) identification. Our results show that iBOT pretrained on natural images achieves the highest out-of-distribution generalization, with AUROCs of 0.80-0.97, outperforming domain-specific models, which achieved AUROCs of 0.78-0.96 and a baseline ViT-L with no pretraining, which achieved AUROCs of 0.68-0.91. These findings highlight the value of foundation models in improving AMD identification and challenge the assumption that in-domain pretraining is necessary. Furthermore, we release BRAMD, an open-access dataset (n=587) of DFIs with AMD labels from Brazil. 

**Abstract (ZH)**: 自监督学习（SSL）使视觉变换器（ViTs）能够从大规模自然图像数据集中学习 robust 表示，从而增强其跨领域的泛化能力。在视网膜成像中，预训练于自然数据或眼科数据的基础模型显示出前景，但领域内预训练的好处尚不明确。为了探究这一问题，我们对六种自监督预训练的ViTs在七个数字视网膜图像（DFI）数据集上进行了基准测试，这些数据集总计包含70,000张由专家标注的图像，用于中晚期年龄相关黄斑变性（AMD）识别任务。研究结果表明，预训练于自然图像的iBOT取得了最佳的跨领域泛化性能，AUROCs为0.80-0.97，优于专门领域模型（AUROCs为0.78-0.96）及无预训练的基础ViT-L（AUROCs为0.68-0.91）。这些发现突显了基础模型在提高AMD识别方面的价值，并挑战了领域内预训练必要的假设。此外，我们发布了BRAMD，这是一个包含来自巴西587张标注有AMD标签的数字视网膜图像的开源数据集。 

---
# PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes 

**Title (ZH)**: PlaceIt3D：面向真实3D场景的语言引导对象放置 

**Authors**: Ahmed Abdelreheem, Filippo Aleotti, Jamie Watson, Zawar Qureshi, Abdelrahman Eldesokey, Peter Wonka, Gabriel Brostow, Sara Vicente, Guillermo Garcia-Hernando  

**Link**: [PDF](https://arxiv.org/pdf/2505.05288)  

**Abstract**: We introduce the novel task of Language-Guided Object Placement in Real 3D Scenes. Our model is given a 3D scene's point cloud, a 3D asset, and a textual prompt broadly describing where the 3D asset should be placed. The task here is to find a valid placement for the 3D asset that respects the prompt. Compared with other language-guided localization tasks in 3D scenes such as grounding, this task has specific challenges: it is ambiguous because it has multiple valid solutions, and it requires reasoning about 3D geometric relationships and free space. We inaugurate this task by proposing a new benchmark and evaluation protocol. We also introduce a new dataset for training 3D LLMs on this task, as well as the first method to serve as a non-trivial baseline. We believe that this challenging task and our new benchmark could become part of the suite of benchmarks used to evaluate and compare generalist 3D LLM models. 

**Abstract (ZH)**: 语言引导的三维场景对象放置任务 

---
# Software Development Life Cycle Perspective: A Survey of Benchmarks for CodeLLMs and Agents 

**Title (ZH)**: 软件开发生命周期视角：代码LLMs和代理的基准调研 

**Authors**: Kaixin Wang, Tianlin Li, Xiaoyu Zhang, Chong Wang, Weisong Sun, Yang Liu, Bin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.05283)  

**Abstract**: Code large language models (CodeLLMs) and agents have shown great promise in tackling complex software engineering this http URL to traditional software engineering methods, CodeLLMs and agents offer stronger abilities, and can flexibly process inputs and outputs in both natural and code. Benchmarking plays a crucial role in evaluating the capabilities of CodeLLMs and agents, guiding their development and deployment. However, despite their growing significance, there remains a lack of comprehensive reviews of benchmarks for CodeLLMs and agents. To bridge this gap, this paper provides a comprehensive review of existing benchmarks for CodeLLMs and agents, studying and analyzing 181 benchmarks from 461 relevant papers, covering the different phases of the software development life cycle (SDLC). Our findings reveal a notable imbalance in the coverage of current benchmarks, with approximately 60% focused on the software development phase in SDLC, while requirements engineering and software design phases receive minimal attention at only 5% and 3%, respectively. Additionally, Python emerges as the dominant programming language across the reviewed benchmarks. Finally, this paper highlights the challenges of current research and proposes future directions, aiming to narrow the gap between the theoretical capabilities of CodeLLMs and agents and their application in real-world scenarios. 

**Abstract (ZH)**: Code大模型（CodeLLMs）和代理在复杂软件工程中的应用前景：面向传统软件工程方法，CodeLLMs和代理提供了更强的能力，并且能够在自然语言和代码之间灵活处理输入和输出。基准测试在评估CodeLLMs和代理的能力中扮演着至关重要的角色，指导其开发和部署。然而，尽管它们的重要性不断提高，仍缺乏对CodeLLMs和代理基准的全面综述。为了填补这一空白，本论文对现有的CodeLLMs和代理基准进行了全面综述，研究并分析了461篇相关论文中的181个基准，涵盖了软件开发生命周期（SDLC）的不同阶段。我们的研究发现，当前基准的覆盖范围存在明显不平衡，约60%的关注于开发阶段，而需求工程和设计阶段分别仅占5%和3%。此外，Python在所审阅的基准中占据主导地位。最后，本文指出现有研究中的挑战，并提出未来的研究方向，旨在缩小CodeLLMs和代理的理论能力与其在实际应用场景中的应用之间的差距。 

---
# T-T: Table Transformer for Tagging-based Aspect Sentiment Triplet Extraction 

**Title (ZH)**: T-T: 基于标签的aspect情感三元组提取的表格变换器 

**Authors**: Kun Peng, Chaodong Tong, Cong Cao, Hao Peng, Qian Li, Guanlin Wu, Lei Jiang, Yanbing Liu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05271)  

**Abstract**: Aspect sentiment triplet extraction (ASTE) aims to extract triplets composed of aspect terms, opinion terms, and sentiment polarities from given sentences. The table tagging method is a popular approach to addressing this task, which encodes a sentence into a 2-dimensional table, allowing for the tagging of relations between any two words. Previous efforts have focused on designing various downstream relation learning modules to better capture interactions between tokens in the table, revealing that a stronger capability to capture relations can lead to greater improvements in the model. Motivated by this, we attempt to directly utilize transformer layers as downstream relation learning modules. Due to the powerful semantic modeling capability of transformers, it is foreseeable that this will lead to excellent improvement. However, owing to the quadratic relation between the length of the table and the length of the input sentence sequence, using transformers directly faces two challenges: overly long table sequences and unfair local attention interaction. To address these challenges, we propose a novel Table-Transformer (T-T) for the tagging-based ASTE method. Specifically, we introduce a stripe attention mechanism with a loop-shift strategy to tackle these challenges. The former modifies the global attention mechanism to only attend to a 2-dimensional local attention window, while the latter facilitates interaction between different attention windows. Extensive and comprehensive experiments demonstrate that the T-T, as a downstream relation learning module, achieves state-of-the-art performance with lower computational costs. 

**Abstract (ZH)**: 面向表格的aspect情感三元组提取（Table-based Aspect Sentiment Triplet Extraction, ASTE）旨在从给定句子中提取由aspect术语、意见术语和情感极性组成的三元组。表标记方法是解决这一任务的一种流行方法，该方法将句子编码为二维表，允许对任何两个词之间的关系进行标注。先前的努力集中在设计各种下游关系学习模块，以更好地捕捉表中token之间的交互，揭示出捕捉关系的能力越强，模型的改进程度越大。受此启发，我们尝试直接利用transformer层作为下游关系学习模块。由于transformer强大的语义建模能力，这预计将导致显著的改进。然而，由于表的长度与输入句子序列长度之间存在二次关系，直接使用transformer面临两个挑战：过长的表序列和不公的局部注意力交互。为解决这些挑战，我们提出了一个新颖的Table-Transformer（T-T）用于基于表标记的ASTE方法。具体地，我们引入了一种带循环移位策略的条带注意力机制以应对这些挑战。前者将全局注意力机制修改为仅关注二维局部注意力窗口，而后者促进了不同注意力窗口之间的交互。详尽且综合的实验表明，作为下游关系学习模块，T-T实现了最先进的性能，并具有更低的计算成本。 

---
# Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration 

**Title (ZH)**: 基于状态建模和对抗性探索的协作多智能体 reinforcement learning 改进方法 

**Authors**: Andreas Kontogiannis, Konstantinos Papathanasiou, Yi Shen, Giorgos Stamou, Michael M. Zavlanos, George Vouros  

**Link**: [PDF](https://arxiv.org/pdf/2505.05262)  

**Abstract**: Learning to cooperate in distributed partially observable environments with no communication abilities poses significant challenges for multi-agent deep reinforcement learning (MARL). This paper addresses key concerns in this domain, focusing on inferring state representations from individual agent observations and leveraging these representations to enhance agents' exploration and collaborative task execution policies. To this end, we propose a novel state modelling framework for cooperative MARL, where agents infer meaningful belief representations of the non-observable state, with respect to optimizing their own policies, while filtering redundant and less informative joint state information. Building upon this framework, we propose the MARL SMPE algorithm. In SMPE, agents enhance their own policy's discriminative abilities under partial observability, explicitly by incorporating their beliefs into the policy network, and implicitly by adopting an adversarial type of exploration policies which encourages agents to discover novel, high-value states while improving the discriminative abilities of others. Experimentally, we show that SMPE outperforms state-of-the-art MARL algorithms in complex fully cooperative tasks from the MPE, LBF, and RWARE benchmarks. 

**Abstract (ZH)**: 在无通信能力的分布式部分可观测环境中学习合作对多智能体深度强化学习（MARL）提出了重大挑战。本文针对该领域的关键问题，集中于从个体智能体观察中推断状态表示，并利用这些表示增强智能体的探索和协作任务执行策略。为此，我们提出了一种新的合作MARL的状态建模框架，其中智能体推断与优化自身策略相关的有意义的信念表示，同时过滤冗余和低信息量的联合状态信息。在此基础上，我们提出了MARL SMPE算法。在SMPE中，智能体通过将自身的信念融入策略网络中，明确地增强自身策略的鉴别能力，并通过采用一种对抗型的探索策略，隐式地促进自身和他人鉴别能力的提升，同时发现新的、高价值状态。实验结果显示，SMPE在MPE、LBF和RWARE基准中的复杂完全协同任务中优于最先进的MARL算法。 

---
# Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning 

**Title (ZH)**: 把CASH onto bandits：一个自动机器学习的极大K臂问题 

**Authors**: Amir Rezaei Balef, Claire Vernade, Katharina Eggensperger  

**Link**: [PDF](https://arxiv.org/pdf/2505.05226)  

**Abstract**: The Combined Algorithm Selection and Hyperparameter optimization (CASH) is a challenging resource allocation problem in the field of AutoML. We propose MaxUCB, a max $k$-armed bandit method to trade off exploring different model classes and conducting hyperparameter optimization. MaxUCB is specifically designed for the light-tailed and bounded reward distributions arising in this setting and, thus, provides an efficient alternative compared to classic max $k$-armed bandit methods assuming heavy-tailed reward distributions. We theoretically and empirically evaluate our method on four standard AutoML benchmarks, demonstrating superior performance over prior approaches. 

**Abstract (ZH)**: 结合算法选择与超参数优化的MaxUCB方法：一种适用于AutoML领域的高效资源分配策略 

---
# Incentive-Aware Machine Learning; Robustness, Fairness, Improvement & Causality 

**Title (ZH)**: 激励aware机器学习；稳健性、公平性、改进与因果性 

**Authors**: Chara Podimata  

**Link**: [PDF](https://arxiv.org/pdf/2505.05211)  

**Abstract**: The article explores the emerging domain of incentive-aware machine learning (ML), which focuses on algorithmic decision-making in contexts where individuals can strategically modify their inputs to influence outcomes. It categorizes the research into three perspectives: robustness, aiming to design models resilient to "gaming"; fairness, analyzing the societal impacts of such systems; and improvement/causality, recognizing situations where strategic actions lead to genuine personal or societal improvement. The paper introduces a unified framework encapsulating models for these perspectives, including offline, online, and causal settings, and highlights key challenges such as differentiating between gaming and improvement and addressing heterogeneity among agents. By synthesizing findings from diverse works, we outline theoretical advancements and practical solutions for robust, fair, and causally-informed incentive-aware ML systems. 

**Abstract (ZH)**: 本文探讨了激励感知机器学习这一新兴领域，重点关注个体可以通过战略性修改输入来影响结果的环境中的算法决策。文章从三个方面分类研究：鲁棒性，旨在设计能够在“游戏”面前坚固的模型；公平性，分析此类系统的社会影响；以及改进/因果性，认识到战略性行为可能导致真实的个人或社会改进。文章介绍了一个统一框架，涵盖这些视角下的模型，包括离线、在线和因果设置，并强调了识别“游戏”与改进之间的差异和应对代理异质性等关键挑战。通过综合多元研究的发现，本文概述了稳健、公平和因果驱动的激励感知机器学习系统的理论进展和实用解决方案。 

---
# LAPSO: A Unified Optimization View for Learning-Augmented Power System Operations 

**Title (ZH)**: LAPSO：学习增强电力系统运营的统一优化视角 

**Authors**: Wangkun Xu, Zhongda Chu, Fei Teng  

**Link**: [PDF](https://arxiv.org/pdf/2505.05203)  

**Abstract**: With the high penetration of renewables, traditional model-based power system operation is challenged to deliver economic, stable, and robust decisions. Machine learning has emerged as a powerful modeling tool for capturing complex dynamics to address these challenges. However, its separate design often lacks systematic integration with existing methods. To fill the gap, this paper proposes a holistic framework of Learning-Augmented Power System Operations (LAPSO, pronounced as Lap-So). Adopting a native optimization perspective, LAPSO is centered on the operation stage and aims to break the boundary between temporally siloed power system tasks, such as forecast, operation and control, while unifying the objectives of machine learning and model-based optimizations at both training and inference stages. Systematic analysis and simulations demonstrate the effectiveness of applying LAPSO in designing new integrated algorithms, such as stability-constrained optimization (SCO) and objective-based forecasting (OBF), while enabling end-to-end tracing of different sources of uncertainties. In addition, a dedicated Python package-lapso is introduced to automatically augment existing power system optimization models with learnable components. All code and data are available at this https URL. 

**Abstract (ZH)**: 随着可再生能源的高渗透率，传统的基于模型的电力系统运行模式面临着提供经济、稳定和 robust 决策的挑战。机器学习作为一种强大的建模工具，被用于捕捉复杂动力学以应对这些挑战。然而，其单独设计往往缺乏与现有方法的系统集成。为了弥补这一差距，本文提出了一种整体框架，称为学习增强电力系统运行（LAPSO，发音为 Lap-So）。从原生优化视角出发，LAPSO以运行阶段为中心，旨在打破时间上分离的电力系统任务（如预测、运行和控制）之间的边界，同时在训练和推理阶段统一机器学习和基于模型优化的目标。系统分析和仿真表明，LAPSO在设计新的集成算法（如稳定性约束优化（SCO）和目标导向预测（OBF））方面有效，同时允许对不同来源的不确定性进行端到端跟踪。此外，还介绍了一个专门的 Python 包 lapso，用于自动增强现有的电力系统优化模型中的可学习组件。所有代码和数据可在以下网址获得。 

---
# Concept-Based Unsupervised Domain Adaptation 

**Title (ZH)**: 基于概念的无监督领域适应 

**Authors**: Xinyue Xu, Yueying Hu, Hui Tang, Yi Qin, Lu Mi, Hao Wang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.05195)  

**Abstract**: Concept Bottleneck Models (CBMs) enhance interpretability by explaining predictions through human-understandable concepts but typically assume that training and test data share the same distribution. This assumption often fails under domain shifts, leading to degraded performance and poor generalization. To address these limitations and improve the robustness of CBMs, we propose the Concept-based Unsupervised Domain Adaptation (CUDA) framework. CUDA is designed to: (1) align concept representations across domains using adversarial training, (2) introduce a relaxation threshold to allow minor domain-specific differences in concept distributions, thereby preventing performance drop due to over-constraints of these distributions, (3) infer concepts directly in the target domain without requiring labeled concept data, enabling CBMs to adapt to diverse domains, and (4) integrate concept learning into conventional domain adaptation (DA) with theoretical guarantees, improving interpretability and establishing new benchmarks for DA. Experiments demonstrate that our approach significantly outperforms the state-of-the-art CBM and DA methods on real-world datasets. 

**Abstract (ZH)**: 基于概念的无监督领域适应 (CUDA) 框架 

---
# Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks 

**Title (ZH)**: 通过自信息重写攻击揭示文本水印的弱点 

**Authors**: Yixin Cheng, Hongcheng Guo, Yangming Li, Leonid Sigal  

**Link**: [PDF](https://arxiv.org/pdf/2505.05190)  

**Abstract**: Text watermarking aims to subtly embed statistical signals into text by controlling the Large Language Model (LLM)'s sampling process, enabling watermark detectors to verify that the output was generated by the specified model. The robustness of these watermarking algorithms has become a key factor in evaluating their effectiveness. Current text watermarking algorithms embed watermarks in high-entropy tokens to ensure text quality. In this paper, we reveal that this seemingly benign design can be exploited by attackers, posing a significant risk to the robustness of the watermark. We introduce a generic efficient paraphrasing attack, the Self-Information Rewrite Attack (SIRA), which leverages the vulnerability by calculating the self-information of each token to identify potential pattern tokens and perform targeted attack. Our work exposes a widely prevalent vulnerability in current watermarking algorithms. The experimental results show SIRA achieves nearly 100% attack success rates on seven recent watermarking methods with only 0.88 USD per million tokens cost. Our approach does not require any access to the watermark algorithms or the watermarked LLM and can seamlessly transfer to any LLM as the attack model, even mobile-level models. Our findings highlight the urgent need for more robust watermarking. 

**Abstract (ZH)**: 文本水印旨在通过控制大型语言模型（LLM）的采样过程，微妙地将统计信号嵌入文本中，从而使水印检测器能够验证输出是由指定模型生成的。这些水印算法的鲁棒性已成为评估其有效性的重要因素。当前的文本水印算法将水印嵌入高熵令牌以确保文本质量。在本文中，我们揭示这一看似无害的设计可以被攻击者利用，对水印的鲁棒性构成重大威胁。我们提出了一种高效的通用改写攻击方法，即自我信息重写攻击（SIRA），该方法利用漏洞通过计算每个令牌的自我信息来识别潜在的模式令牌并执行有针对性的攻击。我们的工作揭示了当前水印算法中广泛存在的漏洞。实验结果表明，SIRA在仅需每百万令牌0.88美元成本的情况下，对七种最近的方法实现了近乎100%的攻击成功率。我们的方法不需要访问水印算法或水印的LLM，并且可以无缝转移到任何LLM，甚至是移动级别模型作为攻击模型。我们的研究结果突显了对更鲁棒水印方法的迫切需要。 

---
# Biomed-DPT: Dual Modality Prompt Tuning for Biomedical Vision-Language Models 

**Title (ZH)**: Biomed-DPT: 双模态提示调整方法用于生物医学视觉语言模型 

**Authors**: Wei Peng, Kang Liu, Jianchen Hu, Meng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05189)  

**Abstract**: Prompt learning is one of the most effective paradigms for adapting pre-trained vision-language models (VLMs) to the biomedical image classification tasks in few shot scenarios. However, most of the current prompt learning methods only used the text prompts and ignored the particular structures (such as the complex anatomical structures and subtle pathological features) in the biomedical images. In this work, we propose Biomed-DPT, a knowledge-enhanced dual modality prompt tuning technique. In designing the text prompt, Biomed-DPT constructs a dual prompt including the template-driven clinical prompts and the large language model (LLM)-driven domain-adapted prompts, then extracts the clinical knowledge from the domain-adapted prompts through the knowledge distillation technique. In designing the vision prompt, Biomed-DPT introduces the zero vector as a soft prompt to leverage attention re-weighting so that the focus on non-diagnostic regions and the recognition of non-critical pathological features are avoided. Biomed-DPT achieves an average classification accuracy of 66.14\% across 11 biomedical image datasets covering 9 modalities and 10 organs, with performance reaching 78.06\% in base classes and 75.97\% in novel classes, surpassing the Context Optimization (CoOp) method by 6.20\%, 3.78\%, and 8.04\%, respectively. Our code are available at \underline{this https URL}. 

**Abstract (ZH)**: 基于知识增强的双模态提示调谐技术Biomed-DPT 

---
# Stochastic Variational Propagation: Local, Scalable and Efficient Alternative to Backpropagation 

**Title (ZH)**: 随机变分传播：反向传播的局部、可扩展且efficient的替代方法 

**Authors**: Bojian Yin, Federico Corradi  

**Link**: [PDF](https://arxiv.org/pdf/2505.05181)  

**Abstract**: Backpropagation (BP) is the cornerstone of deep learning, but its reliance on global gradient synchronization limits scalability and imposes significant memory overhead. We propose Stochastic Variational Propagation (SVP), a scalable alternative that reframes training as hierarchical variational inference. SVP treats layer activations as latent variables and optimizes local Evidence Lower Bounds (ELBOs), enabling independent, local updates while preserving global coherence. However, directly applying KL divergence in layer-wise ELBOs risks inter-layer's representation collapse due to excessive compression. To prevent this, SVP projects activations into low-dimensional spaces via fixed random matrices, ensuring information preservation and representational diversity. Combined with a feature alignment loss for inter-layer consistency, SVP achieves competitive accuracy with BP across diverse architectures (MLPs, CNNs, Transformers) and datasets (MNIST to ImageNet), reduces memory usage by up to 4x, and significantly improves scalability. More broadly, SVP introduces a probabilistic perspective to deep representation learning, opening pathways toward more modular and interpretable neural network design. 

**Abstract (ZH)**: Stochastic Variational Propagation: A Scalable Alternative for Deep Learning 

---
# Dukawalla: Voice Interfaces for Small Businesses in Africa 

**Title (ZH)**: Dukawalla: 非洲地区小型企业的声音界面 

**Authors**: Elizabeth Ankrah, Stephanie Nyairo, Mercy Muchai, Kagonya Awori, Millicent Ochieng, Mark Kariuki, Jacki O'Neill  

**Link**: [PDF](https://arxiv.org/pdf/2505.05170)  

**Abstract**: Small and medium sized businesses often struggle with data driven decision making do to a lack of advanced analytics tools, especially in African countries where they make up a majority of the workforce. Though many tools exist they are not designed to fit into the ways of working of SMB workers who are mobile first, have limited time to learn new workflows, and for whom social and business are tightly coupled. To address this, the Dukawalla prototype was created. This intelligent assistant bridges the gap between raw business data, and actionable insights by leveraging voice interaction and the power of generative AI. Dukawalla provides an intuitive way for business owners to interact with their data, aiding in informed decision making. This paper examines Dukawalla's deployment across SMBs in Nairobi, focusing on their experiences using this voice based assistant to streamline data collection and provide business insights 

**Abstract (ZH)**: 小型和中型企业常常因缺乏高级数据分析工具而难以进行数据驱动的决策，特别是在非洲国家，中小企业占据了大多数劳动力。虽然许多工具已经存在，但它们并未设计成适合移动优先的中小企业员工的工作方式，这些员工没有太多时间学习新的工作流程，而且他们的社交和商业活动紧密相连。为解决这个问题，我们创建了Dukawalla原型。这个智能助手通过利用语音交互和生成式AI的力量，在原始业务数据和可操作的洞察之间架起桥梁。Dukawalla为中小企业主提供了一种直观的数据交互方式，有助于做出明智的决策。本文探讨了Dukawalla在 Nairobi 的中小企业中的部署情况，重点关注这些企业如何使用基于语音的助手来简化数据收集并提供商业见解。 

---
# Understanding In-context Learning of Addition via Activation Subspaces 

**Title (ZH)**: 理解通过激活子空间进行的加法上下文学习 

**Authors**: Xinyan Hu, Kayo Yin, Michael I. Jordan, Jacob Steinhardt, Lijie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.05145)  

**Abstract**: To perform in-context learning, language models must extract signals from individual few-shot examples, aggregate these into a learned prediction rule, and then apply this rule to new examples. How is this implemented in the forward pass of modern transformer models? To study this, we consider a structured family of few-shot learning tasks for which the true prediction rule is to add an integer $k$ to the input. We find that Llama-3-8B attains high accuracy on this task for a range of $k$, and localize its few-shot ability to just three attention heads via a novel optimization approach. We further show the extracted signals lie in a six-dimensional subspace, where four of the dimensions track the unit digit and the other two dimensions track overall magnitude. We finally examine how these heads extract information from individual few-shot examples, identifying a self-correction mechanism in which mistakes from earlier examples are suppressed by later examples. Our results demonstrate how tracking low-dimensional subspaces across a forward pass can provide insight into fine-grained computational structures. 

**Abstract (ZH)**: 语言模型为了进行上下文学习，必须从个别少样本示例中提取信号，将这些信号整合成一个学习中的预测规则，然后将此规则应用于新示例。这种机制是如何在现代变压器模型的前向传播过程中实现的？为了研究这一点，我们考虑了一类结构化的少样本学习任务，其中真实的预测规则是将整数 \(k\) 加到输入中。我们发现 Llama-3-8B 在 \(k\) 的多种情况上都实现了高精度，并通过一种新型优化方法将其实现的少样本能力局部化到仅三个注意力头中。我们进一步展示了提取的信号位于一个六维子空间中，其中四个维度跟踪个位数，另外两个维度跟踪整体大小。最后，我们检查了这些头如何从个别少样本示例中提取信息，识别出一种自我修正机制，即早期示例中的错误被后来的示例抑制。我们的结果展示了在整个前向传播过程中跟踪低维子空间如何提供对精细计算结构的洞察。 

---
# Guiding Evolutionary AutoEncoder Training with Activation-Based Pruning Operators 

**Title (ZH)**: 基于激活值的剪枝操作引导自动编码器进化训练 

**Authors**: Steven Jorgensen, Erik Hemberg, Jamal Toutouh, Una-May O'Reilly  

**Link**: [PDF](https://arxiv.org/pdf/2505.05138)  

**Abstract**: This study explores a novel approach to neural network pruning using evolutionary computation, focusing on simultaneously pruning the encoder and decoder of an autoencoder. We introduce two new mutation operators that use layer activations to guide weight pruning. Our findings reveal that one of these activation-informed operators outperforms random pruning, resulting in more efficient autoencoders with comparable performance to canonically trained models. Prior work has established that autoencoder training is effective and scalable with a spatial coevolutionary algorithm that cooperatively coevolves a population of encoders with a population of decoders, rather than one autoencoder. We evaluate how the same activity-guided mutation operators transfer to this context. We find that random pruning is better than guided pruning, in the coevolutionary setting. This suggests activation-based guidance proves more effective in low-dimensional pruning environments, where constrained sample spaces can lead to deviations from true uniformity in randomization. Conversely, population-driven strategies enhance robustness by expanding the total pruning dimensionality, achieving statistically uniform randomness that better preserves system dynamics. We experiment with pruning according to different schedules and present best combinations of operator and schedule for the canonical and coevolving populations cases. 

**Abstract (ZH)**: 该研究探讨了一种使用进化计算进行神经网络剪枝的新型方法，重点关注同时剪枝自编码器的编码器和解码器。我们引入了两种新的变异算子，利用层激活来指导权重剪枝。我们的研究发现，其中一种基于激活的算子优于随机剪枝，从而实现了更具效率的自编码器，并且在性能上与传统训练模型相当。之前的研究表明，使用时空协作演化算法可以有效且可扩展地训练自编码器，该算法协同演化编码器和解码器种群，而非单个自编码器。我们评估了相同活动引导的变异算子在这种上下文中的应用。我们发现，在协同演化设置中，随机剪枝优于引导剪枝，这表明基于激活的引导在低维度剪枝环境中更为有效，受限的样本空间可能导致随机化偏离真正的均匀性。相反，种群驱动的策略通过扩展总剪枝维度来增强鲁棒性，实现统计上均匀的随机性，从而更好地保持系统动力学。我们根据不同的剪枝时间表进行实验，并展示了在标准和协同演化种群情况下最优的算子和时间表组合。 

---
# Beyond Low-rank Decomposition: A Shortcut Approach for Efficient On-Device Learning 

**Title (ZH)**: 超越低秩分解：一种高效的本地设备学习快捷方法 

**Authors**: Le-Trung Nguyen, Ael Quelennec, Van-Tam Nguyen, Enzo Tartaglione  

**Link**: [PDF](https://arxiv.org/pdf/2505.05086)  

**Abstract**: On-device learning has emerged as a promising direction for AI development, particularly because of its potential to reduce latency issues and mitigate privacy risks associated with device-server communication, while improving energy efficiency. Despite these advantages, significant memory and computational constraints still represent major challenges for its deployment. Drawing on previous studies on low-rank decomposition methods that address activation memory bottlenecks in backpropagation, we propose a novel shortcut approach as an alternative. Our analysis and experiments demonstrate that our method can reduce activation memory usage, even up to $120.09\times$ compared to vanilla training, while also reducing overall training FLOPs up to $1.86\times$ when evaluated on traditional benchmarks. 

**Abstract (ZH)**: 设备端学习已成为AI发展的一个有前途的方向，特别是在减少设备-服务器通信带来的延迟问题和隐私风险、提升能源效率方面展现出潜力。尽管存在这些优势，内存和计算能力的限制仍然是其广泛应用的主要挑战。基于以往针对反向传播中激活记忆瓶颈的低秩分解方法研究，我们提出了一种新颖的捷径方法作为替代方案。我们的分析和实验表明，该方法可以减少激活记忆使用量，最多可达120.09倍，与标准训练相比，同时在传统基准上将整体训练FLOPs降低至最多1.86倍。 

---
# FG-CLIP: Fine-Grained Visual and Textual Alignment 

**Title (ZH)**: FG-CLIP: 细粒度视觉和文本对齐 

**Authors**: Chunyu Xie, Bin Wang, Fanjing Kong, Jincheng Li, Dawei Liang, Gengshen Zhang, Dawei Leng, Yuhui Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05071)  

**Abstract**: Contrastive Language-Image Pre-training (CLIP) excels in multimodal tasks such as image-text retrieval and zero-shot classification but struggles with fine-grained understanding due to its focus on coarse-grained short captions. To address this, we propose Fine-Grained CLIP (FG-CLIP), which enhances fine-grained understanding through three key innovations. First, we leverage large multimodal models to generate 1.6 billion long caption-image pairs for capturing global-level semantic details. Second, a high-quality dataset is constructed with 12 million images and 40 million region-specific bounding boxes aligned with detailed captions to ensure precise, context-rich representations. Third, 10 million hard fine-grained negative samples are incorporated to improve the model's ability to distinguish subtle semantic differences. Corresponding training methods are meticulously designed for these data. Extensive experiments demonstrate that FG-CLIP outperforms the original CLIP and other state-of-the-art methods across various downstream tasks, including fine-grained understanding, open-vocabulary object detection, image-text retrieval, and general multimodal benchmarks. These results highlight FG-CLIP's effectiveness in capturing fine-grained image details and improving overall model performance. The related data, code, and models are available at this https URL. 

**Abstract (ZH)**: 细粒度CLIP（FG-CLIP）：通过三种关键创新增强细粒度理解 

---
# Teochew-Wild: The First In-the-wild Teochew Dataset with Orthographic Annotations 

**Title (ZH)**: 潮州野生：首个带有音节注释的潮州方言野外数据集 

**Authors**: Linrong Pan, Chenglong Jiang, Gaoze Hou, Ying Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.05056)  

**Abstract**: This paper reports the construction of the Teochew-Wild, a speech corpus of the Teochew dialect. The corpus includes 18.9 hours of in-the-wild Teochew speech data from multiple speakers, covering both formal and colloquial expressions, with precise orthographic and pinyin annotations. Additionally, we provide supplementary text processing tools and resources to propel research and applications in speech tasks for this low-resource language, such as automatic speech recognition (ASR) and text-to-speech (TTS). To the best of our knowledge, this is the first publicly available Teochew dataset with accurate orthographic annotations. We conduct experiments on the corpus, and the results validate its effectiveness in ASR and TTS tasks. 

**Abstract (ZH)**: 本论文报告了潮州野生语料库的构建，该语料库包含来自多位讲者的18.9小时的潮州方言野生语音数据，涵盖正式和非正式表达，并附有精确的拼写和拼音注释。此外，我们还提供了辅助文本处理工具和资源，以促进对这种低资源语言的语音任务研究和应用，如自动语音识别（ASR）和文本到语音（TTS）。据我们所知，这是首个包含准确拼写注释的公开潮州方言数据集。我们在语料库上进行了实验，结果验证了其在ASR和TTS任务中的有效性。 

---
# Direct Image Classification from Fourier Ptychographic Microscopy Measurements without Reconstruction 

**Title (ZH)**: 直接从傅里叶 Ptychographic 显微镜测量中进行图像分类无需重建。 

**Authors**: Navya Sonal Agarwal, Jan Philipp Schneider, Kanchana Vaishnavi Gandikota, Syed Muhammad Kazim, John Meshreki, Ivo Ihrke, Michael Moeller  

**Link**: [PDF](https://arxiv.org/pdf/2505.05054)  

**Abstract**: The computational imaging technique of Fourier Ptychographic Microscopy (FPM) enables high-resolution imaging with a wide field of view and can serve as an extremely valuable tool, e.g. in the classification of cells in medical applications. However, reconstructing a high-resolution image from tens or even hundreds of measurements is computationally expensive, particularly for a wide field of view. Therefore, in this paper, we investigate the idea of classifying the image content in the FPM measurements directly without performing a reconstruction step first. We show that Convolutional Neural Networks (CNN) can extract meaningful information from measurement sequences, significantly outperforming the classification on a single band-limited image (up to 12 %) while being significantly more efficient than a reconstruction of a high-resolution image. Furthermore, we demonstrate that a learned multiplexing of several raw measurements allows maintaining the classification accuracy while reducing the amount of data (and consequently also the acquisition time) significantly. 

**Abstract (ZH)**: Fourier Ptychographic Microscopy中的图像内容直接分类：无需重建的高效分类方法 

---
# Image-Text Relation Prediction for Multilingual Tweets 

**Title (ZH)**: 多语言推文中图像-文本关系预测 

**Authors**: Matīss Rikters, Edison Marrese-Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2505.05040)  

**Abstract**: Various social networks have been allowing media uploads for over a decade now. Still, it has not always been clear what is their relation with the posted text or even if there is any at all. In this work, we explore how multilingual vision-language models tackle the task of image-text relation prediction in different languages, and construct a dedicated balanced benchmark data set from Twitter posts in Latvian along with their manual translations into English. We compare our results to previous work and show that the more recently released vision-language model checkpoints are becoming increasingly capable at this task, but there is still much room for further improvement. 

**Abstract (ZH)**: 各种社交网络允许用户上传媒体已有十多年的时间，但这些媒体与发布的文本之间的关系尚不明确，甚至有时并不存在任何关系。在这项工作中，我们探究多语言视觉-语言模型在不同语言下如何处理图像-文本关系预测任务，并从拉脱维亚语的Twitter帖子及其手工翻译成英文的数据集中构建了一个专门的平衡基准数据集。我们将我们的结果与以往的工作进行比较，并表明最近发布的视觉-语言模型预训练检查点在这一任务上的能力不断提升，但仍有很多改进的空间。 

---
# Generating Reliable Synthetic Clinical Trial Data: The Role of Hyperparameter Optimization and Domain Constraints 

**Title (ZH)**: 生成可靠的合成临床试验数据：超参数优化和领域约束的作用 

**Authors**: Waldemar Hahn, Jan-Niklas Eckardt, Christoph Röllig, Martin Sedlmayr, Jan Moritz Middeke, Markus Wolfien  

**Link**: [PDF](https://arxiv.org/pdf/2505.05019)  

**Abstract**: The generation of synthetic clinical trial data offers a promising approach to mitigating privacy concerns and data accessibility limitations in medical research. However, ensuring that synthetic datasets maintain high fidelity, utility, and adherence to domain-specific constraints remains a key challenge. While hyperparameter optimization (HPO) has been shown to improve generative model performance, the effectiveness of different optimization strategies for synthetic clinical data remains unclear. This study systematically evaluates four HPO strategies across eight generative models, comparing single-metric optimization against compound metric optimization approaches. Our results demonstrate that HPO consistently improves synthetic data quality, with TVAE, CTGAN, and CTAB-GAN+ achieving improvements of up to 60%, 39%, and 38%, respectively. Compound metric optimization outperformed single-metric strategies, producing more balanced and generalizable synthetic datasets. Interestingly, HPO alone is insufficient to ensure clinically valid synthetic data, as all models exhibited violations of fundamental survival constraints. Preprocessing and postprocessing played a crucial role in reducing these violations, as models lacking robust processing steps produced invalid data in up to 61% of cases. These findings underscore the necessity of integrating explicit domain knowledge alongside HPO to create high quality synthetic datasets. Our study provides actionable recommendations for improving synthetic data generation, with future research needed to refine metric selection and validate these findings on larger datasets to enhance clinical applicability. 

**Abstract (ZH)**: 合成临床试验数据的生成为缓解医疗研究中的隐私担忧和数据可访问性限制提供了有前景的方法。然而，确保合成数据集保持高保真度、实用性和遵循领域特定约束仍然是一个关键挑战。虽然超参数优化（HPO）已被证明可以提高生成模型的性能，但不同优化策略对合成临床数据的有效性尚不清楚。本研究系统地评估了四种HPO策略在八种生成模型中的应用，比较了单指标优化与复合指标优化方法。研究结果表明，HPO一致地提高了合成数据的质量，其中TVAE、CTGAN和CTAB-GAN+分别实现了多达60%、39%和38%的改进。复合指标优化优于单指标策略，产生了更平衡和更具通用性的合成数据集。有趣的是，仅HPO不足以确保临床有效的合成数据，所有模型均表现出违反基本生存约束的现象。预处理和后处理在减少这些违反现象方面发挥了关键作用，缺乏稳健处理步骤的模型在多达61%的情况下产生了无效数据。这些发现强调了在HPO过程中结合显式领域知识的必要性，以生成高质量的合成数据集。本研究提供了改进合成数据生成的可操作建议，未来研究需要进一步细化指标选择并在更大规模的数据集上验证这些发现，以增强临床应用性。 

---
# An Agent-Based Modeling Approach to Free-Text Keyboard Dynamics for Continuous Authentication 

**Title (ZH)**: 基于代理模型的自由文本键盘动态建模连续认证方法 

**Authors**: Roberto Dillon, Arushi  

**Link**: [PDF](https://arxiv.org/pdf/2505.05015)  

**Abstract**: Continuous authentication systems leveraging free-text keyboard dynamics offer a promising additional layer of security in a multifactor authentication setup that can be used in a transparent way with no impact on user experience. This study investigates the efficacy of behavioral biometrics by employing an Agent-Based Model (ABM) to simulate diverse typing profiles across mechanical and membrane keyboards. Specifically, we generated synthetic keystroke data from five unique agents, capturing features related to dwell time, flight time, and error rates within sliding 5-second windows updated every second. Two machine learning approaches, One-Class Support Vector Machine (OC-SVM) and Random Forest (RF), were evaluated for user verification. Results revealed a stark contrast in performance: while One-Class SVM failed to differentiate individual users within each group, Random Forest achieved robust intra-keyboard user recognition (Accuracy > 0.7) but struggled to generalize across keyboards for the same user, highlighting the significant impact of keyboard hardware on typing behavior. These findings suggest that: (1) keyboard-specific user profiles may be necessary for reliable authentication, and (2) ensemble methods like RF outperform One-Class SVM in capturing fine-grained user-specific patterns. 

**Abstract (ZH)**: 基于自由文本键盘动态的持续认证系统在多重因素认证设置中提供了一种增强的安全层，可以在不影响用户体验的情况下以透明方式使用。本研究通过运用基于代理的模型（ABM）模拟机械键盘和薄膜键盘的多样化打字模式，探讨行为生物特征的有效性。具体地，我们从五个独特的代理生成合成按键数据，在每秒更新一次的滑动5秒窗口内捕捉与停留时间、飞行时间和错误率相关的特点。评估了Two机器学习方法，即One-Class支持向量机（OC-SVM）和随机森林（RF），用于用户验证。结果表明，One-Class SVM在区分同一组内的个体用户方面表现不佳，而随机森林实现了稳健的键盘内用户识别（准确率＞0.7），但在同一用户跨键盘上的泛化方面遇到困难，突显了键盘硬件对打字行为的显著影响。这些发现表明：（1）特定键盘的用户配置文件可能是可靠认证所必需的，（2）集成方法如随机森林在捕捉细粒度的用户特定模式方面优于One-Class SVM。 

---
# StabStitch++: Unsupervised Online Video Stitching with Spatiotemporal Bidirectional Warps 

**Title (ZH)**: StabStitch++: 基于时空双向变换的无监督在线视频拼接 

**Authors**: Lang Nie, Chunyu Lin, Kang Liao, Yun Zhang, Shuaicheng Liu, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.05001)  

**Abstract**: We retarget video stitching to an emerging issue, named warping shake, which unveils the temporal content shakes induced by sequentially unsmooth warps when extending image stitching to video stitching. Even if the input videos are stable, the stitched video can inevitably cause undesired warping shakes and affect the visual experience. To address this issue, we propose StabStitch++, a novel video stitching framework to realize spatial stitching and temporal stabilization with unsupervised learning simultaneously. First, different from existing learning-based image stitching solutions that typically warp one image to align with another, we suppose a virtual midplane between original image planes and project them onto it. Concretely, we design a differentiable bidirectional decomposition module to disentangle the homography transformation and incorporate it into our spatial warp, evenly spreading alignment burdens and projective distortions across two views. Then, inspired by camera paths in video stabilization, we derive the mathematical expression of stitching trajectories in video stitching by elaborately integrating spatial and temporal warps. Finally, a warp smoothing model is presented to produce stable stitched videos with a hybrid loss to simultaneously encourage content alignment, trajectory smoothness, and online collaboration. Compared with StabStitch that sacrifices alignment for stabilization, StabStitch++ makes no compromise and optimizes both of them simultaneously, especially in the online mode. To establish an evaluation benchmark and train the learning framework, we build a video stitching dataset with a rich diversity in camera motions and scenes. Experiments exhibit that StabStitch++ surpasses current solutions in stitching performance, robustness, and efficiency, offering compelling advancements in this field by building a real-time online video stitching system. 

**Abstract (ZH)**: 面向 warp shake 问题的视频拼接研究：一种无监督学习驱动的空间拼接与 temporal 稳定同步框架 

---
# Rethinking Invariance in In-context Learning 

**Title (ZH)**: 重思上下文学习中的不变性 

**Authors**: Lizhe Fang, Yifei Wang, Khashayar Gatmiry, Lei Fang, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04994)  

**Abstract**: In-Context Learning (ICL) has emerged as a pivotal capability of auto-regressive large language models, yet it is hindered by a notable sensitivity to the ordering of context examples regardless of their mutual independence. To address this issue, recent studies have introduced several variant algorithms of ICL that achieve permutation invariance. However, many of these do not exhibit comparable performance with the standard auto-regressive ICL algorithm. In this work, we identify two crucial elements in the design of an invariant ICL algorithm: information non-leakage and context interdependence, which are not simultaneously achieved by any of the existing methods. These investigations lead us to the proposed Invariant ICL (InvICL), a methodology designed to achieve invariance in ICL while ensuring the two properties. Empirically, our findings reveal that InvICL surpasses previous models, both invariant and non-invariant, in most benchmark datasets, showcasing superior generalization capabilities across varying input lengths. Code is available at this https URL. 

**Abstract (ZH)**: 基于上下文学习（ICL）已成为自回归大型语言模型的关键能力，但其对上下文示例排序的敏感性限制了其发展，尤其是示例之间并非相互独立时。为解决这一问题，近期研究引入了几种ICI的变体算法以实现排列不变性，但许多变体算法的性能不及标准的自回归ICI算法。本文识别出设计一个不变的ICL算法的两个关键要素：信息不泄露和上下文互依赖性，这两个要素至今未由任何现有方法同时实现。这些研究推动我们提出了不变ICL（InvICL），一种旨在实现ICL不变性同时确保上述两个属性的方法。我们的实验发现表明，InvICL在大多数基准数据集中超越了之前的所有模型，展现出更强的泛化能力，覆盖不同输入长度。代码可在以下链接获取：this https URL。 

---
# Decomposition of Probabilities of Causation with Two Mediators 

**Title (ZH)**: 两中介变量下因果概率的分解 

**Authors**: Yuta Kawakami, Jin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.04983)  

**Abstract**: Mediation analysis for probabilities of causation (PoC) provides a fundamental framework for evaluating the necessity and sufficiency of treatment in provoking an event through different causal pathways. One of the primary objectives of causal mediation analysis is to decompose the total effect into path-specific components. In this study, we investigate the path-specific probability of necessity and sufficiency (PNS) to decompose the total PNS into path-specific components along distinct causal pathways between treatment and outcome, incorporating two mediators. We define the path-specific PNS for decomposition and provide an identification theorem. Furthermore, we conduct numerical experiments to assess the properties of the proposed estimators from finite samples and demonstrate their practical application using a real-world educational dataset. 

**Abstract (ZH)**: 概率介导分析中的概率成因（PoC）的中介分析提供了一种基本框架，用于评估不同因果路径中治疗诱发事件的必要性和充分性。因果中介分析的主要目标之一是将总效应分解为特定路径的组件。在本研究中，我们探讨特定路径的概率必要性和充分性（PNS），将其总PNS分解为治疗与结果之间不同因果路径的特定路径组件，同时考虑两个中介变量。我们定义了用于分解的特定路径PNS，并提供了识别定理。此外，我们通过数值实验评估了所提估计量在有限样本中的性质，并使用实际教育数据集展示了其实际应用。 

---
# ChainMarks: Securing DNN Watermark with Cryptographic Chain 

**Title (ZH)**: ChainMarks: 用加密链保护DNN水印 

**Authors**: Brian Choi, Shu Wang, Isabelle Choi, Kun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.04977)  

**Abstract**: With the widespread deployment of deep neural network (DNN) models, dynamic watermarking techniques are being used to protect the intellectual property of model owners. However, recent studies have shown that existing watermarking schemes are vulnerable to watermark removal and ambiguity attacks. Besides, the vague criteria for determining watermark presence further increase the likelihood of such attacks. In this paper, we propose a secure DNN watermarking scheme named ChainMarks, which generates secure and robust watermarks by introducing a cryptographic chain into the trigger inputs and utilizes a two-phase Monte Carlo method for determining watermark presence. First, ChainMarks generates trigger inputs as a watermark dataset by repeatedly applying a hash function over a secret key, where the target labels associated with trigger inputs are generated from the digital signature of model owner. Then, the watermarked model is produced by training a DNN over both the original and watermark datasets. To verify watermarks, we compare the predicted labels of trigger inputs with the target labels and determine ownership with a more accurate decision threshold that considers the classification probability of specific models. Experimental results show that ChainMarks exhibits higher levels of robustness and security compared to state-of-the-art watermarking schemes. With a better marginal utility, ChainMarks provides a higher probability guarantee of watermark presence in DNN models with the same level of watermark accuracy. 

**Abstract (ZH)**: ChainMarks：一种基于加密链的深度神经网络水印方案 

---
# AI and Vision based Autonomous Navigation of Nano-Drones in Partially-Known Environments 

**Title (ZH)**: 基于视觉的纳航器在部分未知环境中的自主导航技术 

**Authors**: Mattia Sartori, Chetna Singhal, Neelabhro Roy, Davide Brunelli, James Gross  

**Link**: [PDF](https://arxiv.org/pdf/2505.04972)  

**Abstract**: The miniaturisation of sensors and processors, the advancements in connected edge intelligence, and the exponential interest in Artificial Intelligence are boosting the affirmation of autonomous nano-size drones in the Internet of Robotic Things ecosystem. However, achieving safe autonomous navigation and high-level tasks such as exploration and surveillance with these tiny platforms is extremely challenging due to their limited resources. This work focuses on enabling the safe and autonomous flight of a pocket-size, 30-gram platform called Crazyflie 2.1 in a partially known environment. We propose a novel AI-aided, vision-based reactive planning method for obstacle avoidance under the ambit of Integrated Sensing, Computing and Communication paradigm. We deal with the constraints of the nano-drone by splitting the navigation task into two parts: a deep learning-based object detector runs on the edge (external hardware) while the planning algorithm is executed onboard. The results show the ability to command the drone at $\sim8$ frames-per-second and a model performance reaching a COCO mean-average-precision of $60.8$. Field experiments demonstrate the feasibility of the solution with the drone flying at a top speed of $1$ m/s while steering away from an obstacle placed in an unknown position and reaching the target destination. The outcome highlights the compatibility of the communication delay and the model performance with the requirements of the real-time navigation task. We provide a feasible alternative to a fully onboard implementation that can be extended to autonomous exploration with nano-drones. 

**Abstract (ZH)**: 纳米传感器与处理器的微型化、边缘智能的进步以及人工智能的指数级兴趣推动了自主纳米无人机在物联网机器人生态系统中的确认。然而，由于这些小型平台资源有限，要在部分已知环境中实现安全自主导航和探索、监控等高级任务极具挑战性。本文旨在使一款30克级、名为Crazyflie 2.1的口袋大小无人机在部分已知环境中实现安全自主飞行。我们提出了一种基于集成感知、计算和通信范式的新型人工智能辅助、视觉导向的反应规划方法，用于障碍物避让。我们通过将导航任务分为两部分来处理纳米无人机的约束条件：深度学习对象检测器在边缘（外部硬件）上运行，而规划算法则在机载上执行。实验结果表明，该方法能够在约8帧每秒的速度下控制无人机，并达到COCO平均准确率为60.8的模型性能。实地试验表明，无人机能够在最大速度为1米/秒的情况下避开未知位置的障碍物并到达目标。结果 Highlights 了通信延迟和模型性能与实时导航任务需求的兼容性。本文提供了可扩展到纳米无人机自主探索的全机载实现的可行替代方案。 

---
# Moments of Causal Effects 

**Title (ZH)**: 因果效应的矩 

**Authors**: Yuta Kawakami, Jin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.04971)  

**Abstract**: The moments of random variables are fundamental statistical measures for characterizing the shape of a probability distribution, encompassing metrics such as mean, variance, skewness, and kurtosis. Additionally, the product moments, including covariance and correlation, reveal the relationships between multiple random variables. On the other hand, the primary focus of causal inference is the evaluation of causal effects, which are defined as the difference between two potential outcomes. While traditional causal effect assessment focuses on the average causal effect, this work provides definitions, identification theorems, and bounds for moments and product moments of causal effects to analyze their distribution and relationships. We conduct experiments to illustrate the estimation of the moments of causal effects from finite samples and demonstrate their practical application using a real-world medical dataset. 

**Abstract (ZH)**: 随机变量的矩是描述概率分布形状的基本统计量，包括均值、方差、偏度和峰度等指标。此外，包含协方差和相关系数的乘积矩揭示了多个随机变量之间的关系。另一方面，因果推断的主要关注点是评估因果效应，这些效应定义为两个潜在结果之间的差异。虽然传统的因果效应评估主要关注平均因果效应，但本工作提供了因果效应矩和乘积矩的定义、识别定理及其界，以便分析其分布和关系。我们通过实验说明如何从有限样本中估计因果效应的矩，并通过实际医疗数据集演示其实际应用。 

---
# ADD: Physics-Based Motion Imitation with Adversarial Differential Discriminators 

**Title (ZH)**: 基于物理的运动模仿与对抗微分判别器 

**Authors**: Ziyu Zhang, Sergey Bashkirov, Dun Yang, Michael Taylor, Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04961)  

**Abstract**: Multi-objective optimization problems, which require the simultaneous optimization of multiple terms, are prevalent across numerous applications. Existing multi-objective optimization methods often rely on manually tuned aggregation functions to formulate a joint optimization target. The performance of such hand-tuned methods is heavily dependent on careful weight selection, a time-consuming and laborious process. These limitations also arise in the setting of reinforcement-learning-based motion tracking for physically simulated characters, where intricately crafted reward functions are typically used to achieve high-fidelity results. Such solutions not only require domain expertise and significant manual adjustment, but also limit the applicability of the resulting reward function across diverse skills. To bridge this gap, we present a novel adversarial multi-objective optimization technique that is broadly applicable to a range of multi-objective optimization problems, including motion tracking. The proposed adversarial differential discriminator receives a single positive sample, yet is still effective at guiding the optimization process. We demonstrate that our technique can enable characters to closely replicate a variety of acrobatic and agile behaviors, achieving comparable quality to state-of-the-art motion-tracking methods, without relying on manually tuned reward functions. Results are best visualized through this https URL. 

**Abstract (ZH)**: 多目标优化问题要求同时优化多个目标，在众多应用中普遍存在。现有的多目标优化方法通常依赖手动调参的聚合函数来制定联合优化目标。这类手动调参的方法其性能高度依赖于权重的选择，这是一个耗时且劳动密集的过程。这些限制也同样存在于基于强化学习的动作跟踪中，通常需要精心设计奖励函数来实现高保真效果。这种解决方案不仅需要领域专业知识和大量的手动调整，还限制了所得奖励函数在不同技能中的应用。为了解决这个问题，我们提出了一种新型的对抗多目标优化技术，该技术广泛适用于多种多目标优化问题，包括动作跟踪。提出的对抗差分鉴别器仅接收一个正样本，但仍能有效地指导优化过程。我们的技术能够使角色精确再现各种杂技和敏捷行为，达到与当前最优动作跟踪方法相当的品质，而不依赖手动调参的奖励函数。结果可通过此链接可视化：https://www.example.com results。 

---
# Graffe: Graph Representation Learning via Diffusion Probabilistic Models 

**Title (ZH)**: Graffe: 图表示学习 via 扩散概率模型 

**Authors**: Dingshuo Chen, Shuchen Xue, Liuji Chen, Yingheng Wang, Qiang Liu, Shu Wu, Zhi-Ming Ma, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04956)  

**Abstract**: Diffusion probabilistic models (DPMs), widely recognized for their potential to generate high-quality samples, tend to go unnoticed in representation learning. While recent progress has highlighted their potential for capturing visual semantics, adapting DPMs to graph representation learning remains in its infancy. In this paper, we introduce Graffe, a self-supervised diffusion model proposed for graph representation learning. It features a graph encoder that distills a source graph into a compact representation, which, in turn, serves as the condition to guide the denoising process of the diffusion decoder. To evaluate the effectiveness of our model, we first explore the theoretical foundations of applying diffusion models to representation learning, proving that the denoising objective implicitly maximizes the conditional mutual information between data and its representation. Specifically, we prove that the negative logarithm of the denoising score matching loss is a tractable lower bound for the conditional mutual information. Empirically, we conduct a series of case studies to validate our theoretical insights. In addition, Graffe delivers competitive results under the linear probing setting on node and graph classification tasks, achieving state-of-the-art performance on 9 of the 11 real-world datasets. These findings indicate that powerful generative models, especially diffusion models, serve as an effective tool for graph representation learning. 

**Abstract (ZH)**: 自监督扩散模型Grafite及其在图表示学习中的应用 

---
# Chain-of-Thought Tokens are Computer Program Variables 

**Title (ZH)**: Chain-of-Thought Tokens是计算机程序变量 

**Authors**: Fangwei Zhu, Peiyi Wang, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2505.04955)  

**Abstract**: Chain-of-thoughts (CoT) requires large language models (LLMs) to generate intermediate steps before reaching the final answer, and has been proven effective to help LLMs solve complex reasoning tasks. However, the inner mechanism of CoT still remains largely unclear. In this paper, we empirically study the role of CoT tokens in LLMs on two compositional tasks: multi-digit multiplication and dynamic programming. While CoT is essential for solving these problems, we find that preserving only tokens that store intermediate results would achieve comparable performance. Furthermore, we observe that storing intermediate results in an alternative latent form will not affect model performance. We also randomly intervene some values in CoT, and notice that subsequent CoT tokens and the final answer would change correspondingly. These findings suggest that CoT tokens may function like variables in computer programs but with potential drawbacks like unintended shortcuts and computational complexity limits between tokens. The code and data are available at this https URL. 

**Abstract (ZH)**: 链思（CoT）要求大型语言模型（LLMs）在到达最终答案之前生成中间步骤，并已被证明有助于解决复杂的推理任务。然而，CoT的内在机制仍然 largely unclear。在本文中，我们实证研究了CoT令牌在LLMs上的作用，针对两个组合任务：多位数乘法和动态规划。虽然CoT对于解决这些问题是必要的，我们发现仅保留存储中间结果的令牌即可实现相当的性能。此外，我们观察到以另一种潜在的隐式形式存储中间结果不会影响模型性能。我们还随机干预了CoT中的某些值，注意到后续的CoT令牌和最终答案会相应地发生变化。这些发现表明，CoT令牌可能类似于计算机程序中的变量，但可能存在未预期的捷径和令牌之间计算复杂性的局限性。相关代码和数据可在以下链接获取：this https URL。 

---
# T2VTextBench: A Human Evaluation Benchmark for Textual Control in Video Generation Models 

**Title (ZH)**: T2VTextBench: 用于视频生成模型中文本控制的人工评估基准 

**Authors**: Xuyang Guo, Jiayan Huo, Zhenmei Shi, Zhao Song, Jiahao Zhang, Jiale Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04946)  

**Abstract**: Thanks to recent advancements in scalable deep architectures and large-scale pretraining, text-to-video generation has achieved unprecedented capabilities in producing high-fidelity, instruction-following content across a wide range of styles, enabling applications in advertising, entertainment, and education. However, these models' ability to render precise on-screen text, such as captions or mathematical formulas, remains largely untested, posing significant challenges for applications requiring exact textual accuracy. In this work, we introduce T2VTextBench, the first human-evaluation benchmark dedicated to evaluating on-screen text fidelity and temporal consistency in text-to-video models. Our suite of prompts integrates complex text strings with dynamic scene changes, testing each model's ability to maintain detailed instructions across frames. We evaluate ten state-of-the-art systems, ranging from open-source solutions to commercial offerings, and find that most struggle to generate legible, consistent text. These results highlight a critical gap in current video generators and provide a clear direction for future research aimed at enhancing textual manipulation in video synthesis. 

**Abstract (ZH)**: 基于文本的时间一致性屏幕文本评估基准：T2VTextBench 

---
# Structural Alignment in Link Prediction 

**Title (ZH)**: 链接预测中的结构对齐 

**Authors**: Jeffrey Seathrún Sardina  

**Link**: [PDF](https://arxiv.org/pdf/2505.04939)  

**Abstract**: While Knowledge Graphs (KGs) have become increasingly popular across various scientific disciplines for their ability to model and interlink huge quantities of data, essentially all real-world KGs are known to be incomplete. As such, with the growth of KG use has been a concurrent development of machine learning tools designed to predict missing information in KGs, which is referred to as the Link Prediction Task. The majority of state-of-the-art link predictors to date have followed an embedding-based paradigm. In this paradigm, it is assumed that the information content of a KG is best represented by the (individual) vector representations of its nodes and edges, and that therefore node and edge embeddings are particularly well-suited to performing link prediction.
This thesis proposes an alternative perspective on the field's approach to link prediction and KG data modelling. Specifically, this work re-analyses KGs and state-of-the-art link predictors from a graph-structure-first perspective that models the information content of a KG in terms of whole triples, rather than individual nodes and edges.
Following a literature review and two core sets of experiments, this thesis concludes that a structure-first perspective on KGs and link prediction is both viable and useful for understanding KG learning and for enabling cross-KG transfer learning for the link prediction task. This observation is used to create and propose the Structural Alignment Hypothesis, which postulates that link prediction can be understood and modelled as a structural task.
All code and data used for this thesis are open-sourced. This thesis was written bilingually, with the main document in English and an informal extended summary in Irish. An Irish-language translation dictionary of machine learning terms (the Foclóir Tráchtais) created for this work is open-sourced as well. 

**Abstract (ZH)**: 虽然知识图谱（KGs）在各个科学领域因其建模大量数据并实现数据链接的能力而变得日益流行，但实际上所有现实世界的KGs都存在着不完整性。随着KGs应用的增长，开发用于预测KG中缺失信息的机器学习工具也得到了同步发展，这一任务被称为链接预测任务。迄今为止，大多数先进的链接预测器都遵循基于嵌入的方法。在这种方法中，假设知识图谱的信息内容可以用其节点和边的向量表示最好，并且因此节点和边的嵌入特别适合进行链接预测。

本文对该领域在链接预测和KG数据建模方面的做法提出了不同的观点。具体而言，本文从图结构优先的角度重新分析了KGs和最先进的链接预测器，以整三元组来建模KG的信息内容，而不是单独的节点和边。

在文献综述和两组核心实验之后，本文得出结论，从图结构优先的角度理解KGs和链接预测既是可行的，也有助于理解KG学习，并使链接预测任务跨KG迁移学习成为可能。本文基于这一观察提出了结构对齐假说，认为链接预测可以被理解为一个结构任务并进行建模。

本文使用的所有代码和数据均已开源。本文使用英语为主文档，并附有爱尔兰语的非正式扩展摘要。本文还开发了一个机器学习术语的爱尔兰语翻译词典（Foclóir Tráchtais），该词典也已开源。 

---
# Fair Uncertainty Quantification for Depression Prediction 

**Title (ZH)**: 抑郁症预测中的公平不确定性量化 

**Authors**: Yonghong Li, Xiuzhuang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.04931)  

**Abstract**: Trustworthy depression prediction based on deep learning, incorporating both predictive reliability and algorithmic fairness across diverse demographic groups, is crucial for clinical application. Recently, achieving reliable depression predictions through uncertainty quantification has attracted increasing attention. However, few studies have focused on the fairness of uncertainty quantification (UQ) in depression prediction. In this work, we investigate the algorithmic fairness of UQ, namely Equal Opportunity Coverage (EOC) fairness, and propose Fair Uncertainty Quantification (FUQ) for depression prediction. FUQ pursues reliable and fair depression predictions through group-based analysis. Specifically, we first group all the participants by different sensitive attributes and leverage conformal prediction to quantify uncertainty within each demographic group, which provides a theoretically guaranteed and valid way to quantify uncertainty for depression prediction and facilitates the investigation of fairness across different demographic groups. Furthermore, we propose a fairness-aware optimization strategy that formulates fairness as a constrained optimization problem under EOC constraints. This enables the model to preserve predictive reliability while adapting to the heterogeneous uncertainty levels across demographic groups, thereby achieving optimal fairness. Through extensive evaluations on several visual and audio depression datasets, our approach demonstrates its effectiveness. 

**Abstract (ZH)**: 基于深度学习的可信赖抑郁预测：结合预测可靠性与跨不同人群组的算法公平性对于临床应用至关重要。最近，通过不确定性量化实现可靠的抑郁预测吸引了越来越多的关注。然而，很少有研究关注抑郁预测中不确定性量化（UQ）的公平性。在本文中，我们研究了UQ的算法公平性，即平等机会覆盖（EOC）公平性，并提出了一种新的方法，即公平不确定性量化（FUQ）以用于抑郁预测。FUQ 通过基于群体的分析追求可靠的和公平的抑郁预测。具体来说，我们首先根据不同的敏感属性对所有参与者进行分组，并利用校准预测来量化每个群体内的不确定性，这种方法为抑郁预测提供了一种理论保证且有效的不确定性量化方式，并有助于在不同人群组之间调查公平性问题。此外，我们提出了一个公平性感知的优化策略，将公平性作为在EOC约束下的约束优化问题进行建模。这种方法使得模型在保持预测可靠性的同时，能够适应不同人群组之间的异质不确定性水平，从而实现最优公平性。通过在多个视觉和音频抑郁数据集上的广泛评估，我们的方法证实了其有效性。 

---
# Physics-Assisted and Topology-Informed Deep Learning for Weather Prediction 

**Title (ZH)**: 物理学辅助和拓扑驱动的深度学习天气预报 

**Authors**: Jiaqi Zheng, Qing Ling, Yerong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04918)  

**Abstract**: Although deep learning models have demonstrated remarkable potential in weather prediction, most of them overlook either the \textbf{physics} of the underlying weather evolution or the \textbf{topology} of the Earth's surface. In light of these disadvantages, we develop PASSAT, a novel Physics-ASSisted And Topology-informed deep learning model for weather prediction. PASSAT attributes the weather evolution to two key factors: (i) the advection process that can be characterized by the advection equation and the Navier-Stokes equation; (ii) the Earth-atmosphere interaction that is difficult to both model and calculate. PASSAT also takes the topology of the Earth's surface into consideration, other than simply treating it as a plane. With these considerations, PASSAT numerically solves the advection equation and the Navier-Stokes equation on the spherical manifold, utilizes a spherical graph neural network to capture the Earth-atmosphere interaction, and generates the initial velocity fields that are critical to solving the advection equation from the same spherical graph neural network. In the $5.625^\circ$-resolution ERA5 data set, PASSAT outperforms both the state-of-the-art deep learning-based weather prediction models and the operational numerical weather prediction model IFS T42. Code and checkpoint are available at this https URL. 

**Abstract (ZH)**: 虽然深度学习模型在天气预测方面展示了 remarkable 的潜力，但大多数模型要么忽略了基础天气演变过程的 **物理规律**，要么忽略了地球表面的 **拓扑结构**。鉴于这些缺点，我们提出了 PASSAT，这是一种新的结合物理辅助和拓扑信息的深度学习模型，用于天气预测。PASSAT 将天气演变归因于两个关键因素：（i）可以由对流方程和纳维-斯托克斯方程描述的对流过程；（ii）地球与大气的相互作用，这一作用既难以建模也难以计算。PASSAT 还考虑了地球表面的拓扑结构，而不仅仅是将其视为平面。通过这些考虑，PASSAT 在球面流形上数值求解对流方程和纳维-斯托克斯方程，利用球面图神经网络捕获地球与大气之间的相互作用，并从同一个球面图神经网络生成用于求解对流方程的初始速度场。在 $5.625^\circ$ 分辨率的 ERA5 数据集中，PASSAT 在天气预测性能上超过了最先进的基于深度学习的天气预测模型和现有的操作数值天气预报模型 IFS T42。代码和检查点可在以下链接获取。 

---
# An Open-Source Dual-Loss Embedding Model for Semantic Retrieval in Higher Education 

**Title (ZH)**: 开源双损失嵌入模型在高等教育中的语义检索 

**Authors**: Ramteja Sajja, Yusuf Sermet, Ibrahim Demir  

**Link**: [PDF](https://arxiv.org/pdf/2505.04916)  

**Abstract**: Recent advances in AI have catalyzed the adoption of intelligent educational tools, yet many semantic retrieval systems remain ill-suited to the unique linguistic and structural characteristics of academic content. This study presents two open-source embedding models fine-tuned for educational question answering, particularly in the context of course syllabi. A synthetic dataset of 3,197 sentence pairs, spanning synonymous terminology, paraphrased questions, and implicit-explicit mappings, was constructed through a combination of manual curation and large language model (LLM)-assisted generation. Two training strategies were evaluated: (1) a baseline model fine-tuned using MultipleNegativesRankingLoss (MNRL), and (2) a dual-loss model that combines MNRL with CosineSimilarityLoss to improve both semantic ranking and similarity calibration. Evaluations were conducted on 28 university course syllabi using a fixed set of natural language questions categorized into course, faculty, and teaching assistant information. Results demonstrate that both fine-tuned models outperform strong open-source baselines, including all-MiniLM-L6-v2 and multi-qa-MiniLM-L6-cos-v1, and that the dual-loss model narrows the performance gap with high-performing proprietary embeddings such as OpenAI's text-embedding-3 series. This work contributes reusable, domain-aligned embedding models and provides a replicable framework for educational semantic retrieval, supporting downstream applications such as academic chatbots, retrieval-augmented generation (RAG) systems, and learning management system (LMS) integrations. 

**Abstract (ZH)**: Recent Advances in AIhave Catalyzed the Adoption of Intelligent Educational Tools: Fine-Tuned Embedding Models for Academic Question Answering in Course Syllabi 

---
# SpatialPrompting: Keyframe-driven Zero-Shot Spatial Reasoning with Off-the-Shelf Multimodal Large Language Models 

**Title (ZH)**: 空间提示：以关键帧驱动的零样本空间推理与即用型多模态大型语言模型 

**Authors**: Shun Taguchi, Hideki Deguchi, Takumi Hamazaki, Hiroyuki Sakai  

**Link**: [PDF](https://arxiv.org/pdf/2505.04911)  

**Abstract**: This study introduces SpatialPrompting, a novel framework that harnesses the emergent reasoning capabilities of off-the-shelf multimodal large language models to achieve zero-shot spatial reasoning in three-dimensional (3D) environments. Unlike existing methods that rely on expensive 3D-specific fine-tuning with specialized 3D inputs such as point clouds or voxel-based features, SpatialPrompting employs a keyframe-driven prompt generation strategy. This framework uses metrics such as vision-language similarity, Mahalanobis distance, field of view, and image sharpness to select a diverse and informative set of keyframes from image sequences and then integrates them with corresponding camera pose data to effectively abstract spatial relationships and infer complex 3D structures. The proposed framework not only establishes a new paradigm for flexible spatial reasoning that utilizes intuitive visual and positional cues but also achieves state-of-the-art zero-shot performance on benchmark datasets, such as ScanQA and SQA3D, across several metrics. The proposed method effectively eliminates the need for specialized 3D inputs and fine-tuning, offering a simpler and more scalable alternative to conventional approaches. 

**Abstract (ZH)**: SpatialPrompting: 一种利用现成多模态大语言模型新兴推理能力实现三维环境零样本空间推理的新型框架 

---
# Precise gradient descent training dynamics for finite-width multi-layer neural networks 

**Title (ZH)**: 有限宽度多层神经网络的精确梯度下降训练动力学 

**Authors**: Qiyang Han, Masaaki Imaizumi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04898)  

**Abstract**: In this paper, we provide the first precise distributional characterization of gradient descent iterates for general multi-layer neural networks under the canonical single-index regression model, in the `finite-width proportional regime' where the sample size and feature dimension grow proportionally while the network width and depth remain bounded. Our non-asymptotic state evolution theory captures Gaussian fluctuations in first-layer weights and concentration in deeper-layer weights, and remains valid for non-Gaussian features.
Our theory differs from existing neural tangent kernel (NTK), mean-field (MF) theories and tensor program (TP) in several key aspects. First, our theory operates in the finite-width regime whereas these existing theories are fundamentally infinite-width. Second, our theory allows weights to evolve from individual initializations beyond the lazy training regime, whereas NTK and MF are either frozen at or only weakly sensitive to initialization, and TP relies on special initialization schemes. Third, our theory characterizes both training and generalization errors for general multi-layer neural networks beyond the uniform convergence regime, whereas existing theories study generalization almost exclusively in two-layer settings.
As a statistical application, we show that vanilla gradient descent can be augmented to yield consistent estimates of the generalization error at each iteration, which can be used to guide early stopping and hyperparameter tuning. As a further theoretical implication, we show that despite model misspecification, the model learned by gradient descent retains the structure of a single-index function with an effective signal determined by a linear combination of the true signal and the initialization. 

**Abstract (ZH)**: 在有限宽度比例 regime 下广义多层神经网络在单指数量化回归模型下梯度下降迭代的首个精确分布描述：不同于现有神经 tangent 核（NTK）、均场（MF）理论和张量程序（TP）的若干关键差异 

---
# Clustering with Communication: A Variational Framework for Single Cell Representation Learning 

**Title (ZH)**: 通信驱动聚类：单细胞表示学习的变分框架 

**Authors**: Cong Qi, Yeqing Chen, Jie Zhang, Wei Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04891)  

**Abstract**: Single-cell RNA sequencing (scRNA-seq) has revealed complex cellular heterogeneity, but recent studies emphasize that understanding biological function also requires modeling cell-cell communication (CCC), the signaling interactions mediated by ligand-receptor pairs that coordinate cellular behavior. Tools like CellChat have demonstrated that CCC plays a critical role in processes such as cell differentiation, tissue regeneration, and immune response, and that transcriptomic data inherently encodes rich information about intercellular signaling. We propose CCCVAE, a novel variational autoencoder framework that incorporates CCC signals into single-cell representation learning. By leveraging a communication-aware kernel derived from ligand-receptor interactions and a sparse Gaussian process, CCCVAE encodes biologically informed priors into the latent space. Unlike conventional VAEs that treat each cell independently, CCCVAE encourages latent embeddings to reflect both transcriptional similarity and intercellular signaling context. Empirical results across four scRNA-seq datasets show that CCCVAE improves clustering performance, achieving higher evaluation scores than standard VAE baselines. This work demonstrates the value of embedding biological priors into deep generative models for unsupervised single-cell analysis. 

**Abstract (ZH)**: 单细胞RNA测序(scRNA-seq)揭示了复杂的细胞异质性，但最近的研究强调，理解生物功能还需要建模细胞间通讯(CCC)，即由配体-受体配对介导的信号相互作用，协调细胞行为。像CellChat这样的工具已经证明，CCC在细胞分化、组织再生和免疫反应等过程中扮演了关键角色，并且转录组数据本身就包含了丰富的细胞间信号传导信息。我们提出了一种新颖的变分自编码器框架CCCVAE，该框架将CCC信号整合到单细胞表示学习中。CCCVAE通过利用源自配体-受体相互作用的通信感知核和稀疏高斯过程，将生物学先验知识编码到潜在空间中。与传统的独立处理每个细胞的VAE不同，CCCVAE鼓励潜在嵌入反映转录相似性和细胞间信号传导上下文。在四个scRNA-seq数据集上的实验证明，CCCVAE提高了聚类性能，评估得分高于标准VAE基线。这项工作展示了将生物学先验嵌入到深度生成模型中进行无监督单细胞分析的价值。 

---
# Cross-Branch Orthogonality for Improved Generalization in Face Deepfake Detection 

**Title (ZH)**: 跨支路正交性以提高面部深度合成检测的泛化能力 

**Authors**: Tharindu Fernando, Clinton Fookes, Sridha Sridharan, Simon Denman  

**Link**: [PDF](https://arxiv.org/pdf/2505.04888)  

**Abstract**: Remarkable advancements in generative AI technology have given rise to a spectrum of novel deepfake categories with unprecedented leaps in their realism, and deepfakes are increasingly becoming a nuisance to law enforcement authorities and the general public. In particular, we observe alarming levels of confusion, deception, and loss of faith regarding multimedia content within society caused by face deepfakes, and existing deepfake detectors are struggling to keep up with the pace of improvements in deepfake generation. This is primarily due to their reliance on specific forgery artifacts, which limits their ability to generalise and detect novel deepfake types. To combat the spread of malicious face deepfakes, this paper proposes a new strategy that leverages coarse-to-fine spatial information, semantic information, and their interactions while ensuring feature distinctiveness and reducing the redundancy of the modelled features. A novel feature orthogonality-based disentanglement strategy is introduced to ensure branch-level and cross-branch feature disentanglement, which allows us to integrate multiple feature vectors without adding complexity to the feature space or compromising generalisation. Comprehensive experiments on three public benchmarks: FaceForensics++, Celeb-DF, and the Deepfake Detection Challenge (DFDC) show that these design choices enable the proposed approach to outperform current state-of-the-art methods by 5% on the Celeb-DF dataset and 7% on the DFDC dataset in a cross-dataset evaluation setting. 

**Abstract (ZH)**: 显著的生成AI技术进步催生了一系列前所未有的高真实度深伪类别，深伪内容日益成为执法机构和普通公众的困扰。特别是，我们观察到社会上关于多媒体内容存在着令人担忧的混淆、欺骗和信任损失现象，现有的深伪检测方法难以跟上深伪生成技术的快速发展。这主要是因为它们依赖于特定的伪造特征，限制了它们对新型深伪类型的泛化和检测能力。为应对恶意面部深伪的传播，本文提出了一种新策略，该策略结合了粗细尺度空间信息、语义信息及其交互作用，同时确保特征的差异性和减少模型特征的冗余性。引入了一种新颖的基于特征正交性的解耦策略，确保分支级和跨分支特征的解耦，从而在不会增加特征空间复杂性或损害泛化能力的情况下，能够整合多个特征向量。在三个公开基准FaceForensics++、Celeb-DF和Deepfake Detection Challenge (DFDC)上的全面实验表明，这些设计选择使所提出的方法在Celeb-DF数据集上的检测性能比当前最先进的方法高出5%，在DFDC数据集上的检测性能高出7%，特别是在跨数据集评估设置中。 

---
# QBR: A Question-Bank-Based Approach to Fine-Grained Legal Knowledge Retrieval for the General Public 

**Title (ZH)**: 基于问题集的方法：面向普通公众的细粒度法律知识检索 

**Authors**: Mingruo Yuan, Ben Kao, Tien-Hsuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04883)  

**Abstract**: Retrieval of legal knowledge by the general public is a challenging problem due to the technicality of the professional knowledge and the lack of fundamental understanding by laypersons on the subject. Traditional information retrieval techniques assume that users are capable of formulating succinct and precise queries for effective document retrieval. In practice, however, the wide gap between the highly technical contents and untrained users makes legal knowledge retrieval very difficult. We propose a methodology, called QBR, which employs a Questions Bank (QB) as an effective medium for bridging the knowledge gap. We show how the QB is used to derive training samples to enhance the embedding of knowledge units within documents, which leads to effective fine-grained knowledge retrieval. We discuss and evaluate through experiments various advantages of QBR over traditional methods. These include more accurate, efficient, and explainable document retrieval, better comprehension of retrieval results, and highly effective fine-grained knowledge retrieval. We also present some case studies and show that QBR achieves social impact by assisting citizens to resolve everyday legal concerns. 

**Abstract (ZH)**: 普通公众获取法律知识是一个具有挑战性的问题，由于专业知识的技术性以及非专业人士对该主题的基本理解不足。传统的信息检索技术假设用户能够制定简洁和精确的查询以实现有效的文档检索。然而，在实践中，高度技术性的内容与未受训练的用户之间的巨大差距使得法律知识检索非常困难。我们提出了一种名为QBR的方法，该方法利用知识问答库（QB）作为弥合知识差距的有效媒介。我们展示了如何利用QB来生成训练样本以增强文档中的知识单元嵌入，从而实现有效的细粒度知识检索。我们通过实验讨论并评估了QBR相对于传统方法的各种优势，包括更准确、更高效的文档检索、更好的检索结果理解以及非常有效的细粒度知识检索。我们还呈现了一些案例研究，并展示了QBR通过帮助公民解决日常生活中的法律问题而产生的社会影响。 

---
# ConCISE: Confidence-guided Compression in Step-by-step Efficient Reasoning 

**Title (ZH)**: ConCISE: 基于信心引导的逐步高效推理压缩 

**Authors**: Ziqing Qiao, Yongheng Deng, Jiali Zeng, Dong Wang, Lai Wei, Fandong Meng, Jie Zhou, Ju Ren, Yaoxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04881)  

**Abstract**: Large Reasoning Models (LRMs) perform strongly in complex reasoning tasks via Chain-of-Thought (CoT) prompting, but often suffer from verbose outputs caused by redundant content, increasing computational overhead, and degrading user experience. Existing compression methods either operate post-hoc pruning, risking disruption to reasoning coherence, or rely on sampling-based selection, which fails to intervene effectively during generation. In this work, we introduce a confidence-guided perspective to explain the emergence of redundant reflection in LRMs, identifying two key patterns: Confidence Deficit, where the model reconsiders correct steps due to low internal confidence, and Termination Delay, where reasoning continues even after reaching a confident answer. Based on this analysis, we propose ConCISE (Confidence-guided Compression In Step-by-step Efficient Reasoning), a framework that simplifies reasoning chains by reinforcing the model's confidence during inference, thus preventing the generation of redundant reflection steps. It integrates Confidence Injection to stabilize intermediate steps and Early Stopping to terminate reasoning when confidence is sufficient. Extensive experiments demonstrate that fine-tuning LRMs on ConCISE-generated data yields significantly shorter outputs, reducing length by up to approximately 50% under SimPO, while maintaining high task accuracy. ConCISE consistently outperforms existing baselines across multiple reasoning benchmarks. 

**Abstract (ZH)**: Large Reasoning Models (LRMs)通过Chain-of-Thought (CoT)提示在复杂推理任务中表现出色，但常常因冗余内容导致冗长输出，增加计算开销并降低用户体验。现有压缩方法要么在事后剪枝操作，有可能破坏推理连贯性，要么依赖于基于采样的选择，这在生成过程中无法有效干预。在本项工作中，我们引入了一种基于信心的视角来解释LRMs中冗余反思的产生，识别出两种关键模式：信心赤字，模型因内部信心不足而重新考虑正确的步骤；以及终止延迟，即使在得到自信的答案后，推理仍然继续。基于此分析，我们提出了ConCISE（基于信心的压缩在逐步高效推理中），一种框架，在推理过程中增强模型的信心，从而防止生成冗余反思步骤。该框架集成了信心注入以稳定中间步骤，并采用早期停止以在信心充足时终止推理。广泛实验证明，通过对ConCISE生成的数据进行微调，可以显著缩短输出长度，在SimPO下减少约50%的长度，同时保持高任务准确性。ConCISE在多种推理基准测试中均优于现有基线。 

---
# GroverGPT-2: Simulating Grover's Algorithm via Chain-of-Thought Reasoning and Quantum-Native Tokenization 

**Title (ZH)**: GroverGPT-2：通过链式推理和量子原生标记化模拟Grover算法 

**Authors**: Min Chen, Jinglei Cheng, Pingzhi Li, Haoran Wang, Tianlong Chen, Junyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04880)  

**Abstract**: Quantum computing offers theoretical advantages over classical computing for specific tasks, yet the boundary of practical quantum advantage remains an open question. To investigate this boundary, it is crucial to understand whether, and how, classical machines can learn and simulate quantum algorithms. Recent progress in large language models (LLMs) has demonstrated strong reasoning abilities, prompting exploration into their potential for this challenge. In this work, we introduce GroverGPT-2, an LLM-based method for simulating Grover's algorithm using Chain-of-Thought (CoT) reasoning and quantum-native tokenization. Building on its predecessor, GroverGPT-2 performs simulation directly from quantum circuit representations while producing logically structured and interpretable outputs. Our results show that GroverGPT-2 can learn and internalize quantum circuit logic through efficient processing of quantum-native tokens, providing direct evidence that classical models like LLMs can capture the structure of quantum algorithms. Furthermore, GroverGPT-2 outputs interleave circuit data with natural language, embedding explicit reasoning into the simulation. This dual capability positions GroverGPT-2 as a prototype for advancing machine understanding of quantum algorithms and modeling quantum circuit logic. We also identify an empirical scaling law for GroverGPT-2 with increasing qubit numbers, suggesting a path toward scalable classical simulation. These findings open new directions for exploring the limits of classical simulatability, enhancing quantum education and research, and laying groundwork for future foundation models in quantum computing. 

**Abstract (ZH)**: 量子计算在特定任务上提供了理论上的优势，但实用的量子优势边界仍然是一个开放问题。为了探讨这一边界，了解经典机器能否学习和模拟量子算法至关重要。近年来，大型语言模型（LLMs）的进步展现出强大的推理能力，促使人们探索其在这项挑战中的潜力。在这项工作中，我们引入了GroverGPT-2，这是一种基于LLM的方法，通过链式思考（CoT）推理和量子本征标记来模拟Grover算法。GroverGPT-2在其前身的基础上，直接从量子电路表示中进行模拟，生成逻辑结构化和可解释的输出。我们的结果显示，GroverGPT-2可以通过高效处理量子本征标记来学习和内化量子电路逻辑，提供了经典模型如LLMs可以捕捉量子算法结构的直接证据。此外，GroverGPT-2输出中嵌入了量子电路数据与自然语言的交织，将显式推理嵌入到模拟中。这种双重能力使GroverGPT-2成为推进机器对量子算法的理解和建模量子电路逻辑的原型。我们还识别了GroverGPT-2随量子比特数量增加的实证缩放定律，这为可扩展的经典模拟指出了道路。这些发现为探索经典模拟极限提供了新方向，增强了量子教育与研究，并为未来量子计算的基础模型奠定了基础。 

---
# Learning from Loss Landscape: Generalizable Mixed-Precision Quantization via Adaptive Sharpness-Aware Gradient Aligning 

**Title (ZH)**: 基于损失景观学习：自适应锐度感知梯度对齐的可迁移混合精度量化 

**Authors**: Lianbo Ma, Jianlun Ma, Yuee Zhou, Guoyang Xie, Qiang He, Zhichao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04877)  

**Abstract**: Mixed Precision Quantization (MPQ) has become an essential technique for optimizing neural network by determining the optimal bitwidth per layer. Existing MPQ methods, however, face a major hurdle: they require a computationally expensive search for quantization policies on large-scale datasets. To resolve this issue, we introduce a novel approach that first searches for quantization policies on small datasets and then generalizes them to large-scale datasets. This approach simplifies the process, eliminating the need for large-scale quantization fine-tuning and only necessitating model weight adjustment. Our method is characterized by three key techniques: sharpness-aware minimization for enhanced quantization generalization, implicit gradient direction alignment to handle gradient conflicts among different optimization objectives, and an adaptive perturbation radius to accelerate optimization. Both theoretical analysis and experimental results validate our approach. Using the CIFAR10 dataset (just 0.5\% the size of ImageNet training data) for MPQ policy search, we achieved equivalent accuracy on ImageNet with a significantly lower computational cost, while improving efficiency by up to 150% over the baselines. 

**Abstract (ZH)**: 基于小型数据集的迁移量化策略：Mixed Precision Quantization (MPQ) 已成为通过确定每层的最优位宽来优化神经网络的关键技术。现有 MPQ 方法面临的主要挑战是，在大规模数据集上寻找量化策略计算成本高昂。为解决这一问题，我们提出了一种新颖的方法，该方法首先在小型数据集上搜索量化策略，然后将这些策略推广到大规模数据集。该方法简化了过程，消除了对大规模量化微调的需求，只需调整模型权重。我们的方法具有三个关键技术：增强量化泛化的尖锐性感知最小化、处理不同优化目标之间梯度冲突的隐式梯度方向对齐，以及加速优化的自适应扰动半径。理论分析和实验结果均验证了该方法的有效性。使用仅相当于 ImageNet 训练数据0.5％大小的CIFAR10数据集进行 MPQ 策略搜索，我们在 ImageNet 上实现了相当于的精度，计算成本显著降低，同时相对于 baseline 方法效率提升高达150％。 

---
# Federated Learning for Cyber Physical Systems: A Comprehensive Survey 

**Title (ZH)**: 联邦学习在 cyber-physical 系统中的应用：一个综合综述 

**Authors**: Minh K. Quan, Pubudu N. Pathirana, Mayuri Wijayasundara, Sujeeva Setunge, Dinh C. Nguyen, Christopher G. Brinton, David J. Love, H. Vincent Poor  

**Link**: [PDF](https://arxiv.org/pdf/2505.04873)  

**Abstract**: The integration of machine learning (ML) in cyber physical systems (CPS) is a complex task due to the challenges that arise in terms of real-time decision making, safety, reliability, device heterogeneity, and data privacy. There are also open research questions that must be addressed in order to fully realize the potential of ML in CPS. Federated learning (FL), a distributed approach to ML, has become increasingly popular in recent years. It allows models to be trained using data from decentralized sources. This approach has been gaining popularity in the CPS field, as it integrates computer, communication, and physical processes. Therefore, the purpose of this work is to provide a comprehensive analysis of the most recent developments of FL-CPS, including the numerous application areas, system topologies, and algorithms developed in recent years. The paper starts by discussing recent advances in both FL and CPS, followed by their integration. Then, the paper compares the application of FL in CPS with its applications in the internet of things (IoT) in further depth to show their connections and distinctions. Furthermore, the article scrutinizes how FL is utilized in critical CPS applications, e.g., intelligent transportation systems, cybersecurity services, smart cities, and smart healthcare solutions. The study also includes critical insights and lessons learned from various FL-CPS implementations. The paper's concluding section delves into significant concerns and suggests avenues for further research in this fast-paced and dynamic era. 

**Abstract (ZH)**: 机器学习在 cyber 物理系统中的集成：联邦学习的进展与挑战 

---
# Auto-regressive transformation for image alignment 

**Title (ZH)**: 自动回归转换for图像对齐 

**Authors**: Kanggeon Lee, Soochahn Lee, Kyoung Mu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.04864)  

**Abstract**: Existing methods for image alignment struggle in cases involving feature-sparse regions, extreme scale and field-of-view differences, and large deformations, often resulting in suboptimal accuracy. Robustness to these challenges improves through iterative refinement of the transformation field while focusing on critical regions in multi-scale image representations. We thus propose Auto-Regressive Transformation (ART), a novel method that iteratively estimates the coarse-to-fine transformations within an auto-regressive framework. Leveraging hierarchical multi-scale features, our network refines the transformations using randomly sampled points at each scale. By incorporating guidance from the cross-attention layer, the model focuses on critical regions, ensuring accurate alignment even in challenging, feature-limited conditions. Extensive experiments across diverse datasets demonstrate that ART significantly outperforms state-of-the-art methods, establishing it as a powerful new method for precise image alignment with broad applicability. 

**Abstract (ZH)**: 现有的图像对齐方法在特征稀疏区域、极端尺度和视场差异以及大形变的情况下表现不佳，常常导致亚最优的准确性。通过在多尺度图像表示中迭代细化变换场并聚焦关键区域，可以提高对这些挑战的鲁棒性。因此，我们提出了自回归变换（ART），这是一种在自回归框架内迭代估计粗到细变换的新方法。利用分层多尺度特征，我们的网络在每个尺度上随机采样点以细化变换。通过引入交叉注意力层的指导，模型能够聚焦于关键区域，即使在特征受限的挑战条件下也能确保准确对齐。广泛的实验证明，ART在多个数据集上显著优于现有最先进的方法，确立了其作为一种适用于精确图像对齐的强大新方法的地位。 

---
# D-CODA: Diffusion for Coordinated Dual-Arm Data Augmentation 

**Title (ZH)**: D-CODA: 扩散驱动的协调双臂数据增强 

**Authors**: I-Chun Arthur Liu, Jason Chen, Gaurav Sukhatme, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2505.04860)  

**Abstract**: Learning bimanual manipulation is challenging due to its high dimensionality and tight coordination required between two arms. Eye-in-hand imitation learning, which uses wrist-mounted cameras, simplifies perception by focusing on task-relevant views. However, collecting diverse demonstrations remains costly, motivating the need for scalable data augmentation. While prior work has explored visual augmentation in single-arm settings, extending these approaches to bimanual manipulation requires generating viewpoint-consistent observations across both arms and producing corresponding action labels that are both valid and feasible. In this work, we propose Diffusion for COordinated Dual-arm Data Augmentation (D-CODA), a method for offline data augmentation tailored to eye-in-hand bimanual imitation learning that trains a diffusion model to synthesize novel, viewpoint-consistent wrist-camera images for both arms while simultaneously generating joint-space action labels. It employs constrained optimization to ensure that augmented states involving gripper-to-object contacts adhere to constraints suitable for bimanual coordination. We evaluate D-CODA on 5 simulated and 3 real-world tasks. Our results across 2250 simulation trials and 300 real-world trials demonstrate that it outperforms baselines and ablations, showing its potential for scalable data augmentation in eye-in-hand bimanual manipulation. Our project website is at: this https URL. 

**Abstract (ZH)**: 基于手腕相机的双臂协调数据增强方法（D-CODA）：面向 Eye-in-Hand 双臂模仿学习的离线数据增强 

---
# PR2: Peephole Raw Pointer Rewriting with LLMs for Translating C to Safer Rust 

**Title (ZH)**: PR2: 使用LLMs的崽孔原始指针重写以将C翻译为更安全的Rust 

**Authors**: Yifei Gao, Chengpeng Wang, Pengxiang Huang, Xuwei Liu, Mingwei Zheng, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04852)  

**Abstract**: There has been a growing interest in translating C code to Rust due to Rust's robust memory and thread safety guarantees. Tools such as C2RUST enable syntax-guided transpilation from C to semantically equivalent Rust code. However, the resulting Rust programs often rely heavily on unsafe constructs--particularly raw pointers--which undermines Rust's safety guarantees. This paper aims to improve the memory safety of Rust programs generated by C2RUST by eliminating raw pointers. Specifically, we propose a peephole raw pointer rewriting technique that lifts raw pointers in individual functions to appropriate Rust data structures. Technically, PR2 employs decision-tree-based prompting to guide the pointer lifting process. Additionally, it leverages code change analysis to guide the repair of errors introduced during rewriting, effectively addressing errors encountered during compilation and test case execution. We implement PR2 as a prototype and evaluate it using gpt-4o-mini on 28 real-world C projects. The results show that PR2 successfully eliminates 13.22% of local raw pointers across these projects, significantly enhancing the safety of the translated Rust code. On average, PR2 completes the transformation of a project in 5.44 hours, at an average cost of $1.46. 

**Abstract (ZH)**: 基于决策树提示的无损指针重写技术以提升由C2RUST生成的Rust程序的内存安全性 

---
# Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards 

**Title (ZH)**: 基于 evolving leaderboards 的 LLM 忠实性在 RAG 中的基准测试 

**Authors**: Manveer Singh Tamber, Forrest Sheng Bao, Chenyu Xu, Ge Luo, Suleman Kazi, Minseok Bae, Miaoran Li, Ofer Mendelevitch, Renyi Qu, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04847)  

**Abstract**: Hallucinations remain a persistent challenge for LLMs. RAG aims to reduce hallucinations by grounding responses in contexts. However, even when provided context, LLMs still frequently introduce unsupported information or contradictions. This paper presents our efforts to measure LLM hallucinations with a focus on summarization tasks, assessing how often various LLMs introduce hallucinations when summarizing documents. We discuss Vectara's existing LLM hallucination leaderboard, based on the Hughes Hallucination Evaluation Model (HHEM). While HHEM and Vectara's Hallucination Leaderboard have garnered great research interest, we examine challenges faced by HHEM and current hallucination detection methods by analyzing the effectiveness of these methods on existing hallucination datasets. To address these limitations, we propose FaithJudge, an LLM-as-a-judge approach guided by few-shot human hallucination annotations, which substantially improves automated LLM hallucination evaluation over current methods. We introduce an enhanced hallucination leaderboard centered on FaithJudge, alongside our current hallucination leaderboard, enabling more reliable benchmarking of LLMs for hallucinations in RAG. 

**Abstract (ZH)**: LLMs中的幻觉仍然是一个持续的挑战。RAG通过将响应与上下文联系起来以减少幻觉。然而，即使提供了上下文，LLMs仍然经常引入未支持的信息或矛盾。本文旨在通过总结任务衡量LLMs的幻觉，评估各种LLMs在总结文档时引入幻觉的频率。我们讨论了Vectara现有的基于Hughes Hallucination Evaluation Model (HHEM)的LLM幻觉排行榜。虽然HHEM和Vectara的幻觉排行榜吸引了大量研究兴趣，但我们通过分析这些方法在现有幻觉数据集上的有效性，考察了HHEM和其他当前幻觉检测方法面临的一些挑战。为了解决这些限制，我们提出了FaithJudge，这是一种由少量人类幻觉注释引导的LLM作为裁判的方法，相较于当前的方法，FaithJudge显著提高了自动评估LLM幻觉的效果。我们引入了一个聚焦于FaithJudge的增强幻觉排行榜，并与当前的幻觉排行榜一起使用，为RAG中的LLM幻觉基准测试提供更可靠的指标。 

---
# Putting the Value Back in RL: Better Test-Time Scaling by Unifying LLM Reasoners With Verifiers 

**Title (ZH)**: 将价值带回RL：通过统一LLM推理器与验证器提高测试时缩放效果 

**Authors**: Kusha Sareen, Morgane M Moss, Alessandro Sordoni, Rishabh Agarwal, Arian Hosseini  

**Link**: [PDF](https://arxiv.org/pdf/2505.04842)  

**Abstract**: Prevalent reinforcement learning~(RL) methods for fine-tuning LLM reasoners, such as GRPO or Leave-one-out PPO, abandon the learned value function in favor of empirically estimated returns. This hinders test-time compute scaling that relies on using the value-function for verification. In this work, we propose RL$^V$ that augments any ``value-free'' RL method by jointly training the LLM as both a reasoner and a generative verifier using RL-generated data, adding verification capabilities without significant overhead. Empirically, RL$^V$ boosts MATH accuracy by over 20\% with parallel sampling and enables $8-32\times$ efficient test-time compute scaling compared to the base RL method. RL$^V$ also exhibits strong generalization capabilities for both easy-to-hard and out-of-domain tasks. Furthermore, RL$^V$ achieves $1.2-1.6\times$ higher performance when jointly scaling parallel and sequential test-time compute with a long reasoning R1 model. 

**Abstract (ZH)**: 基于值的强化学习方法RL<sup>V</sup>用于增强LLM推理器的细调，同时显著减少计算开销 

---
# Quantum-Inspired Optimization Process for Data Imputation 

**Title (ZH)**: 量子启发的优化过程用于数据插补 

**Authors**: Nishikanta Mohanty, Bikash K. Behera, Badsah Mukherjee, Christopher Ferrie  

**Link**: [PDF](https://arxiv.org/pdf/2505.04841)  

**Abstract**: Data imputation is a critical step in data pre-processing, particularly for datasets with missing or unreliable values. This study introduces a novel quantum-inspired imputation framework evaluated on the UCI Diabetes dataset, which contains biologically implausible missing values across several clinical features. The method integrates Principal Component Analysis (PCA) with quantum-assisted rotations, optimized through gradient-free classical optimizers -COBYLA, Simulated Annealing, and Differential Evolution to reconstruct missing values while preserving statistical fidelity. Reconstructed values are constrained within +/-2 standard deviations of original feature distributions, avoiding unrealistic clustering around central tendencies. This approach achieves a substantial and statistically significant improvement, including an average reduction of over 85% in Wasserstein distance and Kolmogorov-Smirnov test p-values between 0.18 and 0.22, compared to p-values > 0.99 in classical methods such as Mean, KNN, and MICE. The method also eliminates zero-value artifacts and enhances the realism and variability of imputed data. By combining quantum-inspired transformations with a scalable classical framework, this methodology provides a robust solution for imputation tasks in domains such as healthcare and AI pipelines, where data quality and integrity are crucial. 

**Abstract (ZH)**: 基于量子启发的新型插补框架在UCI糖尿病数据集上的评估 

---
# Piecewise Constant Spectral Graph Neural Network 

**Title (ZH)**: 分段常数谱图神经网络 

**Authors**: Vahan Martirosyan, Jhony H. Giraldo, Fragkiskos D. Malliaros  

**Link**: [PDF](https://arxiv.org/pdf/2505.04808)  

**Abstract**: Graph Neural Networks (GNNs) have achieved significant success across various domains by leveraging graph structures in data. Existing spectral GNNs, which use low-degree polynomial filters to capture graph spectral properties, may not fully identify the graph's spectral characteristics because of the polynomial's small degree. However, increasing the polynomial degree is computationally expensive and beyond certain thresholds leads to performance plateaus or degradation. In this paper, we introduce the Piecewise Constant Spectral Graph Neural Network(PieCoN) to address these challenges. PieCoN combines constant spectral filters with polynomial filters to provide a more flexible way to leverage the graph structure. By adaptively partitioning the spectrum into intervals, our approach increases the range of spectral properties that can be effectively learned. Experiments on nine benchmark datasets, including both homophilic and heterophilic graphs, demonstrate that PieCoN is particularly effective on heterophilic datasets, highlighting its potential for a wide range of applications. 

**Abstract (ZH)**: Piecewise Constant Spectral Graph Neural Network (PieCoN) 

---
# ORBIT-2: Scaling Exascale Vision Foundation Models for Weather and Climate Downscaling 

**Title (ZH)**: ORBIT-2: 扩展级谢스cale视觉基础模型用于天气和气候下scalings 

**Authors**: Xiao Wang, Jong-Youl Choi, Takuya Kurihaya, Isaac Lyngaas, Hong-Jun Yoon, Ming Fan, Nasik Muhammad Nafi, Aristeidis Tsaris, Ashwin M. Aji, Maliha Hossain, Mohamed Wahib, Dali Wang, Peter Thornton, Prasanna Balaprakash, Moetasim Ashfaq, Dan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04802)  

**Abstract**: Sparse observations and coarse-resolution climate models limit effective regional decision-making, underscoring the need for robust downscaling. However, existing AI methods struggle with generalization across variables and geographies and are constrained by the quadratic complexity of Vision Transformer (ViT) self-attention. We introduce ORBIT-2, a scalable foundation model for global, hyper-resolution climate downscaling. ORBIT-2 incorporates two key innovations: (1) Residual Slim ViT (Reslim), a lightweight architecture with residual learning and Bayesian regularization for efficient, robust prediction; and (2) TILES, a tile-wise sequence scaling algorithm that reduces self-attention complexity from quadratic to linear, enabling long-sequence processing and massive parallelism. ORBIT-2 scales to 10 billion parameters across 32,768 GPUs, achieving up to 1.8 ExaFLOPS sustained throughput and 92-98% strong scaling efficiency. It supports downscaling to 0.9 km global resolution and processes sequences up to 4.2 billion tokens. On 7 km resolution benchmarks, ORBIT-2 achieves high accuracy with R^2 scores in the range of 0.98 to 0.99 against observation data. 

**Abstract (ZH)**: 稀疏的观测数据和粗分辨率的气候模型限制了有效的区域决策制定，强调了需要稳健的降尺度方法。然而，现有的AI方法在变量和地理方面的泛化能力有限，并且受到Vision Transformer（ViT）自注意力二次复杂性的约束。我们引入了ORBIT-2，这是一种适用于全球高分辨率气候降尺度的大规模基础模型。ORBIT-2 包含两项关键创新：（1）残差轻量级ViT（Reslim），一种具有残差学习和贝叶斯正则化的轻量级架构，用于高效稳健的预测；（2）适用于序列缩放的TILES算法，将自注意力复杂性从二次降低到线性，从而实现长序列处理和大规模并行处理。ORBIT-2 跨32,768块GPU可扩展至100亿参数，实现高达1.8 ExaFLOPS的持续吞吐量，并具有92-98%的强扩展效率。它支持将其分辨率细分为0.9公里的全球分辨率，并处理最长42亿个标记的序列。在7公里分辨率的基准测试中，ORBIT-2 在观测数据的 R² 得分范围为0.98到0.99，实现了高精度。 

---
# Confabulation dynamics in a reservoir computer: Filling in the gaps with untrained attractors 

**Title (ZH)**: reservoir计算机中的虚构动态：用未训练的吸引子填补空白 

**Authors**: Jack O'Hagan, Andrew Keane, Andrew Flynn  

**Link**: [PDF](https://arxiv.org/pdf/2505.04792)  

**Abstract**: Artificial Intelligence has advanced significantly in recent years thanks to innovations in the design and training of artificial neural networks (ANNs). Despite these advancements, we still understand relatively little about how elementary forms of ANNs learn, fail to learn, and generate false information without the intent to deceive, a phenomenon known as `confabulation'. To provide some foundational insight, in this paper we analyse how confabulation occurs in reservoir computers (RCs): a dynamical system in the form of an ANN. RCs are particularly useful to study as they are known to confabulate in a well-defined way: when RCs are trained to reconstruct the dynamics of a given attractor, they sometimes construct an attractor that they were not trained to construct, a so-called `untrained attractor' (UA). This paper sheds light on the role played by UAs when reconstruction fails and their influence when modelling transitions between reconstructed attractors. Based on our results, we conclude that UAs are an intrinsic feature of learning systems whose state spaces are bounded, and that this means of confabulation may be present in systems beyond RCs. 

**Abstract (ZH)**: 近年来，由于人工神经网络（ANNs）在设计和训练方面的创新，人工智能取得了显著进展。尽管取得了这些进步，我们仍对基本形式的ANNs如何学习、无法学习以及生成虚假信息（无意欺骗的现象，称之为“自欺”）知之甚少。为了提供一些基础性的见解，本文分析了“水库计算机”（RCs）中自欺现象是如何发生的：作为一种动态系统，ANN的形式，RCs以其在特定情况下自欺的方式而特别有用：当RCs被训练以重构给定吸引子的动力学时，有时会构建一个它们没有被训练的吸引子，称为“未训练吸引子”（UA）。本文探讨了当重构失败时未训练吸引子的作用及其在建模重构吸引子之间过渡时的影响。根据我们的研究结果，我们得出结论认为，未训练吸引子是学习系统中固有的特征，这意味着这种自欺手段可能存在于RCs之外的系统中。 

---
# Replay to Remember (R2R): An Efficient Uncertainty-driven Unsupervised Continual Learning Framework Using Generative Replay 

**Title (ZH)**: 回忆以记住 (R2R): 一种基于不确定性驱动的生成回放高效无监督连续学习框架 

**Authors**: Sriram Mandalika, Harsha Vardhan, Athira Nambiar  

**Link**: [PDF](https://arxiv.org/pdf/2505.04787)  

**Abstract**: Continual Learning entails progressively acquiring knowledge from new data while retaining previously acquired knowledge, thereby mitigating ``Catastrophic Forgetting'' in neural networks. Our work presents a novel uncertainty-driven Unsupervised Continual Learning framework using Generative Replay, namely ``Replay to Remember (R2R)''. The proposed R2R architecture efficiently uses unlabelled and synthetic labelled data in a balanced proportion using a cluster-level uncertainty-driven feedback mechanism and a VLM-powered generative replay module. Unlike traditional memory-buffer methods that depend on pretrained models and pseudo-labels, our R2R framework operates without any prior training. It leverages visual features from unlabeled data and adapts continuously using clustering-based uncertainty estimation coupled with dynamic thresholding. Concurrently, a generative replay mechanism along with DeepSeek-R1 powered CLIP VLM produces labelled synthetic data representative of past experiences, resembling biological visual thinking that replays memory to remember and act in new, unseen tasks. Extensive experimental analyses are carried out in CIFAR-10, CIFAR-100, CINIC-10, SVHN and TinyImageNet datasets. Our proposed R2R approach improves knowledge retention, achieving a state-of-the-art performance of 98.13%, 73.06%, 93.41%, 95.18%, 59.74%, respectively, surpassing state-of-the-art performance by over 4.36%. 

**Abstract (ZH)**: 持续学习涉及从新数据中逐步获取知识同时保留之前获取的知识，从而减轻神经网络中的“灾难性遗忘”。我们的工作提出了一种基于不确定性驱动的无监督持续学习框架，名为“回忆以记起（R2R）”，该框架利用生成回放机制。提出的R2R架构通过簇级不确定性驱动的反馈机制和基于VLM的生成回放模块，高效地在平衡比例下使用未标记和合成标记数据。与依赖预训练模型和伪标签的传统记忆缓冲方法不同，我们的R2R框架无需任何先验训练。它利用未标记数据的视觉特征，并结合基于聚类的不确定性估计和动态阈值进行持续适应。同时，生成回放机制与DeepSeek-R1增强的CLIP VLM合作生成代表过往经历的标记合成数据，类似于生物视觉思考中的回忆记忆以记住并在新、未见任务中采取行动。在CIFAR-10、CIFAR-100、CINIC-10、SVHN和TinyImageNet数据集上进行了广泛实验分析。我们提出的R2R方法提高了知识保留，分别达到了98.13%、73.06%、93.41%、95.18%、59.74%的性能，超过现有最佳性能4.36%以上。 

---
# Flower Across Time and Media: Sentiment Analysis of Tang Song Poetry and Visual Correspondence 

**Title (ZH)**: .time与媒介中的花朵：唐宋诗词的情感分析与视觉对应研究 

**Authors**: Shuai Gong, Tiange Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.04785)  

**Abstract**: The Tang (618 to 907) and Song (960 to 1279) dynasties witnessed an extraordinary flourishing of Chinese cultural expression, where floral motifs served as a dynamic medium for both poetic sentiment and artistic design. While previous scholarship has examined these domains independently, the systematic correlation between evolving literary emotions and visual culture remains underexplored. This study addresses that gap by employing BERT-based sentiment analysis to quantify emotional patterns in floral imagery across Tang Song poetry, then validating these patterns against contemporaneous developments in decorative this http URL approach builds upon recent advances in computational humanities while remaining grounded in traditional sinological methods. By applying a fine tuned BERT model to analyze peony and plum blossom imagery in classical poetry, we detect measurable shifts in emotional connotations between the Tang and Song periods. These textual patterns are then cross berenced with visual evidence from textiles, ceramics, and other material culture, revealing previously unrecognized synergies between literary expression and artistic representation. 

**Abstract (ZH)**: 唐代（618-907）和宋代（960-1279）见证了 Chinese 文化表达的 extraordinary 繁荣，其中花饰图案成为抒情和艺术设计的动态媒介。尽管以往的研究已经分别探讨了这些领域，但文学情感演变与视觉文化的系统性关联仍待深入探索。本研究通过使用基于 BERT 的情感分析来量化唐宋诗歌中花饰意象的情感模式，然后将这些模式与同期装饰艺术的发展进行比对，从而填补了这一空白。该研究建立在人文计算的最新进展之上，同时仍然根植于传统的汉学方法。通过应用微调后的 BERT 模型分析古典诗歌中的牡丹和梅花意象，我们发现唐代和宋代之间情感内涵存在可量化的转变。然后将这些文本模式与纺织品、陶瓷和其他物质文化中的视觉证据进行交叉验证，揭示了文学表达与艺术表现之间的先前未被认识的协同关系。 

---
# A Proposal for Evaluating the Operational Risk for ChatBots based on Large Language Models 

**Title (ZH)**: 基于大型语言模型的聊天机器人操作风险评估Proposal 

**Authors**: Pedro Pinacho-Davidson, Fernando Gutierrez, Pablo Zapata, Rodolfo Vergara, Pablo Aqueveque  

**Link**: [PDF](https://arxiv.org/pdf/2505.04784)  

**Abstract**: The emergence of Generative AI (Gen AI) and Large Language Models (LLMs) has enabled more advanced chatbots capable of human-like interactions. However, these conversational agents introduce a broader set of operational risks that extend beyond traditional cybersecurity considerations. In this work, we propose a novel, instrumented risk-assessment metric that simultaneously evaluates potential threats to three key stakeholders: the service-providing organization, end users, and third parties. Our approach incorporates the technical complexity required to induce erroneous behaviors in the chatbot--ranging from non-induced failures to advanced prompt-injection attacks--as well as contextual factors such as the target industry, user age range, and vulnerability severity. To validate our metric, we leverage Garak, an open-source framework for LLM vulnerability testing. We further enhance Garak to capture a variety of threat vectors (e.g., misinformation, code hallucinations, social engineering, and malicious code generation). Our methodology is demonstrated in a scenario involving chatbots that employ retrieval-augmented generation (RAG), showing how the aggregated risk scores guide both short-term mitigation and longer-term improvements in model design and deployment. The results underscore the importance of multi-dimensional risk assessments in operationalizing secure, reliable AI-driven conversational systems. 

**Abstract (ZH)**: 生成式人工智能和大型语言模型的 emergence 使得更加先进的聊天机器人能够实现类人的交互。然而，这些对话代理引入了一套更广泛的操作风险，超出了传统的网络安全考量。在本研究中，我们提出了一种新颖的、可操作的风险评估指标，同时评估对服务提供组织、终端用户和第三方三个关键利益相关者的潜在威胁。我们的方法纳入了诱导聊天机器人错误行为所需的的技术复杂度，涵盖了从未诱导故障到高级提示注入攻击的各方面，同时考虑了目标行业、用户年龄范围和漏洞严重性等上下文因素。为了验证我们的指标，我们利用Garak，一个开源的大型语言模型漏洞测试框架。我们进一步增强了Garak以捕获各种威胁向量（如虚假信息、代码错觉、社会工程和恶意代码生成）。这种方法在涉及检索增强生成（RAG）的聊天机器人场景中得到了演示，展示了综合风险评分如何指导短期缓解措施和长时间内的模型设计与部署改进。研究结果强调了在实现安全可靠的AI驱动对话系统时进行多维度风险评估的重要性。 

---
# Exploring Zero-Shot App Review Classification with ChatGPT: Challenges and Potential 

**Title (ZH)**: 基于ChatGPT的零样本应用评论分类：挑战与潜力 

**Authors**: Mohit Chaudhary, Chirag Jain, Preethu Rose Anish  

**Link**: [PDF](https://arxiv.org/pdf/2505.04759)  

**Abstract**: App reviews are a critical source of user feedback, offering valuable insights into an app's performance, features, usability, and overall user experience. Effectively analyzing these reviews is essential for guiding app development, prioritizing feature updates, and enhancing user satisfaction. Classifying reviews into functional and non-functional requirements play a pivotal role in distinguishing feedback related to specific app features (functional requirements) from feedback concerning broader quality attributes, such as performance, usability, and reliability (non-functional requirements). Both categories are integral to informed development decisions. Traditional approaches to classifying app reviews are hindered by the need for large, domain-specific datasets, which are often costly and time-consuming to curate. This study explores the potential of zero-shot learning with ChatGPT for classifying app reviews into four categories: functional requirement, non-functional requirement, both, or neither. We evaluate ChatGPT's performance on a benchmark dataset of 1,880 manually annotated reviews from ten diverse apps spanning multiple domains. Our findings demonstrate that ChatGPT achieves a robust F1 score of 0.842 in review classification, despite certain challenges and limitations. Additionally, we examine how factors such as review readability and length impact classification accuracy and conduct a manual analysis to identify review categories more prone to misclassification. 

**Abstract (ZH)**: 应用程序评论是用户反馈的关键来源，提供了关于应用程序性能、功能、可用性和整体用户体验的重要见解。有效地分析这些评论对于指导应用程序开发、优先考虑功能更新以及提升用户满意度至关重要。将评论分类为功能性要求和非功能性要求在区分与具体应用程序功能相关的反馈（功能性要求）和与更广泛的Quality属性相关的反馈（非功能性要求）方面起到关键作用。两类反馈对于明智的开发决策都是不可或缺的。传统的应用程序评论分类方法受限于需要大量的特定领域数据集，这些数据集往往成本高昂且耗时。本研究探讨了使用ChatGPT进行零-shot学习在将应用程序评论分类为四种类别（功能性要求、非功能性要求、两者皆是或两者都不是）方面的潜力。我们评估了ChatGPT在1,880条人工标注的评论基准数据集中（来自十个不同领域的应用程序）的表现。研究结果表明，尽管存在某些挑战和限制，ChatGPT在评论分类中的F1分数达到了0.842。此外，我们还研究了评论可读性和长度等因素如何影响分类准确性，并进行了人工分析以识别更易出现分类错误的评论类别。 

---
# When Bad Data Leads to Good Models 

**Title (ZH)**: 当不良数据造就了良好模型 

**Authors**: Kenneth Li, Yida Chen, Fernanda Viégas, Martin Wattenberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.04741)  

**Abstract**: In large language model (LLM) pretraining, data quality is believed to determine model quality. In this paper, we re-examine the notion of "quality" from the perspective of pre- and post-training co-design. Specifically, we explore the possibility that pre-training on more toxic data can lead to better control in post-training, ultimately decreasing a model's output toxicity. First, we use a toy experiment to study how data composition affects the geometry of features in the representation space. Next, through controlled experiments with Olmo-1B models trained on varying ratios of clean and toxic data, we find that the concept of toxicity enjoys a less entangled linear representation as the proportion of toxic data increases. Furthermore, we show that although toxic data increases the generational toxicity of the base model, it also makes the toxicity easier to remove. Evaluations on Toxigen and Real Toxicity Prompts demonstrate that models trained on toxic data achieve a better trade-off between reducing generational toxicity and preserving general capabilities when detoxifying techniques such as inference-time intervention (ITI) are applied. Our findings suggest that, with post-training taken into account, bad data may lead to good models. 

**Abstract (ZH)**: 在大型语言模型（LLM）预训练中，数据质量被认为决定模型质量。本文从预训练和后训练协同设计的角度重新审视“质量”的概念。具体而言，我们探索了使用更具毒性的数据进行预训练是否能够在后训练中更好地控制模型输出的毒性，从而最终降低模型的输出毒性。首先，我们通过一个玩具实验研究数据组成如何影响表示空间中特征的几何结构。其次，通过使用Olmo-1B模型在不同比例的干净和有毒数据上进行训练的受控实验，我们发现随着有毒数据比例的增加，毒性概念在表示空间中享有更少纠缠的线性表示。此外，我们表明，尽管有毒数据增加了基模型生成的毒性，但它也使得去除毒性更加容易。在Toxigen和Real Toxicity Prompts上的评估表明，在应用推理时干预（ITI）等去毒技术时，使用有毒数据训练的模型在降低生成毒性与保留通用能力之间实现了更好的权衡。我们的研究结果表明，考虑到后训练因素，坏数据可能导致好模型。 

---
# QBD-RankedDataGen: Generating Custom Ranked Datasets for Improving Query-By-Document Search Using LLM-Reranking with Reduced Human Effort 

**Title (ZH)**: QBD-RankedDataGen: 生成定制排序数据集以利用LLM重排序改进查询-by-文档搜索，同时减少人工努力 

**Authors**: Sriram Gopalakrishnan, Sunandita Patra  

**Link**: [PDF](https://arxiv.org/pdf/2505.04732)  

**Abstract**: The Query-By-Document (QBD) problem is an information retrieval problem where the query is a document, and the retrieved candidates are documents that match the query document, often in a domain or query specific manner. This can be crucial for tasks such as patent matching, legal or compliance case retrieval, and academic literature review. Existing retrieval methods, including keyword search and document embeddings, can be optimized with domain-specific datasets to improve QBD search performance. However, creating these domain-specific datasets is often costly and time-consuming. Our work introduces a process to generate custom QBD-search datasets and compares a set of methods to use in this problem, which we refer to as QBD-RankedDatagen. We provide a comparative analysis of our proposed methods in terms of cost, speed, and the human interface with the domain experts. The methods we compare leverage Large Language Models (LLMs) which can incorporate domain expert input to produce document scores and rankings, as well as explanations for human review. The process and methods for it that we present can significantly reduce human effort in dataset creation for custom domains while still obtaining sufficient expert knowledge for tuning retrieval models. We evaluate our methods on QBD datasets from the Text Retrieval Conference (TREC) and finetune the parameters of the BM25 model -- which is used in many industrial-strength search engines like OpenSearch -- using the generated data. 

**Abstract (ZH)**: 基于文档的查询（QBD）问题是一种信息检索问题，其中查询是一个文档，检索的候选项是与查询文档匹配的文档，通常是在特定领域或查询的基础上进行匹配。这对于专利匹配、法律或合规案例检索以及学术文献回顾等任务至关重要。现有的检索方法，包括关键词搜索和文档嵌入，可以通过使用特定领域的数据集进行优化以提高QBD搜索性能。然而，创建这些特定领域的数据集通常成本高且耗时。我们的工作介绍了一种生成自定义QBD搜索数据集的过程，并比较了一系列在该问题中使用的方法，我们将其称为QBD-RankedDatagen。我们从成本、速度以及与领域专家的人机接口方面对提出的这些方法进行了比较分析。我们比较的方法利用了大型语言模型（LLMs），可以结合领域专家的输入来生成文档评分和排名，以及供人工审核的解释。我们提出的过程和方法可以显著减少在自定义领域创建数据集所需的人工努力，同时仍然能够获得足够的专家知识来调整检索模型。我们在Text Retrieval Conference (TREC)提供的QBD数据集上评估了我们的方法，并使用生成的数据对BM25模型进行了微调，该模型广泛应用于许多工业级搜索引擎，如OpenSearch。 

---
# Geometric Fault-Tolerant Neural Network Tracking Control of Unknown Systems on Matrix Lie Groups 

**Title (ZH)**: 未知系统的矩阵李群上几何容错神经网络跟踪控制 

**Authors**: Robin Chhabra, Farzaneh Abdollahi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04725)  

**Abstract**: We present a geometric neural network-based tracking controller for systems evolving on matrix Lie groups under unknown dynamics, actuator faults, and bounded disturbances. Leveraging the left-invariance of the tangent bundle of matrix Lie groups, viewed as an embedded submanifold of the vector space $\R^{N\times N}$, we propose a set of learning rules for neural network weights that are intrinsically compatible with the Lie group structure and do not require explicit parameterization. Exploiting the geometric properties of Lie groups, this approach circumvents parameterization singularities and enables a global search for optimal weights. The ultimate boundedness of all error signals -- including the neural network weights, the coordinate-free configuration error function, and the tracking velocity error -- is established using Lyapunov's direct method. To validate the effectiveness of the proposed method, we provide illustrative simulation results for decentralized formation control of multi-agent systems on the Special Euclidean group. 

**Abstract (ZH)**: 基于几何神经网络的矩阵李群上具有未知动力学、执行器故障和有界干扰系统的跟踪控制器研究 

---
# Advanced Deep Learning Approaches for Automated Recognition of Cuneiform Symbols 

**Title (ZH)**: 先进深度学习方法在楔形符号自动识别中的应用 

**Authors**: Shahad Elshehaby, Alavikunhu Panthakkan, Hussain Al-Ahmad, Mina Al-Saad  

**Link**: [PDF](https://arxiv.org/pdf/2505.04678)  

**Abstract**: This paper presents a thoroughly automated method for identifying and interpreting cuneiform characters via advanced deep-learning algorithms. Five distinct deep-learning models were trained on a comprehensive dataset of cuneiform characters and evaluated according to critical performance metrics, including accuracy and precision. Two models demonstrated outstanding performance and were subsequently assessed using cuneiform symbols from the Hammurabi law acquisition, notably Hammurabi Law 1. Each model effectively recognized the relevant Akkadian meanings of the symbols and delivered precise English translations. Future work will investigate ensemble and stacking approaches to optimize performance, utilizing hybrid architectures to improve detection accuracy and reliability. This research explores the linguistic relationships between Akkadian, an ancient Mesopotamian language, and Arabic, emphasizing their historical and cultural linkages. This study demonstrates the capability of deep learning to decipher ancient scripts by merging computational linguistics with archaeology, therefore providing significant insights for the comprehension and conservation of human history. 

**Abstract (ZH)**: 本文提出了一种完全自动化的方法，通过先进深度学习算法识别和解释楔形文字字符。五种不同的深度学习模型在楔形文字字符的全面数据集上进行训练，并根据关键性能指标（包括准确率和精密度）进行评估。其中两种模型表现突出，并随后使用汉谟拉比法典中的楔形文字符号（尤其是汉谟拉比法典第1条）进行了评估，每种模型都能有效地识别相关的阿卡德语意义，并提供精确的英语翻译。未来工作将研究集成和堆叠方法以优化性能，并利用混合架构提高检测准确性和可靠性。本文探讨了阿卡德语与阿拉伯语之间的语言关系，强调了它们的历史和文化联系。该研究展示了深度学习在合并计算语言学与考古学以解译古代文字方面的能力，从而为理解和保护人类历史提供了重要的见解。 

---
# Proceedings The 13th International Workshop on Theorem proving components for Educational software 

**Title (ZH)**: 第13届定理证明组件在教育软件研讨会论文集 

**Authors**: Julien Narboux, Walther Neuper, Pedro Quaresma  

**Link**: [PDF](https://arxiv.org/pdf/2505.04677)  

**Abstract**: The ThEdu series pursues the smooth transition from an intuitive way of doing mathematics at secondary school to a more formal approach to the subject in STEM education while favoring software support for this transition by exploiting the power of theorem-proving technologies.  What follows is a brief description of how the present volume contributes to this enterprise.  The 13th International Workshop on Theorem Proving Components for Educational Software (ThEdu'24), was a satellite event of the CADE29, part of IJCAR 2024, Nancy, France. ThEdu'24 was a vibrant workshop, with one invited talk by Jeremy Avigad (Carnegie Mellon University) and 14 submitted talks. An open call for papers was then issued and attracted 9 submissions. Eight of those submissions have been accepted by our reviewers. The resulting revised papers are collected in the present volume. The contributions in this volume are a faithful representation of the wide spectrum of ThEdu, ranging from those more focused on the automated deduction research, not losing track of the possible applications in an educational setting, to those focused on the applications, in educational settings, of automated deduction tools and methods. We, the volume editors, hope that this collection of papers will further promote the development of theorem-proving-based software and that it will allow to improve the mutual understanding between computer scientists, mathematicians, and stakeholders in education. While this volume goes to press, the next edition of the ThEdu workshop is being prepared: ThEdu'25 will be a satellite event of the 30th international Conference on Automated DEduction (CADE-30), July 28th - August 2nd, 2025, Stuttgart, Germany. 

**Abstract (ZH)**: The ThEdu 系列追求从中学直观的数学方法平滑过渡到 STEM 教育中更为形式化的处理方法，并通过利用定理证明技术的力量来支持这一过渡。以下是本卷对此项工作的贡献的简要描述。第十三届定理证明组件教育软件国际研讨会（ThEdu'24）是 CADE29（IJCAR 2024 的卫星会议）的一部分，在法国南锡举行。ThEdu'24 是一个充满活力的工作组，包括一场由 Jeremy Avigad（卡内基梅隆大学）举办的特邀演讲，以及 14 场提交的演讲。随后发出了一篇论文征稿，并收到了 9 篇投稿。其中 8 篇投稿被我们的审稿人接受。这些经过修订的论文收录在本卷中。本卷中的贡献忠实反映了 ThEdu 的广泛范围，从更多关注自动推理研究，但不忽视教育应用，到更多关注自动化推理工具和方法在教育环境中的应用。我们，作为该卷的编辑，希望这些论文的集合将进一步促进基于定理证明的软件的发展，并有助于提高计算机科学家、数学家和教育利益相关者之间的相互理解。与此同时，ThEdu 工作坊的下一届会议正在筹备中：ThEdu'25 将作为第 30 届自动推理国际会议（CADE-30）的卫星会议，在 2025 年 7 月 28 日至 8 月 2 日在德国斯图加特举行。 

---
# REVEAL: Multi-turn Evaluation of Image-Input Harms for Vision LLM 

**Title (ZH)**: REVEAL: 图像输入危害的多轮评估 for 视觉LLM 

**Authors**: Madhur Jindal, Saurabh Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2505.04673)  

**Abstract**: Vision Large Language Models (VLLMs) represent a significant advancement in artificial intelligence by integrating image-processing capabilities with textual understanding, thereby enhancing user interactions and expanding application domains. However, their increased complexity introduces novel safety and ethical challenges, particularly in multi-modal and multi-turn conversations. Traditional safety evaluation frameworks, designed for text-based, single-turn interactions, are inadequate for addressing these complexities. To bridge this gap, we introduce the REVEAL (Responsible Evaluation of Vision-Enabled AI LLMs) Framework, a scalable and automated pipeline for evaluating image-input harms in VLLMs. REVEAL includes automated image mining, synthetic adversarial data generation, multi-turn conversational expansion using crescendo attack strategies, and comprehensive harm assessment through evaluators like GPT-4o.
We extensively evaluated five state-of-the-art VLLMs, GPT-4o, Llama-3.2, Qwen2-VL, Phi3.5V, and Pixtral, across three important harm categories: sexual harm, violence, and misinformation. Our findings reveal that multi-turn interactions result in significantly higher defect rates compared to single-turn evaluations, highlighting deeper vulnerabilities in VLLMs. Notably, GPT-4o demonstrated the most balanced performance as measured by our Safety-Usability Index (SUI) followed closely by Pixtral. Additionally, misinformation emerged as a critical area requiring enhanced contextual defenses. Llama-3.2 exhibited the highest MT defect rate ($16.55 \%$) while Qwen2-VL showed the highest MT refusal rate ($19.1 \%$). 

**Abstract (ZH)**: 视觉大型语言模型（VLLMs）：负责任的评估框架（REVEAL） 

---
# Personalized Risks and Regulatory Strategies of Large Language Models in Digital Advertising 

**Title (ZH)**: 大型语言模型在数字广告中的个性化风险与监管策略 

**Authors**: Haoyang Feng, Yanjun Dai, Yuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04665)  

**Abstract**: Although large language models have demonstrated the potential for personalized advertising recommendations in experimental environments, in actual operations, how advertising recommendation systems can be combined with measures such as user privacy protection and data security is still an area worthy of in-depth discussion. To this end, this paper studies the personalized risks and regulatory strategies of large language models in digital advertising. This study first outlines the principles of Large Language Model (LLM), especially the self-attention mechanism based on the Transformer architecture, and how to enable the model to understand and generate natural language text. Then, the BERT (Bidirectional Encoder Representations from Transformers) model and the attention mechanism are combined to construct an algorithmic model for personalized advertising recommendations and user factor risk protection. The specific steps include: data collection and preprocessing, feature selection and construction, using large language models such as BERT for advertising semantic embedding, and ad recommendations based on user portraits. Then, local model training and data encryption are used to ensure the security of user privacy and avoid the leakage of personal data. This paper designs an experiment for personalized advertising recommendation based on a large language model of BERT and verifies it with real user data. The experimental results show that BERT-based advertising push can effectively improve the click-through rate and conversion rate of advertisements. At the same time, through local model training and privacy protection mechanisms, the risk of user privacy leakage can be reduced to a certain extent. 

**Abstract (ZH)**: 大型语言模型在数字广告中的个性化风险与监管策略研究 

---
# Advancing 3D Medical Image Segmentation: Unleashing the Potential of Planarian Neural Networks in Artificial Intelligence 

**Title (ZH)**: 推进3D医学图像分割：释放Planarian神经网络在人工智能中的潜力 

**Authors**: Ziyuan Huang, Kevin Huggins, Srikar Bellur  

**Link**: [PDF](https://arxiv.org/pdf/2505.04664)  

**Abstract**: Our study presents PNN-UNet as a method for constructing deep neural networks that replicate the planarian neural network (PNN) structure in the context of 3D medical image data. Planarians typically have a cerebral structure comprising two neural cords, where the cerebrum acts as a coordinator, and the neural cords serve slightly different purposes within the organism's neurological system. Accordingly, PNN-UNet comprises a Deep-UNet and a Wide-UNet as the nerve cords, with a densely connected autoencoder performing the role of the brain. This distinct architecture offers advantages over both monolithic (UNet) and modular networks (Ensemble-UNet). Our outcomes on a 3D MRI hippocampus dataset, with and without data augmentation, demonstrate that PNN-UNet outperforms the baseline UNet and several other UNet variants in image segmentation. 

**Abstract (ZH)**: 我们的研究提出了PNN-UNet方法，用于在3D医学图像数据中构建模仿扁虫神经网络（PNN）结构的深度神经网络。PNN-UNet包括一个Deep-UNet和一个Wide-UNet作为神经索，而一个密集连接的自编码器执行大脑的功能。这种独特的架构在整体（UNet）和模块化网络（Ensemble-UNet）之上具有优势。在无数据增强和有数据增强的3D MRI海马体数据集上的结果表明，PNN-UNet在图像分割方面优于基线UNet和其他几种UNet变体。 

---
# Advancing Conversational Diagnostic AI with Multimodal Reasoning 

**Title (ZH)**: 基于多模态推理推进对话诊断AI 

**Authors**: Khaled Saab, Jan Freyberg, Chunjong Park, Tim Strother, Yong Cheng, Wei-Hung Weng, David G.T. Barrett, David Stutz, Nenad Tomasev, Anil Palepu, Valentin Liévin, Yash Sharma, Roma Ruparel, Abdullah Ahmed, Elahe Vedadi, Kimberly Kanada, Cian Hughes, Yun Liu, Geoff Brown, Yang Gao, Sean Li, S. Sara Mahdavi, James Manyika, Katherine Chou, Yossi Matias, Avinatan Hassidim, Dale R. Webster, Pushmeet Kohli, S.M. Ali Eslami, Joëlle Barral, Adam Rodman, Vivek Natarajan, Mike Schaekermann, Tao Tu, Alan Karthikesalingam, Ryutaro Tanno  

**Link**: [PDF](https://arxiv.org/pdf/2505.04653)  

**Abstract**: Large Language Models (LLMs) have demonstrated great potential for conducting diagnostic conversations but evaluation has been largely limited to language-only interactions, deviating from the real-world requirements of remote care delivery. Instant messaging platforms permit clinicians and patients to upload and discuss multimodal medical artifacts seamlessly in medical consultation, but the ability of LLMs to reason over such data while preserving other attributes of competent diagnostic conversation remains unknown. Here we advance the conversational diagnosis and management performance of the Articulate Medical Intelligence Explorer (AMIE) through a new capability to gather and interpret multimodal data, and reason about this precisely during consultations. Leveraging Gemini 2.0 Flash, our system implements a state-aware dialogue framework, where conversation flow is dynamically controlled by intermediate model outputs reflecting patient states and evolving diagnoses. Follow-up questions are strategically directed by uncertainty in such patient states, leading to a more structured multimodal history-taking process that emulates experienced clinicians. We compared AMIE to primary care physicians (PCPs) in a randomized, blinded, OSCE-style study of chat-based consultations with patient actors. We constructed 105 evaluation scenarios using artifacts like smartphone skin photos, ECGs, and PDFs of clinical documents across diverse conditions and demographics. Our rubric assessed multimodal capabilities and other clinically meaningful axes like history-taking, diagnostic accuracy, management reasoning, communication, and empathy. Specialist evaluation showed AMIE to be superior to PCPs on 7/9 multimodal and 29/32 non-multimodal axes (including diagnostic accuracy). The results show clear progress in multimodal conversational diagnostic AI, but real-world translation needs further research. 

**Abstract (ZH)**: 大型语言模型（LLMs）在进行诊断对话方面展现了巨大潜力，但目前的评估主要集中在语言交互上，偏离了远程医疗服务的实际需求。即时通讯平台允许医生和患者在医疗咨询中无缝上传和讨论多模态医疗 artifacts，但大型语言模型在处理这类数据的同时保留高效诊断对话的其他特征的能力仍未知。通过引入一种新的能力来收集和解释多模态数据，并在咨询过程中精确地推理这些数据，我们提升了综合医学智能探索器（AMIE）的对话诊断和管理性能。利用Gemini 2.0 Flash，我们的系统实现了具备状态感知的对话框架，其中对话流程动态受到反映患者状态和演化诊断的中间模型输出的控制。策略性地引导后续问题依据患者状态的不确定性，这会产生一个结构化的多模态病史采集过程，逼近临床经验丰富的医生。我们在一项随机、盲法、类似OSCE的基于聊天的咨询研究中将AMIE与全科医生（PCPs）进行了比较。我们构建了105个评估场景，使用智能手机皮肤照片、心电图以及临床文件的PDF等多样的artifact，覆盖不同的条件和人口统计信息。我们的评分标准评估了多模态能力及其他具有临床意义的维度，如病史采集、诊断准确性、管理推理、沟通和同理心。专科评估显示，AMIE在7/9个多模态轴和29/32个非多模态轴（包括诊断准确性）上优于全科医生。结果表明，在多模态对话诊断AI方面取得了明显的进步，但实际应用仍需进一步研究。 

---
# Multimodal Benchmarking and Recommendation of Text-to-Image Generation Models 

**Title (ZH)**: 多模态文本到图像生成模型的基准测试与推荐 

**Authors**: Kapil Wanaskar, Gaytri Jena, Magdalini Eirinaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.04650)  

**Abstract**: This work presents an open-source unified benchmarking and evaluation framework for text-to-image generation models, with a particular focus on the impact of metadata augmented prompts. Leveraging the DeepFashion-MultiModal dataset, we assess generated outputs through a comprehensive set of quantitative metrics, including Weighted Score, CLIP (Contrastive Language Image Pre-training)-based similarity, LPIPS (Learned Perceptual Image Patch Similarity), FID (Frechet Inception Distance), and retrieval-based measures, as well as qualitative analysis. Our results demonstrate that structured metadata enrichments greatly enhance visual realism, semantic fidelity, and model robustness across diverse text-to-image architectures. While not a traditional recommender system, our framework enables task-specific recommendations for model selection and prompt design based on evaluation metrics. 

**Abstract (ZH)**: 本研究提出了一种开源统一的文本到图像生成模型基准测试与评估框架，特别关注元数据增强提示的影响。通过利用DeepFashion-MultiModal数据集，我们使用一系列定量指标进行了评估，包括加权分数、CLIP基于的相似性、LPIPS（学习感知图像块相似性）、FID（弗谢赫特 inception 距离）以及检索基评估指标，并结合定性分析。我们的结果表明，结构化的元数据增强显著提升了跨多种文本到图像架构的视觉真实性、语义准确性和模型稳健性。虽然不是传统的推荐系统，但该框架基于评估指标提供了任务特定的模型选择和提示设计建议。 

---
# Rethinking Multimodal Sentiment Analysis: A High-Accuracy, Simplified Fusion Architecture 

**Title (ZH)**: 重塑多模态情感分析：一种高精度简化融合架构 

**Authors**: Nischal Mandal, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.04642)  

**Abstract**: Multimodal sentiment analysis, a pivotal task in affective computing, seeks to understand human emotions by integrating cues from language, audio, and visual signals. While many recent approaches leverage complex attention mechanisms and hierarchical architectures, we propose a lightweight, yet effective fusion-based deep learning model tailored for utterance-level emotion classification. Using the benchmark IEMOCAP dataset, which includes aligned text, audio-derived numeric features, and visual descriptors, we design a modality-specific encoder using fully connected layers followed by dropout regularization. The modality-specific representations are then fused using simple concatenation and passed through a dense fusion layer to capture cross-modal interactions. This streamlined architecture avoids computational overhead while preserving performance, achieving a classification accuracy of 92% across six emotion categories. Our approach demonstrates that with careful feature engineering and modular design, simpler fusion strategies can outperform or match more complex models, particularly in resource-constrained environments. 

**Abstract (ZH)**: 多模态情感分析是情感计算中的一个关键任务，旨在通过综合语言、音频和视觉信号来理解人类情感。虽然许多最近的方法利用了复杂的注意力机制和分层结构，我们提出了一种轻量级但有效的基于融合的深度学习模型，专门用于短语级别情感分类。利用包含对齐文本、音频衍生数值特征和视觉描述的基准IEMOCAP数据集，我们设计了模态特定编码器，使用全连接层后接 dropout 正则化。模态特定表示随后通过简单的串联融合，并通过密集融合层捕捉跨模态交互。这种精简的架构避免了计算开销同时保持了性能，实现了六个情感类别上的分类准确率92%。我们的方法证明了在精心特征工程和模块化设计下，简单的融合策略可以在资源受限环境中超越或匹配更复杂模型。 

---
# Language translation, and change of accent for speech-to-speech task using diffusion model 

**Title (ZH)**: 使用扩散模型进行语言翻译及语音换音的语音到语音任务 

**Authors**: Abhishek Mishra, Ritesh Sur Chowdhury, Vartul Bahuguna, Isha Pandey, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04639)  

**Abstract**: Speech-to-speech translation (S2ST) aims to convert spoken input in one language to spoken output in another, typically focusing on either language translation or accent adaptation. However, effective cross-cultural communication requires handling both aspects simultaneously - translating content while adapting the speaker's accent to match the target language context. In this work, we propose a unified approach for simultaneous speech translation and change of accent, a task that remains underexplored in current literature. Our method reformulates the problem as a conditional generation task, where target speech is generated based on phonemes and guided by target speech features. Leveraging the power of diffusion models, known for high-fidelity generative capabilities, we adapt text-to-image diffusion strategies by conditioning on source speech transcriptions and generating Mel spectrograms representing the target speech with desired linguistic and accentual attributes. This integrated framework enables joint optimization of translation and accent adaptation, offering a more parameter-efficient and effective model compared to traditional pipelines. 

**Abstract (ZH)**: 跨文化口语翻译与口音转换（S2ST）旨在将一种语言的口语输入转换为另一种语言的口语输出，通常专注于语言翻译或口音适应。然而，有效的跨文化沟通要求同时处理这两个方面：内容翻译与说话人口音的适配以匹配目标语言背景。在本文中，我们提出了一种统一的方法，用于同时进行口语翻译和口音转换，这是一个当前文献中研究不足的任务。我们的方法将问题重新表述为条件生成任务，其中目标口语基于音素生成并由目标口语特征引导。利用扩散模型的强大生成能力，我们通过在源口语转录的基础上进行条件设置，并生成表示具有所需语言和口音属性的目标口语的梅尔频谱图，来适应文本到图像的扩散策略。通过这种集成框架，可以实现翻译和口音转换的联合优化，相比传统流水线方法，提供了一个更具参数效率和有效性的模型。 

---
# Adaptive Token Boundaries: Integrating Human Chunking Mechanisms into Multimodal LLMs 

**Title (ZH)**: 自适应TOKEN边界：将人类段落划分机制集成到多模态LLMs中 

**Authors**: Dongxing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04637)  

**Abstract**: Recent advancements in multimodal large language models (MLLMs) have demonstrated remarkable capabilities in processing diverse data types, yet significant disparities persist between human cognitive processes and computational approaches to multimodal information integration. This research presents a systematic investigation into the parallels between human cross-modal chunking mechanisms and token representation methodologies in MLLMs. Through empirical studies comparing human performance patterns with model behaviors across visual-linguistic tasks, we demonstrate that conventional static tokenization schemes fundamentally constrain current models' capacity to simulate the dynamic, context-sensitive nature of human information processing. We propose a novel framework for dynamic cross-modal tokenization that incorporates adaptive boundaries, hierarchical representations, and alignment mechanisms grounded in cognitive science principles. Quantitative evaluations demonstrate that our approach yields statistically significant improvements over state-of-the-art models on benchmark tasks (+7.8% on Visual Question Answering, +5.3% on Complex Scene Description) while exhibiting more human-aligned error patterns and attention distributions. These findings contribute to the theoretical understanding of the relationship between human cognition and artificial intelligence, while providing empirical evidence for developing more cognitively plausible AI systems. 

**Abstract (ZH)**: 最近在多模态大语言模型方面取得的进展展示了在处理多种数据类型方面的卓越能力，但人类认知过程与计算方法在多模态信息集成方面的差距依然显著。本研究系统探讨了人类跨模态片段化机制与多模态大语言模型中词元表示方法之间的相似性。通过比较人类和模型在视觉语言任务中的绩效模式，我们证明了传统的静态词元划分方案从根本上限制了当前模型模拟人类信息处理的动态性和上下文敏感性能力。我们提出了一种新的动态跨模态词元化框架，该框架结合了适应性边界、分层表示和基于认知科学原理的对齐机制。定量评估表明，我们的方法在基准任务中比最先进的模型表现出了统计意义上的显著提升（视觉问答任务上+7.8%，复杂场景描述任务上+5.3%），并且展现出更符合人类错误模式和注意力分布的特点。这些发现不仅丰富了人类认知与人工智能关系的理论理解，还为开发更加认知合理的AI系统提供了实证支持。 

---
# From Dialect Gaps to Identity Maps: Tackling Variability in Speaker Verification 

**Title (ZH)**: 从方言差距到身份映射：应对说话人验证中的变异性 

**Authors**: Abdulhady Abas Abdullah, Soran Badawi, Dana A. Abdullah, Dana Rasul Hamad, Hanan Abdulrahman Taher, Sabat Salih Muhamad, Aram Mahmood Ahmed, Bryar A. Hassan, Sirwan Abdolwahed Aula, Tarik A. Rashid  

**Link**: [PDF](https://arxiv.org/pdf/2505.04629)  

**Abstract**: The complexity and difficulties of Kurdish speaker detection among its several dialects are investigated in this work. Because of its great phonetic and lexical differences, Kurdish with several dialects including Kurmanji, Sorani, and Hawrami offers special challenges for speaker recognition systems. The main difficulties in building a strong speaker identification system capable of precisely identifying speakers across several dialects are investigated in this work. To raise the accuracy and dependability of these systems, it also suggests solutions like sophisticated machine learning approaches, data augmentation tactics, and the building of thorough dialect-specific corpus. The results show that customized strategies for every dialect together with cross-dialect training greatly enhance recognition performance. 

**Abstract (ZH)**: Kurdish方言 Speaker识别的复杂性与困难及其解决方案研究 

---
# How Social is It? A Benchmark for LLMs' Capabilities in Multi-user Multi-turn Social Agent Tasks 

**Title (ZH)**: 多用户多轮社会代理任务能力基准：它有多社会？ 

**Authors**: Yusen Wu, Junwu Xiong, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04628)  

**Abstract**: Expanding the application of large language models (LLMs) to societal life, instead of primary function only as auxiliary assistants to communicate with only one person at a time, necessitates LLMs' capabilities to independently play roles in multi-user, multi-turn social agent tasks within complex social settings. However, currently the capability has not been systematically measured with available benchmarks. To address this gap, we first introduce an agent task leveling framework grounded in sociological principles. Concurrently, we propose a novel benchmark, How Social Is It (we call it HSII below), designed to assess LLM's social capabilities in comprehensive social agents tasks and benchmark representative models. HSII comprises four stages: format parsing, target selection, target switching conversation, and stable conversation, which collectively evaluate the communication and task completion capabilities of LLMs within realistic social interaction scenarios dataset, HSII-Dataset. The dataset is derived step by step from news dataset. We perform an ablation study by doing clustering to the dataset. Additionally, we investigate the impact of chain of thought (COT) method on enhancing LLMs' social performance. Since COT cost more computation, we further introduce a new statistical metric, COT-complexity, to quantify the efficiency of certain LLMs with COTs for specific social tasks and strike a better trade-off between measurement of correctness and efficiency. Various results of our experiments demonstrate that our benchmark is well-suited for evaluating social skills in LLMs. 

**Abstract (ZH)**: 扩展大型语言模型在社会生活中的应用，而非仅作为辅助助手与单一用户交流，要求LLMs具备在复杂社会环境中独立承担多用户、多轮社会智能代理任务的能力。然而，目前尚无系统化的基准来衡量这一能力。为填补这一空白，我们首先引入了一个基于社会学原理的任务层级框架。同时，我们提出了一种新的基准——How Social Is It（我们将其简称为HSII）——用于评估LLMs在综合社会智能代理任务中的社会能力，并以此作为代表性模型的基准。HSII包括四个阶段：格式解析、目标选择、目标切换对话和稳定对话，旨在评估LLMs在现实社会交互场景中的沟通和任务完成能力，数据集为HSII-Dataset。该数据集从新闻数据集逐步衍生而来。我们通过聚类分析对该数据集进行了消融研究，并探讨了思维链方法（COT）对提升LLMs社会表现的影响。由于COT消耗更多计算资源，我们进一步引入了一个新的统计指标——COT复杂性，以量化具有特定COT的LLMs在特定社会任务中的效率，并在正确性和效率之间找到更好的权衡。各种实验结果表明，我们的基准对评估LLMs的社会技能非常合适。 

---
# Toward Holistic Evaluation of Recommender Systems Powered by Generative Models 

**Title (ZH)**: 面向生成模型驱动的推荐系统综合性评估 

**Authors**: Yashar Deldjoo, Nikhil Mehta, Maheswaran Sathiamoorthy, Shuai Zhang, Pablo Castells, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2504.06667)  

**Abstract**: Recommender systems powered by generative models (Gen-RecSys) extend beyond classical item ranking by producing open-ended content, which simultaneously unlocks richer user experiences and introduces new risks. On one hand, these systems can enhance personalization and appeal through dynamic explanations and multi-turn dialogues. On the other hand, they might venture into unknown territory-hallucinating nonexistent items, amplifying bias, or leaking private information. Traditional accuracy metrics cannot fully capture these challenges, as they fail to measure factual correctness, content safety, or alignment with user intent.
This paper makes two main contributions. First, we categorize the evaluation challenges of Gen-RecSys into two groups: (i) existing concerns that are exacerbated by generative outputs (e.g., bias, privacy) and (ii) entirely new risks (e.g., item hallucinations, contradictory explanations). Second, we propose a holistic evaluation approach that includes scenario-based assessments and multi-metric checks-incorporating relevance, factual grounding, bias detection, and policy compliance. Our goal is to provide a guiding framework so researchers and practitioners can thoroughly assess Gen-RecSys, ensuring effective personalization and responsible deployment. 

**Abstract (ZH)**: 基于生成模型的推荐系统（Gen-RecSys）超越了经典项排序，通过生成开放式内容同时提升了用户体验并引入了新的风险。一方面，这些系统可以通过动态解释和多轮对话增强个性化和吸引力。另一方面，它们可能闯入未知领域，如虚构不存在的项目、放大偏见或泄露私人信息。传统准确性指标无法充分捕捉这些挑战，因为它们未能衡量事实正确性、内容安全性或与用户意图的一致性。

本文主要贡献两个方面：首先，我们将Gen-RecSys的评估挑战分为两类：（i）生成输出加剧的现有问题（如偏见、隐私），和（ii）全新的风险（如项目幻觉、矛盾的解释）。其次，我们提出了一种综合的评估方法，包括基于场景的评估和多指标检查，涵盖了相关性、事实基础、偏见检测和政策合规性。我们的目标是提供一个指导框架，使研究者和实践者能够全面评估Gen-RecSys，确保有效的个性化和负责任的部署。 

---
