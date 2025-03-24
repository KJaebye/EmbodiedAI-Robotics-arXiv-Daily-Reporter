# HCAST: Human-Calibrated Autonomy Software Tasks 

**Title (ZH)**: Human-calibrated Autonomy Software Tasks 

**Authors**: David Rein, Joel Becker, Amy Deng, Seraphina Nix, Chris Canal, Daniel O'Connel, Pip Arnott, Ryan Bloom, Thomas Broadley, Katharyn Garcia, Brian Goodrich, Max Hasin, Sami Jawhar, Megan Kinniment, Thomas Kwa, Aron Lajko, Nate Rush, Lucas Jun Koba Sato, Sydney Von Arx, Ben West, Lawrence Chan, Elizabeth Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2503.17354)  

**Abstract**: To understand and predict the societal impacts of highly autonomous AI systems, we need benchmarks with grounding, i.e., metrics that directly connect AI performance to real-world effects we care about. We present HCAST (Human-Calibrated Autonomy Software Tasks), a benchmark of 189 machine learning engineering, cybersecurity, software engineering, and general reasoning tasks. We collect 563 human baselines (totaling over 1500 hours) from people skilled in these domains, working under identical conditions as AI agents, which lets us estimate that HCAST tasks take humans between one minute and 8+ hours. Measuring the time tasks take for humans provides an intuitive metric for evaluating AI capabilities, helping answer the question "can an agent be trusted to complete a task that would take a human X hours?" We evaluate the success rates of AI agents built on frontier foundation models, and we find that current agents succeed 70-80% of the time on tasks that take humans less than one hour, and less than 20% of the time on tasks that take humans more than 4 hours. 

**Abstract (ZH)**: 为了理解并预测高度自主人工智能系统对社会的影响，我们需要具备接地性的基准，即直接将AI性能与我们关心的实际效果连接起来的指标。我们提出了HCAST（Human-Calibrated Autonomy Software Tasks），一个包含189项机器学习工程、网络安全、软件工程和一般推理任务的基准。我们从这些领域中熟练的人们那里收集了563项人类基准（总计超过1500小时的工作时间），工作条件与AI代理相同，以便估计HCAST任务所需的时间为1分钟至8小时以上。测量任务所需的时间为一个直观的指标，用于评估AI能力，帮助回答“一个代理能否被信任完成一个需要人类X小时的任务？”的问题。我们评估了基于前沿基础模型构建的AI代理的成功率，并发现当前代理在人类可以在不到一小时内完成的任务中成功率达到70-80%，而在人类需要超过4小时才能完成的任务中成功率低于20%。 

---
# Capturing Individual Human Preferences with Reward Features 

**Title (ZH)**: 基于奖励特征捕捉个体人类偏好 

**Authors**: André Barreto, Vincent Dumoulin, Yiran Mao, Nicolas Perez-Nieves, Bobak Shahriari, Yann Dauphin, Doina Precup, Hugo Larochelle  

**Link**: [PDF](https://arxiv.org/pdf/2503.17338)  

**Abstract**: Reinforcement learning from human feedback usually models preferences using a reward model that does not distinguish between people. We argue that this is unlikely to be a good design choice in contexts with high potential for disagreement, like in the training of large language models. We propose a method to specialise a reward model to a person or group of people. Our approach builds on the observation that individual preferences can be captured as a linear combination of a set of general reward features. We show how to learn such features and subsequently use them to quickly adapt the reward model to a specific individual, even if their preferences are not reflected in the training data. We present experiments with large language models comparing the proposed architecture with a non-adaptive reward model and also adaptive counterparts, including models that do in-context personalisation. Depending on how much disagreement there is in the training data, our model either significantly outperforms the baselines or matches their performance with a simpler architecture and more stable training. 

**Abstract (ZH)**: 从人类反馈中进行强化学习通常使用一个不区分个体的奖励模型。我们在大型语言模型训练等存在高争议风险的背景下，认为这种设计可能不是一个好的选择。我们提出了一种将奖励模型专用于特定个人或群体的方法。我们基于个体偏好可以表示为一组通用奖励特征线性组合的观察，展示了一种方法来学习这些特征，并利用它们快速适应特定个体的奖励模型，即使个体的偏好在训练数据中未得到反映。我们通过与非自适应奖励模型以及各种自适应模型（包括进行上下文个性化调整的模型）进行大型语言模型实验，展示了该架构的表现。当训练数据中的分歧较大时，我们的模型显著优于基准模型；当分歧较小时，通过更简单的架构和更稳定的训练，我们的模型能够与基准模型取得相似的性能。 

---
# Breaking the Symmetries of Indistinguishable Objects 

**Title (ZH)**: 破坏不可区分对象的对称性 

**Authors**: Ozgur Akgun, Mun See Chang, Ian P. Gent, Christopher Jefferson  

**Link**: [PDF](https://arxiv.org/pdf/2503.17251)  

**Abstract**: Indistinguishable objects often occur when modelling problems in constraint programming, as well as in other related paradigms. They occur when objects can be viewed as being drawn from a set of unlabelled objects, and the only operation allowed on them is equality testing. For example, the golfers in the social golfer problem are indistinguishable. If we do label the golfers, then any relabelling of the golfers in one solution gives another valid solution. Therefore, we can regard the symmetric group of size $n$ as acting on a set of $n$ indistinguishable objects. In this paper, we show how we can break the symmetries resulting from indistinguishable objects. We show how symmetries on indistinguishable objects can be defined properly in complex types, for example in a matrix indexed by indistinguishable objects. We then show how the resulting symmetries can be broken correctly. In Essence, a high-level modelling language, indistinguishable objects are encapsulated in "unnamed types". We provide an implementation of complete symmetry breaking for unnamed types in Essence. 

**Abstract (ZH)**: 不可区分的对象在约束编程及其相关范式中建模问题时经常出现。当对象可以被视为未标记对象集合中的元素，且唯一允许的操作仅为相等性测试时，就会出现不可区分的对象。例如，在社交高尔夫问题中，高尔夫球手是不可区分的。如果我们给高尔夫球手加上标签，那么在某一解中的高尔夫球手的任何重新标记都将给出另一有效解。因此，我们可以将大小为 \( n \) 的对称群视为作用于 \( n \) 个不可区分对象集合上的群。在本文中，我们展示了如何打破由不可区分对象导致的对称性。我们展示了如何在复杂类型中，例如由不可区分对象索引的矩阵中，正确定义不可区分对象上的对称性，并展示如何正确打破这些对称性。在Essence这一高级建模语言中，不可区分的对象被封装在“无名类型”中。我们在Essence中提供了一种完整的对称性打破实现用于无名类型。 

---
# A Guide to Bayesian Networks Software Packages for Structure and Parameter Learning -- 2025 Edition 

**Title (ZH)**: Bayesian网络软件包指南——2025年版 

**Authors**: Joverlyn Gaudillo, Nicole Astrologo, Fabio Stella, Enzo Acerbi, Francesco Canonaco  

**Link**: [PDF](https://arxiv.org/pdf/2503.17025)  

**Abstract**: A representation of the cause-effect mechanism is needed to enable artificial intelligence to represent how the world works. Bayesian Networks (BNs) have proven to be an effective and versatile tool for this task. BNs require constructing a structure of dependencies among variables and learning the parameters that govern these relationships. These tasks, referred to as structural learning and parameter learning, are actively investigated by the research community, with several algorithms proposed and no single method having established itself as standard. A wide range of software, tools, and packages have been developed for BNs analysis and made available to academic researchers and industry practitioners. As a consequence of having no one-size-fits-all solution, moving the first practical steps and getting oriented into this field is proving to be challenging to outsiders and beginners. In this paper, we review the most relevant tools and software for BNs structural and parameter learning to date, providing our subjective recommendations directed to an audience of beginners. In addition, we provide an extensive easy-to-consult overview table summarizing all software packages and their main features. By improving the reader understanding of which available software might best suit their needs, we improve accessibility to the field and make it easier for beginners to take their first step into it. 

**Abstract (ZH)**: 一种表示因果机制的方法对于使人工智能能够代表世界是如何运作的至关重要。贝叶斯网络（BNs）已被证明是一种有效且多功能的工具来完成这一任务。构建变量之间依赖关系的结构并学习控制这些关系的参数是必要的任务，被称为结构学习和参数学习，这些任务正在研究界中积极研究之中，已有多种算法被提出，但尚未有单一方法确立为标准。为BNs分析开发了多种软件、工具和包，并已提供给学术研究人员和工业从业者使用。由于没有一刀切的解决方案，对于门外汉和初学者而言，迈出第一步并进入这一领域变得具有挑战性。在本文中，我们回顾了迄今为止与BNs结构和参数学习相关的最相关工具和软件，为初学者提供了我们主观的推荐。此外，我们提供了涵盖所有软件包及其主要功能的详尽且易于查阅的概述表。通过提高读者对可用软件的了解，使他们能够更准确地满足自身需求，我们提高了该领域的可访问性，并使初学者更易于迈入这一领域。 

---
# Real-Time Diffusion Policies for Games: Enhancing Consistency Policies with Q-Ensembles 

**Title (ZH)**: 实时扩散策略在游戏中的应用：结合Q-集成提升一致性策略 

**Authors**: Ruoqi Zhang, Ziwei Luo, Jens Sjölund, Per Mattsson, Linus Gisslén, Alessandro Sestini  

**Link**: [PDF](https://arxiv.org/pdf/2503.16978)  

**Abstract**: Diffusion models have shown impressive performance in capturing complex and multi-modal action distributions for game agents, but their slow inference speed prevents practical deployment in real-time game environments. While consistency models offer a promising approach for one-step generation, they often suffer from training instability and performance degradation when applied to policy learning. In this paper, we present CPQE (Consistency Policy with Q-Ensembles), which combines consistency models with Q-ensembles to address these this http URL leverages uncertainty estimation through Q-ensembles to provide more reliable value function approximations, resulting in better training stability and improved performance compared to classic double Q-network methods. Our extensive experiments across multiple game scenarios demonstrate that CPQE achieves inference speeds of up to 60 Hz -- a significant improvement over state-of-the-art diffusion policies that operate at only 20 Hz -- while maintaining comparable performance to multi-step diffusion approaches. CPQE consistently outperforms state-of-the-art consistency model approaches, showing both higher rewards and enhanced training stability throughout the learning process. These results indicate that CPQE offers a practical solution for deploying diffusion-based policies in games and other real-time applications where both multi-modal behavior modeling and rapid inference are critical requirements. 

**Abstract (ZH)**: CPQE：一致性策略与Q集成结合解决扩散模型的应用问题 

---
# Neural-Guided Equation Discovery 

**Title (ZH)**: 神经引导的方程发现 

**Authors**: Jannis Brugger, Mattia Cerrato, David Richter, Cedric Derstroff, Daniel Maninger, Mira Mezini, Stefan Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2503.16953)  

**Abstract**: Deep learning approaches are becoming increasingly attractive for equation discovery. We show the advantages and disadvantages of using neural-guided equation discovery by giving an overview of recent papers and the results of experiments using our modular equation discovery system MGMT ($\textbf{M}$ulti-Task $\textbf{G}$rammar-Guided $\textbf{M}$onte-Carlo $\textbf{T}$ree Search for Equation Discovery). The system uses neural-guided Monte-Carlo Tree Search (MCTS) and supports both supervised and reinforcement learning, with a search space defined by a context-free grammar. We summarize seven desirable properties of equation discovery systems, emphasizing the importance of embedding tabular data sets for such learning approaches. Using the modular structure of MGMT, we compare seven architectures (among them, RNNs, CNNs, and Transformers) for embedding tabular datasets on the auxiliary task of contrastive learning for tabular data sets on an equation discovery task. For almost all combinations of modules, supervised learning outperforms reinforcement learning. Moreover, our experiments indicate an advantage of using grammar rules as action space instead of tokens. Two adaptations of MCTS -- risk-seeking MCTS and AmEx-MCTS -- can improve equation discovery with that kind of search. 

**Abstract (ZH)**: 深度学习方法在方程发现中的应用日益吸引人。我们通过概述近期论文并使用我们模块化的方程发现系统MGMT（基于上下文无关文法的多任务语法引导蒙特卡洛树搜索方程发现）的实验结果，展示了神经引导方程发现的优势与局限。我们总结了方程发现系统所需的七种 desirable 属性，强调了嵌入表格数据集对于此类学习方法的重要性。通过MGMT的模块化结构，我们将七个架构（包括RNN、CNN和Transformer）进行比较，用于嵌入表格数据集的辅助任务（对比学习）以表格数据集进行方程发现任务。对于大多数模块组合，监督学习优于强化学习。此外，我们的实验表明，使用语法规则作为动作空间优于使用标记。两种MCTS的改编——风险寻求MCTS和AmEx-MCTS——可以提高这种搜索方式下的方程发现能力。 

---
# Interpretable Machine Learning for Oral Lesion Diagnosis through Prototypical Instances Identification 

**Title (ZH)**: 基于原型实例识别的可解释机器学习在口腔病损诊断中的应用 

**Authors**: Alessio Cascione, Mattia Setzu, Federico A. Galatolo, Mario G.C.A. Cimino, Riccardo Guidotti  

**Link**: [PDF](https://arxiv.org/pdf/2503.16938)  

**Abstract**: Decision-making processes in healthcare can be highly complex and challenging. Machine Learning tools offer significant potential to assist in these processes. However, many current methodologies rely on complex models that are not easily interpretable by experts. This underscores the need to develop interpretable models that can provide meaningful support in clinical decision-making. When approaching such tasks, humans typically compare the situation at hand to a few key examples and representative cases imprinted in their memory. Using an approach which selects such exemplary cases and grounds its predictions on them could contribute to obtaining high-performing interpretable solutions to such problems. To this end, we evaluate PivotTree, an interpretable prototype selection model, on an oral lesion detection problem, specifically trying to detect the presence of neoplastic, aphthous and traumatic ulcerated lesions from oral cavity images. We demonstrate the efficacy of using such method in terms of performance and offer a qualitative and quantitative comparison between exemplary cases and ground-truth prototypes selected by experts. 

**Abstract (ZH)**: 医疗保健中的决策过程可能极为复杂且具挑战性。机器学习工具在这些过程中提供了显著的辅助潜力。然而，许多现有方法依赖于难于专家解释的复杂模型。这突显了开发可解释模型的需求，这些模型可以在临床决策中提供有意义的支持。在处理这类任务时，人类通常将其手中的情况与记忆中的一些关键示例和代表性案例进行对比。采用选择此类典型示例并基于它们进行预测的方法，有助于获得高性能且可解释的解决方案。为此，我们评估了PivotTree这一可解释原型选择模型在口腔病损检测问题上的应用，具体目标是从口腔腔隙图像中检测瘤性、复发性和外伤性溃疡性病损。我们展示了该方法在性能上的有效性，并提供了专家选定的典型示例与真实原型之间的定性和定量比较。 

---
# A New Segment Routing method with Swap Node Selection Strategy Based on Deep Reinforcement Learning for Software Defined Network 

**Title (ZH)**: 基于深度强化学习的换节点选择策略的新型段路由方法 

**Authors**: Miao Ye, Jihao Zheng, Qiuxiang Jiang, Yuan Huang, Ziheng Wang, Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16914)  

**Abstract**: The existing segment routing (SR) methods need to determine the routing first and then use path segmentation approaches to select swap nodes to form a segment routing path (SRP). They require re-segmentation of the path when the routing changes. Furthermore, they do not consider the flow table issuance time, which cannot maximize the speed of issuance flow table. To address these issues, this paper establishes an optimization model that can simultaneously form routing strategies and path segmentation strategies for selecting the appropriate swap nodes to reduce flow table issuance time. It also designs an intelligent segment routing algorithm based on deep reinforcement learning (DRL-SR) to solve the proposed model. First, a traffic matrix is designed as the state space for the deep reinforcement learning agent; this matrix includes multiple QoS performance indicators, flow table issuance time overhead and SR label stack depth. Second, the action selection strategy and corresponding reward function are designed, where the agent selects the next node considering the routing; in addition, the action selection strategy whether the newly added node is selected as the swap node and the corresponding reward function are designed considering the time cost factor for the controller to issue the flow table to the swap node. Finally, a series of experiments and their results show that, compared with the existing methods, the designed segmented route optimization model and the intelligent solution algorithm (DRL-SR) can reduce the time overhead required to complete the segmented route establishment task while optimizing performance metrics such as throughput, delays and packet losses. 

**Abstract (ZH)**: 基于深度强化学习的现有分段路由优化模型与智能算法 

---
# MAPS: A Multi-Agent Framework Based on Big Seven Personality and Socratic Guidance for Multimodal Scientific Problem Solving 

**Title (ZH)**: MAPS：基于“大七”人格和苏格拉底引导的多模态科学问题解决多agent框架 

**Authors**: Jian Zhang, Zhiyuan Wang, Zhangqi Wang, Xinyu Zhang, Fangzhi Xu, Qika Lin, Rui Mao, Erik Cambria, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16905)  

**Abstract**: Multimodal scientific problems (MSPs) involve complex issues that require the integration of multiple modalities, such as text and diagrams, presenting a significant challenge in artificial intelligence. While progress has been made in addressing traditional scientific problems, MSPs still face two primary issues: the challenge of multi-modal comprehensive reasoning in scientific problem-solving and the lack of reflective and rethinking capabilities. To address these issues, we introduce a Multi-Agent framework based on the Big Seven Personality and Socratic guidance (MAPS). This framework employs seven distinct agents that leverage feedback mechanisms and the Socratic method to guide the resolution of MSPs. To tackle the first issue, we propose a progressive four-agent solving strategy, where each agent focuses on a specific stage of the problem-solving process. For the second issue, we introduce a Critic agent, inspired by Socratic questioning, which prompts critical thinking and stimulates autonomous learning. We conduct extensive experiments on the EMMA, Olympiad, and MathVista datasets, achieving promising results that outperform the current SOTA model by 15.84% across all tasks. Meanwhile, the additional analytical experiments also verify the model's progress as well as generalization ability. 

**Abstract (ZH)**: 多模态科学问题（MSPs）涉及复杂问题，需要整合多种模态，如文本和图表，这为人工智能带来了显著挑战。尽管在解决传统科学问题方面取得了进展，MSPs仍然面临两个主要问题：科学问题解决中的多模态综合推理挑战以及缺乏反思和重思能力。为解决这些问题，我们引入了一种基于Big Seven人格和苏格拉底引导的多代理框架（MAPS）。该框架利用七个不同的代理并采用反馈机制和苏格拉底方法来引导MSPs的解决过程。为应对第一个问题，我们提出了一种逐步四代理解决策略，每个代理专注于问题解决过程中的特定阶段。为应对第二个问题，我们引入了评论代理（Critic agent），受到苏格拉底提问的启发，它促进批判性思维并激发自主学习。我们在EMMA、奥林匹亚和MathVista数据集上进行了广泛的实验，取得了令人鼓舞的结果，与当前的SOTA模型相比，在所有任务上性能提高15.84%。同时，额外的分析实验也验证了模型的进步及其通用性。 

---
# MARS: A Multi-Agent Framework Incorporating Socratic Guidance for Automated Prompt Optimization 

**Title (ZH)**: MARS：一个融合苏格拉底引导的多Agent框架，用于自动化提示优化 

**Authors**: Jian Zhang, Zhangqi Wang, Haiping Zhu, Jun Liu, Qika Lin, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2503.16874)  

**Abstract**: The basic question-answering format of large language models involves inputting a prompt and receiving a response, and the quality of the prompt directly impacts the effectiveness of the response. Automated Prompt Optimization (APO) aims to break free from the cognitive biases of manually designed prompts and explores a broader design space for prompts. However, existing APO methods suffer from limited flexibility of fixed templates and inefficient search in prompt spaces as key issues. To this end, we propose a Multi-Agent framework Incorporating Socratic guidance (MARS), which utilizes multi-agent fusion technology for automatic planning, with gradual continuous optimization and evaluation. Specifically, MARS comprises seven agents, each with distinct functionalities, which autonomously use the Planner to devise an optimization path that ensures flexibility. Additionally, it employs a Teacher-Critic-Student Socratic dialogue pattern to iteratively optimize the prompts while conducting effective search. We conduct extensive experiments on various datasets to validate the effectiveness of our method, and perform additional analytical experiments to assess the model's advancement as well as the interpretability. 

**Abstract (ZH)**: 多代理结合苏格拉底引导的自动提示优化框架（MARS） 

---
# In-House Evaluation Is Not Enough: Towards Robust Third-Party Flaw Disclosure for General-Purpose AI 

**Title (ZH)**: 内部评估不够：走向稳健的通用人工智能第三方漏洞披露 

**Authors**: Shayne Longpre, Kevin Klyman, Ruth E. Appel, Sayash Kapoor, Rishi Bommasani, Michelle Sahar, Sean McGregor, Avijit Ghosh, Borhane Blili-Hamelin, Nathan Butters, Alondra Nelson, Amit Elazari, Andrew Sellars, Casey John Ellis, Dane Sherrets, Dawn Song, Harley Geiger, Ilona Cohen, Lauren McIlvenny, Madhulika Srikumar, Mark M. Jaycox, Markus Anderljung, Nadine Farid Johnson, Nicholas Carlini, Nicolas Miailhe, Nik Marda, Peter Henderson, Rebecca S. Portnoff, Rebecca Weiss, Victoria Westerhoff, Yacine Jernite, Rumman Chowdhury, Percy Liang, Arvind Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16861)  

**Abstract**: The widespread deployment of general-purpose AI (GPAI) systems introduces significant new risks. Yet the infrastructure, practices, and norms for reporting flaws in GPAI systems remain seriously underdeveloped, lagging far behind more established fields like software security. Based on a collaboration between experts from the fields of software security, machine learning, law, social science, and policy, we identify key gaps in the evaluation and reporting of flaws in GPAI systems. We call for three interventions to advance system safety. First, we propose using standardized AI flaw reports and rules of engagement for researchers in order to ease the process of submitting, reproducing, and triaging flaws in GPAI systems. Second, we propose GPAI system providers adopt broadly-scoped flaw disclosure programs, borrowing from bug bounties, with legal safe harbors to protect researchers. Third, we advocate for the development of improved infrastructure to coordinate distribution of flaw reports across the many stakeholders who may be impacted. These interventions are increasingly urgent, as evidenced by the prevalence of jailbreaks and other flaws that can transfer across different providers' GPAI systems. By promoting robust reporting and coordination in the AI ecosystem, these proposals could significantly improve the safety, security, and accountability of GPAI systems. 

**Abstract (ZH)**: 通用人工智能系统中普遍应用引入了显著的新风险。然而，用于报告通用人工智能系统缺陷的基础设施、实践和规范发展严重不足，远远落后于软件安全等更加成熟的研究领域。基于来自软件安全、机器学习、法律、社会科学和政策领域的专家合作，我们指出了评估和报告通用人工智能系统缺陷的关键缺口。我们呼吁采取三项干预措施以推进系统安全。首先，我们提议使用标准化的人工智能缺陷报告和研究人员的参与规则，以便更轻松地提交、重现和处理通用人工智能系统的缺陷。其次，我们提议通用人工智能系统提供者采纳广泛的缺陷披露计划，借鉴漏洞赏金制度，并提供法律安全港以保护研究人员。第三，我们倡导开发改进的基础设施以协调分布可能受影响的众多利益相关方的缺陷报告。这些干预措施由于跨不同提供商的人工智能系统中普遍存在越狱和其他缺陷而变得越来越迫切。通过在人工智能生态系统中促进稳健的报告和协调，这些提案有望显著提高通用人工智能系统的安全、安全性和问责制。 

---
# A Learnability Analysis on Neuro-Symbolic Learning 

**Title (ZH)**: 神经符号学习的可学习性分析 

**Authors**: Hao-Yuan He, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16797)  

**Abstract**: This paper analyzes the learnability of neuro-symbolic (NeSy) tasks within hybrid systems. We show that the learnability of NeSy tasks can be characterized by their derived constraint satisfaction problems (DCSPs). Specifically, a task is learnable if the corresponding DCSP has a unique solution; otherwise, it is unlearnable. For learnable tasks, we establish error bounds by exploiting the clustering property of the hypothesis space. Additionally, we analyze the asymptotic error for general NeSy tasks, showing that the expected error scales with the disagreement among solutions. Our results offer a principled approach to determining learnability and provide insights into the design of new algorithms. 

**Abstract (ZH)**: 本文分析了混合系统中神经符号（NeSy）任务的学习性。我们表明，NeSy任务的学习性可以通过其派生的约束满足问题（DCSP）来表征。具体来说，如果对应的DCSP有一个唯一解，则该任务是可学习的；否则，它是不可学习的。对于可学习的任务，我们通过利用假设空间的聚类性质建立了误差界。此外，我们分析了一般NeSy任务的渐近误差，指出预期误差与解之间的分歧成比例关系。我们的结果提供了一种原则性的方法来确定学习性，并为新算法的设计提供了洞察。 

---
# Does Chain-of-Thought Reasoning Help Mobile GUI Agent? An Empirical Study 

**Title (ZH)**: 链式思维推理有助于移动GUI代理？一项实证研究 

**Authors**: Li Zhang, Longxi Gao, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16788)  

**Abstract**: Reasoning capabilities have significantly improved the performance of vision-language models (VLMs) in domains such as mathematical problem-solving, coding, and visual question-answering. However, their impact on real-world applications remains unclear. This paper presents the first empirical study on the effectiveness of reasoning-enabled VLMs in mobile GUI agents, a domain that requires interpreting complex screen layouts, understanding user instructions, and executing multi-turn interactions. We evaluate two pairs of commercial models--Gemini 2.0 Flash and Claude 3.7 Sonnet--comparing their base and reasoning-enhanced versions across two static benchmarks (ScreenSpot and AndroidControl) and one interactive environment (AndroidWorld). We surprisingly find the Claude 3.7 Sonnet reasoning model achieves state-of-the-art performance on AndroidWorld. However, reasoning VLMs generally offer marginal improvements over non-reasoning models on static benchmarks and even degrade performance in some agent setups. Notably, reasoning and non-reasoning VLMs fail on different sets of tasks, suggesting that reasoning does have an impact, but its benefits and drawbacks counterbalance each other. We attribute these inconsistencies to the limitations of benchmarks and VLMs. Based on the findings, we provide insights for further enhancing mobile GUI agents in terms of benchmarks, VLMs, and their adaptability in dynamically invoking reasoning VLMs. The experimental data are publicly available at this https URL. 

**Abstract (ZH)**: 基于推理的视觉-语言模型在移动GUI代理中的有效性研究 

---
# SuperARC: A Test for General and Super Intelligence Based on First Principles of Recursion Theory and Algorithmic Probability 

**Title (ZH)**: SuperARC：基于递归理论和算法概率的基本原理测试通用和超智能能力 

**Authors**: Alberto Hernández-Espinosa, Luan Ozelim, Felipe S. Abrahão, Hector Zenil  

**Link**: [PDF](https://arxiv.org/pdf/2503.16743)  

**Abstract**: We introduce an open-ended test grounded in algorithmic probability that can avoid benchmark contamination in the quantitative evaluation of frontier models in the context of their Artificial General Intelligence (AGI) and Superintelligence (ASI) claims. Unlike other tests, this test does not rely on statistical compression methods (such as GZIP or LZW), which are more closely related to Shannon entropy than to Kolmogorov complexity. The test challenges aspects related to features of intelligence of fundamental nature such as synthesis and model creation in the context of inverse problems (generating new knowledge from observation). We argue that metrics based on model abstraction and optimal Bayesian inference for planning can provide a robust framework for testing intelligence, including natural intelligence (human and animal), narrow AI, AGI, and ASI. Our results show no clear evidence of LLM convergence towards a defined level of intelligence, particularly AGI or ASI. We found that LLM model versions tend to be fragile and incremental, as new versions may perform worse than older ones, with progress largely driven by the size of training data. The results were compared with a hybrid neurosymbolic approach that theoretically guarantees model convergence from optimal inference based on the principles of algorithmic probability and Kolmogorov complexity. The method outperforms LLMs in a proof-of-concept on short binary sequences. Our findings confirm suspicions regarding the fundamental limitations of LLMs, exposing them as systems optimised for the perception of mastery over human language. Progress among different LLM versions from the same developers was found to be inconsistent and limited, particularly in the absence of a solid symbolic counterpart. 

**Abstract (ZH)**: 基于算法概率的开放性测试：避免在前沿模型的AGI和ASI主张的量化评估中出现基准污染 

---
# Towards Agentic Recommender Systems in the Era of Multimodal Large Language Models 

**Title (ZH)**: 向代理型推荐系统迈进：多模态大语言模型时代 

**Authors**: Chengkai Huang, Junda Wu, Yu Xia, Zixu Yu, Ruhan Wang, Tong Yu, Ruiyi Zhang, Ryan A. Rossi, Branislav Kveton, Dongruo Zhou, Julian McAuley, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16734)  

**Abstract**: Recent breakthroughs in Large Language Models (LLMs) have led to the emergence of agentic AI systems that extend beyond the capabilities of standalone models. By empowering LLMs to perceive external environments, integrate multimodal information, and interact with various tools, these agentic systems exhibit greater autonomy and adaptability across complex tasks. This evolution brings new opportunities to recommender systems (RS): LLM-based Agentic RS (LLM-ARS) can offer more interactive, context-aware, and proactive recommendations, potentially reshaping the user experience and broadening the application scope of RS. Despite promising early results, fundamental challenges remain, including how to effectively incorporate external knowledge, balance autonomy with controllability, and evaluate performance in dynamic, multimodal settings. In this perspective paper, we first present a systematic analysis of LLM-ARS: (1) clarifying core concepts and architectures; (2) highlighting how agentic capabilities -- such as planning, memory, and multimodal reasoning -- can enhance recommendation quality; and (3) outlining key research questions in areas such as safety, efficiency, and lifelong personalization. We also discuss open problems and future directions, arguing that LLM-ARS will drive the next wave of RS innovation. Ultimately, we foresee a paradigm shift toward intelligent, autonomous, and collaborative recommendation experiences that more closely align with users' evolving needs and complex decision-making processes. 

**Abstract (ZH)**: 近期大型语言模型的突破推动了具有自主能力的AI系统的出现，这些系统超越了单一模型的功能。通过赋予大型语言模型感知外部环境、整合多模态信息以及与各种工具交互的能力，这些自主系统在复杂任务中表现出更大的自主性和适应性。这种演变为推荐系统（RS）带来了新的机会：基于大型语言模型的自主推荐系统（LLM-ARS）可以提供更加互动、情境意识强且主动的推荐，有可能重塑用户体验并扩大推荐系统的应用范围。尽管早期结果颇具前景，但仍存在一些根本性的挑战，包括如何有效融入外部知识、如何平衡自主性和可控性以及如何在动态的多模态环境中评估性能。在本文中，我们首先对LLM-ARS进行了系统的分析：（1）阐明核心概念和架构；（2）强调自主能力如规划、记忆和多模态推理如何提高推荐质量；（3）概述关于安全性、效率和终生个性化等领域的关键研究问题。我们还讨论了开放性问题和未来方向，认为LLM-ARS将推动推荐系统创新的下一波浪潮。最终，我们预见到一种智能的、自主的和合作的推荐体验范式的转变，这种转变更能够与用户不断变化的需求和复杂的决策过程相契合。 

---
# Towards Automated Semantic Interpretability in Reinforcement Learning via Vision-Language Models 

**Title (ZH)**: 基于视觉-语言模型的强化学习自动语义可解释性研究 

**Authors**: Zhaoxin Li, Zhang Xi-Jia, Batuhan Altundas, Letian Chen, Rohan Paleja, Matthew Gombolay  

**Link**: [PDF](https://arxiv.org/pdf/2503.16724)  

**Abstract**: Semantic Interpretability in Reinforcement Learning (RL) enables transparency, accountability, and safer deployment by making the agent's decisions understandable and verifiable. Achieving this, however, requires a feature space composed of human-understandable concepts, which traditionally rely on human specification and fail to generalize to unseen environments. In this work, we introduce Semantically Interpretable Reinforcement Learning with Vision-Language Models Empowered Automation (SILVA), an automated framework that leverages pre-trained vision-language models (VLM) for semantic feature extraction and interpretable tree-based models for policy optimization. SILVA first queries a VLM to identify relevant semantic features for an unseen environment, then extracts these features from the environment. Finally, it trains an Interpretable Control Tree via RL, mapping the extracted features to actions in a transparent and interpretable manner. To address the computational inefficiency of extracting features directly with VLMs, we develop a feature extraction pipeline that generates a dataset for training a lightweight convolutional network, which is subsequently used during RL. By leveraging VLMs to automate tree-based RL, SILVA removes the reliance on human annotation previously required by interpretable models while also overcoming the inability of VLMs alone to generate valid robot policies, enabling semantically interpretable reinforcement learning without human-in-the-loop. 

**Abstract (ZH)**: 基于视觉-语言模型赋能自动化的方法的语义可解释强化学习（SILVA） 

---
# Empowering Medical Multi-Agents with Clinical Consultation Flow for Dynamic Diagnosis 

**Title (ZH)**: 基于临床咨询流程赋能医疗多智能体动态诊断 

**Authors**: Sihan Wang, Suiyang Jiang, Yibo Gao, Boming Wang, Shangqi Gao, Xiahai Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16547)  

**Abstract**: Traditional AI-based healthcare systems often rely on single-modal data, limiting diagnostic accuracy due to incomplete information. However, recent advancements in foundation models show promising potential for enhancing diagnosis combining multi-modal information. While these models excel in static tasks, they struggle with dynamic diagnosis, failing to manage multi-turn interactions and often making premature diagnostic decisions due to insufficient persistence in information this http URL address this, we propose a multi-agent framework inspired by consultation flow and reinforcement learning (RL) to simulate the entire consultation process, integrating multiple clinical information for effective diagnosis. Our approach incorporates a hierarchical action set, structured from clinic consultation flow and medical textbook, to effectively guide the decision-making process. This strategy improves agent interactions, enabling them to adapt and optimize actions based on the dynamic state. We evaluated our framework on a public dynamic diagnosis benchmark. The proposed framework evidentially improves the baseline methods and achieves state-of-the-art performance compared to existing foundation model-based methods. 

**Abstract (ZH)**: 基于传统单模态数据的AI医疗系统限制了诊断准确性，而近年来的基础模型进展显示了通过整合多模态信息增强诊断的潜力。尽管这些模型在静态任务中表现优异，但在动态诊断中却难以应对多轮交互，常因信息不足而过早做出诊断决策。为解决这些问题，我们提出了一种受咨询流程启发并结合强化学习的多代理框架，以模拟完整的咨询过程，整合多种临床信息进行有效诊断。该方法采用层次化动作集，从临床咨询流程和医学教科书结构化而来，有效指导决策过程。此策略增强了代理间的互动，使它们能够根据动态状态进行适应和优化。我们在一个公开的动态诊断基准上评估了该框架。提出的框架显著改进了基线方法，并在与现有基于基础模型的方法相比时达到了最先进的性能。 

---
# Improving Interactive Diagnostic Ability of a Large Language Model Agent Through Clinical Experience Learning 

**Title (ZH)**: 通过临床经验学习提升大型语言模型代理的交互诊断能力 

**Authors**: Zhoujian Sun, Ziyi Liu, Cheng Luo, Jiebin Chu, Zhengxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16463)  

**Abstract**: Recent advances in large language models (LLMs) have shown promising results in medical diagnosis, with some studies indicating superior performance compared to human physicians in specific scenarios. However, the diagnostic capabilities of LLMs are often overestimated, as their performance significantly deteriorates in interactive diagnostic settings that require active information gathering. This study investigates the underlying mechanisms behind the performance degradation phenomenon and proposes a solution. We identified that the primary deficiency of LLMs lies in the initial diagnosis phase, particularly in information-gathering efficiency and initial diagnosis formation, rather than in the subsequent differential diagnosis phase. To address this limitation, we developed a plug-and-play method enhanced (PPME) LLM agent, leveraging over 3.5 million electronic medical records from Chinese and American healthcare facilities. Our approach integrates specialized models for initial disease diagnosis and inquiry into the history of the present illness, trained through supervised and reinforcement learning techniques. The experimental results indicate that the PPME LLM achieved over 30% improvement compared to baselines. The final diagnostic accuracy of the PPME LLM in interactive diagnostic scenarios approached levels comparable to those achieved using complete clinical data. These findings suggest a promising potential for developing autonomous diagnostic systems, although further validation studies are needed. 

**Abstract (ZH)**: 近期大规模语言模型在医疗诊断方面的进展显示出有希望的结果，一些研究在特定场景中表明其性能优于人类医师。然而，大规模语言模型的诊断能力往往被高估，在需要主动信息收集的交互式诊断环境中，其性能显著下降。本研究探讨了性能下降现象背后的机制并提出了解决方案。我们发现，大规模语言模型的主要不足在于初步诊断阶段，特别是信息收集效率和初步诊断形成上，而不是在后续的鉴别诊断阶段。为解决这一限制，我们开发了一种插件增强的大规模语言模型代理（PPME LLM），利用来自中国和美国医疗服务设施的超过350万份电子医疗记录。我们的方法通过监督学习和强化学习技术将专门用于初步疾病诊断和现病史查询的模型结合起来。实验结果表明，PPME LLM相比于基线模型实现了超过30%的改进。在交互式诊断场景中，PPME LLM的最终诊断准确性接近使用完整临床数据所达到的水平。这些发现表明了自主诊断系统开发的巨大潜力，但还需要进一步的验证研究。 

---
# NdLinear Is All You Need for Representation Learning 

**Title (ZH)**: NdLinear 皆你所需，用于表示学习 

**Authors**: Alex Reneau, Jerry Yao-Chieh Hu, Zhongfang Zhuang, Ting-Chun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17353)  

**Abstract**: Many high-impact machine learning tasks involve multi-dimensional data (e.g., images, volumetric medical scans, multivariate time-series). Yet, most neural architectures flatten inputs, discarding critical cross-dimension information. We introduce NdLinear, a novel linear transformation that preserves these structures without extra overhead. By operating separately along each dimension, NdLinear captures dependencies that standard fully connected layers overlook. Extensive experiments across convolutional, recurrent, and transformer-based networks show significant improvements in representational power and parameter efficiency. Crucially, NdLinear serves as a foundational building block for large-scale foundation models by operating on any unimodal or multimodal data in its native form. This removes the need for flattening or modality-specific preprocessing. Ndlinear rethinks core architectural priorities beyond attention, enabling more expressive, context-aware models at scale. We propose NdLinear as a drop-in replacement for standard linear layers -- marking an important step toward next-generation neural architectures. 

**Abstract (ZH)**: NdLinear：一种保留多维结构的新型线性变换 

---
# Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation 

**Title (ZH)**: 调整你的节奏：基于门控增强节奏感知特征表示的高对齐舞蹈姿态生成 

**Authors**: Congyi Fan, Jian Guan, Xuanjia Zhao, Dongli Xu, Youtian Lin, Tong Ye, Pengming Feng, Haiwei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.17340)  

**Abstract**: Automatically generating natural, diverse and rhythmic human dance movements driven by music is vital for virtual reality and film industries. However, generating dance that naturally follows music remains a challenge, as existing methods lack proper beat alignment and exhibit unnatural motion dynamics. In this paper, we propose Danceba, a novel framework that leverages gating mechanism to enhance rhythm-aware feature representation for music-driven dance generation, which achieves highly aligned dance poses with enhanced rhythmic sensitivity. Specifically, we introduce Phase-Based Rhythm Extraction (PRE) to precisely extract rhythmic information from musical phase data, capitalizing on the intrinsic periodicity and temporal structures of music. Additionally, we propose Temporal-Gated Causal Attention (TGCA) to focus on global rhythmic features, ensuring that dance movements closely follow the musical rhythm. We also introduce Parallel Mamba Motion Modeling (PMMM) architecture to separately model upper and lower body motions along with musical features, thereby improving the naturalness and diversity of generated dance movements. Extensive experiments confirm that Danceba outperforms state-of-the-art methods, achieving significantly better rhythmic alignment and motion diversity. Project page: this https URL . 

**Abstract (ZH)**: 自动生成与音乐节奏自然契合、多样且富有韵律的人类舞蹈动作对于虚拟现实和电影行业至关重要。然而，生成能够自然跟随音乐的舞蹈仍然存在挑战，因为现有方法缺乏恰当的节奏对齐，表现出不自然的运动动态。在本文中，我们提出Danceba，一个新颖的框架，利用门控机制增强音乐感知特征表示，以实现高度对齐且增强节奏敏感性的舞蹈姿态。具体来说，我们引入基于相位的节奏提取（PRE）以精确提取音乐相位数据中的节奏信息，充分利用音乐的固有周期性和时间结构。此外，我们提出时间门控因果注意力（TGCA）以聚焦全局节奏特征，确保舞蹈动作紧密跟随音乐节奏。我们还引入并行马amba运动建模（PMMM）架构，分别建模上身和下身运动及其与音乐特征的关系，从而提高生成舞蹈动作的自然性和多样性。广泛实验结果显示，Danceba在节奏对齐和运动多样性方面显著优于现有方法。项目页面：this https URL。 

---
# Can AI expose tax loopholes? Towards a new generation of legal policy assistants 

**Title (ZH)**: AI能否揭露税收漏洞？迈向新一代法律政策助手 

**Authors**: Peter Fratrič, Nils Holzenberger, David Restrepo Amariles  

**Link**: [PDF](https://arxiv.org/pdf/2503.17339)  

**Abstract**: The legislative process is the backbone of a state built on solid institutions. Yet, due to the complexity of laws -- particularly tax law -- policies may lead to inequality and social tensions. In this study, we introduce a novel prototype system designed to address the issues of tax loopholes and tax avoidance. Our hybrid solution integrates a natural language interface with a domain-specific language tailored for planning. We demonstrate on a case study how tax loopholes and avoidance schemes can be exposed. We conclude that our prototype can help enhance social welfare by systematically identifying and addressing tax gaps stemming from loopholes. 

**Abstract (ZH)**: 立法过程是基于坚实机构的国家的核心。然而，由于法律——特别是税法——的复杂性，政策可能导致不平等和社会紧张。在本研究中，我们介绍了一种新型原型系统，旨在解决税收漏洞和税收规避的问题。我们的混合解决方案结合了自然语言界面和一个针对规划的领域专用语言。我们通过一个案例研究展示了如何暴露税收漏洞和规避方案。我们得出结论，我们的原型可以通过系统识别和解决源自漏洞的税差，从而有助于提高社会福利。 

---
# Efficient Intent-Based Filtering for Multi-Party Conversations Using Knowledge Distillation from LLMs 

**Title (ZH)**: 使用来自大规模语言模型的知识蒸馏进行高效基于意图的多方对话过滤 

**Authors**: Reem Gody, Mohamed Abdelghaffar, Mohammed Jabreel, Ahmed Tawfik  

**Link**: [PDF](https://arxiv.org/pdf/2503.17336)  

**Abstract**: Large language models (LLMs) have showcased remarkable capabilities in conversational AI, enabling open-domain responses in chat-bots, as well as advanced processing of conversations like summarization, intent classification, and insights generation. However, these models are resource-intensive, demanding substantial memory and computational power. To address this, we propose a cost-effective solution that filters conversational snippets of interest for LLM processing, tailored to the target downstream application, rather than processing every snippet. In this work, we introduce an innovative approach that leverages knowledge distillation from LLMs to develop an intent-based filter for multi-party conversations, optimized for compute power constrained environments. Our method combines different strategies to create a diverse multi-party conversational dataset, that is annotated with the target intents and is then used to fine-tune the MobileBERT model for multi-label intent classification. This model achieves a balance between efficiency and performance, effectively filtering conversation snippets based on their intents. By passing only the relevant snippets to the LLM for further processing, our approach significantly reduces overall operational costs depending on the intents and the data distribution as demonstrated in our experiments. 

**Abstract (ZH)**: 大型语言模型（LLMs）在对话AI领域展现出了 remarkable 的能力，能够生成开放领域对话，并对对话进行总结、意图分类和见解生成。然而，这些模型资源密集，需要大量内存和计算能力。为此，我们提出了一种经济高效的解决方案，通过对目标下游应用感兴趣的对话片段进行过滤以便LLM处理，而不是处理每个片段。在本文中，我们介绍了一种创新的方法，利用LLMs的知识蒸馏开发基于意图的过滤器，以适应计算能力受限的环境。我们的方法结合不同的策略创建了一个多元对话数据集，并对其进行标注以目标意图，然后用于微调MobileBERT模型以实现多标签意图分类。该模型在效率和性能之间达到了平衡，能够根据对话片段的意图对其进行有效过滤。通过只将相关的片段传递给LLM进行进一步处理，我们的方法在实验中展示了依赖于意图和数据分布的情况下显著降低了总体运营成本。 

---
# CVE-Bench: A Benchmark for AI Agents' Ability to Exploit Real-World Web Application Vulnerabilities 

**Title (ZH)**: CVE-Bench: 一个评估AI代理发现真实世界Web应用漏洞能力的基准测试 

**Authors**: Yuxuan Zhu, Antony Kellermann, Dylan Bowman, Philip Li, Akul Gupta, Adarsh Danda, Richard Fang, Conner Jensen, Eric Ihli, Jason Benn, Jet Geronimo, Avi Dhir, Sudhit Rao, Kaicheng Yu, Twm Stone, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17332)  

**Abstract**: Large language model (LLM) agents are increasingly capable of autonomously conducting cyberattacks, posing significant threats to existing applications. This growing risk highlights the urgent need for a real-world benchmark to evaluate the ability of LLM agents to exploit web application vulnerabilities. However, existing benchmarks fall short as they are limited to abstracted Capture the Flag competitions or lack comprehensive coverage. Building a benchmark for real-world vulnerabilities involves both specialized expertise to reproduce exploits and a systematic approach to evaluating unpredictable threats. To address this challenge, we introduce CVE-Bench, a real-world cybersecurity benchmark based on critical-severity Common Vulnerabilities and Exposures. In CVE-Bench, we design a sandbox framework that enables LLM agents to exploit vulnerable web applications in scenarios that mimic real-world conditions, while also providing effective evaluation of their exploits. Our evaluation shows that the state-of-the-art agent framework can resolve up to 13% of vulnerabilities. 

**Abstract (ZH)**: 大规模语言模型（LLM）代理日益具备自主开展网络攻击的能力，对现有应用构成重大威胁。这一不断增长的风险突显了建立实际世界基准以评估LLM代理利用Web应用漏洞能力的紧迫需求。然而，现有基准不足，因为它们局限于抽象的Capture the Flag比赛或缺乏全面覆盖。建立针对实际漏洞的基准需要专门的知识来重现漏洞利用，同时也需要系统的方法来评估不可预测的威胁。为应对这一挑战，我们介绍了一个基于关键严重性通用漏洞和曝光（CVE）的实际世界网络安全基准——CVE-Bench。在CVE-Bench中，我们设计了一个沙盒框架，使LLM代理能够在模拟实际世界条件的场景中利用漏洞的Web应用，同时有效评估其漏洞利用。我们的评估表明，最先进的代理框架可以解决多达13%的漏洞。 

---
# LLM+MAP: Bimanual Robot Task Planning using Large Language Models and Planning Domain Definition Language 

**Title (ZH)**: LLM+MAP: 使用大型语言模型和规划领域定义语言的双臂机器人任务规划 

**Authors**: Kun Chu, Xufeng Zhao, Cornelius Weber, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2503.17309)  

**Abstract**: Bimanual robotic manipulation provides significant versatility, but also presents an inherent challenge due to the complexity involved in the spatial and temporal coordination between two hands. Existing works predominantly focus on attaining human-level manipulation skills for robotic hands, yet little attention has been paid to task planning on long-horizon timescales. With their outstanding in-context learning and zero-shot generation abilities, Large Language Models (LLMs) have been applied and grounded in diverse robotic embodiments to facilitate task planning. However, LLMs still suffer from errors in long-horizon reasoning and from hallucinations in complex robotic tasks, lacking a guarantee of logical correctness when generating the plan. Previous works, such as LLM+P, extended LLMs with symbolic planners. However, none have been successfully applied to bimanual robots. New challenges inevitably arise in bimanual manipulation, necessitating not only effective task decomposition but also efficient task allocation. To address these challenges, this paper introduces LLM+MAP, a bimanual planning framework that integrates LLM reasoning and multi-agent planning, automating effective and efficient bimanual task planning. We conduct simulated experiments on various long-horizon manipulation tasks of differing complexity. Our method is built using GPT-4o as the backend, and we compare its performance against plans generated directly by LLMs, including GPT-4o, V3 and also recent strong reasoning models o1 and R1. By analyzing metrics such as planning time, success rate, group debits, and planning-step reduction rate, we demonstrate the superior performance of LLM+MAP, while also providing insights into robotic reasoning. Code is available at this https URL. 

**Abstract (ZH)**: 双臂机器人操作提供了显著的灵活性，但同时也由于在空间和时间上协调两个手的复杂性而带来了内在的挑战。现有研究主要集中在使机器人的手达到人类级别的操作技能，但在长时尺度的任务规划方面关注较少。凭借卓越的上下文学习能力和零样本生成能力，大型语言模型（LLMs）已在各种机器人类体中得到应用，以促进任务规划。然而，LLMs 在长时尺度推理中仍存在错误，并且在复杂机器人任务中容易产生幻想，无法保证逻辑正确性。先前的研究，如LLM+P，将LLMs与符号规划结合，但尚未成功应用于双臂机器人。双臂操作带来了新的挑战，不仅需要有效的任务分解，还需要高效的任务分配。为应对这些挑战，本文引入了LLM+MAP，这是一种结合LLM推理与多代理规划的双臂规划框架，实现有效且高效的双臂任务规划。我们在不同复杂性的长时尺度操作任务中进行了模拟实验。我们使用GPT-4o作为后端构建了该方法，并将其性能与直接由LLMs生成的计划进行比较，包括GPT-4o、V3以及最近的强推理模型o1和R1。通过分析规划时间、成功率、群体折扣和规划步长减少率等指标，我们展示了LLM+MAP的优越性能，并为机器人推理提供了见解。代码可访问此处：this https URL。 

---
# Preference-Guided Diffusion for Multi-Objective Offline Optimization 

**Title (ZH)**: 基于偏好引导的多目标离线优化扩散方法 

**Authors**: Yashas Annadani, Syrine Belakaria, Stefano Ermon, Stefan Bauer, Barbara E Engelhardt  

**Link**: [PDF](https://arxiv.org/pdf/2503.17299)  

**Abstract**: Offline multi-objective optimization aims to identify Pareto-optimal solutions given a dataset of designs and their objective values. In this work, we propose a preference-guided diffusion model that generates Pareto-optimal designs by leveraging a classifier-based guidance mechanism. Our guidance classifier is a preference model trained to predict the probability that one design dominates another, directing the diffusion model toward optimal regions of the design space. Crucially, this preference model generalizes beyond the training distribution, enabling the discovery of Pareto-optimal solutions outside the observed dataset. We introduce a novel diversity-aware preference guidance, augmenting Pareto dominance preference with diversity criteria. This ensures that generated solutions are optimal and well-distributed across the objective space, a capability absent in prior generative methods for offline multi-objective optimization. We evaluate our approach on various continuous offline multi-objective optimization tasks and find that it consistently outperforms other inverse/generative approaches while remaining competitive with forward/surrogate-based optimization methods. Our results highlight the effectiveness of classifier-guided diffusion models in generating diverse and high-quality solutions that approximate the Pareto front well. 

**Abstract (ZH)**: 离线多目标优化旨在给定设计及其目标值的数据集时，识别帕累托最优解。本文提出了一种偏好引导扩散模型，通过基于分类器的引导机制生成帕累托最优设计。我们的引导分类器是一个训练好的偏好模型，用于预测一个设计支配另一个设计的概率，从而引导扩散模型向设计空间的最优区域发展。关键在于，该偏好模型可以泛化到训练分布之外，从而在观察到的数据集之外发现帕累托最优解。我们引入了一种新的多样性感知偏好引导，将支配偏好与多样性标准相结合，确保生成的解决方案不仅是最优的，而且在目标空间中分布良好，这是以往离线多目标优化的生成方法所欠缺的能力。我们在各种连续离线多目标优化任务上评估了我们的方法，发现它在所有其他逆向/生成方法中表现更优，并且与前向/代理基优化方法具有竞争力。我们的结果强调了分类器引导扩散模型在生成多样且高质量解决方案方面的有效性，这些解决方案能够很好地逼近帕累托前沿。 

---
# KL3M Tokenizers: A Family of Domain-Specific and Character-Level Tokenizers for Legal, Financial, and Preprocessing Applications 

**Title (ZH)**: KL3M分词器：适用于法律、金融及预处理应用的领域特定字符级分词器 

**Authors**: Michael J Bommarito, Daniel Martin Katz, Jillian Bommarito  

**Link**: [PDF](https://arxiv.org/pdf/2503.17247)  

**Abstract**: We present the KL3M tokenizers, a family of specialized tokenizers for legal, financial, and governmental text. Despite established work on tokenization, specialized tokenizers for professional domains remain understudied. Our paper offers two main contributions to this area.
First, we introduce domain-specific BPE tokenizers for legal, financial, and governmental text. Our kl3m-004-128k-cased tokenizer uses 9-17% fewer tokens than GPT-4o and Llama3 for domain-specific documents, despite having a smaller vocabulary. For specialized terminology, our cased tokenizer is even more efficient, using up to 83% fewer tokens for legal terms and 39% fewer tokens for financial terms.
Second, we develop character-level BPE tokenizers (4K, 8K, and 16K vocabulary sizes) for text correction tasks like OCR post-processing. These tokenizers keep consistent token boundaries between error-containing and correct text, making it easier for models to learn correction patterns.
These tokenizers help professional applications by fitting more text in context windows, reducing computational needs, and preserving the meaning of domain-specific terms. Our analysis shows these efficiency gains directly benefit the processing of long legal and financial documents. We release all tokenizers and code through GitHub and Hugging Face to support further research in specialized tokenization. 

**Abstract (ZH)**: KL3M分词器：面向法律、金融和政府文本的专业分词技术 

---
# SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging 

**Title (ZH)**: SafeMERGE: 在精选分层模型合并中保留细调大型语言模型的安全对齐 

**Authors**: Aladin Djuhera, Swanand Ravindra Kadhe, Farhan Ahmed, Syed Zawad, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2503.17239)  

**Abstract**: Fine-tuning large language models (LLMs) on downstream tasks can inadvertently erode their safety alignment, even for benign fine-tuning datasets. We address this challenge by proposing SafeMERGE, a post-fine-tuning framework that preserves safety while maintaining task utility. It achieves this by selectively merging fine-tuned and safety-aligned model layers only when those deviate from safe behavior, measured by a cosine similarity criterion. We evaluate SafeMERGE against other fine-tuning- and post-fine-tuning-stage approaches for Llama-2-7B-Chat and Qwen-2-7B-Instruct models on GSM8K and PubMedQA tasks while exploring different merging strategies. We find that SafeMERGE consistently reduces harmful outputs compared to other baselines without significantly sacrificing performance, sometimes even enhancing it. The results suggest that our selective, subspace-guided, and per-layer merging method provides an effective safeguard against the inadvertent loss of safety in fine-tuned LLMs while outperforming simpler post-fine-tuning-stage defenses. 

**Abstract (ZH)**: Fine-tuning 大型语言模型 (LLMs) 在下游任务上的调整可能会无意中削弱其安全性对齐，即使是对于 benign 细调数据集也是如此。我们提出了一种名为 SafeMERGE 的后细调框架，该框架在保持安全性的前提下维持任务效用。通过仅在那些行为偏离安全标准的 fine-tuned 和安全性对齐的模型层之间选择性地进行合并，SafeMERGE 实现了这一目标，合并的方式基于余弦相似性的标准。我们在 Llama-2-7B-Chat 和 Qwen-2-7B-Instruct 模型上，以 GSM8K 和 PubMedQA 任务为评价标准，探讨了不同的合并策略。研究结果表明，SafeMERGE 在减少有害输出方面始终优于其他基准方法，而不显著牺牲性能，有时甚至可以提升性能。结果表明，我们的选择性、子空间引导的逐层合并方法提供了一种有效的保护措施，以防止在细调的大型语言模型中意外失去安全性，并且优于简单的后细调阶段防御措施。 

---
# Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID 

**Title (ZH)**: 强 baseline：基于 YOLOv12 和 BoT-SORT-ReID 的多无人机跟踪 

**Authors**: Yu-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17237)  

**Abstract**: Detecting and tracking multiple unmanned aerial vehicles (UAVs) in thermal infrared video is inherently challenging due to low contrast, environmental noise, and small target sizes. This paper provides a straightforward approach to address multi-UAV tracking in thermal infrared video, leveraging recent advances in detection and tracking. Instead of relying on the YOLOv5 with the DeepSORT pipeline, we present a tracking framework built on YOLOv12 and BoT-SORT, enhanced with tailored training and inference strategies. We evaluate our approach following the metrics from the 4th Anti-UAV Challenge and demonstrate competitive performance. Notably, we achieve strong results without using contrast enhancement or temporal information fusion to enrich UAV features, highlighting our approach as a "Strong Baseline" for the multi-UAV tracking task. We provide implementation details, in-depth experimental analysis, and a discussion of potential improvements. The code is available at this https URL . 

**Abstract (ZH)**: 基于热红外视频的多无人机检测与跟踪：一种简单有效的框架 

---
# FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs 

**Title (ZH)**: FactSelfCheck: LLMs的事实级黑盒幻觉检测 

**Authors**: Albert Sawczyn, Jakub Binkowski, Denis Janiak, Bogdan Gabrys, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2503.17229)  

**Abstract**: Large Language Models (LLMs) frequently generate hallucinated content, posing significant challenges for applications where factuality is crucial. While existing hallucination detection methods typically operate at the sentence level or passage level, we propose FactSelfCheck, a novel black-box sampling-based method that enables fine-grained fact-level detection. Our approach represents text as knowledge graphs consisting of facts in the form of triples. Through analyzing factual consistency across multiple LLM responses, we compute fine-grained hallucination scores without requiring external resources or training data. Our evaluation demonstrates that FactSelfCheck performs competitively with leading sampling-based methods while providing more detailed insights. Most notably, our fact-level approach significantly improves hallucination correction, achieving a 35% increase in factual content compared to the baseline, while sentence-level SelfCheckGPT yields only an 8% improvement. The granular nature of our detection enables more precise identification and correction of hallucinated content. 

**Abstract (ZH)**: 大型语言模型（LLMs）经常生成虚构内容，这对事实至关重要的应用构成了重大挑战。虽然现有的虚构内容检测方法通常在句子或段落级别进行操作，但我们提出了FactSelfCheck，这是一种新颖的黑盒抽样基于方法，能够实现细粒度的事实级检测。我们的方法将文本表示为由三元组形式的事实构成的知识图谱。通过对多个LLM响应中的事实一致性进行分析，我们可以在不依赖外部资源或训练数据的情况下计算细粒度的虚构得分。我们的评估表明，FactSelfCheck在性能上与领先的抽样基于方法相当，同时提供更详细的洞察。尤为值得注意的是，我们基于事实的方法在虚构内容纠正方面显著提升，与基线相比，事实内容提高了35%，而基于句子级别的SelfCheckGPT仅提高了8%。我们检测的粒度化特性能够更精确地识别和纠正虚构内容。 

---
# Neuro-Symbolic Scene Graph Conditioning for Synthetic Image Dataset Generation 

**Title (ZH)**: 基于神经-符号场景图的合成图像数据集生成 

**Authors**: Giacomo Savazzi, Eugenio Lomurno, Cristian Sbrolli, Agnese Chiatti, Matteo Matteucci  

**Link**: [PDF](https://arxiv.org/pdf/2503.17224)  

**Abstract**: As machine learning models increase in scale and complexity, obtaining sufficient training data has become a critical bottleneck due to acquisition costs, privacy constraints, and data scarcity in specialised domains. While synthetic data generation has emerged as a promising alternative, a notable performance gap remains compared to models trained on real data, particularly as task complexity grows. Concurrently, Neuro-Symbolic methods, which combine neural networks' learning strengths with symbolic reasoning's structured representations, have demonstrated significant potential across various cognitive tasks. This paper explores the utility of Neuro-Symbolic conditioning for synthetic image dataset generation, focusing specifically on improving the performance of Scene Graph Generation models. The research investigates whether structured symbolic representations in the form of scene graphs can enhance synthetic data quality through explicit encoding of relational constraints. The results demonstrate that Neuro-Symbolic conditioning yields significant improvements of up to +2.59% in standard Recall metrics and +2.83% in No Graph Constraint Recall metrics when used for dataset augmentation. These findings establish that merging Neuro-Symbolic and generative approaches produces synthetic data with complementary structural information that enhances model performance when combined with real data, providing a novel approach to overcome data scarcity limitations even for complex visual reasoning tasks. 

**Abstract (ZH)**: 随着机器学习模型的规模和复杂性增加，由于获取成本、隐私约束及专业领域数据稀缺性，获得足够的训练数据已成为关键瓶颈。尽管合成数据生成已经成为一个有前景的替代方案，但在复杂任务下与使用真实数据训练的模型相比仍存在显著性能差距。与此同时，神经符号方法结合了神经网络的学习优势和符号推理的结构化表示，在各种认知任务中展现了显著潜力。本文探索了神经符号条件化在合成图像数据集生成中的应用，特别是着重于提高场景图生成模型的性能。研究探讨了通过显式编码关系约束来提升基于场景图的结构化符号表示是否能改善合成数据质量。结果表明，使用神经符号条件化在数据集扩充时，标准召回率指标可提升多达2.59%，在无图约束召回率指标上可提升2.83%。这些发现证实，将神经符号方法与生成方法相结合，可以生成具有补充结构信息的合成数据，当与真实数据结合时能提升模型性能，为克服复杂视觉推理任务中的数据稀缺性限制提供了全新方法。 

---
# Automating Adjudication of Cardiovascular Events Using Large Language Models 

**Title (ZH)**: 使用大型语言模型自动 adjudicate 心血管事件 

**Authors**: Sonish Sivarajkumar, Kimia Ameri, Chuqin Li, Yanshan Wang, Min Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17222)  

**Abstract**: Cardiovascular events, such as heart attacks and strokes, remain a leading cause of mortality globally, necessitating meticulous monitoring and adjudication in clinical trials. This process, traditionally performed manually by clinical experts, is time-consuming, resource-intensive, and prone to inter-reviewer variability, potentially introducing bias and hindering trial progress. This study addresses these critical limitations by presenting a novel framework for automating the adjudication of cardiovascular events in clinical trials using Large Language Models (LLMs). We developed a two-stage approach: first, employing an LLM-based pipeline for event information extraction from unstructured clinical data and second, using an LLM-based adjudication process guided by a Tree of Thoughts approach and clinical endpoint committee (CEC) guidelines. Using cardiovascular event-specific clinical trial data, the framework achieved an F1-score of 0.82 for event extraction and an accuracy of 0.68 for adjudication. Furthermore, we introduce the CLEART score, a novel, automated metric specifically designed for evaluating the quality of AI-generated clinical reasoning in adjudicating cardiovascular events. This approach demonstrates significant potential for substantially reducing adjudication time and costs while maintaining high-quality, consistent, and auditable outcomes in clinical trials. The reduced variability and enhanced standardization also allow for faster identification and mitigation of risks associated with cardiovascular therapies. 

**Abstract (ZH)**: 使用大型语言模型自动化的临床试验心血管事件裁决框架 

---
# PP-DocLayout: A Unified Document Layout Detection Model to Accelerate Large-Scale Data Construction 

**Title (ZH)**: PP-DocLayout: 一种统一的文档布局检测模型，以加速大规模数据构建 

**Authors**: Ting Sun, Cheng Cui, Yuning Du, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17213)  

**Abstract**: Document layout analysis is a critical preprocessing step in document intelligence, enabling the detection and localization of structural elements such as titles, text blocks, tables, and formulas. Despite its importance, existing layout detection models face significant challenges in generalizing across diverse document types, handling complex layouts, and achieving real-time performance for large-scale data processing. To address these limitations, we present PP-DocLayout, which achieves high precision and efficiency in recognizing 23 types of layout regions across diverse document formats. To meet different needs, we offer three models of varying scales. PP-DocLayout-L is a high-precision model based on the RT-DETR-L detector, achieving 90.4% mAP@0.5 and an end-to-end inference time of 13.4 ms per page on a T4 GPU. PP-DocLayout-M is a balanced model, offering 75.2% mAP@0.5 with an inference time of 12.7 ms per page on a T4 GPU. PP-DocLayout-S is a high-efficiency model designed for resource-constrained environments and real-time applications, with an inference time of 8.1 ms per page on a T4 GPU and 14.5 ms on a CPU. This work not only advances the state of the art in document layout analysis but also provides a robust solution for constructing high-quality training data, enabling advancements in document intelligence and multimodal AI systems. Code and models are available at this https URL . 

**Abstract (ZH)**: 文档布局分析是文档智能中的关键预处理步骤，能够检测和定位标题、文本块、表格和公式等结构元素。为了应对现有布局检测模型在跨文档类型泛化、处理复杂布局以及大规模数据实时处理方面的局限性，我们提出了PP-DocLayout，它在T4 GPU上实现了每页13.4 ms的端到端推理时间，精确识别23种不同类型的布局区域。PP-DocLayout-L是基于RT-DETR-L检测器的高精度模型，实现90.4%的mAP@0.5。PP-DocLayout-M是平衡模型，实现75.2%的mAP@0.5，每页推理时间为12.7 ms。PP-DocLayout-S是高效模型，适用于资源受限环境和实时应用，T4 GPU上的每页推理时间为8.1 ms，CPU上的时间为14.5 ms。本工作不仅推进了文档布局分析的技术前沿，还提供了构建高质量训练数据的稳健解决方案，推动了文档智能和多模态AI系统的发展。代码和模型可在此处访问：this https URL。 

---
# TreeSynth: Synthesizing Diverse Data from Scratch via Tree-Guided Subspace Partitioning 

**Title (ZH)**: TreeSynth：基于树引导子空间划分的从零合成多样数据方法 

**Authors**: Sheng Wang, Pengan Chen, Jingqi Zhou, Qintong Li, Jingwei Dong, Jiahui Gao, Boyang Xue, Jiyue Jiang, Lingpeng Kong, Chuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17195)  

**Abstract**: Model customization requires high-quality and diverse datasets, but acquiring such data remains challenging and costly. Although large language models (LLMs) can synthesize training data, current approaches are constrained by limited seed data, model bias and insufficient control over the generation process, resulting in limited diversity and biased distribution with the increase of data scales. To tackle this challenge, we present TreeSynth, a tree-guided subspace-based data synthesis framework that recursively partitions the entire data space into hierar-chical subspaces, enabling comprehensive and diverse scaling of data synthesis. Briefly, given a task-specific description, we construct a data space partitioning tree by iteratively executing criteria determination and subspace coverage steps. This hierarchically divides the whole space (i.e., root node) into mutually exclusive and complementary atomic subspaces (i.e., leaf nodes). By collecting synthesized data according to the attributes of each leaf node, we obtain a diverse dataset that fully covers the data space. Empirically, our extensive experiments demonstrate that TreeSynth surpasses both human-designed datasets and the state-of-the-art data synthesis baselines, achieving maximum improvements of 45.2% in data diversity and 17.6% in downstream task performance across various models and tasks. Hopefully, TreeSynth provides a scalable solution to synthesize diverse and comprehensive datasets from scratch without human intervention. 

**Abstract (ZH)**: 基于树引导的子空间数据合成框架：TreeSynth 

---
# D2Fusion: Dual-domain Fusion with Feature Superposition for Deepfake Detection 

**Title (ZH)**: D2Fusion: 双 domain 融合与特征叠加方法在深度假信息检测中的应用 

**Authors**: Xueqi Qiu, Xingyu Miao, Fan Wan, Haoran Duan, Tejal Shah, Varun Ojhab, Yang Longa, Rajiv Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2503.17184)  

**Abstract**: Deepfake detection is crucial for curbing the harm it causes to society. However, current Deepfake detection methods fail to thoroughly explore artifact information across different domains due to insufficient intrinsic interactions. These interactions refer to the fusion and coordination after feature extraction processes across different domains, which are crucial for recognizing complex forgery clues. Focusing on more generalized Deepfake detection, in this work, we introduce a novel bi-directional attention module to capture the local positional information of artifact clues from the spatial domain. This enables accurate artifact localization, thus addressing the coarse processing with artifact features. To further address the limitation that the proposed bi-directional attention module may not well capture global subtle forgery information in the artifact feature (e.g., textures or edges), we employ a fine-grained frequency attention module in the frequency domain. By doing so, we can obtain high-frequency information in the fine-grained features, which contains the global and subtle forgery information. Although these features from the diverse domains can be effectively and independently improved, fusing them directly does not effectively improve the detection performance. Therefore, we propose a feature superposition strategy that complements information from spatial and frequency domains. This strategy turns the feature components into the form of wave-like tokens, which are updated based on their phase, such that the distinctions between authentic and artifact features can be amplified. Our method demonstrates significant improvements over state-of-the-art (SOTA) methods on five public Deepfake datasets in capturing abnormalities across different manipulated operations and real-life. 

**Abstract (ZH)**: 深度生成伪造检测对于遏制其对社会的危害至关重要。然而，当前的深度生成伪造检测方法由于内在交互不足，未能充分探索不同域之间的艺术信息。这些交互是指不同域在特征提取过程后的融合和协调，对于识别复杂的伪造线索至关重要。为了进行更通用的深度生成伪造检测，本工作引入了一个新颖的双向注意力模块，从空间域捕捉艺术线索的局部位置信息，从而实现准确的艺术品局部化，解决粗略处理的艺术特征问题。为了进一步解决所提议的双向注意力模块可能难以在艺术特征中很好地捕捉全局微细伪造信息（如纹理或边缘）的问题，我们在频率域中采用了细粒度频率注意力模块。通过这样做，我们可以获取细粒度特征中的高频信息，这些信息包含全局和微细的伪造信息。尽管来自不同领域的这些特征可以有效且独立地提高，直接融合它们并不能显著提高检测性能。因此，我们提出了一种特征叠加策略，以补充空间域和频率域的信息。该策略将特征组件转换为波形标记的形式，并基于其相位更新，从而放大了真实和艺术特征之间的差异。我们的方法在五个公开的深度生成伪造数据集上，在不同操作和现实生活中的异常检测方面，显著优于现有最佳方法。 

---
# LLMs Love Python: A Study of LLMs' Bias for Programming Languages and Libraries 

**Title (ZH)**: LLMs偏好Python：关于LLMs对编程语言和库的偏见研究 

**Authors**: Lukas Twist, Jie M. Zhang, Mark Harman, Don Syme, Joost Noppen, Detlef Nauck  

**Link**: [PDF](https://arxiv.org/pdf/2503.17181)  

**Abstract**: Programming language and library choices are crucial to software reliability and security. Poor or inconsistent choices can lead to increased technical debt, security vulnerabilities, and even catastrophic failures in safety-critical systems. As Large Language Models (LLMs) play an increasing role in code generation, it is essential to understand how they make these decisions. However, little is known about their preferences when selecting programming languages and libraries for different coding tasks. To fill this gap, this study provides the first in-depth investigation into LLM preferences for programming languages and libraries used when generating code. We assess the preferences of eight diverse LLMs by prompting them to complete various coding tasks, including widely-studied benchmarks and the more practical task of generating the initial structural code for new projects (a crucial step that often determines a project's language or library choices).
Our findings reveal that LLMs heavily favour Python when solving language-agnostic problems, using it in 90%-97% of cases for benchmark tasks. Even when generating initial project code where Python is not a suitable language, it remains the most-used language in 58% of instances. Moreover, LLMs contradict their own language recommendations in 83% of project initialisation tasks, raising concerns about their reliability in guiding language selection. Similar biases toward well-established libraries further create serious discoverability challenges for newer open-source projects. These results highlight the need to improve LLMs' adaptability to diverse programming contexts and to develop mechanisms for mitigating programming language and library bias. 

**Abstract (ZH)**: 编程语言和库的选择对软件可靠性和安全性至关重要。不恰当或不一致的选择可能增加技术债务、安全漏洞，甚至导致安全关键系统出现灾难性故障。随着大型语言模型（LLMs）在代码生成中扮演越来越重要的角色，了解它们如何做出这些选择变得尤为重要。然而，关于它们在不同编程任务中选择编程语言和库的偏好知之甚少。为此，本研究首次深入探讨了LLMs在生成代码时选择编程语言和库的偏好。我们通过促使八种不同的LLMs完成各种编程任务，包括广泛研究的基准测试和生成新项目初始结构代码的更为实际的任务（这一步骤通常决定了项目的语言或库选择），来评估它们的偏好。

我们的研究发现，当解决跨语言问题时，LLMs强烈偏好使用Python，超过90%-97%的情况下会选择Python来完成基准测试任务。即使在生成初始项目代码的情况下，尽管Python不是最适合的语言，它仍然是58%情况下最常用的语言。此外，在58%的项目初始化任务中，LLMs与它们自己的语言推荐相悖，这引发了对其在指导语言选择方面可靠性的担忧。类似地，对现有库的偏向进一步给新的开源项目带来了严重的可发现性挑战。这些结果表明，需要改进LLMs在不同编程环境中的适应性，并开发机制以减轻编程语言和库的偏向。 

---
# DiTEC-WDN: A Large-Scale Dataset of Water Distribution Network Scenarios under Diverse Hydraulic Conditions 

**Title (ZH)**: DiTEC-WDN：多种水力条件下大规模水 Distribution Network 情景数据集 

**Authors**: Huy Truong, Andrés Tello, Alexander Lazovik, Victoria Degeler  

**Link**: [PDF](https://arxiv.org/pdf/2503.17167)  

**Abstract**: Privacy restrictions hinder the sharing of real-world Water Distribution Network (WDN) models, limiting the application of emerging data-driven machine learning, which typically requires extensive observations. To address this challenge, we propose the dataset DiTEC-WDN that comprises 36,000 unique scenarios simulated over either short-term (24 hours) or long-term (1 year) periods. We constructed this dataset using an automated pipeline that optimizes crucial parameters (e.g., pressure, flow rate, and demand patterns), facilitates large-scale simulations, and records discrete, synthetic but hydraulically realistic states under standard conditions via rule validation and post-hoc analysis. With a total of 228 million generated graph-based states, DiTEC-WDN can support a variety of machine-learning tasks, including graph-level, node-level, and link-level regression, as well as time-series forecasting. This contribution, released under a public license, encourages open scientific research in the critical water sector, eliminates the risk of exposing sensitive data, and fulfills the need for a large-scale water distribution network benchmark for study comparisons and scenario analysis. 

**Abstract (ZH)**: 隐私限制阻碍了真实水分布网络(WDN)模型的共享，限制了新兴数据驱动机器学习的应用，后者通常需要大量观测数据。为应对这一挑战，我们提出了由36,000个独特场景组成的DiTEC-WDN数据集，这些场景模拟了短期（24小时）或长期（1年）的时间段。我们通过一个自动化管道优化关键参数（如压力、流量和需求模式），实现大规模模拟，并通过规则验证和事后分析记录在标准条件下的离散、合成但水力现实的状态，总共生成了2.28亿个基于图的状态。DiTEC-WDN可以支持图级、节点级和连接级回归以及时间序列预测等多种机器学习任务。该贡献在公共许可下发布，促进了关键水领域中的开放科学研究，消除了暴露敏感数据的风险，并满足了大规模水分布网络基准测试的需求，以用于研究比较和情景分析。 

---
# Temporal-Guided Spiking Neural Networks for Event-Based Human Action Recognition 

**Title (ZH)**: 基于时间引导的事件驱动人体动作识别Spiking神经网络 

**Authors**: Siyuan Yang, Shilin Lu, Shizheng Wang, Meng Hwa Er, Zengwei Zheng, Alex C. Kot  

**Link**: [PDF](https://arxiv.org/pdf/2503.17132)  

**Abstract**: This paper explores the promising interplay between spiking neural networks (SNNs) and event-based cameras for privacy-preserving human action recognition (HAR). The unique feature of event cameras in capturing only the outlines of motion, combined with SNNs' proficiency in processing spatiotemporal data through spikes, establishes a highly synergistic compatibility for event-based HAR. Previous studies, however, have been limited by SNNs' ability to process long-term temporal information, essential for precise HAR. In this paper, we introduce two novel frameworks to address this: temporal segment-based SNN (\textit{TS-SNN}) and 3D convolutional SNN (\textit{3D-SNN}). The \textit{TS-SNN} extracts long-term temporal information by dividing actions into shorter segments, while the \textit{3D-SNN} replaces 2D spatial elements with 3D components to facilitate the transmission of temporal information. To promote further research in event-based HAR, we create a dataset, \textit{FallingDetection-CeleX}, collected using the high-resolution CeleX-V event camera $(1280 \times 800)$, comprising 7 distinct actions. Extensive experimental results show that our proposed frameworks surpass state-of-the-art SNN methods on our newly collected dataset and three other neuromorphic datasets, showcasing their effectiveness in handling long-range temporal information for event-based HAR. 

**Abstract (ZH)**: 基于事件的相机和脉冲神经网络在隐私保护的人体动作识别中的前景探索：TS-SNN和3D-SNN框架 

---
# Leveraging Language Models for Out-of-Distribution Recovery in Reinforcement Learning 

**Title (ZH)**: 利用语言模型进行强化学习中的分布外恢复 

**Authors**: Chan Kim, Seung-Woo Seo, Seong-Woo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.17125)  

**Abstract**: Deep Reinforcement Learning (DRL) has demonstrated strong performance in robotic control but remains susceptible to out-of-distribution (OOD) states, often resulting in unreliable actions and task failure. While previous methods have focused on minimizing or preventing OOD occurrences, they largely neglect recovery once an agent encounters such states. Although the latest research has attempted to address this by guiding agents back to in-distribution states, their reliance on uncertainty estimation hinders scalability in complex environments. To overcome this limitation, we introduce Language Models for Out-of-Distribution Recovery (LaMOuR), which enables recovery learning without relying on uncertainty estimation. LaMOuR generates dense reward codes that guide the agent back to a state where it can successfully perform its original task, leveraging the capabilities of LVLMs in image description, logical reasoning, and code generation. Experimental results show that LaMOuR substantially enhances recovery efficiency across diverse locomotion tasks and even generalizes effectively to complex environments, including humanoid locomotion and mobile manipulation, where existing methods struggle. The code and supplementary materials are available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于语言模型的异分布恢复（LaMOuR）：增强机器人控制中的异常状态恢复性能 

---
# The CASTLE 2024 Dataset: Advancing the Art of Multimodal Understanding 

**Title (ZH)**: CASTLE 2024 数据集：推动多模态理解的艺术 

**Authors**: Luca Rossetto, Werner Bailer, Duc-Tien Dang-Nguyen, Graham Healy, Björn Þór Jónsson, Onanong Kongmeesub, Hoang-Bao Le, Stevan Rudinac, Klaus Schöffmann, Florian Spiess, Allie Tran, Minh-Triet Tran, Quang-Linh Tran, Cathal Gurrin  

**Link**: [PDF](https://arxiv.org/pdf/2503.17116)  

**Abstract**: Egocentric video has seen increased interest in recent years, as it is used in a range of areas. However, most existing datasets are limited to a single perspective. In this paper, we present the CASTLE 2024 dataset, a multimodal collection containing ego- and exo-centric (i.e., first- and third-person perspective) video and audio from 15 time-aligned sources, as well as other sensor streams and auxiliary data. The dataset was recorded by volunteer participants over four days in a fixed location and includes the point of view of 10 participants, with an additional 5 fixed cameras providing an exocentric perspective. The entire dataset contains over 600 hours of UHD video recorded at 50 frames per second. In contrast to other datasets, CASTLE 2024 does not contain any partial censoring, such as blurred faces or distorted audio. The dataset is available via this https URL. 

**Abstract (ZH)**: 自视角视频近年来引起了广泛关注，因为它在多个领域中被应用。然而，现有大多数数据集仅限于单一视角。本文介绍了CASTLE 2024数据集，该数据集包含15个时间对齐的视觉和音频来源的多模态集合，以及其他传感器流和辅助数据，来自固定位置的志愿者参与者录制了为期四天的数据，其中包括10名参与者的第一人称和第三人称视角，另有5台固定相机提供第三人称视角。整个数据集包含超过600小时的50帧/秒录制的超高清视频。与现有数据集不同，CASTLE 2024没有包含任何部分遮挡，如模糊的脸部或失真的音频。该数据集可通过此链接访问。 

---
# FFaceNeRF: Few-shot Face Editing in Neural Radiance Fields 

**Title (ZH)**: FFaceNeRF：神经辐射场中的少量样本面部编辑 

**Authors**: Kwan Yun, Chaelin Kim, Hangyeul Shin, Junyong Noh  

**Link**: [PDF](https://arxiv.org/pdf/2503.17095)  

**Abstract**: Recent 3D face editing methods using masks have produced high-quality edited images by leveraging Neural Radiance Fields (NeRF). Despite their impressive performance, existing methods often provide limited user control due to the use of pre-trained segmentation masks. To utilize masks with a desired layout, an extensive training dataset is required, which is challenging to gather. We present FFaceNeRF, a NeRF-based face editing technique that can overcome the challenge of limited user control due to the use of fixed mask layouts. Our method employs a geometry adapter with feature injection, allowing for effective manipulation of geometry attributes. Additionally, we adopt latent mixing for tri-plane augmentation, which enables training with a few samples. This facilitates rapid model adaptation to desired mask layouts, crucial for applications in fields like personalized medical imaging or creative face editing. Our comparative evaluations demonstrate that FFaceNeRF surpasses existing mask based face editing methods in terms of flexibility, control, and generated image quality, paving the way for future advancements in customized and high-fidelity 3D face editing. The code is available on the {\href{this https URL}{project-page}}. 

**Abstract (ZH)**: Recent 3D人脸编辑方法利用掩模并通过神经辐射场（NeRF）产生了高质量的编辑图像。尽管这些方法表现令人印象深刻，但现有方法往往因使用预训练分割掩模而提供了有限的用户控制。为了利用具有期望布局的掩模，需要一个庞大的训练数据集，这具有挑战性。我们提出了FFaceNeRF，一种基于NeRF的人脸编辑技术，可以克服由于固定掩模布局导致的有限用户控制问题。该方法采用带特征注入的几何适配器，能够有效操纵几何属性。此外，我们采用潜在掺混进行三平面增强，这使得使用少量样本进行训练成为可能。这促进了模型对期望掩模布局的快速适应，对于个性化医疗成像或创意人脸编辑等领域至关重要。我们的比较评估表明，FFaceNeRF 在灵活性、控制能力和生成图像质量方面超越了现有的基于掩模的人脸编辑方法，为定制和高保真3D人脸编辑的未来进展铺平了道路。源代码可在项目页面（this https URL）获取。 

---
# Does a Rising Tide Lift All Boats? Bias Mitigation for AI-based CMR Segmentation 

**Title (ZH)**: 水涨船高：基于AI的CMR分割中的偏见缓解 

**Authors**: Tiarna Lee, Esther Puyol-Antón, Bram Ruijsink, Miaojing Shi, Andrew P. King  

**Link**: [PDF](https://arxiv.org/pdf/2503.17089)  

**Abstract**: Artificial intelligence (AI) is increasingly being used for medical imaging tasks. However, there can be biases in the resulting models, particularly when they were trained using imbalanced training datasets. One such example has been the strong race bias effect in cardiac magnetic resonance (CMR) image segmentation models. Although this phenomenon has been reported in a number of publications, little is known about the effectiveness of bias mitigation algorithms in this domain. We aim to investigate the impact of common bias mitigation methods to address bias between Black and White subjects in AI-based CMR segmentation models. Specifically, we use oversampling, importance reweighing and Group DRO as well as combinations of these techniques to mitigate the race bias. Furthermore, motivated by recent findings on the root causes of AI-based CMR segmentation bias, we evaluate the same methods using models trained and evaluated on cropped CMR images. We find that bias can be mitigated using oversampling, significantly improving performance for the underrepresented Black subjects whilst not significantly reducing the majority White subjects' performance. Group DRO also improves performance for Black subjects but not significantly, while reweighing decreases performance for Black subjects. Using a combination of oversampling and Group DRO also improves performance for Black subjects but not significantly. Using cropped images increases performance for both races and reduces the bias, whilst adding oversampling as a bias mitigation technique with cropped images reduces the bias further. 

**Abstract (ZH)**: 人工智能在心脏磁共振图像分割中的种族偏差及其缓解方法研究 

---
# Deterministic AI Agent Personality Expression through Standard Psychological Diagnostics 

**Title (ZH)**: 确定性人工智能代理心理特征的标准化心理诊断表达 

**Authors**: J. M. Diederik Kruijssen, Nicholas Emmons  

**Link**: [PDF](https://arxiv.org/pdf/2503.17085)  

**Abstract**: Artificial intelligence (AI) systems powered by large language models have become increasingly prevalent in modern society, enabling a wide range of applications through natural language interaction. As AI agents proliferate in our daily lives, their generic and uniform expressiveness presents a significant limitation to their appeal and adoption. Personality expression represents a key prerequisite for creating more human-like and distinctive AI systems. We show that AI models can express deterministic and consistent personalities when instructed using established psychological frameworks, with varying degrees of accuracy depending on model capabilities. We find that more advanced models like GPT-4o and o1 demonstrate the highest accuracy in expressing specified personalities across both Big Five and Myers-Briggs assessments, and further analysis suggests that personality expression emerges from a combination of intelligence and reasoning capabilities. Our results reveal that personality expression operates through holistic reasoning rather than question-by-question optimization, with response-scale metrics showing higher variance than test-scale metrics. Furthermore, we find that model fine-tuning affects communication style independently of personality expression accuracy. These findings establish a foundation for creating AI agents with diverse and consistent personalities, which could significantly enhance human-AI interaction across applications from education to healthcare, while additionally enabling a broader range of more unique AI agents. The ability to quantitatively assess and implement personality expression in AI systems opens new avenues for research into more relatable, trustworthy, and ethically designed AI. 

**Abstract (ZH)**: 由大型语言模型驱动的人工智能系统在现代社会中变得越来越普遍，通过自然语言交互实现广泛的应用。随着人工智能代理在日常生活中增多，它们的一般性和统一性表达形式对其吸引力和采用率构成显著限制。个性表达是创造更加人性化和独特的人工智能系统的关键前提。我们展示，在使用现有心理学框架进行指令时，人工智能模型可以表现出确定性和一致性的个性特征，其准确程度根据不同模型的能力而有所不同。我们发现，更先进的模型如GPT-4o和o1在Big Five和Myers-Briggs评估中表现出最高的个性表达准确性，进一步的分析表明，个性表达源自智能和推理能力的结合。我们的研究结果揭示了个性表达通过整体推理而非逐题优化机制发挥作用，响应尺度的度量标准显示出比测试尺度的度量标准更高的变异程度。此外，我们发现模型微调对沟通风格的影响独立于个性表达的准确性。这些发现为创建具有多样化和一致性的个性的人工智能代理奠定了基础，这可以在从教育到医疗保健等应用中显著增强人机交互，并且还可以使更独特的人工智能代理得以实现。在人工智能系统中定量评估和实施个性表达的能力为更加相关、可信和伦理设计的人工智能的更多研究打开了新的途径。 

---
# A Thorough Assessment of the Non-IID Data Impact in Federated Learning 

**Title (ZH)**: 非IID数据对联邦学习影响的彻底评估 

**Authors**: Daniel M. Jimenez-Gutierrez, Mehrdad Hassanzadeh, Aris Anagnostopoulos, Ioannis Chatzigiannakis, Andrea Vitaletti  

**Link**: [PDF](https://arxiv.org/pdf/2503.17070)  

**Abstract**: Federated learning (FL) allows collaborative machine learning (ML) model training among decentralized clients' information, ensuring data privacy. The decentralized nature of FL deals with non-independent and identically distributed (non-IID) data. This open problem has notable consequences, such as decreased model performance and more significant convergence times. Despite its importance, experimental studies systematically addressing all types of data heterogeneity (a.k.a. non-IIDness) remain scarce. We aim to fill this gap by assessing and quantifying the non-IID effect through a thorough empirical analysis. We use the Hellinger Distance (HD) to measure differences in distribution among clients. Our study benchmarks four state-of-the-art strategies for handling non-IID data, including label, feature, quantity, and spatiotemporal skewness, under realistic and controlled conditions. This is the first comprehensive analysis of the spatiotemporal skew effect in FL. Our findings highlight the significant impact of label and spatiotemporal skew non-IID types on FL model performance, with notable performance drops occurring at specific HD thresholds. Additionally, the FL performance is heavily affected mainly when the non-IIDness is extreme. Thus, we provide recommendations for FL research to tackle data heterogeneity effectively. Our work represents the most extensive examination of non-IIDness in FL, offering a robust foundation for future research. 

**Abstract (ZH)**: 联邦学习(Federal Learning, FL)允许去中心化客户端协作进行机器学习(ML)模型训练，确保数据隐私。FL的去中心化特性处理非独立且非同分布（non-IID）数据。这一开放问题导致了模型性能下降和收敛时间加长等显著后果。尽管其重要性不言而喻，但系统性地综合研究所有类型的数据异质性（即non-IID性）的实验研究仍然很少。我们旨在通过彻底的实证分析评估和量化non-IID效应。我们使用Hellinger距离（HD）来衡量客户端之间分布的差异。我们的研究在现实和受控条件下基准测试了四种最先进的处理non-IID数据的策略，包括标签、特征、数量和时空偏斜。这是我们首次对FL中时空偏斜效应进行全面分析。我们的研究结果强调了标签和时空偏斜non-IID类型对FL模型性能的显著影响，在特定的HD阈值下出现显著性能下降。此外，当non-IID性极端时，FL性能受到严重影响。因此，我们为有效应对数据异质性提供了研究建议。我们的工作是迄今为止对FL中non-IID性最全面的研究，为未来研究提供了坚实的基础。 

---
# PVChat: Personalized Video Chat with One-Shot Learning 

**Title (ZH)**: PVChat：基于单次学习的个性化视频聊天 

**Authors**: Yufei Shi, Weilong Yan, Gang Xu, Yumeng Li, Yuchen Li, Zhenxi Li, Fei Richard Yu, Ming Li, Si Yong Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2503.17069)  

**Abstract**: Video large language models (ViLLMs) excel in general video understanding, e.g., recognizing activities like talking and eating, but struggle with identity-aware comprehension, such as "Wilson is receiving chemotherapy" or "Tom is discussing with Sarah", limiting their applicability in smart healthcare and smart home environments. To address this limitation, we propose a one-shot learning framework PVChat, the first personalized ViLLM that enables subject-aware question answering (QA) from a single video for each subject. Our approach optimizes a Mixture-of-Heads (MoH) enhanced ViLLM on a synthetically augmented video-QA dataset, leveraging a progressive image-to-video learning strategy. Specifically, we introduce an automated augmentation pipeline that synthesizes identity-preserving positive samples and retrieves hard negatives from existing video corpora, generating a diverse training dataset with four QA types: existence, appearance, action, and location inquiries. To enhance subject-specific learning, we propose a ReLU Routing MoH attention mechanism, alongside two novel objectives: (1) Smooth Proximity Regularization for progressive learning through exponential distance scaling and (2) Head Activation Enhancement for balanced attention routing. Finally, we adopt a two-stage training strategy, transitioning from image pre-training to video fine-tuning, enabling a gradual learning process from static attributes to dynamic representations. We evaluate PVChat on diverse datasets covering medical scenarios, TV series, anime, and real-world footage, demonstrating its superiority in personalized feature understanding after learning from a single video, compared to state-of-the-art ViLLMs. 

**Abstract (ZH)**: 基于视频的个性化大型语言模型（PVChat）：一种单视频学习框架，实现主体意识下的问答能力 

---
# Replay4NCL: An Efficient Memory Replay-based Methodology for Neuromorphic Continual Learning in Embedded AI Systems 

**Title (ZH)**: Replay4NCL: 一种基于内存重演的嵌入式人工智能系统中神经形态持续学习高效方法 

**Authors**: Mishal Fatima Minhas, Rachmad Vidya Wicaksana Putra, Falah Awwad, Osman Hasan, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2503.17061)  

**Abstract**: Neuromorphic Continual Learning (NCL) paradigm leverages Spiking Neural Networks (SNNs) to enable continual learning (CL) capabilities for AI systems to adapt to dynamically changing environments. Currently, the state-of-the-art employ a memory replay-based method to maintain the old knowledge. However, this technique relies on long timesteps and compression-decompression steps, thereby incurring significant latency and energy overheads, which are not suitable for tightly-constrained embedded AI systems (e.g., mobile agents/robotics). To address this, we propose Replay4NCL, a novel efficient memory replay-based methodology for enabling NCL in embedded AI systems. Specifically, Replay4NCL compresses the latent data (old knowledge), then replays them during the NCL training phase with small timesteps, to minimize the processing latency and energy consumption. To compensate the information loss from reduced spikes, we adjust the neuron threshold potential and learning rate settings. Experimental results on the class-incremental scenario with the Spiking Heidelberg Digits (SHD) dataset show that Replay4NCL can preserve old knowledge with Top-1 accuracy of 90.43% compared to 86.22% from the state-of-the-art, while effectively learning new tasks, achieving 4.88x latency speed-up, 20% latent memory saving, and 36.43% energy saving. These results highlight the potential of our Replay4NCL methodology to further advances NCL capabilities for embedded AI systems. 

**Abstract (ZH)**: 神经形态连续学习（NCL）范式利用脉冲神经网络（SNNs）为AI系统提供连续学习能力，使其能够适应动态变化的环境。当前最先进的方法依赖于记忆回放技术来保留旧的知识，但该技术需要较长的时间步长和压缩解压缩步骤，从而导致显著的延迟和能量开销，这不适合紧约束嵌入式AI系统（例如移动代理/机器人）。为此，我们提出了Replay4NCL，一种新颖有效的记忆回放方法，以在嵌入式AI系统中实现NCL。具体而言，Replay4NCL压缩潜在数据（旧知识），然后在NCL训练阶段以短时间步长回放它们，以最小化处理延迟和能量消耗。为了抵消减少的脉冲带来的信息损失，我们调整了神经元阈值势能和学习率设置。在Spiking Heidelberg Digits (SHD)数据集上的类别增量场景实验结果表明，与最先进的技术相比，Replay4NCL在Top-1精度上保留了90.43%的旧知识，同时有效地学习新任务，实现了4.88倍的延迟加速、20%的潜在内存节省和36.43%的能量节省。这些结果突显了Replay4NCL方法在进一步推进嵌入式AI系统中NCL能力方面的潜力。 

---
# Data-Driven Optimization of EV Charging Station Placement Using Causal Discovery 

**Title (ZH)**: 使用因果发现驱动的电动汽车充电站位置优化 

**Authors**: Julius Stephan Junker, Rong Hu, Ziyue Li, Wolfgang Ketter  

**Link**: [PDF](https://arxiv.org/pdf/2503.17055)  

**Abstract**: This paper addresses the critical challenge of optimizing electric vehicle charging station placement through a novel data-driven methodology employing causal discovery techniques. While traditional approaches prioritize economic factors or power grid constraints, they often neglect empirical charging patterns that ultimately determine station utilization. We analyze extensive charging data from Palo Alto and Boulder (337,344 events across 100 stations) to uncover latent relationships between station characteristics and utilization. Applying structural learning algorithms (NOTEARS and DAGMA) to this data reveals that charging demand is primarily determined by three factors: proximity to amenities, EV registration density, and adjacency to high-traffic routes. These findings, consistent across multiple algorithms and urban contexts, challenge conventional infrastructure distribution strategies. We develop an optimization framework that translates these insights into actionable placement recommendations, identifying locations likely to experience high utilization based on the discovered dependency structures. The resulting site selection model prioritizes strategic clustering in high-amenity areas with substantial EV populations rather than uniform spatial distribution. Our approach contributes a framework that integrates empirical charging behavior into infrastructure planning, potentially enhancing both station utilization and user convenience. By focusing on data-driven insights instead of theoretical distribution models, we provide a more effective strategy for expanding charging networks that can adjust to various stages of EV market development. 

**Abstract (ZH)**: 本文通过一种新颖的数据驱动方法和因果发现技术，解决了电动车辆充电站布局优化的关键挑战。传统方法通常重视经济因素或电网约束，但往往忽视了实际充电模式，后者最终决定了充电站的使用情况。我们通过分析帕洛阿托和博尔德的大量充电数据（共计337,344个事件，覆盖100个充电站）来揭示充电站特性与使用情况之间的潜在关系。应用结构学习算法（NOTEARS和DAGMA）对这些数据进行分析，结果显示充电需求主要由三个因素决定：靠近便利设施的距离、电动汽车注册密度以及靠近繁忙路线的位置。这些发现一致地挑战了传统基础设施分布策略的有效性。我们开发了一种优化框架，将这些洞见转化为可操作的布局建议，根据发现的依赖结构确定了高利用概率的地点。最终的选址模型优先考虑在高便利设施区域和大量电动汽车人口集中的地区战略性地聚集充电站，而不是均匀的空间分布。本文提供的框架将实际的充电行为集成到基础设施规划中，可能提高充电站的使用率和用户便利性。通过聚焦数据驱动的洞察而非理论分布模型，我们提供了一种更有效的策略，以便根据电动汽车市场发展的不同阶段调整充电网络的扩展。 

---
# HAPI: A Model for Learning Robot Facial Expressions from Human Preferences 

**Title (ZH)**: HAPI：一种从人类偏好学习机器人面部表情的模型 

**Authors**: Dongsheng Yang, Qianying Liu, Wataru Sato, Takashi Minato, Chaoran Liu, Shin'ya Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2503.17046)  

**Abstract**: Automatic robotic facial expression generation is crucial for human-robot interaction, as handcrafted methods based on fixed joint configurations often yield rigid and unnatural behaviors. Although recent automated techniques reduce the need for manual tuning, they tend to fall short by not adequately bridging the gap between human preferences and model predictions-resulting in a deficiency of nuanced and realistic expressions due to limited degrees of freedom and insufficient perceptual integration. In this work, we propose a novel learning-to-rank framework that leverages human feedback to address this discrepancy and enhanced the expressiveness of robotic faces. Specifically, we conduct pairwise comparison annotations to collect human preference data and develop the Human Affective Pairwise Impressions (HAPI) model, a Siamese RankNet-based approach that refines expression evaluation. Results obtained via Bayesian Optimization and online expression survey on a 35-DOF android platform demonstrate that our approach produces significantly more realistic and socially resonant expressions of Anger, Happiness, and Surprise than those generated by baseline and expert-designed methods. This confirms that our framework effectively bridges the gap between human preferences and model predictions while robustly aligning robotic expression generation with human affective responses. 

**Abstract (ZH)**: 自动机器人面部表情生成对于人机交互至关重要，因为基于固定关节配置的手工制作方法往往会产生僵硬和不自然的行为。虽然最近的自动化技术减少了手动调整的需求，但它们往往由于未能充分弥合人类偏好与模型预测之间的差距而有所不足，导致表情细腻和真实性不足，这主要是因为自由度有限和感知整合不足。在本项工作中，我们提出了一种新颖的学习排名框架，利用人类反馈来解决这一差异，并增强了机器人类别表情的表现力。具体而言，我们进行了成对比较注释以收集人类偏好数据，并开发了基于双胞胎排名网络的Human Affective Pairwise Impressions (HAPI) 模型，该模型用于细化表情评估。通过贝叶斯优化和在线表情调查获得的结果表明，与基线方法和专家设计的方法相比，我们的方法生成了显著更加真实和具有社会共鸣效果的愤怒、快乐和惊讶表情。这证明了我们的框架有效弥合了人类偏好与模型预测之间的差距，并且能够稳健地将机器人类别表情生成与人类情感反应对齐。 

---
# Summarization Metrics for Spanish and Basque: Do Automatic Scores and LLM-Judges Correlate with Humans? 

**Title (ZH)**: 西班牙语和巴斯克语摘要评价指标：自动分数和LLM评审员与人类评分相关吗？ 

**Authors**: Jeremy Barnes, Naiara Perez, Alba Bonet-Jover, Begoña Altuna  

**Link**: [PDF](https://arxiv.org/pdf/2503.17039)  

**Abstract**: Studies on evaluation metrics and LLM-as-a-Judge models for automatic text summarization have largely been focused on English, limiting our understanding of their effectiveness in other languages. Through our new dataset BASSE (BAsque and Spanish Summarization Evaluation), we address this situation by collecting human judgments on 2,040 abstractive summaries in Basque and Spanish, generated either manually or by five LLMs with four different prompts. For each summary, annotators evaluated five criteria on a 5-point Likert scale: coherence, consistency, fluency, relevance, and 5W1H. We use these data to reevaluate traditional automatic metrics used for evaluating summaries, as well as several LLM-as-a-Judge models that show strong performance on this task in English. Our results show that currently proprietary judge LLMs have the highest correlation with human judgments, followed by criteria-specific automatic metrics, while open-sourced judge LLMs perform poorly. We release BASSE and our code publicly, along with the first large-scale Basque summarization dataset containing 22,525 news articles with their subheads. 

**Abstract (ZH)**: 关于自动文本摘要的评估指标和LLM-as-a-Judge模型的研究主要集中在英语上，限制了我们对其在其他语言中的有效性的理解。通过我们的新数据集BASSE（巴斯克语和西班牙语摘要评价），我们收集了2040个巴斯克语和西班牙语的抽象总结的人工判断，这些总结要么是人工生成的，要么是由五个具有四种不同提示的LLM生成的。每个摘要的注释者根据一致性、连贯性、流畅性、相关性以及5W1H这五个标准在5点李克特量表上进行评价。我们使用这些数据重新评估传统的自动评估摘要的方法，以及在该任务上显示强大性能的几种LLM-as-a-Judge模型。结果显示，当前的专有法官LLM与人力判断的相关性最高，其次是特定标准的自动评估指标，而开源的法官LLM表现不佳。我们已将BASSE数据集、代码以及包含22,525篇新闻文章及其副标题的首个大规模巴斯克语摘要数据集公开发布。 

---
# An Attentive Representative Sample Selection Strategy Combined with Balanced Batch Training for Skin Lesion Segmentation 

**Title (ZH)**: 带有平衡批次训练的注意力代表性样本选择策略用于皮肤病变分割 

**Authors**: Stephen Lloyd-Brown, Susan Francis, Caroline Hoad, Penny Gowland, Karen Mullinger, Andrew French, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17034)  

**Abstract**: An often overlooked problem in medical image segmentation research is the effective selection of training subsets to annotate from a complete set of unlabelled data. Many studies select their training sets at random, which may lead to suboptimal model performance, especially in the minimal supervision setting where each training image has a profound effect on performance outcomes. This work aims to address this issue. We use prototypical contrasting learning and clustering to extract representative and diverse samples for annotation. We improve upon prior works with a bespoke cluster-based image selection process. Additionally, we introduce the concept of unsupervised balanced batch dataloading to medical image segmentation, which aims to improve model learning with minimally annotated data. We evaluated our method on a public skin lesion dataset (ISIC 2018) and compared it to another state-of-the-art data sampling method. Our method achieved superior performance in a low annotation budget scenario. 

**Abstract (ZH)**: 医学图像分割研究中常被忽视的一个问题是如何有效从完全未标注数据集中选择注释训练子集。许多研究随机选择训练集，这可能导致模型性能不佳，尤其是在最小监督设置中，每个训练图像对性能结果有深远影响。本工作旨在解决这一问题。我们采用原型对比学习和聚类来提取具有代表性和多样性的样本进行注释。我们通过定制的基于聚类的图像选择过程改进了先前工作。此外，我们提出了无监督平衡批次数据加载的概念，应用于医学图像分割，旨在使用少量标注数据提高模型学习效果。我们在一个公开的皮肤病变数据集（ISIC 2018）上评估了我们的方法，并将其与另一种最先进的数据采样方法进行了比较。在低标注预算场景中，我们的方法表现出更优的性能。 

---
# Exploring the Efficacy of Partial Denoising Using Bit Plane Slicing for Enhanced Fracture Identification: A Comparative Study of Deep Learning-Based Approaches and Handcrafted Feature Extraction Techniques 

**Title (ZH)**: 基于位平面分割的局部去噪方法增强骨折识别效果探索：深度学习方法与手工特征提取技术的比较研究 

**Authors**: Snigdha Paul, Sambit Mallick, Anindya Sen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17030)  

**Abstract**: Computer vision has transformed medical diagnosis, treatment, and research through advanced image processing and machine learning techniques. Fracture classification, a critical area in healthcare, has greatly benefited from these advancements, yet accurate detection is challenged by complex patterns and image noise. Bit plane slicing enhances medical images by reducing noise interference and extracting informative features. This research explores partial denoising techniques to provide practical solutions for improved fracture analysis, ultimately enhancing patient care. The study explores deep learning model DenseNet and handcrafted feature extraction. Decision Tree and Random Forest, were employed to train and evaluate distinct image representations. These include the original image, the concatenation of the four bit planes from the LSB as well as MSB, the fully denoised image, and an image consisting of 6 bit planes from MSB and 2 denoised bit planes from LSB. The purpose of forming these diverse image representations is to analyze SNR as well as classification accuracy and identify the bit planes that contain the most informative features. Moreover, the study delves into the significance of partial denoising techniques in preserving crucial features, leading to improvements in classification results. Notably, this study shows that employing the Random Forest classifier, the partially denoised image representation exhibited a testing accuracy of 95.61% surpassing the performance of other image representations. The outcomes of this research provide valuable insights into the development of efficient preprocessing, feature extraction and classification approaches for fracture identification. By enhancing diagnostic accuracy, these advancements hold the potential to positively impact patient care and overall medical outcomes. 

**Abstract (ZH)**: 计算机视觉通过对先进图像处理和机器学习技术的应用，已革新了医疗诊断、治疗和研究。骨折分类这一医疗领域关键环节极大地受益于这些进步，但准确检测仍受复杂模式和图像噪声的挑战。位平面切片通过减少噪声干扰并提取有用特征，增强了医疗图像。本研究探讨部分去噪技术，以提供提高骨折分析的实用解决方案，最终提升患者护理质量。研究探讨了深度学习模型DenseNet和手工特征提取，并使用决策树和随机森林对不同的图像表示进行训练和评估。这些表示包括原始图像、最下位平面(LSB)和最上位平面(MSB)的四位位平面拼接、完全去噪图像，以及由MSB的6位平面和LSB的2个去噪位平面组成的图像。形成这些多样化的图像表示旨在分析信噪比和分类准确性，并确定包含最有用特征的位平面。此外，研究深入探讨了部分去噪技术在保持重要特征方面的意义，从而改善分类结果。值得注意的是，本研究显示，使用随机森林分类器的部分去噪图像表示在测试中的准确率为95.61%，超过了其他图像表示的性能。本研究的结果为骨折识别的高效预处理、特征提取和分类方法的发展提供了宝贵的见解。通过提升诊断准确性，这些进步有望对患者护理和整体医疗结果产生积极影响。 

---
# Symbolic Audio Classification via Modal Decision Tree Learning 

**Title (ZH)**: 模态决策树学习驱动的符号音频分类 

**Authors**: Enrico Marzano, Giovanni Pagliarini, Riccardo Pasini, Guido Sciavicco, Ionel Eduard Stan  

**Link**: [PDF](https://arxiv.org/pdf/2503.17018)  

**Abstract**: The range of potential applications of acoustic analysis is wide. Classification of sounds, in particular, is a typical machine learning task that received a lot of attention in recent years. The most common approaches to sound classification are sub-symbolic, typically based on neural networks, and result in black-box models with high performances but very low transparency. In this work, we consider several audio tasks, namely, age and gender recognition, emotion classification, and respiratory disease diagnosis, and we approach them with a symbolic technique, that is, (modal) decision tree learning. We prove that such tasks can be solved using the same symbolic pipeline, that allows to extract simple rules with very high accuracy and low complexity. In principle, all such tasks could be associated to an autonomous conversation system, which could be useful in different contexts, such as an automatic reservation agent for an hospital or a clinic. 

**Abstract (ZH)**: 声学分析潜在应用范围广泛。声音分类是一项典型的机器学习任务，近年来受到了广泛关注。声音分类的最常见方法是无符号的，通常基于神经网络，产生高性能但透明度极低的黑盒模型。在本工作中，我们考虑了年龄和性别识别、情绪分类以及呼吸道疾病诊断等几个音频任务，并采用符号技术，即（模态）决策树学习方法来解决这些问题。我们证明了这些任务可以使用相同的符号管道解决，该管道能够提取简单规则并达到非常高准确性且低复杂度。原则上，所有此类任务都可以与自主对话系统相关联，在医院或诊所等不同场景中可能很有用。 

---
# Developing Critical Thinking in Second Language Learners: Exploring Generative AI like ChatGPT as a Tool for Argumentative Essay Writing 

**Title (ZH)**: 在第二语言学习者中培养批判性思维：探索像ChatGPT这样的生成型AI作为论说文写作工具的研究 

**Authors**: Simon Suh, Jihyuk Bang, Ji Woo Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.17013)  

**Abstract**: This study employs the Paul-Elder Critical Thinking Model and Tan's argumentative writing framework to create a structured methodology. This methodology, ChatGPT Guideline for Critical Argumentative Writing (CGCAW) framework, integrates the models with ChatGPT's capabilities to guide L2 learners in utilizing ChatGPT to enhance their critical thinking skills. A quantitative experiment was conducted with 10 participants from a state university, divided into experimental and control groups. The experimental group utilized the CGCAW framework, while the control group used ChatGPT without specific guidelines. Participants wrote an argumentative essay within a 40-minute timeframe, and essays were evaluated by three assessors: ChatGPT, Grammarly, and a course instructor. Results indicated that the experimental group showed improvements in clarity, logical coherence, and use of evidence, demonstrating ChatGPT's potential to enhance specific aspects of argumentative writing. However, the control group performed better in overall language mechanics and articulation of main arguments, indicating areas where the CGCAW framework could be further refined. This study highlights the need for further research to optimize the use of AI tools like ChatGPT in L2 learning environments to enhance critical thinking and writing skills. 

**Abstract (ZH)**: 基于Paul-Elder批判性思维模型和Tan的论说文框架的ChatGPT论说文指南（CGCAW）方法论：一项促进二语学习者批判性思维和论说文写作能力的研究 

---
# Targetless 6DoF Calibration of LiDAR and 2D Scanning Radar Based on Cylindrical Occupancy 

**Title (ZH)**: 基于圆柱体 occupancy 的无目标六自由度 LiDAR 和 2D 扫描雷达标定 

**Authors**: Weimin Wang, Yu Du, Ting Yang, Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17002)  

**Abstract**: Owing to the capability for reliable and all-weather long-range sensing, the fusion of LiDAR and Radar has been widely applied to autonomous vehicles for robust perception. In practical operation, well manually calibrated extrinsic parameters, which are crucial for the fusion of multi-modal sensors, may drift due to the vibration. To address this issue, we present a novel targetless calibration approach, termed LiRaCo, for the extrinsic 6DoF calibration of LiDAR and Radar sensors. Although both types of sensors can obtain geometric information, bridging the geometric correspondences between multi-modal data without any clues of explicit artificial markers is nontrivial, mainly due to the low vertical resolution of scanning Radar. To achieve the targetless calibration, LiRaCo leverages a spatial occupancy consistency between LiDAR point clouds and Radar scans in a common cylindrical representation, considering the increasing data sparsity with distance for both sensors. Specifically, LiRaCo expands the valid Radar scanned pixels into 3D occupancy grids to constrain LiDAR point clouds based on spatial consistency. Consequently, a cost function involving extrinsic calibration parameters is formulated based on the spatial overlap of 3D grids and LiDAR points. Extrinsic parameters are finally estimated by optimizing the cost function. Comprehensive quantitative and qualitative experiments on two real outdoor datasets with different LiDAR sensors demonstrate the feasibility and accuracy of the proposed method. The source code will be publicly available. 

**Abstract (ZH)**: 基于LiDAR和雷达的无目标外参6DoFcalibration方法LiRaCo 

---
# Enabling Versatile Controls for Video Diffusion Models 

**Title (ZH)**: 支持视频扩散模型的多功能控制 

**Authors**: Xu Zhang, Hao Zhou, Haoming Qin, Xiaobin Lu, Jiaxing Yan, Guanzhong Wang, Zeyu Chen, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16983)  

**Abstract**: Despite substantial progress in text-to-video generation, achieving precise and flexible control over fine-grained spatiotemporal attributes remains a significant unresolved challenge in video generation research. To address these limitations, we introduce VCtrl (also termed PP-VCtrl), a novel framework designed to enable fine-grained control over pre-trained video diffusion models in a unified manner. VCtrl integrates diverse user-specified control signals-such as Canny edges, segmentation masks, and human keypoints-into pretrained video diffusion models via a generalizable conditional module capable of uniformly encoding multiple types of auxiliary signals without modifying the underlying generator. Additionally, we design a unified control signal encoding pipeline and a sparse residual connection mechanism to efficiently incorporate control representations. Comprehensive experiments and human evaluations demonstrate that VCtrl effectively enhances controllability and generation quality. The source code and pre-trained models are publicly available and implemented using the PaddlePaddle framework at this http URL. 

**Abstract (ZH)**: 尽管在文本到视频生成方面取得了显著进展，但在视频生成研究中实现精确灵活的细粒度时空属性控制仍然是一个重要的未解决问题。为了解决这些局限性，我们引入了VCtrl（也称为PP-VCtrl）这一新型框架，旨在以统一的方式对预训练的视频扩散模型进行细粒度控制。VCtrl通过一个通用的条件模块将用户指定的控制信号（如Canny边缘、分割掩码和人体关键点）整合到预训练的视频扩散模型中，该模块能够均匀编码多种类型的辅助信号而无需修改底层生成器。此外，我们设计了一种统一的控制信号编码管道和一种稀疏残差连接机制，以高效地整合控制表示。全面的实验和人工评估表明，VCtrl显著提高了可控性和生成质量。源代码和预训练模型已公开，并使用PaddlePaddle框架在以下网址实现：this http URL。 

---
# Token Dynamics: Towards Efficient and Dynamic Video Token Representation for Video Large Language Models 

**Title (ZH)**: Token 动态分析：面向高效且动态的视频令牌表示以用于视频大规模语言模型 

**Authors**: Haichao Zhang, Zhuowei Li, Dimitris Metaxas, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16980)  

**Abstract**: Token-based video representation has emerged as a promising approach for enabling large language models to interpret video content. However, existing token reduction techniques, such as token pruning and token merging, often disrupt essential spatial-temporal positional embeddings, failing to adequately balance computational efficiency with fewer tokens. Consequently, these methods result in relatively lengthy token sequences, limiting their applicability in scenarios requiring extreme token compression, such as video large language models. In this paper, we introduce the novel task of extreme short token reduction, aiming to represent extensive video sequences with a minimal number of tokens. To address this challenge, we propose Token Dynamics, a new video representation framework that dynamically reduces token count while preserving spatial-temporal coherence. Specifically, we disentangle video representations by separating visual embeddings from grid-level motion information, structuring them into: 1. a concise token base, created by clustering tokens that describe object-level content; 2. a token dynamics map, capturing detailed spatial-temporal motion patterns across grids. Furthermore, we introduce a cross-dynamics attention mechanism that integrates motion features into the token base without increasing token length, thereby maintaining compactness and spatial-temporal integrity. The experiments demonstrate a reduction of token count to merely 0.07% of the original tokens, with only a minor performance drop of 1.13%. Additionally, we propose two novel subtasks within extreme token reduction (fixed-length and adaptive-length compression), both effectively representing long token sequences for video-language tasks. Our method offers significantly lower theoretical complexity, fewer tokens, and enhanced throughput, thus providing an efficient solution for video LLMs. 

**Abstract (ZH)**: 基于令牌的视频表示作为使大型语言模型能够解释视频内容的一种有前途的方法已经 emergence。然而，现有的令牌缩减技术，例如令牌裁剪和令牌合并，往往会破坏重要的空时位置嵌入，无法充分平衡计算效率和较少的令牌数量。因此，这些方法会导致相对较长的令牌序列，限制了其在需要极端令牌压缩的场景（如视频大型语言模型）中的应用。在本文中，我们引入了一项全新的极端短令牌缩减任务，旨在使用最少的令牌表示大量的视频序列。为了解决这一挑战，我们提出了一种新的视频表示框架——令牌动态，它可以在保持空时连贯性的同时动态减少令牌数量。具体而言，我们通过分离视觉嵌入和网格级别的运动信息来解耦视频表示，将它们结构化为：1. 一个简洁的令牌基础，通过聚类描述对象级内容的令牌创建；2. 一个令牌动态映射，捕捉网格之间详细的空时运动模式。此外，我们还引入了一种跨动态注意力机制，可以在不增加令牌长度的情况下将运动特征整合到令牌基础中，从而保持紧凑性和空时完整性。实验结果表明，令牌数量减少了原始令牌的0.07%，性能仅下降了1.13%。此外，我们还在极端令牌缩减中提出了两个新的子任务（固定长度和自适应长度压缩），两者都能有效地表示长令牌序列以供视频-语言任务使用。我们的方法提供了显著更低的理论复杂度、更少的令牌和增强的吞吐量，从而为视频LLMs提供了高效的解决方案。 

---
# GeoT: Geometry-guided Instance-dependent Transition Matrix for Semi-supervised Tooth Point Cloud Segmentation 

**Title (ZH)**: GeoT: 基于几何引导的实例自适应转换矩阵的半监督牙齿点云分割 

**Authors**: Weihao Yu, Xiaoqing Guo, Chenxin Li, Yifan Liu, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16976)  

**Abstract**: Achieving meticulous segmentation of tooth point clouds from intra-oral scans stands as an indispensable prerequisite for various orthodontic applications. Given the labor-intensive nature of dental annotation, a significant amount of data remains unlabeled, driving increasing interest in semi-supervised approaches. One primary challenge of existing semi-supervised medical segmentation methods lies in noisy pseudo labels generated for unlabeled data. To address this challenge, we propose GeoT, the first framework that employs instance-dependent transition matrix (IDTM) to explicitly model noise in pseudo labels for semi-supervised dental segmentation. Specifically, to handle the extensive solution space of IDTM arising from tens of thousands of dental points, we introduce tooth geometric priors through two key components: point-level geometric regularization (PLGR) to enhance consistency between point adjacency relationships in 3D and IDTM spaces, and class-level geometric smoothing (CLGS) to leverage the fixed spatial distribution of tooth categories for optimal IDTM estimation. Extensive experiments performed on the public Teeth3DS dataset and private dataset demonstrate that our method can make full utilization of unlabeled data to facilitate segmentation, achieving performance comparable to fully supervised methods with only $20\%$ of the labeled data. 

**Abstract (ZH)**: 实现口腔扫描牙齿点云的精细分割是各种正畸应用不可或缺的前提。鉴于牙齿注释的劳动密集型特性，大量数据仍未标注，这推动了对半监督方法的日益浓厚兴趣。现有的半监督医学分割方法的主要挑战之一是在未标注数据上生成的噪声伪标签。为应对这一挑战，我们提出了GeoT框架，它是首个采用实例依赖转换矩阵（IDTM）明确建模半监督牙齿分割中伪标签噪声的方法。具体而言，为了处理源自数万个牙齿点的IDTM的庞大解决方案空间，我们通过两个关键组件引入牙齿几何先验：点级几何正则化（PLGR），以增强3D中点邻接关系和IDTM空间之间的一致性；以及类别级几何平滑（CLGS），以利用牙齿类别固定的空间分布实现最优IDTM估计。在公共Teeth3DS数据集和私有数据集上进行的大量实验表明，我们的方法能够充分利用未标注数据以促进分割，仅使用20%的标注数据即可达到与全监督方法相当的性能。 

---
# Assessing Consistency and Reproducibility in the Outputs of Large Language Models: Evidence Across Diverse Finance and Accounting Tasks 

**Title (ZH)**: 评估大型语言模型输出的一致性和可重复性：跨 diverse 金融和会计任务的证据 

**Authors**: Julian Junyan Wang, Victor Xiaoqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16974)  

**Abstract**: This study provides the first comprehensive assessment of consistency and reproducibility in Large Language Model (LLM) outputs in finance and accounting research. We evaluate how consistently LLMs produce outputs given identical inputs through extensive experimentation with 50 independent runs across five common tasks: classification, sentiment analysis, summarization, text generation, and prediction. Using three OpenAI models (GPT-3.5-turbo, GPT-4o-mini, and GPT-4o), we generate over 3.4 million outputs from diverse financial source texts and data, covering MD&As, FOMC statements, finance news articles, earnings call transcripts, and financial statements. Our findings reveal substantial but task-dependent consistency, with binary classification and sentiment analysis achieving near-perfect reproducibility, while complex tasks show greater variability. More advanced models do not consistently demonstrate better consistency and reproducibility, with task-specific patterns emerging. LLMs significantly outperform expert human annotators in consistency and maintain high agreement even where human experts significantly disagree. We further find that simple aggregation strategies across 3-5 runs dramatically improve consistency. Simulation analysis reveals that despite measurable inconsistency in LLM outputs, downstream statistical inferences remain remarkably robust. These findings address concerns about what we term "G-hacking," the selective reporting of favorable outcomes from multiple Generative AI runs, by demonstrating that such risks are relatively low for finance and accounting tasks. 

**Abstract (ZH)**: 本研究提供了对金融与会计研究中大型语言模型（LLM）输出一致性和可再现性的首次全面评估。通过在五个常见任务（分类、情感分析、总结、文本生成和预测）上进行广泛实验（共50次独立运行），评估了给定相同输入时LLM的输出一致性。使用三个OpenAI模型（GPT-3.5-turbo、GPT-4o-mini和GPT-4o），我们从多样化的金融源文本和数据中生成了超过340万条输出，涵盖MD&A、FOMC声明、金融新闻文章、收益会议 transcript 和财务报表。研究发现，输出在任务上表现出显著但任务依赖的一致性，二分类和情感分析几乎完全可再现，而复杂任务则表现出更大的变化性。进阶模型在一致性与可再现性上并不始终表现得更好，且出现特定任务模式。LLM在一致性上显著优于专家人工注释者，并在专家分歧显著的情况下仍保持高水平的一致性。进一步发现，简单聚合策略（跨3-5次运行）显著提高了一致性。模拟分析表明，尽管LLM输出存在可测量的不一致性，下游统计推断依然表现出色。这些发现缓解了所谓的“G-黑客”现象，即从多个生成AI运行中选择性报告有利结果的问题，证明了在金融与会计任务中这种风险较低。 

---
# ARFlow: Human Action-Reaction Flow Matching with Physical Guidance 

**Title (ZH)**: ARFlow: 基于物理指导的人类动作-反应流匹配 

**Authors**: Wentao Jiang, Jingya Wang, Haotao Lu, Kaiyang Ji, Baoxiong Jia, Siyuan Huang, Ye Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16973)  

**Abstract**: Human action-reaction synthesis, a fundamental challenge in modeling causal human interactions, plays a critical role in applications ranging from virtual reality to social robotics. While diffusion-based models have demonstrated promising performance, they exhibit two key limitations for interaction synthesis: reliance on complex noise-to-reaction generators with intricate conditional mechanisms, and frequent physical violations in generated motions. To address these issues, we propose Action-Reaction Flow Matching (ARFlow), a novel framework that establishes direct action-to-reaction mappings, eliminating the need for complex conditional mechanisms. Our approach introduces two key innovations: an x1-prediction method that directly outputs human motions instead of velocity fields, enabling explicit constraint enforcement; and a training-free, gradient-based physical guidance mechanism that effectively prevents body penetration artifacts during sampling. Extensive experiments on NTU120 and Chi3D datasets demonstrate that ARFlow not only outperforms existing methods in terms of Fréchet Inception Distance and motion diversity but also significantly reduces body collisions, as measured by our new Intersection Volume and Intersection Frequency metrics. 

**Abstract (ZH)**: 基于人类动作-反作用合成的人类因果交互建模是一项基本挑战，对于从虚拟现实到社会机器人应用等领域起着关键作用。尽管基于扩散的方法已经显示出有希望的性能，但它们在交互合成方面存在两个关键局限性：对复杂噪声到反作用生成器的依赖性及生成动作中的频繁物理违反现象。为了解决这些问题，我们提出了一种新的框架，称为动作-反作用流动匹配（ARFlow），该框架直接建立了动作到反作用的映射，消除了复杂条件机制的需要。我们的方法引入了两个关键创新：一种x1预测方法，直接输出人类动作而不是速度场，从而允许显式约束的施加；以及一种无需训练的梯度基于物理引导机制，在采样过程中有效防止身体穿插现象。在NTU120和Chi3D数据集上的广泛实验表明，ARFlow不仅在弗雷歇入学距离和动作多样性方面优于现有方法，还通过我们提出的新交集体积和交集频率度量显著减少了身体碰撞。 

---
# From Faces to Voices: Learning Hierarchical Representations for High-quality Video-to-Speech 

**Title (ZH)**: 从面部到声音：学习多层次表示以实现高质量视频到语音转换 

**Authors**: Ji-Hoon Kim, Jeongsoo Choi, Jaehun Kim, Chaeyoung Jung, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2503.16956)  

**Abstract**: The objective of this study is to generate high-quality speech from silent talking face videos, a task also known as video-to-speech synthesis. A significant challenge in video-to-speech synthesis lies in the substantial modality gap between silent video and multi-faceted speech. In this paper, we propose a novel video-to-speech system that effectively bridges this modality gap, significantly enhancing the quality of synthesized speech. This is achieved by learning of hierarchical representations from video to speech. Specifically, we gradually transform silent video into acoustic feature spaces through three sequential stages -- content, timbre, and prosody modeling. In each stage, we align visual factors -- lip movements, face identity, and facial expressions -- with corresponding acoustic counterparts to ensure the seamless transformation. Additionally, to generate realistic and coherent speech from the visual representations, we employ a flow matching model that estimates direct trajectories from a simple prior distribution to the target speech distribution. Extensive experiments demonstrate that our method achieves exceptional generation quality comparable to real utterances, outperforming existing methods by a significant margin. 

**Abstract (ZH)**: 本研究的目标是从静音说话人脸视频中生成高质量语音，这一任务也被称为视频到语音合成。视频到语音合成中的一个重大挑战在于静音视频与多维语音之间巨大的模态差距。本文提出了一种新颖的视频到语音系统，有效地弥合了这一模态差距，大幅提升了合成语音的质量。这一目标通过对视频到语音进行层次化表示学习实现。具体而言，我们通过三个连续阶段逐渐将静音视频转换为声学特征空间——内容建模、音色建模和语调建模。在每个阶段中，我们通过将视觉因素——唇部运动、面部身份和面部表情——与相应的声学对应物对齐，确保无缝转换。此外，为了从视觉表示生成现实且连贯的语音，我们采用了流匹配模型，该模型从简单的先验分布直接估计到目标语音分布的轨迹。广泛实验表明，本方法的生成质量出众，可与真实语音媲美，显著优于现有方法。 

---
# On-Sensor Convolutional Neural Networks with Early-Exits 

**Title (ZH)**: 基于传感器的卷积神经网络及早期退出机制 

**Authors**: Hazem Hesham Yousef Shalby, Arianna De Vecchi, Alice Scandelli, Pietro Bartoli, Diana Trojaniello, Manuel Roveri, Federica Villa  

**Link**: [PDF](https://arxiv.org/pdf/2503.16939)  

**Abstract**: Tiny Machine Learning (TinyML) is a novel research field aiming at integrating Machine Learning (ML) within embedded devices with limited memory, computation, and energy. Recently, a new branch of TinyML has emerged, focusing on integrating ML directly into the sensors to further reduce the power consumption of embedded devices. Interestingly, despite their state-of-the-art performance in many tasks, none of the current solutions in the literature aims to optimize the implementation of Convolutional Neural Networks (CNNs) operating directly into sensors. In this paper, we introduce for the first time in the literature the optimized design and implementation of Depth-First CNNs operating on the Intelligent Sensor Processing Unit (ISPU) within an Inertial Measurement Unit (IMU) by STMicroelectronics. Our approach partitions the CNN between the ISPU and the microcontroller (MCU) and employs an Early-Exit mechanism to stop the computations on the IMU when enough confidence about the results is achieved, hence significantly reducing power consumption. When using a NUCLEO-F411RE board, this solution achieved an average current consumption of 4.8 mA, marking an 11% reduction compared to the regular inference pipeline on the MCU, while having equal accuracy. 

**Abstract (ZH)**: TinyML中基于ISPU的优化Depth-First CNN设计与实现：一种减少IMU功耗的方法 

---
# Rude Humans and Vengeful Robots: Examining Human Perceptions of Robot Retaliatory Intentions in Professional Settings 

**Title (ZH)**: 粗鲁的人和报复心强的机器人：探究专业环境中人类对机器人报复意图的感知 

**Authors**: Kate Letheren, Nicole Robinson  

**Link**: [PDF](https://arxiv.org/pdf/2503.16932)  

**Abstract**: Humans and robots are increasingly working in personal and professional settings. In workplace settings, humans and robots may work together as colleagues, potentially leading to social expectations, or violation thereof. Extant research has primarily sought to understand social interactions and expectations in personal rather than professional settings, and none of these studies have examined negative outcomes arising from violations of social expectations. This paper reports the results of a 2x3 online experiment that used a unique first-person perspective video to immerse participants in a collaborative workplace setting. The results are nuanced and reveal that while robots are expected to act in accordance with social expectations despite human behavior, there are benefits for robots perceived as being the bigger person in the face of human rudeness. Theoretical and practical implications are provided which discuss the import of these findings for the design of social robots. 

**Abstract (ZH)**: 人类和机器人在个人和专业环境中 increasingly 工作。在工作场所，人类和机器人可能互相作为同事合作，这可能导致社会期望或其违背。现有研究主要关注个人而非专业环境中的社会互动和期望，且这些研究均未探讨违背社会期望所产生的负面影响。本文报告了一项使用独特第一人称视角视频的 2x3 在线实验结果，使参与者沉浸在协作工作环境中。结果显示，尽管机器人应根据社会期望行事，但被视为在面对人类粗鲁行为时更大度的机器人有其优势。文章提出了理论和实践意义，讨论了这些发现对设计社会机器人的重要性。 

---
# TEMPO: Temporal Preference Optimization of Video LLMs via Difficulty Scheduling and Pre-SFT Alignment 

**Title (ZH)**: TEMPO：通过难度调度和预SFT对齐优化视频LLMs的时间偏好优化 

**Authors**: Shicheng Li, Lei Li, Kun Ouyang, Shuhuai Ren, Yuanxin Liu, Yuanxing Zhang, Fuzheng Zhang, Lingpeng Kong, Qi Liu, Xu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.16929)  

**Abstract**: Video Large Language Models (Video LLMs) have achieved significant success by leveraging a two-stage paradigm: pretraining on large-scale video-text data for vision-language alignment, followed by supervised fine-tuning (SFT) for task-specific capabilities. However, existing approaches struggle with temporal reasoning due to weak temporal correspondence in the data and reliance on the next-token prediction paradigm during training. To address these limitations, we propose TEMPO (TEMporal Preference Optimization), a systematic framework that enhances Video LLMs' temporal reasoning capabilities through Direct Preference Optimization (DPO). To facilitate this, we introduce an automated preference data generation pipeline that systematically constructs preference pairs by selecting videos that are rich in temporal information, designing video-specific perturbation strategies, and finally evaluating model responses on clean and perturbed video inputs. Our temporal alignment features two key innovations: curriculum learning which that progressively increases perturbation difficulty to improve model robustness and adaptability; and ``Pre-SFT Alignment'', applying preference optimization before instruction tuning to prioritize fine-grained temporal comprehension. Extensive experiments demonstrate that our approach consistently improves Video LLM performance across multiple benchmarks with a relatively small set of self-generated DPO data. We further analyze the transferability of DPO data across architectures and the role of difficulty scheduling in optimization. Our findings highlight our TEMPO as a scalable and efficient complement to SFT-based methods, paving the way for developing reliable Video LLMs. 

**Abstract (ZH)**: 视频大型语言模型（Video LLMs）通过利用两阶段范式取得了显著成功：在大规模视频-文本数据上进行预训练以实现视觉-语言对齐，随后进行监督微调（SFT）以获得特定任务的能力。然而，现有方法在处理时间推理方面存在困难，因为数据中的时间对应关系较弱，并且在训练过程中依赖于下一个令牌预测范式。为了解决这些局限性，我们提出了一种系统框架TEMPO（TEMporal Preference Optimization），通过直接偏好优化（DPO）增强视频大型语言模型的时间推理能力。为了促进这一点，我们引入了一种自动偏好数据生成管道，该管道系统地构建偏好对，通过选择富含时间信息的视频、设计视频特定的扰动策略，并最终评估模型对干净和扰动视频输入的响应。我们的时间对齐包括两个关键创新：循序渐进的学习方法，逐步提高扰动难度以提高模型的稳健性和适应性；以及“预SFT对齐”，在指令微调之前应用偏好优化以优先考虑细粒度的时间理解。广泛的经验表明，通过相对较小的自动生成DPO数据集，我们的方法在多个基准上持续提高了视频大型语言模型的性能。我们进一步分析了DPO数据在不同架构中的可迁移性以及难度调度在优化中的作用。我们的研究结果突显了TEMPO作为SFT方法的可扩展和高效补充的重要性，为开发可靠的视频大型语言模型铺平了道路。 

---
# RustEvo^2: An Evolving Benchmark for API Evolution in LLM-based Rust Code Generation 

**Title (ZH)**: RustEvo^2: 一种基于API演化的LLM驱动的Rust代码生成演变基准测试 

**Authors**: Linxi Liang, Jing Gong, Mingwei Liu, Chong Wang, Guangsheng Ou, Yanlin Wang, Xin Peng, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16922)  

**Abstract**: Large Language Models (LLMs) have become pivotal tools for automating code generation in software development. However, these models face significant challenges in producing version-aware code for rapidly evolving languages like Rust, where frequent Application Programming Interfaces (API) changes across versions lead to compatibility issues and correctness errors. Existing benchmarks lack systematic evaluation of how models navigate API transitions, relying on labor-intensive manual curation and offering limited version-specific insights. To address this gap, we present RustEvo, a novel framework for constructing dynamic benchmarks that evaluate the ability of LLMs to adapt to evolving Rust APIs. RustEvo automates dataset creation by synthesizing 588 API changes (380 from Rust standard libraries, 208 from 15 third-party crates) into programming tasks mirroring real-world challenges. These tasks cover four API evolution categories: Stabilizations, Signature Changes, Behavioral Changes, and Deprecations, reflecting their actual distribution in the Rust ecosystem.
Experiments on state-of-the-art (SOTA) LLMs reveal significant performance variations: models achieve a 65.8% average success rate on stabilized APIs but only 38.0% on behavioral changes, highlighting difficulties in detecting semantic shifts without signature alterations. Knowledge cutoff dates strongly influence performance, with models scoring 56.1% on before-cutoff APIs versus 32.5% on after-cutoff tasks. Retrieval-Augmented Generation (RAG) mitigates this gap, improving success rates by 13.5% on average for APIs released after model training. Our findings underscore the necessity of our evolution-aware benchmarks to advance the adaptability of LLMs in fast-paced software ecosystems. The framework and the benchmarks are publicly released at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已成为软件开发中自动化代码生成的关键工具。然而，这些模型在生成版本感知代码时面临巨大挑战，特别是在像Rust这样的快速演进语言中，频繁的应用程序编程接口（API）变化导致兼容性问题和正确性错误。现有基准测试缺乏系统性评估模型如何导航API过渡，依赖于耗时的手动整理，并提供有限的版本特定洞察。为弥补这一差距，我们提出了RustEvo，一种新的框架，用于构建动态基准测试以评估LLMs适应Rust API演进的能力。RustEvo通过合成588个API变化（包括380个来自Rust标准库和208个来自15个第三方crate），创建模拟现实挑战的编程任务。这些任务涵盖了四个API演进类别：稳定化、签名变化、行为变化和弃用，反映了Rust生态系统中这些类别的实际分布。 

---
# When Preferences Diverge: Aligning Diffusion Models with Minority-Aware Adaptive DPO 

**Title (ZH)**: 当偏好不一致时：面向少数群体意识的自适应DPO对齐扩散模型 

**Authors**: Lingfan Zhang, Chen Liu, Chengming Xu, Kai Hu, Donghao Luo, Chengjie Wang, Yanwei Fu, Yuan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16921)  

**Abstract**: In recent years, the field of image generation has witnessed significant advancements, particularly in fine-tuning methods that align models with universal human preferences. This paper explores the critical role of preference data in the training process of diffusion models, particularly in the context of Diffusion-DPO and its subsequent adaptations. We investigate the complexities surrounding universal human preferences in image generation, highlighting the subjective nature of these preferences and the challenges posed by minority samples in preference datasets. Through pilot experiments, we demonstrate the existence of minority samples and their detrimental effects on model performance. We propose Adaptive-DPO -- a novel approach that incorporates a minority-instance-aware metric into the DPO objective. This metric, which includes intra-annotator confidence and inter-annotator stability, distinguishes between majority and minority samples. We introduce an Adaptive-DPO loss function which improves the DPO loss in two ways: enhancing the model's learning of majority labels while mitigating the negative impact of minority samples. Our experiments demonstrate that this method effectively handles both synthetic minority data and real-world preference data, paving the way for more effective training methodologies in image generation tasks. 

**Abstract (ZH)**: 近年来，图像生成领域取得了显著进展，尤其是在使模型与普遍人类偏好相一致的微调方法方面。本文探讨了偏好数据在扩散模型训练过程中的关键作用，特别是在Diffusion-DPO及其后续改进方法的上下文中。我们研究了图像生成中普遍人类偏好的复杂性，突显了这些偏好的主观性质以及偏好数据集中少数样本带来的挑战。通过初步实验，我们证明了少数样本的存在及其对模型性能的负面影响。我们提出了Adaptive-DPO——一种新颖的方法，该方法将少数实例感知度量纳入DPO目标中。该度量包括注释者内部信心和注释者之间的一致性，以区分多数样本和少数样本。我们引入了一种Adaptive-DPO损失函数，该函数通过增强模型对多数标签的学习并减轻少数样本的负面影响，来改进DPO损失。我们的实验表明，该方法能够有效地处理合成的少数数据和真实世界的偏好数据，为图像生成任务的有效培训方法铺平了道路。 

---
# Deep Learning for Human Locomotion Analysis in Lower-Limb Exoskeletons: A Comparative Study 

**Title (ZH)**: 深度学习在下肢外骨骼中的人体运动分析：一项比较研究 

**Authors**: Omar Coser, Christian Tamantini, Matteo Tortora, Leonardo Furia, Rosa Sicilia, Loredana Zollo, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2503.16904)  

**Abstract**: Wearable robotics for lower-limb assistance have become a pivotal area of research, aiming to enhance mobility for individuals with physical impairments or augment the performance of able-bodied users. Accurate and adaptive control systems are essential to ensure seamless interaction between the wearer and the robotic device, particularly when navigating diverse and dynamic terrains. Despite the recent advances in neural networks for time series analysis, no attempts have been directed towards the classification of ground conditions, categorized into five classes and subsequently determining the ramp's slope and stair's height. In this respect, this paper presents an experimental comparison between eight deep neural network backbones to predict high-level locomotion parameters across diverse terrains.
All the models are trained on the publicly available CAMARGO 2021 dataset. IMU-only data equally or outperformed IMU+EMG inputs, promoting a cost-effective and efficient design. Indeeds, using three IMU sensors, the LSTM achieved high terrain classification accuracy (0.94 +- 0.04) and precise ramp slope (1.95 +- 0.58°) and the CNN-LSTM a stair height (15.65 +- 7.40 mm) estimations. As a further contribution, SHAP analysis justified sensor reduction without performance loss, ensuring a lightweight setup. The system operates with ~2 ms inference time, supporting real-time applications. The code is code available at this https URL. 

**Abstract (ZH)**: 穿戴式机器人在下肢辅助领域的研究已成为关键研究方向，旨在提升身体残疾个体的移动性或增强健全个体的表现。准确且适应性强的控制系统对于确保穿戴者与机器人设备之间无缝交互至关重要，尤其是在穿越多样且动态地形时。尽管在时间序列分析的神经网络方面取得了最近的进步，但仍没有尝试对地面条件进行分类，将其分为五类，并据此确定斜坡的坡度和阶梯的高度。就此而言，本文对比了八种深度神经网络骨干模型在不同地形下预测高级步行参数的实验效果。所有模型均在公开可用的CAMARGO 2021数据集上进行训练。仅使用IMU数据或IMU+EMG输入方式可实现同等甚至更优的效果，促进成本效益高且高效的系统设计。确实，使用三个IMU传感器时，LSTM 在地形分类准确性上达到0.94 ± 0.04，并精确估计斜坡坡度为1.95 ± 0.58°，而CNN-LSTM 估计阶梯高度为15.65 ± 7.40 mm。另外，SHAP 分析证明了在不损失性能的情况下减少传感器数量的有效性，确保轻量化配置。该系统具有约2 ms的推理时间，支持实时应用。代码可通过此链接访问。 

---
# Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification 

**Title (ZH)**: 基于分类器引导的CLIP知识蒸馏用于无监督多标签分类 

**Authors**: Dongseob Kim, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2503.16873)  

**Abstract**: Multi-label classification is crucial for comprehensive image understanding, yet acquiring accurate annotations is challenging and costly. To address this, a recent study suggests exploiting unsupervised multi-label classification leveraging CLIP, a powerful vision-language model. Despite CLIP's proficiency, it suffers from view-dependent predictions and inherent bias, limiting its effectiveness. We propose a novel method that addresses these issues by leveraging multiple views near target objects, guided by Class Activation Mapping (CAM) of the classifier, and debiasing pseudo-labels derived from CLIP predictions. Our Classifier-guided CLIP Distillation (CCD) enables selecting multiple local views without extra labels and debiasing predictions to enhance classification performance. Experimental results validate our method's superiority over existing techniques across diverse datasets. The code is available at this https URL. 

**Abstract (ZH)**: 利用CLIP进行去偏见的多标签分类：基于分类器引导的CLIP蒸馏（Classifier-guided CLIP Distillation） 

---
# Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs 

**Title (ZH)**: 稀疏逻辑采样：在大规模语言模型中加速知识蒸馏 

**Authors**: Anshumann, Mohd Abbas Zaidi, Akhil Kedia, Jinwoo Ahn, Taehwak Kwon, Kangwook Lee, Haejun Lee, Joohyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.16870)  

**Abstract**: Knowledge distillation can be a cost-effective technique to distill knowledge in Large Language Models, if the teacher output logits can be pre-computed and cached. However, successfully applying this to pre-training remains largely unexplored. In this work, we prove that naive approaches for sparse knowledge distillation such as caching Top-K probabilities, while intuitive, provide biased estimates of teacher probability distribution to the student, resulting in suboptimal performance and calibration. We propose an importance-sampling-based method `Random Sampling Knowledge Distillation', which provides unbiased estimates, preserves the gradient in expectation, and requires storing significantly sparser logits. Our method enables faster training of student models with marginal overhead (<10%) compared to cross-entropy based training, while maintaining competitive performance compared to full distillation, across a range of model sizes from 300M to 3B. 

**Abstract (ZH)**: 基于重要性采样的随机采样知识蒸馏在大规模语言模型中提供无偏估计并保持梯度，在不同模型大小（从300M到3B）下实现较快的训练速度同时保持竞争力性能。 

---
# MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering 

**Title (ZH)**: MTBench: 多模态时间序列基准数据集用于时间推理和问答 

**Authors**: Jialin Chen, Aosong Feng, Ziyu Zhao, Juan Garza, Gaukhar Nurbek, Cheng Qin, Ali Maatouk, Leandros Tassiulas, Yifeng Gao, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2503.16858)  

**Abstract**: Understanding the relationship between textual news and time-series evolution is a critical yet under-explored challenge in applied data science. While multimodal learning has gained traction, existing multimodal time-series datasets fall short in evaluating cross-modal reasoning and complex question answering, which are essential for capturing complex interactions between narrative information and temporal patterns. To bridge this gap, we introduce Multimodal Time Series Benchmark (MTBench), a large-scale benchmark designed to evaluate large language models (LLMs) on time series and text understanding across financial and weather domains. MTbench comprises paired time series and textual data, including financial news with corresponding stock price movements and weather reports aligned with historical temperature records. Unlike existing benchmarks that focus on isolated modalities, MTbench provides a comprehensive testbed for models to jointly reason over structured numerical trends and unstructured textual narratives. The richness of MTbench enables formulation of diverse tasks that require a deep understanding of both text and time-series data, including time-series forecasting, semantic and technical trend analysis, and news-driven question answering (QA). These tasks target the model's ability to capture temporal dependencies, extract key insights from textual context, and integrate cross-modal information. We evaluate state-of-the-art LLMs on MTbench, analyzing their effectiveness in modeling the complex relationships between news narratives and temporal patterns. Our findings reveal significant challenges in current models, including difficulties in capturing long-term dependencies, interpreting causality in financial and weather trends, and effectively fusing multimodal information. 

**Abstract (ZH)**: 理解文本新闻与时间序列演化的关系是应用数据科学中一个关键但尚未充分探索的挑战。尽管多模态学习正逐渐成为热点，但现有的多模态时间序列数据集在评估跨模态推理和复杂问题回答方面仍存在不足，这是捕捉叙事信息与时间模式之间复杂互动的关键。为此，我们介绍了多模态时间序列基准（MTBench），这是一个大规模基准，旨在评估大规模语言模型（LLMs）在金融和天气领域的时间序列和文本理解能力。MTBench 包含成对的时间序列和文本数据，包括与股票价格变动对应的金融新闻和与历史温度记录对齐的天气报告。不同于现有主要关注孤立模态的基准，MTBench 提供了一个全面的测试环境，让模型能够同时推理结构化数值趋势和非结构化文本叙事。MTBench 的丰富性使其能够定义多种需要深入理解文本和时间序列数据的任务，包括时间序列预测、语义和技术趋势分析、以及由新闻驱动的问题回答（QA）。这些任务旨在评估模型捕捉时间依赖性、从文本上下文中提取关键见解并整合跨模态信息的能力。我们在 MTBench 上评估了最先进的 LLMs，分析了它们在建模新闻叙事与时间模式之间复杂关系方面的有效性。我们的研究发现了当前模型的重要挑战，包括难以捕捉长期依赖性、解释金融和天气趋势中的因果关系，以及有效融合多模态信息。 

---
# Imagine to Hear: Auditory Knowledge Generation can be an Effective Assistant for Language Models 

**Title (ZH)**: 想象听见：听觉知识生成可以成为语言模型的有效辅助工具 

**Authors**: Suho Yoo, Hyunjong Ok, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.16853)  

**Abstract**: Language models pretrained on text-only corpora often struggle with tasks that require auditory commonsense knowledge. Previous work addresses this problem by augmenting the language model to retrieve knowledge from external audio databases. This approach has several limitations, such as the potential lack of relevant audio in databases and the high costs associated with constructing and querying the databases. To address these issues, we propose Imagine to Hear, a novel approach that dynamically generates auditory knowledge using generative models. Our framework detects multiple audio-related textual spans from the given prompt and generates corresponding auditory knowledge. We develop several mechanisms to efficiently process multiple auditory knowledge, including a CLAP-based rejection sampler and a language-audio fusion module. Our experiments show that our method achieves state-of-the-art performance on AuditoryBench without relying on external databases, highlighting the effectiveness of our generation-based approach. 

**Abstract (ZH)**: 语言模型在仅文本数据预训练的情况下往往难以完成需要听觉常识知识的任务。为解决这一问题，已有工作通过扩展语言模型以从外部音频数据库检索知识予以应对，但这种方法存在一些局限性，如数据库中缺乏相关音频以及构建和查询数据库所导致的高昂成本。为应对这些问题，我们提出了一种名为Imagine to Hear的新型方法，该方法利用生成模型动态生成听觉知识。我们的框架能够从给定提示中检测出多个与音频相关的文本片段，并生成相应的听觉知识。我们开发了几种机制来高效处理多样的听觉知识，包括基于CLAP的拒绝采样器和语言-音频融合模块。实验结果表明，我们的方法在不依赖外部数据库的情况下，在AuditoryBench上实现了最先进的性能，突显了基于生成的方法的有效性。 

---
# Casual Inference via Style Bias Deconfounding for Domain Generalization 

**Title (ZH)**: 基于风格偏差去混淆的因果推断在领域泛化的应用 

**Authors**: Jiaxi Li, Di Lin, Hao Chen, Hongying Liu, Liang Wan, Wei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16852)  

**Abstract**: Deep neural networks (DNNs) often struggle with out-of-distribution data, limiting their reliability in diverse realworld applications. To address this issue, domain generalization methods have been developed to learn domain-invariant features from single or multiple training domains, enabling generalization to unseen testing domains. However, existing approaches usually overlook the impact of style frequency within the training set. This oversight predisposes models to capture spurious visual correlations caused by style confounding factors, rather than learning truly causal representations, thereby undermining inference reliability. In this work, we introduce Style Deconfounding Causal Learning (SDCL), a novel causal inference-based framework designed to explicitly address style as a confounding factor. Our approaches begins with constructing a structural causal model (SCM) tailored to the domain generalization problem and applies a backdoor adjustment strategy to account for style influence. Building on this foundation, we design a style-guided expert module (SGEM) to adaptively clusters style distributions during training, capturing the global confounding style. Additionally, a back-door causal learning module (BDCL) performs causal interventions during feature extraction, ensuring fair integration of global confounding styles into sample predictions, effectively reducing style bias. The SDCL framework is highly versatile and can be seamlessly integrated with state-of-the-art data augmentation techniques. Extensive experiments across diverse natural and medical image recognition tasks validate its efficacy, demonstrating superior performance in both multi-domain and the more challenging single-domain generalization scenarios. 

**Abstract (ZH)**: Style Deconfounding Causal Learning：基于因果推断的风格去混淆学习 

---
# Physics-Informed Neural Network Surrogate Models for River Stage Prediction 

**Title (ZH)**: 物理知情神经网络 surrogate 模型在河流水位预测中的应用 

**Authors**: Maximilian Zoch, Edward Holmberg, Pujan Pokhrel, Ken Pathak, Steven Sloan, Kendall Niles, Jay Ratcliff, Maik Flanagin, Elias Ioup, Christian Guetl, Mahdi Abdelguerfi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16850)  

**Abstract**: This work investigates the feasibility of using Physics-Informed Neural Networks (PINNs) as surrogate models for river stage prediction, aiming to reduce computational cost while maintaining predictive accuracy. Our primary contribution demonstrates that PINNs can successfully approximate HEC-RAS numerical solutions when trained on a single river, achieving strong predictive accuracy with generally low relative errors, though some river segments exhibit higher deviations.
By integrating the governing Saint-Venant equations into the learning process, the proposed PINN-based surrogate model enforces physical consistency and significantly improves computational efficiency compared to HEC-RAS. We evaluate the model's performance in terms of accuracy and computational speed, demonstrating that it closely approximates HEC-RAS predictions while enabling real-time inference.
These results highlight the potential of PINNs as effective surrogate models for single-river hydrodynamics, offering a promising alternative for computationally efficient river stage forecasting. Future work will explore techniques to enhance PINN training stability and robustness across a more generalized multi-river model. 

**Abstract (ZH)**: 本文研究了将物理学告知神经网络（PINNs）作为河流水位预测的代理模型的可行性，旨在减少计算成本同时保持预测准确性。主要贡献在于证明了在单一河流上训练的PINNs能够成功逼近HEC-RAS数值解，总体相对误差较低，尽管某些河段表现出较高偏差。

通过将圣维南方程集成到学习过程中，所提出的基于PINN的代理模型确保了物理一致性，并大大提高了计算效率，相比于HEC-RAS。我们从准确性和计算速度两方面评估了该模型的性能，结果表明它可以逼近HEC-RAS的预测结果并支持实时推断。

这些结果突显了PINNs作为单一河流水动力学有效代理模型的潜力，为计算高效的河流水位预报提供了有希望的替代方案。未来工作将探索增强PINN训练稳定性和健壯性的技术，以适用于更通用的多河模型。 

---
# The Deployment of End-to-End Audio Language Models Should Take into Account the Principle of Least Privilege 

**Title (ZH)**: 端到端音频语言模型的部署应考虑最小权限原则 

**Authors**: Luxi He, Xiangyu Qi, Michel Liao, Inyoung Cheong, Prateek Mittal, Danqi Chen, Peter Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2503.16833)  

**Abstract**: We are at a turning point for language models that accept audio input. The latest end-to-end audio language models (Audio LMs) process speech directly instead of relying on a separate transcription step. This shift preserves detailed information, such as intonation or the presence of multiple speakers, that would otherwise be lost in transcription. However, it also introduces new safety risks, including the potential misuse of speaker identity cues and other sensitive vocal attributes, which could have legal implications. In this position paper, we urge a closer examination of how these models are built and deployed. We argue that the principle of least privilege should guide decisions on whether to deploy cascaded or end-to-end models. Specifically, evaluations should assess (1) whether end-to-end modeling is necessary for a given application; and (2), the appropriate scope of information access. Finally, We highlight related gaps in current audio LM benchmarks and identify key open research questions, both technical and policy-related, that must be addressed to enable the responsible deployment of end-to-end Audio LMs. 

**Abstract (ZH)**: 语言模型接受音频输入正处在一个转折点。最新的端到端音频语言模型（Audio LMs）直接处理语音而非依赖于单独的转录步骤。这一转变保留了诸如语调或多说话者存在的详细信息，这些信息在转录过程中会丢失。然而，这也引入了新的安全风险，包括可能误用说话者身份线索和其他敏感的嗓音属性，这可能会带来法律上的影响。在本文中，我们呼吁更仔细地审视这些模型的构建和部署方式。我们认为最小权限原则应指导是否部署级联或端到端模型的决策。具体而言，评估应考虑（1）给定应用是否需要端到端建模；（2）适当的信息访问范围。最后，我们指出现有音频LM基准中的相关缺口，并识别必须解决的关键开放研究问题，包括技术和政策相关问题，以确保端到端音频LMs的负责任部署。 

---
# DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation 

**Title (ZH)**: DyWA：动态自适应世界动作模型以实现泛化非拾取操作 

**Authors**: Jiangran Lyu, Ziming Li, Xuesong Shi, Chaoyi Xu, Yizhou Wang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16806)  

**Abstract**: Nonprehensile manipulation is crucial for handling objects that are too thin, large, or otherwise ungraspable in unstructured environments. While conventional planning-based approaches struggle with complex contact modeling, learning-based methods have recently emerged as a promising alternative. However, existing learning-based approaches face two major limitations: they heavily rely on multi-view cameras and precise pose tracking, and they fail to generalize across varying physical conditions, such as changes in object mass and table friction. To address these challenges, we propose the Dynamics-Adaptive World Action Model (DyWA), a novel framework that enhances action learning by jointly predicting future states while adapting to dynamics variations based on historical trajectories. By unifying the modeling of geometry, state, physics, and robot actions, DyWA enables more robust policy learning under partial observability. Compared to baselines, our method improves the success rate by 31.5% using only single-view point cloud observations in the simulation. Furthermore, DyWA achieves an average success rate of 68% in real-world experiments, demonstrating its ability to generalize across diverse object geometries, adapt to varying table friction, and robustness in challenging scenarios such as half-filled water bottles and slippery surfaces. 

**Abstract (ZH)**: 非抓取 manipulation 对于处理细长、过大或在非结构化环境中无法抓住的对象至关重要。虽然基于规划的传统方法在复杂的接触建模上遇到困难，但基于学习的方法最近 emerged 作为一种有前途的替代方案。然而，现有的基于学习的方法面临两大主要限制：它们严重依赖多视图相机和精确的姿态跟踪，并且无法在不同的物理条件下进行泛化，例如物体质量的变化和桌面摩擦的变化。为了解决这些挑战，我们提出了动态自适应世界动作模型（DyWA），这是一种新的框架，通过联合预测未来状态并在根据历史轨迹适应动力学变化的基础上增强动作学习。通过统一建模几何、状态、物理和机器人动作，DyWA 在部分可观测条件下实现了更稳健的策略学习。与基线方法相比，仅使用单视角点云观测，我们的方法在模拟实验中将成功率提高了 31.5%。此外，在真实世界实验中，DyWA 达到了 68% 的平均成功率，展示了其在不同几何形状的物体、适应不同的桌面摩擦以及在挑战性场景（如半满的水瓶和滑动表面）中的泛化能力和鲁棒性。 

---
# Auto-Regressive Diffusion for Generating 3D Human-Object Interactions 

**Title (ZH)**: 自回归扩散生成3D人体物交互 

**Authors**: Zichen Geng, Zeeshan Hayder, Wei Liu, Ajmal Saeed Mian  

**Link**: [PDF](https://arxiv.org/pdf/2503.16801)  

**Abstract**: Text-driven Human-Object Interaction (Text-to-HOI) generation is an emerging field with applications in animation, video games, virtual reality, and robotics. A key challenge in HOI generation is maintaining interaction consistency in long sequences. Existing Text-to-Motion-based approaches, such as discrete motion tokenization, cannot be directly applied to HOI generation due to limited data in this domain and the complexity of the modality. To address the problem of interaction consistency in long sequences, we propose an autoregressive diffusion model (ARDHOI) that predicts the next continuous token. Specifically, we introduce a Contrastive Variational Autoencoder (cVAE) to learn a physically plausible space of continuous HOI tokens, thereby ensuring that generated human-object motions are realistic and natural. For generating sequences autoregressively, we develop a Mamba-based context encoder to capture and maintain consistent sequential actions. Additionally, we implement an MLP-based denoiser to generate the subsequent token conditioned on the encoded context. Our model has been evaluated on the OMOMO and BEHAVE datasets, where it outperforms existing state-of-the-art methods in terms of both performance and inference speed. This makes ARDHOI a robust and efficient solution for text-driven HOI tasks 

**Abstract (ZH)**: 基于文本的人机物交互（Text-to-HOI）生成是动画、电子游戏、虚拟现实和机器人领域的新兴领域。长序列中交互一致性保持是HOI生成的关键挑战。现有的基于文本到运动的方法，如离散运动token化，由于该领域数据有限且模态复杂性高，无法直接应用于HOI生成。为了解决长序列中交互一致性的问题，我们提出了一种自回归扩散模型（ARDHOI），用于预测下一个连续token。具体地，我们引入了一种对比变分自编码器（cVAE）以学习物理上合理的连续HOI token空间，从而保证生成的人机物运动具有现实感和自然性。为了自回归地生成序列，我们开发了一种基于Mamba的上下文编码器，以捕获和保持一致的序列动作。此外，我们实现了一种基于MLP的去噪器，以在编码的上下文条件下生成后续token。我们在OMOMO和BEHAVE数据集上对模型进行了评估，结果显示在性能和推理速度上均优于现有最先进的方法。这使得ARDHOI成为文本驱动HOI任务的 robust 和高效解决方案。 

---
# Causally Aligned Curriculum Learning 

**Title (ZH)**: 因果对齐课程学习 

**Authors**: Mingxuan Li, Junzhe Zhang, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2503.16799)  

**Abstract**: A pervasive challenge in Reinforcement Learning (RL) is the "curse of dimensionality" which is the exponential growth in the state-action space when optimizing a high-dimensional target task. The framework of curriculum learning trains the agent in a curriculum composed of a sequence of related and more manageable source tasks. The expectation is that when some optimal decision rules are shared across source tasks and the target task, the agent could more quickly pick up the necessary skills to behave optimally in the environment, thus accelerating the learning process. However, this critical assumption of invariant optimal decision rules does not necessarily hold in many practical applications, specifically when the underlying environment contains unobserved confounders. This paper studies the problem of curriculum RL through causal lenses. We derive a sufficient graphical condition characterizing causally aligned source tasks, i.e., the invariance of optimal decision rules holds. We further develop an efficient algorithm to generate a causally aligned curriculum, provided with qualitative causal knowledge of the target task. Finally, we validate our proposed methodology through experiments in discrete and continuous confounded tasks with pixel observations. 

**Abstract (ZH)**: 强化学习（RL）中的一个普遍挑战是“维度灾难”，即在优化高维目标任务时状态-动作空间的指数级增长。课程学习框架通过一系列相关且更易于管理的源任务来训练代理。期望通过一些最优决策规则在源任务和目标任务中保持一致，代理能够更快地掌握必要的技能以在环境中表现最优，从而加速学习过程。然而，在许多实际应用中，这一关键假设——最优决策规则的不变性——并不一定成立，特别是在潜在环境中存在未观察到的混杂因素时。本文从因果视角研究课程强化学习问题。我们推导出一个充分的图形条件，刻画因果对齐的源任务，即最优决策规则的不变性成立。进一步地，我们开发了一个高效算法，在提供目标任务的定性因果知识的前提下生成因果对齐的课程。最后，我们通过在具有像素观察的离散和连续混杂任务中的实验验证了所提出的方法论。 

---
# "The Diagram is like Guardrails": Structuring GenAI-assisted Hypotheses Exploration with an Interactive Shared Representation 

**Title (ZH)**: “图表如同护栏”：通过交互式共享表示结构化AI辅助假设探索 

**Authors**: Zijian Ding, Michelle Brachman, Joel Chan, Werner Geyer  

**Link**: [PDF](https://arxiv.org/pdf/2503.16791)  

**Abstract**: Data analysis encompasses a spectrum of tasks, from high-level conceptual reasoning to lower-level execution. While AI-powered tools increasingly support execution tasks, there remains a need for intelligent assistance in conceptual tasks. This paper investigates the design of an ordered node-link tree interface augmented with AI-generated information hints and visualizations, as a potential shared representation for hypothesis exploration. Through a design probe (n=22), participants generated diagrams averaging 21.82 hypotheses. Our findings showed that the node-link diagram acts as "guardrails" for hypothesis exploration, facilitating structured workflows, providing comprehensive overviews, and enabling efficient backtracking. The AI-generated information hints, particularly visualizations, aided users in transforming abstract ideas into data-backed concepts while reducing cognitive load. We further discuss how node-link diagrams can support both parallel exploration and iterative refinement in hypothesis formulation, potentially enhancing the breadth and depth of human-AI collaborative data analysis. 

**Abstract (ZH)**: 数据分析涵盖了从高层次的概念推理到低层次的执行任务的谱系。尽管AI辅助工具越来越多地支持执行任务，但在概念任务上仍需智能辅助。本文探讨了一种有序节点链接树界面的设计，该界面结合了AI生成的信息提示和可视化，作为假设探索的潜在共享表示形式。通过一项设计探查（n=22），参与者生成了平均21.82个假设图。我们的研究发现表明，节点链接图充当了“赛道”，促进了结构化的工作流程，提供了全面的概述，并使回溯变得高效。AI生成的信息提示，尤其是可视化，帮助用户将抽象的想法转化为数据支持的概念，从而减少认知负担。我们进一步讨论了节点链接图如何支持假设制定过程中的并行探索和迭代细化，从而可能增强人类-AI协作数据分析的广度和深度。 

---
# Learning Part Knowledge to Facilitate Category Understanding for Fine-Grained Generalized Category Discovery 

**Title (ZH)**: 学习部分知识以促进细粒度泛化类别的理解与发现 

**Authors**: Enguang Wang, Zhimao Peng, Zhengyuan Xie, Haori Lu, Fei Yang, Xialei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16782)  

**Abstract**: Generalized Category Discovery (GCD) aims to classify unlabeled data containing both seen and novel categories. Although existing methods perform well on generic datasets, they struggle in fine-grained scenarios. We attribute this difficulty to their reliance on contrastive learning over global image features to automatically capture discriminative cues, which fails to capture the subtle local differences essential for distinguishing fine-grained categories. Therefore, in this paper, we propose incorporating part knowledge to address fine-grained GCD, which introduces two key challenges: the absence of annotations for novel classes complicates the extraction of the part features, and global contrastive learning prioritizes holistic feature invariance, inadvertently suppressing discriminative local part patterns. To address these challenges, we propose PartGCD, including 1) Adaptive Part Decomposition, which automatically extracts class-specific semantic parts via Gaussian Mixture Models, and 2) Part Discrepancy Regularization, enforcing explicit separation between part features to amplify fine-grained local part distinctions.
Experiments demonstrate state-of-the-art performance across multiple fine-grained benchmarks while maintaining competitiveness on generic datasets, validating the effectiveness and robustness of our approach. 

**Abstract (ZH)**: 广义类别发现（GCD）旨在对包含已见类别和新颖类别的未标记数据进行分类。尽管现有方法在通用数据集中表现良好，但在细粒度场景中却面临挑战。我们将这一困难归因于它们依赖于全局图像特征的对比学习自动捕捉区分性线索的方法，这种方法未能捕捉到区分细粒度类别所必需的微妙局部差异。因此，在本文中，我们提出将部分知识融入细粒度GCD中，这引入了两个关键挑战：新颖类别的标注缺失使得部分特征的提取变得复杂，而全局对比学习 prioritizes 整体特征不变性，无意中抑制了区分性局部部分模式。为了解决这些挑战，我们提出了PartGCD，包括1）自适应部分分解，通过高斯混合模型自动提取类特定的语义部分，以及2）部分差异正则化，强制部分特征之间的明确分离以放大细粒度局部部分的区别。实验结果在多个细粒度基准上展示了最先进的性能，同时在通用数据集上保持竞争力，验证了我们方法的有效性和鲁棒性。 

---
# Chain-of-Tools: Utilizing Massive Unseen Tools in the CoT Reasoning of Frozen Language Models 

**Title (ZH)**: 工具链：在冻结语言模型的CoT推理中利用大量未见过的工具 

**Authors**: Mengsong Wu, Tong Zhu, Han Han, Xiang Zhang, Wenbiao Shao, Wenliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.16779)  

**Abstract**: Tool learning can further broaden the usage scenarios of large language models (LLMs). However most of the existing methods either need to finetune that the model can only use tools seen in the training data, or add tool demonstrations into the prompt with lower efficiency. In this paper, we present a new Tool Learning method Chain-of-Tools. It makes full use of the powerful semantic representation capability of frozen LLMs to finish tool calling in CoT reasoning with a huge and flexible tool pool which may contain unseen tools. Especially, to validate the effectiveness of our approach in the massive unseen tool scenario, we construct a new dataset SimpleToolQuestions. We conduct experiments on two numerical reasoning benchmarks (GSM8K-XL and FuncQA) and two knowledge-based question answering benchmarks (KAMEL and SimpleToolQuestions). Experimental results show that our approach performs better than the baseline. We also identify dimensions of the model output that are critical in tool selection, enhancing the model interpretability. Our code and data are available at: this https URL . 

**Abstract (ZH)**: Tool学习可以进一步拓宽大规模语言模型的使用场景。然而，现有方法要么需要微调模型只能使用训练数据中 seen 的工具，要么在提示中添加工具演示以降低效率。在本文中，我们提出了一种新的Tool学习方法：Chain-of-Tools。该方法充分利用了冻结的大规模语言模型的强大语义表示能力，在包含未seen工具的庞大且灵活的工具池中进行CoT推理以完成工具调用。特别是，为了验证我们的方法在大量未seen工具场景下的有效性，我们构建了一个新的数据集SimpleToolQuestions。我们在两个数值推理基准（GSM8K-XL和FuncQA）和两个基于知识的问答基准（KAMEL和SimpleToolQuestions）上进行了实验。实验结果表明，我们的方法优于baseline。我们还识别了模型输出中对于工具选择至关重要的维度，提高了模型的可解释性。我们的代码和数据可在以下链接获取：this https URL。 

---
# Dynamic Attention Mechanism in Spatiotemporal Memory Networks for Object Tracking 

**Title (ZH)**: 时空记忆网络中动态注意力机制的研究 

**Authors**: Meng Zhou, Jiadong Xie, Mingsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16768)  

**Abstract**: Mainstream visual object tracking frameworks predominantly rely on template matching paradigms. Their performance heavily depends on the quality of template features, which becomes increasingly challenging to maintain in complex scenarios involving target deformation, occlusion, and background clutter. While existing spatiotemporal memory-based trackers emphasize memory capacity expansion, they lack effective mechanisms for dynamic feature selection and adaptive fusion. To address this gap, we propose a Dynamic Attention Mechanism in Spatiotemporal Memory Network (DASTM) with two key innovations: 1) A differentiable dynamic attention mechanism that adaptively adjusts channel-spatial attention weights by analyzing spatiotemporal correlations between the templates and memory features; 2) A lightweight gating network that autonomously allocates computational resources based on target motion states, prioritizing high-discriminability features in challenging scenarios. Extensive evaluations on OTB-2015, VOT 2018, LaSOT, and GOT-10K benchmarks demonstrate our DASTM's superiority, achieving state-of-the-art performance in success rate, robustness, and real-time efficiency, thereby offering a novel solution for real-time tracking in complex environments. 

**Abstract (ZH)**: 主流的视觉对象跟踪框架主要依赖模板匹配 paradigm。它们的性能高度依赖于模板特征的质量，在涉及目标变形、遮挡和背景杂乱的复杂场景中，这一依赖关系变得越来越具挑战性。虽然现有的基于时空记忆的跟踪器侧重于扩大记忆容量，但缺乏有效的动态特征选择和自适应融合机制。为解决这一问题，我们提出了一种时空记忆网络中的动态注意力机制 (DASTM)，其中包括两项关键创新：1) 一个可微分的动态注意力机制，通过分析模板与记忆特征之间的时空相关性，自适应调整通道-空间注意力权重；2) 一个轻量级门控网络，能够根据目标运动状态自主分配计算资源，在挑战性场景中优先选择高可区分特征。在 OTB-2015、VOT 2018、LaSOT 和 GOT-10K 基准上的广泛评估表明，我们的 DASTM 超越了现有方法，在成功率、鲁棒性和实时效率方面达到最佳性能，从而为复杂环境下的实时跟踪提供了一种新的解决方案。 

---
# QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge 

**Title (ZH)**: QuartDepth：边缘实时深度估计的后训练量化 

**Authors**: Xuan Shen, Weize Ma, Jing Liu, Changdi Yang, Rui Ding, Quanyi Wang, Henghui Ding, Wei Niu, Yanzhi Wang, Pu Zhao, Jun Lin, Jiuxiang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16709)  

**Abstract**: Monocular Depth Estimation (MDE) has emerged as a pivotal task in computer vision, supporting numerous real-world applications. However, deploying accurate depth estimation models on resource-limited edge devices, especially Application-Specific Integrated Circuits (ASICs), is challenging due to the high computational and memory demands. Recent advancements in foundational depth estimation deliver impressive results but further amplify the difficulty of deployment on ASICs. To address this, we propose QuartDepth which adopts post-training quantization to quantize MDE models with hardware accelerations for ASICs. Our approach involves quantizing both weights and activations to 4-bit precision, reducing the model size and computation cost. To mitigate the performance degradation, we introduce activation polishing and compensation algorithm applied before and after activation quantization, as well as a weight reconstruction method for minimizing errors in weight quantization. Furthermore, we design a flexible and programmable hardware accelerator by supporting kernel fusion and customized instruction programmability, enhancing throughput and efficiency. Experimental results demonstrate that our framework achieves competitive accuracy while enabling fast inference and higher energy efficiency on ASICs, bridging the gap between high-performance depth estimation and practical edge-device applicability. Code: this https URL 

**Abstract (ZH)**: 单目深度估计(MDE)已成为计算机视觉中的一个重要任务，支持众多实际应用。然而，在资源受限的边缘设备，尤其是应用特定集成电路(ASICs)上部署准确的深度估计模型颇具挑战性，因为这需要高计算能力和内存需求。近期基础深度估计的发展虽然取得了显著成果，但进一步加剧了其在ASICs上的部署难度。为解决这一问题，我们提出了QuartDepth，采用后训练量化技术结合硬件加速器为ASICs量化MDE模型。我们的方法将权重和激活量化为4位精度，减小模型大小和计算成本。为了缓解性能下降，我们引入了在激活量化前后应用的激活优化和补偿算法，以及一种权重重建方法以最小化权重量化误差。此外，我们设计了一种灵活且可编程的硬件加速器，支持内核融合和自定义指令编程，提高吞吐量和效率。实验结果表明，我们的框架在ASICs上实现了 competitive准确度，支持快速推理和更高能效，填补了高性能深度估计与实际边缘设备应用之间的差距。代码：this https URL。 

---
# Limits of trust in medical AI 

**Title (ZH)**: 医疗AI中的信任极限 

**Authors**: Joshua Hatherley  

**Link**: [PDF](https://arxiv.org/pdf/2503.16692)  

**Abstract**: Artificial intelligence (AI) is expected to revolutionize the practice of medicine. Recent advancements in the field of deep learning have demonstrated success in a variety of clinical tasks: detecting diabetic retinopathy from images, predicting hospital readmissions, aiding in the discovery of new drugs, etc. AI's progress in medicine, however, has led to concerns regarding the potential effects of this technology upon relationships of trust in clinical practice. In this paper, I will argue that there is merit to these concerns, since AI systems can be relied upon, and are capable of reliability, but cannot be trusted, and are not capable of trustworthiness. Insofar as patients are required to rely upon AI systems for their medical decision-making, there is potential for this to produce a deficit of trust in relationships in clinical practice. 

**Abstract (ZH)**: 人工智能（AI）有望革新医学实践。最近深度学习领域的进展已在多种临床任务中显示出了成功，例如从图像中检测糖尿病视网膜病变、预测再住院、辅助发现新药物等。然而，医学中人工智能的进步也引发了对于该技术可能影响临床实践中信任关系的潜在影响的担忧。本文将论证这些担忧有一定的合理性，因为虽然人工智能系统可以依赖且具有可靠性，但无法被视为可信赖的且不具备值得信任的品质。鉴于患者可能需要依赖人工智能系统进行医疗决策，这可能导致临床实践中信任关系的缺失。 

---
# GAIR: Improving Multimodal Geo-Foundation Model with Geo-Aligned Implicit Representations 

**Title (ZH)**: GAIR: 基于地理对齐隐式表示提高多模态地理基础模型性能 

**Authors**: Zeping Liu, Fan Zhang, Junfeng Jiao, Ni Lao, Gengchen Mai  

**Link**: [PDF](https://arxiv.org/pdf/2503.16683)  

**Abstract**: Advancements in vision and language foundation models have inspired the development of geo-foundation models (GeoFMs), enhancing performance across diverse geospatial tasks. However, many existing GeoFMs primarily focus on overhead remote sensing (RS) data while neglecting other data modalities such as ground-level imagery. A key challenge in multimodal GeoFM development is to explicitly model geospatial relationships across modalities, which enables generalizability across tasks, spatial scales, and temporal contexts. To address these limitations, we propose GAIR, a novel multimodal GeoFM architecture integrating overhead RS data, street view (SV) imagery, and their geolocation metadata. We utilize three factorized neural encoders to project an SV image, its geolocation, and an RS image into the embedding space. The SV image needs to be located within the RS image's spatial footprint but does not need to be at its geographic center. In order to geographically align the SV image and RS image, we propose a novel implicit neural representations (INR) module that learns a continuous RS image representation and looks up the RS embedding at the SV image's geolocation. Next, these geographically aligned SV embedding, RS embedding, and location embedding are trained with contrastive learning objectives from unlabeled data. We evaluate GAIR across 10 geospatial tasks spanning RS image-based, SV image-based, and location embedding-based benchmarks. Experimental results demonstrate that GAIR outperforms state-of-the-art GeoFMs and other strong baselines, highlighting its effectiveness in learning generalizable and transferable geospatial representations. 

**Abstract (ZH)**: 视觉和语言基础模型的进步激发了地理基础模型（GeoFMs）的发展，提升了多样化的地理空间任务性能。然而，许多现有的GeoFMs主要集中在高空遥感（RS）数据上，而忽视了地面图像等其他数据模态。多模态GeoFM发展中的一项关键挑战是明确建模跨模态的地理空间关系，这使得任务、空间尺度和时间上下文的泛化能力得以增强。为解决这些限制，我们提出了GAIR，这是一种新颖的多模态GeoFM架构，结合了高空RS数据、街景（SV）图像及其地理定位元数据。我们利用三个因子化的神经编码器将SV图像、其地理定位和RS图像投影到嵌入空间。SV图像需要位于RS图像的地理区域内，但不一定在其地理中心。为了地理对齐SV图像和RS图像，我们提出了一种新颖的隐式神经表示（INR）模块，该模块学习连续的RS图像表示，并在SV图像的地理定位处查找RS嵌入。随后，这些地理对齐的SV嵌入、RS嵌入和位置嵌入是从未标记数据中使用对比学习目标进行训练的。我们在涵盖RS图像基于、SV图像基于和位置嵌入基于的10项地理空间任务基准上评估了GAIR。实验结果表明，GAIR在多项指标上优于最先进的GeoFMs和其他强大的基线，突显了其在学习泛化和可转移地理空间表示方面的有效性。 

---
# GauRast: Enhancing GPU Triangle Rasterizers to Accelerate 3D Gaussian Splatting 

**Title (ZH)**: GauRast: 提升GPU三角光栅化以加速3D高斯点阵化 

**Authors**: Sixu Li, Ben Keller, Yingyan Celine Lin, Brucek Khailany  

**Link**: [PDF](https://arxiv.org/pdf/2503.16681)  

**Abstract**: 3D intelligence leverages rich 3D features and stands as a promising frontier in AI, with 3D rendering fundamental to many downstream applications. 3D Gaussian Splatting (3DGS), an emerging high-quality 3D rendering method, requires significant computation, making real-time execution on existing GPU-equipped edge devices infeasible. Previous efforts to accelerate 3DGS rely on dedicated accelerators that require substantial integration overhead and hardware costs. This work proposes an acceleration strategy that leverages the similarities between the 3DGS pipeline and the highly optimized conventional graphics pipeline in modern GPUs. Instead of developing a dedicated accelerator, we enhance existing GPU rasterizer hardware to efficiently support 3DGS operations. Our results demonstrate a 23$\times$ increase in processing speed and a 24$\times$ reduction in energy consumption, with improvements yielding 6$\times$ faster end-to-end runtime for the original 3DGS algorithm and 4$\times$ for the latest efficiency-improved pipeline, achieving 24 FPS and 46 FPS respectively. These enhancements incur only a minimal area overhead of 0.2\% relative to the entire SoC chip area, underscoring the practicality and efficiency of our approach for enabling 3DGS rendering on resource-constrained platforms. 

**Abstract (ZH)**: 3D智能利用丰富的3D特征，在AI领域展现出有前景的研究前沿，其中3D渲染对于许多下游应用至关重要。3D高斯斑点图（3DGS）是一种新兴的高质量3D渲染方法，但由于其所需的巨大计算量，在现有GPU装备的边缘设备上实现实时执行是不切实际的。先前加速3DGS的努力依赖于专用加速器，这需要大量的集成开销和硬件成本。本工作提出了一种加速策略，该策略利用了3DGS管道与现代GPU中高度优化的传统图形管道之间的相似性。我们未开发专用加速器，而是增强现有GPU的光栅化硬件以高效支持3DGS操作。实验结果表明，处理速度提高了23倍，能耗降低了24倍，从而使原3DGS算法的端到端运行时间提高了6倍，而最新效率提升的管道则提高了4倍，分别达到了24 FPS和46 FPS。这些增强仅相对于整个SOC芯片面积产生了0.2%的最小面积开销，突显了我们方法在资源受限平台上实现3DGS渲染的实用性和效率。 

---
# Echoes of Power: Investigating Geopolitical Bias in US and China Large Language Models 

**Title (ZH)**: 权力回声：探究美国和中国大型语言模型中的地缘政治偏见 

**Authors**: Andre G. C. Pacheco, Athus Cavalini, Giovanni Comarela  

**Link**: [PDF](https://arxiv.org/pdf/2503.16679)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for generating human-like text, transforming human-machine interactions. However, their widespread adoption has raised concerns about their potential to influence public opinion and shape political narratives. In this work, we investigate the geopolitical biases in US and Chinese LLMs, focusing on how these models respond to questions related to geopolitics and international relations. We collected responses from ChatGPT and DeepSeek to a set of geopolitical questions and evaluated their outputs through both qualitative and quantitative analyses. Our findings show notable biases in both models, reflecting distinct ideological perspectives and cultural influences. However, despite these biases, for a set of questions, the models' responses are more aligned than expected, indicating that they can address sensitive topics without necessarily presenting directly opposing viewpoints. This study highlights the potential of LLMs to shape public discourse and underscores the importance of critically assessing AI-generated content, particularly in politically sensitive contexts. 

**Abstract (ZH)**: 大型语言模型（LLMs）作为生成类人类文本的强大工具，正在改变人机交互方式。然而，它们的广泛应用也引发了对其可能影响公众意见和塑造政治叙事潜在影响的担忧。在这项研究中，我们调查了中美LLMs在地缘政治偏见方面的情况，重点关注这些模型对与地缘政治和国际关系相关问题的响应。我们收集了ChatGPT和DeepSeek对一组地缘政治问题的回答，并通过定性和定量分析评估了它们的输出。研究结果表明，这两种模型都存在明显的偏见，反映出不同的意识形态视角和文化影响。然而，尽管存在这些偏见，对于一组问题，模型的回答比预期更加一致，表明它们可以在不直接呈现对立观点的情况下处理敏感话题。本研究强调了LLMs在塑造公众话语方面的潜力，并强调了在政治敏感背景下批判性评估AI生成内容的重要性。 

---
# Accelerating Transformer Inference and Training with 2:4 Activation Sparsity 

**Title (ZH)**: 用2:4激活稀疏性加速Transformer推理和训练 

**Authors**: Daniel Haziza, Timothy Chou, Dhruv Choudhary, Luca Wehrstedt, Francisco Massa, Jiecao Yu, Geonhwa Jeong, Supriya Rao, Patrick Labatut, Jesse Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.16672)  

**Abstract**: In this paper, we demonstrate how to leverage 2:4 sparsity, a popular hardware-accelerated GPU sparsity pattern, to activations to accelerate large language model training and inference. Crucially we exploit the intrinsic sparsity found in Squared-ReLU activations to provide this acceleration with no accuracy loss. Our approach achieves up to 1.3x faster Feed Forward Network (FFNs) in both the forwards and backwards pass. This work highlights the potential for sparsity to play a key role in accelerating large language model training and inference. 

**Abstract (ZH)**: 本文展示了如何利用2:4稀疏性这一流行的硬件加速GPU稀疏模式来加速大型语言模型的训练和推理，同时通过对Squared-ReLU激活中固有的稀疏性的利用，实现了零准确率损失的加速。我们的方法在前向和反向传播中分别实现了最多1.3倍更快的前馈网络(FFNs)速度。本文突显了稀疏性在加速大型语言模型训练和推理中的潜在关键作用。 

---
# Aligning Text-to-Music Evaluation with Human Preferences 

**Title (ZH)**: 文本到音乐评估的人类偏好对齐 

**Authors**: Yichen Huang, Zachary Novack, Koichi Saito, Jiatong Shi, Shinji Watanabe, Yuki Mitsufuji, John Thickstun, Chris Donahue  

**Link**: [PDF](https://arxiv.org/pdf/2503.16669)  

**Abstract**: Despite significant recent advances in generative acoustic text-to-music (TTM) modeling, robust evaluation of these models lags behind, relying in particular on the popular Fréchet Audio Distance (FAD). In this work, we rigorously study the design space of reference-based divergence metrics for evaluating TTM models through (1) designing four synthetic meta-evaluations to measure sensitivity to particular musical desiderata, and (2) collecting and evaluating on MusicPrefs, the first open-source dataset of human preferences for TTM systems. We find that not only is the standard FAD setup inconsistent on both synthetic and human preference data, but that nearly all existing metrics fail to effectively capture desiderata, and are only weakly correlated with human perception. We propose a new metric, the MAUVE Audio Divergence (MAD), computed on representations from a self-supervised audio embedding model. We find that this metric effectively captures diverse musical desiderata (average rank correlation 0.84 for MAD vs. 0.49 for FAD and also correlates more strongly with MusicPrefs (0.62 vs. 0.14). 

**Abstract (ZH)**: 尽管生成性声学文本到音乐（TTM）建模领域取得了显著的近期进展，但对这些模型的稳健评估仍落后于实际进展，特别是依赖于流行的Fréchet音频距离（FAD）。在本文中，我们通过（1）设计四种合成元评估来衡量对特定音乐需求的敏感性，以及（2）收集并评估首个开源的基于人类偏好的TTS系统数据集MusicPrefs，系统地研究基于参考的发散度度量的设计空间。我们发现，标准的FAD设置不仅在合成和人类偏好数据上不一致，而且几乎所有现有的度量都无法有效捕捉需求，且仅与人类感知呈弱相关性。我们提出了一种新的度量标准——自监督音频嵌入模型上的MAUVE音频发散度（MAD），发现该度量能更有效地捕捉多样化的音乐需求（MAD与FAD的平均等级相关性为0.84，而FAD为0.49），并且与MusicPrefs的相关性更强（0.62 vs. 0.14）。 

---
# Code Evolution Graphs: Understanding Large Language Model Driven Design of Algorithms 

**Title (ZH)**: 代码演化图：理解大型语言模型驱动的算法设计 

**Authors**: Niki van Stein, Anna V. Kononova, Lars Kotthoff, Thomas Bäck  

**Link**: [PDF](https://arxiv.org/pdf/2503.16668)  

**Abstract**: Large Language Models (LLMs) have demonstrated great promise in generating code, especially when used inside an evolutionary computation framework to iteratively optimize the generated algorithms. However, in some cases they fail to generate competitive algorithms or the code optimization stalls, and we are left with no recourse because of a lack of understanding of the generation process and generated codes. We present a novel approach to mitigate this problem by enabling users to analyze the generated codes inside the evolutionary process and how they evolve over repeated prompting of the LLM. We show results for three benchmark problem classes and demonstrate novel insights. In particular, LLMs tend to generate more complex code with repeated prompting, but additional complexity can hurt algorithmic performance in some cases. Different LLMs have different coding ``styles'' and generated code tends to be dissimilar to other LLMs. These two findings suggest that using different LLMs inside the code evolution frameworks might produce higher performing code than using only one LLM. 

**Abstract (ZH)**: 大型语言模型在进化计算框架中生成代码并迭代优化代码方面展现了巨大的潜力，但有时无法生成竞争性的算法或代码优化停滞不前，由于缺乏对生成过程和生成代码的理解，我们对此束手无策。我们提出了一种新的方法，使用户能够在进化过程中分析生成的代码及其在重复提示LLM时如何演变。我们展示了三种基准问题类别的结果，并展示了新的见解。具体而言，随着重复提示LLM，生成的代码变得更加复杂，但额外的复杂性在某些情况下可能损害算法性能。不同LLM具有不同的编码风格，生成的代码往往与其他LLM生成的代码不同。这两项发现表明，在代码进化框架中使用不同的LLM可能产生性能更高的代码，而不仅仅使用一个LLM。 

---
# MobilePlantViT: A Mobile-friendly Hybrid ViT for Generalized Plant Disease Image Classification 

**Title (ZH)**: MobilePlantViT：一种适用于通用植物病害图像分类的移动友好型混合ViT 

**Authors**: Moshiur Rahman Tonmoy, Md. Mithun Hossain, Nilanjan Dey, M. F. Mridha  

**Link**: [PDF](https://arxiv.org/pdf/2503.16628)  

**Abstract**: Plant diseases significantly threaten global food security by reducing crop yields and undermining agricultural sustainability. AI-driven automated classification has emerged as a promising solution, with deep learning models demonstrating impressive performance in plant disease identification. However, deploying these models on mobile and edge devices remains challenging due to high computational demands and resource constraints, highlighting the need for lightweight, accurate solutions for accessible smart agriculture systems. To address this, we propose MobilePlantViT, a novel hybrid Vision Transformer (ViT) architecture designed for generalized plant disease classification, which optimizes resource efficiency while maintaining high performance. Extensive experiments across diverse plant disease datasets of varying scales show our model's effectiveness and strong generalizability, achieving test accuracies ranging from 80% to over 99%. Notably, with only 0.69 million parameters, our architecture outperforms the smallest versions of MobileViTv1 and MobileViTv2, despite their higher parameter counts. These results underscore the potential of our approach for real-world, AI-powered automated plant disease classification in sustainable and resource-efficient smart agriculture systems. All codes will be available in the GitHub repository: this https URL 

**Abstract (ZH)**: 植物疾病严重威胁全球粮食安全，通过降低作物产量和削弱农业可持续性。基于AI的自动化分类方法 emerge 作为一项有前景的解决方案，深度学习模型在植物病害识别方面展示了出色的性能。然而，将这些模型部署在移动和边缘设备上仍然面临挑战，由于计算需求高和资源限制，突显了轻量级、准确的解决方案对于可访问的智能农业系统的需求。为了解决这一问题，我们提出了MobilePlantViT，这是一种新型混合Vision Transformer (ViT) 架构，旨在实现泛化的植物病害分类，该架构优化了资源效率，同时保持高性能。跨不同规模的多种植物病害数据集的大量实验显示了我们模型的有效性和强大的泛化能力，其测试准确率从80%到超过99%不等。值得注意的是，尽管MobileViTv1和MobileViTv2的参数量更高，我们的架构仅包含0.69百万个参数，但仍表现出色。这些结果强调了我们方法在可持续和资源高效的智能农业系统中实现AI驱动的自动化植物病害分类的潜力。所有代码将在GitHub仓库中提供：this https URL 

---
# Classification of User Reports for Detection of Faulty Computer Components using NLP Models: A Case Study 

**Title (ZH)**: 基于NLP模型的用户报告分类研究：以检测故障计算机组件为例 

**Authors**: Maria de Lourdes M. Silva, André L. C. Mendonça, Eduardo R. D. Neto, Iago C. Chaves, Felipe T. Brito, Victor A. E. Farias, Javam C. Machado  

**Link**: [PDF](https://arxiv.org/pdf/2503.16614)  

**Abstract**: Computer manufacturers typically offer platforms for users to report faults. However, there remains a significant gap in these platforms' ability to effectively utilize textual reports, which impedes users from describing their issues in their own words. In this context, Natural Language Processing (NLP) offers a promising solution, by enabling the analysis of user-generated text. This paper presents an innovative approach that employs NLP models to classify user reports for detecting faulty computer components, such as CPU, memory, motherboard, video card, and more. In this work, we build a dataset of 341 user reports obtained from many sources. Additionally, through extensive experimental evaluation, our approach achieved an accuracy of 79% with our dataset. 

**Abstract (ZH)**: 计算机制造商通常为用户提供报告故障的平台。然而，这些平台在有效利用文本报告方面仍存在显著差距，阻碍了用户用他们自己的语言描述问题。在此背景下，自然语言处理（NLP）提供了一种有前景的解决方案，通过使用户生成的文本得以分析。本文提出了一种创新方法，利用NLP模型对用户报告进行分类，以检测故障的计算机组件，如CPU、内存、主板、显卡等。在此工作中，我们构建了一个包含341份用户报告的数据集，这些报告来自多个来源。此外，通过广泛的经验性评估，我们的方法在数据集上的准确率达到79%。 

---
# A Recipe for Generating 3D Worlds From a Single Image 

**Title (ZH)**: 从单张图片生成3D世界的 recipe 

**Authors**: Katja Schwarz, Denys Rozumnyi, Samuel Rota Bulò, Lorenzo Porzi, Peter Kontschieder  

**Link**: [PDF](https://arxiv.org/pdf/2503.16611)  

**Abstract**: We introduce a recipe for generating immersive 3D worlds from a single image by framing the task as an in-context learning problem for 2D inpainting models. This approach requires minimal training and uses existing generative models. Our process involves two steps: generating coherent panoramas using a pre-trained diffusion model and lifting these into 3D with a metric depth estimator. We then fill unobserved regions by conditioning the inpainting model on rendered point clouds, requiring minimal fine-tuning. Tested on both synthetic and real images, our method produces high-quality 3D environments suitable for VR display. By explicitly modeling the 3D structure of the generated environment from the start, our approach consistently outperforms state-of-the-art, video synthesis-based methods along multiple quantitative image quality metrics. Project Page: this https URL 

**Abstract (ZH)**: 我们提出了一种生成单张图像的沉浸式3D世界的食谱，将任务重新定义为针对2D修复模型的上下文学习问题。该方法需要少量训练，并利用现有的生成模型。我们的过程分为两步：使用预训练的扩散模型生成连贯的全景图，并用度量深度估计器将其提升到3D。然后，我们通过将修复模型条件化于渲染的点云来填充未观察到的区域，只需少量微调。在合成和真实图像上测试后，我们的方法生成适用于VR显示的高度高质量的3D环境。从一开始就明确建模生成环境的3D结构，我们的方法在多个定量图像质量指标上始终优于基于视频合成的方法。项目页面：这个链接。 

---
# Explainable AI-Guided Efficient Approximate DNN Generation for Multi-Pod Systolic Arrays 

**Title (ZH)**: 可解释AI引导的高效近似DNN生成方法用于多节点 systolic阵列 

**Authors**: Ayesha Siddique, Khurram Khalil, Khaza Anuarul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2503.16583)  

**Abstract**: Approximate deep neural networks (AxDNNs) are promising for enhancing energy efficiency in real-world devices. One of the key contributors behind this enhanced energy efficiency in AxDNNs is the use of approximate multipliers. Unfortunately, the simulation of approximate multipliers does not usually scale well on CPUs and GPUs. As a consequence, this slows down the overall simulation of AxDNNs aimed at identifying the appropriate approximate multipliers to achieve high energy efficiency with a minimum accuracy loss. To address this problem, we present a novel XAI-Gen methodology, which leverages the analytical model of the emerging hardware accelerator (e.g., Google TPU v4) and explainable artificial intelligence (XAI) to precisely identify the non-critical layers for approximation and quickly discover the appropriate approximate multipliers for AxDNN layers. Our results show that XAI-Gen achieves up to 7x lower energy consumption with only 1-2% accuracy loss. We also showcase the effectiveness of the XAI-Gen approach through a neural architecture search (XAI-NAS) case study. Interestingly, XAI-NAS achieves 40\% higher energy efficiency with up to 5x less execution time when compared to the state-of-the-art NAS methods for generating AxDNNs. 

**Abstract (ZH)**: 基于可解释人工智能的AxDNNs能量效率增强方法：XAI-Gen 

---
# Machine Learning-Based Genomic Linguistic Analysis (Gene Sequence Feature Learning): A Case Study on Predicting Heavy Metal Response Genes in Rice 

**Title (ZH)**: 基于机器学习的基因组语言分析（基因序列特征学习）：预测水稻重金属响应基因的案例研究 

**Authors**: Ruiqi Yang, Jianxu Wang, Wei Yuan, Xun Wang, Mei Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16582)  

**Abstract**: This study explores the application of machine learning-based genetic linguistics for identifying heavy metal response genes in rice (Oryza sativa). By integrating convolutional neural networks and random forest algorithms, we developed a hybrid model capable of extracting and learning meaningful features from gene sequences, such as k-mer frequencies and physicochemical properties. The model was trained and tested on datasets of genes, achieving high predictive performance (precision: 0.89, F1-score: 0.82). RNA-seq and qRT-PCR experiments conducted on rice leaves which exposed to Hg0, revealed differential expression of genes associated with heavy metal responses, which validated the model's predictions. Co-expression network analysis identified 103 related genes, and a literature review indicated that these genes are highly likely to be involved in heavy metal-related biological processes. By integrating and comparing the analysis results with those of differentially expressed genes (DEGs), the validity of the new machine learning method was further demonstrated. This study highlights the efficacy of combining machine learning with genetic linguistics for large-scale gene prediction. It demonstrates a cost-effective and efficient approach for uncovering molecular mechanisms underlying heavy metal responses, with potential applications in developing stress-tolerant crop varieties. 

**Abstract (ZH)**: 基于机器学习的遗传语言学在水稻（Oryza sativa）中识别重金属响应基因的应用研究 

---
# Investigating Retrieval-Augmented Generation in Quranic Studies: A Study of 13 Open-Source Large Language Models 

**Title (ZH)**: 调查《古兰经》研究中的检索增强生成：13个开源大规模语言模型的研究 

**Authors**: Zahra Khalila, Arbi Haza Nasution, Winda Monika, Aytug Onan, Yohei Murakami, Yasir Bin Ismail Radi, Noor Mohammad Osmani  

**Link**: [PDF](https://arxiv.org/pdf/2503.16581)  

**Abstract**: Accurate and contextually faithful responses are critical when applying large language models (LLMs) to sensitive and domain-specific tasks, such as answering queries related to quranic studies. General-purpose LLMs often struggle with hallucinations, where generated responses deviate from authoritative sources, raising concerns about their reliability in religious contexts. This challenge highlights the need for systems that can integrate domain-specific knowledge while maintaining response accuracy, relevance, and faithfulness. In this study, we investigate 13 open-source LLMs categorized into large (e.g., Llama3:70b, Gemma2:27b, QwQ:32b), medium (e.g., Gemma2:9b, Llama3:8b), and small (e.g., Llama3.2:3b, Phi3:3.8b). A Retrieval-Augmented Generation (RAG) is used to make up for the problems that come with using separate models. This research utilizes a descriptive dataset of Quranic surahs including the meanings, historical context, and qualities of the 114 surahs, allowing the model to gather relevant knowledge before responding. The models are evaluated using three key metrics set by human evaluators: context relevance, answer faithfulness, and answer relevance. The findings reveal that large models consistently outperform smaller models in capturing query semantics and producing accurate, contextually grounded responses. The Llama3.2:3b model, even though it is considered small, does very well on faithfulness (4.619) and relevance (4.857), showing the promise of smaller architectures that have been well optimized. This article examines the trade-offs between model size, computational efficiency, and response quality while using LLMs in domain-specific applications. 

**Abstract (ZH)**: 准确且上下文忠实的响应对于将大型语言模型应用于宗教研究等敏感和领域特定任务至关重要。通用大型语言模型常常会遇到幻觉问题，导致生成的响应偏离权威来源，这在宗教场景中引发了其可靠性的担忧。这一挑战凸显了系统整合领域特定知识并保持响应准确性、相关性和忠实性的重要性。本研究调查了13个开源大型语言模型（包括Llama3:70b、Gemma2:27b、QwQ:32b等大型模型，Gemma2:9b、Llama3:8b等中型模型，以及Llama3.2:3b、Phi3:3.8b等小型模型），并使用检索增强生成（RAG）来弥补单独模型使用时的问题。本研究利用描述性的古兰经章节数据集，包括114个章节的意义、历史背景和特性，使模型能在回应前获取相关知识。模型使用由人类评估者设定的三个关键指标进行评估：上下文相关性、答案忠实性和答案相关性。研究发现，大型模型在捕捉查询语义和生成准确、上下文相关响应方面始终优于小型模型。尽管Llama3.2:3b模型属于小型模型，但其在忠实性（4.619）和相关性（4.857）方面表现优异，显示出已优化的小型架构的潜力。本文探讨了在领域特定应用中使用大型语言模型时，模型规模、计算效率与响应质量之间的权衡。 

---
# Feature selection strategies for optimized heart disease diagnosis using ML and DL models 

**Title (ZH)**: 用于优化心脏疾病诊断的ML和DL模型的特征选择策略 

**Authors**: Bilal Ahmad, Jinfu Chen, Haibao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.16577)  

**Abstract**: Heart disease remains one of the leading causes of morbidity and mortality worldwide, necessitating the development of effective diagnostic tools to enable early diagnosis and clinical decision-making. This study evaluates the impact of feature selection techniques Mutual Information (MI), Analysis of Variance (ANOVA), and Chi-Square on the predictive performance of various machine learning (ML) and deep learning (DL) models using a dataset of clinical indicators for heart disease. Eleven ML/DL models were assessed using metrics such as precision, recall, AUC score, F1-score, and accuracy. Results indicate that MI outperformed other methods, particularly for advanced models like neural networks, achieving the highest accuracy of 82.3% and recall score of 0.94. Logistic regression (accuracy 82.1%) and random forest (accuracy 80.99%) also demonstrated improved performance with MI. Simpler models such as Naive Bayes and decision trees achieved comparable results with ANOVA and Chi-Square, yielding accuracies of 76.45% and 75.99%, respectively, making them computationally efficient alternatives. Conversely, k Nearest Neighbors (KNN) and Support Vector Machines (SVM) exhibited lower performance, with accuracies ranging between 51.52% and 54.43%, regardless of the feature selection method. This study provides a comprehensive comparison of feature selection methods for heart disease prediction, demonstrating the critical role of feature selection in optimizing model performance. The results offer practical guidance for selecting appropriate feature selection techniques based on the chosen classification algorithm, contributing to the development of more accurate and efficient diagnostic tools for enhanced clinical decision-making in cardiology. 

**Abstract (ZH)**: Heart Disease 预测中特征选择技术的影响：互信息、方差分析和卡方检验在机器学习和深度学习模型中的绩效评估 

---
# Extract, Match, and Score: An Evaluation Paradigm for Long Question-context-answer Triplets in Financial Analysis 

**Title (ZH)**: 提取、匹配和评分：金融分析中长期问题-背景-答案 triplet的评估范式 

**Authors**: Bo Hu, Han Yuan, Vlad Pandelea, Wuqiong Luo, Yingzhu Zhao, Zheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.16575)  

**Abstract**: The rapid advancement of large language models (LLMs) has sparked widespread adoption across diverse applications, making robust evaluation frameworks crucial for assessing their performance. While conventional evaluation metrics remain applicable for shorter texts, their efficacy diminishes when evaluating the quality of long-form answers. This limitation is particularly critical in real-world scenarios involving extended questions, extensive context, and long-form answers, such as financial analysis or regulatory compliance. In this paper, we use a practical financial use case to illustrate applications that handle "long question-context-answer triplets". We construct a real-world financial dataset comprising long triplets and demonstrate the inadequacies of traditional metrics. To address this, we propose an effective Extract, Match, and Score (EMS) evaluation approach tailored to the complexities of long-form LLMs' outputs, providing practitioners with a reliable methodology for assessing LLMs' performance in complex real-world scenarios. 

**Abstract (ZH)**: 大规模语言模型的迅速发展促成了其在多种应用中的广泛应用，因此构建 robust 的评估框架对于评估其性能至关重要。虽然传统的评估指标适用于较短的文本，但在评估长文答复的质量时效果减弱。这一限制在涉及扩展问题、丰富背景和长文答复的实际场景中尤为重要，例如财务分析或合规性。本文通过一个实际的财务应用场景来展示处理“长问题-背景-答复三元组”的应用。我们构建了一个包含长三元组的现实世界财务数据集，并展示了传统指标的不足。为了解决这个问题，我们提出了一种适应长文生成模型输出复杂性的有效 Extract-Match-Score (EMS) 评估方法，为评估复杂实际场景中大语言模型的性能提供了可靠的方法。 

---
# AUV Acceleration Prediction Using DVL and Deep Learning 

**Title (ZH)**: AUV 加速预测：基于DVL和深度学习的方法 

**Authors**: Yair Stolero, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2503.16573)  

**Abstract**: Autonomous underwater vehicles (AUVs) are essential for various applications, including oceanographic surveys, underwater mapping, and infrastructure inspections. Accurate and robust navigation are critical to completing these tasks. To this end, a Doppler velocity log (DVL) and inertial sensors are fused together. Recently, a model-based approach demonstrated the ability to extract the vehicle acceleration vector from DVL velocity measurements. Motivated by this advancement, in this paper we present an end-to-end deep learning approach to estimate the AUV acceleration vector based on past DVL velocity measurements. Based on recorded data from sea experiments, we demonstrate that the proposed method improves acceleration vector estimation by more than 65% compared to the model-based approach by using data-driven techniques. As a result of our data-driven approach, we can enhance navigation accuracy and reliability in AUV applications, contributing to more efficient and effective underwater missions through improved accuracy and reliability. 

**Abstract (ZH)**: 自主水下机器人（AUVs）在海洋学调查、水下测绘和基础设施检测等多种应用中至关重要。精确可靠的导航对于完成这些任务至关重要。为此，将多普勒速度 log (DVL) 和惯性传感器融合在一起。近期，一种基于模型的方法展示了从 DVL 速度测量中提取水下机器人加速度矢量的能力。受此进展的启发，本文提出了一种基于端到端深度学习的方法，通过利用过去 DVL 速度测量数据估算 AUV 的加速度矢量。通过对海洋实验记录的数据进行分析，我们证明所提出的方法在使用数据驱动技术的情况下，相比基于模型的方法，加速度矢量估计性能提高超过 65%。由于我们采用数据驱动的方法，可以提高 AUV 应用中的导航精度和可靠性，从而通过增强准确性和可靠性来实现更高效有效的水下任务。 

---
# Efficient ANN-Guided Distillation: Aligning Rate-based Features of Spiking Neural Networks through Hybrid Block-wise Replacement 

**Title (ZH)**: 基于ANN引导的蒸馏：通过混合块级替代对.spike神经网络的速率基特征进行对齐 

**Authors**: Shu Yang, Chengting Yu, Lei Liu, Hanzhi Ma, Aili Wang, Erping Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16572)  

**Abstract**: Spiking Neural Networks (SNNs) have garnered considerable attention as a potential alternative to Artificial Neural Networks (ANNs). Recent studies have highlighted SNNs' potential on large-scale datasets. For SNN training, two main approaches exist: direct training and ANN-to-SNN (ANN2SNN) conversion. To fully leverage existing ANN models in guiding SNN learning, either direct ANN-to-SNN conversion or ANN-SNN distillation training can be employed. In this paper, we propose an ANN-SNN distillation framework from the ANN-to-SNN perspective, designed with a block-wise replacement strategy for ANN-guided learning. By generating intermediate hybrid models that progressively align SNN feature spaces to those of ANN through rate-based features, our framework naturally incorporates rate-based backpropagation as a training method. Our approach achieves results comparable to or better than state-of-the-art SNN distillation methods, showing both training and learning efficiency. 

**Abstract (ZH)**: 基于ANN视角的ANN-SNN蒸馏框架：块级替换策略引导的SNN训练 

---
# Gene42: Long-Range Genomic Foundation Model With Dense Attention 

**Title (ZH)**: Gene42: 长范围基因组基础模型与密集注意力 

**Authors**: Kirill Vishniakov, Boulbaba Ben Amor, Engin Tekin, Nancy A. ElNaker, Karthik Viswanathan, Aleksandr Medvedev, Aahan Singh, Maryam Nadeem, Mohammad Amaan Sayeed, Praveenkumar Kanithi, Tiago Magalhaes, Natalia Vassilieva, Dwarikanath Mahapatra, Marco Pimentel, and Shadab Khan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16565)  

**Abstract**: We introduce Gene42, a novel family of Genomic Foundation Models (GFMs) designed to manage context lengths of up to 192,000 base pairs (bp) at a single-nucleotide resolution. Gene42 models utilize a decoder-only (LLaMA-style) architecture with a dense self-attention mechanism. Initially trained on fixed-length sequences of 4,096 bp, our models underwent continuous pretraining to extend the context length to 192,000 bp. This iterative extension allowed for the comprehensive processing of large-scale genomic data and the capture of intricate patterns and dependencies within the human genome. Gene42 is the first dense attention model capable of handling such extensive long context lengths in genomics, challenging state-space models that often rely on convolutional operators among other mechanisms. Our pretrained models exhibit notably low perplexity values and high reconstruction accuracy, highlighting their strong ability to model genomic data. Extensive experiments on various genomic benchmarks have demonstrated state-of-the-art performance across multiple tasks, including biotype classification, regulatory region identification, chromatin profiling prediction, variant pathogenicity prediction, and species classification. The models are publicly available at this http URL. 

**Abstract (ZH)**: Gene42：一种新型的基因组基础模型家族，设计用于管理高达192,000碱基对的上下文长度 

---
# Chem42: a Family of chemical Language Models for Target-aware Ligand Generation 

**Title (ZH)**: Chem42：一种面向靶点的化学语言模型家族，用于配体生成 

**Authors**: Aahan Singh, Engin Tekin, Maryam Nadeem, Nancy A. ElNaker, Mohammad Amaan Sayeed, Natalia Vassilieva, Boulbaba Ben Amor  

**Link**: [PDF](https://arxiv.org/pdf/2503.16563)  

**Abstract**: Revolutionizing drug discovery demands more than just understanding molecular interactions - it requires generative models that can design novel ligands tailored to specific biological targets. While chemical Language Models (cLMs) have made strides in learning molecular properties, most fail to incorporate target-specific insights, restricting their ability to drive de-novo ligand generation. Chem42, a cutting-edge family of generative chemical Language Models, is designed to bridge this gap. By integrating atomic-level interactions with multimodal inputs from Prot42, a complementary protein Language Model, Chem42 achieves a sophisticated cross-modal representation of molecular structures, interactions, and binding patterns. This innovative framework enables the creation of structurally valid, synthetically accessible ligands with enhanced target specificity. Evaluations across diverse protein targets confirm that Chem42 surpasses existing approaches in chemical validity, target-aware design, and predicted binding affinity. By reducing the search space of viable drug candidates, Chem42 could accelerate the drug discovery pipeline, offering a powerful generative AI tool for precision medicine. Our Chem42 models set a new benchmark in molecule property prediction, conditional molecule generation, and target-aware ligand design. The models are publicly available at this http URL. 

**Abstract (ZH)**: Revolutionizing 药物发现需求的不仅仅是理解分子相互作用——还需要能够设计针对特定生物靶标的新颖配体的生成模型。Chem42：一种先进的生成化学语言模型，通过集成原子级相互作用与互补蛋白质语言模型Prot42的多模态输入，实现分子结构、相互作用及结合模式的复杂跨模态表示，从而能够创建结构合理、易于合成且具有增强靶标特异性的配体。Chem42在多种蛋白质靶标下的评估表明，它在化学有效性、目标感知设计以及预测结合亲和力方面超过了现有方法。通过缩小可行药物候选物的搜索空间，Chem42可以加速药物发现流程，提供一种强大的生成人工智能工具，用于精准医疗。我们的Chem42模型在分子性质预测、条件分子生成及目标感知配体设计方面设立了新的基准。模型已公开，访问地址见相应链接。 

---
# Advancing Problem-Based Learning in Biomedical Engineering in the Era of Generative AI 

**Title (ZH)**: 生成人工智能时代基于问题的学习在生物医学工程中的推进 

**Authors**: Micky C. Nnamdi, J. Ben Tamo, Wenqi Shi, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16558)  

**Abstract**: Problem-Based Learning (PBL) has significantly impacted biomedical engineering (BME) education since its introduction in the early 2000s, effectively enhancing critical thinking and real-world knowledge application among students. With biomedical engineering rapidly converging with artificial intelligence (AI), integrating effective AI education into established curricula has become challenging yet increasingly necessary. Recent advancements, including AI's recognition by the 2024 Nobel Prize, have highlighted the importance of training students comprehensively in biomedical AI. However, effective biomedical AI education faces substantial obstacles, such as diverse student backgrounds, limited personalized mentoring, constrained computational resources, and difficulties in safely scaling hands-on practical experiments due to privacy and ethical concerns associated with biomedical data. To overcome these issues, we conducted a three-year (2021-2023) case study implementing an advanced PBL framework tailored specifically for biomedical AI education, involving 92 undergraduate and 156 graduate students from the joint Biomedical Engineering program of Georgia Institute of Technology and Emory University. Our approach emphasizes collaborative, interdisciplinary problem-solving through authentic biomedical AI challenges. The implementation led to measurable improvements in learning outcomes, evidenced by high research productivity (16 student-authored publications), consistently positive peer evaluations, and successful development of innovative computational methods addressing real biomedical challenges. Additionally, we examined the role of generative AI both as a teaching subject and an educational support tool within the PBL framework. Our study presents a practical and scalable roadmap for biomedical engineering departments aiming to integrate robust AI education into their curricula. 

**Abstract (ZH)**: 基于问题的学习（PBL）自2000年初次引入以来，显著影响了生物医学工程（BME）教育，有效提升了学生的批判性思维和现实应用知识。随着生物医学工程与人工智能（AI）的迅速融合，将有效的AI教育整合到现有课程中变得既具挑战性又日益必要。近期的进展，包括AI在2024年诺贝尔奖中的认可，强调了全面培训生物医学AI学生的重要性。然而，有效的生物医学AI教育面临诸多障碍，如学生背景多样、个性化指导有限、计算资源受限以及由于生物医学数据的隐私和伦理问题而难以安全扩展实践性实验。为克服这些障碍，我们在2021-2023年间对一个定制化的PBL框架在生物医学AI教育中的应用进行了为期三年的案例研究，涉及来自乔治亚理工学院和埃默里大学联合生物医学工程项目的92名本科生和156名研究生。我们的方法强调通过真实的生物医学AI挑战实现跨学科的合作问题解决。实施结果显示在学习成果上取得了可测量的改善，体现在高科研生产力（16篇由学生作者撰写的论文）、持续的积极同侪评估以及成功开发了创新的计算方法来解决实际的生物医学挑战。此外，我们还探讨了生成式AI在PBL框架中的教学主题和教育支持工具的作用。我们的研究提供了一条实用且可扩展的道路，供生物医学工程系参考，以将坚实的AI教育整合到其课程中。 

---
# Reliable Radiologic Skeletal Muscle Area Assessment -- A Biomarker for Cancer Cachexia Diagnosis 

**Title (ZH)**: 可靠的放射学骨骼肌面积评估——癌症恶病质诊断的生物标志物 

**Authors**: Sabeen Ahmed, Nathan Parker, Margaret Park, Daniel Jeong, Lauren Peres, Evan W. Davis, Jennifer B. Permuth, Erin Siegel, Matthew B. Schabath, Yasin Yilmaz, Ghulam Rasool  

**Link**: [PDF](https://arxiv.org/pdf/2503.16556)  

**Abstract**: Cancer cachexia is a common metabolic disorder characterized by severe muscle atrophy which is associated with poor prognosis and quality of life. Monitoring skeletal muscle area (SMA) longitudinally through computed tomography (CT) scans, an imaging modality routinely acquired in cancer care, is an effective way to identify and track this condition. However, existing tools often lack full automation and exhibit inconsistent accuracy, limiting their potential for integration into clinical workflows. To address these challenges, we developed SMAART-AI (Skeletal Muscle Assessment-Automated and Reliable Tool-based on AI), an end-to-end automated pipeline powered by deep learning models (nnU-Net 2D) trained on mid-third lumbar level CT images with 5-fold cross-validation, ensuring generalizability and robustness. SMAART-AI incorporates an uncertainty-based mechanism to flag high-error SMA predictions for expert review, enhancing reliability. We combined the SMA, skeletal muscle index, BMI, and clinical data to train a multi-layer perceptron (MLP) model designed to predict cachexia at the time of cancer diagnosis. Tested on the gastroesophageal cancer dataset, SMAART-AI achieved a Dice score of 97.80% +/- 0.93%, with SMA estimated across all four datasets in this study at a median absolute error of 2.48% compared to manual annotations with SliceOmatic. Uncertainty metrics-variance, entropy, and coefficient of variation-strongly correlated with SMA prediction errors (0.83, 0.76, and 0.73 respectively). The MLP model predicts cachexia with 79% precision, providing clinicians with a reliable tool for early diagnosis and intervention. By combining automation, accuracy, and uncertainty awareness, SMAART-AI bridges the gap between research and clinical application, offering a transformative approach to managing cancer cachexia. 

**Abstract (ZH)**: 癌症恶病质是一种常见的代谢障碍，以严重的肌肉萎缩为特征，与不良的预后和生活质量相关。通过计算机断层扫描（CT）扫描纵向监测骨骼肌面积（SMA）是识别和追踪这种状况的有效方法。然而，现有工具往往缺乏完全自动化并且表现出不一致的准确性，限制了其在临床工作流程中的集成。为解决这些挑战，我们开发了SMAART-AI（基于AI的骨骼肌评估自动化可靠工具），该工具基于深度学习模型（nnU-Net 2D），经过5折交叉验证训练，以确保泛化能力和鲁棒性。SMAART-AI Incorporates一种基于不确定性的机制，对高误差SMA预测进行专家审查，提高可靠性。我们结合SMA、骨骼肌指数、BMI和临床数据，训练了一个多层感知机（MLP）模型，用于预测癌症诊断时的恶病质。在胃食管癌数据集上测试，SMAART-AI实现了97.80%±0.93%的Dice分数，与使用SliceOmatic的手动注释相比，在所有四个数据集中的SMA估算中位绝对误差为2.48%。不确定性度量——方差、熵和变异系数——与SMA预测误差强烈相关（分别为0.83、0.76和0.73）。MLP模型以79%的精度预测恶病质，为临床早期诊断和干预提供了一个可靠工具。通过结合自动化、准确性和不确定性意识，SMAART-AI弥合了科研与临床应用之间的差距，提供了一种管理癌症恶病质的变革性方法。 

---
# A Comprehensive Survey on Architectural Advances in Deep CNNs: Challenges, Applications, and Emerging Research Directions 

**Title (ZH)**: 深度CNN架构进展综述：挑战、应用及新兴研究方向 

**Authors**: Saddam Hussain Khan, Rashid Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2503.16546)  

**Abstract**: Deep Convolutional Neural Networks (CNNs) have significantly advanced deep learning, driving breakthroughs in computer vision, natural language processing, medical diagnosis, object detection, and speech recognition. Architectural innovations including 1D, 2D, and 3D convolutional models, dilated and grouped convolutions, depthwise separable convolutions, and attention mechanisms address domain-specific challenges and enhance feature representation and computational efficiency. Structural refinements such as spatial-channel exploitation, multi-path design, and feature-map enhancement contribute to robust hierarchical feature extraction and improved generalization, particularly through transfer learning. Efficient preprocessing strategies, including Fourier transforms, structured transforms, low-precision computation, and weight compression, optimize inference speed and facilitate deployment in resource-constrained environments. This survey presents a unified taxonomy that classifies CNN architectures based on spatial exploitation, multi-path structures, depth, width, dimensionality expansion, channel boosting, and attention mechanisms. It systematically reviews CNN applications in face recognition, pose estimation, action recognition, text classification, statistical language modeling, disease diagnosis, radiological analysis, cryptocurrency sentiment prediction, 1D data processing, video analysis, and speech recognition. In addition to consolidating architectural advancements, the review highlights emerging learning paradigms such as few-shot, zero-shot, weakly supervised, federated learning frameworks and future research directions include hybrid CNN-transformer models, vision-language integration, generative learning, etc. This review provides a comprehensive perspective on CNN's evolution from 2015 to 2025, outlining key innovations, challenges, and opportunities. 

**Abstract (ZH)**: 深度卷积神经网络（CNNs）极大地推动了深度学习的发展，促进了计算机视觉、自然语言处理、医学诊断、对象检测和语音识别等领域的突破。包括1D、2D和3D卷积模型、膨胀卷积、分组卷积、深度可分离卷积和注意力机制在内的架构创新解决了领域特定的挑战，提升了特征表示和计算效率。结构精炼如空域-通道利用、多路径设计和特征图增强促进了鲁棒的层级特征提取和泛化能力，特别是在迁移学习方面。高效的预处理策略，包括傅里叶变换、结构化变换、低精度计算和权重压缩，优化了推理速度并促进了资源受限环境下的部署。本文综述提出了一个统一的分类体系，根据空域利用、多路径结构、深度、宽度、维度扩展、通道增强和注意力机制对CNN架构进行分类。系统回顾了CNN在面部识别、姿态估计、动作识别、文本分类、统计语言建模、疾病诊断、影像分析、加密货币情绪预测、1D数据处理、视频分析和语音识别等方面的应用。此外，综述还强调了 emerging learning paradigms 如少样本学习、零样本学习、弱监督学习及联邦学习框架，并指出了未来的研究方向，包括混合CNN-Transformer模型、视觉-语言集成、生成学习等。本文对2015年至2025年间CNN的发展进行了综合概述，概述了关键创新、挑战和机遇。 

---
# Causal Discovery and Counterfactual Reasoning to Optimize Persuasive Dialogue Policies 

**Title (ZH)**: 因果发现与反事实推理以优化说服性对话策略 

**Authors**: Donghuo Zeng, Roberto Legaspi, Yuewen Sun, Xinshuai Dong, Kazushi Ikeda, Peter Spirtes, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16544)  

**Abstract**: Tailoring persuasive conversations to users leads to more effective persuasion. However, existing dialogue systems often struggle to adapt to dynamically evolving user states. This paper presents a novel method that leverages causal discovery and counterfactual reasoning for optimizing system persuasion capability and outcomes. We employ the Greedy Relaxation of the Sparsest Permutation (GRaSP) algorithm to identify causal relationships between user and system utterance strategies, treating user strategies as states and system strategies as actions. GRaSP identifies user strategies as causal factors influencing system responses, which inform Bidirectional Conditional Generative Adversarial Networks (BiCoGAN) in generating counterfactual utterances for the system. Subsequently, we use the Dueling Double Deep Q-Network (D3QN) model to utilize counterfactual data to determine the best policy for selecting system utterances. Our experiments with the PersuasionForGood dataset show measurable improvements in persuasion outcomes using our approach over baseline methods. The observed increase in cumulative rewards and Q-values highlights the effectiveness of causal discovery in enhancing counterfactual reasoning and optimizing reinforcement learning policies for online dialogue systems. 

**Abstract (ZH)**: 面向用户的有说服力的对话定制能够增强说服效果。然而，现有的对话系统往往难以适应动态变化的用户状态。本文提出了一种利用因果发现和反事实推理的方法，以优化系统的说服能力和成果。我们采用Greedy Relaxation of the Sparsest Permutation (GRaSP) 算法来识别用户和系统话语策略之间的因果关系，将用户策略视为状态，系统策略视为动作。GRaSP识别用户策略作为影响系统响应的因果因子，并指导Bidirectional Conditional Generative Adversarial Networks (BiCoGAN) 生成系统的反事实话语。随后，我们使用Dueling Double Deep Q-Network (D3QN) 模型利用反事实数据确定选择系统话语的最佳策略。我们的实验结果表明，使用本文方法相比基准方法在说服成果方面有可测量的改进。观察到累积奖励和Q值的增加进一步表明因果发现在增强反事实推理和优化在线对话系统强化学习策略方面的有效性。 

---
# Gender and content bias in Large Language Models: a case study on Google Gemini 2.0 Flash Experimental 

**Title (ZH)**: 大型语言模型中的性别和内容偏见：对Google Gemini 2.0 Flash Experimental的案例研究 

**Authors**: Roberto Balestri  

**Link**: [PDF](https://arxiv.org/pdf/2503.16534)  

**Abstract**: This study evaluates the biases in Gemini 2.0 Flash Experimental, a state-of-the-art large language model (LLM) developed by Google, focusing on content moderation and gender disparities. By comparing its performance to ChatGPT-4o, examined in a previous work of the author, the analysis highlights some differences in ethical moderation practices. Gemini 2.0 demonstrates reduced gender bias, notably with female-specific prompts achieving a substantial rise in acceptance rates compared to results obtained by ChatGPT-4o. It adopts a more permissive stance toward sexual content and maintains relatively high acceptance rates for violent prompts, including gender-specific cases. Despite these changes, whether they constitute an improvement is debatable. While gender bias has been reduced, this reduction comes at the cost of permitting more violent content toward both males and females, potentially normalizing violence rather than mitigating harm. Male-specific prompts still generally receive higher acceptance rates than female-specific ones. These findings underscore the complexities of aligning AI systems with ethical standards, highlighting progress in reducing certain biases while raising concerns about the broader implications of the model's permissiveness. Ongoing refinements are essential to achieve moderation practices that ensure transparency, fairness, and inclusivity without amplifying harmful content. 

**Abstract (ZH)**: 本研究评估了由谷歌开发的最先进的大型语言模型（LLM）Gemini 2.0 Flash Experimental中的偏见问题，重点关注内容审核和性别差距。通过将其性能与作者在之前工作中研究的ChatGPT-4o进行比较，分析突显了一些伦理审核实践的差异。Gemini 2.0展示了减少性别偏见的情况，特别是在针对女性的具体提示中，接受率大幅提升，高于ChatGPT-4o的结果。它在性内容方面采取了更为宽松的立场，并且对于暴力提示（包括性别特定情况）保持了较高的接受率。尽管存在这些变化，但是否构成了改进仍然是值得争议的问题。虽然性别偏见有所减少，但这种减少是以容忍更多针对男女的暴力内容为代价的，这可能会正常化暴力行为，而不是减轻伤害。男性特定提示的接受率仍然普遍高于女性特定提示。这些发现突显了将AI系统与伦理标准对齐的复杂性，尽管减少了某些偏见，但仍引起了关于模型宽松程度更广泛影响的担忧。持续优化对于实现确保透明度、公平性和包容性的审核实践至关重要，同时不放大有害内容。 

---
# From Patient Consultations to Graphs: Leveraging LLMs for Patient Journey Knowledge Graph Construction 

**Title (ZH)**: 从患者咨询到图构建：利用大语言模型构建患者旅程知识图谱 

**Authors**: Hassan S. Al Khatib, Sudip Mittal, Shahram Rahimi, Nina Marhamati, Sean Bozorgzad  

**Link**: [PDF](https://arxiv.org/pdf/2503.16533)  

**Abstract**: The transition towards patient-centric healthcare necessitates a comprehensive understanding of patient journeys, which encompass all healthcare experiences and interactions across the care spectrum. Existing healthcare data systems are often fragmented and lack a holistic representation of patient trajectories, creating challenges for coordinated care and personalized interventions. Patient Journey Knowledge Graphs (PJKGs) represent a novel approach to addressing the challenge of fragmented healthcare data by integrating diverse patient information into a unified, structured representation. This paper presents a methodology for constructing PJKGs using Large Language Models (LLMs) to process and structure both formal clinical documentation and unstructured patient-provider conversations. These graphs encapsulate temporal and causal relationships among clinical encounters, diagnoses, treatments, and outcomes, enabling advanced temporal reasoning and personalized care insights. The research evaluates four different LLMs, such as Claude 3.5, Mistral, Llama 3.1, and Chatgpt4o, in their ability to generate accurate and computationally efficient knowledge graphs. Results demonstrate that while all models achieved perfect structural compliance, they exhibited variations in medical entity processing and computational efficiency. The paper concludes by identifying key challenges and future research directions. This work contributes to advancing patient-centric healthcare through the development of comprehensive, actionable knowledge graphs that support improved care coordination and outcome prediction. 

**Abstract (ZH)**: 基于患者的医疗知识图谱构建方法学：利用大规模语言模型整合结构化与非结构化患者数据以支持个性化医疗决策 

---
# Modelling Emotions in Face-to-Face Setting: The Interplay of Eye-Tracking, Personality, and Temporal Dynamics 

**Title (ZH)**: 在面对面互动中建模情绪：眼动、人格和时间动态的交互作用 

**Authors**: Meisam Jamshidi Seikavandi, Jostein Fimland, Maria Barrett, Paolo Burelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.16532)  

**Abstract**: Accurate emotion recognition is pivotal for nuanced and engaging human-computer interactions, yet remains difficult to achieve, especially in dynamic, conversation-like settings. In this study, we showcase how integrating eye-tracking data, temporal dynamics, and personality traits can substantially enhance the detection of both perceived and felt emotions. Seventy-three participants viewed short, speech-containing videos from the CREMA-D dataset, while being recorded for eye-tracking signals (pupil size, fixation patterns), Big Five personality assessments, and self-reported emotional states. Our neural network models combined these diverse inputs including stimulus emotion labels for contextual cues and yielded marked performance gains compared to the state-of-the-art. Specifically, perceived valence predictions reached a macro F1-score of 0.76, and models incorporating personality traits and stimulus information demonstrated significant improvements in felt emotion accuracy. These results highlight the benefit of unifying physiological, individual and contextual factors to address the subjectivity and complexity of emotional expression. Beyond validating the role of user-specific data in capturing subtle internal states, our findings inform the design of future affective computing and human-agent systems, paving the way for more adaptive and cross-individual emotional intelligence in real-world interactions. 

**Abstract (ZH)**: 准确的情绪识别对于细腻和引人入胜的人机交互至关重要，但在动态、对话式的环境中仍难以实现。本研究展示了如何通过整合眼动追踪数据、时间动态和个性特质，大幅提高对感知和体验情绪的检测能力。七十名参与者观看了包含言语的CREMA-D数据集中的简短视频，并被记录了眼动信号（瞳孔大小、注视模式）、五大人格特质评估以及自我报告的情绪状态。我们的神经网络模型结合了这些多样的输入，包括刺激情绪标签作为上下文线索，相比现有最佳方法取得了显著的性能提升。特别是，感知的主观价值预测达到了宏观F1分数0.76，包含个性特质和刺激信息的模型在体验情绪准确性上表现出显著改进。这些结果强调了整合生理、个体和情境因素以解决情绪表达的主观性和复杂性的益处。我们不仅验证了用户特定数据在捕捉微妙内在状态方面的作用，还为未来的计算情感和人机系统的设计提供了指导，铺就了在实际交互中实现更适应性和跨个体的情感智能的道路。 

---
# Enhancing LLM Generation with Knowledge Hypergraph for Evidence-Based Medicine 

**Title (ZH)**: 增强LLM生成能力以支持基于证据的医学知识图谱 

**Authors**: Chengfeng Dou, Ying Zhang, Zhi Jin, Wenpin Jiao, Haiyan Zhao, Yongqiang Zhao, Zhengwei Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16530)  

**Abstract**: Evidence-based medicine (EBM) plays a crucial role in the application of large language models (LLMs) in healthcare, as it provides reliable support for medical decision-making processes. Although it benefits from current retrieval-augmented generation~(RAG) technologies, it still faces two significant challenges: the collection of dispersed evidence and the efficient organization of this evidence to support the complex queries necessary for EBM. To tackle these issues, we propose using LLMs to gather scattered evidence from multiple sources and present a knowledge hypergraph-based evidence management model to integrate these evidence while capturing intricate relationships. Furthermore, to better support complex queries, we have developed an Importance-Driven Evidence Prioritization (IDEP) algorithm that utilizes the LLM to generate multiple evidence features, each with an associated importance score, which are then used to rank the evidence and produce the final retrieval results. Experimental results from six datasets demonstrate that our approach outperforms existing RAG techniques in application domains of interest to EBM, such as medical quizzing, hallucination detection, and decision support. Testsets and the constructed knowledge graph can be accessed at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于证据的医学（EBM）在医疗领域应用大型语言模型（LLMs）中扮演着重要角色，因为它为医学决策过程提供了可靠的支撑。尽管EBM受益于当前的检索增强生成（RAG）技术，但它仍然面临两个重大挑战：分散证据的收集和组织，以支持EBM所需的复杂查询。为解决这些问题，我们提出使用LLM从多个源收集分散的证据，并通过知识超图为基础的证据管理模型整合这些证据，同时捕获复杂的相互关系。此外，为了更好地支持复杂查询，我们开发了一种基于重要性驱动的证据优先级排序（IDEP）算法，该算法利用LLM生成多个具有相关重要性评分的证据特征，然后根据这些特征对证据进行排序并产生最终检索结果。来自六个数据集的实验结果表明，我们的方法在适用于EBM的应用领域（如医学测验、幻觉检测和决策支持）中优于现有RAG技术。测试集和构建的知识图谱可访问 <https://this https URL>。 

---
# Safety Evaluation and Enhancement of DeepSeek Models in Chinese Contexts 

**Title (ZH)**: DeepSeek模型在中文环境下的安全性评估与提升 

**Authors**: Wenjing Zhang, Xuejiao Lei, Zhaoxiang Liu, Limin Han, Jiaojiao Zhao, Beibei Huang, Zhenhong Long, Junting Guo, Meijuan An, Rongjia Du, Ning Wang, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2503.16529)  

**Abstract**: DeepSeek-R1, renowned for its exceptional reasoning capabilities and open-source strategy, is significantly influencing the global artificial intelligence landscape. However, it exhibits notable safety shortcomings. Recent research conducted by Robust Intelligence, a subsidiary of Cisco, in collaboration with the University of Pennsylvania, revealed that DeepSeek-R1 achieves a 100\% attack success rate when processing harmful prompts. Furthermore, multiple security firms and research institutions have identified critical security vulnerabilities within the model. Although China Unicom has uncovered safety vulnerabilities of R1 in Chinese contexts, the safety capabilities of the remaining distilled models in the R1 series have not yet been comprehensively evaluated. To address this gap, this study utilizes the comprehensive Chinese safety benchmark CHiSafetyBench to conduct an in-depth safety evaluation of the DeepSeek-R1 series distilled models. The objective is to assess the safety capabilities of these models in Chinese contexts both before and after distillation, and to further elucidate the adverse effects of distillation on model safety. Building on these findings, we implement targeted safety enhancements for six distilled models. Evaluation results indicate that the enhanced models achieve significant improvements in safety while maintaining reasoning capabilities without notable degradation. We open-source the safety-enhanced models at this https URL to serve as a valuable resource for future research and optimization of DeepSeek models. 

**Abstract (ZH)**: DeepSeek-R1系列精简模型的安全性评估与增强：基于CHiSafetyBench的综合研究 

---
# HDLCoRe: A Training-Free Framework for Mitigating Hallucinations in LLM-Generated HDL 

**Title (ZH)**: HDLCoRe: 无需训练的框架，用于减轻LLM生成的HDL中的幻觉 

**Authors**: Heng Ping, Shixuan Li, Peiyu Zhang, Anzhe Cheng, Shukai Duan, Nikos Kanakaris, Xiongye Xiao, Wei Yang, Shahin Nazarian, Andrei Irimia, Paul Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16528)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable capabilities in code generation tasks. However, when applied to hardware description languages (HDL), these models exhibit significant limitations due to data scarcity, resulting in hallucinations and incorrect code generation. To address these challenges, we propose HDLCoRe, a training-free framework that enhances LLMs' HDL generation capabilities through prompt engineering techniques and retrieval-augmented generation (RAG). Our approach consists of two main components: (1) an HDL-aware Chain-of-Thought (CoT) prompting technique with self-verification that classifies tasks by complexity and type, incorporates domain-specific knowledge, and guides LLMs through step-by-step self-simulation for error correction; and (2) a two-stage heterogeneous RAG system that addresses formatting inconsistencies through key component extraction and efficiently retrieves relevant HDL examples through sequential filtering and re-ranking. HDLCoRe eliminates the need for model fine-tuning while substantially improving LLMs' HDL generation capabilities. Experimental results demonstrate that our framework achieves superior performance on the RTLLM2.0 benchmark, significantly reducing hallucinations and improving both syntactic and functional correctness. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）在代码生成任务中展现了显著的能力。然而，当应用于硬件描述语言（HDL）时，这些模型由于数据稀缺性表现出显著的局限性，导致生成幻觉和错误代码。为了应对这些挑战，我们提出了一种无需微调的HDLCoRe框架，通过提示工程技术和服务检索增强生成（RAG）技术来增强LLMs的HDL生成能力。我们的方法包括两个主要组成部分：（1）一种awareHDLCoRe有序思考（CoT）提示技术，带有自我验证功能，通过分类任务的复杂性和类型、整合领域特定知识，并通过逐步自我模拟引导LLMs进行错误校正；以及（2）一种两阶段异构RAG系统，通过关键组件提取解决格式不一致问题，并通过顺序过滤和重新排名高效检索相关HDL示例。HDLCoRe消除了模型微调的需要，同时大幅提高LLMs的HDL生成能力。实验结果表明，我们的框架在RTLLM2.0基准测试中实现了优异的性能，显著减少了幻觉，并提高了语法和功能正确性。 

---
# LLM Generated Persona is a Promise with a Catch 

**Title (ZH)**: LLM生成的人格是一种有 promise 的承诺，但也伴随着问题。 

**Authors**: Ang Li, Haozhe Chen, Hongseok Namkoong, Tianyi Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16527)  

**Abstract**: The use of large language models (LLMs) to simulate human behavior has gained significant attention, particularly through personas that approximate individual characteristics. Persona-based simulations hold promise for transforming disciplines that rely on population-level feedback, including social science, economic analysis, marketing research, and business operations. Traditional methods to collect realistic persona data face significant challenges. They are prohibitively expensive and logistically challenging due to privacy constraints, and often fail to capture multi-dimensional attributes, particularly subjective qualities. Consequently, synthetic persona generation with LLMs offers a scalable, cost-effective alternative. However, current approaches rely on ad hoc and heuristic generation techniques that do not guarantee methodological rigor or simulation precision, resulting in systematic biases in downstream tasks. Through extensive large-scale experiments including presidential election forecasts and general opinion surveys of the U.S. population, we reveal that these biases can lead to significant deviations from real-world outcomes. Our findings underscore the need to develop a rigorous science of persona generation and outline the methodological innovations, organizational and institutional support, and empirical foundations required to enhance the reliability and scalability of LLM-driven persona simulations. To support further research and development in this area, we have open-sourced approximately one million generated personas, available for public access and analysis at this https URL. 

**Abstract (ZH)**: 大型语言模型在模拟人类行为中的应用，特别是在基于个性特征的模拟方面已引起广泛关注。这种基于个性的模拟为依赖于群体反馈的学科带来了变革潜力，包括社会科学、经济分析、市场营销研究和企业运营。传统方法收集真实个性数据面临重大挑战。由于隐私限制，这些方法费用高昂且在物流上具有挑战性，且往往无法捕捉多维度特征，特别是主观品质。因此，使用大型语言模型生成合成个性提供了一种可扩展且成本效益高的替代方案。然而，目前的方法依赖于难以保证方法论严谨性和模拟精确性的随意生成技术，导致下游任务中出现系统性偏差。通过包括美国总统选举预测和一般公众意见调查在内的大规模实验，我们揭示了这些偏差可能导致与现实世界结果的重大偏差。我们的研究强调了需要发展严谨的个性生成科学，并概述了提高大型语言模型驱动的个性模拟可靠性和可扩展性的方法论创新、组织和制度支持以及实证基础。为支持该领域的进一步研究和发展，我们已开源约一百万个生成的个性特征，并可通过以下链接进行公共访问和分析：[此 https URL]。 

---
# KVShare: Semantic-Aware Key-Value Cache Sharing for Efficient Large Language Model Inference 

**Title (ZH)**: KVShare：具有语义意识的键值缓存共享以实现高效的大型语言模型推理 

**Authors**: Huan Yang, Renji Zhang, Deyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16525)  

**Abstract**: This paper presents KVShare, a multi-user Key-Value (KV) Cache sharing technology based on semantic similarity, designed to enhance the inference efficiency of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). Addressing the limitations of existing prefix caching (strict text prefix matching) and semantic caching (loss of response diversity), KVShare achieves fine-grained KV cache reuse through semantic alignment algorithms and differential editing operations. Experiments on real-world user conversation datasets demonstrate that KVShare improves KV cache hit rates by over 60%, while maintaining output quality comparable to full computation (no significant degradation in BLEU and Rouge-L metrics). This approach effectively reduces GPU resource consumption and is applicable to scenarios with repetitive queries, such as healthcare and education. 

**Abstract (ZH)**: 基于语义相似性的多用户键值缓存技术KVShare：提高大型语言模型和多模态大型语言模型的推理效率 

---
# Mind2: Mind-to-Mind Emotional Support System with Bidirectional Cognitive Discourse Analysis 

**Title (ZH)**: Mind2：双向认知话语分析的情感支持系统 

**Authors**: Shi Yin Hong, Uttamasha Oyshi, Quan Mai, Gibson Nkhata, Susan Gauch  

**Link**: [PDF](https://arxiv.org/pdf/2503.16523)  

**Abstract**: Emotional support (ES) systems alleviate users' mental distress by generating strategic supportive dialogues based on diverse user situations. However, ES systems are limited in their ability to generate effective ES dialogues that include timely context and interpretability, hindering them from earning public trust. Driven by cognitive models, we propose Mind-to-Mind (Mind2), an ES framework that approaches interpretable ES context modeling for the ES dialogue generation task from a discourse analysis perspective. Specifically, we perform cognitive discourse analysis on ES dialogues according to our dynamic discourse context propagation window, which accommodates evolving context as the conversation between the ES system and user progresses. To enhance interpretability, Mind2 prioritizes details that reflect each speaker's belief about the other speaker with bidirectionality, integrating Theory-of-Mind, physiological expected utility, and cognitive rationality to extract cognitive knowledge from ES conversations. Experimental results support that Mind2 achieves competitive performance versus state-of-the-art ES systems while trained with only 10\% of the available training data. 

**Abstract (ZH)**: 情感支持系统通过生成基于多样化用户情况的策略性支持对话来缓解用户的心理压力。然而，情感支持系统在生成包含及时上下文和可解释性的有效情感支持对话方面能力有限，阻碍了它们获得公众信任。基于认知模型，我们提出Mind-to-Mind（Mind2）框架，该框架从话语分析的角度出发，旨在进行可解释的情感支持上下文建模以生成情感支持对话。具体而言，我们根据动态话语上下文传播窗口对情感支持对话进行认知话语分析，以适应对话过程中逐渐变化的上下文。为了增强可解释性，Mind2优先考虑反映每位对话者对另一方信念的细节，并通过双向方式整合心智理论、生理预期效用和认知理性从情感支持对话中提取认知知识。实验结果表明，Mind2仅使用可用训练数据的10%即可实现与最先进的情感支持系统相当的性能。 

---
# Not All Personas Are Worth It: Culture-Reflective Persona Data Augmentation 

**Title (ZH)**: 并非所有persona都值得使用：反映文化的人设数据增强 

**Authors**: Ji-Eun Han, Yoonseok Heo  

**Link**: [PDF](https://arxiv.org/pdf/2503.16520)  

**Abstract**: Incorporating personas into conversational AI models is crucial for achieving authentic and engaging interactions. However, the cultural diversity and adaptability of existing persona datasets is often overlooked, reducing their efficacy in building culturally aware AI systems. To address this issue, we propose a two-step pipeline for generating culture-specific personas and introduce KoPersona, a dataset comprising 200,000 personas designed to capture Korean cultural values, behaviors, and social nuances. A comprehensive evaluation through various metrics validates the quality of KoPersona and its relevance to Korean culture. This work not only contributes to persona-based research, but also establishes a scalable approach for creating culturally relevant personas adaptable to various languages and cultural contexts. 

**Abstract (ZH)**: 将人格融入对话AI模型对于实现真实且引人入胜的交互至关重要。然而，现有人格数据集的文化多样性和适应性往往被忽视，这限制了其在构建文化aware AI系统中的效果。为解决这一问题，我们提出了一种两步管道生成文化特定的人格，并介绍了包含200,000个人格的KoPersona数据集，旨在捕捉韩国文化价值观、行为和社会细微差别。通过多种指标进行全面评估验证了KoPersona的质量及其与韩国文化的相关性。本研究不仅推进了基于人格的研究，还建立了一种可扩展的方法，用于创建适用于多种语言和文化背景的相关人格。 

---
# Advancing Human-Machine Teaming: Concepts, Challenges, and Applications 

**Title (ZH)**: 提升人机协同：概念、挑战与应用 

**Authors**: Dian Chen, Han Jun Yoon, Zelin Wan, Nithin Alluru, Sang Won Lee, Richard He, Terrence J. Moore, Frederica F. Nelson, Sunghyun Yoon, Hyuk Lim, Dan Dongseong Kim, Jin-Hee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2503.16518)  

**Abstract**: Human-Machine Teaming (HMT) is revolutionizing collaboration across domains such as defense, healthcare, and autonomous systems by integrating AI-driven decision-making, trust calibration, and adaptive teaming. This survey presents a comprehensive taxonomy of HMT, analyzing theoretical models, including reinforcement learning, instance-based learning, and interdependence theory, alongside interdisciplinary methodologies. Unlike prior reviews, we examine team cognition, ethical AI, multi-modal interactions, and real-world evaluation frameworks. Key challenges include explainability, role allocation, and scalable benchmarking. We propose future research in cross-domain adaptation, trust-aware AI, and standardized testbeds. By bridging computational and social sciences, this work lays a foundation for resilient, ethical, and scalable HMT systems. 

**Abstract (ZH)**: Human-Machine Teaming：跨领域重塑协作的理论与方法综述 

---
# From G-Factor to A-Factor: Establishing a Psychometric Framework for AI Literacy 

**Title (ZH)**: 从G因子到A因子：建立人工智能素养的心理测量框架 

**Authors**: Ning Li, Wenming Deng, Jiatan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.16517)  

**Abstract**: This research addresses the growing need to measure and understand AI literacy in the context of generative AI technologies. Through three sequential studies involving a total of 517 participants, we establish AI literacy as a coherent, measurable construct with significant implications for education, workforce development, and social equity. Study 1 (N=85) revealed a dominant latent factor - termed the "A-factor" - that accounts for 44.16% of variance across diverse AI interaction tasks. Study 2 (N=286) refined the measurement tool by examining four key dimensions of AI literacy: communication effectiveness, creative idea generation, content evaluation, and step-by-step collaboration, resulting in an 18-item assessment battery. Study 3 (N=146) validated this instrument in a controlled laboratory setting, demonstrating its predictive validity for real-world task performance. Results indicate that AI literacy significantly predicts performance on complex, language-based creative tasks but shows domain specificity in its predictive power. Additionally, regression analyses identified several significant predictors of AI literacy, including cognitive abilities (IQ), educational background, prior AI experience, and training history. The multidimensional nature of AI literacy and its distinct factor structure provide evidence that effective human-AI collaboration requires a combination of general and specialized abilities. These findings contribute to theoretical frameworks of human-AI collaboration while offering practical guidance for developing targeted educational interventions to promote equitable access to the benefits of generative AI technologies. 

**Abstract (ZH)**: 本研究针对生成型人工智能技术背景下的AI素养 measurement and understanding growing needs, 通过三项 sequential 研究共涉及517名参与者, 确立了AI素养作为一个连贯且可衡量的建构, 并对教育、劳动力发展和社会公平具有重要意义。研究1 (N=85) 揭示了一个主导的潜在因素——称为“A因子”——解释了多种AI交互任务中44.16%的变异。研究2 (N=286) 通过探索AI素养的四个关键维度——沟通有效性、创意理念生成、内容评价和逐步协作——完善了测量工具, 形成了一个包含18项的测评电池。研究3 (N=146) 在受控实验室环境中验证了该工具, 证明其对实际任务表现具有预测有效性。结果表明, AI素养在复杂语言创造性任务上的表现具有显著性预测, 但在预测能力上表现出领域特异性。此外, 回归分析发现了几个与AI素养显著相关的预测因子, 包括认知能力(IQ)、教育背景、先前的AI经验以及培训历史。AI素养的多维度性质及其独特的因子结构提供了有力证据, 即有效的AI人机协作需要一般能力和专业能力的结合。这些发现为人类-人工智能协作的理论框架做出了贡献, 同时也为促进公平获取生成型人工智能技术带来的好处提供实用指导。 

---
# Using LLMs for Automated Privacy Policy Analysis: Prompt Engineering, Fine-Tuning and Explainability 

**Title (ZH)**: 使用大语言模型进行自动化隐私政策分析：提示工程、微调与解释性 

**Authors**: Yuxin Chen, Peng Tang, Weidong Qiu, Shujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16516)  

**Abstract**: Privacy policies are widely used by digital services and often required for legal purposes. Many machine learning based classifiers have been developed to automate detection of different concepts in a given privacy policy, which can help facilitate other automated tasks such as producing a more reader-friendly summary and detecting legal compliance issues. Despite the successful applications of large language models (LLMs) to many NLP tasks in various domains, there is very little work studying the use of LLMs for automated privacy policy analysis, therefore, if and how LLMs can help automate privacy policy analysis remains under-explored. To fill this research gap, we conducted a comprehensive evaluation of LLM-based privacy policy concept classifiers, employing both prompt engineering and LoRA (low-rank adaptation) fine-tuning, on four state-of-the-art (SOTA) privacy policy corpora and taxonomies. Our experimental results demonstrated that combining prompt engineering and fine-tuning can make LLM-based classifiers outperform other SOTA methods, \emph{significantly} and \emph{consistently} across privacy policy corpora/taxonomies and concepts. Furthermore, we evaluated the explainability of the LLM-based classifiers using three metrics: completeness, logicality, and comprehensibility. For all three metrics, a score exceeding 91.1\% was observed in our evaluation, indicating that LLMs are not only useful to improve the classification performance, but also to enhance the explainability of detection results. 

**Abstract (ZH)**: 基于大型语言模型的隐私政策概念分类器的综合评估：从提示工程到LoRA微调 

---
# Highlighting Case Studies in LLM Literature Review of Interdisciplinary System Science 

**Title (ZH)**: LLM领域跨学科系统科学文献综述中的案例研究亮点 

**Authors**: Lachlan McGinness, Peter Baumgartner  

**Link**: [PDF](https://arxiv.org/pdf/2503.16515)  

**Abstract**: Large Language Models (LLMs) were used to assist four Commonwealth Scientific and Industrial Research Organisation (CSIRO) researchers to perform systematic literature reviews (SLR). We evaluate the performance of LLMs for SLR tasks in these case studies. In each, we explore the impact of changing parameters on the accuracy of LLM responses. The LLM was tasked with extracting evidence from chosen academic papers to answer specific research questions. We evaluate the models' performance in faithfully reproducing quotes from the literature and subject experts were asked to assess the model performance in answering the research questions. We developed a semantic text highlighting tool to facilitate expert review of LLM responses.
We found that state of the art LLMs were able to reproduce quotes from texts with greater than 95% accuracy and answer research questions with an accuracy of approximately 83%. We use two methods to determine the correctness of LLM responses; expert review and the cosine similarity of transformer embeddings of LLM and expert answers. The correlation between these methods ranged from 0.48 to 0.77, providing evidence that the latter is a valid metric for measuring semantic similarity. 

**Abstract (ZH)**: 大型语言模型（LLMs）被用于辅助四名 Commonwealth Scientific and Industrial Research Organisation (CSIRO) 研究人员进行系统文献综述（SLR）。我们在这些案例研究中评估了LLMs在SLR任务中的性能。在每个案例中，我们探索了改变参数对LLM响应准确性的影响。LLM的任务是从选定的学术论文中提取证据以回答特定的研究问题。我们评估了模型在忠实再现文献引文方面的表现，并邀请领域专家对模型回答研究问题的性能进行评估。我们开发了一种语义文本高亮工具，以促进领域专家对LLM响应的审查。我们发现，最先进的LLMs能够在超过95%的准确率下忠实再现文本引文，并在大约83%的准确率下回答研究问题。我们使用两种方法来确定LLM响应的正确性；专家审查和转换器嵌入表示的余弦相似度。这些方法的相关性范围从0.48到0.77，提供了后者是衡量语义相似性的有效度量标准的证据。 

---
# VeriMind: Agentic LLM for Automated Verilog Generation with a Novel Evaluation Metric 

**Title (ZH)**: VeriMind: 自主代理模型在新型评价指标下的自动化Verilog生成 

**Authors**: Bardia Nadimi, Ghali Omar Boutaib, Hao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16514)  

**Abstract**: Designing Verilog modules requires meticulous attention to correctness, efficiency, and adherence to design specifications. However, manually writing Verilog code remains a complex and time-consuming task that demands both expert knowledge and iterative refinement. Leveraging recent advancements in large language models (LLMs) and their structured text generation capabilities, we propose VeriMind, an agentic LLM framework for Verilog code generation that significantly automates and optimizes the synthesis process. Unlike traditional LLM-based code generators, VeriMind employs a structured reasoning approach: given a user-provided prompt describing design requirements, the system first formulates a detailed train of thought before the final Verilog code is generated. This multi-step methodology enhances interpretability, accuracy, and adaptability in hardware design. In addition, we introduce a novel evaluation metric-pass@ARC-which combines the conventional pass@k measure with Average Refinement Cycles (ARC) to capture both success rate and the efficiency of iterative refinement. Experimental results on diverse hardware design tasks demonstrated that our approach achieved up to $8.3\%$ improvement on pass@k metric and $8.1\%$ on pass@ARC metric. These findings underscore the transformative potential of agentic LLMs in automated hardware design, RTL development, and digital system synthesis. 

**Abstract (ZH)**: 基于大型语言模型的VeriMind框架：一种用于Verilog代码生成的代理型LLM框架 

---
# Medifact at PerAnsSumm 2025: Leveraging Lightweight Models for Perspective-Specific Summarization of Clinical Q&A Forums 

**Title (ZH)**: Medifact在PerAnsSumm 2025：利用轻量级模型进行临床问答论坛视角特定总结 

**Authors**: Nadia Saeed  

**Link**: [PDF](https://arxiv.org/pdf/2503.16513)  

**Abstract**: The PerAnsSumm 2025 challenge focuses on perspective-aware healthcare answer summarization (Agarwal et al., 2025). This work proposes a few-shot learning framework using a Snorkel-BART-SVM pipeline for classifying and summarizing open-ended healthcare community question-answering (CQA). An SVM model is trained with weak supervision via Snorkel, enhancing zero-shot learning. Extractive classification identifies perspective-relevant sentences, which are then summarized using a pretrained BART-CNN model. The approach achieved 12th place among 100 teams in the shared task, demonstrating computational efficiency and contextual accuracy. By leveraging pretrained summarization models, this work advances medical CQA research and contributes to clinical decision support systems. 

**Abstract (ZH)**: PerAnsSumm 2025 挑战聚焦于视角感知的医疗问答摘要 (Agarwal 等, 2025) 

---
# Token-Level Uncertainty-Aware Objective for Language Model Post-Training 

**Title (ZH)**: 基于令牌级别的不确定性意识目标函数的模型后训练 

**Authors**: Tingkai Liu, Ari S. Benjamin, Anthony M. Zador  

**Link**: [PDF](https://arxiv.org/pdf/2503.16511)  

**Abstract**: In the current work, we connect token-level uncertainty in causal language modeling to two types of training objectives: 1) masked maximum likelihood (MLE), 2) self-distillation. We show that masked MLE is effective in reducing epistemic uncertainty, and serve as an effective token-level automatic curriculum learning technique. However, masked MLE is prone to overfitting and requires self-distillation regularization to improve or maintain performance on out-of-distribution tasks. We demonstrate significant performance gain via the proposed training objective - combined masked MLE and self-distillation - across multiple architectures (Gemma, LLaMA, Phi) and datasets (Alpaca, ShareGPT, GSM8K), mitigating overfitting while maintaining adaptability during post-training. Our findings suggest that uncertainty-aware training provides an effective mechanism for enhancing language model training. 

**Abstract (ZH)**: 在当前工作中，我们将因果语言模型中的 token 级别不确定性与两种训练目标连接起来：1) 掩码最大似然估计 (MLE)，2) 自我蒸馏。我们表明，掩码 MLE 在减少epistemic不确定性方面有效，并作为有效的 token 级别自动 Curriculum Learning 技术。然而，掩码 MLE 容易过拟合，并需要通过自我蒸馏正则化来提高或保持在分布外任务中的性能。通过提出的训练目标——结合掩码 MLE 和自我蒸馏——我们在多个架构 (Gemma, LLaMA, Phi) 和数据集 (Alpaca, ShareGPT, GSM8K) 上实现了显著的性能提升，同时在后训练期间减轻过拟合并保持适应性。我们的研究结果表明，不确定性感知的训练为增强语言模型训练提供了一种有效机制。 

---
# Conversational AI as a Coding Assistant: Understanding Programmers' Interactions with and Expectations from Large Language Models for Coding 

**Title (ZH)**: Conversational AI作为编程辅助：理解程序员与大规模语言模型在编程中的互动及其期望 

**Authors**: Mehmet Akhoroz, Caglar Yildirim  

**Link**: [PDF](https://arxiv.org/pdf/2503.16508)  

**Abstract**: Conversational AI interfaces powered by large language models (LLMs) are increasingly used as coding assistants. However, questions remain about how programmers interact with LLM-based conversational agents, the challenges they encounter, and the factors influencing adoption. This study investigates programmers' usage patterns, perceptions, and interaction strategies when engaging with LLM-driven coding assistants. Through a survey, participants reported both the benefits, such as efficiency and clarity of explanations, and the limitations, including inaccuracies, lack of contextual awareness, and concerns about over-reliance. Notably, some programmers actively avoid LLMs due to a preference for independent learning, distrust in AI-generated code, and ethical considerations. Based on our findings, we propose design guidelines for improving conversational coding assistants, emphasizing context retention, transparency, multimodal support, and adaptability to user preferences. These insights contribute to the broader understanding of how LLM-based conversational agents can be effectively integrated into software development workflows while addressing adoption barriers and enhancing usability. 

**Abstract (ZH)**: 由大型语言模型（LLMs）驱动的对话式AI接口作为编码助手的使用：程序员的交互模式、挑战及影响因素研究 

---
# Fewer Than 1% of Explainable AI Papers Validate Explainability with Humans 

**Title (ZH)**: 少于1%的可解释人工智能论文通过人类验证解释性。 

**Authors**: Ashley Suh, Isabelle Hurley, Nora Smith, Ho Chit Siu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16507)  

**Abstract**: This late-breaking work presents a large-scale analysis of explainable AI (XAI) literature to evaluate claims of human explainability. We collaborated with a professional librarian to identify 18,254 papers containing keywords related to explainability and interpretability. Of these, we find that only 253 papers included terms suggesting human involvement in evaluating an XAI technique, and just 128 of those conducted some form of a human study. In other words, fewer than 1% of XAI papers (0.7%) provide empirical evidence of human explainability when compared to the broader body of XAI literature. Our findings underscore a critical gap between claims of human explainability and evidence-based validation, raising concerns about the rigor of XAI research. We call for increased emphasis on human evaluations in XAI studies and provide our literature search methodology to enable both reproducibility and further investigation into this widespread issue. 

**Abstract (ZH)**: 这项 Late-Breaking 工作呈现了对可解释 AI (XAI) 文献的大规模分析，以评估人类可解释性的主张。我们与专业图书管理员合作，识别出包含与可解释性及可理解性相关关键词的 18,254 篇论文。在这之中，我们发现仅有 253 篇论文包含了表明人类参与评估 XAI 技术的术语，且只有其中的 128 篇进行了某种形式的人类研究。换句话说，在与更广泛的 XAI 文献相比时，提供人类可解释性实证证据的 XAI 论文不到 1%（0.7%）。我们的研究结果强调了人类可解释性主张与基于证据的验证之间的重要差距，引发了对 XAI 研究严谨性的担忧。我们呼吁在 XAI 研究中加强对人类评估的重视，并提供我们的文献搜索方法以实现可重复性和进一步研究这一普遍问题。 

---
# Stakeholder Perspectives on Whether and How Social Robots Can Support Mediation and Advocacy for Higher Education Students with Disabilities 

**Title (ZH)**: 利益相关者视角下的社会机器人在支持高等教育残疾人学生调解和倡导中的作用与方式 

**Authors**: Alva Markelius, Julie Bailey, Jenny L. Gibson, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2503.16499)  

**Abstract**: This paper presents an iterative, participatory, empirical study that examines the potential of using artificial intelligence, such as social robots and large language models, to support mediation and advocacy for students with disabilities in higher education. Drawing on qualitative data from interviews and focus groups conducted with various stakeholders, including disabled students, disabled student representatives, and disability practitioners at the University of Cambridge, this study reports findings relating to understanding the problem space, ideating robotic support and participatory co-design of advocacy support robots. The findings highlight the potential of these technologies in providing signposting and acting as a sounding board or study companion, while also addressing limitations in empathic understanding, trust, equity, and accessibility. We discuss ethical considerations, including intersectional biases, the double empathy problem, and the implications of deploying social robots in contexts shaped by structural inequalities. Finally, we offer a set of recommendations and suggestions for future research, rethinking the notion of corrective technological interventions to tools that empower and amplify self-advocacy. 

**Abstract (ZH)**: 本研究呈献了一种迭代的、参与式的、实证的研究方法，探讨使用人工智能，如社会机器人和大型语言模型，来支持高等教育中残疾学生的调解和倡导的可能性。基于与多方利益相关者（包括剑桥大学的残疾学生、残疾学生代表和残疾人从业人员）进行的访谈和焦点小组的定性数据，本研究报告了关于理解问题领域、构思机器人支持以及参与式共同设计倡导支持机器人的发现。这些发现突显了这些技术在提供方向和充当提示板或学习伴侣方面的潜力，同时也指出了同理理解、信任、公平性和可访问性方面存在的局限性。我们讨论了伦理考虑，包括交叉偏见、同理心不足的问题以及在由结构性不平等塑造的背景下部署社会机器人的含义。最后，我们提出了对未来研究的建议和建议，重新思考矫正性技术干预措施的概念，使之成为增强自我倡导的工具。 

---
# Llms, Virtual Users, and Bias: Predicting Any Survey Question Without Human Data 

**Title (ZH)**: 大语言模型、虚拟用户和偏见：无需人类数据预测任何调研问题 

**Authors**: Enzo Sinacola, Arnault Pachot, Thierry Petit  

**Link**: [PDF](https://arxiv.org/pdf/2503.16498)  

**Abstract**: Large Language Models (LLMs) offer a promising alternative to traditional survey methods, potentially enhancing efficiency and reducing costs. In this study, we use LLMs to create virtual populations that answer survey questions, enabling us to predict outcomes comparable to human responses. We evaluate several LLMs-including GPT-4o, GPT-3.5, Claude 3.5-Sonnet, and versions of the Llama and Mistral models-comparing their performance to that of a traditional Random Forests algorithm using demographic data from the World Values Survey (WVS). LLMs demonstrate competitive performance overall, with the significant advantage of requiring no additional training data. However, they exhibit biases when predicting responses for certain religious and population groups, underperforming in these areas. On the other hand, Random Forests demonstrate stronger performance than LLMs when trained with sufficient data. We observe that removing censorship mechanisms from LLMs significantly improves predictive accuracy, particularly for underrepresented demographic segments where censored models struggle. These findings highlight the importance of addressing biases and reconsidering censorship approaches in LLMs to enhance their reliability and fairness in public opinion research. 

**Abstract (ZH)**: 大规模语言模型（LLMs）为传统调查方法提供了有前景的替代方案，有可能提高效率并降低成本。本研究使用LLMs创建虚拟人口以回答调查问题，从而能够预测与人类回应相近的结果。我们评估了几种LLMs，包括GPT-4o、GPT-3.5、Claude 3.5-Sonnet以及Llama和Mistral模型的不同版本，并将它们的性能与使用世界价值观调查（WVS）的人口统计数据训练的传统随机森林算法进行比较。总体而言，LLMs展示出了竞争力，其显著优势在于不需要额外的训练数据。然而，它们在预测某些宗教和人口群体的回答时表现出偏差，这些区域的性能欠佳。另一方面，当使用足够数据训练时，随机森林表现出比LLMs更好的性能。我们发现，移除LLMs中的审核机制显著提高了预测准确性，尤其是对于审核模型难以处理的未充分代表的人口细分群体。这些发现强调了在公共意见研究中解决偏见并重新考虑审核方法的重要性，以提高LLMs的可靠性和公平性。 

---
# Effective Yet Ephemeral Propaganda Defense: There Needs to Be More than One-Shot Inoculation to Enhance Critical Thinking 

**Title (ZH)**: 有效的但短暂的 propaganda 防御：提高批判性思维不仅需要一次性的免疫手段 

**Authors**: Nicolas Hoferer, Kilian Sprenkamp, Dorian Christoph Quelle, Daniel Gordon Jones, Zoya Katashinskaya, Alexandre Bovet, Liudmila Zavolokina  

**Link**: [PDF](https://arxiv.org/pdf/2503.16497)  

**Abstract**: In today's media landscape, propaganda distribution has a significant impact on society. It sows confusion, undermines democratic processes, and leads to increasingly difficult decision-making for news readers. We investigate the lasting effect on critical thinking and propaganda awareness on them when using a propaganda detection and contextualization tool. Building on inoculation theory, which suggests that preemptively exposing individuals to weakened forms of propaganda can improve their resilience against it, we integrate Kahneman's dual-system theory to measure the tools' impact on critical thinking. Through a two-phase online experiment, we measure the effect of several inoculation doses. Our findings show that while the tool increases critical thinking during its use, this increase vanishes without access to the tool. This indicates a single use of the tool does not create a lasting impact. We discuss the implications and propose possible approaches to improve the resilience against propaganda in the long-term. 

**Abstract (ZH)**: 当前媒体环境下，宣传信息的分发对社会产生了显著影响。它播下了困惑，削弱了民主进程，并导致新闻读者在决策时面临越来越多的困难。我们探讨了使用宣传检测与语境化工具时对批判性思维和宣传意识的持久影响。基于假设提前接触减弱形式的宣传可以增强个体对其的抵抗力的接种理论，我们结合卡尼曼的双系统理论来衡量该工具对批判性思维的影响。通过两阶段在线实验，我们测量了几种接种剂量的效果。研究发现，尽管在使用工具期间批判性思维增加了，但没有工具访问时这种增加会消失，这表明单次使用工具并不会产生持久影响。我们讨论了这些发现的意义，并提出了增强长期抵御宣传能力的可能方法。 

---
# The Impact of Generative AI Coding Assistants on Developers Who Are Visually Impaired 

**Title (ZH)**: 视障开发者使用生成式AI编程助手的影响研究 

**Authors**: Claudia Flores-Saviaga, Benjamin V. Hanrahan, Kashif Imteyaz, Steven Clarke, Saiph Savage  

**Link**: [PDF](https://arxiv.org/pdf/2503.16491)  

**Abstract**: The rapid adoption of generative AI in software development has impacted the industry, yet its effects on developers with visual impairments remain largely unexplored. To address this gap, we used an Activity Theory framework to examine how developers with visual impairments interact with AI coding assistants. For this purpose, we conducted a study where developers who are visually impaired completed a series of programming tasks using a generative AI coding assistant. We uncovered that, while participants found the AI assistant beneficial and reported significant advantages, they also highlighted accessibility challenges. Specifically, the AI coding assistant often exacerbated existing accessibility barriers and introduced new challenges. For example, it overwhelmed users with an excessive number of suggestions, leading developers who are visually impaired to express a desire for ``AI timeouts.'' Additionally, the generative AI coding assistant made it more difficult for developers to switch contexts between the AI-generated content and their own code. Despite these challenges, participants were optimistic about the potential of AI coding assistants to transform the coding experience for developers with visual impairments. Our findings emphasize the need to apply activity-centered design principles to generative AI assistants, ensuring they better align with user behaviors and address specific accessibility needs. This approach can enable the assistants to provide more intuitive, inclusive, and effective experiences, while also contributing to the broader goal of enhancing accessibility in software development. 

**Abstract (ZH)**: 视觉障碍开发者中生成式AI在软件开发中的影响研究 

---
# PythonPal: Enhancing Online Programming Education through Chatbot-Driven Personalized Feedback 

**Title (ZH)**: PythonPal: 通过基于聊天机器人的个性化反馈提升在线编程教育 

**Authors**: Sirinda Palahan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16487)  

**Abstract**: The rise of online programming education has necessitated more effective, personalized interactions, a gap that PythonPal aims to fill through its innovative learning system integrated with a chatbot. This research delves into PythonPal's potential to enhance the online learning experience, especially in contexts with high student-to-teacher ratios where there is a need for personalized feedback. PythonPal's design, featuring modules for conversation, tutorials, and exercises, was evaluated through student interactions and feedback. Key findings reveal PythonPal's proficiency in syntax error recognition and user query comprehension, with its intent classification model showing high accuracy. The system's performance in error feedback, though varied, demonstrates both strengths and areas for enhancement. Student feedback indicated satisfactory query understanding and feedback accuracy but also pointed out the need for faster responses and improved interaction quality. PythonPal's deployment promises to significantly enhance online programming education by providing immediate, personalized feedback and interactive learning experiences, fostering a deeper understanding of programming concepts among students. These benefits mark a step forward in addressing the challenges of distance learning, making programming education more accessible and effective. 

**Abstract (ZH)**: 在线编程教育的兴起 necessitated 更有效的个性化互动，PythonPal 通过其集成聊天机器人的创新学习系统旨在填补这一空白。本研究探讨了 PythonPal 在提高在线学习体验方面的潜在作用，尤其是在学生与教师比例高、需要个性化反馈的情况下。PythonPal 的设计包括对话模块、教程模块和练习模块，通过学生互动和反馈进行了评估。关键发现表明，PythonPal 在语法错误识别和用户查询理解方面表现出色，其意图分类模型准确性较高。系统在错误反馈方面的表现虽有差异，但仍显示出其优缺点。学生反馈显示查询理解和反馈准确性令人满意，但也指出了需要更快响应和改进互动质量的需求。PythonPal 的部署有望显著提升在线编程教育，通过提供即时个性化反馈和互动学习体验，促进学生对编程概念的深入理解。这些好处标志着在应对远程学习挑战方面迈出了一步，使编程教育更具访问性和有效性。 

---
# Accodemy: AI Powered Code Learning Platform to Assist Novice Programmers in Overcoming the Fear of Coding 

**Title (ZH)**: Accodemy：AI驱动的代码学习平台，帮助 novice programmers 克服编码恐惧 

**Authors**: M.A.F. Aamina, V. Kavishcan, W.M.P.B.B. Jayaratne, K.K.D.S.N. Kannangara, A.A. Aamil, Achini Adikari  

**Link**: [PDF](https://arxiv.org/pdf/2503.16486)  

**Abstract**: Computer programming represents a rapidly evolving and sought-after career path in the 21st century. Nevertheless, novice learners may find the process intimidating for several reasons, such as limited and highly competitive career opportunities, peer and parental pressure for academic success, and course difficulties. These factors frequently contribute to anxiety and eventual dropout as a result of fear. Furthermore, research has demonstrated that beginners are significantly deterred by the fear of failure, which results in programming anxiety and and a sense of being overwhelmed by intricate topics, ultimately leading to dropping out. This project undertakes an exploration beyond the scope of conventional code learning platforms by identifying and utilising effective and personalised strategies of learning. The proposed solution incorporates features such as AI-generated challenging questions, mindfulness quotes, and tips to motivate users, along with an AI chatbot that functions as a motivational aid. In addition, the suggested solution integrates personalized roadmaps and gamification elements to maintain user involvement. The project aims to systematically monitor the progress of novice programmers and enhance their knowledge of coding with a personalised, revised curriculum to help mitigate the fear of coding and boost confidence. 

**Abstract (ZH)**: 21世纪中计算机编程代表一种快速演变且需求旺盛的职业路径。然而，初学者可能因多种原因感到望而却步，包括有限且竞争激烈的职业机会、同龄人和父母对学术成功的压力以及课程难度。这些因素常导致焦虑，最终因恐惧而辍学。此外，研究证实，初学者对失败的恐惧显著地阻碍了他们的学习，导致编程焦虑和面对复杂主题时感到不知所措，最终导致辍学。本项目超越了传统代码学习平台的范畴，通过识别并利用有效的个性化学习策略来开展探索。所提议的解决方案包括AI生成的挑战性问题、正念名言以及激励提示，同时还包括一个作为激励工具的AI聊天机器人。此外，所提议的解决方案结合了个性化路线图和游戏化元素以保持用户参与。该项目旨在系统地监控初学者程序员的进展，并通过个性化修订后的课程体系增强他们的编程知识，从而减轻编程恐惧并提升信心。 

---
# Optimizing Generative AI's Accuracy and Transparency in Inductive Thematic Analysis: A Human-AI Comparison 

**Title (ZH)**: 优化生成式人工智能在归纳主题分析中的准确性和透明度：人类-人工智能比较研究 

**Authors**: Matthew Nyaaba, Min SungEun, Mary Abiswin Apam, Kwame Owoahene Acheampong, Emmanuel Dwamena  

**Link**: [PDF](https://arxiv.org/pdf/2503.16485)  

**Abstract**: This study explores the use of OpenAI's API for inductive thematic analysis, employing a stepwise strategy to enhance transparency and traceability in GenAI-generated coding. A five-phase analysis and evaluation process were followed. Using the stepwise prompt, GenAI effectively generated codes with supporting statements and references, categorized themes, and developed broader interpretations by linking them to real-world contexts. While GenAI performed at a comparable level to human coders in coding and theming, it exhibited a more generalized and conceptual approach to interpretation, whereas human coders provided more specific, theme-based interpretations. Mapping these processes onto Naeem et al.'s (2023) six-step thematic analysis framework, GenAI covered four out of the six steps, while human coders followed three steps. Although GenAI's coding, theming, and interpretation align with keywording, coding, theming, and interpretation in Naeem et al.'s framework, human coders' interpretations were more closely tied to themes rather than broader conceptualization. This study positions GenAI as a viable tool for conducting inductive thematic analysis with minimal human intervention, offering an efficient and structured approach to qualitative data analysis. Future research should explore the development of specialized prompts that align GenAI's inductive thematic analysis with established qualitative research frameworks. 

**Abstract (ZH)**: 本研究探讨了使用OpenAI的API进行归纳主题分析的方法，采用逐步策略以增强生成式人工智能生成编码的透明度和可追溯性。遵循五个阶段的分析和评估过程。通过逐步提示，生成式人工智能有效地生成了带有支持性陈述和参考的编码，分类了主题，并通过与现实世界情境的联系发展了更广泛的理解。尽管生成式人工智能在编码和主题化方面的表现与人类编码员相当，但在解释方面，生成式人工智能采用了一种更概括和概念化的做法，而人类编码员则提供了更具体且基于主题的解释。将这些过程映射到Naeem等人（2023）提出的六阶段主题分析框架，生成式人工智能覆盖了其中的四个阶段，而人类编码员遵循了三个阶段。虽然生成式人工智能的编码、主题化和解释与Naeem等人框架中的关键词提取、编码、主题化和解释相契合，但人类编码员的解释更紧密地与具体主题相关，而非广泛的概念化。本研究将生成式人工智能定位为一种在最少人工干预下进行归纳主题分析的有效工具，提供了一种高效且结构化的方法来分析定性数据。未来的研究应探索开发专门的提示，使生成式人工智能的归纳主题分析与已建立的定性研究框架相一致。 

---
# AI-Powered Episodic Future Thinking 

**Title (ZH)**: AI驱动的 episodic 未来思考 

**Authors**: Sareh Ahmadi, Michelle Rockwell, Megan Stuart, Allison Tegge, Xuan Wang, Jeffrey Stein, Edward A. Fox  

**Link**: [PDF](https://arxiv.org/pdf/2503.16484)  

**Abstract**: Episodic Future Thinking (EFT) is an intervention that involves vividly imagining personal future events and experiences in detail. It has shown promise as an intervention to reduce delay discounting - the tendency to devalue delayed rewards in favor of immediate gratification - and to promote behavior change in a range of maladaptive health behaviors. We present EFTeacher, an AI chatbot powered by the GPT-4-Turbo large language model, designed to generate EFT cues for users with lifestyle-related conditions. To evaluate the chatbot, we conducted a user study that included usability assessments and user evaluations based on content characteristics questionnaires, followed by semi-structured interviews. The study provides qualitative insights into participants' experiences and interactions with the chatbot and its usability. Our findings highlight the potential application of AI chatbots based on Large Language Models (LLMs) in EFT interventions, and offer design guidelines for future behavior-oriented applications. 

**Abstract (ZH)**: episodic未来思考(EFT)是一种干预措施，涉及生动地详细想象个人未来事件和体验。它显示出减少延迟折扣（倾向于以即时满足为优先而低估延迟奖励）以及促进各种不良健康行为改变的潜力。我们介绍了EFTeacher，一个基于GPT-4-Turbo大规模语言模型的AI聊天机器人，旨在为与生活方式相关条件的用户提供EFT提示。为评估聊天机器人，我们进行了一项用户研究，包括使用可用性评估和基于内容特性问卷的用户评价，随后进行了半结构化访谈。该研究提供了关于参与者与聊天机器人互动经验及可用性的定性见解。我们的研究结果强调了基于大规模语言模型（LLMs）的AI聊天机器人在EFT干预中的潜在应用，并为未来目标行为应用的设计提供了指导方针。 

---
# Human Preferences for Constructive Interactions in Language Model Alignment 

**Title (ZH)**: 人类在语言模型对齐中对建设性互动的偏好 

**Authors**: Yara Kyrychenko, Jon Roozenbeek, Brandon Davidson, Sander van der Linden, Ramit Debnath  

**Link**: [PDF](https://arxiv.org/pdf/2503.16480)  

**Abstract**: As large language models (LLMs) enter the mainstream, aligning them to foster constructive dialogue rather than exacerbate societal divisions is critical. Using an individualized and multicultural alignment dataset of over 7,500 conversations of individuals from 74 countries engaging with 21 LLMs, we examined how linguistic attributes linked to constructive interactions are reflected in human preference data used for training AI. We found that users consistently preferred well-reasoned and nuanced responses while rejecting those high in personal storytelling. However, users who believed that AI should reflect their values tended to place less preference on reasoning in LLM responses and more on curiosity. Encouragingly, we observed that users could set the tone for how constructive their conversation would be, as LLMs mirrored linguistic attributes, including toxicity, in user queries. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）进入主流，引导它们促进建设性对话而不是加剧社会分歧至关重要。通过使用包含来自74个国家的7,500多场对话的数据集，这些对话涉及21种LLM，并具有个性化和多元文化的特点，我们研究了与建设性互动相关的语言属性如何在用于训练AI的人类偏好数据中体现。我们发现，用户倾向于偏好有充足理由和细腻的回应，而非过度个人化的的故事讲述。然而，那些认为AI应该反映其价值观的用户倾向于在LLM回应中较少看重理性分析，而更看重好奇心。令人鼓舞的是，我们观察到用户能够通过其查询的语言属性，包括有害内容，来设定对话的建设性基调。 

---
# LeRAAT: LLM-Enabled Real-Time Aviation Advisory Tool 

**Title (ZH)**: LeRAAT: LLM驱动的实时航空 advisement 工具 

**Authors**: Marc R. Schlichting, Vale Rasmussen, Heba Alazzeh, Houjun Liu, Kiana Jafari, Amelia F. Hardy, Dylan M. Asmar, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2503.16477)  

**Abstract**: In aviation emergencies, high-stakes decisions must be made in an instant. Pilots rely on quick access to precise, context-specific information -- an area where emerging tools like large language models (LLMs) show promise in providing critical support. This paper introduces LeRAAT, a framework that integrates LLMs with the X-Plane flight simulator to deliver real-time, context-aware pilot assistance. The system uses live flight data, weather conditions, and aircraft documentation to generate recommendations aligned with aviation best practices and tailored to the particular situation. It employs a Retrieval-Augmented Generation (RAG) pipeline that extracts and synthesizes information from aircraft type-specific manuals, including performance specifications and emergency procedures, as well as aviation regulatory materials, such as FAA directives and standard operating procedures. We showcase the framework in both a virtual reality and traditional on-screen simulation, supporting a wide range of research applications such as pilot training, human factors research, and operational decision support. 

**Abstract (ZH)**: 在航空紧急事件中，高风险决策必须瞬时作出。飞行员依赖于快速获取精确的、情境相关的信息——这是新兴工具如大规模语言模型（LLMs）展现出关键支持潜力的领域。本文介绍了一种名为LeRAAT的框架，该框架将LLMs与X-Plane飞行模拟器集成，以提供实时的情境感知飞行员辅助。该系统利用实时飞行数据、天气状况和航空器文档来生成符合航空最佳实践且针对特定情境的建议。它采用检索增强生成（RAG）管道，从特定类型的航空器手册中提取和合成信息，包括性能规范和应急程序，以及航空监管材料，如联邦航空管理局（FAA）的指令和标准操作程序。我们通过虚拟现实和传统屏幕模拟展示了该框架，支持飞行员培训、人因研究和操作决策支持等多种研究应用。 

---
# From Voices to Worlds: Developing an AI-Powered Framework for 3D Object Generation in Augmented Reality 

**Title (ZH)**: 从声音到世界：开发一种基于人工智能的增强现实三维对象生成框架 

**Authors**: Majid Behravan, Denis Gracanin  

**Link**: [PDF](https://arxiv.org/pdf/2503.16474)  

**Abstract**: This paper presents Matrix, an advanced AI-powered framework designed for real-time 3D object generation in Augmented Reality (AR) environments. By integrating a cutting-edge text-to-3D generative AI model, multilingual speech-to-text translation, and large language models (LLMs), the system enables seamless user interactions through spoken commands. The framework processes speech inputs, generates 3D objects, and provides object recommendations based on contextual understanding, enhancing AR experiences. A key feature of this framework is its ability to optimize 3D models by reducing mesh complexity, resulting in significantly smaller file sizes and faster processing on resource-constrained AR devices. Our approach addresses the challenges of high GPU usage, large model output sizes, and real-time system responsiveness, ensuring a smoother user experience. Moreover, the system is equipped with a pre-generated object repository, further reducing GPU load and improving efficiency. We demonstrate the practical applications of this framework in various fields such as education, design, and accessibility, and discuss future enhancements including image-to-3D conversion, environmental object detection, and multimodal support. The open-source nature of the framework promotes ongoing innovation and its utility across diverse industries. 

**Abstract (ZH)**: 这篇论文介绍了一种名为Matrix的先进AI驱动框架，用于 augmented reality (AR) 环境中的实时3D物体生成。通过整合前沿的文字到3D生成AI模型、多语言语音到文本翻译以及大语言模型（LLMs），该系统能够通过语音命令实现无缝用户交互。该框架处理语音输入，生成3D物体，并基于上下文理解提供物体推荐，从而增强AR体验。该框架的一个关键功能是通过减少网格复杂性优化3D模型，从而在资源受限的AR设备上显著减小文件大小并加快处理速度。我们的方法解决了一系列挑战，包括高GPU使用率、大的模型输出尺寸以及实时系统的响应性，从而确保更流畅的用户体验。此外，系统配备有预生成的物体库，进一步减轻GPU负载并提高效率。我们展示了该框架在教育、设计和无障碍等多种领域的实际应用，并探讨了未来增强的功能，包括图像到3D的转换、环境物体检测以及多模态支持。该框架的开源性质促进了持续创新并使其在各个行业中具有广泛应用价值。 

---
# Human-AI Interaction Design Standards 

**Title (ZH)**: 人类-人工智能交互设计规范 

**Authors**: Chaoyi Zhao, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16472)  

**Abstract**: The rapid development of artificial intelligence (AI) has significantly transformed human-computer interactions, making it essential to establish robust design standards to ensure effective, ethical, and human-centered AI (HCAI) solutions. Standards serve as the foundation for the adoption of new technologies, and human-AI interaction (HAII) standards are critical to supporting the industrialization of AI technology by following an HCAI approach. These design standards aim to provide clear principles, requirements, and guidelines for designing, developing, deploying, and using AI systems, enhancing the user experience and performance of AI systems. Despite their importance, the creation and adoption of HCAI-based interaction design standards face challenges, including the absence of universal frameworks, the inherent complexity of HAII, and the ethical dilemmas that arise in such systems. This chapter provides a comparative analysis of HAII versus traditional human-computer interaction (HCI) and outlines guiding principles for HCAI-based design. It explores international, regional, national, and industry standards related to HAII design from an HCAI perspective and reviews design guidelines released by leading companies such as Microsoft, Google, and Apple. Additionally, the chapter highlights tools available for implementing HAII standards and presents case studies of human-centered interaction design for AI systems in diverse fields, including healthcare, autonomous vehicles, and customer service. It further examines key challenges in developing HAII standards and suggests future directions for the field. Emphasizing the importance of ongoing collaboration between AI designers, developers, and experts in human factors and HCI, this chapter stresses the need to advance HCAI-based interaction design standards to ensure human-centered AI solutions across various domains. 

**Abstract (ZH)**: 快速发展的人工智能（AI）已显著改变了人机交互，建立坚实的设计标准以确保有效、伦理和以人为本的AI（HCAI）解决方案变得至关重要。这些设计标准旨在为设计、开发、部署和使用AI系统提供清晰的原则、要求和指南，增强AI系统的用户体验和性能。尽管它们很重要，但基于HCAI的交互设计标准的创建和采用仍面临挑战，包括缺乏通用框架、HAII固有的复杂性以及此类系统中出现的伦理困境。本章对HAII与传统人机交互（HCI）进行了比较分析，并概述了基于HCAI的设计指导原则。本章还从HCAI的角度概述了与HAII设计相关的国际、区域、国家和行业标准，并回顾了微软、谷歌和苹果等领先公司发布的设计指南。此外，本章强调了实施HAII标准的可用工具，并展示了AI系统的人本交互设计案例研究，涉及医疗保健、自动驾驶车辆和客户服务等多个领域。本章进一步探讨了开发HAII标准的关键挑战，并提出了该领域的未来方向。强调人工智能设计师、开发者与人因和HCI专家之间持续合作的重要性，本章强调需推进基于HCAI的交互设计标准，以确保在各个领域提供以人为本的AI解决方案。 

---
# A Review of Brain-Computer Interface Technologies: Signal Acquisition Methods and Interaction Paradigms 

**Title (ZH)**: 脑机接口技术综述：信号获取方法与交互模式 

**Authors**: Yifan Wang, Cheng Jiang, Chenzhong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16471)  

**Abstract**: Brain-Computer Interface (BCI) technology facilitates direct communication between the human brain and external devices, representing a substantial advancement in human-machine interaction. This review provides an in-depth analysis of various BCI paradigms, including classic paradigms, current classifications, and hybrid paradigms, each with distinct characteristics and applications. Additionally, we explore a range of signal acquisition methods, classified into non-implantation, intervention, and implantation techniques, elaborating on their principles and recent advancements. By examining the interdependence between paradigms and signal acquisition technologies, this review offers a comprehensive perspective on how innovations in one domain propel progress in the other. The goal is to present insights into the future development of more efficient, user-friendly, and versatile BCI systems, emphasizing the synergy between paradigm design and signal acquisition techniques and their potential to transform the field. 

**Abstract (ZH)**: 脑-机接口（BCI）技术促进了人脑与外部设备之间的直接通信，代表了人机交互领域的重要进展。本文综述了各种BCI范式，包括经典范式、当前分类和混合范式，每种范式都有其独特的特征和应用。此外，还探讨了多种信号获取方法，分为非植入、介入和植入技术，并详细阐述了其原理和最新进展。通过分析范式与信号获取技术之间的相互依赖性，本文提供了BCI领域如何在二者创新中共同进步的全面视角。目标是呈现更高效率、用户友好且更具灵活性的BCI系统的发展洞察，强调范式设计与信号获取技术之间的协同作用及其对领域的潜在影响。 

---
# Towards properly implementing Theory of Mind in AI systems: An account of four misconceptions 

**Title (ZH)**: 关于在AI系统中适当实现理论心智的探讨：四种误解的阐释 

**Authors**: Ramira van der Meulen, Rineke Verbrugge, Max van Duijn  

**Link**: [PDF](https://arxiv.org/pdf/2503.16468)  

**Abstract**: The search for effective collaboration between humans and computer systems is one of the biggest challenges in Artificial Intelligence. One of the more effective mechanisms that humans use to coordinate with one another is theory of mind (ToM). ToM can be described as the ability to `take someone else's perspective and make estimations of their beliefs, desires and intentions, in order to make sense of their behaviour and attitudes towards the world'. If leveraged properly, this skill can be very useful in Human-AI collaboration.
This introduces the question how we implement ToM when building an AI system. Humans and AI Systems work quite differently, and ToM is a multifaceted concept, each facet rooted in different research traditions across the cognitive and developmental sciences. We observe that researchers from artificial intelligence and the computing sciences, ourselves included, often have difficulties finding their way in the ToM literature. In this paper, we identify four common misconceptions around ToM that we believe should be taken into account when developing an AI system. We have hyperbolised these misconceptions for the sake of the argument, but add nuance in their discussion.
The misconceptions we discuss are:
(1) "Humans Use a ToM Module, So AI Systems Should As Well".
(2) "Every Social Interaction Requires (Advanced) ToM".
(3) "All ToM is the Same".
(4) "Current Systems Already Have ToM".
After discussing the misconception, we end each section by providing tentative guidelines on how the misconception can be overcome. 

**Abstract (ZH)**: 人类与计算机系统之间有效协作的搜索是人工智能领域最大的挑战之一。其中一种更为有效的人类相互协调机制是心理理论（ToM）。ToM 可以描述为“站在他人角度思考，并对其信念、欲望和意图进行估算，从而理解其行为和对世界的态度”的能力。如若合理运用，这一技能对于人机协作大有裨益。
本论文探讨如何在构建AI系统时实现ToM。人类与AI系统的工作方式大不相同，而ToM是一个多维度的概念，每个维度源于认知科学和发育科学的不同研究传统。我们观察到，来自人工智能和计算科学的研究者，包括我们在内，往往难以在ToM文献中找到方向。在本文中，我们识别出四种关于ToM的常见误解，并认为在开发AI系统时应考虑这些误解。为了论点的需要，我们夸大了这些误解，但在讨论中增添了细微差别。
我们讨论的误解包括：
(1) “人类使用ToM模块，因此AI系统也应该有”。
(2) “每次社交互动都需要（高级）ToM”。
(3) “所有的ToM都一样”。
(4) “当前的系统已经具备ToM”。
在讨论每个误解后，我们为克服这些误解提供了初步指南。 

---
# Enhancing Explainability with Multimodal Context Representations for Smarter Robots 

**Title (ZH)**: 基于多模态上下文表示以提高可解释性让机器人更智能 

**Authors**: Anargh Viswanath, Lokesh Veeramacheneni, Hendrik Buschmeier  

**Link**: [PDF](https://arxiv.org/pdf/2503.16467)  

**Abstract**: Artificial Intelligence (AI) has significantly advanced in recent years, driving innovation across various fields, especially in robotics. Even though robots can perform complex tasks with increasing autonomy, challenges remain in ensuring explainability and user-centered design for effective interaction. A key issue in Human-Robot Interaction (HRI) is enabling robots to effectively perceive and reason over multimodal inputs, such as audio and vision, to foster trust and seamless collaboration. In this paper, we propose a generalized and explainable multimodal framework for context representation, designed to improve the fusion of speech and vision modalities. We introduce a use case on assessing 'Relevance' between verbal utterances from the user and visual scene perception of the robot. We present our methodology with a Multimodal Joint Representation module and a Temporal Alignment module, which can allow robots to evaluate relevance by temporally aligning multimodal inputs. Finally, we discuss how the proposed framework for context representation can help with various aspects of explainability in HRI. 

**Abstract (ZH)**: 人工智能（AI）在近年来取得了显著进步，驱动着各个领域创新，特别是在机器人技术领域。尽管机器人能够执行日益复杂的任务并实现越来越多的自主性，但在确保可解释性和以用户为中心的设计方面仍然存在挑战，以实现有效的交互。在人机交互（HRI）中，一个关键问题是如何让机器人有效感知和推理多模态输入（如音频和视觉），以促进信任和无缝协作。本文提出了一种泛化且可解释的多模态框架，用于上下文表示，旨在提高语音和视觉模态的融合。我们介绍了一个评估“相关性”的用例，该用例涉及用户口头表述与机器人视觉场景感知之间的相关性。我们提出了多模态联合表示模块和时间对齐模块的方法，以允许机器人通过时间对齐多模态输入来评估相关性。最后，我们讨论了所提出的上下文表示框架如何在HRI中帮助提高解释性。 

---
# ACE, Action and Control via Explanations: A Proposal for LLMs to Provide Human-Centered Explainability for Multimodal AI Assistants 

**Title (ZH)**: ACE：动作与控制借助解释：为多模态AI助手机给人中心化可解释性的提案 

**Authors**: Elizabeth Anne Watkins, Emanuel Moss, Ramesh Manuvinakurike, Meng Shi, Richard Beckwith, Giuseppe Raffa  

**Link**: [PDF](https://arxiv.org/pdf/2503.16466)  

**Abstract**: In this short paper we address issues related to building multimodal AI systems for human performance support in manufacturing domains. We make two contributions: we first identify challenges of participatory design and training of such systems, and secondly, to address such challenges, we propose the ACE paradigm: "Action and Control via Explanations". Specifically, we suggest that LLMs can be used to produce explanations in the form of human interpretable "semantic frames", which in turn enable end users to provide data the AI system needs to align its multimodal models and representations, including computer vision, automatic speech recognition, and document inputs. ACE, by using LLMs to "explain" using semantic frames, will help the human and the AI system to collaborate, together building a more accurate model of humans activities and behaviors, and ultimately more accurate predictive outputs for better task support, and better outcomes for human users performing manual tasks. 

**Abstract (ZH)**: 在本短文中，我们探讨了构建适用于制造领域的人机协作智能系统面临的问题，并提出了两个贡献：首先，我们识别出了参与设计和培训这类系统所面临的挑战；其次，为了应对这些挑战，我们提出了ACE范式：“通过解释进行行动与控制”。具体而言，我们建议可以利用大规模语言模型（LLM）生成人类可解释的“语义框架”形式的解释，从而使得最终用户能够提供AI系统所需的用于其跨模态模型和表示的数据，包括计算机视觉、自动语音识别和文档输入。通过利用LLM用语义框架“解释”的方式，ACE将有助于人与智能系统合作，共同构建更准确的人类活动和行为模型，并最终提供更准确的任务预测输出，以更好地支持进行手工任务的人类用户。 

---
# OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents 

**Title (ZH)**: OS-Kairos: 适应性交互的MLLM驱动GUI代理模型 

**Authors**: Pengzhou Cheng, Zheng Wu, Zongru Wu, Aston Zhang, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16465)  

**Abstract**: Autonomous graphical user interface (GUI) agents powered by multimodal large language models have shown great promise. However, a critical yet underexplored issue persists: over-execution, where the agent executes tasks in a fully autonomous way, without adequate assessment of its action confidence to compromise an adaptive human-agent collaboration. This poses substantial risks in complex scenarios, such as those involving ambiguous user instructions, unexpected interruptions, and environmental hijacks. To address the issue, we introduce OS-Kairos, an adaptive GUI agent capable of predicting confidence levels at each interaction step and efficiently deciding whether to act autonomously or seek human intervention. OS-Kairos is developed through two key mechanisms: (i) collaborative probing that annotates confidence scores at each interaction step; (ii) confidence-driven interaction that leverages these confidence scores to elicit the ability of adaptive interaction. Experimental results show that OS-Kairos substantially outperforms existing models on our curated dataset featuring complex scenarios, as well as on established benchmarks such as AITZ and Meta-GUI, with 24.59\%$\sim$87.29\% improvements in task success rate. OS-Kairos facilitates an adaptive human-agent collaboration, prioritizing effectiveness, generality, scalability, and efficiency for real-world GUI interaction. The dataset and codes are available at this https URL. 

**Abstract (ZH)**: 由多模态大语言模型驱动的自主图形用户界面（GUI）代理展现了巨大的潜力。然而，一个关键而未被充分探索的问题依然存在：过度执行，代理在完全自主执行任务时，缺乏对其行为信心的充分评估，从而破坏了适应性的人机协作。在复杂场景中，如模糊用户指令、意外中断和环境干预的情况下，这带来了显著的风险。为了解决这一问题，我们提出了OS-Kairos，这是一种能够预测每次交互步骤的信心水平并在必要时寻求人类干预以决定是否自主行动的适应性GUI代理。OS-Kairos通过两种关键机制开发：（i）协作探针，为每个交互步骤标注信心分数；（ii）信心驱动交互，利用这些信心分数以实现适应性交互的能力。实验结果显示，OS-Kairos在我们的精心设计的数据集以及AITZ和Meta-GUI等标准基准上表现出色，任务成功率提高了24.59%至87.29%。OS-Kairos促进了适应性的人机协作，优先考虑实际应用场景中GUI交互的有效性、通用性、可扩展性和效率。数据集和代码可在以下链接获取：this https URL 

---
# Human-Centered AI in Multidisciplinary Medical Discussions: Evaluating the Feasibility of a Chat-Based Approach to Case Assessment 

**Title (ZH)**: 面向人的AI在多学科医疗讨论中的应用：基于聊天的病例评估可行性评估 

**Authors**: Shinnosuke Sawano, Satoshi Kodera  

**Link**: [PDF](https://arxiv.org/pdf/2503.16464)  

**Abstract**: In this study, we investigate the feasibility of using a human-centered artificial intelligence (AI) chat platform where medical specialists collaboratively assess complex cases. As the target population for this platform, we focus on patients with cardiovascular diseases who are in a state of multimorbidity, that is, suffering from multiple chronic conditions. We evaluate simulated cases with multiple diseases using a chat application by collaborating with physicians to assess feasibility, efficiency gains through AI utilization, and the quantification of discussion content. We constructed simulated cases based on past case reports, medical errors reports and complex cases of cardiovascular diseases experienced by the physicians. The analysis of discussions across five simulated cases demonstrated a significant reduction in the time required for summarization using AI, with an average reduction of 79.98\%. Additionally, we examined hallucination rates in AI-generated summaries used in multidisciplinary medical discussions. The overall hallucination rate ranged from 1.01\% to 5.73\%, with an average of 3.62\%, whereas the harmful hallucination rate varied from 0.00\% to 2.09\%, with an average of 0.49\%. Furthermore, morphological analysis demonstrated that multidisciplinary assessments enabled a more complex and detailed representation of medical knowledge compared with single physician assessments. We examined structural differences between multidisciplinary and single physician assessments using centrality metrics derived from the knowledge graph. In this study, we demonstrated that AI-assisted summarization significantly reduced the time required for medical discussions while maintaining structured knowledge representation. These findings can support the feasibility of AI-assisted chat-based discussions as a human-centered approach to multidisciplinary medical decision-making. 

**Abstract (ZH)**: 本研究探讨了使用以人为中心的人工智能聊天平台进行医疗专家协作评估复杂病例的可行性，该平台的目标人群是患有心血管疾病的多病态患者，即同时患有多种慢性疾病的患者。通过与医生合作使用聊天应用评估模拟病例，探讨人工智能应用带来的效率提升以及讨论内容的量化。我们根据过去的病例报告、医疗错误报告和医生经历的心血管复杂病例构建了模拟病例。跨五个模拟病例的讨论分析显示，使用人工智能进行总结的时间明显减少，平均减少79.98%。此外，我们还检查了多学科医疗讨论中人工智能生成总结的幻觉率，总体幻觉率在1.01%至5.73%之间，平均为3.62%，有害幻觉率在0.00%至2.09%之间，平均为0.49%。进一步的形态分析表明，多学科评估比单科室医生评估提供了更复杂和详细的医学知识表示。通过知识图谱中得到的中心性指标研究了多学科和单科室医生评估之间的结构差异。本研究证明，人工智能辅助总结可在保持结构性知识表示的同时，显著缩短医疗讨论所需时间。这些发现支持了人工智能辅助基于聊天的讨论作为以人为中心的多学科医疗决策方法的可行性。 

---
# Rank-O-ToM: Unlocking Emotional Nuance Ranking to Enhance Affective Theory-of-Mind 

**Title (ZH)**: Rank-O-ToM: 解锁情感细微差别的排名以增强情感共情理论 

**Authors**: JiHyun Kim, JuneHyoung Kwon, MiHyeon Kim, Eunju Lee, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.16461)  

**Abstract**: Facial Expression Recognition (FER) plays a foundational role in enabling AI systems to interpret emotional nuances, a critical aspect of affective Theory of Mind (ToM). However, existing models often struggle with poor calibration and a limited capacity to capture emotional intensity and complexity. To address this, we propose Ranking the Emotional Nuance for Theory of Mind (Rank-O-ToM), a framework that leverages ordinal ranking to align confidence levels with the emotional spectrum. By incorporating synthetic samples reflecting diverse affective complexities, Rank-O-ToM enhances the nuanced understanding of emotions, advancing AI's ability to reason about affective states. 

**Abstract (ZH)**: 情绪精细排序用于理论心智（Rank-O-ToM）：一种利用序数排序提升情绪理解的框架 

---
# Beyond Final Answers: Evaluating Large Language Models for Math Tutoring 

**Title (ZH)**: 超越最终答案：评估大型语言模型在数学辅导中的应用 

**Authors**: Adit Gupta, Jennifer Reddig, Tommaso Calo, Daniel Weitekamp, Christopher J. MacLellan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16460)  

**Abstract**: Researchers have made notable progress in applying Large Language Models (LLMs) to solve math problems, as demonstrated through efforts like GSM8k, ProofNet, AlphaGeometry, and MathOdyssey. This progress has sparked interest in their potential use for tutoring students in mathematics. However, the reliability of LLMs in tutoring contexts -- where correctness and instructional quality are crucial -- remains underexplored. Moreover, LLM problem-solving capabilities may not necessarily translate into effective tutoring support for students. In this work, we present two novel approaches to evaluate the correctness and quality of LLMs in math tutoring contexts. The first approach uses an intelligent tutoring system for college algebra as a testbed to assess LLM problem-solving capabilities. We generate benchmark problems using the tutor, prompt a diverse set of LLMs to solve them, and compare the solutions to those generated by the tutor. The second approach evaluates LLM as tutors rather than problem solvers. We employ human evaluators, who act as students seeking tutoring support from each LLM. We then assess the quality and correctness of the support provided by the LLMs via a qualitative coding process. We applied these methods to evaluate several ChatGPT models, including 3.5 Turbo, 4, 4o, o1-mini, and o1-preview. Our findings show that when used as problem solvers, LLMs generate correct final answers for 85.5% of the college algebra problems tested. When employed interactively as tutors, 90% of LLM dialogues show high-quality instructional support; however, many contain errors -- only 56.6% are entirely correct. We conclude that, despite their potential, LLMs are not yet suitable as intelligent tutors for math without human oversight or additional mechanisms to ensure correctness and quality. 

**Abstract (ZH)**: 研究人员在利用大型语言模型（LLMs）解决数学问题方面取得了显著进展，如GSM8k、ProofNet、AlphaGeometry和MathOdyssey等努力所展示的那样。这一进展引发了它们在数学教学中的潜在应用兴趣。然而，LLMs在教学情境下的可靠性——正确性和教学质量至关重要——仍需进一步探索。此外，LLMs的解题能力不一定能转化为有效的学生教学支持。在本项工作中，我们提出了两种新的方法来评估LLMs在数学教学情境下的正确性和质量。第一种方法使用一个大学代数智能教学系统作为实验平台，评估LLMs的解题能力。我们利用该系统生成基准问题，促使一系列不同的LLMs求解这些问题，并将它们的答案与教师生成的答案进行比较。第二种方法将LLMs评估为辅导者而非解题者。我们采用了人类评价者，他们作为寻求LLMs教学支持的学生。然后，我们通过定性的编码过程来评估LLMs提供的支持的质量和正确性。我们应用这些方法评估了多个ChatGPT模型，包括3.5 Turbo、4、4o、o1-mini和o1-preview。研究结果显示，当作为解题者使用时，LLMs在测试的大学代数问题中生成正确最终答案的比例为85.5%。当作为交互式辅导者使用时，90%的LLM对话显示出高质量的教学支持，但其中许多包含错误，仅56.6%是完全正确的。因此，尽管潜力巨大，但在没有人类监督或额外机制确保正确性和质量的情况下，LLMs尚未适合用作数学智能辅导者。 

---
# Integrating Personality into Digital Humans: A Review of LLM-Driven Approaches for Virtual Reality 

**Title (ZH)**: 将人格融入数字人类：基于LLM的虚拟现实方法综述 

**Authors**: Iago Alves Brito, Julia Soares Dollis, Fernanda Bufon Färber, Pedro Schindler Freire Brasil Ribeiro, Rafael Teixeira Sousa, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2503.16457)  

**Abstract**: The integration of large language models (LLMs) into virtual reality (VR) environments has opened new pathways for creating more immersive and interactive digital humans. By leveraging the generative capabilities of LLMs alongside multimodal outputs such as facial expressions and gestures, virtual agents can simulate human-like personalities and emotions, fostering richer and more engaging user experiences. This paper provides a comprehensive review of methods for enabling digital humans to adopt nuanced personality traits, exploring approaches such as zero-shot, few-shot, and fine-tuning. Additionally, it highlights the challenges of integrating LLM-driven personality traits into VR, including computational demands, latency issues, and the lack of standardized evaluation frameworks for multimodal interactions. By addressing these gaps, this work lays a foundation for advancing applications in education, therapy, and gaming, while fostering interdisciplinary collaboration to redefine human-computer interaction in VR. 

**Abstract (ZH)**: 大型语言模型（LLMs）与虚拟现实（VR）环境的集成开启了创建更具沉浸感和互动性的数字人类的新途径。通过利用LLMs的生成能力以及面部表情和手势等多模态输出，虚拟代理可以模拟类似人类的性格和情绪，促进更丰富和更具吸引力的用户体验。本文提供了一种全面的方法回顾，探讨了零样本、少量样本和微调等方法，使数字人类能够采用细腻的性格特征。同时，本文还突出了将LLMs驱动的性格特征集成到VR中所面临的挑战，包括计算需求、延迟问题以及缺乏针对多模态交互的标准评估框架。通过解决这些差距，本文为基础视听觉交互应用在教育、治疗和游戏领域的发展奠定了基础，并促进了跨学科合作，以重新定义VR中的交互方式。 

---
# Position: Beyond Assistance -- Reimagining LLMs as Ethical and Adaptive Co-Creators in Mental Health Care 

**Title (ZH)**: 位置：超越辅助——重新构想在心理健康护理中作为伦理化和适应性共创者的语言模型 

**Authors**: Abeer Badawi, Md Tahmid Rahman Laskar, Jimmy Xiangji Huang, Shaina Raza, Elham Dolatabadi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16456)  

**Abstract**: This position paper argues for a fundamental shift in how Large Language Models (LLMs) are integrated into the mental health care domain. We advocate for their role as co-creators rather than mere assistive tools. While LLMs have the potential to enhance accessibility, personalization, and crisis intervention, their adoption remains limited due to concerns about bias, evaluation, over-reliance, dehumanization, and regulatory uncertainties. To address these challenges, we propose two structured pathways: SAFE-i (Supportive, Adaptive, Fair, and Ethical Implementation) Guidelines for ethical and responsible deployment, and HAAS-e (Human-AI Alignment and Safety Evaluation) Framework for multidimensional, human-centered assessment. SAFE-i provides a blueprint for data governance, adaptive model engineering, and real-world integration, ensuring LLMs align with clinical and ethical standards. HAAS-e introduces evaluation metrics that go beyond technical accuracy to measure trustworthiness, empathy, cultural sensitivity, and actionability. We call for the adoption of these structured approaches to establish a responsible and scalable model for LLM-driven mental health support, ensuring that AI complements-rather than replaces-human expertise. 

**Abstract (ZH)**: 这一立场论文主张在精神健康护理领域对大型语言模型（LLMs）的整合方式进行根本性转变。我们提倡将其视为协作共创者而非仅仅是辅助工具。尽管LLMs有可能提高可及性、个性化和危机干预，但由于偏见、评估、过度依赖、去人性化和监管不确定性等方面的担忧，其采用仍受到限制。为解决这些挑战，我们提出两条结构化的路径：SAFE-i（支持性、适应性、公平性和伦理实施）准则，用于负责任和伦理的部署，以及HAAS-e（人类-人工智能契合与安全性评估）框架，用于多维度的人本评估。SAFE-i为数据治理、适应性模型工程和实际整合提供了蓝图，确保LLMs符合临床和伦理标准。HAAS-e引入了超越技术准确性的评估标准，以衡量可信度、同理心、文化敏感性和可操作性。我们呼吁采用这些结构化方法，建立LLM驱动精神健康支持的负责任和可扩展模式，确保人工智能补充而非取代人类专长。 

---
# Bridging Structural Dynamics and Biomechanics: Human Motion Estimation through Footstep-Induced Floor Vibrations 

**Title (ZH)**: 结构动力学与生物力学的桥梁：通过脚步引起的地面振动进行人体运动估计 

**Authors**: Yiwen Dong, Jessica Rose, Hae Young Noh  

**Link**: [PDF](https://arxiv.org/pdf/2503.16455)  

**Abstract**: Quantitative estimation of human joint motion in daily living spaces is essential for early detection and rehabilitation tracking of neuromusculoskeletal disorders (e.g., Parkinson's) and mitigating trip and fall risks for older adults. Existing approaches involve monitoring devices such as cameras, wearables, and pressure mats, but have operational constraints such as direct line-of-sight, carrying devices, and dense deployment. To overcome these limitations, we leverage gait-induced floor vibration to estimate lower-limb joint motion (e.g., ankle, knee, and hip flexion angles), allowing non-intrusive and contactless gait health monitoring in people's living spaces. To overcome the high uncertainty in lower-limb movement given the limited information provided by the gait-induced floor vibrations, we formulate a physics-informed graph to integrate domain knowledge of gait biomechanics and structural dynamics into the model. Specifically, different types of nodes represent heterogeneous information from joint motions and floor vibrations; Their connecting edges represent the physiological relationships between joints and forces governed by gait biomechanics, as well as the relationships between forces and floor responses governed by the structural dynamics. As a result, our model poses physical constraints to reduce uncertainty while allowing information sharing between the body and the floor to make more accurate predictions. We evaluate our approach with 20 participants through a real-world walking experiment. We achieved an average of 3.7 degrees of mean absolute error in estimating 12 joint flexion angles (38% error reduction from baseline), which is comparable to the performance of cameras and wearables in current medical practices. 

**Abstract (ZH)**: 日常生活中人体关节运动的定量估计对于早期检测和康复跟踪神经肌骨疾病（如帕金森病）以及降低老年人跌倒风险至关重要。现有的方法涉及监控设备如摄像头、可穿戴设备和压力垫，但存在直接视线、携带设备和密集部署的操作限制。为克服这些限制，我们利用步态引起的地板振动来估计下肢关节运动（如踝关节、膝关节和髋关节的屈曲角度），从而在人们的居住空间中实现非侵入性和无接触的步态健康监测。为克服由步态引起的地板振动提供的有限信息导致的下肢运动高度不确定性，我们构建了一个物理信息图以整合步态 biomechanics 和结构动力学领域的专业知识。具体来说，不同类型节点代表关节运动和地板振动的异质信息；它们之间的连接边表示步态 biomechanics 控制下的关节和力之间的生理关系，以及结构动力学控制下的力与地板响应之间的关系。结果，我们的模型通过施加物理约束来降低不确定性，并允许身体和地板之间的信息共享，从而提高预测准确性。我们通过真实世界的步行实验对我们的方法进行了评估。我们平均在估计12个关节屈曲角度方面取得了3.7度的平均绝对误差（相较于基线的误差减少了38%），其性能与目前临床实践中摄像头和可穿戴设备的性能相当。 

---
# An Audio-Visual Fusion Emotion Generation Model Based on Neuroanatomical Alignment 

**Title (ZH)**: 基于神经解剖对齐的视听融合情感生成模型 

**Authors**: Haidong Wang, Qia Shan, JianHua Zhang, PengFei Xiao, Ao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16454)  

**Abstract**: In the field of affective computing, traditional methods for generating emotions predominantly rely on deep learning techniques and large-scale emotion datasets. However, deep learning techniques are often complex and difficult to interpret, and standardizing large-scale emotional datasets are difficult and costly to establish. To tackle these challenges, we introduce a novel framework named Audio-Visual Fusion for Brain-like Emotion Learning(AVF-BEL). In contrast to conventional brain-inspired emotion learning methods, this approach improves the audio-visual emotion fusion and generation model through the integration of modular components, thereby enabling more lightweight and interpretable emotion learning and generation processes. The framework simulates the integration of the visual, auditory, and emotional pathways of the brain, optimizes the fusion of emotional features across visual and auditory modalities, and improves upon the traditional Brain Emotional Learning (BEL) model. The experimental results indicate a significant improvement in the similarity of the audio-visual fusion emotion learning generation model compared to single-modality visual and auditory emotion learning and generation model. Ultimately, this aligns with the fundamental phenomenon of heightened emotion generation facilitated by the integrated impact of visual and auditory stimuli. This contribution not only enhances the interpretability and efficiency of affective intelligence but also provides new insights and pathways for advancing affective computing technology. Our source code can be accessed here: this https URL}{this https URL. 

**Abstract (ZH)**: 在情感计算领域，传统的用于生成情感的方法主要依赖于深度学习技术和大规模情感数据集。然而，深度学习技术往往复杂且难以解释，建立标准化的大规模情感数据集也极为困难和成本高昂。为解决这些挑战，我们提出了一种名为Audio-Visual Fusion for Brain-like Emotion Learning (AVF-BEL)的新颖框架。与传统的基于大脑的情感学习方法不同，该方法通过模块化组件的整合，增强了听觉和视觉情感融合与生成模型，从而使情感学习和生成过程更加轻量级和易于解释。该框架模拟了大脑的视觉、听觉和情绪通路的整合，优化了视觉和听觉模态之间的情感特征融合，并改进了传统的Brain Emotional Learning (BEL)模型。实验结果表明，与单一模态视觉和听觉情感学习和生成模型相比，听觉-视觉融合情感学习生成模型的相似性有了显著提高。最终，这与视觉和听觉刺激的综合影响引发的情感增强现象保持一致。这一贡献不仅提高了情感智能的可解释性和效率，还为推进情感计算技术提供了新的见解和路径。源代码可在此访问：this https URL this https URL。 

---
# Towards Biomarker Discovery for Early Cerebral Palsy Detection: Evaluating Explanations Through Kinematic Perturbations 

**Title (ZH)**: 面向早期脑瘫检测的生物标志物发现：通过运动动力学扰动评估解释性分析 

**Authors**: Kimji N. Pellano, Inga Strümke, Daniel Groos, Lars Adde, Pål Haugen, Espen Alexander F. Ihlen  

**Link**: [PDF](https://arxiv.org/pdf/2503.16452)  

**Abstract**: Cerebral Palsy (CP) is a prevalent motor disability in children, for which early detection can significantly improve treatment outcomes. While skeleton-based Graph Convolutional Network (GCN) models have shown promise in automatically predicting CP risk from infant videos, their "black-box" nature raises concerns about clinical explainability. To address this, we introduce a perturbation framework tailored for infant movement features and use it to compare two explainable AI (XAI) methods: Class Activation Mapping (CAM) and Gradient-weighted Class Activation Mapping (Grad-CAM). First, we identify significant and non-significant body keypoints in very low- and very high-risk infant video snippets based on the XAI attribution scores. We then conduct targeted velocity and angular perturbations, both individually and in combination, on these keypoints to assess how the GCN model's risk predictions change. Our results indicate that velocity-driven features of the arms, hips, and legs have a dominant influence on CP risk predictions, while angular perturbations have a more modest impact. Furthermore, CAM and Grad-CAM show partial convergence in their explanations for both low- and high-risk CP groups. Our findings demonstrate the use of XAI-driven movement analysis for early CP prediction and offer insights into potential movement-based biomarker discovery that warrant further clinical validation. 

**Abstract (ZH)**: 脑瘫（CP）是一种在儿童中常见的运动障碍，早期检测可以显著改善治疗效果。虽然基于骨架的图卷积网络（GCN）模型在自动预测婴儿视频中的CP风险方面显示出潜力，但其“黑盒”性质引发了关于临床解释性的担忧。为解决这一问题，我们引入了一种针对婴儿运动特征的扰动框架，并使用该框架比较了两种可解释人工智能（XAI）方法：Class Activation Mapping（CAM）和Gradient-weighted Class Activation Mapping（Grad-CAM）。首先，我们根据XAI归属评分，在极高风险和极低风险婴儿视频片段中识别出显著和不显著的体表关键点。然后，我们在这些关键点上进行有针对性的速度和角度扰动，单独和组合进行，评估GCN模型的风险预测如何变化。结果显示，手臂、髋部和腿部的速度驱动特征对CP风险预测起主导作用，而角度扰动能产生较小的影响。此外，CAM和Grad-CAM在低风险和高风险CP组的解释方面表现出部分一致性。我们的研究结果展示了通过XAI驱动的运动分析进行早期CP预测的应用，并提供了有关潜在运动基线生物标志物发现的见解，这些见解值得进一步的临床验证。 

---
# Think-Then-React: Towards Unconstrained Human Action-to-Reaction Generation 

**Title (ZH)**: 思考后再反应：迈向无约束的人类动作到反应生成 

**Authors**: Wenhui Tan, Boyuan Li, Chuhao Jin, Wenbing Huang, Xiting Wang, Ruihua Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.16451)  

**Abstract**: Modeling human-like action-to-reaction generation has significant real-world applications, like human-robot interaction and games. Despite recent advancements in single-person motion generation, it is still challenging to well handle action-to-reaction generation, due to the difficulty of directly predicting reaction from action sequence without prompts, and the absence of a unified representation that effectively encodes multi-person motion. To address these challenges, we introduce Think-Then-React (TTR), a large language-model-based framework designed to generate human-like reactions. First, with our fine-grained multimodal training strategy, TTR is capable to unify two processes during inference: a thinking process that explicitly infers action intentions and reasons corresponding reaction description, which serve as semantic prompts, and a reacting process that predicts reactions based on input action and the inferred semantic prompts. Second, to effectively represent multi-person motion in language models, we propose a unified motion tokenizer by decoupling egocentric pose and absolute space features, which effectively represents action and reaction motion with same encoding. Extensive experiments demonstrate that TTR outperforms existing baselines, achieving significant improvements in evaluation metrics, such as reducing FID from 3.988 to 1.942. 

**Abstract (ZH)**: 基于大型语言模型的Think-Then-React框架：生成人类反应的挑战与解决方案 

---
# Mitigating the Uncanny Valley Effect in Hyper-Realistic Robots: A Student-Centered Study on LLM-Driven Conversations 

**Title (ZH)**: 缓解超real机器人中的毛骨悚然谷效应：基于学生的LLM驱动对话研究 

**Authors**: Hangyeol Kang, Thiago Freitas dos Santos, Maher Ben Moussa, Nadia Magnenat-Thalmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.16449)  

**Abstract**: The uncanny valley effect poses a significant challenge in the development and acceptance of hyper-realistic social robots. This study investigates whether advanced conversational capabilities powered by large language models (LLMs) can mitigate this effect in highly anthropomorphic robots. We conducted a user study with 80 participants interacting with Nadine, a hyper-realistic humanoid robot equipped with LLM-driven communication skills. Through pre- and post-interaction surveys, we assessed changes in perceptions of uncanniness, conversational quality, and overall user experience. Our findings reveal that LLM-enhanced interactions significantly reduce feelings of eeriness while fostering more natural and engaging conversations. Additionally, we identify key factors influencing user acceptance, including conversational naturalness, human-likeness, and interestingness. Based on these insights, we propose design recommendations to enhance the appeal and acceptability of hyper-realistic robots in social contexts. This research contributes to the growing field of human-robot interaction by offering empirical evidence on the potential of LLMs to bridge the uncanny valley, with implications for the future development of social robots. 

**Abstract (ZH)**: 超现实社会机器人中拟人化效果的挑战：大型语言模型增强对话能力的缓解作用研究 

---
# FINCH: Locally Visualizing Higher-Order Feature Interactions in Black Box Models 

**Title (ZH)**: FINCH: 局部可视化黑盒模型中的高阶特征交互 

**Authors**: Anna Kleinau, Bernhard Preim, Monique Meuschke  

**Link**: [PDF](https://arxiv.org/pdf/2503.16445)  

**Abstract**: In an era where black-box AI models are integral to decision-making across industries, robust methods for explaining these models are more critical than ever. While these models leverage complex feature interplay for accurate predictions, most explanation methods only assign relevance to individual features. There is a research gap in methods that effectively illustrate interactions between features, especially in visualizing higher-order interactions involving multiple features, which challenge conventional representation methods. To address this challenge in local explanations focused on individual instances, we employ a visual, subset-based approach to reveal relevant feature interactions. Our visual analytics tool FINCH uses coloring and highlighting techniques to create intuitive, human-centered visualizations, and provides additional views that enable users to calibrate their trust in the model and explanations. We demonstrate FINCH in multiple case studies, demonstrating its generalizability, and conducted an extensive human study with machine learning experts to highlight its helpfulness and usability. With this approach, FINCH allows users to visualize feature interactions involving any number of features locally. 

**Abstract (ZH)**: 在黑盒AI模型广泛应用于各行业的决策时代，有效的解释方法比以往任何时候都更加重要。尽管这些模型依靠复杂的功能交互进行准确预测，但大多数解释方法仅对单个特征赋以相关性。在展示特征之间有效交互的方法，尤其是高阶交互，方面存在研究空白，这些交互涉及多个特征并挑战传统的表示方法。为解决这一挑战，我们采用基于子集的可视化方法来揭示相关特征交互。我们的可视化分析工具FINCH使用着色和突出显示技术创建直观的人类中心可视化，并提供了额外的视角，使用户能够校准其对模型和解释的信任度。我们通过多个案例研究展示了FINCH的普适性，并进行了广泛的专家研究来突出其帮助性和易用性。通过这种方法，FINCH允许用户局部可视化任何数量特征之间的交互。 

---
# Conversational Explanations: Discussing Explainable AI with Non-AI Experts 

**Title (ZH)**: 对话式解释：与非AI专家讨论可解释的人工智能 

**Authors**: Tong Zhang, Mengao Zhang, Wei Yan Low, X. Jessie Yang, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16444)  

**Abstract**: Explainable AI (XAI) aims to provide insights into the decisions made by AI models. To date, most XAI approaches provide only one-time, static explanations, which cannot cater to users' diverse knowledge levels and information needs. Conversational explanations have been proposed as an effective method to customize XAI explanations. However, building conversational explanation systems is hindered by the scarcity of training data. Training with synthetic data faces two main challenges: lack of data diversity and hallucination in the generated data. To alleviate these issues, we introduce a repetition penalty to promote data diversity and exploit a hallucination detector to filter out untruthful synthetic conversation turns. We conducted both automatic and human evaluations on the proposed system, fEw-shot Multi-round ConvErsational Explanation (EMCEE). For automatic evaluation, EMCEE achieves relative improvements of 81.6% in BLEU and 80.5% in ROUGE compared to the baselines. EMCEE also mitigates the degeneration of data quality caused by training on synthetic data. In human evaluations (N=60), EMCEE outperforms baseline models and the control group in improving users' comprehension, acceptance, trust, and collaboration with static explanations by large margins. Through a fine-grained analysis of model responses, we further demonstrate that training on self-generated synthetic data improves the model's ability to generate more truthful and understandable answers, leading to better user interactions. To the best of our knowledge, this is the first conversational explanation method that can answer free-form user questions following static explanations. 

**Abstract (ZH)**: 可解释的人工智能（XAI）旨在提供对AI模型决策的洞察。目前，大多数XAI方法仅提供一次性、静态的解释，无法满足用户多样化的知识水平和信息需求。会话解释已被提议作为一种有效的方法来定制XAI解释。然而，构建会话解释系统受到训练数据稀缺的阻碍。使用合成数据训练面临两个主要挑战：数据多样性不足和生成数据中的幻觉。为了解决这些问题，我们引入了重复惩罚以促进数据多样性，并利用幻觉检测器过滤掉不真实的合成对话轮。我们对提出的系统fEw-shot Multi-round ConvErsational Explanation (EMCEE)进行了自动和人工评估。自动评估结果显示，EMCEE在BLEU上相对提高了81.6%，在ROUGE上相对提高了80.5%，优于基准模型。EMCEE还缓解了由于使用合成数据训练而导致的数据质量退化。在人工评估（N=60）中，EMCEE在提高用户对静态解释的理解、接受度、信任度和合作度方面显著优于基准模型和对照组。通过对模型响应的精细分析，我们进一步证明，使用自我生成的合成数据训练可以提高模型生成更加真实和易于理解的答案的能力，从而改善用户交互。据我们所知，这是首个能够根据静态解释回答自由形式用户问题的会话解释方法。 

---
# Situational Agency: The Framework for Designing Behavior in Agent-based art 

**Title (ZH)**: 情景代理：基于代理的艺术行为设计框架 

**Authors**: Ary-Yue Huang, Varvara Guljajeva  

**Link**: [PDF](https://arxiv.org/pdf/2503.16442)  

**Abstract**: In the context of artificial life art and agent-based art, this paper draws on Simon Penny's {\itshape Aesthetic of Behavior} theory and Sofian Audry's discussions on behavior computation to examine how artists design agent behaviors and the ensuing aesthetic experiences. We advocate for integrating the environment in which agents operate as the context for behavioral design, positing that the environment emerges through continuous interactions among agents, audiences, and other entities, forming an evolving network of meanings generated by these interactions. Artists create contexts by deploying and guiding these computational systems, audience participation, and agent behaviors through artist strategies. This framework is developed by analysing two categories of agent-based artworks, exploring the intersection of computational systems, audience participation, and artistic strategies in creating aesthetic experiences. This paper seeks to provide a contextual foundation and framework for designing agents' behaviors by conducting a comparative study focused on behavioural design strategies by the artists. 

**Abstract (ZH)**: 人工智能艺术与代理基础艺术的背景下，本文借鉴Simon Penny的《行为美学》理论及Sofian Audry关于行为计算的讨论，探讨艺术家如何设计代理行为及其引发的审美体验。我们主张将代理运行的环境纳入行为设计的背景中，认为环境通过代理、观众及其他实体的持续互动不断涌现，形成由这些互动生成的意义演变网络。艺术家通过部署和引导这些计算系统、观众参与及代理行为，创造出特定的策展背景。本文通过对两类基于代理的艺术作品进行分析，探讨计算系统、观众参与及艺术策略在创造审美体验中的交集，旨在提供一种设计代理行为的背景框架，并通过对比研究艺术家的行为设计策略，为设计代理行为提供理论基础。 

---
# Safe and Efficient Social Navigation through Explainable Safety Regions Based on Topological Features 

**Title (ZH)**: 基于拓扑特征的可解释安全区域的社会导航安全与高效性 

**Authors**: Victor Toscano-Duran, Sara Narteni, Alberto Carlevaro, Rocio Gonzalez-Diaz, Maurizio Mongelli, Jerome Guzzi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16441)  

**Abstract**: The recent adoption of artificial intelligence (AI) in robotics has driven the development of algorithms that enable autonomous systems to adapt to complex social environments. In particular, safe and efficient social navigation is a key challenge, requiring AI not only to avoid collisions and deadlocks but also to interact intuitively and predictably with its surroundings. To date, methods based on probabilistic models and the generation of conformal safety regions have shown promising results in defining safety regions with a controlled margin of error, primarily relying on classification approaches and explicit rules to describe collision-free navigation conditions.
This work explores how topological features contribute to explainable safety regions in social navigation. Instead of using behavioral parameters, we leverage topological data analysis to classify and characterize different simulation behaviors. First, we apply global rule-based classification to distinguish between safe (collision-free) and unsafe scenarios based on topological properties. Then, we define safety regions, $S_\varepsilon$, in the topological feature space, ensuring a maximum classification error of $\varepsilon$. These regions are built with adjustable SVM classifiers and order statistics, providing robust decision boundaries. Local rules extracted from these regions enhance interpretability, keeping the decision-making process transparent.
Our approach initially separates simulations with and without collisions, outperforming methods that not incorporate topological features. It offers a deeper understanding of robot interactions within a navigable space. We further refine safety regions to ensure deadlock-free simulations and integrate both aspects to define a compliant simulation space that guarantees safe and efficient navigation. 

**Abstract (ZH)**: 近期人工智能在 robotics 中的应用推动了能够适应复杂社交环境的自主系统算法的发展。特别是在确保安全和高效的社交导航方面，这是一大关键挑战，不仅要求 AI 避免碰撞和死锁，还需要与周围环境进行直观和可预测的交互。到目前为止，基于概率模型和生成符合安全区域的方法已经在控制误差范围内定义安全区域方面显示出有前景的结果，主要依赖分类方法和明确规则来描述无碰撞导航条件。

本工作探讨拓扑特征如何贡献于可解释的安全区域在社交导航中的作用。我们不使用行为参数，而是利用拓扑数据分析来对不同的仿真行为进行分类和特征化。首先，我们应用全局规则分类，根据拓扑性质区分安全（无碰撞）和不安全场景。然后，在拓扑特征空间中定义安全区域 $S_\varepsilon$，确保分类误差的最大值为 $\varepsilon$。这些区域由可调 SVM 分类器和顺序统计量构建，提供稳健的决策边界。从这些区域中提取的局部规则增强了可解释性，保持决策过程透明。

我们的方法最初将无碰撞和有碰撞的仿真进行了区分，优于未纳入拓扑特征的方法，提供了对机器人在可导航空间中交互的更深层次理解。我们进一步细化安全区域以确保无死锁仿真，并将两者相结合定义一个符合安全和高效导航的仿真空间。 

---
# Cause-effect perception in an object place task 

**Title (ZH)**: 物体摆放任务中的因果感知 

**Authors**: Nikolai Bahr, Christoph Zetzsche, Jaime Maldonado, Kerstin Schill  

**Link**: [PDF](https://arxiv.org/pdf/2503.16440)  

**Abstract**: Algorithmic causal discovery is based on formal reasoning and provably converges toward the optimal solution. However, since some of the underlying assumptions are often not met in practice no applications for autonomous everyday life competence are yet available. Humans on the other hand possess full everyday competence and develop cognitive models in a data efficient manner with the ability to transfer knowledge between and to new situations. Here we investigate the causal discovery capabilities of humans in an object place task in virtual reality (VR) with haptic feedback and compare the results to the state of the art causal discovery algorithms FGES, PC and FCI. In addition we use the algorithms to analyze causal relations between sensory information and the kinematic parameters of human behavior.
Our findings show that the majority of participants were able to determine which variables are causally related. This is in line with causal discovery algorithms like PC, which recover causal dependencies in the first step. However, unlike such algorithms which can identify causes and effects in our test configuration, humans are unsure in determining a causal direction. Regarding the relation between the sensory information provided to the participants and their placing actions (i.e. their kinematic parameters) the data yields a surprising dissociation of the subjects knowledge and the sensorimotor level. Knowledge of the cause-effect pairs, though undirected, should suffice to improve subject's movements. Yet a detailed causal analysis provides little evidence for any such influence. This, together with the reports of the participants, implies that instead of exploiting their consciously perceived information they leave it to the sensorimotor level to control the movement. 

**Abstract (ZH)**: 人类在虚拟现实中的物体放置任务中因果发现能力研究及与先进因果发现算法的比较 

---
# DreamLLM-3D: Affective Dream Reliving using Large Language Model and 3D Generative AI 

**Title (ZH)**: DreamLLM-3D: 基于大型语言模型和3D生成AI的情绪梦境重温 

**Authors**: Pinyao Liu, Keon Ju Lee, Alexander Steinmaurer, Claudia Picard-Deland, Michelle Carr, Alexandra Kitson  

**Link**: [PDF](https://arxiv.org/pdf/2503.16439)  

**Abstract**: We present DreamLLM-3D, a composite multimodal AI system behind an immersive art installation for dream re-experiencing. It enables automated dream content analysis for immersive dream-reliving, by integrating a Large Language Model (LLM) with text-to-3D Generative AI. The LLM processes voiced dream reports to identify key dream entities (characters and objects), social interaction, and dream sentiment. The extracted entities are visualized as dynamic 3D point clouds, with emotional data influencing the color and soundscapes of the virtual dream environment. Additionally, we propose an experiential AI-Dreamworker Hybrid paradigm. Our system and paradigm could potentially facilitate a more emotionally engaging dream-reliving experience, enhancing personal insights and creativity. 

**Abstract (ZH)**: 我们提出DreamLLM-3D，一个结合多模态AI系统的沉浸式艺术装置，用于梦境再体验。该系统通过将大型语言模型（LLM）与文本到3D生成AI集成，实现自动化的梦境内容分析，以支持沉浸式梦境再体验。此外，我们提出了 experiential AI-Dreamworker Hybrid 帕累托改进。我们的系统和帕累托改进有望促进更富有情感共鸣的梦境再体验，增强个人洞察力和创造力。 

---
# Haunted House: A text-based game for comparing the flexibility of mental models in humans and LLMs 

**Title (ZH)**: 鬼屋：基于文本的游戏，用于比较人类和大语言模型的心理模型灵活性 

**Authors**: Brett Puppart, Paul-Henry Paltmann, Jaan Aru  

**Link**: [PDF](https://arxiv.org/pdf/2503.16437)  

**Abstract**: This study introduces "Haunted House" a novel text-based game designed to compare the performance of humans and large language models (LLMs) in model-based reasoning. Players must escape from a house containing nine rooms in a 3x3 grid layout while avoiding the ghost. They are guided by verbal clues that they get each time they move. In Study 1, the results from 98 human participants revealed a success rate of 31.6%, significantly outperforming seven state-of-the-art LLMs tested. Out of 140 attempts across seven LLMs, only one attempt resulted in a pass by Claude 3 Opus. Preliminary results suggested that GPT o3-mini-high performance might be higher, but not at the human level. Further analysis of 29 human participants' moves in Study 2 indicated that LLMs frequently struggled with random and illogical moves, while humans exhibited such errors less frequently. Our findings suggest that current LLMs encounter difficulties in tasks that demand active model-based reasoning, offering inspiration for future benchmarks. 

**Abstract (ZH)**: 这项研究介绍了“鬼屋”这款新型文本游戏，旨在通过模型基于的推理任务比较人类和大型语言模型（LLMs）的表现。玩家必须在一个3x3网格布局的房子里通过九个房间，同时避免遇鬼。他们通过每次移动获得言语线索。研究1的98名人类参与者的结果显示成功率为31.6%，显著优于测试的七种最先进的LLMs。在七种LLM的140次尝试中，仅Claude 3 Opus有一次通过。初步结果显示，GPT o3-mini-high性能可能更高，但尚未达到人类水平。研究2中29名人类参与者的行为分析表明，LLMs在处理随机和不合逻辑的移动时经常遇到困难，而 humans则较少出现此类错误。我们的研究发现表明，当前的LLMs在需要主动模型推理的任务中存在困难，为未来的基准测试提供了启示。 

---
# Enhancing Human-Robot Collaboration through Existing Guidelines: A Case Study Approach 

**Title (ZH)**: 基于现有指南增强人机协作：一个案例研究方法 

**Authors**: Yutaka Matsubara, Akihisa Morikawa, Daichi Mizuguchi, Kiyoshi Fujiwara  

**Link**: [PDF](https://arxiv.org/pdf/2503.16436)  

**Abstract**: As AI systems become more prevalent, concerns about their development, operation, and societal impact intensify. Establishing ethical, social, and safety standards amidst evolving AI capabilities poses significant challenges. Global initiatives are underway to establish guidelines for AI system development and operation. With the increasing use of collaborative human-AI task execution, it's vital to continuously adapt AI systems to meet user and environmental needs. Failure to synchronize AI evolution with changes in users and the environment could result in ethical and safety issues. This paper evaluates the applicability of existing guidelines in human-robot collaborative systems, assesses their effectiveness, and discusses limitations. Through a case study, we examine whether our target system meets requirements outlined in existing guidelines and propose improvements to enhance human-robot interactions. Our contributions provide insights into interpreting and applying guidelines, offer concrete examples of system enhancement, and highlight their applicability and limitations. We believe these contributions will stimulate discussions and influence system assurance and certification in future AI-infused critical systems. 

**Abstract (ZH)**: 随着人工智能系统的普及，对其开发、运营及其社会影响的担忧日益增加。在不断演进的人工智能能力背景下建立伦理、社会和安全标准面临重大挑战。全球范围内正积极推进人工智能系统开发和运营的指导原则。随着人类与人工智能协作任务执行的不断增加，持续适应人工智能系统以满足用户和环境需求变得至关重要。未能同步人工智能进化与用户及环境变化可能导致伦理和安全问题。本文评估了现有指导原则在人机协作系统中的适用性，评估其有效性并讨论其局限性。通过案例研究，我们分析了目标系统是否满足现有指导原则的要求，并提出了改进措施以增强人机交互。我们的贡献提供了对指导原则的理解和应用的见解，提供了系统增强的具体示例，并指出了其适用性和局限性。我们相信这些贡献将激发讨论并影响未来包含人工智能的关键系统的保证和认证。 

---
# Interactive Sketchpad: An Interactive Multimodal System for Collaborative, Visual Problem-Solving 

**Title (ZH)**: 交互式绘图板：一种用于协作性可视化问题解决的多模态交互系统 

**Authors**: Steven-Shine Chen, Jimin Lee, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16434)  

**Abstract**: Humans have long relied on visual aids like sketches and diagrams to support reasoning and problem-solving. Visual tools, like auxiliary lines in geometry or graphs in calculus, are essential for understanding complex ideas. However, many tutoring systems remain text-based, providing feedback only through natural language. Leveraging recent advances in Large Multimodal Models (LMMs), this paper introduces Interactive Sketchpad, a tutoring system that combines language-based explanations with interactive visualizations to enhance learning. Built on a pre-trained LMM, Interactive Sketchpad is fine-tuned to provide step-by-step guidance in both text and visuals, enabling natural multimodal interaction with the student. Accurate and robust diagrams are generated by incorporating code execution into the reasoning process. User studies conducted on math problems such as geometry, calculus, and trigonometry demonstrate that Interactive Sketchpad leads to improved task comprehension, problem-solving accuracy, and engagement levels, highlighting its potential for transforming educational technologies. 

**Abstract (ZH)**: 人类长期以来依靠简图和图表等视觉辅助工具来支持推理和问题解决。视觉工具，如几何学中的辅助线或微积分中的图表，对于理解复杂概念至关重要。然而，许多辅导系统仍然基于文本，仅通过自然语言提供反馈。利用大型多模态模型（LMM）的 recent 进展，本文介绍了互动简图板，这是一种结合基于语言的解释与互动可视化来增强学习的辅导系统。基于预训练的 LMM，互动简图板经过微调，能够同时在文本和可视化方面提供逐步指导，实现与学生的自然多模态交互。通过将代码执行纳入推理过程，生成了准确且稳健的图表。在几何、微积分和三角学等数学问题上的用户研究表明，互动简图板有助于提高任务理解力、问题解决的准确性以及参与度，凸显了其对改造教育技术的潜力。 

---
# Multimodal Transformer Models for Turn-taking Prediction: Effects on Conversational Dynamics of Human-Agent Interaction during Cooperative Gameplay 

**Title (ZH)**: 多模态 Transformer 模型在合作游戏过程中人类-代理交互对话动力学中的轮流预测效果 

**Authors**: Young-Ho Bae, Casey C. Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2503.16432)  

**Abstract**: This study investigates multimodal turn-taking prediction within human-agent interactions (HAI), particularly focusing on cooperative gaming environments. It comprises both model development and subsequent user study, aiming to refine our understanding and improve conversational dynamics in spoken dialogue systems (SDSs). For the modeling phase, we introduce a novel transformer-based deep learning (DL) model that simultaneously integrates multiple modalities - text, vision, audio, and contextual in-game data to predict turn-taking events in real-time. Our model employs a Crossmodal Transformer architecture to effectively fuse information from these diverse modalities, enabling more comprehensive turn-taking predictions. The model demonstrates superior performance compared to baseline models, achieving 87.3% accuracy and 83.0% macro F1 score. A human user study was then conducted to empirically evaluate the turn-taking DL model in an interactive scenario with a virtual avatar while playing the game "Dont Starve Together", comparing a control condition without turn-taking prediction (n=20) to an experimental condition with our model deployed (n=40). Both conditions included a mix of English and Korean speakers, since turn-taking cues are known to vary by culture. We then analyzed the interaction quality, examining aspects such as utterance counts, interruption frequency, and participant perceptions of the avatar. Results from the user study suggest that our multimodal turn-taking model not only enhances the fluidity and naturalness of human-agent conversations, but also maintains a balanced conversational dynamic without significantly altering dialogue frequency. The study provides in-depth insights into the influence of turn-taking abilities on user perceptions and interaction quality, underscoring the potential for more contextually adaptive and responsive conversational agents. 

**Abstract (ZH)**: 本研究探讨了人类-代理交互（HAI）中的多模态轮替预测，重点关注合作游戏环境。该研究涵盖了模型开发和后续用户研究，旨在深化我们对对话式对话系统（SDSs）中对话动力学的理解并予以改进。在建模阶段，我们提出了一种新颖的基于变换器的深度学习（DL）模型，该模型能够同时整合文本、视觉、音频和游戏内上下文等多种模态信息，以实现实时轮替事件预测。该模型采用跨模态变换器架构，有效融合了这些多样模态的信息，从而实现更加全面的轮替预测。与基准模型相比，该模型展现出卓越的性能，准确率为87.3%，宏F1分为83.0%。随后，我们进行了人类使用者研究，在“Dont Starve Together”游戏中与虚拟角色互动，对比了包含轮替预测实验条件（n=40）和无轮替预测控制条件（n=20）的效果。研究对象包括英语和韩语使用者，因为轮替提示会因文化差异而异。我们分析了交互质量，包括话语数量、打断频率以及参与者对虚拟角色的感知等方面。用户研究结果表明，多模态轮替模型不仅提升了人类-代理对话的流畅性和自然性，还维持了平衡的对话动态，未显著改变对话频率。研究为轮替能力对用户体验和交互质量的影响提供了深入洞察，突显了更加上下文相关和响应式对话代理的潜力。 

---
# OpenAI's Approach to External Red Teaming for AI Models and Systems 

**Title (ZH)**: OpenAI针对AI模型和系统的外部红队演练方法 

**Authors**: Lama Ahmad, Sandhini Agarwal, Michael Lampe, Pamela Mishkin  

**Link**: [PDF](https://arxiv.org/pdf/2503.16431)  

**Abstract**: Red teaming has emerged as a critical practice in assessing the possible risks of AI models and systems. It aids in the discovery of novel risks, stress testing possible gaps in existing mitigations, enriching existing quantitative safety metrics, facilitating the creation of new safety measurements, and enhancing public trust and the legitimacy of AI risk assessments. This white paper describes OpenAI's work to date in external red teaming and draws some more general conclusions from this work. We describe the design considerations underpinning external red teaming, which include: selecting composition of red team, deciding on access levels, and providing guidance required to conduct red teaming. Additionally, we show outcomes red teaming can enable such as input into risk assessment and automated evaluations. We also describe the limitations of external red teaming, and how it can fit into a broader range of AI model and system evaluations. Through these contributions, we hope that AI developers and deployers, evaluation creators, and policymakers will be able to better design red teaming campaigns and get a deeper look into how external red teaming can fit into model deployment and evaluation processes. These methods are evolving and the value of different methods continues to shift as the ecosystem around red teaming matures and models themselves improve as tools for red teaming. 

**Abstract (ZH)**: 红队演练已成为评估AI模型和系统潜在风险的关键实践。它有助于发现新型风险，对现有缓解措施可能存在的漏洞进行压力测试，丰富现有的定量安全性指标，促进新安全性度量的创建，以及增强公众对AI风险评估的信任和合法性。本白皮书描述了OpenAI迄今为止在外部红队演练方面的工作，并从这些工作中得出一些更普遍的结论。我们描述了外部红队演练的设计考虑因素，包括：选择红队组成、决定访问级别以及提供进行红队演练所需的指导。此外，我们展示了红队演练所能带来的成果，如风险评估的输入和自动化评估。我们还描述了外部红队演练的局限性，并说明它如何适应更广泛范围的AI模型和系统评估。通过这些贡献，我们希望AI开发者和部署者、评估创建者和政策制定者能够更好地设计红队演练活动，并更深入地了解外部红队演练如何融入模型部署和评估过程。这些方法正在不断发展，不同的方法的价值也会随着红队演练生态圈的成熟和模型自身作为红队工具的改进而变化。 

---
# CLIP-PING: Boosting Lightweight Vision-Language Models with Proximus Intrinsic Neighbors Guidance 

**Title (ZH)**: CLIP-PING：使用Proximus内在邻域指导增强轻量级视觉语言模型 

**Authors**: Chu Myaet Thwal, Ye Lin Tun, Minh N. H. Nguyen, Eui-Nam Huh, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2412.03871)  

**Abstract**: Beyond the success of Contrastive Language-Image Pre-training (CLIP), recent trends mark a shift toward exploring the applicability of lightweight vision-language models for resource-constrained scenarios. These models often deliver suboptimal performance when relying solely on a single image-text contrastive learning objective, spotlighting the need for more effective training mechanisms that guarantee robust cross-modal feature alignment. In this work, we propose CLIP-PING: Contrastive Language-Image Pre-training with Proximus Intrinsic Neighbors Guidance, a novel yet simple and efficient training paradigm designed to boost the performance of lightweight vision-language models with minimal computational overhead and lower data demands. CLIP-PING bootstraps unimodal features extracted from arbitrary pre-trained encoders to obtain intrinsic guidance of proximus neighbor samples, i.e., nearest-neighbor (NN) and cross nearest-neighbor (XNN). We find that extra contrastive supervision from these neighbors substantially boosts cross-modal alignment, enabling lightweight models to learn more generic features with rich semantic diversity. Extensive experiments reveal that CLIP-PING notably surpasses its peers in zero-shot generalization and cross-modal retrieval tasks. Specifically, a 5.5% gain on zero-shot ImageNet1K classification with 10.7% (I2T) and 5.7% (T2I) on Flickr30K retrieval, compared to the original CLIP when using ViT-XS image encoder trained on 3 million (image, text) pairs. Moreover, CLIP-PING showcases a strong transferability under the linear evaluation protocol across several downstream tasks. 

**Abstract (ZH)**: 超越CLIP的成功：面向资源受限场景的轻量级跨模态模型训练新趋势——CLIP-PING：基于近邻样本内在指导的对比语言-图像预训练 

---
# OnDev-LCT: On-Device Lightweight Convolutional Transformers towards federated learning 

**Title (ZH)**: OnDev-LCT: 边缘设备轻量级卷积变换器用于联邦学习 

**Authors**: Chu Myaet Thwal, Minh N.H. Nguyen, Ye Lin Tun, Seong Tae Kim, My T. Thai, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2401.11652)  

**Abstract**: Federated learning (FL) has emerged as a promising approach to collaboratively train machine learning models across multiple edge devices while preserving privacy. The success of FL hinges on the efficiency of participating models and their ability to handle the unique challenges of distributed learning. While several variants of Vision Transformer (ViT) have shown great potential as alternatives to modern convolutional neural networks (CNNs) for centralized training, the unprecedented size and higher computational demands hinder their deployment on resource-constrained edge devices, challenging their widespread application in FL. Since client devices in FL typically have limited computing resources and communication bandwidth, models intended for such devices must strike a balance between model size, computational efficiency, and the ability to adapt to the diverse and non-IID data distributions encountered in FL. To address these challenges, we propose OnDev-LCT: Lightweight Convolutional Transformers for On-Device vision tasks with limited training data and resources. Our models incorporate image-specific inductive biases through the LCT tokenizer by leveraging efficient depthwise separable convolutions in residual linear bottleneck blocks to extract local features, while the multi-head self-attention (MHSA) mechanism in the LCT encoder implicitly facilitates capturing global representations of images. Extensive experiments on benchmark image datasets indicate that our models outperform existing lightweight vision models while having fewer parameters and lower computational demands, making them suitable for FL scenarios with data heterogeneity and communication bottlenecks. 

**Abstract (ZH)**: 联邦学习（FL）作为一种在多个边缘设备上协作训练机器学习模型的同时保护隐私的有潜力的方法已经 Emerged。FL 的成功依赖于参与模型的效率及其处理分布式学习独特挑战的能力。尽管几种变体的 Vision Transformer（ViT）显示出巨大的潜力，作为现代卷积神经网络（CNNs）的替代品用于中心化训练，但它们前所未有的规模和更高的计算需求阻碍了它们在资源受限的边缘设备上的部署，挑战了它们在 FL 中的广泛应用。由于 FL 中的客户端设备通常具有有限的计算资源和通信带宽，针对此类设备的模型必须在模型规模、计算效率以及适应 FL 中遇到的多样化和非同态数据分布之间取得平衡。为了应对这些挑战，我们提出了 OnDev-LCT：适用于有限训练数据和资源的 On-Device 视觉任务的轻量级卷积变压器。我们的模型通过 LCT 分词器结合图像特定的归纳偏差，利用残差线性瓶颈块中的高效深度可分离卷积提取局部特征，而 LCT 编码器中多头自注意力（MHSA）机制隐式地促进了图像全局表示的捕捉。在基准图像数据集上的 extensive 实验表明，我们的模型在参数更少和计算需求更低的情况下优于现有的轻量级视觉模型，使其适合在具有数据异质性和通信瓶颈的 FL 场景中应用。 

---
